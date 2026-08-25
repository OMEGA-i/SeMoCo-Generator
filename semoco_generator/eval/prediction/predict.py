"""Unconditional motion prediction: roll out a prefix and write SMPL-22 predictions.

Protocol: no text conditioning, observe the first ``--predict-ratio`` of a clip's
tokens, autoregressively generate the rest, decode through the frozen tokenizer
and SOMA-X forward kinematics, then keep the SMPL-22 joint subset. Predictions
land as ``{cid}_pred.npy`` ``[T, 22, 3]``, which
:mod:`semoco_generator.eval.prediction.score` turns into ADE/FDE.

With ``--num-samples K > 1`` this draws K independent rollouts per clip and keeps
the min-ADE and min-FDE picks (``{cid}_pred.npy`` and ``{cid}_predfde.npy``), the
standard stochastic human-motion-prediction protocol. Selection needs a ground
truth, so ``--gt-dir`` (from ``build_gt``) is required in that case.

Clips are bucketed by exact token length, so every clip in a rollout batch shares
one ``(prefix_len, num_steps)``. There is no cross-clip padding: the result is
bit-identical to a per-clip rollout while amortizing the GPU launch.

Example::

    python -m semoco_generator.eval.prediction.predict \\
        --checkpoint runs/motion_gpt_150m/model/best.pt \\
        --tokenizer-checkpoint checkpoints/tokenizer/split_branch_sem.pt \\
        --codes-root local://t2m_codes --parquet-dir <release>/derived_umr_<hash> \\
        --split test --out-dir local://pred_out/test --max-tokens 31
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ...dataset.umr_parquet import load_rows_by_rec_id
from ...local_uri import resolve_local_uri
from ...paths import ensure_tokenizer_on_path
from ...tokenizer_bridge import FrozenMotionTokenizer
from ..decode_rollout import codes_to_joints
from ..rollout import SamplingConfig, default_motion_sampling, load_model, rollout
from .build_gt import safe_cid
from .metrics import DEFAULT_PREDICT_RATIO, ade, fde

# Anchor columns only: decoding needs the frame-0 seed, not the [T, 499] features.
_ANCHOR_COLS = ["init_root_pos", "init_root_rot6d", "init_joints76_rot6d", "identity_coeffs"]


def _load_smpl22_fn():
    ensure_tokenizer_on_path()
    from eval.recon_common import soma77_to_smpl22  # type: ignore

    return soma77_to_smpl22


def _load_codes_store(codes_root: Path, split: str):
    codes = np.load(codes_root / f"{split}.codes.npy", mmap_mode="r")
    index = json.loads((codes_root / f"{split}.index.json").read_text())
    first = index[0]
    ks = "code_start" if "code_start" in first else "start"
    kl = "code_len" if "code_len" in first else "length"
    return codes, [(str(e["rec_id"]), int(e[ks]), int(e[kl])) for e in index]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="trained motion prior (a motion_gpt_*.yaml run, use_text=false)")
    ap.add_argument("--tokenizer-checkpoint", default=None)
    ap.add_argument("--codes-root", required=True, help="dir with <split>.codes.npy/.index.json")
    ap.add_argument("--parquet-dir", required=True,
                    help="derived UMR parquet, read for the frame-0 anchors")
    ap.add_argument("--gt-dir", default=None,
                    help="{cid}_gt.npy from build_gt; required for best-of-K selection")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--predict-ratio", type=float, default=DEFAULT_PREDICT_RATIO,
                    help="fraction of tokens observed before generation starts")
    ap.add_argument("--min-tokens", type=int, default=5,
                    help="skip clips shorter than this, which have no usable future segment")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="cap clip length to this many tokens (0=off); 31 tokens is the 2s horizon")
    ap.add_argument("--rec-id-list", default=None,
                    help="optional file, one rec_id per line, to restrict the clip set")
    ap.add_argument("--limit", type=int, default=0, help="cap #clips (0 = all)")
    ap.add_argument("--num-shards", type=int, default=1, help="split clips across N processes")
    ap.add_argument("--shard-idx", type=int, default=0, help="this process's shard [0..N-1]")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=1,
                    help="best-of-K: draw K stochastic rollouts per clip and keep the best")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None,
                    help="scalar temperature override (default: per-codebook schedule)")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log-every", type=int, default=500)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    k = max(1, args.num_samples)
    if k > 1 and not args.gt_dir:
        raise SystemExit("--num-samples > 1 selects against the ground truth: pass --gt-dir")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    out_dir = resolve_local_uri(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codes_root = resolve_local_uri(args.codes_root)
    gt_dir = resolve_local_uri(args.gt_dir) if args.gt_dir else None

    model, _ = load_model(args.checkpoint, device=device)
    tok = FrozenMotionTokenizer.load(args.tokenizer_checkpoint, device=device)
    soma77_to_smpl22 = _load_smpl22_fn()
    q = model.cfg.num_codebooks
    if args.temperature is not None:
        sampling = SamplingConfig(temperature=args.temperature, top_p=args.top_p, top_k=0)
    else:
        sampling = default_motion_sampling(q)

    codes, entries = _load_codes_store(codes_root, args.split)
    if args.rec_id_list:
        want = {ln.strip() for ln in Path(args.rec_id_list).read_text().splitlines() if ln.strip()}
        entries = [e for e in entries if e[0] in want]

    ratio = args.predict_ratio
    cap = args.max_tokens
    usable = []
    for rid, start, length in entries:
        if length < args.min_tokens:
            continue
        eff = min(length, cap) if cap > 0 else length
        plen = max(1, int(eff * ratio))
        if eff - plen < 1:  # need at least one generated token
            continue
        usable.append((rid, start, eff, plen))
    if args.limit:
        usable = usable[: args.limit]
    if args.num_shards > 1:
        usable = usable[args.shard_idx :: args.num_shards]
    n_before = len(usable)

    listing = [f.name for f in out_dir.iterdir()] if out_dir.is_dir() else []
    have_pred = {f[: -len("_pred.npy")] for f in listing if f.endswith("_pred.npy")}
    have_fde = {f[: -len("_predfde.npy")] for f in listing if f.endswith("_predfde.npy")}
    # A best-of-K clip only counts as done once both picks are on disk.
    done = (have_pred & have_fde) if k > 1 else have_pred
    usable = [u for u in usable if safe_cid(u[0]) not in done]
    print(f"[pred] split={args.split} clips_total={len(entries)} usable={n_before} "
          f"todo={len(usable)} shard={args.shard_idx}/{args.num_shards} ratio={ratio} "
          f"min_tokens={args.min_tokens} max_tokens={cap} K={k} Q={q}", flush=True)

    rec_ids = [u[0] for u in usable]
    print(f"[pred] scanning parquet for {len(rec_ids)} anchors ...", flush=True)
    anchors = load_rows_by_rec_id(args.parquet_dir, args.split, rec_ids, cols=_ANCHOR_COLS)
    missing = [r for r in rec_ids if r not in anchors]
    if missing:
        print(f"[pred] WARN {len(missing)} rec_ids have no parquet anchor (skipped), "
              f"e.g. {missing[:3]}", flush=True)
        usable = [u for u in usable if u[0] in anchors]

    buckets: dict[int, list[tuple[str, int, int, int]]] = defaultdict(list)
    for u in usable:
        buckets[u[2]].append(u)

    written = 0
    t0 = time.time()
    bs = max(1, args.batch_size)
    for length in sorted(buckets):
        group = buckets[length]
        for i in range(0, len(group), bs):
            chunk = group[i : i + bs]
            plen = chunk[0][3]
            num_steps = length - plen
            prefix = np.stack([np.asarray(codes[s : s + plen], dtype=np.int64)
                               for _, s, _, _ in chunk])
            prefix_t = torch.from_numpy(prefix).to(device)  # [B, plen, Q]

            gt22 = [None] * len(chunk)
            if gt_dir is not None:
                for b, (rid, _, _, _) in enumerate(chunk):
                    gp = gt_dir / f"{safe_cid(rid)}_gt.npy"
                    if gp.exists():
                        gt22[b] = np.load(gp).astype(np.float32)

            best_ade = [float("inf")] * len(chunk)
            best_fde = [float("inf")] * len(chunk)
            best_pred = [None] * len(chunk)
            best_pred_fde = [None] * len(chunk)
            for _ in range(k):
                gen = rollout(model, prefix_t, num_steps, sampling=sampling, device=device)
                gen = gen.cpu().numpy().astype(np.int64)  # [B, L, Q]
                for b, (rid, _, _, _) in enumerate(chunk):
                    pred_j = codes_to_joints(tok, gen[b], anchors[rid], device=device)["joints77"]
                    pred22 = soma77_to_smpl22(pred_j).astype(np.float32)
                    if k == 1 or gt22[b] is None:
                        if best_pred[b] is None:
                            best_pred[b] = pred22
                            best_pred_fde[b] = pred22
                        continue
                    a = ade(pred22, gt22[b], ratio)
                    if a < best_ade[b]:
                        best_ade[b] = a
                        best_pred[b] = pred22
                    d = fde(pred22, gt22[b])
                    if d < best_fde[b]:
                        best_fde[b] = d
                        best_pred_fde[b] = pred22

            for b, (rid, _, _, _) in enumerate(chunk):
                cid = safe_cid(rid)
                np.save(out_dir / f"{cid}_pred.npy", best_pred[b])
                if k > 1:  # the min-FDE pick only differs from min-ADE for best-of-K
                    np.save(out_dir / f"{cid}_predfde.npy", best_pred_fde[b])
                written += 1
            if written and (written % args.log_every < bs):
                rate = written / max(1e-9, time.time() - t0)
                print(f"[pred] {written}/{len(usable)} clips | {rate:.1f} clips/s "
                      f"| L~{length} pfx={plen} steps={num_steps} K={k}", flush=True)

    dt = time.time() - t0
    print(f"[pred] done: wrote {written} clips to {out_dir} in {dt / 60:.1f} min "
          f"({written / max(1e-9, dt):.1f} clips/s)", flush=True)


if __name__ == "__main__":
    main()
