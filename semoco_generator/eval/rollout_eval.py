"""Batch rollout + SOMA77 visualization over a test-split sample.

Deterministically selects ``--num-samples`` clips from the derived parquet split
(fixed ``--seed``), runs 2s-prefix -> 8s rollout for each, decodes to SOMA77 FK
joints, and writes per-clip viewer ``.npz`` (pred + gt) plus an ``index.json``
describing the selection. Selection is reproducible: same seed + parquet ->
same recordings.

Example::

    python -m semoco_generator.eval.rollout_eval \
        --checkpoint runs/mgpt_codear_150m/model/best.pt \
        --parquet-dir <derived_umr_dir> \
        --split test --num-samples 100 --seed 0 \
        --prefix-tokens 25 --rollout-tokens 100 \
        --out-dir runs/mgpt_codear_150m/eval_test --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..dataset.umr_parquet import iter_parquet_rows, load_rows_by_rec_id
from ..local_uri import resolve_local_uri
from ..paths import default_checkpoint
from ..tokenizer_bridge import FrozenMotionTokenizer, soma_skeleton_edges
from . import metrics as M
from .rollout import load_model
from .visualize_soma77 import build_sampling, encode_clip, generate_and_decode, save_demo


@torch.no_grad()
def _clip_token_logits(model, codes: np.ndarray, device, amp_dtype):
    """Teacher-forced logits/targets/mask for one clip (contiguous causal)."""
    inp = torch.from_numpy(codes[:-1].astype(np.int64)).unsqueeze(0).to(device)
    tgt = torch.from_numpy(codes[1:].astype(np.int64)).unsqueeze(0).to(device)
    mask = torch.ones(1, inp.shape[1], dtype=torch.bool, device=device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
        logits = model.forward_packed(
            inp, tgt,
            segment_ids=torch.zeros(1, inp.shape[1], dtype=torch.long, device=device),
            positions=torch.arange(inp.shape[1], device=device).unsqueeze(0),
        )
    return logits, tgt, mask


def _foot_indices():
    from ..paths import ensure_tokenizer_on_path  # noqa: WPS433
    ensure_tokenizer_on_path()
    from data.umr_schema import FOOT_CONTACT_SOMA77_INDICES, SLICE_FOOT_CONTACT  # noqa: WPS433
    return list(FOOT_CONTACT_SOMA77_INDICES), SLICE_FOOT_CONTACT


def _select_rec_ids(rec_ids: list[str], num: int, seed: int) -> list[str]:
    """Deterministic random selection (fixed seed -> fixed order)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rec_ids))
    return [rec_ids[i] for i in order]  # caller takes valid clips until ``num``


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch rollout + SOMA77 viz over test split.")
    parser.add_argument("--checkpoint", type=str, required=True, help="trained MotionGPT checkpoint")
    parser.add_argument("--tokenizer-checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--parquet-dir", type=str, required=True,
                        help="derived_umr_* dir (features + anchor source); local:// or path")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="deterministic selection seed")
    parser.add_argument("--prefix-tokens", type=int, default=25)
    parser.add_argument("--rollout-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--out-dir", type=str, required=True, help="local:// URI or path")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    parquet_dir = resolve_local_uri(args.parquet_dir)
    out_dir = resolve_local_uri(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = FrozenMotionTokenizer.load(args.tokenizer_checkpoint or default_checkpoint(), device=args.device)
    model, _ = load_model(args.checkpoint, device=args.device)
    sampling = build_sampling(model.cfg.num_codebooks, args.temperature, args.top_p, args.top_k)
    edges = soma_skeleton_edges()

    # Pass 1 (cheap: rec_id only) -> deterministic order; then load features +
    # anchor for a capped candidate set (headroom for too-short skips).
    rec_ids = [str(r["rec_id"]) for r in iter_parquet_rows(parquet_dir, args.split, cols=[])]
    ordered = _select_rec_ids(rec_ids, args.num_samples, args.seed)
    cap = min(len(ordered), max(args.num_samples * 4, args.num_samples + 64))
    candidates = ordered[:cap]
    clips = load_rows_by_rec_id(parquet_dir, args.split, candidates)
    print(f"[eval] split={args.split} pool={len(rec_ids)} target={args.num_samples} "
          f"candidates={len(candidates)} seed={args.seed}")

    q = model.cfg.num_codebooks
    amp = torch.bfloat16 if "cuda" in args.device else torch.float32
    foot_idx, foot_slice = _foot_indices()
    tok_agg = M.new_accumulator(q)                 # teacher-forced token metrics
    gen_stats = {"repeat_rate": [], "foot_skate": [], "jerk": []}  # rollout diagnostics
    prefix_frames = args.prefix_tokens * tok.stride

    selected: list[dict] = []
    skipped = 0
    t0 = time.time()
    for rec_id in candidates:
        if len(selected) >= args.num_samples:
            break
        clip = clips.get(rec_id)
        if clip is None:
            skipped += 1
            continue
        try:
            codes = encode_clip(tok, clip)                                 # [T, Q]
            if codes.shape[0] < args.prefix_tokens + 2:
                raise ValueError("too short")
            # Teacher-forced token metrics on the full clip.
            logits, tgt, mask = _clip_token_logits(model, codes, torch.device(args.device), amp)
            M.accumulate(tok_agg, logits, tgt, mask)
            # Rollout + decode (reuses the codes we just encoded).
            result = generate_and_decode(
                model, tok, clip,
                prefix_tokens=args.prefix_tokens, rollout_tokens=args.rollout_tokens,
                sampling=sampling, device=args.device, all_codes=codes,
            )
        except ValueError:
            skipped += 1
            continue

        # Generation diagnostics on the rolled-out region.
        pred = result["pred"]
        foot_contact = np.clip(pred["features"][:, foot_slice], 0.0, 1.0)
        gen_stats["repeat_rate"].append(M.repeat_rate(result["gen_codes"], start=args.prefix_tokens))
        gen_stats["jerk"].append(M.joint_jerk(pred["joints77"], result["fps"], start=prefix_frames))
        gen_stats["foot_skate"].append(
            M.foot_skate(pred["joints77"], foot_contact, foot_idx, result["fps"], start=prefix_frames))

        meta = save_demo(out_dir, result, edges, meta_extra={
            "split": args.split, "checkpoint": args.checkpoint,
            "tokenizer_checkpoint": tok.spec.checkpoint, "token_rate": tok.spec.token_rate,
        })
        selected.append(meta)
        if len(selected) % 10 == 0:
            rate = len(selected) / max(1e-9, time.time() - t0)
            print(f"[eval] {len(selected)}/{args.num_samples} clips ({rate:.2f}/s, skipped {skipped})")

    token_metrics = M.finalize(tok_agg, q, prefix="test/")
    gen_metrics = {f"gen/{k}": (float(np.mean(v)) if v else 0.0) for k, v in gen_stats.items()}
    index = {
        "split": args.split,
        "parquet_dir": str(parquet_dir),
        "seed": args.seed,
        "num_requested": args.num_samples,
        "num_selected": len(selected),
        "skipped_too_short": skipped,
        "prefix_tokens": args.prefix_tokens,
        "rollout_tokens": args.rollout_tokens,
        "checkpoint": args.checkpoint,
        "tokenizer_checkpoint": tok.spec.checkpoint,
        "token_metrics": token_metrics,
        "generation_metrics": gen_metrics,
        "rec_ids": [m["rec_id"] for m in selected],
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(
        {"token_metrics": token_metrics, "generation_metrics": gen_metrics,
         "num_clips": len(selected), "checkpoint": args.checkpoint}, indent=2))
    print(
        f"[eval] done: {len(selected)} clips -> {out_dir} (skipped {skipped})\n"
        f"[eval] test ppl_mean={token_metrics['test/ppl_mean']:.2f} "
        f"ce_mean={token_metrics['test/ce_mean']:.4f} top1_q0={token_metrics['test/top1_q0']:.3f} | "
        f"repeat={gen_metrics['gen/repeat_rate']:.3f} foot_skate={gen_metrics['gen/foot_skate']:.3f} "
        f"jerk={gen_metrics['gen/jerk']:.1f}")


if __name__ == "__main__":
    main()
