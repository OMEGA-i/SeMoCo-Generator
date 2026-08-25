"""Precompute SOMA/TMR GT joints + embeddings into the packed GT cache.

Writes go through :mod:`eval.cache` (``save_tmr_gt_joints`` / ``save_tmr_gt_motion``
/ ``save_tmr_text``) into the packed ``ShardedCacheStore`` under
``<data-root>/eval_cache/v2`` — scopes ``tmr_gt_joints``, ``tmr_gt_motion``, ``tmr_text``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ....local_uri import resolve_local_uri
from ....dataset.umr_parquet import load_joints77_batch
from ... import cache as C
from ...tmr import load_tmr
from .dataset import SomaTMRDataset
from .protocol import TMR_MODEL
from .runner import _tmr_encode_batch  # shared batch TMR encode


def _shard(items: list, shard_index: int, num_shards: int) -> list:
    if num_shards <= 1:
        return items
    return [x for i, x in enumerate(items) if i % num_shards == shard_index]


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute TMR GT into stable per-clip cache")
    p.add_argument("--codes-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--tmr-model", default=TMR_MODEL)
    p.add_argument("--text-encoder", choices=("flan", "siglip", "qwen3"), default="flan")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fk-device", default="cpu")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-tok", type=int, default=125)
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--text-batch", type=int, default=64)
    p.add_argument("--parquet-dir", default=None,
                   help="UMR release dir for GT joints (default: resolve from codes_root meta.json)")
    p.add_argument("--missing-only", action="store_true", default=True)
    p.add_argument("--encode-all", action="store_false", dest="missing_only")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"bad shard: index={args.shard_index} num_shards={args.num_shards}")

    codes_root = resolve_local_uri(args.codes_root)
    ds = SomaTMRDataset(
        codes_root, args.split, text_encoder=args.text_encoder,
        limit=args.limit, seed=0, max_tok=args.max_tok,
    )
    ds_sig = C.dataset_sig(codes_root, args.split)
    emb_sig = C.tmr_gt_sig(args.tmr_model, store=ds_sig)
    print(
        f"[tmr-precompute] clips={len(ds)} shard={args.shard_index}/{args.num_shards} "
        f"emb_sig={emb_sig} cache={C.cache_root()}",
        flush=True,
    )

    need = [
        c for c in ds.clips
        if (not args.missing_only) or (not C.probe_tmr_gt_motion(emb_sig, c.clip_id))
    ]
    need = _shard(need, args.shard_index, args.num_shards)
    need_text = []
    if not args.no_text:
        caps = list(dict.fromkeys(c.caption for c in ds.clips))
        need_text = [
            cp for cp in caps
            if (not args.missing_only) or (not C.probe_tmr_text(args.tmr_model, cp))
        ]
        need_text = _shard(need_text, args.shard_index, args.num_shards)
    print(
        f"[tmr-precompute] shard work motion={len(need)} text={len(need_text)} dry_run={args.dry_run}",
        flush=True,
    )
    if args.dry_run:
        return
    if not need and not need_text:
        print("[tmr-precompute] nothing to do", flush=True)
        return

    tmr = load_tmr(device=args.device, modelname=args.tmr_model, rprecision=not args.no_text)
    gt_src_fps = 50.0  # UMR release source FPS

    t0 = time.time()

    # --- Phase 1: read GT joints directly from original Parquet data ---
    # No tokenizer decode — joints77_pos was precomputed during cursor-realign export.
    joints_by_clip: dict[str, np.ndarray] = {}
    from ...datasets.release_subset import resolve_parquet_dir
    codes_root = resolve_local_uri(args.codes_root)
    parquet_dir = resolve_parquet_dir(codes_root, args.split, cli_value=args.parquet_dir)

    # Separate cached vs pending
    pending_recs: dict[str, object] = {}  # rec_id → clip
    for c in need:
        j = C.load_tmr_gt_joints(c.clip_id, store=ds_sig)
        if j is not None:
            joints_by_clip[c.clip_id] = j
        elif c.rec_id:
            pending_recs[c.rec_id] = c

    if pending_recs:
        joints_batch = load_joints77_batch(parquet_dir, args.split, list(pending_recs.keys()))
        for rec_id, j in joints_batch.items():
            c = pending_recs[rec_id]
            max_frames = args.max_tok * 4  # temporal_stride=4
            if j.shape[0] > max_frames:
                j = j[:max_frames]
            joints_by_clip[c.clip_id] = j
            C.save_tmr_gt_joints(c.clip_id, j, store=ds_sig)
        missed = set(pending_recs) - set(joints_batch)
        for rec_id in missed:
            print(f"[tmr-precompute] skip {rec_id}: not found in parquet", flush=True)

    # --- Phase 2: batch TMR encode ---
    pending_enc = [
        c for c in need
        if (not C.probe_tmr_gt_motion(emb_sig, c.clip_id)) and c.clip_id in joints_by_clip
    ]
    if pending_enc:
        joints = [joints_by_clip[c.clip_id] for c in pending_enc]
        fpses = [gt_src_fps] * len(pending_enc)
        embs = _tmr_encode_batch(tmr, joints, fpses, args.device)
        for c, emb in zip(pending_enc, embs):
            C.save_tmr_gt_motion(emb_sig, c.clip_id, emb)

    n_dec = len(joints_by_clip)
    n_enc = len(pending_enc)
    if n_dec or n_enc:
        elapsed = time.time() - t0
        print(
            f"[tmr-precompute] {n_dec} joints loaded (parquet) + "
            f"{n_enc} TMR-encoded (batched) in {elapsed:.1f}s",
            flush=True,
        )

    if need_text:
        bs = max(1, int(args.text_batch))
        for s in range(0, len(need_text), bs):
            batch = need_text[s : s + bs]
            with torch.inference_mode():
                e = np.asarray(tmr.encode_raw_text(batch, unit_vector=True).float().cpu().numpy())
            for cp, v in zip(batch, e):
                C.save_tmr_text(args.tmr_model, cp, v)
            print(f"[tmr-precompute] text {min(s + bs, len(need_text))}/{len(need_text)}", flush=True)

    print(f"[tmr-precompute] done in {time.time() - t0:.1f}s emb_sig={emb_sig}", flush=True)


if __name__ == "__main__":
    main()
