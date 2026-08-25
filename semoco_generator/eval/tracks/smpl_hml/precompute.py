"""Precompute HumanML3D ``text_mot_match`` GT embeddings into the packed GT cache.

Writes go through :mod:`eval.cache` (``save_hml_gt_motion`` / ``save_hml_gt_text``)
into the packed ``ShardedCacheStore`` under ``<data-root>/eval_cache/v2`` — scopes
``hml_gt_motion`` and ``hml_gt_text``, keyed by ``gt_sig``/``clip_id``/``text_key``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ... import cache as C
from .dataset import HumanML3DDataset
from .hml_evaluator import TextMotMatchEvaluator
from .protocol import DEFAULT_CHECKPOINT, DEFAULT_MEAN_STD, HML_SUBSET_PROTOCOL
from .word_vectorizer import load_word_vectorizer, resolve_glove_root


def _load_mean_std(meta_dir: Path):
    mean_p, std_p = meta_dir / "mean.npy", meta_dir / "std.npy"
    if mean_p.is_file() and std_p.is_file():
        return np.load(mean_p).astype(np.float32), np.load(std_p).astype(np.float32)
    return None, None


def _m_length(clip) -> int | None:
    meta = clip.metadata or {}
    return int(meta.get("m_length") or meta.get("n_frames") or 0) or None


def _gt_feature(ds: HumanML3DDataset, index: int) -> np.ndarray:
    return ds.motion(index)


def _shard(items: list, shard_index: int, num_shards: int) -> list:
    if num_shards <= 1:
        return items
    return [x for i, x in enumerate(items) if i % num_shards == shard_index]


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute HumanML GT embeddings (stable per-clip cache)")
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--evaluator-checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--mean-std-dir", default=str(DEFAULT_MEAN_STD))
    p.add_argument("--glove-root", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--hml-protocol", choices=("official_hml_eval", "legacy_full_test"),
                   default=HML_SUBSET_PROTOCOL)
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--no-official-encode", action="store_true")
    p.add_argument("--missing-only", action="store_true", default=True)
    p.add_argument("--encode-all", action="store_false", dest="missing_only")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--encode-batch", type=int, default=32)
    args = p.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"bad shard: index={args.shard_index} num_shards={args.num_shards}")

    data_root = Path(args.data_root)
    ds = HumanML3DDataset(
        data_root, args.split, limit=args.limit, seed=0, protocol=args.hml_protocol,
    )
    gt_sig = C.hml_gt_sig(
        args.evaluator_checkpoint,
        official_encode=not args.no_official_encode,
        hml_protocol=args.hml_protocol,
        data=data_root.name,
        mean_std=args.mean_std_dir,
        glove=args.glove_root,
    )
    text_sig = C.hml_gt_text_sig(
        args.evaluator_checkpoint,
        official_encode=not args.no_official_encode,
        hml_protocol=args.hml_protocol,
        data=data_root.name,
        glove=args.glove_root,
    )
    print(
        f"[hml-precompute] clips={len(ds)} shard={args.shard_index}/{args.num_shards} "
        f"gt_sig={gt_sig} text_sig={text_sig} cache={C.cache_root()}",
        flush=True,
    )

    need_motion = []
    need_text = []  # (clip, tokens, text_key)
    for i, clip in enumerate(ds.clips):
        if (not args.missing_only) or (not C.probe_hml_gt_motion(gt_sig, clip.clip_id)):
            need_motion.append((i, clip))
        if not args.no_text:
            toks = list((clip.metadata or {}).get("tokens") or [])
            tk = C.text_key(clip.caption, toks)
            if (not args.missing_only) or (not C.probe_hml_gt_text(text_sig, tk)):
                need_text.append((clip, toks, tk))

    need_motion = _shard(need_motion, args.shard_index, args.num_shards)
    # Dedup text keys globally, then shard unique keys so each caption is encoded once.
    if need_text:
        seen: dict[str, tuple] = {}
        for clip, toks, tk in need_text:
            seen.setdefault(tk, (clip, toks, tk))
        need_text = _shard(list(seen.values()), args.shard_index, args.num_shards)

    print(
        f"[hml-precompute] shard work motion={len(need_motion)} text={len(need_text)} "
        f"dry_run={args.dry_run}",
        flush=True,
    )
    if args.dry_run:
        return
    if not need_motion and not need_text:
        print("[hml-precompute] nothing to do", flush=True)
        return

    mean, std = _load_mean_std(Path(args.mean_std_dir))
    wv = None
    if need_text:
        glove = resolve_glove_root([args.glove_root] if args.glove_root else None)
        if glove is None:
            raise SystemExit("GloVe missing; pass --glove-root or use --no-text")
        wv = load_word_vectorizer(glove)
    evaluator = TextMotMatchEvaluator(
        args.evaluator_checkpoint,
        device=args.device,
        word_vectorizer=wv,
        mean=mean,
        std=std,
        official_protocol=not args.no_official_encode,
    )

    t0 = time.time()
    batch = max(1, int(args.encode_batch))
    for s in range(0, len(need_motion), batch):
        chunk = need_motion[s : s + batch]
        feats = [_gt_feature(ds, i) for i, _ in chunk]
        lens = [_m_length(c) for _, c in chunk]
        embs = evaluator.encode_motion(feats, lengths=lens)
        for (_, c), e in zip(chunk, embs):
            C.save_hml_gt_motion(gt_sig, c.clip_id, e)
        print(
            f"[hml-precompute] motion {min(s + batch, len(need_motion))}/{len(need_motion)}",
            flush=True,
        )

    if need_text:
        for s in range(0, len(need_text), batch):
            chunk = need_text[s : s + batch]
            embs = evaluator.encode_text(
                [clip.caption for clip, _, _ in chunk],
                tokens=[toks for _, toks, _ in chunk],
            )
            for (_, _, tk), e in zip(chunk, embs):
                C.save_hml_gt_text(text_sig, tk, e)
            print(
                f"[hml-precompute] text {min(s + batch, len(need_text))}/{len(need_text)}",
                flush=True,
            )

    print(f"[hml-precompute] done in {time.time() - t0:.1f}s gt_sig={gt_sig}", flush=True)


if __name__ == "__main__":
    main()
