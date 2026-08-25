#!/usr/bin/env python3
"""Verify precomputed TMR motion latents and code-store alignment."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from semoco_generator.local_uri import resolve_local_uri


def _sample_indices(size: int) -> list[int]:
    if size < 1:
        return []
    return sorted({0, size // 2, size - 1})


def verify_alignment(
    codes_root: Path,
    motion_latents_root: Path,
    parquet_dir: Path,
    split: str,
) -> None:
    """Spot-check record ordering and text-embedding slices."""
    motion = np.load(motion_latents_root / f"{split}.motion_latents.npy")
    shared = json.loads((codes_root / f"{split}.index.json").read_text())
    text_index = json.loads(
        (codes_root / f"{split}.text_index.flan.json").read_text()
    )
    aligned_size = min(len(motion), len(shared), len(text_index))
    print(
        f"[verify] aligned records: motion={len(motion)} shared={len(shared)} "
        f"text_index={len(text_index)}"
    )
    if aligned_size < 1:
        raise ValueError(f"{split} has no aligned records")

    from semoco_generator.dataset.umr_parquet import _build_rec_id_index

    parquet_index = _build_rec_id_index(parquet_dir, split)
    parquet_rec_ids = sorted(
        parquet_index,
        key=lambda rec_id: (parquet_index[rec_id][0], parquet_index[rec_id][1]),
    )
    if len(parquet_rec_ids) < aligned_size:
        raise ValueError(
            f"{split} parquet has {len(parquet_rec_ids)} records, "
            f"but aligned stores have {aligned_size}"
        )

    mismatches = 0
    for index in _sample_indices(aligned_size):
        expected = parquet_rec_ids[index]
        actual = shared[index]["rec_id"]
        status = "OK" if expected == actual else "MISMATCH"
        mismatches += int(expected != actual)
        caption = str(shared[index].get("caption", ""))[:60]
        print(f"  [{index}] {status} rec_id={actual} caption={caption}...")
    if mismatches:
        raise ValueError(f"{split} has {mismatches} record-order mismatches")

    text_embeddings = np.load(
        codes_root / f"{split}.text_emb.flan.npy", mmap_mode="r"
    )
    for index in _sample_indices(aligned_size):
        entry = text_index[index]
        start = int(entry["text_start"])
        length = int(entry["text_len"])
        embedding = np.asarray(
            text_embeddings[start : start + length], dtype=np.float32
        )
        print(
            f"  [{index}] text_emb={embedding.shape} "
            f"mean={embedding.mean():.4f} std={embedding.std():.4f}"
        )


def verify_latent_quality(motion_latents_root: Path, split: str) -> None:
    """Validate shape, finiteness, and aggregate latent statistics."""
    motion = np.load(motion_latents_root / f"{split}.motion_latents.npy")
    if motion.ndim != 2 or motion.shape[0] < 1:
        raise ValueError(f"{split} motion latents must be non-empty [N,D], got {motion.shape}")
    if not np.isfinite(motion).all():
        raise ValueError(f"{split} motion latents contain NaN or Inf")

    mean = motion.mean(axis=0)
    std = motion.std(axis=0)
    norms = np.linalg.norm(motion, axis=1)
    print(f"[verify] latent quality ({split}, {len(motion)} clips)")
    print(f"  shape: {motion.shape}")
    print(f"  mean per dim: {mean.mean():.6f} +/- {mean.std():.4f}")
    print(f"  std per dim:  {std.mean():.4f} +/- {std.std():.4f}")
    print(
        f"  L2 norms: mean={norms.mean():.4f} std={norms.std():.4f} "
        f"min={norms.min():.4f} max={norms.max():.4f}"
    )
    if abs(float(mean.mean())) > 0.1:
        raise ValueError(f"{split} latent mean is too far from zero")


def verify_live_encoding(
    motion_latents_root: Path,
    parquet_dir: Path,
    *,
    split: str,
    device: str,
    tmr_model: str,
) -> None:
    """Compare selected latents with on-the-fly TMR motion encoding."""
    warnings.filterwarnings("ignore", message="Already found a `peft_config`")

    from semoco_generator.dataset.umr_parquet import (
        _build_rec_id_index,
        load_joints77_batch,
    )
    from semoco_generator.eval.tmr import load_tmr

    tmr = load_tmr(
        modelname=tmr_model,
        device=device,
        rprecision=False,
        defer_text_encoder=True,
    )
    parquet_index = _build_rec_id_index(parquet_dir, split)
    rec_ids = sorted(
        parquet_index,
        key=lambda rec_id: (parquet_index[rec_id][0], parquet_index[rec_id][1]),
    )
    precomputed = np.load(
        motion_latents_root / f"{split}.motion_latents.npy", mmap_mode="r"
    )

    for index in (0, 100, 1000, 10000):
        if index >= min(len(rec_ids), len(precomputed)):
            continue
        rec_id = rec_ids[index]
        joints = load_joints77_batch(parquet_dir, split, [rec_id]).get(rec_id)
        if joints is None:
            raise ValueError(f"{split}/{rec_id} is missing from parquet")
        joints_tensor = torch.from_numpy(joints).to(device).unsqueeze(0)
        lengths = torch.tensor([len(joints)], device=device)
        with torch.inference_mode():
            live = tmr.encode_motion(
                posed_joints=joints_tensor,
                lengths=lengths,
                unit_vector=False,
            )[0]
        live_array = live.float().cpu().numpy()
        expected = np.asarray(precomputed[index], dtype=np.float32)
        cosine = float(
            np.dot(live_array, expected)
            / (np.linalg.norm(live_array) * np.linalg.norm(expected))
        )
        print(f"  [{index}] rec_id={rec_id} cosine={cosine:.6f}")
        if cosine <= 0.999:
            raise ValueError(
                f"live TMR mismatch for {split}/{rec_id}: cosine={cosine:.6f}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes-root", required=True)
    parser.add_argument("--motion-latents-root", type=Path, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tmr-model", default="tmr-soma-rp")
    parser.add_argument("--skip-live-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    codes_root = resolve_local_uri(args.codes_root)
    motion_root = args.motion_latents_root.expanduser().resolve()
    parquet_dir = args.parquet_dir.expanduser().resolve()

    available_splits: list[str] = []
    for split in args.splits:
        path = motion_root / f"{split}.motion_latents.npy"
        if not path.is_file():
            print(f"[verify] {path} is not ready; skipping")
            continue
        available_splits.append(split)
        verify_alignment(codes_root, motion_root, parquet_dir, split)
        verify_latent_quality(motion_root, split)

    if not args.skip_live_check and available_splits:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print("[verify] CUDA is unavailable; skipping live encoding check")
        else:
            verify_live_encoding(
                motion_root,
                parquet_dir,
                split=available_splits[0],
                device=args.device,
                tmr_model=args.tmr_model,
            )
    print("[verify] verification complete")


if __name__ == "__main__":
    main()
