"""Pre-compute frozen TMR motion encoder latents for all clips.

Loads the motion encoder from ``tmr-soma-rp``, uses ``load_joints77_batch``
for direct parquet access (no full-scan), and saves ``[N, 256]`` float32
latent arrays.  Supports multi-GPU via ``--shard-id`` / ``--num-shards``.

Usage::

    # 6-GPU parallel
    for gpu in 0 1 2 3 4 5; do
        python -m semoco_generator.tools.precompute_tmr_motion_latents \
            --parquet-dir .../derived_umr_<hash> \
            --out-root local://tmr_soma_flan --split train \
            --batch-size 128 --device cuda:$gpu \
            --shard-id $gpu --num-shards 6 &
    done
    wait
"""

from __future__ import annotations

import argparse
import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-compute TMR motion latents")
    p.add_argument("--parquet-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, default=Path("local://tmr_soma_flan"))
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def load_tmr_motion_encoder(device: str):
    """Load TMR-SOMA-RP motion encoder and motion rep (no LLM2Vec)."""
    import warnings
    warnings.filterwarnings("ignore", message="Already found a `peft_config`")

    from kimodo.model.tmr import ACTORStyleEncoder
    from kimodo.model.loading import load_checkpoint_state_dict
    from kimodo.motion_rep import TMRMotionRep
    from kimodo.skeleton import SOMASkeleton30

    from semoco_generator.eval.tmr.kimodo_compat import resolve_tmr_checkpoint

    try:
        ckpt_dir = resolve_tmr_checkpoint("tmr-soma-rp", local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            "TMR-SOMA-RP is not available locally. Run "
            "`python -m semoco_generator.tools.fetch_assets fetch --asset tmr-soma-rp` first."
        ) from exc
    s = ckpt_dir / "stats" / "motion"
    if not all((s / name / "mean.npy").is_file() for name in ("global_root", "local_root", "body")):
        raise FileNotFoundError(f"TMR-SOMA-RP snapshot is incomplete: {ckpt_dir}")

    skeleton = SOMASkeleton30()
    motion_rep = TMRMotionRep(skeleton, fps=30, stats_path=str(ckpt_dir / "stats" / "motion"))

    motion_encoder = ACTORStyleEncoder(
        motion_rep=motion_rep, llm_shape=None, vae=True,
        latent_dim=256, ff_size=1024, num_layers=6, num_heads=4,
        dropout=0.1, activation="gelu",
    )
    motion_encoder.load_state_dict(
        load_checkpoint_state_dict(ckpt_dir / "last_weights" / "motion_encoder.pt"),
    )
    motion_encoder = motion_encoder.to(device).eval()
    for p in motion_encoder.parameters():
        p.requires_grad_(False)

    print(f"[precompute] motion encoder loaded: {ckpt_dir}", flush=True)
    return motion_encoder, motion_rep, skeleton


def encode_batch(motion_encoder, motion_rep, skeleton, joints_list, lengths, device):
    """Encode [B, T, 77, 3] joints → [B, 256] mu latents."""
    B = len(joints_list)
    Tmax = max(lengths)

    padded = np.zeros((B, Tmax, 77, 3), dtype=np.float32)
    for i, (j, l) in enumerate(zip(joints_list, lengths)):
        padded[i, :l] = j

    joints_t = torch.from_numpy(padded).to(device)
    lengths_t = torch.tensor(lengths, device=device)

    # Slice 77 → 30 joints
    from kimodo.skeleton import build_skeleton
    from kimodo.motion_rep.feature_utils import length_to_mask

    skel_slice = motion_rep.skeleton.get_skel_slice(build_skeleton(77))
    joints_t = joints_t[..., skel_slice, :]

    # Cap to 500 frames
    max_frames = 500
    if Tmax > max_frames:
        new_joints = torch.zeros(B, max_frames, joints_t.shape[2], 3,
                                 dtype=joints_t.dtype, device=joints_t.device)
        new_lengths = torch.zeros(B, dtype=torch.long, device=lengths_t.device)
        for i in range(B):
            n = int(lengths_t[i].item())
            n_new = min(n, max_frames)
            idx = torch.linspace(0, n - 1, n_new, dtype=torch.long, device=joints_t.device)
            new_joints[i, :n_new] = joints_t[i, idx]
            new_lengths[i] = n_new
        joints_t = new_joints
        lengths_t = new_lengths
        Tmax = max_frames

    features = motion_rep(
        posed_joints=joints_t, to_canonicalize=True, to_normalize=True, lengths=lengths_t,
    )
    mask = length_to_mask(lengths_t, device=device)

    with torch.no_grad():
        encoded = motion_encoder({"x": features, "mask": mask})
        mu, _ = encoded.unbind(1)

    return mu.cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()

    motion_encoder, motion_rep, skeleton = load_tmr_motion_encoder(args.device)

    # ---- Get rec_ids in order ----
    from semoco_generator.dataset.umr_parquet import _build_rec_id_index, load_joints77_batch

    ridx = _build_rec_id_index(args.parquet_dir, args.split)
    all_rec_ids = sorted(ridx.keys(), key=lambda r: (ridx[r][0], ridx[r][1]))

    # ---- Shard split ----
    if args.num_shards > 1:
        chunk = (len(all_rec_ids) + args.num_shards - 1) // args.num_shards
        start = args.shard_id * chunk
        end = min(start + chunk, len(all_rec_ids))
        all_rec_ids = all_rec_ids[start:end]
        print(f"[precompute] GPU {args.shard_id}/{args.num_shards}: "
              f"rec_ids [{start}:{end}] ({len(all_rec_ids)} clips)", flush=True)
    else:
        print(f"[precompute] {len(all_rec_ids)} clips in {args.split} split", flush=True)

    # ---- Output path ----
    out_path = args.out_root / f"{args.split}.motion_latents.npy"
    if args.num_shards > 1:
        out_path = args.out_root / f"{args.split}.motion_latents_s{args.shard_id:02d}.npy"

    all_latents = np.zeros((len(all_rec_ids), 256), dtype=np.float32)

    # ---- Find which parquet shards overlap with our rec_id range ------------
    # Each rec_id maps to (shard_file, row_index). Collect the unique shard
    # files that contain our rec_ids, preserving the order from _index.json.
    shard_order = []
    seen_shards = set()
    for rid in all_rec_ids:
        shard_file = ridx[rid][0]
        if shard_file not in seen_shards:
            seen_shards.add(shard_file)
            shard_order.append(shard_file)

    print(f"[precompute] scanning {len(shard_order)}/{len(set(s[0] for s in ridx.values()))} "
          f"shards for {len(all_rec_ids)} rec_ids", flush=True)

    # ---- Stream through only relevant shards --------------------------------
    import pyarrow.parquet as pq
    from semoco_generator.dataset.umr_parquet import _list_col_np

    rec_id_to_idx = {rid: i for i, rid in enumerate(all_rec_ids)}
    total_wanted = len(all_rec_ids)
    processed = 0
    shard_dir = args.parquet_dir / args.split

    pbar = tqdm(total=total_wanted, desc=f"[{args.split}]",
                disable=(args.shard_id != 0))

    joints_buf: list[tuple[int, np.ndarray, int]] = []
    bs = args.batch_size

    # Limit PyArrow I/O threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    for shard_file in shard_order:
        shard_path = shard_dir / Path(shard_file).name
        if not shard_path.is_file():
            continue
        pf = pq.ParquetFile(shard_path, memory_map=True)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["rec_id", "num_records", "joints77_pos"])
            rec_ids_col = table.column("rec_id").to_pylist()
            nrecs_col = table.column("num_records").to_numpy()

            # Check if this row group has any wanted rec_ids
            matching = [i for i, rid in enumerate(rec_ids_col) if rid in rec_id_to_idx]
            if not matching:
                continue

            j77_col = table.column("joints77_pos").combine_chunks()
            j77_flat = j77_col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            j77_offsets = j77_col.offsets.to_numpy()

            for i in matching:
                rid = rec_ids_col[i]
                idx = rec_id_to_idx[rid]
                n = int(nrecs_col[i])
                raw = j77_flat[j77_offsets[i]:j77_offsets[i + 1]]
                joints = raw.reshape(n + 1, 77, 3)[1:]
                joints_buf.append((idx, joints, n))
                processed += 1

                if len(joints_buf) >= bs:
                    indices, jlist, lens = zip(*joints_buf)
                    latents = encode_batch(motion_encoder, motion_rep, skeleton,
                                           list(jlist), list(lens), args.device)
                    for ii, mu in zip(indices, latents):
                        all_latents[ii] = mu
                    joints_buf.clear()
                    pbar.update(len(indices))

            if processed >= total_wanted:
                break
        if processed >= total_wanted:
            break

    # Final batch
    if joints_buf:
        indices, jlist, lens = zip(*joints_buf)
        latents = encode_batch(motion_encoder, motion_rep, skeleton,
                               list(jlist), list(lens), args.device)
        for ii, mu in zip(indices, latents):
            all_latents[ii] = mu
        pbar.update(len(indices))

    pbar.close()

    # ---- Save (tmp first, then move) ----

    # ---- Save (tmp first, then move) ----
    tmp_path = Path(tempfile.mktemp(suffix=".npy", dir="/tmp"))
    np.save(tmp_path, all_latents)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_path), str(out_path))
    print(f"[precompute] saved {all_latents.shape} → {out_path}  "
          f"({all_latents.nbytes / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
