"""Pre-compute TMR motion FEATURES (not latents) for joint training.

Like ``precompute_tmr_motion_latents.py`` but stops at ``TMRMotionRep`` output
(186-dim per frame) instead of running through the motion encoder.  The encoder
is trained jointly in the unfrozen variant so we need raw features as input.

Output per split:
  ``{split}.motion_features.npy``  [sum_T, 186] float16  (mmap-friendly)
  ``{split}.motion_index.json``    [{feat_start, feat_len}] per clip
"""

from __future__ import annotations

import argparse, json, os, tempfile, shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_MAX_FRAMES = 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, default=Path("local://tmr_soma_flan"))
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def load_motion_rep():
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
    print(f"[features] motion_rep loaded from {ckpt_dir}", flush=True)
    return motion_rep, skeleton


def extract_features_batch(motion_rep, skeleton, joints_list, lengths, device):
    """[B, T, 77, 3] joints → list of [T_i, 186] features."""
    from kimodo.skeleton import build_skeleton

    B = len(joints_list)
    Tmax = min(max(lengths), _MAX_FRAMES)

    padded = np.zeros((B, Tmax, 77, 3), dtype=np.float32)
    for i, (j, l) in enumerate(zip(joints_list, lengths)):
        l = min(l, Tmax)
        if l < len(j):
            idx = np.linspace(0, len(j) - 1, l, dtype=int)
            padded[i, :l] = j[idx]
        else:
            padded[i, :l] = j[:l]

    joints_t = torch.from_numpy(padded).to(device)
    lengths_t = torch.tensor([min(l, Tmax) for l in lengths], device=device)

    skel_slice = motion_rep.skeleton.get_skel_slice(build_skeleton(77))
    joints_t = joints_t[..., skel_slice, :]

    features = motion_rep(
        posed_joints=joints_t, to_canonicalize=True, to_normalize=True, lengths=lengths_t,
    )  # [B, Tmax, 186]

    feats = []
    for i in range(B):
        L = int(lengths_t[i].item())
        feats.append(features[i, :L].cpu().numpy().astype(np.float16))
    return feats


def main() -> None:
    args = parse_args()

    motion_rep, skeleton = load_motion_rep()

    from semoco_generator.dataset.umr_parquet import _build_rec_id_index

    ridx = _build_rec_id_index(args.parquet_dir, args.split)
    all_rec_ids = sorted(ridx.keys(), key=lambda r: (ridx[r][0], ridx[r][1]))

    if args.num_shards > 1:
        chunk = (len(all_rec_ids) + args.num_shards - 1) // args.num_shards
        start = args.shard_id * chunk
        end = min(start + chunk, len(all_rec_ids))
        all_rec_ids = all_rec_ids[start:end]
        print(f"[features] GPU {args.shard_id}/{args.num_shards}: "
              f"[{start}:{end}] ({len(all_rec_ids)} clips)", flush=True)

    # Find relevant shards
    shard_order = []
    seen = set()
    for rid in all_rec_ids:
        sf = ridx[rid][0]
        if sf not in seen:
            seen.add(sf)
            shard_order.append(sf)

    import pyarrow.parquet as pq
    from semoco_generator.dataset.umr_parquet import _list_col_np

    rec_to_idx = {rid: i for i, rid in enumerate(all_rec_ids)}
    total = len(all_rec_ids)
    processed = 0
    shard_dir = args.parquet_dir / args.split
    bs = args.batch_size

    all_features = []        # flat list of [T_i, 186] float16 arrays
    feat_index = []           # [{feat_start, feat_len}]
    feat_offset = 0

    pbar = tqdm(total=total, desc=f"[{args.split}]", disable=(args.shard_id != 0))

    jbuf, ibuf = [], []
    for shard_file in shard_order:
        shard_path = shard_dir / Path(shard_file).name
        if not shard_path.is_file():
            continue
        pf = pq.ParquetFile(shard_path, memory_map=True)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["rec_id", "num_records", "joints77_pos"])
            recs = table.column("rec_id").to_pylist()
            nrecs = table.column("num_records").to_numpy()
            matching = [i for i, rid in enumerate(recs) if rid in rec_to_idx]
            if not matching:
                continue
            j77 = table.column("joints77_pos").combine_chunks()
            flat = j77.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            offs = j77.offsets.to_numpy()
            for i in matching:
                idx = rec_to_idx[recs[i]]
                n = int(nrecs[i])
                raw = flat[offs[i]:offs[i + 1]]
                joints = raw.reshape(n + 1, 77, 3)[1:]
                jbuf.append((idx, joints, n))
                processed += 1
                if len(jbuf) >= bs:
                    indices, jlist, lens = zip(*jbuf)
                    feats = extract_features_batch(motion_rep, skeleton, list(jlist), list(lens), args.device)
                    for ii, f in zip(indices, feats):
                        ibuf.append((ii, f))
                    jbuf.clear()
                    pbar.update(len(indices))
            if processed >= total:
                break
        if processed >= total:
            break

    if jbuf:
        indices, jlist, lens = zip(*jbuf)
        feats = extract_features_batch(motion_rep, skeleton, list(jlist), list(lens), args.device)
        for ii, f in zip(indices, feats):
            ibuf.append((ii, f))
        pbar.update(len(indices))

    pbar.close()

    # Sort by original index and flatten
    ibuf.sort(key=lambda x: x[0])
    feat_index = [None] * len(ibuf)
    for idx, feats in ibuf:
        feat_index[idx] = {"feat_start": feat_offset, "feat_len": len(feats)}
        feat_offset += len(feats)
        all_features.append(feats)

    full = np.concatenate(all_features, axis=0).astype(np.float16)

    # Output paths
    feat_path = args.out_root / f"{args.split}.motion_features.npy"
    idx_path = args.out_root / f"{args.split}.motion_index.json"
    if args.num_shards > 1:
        feat_path = args.out_root / f"{args.split}.motion_features_s{args.shard_id:02d}.npy"
        idx_path = args.out_root / f"{args.split}.motion_index_s{args.shard_id:02d}.json"

    tmp = Path(tempfile.mktemp(suffix=".npy", dir="/tmp"))
    np.save(tmp, full)
    args.out_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(feat_path))
    idx_path.write_text(json.dumps(feat_index))
    print(f"[features] saved {full.shape} → {feat_path}  "
          f"({full.nbytes / 1e6:.1f} MB) + {len(feat_index)} index entries", flush=True)


if __name__ == "__main__":
    main()
