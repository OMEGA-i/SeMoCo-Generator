#!/usr/bin/env python3
"""Precompute SOMA-compatible per-clip anchors from HML GT joints via SMPL fitting.

For each HML test clip:
  1. Load first frame of GT joints22 from HumanML3D/new_joints/<rec_id>.npy
  2. Fit SMPL body parameters (global_orient, body_pose, transl) via gradient descent
  3. Convert fitted axis-angle rotations → rot6d (first two columns of rotation matrix)
  4. Map SMPL-24 body joints → SOMA77 joint indices
  5. Save as {clip_id: anchor_dict} pickle

Output: <codes_root>/hml_per_clip_anchors.pkl
  {clip_id: {"init_root_pos": [3], "init_root_rot6d": [6],
             "init_joints76_rot6d": [76,6], "identity_coeffs": [10]}}

GPU-accelerated: processes clips in batches of 512, each with T=1 frame.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semoco_generator.eval.motion_ops.smpl_utils import create_smpl
from semoco_generator.eval.tracks.smpl_hml.dataset import HumanML3DDataset
from semoco_generator.paths import humanml3d_root
from tqdm import tqdm


# SMPL-24 body joints → SOMA77 FK indices
# From conversions.py:_SMPL24_TO_SOMA77_INDEX
_SMPL24_TO_SOMA77: list[int] = [
    0, 67, 72, 1, 68, 73, 2, 69, 74, 3, 70, 75,
    4, 11, 39, 6, 12, 40, 13, 41, 14, 42, 14, 42,
]

# Eye for rot6d identity
_EYE6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def axis_angle_to_rot6d(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle [..., 3] → rot6d [..., 6] (first two cols of rot matrix)."""
    angle = torch.norm(aa, dim=-1, keepdim=True)
    axis = aa / angle.clamp_min(1e-8)
    # Rodrigues: R = I + sin(θ)[a]× + (1-cos(θ))[a]×²
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    # Skew-symmetric cross-product matrix
    a1, a2, a3 = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(a1)
    K = torch.stack([
        torch.stack([zero, -a3, a2], dim=-1),
        torch.stack([a3, zero, -a1], dim=-1),
        torch.stack([-a2, a1, zero], dim=-1),
    ], dim=-2)
    K2 = K @ K
    R = torch.eye(3, device=aa.device, dtype=aa.dtype) + sin_a.unsqueeze(-1) * K + (1 - cos_a).unsqueeze(-1) * K2
    # rot6d = first two columns
    return R[..., :2].reshape(*aa.shape[:-1], 6)


def fit_smpl_first_frame_batch(
    joints_batch: torch.Tensor,  # [B, 1, 22, 3] or [B, 22, 3]
    smpl,
    *,
    fit_steps: int = 100,
    lr: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit SMPL to a batch of single-frame joint positions.

    Returns (global_orient, body_pose, transl) each [B, ...] on the same device.
    """
    if joints_batch.dim() == 4:
        joints_batch = joints_batch[:, 0]  # [B, 22, 3]
    B = joints_batch.shape[0]
    dev = joints_batch.device
    tgt = joints_batch.to(dev, dtype=torch.float32)

    go = torch.zeros(B, 1, 3, device=dev, requires_grad=True)
    bp = torch.zeros(B, 23, 3, device=dev, requires_grad=True)
    tl = tgt[:, 0, :].clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([go, bp, tl], lr=lr)
    for _ in range(fit_steps):
        opt.zero_grad()
        out = smpl(global_orient=go, body_pose=bp, transl=tl)
        torch.nn.functional.mse_loss(out.joints[:, :22, :], tgt).backward()
        opt.step()

    with torch.no_grad():
        return go.detach(), bp.detach(), tl.detach()


def build_anchor_from_smpl_params(
    global_orient: np.ndarray,  # [3] axis-angle
    body_pose: np.ndarray,      # [23, 3] axis-angle
    transl: np.ndarray,         # [3]
) -> dict[str, np.ndarray]:
    """Build a SOMA-compatible anchor dict from fitted SMPL parameters.

    Maps SMPL body joints (23 body + 1 root = 24 total) to SOMA77 (77 joints).
    Non-SMPL SOMA joints (fingers, face, etc.) get identity rotations.
    """
    go_t = torch.from_numpy(global_orient.reshape(1, 3)).float()
    bp_t = torch.from_numpy(body_pose.reshape(23, 3)).float()

    # Root rotation: global_orient → rot6d
    init_root_rot6d = axis_angle_to_rot6d(go_t).numpy().reshape(6).astype(np.float32)

    # Body joint rotations: body_pose → rot6d, mapped to SOMA77 indices
    bp_rot6d = axis_angle_to_rot6d(bp_t).numpy().reshape(23, 6).astype(np.float32)

    # Initialize all 76 SOMA joint rotations to identity
    init_joints76_rot6d = np.tile(_EYE6, (76, 1)).astype(np.float32)

    # Map SMPL body joints (indices 1-23 in SMPL-24) to SOMA77
    # SMPL-24 index → SOMA77 index mapping
    for smpl_idx in range(1, 24):  # SMPL body joints 1-23 (skip root=0)
        soma_idx = _SMPL24_TO_SOMA77[smpl_idx]
        if soma_idx >= 1 and soma_idx < 77:
            # body_pose index is smpl_idx - 1 (body_pose has 23 joints, indices 0-22)
            init_joints76_rot6d[soma_idx - 1] = bp_rot6d[smpl_idx - 1]

    return {
        "init_root_pos": transl.astype(np.float32),
        "init_root_rot6d": init_root_rot6d,
        "init_joints76_rot6d": init_joints76_rot6d,
        "identity_coeffs": np.zeros((10,), dtype=np.float32),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=humanml3d_root())
    p.add_argument("--output", default=None,
                   help="defaults to <data-root>/hml_per_clip_anchors.pkl, "
                        "which is where the HumanML3D track looks for it")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--fit-steps", type=int, default=100)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    device = args.device
    data_root = Path(args.data_root)
    if args.output is None:
        args.output = data_root / "hml_per_clip_anchors.pkl"

    # Load HML test prompts
    print("Loading HML test dataset...")
    ds = HumanML3DDataset(data_root, "test", limit=args.limit, seed=0, protocol="official_hml_eval")
    print(f"  Clips: {len(ds.clips)}")

    # Build clip → rec_id mapping
    clip_rec_ids: list[tuple[str, str]] = []  # (clip_id, rec_id)
    for clip in ds.clips:
        meta = clip.metadata or {}
        rec_id = str(meta.get("rec_id", "")).strip()
        if not rec_id:
            # Fall back: extract rec_id from clip_id (format: hml:test:NNNNNN or hml:test:NNNNNN:segN)
            parts = clip.clip_id.split(":")
            if len(parts) >= 3:
                rec_id = parts[2].split(":")[0]  # "009377" from "009377:seg1"
            else:
                rec_id = clip.clip_id
        clip_rec_ids.append((clip.clip_id, rec_id))

    # SMPL model (reuse across fits with varying batch sizes)
    print(f"Creating SMPL model (batch_size={args.batch_size})...")
    smpl = create_smpl(args.batch_size, device)

    # Process in batches
    anchors: dict[str, dict[str, np.ndarray]] = {}
    new_joints_dir = data_root / "new_joints"
    failed = 0
    not_found = 0

    for start in tqdm(range(0, len(clip_rec_ids), args.batch_size), desc="Fitting anchors"):
        batch_clips = clip_rec_ids[start:start + args.batch_size]
        B = len(batch_clips)

        # Load first frames
        first_frames = []
        valid_indices = []
        for i, (clip_id, rec_id) in enumerate(batch_clips):
            joint_path = new_joints_dir / f"{rec_id}.npy"
            if not joint_path.is_file():
                # Try 6-digit zero-padded format
                joint_path = new_joints_dir / f"{int(rec_id):06d}.npy" if rec_id.isdigit() else joint_path
            if not joint_path.is_file():
                not_found += 1
                continue
            try:
                joints = np.load(joint_path).astype(np.float32)
                if joints.ndim == 3:
                    first_frame = joints[0:1, :22, :]  # [1, 22, 3]
                else:
                    first_frame = joints[:22, :].reshape(1, 22, 3)
                first_frames.append(first_frame)
                valid_indices.append(i)
            except Exception:
                failed += 1
                continue

        if not first_frames:
            continue

        # Stack into batch
        batch_joints = np.concatenate(first_frames, axis=0)  # [B_valid, 22, 3]
        B_valid = batch_joints.shape[0]

        # Resize SMPL batch if needed
        if B_valid != args.batch_size:
            smpl = create_smpl(B_valid, device)

        joints_t = torch.from_numpy(batch_joints).to(device, dtype=torch.float32)

        try:
            go, bp, tl = fit_smpl_first_frame_batch(
                joints_t, smpl, fit_steps=args.fit_steps, lr=0.05,
            )
        except Exception as e:
            print(f"\n  Batch {start}: fit failed: {e}")
            failed += B_valid
            continue

        # Build anchors
        for j, i in enumerate(valid_indices):
            clip_id = batch_clips[i][0]
            anchor = build_anchor_from_smpl_params(
                global_orient=go[j].cpu().numpy(),
                body_pose=bp[j].cpu().numpy(),
                transl=tl[j].cpu().numpy(),
            )
            anchors[clip_id] = anchor

        # Reset SMPL to original batch size for next iteration
        if B_valid != args.batch_size:
            smpl = create_smpl(args.batch_size, device)

    print(f"\nResults:")
    print(f"  Anchors computed: {len(anchors)}/{len(ds.clips)}")
    print(f"  Not found: {not_found}")
    print(f"  Failed fits: {failed}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(anchors, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
