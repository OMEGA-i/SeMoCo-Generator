"""SMPL mesh vertices → SOMA77 joints via SOMA-X PoseInversion.

Uses low-LOD SOMA (4505 verts) for memory efficiency (~0.4GB).
PI processes all frames in one batch; FK runs per-frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...paths import ensure_tokenizer_on_path


def _soma_x_assets() -> Path:
    ensure_tokenizer_on_path()
    from data.soma77_fk import default_soma_data_root  # noqa: WPS433
    return default_soma_data_root()


def _build_soma(device):
    ensure_tokenizer_on_path()
    from data.soma77_fk import _ensure_soma_on_path, _soma_init_context  # noqa: WPS433
    _ensure_soma_on_path()
    from soma.soma import SOMALayer  # noqa: WPS433
    with _soma_init_context():
        return SOMALayer(_soma_x_assets(), low_lod=True,
                         identity_model_type="smpl",
                         device=torch.device(device), mode="warp")


class SOMAConverter:
    """SMPL mesh vertices → SOMA77 joints. Build once, convert many."""

    def __init__(self, device="cuda", *, fit_iters: int = 50, fit_lr: float = 5e-3):
        self.device = device
        # Autograd-FK fit iterations. The analytical warp kabsch solver
        # (body_iters/full_iters) is non-deterministic run-to-run (warp atomics),
        # which injects ~0.1-0.8 of joint noise into SOMA77 and destabilizes TMR
        # scores. Autograd-only fitting (no analytical solve) is deterministic
        # (rot diff ~1e-4) and accurate (~3mm vertex error).
        self.fit_iters = int(fit_iters)
        self.fit_lr = float(fit_lr)
        self._soma = _build_soma(device)

    @torch.no_grad()
    def vertices_to_soma77(self, vertices: torch.Tensor) -> np.ndarray:
        if vertices.dim() != 3:
            raise ValueError(f"Expected [T, V, 3], got {tuple(vertices.shape)}")

        verts = vertices.to(torch.device(self.device), dtype=torch.float32)
        T = verts.shape[0]

        from soma.pose_inversion import PoseInversion  # noqa: WPS433
        from soma.geometry.rig_utils import remove_joint_orient_local  # noqa: WPS433
        from soma.geometry.transforms import matrix_to_rotvec  # noqa: WPS433

        num_b = int(self._soma.identity_model.num_identity_coeffs)
        betas = torch.zeros((1, num_b), device=self.device, dtype=torch.float32)
        inv = PoseInversion(self._soma, low_lod=True)
        inv.prepare_identity(betas)

        # PoseInversion.fit expects (B, V, 3) with frames as batch — NOT (1, T, V, 3).
        # Use deterministic autograd-only fitting (skip the non-deterministic
        # analytical warp kabsch solve). Requires grad, so override the outer
        # no_grad for this call.
        with torch.enable_grad():
            r = inv.fit(
                verts,
                body_iters=0,
                finger_iters=0,
                full_iters=0,
                autograd_iters=self.fit_iters,
                autograd_lr=self.fit_lr,
            )
        rot = r["rotations"]              # [T, 78, 3, 3]
        transl = r["root_translation"]    # [T, 3]

        # FK over all T frames at once (frames as batch). Pass relative rotation
        # matrices with pose2rot=False (the axis-angle pose2rot=True path has a
        # batched-view bug in soma.pose; matrices are batch-safe and match the
        # per-frame result to ~1e-6). Falls back to per-frame on any shape issue.
        try:
            rel = remove_joint_orient_local(
                rot, self._soma._t_pose_orient, self._soma._t_pose_orient_parent_T)  # [T,78,3,3]
            p77_mat = rel[:, 1:, :, :].contiguous()  # strip virtual Root -> [T,77,3,3]
            self._soma.prepare_identity(betas.expand(T, -1).contiguous(), global_scale=1.0)
            out = self._soma.pose(p77_mat, transl=transl, pose2rot=False)
            joints = out["joints"].detach()
        except (RuntimeError, ValueError, IndexError):
            all_joints = []
            for i in range(T):
                rel = remove_joint_orient_local(
                    rot[i:i+1], self._soma._t_pose_orient,
                    self._soma._t_pose_orient_parent_T)
                pv = matrix_to_rotvec(rel.reshape(-1, 3, 3)).reshape(1, -1, 3)
                p77 = pv[:, 1:, :]
                self._soma.prepare_identity(betas, global_scale=1.0)
                out = self._soma.pose(p77, transl=transl[i:i+1], pose2rot=True)
                all_joints.append(out["joints"].detach())
            joints = torch.cat(all_joints, dim=0)

        joints = joints.cpu().numpy().astype(np.float32)
        assert joints.shape[1] == 77
        return joints
