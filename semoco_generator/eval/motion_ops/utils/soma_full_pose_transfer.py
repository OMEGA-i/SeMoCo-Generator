"""Deterministic SOMA full-pose to SMPL-X body-pose transfer.

This module consumes the complete pose carried by a normal converted
``soma77`` clip, rather than a ``joints22`` projection.  Keeping the model's
rotations preserves head and terminal-foot orientation that is mathematically
absent from a position-only body22 IK objective.

The transfer applies SOMA-X's ``relative_to_soma_joint_orient`` convention,
computes semantic SOMA world-rotation deltas, and uses those rotations as an
SO(3) prior while refining the unavoidable SOMA/SMPL-X bind-skeleton position
residual.  It never falls back to a positions-only IK fit or post-fit joint
editing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from ...metrics import resample_fps
from ...soma_pose import (
    KIMODO_POSE_CONVENTION,
    SEMOCO_POSE_CONVENTION,
    SomaPose,
)
from ...tracks.smpl_hml.conversion import soma77_to_joints22
from .smplx_fit import (
    NUM_BETAS,
    NUM_BODY_JOINTS,
    _body22_rest_skeleton,
    _body22_world_rotations,
    _fit_quality_metrics,
    _geodesic_rotation_angles,
    _joint_weights,
    _matrix_to_axis_angle,
    _smplx_body22_fk,
)


# Each output body22 joint maps to the SOMA77 joint with the same anatomical
# role.  This is deliberately a *semantic* map: we transfer world motion
# deltas, never pretend the two rigs have equal bind joints or bone lengths.
SMPLX22_TO_SOMA77: tuple[int, ...] = (
    0,   # pelvis       <- Hips
    67,  # left hip     <- LeftLeg
    72,  # right hip    <- RightLeg
    1,   # spine1       <- Spine1
    68,  # left knee    <- LeftShin
    73,  # right knee   <- RightShin
    2,   # spine2       <- Spine2
    69,  # left ankle   <- LeftFoot
    74,  # right ankle  <- RightFoot
    3,   # spine3       <- Chest
    70,  # left foot    <- LeftToeBase
    75,  # right foot   <- RightToeBase
    4,   # neck         <- Neck1
    11,  # left collar  <- LeftShoulder
    39,  # right collar <- RightShoulder
    6,   # head         <- Head
    12,  # left shoulder<- LeftArm
    40,  # right shoulder <- RightArm
    13,  # left elbow   <- LeftForeArm
    41,  # right elbow  <- RightForeArm
    14,  # left wrist   <- LeftHand
    42,  # right wrist  <- RightHand
)

FORMAT_NAME_SOMA_TRANSFER = "smplx_body22_soma77_transfer_aa"

# The rotation prior is deliberately stronger at the three joints that a
# positions-only body22 fit cannot observe faithfully: head and terminal feet.
# Spine/collar rotations carry different bind semantics between SOMA and
# SMPL-X, so they contribute less to the pose prior.
_POSE_PRIOR_ENDPOINT_JOINTS = (10, 11, 15)
_POSE_PRIOR_SOFT_BIND_JOINTS = (3, 6, 9, 13, 14)


@dataclass(frozen=True)
class SomaWorldPose:
    """Source-neutral SOMA motion ready for semantic SMPL-X transfer."""

    world_motion_deltas77: np.ndarray
    joints77: np.ndarray
    fps: float
    source_pose_convention: str

    def __post_init__(self) -> None:
        world = np.asarray(self.world_motion_deltas77, dtype=np.float32)
        joints = np.asarray(self.joints77, dtype=np.float32)
        if world.ndim != 4 or world.shape[1:] != (77, 3, 3):
            raise ValueError(
                "world_motion_deltas77 must be [T,77,3,3], "
                f"got {world.shape}"
            )
        if joints.shape != (world.shape[0], 77, 3):
            raise ValueError(
                f"joints77 must be [{world.shape[0]},77,3], got {joints.shape}"
            )
        if world.shape[0] < 1:
            raise ValueError("SOMA world pose must contain at least one frame")
        if not np.isfinite(world).all() or not np.isfinite(joints).all():
            raise ValueError("SOMA world pose contains non-finite values")
        if not np.isfinite(float(self.fps)) or float(self.fps) <= 0.0:
            raise ValueError(f"fps must be finite and positive, got {self.fps!r}")
        if not self.source_pose_convention:
            raise ValueError("source_pose_convention must be non-empty")

        gram = np.swapaxes(world, -1, -2) @ world
        eye = np.eye(3, dtype=np.float32)
        orth_error = float(np.max(np.abs(gram - eye)))
        det_error = float(np.max(np.abs(np.linalg.det(world) - 1.0)))
        if orth_error > 3e-3 or det_error > 3e-3:
            raise ValueError(
                "world_motion_deltas77 must contain proper rotation matrices "
                f"(orth_error={orth_error:.3g}, det_error={det_error:.3g})"
            )

        object.__setattr__(self, "world_motion_deltas77", world)
        object.__setattr__(self, "joints77", joints)


def _soma_world_pose_from_semoco(source: SomaPose) -> SomaWorldPose:
    """Apply SOMA-X joint-orient semantics to one Semoco pose."""
    if source.pose_convention != SEMOCO_POSE_CONVENTION:
        raise ValueError(
            f"unsupported Semoco pose convention {source.pose_convention!r}; "
            f"expected {SEMOCO_POSE_CONVENTION!r}"
        )
    return SomaWorldPose(
        world_motion_deltas77=_soma_world_motion_deltas(source.rotmat77),
        joints77=source.joints77,
        fps=float(source.fps),
        source_pose_convention=source.pose_convention,
    )


def _soma_world_pose_from_kimodo(
    source: SomaPose,
    *,
    device: str = "cpu",
    max_fk_error_m: float = 1e-4,
) -> SomaWorldPose:
    """Use KiMoDo's official SOMASkeleton77 FK to obtain world rotations."""
    if source.pose_convention != KIMODO_POSE_CONVENTION:
        raise ValueError(
            f"unsupported KiMoDo pose convention {source.pose_convention!r}; "
            f"expected {KIMODO_POSE_CONVENTION!r}"
        )

    from kimodo.skeleton import SOMASkeleton77

    skeleton = SOMASkeleton77().to(device).eval()
    with torch.inference_mode():
        local = torch.as_tensor(
            np.asarray(source.rotmat77).copy(),
            dtype=torch.float32,
            device=device,
        )
        root = torch.as_tensor(
            np.asarray(source.transl).copy(),
            dtype=torch.float32,
            device=device,
        )
        world, recovered, _without_root = skeleton.fk(local, root)
    world_np = world.cpu().numpy().astype(np.float32, copy=False)
    recovered_np = recovered.cpu().numpy().astype(np.float32, copy=False)
    max_error = float(np.max(np.abs(recovered_np - source.joints77)))
    if not np.isfinite(max_error) or max_error > float(max_fk_error_m):
        raise ValueError(
            "KiMoDo official FK does not reproduce the cached joints77: "
            f"max_abs_error_m={max_error:.8g}"
        )
    return SomaWorldPose(
        world_motion_deltas77=world_np,
        joints77=source.joints77,
        fps=float(source.fps),
        source_pose_convention=source.pose_convention,
    )


def soma_world_pose(
    source: SomaPose,
    *,
    device: str = "cpu",
) -> SomaWorldPose:
    """Resolve a model-specific local pose through its official SOMA rig."""
    if source.pose_convention == SEMOCO_POSE_CONVENTION:
        return _soma_world_pose_from_semoco(source)
    if source.pose_convention == KIMODO_POSE_CONVENTION:
        return _soma_world_pose_from_kimodo(source, device=device)
    raise ValueError(f"unsupported SOMA pose convention {source.pose_convention!r}")


def _soma_assets_root() -> Path:
    from ....paths import soma_tokenizer_root

    return soma_tokenizer_root() / "third_party" / "SOMA-X" / "assets"


@lru_cache(maxsize=1)
def _soma_rig() -> tuple[np.ndarray, np.ndarray]:
    """Load the official SOMA-X rest-orient and parent tables once."""
    asset = _soma_assets_root() / "SOMA_neutral.npz"
    if not asset.is_file():
        raise FileNotFoundError(
            "official SOMA-X rig asset is unavailable: "
            f"expected {asset}"
        )
    with np.load(asset, allow_pickle=False) as data:
        parents = np.asarray(data["joint_parent_ids"], dtype=np.int64)
        orient = np.asarray(data["t_pose_world"], dtype=np.float32)[..., :3, :3]
    if parents.shape != (78,) or orient.shape != (78, 3, 3):
        raise ValueError(
            "unexpected SOMA-X rig shape: "
            f"parents={parents.shape}, t_pose_world={orient.shape}"
        )
    if int(parents[0]) != 0 or int(parents[1]) != 0:
        raise ValueError("unexpected SOMA-X Root/Hips parent convention")
    return parents, orient


def _soma_world_motion_deltas(rotmat77: np.ndarray) -> np.ndarray:
    """Apply official SOMA-X joint-orient semantics and return [T,77] deltas.

    Semoco ``SomaPose.rotmat77`` is local and relative to SOMA's joint-orient
    basis.  SOMA-X applies ``orient[parent].T @ local @ orient[joint]`` before
    forward kinematics.  The returned rotations are world motion deltas from
    the official neutral rest frame, so they are meaningful across rigs.
    """
    matrices = np.asarray(rotmat77, dtype=np.float32)
    if matrices.ndim != 4 or matrices.shape[1:] != (77, 3, 3):
        raise ValueError(f"rotmat77 must be [T,77,3,3], got {matrices.shape}")
    if not np.isfinite(matrices).all():
        raise ValueError("rotmat77 contains non-finite values")

    parents, orient = _soma_rig()
    frames = int(matrices.shape[0])
    identity = np.eye(3, dtype=np.float32)
    relative = np.concatenate(
        (np.broadcast_to(identity, (frames, 1, 3, 3)), matrices), axis=1,
    )
    parent_orient_t = np.swapaxes(orient[parents], -1, -2)
    absolute_local = parent_orient_t[None] @ relative @ orient[None]

    world = np.empty_like(absolute_local)
    world[:, 0] = absolute_local[:, 0]
    for joint in range(1, 78):
        world[:, joint] = world[:, parents[joint]] @ absolute_local[:, joint]

    rest_local = parent_orient_t @ orient
    rest_world = np.empty_like(rest_local)
    rest_world[0] = rest_local[0]
    for joint in range(1, 78):
        rest_world[joint] = rest_world[parents[joint]] @ rest_local[joint]
    # Drop virtual Root so output index 0 remains SOMA77 Hips.
    return world[:, 1:] @ np.swapaxes(rest_world[1:], -1, -2)[None]


def _resample_rotations(
    rotations: np.ndarray,
    *,
    src_fps: float,
    dst_fps: float,
) -> np.ndarray:
    """Geodesically resample ``[T,N,3,3]`` rotations to the package FPS."""
    values = np.asarray(rotations, dtype=np.float32)
    if values.ndim != 4 or values.shape[-2:] != (3, 3):
        raise ValueError(f"rotations must be [T,N,3,3], got {values.shape}")
    frames = int(values.shape[0])
    if frames < 2 or abs(float(src_fps) - float(dst_fps)) < 1e-6:
        return values.copy()
    src_t = np.arange(frames, dtype=np.float64) / float(src_fps)
    out_frames = max(2, int(np.floor((frames - 1) / float(src_fps) * float(dst_fps))) + 1)
    dst_t = np.clip(np.arange(out_frames, dtype=np.float64) / float(dst_fps), 0.0, src_t[-1])
    out = np.empty((out_frames, values.shape[1], 3, 3), dtype=np.float32)
    for joint in range(values.shape[1]):
        out[:, joint] = Slerp(src_t, Rotation.from_matrix(values[:, joint]))(dst_t).as_matrix()
    return out


def _smplx_local_from_world(world: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert target world rotations into SMPL-X root and 21 local rotations."""
    values = torch.as_tensor(world, dtype=torch.float32, device=device)
    if values.ndim != 4 or tuple(values.shape[1:]) != (22, 3, 3):
        raise ValueError(f"world rotations must be [T,22,3,3], got {tuple(values.shape)}")
    _, parents = _body22_rest_skeleton(str(device))
    local = torch.empty_like(values)
    local[:, 0] = values[:, 0]
    for joint in range(1, 22):
        parent = int(parents[joint])
        local[:, joint] = values[:, parent].transpose(-1, -2) @ values[:, joint]
    return _matrix_to_axis_angle(local[:, 0]), _matrix_to_axis_angle(local[:, 1:])


def _semantic_world_rotation_error_deg(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    expected_world: np.ndarray,
) -> dict[str, float]:
    """Report direct semantic head/foot transfer residuals for audit."""
    observed = _body22_world_rotations(global_orient, body_pose)
    target = torch.as_tensor(expected_world, dtype=observed.dtype, device=observed.device)
    relative = target.transpose(-1, -2) @ observed
    cosine = (
        relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ).mul(0.5).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine))
    return {
        "soma_transfer_head_world_error_deg": float(angles[:, 15].max().item()),
        "soma_transfer_left_foot_world_error_deg": float(angles[:, 10].max().item()),
        "soma_transfer_right_foot_world_error_deg": float(angles[:, 11].max().item()),
    }


def _pose_prior_joint_weights(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return anatomical weights for the full-SOMA world-rotation prior."""
    weights = torch.ones(22, device=device, dtype=dtype)
    weights[list(_POSE_PRIOR_ENDPOINT_JOINTS)] = 4.0
    weights[list(_POSE_PRIOR_SOFT_BIND_JOINTS)] = 0.5
    return weights


def _sequence_mean(per_frame: torch.Tensor, boundaries: list[tuple[int, int]]) -> torch.Tensor:
    """Average each clip equally instead of favoring longer source motions."""
    return torch.stack([
        per_frame[start:end].mean() for start, end in boundaries
    ]).mean()


class SomaFullPoseToSMPLXParams:
    """Transfer an official full SOMA pose into SMPL-X body parameters.

    The converted clip contains the model's native rotations. We use them for
    a semantic world-rotation initialization and keep them as an explicit
    geodesic prior while fitting SMPL-X joint positions.
    """

    def __init__(
        self,
        device: str = "cuda",
        *,
        fit_steps: int = 250,
        lr: float = 0.03,
        pose_prior_weight: float = 0.01,
    ) -> None:
        self._device = device
        self._fit_steps = int(fit_steps)
        self._lr = float(lr)
        self._pose_prior_weight = float(pose_prior_weight)

    def transfer(self, source: SomaPose, *, fps: float) -> dict[str, Any]:
        """Return one package-ready, full-pose-prior SMPL-X export at ``fps``."""
        return self.transfer_many([source], fps=fps)[0]

    def transfer_many(
        self,
        sources: list[SomaPose],
        *,
        fps: float,
    ) -> list[dict[str, Any]]:
        """Transfer model poses through their official SOMA semantics."""
        return self.transfer_world_many(
            [soma_world_pose(source, device=self._device) for source in sources],
            fps=fps,
        )

    def transfer_world(
        self,
        source: SomaWorldPose,
        *,
        fps: float,
    ) -> dict[str, Any]:
        """Transfer one validated source-neutral SOMA world pose."""
        return self.transfer_world_many([source], fps=fps)[0]

    def transfer_world_many(
        self,
        sources: list[SomaWorldPose],
        *,
        fps: float,
    ) -> list[dict[str, Any]]:
        """Transfer several validated SOMA world poses in one batched solve."""
        if not sources:
            return []
        if not np.isfinite(float(fps)) or float(fps) <= 0.0:
            raise ValueError(f"fps must be finite and positive, got {fps!r}")
        if self._fit_steps < 0:
            raise ValueError(f"fit_steps must be non-negative, got {self._fit_steps}")
        if self._lr <= 0.0 or not np.isfinite(self._lr):
            raise ValueError(f"lr must be finite and positive, got {self._lr!r}")
        if self._pose_prior_weight < 0.0 or not np.isfinite(self._pose_prior_weight):
            raise ValueError(
                "pose_prior_weight must be finite and non-negative, got "
                f"{self._pose_prior_weight!r}"
            )

        worlds: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for source in sources:
            worlds.append(_resample_rotations(
                source.world_motion_deltas77[
                    :, np.asarray(SMPLX22_TO_SOMA77, dtype=np.int64)
                ],
                src_fps=float(source.fps),
                dst_fps=float(fps),
            ))
            targets.append(resample_fps(
                soma77_to_joints22(source.joints77), float(source.fps), float(fps),
            ))

        device = torch.device(self._device)
        target_world_np = np.concatenate(worlds, axis=0)
        target_joints_np = np.concatenate(targets, axis=0)
        boundaries: list[tuple[int, int]] = []
        start = 0
        for target in targets:
            end = start + int(target.shape[0])
            boundaries.append((start, end))
            start = end

        target_world = torch.as_tensor(target_world_np, dtype=torch.float32, device=device)
        target_joints = torch.as_tensor(target_joints_np, dtype=torch.float32, device=device)
        global_orient, local_body = _smplx_local_from_world(target_world_np, device)
        body_pose = local_body.reshape(-1, NUM_BODY_JOINTS * 3)
        rest, _ = _body22_rest_skeleton(str(device))
        rest = rest.to(dtype=target_joints.dtype, device=device)

        # `source.transl` is SOMA's virtual Root.  `joints22[:, 0]` is the
        # canonical source Hips point, which is the actual SMPL-X pelvis
        # target. Keeping it fixed prevents the spatial solve from hiding a
        # coordinate-convention error in global translation.
        transl = (target_joints[:, 0] - rest[0]).detach()
        global_orient = global_orient.detach().clone().requires_grad_(True)
        body_pose = body_pose.detach().clone().requires_grad_(True)
        position_weights = _joint_weights(device=device, dtype=target_joints.dtype)
        pose_weights = _pose_prior_joint_weights(device=device, dtype=target_joints.dtype)

        optimizer = torch.optim.Adam((global_orient, body_pose), lr=self._lr)
        for _ in range(self._fit_steps):
            optimizer.zero_grad(set_to_none=True)
            predicted = _smplx_body22_fk(global_orient, body_pose, transl)
            spatial_per_frame = (
                (predicted - target_joints).square() * position_weights[None, :, None]
            ).sum(dim=(1, 2)) / (3.0 * position_weights.sum())
            observed_world = _body22_world_rotations(global_orient, body_pose)
            rotation_angles = _geodesic_rotation_angles(
                observed_world, target_world, differentiable=True,
            )
            prior_per_frame = (
                rotation_angles.square() * pose_weights[None]
            ).sum(dim=1) / pose_weights.sum()
            loss = _sequence_mean(spatial_per_frame, boundaries)
            loss = loss + self._pose_prior_weight * _sequence_mean(prior_per_frame, boundaries)
            loss.backward()
            optimizer.step()

        results: list[dict[str, Any]] = []
        with torch.no_grad():
            for source, (clip_start, clip_end), clip_target_np, clip_world_np in zip(
                sources, boundaries, targets, worlds, strict=True,
            ):
                clip_go = global_orient[clip_start:clip_end]
                clip_bp = body_pose[clip_start:clip_end]
                clip_tl = transl[clip_start:clip_end]
                clip_target = target_joints[clip_start:clip_end]
                clip_predicted = _smplx_body22_fk(clip_go, clip_bp, clip_tl)
                quality = _fit_quality_metrics(
                    clip_predicted,
                    clip_target,
                    clip_go,
                    body_pose=clip_bp,
                )
                semantic = _semantic_world_rotation_error_deg(
                    clip_go, clip_bp, clip_world_np,
                )
                root_residual = torch.linalg.vector_norm(
                    clip_predicted[:, 0] - clip_target[:, 0], dim=-1,
                ) * 1000.0
                results.append({
                    "joints22": clip_target_np.astype(np.float32, copy=False),
                    "smplx_fk_joints22": clip_predicted.cpu().numpy().astype(np.float32),
                    "transl": clip_tl.cpu().numpy().astype(np.float32),
                    "global_orient": clip_go.cpu().numpy().astype(np.float32),
                    "body_pose": clip_bp.cpu().numpy().astype(np.float32),
                    "betas": np.zeros(NUM_BETAS, dtype=np.float32),
                    "fit_mse": float(torch.nn.functional.mse_loss(
                        clip_predicted, clip_target,
                    ).item()),
                    "format": FORMAT_NAME_SOMA_TRANSFER,
                    "retarget_method": "soma_full_pose_prior_smplx_refinement",
                    "source_pose_convention": source.source_pose_convention,
                    "soma_transfer_root_position_error_mm": float(root_residual.max().item()),
                    "soma_transfer_pose_prior_weight": float(self._pose_prior_weight),
                    **quality,
                    **semantic,
                })
        return results


__all__ = [
    "FORMAT_NAME_SOMA_TRANSFER",
    "SMPLX22_TO_SOMA77",
    "SomaWorldPose",
    "SomaFullPoseToSMPLXParams",
    "soma_world_pose",
]
