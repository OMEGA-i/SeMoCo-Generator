"""Official HumanML3D ``process_file`` feature extraction (EricGuo5513).

``new_joints`` on disk are ``recover_from_ric(new_joint_vecs)``, already in the
canonical floor / face-Z+ frame. Regenerating features from ``new_joints`` must
skip the uniform-skeleton / put-on-floor / face-Z+ stages (``already_aligned``).
"""

from __future__ import annotations

import numpy as np
import torch

from .paramUtil import t2m_kinematic_chain, t2m_raw_offsets
from .quaternion import (
    qbetween_np,
    qfix,
    qinv_np,
    qmul_np,
    qrot_np,
    quaternion_to_cont6d_np,
)
from .skeleton import Skeleton

# Lower legs / feet / face (official HumanML3D constants).
L_IDX1, L_IDX2 = 5, 8
FID_R, FID_L = [8, 11], [7, 10]
FACE_JOINT_INDX = [2, 1, 17, 16]
JOINTS_NUM = 22
N_RAW_OFFSETS = torch.from_numpy(t2m_raw_offsets).float()
KINEMATIC_CHAIN = t2m_kinematic_chain


def uniform_skeleton(positions: np.ndarray, target_offset: torch.Tensor) -> np.ndarray:
    src_skel = Skeleton(N_RAW_OFFSETS, KINEMATIC_CHAIN, "cpu")
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0])).numpy()
    tgt_offset = target_offset.numpy()
    src_leg_len = np.abs(src_offset[L_IDX1]).max() + np.abs(src_offset[L_IDX2]).max()
    tgt_leg_len = np.abs(tgt_offset[L_IDX1]).max() + np.abs(tgt_offset[L_IDX2]).max()
    scale_rt = tgt_leg_len / src_leg_len
    # Do NOT scale the root trajectory by scale_rt.
    #
    # The original HumanML3D code multiplies positions[:, 0] by scale_rt under
    # the assumption that the source and target skeletons are scaled versions of
    # the same character — leg length maps 1:1 to stride length, so trajectory
    # scaling keeps the "number of steps" semantics consistent.
    #
    # That assumption breaks when the source skeleton is SMPL (FK output from
    # HyMotion rot6d) and the target skeleton is the HumanML3D canonical
    # example (000021.npy). SMPL and HML have genuinely different bone
    # *proportions* (up to 27 %), not just a uniform scale difference, so
    # scale_rt is a measurement artefact, not a meaningful height ratio.
    # Multiplying the root trajectory by scale_rt scales the global velocity by
    # 10-30 %, which directly poisons the first 4 (root-velocity) dimensions of
    # the 263-D feature, and through normalisation contaminates the entire
    # evaluator embedding → catastrophic FID / R-precision / Diversity.
    #
    # GT features use already_aligned=True (no uniform_skeleton at all), and
    # Semoco / Kimodo produce joints22 from soma77 (already close to the HML
    # canonical skeleton, so scale_rt ≈ 1.0), so this change is a no-op for
    # every model except HyMotion — exactly the one it fixes.
    tgt_root_pos = positions[:, 0]  # was: positions[:, 0] * scale_rt
    quat_params = src_skel.inverse_kinematics_np(positions, FACE_JOINT_INDX)
    src_skel.set_offset(target_offset)
    return src_skel.forward_kinematics_np(quat_params, tgt_root_pos)


def _align_positions(positions: np.ndarray, tgt_offsets: torch.Tensor | None) -> np.ndarray:
    if tgt_offsets is not None:
        positions = uniform_skeleton(positions, tgt_offsets)
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions = positions.copy()
    positions[:, :, 1] -= floor_height
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    return qrot_np(root_quat_init, positions)


def _foot_detect(positions: np.ndarray, thres: float):
    velfactor = np.array([thres, thres])
    feet_l_x = (positions[1:, FID_L, 0] - positions[:-1, FID_L, 0]) ** 2
    feet_l_y = (positions[1:, FID_L, 1] - positions[:-1, FID_L, 1]) ** 2
    feet_l_z = (positions[1:, FID_L, 2] - positions[:-1, FID_L, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)
    feet_r_x = (positions[1:, FID_R, 0] - positions[:-1, FID_R, 0]) ** 2
    feet_r_y = (positions[1:, FID_R, 1] - positions[:-1, FID_R, 1]) ** 2
    feet_r_z = (positions[1:, FID_R, 2] - positions[:-1, FID_R, 2]) ** 2
    feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)
    return feet_l, feet_r


def _features_from_global(positions: np.ndarray, feet_thre: float) -> np.ndarray:
    """Build 263-D features from already-aligned global joints ``[T,22,3]``."""
    global_positions = positions.copy()
    feet_l, feet_r = _foot_detect(positions, feet_thre)

    skel = Skeleton(N_RAW_OFFSETS, KINEMATIC_CHAIN, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, FACE_JOINT_INDX, smooth_forward=True)
    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    r_rot = quat_params[:, 0].copy()
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = qrot_np(r_rot[1:], velocity)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))

    # RIC (root-relative, facing Z+).
    ric = positions.copy()
    ric[..., 0] -= ric[:, 0:1, 0]
    ric[..., 2] -= ric[:, 0:1, 2]
    ric = qrot_np(np.repeat(r_rot[:, None], ric.shape[1], axis=1), ric)

    root_y = ric[:, 0, 1:2]
    r_velocity = np.arcsin(np.clip(r_velocity[:, 2:3], -1.0, 1.0))
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
    ric_data = ric[:, 1:].reshape(len(ric), -1)
    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    ).reshape(len(r_rot) - 1, -1)

    data = np.concatenate(
        [root_data, ric_data[:-1], rot_data[:-1], local_vel, feet_l, feet_r],
        axis=-1,
    ).astype(np.float32)
    if data.shape[-1] != 263:
        raise ValueError(f"expected 263-D HumanML features, got {data.shape}")
    return data


def process_file(
    positions: np.ndarray,
    feet_thre: float = 0.002,
    *,
    tgt_offsets: torch.Tensor | None = None,
    already_aligned: bool = False,
) -> np.ndarray:
    """Official HumanML3D feature extraction.

    Parameters
    ----------
    positions
        ``[T, J, 3]`` joint positions (use first 22 joints).
    already_aligned
        If True, skip uniform-skeleton / floor / face-Z+ (for ``new_joints``).
    tgt_offsets
        Required when ``already_aligned=False`` (from example skeleton 000021).
    """
    positions = np.asarray(positions, dtype=np.float32)[:, :JOINTS_NUM]
    if positions.ndim != 3 or positions.shape[1] != JOINTS_NUM:
        raise ValueError(f"expected [T,{JOINTS_NUM},3], got {positions.shape}")
    if positions.shape[0] < 2:
        raise ValueError(f"need at least 2 frames, got {positions.shape[0]}")
    if already_aligned:
        aligned = positions.copy()
    else:
        if tgt_offsets is None:
            raise ValueError("tgt_offsets required when already_aligned=False")
        aligned = _align_positions(positions, tgt_offsets)
    return _features_from_global(aligned, feet_thre)


def offsets_from_example(example_joints: np.ndarray) -> torch.Tensor:
    """Compute target skeleton offsets from one example pose ``[T|1, J, 3]``."""
    j = np.asarray(example_joints, dtype=np.float32)
    if j.ndim == 3:
        j = j[0]
    j = j[:JOINTS_NUM]
    skel = Skeleton(N_RAW_OFFSETS, KINEMATIC_CHAIN, "cpu")
    return skel.get_offsets_joints(torch.from_numpy(j))


__all__ = [
    "JOINTS_NUM",
    "offsets_from_example",
    "process_file",
    "uniform_skeleton",
]
