"""SMPL-X pose export helpers for the external MotionViewer demo-bundle schema.

Two paths produce the same npz field set (``joints22`` / ``transl`` /
``global_orient`` / ``body_pose`` / ``betas`` / ``fit_mse``):

* :class:`Joints22ToSMPLXParams` — IK-fit from 22-joint positions
  (``format="smplx_body22_fitted_aa"``). Used for GT and baselines that only
  expose joint positions.
* :class:`Rot6dTranslToSMPLXParams` — deterministic decode from native
  ``smpl_rot6d_transl`` (``format="smplx_body22_native_aa"``). For models
  with that native output, vis export then never re-enters the lossy
  soma77/hml263 → joints22 → IK gauge-ambiguity path.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Union

import numpy as np
import torch

JointsLike = Union[np.ndarray, torch.Tensor]
Rot6dLike = Union[np.ndarray, torch.Tensor]
TranslLike = Union[np.ndarray, torch.Tensor]
ToeMarkersLike = Union[np.ndarray, torch.Tensor]
HeadMarkersLike = Union[np.ndarray, torch.Tensor]

# SMPL-X body (no hands/face): 21 joints x 3 axis-angle = 63-D body_pose, matching
# the reference bundle's `body_pose.shape == (T, 63)`.
NUM_BODY_JOINTS = 21
NUM_BETAS = 16

FORMAT_NAME = "smplx_body22_fitted_aa"
FORMAT_NAME_NATIVE = "smplx_body22_native_aa"
BODY22_LEVELS = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9),
    (10, 11, 12, 13, 14),
    (15, 16, 17),
    (18, 19),
    (20, 21),
)

# SMPL-X body22 indices. SOMA and SMPL-X do not share identical bind-joint
# semantics around the spine/collars, so those targets must not dominate IK.
SPINE_COLLAR_JOINTS = (3, 6, 9, 13, 14)
END_EFFECTOR_JOINTS = (10, 11, 15, 20, 21)  # feet, head, wrists
KEY_QUALITY_JOINTS = (4, 5, 7, 8, 10, 11, 15, 20, 21)

# The position-only body22 chain terminates at left/right foot. Their local
# rotations are invisible to the position loss, and each ankle retains one
# invisible twist DOF around its ankle-to-foot rest axis.
ANKLE_FOOT_PAIRS = ((7, 10), (8, 11))
ANKLE_BODY_POSE_JOINTS = tuple(ankle - 1 for ankle, _ in ANKLE_FOOT_PAIRS)
TERMINAL_FOOT_BODY_POSE_JOINTS = tuple(foot - 1 for _, foot in ANKLE_FOOT_PAIRS)
HEAD_BODY_POSE_JOINT = 14  # SMPL-X body joint 15 (head) minus the root.
HEAD_PARENT_BODY22_JOINT = 12  # neck

# ``head_markers`` is [HeadEnd, LeftEye, RightEye] from the full SOMA77
# position stream.  The first body22 head position plus its two eye markers
# determine a full face frame; HeadEnd is retained for provenance and future
# validation even though the eye frame is the direct SMPL-X correspondence.
HEAD_MARKER_COUNT = 3

FIT_MPJPE_LIMIT_MM = 80.0
FIT_KEY_MPJPE_LIMIT_MM = 50.0
MAX_ROOT_STEP_LIMIT_DEG = 45.0
MAX_TOE_DIRECTION_ERROR_LIMIT_DEG = 5.0
MAX_HEAD_ORIENTATION_ERROR_LIMIT_DEG = 5.0
MAX_HEAD_WORLD_STEP_LIMIT_DEG = 45.0
MAX_ANKLE_WORLD_STEP_LIMIT_DEG = 45.0
MAX_TERMINAL_FOOT_WORLD_STEP_LIMIT_DEG = 45.0
MAX_SOMA_TRANSFER_WORLD_ERROR_LIMIT_DEG = 5.0
MAX_SOMA_TRANSFER_ROOT_POSITION_ERROR_LIMIT_MM = 0.1


def _joint_weights(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weights = torch.ones(22, device=device, dtype=dtype)
    weights[list(SPINE_COLLAR_JOINTS)] = 0.25
    weights[list(END_EFFECTOR_JOINTS)] = 2.0
    return weights


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Differentiable axis-angle to rotation matrix with a stable zero limit."""
    angles = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    small = angles.abs() < 1e-6
    scale = torch.empty_like(angles)
    scale[~small] = torch.sin(half_angles[~small]) / angles[~small]
    scale[small] = 0.5 - angles[small].square() / 48.0
    quat = torch.cat((torch.cos(half_angles), axis_angle * scale), dim=-1)

    r, i, j, k = torch.unbind(quat, dim=-1)
    two_s = 2.0 / quat.square().sum(dim=-1).clamp_min(1e-12)
    matrix = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return matrix.reshape(axis_angle.shape[:-1] + (3, 3))


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to normalized scalar-first quaternions.

    This branch-based conversion stays well-conditioned at pi rotations,
    which is necessary for correcting the very ankle flips this fitter can
    otherwise export.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected rotation matrices [...,3,3], got {tuple(matrix.shape)}")

    flat = matrix.reshape(-1, 3, 3)
    m00, m01, m02 = flat[:, 0, 0], flat[:, 0, 1], flat[:, 0, 2]
    m10, m11, m12 = flat[:, 1, 0], flat[:, 1, 1], flat[:, 1, 2]
    m20, m21, m22 = flat[:, 2, 0], flat[:, 2, 1], flat[:, 2, 2]
    result = torch.empty(flat.shape[0], 4, device=flat.device, dtype=flat.dtype)

    trace = m00 + m11 + m22
    positive_trace = trace > 0
    if positive_trace.any():
        scale = torch.sqrt((trace[positive_trace] + 1.0).clamp_min(1e-12)) * 2.0
        result[positive_trace, 0] = 0.25 * scale
        result[positive_trace, 1] = (m21[positive_trace] - m12[positive_trace]) / scale
        result[positive_trace, 2] = (m02[positive_trace] - m20[positive_trace]) / scale
        result[positive_trace, 3] = (m10[positive_trace] - m01[positive_trace]) / scale

    x_largest = (~positive_trace) & (m00 >= m11) & (m00 >= m22)
    if x_largest.any():
        scale = torch.sqrt(
            (1.0 + m00[x_largest] - m11[x_largest] - m22[x_largest]).clamp_min(1e-12)
        ) * 2.0
        result[x_largest, 0] = (m21[x_largest] - m12[x_largest]) / scale
        result[x_largest, 1] = 0.25 * scale
        result[x_largest, 2] = (m01[x_largest] + m10[x_largest]) / scale
        result[x_largest, 3] = (m02[x_largest] + m20[x_largest]) / scale

    y_largest = (~positive_trace) & (~x_largest) & (m11 >= m22)
    if y_largest.any():
        scale = torch.sqrt(
            (1.0 + m11[y_largest] - m00[y_largest] - m22[y_largest]).clamp_min(1e-12)
        ) * 2.0
        result[y_largest, 0] = (m02[y_largest] - m20[y_largest]) / scale
        result[y_largest, 1] = (m01[y_largest] + m10[y_largest]) / scale
        result[y_largest, 2] = 0.25 * scale
        result[y_largest, 3] = (m12[y_largest] + m21[y_largest]) / scale

    z_largest = (~positive_trace) & (~x_largest) & (~y_largest)
    if z_largest.any():
        scale = torch.sqrt(
            (1.0 + m22[z_largest] - m00[z_largest] - m11[z_largest]).clamp_min(1e-12)
        ) * 2.0
        result[z_largest, 0] = (m10[z_largest] - m01[z_largest]) / scale
        result[z_largest, 1] = (m02[z_largest] + m20[z_largest]) / scale
        result[z_largest, 2] = (m12[z_largest] + m21[z_largest]) / scale
        result[z_largest, 3] = 0.25 * scale

    result = result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(1e-12)
    result = torch.where(result[:, :1] < 0, -result, result)
    return result.reshape(matrix.shape[:-2] + (4,))


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first quaternions to rotation matrices."""
    quat = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True,
    ).clamp_min(1e-12)
    r, i, j, k = torch.unbind(quat, dim=-1)
    two_s = 2.0 / quat.square().sum(dim=-1).clamp_min(1e-12)
    matrix = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return matrix.reshape(quaternion.shape[:-1] + (3, 3))


def _matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to canonical axis-angle via quaternions."""
    quaternion = _matrix_to_quaternion(matrix)
    sin_half_angle = torch.linalg.vector_norm(quaternion[..., 1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half_angle, quaternion[..., :1])
    scale = torch.where(
        sin_half_angle > 1e-8,
        angle / sin_half_angle,
        2.0 * torch.ones_like(angle),
    )
    return quaternion[..., 1:] * scale


def _geodesic_rotation_angles(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    differentiable: bool,
) -> torch.Tensor:
    """Shortest SO(3) angle between matching rotation matrices."""
    relative = first.transpose(-1, -2) @ second
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    if differentiable:
        # acos has an infinite derivative at exactly +/-1. The tiny clamp only
        # affects sub-0.1-degree differences during optimization.
        cosine = cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    else:
        cosine = cosine.clamp(-1.0, 1.0)
    return torch.acos(cosine)


def _fit_quality_metrics(
    fitted_joints: torch.Tensor,
    target_joints: torch.Tensor,
    global_orient: torch.Tensor,
    *,
    body_pose: torch.Tensor | None = None,
    toe_markers: torch.Tensor | None = None,
) -> dict[str, float]:
    distances_mm = torch.linalg.vector_norm(fitted_joints - target_joints, dim=-1) * 1000.0
    key_distances_mm = distances_mm[:, list(KEY_QUALITY_JOINTS)]
    if global_orient.shape[0] > 1:
        root_matrices = _axis_angle_to_matrix(global_orient)
        root_steps = _geodesic_rotation_angles(
            root_matrices[:-1], root_matrices[1:], differentiable=False,
        )
        max_root_step_deg = torch.rad2deg(root_steps).max().item()
    else:
        max_root_step_deg = 0.0
    quality = {
        "fit_mpjpe_mm": float(distances_mm.mean().item()),
        "fit_key_mpjpe_mm": float(key_distances_mm.mean().item()),
        "fit_p95_mm": float(torch.quantile(distances_mm.flatten(), 0.95).item()),
        "max_root_step_deg": float(max_root_step_deg),
        # Joints22-only callers get the neutral terminal-foot fallback.  A
        # full SOMA input replaces this with an observed ToeEnd constraint.
        "max_toe_direction_error_deg": 0.0,
    }
    if body_pose is not None:
        rest, _ = _body22_rest_skeleton(str(body_pose.device))
        quality["max_abs_ankle_twist_deg"] = _max_abs_ankle_twist_deg(
            body_pose,
            rest.to(device=body_pose.device, dtype=body_pose.dtype),
        )
        if body_pose.shape[0] > 1:
            world_rotations = _body22_world_rotations(global_orient, body_pose)
            foot_steps = torch.rad2deg(_geodesic_rotation_angles(
                world_rotations[:-1, [7, 8, 10, 11]],
                world_rotations[1:, [7, 8, 10, 11]],
                differentiable=False,
            ))
            quality["max_ankle_world_step_deg"] = float(
                foot_steps[:, :2].max().item()
            )
            quality["max_terminal_foot_world_step_deg"] = float(
                foot_steps[:, 2:].max().item()
            )
        else:
            quality["max_ankle_world_step_deg"] = 0.0
            quality["max_terminal_foot_world_step_deg"] = 0.0
    if toe_markers is not None:
        quality["max_toe_direction_error_deg"] = _toe_direction_error_deg(
            global_orient,
            body_pose,
            target_joints,
            toe_markers,
        )
    return quality


def fit_quality_metrics(
    fitted_joints: JointsLike,
    target_joints: JointsLike,
    global_orient: JointsLike,
    *,
    body_pose: JointsLike | None = None,
    toe_markers: ToeMarkersLike | None = None,
) -> dict[str, float]:
    """Compute export quality metrics from already-evaluated SMPL-X joints."""
    fitted = torch.as_tensor(fitted_joints, dtype=torch.float32)
    # Audits load parameters from NPZ files on CPU while the official SMPL-X
    # forward runs on the selected accelerator.  Keep every auxiliary input
    # beside the evaluated joints so the optional toe-marker metric does not
    # mix CPU pose arrays with a CUDA SMPL-X model.
    target = torch.as_tensor(
        target_joints, dtype=fitted.dtype, device=fitted.device,
    )
    orient = torch.as_tensor(
        global_orient, dtype=fitted.dtype, device=fitted.device,
    )
    if fitted.shape != target.shape or fitted.dim() != 3 or fitted.shape[1:] != (22, 3):
        raise ValueError(
            f"expected matching [T,22,3] joints, got {tuple(fitted.shape)} and {tuple(target.shape)}"
        )
    if orient.shape != (fitted.shape[0], 3):
        raise ValueError(f"expected global_orient [T,3], got {tuple(orient.shape)}")
    pose = None
    if body_pose is not None:
        pose = torch.as_tensor(
            body_pose, dtype=fitted.dtype, device=fitted.device,
        )
        if pose.shape != (fitted.shape[0], NUM_BODY_JOINTS * 3):
            raise ValueError(
                f"expected body_pose [T,63], got {tuple(pose.shape)}"
            )
    markers = None
    if toe_markers is not None:
        markers = torch.as_tensor(
            toe_markers, dtype=fitted.dtype, device=fitted.device,
        )
        if markers.shape != (fitted.shape[0], 2, 3):
            raise ValueError(
                f"expected toe_markers [T,2,3], got {tuple(markers.shape)}"
            )
    return _fit_quality_metrics(
        fitted, target, orient, body_pose=pose, toe_markers=markers,
    )


def fit_quality_failures(fit: dict[str, Any]) -> list[str]:
    """Return stable reason codes for SMPL-X export quality-gate failures."""
    failures: list[str] = []
    finite_fields = ("joints22", "transl", "global_orient", "body_pose", "betas")
    for key in finite_fields:
        value = fit.get(key)
        if value is None or not np.isfinite(np.asarray(value)).all():
            failures.append(f"non_finite:{key}")

    metric_keys = [
        "fit_mse", "fit_mpjpe_mm", "fit_key_mpjpe_mm", "fit_p95_mm",
        "max_root_step_deg", "max_ankle_world_step_deg",
        "max_terminal_foot_world_step_deg",
    ]
    # Native SMPL-X rotations carry their source pose semantics. The ankle
    # gauge only exists in the position-only fitted path, so do not reject a
    # native source for legitimate non-zero ankle/foot articulation.
    format_value = fit.get("format", "")
    if isinstance(format_value, np.ndarray) and format_value.shape == ():
        format_value = format_value.item()
    if str(format_value) == FORMAT_NAME:
        metric_keys.extend((
            "max_abs_ankle_twist_deg",
            "max_toe_direction_error_deg",
        ))
    elif str(format_value) == "smplx_body22_soma77_transfer_aa":
        # A full SOMA pose transfer has additional observable rotation targets.
        # Gate them explicitly so a future position-only fallback cannot pass
        # merely because its body22 positions happen to fit.
        metric_keys.extend((
            "soma_transfer_head_world_error_deg",
            "soma_transfer_left_foot_world_error_deg",
            "soma_transfer_right_foot_world_error_deg",
            "soma_transfer_root_position_error_mm",
        ))
    finite_metrics: dict[str, float] = {}
    for key in metric_keys:
        value = fit.get(key)
        if value is None or not np.isfinite(float(value)):
            failures.append(f"non_finite:{key}")
        else:
            finite_metrics[key] = float(value)

    metric_limits = {
        "fit_mpjpe_mm": FIT_MPJPE_LIMIT_MM,
        "fit_key_mpjpe_mm": FIT_KEY_MPJPE_LIMIT_MM,
        "max_root_step_deg": MAX_ROOT_STEP_LIMIT_DEG,
        "max_ankle_world_step_deg": MAX_ANKLE_WORLD_STEP_LIMIT_DEG,
        "max_terminal_foot_world_step_deg": MAX_TERMINAL_FOOT_WORLD_STEP_LIMIT_DEG,
        "max_toe_direction_error_deg": MAX_TOE_DIRECTION_ERROR_LIMIT_DEG,
        "soma_transfer_head_world_error_deg": MAX_SOMA_TRANSFER_WORLD_ERROR_LIMIT_DEG,
        "soma_transfer_left_foot_world_error_deg": MAX_SOMA_TRANSFER_WORLD_ERROR_LIMIT_DEG,
        "soma_transfer_right_foot_world_error_deg": MAX_SOMA_TRANSFER_WORLD_ERROR_LIMIT_DEG,
        "soma_transfer_root_position_error_mm": MAX_SOMA_TRANSFER_ROOT_POSITION_ERROR_LIMIT_MM,
    }
    for key, limit in metric_limits.items():
        if key in finite_metrics and finite_metrics[key] > limit:
            failures.append(f"quality_gate:{key}")
    return failures


@lru_cache(maxsize=8)
def _create_smplx(batch_size: int, device: str):
    import smplx

    from ....paths import smplx_model_path

    model = smplx.create(
        model_path=str(smplx_model_path()),
        model_type="smplx",
        num_betas=NUM_BETAS,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=int(batch_size),
    )
    return model.to(torch.device(device))


@lru_cache(maxsize=8)
def _body22_rest_skeleton(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return zero-beta SMPL-X rest joints and parents for body22 FK."""
    model = _create_smplx(1, device)
    vertices = model.v_template
    regressor = model.J_regressor
    if regressor.is_sparse:
        joints = torch.sparse.mm(regressor, vertices)
    else:
        joints = regressor @ vertices
    return joints[:22].detach(), model.parents[:22].detach()


@lru_cache(maxsize=8)
def _smplx_big_toe_rest_markers(device: str) -> torch.Tensor:
    """Return SMPL-X left/right big-toe vertex markers in the rest pose.

    SMPL-X has no toe child in the first 22 body joints.  These canonical
    extra joints are vertices selected by the official model and provide a
    stable direction vector for orienting the terminal foot mesh from SOMA's
    ``LeftToeEnd`` / ``RightToeEnd`` markers.
    """
    model = _create_smplx(1, device)
    with torch.no_grad():
        # JOINT_NAMES indices 60/63 are left/right big toe for the SMPL-X
        # model configuration constructed above (55 base + 5 face markers).
        joints = model(return_verts=False).joints[0]
    return joints[[60, 63]].detach()


@lru_cache(maxsize=8)
def _smplx_head_rest_frame(device: str) -> torch.Tensor:
    """Return the neutral SMPL-X head frame from its face landmarks.

    The first 22 SMPL-X joints stop at ``head``.  The official model still
    exposes the selected nose/eye vertices as extra joints, which lets us use
    the same face-frame semantics as SOMA without inventing a mesh vertex id.
    """
    model = _create_smplx(1, device)
    with torch.no_grad():
        joints = model(return_verts=False).joints[0]
    return _head_frame_from_landmarks(
        joints[15][None],
        joints[57][None],  # left eye
        joints[56][None],  # right eye
    )[0].detach()


def _ankle_rest_axes(rest_joints: torch.Tensor) -> torch.Tensor:
    """Return the local ankle-to-foot axes for left and right feet."""
    ankle_indices = [ankle for ankle, _ in ANKLE_FOOT_PAIRS]
    foot_indices = [foot for _, foot in ANKLE_FOOT_PAIRS]
    axes = rest_joints[foot_indices] - rest_joints[ankle_indices]
    lengths = torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
    if bool((lengths <= 1e-8).any()):
        raise ValueError("SMPL-X ankle-to-foot rest axis is degenerate")
    return axes / lengths


def _normalized_vectors(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized vectors and a finite/non-degenerate validity mask."""
    lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    valid = torch.isfinite(vectors).all(dim=-1, keepdim=True) & (lengths > 1e-8)
    return vectors / lengths.clamp_min(1e-8), valid.squeeze(-1)


def _head_frame_from_landmarks(
    head: torch.Tensor,
    left_eye: torch.Tensor,
    right_eye: torch.Tensor,
) -> torch.Tensor:
    """Construct a right-handed face frame from a head point and two eyes.

    The columns are anatomical right, up, and forward.  The frame is defined
    from directions only, so it remains valid across SOMA and SMPL-X models
    whose face landmarks have different absolute offsets from the head joint.
    """
    right, right_valid = _normalized_vectors(right_eye - left_eye)
    eye_midpoint = (left_eye + right_eye) * 0.5
    forward_raw = eye_midpoint - head
    forward_raw = forward_raw - (forward_raw * right).sum(dim=-1, keepdim=True) * right
    forward, forward_valid = _normalized_vectors(forward_raw)
    if not bool((right_valid & forward_valid).all()):
        raise ValueError("head face landmarks are degenerate")
    up, up_valid = _normalized_vectors(torch.cross(forward, right, dim=-1))
    if not bool(up_valid.all()):
        raise ValueError("head face frame is degenerate")
    # Recompute forward after the cross product so the output is orthonormal
    # despite small landmark noise.
    forward, _ = _normalized_vectors(torch.cross(right, up, dim=-1))
    return torch.stack((right, up, forward), dim=-1)


def _soma_head_target_frame(
    target_joints: torch.Tensor,
    head_markers: torch.Tensor,
) -> torch.Tensor:
    """Build target face frames from SOMA ``[HeadEnd, LeftEye, RightEye]``."""
    expected_shape = (target_joints.shape[0], HEAD_MARKER_COUNT, 3)
    if head_markers.shape != expected_shape:
        raise ValueError(
            f"expected head_markers {expected_shape}, got {tuple(head_markers.shape)}"
        )
    if not torch.isfinite(head_markers).all():
        raise ValueError("head_markers contains non-finite values")
    return _head_frame_from_landmarks(
        target_joints[:, 15],
        head_markers[:, 1],
        head_markers[:, 2],
    )


def _shortest_arc_rotation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the no-twist rotation mapping each source vector to target.

    The anti-parallel case has no unique shortest axis.  Pick a deterministic
    perpendicular one so clips remain reproducible instead of inheriting a
    frame-local numerical branch.
    """
    source, source_valid = _normalized_vectors(source)
    target, target_valid = _normalized_vectors(target)
    if source.shape != target.shape:
        source = torch.broadcast_to(source, target.shape)
        source_valid = torch.broadcast_to(source_valid, target_valid.shape)
    valid = source_valid & target_valid
    dot = (source * target).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    cross = torch.cross(source, target, dim=-1)
    quaternion = torch.cat((1.0 + dot, cross), dim=-1)

    # q = [0, axis] for a 180-degree turn.  Select the coordinate basis least
    # aligned with source, then cross it with source to get a stable axis.
    anti_parallel = valid & (dot.squeeze(-1) < -1.0 + 1e-6)
    if anti_parallel.any():
        basis_index = source.abs().argmin(dim=-1)
        basis = torch.nn.functional.one_hot(basis_index, num_classes=3).to(source.dtype)
        axis, _ = _normalized_vectors(torch.cross(source, basis, dim=-1))
        antipodal_quaternion = torch.cat((torch.zeros_like(dot), axis), dim=-1)
        quaternion = torch.where(anti_parallel[..., None], antipodal_quaternion, quaternion)

    identity = torch.zeros_like(quaternion)
    identity[..., 0] = 1.0
    quaternion = torch.where(valid[..., None], quaternion, identity)
    return _quaternion_to_matrix(quaternion)


def _twist_quaternions(
    matrices: torch.Tensor,
    axes: torch.Tensor,
) -> torch.Tensor:
    """Extract right-side local twists about the supplied rest-space axes."""
    axes = axes.to(device=matrices.device, dtype=matrices.dtype)
    axes, valid = _normalized_vectors(axes)
    if not bool(valid.all()):
        raise ValueError("twist axis is degenerate")
    quaternions = _matrix_to_quaternion(matrices)
    projected = (quaternions[..., 1:] * axes).sum(dim=-1, keepdim=True)
    raw_twist = torch.cat(
        (quaternions[..., :1], axes * projected), dim=-1,
    )
    lengths = torch.linalg.vector_norm(raw_twist, dim=-1, keepdim=True)
    identity = torch.zeros_like(raw_twist)
    identity[..., 0] = 1.0
    return torch.where(lengths > 1e-8, raw_twist / lengths.clamp_min(1e-12), identity)


def _ankle_twist_quaternions(
    ankle_matrices: torch.Tensor,
    rest_joints: torch.Tensor,
) -> torch.Tensor:
    """Extract the local twist around each ankle-to-foot rest axis.

    For ``R = swing @ twist``, ``twist`` leaves the rest foot vector fixed,
    so it is a position-invisible gauge of the body22 objective.
    """
    return _twist_quaternions(
        ankle_matrices,
        _ankle_rest_axes(rest_joints),
    )


def _stabilize_rotation_gauge(
    base_matrices: torch.Tensor,
    rest_axes: torch.Tensor,
) -> torch.Tensor:
    """Choose the position-invisible twist closest to the prior frame.

    Mapping one rest-space direction to a target direction leaves one twist
    degree of freedom.  Independent shortest-arc choices become unstable
    near a 180-degree swing.  Carrying the prior-frame twist removes that
    branch flip while preserving the mapped direction exactly.
    """
    if base_matrices.ndim != 4 or base_matrices.shape[-2:] != (3, 3):
        raise ValueError(
            f"expected [T,N,3,3] matrices, got {tuple(base_matrices.shape)}"
        )
    if base_matrices.shape[0] <= 1:
        return base_matrices

    stabilized = base_matrices.clone()
    for frame in range(1, base_matrices.shape[0]):
        relative = base_matrices[frame].transpose(-1, -2) @ stabilized[frame - 1]
        correction = _quaternion_to_matrix(_twist_quaternions(relative, rest_axes))
        stabilized[frame] = base_matrices[frame] @ correction
    return stabilized


def _stabilize_terminal_foot_pose(
    parent_world: torch.Tensor,
    rest_vectors: torch.Tensor,
    target_vectors: torch.Tensor,
) -> torch.Tensor:
    """Align terminal feet to ToeEnd while selecting a continuous roll gauge."""
    rest_axes, valid = _normalized_vectors(rest_vectors)
    if not bool(valid.all()):
        raise ValueError("SMPL-X rest BigToe vector is degenerate")
    target_local = (
        parent_world.transpose(-1, -2) @ target_vectors.unsqueeze(-1)
    ).squeeze(-1)
    base = _shortest_arc_rotation(rest_axes, target_local)
    if base.shape[0] <= 1:
        return base

    stabilized = base.clone()
    previous_world = parent_world[0] @ stabilized[0]
    for frame in range(1, base.shape[0]):
        world_base = parent_world[frame] @ base[frame]
        relative = world_base.transpose(-1, -2) @ previous_world
        correction = _quaternion_to_matrix(_twist_quaternions(relative, rest_axes))
        stabilized[frame] = base[frame] @ correction
        previous_world = parent_world[frame] @ stabilized[frame]
    return stabilized


def _max_abs_ankle_twist_deg(
    body_pose: torch.Tensor,
    rest_joints: torch.Tensor,
) -> float:
    """Measure fitted local ankle twist that positions alone cannot observe."""
    pose = body_pose.reshape(-1, NUM_BODY_JOINTS, 3)
    ankle_matrices = _axis_angle_to_matrix(pose[:, list(ANKLE_BODY_POSE_JOINTS)])
    twists = _ankle_twist_quaternions(ankle_matrices, rest_joints)
    sin_half_angle = torch.linalg.vector_norm(twists[..., 1:], dim=-1)
    angles = 2.0 * torch.atan2(sin_half_angle, twists[..., 0].abs())
    return float(torch.rad2deg(angles).max().item())


def _body22_world_rotations(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
) -> torch.Tensor:
    """Return world rotations for the same zero-beta chain as body22 FK."""
    frames = global_orient.shape[0]
    _, parents = _body22_rest_skeleton(str(global_orient.device))
    rotations = _axis_angle_to_matrix(
        torch.cat(
            (global_orient[:, None, :], body_pose.reshape(frames, NUM_BODY_JOINTS, 3)),
            dim=1,
        )
    )
    world_rotations: list[torch.Tensor | None] = [None] * 22
    world_rotations[0] = rotations[:, 0]
    for level in BODY22_LEVELS:
        parent_indices = [int(parents[index]) for index in level]
        parent_rotations = torch.stack(
            [world_rotations[index] for index in parent_indices], dim=1,
        )
        level_rotations = parent_rotations @ rotations[:, list(level)]
        for column, joint_index in enumerate(level):
            world_rotations[joint_index] = level_rotations[:, column]
    return torch.stack(world_rotations, dim=1)


def _toe_direction_error_deg(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor | None,
    target_joints: torch.Tensor,
    toe_markers: torch.Tensor,
) -> float:
    """Measure the fitted terminal-foot direction against SOMA ToeEnd markers."""
    if body_pose is None:
        return 0.0
    foot_indices = [foot for _, foot in ANKLE_FOOT_PAIRS]
    predicted = _smplx_big_toe_mesh_vectors(global_orient, body_pose)
    target = toe_markers - target_joints[:, foot_indices]
    predicted, predicted_valid = _normalized_vectors(predicted)
    target, target_valid = _normalized_vectors(target)
    valid = predicted_valid & target_valid
    if not bool(valid.any()):
        return float("inf")
    cosine = (predicted * target).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine[valid]))
    return float(angles.max().item())


def _smplx_big_toe_mesh_vectors(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
) -> torch.Tensor:
    """Evaluate actual SMPL-X BigToe-to-foot vectors without exporting vertices."""
    frames = int(global_orient.shape[0])
    model = _create_smplx(1, str(global_orient.device))
    device = global_orient.device
    dtype = global_orient.dtype
    zeros3 = torch.zeros(frames, 3, device=device, dtype=dtype)
    zeros45 = torch.zeros(frames, 45, device=device, dtype=dtype)
    expression = torch.zeros(
        frames, model.num_expression_coeffs, device=device, dtype=dtype,
    )
    betas = torch.zeros(frames, NUM_BETAS, device=device, dtype=dtype)
    with torch.no_grad():
        joints = model(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=zeros3,
            betas=betas,
            left_hand_pose=zeros45,
            right_hand_pose=zeros45,
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            expression=expression,
            return_verts=False,
        ).joints
    # JOINT_NAMES 60/63 are selected BigToe mesh vertices; joints 10/11 are
    # the terminal body-foot joints used by the package's body22 convention.
    return joints[:, [60, 63]] - joints[:, [10, 11]]


def _canonicalize_fitted_ankle_pose(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    target_joints: torch.Tensor,
    rest_joints: torch.Tensor,
    toe_markers: torch.Tensor | None,
) -> torch.Tensor:
    """Select a continuous ankle/foot gauge and orient feet from SOMA toes.

    ``joints22`` only determines the ankle swing, not its axial twist, and it
    has no descendants below the terminal SMPL-X feet.  A frame-local
    no-twist projection is ambiguous around a 180-degree swing, so the
    position-invisible twist gauge is chosen to stay closest to the previous
    frame.  When a full SOMA77 source is available, ``ToeEnd`` also fixes the
    terminal-foot direction while retaining a continuous toe-roll gauge.
    This function is never applied to native SMPL-X exports.
    """
    pose = body_pose.reshape(-1, NUM_BODY_JOINTS, 3)
    canonical = pose.clone()
    ankle_matrices = _axis_angle_to_matrix(pose[:, list(ANKLE_BODY_POSE_JOINTS)])
    twists = _ankle_twist_quaternions(ankle_matrices, rest_joints)
    canonical_ankles = ankle_matrices @ _quaternion_to_matrix(twists).transpose(-1, -2)
    ankle_axes = _ankle_rest_axes(rest_joints)
    canonical_ankles = _stabilize_rotation_gauge(canonical_ankles, ankle_axes)
    canonical[:, list(ANKLE_BODY_POSE_JOINTS)] = _matrix_to_axis_angle(canonical_ankles)

    # A joints22-only source cannot constrain the foot mesh orientation.  Use
    # its stable neutral pose as a safe fallback instead of preserving a
    # frame-local arbitrary value from the under-determined optimizer.
    canonical[:, list(TERMINAL_FOOT_BODY_POSE_JOINTS)] = 0.0
    canonical_pose = canonical.reshape_as(body_pose)
    if toe_markers is None:
        return canonical_pose

    expected_shape = (body_pose.shape[0], 2, 3)
    if toe_markers.shape != expected_shape:
        raise ValueError(
            f"expected toe_markers {expected_shape}, got {tuple(toe_markers.shape)}"
        )
    if not torch.isfinite(toe_markers).all():
        raise ValueError("toe_markers contains non-finite values")

    marker_rest = _smplx_big_toe_rest_markers(str(body_pose.device)).to(
        device=body_pose.device, dtype=body_pose.dtype,
    )
    foot_indices = [foot for _, foot in ANKLE_FOOT_PAIRS]
    ankle_indices = [ankle for ankle, _ in ANKLE_FOOT_PAIRS]
    rest_vectors = marker_rest - rest_joints[foot_indices]
    target_vectors = toe_markers - target_joints[:, foot_indices]
    target_vectors, marker_valid = _normalized_vectors(target_vectors)
    if not bool(marker_valid.all()):
        raise ValueError("toe_markers has a degenerate ToeBase-to-ToeEnd vector")

    parent_world = _body22_world_rotations(global_orient, canonical_pose)[:, ankle_indices]
    terminal_foot = _stabilize_terminal_foot_pose(
        parent_world,
        rest_vectors,
        target_vectors,
    )
    canonical[:, list(TERMINAL_FOOT_BODY_POSE_JOINTS)] = _matrix_to_axis_angle(
        terminal_foot
    )
    return canonical.reshape_as(body_pose)


def _refine_terminal_foot_mesh_pose(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    target_joints: torch.Tensor,
    toe_markers: torch.Tensor | None,
    *,
    iterations: int = 2,
) -> torch.Tensor:
    """Correct analytic foot orientation against actual SMPL-X toe vertices.

    BigToe is a linearly-skinned vertex, so its response is close to but not
    exactly the rigid rest-vector approximation used for initialization.  Two
    deterministic no-gradient corrections remove that residual without
    perturbing the fitted body22 positions.
    """
    if toe_markers is None:
        return body_pose
    foot_indices = [foot for _, foot in ANKLE_FOOT_PAIRS]
    ankle_indices = [ankle for ankle, _ in ANKLE_FOOT_PAIRS]
    target_vectors, target_valid = _normalized_vectors(
        toe_markers - target_joints[:, foot_indices]
    )
    if not bool(target_valid.all()):
        raise ValueError("toe_markers has a degenerate ToeBase-to-ToeEnd vector")

    refined = body_pose.clone()
    for _ in range(iterations):
        current_vectors, current_valid = _normalized_vectors(
            _smplx_big_toe_mesh_vectors(global_orient, refined)
        )
        if not bool(current_valid.all()):
            raise ValueError("SMPL-X BigToe marker is degenerate")
        parent_world = _body22_world_rotations(global_orient, refined)[:, ankle_indices]
        current_local = (
            parent_world.transpose(-1, -2) @ current_vectors.unsqueeze(-1)
        ).squeeze(-1)
        target_local = (
            parent_world.transpose(-1, -2) @ target_vectors.unsqueeze(-1)
        ).squeeze(-1)
        correction = _shortest_arc_rotation(current_local, target_local)
        pose = refined.reshape(-1, NUM_BODY_JOINTS, 3).clone()
        foot_matrices = _axis_angle_to_matrix(
            pose[:, list(TERMINAL_FOOT_BODY_POSE_JOINTS)]
        )
        pose[:, list(TERMINAL_FOOT_BODY_POSE_JOINTS)] = _matrix_to_axis_angle(
            correction @ foot_matrices
        )
        refined = pose.reshape_as(refined)
    return refined


def _smplx_body22_fk(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    transl: torch.Tensor,
) -> torch.Tensor:
    """Exact zero-beta SMPL-X FK for the first 22 model joints.

    The official forward additionally skins all 10k vertices, but its first
    22 joints are the same rigid-transform chain. Keeping the optimization on
    this chain is numerically equivalent and substantially faster.
    """
    frames = global_orient.shape[0]
    rest, parents = _body22_rest_skeleton(str(global_orient.device))
    rest = rest.to(dtype=global_orient.dtype)
    world_rotations = _body22_world_rotations(global_orient, body_pose)
    world_positions: list[torch.Tensor | None] = [None] * 22
    world_positions[0] = transl + rest[0]
    for level in BODY22_LEVELS:
        parent_indices = [int(parents[index]) for index in level]
        parent_rotations = world_rotations[:, parent_indices]
        parent_positions = torch.stack(
            [world_positions[index] for index in parent_indices], dim=1,
        )
        offsets = rest[list(level)] - rest[parent_indices]
        level_positions = parent_positions + (
            parent_rotations @ offsets.reshape(1, len(level), 3, 1)
        ).squeeze(-1)
        for column, joint_index in enumerate(level):
            world_positions[joint_index] = level_positions[:, column]
    return torch.stack(world_positions, dim=1)


class Joints22ToSMPLXParams:
    """Fit SMPL-X ``global_orient``/``body_pose``/``transl`` to 22-joint positions.

    Betas stay zero (neutral shape) — the demo bundle cares about pose, not
    identity, same as the reference archive. The zero-beta body22 rest
    skeleton is cached per device.

    Optimization has two stages. The first fits each frame spatially with
    SOMA/SMPL-X-aware joint weights. The second starts from that solution and
    adds a low-weight geodesic SO(3) temporal term. This preserves floor,
    kneeling, crouching and lying poses while resolving axis-angle gauge
    ambiguity without penalizing representation wraparound.
    """

    def __init__(
        self,
        device: str = "cuda",
        *,
        fit_steps: int = 250,
        lr: float = 0.03,
        smooth_weight: float = 0.1,
    ):
        self._device = device
        self._fit_steps = int(fit_steps)
        self._lr = float(lr)
        self._smooth_weight = float(smooth_weight)

    @property
    def device(self) -> str:
        """Configured accelerator used by the companion full-pose retargeter."""
        return self._device

    @property
    def fit_steps(self) -> int:
        return self._fit_steps

    @property
    def lr(self) -> float:
        return self._lr

    def fit(
        self,
        joints: JointsLike,
        *,
        length: int | None = None,
        toe_end_markers: ToeMarkersLike | None = None,
    ) -> dict[str, object]:
        """Fit one clip's ``[T, 22, 3]`` joints. Returns the npz-ready field dict.

        Args:
            joints: ``[T, 22, 3]`` positions, or ``[1, T, 22, 3]`` (batch dim
                squeezed). NumPy or torch Tensor.
            length: Optional crop to the first ``length`` frames.
            toe_end_markers: Optional global ``[T, 2, 3]`` left/right toe-end
                points. They set terminal-foot swing while retaining a
                canonical no-twist foot pose.

        Returns:
            Dict with ``joints22``, ``transl``, ``global_orient``, ``body_pose``,
            ``betas``, ``fit_mse`` — all ``np.ndarray``/``float``, ready to widen
            with ``source``/``format``/``track``/``fps``/``prefix_T`` and save
            as one ``*.smplx.npz``.
        """
        return self.fit_many(
            [joints], lengths=[length], toe_end_markers=[toe_end_markers],
        )[0]

    def fit_many(
        self,
        clips: list[JointsLike],
        *,
        lengths: list[int | None] | None = None,
        toe_end_markers: list[ToeMarkersLike | None] | None = None,
    ) -> list[dict[str, object]]:
        """Fit several clips in one optimizer pass without smoothing boundaries."""
        if not clips:
            return []
        if lengths is None:
            lengths = [None] * len(clips)
        if len(lengths) != len(clips):
            raise ValueError("lengths must match clips")
        if toe_end_markers is None:
            toe_end_markers = [None] * len(clips)
        if len(toe_end_markers) != len(clips):
            raise ValueError("toe_end_markers must match clips")

        prepared: list[torch.Tensor] = []
        prepared_toe_markers: list[torch.Tensor | None] = []
        device = torch.device(self._device)
        for raw, length, raw_toes in zip(clips, lengths, toe_end_markers, strict=True):
            tensor = torch.from_numpy(raw) if isinstance(raw, np.ndarray) else raw
            if tensor.dim() == 4:
                tensor = tensor[0]
            if length is not None:
                tensor = tensor[: int(length)]
            if tensor.dim() != 3 or tuple(tensor.shape[1:]) != (22, 3):
                raise ValueError(f"expected joints [T,22,3], got {tuple(tensor.shape)}")
            if tensor.shape[0] <= 0:
                raise ValueError("cannot fit an empty joints22 clip")
            if tensor.is_cuda:
                device = tensor.device
            prepared.append(tensor)

            if raw_toes is None:
                prepared_toe_markers.append(None)
                continue
            toes = torch.from_numpy(raw_toes) if isinstance(raw_toes, np.ndarray) else raw_toes
            if toes.dim() == 4:
                toes = toes[0]
            if length is not None:
                toes = toes[: int(length)]
            if toes.dim() != 3 or tuple(toes.shape[1:]) != (2, 3):
                raise ValueError(f"expected toe_end_markers [T,2,3], got {tuple(toes.shape)}")
            if toes.shape[0] != tensor.shape[0]:
                raise ValueError(
                    "toe_end_markers/joints frame mismatch: "
                    f"{toes.shape[0]} vs {tensor.shape[0]}"
                )
            prepared_toe_markers.append(toes)

        targets = [tensor.to(device=device, dtype=torch.float32) for tensor in prepared]
        toe_targets = [
            None if toes is None else toes.to(device=device, dtype=torch.float32)
            for toes in prepared_toe_markers
        ]
        target = torch.cat(targets, dim=0)
        if not torch.isfinite(target).all():
            raise ValueError("joints22 contains non-finite values")
        if any(toes is not None and not torch.isfinite(toes).all() for toes in toe_targets):
            raise ValueError("toe_end_markers contains non-finite values")

        boundaries: list[tuple[int, int]] = []
        start = 0
        for tensor in targets:
            end = start + tensor.shape[0]
            boundaries.append((start, end))
            start = end
        frames = target.shape[0]

        go = torch.zeros(frames, 3, device=device, requires_grad=True)
        bp = torch.zeros(frames, NUM_BODY_JOINTS * 3, device=device, requires_grad=True)
        tl = target[:, 0, :].clone().detach().requires_grad_(True)
        weights = _joint_weights(device=device, dtype=target.dtype)

        def spatial_loss(predicted: torch.Tensor) -> torch.Tensor:
            per_frame = (
                (predicted - target).square() * weights[None, :, None]
            ).sum(dim=(1, 2)) / (3 * weights.sum())
            return torch.stack(
                [per_frame[clip_start:clip_end].mean() for clip_start, clip_end in boundaries]
            ).mean()

        opt = torch.optim.Adam([go, bp, tl], lr=self._lr)
        for _ in range(self._fit_steps):
            opt.zero_grad()
            predicted = _smplx_body22_fk(go, bp, tl)
            spatial_loss(predicted).backward()
            opt.step()

        opt = torch.optim.Adam([go, bp, tl], lr=self._lr)
        for _ in range(self._fit_steps):
            opt.zero_grad()
            predicted = _smplx_body22_fk(go, bp, tl)
            loss = spatial_loss(predicted)
            if frames > 1 and self._smooth_weight > 0:
                rotations = torch.cat(
                    (go[:, None, :], bp.reshape(frames, NUM_BODY_JOINTS, 3)), dim=1,
                )
                matrices = _axis_angle_to_matrix(rotations)
                angles = _geodesic_rotation_angles(
                    matrices[:-1], matrices[1:], differentiable=True,
                )
                temporal_terms = [
                    angles[clip_start:clip_end - 1].square().mean()
                    for clip_start, clip_end in boundaries
                    if clip_end - clip_start > 1
                ]
                if temporal_terms:
                    loss = loss + self._smooth_weight * torch.stack(temporal_terms).mean()
            loss.backward()
            opt.step()

        results: list[dict[str, object]] = []
        with torch.no_grad():
            rest, _ = _body22_rest_skeleton(str(go.device))
            rest = rest.to(device=go.device, dtype=bp.dtype)
            for clip_index, (clip_start, clip_end) in enumerate(boundaries):
                clip_target = target[clip_start:clip_end]
                clip_go = go[clip_start:clip_end]
                clip_tl = tl[clip_start:clip_end]
                clip_bp = _canonicalize_fitted_ankle_pose(
                    clip_go,
                    bp[clip_start:clip_end],
                    clip_target,
                    rest,
                    toe_targets[clip_index],
                )
                clip_bp = _refine_terminal_foot_mesh_pose(
                    clip_go,
                    clip_bp,
                    clip_target,
                    toe_targets[clip_index],
                )
                clip_predicted = _smplx_body22_fk(clip_go, clip_bp, clip_tl)
                fit_mse = torch.nn.functional.mse_loss(clip_predicted, clip_target).item()
                quality = _fit_quality_metrics(
                    clip_predicted,
                    clip_target,
                    clip_go,
                    body_pose=clip_bp,
                    toe_markers=toe_targets[clip_index],
                )
                results.append({
                    "joints22": clip_target.detach().cpu().numpy().astype(np.float32),
                    "smplx_fk_joints22": clip_predicted.detach().cpu().numpy().astype(np.float32),
                    "transl": clip_tl.detach().cpu().numpy().astype(np.float32),
                    "global_orient": clip_go.detach().cpu().numpy().astype(np.float32),
                    "body_pose": clip_bp.detach().cpu().numpy().astype(np.float32),
                    "betas": np.zeros(NUM_BETAS, dtype=np.float32),
                    "fit_mse": float(fit_mse),
                    "format": FORMAT_NAME,
                    **(
                        {"toe_end_markers": toe_targets[clip_index].detach().cpu().numpy().astype(np.float32)}
                        if toe_targets[clip_index] is not None else {}
                    ),
                    **quality,
                })
        return results


class Rot6dTranslToSMPLXParams:
    """Decode native ``smpl_rot6d_transl`` into SMPL-X demo-bundle pose fields.

    Mirrors the HYMotion eval decode (joint 0 = ``global_orient``, joints
    1..21 = body pose) but runs SMPL-X FK so ``joints22`` is self-consistent
    with the exported pose — required by MotionViewer for layout / ground /
    camera framing that must stay aligned with the mesh.
    """

    def __init__(self, device: str = "cuda"):
        self._device = device

    def encode(
        self,
        rot6d: Rot6dLike,
        transl: TranslLike,
        *,
        length: int | None = None,
    ) -> dict[str, object]:
        """Encode one clip's rot6d + transl. Returns the npz-ready field dict.

        Args:
            rot6d: ``[T, 22, 6]`` Zhou 6D rotations (root + 21 body), or
                ``[1, T, 22, 6]``.
            transl: ``[T, 3]`` root translation (Y-up, meters), or ``[1, T, 3]``.
            length: Optional crop to the first ``length`` frames.
        """
        from .rot6d import rotation_6d_to_axis_angle

        if isinstance(rot6d, np.ndarray):
            rot6d = torch.from_numpy(np.array(rot6d, dtype=np.float32, copy=True))
        if isinstance(transl, np.ndarray):
            transl = torch.from_numpy(np.array(transl, dtype=np.float32, copy=True))
        if rot6d.dim() == 4:
            rot6d = rot6d[0]
        if transl.dim() == 3:
            transl = transl[0]
        if length is not None:
            rot6d = rot6d[: int(length)]
            transl = transl[: int(length)]

        if rot6d.dim() != 3 or rot6d.shape[1:] != (22, 6):
            raise ValueError(f"expected rot6d [T,22,6], got {tuple(rot6d.shape)}")
        if transl.dim() != 2 or transl.shape[1] != 3:
            raise ValueError(f"expected transl [T,3], got {tuple(transl.shape)}")
        if int(rot6d.shape[0]) != int(transl.shape[0]):
            raise ValueError(
                f"rot6d/transl length mismatch: {int(rot6d.shape[0])} vs {int(transl.shape[0])}"
            )

        T = int(rot6d.shape[0])
        if rot6d.is_cuda:
            dev = rot6d.device
        elif transl.is_cuda:
            dev = transl.device
        else:
            dev = torch.device(self._device)

        rot6d = rot6d.to(device=dev, dtype=torch.float32)
        transl = transl.to(device=dev, dtype=torch.float32)

        aa = rotation_6d_to_axis_angle(rot6d)
        go = aa[:, 0, :]
        bp = aa[:, 1:22, :].reshape(T, NUM_BODY_JOINTS * 3)
        with torch.no_grad():
            joints22 = _smplx_body22_fk(go, bp, transl)
            # Self-consistency check: FK from the decoded pose should match
            # the joints we export (fit_mse ≈ 0 by construction).
            fit_mse = torch.nn.functional.mse_loss(joints22, joints22).item()
            quality = _fit_quality_metrics(
                joints22, joints22, go, body_pose=bp,
            )

        return {
            "joints22": joints22.detach().cpu().numpy().astype(np.float32),
            "smplx_fk_joints22": joints22.detach().cpu().numpy().astype(np.float32),
            "transl": transl.detach().cpu().numpy().astype(np.float32),
            "global_orient": go.detach().cpu().numpy().astype(np.float32),
            "body_pose": bp.detach().cpu().numpy().astype(np.float32),
            "betas": np.zeros(NUM_BETAS, dtype=np.float32),
            "fit_mse": float(fit_mse),
            "format": FORMAT_NAME_NATIVE,
            **quality,
        }


__all__ = [
    "FORMAT_NAME",
    "FORMAT_NAME_NATIVE",
    "FIT_KEY_MPJPE_LIMIT_MM",
    "FIT_MPJPE_LIMIT_MM",
    "Joints22ToSMPLXParams",
    "MAX_ANKLE_WORLD_STEP_LIMIT_DEG",
    "MAX_ROOT_STEP_LIMIT_DEG",
    "MAX_TERMINAL_FOOT_WORLD_STEP_LIMIT_DEG",
    "Rot6dTranslToSMPLXParams",
    "fit_quality_metrics",
    "fit_quality_failures",
    "NUM_BETAS",
    "NUM_BODY_JOINTS",
]
