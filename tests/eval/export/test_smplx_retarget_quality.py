"""Regression tests for SMPL-X fitting and SOMA retarget quality."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from semoco_generator.eval.motion_ops.utils.smplx_fit import (
    FORMAT_NAME,
    FORMAT_NAME_NATIVE,
    NUM_BETAS,
    Joints22ToSMPLXParams,
    Rot6dTranslToSMPLXParams,
    _canonicalize_fitted_ankle_pose,
    _create_smplx,
    _max_abs_ankle_twist_deg,
    _smplx_body22_fk,
    _body22_rest_skeleton,
    fit_quality_failures,
    fit_quality_metrics,
)
from semoco_generator.eval.soma_pose import (
    soma_pose_dynamics,
    soma_pose_dynamics_failures,
)

pytestmark = pytest.mark.requires_smplx


def _synthetic_joints22(frames: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    joints = np.zeros((frames, 22, 3), dtype=np.float32)
    joints[:, 1:] = 0.15 * rng.normal(size=(frames, 21, 3)).astype(np.float32)
    joints[:, 0] = np.cumsum(
        0.01 * rng.normal(size=(frames, 3)).astype(np.float32), axis=0,
    )
    return joints


def _synthetic_rot6d_transl(
    frames: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    rot6d = np.broadcast_to(identity, (frames, 22, 6)).copy()
    rot6d += 0.02 * rng.normal(size=rot6d.shape).astype(np.float32)
    transl = np.cumsum(
        0.01 * rng.normal(size=(frames, 3)).astype(np.float32), axis=0,
    )
    return rot6d, transl


def test_joints22_fitter_returns_package_schema() -> None:
    fitter = Joints22ToSMPLXParams(device="cpu", fit_steps=3, lr=0.05)
    fit = fitter.fit(_synthetic_joints22(5, seed=0))

    assert set(fit) >= {
        "joints22", "transl", "global_orient", "body_pose", "betas",
        "fit_mse", "format",
    }
    assert fit["joints22"].shape == (5, 22, 3)
    assert fit["transl"].shape == (5, 3)
    assert fit["global_orient"].shape == (5, 3)
    assert fit["body_pose"].shape == (5, 63)
    assert fit["betas"].shape == (NUM_BETAS,)
    assert np.allclose(fit["betas"], 0.0)
    assert isinstance(fit["fit_mse"], float)
    assert fit["format"] == FORMAT_NAME


def test_joints22_fitter_accepts_batched_cropped_input() -> None:
    fitter = Joints22ToSMPLXParams(device="cpu", fit_steps=2, lr=0.05)
    fit = fitter.fit(_synthetic_joints22(8, seed=1)[None], length=4)

    assert fit["joints22"].shape == (4, 22, 3)
    assert fit["body_pose"].shape == (4, 63)


def test_native_rot6d_encoder_returns_package_schema() -> None:
    encoder = Rot6dTranslToSMPLXParams(device="cpu")
    rot6d, transl = _synthetic_rot6d_transl(5, seed=10)
    fit = encoder.encode(rot6d, transl)

    assert fit["joints22"].shape == (5, 22, 3)
    assert fit["transl"].shape == (5, 3)
    assert fit["global_orient"].shape == (5, 3)
    assert fit["body_pose"].shape == (5, 63)
    assert fit["betas"].shape == (NUM_BETAS,)
    assert np.allclose(fit["betas"], 0.0)
    assert fit["format"] == FORMAT_NAME_NATIVE
    assert fit["fit_mse"] == 0.0
    np.testing.assert_allclose(fit["transl"], transl, atol=1e-6)


def test_native_rot6d_joints_are_fk_self_consistent() -> None:
    encoder = Rot6dTranslToSMPLXParams(device="cpu")
    rot6d, transl = _synthetic_rot6d_transl(4, seed=11)
    fit = encoder.encode(rot6d, transl)

    model = _create_smplx(4, "cpu")
    with torch.no_grad():
        output = model(
            global_orient=torch.from_numpy(fit["global_orient"]),
            body_pose=torch.from_numpy(fit["body_pose"]),
            transl=torch.from_numpy(fit["transl"]),
            betas=torch.zeros(4, NUM_BETAS),
        )
    expected = output.joints[:, :22].numpy().astype(np.float32)

    np.testing.assert_allclose(fit["joints22"], expected, atol=1e-5)


def test_native_rot6d_encoder_accepts_batched_cropped_input() -> None:
    encoder = Rot6dTranslToSMPLXParams(device="cpu")
    rot6d, transl = _synthetic_rot6d_transl(8, seed=12)
    fit = encoder.encode(rot6d[None], transl[None], length=3)

    assert fit["joints22"].shape == (3, 22, 3)
    assert fit["body_pose"].shape == (3, 63)
    assert fit["format"] == FORMAT_NAME_NATIVE


def test_root_step_uses_geodesic_rotation_distance() -> None:
    joints = np.zeros((2, 22, 3), dtype=np.float32)
    orient = np.zeros((2, 3), dtype=np.float32)
    orient[:, 1] = np.deg2rad([179.0, -179.0])

    body_pose = np.zeros((2, 63), dtype=np.float32)
    quality = fit_quality_metrics(joints, joints, orient, body_pose=body_pose)

    assert quality["max_root_step_deg"] == pytest.approx(2.0, abs=0.01)
    fit = {
        "joints22": joints,
        "transl": np.zeros((2, 3), dtype=np.float32),
        "global_orient": orient,
        "body_pose": body_pose,
        "betas": np.zeros(16, dtype=np.float32),
        "fit_mse": 0.0,
        **quality,
    }
    assert fit_quality_failures(fit) == []


def test_world_foot_step_gate_rejects_post_fit_rotation_flip() -> None:
    """A foot mesh flip must fail even when root and positions are stable."""
    joints = np.zeros((2, 22, 3), dtype=np.float32)
    orient = np.zeros((2, 3), dtype=np.float32)
    body_pose = np.zeros((2, 63), dtype=np.float32)
    body_pose[1, 18:21] = np.array([0.0, 0.0, np.pi / 2.0], dtype=np.float32)
    quality = fit_quality_metrics(joints, joints, orient, body_pose=body_pose)
    fit = {
        "joints22": joints,
        "transl": np.zeros((2, 3), dtype=np.float32),
        "global_orient": orient,
        "body_pose": body_pose,
        "betas": np.zeros(16, dtype=np.float32),
        "fit_mse": 0.0,
        **quality,
    }

    assert quality["max_ankle_world_step_deg"] > 45.0
    assert quality["max_terminal_foot_world_step_deg"] > 45.0
    assert "quality_gate:max_ankle_world_step_deg" in fit_quality_failures(fit)
    assert "quality_gate:max_terminal_foot_world_step_deg" in fit_quality_failures(fit)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA SMPL-X runtime",
)
def test_quality_metrics_accepts_cpu_serialized_inputs_for_cuda_fk() -> None:
    """The independent FK audit must handle NPZ arrays loaded on CPU."""
    frames = 2
    device = "cuda:0"
    global_orient = torch.zeros(frames, 3, device=device)
    body_pose = torch.zeros(frames, 63, device=device)
    transl = torch.zeros(frames, 3, device=device)
    fitted = _smplx_body22_fk(global_orient, body_pose, transl)
    target = fitted.cpu().numpy()
    toe_markers = target[:, [10, 11]].copy()
    toe_markers[:, :, 0] += 0.1

    quality = fit_quality_metrics(
        fitted,
        target,
        global_orient.cpu().numpy(),
        body_pose=body_pose.cpu().numpy(),
        toe_markers=toe_markers,
    )

    assert np.isfinite(quality["max_toe_direction_error_deg"])


def test_ankle_axial_twist_is_unobservable_but_canonicalized() -> None:
    """A foot-axis twist must not move body22, but must not survive export."""
    import torch

    rest, _ = _body22_rest_skeleton("cpu")
    axis = rest[10] - rest[7]
    axis = axis / torch.linalg.vector_norm(axis)
    frames = 2
    global_orient = torch.zeros(frames, 3)
    transl = torch.zeros(frames, 3)
    pose = torch.zeros(frames, 63)
    pose[:, 18:21] = axis * (np.pi / 2.0)
    target = _smplx_body22_fk(global_orient, torch.zeros_like(pose), transl)
    twisted = _smplx_body22_fk(global_orient, pose, transl)

    assert torch.allclose(twisted, target, atol=1e-6)
    assert _max_abs_ankle_twist_deg(pose, rest) == pytest.approx(90.0, abs=0.1)

    canonical = _canonicalize_fitted_ankle_pose(
        global_orient, pose, target, rest, None,
    )
    assert _max_abs_ankle_twist_deg(canonical, rest) < 1e-3
    assert torch.allclose(
        _smplx_body22_fk(global_orient, canonical, transl), target, atol=1e-6,
    )


def test_raw_foot_dynamics_rejects_implausible_single_frame_jump() -> None:
    joints = np.zeros((2, 77, 3), dtype=np.float32)
    joints[:, 70, 0] = 0.0
    joints[:, 71, 0] = 0.1
    joints[:, 75, 0] = 1.0
    joints[:, 76, 0] = 1.1
    joints[1, 75, 0] += 0.3
    joints[1, 76, 0] = joints[1, 75, 0]
    joints[1, 76, 1] = 0.1

    dynamics = soma_pose_dynamics(joints, fps=50.0)
    failures = soma_pose_dynamics_failures(dynamics)

    assert dynamics["raw_max_foot_speed_mps"] > 10.0
    assert dynamics["raw_max_toe_angular_velocity_deg_s"] > 900.0
    assert "quality_gate:raw_foot_speed_mps" in failures
    assert "quality_gate:raw_toe_angular_velocity_deg_s" in failures
