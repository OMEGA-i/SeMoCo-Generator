"""Contract tests for full-SOMA-pose SMPL-X exports."""

from __future__ import annotations

import numpy as np

from semoco_generator.eval.motion_ops.utils.smplx_fit import fit_quality_failures


def test_full_pose_quality_gate_rejects_lost_terminal_foot_orientation() -> None:
    """The transfer format cannot silently degrade into a position-only fit."""
    fit = {
        "joints22": np.zeros((2, 22, 3), dtype=np.float32),
        "transl": np.zeros((2, 3), dtype=np.float32),
        "global_orient": np.zeros((2, 3), dtype=np.float32),
        "body_pose": np.zeros((2, 63), dtype=np.float32),
        "betas": np.zeros(16, dtype=np.float32),
        "fit_mse": 0.0,
        "fit_mpjpe_mm": 0.0,
        "fit_key_mpjpe_mm": 0.0,
        "fit_p95_mm": 0.0,
        "max_root_step_deg": 0.0,
        "max_ankle_world_step_deg": 0.0,
        "max_terminal_foot_world_step_deg": 0.0,
        "format": "smplx_body22_soma77_transfer_aa",
        "soma_transfer_head_world_error_deg": 0.0,
        "soma_transfer_left_foot_world_error_deg": 0.0,
        "soma_transfer_right_foot_world_error_deg": 5.1,
        "soma_transfer_root_position_error_mm": 0.0,
    }

    assert "quality_gate:soma_transfer_right_foot_world_error_deg" in fit_quality_failures(fit)
