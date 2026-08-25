"""Conversions and temporal normalization for the SOMA/TMR track."""
from __future__ import annotations
import numpy as np
from ...metrics import resample_fps
from ..smpl_hml.conversion import soma77_to_joints22


def smpl_vertices_to_soma77(vertices: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    from ...motion_ops.soma_converter import SOMAConverter
    import torch
    return SOMAConverter(device=device).vertices_to_soma77(torch.as_tensor(vertices)).astype(np.float32)


def joints22_to_soma77(joints22: np.ndarray) -> np.ndarray:
    """Embed 22 joints into SOMA only when no SMPL reconstruction is available."""
    j = np.asarray(joints22, np.float32)
    if j.ndim != 3 or j.shape[1:] != (22, 3):
        raise ValueError(f"expected [T,22,3], got {j.shape}")
    out = np.repeat(j[:, :1], 77, axis=1)
    from ..smpl_hml.conversion import _SMPL24_TO_SOMA77_INDEX
    out[:, np.asarray(_SMPL24_TO_SOMA77_INDEX[:22])] = j
    return out


def resample_for_tmr(joints: np.ndarray, src_fps: float) -> np.ndarray:
    return resample_fps(np.asarray(joints, np.float32), src_fps, 30.0)


__all__ = ["smpl_vertices_to_soma77", "joints22_to_soma77", "resample_for_tmr"]
