"""Conversions into the official 22-joint / 263D HumanML representation."""
from __future__ import annotations

import numpy as np

from ....paths import humanml3d_root
from .vendor.motion_process import offsets_from_example, process_file

# SMPL-24 body joints → SOMA77 FK indices (pelvis..wrists). Kept local so this
# track does not hard-require the tokenizer repo at import time.
_SMPL24_TO_SOMA77_INDEX: tuple[int, ...] = (
    0, 67, 72, 1, 68, 73, 2, 69, 74, 3, 70, 75, 4, 11, 39, 6, 12, 40, 13, 41, 14, 42, 14, 42,
)

_OFFSETS_CACHE: object | None = None
_DEFAULT_EXAMPLE = humanml3d_root() / "new_joints" / "000021.npy"


def _tgt_offsets():
    global _OFFSETS_CACHE
    if _OFFSETS_CACHE is None:
        example = np.load(_DEFAULT_EXAMPLE).astype(np.float32)
        _OFFSETS_CACHE = offsets_from_example(example)
    return _OFFSETS_CACHE


def soma77_to_joints22(joints77: np.ndarray) -> np.ndarray:
    """Select the SMPL-24 first 22 joints from a SOMA77 clip."""
    j = np.asarray(joints77, dtype=np.float32)
    if j.ndim != 3 or j.shape[1:] != (77, 3):
        raise ValueError(f"expected [T,77,3], got {j.shape}")
    return j[:, np.asarray(_SMPL24_TO_SOMA77_INDEX[:22]), :].copy()


def joints22_to_hml263(joints22: np.ndarray, fps: float = 20.0) -> np.ndarray:
    """Encode joint positions with the official HumanML3D ``process_file``.

    ``fps`` is accepted for API compatibility; HumanML features are defined on
    the 20 Hz joint stream (resample before calling when needed).
    """
    del fps  # official process_file operates on the 20 Hz joint sequence
    j = np.asarray(joints22, dtype=np.float32)
    if j.ndim != 3 or j.shape[1] < 22:
        raise ValueError(f"expected [T,>=22,3], got {j.shape}")
    j = j[:, :22]
    if j.shape[0] < 2:
        # Pad a static frame so process_file can run; caller should avoid this.
        j = np.concatenate([j, j], axis=0)
    return process_file(j, already_aligned=False, tgt_offsets=_tgt_offsets()).astype(np.float32)


def soma77_to_hml263(joints77: np.ndarray, fps: float = 20.0) -> np.ndarray:
    """SOMA77 -> joints22 mapped subset -> HumanML 263D."""
    return joints22_to_hml263(soma77_to_joints22(joints77), fps=fps)


__all__ = ["soma77_to_joints22", "joints22_to_hml263", "soma77_to_hml263"]
