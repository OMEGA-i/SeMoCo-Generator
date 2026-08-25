"""Decode generated motion codes back to UMR-499 features / SOMA-77 joints.

This module is pure orchestration: it resolves the frame-0 anchor from a
derived-parquet clip row and delegates all tokenizer access to
:mod:`semoco_generator.tokenizer_bridge`.

    codes [T_tok, Q]
        -> FrozenMotionTokenizer.decode                  -> features [T, 499]
        -> FrozenMotionTokenizer.decode_to_joints_arrays -> joints77 [T, 77, 3]

Codes never carry the frame-0 seed, so the anchor needed to decode them back to
absolute motion comes from the clip's parquet row (``init_root_pos`` /
``init_root_rot6d`` / ``init_joints76_rot6d`` / ``identity_coeffs``; see
``dataset/umr_parquet.py``).
"""

from __future__ import annotations

import numpy as np

from ..tokenizer_bridge import FrozenMotionTokenizer


def codes_to_features(tokenizer: FrozenMotionTokenizer, codes: np.ndarray) -> np.ndarray:
    """``codes [T_tok, Q]`` -> reconstructed ``features [T, 499]`` (physical)."""
    return tokenizer.decode(codes)


def codes_to_joints(
    tokenizer: FrozenMotionTokenizer,
    codes: np.ndarray,
    clip: dict,
    *,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Full decode path -> ``{joints77 [T,77,3], root [T,3], features [T,499]}``.

    ``clip`` is a derived-parquet row carrying the frame-0 anchor; see
    ``dataset/umr_parquet.py::CLIP_COLS``.
    """
    return tokenizer.decode_to_joints_arrays(
        codes,
        init_root_pos=clip["init_root_pos"],
        init_root_rot6d=clip["init_root_rot6d"],
        init_joints76_rot6d=clip["init_joints76_rot6d"],
        identity_coeffs=clip["identity_coeffs"],
        device=device,
    )


__all__ = ["codes_to_features", "codes_to_joints"]
