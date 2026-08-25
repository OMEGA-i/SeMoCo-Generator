"""ADE / FDE over the future window of a motion prediction.

Both metrics take a pair of SMPL-22 joint sequences ``[T, 22, 3]`` in metres.
The scored window is the future segment only: with ``ratio=0.2`` the first 20%
of frames are the observation the model was conditioned on, and the metrics run
over the remaining 80%. Joints are flattened to 66-d per frame, so the per-frame
error is a single Euclidean distance; ADE averages it over the window and FDE
reads it at the last frame.
"""

from __future__ import annotations

import numpy as np

DEFAULT_PREDICT_RATIO = 0.2


def future_window(n_frames: int, ratio: float = DEFAULT_PREDICT_RATIO) -> int:
    """First scored frame index, or ``-1`` when the clip is too short."""
    if n_frames < 2:
        return -1
    start = max(1, int(n_frames * ratio))
    return -1 if start >= n_frames else start


def ade(pred: np.ndarray, gt: np.ndarray, ratio: float = DEFAULT_PREDICT_RATIO) -> float:
    """Average displacement error over the future window, in metres."""
    n = min(pred.shape[0], gt.shape[0])
    start = future_window(n, ratio)
    if start < 0:
        return float("inf")
    p = pred[start:n].reshape(n - start, -1)
    g = gt[start:n].reshape(n - start, -1)
    return float(np.linalg.norm(p - g, axis=-1).mean())


def fde(pred: np.ndarray, gt: np.ndarray) -> float:
    """Final displacement error, in metres.

    Takes no ratio: the last frame is the last frame regardless of where the
    scored window starts.
    """
    n = min(pred.shape[0], gt.shape[0])
    if n < 2:
        return float("inf")
    return float(np.linalg.norm(pred[n - 1].reshape(-1) - gt[n - 1].reshape(-1)))


__all__ = ["DEFAULT_PREDICT_RATIO", "ade", "fde", "future_window"]
