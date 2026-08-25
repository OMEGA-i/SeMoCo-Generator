"""Rotation 6D helpers shared by baseline adapters."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """Convert Zhou et al. 6D rotations ``(*, 6)`` to matrices ``(*, 3, 3)``.

    The 6-D vector is *interleaved* as the rows of a 3x2 matrix
    ``(a1_0, a2_0, a1_1, a2_1, a1_2, a2_2)``; the two basis vectors are its
    columns. Splitting it as ``[:3]`` / ``[3:]`` instead mixes elements across
    columns and yields invalid rotations.
    """
    x = d6.view(*d6.shape[:-1], 3, 2)
    a1 = x[..., 0]  # first column of the 3×2
    a2 = x[..., 1]  # second column
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_axis_angle(rm: Tensor) -> Tensor:
    """Convert rotation matrices ``(*, 3, 3)`` to axis-angle ``(*, 3)``."""
    t = rm[..., 0, 0] + rm[..., 1, 1] + rm[..., 2, 2]
    cos = ((t - 1.0) * 0.5).clamp(-1.0 + 1e-8, 1.0 - 1e-8)
    th = torch.acos(cos)
    rx = rm[..., 2, 1] - rm[..., 1, 2]
    ry = rm[..., 0, 2] - rm[..., 2, 0]
    rz = rm[..., 1, 0] - rm[..., 0, 1]
    sn = torch.sqrt(rx * rx + ry * ry + rz * rz).clamp_min(1e-8)
    ax = torch.stack([rx, ry, rz], dim=-1) / sn.unsqueeze(-1)
    return ax * th.unsqueeze(-1)


def rotation_6d_to_axis_angle(d6: Tensor) -> Tensor:
    """Convert Zhou et al. 6D rotations ``(*, 6)`` to axis-angle ``(*, 3)``."""
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))
