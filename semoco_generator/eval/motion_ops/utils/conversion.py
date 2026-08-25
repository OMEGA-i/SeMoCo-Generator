"""Shared motion → SMPL mesh conversion helpers for baseline adapters.

A baseline that emits HumanML3D-style 22-joint positions -- directly, or
after recovering them from a feature vector -- needs the same final step: fit
SMPL body parameters to those 22 joints and read back mesh vertices for the
shared :class:`SOMAConverter`.

:class:`Joints22ToSMPLVertices` owns the SMPL model cache and the fitting
loop so each runtime does not duplicate it.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from ..smpl_utils import create_smpl

JointsLike = Union[np.ndarray, torch.Tensor]


class Joints22ToSMPLVertices:
    """Fit SMPL mesh vertices to HumanML3D-style 22-joint positions.

    A gradient-based loop: optimize ``global_orient`` / ``body_pose`` /
    ``transl`` so the SMPL body joints ``[:22]`` match the target positions,
    then read back
    ``vertices`` ``[T, 6890, 3]``.

    A single instance caches one SMPL model per ``(batch_size, device)``
    pair, so it can be shared across prompts/seeds.
    """

    def __init__(self, device: str = "cuda", *, fit_steps: int = 50, lr: float = 0.05):
        self._device = device
        self._fit_steps = int(fit_steps)
        self._lr = float(lr)
        self._smpl_cache: dict[tuple[int, str], object] = {}

    def _smpl(self, batch_size: int, device: torch.device):
        key = (int(batch_size), str(device))
        if key not in self._smpl_cache:
            self._smpl_cache[key] = create_smpl(int(batch_size), str(device))
        return self._smpl_cache[key]

    def joints22_to_vertices(
        self,
        joints: JointsLike,
        *,
        length: int | None = None,
    ) -> torch.Tensor:
        """Convert 22-joint positions ``[T, 22, 3]`` to SMPL vertices ``[T, 6890, 3]``.

        Args:
            joints: ``[T, 22, 3]`` positions, or ``[1, T, 22, 3]`` (batch
                dim is squeezed).  NumPy or torch Tensor.
            length: Optional crop to the first ``length`` frames.

        Returns:
            ``torch.Tensor`` of shape ``[T, 6890, 3]`` on the runtime device.
        """
        if isinstance(joints, np.ndarray):
            joints = torch.from_numpy(joints)
        if joints.dim() == 4:
            joints = joints[0]
        if length is not None:
            joints = joints[: int(length)]

        T = int(joints.shape[0])
        dev = joints.device if joints.is_cuda else torch.device(self._device)
        tgt = joints.to(device=dev, dtype=torch.float32)

        smpl = self._smpl(T, dev)
        go = torch.zeros(T, 1, 3, device=dev, requires_grad=True)
        bp = torch.zeros(T, 23, 3, device=dev, requires_grad=True)
        tl = tgt[:, 0, :].clone().detach().requires_grad_(True)

        opt = torch.optim.Adam([go, bp, tl], lr=self._lr)
        for _ in range(self._fit_steps):
            opt.zero_grad()
            out = smpl(global_orient=go, body_pose=bp, transl=tl)
            torch.nn.functional.mse_loss(out.joints[:, :22, :], tgt).backward()
            opt.step()

        with torch.no_grad():
            return smpl(global_orient=go, body_pose=bp, transl=tl).vertices
