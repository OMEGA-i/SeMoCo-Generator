"""``MotionBatch``: device-resident internal representation for conversion edges.

Per the production plan's "Conversion And Data Movement Plan": conversion
internals should carry GPU tensors across GPU-heavy chains (e.g.
``joints22 -> smpl_vertices -> soma77``) instead of converting to CPU numpy
after every edge and back to GPU for the next one. ``MotionBatch`` is that
internal batch representation; :class:`~semoco_generator.eval.schema.MotionClip`
remains the adapter at cache/report boundaries (never the long-term internal
format for multi-edge GPU chains).

Rule from the plan, restated:

* Conversion internals use ``MotionBatch``.
* Cache and report interfaces use serialized artifacts (``MotionClip`` / numpy).
* Legacy ``MotionClip`` stacking helpers remain for edges that only need CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from .schema import MotionClip, MotionRep


@dataclass
class MotionBatch:
    """Stacked-along-time batch of clips sharing one representation.

    ``data`` is ``[total_frames, ...]`` (all clips concatenated along the
    leading axis); ``lengths`` records each original clip's frame count so
    :meth:`split` can recover per-clip :class:`MotionClip` objects.
    """

    rep: MotionRep
    data: torch.Tensor
    lengths: list[int]
    fps: float | list[float]
    aux: dict[str, torch.Tensor] = field(default_factory=dict)
    device: str = "cpu"

    # ------------------------------------------------------------------
    @classmethod
    def from_clips(cls, clips: Sequence[MotionClip], *, device: str | torch.device = "cpu") -> "MotionBatch":
        if not clips:
            raise ValueError("MotionBatch.from_clips requires at least one clip")
        rep = clips[0].rep
        lengths = [int(np.asarray(c.array).shape[0]) for c in clips]
        fps = float(clips[0].fps)
        dev = torch.device(device)

        arrays = [np.asarray(c.array, dtype=np.float32) for c in clips]
        data = torch.as_tensor(np.concatenate(arrays, axis=0)).to(dev)

        aux_keys = set()
        for c in clips:
            aux_keys.update(c.aux.keys())
        aux: dict[str, torch.Tensor] = {}
        for k in aux_keys:
            parts = [np.asarray(c.aux[k], dtype=np.float32) for c in clips]
            aux[k] = torch.as_tensor(np.concatenate(parts, axis=0)).to(dev)

        return cls(rep=rep, data=data, lengths=lengths, fps=fps, aux=aux, device=str(dev))

    # ------------------------------------------------------------------
    def to_device(self, device: str | torch.device, *, non_blocking: bool = True) -> "MotionBatch":
        dev = torch.device(device)
        if torch.device(self.device) == dev:
            return self
        data = self.data.to(dev, non_blocking=non_blocking)
        aux = {k: v.to(dev, non_blocking=non_blocking) for k, v in self.aux.items()}
        return MotionBatch(rep=self.rep, data=data, lengths=list(self.lengths), fps=self.fps, aux=aux, device=str(dev))

    def split(self) -> list[MotionClip]:
        """Split back into per-clip :class:`MotionClip` objects. Only call this
        at storage/metric edges — never chain another GPU edge off of it."""
        bounds = [0]
        for length in self.lengths:
            bounds.append(bounds[-1] + length)
        data_np = self.data.detach().cpu().numpy()
        aux_np = {k: v.detach().cpu().numpy() for k, v in self.aux.items()}
        fps_list = self.fps if isinstance(self.fps, (list, tuple)) else [self.fps] * len(self.lengths)

        out: list[MotionClip] = []
        for i in range(len(self.lengths)):
            s, e = bounds[i], bounds[i + 1]
            clip_aux = {k: v[s:e] for k, v in aux_np.items()}
            out.append(MotionClip(rep=self.rep, array=data_np[s:e], fps=float(fps_list[i]), aux=clip_aux))
        return out

    def slice_by_frame_budget(self, max_frames: int) -> list["MotionBatch"]:
        """Split into VRAM-safe chunks along the stacked frame axis.

        Chunk boundaries do **not** need to align with original clip
        boundaries — per-frame model ops (SMPL FK, SOMA pose inversion) treat
        every frame independently. Use the *original* (un-chunked) batch's
        ``lengths`` to re-split a concatenated model output back into clips,
        not the chunk's own ``lengths`` (which is just its own frame count).
        """
        total = int(self.data.shape[0])
        max_frames = max(1, int(max_frames))
        if total <= max_frames:
            return [self]
        chunks: list[MotionBatch] = []
        for start in range(0, total, max_frames):
            end = min(start + max_frames, total)
            chunks.append(MotionBatch(
                rep=self.rep,
                data=self.data[start:end],
                lengths=[end - start],
                fps=self.fps,
                aux={k: v[start:end] for k, v in self.aux.items()},
                device=self.device,
            ))
        return chunks

    def estimate_bytes(self) -> int:
        total = int(self.data.numel()) * int(self.data.element_size())
        for v in self.aux.values():
            total += int(v.numel()) * int(v.element_size())
        return total


def concat_batches(batches: Sequence[MotionBatch]) -> MotionBatch:
    """Concatenate same-rep batches along the frame axis (e.g. re-joining
    ``slice_by_frame_budget`` chunks' model outputs before a final ``split()``)."""
    if not batches:
        raise ValueError("concat_batches requires at least one batch")
    if len(batches) == 1:
        return batches[0]
    rep = batches[0].rep
    device = batches[0].device
    data = torch.cat([b.data for b in batches], dim=0)
    lengths: list[int] = []
    for b in batches:
        lengths.extend(b.lengths)
    aux_keys = set()
    for b in batches:
        aux_keys.update(b.aux.keys())
    aux = {k: torch.cat([b.aux[k] for b in batches], dim=0) for k in aux_keys if all(k in b.aux for b in batches)}
    fps = batches[0].fps
    return MotionBatch(rep=rep, data=data, lengths=lengths, fps=fps, aux=aux, device=device)


__all__ = ["MotionBatch", "concat_batches"]
