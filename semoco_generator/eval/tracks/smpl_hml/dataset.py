"""Official HumanML3D caption and 263-D feature loader."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from ...datasets.release_subset import EvalClip, build_subset_from_humanml3d
from .paths import resolve_humanml_asset
from .protocol import HML_SUBSET_PROTOCOL


class HumanML3DDataset:
    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        *,
        limit: int | None = None,
        seed: int = 0,
        protocol: Literal["official_hml_eval", "legacy_full_test"] = HML_SUBSET_PROTOCOL,  # type: ignore[assignment]
    ) -> None:
        self.root = Path(root)
        self.protocol = protocol
        self.clips = build_subset_from_humanml3d(
            self.root, split=split, limit=limit, seed=seed, protocol=protocol
        )

    def __len__(self) -> int:
        return len(self.clips)

    def clip(self, index: int) -> EvalClip:
        return self.clips[index]

    def motion(self, index: int) -> np.ndarray:
        clip = self.clips[index]
        mid = clip.rec_id
        path = resolve_humanml_asset(self.root, "new_joint_vecs", mid)
        if path is None:
            raise FileNotFoundError(
                f"missing HumanML new_joint_vecs for {mid} under {self.root / 'new_joint_vecs'}"
            )
        data = np.load(path).astype(np.float32)
        if data.ndim != 2 or data.shape[1] != 263:
            raise ValueError(f"expected HumanML [T,263] at {path}, got {data.shape}")
        meta = clip.metadata or {}
        frame_start = int(meta.get("frame_start") or 0)
        frame_end = meta.get("frame_end")
        if frame_end is not None:
            data = data[frame_start: int(frame_end)]
        elif frame_start:
            data = data[frame_start:]
        m_length = meta.get("m_length")
        if m_length is not None:
            data = data[: int(m_length)]
        return data


__all__ = ["HumanML3DDataset"]
