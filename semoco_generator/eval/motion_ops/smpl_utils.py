"""Shared SMPL body-model helpers for baseline → SOMA conversion."""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=8)
def create_smpl(batch_size: int, device: str):
    """Build an SMPL model (6890 verts) matching :class:`SOMAConverter` identity."""
    import smplx

    from ...paths import smpl_model_path

    model = smplx.create(
        model_path=str(smpl_model_path()),
        model_type="smpl",
        use_pca=False,
        flat_hand_mean=True,
        batch_size=int(batch_size),
    )
    return model.to(torch.device(device))
