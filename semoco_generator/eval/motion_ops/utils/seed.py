"""Shared RNG helpers for baseline adapters."""

from __future__ import annotations


def seed_all(seed: int) -> None:
    """Seed torch (and CUDA if available) for a deterministic generation step."""
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
