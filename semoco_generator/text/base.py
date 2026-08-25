"""Abstract base class for text embedding models.

All text encoders used for SeMoCo-Generator conditioning must implement
``encode()`` (returns per-token ``last_hidden_state`` embeddings) and
override the ``load()`` classmethod.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class TextEncoder(ABC):
    """Base for all frozen text embedding models.

    Subclasses must implement ``encode()`` and override ``load()``.
    ``load()`` is deliberately **not** abstract — each encoder has its own
    tokenizer class, model class, and default hyperparameters, so a single
    abstract signature would be too restrictive.
    """

    name: str           # short key, e.g. "flan"
    model_id: str       # HF repo id or local path passed to from_pretrained
    clip_dim: int       # embedding dimension
    device: torch.device
    max_length: int

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        device: str | torch.device = "cuda",
        max_length: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
    ) -> "TextEncoder":
        """Load the model. Subclasses MUST override this."""
        raise NotImplementedError(f"{cls.__name__}.load() not implemented")

    @abstractmethod
    @torch.no_grad()
    def encode(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch-encode captions.

        Returns:
            emb  ``[B, L, clip_dim]`` float32  — per-token embeddings
            mask ``[B, L]`` bool               — ``True`` for real tokens
        """
        ...

    @torch.no_grad()
    def encode_one(self, caption: str) -> tuple[np.ndarray, np.ndarray]:
        """Single caption → numpy arrays (convenience for export)."""
        emb, mask = self.encode([caption])
        return emb[0].cpu().numpy(), mask[0].cpu().numpy()
