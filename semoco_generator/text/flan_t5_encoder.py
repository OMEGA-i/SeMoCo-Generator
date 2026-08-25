"""Frozen Flan-T5 text encoder for word-level text conditioning.

Mirrors the paper's / MotionMillion's setup (``T5EncoderModel`` word-level
embeddings, not a pooled sentence vector): a caption is tokenized and passed
through a frozen encoder, yielding per-word embeddings ``[L, clip_dim]`` plus an
attention mask ``[L]``. The 3B encoder is heavy, so text embeddings are computed
**offline once** by ``tools/export_t2m_dataset.py`` and cached to disk; training
and rollout consume the cache and never load the encoder. It is only
instantiated again at generation time for arbitrary user prompts.

    enc = FlanT5Encoder.load(device="cuda")           # clip_dim == enc.clip_dim
    emb, mask = enc.encode(["a person walks forward"]) # emb [B,L,clip_dim], mask [B,L]
"""

from __future__ import annotations

import numpy as np
import torch

from .base import TextEncoder

# Flan-T5 encoder output width by model size (== hidden size).
_CLIP_DIM = {
    "google/flan-t5-base": 768,
    "google/flan-t5-large": 1024,
    "google/flan-t5-xl": 2048,
    "google/flan-t5-xxl": 4096,
}


class FlanT5Encoder(TextEncoder):
    """Thin wrapper around a frozen ``T5EncoderModel`` (word-level embeddings)."""

    DEFAULT_MODEL_ID = "google/flan-t5-xl"

    def __init__(self, tokenizer, model, *, name: str, clip_dim: int, device: torch.device, max_length: int):
        self._tok = tokenizer
        self._model = model
        self.name = name
        self.model_id = name
        self.clip_dim = clip_dim
        self.device = device
        self.max_length = max_length

    @classmethod
    def load(
        cls,
        model_id: str | None = None,
        *,
        name: str | None = None,          # legacy alias for model_id
        device: str | torch.device = "cuda",
        max_length: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
    ) -> "FlanT5Encoder":
        from transformers import T5EncoderModel, T5Tokenizer  # noqa: WPS433 (heavy, deferred)

        mid = model_id or name or cls.DEFAULT_MODEL_ID
        dev = torch.device(device if (torch.cuda.is_available() or "cuda" not in str(device)) else "cpu")
        tok = T5Tokenizer.from_pretrained(mid, local_files_only=local_files_only)
        model = T5EncoderModel.from_pretrained(mid, local_files_only=local_files_only)
        model = model.to(dev)
        if dev.type == "cuda":
            model = model.to(dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        clip_dim = int(getattr(model.config, "d_model", _CLIP_DIM.get(mid, 2048)))
        return cls(tok, model, name=mid, clip_dim=clip_dim, device=dev, max_length=max_length)

    @torch.no_grad()
    def encode(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """``list[str]`` -> (``emb [B, L, clip_dim]`` float32, ``mask [B, L]`` bool).

        ``L`` is the batch-max token length (padded); ``mask`` marks real tokens.
        """
        enc = self._tok(
            list(captions),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        out = self._model(input_ids=input_ids, attention_mask=attn)
        emb = out.last_hidden_state.float()            # [B, L, clip_dim]
        return emb, attn.bool()

    @torch.no_grad()
    def encode_one(self, caption: str) -> tuple[np.ndarray, np.ndarray]:
        """Single caption -> (``emb [L, clip_dim]`` float32, ``mask [L]`` bool) numpy."""
        emb, mask = self.encode([caption])
        return emb[0].cpu().numpy(), mask[0].cpu().numpy()
