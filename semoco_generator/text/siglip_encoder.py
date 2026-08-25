"""Frozen SigLIP text encoder for text conditioning.

Uses ``SiglipTextModel`` (text tower only — NOT the full vision+text
``SiglipModel``) to produce per-token embeddings ``[B, L, 1152]``.
Bidirectional attention, last-token pooling in the native model; we use raw
``last_hidden_state`` for per-token conditioning.

    enc = SigLIPEncoder.load(device="cuda")
    emb, mask = enc.encode(["a person walks forward"])  # [1, L, 1152], [1, L]
"""

from __future__ import annotations

import torch

from .base import TextEncoder


class SigLIPEncoder(TextEncoder):
    """Frozen SigLIP text encoder — word-level embeddings from the text tower."""

    DEFAULT_MODEL_ID = "google/siglip-so400m-patch14-384"

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
        device: str | torch.device = "cuda",
        max_length: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
    ) -> "SigLIPEncoder":
        from transformers import SiglipTextModel, AutoTokenizer  # noqa: WPS433 (heavy, deferred)

        mid = model_id or cls.DEFAULT_MODEL_ID
        dev = torch.device(device if (torch.cuda.is_available() or "cuda" not in str(device)) else "cpu")
        tok = AutoTokenizer.from_pretrained(mid, local_files_only=local_files_only)
        model = SiglipTextModel.from_pretrained(mid, local_files_only=local_files_only)
        model = model.to(dev)
        if dev.type == "cuda":
            model = model.to(dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        clip_dim = int(model.config.hidden_size)
        return cls(tok, model, name=mid, clip_dim=clip_dim, device=dev, max_length=max_length)

    @torch.no_grad()
    def encode(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """``list[str]`` -> (``emb [B, L, clip_dim]`` float32, ``mask [B, L]`` bool)."""
        enc = self._tok(
            list(captions),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        # SigLIP tokenizer does not return attention_mask — build it from pad_token_id
        if "attention_mask" in enc:
            attn = enc["attention_mask"].to(self.device)
        else:
            pad_id = self._tok.pad_token_id or 1
            attn = (input_ids != pad_id).to(self.device)
        out = self._model(input_ids=input_ids, attention_mask=attn)
        emb = out.last_hidden_state.float()            # [B, L, clip_dim]
        return emb, attn.bool()
