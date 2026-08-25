"""Frozen Qwen3 Embedding encoder for text conditioning.

Uses ``AutoModel`` (resolves to ``Qwen3Model`` — the base transformer without
the LM head; the checkpoint has no ``lm_head.weight``).  Produces per-token
embeddings ``[B, L, 2560]`` with **causal** (left-to-right) attention — a
semantic difference from FlanT5/SigLIP that ablation studies evaluate.

The model is a sentence-transformers model with last-token pooling in its
native pipeline; we bypass the pooling layer and use raw ``last_hidden_state``
for per-token conditioning.

    enc = Qwen3Encoder.load(device="cuda")
    emb, mask = enc.encode(["a person walks forward"])  # [1, L, 2560], [1, L]
"""

from __future__ import annotations

import torch

from .base import TextEncoder


class Qwen3Encoder(TextEncoder):
    """Frozen Qwen3 Embedding encoder — causal per-token embeddings."""

    # Qwen3 is natively supported in transformers ≥ 4.51 — no trust_remote_code needed.
    DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-4B"

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
        max_length: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
    ) -> "Qwen3Encoder":
        from transformers import AutoModel, AutoTokenizer  # noqa: WPS433 (heavy, deferred)

        mid = model_id or cls.DEFAULT_MODEL_ID
        dev = torch.device(device if (torch.cuda.is_available() or "cuda" not in str(device)) else "cpu")
        tok = AutoTokenizer.from_pretrained(mid, local_files_only=local_files_only)
        # Left-padding ensures last-token semantics for decoder-only models.
        tok.padding_side = "left"
        model = AutoModel.from_pretrained(mid, local_files_only=local_files_only)
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
        """``list[str]`` -> (``emb [B, L, clip_dim]`` float32, ``mask [B, L]`` bool).

        Returns per-token ``last_hidden_state`` (causal context — each token
        only attends to preceding tokens).
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
