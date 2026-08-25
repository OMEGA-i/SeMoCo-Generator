"""Packet embedding: per-codebook tables summed into one token embedding.

A motion packet ``m_t = [q0_t, ..., q_{Q-1}_t]`` is embedded as

    e_t = sum_i E_i(q_t^i)

(plus positional info added by the backbone via RoPE). The time axis stays at
the low (12.5Hz) packet rate; codebooks are NOT unrolled into the sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import MotionGPTConfig


class PacketEmbedding(nn.Module):
    def __init__(self, cfg: MotionGPTConfig) -> None:
        super().__init__()
        self.num_codebooks = cfg.num_codebooks
        sizes = cfg.resolved_codebook_sizes()
        self.tables = nn.ModuleList(
            [nn.Embedding(size, cfg.d_model) for size in sizes]
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for table in self.tables:
            nn.init.normal_(table.weight, mean=0.0, std=0.02)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes [B, T, Q]`` (long) -> ``[B, T, d_model]``."""
        if codes.shape[-1] != self.num_codebooks:
            raise ValueError(f"expected Q={self.num_codebooks}; got {codes.shape[-1]}")
        emb = self.tables[0](codes[..., 0])
        for i in range(1, self.num_codebooks):
            emb = emb + self.tables[i](codes[..., i])
        return self.dropout(emb)
