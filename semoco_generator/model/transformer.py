"""Decoder-only causal transformer blocks (RoPE + RMSNorm + SwiGLU + KV cache).

A clean, dependency-free backbone in the Qwen3-style: pre-norm RMSNorm,
rotary position embeddings, grouped-query-capable attention, and a SwiGLU MLP.
Supports incremental decoding via a per-layer ``KVCache`` for AR rollout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MotionGPTConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # [seq_len, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)          # [seq_len, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, H, T, D]. cos/sin are either [T, D] (shared contiguous positions)
    # or [B, T, D] (per-sequence positions, e.g. per-document RoPE reset).
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, T, D]
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        cos = cos.unsqueeze(1)                # [B, 1, T, D]
        sin = sin.unsqueeze(1)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


@dataclass
class KVCache:
    k: torch.Tensor  # [B, H_kv, T, D]
    v: torch.Tensor  # [B, H_kv, T, D]

    @property
    def length(self) -> int:
        return self.k.shape[2]


class Attention(nn.Module):
    def __init__(self, cfg: MotionGPTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)

        # QK-Norm (Qwen3-TTS style): RMSNorm on Q and K per head after linear
        # projection, before RoPE.  Stabilises training at scale.
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cache: KVCache | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        B, T, _ = x.shape
        # Linear projection -> reshape -> QK-Norm per head (Qwen3-TTS style) -> transpose
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)                                    # [B, H, T, D]
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)                                    # [B, Hkv, T, D]
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        new_cache: KVCache | None = None
        past_len = 0
        if cache is not None:
            past_len = cache.k.shape[2]
            k = torch.cat([cache.k, k], dim=2)
            v = torch.cat([cache.v, v], dim=2)
            new_cache = KVCache(k=k, v=v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        dropout_p = self.dropout if self.training else 0.0
        if doc_mask is not None:
            # Packed training: pre-built block-diagonal causal mask [B,1,T,T]
            # (already encodes causality, so is_causal=False).
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=doc_mask, is_causal=False, dropout_p=dropout_p
            )
        else:
            # Masking cases:
            #  * T == 1 (decode): query attends to all cached keys (all in the past).
            #  * past_len == 0, T > 1 (prefill): standard causal over the block.
            #  * past_len > 0, T > 1: explicit mask (causal within block + full past).
            attn_mask = None
            is_causal = False
            if T == 1:
                is_causal = False
            elif past_len == 0:
                is_causal = True
            else:
                total = past_len + T
                allow = torch.ones(T, total, dtype=torch.bool, device=q.device)
                block_causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
                allow[:, past_len:] = block_causal
                attn_mask = allow.view(1, 1, T, total)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p
            )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: MotionGPTConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)  # up
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)  # down
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: MotionGPTConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cache: KVCache | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        h, new_cache = self.attn(self.attn_norm(x), cos, sin, cache=cache, doc_mask=doc_mask)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache
