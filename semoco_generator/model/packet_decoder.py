"""Packet decoder - q0 time-axis AR + q1..qN codebook-axis AR.

This follows the Qwen3-TTS code-predictor pattern exactly: a small causal
transformer with its own RoPE, QK-Norm, RMSNorm, and SwiGLU — matching the
backbone's convention throughout.  Each residual codebook gets its own embedding
table and LM head; the transformer is shared across all codebook steps.

The code predictor is *independent* from the backbone: it always uses
``code_pred_hidden`` (default 1024), ``code_pred_heads`` (default 8), etc.
A bridge projection (``code_proj``) connects the backbone hidden states to the
code predictor dimension when they differ.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MotionGPTConfig
from .transformer import RMSNorm, _rotate_half


def _build_code_rope(
    seq_len: int, head_dim: int, theta: float, device, dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE cache for the code-axis positions (max Q+1 = 17 tokens)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_code_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place RoPE for ``[N, H, L, D]`` code-axis tensors."""
    cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, L, D]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class CodeAttention(nn.Module):
    """Multi-head self-attention for the code-axis predictor with GQA support.

    Uses explicit Q/K/V projections with QK-Norm per head before RoPE —
    matching Qwen3-TTS exactly.
    """

    def __init__(
        self, d_model: int, n_heads: int, n_kv_heads: int,
        head_dim: int, dropout: float, eps: float,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout

        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # QK-Norm (Qwen3-TTS style)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        N, L, _ = x.shape
        # Project, reshape, QK-Norm (per head, before RoPE), then transpose
        q = self.wq(x).view(N, L, self.n_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)                       # [N, Hq, L, D]
        k = self.wk(x).view(N, L, self.n_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)                       # [N, Hkv, L, D]
        v = self.wv(x).view(N, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_code_rope(q, k, cos, sin)

        # GQA: repeat KV heads to match query heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, dropout_p=dropout_p,
        )
        out = out.transpose(1, 2).contiguous().view(N, L, -1)
        return self.wo(out)


class CodeBlock(nn.Module):
    """Transformer block for the code-axis predictor.

    RMSNorm + CodeAttention (with QK-Norm + RoPE + GQA) + SwiGLU FFN.
    Identical convention to the backbone ``Block``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        ffn_hidden: int,
        dropout: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.attn = CodeAttention(d_model, n_heads, n_kv_heads, head_dim, dropout, eps)
        self.ffn_norm = RMSNorm(d_model, eps=eps)
        self.w1 = nn.Linear(d_model, ffn_hidden, bias=False)   # gate
        self.w3 = nn.Linear(d_model, ffn_hidden, bias=False)   # up
        self.w2 = nn.Linear(ffn_hidden, d_model, bias=False)   # down
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.attn_norm(x)
        h = self.attn(h, cos, sin, mask)
        x = x + h
        h = self.ffn_norm(x)
        h = self.dropout(self.w2(F.silu(self.w1(h)) * self.w3(h)))
        return x + h


class PacketDecoder(nn.Module):
    """Codebook-axis AR predictor with independent dimension from backbone.

    q0 is predicted directly from backbone hidden states through ``q0_head``.
    q1..qN go through a bridge projection + shared causal transformer + per-codebook
    linear heads.  All internal dimensions are governed by ``code_pred_*`` config
    fields, matching Qwen3-TTS's fixed-size code predictor pattern.
    """

    def __init__(self, cfg: MotionGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_codebooks = cfg.num_codebooks
        self.sizes = cfg.resolved_codebook_sizes()
        if len(set(self.sizes)) != 1:
            raise ValueError("codebook-axis predictor currently expects uniform vocab sizes")

        # ---- code predictor dimensions (independent from backbone) ----
        cp = cfg.code_pred_hidden
        cp_heads = cfg.code_pred_heads
        cp_kv = cfg.code_pred_kv_heads_resolved
        cp_hd = cfg.code_pred_head_dim
        cp_ffn = cfg.code_pred_ffn_hidden

        # Bridge: backbone hidden → code predictor (identity when dims match)
        self.code_proj: nn.Module
        if cfg.d_model != cp:
            self.code_proj = nn.Linear(cfg.d_model, cp, bias=False)
        else:
            self.code_proj = nn.Identity()

        # q0 head: backbone dim → vocab (no code predictor involvement)
        self.q0_head = nn.Linear(cfg.d_model, self.sizes[0], bias=False)

        # Previous-codebook embeddings (q0..q_{K-1} → code predictor)
        self.prev_code_embeddings = nn.ModuleList(
            [nn.Embedding(self.sizes[i], cp) for i in range(max(0, self.num_codebooks - 1))]
        )

        # Code predictor transformer blocks
        self.code_blocks = nn.ModuleList(
            [
                CodeBlock(cp, cp_heads, cp_kv, cp_hd, cp_ffn, cfg.dropout, cfg.norm_eps)
                for _ in range(cfg.code_pred_layers)
            ]
        )
        self.code_norm = RMSNorm(cp, eps=cfg.norm_eps)

        # RoPE cache for the code-axis (max Q+1 positions)
        code_max_len = cfg.num_codebooks + 1
        cos, sin = _build_code_rope(
            code_max_len, cp_hd, cfg.rope_theta,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        self.register_buffer("code_rope_cos", cos, persistent=False)
        self.register_buffer("code_rope_sin", sin, persistent=False)

        # Per-codebook residual output heads
        self.residual_heads = nn.ModuleList(
            [nn.Linear(cp, self.sizes[i], bias=False) for i in range(1, self.num_codebooks)]
        )

        self.apply(self._init_code_weights)

    @staticmethod
    def _init_code_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Boolean causal mask for SDPA where ``True`` means *allowed*."""
        return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))

    def _code_rope_slice(self, length: int, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.code_rope_cos[:length].to(dtype)
        sin = self.code_rope_sin[:length].to(dtype)
        return cos, sin

    def _code_axis(self, seq: torch.Tensor) -> torch.Tensor:
        """Run the causal codebook-axis transformer over ``seq [N, L, D]``."""
        L = seq.shape[1]
        cos, sin = self._code_rope_slice(L, seq.dtype)
        mask = self._causal_mask(L, seq.device)
        for block in self.code_blocks:
            seq = block(seq, cos, sin, mask)
        return self.code_norm(seq)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, hidden: torch.Tensor, target_codes: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Teacher-forced packet logits.

        ``hidden`` is ``[B, T, d_model]`` (backbone output).
        ``target_codes`` is the next-packet target ``[B, T, Q]`` and is required
        for q1..qN teacher forcing.
        """
        q0_logits = self.q0_head(hidden)                     # [B, T, V] from backbone
        if self.num_codebooks == 1 or target_codes is None:
            return [q0_logits]

        # Project backbone hidden → code predictor dimension
        cp_input = self.code_proj(hidden)                    # [B, T, cp_hidden]
        B, T, D = cp_input.shape

        prev = [
            self.prev_code_embeddings[i](target_codes[..., i]).reshape(B * T, 1, D)
            for i in range(self.num_codebooks - 1)
        ]
        seq = torch.cat([cp_input.reshape(B * T, 1, D), *prev], dim=1)  # [B*T, Q, D]
        code_hidden = self._code_axis(seq)                               # [B*T, Q, D]

        residual_logits = [
            head(code_hidden[:, i]).reshape(B, T, -1)
            for i, head in enumerate(self.residual_heads, start=1)
        ]
        return [q0_logits, *residual_logits]

    def next_code_logits(self, hidden: torch.Tensor, prefix_codes: torch.Tensor) -> torch.Tensor:
        """Predict the next residual codebook during rollout.

        ``hidden`` is ``[B, d_model]`` for one packet.
        ``prefix_codes`` contains sampled codes ``[B, K]`` for q0..q(K-1).
        Returns logits for qK ``[B, V]``.
        """
        if prefix_codes.dim() != 2:
            raise ValueError("prefix_codes must be [B, K]")
        B, K = prefix_codes.shape
        if K <= 0 or K >= self.num_codebooks:
            raise ValueError(f"expected 1 <= K < {self.num_codebooks}; got {K}")

        cp_h = self.code_proj(hidden)                        # [B, cp_hidden]
        parts = [cp_h.unsqueeze(1)]                          # [B, 1, cp_hidden]
        for i in range(K):
            parts.append(self.prev_code_embeddings[i](prefix_codes[:, i]).unsqueeze(1))
        seq = torch.cat(parts, dim=1)                        # [B, K+1, cp_hidden]
        code_hidden = self._code_axis(seq)
        return self.residual_heads[K - 1](code_hidden[:, K])

    @torch.no_grad()
    def greedy_codes(self, hidden: torch.Tensor) -> torch.Tensor:
        """Free-run the codebook axis with argmax prefixes (no GT teacher force).

        ``hidden [B, T, d_model]`` -> predicted packet codes ``[B, T, Q]``.
        """
        B, T, _ = hidden.shape
        q0 = self.q0_head(hidden).argmax(dim=-1)             # [B, T]
        if self.num_codebooks == 1:
            return q0.unsqueeze(-1)

        cp_h = self.code_proj(hidden.reshape(B * T, -1))      # [B*T, cp_hidden]
        prefix = q0.reshape(B * T, 1)
        out = [q0]
        for _k in range(1, self.num_codebooks):
            # Build sequence: [projected_hidden, emb(q0), ..., emb(q_{k-1})]
            parts = [cp_h.unsqueeze(1)]
            for i in range(prefix.shape[1]):
                parts.append(self.prev_code_embeddings[i](prefix[:, i]).unsqueeze(1))
            seq = torch.cat(parts, dim=1)                    # [B*T, k+1, cp_hidden]
            ch = self._code_axis(seq)
            logits = self.residual_heads[_k - 1](ch[:, _k])   # [B*T, V]
            pred = logits.argmax(dim=-1)
            out.append(pred.reshape(B, T))
            prefix = torch.cat([prefix, pred.unsqueeze(1)], dim=1)
        return torch.stack(out, dim=-1)
