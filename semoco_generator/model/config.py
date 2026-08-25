"""Configuration for the SeMoCo-Generator backbone + packet heads.

Qwen3TTS-aligned: independent code predictor (fixed 1024-dim), GQA support,
head_dim=128, RoPE theta=1M.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MotionGPTConfig:
    # Packet / codebook geometry (must match the frozen tokenizer export).
    num_codebooks: int = 16
    codebook_size: int = 1024
    # Per-codebook vocab sizes; defaults to uniform ``codebook_size``.
    codebook_sizes: list[int] | None = None

    # Backbone (decoder-only causal transformer).
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 0                   # 0 → full MHA (== n_heads); >0 → GQA
    ffn_hidden: int = 2048                # SwiGLU inner dim (8/3 * d_model preferred)
    rope_theta: float = 1000000.0         # Qwen3TTS: 1M for fine position encoding
    norm_eps: float = 1e-5
    dropout: float = 0.0
    max_seq_len: int = 4096               # RoPE table cap

    # Sequence.
    context_length: int = 256

    # Per-codebook CE loss weights (coarse > residual). Length must match
    # ``num_codebooks`` when provided; otherwise a sensible default is built.
    loss_weights: list[float] | None = None

    # ------------------------------------------------------------------
    # Codebook-axis predictor (independent from backbone — Qwen3TTS pattern).
    # All code predictors are identical across model sizes:
    #   1024-dim, 8 heads, 4 KV heads, 3072 FFN, 5 layers.
    # A bridge projection (d_model → code_pred_hidden) is inserted when dims differ.
    # ------------------------------------------------------------------
    code_pred_hidden: int = 1024
    code_pred_layers: int = 5
    code_pred_heads: int = 8
    code_pred_kv_heads: int = 0           # 0 → full MHA (== code_pred_heads); >0 → GQA
    code_pred_ffn_hidden: int = 3072
    code_axis_loss_scale: float = 0.3
    code_axis_start_step: int = 0         # 0 → joint from step 0; >0 → q0-only until this step
    code_axis_warmup_steps: int = 0       # 0 → instant; >0 → linear warmup after start_step
    gradient_checkpointing: bool = False  # trade compute for memory

    # ------------------------------------------------------------------
    # Text2Motion conditioning (v1). When ``use_text`` is False the model is
    # the original motion-only packet LM and none of the text modules exist.
    # ------------------------------------------------------------------
    use_text: bool = False
    clip_dim: int = 2048                   # Text embedding dimension (matches chosen encoder)
    text_cond_prob: float = 0.1            # CFG: prob of dropping text -> null
    eos_loss_scale: float = 1.0            # weight of the motion-EOS BCE head

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def resolved_codebook_sizes(self) -> list[int]:
        if self.codebook_sizes is not None:
            if len(self.codebook_sizes) != self.num_codebooks:
                raise ValueError("codebook_sizes length != num_codebooks")
            return list(self.codebook_sizes)
        return [self.codebook_size] * self.num_codebooks

    def resolved_loss_weights(self) -> list[float]:
        if self.loss_weights is not None:
            if len(self.loss_weights) != self.num_codebooks:
                raise ValueError("loss_weights length != num_codebooks")
            return list(self.loss_weights)
        # Coarse codebooks (q0/q1) weighted higher; fine residual codebooks
        # (last 4) weighted lower.  Matches the 16-codebook default tokenizer.
        defaults = [1.5, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 0.7]
        if self.num_codebooks == len(defaults):
            return defaults
        # Generic fallback: q0=1.5, q1=1.2, middle=1.0, last four=0.7.
        w = [1.0] * self.num_codebooks
        if self.num_codebooks >= 1:
            w[0] = 1.5
        if self.num_codebooks >= 2:
            w[1] = 1.2
        for k in range(max(0, self.num_codebooks - 2), self.num_codebooks):
            if k >= 2:
                w[k] = 0.7
        return w

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    @property
    def code_pred_head_dim(self) -> int:
        if self.code_pred_hidden % self.code_pred_heads != 0:
            raise ValueError("code_pred_hidden must be divisible by code_pred_heads")
        return self.code_pred_hidden // self.code_pred_heads

    @property
    def code_pred_kv_heads_resolved(self) -> int:
        return self.code_pred_kv_heads or self.code_pred_heads
