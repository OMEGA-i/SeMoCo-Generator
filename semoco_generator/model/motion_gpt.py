"""SeMoCo-Generator - packet-level multi-codebook causal LM.

Time-axis autoregression over 12.5Hz motion packets. The backbone predicts q0;
a small codebook-axis causal predictor generates q1..qN within each packet.
Forward consumes ``codes [B, T, Q]`` and predicts the next packet at every
position. Supports KV-cached incremental decoding for rollout.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MotionGPTConfig
from .packet_embedding import PacketEmbedding
from .packet_decoder import PacketDecoder
from .transformer import Block, KVCache, RMSNorm, build_rope_cache


class MotionGPT(nn.Module):
    def __init__(self, cfg: MotionGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = PacketEmbedding(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.decoder = PacketDecoder(cfg)

        # Text2motion conditioning modules (only when enabled).
        if cfg.use_text:
            # Project frozen per-token text embeddings into the model width.
            # Encoders differ wildly in raw scale (Flan ~O(1), Qwen3 ~O(100));
            # callers must go through ``project_text`` which L2-normalizes first.
            self.text_proj = nn.Linear(cfg.clip_dim, cfg.d_model, bias=True)
            # Learnable null-text token for classifier-free guidance.
            self.null_text = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            # Learnable motion BOS packet embedding (prefill seed after text).
            self.motion_bos = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            # Binary stop head: predicts EOS at the position after the last frame.
            self.eos_head = nn.Linear(cfg.d_model, 1, bias=True)
            nn.init.normal_(self.null_text, mean=0.0, std=0.02)
            nn.init.normal_(self.motion_bos, mean=0.0, std=0.02)

        cos, sin = build_rope_cache(
            cfg.max_seq_len, cfg.head_dim, cfg.rope_theta,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale residual projections (Qwen3TTS-style) for stable deep training.
        # Backbone and code predictor use DIFFERENT depth-based scaling:
        #   backbone blocks  → 1/sqrt(n_layers)
        #   code_pred blocks → 1/sqrt(code_pred_layers)
        backbone_scale = 1.0 / math.sqrt(cfg.n_layers)
        code_scale = 1.0 / math.sqrt(max(1, cfg.code_pred_layers))
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                scale = code_scale if name.startswith("decoder.code_blocks") else backbone_scale
                with torch.no_grad():
                    p.mul_(scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def _rope_slice(self, start: int, length: int, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        end = start + length
        if end > self.cfg.max_seq_len:
            raise ValueError(f"sequence position {end} exceeds max_seq_len={self.cfg.max_seq_len}")
        cos = self.rope_cos[start:end].to(dtype)
        sin = self.rope_sin[start:end].to(dtype)
        return cos, sin

    def _rope_gather(self, positions: torch.Tensor, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-position RoPE (positions ``[B, T]`` -> cos/sin ``[B, T, D]``)."""
        if int(positions.max()) >= self.cfg.max_seq_len:
            raise ValueError(f"position >= max_seq_len={self.cfg.max_seq_len}")
        cos = self.rope_cos[positions].to(dtype)   # [B, T, D]
        sin = self.rope_sin[positions].to(dtype)
        return cos, sin

    # ------------------------------------------------------------------
    # Mask helpers (public — used by training, eval, and rollout).
    # ------------------------------------------------------------------
    @staticmethod
    def next_packet_mask(valid: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        """Mask for next-packet prediction in packed training.

        ``valid``       ``[B, T]`` bool (True = real token).
        ``segment_ids`` ``[B, T]`` long (per-clip id; -1 = padding).

        Returns ``[B, T-1]`` bool — True where the target position is valid AND
        both the input and target belong to the same clip (no cross-clip prediction
        across packing boundaries).
        """
        return valid[:, 1:] & valid[:, :-1] & (segment_ids[:, 1:] == segment_ids[:, :-1])

    @staticmethod
    def doc_causal_mask(segment_ids: torch.Tensor) -> torch.Tensor:
        """Block-diagonal causal mask from ``segment_ids [B, T]`` -> bool ``[B, 1, T, T]``.

        A query attends to a key iff they share a segment id and the key is not
        in the future. The diagonal is always allowed (segment_ids[i]==segment_ids[i]),
        so padding rows never become fully masked (no NaNs).
        """
        same = segment_ids[:, :, None] == segment_ids[:, None, :]   # [B, T, T]
        T = segment_ids.shape[1]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=segment_ids.device))
        return (same & causal).unsqueeze(1)

    @staticmethod
    def mixed_attn_mask(role: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Prefix + mixed-attention mask -> bool ``[B, 1, S, S]``.

        ``role``  ``[B, S]``: 0 = text token, 1 = motion token (incl. motion BOS).
        ``valid`` ``[B, S]``: True where the key is a real (non-padding) token.

        Rules (the paper's mixed attention): text attends **bidirectionally** to
        text; motion attends to **all** text and **causally** to motion; text
        never attends to motion. The diagonal is always allowed so padding query
        rows are never fully masked (no NaNs).
        """
        B, S = role.shape
        dev = role.device
        is_text = role == 0                       # [B, S]
        is_mot = role == 1
        seq = torch.arange(S, device=dev)
        causal = (seq[None, :] <= seq[:, None])    # [S, S] key <= query
        qi_text = is_text[:, :, None]
        qi_mot = is_mot[:, :, None]
        kj_text = is_text[:, None, :]
        kj_mot = is_mot[:, None, :]
        kj_valid = valid[:, None, :]
        allow = kj_valid & (
            (qi_text & kj_text)
            | (qi_mot & kj_text)
            | (qi_mot & kj_mot & causal[None])
        )
        eye = torch.eye(S, dtype=torch.bool, device=dev)[None]
        allow = allow | eye
        return allow.unsqueeze(1)                   # [B, 1, S, S]

    # ------------------------------------------------------------------
    # Shared backbone
    # ------------------------------------------------------------------
    def _run_backbone(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        caches: list[KVCache] | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        """Run all backbone blocks + final RMSNorm. Returns (hidden, new_caches)."""
        new_caches: list[KVCache] | None = [] if caches is not None else None
        use_ckpt = self.training and self.cfg.gradient_checkpointing and caches is None
        for i, block in enumerate(self.blocks):
            layer_cache = caches[i] if caches is not None else None
            if use_ckpt:
                x, nc = torch.utils.checkpoint.checkpoint(
                    lambda x, cos, sin, cache, doc_mask: block(x, cos, sin, cache=cache, doc_mask=doc_mask),
                    x, cos, sin, layer_cache, doc_mask,
                    use_reentrant=False,
                )
            else:
                x, nc = block(x, cos, sin, cache=layer_cache, doc_mask=doc_mask)
            if new_caches is not None:
                new_caches.append(nc)
        return self.norm(x), new_caches

    # ------------------------------------------------------------------
    # Public forward paths (no boolean flags — fixed return types).
    # ------------------------------------------------------------------
    def forward_packed(
        self,
        codes: torch.Tensor,
        target_codes: torch.Tensor,
        segment_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Training forward with document-masked packing.

        ``codes [B, T, Q]``, ``segment_ids [B, T]``, ``positions [B, T]``
        -> per-codebook logits ``[B, T, V_i]``.
        """
        x = self.embed(codes)
        cos, sin = self._rope_gather(positions, x.dtype)
        mask = self.doc_causal_mask(segment_ids)
        x, _ = self._run_backbone(x, cos, sin, doc_mask=mask)
        return self.decoder(x, target_codes=target_codes)

    def forward_kv(
        self,
        codes: torch.Tensor,
        caches: list[KVCache],
        start_pos: int,
        *,
        return_hidden: bool = False,
    ) -> tuple:
        """KV-cached forward for rollout (single- or multi-step).

        ``codes [B, T, Q]`` -> ``(logits, new_caches)`` or
        ``(logits, new_caches, hidden)`` when ``return_hidden=True``.
        """
        x = self.embed(codes)
        cos, sin = self._rope_slice(start_pos, x.shape[1], x.dtype)
        x, new_caches = self._run_backbone(x, cos, sin, caches=caches)
        logits = self.decoder(x)
        if return_hidden:
            return logits, new_caches, x
        return logits, new_caches

    def forward_embeds(
        self,
        inputs_embeds: torch.Tensor,
        *,
        attn_mask: torch.Tensor,
        caches: list[KVCache] | None = None,
        start_pos: int = 0,
        run_decoder: bool = True,
        return_hidden: bool = False,
    ) -> tuple:
        """Forward from pre-computed embeddings (text+motion prefill / decode).

        ``inputs_embeds [B, S, D]`` with an explicit ``attn_mask [B, 1, S, S]``.
        ``run_decoder=False`` returns ``(new_caches, hidden)`` so the caller can
        slice the motion span before running the packet decoder.
        """
        cos, sin = self._rope_slice(start_pos, inputs_embeds.shape[1], inputs_embeds.dtype)
        x, new_caches = self._run_backbone(inputs_embeds, cos, sin, caches=caches, doc_mask=attn_mask)
        if not run_decoder:
            return new_caches, x
        logits = self.decoder(x)
        if return_hidden:
            return logits, new_caches, x
        return logits, new_caches

    # ------------------------------------------------------------------
    # Packet-level decode (public seam for rollout)
    # ------------------------------------------------------------------
    def decode_packet(
        self, hidden: torch.Tensor, prefix_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict logits for the next residual codebook during rollout.

        ``hidden`` is ``[B, D]`` for one packet. ``prefix_codes [B, K]``
        contains already-sampled codes q0..q_{K-1}.  If ``prefix_codes`` is
        ``None``, returns q0 logits only.
        """
        if prefix_codes is None:
            return self.decoder.q0_head(hidden.unsqueeze(1)).squeeze(1)
        return self.decoder.next_code_logits(hidden, prefix_codes)

    # ------------------------------------------------------------------
    # Text2Motion teacher-forced forward
    # ------------------------------------------------------------------
    # Text conditioning helpers
    # ------------------------------------------------------------------
    def project_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        """L2-normalize per token, then project ``clip_dim -> d_model``.

        Raw encoder states are not scale-compatible across Flan / SigLIP / Qwen3;
        normalizing here keeps the shared per-token conditioning interface without
        re-exporting caches. Zero pad rows stay zero (``F.normalize`` eps path).
        """
        return self.text_proj(F.normalize(text_emb, dim=-1))

    def replace_text_with_null(
        self, text_x: torch.Tensor, drop_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Replace projected text with the learned null token for CFG training.

        ``text_x [B, L, d_model]``, ``drop_mask [B]`` bool (True = drop this sample).
        Returns ``text_x`` with dropped rows replaced by ``self.null_text``.
        If ``drop_mask`` is None, returns ``text_x`` unchanged.
        """
        if drop_mask is None:
            return text_x
        B = text_x.shape[0]
        null = self.null_text.to(text_x.dtype).expand(B, text_x.shape[1], -1)
        return torch.where(drop_mask.view(B, 1, 1), null, text_x)

    @staticmethod
    def cfg_guide(cond: torch.Tensor, uncond: torch.Tensor, scale: float) -> torch.Tensor:
        """Classifier-free guidance: ``uncond + scale * (cond - uncond)``."""
        if scale == 1.0:
            return cond
        return uncond + scale * (cond - uncond)

    # ------------------------------------------------------------------
    def t2m_train_step(
        self,
        text_emb: torch.Tensor,
        text_valid: torch.Tensor,
        motion_codes: torch.Tensor,
        motion_valid: torch.Tensor,
        *,
        drop_text: torch.Tensor | None = None,
        return_pred_hidden: bool = False,
    ) -> tuple:
        """Teacher-forced text2motion pass.

        ``text_emb``    ``[B, Lt, clip_dim]`` frozen Flan-T5 word embeddings.
        ``text_valid``  ``[B, Lt]`` bool (True = real word).
        ``motion_codes````[B, Tm, Q]`` long packet codes.
        ``motion_valid````[B, Tm]`` bool (True = real motion frame).
        ``drop_text``   ``[B]`` bool (CFG: replace text with the null token).

        Returns ``(q_logits, eos_logits)`` where ``q_logits[i]`` is ``[B, Tm, V]``
        predicting ``motion_codes[..., i]`` (BOS->m0, m0->m1, ...), and
        ``eos_logits`` is ``[B, Tm+1]`` (stop logit at each motion-block position;
        index 0 = BOS).  When ``return_pred_hidden=True``, also returns the
        motion-span hidden ``[B, Tm, D]`` used for packet decoding (for
        free-running residual metrics).
        """
        if not self.cfg.use_text:
            raise RuntimeError("t2m_train_step requires cfg.use_text=True")
        B, Tm, _ = motion_codes.shape
        text_x = self.project_text(text_emb)                     # [B, Lt, D]
        text_x = self.replace_text_with_null(text_x, drop_text)
        mot_emb = self.embed(motion_codes)                       # [B, Tm, D]
        bos = self.motion_bos.to(mot_emb.dtype).expand(B, 1, -1)
        mot_x = torch.cat([bos, mot_emb], dim=1)                 # [B, Tm+1, D]
        x = torch.cat([text_x, mot_x], dim=1)                    # [B, S, D]

        Lt = text_x.shape[1]
        dev = x.device
        role = torch.cat(
            [
                torch.zeros(B, Lt, dtype=torch.long, device=dev),
                torch.ones(B, Tm + 1, dtype=torch.long, device=dev),
            ],
            dim=1,
        )
        valid = torch.cat(
            [
                text_valid.bool(),
                torch.ones(B, 1, dtype=torch.bool, device=dev),   # BOS
                motion_valid.bool(),
            ],
            dim=1,
        )
        mask = self.mixed_attn_mask(role, valid)
        _, hidden = self.forward_embeds(x, attn_mask=mask, run_decoder=False)
        mot_hidden = hidden[:, Lt:, :]                            # [B, Tm+1, D]
        pred_hidden = mot_hidden[:, :Tm, :]
        q_logits = self.decoder(pred_hidden, target_codes=motion_codes)
        eos_logits = self.eos_head(mot_hidden).squeeze(-1)       # [B, Tm+1]
        if return_pred_hidden:
            return q_logits, eos_logits, pred_hidden
        return q_logits, eos_logits

    def init_caches(self, batch_size: int, device, dtype) -> list[KVCache]:
        hd = self.cfg.head_dim
        hkv = self.cfg.kv_heads
        empty = lambda: torch.zeros(batch_size, hkv, 0, hd, device=device, dtype=dtype)
        return [KVCache(k=empty(), v=empty()) for _ in range(self.cfg.n_layers)]

    # ------------------------------------------------------------------
    @staticmethod
    def packet_ce_loss(
        cfg,
        logits: list[torch.Tensor],
        targets: torch.Tensor,
        mask: torch.Tensor,
        *,
        code_axis_scale: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Weighted per-codebook cross entropy.

        ``logits``  : list of ``[B, T, V_i]``
        ``targets`` : ``[B, T, Q]`` long (next-packet codes)
        ``mask``    : ``[B, T]`` bool (True = compute loss here)
        ``code_axis_scale`` : override for ``cfg.code_axis_loss_scale``
            (used for gradual warmup of residual loss).

        Returns ``(total_loss, metrics)`` where metrics holds per-codebook CE,
        top-1 accuracy, and the valid-token count.
        """
        weights = cfg.resolved_loss_weights()
        flat_mask = mask.reshape(-1)
        n_valid = flat_mask.sum().clamp_min(1)
        ce_values: list[torch.Tensor] = []
        metrics: dict[str, torch.Tensor] = {}
        for i, logit in enumerate(logits):
            V = logit.shape[-1]
            lg = logit.reshape(-1, V)
            tg = targets[..., i].reshape(-1)
            ce_per = F.cross_entropy(lg, tg, reduction="none")
            ce = (ce_per * flat_mask).sum() / n_valid
            with torch.no_grad():
                pred = lg.argmax(dim=-1)
                acc = ((pred == tg) & flat_mask).sum() / n_valid
            ce_values.append(ce)
            metrics[f"ce_q{i}"] = ce.detach()
            metrics[f"acc_q{i}"] = acc

        q0_loss = weights[0] * ce_values[0]
        if len(ce_values) > 1:
            residual_weights = weights[1:]
            residual_total = torch.zeros((), device=targets.device, dtype=torch.float32)
            residual_wsum = 0.0
            for ce, w in zip(ce_values[1:], residual_weights, strict=True):
                residual_total = residual_total + w * ce
                residual_wsum += w
            code_axis_loss = residual_total / max(residual_wsum, 1e-8)
            scale = code_axis_scale if code_axis_scale is not None else cfg.code_axis_loss_scale
            total = q0_loss + scale * code_axis_loss
        else:
            code_axis_loss = torch.zeros((), device=targets.device, dtype=torch.float32)
            total = q0_loss

        metrics["loss_q0"] = q0_loss.detach()
        metrics["loss_code_axis"] = code_axis_loss.detach()
        metrics["loss"] = total.detach()
        metrics["n_valid"] = n_valid.detach()
        return total, metrics
