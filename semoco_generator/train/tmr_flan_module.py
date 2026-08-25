"""``TMRFlanModule`` — trainable Flan-T5 text encoder with InfoNCE + latent + KL losses.

Matches the original TMR training (Mathux/TMR, ICCV 2023) but adapted for
frozen motion encoder:

- **No reconstruction loss** (motion decoder is frozen)
- **No cross-modal KL** (motion encoder is frozen, can't adapt)
- **Contrastive (InfoNCE)** on L2-normalized latents with learnable temperature
- **Latent alignment** (SmoothL1) between text-mu and frozen motion-mu
- **KL prior** regularising text distribution toward N(0,1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# KLLoss — matches original TMR implementation exactly
# ---------------------------------------------------------------------------

class KLLoss:
    """KL divergence between two diagonal Gaussians: KL(q ‖ p).

    From ``src/model/losses.py`` in Mathux/TMR.
    """

    def __call__(
        self,
        q: tuple[torch.Tensor, torch.Tensor],
        p: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        mu_q, logvar_q = q
        mu_p, logvar_p = p
        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        return 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio).mean()


# ---------------------------------------------------------------------------
# Module config — simple dataclass (base_trainer calls asdict(mcfg))
# ---------------------------------------------------------------------------

@dataclass
class TMRFlanConfig:
    """Configuration for the TMR Flan-T5 text encoder."""
    latent_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4
    ff_size: int = 1024
    dropout: float = 0.1
    temperature: float = 0.1
    text_input_dim: int = 2048  # Flan-T5-XL clip_dim


# ---------------------------------------------------------------------------
# TMRFlanModule
# ---------------------------------------------------------------------------

class TMRFlanModule(nn.Module):
    """Trainable text-encoder wrapper with InfoNCE + latent + KL losses.

    Parameters
    ----------
    text_encoder
        ``ACTORStyleEncoder(llm_shape=(-1, 2048), vae=True, num_layers=6, ...)``
    temperature
        Initial InfoNCE temperature (default 0.1, from original TMR config).
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(
            torch.tensor([np.log(1.0 / temperature)], dtype=torch.float32),
        )
        self.latent_loss_fn = nn.SmoothL1Loss()
        self.kl_loss_fn = KLLoss()

    def num_parameters(self) -> int:
        """Total trainable parameter count (used by base_trainer for logging)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device,
        amp_dtype: torch.dtype,
        *,
        step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute InfoNCE + latent + KL losses.

        Returns:
            (scalar_loss, metrics_dict)
        """
        B = len(batch["motion_latent"])

        motion_mu = batch["motion_latent"].to(device, non_blocking=True)   # [B, 256]
        text_emb = batch["text_emb"].to(device, non_blocking=True)          # [B, Lmax, 2048]
        text_valid = batch["text_valid"].to(device, non_blocking=True)      # [B, Lmax]

        # ---- Text encoder forward -------------------------------------------
        text_inputs = {"x": text_emb, "mask": text_valid}
        text_encoded = self.text_encoder(text_inputs)  # [B, 2, 256]
        t_mu, t_logvar = text_encoded.unbind(1)          # [B, 256], [B, 256]

        # ---- VAE reparameterization (matches original TMR training) ---------
        t_std = t_logvar.exp().pow(0.5)
        t_z = t_mu + torch.randn_like(t_std) * t_std    # [B, 256]

        # ---- L2-normalize for contrastive -----------------------------------
        t_z_norm = F.normalize(t_z, dim=-1)
        m_z_norm = F.normalize(motion_mu, dim=-1)

        # ---- InfoNCE contrastive loss (λ = 0.1) -----------------------------
        scale = self.logit_scale.exp()
        logits = scale * (t_z_norm @ m_z_norm.T)        # [B, B]
        labels = torch.arange(B, device=device)
        contrastive = (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        ) / 2.0

        # ---- Latent alignment loss (λ = 1e-5) -------------------------------
        latent_loss = self.latent_loss_fn(t_mu, motion_mu)

        # ---- KL prior loss (λ = 1e-5) ---------------------------------------
        zero_ref = (torch.zeros_like(t_mu), torch.zeros_like(t_logvar))
        kl_loss = self.kl_loss_fn((t_mu, t_logvar), zero_ref)

        # ---- Weighted sum (from original TMR loss weights) ------------------
        loss = 0.1 * contrastive + 1e-5 * latent_loss + 1e-5 * kl_loss

        metrics = {
            "loss": loss.detach(),
            "contrastive": contrastive.detach(),
            "latent": latent_loss.detach(),
            "kl": kl_loss.detach(),
            "scale": scale.detach(),
        }
        return loss, metrics
