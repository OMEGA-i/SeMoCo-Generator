"""``TMRFlanTEMOSModule`` — TEMOS phase: joint encoder + decoder training.

Motion encoder + text encoder + decoder all trainable from warm-start.
Matches original TEMOS losses: reconstruction (λ=1.0), contrastive (λ=0.1),
KL prior (λ=1e-5).  No cross-modal KL — that is added in the TMR phase.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tmr_flan_module import KLLoss


class TMRFlanTEMOSModule(nn.Module):
    """TEMOS phase: train encoders + decoder jointly.

    Parameters
    ----------
    text_encoder: ACTORStyleEncoder(llm_shape=(-1,2048), vae=True, 6 layers)
    motion_encoder: ACTORStyleEncoder(motion_rep=..., vae=True, 6 layers)
    motion_decoder: ACTORStyleDecoder(nfeats=186, ...)
    temperature: InfoNCE temperature (default 0.1).
    """

    def __init__(self, text_encoder, motion_encoder, motion_decoder, temperature=0.1):
        super().__init__()
        self.text_encoder = text_encoder
        self.motion_encoder = motion_encoder
        self.motion_decoder = motion_decoder
        self.logit_scale = nn.Parameter(
            torch.tensor([np.log(1.0 / temperature)], dtype=torch.float32),
        )
        self.latent_loss_fn = nn.SmoothL1Loss()
        self.recons_loss_fn = nn.SmoothL1Loss()
        self.kl_loss_fn = KLLoss()

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch, device, amp_dtype, *, step=0):
        B = len(batch["motion_feat_valid"])

        # ---- Ground truth motion features -----------------------------------
        motion_feat = batch["motion_feat"].to(device, non_blocking=True)
        motion_valid = batch["motion_feat_valid"].to(device, non_blocking=True)

        # ---- Motion encoder forward -----------------------------------------
        motion_inputs = {"x": motion_feat, "mask": motion_valid}
        motion_encoded = self.motion_encoder(motion_inputs)
        m_mu, m_logvar = motion_encoded.unbind(1)

        # ---- Text encoder forward -------------------------------------------
        text_emb = batch["text_emb"].to(device, non_blocking=True)
        text_valid = batch["text_valid"].to(device, non_blocking=True)
        text_inputs = {"x": text_emb, "mask": text_valid}
        text_encoded = self.text_encoder(text_inputs)
        t_mu, t_logvar = text_encoded.unbind(1)

        # ---- VAE reparameterization -----------------------------------------
        t_std = t_logvar.exp().pow(0.5)
        t_z = t_mu + torch.randn_like(t_std) * t_std
        m_std = m_logvar.exp().pow(0.5)
        m_z = m_mu + torch.randn_like(m_std) * m_std

        # ---- L2-normalize ---------------------------------------------------
        t_z_n = F.normalize(t_z, dim=-1)
        m_z_n = F.normalize(m_z, dim=-1)

        # ---- Reconstruction: decode from motion latent (λ=1.0) --------------
        m_recon = self.motion_decoder({"z": m_z, "mask": motion_valid})
        recons = self.recons_loss_fn(m_recon, motion_feat)

        # ---- InfoNCE contrastive (λ=0.1) ------------------------------------
        scale = self.logit_scale.exp()
        logits = scale * (t_z_n @ m_z_n.T)
        labels = torch.arange(B, device=device)
        contrastive = (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        ) / 2.0

        # ---- Latent alignment (λ=1e-5) --------------------------------------
        latent = self.latent_loss_fn(t_mu, m_mu)

        # ---- KL prior (λ=1e-5) ----------------------------------------------
        zero_ref = (torch.zeros_like(m_mu), torch.zeros_like(m_logvar))
        kl = (self.kl_loss_fn((m_mu, m_logvar), zero_ref) +
              self.kl_loss_fn((t_mu, t_logvar), zero_ref)) / 2.0

        # ---- Total (matching original TEMOS weights) ------------------------
        loss = 1.0 * recons + 0.1 * contrastive + 1e-5 * latent + 1e-5 * kl

        metrics = {
            "loss": loss.detach(),
            "recons": recons.detach(),
            "contrastive": contrastive.detach(),
            "latent": latent.detach(),
            "kl": kl.detach(),
            "scale": scale.detach(),
        }
        return loss, metrics
