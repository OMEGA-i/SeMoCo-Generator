"""``TMRFlanJointModule`` — joint text+motion encoder + decoder training.

Both encoders and decoder are trainable.  Includes reconstruction loss (weight
1.0, matching original TMR) to prevent latent space drift and overfitting.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tmr_flan_module import KLLoss


# ---------------------------------------------------------------------------
# ACTORStyleDecoder — matches original TMR (Mathux/TMR)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000.0, -torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class ACTORStyleDecoder(nn.Module):
    """TransformerDecoder that reconstructs motion features from a latent vector.

    Matches the original TMR implementation (Mathux/TMR, ICCV 2023).
    """

    def __init__(self, nfeats=186, latent_dim=256, ff_size=1024, num_layers=6,
                 num_heads=4, dropout=0.1, activation="gelu"):
        super().__init__()
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, activation=activation, batch_first=True,
        )
        self.seqTransDecoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.final_layer = nn.Linear(latent_dim, nfeats)

    def forward(self, z_dict):
        z = z_dict["z"]       # [B, latent_dim]
        mask = z_dict["mask"] # [B, nframes] bool (True=valid)
        B, nframes = mask.shape
        memory = z.unsqueeze(1)  # [B, 1, latent_dim]
        queries = torch.zeros(B, nframes, self.latent_dim, device=z.device)
        queries = self.sequence_pos_encoding(queries)
        output = self.seqTransDecoder(
            queries, memory,
            tgt_key_padding_mask=~mask,
        )
        output = self.final_layer(output)  # [B, nframes, nfeats]
        output[~mask] = 0.0
        return output


class TMRFlanJointModule(nn.Module):
    """Joint text + motion encoder + decoder training (full TMR loss).

    Parameters
    ----------
    text_encoder: ACTORStyleEncoder(llm_shape=(-1,2048), vae=True, 6 layers)
    motion_encoder: ACTORStyleEncoder(motion_rep=..., vae=True, 6 layers)
    motion_decoder: ACTORStyleDecoder(nfeats=186, ...)
        Warm-started from tmr-soma-rp, trainable.
    temperature: InfoNCE temperature (default 0.1).
    """

    def __init__(self, text_encoder, motion_encoder, motion_decoder, temperature=0.1):
        super().__init__()
        self.text_encoder = text_encoder
        self.motion_encoder = motion_encoder
        self.motion_decoder = motion_decoder
        self.logit_scale = nn.Parameter(torch.tensor([np.log(1.0 / temperature)], dtype=torch.float32))
        self.latent_loss_fn = nn.SmoothL1Loss()
        self.recons_loss_fn = nn.SmoothL1Loss()
        self.kl_loss_fn = KLLoss()

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch, device, amp_dtype, *, step=0):
        B = len(batch["motion_feat_valid"])

        # ---- Ground truth motion features -----------------------------------
        motion_feat = batch["motion_feat"].to(device, non_blocking=True)   # [B, Tmax, 186]
        motion_valid = batch["motion_feat_valid"].to(device, non_blocking=True)

        # ---- Motion encoder forward -----------------------------------------
        motion_inputs = {"x": motion_feat, "mask": motion_valid}
        motion_encoded = self.motion_encoder(motion_inputs)  # [B, 2, 256]
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

        # ---- Reconstruction (λ=1.0) — decode both latents, compare to GT ----
        t_recon = self.motion_decoder({"z": t_z, "mask": motion_valid})  # [B, Tmax, 186]
        m_recon = self.motion_decoder({"z": m_z, "mask": motion_valid})  # [B, Tmax, 186]
        recons = (self.recons_loss_fn(t_recon, motion_feat) +
                  self.recons_loss_fn(m_recon, motion_feat)) / 2.0
        # Zero out padded positions
        recons = recons * motion_valid.float().mean()  # scale by valid ratio

        # ---- InfoNCE contrastive (λ=0.1) ------------------------------------
        scale = self.logit_scale.exp()
        logits = scale * (t_z_n @ m_z_n.T)
        labels = torch.arange(B, device=device)
        contrastive = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0

        # ---- Latent alignment (λ=1e-5) --------------------------------------
        latent = self.latent_loss_fn(t_mu, m_mu)

        # ---- KL losses (matching original TMR) ------------------------------
        zero_ref = (torch.zeros_like(t_mu), torch.zeros_like(t_logvar))
        kl_prior = (self.kl_loss_fn((t_mu, t_logvar), zero_ref) +
                    self.kl_loss_fn((m_mu, m_logvar), zero_ref)) / 2.0
        kl_cross = (self.kl_loss_fn((t_mu, t_logvar), (m_mu.detach(), m_logvar.detach())) +
                    self.kl_loss_fn((m_mu, m_logvar), (t_mu.detach(), t_logvar.detach()))) / 2.0

        # ---- Total (weights matching original TMR) --------------------------
        loss = 1.0 * recons + 0.1 * contrastive + 1e-5 * latent + 1e-5 * kl_prior + 1e-5 * kl_cross

        metrics = {
            "loss": loss.detach(),
            "recons": recons.detach(),
            "contrastive": contrastive.detach(),
            "latent": latent.detach(),
            "kl_prior": kl_prior.detach(),
            "kl_cross": kl_cross.detach(),
            "scale": scale.detach(),
        }
        return loss, metrics
