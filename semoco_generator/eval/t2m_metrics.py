"""T2M-specific loss and validation helpers.

These were previously embedded in ``train/train_t2m.py``, forcing the eval layer
to import from the training layer.  Now both ``train_t2m.py`` and
``eval/t2m_token_eval.py`` import from here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..model import MotionGPT


def eos_targets(motion_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """From ``motion_valid [B, Tm]`` -> (``eos_target [B, Tm+1]``, ``eos_mask [B, Tm+1]``).

    EOS fires at motion-block index ``n`` = number of real frames (BOS is index 0),
    i.e. right after the last frame. Loss is computed over indices ``0..n``.
    """
    n = motion_valid.sum(dim=1)                               # [B]
    Tb = motion_valid.shape[1] + 1
    ar = torch.arange(Tb, device=motion_valid.device).unsqueeze(0)   # [1, Tb]
    eos_target = (ar == n.unsqueeze(1)).float()
    eos_mask = (ar <= n.unsqueeze(1))
    return eos_target, eos_mask


def t2m_losses(
    core: MotionGPT,
    batch: dict,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    cfg_drop: float,
    global_step: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Compute T2M training loss (code CE + EOS BCE + CFG dropout)."""
    text_emb = batch["text_emb"].to(device, non_blocking=True)
    text_valid = batch["text_valid"].to(device, non_blocking=True)
    motion_codes = batch["motion_codes"].to(device, non_blocking=True)
    motion_valid = batch["motion_valid"].to(device, non_blocking=True)

    B = motion_codes.shape[0]
    drop_text = None
    if cfg_drop > 0:
        drop_text = torch.rand(B, device=device) < cfg_drop

    # Staged training: q0-only before code_axis_start_step, then warm up code_axis loss
    code_axis_scale = None
    start = core.cfg.code_axis_start_step
    warmup = core.cfg.code_axis_warmup_steps
    if start > 0 and global_step < start:
        code_axis_scale = 0.0                         # stage 1: backbone + q0 only
    elif warmup > 0 and global_step < start + warmup:
        code_axis_scale = ((global_step - start) / warmup) * core.cfg.code_axis_loss_scale

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
        q_logits, eos_logits = core.t2m_train_step(
            text_emb, text_valid, motion_codes, motion_valid, drop_text=drop_text,
        )
        code_loss, metrics = core.packet_ce_loss(
            core.cfg, q_logits, motion_codes, motion_valid,
            code_axis_scale=code_axis_scale,
        )
        eos_tgt, eos_msk = eos_targets(motion_valid)
        bce = F.binary_cross_entropy_with_logits(eos_logits.float(), eos_tgt, reduction="none")
        eos_loss = (bce * eos_msk).sum() / eos_msk.sum().clamp_min(1)
        total = code_loss + core.cfg.eos_loss_scale * eos_loss

    with torch.no_grad():
        eos_pred = (eos_logits > 0)
        eos_acc = ((eos_pred == (eos_tgt > 0.5)) & eos_msk).sum() / eos_msk.sum().clamp_min(1)
    metrics["loss_eos"] = eos_loss.detach()
    metrics["acc_eos"] = eos_acc
    metrics["loss_total"] = total.detach()
    return total, metrics


@torch.no_grad()
def codebook_free_run_acc(
    core: MotionGPT,
    pred_hidden: torch.Tensor,
    motion_codes: torch.Tensor,
    motion_valid: torch.Tensor,
) -> dict[str, float]:
    """Per-codebook acc with **greedy free-running** codebook-axis prefixes.

    Time axis stays teacher-forced (``pred_hidden`` from GT previous packets).
    Within each packet, ``q_k`` is conditioned on the model's own argmax
    ``q0..q_{k-1}`` — not GT.  ``roll_acc_q0`` should match TF ``acc_q0``;
    deeper ``roll_acc_q*`` expose residual exposure bias that TF hides.
    """
    roll = core.decoder.greedy_codes(pred_hidden)                 # [B, T, Q]
    q = motion_codes.shape[-1]
    n_valid = motion_valid.sum().clamp_min(1).float()
    out: dict[str, float] = {}
    for k in range(q):
        hit = ((roll[..., k] == motion_codes[..., k]) & motion_valid).sum().float()
        out[f"roll_acc_q{k}"] = float(hit / n_valid)
    out["roll_acc_mean"] = sum(out[f"roll_acc_q{k}"] for k in range(q)) / max(q, 1)
    return out


@torch.no_grad()
def eval_t2m(
    core: MotionGPT,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_batches: int = 50,
) -> dict[str, float]:
    """Run T2M validation over ``max_batches`` batches.

    Logs teacher-forced CE/acc plus codebook-axis free-running ``val/roll_acc_*``.
    """
    was_training = core.training
    core.eval()
    q = core.cfg.num_codebooks
    agg: dict[str, float] = {}
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        text_emb = batch["text_emb"].to(device, non_blocking=True)
        text_valid = batch["text_valid"].to(device, non_blocking=True)
        motion_codes = batch["motion_codes"].to(device, non_blocking=True)
        motion_valid = batch["motion_valid"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            q_logits, eos_logits, pred_hidden = core.t2m_train_step(
                text_emb, text_valid, motion_codes, motion_valid,
                drop_text=None, return_pred_hidden=True,
            )
            code_loss, metrics = core.packet_ce_loss(core.cfg, q_logits, motion_codes, motion_valid)
            eos_tgt, eos_msk = eos_targets(motion_valid)
            bce = F.binary_cross_entropy_with_logits(eos_logits.float(), eos_tgt, reduction="none")
            eos_loss = (bce * eos_msk).sum() / eos_msk.sum().clamp_min(1)
            total = code_loss + core.cfg.eos_loss_scale * eos_loss
            roll = codebook_free_run_acc(core, pred_hidden, motion_codes, motion_valid)

        eos_pred = eos_logits > 0
        eos_acc = ((eos_pred == (eos_tgt > 0.5)) & eos_msk).sum() / eos_msk.sum().clamp_min(1)

        agg["val/loss"] = agg.get("val/loss", 0.0) + float(total)
        agg["val/loss_eos"] = agg.get("val/loss_eos", 0.0) + float(eos_loss)
        agg["val/acc_eos"] = agg.get("val/acc_eos", 0.0) + float(eos_acc)
        for k in range(q):
            agg[f"val/ce_q{k}"] = agg.get(f"val/ce_q{k}", 0.0) + float(metrics[f"ce_q{k}"])
            agg[f"val/acc_q{k}"] = agg.get(f"val/acc_q{k}", 0.0) + float(metrics[f"acc_q{k}"])
            agg[f"val/roll_acc_q{k}"] = agg.get(f"val/roll_acc_q{k}", 0.0) + roll[f"roll_acc_q{k}"]
        agg["val/roll_acc_mean"] = agg.get("val/roll_acc_mean", 0.0) + roll["roll_acc_mean"]
        n += 1
    if was_training:
        core.train()
    n = max(n, 1)
    out = {k: v / n for k, v in agg.items()}
    out["val/ce_mean"] = sum(out[f"val/ce_q{k}"] for k in range(q)) / q
    return out
