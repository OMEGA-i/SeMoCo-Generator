"""Train a Flan-T5 TMR text encoder with contrastive learning (frozen motion encoder).

Loads pre-computed motion latents + Flan-T5 text embeddings, trains only the
ACTORStyleEncoder text side with InfoNCE + latent + KL losses.

Examples::

    # single GPU
    python -m semoco_generator.train.train_tmr_flan --config configs/tmr_flan.yaml

    # DDP
    torchrun --nproc_per_node=8 -m semoco_generator.train.train_tmr_flan \\
        --config configs/tmr_flan.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from ..dataset.tmr_flan_dataset import TMRFlanDataset, collate_tmr_flan
from kimodo.model.tmr import ACTORStyleEncoder
from ..local_uri import resolve_local_uri
from .base_trainer import run_training
from .tmr_flan_module import TMRFlanConfig, TMRFlanModule
from .utils import dist_info, log


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TMR-Flan text encoder.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="override train.out_dir")
    parser.add_argument("--resume", type=str, default=None,
                        help="resume from checkpoint")
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text())
    data_cfg = payload.get("data", {})
    model_cfg = payload.get("model", {})
    optim_cfg = payload.get("optim", {})
    train_cfg = payload.get("train", {})
    swan_cfg = payload.get("swanlab", {})

    rank, world, local, is_ddp = dist_info()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if is_ddp:
        dist.init_process_group(backend="nccl")

    seed = int(train_cfg.get("seed", 3407)) + rank
    torch.manual_seed(seed)

    # ---- data ---------------------------------------------------------------
    train_ds = TMRFlanDataset(
        codes_root=data_cfg["codes_root"],
        motion_latents_root=data_cfg["motion_latents_root"],
        split="train",
    )
    val_ds = TMRFlanDataset(
        codes_root=data_cfg["codes_root"],
        motion_latents_root=data_cfg["motion_latents_root"],
        split="val",
    )
    log(rank, f"[data] train={len(train_ds)}  val={len(val_ds)}  "
              f"clip_dim={train_ds.clip_dim}")

    # ---- model --------------------------------------------------------------
    mcfg = TMRFlanConfig(
        latent_dim=int(model_cfg.get("latent_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        ff_size=int(model_cfg.get("ff_size", 1024)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        temperature=float(model_cfg.get("temperature", 0.1)),
        text_input_dim=int(model_cfg.get("text_input_dim", train_ds.clip_dim)),
    )
    if mcfg.text_input_dim != train_ds.clip_dim:
        raise ValueError(
            f"model text_input_dim={mcfg.text_input_dim} != "
            f"store clip_dim={train_ds.clip_dim}"
        )

    # Build text encoder from scratch (same arch as TMR-SOMA-RP but Flan-T5 input)
    text_encoder = ACTORStyleEncoder(
        motion_rep=None,
        llm_shape=(-1, mcfg.text_input_dim),
        vae=True,
        latent_dim=mcfg.latent_dim,
        ff_size=mcfg.ff_size,
        num_layers=mcfg.num_layers,
        num_heads=mcfg.num_heads,
        dropout=mcfg.dropout,
        activation="gelu",
    ).to(device)

    module = TMRFlanModule(
        text_encoder=text_encoder,
        temperature=mcfg.temperature,
    ).to(device)

    log(rank, f"[model] trainable params={module.num_parameters() / 1e6:.1f}M  "
              f"text_dim={mcfg.text_input_dim}  latent_dim={mcfg.latent_dim}  "
              f"layers={mcfg.num_layers}  heads={mcfg.num_heads}")

    # ---- per-trainer callbacks ----------------------------------------------
    def compute_loss(core, batch, dev, amp_dtype, *, step=0):
        return core(batch, dev, amp_dtype, step=step)

    def compute_val_metrics(core, loader, dev, amp_dtype):
        """InfoNCE loss on validation set."""
        core.eval()
        total_loss = 0.0
        total_contrastive = 0.0
        total_latent = 0.0
        total_kl = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                if count >= 50:  # cap at 50 batches for speed
                    break
                _, m = core(batch, dev, amp_dtype, step=0)
                bs = len(batch["motion_latent"])
                total_loss += float(m["loss"].item()) * bs
                total_contrastive += float(m["contrastive"].item()) * bs
                total_latent += float(m["latent"].item()) * bs
                total_kl += float(m["kl"].item()) * bs
                count += bs
        core.train()
        return {
            "val/ce_mean": total_contrastive / max(count, 1),
            "val/loss": total_loss / max(count, 1),
            "val/latent": total_latent / max(count, 1),
            "val/kl": total_kl / max(count, 1),
        }

    def train_log_metrics(metrics, step, dt):
        return {
            "train/contrastive": float(metrics["contrastive"].item()),
            "train/latent": float(metrics["latent"].item()),
            "train/kl": float(metrics["kl"].item()),
            "train/scale": float(metrics["scale"].item()),
            "train/temperature": 1.0 / float(metrics["scale"].item()),
        }

    # ---- run ----------------------------------------------------------------
    out_dir = resolve_local_uri(args.out_dir or train_cfg.get("out_dir", "runs/tmr-soma-flan"))

    run_training(
        model=module, mcfg=mcfg, train_ds=train_ds, val_ds=val_ds,
        collate_fn=collate_tmr_flan, compute_loss=compute_loss,
        compute_val_metrics=compute_val_metrics, train_log_metrics=train_log_metrics,
        optim_cfg=optim_cfg, train_cfg=train_cfg, swan_cfg=swan_cfg,
        out_dir=out_dir, payload=payload, resume_from=args.resume,
        rank=rank, world=world, local=local, is_ddp=is_ddp, device=device, seed=seed,
    )

    # ---- post-training: export text_encoder.pt for TMR checkpoint assembly ----
    if rank == 0:
        ckpt_dir = out_dir / "model"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(module.text_encoder.state_dict(), ckpt_dir / "text_encoder.pt")
        log(rank, f"[export] text_encoder.pt saved to {ckpt_dir / 'text_encoder.pt'}")


if __name__ == "__main__":
    main()
