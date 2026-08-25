"""Train a Flan-T5 TMR text+motion encoders + decoder (TEMOS phase).

Both encoders and decoder are warm-started from NVIDIA TMR-SOMA-RP and trained
with reconstruction + contrastive + KL losses.

Examples::

    torchrun --nproc_per_node=8 -m semoco_generator.train.train_tmr_flan_temos \
        --config configs/tmr_flan_temos.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from ..dataset.tmr_flan_joint_dataset import TMRFlanJointDataset, collate_tmr_flan_joint
from kimodo.model.tmr import ACTORStyleEncoder
from kimodo.model.loading import load_checkpoint_state_dict
from kimodo.motion_rep import TMRMotionRep
from kimodo.skeleton import SOMASkeleton30
from ..local_uri import resolve_local_uri
from .base_trainer import run_training
from .tmr_flan_module import TMRFlanConfig
from .tmr_flan_joint_module import ACTORStyleDecoder
from .tmr_flan_temos_module import TMRFlanTEMOSModule
from .utils import dist_info, log

def _default_soma_snapshot() -> Path:
    """Resolve TMR through the shared Kimodo/HF resolver, not cache internals."""
    from ..eval.tmr.kimodo_compat import resolve_tmr_checkpoint

    return resolve_tmr_checkpoint("tmr-soma-rp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TMR-Flan TEMOS phase.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
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
    train_ds = TMRFlanJointDataset(
        codes_root=data_cfg["codes_root"],
        features_root=data_cfg["features_root"],
        split="train",
    )
    val_ds = TMRFlanJointDataset(
        codes_root=data_cfg["codes_root"],
        features_root=data_cfg["features_root"],
        split="val",
    )
    log(rank, f"[data] train={len(train_ds)}  val={len(val_ds)}  "
              f"text_dim={train_ds.text_dim}")

    # ---- model --------------------------------------------------------------
    mcfg = TMRFlanConfig(
        latent_dim=int(model_cfg.get("latent_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        ff_size=int(model_cfg.get("ff_size", 1024)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        temperature=float(model_cfg.get("temperature", 0.1)),
        text_input_dim=int(model_cfg.get("text_input_dim", train_ds.text_dim)),
    )

    snap = Path(model_cfg.get("warm_start_snapshot", str(_default_soma_snapshot())))

    # ---- Motion encoder (warm-start from TMR-SOMA-RP, trainable) ------------
    skeleton = SOMASkeleton30()
    motion_rep = TMRMotionRep(skeleton, fps=30, stats_path=str(snap / "stats" / "motion"))
    motion_encoder = ACTORStyleEncoder(
        motion_rep=motion_rep, llm_shape=None, vae=True,
        latent_dim=mcfg.latent_dim, ff_size=mcfg.ff_size,
        num_layers=mcfg.num_layers, num_heads=mcfg.num_heads,
        dropout=mcfg.dropout, activation="gelu",
    )
    motion_encoder.load_state_dict(
        load_checkpoint_state_dict(snap / "last_weights" / "motion_encoder.pt"),
    )

    # ---- Text encoder (warm-start from motion encoder weights) ---------------
    text_encoder = ACTORStyleEncoder(
        motion_rep=None, llm_shape=(-1, mcfg.text_input_dim), vae=True,
        latent_dim=mcfg.latent_dim, ff_size=mcfg.ff_size,
        num_layers=mcfg.num_layers, num_heads=mcfg.num_heads,
        dropout=mcfg.dropout, activation="gelu",
    )
    # Copy transformer weights from motion encoder (shared architecture)
    msd = motion_encoder.state_dict()
    tsd = text_encoder.state_dict()
    for k in tsd:
        if k in msd and tsd[k].shape == msd[k].shape:
            tsd[k].copy_(msd[k])
    text_encoder.load_state_dict(tsd)

    # ---- Motion decoder (warm-start from TMR-SOMA-RP, trainable) ------------
    motion_decoder = ACTORStyleDecoder(
        nfeats=186, latent_dim=mcfg.latent_dim, ff_size=mcfg.ff_size,
        num_layers=mcfg.num_layers, num_heads=mcfg.num_heads,
        dropout=mcfg.dropout, activation="gelu",
    )
    motion_decoder.load_state_dict(
        load_checkpoint_state_dict(snap / "last_weights" / "motion_decoder.pt"),
    )

    module = TMRFlanTEMOSModule(
        text_encoder=text_encoder,
        motion_encoder=motion_encoder,
        motion_decoder=motion_decoder,
        temperature=mcfg.temperature,
    ).to(device)

    log(rank, f"[model] trainable params={module.num_parameters() / 1e6:.1f}M  "
              f"text_dim={mcfg.text_input_dim}  latent_dim={mcfg.latent_dim}  "
              f"layers={mcfg.num_layers}  heads={mcfg.num_heads}  TEMOS")

    # ---- per-trainer callbacks ----------------------------------------------
    def compute_loss(core, batch, dev, amp_dtype, *, step=0):
        return core(batch, dev, amp_dtype, step=step)

    def compute_val_metrics(core, loader, dev, amp_dtype):
        """InfoNCE loss on validation set."""
        core.eval()
        total_contrastive = 0.0
        total_recons = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                if count >= 50 * int(max(1, len(batch["motion_feat_valid"]))):
                    break
                _, m = core(batch, dev, amp_dtype, step=0)
                bs = len(batch["motion_feat_valid"])
                total_contrastive += float(m["contrastive"].item()) * bs
                total_recons += float(m["recons"].item()) * bs
                count += bs
        core.train()
        return {
            "val/ce_mean": total_contrastive / max(count, 1),
            "val/recons": total_recons / max(count, 1),
        }

    def train_log_metrics(metrics, step, dt):
        return {
            "train/contrastive": float(metrics["contrastive"].item()),
            "train/recons": float(metrics["recons"].item()),
            "train/latent": float(metrics["latent"].item()),
            "train/kl": float(metrics["kl"].item()),
            "train/scale": float(metrics["scale"].item()),
            "train/temperature": 1.0 / float(metrics["scale"].item()),
        }

    # ---- checkpoint export (extra tensors) ----------------------------------
    def _ckpt_extra(val_metrics=None):
        extra = {}
        if val_metrics:
            extra["val_metrics"] = val_metrics
        extra["motion_decoder_state"] = module.motion_decoder.state_dict()
        return extra

    # ---- run ----------------------------------------------------------------
    out_dir = resolve_local_uri(args.out_dir or train_cfg.get("out_dir", "runs/tmr-soma-flan-temos"))

    run_training(
        model=module, mcfg=mcfg, train_ds=train_ds, val_ds=val_ds,
        collate_fn=collate_tmr_flan_joint, compute_loss=compute_loss,
        compute_val_metrics=compute_val_metrics, train_log_metrics=train_log_metrics,
        optim_cfg=optim_cfg, train_cfg=train_cfg, swan_cfg=swan_cfg,
        out_dir=out_dir, payload=payload, resume_from=args.resume,
        rank=rank, world=world, local=local, is_ddp=is_ddp, device=device, seed=seed,
        ckpt_extra=_ckpt_extra,
    )

    # ---- post-training: export encoders + decoder for TMR phase -------------
    if rank == 0:
        ckpt_dir = out_dir / "model"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(module.motion_encoder.state_dict(), ckpt_dir / "motion_encoder.pt")
        torch.save(module.text_encoder.state_dict(), ckpt_dir / "text_encoder.pt")
        torch.save(module.motion_decoder.state_dict(), ckpt_dir / "motion_decoder.pt")
        log(rank, f"[export] encoders + decoder saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
