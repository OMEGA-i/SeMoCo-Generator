"""Stage 1 - train SeMoCo-Generator (next-packet prediction, teacher forcing).

Single-GPU or torchrun-DDP. Loads packed motion codes exported by
``tools/export_motion_codes.py``, trains the packet LM with weighted per-codebook
cross entropy, and logs token-level metrics (per-codebook CE, top-1/top-5
accuracy, perplexity).

Examples::

    # single GPU
    python -m semoco_generator.train.train_motion_gpt \
        --config configs/motion_gpt_150m.yaml

    # 4-GPU DDP
    torchrun --nproc_per_node=4 -m semoco_generator.train.train_motion_gpt \
        --config configs/motion_gpt_150m.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from ..dataset import MotionCodeDataset, collate_motion_codes
from ..eval.metrics import compute_token_metrics
from ..local_uri import resolve_local_uri
from ..model import MotionGPT, MotionGPTConfig
from .base_trainer import run_training
from .utils import dist_info, log


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SeMoCo-Generator (Stage 1).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=str, default=None, help="override train.out_dir")
    parser.add_argument("--resume", type=str, default=None, help="resume from checkpoint")
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

    # ---- data ----
    codes_root = resolve_local_uri(data_cfg["codes_root"])
    ctx = int(data_cfg.get("context_length", 256))
    pack = bool(data_cfg.get("pack", True))
    train_ds = MotionCodeDataset(
        codes_root, data_cfg.get("train_split", "train"),
        context_length=ctx, pack=pack, shuffle=True, seed=seed, rank=rank,
    )
    val_split = data_cfg.get("val_split", "val")
    val_ds = None
    if (Path(codes_root) / f"{val_split}.codes.npy").is_file():
        val_ds = MotionCodeDataset(
            codes_root, val_split, context_length=ctx, pack=pack, shuffle=False, seed=0,
        )

    # ---- model ----
    mcfg = MotionGPTConfig(
        num_codebooks=int(model_cfg.get("num_codebooks", train_ds.num_codebooks)),
        codebook_size=int(model_cfg.get("codebook_size", train_ds.codebook_size)),
        d_model=int(model_cfg.get("d_model", 768)),
        n_layers=int(model_cfg.get("n_layers", 12)),
        n_heads=int(model_cfg.get("n_heads", 12)),
        n_kv_heads=int(model_cfg.get("n_kv_heads", 0) or 0),
        ffn_hidden=int(model_cfg.get("ffn_hidden", 2048)),
        rope_theta=float(model_cfg.get("rope_theta", 1000000.0)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        max_seq_len=int(model_cfg.get("max_seq_len", max(4096, ctx))),
        context_length=ctx,
        loss_weights=model_cfg.get("loss_weights"),
        code_pred_hidden=int(model_cfg.get("code_pred_hidden", 1024)),
        code_pred_layers=int(model_cfg.get("code_pred_layers", 5)),
        code_pred_heads=int(model_cfg.get("code_pred_heads") or 8),
        code_pred_kv_heads=int(model_cfg.get("code_pred_kv_heads", 0) or 0),
        code_pred_ffn_hidden=int(model_cfg.get("code_pred_ffn_hidden") or 3072),
        code_axis_loss_scale=float(model_cfg.get("code_axis_loss_scale", 0.3)),
        code_axis_start_step=int(model_cfg.get("code_axis_start_step", 0)),
        code_axis_warmup_steps=int(model_cfg.get("code_axis_warmup_steps", 0)),
    )
    if mcfg.num_codebooks != train_ds.num_codebooks:
        log(rank, f"[warn] model codebook geometry != data ({train_ds.num_codebooks}x{train_ds.codebook_size})")

    model = MotionGPT(mcfg).to(device)
    log(rank, f"[model] params={model.num_parameters() / 1e6:.1f}M  ctx={ctx}  Q={mcfg.num_codebooks}")

    out_dir = resolve_local_uri(args.out_dir or train_cfg.get("out_dir", "runs/mgpt"))

    # ---- per-trainer callbacks ----
    def compute_loss(core, batch, dev, amp_dtype, *, step=0):
        codes = batch["motion_codes"].to(dev, non_blocking=True)
        valid = batch["valid_mask"].to(dev, non_blocking=True)
        seg = batch["segment_ids"].to(dev, non_blocking=True)
        pos = batch["positions"].to(dev, non_blocking=True)
        inp, tgt = codes[:, :-1], codes[:, 1:]
        tmask = core.next_packet_mask(valid, seg)

        # Staged training: q0-only before code_axis_start_step
        code_axis_scale = None
        start = core.cfg.code_axis_start_step
        warmup = core.cfg.code_axis_warmup_steps
        if start > 0 and step < start:
            code_axis_scale = 0.0
        elif warmup > 0 and step < start + warmup:
            code_axis_scale = ((step - start) / warmup) * core.cfg.code_axis_loss_scale

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=dev.type == "cuda"):
            logits = core.forward_packed(inp, tgt, seg[:, :-1], pos[:, :-1])
            return core.packet_ce_loss(core.cfg, logits, tgt, tmask, code_axis_scale=code_axis_scale)

    def compute_val_metrics(core, loader, dev, amp_dtype):
        return compute_token_metrics(core, loader, dev, amp_dtype, core.cfg.num_codebooks,
                                     max_batches=50, prefix="val/")

    def train_log_metrics(metrics, step, dt):
        bs = int(train_cfg.get("batch_size", 4))
        ctx_val = int(train_cfg.get("context_length", 2048))
        tps = (step + 1) * bs * world * ctx_val / max(dt, 1e-6)
        return {
            "train/tok_per_s": tps,
            **{f"train/ce_q{i}": float(metrics[f"ce_q{i}"].item()) for i in range(mcfg.num_codebooks)},
            **{f"train/acc_q{i}": float(metrics[f"acc_q{i}"].item()) for i in range(mcfg.num_codebooks)},
        }

    run_training(
        model=model, mcfg=mcfg, train_ds=train_ds, val_ds=val_ds,
        collate_fn=collate_motion_codes, compute_loss=compute_loss,
        compute_val_metrics=compute_val_metrics, train_log_metrics=train_log_metrics,
        optim_cfg=optim_cfg, train_cfg=train_cfg, swan_cfg=swan_cfg,
        out_dir=out_dir, payload=payload, resume_from=args.resume,
        rank=rank, world=world, local=local, is_ddp=is_ddp, device=device, seed=seed,
    )


if __name__ == "__main__":
    main()
