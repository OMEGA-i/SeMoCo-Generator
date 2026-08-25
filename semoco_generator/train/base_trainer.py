"""Shared training loop used by all SeMoCo-Generator trainers.

Each specific trainer (``train_motion_gpt.py``, ``train_t2m.py``) becomes a thin
configurator that builds the model, dataset, optimizer, and provides a
``compute_loss`` / ``compute_val_metrics`` callable.  This module encapsulates
the common loop, logging, checkpointing, and distributed setup.
"""

from __future__ import annotations

import json
import os
import time

# Reduce CUDA memory fragmentation — must be set before any torch.cuda usage.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# One-time cuDNN auto-tune per operator shape (first ~50 steps, ~3 min).
# After that, all subsequent steps use the fastest cached kernel.
torch.backends.cudnn.benchmark = True

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..local_uri import resolve_local_uri
from .utils import dist_info, init_swanlab, is_main, log, lr_at


def build_optimizer(
    model: nn.Module,
    optim_cfg: dict,
) -> torch.optim.Optimizer:
    """Build AdamW with weight-decay / no-decay parameter groups."""
    base_lr = float(optim_cfg.get("lr", 3e-4))
    wd = float(optim_cfg.get("weight_decay", 0.1))
    betas = tuple(optim_cfg.get("betas", [0.9, 0.95]))
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=base_lr, betas=betas,
    )


def save_checkpoint(
    *,
    core: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict,
    data_meta: dict,
    base_lr: float,
    out_dir: Path,
    name: str,
    step: int,
    rank: int,
    extra: dict | None = None,
) -> None:
    """Save a training checkpoint (rank-0 only)."""
    if not is_main(rank):
        return
    ckpt = {
        "model": core.state_dict(),
        "model_config": model_config,
        "data_meta": data_meta,
        "step": step,
        "optimizer": optimizer.state_dict(),
        "lr": base_lr,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, out_dir / "model" / name)


def build_dataloaders(
    train_ds,
    val_ds,
    *,
    batch_size: int,
    num_workers: int,
    collate_fn: Callable,
    rank: int,
    world: int,
    is_ddp: bool,
) -> tuple[DataLoader, DataLoader | None, DistributedSampler | None]:
    """Build train + val DataLoaders with optional DDP sampler."""
    train_sampler = None
    if is_ddp:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=True, drop_last=True,
            persistent_workers=num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=True, drop_last=True,
            persistent_workers=num_workers > 0,
        )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
        )
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_training(
    *,
    model: nn.Module,
    mcfg,
    train_ds,
    val_ds,
    collate_fn: Callable,
    compute_loss: Callable[[nn.Module, dict, torch.device, torch.dtype], tuple[torch.Tensor, dict]],
    compute_val_metrics: Callable[[nn.Module, DataLoader, torch.device, torch.dtype], dict[str, float]],
    train_log_metrics: Callable[[dict, int, float], dict] | None = None,
    optim_cfg: dict,
    train_cfg: dict,
    swan_cfg: dict,
    out_dir: Path,
    payload: dict,
    rank: int,
    world: int,
    local: int,
    is_ddp: bool,
    device: torch.device,
    seed: int,
    resume_from: str | None = None,
) -> None:
    """Run the full training loop.

    ``compute_loss(core, batch, device, amp_dtype) -> (loss, metrics)`` is the
    only per-trainer logic.  It is called inside ``autocast``.
    ``compute_val_metrics(core, loader, device, amp_dtype) -> dict`` provides
    validation metrics logged every ``val_every`` steps.
    ``train_log_metrics(metrics, step, dt) -> dict`` optionally returns extra
    swanlab keys (if None, only loss/lr are logged).
    """

    # ---- distributed ----
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 4))
    if is_ddp:
        model = DDP(model, device_ids=[local], output_device=local)
    core = model.module if is_ddp else model

    # ---- optimizer ----
    base_lr = float(optim_cfg.get("lr", 3e-4))
    optimizer = build_optimizer(core, optim_cfg)
    grad_clip = float(optim_cfg.get("grad_clip", 1.0))
    warmup_steps = int(optim_cfg.get("warmup_steps", 1000))
    max_steps = int(train_cfg.get("max_steps", 50000))
    amp_dtype = torch.bfloat16 if train_cfg.get("precision", "bf16") == "bf16" else torch.float16
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))

    # ---- resume ----
    start_step = 0
    best_val = float("inf")
    epoch = 0
    swanlab_run_id: str | None = None
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=True)
        core.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt.get("optimizer", {}))
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("val_metrics", {}).get("val/ce_mean", float("inf")))
        swanlab_run_id = ckpt.get("swanlab_run_id")
        log(rank, f"[resume] loaded step {start_step} from {resume_from}  best_val={best_val:.4f}")

    # ---- data loaders ----
    train_loader, val_loader, train_sampler = build_dataloaders(
        train_ds, val_ds, batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_fn, rank=rank, world=world, is_ddp=is_ddp,
    )

    # ---- output & logging ----
    (out_dir / "model").mkdir(parents=True, exist_ok=True)
    if resume_from is None:
        (out_dir / "config_resolved.json").write_text(
            json.dumps({"model": asdict(mcfg), "config": payload}, indent=2),
        )
    swan = init_swanlab(
        swan_cfg, name=out_dir.name,
        config={"model": asdict(mcfg), "data": train_ds.meta, "optim": optim_cfg,
                "train": train_cfg, "params_M": round(core.num_parameters() / 1e6, 1)},
        is_main_proc=is_main(rank),
        resume_id=swanlab_run_id,
        out_dir=str(out_dir),
    )
    if getattr(swan, "id", None):
        swanlab_run_id = swan.id
        log(rank, f"[swanlab] run id={swanlab_run_id}" + (f"  {swan.url}" if getattr(swan, "url", None) else ""))

    def _ckpt_extra(**kwargs) -> dict:
        extra = dict(kwargs)
        if swanlab_run_id:
            extra["swanlab_run_id"] = swanlab_run_id
        return extra

    log_every = int(train_cfg.get("log_every", 50))
    val_every = int(train_cfg.get("val_every", 1000))
    ckpt_every = int(train_cfg.get("ckpt_every", 5000))

    # ---- loop ----
    model.train()
    step = start_step
    t0 = time.time()
    done = False
    while not done:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            lr = lr_at(step, base_lr, warmup_steps, max_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            loss, metrics = compute_loss(core, batch, device, amp_dtype, step=step)
            (loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) >= max_steps:
                gn = 0.0
                if grad_clip > 0:
                    gn = torch.nn.utils.clip_grad_norm_(core.parameters(), grad_clip).item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # Store grad norm immediately so it's available at next log step
                metrics["grad_norm"] = gn

            if step % log_every == 0:
                dt = time.time() - t0
                log(rank, f"step {step}/{max_steps} loss {loss.item():.4f} lr {lr:.2e} gn {metrics.get('grad_norm', 0):.2f}")
                if is_main(rank):
                    extra = train_log_metrics(metrics, step, dt) if train_log_metrics else {}
                    swan.log({"train/loss": float(loss.item()), "train/lr": lr, **extra}, step=step)

            if val_loader is not None and step > 0 and step % val_every == 0:
                vm = compute_val_metrics(core, val_loader, device, amp_dtype)
                val_loss = vm.get("val/ce_mean", vm.get("val/loss", float("inf")))
                log(rank, f"[val] step {step}  best={best_val:.4f}  cur={val_loss:.4f}")
                if is_main(rank):
                    swan.log(vm, step=step)
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(
                            core=core, optimizer=optimizer,
                            model_config=asdict(mcfg), data_meta=train_ds.meta,
                            base_lr=base_lr, out_dir=out_dir, name="best.pt",
                            step=step, rank=rank,
                            extra=_ckpt_extra(val_metrics=vm),
                        )

            if step > 0 and step % ckpt_every == 0:
                save_checkpoint(
                    core=core, optimizer=optimizer, model_config=asdict(mcfg),
                    data_meta=train_ds.meta, base_lr=base_lr, out_dir=out_dir,
                    name="latest.pt", step=step, rank=rank,
                    extra=_ckpt_extra(),
                )

            step += 1
            if step >= max_steps:
                done = True
                break
        epoch += 1

    save_checkpoint(
        core=core, optimizer=optimizer, model_config=asdict(mcfg),
        data_meta=train_ds.meta, base_lr=base_lr, out_dir=out_dir,
        name="latest.pt", step=step, rank=rank,
        extra=_ckpt_extra(),
    )
    swan.finish()
    log(rank, f"[done] trained {step} steps; best val {best_val:.4f}")
    if is_ddp:
        dist.destroy_process_group()
