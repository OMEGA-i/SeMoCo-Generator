"""Stage 1 (text2motion) - train the text-conditioned packet LM.

Loads the paired {motion codes, Flan-T5 word embeddings} store exported by
``tools/export_t2m_dataset.py`` and trains the generator: a text
prefix (bidirectional) + causal motion packets (mixed attention), teacher-forced
next-packet cross entropy on the motion span only, plus a binary EOS head and
classifier-free-guidance text dropout.

Examples::

    # single GPU
    python -m semoco_generator.train.train_t2m --config configs/t2m_150m_flan.yaml

    # DDP
    torchrun --nproc_per_node=8 -m semoco_generator.train.train_t2m \
        --config configs/t2m_150m_flan.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from ..dataset import T2MCodeDataset, collate_t2m
from ..eval.t2m_metrics import eval_t2m, t2m_losses
from ..local_uri import resolve_local_uri
from ..model import MotionGPT, MotionGPTConfig
from .base_trainer import run_training
from .utils import dist_info, log


def _load_pretrained(model: MotionGPT, ckpt_path: str, rank: int) -> None:
    """Load motion-pretrained weights into a T2M model.

    Only shared modules (embed, blocks, norm, decoder) are loaded.
    T2M-specific modules (text_proj, null_text, motion_bos, eos_head)
    keep their random initialization.
    """
    from .utils import log as _log
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pretrained_state = ckpt.get("model", ckpt)
    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for name, param in pretrained_state.items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name].copy_(param)
            loaded += 1
        else:
            skipped += 1
    missing = sum(1 for n in model_state if n not in pretrained_state)
    _log(rank, f"[pretrained] loaded {loaded} params, skipped {skipped}, "
              f"T2M-only random init: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train text2motion SeMoCo-Generator (Stage 1).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=str, default=None, help="override train.out_dir")
    parser.add_argument("--resume", type=str, default=None, help="resume from checkpoint")
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text())
    data_cfg = payload.get("data", {})
    model_cfg = payload.get("model", {})
    text_cfg = payload.get("text", {})
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
    text_enc_key = text_cfg.get("encode", None)
    max_motion_tok = int(data_cfg.get("max_motion_tok", 300))
    train_ds = T2MCodeDataset(
        codes_root, data_cfg.get("train_split", "train"),
        text_encoder_key=text_enc_key, max_motion_tok=max_motion_tok,
    )
    val_split = data_cfg.get("val_split", "val")
    val_ds = None
    if (Path(codes_root) / f"{val_split}.codes.npy").is_file():
        val_ds = T2MCodeDataset(
            codes_root, val_split, text_encoder_key=text_enc_key,
            max_motion_tok=max_motion_tok,
        )

    # ---- model ----
    ctx = int(text_cfg.get("text_max_length", 64)) + max_motion_tok + 8
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
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
        use_text=True,
        clip_dim=int(model_cfg.get("clip_dim", train_ds.clip_dim)),
        text_cond_prob=float(text_cfg.get("cfg_prob", 0.1)),
        eos_loss_scale=float(model_cfg.get("eos_loss_scale", 1.0)),
    )
    if mcfg.clip_dim != train_ds.clip_dim:
        raise ValueError(f"model clip_dim={mcfg.clip_dim} != store clip_dim={train_ds.clip_dim}")

    pretrained = model_cfg.get("pretrained")
    model = MotionGPT(mcfg).to(device)
    if pretrained:
        _load_pretrained(model, pretrained, rank)
    log(rank, f"[model] params={model.num_parameters() / 1e6:.1f}M  Q={mcfg.num_codebooks}  "
              f"clip_dim={mcfg.clip_dim}  max_motion_tok={max_motion_tok}"
              + (f"  pretrained={pretrained}" if pretrained else ""))

    out_dir = resolve_local_uri(args.out_dir or train_cfg.get("out_dir", "runs/t2m"))
    cfg_drop = mcfg.text_cond_prob

    # ---- per-trainer callbacks ----
    def compute_loss(core, batch, dev, amp_dtype, *, step=0):
        return t2m_losses(core, batch, dev, amp_dtype, cfg_drop=cfg_drop, global_step=step)

    def compute_val_metrics(core, loader, dev, amp_dtype):
        return eval_t2m(core, loader, dev, amp_dtype, max_batches=50)

    def train_log_metrics(metrics, step, dt):
        return {
            "train/loss_eos": float(metrics["loss_eos"].item()),
            "train/acc_eos": float(metrics["acc_eos"].item()),
            **{f"train/ce_q{i}": float(metrics[f"ce_q{i}"].item()) for i in range(mcfg.num_codebooks)},
            **{f"train/acc_q{i}": float(metrics[f"acc_q{i}"].item()) for i in range(mcfg.num_codebooks)},
        }

    run_training(
        model=model, mcfg=mcfg, train_ds=train_ds, val_ds=val_ds,
        collate_fn=collate_t2m, compute_loss=compute_loss,
        compute_val_metrics=compute_val_metrics, train_log_metrics=train_log_metrics,
        optim_cfg=optim_cfg, train_cfg=train_cfg, swan_cfg=swan_cfg,
        out_dir=out_dir, payload=payload, resume_from=args.resume,
        rank=rank, world=world, local=local, is_ddp=is_ddp, device=device, seed=seed,
    )


if __name__ == "__main__":
    main()
