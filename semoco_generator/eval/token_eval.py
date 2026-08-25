"""Teacher-forced token-level evaluation over a code-store split.

Computes the definitive next-packet metrics (per-codebook CE / perplexity /
top-1 / top-5, plus aggregate ``ce_mean`` / ``ppl_mean``) over a whole split's
exported codes, and writes them to JSON. Use this for a clear, reproducible
quantitative reference on val / test (requires that split's codes exported via
``tools/export_motion_codes``).

Example::

    python -m semoco_generator.eval.token_eval \
        --checkpoint runs/mgpt_codear_150m/model/best.pt \
        --codes-root local://codes_s4 --split test \
        --out runs/mgpt_codear_150m/token_metrics_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..dataset import MotionCodeDataset, collate_motion_codes
from ..local_uri import resolve_local_uri
from .metrics import compute_token_metrics
from .rollout import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-forced token metrics over a split.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--codes-root", type=str, required=True, help="export dir (local:// ok)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0, help="0 = whole split")
    parser.add_argument("--out", type=str, default=None, help="metrics JSON (default runs alongside ckpt)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    model, _ = load_model(args.checkpoint, device=device)
    amp = torch.bfloat16 if device.type == "cuda" else torch.float32

    ds = MotionCodeDataset(
        args.codes_root, args.split, context_length=args.context_length,
        pack=True, shuffle=False,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_motion_codes, pin_memory=True,
    )
    print(f"[token_eval] split={args.split} clips={len(ds.index)} ctx={args.context_length}")

    metrics = compute_token_metrics(
        model, loader, device, amp, model.cfg.num_codebooks,
        max_batches=(args.max_batches or None), prefix=f"{args.split}/",
    )
    metrics["checkpoint"] = args.checkpoint
    metrics["split"] = args.split
    metrics["num_codebooks"] = model.cfg.num_codebooks

    out = Path(args.out) if args.out else (Path(args.checkpoint).resolve().parents[1] / f"token_metrics_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    key = f"{args.split}/"
    print(
        f"[token_eval] ce_mean={metrics[key + 'ce_mean']:.4f} ppl_mean={metrics[key + 'ppl_mean']:.2f} "
        f"top1_q0={metrics[key + 'top1_q0']:.3f}  -> {out}"
    )


if __name__ == "__main__":
    main()
