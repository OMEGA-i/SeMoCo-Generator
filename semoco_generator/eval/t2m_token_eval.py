"""Teacher-forced token metrics for a trained text2motion model (lightweight eval).

Reports per-codebook cross-entropy, perplexity, top-1 accuracy, and EOS accuracy
on a split of the paired store. This is the fast quantitative check; TMR-based
R-precision/FID lives in the ``soma_tmr`` evaluation track
(``python -m semoco_generator.eval.cli run --track soma_tmr``).

Example::

    python -m semoco_generator.eval.t2m_token_eval \\
        --checkpoint runs/t2m_150m_flan/model/best.pt \\
        --codes-root local://t2m_codes \\
        --split test --text-encoder flan --max-batches 20 --out runs/eval/token_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..dataset import T2MCodeDataset, collate_t2m
from ..eval.rollout import load_model
from .t2m_metrics import eval_t2m


def main() -> None:
    parser = argparse.ArgumentParser(description="Text2Motion teacher-forced token metrics.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--codes-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--max-motion-tok", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--text-encoder",
        choices=["flan", "siglip", "qwen3"],
        default=None,
        help="Paired-store frozen text embedding encoder. Default: from ckpt data_meta.",
    )
    parser.add_argument("--out", type=str, default=None, help="Optional JSON artifact path.")
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device=dev)
    if not model.cfg.use_text:
        raise SystemExit("checkpoint is not a text2motion model")

    text_enc = (
        args.text_encoder
        or (ckpt.get("data_meta") or {}).get("encode_key")
        or ckpt.get("text_encoder_key")
        or "flan"
    )
    ds = T2MCodeDataset(
        args.codes_root,
        args.split,
        text_encoder_key=text_enc,
        max_motion_tok=args.max_motion_tok,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_t2m,
        pin_memory=True,
    )
    amp_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

    vm = eval_t2m(model, loader, dev, amp_dtype, max_batches=args.max_batches)
    report = {k: round(v, 5) for k, v in vm.items()}
    report["ppl_mean"] = round(math.exp(vm["val/ce_mean"]), 3)
    report["text_encoder"] = text_enc
    report["n_clips"] = len(ds)
    report["checkpoint"] = str(args.checkpoint)
    report["codes_root"] = str(args.codes_root)
    report["split"] = args.split
    report["max_batches"] = args.max_batches
    print(json.dumps(report, indent=2))
    print(
        f"\n[t2m-token-eval] split={args.split} clips={len(ds)} text_encoder={text_enc} "
        f"ce_mean={vm['val/ce_mean']:.4f} ppl={report['ppl_mean']} "
        f"top1_q0={vm['val/acc_q0']:.3f} eos_acc={vm['val/acc_eos']:.3f}"
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
