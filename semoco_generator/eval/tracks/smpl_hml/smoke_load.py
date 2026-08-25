"""Smoke / readiness check for HumanML3D ``text_mot_match`` evaluator load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .hml_evaluator import TextMotMatchEvaluator
from .protocol import DEFAULT_CHECKPOINT, DEFAULT_MEAN_STD
from .conversion import joints22_to_hml263
from .word_vectorizer import load_word_vectorizer, resolve_glove_root


def main() -> None:
    p = argparse.ArgumentParser(description="Verify text_mot_match evaluator loads and encodes motion/text")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--mean-std-dir", default=str(DEFAULT_MEAN_STD))
    p.add_argument("--glove-root", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    mean_p = Path(args.mean_std_dir) / "mean.npy"
    std_p = Path(args.mean_std_dir) / "std.npy"
    glove = resolve_glove_root([args.glove_root] if args.glove_root else None)
    report = {
        "checkpoint": str(ckpt),
        "checkpoint_exists": ckpt.is_file(),
        "mean_exists": mean_p.is_file(),
        "std_exists": std_p.is_file(),
        "glove_root": str(glove) if glove else None,
        "glove_ready": glove is not None,
        "device": args.device,
    }
    if not ckpt.is_file():
        raise SystemExit(f"missing evaluator checkpoint: {ckpt} (unzip t2m.zip under HumanML3D ckpt root)")
    mean = np.load(mean_p).astype(np.float32) if mean_p.is_file() else None
    std = np.load(std_p).astype(np.float32) if std_p.is_file() else None

    wv = None
    if glove is not None:
        wv = load_word_vectorizer(glove)
        report["word_vectorizer_size"] = len(wv)

    ev = TextMotMatchEvaluator(ckpt, device=args.device, mean=mean, std=std, word_vectorizer=wv)
    report["loaded"] = True
    report["movement_params"] = sum(x.numel() for x in ev.movement.parameters())
    report["motion_params"] = sum(x.numel() for x in ev.motion.parameters())
    report["text_params"] = sum(x.numel() for x in ev.text.parameters())

    rng = np.random.default_rng(0)
    feats = []
    for _ in range(4):
        j = rng.normal(size=(48, 22, 3)).astype(np.float32)
        j[..., 1] = np.abs(j[..., 1])
        feats.append(joints22_to_hml263(j, 20.0))
    emb = ev.encode_motion(feats)
    report["motion_emb_shape"] = list(emb.shape)

    report["text_encode_ready"] = ev.word_vectorizer is not None
    if ev.word_vectorizer is not None:
        text_emb = ev.encode_text(["a person walks forward", "someone jumps up and down"])
        report["text_emb_shape"] = list(text_emb.shape)
        report["note"] = "Motion + text encode OK."
    else:
        report["note"] = (
            "Motion path OK. Text path needs HumanML WordVectorizer/GloVe "
            "(<data-root>/glove/our_vab_*)."
        )
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    if not report["text_encode_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
