"""Text2Motion inference: prompt -> motion codes -> SOMA77 joints (renderable).

Loads a trained text2motion checkpoint, encodes prompts with the frozen Flan-T5
encoder, autoregressively samples motion packets (with CFG + EOS stop), decodes
them back to SOMA77 joints via the frozen tokenizer, and writes per-prompt
``.npz`` files compatible with ``eval/render_skeleton.py``.

Anchor: text-only generation has no ground-truth frame-0 pose, so the canonical
(rest-pose) anchor is used by default. Pass ``--anchor gt`` with ``--store``/
``--rows`` to reuse stored anchors (e.g. for reconstruction-style comparison).

Example::

    python -m semoco_generator.eval.t2m_infer \
        --checkpoint runs/t2m_150m_flan/model/best.pt \
        --prompts "a person walks forward" "a person jumps in place" \
        --out-dir runs/eval/infer/t2m_150m_flan --render mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..local_uri import resolve_local_uri
from ..paths import default_checkpoint
from ..text import get_encoder_cls
from ..tokenizer_bridge import FrozenMotionTokenizer, canonical_anchor, soma_skeleton_edges
from .rollout import SamplingConfig, generate_from_text, load_model


def _slug(text: str, i: int) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in text.lower())[:40].strip("_")
    return f"{i:03d}_{keep or 'clip'}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Text2Motion inference + decode to SOMA77 joints.")
    parser.add_argument("--checkpoint", type=str, required=True, help="trained t2m .pt")
    parser.add_argument("--prompts", type=str, nargs="+", default=None)
    parser.add_argument("--prompts-file", type=str, default=None, help="one prompt per line")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=str, default=None, help="frozen codec .pt")
    parser.add_argument("--text-encoder", type=str, default="flan",
                        choices=["flan", "siglip", "qwen3"], help="registered text encoder key")
    parser.add_argument("--text-encoder-model", type=str, default=None,
                        help="override model ID (defaults to built-in per key)")
    parser.add_argument("--text-max-length", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--max-tok", type=int, default=300)
    parser.add_argument("--eos-thresh", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fk-device", type=str, default="cpu")
    parser.add_argument("--render", type=str, default=None, choices=["gif", "mp4"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    prompts: list[str] = list(args.prompts or [])
    if args.prompts_file:
        prompts += [ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()]
    if not prompts:
        raise SystemExit("no prompts given (use --prompts or --prompts-file)")

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = resolve_local_uri(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[t2m] loading model: {args.checkpoint}", flush=True)
    model, ckpt = load_model(args.checkpoint, device=dev)
    if not model.cfg.use_text:
        raise SystemExit("checkpoint is not a text2motion model (use_text=False)")

    enc_cls = get_encoder_cls(args.text_encoder)
    model_id = args.text_encoder_model or enc_cls.DEFAULT_MODEL_ID
    print(f"[t2m] loading text encoder: {args.text_encoder} -> {model_id}", flush=True)
    text_enc = enc_cls.load(
        model_id, device=dev, max_length=args.text_max_length,
        local_files_only=args.local_files_only,
    )
    if text_enc.clip_dim != model.cfg.clip_dim:
        raise SystemExit(f"text clip_dim {text_enc.clip_dim} != model {model.cfg.clip_dim}")

    tok_ckpt = Path(args.tokenizer_checkpoint) if args.tokenizer_checkpoint else default_checkpoint()
    print(f"[t2m] loading frozen tokenizer: {tok_ckpt}", flush=True)
    tok = FrozenMotionTokenizer.load(tok_ckpt, device=dev)
    edges = soma_skeleton_edges()
    token_rate = tok.spec.token_rate
    joint_fps = tok.spec.source_fps  # decoded joints are at source fps (50)

    text_emb, text_mask = text_enc.encode(prompts)                       # [B, L, dim], [B, L]
    sampling = SamplingConfig(temperature=args.temperature, top_p=args.top_p, top_k=0)
    print(f"[t2m] generating {len(prompts)} clips (cfg={args.cfg_scale}, max_tok={args.max_tok})...", flush=True)
    seqs = generate_from_text(
        model, text_emb, text_mask, max_tok=args.max_tok, cfg_scale=args.cfg_scale,
        eos_thresh=args.eos_thresh, sampling=sampling, device=dev,
    )

    anchor = canonical_anchor(identity_dim=10)
    manifest = []
    for i, (prompt, codes) in enumerate(zip(prompts, seqs, strict=True)):
        n_tok = int(codes.shape[0])
        slug = _slug(prompt, i)
        if n_tok < 2:
            print(f"[t2m] {slug}: EOS immediately (0 frames), skipping decode", flush=True)
            manifest.append({"index": i, "prompt": prompt, "tokens": n_tok, "npz": None})
            continue
        codes_np = codes.cpu().numpy().astype(np.int64)
        out = tok.decode_to_joints_arrays(
            codes_np,
            init_root_pos=anchor["init_root_pos"],
            init_root_rot6d=anchor["init_root_rot6d"],
            init_joints76_rot6d=anchor["init_joints76_rot6d"],
            identity_coeffs=anchor["identity_coeffs"],
            device=args.fk_device,
        )
        npz_path = out_dir / f"{slug}.pred.npz"        # .pred.npz matches render_skeleton
        np.savez_compressed(
            npz_path,
            joints77=out["joints77"].astype(np.float32),
            edges=edges,
            fps=np.float32(joint_fps),
            prefix_frames=np.int64(0),
            caption=prompt,
        )
        dur = out["joints77"].shape[0] / joint_fps
        print(f"[t2m] {slug}: {n_tok} tok -> {out['joints77'].shape[0]} frames ({dur:.1f}s) -> {npz_path.name}", flush=True)
        manifest.append({"index": i, "prompt": prompt, "tokens": n_tok,
                         "frames": int(out["joints77"].shape[0]), "slug": slug, "npz": npz_path.name})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[t2m] wrote {len(manifest)} clips to {out_dir}", flush=True)

    if args.render:
        from .render_skeleton import render_clip
        render_dir = out_dir / "render"
        render_dir.mkdir(exist_ok=True)
        for m in manifest:
            if not m["npz"]:
                continue
            slug = m["slug"]
            try:
                p = render_clip(slug, out_dir, render_dir / f"{slug}.{args.render}", fmt=args.render)
                print(f"[t2m] rendered {p}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[t2m] render failed for {slug}: {exc}", flush=True)


if __name__ == "__main__":
    main()
