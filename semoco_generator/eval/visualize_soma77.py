"""Stage 2 demo - 2s prefix -> 8s prediction, decoded to SOMA77 joints.

Encodes a recording with the frozen tokenizer, takes the first ``--prefix-tokens``
as the conditioning prefix, rolls out ``--rollout-tokens`` packets with
SeMoCo-Generator, decodes the full sequence back to SOMA77 FK joints, and writes a
viewer-friendly ``.npz`` (``joints77``, ``root``, ``fps``, ``prefix_frames``,
``edges``) plus a ground-truth decode for overlay.

The per-clip routine (:func:`generate_and_decode` + :func:`save_demo`) is reused
by :mod:`semoco_generator.eval.rollout_eval` for batch test-split evaluation.

Example::

    python -m semoco_generator.eval.visualize_soma77 \
        --checkpoint runs/mgpt_codear_150m/model/best.pt \
        --parquet-dir <derived_umr_dir> --split test \
        --rec-id rec_01KRWDM8WV0GJ6GMFGFJ5PXGDV \
        --prefix-tokens 25 --rollout-tokens 100 \
        --out-dir runs/mgpt_codear_150m/demo --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..dataset.umr_parquet import load_rows_by_rec_id
from ..local_uri import resolve_local_uri
from ..paths import default_checkpoint
from ..tokenizer_bridge import FrozenMotionTokenizer, soma_skeleton_edges
from ..tokenizer_bridge import decode_codes_to_joints as codes_to_joints
from .rollout import SamplingConfig, default_motion_sampling, load_model, rollout


def encode_clip(tok: FrozenMotionTokenizer, clip: dict) -> np.ndarray:
    """Encode a derived-parquet clip's ``features [T,499]`` -> codes ``[T_tok, Q]``."""
    return tok.encode(np.asarray(clip["features"], dtype=np.float32))


def generate_and_decode(
    model,
    tok: FrozenMotionTokenizer,
    clip: dict,
    *,
    prefix_tokens: int,
    rollout_tokens: int,
    sampling: SamplingConfig,
    device: str,
    all_codes: np.ndarray | None = None,
) -> dict:
    """Encode -> rollout -> decode one clip. Returns pred/gt joints + codes.

    ``clip`` is a derived-parquet row (features + frame-0 anchor). ``all_codes``
    (the full clip's encoded codes) may be passed in to skip re-encoding.
    """
    rec_id = str(clip["rec_id"])
    if all_codes is None:
        all_codes = encode_clip(tok, clip)  # [T_tok, Q]
    if all_codes.shape[0] < prefix_tokens + 2:
        raise ValueError(f"recording {rec_id} too short: {all_codes.shape[0]} tokens")
    prefix = all_codes[:prefix_tokens]

    prefix_t = torch.from_numpy(prefix.astype(np.int64))
    gen = rollout(model, prefix_t, rollout_tokens, sampling=sampling, device=device)
    gen_codes = gen.squeeze(0).cpu().numpy()  # [prefix + rollout, Q]

    pred = codes_to_joints(tok, gen_codes, clip, device=device)
    gt_len = min(all_codes.shape[0], gen_codes.shape[0])
    gt = codes_to_joints(tok, all_codes[:gt_len], clip, device=device)
    return {
        "rec_id": rec_id,
        "pred": pred,
        "gt": gt,
        "gen_codes": gen_codes,
        "prefix_frames": int(prefix_tokens * tok.stride),
        "fps": float(tok.spec.source_fps),
        "total_tokens": int(all_codes.shape[0]),
    }


def save_demo(out_dir: Path, result: dict, edges: np.ndarray, *, meta_extra: dict | None = None) -> dict:
    """Persist pred/gt viewer npz + generated codes + meta for one recording."""
    rec_id = result["rec_id"]
    pred, gt = result["pred"], result["gt"]
    fps, prefix_frames = result["fps"], result["prefix_frames"]
    np.savez_compressed(
        out_dir / f"{rec_id}.pred.npz",
        joints77=pred["joints77"], root=pred["root"], edges=edges,
        fps=np.float32(fps), prefix_frames=np.int32(prefix_frames),
    )
    np.savez_compressed(
        out_dir / f"{rec_id}.gt.npz",
        joints77=gt["joints77"], root=gt["root"], edges=edges, fps=np.float32(fps),
    )
    np.save(out_dir / f"{rec_id}.codes.npy", result["gen_codes"].astype(np.int16))
    meta = {
        "rec_id": rec_id,
        "prefix_frames": prefix_frames,
        "fps": fps,
        "pred_frames": int(pred["joints77"].shape[0]),
        "total_tokens": result["total_tokens"],
    }
    if meta_extra:
        meta.update(meta_extra)
    (out_dir / f"{rec_id}.meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def build_sampling(q: int, temperature: float | None, top_p: float, top_k: int = 0) -> SamplingConfig:
    if temperature is not None:
        return SamplingConfig(temperature=temperature, top_p=top_p, top_k=top_k)
    sampling = default_motion_sampling(q)
    sampling.top_k = top_k
    return sampling


def main() -> None:
    parser = argparse.ArgumentParser(description="SeMoCo-Generator rollout + SOMA77 viz (Stage 2).")
    parser.add_argument("--checkpoint", type=str, required=True, help="trained MotionGPT checkpoint")
    parser.add_argument("--tokenizer-checkpoint", type=str, default=None, help="frozen codec checkpoint")
    parser.add_argument("--rec-id", type=str, required=True, help="recording id for prefix + anchor")
    parser.add_argument("--parquet-dir", type=str, required=True,
                        help="derived_umr_* dir (features + anchor source); local:// or path")
    parser.add_argument("--split", type=str, default="test", help="split holding --rec-id")
    parser.add_argument("--prefix-tokens", type=int, default=25, help="prefix length (25 ~= 2s @12.5Hz)")
    parser.add_argument("--rollout-tokens", type=int, default=100, help="generated length (100 ~= 8s)")
    parser.add_argument("--temperature", type=float, default=None, help="uniform temperature override")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0, help="uniform top-k override (0 = disabled)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, required=True, help="local:// URI or path")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    parquet_dir = resolve_local_uri(args.parquet_dir)
    out_dir = resolve_local_uri(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = load_rows_by_rec_id(parquet_dir, args.split, [args.rec_id])
    if args.rec_id not in clips:
        raise SystemExit(f"rec_id {args.rec_id!r} not found in {parquet_dir}/{args.split}")

    tok = FrozenMotionTokenizer.load(args.tokenizer_checkpoint or default_checkpoint(), device=args.device)
    model, _ = load_model(args.checkpoint, device=args.device)
    sampling = build_sampling(model.cfg.num_codebooks, args.temperature, args.top_p, args.top_k)
    edges = soma_skeleton_edges()

    result = generate_and_decode(
        model, tok, clips[args.rec_id],
        prefix_tokens=args.prefix_tokens, rollout_tokens=args.rollout_tokens,
        sampling=sampling, device=args.device,
    )
    save_demo(out_dir, result, edges, meta_extra={
        "checkpoint": args.checkpoint, "tokenizer_checkpoint": tok.spec.checkpoint,
        "token_rate": tok.spec.token_rate,
    })
    print(f"[viz] wrote {out_dir / args.rec_id}.pred.npz ({result['pred']['joints77'].shape[0]} frames)")


if __name__ == "__main__":
    main()
