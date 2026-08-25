"""Stage 0 - export frozen motion codes for SeMoCo-Generator training.

Encodes each clip's ``features [T,499]`` UMR stream (read from the release's
derived UMR parquet shards) through the frozen tokenizer and writes a packed
code store:

    <out-dir>/<split>.codes.npy      # int16 [sum_T_tok, Q] (ragged clips, packed)
    <out-dir>/<split>.index.json     # [{rec_id, start, length}, ...]
    <out-dir>/<split>.meta.json      # token_rate / Q / codebook_size / stride / ckpt

The anchor needed to decode codes back to motion is NOT duplicated here; it is
read on demand from the same derived parquet at decode/visualization time
(see ``tokenizer_bridge.decode_codes_to_joints``).

Example::

    python -m semoco_generator.tools.export_motion_codes \
        --parquet-dir <release>/derived_umr_<hash> \
        --split train --out-dir local://codes_s4 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..dataset.umr_parquet import iter_parquet_rows
from ..local_uri import resolve_local_uri
from ..paths import default_checkpoint
from ..tokenizer_bridge import FrozenMotionTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frozen motion codes (Stage 0).")
    parser.add_argument("--checkpoint", type=str, default=None, help="tokenizer checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="train", help="split name (train/val/test)")
    parser.add_argument(
        "--parquet-dir", type=str, required=True,
        help="derived_umr_* dir containing {train,val,test}/*.parquet (local:// or path)",
    )
    parser.add_argument(
        "--out-dir", type=str, required=True,
        help="output dir for packed codes (local:// URI or path, "
             "e.g. local://codes_s4)",
    )
    parser.add_argument("--limit", type=int, default=0, help="cap number of clips (0 = all)")
    parser.add_argument("--min-tokens", type=int, default=2, help="skip clips shorter than this")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    parquet_dir = Path(resolve_local_uri(args.parquet_dir))
    out_dir = resolve_local_uri(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint) if args.checkpoint else default_checkpoint()
    print(f"[export] loading frozen tokenizer: {ckpt}")
    tok = FrozenMotionTokenizer.load(ckpt, device=args.device)
    spec = tok.spec
    print(
        f"[export] codec: Q={spec.num_codebooks} codebook={spec.codebook_size} "
        f"stride={spec.temporal_stride} token_rate={spec.token_rate:.2f}Hz"
    )

    buffers: list[np.ndarray] = []
    index: list[dict] = []
    cursor = 0
    skipped = 0
    seen = 0
    t0 = time.time()
    for row in iter_parquet_rows(parquet_dir, args.split, cols=["features"]):
        if args.limit and seen >= args.limit:
            break
        seen += 1
        features = np.asarray(row["features"], dtype=np.float32)
        codes = tok.encode(features)  # [T_tok, Q] int64
        if codes.shape[0] < args.min_tokens:
            skipped += 1
            continue
        codes16 = codes.astype(np.int16, copy=False)
        buffers.append(codes16)
        index.append({"rec_id": str(row["rec_id"]), "start": cursor, "length": int(codes16.shape[0])})
        cursor += int(codes16.shape[0])
        if seen % args.log_every == 0:
            rate = seen / max(1e-9, time.time() - t0)
            print(
                f"[export] {seen} clips seen | {len(index)} written | {cursor} tokens | "
                f"{rate:.1f} clips/s | skipped {skipped}"
            )

    if not buffers:
        raise RuntimeError("no clips encoded; check --parquet-dir / --split")

    packed = np.concatenate(buffers, axis=0)  # [sum_T_tok, Q]
    codes_path = out_dir / f"{args.split}.codes.npy"
    index_path = out_dir / f"{args.split}.index.json"
    meta_path = out_dir / f"{args.split}.meta.json"
    np.save(codes_path, packed)
    index_path.write_text(json.dumps(index))
    meta = {
        "split": args.split,
        "num_clips": len(index),
        "num_tokens": int(packed.shape[0]),
        "num_codebooks": spec.num_codebooks,
        "codebook_size": spec.codebook_size,
        "temporal_stride": spec.temporal_stride,
        "source_fps": spec.source_fps,
        "token_rate": spec.token_rate,
        "checkpoint": spec.checkpoint,
        "parquet_dir": str(parquet_dir),
        "skipped": skipped,
        "dtype": "int16",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[export] done: {len(index)} clips, {packed.shape[0]} tokens -> {codes_path} "
        f"(skipped {skipped})"
    )


if __name__ == "__main__":
    main()
