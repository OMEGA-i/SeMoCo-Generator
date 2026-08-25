"""Stage 0 (text2motion) — export motion-code + text-embedding store.

Shared (encoder-agnostic) files:
    <out>/<split>.codes.npy     int16   [sum_T_tok, Q]
    <out>/<split>.anchor.npy    float32 [N, 465]
    <out>/<split>.identity.npy  float32 [N, C]
    <out>/<split>.index.json    [{rec_id, code_start, code_len, row, caption}]
    <out>/<split>.meta.json     codec metadata

Per-encoder files (key = flan|siglip|qwen3):
    <out>/<split>.text_emb.<key>.npy       float16 [sum_L, clip_dim]
    <out>/<split>.text_index.<key>.json    [{text_start, text_len}]  (aligned by row with index.json)
    <out>/<split>.meta.<key>.json          {clip_dim, encode_key, ...}

Examples::

    # Text-only first (all three encoders, no motion encoding):
    python -m semoco_generator.tools.export_t2m_dataset \\
        --parquet-dir <release>/derived_umr --split train \\
        --out-dir local://t2m_codes_s4 --text-encoder flan --text-only
    python -m semoco_generator.tools.export_t2m_dataset \\
        --split train --out-dir local://t2m_codes_s4 \\
        --text-encoder siglip --text-only
    python -m semoco_generator.tools.export_t2m_dataset \\
        --split train --out-dir local://t2m_codes_s4 \\
        --text-encoder qwen3 --text-only

    # Full export (motion codes + text, reads parquet once):
    python -m semoco_generator.tools.export_t2m_dataset \\
        --parquet-dir <release>/derived_umr --split train \\
        --out-dir local://t2m_codes_s4 --text-encoder flan --limit 64
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..local_uri import resolve_local_uri
from ..paths import default_checkpoint
from ..tokenizer_bridge import FrozenMotionTokenizer

_ANCHOR_DIM = 3 + 6 + 456  # init_root_pos + init_root_rot6d + init_joints76_rot6d


# Columns needed for full export (motion + text).
_FULL_COLS = [
    "rec_id", "num_records", "features", "text",
    "init_root_pos", "init_root_rot6d", "init_joints76_rot6d", "identity_coeffs",
]
# Text-only: skip heavy motion columns (~1.8 GB/split → ~50 MB/split of captions).
_TEXT_ONLY_COLS = ["rec_id", "text"]


def _list_col_np(table, name: str) -> tuple[np.ndarray, np.ndarray]:
    """A ``list<float>`` column -> (flat values ``float32``, int offsets)."""
    arr = table.column(name).combine_chunks()
    return (
        arr.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False),
        arr.offsets.to_numpy(),
    )


def _iter_parquet_rows(parquet_dir: Path, split: str, *, text_only: bool = False):
    """Yield per-clip dicts from all shards of ``split``, in file order.

    When ``text_only``, only reads ``rec_id`` + ``text`` (skips heavy motion
    columns — ~100× less I/O).
    """
    import pyarrow.parquet as pq

    cols = _TEXT_ONLY_COLS if text_only else _FULL_COLS
    shard_dir = parquet_dir / split
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {shard_dir}")
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=cols)
            rec_ids = table.column("rec_id").to_pylist()
            texts = table.column("text").to_pylist()
            if text_only:
                for i in range(len(rec_ids)):
                    yield {"rec_id": rec_ids[i], "text": texts[i]}
            else:
                nrecs = table.column("num_records").to_numpy()
                fv, fo = _list_col_np(table, "features")
                rp_v, rp_o = _list_col_np(table, "init_root_pos")
                rr_v, rr_o = _list_col_np(table, "init_root_rot6d")
                jr_v, jr_o = _list_col_np(table, "init_joints76_rot6d")
                id_v, id_o = _list_col_np(table, "identity_coeffs")
                for i in range(len(rec_ids)):
                    n = int(nrecs[i])
                    yield {
                        "rec_id": rec_ids[i],
                        "text": texts[i],
                        "features": fv[fo[i] : fo[i + 1]].reshape(n, 499),
                        "init_root_pos": rp_v[rp_o[i] : rp_o[i + 1]],
                        "init_root_rot6d": rr_v[rr_o[i] : rr_o[i + 1]],
                        "init_joints76_rot6d": jr_v[jr_o[i] : jr_o[i + 1]],
                        "identity_coeffs": id_v[id_o[i] : id_o[i + 1]],
                    }


def _encode_motion(parquet_dir: Path, args, tok, stride: int, *, out_dir: Path | None = None):
    """Encode motion for all clips; return (code_buf, anchor_buf, identity_buf, shared_index).

    When ``out_dir`` is provided, writes incrementally to temp flat files at each
    flush to avoid OOM on large datasets, then assembles final .npy at the end.
    """
    from collections import defaultdict

    code_buf: list[np.ndarray] = [] if out_dir is None else None
    anchor_buf: list[np.ndarray] = [] if out_dir is None else None
    identity_buf: list[np.ndarray] = [] if out_dir is None else None
    shared_index: list[dict] = []
    code_cursor = 0
    skipped = 0
    t0 = time.time()

    # Incremental disk state
    _tmp_code_fp = None; _tmp_anchor_fp = None; _tmp_id_fp = None
    if out_dir is not None:
        split = args.split
        _tmp_code_fp = out_dir / f".{split}.codes.tmp"
        _tmp_anchor_fp = out_dir / f".{split}.anchor.tmp"
        _tmp_id_fp = out_dir / f".{split}.identity.tmp"
        for fp in [_tmp_code_fp, _tmp_anchor_fp, _tmp_id_fp]:
            fp.unlink(missing_ok=True)

    pend_feats: list[np.ndarray] = []
    pend_text: list[str] = []
    pend_row: list[dict] = []

    def flush_batch() -> None:
        nonlocal code_cursor, skipped
        N = len(pend_feats)
        if N == 0:
            return
        codes_by_j: list[np.ndarray | None] = [None] * N
        buckets: dict[int, list[int]] = defaultdict(list)
        for j, f in enumerate(pend_feats):
            buckets[(int(f.shape[0]) // stride) * stride].append(j)
        for keep, js in buckets.items():
            if keep <= 0:
                continue
            for cs in range(0, len(js), args.codec_batch):
                sub = js[cs : cs + args.codec_batch]
                for j, c in zip(sub, tok.encode_batch([pend_feats[j] for j in sub]), strict=True):
                    codes_by_j[j] = c
        for j, row in enumerate(pend_row):
            codes = codes_by_j[j]
            if codes is None or codes.shape[0] < args.min_tokens:
                skipped += 1
                continue
            n_tok = int(codes.shape[0])
            c16 = codes.astype(np.int16, copy=False)
            av = _anchor_vector(row)
            iv = np.asarray(row["identity_coeffs"], dtype=np.float32).reshape(-1)
            if out_dir is not None:
                with open(_tmp_code_fp, "ab") as fc, open(_tmp_anchor_fp, "ab") as fa, open(_tmp_id_fp, "ab") as fi:
                    c16.tofile(fc); av.tofile(fa); iv.tofile(fi)
            else:
                code_buf.append(c16)
                anchor_buf.append(av)
                identity_buf.append(iv)
            shared_index.append({
                "rec_id": str(row["rec_id"]),
                "code_start": code_cursor, "code_len": n_tok,
                "row": len(shared_index),
                "caption": pend_text[j],
            })
            code_cursor += n_tok
        pend_feats.clear(); pend_text.clear(); pend_row.clear()

    seen = 0
    for row in _iter_parquet_rows(parquet_dir, args.split):
        if args.limit and seen >= args.limit:
            break
        seen += 1
        text = row.get("text") or ""
        if not text:
            skipped += 1
            continue
        pend_feats.append(row["features"]); pend_text.append(text); pend_row.append(row)
        if len(pend_feats) >= args.flush_size:
            flush_batch()
        if seen % args.log_every == 0:
            rate = seen / max(1e-9, time.time() - t0)
            print(f"[export-t2m] {seen} clips seen | {len(shared_index)} written | "
                  f"{code_cursor} tok | {rate:.1f} clips/s | skipped {skipped}", flush=True)
    flush_batch()

    if out_dir is not None:
        # Convert flat temp files to proper .npy files and write index/meta
        Q = int(tok.spec.num_codebooks)
        N = len(shared_index)
        codes = np.fromfile(_tmp_code_fp, dtype=np.int16).reshape(-1, Q)
        anchors = np.fromfile(_tmp_anchor_fp, dtype=np.float32).reshape(N, _ANCHOR_DIM)
        id_data = np.fromfile(_tmp_id_fp, dtype=np.float32)
        id_dim = id_data.size // N if N > 0 else 0
        identities = id_data.reshape(N, id_dim) if id_dim > 0 else id_data.reshape(N, -1)
        ckpt = Path(args.checkpoint) if args.checkpoint else default_checkpoint()
        # Write directly (match _write_shared_files output without the extra stack dim)
        np.save(out_dir / f"{args.split}.codes.npy", codes)
        np.save(out_dir / f"{args.split}.anchor.npy", anchors)
        np.save(out_dir / f"{args.split}.identity.npy", identities)
        index_out = []
        for e in shared_index:
            index_out.append({"rec_id": e["rec_id"], "code_start": e["code_start"],
                              "code_len": e["code_len"], "row": e["row"], "caption": e.get("caption", "")})
        (out_dir / f"{args.split}.index.json").write_text(json.dumps(index_out))
        shared_meta = {
            "split": args.split, "num_clips": N, "num_tokens": int(codes.shape[0]),
            "num_codebooks": Q, "codebook_size": int(tok.spec.codebook_size),
            "temporal_stride": int(tok.spec.temporal_stride),
            "token_rate": float(tok.spec.token_rate), "source_fps": float(tok.spec.source_fps),
            "anchor_dim": _ANCHOR_DIM, "identity_dim": int(identities.shape[1]),
            "tokenizer_checkpoint": str(ckpt),
        }
        (out_dir / f"{args.split}.meta.json").write_text(json.dumps(shared_meta, indent=2))
        for fp in [_tmp_code_fp, _tmp_anchor_fp, _tmp_id_fp]:
            fp.unlink(missing_ok=True)
        return None, None, None, shared_index

    if not code_buf:
        raise RuntimeError("no clips encoded; check --parquet-dir / --split / text column")
    return code_buf, anchor_buf, identity_buf, shared_index


def _write_shared_files(out_dir: Path, split: str, code_buf, anchor_buf, identity_buf,
                        shared_index, spec, ckpt: Path):
    """Write shared store files (codes, anchor, identity, index, meta)."""
    codes_packed = np.concatenate(code_buf, axis=0)
    anchors = np.stack(anchor_buf, axis=0)
    identities = np.stack(identity_buf, axis=0)

    np.save(out_dir / f"{split}.codes.npy", codes_packed)
    np.save(out_dir / f"{split}.anchor.npy", anchors)
    np.save(out_dir / f"{split}.identity.npy", identities)

    shared_index_out = []
    for e in shared_index:
        shared_index_out.append({
            "rec_id": e["rec_id"],
            "code_start": e["code_start"],
            "code_len": e["code_len"],
            "row": e["row"],
            "caption": e.get("caption", ""),
        })
    (out_dir / f"{split}.index.json").write_text(json.dumps(shared_index_out))

    shared_meta = {
        "split": split,
        "num_clips": len(shared_index),
        "num_tokens": int(codes_packed.shape[0]),
        "num_codebooks": int(spec.num_codebooks),
        "codebook_size": int(spec.codebook_size),
        "temporal_stride": int(spec.temporal_stride),
        "token_rate": float(spec.token_rate),
        "source_fps": float(spec.source_fps),
        "anchor_dim": _ANCHOR_DIM,
        "identity_dim": int(identities.shape[1]),
        "tokenizer_checkpoint": str(ckpt),
    }
    (out_dir / f"{split}.meta.json").write_text(json.dumps(shared_meta, indent=2))


def _anchor_vector(row: dict) -> np.ndarray:
    v = np.zeros((_ANCHOR_DIM,), dtype=np.float32)
    v[0:3] = np.asarray(row["init_root_pos"], dtype=np.float32)
    v[3:9] = np.asarray(row["init_root_rot6d"], dtype=np.float32)
    v[9:465] = np.asarray(row["init_joints76_rot6d"], dtype=np.float32).reshape(-1)
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paired motion-code + text-embedding store (T2M Stage 0).")
    parser.add_argument("--parquet-dir", type=str, default=None,
                        help="derived_umr_* dir  (required for full export; optional with --text-only)")
    parser.add_argument("--split", type=str, nargs="*", default=None,
                        help="splits to process (default: train val test)")
    parser.add_argument("--out-dir", type=str, required=True, help="output store dir (local:// or path)")
    parser.add_argument("--checkpoint", type=str, default=None, help="frozen tokenizer .pt  (full export only)")
    parser.add_argument("--text-encoder", type=str, default="flan",
                        choices=["flan", "siglip", "qwen3"], help="registered text encoder key")
    parser.add_argument("--text-encoder-model", type=str, default=None,
                        help="override model ID (defaults to built-in per key)")
    parser.add_argument("--text-max-length", type=int, default=64)
    parser.add_argument("--flush-size", type=int, default=2048, help="clips buffered per flush  (full export)")
    parser.add_argument("--text-subbatch", type=int, default=128, help="captions per encoder forward pass")
    parser.add_argument("--codec-batch", type=int, default=256, help="max clips per batched codec encode  (full export)")
    parser.add_argument("--limit", type=int, default=0, help="cap number of clips (0 = all)")
    parser.add_argument("--min-tokens", type=int, default=2, help="skip clips shorter than this  (full export)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--source-fps", type=float, default=50.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--local-files-only", action="store_true", help="offline load (HF cache / local path)")
    parser.add_argument("--text-only", action="store_true",
                        help="skip motion encoding: encode text only")
    parser.add_argument("--codes-only", action="store_true",
                        help="skip text encoding: encode motion codes only (updates shared files, "
                             "preserves existing text_emb.*)")
    args = parser.parse_args()

    out_dir = resolve_local_uri(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = args.split or ["train", "val", "test"]

    for split in splits:
        args.split = split
        _export_split(args, out_dir)
    return


def _export_split(args: argparse.Namespace, out_dir: Path) -> None:
    """Export a single split."""
    print(f"\n{'='*60}\n[export-t2m] split={args.split}\n{'='*60}", flush=True)

    # ---- codes-only: encode motion only, skip text encoder entirely ----
    if args.codes_only:
        if args.parquet_dir is None:
            raise SystemExit("--parquet-dir is required for --codes-only")
        ckpt = Path(args.checkpoint) if args.checkpoint else default_checkpoint()
        print(f"[export-t2m] frozen tokenizer: {ckpt}", flush=True)
        tok = FrozenMotionTokenizer.load(ckpt, device=args.device, source_fps=args.source_fps)
        spec = tok.spec
        stride = int(spec.temporal_stride)
        print(f"[export-t2m] codec: Q={spec.num_codebooks} codebook={spec.codebook_size} "
              f"stride={spec.temporal_stride} token_rate={spec.token_rate:.2f}Hz", flush=True)

        parquet_dir = Path(resolve_local_uri(args.parquet_dir))
        _encode_motion(parquet_dir, args, tok, stride, out_dir=out_dir)
        print(f"[export-t2m] codes-only done: split={args.split}", flush=True)
        return

    # ---- resolve text encoder (for text-only and full export) ----
    from ..text import get_encoder_cls

    enc_cls = get_encoder_cls(args.text_encoder)
    model_id = args.text_encoder_model or enc_cls.DEFAULT_MODEL_ID
    print(f"[export-t2m] text encoder: {args.text_encoder} → {model_id}", flush=True)
    text_enc = enc_cls.load(
        model_id, device=args.device, max_length=args.text_max_length,
        local_files_only=args.local_files_only,
    )
    print(f"[export-t2m] text clip_dim={text_enc.clip_dim}", flush=True)

    suffix = args.text_encoder  # "flan", "siglip", "qwen3"

    # ---- text-only mode ----
    if args.text_only:
        index_path = out_dir / f"{args.split}.index.json"

        if args.parquet_dir:
            # Read captions directly from parquet (no existing index needed)
            parquet_dir = Path(resolve_local_uri(args.parquet_dir))
            captions = []
            shared_index = []
            seen = 0
            for row in _iter_parquet_rows(parquet_dir, args.split, text_only=True):
                if args.limit and seen >= args.limit:
                    break
                seen += 1
                text = row.get("text") or ""
                if not text:
                    continue
                captions.append(text)
                shared_index.append({
                    "rec_id": str(row["rec_id"]),
                    "code_start": 0, "code_len": 0,   # placeholder — filled by full export later
                    "row": len(shared_index),
                    "caption": text,
                })
            print(f"[export-t2m] --text-only: {len(captions)} captions from parquet {parquet_dir}/{args.split}",
                  flush=True)
            # Write shared index (without real code offsets) so subsequent --text-only runs find it
            (index_path).write_text(json.dumps(shared_index))
        elif index_path.is_file():
            shared_index = json.loads(index_path.read_text())
            captions = [e["caption"] for e in shared_index]
            print(f"[export-t2m] --text-only: encoding {len(captions)} captions from {index_path}", flush=True)
        else:
            raise FileNotFoundError(
                f"Shared index not found: {index_path}. "
                f"Pass --parquet-dir for first --text-only run, or run a full export first."
            )

        text_packed_list: list[np.ndarray] = []
        text_cursor = 0
        text_index: list[dict] = []
        log_every_batch = max(1, 10000 // args.text_subbatch)
        for cs in range(0, len(captions), args.text_subbatch):
            caps = captions[cs : cs + args.text_subbatch]
            emb, mask = text_enc.encode(caps)
            lens = mask.sum(dim=1).tolist()
            emb = emb.cpu().numpy().astype(np.float16)
            for k in range(len(caps)):
                L = int(lens[k])
                text_packed_list.append(emb[k, :L])
                text_index.append({"text_start": text_cursor, "text_len": L})
                text_cursor += L
            if (cs // args.text_subbatch) % log_every_batch == 0:
                pct = min(cs + len(caps), len(captions)) * 100.0 / len(captions)
                print(f"[export-t2m] encoding: {min(cs + len(caps), len(captions))}/{len(captions)} "
                      f"({pct:.0f}%)  |  {text_cursor} text tokens", flush=True)

        text_packed = np.concatenate(text_packed_list, axis=0)

        # text_index.{key}.json: text offsets only, aligned by row with shared index.json.
        # Code offsets live in shared index.json — no duplication, no staleness.
        np.save(out_dir / f"{args.split}.text_emb.{suffix}.npy", text_packed)
        (out_dir / f"{args.split}.text_index.{suffix}.json").write_text(json.dumps(text_index))
        (out_dir / f"{args.split}.meta.{suffix}.json").write_text(json.dumps({
            "split": args.split,
            "clip_dim": int(text_enc.clip_dim),
            "encode_key": args.text_encoder,
            "text_model_id": model_id,
            "text_max_length": int(args.text_max_length),
        }, indent=2))
        print(f"[export-t2m] --text-only done: {len(text_index)} clips | "
              f"text_emb {text_packed.shape} | → {args.split}.text_emb.{suffix}.npy", flush=True)
        return

    # ---- full export: motion codes + text ----
    if args.parquet_dir is None:
        raise SystemExit("--parquet-dir is required for full export (or use --text-only)")

    parquet_dir = Path(resolve_local_uri(args.parquet_dir))

    ckpt = Path(args.checkpoint) if args.checkpoint else default_checkpoint()
    print(f"[export-t2m] frozen tokenizer: {ckpt}", flush=True)
    tok = FrozenMotionTokenizer.load(ckpt, device=args.device, source_fps=args.source_fps)
    spec = tok.spec
    stride = int(spec.temporal_stride)
    print(f"[export-t2m] codec: Q={spec.num_codebooks} codebook={spec.codebook_size} "
          f"stride={spec.temporal_stride} token_rate={spec.token_rate:.2f}Hz", flush=True)

    code_buf: list[np.ndarray] = []
    text_buf: list[np.ndarray] = []
    anchor_buf: list[np.ndarray] = []
    identity_buf: list[np.ndarray] = []
    shared_index: list[dict] = []
    code_cursor = 0
    text_cursor = 0
    skipped = 0
    t0 = time.time()

    # Buffer clips into text-batches so Flan-T5 runs on padded batches.
    pend_feats: list[np.ndarray] = []
    pend_text: list[str] = []
    pend_row: list[dict] = []

    def flush_batch() -> None:
        nonlocal code_cursor, text_cursor, skipped
        N = len(pend_text)
        if N == 0:
            return
        # Text: sub-batched Flan-T5; store each clip's trimmed [L, dim] embedding.
        temb_by_j: list[np.ndarray | None] = [None] * N
        for cs in range(0, N, args.text_subbatch):
            caps = pend_text[cs : cs + args.text_subbatch]
            emb, mask = text_enc.encode(caps)            # [b, Lmax, dim], [b, Lmax]
            lens = mask.sum(dim=1).tolist()
            emb = emb.cpu().numpy().astype(np.float16)
            for k in range(len(caps)):
                temb_by_j[cs + k] = emb[k, : int(lens[k])]
        # Codec: bucket by EXACT trimmed length so batches have zero padding
        # (reconstruction-equivalent to per-clip encode; ~12x faster). Re-map to
        # original order so text/anchor stay aligned.
        codes_by_j: list[np.ndarray | None] = [None] * N
        buckets: dict[int, list[int]] = defaultdict(list)
        for j, f in enumerate(pend_feats):
            buckets[(int(f.shape[0]) // stride) * stride].append(j)
        for keep, js in buckets.items():
            if keep <= 0:
                continue
            for cs in range(0, len(js), args.codec_batch):
                sub = js[cs : cs + args.codec_batch]
                for j, c in zip(sub, tok.encode_batch([pend_feats[j] for j in sub]), strict=True):
                    codes_by_j[j] = c
        for j, row in enumerate(pend_row):
            codes = codes_by_j[j]
            if codes is None or codes.shape[0] < args.min_tokens:
                skipped += 1
                continue
            n_tok = int(codes.shape[0])
            temb = temb_by_j[j]
            L = int(temb.shape[0])
            code_buf.append(codes.astype(np.int16, copy=False))
            text_buf.append(temb)                        # [L, dim] fp16
            anchor_buf.append(_anchor_vector(row))
            identity_buf.append(np.asarray(row["identity_coeffs"], dtype=np.float32).reshape(-1))
            shared_index.append({
                "rec_id": str(row["rec_id"]),
                "code_start": code_cursor, "code_len": n_tok,
                "text_start": text_cursor, "text_len": L,
                "row": len(shared_index),
                "caption": pend_text[j],       # raw text (small; enables TMR R-precision)
            })
            code_cursor += n_tok
            text_cursor += L
        pend_feats.clear(); pend_text.clear(); pend_row.clear()

    seen = 0
    for row in _iter_parquet_rows(parquet_dir, args.split):
        if args.limit and seen >= args.limit:
            break
        seen += 1
        text = row.get("text") or ""
        if not text:
            skipped += 1
            continue
        pend_feats.append(row["features"]); pend_text.append(text); pend_row.append(row)
        if len(pend_text) >= args.flush_size:
            flush_batch()
        if seen % args.log_every == 0:
            rate = seen / max(1e-9, time.time() - t0)
            print(f"[export-t2m] {seen} clips seen | {len(shared_index)} written | "
                  f"{code_cursor} tok | {rate:.1f} clips/s | skipped {skipped}", flush=True)
    flush_batch()

    if not code_buf:
        raise RuntimeError("no clips encoded; check --parquet-dir / --split / text column")

    codes_packed = np.concatenate(code_buf, axis=0)              # [sum_T_tok, Q]
    text_packed = np.concatenate(text_buf, axis=0)              # [sum_L, dim]
    anchors = np.stack(anchor_buf, axis=0)                      # [N, 465]
    identities = np.stack(identity_buf, axis=0)                 # [N, C]

    # --- shared files (motion codes, anchors, identity — encoder-agnostic) ---
    np.save(out_dir / f"{args.split}.codes.npy", codes_packed)
    np.save(out_dir / f"{args.split}.anchor.npy", anchors)
    np.save(out_dir / f"{args.split}.identity.npy", identities)

    # Shared index: code offsets + captions (no text offsets — those are per-encoder).
    shared_index_out = []
    for e in shared_index:
        shared_index_out.append({
            "rec_id": e["rec_id"],
            "code_start": e["code_start"],
            "code_len": e["code_len"],
            "row": e["row"],
            "caption": e.get("caption", ""),
        })
    (out_dir / f"{args.split}.index.json").write_text(json.dumps(shared_index_out))

    # shared meta: codec metadata only (encoder-agnostic)
    shared_meta = {
        "split": args.split,
        "num_clips": len(shared_index),
        "num_tokens": int(codes_packed.shape[0]),
        "num_codebooks": int(spec.num_codebooks),
        "codebook_size": int(spec.codebook_size),
        "temporal_stride": int(spec.temporal_stride),
        "token_rate": float(spec.token_rate),
        "source_fps": float(spec.source_fps),
        "anchor_dim": _ANCHOR_DIM,
        "identity_dim": int(identities.shape[1]),
        "tokenizer_checkpoint": str(ckpt),
    }
    (out_dir / f"{args.split}.meta.json").write_text(json.dumps(shared_meta, indent=2))

    # --- encoder-specific: text offsets only (aligned by row with shared index) ---
    text_index_out = []
    for e in shared_index:
        text_index_out.append({
            "text_start": e["text_start"],
            "text_len": e["text_len"],
        })
    np.save(out_dir / f"{args.split}.text_emb.{suffix}.npy", text_packed)
    (out_dir / f"{args.split}.text_index.{suffix}.json").write_text(json.dumps(text_index_out))
    encoder_meta = {
        "split": args.split,
        "clip_dim": int(text_enc.clip_dim),
        "encode_key": args.text_encoder,
        "text_model_id": model_id,
        "text_encode": model_id,
        "text_max_length": int(args.text_max_length),
    }
    (out_dir / f"{args.split}.meta.{suffix}.json").write_text(json.dumps(encoder_meta, indent=2))

    print(f"[export-t2m] done: {len(shared_index)} clips | codes {codes_packed.shape} "
          f"| text_emb {text_packed.shape} | skipped {skipped} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
