"""Shared reader for the release's derived UMR parquet shards.

The release pipeline (soma ``tools/build_umr_release.py``) freezes each clip as a
row under ``releases/<id>/derived_umr_<hash>/{train,val,test}/*.parquet`` carrying
the ``features [T,499]`` UMR stream, a ``text`` caption, and the frame-0 decode
anchor. Both the v0 motion-code export and the v1 paired-store export read from
here (no per-clip ``umr499.npz`` exists in the release paradigm).

Reads only the requested columns and converts Arrow ``list<float>`` arrays to
NumPy via flat values + offsets (no ``to_pylist`` on the big per-clip feature
lists, which dominates ingest time). ``features`` is reshaped to ``[n, dim]``;
other list columns are yielded as flat ``float32`` arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Anchor layout: init_root_pos(3) + init_root_rot6d(6) + init_joints76_rot6d(456).
ANCHOR_DIM = 3 + 6 + 456

_STR_COLS = {"rec_id", "text"}
_LIST_COLS = {
    "features",
    "init_root_pos",
    "init_root_rot6d",
    "init_joints76_rot6d",
    "identity_coeffs",
    "joints77_pos",
    "joint_orient",
}


def _list_col_np(table, name: str) -> tuple[np.ndarray, np.ndarray]:
    """A ``list<float>`` column -> (flat values ``float32``, int offsets)."""
    arr = table.column(name).combine_chunks()
    return (
        arr.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False),
        arr.offsets.to_numpy(),
    )


def iter_parquet_rows(parquet_dir, split: str, cols, features_dim: int = 499):
    """Yield per-clip dicts from all shards of ``split``, in file order.

    ``cols`` is the set of payload columns to extract (besides ``rec_id`` /
    ``num_records``, which are always included). ``features`` is reshaped to
    ``[num_records, features_dim]``; other list columns come back flat.
    """
    import pyarrow.parquet as pq

    cols = list(dict.fromkeys(cols))
    shard_dir = Path(parquet_dir) / split
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {shard_dir}")

    read_cols = list(dict.fromkeys(["rec_id", "num_records", *cols]))
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=read_cols)
            rec_ids = table.column("rec_id").to_pylist()          # strings: cheap
            nrecs = table.column("num_records").to_numpy()
            str_data = {c: table.column(c).to_pylist() for c in cols if c in _STR_COLS}
            list_data = {c: _list_col_np(table, c) for c in cols if c in _LIST_COLS}
            for i in range(len(rec_ids)):
                n = int(nrecs[i])
                row: dict = {"rec_id": rec_ids[i], "num_records": n}
                for c in cols:
                    if c in _STR_COLS:
                        row[c] = str_data[c][i]
                    elif c in _LIST_COLS:
                        v, o = list_data[c]
                        seg = v[o[i]:o[i + 1]]
                        row[c] = seg.reshape(n, features_dim) if c == "features" else seg
                yield row


def anchor_vector(row: dict) -> np.ndarray:
    """Pack a row's frame-0 anchor fields into a single ``[ANCHOR_DIM]`` vector."""
    v = np.zeros((ANCHOR_DIM,), dtype=np.float32)
    v[0:3] = np.asarray(row["init_root_pos"], dtype=np.float32)
    v[3:9] = np.asarray(row["init_root_rot6d"], dtype=np.float32)
    v[9:465] = np.asarray(row["init_joints76_rot6d"], dtype=np.float32).reshape(-1)
    return v


# features + frame-0 anchor: everything needed to encode a clip and decode its
# codes back to absolute SOMA77 joints (via decode_to_joints_arrays).
CLIP_COLS = [
    "features",
    "init_root_pos", "init_root_rot6d", "init_joints76_rot6d", "identity_coeffs",
]

# CLIP_COLS + precomputed world-space joints (no FK needed).
GT_COLS = CLIP_COLS + ["joints77_pos"]


def joints77_from_row(row: dict) -> "np.ndarray":
    """Extract ``joints77_pos`` from a parquet row, drop anchor frame, reshape.

    Parquet stores ``(num_records + 1)`` frames (anchor frame included).
    The current decode pipeline outputs ``joints[1:]`` (anchor frame dropped),
    so we match that convention: return ``[num_records, 77, 3]``.
    """
    raw = np.asarray(row["joints77_pos"], dtype=np.float32)
    n = int(row["num_records"])
    return raw.reshape(n + 1, 77, 3)[1:]  # → [n, 77, 3]


def load_rows_by_rec_id(parquet_dir, split: str, rec_ids, cols=CLIP_COLS) -> dict:
    """Return ``{rec_id: row}`` for the requested ids (single scan, stops early)."""
    want = set(rec_ids)
    out: dict = {}
    if not want:
        return out
    for row in iter_parquet_rows(parquet_dir, split, cols=cols):
        if row["rec_id"] in want:
            out[row["rec_id"]] = row
            if len(out) == len(want):
                break
    return out


def load_features_by_rec_id(parquet_dir, split: str, rec_ids) -> dict:
    """Return ``{rec_id: features[n,499]}`` for the requested ids (single scan)."""
    return {
        rid: row["features"]
        for rid, row in load_rows_by_rec_id(parquet_dir, split, rec_ids, cols=["features"]).items()
    }


def _build_rec_id_index(parquet_dir, split: str) -> dict:
    """Build ``{rec_id: (shard_file, row_index)}`` from ``_index.json``.

    This makes targeted parquet reads possible — no full-scan needed.
    """
    import json as _json

    index_path = Path(parquet_dir) / "_index.json"
    index = _json.loads(index_path.read_text())
    mapping = {}
    for shard in index["splits"][split]["shards"]:
        shard_file = shard["file"]
        for row_idx, rec_id in enumerate(shard["rec_ids"]):
            mapping[rec_id] = (shard_file, row_idx)
    return mapping


# Module-level cache for rec_id → (shard, row) index
_rec_id_index_cache: dict[tuple, dict] = {}


def _get_rec_id_index(parquet_dir, split: str) -> dict:
    key = (str(parquet_dir), split)
    if key not in _rec_id_index_cache:
        _rec_id_index_cache[key] = _build_rec_id_index(parquet_dir, split)
    return _rec_id_index_cache[key]


def load_joints77_batch(
    parquet_dir, split: str, rec_ids: list[str],
) -> dict:
    """Return ``{rec_id: joints77[n,77,3]}`` using targeted parquet reads.

    Uses ``_index.json`` to read only the parquet shards that actually
    contain the requested rec_ids — avoids scanning all 29 test shards.

    Memory-safe: filters matching rows with ``table.take()`` **before** calling
    ``combine_chunks()``, so only the requested rows (not the entire row group)
    are materialised.
    """
    import os as _os

    import pyarrow as pa
    import pyarrow.parquet as pq

    # Limit PyArrow's I/O thread pool — 3 GPU processes × default-thread-count
    # (one per CPU core) can spawn 600+ threads and exhaust system limits.
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    ridx = _get_rec_id_index(parquet_dir, split)

    # Build a set for O(1) rec_id lookups during row-group filtering.
    wanted_rec_ids = set(rec_ids)

    # Group rec_ids by shard file
    by_shard: dict[str, list[tuple[int, str]]] = {}
    for rec_id in rec_ids:
        entry = ridx.get(rec_id)
        if entry is None:
            continue
        shard_file, row_idx = entry
        by_shard.setdefault(shard_file, []).append((row_idx, rec_id))

    out: dict = {}
    shard_dir = Path(parquet_dir) / split
    n_shards = len(by_shard)

    for shard_idx, (shard_file, entries) in enumerate(by_shard.items()):
        shard_path = shard_dir / Path(shard_file).name
        if not shard_path.is_file():
            continue
        want_rows = {row_idx for row_idx, _ in entries}
        row_to_rec_id = {row_idx: rec_id for row_idx, rec_id in entries}

        pf = pq.ParquetFile(shard_path, memory_map=True)
        n_groups = pf.num_row_groups

        for rg in range(n_groups):
            rg_rows = pf.metadata.row_group(rg).num_rows
            rg_start = sum(pf.metadata.row_group(i).num_rows for i in range(rg))
            # Quick check: does this row group overlap with wanted rows?
            wanted_in_rg = any(rg_start <= r < rg_start + rg_rows for r in want_rows)
            if not wanted_in_rg:
                continue

            table = pf.read_row_group(rg, columns=["rec_id", "num_records", "joints77_pos"])
            rec_ids_col = table.column("rec_id").to_pylist()

            # Find which rows in this row group match our wanted set.
            matching_indices = [
                i for i, rid in enumerate(rec_ids_col) if rid in wanted_rec_ids
            ]
            if not matching_indices:
                continue

            # Take ONLY matching rows before combine_chunks — this is the key
            # memory fix: a full row group holds ~6000 clips × 500 frames × 77×3
            # float32 ≈ 2.8 GB of joints77_pos data.  combine_chunks on the
            # filtered table materialises only the few hundred matching rows.
            filtered = table.take(matching_indices)
            rec_ids_col = filtered.column("rec_id").to_pylist()
            nrecs_col = filtered.column("num_records").to_numpy()

            j77_col = filtered.column("joints77_pos").combine_chunks()
            j77_flat = j77_col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            j77_offsets = j77_col.offsets.to_numpy()

            for i in range(len(rec_ids_col)):
                rec_id = rec_ids_col[i]
                n = int(nrecs_col[i])
                raw = j77_flat[j77_offsets[i]:j77_offsets[i + 1]]
                out[rec_id] = raw.reshape(n + 1, 77, 3)[1:]  # drop anchor frame

        if shard_idx % 5 == 0 or shard_idx == n_shards - 1:
            print(f"  [parquet] shard {shard_idx + 1}/{n_shards} done, "
                  f"{len(out)}/{len(wanted_rec_ids)} recs found so far", flush=True)

    return out


__all__ = [
    "ANCHOR_DIM",
    "CLIP_COLS",
    "GT_COLS",
    "anchor_vector",
    "iter_parquet_rows",
    "joints77_from_row",
    "load_features_by_rec_id",
    "load_joints77_batch",
    "load_rows_by_rec_id",
]
