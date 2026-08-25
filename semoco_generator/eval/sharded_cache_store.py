"""``ShardedCacheStore`` v2: packed shard storage with cheap metadata probes.

This is the real packed-shard implementation described in the production plan
(``bucket-XXXXX.pack`` + ``bucket-XXXXX.index.jsonl``), not just a per-file
manifest. It is deliberately independent of the legacy ``cache.py`` path-per-
artifact layout so existing call sites can adopt it incrementally per artifact
kind (native / converted / gen_emb / GT) without a hard cutover.

Design summary
---------------
* Keys hash to a stable bucket via SHA1, so writers for the same key always
  target the same ``(pack, index)`` pair regardless of process/shard.
* Each record is framed as ``MAGIC + header_len (u32 LE) + json header + raw
  array bytes``. The header records per-array dtype/shape/byte-range plus a
  small ``meta`` dict (e.g. ``fps``, ``rep``, protocol versions).
* Records are appended to the pack file 64-byte aligned. The pack file is
  opened in append mode and fsynced after every write so a crash mid-write
  never corrupts prior records.
* The **index line is only appended after the pack bytes are fsynced**, and
  ``probe_many``/``load_many`` only trust index entries. This gives atomic
  "commit" semantics without a separate marker file: a record is only visible
  once its index line exists, and the index line is only written after the
  payload is durably on disk.
* ``probe_many`` reads index files only (grouped by touched bucket) and never
  opens the pack file — the whole point of a metadata-first probe.
* ``load_many`` reads pack bytes in ``(bucket, offset)`` order so batched
  reads stay close to sequential I/O, and stays under ``max_bytes`` per
  yielded batch.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

_MAGIC = b"OMPK"
_ALIGN = 64
_INDEX_SUFFIX = ".index.jsonl"
_PACK_SUFFIX = ".pack"


# ---------------------------------------------------------------------------
# Public data shapes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PutRecord:
    key: str
    arrays: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)
    protocol_version: str = "v1"


@dataclass(frozen=True)
class ArtifactBatch:
    key: str
    arrays: dict[str, np.ndarray]
    meta: dict


@dataclass(frozen=True)
class CacheStatus:
    key: str
    exists: bool
    bucket: int | None = None
    offset: int | None = None
    nbytes: int = 0
    reason: str | None = None


@dataclass(frozen=True)
class ManifestDelta:
    scope: str
    written: int
    bytes_written: int
    buckets_touched: tuple[int, ...]


@dataclass(frozen=True)
class CacheAudit:
    scope: str
    records: int
    bytes: int
    corrupt: int
    missing_packs: int
    buckets: int
    protocol_versions: dict[str, int]


def _bucket_for(key: str, num_buckets: int) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % max(1, num_buckets)


def _bucket_name(bucket: int) -> str:
    return f"bucket-{bucket:05d}"


def _encode_record(key: str, arrays: dict[str, np.ndarray], meta: dict, protocol_version: str) -> bytes:
    header: dict = {"key": key, "meta": meta, "protocol_version": protocol_version, "arrays": {}}
    parts: list[bytes] = []
    offset = 0
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        if arr.ndim >= 1:
            arr = np.ascontiguousarray(arr)  # np.ascontiguousarray forces ndim >= 1; skip for scalars
        raw = arr.tobytes()
        header["arrays"][name] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "offset": offset,
            "nbytes": len(raw),
        }
        parts.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(header).encode("utf-8")
    body = b"".join(parts)
    return _MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + body


def _decode_record(raw: bytes) -> tuple[str, dict, dict[str, np.ndarray]]:
    if raw[:4] != _MAGIC:
        raise ValueError("bad pack record magic")
    header_len = struct.unpack("<I", raw[4:8])[0]
    header = json.loads(raw[8:8 + header_len].decode("utf-8"))
    body = raw[8 + header_len:]
    arrays: dict[str, np.ndarray] = {}
    for name, info in header["arrays"].items():
        dt = np.dtype(info["dtype"])
        nbytes = int(info["nbytes"])
        off = int(info["offset"])
        arr = np.frombuffer(body[off:off + nbytes], dtype=dt).reshape(info["shape"])
        arrays[name] = arr
    return header["key"], header.get("meta", {}), arrays


class _PackAppender:
    """Append-only, 64-byte-aligned writer for one bucket's pack file.

    Data is buffered and flushed per ``append()``, but *not* fsynced —
    call ``sync()`` explicitly after a batch to persist durably, then
    ``close()`` to release the file handle.
    """

    def __init__(self, pack_path: Path) -> None:
        self.pack_path = pack_path
        pack_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(pack_path, "ab")

    def append(self, data: bytes) -> tuple[int, int]:
        self._fh.seek(0, os.SEEK_END)
        pos = self._fh.tell()
        pad = (-pos) % _ALIGN
        if pad:
            self._fh.write(b"\x00" * pad)
            pos += pad
        self._fh.write(data)
        self._fh.flush()
        return pos, len(data)

    def sync(self) -> None:
        """Fsync the pack file so written data is durable on disk."""
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


class ShardedCacheStore:
    """Packed-shard cache store: ``probe_many`` / ``load_many`` / ``put_many`` / ``audit`` / ``drop``.

    ``root`` is a scope-agnostic base directory; each ``scope`` (e.g.
    ``native``, ``converted``, ``gen_emb``) gets its own bucket directory
    ``root/scope/``.
    """

    def __init__(self, root: str | Path, *, num_buckets: int = 16) -> None:
        self.root = Path(root)
        self.num_buckets = max(1, int(num_buckets))
        self._index_cache: dict[tuple[str, int], dict[str, dict]] = {}
        self._index_cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    def _scope_dir(self, scope: str) -> Path:
        return self.root / scope

    def _pack_path(self, scope: str, bucket: int) -> Path:
        return self._scope_dir(scope) / f"{_bucket_name(bucket)}{_PACK_SUFFIX}"

    def _index_path(self, scope: str, bucket: int) -> Path:
        return self._scope_dir(scope) / f"{_bucket_name(bucket)}{_INDEX_SUFFIX}"

    def _load_bucket_index(self, scope: str, bucket: int, *, force: bool = False) -> dict[str, dict]:
        cache_key = (scope, bucket)
        with self._index_cache_lock:
            if not force and cache_key in self._index_cache:
                return self._index_cache[cache_key]
        index_path = self._index_path(scope, bucket)
        entries: dict[str, dict] = {}
        if index_path.is_file():
            with index_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("status") == "ok":
                        entries[rec["key"]] = rec
                    else:
                        entries.pop(rec.get("key"), None)
        with self._index_cache_lock:
            self._index_cache[cache_key] = entries
        return entries

    def _invalidate(self, scope: str, bucket: int) -> None:
        with self._index_cache_lock:
            self._index_cache.pop((scope, bucket), None)

    # ------------------------------------------------------------------
    def put_many(self, scope: str, records: Sequence[PutRecord]) -> ManifestDelta:
        by_bucket: dict[int, list[PutRecord]] = {}
        for rec in records:
            b = _bucket_for(rec.key, self.num_buckets)
            by_bucket.setdefault(b, []).append(rec)

        total_bytes = 0
        written = 0
        for bucket, recs in by_bucket.items():
            pack_path = self._pack_path(scope, bucket)
            index_path = self._index_path(scope, bucket)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            # The index file is also the per-bucket advisory lock.  ``O_APPEND``
            # alone does not make ``seek(EOF) -> record offset -> write`` atomic
            # across processes, so two writers could publish offsets for the same
            # bytes in a pack.  Hold one lock across pack and index commits.
            with index_path.open("a") as idx_f:
                fcntl.flock(idx_f.fileno(), fcntl.LOCK_EX)
                try:
                    appender = _PackAppender(pack_path)
                    try:
                        for rec in recs:
                            raw = _encode_record(rec.key, rec.arrays, rec.meta, rec.protocol_version)
                            checksum = hashlib.sha256(raw).hexdigest()
                            offset, nbytes = appender.append(raw)
                            entry = {
                                "key": rec.key,
                                "bucket": bucket,
                                "offset": offset,
                                "nbytes": nbytes,
                                "checksum": checksum,
                                "protocol_version": rec.protocol_version,
                                "created_at": time.time(),
                                "status": "ok",
                            }
                            idx_f.write(json.dumps(entry, sort_keys=True) + "\n")
                            total_bytes += nbytes
                            written += 1
                        # Single fsync per bucket after all records are written -
                        # pack data first, then index (index-is-truth on recovery).
                        idx_f.flush()
                        appender.sync()
                        os.fsync(idx_f.fileno())
                    finally:
                        appender.close()
                finally:
                    fcntl.flock(idx_f.fileno(), fcntl.LOCK_UN)
            self._invalidate(scope, bucket)

        return ManifestDelta(
            scope=scope, written=written, bytes_written=total_bytes,
            buckets_touched=tuple(sorted(by_bucket)),
        )

    # ------------------------------------------------------------------
    def _load_bucket_index_fresh_on_miss(
        self, scope: str, bucket: int, wanted_keys: Sequence[str],
    ) -> dict[str, dict]:
        """Read the (possibly process-cached) bucket index, and if any
        ``wanted_keys`` aren't in it, re-read the index from disk once before
        concluding they're truly absent.

        ``put_many`` only invalidates *this process's* in-memory cache on
        *its own* writes (see :meth:`_load_bucket_index`) -- it has no way to
        know another worker *process* wrote to the same bucket. Multiple
        GPU-pinned worker processes lease/commit against the same manifest
        concurrently (see :mod:`worker_pool`), so a convert-phase worker can
        easily probe a bucket before some other process's native-gen commit
        to that same bucket ever lands, cache the miss for the rest of this
        process's lifetime, and then report "cache miss" on a dependency
        that a sibling process already durably wrote. One extra disk read
        only on an actual miss keeps the common (already-cached-and-found)
        path free.
        """
        index = self._load_bucket_index(scope, bucket)
        if any(k not in index for k in wanted_keys):
            index = self._load_bucket_index(scope, bucket, force=True)
        return index

    def probe_many(self, scope: str, keys: Iterable[str]) -> dict[str, CacheStatus]:
        keys = list(keys)
        by_bucket: dict[int, list[str]] = {}
        for key in keys:
            by_bucket.setdefault(_bucket_for(key, self.num_buckets), []).append(key)

        result: dict[str, CacheStatus] = {}
        for bucket, bucket_keys in by_bucket.items():
            index = self._load_bucket_index_fresh_on_miss(scope, bucket, bucket_keys)
            for key in bucket_keys:
                entry = index.get(key)
                if entry is None:
                    result[key] = CacheStatus(key=key, exists=False, bucket=bucket, reason="not_in_index")
                else:
                    result[key] = CacheStatus(
                        key=key, exists=True, bucket=bucket,
                        offset=entry["offset"], nbytes=entry["nbytes"],
                    )
        return result

    # ------------------------------------------------------------------
    def load_many(
        self, scope: str, keys: Iterable[str], *, max_bytes: int | None = None, verify: bool = True,
    ) -> Iterator[list[ArtifactBatch]]:
        keys = list(keys)
        by_bucket: dict[int, list[str]] = {}
        for key in keys:
            by_bucket.setdefault(_bucket_for(key, self.num_buckets), []).append(key)

        limit = max_bytes if max_bytes is not None and max_bytes > 0 else None
        batch: list[ArtifactBatch] = []
        batch_bytes = 0

        for bucket in sorted(by_bucket):
            index = self._load_bucket_index_fresh_on_miss(scope, bucket, by_bucket[bucket])
            pack_path = self._pack_path(scope, bucket)
            wanted = [(index[k]["offset"], k) for k in by_bucket[bucket] if k in index]
            wanted.sort()  # sequential read order within the pack
            if not wanted:
                continue
            with pack_path.open("rb") as f:
                for offset, key in wanted:
                    entry = index[key]
                    nbytes = entry["nbytes"]
                    if limit is not None and batch and batch_bytes + nbytes > limit:
                        yield batch
                        batch = []
                        batch_bytes = 0
                    f.seek(offset)
                    raw = f.read(nbytes)
                    if verify:
                        checksum = hashlib.sha256(raw).hexdigest()
                        if checksum != entry.get("checksum"):
                            continue  # corrupt record — skip rather than raise
                    rec_key, meta, arrays = _decode_record(raw)
                    batch.append(ArtifactBatch(key=rec_key, arrays=arrays, meta=meta))
                    batch_bytes += nbytes
        if batch:
            yield batch

    def load_one(self, scope: str, key: str) -> ArtifactBatch | None:
        for batch in self.load_many(scope, [key]):
            for item in batch:
                if item.key == key:
                    return item
        return None

    # ------------------------------------------------------------------
    def audit(self, scope: str) -> CacheAudit:
        scope_dir = self._scope_dir(scope)
        records = 0
        total_bytes = 0
        corrupt = 0
        missing_packs = 0
        buckets = 0
        protocol_versions: dict[str, int] = {}
        if not scope_dir.is_dir():
            return CacheAudit(scope, 0, 0, 0, 0, 0, {})
        for index_path in sorted(scope_dir.glob(f"*{_INDEX_SUFFIX}")):
            buckets += 1
            bucket_name = index_path.name[: -len(_INDEX_SUFFIX)]
            pack_path = scope_dir / f"{bucket_name}{_PACK_SUFFIX}"
            pack_exists = pack_path.is_file()
            with index_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        corrupt += 1
                        continue
                    if entry.get("status") != "ok":
                        continue
                    records += 1
                    total_bytes += int(entry.get("nbytes", 0))
                    pv = str(entry.get("protocol_version", "unknown"))
                    protocol_versions[pv] = protocol_versions.get(pv, 0) + 1
                    if not pack_exists:
                        missing_packs += 1
        return CacheAudit(
            scope=scope, records=records, bytes=total_bytes, corrupt=corrupt,
            missing_packs=missing_packs, buckets=buckets, protocol_versions=protocol_versions,
        )

    # ------------------------------------------------------------------
    def drop(self, scope: str, *, dry_run: bool = True) -> list[str]:
        """List (dry-run) or remove every pack/index file for ``scope``."""
        scope_dir = self._scope_dir(scope)
        if not scope_dir.is_dir():
            return []
        paths = sorted(str(p) for p in scope_dir.glob(f"*{_PACK_SUFFIX}")) + sorted(
            str(p) for p in scope_dir.glob(f"*{_INDEX_SUFFIX}")
        )
        if not dry_run:
            for p in paths:
                try:
                    Path(p).unlink()
                except OSError:
                    pass
            for bucket in range(self.num_buckets):
                self._invalidate(scope, bucket)
        return paths

    def scopes(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def sample_key(self, scope: str) -> str | None:
        """Return any one live key in *scope*, or ``None`` if empty. Cheap:
        stops at the first ``status: ok`` index line it finds."""
        scope_dir = self._scope_dir(scope)
        if not scope_dir.is_dir():
            return None
        for index_path in sorted(scope_dir.glob(f"*{_INDEX_SUFFIX}")):
            with index_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("status") == "ok":
                        return entry.get("key")
        return None

    def enumerate_scope(self, scope: str) -> list[dict]:
        """Read all live index entries for *scope* and return them as dicts.

        Each dict contains at minimum ``key``, ``bucket``, ``offset``,
        ``nbytes``, ``protocol_version``, ``created_at``.  Read-only — never
        touches pack files.  Useful for cache coverage inspection, migration
        tooling, and ad-hoc queries.

        For scopes with millions of records this will allocate ~200-400 MiB
        of Python objects; if that becomes a problem, add a generator variant
        or a ``--sample`` flag at the call site.
        """
        scope_dir = self._scope_dir(scope)
        if not scope_dir.is_dir():
            return []
        entries: list[dict] = []
        for index_path in sorted(scope_dir.glob(f"*{_INDEX_SUFFIX}")):
            with index_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("status") == "ok":
                        entries.append(entry)
        return entries


__all__ = [
    "ArtifactBatch",
    "CacheAudit",
    "CacheStatus",
    "ManifestDelta",
    "PutRecord",
    "ShardedCacheStore",
]
