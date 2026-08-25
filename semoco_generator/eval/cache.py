"""Eval cache: durable shared truths + run-local volatile artifacts.

Backed entirely by the packed-shard stores (:mod:`sharded_cache_store`,
:mod:`run_artifact_store`) — no per-clip ``.npy``/``.npz`` files, no sidecar
checksums, no separate transitional manifest index. This is the *only*
storage backend now; there is no legacy small-file fallback to keep in sync.

**Durable shared cache** (under ``SEMOCO_EVAL_CACHE_ROOT`` / ``<data-root>/eval_cache/v2``):

* HumanML3D GT motion/text embeddings  (clip / caption + evaluator sig)
* SOMA/TMR GT joints + motion embeddings + caption text embeddings
* Model **native** outputs (model + ckpt + clip + seed + cfg) — same prompts
  regenerate the same native motion, independent of our conversion graph

**Run-local / volatile artifacts** (default under ``<out_root>/run_artifacts/cache_v2``):

* converted targets (depend on ``CONVERSION_GRAPH_VERSION`` / conversion bugs)
* generated-motion embeddings (depend on converted targets)

Pass ``run_root=...`` only to converted/gen APIs. Native always uses the shared
cache so HML and TMR tracks can reuse the same generations.

The ``*_path()`` functions below no longer point at literal per-clip files —
they are canonical identity strings used to derive stable packed-store keys,
kept around because their hierarchical shape (dataset/model/ckpt/...) is
useful for logging/debugging and is asserted on by existing tests.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

from .cache_utils import cfg_str, packed_cache_root, safe

from .run_artifact_store import (
    RunArtifactStore,
    arrays_to_clip,
    clip_to_arrays,
    converted_key,
    gen_embedding_key,
)
from .schema import MotionClip, MotionRep
from .sharded_cache_store import PutRecord, ShardedCacheStore

from .cache_versions import (
    CONVERSION_GRAPH_VERSION,
    GENERATION_PROTOCOL_VERSION,
    GEN_PROTOCOL_VERSION_STEP,
    GT_PROTOCOL_VERSION,
)

# Model IDs whose native cache key includes dataset_sig because they consume
# motion codes from the code store (e.g. SeMoCo-Generator).  Baseline models
# generate purely from text prompts and omit dataset_sig so code-store changes
# do not invalidate their cached native outputs.
_DATASET_DEPENDENT_MODELS: frozenset = frozenset({"semoco"})

# Subdir name under a protocol out_root for volatile generation/conversion artifacts.
RUN_ARTIFACTS_DIR = "run_artifacts"

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
def cache_root() -> Path:
    """Durable shared cache root (GT truths + native generations)."""
    env = os.environ.get("SEMOCO_EVAL_CACHE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    from ..paths import datasets_root

    return datasets_root() / "eval_cache"


def run_artifacts_root(out_root: str | Path | None = None) -> Path:
    """Volatile conversion/gen-embedding root for one eval protocol run.

    Prefer ``out_root / run_artifacts``. Falls back to
    ``cache_root() / "run_artifacts"`` only when no out_root is given (tests).
    """
    if out_root is not None:
        return Path(out_root) / RUN_ARTIFACTS_DIR
    env = os.environ.get("SEMOCO_EVAL_RUN_ARTIFACTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return cache_root() / RUN_ARTIFACTS_DIR


def _artifact_root(run_root: str | Path | None) -> Path:
    return Path(run_root) if run_root is not None else run_artifacts_root()


# ---------------------------------------------------------------------------
# Packed-store singletons (process-wide, so repeated probes reuse warm bucket
# index caches instead of re-reading index files from disk every call).
# ---------------------------------------------------------------------------
_V2_STORES: dict[str, ShardedCacheStore] = {}
_RUN_STORES: dict[str, RunArtifactStore] = {}


def v2_root() -> Path:
    return packed_cache_root(cache_root())


def _v2_store() -> ShardedCacheStore:
    root = str(v2_root())
    store = _V2_STORES.get(root)
    if store is None:
        store = ShardedCacheStore(v2_root(), num_buckets=32)
        _V2_STORES[root] = store
    return store


def _run_store(run_root: str | Path | None) -> RunArtifactStore:
    root = str(_artifact_root(run_root))
    store = _RUN_STORES.get(root)
    if store is None:
        store = RunArtifactStore(root, num_buckets=16)
        _RUN_STORES[root] = store
    return store


def _key_from_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Signatures / small helpers
# ---------------------------------------------------------------------------
# safe() imported from .cache_utils


def ckpt_sig(path: str | Path | None) -> str:
    """Content-based checkpoint signature: sha256(first_1MiB + last_1MiB + file_size).

    Stable across file moves, copies, and mtime changes — only file content
    matters.  Reads at most 2 MiB regardless of file size, so hashing is
    near-instant even for multi-GB checkpoints.
    """
    if path is None:
        return "none"
    p = Path(path)
    try:
        file_size = p.stat().st_size
        with open(p, "rb") as f:
            head = f.read(1_048_576)  # first 1 MiB
            if file_size > 2_097_152:
                f.seek(-1_048_576, os.SEEK_END)
            tail = f.read(1_048_576)  # last 1 MiB
        payload = head + tail + str(file_size).encode()
        return hashlib.sha256(payload).hexdigest()[:12]
    except OSError:
        return "missing"



def text_key(caption: str, tokens: list[str] | None = None) -> str:
    payload = str(caption) + "\x00" + " ".join(tokens or [])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def dataset_sig(root: str | Path, split: str, extra: str = "") -> str:
    """Signature of a prompt/GT source (store or HumanML tree) + split."""
    name = safe(Path(str(root)).name)
    tail = f"_{safe(extra)}" if extra else ""
    return f"{name}_{safe(split)}{tail}"


def hml_gt_sig(
    evaluator_ckpt: str | Path | None,
    *,
    official_encode: bool,
    hml_protocol: str,
    data: str = "",
    mean_std: str | Path | None = None,
    glove: str | None = None,
) -> str:
    parts = [
        GT_PROTOCOL_VERSION,
        ckpt_sig(evaluator_ckpt),
        "off" if official_encode else "raw",
        safe(hml_protocol),
    ]
    if data:
        parts.append(safe(data))
    if mean_std is not None:
        ms = Path(mean_std)
        parts.append(ckpt_sig(ms / "mean.npy") if (ms / "mean.npy").is_file() else "noms")
    if glove:
        parts.append(safe(Path(str(glove)).name))
    return "_".join(parts)


def hml_gt_text_sig(
    evaluator_ckpt: str | Path | None,
    *,
    official_encode: bool,
    hml_protocol: str,
    data: str = "",
    glove: str | None = None,
) -> str:
    """GT text-embedding signature — same as ``hml_gt_sig`` but WITHOUT ``mean_std``.

    Text embeddings (GloVe vectors via ``TextMotMatchEvaluator``) do not depend
    on motion normalization parameters, so ``mean_std`` is excluded from the
    cache key.  This prevents spurious cache misses when ``Mean.npy`` /
    ``mean.npy`` casing differs on disk.
    """
    parts = [
        GT_PROTOCOL_VERSION,
        ckpt_sig(evaluator_ckpt),
        "off" if official_encode else "raw",
        safe(hml_protocol),
    ]
    if data:
        parts.append(safe(data))
    if glove:
        parts.append(safe(Path(str(glove)).name))
    return "_".join(parts)


def tmr_gt_sig(
    tmr_model: str,
    *,
    store: str = "",
) -> str:
    """Stable GT signature — data-version based, no tokenizer dependency.

    GT joints now come directly from original Parquet data, so the cache key
    only needs to identify the data release (via ``GT_PROTOCOL_VERSION``) and
    the TMR model variant.
    """
    tail = f"__{safe(store)}" if store else ""
    return f"{GT_PROTOCOL_VERSION}__{safe(tmr_model)}{tail}"


# ---------------------------------------------------------------------------
# Durable: native model output (shared across tracks / runs)
# ---------------------------------------------------------------------------
def native_path(
    model_id: str,
    ckpt_signature: str,
    clip_id: str,
    seed: int,
    cfg: float | None,
    *,
    dataset: str = "",
) -> Path:
    """Canonical identity path for one native generation (key source, not a literal file)."""
    if model_id in _DATASET_DEPENDENT_MODELS:
        return (
            cache_root()
            / "native"
            / safe(dataset)
            / safe(model_id)
            / safe(ckpt_signature)
            / f"{safe(clip_id)}_s{int(seed)}_cfg{cfg_str(cfg)}_{GENERATION_PROTOCOL_VERSION}.npz"
        )
    # Baseline models generate from text only — omit dataset_sig so code-store
    # changes don't invalidate cached native outputs.
    return (
        cache_root()
        / "native"
        / safe(model_id)
        / safe(ckpt_signature)
        / f"{safe(clip_id)}_s{int(seed)}_cfg{cfg_str(cfg)}_{GENERATION_PROTOCOL_VERSION}.npz"
    )


def _native_key(model_id, ckpt_signature, clip_id, seed, cfg, *, dataset="") -> str:
    return _key_from_path(
        native_path(model_id, ckpt_signature, clip_id, seed, cfg, dataset=dataset), cache_root(),
    )


def load_native(
    model_id,
    ckpt_signature,
    clip_id,
    seed,
    cfg,
    *,
    dataset="",
    run_root: str | Path | None = None,  # ignored; kept for call-site compatibility
) -> MotionClip | None:
    del run_root
    key = _native_key(model_id, ckpt_signature, clip_id, seed, cfg, dataset=dataset)
    item = _v2_store().load_one("native", key)
    if item is None:
        return None
    return arrays_to_clip(item.meta.get("rep", ""), item.arrays)


def load_native_many(
    model_id,
    ckpt_signature,
    items: list[tuple[str, int, float | None]],
    *,
    dataset: str = "",
) -> dict[tuple[str, int, float | None], MotionClip | None]:
    """Bulk load across (clip_id, seed, cfg) tuples.

    Returns a dict keyed by ``(clip_id, seed, cfg)`` → MotionClip or None.
    Uses a single ``load_many`` call per bucket, reading pack records
    sequentially instead of one seek per clip.
    """
    key_by_tuple: dict[tuple, str] = {}
    for clip_id, seed, cfg in items:
        tup = (clip_id, int(seed), cfg)
        key_by_tuple[tup] = _native_key(model_id, ckpt_signature, clip_id, int(seed), cfg, dataset=dataset)

    # load_many yields batches of ArtifactBatch; collect into a key→item dict
    loaded: dict[str, object] = {}
    for batch in _v2_store().load_many("native", list(key_by_tuple.values())):
        for artifact in batch:
            loaded[artifact.key] = artifact

    result: dict[tuple[str, int, float | None], MotionClip | None] = {}
    for tup, key in key_by_tuple.items():
        item = loaded.get(key)
        result[tup] = arrays_to_clip(item.meta.get("rep", ""), item.arrays) if item else None
    return result


def probe_native(
    model_id,
    ckpt_signature,
    clip_id,
    seed,
    cfg,
    *,
    dataset="",
    run_root: str | Path | None = None,
) -> bool:
    del run_root
    key = _native_key(model_id, ckpt_signature, clip_id, seed, cfg, dataset=dataset)
    return _v2_store().probe_many("native", [key])[key].exists


def probe_native_many(
    model_id,
    ckpt_signature,
    clip_ids: list[str],
    seed,
    cfg,
    *,
    dataset="",
) -> dict[str, bool]:
    """Bulk probe (one packed-store call) across ``clip_ids`` for one
    (model, ckpt, seed, cfg, dataset) group — the warm-cache hot path
    for ``ensure_native``'s skip-existing scan."""
    key_by_clip = {
        c: _native_key(model_id, ckpt_signature, c, seed, cfg, dataset=dataset) for c in clip_ids
    }
    status = _v2_store().probe_many("native", list(key_by_clip.values()))
    return {c: status[key_by_clip[c]].exists for c in clip_ids}


def save_native(
    model_id,
    ckpt_signature,
    clip_id,
    seed,
    cfg,
    clip: MotionClip,
    *,
    dataset="",
    run_root: str | Path | None = None,  # ignored; kept for call-site compatibility
) -> None:
    del run_root
    key = _native_key(model_id, ckpt_signature, clip_id, seed, cfg, dataset=dataset)
    _v2_store().put_many(
        "native", [PutRecord(key=key, arrays=clip_to_arrays(clip), meta={"rep": str(clip.rep)})],
    )


def save_native_many(
    model_id,
    ckpt_signature,
    items: list[tuple[str, int, float | None, MotionClip]],
    *,
    dataset="",
) -> None:
    """Save multiple native clips in a single ``put_many`` call.

    Each item is ``(clip_id, seed, cfg, clip)``.  This batches all records
    into one pack+index fsync instead of one per clip, dramatically reducing
    I/O wait on network filesystems (CephFS).
    """
    if not items:
        return
    records = [
        PutRecord(
            key=_native_key(model_id, ckpt_signature, clip_id, seed, cfg, dataset=dataset),
            arrays=clip_to_arrays(clip),
            meta={"rep": str(clip.rep)},
        )
        for clip_id, seed, cfg, clip in items
    ]
    _v2_store().put_many("native", records)


# ---------------------------------------------------------------------------
# Volatile: converted target (run-local)
# ---------------------------------------------------------------------------
def converted_path(
    model_id,
    ckpt_signature,
    clip_id,
    seed,
    cfg,
    target_rep: MotionRep,
    *,
    dataset: str = "",
    run_root: str | Path | None = None,
) -> Path:
    """Canonical identity path for one converted target (key source, not a literal file)."""
    return (
        _artifact_root(run_root)
        / "converted"
        / safe(dataset)
        / safe(model_id)
        / safe(ckpt_signature)
        / safe(target_rep)
        / f"{safe(clip_id)}_s{int(seed)}_cfg{cfg_str(cfg)}_{CONVERSION_GRAPH_VERSION}.npz"
    )


def load_converted(
    model_id,
    ckpt_signature,
    clip_id,
    seed,
    cfg,
    target_rep,
    *,
    dataset="",
    run_root: str | Path | None = None,
) -> MotionClip | None:
    return _run_store(run_root).load_converted(
        model_id, ckpt_signature, clip_id, seed, cfg, target_rep,
        dataset_sig=dataset, conversion_version=CONVERSION_GRAPH_VERSION,
    )


def probe_converted(
    model_id,
    ckpt_signature,
    clip_id,
    seed,
    cfg,
    target_rep,
    *,
    dataset="",
    run_root: str | Path | None = None,
) -> bool:
    key = converted_key(
        model_id, ckpt_signature, clip_id, seed, cfg, target_rep,
        dataset_sig=dataset, conversion_version=CONVERSION_GRAPH_VERSION,
    )
    return _run_store(run_root).probe_converted_many([key])[key]


def probe_converted_many(
    model_id,
    ckpt_signature,
    clip_ids: list[str],
    seed,
    cfg,
    target_rep,
    *,
    dataset="",
    run_root: str | Path | None = None,
) -> dict[str, bool]:
    """Bulk probe (one packed-store call) across ``clip_ids`` — the warm-cache
    hot path for ``ensure_target``'s skip-existing scan."""
    key_by_clip = {
        c: converted_key(
            model_id, ckpt_signature, c, seed, cfg, target_rep,
            dataset_sig=dataset, conversion_version=CONVERSION_GRAPH_VERSION,
        )
        for c in clip_ids
    }
    status = _run_store(run_root).probe_converted_many(list(key_by_clip.values()))
    return {c: status[key_by_clip[c]] for c in clip_ids}


def save_converted(
    model_id,
    ckpt_signature,
    clip_id,
    seed,
    cfg,
    target_rep,
    clip: MotionClip,
    *,
    dataset="",
    run_root: str | Path | None = None,
) -> None:
    _run_store(run_root).put_converted(
        model_id, ckpt_signature, clip_id, seed, cfg, target_rep, clip,
        dataset_sig=dataset, conversion_version=CONVERSION_GRAPH_VERSION,
    )


def save_converted_many(
    model_id,
    ckpt_signature,
    items: list[tuple[str, int, float | None, str, object]],
    *,
    dataset="",
    run_root: str | Path | None = None,
) -> None:
    """Save multiple converted targets in a single ``put_many`` call.

    Each item is ``(clip_id, seed, cfg, target_rep, clip)``.
    """
    if not items:
        return
    records = [
        (model_id, ckpt_signature, clip_id, seed, cfg, target_rep, clip)
        for clip_id, seed, cfg, target_rep, clip in items
    ]
    _run_store(run_root).put_converted_many(
        records, dataset_sig=dataset, conversion_version=CONVERSION_GRAPH_VERSION,
    )


# ---------------------------------------------------------------------------
# Stable: HumanML3D GT caches
# ---------------------------------------------------------------------------
def hml_gt_motion_path(gt_sig: str, clip_id: str) -> Path:
    return cache_root() / "hml_gt" / safe(gt_sig) / "motion_emb" / f"{safe(clip_id)}.npy"


def load_hml_gt_motion(gt_sig, clip_id) -> np.ndarray | None:
    key = _key_from_path(hml_gt_motion_path(gt_sig, clip_id), cache_root())
    item = _v2_store().load_one("hml_gt_motion", key)
    return None if item is None else np.asarray(item.arrays["array"], dtype=np.float32)


def load_hml_gt_motion_many(gt_sig, clip_ids: list[str]) -> dict[str, np.ndarray | None]:
    """Batch-load HML GT motion embeddings for multiple clips.

    Returns a dict mapping ``clip_id`` → embedding array (or None if missing).
    Uses a single ``load_many`` call, reading pack records sequentially per bucket
    instead of one seek per clip.
    """
    root = cache_root()
    key_by_clip = {c: _key_from_path(hml_gt_motion_path(gt_sig, c), root) for c in clip_ids}
    loaded: dict[str, np.ndarray] = {}
    for batch in _v2_store().load_many("hml_gt_motion", list(key_by_clip.values())):
        for item in batch:
            loaded[item.key] = np.asarray(item.arrays["array"], dtype=np.float32)
    return {c: loaded.get(key_by_clip[c]) for c in clip_ids}


def probe_hml_gt_motion(gt_sig, clip_id) -> bool:
    key = _key_from_path(hml_gt_motion_path(gt_sig, clip_id), cache_root())
    return _v2_store().probe_many("hml_gt_motion", [key])[key].exists


def save_hml_gt_motion(gt_sig, clip_id, emb: np.ndarray) -> None:
    key = _key_from_path(hml_gt_motion_path(gt_sig, clip_id), cache_root())
    _v2_store().put_many("hml_gt_motion", [PutRecord(key=key, arrays={"array": np.asarray(emb, dtype=np.float32)})])


def hml_gt_text_path(text_sig: str, tkey: str) -> Path:
    return cache_root() / "hml_gt" / safe(text_sig) / "text_emb" / f"{tkey}.npy"


def load_hml_gt_text(text_sig, tkey) -> np.ndarray | None:
    key = _key_from_path(hml_gt_text_path(text_sig, tkey), cache_root())
    item = _v2_store().load_one("hml_gt_text", key)
    return None if item is None else np.asarray(item.arrays["array"], dtype=np.float32)


def load_hml_gt_text_many(text_sig, tkeys: list[tuple[str, str]]) -> dict[tuple[str, str], np.ndarray | None]:
    """Batch-load HML GT text embeddings.

    *tkeys* is a list of ``(caption, tkey)`` tuples where *tkey* is a pre-computed
    text key (from :func:`text_key`).  Returns a dict mapping ``(caption, tkey)``
    → embedding array (or None if missing).
    """
    root = cache_root()
    key_by_tkey: dict[str, tuple[str, str]] = {}
    keys: list[str] = []
    for caption, tkey in tkeys:
        k = _key_from_path(hml_gt_text_path(text_sig, tkey), root)
        key_by_tkey[k] = (caption, tkey)
        keys.append(k)
    loaded: dict[str, np.ndarray] = {}
    for batch in _v2_store().load_many("hml_gt_text", keys):
        for item in batch:
            loaded[item.key] = np.asarray(item.arrays["array"], dtype=np.float32)
    return {key_by_tkey[k]: loaded.get(k) for k in keys}


def probe_hml_gt_text(text_sig, tkey) -> bool:
    key = _key_from_path(hml_gt_text_path(text_sig, tkey), cache_root())
    return _v2_store().probe_many("hml_gt_text", [key])[key].exists


def save_hml_gt_text(text_sig, tkey, emb: np.ndarray) -> None:
    key = _key_from_path(hml_gt_text_path(text_sig, tkey), cache_root())
    _v2_store().put_many("hml_gt_text", [PutRecord(key=key, arrays={"array": np.asarray(emb, dtype=np.float32)})])


def list_missing_hml_gt_motion(gt_sig: str, clip_ids: list[str]) -> list[str]:
    key_by_clip = {c: _key_from_path(hml_gt_motion_path(gt_sig, c), cache_root()) for c in clip_ids}
    status = _v2_store().probe_many("hml_gt_motion", list(key_by_clip.values()))
    return [c for c in clip_ids if not status[key_by_clip[c]].exists]


def list_missing_hml_gt_text(text_sig: str, captions_and_tokens: list[tuple[str, list[str]]]) -> list[str]:
    """Return text_keys that are missing from the HML GT text cache."""
    tk_by_key: dict[str, str] = {}
    for caption, tokens in captions_and_tokens:
        tk = text_key(caption, tokens)
        key = _key_from_path(hml_gt_text_path(text_sig, tk), cache_root())
        tk_by_key[key] = tk
    status = _v2_store().probe_many("hml_gt_text", list(tk_by_key.keys()))
    return [tk for key, tk in tk_by_key.items() if not status[key].exists]


# ---------------------------------------------------------------------------
# Stable: SOMA/TMR GT caches
# ---------------------------------------------------------------------------
def tmr_gt_joints_path(
    gt_sig: str,
    clip_id: str,
    *,
    store: str = "",
) -> Path:
    sub = f"{safe(gt_sig)}" + (f"__{safe(store)}" if store else "")
    return cache_root() / "tmr_gt" / sub / "joints77" / f"{safe(clip_id)}.npy"


def load_tmr_gt_joints(gt_sig, clip_id, *, store="") -> np.ndarray | None:
    key = _key_from_path(tmr_gt_joints_path(gt_sig, clip_id, store=store), cache_root())
    item = _v2_store().load_one("tmr_gt_joints", key)
    return None if item is None else np.asarray(item.arrays["array"], dtype=np.float32)


def probe_tmr_gt_joints(gt_sig, clip_id, *, store="") -> bool:
    key = _key_from_path(tmr_gt_joints_path(gt_sig, clip_id, store=store), cache_root())
    return _v2_store().probe_many("tmr_gt_joints", [key])[key].exists


def save_tmr_gt_joints(gt_sig, clip_id, joints: np.ndarray, *, store="") -> None:
    key = _key_from_path(tmr_gt_joints_path(gt_sig, clip_id, store=store), cache_root())
    _v2_store().put_many(
        "tmr_gt_joints", [PutRecord(key=key, arrays={"array": np.asarray(joints, dtype=np.float32)})],
    )


def save_tmr_gt_joints_batch(gt_sig, items: list[tuple[str, np.ndarray]], *, store="") -> int:
    """Batch-save GT joints — one fsync per bucket instead of one per clip.

    ``items`` is a list of ``(clip_id, joints_array)`` tuples.  Returns the
    number of records written.  Use this instead of calling
    :func:`save_tmr_gt_joints` in a tight loop; on CephFS / network
    filesystems the per-call fsync makes single-record writes 100-1000×
    slower than a single batched put.
    """
    if not items:
        return 0
    root = cache_root()
    records = [
        PutRecord(
            key=_key_from_path(tmr_gt_joints_path(gt_sig, cid, store=store), root),
            arrays={"array": np.asarray(j, dtype=np.float32)},
        )
        for cid, j in items
    ]
    delta = _v2_store().put_many("tmr_gt_joints", records)
    return delta.written


def tmr_gt_motion_path(gt_sig: str, clip_id: str) -> Path:
    return cache_root() / "tmr_gt" / safe(gt_sig) / "motion_emb" / f"{safe(clip_id)}.npy"


def load_tmr_gt_motion(gt_sig, clip_id) -> np.ndarray | None:
    key = _key_from_path(tmr_gt_motion_path(gt_sig, clip_id), cache_root())
    item = _v2_store().load_one("tmr_gt_motion", key)
    return None if item is None else np.asarray(item.arrays["array"], dtype=np.float32)


def load_tmr_gt_motion_many(gt_sig, clip_ids: list[str]) -> dict[str, np.ndarray | None]:
    """Batch-load TMR GT motion embeddings for multiple clips."""
    root = cache_root()
    key_by_clip = {c: _key_from_path(tmr_gt_motion_path(gt_sig, c), root) for c in clip_ids}
    loaded: dict[str, np.ndarray] = {}
    for batch in _v2_store().load_many("tmr_gt_motion", list(key_by_clip.values())):
        for item in batch:
            loaded[item.key] = np.asarray(item.arrays["array"], dtype=np.float32)
    return {c: loaded.get(key_by_clip[c]) for c in clip_ids}


def probe_tmr_gt_motion(gt_sig, clip_id) -> bool:
    key = _key_from_path(tmr_gt_motion_path(gt_sig, clip_id), cache_root())
    return _v2_store().probe_many("tmr_gt_motion", [key])[key].exists


def save_tmr_gt_motion(gt_sig, clip_id, emb: np.ndarray) -> None:
    key = _key_from_path(tmr_gt_motion_path(gt_sig, clip_id), cache_root())
    _v2_store().put_many("tmr_gt_motion", [PutRecord(key=key, arrays={"array": np.asarray(emb, dtype=np.float32)})])


def save_tmr_gt_motion_batch(gt_sig, items: list[tuple[str, np.ndarray]]) -> int:
    """Batch-save TMR motion embeddings — one fsync per bucket.

    ``items`` is a list of ``(clip_id, embedding_array)`` tuples.
    """
    if not items:
        return 0
    root = cache_root()
    records = [
        PutRecord(
            key=_key_from_path(tmr_gt_motion_path(gt_sig, cid), root),
            arrays={"array": np.asarray(e, dtype=np.float32)},
        )
        for cid, e in items
    ]
    delta = _v2_store().put_many("tmr_gt_motion", records)
    return delta.written


def tmr_text_path(tmr_model: str, caption: str) -> Path:
    return cache_root() / "tmr_text" / safe(tmr_model) / f"{text_key(caption)}.npy"


def load_tmr_text(tmr_model, caption) -> np.ndarray | None:
    key = _key_from_path(tmr_text_path(tmr_model, caption), cache_root())
    item = _v2_store().load_one("tmr_text", key)
    return None if item is None else np.asarray(item.arrays["array"], dtype=np.float32)


def load_tmr_text_many(tmr_model, captions: list[str]) -> dict[str, np.ndarray | None]:
    """Batch-load TMR text embeddings for multiple captions."""
    root = cache_root()
    key_by_cap = {c: _key_from_path(tmr_text_path(tmr_model, c), root) for c in captions}
    loaded: dict[str, np.ndarray] = {}
    for batch in _v2_store().load_many("tmr_text", list(key_by_cap.values())):
        for item in batch:
            loaded[item.key] = np.asarray(item.arrays["array"], dtype=np.float32)
    return {c: loaded.get(key_by_cap[c]) for c in captions}


def probe_tmr_text(tmr_model, caption) -> bool:
    key = _key_from_path(tmr_text_path(tmr_model, caption), cache_root())
    return _v2_store().probe_many("tmr_text", [key])[key].exists


def save_tmr_text(tmr_model, caption, emb: np.ndarray) -> None:
    key = _key_from_path(tmr_text_path(tmr_model, caption), cache_root())
    _v2_store().put_many("tmr_text", [PutRecord(key=key, arrays={"array": np.asarray(emb, dtype=np.float32)})])


def list_missing_tmr_gt_motion(gt_sig: str, clip_ids: list[str]) -> list[str]:
    key_by_clip = {c: _key_from_path(tmr_gt_motion_path(gt_sig, c), cache_root()) for c in clip_ids}
    status = _v2_store().probe_many("tmr_gt_motion", list(key_by_clip.values()))
    return [c for c in clip_ids if not status[key_by_clip[c]].exists]


def list_missing_tmr_text(tmr_model: str, captions: list[str]) -> list[str]:
    key_by_caption = {c: _key_from_path(tmr_text_path(tmr_model, c), cache_root()) for c in captions}
    status = _v2_store().probe_many("tmr_text", list(key_by_caption.values()))
    return [c for c in captions if not status[key_by_caption[c]].exists]


# ---------------------------------------------------------------------------
# Volatile: generated-motion embedding (run-local)
# ---------------------------------------------------------------------------
def gen_motion_path(
    track: str,
    eval_sig: str,
    model_id: str,
    sig: str,
    clip_id: str,
    seed: int,
    cfg: float | None,
    *,
    dataset: str = "",
    run_root: str | Path | None = None,
) -> Path:
    """Canonical identity path for one gen-embedding (key source, not a literal file)."""
    return (
        _artifact_root(run_root)
        / "gen_emb"
        / safe(track)
        / safe(dataset)
        / safe(eval_sig)
        / f"{safe(model_id)}_{safe(sig)}"
        / f"{safe(clip_id)}_s{int(seed)}_cfg{cfg_str(cfg)}.npy"
    )


def load_gen_motion(
    track,
    eval_sig,
    model_id,
    sig,
    clip_id,
    seed,
    cfg,
    *,
    dataset="",
    run_root: str | Path | None = None,
) -> np.ndarray | None:
    return _run_store(run_root).load_gen_embedding(
        track, model_id, sig, clip_id, seed, cfg, dataset_sig=dataset, eval_sig=eval_sig,
    )


def load_gen_motion_many(
    track,
    eval_sig,
    model_id,
    sig,
    clip_ids: list[str],
    seed,
    cfg,
    *,
    dataset="",
    run_root: str | Path | None = None,
) -> dict[str, np.ndarray | None]:
    """Batch-load gen-motion embeddings for multiple clips of one model.

    Returns a dict mapping ``clip_id`` → embedding array (or None if missing).
    Uses :meth:`RunArtifactStore.load_many_gen_embeddings` for sequential per-bucket I/O.
    """
    from .run_artifact_store import gen_embedding_key as _gek

    key_by_clip = {
        c: _gek(track, model_id, sig, c, seed, cfg, dataset_sig=dataset, eval_sig=eval_sig)
        for c in clip_ids
    }
    store = _run_store(run_root)
    loaded: dict[str, np.ndarray] = {}
    for batch in store.load_many_gen_embeddings(list(key_by_clip.values())):
        for key, emb in batch:
            loaded[key] = emb
    return {c: loaded.get(key_by_clip[c]) for c in clip_ids}


def load_converted_many(
    model_id,
    sig,
    clip_ids: list[str],
    seed,
    cfg,
    target_rep,
    *,
    dataset="",
    run_root: str | Path | None = None,
) -> dict[str, "MotionClip | None"]:
    """Batch-load converted targets for multiple clips of one model.

    Returns a dict mapping ``clip_id`` → :class:`MotionClip` (or None if missing).
    """
    from .run_artifact_store import converted_key as _ck

    key_by_clip = {
        c: _ck(model_id, sig, c, seed, cfg, target_rep,
                dataset_sig=dataset, conversion_version=CONVERSION_GRAPH_VERSION)
        for c in clip_ids
    }
    store = _run_store(run_root)
    loaded: dict[str, object] = {}
    for batch in store.load_many_converted(list(key_by_clip.values())):
        for key, clip in batch:
            loaded[key] = clip
    return {c: loaded.get(key_by_clip[c]) for c in clip_ids}


def probe_gen_motion(
    track,
    eval_sig,
    model_id,
    sig,
    clip_id,
    seed,
    cfg,
    *,
    dataset="",
    run_root: str | Path | None = None,
) -> bool:
    key = gen_embedding_key(track, model_id, sig, clip_id, seed, cfg, dataset_sig=dataset, eval_sig=eval_sig)
    return _run_store(run_root).probe_gen_embedding_many([key])[key]


def probe_gen_motion_many(
    track,
    eval_sig,
    model_id,
    sig,
    clip_ids: list[str],
    seed,
    cfg,
    *,
    dataset="",
    run_root: str | Path | None = None,
) -> dict[str, bool]:
    """Bulk probe (one packed-store call) across ``clip_ids``."""
    key_by_clip = {
        c: gen_embedding_key(track, model_id, sig, c, seed, cfg, dataset_sig=dataset, eval_sig=eval_sig)
        for c in clip_ids
    }
    status = _run_store(run_root).probe_gen_embedding_many(list(key_by_clip.values()))
    return {c: status[key_by_clip[c]] for c in clip_ids}


def save_gen_motion(
    track,
    eval_sig,
    model_id,
    sig,
    clip_id,
    seed,
    cfg,
    emb: np.ndarray,
    *,
    dataset="",
    run_root: str | Path | None = None,
) -> None:
    _run_store(run_root).put_gen_embedding(
        track, model_id, sig, clip_id, seed, cfg, emb, dataset_sig=dataset, eval_sig=eval_sig,
    )


def save_gen_motion_many(
    track,
    eval_sig,
    model_id,
    sig,
    items: list[tuple[str, int, float | None, np.ndarray]],
    *,
    dataset="",
    run_root: str | Path | None = None,
) -> None:
    """Save multiple gen-motion embeddings in a single ``put_many`` call.

    Each item is ``(clip_id, seed, cfg, emb)``.
    """
    if not items:
        return
    store = _run_store(run_root)
    records = [
        PutRecord(
            key=gen_embedding_key(
                track, model_id, sig, clip_id, seed, cfg,
                dataset_sig=dataset, eval_sig=eval_sig,
            ),
            arrays={"array": np.asarray(emb, dtype=np.float32)},
            meta={},
        )
        for clip_id, seed, cfg, emb in items
    ]
    store.store.put_many("gen_emb", records)


__all__ = [
    "CONVERSION_GRAPH_VERSION",
    "GENERATION_PROTOCOL_VERSION",
    "GT_PROTOCOL_VERSION",
    "RUN_ARTIFACTS_DIR",
    "cache_root",
    "v2_root",
    "run_artifacts_root",
    "ckpt_sig",
    "dataset_sig",
    "hml_gt_sig",
    "tmr_gt_sig",
    "text_key",
    # Scope instances (preferred API)
    "native", "converted", "hml_gt_motion", "hml_gt_text",
    "tmr_gt_joints", "tmr_gt_motion", "tmr_text", "gen_motion",
    # Legacy flat functions (delegation wrappers, kept for backward compat)
    "load_native", "save_native", "save_native_many", "native_path", "probe_native", "probe_native_many",
    "load_converted", "save_converted", "save_converted_many", "converted_path", "probe_converted", "probe_converted_many",
    "load_hml_gt_motion", "save_hml_gt_motion", "hml_gt_motion_path", "probe_hml_gt_motion",
    "load_hml_gt_text", "save_hml_gt_text", "hml_gt_text_path", "probe_hml_gt_text",
    "list_missing_hml_gt_motion", "list_missing_hml_gt_text",
    "load_tmr_gt_joints", "save_tmr_gt_joints", "tmr_gt_joints_path", "probe_tmr_gt_joints",
    "load_tmr_gt_motion", "save_tmr_gt_motion", "tmr_gt_motion_path", "probe_tmr_gt_motion",
    "load_tmr_text", "save_tmr_text", "tmr_text_path", "probe_tmr_text",
    "list_missing_tmr_gt_motion", "list_missing_tmr_text",
    "load_gen_motion", "save_gen_motion", "save_gen_motion_many", "gen_motion_path", "probe_gen_motion", "probe_gen_motion_many",
]


# ---------------------------------------------------------------------------
# Artifact scopes — grouped cache operations for each artifact kind.
#
# Each scope bundles the probe/load/save functions for one artifact kind
# so callers can write ``C.native.probe(...)`` instead of remembering
# which of the 48 flat ``probe_X`` / ``load_X`` / ``save_X`` functions
# applies.  The flat function names remain as delegation wrappers for
# backward compatibility.
# ---------------------------------------------------------------------------

class ArtifactScope:
    """Grouped cache operations for one artifact kind.

    Usage::

        C.native.probe(model_id=..., clip_id=...)
        clip = C.native.load(model_id=..., clip_id=...)
        C.native.save(clip, model_id=..., clip_id=...)
    """

    __slots__ = ("name", "probe", "load", "save", "probe_many", "load_many", "save_many", "list_missing")

    def __init__(self, name, *, probe, load, save, probe_many=None, load_many=None, save_many=None, list_missing=None):
        self.name = name
        self.probe = probe
        self.load = load
        self.save = save
        self.probe_many = probe_many
        self.load_many = load_many
        self.save_many = save_many
        self.list_missing = list_missing

    def __repr__(self) -> str:
        return f"ArtifactScope({self.name!r})"


native = ArtifactScope("native",
    probe=probe_native, load=load_native, save=save_native,
    probe_many=probe_native_many, load_many=load_native_many, save_many=save_native_many)

converted = ArtifactScope("converted",
    probe=probe_converted, load=load_converted, save=save_converted,
    probe_many=probe_converted_many, load_many=load_converted_many, save_many=save_converted_many)

hml_gt_motion = ArtifactScope("hml_gt_motion",
    probe=probe_hml_gt_motion, load=load_hml_gt_motion, save=save_hml_gt_motion,
    load_many=load_hml_gt_motion_many, list_missing=list_missing_hml_gt_motion)

hml_gt_text = ArtifactScope("hml_gt_text",
    probe=probe_hml_gt_text, load=load_hml_gt_text, save=save_hml_gt_text,
    load_many=load_hml_gt_text_many, list_missing=list_missing_hml_gt_text)

tmr_gt_joints = ArtifactScope("tmr_gt_joints",
    probe=probe_tmr_gt_joints, load=load_tmr_gt_joints, save=save_tmr_gt_joints)

tmr_gt_motion = ArtifactScope("tmr_gt_motion",
    probe=probe_tmr_gt_motion, load=load_tmr_gt_motion, save=save_tmr_gt_motion,
    load_many=load_tmr_gt_motion_many, list_missing=list_missing_tmr_gt_motion)

tmr_text = ArtifactScope("tmr_text",
    probe=probe_tmr_text, load=load_tmr_text, save=save_tmr_text,
    load_many=load_tmr_text_many, list_missing=list_missing_tmr_text)

gen_motion = ArtifactScope("gen_motion",
    probe=probe_gen_motion, load=load_gen_motion, save=save_gen_motion,
    probe_many=probe_gen_motion_many, load_many=load_gen_motion_many, save_many=save_gen_motion_many)
