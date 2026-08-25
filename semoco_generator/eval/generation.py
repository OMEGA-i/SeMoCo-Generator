"""Generation + conversion shared by both tracks.

* ``ensure_native``: durable shared native cache (model + clip + seed + cfg).
  Same prompts reuse generations across HML / TMR / smoke / full runs.
* ``ensure_target``: convert native -> track target into *run-local* artifacts
  under ``run_root`` (depends on conversion graph; not durable truth).

Stable GT / evaluator embeddings live in ``eval.cache`` under ``cache_root()``.
Scoring reads converted targets via ``load_generated`` from run-local storage.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from .artifacts import EvaluationArtifacts
from .cache_async import AsyncCacher
from .conversions import ConversionContext, ConversionGraph
from .models.base import MotionModel
from .schema import ModelInput, MotionRep, TrackInput


def _shard(track_inputs: Sequence[TrackInput], shard_index: int, num_shards: int) -> list[TrackInput]:
    return [c for i, c in enumerate(track_inputs) if i % num_shards == shard_index]


# Max clips per conversion chunk in ensure_target.  Higher values amortise GPU
# kernel-launch overhead across more clips, but also increase peak CPU memory
# for intermediates (SMPL vertices, SOMA features).  500 clips keeps peak CPU
# intermediates under ~4 GB for the heaviest conversion path (smpl_rot6d_transl
# → smpl_vertices → soma77).
_CONVERT_CHUNK_SIZE = 500


def ensure_native(
    model: MotionModel,
    track_inputs: Sequence[TrackInput],
    *,
    seeds: Sequence[int],
    cfg_scale: float | None,
    dataset_sig: str = "",
    shard_index: int = 0,
    num_shards: int = 1,
    skip_existing: bool = True,
    batch_size: int = 8,
    failures_path: str | Path | None = None,
    run_root: str | Path | None = None,  # ignored — native always goes to shared cache
) -> None:
    """Fill the shared native cache for the selected shard.

    Effective batch size is clamped to ``model.schema.max_safe_batch`` so callers
    can keep a large default while preserving deterministic, batch-invariant
    generation semantics.
    """
    sig = model.weight_signature()
    mid = model.schema.model_id
    artifacts = EvaluationArtifacts.for_run(run_root, dataset_sig=dataset_sig)
    native_artifacts = artifacts.model(mid, sig).native
    selected = _shard(track_inputs, shard_index, num_shards)
    safe = max(1, int(getattr(model.schema, "max_safe_batch", 1) or 1))
    batch_size = max(1, min(int(batch_size), safe))
    cacher = AsyncCacher()

    for seed in seeds:
        if skip_existing:
            warm = native_artifacts.probe_many([c.prompt_id for c in selected], int(seed), cfg_scale)
            pending = [c for c in selected if not warm[c.prompt_id]]
        else:
            pending = list(selected)
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            inputs = [
                ModelInput(
                    prompt_id=c.prompt_id, text=c.caption, length=c.length,
                    seed=int(seed), cfg_scale=cfg_scale,
                )
                for c in batch
            ]
            t0 = time.perf_counter()
            try:
                outputs = model.generate(inputs)
            except Exception as exc:  # keep partial runs alive
                _log(failures_path, mid, [c.prompt_id for c in batch], int(seed), stage="generation", exc=exc)
                print(
                    f"[{mid}] batch gen failed seed={seed} size={len(batch)}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            out_by_id = {o.prompt_id: o for o in outputs}
            ok = 0
            units = 0
            native_batch: list[tuple[str, int, float | None, object]] = []
            for c in batch:
                out = out_by_id.get(c.prompt_id)
                if out is None or out.native_motion is None or out.status != "ok":
                    _log(
                        failures_path, mid, [c.prompt_id], int(seed), stage="generation",
                        msg=(out.error if out else "no output"),
                    )
                    continue
                ok += 1
                units += int(getattr(out.native_motion, "num_frames", 0) or 0)
                native_batch.append((c.prompt_id, int(seed), cfg_scale, out.native_motion))
            if native_batch:
                cacher.submit(native_artifacts.save_many, native_batch)
            dt = max(time.perf_counter() - t0, 1e-9)
            print(
                f"[{mid}] native seed={seed} {start + len(batch)}/{len(pending)} "
                f"(bs={batch_size} ok={ok}/{len(batch)} time={dt:.2f}s "
                f"clips/s={ok / dt:.2f} units/s={units / dt:.2f})",
                flush=True,
            )

    cacher.flush_all()


def _duration_s(c: TrackInput) -> float | None:
    if c.length.seconds is not None:
        return float(c.length.seconds)
    return None


def ensure_target(
    model: MotionModel,
    track_inputs: Sequence[TrackInput],
    *,
    target_rep: MotionRep,
    graph: ConversionGraph,
    ctx: ConversionContext,
    seeds: Sequence[int],
    cfg_scale: float | None,
    dataset_sig: str = "",
    shard_index: int = 0,
    num_shards: int = 1,
    skip_existing: bool = True,
    failures_path: str | Path | None = None,
    run_root: str | Path | None = None,
) -> list[str]:
    """Convert shared native -> run-local target using batched conversion.

    Clips are grouped by native representation and converted in a single
    ``graph.convert_batch()`` call per group, amortizing GPU kernel-launch
    overhead for SMPL / SOMA / fitting edges.
    """
    sig = model.weight_signature()
    mid = model.schema.model_id
    artifacts = EvaluationArtifacts.for_run(run_root, dataset_sig=dataset_sig)
    model_artifacts = artifacts.model(mid, sig)
    native_artifacts = model_artifacts.native
    converted = model_artifacts.converted(target_rep)
    selected = _shard(track_inputs, shard_index, num_shards)
    completed: list[str] = []
    cacher = AsyncCacher()

    for seed in seeds:
        # Collect clips that need conversion + their natives
        pending: list[tuple[TrackInput, MotionClip]] = []
        converted_warm: dict[str, bool] = {}
        if skip_existing:
            converted_warm = converted.probe_many(
                [c.prompt_id for c in selected], int(seed), cfg_scale,
            )

        # Collect (clip_id, seed, cfg) tuples for batch load
        _to_load: list[tuple[str, int, float | None]] = []
        _load_clips: list[TrackInput] = []
        for c in selected:
            if skip_existing and converted_warm[c.prompt_id]:
                if int(seed) == int(seeds[0]):
                    completed.append(c.prompt_id)
                continue
            _to_load.append((c.prompt_id, int(seed), cfg_scale))
            _load_clips.append(c)

        # Batch-load all natives in one call (reads pack files sequentially per bucket)
        _natives = native_artifacts.load_many(_to_load) if _to_load else {}

        for c in _load_clips:
            _key = (c.prompt_id, int(seed), cfg_scale)
            native = _natives.get(_key)
            if native is None:
                _log(failures_path, mid, [c.prompt_id], int(seed), stage="conversion", msg="native cache miss")
                continue
            # prompt_id is conversion context, not part of the native motion
            # payload. Attach it only in memory so Semoco can select a per-clip
            # decode anchor without invalidating or changing durable native
            # cache records.
            if native.rep == "motion_codes":
                native.aux = dict(native.aux)
                native.aux["prompt_id"] = c.prompt_id  # type: ignore[assignment]
            pending.append((c, native))

        if not pending:
            continue

        total_by_rep: dict[str, int] = {}
        for chunk_start in range(0, len(pending), _CONVERT_CHUNK_SIZE):
            chunk = pending[chunk_start:chunk_start + _CONVERT_CHUNK_SIZE]

            # Group by native representation (one model = one rep, but be defensive)
            by_rep: dict[str, list[tuple[TrackInput, MotionClip]]] = {}
            for c, native in chunk:
                by_rep.setdefault(native.rep, []).append((c, native))

            for rep, group in by_rep.items():
                total_by_rep[rep] = total_by_rep.get(rep, 0) + len(group)
                group_inputs = [t[0] for t in group]
                group_natives = [t[1] for t in group]
                t0 = time.perf_counter()
                try:
                    results = graph.convert_batch(group_natives, target_rep, ctx)
                except Exception as exc:
                    _log(failures_path, mid, [c.prompt_id for c in group_inputs],
                         int(seed), stage="conversion_batch", exc=exc)
                    print(
                        f"[{mid}] batch convert failed seed={seed} rep={rep} size={len(group)}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue

                units = 0
                converted_batch: list[tuple[str, int, float | None, MotionClip]] = []
                for target_clip, ci in zip(results, group_inputs):
                    units += int(getattr(target_clip, "num_frames", 0) or 0)
                    converted_batch.append((ci.prompt_id, int(seed), cfg_scale, target_clip))
                    if int(seed) == int(seeds[0]):
                        completed.append(ci.prompt_id)
                if converted_batch:
                    cacher.submit(converted.save_many, converted_batch)
                dt = max(time.perf_counter() - t0, 1e-9)
                print(
                    f"[{mid}] convert seed={seed} rep={rep} size={len(group)} "
                    f"time={dt:.2f}s clips/s={len(group) / dt:.2f} units/s={units / dt:.2f}",
                    flush=True,
                )

        if pending:
            print(
                f"[{mid}] converted seed={seed} {len(pending)} clips (chunked {_CONVERT_CHUNK_SIZE}, {len(total_by_rep)} rep groups)",
                flush=True,
            )

        cacher.flush_all()

    return completed


def load_generated(
    model: MotionModel,
    clip_id: str,
    seed: int,
    cfg_scale: float | None,
    target_rep: MotionRep,
    *,
    dataset_sig: str = "",
    run_root: str | Path | None = None,
) -> tuple[np.ndarray, float] | None:
    """Load a converted target (array, fps) from run-local artifacts, if present."""
    artifacts = EvaluationArtifacts.for_run(run_root, dataset_sig=dataset_sig)
    clip = artifacts.model(model.schema.model_id, model.weight_signature()).converted(target_rep).load(
        clip_id, int(seed), cfg_scale,
    )
    if clip is None:
        return None
    return np.asarray(clip.array, dtype=np.float32), float(clip.fps)


def _log(path, model, prompt_ids, seed, *, stage, exc=None, msg=None) -> None:
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    detail = msg or (f"{type(exc).__name__}: {exc}" if exc else "unknown")
    with p.open("a") as f:
        for pid in prompt_ids:
            f.write(
                json.dumps({
                    "stage": stage, "model": model, "prompt_id": pid,
                    "seed": int(seed), "error": detail,
                })
                + "\n"
            )


__all__ = ["ensure_native", "ensure_target", "load_generated"]
