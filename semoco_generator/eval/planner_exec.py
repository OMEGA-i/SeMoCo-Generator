"""Wires :mod:`planner` + :mod:`worker_pool` into a track runner's real
per-model ``native-gen`` / ``convert`` / ``gen-embed`` execution.

Track runners (``tracks/smpl_hml/runner.py``, ``tracks/soma_tmr/runner.py``)
build small closures over their own ``ensure_native`` / ``ensure_target`` /
``_encode_gen_shard`` calls and hand them to :func:`run_model_with_planner`,
which does the scheduling-specific plumbing: build-or-read the shared
manifest, filter work units to one model, translate
:class:`resource_guard.ResourceExceeded` into the pool's retry-at-smaller-
batch signal, and run the lease/commit state machine.

Multiple GPU-pinned worker *processes* can call this against the same
``out_root`` concurrently (e.g. one per physical GPU): :class:`WorkUnit`
identities are derived deterministically from ``run_id`` (the protocol id,
not a timestamp), so every process reads back an identical manifest and
dynamically leases whichever units are still unclaimed — no static
shard-index partitioning required.
"""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .planner import TrackPromptCost, WorkUnit, build_track_manifest, read_manifest, write_manifest
from .resource_guard import ResourceExceeded as GuardResourceExceeded
from .worker_pool import GPUWorkerPool, ResourceExceeded as PoolResourceExceeded, WorkerPoolResult

PhaseFn = Callable[[WorkUnit], bool]


def build_or_read_track_manifest(
    *,
    out_root: Path,
    track: str,
    models: Sequence[str],
    split: str,
    dataset: str,
    prompts: Sequence[TrackPromptCost],
    num_shards: int,
    run_id: str,
) -> list[WorkUnit]:
    """Deterministic manifest shared by every worker process for this
    protocol run. Built once under an exclusive ``flock`` (the first process
    to see a missing ``run.json`` builds and writes it; every other process
    blocks on the same lock rather than racing to read a manifest that's
    still mid-write, then finds ``run.json`` already there and skips
    straight to reading). The same ``run_id`` + ``prompts`` always produce
    byte-identical LPT bins, so unit_ids line up exactly across processes
    regardless of who ends up doing the actual build."""
    manifest_dir = out_root / "planner"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_dir / ".manifest_build.lock"
    with lock_path.open("a") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            if not (manifest_dir / "run.json").is_file():
                run, units = build_track_manifest(
                    track=track, models=list(models), split=split, dataset=dataset,
                    prompts=list(prompts), num_shards=num_shards, run_id=run_id,
                )
                write_manifest(manifest_dir, run, units)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    _run, units = read_manifest(manifest_dir)
    return units


def _guard_wrapped(fn: PhaseFn) -> PhaseFn:
    def wrapped(unit: WorkUnit) -> bool:
        try:
            return fn(unit)
        except GuardResourceExceeded as exc:
            # Re-raised as the pool's own exception type so it takes the
            # retry-at-smaller-batch path instead of an immediate quarantine.
            raise PoolResourceExceeded(str(exc)) from exc
    return wrapped


def run_model_with_planner(
    *,
    out_root: Path,
    model_id: str,
    units: list[WorkUnit],
    native_fn: PhaseFn,
    convert_fn: PhaseFn,
    embed_fn: PhaseFn,
    worker_id: str | None = None,
    lease_ttl: float = 600.0,
    max_retries: int = 2,
) -> WorkerPoolResult:
    """Run one model's native-gen -> convert -> gen-embed chain through the
    lease/commit worker pool. Only units for ``model_id`` are passed to the
    pool, so this is safe to call once per model in the existing "load model,
    do its work, close model" loop without pulling in other models' units."""
    model_units = [u for u in units if u.model == model_id]
    pool = GPUWorkerPool(
        out_root / "planner_state", model_units,
        worker_id=worker_id or f"worker-{os.getpid()}", lease_ttl=lease_ttl, max_retries=max_retries,
    )
    phase_fns = {
        "native-gen": _guard_wrapped(native_fn),
        "convert": _guard_wrapped(convert_fn),
        "gen-embed": _guard_wrapped(embed_fn),
    }
    return pool.run(phase_fns)


@dataclass
class LoadedModel:
    """One track's lazily-loaded model handle plus whatever per-model state
    the native/convert/embed phase closures need (conversion context,
    effective cfg-scale, etc). Fields beyond ``model`` are intentionally
    free-form (``extra``) since HML/TMR tracks need different bits."""

    model: Any
    ctx: Any = None
    eff_cfg: float | None = None
    extra: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


ModelLoaderFn = Callable[[str], LoadedModel]
PhaseFnFactory = Callable[[LoadedModel], PhaseFn]


def run_global_planner(
    *,
    out_root: Path,
    units: list[WorkUnit],
    model_loader: ModelLoaderFn,
    native_fn_factory: PhaseFnFactory,
    convert_fn_factory: PhaseFnFactory,
    embed_fn_factory: PhaseFnFactory,
    worker_id: str | None = None,
    lease_ttl: float = 600.0,
    max_retries: int = 2,
    on_model_closed: Callable[[str, LoadedModel], None] | None = None,
) -> WorkerPoolResult:
    """Global scheduler spanning *every* model's units in one lease/commit
    pool — this is what actually fixes GPU idling under multi-model tracks.

    The previous per-model ``run_model_with_planner()`` call only ever leased
    units for the ONE model a runner's outer loop had already loaded, so a
    worker that ran out of ready work for that model would exit even while
    heavy units for a different (still-unclaimed) model sat ready. Here,
    ``units`` spans every model in the track's manifest; models are loaded
    lazily (only when a leased unit actually needs them) and cached for the
    remainder of this call, so a worker naturally keeps leasing whatever
    ready unit — light or heavy, any model — is next by
    :func:`~semoco_generator.eval.planner.unit_priority` order instead of
    idling once its "current" model runs dry.

    At most one model is resident (loaded) at a time per call — the same
    invariant the old per-model loop had. Switching to a different model
    closes whatever was previously loaded *first*: without this, a worker
    that touches several models over the life of one ``run_global_planner``
    call would accumulate every one of their weights in VRAM simultaneously
    (they'd only ever get closed in the final ``finally`` below), which is a
    real CUDA-OOM risk on top of whatever headroom :class:`ResourceGuard`
    assumed for a single model. In practice workers stick to one model for
    long stretches anyway, since :func:`~semoco_generator.eval.planner.unit_priority`
    sorts same-model units together, so this rarely forces extra reloads.
    """
    handles: dict[str, LoadedModel] = {}
    phase_cache: dict[str, dict[str, PhaseFn]] = {}

    def _close_model(model_id: str) -> None:
        handle = handles.pop(model_id, None)
        phase_cache.pop(model_id, None)
        if handle is None:
            return
        close = getattr(handle.model, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 — best-effort cleanup, never mask the real result
                pass
        if on_model_closed is not None:
            try:
                on_model_closed(model_id, handle)
            except Exception:  # noqa: BLE001
                pass

    def _dispatch(unit: WorkUnit) -> bool:
        if unit.model not in handles:
            for stale_model_id in list(handles.keys()):
                _close_model(stale_model_id)
            handle = model_loader(unit.model)
            handles[unit.model] = handle
            phase_cache[unit.model] = {
                "native-gen": native_fn_factory(handle),
                "convert": convert_fn_factory(handle),
                "gen-embed": embed_fn_factory(handle),
            }
        fn = phase_cache[unit.model][unit.phase]
        return fn(unit)

    pool = GPUWorkerPool(
        out_root / "planner_state", units,
        worker_id=worker_id or f"worker-{os.getpid()}", lease_ttl=lease_ttl, max_retries=max_retries,
    )
    dispatch = _guard_wrapped(_dispatch)
    phase_fns = {"native-gen": dispatch, "convert": dispatch, "gen-embed": dispatch}
    try:
        return pool.run(phase_fns)
    finally:
        for model_id in list(handles.keys()):
            _close_model(model_id)


__all__ = [
    "LoadedModel",
    "ModelLoaderFn",
    "PhaseFn",
    "PhaseFnFactory",
    "build_or_read_track_manifest",
    "run_global_planner",
    "run_model_with_planner",
]
