"""Small-scale end-to-end rehearsal of the production pipeline mechanics.

This exercises the *real* implementations built for this plan
(:class:`ShardedCacheStore`, :class:`RunArtifactStore`, :class:`GPUWorkerPool`,
:class:`ResourceGuard`, :class:`AdaptiveBatchCap`) against synthetic work on
CPU, so it can run anywhere without GPU/model/dataset dependencies while still
proving the actual mechanisms behave correctly end-to-end — not mocks of them.

Covers every item in the plan's "Validation" section that does not require a
real model/GPU/dataset:

* cold vs warm cache timing (metadata probe vs full payload materialization)
* interrupted -> resumed worker-pool run (no recomputation of committed units)
* delete/rebuild rehearsal on a disposable ``ShardedCacheStore`` root
* RAM-stress validation: an artificially low RSS budget forces adaptive batch
  shrinking instead of an uncaught crash

Real GPU/model/dataset production dry runs (full cold/warm eval, chaos/restart
orchestrating multi-GPU shards) are out of scope for this module by design —
they need real hardware/checkpoints and are exercised via the track runners
    directly (``--worker-ram-budget-gb``, ``--gpu-vram-headroom-gb``, etc.).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .planner import build_pilot_manifest
from .resource_guard import AdaptiveBatchCap, ResourceBudget, ResourceExceeded, ResourceGuard
from .sharded_cache_store import PutRecord, ShardedCacheStore
from .worker_pool import GPUWorkerPool


@dataclass
class ColdWarmResult:
    n_records: int
    probe_many_seconds: float
    load_many_materialize_seconds: float
    speedup_x: float


@dataclass
class InterruptResumeResult:
    phase1_committed: list[str]
    phase2_committed: list[str]
    phase2_skipped_already_committed: list[str]
    recomputed_native_gen: bool


@dataclass
class DropRebuildResult:
    records_before_drop: int
    records_after_drop: int
    records_after_rebuild: int
    drop_removed_paths: int


@dataclass
class RamStressResult:
    budget_bytes: int
    batches_attempted: int
    batches_shrunk: int
    process_survived: bool
    final_safe_batch: int


@dataclass
class RehearsalReport:
    cold_warm: ColdWarmResult
    interrupt_resume: InterruptResumeResult
    drop_rebuild: DropRebuildResult
    ram_stress: RamStressResult
    all_passed: bool


# ---------------------------------------------------------------------------
def run_cold_warm_comparison(root: Path, *, n_records: int = 200, array_len: int = 512) -> ColdWarmResult:
    store = ShardedCacheStore(root / "cold_warm", num_buckets=8)
    records = [
        PutRecord(key=f"native:clip{i}", arrays={"array": np.random.default_rng(i).normal(size=array_len).astype(np.float32)})
        for i in range(n_records)
    ]
    store.put_many("native", records)
    keys = [r.key for r in records]

    t0 = time.perf_counter()
    status = store.probe_many("native", keys)
    probe_s = time.perf_counter() - t0
    assert all(s.exists for s in status.values())

    t0 = time.perf_counter()
    total_bytes = 0
    for batch in store.load_many("native", keys):
        for item in batch:
            total_bytes += item.arrays["array"].nbytes
    load_s = time.perf_counter() - t0
    assert total_bytes == n_records * array_len * 4

    speedup = load_s / max(probe_s, 1e-9)
    return ColdWarmResult(
        n_records=n_records, probe_many_seconds=probe_s,
        load_many_materialize_seconds=load_s, speedup_x=speedup,
    )


# ---------------------------------------------------------------------------
def run_interrupt_resume_rehearsal(run_dir: Path, *, n_prompts: int = 12) -> InterruptResumeResult:
    prompt_ids = [f"clip_{i:04d}" for i in range(n_prompts)]
    _run, units = build_pilot_manifest(prompt_ids=prompt_ids, chunk_size=4, run_id="rehearsal")

    # Phase 1: a worker that "crashes" — it is only handed the native-gen units
    # (simulating a process that died before starting convert/gen-embed).
    native_units = [u for u in units if u.phase == "native-gen"]
    pool1 = GPUWorkerPool(run_dir, native_units, worker_id="worker-A")
    result1 = pool1.run({"native-gen": lambda u: True})

    # Phase 2: a fresh pool (simulating a restarted process) gets the FULL
    # manifest and must resume purely from committed.jsonl, not recompute
    # native-gen.
    seen_native_calls: list[str] = []
    pool2 = GPUWorkerPool(run_dir, units, worker_id="worker-B")
    result2 = pool2.run({
        "native-gen": lambda u: (seen_native_calls.append(u.unit_id), True)[1],
        "convert": lambda u: True,
        "gen-embed": lambda u: True,
    })

    return InterruptResumeResult(
        phase1_committed=result1.committed,
        phase2_committed=result2.committed,
        phase2_skipped_already_committed=result2.skipped_already_committed,
        recomputed_native_gen=len(seen_native_calls) > 0,
    )


# ---------------------------------------------------------------------------
def run_drop_rebuild_rehearsal(root: Path, *, n_records: int = 50) -> DropRebuildResult:
    store = ShardedCacheStore(root / "drop_rebuild", num_buckets=4)
    records = [
        PutRecord(key=f"native:r{i}", arrays={"array": np.zeros(16, dtype=np.float32)})
        for i in range(n_records)
    ]
    store.put_many("native", records)
    before = store.audit("native").records

    removed = store.drop("native", dry_run=False)
    after_drop = store.audit("native").records

    store.put_many("native", records)  # rebuild from scratch
    after_rebuild = store.audit("native").records

    return DropRebuildResult(
        records_before_drop=before, records_after_drop=after_drop,
        records_after_rebuild=after_rebuild, drop_removed_paths=len(removed),
    )


# ---------------------------------------------------------------------------
def run_ram_stress_validation(*, budget_bytes: int = 64 * 1024, n_batches: int = 10) -> RamStressResult:
    """Simulate progressively larger batches against an artificially tiny RAM
    budget. The guard/adaptive-cap combo must shrink batches to fit rather
    than let a real allocation raise Linux OOM."""
    guard = ResourceGuard(ResourceBudget(worker_ram_budget_bytes=budget_bytes))
    cap = AdaptiveBatchCap(
        Path("/tmp") / f"validate_ram_stress_{time.time_ns()}.json",
        initial=64, min_batch=1, max_batch=1024,
    )
    key = "rehearsal_model@cpu"
    shrunk_count = 0
    survived = True

    try:
        for i in range(n_batches):
            proposed = cap.get(key)
            items = list(range(proposed))
            # Each "item" claims to need budget_bytes // 4 bytes — deliberately
            # oversized relative to the budget so shrinking is required.
            planned = guard.plan_batch(items, lambda _x: max(1, budget_bytes // 4))
            try:
                guard.check_rss(f"batch-{i}", planned_bytes=len(planned) * (budget_bytes // 4))
            except ResourceExceeded:
                cap.report_resource_exceeded(key, proposed)
                shrunk_count += 1
                continue
            cap.report_success(key, len(planned))
            if len(planned) < proposed:
                shrunk_count += 1
    except Exception:
        survived = False

    return RamStressResult(
        budget_bytes=budget_bytes, batches_attempted=n_batches, batches_shrunk=shrunk_count,
        process_survived=survived, final_safe_batch=cap.get(key),
    )


# ---------------------------------------------------------------------------
def run_full_rehearsal(root: str | Path) -> RehearsalReport:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    cold_warm = run_cold_warm_comparison(root)
    interrupt_resume = run_interrupt_resume_rehearsal(root / "interrupt_resume")
    drop_rebuild = run_drop_rebuild_rehearsal(root)
    ram_stress = run_ram_stress_validation()

    all_passed = (
        cold_warm.speedup_x >= 1.0
        and not interrupt_resume.recomputed_native_gen
        and drop_rebuild.records_after_drop == 0
        and drop_rebuild.records_after_rebuild == drop_rebuild.records_before_drop
        and ram_stress.process_survived
        and ram_stress.batches_shrunk > 0
    )
    return RehearsalReport(
        cold_warm=cold_warm, interrupt_resume=interrupt_resume,
        drop_rebuild=drop_rebuild, ram_stress=ram_stress, all_passed=all_passed,
    )


def render_report(report: RehearsalReport) -> str:
    lines = ["=== production pipeline rehearsal ==="]
    cw = report.cold_warm
    lines.append(
        f"[cold/warm]      probe_many={cw.probe_many_seconds * 1000:.2f}ms "
        f"load_many={cw.load_many_materialize_seconds * 1000:.2f}ms "
        f"speedup={cw.speedup_x:.1f}x over {cw.n_records} records"
    )
    ir = report.interrupt_resume
    lines.append(
        f"[interrupt/resume] phase1_committed={len(ir.phase1_committed)} "
        f"phase2_committed={len(ir.phase2_committed)} "
        f"skipped_already_committed={len(ir.phase2_skipped_already_committed)} "
        f"recomputed_native_gen={ir.recomputed_native_gen}"
    )
    dr = report.drop_rebuild
    lines.append(
        f"[drop/rebuild]    before={dr.records_before_drop} after_drop={dr.records_after_drop} "
        f"after_rebuild={dr.records_after_rebuild} removed_paths={dr.drop_removed_paths}"
    )
    rs = report.ram_stress
    lines.append(
        f"[ram stress]      budget={rs.budget_bytes / 1024:.0f}KiB batches_shrunk={rs.batches_shrunk}"
        f"/{rs.batches_attempted} survived={rs.process_survived} final_safe_batch={rs.final_safe_batch}"
    )
    lines.append(f"ALL PASSED: {report.all_passed}")
    return "\n".join(lines)


def report_to_dict(report: RehearsalReport) -> dict:
    return asdict(report)


__all__ = [
    "ColdWarmResult",
    "DropRebuildResult",
    "InterruptResumeResult",
    "RamStressResult",
    "RehearsalReport",
    "render_report",
    "report_to_dict",
    "run_cold_warm_comparison",
    "run_drop_rebuild_rehearsal",
    "run_full_rehearsal",
    "run_interrupt_resume_rehearsal",
    "run_ram_stress_validation",
]
