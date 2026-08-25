"""Tests for planner_exec: wiring EvalPlanner + GPUWorkerPool into a track
runner's real per-model phase functions (mocked phase work, no real GPU)."""

from __future__ import annotations

import json
import multiprocessing
import threading
from pathlib import Path

from semoco_generator.eval.planner import TrackPromptCost
from semoco_generator.eval.planner_exec import build_or_read_track_manifest, run_model_with_planner
from semoco_generator.eval.resource_guard import ResourceExceeded as GuardResourceExceeded
from semoco_generator.eval.resource_guard import ResourceExceededRecord
from semoco_generator.eval.worker_pool import WorkQueueStore


def _prompts(n: int) -> list[TrackPromptCost]:
    return [TrackPromptCost(f"clip_{i:03d}", float(2 + i % 5)) for i in range(n)]


def test_build_or_read_track_manifest_is_idempotent(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    kwargs = dict(
        out_root=d, track="smpl_hml", models=["semoco", "baseline1"], split="test",
        dataset="HumanML3D", prompts=_prompts(12), num_shards=3, run_id="proto-stable-1",
    )
    units_a = build_or_read_track_manifest(**kwargs)
    units_b = build_or_read_track_manifest(**kwargs)  # second call must read back, not rebuild
    assert [u.unit_id for u in units_a] == [u.unit_id for u in units_b]
    assert {u.model for u in units_a} == {"semoco", "baseline1"}
    assert (d / "planner" / "run.json").is_file()
    assert (d / "planner" / "work_units.jsonl").is_file()
    print("build_or_read_track_manifest idempotent OK")


def _build_manifest_worker(out_root: str, models: list[str], n_prompts: int, run_id: str) -> int:
    """Module-level (picklable) target for a real OS process racing to build
    the manifest concurrently with siblings. Returns the unit count it read
    back, mirroring exactly what a real GPU worker does at startup."""
    units = build_or_read_track_manifest(
        out_root=Path(out_root), track="smpl_hml", models=models, split="test",
        dataset="HumanML3D", prompts=_prompts(n_prompts), num_shards=48, run_id=run_id,
    )
    return len(units)


def test_build_or_read_track_manifest_survives_concurrent_processes(tmp_path: Path | None = None):
    """Regression test for the real-world bug: 3 GPU worker *processes*
    (--shard-index 0/1/2, --scheduler planner) all call
    build_or_read_track_manifest for the same run_id at ~the same time. Before
    the fcntl.flock + atomic-rename fix, whichever process didn't win the
    "build" race could read run.json the instant it existed but before
    work_units.jsonl was fully flushed, and silently get back 0 units (the
    process would then log "0 work units" and exit immediately, leaving that
    GPU idle for the rest of the run). Every one of N racing processes must
    see the full, byte-identical manifest -- never a partial read."""
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    models = ["baseline2", "baseline3", "baseline1", "baseline5", "baseline4", "semoco"]
    n_prompts = 400  # large enough that writing work_units.jsonl is not instantaneous
    run_id = "proto-race-1"
    expected = len(models) * 48 * 3  # models x LPT bins x (native-gen, convert, gen-embed)

    n_procs = 6
    with multiprocessing.get_context("fork").Pool(n_procs) as pool:
        results = pool.starmap(
            _build_manifest_worker,
            [(str(d), models, n_prompts, run_id) for _ in range(n_procs)],
        )
    assert results == [expected] * n_procs, (
        f"every racing process must read back all {expected} units, got {results}"
    )
    print("build_or_read_track_manifest survives concurrent racing processes OK")


def test_run_model_with_planner_filters_by_model_and_commits(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    units = build_or_read_track_manifest(
        out_root=d, track="smpl_hml", models=["semoco", "baseline1"], split="test",
        dataset="HumanML3D", prompts=_prompts(8), num_shards=2, run_id="proto-filter-1",
    )
    calls: list[str] = []

    def make_fn(phase: str):
        def fn(unit) -> bool:
            calls.append(f"{phase}:{unit.model}:{unit.unit_id}")
            return True
        return fn

    result = run_model_with_planner(
        out_root=d, model_id="semoco", units=units,
        native_fn=make_fn("native"), convert_fn=make_fn("convert"), embed_fn=make_fn("embed"),
        worker_id="test-worker",
    )
    assert all(":baseline1:" not in c for c in calls)  # only the requested model ran
    assert any(":semoco:" in c for c in calls)
    assert len(result.committed) == sum(1 for u in units if u.model == "semoco")
    assert not result.quarantined and not result.blocked
    print("run_model_with_planner filters by model + commits OK")


def test_run_model_with_planner_translates_resource_exceeded_to_retry(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    units = build_or_read_track_manifest(
        out_root=d, track="smpl_hml", models=["semoco"], split="test",
        dataset="HumanML3D", prompts=_prompts(4), num_shards=1, run_id="proto-oom-1",
    )
    attempts = {"native": 0}

    def flaky_native(unit) -> bool:
        attempts["native"] += 1
        if attempts["native"] == 1:
            rec = ResourceExceededRecord(stage="native-gen", kind="rss", detail="fake OOM", at=0.0)
            raise GuardResourceExceeded("fake OOM", rec)
        return True

    result = run_model_with_planner(
        out_root=d, model_id="semoco", units=units,
        native_fn=flaky_native, convert_fn=lambda u: True, embed_fn=lambda u: True,
        worker_id="test-worker", max_retries=2,
    )
    # First native-gen attempt raised resource_guard.ResourceExceeded -> the
    # pool's own ResourceExceeded retry path (not an immediate quarantine) ->
    # second attempt succeeds -> whole chain commits.
    assert attempts["native"] == 2
    assert not result.quarantined
    native_id = next(u.unit_id for u in units if u.phase == "native-gen")
    assert native_id in result.retried
    assert native_id in result.committed
    print("run_model_with_planner resource_exceeded -> retry (not quarantine) OK")


def test_concurrent_ledger_appends_stay_valid_jsonl(tmp_path: Path | None = None):
    """Real thread-level concurrency smoke test for the flock hardening added
    to WorkQueueStore._append_jsonl: many concurrent writers must never
    interleave into a corrupt/partial JSON line."""
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    store = WorkQueueStore(d)
    n_threads = 8
    n_each = 25

    def writer(idx: int) -> None:
        for i in range(n_each):
            store.record_failure(f"unit-{idx}-{i}", f"worker-{idx}", detail="x" * 200)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = store.failures_path.read_text().strip().splitlines()
    assert len(lines) == n_threads * n_each
    for line in lines:
        json.loads(line)  # raises if any line is corrupt/partial
    print("concurrent ledger appends stay valid JSONL OK")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_build_or_read_track_manifest_is_idempotent(d)
    with tempfile.TemporaryDirectory() as d:
        test_build_or_read_track_manifest_survives_concurrent_processes(d)
    with tempfile.TemporaryDirectory() as d:
        test_run_model_with_planner_filters_by_model_and_commits(d)
    with tempfile.TemporaryDirectory() as d:
        test_run_model_with_planner_translates_resource_exceeded_to_retry(d)
    with tempfile.TemporaryDirectory() as d:
        test_concurrent_ledger_appends_stay_valid_jsonl(d)
    print("\nALL PLANNER EXEC TESTS PASSED")
