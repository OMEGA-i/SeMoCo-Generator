"""Tests for the GPUWorkerPool lease/commit resumable state machine."""

from __future__ import annotations

from pathlib import Path

from semoco_generator.eval.planner import WorkUnit
from semoco_generator.eval.worker_pool import GPUWorkerPool, ResourceExceeded, WorkQueueStore


def _chain_units() -> list[WorkUnit]:
    return [
        WorkUnit("u-native", "smpl_hml", "semoco", "native-gen", ["c1", "c2"], 2, []),
        WorkUnit("u-convert", "smpl_hml", "semoco", "convert", ["c1", "c2"], 2, ["u-native"]),
        WorkUnit("u-embed", "smpl_hml", "semoco", "gen-embed", ["c1", "c2"], 2, ["u-convert"]),
    ]


def test_dependency_order_is_respected(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    pool = GPUWorkerPool(d, _chain_units())
    order: list[str] = []

    def make_fn(name):
        def fn(unit):
            order.append(unit.unit_id)
            return True
        return fn

    result = pool.run({"native-gen": make_fn("n"), "convert": make_fn("c"), "gen-embed": make_fn("e")})
    assert result.committed == ["u-native", "u-convert", "u-embed"]
    assert order == ["u-native", "u-convert", "u-embed"]
    print("dependency order respected OK")


def test_resume_skips_already_committed_units(tmp_path: Path | None = None):
    """Worker A commits native-gen only (e.g. it was killed before starting
    convert). Worker B resumes with the *full* unit set against the same
    run_dir and must not redo native-gen — only the manifest commit record
    decides that, never logs or process state."""
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    units = _chain_units()

    calls_1: list[str] = []
    pool1 = GPUWorkerPool(d, [units[0]])  # this worker only owns native-gen
    pool1.run({"native-gen": lambda u: (calls_1.append(u.unit_id), True)[1]})
    assert calls_1 == ["u-native"]

    calls_2: list[str] = []
    pool2 = GPUWorkerPool(d, units)  # fresh pool, full unit set, simulates resumed run
    result2 = pool2.run({
        "native-gen": lambda u: (calls_2.append(u.unit_id), True)[1],
        "convert": lambda u: (calls_2.append(u.unit_id), True)[1],
        "gen-embed": lambda u: (calls_2.append(u.unit_id), True)[1],
    })
    assert "u-native" not in calls_2  # already committed, must not re-run
    assert calls_2 == ["u-convert", "u-embed"]
    assert result2.committed == ["u-convert", "u-embed"]
    assert result2.skipped_already_committed == ["u-native"]
    print("resume skips already-committed units OK")


def test_lease_expiry_releases_unit_to_new_worker(tmp_path: Path | None = None):
    import tempfile
    import time

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    units = [WorkUnit("u1", "smpl_hml", "baseline1", "native-gen", ["c1"], 1, [])]

    stale_pool = GPUWorkerPool(d, units, worker_id="worker-stale", lease_ttl=0.05)
    # Lease it but crash before committing (simulate interruption): manually
    # take the lease without running the phase.
    stale_pool.queue.lease("u1", "worker-stale", ttl=0.05)
    time.sleep(0.1)  # let the lease expire

    fresh_pool = GPUWorkerPool(d, units, worker_id="worker-fresh")
    calls = []
    result = fresh_pool.run({"native-gen": lambda u: (calls.append(u.unit_id), True)[1]})
    assert calls == ["u1"]
    assert result.committed == ["u1"]
    print("lease expiry releases unit to new worker OK")


def test_active_lease_blocks_other_worker_until_expiry(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    units = [WorkUnit("u1", "smpl_hml", "baseline1", "native-gen", ["c1"], 1, [])]
    holder = GPUWorkerPool(d, units, worker_id="worker-holder", lease_ttl=600.0)
    holder.queue.lease("u1", "worker-holder", ttl=600.0)  # still active

    other = GPUWorkerPool(d, units, worker_id="worker-other")
    calls = []
    result = other.run({"native-gen": lambda u: (calls.append(u.unit_id), True)[1]})
    assert calls == []
    assert result.blocked == ["u1"]
    print("active lease blocks other worker OK")


def test_quarantine_after_max_retries(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    units = [WorkUnit("u1", "smpl_hml", "baseline2", "native-gen", ["c1"], 1, [])]
    pool = GPUWorkerPool(d, units, max_retries=1)
    result = pool.run({"native-gen": lambda u: False})
    assert result.quarantined == ["u1"]
    assert "u1" in pool.queue.quarantined_unit_ids()
    print("quarantine after max retries OK")


def test_resource_exceeded_triggers_retry_then_quarantine(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    units = [WorkUnit("u1", "smpl_hml", "baseline5", "convert", ["c1"], 1, [])]

    def always_exceeds(unit):
        raise ResourceExceeded("simulated RSS budget exceeded")

    pool = GPUWorkerPool(d, units, max_retries=2)
    result = pool.run({"convert": always_exceeds})
    assert result.quarantined == ["u1"]
    assert len(result.retried) == 2  # retried twice before giving up
    failures = pool.queue._read_jsonl(pool.queue.failures_path)
    assert all("RSS budget exceeded" in f["detail"] for f in failures)
    print("resource_exceeded retry-then-quarantine OK")


def test_blocked_units_report_when_dependency_never_completes(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    units = _chain_units()
    pool = GPUWorkerPool(d, units, max_retries=0)
    result = pool.run({
        "native-gen": lambda u: False,  # fails immediately, quarantined (max_retries=0)
        "convert": lambda u: True,
        "gen-embed": lambda u: True,
    })
    assert result.quarantined == ["u-native"]
    assert result.blocked == ["u-convert", "u-embed"]
    print("blocked units reported when dependency never completes OK")


if __name__ == "__main__":
    test_dependency_order_is_respected()
    test_resume_skips_already_committed_units()
    test_lease_expiry_releases_unit_to_new_worker()
    test_active_lease_blocks_other_worker_until_expiry()
    test_quarantine_after_max_retries()
    test_resource_exceeded_triggers_retry_then_quarantine()
    test_blocked_units_report_when_dependency_never_completes()
    print("\nALL WORKER POOL TESTS PASSED")
