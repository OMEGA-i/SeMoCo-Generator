"""``GPUWorkerPool``: lease/commit state machine over planner work units.

Implements the worker-scheduling contract from the production plan:

    planned -> leased -> running -> committed -> verified
    running -> retry_small_batch -> committed
    running -> failed -> quarantined

A committed manifest delta (``committed.jsonl``) is the source of truth for
resume — log lines are never consulted. Leases (``leases.jsonl``) are
advisory and TTL-based, so an interrupted worker's units become available
again automatically without any manual cleanup.

This module is intentionally execution-agnostic: callers pass one callable
per phase (``native-gen`` / ``convert`` / ``gen-embed``) that does the real
work for one :class:`~semoco_generator.eval.planner.WorkUnit` and returns
``True``/``False``. This lets the pool be exercised with fakes in tests and
wired to ``ensure_native`` / ``ensure_target`` / embedding steps in real
runners without duplicating scheduling logic in every track.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .planner import WorkUnit, unit_priority

LEASE_TTL_DEFAULT = 600.0
DEFAULT_MAX_RETRIES = 2


@dataclass(frozen=True)
class LeaseRecord:
    unit_id: str
    worker_id: str
    leased_at: float
    expires_at: float


@dataclass(frozen=True)
class CommitRecord:
    unit_id: str
    worker_id: str
    committed_at: float
    status: str  # "committed" | "quarantined"
    detail: str = ""


@dataclass(frozen=True)
class ResourceExceededRecord:
    """Structured fail-fast record — see the plan's OOM prevention gate.

    Distinct from a normal phase failure: this means the phase function
    raised/reported a resource budget violation, so the pool should retry at
    a smaller batch rather than immediately quarantining.
    """

    unit_id: str
    worker_id: str
    at: float
    detail: str


class ResourceExceeded(RuntimeError):
    """Raise from a phase function to trigger retry-at-smaller-batch instead
    of immediate quarantine (mirrors ``resource_guard.ResourceExceeded`` but
    kept independent so this module has no hard dependency on it)."""


class WorkQueueStore:
    """Append-only lease/commit/failure ledger backing resumable execution."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.leases_path = self.run_dir / "leases.jsonl"
        self.committed_path = self.run_dir / "committed.jsonl"
        self.failures_path = self.run_dir / "failures.jsonl"
        self.lock_path = self.run_dir / ".queue.lock"

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        out: list[dict] = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def _append_jsonl(self, path: Path, rec: dict, *, fsync: bool = True) -> None:
        """Append one JSON line, holding an exclusive ``flock`` for the whole
        write. Multiple worker *processes* (e.g. one per GPU) append to the
        same ledger files concurrently; without this lock two interleaved
        writes could corrupt a line and desync resume state.

        Args:
            path: JSONL file to append to.
            rec: Record dict to serialize.
            fsync: If True, call ``os.fsync()`` after write for crash safety.
                Leases are TTL-based and do not need fsync (a lost lease write
                is indistinguishable from an expired lease). Commits are the
                source of truth for resume and keep fsync=True.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                f.flush()
                if fsync:
                    os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def committed_unit_ids(self) -> set[str]:
        return {r["unit_id"] for r in self._read_jsonl(self.committed_path) if r.get("status") == "committed"}

    def quarantined_unit_ids(self) -> set[str]:
        return {r["unit_id"] for r in self._read_jsonl(self.committed_path) if r.get("status") == "quarantined"}

    def retry_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self._read_jsonl(self.failures_path):
            uid = r.get("unit_id")
            if uid:
                counts[uid] = counts.get(uid, 0) + 1
        return counts

    def active_leases(self, *, now: float | None = None) -> dict[str, dict]:
        now = time.time() if now is None else now
        leases: dict[str, dict] = {}
        for rec in self._read_jsonl(self.leases_path):
            if rec["expires_at"] > now:
                leases[rec["unit_id"]] = rec
            else:
                leases.pop(rec["unit_id"], None)
        return leases

    def lease(self, unit_id: str, worker_id: str, *, ttl: float = LEASE_TTL_DEFAULT) -> LeaseRecord:
        now = time.time()
        rec = LeaseRecord(unit_id=unit_id, worker_id=worker_id, leased_at=now, expires_at=now + ttl)
        self._append_jsonl(self.leases_path, asdict(rec), fsync=False)
        return rec

    def try_lease(self, unit_id: str, worker_id: str, *, ttl: float = LEASE_TTL_DEFAULT) -> LeaseRecord | None:
        """Atomically acquire a lease if the unit is neither committed nor
        actively leased by another worker.

        This is the cross-process critical section for multi-GPU workers. The
        old read-active-leases-then-append sequence allowed two processes to
        observe "unleased" concurrently and both append a lease for the same
        unit. Holding one lock across the read and append makes lease acquire
        the source of truth.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if unit_id in self.committed_unit_ids() or unit_id in self.quarantined_unit_ids():
                    return None
                active = self.active_leases()
                lease = active.get(unit_id)
                if lease is not None and lease["worker_id"] != worker_id:
                    return None
                return self.lease(unit_id, worker_id, ttl=ttl)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def commit(self, unit_id: str, worker_id: str, *, status: str = "committed", detail: str = "") -> None:
        rec = CommitRecord(unit_id=unit_id, worker_id=worker_id, committed_at=time.time(), status=status, detail=detail)
        self._append_jsonl(self.committed_path, asdict(rec))

    def record_failure(self, unit_id: str, worker_id: str, detail: str) -> None:
        self._append_jsonl(
            self.failures_path,
            {"unit_id": unit_id, "worker_id": worker_id, "detail": detail, "at": time.time()},
        )


PhaseFn = Callable[[WorkUnit], bool]
UnitPriorityFn = Callable[[WorkUnit], float]


@dataclass
class WorkerPoolResult:
    committed: list[str] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    skipped_already_committed: list[str] = field(default_factory=list)
    retried: list[str] = field(default_factory=list)


class GPUWorkerPool:
    """Executes a fixed set of :class:`WorkUnit` against phase callbacks,
    honoring dependency order, leases, and resumable commit state."""

    def __init__(
        self,
        run_dir: str | Path,
        units: list[WorkUnit],
        *,
        worker_id: str | None = None,
        lease_ttl: float = LEASE_TTL_DEFAULT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        priority_fn: UnitPriorityFn | None = None,
    ) -> None:
        self.queue = WorkQueueStore(run_dir)
        priority_fn = priority_fn or unit_priority
        # Highest-priority units come first so that, in a global multi-model
        # pool, concurrent workers preferentially race for the heaviest
        # remaining work instead of only discovering it once every lighter
        # model/phase has already been claimed (the tail-skew / idle-GPU
        # failure mode this ordering exists to prevent).
        ordered = sorted(units, key=lambda u: -priority_fn(u))
        self.units = {u.unit_id: u for u in ordered}
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.lease_ttl = lease_ttl
        self.max_retries = max_retries

    @staticmethod
    def _ready(unit: WorkUnit, committed: set[str]) -> bool:
        return all(dep in committed for dep in unit.depends_on)

    def run(self, phase_fns: dict[str, PhaseFn]) -> WorkerPoolResult:
        committed = self.queue.committed_unit_ids()
        quarantined = self.queue.quarantined_unit_ids()
        retry_counts = self.queue.retry_counts()
        result = WorkerPoolResult()

        pending = [u for u in self.units.values() if u.unit_id not in committed and u.unit_id not in quarantined]
        for u in self.units.values():
            if u.unit_id in committed:
                result.skipped_already_committed.append(u.unit_id)

        progressed = True
        while pending and progressed:
            progressed = False
            still_pending: list[WorkUnit] = []
            # Refresh cross-process state every pass. Other GPU workers may
            # have committed dependencies while this process was running a
            # different unit, so stale in-memory state would falsely report
            # "blocked".
            committed = self.queue.committed_unit_ids()
            quarantined = self.queue.quarantined_unit_ids()
            for unit in pending:
                if unit.unit_id in committed or unit.unit_id in quarantined:
                    continue
                if not self._ready(unit, committed):
                    still_pending.append(unit)
                    continue
                lease = self.queue.try_lease(unit.unit_id, self.worker_id, ttl=self.lease_ttl)
                if lease is None:
                    still_pending.append(unit)  # another worker still holds this lease
                    continue

                fn = phase_fns.get(unit.phase)
                ok = False
                detail = ""
                resource_exceeded = False
                try:
                    ok = bool(fn(unit)) if fn is not None else False
                    if not ok:
                        detail = "phase_fn returned falsy"
                except ResourceExceeded as exc:
                    resource_exceeded = True
                    detail = str(exc)
                except Exception as exc:  # noqa: BLE001 — record and continue other units
                    detail = f"{type(exc).__name__}: {exc}"

                if ok:
                    self.queue.commit(unit.unit_id, self.worker_id)
                    committed.add(unit.unit_id)
                    result.committed.append(unit.unit_id)
                    progressed = True
                    continue

                self.queue.record_failure(unit.unit_id, self.worker_id, detail)
                tries = retry_counts.get(unit.unit_id, 0) + 1
                retry_counts[unit.unit_id] = tries
                if resource_exceeded and tries <= self.max_retries:
                    # Retry-at-smaller-batch: caller's phase_fn is expected to read
                    # `retry_counts`-driven state itself (e.g. via unit_id lookups) or
                    # honor a smaller default; here we simply re-queue the unit.
                    result.retried.append(unit.unit_id)
                    still_pending.append(unit)
                    progressed = True
                    continue
                if (not resource_exceeded) and tries <= self.max_retries:
                    result.retried.append(unit.unit_id)
                    still_pending.append(unit)
                    progressed = True
                    continue

                self.queue.commit(unit.unit_id, self.worker_id, status="quarantined", detail=detail)
                quarantined.add(unit.unit_id)
                result.quarantined.append(unit.unit_id)
                progressed = True
            pending = still_pending

        for unit in pending:
            result.blocked.append(unit.unit_id)
        return result


__all__ = [
    "CommitRecord",
    "DEFAULT_MAX_RETRIES",
    "GPUWorkerPool",
    "LEASE_TTL_DEFAULT",
    "LeaseRecord",
    "PhaseFn",
    "ResourceExceeded",
    "ResourceExceededRecord",
    "UnitPriorityFn",
    "WorkQueueStore",
    "WorkerPoolResult",
]
