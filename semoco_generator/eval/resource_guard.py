"""Lightweight host-RAM / GPU-VRAM safety checks for eval runners.

This module owns the production plan's ``ResourceGuard`` details:

* check process RSS against a caller-supplied budget before/after risky stages
* check GPU free-memory headroom before loading / running heavy models
* chunk large in-memory staging lists by estimated bytes instead of collecting a
  whole shard's converted motions in RAM at once
* aggressively release Python / CUDA caches between model lifecycles
* record every budget violation as a structured ``resource_exceeded`` entry
  (``failures.jsonl``-compatible), so Linux OOM is treated as a pipeline bug
  rather than the normal failure path
* learn and persist a safe batch size per ``(model, device)`` key: shrink by
  half on ``ResourceExceeded``, grow gradually on repeated success, so later
  runs start from the last known-safe value instead of re-discovering it
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import numpy as np
import psutil
import torch

T = TypeVar("T")


@dataclass(frozen=True)
class ResourceExceededRecord:
    """Structured fail-fast record for one resource budget violation.

    Distinct from a generic exception string so downstream tooling (worker
    pool failure logs, run summaries) can count/aggregate by ``kind`` without
    parsing free text.
    """

    stage: str
    kind: str  # "rss" | "gpu_vram" | "metric_memory"
    detail: str
    at: float
    rss_bytes: int | None = None
    planned_bytes: int | None = None
    budget_bytes: int | None = None
    free_bytes: int | None = None


class ResourceExceeded(RuntimeError):
    """Fail-fast resource budget violation. Carries the structured record
    that triggered it in :attr:`record`."""

    def __init__(self, message: str, record: ResourceExceededRecord | None = None) -> None:
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class ResourceBudget:
    worker_ram_budget_bytes: int | None = None
    gpu_vram_headroom_bytes: int | None = None
    stage_bytes_limit: int | None = None

    @classmethod
    def from_args(cls, args) -> "ResourceBudget":
        ram_gb = getattr(args, "worker_ram_budget_gb", None)
        vram_gb = getattr(args, "gpu_vram_headroom_gb", None)
        stage_gb = getattr(args, "stage_bytes_gb", None)
        return cls(
            worker_ram_budget_bytes=None if ram_gb in (None, 0) else int(float(ram_gb) * (1 << 30)),
            gpu_vram_headroom_bytes=None if vram_gb in (None, 0) else int(float(vram_gb) * (1 << 30)),
            stage_bytes_limit=None if stage_gb in (None, 0) else int(float(stage_gb) * (1 << 30)),
        )


class ResourceGuard:
    def __init__(
        self, budget: ResourceBudget, *,
        failures_path: str | Path | None = None,
        empty_cache_interval: int = 0,
    ) -> None:
        self.budget = budget
        self._proc = psutil.Process()
        self.failures_path = Path(failures_path) if failures_path else None
        self._empty_cache_interval = int(empty_cache_interval)
        self._empty_cache_counter = 0

    def _record_failure(self, rec: ResourceExceededRecord) -> None:
        if self.failures_path is None:
            return
        self.failures_path.parent.mkdir(parents=True, exist_ok=True)
        with self.failures_path.open("a") as f:
            f.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
            f.flush()

    def rss_bytes(self) -> int:
        return int(self._proc.memory_info().rss)

    def check_rss(self, stage: str, *, planned_bytes: int = 0) -> None:
        cap = self.budget.worker_ram_budget_bytes
        if cap is None:
            return
        rss = self.rss_bytes()
        if rss + int(planned_bytes) > cap:
            detail = (
                f"{stage}: RSS budget exceeded "
                f"(rss={rss / (1 << 30):.2f} GiB planned={planned_bytes / (1 << 30):.2f} GiB "
                f"budget={cap / (1 << 30):.2f} GiB)"
            )
            rec = ResourceExceededRecord(
                stage=stage, kind="rss", detail=detail, at=time.time(),
                rss_bytes=rss, planned_bytes=int(planned_bytes), budget_bytes=cap,
            )
            self._record_failure(rec)
            raise ResourceExceeded(detail, rec)

    def check_gpu_headroom(self, stage: str, device: str | torch.device) -> None:
        cap = self.budget.gpu_vram_headroom_bytes
        if cap is None or not torch.cuda.is_available():
            return
        dev = torch.device(device)
        if dev.type != "cuda":
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
        if free_bytes < cap:
            detail = (
                f"{stage}: GPU free memory below headroom "
                f"(free={free_bytes / (1 << 30):.2f} GiB total={total_bytes / (1 << 30):.2f} GiB "
                f"required_headroom={cap / (1 << 30):.2f} GiB)"
            )
            rec = ResourceExceededRecord(
                stage=stage, kind="gpu_vram", detail=detail, at=time.time(),
                budget_bytes=cap, free_bytes=free_bytes,
            )
            self._record_failure(rec)
            raise ResourceExceeded(detail, rec)

    def check_metric_memory(self, stage: str, *, planned_bytes: int) -> None:
        """Metric-memory gate: retrieval/duplicate-aware metrics must prove
        their planned block size fits within the RAM budget before allocating."""
        cap = self.budget.worker_ram_budget_bytes
        if cap is None:
            return
        rss = self.rss_bytes()
        if rss + int(planned_bytes) > cap:
            detail = (
                f"{stage}: metric memory budget exceeded "
                f"(rss={rss / (1 << 30):.2f} GiB planned={planned_bytes / (1 << 30):.2f} GiB "
                f"budget={cap / (1 << 30):.2f} GiB)"
            )
            rec = ResourceExceededRecord(
                stage=stage, kind="metric_memory", detail=detail, at=time.time(),
                rss_bytes=rss, planned_bytes=int(planned_bytes), budget_bytes=cap,
            )
            self._record_failure(rec)
            raise ResourceExceeded(detail, rec)

    def stage_bytes_limit(self) -> int:
        if self.budget.stage_bytes_limit is not None:
            return max(1, int(self.budget.stage_bytes_limit))
        if self.budget.worker_ram_budget_bytes is not None:
            # Leave most of the process budget for model weights / CUDA state / caller stacks.
            return max(256 * 1024 * 1024, self.budget.worker_ram_budget_bytes // 8)
        return 512 * 1024 * 1024

    def chunk_by_bytes(self, items: Iterable[T], size_fn: Callable[[T], int]) -> list[list[T]]:
        limit = self.stage_bytes_limit()
        chunks: list[list[T]] = []
        cur: list[T] = []
        cur_bytes = 0
        for item in items:
            item_bytes = max(0, int(size_fn(item)))
            if cur and cur_bytes + item_bytes > limit:
                chunks.append(cur)
                cur = []
                cur_bytes = 0
            cur.append(item)
            cur_bytes += item_bytes
        if cur:
            chunks.append(cur)
        return chunks

    def plan_batch(self, items: list[T], size_fn: Callable[[T], int], *, max_items: int | None = None) -> list[T]:
        """Return a prefix of ``items`` that fits both ``max_items`` and the
        stage byte budget — "shrink batch before allocation" from the plan's
        ResourceGuard enforcement rules, rather than allocating first and
        discovering the overrun after the fact."""
        limit_bytes = self.stage_bytes_limit()
        limit_items = len(items) if max_items is None else max(1, int(max_items))
        out: list[T] = []
        total = 0
        for item in items[:limit_items]:
            b = max(0, int(size_fn(item)))
            if out and total + b > limit_bytes:
                break
            out.append(item)
            total += b
        return out

    def cleanup(self) -> None:
        gc.collect()
        # Only call empty_cache when configured to do so periodically or when
        # memory pressure is explicitly checked via check_empty_cache_on_pressure.
        self.maybe_empty_cache()

    def maybe_empty_cache(self, *, force: bool = False) -> None:
        """Conditionally release cached CUDA memory.

        When *force* is True, always calls ``torch.cuda.empty_cache()``.
        When *empty_cache_interval* > 0, calls it every N invocations.
        Otherwise skips (default behaviour preserved — no cache clearing).
        """
        if not torch.cuda.is_available():
            return
        if force:
            torch.cuda.empty_cache()
            return
        if self._empty_cache_interval > 0:
            self._empty_cache_counter += 1
            if self._empty_cache_counter >= self._empty_cache_interval:
                self._empty_cache_counter = 0
                torch.cuda.empty_cache()

    def check_empty_cache_on_pressure(self, device: str | torch.device) -> None:
        """Call ``torch.cuda.empty_cache()`` if GPU memory fragmentation is high.

        Uses ``memory_stats()`` to detect when the allocator holds many cached
        blocks relative to active allocated memory — a sign of fragmentation
        that can prevent large contiguous allocations after near-OOM batches.
        """
        if not torch.cuda.is_available():
            return
        dev = torch.device(device)
        if dev.type != "cuda":
            return
        stats = torch.cuda.memory_stats(dev)
        active_bytes = stats.get("active_bytes.all.current", 0)
        reserved_bytes = stats.get("reserved_bytes.all.current", 0)
        if reserved_bytes > 0 and active_bytes > 0:
            # If >50% of reserved memory is cached but not actively used,
            # fragmentation is likely. Clear to defragment for next allocation.
            if active_bytes / reserved_bytes < 0.5:
                torch.cuda.empty_cache()

    def check_generation_headroom(
        self,
        stage: str,
        device: str | torch.device,
        *,
        estimated_kv_cache_bytes: int = 0,
        model_params_bytes: int = 0,
        safety_margin: float = 0.10,
    ) -> None:
        """Check whether estimated GPU memory for generation fits available VRAM.

        Raises :class:`ResourceExceeded` if the estimated allocation would
        exceed total VRAM minus the safety margin. The caller (runner or
        worker pool) can catch this to trigger batch-size reduction via
        :class:`AdaptiveBatchCap`.
        """
        if not torch.cuda.is_available():
            return
        dev = torch.device(device)
        if dev.type != "cuda":
            return
        total_bytes, _ = torch.cuda.mem_get_info(dev)
        allocated_bytes = torch.cuda.memory_allocated(dev)
        estimated_total = int(estimated_kv_cache_bytes) + int(model_params_bytes)
        safe_max = int(total_bytes * (1.0 - safety_margin))
        if estimated_total > safe_max:
            free_bytes = total_bytes - allocated_bytes
            detail = (
                f"{stage}: estimated gen memory {estimated_total / (1 << 30):.1f} GiB "
                f"(kv_cache={int(estimated_kv_cache_bytes) / (1 << 30):.1f} GiB + "
                f"params={int(model_params_bytes) / (1 << 30):.1f} GiB) "
                f"exceeds safe limit {safe_max / (1 << 30):.1f} GiB "
                f"({safety_margin * 100:.0f}% margin, "
                f"total VRAM {total_bytes / (1 << 30):.1f} GiB)"
            )
            rec = ResourceExceededRecord(
                stage=stage, kind="gpu_vram", detail=detail, at=time.time(),
                budget_bytes=safe_max, free_bytes=free_bytes,
            )
            self._record_failure(rec)
            raise ResourceExceeded(detail, rec)

    def validate_model_device(
        self,
        model: "torch.nn.Module",
        expected_device: str | torch.device,
        *,
        model_name: str = "unknown",
    ) -> None:
        """Check that all model parameters are on the expected device.

        Raises RuntimeError if any parameter is on a different device.
        Call this after model loading to catch Hydra/deep-config device
        placement bugs early.
        """
        expected = torch.device(expected_device)
        off_device: list[str] = []
        if hasattr(model, "named_parameters"):
            for name, param in model.named_parameters():
                if param.device != expected:
                    off_device.append(f"  {name}: on {param.device}, expected {expected}")
        elif hasattr(model, "parameters"):
            for i, param in enumerate(model.parameters()):
                if param.device != expected:
                    off_device.append(f"  param_{i}: on {param.device}, expected {expected}")
        if off_device:
            detail = (
                f"[{model_name}] {len(off_device)} parameter(s) not on expected "
                f"device {expected}:\n" + "\n".join(off_device[:20])
            )
            if len(off_device) > 20:
                detail += f"\n  ... and {len(off_device) - 20} more"
            raise RuntimeError(detail)


class AdaptiveBatchCap:
    """Learns and persists a safe batch size per ``(model, device)`` key.

    Starts from ``initial``. On repeated success, grows toward ``max_batch``.
    On :class:`ResourceExceeded`, halves toward ``min_batch``. Persisted as a
    small JSON file so a later run/process starts from the last known-safe
    value instead of re-discovering it from a cold guess (the plan's "persist
    the learned safe value per model/GPU").
    """

    def __init__(
        self, path: str | Path, *, initial: int = 8, min_batch: int = 1,
        max_batch: int = 64, growth_factor: float = 1.5,
    ) -> None:
        self.path = Path(path)
        self.initial = max(1, int(initial))
        self.min_batch = max(1, int(min_batch))
        self.max_batch = max(self.min_batch, int(max_batch))
        self.growth_factor = float(growth_factor)
        self._state = self._load()

    def _load(self) -> dict[str, int]:
        if self.path.is_file():
            try:
                return {k: int(v) for k, v in json.loads(self.path.read_text()).items()}
            except (json.JSONDecodeError, OSError, ValueError):
                return {}
        return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp")
        tmp.write_text(json.dumps(self._state, sort_keys=True, indent=2))
        tmp.replace(self.path)

    def get(self, key: str) -> int:
        return int(self._state.get(key, self.initial))

    def report_success(self, key: str, batch_size: int) -> int:
        cur = self._state.get(key, self.initial)
        if int(batch_size) >= cur:
            grown = min(self.max_batch, max(cur + 1, int(round(cur * self.growth_factor))))
            self._state[key] = grown
            self._save()
        return int(self._state.get(key, self.initial))

    def report_resource_exceeded(self, key: str, batch_size: int) -> int:
        shrunk = max(self.min_batch, int(batch_size) // 2)
        self._state[key] = shrunk
        self._save()
        return shrunk


def array_nbytes(arr: np.ndarray) -> int:
    return int(np.asarray(arr).nbytes)


def estimate_kv_cache_bytes(
    num_heads: int,
    head_dim: int,
    num_layers: int,
    batch_size: int,
    max_seq_len: int,
    *,
    dtype_bytes: int = 2,
) -> int:
    """Estimate GPU memory required for KV cache during autoregressive generation.

    KV cache = 2 (keys + values) × num_layers × batch_size × num_kv_heads
              × max_seq_len × head_dim × dtype_bytes

    With CFG (classifier-free guidance), effective batch doubles (cond + uncond).
    Default *dtype_bytes*=2 corresponds to bfloat16.
    """
    return (
        2
        * int(num_layers)
        * int(batch_size)
        * int(num_heads)
        * int(max_seq_len)
        * int(head_dim)
        * int(dtype_bytes)
    )


__all__ = [
    "AdaptiveBatchCap",
    "ResourceBudget",
    "ResourceExceeded",
    "ResourceExceededRecord",
    "ResourceGuard",
    "array_nbytes",
    "estimate_kv_cache_bytes",
]
