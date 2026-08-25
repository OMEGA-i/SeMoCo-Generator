"""Tests for lightweight eval resource guards."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from semoco_generator.eval.resource_guard import (
    AdaptiveBatchCap,
    ResourceBudget,
    ResourceExceeded,
    ResourceGuard,
    array_nbytes,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class _Args:
    worker_ram_budget_gb = 8
    gpu_vram_headroom_gb = 4
    stage_bytes_gb = 1


def test_budget_from_args_converts_gb_to_bytes():
    budget = ResourceBudget.from_args(_Args())
    assert budget.worker_ram_budget_bytes == 8 * (1 << 30)
    assert budget.gpu_vram_headroom_bytes == 4 * (1 << 30)
    assert budget.stage_bytes_limit == 1 * (1 << 30)
    print("budget_from_args OK")


def test_chunk_by_bytes_respects_stage_limit():
    budget = ResourceBudget(stage_bytes_limit=10)
    guard = ResourceGuard(budget)
    items = [3, 4, 5, 6]
    chunks = guard.chunk_by_bytes(items, lambda x: x)
    assert chunks == [[3, 4], [5], [6]]
    print("chunk_by_bytes OK")


def test_stage_limit_defaults_to_fraction_of_worker_budget():
    budget = ResourceBudget(worker_ram_budget_bytes=8 * (1 << 30))
    guard = ResourceGuard(budget)
    assert guard.stage_bytes_limit() == 1 * (1 << 30)
    print("stage limit default OK")


def test_check_rss_raises_when_planned_bytes_exceed_budget():
    guard = ResourceGuard(ResourceBudget(worker_ram_budget_bytes=1))
    try:
        guard.check_rss("test", planned_bytes=1)
    except ResourceExceeded as e:
        assert "RSS budget exceeded" in str(e)
    else:
        raise AssertionError("expected ResourceExceeded")
    print("check_rss fail-fast OK")


def test_array_nbytes_matches_numpy():
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    assert array_nbytes(arr) == arr.nbytes
    print("array_nbytes OK")


def test_check_rss_writes_structured_failure_record(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    failures_path = d / "failures.jsonl"
    guard = ResourceGuard(ResourceBudget(worker_ram_budget_bytes=1), failures_path=failures_path)
    try:
        guard.check_rss("stage-x", planned_bytes=100)
    except ResourceExceeded as e:
        assert e.record is not None
        assert e.record.kind == "rss"
        assert e.record.stage == "stage-x"
    else:
        raise AssertionError("expected ResourceExceeded")

    import json

    lines = failures_path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["kind"] == "rss"
    assert rec["planned_bytes"] == 100
    print("check_rss structured failure record OK")


def test_check_metric_memory_gate():
    guard = ResourceGuard(ResourceBudget(worker_ram_budget_bytes=1 << 30))
    with patch.object(guard, "rss_bytes", return_value=1024):
        guard.check_metric_memory("retrieval", planned_bytes=1024)
    tiny_guard = ResourceGuard(ResourceBudget(worker_ram_budget_bytes=1))
    try:
        tiny_guard.check_metric_memory("retrieval", planned_bytes=10 * (1 << 30))
    except ResourceExceeded as e:
        assert e.record.kind == "metric_memory"
    else:
        raise AssertionError("expected ResourceExceeded")
    print("check_metric_memory gate OK")


def test_plan_batch_shrinks_to_fit_byte_budget():
    guard = ResourceGuard(ResourceBudget(stage_bytes_limit=25))
    items = [10, 10, 10, 10]
    planned = guard.plan_batch(items, lambda x: x)
    assert planned == [10, 10]  # 3rd item would push total to 30 > 25
    assert sum(planned) <= 25
    print("plan_batch shrinks to byte budget OK")


def test_plan_batch_respects_max_items():
    guard = ResourceGuard(ResourceBudget())  # no byte limit configured beyond default
    items = list(range(100))
    planned = guard.plan_batch(items, lambda x: 1, max_items=5)
    assert len(planned) == 5
    print("plan_batch respects max_items OK")


def test_adaptive_batch_cap_shrinks_on_resource_exceeded_and_grows_on_success(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    cap_path = d / "adaptive_batch_cap.json"
    cap = AdaptiveBatchCap(cap_path, initial=8, min_batch=1, max_batch=64, growth_factor=2.0)

    assert cap.get("semoco@cuda:0") == 8
    shrunk = cap.report_resource_exceeded("semoco@cuda:0", 8)
    assert shrunk == 4
    assert cap.get("semoco@cuda:0") == 4

    grown = cap.report_success("semoco@cuda:0", 4)
    assert grown == 8  # 4 * growth_factor(2.0) = 8
    assert cap.get("semoco@cuda:0") == 8

    # Persistence: a fresh AdaptiveBatchCap instance reads the learned value back.
    cap2 = AdaptiveBatchCap(cap_path, initial=8)
    assert cap2.get("semoco@cuda:0") == 8
    print("AdaptiveBatchCap shrink/grow/persist OK")


def test_adaptive_batch_cap_never_exceeds_max_or_drops_below_min(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    cap = AdaptiveBatchCap(d / "cap.json", initial=8, min_batch=2, max_batch=16, growth_factor=3.0)
    for _ in range(5):
        cap.report_success("k", cap.get("k"))
    assert cap.get("k") <= 16
    for _ in range(10):
        cap.report_resource_exceeded("k", cap.get("k"))
    assert cap.get("k") >= 2
    print("AdaptiveBatchCap respects min/max bounds OK")


def test_check_gpu_headroom_against_real_cuda_device():
    """Real-GPU smoke test (skipped if no CUDA visible): confirms
    ``check_gpu_headroom`` reads genuine ``torch.cuda.mem_get_info`` for the
    current device rather than a mocked/fake value, in both directions
    (passes when headroom is generous, raises when the required headroom is
    set absurdly high)."""
    if torch is None or not torch.cuda.is_available():
        print("test_check_gpu_headroom_against_real_cuda_device SKIPPED (no CUDA)")
        return
    dev = torch.device("cuda:0")
    free_b, total_b = torch.cuda.mem_get_info(dev)

    guard = ResourceGuard(ResourceBudget(gpu_vram_headroom_bytes=1))  # 1 byte — trivially satisfied
    guard.check_gpu_headroom("light-gpu-test", dev)  # must not raise

    guard_tight = ResourceGuard(ResourceBudget(gpu_vram_headroom_bytes=total_b * 10))  # impossible headroom
    try:
        guard_tight.check_gpu_headroom("light-gpu-test", dev)
    except ResourceExceeded as e:
        assert e.record.kind == "gpu_vram"
        assert e.record.free_bytes == free_b
    else:
        raise AssertionError("expected ResourceExceeded when headroom exceeds total device memory")
    print(
        f"check_gpu_headroom real-CUDA OK "
        f"(device={torch.cuda.get_device_name(0)} free={free_b / (1 << 30):.1f}GiB/{total_b / (1 << 30):.1f}GiB)"
    )


if __name__ == "__main__":
    test_budget_from_args_converts_gb_to_bytes()
    test_chunk_by_bytes_respects_stage_limit()
    test_stage_limit_defaults_to_fraction_of_worker_budget()
    test_check_rss_raises_when_planned_bytes_exceed_budget()
    test_array_nbytes_matches_numpy()
    test_check_rss_writes_structured_failure_record()
    test_check_metric_memory_gate()
    test_plan_batch_shrinks_to_fit_byte_budget()
    test_plan_batch_respects_max_items()
    test_adaptive_batch_cap_shrinks_on_resource_exceeded_and_grows_on_success()
    test_adaptive_batch_cap_never_exceeds_max_or_drops_below_min()
    test_check_gpu_headroom_against_real_cuda_device()
    print("\nALL RESOURCE GUARD TESTS PASSED")
