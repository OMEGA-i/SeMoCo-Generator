"""Tests for the global multi-model scheduler fix (GPU-idle tail skew).

Covers three layers:

* :func:`planner.unit_priority` — per-model/per-phase weighting.
* :class:`worker_pool.GPUWorkerPool` — priority-ordered leasing across models.
* :func:`planner_exec.run_global_planner` — the actual fix: one worker leases
  ready units for *any* model in a single pool, instead of being scoped to
  whichever model an outer loop happened to load, so a worker idle on a light
  model immediately picks up a heavier model's remaining units.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from semoco_generator.eval import planner as planner_mod
from semoco_generator.eval.planner import TrackPromptCost, WorkUnit, unit_priority
from semoco_generator.eval.planner_exec import (
    LoadedModel,
    build_or_read_track_manifest,
    run_global_planner,
)
from semoco_generator.eval.worker_pool import GPUWorkerPool

# Scheduling behaviour is what these tests are about, not any particular
# model. Supply the cost spread here so the tests stay meaningful regardless of
# which models happen to be registered in the shipped weight table.
HEAVY = "heavy-model"
LIGHT = "light-model"


@pytest.fixture(autouse=True)
def _model_cost_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(planner_mod.MODEL_COST_WEIGHTS, HEAVY, 3.0)
    monkeypatch.setitem(planner_mod.MODEL_COST_WEIGHTS, LIGHT, 1.0)


def test_unit_priority_favors_heavier_model_and_native_gen():
    heavy_native = WorkUnit("u1", "soma_tmr", HEAVY, "native-gen", ["a", "b"], 2, [])
    light_native = WorkUnit("u2", "soma_tmr", LIGHT, "native-gen", ["a", "b"], 2, [])
    heavy_convert = WorkUnit("u3", "soma_tmr", HEAVY, "convert", ["a", "b"], 2, ["u1"])

    # The heavier model outranks a lighter one at the same phase.
    assert unit_priority(heavy_native) > unit_priority(light_native)
    # native-gen outranks convert/gen-embed for the same model.
    assert unit_priority(heavy_native) > unit_priority(heavy_convert)
    print("unit_priority favors heavier model + native-gen OK")


def test_worker_pool_orders_pending_by_priority_across_models(tmp_path: Path | None = None):
    """A global pool spanning two models must attempt the heavy model's
    native-gen units before the light model's, given equal readiness."""
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    units = [
        WorkUnit("light-native", "soma_tmr", LIGHT, "native-gen", ["a"], 1, []),
        WorkUnit("heavy-native", "soma_tmr", HEAVY, "native-gen", ["b"], 1, []),
    ]
    pool = GPUWorkerPool(d, units)
    order = list(pool.units.keys())
    assert order == ["heavy-native", "light-native"]
    print("GPUWorkerPool orders pending by priority across models OK")


def _prompts(n: int) -> list[TrackPromptCost]:
    return [TrackPromptCost(f"clip_{i:03d}", float(2 + i % 5)) for i in range(n)]


def test_run_global_planner_processes_every_model_in_one_call(tmp_path: Path | None = None):
    """The core fix: one run_global_planner() call spanning both models must
    load each model lazily exactly once and commit every unit for both,
    without an outer per-model loop."""
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    units = build_or_read_track_manifest(
        out_root=d, track="soma_tmr", models=[HEAVY, LIGHT], split="test",
        dataset="codes", prompts=_prompts(8), num_shards=2, run_id="proto-global-1",
    )

    load_calls: list[str] = []
    closed: list[str] = []
    phase_calls: list[str] = []

    def model_loader(model_id: str) -> LoadedModel:
        load_calls.append(model_id)
        return LoadedModel(model=model_id)  # plain str stands in for a real MotionModel here

    def _factory(phase: str):
        def factory(handle: LoadedModel):
            def fn(unit) -> bool:
                phase_calls.append(f"{phase}:{handle.model}:{unit.unit_id}")
                return True
            return fn
        return factory

    result = run_global_planner(
        out_root=d, units=units, model_loader=model_loader,
        native_fn_factory=_factory("native-gen"), convert_fn_factory=_factory("convert"),
        embed_fn_factory=_factory("gen-embed"),
        worker_id="test-worker",
        on_model_closed=lambda mid, _handle: closed.append(mid),
    )

    assert sorted(load_calls) == sorted([HEAVY, LIGHT])  # each model loaded exactly once
    assert len(result.committed) == len(units)
    assert not result.quarantined and not result.blocked
    assert any(c.startswith(f"native-gen:{HEAVY}:") for c in phase_calls)
    assert any(c.startswith(f"native-gen:{LIGHT}:") for c in phase_calls)
    assert sorted(closed) == sorted([HEAVY, LIGHT])  # every loaded model gets closed
    print("run_global_planner processes every model in one call OK")


def test_run_global_planner_never_holds_two_models_resident_at_once(tmp_path: Path | None = None):
    """Regression test for a real CUDA-OOM bug hit in production: models were
    only ever closed in the final ``finally`` block, so a worker touching
    several models over one ``run_global_planner`` call accumulated every
    one of their weights in VRAM simultaneously. At most one model must be
    loaded (present in ``handles``) at any instant during the run."""
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    units = build_or_read_track_manifest(
        out_root=d, track="soma_tmr", models=[HEAVY, "third-model", LIGHT], split="test",
        dataset="codes", prompts=_prompts(10), num_shards=3, run_id="proto-residency-1",
    )

    max_resident = 0
    resident: set[str] = set()
    load_calls: list[str] = []
    close_calls: list[str] = []

    def model_loader(model_id: str) -> LoadedModel:
        nonlocal max_resident
        load_calls.append(model_id)
        resident.add(model_id)
        max_resident = max(max_resident, len(resident))
        return LoadedModel(model=model_id)

    def _factory(_phase: str):
        def factory(_handle: LoadedModel):
            return lambda unit: True
        return factory

    def on_closed(model_id: str, _handle: LoadedModel) -> None:
        resident.discard(model_id)
        close_calls.append(model_id)

    result = run_global_planner(
        out_root=d, units=units, model_loader=model_loader,
        native_fn_factory=_factory("native-gen"), convert_fn_factory=_factory("convert"),
        embed_fn_factory=_factory("gen-embed"),
        worker_id="test-worker", on_model_closed=on_closed,
    )

    assert max_resident == 1, f"expected at most 1 resident model at a time, saw {max_resident}"
    assert len(result.committed) == len(units)
    assert not resident  # everything closed by the end
    assert set(close_calls) == {HEAVY, "third-model", LIGHT}
    print("run_global_planner never holds two models resident at once OK")


def test_idle_worker_helps_heavy_model_instead_of_exiting(tmp_path: Path | None = None):
    """Reproduces the GPU-idle bug directly: a worker that already committed
    every unit for the light model must still process the heavy model's
    remaining units when handed the full manifest — it must NOT exit early
    just because it has nothing left to do for whichever model happens to be
    first. This is the exact failure mode observed in a real sharded run
    (a worker reporting ``committed=0 ... GPU work done`` while its peers still
    had unclaimed heavy-model units)."""
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    units = build_or_read_track_manifest(
        out_root=d, track="soma_tmr", models=[HEAVY, LIGHT], split="test",
        dataset="codes", prompts=_prompts(6), num_shards=1, run_id="proto-skew-1",
    )

    # Simulate an earlier worker that raced ahead and already committed every
    # light-model unit (the "light model already done" state).
    light_units = [u for u in units if u.model == LIGHT]
    early_worker_loads: list[str] = []

    def early_loader(model_id: str) -> LoadedModel:
        early_worker_loads.append(model_id)
        return LoadedModel(model=model_id)

    def _noop_factory(_handle: LoadedModel):
        return lambda unit: True

    run_global_planner(
        out_root=d, units=light_units, model_loader=early_loader,
        native_fn_factory=_noop_factory, convert_fn_factory=_noop_factory, embed_fn_factory=_noop_factory,
        worker_id="worker-early",
    )
    assert early_worker_loads == [LIGHT]

    # Now a second worker gets the FULL manifest (both models). Under the old
    # per-model design this worker would only ever see LIGHT (fully
    # committed already) and exit with committed=0. Under the global
    # scheduler it must instead pick up the still-unclaimed heavy-model units.
    late_worker_loads: list[str] = []

    def late_loader(model_id: str) -> LoadedModel:
        late_worker_loads.append(model_id)
        return LoadedModel(model=model_id)

    result = run_global_planner(
        out_root=d, units=units, model_loader=late_loader,
        native_fn_factory=_noop_factory, convert_fn_factory=_noop_factory, embed_fn_factory=_noop_factory,
        worker_id="worker-late",
    )

    assert late_worker_loads == [HEAVY]  # never re-loads the already-committed light model
    heavy_units = [u for u in units if u.model == HEAVY]
    assert len(result.committed) == len(heavy_units)
    assert set(result.skipped_already_committed) == {u.unit_id for u in light_units}
    print("idle worker helps heavy model instead of exiting early OK")


if __name__ == "__main__":
    import tempfile

    test_unit_priority_favors_heavier_model_and_native_gen()
    with tempfile.TemporaryDirectory() as d:
        test_worker_pool_orders_pending_by_priority_across_models(d)
    with tempfile.TemporaryDirectory() as d:
        test_run_global_planner_processes_every_model_in_one_call(d)
    with tempfile.TemporaryDirectory() as d:
        test_run_global_planner_never_holds_two_models_resident_at_once(d)
    with tempfile.TemporaryDirectory() as d:
        test_idle_worker_helps_heavy_model_instead_of_exiting(d)
    print("\nALL GLOBAL SCHEDULER TESTS PASSED")
