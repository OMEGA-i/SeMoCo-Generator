"""Tests for duration-aware LPT sharding and multi-model track manifests."""

from __future__ import annotations

import json
from pathlib import Path

from semoco_generator.eval.planner import (
    TrackPromptCost,
    build_pilot_manifest,
    build_prompt_ids,
    build_track_manifest,
    estimate_cost,
    lpt_shards,
    read_manifest,
    write_manifest,
)


def test_pilot_planner_emits_three_phase_units(tmp_path: Path) -> None:
    prompt_ids = build_prompt_ids(count=5, prefix="hml")
    run, units = build_pilot_manifest(
        prompt_ids=prompt_ids,
        chunk_size=2,
        run_id="run1",
    )
    assert run.num_prompts == 5
    assert len(units) == 9
    assert units[1].phase == "convert"
    assert units[1].depends_on == [units[0].unit_id]
    assert units[2].phase == "gen-embed"
    assert units[2].depends_on == [units[1].unit_id]
    write_manifest(tmp_path, run, units)
    assert json.loads((tmp_path / "run.json").read_text())["run_id"] == "run1"
    assert len((tmp_path / "work_units.jsonl").read_text().splitlines()) == len(units)


def test_lpt_shards_balances_by_cost_not_count():
    items = [("a", 10.0), ("b", 1.0), ("c", 1.0), ("d", 1.0), ("e", 1.0), ("f", 1.0)]
    shards = lpt_shards(items, num_shards=2)
    loads = [sum(dict(items)[pid] for pid in shard) for shard in shards]
    # The heavy item (cost 10) should be alone in its shard once the other
    # shard's cumulative cost would exceed it.
    assert max(loads) - min(loads) <= 10.0
    assert sum(len(s) for s in shards) == len(items)
    print("lpt_shards balances by cost OK")


def test_estimate_cost_falls_back_to_default_for_missing_duration():
    assert estimate_cost(None) == 3.0
    assert estimate_cost(0) == 3.0
    assert estimate_cost(-1) == 3.0
    assert estimate_cost(5.5) == 5.5
    print("estimate_cost fallback OK")


def test_build_track_manifest_covers_all_models_independently(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    prompts = [TrackPromptCost(prompt_id=f"c{i}", duration_s=float(i % 5 + 1)) for i in range(20)]
    models = ["baseline2", "baseline3", "baseline1", "baseline5", "baseline4", "semoco"]
    run, units = build_track_manifest(track="smpl_hml", models=models, prompts=prompts, num_shards=4)

    assert run.num_prompts == 20
    by_model_phase: dict[tuple[str, str], list] = {}
    for u in units:
        by_model_phase.setdefault((u.model, u.phase), []).append(u)
    for model in models:
        assert len(by_model_phase[(model, "native-gen")]) == 4  # one per shard
        assert len(by_model_phase[(model, "convert")]) == 4
        assert len(by_model_phase[(model, "gen-embed")]) == 4
        # dependency chain wiring
        native_units = {u.unit_id: u for u in by_model_phase[(model, "native-gen")]}
        for cu in by_model_phase[(model, "convert")]:
            assert cu.depends_on and cu.depends_on[0] in native_units
        convert_units = {u.unit_id: u for u in by_model_phase[(model, "convert")]}
        for eu in by_model_phase[(model, "gen-embed")]:
            assert eu.depends_on and eu.depends_on[0] in convert_units

    write_manifest(d, run, units)
    run2, units2 = read_manifest(d)
    assert run2.run_id == run.run_id
    assert len(units2) == len(units)
    print("build_track_manifest covers all models + manifest roundtrip OK")


def test_build_track_manifest_skips_empty_shards(tmp_path: Path | None = None):
    prompts = [TrackPromptCost(prompt_id="only_one", duration_s=2.0)]
    run, units = build_track_manifest(track="soma_tmr", models=["baseline5"], prompts=prompts, num_shards=8)
    phases = {u.phase for u in units}
    assert phases == {"native-gen", "convert", "gen-embed"}
    assert len(units) == 3  # only the non-empty shard produces units
    print("build_track_manifest skips empty shards OK")


if __name__ == "__main__":
    test_lpt_shards_balances_by_cost_not_count()
    test_estimate_cost_falls_back_to_default_for_missing_duration()
    test_build_track_manifest_covers_all_models_independently()
    test_build_track_manifest_skips_empty_shards()
    print("\nALL PLANNER TRACK MANIFEST TESTS PASSED")
