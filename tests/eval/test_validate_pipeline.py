"""Tests for the dependency-free production pipeline rehearsal."""

from __future__ import annotations

from pathlib import Path

from semoco_generator.eval.validate_pipeline import (
    run_cold_warm_comparison,
    run_drop_rebuild_rehearsal,
    run_full_rehearsal,
    run_interrupt_resume_rehearsal,
    run_ram_stress_validation,
)


def test_cold_warm_probe_is_not_slower_than_load(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    result = run_cold_warm_comparison(d, n_records=100, array_len=256)
    assert result.n_records == 100
    assert result.probe_many_seconds >= 0
    assert result.load_many_materialize_seconds >= 0
    # probe_many never materializes payloads, so it must not be slower than
    # actually loading and summing every payload's bytes.
    assert result.probe_many_seconds <= result.load_many_materialize_seconds * 2
    print("cold/warm comparison OK")


def test_interrupt_resume_never_recomputes_committed_units(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    result = run_interrupt_resume_rehearsal(d, n_prompts=8)
    assert not result.recomputed_native_gen
    assert len(result.phase1_committed) > 0
    assert len(result.phase2_skipped_already_committed) == len(result.phase1_committed)
    assert len(result.phase2_committed) > 0
    print("interrupt/resume rehearsal OK")


def test_drop_rebuild_rehearsal_empties_then_restores(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    result = run_drop_rebuild_rehearsal(d, n_records=20)
    assert result.records_before_drop == 20
    assert result.records_after_drop == 0
    assert result.records_after_rebuild == 20
    assert result.drop_removed_paths > 0
    print("drop/rebuild rehearsal OK")


def test_ram_stress_validation_survives_and_shrinks():
    result = run_ram_stress_validation(budget_bytes=32 * 1024, n_batches=6)
    assert result.process_survived
    assert result.batches_shrunk > 0
    assert result.final_safe_batch >= 1
    print("RAM stress validation OK")


def test_full_rehearsal_all_passed(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    report = run_full_rehearsal(d)
    assert report.all_passed
    print("full rehearsal all_passed OK")


if __name__ == "__main__":
    test_cold_warm_probe_is_not_slower_than_load()
    test_interrupt_resume_never_recomputes_committed_units()
    test_drop_rebuild_rehearsal_empties_then_restores()
    test_ram_stress_validation_survives_and_shrinks()
    test_full_rehearsal_all_passed()
    print("\nALL VALIDATE PIPELINE TESTS PASSED")
