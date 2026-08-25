"""Tests for the Phase 0 eval-cache audit (read-only measurement, no eval reruns)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from semoco_generator.eval.cache_audit import (
    audit_packed_store,
    discover_logs,
    discover_run_artifact_roots,
    estimate_full_gallery,
    parse_log,
    render_text,
    run_audit,
    scan_tree,
)
from semoco_generator.eval.cache_utils import packed_cache_root


def _write(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_scan_tree_counts_bytes_and_sidecars(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    root = d / "native"
    _write(root / "a.npz", b"12345")
    _write(root / "a.npz.sha256", b"deadbeef")  # has sidecar
    _write(root / "b.npz", b"1234567890")  # missing sidecar
    _write(root / "sub" / "c.npy", b"1")

    stats = scan_tree(root)
    assert stats.exists
    assert stats.total_files == 3  # sidecar files are not counted as data files
    assert stats.sidecar_files == 1
    assert stats.by_ext[".npz"].count == 2
    assert stats.by_ext[".npz"].with_sidecar == 1
    assert stats.by_ext[".npz"].without_sidecar == 1
    assert stats.by_ext[".npy"].count == 1
    assert stats.total_bytes == 5 + 10 + 1
    print("scan_tree counts OK")


def test_scan_tree_missing_root_is_reported_not_raised():
    stats = scan_tree(Path("/nonexistent/definitely/not/here"))
    assert not stats.exists
    assert stats.total_files == 0
    print("scan_tree missing root OK")


def test_discover_run_artifact_roots_and_logs(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    runs = d / "runs" / "eval"
    (runs / "smpl_hml" / "proto1" / "run_artifacts" / "converted").mkdir(parents=True)
    (runs / "soma_tmr" / "proto2" / "run_artifacts" / "gen_emb").mkdir(parents=True)
    _write(runs / "main" / "hml_gen_shard0.log", b"line one\nline two\n")

    roots = discover_run_artifact_roots(runs)
    assert len(roots) == 2
    assert all(p.name == "run_artifacts" for p in roots)

    logs = discover_logs(runs)
    assert len(logs) == 1
    assert logs[0].name == "hml_gen_shard0.log"
    print("discover_run_artifact_roots/logs OK")


def test_parse_log_counts_reject_and_fallback_lines(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    log_path = d / "shard0.log"
    log_path.write_text(
        "\n".join(
            [
                "[smpl_hml] subset protocol=official_hml_eval clips=100",
                "cache rejected (no sidecar): /some/path/a.npz",
                "cache rejected (no sidecar): /some/path/b.npz",
                "cache integrity failure: /some/path/c.npz (expected=aa actual=bb)",
                "[semoco] 16/16 prompts missing from store; falling back to live flan encode",
                "[semoco] native seed=0 16/100 (bs=16)",
            ]
        )
        + "\n"
    )
    stats = parse_log(log_path)
    assert stats.lines == 6
    assert stats.cache_rejected == 2
    assert stats.cache_corrupt == 1
    assert stats.live_encode_fallback == 1
    assert stats.last_progress_line == "[semoco] native seed=0 16/100 (bs=16)"
    print("parse_log OK")


def test_estimate_full_gallery_matches_naive_broadcast_bytes():
    est = estimate_full_gallery("hml", n=1000, d=512, dtype_bytes=4)
    assert est.full_gallery_bytes == 1000 * 1000 * 512 * 4
    assert est.dup_aware_sim_bytes == 1000 * 1000 * 8
    # Sanity: matches what np.linalg.norm(t[:,None,:] - m[None,:,:], axis=-1) would allocate.
    n, dd = 8, 4
    t = np.zeros((n, dd), dtype=np.float32)
    m = np.zeros((n, dd), dtype=np.float32)
    broadcast_bytes = (t[:, None, :] - m[None, :, :]).nbytes
    assert broadcast_bytes == estimate_full_gallery("x", n, dd).full_gallery_bytes
    print("estimate_full_gallery OK")


def test_run_audit_end_to_end_on_synthetic_cache(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    cache_root = d / "eval_cache"
    runs_root = d / "runs" / "eval"

    # hml_gt with one clip missing its sidecar (simulates the reported ~15k gap).
    motion_dir = cache_root / "hml_gt" / "sigA" / "motion_emb"
    motion_dir.mkdir(parents=True)
    np.save(motion_dir / "clip0.npy", np.zeros(512, dtype=np.float32))
    _write(motion_dir / "clip0.npy.sha256", b"abc")
    np.save(motion_dir / "clip1.npy", np.zeros(512, dtype=np.float32))  # no sidecar

    native_dir = cache_root / "native" / "dsA" / "modelA"
    native_dir.mkdir(parents=True)
    _write(native_dir / "c1.npz", b"0123456789")
    _write(native_dir / "c1.npz.sha256", b"abc")

    (runs_root / "smpl_hml" / "proto1" / "run_artifacts" / "converted").mkdir(parents=True)
    log_dir = runs_root / "main"
    _write(log_dir / "shard0.log", b"cache rejected (no sidecar): x\n")

    report = run_audit(cache_root=cache_root, runs_root=runs_root, include_gpu=False)

    assert report.cache_root == str(cache_root)
    assert "hml_gt" in report.cache_families
    assert report.cache_families["hml_gt"].total_files == 2
    assert report.cache_families["hml_gt"].by_ext[".npy"].without_sidecar == 1
    assert "native" in report.cache_families
    assert any(k.endswith("::converted") for k in report.run_artifacts)
    assert len(report.logs) == 1
    assert report.host_ram is not None
    assert len(report.memory_estimates) == 1
    assert report.memory_estimates[0].n == 2  # both clip0 + clip1 counted
    assert report.memory_estimates[0].d == 512

    text = render_text(report)
    assert "hml_gt" in text
    assert "sidecar_coverage" in text
    print("run_audit end-to-end OK")


def test_run_audit_does_not_offer_legacy_drop_for_unmanaged_files(tmp_path: Path):
    cache_root = tmp_path / "eval_cache"
    _write(cache_root / "release_clip_meta" / "release.jsonl", b"metadata")

    report = run_audit(cache_root=cache_root, include_gpu=False, include_logs=False)

    assert any("unmanaged non-v2" in note for note in report.notes)
    assert not any("eval cache drop --legacy" in note for note in report.notes)


def test_run_audit_reports_real_packed_v2_stores(tmp_path: Path | None = None, monkeypatch=None):
    """A real (non-synthetic) native + GT write through cache.py must show up
    in run_audit()'s packed_stores section with correct record/byte counts
    and a correctly-sampled embedding dim for the memory estimate."""
    import os
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    cache_root = d / "eval_cache"
    old_env = os.environ.get("SEMOCO_EVAL_CACHE_ROOT")
    os.environ["SEMOCO_EVAL_CACHE_ROOT"] = str(cache_root)
    try:
        import importlib

        from semoco_generator.eval import cache as C

        importlib.reload(C)  # pick up the new SEMOCO_EVAL_CACHE_ROOT + clear singleton stores
        from semoco_generator.eval.schema import MotionClip

        C.save_hml_gt_motion("sigA", "clip0", np.zeros(512, dtype=np.float32))
        C.save_hml_gt_motion("sigA", "clip1", np.zeros(512, dtype=np.float32))
        C.save_native(
            "modelA", "ckptA", "clip0", 0, 2.5,
            MotionClip(rep="joints22", array=np.zeros((4, 22, 3), dtype=np.float32), fps=20.0), dataset="dsA",
        )

        packed = audit_packed_store(packed_cache_root(cache_root))
        assert packed["hml_gt_motion"].records == 2
        assert packed["native"].records == 1
        assert packed["hml_gt_motion"].corrupt == 0
        assert packed["hml_gt_motion"].missing_packs == 0

        report = run_audit(cache_root=cache_root, include_gpu=False, include_logs=False)
        assert report.packed_stores["v2::hml_gt_motion"].records == 2
        assert report.packed_stores["v2::native"].records == 1
        assert len(report.memory_estimates) == 1
        assert report.memory_estimates[0].n == 2
        assert report.memory_estimates[0].d == 512

        text = render_text(report)
        assert "v2::hml_gt_motion" in text
        assert "v2::native" in text
    finally:
        if old_env is None:
            os.environ.pop("SEMOCO_EVAL_CACHE_ROOT", None)
        else:
            os.environ["SEMOCO_EVAL_CACHE_ROOT"] = old_env
        import importlib

        from semoco_generator.eval import cache as C

        importlib.reload(C)
    print("run_audit packed v2 stores OK")


if __name__ == "__main__":
    import tempfile

    test_scan_tree_missing_root_is_reported_not_raised()
    test_estimate_full_gallery_matches_naive_broadcast_bytes()
    with tempfile.TemporaryDirectory() as d:
        test_scan_tree_counts_bytes_and_sidecars(d)
    with tempfile.TemporaryDirectory() as d:
        test_discover_run_artifact_roots_and_logs(d)
    with tempfile.TemporaryDirectory() as d:
        test_parse_log_counts_reject_and_fallback_lines(d)
    with tempfile.TemporaryDirectory() as d:
        test_run_audit_end_to_end_on_synthetic_cache(d)
    with tempfile.TemporaryDirectory() as d:
        test_run_audit_reports_real_packed_v2_stores(d)
    print("\nALL CACHE AUDIT TESTS PASSED")
