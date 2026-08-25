"""Tests for eval cache drop admin helpers (dry-run + apply)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from semoco_generator.eval.cache_admin import (
    apply_drop,
    apply_v2_scope_drop,
    plan_legacy_drop,
    plan_v2_scope_drop,
)
from semoco_generator.eval.cache_audit import run_audit
from semoco_generator.eval.cache_utils import packed_cache_root
from semoco_generator.eval.run_artifact_store import CACHE_V2_DIRNAME
from semoco_generator.eval.sharded_cache_store import PutRecord, ShardedCacheStore


def test_plan_legacy_drop_lists_durable_families_only(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    cache_root = d / "eval_cache"
    for name in ("hml_gt", "native", "tmr_gt", "tmr_text", "release_clip_meta"):
        (cache_root / name).mkdir(parents=True)
    (cache_root / "native" / "x.bin").write_bytes(b"0123456789")

    targets = plan_legacy_drop(cache_root=cache_root, runs_root=None)
    paths = {t.path for t in targets}
    assert str(cache_root / "native") in paths
    assert str(cache_root / "hml_gt") in paths
    assert str(cache_root / "release_clip_meta") not in paths  # never a drop target
    native_target = next(t for t in targets if t.path == str(cache_root / "native"))
    assert native_target.size_bytes == 10
    print("plan_legacy_drop scope OK")


def test_legacy_operations_reject_packed_v2_root(tmp_path: Path):
    packed_root = packed_cache_root(tmp_path / "eval_cache")
    (packed_root / "native").mkdir(parents=True)

    with pytest.raises(ValueError, match="packed-v2"):
        plan_legacy_drop(cache_root=packed_root)
    with pytest.raises(ValueError, match="packed-v2"):
        run_audit(cache_root=packed_root, include_gpu=False, include_logs=False)


def test_plan_legacy_drop_includes_run_artifacts(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    cache_root = d / "eval_cache"
    runs_root = d / "runs" / "eval"
    (cache_root / "native").mkdir(parents=True)
    ra = runs_root / "smpl_hml" / "proto1" / "run_artifacts"
    (ra / "converted").mkdir(parents=True)
    (ra / "gen_emb").mkdir(parents=True)
    (ra / CACHE_V2_DIRNAME).mkdir(parents=True)  # must NOT be targeted by --legacy

    targets = plan_legacy_drop(cache_root=cache_root, runs_root=runs_root)
    paths = {t.path for t in targets}
    assert str(ra / "converted") in paths
    assert str(ra / "gen_emb") in paths
    assert str(ra / CACHE_V2_DIRNAME) not in paths
    print("plan_legacy_drop run_artifacts scope OK")


def test_apply_drop_removes_listed_targets(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    cache_root = d / "eval_cache"
    (cache_root / "native" / "sub").mkdir(parents=True)
    (cache_root / "hml_gt").mkdir(parents=True)

    targets = plan_legacy_drop(cache_root=cache_root, runs_root=None)
    removed = apply_drop(targets)
    assert set(removed) == {str(cache_root / "native"), str(cache_root / "hml_gt")}
    assert not (cache_root / "native").exists()
    assert not (cache_root / "hml_gt").exists()
    print("apply_drop removes targets OK")


def test_v2_scope_drop_plan_and_apply(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    packed_root = packed_cache_root(d / "durable")
    store = ShardedCacheStore(packed_root, num_buckets=2)
    store.put_many("native", [PutRecord(key="native:a", arrays={"array": np.zeros(4, dtype=np.float32)})])
    store.put_many("gen_emb", [PutRecord(key="gen_emb:a", arrays={"array": np.zeros(4, dtype=np.float32)})])

    all_targets = plan_v2_scope_drop(v2_root=packed_root)
    assert len(all_targets) >= 4  # 2 scopes x (pack+index)

    scoped_targets = plan_v2_scope_drop(v2_root=packed_root, scopes=["native"])
    assert all("native" in Path(t.path).parent.as_posix() for t in scoped_targets)

    removed = apply_v2_scope_drop(v2_root=packed_root, scopes=["native"])
    assert len(removed) >= 2
    assert store.audit("native").records == 0
    assert store.audit("gen_emb").records == 1  # untouched
    print("v2 scope drop plan/apply OK")


if __name__ == "__main__":
    test_plan_legacy_drop_lists_durable_families_only()
    test_plan_legacy_drop_includes_run_artifacts()
    test_apply_drop_removes_listed_targets()
    test_v2_scope_drop_plan_and_apply()
    print("\nALL CACHE ADMIN TESTS PASSED")
