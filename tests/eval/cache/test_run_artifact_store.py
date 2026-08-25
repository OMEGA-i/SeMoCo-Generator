"""Tests for RunArtifactStore (packed-shard converted/gen_emb artifacts)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from semoco_generator.eval.run_artifact_store import RunArtifactStore
from semoco_generator.eval.schema import MotionClip


def test_converted_roundtrip(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    store = RunArtifactStore(d)
    clip = MotionClip(rep="soma77", array=np.ones((5, 77, 3), dtype=np.float32), fps=30.0,
                       aux={"transl": np.zeros((5, 3), dtype=np.float32)})
    store.put_converted("baseline1", "sigA", "c1", 0, 1.5, "soma77", clip, dataset_sig="ds1")

    got = store.load_converted("baseline1", "sigA", "c1", 0, 1.5, "soma77", dataset_sig="ds1")
    assert got is not None
    assert got.rep == "soma77"
    np.testing.assert_array_equal(got.array, clip.array)
    np.testing.assert_array_equal(got.aux["transl"], clip.aux["transl"])

    assert store.load_converted("baseline1", "sigA", "c1", 0, 3.0, "soma77", dataset_sig="ds1") is None
    print("RunArtifactStore converted roundtrip OK")


def test_gen_embedding_roundtrip_and_probe_many(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    store = RunArtifactStore(d)
    for i in range(5):
        store.put_gen_embedding(
            "smpl_hml", "baseline1", "sigA", f"c{i}", 0, 1.5,
            np.full(8, float(i), dtype=np.float32), dataset_sig="ds1", eval_sig="ev1",
        )

    from semoco_generator.eval.run_artifact_store import gen_embedding_key
    keys = [
        gen_embedding_key("smpl_hml", "baseline1", "sigA", f"c{i}", 0, 1.5, dataset_sig="ds1", eval_sig="ev1")
        for i in range(6)
    ]
    status = store.probe_gen_embedding_many(keys)
    assert all(status[k] for k in keys[:5])
    assert not status[keys[5]]

    emb2 = store.load_gen_embedding("smpl_hml", "baseline1", "sigA", "c2", 0, 1.5, dataset_sig="ds1", eval_sig="ev1")
    assert emb2 is not None
    np.testing.assert_array_equal(emb2, np.full(8, 2.0, dtype=np.float32))
    print("RunArtifactStore gen_embedding roundtrip + probe_many OK")


def test_audit_and_drop(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    store = RunArtifactStore(d)
    store.put_gen_embedding("smpl_hml", "baseline1", "sigA", "c1", 0, None, np.zeros(4, dtype=np.float32))
    audit = store.audit("gen_emb")
    assert audit.records == 1
    dropped = store.drop("gen_emb", dry_run=False)
    assert len(dropped) >= 2
    assert store.audit("gen_emb").records == 0
    print("RunArtifactStore audit/drop OK")


if __name__ == "__main__":
    test_converted_roundtrip()
    test_gen_embedding_roundtrip_and_probe_many()
    test_audit_and_drop()
    print("\nALL RUN ARTIFACT STORE TESTS PASSED")
