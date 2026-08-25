"""Tests for the packed-shard ShardedCacheStore v2."""

from __future__ import annotations

from pathlib import Path
import threading
import time

import numpy as np

from semoco_generator.eval.sharded_cache_store import PutRecord, ShardedCacheStore


def test_put_many_serializes_same_bucket_writers(tmp_path: Path, monkeypatch) -> None:
    """Two writers must never compute offsets for the same pack concurrently."""
    import semoco_generator.eval.sharded_cache_store as module

    original_append = module._PackAppender.append
    start = threading.Barrier(2)
    active = 0
    overlapped = False
    guard = threading.Lock()

    def observed_append(self, data: bytes):
        nonlocal active, overlapped
        with guard:
            active += 1
            overlapped = overlapped or active > 1
        try:
            # Make any absent writer lock deterministic rather than timing-led.
            time.sleep(0.05)
            return original_append(self, data)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(module._PackAppender, "append", observed_append)

    def writer(worker: int) -> None:
        start.wait()
        store = ShardedCacheStore(tmp_path, num_buckets=1)
        store.put_many(
            "native",
            [PutRecord(
                key=f"native:writer-{worker}",
                arrays={"array": np.full(1024, worker, dtype=np.float32)},
            )],
        )

    threads = [threading.Thread(target=writer, args=(worker,)) for worker in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not overlapped
    store = ShardedCacheStore(tmp_path, num_buckets=1)
    loaded = [
        item.key
        for batch in store.load_many("native", ["native:writer-0", "native:writer-1"])
        for item in batch
    ]
    assert set(loaded) == {"native:writer-0", "native:writer-1"}


def test_put_probe_load_roundtrip(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    store = ShardedCacheStore(d, num_buckets=4)

    records = [
        PutRecord(key=f"native:clip{i}", arrays={"array": np.arange(i + 1, dtype=np.float32)},
                  meta={"fps": 30.0, "rep": "motion_codes"})
        for i in range(10)
    ]
    delta = store.put_many("native", records)
    assert delta.written == 10
    assert delta.bytes_written > 0

    keys = [r.key for r in records]
    status = store.probe_many("native", keys)
    assert all(status[k].exists for k in keys)
    assert status[keys[0]].nbytes > 0

    missing_status = store.probe_many("native", ["native:does_not_exist"])
    assert not missing_status["native:does_not_exist"].exists

    loaded_keys = set()
    for batch in store.load_many("native", keys):
        for item in batch:
            loaded_keys.add(item.key)
            idx = int(item.key.replace("native:clip", ""))
            np.testing.assert_array_equal(item.arrays["array"], np.arange(idx + 1, dtype=np.float32))
            assert item.meta["fps"] == 30.0
    assert loaded_keys == set(keys)
    print("sharded cache store roundtrip OK")


def test_load_many_respects_max_bytes_batching(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    store = ShardedCacheStore(d, num_buckets=1)
    records = [
        PutRecord(key=f"gen_emb:k{i}", arrays={"array": np.ones(256, dtype=np.float32)})
        for i in range(20)
    ]
    store.put_many("gen_emb", records)
    keys = [r.key for r in records]

    batches = list(store.load_many("gen_emb", keys, max_bytes=256 * 4 * 3))
    assert len(batches) > 1
    total = sum(len(b) for b in batches)
    assert total == len(keys)
    print("load_many batches by max_bytes OK")


def test_audit_counts_records_and_bytes(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    store = ShardedCacheStore(d, num_buckets=2)
    records = [
        PutRecord(key=f"converted:c{i}", arrays={"array": np.zeros(8, dtype=np.float32)})
        for i in range(6)
    ]
    store.put_many("converted", records)
    audit = store.audit("converted")
    assert audit.records == 6
    assert audit.bytes > 0
    assert audit.corrupt == 0
    assert audit.missing_packs == 0
    print("audit counts OK")


def test_load_many_skips_corrupt_record(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    store = ShardedCacheStore(d, num_buckets=1)
    store.put_many("native", [PutRecord(key="native:a", arrays={"array": np.arange(4, dtype=np.float32)})])

    pack_path = store._pack_path("native", 0)
    data = bytearray(pack_path.read_bytes())
    # Flip a byte inside the payload region (after header) to corrupt checksum.
    data[-1] ^= 0xFF
    pack_path.write_bytes(bytes(data))

    batches = list(store.load_many("native", ["native:a"]))
    total_items = sum(len(b) for b in batches)
    assert total_items == 0  # corrupt record is skipped, not raised
    print("load_many skips corrupt record OK")


def test_drop_dry_run_then_apply(tmp_path: Path | None = None):
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    store = ShardedCacheStore(d, num_buckets=2)
    store.put_many("native", [PutRecord(key="native:a", arrays={"array": np.zeros(4, dtype=np.float32)})])

    dry = store.drop("native", dry_run=True)
    assert len(dry) >= 2  # at least one pack + one index
    for p in dry:
        assert Path(p).is_file()

    applied = store.drop("native", dry_run=False)
    assert applied == dry
    for p in applied:
        assert not Path(p).exists()
    assert store.audit("native").records == 0
    print("drop dry-run then apply OK")


def test_probe_and_load_see_another_processs_write_after_a_stale_cached_miss(
    tmp_path: Path | None = None,
):
    """Regression test for a real bug hit while validating the planner's
    global scheduler: `_load_bucket_index` caches a bucket's index in
    *process* memory and only invalidates it on *that same instance's own*
    `put_many` calls. With multiple GPU-pinned worker *processes* leasing
    work against the same manifest, one process's `probe_native_many` can
    run before a *different* process's native-gen commit ever lands in that
    bucket, cache the "not found" result, and then a later `load_native`
    call in the *same* process (e.g. the convert phase depending on that
    native-gen unit) incorrectly reports "native cache miss" even though the
    other process already durably wrote it. Two separate `ShardedCacheStore`
    instances against the same root stand in for two worker processes."""
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    reader = ShardedCacheStore(d, num_buckets=1)  # stands in for the convert-phase worker
    writer = ShardedCacheStore(d, num_buckets=1)  # stands in for the native-gen worker

    # 1. The "convert" worker probes before anyone has written -- caches an
    #    empty index for this bucket in its own process memory.
    miss = reader.probe_many("native", ["native:a"])
    assert not miss["native:a"].exists

    # 2. A *different* process (`writer`) does the native-gen write.
    writer.put_many(
        "native", [PutRecord(key="native:a", arrays={"array": np.arange(4, dtype=np.float32)})],
    )

    # 3. The "convert" worker (`reader`) probes/loads again in the *same*
    #    process lifetime, without ever calling `put_many` itself (so its
    #    per-instance cache was never explicitly invalidated). Before the
    #    fix, this would still report a miss.
    status = reader.probe_many("native", ["native:a"])
    assert status["native:a"].exists, "must see the other process's write, not a stale cached miss"

    loaded = [item for batch in reader.load_many("native", ["native:a"]) for item in batch]
    assert len(loaded) == 1
    np.testing.assert_array_equal(loaded[0].arrays["array"], np.arange(4, dtype=np.float32))
    print("probe/load see another process's write after a stale cached miss OK")


def test_uncommitted_pack_write_is_not_visible(tmp_path: Path | None = None):
    """Simulate a crash between the pack write and the index write: the
    record must stay invisible to probe/load until the index line exists."""
    import tempfile

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    store = ShardedCacheStore(d, num_buckets=1)
    pack_path = store._pack_path("native", 0)
    pack_path.parent.mkdir(parents=True)
    from semoco_generator.eval.sharded_cache_store import _encode_record

    raw = _encode_record("native:ghost", {"array": np.zeros(2, dtype=np.float32)}, {}, "v1")
    pack_path.write_bytes(raw)  # pack bytes exist, but no index entry written

    status = store.probe_many("native", ["native:ghost"])
    assert not status["native:ghost"].exists
    loaded = list(store.load_many("native", ["native:ghost"]))
    assert sum(len(b) for b in loaded) == 0
    print("uncommitted pack write stays invisible OK")


if __name__ == "__main__":
    test_put_probe_load_roundtrip()
    test_load_many_respects_max_bytes_batching()
    test_audit_counts_records_and_bytes()
    test_load_many_skips_corrupt_record()
    test_drop_dry_run_then_apply()
    test_probe_and_load_see_another_processs_write_after_a_stale_cached_miss()
    test_uncommitted_pack_write_is_not_visible()
    print("\nALL SHARDED CACHE STORE TESTS PASSED")
