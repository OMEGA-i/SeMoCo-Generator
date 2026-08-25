"""Numeric-parity + memory-boundedness tests for blockwise retrieval metrics.

Reference implementations here are deliberately re-derived (not imported from
production code) so a regression in ``streaming_metrics`` can't silently pass
by comparing against itself.
"""

from __future__ import annotations

import numpy as np

from semoco_generator.eval.metrics import (
    duplicate_text_groups,
    r_precision_full_gallery,
    r_precision_full_gallery_dup_aware,
)
from semoco_generator.eval.streaming_metrics import (
    _plan_block_sizes,
    _plan_gemm_block_sizes,
    blockwise_duplicate_text_groups,
    blockwise_full_gallery_rank,
    blockwise_full_gallery_rank_dup_aware,
)


def _brute_force_full_gallery(text_emb: np.ndarray, motion_emb: np.ndarray, topk=(1, 2, 3, 5, 10)):
    n = min(len(text_emb), len(motion_emb))
    t, m = text_emb[:n], motion_emb[:n]
    d = np.linalg.norm(t[:, None, :] - m[None, :, :], axis=-1)
    ranks = d.argsort(axis=1)
    positions = np.asarray([int(np.where(ranks[i] == i)[0][0]) for i in range(n)], dtype=np.int64)
    out = {f"R@{k}": float((positions < k).mean()) for k in topk}
    out["MedR"] = float(np.median(positions) + 1)
    return out


def _brute_force_dup_groups(text_emb: np.ndarray, sim_thresh: float = 0.99) -> list[set[int]]:
    n = len(text_emb)
    t = text_emb.astype(np.float64, copy=False)
    t = t / np.clip(np.linalg.norm(t, axis=-1, keepdims=True), 1e-8, None)
    sim = t @ t.T
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > sim_thresh:
                union(i, j)
    groups: dict[int, set[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), set()).add(i)
    return list(groups.values())


def _brute_force_dup_aware(text_emb, motion_emb, topk=(1, 2, 3, 5, 10), sim_thresh=0.99):
    n = min(len(text_emb), len(motion_emb))
    t, m = text_emb[:n], motion_emb[:n]
    groups = _brute_force_dup_groups(t, sim_thresh=sim_thresh)
    membership = np.empty(n, dtype=object)
    for g in groups:
        g_list = sorted(g)
        for i in g_list:
            membership[i] = g_list
    d = np.linalg.norm(t[:, None, :] - m[None, :, :], axis=-1)
    ranks = d.argsort(axis=1)
    positions = []
    for i in range(n):
        allowed = set(membership[i])
        pos = next(int(r) for r, j in enumerate(ranks[i]) if int(j) in allowed)
        positions.append(pos)
    pos_arr = np.asarray(positions, dtype=np.int64)
    out = {f"R@{k}": float((pos_arr < k).mean()) for k in topk}
    out["MedR"] = float(np.median(pos_arr) + 1)
    return out


def test_blockwise_full_gallery_matches_brute_force_small_blocks():
    rng = np.random.default_rng(0)
    n, d = 47, 8
    text = rng.normal(size=(n, d)).astype(np.float32)
    motion = rng.normal(size=(n, d)).astype(np.float32)

    expected = _brute_force_full_gallery(text, motion)
    # Force tiny blocks (query_block/gallery_block << n) so multi-block logic is exercised.
    got = r_precision_full_gallery(text, motion, max_block_bytes=64)
    for k in expected:
        assert abs(expected[k] - got[k]) < 1e-9, (k, expected[k], got[k])
    print("blockwise full-gallery matches brute force OK")


def test_blockwise_full_gallery_various_block_sizes_agree():
    rng = np.random.default_rng(1)
    n, d = 30, 16
    text = rng.normal(size=(n, d)).astype(np.float32)
    motion = rng.normal(size=(n, d)).astype(np.float32)
    ref = blockwise_full_gallery_rank(text, motion, max_block_bytes=10 * 1024 * 1024)
    for max_bytes in (32, 128, 1024, 1 << 20):
        got = blockwise_full_gallery_rank(text, motion, max_block_bytes=max_bytes)
        for k in ref:
            assert abs(ref[k] - got[k]) < 1e-6, (max_bytes, k, ref[k], got[k])
    print("blockwise full-gallery stable across block sizes OK")


def test_blockwise_duplicate_groups_match_brute_force():
    rng = np.random.default_rng(2)
    n, d = 40, 6
    base = rng.normal(size=(10, d))
    # Build clusters of near-duplicate captions (3-4 members each).
    text = np.concatenate([base[i % 10] + rng.normal(scale=1e-4, size=d) for i in range(n)]).reshape(n, d)

    expected = _brute_force_dup_groups(text, sim_thresh=0.9999)
    got = blockwise_duplicate_text_groups(text, sim_thresh=0.9999, max_block_bytes=64)
    expected_sorted = sorted(sorted(g) for g in expected)
    got_sorted = sorted(sorted(g) for g in got)
    assert expected_sorted == got_sorted
    print("blockwise duplicate groups match brute force OK")


def test_blockwise_dup_aware_rank_matches_brute_force():
    rng = np.random.default_rng(3)
    n, d = 24, 5
    base = rng.normal(size=(8, d))
    text = np.stack([base[i % 8] + rng.normal(scale=1e-4, size=d) for i in range(n)])
    motion = rng.normal(size=(n, d))

    expected = _brute_force_dup_aware(text, motion, sim_thresh=0.999)
    got = r_precision_full_gallery_dup_aware(text, motion, sim_thresh=0.999, max_block_bytes=64)
    for k in expected:
        assert abs(expected[k] - got[k]) < 1e-9, (k, expected[k], got[k])
    print("blockwise dup-aware rank matches brute force OK")


def test_public_wrappers_delegate_to_streaming_module():
    rng = np.random.default_rng(4)
    n, d = 12, 4
    text = rng.normal(size=(n, d))
    motion = rng.normal(size=(n, d))
    assert r_precision_full_gallery(text, motion) == blockwise_full_gallery_rank(text, motion)
    groups_a = duplicate_text_groups(text, sim_thresh=0.5)
    groups_b = blockwise_duplicate_text_groups(text, sim_thresh=0.5)
    assert sorted(sorted(g) for g in groups_a) == sorted(sorted(g) for g in groups_b)
    print("metrics.py wrappers delegate to streaming_metrics OK")


def test_edge_cases_n_less_than_2_return_nan():
    out = r_precision_full_gallery(np.zeros((1, 4)), np.zeros((1, 4)))
    assert all(v != v for v in out.values())  # all NaN
    out2 = r_precision_full_gallery_dup_aware(np.zeros((0, 4)), np.zeros((0, 4)))
    assert all(v != v for v in out2.values())
    assert duplicate_text_groups(np.zeros((0, 4))) == []
    print("edge cases (n<2) OK")


def test_plan_block_sizes_bounds_memory():
    """Never plan a block whose [q,g,D] float32 footprint exceeds the budget by more
    than one row/col (i.e. the planner actually enforces the cap, not just documents it)."""
    for n_query, n_gallery, d, budget in (
        (12221, 12221, 512, 256 * 1024 * 1024),
        (50000, 50000, 256, 64 * 1024 * 1024),
        (3, 3, 8, 16),
    ):
        qb, gb = _plan_block_sizes(n_query, n_gallery, d, max_block_bytes=budget)
        assert qb >= 1 and gb >= 1
        assert qb <= n_query and gb <= n_gallery
        footprint = qb * gb * d * 4
        # Single-row/col blocks (qb==1 or gb==1) can exceed budget if D alone is
        # large; otherwise the planner must respect the cap.
        if qb > 1 and gb > 1:
            assert footprint <= budget * 1.05
    print("plan_block_sizes bounds memory OK")


def test_plan_gemm_block_sizes_bounds_memory():
    for n_query, n_gallery, budget in (
        (12221, 12221, 256 * 1024 * 1024),
        (50000, 50000, 64 * 1024 * 1024),
        (3, 3, 16),
    ):
        qb, gb = _plan_gemm_block_sizes(n_query, n_gallery, max_block_bytes=budget)
        assert qb >= 1 and gb >= 1
        assert qb <= n_query and gb <= n_gallery
        footprint = qb * gb * 4
        if qb > 1 and gb > 1:
            assert footprint <= budget * 1.05
    print("plan_gemm_block_sizes bounds memory OK")


def test_invalid_block_budget_is_rejected():
    text = np.zeros((4, 4), dtype=np.float32)
    motion = np.zeros((4, 4), dtype=np.float32)
    try:
        blockwise_full_gallery_rank(text, motion, max_block_bytes=0)
    except ValueError as e:
        assert "max_block_bytes" in str(e)
    else:
        raise AssertionError("expected ValueError for max_block_bytes=0")
    print("invalid block budget rejected OK")


if __name__ == "__main__":
    test_blockwise_full_gallery_matches_brute_force_small_blocks()
    test_blockwise_full_gallery_various_block_sizes_agree()
    test_blockwise_duplicate_groups_match_brute_force()
    test_blockwise_dup_aware_rank_matches_brute_force()
    test_public_wrappers_delegate_to_streaming_module()
    test_edge_cases_n_less_than_2_return_nan()
    test_plan_block_sizes_bounds_memory()
    test_plan_gemm_block_sizes_bounds_memory()
    test_invalid_block_budget_is_rejected()
    print("\nALL STREAMING METRICS TESTS PASSED")
