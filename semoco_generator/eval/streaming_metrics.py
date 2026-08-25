"""Blockwise (host-RAM-bounded) retrieval metrics.

``metrics.r_precision_full_gallery`` / ``r_precision_full_gallery_dup_aware`` /
``duplicate_text_groups`` used to materialize a full ``N x N x D`` broadcast
(and an ``N x N`` Python double loop for duplicate grouping). For a full
HumanML3D eval (``N ~= 12k``, ``D = 512``) that broadcast alone is on the
order of hundreds of GB of host RAM — see ``eval cache audit``'s
``metric memory estimates`` section, which measures this from the real
cached embedding count/dim rather than a guessed constant.

This module computes the same rankings query-block x gallery-block at a time,
bounded by ``max_block_bytes``, and never allocates more than one block's
worth of ``[query_block, gallery_block]`` distances at once. Callers get the
exact same metric names/semantics (``R@k`` / ``MedR``) — chunking is an
internal implementation detail.

Rank definition (matches the original ``argsort``-based implementation for
the non-tied case): a query's rank is the number of gallery items strictly
closer than its own positive match (0-indexed; ``R@k`` counts rank ``< k``).
"""

from __future__ import annotations

import os

import numpy as np

_DEFAULT_BLOCK_MB = int(os.environ.get("SEMOCO_EVAL_METRIC_BLOCK_MB", "256"))
DEFAULT_MAX_BLOCK_BYTES = _DEFAULT_BLOCK_MB * 1024 * 1024
"""Cap for one ``[query_block, gallery_block]`` float32 distance block."""


def _nan_rank_dict(topk: tuple[int, ...]) -> dict[str, float]:
    out = {f"R@{k}": float("nan") for k in topk}
    out["MedR"] = float("nan")
    return out


def _validate_block_budget(max_block_bytes: int) -> int:
    """Reject invalid memory budgets early.

    Retrieval is intentionally conservative by default because the caller may be
    inside a container/pod whose real memory limit is lower than the host's
    visible RAM. The default stays at 256MB unless explicitly overridden.
    """
    max_block_bytes = int(max_block_bytes)
    if max_block_bytes <= 0:
        raise ValueError(f"max_block_bytes must be > 0, got {max_block_bytes}")
    return max_block_bytes


def _plan_block_sizes(n_query: int, n_gallery: int, d: int, *, max_block_bytes: int) -> tuple[int, int]:
    """Pick ``(query_block, gallery_block)`` so a float32 distance block of shape
    ``[query_block, gallery_block]`` computed from ``D``-dim embeddings stays
    within ``max_block_bytes`` (the broadcast subtraction is the dominant cost:
    ``query_block * gallery_block * D * 4`` bytes)."""
    max_block_bytes = _validate_block_budget(max_block_bytes)
    n_query = max(1, n_query)
    n_gallery = max(1, n_gallery)
    d = max(1, d)
    max_elems = max(1, max_block_bytes // (d * 4))
    gallery_block = max(1, min(n_gallery, max_elems))
    query_block = max(1, min(n_query, max_elems // gallery_block))
    return query_block, gallery_block


def _plan_gemm_block_sizes(n_query: int, n_gallery: int, *, max_block_bytes: int) -> tuple[int, int]:
    """Pick ``(query_block, gallery_block)`` for the GEMM-based squared-distance
    identity path, where the materialized block is only ``[query_block,
    gallery_block]`` (no ``D`` factor — ``D`` only affects matmul FLOPs, not the
    output block's memory footprint)."""
    max_block_bytes = _validate_block_budget(max_block_bytes)
    n_query = max(1, n_query)
    n_gallery = max(1, n_gallery)
    max_elems = max(1, max_block_bytes // 4)
    gallery_block = max(1, min(n_gallery, max_elems))
    query_block = max(1, min(n_query, max_elems // gallery_block))
    return query_block, gallery_block


def blockwise_full_gallery_rank(
    text_emb: np.ndarray,
    motion_emb: np.ndarray,
    *,
    topk: tuple[int, ...] = (1, 2, 3, 5, 10),
    max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
) -> dict[str, float]:
    """R@k + MedR against the full gallery, computed in bounded blocks.

    Never allocates an ``N x N x D`` (or even ``N x N``) array; peak extra
    memory is one ``[query_block, gallery_block]`` squared-distance block plus
    ``O(N)`` rank bookkeeping. Uses the ``||a-b||^2 = ||a||^2 + ||b||^2 -
    2*a@b^T`` identity so the dominant cost is a BLAS GEMM per block instead of
    an elementwise broadcast+norm loop (rank comparisons only need squared
    distances since both sides are non-negative, so ``sqrt`` is skipped).
    """
    n = min(len(text_emb), len(motion_emb))
    if n < 2:
        return _nan_rank_dict(topk)
    t = np.ascontiguousarray(text_emb[:n], dtype=np.float32)
    m = np.ascontiguousarray(motion_emb[:n], dtype=np.float32)
    t_sq = np.einsum("ij,ij->i", t, t)
    m_sq = np.einsum("ij,ij->i", m, m)
    qblock, gblock = _plan_gemm_block_sizes(n, n, max_block_bytes=max_block_bytes)

    positions = np.empty(n, dtype=np.int64)
    for qs in range(0, n, qblock):
        qe = min(qs + qblock, n)
        tq = t[qs:qe]
        self_block_sq = t_sq[qs:qe, None] + m_sq[None, qs:qe] - 2.0 * (tq @ m[qs:qe].T)
        self_sq = np.diag(self_block_sq).copy()
        rank_count = np.zeros(qe - qs, dtype=np.int64)
        for gs in range(0, n, gblock):
            ge = min(gs + gblock, n)
            dblock_sq = t_sq[qs:qe, None] + m_sq[None, gs:ge] - 2.0 * (tq @ m[gs:ge].T)
            self_mask = (np.arange(qs, qe) >= gs) & (np.arange(qs, qe) < ge)
            if np.any(self_mask):
                rows = np.nonzero(self_mask)[0]
                cols = np.arange(qs, qe)[self_mask] - gs
                dblock_sq[rows, cols] = self_sq[rows]
            rank_count += (dblock_sq < self_sq[:, None]).sum(axis=1)
        positions[qs:qe] = rank_count

    out = {f"R@{k}": float((positions < k).mean()) for k in topk}
    out["MedR"] = float(np.median(positions) + 1)
    return out


def blockwise_duplicate_text_groups(
    text_emb: np.ndarray,
    *,
    sim_thresh: float = 0.99,
    max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
) -> list[set[int]]:
    """Group near-duplicate captions by text-text cosine similarity.

    Computes the ``[query_block, N]`` similarity matrix one block of query
    rows at a time (GEMM), instead of a full ``N x N`` matrix plus an ``N^2``
    Python double loop. Union-find calls only happen for pairs above
    ``sim_thresh``, which is rare, so this stays fast even for large ``N``.
    """
    n = len(text_emb)
    if n == 0:
        return []
    t = text_emb.astype(np.float32, copy=False)
    t = t / np.clip(np.linalg.norm(t, axis=-1, keepdims=True), 1e-8, None)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    block = max(1, min(n, max_block_bytes // max(n * 4, 1)))
    for qs in range(0, n, block):
        qe = min(qs + block, n)
        sim_block = t[qs:qe] @ t.T  # [block, n], float32
        ii, jj = np.nonzero(sim_block > sim_thresh)
        for local_i, j in zip(ii.tolist(), jj.tolist()):
            i = qs + local_i
            if j > i:
                union(i, j)

    groups: dict[int, set[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), set()).add(i)
    return list(groups.values())


def blockwise_full_gallery_rank_dup_aware(
    text_emb: np.ndarray,
    motion_emb: np.ndarray,
    *,
    topk: tuple[int, ...] = (1, 2, 3, 5, 10),
    sim_thresh: float = 0.99,
    max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
) -> dict[str, float]:
    """Full-gallery R@k where retrieving any motion in a near-duplicate text
    group counts as a hit (Kimodo benchmark semantics), computed in bounded
    blocks with two gallery passes per query block (find each query's best
    allowed distance, then count how many gallery items beat it).
    """
    n = min(len(text_emb), len(motion_emb))
    if n < 2:
        return _nan_rank_dict(topk)
    t = np.ascontiguousarray(text_emb[:n], dtype=np.float32)
    m = np.ascontiguousarray(motion_emb[:n], dtype=np.float32)

    groups = blockwise_duplicate_text_groups(t, sim_thresh=sim_thresh, max_block_bytes=max_block_bytes)
    group_of = np.empty(n, dtype=np.int64)
    for gi, g in enumerate(groups):
        for idx in g:
            group_of[idx] = gi

    t_sq = np.einsum("ij,ij->i", t, t)
    m_sq = np.einsum("ij,ij->i", m, m)
    qblock, gblock = _plan_gemm_block_sizes(n, n, max_block_bytes=max_block_bytes)

    positions = np.empty(n, dtype=np.int64)
    for qs in range(0, n, qblock):
        qe = min(qs + qblock, n)
        tq = t[qs:qe]
        q_group = group_of[qs:qe]
        d_min_sq = np.full(qe - qs, np.inf, dtype=np.float32)
        for gs in range(0, n, gblock):
            ge = min(gs + gblock, n)
            dblock_sq = t_sq[qs:qe, None] + m_sq[None, gs:ge] - 2.0 * (tq @ m[gs:ge].T)
            self_mask = (np.arange(qs, qe) >= gs) & (np.arange(qs, qe) < ge)
            if np.any(self_mask):
                rows = np.nonzero(self_mask)[0]
                cols = np.arange(qs, qe)[self_mask] - gs
                dblock_sq[rows, cols] = t_sq[qs:qe][rows] + m_sq[qs:qe][rows] - 2.0 * np.sum(tq[rows] * m[qs:qe][rows], axis=1)
            allowed = group_of[gs:ge][None, :] == q_group[:, None]
            masked = np.where(allowed, dblock_sq, np.inf)
            d_min_sq = np.minimum(d_min_sq, masked.min(axis=1))
        rank_count = np.zeros(qe - qs, dtype=np.int64)
        for gs in range(0, n, gblock):
            ge = min(gs + gblock, n)
            dblock_sq = t_sq[qs:qe, None] + m_sq[None, gs:ge] - 2.0 * (tq @ m[gs:ge].T)
            self_mask = (np.arange(qs, qe) >= gs) & (np.arange(qs, qe) < ge)
            if np.any(self_mask):
                rows = np.nonzero(self_mask)[0]
                cols = np.arange(qs, qe)[self_mask] - gs
                dblock_sq[rows, cols] = d_min_sq[rows]
            rank_count += (dblock_sq < d_min_sq[:, None]).sum(axis=1)
        positions[qs:qe] = rank_count

    out = {f"R@{k}": float((positions < k).mean()) for k in topk}
    out["MedR"] = float(np.median(positions) + 1)
    return out


__all__ = [
    "DEFAULT_MAX_BLOCK_BYTES",
    "blockwise_full_gallery_rank",
    "blockwise_duplicate_text_groups",
    "blockwise_full_gallery_rank_dup_aware",
]
