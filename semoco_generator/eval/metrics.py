"""Shared quantitative metrics for SeMoCo-Generator (train val + test).

Token-level (teacher-forced next-packet prediction):
    - per-codebook cross entropy (CE) and perplexity (ppl = exp(CE))
    - per-codebook top-1 / top-5 accuracy
    - aggregate ``ce_mean`` / ``ppl_mean`` over codebooks

Generation-level diagnostics (on autoregressive rollouts, no GT alignment):
    - ``repeat_rate``: fraction of generated packets identical to the previous one
      (collapse detector)
    - ``foot_skate``: mean horizontal foot speed while a foot is in contact (m/s)
    - ``jerk``: mean joint jerk magnitude (3rd time-derivative; lower = smoother)

TMR-space T2M suite (same *names* as HumanML3D; **not** numerically
comparable to HumanML3D-table numbers — embeddings come from TMR-SOMA + LLM2Vec):
    - ``fid``, ``r_precision`` (R@k + MedR), ``mm_dist``, ``diversity``,
      ``multimodality``, ``t2m_sim``

These are pure (numpy/torch only) so the module stays dependency-light; foot
indices for ``foot_skate`` are passed in by the caller (kept out of here to avoid
a soma import).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import streaming_metrics as _streaming


def new_accumulator(num_codebooks: int) -> dict:
    agg = {"n_tokens": 0.0}
    for i in range(num_codebooks):
        agg[f"ce_q{i}"] = 0.0
        agg[f"top1_q{i}"] = 0.0
        agg[f"top5_q{i}"] = 0.0
    return agg


@torch.no_grad()
def accumulate(agg: dict, logits: list[torch.Tensor], targets: torch.Tensor, mask: torch.Tensor) -> None:
    """Accumulate weighted-by-token CE / top-k hits over a batch.

    ``logits``: list of ``[B,T,V_i]``; ``targets`` ``[B,T,Q]``; ``mask`` ``[B,T]`` bool.
    """
    flat = mask.reshape(-1)
    n_valid = float(flat.sum().item())
    if n_valid == 0:
        return
    for i, lg in enumerate(logits):
        V = lg.shape[-1]
        lgf = lg.reshape(-1, V).float()
        tg = targets[..., i].reshape(-1)
        ce = (F.cross_entropy(lgf, tg, reduction="none") * flat).sum().item()
        top5 = lgf.topk(5, dim=-1).indices
        hit5 = ((top5 == tg.unsqueeze(-1)).any(-1) & flat).sum().item()
        hit1 = ((lgf.argmax(-1) == tg) & flat).sum().item()
        agg[f"ce_q{i}"] += ce
        agg[f"top1_q{i}"] += hit1
        agg[f"top5_q{i}"] += hit5
    agg["n_tokens"] += n_valid


def finalize(agg: dict, num_codebooks: int, *, prefix: str = "") -> dict:
    """Turn accumulated sums into per-codebook + aggregate metrics."""
    n = max(agg["n_tokens"], 1.0)
    out: dict[str, float] = {}
    mean_ce = 0.0
    for i in range(num_codebooks):
        ce = agg[f"ce_q{i}"] / n
        out[f"{prefix}ce_q{i}"] = ce
        out[f"{prefix}ppl_q{i}"] = math.exp(min(ce, 20.0))
        out[f"{prefix}top1_q{i}"] = agg[f"top1_q{i}"] / n
        out[f"{prefix}top5_q{i}"] = agg[f"top5_q{i}"] / n
        mean_ce += ce
    mean_ce /= max(num_codebooks, 1)
    out[f"{prefix}ce_mean"] = mean_ce
    out[f"{prefix}ppl_mean"] = math.exp(min(mean_ce, 20.0))
    out[f"{prefix}n_tokens"] = float(agg["n_tokens"])
    return out


@torch.no_grad()
def compute_token_metrics(
    model,
    loader,
    device,
    amp_dtype,
    num_codebooks: int,
    *,
    max_batches: int | None = None,
    prefix: str = "",
) -> dict:
    """Teacher-forced token metrics over a (packed) dataloader."""
    was_training = model.training
    model.eval()
    agg = new_accumulator(num_codebooks)
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        codes = batch["motion_codes"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        seg = batch["segment_ids"].to(device, non_blocking=True)
        pos = batch["positions"].to(device, non_blocking=True)
        inp, tgt = codes[:, :-1], codes[:, 1:]
        tmask = model.next_packet_mask(valid, seg)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model.forward_packed(inp, tgt, seg[:, :-1], pos[:, :-1])
        accumulate(agg, logits, tgt, tmask)
    if was_training:
        model.train()
    return finalize(agg, num_codebooks, prefix=prefix)


# ---------------------------------------------------------------------------
# Generation-level diagnostics
# ---------------------------------------------------------------------------
def repeat_rate(codes: np.ndarray, start: int = 0) -> float:
    """Fraction of packets (from ``start``) identical to the previous packet."""
    seg = codes[start:]
    if seg.shape[0] < 2:
        return 0.0
    same = np.all(seg[1:] == seg[:-1], axis=-1)
    return float(same.mean())


def joint_jerk(joints: np.ndarray, fps: float, start: int = 0) -> float:
    """Mean joint jerk magnitude (3rd derivative) over frames from ``start`` (m/s^3)."""
    j = joints[start:]
    if j.shape[0] < 4:
        return 0.0
    dt = 1.0 / max(fps, 1e-6)
    jerk = np.diff(j, n=3, axis=0) / (dt ** 3)
    return float(np.linalg.norm(jerk, axis=-1).mean())


# ---------------------------------------------------------------------------
# TMR / cross-model evaluation metrics (shared by the soma_tmr track)
# ---------------------------------------------------------------------------
def resample_fps(joints: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linear-resample ``joints [T, J, 3]`` from ``src_fps`` to ``dst_fps``."""
    T = joints.shape[0]
    if T < 2 or abs(src_fps - dst_fps) < 1e-6:
        return joints
    src_t = np.arange(T) / src_fps
    # floor() not round() — the time span is [0, (T-1)/src_fps]; we need
    # floor((T-1)/src_fps * dst_fps) + 1 samples so the last sample time
    # never exceeds src_t[-1].  round() adds a spurious trailing frame for
    # ~40 % of Semoco (50→20) clips and ~33 % of Kimodo (30→20) clips,
    # introducing an interpolated boundary frame that shifts the evaluator
    # embedding.
    n_dst = max(2, int(np.floor((T - 1) / src_fps * dst_fps)) + 1)
    dst_t = np.clip(np.arange(n_dst) / dst_fps, 0, src_t[-1])
    flat = joints.reshape(T, -1)
    # Vectorised over all feature columns — scipy.interp1d with axis=0
    # processes all columns in one C-level call (avoid 231× Python np.interp).
    from scipy.interpolate import interp1d
    out = interp1d(src_t, flat, axis=0, kind="linear", copy=False)(dst_t)
    return out.reshape(n_dst, joints.shape[1], joints.shape[2]).astype(np.float32)


def r_precision(
    text_emb: np.ndarray,
    motion_emb: np.ndarray,
    *,
    batch: int = 32,
    topk: tuple[int, ...] = (1, 2, 3, 5, 10),
    seed: int = 0,
) -> dict[str, float]:
    """R@k + MedR over random batches: does a caption retrieve its own motion?

    Gallery size is ``batch`` (classic HumanML batch-32 by default). R@k for k >= batch is
    skipped. MedR is 1-indexed median rank of the correct motion.
    """
    N = text_emb.shape[0]
    active_k = tuple(k for k in topk if k < batch)
    if N < 2:
        out = {f"R@{k}": float("nan") for k in active_k}
        out["MedR"] = float("nan")
        return out
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    hits = {k: 0 for k in active_k}
    ranks_all: list[int] = []
    total = 0
    for s in range(0, N - 1, batch):
        idx = order[s : s + batch]
        if len(idx) < 2:
            break
        d = np.linalg.norm(text_emb[idx][:, None, :] - motion_emb[idx][None, :, :], axis=-1)
        ranks = d.argsort(axis=1)
        for i in range(len(idx)):
            pos = int(np.where(ranks[i] == i)[0][0])
            ranks_all.append(pos)
            for k in active_k:
                hits[k] += int(pos < k)
        total += len(idx)
    out = {f"R@{k}": hits[k] / max(total, 1) for k in active_k}
    out["MedR"] = float(np.median(ranks_all) + 1) if ranks_all else float("nan")
    return out


def r_precision_full_gallery(
    text_emb: np.ndarray,
    motion_emb: np.ndarray,
    *,
    topk: tuple[int, ...] = (1, 2, 3, 5, 10),
    max_block_bytes: int = _streaming.DEFAULT_MAX_BLOCK_BYTES,
) -> dict[str, float]:
    """R@k + MedR against the full gallery (default dual-track protocol).

    Unlike :func:`r_precision`, every caption ranks against all motions. Prefer
    this for publishable Semoco/SOMA and modern HumanML runs; keep batch-32 only
    for legacy HumanML batch-32 table compatibility.

    Computed blockwise (see :mod:`streaming_metrics`) so a full ``N x N x D``
    broadcast is never materialized — for ``N ~= 12k``, ``D = 512`` that naive
    broadcast would be hundreds of GB of host RAM.
    """
    return _streaming.blockwise_full_gallery_rank(
        text_emb, motion_emb, topk=topk, max_block_bytes=max_block_bytes,
    )


def duplicate_text_groups(
    text_emb: np.ndarray,
    *,
    sim_thresh: float = 0.99,
    max_block_bytes: int = _streaming.DEFAULT_MAX_BLOCK_BYTES,
) -> list[set[int]]:
    """Group near-duplicate captions by text-text cosine similarity (Kimodo).

    Computed blockwise (see :mod:`streaming_metrics`) instead of a full
    ``N x N`` matrix plus an ``O(N^2)`` Python double loop.
    """
    return _streaming.blockwise_duplicate_text_groups(
        text_emb, sim_thresh=sim_thresh, max_block_bytes=max_block_bytes,
    )


def r_precision_full_gallery_dup_aware(
    text_emb: np.ndarray,
    motion_emb: np.ndarray,
    *,
    topk: tuple[int, ...] = (1, 2, 3, 5, 10),
    sim_thresh: float = 0.99,
    max_block_bytes: int = _streaming.DEFAULT_MAX_BLOCK_BYTES,
) -> dict[str, float]:
    """Full-gallery R@k where any motion in a near-duplicate text group counts.

    Matches Kimodo benchmark retrieval semantics: captions with text-text cosine
    similarity > ``sim_thresh`` are grouped, and retrieving any group member is
    treated as a hit for that query. Computed blockwise (see
    :mod:`streaming_metrics`); never materializes ``N x N x D``.
    """
    return _streaming.blockwise_full_gallery_rank_dup_aware(
        text_emb, motion_emb, topk=topk, sim_thresh=sim_thresh, max_block_bytes=max_block_bytes,
    )


def compute_embedding_metrics(
    *,
    gen_emb: np.ndarray,
    gt_emb: np.ndarray | None = None,
    text_emb: np.ndarray | None = None,
    gen_emb_by_prompt: np.ndarray | None = None,
    retrieval_protocol: str = "full_gallery",
    batch: int = 32,
    diversity_pairs: int = 300,
    seed: int = 0,
    topk: tuple[int, ...] = (1, 2, 3, 5, 10),
    kimodo_metrics: bool = False,
) -> dict[str, float]:
    """Shared embedding metrics with unified lowercase keys for both tracks."""
    out: dict[str, float] = {
        "fid": float("nan"),
        "fid_gen_gt": float("nan"),
        "fid_gen_text": float("nan"),
        "fid_gt_text": float("nan"),
        "r1": float("nan"),
        "r2": float("nan"),
        "r3": float("nan"),
        "r5": float("nan"),
        "r10": float("nan"),
        "medr": float("nan"),
        "matching": float("nan"),
        "t2m_sim": float("nan"),
        "t2m_sim_kimodo": float("nan"),
        "diversity": diversity(gen_emb, pairs=diversity_pairs, seed=seed),
        "multimodality": float("nan"),
    }
    if gt_emb is not None and len(gt_emb) >= 2 and len(gen_emb) >= 2:
        out["fid"] = fid(gen_emb, gt_emb)
        out["fid_gen_gt"] = out["fid"]
    if text_emb is not None and len(text_emb) >= 2:
        out["matching"] = mm_dist(text_emb, gen_emb)
        raw_sim = t2m_sim(text_emb, gen_emb)
        out["t2m_sim"] = raw_sim
        out["t2m_sim_kimodo"] = t2m_sim_kimodo(text_emb, gen_emb)
        if kimodo_metrics:
            # Kimodo extras live behind --kimodo-metrics, not a separate gallery protocol.
            rp = r_precision_full_gallery_dup_aware(text_emb, gen_emb, topk=topk)
        elif retrieval_protocol == "batch32":
            rp = r_precision(text_emb, gen_emb, batch=32, topk=topk, seed=seed)
        elif retrieval_protocol == "batch256":
            rp = r_precision(text_emb, gen_emb, batch=256, topk=topk, seed=seed)
        else:
            rp = r_precision_full_gallery(text_emb, gen_emb, topk=topk)
        out["r1"] = rp.get("R@1", float("nan"))
        out["r2"] = rp.get("R@2", float("nan"))
        out["r3"] = rp.get("R@3", float("nan"))
        out["r5"] = rp.get("R@5", float("nan"))
        out["r10"] = rp.get("R@10", float("nan"))
        out["medr"] = rp.get("MedR", float("nan"))
        if kimodo_metrics and len(gen_emb) >= 2:
            out["fid_gen_text"] = fid(gen_emb, text_emb[: len(gen_emb)])
            if gt_emb is not None and len(gt_emb) >= 2:
                out["fid_gt_text"] = fid(gt_emb[: len(text_emb)], text_emb[: len(gt_emb)])
    if gen_emb_by_prompt is not None:
        out["multimodality"] = multimodality(gen_emb_by_prompt)
    return out


def fid(gen_emb: np.ndarray, gt_emb: np.ndarray) -> float:
    """Frechet distance between generated and GT motion embedding distributions."""
    from scipy import linalg
    mu_g, cov_g = gen_emb.mean(0), np.cov(gen_emb, rowvar=False)
    mu_r, cov_r = gt_emb.mean(0), np.cov(gt_emb, rowvar=False)
    diff = mu_g - mu_r
    covmean = linalg.sqrtm(cov_g @ cov_r)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_g + cov_r - 2.0 * covmean))


def mm_dist(text_emb: np.ndarray, motion_emb: np.ndarray) -> float:
    """Mean L2 distance between paired text and motion embeddings (↓ better)."""
    n = min(len(text_emb), len(motion_emb))
    if n == 0:
        return float("nan")
    return float(np.linalg.norm(text_emb[:n] - motion_emb[:n], axis=-1).mean())


def diversity(motion_emb: np.ndarray, *, pairs: int = 300, seed: int = 0) -> float:
    """Mean pairwise L2 over random motion-embedding pairs (→ closer to Real better)."""
    n = motion_emb.shape[0]
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, size=pairs)
    b = rng.integers(0, n, size=pairs)
    m = a != b
    if not np.any(m):
        return float("nan")
    return float(np.linalg.norm(motion_emb[a[m]] - motion_emb[b[m]], axis=1).mean())


def multimodality(motion_emb_by_prompt: np.ndarray) -> float:
    """Within-prompt diversity: mean pairwise L2 across seeds per caption (↑ better).

    ``motion_emb_by_prompt`` shape ``[N, S, D]`` with ``S >= 2``.
    """
    if motion_emb_by_prompt.ndim != 3 or motion_emb_by_prompt.shape[1] < 2:
        return float("nan")
    n, s, _ = motion_emb_by_prompt.shape
    dists: list[float] = []
    for i in range(n):
        for a in range(s):
            for b in range(a + 1, s):
                dists.append(float(np.linalg.norm(
                    motion_emb_by_prompt[i, a] - motion_emb_by_prompt[i, b]
                )))
    return float(np.mean(dists)) if dists else float("nan")


def t2m_sim(text_emb: np.ndarray, motion_emb: np.ndarray) -> float:
    """Mean cosine similarity between paired text and motion embeddings (↑ better).

    Assumes unit-normalized TMR embeddings (``unit_vector=True``); falls back to
    explicit L2-normalization otherwise. Range is ``[-1, 1]``.
    """
    n = min(len(text_emb), len(motion_emb))
    if n == 0:
        return float("nan")
    t = text_emb[:n].astype(np.float64, copy=False)
    m = motion_emb[:n].astype(np.float64, copy=False)
    t = t / np.clip(np.linalg.norm(t, axis=-1, keepdims=True), 1e-8, None)
    m = m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-8, None)
    return float((t * m).sum(axis=-1).mean())


def t2m_sim_kimodo(text_emb: np.ndarray, motion_emb: np.ndarray) -> float:
    """Kimodo benchmark text-motion similarity: ``cosine / 2 + 0.5`` in ``[0, 1]``."""
    raw = t2m_sim(text_emb, motion_emb)
    if raw != raw:  # NaN
        return float("nan")
    return float(raw / 2.0 + 0.5)


def estimate_foot_contact(
    joints: np.ndarray,
    foot_indices,
    fps: float,
    *,
    vel_thresh: float = 0.15,
    height_thresh: float = 0.12,
) -> np.ndarray:
    """Heuristic foot-contact labels from joint height + velocity.

    ``joints [T, J, 3]`` (y-up). Returns ``[T, len(foot_indices)]`` in {0, 1}.
    """
    idx = list(foot_indices)
    feet = joints[:, idx, :]  # [T, F, 3]
    T = feet.shape[0]
    if T < 2:
        return np.zeros((T, len(idx)), dtype=np.float32)
    dt = 1.0 / max(fps, 1e-6)
    vel = np.zeros_like(feet)
    vel[1:] = np.diff(feet, axis=0) / dt
    vel[0] = vel[1]
    speed = np.linalg.norm(vel, axis=-1)
    height = feet[..., 1]
    return ((speed < vel_thresh) & (height < height_thresh)).astype(np.float32)


def tmr_alignment_metrics(
    text_emb: np.ndarray,
    motion_emb: np.ndarray,
    *,
    batch: int = 32,
    diversity_pairs: int = 300,
    seed: int = 0,
) -> dict[str, float]:
    """Bundle MMDist / Diversity / t2m_sim / R@k / MedR for a TMR embedding set."""
    out: dict[str, float] = {
        "MMDist": mm_dist(text_emb, motion_emb),
        "Diversity": diversity(motion_emb, pairs=diversity_pairs, seed=seed),
        "t2m_sim": t2m_sim(text_emb, motion_emb),
    }
    out.update(r_precision(text_emb, motion_emb, batch=batch, seed=seed))
    return out


def foot_skate(
    joints: np.ndarray,
    foot_contact: np.ndarray,
    foot_indices,
    fps: float,
    start: int = 0,
    contact_thresh: float = 0.5,
) -> float:
    """Mean horizontal foot speed (m/s) while that foot is in contact, from ``start``.

    ``joints [T,77,3]``, ``foot_contact [T,4]`` (per ``foot_indices``), up axis = y.
    """
    j = joints[start:]
    c = foot_contact[start:]
    T = min(j.shape[0], c.shape[0])
    if T < 2:
        return 0.0
    j = j[:T]; c = c[:T]
    feet = j[:, list(foot_indices), :]                 # [T,4,3]
    dt = 1.0 / max(fps, 1e-6)
    vel = np.diff(feet, axis=0) / dt                    # [T-1,4,3]
    horiz = np.linalg.norm(vel[..., [0, 2]], axis=-1)   # [T-1,4] xz speed
    in_contact = (c[1:] > contact_thresh)               # [T-1,4]
    denom = float(in_contact.sum())
    if denom == 0:
        return 0.0
    return float((horiz * in_contact).sum() / denom)
