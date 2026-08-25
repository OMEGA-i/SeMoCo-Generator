"""Unified evaluation score row shared by both tracks.

This records the *results* of a run (metrics, counts, timing), not paper-table
bookkeeping. There is no native-matrix / comparability / paper-anchor metadata;
conversion provenance, when useful for debugging, lives in ``extras``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

MetricSpace = Literal["humanml3d", "soma_tmr"]
RetrievalProtocol = Literal["full_gallery", "batch32", "batch256"]
TrackName = Literal["smpl_hml", "soma_tmr"]


@dataclass
class EvalScore:
    """Publishable per-model score row for either evaluation track."""

    model: str
    track: TrackName
    dataset: str
    split: str
    evaluator: str
    evaluator_checkpoint: str
    metric_space: MetricSpace
    retrieval_protocol: RetrievalProtocol
    target_rep: str
    num_prompts: int
    num_success: int
    num_failed: int
    num_seeds: int
    protocol_id: str
    subset: str | None = None  # source group label when scoring per-subset

    # Core embedding metrics (same names across tracks).
    fid: float | None = None
    r1: float | None = None
    r2: float | None = None
    r3: float | None = None
    r5: float | None = None
    r10: float | None = None
    medr: float | None = None
    matching: float | None = None
    t2m_sim: float | None = None
    diversity: float | None = None
    multimodality: float | None = None

    # Engineering metrics (computed from joints when available).
    foot_skate: float | None = None
    jerk: float | None = None
    length_mean_s: float | None = None
    length_std_s: float | None = None
    eos_rate: float | None = None
    max_len_rate: float | None = None
    failure_rate: float = 0.0
    sec_per_clip: float | None = None

    notes: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_embedding_metrics(
        cls,
        *,
        model: str,
        track: TrackName,
        dataset: str,
        split: str,
        evaluator: str,
        evaluator_checkpoint: str,
        metric_space: MetricSpace,
        retrieval_protocol: RetrievalProtocol,
        target_rep: str,
        protocol_id: str,
        num_prompts: int,
        num_success: int,
        num_failed: int,
        num_seeds: int,
        emb: dict[str, float],
        **kwargs: Any,
    ) -> "EvalScore":
        """Build an :class:`EvalScore` from shared embedding-metric keys."""
        return cls(
            model=model,
            track=track,
            dataset=dataset,
            split=split,
            evaluator=evaluator,
            evaluator_checkpoint=evaluator_checkpoint,
            metric_space=metric_space,
            retrieval_protocol=retrieval_protocol,
            target_rep=target_rep,
            num_prompts=num_prompts,
            num_success=num_success,
            num_failed=num_failed,
            num_seeds=num_seeds,
            protocol_id=protocol_id,
            fid=_maybe_float(emb.get("fid")),
            r1=_maybe_float(emb.get("r1")),
            r2=_maybe_float(emb.get("r2")),
            r3=_maybe_float(emb.get("r3")),
            r5=_maybe_float(emb.get("r5")),
            r10=_maybe_float(emb.get("r10")),
            medr=_maybe_float(emb.get("medr")),
            matching=_maybe_float(emb.get("matching")),
            t2m_sim=_maybe_float(emb.get("t2m_sim")),
            diversity=_maybe_float(emb.get("diversity")),
            multimodality=_maybe_float(emb.get("multimodality")),
            failure_rate=float(num_failed) / max(num_prompts * max(num_seeds, 1), 1),
            **kwargs,
        )


def _maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


__all__ = [
    "EvalScore",
    "MetricSpace",
    "RetrievalProtocol",
    "TrackName",
]
