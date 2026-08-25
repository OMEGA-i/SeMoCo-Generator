"""Frozen evaluation protocol helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..score_schema import MetricSpace, RetrievalProtocol, TrackName


@dataclass
class EvalProtocol:
    """Frozen protocol describing one evaluation run."""

    protocol_id: str
    track: TrackName
    dataset: str
    split: str
    subset_source: str
    subset_ids: list[str]
    evaluator: str
    evaluator_checkpoint: str
    metric_space: MetricSpace
    retrieval_protocol: RetrievalProtocol = "full_gallery"
    fps_eval: float = 20.0
    duration_policy: str = "gt_or_fixed"
    default_duration_s: float = 5.0
    num_seeds: int = 1
    seed0: int = 0
    cfg_scale: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "EvalProtocol":
        obj = json.loads(Path(path).read_text())
        return cls(**obj)


def make_protocol_id(
    *,
    track: str,
    dataset: str,
    split: str,
    subset_ids: list[str],
    evaluator: str,
    retrieval_protocol: str,
    num_seeds: int,
    seed0: int,
    fps_eval: float = 20.0,
    subset_source: str = "",
    extras: dict | None = None,
    notes: list | None = None,
    **__: object,
) -> str:
    """Stable short id from the frozen subset + evaluator knobs."""
    h = hashlib.sha1()
    payload = {
        "track": track,
        "dataset": dataset,
        "split": split,
        "subset_ids": list(subset_ids),
        "evaluator": evaluator,
        "retrieval_protocol": retrieval_protocol,
        "num_seeds": num_seeds,
        "seed0": seed0,
    }
    h.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return f"{track}-{split}-{h.hexdigest()[:10]}"


ModeName = Literal["smoke", "main"]


def default_smoke_limit(track: TrackName) -> int:
    return 8 if track == "smpl_hml" else 16


__all__ = [
    "EvalProtocol",
    "ModeName",
    "default_smoke_limit",
    "make_protocol_id",
]
