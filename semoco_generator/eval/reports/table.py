"""Single CSV/JSON report exporter for both evaluation tracks.

The two tracks share one writer; only the column set differs (SOMA-TMR adds
motion-quality columns). No comparability/native-matrix columns.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from ..score_schema import EvalScore

_COMMON_COLS = [
    "model",
    "dataset",
    "split",
    "metric_space",
    "retrieval_protocol",
    "target_rep",
    "subset",
    "num_prompts",
    "num_success",
    "num_failed",
    "fid",
    "r1",
    "r2",
    "r3",
    "r5",
    "r10",
    "medr",
    "matching",
    "t2m_sim",
    "diversity",
    "multimodality",
]

HUMANML_COLUMNS = _COMMON_COLS + ["failure_rate", "sec_per_clip"]

SOMA_TMR_COLUMNS = _COMMON_COLS + [
    "foot_skate",
    "jerk",
    "length_mean_s",
    "length_std_s",
    "eos_rate",
    "max_len_rate",
    "failure_rate",
    "sec_per_clip",
]


def export_scores(
    scores: Sequence[EvalScore],
    path: str | Path,
    *,
    columns: Sequence[str] = HUMANML_COLUMNS,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in scores:
            row = s.to_dict()
            w.writerow({k: row.get(k) for k in cols})
    out.with_suffix(".json").write_text(
        json.dumps([s.to_dict() for s in scores], indent=2) + "\n"
    )
    return out


__all__ = ["HUMANML_COLUMNS", "SOMA_TMR_COLUMNS", "export_scores"]
