"""Frozen defaults for the HumanML3D ``text_mot_match`` evaluation track."""
from __future__ import annotations

import os
from pathlib import Path

from ....paths import baseline_checkpoint_root

FPS = 20.0
UNIT_LENGTH = 4
MAX_MOTION_LENGTH = 196
MAX_TEXT_LEN = 20
MIN_MOTION_LEN = 40
MAX_MOTION_LEN_EXCLUSIVE = 200
RETRIEVAL_PROTOCOL = "batch32"
HML_SUBSET_PROTOCOL = "official_hml_eval"
_HML_ROOT = baseline_checkpoint_root() / "HumanML3D" / "t2m"
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "SEMOCO_HML_EVALUATOR_CHECKPOINT",
        _HML_ROOT / "text_mot_match" / "model" / "finest.tar",
    )
).expanduser()
DEFAULT_MEAN_STD = Path(
    os.environ.get(
        "SEMOCO_HML_MEAN_STD_DIR",
        _HML_ROOT / "Comp_v6_KLD01" / "meta",
    )
).expanduser()
DEFAULT_OPT = DEFAULT_CHECKPOINT.parents[1] / "opt.txt"

__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_MEAN_STD",
    "DEFAULT_OPT",
    "FPS",
    "HML_SUBSET_PROTOCOL",
    "MAX_MOTION_LENGTH",
    "MAX_MOTION_LEN_EXCLUSIVE",
    "MAX_TEXT_LEN",
    "MIN_MOTION_LEN",
    "RETRIEVAL_PROTOCOL",
    "UNIT_LENGTH",
]
