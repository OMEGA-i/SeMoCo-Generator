"""Model schemas, target matrices, and the model loader.

Two evaluation matrices:

* ``humanml3d_eval``: models are scored on the official HumanML3D
  ``text_mot_match`` evaluator.
* ``soma_tmr_eval``: models are scored on the SOMA-TMR retrieval evaluator
  over a SOMA-representation test store.

Only our own model ships here. To score an external baseline, register a
:class:`ModelSchema` in :data:`MODEL_SCHEMAS`, add its ID to the relevant
matrix tuple, and return your adapter from :func:`load_model` — the track
runners and cache layer need no other change.
"""

from __future__ import annotations

from typing import Any

from ..cache_versions import GEN_ALIGN_VERSION
from ..schema import ModelSchema
from .base import MotionModel

# Every model in the HumanML3D matrix.
HUMANML3D_MODELS: tuple[str, ...] = ("semoco",)

# SOMA-TMR matrix.
SOMA_TMR_MODELS: tuple[str, ...] = ("semoco",)


MODEL_SCHEMAS: dict[str, ModelSchema] = {
    "semoco": ModelSchema(
        model_id="semoco",
        role="ours",
        text_input="frozen_text_embedding_or_live_encode",
        length_input="target_tok_from_duration",
        native_output="motion_codes",
        native_fps=50.0,
        generation_controls=("cfg_scale", "seed", "max_tok"),
        required_assets=("semoco_checkpoint", "motion_tokenizer"),
        max_safe_batch=128,  # H100 80GB, test batch=128
    ),
}

# Native default classifier-free-guidance scales (None => model default / n/a).
DEFAULT_CFG: dict[str, float | None] = {
    "semoco": 4.0,
}

# Accepted spellings for a model ID, mapped to its canonical form.
_ALIASES = {
    "semoco": "semoco",
    "semocogenerator": "semoco",
}


def normalize_model_name(name: str) -> str:
    key = str(name).lower().replace("-", "").replace("_", "")
    return _ALIASES.get(key, key)


def get_model_schema(name: str) -> ModelSchema | None:
    return MODEL_SCHEMAS.get(normalize_model_name(name))


def default_cfg_for(name: str, override: float | None = None) -> float | None:
    if override is not None:
        return float(override)
    return DEFAULT_CFG.get(normalize_model_name(name))


def weight_signature(name: str, **kwargs: Any) -> str:
    """Compute a model's cache signature WITHOUT loading it.

    This is the single source of truth for every model's cache signature;
    :meth:`.semoco.SemocoModel.weight_signature` delegates here rather than
    keeping a second, easily-desynced copy of the format string.
    """
    from ..cache import ckpt_sig

    key = normalize_model_name(name)
    if key == "semoco":
        eos_thresh = float(kwargs.get("eos_thresh", 1.01))
        return (
            f"semoco_{ckpt_sig(kwargs.get('checkpoint'))}_"
            f"{ckpt_sig(kwargs.get('tokenizer_checkpoint'))}_mt{int(kwargs.get('max_tok', 125))}"
            f"_eos{eos_thresh:g}"
        )
    raw = name.lower().replace("-", "")
    return f"{key}_{raw}_{GEN_ALIGN_VERSION}"


def load_model(name: str, **kwargs: Any) -> MotionModel:
    """Instantiate a :class:`MotionModel` for ``name``."""
    key = normalize_model_name(name)
    if key == "semoco":
        from .semoco import SemocoModel

        return SemocoModel(**kwargs)
    raise KeyError(
        f"unknown model {name!r}; choices: {', '.join(sorted(MODEL_SCHEMAS))}"
    )


__all__ = [
    "DEFAULT_CFG",
    "GEN_ALIGN_VERSION",
    "HUMANML3D_MODELS",
    "MODEL_SCHEMAS",
    "SOMA_TMR_MODELS",
    "default_cfg_for",
    "get_model_schema",
    "load_model",
    "normalize_model_name",
    "weight_signature",
]
