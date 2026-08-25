"""Evaluation model interface, adapters, and registry."""

from .base import MotionModel
from .registry import (
    DEFAULT_CFG,
    HUMANML3D_MODELS,
    MODEL_SCHEMAS,
    SOMA_TMR_MODELS,
    default_cfg_for,
    get_model_schema,
    load_model,
    normalize_model_name,
    weight_signature,
)

__all__ = [
    "DEFAULT_CFG",
    "HUMANML3D_MODELS",
    "MODEL_SCHEMAS",
    "MotionModel",
    "SOMA_TMR_MODELS",
    "default_cfg_for",
    "get_model_schema",
    "load_model",
    "normalize_model_name",
    "weight_signature",
]
