"""Unified inference interface for evaluation models.

A :class:`MotionModel` normalizes one model's loading, text conditioning,
length conditioning, sampling, and *native* output packaging. It never performs
track-specific motion conversion; that is the job of the
:class:`~semoco_generator.eval.conversions.ConversionGraph`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..schema import ModelInput, ModelOutput, ModelSchema


class MotionModel(ABC):
    """A text-to-motion model that emits its native representation."""

    schema: ModelSchema

    @abstractmethod
    def generate(self, inputs: Sequence[ModelInput]) -> list[ModelOutput]:
        """Generate one :class:`ModelOutput` per input (aligned, same order)."""

    def weight_signature(self) -> str:
        """Stable signature of this model's weights, for cache keying.

        Default is the model id (baseline weights are static on disk); models
        loaded from a mutable checkpoint should include a checkpoint signature.
        """
        return self.schema.model_id

    def close(self) -> None:
        return None


__all__ = ["MotionModel"]
