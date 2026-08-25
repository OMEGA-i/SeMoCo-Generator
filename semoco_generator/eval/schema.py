"""Schema-first core types for the two evaluation tracks.

The eval pipeline is deliberately small and explicit:

    TrackInput -> MotionModel.generate(ModelInput) -> native MotionClip
                -> ConversionGraph.convert -> target MotionClip -> EvalScore

Each model owns a :class:`ModelSchema` (what text/length it consumes and what
native motion representation it emits). Runners are thin executors that map track
prompts into model inputs and map native model outputs into the track's
required representation.

This module is pure-python (no torch/numpy-heavy work at import time) so it can
be imported by reporting, tests, and CLI wiring without pulling GPU deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

@dataclass(frozen=True)
class ModelSignature:
    """Lightweight identifier for cache key derivation without loading the model."""
    model_id: str
    weight_sig: str
    native_rep: MotionRep

# ---------------------------------------------------------------------------
# Motion representations
# ---------------------------------------------------------------------------
# The full set of motion representations the conversion graph knows about.
# Only a handful act as canonical relays (soma77, joints22); the rest are
# model-native entry points or evaluator targets.
MotionRep = Literal[
    "motion_codes",        # Semoco tokenizer codes [T, Q] int
    "soma77",              # SOMA77 posed joints [T, 77, 3]
    "joints22",            # HumanML-style 22 joints [T, 22, 3]
    "smpl_rot6d_transl",   # SMPL rot6d [T, 22, 6] + transl aux [T, 3]
    "smpl_vertices",       # SMPL mesh vertices [T, V, 3]
    "hml263",              # HumanML3D 263-D feature vector [T, 263]
]

TrackId = Literal["humanml3d_eval", "soma_tmr_eval"]
ModelRole = Literal["baseline", "ours"]


# ---------------------------------------------------------------------------
# Length specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LengthSpec:
    """How long a clip should be, expressed in whatever unit is authoritative.

    ``controlled=False`` marks a model whose length is decided by its own
    EOS rather than by the requested duration.
    """

    seconds: float | None = None
    frames: int | None = None
    tokens: int | None = None
    controlled: bool = True

    def to_frames(self, fps: float, *, default: int = 100) -> int:
        if self.frames is not None:
            return max(1, int(self.frames))
        if self.seconds is not None:
            return max(1, int(round(float(self.seconds) * float(fps))))
        return int(default)

    def to_seconds(self, fps: float, *, default: float = 5.0) -> float:
        if self.seconds is not None:
            return float(self.seconds)
        if self.frames is not None:
            return float(self.frames) / max(float(fps), 1e-6)
        return float(default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seconds": self.seconds,
            "frames": self.frames,
            "tokens": self.tokens,
            "controlled": self.controlled,
        }


# ---------------------------------------------------------------------------
# Motion payload
# ---------------------------------------------------------------------------
@dataclass
class MotionClip:
    """One motion clip in exactly one representation.

    ``array`` is the primary payload; ``aux`` carries companion arrays that a
    single representation genuinely needs (e.g. ``transl`` for
    ``smpl_rot6d_transl``). Derived representations are produced on demand by
    the :class:`~semoco_generator.eval.conversions.ConversionGraph`, never
    stored as optional fields on a shared bag.
    """

    rep: MotionRep
    array: np.ndarray
    fps: float
    aux: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.array.shape[0])


# ---------------------------------------------------------------------------
# Model schema / IO
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSchema:
    """Static description of one model's input/output contract."""

    model_id: str
    role: ModelRole
    text_input: str          # e.g. raw_caption / instruction_template / frozen_text_embedding
    length_input: str        # e.g. duration_frames / official_m_length / model_eos
    native_output: MotionRep
    native_fps: float
    generation_controls: tuple[str, ...] = ()
    required_assets: tuple[str, ...] = ()
    # Max batch size that keeps per-clip outputs invariant to batch mates / size /
    # shard layout. A model whose sampling RNG is shared across the batch must
    # keep this at 1, otherwise results depend on how clips were grouped.
    max_safe_batch: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "text_input": self.text_input,
            "length_input": self.length_input,
            "native_output": self.native_output,
            "native_fps": self.native_fps,
            "generation_controls": list(self.generation_controls),
            "required_assets": list(self.required_assets),
            "max_safe_batch": int(self.max_safe_batch),
        }


@dataclass
class ModelInput:
    """One generation request for a single (prompt, seed)."""

    prompt_id: str
    text: str
    length: LengthSpec
    seed: int = 0
    cfg_scale: float | None = None
    controls: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    """Native output of one generation request."""

    model_id: str
    prompt_id: str
    seed: int
    native_motion: MotionClip | None
    status: Literal["ok", "failed"] = "ok"
    error: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackInput:
    """One evaluation prompt with its GT reference."""

    prompt_id: str
    rec_id: str
    caption: str
    length: LengthSpec
    text_payload: dict[str, Any] = field(default_factory=dict)
    gt_ref: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "LengthSpec",
    "ModelInput",
    "ModelOutput",
    "ModelRole",
    "ModelSignature",
    "ModelSchema",
    "MotionClip",
    "MotionRep",
    "TrackId",
    "TrackInput",
]
