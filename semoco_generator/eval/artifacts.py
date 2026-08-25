"""Evaluation artifact access behind one run-aware interface.

``cache.py`` deliberately remains the compatibility facade for cache keys and
physical storage.  This module is the higher-level seam used by evaluation
workflows: callers bind a dataset and optional run root once, then obtain a
small handle for the artifact family they need.  In particular, callers do
not need to remember that native motions are durable while converted motions
and generated embeddings are run-local.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cache as C
from .schema import MotionClip, MotionRep


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Artifact access scoped to one dataset and, optionally, one eval run.

    ``run_root`` is required only by artifact families whose identity is tied
    to a conversion or evaluation run.  Durable native and GT artifacts never
    use it.  The class intentionally contains no storage implementation; that
    remains centralized in :mod:`cache` so keys and data locations do not
    change during this refactor.
    """

    dataset_sig: str = ""
    run_root: Path | None = None

    @classmethod
    def for_run(
        cls,
        run_root: str | Path | None,
        *,
        dataset_sig: str = "",
    ) -> "EvaluationArtifacts":
        return cls(
            dataset_sig=dataset_sig,
            run_root=Path(run_root) if run_root is not None else None,
        )

    def model(self, model_id: str, weight_signature: str) -> "ModelArtifacts":
        return ModelArtifacts(self, model_id, weight_signature)

    def tmr_ground_truth(self, signature: str) -> "TMRGroundTruthArtifacts":
        return TMRGroundTruthArtifacts(self, signature)


@dataclass(frozen=True)
class ModelArtifacts:
    """All cached artifacts belonging to one model weight signature."""

    artifacts: EvaluationArtifacts
    model_id: str
    weight_signature: str

    @property
    def native(self) -> "NativeMotionArtifacts":
        return NativeMotionArtifacts(self)

    def converted(self, target_rep: MotionRep) -> "ConvertedMotionArtifacts":
        return ConvertedMotionArtifacts(self, target_rep)

    def embeddings(self, track: str, evaluation_signature: str) -> "GeneratedEmbeddingArtifacts":
        return GeneratedEmbeddingArtifacts(self, track, evaluation_signature)


@dataclass(frozen=True)
class NativeMotionArtifacts:
    """Durable native motions shared across evaluation runs."""

    model: ModelArtifacts

    def probe_many(self, clip_ids: list[str], seed: int, cfg: float | None) -> dict[str, bool]:
        return C.probe_native_many(
            self.model.model_id,
            self.model.weight_signature,
            clip_ids,
            seed,
            cfg,
            dataset=self.model.artifacts.dataset_sig,
        )

    def load_many(
        self,
        items: list[tuple[str, int, float | None]],
    ) -> dict[tuple[str, int, float | None], MotionClip | None]:
        return C.load_native_many(
            self.model.model_id,
            self.model.weight_signature,
            items,
            dataset=self.model.artifacts.dataset_sig,
        )

    def load(self, clip_id: str, seed: int, cfg: float | None) -> MotionClip | None:
        return C.load_native(
            self.model.model_id,
            self.model.weight_signature,
            clip_id,
            seed,
            cfg,
            dataset=self.model.artifacts.dataset_sig,
        )

    def save_many(self, items: list[tuple[str, int, float | None, MotionClip]]) -> None:
        C.save_native_many(
            self.model.model_id,
            self.model.weight_signature,
            items,
            dataset=self.model.artifacts.dataset_sig,
        )


@dataclass(frozen=True)
class ConvertedMotionArtifacts:
    """Run-local converted motions for one target representation."""

    model: ModelArtifacts
    target_rep: MotionRep

    def probe_many(self, clip_ids: list[str], seed: int, cfg: float | None) -> dict[str, bool]:
        return C.probe_converted_many(
            self.model.model_id,
            self.model.weight_signature,
            clip_ids,
            seed,
            cfg,
            self.target_rep,
            dataset=self.model.artifacts.dataset_sig,
            run_root=self.model.artifacts.run_root,
        )

    def load(self, clip_id: str, seed: int, cfg: float | None) -> MotionClip | None:
        return C.load_converted(
            self.model.model_id,
            self.model.weight_signature,
            clip_id,
            seed,
            cfg,
            self.target_rep,
            dataset=self.model.artifacts.dataset_sig,
            run_root=self.model.artifacts.run_root,
        )

    def load_many(
        self,
        clip_ids: list[str],
        seed: int,
        cfg: float | None,
    ) -> dict[str, MotionClip | None]:
        return C.load_converted_many(
            self.model.model_id,
            self.model.weight_signature,
            clip_ids,
            seed,
            cfg,
            self.target_rep,
            dataset=self.model.artifacts.dataset_sig,
            run_root=self.model.artifacts.run_root,
        )

    def save_many(self, items: list[tuple[str, int, float | None, MotionClip]]) -> None:
        C.save_converted_many(
            self.model.model_id,
            self.model.weight_signature,
            [(clip_id, seed, cfg, self.target_rep, clip) for clip_id, seed, cfg, clip in items],
            dataset=self.model.artifacts.dataset_sig,
            run_root=self.model.artifacts.run_root,
        )


@dataclass(frozen=True)
class GeneratedEmbeddingArtifacts:
    """Run-local evaluator embeddings for one model and evaluator signature."""

    model: ModelArtifacts
    track: str
    evaluation_signature: str

    def probe_many(self, clip_ids: list[str], seed: int, cfg: float | None) -> dict[str, bool]:
        return C.probe_gen_motion_many(
            self.track,
            self.evaluation_signature,
            self.model.model_id,
            self.model.weight_signature,
            clip_ids,
            seed,
            cfg,
            dataset=self.model.artifacts.dataset_sig,
            run_root=self.model.artifacts.run_root,
        )

    def load_many(
        self,
        clip_ids: list[str],
        seed: int,
        cfg: float | None,
    ) -> dict[str, np.ndarray | None]:
        return C.load_gen_motion_many(
            self.track,
            self.evaluation_signature,
            self.model.model_id,
            self.model.weight_signature,
            clip_ids,
            seed,
            cfg,
            dataset=self.model.artifacts.dataset_sig,
            run_root=self.model.artifacts.run_root,
        )

    def save_many(self, items: list[tuple[str, int, float | None, np.ndarray]]) -> None:
        C.save_gen_motion_many(
            self.track,
            self.evaluation_signature,
            self.model.model_id,
            self.model.weight_signature,
            items,
            dataset=self.model.artifacts.dataset_sig,
            run_root=self.model.artifacts.run_root,
        )


@dataclass(frozen=True)
class TMRGroundTruthArtifacts:
    """Durable TMR GT artifacts for one evaluator signature."""

    artifacts: EvaluationArtifacts
    signature: str

    def load_joints(self, clip_id: str) -> np.ndarray | None:
        return C.load_tmr_gt_joints(
            self.signature,
            clip_id,
            store=self.artifacts.dataset_sig,
        )


__all__ = [
    "ConvertedMotionArtifacts",
    "EvaluationArtifacts",
    "GeneratedEmbeddingArtifacts",
    "ModelArtifacts",
    "NativeMotionArtifacts",
    "TMRGroundTruthArtifacts",
]
