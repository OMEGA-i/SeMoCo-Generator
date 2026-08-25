"""Central model selection and scoring-entry construction for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .checkpoints import CheckpointSpec


SEMOCO_MODEL_IDS = frozenset({"semoco", "semocogenerator"})


def _is_semoco_name(name: str) -> bool:
    return name.lower().replace("-", "") in SEMOCO_MODEL_IDS


def default_baselines(track: str) -> tuple[str, ...]:
    """Return the registered non-Semoco baseline IDs for *track*."""
    if track == "smpl_hml":
        from .models import HUMANML3D_MODELS

        models = HUMANML3D_MODELS
    elif track == "soma_tmr":
        from .models import SOMA_TMR_MODELS

        models = SOMA_TMR_MODELS
    else:
        raise ValueError(f"unsupported evaluation track {track!r}")
    return tuple(model for model in models if not _is_semoco_name(model))


@dataclass(frozen=True)
class ModelScoreEntry:
    """One model's cache identity and display label for aggregate scoring."""

    model_id: str
    signature_kwargs: dict[str, object]
    display_name: str

    def as_tuple(self) -> tuple[str, dict, str]:
        """Compatibility shape consumed by existing aggregation functions."""
        return self.model_id, dict(self.signature_kwargs), self.display_name


@dataclass(frozen=True)
class ModelPlan:
    """Selected baseline/Semoco models and their scoring entries.

    The plan is deliberately independent of argparse and model loading.  Both
    track runners and cache-only aggregation can therefore derive the exact
    same cache signature inputs from their already-resolved checkpoint specs.
    """

    baselines: tuple[str, ...]
    semoco_specs: tuple[CheckpointSpec, ...]

    @property
    def model_ids(self) -> list[str]:
        return [*self.baselines, *(spec.name for spec in self.semoco_specs)]

    @property
    def score_entries(self) -> list[ModelScoreEntry]:
        entries = [ModelScoreEntry(model, {}, model) for model in self.baselines]
        entries.extend(
            ModelScoreEntry(
                "semoco",
                {
                    "checkpoint": str(spec.model_ckpt),
                    "tokenizer_checkpoint": str(spec.tokenizer_ckpt),
                    "max_tok": spec.max_tok,
                    "codes_root": str(spec.codes_root),
                },
                spec.name,
            )
            for spec in self.semoco_specs
        )
        return entries

    def scoring_tuples(self) -> list[tuple[str, dict, str]]:
        return [entry.as_tuple() for entry in self.score_entries]


def build_model_plan(
    requested_models: Sequence[str] | None,
    semoco_specs: Sequence[CheckpointSpec],
    *,
    default_models: Sequence[str],
    include_baselines: bool = True,
    default_when_unspecified: bool = True,
) -> ModelPlan:
    """Resolve baseline selection while preserving legacy omission semantics.

    ``None`` means an aggregate caller omitted ``--models`` entirely and gets
    all default baselines.  An empty explicit list only gets defaults when no
    Semoco checkpoint was requested, matching the existing track-runner
    behavior.  ``_none_`` remains a supported cache-aggregate sentinel.
    """
    specs = tuple(semoco_specs)
    defaults = tuple(model for model in default_models if not _is_semoco_name(model))
    selected = tuple(
        model
        for model in (requested_models or ())
        if model not in {"", "_none_"} and not _is_semoco_name(model)
    )
    if not include_baselines:
        selected = ()
    elif requested_models is None:
        selected = defaults
    elif not selected and not specs and default_when_unspecified:
        selected = defaults
    return ModelPlan(selected, specs)


def build_track_model_plan(
    track: str,
    requested_models: Sequence[str] | None,
    semoco_specs: Sequence[CheckpointSpec],
    *,
    include_baselines: bool = True,
    default_when_unspecified: bool = True,
) -> ModelPlan:
    return build_model_plan(
        requested_models,
        semoco_specs,
        default_models=default_baselines(track),
        include_baselines=include_baselines,
        default_when_unspecified=default_when_unspecified,
    )


__all__ = [
    "ModelPlan",
    "ModelScoreEntry",
    "build_model_plan",
    "build_track_model_plan",
    "default_baselines",
]
