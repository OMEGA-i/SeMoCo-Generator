"""Model-plan compatibility coverage for runner and aggregate callers."""

from __future__ import annotations

from pathlib import Path

from semoco_generator.eval.cache_aggregate import build_model_entries
from semoco_generator.eval.checkpoints import CheckpointSpec
from semoco_generator.eval.model_plan import build_model_plan, build_track_model_plan


def _semoco_spec() -> CheckpointSpec:
    return CheckpointSpec(
        name="semoco-test",
        model_ckpt=Path("/models/semoco.pt"),
        tokenizer_ckpt=Path("/models/tokenizer.pt"),
        codes_root=Path("/codes"),
        text_encoder="flan",
        max_tok=125,
    )


def test_track_plan_defaults_to_registered_baselines_without_semoco() -> None:
    """No baseline ships built in, so the default plan is empty until one is registered."""
    plan = build_track_model_plan("soma_tmr", [], [])

    assert plan.baselines == ()
    assert plan.model_ids == []


def test_explicit_semoco_plan_does_not_add_runner_baselines() -> None:
    plan = build_model_plan(
        [],
        [_semoco_spec()],
        default_models=("baseline5", "semoco"),
    )

    assert plan.baselines == ()
    assert plan.model_ids == ["semoco-test"]
    assert plan.scoring_tuples() == [
        (
            "semoco",
            {
                "checkpoint": "/models/semoco.pt",
                "tokenizer_checkpoint": "/models/tokenizer.pt",
                "max_tok": 125,
                "codes_root": "/codes",
            },
            "semoco-test",
        ),
    ]


def test_cache_aggregate_preserves_empty_models_sentinel() -> None:
    assert build_model_entries([], [], track="smpl_hml") == []
    defaults = build_model_entries(None, [], track="smpl_hml")
    assert [entry[0] for entry in defaults] == []
