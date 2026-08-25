"""Tests for generation and conversion performance logging."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from semoco_generator.eval.generation import ensure_native, ensure_target
from semoco_generator.eval.schema import (
    LengthSpec,
    ModelInput,
    ModelOutput,
    ModelSchema,
    MotionClip,
    TrackInput,
)


class _FakeModel:
    def __init__(self) -> None:
        self.schema = ModelSchema(
            model_id="fake",
            role="ours",
            text_input="raw_caption",
            length_input="seconds",
            native_output="motion_codes",
            native_fps=30.0,
            max_safe_batch=8,
        )

    def weight_signature(self) -> str:
        return "sig"

    def generate(self, inputs: list[ModelInput]) -> list[ModelOutput]:
        return [
            ModelOutput(
                model_id="fake",
                prompt_id=item.prompt_id,
                seed=item.seed,
                native_motion=MotionClip(
                    rep="motion_codes",
                    array=np.zeros((5, 2), dtype=np.int64),
                    fps=30.0,
                ),
            )
            for item in inputs
        ]


class _FakeGraph:
    def convert_batch(self, natives, target_rep, ctx):
        del ctx
        return [
            MotionClip(
                rep=target_rep,
                array=np.zeros((7, 22, 3), dtype=np.float32),
                fps=20.0,
            )
            for _native in natives
        ]


def _track_inputs() -> list[TrackInput]:
    return [
        TrackInput(
            prompt_id=prompt_id,
            rec_id=prompt_id,
            caption=caption,
            length=LengthSpec(seconds=1.0),
        )
        for prompt_id, caption in (("a", "alpha"), ("b", "beta"))
    ]


def test_ensure_native_logs_performance() -> None:
    output = io.StringIO()
    with patch(
        "semoco_generator.eval.artifacts.C.save_native_many"
    ), redirect_stdout(output):
        ensure_native(
            _FakeModel(),
            _track_inputs(),
            seeds=[0],
            cfg_scale=None,
            batch_size=2,
            skip_existing=False,
        )
    text = output.getvalue()
    assert "clips/s=" in text
    assert "units/s=" in text
    assert "[fake] native seed=0" in text


def test_ensure_target_logs_performance() -> None:
    native = MotionClip(
        rep="motion_codes",
        array=np.zeros((5, 2), dtype=np.int64),
        fps=30.0,
    )
    natives = {(prompt_id, 0, None): native for prompt_id in ("a", "b")}
    output = io.StringIO()
    with patch(
        "semoco_generator.eval.artifacts.C.load_native_many",
        return_value=natives,
    ), patch(
        "semoco_generator.eval.artifacts.C.save_converted_many"
    ), redirect_stdout(output):
        ensure_target(
            _FakeModel(),
            _track_inputs(),
            target_rep="hml263",
            graph=_FakeGraph(),
            ctx=object(),
            seeds=[0],
            cfg_scale=None,
            skip_existing=False,
        )
    text = output.getvalue()
    assert "convert seed=0 rep=motion_codes" in text
    assert "clips/s=" in text
    assert "units/s=" in text
