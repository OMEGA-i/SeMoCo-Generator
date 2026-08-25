"""Tests for Semoco eval batching / independent sampling streams."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import torch

from semoco_generator.eval.models.semoco import SemocoModel
from semoco_generator.eval.rollout import _sample_one
from semoco_generator.eval.schema import LengthSpec, ModelInput


def test_semoco_default_eos_thresh_forces_full_target_tok_decode():
    """Regression test for the GT-duration-alignment bug: every other eval
    baseline forces its decode length to exactly match the GT-derived target
    duration, but Semoco's learned EOS head could previously stop generation
    early (observed on ~20% of SOMA clips, often near half the requested
    length), which is not a fair length-matched comparison. The default must
    be unreachable by a sigmoid probability (bounded in [0, 1]) so decoding
    always runs the full `target_tok` packets."""
    default = inspect.signature(SemocoModel.__init__).parameters["eos_thresh"].default
    assert default > 1.0, f"default eos_thresh={default} can still trigger early EOS stop"
    print("semoco default eos_thresh forces full target_tok decode OK")


def test_semoco_weight_signature_matches_free_function_form():
    """Regression test for a real bug hit while validating this fix: the CPU
    aggregation pass recomputes each model's signature via the free-function
    `registry.weight_signature(name, **kwargs)` *without* loading the model
    (see tracks/*/runner.py `_aggregate`), while generation uses the loaded
    instance's `model.weight_signature()`. These two silently went out of
    sync when `eos_thresh` was added only to the instance method, and
    aggregate found "0 run-local gens" for every single model. Both must
    delegate to the exact same underlying function so they can never
    desync again."""
    from semoco_generator.eval.models.registry import weight_signature as free_weight_signature

    model = SemocoModel.__new__(SemocoModel)
    model._checkpoint_path = "/fake/ckpt.pt"
    model._tokenizer_path = "/fake/tok.pt"
    model.max_tok = 125
    model.eos_thresh = 1.01
    instance_sig = model.weight_signature()
    free_sig = free_weight_signature(
        "semoco", checkpoint="/fake/ckpt.pt", tokenizer_checkpoint="/fake/tok.pt",
        max_tok=125, eos_thresh=1.01,
    )
    assert instance_sig == free_sig
    # And critically: the aggregation call site never passes `eos_thresh`
    # explicitly (no CLI flag for it exists), so the free function's default
    # must match SemocoModel's constructor default or this desyncs again.
    default_eos = inspect.signature(SemocoModel.__init__).parameters["eos_thresh"].default
    free_sig_no_eos_kwarg = free_weight_signature(
        "semoco", checkpoint="/fake/ckpt.pt", tokenizer_checkpoint="/fake/tok.pt", max_tok=125,
    )
    assert free_sig_no_eos_kwarg == free_weight_signature(
        "semoco", checkpoint="/fake/ckpt.pt", tokenizer_checkpoint="/fake/tok.pt",
        max_tok=125, eos_thresh=default_eos,
    )
    print("semoco weight_signature matches free function form OK")


def test_semoco_weight_signature_is_namespaced_by_eos_thresh():
    """The native/converted/gen-embed cache key is derived from
    weight_signature(); it must change if eos_thresh changes so stale
    early-stopped generations from before this fix are never silently
    reused (they simply become unreachable, orphaned cache entries)."""
    model = SemocoModel.__new__(SemocoModel)
    model._checkpoint_path = "/fake/ckpt.pt"
    model._tokenizer_path = "/fake/tok.pt"
    model.max_tok = 125
    model.eos_thresh = 1.01
    sig_a = model.weight_signature()
    model.eos_thresh = 0.5
    sig_b = model.weight_signature()
    assert sig_a != sig_b
    assert "eos1.01" in sig_a
    assert "eos0.5" in sig_b
    print("semoco weight_signature namespaced by eos_thresh OK")


def test_sample_one_with_generators_matches_per_row_sampling():
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.25, 4.0]], dtype=torch.float32)
    gens = [torch.Generator().manual_seed(123), torch.Generator().manual_seed(123)]
    batched = _sample_one(logits, 1.0, 1.0, generators=gens)

    expected_rows = []
    for row in logits:
        gen = torch.Generator().manual_seed(123)
        expected_rows.append(_sample_one(row.unsqueeze(0), 1.0, 1.0, generators=[gen])[0])
    expected = torch.stack(expected_rows)
    assert torch.equal(batched, expected)
    print("sample_one generators preserve per-row sampling OK")


def test_semoco_generate_batches_by_cfg_and_trims_to_target_tok():
    model = SemocoModel.__new__(SemocoModel)
    model.device = torch.device("cpu")
    model.fps = 50.0
    model.max_tok = 125
    model.eos_thresh = 1.01
    model._text_encoder_key = "flan"
    model.model = object()

    def fake_text_batch(prompts):
        b = len(prompts)
        return torch.zeros((b, 2, 3), dtype=torch.float32), torch.ones((b, 2), dtype=torch.bool)

    model._text_batch = fake_text_batch

    calls: list[tuple[int, float, int]] = []

    def fake_generate_from_text(
        _model, text_emb, text_valid, *, max_tok, cfg_scale, eos_thresh, device, generators,
    ):
        assert text_emb.shape[0] == len(generators)
        assert text_valid.shape[0] == text_emb.shape[0]
        assert eos_thresh == model.eos_thresh
        calls.append((text_emb.shape[0], float(cfg_scale), int(max_tok)))
        q = 4
        return [torch.zeros((max_tok, q), dtype=torch.long) + i for i in range(text_emb.shape[0])]

    inputs = [
        ModelInput(prompt_id="a", text="p0", length=LengthSpec(seconds=2.0), seed=7, cfg_scale=None),   # tok 25, cfg 3.0
        ModelInput(prompt_id="b", text="p1", length=LengthSpec(seconds=2.0), seed=7, cfg_scale=None),   # same bucket
        ModelInput(prompt_id="c", text="p2", length=LengthSpec(seconds=4.0), seed=7, cfg_scale=None),   # tok 50
        ModelInput(prompt_id="d", text="p3", length=LengthSpec(seconds=2.0), seed=7, cfg_scale=1.5),    # cfg split
    ]

    with patch("semoco_generator.eval.models.semoco.generate_from_text", side_effect=fake_generate_from_text):
        outputs = model.generate(inputs)

    assert sorted(calls) == sorted([(3, 3.0, 50), (1, 1.5, 25)])
    by_id = {o.prompt_id: o for o in outputs}
    assert by_id["a"].native_motion is not None and by_id["a"].native_motion.array.shape[0] == 25
    assert by_id["b"].native_motion is not None and by_id["b"].native_motion.array.shape[0] == 25
    assert by_id["c"].native_motion is not None and by_id["c"].native_motion.array.shape[0] == 50
    assert by_id["d"].native_motion is not None and by_id["d"].provenance["effective_cfg"] == 1.5
    assert by_id["a"].provenance["batch_size"] == 3
    print("semoco generate batching by cfg and per-sample trimming OK")


if __name__ == "__main__":
    test_sample_one_with_generators_matches_per_row_sampling()
    test_semoco_default_eos_thresh_forces_full_target_tok_decode()
    test_semoco_weight_signature_matches_free_function_form()
    test_semoco_weight_signature_is_namespaced_by_eos_thresh()
    test_semoco_generate_batches_by_cfg_and_trims_to_target_tok()
    print("\nALL SEMOCO BATCHING TESTS PASSED")
