"""Tests for the soma_tmr text-encode tail fix: a worker whose caption shard
has nothing missing must never touch the (~16 GB) LLM2Vec encoder.

``main()`` in ``tracks/soma_tmr/runner.py`` now computes the missing-caption
list itself and only loads LLM2Vec + calls ``_encode_text_shard`` when that
list is non-empty. These tests exercise ``_encode_text_shard`` directly
(the function main() gates on) to confirm the missing-detection and the
early-return-without-touching-``tmr`` behavior it relies on."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import torch

from semoco_generator.eval.tracks.soma_tmr.runner import _encode_text_shard


@dataclass
class _FakeClip:
    caption: str


class _ExplodingTMR:
    """Stands in for the real TMR/LLM2Vec model — raises if ever called, so
    tests can assert the encode path was never touched."""

    def encode_raw_text(self, *args, **kwargs):
        raise AssertionError("encode_raw_text must not be called when nothing is missing")


class _Args:
    tmr_model = "soma-rp"
    text_batch_size = 64
    shard_index = 0


def test_encode_text_shard_skips_tmr_when_nothing_missing():
    selected = [_FakeClip("a person walks forward"), _FakeClip("a person waves")]
    with patch("semoco_generator.eval.tracks.soma_tmr.runner.C.probe_tmr_text", return_value=True):
        _encode_text_shard(_ExplodingTMR(), selected, _Args())  # must not raise
    print("_encode_text_shard skips TMR/LLM2Vec when nothing missing OK")


def test_encode_text_shard_skips_tmr_with_precomputed_empty_missing():
    """Mirrors the main() gating path exactly: caller precomputes ``missing``
    once and only reaches this function (and only loads LLM2Vec upstream) if
    it is non-empty. An empty explicit list must short-circuit identically to
    the auto-detected case above."""
    selected = [_FakeClip("a person sits down")]
    _encode_text_shard(_ExplodingTMR(), selected, _Args(), missing=[])  # must not raise
    print("_encode_text_shard skips TMR with precomputed empty missing list OK")


def test_encode_text_shard_encodes_only_missing_captions():
    selected = [_FakeClip("cached caption"), _FakeClip("new caption")]
    saved: list[tuple[str, str, np.ndarray]] = []

    def fake_probe(tmr_model, caption):
        return caption == "cached caption"  # only "new caption" is missing

    def fake_save(tmr_model, caption, emb):
        saved.append((tmr_model, caption, emb))

    class _StubTMR:
        def encode_raw_text(self, batch, unit_vector=True):
            assert batch == ["new caption"]
            return torch.ones(len(batch), 4)

    with patch("semoco_generator.eval.tracks.soma_tmr.runner.C.probe_tmr_text", side_effect=fake_probe), \
         patch("semoco_generator.eval.tracks.soma_tmr.runner.C.save_tmr_text", side_effect=fake_save):
        _encode_text_shard(_StubTMR(), selected, _Args())

    assert len(saved) == 1
    assert saved[0][1] == "new caption"
    print("_encode_text_shard encodes only the missing captions OK")


if __name__ == "__main__":
    test_encode_text_shard_skips_tmr_when_nothing_missing()
    test_encode_text_shard_skips_tmr_with_precomputed_empty_missing()
    test_encode_text_shard_encodes_only_missing_captions()
    print("\nALL TMR TEXT TAIL TESTS PASSED")
