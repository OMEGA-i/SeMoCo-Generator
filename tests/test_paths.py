"""External-path resolution stays portable and explicitly overridable."""

from __future__ import annotations

import semoco_generator.paths as paths
from semoco_generator.eval.motion_ops.paths import llm2vec_merged_cache_root
from semoco_generator.paths import baseline_checkpoint_root, datasets_root, default_checkpoint


def test_dataset_and_baseline_roots_honor_environment_overrides(
    monkeypatch, tmp_path
) -> None:
    data_root = tmp_path / "datasets"
    checkpoint_root = tmp_path / "checkpoints"
    monkeypatch.setenv("MOTIONVERSE_DATA_ROOT", str(data_root))
    monkeypatch.setenv("SEMOCO_BASELINE_CKPT_ROOT", str(checkpoint_root))

    assert datasets_root() == data_root.resolve()
    assert baseline_checkpoint_root() == checkpoint_root.resolve()


def test_default_checkpoint_prefers_explicit_environment_override(monkeypatch, tmp_path):
    checkpoint = tmp_path / "tokenizer.pt"
    checkpoint.write_bytes(b"tokenizer")
    monkeypatch.setenv("SOMA_TOKENIZER_CHECKPOINT", str(checkpoint))

    assert default_checkpoint() == checkpoint.resolve()


def test_default_checkpoint_uses_configured_registry(monkeypatch, tmp_path):
    checkpoint = tmp_path / "registered-tokenizer.pt"
    checkpoint.write_bytes(b"tokenizer")
    monkeypatch.delenv("SOMA_TOKENIZER_CHECKPOINT", raising=False)
    monkeypatch.setattr(paths, "_registered_default_checkpoint", lambda: checkpoint)

    assert default_checkpoint() == checkpoint


def test_llm2vec_merged_cache_uses_baseline_root_and_honors_override(monkeypatch, tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    monkeypatch.setenv("SEMOCO_BASELINE_CKPT_ROOT", str(checkpoint_root))
    monkeypatch.delenv("LLM2VEC_MERGED_CACHE", raising=False)

    assert llm2vec_merged_cache_root() == (
        checkpoint_root / "runtime" / "llm2vec-merged"
    ).resolve()

    override = tmp_path / "fast-cache"
    monkeypatch.setenv("LLM2VEC_MERGED_CACHE", str(override))
    assert llm2vec_merged_cache_root() == override.resolve()
