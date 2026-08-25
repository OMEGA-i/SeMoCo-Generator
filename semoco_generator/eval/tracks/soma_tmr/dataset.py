"""T2M code-store subset loader for SOMA/TMR evaluation."""
from __future__ import annotations
from pathlib import Path
from ...datasets.release_subset import EvalClip, build_subset_from_t2m_store
from ....dataset import T2MCodeDataset


class SomaTMRDataset:
    def __init__(self, codes_root: str | Path, split: str = "test", *, text_encoder: str = "flan",
                 limit: int | None = None, seed: int = 0, max_tok: int = 125,
                 subset_map: dict[str, str] | None = None) -> None:
        self.codes = T2MCodeDataset(codes_root, split, text_encoder_key=text_encoder, max_motion_tok=max_tok)
        self.clips = build_subset_from_t2m_store(codes_root, split=split, limit=limit, seed=seed,
                                                  max_tokens=max_tok, subset_map=subset_map)

    def __len__(self) -> int: return len(self.clips)
    def clip(self, i: int) -> EvalClip: return self.clips[i]


__all__ = ["SomaTMRDataset"]
