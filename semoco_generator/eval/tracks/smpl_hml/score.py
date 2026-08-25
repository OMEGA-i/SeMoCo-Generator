"""Metric packing for HumanML3D ``text_mot_match`` embeddings."""
from __future__ import annotations
import numpy as np
from ...metrics import compute_embedding_metrics


def score_embeddings(gen: np.ndarray, gt: np.ndarray, *, text: np.ndarray | None = None,
                     retrieval_protocol: str = "full_gallery", seed: int = 0,
                     kimodo_metrics: bool = True) -> dict[str, float]:
    return compute_embedding_metrics(gen_emb=gen, gt_emb=gt, text_emb=text,
                                     retrieval_protocol=retrieval_protocol, seed=seed,
                                     kimodo_metrics=kimodo_metrics)


__all__ = ["score_embeddings"]
