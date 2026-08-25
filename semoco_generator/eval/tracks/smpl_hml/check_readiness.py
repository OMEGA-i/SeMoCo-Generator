"""Validate assets required for the HumanML3D evaluation track."""

from __future__ import annotations

from pathlib import Path

from .paths import count_humanml_assets
from .protocol import DEFAULT_CHECKPOINT, DEFAULT_MEAN_STD
from .word_vectorizer import resolve_glove_root


def check_readiness(
    data_root: str | Path,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    *,
    mean_std_dir: str | Path = DEFAULT_MEAN_STD,
    glove_root: str | Path | None = None,
    require_text: bool = True,
) -> list[str]:
    """Return missing asset paths / labels for the smpl_hml track.

    When ``require_text`` is True (default), GloVe / WordVectorizer assets are
    required so R-precision / Matching can run. Motion-only FID/Diversity can
    set ``require_text=False``.
    Accepts both flat and two-digit-sharded ``texts/`` trees.
    """
    root = Path(data_root)
    required = [
        root / "test.txt",
        root / "texts",
        root / "new_joint_vecs",
        root / "Mean.npy",
        root / "Std.npy",
        Path(checkpoint),
        Path(mean_std_dir) / "mean.npy",
        Path(mean_std_dir) / "std.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]

    _, n_vecs = count_humanml_assets(root, "new_joint_vecs")
    if (root / "new_joint_vecs").is_dir() and n_vecs == 0:
        missing.append(f"{root / 'new_joint_vecs'} (empty)")

    _, n_texts = count_humanml_assets(root, "texts")
    if (root / "texts").is_dir() and n_texts == 0:
        missing.append(f"{root / 'texts'} (empty)")

    if require_text:
        candidates = [glove_root] if glove_root else None
        glove = resolve_glove_root(candidates)
        if glove is None:
            missing.append(
                "glove/our_vab_{data.npy,words.pkl,idx.pkl} "
                "(WordVectorizer; set --glove-root or place under <data-root>/glove)"
            )
    return missing


__all__ = ["check_readiness"]
