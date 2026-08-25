"""Runtime cache locations for the retrieval-teacher text encoder."""

from __future__ import annotations

import os
from pathlib import Path

from ...paths import baseline_checkpoint_root


def llm2vec_merged_cache_root() -> Path:
    """Return the writable LLM2Vec merged-weight cache location.

    This derived runtime artifact belongs with the external checkpoints rather
    than in a developer-specific home-directory cache. The dedicated override
    exists for shared installations with a separate fast scratch volume.
    """
    env = os.environ.get("LLM2VEC_MERGED_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return baseline_checkpoint_root() / "runtime" / "llm2vec-merged"
