"""``local://`` URI helpers, matching the tokenizer repo's convention.

Every long-lived data artifact is addressed under a single ``data_root`` via a
``local://`` URI, so configs stay portable across machines::

    local://recordings/<recording_id>/umr499.npz
    local://manifests/<scale>/{all,train,val,test}.txt
    local://t2m_codes/<split>.codes.npy        # exported code store

URIs resolve against, in priority order: an explicit ``data_root`` argument,
the ``MOTIONVERSE_DATA_ROOT`` environment variable (shared with the tokenizer
repo so both resolve to the same root), or the sibling ``semoco-MotionVerse/``
directory.

Plain absolute / relative filesystem paths pass through unchanged, so the
helpers are safe to wrap around any path argument.
"""

from __future__ import annotations

import os
from pathlib import Path

from .paths import ENV_DATA_ROOT, datasets_root

LOCAL_URI_PREFIX = "local://"


def default_data_root() -> Path:
    """Ambient ``data_root``: ``$MOTIONVERSE_DATA_ROOT`` if set, else the dataset root."""
    env = os.environ.get(ENV_DATA_ROOT)
    return Path(env) if env else datasets_root()


def resolve_local_uri(uri: str | Path, data_root: str | Path | None = None) -> Path:
    """Resolve ``local://X`` -> ``<data_root>/X``; pass-through otherwise."""
    s = str(uri)
    if s.startswith(LOCAL_URI_PREFIX):
        rel = s[len(LOCAL_URI_PREFIX):]
        root = Path(data_root) if data_root is not None else default_data_root()
        return root / rel
    return Path(s)


def to_local_uri(path: str | Path, data_root: str | Path | None = None) -> str:
    """Inverse: ``<data_root>/X`` -> ``local://X`` when ``path`` is under root."""
    p = Path(path)
    root = Path(data_root) if data_root is not None else default_data_root()
    try:
        rel = p.resolve().relative_to(root.resolve())
        return f"{LOCAL_URI_PREFIX}{rel}"
    except (ValueError, OSError):
        return str(p)


__all__ = [
    "ENV_DATA_ROOT",
    "LOCAL_URI_PREFIX",
    "default_data_root",
    "resolve_local_uri",
    "to_local_uri",
]
