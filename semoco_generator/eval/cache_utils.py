"""Shared utilities for the eval cache pipeline.

Functions that are used by multiple cache modules live here so they aren't
copy-pasted.  This module has zero domain knowledge — pure formatting, string
sanitisation, and filesystem discovery helpers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

LEGACY_DURABLE_FAMILIES = ("hml_gt", "native", "tmr_gt", "tmr_text")
PACKED_CACHE_DIRNAME = "v2"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_bytes(n: float) -> str:
    """Human-readable byte count (e.g. ``"1.5G"``, ``"340K"``, ``"64B"``)."""
    x = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if x < 1024 or unit == "T":
            return f"{x:.1f}{unit}" if unit != "B" else f"{int(x)}B"
        x /= 1024
    return f"{x:.1f}T"


def cfg_str(cfg: float | None) -> str:
    """Format a CFG scale float for cache key embedding."""
    return "none" if cfg is None else f"{float(cfg):.4g}"


def safe(s: str) -> str:
    """Sanitize a string for use in path names and cache keys."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "x"


def require_durable_cache_root(root: Path, *, operation: str) -> Path:
    """Reject a packed-v2 directory where a durable cache root is required.

    Durable cache roots contain a ``v2/`` directory.  Passing that child to a
    legacy scan used to make valid packed scopes look like stale directories,
    and a subsequent legacy drop could remove live cache data.
    """
    path = Path(root)
    if path.name == PACKED_CACHE_DIRNAME or path.resolve().name == PACKED_CACHE_DIRNAME:
        raise ValueError(
            f"{operation} expects the durable cache root containing "
            f"'{PACKED_CACHE_DIRNAME}/', "
            f"not the packed-v2 directory: {path}. Pass {path.parent} instead."
        )
    return path


def packed_cache_root(durable_root: Path) -> Path:
    """Return the packed-store directory owned by a durable cache root."""
    return Path(durable_root) / PACKED_CACHE_DIRNAME


# ---------------------------------------------------------------------------
# Filesystem discovery
# ---------------------------------------------------------------------------

def discover_run_artifact_roots(runs_root: Path, *, max_depth: int = 4) -> list[Path]:
    """Find every ``run_artifacts`` directory under *runs_root* (bounded depth)."""
    if not runs_root.is_dir():
        return []
    found: list[Path] = []
    stack: list[tuple[Path, int]] = [(runs_root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            try:
                if not e.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            if e.name == "run_artifacts":
                found.append(Path(e.path))
                continue
            if depth < max_depth:
                stack.append((Path(e.path), depth + 1))
    return sorted(found)


def discover_logs(runs_root: Path) -> list[Path]:
    """Find all ``.log`` files under *runs_root*."""
    if not runs_root.is_dir():
        return []
    return sorted(runs_root.rglob("*.log"))


def discover_run_cache_v2_roots(runs_root: Path, *, max_depth: int = 4) -> list[Path]:
    """Find every ``run_artifacts/cache_v2`` directory under *runs_root*."""
    roots = discover_run_artifact_roots(runs_root, max_depth=max_depth)
    return [r / "cache_v2" for r in roots if (r / "cache_v2").is_dir()]


__all__ = [
    "cfg_str",
    "discover_logs",
    "discover_run_artifact_roots",
    "discover_run_cache_v2_roots",
    "fmt_bytes",
    "LEGACY_DURABLE_FAMILIES",
    "PACKED_CACHE_DIRNAME",
    "packed_cache_root",
    "require_durable_cache_root",
    "safe",
]
