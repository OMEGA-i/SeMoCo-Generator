"""``eval cache drop`` / ``eval cache rebuild`` administrative commands.

Kept separate from :mod:`cache_audit` (read-only) so the destructive path is
easy to review in isolation. Every removal here is dry-run by default; nothing
is deleted unless the caller explicitly opts in.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .cache_utils import (
    LEGACY_DURABLE_FAMILIES,
    discover_run_artifact_roots,
    require_durable_cache_root,
)
from .sharded_cache_store import ShardedCacheStore

LEGACY_RUN_ARTIFACT_FAMILIES = ("converted", "gen_emb")


@dataclass(frozen=True)
class DropTarget:
    path: str
    kind: str  # "legacy_durable" | "legacy_run_artifacts" | "manifest_index" | "v2_scope"
    size_bytes: int


def _du(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def plan_legacy_drop(
    *, cache_root: Path, runs_root: Path | None = None, include_manifest_index: bool = False,
) -> list[DropTarget]:
    """List (never delete) legacy per-file cache/run-artifact directories."""
    cache_root = require_durable_cache_root(cache_root, operation="legacy cache drop")
    targets: list[DropTarget] = []
    for name in LEGACY_DURABLE_FAMILIES:
        p = cache_root / name
        if p.is_dir():
            targets.append(DropTarget(path=str(p), kind="legacy_durable", size_bytes=_du(p)))
    if include_manifest_index:
        p = cache_root / "_manifest"
        if p.is_dir():
            targets.append(DropTarget(path=str(p), kind="manifest_index", size_bytes=_du(p)))
    if runs_root is not None:
        for run_artifacts in discover_run_artifact_roots(runs_root):
            for name in LEGACY_RUN_ARTIFACT_FAMILIES:
                p = run_artifacts / name
                if p.is_dir():
                    targets.append(DropTarget(path=str(p), kind="legacy_run_artifacts", size_bytes=_du(p)))
    return targets


def apply_drop(targets: list[DropTarget]) -> list[str]:
    """Actually remove every target directory. Returns the removed paths."""
    removed: list[str] = []
    for t in targets:
        p = Path(t.path)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed.append(t.path)
        elif p.is_file():
            p.unlink(missing_ok=True)
            removed.append(t.path)
    return removed


def plan_v2_scope_drop(*, v2_root: Path, scopes: list[str] | None = None) -> list[DropTarget]:
    """List (never delete) ShardedCacheStore v2 pack/index files for given scopes.

    If ``scopes`` is None, targets every scope currently present under
    ``v2_root``.
    """
    store = ShardedCacheStore(v2_root)
    want = scopes if scopes else store.scopes()
    targets: list[DropTarget] = []
    for scope in want:
        for p in store.drop(scope, dry_run=True):
            targets.append(DropTarget(path=p, kind="v2_scope", size_bytes=_du(Path(p))))
    return targets


def apply_v2_scope_drop(*, v2_root: Path, scopes: list[str] | None = None) -> list[str]:
    store = ShardedCacheStore(v2_root)
    want = scopes if scopes else store.scopes()
    removed: list[str] = []
    for scope in want:
        removed.extend(store.drop(scope, dry_run=False))
    return removed


__all__ = [
    "DropTarget",
    "LEGACY_DURABLE_FAMILIES",
    "LEGACY_RUN_ARTIFACT_FAMILIES",
    "apply_drop",
    "apply_v2_scope_drop",
    "plan_legacy_drop",
    "plan_v2_scope_drop",
]
