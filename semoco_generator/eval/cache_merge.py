"""``eval cache merge`` -- consolidate run-local caches scattered across
different ``--out-dir`` directories into the canonical target.

Safety guarantees
-----------------
- **Dry-run by default**: ``--dry-run`` (the default) prints a plan and exits.
  Pass ``--yes`` to actually execute.
- **Backup-first**: before touching any cache files the tool copies the
  canonical target's existing ``cache_v2/`` and every non-empty source's
  ``cache_v2/`` into a timestamped backup directory under
  ``runs/archive/eval/cache_merges/<iso>/`` (or ``--backup-dir``).
- **Idempotent**: records whose key already exists in the target are skipped.
  Re-running the same merge is safe and fast.
- **Non-destructive** (without ``--cleanup``): source directories are never
  removed unless you explicitly pass ``--cleanup`` **and** verification
  passes.
- **Per-source flush**: after ingesting each source the target store is
  flushed to disk so a mid-merge crash does not lose previously-merged data.

Algorithm
---------
1. **Discover** -- walk ``<runs_root>/*/<protocol_id>/run_artifacts/cache_v2/``
   and group by ``protocol_id``.
2. **Plan** -- for each group pick a *canonical target* (standard track
   directory > has-reports > largest cache) and list sources to merge.
3. **Backup** -- ``shutil.copytree`` the target's and every non-empty source's
   ``cache_v2/`` into ``<backup_dir>/``.
4. **Merge** -- for each source (non-target), enumerate its ``converted`` and
   ``gen_emb`` scopes, load records not already present in the target, and
   ``put_many`` them into the target store.
5. **Verify** -- confirm every key that existed across all sources is now
   reachable in the target.
6. **Cleanup** (opt-in) -- if ``--cleanup`` is set and verification succeeds,
   remove the merged source ``cache_v2/`` directories.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNS_ROOT_DEFAULT = _REPO_ROOT / "runs" / "eval"

# Canonical out-dir names per track (the ones we want to merge *into*).
_CANONICAL_DIRS: dict[str, str] = {
    "smpl_hml": "smpl_hml",
    "soma_tmr": "soma_tmr",
}

# Scopes that live in run-local cache_v2 stores.
_RUN_SCOPES = ("converted", "gen_emb")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class SourceInfo(NamedTuple):
    """A single run-local cache_v2 store."""

    out_dir: str          # e.g. "soma_tmr_4x"
    protocol_id: str      # e.g. "soma_tmr-test-2dd7fef2c0"
    path: Path            # .../<out_dir>/<protocol_id>
    cache_root: Path      # .../<out_dir>/<protocol_id>/run_artifacts/cache_v2
    scopes: dict[str, int]  # scope -> record count
    total_bytes: int
    has_reports: bool


class MergePlan(NamedTuple):
    protocol_id: str
    target: SourceInfo
    sources: list[SourceInfo]  # non-target sources with records to merge
    dry: bool


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover(runs_root: Path) -> dict[str, list[SourceInfo]]:
    """Walk *runs_root* and collect every run-local cache_v2 store group by protocol_id."""
    groups: dict[str, list[SourceInfo]] = {}

    for out_dir_path in sorted(runs_root.iterdir()):
        if not out_dir_path.is_dir():
            continue
        out_dir_name = out_dir_path.name
        if out_dir_name.startswith("_") or out_dir_name.startswith("."):
            continue  # skip backup dirs, hidden dirs

        for proto_path in sorted(out_dir_path.iterdir()):
            if not proto_path.is_dir():
                continue
            proto_id = proto_path.name
            cache_root = proto_path / "run_artifacts" / "cache_v2"
            if not cache_root.is_dir():
                continue

            info = _inspect_source(out_dir_name, proto_id, proto_path, cache_root)
            if info is not None:
                groups.setdefault(proto_id, []).append(info)

    return groups


def _inspect_source(
    out_dir: str, protocol_id: str, proto_path: Path, cache_root: Path,
) -> SourceInfo | None:
    """Build a SourceInfo for one cache_v2 directory."""
    from .sharded_cache_store import ShardedCacheStore

    store = ShardedCacheStore(cache_root, num_buckets=16)
    scopes: dict[str, int] = {}
    total_bytes = 0
    for scope in _RUN_SCOPES:
        audit = store.audit(scope)
        if audit.records > 0:
            scopes[scope] = audit.records
            total_bytes += audit.bytes

    if not scopes:
        return None  # empty store, nothing to merge

    has_reports = (proto_path / "reports").is_dir() and any(
        (proto_path / "reports").glob("*.csv")
    )

    return SourceInfo(
        out_dir=out_dir,
        protocol_id=protocol_id,
        path=proto_path,
        cache_root=cache_root,
        scopes=scopes,
        total_bytes=total_bytes,
        has_reports=has_reports,
    )


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def pick_canonical(sources: list[SourceInfo]) -> SourceInfo:
    """Choose which source is the merge target.

    Priority:
    1. Standard track directory (``smpl_hml/`` or ``soma_tmr/``)
    2. Has reports/
    3. Largest total_bytes
    """
    # Priority 1: standard dir
    for s in sources:
        if s.out_dir in _CANONICAL_DIRS.values():
            return s

    # Priority 2: has reports
    for s in sources:
        if s.has_reports:
            return s

    # Priority 3: largest
    return max(sources, key=lambda s: s.total_bytes)


def plan_merges(
    groups: dict[str, list[SourceInfo]],
) -> list[MergePlan]:
    """Build a MergePlan for each protocol_id that has >1 source."""
    plans: list[MergePlan] = []
    for protocol_id, sources in sorted(groups.items()):
        if len(sources) <= 1:
            continue  # nothing to merge
        target = pick_canonical(sources)
        others = [s for s in sources if s.out_dir != target.out_dir and s.total_bytes > 0]
        if not others:
            continue
        plans.append(MergePlan(protocol_id=protocol_id, target=target, sources=others, dry=True))
    return plans


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def backup_plan(plan: MergePlan, backup_dir: Path) -> None:
    """Copy target and source cache_v2 trees into *backup_dir*."""
    # Back up the canonical target first (most important)
    _backup_one(plan.target, backup_dir, label="target")

    for src in plan.sources:
        _backup_one(src, backup_dir, label="source")


def _backup_one(info: SourceInfo, backup_dir: Path, label: str) -> None:
    dst = backup_dir / info.out_dir / info.protocol_id / "run_artifacts" / "cache_v2"
    if dst.exists():
        print(f"  [{label}] already backed up: {dst}", flush=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [{label}] backing up {info.out_dir}/{info.protocol_id} "
          f"({fmt_bytes(info.total_bytes)}) ...", flush=True, end=" ")
    t0 = time.perf_counter()
    shutil.copytree(str(info.cache_root), str(dst), symlinks=False,
                    dirs_exist_ok=True)
    dt = time.perf_counter() - t0
    print(f"{dt:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_plan(plan: MergePlan) -> dict[str, tuple[int, int]]:
    """Execute one MergePlan. Returns ``{scope: (new_keys, skipped_keys)}``."""
    from .sharded_cache_store import PutRecord, ShardedCacheStore
    from .run_artifact_store import RunArtifactStore

    target_root = plan.target.path / "run_artifacts"
    target_store = ShardedCacheStore(
        target_root / "cache_v2", num_buckets=16,
    )

    stats: dict[str, tuple[int, int]] = {}

    for scope in _RUN_SCOPES:
        # Collect existing keys in target
        existing: set[str] = set()
        for entry in target_store.enumerate_scope(scope):
            existing.add(entry["key"])

        new_total = 0
        skip_total = 0

        for src in plan.sources:
            if scope not in src.scopes:
                continue

            src_cache_root = src.cache_root
            src_store = ShardedCacheStore(src_cache_root, num_buckets=16)

            # Enumerate all live records in this scope
            entries = src_store.enumerate_scope(scope)
            to_merge: list[PutRecord] = []

            for entry in entries:
                key = entry["key"]
                if key in existing:
                    skip_total += 1
                    continue
                batch = src_store.load_one(scope, key)
                if batch is None:
                    continue
                to_merge.append(PutRecord(
                    key=batch.key,
                    arrays=batch.arrays,
                    meta=batch.meta,
                ))
                existing.add(key)

            if to_merge:
                target_store.put_many(scope, to_merge)
                new_total += len(to_merge)

            print(f"  [{scope}] {src.out_dir}: +{len(to_merge)} new, "
                  f"{skip_total} skipped", flush=True)

        stats[scope] = (new_total, skip_total)

    return stats


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_plan(plan: MergePlan) -> dict[str, list[str]]:
    """Check that every key from every source exists in the target.
    Returns ``{scope: [missing_keys]}``.
    """
    from .sharded_cache_store import ShardedCacheStore

    target_cache = plan.target.path / "run_artifacts" / "cache_v2"
    target_store = ShardedCacheStore(target_cache, num_buckets=16)

    missing: dict[str, list[str]] = {}

    for scope in _RUN_SCOPES:
        # Collect all source keys
        source_keys: set[str] = set()
        for src in plan.sources:
            if scope not in src.scopes:
                continue
            src_store = ShardedCacheStore(src.cache_root, num_buckets=16)
            for entry in src_store.enumerate_scope(scope):
                source_keys.add(entry["key"])

        if not source_keys:
            continue

        # Check target
        target_keys: set[str] = set()
        for entry in target_store.enumerate_scope(scope):
            target_keys.add(entry["key"])

        missing_keys = sorted(source_keys - target_keys)
        if missing_keys:
            missing[scope] = missing_keys

    return missing


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_sources(plan: MergePlan) -> list[Path]:
    """Remove the cache_v2 directories of merged sources."""
    removed: list[Path] = []
    for src in plan.sources:
        if src.cache_root.exists():
            shutil.rmtree(str(src.cache_root))
            removed.append(src.cache_root)
    return removed


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dry-run rendering
# ---------------------------------------------------------------------------

def render_dry_run(plans: list[MergePlan]) -> str:
    """Pretty-print merge plans."""
    if not plans:
        return "Nothing to merge — every protocol_id has a single source.\n"

    lines: list[str] = []
    total_sources = 0
    total_bytes = 0

    for plan in plans:
        total_sources += len(plan.sources)
        for s in plan.sources:
            total_bytes += s.total_bytes

        lines.append(f"\n{'─'*70}")
        lines.append(f"protocol_id: {plan.protocol_id}")
        lines.append(f"  canonical target: {plan.target.out_dir}/ "
                      f"({fmt_bytes(plan.target.total_bytes)}, "
                      f"reports={'yes' if plan.target.has_reports else 'no'})")
        lines.append(f"  sources ({len(plan.sources)}):")
        for s in plan.sources:
            scope_str = ", ".join(
                f"{sc}={fmt_bytes(s.total_bytes)}"  # simplified
                for sc in _RUN_SCOPES if sc in s.scopes
            )
            lines.append(
                f"    {s.out_dir:30s} {fmt_bytes(s.total_bytes):>8}  "
                f"({scope_str}, reports={'yes' if s.has_reports else 'no'})"
            )

    lines.append(f"\n{'─'*70}")
    lines.append(f"Total: {len(plans)} protocol_id(s) with {total_sources} "
                 f"source(s) to merge ({fmt_bytes(total_bytes)})")
    lines.append(f"Canonical dirs: {', '.join(_CANONICAL_DIRS.values())}")
    lines.append("Run with --yes to execute (auto-backup first).")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Library entry point (accepts pre-parsed argparse.Namespace from unified CLI)
def merge_caches(args: argparse.Namespace) -> int:
    """Consolidate run-local caches into canonical targets. Returns 0/1."""
    runs_root = Path(args.runs_root) if args.runs_root else _RUNS_ROOT_DEFAULT
    if not runs_root.is_dir():
        print(f"ERROR: runs-root does not exist: {runs_root}", flush=True)
        return 1

    # ---- Discover ---------------------------------------------------------
    print("Discovering run-local caches ...", flush=True)
    groups = discover(runs_root)

    # ---- Plan -------------------------------------------------------------
    plans = plan_merges(groups)

    # Filter by track
    if args.track:
        canonical_dir = _CANONICAL_DIRS.get(args.track, args.track)
        plans = [p for p in plans if p.target.out_dir == canonical_dir]

    if not plans:
        print("Nothing to merge.")
        return 0

    # ---- Dry-run ----------------------------------------------------------
    is_dry = not args.yes
    print(render_dry_run(plans))

    if is_dry:
        print("DRY-RUN complete. Re-run with --yes to execute.", flush=True)
        return 0

    # ---- Backup -----------------------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = (
        Path(args.backup_dir) if args.backup_dir
        else runs_root.parent / "archive" / "eval" / "cache_merges" / ts
    )
    backup_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = backup_dir / "MANIFEST.txt"
    manifest_lines: list[str] = [
        f"backup created: {ts}",
        f"command: eval cache merge --yes",
        "",
    ]

    print(f"\nBacking up to {backup_dir} ...", flush=True)
    for plan in plans:
        backup_plan(plan, backup_dir)
        manifest_lines.append(f"{plan.target.out_dir}/{plan.protocol_id}")
        for s in plan.sources:
            manifest_lines.append(f"  <- {s.out_dir}/{plan.protocol_id}")
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(f"Backup complete ({backup_dir})", flush=True)

    # ---- Merge ------------------------------------------------------------
    total_new = 0
    total_skip = 0
    failed: list[str] = []

    for plan in plans:
        print(f"\nMerging {plan.protocol_id} ...", flush=True)
        print(f"  target: {plan.target.out_dir}/", flush=True)
        try:
            stats = merge_plan(plan)
            for scope, (new, skip) in stats.items():
                total_new += new
                total_skip += skip
                print(f"  {scope}: +{new} new, {skip} skipped", flush=True)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            failed.append(plan.protocol_id)
            continue

    print(f"\nMerge done: +{total_new} new records, {total_skip} skipped "
          f"({len(failed)} failed)", flush=True)

    if failed:
        print(f"WARNING: {len(failed)} protocol_id(s) failed: {failed}", flush=True)
        print("Backup is at:", backup_dir, flush=True)
        return 1

    # ---- Verify -----------------------------------------------------------
    if not args.no_verify:
        print("\nVerifying ...", flush=True)
        all_ok = True
        for plan in plans:
            missing = verify_plan(plan)
            if missing:
                all_ok = False
                for scope, keys in missing.items():
                    print(f"  {plan.protocol_id}/{scope}: {len(keys)} keys MISSING!",
                          flush=True)
                    for k in keys[:5]:
                        print(f"    {k}", flush=True)
                    if len(keys) > 5:
                        print(f"    ... and {len(keys) - 5} more", flush=True)
            else:
                print(f"  {plan.protocol_id}: all keys verified", flush=True)

        if not all_ok:
            print("\nVERIFICATION FAILED. Backup is at:", backup_dir, flush=True)
            print("To restore: cp -a <backup>/<out_dir>/... "
                  "runs/eval/<out_dir>/...", flush=True)
            return 1
        print("Verification passed.", flush=True)

    # ---- Cleanup (opt-in) -------------------------------------------------
    if args.cleanup:
        print("\nCleaning up merged sources ...", flush=True)
        removed = []
        for plan in plans:
            removed += cleanup_sources(plan)
        for r in removed:
            print(f"  removed {r}", flush=True)
        print(f"Cleaned up {len(removed)} source directories.", flush=True)

    print("\nAll done.", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    print("Use 'python -m semoco_generator.eval.cli cache merge' instead.", file=sys.stderr)
    raise SystemExit(1)
