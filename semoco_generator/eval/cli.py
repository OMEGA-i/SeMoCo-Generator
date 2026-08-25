"""Unified ``eval`` CLI entry point.

Usage::

    python -m semoco_generator.eval.cli cache audit [--root ...] [--runs-root ...]

Subcommands::

    python -m semoco_generator.eval.cli cache audit

The command group is deliberately named ``cache`` (not ``cache_v2`` /
``eval_cache_v2``): the CLI surface stays stable even as the storage
implementation behind it changes across pipeline phases.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .cache_admin import apply_drop, apply_v2_scope_drop, plan_legacy_drop, plan_v2_scope_drop
from .cache_audit import render_text, run_audit
from .cache_coverage import build_report as build_coverage_report, render_coverage, report_to_dict as coverage_to_dict
from .planner import build_pilot_manifest, build_prompt_ids, write_manifest
from .cache_utils import fmt_bytes, packed_cache_root, require_durable_cache_root
from .reports.table import HUMANML_COLUMNS, SOMA_TMR_COLUMNS
from .validate_pipeline import render_report, report_to_dict, run_full_rehearsal

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _add_cache_audit_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--root", default=None,
        help="Durable cache parent containing v2/ (default: $SEMOCO_EVAL_CACHE_ROOT or <workspace>/semoco-MotionVerse/eval_cache)",
    )
    sp.add_argument(
        "--runs-root", default=None,
        help="runs/eval root to scan for run_artifacts + shard logs (default: <repo>/runs/eval)",
    )
    sp.add_argument(
        "--run-artifacts-root", action="append", default=None, dest="extra_run_artifacts",
        help="Extra run_artifacts root(s) to include beyond --runs-root discovery (repeatable)",
    )
    sp.add_argument("--no-gpu", action="store_true", help="Skip nvidia-smi GPU query")
    sp.add_argument("--no-logs", action="store_true", help="Skip shard log parsing")
    sp.add_argument("--json-out", default=None, help="Write the full report as JSON to this path")
    sp.add_argument("--quiet", action="store_true", help="Suppress human-readable text output")


def _cmd_cache_audit(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_root) if args.runs_root else _REPO_ROOT / "runs" / "eval"
    extra_roots = [Path(p) for p in (args.extra_run_artifacts or [])]
    try:
        report = run_audit(
            cache_root=Path(args.root) if args.root else None,
            runs_root=runs_root,
            extra_run_artifact_roots=extra_roots,
            include_gpu=not args.no_gpu,
            include_logs=not args.no_logs,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(render_text(report))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(report), indent=2, default=str) + "\n")
        print(f"\nwrote {out_path}")
    return 0


def _cmd_plan_pilot(args: argparse.Namespace) -> int:
    if args.prompt_ids_jsonl:
        prompt_ids = []
        with Path(args.prompt_ids_jsonl).open("r") as f:
            for line in f:
                if line.strip():
                    prompt_ids.append(json.loads(line).get("prompt_id") or json.loads(line).get("clip_id"))
        prompt_ids = [str(x) for x in prompt_ids if x]
    else:
        prompt_ids = build_prompt_ids(count=args.num_prompts, prefix=args.prompt_prefix)
    run, units = build_pilot_manifest(
        track=args.track,
        model=args.model,
        split=args.split,
        dataset=args.dataset,
        prompt_ids=prompt_ids,
        chunk_size=args.chunk_size,
        run_id=args.run_id,
    )
    write_manifest(args.out_dir, run, units)
    print(f"wrote {args.out_dir}/run.json and work_units.jsonl ({len(units)} units)")
    return 0


def _cmd_cache_drop(args: argparse.Namespace) -> int:
    from .cache import cache_root as get_cache_root

    cache_root = Path(args.root) if args.root else get_cache_root()
    runs_root = Path(args.runs_root) if args.runs_root else _REPO_ROOT / "runs" / "eval"

    try:
        if args.legacy or args.v2:
            cache_root = require_durable_cache_root(cache_root, operation="cache drop")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    v2_root = Path(args.v2_root) if args.v2_root else (packed_cache_root(cache_root) if args.v2 else None)

    targets = []
    if args.legacy:
        targets += plan_legacy_drop(
            cache_root=cache_root, runs_root=runs_root,
            include_manifest_index=args.include_manifest_index,
        )
    if v2_root is not None:
        targets += plan_v2_scope_drop(v2_root=v2_root, scopes=args.scope or None)
    if not targets:
        print("nothing to drop (use --legacy and/or --v2/--v2-root)")
        return 0

    total_bytes = sum(t.size_bytes for t in targets)
    print(f"{'APPLY' if args.yes else 'DRY-RUN'} drop plan ({len(targets)} targets, {fmt_bytes(total_bytes)})")
    for t in targets:
        print(f"  {t.kind:22} {fmt_bytes(t.size_bytes):>8}  {t.path}")

    if not args.yes:
        print("\nNo changes made. Re-run with --yes to actually delete.")
        return 0

    removed = []
    if args.legacy:
        removed += apply_drop([t for t in targets if t.kind != "v2_scope"])
    if v2_root is not None:
        removed += apply_v2_scope_drop(v2_root=v2_root, scopes=args.scope or None)
    print(f"\nremoved {len(removed)} paths")
    return 0


def _cmd_tmr_list(args: argparse.Namespace) -> int:
    """Print a table of registered TMR models."""
    from .tmr import TMR_REGISTRY
    if not TMR_REGISTRY:
        print("(no TMR models registered)")
        return 0
    print(f"{'NAME':<24} {'TEXT_ENCODER':<14} {'MOTION_ENCODER':<24}")
    print("-" * 62)
    for entry in TMR_REGISTRY.values():
        me = f"shared from {entry.motion_encoder_from}" if entry.motion_encoder_from else "(own)"
        print(f"{entry.name:<24} {entry.text_encoder_kind:<14} {me:<24}")
    print(f"\n{len(TMR_REGISTRY)} model(s) registered.")
    return 0


def _add_run_args(p: argparse.ArgumentParser) -> None:
    """Register shared args, then delegate track options to track adapters."""
    from .tracks.shared_runner import _add_shared_args
    from .tracks.smpl_hml.runner import add_cli_args as add_hml_args
    from .tracks.soma_tmr.runner import add_cli_args as add_tmr_args

    p.add_argument("--track", required=True, choices=["smpl_hml", "soma_tmr"],
                   help="Evaluation track")
    _add_shared_args(p, adapter=None)
    p.add_argument("--models", nargs="*", default=[],
                   help="Baseline model IDs (default: all baselines for the track)")
    add_hml_args(p, require_data_root=False)
    # Preserve the unified CLI's historical opt-in default.  The direct TMR
    # adapter keeps its established default through its own registration.
    add_tmr_args(p, require_codes_root=False, kimodo_metrics_default=False)


def _cmd_run(args: argparse.Namespace) -> int:
    """Dispatch to the appropriate track runner with pre-parsed args."""
    if args.track == "soma_tmr":
        from .tracks.soma_tmr.runner import _run_track
    elif args.track == "smpl_hml":
        from .tracks.smpl_hml.runner import _run_track
    else:
        print(f"ERROR: unknown track: {args.track}", flush=True)
        return 1
    _run_track(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eval", description="Semoco eval pipeline CLI")
    sub = p.add_subparsers(dest="command", required=True)

    cache_p = sub.add_parser("cache", help="Eval cache inspection / maintenance")
    cache_sub = cache_p.add_subparsers(dest="cache_command", required=True)

    audit_p = cache_sub.add_parser(
        "audit",
        help="Read-only cache / run-artifact / log audit (Phase 0 measurement; never mutates anything)",
    )
    _add_cache_audit_args(audit_p)
    audit_p.set_defaults(func=_cmd_cache_audit)

    drop_p = cache_sub.add_parser(
        "drop",
        help="List or delete legacy per-file cache/run-artifact directories and/or v2 pack scopes "
             "(dry-run by default; pass --yes to actually delete)",
    )
    drop_p.add_argument("--root", default=None, help="Durable cache parent containing v2/ (for --legacy or --v2)")
    drop_p.add_argument("--runs-root", default=None, help="runs/eval root to scan for legacy run_artifacts")
    drop_p.add_argument("--legacy", action="store_true", help="Target legacy per-file cache/run-artifact dirs")
    drop_p.add_argument(
        "--include-manifest-index", action="store_true",
        help="Also drop the transitional _manifest/ probe index under --root",
    )
    drop_p.add_argument(
        "--v2", action="store_true",
        help="Target the default durable v2 store (--root or cache_root() / 'v2')",
    )
    drop_p.add_argument(
        "--v2-root", default=None,
        help="Explicit ShardedCacheStore v2 root (e.g. a run's run_artifacts/cache_v2); overrides --v2",
    )
    drop_p.add_argument(
        "--scope", action="append", default=None,
        help="Limit --v2/--v2-root drop to this scope (repeatable); default is every scope present",
    )
    drop_p.add_argument("--yes", action="store_true", help="Actually delete (default: dry-run)")
    drop_p.set_defaults(func=_cmd_cache_drop)

    coverage_p = cache_sub.add_parser(
        "coverage",
        help="Inspect cache contents organized by track and model (read-only, never mutates)",
    )
    _add_cache_coverage_args(coverage_p)
    coverage_p.set_defaults(func=_cmd_cache_coverage)

    aggregate_p = cache_sub.add_parser(
        "aggregate",
        help="CPU-only scoring from cached embeddings (no GPU, no model loading, no dataset creation)",
    )
    _add_cache_aggregate_args(aggregate_p)
    aggregate_p.set_defaults(func=_cmd_cache_aggregate)

    merge_p = cache_sub.add_parser(
        "merge",
        help="Consolidate run-local caches from different --out-dir directories "
             "into the canonical target (dry-run by default; pass --yes to execute)",
    )
    _add_cache_merge_args(merge_p)
    merge_p.set_defaults(func=_cmd_cache_merge)

    plan_p = sub.add_parser("plan", help="Eval planner/work manifest helpers")
    plan_sub = plan_p.add_subparsers(dest="plan_command", required=True)
    pilot_p = plan_sub.add_parser("pilot", help="Emit a pilot work manifest for one track/model")
    pilot_p.add_argument("--track", default="smpl_hml")
    pilot_p.add_argument("--model", default="semoco")
    pilot_p.add_argument("--split", default="test")
    pilot_p.add_argument("--dataset", default="HumanML3D")
    pilot_p.add_argument("--num-prompts", type=int, default=32)
    pilot_p.add_argument("--prompt-prefix", default="prompt")
    pilot_p.add_argument("--chunk-size", type=int, default=16)
    pilot_p.add_argument("--run-id", default=None)
    pilot_p.add_argument("--prompt-ids-jsonl", default=None)
    pilot_p.add_argument("--out-dir", required=True)
    pilot_p.set_defaults(func=_cmd_plan_pilot)

    validate_p = sub.add_parser("validate", help="Pipeline validation/rehearsal helpers")
    validate_sub = validate_p.add_subparsers(dest="validate_command", required=True)
    rehearsal_p = validate_sub.add_parser(
        "rehearsal",
        help="Run a small, dependency-free rehearsal of cache/worker-pool/resource-guard "
             "mechanics (cold/warm, interrupt/resume, drop/rebuild, RAM stress)",
    )
    rehearsal_p.add_argument("--root", required=True, help="Disposable scratch root for the rehearsal")
    rehearsal_p.add_argument("--json-out", default=None)
    rehearsal_p.set_defaults(func=_cmd_validate_rehearsal)

    run_p = sub.add_parser("run", help="Run an evaluation track (smpl_hml or soma_tmr)")
    _add_run_args(run_p)
    run_p.set_defaults(func=_cmd_run)

    tmr_p = sub.add_parser("tmr", help="TMR evaluator model registry")
    tmr_sub = tmr_p.add_subparsers(dest="tmr_command", required=True)
    list_p = tmr_sub.add_parser("list", help="List registered TMR models and their metadata")
    list_p.set_defaults(func=_cmd_tmr_list)

    return p


def _add_cache_coverage_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--root", default=None,
        help="Durable eval cache v2 root (default: $SEMOCO_EVAL_CACHE_ROOT/v2)",
    )
    sp.add_argument(
        "--runs-root", default=None,
        help="runs/eval root to auto-discover run-local caches (default: <repo>/runs/eval)",
    )
    sp.add_argument(
        "--run-root", action="append", default=None, dest="run_roots",
        help="Explicit run-local cache_v2 root(s) to inspect (repeatable)",
    )
    sp.add_argument(
        "--track", default=None, choices=["smpl_hml", "soma_tmr"],
        help="Show only one track",
    )
    sp.add_argument(
        "--model", default=None,
        help="Show only one model ID (e.g. semoco, kimodo)",
    )
    sp.add_argument("--json-out", default=None, help="Write structured JSON report to this path")
    sp.add_argument("--quiet", action="store_true", help="Suppress human-readable text output")


def _cmd_cache_coverage(args: argparse.Namespace) -> int:
    from .cache import v2_root as default_v2_root

    v2_root = Path(args.root) if args.root else default_v2_root()
    runs_root = Path(args.runs_root) if args.runs_root else _REPO_ROOT / "runs" / "eval"
    run_roots = [Path(p) for p in (args.run_roots or [])]

    report = build_coverage_report(
        v2_root=v2_root,
        run_artifact_roots=run_roots or None,
        runs_root=runs_root,
    )

    # Apply filters
    if args.track:
        keep = {args.track}
        report.tracks = {k: v for k, v in report.tracks.items() if k in keep}
        report.run_local_tracks = {k: v for k, v in report.run_local_tracks.items() if k in keep}

    if args.model:
        for ts in list(report.tracks.values()) + list(report.run_local_tracks.values()):
            for ss in list(ts.scopes.values()):
                if ss.by_model:
                    ss.by_model = {
                        mid: mb for mid, mb in ss.by_model.items()
                        if mid == args.model or args.model in mid
                    }

    if not args.quiet:
        print(render_coverage(report))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(coverage_to_dict(report), indent=2, default=str) + "\n")
        print(f"\nwrote {out_path}")
    return 0


def _cmd_validate_rehearsal(args: argparse.Namespace) -> int:
    report = run_full_rehearsal(args.root)
    print(render_report(report))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report_to_dict(report), indent=2, default=str) + "\n")
        print(f"\nwrote {out_path}")
    return 0 if report.all_passed else 1


# ---------------------------------------------------------------------------
# cache aggregate — CPU-only scoring from existing cache
# ---------------------------------------------------------------------------

def _add_cache_aggregate_args(sp: argparse.ArgumentParser) -> None:
    from .cache_aggregate import add_args
    add_args(sp)


def _cmd_cache_aggregate(args: argparse.Namespace) -> int:
    from .cache_aggregate import aggregate_from_cache
    return aggregate_from_cache(args)


def _add_cache_merge_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--runs-root", default=None,
                    help="runs/eval root to scan (default: <repo>/runs/eval)")
    sp.add_argument("--track", default=None, choices=["smpl_hml", "soma_tmr"],
                    help="Merge only this track")
    sp.add_argument("--dry-run", action="store_true", default=True,
                    help="Print merge plan without executing (default)")
    sp.add_argument("--yes", action="store_true", default=False,
                    help="Execute the merge (auto-backup first)")
    sp.add_argument("--backup-dir", default=None,
                    help="Explicit backup directory")
    sp.add_argument("--cleanup", action="store_true", default=False,
                    help="After successful merge+verification, remove merged source cache_v2/ dirs")
    sp.add_argument("--no-verify", action="store_true", default=False,
                    help="Skip post-merge verification (not recommended)")


def _cmd_cache_merge(args: argparse.Namespace) -> int:
    from .cache_merge import merge_caches
    return merge_caches(args)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
