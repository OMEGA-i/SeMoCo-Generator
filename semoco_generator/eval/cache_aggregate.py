"""``eval cache aggregate`` — CPU-only scoring from cached embeddings.

Extracted from ``cli.py`` so the orchestration logic lives behind a clean seam.
The compatibility utility :func:`build_model_entries` delegates to the shared
model plan used by track runners.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import cache as C
from .datasets.protocol import EvalProtocol
from .datasets.release_subset import load_subset
from .checkpoints import CheckpointSpec, load_registry, resolve
from .model_plan import build_track_model_plan
from .reports.table import HUMANML_COLUMNS, SOMA_TMR_COLUMNS
from .score_schema import EvalScore
from .sharded_cache_store import ShardedCacheStore


# ---------------------------------------------------------------------------
# Argument registration
# ---------------------------------------------------------------------------

def add_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--out-dir", required=True,
                    help="Path to a protocol output directory containing protocol.json, "
                         "prompts.jsonl, and run_artifacts/cache_v2/")
    sp.add_argument("--models", nargs="*", default=None,
                    help="Baseline model IDs to score (default: all baseline models for the track)")
    sp.add_argument("--semoco-model", default=None,
                    help="Semoco checkpoint SPEC (name, @group, comma-separated, 'all', or raw path)")
    sp.add_argument("--output", default=None,
                    help="CSV output path (default: <out-dir>/reports/<table>.csv)")
    sp.add_argument("--parquet-dir", default=None,
                    help="UMR release dir for per-subset provenance labels")
    sp.add_argument("--subsets", default=None,
                    help="Comma-separated subset names to filter. Requires --parquet-dir.")
    sp.add_argument("--limit-per-subset", type=int, default=None,
                    help="Sample N clips per subset (deterministic seed=0)")
    sp.add_argument("--repeat", type=int, default=1,
                    help="Repeat aggregation N times with different random seeds")
    sp.add_argument("--retrieval-protocol", default=None,
                    choices=("full_gallery", "batch32", "batch256"),
                    help="Override retrieval protocol from protocol.json")
    sp.add_argument("--lazy-generate", action="store_true",
                    help="Load models and generate missing natives/converts/gen_emb on the fly "
                         "(requires --device). Model stays loaded across --repeat iterations.")
    sp.add_argument("--device", default="cuda:0",
                    help="GPU device for --lazy-generate (default: cuda:0)")


# ---------------------------------------------------------------------------
# Shared model-list builder
# ---------------------------------------------------------------------------

def build_model_entries(
    baseline_ids: list[str] | None,
    semoco_specs: list[CheckpointSpec],
    *,
    track: str,
) -> list[tuple[str, dict, str]]:
    """Build legacy aggregation tuples via :class:`model_plan.ModelPlan`.

    ``[]`` remains distinct from omission here: cache aggregation may use an
    explicit empty list to score only resolved Semoco checkpoints.
    """
    return build_track_model_plan(
        track,
        baseline_ids,
        semoco_specs,
        default_when_unspecified=False,
    ).scoring_tuples()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_METRIC_FIELDS = ["fid", "r1", "r2", "r3", "r5", "r10", "medr",
                  "matching", "t2m_sim", "diversity", "multimodality",
                  "foot_skate", "jerk", "length_mean_s", "length_std_s"]


def aggregate_from_cache(args: argparse.Namespace) -> int:
    """CPU-only: score all cached gen_emb entries for a completed eval run."""
    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        print(f"ERROR: --out-dir does not exist: {out_dir}", flush=True)
        return 1

    protocol_path = out_dir / "protocol.json"
    if not protocol_path.is_file():
        print(f"ERROR: protocol.json not found at {protocol_path}", flush=True)
        return 1

    protocol = EvalProtocol.load(protocol_path)
    track = protocol.track
    protocol_id = protocol.protocol_id
    ds_sig = protocol.dataset
    split = protocol.split
    evaluator_name = protocol.evaluator
    evaluator_checkpoint = protocol.evaluator_checkpoint
    retrieval_protocol = args.retrieval_protocol or protocol.retrieval_protocol
    seeds = list(range(protocol.num_seeds)) if protocol.num_seeds else [protocol.seed0] or [0]
    cfg_scale = protocol.cfg_scale
    print(f"[aggregate] loaded protocol: track={track} id={protocol_id}", flush=True)

    prompts_path = out_dir / "prompts.jsonl"
    if not prompts_path.is_file():
        print(f"ERROR: prompts.jsonl not found at {prompts_path}", flush=True)
        return 1
    clips = load_subset(prompts_path)
    print(f"[aggregate] loaded {len(clips)} clips from prompts.jsonl", flush=True)

    if args.parquet_dir:
        from .datasets.release_subset import load_subset_labels_from_release
        parquet_dir = Path(args.parquet_dir)
        if not parquet_dir.is_dir():
            print(f"ERROR: --parquet-dir does not exist: {parquet_dir}", flush=True)
            return 1
        subset_map = load_subset_labels_from_release(str(parquet_dir), split)
        if subset_map:
            for c in clips:
                if c.rec_id in subset_map:
                    c.subset = subset_map[c.rec_id]
            labels = sorted(set(v for v in subset_map.values() if v))
            print(f"[aggregate] loaded {len(subset_map)} subset labels, {len(labels)} subsets: {labels}", flush=True)
        else:
            print(f"[aggregate] WARNING: no subset labels found in {parquet_dir}", flush=True)

    if args.subsets:
        wanted = set(s.strip() for s in args.subsets.split(","))
        before = len(clips)
        clips = [c for c in clips if (c.subset or "") in wanted]
        print(f"[aggregate] filtered to subsets {sorted(wanted)}: {len(clips)}/{before} clips", flush=True)

    by_subset_all: dict[str, list] | None = None
    if args.limit_per_subset:
        by_subset_all = {}
        for c in clips:
            by_subset_all.setdefault(c.subset or "unlabeled", []).append(c)
        for sub_name, sub_clips in sorted(by_subset_all.items()):
            print(f"[aggregate]   {sub_name}: {len(sub_clips)} total clips available", flush=True)

    semoco_specs = resolve(load_registry(), args.semoco_model) if args.semoco_model else []
    if semoco_specs:
        print(f"[aggregate] {len(semoco_specs)} Semoco checkpoint(s) resolved", flush=True)
    all_entries = build_model_entries(args.models, semoco_specs, track=track)
    print(f"[aggregate] {len(all_entries)} model(s) to score", flush=True)

    run_root = out_dir / "run_artifacts"
    run_store = ShardedCacheStore(run_root / "cache_v2", num_buckets=16) if (run_root / "cache_v2").is_dir() else None
    discovered_eval_sig, discovered_ds_sig = None, ds_sig
    if run_store is not None:
        sample_key = run_store.sample_key("gen_emb")
        if sample_key:
            parts = sample_key.split(":")
            if len(parts) >= 4:
                discovered_ds_sig = parts[2]
                discovered_eval_sig = parts[3]
                print(f"[aggregate] discovered ds_sig={discovered_ds_sig} eval_sig={discovered_eval_sig} from cache", flush=True)

    from .aggregate import aggregate_hml, aggregate_soma_tmr

    # ---- Lazy-generate: pre-load models once for all repeats -----------------
    lazy_models: dict[str, object] = {}
    lazy_tmr = None
    lazy_device = None
    if args.lazy_generate:
        lazy_device = args.device
        print(f"[aggregate] lazy-generate enabled, device={lazy_device}", flush=True)

        # Load TMR evaluator (for gen_emb)
        if track == "soma_tmr":
            from .tmr import load_tmr
            lazy_tmr = load_tmr(device=lazy_device, modelname=evaluator_name,
                                rprecision=not getattr(args, 'no_rprecision', False))
            print(f"[aggregate] loaded TMR evaluator: {evaluator_name}", flush=True)

        from .models.registry import normalize_model_name
        for model_name, sig_kwargs, display_name in all_entries:
            mid = normalize_model_name(model_name)
            print(f"[aggregate]   loading {display_name}...", flush=True)

            if mid == "semoco":
                from .models.semoco import SemocoModel
                model = SemocoModel(
                    checkpoint=sig_kwargs.get("checkpoint"),
                    tokenizer_checkpoint=sig_kwargs.get("tokenizer_checkpoint"),
                    codes_root=sig_kwargs.get("codes_root"),
                    text_encoder=sig_kwargs.get("text_encoder", "flan"),
                    device=lazy_device,
                    max_tok=sig_kwargs.get("max_tok", 125),
                )
            else:
                from .models import load_model
                model = load_model(model_name, device=lazy_device)
            lazy_models[display_name] = model
            print(f"[aggregate]   {display_name}: loaded ✓", flush=True)

    all_run_scores: dict[int, list] = {}
    for repeat_seed in range(args.repeat):
        run_clips = clips
        if by_subset_all is not None:
            rng = np.random.default_rng(repeat_seed)
            run_clips = []
            for sub_name, sub_clips in sorted(by_subset_all.items()):
                n = min(args.limit_per_subset, len(sub_clips))
                idx = np.sort(rng.choice(len(sub_clips), size=n, replace=False))
                run_clips.extend([sub_clips[int(i)] for i in idx])
            if repeat_seed == 0:
                print(f"[aggregate] sampled {len(run_clips)} total clips per run", flush=True)
        if args.repeat > 1:
            print(f"[aggregate] repeat {repeat_seed + 1}/{args.repeat} (seed={repeat_seed})...", flush=True)
        write = (args.repeat == 1)
        if track == "smpl_hml":
            if discovered_eval_sig is None:
                print("ERROR: could not discover HML gt_sig; run-local gen_emb empty?", flush=True)
                return 1
            run_scores = aggregate_hml(clips=run_clips, gt_sig=discovered_eval_sig,
                                       ds_sig=discovered_ds_sig, out_root=out_dir,
                                       protocol_id=protocol_id, models=all_entries,
                                       evaluator_checkpoint=str(evaluator_checkpoint),
                                       retrieval_protocol=retrieval_protocol,
                                       cfg_scale=cfg_scale, seeds=seeds, split=split,
                                       dataset_name=discovered_ds_sig, run_root=run_root,
                                       write_scores=write,
                                       lazy_models=lazy_models if args.lazy_generate else None,
                                       lazy_device=lazy_device)
        elif track == "soma_tmr":
            if discovered_eval_sig is None:
                print("ERROR: could not discover TMR emb_sig; run-local gen_emb empty?", flush=True)
                return 1
            run_scores = aggregate_soma_tmr(clips=run_clips, emb_sig=discovered_eval_sig,
                                            ds_sig=discovered_ds_sig, out_root=out_dir,
                                            protocol_id=protocol_id, models=all_entries,
                                            tmr_model=evaluator_name,
                                            retrieval_protocol=retrieval_protocol,
                                            cfg_scale=cfg_scale, seeds=seeds, split=split,
                                            dataset_name=discovered_ds_sig, run_root=run_root,
                                            write_scores=write,
                                            lazy_models=lazy_models if args.lazy_generate else None,
                                            lazy_tmr=lazy_tmr,
                                            lazy_device=lazy_device)
        else:
            print(f"ERROR: unknown track '{track}' in protocol.json", flush=True)
            return 1
        all_run_scores[repeat_seed] = run_scores

    # ---- Clean up lazy models ----
    for model in lazy_models.values():
        try:
            model.close()
        except Exception:
            pass
    if lazy_tmr is not None:
        from .tmr import close_tmr
        close_tmr(lazy_tmr)

    if args.repeat > 1:
        cols = SOMA_TMR_COLUMNS if track == "soma_tmr" else HUMANML_COLUMNS
        summary_scores = _compute_repeat_summary(all_run_scores, args.repeat, cols)
        (out_dir / "scores").mkdir(parents=True, exist_ok=True)
        for seed, run_scores in all_run_scores.items():
            for s in run_scores:
                key = s.model.replace("/", "_")
                (out_dir / "scores" / f"{key}_seed{seed}.json").write_text(
                    json.dumps(s.to_dict(), indent=2) + "\n")
        (out_dir / "reports").mkdir(parents=True, exist_ok=True)
        report_name = "humanml3d_table.csv" if track == "smpl_hml" else "semoco_soma_table.csv"
        _export_repeat_summary_csv(summary_scores,
                                   Path(args.output) if args.output else out_dir / "reports" / report_name,
                                   cols)
        print(f"[aggregate] wrote {args.output or out_dir / 'reports' / report_name} "
              f"(mean±std across {args.repeat} runs)", flush=True)
    else:
        print(f"[aggregate] done — {len(all_run_scores.get(0, []))} score(s) "
              f"written to {out_dir / 'reports'}", flush=True)
    return 0


def _compute_repeat_summary(all_run_scores, n_repeats, columns):
    groups: dict[tuple, list] = defaultdict(list)
    for _seed, scores in all_run_scores.items():
        for s in scores:
            groups[(s.model, s.subset)].append(s)
    summary: list[dict] = []
    for (model, subset), score_list in sorted(groups.items()):
        row = {"model": model, "subset": subset or "",
               "num_prompts": score_list[0].num_prompts,
               "num_success": int(_safe_mean([s.num_success for s in score_list])),
               "num_failed": int(_safe_mean([s.num_failed for s in score_list])),
               "_n_repeats": len(score_list)}
        for field in _METRIC_FIELDS:
            values = [getattr(s, field, None) for s in score_list]
            values = [v for v in values if v is not None and v == v]
            row[field] = _safe_mean(values) if values else None
            row[f"{field}_std"] = _safe_std(values) if values else None
        first = score_list[0]
        for f in ["track", "dataset", "split", "evaluator", "evaluator_checkpoint",
                   "metric_space", "retrieval_protocol", "target_rep", "protocol_id",
                   "num_seeds", "failure_rate", "sec_per_clip"]:
            if f not in row:
                row[f] = getattr(first, f, None) if hasattr(first, f) else first.to_dict().get(f, "")
        summary.append(row)
    return summary


def _export_repeat_summary_csv(summary_rows, path, columns):
    import csv as _csv
    expanded_cols: list[str] = []
    for c in columns:
        expanded_cols.append(c)
        if c in _METRIC_FIELDS:
            expanded_cols.append(f"{c}_std")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=expanded_cols, extrasaction="ignore")
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    path.with_suffix(".json").write_text(json.dumps(summary_rows, indent=2) + "\n")


def _safe_mean(values):
    return float(np.mean([float(v) for v in values]))


def _safe_std(values):
    if len(values) < 2:
        return 0.0
    return float(np.std([float(v) for v in values], ddof=1))


__all__ = ["add_args", "aggregate_from_cache", "build_model_entries"]
