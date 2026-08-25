"""Shared eval runner skeleton — parameterized by a :class:`TrackAdapter`.

Both eval tracks (``soma_tmr``, ``smpl_hml``) share the same orchestration:
arg parsing, model-list building, legacy/planner dispatch, sharding, and
aggregation dispatch.  The track-specific logic (evaluator loading, GT
encoding, gen encoding, per-clip embedding loading, motion-quality scoring,
text encoding) is injected via HOOK callables on the adapter.

Usage::

    from ..shared_runner import TrackAdapter, run_eval

    def main() -> None:
        adapter = TrackAdapter(
            track_name="soma_tmr",
            target_rep="soma77",
            ...
            encode_gt_shard=_encode_gt_shard,
            aggregate_fn=_aggregate,
        )
        run_eval(adapter)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..checkpoints import CheckpointSpec, load_registry, resolve
from ..model_plan import ModelPlan, build_model_plan
from ..resource_guard import ResourceBudget, ResourceGuard
from ..schema import MotionRep, TrackInput


# ---------------------------------------------------------------------------
# Tiny shared utilities (previously duplicated in both runners)
# ---------------------------------------------------------------------------

def _build_protocol_extras(args: argparse.Namespace) -> dict:
    """Collect track-specific metadata for :class:`EvalProtocol.extras`."""
    extras: dict = {}
    parquet_dir = getattr(args, "parquet_dir", None)
    if parquet_dir:
        extras["parquet_dir"] = str(Path(parquet_dir).resolve())
    return extras


def _shard(items: list, shard_index: int, num_shards: int) -> list:
    return [c for i, c in enumerate(items) if i % num_shards == shard_index]


def _build_model_plan(
    args: argparse.Namespace,
    semoco_specs: list[CheckpointSpec],
    default_models: Sequence[str],
) -> ModelPlan:
    """Resolve runner model selection via the shared immutable plan."""
    return build_model_plan(
        getattr(args, "models", []),
        semoco_specs,
        default_models=default_models,
        include_baselines=not getattr(args, "semoco_only", False),
    )


def _add_shared_args(p: argparse.ArgumentParser, adapter: TrackAdapter | None = None) -> None:
    """Register the CLI arguments shared by both tracks."""
    p.add_argument("--split", default=adapter.default_split if adapter else "test")
    p.add_argument("--mode", choices=("main", "smoke"), default="main")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fk-device", default="cuda")
    p.add_argument("--seeds", default="0")
    p.add_argument("--cfg-scale", type=float, default=None)
    p.add_argument("--max-tok", type=int, default=125)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--convert-only", action="store_true",
                   help="Skip generation; run conversion+embedding+aggregate from native cache")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip cached generations (default).")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="Force re-generation.")
    p.add_argument("--retrieval-protocol",
                   default=adapter.retrieval_protocol_default if adapter else "batch32",
                   choices=("full_gallery", "batch32", "batch256"))
    p.add_argument("--gen-batch-size", type=int, default=adapter.gen_batch_size_default if adapter else 32)
    p.add_argument("--out-dir", default=adapter.out_dir_default if adapter else "runs/eval")
    p.add_argument("--semoco-checkpoint", default=None,
                   help="Legacy: path to a single Semoco checkpoint (--semoco-model preferred).")
    p.add_argument("--semoco-model", default=None,
                   help="Checkpoint name, @group, comma-separated list, raw path, or 'all'.")
    p.add_argument("--semoco-only", action="store_true")
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--text-encoder", default=None,
                   help="Text encoder key (flan, llm2vec, etc.).")
    p.add_argument("--semoco-tokenizer", "--tokenizer-checkpoint", default=None,
                   dest="semoco_tokenizer",
                   help="Path to the frozen tokenizer checkpoint (overrides auto-discovery).")
    p.add_argument("--semoco-codes-root", default=None,
                   help="Legacy: override Semoco codes root.")
    # Planner / scheduler
    p.add_argument("--scheduler", default="legacy",
                   choices=("legacy", "planner"))
    p.add_argument("--planner-shards", type=int, default=1,
                   help="Number of concurrent GPU workers for planner mode.")
    p.add_argument("--lease-ttl-s", type=int, default=600,
                   help="Lease TTL in seconds (planner mode).")
    # Resource budget
    p.add_argument("--worker-ram-budget-gb", type=float, default=200.0)
    p.add_argument("--gpu-vram-headroom-gb", type=float, default=2.0)
    p.add_argument("--stage-bytes-gb", type=float, default=30.0)
    # Track-specific extras (adapter hook)
    if adapter is not None and adapter.add_args is not None:
        adapter.add_args(p)


# ---------------------------------------------------------------------------
# TrackAdapter
# ---------------------------------------------------------------------------

@dataclass
class TrackAdapter:
    """Per-track constants and HOOK callables for the shared runner.

    CONSTANT fields are simple values (strings, ints, floats).  HOOK fields
    are callables that encapsulate track-specific logic (GT loading, evaluator
    creation, scoring, etc.).  Each HOOK's signature is documented inline.
    """

    # ---- CONSTANTs ----
    track_name: str                              # "soma_tmr" | "smpl_hml"
    target_rep: MotionRep                        # "soma77" | "hml263"
    eval_fps: float                              # 30.0 | 20.0
    smoke_limit: int                             # per-track smoke-test prompt count
    model_list: tuple[str, ...]                  # SOMA_TMR_MODELS | HUMANML3D_MODELS
    report_columns: Sequence[str]                # SOMA_TMR_COLUMNS | HUMANML_COLUMNS
    report_filename: str                         # "semoco_soma_table.csv" | "humanml3d_table.csv"
    metric_space: str                            # "soma_tmr" | "humanml3d"
    evaluator_name: str                          # "tmr-soma-rp" | "text_mot_match"
    retrieval_protocol_default: str = "batch32"
    gen_batch_size_default: int = 32
    out_dir_default: str = "out/eval"
    default_split: str = "test"
    has_motion_quality: bool = False             # soma_tmr collects joints/fps; smpl_hml does not
    model_fps_map: dict[str, float] = field(default_factory=dict)
    evaluator_checkpoint: str | None = None      # overrides evaluator_name for protocol (HML needs file path)
    subset_source: str | None = None             # overrides track_name for protocol (HML uses "humanml3d")

    # ---- HOOKs ----
    # Evaluator checkpoint resolver: (args) -> str (defaults to evaluator_name if not set)
    evaluator_checkpoint_fn: Callable[..., str] | None = None

    # CLI extras
    add_args: Callable[[argparse.ArgumentParser], None] | None = None

    # Dataset factory: (args, limit, ...) -> dataset object
    create_dataset: Callable[..., Any] | None = None

    # Check readiness: (args) -> list of missing assets (empty = ready)
    check_readiness: Callable[..., list[str]] | None = None

    # Signatures
    dataset_sig_fn: Callable[..., str] | None = None

    # Evaluator factory: (args, ...) -> evaluator object
    load_evaluator: Callable[..., Any] | None = None

    # GT encoding: (ds, selected, evaluator, args, ds_sig) -> None
    encode_gt_shard: Callable[..., None] | None = None

    # Gen encoding: (model, selected, evaluator, args, seeds, eff_cfg, ds_sig, run_root, guard) -> None
    encode_gen_shard: Callable[..., None] | None = None

    # Text encoding (TMR only; None for HML): (evaluator, selected, args, ...) -> None
    encode_text_shard: Callable[..., None] | None = None

    # Semoco spec resolution from legacy --semoco-checkpoint: (args, ...) -> CheckpointSpec
    resolve_semoco_spec: Callable[..., CheckpointSpec] | None = None

    # Track input builder: (clips) -> list[TrackInput]
    build_track_inputs: Callable[..., list[TrackInput]] | None = None

    # Aggregation: (ds, args, ds_sig, out_root, protocol, protocol_id, seeds, *, run_root, guard, semoco_specs) -> list
    aggregate_fn: Callable[..., Any] | None = None

    # Cleanup (e.g., close_tmr): (evaluator) -> None
    cleanup: Callable[[Any], None] | None = None

    # Planner dispatch: (manifest, ds, track_inputs, all_model_ids, out_root, args,
    #   seeds, ds_sig, evaluator, graph, failures, run_root, guard, text_enc,
    #   semoco_specs) -> None
    run_planner: Callable[..., None] | None = None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_eval(adapter: TrackAdapter, args: argparse.Namespace | None = None) -> None:
    """Shared eval orchestration — arg parse, setup, GPU work, aggregate."""
    import json
    from pathlib import Path

    import torch

    from .. import cache as C
    from ..conversions import ConversionContext, build_default_graph
    from ..datasets.protocol import EvalProtocol, make_protocol_id
    from ..datasets.release_subset import write_subset
    from ..generation import ensure_native, ensure_target
    from ..models import default_cfg_for, load_model, weight_signature
    from ..planner import TrackPromptCost
    from ..planner_exec import build_or_read_track_manifest

    # -- Arg parsing ----------------------------------------------------------
    if args is None:
        p = argparse.ArgumentParser(
            description=f"{adapter.track_name} evaluation track runner"
        )
        p.add_argument("--models", nargs="*", default=list(adapter.model_list))
        _add_shared_args(p, adapter)
        args = p.parse_args()

    # -- Auto-configure environment --------------------------------------------
    if adapter.track_name == "soma_tmr":
        import os as _os
        _os.environ["no_proxy"] = "127.0.0.1,localhost"
        _os.environ["NO_PROXY"] = "127.0.0.1,localhost"

    # -- Device validation ----------------------------------------------------
    if not args.aggregate_only:
        from ..device_utils import validate_device as _validate_device
        _validate_device(
            args.device, num_shards=args.num_shards, scheduler=args.scheduler,
            stage=adapter.track_name,
        )

    # -- Limit ----------------------------------------------------------------
    limit = args.limit if args.limit is not None else (
        adapter.smoke_limit if args.mode == "smoke" else None
    )

    # -- Text encoder inference -----------------------------------------------
    text_enc = args.text_encoder
    if text_enc is None and args.semoco_checkpoint:
        ckpt = torch.load(args.semoco_checkpoint, map_location="cpu", weights_only=False)
        text_enc = (ckpt.get("data_meta") or {}).get("encode_key") or ckpt.get("text_encoder_key") or "flan"
    text_enc = text_enc or "flan"

    # -- Semoco spec resolution ------------------------------------------------
    semoco_specs: list[CheckpointSpec] = []
    if args.semoco_model:
        semoco_specs = resolve(load_registry(), args.semoco_model)
    elif args.semoco_checkpoint:
        assert adapter.resolve_semoco_spec is not None, "track must provide resolve_semoco_spec"
        semoco_specs = [adapter.resolve_semoco_spec(args, text_enc)]

    # -- Dataset --------------------------------------------------------------
    assert adapter.create_dataset is not None, "track must provide create_dataset"
    ds = adapter.create_dataset(args, limit=limit, text_enc=text_enc)

    # -- Seeds ----------------------------------------------------------------
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        seeds = [0]

    # -- Protocol ID + out_root -----------------------------------------------
    if not args.aggregate_only:
        assert adapter.check_readiness is not None, "track must provide check_readiness"
        missing_assets = adapter.check_readiness(args)
        if missing_assets:
            raise FileNotFoundError(
                f"Missing required assets for {adapter.track_name} track: {missing_assets}"
            )
    assert adapter.dataset_sig_fn is not None, "track must provide dataset_sig_fn"
    ds_sig = adapter.dataset_sig_fn(args)

    protocol_id = make_protocol_id(
        track=adapter.track_name,
        dataset=ds_sig,
        split=args.split,
        subset_ids=[c.clip_id for c in ds.clips],
        evaluator=adapter.evaluator_name,
        retrieval_protocol=args.retrieval_protocol,
        num_seeds=len(seeds),
        seed0=seeds[0] if seeds else 0,
    )
    out_root = Path(args.out_dir) / protocol_id
    out_root.mkdir(parents=True, exist_ok=True)
    write_subset(ds.clips, out_root / "prompts.jsonl")

    evaluator_checkpoint = (
        adapter.evaluator_checkpoint_fn(args) if adapter.evaluator_checkpoint_fn is not None
        else adapter.evaluator_checkpoint or adapter.evaluator_name
    )
    protocol = EvalProtocol(
        protocol_id=protocol_id,
        track=adapter.track_name,
        dataset=ds_sig,
        split=args.split,
        evaluator=adapter.evaluator_name,
        evaluator_checkpoint=evaluator_checkpoint,
        metric_space=adapter.metric_space,
        retrieval_protocol=args.retrieval_protocol,
        num_seeds=len(seeds),
        seed0=seeds[0] if seeds else 0,
        fps_eval=adapter.eval_fps,
        cfg_scale=args.cfg_scale,
        subset_source=adapter.subset_source or adapter.track_name,
        subset_ids=[c.clip_id for c in ds.clips],
        extras=_build_protocol_extras(args),
        notes=[],
    )
    protocol.save(out_root / "protocol.json")

    # -- Run root + resource guard --------------------------------------------
    run_root = C.run_artifacts_root(out_root)
    guard = ResourceGuard(ResourceBudget.from_args(args))

    # -- Model list -----------------------------------------------------------
    model_plan = _build_model_plan(
        args, semoco_specs, adapter.model_list,
    )
    baseline_models = list(model_plan.baselines)
    semoco_specs = list(model_plan.semoco_specs)
    all_model_ids = model_plan.model_ids

    # -- Aggregate-only early return ------------------------------------------
    if args.aggregate_only:
        assert adapter.aggregate_fn is not None, "track must provide aggregate_fn"
        adapter.aggregate_fn(
            ds, args, ds_sig, out_root, protocol, protocol_id, seeds,
            run_root=run_root, guard=guard, semoco_specs=semoco_specs,
        )
        return

    # -- GPU work -------------------------------------------------------------
    selected = _shard(ds.clips, args.shard_index, args.num_shards)

    # GT encode
    assert adapter.load_evaluator is not None, "track must provide load_evaluator"
    assert adapter.encode_gt_shard is not None, "track must provide encode_gt_shard"
    evaluator = adapter.load_evaluator(args, ds_sig=ds_sig)
    adapter.encode_gt_shard(ds, selected, evaluator, args, ds_sig)

    # Track inputs
    assert adapter.build_track_inputs is not None, "track must provide build_track_inputs"
    track_inputs = adapter.build_track_inputs(ds.clips)

    # Conversion graph
    graph = build_default_graph()

    # Planner manifest (pre-built for planner mode)
    failures: Path | None = out_root / "failures.jsonl" if args.shard_index == 0 else None
    if failures is not None and failures.exists():
        failures.unlink()

    if args.scheduler == "planner":
        prompts = [TrackPromptCost(t.prompt_id, t.length.seconds) for t in track_inputs]
        manifest = build_or_read_track_manifest(
            out_root=out_root,
            track=adapter.track_name,
            models=all_model_ids,
            split=args.split,
            dataset=ds_sig,
            prompts=prompts,
            num_shards=args.planner_shards,
            run_id=protocol_id,
        )
        # Text encode (planner mode — pre-encode before worker pool)
        if adapter.encode_text_shard is not None:
            adapter.encode_text_shard(evaluator, selected, args, ds_sig=ds_sig, run_root=run_root)
        # Delegate to track-specific planner implementation
        if adapter.run_planner is not None:
            adapter.run_planner(
                manifest=manifest, ds=ds, track_inputs=track_inputs,
                all_model_ids=all_model_ids, out_root=out_root, args=args,
                seeds=seeds, ds_sig=ds_sig, evaluator=evaluator, graph=graph,
                failures=failures, run_root=run_root, guard=guard,
                text_enc=text_enc, semoco_specs=semoco_specs,
            )
        return  # planner never auto-aggregates (multiple concurrent workers)

    # --- Legacy (single-process) loop ---------------------------------------
    for model_name in baseline_models:
        mid = _run_one_model(
            adapter, model_name, {}, model_name, track_inputs, selected,
            graph, args, seeds, ds_sig, evaluator, failures, run_root, guard,
            is_semoco=False,
        )
    for spec in semoco_specs:
        sig_kwargs = {"checkpoint": str(spec.model_ckpt),
                      "tokenizer_checkpoint": str(spec.tokenizer_ckpt),
                      "max_tok": spec.max_tok,
                      "codes_root": str(spec.codes_root),
                      "text_encoder": spec.text_encoder,
                      "fk_device": args.fk_device}
        _run_one_model(
            adapter, "semoco", sig_kwargs, spec.name,
            track_inputs, selected, graph, args, seeds, ds_sig,
            evaluator, failures, run_root, guard, is_semoco=True,
        )

    # Text encode (deferred — TMR only)
    if adapter.encode_text_shard is not None:
        adapter.encode_text_shard(evaluator, selected, args, ds_sig=ds_sig, run_root=run_root)

    # Auto-aggregate (single-shard legacy only)
    if args.num_shards == 1:
        assert adapter.aggregate_fn is not None
        adapter.aggregate_fn(
            ds, args, ds_sig, out_root, protocol, protocol_id, seeds,
            run_root=run_root, guard=guard, semoco_specs=semoco_specs,
        )

    # Cleanup
    if adapter.cleanup is not None:
        adapter.cleanup(evaluator)


def _run_one_model(
    adapter: TrackAdapter,
    model_name: str,
    sig_kwargs: dict,
    display_name: str,
    track_inputs: list,
    selected: list,
    graph,
    args,
    seeds: list[int],
    ds_sig: str,
    evaluator,
    failures,
    run_root,
    guard: ResourceGuard,
    *,
    is_semoco: bool = False,
) -> str:
    """Generate + convert + encode one model (baseline or Semoco)."""
    from ..models import default_cfg_for, load_model, weight_signature
    from ..generation import ensure_native, ensure_target
    from ..conversions import ConversionContext

    mid = model_name
    sig = weight_signature(model_name, **sig_kwargs)
    eff_cfg = default_cfg_for(model_name, args.cfg_scale)

    load_kwargs: dict = {"device": args.device}
    guard.check_gpu_headroom(f"{model_name}: before load_model", args.device)

    if is_semoco:
        from ..models.semoco import SemocoModel
        # Merge semoco-specific kwargs
        load_kwargs["checkpoint"] = sig_kwargs.get("checkpoint")
        load_kwargs["tokenizer_checkpoint"] = sig_kwargs.get("tokenizer_checkpoint")
        load_kwargs["max_tok"] = sig_kwargs.get("max_tok", 125)
        load_kwargs["text_encoder"] = sig_kwargs.get("text_encoder") or args.text_encoder or "flan"
        load_kwargs["codes_root"] = sig_kwargs.get("codes_root")
        load_kwargs["fk_device"] = sig_kwargs.get("fk_device", args.fk_device)
        model = SemocoModel(**load_kwargs)
    else:
        model = load_model(model_name, **load_kwargs)

    guard.validate_model_device(model, args.device, model_name=getattr(model, 'schema', None) and model.schema.model_id)
    try:
        guard.check_rss(f"{display_name}: after load_model")
        ctx = ConversionContext(device=args.device, fk_device=args.fk_device)
        if is_semoco:
            ctx.semoco_tokenizer = model.tokenizer
            ctx.semoco_anchor = model.anchor
            if model.dataset is not None:
                prompts_with_rec = [(t.prompt_id, t.rec_id or "", t.caption) for t in track_inputs]
                ctx.prompt_id_anchors = model.build_prompt_anchor_map(prompts_with_rec)
        if hasattr(model, 'estimate_generation_memory'):
            mem_info = model.estimate_generation_memory(
                args.gen_batch_size,
                getattr(model, 'max_tok', args.max_tok),
                cfg_scale=eff_cfg,
            )
            guard.check_generation_headroom(
                f"{display_name}: before generation",
                args.device,
                estimated_kv_cache_bytes=mem_info.get("kv_cache_bytes", 0),
                model_params_bytes=mem_info.get("param_bytes", 0),
            )
        ensure_native(
            model, track_inputs, seeds=seeds, cfg_scale=eff_cfg,
            dataset_sig=ds_sig, shard_index=args.shard_index,
            num_shards=args.num_shards, skip_existing=args.skip_existing,
            batch_size=args.gen_batch_size, failures_path=failures,
        )
        ensure_target(
            model, track_inputs, target_rep=adapter.target_rep,
            graph=graph, ctx=ctx, seeds=seeds, cfg_scale=eff_cfg,
            dataset_sig=ds_sig, shard_index=args.shard_index,
            num_shards=args.num_shards, skip_existing=args.skip_existing,
            failures_path=failures, run_root=run_root,
        )
        adapter.encode_gen_shard(
            model, selected, evaluator, args, seeds, eff_cfg,
            ds_sig=ds_sig, run_root=run_root, guard=guard,
        )
    finally:
        model.close()
        guard.cleanup()
        guard.check_empty_cache_on_pressure(args.device)
    return mid
