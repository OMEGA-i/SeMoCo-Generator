"""End-to-end SOMA/TMR evaluation track runner (sharded, resumable).

Durable shared cache (``SEMOCO_EVAL_CACHE_ROOT``):
  - GT joints / motion / text embeddings
  - model native generations (reuse across tracks)

Run-local under ``<out_root>/run_artifacts``:
  - converted soma77 targets
  - generated-motion embeddings

Shard phase (GPU): GT encode, native gen, convert, gen encode.
Aggregate phase (CPU, ``--aggregate-only``): score from stable GT + run-local gen emb.

Typical sharded full run:
    for shard in 0..N-1:  python -m semoco_generator.eval.cli run --track soma_tmr ... --num-shards N --shard-index shard
    python -m semoco_generator.eval.cli run --track soma_tmr ... --aggregate-only

``--scheduler planner`` replaces the static ``shard_index % num_shards`` split
for native-gen/convert/gen-embed with :mod:`planner_exec` (EvalPlanner LPT
bins + GPUWorkerPool lease/commit). Multiple GPU-pinned processes can point
at the same ``--out-dir`` concurrently and will dynamically split ready work
instead of a fixed partition; run ``--aggregate-only`` once after all of them
finish (never auto-aggregated inline, since no single process can know when
every concurrent worker is done).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ....local_uri import resolve_local_uri
from ....paths import default_checkpoint
from ....tokenizer_bridge import FrozenMotionTokenizer
from ... import cache as C
from ...checkpoints import CheckpointSpec
from ...conversions import ConversionContext
from ...generation import ensure_native, ensure_target
from ...metrics import resample_fps
from ...models import SOMA_TMR_MODELS, default_cfg_for, load_model
from ...planner_exec import LoadedModel
from ...reports import SOMA_TMR_COLUMNS
from ...resource_guard import ResourceGuard, array_nbytes
from ...schema import LengthSpec, TrackInput
from ...tmr import TMR_REGISTRY, close_tmr, load_tmr, load_tmr_multi
from .check_readiness import check_readiness
from .dataset import SomaTMRDataset
from .protocol import MODEL_FPS, RETRIEVAL_PROTOCOL, TMR_MODEL

TARGET_REP = "soma77"
EVAL_FPS = 30.0
TRACK = "soma_tmr"


def add_cli_args(
    p: argparse.ArgumentParser,
    *,
    require_codes_root: bool = False,
    kimodo_metrics_default: bool = True,
) -> None:
    """Register SOMA/TMR-specific options on either supported CLI entry point."""
    p.add_argument("--codes-root", required=require_codes_root, help="Code store root (local:// or path)")
    p.add_argument("--tmr-model", default=TMR_MODEL,
                   help="Comma-separated TMR model names")
    p.add_argument("--no-rprecision", action="store_true",
                   help="Skip R-precision (FID-only, no text encoder)")
    p.add_argument("--text-batch-size", type=int, default=256, help="Batch size for text encoding")
    p.add_argument("--parquet-dir", default=None,
                   help="derived_umr_* release dir for per-subset provenance labels")
    p.add_argument("--kimodo-metrics", action="store_true", default=kimodo_metrics_default,
                   help="Compute KiMoDo-specific duplicate-aware metrics")
    p.add_argument("--no-kimodo-metrics", dest="kimodo_metrics", action="store_false")


def _parse_tmr_models(args) -> list[str]:
    """Parse comma-separated ``--tmr-model`` into a list of model names."""
    raw = getattr(args, "tmr_model", None) or TMR_MODEL
    return [m.strip() for m in str(raw).split(",") if m.strip()]


from ..shared_runner import TrackAdapter


def _tmr_encode_batch(
    tmr,
    joints_list: list[np.ndarray],
    fps_list: list[float],
    device: str,
    batch_size: int = 64,
) -> list[np.ndarray]:
    """TMR-encode multiple clips in batched forward passes.

    Clips are resampled to ``EVAL_FPS``, sorted by length to limit padding, then
    padded to a common length within each micro-batch and encoded together.
    """
    if not joints_list:
        return []

    # Resample all clips first
    resampled = [
        resample_fps(np.asarray(j, np.float32), float(fps), EVAL_FPS)
        for j, fps in zip(joints_list, fps_list)
    ]

    results: list[np.ndarray | None] = [None] * len(resampled)

    # Sort by length (descending) to minimise padding within each micro-batch
    indexed = sorted(enumerate(resampled), key=lambda x: x[1].shape[0], reverse=True)

    for start in range(0, len(indexed), batch_size):
        group = indexed[start : start + batch_size]
        idxs = [i for i, _ in group]
        arrays = [a for _, a in group]
        max_t = max(a.shape[0] for a in arrays)

        # Pad to max_t
        padded = []
        lengths = []
        for a in arrays:
            lengths.append(a.shape[0])
            if a.shape[0] < max_t:
                pad = np.zeros((max_t - a.shape[0], a.shape[1], a.shape[2]), dtype=np.float32)
                padded.append(np.concatenate([a, pad], axis=0))
            else:
                padded.append(a)

        batch = torch.from_numpy(np.stack(padded)).float().to(device)  # [B, max_t, 77, 3]
        lengths_t = torch.as_tensor(lengths, device=device)
        with torch.inference_mode():
            embs = tmr.encode_motion(batch, lengths=lengths_t, unit_vector=True)  # [B, D] or [B, 1, D]
        embs = embs.squeeze(1) if embs.dim() == 3 else embs  # [B, D]

        for idx, emb in zip(idxs, embs):
            results[idx] = np.asarray(emb.float().cpu().numpy(), dtype=np.float32)

    # Any None entries indicate an unexpected shape — shouldn't happen
    if any(r is None for r in results):
        raise RuntimeError("_tmr_encode_batch produced None entries — shape mismatch?")

    return results  # type: ignore[return-value]


def _build_track_inputs(clips) -> list[TrackInput]:
    return [
        TrackInput(prompt_id=clip.clip_id, rec_id=clip.rec_id, caption=clip.caption,
                   length=LengthSpec(seconds=clip.duration_s))
        for clip in clips
    ]


def _semoco_spec_from_legacy_args(args, codes_root, tok_ckpt, text_enc) -> CheckpointSpec:
    """Build a :class:`CheckpointSpec` from legacy ``--semoco-checkpoint`` etc. args."""
    from ...checkpoints import discover_from_dir

    ckpt = Path(args.semoco_checkpoint)
    run_dir = ckpt.parent.parent
    discovered = discover_from_dir(run_dir)
    if discovered is not None:
        return CheckpointSpec(
            name=discovered.name,
            model_ckpt=discovered.model_ckpt,
            tokenizer_ckpt=tok_ckpt,
            codes_root=codes_root,
            text_encoder=text_enc,
            max_tok=args.max_tok,
        )
    return CheckpointSpec(
        name=ckpt.parent.parent.name,
        model_ckpt=ckpt,
        tokenizer_ckpt=tok_ckpt,
        codes_root=codes_root,
        text_encoder=text_enc,
        max_tok=args.max_tok,
    )


# ---- TrackAdapter HOOK wrappers -----------------------------------------------

def _tmr_check_readiness(args):
    codes_root = resolve_local_uri(args.codes_root)
    return check_readiness(codes_root, args.split, text_encoder=args.text_encoder,
                           checkpoint=args.semoco_checkpoint,
                           tokenizer=args.semoco_tokenizer)

def _tmr_dataset_sig(args):
    codes_root = resolve_local_uri(args.codes_root)
    return C.dataset_sig(codes_root, args.split)

def _tmr_resolve_semoco_spec(args, text_enc):
    codes_root = resolve_local_uri(args.codes_root)
    tok_ckpt = Path(args.semoco_tokenizer) if args.semoco_tokenizer else default_checkpoint()
    if not tok_ckpt.is_file():
        meta_path = codes_root / f"{args.split}.meta.json"
        if meta_path.is_file():
            cand = Path(json.loads(meta_path.read_text()).get("tokenizer_checkpoint") or "")
            if cand.is_file():
                tok_ckpt = cand
    return _semoco_spec_from_legacy_args(args, codes_root, tok_ckpt, text_enc)

def _tmr_create_dataset(args, *, limit, text_enc):
    codes_root = resolve_local_uri(args.codes_root)
    subset_map = None
    if args.parquet_dir:
        from ...datasets.release_subset import load_subset_labels_from_release
        parquet_dir = resolve_local_uri(args.parquet_dir)
        print(f"[soma_tmr] loading subset labels from {parquet_dir}/{args.split}", flush=True)
        subset_map = load_subset_labels_from_release(parquet_dir, args.split)
        labels = sorted(set(subset_map.values()))
        print(f"[soma_tmr] {len(subset_map)} clips → {len(labels)} subsets: {labels}", flush=True)
    return SomaTMRDataset(codes_root, args.split, text_encoder=text_enc, limit=limit, seed=0,
                           max_tok=args.max_tok, subset_map=subset_map)

def _tmr_load_evaluator(args, *, ds_sig):
    codes_root = resolve_local_uri(args.codes_root)
    tok_ckpt = Path(args.semoco_tokenizer) if args.semoco_tokenizer else default_checkpoint()
    if not tok_ckpt.is_file():
        meta_path = codes_root / f"{args.split}.meta.json"
        if meta_path.is_file():
            cand = Path(json.loads(meta_path.read_text()).get("tokenizer_checkpoint") or "")
            if cand.is_file():
                tok_ckpt = cand
    tok = FrozenMotionTokenizer.load(tok_ckpt, device=args.device)
    gt_src_fps = float(tok.spec.source_fps)

    tmr_models = _parse_tmr_models(args)
    tmr_dict = load_tmr_multi(
        device=args.device, modelnames=tmr_models,
        rprecision=not args.no_rprecision,
    )
    return (tmr_dict, tok, gt_src_fps)

def _tmr_encode_gt_shard_hook(ds, selected, evaluator, args, ds_sig):
    tmr_dict, _tok, _gt_src_fps = evaluator
    for tmr_name, tmr in tmr_dict.items():
        emb_sig = C.tmr_gt_sig(tmr_name, store=ds_sig)
        _encode_gt_shard(ds, selected, tmr, args, emb_sig, ds_sig)

def _tmr_encode_gen_shard_hook(model, selected, evaluator, args, seeds, eff_cfg, *, ds_sig, run_root, guard):
    tmr_dict, _tok, _fps = evaluator
    for tmr_name, tmr in tmr_dict.items():
        emb_sig = C.tmr_gt_sig(tmr_name, store=ds_sig)
        _encode_gen_shard(model, selected, tmr, args, seeds, eff_cfg,
                          emb_sig=emb_sig, ds_sig=ds_sig, run_root=run_root, guard=guard)

def _tmr_encode_text_shard_hook(evaluator, selected, args, *, ds_sig, run_root):
    if args.no_rprecision:
        return
    tmr_dict, _tok, _fps = evaluator
    for tmr_name, tmr in tmr_dict.items():
        _encode_text_shard(tmr, selected, args, tmr_model=tmr_name)


def _run_track(args: argparse.Namespace | None = None) -> None:
    """Build adapter and dispatch to shared runner.

    When *args* is provided (from the unified ``eval run`` CLI), internal
    arg-parsing is skipped.  When *args* is ``None``, arguments are parsed
    from ``sys.argv`` (legacy direct-module invocation).
    """
    from ..shared_runner import TrackAdapter, run_eval

    def _tmr_run_planner(*, manifest, ds, track_inputs, all_model_ids, out_root, args,
                          seeds, ds_sig, evaluator, graph, failures, run_root, guard,
                          text_enc, semoco_specs):
        """Planner dispatch: build closures and call planner_exec.run_global_planner."""
        from ...conversions import ConversionContext
        from ...models import default_cfg_for, load_model
        from ...models.semoco import SemocoModel
        from ...planner_exec import LoadedModel, run_global_planner as _planner_run

        del all_model_ids  # resolved from semoco_specs and args.models internally
        tmr, _tok, _fps = evaluator
        emb_sig = C.tmr_gt_sig(args.tmr_model, store=ds_sig)
        codes_root = resolve_local_uri(args.codes_root)
        tok_ckpt = Path(args.semoco_tokenizer) if args.semoco_tokenizer else default_checkpoint()
        if not tok_ckpt.is_file():
            meta_path = codes_root / f"{args.split}.meta.json"
            if meta_path.is_file():
                cand = Path(json.loads(meta_path.read_text()).get("tokenizer_checkpoint") or "")
                if cand.is_file():
                    tok_ckpt = cand

        semoco_by_name = {s.name: s for s in (semoco_specs or [])}
        ti_by_id = {t.prompt_id: t for t in track_inputs}
        clip_by_id = {c.clip_id: c for c in ds.clips}

        def model_loader(model_id: str) -> LoadedModel:
            load_kwargs: dict = {"device": args.device}
            semoco_spec = semoco_by_name.get(model_id)
            if semoco_spec is not None:
                model = SemocoModel(
                    checkpoint=str(semoco_spec.model_ckpt),
                    tokenizer_checkpoint=str(semoco_spec.tokenizer_ckpt),
                    codes_root=str(semoco_spec.codes_root),
                    text_encoder=semoco_spec.text_encoder,
                    device=args.device,
                    fk_device=args.fk_device,
                    max_tok=semoco_spec.max_tok,
                )
            else:
                model = load_model(model_id, **load_kwargs)
            guard.check_gpu_headroom(f"{model_id}: before load_model", args.device)
            guard.check_rss(f"{model.schema.model_id}: after load_model")
            ctx = ConversionContext(device=args.device, fk_device=args.fk_device)
            if getattr(model, "tokenizer", None) is not None:
                ctx.semoco_tokenizer = model.tokenizer
                ctx.semoco_anchor = model.anchor
                if hasattr(model, "build_prompt_anchor_map") and getattr(model, "dataset", None) is not None:
                    prompts_with_rec = [(t.prompt_id, t.rec_id or "", t.caption) for t in track_inputs]
                    ctx.prompt_id_anchors = model.build_prompt_anchor_map(prompts_with_rec)
            eff_cfg = default_cfg_for(model.schema.model_id, args.cfg_scale)
            return LoadedModel(model=model, ctx=ctx, eff_cfg=eff_cfg)

        def native_fn_factory(handle: LoadedModel):
            mid = handle.model.schema.model_id
            def fn(unit) -> bool:
                inputs = [ti_by_id[pid] for pid in unit.prompt_ids if pid in ti_by_id]
                if inputs:
                    guard.check_rss(f"{mid}: native-gen {unit.unit_id}")
                    ensure_native(handle.model, inputs, seeds=seeds, cfg_scale=handle.eff_cfg,
                                  dataset_sig=ds_sig, shard_index=0, num_shards=1,
                                  skip_existing=args.skip_existing,
                                  batch_size=args.gen_batch_size, failures_path=failures)
                return True
            return fn

        def convert_fn_factory(handle: LoadedModel):
            mid = handle.model.schema.model_id
            def fn(unit) -> bool:
                inputs = [ti_by_id[pid] for pid in unit.prompt_ids if pid in ti_by_id]
                if inputs:
                    guard.check_rss(f"{mid}: convert {unit.unit_id}")
                    ensure_target(handle.model, inputs, target_rep=TARGET_REP, graph=graph,
                                  ctx=handle.ctx, seeds=seeds, cfg_scale=handle.eff_cfg,
                                  dataset_sig=ds_sig, shard_index=0, num_shards=1,
                                  skip_existing=args.skip_existing, failures_path=failures,
                                  run_root=run_root)
                return True
            return fn

        def embed_fn_factory(handle: LoadedModel):
            mid = handle.model.schema.model_id
            def fn(unit) -> bool:
                clips = [clip_by_id[pid] for pid in unit.prompt_ids if pid in clip_by_id]
                if clips:
                    guard.check_rss(f"{mid}: gen-embed {unit.unit_id}")
                    _encode_gen_shard(handle.model, clips, tmr, args, seeds, handle.eff_cfg,
                                      emb_sig=emb_sig, ds_sig=ds_sig, run_root=run_root,
                                      guard=guard)
                return True
            return fn

        result = _planner_run(
            out_root=out_root, units=manifest, model_loader=model_loader,
            native_fn_factory=native_fn_factory, convert_fn_factory=convert_fn_factory,
            embed_fn_factory=embed_fn_factory,
            worker_id=f"gpu-{args.device.replace(':', '_')}-{os.getpid()}",
            lease_ttl=args.lease_ttl_s,
            on_model_closed=lambda _mid, _handle: guard.cleanup(),
        )
        print(
            f"[soma_tmr] planner pool (global, {len(args.models)} models): "
            f"committed={len(result.committed)} "
            f"skipped_already_committed={len(result.skipped_already_committed)} "
            f"quarantined={len(result.quarantined)} retried={len(result.retried)} "
            f"blocked={len(result.blocked)}",
            flush=True,
        )
        if result.blocked:
            print(f"[soma_tmr] WARNING: {len(result.blocked)} planner units still blocked "
                  f"(dependency never completed — check for a dead/expired lease)", flush=True)

    adapter = TrackAdapter(
        track_name=TRACK,
        target_rep=TARGET_REP,
        eval_fps=EVAL_FPS,
        smoke_limit=16,
        model_list=SOMA_TMR_MODELS,
        report_columns=SOMA_TMR_COLUMNS,
        report_filename="semoco_soma_table.csv",
        metric_space="soma_tmr",
        evaluator_name=args.tmr_model if args is not None else "tmr-soma-rp",
        retrieval_protocol_default=RETRIEVAL_PROTOCOL,
        gen_batch_size_default=8,
        out_dir_default="runs/eval/soma_tmr",
        has_motion_quality=True,
        model_fps_map=MODEL_FPS,

        # HOOKs
        add_args=lambda p: add_cli_args(
            p,
            require_codes_root=True,
            kimodo_metrics_default=True,
        ),
        create_dataset=_tmr_create_dataset,
        check_readiness=_tmr_check_readiness,
        dataset_sig_fn=_tmr_dataset_sig,
        load_evaluator=_tmr_load_evaluator,
        encode_gt_shard=_tmr_encode_gt_shard_hook,
        encode_gen_shard=_tmr_encode_gen_shard_hook,
        encode_text_shard=_tmr_encode_text_shard_hook,
        resolve_semoco_spec=_tmr_resolve_semoco_spec,
        build_track_inputs=_build_track_inputs,
        aggregate_fn=_tmr_aggregate,
        run_planner=_tmr_run_planner,
        cleanup=lambda ev: [close_tmr(tmr) for tmr in ev[0].values()],
    )
    run_eval(adapter, args)


def main() -> None:
    """Legacy entry point — delegates to :func:`_run_track`."""
    _run_track()


def _tmr_aggregate(ds, args, ds_sig, out_root, protocol, protocol_id, seeds, *, run_root, guard, semoco_specs=None):
    """Adapter wrapper: match shared runner's aggregate_fn signature. Multi-TMR loop."""
    codes_root = resolve_local_uri(args.codes_root)
    text_enc = args.text_encoder or "flan"
    tok_ckpt = Path(args.semoco_tokenizer) if args.semoco_tokenizer else default_checkpoint()
    tmr_models = _parse_tmr_models(args)

    if semoco_specs is None:
        semoco_specs = []

    from ...model_plan import build_model_plan

    all_entries = build_model_plan(
        args.models,
        semoco_specs,
        default_models=SOMA_TMR_MODELS,
        include_baselines=not getattr(args, "semoco_only", False),
    ).scoring_tuples()

    from ...aggregate import aggregate_soma_tmr
    from ...reports import SOMA_TMR_COLUMNS, export_scores

    all_scores: list = []
    for tmr_model_name in tmr_models:
        emb_sig = C.tmr_gt_sig(tmr_model_name, store=ds_sig)
        scores = aggregate_soma_tmr(
            clips=ds.clips, emb_sig=emb_sig, ds_sig=ds_sig, out_root=out_root,
            protocol_id=protocol_id, models=all_entries, tmr_model=tmr_model_name,
            retrieval_protocol=args.retrieval_protocol,
            kimodo_metrics=bool(args.kimodo_metrics), no_rprecision=args.no_rprecision,
            cfg_scale=args.cfg_scale, seeds=seeds, split=args.split,
            dataset_name=protocol.dataset, model_fps_map=MODEL_FPS,
            guard=guard, run_root=run_root, write_scores=True,
        )
        all_scores.extend(scores)

    if all_scores:
        (out_root / "reports").mkdir(parents=True, exist_ok=True)
        export_scores(all_scores, out_root / "reports" / "semoco_soma_table.csv", columns=SOMA_TMR_COLUMNS)
        print(f"[soma_tmr] wrote {out_root / 'reports' / 'semoco_soma_table.csv'} ({len(all_scores)} rows)", flush=True)

    return all_scores


def _encode_gt_shard(ds, selected, tmr, args, emb_sig, ds_sig):
    """Load GT joints from original Parquet data, then TMR-encode motion embeddings.

    No tokenizer involved — joints77_pos was precomputed during cursor-realign
    export and stored directly in the Parquet release.
    """
    gt_src_fps = 50.0  # UMR release source FPS
    max_tok = args.max_tok

    need = [c for c in selected if not C.probe_tmr_gt_motion(emb_sig, c.clip_id)]
    if not need:
        print(f"[soma_tmr] shard {args.shard_index}: all GT already cached, skip", flush=True)
        return

    print(f"[soma_tmr] shard {args.shard_index}: computing GT for {len(need)}/{len(selected)} clips "
          f"(reading from parquet...)", flush=True)

    # ---- Phase 1: read GT joints directly from Parquet ----
    from ....dataset.umr_parquet import load_joints77_batch
    from ...datasets.release_subset import resolve_parquet_dir

    codes_root = resolve_local_uri(args.codes_root)
    parquet_dir = resolve_parquet_dir(codes_root, args.split, cli_value=args.parquet_dir)
    joints_by_clip: dict[str, np.ndarray] = {}

    # Load cached joints first
    pending_recs: dict[str, object] = {}
    for c in need:
        j = C.load_tmr_gt_joints(emb_sig, c.clip_id, store=ds_sig)
        if j is not None:
            joints_by_clip[c.clip_id] = j
        elif c.rec_id:
            pending_recs[c.rec_id] = c

    if pending_recs:
        joints_batch = load_joints77_batch(parquet_dir, "test", list(pending_recs.keys()))
        save_items: list[tuple[str, "np.ndarray"]] = []
        for rec_id, j in joints_batch.items():
            c = pending_recs[rec_id]
            max_frames = max_tok * 4  # temporal_stride=4
            if j.shape[0] > max_frames:
                j = j[:max_frames]
            joints_by_clip[c.clip_id] = j
            save_items.append((c.clip_id, j))

        if save_items:
            print(f"[soma_tmr] shard {args.shard_index}: saving {len(save_items)} GT joints "
                  f"(batched, one fsync per bucket)...", flush=True)
            n = C.save_tmr_gt_joints_batch(emb_sig, save_items, store=ds_sig)
            print(f"[soma_tmr] shard {args.shard_index}: saved {n} GT joints ✓", flush=True)

        missed = set(pending_recs) - set(joints_batch)
        for rec_id in missed:
            print(f"[soma_tmr] shard {args.shard_index}: skip {rec_id}: not found in parquet", flush=True)

        if joints_batch:
            print(f"[soma_tmr] shard {args.shard_index}: loaded {len(joints_batch)} GT clips from parquet ✓",
                  flush=True)

    # ---- Phase 2: TMR motion encode (batched) ----
    pending_enc = [
        c for c in need
        if (not C.probe_tmr_gt_motion(emb_sig, c.clip_id)) and c.clip_id in joints_by_clip
    ]
    if pending_enc:
        print(f"[soma_tmr] shard {args.shard_index}: TMR-encoding {len(pending_enc)} GT motions "
              f"(batch on {args.device})...", flush=True)
        joints = [joints_by_clip[c.clip_id] for c in pending_enc]
        fpses = [gt_src_fps] * len(pending_enc)
        embs = _tmr_encode_batch(tmr, joints, fpses, args.device)
        # Batch save — one fsync per bucket instead of one per clip.
        emb_items = [(c.clip_id, e) for c, e in zip(pending_enc, embs)]
        C.save_tmr_gt_motion_batch(emb_sig, emb_items)
        print(f"[soma_tmr] shard {args.shard_index}: encoded {len(pending_enc)} GT motions ✓", flush=True)


def _encode_text_shard(tmr, selected, args, *, missing: list[str] | None = None):
    if missing is None:
        caps = list(dict.fromkeys(c.caption for c in selected))
        missing = [cp for cp in caps if not C.probe_tmr_text(args.tmr_model, cp)]
    if not missing:
        return
    bs = max(1, int(args.text_batch_size))
    for s in range(0, len(missing), bs):
        batch = missing[s:s + bs]
        with torch.inference_mode():
            e = np.asarray(tmr.encode_raw_text(batch, unit_vector=True).float().cpu().numpy())
        for cp, v in zip(batch, e):
            C.save_tmr_text(args.tmr_model, cp, v)
    print(f"[soma_tmr] shard {args.shard_index}: encoded {len(missing)} captions", flush=True)


def _encode_gen_shard(model, selected, tmr, args, seeds, eff_cfg, *, emb_sig, ds_sig, run_root, guard):
    """TMR-encode generated soma77 targets (batched per seed)."""
    mid = model.schema.model_id
    sig = model.weight_signature()
    n = 0
    for seed in seeds:
        pending: list[tuple[TrackInput, np.ndarray, float]] = []  # (clip, array, fps)
        warm = C.probe_gen_motion_many(
            TRACK, emb_sig, mid, sig, [c.clip_id for c in selected], int(seed), eff_cfg,
            dataset=ds_sig, run_root=run_root,
        )
        for c in selected:
            if warm[c.clip_id]:
                continue
            conv = C.load_converted(
                mid, sig, c.clip_id, int(seed), eff_cfg, TARGET_REP,
                dataset=ds_sig, run_root=run_root,
            )
            if conv is None:
                continue
            fps = float(conv.fps) or float(MODEL_FPS.get(mid, EVAL_FPS))
            pending.append((c, np.asarray(conv.array, np.float32), fps))

        if not pending:
            continue

        for chunk in guard.chunk_by_bytes(pending, lambda item: array_nbytes(item[1])):
            chunk_bytes = sum(array_nbytes(item[1]) for item in chunk)
            guard.check_rss(f"{mid}: before tmr encode_motion", planned_bytes=chunk_bytes)
            clips, arrays, fpses = zip(*chunk)
            embs = _tmr_encode_batch(tmr, list(arrays), list(fpses), args.device)
            gen_batch = [
                (c.clip_id, int(seed), eff_cfg, emb)
                for c, emb in zip(clips, embs)
            ]
            C.save_gen_motion_many(
                TRACK, emb_sig, mid, sig, gen_batch,
                dataset=ds_sig, run_root=run_root,
            )
            n += len(embs)
    if n:
        print(f"[soma_tmr] shard {args.shard_index}: {mid} encoded {n} gen motions (batched)", flush=True)


if __name__ == "__main__":
    main()
