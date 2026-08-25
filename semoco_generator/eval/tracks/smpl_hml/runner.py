"""End-to-end HumanML3D evaluation track runner (sharded, resumable).

Durable shared cache (``SEMOCO_EVAL_CACHE_ROOT``):
  - GT motion/text embeddings (evaluator truths)
  - model native generations (same prompts reuse across tracks)

Run-local under ``<out_root>/run_artifacts``:
  - converted hml263 targets (conversion-graph dependent)
  - generated-motion embeddings

Shard phase (GPU): fill missing GT, generate natives, convert, encode gens.
Aggregate phase (CPU, ``--aggregate-only``): stable GT + run-local gen emb → score.

Typical sharded full run:
    # optional: prewarm GT via tracks.smpl_hml.precompute
    for shard in 0..N-1:  python -m semoco_generator.eval.cli run --track smpl_hml ... --num-shards N --shard-index shard
    python -m semoco_generator.eval.cli run --track smpl_hml ... --aggregate-only

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
import os
from pathlib import Path

import numpy as np

from ... import cache as C
from ...checkpoints import CheckpointSpec
from ...conversions import ConversionContext
from ...generation import ensure_native, ensure_target
from ...models import HUMANML3D_MODELS, default_cfg_for, load_model
from ...planner_exec import LoadedModel, run_global_planner
from ...reports import HUMANML_COLUMNS
from ...resource_guard import ResourceGuard, array_nbytes
from ...schema import LengthSpec, TrackInput
from .check_readiness import check_readiness
from .dataset import HumanML3DDataset
from .hml_evaluator import TextMotMatchEvaluator
from .paths import resolve_humanml_asset
from .protocol import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MEAN_STD,
    FPS,
    HML_SUBSET_PROTOCOL,
    RETRIEVAL_PROTOCOL,
)
from .word_vectorizer import load_word_vectorizer, resolve_glove_root

TARGET_REP = "hml263"
TRACK = "smpl_hml"


def add_cli_args(p: argparse.ArgumentParser, *, require_data_root: bool = False) -> None:
    """Register HumanML-specific options on either supported CLI entry point."""
    p.add_argument("--data-root", required=require_data_root, help="Path to HumanML3D directory")
    p.add_argument("--evaluator-checkpoint", default=str(DEFAULT_CHECKPOINT),
                   help="Path to text_mot_match evaluator checkpoint")
    p.add_argument("--mean-std-dir", default=str(DEFAULT_MEAN_STD),
                   help="Path to HumanML Mean/Std directory")
    p.add_argument("--glove-root", default=None, help="Path to GloVe vectors directory")
    p.add_argument("--motion-only", action="store_true", help="Skip text embeddings (FID-only)")
    p.add_argument("--hml-protocol", choices=("official_hml_eval", "legacy_full_test"),
                   default=HML_SUBSET_PROTOCOL,
                   help="HumanML subset/evaluation protocol")
    # Accepted for compatibility with the unified CLI's former spelling.
    p.add_argument("--official-encode", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-official-encode", action="store_true", help="Use raw encode path")


def _load_mean_std(meta_dir: Path | None):
    if meta_dir is None:
        return None, None
    mean_p, std_p = meta_dir / "mean.npy", meta_dir / "std.npy"
    if mean_p.is_file() and std_p.is_file():
        return np.load(mean_p).astype(np.float32), np.load(std_p).astype(np.float32)
    return None, None


def _gt_feature(clip, data_root: Path) -> np.ndarray:
    src = resolve_humanml_asset(data_root, "new_joint_vecs", clip.rec_id)
    if src is None:
        raise FileNotFoundError(f"missing HumanML new_joint_vecs for {clip.rec_id}")
    arr = np.load(src).astype(np.float32)
    meta = clip.metadata or {}
    fs = int(meta.get("frame_start") or 0)
    fe = meta.get("frame_end")
    arr = arr[fs: int(fe)] if fe is not None else arr[fs:]
    if meta.get("m_length") is not None:
        arr = arr[: int(meta["m_length"])]
    return arr


def _m_length(clip) -> int | None:
    meta = clip.metadata or {}
    return int(meta.get("m_length") or meta.get("n_frames") or 0) or None


def _build_track_inputs(clips) -> list[TrackInput]:
    out: list[TrackInput] = []
    for clip in clips:
        meta = clip.metadata or {}
        out.append(TrackInput(prompt_id=clip.clip_id, rec_id=clip.rec_id, caption=clip.caption,
                              length=LengthSpec(seconds=clip.duration_s),
                              text_payload={"tokens": list(meta.get("tokens") or [])}))
    return out


def _semoco_spec_from_legacy_args(args, text_enc: str) -> CheckpointSpec:
    """Build a :class:`CheckpointSpec` from legacy ``--semoco-checkpoint`` etc. args."""
    from pathlib import Path as _Path
    from ...checkpoints import discover_from_dir

    ckpt = _Path(args.semoco_checkpoint)
    # Try auto-discovery from the run directory first
    run_dir = ckpt.parent.parent  # model/best.pt → runs/<name>
    discovered = discover_from_dir(run_dir)
    if discovered is not None:
        # Override with any explicit args
        return CheckpointSpec(
            name=discovered.name,
            model_ckpt=discovered.model_ckpt,
            tokenizer_ckpt=(
                _Path(args.semoco_tokenizer) if args.semoco_tokenizer
                else discovered.tokenizer_ckpt
            ),
            codes_root=(
                _Path(args.semoco_codes_root) if args.semoco_codes_root
                else discovered.codes_root
            ),
            text_encoder=text_enc,
            max_tok=args.max_tok,
        )
    # Fallback: build from scratch (requires explicit codes_root)
    if not args.semoco_tokenizer:
        raise SystemExit("--semoco-tokenizer is required (cannot auto-discover)")
    return CheckpointSpec(
        name=ckpt.parent.parent.name,
        model_ckpt=ckpt,
        tokenizer_ckpt=_Path(args.semoco_tokenizer),
        codes_root=_Path(args.semoco_codes_root) if args.semoco_codes_root else _Path("."),
        text_encoder=text_enc,
        max_tok=args.max_tok,
    )


def _run_track(args: argparse.Namespace | None = None) -> None:
    """Build adapter and dispatch to shared runner.

    When *args* is provided (from the unified ``eval run`` CLI), internal
    arg-parsing is skipped.  When *args* is ``None``, arguments are parsed
    from ``sys.argv`` (legacy direct-module invocation).
    """
    from pathlib import Path as _Path
    from ..shared_runner import TrackAdapter, run_eval

    # ---- gt_sig derived from args (used by multiple hooks) ----
    def _hml_gt_sig(args):
        return C.hml_gt_sig(
            args.evaluator_checkpoint,
            official_encode=not args.no_official_encode,
            hml_protocol=args.hml_protocol,
            data=_Path(args.data_root).name,
            mean_std=args.mean_std_dir,
            glove=args.glove_root,
        )

    # ---- Hook implementations ----
    def _hml_create_dataset(args, *, limit, text_enc):
        del text_enc  # HML dataset does not need text_encoder
        return HumanML3DDataset(args.data_root, args.split, limit=limit, seed=0,
                                protocol=args.hml_protocol)

    def _hml_check_readiness(args):
        return check_readiness(
            args.data_root,
            args.evaluator_checkpoint or DEFAULT_CHECKPOINT,
            mean_std_dir=args.mean_std_dir or DEFAULT_MEAN_STD,
            glove_root=args.glove_root,
            require_text=not getattr(args, 'motion_only', False),
        )

    def _hml_dataset_sig(args):
        return C.dataset_sig(args.data_root, args.split)

    def _hml_evaluator_checkpoint_fn(args):
        return str(args.evaluator_checkpoint or DEFAULT_CHECKPOINT)

    def _hml_load_evaluator(args, *, ds_sig):
        del ds_sig
        mean, std = _load_mean_std(_Path(args.mean_std_dir or DEFAULT_MEAN_STD))
        wv = None
        if not getattr(args, 'motion_only', False):
            glove = resolve_glove_root([args.glove_root] if args.glove_root else None)
            if glove is not None:
                wv = load_word_vectorizer(glove)
        return TextMotMatchEvaluator(
            args.evaluator_checkpoint or str(DEFAULT_CHECKPOINT), device=args.device,
            word_vectorizer=wv, mean=mean, std=std,
            official_protocol=not args.no_official_encode,
        )

    def _hml_encode_gt_shard(ds, selected, evaluator, args, ds_sig):
        del ds  # HML uses data_root from args instead
        gt_sig = _hml_gt_sig(args)
        data_root = _Path(args.data_root)
        return _encode_gt_shard(selected, evaluator, data_root, gt_sig, args)

    def _hml_encode_gen_shard(model, selected, evaluator, args, seeds, eff_cfg, *,
                               ds_sig, run_root, guard):
        gt_sig = _hml_gt_sig(args)
        return _encode_gen_shard(model, selected, evaluator, gt_sig, ds_sig, args, seeds,
                                 eff_cfg, run_root=run_root, guard=guard)

    def _hml_aggregate(ds, args, ds_sig, out_root, protocol, protocol_id, seeds, *,
                        run_root, guard, semoco_specs=None):
        gt_sig = _hml_gt_sig(args)
        return _aggregate(ds, args, gt_sig, ds_sig, out_root, protocol, protocol_id, seeds,
                          run_root=run_root, guard=guard, semoco_specs=semoco_specs)

    def _hml_resolve_semoco_spec(args, text_enc):
        return _semoco_spec_from_legacy_args(args, text_enc)

    def _hml_run_planner(*, manifest, ds, track_inputs, all_model_ids, out_root, args,
                          seeds, ds_sig, evaluator, graph, failures, run_root, guard,
                          text_enc, semoco_specs):
        del all_model_ids  # HML planner resolves models from spec names internally
        gt_sig = _hml_gt_sig(args)
        return _run_global_planner(
            ds=ds, track_inputs=track_inputs, units=manifest, out_root=out_root,
            args=args, seeds=seeds, ds_sig=ds_sig, gt_sig=gt_sig, evaluator=evaluator,
            graph=graph, failures=failures, run_root=run_root, guard=guard,
            text_enc=text_enc, semoco_specs=semoco_specs,
        )

    # ---- Build adapter and run ----
    adapter = TrackAdapter(
        track_name=TRACK,
        target_rep=TARGET_REP,
        eval_fps=FPS,
        smoke_limit=8,
        model_list=HUMANML3D_MODELS,
        report_columns=HUMANML_COLUMNS,
        report_filename="humanml3d_table.csv",
        metric_space="humanml3d",
        evaluator_name="text_mot_match",
        subset_source="humanml3d",
        retrieval_protocol_default=RETRIEVAL_PROTOCOL,
        gen_batch_size_default=8,
        out_dir_default="runs/eval/smpl_hml",

        add_args=lambda p: add_cli_args(p, require_data_root=True),
        evaluator_checkpoint_fn=_hml_evaluator_checkpoint_fn,
        create_dataset=_hml_create_dataset,
        check_readiness=_hml_check_readiness,
        dataset_sig_fn=_hml_dataset_sig,
        load_evaluator=_hml_load_evaluator,
        encode_gt_shard=_hml_encode_gt_shard,
        encode_gen_shard=_hml_encode_gen_shard,
        build_track_inputs=_build_track_inputs,
        aggregate_fn=_hml_aggregate,
        resolve_semoco_spec=_hml_resolve_semoco_spec,
        run_planner=_hml_run_planner,
    )
    run_eval(adapter, args)


def main() -> None:
    """Legacy entry point — delegates to :func:`_run_track`."""
    _run_track()


def _run_global_planner(
    *, ds, track_inputs, units, out_root, args, seeds, ds_sig, gt_sig, evaluator, graph,
    failures, run_root, guard, text_enc, semoco_specs=None,
) -> None:
    """Global scheduler spanning every model in ``all_model_ids``: one worker
    dynamically leases whichever ready unit is available across ALL models
    (not just whichever model an outer loop happened to load), so a worker
    that runs out of ready work for a light model immediately helps with a
    heavier model's remaining units instead of exiting early. Models are
    loaded lazily on first use and closed once every worker unit is done."""
    if semoco_specs is None:
        semoco_specs = []
    semoco_by_name = {s.name: s for s in semoco_specs}
    ti_by_id = {t.prompt_id: t for t in track_inputs}
    clip_by_id = {c.clip_id: c for c in ds.clips}

    def model_loader(model_id: str) -> LoadedModel:
        load_kwargs: dict = {"device": args.device}
        # Check if this is an Semoco spec name
        semoco_spec = semoco_by_name.get(model_id)
        if semoco_spec is not None:
            from ...models.semoco import SemocoModel

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
            # Baseline
            model = load_model(model_id, **load_kwargs)

        guard.check_gpu_headroom(f"{model_id}: before load_model", args.device)
        # model already loaded above; just RSS check
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
                ensure_native(handle.model, inputs, seeds=seeds, cfg_scale=handle.eff_cfg, dataset_sig=ds_sig,
                              shard_index=0, num_shards=1, skip_existing=args.skip_existing,
                              batch_size=args.gen_batch_size, failures_path=failures, run_root=run_root)
            return True
        return fn

    def convert_fn_factory(handle: LoadedModel):
        mid = handle.model.schema.model_id

        def fn(unit) -> bool:
            inputs = [ti_by_id[pid] for pid in unit.prompt_ids if pid in ti_by_id]
            if inputs:
                guard.check_rss(f"{mid}: convert {unit.unit_id}")
                ensure_target(handle.model, inputs, target_rep=TARGET_REP, graph=graph, ctx=handle.ctx,
                              seeds=seeds, cfg_scale=handle.eff_cfg, dataset_sig=ds_sig, shard_index=0,
                              num_shards=1, skip_existing=args.skip_existing, failures_path=failures,
                              run_root=run_root)
            return True
        return fn

    def embed_fn_factory(handle: LoadedModel):
        mid = handle.model.schema.model_id

        def fn(unit) -> bool:
            clips = [clip_by_id[pid] for pid in unit.prompt_ids if pid in clip_by_id]
            if clips:
                guard.check_rss(f"{mid}: gen-embed {unit.unit_id}")
                _encode_gen_shard(handle.model, clips, evaluator, gt_sig, ds_sig, args, seeds, handle.eff_cfg,
                                  run_root=run_root, guard=guard)
            return True
        return fn

    result = run_global_planner(
        out_root=out_root, units=units, model_loader=model_loader,
        native_fn_factory=native_fn_factory, convert_fn_factory=convert_fn_factory,
        embed_fn_factory=embed_fn_factory,
        worker_id=f"gpu-{args.device.replace(':', '_')}-{os.getpid()}", lease_ttl=args.lease_ttl_s,
        on_model_closed=lambda _mid, _handle: guard.cleanup(),
    )
    print(
        f"[smpl_hml] planner pool (global, {len(args.models)} models): committed={len(result.committed)} "
        f"skipped_already_committed={len(result.skipped_already_committed)} "
        f"quarantined={len(result.quarantined)} retried={len(result.retried)} "
        f"blocked={len(result.blocked)}",
        flush=True,
    )
    if result.blocked:
        print(f"[smpl_hml] WARNING: {len(result.blocked)} planner units still blocked "
              f"(dependency never completed — check for a dead/expired lease)", flush=True)


def _encode_gt_shard(selected, evaluator, data_root, gt_sig, args):
    # GT motion embeddings
    miss = [c for c in selected if not C.probe_hml_gt_motion(gt_sig, c.clip_id)]
    if miss:
        feats = [_gt_feature(c, data_root) for c in miss]
        lens = [_m_length(c) for c in miss]
        embs = evaluator.encode_motion(feats, lengths=lens)
        for c, e in zip(miss, embs):
            C.save_hml_gt_motion(gt_sig, c.clip_id, e)
        print(f"[smpl_hml] shard {args.shard_index}: encoded {len(miss)} GT motions", flush=True)
    # GT text embeddings
    if args.motion_only:
        return
    tmiss, tkeys = [], []
    for c in selected:
        toks = list((c.metadata or {}).get("tokens") or [])
        tk = C.text_key(c.caption, toks)
        if not C.probe_hml_gt_text(gt_sig, tk):
            tmiss.append((c, toks, tk))
    if tmiss:
        try:
            embs = evaluator.encode_text([c.caption for c, _, _ in tmiss], tokens=[t for _, t, _ in tmiss])
            for (c, _t, tk), e in zip(tmiss, embs):
                C.save_hml_gt_text(gt_sig, tk, e)
            print(f"[smpl_hml] shard {args.shard_index}: encoded {len(tmiss)} GT texts", flush=True)
        except RuntimeError as exc:
            print(f"[smpl_hml] text encode skipped: {exc}", flush=True)


def _encode_gen_shard(model, selected, evaluator, gt_sig, ds_sig, args, seeds, eff_cfg, *, run_root, guard):
    mid = model.schema.model_id
    sig = model.weight_signature()
    for seed in seeds:
        pending: list[tuple[object, np.ndarray, int]] = []
        warm = C.probe_gen_motion_many(
            TRACK, gt_sig, mid, sig, [c.clip_id for c in selected], int(seed), eff_cfg,
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
            feat = np.asarray(conv.array, np.float32)
            ml = _m_length(c)
            pending.append((c, feat, min(int(ml), int(feat.shape[0])) if ml else int(feat.shape[0])))
        if not pending:
            continue
        encoded = 0
        for chunk in guard.chunk_by_bytes(pending, lambda item: array_nbytes(item[1])):
            chunk_bytes = sum(array_nbytes(item[1]) for item in chunk)
            guard.check_rss(f"{mid}: before hml encode_motion", planned_bytes=chunk_bytes)
            miss = [item[0] for item in chunk]
            feats = [item[1] for item in chunk]
            lens = [item[2] for item in chunk]
            embs = evaluator.encode_motion(feats, lengths=lens)
            gen_batch = [
                (c.clip_id, int(seed), eff_cfg, e)
                for c, e in zip(miss, embs)
            ]
            C.save_gen_motion_many(
                TRACK, gt_sig, mid, sig, gen_batch,
                dataset=ds_sig, run_root=run_root,
            )
            encoded += len(chunk)
        print(f"[smpl_hml] shard {args.shard_index}: {mid} encoded {encoded} gen motions", flush=True)


def _aggregate(ds, args, gt_sig, ds_sig, out_root, protocol, protocol_id, seeds, *, run_root, guard,
               semoco_specs=None):
    if semoco_specs is None:
        semoco_specs = []

    from ...model_plan import build_model_plan

    all_entries = build_model_plan(
        args.models,
        semoco_specs,
        default_models=HUMANML3D_MODELS,
        include_baselines=not getattr(args, "semoco_only", False),
    ).scoring_tuples()

    # Delegate to standalone aggregate function
    from ...aggregate import aggregate_hml

    return aggregate_hml(
        clips=ds.clips,
        gt_sig=gt_sig,
        ds_sig=ds_sig,
        out_root=out_root,
        protocol_id=protocol_id,
        models=all_entries,
        evaluator_checkpoint=str(args.evaluator_checkpoint),
        retrieval_protocol=args.retrieval_protocol,
        motion_only=args.motion_only,
        cfg_scale=args.cfg_scale,
        seeds=seeds,
        split=args.split,
        dataset_name=protocol.dataset,
        guard=guard,
        run_root=run_root,
        write_scores=True,
    )


if __name__ == "__main__":
    main()
