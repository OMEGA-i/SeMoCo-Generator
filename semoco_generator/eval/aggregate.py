"""Standalone, lightweight aggregation from cached embeddings.

CPU-only -- zero model loading, zero dataset creation, zero GPU init.
Reads gen/GT/text embeddings from the packed-shard cache stores and produces
:class:`EvalScore` rows for CSV/JSON export.

Usage::

    from semoco_generator.eval.aggregate import aggregate_hml, aggregate_soma_tmr
    from semoco_generator.eval.datasets.release_subset import load_subset

    clips = load_subset(out_root / "prompts.jsonl")
    scores = aggregate_hml(clips, gt_sig=..., ds_sig=..., out_root=..., ...)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from . import cache as C
from .models.registry import default_cfg_for, normalize_model_name, weight_signature
from .reports.table import HUMANML_COLUMNS, SOMA_TMR_COLUMNS, export_scores
from .resource_guard import ResourceBudget, ResourceGuard, array_nbytes
from .score_schema import EvalScore

# ---------------------------------------------------------------------------
# HML (HumanML3D / text_mot_match) aggregation
# ---------------------------------------------------------------------------

TRACK_HML = "smpl_hml"
TARGET_REP_HML = "hml263"


def aggregate_hml(
    clips: list,  # list[EvalClip]
    gt_sig: str,
    ds_sig: str,
    out_root: str | Path,
    protocol_id: str,
    models: list[tuple[str, dict, str]],  # (model_name, sig_kwargs, display_name)
    *,
    evaluator_checkpoint: str,
    retrieval_protocol: str = "batch32",
    motion_only: bool = False,
    cfg_scale: float | None = None,
    seeds: Sequence[int] = (0,),
    split: str = "test",
    dataset_name: str = "",
    guard: ResourceGuard | None = None,
    run_root: str | Path | None = None,
    write_scores: bool = True,
    lazy_models: dict[str, object] | None = None,
    lazy_device: str | None = None,
) -> list[EvalScore]:
    """CPU-only HML scoring from cached gen/GT embeddings.

    With *lazy_models* and *lazy_device*, missing gen_emb entries trigger
    on-the-fly native generation -> conversion -> encoding, filling caches as
    they go.  Models stay loaded across calls so repeat sampling converges.
    """
    out_root = Path(out_root)
    if run_root is None:
        run_root = out_root / "run_artifacts"
    if guard is None:
        guard = ResourceGuard(ResourceBudget())  # no-op (all budgets None)
    if not seeds:
        seeds = (0,)
    seed0 = seeds[0]
    scores: list[EvalScore] = []

    clip_ids = [c.clip_id for c in clips]

    # ---- Pre-load GT data once (shared across all models) ----
    print(f"[smpl_hml] batch-loading GT motion ({len(clip_ids)} clips)...", flush=True)
    gt_map = C.load_hml_gt_motion_many(gt_sig, clip_ids)

    text_keys: list[tuple[str, str]] = []
    for c in clips:
        toks = list((c.metadata or {}).get("tokens") or [])
        text_keys.append((c.caption, C.text_key(c.caption, toks)))
    if not motion_only:
        print(f"[smpl_hml] batch-loading GT text ({len(text_keys)} entries)...", flush=True)
        text_map = C.load_hml_gt_text_many(gt_sig, text_keys)
    else:
        text_map = {}

    # ---- Per-model: batch-load gen_emb, then score ----
    for model_name, sig_kwargs, display_name in models:
        mid = normalize_model_name(model_name)
        sig = weight_signature(model_name, **sig_kwargs)
        eff_cfg = default_cfg_for(mid, cfg_scale)
        score_key = display_name if sig_kwargs else mid

        print(f"[smpl_hml] batch-loading gen_emb for {score_key} ({len(clip_ids)} clips)...", flush=True)
        gen_map = C.load_gen_motion_many(
            TRACK_HML, gt_sig, mid, sig, clip_ids, seed0, eff_cfg,
            dataset=ds_sig, run_root=run_root,
        )

        # ---- Lazy-generate: fill missing gen_emb on the fly ----
        lazy_model = (lazy_models or {}).get(display_name) if lazy_models else None
        if lazy_model is not None and lazy_device is not None:
            from .generation import ensure_native, ensure_target
            from .conversions import ConversionContext, build_default_graph
            from .tracks.smpl_hml.hml_evaluator import TextMotMatchEvaluator

            missing_clips = [c for c in clips if gen_map.get(c.clip_id) is None]
            if missing_clips:
                print(f"[smpl_hml] {score_key}: lazy-generating {len(missing_clips)} missing clips...", flush=True)
                track_inputs = []
                for c in missing_clips:
                    from .schema import LengthSpec, TrackInput
                    dur = getattr(c, 'duration_s', 5.0) or 5.0
                    track_inputs.append(TrackInput(prompt_id=c.clip_id, rec_id=getattr(c, 'rec_id', ''),
                                                   caption=c.caption, length=LengthSpec(seconds=dur)))

                ensure_native(lazy_model, track_inputs, seeds=seeds, cfg_scale=eff_cfg,
                              dataset_sig=ds_sig, shard_index=0, num_shards=1,
                              skip_existing=True, batch_size=256, failures_path=None)

                graph = build_default_graph()
                ctx = ConversionContext(device=lazy_device, fk_device="cpu")
                if hasattr(lazy_model, 'tokenizer'):
                    ctx.semoco_tokenizer = lazy_model.tokenizer
                    ctx.semoco_anchor = lazy_model.anchor
                ensure_target(lazy_model, track_inputs, target_rep=TARGET_REP_HML,
                              graph=graph, ctx=ctx, seeds=seeds, cfg_scale=eff_cfg,
                              dataset_sig=ds_sig, shard_index=0, num_shards=1,
                              skip_existing=True, failures_path=None, run_root=run_root)

                # Encode gen_emb using the cached evaluator (text_mot_match)
                import json as _json
                evaluator_json = _json.loads((Path(out_root).parent / "protocol.json").read_text())
                eval_ckpt = evaluator_json.get("evaluator_checkpoint", "")
                evaluator = TextMotMatchEvaluator(eval_ckpt, device=lazy_device)
                evaluator.eval()

                pending_enc = []
                for c in missing_clips:
                    conv = C.load_converted(mid, sig, c.clip_id, seeds[0], eff_cfg, TARGET_REP_HML,
                                            dataset=ds_sig, run_root=run_root)
                    if conv is not None:
                        arr = np.asarray(conv.array, np.float32)
                        if arr.ndim == 3 and arr.shape[-1] == 263:
                            pending_enc.append((c, arr))

                if pending_enc:
                    motions = [a for _, a in pending_enc]
                    embs = evaluator.encode_motion(motions)
                    gen_batch = [(c.clip_id, int(seeds[0]), eff_cfg, emb)
                                 for (c, _), emb in zip(pending_enc, np.asarray(embs))]
                    C.save_gen_motion_many(TRACK_HML, gt_sig, mid, sig, gen_batch,
                                           dataset=ds_sig, run_root=run_root)

                gen_map = C.load_gen_motion_many(TRACK_HML, gt_sig, mid, sig, clip_ids,
                                                  seeds[0], eff_cfg, dataset=ds_sig, run_root=run_root)
                print(f"[smpl_hml] {score_key}: lazy-generate done ✓", flush=True)

        # ---- Inner loop: dict lookups only (no I/O) ----
        gen_embs, gt_embs, texts = [], [], []
        skipped_nonfinite = 0
        for c in clips:
            ge = gen_map.get(c.clip_id)
            gt = gt_map.get(c.clip_id)
            if ge is None or gt is None:
                continue
            if not (np.isfinite(ge).all() and np.isfinite(gt).all()):
                skipped_nonfinite += 1
                continue
            gen_embs.append(ge)
            gt_embs.append(gt)
            texts.append(text_map.get((c.caption, C.text_key(c.caption, list((c.metadata or {}).get("tokens") or [])))))

        if skipped_nonfinite:
            print(f"[smpl_hml] {score_key}: skipped {skipped_nonfinite} non-finite emb clips", flush=True)

        # Filter clips without text embeddings so one missing text doesn't
        # zero out R@k for the entire model (same pattern as SOMA _score_one).
        if not motion_only and any(t is None for t in texts):
            aligned = [(g, gt, t) for g, gt, t in zip(gen_embs, gt_embs, texts) if t is not None]
            missing_text = len(texts) - len(aligned)
            if missing_text:
                print(f"[smpl_hml] {score_key}: {missing_text} clip(s) missing text, "
                      f"R@k computed on {len(aligned)}", flush=True)
            if len(aligned) >= 2:
                gen_embs, gt_embs, texts = map(list, zip(*aligned))

        if len(gen_embs) < 2:
            print(f"[smpl_hml] {score_key}: insufficient run-local gens ({len(gen_embs)})", flush=True)
            continue

        planned_bytes = (
            sum(array_nbytes(x) for x in gen_embs)
            + sum(array_nbytes(x) for x in gt_embs)
            + sum(0 if t is None else array_nbytes(t) for t in texts)
        )
        guard.check_rss(f"{score_key}: before aggregate np.stack", planned_bytes=planned_bytes)

        gen_emb = np.stack(gen_embs)
        gt_emb = np.stack(gt_embs)
        text_emb = np.stack(texts) if (not motion_only and all(t is not None for t in texts)) else None

        from .tracks.smpl_hml.score import score_embeddings

        emb = score_embeddings(gen_emb, gt_emb, text=text_emb,
                               retrieval_protocol=retrieval_protocol, seed=seed0)
        ok = len(gen_embs)
        score = EvalScore.from_embedding_metrics(
            model=score_key, track=TRACK_HML, dataset=dataset_name or ds_sig,
            split=split, evaluator="text_mot_match",
            evaluator_checkpoint=str(evaluator_checkpoint), metric_space="humanml3d",
            retrieval_protocol=retrieval_protocol,
            target_rep=TARGET_REP_HML, protocol_id=protocol_id,
            num_prompts=len(clips), num_success=ok,
            num_failed=len(clips) - ok, num_seeds=len(seeds), emb=emb,
            extras={"effective_cfg": eff_cfg, "semoco_checkpoint": display_name if sig_kwargs else None},
        )

        if write_scores:
            (out_root / "scores").mkdir(parents=True, exist_ok=True)
            (out_root / "scores" / f"{score_key}.json").write_text(
                json.dumps(score.to_dict(), indent=2) + "\n"
            )
        scores.append(score)
        print(json.dumps(score.to_dict(), indent=2), flush=True)

    if write_scores and scores:
        (out_root / "reports").mkdir(parents=True, exist_ok=True)
        export_scores(scores, out_root / "reports" / "humanml3d_table.csv", columns=HUMANML_COLUMNS)
        print(f"[smpl_hml] wrote {out_root / 'reports' / 'humanml3d_table.csv'}", flush=True)

    return scores


# ---------------------------------------------------------------------------
# SOMA/TMR aggregation
# ---------------------------------------------------------------------------

TRACK_TMR = "soma_tmr"
TARGET_REP_TMR = "soma77"
EVAL_FPS_TMR = 30.0
# Model-native FPS, used when a cached clip has no meta.json to read it from.
MODEL_FPS_TMR: dict[str, float] = {
    "semoco": 50.0,
}


def _score_one_tmr(
    mid: str,
    score_key: str,
    sig_kwargs: dict,
    display_name: str,
    gen_list: list[np.ndarray],
    gt_list: list[np.ndarray],
    texts_list: list[np.ndarray | None],
    jl_list: list[np.ndarray],
    fl_list: list[float],
    *,
    retrieval_protocol: str,
    no_rprecision: bool,
    kimodo_metrics: bool,
    seed0: int,
    num_seeds: int,
    eff_cfg: float | None,
    skipped_nonfinite: int,
    out_root: Path,
    protocol_id: str,
    dataset_name: str,
    split: str,
    evaluator_name: str,
    evaluator_checkpoint: str,
    write_scores: bool,
    subset_name: str | None = None,
) -> EvalScore | None:
    """Compute and write a single EvalScore from pre-collected embedding lists."""
    if len(gen_list) < 2:
        print(f"[soma_tmr] {score_key}: insufficient gens ({len(gen_list)})", flush=True)
        return None

    # Deferred import -- only track that needs these scoring functions
    from .tracks.soma_tmr.score import motion_quality, score_embeddings

    # Filter out clips without text embeddings to keep arrays aligned
    if not no_rprecision:
        aligned = [(g, gt, t, j, f) for g, gt, t, j, f
                   in zip(gen_list, gt_list, texts_list, jl_list, fl_list)
                   if t is not None]
        if len(aligned) < len(gen_list):
            print(f"[soma_tmr] {score_key}: {len(gen_list) - len(aligned)} clips missing text, "
                  f"R@k computed on {len(aligned)}", flush=True)
        if len(aligned) >= 2:
            gen_list, gt_list, texts_list, jl_list, fl_list = (  # type: ignore[assignment]
                [x[0] for x in aligned], [x[1] for x in aligned], [x[2] for x in aligned],
                [x[3] for x in aligned], [x[4] for x in aligned],
            )

    gen_emb = np.stack(gen_list)
    gt_emb = np.stack(gt_list)
    text_emb = np.stack(texts_list) if (not no_rprecision and all(t is not None for t in texts_list)) else None

    emb = score_embeddings(
        gen_emb, gt_emb, text=text_emb,
        retrieval_protocol=retrieval_protocol, seed=seed0,
        kimodo_metrics=bool(kimodo_metrics),
    )
    mqs = [motion_quality(j, f) for j, f in zip(jl_list, fl_list)]
    fs = [m["foot_skate"] for m in mqs]
    jk = [m["jerk"] for m in mqs]
    lens = [float(j.shape[0]) / max(f, 1e-6) for j, f in zip(jl_list, fl_list)]
    num_prompts = len(gen_list) + skipped_nonfinite
    extras: dict = {
        "effective_cfg": eff_cfg,
        "semoco_checkpoint": display_name if sig_kwargs else None,
    }
    extras.update({k: emb[k] for k in ("t2m_sim_kimodo", "fid_gen_text", "fid_gt_text") if k in emb})

    score = EvalScore.from_embedding_metrics(
        model=score_key, track=TRACK_TMR, dataset=dataset_name,
        split=split, evaluator=evaluator_name,
        evaluator_checkpoint=evaluator_checkpoint, metric_space="soma_tmr",
        retrieval_protocol=retrieval_protocol,
        target_rep=TARGET_REP_TMR, protocol_id=protocol_id,
        num_prompts=num_prompts, num_success=len(gen_list),
        num_failed=skipped_nonfinite, num_seeds=num_seeds, emb=emb,
        foot_skate=float(np.mean(fs)) if fs else None,
        jerk=float(np.mean(jk)) if jk else None,
        length_mean_s=float(np.mean(lens)) if lens else None,
        length_std_s=float(np.std(lens)) if lens else None,
        extras=extras,
        subset=subset_name,
    )

    if write_scores:
        (out_root / "scores").mkdir(parents=True, exist_ok=True)
        safe_eval = evaluator_name.replace("/", "_").replace(" ", "_")
        score_file = f"{score_key.replace('/', '_')}_{safe_eval}.json"
        (out_root / "scores" / score_file).write_text(
            json.dumps(score.to_dict(), indent=2) + "\n"
        )

    if subset_name:
        print(f"[soma_tmr] {score_key}: fid={score.fid:.4f} n={len(gen_list)}", flush=True)
    else:
        print(json.dumps(score.to_dict(), indent=2), flush=True)
    return score


def aggregate_soma_tmr(
    clips: list,  # list[EvalClip]
    emb_sig: str,
    ds_sig: str,
    out_root: str | Path,
    protocol_id: str,
    models: list[tuple[str, dict, str]],  # (model_name, sig_kwargs, display_name)
    *,
    tmr_model: str = "tmr-soma-rp",
    retrieval_protocol: str = "batch32",
    kimodo_metrics: bool = True,
    no_rprecision: bool = False,
    cfg_scale: float | None = None,
    seeds: Sequence[int] = (0,),
    split: str = "test",
    dataset_name: str = "",
    model_fps_map: dict[str, float] | None = None,
    guard: ResourceGuard | None = None,
    run_root: str | Path | None = None,
    write_scores: bool = True,
    lazy_models: dict[str, object] | None = None,
    lazy_tmr: object | None = None,
    lazy_device: str | None = None,
) -> list[EvalScore]:
    """CPU-only SOMA/TMR scoring from cached gen/GT/text embeddings.

    Args:
        clips: ``EvalClip`` list (from ``load_subset(prompts.jsonl)``).
        emb_sig: TMR GT signature (from ``C.tmr_gt_sig(...)``).
        ds_sig: Dataset signature (from ``C.dataset_sig(...)``).
        out_root: Protocol output directory (scores/ and reports/ go here).
        protocol_id: Stable protocol identifier.
        models: List of ``(model_name, sig_kwargs, display_name)`` -- same format
            as built by the track runners.
        tmr_model: TMR model name for text-embedding cache keys.
        retrieval_protocol: ``"full_gallery"`` or ``"batch32"``.
        kimodo_metrics: If True, compute duplicate-aware R-precision.
        no_rprecision: If True, skip text embeddings entirely.
        cfg_scale: Override default CFG scale.
        seeds: Random seeds used (only ``seeds[0]`` is read).
        split: Dataset split name (for score metadata).
        dataset_name: Human-readable dataset label (for score metadata).
        model_fps_map: Per-model source FPS overrides.
        guard: Optional :class:`ResourceGuard` for RSS checks; a no-op guard
            with infinite budget is used when *None*.
        run_root: Run-local artifact root (default: ``out_root / "run_artifacts"``).
        write_scores: If True, write per-model ``scores/<key>.json`` and the
            final ``reports/semoco_soma_table.csv``.

    Returns:
        List of :class:`EvalScore` objects (also written to disk when
        *write_scores* is True).
    """
    out_root = Path(out_root)
    if run_root is None:
        run_root = out_root / "run_artifacts"
    if guard is None:
        guard = ResourceGuard(ResourceBudget())  # no-op (all budgets None)
    if not seeds:
        seeds = (0,)
    seed0 = seeds[0]
    fps_map = dict(model_fps_map or {})
    scores: list[EvalScore] = []

    dataset_label = dataset_name or ds_sig
    clip_ids = [c.clip_id for c in clips]
    captions = [c.caption for c in clips]

    # ---- Pre-load GT data once (shared across all models) ----
    print(f"[soma_tmr] batch-loading GT motion ({len(clip_ids)} clips)...", flush=True)
    gt_map = C.load_tmr_gt_motion_many(emb_sig, clip_ids)

    if not no_rprecision:
        print(f"[soma_tmr] batch-loading TMR text ({len(captions)} captions)...", flush=True)
        text_map = C.load_tmr_text_many(tmr_model, captions)
    else:
        text_map = {}

    # ---- Per-model: batch-load gen_emb + converted, then score ----
    for model_name, sig_kwargs, display_name in models:
        mid = normalize_model_name(model_name)
        sig = weight_signature(model_name, **sig_kwargs)
        score_key = display_name if sig_kwargs else mid
        eff_cfg = default_cfg_for(mid, cfg_scale)
        default_fps = float(fps_map.get(mid, MODEL_FPS_TMR.get(mid, EVAL_FPS_TMR)))

        print(f"[soma_tmr] batch-loading gen_emb for {score_key} ({len(clip_ids)} clips)...", flush=True)
        gen_map = C.load_gen_motion_many(
            TRACK_TMR, emb_sig, mid, sig, clip_ids, seed0, eff_cfg,
            dataset=ds_sig, run_root=run_root,
        )

        print(f"[soma_tmr] batch-loading converted for {score_key} ({len(clip_ids)} clips)...", flush=True)
        conv_map = C.load_converted_many(
            mid, sig, clip_ids, seed0, eff_cfg, TARGET_REP_TMR,
            dataset=ds_sig, run_root=run_root,
        )

        # ---- Lazy-generate: fill missing gen_emb / converted on the fly ----
        lazy_model = (lazy_models or {}).get(display_name) if lazy_models else None
        if lazy_model is not None and lazy_tmr is not None and lazy_device is not None:
            from .generation import ensure_native, ensure_target
            from .tracks.soma_tmr.runner import _tmr_encode_batch
            from .conversions import ConversionContext, build_default_graph

            missing_clips = [c for c in clips if gen_map.get(c.clip_id) is None or conv_map.get(c.clip_id) is None]
            if missing_clips:
                print(f"[soma_tmr] {score_key}: lazy-generating {len(missing_clips)} missing clips...", flush=True)
                # 1. Build TrackInputs
                track_inputs = []
                for c in missing_clips:
                    from .schema import LengthSpec, TrackInput
                    dur = getattr(c, 'duration_s', 5.0) or 5.0
                    track_inputs.append(TrackInput(prompt_id=c.clip_id, rec_id=getattr(c, 'rec_id', ''),
                                                   caption=c.caption, length=LengthSpec(seconds=dur)))

                # 2. Generate native (probe-skip, writes to durable v2)
                ensure_native(lazy_model, track_inputs, seeds=seeds, cfg_scale=eff_cfg,
                              dataset_sig=ds_sig, shard_index=0, num_shards=1,
                              skip_existing=True, batch_size=256, failures_path=None)

                # 3. Convert to soma77
                graph = build_default_graph()
                ctx = ConversionContext(device=lazy_device, fk_device="cpu")
                if hasattr(lazy_model, 'tokenizer'):
                    ctx.semoco_tokenizer = lazy_model.tokenizer
                    ctx.semoco_anchor = lazy_model.anchor
                ensure_target(lazy_model, track_inputs, target_rep=TARGET_REP_TMR,
                              graph=graph, ctx=ctx, seeds=seeds, cfg_scale=eff_cfg,
                              dataset_sig=ds_sig, shard_index=0, num_shards=1,
                              skip_existing=True, failures_path=None, run_root=run_root)

                # 4. Encode gen_emb
                pending_enc = []
                for c in missing_clips:
                    conv = C.load_converted(mid, sig, c.clip_id, seeds[0], eff_cfg, TARGET_REP_TMR,
                                            dataset=ds_sig, run_root=run_root)
                    if conv is not None:
                        fps = float(getattr(conv, 'fps', None) or default_fps)
                        pending_enc.append((c, np.asarray(conv.array, np.float32), fps))

                if pending_enc:
                    joints_list = [j for _, j, _ in pending_enc]
                    fps_list = [f for _, _, f in pending_enc]
                    embs = _tmr_encode_batch(lazy_tmr, joints_list, fps_list, lazy_device)
                    gen_batch = [(c.clip_id, int(seeds[0]), eff_cfg, emb)
                                 for (c, _, _), emb in zip(pending_enc, embs)]
                    C.save_gen_motion_many(TRACK_TMR, emb_sig, mid, sig, gen_batch,
                                           dataset=ds_sig, run_root=run_root)

                # Reload gen/conv maps with newly filled entries
                gen_map = C.load_gen_motion_many(TRACK_TMR, emb_sig, mid, sig, clip_ids,
                                                  seeds[0], eff_cfg, dataset=ds_sig, run_root=run_root)
                conv_map = C.load_converted_many(mid, sig, clip_ids, seeds[0], eff_cfg,
                                                  TARGET_REP_TMR, dataset=ds_sig, run_root=run_root)
                print(f"[soma_tmr] {score_key}: lazy-generate done ✓", flush=True)

        # ---- Inner loop: dict lookups only (no I/O) ----
        by_subset: dict[str, dict] = defaultdict(
            lambda: {"gen": [], "gt": [], "texts": [], "jl": [], "fl": [], "skipped": 0}
        )
        for c in clips:
            ge = gen_map.get(c.clip_id)
            gt = gt_map.get(c.clip_id)
            if ge is None or gt is None:
                continue
            sub = c.subset or "unknown"
            bucket = by_subset[sub]
            if not (np.isfinite(ge).all() and np.isfinite(gt).all()):
                bucket["skipped"] += 1
                continue
            text = text_map.get(c.caption)
            bucket["gen"].append(ge)
            bucket["gt"].append(gt)
            bucket["texts"].append(text)
            conv = conv_map.get(c.clip_id)
            if conv is not None:
                bucket["jl"].append(np.asarray(conv.array, np.float32))
                bucket["fl"].append(float(conv.fps) or default_fps)

        # Derive overall aggregates from per-subset buckets.
        overall_gen = [e for sub in by_subset.values() for e in sub["gen"]]
        overall_gt = [e for sub in by_subset.values() for e in sub["gt"]]
        overall_texts = [e for sub in by_subset.values() for e in sub["texts"]]
        overall_jl = [e for sub in by_subset.values() for e in sub["jl"]]
        overall_fl = [e for sub in by_subset.values() for e in sub["fl"]]
        overall_skipped = sum(sub["skipped"] for sub in by_subset.values())

        if overall_skipped:
            print(f"[soma_tmr] {mid}: skipped {overall_skipped} non-finite emb clips", flush=True)

        # ---- Overall score ----
        if len(overall_gen) >= 2:
            score = _score_one_tmr(
                mid, score_key, sig_kwargs, display_name,
                overall_gen, overall_gt, overall_texts, overall_jl, overall_fl,
                retrieval_protocol=retrieval_protocol,
                no_rprecision=no_rprecision,
                kimodo_metrics=kimodo_metrics,
                seed0=seed0, num_seeds=len(seeds), eff_cfg=eff_cfg,
                skipped_nonfinite=overall_skipped, out_root=out_root,
                protocol_id=protocol_id, dataset_name=dataset_label,
                split=split, evaluator_name=tmr_model,
                evaluator_checkpoint=tmr_model,
                write_scores=write_scores,
            )
            if score is not None:
                scores.append(score)

        # ---- Per-subset scores (only when multiple real subsets exist) ----
        if len(by_subset) > 1:
            for sub_name in sorted(by_subset):
                sub = by_subset[sub_name]
                if len(sub["gen"]) < 2:
                    continue
                sub_score = _score_one_tmr(
                    mid, f"{score_key}/{sub_name}", sig_kwargs, display_name,
                    sub["gen"], sub["gt"], sub["texts"], sub["jl"], sub["fl"],
                    retrieval_protocol=retrieval_protocol,
                    no_rprecision=no_rprecision,
                    kimodo_metrics=kimodo_metrics,
                    seed0=seed0, num_seeds=len(seeds), eff_cfg=eff_cfg,
                    skipped_nonfinite=sub["skipped"], out_root=out_root,
                    protocol_id=protocol_id, dataset_name=dataset_label,
                    split=split, evaluator_name=tmr_model,
                    evaluator_checkpoint=tmr_model,
                    write_scores=write_scores,
                    subset_name=sub_name,
                )
                if sub_score is not None:
                    scores.append(sub_score)

    if write_scores and scores:
        (out_root / "reports").mkdir(parents=True, exist_ok=True)
        export_scores(scores, out_root / "reports" / "semoco_soma_table.csv", columns=SOMA_TMR_COLUMNS)
        print(f"[soma_tmr] wrote {out_root / 'reports' / 'semoco_soma_table.csv'}", flush=True)

    return scores


__all__ = [
    "MODEL_FPS_TMR",
    "aggregate_hml",
    "aggregate_soma_tmr",
]
