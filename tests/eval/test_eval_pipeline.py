"""Tests for the schema-first dual-track eval pipeline (no torch / GPU needed).

Run::

    python -m pytest tests/test_eval_pipeline.py -q
    # or without pytest:
    python tests/test_eval_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from semoco_generator.eval.conversions import ConversionContext, ConversionGraph, build_default_graph
from semoco_generator.eval.models.registry import (
    HUMANML3D_MODELS,
    MODEL_SCHEMAS,
    SOMA_TMR_MODELS,
    default_cfg_for,
    get_model_schema,
    normalize_model_name,
)
from semoco_generator.eval.reports import HUMANML_COLUMNS, SOMA_TMR_COLUMNS, export_scores
from semoco_generator.eval.schema import LengthSpec, MotionClip
from semoco_generator.eval.score_schema import EvalScore


def test_registry_matrices():
    assert set(SOMA_TMR_MODELS).issubset(set(HUMANML3D_MODELS))
    # Every matrix model has a schema.
    for m in HUMANML3D_MODELS:
        assert get_model_schema(m) is not None, m
    # Aliases resolve.
    assert normalize_model_name("semoco") == "semoco"
    assert MODEL_SCHEMAS["semoco"].role == "ours"
    assert default_cfg_for("semoco") == 4.0
    assert default_cfg_for("semoco", 9.0) == 9.0
    print("registry matrices OK")


def test_native_output_reps():
    for model, rep in {"semoco": "motion_codes"}.items():
        assert MODEL_SCHEMAS[model].native_output == rep, model
    print("native reps OK")


def test_track_schemas():
    from semoco_generator.eval.tracks.smpl_hml.protocol import FPS as hml_fps
    from semoco_generator.eval.tracks.smpl_hml.runner import TARGET_REP as hml_rep
    from semoco_generator.eval.tracks.soma_tmr.runner import EVAL_FPS as tmr_fps, TARGET_REP as tmr_rep
    assert hml_rep == "hml263"
    assert hml_fps == 20.0
    assert tmr_rep == "soma77"
    assert tmr_fps == 30.0
    print("track schemas OK")


def test_conversion_paths_are_canonical():
    g = build_default_graph()

    # HumanML3D target: hml263. SMPL-family must reach it via joints22, NOT soma77.
    assert g.find_path("motion_codes", "hml263") == [
        "motion_codes", "soma77", "joints22", "hml263",
    ]
    assert g.find_path("soma77", "hml263") == ["soma77", "joints22", "hml263"]
    assert g.find_path("smpl_rot6d_transl", "hml263") == [
        "smpl_rot6d_transl", "joints22", "hml263",
    ]
    assert g.find_path("joints22", "hml263") == ["joints22", "hml263"]
    assert g.find_path("hml263", "hml263") == ["hml263"]

    # SOMA/TMR target: soma77.
    assert g.find_path("motion_codes", "soma77") == ["motion_codes", "soma77"]
    assert g.find_path("soma77", "soma77") == ["soma77"]
    assert g.find_path("smpl_rot6d_transl", "soma77") == [
        "smpl_rot6d_transl", "smpl_vertices", "soma77",
    ]
    # Direct device-resident edge: 1 hop instead of the old joints22 ->
    # smpl_vertices -> soma77 detour (avoids a GPU->CPU->GPU round trip).
    assert g.find_path("joints22", "soma77") == ["joints22", "soma77"]
    # No route from a HumanML-native feature back into SOMA space (not needed).
    assert g.find_path("hml263", "soma77") is None
    print("conversion paths canonical OK")


def test_length_spec():
    assert LengthSpec(seconds=2.0).to_frames(30) == 60
    assert LengthSpec(frames=45).to_frames(30) == 45
    assert LengthSpec(seconds=3.0).to_seconds(20) == 3.0
    assert LengthSpec(frames=40).to_seconds(20) == 2.0
    assert LengthSpec().to_seconds(20, default=5.0) == 5.0
    print("length spec OK")


def test_score_export(tmp_path: Path | None = None):
    out_dir = Path(tmp_path) if tmp_path is not None else Path("runs/_test_eval_export")
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = {"fid": 0.1, "r1": 0.5, "r2": 0.6, "r3": 0.7, "r5": 0.8, "r10": 0.9,
           "medr": 2.0, "matching": 3.0, "t2m_sim": 0.4, "diversity": 9.0, "multimodality": 1.0}
    score = EvalScore.from_embedding_metrics(
        model="semoco", track="soma_tmr", dataset="store", split="test",
        evaluator="tmr-soma-rp", evaluator_checkpoint="hf://...", metric_space="soma_tmr",
        retrieval_protocol="full_gallery", target_rep="soma77", protocol_id="pid",
        num_prompts=100, num_success=98, num_failed=2, num_seeds=1, emb=emb,
        foot_skate=0.03, jerk=1.2, length_mean_s=3.1, length_std_s=0.5,
        notes=["a", "b"], extras={"conversion_path": ["codes_to_soma77"]},
    )
    d = score.to_dict()
    # Trimmed schema: no native-matrix bookkeeping columns.
    for banned in ("native_rep", "conversion_path", "comparability"):
        assert banned not in d, banned
    assert d["target_rep"] == "soma77"
    csv_path = export_scores([score], out_dir / "table.csv", columns=SOMA_TMR_COLUMNS)
    header = csv_path.read_text().splitlines()[0].split(",")
    assert "comparability" not in header and "native_rep" not in header
    assert "foot_skate" in header and "fid" in header
    assert "notes" not in header
    assert "notes" not in HUMANML_COLUMNS
    assert "notes" not in SOMA_TMR_COLUMNS
    payload = json.loads(csv_path.with_suffix(".json").read_text())
    assert payload[0]["model"] == "semoco"
    # HumanML columns omit the SOMA-only motion-quality columns.
    assert "foot_skate" not in HUMANML_COLUMNS
    print("score export OK")


def test_cache_keys_and_roundtrip(tmp_path: Path | None = None):
    import os
    import tempfile

    from semoco_generator.eval.schema import MotionClip

    d = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    os.environ["SEMOCO_EVAL_CACHE_ROOT"] = str(d)
    run_art = d / "protocol" / "run_artifacts"
    os.environ.pop("SEMOCO_EVAL_RUN_ARTIFACTS_ROOT", None)
    import importlib

    from semoco_generator.eval import cache as Cmod
    importlib.reload(Cmod)  # pick up env root

    # native is durable under shared cache_root (not run_artifacts)
    clip = MotionClip(rep="smpl_rot6d_transl", array=np.ones((4, 22, 6), np.float32),
                      fps=30.0, aux={"transl": np.zeros((4, 3), np.float32)})
    Cmod.save_native("baseline1", "sigA", "c1", 0, 1.5, clip)
    r = Cmod.load_native("baseline1", "sigA", "c1", 0, 1.5)
    assert r is not None and r.rep == "smpl_rot6d_transl" and "transl" in r.aux
    assert Cmod.probe_native("baseline1", "sigA", "c1", 0, 1.5)
    assert Cmod.load_native("baseline1", "sigA", "c1", 0, 3.0) is None   # cfg differs
    assert Cmod.load_native("baseline1", "sigA", "c1", 1, 1.5) is None   # seed differs
    assert Cmod.load_native("baseline1", "sigB", "c1", 0, 1.5) is None   # ckpt sig differs
    assert "native" in str(Cmod.native_path("baseline1", "sigA", "c1", 0, 1.5))
    assert "run_artifacts" not in str(Cmod.native_path("baseline1", "sigA", "c1", 0, 1.5))

    # converted + gen_emb are run-local under run_root
    Cmod.save_converted("baseline1", "sigA", "c1", 0, 1.5, "soma77", clip, run_root=run_art)
    assert Cmod.probe_converted("baseline1", "sigA", "c1", 0, 1.5, "soma77", run_root=run_art)
    assert Cmod.load_converted("baseline1", "sigA", "c1", 0, 1.5, "soma77", run_root=run_art) is not None
    assert Cmod.load_converted("baseline1", "sigA", "c1", 0, 1.5, "hml263", run_root=run_art) is None
    assert "run_artifacts" in str(
        Cmod.converted_path("baseline1", "sigA", "c1", 0, 1.5, "soma77", run_root=run_art)
    )
    Cmod.save_gen_motion("smpl_hml", "gsig", "baseline1", "sigA", "c1", 0, 1.5,
                         np.ones(4, np.float32), run_root=run_art)
    assert Cmod.probe_gen_motion("smpl_hml", "gsig", "baseline1", "sigA", "c1", 0, 1.5, run_root=run_art)
    assert Cmod.load_gen_motion(
        "smpl_hml", "gsig", "baseline1", "sigA", "c1", 0, 1.5, run_root=run_art
    ) is not None

    # GT caches keyed by clip / caption + missing helpers
    Cmod.save_hml_gt_motion("gsig", "hml:test:M0", np.arange(8, dtype=np.float32))
    assert Cmod.probe_hml_gt_motion("gsig", "hml:test:M0")
    assert Cmod.load_hml_gt_motion("gsig", "hml:test:M0").shape == (8,)
    assert Cmod.load_hml_gt_motion("gsig", "hml:test:M1") is None
    assert Cmod.list_missing_hml_gt_motion("gsig", ["hml:test:M0", "hml:test:M1"]) == ["hml:test:M1"]
    Cmod.save_hml_gt_text("gsig", Cmod.text_key("walk", ["a/NOUN"]), np.ones(3, np.float32))
    assert Cmod.probe_hml_gt_text("gsig", Cmod.text_key("walk", ["a/NOUN"]))
    assert Cmod.load_hml_gt_text("gsig", Cmod.text_key("walk", ["a/NOUN"])) is not None
    Cmod.save_tmr_gt_joints("tok", "c1", np.zeros((5, 77, 3), np.float32), store="store_test")
    assert Cmod.probe_tmr_gt_joints("tok", "c1", store="store_test")
    assert Cmod.load_tmr_gt_joints("tok", "c1", store="store_test") is not None
    Cmod.save_tmr_gt_motion("tmr_sig", "c1", np.ones(8, np.float32))
    assert Cmod.probe_tmr_gt_motion("tmr_sig", "c1")
    assert Cmod.list_missing_tmr_gt_motion("tmr_sig", ["c1", "c2"]) == ["c2"]
    Cmod.save_tmr_text("tmr-soma-rp", "a person walks", np.ones(4, np.float32))
    assert Cmod.probe_tmr_text("tmr-soma-rp", "a person walks")
    assert Cmod.load_tmr_text("tmr-soma-rp", "a person walks") is not None
    assert Cmod.load_tmr_text("tmr-soma-rp", "other") is None
    assert Cmod.list_missing_tmr_text("tmr-soma-rp", ["a person walks", "x"]) == ["x"]

    # sigs stable + distinct
    assert Cmod.hml_gt_sig(None, official_encode=True, hml_protocol="official_hml_eval") \
        != Cmod.hml_gt_sig(None, official_encode=False, hml_protocol="official_hml_eval")
    print("cache keys + roundtrip OK")


def test_humanml_process_file_matches_official_vecs():
    """new_joints -> process_file(already_aligned=True) should match new_joint_vecs closely.

    Official disk features were built with the full align pipeline from raw mocap;
    ``new_joints`` are already in the canonical frame, so ``already_aligned=True``
    is the correct re-encode path for regenerating from ``new_joints``.
    """
    from semoco_generator.eval.tracks.smpl_hml.vendor.motion_process import process_file

    from semoco_generator.paths import humanml3d_root

    root = humanml3d_root()
    joints_path = root / "new_joints" / "000021.npy"
    vecs_path = root / "new_joint_vecs" / "000021.npy"
    if not joints_path.is_file() or not vecs_path.is_file():
        print("humanml conversion skip (assets missing)")
        return
    joints = np.load(joints_path).astype(np.float32)
    if joints.ndim == 3 and joints.shape[1] > 22:
        joints = joints[:, :22]
    official = np.load(vecs_path).astype(np.float32)
    recon = process_file(joints, already_aligned=True)
    n = min(len(recon), len(official))
    diff = np.abs(recon[:n] - official[:n])
    mae = float(diff.mean())
    # RIC/rot features should match tightly; foot-contact bits can differ slightly.
    assert mae < 0.05, f"MAE too high: {mae}"
    print(f"humanml process_file MAE={mae:.6f} OK")


def test_joints22_to_hml263_runs_on_example():
    from semoco_generator.eval.tracks.smpl_hml.conversion import joints22_to_hml263

    from semoco_generator.paths import humanml3d_root

    root = humanml3d_root()
    joints_path = root / "new_joints" / "000021.npy"
    if not joints_path.is_file():
        print("joints22_to_hml263 skip (assets missing)")
        return
    joints = np.load(joints_path).astype(np.float32)[:, :22]
    feats = joints22_to_hml263(joints, 20.0)
    assert feats.ndim == 2 and feats.shape[1] == 263
    assert feats.shape[0] == joints.shape[0] - 1  # process_file drops last frame
    print("joints22_to_hml263 shape OK")


if __name__ == "__main__":
    import tempfile

    test_registry_matrices()
    test_native_output_reps()
    test_track_schemas()
    test_conversion_paths_are_canonical()
    test_length_spec()
    with tempfile.TemporaryDirectory() as d:
        test_score_export(d)
    with tempfile.TemporaryDirectory() as d:
        test_cache_keys_and_roundtrip(d)
    test_humanml_process_file_matches_official_vecs()
    test_joints22_to_hml263_runs_on_example()
    print("\nALL EVAL PIPELINE TESTS PASSED")
