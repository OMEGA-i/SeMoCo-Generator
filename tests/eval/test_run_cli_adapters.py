"""The unified CLI delegates track option registration to track adapters."""

from __future__ import annotations

from semoco_generator.eval.cli import build_parser
from semoco_generator.eval.tracks.smpl_hml.protocol import HML_SUBSET_PROTOCOL


def test_unified_run_keeps_track_options_and_track_defaults() -> None:
    args = build_parser().parse_args([
        "run", "--track", "smpl_hml", "--data-root", "/path/to/HumanML3D",
    ])

    assert args.data_root == "/path/to/HumanML3D"
    assert args.hml_protocol == HML_SUBSET_PROTOCOL
    assert args.kimodo_metrics is False


def test_unified_run_accepts_soma_tmr_adapter_options() -> None:
    args = build_parser().parse_args([
        "run", "--track", "soma_tmr", "--codes-root", "local://codes",
        "--tmr-model", "tmr-soma-rp,tmr-soma-flan", "--kimodo-metrics",
    ])

    assert args.codes_root == "local://codes"
    assert args.tmr_model == "tmr-soma-rp,tmr-soma-flan"
    assert args.kimodo_metrics is True
