from __future__ import annotations

import json
import importlib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from semoco_generator.tools.fetch_assets import (
    ASSET_BY_ID,
    AssetInstaller,
    AssetSpec,
    Result,
    main,
    safe_extract_zip,
    select_assets,
)
import semoco_generator.tools.fetch_assets as fetch_assets


def test_default_preset_and_all_are_stable_superset():
    default = {asset.id for asset in select_assets()}
    with_all = {asset.id for asset in select_assets(include_all=True)}

    assert {"tmr-soma-rp", "humanml-evaluator", "humanml3d", "flan-t5-xl"} <= default
    assert default < with_all
    assert {"siglip", "qwen3-embedding-4b"} <= with_all - default


def test_safe_extract_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../outside", "nope")

    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "out")
    assert not (tmp_path / "outside").exists()


def test_archive_install_normalizes_marker_root_and_skips_on_rerun(tmp_path: Path, monkeypatch):
    installer = AssetInstaller(tmp_path)
    spec = ASSET_BY_ID["humanml-evaluator"]
    calls = 0

    def fake_download(_drive_id: str, output: Path) -> None:
        nonlocal calls
        calls += 1
        with zipfile.ZipFile(output, "w") as zf:
            for marker in spec.markers:
                zf.writestr(f"release/{marker}", "x")

    monkeypatch.setattr(installer, "_download_gdrive", fake_download)
    assert installer.install(spec).status == "installed"
    assert (tmp_path / "HumanML3D/t2m" / spec.markers[0]).is_file()
    assert installer.install(spec).status == "skipped"
    assert calls == 1
    state = json.loads((tmp_path / ".semoco-assets.json").read_text())
    assert state["manifest_version"] == 1
    assert "humanml-evaluator" in state["assets"]


def test_force_and_failure_do_not_stop_later_assets(tmp_path: Path, monkeypatch):
    installer = AssetInstaller(tmp_path, force=True)
    first = AssetSpec("test-a", "gdrive_file", drive_id="id-a", target="Test/a.pth")
    second = AssetSpec("test-b", "gdrive_file", drive_id="id-b", target="Test/b.pth")

    def fake_download(drive_id: str, output: Path) -> None:
        if drive_id == first.drive_id:
            raise RuntimeError("network unavailable")
        output.write_bytes(b"second")

    monkeypatch.setattr(installer, "_download_gdrive", fake_download)
    results = [installer.install(spec) for spec in (first, second)]
    assert [result.status for result in results] == ["failed", "installed"]
    assert (tmp_path / "Test/b.pth").read_bytes() == b"second"


def test_cli_list_dry_run_and_verify_json(tmp_path: Path, capsys):
    assert main(["list", "--asset", "humanml-evaluator", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["id"] == "humanml-evaluator"

    assert main(["fetch", "--asset", "humanml-evaluator", "--dry-run", "--root", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["status"] == "planned"
    assert main(["verify", "--asset", "humanml-evaluator", "--root", str(tmp_path), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["results"][0]["status"] == "missing"


def test_external_eval_assets_use_configured_registry(tmp_path: Path, monkeypatch):
    tokenizer = tmp_path / "tokenizer.pt"
    checkpoint = tmp_path / "semoco.pt"
    codes = tmp_path / "codes"
    tokenizer.write_bytes(b"tokenizer")
    checkpoint.write_bytes(b"semoco")
    codes.mkdir()
    spec = SimpleNamespace(
        tokenizer_ckpt=tokenizer,
        model_ckpt=checkpoint,
        codes_root=codes,
    )
    monkeypatch.setattr(fetch_assets, "configured_default_specs", lambda: [spec])
    monkeypatch.setattr(fetch_assets, "default_checkpoint", lambda: tokenizer)
    installer = AssetInstaller(tmp_path / "checkpoints")

    assert installer.verify(ASSET_BY_ID["soma-tokenizer"]).status == "external-present"
    assert installer.verify(ASSET_BY_ID["semoco-checkpoint"]).status == "external-present"
    assert installer.verify(ASSET_BY_ID["t2m-code-store"]).status == "external-present"


@pytest.mark.requires_kimodo
def test_explicit_tmr_offline_resolution_never_falls_back_to_network(monkeypatch):
    hub = importlib.import_module("huggingface_hub")
    compat = importlib.import_module("semoco_generator.eval.tmr.kimodo_compat")

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")
        return "/unexpected-network-download"

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    with pytest.raises(FileNotFoundError):
        compat.resolve_tmr_checkpoint("tmr-soma-rp", local_files_only=True)
    assert calls == [{"repo_id": "nvidia/TMR-SOMA-RP-v1", "local_files_only": True}]
