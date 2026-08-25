"""Install and diagnose public SeMoCo-Generator evaluation assets.

The command intentionally keeps Hub repositories in Hugging Face's normal
cache.  Only files without a Hub source, and generated runtime artifacts, are
installed under ``$SEMOCO_BASELINE_CKPT_ROOT``.  This makes the command safe to
rerun on shared machines and prevents a second copy of multi-GB models.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

from ..paths import (
    baseline_checkpoint_root,
    default_checkpoint,
    glove_root,
    humanml3d_root,
    smpl_model_path,
    smplx_model_path,
)
from ..eval.checkpoints import configured_default_specs

MANIFEST_VERSION = 1
STATE_FILENAME = ".semoco-assets.json"


@dataclass(frozen=True)
class AssetSpec:
    """One independently-installable public asset or local prerequisite."""

    id: str
    kind: str
    source: str | None = None
    revision: str | None = None
    files: tuple[str, ...] = ()
    markers: tuple[str, ...] = ()
    preset: str = "full-eval"
    optional: bool = False
    drive_id: str | None = None
    target: str | None = None
    description: str = ""


# Keep this catalog deliberately declarative.  It doubles as ``assets list``
# and gives us a single reviewable place for public upstream locations.
ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec("tmr-soma-rp", "hf_snapshot", "nvidia/TMR-SOMA-RP-v1", markers=("config.yaml", "stats/motion/body/mean.npy", "last_weights/motion_encoder.pt"), description="SOMA/TMR evaluator"),
    AssetSpec("humanml-evaluator", "gdrive_archive", "https://github.com/EricGuo5513/momask-codes", markers=("text_mot_match/model/finest.tar", "Comp_v6_KLD01/meta/mean.npy", "Comp_v6_KLD01/meta/std.npy"), drive_id="19C_eiEr0kMGlYVJy_yFL6_Dhk3RvmwhM", target="HumanML3D/t2m", description="Official HumanML matching evaluator"),
    AssetSpec("humanml3d", "humanml3d", "jbs99/humanml3d", files=("HumanML3D.zip",), markers=("test.txt", "Mean.npy", "Std.npy", "texts", "new_joint_vecs"), description="HumanML3D evaluation dataset"),
    AssetSpec("glove", "glove", markers=("our_vab_data.npy", "our_vab_words.pkl", "our_vab_idx.pkl"), description="HumanML evaluator GloVe vocabulary"),
    AssetSpec("flan-t5-xl", "hf_snapshot", "google/flan-t5-xl", description="Flan-T5 text encoder (T2M + TMR-Flan)"),
    AssetSpec("llm2vec-base", "hf_snapshot", "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", description="TMR R-precision LLM2Vec base"),
    AssetSpec("llm2vec-adapter", "hf_snapshot", "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", description="TMR R-precision LLM2Vec adapter"),
    AssetSpec("smpl", "external", markers=("smpl/SMPL_NEUTRAL.pkl",), description="External SMPL model ($SMPL_MODEL_PATH)"),
    AssetSpec("smplx", "external", markers=("smplx/SMPLX_NEUTRAL.npz",), description="External SMPL-X model ($SMPLX_MODEL_PATH)"),
    AssetSpec("soma-tokenizer", "external", description="Frozen SeMoCo tokenizer checkpoint ($SOMA_TOKENIZER_ROOT)"),
    AssetSpec("semoco-checkpoint", "external", description="Local T2M run checkpoint"),
    AssetSpec("t2m-code-store", "external", description="Local local:// T2M code store"),
    AssetSpec("siglip", "hf_snapshot", "google/siglip-so400m-patch14-384", preset="all", optional=True, description="Optional text encoder"),
    AssetSpec("qwen3-embedding-4b", "hf_snapshot", "Qwen/Qwen3-Embedding-4B", preset="all", optional=True, description="Optional text encoder"),
)
ASSET_BY_ID = {asset.id: asset for asset in ASSETS}


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Return resolved paths once, preserving their configuration order."""
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


@dataclass
class Result:
    id: str
    status: str
    detail: str = ""
    source: str | None = None
    markers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_assets(ids: Iterable[str] = (), *, include_all: bool = False) -> list[AssetSpec]:
    """Select named assets, or the complete stable public evaluation preset."""
    requested = list(ids)
    unknown = sorted(set(requested) - set(ASSET_BY_ID))
    if unknown:
        raise ValueError(f"unknown asset id(s): {', '.join(unknown)}")
    if requested:
        selected = [ASSET_BY_ID[name] for name in requested]
    else:
        selected = [asset for asset in ASSETS if asset.preset == "full-eval"]
    if include_all:
        seen = {asset.id for asset in selected}
        selected.extend(asset for asset in ASSETS if asset.preset == "all" and asset.id not in seen)
    return selected


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a zip without accepting absolute or parent-traversal members."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            rel = PurePosixPath(info.filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe archive member: {info.filename!r}")
            if not info.filename or info.is_dir():
                continue
            target = destination.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _find_marker_root(stage: Path, markers: tuple[str, ...]) -> Path | None:
    if all((stage / marker).exists() for marker in markers):
        return stage
    first = markers[0] if markers else ""
    for candidate in stage.rglob(Path(first).name):
        parent = candidate
        for _ in Path(first).parts:
            parent = parent.parent
        if all((parent / marker).exists() for marker in markers):
            return parent
    return None


class AssetInstaller:
    def __init__(self, root: Path | None = None, *, offline: bool = False, force: bool = False):
        self.root = Path(root or baseline_checkpoint_root()).expanduser().resolve()
        self.offline = offline
        self.force = force
        self.state_path = self.root / STATE_FILENAME
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.state_path.read_text())
            return raw if raw.get("manifest_version") == MANIFEST_VERSION else {"manifest_version": MANIFEST_VERSION, "assets": {}}
        except (FileNotFoundError, json.JSONDecodeError):
            return {"manifest_version": MANIFEST_VERSION, "assets": {}}

    def _save_state(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        os.replace(temp, self.state_path)

    def _target(self, spec: AssetSpec) -> Path | None:
        if spec.target:
            return self.root / spec.target
        if spec.kind == "humanml3d":
            return humanml3d_root()
        if spec.kind == "glove":
            return glove_root()
        return None

    def _external_markers(self, spec: AssetSpec) -> list[Path]:
        if spec.id == "smpl":
            return [smpl_model_path() / marker for marker in spec.markers]
        if spec.id == "smplx":
            return [smplx_model_path() / marker for marker in spec.markers]
        if spec.id == "soma-tokenizer":
            return [default_checkpoint()]
        default_specs = configured_default_specs()
        if spec.id == "semoco-checkpoint":
            return _unique_paths(item.model_ckpt for item in default_specs)
        if spec.id == "t2m-code-store":
            return _unique_paths(item.codes_root for item in default_specs)
        return []

    def _verify_hf(self, spec: AssetSpec) -> tuple[bool, str, list[str]]:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            if spec.kind == "hf_file":
                files = [Path(hf_hub_download(repo_id=spec.source, filename=name, local_files_only=True)) for name in spec.files]
                return all(p.is_file() for p in files), str(files[0].parent) if files else "", [str(p) for p in files]
            directory = Path(snapshot_download(
                repo_id=spec.source, local_files_only=True,
                allow_patterns=list(spec.files) if spec.files else None,
            ))
            markers = spec.markers or spec.files
            return all((directory / marker).is_file() for marker in markers), str(directory), [str(directory / marker) for marker in markers]
        except Exception as exc:  # noqa: BLE001
            return False, str(exc), []

    def verify(self, spec: AssetSpec) -> Result:
        if spec.kind.startswith("hf_"):
            ready, detail, markers = self._verify_hf(spec)
            return Result(spec.id, "present" if ready else "missing", detail, spec.source, markers)
        if spec.kind == "external":
            try:
                markers = self._external_markers(spec)
            except FileNotFoundError as exc:
                return Result(spec.id, "missing-external", str(exc), markers=[str(exc)])
            if not markers:
                return Result(spec.id, "missing-external", spec.description)
            return Result(spec.id, "external-present" if all(p.exists() for p in markers) else "missing-external", spec.description, markers=[str(p) for p in markers])
        target = self._target(spec)
        if target is None:
            return Result(spec.id, "missing", "no target configured", spec.source)
        if spec.kind == "gdrive_file":
            ready = target.is_file() and target.stat().st_size > 0
            markers = [str(target)]
        else:
            ready = all((target / marker).exists() for marker in spec.markers)
            markers = [str(target / marker) for marker in spec.markers]
        return Result(spec.id, "present" if ready else "missing", str(target), spec.source, markers)

    def _record(self, spec: AssetSpec, result: Result, *, revision: str | None = None, file: Path | None = None) -> None:
        entry: dict[str, Any] = {
            "source": spec.source,
            "revision": revision or spec.revision,
            "installed_at": int(time.time()),
            "markers": result.markers,
        }
        if file and file.is_file():
            entry.update({"size": file.stat().st_size, "sha256": _sha256(file)})
        self.state.setdefault("assets", {})[spec.id] = entry
        self._save_state()

    def _download_gdrive(self, drive_id: str, output: Path) -> None:
        import gdown
        output.parent.mkdir(parents=True, exist_ok=True)
        result = gdown.download(id=drive_id, output=str(output), quiet=True, fuzzy=True)
        if not result or not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Google Drive download failed for {drive_id}")

    def _atomic_directory(self, source: Path, target: Path, markers: tuple[str, ...]) -> None:
        if not all((source / marker).exists() for marker in markers):
            raise RuntimeError(f"download did not contain required marker(s): {', '.join(markers)}")
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = target.with_name(target.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            target.rename(backup)
        try:
            source.rename(target)
        except Exception:
            if backup.exists():
                backup.rename(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)

    def _install_hf(self, spec: AssetSpec) -> str:
        from huggingface_hub import hf_hub_download, snapshot_download
        if spec.kind == "hf_file":
            downloaded = [Path(hf_hub_download(repo_id=spec.source, filename=name, local_files_only=self.offline)) for name in spec.files]
            return downloaded[0].parent.name if downloaded else ""
        directory = Path(snapshot_download(
            repo_id=spec.source, local_files_only=self.offline,
            allow_patterns=list(spec.files) if spec.files else None,
        ))
        return directory.name

    def _install_humanml3d(self, spec: AssetSpec, stage: Path) -> None:
        from huggingface_hub import hf_hub_download
        archive = Path(hf_hub_download(
            repo_id=spec.source, repo_type="dataset", filename="HumanML3D.zip", local_files_only=self.offline,
        ))
        safe_extract_zip(archive, stage)
        nested = stage / "HumanML3D"
        if nested.is_dir():
            promoted = stage.with_name(stage.name + "-root")
            nested.rename(promoted)
            shutil.rmtree(stage)
            promoted.rename(stage)

    def _install_glove(self, target: Path) -> None:
        from ..eval.tracks.smpl_hml.prepare_humanml3d import download_glove
        if self.offline:
            raise RuntimeError("GloVe is not cached under the checkpoint root; offline mode cannot download it")
        download_glove(target)

    def install(self, spec: AssetSpec, *, dry_run: bool = False) -> Result:
        if spec.kind == "external":
            return Result(spec.id, "external", spec.description, spec.source)
        current = self.verify(spec)
        if current.status == "present" and not self.force:
            return Result(spec.id, "skipped", "already verified", spec.source, current.markers)
        if dry_run:
            return Result(spec.id, "planned", spec.description, spec.source, current.markers)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if spec.kind.startswith("hf_"):
                revision = self._install_hf(spec)
                result = self.verify(spec)
                if result.status != "present":
                    raise RuntimeError(result.detail)
                self._record(spec, result, revision=revision)
                result.status = "installed"
                return result
            target = self._target(spec)
            if target is None:
                raise RuntimeError("no target configured")
            if spec.kind == "glove":
                self._install_glove(target)
                result = self.verify(spec)
                if result.status != "present":
                    raise RuntimeError("GloVe download did not produce required files")
                self._record(spec, result)
                result.status = "installed"
                return result
            if self.offline:
                raise RuntimeError("asset is not installed and has no offline source")
            staging_parent = self.root / ".staging"
            staging_parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f"{spec.id}-", dir=staging_parent) as tmp:
                work = Path(tmp)
                if spec.kind == "gdrive_file":
                    downloaded = work / target.name
                    self._download_gdrive(spec.drive_id or "", downloaded)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(downloaded, target)
                    result = self.verify(spec)
                    self._record(spec, result, file=target)
                else:
                    stage = work / "payload"
                    stage.mkdir()
                    if spec.kind == "gdrive_archive":
                        archive = work / "payload.zip"
                        self._download_gdrive(spec.drive_id or "", archive)
                        safe_extract_zip(archive, stage)
                    elif spec.kind == "humanml3d":
                        self._install_humanml3d(spec, stage)
                    else:
                        raise RuntimeError(f"unsupported asset kind: {spec.kind}")
                    payload = _find_marker_root(stage, spec.markers)
                    if payload is None:
                        raise RuntimeError("download did not contain required marker(s): " + ", ".join(spec.markers))
                    # Rename a sibling directory, never a child of the temp dir.
                    final_stage = target.parent / (target.name + ".new")
                    if final_stage.exists():
                        shutil.rmtree(final_stage)
                    shutil.copytree(payload, final_stage)
                    self._atomic_directory(final_stage, target, spec.markers)
                    result = self.verify(spec)
                    self._record(spec, result)
            result.status = "installed"
            return result
        except Exception as exc:  # noqa: BLE001 - aggregate independent failures
            return Result(spec.id, "failed", str(exc), spec.source, current.markers)


def _emit(results: list[Result], json_path: str | None) -> None:
    payload = {"manifest_version": MANIFEST_VERSION, "results": [result.as_dict() for result in results]}
    if json_path is not None:
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        if json_path == "-":
            print(rendered)
        else:
            destination = Path(json_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered + "\n")
            print(f"Wrote {destination}")
        return
    width = max((len(result.id) for result in results), default=5)
    for result in results:
        print(f"{result.id:<{width}}  {result.status:<16}  {result.detail}")
    failures = sum(result.status in {"failed", "missing", "missing-external"} for result in results)
    print(f"\n{len(results) - failures} ready, {failures} missing or failed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "fetch", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--asset", action="append", default=[], metavar="ID", help="asset id (repeatable)")
        p.add_argument("--all", action="store_true", help="include optional public variants")
        p.add_argument("--root", type=Path, help="override SEMOCO_BASELINE_CKPT_ROOT")
        p.add_argument("--json", nargs="?", const="-", help="write JSON report (stdout when no path is given)")
        if name == "fetch":
            p.add_argument("--force", action="store_true", help="reinstall already verified assets")
            p.add_argument("--offline", action="store_true", help="only use cached/local files")
            p.add_argument("--dry-run", action="store_true", help="show planned work without writing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        selected = select_assets(args.asset, include_all=args.all)
    except ValueError as exc:
        print(f"assets: {exc}", file=sys.stderr)
        return 2
    if args.command == "list":
        results = [Result(asset.id, "optional" if asset.optional else "full-eval", asset.description, asset.source) for asset in selected]
        _emit(results, args.json)
        return 0
    installer = AssetInstaller(args.root, offline=getattr(args, "offline", False), force=getattr(args, "force", False))
    if args.command == "verify":
        results = [installer.verify(asset) for asset in selected]
        _emit(results, args.json)
        return 1 if any(result.status in {"missing", "missing-external", "failed"} for result in results) else 0
    # Third-party download helpers occasionally print progress to stdout.  JSON
    # reports must remain parseable, so route that incidental output to stderr.
    stream = contextlib.redirect_stdout(sys.stderr) if args.json is not None else contextlib.nullcontext()
    with stream:
        results = [installer.install(asset, dry_run=args.dry_run) for asset in selected]
    _emit(results, args.json)
    return 1 if any(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
