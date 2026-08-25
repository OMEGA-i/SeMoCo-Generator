"""Prepare / validate a HumanML3D dataset root for the smpl_hml track.

Default writable locations::

    $MOTIONVERSE_DATA_ROOT/HumanML3D
    $MOTIONVERSE_DATA_ROOT/glove

Modes:
  validate  — report missing assets (no writes)
  extract   — extract archive into out-root
  flatten   — hardlink/copy sharded texts|new_joints (XX/id.*) → flat
  prepare   — extract + flatten + ensure glove + ensure new_joint_vecs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

from ....paths import datasets_root, glove_root, humanml3d_root
from .paths import count_humanml_assets
from .word_vectorizer import resolve_glove_root

DEFAULT_ARCHIVE = Path(
    os.environ.get("SEMOCO_HUMANML_ARCHIVE", datasets_root() / "humanml3d.tar.gz")
).expanduser()
DEFAULT_OUT = humanml3d_root()
DEFAULT_GLOVE = glove_root()
GLOVE_FILES = ("our_vab_data.npy", "our_vab_words.pkl", "our_vab_idx.pkl")
GLOVE_URLS = {
    "our_vab_data.npy": "https://raw.githubusercontent.com/EricGuo5513/text-to-motion/main/glove/our_vab_data.npy",
    "our_vab_words.pkl": "https://raw.githubusercontent.com/EricGuo5513/text-to-motion/main/glove/our_vab_words.pkl",
    "our_vab_idx.pkl": "https://raw.githubusercontent.com/EricGuo5513/text-to-motion/main/glove/our_vab_idx.pkl",
}
# HF/yonful-style trees store clips under two-digit shard dirs (e.g. texts/00/000000.txt).
_SHARD_DIRS = ("texts", "new_joints")


def _is_macos_junk(name: str) -> bool:
    base = Path(name).name
    return base.startswith("._") or "__MACOSX" in name or name.endswith(".DS_Store")


def _count_files(root: Path, pattern: str) -> int:
    """Count ``pattern`` files at top level, falling back to one shard level deep."""
    if not root.is_dir():
        return 0
    n = len(list(root.glob(pattern)))
    if n > 0:
        return n
    return len(list(root.glob(f"*/{pattern}")))


def validate_humanml_root(root: Path, *, glove_root: Path | None = None) -> dict:
    root = Path(root)
    required_files = [root / "test.txt", root / "Mean.npy", root / "Std.npy"]
    required_dirs = [root / "texts", root / "new_joint_vecs"]
    missing = [str(p) for p in required_files if not p.exists()]
    missing += [str(p) for p in required_dirs if not p.exists()]
    n_texts_flat, n_texts = count_humanml_assets(root, "texts")
    n_vecs_flat, n_vecs = count_humanml_assets(root, "new_joint_vecs")
    n_joints_flat, n_joints = count_humanml_assets(root, "new_joints")
    glove = resolve_glove_root([glove_root] if glove_root else None)
    notes: list[str] = []
    if n_texts > 0 and n_texts_flat == 0:
        notes.append("texts/ is sharded (XX/id.txt); loaders resolve shards automatically.")
    if n_joints > 0 and n_joints_flat == 0:
        notes.append("new_joints/ is sharded (XX/id.npy); regenerate writes flat new_joint_vecs.")
    if n_joints > 0 and n_vecs == 0:
        notes.append(
            "new_joints present but new_joint_vecs missing; regenerate or download official features."
        )
    if n_vecs == 0 and (root / "new_joint_vecs").is_dir():
        missing.append(f"{root / 'new_joint_vecs'} (empty)")
    if n_texts == 0 and (root / "texts").is_dir():
        missing.append(f"{root / 'texts'} (empty)")
    ready = not missing and glove is not None and n_vecs > 0 and n_texts > 0
    return {
        "root": str(root),
        "missing": missing,
        "n_texts": n_texts_flat,
        "n_texts_including_shards": n_texts,
        "n_new_joint_vecs": n_vecs_flat if n_vecs_flat else n_vecs,
        "n_new_joints": n_joints_flat,
        "n_new_joints_including_shards": n_joints,
        "glove_root": str(glove) if glove else None,
        "glove_ready": glove is not None,
        "ready": ready,
        "notes": notes,
    }


def flatten_shards(out_root: Path, *, dirs: tuple[str, ...] = _SHARD_DIRS) -> dict:
    """Hardlink (or copy) ``dir/XX/id.ext`` → ``dir/id.ext`` for official flat layout."""
    out_root = Path(out_root)
    summary: dict = {"root": str(out_root), "dirs": {}}
    for name in dirs:
        parent = out_root / name
        linked = 0
        copied = 0
        skipped = 0
        if not parent.is_dir():
            summary["dirs"][name] = {"linked": 0, "copied": 0, "skipped": 0, "missing": True}
            continue
        # Walk shard dirs explicitly (faster / clearer than recursive glob on huge trees).
        for shard in sorted(parent.iterdir()):
            if not shard.is_dir() or len(shard.name) != 2:
                continue
            for src in shard.iterdir():
                if not src.is_file() or _is_macos_junk(src.name):
                    continue
                dest = parent / src.name
                if dest.exists():
                    skipped += 1
                    continue
                try:
                    os.link(src, dest)
                    linked += 1
                except OSError:
                    shutil.copy2(src, dest)
                    copied += 1
        n_flat = sum(1 for p in parent.iterdir() if p.is_file())
        summary["dirs"][name] = {
            "linked": linked,
            "copied": copied,
            "skipped": skipped,
            "n_flat": n_flat,
        }
        print(
            f"[prepare-humanml3d] flatten {name}: linked={linked} copied={copied} "
            f"skipped={skipped} n_flat={n_flat}",
            flush=True,
        )
    return summary


def extract_archive(archive: Path, out_root: Path) -> dict:
    archive = Path(archive)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    extracted = 0
    skipped = 0
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.name.endswith(".tar.gz"):
        opener = tarfile.open(archive, "r:gz")
        with opener as tf:
            for m in tf:
                if not m.isfile() or _is_macos_junk(m.name):
                    skipped += 1
                    continue
                # Strip leading HumanML3D/ if present.
                parts = Path(m.name).parts
                if parts and parts[0] == "HumanML3D":
                    rel = Path(*parts[1:]) if len(parts) > 1 else Path()
                else:
                    rel = Path(*parts)
                if not str(rel):
                    continue
                dest = out_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    skipped += 1
                    continue
                src = tf.extractfile(m)
                if src is None:
                    skipped += 1
                    continue
                with dest.open("wb") as f:
                    shutil.copyfileobj(src, f)
                extracted += 1
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.endswith("/") or _is_macos_junk(name):
                    skipped += 1
                    continue
                parts = Path(name).parts
                if parts and parts[0] == "HumanML3D":
                    rel = Path(*parts[1:]) if len(parts) > 1 else Path()
                else:
                    rel = Path(*parts)
                if not str(rel):
                    continue
                dest = out_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    skipped += 1
                    continue
                with zf.open(name) as src, dest.open("wb") as f:
                    shutil.copyfileobj(src, f)
                extracted += 1
    else:
        raise ValueError(f"unsupported archive type: {archive}")
    return {"archive": str(archive), "out_root": str(out_root), "extracted": extracted, "skipped": skipped}


def download_glove(glove_root: Path) -> dict:
    import urllib.request

    glove_root = Path(glove_root)
    glove_root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name, url in GLOVE_URLS.items():
        dest = glove_root / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        print(f"[prepare-humanml3d] downloading {name} ...", flush=True)
        urllib.request.urlretrieve(url, dest)
        downloaded.append(name)
    return {
        "glove_root": str(glove_root),
        "downloaded": downloaded,
        "ready": resolve_glove_root([glove_root]) is not None,
    }


def try_download_humanml3d_zip(dest_dir: Path) -> Path | None:
    """Download official-style HumanML3D.zip from ungated HF if available."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "HumanML3D.zip"
    if zip_path.is_file() and zip_path.stat().st_size > 1_000_000:
        return zip_path
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(
            repo_id="jbs99/humanml3d",
            repo_type="dataset",
            filename="HumanML3D.zip",
            local_dir=str(dest_dir),
        )
        return Path(p)
    except Exception as exc:  # noqa: BLE001
        print(f"[prepare-humanml3d] HF HumanML3D.zip download failed: {exc}", flush=True)
        return None


def ensure_new_joint_vecs(
    out_root: Path,
    *,
    allow_regenerate: bool = True,
    limit: int | None = None,
) -> dict:
    """Ensure ``new_joint_vecs`` exists; optionally regenerate from ``new_joints``."""
    out_root = Path(out_root)
    vec_dir = out_root / "new_joint_vecs"
    joints_dir = out_root / "new_joints"
    n_vecs = len(list(vec_dir.glob("*.npy"))) if vec_dir.is_dir() else 0
    if n_vecs > 0 and limit is None:
        return {"status": "present", "n_new_joint_vecs": n_vecs, "source": "existing"}
    if not allow_regenerate:
        return {"status": "missing", "n_new_joint_vecs": 0, "source": None}
    if not joints_dir.is_dir():
        return {
            "status": "missing",
            "n_new_joint_vecs": 0,
            "source": None,
            "error": "no new_joints/ to regenerate from",
        }
    from .regenerate_joint_vecs import regenerate_from_new_joints

    stats = regenerate_from_new_joints(joints_dir, vec_dir, limit=limit)
    stats["status"] = "regenerated"
    stats["source"] = "new_joints_process_file"
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare / validate HumanML3D assets")
    p.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    p.add_argument("--zip", default=None, help="Optional HumanML3D.zip override")
    p.add_argument("--out-root", default=str(DEFAULT_OUT))
    p.add_argument("--glove-root", default=str(DEFAULT_GLOVE))
    p.add_argument(
        "--mode",
        choices=("validate", "extract", "flatten", "prepare"),
        default="validate",
    )
    p.add_argument("--download-zip", action="store_true", help="Try HF jbs99/humanml3d HumanML3D.zip")
    p.add_argument("--no-regenerate", action="store_true")
    p.add_argument("--regen-limit", type=int, default=None, help="Cap joint_vec regeneration")
    p.add_argument("--out", default="runs/eval/readiness/humanml3d_assets.json")
    args = p.parse_args()

    out_root = Path(args.out_root)
    glove_root = Path(args.glove_root)
    report: dict = {"mode": args.mode}

    if args.mode in {"extract", "prepare"}:
        zip_path = Path(args.zip) if args.zip else None
        if args.download_zip and (zip_path is None or not zip_path.is_file()):
            zip_path = try_download_humanml3d_zip(out_root.parent / "_downloads") or zip_path
        if zip_path and zip_path.is_file():
            report["extract_zip"] = extract_archive(zip_path, out_root)
        elif Path(args.archive).is_file():
            report["extract_tar"] = extract_archive(Path(args.archive), out_root)
        else:
            report["extract_error"] = "no archive/zip available"

    if args.mode in {"flatten", "prepare"}:
        report["flatten"] = flatten_shards(out_root)

    if args.mode == "prepare":
        report["glove"] = download_glove(glove_root)
        report["new_joint_vecs"] = ensure_new_joint_vecs(
            out_root,
            allow_regenerate=not args.no_regenerate,
            limit=args.regen_limit,
        )

    report["validate"] = validate_humanml_root(out_root, glove_root=glove_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if args.mode != "validate" and not report["validate"]["ready"]:
        # flatten-only may still lack joint_vecs; prepare must be ready.
        if args.mode == "prepare":
            raise SystemExit(1)
        if args.mode == "flatten" and report["validate"]["n_texts"] == 0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
