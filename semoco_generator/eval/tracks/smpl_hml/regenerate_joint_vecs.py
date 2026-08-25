"""Regenerate HumanML3D ``new_joint_vecs`` from ``new_joints``.

Official ``new_joints`` are ``recover_from_ric(new_joint_vecs)`` (already aligned).
We therefore extract 263-D features with ``already_aligned=True`` — skipping
uniform-skeleton / floor / face-Z+ so we do not double-normalize.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .vendor.motion_process import JOINTS_NUM, process_file


def _collect_joint_paths(joints_dir: Path) -> list[Path]:
    """Prefer flat ``*.npy``; otherwise gather one-level shard paths ``XX/*.npy``."""
    flat = sorted(p for p in joints_dir.glob("*.npy") if p.is_file())
    if flat:
        return flat
    # Deduplicate by basename if shards somehow overlap.
    by_name: dict[str, Path] = {}
    for p in sorted(joints_dir.glob("*/*.npy")):
        if p.is_file() and len(p.parent.name) == 2:
            by_name.setdefault(p.name, p)
    return sorted(by_name.values(), key=lambda p: p.name)


def regenerate_from_new_joints(
    joints_dir: str | Path,
    vec_dir: str | Path,
    *,
    feet_thre: float = 0.002,
    overwrite: bool = False,
    limit: int | None = None,
    ids: list[str] | None = None,
    workers: int = 1,
) -> dict:
    joints_dir = Path(joints_dir)
    vec_dir = Path(vec_dir)
    if not joints_dir.is_dir():
        raise FileNotFoundError(joints_dir)
    vec_dir.mkdir(parents=True, exist_ok=True)

    if ids is not None:
        from .paths import resolve_humanml_asset

        root = joints_dir.parent
        paths = []
        for mid in ids:
            p = resolve_humanml_asset(root, "new_joints", mid)
            if p is not None:
                paths.append(p)
        paths = sorted(paths, key=lambda p: p.name)
    else:
        paths = _collect_joint_paths(joints_dir)
    if limit is not None:
        paths = paths[: int(limit)]

    ok = 0
    skipped = 0
    failed: list[dict] = []

    def _one(src: Path) -> tuple[str, str | None]:
        dest = vec_dir / src.name
        if dest.is_file() and not overwrite:
            return "skipped", None
        try:
            joints = np.load(src).astype(np.float32)
            if joints.ndim != 3:
                raise ValueError(f"expected [T,J,3], got {joints.shape}")
            if joints.shape[1] < JOINTS_NUM:
                raise ValueError(f"need >= {JOINTS_NUM} joints, got {joints.shape}")
            feats = process_file(joints, feet_thre, already_aligned=True)
            np.save(dest, feats.astype(np.float32))
            return "ok", None
        except Exception as exc:  # noqa: BLE001
            return "failed", f"{src.name}: {exc}"

    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(_regen_worker, str(src), str(vec_dir), feet_thre, overwrite): src for src in paths}
            for i, fut in enumerate(as_completed(futs), 1):
                status, err = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed.append({"file": futs[fut].name, "error": err or "unknown"})
                if i % 500 == 0:
                    print(
                        f"[regenerate-joint-vecs] {i}/{len(paths)} "
                        f"ok={ok} skipped={skipped} failed={len(failed)}",
                        flush=True,
                    )
    else:
        for i, src in enumerate(paths):
            status, err = _one(src)
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed.append({"file": src.name, "error": err or "unknown"})
            if (i + 1) % 500 == 0:
                print(
                    f"[regenerate-joint-vecs] {i + 1}/{len(paths)} "
                    f"ok={ok} skipped={skipped} failed={len(failed)}",
                    flush=True,
                )

    report = {
        "joints_dir": str(joints_dir),
        "vec_dir": str(vec_dir),
        "n_input": len(paths),
        "n_written": ok,
        "n_skipped": skipped,
        "n_failed": len(failed),
        "failed_sample": failed[:20],
        "n_new_joint_vecs": len(list(vec_dir.glob("*.npy"))),
    }
    return report


def _regen_worker(src: str, vec_dir: str, feet_thre: float, overwrite: bool) -> tuple[str, str | None]:
    src_p = Path(src)
    dest = Path(vec_dir) / src_p.name
    if dest.is_file() and not overwrite:
        return "skipped", None
    try:
        joints = np.load(src_p).astype(np.float32)
        if joints.ndim != 3:
            raise ValueError(f"expected [T,J,3], got {joints.shape}")
        if joints.shape[1] < JOINTS_NUM:
            raise ValueError(f"need >= {JOINTS_NUM} joints, got {joints.shape}")
        feats = process_file(joints, feet_thre, already_aligned=True)
        np.save(dest, feats.astype(np.float32))
        return "ok", None
    except Exception as exc:  # noqa: BLE001
        return "failed", str(exc)


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate new_joint_vecs from new_joints")
    p.add_argument("--joints-dir", required=True)
    p.add_argument("--vec-dir", required=True)
    p.add_argument("--feet-thre", type=float, default=0.002)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--split-file", default=None, help="Only regenerate IDs listed in this split txt")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=None, help="Optional JSON report path")
    args = p.parse_args()
    ids = None
    if args.split_file:
        ids = [ln.strip() for ln in Path(args.split_file).read_text().splitlines() if ln.strip()]
    report = regenerate_from_new_joints(
        args.joints_dir,
        args.vec_dir,
        feet_thre=args.feet_thre,
        overwrite=args.overwrite,
        limit=args.limit,
        ids=ids,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    if report["n_written"] == 0 and report["n_skipped"] == 0:
        raise SystemExit(1)
    if report["n_failed"] and report["n_written"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
