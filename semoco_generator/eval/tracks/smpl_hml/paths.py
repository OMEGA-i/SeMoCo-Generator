"""Path helpers for official flat or HF-sharded HumanML3D trees."""

from __future__ import annotations

from pathlib import Path


def resolve_humanml_asset(root: Path, kind: str, mid: str) -> Path | None:
    """Resolve ``texts/<id>.txt`` or ``new_joints|new_joint_vecs/<id>.npy``.

    Supports EricGuo flat layout and HF two-digit shard layout
    (``texts/00/000000.txt``). For mirrored ``M######`` IDs, text files often
    exist only under the unmirrored id — fall back accordingly for ``texts``.
    """
    root = Path(root)
    mid = str(mid).strip()
    if kind == "texts":
        name = f"{mid}.txt"
        parent = root / "texts"
    elif kind in {"new_joints", "new_joint_vecs"}:
        name = f"{mid}.npy"
        parent = root / kind
    else:
        raise ValueError(f"unknown HumanML asset kind: {kind!r}")

    def _try(one: str) -> Path | None:
        fname = f"{one}.txt" if kind == "texts" else f"{one}.npy"
        flat = parent / fname
        if flat.is_file():
            return flat
        if len(one) >= 2:
            sharded = parent / one[:2] / fname
            if sharded.is_file():
                return sharded
        return None

    hit = _try(mid)
    if hit is not None:
        return hit
    if kind == "texts" and mid.startswith("M") and len(mid) > 1:
        return _try(mid[1:])
    return None


def count_humanml_assets(root: Path, kind: str) -> tuple[int, int]:
    """Return ``(n_flat, n_available)`` for texts or npy dirs.

    ``n_available`` prefers the full sharded count when shards exist (partial
    flatten must not hide the complete tree).
    """
    parent = Path(root) / kind
    if not parent.is_dir():
        return 0, 0
    pattern = "*.txt" if kind == "texts" else "*.npy"
    n_flat = sum(1 for p in parent.glob(pattern) if p.is_file())
    n_shard = sum(1 for p in parent.glob(f"*/{pattern}") if p.is_file())
    if n_shard > 0:
        return n_flat, n_shard
    return n_flat, n_flat


__all__ = ["resolve_humanml_asset", "count_humanml_assets"]
