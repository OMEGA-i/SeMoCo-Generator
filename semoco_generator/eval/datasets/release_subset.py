"""Build reproducible evaluation subsets from the two supported sources.

* ``humanml3d``: official HumanML3D tree (``test.txt``/``texts``/``new_joint_vecs``)
* ``t2m_store``: our exported T2M code store (index/meta)

There is no release/parquet scan path; the two evaluation tracks read only
these two sources.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...local_uri import resolve_local_uri

SourceName = Literal["t2m_store", "humanml3d"]


@dataclass
class EvalClip:
    clip_id: str
    rec_id: str
    split: str
    caption: str
    source: SourceName
    release_id: str | None = None
    row: int | None = None
    code_len: int | None = None
    duration_s: float | None = None
    parquet_row_index: int | None = None
    subset: str | None = None  # source group from provenance_json.dataset
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _seeded_sample(items: list[Any], *, limit: int | None, seed: int) -> list[Any]:
    if limit is None or limit >= len(items):
        return list(items)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(items), size=int(limit), replace=False))
    return [items[int(i)] for i in idx]


def _npy_num_rows(path: Path) -> int | None:
    """Read only the ``.npy`` header for row count (no full mmap)."""
    try:
        with Path(path).open("rb") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            else:
                shape, _, _ = np.lib.format._read_array_header(f, version)
        if not shape:
            return None
        return int(shape[0])
    except Exception:  # noqa: BLE001
        return None


def build_subset_from_t2m_store(
    codes_root: str | Path,
    *,
    split: str = "test",
    rec_ids: list[str] | None = None,
    limit: int | None = None,
    seed: int = 0,
    caption_regex: str | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    subset_map: dict[str, str] | None = None,
) -> list[EvalClip]:
    """Read exported T2M index/meta and return stable clips with code_len/duration."""
    root = resolve_local_uri(codes_root)
    index_path = root / f"{split}.index.json"
    meta_path = root / f"{split}.meta.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing {index_path}")
    index = json.loads(index_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    token_rate = float(meta.get("token_rate") or meta.get("tokens_per_second") or 12.5)
    pat = re.compile(caption_regex) if caption_regex else None
    want = set(rec_ids) if rec_ids else None
    clips: list[EvalClip] = []
    for row, e in enumerate(index):
        rid = str(e.get("rec_id") or e.get("id") or f"row{row}")
        if want is not None and rid not in want:
            continue
        caption = str(e.get("caption") or e.get("text") or "")
        if pat is not None and not pat.search(caption):
            continue
        code_len = int(e.get("code_len") or 0)
        if min_tokens is not None and code_len < min_tokens:
            continue
        if max_tokens is not None and code_len > max_tokens:
            continue
        duration_s = float(code_len) / token_rate if code_len > 0 else None
        clips.append(
            EvalClip(
                clip_id=f"{split}:{rid}",
                rec_id=rid,
                split=split,
                caption=caption,
                source="t2m_store",
                release_id=meta.get("release_id"),
                row=row,
                code_len=code_len,
                duration_s=duration_s,
                subset=subset_map.get(rid) if subset_map else None,
                metadata={
                    "token_rate": token_rate,
                    "code_start": e.get("code_start"),
                    "tokenizer_checkpoint": meta.get("tokenizer_checkpoint"),
                },
            )
        )
    return _seeded_sample(clips, limit=limit, seed=seed)


def _parse_humanml_text_line_full(line: str) -> tuple[str, list[str], float, float]:
    """Parse ``caption # tokens # f_tag # to_tag`` from one HumanML text line."""
    parts = line.split("#")
    caption = parts[0].strip()
    tokens: list[str] = []
    if len(parts) > 1:
        tokens = [t for t in parts[1].strip().split() if t]
    f_tag = 0.0
    to_tag = 0.0
    if len(parts) > 3:
        try:
            f_tag = float(parts[2].strip() or 0.0)
        except ValueError:
            f_tag = 0.0
        try:
            to_tag = float(parts[3].strip() or 0.0)
        except ValueError:
            to_tag = 0.0
    if f_tag != f_tag or to_tag != to_tag or abs(f_tag) == float("inf") or abs(to_tag) == float("inf"):
        f_tag, to_tag = 0.0, 0.0
    return caption, tokens, f_tag, to_tag


def _unit_length_round(n_frames: int, *, unit_length: int = 4, max_len: int = 196) -> int:
    """Floor to a multiple of ``unit_length`` and clamp to HumanML max length."""
    n = max(unit_length, int(n_frames))
    n = (n // unit_length) * unit_length
    return min(max_len, max(unit_length, n))


def build_subset_from_humanml3d(
    data_root: str | Path,
    *,
    split: str = "test",
    limit: int | None = None,
    seed: int = 0,
    caption_index: int = 0,
    index_jsonl: str | Path | None = None,
    protocol: Literal["official_hml_eval", "legacy_full_test"] = "official_hml_eval",
    min_motion_len: int = 40,
    max_motion_len: int = 200,
    unit_length: int = 4,
) -> list[EvalClip]:
    """Build clips from an official HumanML3D-style tree.

    ``protocol``:
      - ``official_hml_eval``: expand nonzero ``f_tag/to_tag`` segments, filter
        ``min_motion_len <= len < max_motion_len``, store unit-length ``m_length``.
      - ``legacy_full_test``: one clip per motion ID (first caption), no filter.
    """
    root = resolve_local_uri(data_root)
    split_file = root / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"missing HumanML3D split file: {split_file}")
    ids = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]
    id_limit = limit
    if protocol == "official_hml_eval" and limit is not None:
        id_limit = min(len(ids), max(int(limit) * 4, int(limit)))
    if id_limit is not None and id_limit < len(ids):
        rng = np.random.default_rng(int(seed))
        pick = rng.choice(len(ids), size=int(id_limit), replace=False)
        ids = [ids[i] for i in sorted(pick.tolist())]

    index_map: dict[str, dict[str, Any]] = {}
    candidates: list[Path] = []
    if index_jsonl is not None:
        candidates.append(Path(index_jsonl))
    candidates.extend(
        [
            root / f".eval_index_{split}.jsonl",
            Path("runs/eval/readiness") / f"hml_{split}_index.jsonl",
        ]
    )
    for cand in candidates:
        if cand.is_file():
            for ln in cand.read_text().splitlines():
                if not ln.strip():
                    continue
                obj = json.loads(ln)
                index_map[str(obj["rec_id"])] = obj
            break

    from ..tracks.smpl_hml.paths import resolve_humanml_asset

    def _load_all_caption_rows(mid: str) -> tuple[list[tuple[str, list[str], float, float]], str | None]:
        text_path_p = resolve_humanml_asset(root, "texts", mid)
        if text_path_p is None or not text_path_p.is_file():
            return [], None
        lines = [ln.strip() for ln in text_path_p.read_text().splitlines() if ln.strip()]
        parsed: list[tuple[str, list[str], float, float]] = []
        for ln in lines:
            cap, toks, f_tag, to_tag = _parse_humanml_text_line_full(ln)
            if cap:
                parsed.append((cap, toks, f_tag, to_tag))
        return parsed, str(text_path_p)

    def _load_humanml_caption_tokens(mid: str, caption_index: int) -> tuple[str, list[str], str | None]:
        parsed, text_path = _load_all_caption_rows(mid)
        if not parsed:
            return "", [], text_path
        cap, toks, _, _ = parsed[min(caption_index, len(parsed) - 1)]
        return cap, toks, text_path

    clips: list[EvalClip] = []
    row_i = 0
    for mid in ids:
        vec_path_p = resolve_humanml_asset(root, "new_joint_vecs", mid)
        meta = index_map.get(mid)
        n_frames_full = None
        if meta is not None and meta.get("n_frames") is not None:
            n_frames_full = int(meta["n_frames"])
        elif vec_path_p is not None:
            n_frames_full = _npy_num_rows(vec_path_p)
        vec_path = (
            str(meta.get("vec_path"))
            if meta is not None and meta.get("vec_path")
            else (str(vec_path_p) if vec_path_p else str(root / "new_joint_vecs" / f"{mid}.npy"))
        )

        if protocol == "legacy_full_test":
            if meta is not None and (meta.get("caption") or meta.get("tokens")):
                caption = str(meta.get("caption") or "")
                tokens = list(meta.get("tokens") or [])
                text_path = meta.get("text_path")
                if not caption or not tokens:
                    cap2, toks2, tpath2 = _load_humanml_caption_tokens(mid, caption_index)
                    caption = caption or cap2
                    tokens = tokens or toks2
                    text_path = text_path or tpath2
            else:
                caption, tokens, text_path = _load_humanml_caption_tokens(mid, caption_index)
            if not caption or not tokens:
                continue
            duration_s = float(n_frames_full) / 20.0 if n_frames_full else None
            m_length = _unit_length_round(int(n_frames_full or 0)) if n_frames_full else None
            clips.append(
                EvalClip(
                    clip_id=f"hml:{split}:{mid}",
                    rec_id=mid,
                    split=split,
                    caption=caption,
                    source="humanml3d",
                    row=row_i,
                    duration_s=duration_s,
                    metadata={
                        "n_frames": n_frames_full,
                        "m_length": m_length,
                        "fps": 20.0,
                        "vec_path": vec_path,
                        "text_path": text_path,
                        "tokens": tokens,
                        "start_s": 0.0,
                        "end_s": 0.0,
                        "frame_start": 0,
                        "frame_end": n_frames_full,
                        "caption_index": caption_index,
                        "hml_protocol": protocol,
                    },
                )
            )
            row_i += 1
            continue

        rows, text_path = _load_all_caption_rows(mid)
        if not rows or n_frames_full is None:
            continue
        for cap_i, (caption, tokens, f_tag, to_tag) in enumerate(rows):
            if not caption or not tokens:
                continue
            if abs(f_tag) < 1e-6 and abs(to_tag) < 1e-6:
                frame_start, frame_end = 0, int(n_frames_full)
            else:
                frame_start = int(round(float(f_tag) * 20.0))
                frame_end = int(round(float(to_tag) * 20.0))
                frame_start = max(0, min(frame_start, int(n_frames_full)))
                frame_end = max(frame_start, min(frame_end, int(n_frames_full)))
            n_frames = int(frame_end - frame_start)
            if n_frames < int(min_motion_len) or n_frames >= int(max_motion_len):
                continue
            m_length = _unit_length_round(n_frames, unit_length=unit_length)
            clip_id = (
                f"hml:{split}:{mid}"
                if abs(f_tag) < 1e-6 and abs(to_tag) < 1e-6 and cap_i == 0
                else f"hml:{split}:{mid}:seg{cap_i}"
            )
            clips.append(
                EvalClip(
                    clip_id=clip_id,
                    rec_id=mid,
                    split=split,
                    caption=caption,
                    source="humanml3d",
                    row=row_i,
                    duration_s=float(m_length) / 20.0,
                    metadata={
                        "n_frames": n_frames,
                        "m_length": m_length,
                        "n_frames_full": n_frames_full,
                        "fps": 20.0,
                        "vec_path": vec_path,
                        "text_path": text_path,
                        "tokens": tokens,
                        "start_s": float(f_tag),
                        "end_s": float(to_tag),
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "caption_index": cap_i,
                        "hml_protocol": protocol,
                    },
                )
            )
            row_i += 1

    return _seeded_sample(clips, limit=limit, seed=seed)


def write_subset(subset: list[EvalClip], path: str | Path) -> str:
    """Write prompts.jsonl and return a content hash used in protocol_id."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(c.to_dict(), ensure_ascii=False, sort_keys=True) for c in subset]
    text = "\n".join(lines) + ("\n" if lines else "")
    out.write_text(text)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_subset(path: str | Path) -> list[EvalClip]:
    clips: list[EvalClip] = []
    for ln in Path(path).read_text().splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        clips.append(EvalClip(**obj))
    return clips


def load_subset_labels_from_release(
    parquet_dir: str | Path, split: str
) -> dict[str, str]:
    """Read ``provenance_json.dataset`` for every rec_id in a release split.

    Returns ``{rec_id: subset_label}``, e.g. ``"HumanML3D"``, ``"bones-seed"``,
    ``"mm_MotionGV"``.  Scans all parquet shards once (reads only ``rec_id`` +
    ``provenance_json`` columns).
    """
    import pyarrow.parquet as pq

    subset_map: dict[str, str] = {}
    shard_dir = Path(parquet_dir) / split
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        return subset_map

    for shard in shards:
        pf = pq.ParquetFile(shard, memory_map=True)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["rec_id", "provenance_json"])
            rec_ids = table.column("rec_id").to_pylist()
            provs = table.column("provenance_json").to_pylist()
            for rid, pjson in zip(rec_ids, provs):
                try:
                    prov = json.loads(pjson)
                    subset_map[rid] = str(prov.get("dataset", "")) or None
                except (json.JSONDecodeError, TypeError):
                    print(f"[subset] WARNING: corrupt provenance_json for rec_id={rid}, skipping", flush=True)
                    continue
    return subset_map


# ---------------------------------------------------------------------------
# Parquet directory resolution
# ---------------------------------------------------------------------------

def resolve_parquet_dir(
    codes_root: str | Path,
    split: str = "test",
    *,
    cli_value: str | None = None,
) -> Path:
    """Resolve the parquet release directory for a T2M code store.

    Priority:
    1. Explicit *cli_value* (from ``--parquet-dir``)
    2. ``$SEMOCO_EVAL_PARQUET_DIR`` env var
    3. ``{codes_root}/{split}.meta.json`` field ``"parquet_release_dir"``

    Raises:
        FileNotFoundError: if no parquet directory can be resolved.
    """
    import os as _os

    # 1. CLI arg
    if cli_value:
        from ...local_uri import resolve_local_uri as _resolve
        return _resolve(cli_value)

    # 2. Env var
    env = _os.environ.get("SEMOCO_EVAL_PARQUET_DIR")
    if env:
        return Path(env).expanduser().resolve()

    # 3. Code store metadata
    root = resolve_local_uri(str(codes_root))
    meta_path = root / f"{split}.meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        meta_dir = meta.get("parquet_release_dir")
        if meta_dir:
            from ...local_uri import resolve_local_uri as _resolve
            return _resolve(str(meta_dir))

    raise FileNotFoundError(
        f"Cannot determine parquet release directory for codes_root={root}.\n"
        f"  - Pass --parquet-dir <path>\n"
        f"  - Set $SEMOCO_EVAL_PARQUET_DIR\n"
        f"  - Add 'parquet_release_dir' to {meta_path}"
    )


__all__ = [
    "EvalClip",
    "build_subset_from_humanml3d",
    "build_subset_from_t2m_store",
    "load_subset",
    "load_subset_labels_from_release",
    "resolve_parquet_dir",
    "write_subset",
]
