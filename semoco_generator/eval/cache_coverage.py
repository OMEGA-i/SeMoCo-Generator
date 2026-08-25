"""Read-only cache coverage inspection organized by track and model category.

This module never mutates anything on disk. It reads index files from the
packed v2 store (and optionally run-local artifact stores), parses cache keys
to extract model identity and track affiliation, then renders a report grouped
by track, model category (baseline vs ours), and scope — with every checkpoint
listed as its own row.

Coverage percentages are computed per **(model, dataset)** group: within each
group the checkpoint with the most cached records serves as the 100 % baseline.
Each track only shows the models registered for that track in
``models/registry.py``, so a model scored on one track does not appear as a
gap on another.

Usage::

    python -m semoco_generator.eval.cli cache coverage

Or programmatically::

    from semoco_generator.eval.cache_coverage import build_report, render_coverage
    report = build_report()
    print(render_coverage(report))
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cache import _DATASET_DEPENDENT_MODELS
from .cache_utils import discover_run_cache_v2_roots, fmt_bytes
from .model_plan import SEMOCO_MODEL_IDS
from .models.registry import HUMANML3D_MODELS, MODEL_SCHEMAS, SOMA_TMR_MODELS
from .sharded_cache_store import ShardedCacheStore

# ---------------------------------------------------------------------------
# Track / model constants
# ---------------------------------------------------------------------------
# Derived from the registry rather than copied, so registering an external
# baseline (see models/registry.py) makes it visible to coverage reporting
# without a second edit here.

_KNOWN_MODELS: frozenset = frozenset(MODEL_SCHEMAS)

_SEMOCO_MODELS: frozenset = frozenset(SEMOCO_MODEL_IDS) & _KNOWN_MODELS

# Only these models include dataset in their native cache key.  The key parser
# uses this to disambiguate the baseline format (native/<model_id>/...) from
# the Semoco format (native/<dataset>/<model_id>/...).
_DATASET_DEPENDENT: frozenset = _DATASET_DEPENDENT_MODELS

_HML_MODELS: frozenset = frozenset(HUMANML3D_MODELS)
_TMR_MODELS: frozenset = frozenset(SOMA_TMR_MODELS)

_TRACK_MODELS: dict[str, frozenset] = {"smpl_hml": _HML_MODELS, "soma_tmr": _TMR_MODELS}

# Scope to track mapping.
_SCOPE_TRACK_MAP: dict[str, str] = {
    "native":        "shared",
    "hml_gt_motion": "smpl_hml",
    "hml_gt_text":   "smpl_hml",
    "tmr_gt_joints": "soma_tmr",
    "tmr_gt_motion": "soma_tmr",
    "tmr_text":      "soma_tmr",
    "converted":     "shared",
    "gen_emb":       "shared",
}

_TRACK_LABEL: dict[str, str] = {
    "smpl_hml": "smpl_hml (HumanML3D)",
    "soma_tmr": "soma_tmr (SOMA/TMR)",
}


# ---------------------------------------------------------------------------
# Key parsers
# ---------------------------------------------------------------------------

def _parse_native_key(key: str) -> dict[str, str] | None:
    """Extract *model_id*, *dataset*, *weight_sig*, *clip_id* from a native key.

    Baseline format::

        native/<model_id>/<weight_sig>/<clip_id>_s<seed>_cfg<cfg>_<ver>.npz

    Semoco (dataset-dependent) format::

        native/<dataset>/semoco/<weight_sig>/<clip_id>_s<seed>_cfg<cfg>_<ver>.npz
    """
    parts = key.split("/")
    if len(parts) < 3 or parts[0] != "native":
        return None
    # Baseline: parts[1] is a known model_id
    if parts[1] in _KNOWN_MODELS:
        return {
            "model_id": parts[1],
            "weight_sig": parts[2] if len(parts) > 2 else "",
            "dataset": "",
            "clip_id": parts[3].rsplit("_s", 1)[0] if len(parts) > 3 else "",
        }
    # Semoco: parts[1] is dataset, parts[2] is model_id.
    # Only _DATASET_DEPENDENT models use this format.
    if len(parts) >= 4 and parts[2] in _DATASET_DEPENDENT:
        return {
            "model_id": parts[2],
            "weight_sig": parts[3],
            "dataset": parts[1],
            "clip_id": parts[4].rsplit("_s", 1)[0] if len(parts) > 4 else "",
        }
    return None


def _parse_converted_key(key: str) -> dict[str, str] | None:
    parts = key.split(":")
    if len(parts) >= 3 and parts[0] == "converted":
        return {"model_id": parts[2]}
    return None


def _parse_gen_emb_key(key: str) -> dict[str, str] | None:
    parts = key.split(":")
    if len(parts) >= 5 and parts[0] == "gen_emb":
        return {"track": parts[1], "model_id": parts[4]}
    return None


_KEY_PARSERS = {
    "native": _parse_native_key,
    "converted": _parse_converted_key,
    "gen_emb": _parse_gen_emb_key,
}

_SCOPES_WITH_MODEL_ID = frozenset({"native", "converted", "gen_emb"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CkptBreakdown:
    """One checkpoint's stats within a (model, dataset) group."""
    weight_sig: str
    records: int
    bytes_total: int
    pct_of_max: float | None = None   # None when it IS the max (100 % baseline)


@dataclass
class DatasetGroup:
    """All checkpoints for one (model, dataset) combination."""
    dataset: str                      # "" for baselines (no dataset in key)
    max_records: int
    total_records: int
    ckpts: list[CkptBreakdown] = field(default_factory=list)
    unique_clips: int = 0


@dataclass
class ModelBreakdown:
    """Records in one scope attributed to one model, grouped by dataset."""
    model_id: str
    model_type: str                   # "baseline" | "ours"
    records: int = 0
    bytes_total: int = 0
    dataset_groups: list[DatasetGroup] = field(default_factory=list)


@dataclass
class ScopeSummary:
    """Records in one scope, broken down by model (if applicable)."""
    scope: str
    track: str
    total_records: int = 0
    total_bytes: int = 0
    corrupt: int = 0
    missing_packs: int = 0
    buckets: int = 0
    protocol_versions: dict[str, int] = field(default_factory=dict)
    by_model: dict[str, ModelBreakdown] = field(default_factory=dict)


@dataclass
class TrackSummary:
    track: str
    label: str
    scopes: dict[str, ScopeSummary] = field(default_factory=dict)


@dataclass
class CoverageReport:
    generated_at: str
    cache_root: str
    tracks: dict[str, TrackSummary] = field(default_factory=dict)
    run_local_roots: list[str] = field(default_factory=list)
    run_local_tracks: dict[str, TrackSummary] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_type(model_id: str) -> str:
    return "ours" if model_id in _SEMOCO_MODELS else "baseline"



def _short_sig(weight_sig: str, model_id: str) -> str:
    """Strip the redundant model_id prefix from a weight signature."""
    prefix = f"{model_id}_"
    if weight_sig.startswith(prefix):
        return weight_sig[len(prefix):]
    return weight_sig


def _pct_str(pct: float | None) -> str:
    """Compact percentage / max marker."""
    if pct is None:
        return "  100% ✓"
    if pct >= 99.95:
        return "  100%"
    return f"{pct:6.1f}%"


# ---------------------------------------------------------------------------
# Core inspection
# ---------------------------------------------------------------------------

def _inspect_store(
    store: ShardedCacheStore,
    *,
    parse_keys: bool = True,
) -> dict[str, ScopeSummary]:
    """Audit every scope in *store* and add model-level breakdowns where possible."""
    result: dict[str, ScopeSummary] = {}
    scopes = store.scopes()
    if not scopes:
        return result

    for scope in sorted(scopes):
        audit = store.audit(scope)
        track = _SCOPE_TRACK_MAP.get(scope, "unknown")
        ss = ScopeSummary(
            scope=scope,
            track=track,
            total_records=audit.records,
            total_bytes=audit.bytes,
            corrupt=audit.corrupt,
            missing_packs=audit.missing_packs,
            buckets=audit.buckets,
            protocol_versions=dict(audit.protocol_versions),
        )

        if parse_keys and scope in _SCOPES_WITH_MODEL_ID:
            parser = _KEY_PARSERS.get(scope)
            if parser is not None:
                entries = store.enumerate_scope(scope)

                # (model_id, dataset) -> { weight_sig -> {records, bytes, clips} }
                groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(
                    lambda: defaultdict(lambda: {"records": 0, "bytes": 0, "clips": set()})
                )

                for entry in entries:
                    parsed = parser(entry.get("key", ""))
                    if parsed is None:
                        continue
                    mid = parsed.get("model_id", "unknown")
                    ds = parsed.get("dataset", "")
                    ws = parsed.get("weight_sig", "unknown")
                    cid = parsed.get("clip_id", "")
                    g = groups[(mid, ds)][ws]
                    g["records"] += 1
                    g["bytes"] += int(entry.get("nbytes", 0))
                    if cid:
                        g["clips"].add(cid)

                model_data: dict[str, dict] = defaultdict(lambda: {
                    "records": 0, "bytes": 0, "groups": [],
                })

                for (mid, ds), ckpt_map in groups.items():
                    ckpt_list: list[CkptBreakdown] = []
                    max_rec = 0
                    total_rec = 0
                    all_clips: set[str] = set()
                    for ws, info in ckpt_map.items():
                        ckpt_list.append(CkptBreakdown(
                            weight_sig=ws,
                            records=info["records"],
                            bytes_total=info["bytes"],
                        ))
                        max_rec = max(max_rec, info["records"])
                        total_rec += info["records"]
                        all_clips |= info["clips"]

                    ckpt_list.sort(key=lambda c: -c.records)

                    for c in ckpt_list:
                        if len(ckpt_list) > 1 and c.records < max_rec:
                            c.pct_of_max = 100.0 * c.records / max_rec if max_rec > 0 else None

                    dg = DatasetGroup(
                        dataset=ds,
                        max_records=max_rec,
                        total_records=total_rec,
                        ckpts=ckpt_list,
                        unique_clips=len(all_clips),
                    )
                    model_data[mid]["groups"].append(dg)
                    model_data[mid]["records"] += total_rec
                    model_data[mid]["bytes"] += sum(c.bytes_total for c in ckpt_list)

                for mid, md in sorted(model_data.items()):
                    md["groups"].sort(key=lambda g: (g.dataset == "", g.dataset))
                    ss.by_model[mid] = ModelBreakdown(
                        model_id=mid,
                        model_type=_model_type(mid),
                        records=md["records"],
                        bytes_total=md["bytes"],
                        dataset_groups=md["groups"],
                    )
        result[scope] = ss
    return result


def _distribute_to_tracks(
    scopes: dict[str, ScopeSummary],
) -> dict[str, TrackSummary]:
    """Group per-scope summaries into track-level summaries.

    Shared scopes (native, converted, gen_emb) are copied to every track,
    but *by_model* is filtered so each track only sees its own models.
    """
    tracks: dict[str, TrackSummary] = {}
    for track_key, track_label in _TRACK_LABEL.items():
        tracks[track_key] = TrackSummary(track=track_key, label=track_label)

    for scope, ss in scopes.items():
        if ss.track == "shared":
            for track_key in ("smpl_hml", "soma_tmr"):
                if track_key not in tracks:
                    continue
                track_models = _TRACK_MODELS.get(track_key, set())
                filtered = ScopeSummary(
                    scope=ss.scope, track=ss.track,
                    total_records=ss.total_records, total_bytes=ss.total_bytes,
                    corrupt=ss.corrupt, missing_packs=ss.missing_packs,
                    buckets=ss.buckets, protocol_versions=dict(ss.protocol_versions),
                    by_model={
                        mid: mb for mid, mb in ss.by_model.items()
                        if mid in track_models
                    },
                )
                tracks[track_key].scopes[scope] = filtered
        elif ss.track in tracks:
            tracks[ss.track].scopes[scope] = ss
    return tracks


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    *,
    v2_root: str | Path | None = None,
    run_artifact_roots: list[str | Path] | None = None,
    runs_root: str | Path | None = None,
) -> CoverageReport:
    """Build a coverage report from the durable v2 store and optional run-local stores.

    Args:
        v2_root: Path to the durable v2 cache root (``<cache_root>/v2``).
            Defaults to ``$SEMOCO_EVAL_CACHE_ROOT/v2``.
        run_artifact_roots: Explicit run-local ``cache_v2`` roots to inspect.
        runs_root: ``runs/eval`` directory to auto-discover run-local caches.
    """
    from .cache import v2_root as default_v2_root

    notes: list[str] = []

    root = Path(v2_root) if v2_root is not None else default_v2_root()
    if not root.is_dir():
        notes.append(f"v2 cache root does not exist: {root}")

    report = CoverageReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        cache_root=str(root),
        notes=notes,
    )

    if root.is_dir():
        store = ShardedCacheStore(root, num_buckets=32)
        scopes = _inspect_store(store)
        report.tracks = _distribute_to_tracks(scopes)
    else:
        for track_key, track_label in _TRACK_LABEL.items():
            report.tracks[track_key] = TrackSummary(track=track_key, label=track_label)

    # Run-local caches
    run_roots: list[Path] = [Path(p) for p in (run_artifact_roots or [])]
    if runs_root is not None:
        run_roots += discover_run_cache_v2_roots(Path(runs_root))

    for rr in run_roots:
        if not rr.is_dir():
            report.run_local_roots.append(f"{rr} (missing)")
            continue
        report.run_local_roots.append(str(rr))
        store = ShardedCacheStore(rr, num_buckets=16)
        scopes = _inspect_store(store)
        rl_tracks = _distribute_to_tracks(scopes)
        for tk, ts in rl_tracks.items():
            if tk not in report.run_local_tracks:
                report.run_local_tracks[tk] = TrackSummary(track=tk, label=_TRACK_LABEL.get(tk, tk))
            for sn, ss in ts.scopes.items():
                report.run_local_tracks[tk].scopes[sn] = ss

    return report


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------

def _render_model(mb: ModelBreakdown, *, indent: int = 6) -> list[str]:
    """Flat per-ckpt table: every checkpoint is one row."""
    prefix = " " * indent
    lines: list[str] = []
    type_tag = "ours" if mb.model_type == "ours" else "baseline"

    n_ckpts = sum(len(dg.ckpts) for dg in mb.dataset_groups)
    lines.append(
        f"{prefix}{mb.model_id:<22} {type_tag:>8}  "
        f"{mb.records:>8} records  {fmt_bytes(mb.bytes_total):>8}  "
        f"({len(mb.dataset_groups)} datasets, {n_ckpts} ckpts)"
    )

    has_dataset = any(dg.dataset for dg in mb.dataset_groups)

    if has_dataset:
        # Semoco: ckpt | dataset | records | bytes | complete%
        lines.append(
            f"{prefix}  {'ckpt':<54} {'dataset':<42} {'records':>8}  {'bytes':>8}  compl"
        )
        lines.append(f"{prefix}  {'-'*54} {'-'*42} {'-'*8}  {'-'*8}  -----")
        for dg in mb.dataset_groups:
            ds = dg.dataset or "(no dataset)"
            for cb in dg.ckpts:
                pct = _pct_str(cb.pct_of_max)
                lines.append(
                    f"{prefix}  {_short_sig(cb.weight_sig, mb.model_id):<54} "
                    f"{ds:<42} {cb.records:>8}  {fmt_bytes(cb.bytes_total):>8}  {pct}"
                )
    else:
        # Baseline (no dataset in key): ckpt | records | bytes | complete%
        lines.append(
            f"{prefix}  {'ckpt':<54} {'records':>8}  {'bytes':>8}  compl"
        )
        lines.append(f"{prefix}  {'-'*54} {'-'*8}  {'-'*8}  -----")
        for dg in mb.dataset_groups:
            for cb in dg.ckpts:
                pct = _pct_str(cb.pct_of_max)
                lines.append(
                    f"{prefix}  {_short_sig(cb.weight_sig, mb.model_id):<54} "
                    f"{cb.records:>8}  {fmt_bytes(cb.bytes_total):>8}  {pct}"
                )
    return lines


def _render_scope(ss: ScopeSummary, *, indent: int = 4) -> list[str]:
    prefix = " " * indent
    lines: list[str] = []
    pv_str = ", ".join(f"{k}={v}" for k, v in sorted(ss.protocol_versions.items())) or "n/a"
    flags = ""
    if ss.corrupt or ss.missing_packs:
        flags = f"  [!] corrupt={ss.corrupt} missing_packs={ss.missing_packs}"

    gt_hint = ""
    if not ss.by_model and ss.total_records > 0:
        gt_hint = "  [complete]"

    lines.append(
        f"{prefix}{ss.scope:<18} {ss.total_records:>8} records  "
        f"{fmt_bytes(ss.total_bytes):>8}  buckets={ss.buckets}  "
        f"pv=({pv_str}){flags}{gt_hint}"
    )

    if ss.by_model:
        ours = [(mid, mb) for mid, mb in ss.by_model.items() if mb.model_type == "ours"]
        baselines = [(mid, mb) for mid, mb in ss.by_model.items() if mb.model_type == "baseline"]
        unknown = [(mid, mb) for mid, mb in ss.by_model.items() if mb.model_type not in ("ours", "baseline")]
        if ours:
            lines.append(f"{prefix}  -- Semocos --")
            for _, mb in ours:
                lines.extend(_render_model(mb, indent=indent + 2))
        if baselines:
            lines.append(f"{prefix}  -- Baselines --")
            for _, mb in baselines:
                lines.extend(_render_model(mb, indent=indent + 2))
        if unknown:
            lines.append(f"{prefix}  -- Other --")
            for _, mb in unknown:
                lines.extend(_render_model(mb, indent=indent + 2))
    return lines


def render_coverage(report: CoverageReport) -> str:
    lines: list[str] = []
    lines.append(f"=== Cache Coverage Report @ {report.generated_at} ===")
    lines.append(f"cache_root: {report.cache_root}")
    lines.append("")

    lines.append("[Durable Cache (v2) -- by Track]")
    lines.append("")
    for track_key in ("smpl_hml", "soma_tmr"):
        ts = report.tracks.get(track_key)
        if ts is None:
            continue
        lines.append(f"  Track: {ts.label}")
        if not ts.scopes:
            lines.append("    (no cached data)")
            lines.append("")
            continue
        for scope_name, ss in sorted(ts.scopes.items()):
            lines.extend(_render_scope(ss, indent=4))
        lines.append("")

    if report.run_local_roots:
        lines.append("[Run-Local Caches]")
        lines.append("")
        for rlr in report.run_local_roots:
            lines.append(f"  root: {rlr}")
        lines.append("")
        for track_key in ("smpl_hml", "soma_tmr"):
            ts = report.run_local_tracks.get(track_key)
            if ts is None or not ts.scopes:
                continue
            lines.append(f"  Track: {ts.label}")
            for scope_name, ss in sorted(ts.scopes.items()):
                lines.extend(_render_scope(ss, indent=4))
            lines.append("")

    if report.notes:
        lines.append("[Notes]")
        for n in report.notes:
            lines.append(f"  - {n}")
        lines.append("")

    return "\n".join(lines)


def report_to_dict(report: CoverageReport) -> dict[str, Any]:
    """JSON-serializable dict for ``--json-out``."""
    return asdict(report)


__all__ = [
    "CkptBreakdown",
    "CoverageReport",
    "DatasetGroup",
    "ModelBreakdown",
    "ScopeSummary",
    "TrackSummary",
    "build_report",
    "render_coverage",
    "report_to_dict",
]
