"""Read-only audit of the eval cache (packed v2 stores + run artifacts).

This module never mutates anything on disk. It is the pipeline's "control
panel":

* packed-store record/byte counts per scope (``native``, ``hml_gt_motion``,
  ``converted``, ``gen_emb``, ...) via :class:`ShardedCacheStore.audit`
* stray small-file counts under the cache root — non-empty only means
  something *other* than the packed stores wrote there (old leftovers from
  before the v2 migration, or files placed by hand); a clean install reports
  zero here
* shard-log summaries: cache-reject counts, corrupt-sidecar counts, live
  text-encoder fallbacks, last progress line seen
* host RAM (total/available/used) and current process RSS
* GPU VRAM/utilization via ``nvidia-smi`` (no torch/CUDA import needed)
* empirical full-gallery / duplicate-aware metric memory estimates, derived
  from real cached embedding counts + dims (sampled from the packed GT
  stores) rather than guessed constants

Scanning is a single ``os.scandir`` pass per directory (no ``Path.rglob``
object churn) so it stays cheap even at 10^5-10^6 files.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .cache_utils import (
    LEGACY_DURABLE_FAMILIES,
    PACKED_CACHE_DIRNAME,
    discover_run_artifact_roots,
    discover_run_cache_v2_roots,
    discover_logs,
    fmt_bytes,
    packed_cache_root,
    require_durable_cache_root,
)
from .sharded_cache_store import CacheAudit, ShardedCacheStore

DATA_EXTS = (".npy", ".npz")
SIDECAR_SUFFIX = ".sha256"

# (upper_bound_bytes_exclusive, label) — last bucket catches everything larger.
_SIZE_BUCKETS: tuple[tuple[float, str], ...] = (
    (4 * 1024, "<4K"),
    (64 * 1024, "<64K"),
    (1024 * 1024, "<1M"),
    (16 * 1024 * 1024, "<16M"),
)

_REJECT_PATTERN = "cache rejected (no sidecar)"
_CORRUPT_PATTERN = "cache integrity failure"
_FALLBACK_PATTERN = "falling back to live"
_PROGRESS_MARKERS = ("native seed=", "shard ", "] subset protocol=")


def _bucket(size: int) -> str:
    for limit, label in _SIZE_BUCKETS:
        if size < limit:
            return label
    return ">=16M"


# ---------------------------------------------------------------------------
# Cache / run-artifact tree scanning
# ---------------------------------------------------------------------------
@dataclass
class ExtStats:
    count: int = 0
    bytes: int = 0
    with_sidecar: int = 0
    without_sidecar: int = 0


@dataclass
class FamilyStats:
    root: str
    exists: bool = False
    total_files: int = 0
    total_bytes: int = 0
    sidecar_files: int = 0
    by_ext: dict[str, ExtStats] = field(default_factory=dict)
    size_histogram: dict[str, int] = field(default_factory=dict)
    top_subdirs_by_bytes: list[tuple[str, int, int]] = field(default_factory=list)
    scan_seconds: float = 0.0


def scan_tree(root: Path) -> FamilyStats:
    """One bounded ``os.scandir`` walk of *root*. Read-only; stats + sidecar
    *existence* only (never opens/loads/hashes a data file)."""
    stats = FamilyStats(root=str(root), exists=root.is_dir())
    if not stats.exists:
        return stats
    t0 = time.time()
    top_bytes: dict[str, int] = {}
    top_files: dict[str, int] = {}
    stack: list[str] = [str(root)]
    while stack:
        d = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        names = {e.name for e in entries}
        for e in entries:
            try:
                is_dir = e.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                stack.append(e.path)
                continue
            name = e.name
            if name.endswith(SIDECAR_SUFFIX):
                stats.sidecar_files += 1
                continue
            ext = os.path.splitext(name)[1]
            try:
                size = e.stat(follow_symlinks=False).st_size
            except OSError:
                continue
            stats.total_files += 1
            stats.total_bytes += size
            es = stats.by_ext.setdefault(ext, ExtStats())
            es.count += 1
            es.bytes += size
            if ext in DATA_EXTS:
                if f"{name}{SIDECAR_SUFFIX}" in names:
                    es.with_sidecar += 1
                else:
                    es.without_sidecar += 1
            bucket = _bucket(size)
            stats.size_histogram[bucket] = stats.size_histogram.get(bucket, 0) + 1
            try:
                rel_parts = Path(d).relative_to(root).parts
            except ValueError:
                rel_parts = ()
            top_key = rel_parts[0] if rel_parts else "."
            top_bytes[top_key] = top_bytes.get(top_key, 0) + size
            top_files[top_key] = top_files.get(top_key, 0) + 1
    stats.top_subdirs_by_bytes = sorted(
        ((k, top_files[k], top_bytes[k]) for k in top_bytes), key=lambda t: -t[2]
    )[:15]
    stats.scan_seconds = time.time() - t0
    return stats


# ---------------------------------------------------------------------------
# Packed v2 store auditing
# ---------------------------------------------------------------------------
def audit_packed_store(root: Path, *, num_buckets: int = 16) -> dict[str, CacheAudit]:
    """Audit every scope of a :class:`ShardedCacheStore` rooted at *root*."""
    store = ShardedCacheStore(root, num_buckets=num_buckets)
    return {scope: store.audit(scope) for scope in store.scopes()}


# ---------------------------------------------------------------------------
# Shard log summaries
# ---------------------------------------------------------------------------
@dataclass
class LogStats:
    path: str
    size_bytes: int
    mtime: float
    lines: int = 0
    cache_rejected: int = 0
    cache_corrupt: int = 0
    live_encode_fallback: int = 0
    last_progress_line: str | None = None


def parse_log(path: Path) -> LogStats:
    st = path.stat()
    ls = LogStats(path=str(path), size_bytes=st.st_size, mtime=st.st_mtime)
    last_progress: str | None = None
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                ls.lines += 1
                if _REJECT_PATTERN in line:
                    ls.cache_rejected += 1
                if _CORRUPT_PATTERN in line:
                    ls.cache_corrupt += 1
                if _FALLBACK_PATTERN in line:
                    ls.live_encode_fallback += 1
                if any(marker in line for marker in _PROGRESS_MARKERS):
                    last_progress = line.strip()
    except OSError:
        pass
    ls.last_progress_line = last_progress
    return ls


# ---------------------------------------------------------------------------
# Host RAM / GPU
# ---------------------------------------------------------------------------
@dataclass
class HostRamStats:
    total_bytes: int
    available_bytes: int
    used_bytes: int
    percent: float
    process_rss_bytes: int


def host_ram_stats() -> HostRamStats:
    import psutil

    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    return HostRamStats(
        total_bytes=int(vm.total),
        available_bytes=int(vm.available),
        used_bytes=int(vm.used),
        percent=float(vm.percent),
        process_rss_bytes=int(proc.memory_info().rss),
    )


@dataclass
class GpuStats:
    index: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_pct: float


def gpu_stats(*, timeout: float = 5.0) -> list[GpuStats]:
    """Query GPU VRAM/utilization via ``nvidia-smi`` (no torch/CUDA import,
    so this stays cheap and safe to call even when workers hold GPU memory)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    stats: list[GpuStats] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, total, used, free, util = parts[:6]
        try:
            stats.append(
                GpuStats(
                    index=int(idx),
                    name=name,
                    memory_total_mb=float(total),
                    memory_used_mb=float(used),
                    memory_free_mb=float(free),
                    utilization_pct=float(util),
                )
            )
        except ValueError:
            continue
    return stats


# ---------------------------------------------------------------------------
# Empirical metric-memory estimates
# ---------------------------------------------------------------------------
@dataclass
class MemoryEstimate:
    label: str
    n: int
    d: int
    dtype_bytes: int
    full_gallery_bytes: int
    dup_aware_sim_bytes: int


def estimate_full_gallery(label: str, n: int, d: int, *, dtype_bytes: int = 4) -> MemoryEstimate:
    """Naive ``N x N x D`` broadcast (current ``r_precision_full_gallery``) plus
    the ``N x N`` float64 text-text similarity matrix (duplicate-aware retrieval)."""
    full = int(n) * int(n) * int(d) * int(dtype_bytes)
    dup = int(n) * int(n) * 8
    return MemoryEstimate(
        label=label, n=int(n), d=int(d), dtype_bytes=int(dtype_bytes),
        full_gallery_bytes=full, dup_aware_sim_bytes=dup,
    )


def _count_and_dim(dir_path: Path) -> tuple[int, int | None]:
    if not dir_path.is_dir():
        return 0, None
    count = 0
    dim: int | None = None
    with os.scandir(dir_path) as it:
        for e in it:
            if not e.name.endswith(".npy"):
                continue
            try:
                if not e.is_file():
                    continue
            except OSError:
                continue
            count += 1
            if dim is None:
                try:
                    arr = np.load(e.path, mmap_mode="r")
                    dim = int(arr.shape[-1]) if arr.ndim else 1
                except Exception:
                    pass
    return count, dim


def _best_candidate(root: Path, glob_pat: str) -> tuple[int, int | None, str | None]:
    best: tuple[int, int | None, str | None] = (0, None, None)
    if not root.is_dir():
        return best
    for d in root.glob(glob_pat):
        if not d.is_dir():
            continue
        c, dim = _count_and_dim(d)
        if c > best[0]:
            best = (c, dim, str(d))
    return best


def _packed_gt_candidate(v2_root: Path, scope: str) -> tuple[int, int | None, str]:
    """Record count + embedding dim for a packed GT scope (``hml_gt_motion``,
    ``tmr_gt_motion``), sampled by loading exactly one record."""
    store = ShardedCacheStore(v2_root)
    audit = store.audit(scope)
    if not audit.records:
        return 0, None, f"{v2_root}/{scope}"
    dim: int | None = None
    key = store.sample_key(scope)
    if key is not None:
        item = store.load_one(scope, key)
        if item is not None and "array" in item.arrays:
            arr = item.arrays["array"]
            dim = int(arr.shape[-1]) if arr.ndim else 1
    return audit.records, dim, f"{v2_root}/{scope}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class CacheAuditReport:
    generated_at: str
    cache_root: str
    packed_stores: dict[str, CacheAudit]
    cache_families: dict[str, FamilyStats]
    run_artifacts: dict[str, FamilyStats]
    logs: dict[str, LogStats]
    host_ram: HostRamStats | None
    gpus: list[GpuStats]
    memory_estimates: list[MemoryEstimate]
    notes: list[str]


def run_audit(
    *,
    cache_root: Path | None = None,
    runs_root: Path | None = None,
    extra_run_artifact_roots: list[Path] | None = None,
    include_gpu: bool = True,
    include_logs: bool = True,
) -> CacheAuditReport:
    from .cache import cache_root as default_cache_root

    root = Path(cache_root) if cache_root is not None else default_cache_root()
    root = require_durable_cache_root(root, operation="cache audit")
    notes: list[str] = []

    packed_stores: dict[str, CacheAudit] = {}
    if root.is_dir():
        v2_audit = audit_packed_store(packed_cache_root(root))
        for scope, a in v2_audit.items():
            packed_stores[f"v2::{scope}"] = a
        # Everything *other* than "v2" under the cache root is either a stray
        # leftover from before the packed-store migration or something placed
        # by hand — surface it as a plain small-file scan, never mixed with
        # the packed-store counts above.
        for child in sorted(
            p for p in root.iterdir() if p.is_dir() and p.name != PACKED_CACHE_DIRNAME
        ):
            stats = scan_tree(child)
            if stats.total_files:
                if child.name in LEGACY_DURABLE_FAMILIES:
                    notes.append(
                        f"stray legacy files under {child} (files={stats.total_files}); "
                        "consider `eval cache drop --legacy`"
                    )
                else:
                    notes.append(
                        f"unmanaged non-v2 files under {child} (files={stats.total_files}); "
                        "retained because no cache-drop command owns this path"
                    )
    else:
        notes.append(f"cache root does not exist: {root}")

    cache_families: dict[str, FamilyStats] = {}
    if root.is_dir():
        for child in sorted(
            p for p in root.iterdir() if p.is_dir() and p.name != PACKED_CACHE_DIRNAME
        ):
            cache_families[child.name] = scan_tree(child)

    run_artifact_roots = list(extra_run_artifact_roots or [])
    if runs_root is not None:
        run_artifact_roots += discover_run_artifact_roots(Path(runs_root))

    run_artifacts: dict[str, FamilyStats] = {}
    for rroot in run_artifact_roots:
        if not rroot.is_dir():
            run_artifacts[str(rroot)] = FamilyStats(root=str(rroot), exists=False)
            continue
        for child in sorted(p for p in rroot.iterdir() if p.is_dir() and p.name != "cache_v2"):
            key = f"{rroot}::{child.name}"
            run_artifacts[key] = scan_tree(child)
        cache_v2 = rroot / "cache_v2"
        if cache_v2.is_dir():
            for scope, a in audit_packed_store(cache_v2, num_buckets=16).items():
                packed_stores[f"{rroot}::cache_v2::{scope}"] = a

    logs: dict[str, LogStats] = {}
    if include_logs and runs_root is not None:
        for log_path in discover_logs(Path(runs_root)):
            logs[str(log_path)] = parse_log(log_path)

    host_ram: HostRamStats | None = None
    try:
        host_ram = host_ram_stats()
    except Exception as e:  # pragma: no cover - psutil always present in venv
        notes.append(f"host RAM stats unavailable: {e}")

    gpus: list[GpuStats] = []
    if include_gpu:
        gpus = gpu_stats()
        if not gpus:
            notes.append("no GPU stats available (nvidia-smi missing or failed)")

    memory_estimates: list[MemoryEstimate] = []
    # Prefer the packed v2 store; fall back to legacy per-file globs so an
    # audit taken mid-migration (stray leftovers, no v2 data yet) still works.
    hml_n, hml_d, hml_src = _packed_gt_candidate(packed_cache_root(root), "hml_gt_motion")
    if not hml_n:
        hml_n, hml_d, hml_src = _best_candidate(root, "hml_gt/*/motion_emb")
    if hml_n:
        memory_estimates.append(
            estimate_full_gallery(f"smpl_hml full-gallery retrieval (source={hml_src})", hml_n, hml_d or 512)
        )
    tmr_n, tmr_d, tmr_src = _packed_gt_candidate(packed_cache_root(root), "tmr_gt_motion")
    if not tmr_n:
        tmr_n, tmr_d, tmr_src = _best_candidate(root, "tmr_gt/*/motion_emb")
    if tmr_n:
        memory_estimates.append(
            estimate_full_gallery(f"soma_tmr full-gallery retrieval (source={tmr_src})", tmr_n, tmr_d or 256)
        )
    if not memory_estimates:
        notes.append("no GT motion_emb caches found yet; metric-memory estimate skipped")

    return CacheAuditReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        cache_root=str(root),
        packed_stores=packed_stores,
        cache_families=cache_families,
        run_artifacts=run_artifacts,
        logs=logs,
        host_ram=host_ram,
        gpus=gpus,
        memory_estimates=memory_estimates,
        notes=notes,
    )


def report_to_dict(report: CacheAuditReport) -> dict:
    return asdict(report)


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------

def _render_family(name: str, s: FamilyStats) -> list[str]:
    lines = [f"  {name:<24} root={s.root}"]
    if not s.exists:
        lines.append("      (missing)")
        return lines
    total_data = sum(es.count for ext, es in s.by_ext.items() if ext in DATA_EXTS)
    total_missing_sidecar = sum(es.without_sidecar for ext, es in s.by_ext.items() if ext in DATA_EXTS)
    coverage_str = (
        f"{100.0 * (1 - total_missing_sidecar / total_data):5.1f}%" if total_data else "  n/a"
    )
    lines.append(
        f"      files={s.total_files:>7}  bytes={fmt_bytes(s.total_bytes):>8}  "
        f"sidecars={s.sidecar_files:>7}  sidecar_coverage={coverage_str}  scan={s.scan_seconds:.2f}s"
    )
    for ext, es in sorted(s.by_ext.items(), key=lambda kv: -kv[1].bytes):
        extra = ""
        if ext in DATA_EXTS:
            extra = f"  with_sidecar={es.with_sidecar} without_sidecar={es.without_sidecar}"
        lines.append(f"        {ext or '(noext)':<8} count={es.count:>7}  bytes={fmt_bytes(es.bytes):>8}{extra}")
    hist = "  ".join(f"{k}={v}" for k, v in sorted(s.size_histogram.items()))
    lines.append(f"        size_histogram: {hist}")
    if s.top_subdirs_by_bytes:
        top = ", ".join(f"{k}({f} files, {fmt_bytes(b)})" for k, f, b in s.top_subdirs_by_bytes[:6])
        lines.append(f"        top_children: {top}")
    return lines


def _render_packed(name: str, a: CacheAudit) -> str:
    pv = ", ".join(f"{k}={v}" for k, v in sorted(a.protocol_versions.items())) or "n/a"
    flags = ""
    if a.corrupt or a.missing_packs:
        flags = f"  [!] corrupt={a.corrupt} missing_packs={a.missing_packs}"
    return (
        f"  {name:<40} records={a.records:>8}  bytes={fmt_bytes(a.bytes):>8}  "
        f"buckets={a.buckets:>3}  protocol_versions=({pv}){flags}"
    )


def render_text(report: CacheAuditReport) -> str:
    lines: list[str] = []
    lines.append(f"=== eval cache audit @ {report.generated_at} ===")
    lines.append(f"cache_root: {report.cache_root}")

    lines.append("\n[packed v2 stores]")
    if not report.packed_stores:
        lines.append("  (none found — nothing cached yet)")
    for name, a in sorted(report.packed_stores.items()):
        lines.append(_render_packed(name, a))

    lines.append("\n[stray non-v2 files under cache root — should be empty on a clean install]")
    if not report.cache_families:
        lines.append("  (none found)")
    for name, s in report.cache_families.items():
        lines.extend(_render_family(name, s))

    lines.append("\n[run_artifacts: stray non-cache_v2 files — should be empty on a clean install]")
    if not report.run_artifacts:
        lines.append("  (none found)")
    for name, s in report.run_artifacts.items():
        lines.extend(_render_family(name, s))

    lines.append("\n[shard logs]")
    if not report.logs:
        lines.append("  (none found)")
    total_rejected = sum(l.cache_rejected for l in report.logs.values())
    total_corrupt = sum(l.cache_corrupt for l in report.logs.values())
    total_fallback = sum(l.live_encode_fallback for l in report.logs.values())
    for path, l in report.logs.items():
        lines.append(
            f"  {path}: lines={l.lines} rejected={l.cache_rejected} corrupt={l.cache_corrupt} "
            f"live_fallback={l.live_encode_fallback} size={fmt_bytes(l.size_bytes)}"
        )
        if l.last_progress_line:
            lines.append(f"      last_progress: {l.last_progress_line}")
    if report.logs:
        lines.append(
            f"  TOTAL: rejected={total_rejected} corrupt={total_corrupt} live_fallback={total_fallback}"
        )

    lines.append("\n[host RAM]")
    if report.host_ram:
        r = report.host_ram
        lines.append(
            f"  total={fmt_bytes(r.total_bytes)} available={fmt_bytes(r.available_bytes)} "
            f"used={fmt_bytes(r.used_bytes)} ({r.percent:.1f}%) this_process_rss={fmt_bytes(r.process_rss_bytes)}"
        )
    else:
        lines.append("  (unavailable)")

    lines.append("\n[GPU]")
    if report.gpus:
        for g in report.gpus:
            lines.append(
                f"  gpu{g.index} {g.name}: mem_used={g.memory_used_mb:.0f}MB / {g.memory_total_mb:.0f}MB "
                f"free={g.memory_free_mb:.0f}MB util={g.utilization_pct:.0f}%"
            )
    else:
        lines.append("  (unavailable)")

    lines.append("\n[metric memory estimates]")
    if report.memory_estimates:
        for m in report.memory_estimates:
            lines.append(
                f"  {m.label}: N={m.n} D={m.d} -> full_gallery(NxNxD,f32)={fmt_bytes(m.full_gallery_bytes)}  "
                f"dup_aware_sim(NxN,f64)={fmt_bytes(m.dup_aware_sim_bytes)}"
            )
    else:
        lines.append("  (none)")

    if report.notes:
        lines.append("\n[notes]")
        for n in report.notes:
            lines.append(f"  - {n}")

    return "\n".join(lines)


__all__ = [
    "ExtStats",
    "FamilyStats",
    "LogStats",
    "HostRamStats",
    "GpuStats",
    "MemoryEstimate",
    "CacheAuditReport",
    "scan_tree",
    "audit_packed_store",
    "parse_log",
    "host_ram_stats",
    "gpu_stats",
    "estimate_full_gallery",
    "run_audit",
    "report_to_dict",
    "render_text",
]
