"""Work-manifest planner for the eval worker pool.

Emits deterministic JSON/JSONL artifacts (`run.json` / `work_units.jsonl`) so
:class:`~semoco_generator.eval.worker_pool.GPUWorkerPool` can consume work
units without rescanning directories or parsing logs.

Two manifest builders are provided:

* :func:`build_pilot_manifest` — even-sized chunks over a flat prompt-id
  list. Useful for smoke tests and CLI pilots.
* :func:`build_track_manifest` — duration-aware, longest-processing-time
  (LPT) bin packing over real ``(prompt_id, duration_seconds)`` pairs, so a
  shard balances estimated generation/conversion cost rather than just clip
  count (per the "Aggressive Pipeline Plan" duration-aware sharding).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class WorkUnit:
    unit_id: str
    track: str
    model: str
    phase: str
    prompt_ids: list[str]
    estimated_items: int
    depends_on: list[str]


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    track: str
    model: str
    split: str
    dataset: str
    created_at: float
    num_prompts: int
    chunk_size: int


def _chunks(items: list[str], size: int) -> list[list[str]]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


def build_prompt_ids(*, count: int, prefix: str = "prompt") -> list[str]:
    return [f"{prefix}_{i:06d}" for i in range(int(count))]


@dataclass(frozen=True)
class TrackPromptCost:
    """One planner-visible prompt: id plus an estimated per-clip cost.

    ``duration_s`` is the natural cost proxy (longer clips cost more
    generation/conversion/encode time); pass ``None`` to fall back to
    ``estimate_cost``'s default for models without a duration signal, i.e.
    those whose output length is decided by their own EOS.
    """

    prompt_id: str
    duration_s: float | None = None


def estimate_cost(duration_s: float | None, *, default: float = 3.0) -> float:
    return float(duration_s) if duration_s and duration_s > 0 else float(default)


# Empirical per-model generation cost multipliers, relative to a "1.0" model.
# ``build_track_manifest`` reuses the *same* duration-based LPT shards for
# every model (so shard item counts are comparable across models), but some
# baselines are far slower per second-of-motion than others (a diffusion
# pipeline vs. a single-pass transformer, say). Without a
# model-aware weight, :func:`unit_priority` can't tell a worker to go help
# with the expensive model instead of leaving it for whichever worker happens
# to still be busy at the end — which is exactly the GPU-idle tail-skew
# failure mode this exists to fix.
DEFAULT_MODEL_COST_WEIGHT = 1.0
# Relative wall-cost per clip, used to schedule the expensive models first.
MODEL_COST_WEIGHTS: dict[str, float] = {
    "semoco": 1.3,
}

# native-gen dominates wall time; convert/gen-embed are comparatively cheap
# batched post-processing steps, so they shouldn't compete for lease priority
# against a still-unstarted native-gen unit for a heavy model.
PHASE_COST_WEIGHTS: dict[str, float] = {
    "native-gen": 1.0,
    "convert": 0.15,
    "gen-embed": 0.15,
}


def unit_priority(unit: WorkUnit) -> float:
    """Higher = a worker should attempt/lease this unit before others.

    Lexicographic: the per-model cost weight dominates completely, the
    per-phase cost weight is the tie-break *within* one model, and the unit's
    item count (a duration proxy: LPT already sized shards so heavier
    per-item cost gets fewer items) only breaks remaining ties. This means
    *every* unit for a heavier model outranks *every* unit for a lighter one
    — workers process one model's whole chain (native-gen, then its convert,
    then its gen-embed) before moving to the next model, rather than a plain
    ``model_weight * phase_weight`` product letting a lighter model's
    native-gen units interleave with a heavier model's convert/gen-embed
    units. That interleaving would otherwise force
    :func:`~semoco_generator.eval.planner_exec.run_global_planner` to reload
    models back and forth far more than necessary (each load/close is not
    free, and simultaneously-resident models are a real VRAM/OOM risk).

    Workers process ``pending`` units in descending priority order, so
    multiple concurrent workers preferentially race for the heaviest
    remaining model's work first instead of only discovering it once every
    lighter model is already claimed — this is what fixes a worker idling
    once its current model runs dry while a heavier model still has ready
    work.
    """
    model_weight = MODEL_COST_WEIGHTS.get(unit.model, DEFAULT_MODEL_COST_WEIGHT)
    phase_weight = PHASE_COST_WEIGHTS.get(unit.phase, 1.0)
    items = max(1, unit.estimated_items)
    return model_weight * 1_000_000.0 + phase_weight * 1_000.0 + items


def lpt_shards(items: Sequence[tuple[str, float]], num_shards: int) -> list[list[str]]:
    """Longest-processing-time bin packing: sort by cost desc, assign to the
    currently lightest shard. Balances estimated cost per shard instead of
    just clip count (see the plan's duration-aware sharding requirement)."""
    num_shards = max(1, int(num_shards))
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    loads = [0.0] * num_shards
    for prompt_id, cost in sorted(items, key=lambda x: -x[1]):
        idx = min(range(num_shards), key=lambda i: loads[i])
        shards[idx].append(prompt_id)
        loads[idx] += cost
    return shards


def build_track_manifest(
    *,
    track: str,
    models: Sequence[str],
    split: str = "test",
    dataset: str = "HumanML3D",
    prompts: Sequence[TrackPromptCost],
    num_shards: int = 8,
    run_id: str | None = None,
) -> tuple[RunManifest, list[WorkUnit]]:
    """Duration-aware manifest covering every model in ``models`` for one track.

    Each model gets its own independent LPT sharding (models differ in
    generation cost per second of motion), and each shard becomes a
    ``native-gen -> convert -> gen-embed`` dependency chain. This is the
    entry point for "all baselines + our model" planning: pass every model id
    you want covered and the planner emits one coherent work manifest.
    """
    run_id = run_id or f"{track}-multi-{split}-{int(time.time())}"
    items = [(p.prompt_id, estimate_cost(p.duration_s)) for p in prompts]
    shards = lpt_shards(items, num_shards)
    run = RunManifest(
        run_id=run_id,
        track=track,
        model=",".join(models),
        split=split,
        dataset=dataset,
        created_at=time.time(),
        num_prompts=len(items),
        chunk_size=max(1, len(items) // max(1, num_shards)),
    )
    units: list[WorkUnit] = []
    for model in models:
        for shard_idx, prompt_ids in enumerate(shards):
            if not prompt_ids:
                continue
            native_id = f"{run_id}:{model}:native-gen:{shard_idx:05d}"
            convert_id = f"{run_id}:{model}:convert:{shard_idx:05d}"
            embed_id = f"{run_id}:{model}:gen-embed:{shard_idx:05d}"
            units.append(WorkUnit(native_id, track, model, "native-gen", prompt_ids, len(prompt_ids), []))
            units.append(WorkUnit(convert_id, track, model, "convert", prompt_ids, len(prompt_ids), [native_id]))
            units.append(WorkUnit(embed_id, track, model, "gen-embed", prompt_ids, len(prompt_ids), [convert_id]))
    return run, units


def build_pilot_manifest(
    *,
    track: str = "smpl_hml",
    model: str = "semoco",
    split: str = "test",
    dataset: str = "HumanML3D",
    prompt_ids: list[str],
    chunk_size: int = 64,
    run_id: str | None = None,
) -> tuple[RunManifest, list[WorkUnit]]:
    run_id = run_id or f"{track}-{model}-{split}-{int(time.time())}"
    run = RunManifest(
        run_id=run_id,
        track=track,
        model=model,
        split=split,
        dataset=dataset,
        created_at=time.time(),
        num_prompts=len(prompt_ids),
        chunk_size=max(1, int(chunk_size)),
    )
    units: list[WorkUnit] = []
    for idx, ids in enumerate(_chunks(prompt_ids, chunk_size)):
        native_id = f"{run_id}:native-gen:{idx:05d}"
        convert_id = f"{run_id}:convert:{idx:05d}"
        embed_id = f"{run_id}:gen-embed:{idx:05d}"
        units.append(WorkUnit(native_id, track, model, "native-gen", ids, len(ids), []))
        units.append(WorkUnit(convert_id, track, model, "convert", ids, len(ids), [native_id]))
        units.append(WorkUnit(embed_id, track, model, "gen-embed", ids, len(ids), [convert_id]))
    return run, units


def write_manifest(out_dir: str | Path, run: RunManifest, units: list[WorkUnit]) -> None:
    """Publish the manifest so concurrent readers never observe a partial file.

    Each file is written to a ``.tmp`` sibling and atomically renamed into
    place (POSIX ``rename`` is all-or-nothing on the same filesystem), and
    ``work_units.jsonl`` is published *before* ``run.json``. Callers use
    ``run.json`` existing as the "manifest is ready" signal (see
    :func:`semoco_generator.eval.planner_exec.build_or_read_track_manifest`),
    so by the time a racing process observes it, the units file is already
    complete — never caught mid-write with zero or a truncated prefix of
    lines.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    units_tmp = out / f"work_units.jsonl.{pid}.tmp"
    with units_tmp.open("w") as f:
        for unit in units:
            f.write(json.dumps(asdict(unit), sort_keys=True) + "\n")
    units_tmp.replace(out / "work_units.jsonl")
    run_tmp = out / f"run.json.{pid}.tmp"
    run_tmp.write_text(json.dumps(asdict(run), indent=2, sort_keys=True) + "\n")
    run_tmp.replace(out / "run.json")


def read_manifest(out_dir: str | Path) -> tuple[RunManifest, list[WorkUnit]]:
    """Read back a manifest written by :func:`write_manifest`.

    Used for resume: the worker pool re-derives its ready/blocked state from
    ``committed.jsonl``/``leases.jsonl`` (see :mod:`worker_pool`), but the
    work unit *definitions* themselves come from this manifest, not from
    rescanning directories or parsing shard logs.
    """
    out = Path(out_dir)
    run_data = json.loads((out / "run.json").read_text())
    run = RunManifest(**run_data)
    units: list[WorkUnit] = []
    units_path = out / "work_units.jsonl"
    if units_path.is_file():
        with units_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    units.append(WorkUnit(**json.loads(line)))
    return run, units


__all__ = [
    "DEFAULT_MODEL_COST_WEIGHT",
    "MODEL_COST_WEIGHTS",
    "PHASE_COST_WEIGHTS",
    "RunManifest",
    "TrackPromptCost",
    "WorkUnit",
    "build_pilot_manifest",
    "build_prompt_ids",
    "build_track_manifest",
    "estimate_cost",
    "lpt_shards",
    "read_manifest",
    "unit_priority",
    "write_manifest",
]
