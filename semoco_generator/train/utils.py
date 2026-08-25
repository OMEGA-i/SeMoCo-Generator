"""Shared training utilities used by all trainers.

Import from here instead of cross-importing between ``train_motion_gpt.py``
and ``train_t2m.py``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def dist_info() -> tuple[int, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world, local, world > 1
    return 0, 1, 0, False


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, msg: str) -> None:
    if is_main(rank):
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def lr_at(
    step: int,
    base_lr: float,
    warmup: int,
    max_steps: int,
    min_ratio: float = 0.1,
) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return base_lr * min_ratio
    progress = (step - warmup) / max(1, max_steps - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cos)


# ---------------------------------------------------------------------------
# SwanLab (rank-0 only, no-op fallback)
# ---------------------------------------------------------------------------
class _NullRun:
    id: str | None = None
    url: str | None = None

    def log(self, *a, **k) -> None:
        return None

    def finish(self) -> None:
        return None


def init_swanlab(
    swan_cfg: dict,
    *,
    name: str,
    config: dict,
    is_main_proc: bool,
    resume_id: str | None = None,
    out_dir: str | None = None,
):
    """Init SwanLab. When ``resume_id`` is set, continue that cloud run's curves.

    Also writes ``{out_dir}/swanlab_run.json`` so later resumes can find the id
    even if the checkpoint predates this field.
    """
    if not is_main_proc or not swan_cfg.get("enabled", False):
        return _NullRun()
    try:
        import swanlab  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[swanlab] disabled (import failed: {exc})")
        return _NullRun()

    init_kwargs: dict = {
        "project": swan_cfg.get("project", "VanillaMotionGPT"),
        "workspace": swan_cfg.get("workspace"),
        "experiment_name": swan_cfg.get("experiment_name", name),
        "config": config,
    }
    # Prefer explicit resume_id; else try sidecar from a previous launch in out_dir.
    sid = resume_id
    if not sid and out_dir is not None:
        sidecar = Path(out_dir) / "swanlab_run.json"
        if sidecar.is_file():
            try:
                sid = json.loads(sidecar.read_text()).get("id")
            except Exception:  # noqa: BLE001
                sid = None
    if sid:
        init_kwargs["id"] = sid
        init_kwargs["resume"] = "must"
        print(f"[swanlab] resuming run id={sid}", flush=True)

    try:
        run = swanlab.init(**init_kwargs)
    except Exception as exc:  # noqa: BLE001
        if sid:
            # Cloud run gone / id stale — fall back to a fresh run rather than abort.
            print(f"[swanlab] resume failed ({exc}); starting a new run", flush=True)
            init_kwargs.pop("id", None)
            init_kwargs.pop("resume", None)
            try:
                run = swanlab.init(**init_kwargs)
            except Exception as exc2:  # noqa: BLE001
                print(f"[swanlab] init failed ({exc2}); no-op")
                return _NullRun()
        else:
            print(f"[swanlab] init failed ({exc}); no-op")
            return _NullRun()

    run_id = getattr(run, "id", None) or (swanlab.get_run().id if swanlab.get_run() else None)
    run_url = getattr(run, "url", None)

    if out_dir is not None and run_id:
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "swanlab_run.json").write_text(
                json.dumps({"id": run_id, "url": run_url, "name": init_kwargs.get("experiment_name")}, indent=2),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[swanlab] failed to write sidecar: {exc}")

    class _Run:
        id = run_id
        url = run_url

        def log(self, metrics: dict, *, step: int | None = None) -> None:
            try:
                swanlab.log(metrics, step=step) if step is not None else swanlab.log(metrics)
            except Exception as exc:  # noqa: BLE001
                print(f"[swanlab] log failed: {exc}")

        def finish(self) -> None:
            try:
                swanlab.finish()
            except Exception:  # noqa: BLE001
                pass

    return _Run()
