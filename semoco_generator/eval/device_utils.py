"""Device validation utilities for eval runners.

Ensures CUDA_VISIBLE_DEVICES is consistent with --device arguments and
provides actionable error messages when GPU configuration is wrong.
"""

from __future__ import annotations

import os
import sys

import torch


def validate_device(
    device: str | torch.device,
    *,
    num_shards: int = 1,
    scheduler: str = "legacy",
    stage: str = "eval",
) -> None:
    """Validate CUDA device configuration at startup.

    Checks:
    1. CUDA is available (``torch.cuda.is_available()``)
    2. ``CUDA_VISIBLE_DEVICES`` env var is set and consistent
    3. Requested device index is within visible GPUs
    4. If num_shards > 1 and scheduler is legacy, warn if >1 GPU visible

    Raises SystemExit with an actionable error message if configuration is
    invalid, rather than letting the pipeline crash later with a cryptic
    "No CUDA GPUs are available" at model-load time.
    """
    dev = torch.device(device)

    # Check 1: CUDA availability
    if dev.type == "cuda" and not torch.cuda.is_available():
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        msg = (
            f"\n[{stage}] FATAL: --device {device} requested but no CUDA GPUs are available.\n"
            f"  CUDA_VISIBLE_DEVICES={cvd!r}\n"
            f"  torch.cuda.device_count()={torch.cuda.device_count()}\n"
            f"\n  Possible causes:\n"
            f"    1. CUDA_VISIBLE_DEVICES is not set or is empty.\n"
            f"    2. The env var was not propagated to the Python process\n"
            f"       (e.g. 'nohup env CUDA_VISIBLE_DEVICES=0 ...' silently\n"
            f"        ignores the env prefix when the binary path is relative).\n"
            f"    3. The specified GPU index does not exist on this machine.\n"
            f"\n  Debug:\n"
            f"    $ nvidia-smi -L               # list all GPUs\n"
            f"    $ echo $CUDA_VISIBLE_DEVICES   # check current value\n"
            f"\n  Fix: use absolute paths in launcher scripts and set\n"
            f"       CUDA_VISIBLE_DEVICES=<gpu_id> before the python command.\n"
        )
        print(msg, file=sys.stderr, flush=True)
        raise SystemExit(1)

    # Check 2: Device index validity when CUDA is available
    if dev.type == "cuda":
        num_gpus = torch.cuda.device_count()
        if dev.index is not None and dev.index >= num_gpus:
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            msg = (
                f"\n[{stage}] FATAL: --device {device} index {dev.index} is out of range.\n"
                f"  torch.cuda.device_count()={num_gpus}\n"
                f"  CUDA_VISIBLE_DEVICES={cvd!r}\n"
                f"\n  With CUDA_VISIBLE_DEVICES set, cuda:0 always maps to the\n"
                f"  first visible GPU. Always use --device cuda:0 when the\n"
                f"  launcher pins one GPU per worker via CUDA_VISIBLE_DEVICES.\n"
            )
            print(msg, file=sys.stderr, flush=True)
            raise SystemExit(1)

    # Check 3: Warn if multiple GPUs visible when sharding (legacy scheduler)
    if num_shards > 1 and scheduler == "legacy":
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd:
            visible = [d.strip() for d in cvd.split(",") if d.strip()]
            if len(visible) > 1:
                print(
                    f"[{stage}] WARNING: --num-shards={num_shards} but "
                    f"CUDA_VISIBLE_DEVICES={cvd!r} exposes {len(visible)} GPUs. "
                    "Each worker process should see exactly one GPU. "
                    "The launcher script should set CUDA_VISIBLE_DEVICES=<single_gpu> per worker.",
                    flush=True,
                )

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd:
        print(
            f"[{stage}] NOTE: CUDA_VISIBLE_DEVICES is not set. "
            f"If running multi-GPU eval, each worker should be pinned to one GPU.",
            flush=True,
        )

    print(
        f"[{stage}] device validated: --device={device}, "
        f"CUDA_VISIBLE_DEVICES={cvd!r}, "
        f"visible_count={torch.cuda.device_count() if dev.type == 'cuda' else 'N/A'}",
        flush=True,
    )
