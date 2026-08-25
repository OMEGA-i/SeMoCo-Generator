"""Async cache I/O: overlap disk writes with GPU compute.

Delegates ``save_native_many`` / ``save_converted_many`` /
``save_gen_motion_many`` to a background thread so the main thread can
start the next batch's generation immediately.

Thread-safety
-------------
A single worker thread serialises writes; the process-wide singleton
``ShardedCacheStore`` instances in :mod:`cache` are protected by
``_index_cache_lock`` (added in the same change).  The main thread only
calls ``probe_many`` / ``load_many`` which are read-dominated and
tolerate stale caches (``_load_bucket_index_fresh_on_miss`` handles
cross-process freshness already).

Memory safety
-------------
With ``max_workers=1``, at most one batch worth of data (~2-5 MB of
motion codes) is ever queued.  Generation takes 30-250 s per batch
while a pack-file append + fsync takes ~100 ms, so the worker thread
always drains the queue before the next batch finishes.  No unbounded
accumulation is possible.
"""

from __future__ import annotations

import atexit
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from . import cache as C
from .schema import MotionClip

_DEFAULT_MAX_WORKERS = int(os.environ.get("SEMOCO_CACHE_IO_WORKERS", "1"))


class AsyncCacher:
    """Background-write helper for eval cache operations.

    Usage::

        cacher = AsyncCacher()
        for batch in batches:
            outputs = model.generate(batch)          # GPU compute
            cacher.submit_save_native(mid, sig, ...) # enqueue write
            # next iteration starts immediately — GPU stays busy
        cacher.flush_all()  # before model.close() or program exit
    """

    def __init__(self, max_workers: int = _DEFAULT_MAX_WORKERS) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cache_io",
        )
        self._pending: list[Future] = []
        atexit.register(self.flush_all)

    # ------------------------------------------------------------------
    # Submit helpers — return immediately, do the real I/O in background
    # ------------------------------------------------------------------

    def submit(self, operation: Callable[..., object], /, *args, **kwargs) -> Future:
        """Enqueue a cache operation exposed by a higher-level artifact handle."""
        f = self._executor.submit(operation, *args, **kwargs)
        self._pending.append(f)
        return f

    def submit_save_native(
        self,
        model_id: str,
        ckpt_signature: str,
        items: list[tuple[str, int, float | None, MotionClip]],
        *,
        dataset: str = "",
    ) -> Future:
        """Enqueue a native-motion cache write.  Returns a ``Future``."""
        f = self._executor.submit(
            C.save_native_many, model_id, ckpt_signature, items, dataset=dataset,
        )
        self._pending.append(f)
        return f

    def submit_save_converted(
        self,
        model_id: str,
        ckpt_signature: str,
        items: list[tuple[str, int, float | None, str, object]],
        *,
        dataset: str = "",
        run_root: str | Path | None = None,
    ) -> Future:
        """Enqueue a converted-target cache write.  Returns a ``Future``."""
        f = self._executor.submit(
            C.save_converted_many,
            model_id, ckpt_signature, items,
            dataset=dataset, run_root=run_root,
        )
        self._pending.append(f)
        return f

    def submit_save_gen_motion(
        self,
        track: str,
        eval_sig: str,
        model_id: str,
        ckpt_signature: str,
        items: list[tuple[str, int, float | None, np.ndarray]],
        *,
        dataset: str = "",
        run_root: str | Path | None = None,
    ) -> Future:
        """Enqueue a gen-motion-embedding cache write.  Returns a ``Future``."""
        f = self._executor.submit(
            C.save_gen_motion_many,
            track, eval_sig, model_id, ckpt_signature, items,
            dataset=dataset, run_root=run_root,
        )
        self._pending.append(f)
        return f

    # ------------------------------------------------------------------
    # Flush — block until all pending writes complete
    # ------------------------------------------------------------------

    def flush_all(self, timeout: float | None = None) -> None:
        """Wait for every pending background write to finish.

        Raises on the first exception from any future so fsync / disk-full
        errors surface immediately rather than being silently lost.
        """
        pending = self._pending[:]
        self._pending.clear()
        for f in pending:
            # result() re-raises any exception from the worker thread.
            _ = f.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> AsyncCacher:
        return self

    def __exit__(self, *exc_info) -> None:
        self.flush_all()


__all__ = ["AsyncCacher"]
