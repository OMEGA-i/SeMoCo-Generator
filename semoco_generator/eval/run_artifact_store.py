"""``RunArtifactStore``: packed-shard artifact storage scoped to one protocol run.

Same interface shape as :class:`~semoco_generator.eval.sharded_cache_store.ShardedCacheStore`
(``probe_many`` / ``load_many`` / ``put_many`` / ``audit`` / ``drop``), but the
storage root lives under a run's ``run_artifacts`` directory rather than the
durable shared cache. Callers should not need to know whether artifacts are
packed shards or legacy small files — this module hides that behind
``MotionClip``/embedding-aware helpers.

Key shapes follow the production plan:

* Converted: ``(kind=converted, dataset_sig, model_id, weight_sig, prompt_id,
  seed, cfg, target_rep, conversion_version)``
* Embedding: ``(kind=gen_emb, track, dataset_sig, eval_sig, model_id,
  weight_sig, prompt_id, seed, cfg)``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .cache_utils import cfg_str
from .schema import MotionClip, MotionRep
from .sharded_cache_store import ArtifactBatch, CacheAudit, PutRecord, ShardedCacheStore

CACHE_V2_DIRNAME = "cache_v2"



def converted_key(
    model_id: str, weight_sig: str, prompt_id: str, seed: int, cfg: float | None,
    target_rep: MotionRep, *, dataset_sig: str = "", conversion_version: str = "v1",
) -> str:
    return (
        f"converted:{dataset_sig}:{model_id}:{weight_sig}:{prompt_id}:s{int(seed)}:"
        f"cfg{cfg_str(cfg)}:{target_rep}:{conversion_version}"
    )


def gen_embedding_key(
    track: str, model_id: str, weight_sig: str, prompt_id: str, seed: int, cfg: float | None,
    *, dataset_sig: str = "", eval_sig: str = "",
) -> str:
    return (
        f"gen_emb:{track}:{dataset_sig}:{eval_sig}:{model_id}:{weight_sig}:{prompt_id}:"
        f"s{int(seed)}:cfg{cfg_str(cfg)}"
    )


def clip_to_arrays(clip: MotionClip) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "array": np.asarray(clip.array),
        "fps": np.asarray(float(clip.fps), dtype=np.float64),
    }
    for k, v in clip.aux.items():
        arrays[f"aux__{k}"] = np.asarray(v)
    return arrays


def arrays_to_clip(rep: str, arrays: dict[str, np.ndarray]) -> MotionClip:
    aux = {k[len("aux__"):]: np.asarray(v, dtype=np.float32) for k, v in arrays.items() if k.startswith("aux__")}
    return MotionClip(
        rep=rep,  # type: ignore[arg-type]
        array=np.asarray(arrays["array"], dtype=np.float32),
        fps=float(np.asarray(arrays["fps"])),
        aux=aux,
    )


class RunArtifactStore:
    """Packed-shard artifact store for one protocol run's converted/gen_emb data."""

    def __init__(self, run_root: str | Path, *, num_buckets: int = 8) -> None:
        self.run_root = Path(run_root)
        self.store = ShardedCacheStore(self.run_root / CACHE_V2_DIRNAME, num_buckets=num_buckets)

    # ------------------------------------------------------------------
    # Converted targets
    # ------------------------------------------------------------------
    def put_converted(
        self, model_id, weight_sig, prompt_id, seed, cfg, target_rep, clip: MotionClip,
        *, dataset_sig: str = "", conversion_version: str = "v1",
    ) -> None:
        key = converted_key(
            model_id, weight_sig, prompt_id, seed, cfg, target_rep,
            dataset_sig=dataset_sig, conversion_version=conversion_version,
        )
        self.store.put_many("converted", [
            PutRecord(key=key, arrays=clip_to_arrays(clip), meta={"rep": str(clip.rep)}),
        ])

    def put_converted_many(
        self, items: list[tuple[str, str, str, int, float | None, str, object]],
        *, dataset_sig: str = "", conversion_version: str = "v1",
    ) -> None:
        """Batch-save multiple converted targets in one ``put_many`` call.

        Each item is ``(model_id, weight_sig, prompt_id, seed, cfg, target_rep, clip)``.
        """
        if not items:
            return
        records = [
            PutRecord(
                key=converted_key(
                    model_id, weight_sig, prompt_id, seed, cfg, target_rep,
                    dataset_sig=dataset_sig, conversion_version=conversion_version,
                ),
                arrays=clip_to_arrays(clip),
                meta={"rep": str(clip.rep)},
            )
            for model_id, weight_sig, prompt_id, seed, cfg, target_rep, clip in items
        ]
        self.store.put_many("converted", records)

    def load_converted(
        self, model_id, weight_sig, prompt_id, seed, cfg, target_rep,
        *, dataset_sig: str = "", conversion_version: str = "v1",
    ) -> MotionClip | None:
        key = converted_key(
            model_id, weight_sig, prompt_id, seed, cfg, target_rep,
            dataset_sig=dataset_sig, conversion_version=conversion_version,
        )
        item = self.store.load_one("converted", key)
        if item is None:
            return None
        return arrays_to_clip(item.meta.get("rep", target_rep), item.arrays)

    def probe_converted_many(self, keys: list[str]) -> dict[str, bool]:
        status = self.store.probe_many("converted", keys)
        return {k: v.exists for k, v in status.items()}

    # ------------------------------------------------------------------
    # Generated-motion embeddings
    # ------------------------------------------------------------------
    def put_gen_embedding(
        self, track, model_id, weight_sig, prompt_id, seed, cfg, emb: np.ndarray,
        *, dataset_sig: str = "", eval_sig: str = "",
    ) -> None:
        key = gen_embedding_key(
            track, model_id, weight_sig, prompt_id, seed, cfg,
            dataset_sig=dataset_sig, eval_sig=eval_sig,
        )
        self.store.put_many("gen_emb", [PutRecord(key=key, arrays={"array": np.asarray(emb, dtype=np.float32)})])

    def load_gen_embedding(
        self, track, model_id, weight_sig, prompt_id, seed, cfg,
        *, dataset_sig: str = "", eval_sig: str = "",
    ) -> np.ndarray | None:
        key = gen_embedding_key(
            track, model_id, weight_sig, prompt_id, seed, cfg,
            dataset_sig=dataset_sig, eval_sig=eval_sig,
        )
        item = self.store.load_one("gen_emb", key)
        if item is None:
            return None
        arr = item.arrays.get("array")
        if arr is None:
            arr = item.arrays.get("emb")
        return None if arr is None else np.asarray(arr, dtype=np.float32)

    def probe_gen_embedding_many(self, keys: list[str]) -> dict[str, bool]:
        status = self.store.probe_many("gen_emb", keys)
        return {k: v.exists for k, v in status.items()}

    # ------------------------------------------------------------------
    def load_many_converted(self, keys: list[str], *, max_bytes: int | None = None):
        for batch in self.store.load_many("converted", keys, max_bytes=max_bytes):
            yield [(item.key, arrays_to_clip(item.meta.get("rep", ""), item.arrays)) for item in batch]

    def load_many_gen_embeddings(self, keys: list[str], *, max_bytes: int | None = None):
        for batch in self.store.load_many("gen_emb", keys, max_bytes=max_bytes):
            yield [(item.key, np.asarray((item.arrays.get("array", item.arrays.get("emb"))), dtype=np.float32)) for item in batch]

    # ------------------------------------------------------------------
    def audit(self, scope: str) -> CacheAudit:
        return self.store.audit(scope)

    def drop(self, scope: str, *, dry_run: bool = True) -> list[str]:
        return self.store.drop(scope, dry_run=dry_run)

    def scopes(self) -> list[str]:
        return self.store.scopes()


__all__ = [
    "CACHE_V2_DIRNAME",
    "RunArtifactStore",
    "converted_key",
    "gen_embedding_key",
]
