"""Our additions on top of the upstream ``kimodo`` package.

The retrieval teacher we train replaces kimodo's LLM2Vec text encoder with
Flan-T5, and our precompute paths need a checkpoint resolver with strict
offline semantics. Both live here rather than as a patch to the submodule.
"""

from __future__ import annotations

from pathlib import Path


def resolve_tmr_checkpoint(
    modelname: str, *, local_files_only: bool | None = None
) -> Path:
    """Resolve a registered kimodo/TMR checkpoint in the standard HF cache.

    ``local_files_only=True`` is what precompute jobs and the asset verifier
    want: they should not know the Hugging Face cache layout, and they must
    fail loudly rather than silently start a multi-gigabyte download when the
    assets were never installed. ``None`` keeps kimodo's ``LOCAL_CACHE``
    behaviour.
    """
    from huggingface_hub import snapshot_download
    from kimodo.model.loading import MODEL_NAMES, get_env_var

    try:
        repo_id = MODEL_NAMES[modelname]
    except KeyError:
        raise ValueError(
            f"Model {modelname!r} not found. Available models: {list(MODEL_NAMES)}"
        ) from None

    if local_files_only is not None:
        return Path(snapshot_download(repo_id=repo_id, local_files_only=local_files_only))

    if get_env_var("LOCAL_CACHE", "False").lower() != "true":
        return Path(snapshot_download(repo_id=repo_id))
    try:
        return Path(snapshot_download(repo_id=repo_id, local_files_only=True))
    except Exception:
        return Path(snapshot_download(repo_id=repo_id))


class FlanT5TextEncoder:
    """Adapt :class:`FlanT5Encoder` to kimodo's ``LLM2VecEncoder`` interface.

    ``LLM2VecEncoder.__call__(texts)`` returns ``(emb [B, 1, 4096], lengths [B])``
    whereas ``FlanT5Encoder.encode(texts)`` returns ``(emb [B, L, 2048], mask
    [B, L])``. Converting the mask to lengths lets ``TMR.full_text_encoder()``
    rebuild it through ``length_to_mask()``. Right-padding throughout.
    """

    def __init__(
        self,
        model_id: str = "google/flan-t5-xl",
        device: str = "auto",
        max_length: int = 64,
        dtype: str = "bfloat16",
        llm_dim: int = 2048,
    ) -> None:
        import torch

        from ...text.flan_t5_encoder import FlanT5Encoder

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._enc = FlanT5Encoder.load(
            model_id, device=device, max_length=max_length, dtype=getattr(torch, dtype),
        )
        self.llm_dim = llm_dim

    def to(self, device=None, dtype=None):
        return self

    def eval(self):
        return self

    def __call__(self, texts: list[str]):
        emb, mask = self._enc.encode(list(texts))
        return emb, mask.sum(dim=1).tolist()


def patch_tmr_motion_rep() -> None:
    """Fix an upstream broadcast bug in kimodo's TMR feature representation.

    ``TMRMotionRep.translate_2d`` reads the planar offset as
    ``translation_2d[:, 0]``, which has shape ``[B]`` where broadcasting against
    the ``[B, T]`` root track needs ``[B, 1]``. A batch of one happens to work.
    Beyond that it raises, except when the batch size equals the padded clip
    length, where it instead spreads one clip's offset across frames and returns
    a silently wrong result. ``TMR.encode_motion`` always canonicalizes, so
    every batched encode reaches this. kimodo's sibling ``KimodoMotionRep``
    already uses the correct ``[:, [0]]`` form.

    Idempotent, so it is safe to call on every model load.
    """
    from kimodo.motion_rep.reps.tmr_motionrep import TMRMotionRep
    from kimodo.tools import ensure_batched

    if getattr(TMRMotionRep.translate_2d, "_semoco_broadcast_fix", False):
        return

    @ensure_batched(features=3, translation_2d=2)
    def translate_2d(self, features, translation_2d):
        """Translate root planar position by ``(dx, dz)``."""
        if translation_2d.dim() == 1:
            translation_2d = translation_2d.repeat(features.shape[0], 1)
        new_features = features.clone()
        new_root_pos = new_features[:, :, self.slice_dict["root_pos"]]
        new_root_pos[:, :, 0] += translation_2d[:, [0]]
        new_root_pos[:, :, 2] += translation_2d[:, [1]]
        return new_features

    translate_2d._semoco_broadcast_fix = True
    TMRMotionRep.translate_2d = translate_2d


__all__ = ["FlanT5TextEncoder", "patch_tmr_motion_rep", "resolve_tmr_checkpoint"]
