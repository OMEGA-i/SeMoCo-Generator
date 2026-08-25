"""TMR retrieval evaluator facade.

TMR is a **metric / evaluator** (FID, R-precision on SOMA77), not a text-to-motion
baseline.  The model implementation still lives under the vendored kimodo package
(shared ``load_model`` / skeleton / motion_rep with Kimodo generation); callers
should import from here instead of ``kimodo``.

TMR variants are registered in :data:`TMR_REGISTRY` via :func:`_register`.
Use :func:`load_tmr` for a single model or :func:`load_tmr_multi` to load
multiple variants with automatic motion-encoder sharing.

Each model is fully ready after loading — no deferred setup, no stubs,
no post-construction patching.  Text encoders are loaded inline (one per GPU).
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

from ...paths import repo_root


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class TMREntry:
    """Describes how to load one TMR model variant."""

    name: str
    """Short key used on the CLI (``--tmr-model <name>``)."""

    loader: Callable[..., nn.Module]
    """Callable ``(device, rprecision, **kw) -> nn.Module``."""

    text_encoder_kind: str = "builtin"
    """Display-only label for ``eval tmr list``: ``"builtin"``, ``"flan"``, ``"llm2vec"``."""

    motion_encoder_from: str | None = None
    """If set, :func:`load_tmr_multi` will pass this entry's motion encoder to the
    dependent loader, avoiding a duplicate load."""


TMR_REGISTRY: dict[str, TMREntry] = {}
"""All registered TMR model variants.  Populated by :func:`_register` at import time."""


def _register(entry: TMREntry) -> TMREntry:
    """Register a TMR variant (idempotent — last write wins)."""
    TMR_REGISTRY[entry.name] = entry
    return entry


# ---------------------------------------------------------------------------
# Built-in loaders
# ---------------------------------------------------------------------------

_TMR_FLAN_CKPT: Optional[Path] = None


def _resolve_tmr_flan_ckpt() -> Path:
    """Resolve the ``tmr-soma-flan`` checkpoint directory."""
    global _TMR_FLAN_CKPT
    if _TMR_FLAN_CKPT is not None:
        return _TMR_FLAN_CKPT

    candidates = [
        repo_root() / "runs" / "tmr-soma-flan",
    ]
    for c in candidates:
        if (c / "model" / "text_encoder.pt").is_file():
            _TMR_FLAN_CKPT = c.resolve()
            return _TMR_FLAN_CKPT

    raise FileNotFoundError(
        "tmr-soma-flan checkpoint not found. "
        "Train it first with: python -m semoco_generator.train.train_tmr_flan "
        "--config configs/tmr_flan.yaml"
    )


def _load_tmr_soma_rp(
    device: str | torch.device = "cuda",
    *,
    rprecision: bool = True,
    **_kw,
) -> nn.Module:
    """Load NVIDIA TMR-SOMA-RP-v1 from HuggingFace.

    When *rprecision* is True, LLM2Vec is loaded inline (~16 GB).
    When False, ``raw_text_encoder`` is ``None`` (FID-only).
    """
    from kimodo.model.load_model import load_model

    warnings.filterwarnings(
        "ignore",
        message="Already found a `peft_config` attribute in the model.",
    )

    # When rprecision=False, we only need the motion encoder. Pass a dummy
    # text encoder so load_model() skips LLM2Vec (~16 GB) creation.
    if not rprecision:
        text_encoder = nn.Identity()  # dummy — never called, just prevents LLM2Vec load
    else:
        text_encoder = None  # load_model builds LLM2Vec internally
    return load_model(
        modelname="tmr-soma-rp",
        device=str(device),
        default_family="TMR",
        text_encoder=text_encoder,
    )


def _load_tmr_soma_flan(
    device: str | torch.device = "cuda",
    *,
    rprecision: bool = True,
    motion_encoder: Optional[nn.Module] = None,
    **_kw,
) -> nn.Module:
    """Load the TMR-Flan model (local checkpoint + Flan-T5 live encoder).

    If *motion_encoder* is provided, it is reused instead of loading a second
    copy from HuggingFace.
    """
    from .kimodo_compat import FlanT5TextEncoder
    from kimodo.model.loading import load_checkpoint_state_dict
    from kimodo.model.tmr import ACTORStyleEncoder, TMR

    ckpt_dir = _resolve_tmr_flan_ckpt()

    # ---- Motion encoder (load from local checkpoint) -----------------------
    if motion_encoder is not None:
        mot_enc = motion_encoder
    else:
        from kimodo.motion_rep import TMRMotionRep
        from kimodo.skeleton import SOMASkeleton30

        stats_path = str(ckpt_dir / "model" / "motion_stats")
        mot_enc = ACTORStyleEncoder(
            motion_rep=TMRMotionRep(fps=30.0, skeleton=SOMASkeleton30(),
                                    stats_path=stats_path),
            llm_shape=None,
            vae=True, latent_dim=256, ff_size=1024,
            num_layers=6, num_heads=4, dropout=0.1, activation="gelu",
        )
        mot_enc.load_state_dict(
            load_checkpoint_state_dict(ckpt_dir / "model" / "motion_encoder.pt"),
        )
        mot_enc = mot_enc.to(device).eval()
        for p in mot_enc.parameters():
            p.requires_grad_(False)

    # ---- Text encoder (ACTORStyleEncoder from local checkpoint) -------------
    text_encoder = ACTORStyleEncoder(
        motion_rep=None,
        llm_shape=(-1, 2048),  # Flan-T5-XL clip_dim
        vae=True,
        latent_dim=256,
        ff_size=1024,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
    )
    text_encoder.load_state_dict(
        load_checkpoint_state_dict(ckpt_dir / "model" / "text_encoder.pt"),
    )
    text_encoder = text_encoder.to(device).eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    # ---- Raw text encoder (Flan-T5) -----------------------------------------
    raw_enc = FlanT5TextEncoder(device=str(device)) if rprecision else None

    # ---- Assemble TMR -------------------------------------------------------
    tmr = TMR(
        motion_encoder=mot_enc,
        top_text_encoder=text_encoder,
        vae=True,
        text_encoder=raw_enc,
        device=str(device),
        sample_mean=True,
        unit_vector=True,
        compute_grads=False,
    ).to(device).eval()

    return tmr


# ---- Register built-in variants --------------------------------------------

_register(TMREntry(
    name="tmr-soma-rp",
    loader=_load_tmr_soma_rp,
    text_encoder_kind="llm2vec",
))

_register(TMREntry(
    name="tmr-soma-flan",
    loader=_load_tmr_soma_flan,
    text_encoder_kind="flan",
    motion_encoder_from="tmr-soma-rp",
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_tmr(
    device: str | torch.device = "cuda",
    *,
    modelname: str = "tmr-soma-rp",
    rprecision: bool = True,
) -> nn.Module:
    """Load a pretrained TMR retrieval model.

    The model is fully ready after this call — ``encode_raw_text`` works
    immediately (if *rprecision* is True).  Text encoders are loaded inline:
    one copy per GPU, no shared pool, no deferred setup.

    Parameters
    ----------
    device: Target device string or ``torch.device``.
    modelname: Registered TMR model key. Built-in choices:
        - ``tmr-soma-rp`` → NVIDIA TMR-SOMA-RP-v1 (LLM2Vec, ~16 GB)
        - ``tmr-soma-flan`` → local checkpoint (Flan-T5, ~2 GB)
    rprecision: If False, ``raw_text_encoder`` is ``None`` (FID-only, no
        text encoder loaded).  Use ``--no-rprecision`` for this path.
    """
    entry = TMR_REGISTRY.get(modelname)
    if entry is None:
        raise KeyError(
            f"Unknown TMR model {modelname!r}. "
            f"Registered: {sorted(TMR_REGISTRY)}. "
            f"Use 'eval tmr list' to see all available models."
        )
    from .kimodo_compat import patch_tmr_motion_rep

    patch_tmr_motion_rep()
    return entry.loader(device=str(device), rprecision=rprecision)


def load_tmr_multi(
    device: str | torch.device = "cuda",
    *,
    modelnames: list[str],
    rprecision: bool = True,
) -> dict[str, nn.Module]:
    """Load multiple TMR variants, sharing motion encoders where declared.

    Loads models in order.  When an entry declares ``motion_encoder_from`` and
    the referenced model has already been loaded, its motion encoder is reused
    to avoid a duplicate load (~2 GB GPU RAM saved).

    Returns a dict mapping model name → TMR module.
    """
    from .kimodo_compat import patch_tmr_motion_rep

    patch_tmr_motion_rep()
    tmr_dict: dict[str, nn.Module] = {}
    for name in modelnames:
        entry = TMR_REGISTRY.get(name)
        if entry is None:
            raise KeyError(
                f"Unknown TMR model {name!r}. "
                f"Registered: {sorted(TMR_REGISTRY)}."
            )

        kwargs: dict = dict(device=str(device), rprecision=rprecision)

        # Motion encoder sharing (declarative, no hardcoded names)
        if entry.motion_encoder_from:
            src_name = entry.motion_encoder_from
            if src_name in tmr_dict:
                kwargs["motion_encoder"] = tmr_dict[src_name].motion_encoder

        tmr_dict[name] = entry.loader(**kwargs)

    return tmr_dict


def close_tmr(tmr_model) -> None:
    """Best-effort release of GPU resources for a TMR model."""
    try:
        tmr_model.raw_text_encoder = None
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = [
    "TMR_REGISTRY",
    "TMREntry",
    "_register",
    "close_tmr",
    "load_tmr",
    "load_tmr_multi",
]
