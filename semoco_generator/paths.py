"""Filesystem bridge to the frozen SeMoCo tokenizer repo.

The generator consumes the tokenizer's model/codec code (to load the frozen
checkpoint and decode codes back to motion) but does not vendor it. We add the
tokenizer repo root to ``sys.path`` so its ``models`` / ``data`` packages import
cleanly.

Resolution order for the tokenizer repo root:
    1. ``$SOMA_TOKENIZER_ROOT`` env var, if set;
    2. a sibling ``SeMoCo-Tokenizer`` directory next to this repo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent

# Dataset root that ``local://`` resolves against; override with
# $MOTIONVERSE_DATA_ROOT (the tokenizer repo reads the same variable).
_DEFAULT_DATASETS_ROOT = _WORKSPACE_ROOT / "semoco-MotionVerse"

ENV_DATA_ROOT = "MOTIONVERSE_DATA_ROOT"


def repo_root() -> Path:
    """Return the SeMoCo-Generator repository root."""
    return _REPO_ROOT


def workspace_root() -> Path:
    """Return the workspace containing this repo and its sibling projects."""
    return _WORKSPACE_ROOT


def humanml3d_root() -> Path:
    """Return the default HumanML3D data root."""
    return datasets_root() / "HumanML3D"


def glove_root() -> Path:
    """Return the default HumanML3D GloVe asset root."""
    return datasets_root() / "glove"


def soma_tokenizer_root() -> Path:
    """Return the SeMoCo tokenizer repo root."""
    env = os.environ.get("SOMA_TOKENIZER_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"SOMA_TOKENIZER_ROOT={env!r} is not a directory")
    sibling = _WORKSPACE_ROOT / "SeMoCo-Tokenizer"
    if sibling.is_dir():
        return sibling.resolve()
    raise FileNotFoundError(
        "Could not locate the SeMoCo tokenizer repo; set $SOMA_TOKENIZER_ROOT "
        "(https://github.com/OMEGA-i/SeMoCo-Tokenizer)."
    )


def smpl_model_path() -> Path:
    """Return directory containing ``smpl/SMPL_NEUTRAL.pkl``.

    ``smplx.create(model_path=…, model_type='smpl')`` looks here.
    Resolution: ``$SMPL_MODEL_PATH`` → the tokenizer repo's ``assets/``.
    """
    env = os.environ.get("SMPL_MODEL_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"SMPL_MODEL_PATH={env!r} not found")
    return soma_tokenizer_root() / "assets"


def smplx_model_path() -> Path:
    """Return directory containing ``smplx/SMPLX_NEUTRAL.npz``.

    ``smplx.create(model_path=…, model_type='smplx')`` looks here.
    Resolution: ``$SMPLX_MODEL_PATH`` → the tokenizer repo's ``assets/``.
    """
    env = os.environ.get("SMPLX_MODEL_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"SMPLX_MODEL_PATH={env!r} not found")
    return soma_tokenizer_root() / "assets"


def datasets_root() -> Path:
    """Return the dataset root that ``local://`` URIs resolve against."""
    env = os.environ.get(ENV_DATA_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_DATASETS_ROOT.resolve()


def baseline_checkpoint_root() -> Path:
    """Return the root for downloaded (non-Hugging Face) checkpoints.

    Weights never ship with this package. ``python -m
    semoco_generator.tools.fetch_assets`` installs archives and direct
    downloads here; override with ``$SEMOCO_BASELINE_CKPT_ROOT``, otherwise a
    sibling ``checkpoints/`` directory is used.
    """
    env = os.environ.get("SEMOCO_BASELINE_CKPT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _WORKSPACE_ROOT / "checkpoints"


def ensure_tokenizer_on_path() -> Path:
    """Add the tokenizer repo root to ``sys.path`` (idempotent)."""
    root = soma_tokenizer_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def default_checkpoint() -> Path:
    """Resolve the frozen tokenizer used by the configured default eval runs.

    Resolution is explicit ``$SOMA_TOKENIZER_CHECKPOINT`` first, then the
    tokenizer recorded by the default checkpoint registry. The last resort is
    the tokenizer repo's own default training output, which is what a
    standalone install without an eval registry ends up with.
    """
    env = os.environ.get("SOMA_TOKENIZER_CHECKPOINT")
    if env:
        return Path(env).expanduser().resolve()

    registered = _registered_default_checkpoint()
    if registered is not None:
        return registered

    return (
        soma_tokenizer_root()
        / "runs"
        / "semoco_split_fp"
        / "model"
        / "best.pt"
    )


def _registered_default_checkpoint() -> Path | None:
    """Read the tokenizer from the configured default eval checkpoint.

    The lazy import keeps generic path resolution usable in installations that
    intentionally do not ship evaluation dependencies.
    """
    try:
        from .eval.checkpoints import configured_default_specs

        for spec in configured_default_specs():
            if spec.tokenizer_ckpt.is_file():
                return spec.tokenizer_ckpt
    except (ImportError, OSError):
        pass
    return None
