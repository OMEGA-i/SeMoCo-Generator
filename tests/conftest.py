"""Environment gating so a fresh clone runs green without external downloads.

Three markers opt a test into something the repo cannot bundle:

* ``requires_tokenizer`` — needs the SeMoCo tokenizer repo on disk
  (``$SOMA_TOKENIZER_ROOT`` or a sibling checkout).
* ``requires_smplx`` — needs the registration-gated SMPL-X model
  (``$SMPLX_MODEL_PATH`` or the tokenizer's ``assets/`` tree).
* ``requires_kimodo`` — needs the ``third_party/kimodo`` submodule installed
  along with the ``[tmr]`` extra it imports.

All three skip rather than fail when the dependency is missing. Run them
explicitly with e.g. ``pytest -m requires_tokenizer`` once it is in place.
"""

from __future__ import annotations

import importlib.util

import pytest

_MARKERS = {
    "requires_tokenizer": "needs the SeMoCo-Tokenizer repo; set $SOMA_TOKENIZER_ROOT",
    "requires_smplx": "needs the SMPL-X model; set $SMPLX_MODEL_PATH",
    "requires_kimodo": (
        "needs third_party/kimodo installed; "
        "`uv sync --extra tmr && uv pip install -e third_party/kimodo`"
    ),
}


def pytest_configure(config: pytest.Config) -> None:
    for name, reason in _MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {reason}")


def _tokenizer_available() -> bool:
    from semoco_generator.paths import soma_tokenizer_root

    try:
        soma_tokenizer_root()
    except FileNotFoundError:
        return False
    return True


def _smplx_available() -> bool:
    from semoco_generator.paths import smplx_model_path

    try:
        return (smplx_model_path() / "smplx" / "SMPLX_NEUTRAL.npz").is_file()
    except FileNotFoundError:
        return False


def _kimodo_available() -> bool:
    # A bare find_spec("kimodo") is not enough: the submodule can be importable
    # while its own dependencies (einops, omegaconf, ...) are absent, which is
    # exactly the state a `uv sync` without the [tmr] extra leaves behind.
    if importlib.util.find_spec("kimodo") is None:
        return False
    try:
        importlib.import_module("kimodo.model.loading")
    except ImportError:
        return False
    return True


_PROBES = {
    "requires_tokenizer": _tokenizer_available,
    "requires_smplx": _smplx_available,
    "requires_kimodo": _kimodo_available,
}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    available: dict[str, bool] = {}
    for item in items:
        for name, reason in _MARKERS.items():
            if name not in item.keywords:
                continue
            if name not in available:
                available[name] = _PROBES[name]()
            if not available[name]:
                item.add_marker(pytest.mark.skip(reason=reason))
