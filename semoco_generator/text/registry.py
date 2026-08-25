"""Text encoder registry — maps short keys to encoder classes."""

from __future__ import annotations

from typing import Type

from .base import TextEncoder

_registry: dict[str, Type[TextEncoder]] = {}


def register(key: str, cls: Type[TextEncoder]) -> None:
    """Register a text encoder class under a short key."""
    if not hasattr(cls, "load"):
        raise TypeError(
            f"{cls.__name__} is missing the load() classmethod — "
            f"cannot register as a TextEncoder"
        )
    _registry[key] = cls


def get_encoder_cls(key: str) -> Type[TextEncoder]:
    """Look up a registered encoder class by key.

    Raises:
        KeyError: if *key* is not registered.
    """
    if key not in _registry:
        raise KeyError(
            f"Unknown text encoder {key!r}. "
            f"Available: {list(_registry)}"
        )
    return _registry[key]


def list_encoders() -> list[str]:
    """Return all registered encoder keys."""
    return list(_registry)
