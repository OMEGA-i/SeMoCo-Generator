"""Canonical version constants for all eval caches.

Single source of truth to avoid import cycles between ``cache.py`` and
``models/registry.py``.  Every cache-key-producing function reads these —
no magic strings.

Bump rules (manual only — source changes do NOT auto-invalidate):
* ``GEN_PROTOCOL_VERSION_STEP`` — invalidates ALL native cache.
* ``GT_PROTOCOL_VERSION`` — invalidates GT precomputation (evaluator, tokenizer, stats).
* ``CONVERSION_GRAPH_VERSION`` — invalidates cached converted targets when a
  ConversionGraph edge changes.
* ``GEN_ALIGN_VERSION`` — baked into every model's ``weight_signature()``;
  bump when generation-time post-processing changes clip content.
"""

from __future__ import annotations

# Bump when a ConversionGraph edge changes so cached converted targets invalidate.
CONVERSION_GRAPH_VERSION: str = "v2"

# Bump to invalidate ALL native cache.
# Only change this manually — source code changes do NOT auto-invalidate.
GEN_PROTOCOL_VERSION_STEP: int = 2
"""Increment this integer when old native cache should be discarded."""

GENERATION_PROTOCOL_VERSION: str = str(GEN_PROTOCOL_VERSION_STEP)
"""Embedded in native cache keys.  Manual bump only."""

GT_PROTOCOL_VERSION: str = "v3"
"""Bump when GT precomputation logic changes (evaluator, tokenizer, mean-std)."""

# Bump whenever generation-time post-processing changes the *content* of a
# cached native / converted / gen-embed clip. Entries computed before the change
# are then never silently reused: they become unreachable orphans under the old
# signature. Baked into every model's ``weight_signature()``.
GEN_ALIGN_VERSION: str = "align_gtlen_v1"


__all__ = [
    "CONVERSION_GRAPH_VERSION",
    "GENERATION_PROTOCOL_VERSION",
    "GEN_PROTOCOL_VERSION_STEP",
    "GEN_ALIGN_VERSION",
    "GT_PROTOCOL_VERSION",
]
