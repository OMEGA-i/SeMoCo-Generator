"""Motion-representation conversion helpers shared by the evaluation tracks.

The SOMA/SMPL bridges live here: :class:`SOMAConverter` turns SMPL mesh
vertices into SOMA77 joints, and the ``utils`` submodule holds the rot6d /
joints22 conversions the export path needs.
"""
from __future__ import annotations

from .soma_converter import SOMAConverter

__all__ = ["SOMAConverter"]
