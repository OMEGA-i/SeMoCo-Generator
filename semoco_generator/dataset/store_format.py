"""Shared schema for the motion-code store format.

Both the export tools (``tools/export_*.py``) and the dataset classes
(``MotionCodeDataset``, ``T2MCodeDataset``) read and write these files.
This module defines the expected layout, dtype, and JSON key conventions
so the contract is explicit rather than implicit across four files.

Format version: 1
"""

from __future__ import annotations

from typing import TypedDict


# ---------------------------------------------------------------------------
# Motion-only store (export_motion_codes.py -> MotionCodeDataset)
# ---------------------------------------------------------------------------
class ClipIndexEntry(TypedDict):
    rec_id: str
    start: int   # row offset in .codes.npy
    length: int  # number of token rows


class CodecMeta(TypedDict, total=False):
    split: str
    num_clips: int
    num_tokens: int
    num_codebooks: int
    codebook_size: int
    temporal_stride: int
    source_fps: float
    token_rate: float
    checkpoint: str
    recordings_root: str
    dtype: str  # e.g. "int16"


def motion_store_files(split: str) -> dict[str, str]:
    """Return the three files expected for a motion-only split."""
    return {
        "codes": f"{split}.codes.npy",    # int16 [sum_T, Q]
        "index": f"{split}.index.json",   # list[ClipIndexEntry]
        "meta":  f"{split}.meta.json",    # CodecMeta
    }


# ---------------------------------------------------------------------------
# T2M paired store (export_t2m_dataset.py -> T2MCodeDataset)
# ---------------------------------------------------------------------------
class T2MClipIndexEntry(TypedDict, total=False):
    rec_id: str
    code_start: int
    code_len: int
    row: int
    caption: str


class T2MTextIndexEntry(TypedDict):
    text_start: int
    text_len: int


def t2m_store_files(split: str, text_encoder_key: str | None = None) -> dict[str, str]:
    """Return the files expected for a T2M split.

    Shared (encoder-agnostic) files plus per-encoder text embeddings.
    """
    suffix = f".{text_encoder_key}" if text_encoder_key else ""
    return {
        "codes":      f"{split}.codes.npy",               # int16 [sum_T, Q]
        "anchor":     f"{split}.anchor.npy",               # float32 [N, 465]
        "identity":   f"{split}.identity.npy",             # float32 [N, C]
        "index":      f"{split}.index.json",               # list[T2MClipIndexEntry]
        "meta":       f"{split}.meta.json",                # CodecMeta + anchor_dim + identity_dim
        "text_emb":   f"{split}.text_emb{suffix}.npy",     # float16 [sum_L, clip_dim]
        "text_index": f"{split}.text_index{suffix}.json",  # list[T2MTextIndexEntry]
        "text_meta":  f"{split}.meta{suffix}.json",        # {clip_dim, encode_key, ...}
    }
