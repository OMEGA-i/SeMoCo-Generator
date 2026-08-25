"""``T2MCodeDataset`` - serves paired (text embedding, motion codes) clips.

Reads the store written by ``tools/export_t2m_dataset.py``.

**New format (multi-encoder):**

    <root>/<split>.codes.npy              # motion codes — shared
    <root>/<split>.anchor.npy             # shared
    <root>/<split>.identity.npy           # shared
    <root>/<split>.index.json             # shared: code offsets + captions
    <root>/<split>.meta.json              # codec metadata
    <root>/<split>.text_emb.{key}.npy     # per-encoder embeddings
    <root>/<split>.text_index.{key}.json  # per-encoder text offsets (aligned by row)
    <root>/<split>.meta.{key}.json        # per-encoder metadata (clip_dim etc.)

**Old format (backward compat):**

    <root>/<split>.text_emb.npy           # text embeddings
    <root>/<split>.index.json             # self-contained (code + text offsets)
    <root>/<split>.meta.json              # codec + text-encoder metadata

Pass ``text_encoder_key="siglip"`` to load the suffixed variant.  Omit
(``None``) for backward-compatible unsuffixed loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..local_uri import resolve_local_uri


class T2MCodeDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        text_encoder_key: str | None = None,
        max_motion_tok: int = 300,
        min_motion_tok: int = 2,
    ) -> None:
        self.root = resolve_local_uri(root)
        self.split = split
        self.text_encoder_key = text_encoder_key
        self.max_motion_tok = int(max_motion_tok)
        self.min_motion_tok = int(min_motion_tok)

        suffix = f".{text_encoder_key}" if text_encoder_key else ""

        codes_path = self.root / f"{split}.codes.npy"

        # Shared index (code offsets + captions)
        shared_index_path = self.root / f"{split}.index.json"

        # Text files: suffixed first, unsuffixed fallback (backward compat)
        text_path = self.root / f"{split}.text_emb{suffix}.npy"
        text_index_path = self.root / f"{split}.text_index{suffix}.json"
        meta_path = self.root / f"{split}.meta{suffix}.json"

        if not text_path.is_file():
            if suffix:
                # Backward compat: old stores have no suffix
                text_path = self.root / f"{split}.text_emb.npy"
                text_index_path = self.root / f"{split}.index.json"  # old: text offsets in shared index
                meta_path = self.root / f"{split}.meta.json"
            if not text_path.is_file():
                raise FileNotFoundError(f"missing text_emb: {text_path}")

        self.codes = np.load(codes_path, mmap_mode="r")           # [sum_T_tok, Q] int16
        self.text_emb = np.load(text_path, mmap_mode="r")         # [sum_L, dim] fp16
        self.shared_index = json.loads(shared_index_path.read_text())
        # Per-encoder text offsets (aligned by row with shared_index),
        # or shared index for old-format stores.
        if text_index_path.is_file():
            self.text_index = json.loads(text_index_path.read_text())
        else:
            self.text_index = self.shared_index
        # Codec metadata from shared meta.json; encoder metadata from suffixed meta.
        # Shared meta may not exist yet (--text-only before full export).
        shared_meta_path = self.root / f"{split}.meta.json"
        if shared_meta_path.is_file():
            shared_meta = json.loads(shared_meta_path.read_text())
        else:
            shared_meta = {}
        self.meta = json.loads(meta_path.read_text())
        self.num_codebooks = int(shared_meta.get("num_codebooks", self.meta.get("num_codebooks", 16)))
        self.codebook_size = int(shared_meta.get("codebook_size", self.meta.get("codebook_size", 1024)))
        self.clip_dim = int(self.meta["clip_dim"])

        # Anchors / identities are only needed for decode/eval; load lazily.
        self._anchor_path = self.root / f"{split}.anchor.npy"
        self._identity_path = self.root / f"{split}.identity.npy"
        self._anchors = None
        self._identities = None

        # Drop clips that are too short to train on (need >=1 target frame).
        self._keep = [i for i, e in enumerate(self.shared_index) if int(e.get("code_len", 0)) >= self.min_motion_tok]

    def __len__(self) -> int:
        return len(self._keep)

    def codebook_sizes(self) -> list[int]:
        return [self.codebook_size] * self.num_codebooks

    @property
    def anchors(self) -> np.ndarray:
        if self._anchors is None:
            self._anchors = np.load(self._anchor_path, mmap_mode="r")
        return self._anchors

    @property
    def identities(self) -> np.ndarray:
        if self._identities is None:
            self._identities = np.load(self._identity_path, mmap_mode="r")
        return self._identities

    def entry(self, idx: int) -> dict:
        """Raw index entry (rec_id + offsets) for the ``idx``-th kept clip."""
        e = self.shared_index[self._keep[idx]]
        te = self.text_index[self._keep[idx]]
        return {**e, "text_start": te.get("text_start", 0), "text_len": te.get("text_len", 0)}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        e = self.shared_index[self._keep[idx]]
        te = self.text_index[self._keep[idx]]
        cs, cl = int(e["code_start"]), int(e["code_len"])
        ts, tl = int(te["text_start"]), int(te["text_len"])
        cl = min(cl, self.max_motion_tok)
        codes = np.asarray(self.codes[cs : cs + cl], dtype=np.int64)         # [Tm, Q]
        temb = np.asarray(self.text_emb[ts : ts + tl], dtype=np.float32)     # [L, dim]
        return {
            "motion_codes": torch.from_numpy(codes),
            "text_emb": torch.from_numpy(temb),
        }


def collate_t2m(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Left-pad text, right-pad motion; build validity masks.

    Returns
        text_emb     : ``[B, Lmax, dim]``  (left-padded; zeros on pad)
        text_valid   : ``[B, Lmax]`` bool  (True = real word)
        motion_codes : ``[B, Tmax, Q]`` long (right-padded; zeros on pad)
        motion_valid : ``[B, Tmax]`` bool  (True = real frame)
    """
    B = len(batch)
    dim = batch[0]["text_emb"].shape[-1]
    Q = batch[0]["motion_codes"].shape[-1]
    Lmax = max(int(b["text_emb"].shape[0]) for b in batch)
    Tmax = max(int(b["motion_codes"].shape[0]) for b in batch)

    text_emb = torch.zeros(B, Lmax, dim, dtype=torch.float32)
    text_valid = torch.zeros(B, Lmax, dtype=torch.bool)
    motion_codes = torch.zeros(B, Tmax, Q, dtype=torch.long)
    motion_valid = torch.zeros(B, Tmax, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = int(b["text_emb"].shape[0])
        T = int(b["motion_codes"].shape[0])
        text_emb[i, Lmax - L :] = b["text_emb"]          # LEFT-pad text
        text_valid[i, Lmax - L :] = True
        motion_codes[i, :T] = b["motion_codes"]          # RIGHT-pad motion
        motion_valid[i, :T] = True

    return {
        "text_emb": text_emb,
        "text_valid": text_valid,
        "motion_codes": motion_codes,
        "motion_valid": motion_valid,
    }
