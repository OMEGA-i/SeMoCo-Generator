"""``MotionCodeDataset`` - serves packed motion-code packets for AR training.

Reads the packed store written by ``tools/export_motion_codes.py``:

    <root>/<split>.codes.npy   # int16 [sum_T_tok, Q], clips concatenated
    <root>/<split>.index.json  # [{rec_id, start, length}, ...]
    <root>/<split>.meta.json   # codec metadata

Sequence packing (document-masked)
----------------------------------
Mocap clips are short (~7s avg), so one-clip-per-sample wastes a long context
window on padding. Instead we pack multiple whole clips into one ``ctx``-length
sequence and tag each token with a ``segment_id`` (which clip it belongs to) and
a per-clip ``position`` (RoPE index reset at every clip boundary). The model
then applies a **block-diagonal causal mask** so a token only attends within its
own clip - packing becomes equivalent to per-clip training but with no padding
and full GPU utilisation. Long clips stay intact as a single document (true
long-horizon context up to ``ctx``).

Each sample:

    motion_codes : LongTensor [ctx, Q]
    segment_ids  : LongTensor [ctx]   (per-window clip id; -1 = padding)
    positions    : LongTensor [ctx]   (RoPE index within each clip; 0 on pad)
    valid_mask   : BoolTensor [ctx]   (True = real token)

Two packing modes:
* ``shuffle=True`` (train): each sample greedily fills ``ctx`` with randomly
  sampled clips (fresh every call) -> varied clip combinations across epochs.
* ``shuffle=False`` (val): deterministic contiguous windows over the packed
  stream -> stable metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..local_uri import resolve_local_uri


def _positions_from_segments(seg: np.ndarray) -> np.ndarray:
    """Per-block reset index: 0,1,2,... within each run of equal ``segment_id``."""
    n = seg.shape[0]
    ar = np.arange(n, dtype=np.int64)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = seg[1:] != seg[:-1]
    last_reset = np.maximum.accumulate(np.where(change, ar, 0))
    return ar - last_reset


class MotionCodeDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        context_length: int = 2048,
        pack: bool = True,
        shuffle: bool = True,
        pad_code: int = 0,
        seed: int | None = None,
        rank: int = 0,
    ) -> None:
        self.root = resolve_local_uri(root)
        self.split = split
        self.context_length = int(context_length)
        self.pack = bool(pack)
        self.shuffle = bool(shuffle)
        self.pad_code = int(pad_code)
        self._seed = seed if seed is not None else 0
        self._rank = int(rank)

        codes_path = self.root / f"{split}.codes.npy"
        index_path = self.root / f"{split}.index.json"
        meta_path = self.root / f"{split}.meta.json"
        for p in (codes_path, index_path, meta_path):
            if not p.is_file():
                raise FileNotFoundError(f"missing code store file: {p}")

        self.codes = np.load(codes_path, mmap_mode="r")          # [N, Q] int16
        self.index = json.loads(index_path.read_text())
        self.meta = json.loads(meta_path.read_text())
        self.num_codebooks = int(self.meta["num_codebooks"])
        self.codebook_size = int(self.meta["codebook_size"])
        if self.codes.shape[1] != self.num_codebooks:
            raise ValueError(
                f"codes Q={self.codes.shape[1]} != meta num_codebooks={self.num_codebooks}"
            )

        # Support both motion-specific format (start/length) and T2M format (code_start/code_len)
        first = self.index[0]
        key_start = "code_start" if "code_start" in first else "start"
        key_len = "code_len" if "code_len" in first else "length"
        self.starts = np.asarray([int(e[key_start]) for e in self.index], dtype=np.int64)
        self.lengths = np.asarray([int(e[key_len]) for e in self.index], dtype=np.int64)
        self.total_tokens = int(self.codes.shape[0])

        if self.pack:
            self._num_samples = max(1, self.total_tokens // self.context_length)
        else:
            self._num_samples = len(self.index)

    def __len__(self) -> int:
        return self._num_samples

    def codebook_sizes(self) -> list[int]:
        return [self.codebook_size] * self.num_codebooks

    # ------------------------------------------------------------------
    def _emit(self, codes: np.ndarray, seg: np.ndarray) -> dict[str, torch.Tensor]:
        """Pad to ``ctx`` and build positions; ``seg`` uses -1 for padding."""
        L = self.context_length
        n = codes.shape[0]
        out_codes = np.full((L, self.num_codebooks), self.pad_code, dtype=np.int64)
        out_seg = np.full((L,), -1, dtype=np.int64)
        out_codes[:n] = codes
        out_seg[:n] = seg
        positions = _positions_from_segments(out_seg)
        positions[out_seg < 0] = 0
        valid = out_seg >= 0
        return {
            "motion_codes": torch.from_numpy(out_codes),
            "segment_ids": torch.from_numpy(out_seg),
            "positions": torch.from_numpy(positions),
            "valid_mask": torch.from_numpy(valid),
        }

    def _random_packed(self, idx: int) -> dict[str, torch.Tensor]:
        L = self.context_length
        n_clips = len(self.index)
        # Deterministic per-call seed: reproducible across workers/epochs/ranks.
        wid = 0
        wi = torch.utils.data.get_worker_info()
        if wi is not None:
            wid = wi.id
        sub_seed = hash((self._seed, self._rank, wid, idx)) & 0x7FFFFFFF
        rng = np.random.default_rng(sub_seed)
        codes_parts: list[np.ndarray] = []
        seg_parts: list[np.ndarray] = []
        filled = 0
        seg_id = 0
        while filled < L:
            ci = int(rng.integers(0, n_clips))
            s, ln = int(self.starts[ci]), int(self.lengths[ci])
            remaining = L - filled
            if ln >= L and filled == 0:
                off = int(rng.integers(0, ln - L + 1)) if ln > L else 0
                chunk = np.asarray(self.codes[s + off : s + off + L], dtype=np.int64)
                codes_parts.append(chunk)
                seg_parts.append(np.full((L,), seg_id, dtype=np.int64))
                filled = L
                break
            take = min(ln, remaining)
            chunk = np.asarray(self.codes[s : s + take], dtype=np.int64)
            codes_parts.append(chunk)
            seg_parts.append(np.full((take,), seg_id, dtype=np.int64))
            filled += take
            seg_id += 1
        codes = np.concatenate(codes_parts, axis=0)[:L]
        seg = np.concatenate(seg_parts, axis=0)[:L]
        return self._emit(codes, seg)

    def _contiguous_window(self, idx: int) -> dict[str, torch.Tensor]:
        L = self.context_length
        g0 = idx * L
        g1 = min(g0 + L, self.total_tokens)
        codes = np.asarray(self.codes[g0:g1], dtype=np.int64)
        pos_global = np.arange(g0, g1, dtype=np.int64)
        clip_ids = np.searchsorted(self.starts, pos_global, side="right") - 1
        # Relabel to 0,1,2,... per block so segment ids are window-local.
        change = np.empty(clip_ids.shape[0], dtype=bool)
        if change.shape[0] > 0:
            change[0] = True
            change[1:] = clip_ids[1:] != clip_ids[:-1]
            seg = np.cumsum(change) - 1
        else:
            seg = clip_ids
        return self._emit(codes, seg.astype(np.int64))

    def _single_clip(self, idx: int) -> dict[str, torch.Tensor]:
        s, ln = int(self.starts[idx]), int(self.lengths[idx])
        L = self.context_length
        take = min(ln, L)
        codes = np.asarray(self.codes[s : s + take], dtype=np.int64)
        seg = np.zeros((take,), dtype=np.int64)
        return self._emit(codes, seg)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not self.pack:
            return self._single_clip(idx)
        if self.shuffle:
            return self._random_packed(idx)
        return self._contiguous_window(idx)


def collate_motion_codes(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "motion_codes": torch.stack([b["motion_codes"] for b in batch], dim=0),  # [B, L, Q]
        "segment_ids": torch.stack([b["segment_ids"] for b in batch], dim=0),    # [B, L]
        "positions": torch.stack([b["positions"] for b in batch], dim=0),        # [B, L]
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),      # [B, L]
    }
