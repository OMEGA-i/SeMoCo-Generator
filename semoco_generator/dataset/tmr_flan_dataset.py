"""``TMRFlanDataset`` — paired (motion latent, Flan-T5 text embedding) dataset.

Reads pre-computed motion latents from ``<data-root>/tmr_soma_flan/`` and pre-computed
Flan-T5 per-token text embeddings from the T2M code store.  No Flan-T5 model is
loaded — text embeddings are read via mmap directly.

Collation right-pads text to ``Lmax`` (matches Flan-T5 live-encoding convention).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..local_uri import resolve_local_uri


class TMRFlanDataset(Dataset):
    """Paired (motion_latent, text_emb) for TMR text-encoder contrastive training.

    Parameters
    ----------
    codes_root
        Path to the T2M code store (e.g. ``local://t2m_codes``).
    motion_latents_root
        Directory containing ``{split}.motion_latents.npy`` (``local://`` ok).
    split
        ``"train"``, ``"val"``, or ``"test"``.
    text_encoder_key
        Suffix key for text embeddings (default ``"flan"``).
    """

    def __init__(
        self,
        codes_root: str | Path,
        motion_latents_root: str | Path,
        split: str = "train",
        *,
        text_encoder_key: str = "flan",
    ) -> None:
        self.codes_root = resolve_local_uri(codes_root)
        self.split = split
        self.motion_latents_root = resolve_local_uri(motion_latents_root)

        # ---- Motion latents: [N, 256] float32, fully in RAM ----
        motion_path = self.motion_latents_root / f"{split}.motion_latents.npy"
        if not motion_path.is_file():
            raise FileNotFoundError(
                f"Motion latents not found at {motion_path}. "
                f"Run `python -m semoco_generator.tools.precompute_tmr_motion_latents` first."
            )
        self.motion_latents = np.load(motion_path)  # [N, 256] fp32
        print(f"[tmr_flan] loaded motion latents: {self.motion_latents.shape}  "
              f"({self.motion_latents.nbytes / 1e6:.1f} MB)")

        # ---- Text embeddings: mmap [sum_L, 2048] fp16 ----
        suffix = f".{text_encoder_key}" if text_encoder_key else ""
        text_path = self.codes_root / f"{split}.text_emb{suffix}.npy"
        text_index_path = self.codes_root / f"{split}.text_index{suffix}.json"
        meta_path = self.codes_root / f"{split}.meta{suffix}.json"

        if not text_path.is_file():
            raise FileNotFoundError(f"Text embeddings not found: {text_path}")

        self.text_emb = np.load(text_path, mmap_mode="r")  # [sum_L, 2048] fp16
        self.text_index = json.loads(text_index_path.read_text())

        meta = json.loads(meta_path.read_text())
        self.clip_dim = int(meta["clip_dim"])
        self.encode_key = str(meta.get("encode_key", text_encoder_key))
        self.text_model_id = str(meta.get("text_model_id", ""))

        # ---- Shared index for filtering ----
        shared_index_path = self.codes_root / f"{split}.index.json"
        self.shared_index = json.loads(shared_index_path.read_text())

        # Filter clips that are too short (motion < 2 tokens)
        self._keep = [
            i for i, e in enumerate(self.shared_index)
            if int(e.get("code_len", 0)) >= 2
        ]

        # Verify alignment
        n_motion = len(self.motion_latents)
        n_text = len(self.text_index)
        n_shared = len(self.shared_index)
        assert n_motion == n_text == n_shared, (
            f"Mismatch: motion={n_motion} text_index={n_text} shared_index={n_shared}"
        )

    @property
    def meta(self) -> dict:
        """Metadata dict for swanlab logging (accessed by base_trainer)."""
        return {
            "split": self.split,
            "clip_dim": self.clip_dim,
            "encode_key": self.encode_key,
            "text_model_id": self.text_model_id,
            "num_clips": len(self),
            "num_total": len(self.shared_index),
        }

    def __len__(self) -> int:
        return len(self._keep)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        real = self._keep[idx]
        ti = self.text_index[real]
        L = int(ti["text_len"])
        ts = int(ti["text_start"])

        # Slice text embedding from mmap → [L, 2048] fp32
        text_emb = torch.from_numpy(
            np.asarray(self.text_emb[ts : ts + L], dtype=np.float32)
        )
        motion_latent = torch.from_numpy(self.motion_latents[real])  # [256]

        return {
            "motion_latent": motion_latent,
            "text_emb": text_emb,
            "text_len": L,
        }


def collate_tmr_flan(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Right-pad text to ``Lmax``, stack motion latents.

    Right-padding matches Flan-T5 live-encoding convention (tokens first, pad
    tokens last), which is what ``ACTORStyleEncoder`` expects via
    ``length_to_mask()``.
    """
    B = len(batch)
    dim = batch[0]["text_emb"].shape[-1]  # 2048
    Lmax = max(int(b["text_emb"].shape[0]) for b in batch)

    text_emb = torch.zeros(B, Lmax, dim, dtype=torch.float32)
    text_valid = torch.zeros(B, Lmax, dtype=torch.bool)
    motion_latent = torch.zeros(B, 256, dtype=torch.float32)

    for i, b in enumerate(batch):
        L = int(b["text_emb"].shape[0])
        # RIGHT-pad: real tokens first, pad tokens last
        text_emb[i, :L] = b["text_emb"]
        text_valid[i, :L] = True
        motion_latent[i] = b["motion_latent"]

    return {
        "text_emb": text_emb,
        "text_valid": text_valid,
        "motion_latent": motion_latent,
    }


__all__ = ["TMRFlanDataset", "collate_tmr_flan"]
