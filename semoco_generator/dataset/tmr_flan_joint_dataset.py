"""``TMRFlanJointDataset`` — paired (motion features, Flan-T5 text embedding).

Loads pre-computed motion features ``[T, 186]`` and pre-computed Flan-T5 text
embeddings ``[L, 2048]``, both via mmap.  Motion encoder processes features
directly (not pre-computed latents), enabling joint training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TMRFlanJointDataset(Dataset):
    def __init__(self, codes_root, features_root, split="train", text_encoder_key="flan"):
        from semoco_generator.local_uri import resolve_local_uri
        self.codes_root = resolve_local_uri(codes_root)
        self.split = split
        self.features_root = resolve_local_uri(features_root)

        # ---- Motion features: mmap [sum_T, 186] fp16 ----
        feat_path = self.features_root / f"{split}.motion_features.npy"
        idx_path = self.features_root / f"{split}.motion_index.json"
        if not feat_path.is_file():
            raise FileNotFoundError(f"Motion features not found: {feat_path}")
        self.motion_feat = np.load(feat_path, mmap_mode="r")
        self.motion_index = json.loads(idx_path.read_text())

        # ---- Text embeddings: mmap [sum_L, 2048] fp16 ----
        suffix = f".{text_encoder_key}" if text_encoder_key else ""
        text_path = self.codes_root / f"{split}.text_emb{suffix}.npy"
        text_index_path = self.codes_root / f"{split}.text_index{suffix}.json"
        meta_path = self.codes_root / f"{split}.meta{suffix}.json"

        self.text_emb = np.load(text_path, mmap_mode="r")
        self.text_index = json.loads(text_index_path.read_text())
        self.meta = json.loads(meta_path.read_text())
        self.clip_dim = int(self.meta["clip_dim"])

        # Shared index for filtering
        shared = json.loads((self.codes_root / f"{split}.index.json").read_text())
        self._keep = [i for i, e in enumerate(shared) if int(e.get("code_len", 0)) >= 2]

        # Verify alignment
        assert len(self.motion_index) == len(self.text_index), \
            f"Misaligned: motion={len(self.motion_index)} text={len(self.text_index)}"

    def __len__(self):
        return len(self._keep)

    def __getitem__(self, idx):
        real = self._keep[idx]
        mi = self.motion_index[real]
        ti = self.text_index[real]

        # Motion features [T, 186]
        ms, ml = mi["feat_start"], mi["feat_len"]
        mfeat = torch.from_numpy(
            np.asarray(self.motion_feat[ms:ms + ml], dtype=np.float32)
        )

        # Text embedding [L, 2048]
        ts, tl = ti["text_start"], ti["text_len"]
        temb = torch.from_numpy(
            np.asarray(self.text_emb[ts:ts + tl], dtype=np.float32)
        )

        return {"motion_feat": mfeat, "text_emb": temb,
                "motion_len": ml, "text_len": tl}


def collate_tmr_flan_joint(batch):
    """Right-pad both motion and text."""
    B = len(batch)
    tdim = batch[0]["text_emb"].shape[-1]
    mdim = batch[0]["motion_feat"].shape[-1]

    Lmax = max(b["text_emb"].shape[0] for b in batch)
    Tmax = max(b["motion_feat"].shape[0] for b in batch)

    text_emb = torch.zeros(B, Lmax, tdim, dtype=torch.float32)
    text_valid = torch.zeros(B, Lmax, dtype=torch.bool)
    motion_feat = torch.zeros(B, Tmax, mdim, dtype=torch.float32)
    motion_feat_valid = torch.zeros(B, Tmax, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b["text_emb"].shape[0]
        T = b["motion_feat"].shape[0]
        text_emb[i, :L] = b["text_emb"]
        text_valid[i, :L] = True
        motion_feat[i, :T] = b["motion_feat"]
        motion_feat_valid[i, :T] = True

    return {"text_emb": text_emb, "text_valid": text_valid,
            "motion_feat": motion_feat, "motion_feat_valid": motion_feat_valid}
