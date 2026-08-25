"""HumanML3D ``text_mot_match`` evaluator (classic co-embedding protocol)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .word_vectorizer import POS_enumerator
from .hml_match_modules import MotionEncoderBiGRUCo, MovementConvEncoder, TextEncoderBiGRUCo
from .protocol import DEFAULT_CHECKPOINT, MAX_MOTION_LENGTH, MAX_TEXT_LEN, UNIT_LENGTH


class TextMotMatchEvaluator:
    """Frozen HumanML3D ``text_mot_match`` co-embedding model.

    Motion encoding works with only the evaluator checkpoint. Text encoding
    needs a HumanML ``WordVectorizer`` (GloVe) if R-precision / Matching /
    t2m_sim are required; FID / Diversity can run motion-only.
    """

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        *,
        device: str = "cuda:0",
        word_vectorizer=None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        unit_length: int = UNIT_LENGTH,
        max_motion_length: int = MAX_MOTION_LENGTH,
        max_text_len: int = MAX_TEXT_LEN,
        official_protocol: bool = True,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"text_mot_match checkpoint not found: {self.checkpoint}")
        ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        dim_pose = 263
        dim_word = 300
        dim_pos = len(POS_enumerator)
        dim_movement_enc_hidden = 512
        dim_movement_latent = 512
        dim_text_hidden = 512
        dim_motion_hidden = 1024
        dim_coemb = 512

        self.movement = MovementConvEncoder(dim_pose - 4, dim_movement_enc_hidden, dim_movement_latent)
        self.text = TextEncoderBiGRUCo(dim_word, dim_pos, dim_text_hidden, dim_coemb, device=self.device)
        self.motion = MotionEncoderBiGRUCo(dim_movement_latent, dim_motion_hidden, dim_coemb, device=self.device)
        self.movement.load_state_dict(ckpt["movement_encoder"])
        self.text.load_state_dict(ckpt["text_encoder"])
        self.motion.load_state_dict(ckpt["motion_encoder"])
        self.movement.to(self.device).eval()
        self.text.to(self.device).eval()
        self.motion.to(self.device).eval()

        if word_vectorizer is None:
            try:
                from .word_vectorizer import load_word_vectorizer

                word_vectorizer = load_word_vectorizer()
            except FileNotFoundError:
                word_vectorizer = None
        self.word_vectorizer = word_vectorizer
        self.unit_length = int(unit_length)
        self.max_motion_length = int(max_motion_length)
        self.max_text_len = int(max_text_len)
        self.official_protocol = bool(official_protocol)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def normalize(self, motion: np.ndarray) -> np.ndarray:
        x = np.asarray(motion, dtype=np.float32)
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / np.clip(self.std, 1e-8, None)

    def _prepare_motion(self, motion: np.ndarray, length: int | None) -> tuple[np.ndarray, int]:
        x = self.normalize(motion)
        if length is None:
            length = int(x.shape[0])
        length = max(1, min(int(length), int(x.shape[0]), self.max_motion_length))
        # Official Text2MotionDatasetV2 floors to unit_length before pad.
        if self.official_protocol:
            length = max(self.unit_length, (length // self.unit_length) * self.unit_length)
            length = min(length, self.max_motion_length)
        x = x[:length]
        if self.official_protocol and length < self.max_motion_length:
            pad = np.zeros((self.max_motion_length - length, x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)
        return x.astype(np.float32), length

    @torch.inference_mode()
    def encode_motion(
        self,
        motions: Sequence[np.ndarray],
        lengths: Sequence[int] | None = None,
        *,
        batch_size: int = 256,
    ) -> np.ndarray:
        prepared: list[np.ndarray] = []
        m_lens: list[int] = []
        for i, motion in enumerate(motions):
            req_len = None if lengths is None else int(lengths[i])
            x, m_len = self._prepare_motion(motion, req_len)
            prepared.append(x)
            m_lens.append(m_len)
        if not prepared:
            return np.zeros((0, 512), dtype=np.float32)

        # Under the official protocol every clip is padded to max_motion_length,
        # so a batched forward is numerically identical to the per-clip loop.
        # The non-official path keeps clips unpadded, so fall back to per-clip.
        if not self.official_protocol:
            return self._encode_motion_loop(prepared, m_lens)

        embs: list[np.ndarray] = []
        C = prepared[0].shape[1]
        for s in range(0, len(prepared), max(1, int(batch_size))):
            xb = prepared[s : s + batch_size]
            lb = m_lens[s : s + batch_size]
            arr = np.stack(xb).astype(np.float32)  # [B, max_motion_length, C]
            x_t = torch.from_numpy(arr[..., : C - 4]).float().to(self.device)
            movements = self.movement(x_t)  # [B, T', 512]
            tp = int(movements.shape[1])
            mov_lens = torch.tensor(
                [max(1, min(int(m) // self.unit_length, tp)) for m in lb],
                device=self.device, dtype=torch.long,
            )
            emb = self.motion(movements, mov_lens)  # [B, 512]
            embs.append(emb.float().cpu().numpy())
        return np.concatenate(embs, axis=0).astype(np.float32)

    @torch.inference_mode()
    def _encode_motion_loop(self, prepared: list[np.ndarray], m_lens: list[int]) -> np.ndarray:
        embs: list[np.ndarray] = []
        for x, m_len in zip(prepared, m_lens):
            x_t = torch.from_numpy(x[None, ..., :-4]).float().to(self.device)
            movements = self.movement(x_t)
            mov_len = max(1, int(movements.shape[1]))
            mov_len = min(mov_len, int(movements.shape[1]))
            lengths_t = torch.tensor([mov_len], device=self.device, dtype=torch.long)
            emb = self.motion(movements, lengths_t)[0]
            embs.append(emb.float().cpu().numpy())
        return np.stack(embs).astype(np.float32)

    def encode_text(self, captions: Sequence[str], *, tokens: Sequence[Sequence[str]] | None = None) -> np.ndarray:
        if self.word_vectorizer is None:
            raise RuntimeError(
                "text_mot_match text encoding requires a HumanML WordVectorizer (GloVe). "
                "Pass word_vectorizer=... or run motion-only metrics (FID/Diversity)."
            )
        tokens_batch: list[list[str]] = []
        for i, caption in enumerate(captions):
            if tokens is not None and i < len(tokens) and tokens[i]:
                toks = [str(t) for t in tokens[i] if t]
            else:
                # Prefer HumanML POS tokens embedded as ``word/POS`` in the caption
                # string; otherwise fall back to ``word/OTHER`` (weaker).
                toks = []
                for raw in str(caption).strip().split():
                    if "/" in raw and raw.split("/")[-1] in POS_enumerator:
                        toks.append(raw)
                    else:
                        toks.append(f"{raw.lower()}/OTHER")
            if self.official_protocol:
                # Official Text2MotionDatasetV2 wraps with sos/eos and pads to max_text_len+2.
                body = toks[: self.max_text_len]
                wrapped = ["sos/OTHER"] + body + ["eos/OTHER"]
                while len(wrapped) < self.max_text_len + 2:
                    wrapped.append("unk/OTHER")
                tokens_batch.append(wrapped)
            else:
                tokens_batch.append((toks[:20] if len(toks) > 20 else toks) or ["unk/OTHER"])
        max_len = max(len(t) for t in tokens_batch)
        word_embs = np.zeros((len(tokens_batch), max_len, 300), dtype=np.float32)
        pos_ohot = np.zeros((len(tokens_batch), max_len, len(POS_enumerator)), dtype=np.float32)
        cap_lens: list[int] = []
        for i, toks in enumerate(tokens_batch):
            # Official uses true caption length including sos/eos (before unk pad).
            if self.official_protocol:
                # Find first unk padding start if present after sos/eos construction.
                true_len = len(toks)
                for j, tok in enumerate(toks):
                    if tok == "unk/OTHER" and j >= 2:
                        true_len = j
                        break
                cap_lens.append(max(2, true_len))
            else:
                cap_lens.append(len(toks))
            for j, tok in enumerate(toks):
                try:
                    w, p = self.word_vectorizer[tok]
                except Exception:
                    w, p = self.word_vectorizer["unk/OTHER"]
                word_embs[i, j] = np.asarray(w, dtype=np.float32)
                pos_ohot[i, j] = np.asarray(p, dtype=np.float32)
        order = np.argsort(cap_lens)[::-1].copy()
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        we = torch.from_numpy(word_embs[order]).to(self.device)
        pe = torch.from_numpy(pos_ohot[order]).to(self.device)
        lengths = torch.tensor([cap_lens[i] for i in order], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            emb = self.text(we, pe, lengths).float().cpu().numpy()
        return emb[inv].astype(np.float32)

    def compute_metrics(
        self,
        generated: Sequence[np.ndarray],
        ground_truth: Sequence[np.ndarray],
        captions: Sequence[str] | None = None,
        *,
        retrieval_protocol: str = "full_gallery",
        seed: int = 0,
        gen_lengths: Sequence[int] | None = None,
        gt_lengths: Sequence[int] | None = None,
    ) -> dict[str, float]:
        from .score import score_embeddings

        gen_emb = self.encode_motion(generated, lengths=gen_lengths)
        gt_emb = self.encode_motion(ground_truth, lengths=gt_lengths)
        text_emb = None
        if captions is not None:
            try:
                text_emb = self.encode_text(captions)
            except RuntimeError:
                text_emb = None
        return score_embeddings(
            gen_emb,
            gt_emb,
            text=text_emb,
            retrieval_protocol=retrieval_protocol,
            seed=seed,
        )


# Prefer the protocol name in public APIs.
HumanMLEvaluator = TextMotMatchEvaluator

__all__ = ["HumanMLEvaluator", "TextMotMatchEvaluator"]
