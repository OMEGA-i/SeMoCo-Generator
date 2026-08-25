"""Load and wrap the frozen SeMoCo codec.

THIS IS THE ONLY MODULE THAT TOUCHES THE TOKENIZER REPO.
Everything else (model / dataset / train / rollout) is self-contained and runs
off the exported ``.npy`` code store with no external dependency. Keep the
``soma`` surface quarantined here so the project stays independent.

The checkpoint produced by ``experiments.train_native`` is self-contained: it
carries ``codec_config`` (to rebuild the model), ``net`` (state dict), and
``norm`` (packed mean/std). We mirror ``tools/eval.py::_load_native_codec`` but
expose a small, framework-agnostic API:

    bridge = FrozenMotionTokenizer.load(checkpoint, device="cuda")
    codes  = bridge.encode(features)            # [T, 499] physical -> [T_tok, Q] int64
    rec    = bridge.decode(codes)               # [T_tok, Q] -> [T, 499] physical
    out    = bridge.decode_to_joints(codes, anchor_npz=...)   # -> SOMA77 FK joints

All ``soma`` imports below are deferred (inside functions) and guarded by
``ensure_tokenizer_on_path()`` so importing this module never requires the
tokenizer repo to be present; only calling these methods does.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .paths import default_checkpoint, ensure_tokenizer_on_path


def _cfg_kwargs(cls: type, raw: dict) -> dict:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in allowed}


@dataclass
class TokenizerSpec:
    """Static description of the frozen codec, useful for model config."""

    num_codebooks: int
    codebook_size: int
    temporal_stride: int
    source_fps: float
    checkpoint: str

    @property
    def token_rate(self) -> float:
        return self.source_fps / float(self.temporal_stride)


class FrozenMotionTokenizer:
    """Thin wrapper around a frozen single-stream StructuredVQTokenizer."""

    def __init__(self, model, mean: torch.Tensor, std: torch.Tensor, spec: TokenizerSpec, device: torch.device):
        self._model = model
        self._mean = mean
        self._std = std
        self.spec = spec
        self.device = device

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        checkpoint: str | Path | None = None,
        *,
        device: str | torch.device = "cuda",
        source_fps: float = 50.0,
    ) -> "FrozenMotionTokenizer":
        ensure_tokenizer_on_path()
        from models.umr.structured_vq import (  # noqa: WPS433  (deferred import)
            BackboneConfig,
            CodecConfig,
            GroupConfig,
            QuantizerConfig,
            build_structured_vq,
        )

        ckpt_path = Path(checkpoint) if checkpoint is not None else default_checkpoint()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"tokenizer checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cc = ckpt["codec_config"]
        if cc.get("groups"):
            raise ValueError(
                "SeMoCo-Generator expects a single-stream tokenizer (uniform [T, Q] "
                f"packet); checkpoint {ckpt_path} is part-wise multi-branch."
            )
        cfg = CodecConfig(
            backbone=BackboneConfig(**_cfg_kwargs(BackboneConfig, cc["backbone"])),
            quantizer=QuantizerConfig(**_cfg_kwargs(QuantizerConfig, cc["quantizer"])),
            input_dim=int(cc.get("input_dim", 499)),
            groups=None,
        )
        model = build_structured_vq(cfg)
        net = ckpt["net"]
        model_keys = set(model.state_dict().keys())
        model.load_state_dict({k: v for k, v in net.items() if k in model_keys})
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        dev = torch.device(device if (torch.cuda.is_available() or "cuda" not in str(device)) else "cpu")
        model.to(dev)

        mean = torch.from_numpy(np.asarray(ckpt["norm"]["mean"], dtype=np.float32)).view(1, -1, 1).to(dev)
        std = torch.from_numpy(np.asarray(ckpt["norm"]["std"], dtype=np.float32)).view(1, -1, 1).to(dev)

        # Determine actual codebook count from a test encode — the config's
        # ``num_quantizers`` may not include a semantic VQ layer that the
        # tokenizer appends, so we trust the output shape.
        with torch.no_grad():
            test_input = torch.zeros(1, cfg.input_dim, cfg.backbone.temporal_stride * 4, device=dev)
            test_codes = model.encode_indices(test_input)
            actual_q = int(test_codes.shape[-1])

        spec = TokenizerSpec(
            num_codebooks=actual_q,
            codebook_size=int(cc["quantizer"]["codebook_size"]),
            temporal_stride=int(cc["backbone"]["temporal_stride"]),
            source_fps=float(source_fps),
            checkpoint=str(ckpt_path),
        )
        return cls(model, mean, std, spec, dev)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------
    @property
    def stride(self) -> int:
        return self.spec.temporal_stride

    def _trim(self, length: int) -> int:
        stride = self.stride
        if stride <= 1:
            return length
        return (length // stride) * stride

    @torch.no_grad()
    def encode(self, features: np.ndarray) -> np.ndarray:
        """``features [T, 499]`` (physical units) -> ``codes [T_tok, Q]`` int64."""
        feats = np.asarray(features, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[1] != 499:
            raise ValueError(f"features must be [T, 499]; got {feats.shape}")
        keep = self._trim(feats.shape[0])
        if keep <= 0:
            return np.zeros((0, self.spec.num_codebooks), dtype=np.int64)
        feats = feats[:keep]
        t = torch.from_numpy(feats).to(self.device).unsqueeze(0).transpose(1, 2)  # [1, 499, T]
        t_norm = (t - self._mean) / self._std
        indices = self._model.encode_indices(t_norm)
        if isinstance(indices, dict):
            raise ValueError("part-wise tokenizer is not supported by FrozenMotionTokenizer")
        return indices.squeeze(0).to(torch.int64).cpu().numpy()  # [T_tok, Q]

    @torch.no_grad()
    def encode_batch(self, feats_list: list[np.ndarray]) -> list[np.ndarray]:
        """Batched codec encode: ``list[[T_i,499]]`` -> ``list[[T_tok_i, Q]]`` int64.

        One codec forward over a padded ``[B,499,Tmax]`` tensor (amortizes the
        per-clip Python/launch/transfer overhead). Each clip is trimmed to a
        stride multiple, padded to the batch-max length by **edge replication**
        (repeat the last frame), and its output is trimmed back to its own token
        count. The longest clip in a batch has zero padding (exact); shorter clips
        differ from a standalone encode only in at most the final token(s) within
        the conv receptive field over the replicated tail — kept negligible by
        length-bucketing the caller's batches.
        """
        n = len(feats_list)
        out: list[np.ndarray] = [
            np.zeros((0, self.spec.num_codebooks), dtype=np.int64) for _ in range(n)
        ]
        keeps = [self._trim(int(np.asarray(f).shape[0])) for f in feats_list]
        valid = [i for i in range(n) if keeps[i] > 0]
        if not valid:
            return out
        stride = self.stride
        Tmax = max(keeps[i] for i in valid)
        mean = self._mean.squeeze(0)          # [499, 1]
        std = self._std.squeeze(0)            # [499, 1]
        batch = torch.empty(len(valid), 499, Tmax, device=self.device, dtype=torch.float32)
        for bi, i in enumerate(valid):
            k = keeps[i]
            f = np.asarray(feats_list[i], dtype=np.float32)[:k]        # [k, 499]
            t = torch.from_numpy(f).to(self.device).transpose(0, 1)   # [499, k]
            t = (t - mean) / std
            batch[bi, :, :k] = t
            if k < Tmax:
                batch[bi, :, k:] = t[:, k - 1 : k]                    # edge replicate
        indices = self._model.encode_indices(batch)                   # [B, Tmax/stride, Q]
        if isinstance(indices, dict):
            raise ValueError("part-wise tokenizer is not supported by FrozenMotionTokenizer")
        indices = indices.to(torch.int64).cpu().numpy()
        for bi, i in enumerate(valid):
            out[i] = indices[bi, : keeps[i] // stride]
        return out

    @torch.no_grad()
    def decode_batch(self, codes_list: list[np.ndarray]) -> list[np.ndarray]:
        """Batched codec decode: ``list[[T_tok_i,Q]]`` -> ``list[[T_i,499]]``.

        WARNING: APPROXIMATE for shorter clips. Edge-replicate padding to the
        batch-max length can shift a shorter clip's features by ~0.1-0.2 vs a
        standalone decode (the codec's temporal receptive field spreads the
        padded tail). Do NOT use for ground-truth decoding; use per-clip
        :meth:`decode` / :meth:`decode_to_joints_arrays` when exactness matters.


        One codec forward over a padded ``[B, Tmax_tok, Q]`` index tensor (edge
        replicate the last token for shorter clips), then each clip's output is
        trimmed to its own frame count ``T_tok_i * stride``. The longest clip is
        exact; shorter clips differ only in at most the final frame(s) within the
        conv receptive field over the replicated tail (negligible, and callers
        length-bucket). Mirrors :meth:`encode_batch`.
        """
        from data.umr_schema import SLICE_FOOT_CONTACT  # noqa: WPS433

        n = len(codes_list)
        out: list[np.ndarray] = [np.zeros((0, 499), dtype=np.float32) for _ in range(n)]
        toks = [int(np.asarray(c).shape[0]) for c in codes_list]
        valid = [i for i in range(n) if toks[i] > 0]
        if not valid:
            return out
        q = self.spec.num_codebooks
        stride = self.stride
        tmax = max(toks[i] for i in valid)
        batch = torch.zeros(len(valid), tmax, q, dtype=torch.int64, device=self.device)
        for bi, i in enumerate(valid):
            c = torch.as_tensor(np.asarray(codes_list[i], dtype=np.int64), device=self.device)
            k = toks[i]
            batch[bi, :k] = c
            if k < tmax:
                batch[bi, k:] = c[k - 1 : k]  # edge replicate
        rec_norm = self._model.decode_indices(batch)          # [B, 499, tmax*stride]
        rec = rec_norm * self._std + self._mean
        rec[:, SLICE_FOOT_CONTACT] = rec_norm[:, SLICE_FOOT_CONTACT]
        rec = rec.transpose(1, 2).cpu().numpy()               # [B, tmax*stride, 499]
        for bi, i in enumerate(valid):
            out[i] = rec[bi, : toks[i] * stride].astype(np.float32)
        return out

    def _features_to_joints(
        self,
        features: np.ndarray,
        *,
        init_root_pos: np.ndarray,
        init_root_rot6d: np.ndarray,
        init_joints76_rot6d: np.ndarray,
        identity_coeffs: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """``features [T,499]`` + anchor -> ``joints77 [T-1,77,3]`` (anchor frame dropped)."""
        ensure_tokenizer_on_path()
        from data.umr_schema import CanonicalAnchor  # noqa: WPS433
        from data.umr_to_soma77 import materialize_features_matrices  # noqa: WPS433
        from data.soma77_fk import soma77_joints_world_xyz_from_matrices  # noqa: WPS433

        anchor = CanonicalAnchor(
            init_root_pos=np.asarray(init_root_pos, dtype=np.float32).reshape(3),
            init_root_rot6d=np.asarray(init_root_rot6d, dtype=np.float32).reshape(6),
            init_joints76_rot6d=np.asarray(init_joints76_rot6d, dtype=np.float32).reshape(76, 6),
        )
        identity = np.asarray(identity_coeffs, dtype=np.float32)
        if identity.ndim == 1:
            identity = identity.reshape(1, -1)
        mats = materialize_features_matrices(features, anchor)
        joints = soma77_joints_world_xyz_from_matrices(
            mats.rotmat77, mats.transl, identity, device=device,
        )
        return np.asarray(joints[1:], dtype=np.float32)

    def decode_prefix(self, codes: np.ndarray, num_codebooks: int) -> np.ndarray:
        """Decode using only the first ``num_codebooks`` codebook levels.

        This is a thin wrapper that prefix-truncates the codes tensor before
        passing it to :meth:`decode`. The tokenizer's
        ``StructuredVQTokenizer.decode_indices`` calls
        ``get_layer_embeddings(indices)`` which returns per-layer embeddings
        ``[B, T, K, C]`` and sums only the provided K layers. Passing
        ``codes[..., :num_codebooks]`` therefore naturally reconstructs from
        only the first N codebooks (coarse-to-fine).

        Args:
            codes: ``[T_tok, Q]`` int64 codes, where Q >= num_codebooks.
            num_codebooks: Number of codebook levels to use (1..Q).

        Returns:
            Reconstructed features ``[T, 499]`` float32 in physical units.
        """
        return self.decode(codes[..., :num_codebooks])

    @torch.no_grad()
    def decode(self, codes: np.ndarray | torch.Tensor) -> np.ndarray:
        """``codes [T_tok, Q]`` -> reconstructed ``features [T, 499]`` (physical)."""
        if isinstance(codes, np.ndarray):
            idx = torch.from_numpy(np.asarray(codes, dtype=np.int64))
        else:
            idx = codes.to(torch.int64)
        idx = idx.to(self.device)
        if idx.dim() == 2:
            idx = idx.unsqueeze(0)  # [1, T_tok, Q]
        from data.umr_schema import SLICE_FOOT_CONTACT  # noqa: WPS433

        rec_norm = self._model.decode_indices(idx)          # [1, 499, T]
        rec = rec_norm * self._std + self._mean
        rec[:, SLICE_FOOT_CONTACT] = rec_norm[:, SLICE_FOOT_CONTACT]
        return rec.transpose(1, 2).squeeze(0).cpu().numpy()  # [T, 499]

    # ------------------------------------------------------------------
    # Decode -> SOMA77 skeleton (FK). Kept here so the FK / UMR materialization
    # dependency on the tokenizer repo stays quarantined in this module.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_to_joints(
        self,
        codes: np.ndarray | torch.Tensor,
        *,
        anchor_npz: str | Path,
        device: str = "cpu",
    ) -> dict[str, np.ndarray]:
        """``codes [T_tok, Q]`` + a recording's ``umr499.npz`` anchor -> joints.

        Returns ``{joints77 [T,77,3], root [T,3], features [T,499]}``. The anchor
        (absolute frame-0 seed) and identity coefficients are read from
        ``anchor_npz``; the leading anchor frame is dropped so joints align with
        the reconstructed target frames (mirrors ``tools/dump_viewer_data.py``).
        """
        ensure_tokenizer_on_path()
        from data.umr_schema import CanonicalAnchor  # noqa: WPS433
        from data.umr_to_soma77 import materialize_features_matrices  # noqa: WPS433
        from data.soma77_fk import soma77_joints_world_xyz_from_matrices  # noqa: WPS433

        features = self.decode(codes)
        with np.load(Path(anchor_npz), allow_pickle=False) as data:
            anchor = CanonicalAnchor(
                init_root_pos=np.asarray(data["init_root_pos"], dtype=np.float32),
                init_root_rot6d=np.asarray(data["init_root_rot6d"], dtype=np.float32),
                init_joints76_rot6d=np.asarray(data["init_joints76_rot6d"], dtype=np.float32),
            )
            identity = np.asarray(data["identity_coeffs"], dtype=np.float32)
        mats = materialize_features_matrices(features, anchor)
        joints = soma77_joints_world_xyz_from_matrices(
            mats.rotmat77, mats.transl, identity, device=device,
        )
        return {
            "joints77": np.asarray(joints[1:], dtype=np.float32),
            "root": np.asarray(mats.transl[1:], dtype=np.float32),
            "features": features.astype(np.float32, copy=False),
        }

    @torch.no_grad()
    def decode_to_joints_arrays(
        self,
        codes: np.ndarray | torch.Tensor,
        *,
        init_root_pos: np.ndarray,
        init_root_rot6d: np.ndarray,
        init_joints76_rot6d: np.ndarray,
        identity_coeffs: np.ndarray,
        device: str = "cpu",
    ) -> dict[str, np.ndarray]:
        """Like :meth:`decode_to_joints` but takes anchor arrays directly.

        Used for text2motion generation (no source ``umr499.npz`` exists). The
        anchor / identity can come from the T2M store (GT) or a canonical seed
        (see :func:`canonical_anchor`).
        """
        ensure_tokenizer_on_path()
        from data.umr_schema import CanonicalAnchor  # noqa: WPS433
        from data.umr_to_soma77 import materialize_features_matrices  # noqa: WPS433
        from data.soma77_fk import soma77_joints_world_xyz_from_matrices  # noqa: WPS433

        features = self.decode(codes)
        anchor = CanonicalAnchor(
            init_root_pos=np.asarray(init_root_pos, dtype=np.float32).reshape(3),
            init_root_rot6d=np.asarray(init_root_rot6d, dtype=np.float32).reshape(6),
            init_joints76_rot6d=np.asarray(init_joints76_rot6d, dtype=np.float32).reshape(76, 6),
        )
        identity = np.asarray(identity_coeffs, dtype=np.float32)
        if identity.ndim == 1:
            identity = identity.reshape(1, -1)          # FK expects [1, C] or [T, C]
        mats = materialize_features_matrices(features, anchor)
        joints = soma77_joints_world_xyz_from_matrices(
            mats.rotmat77, mats.transl, identity, device=device,
        )
        return {
            "joints77": np.asarray(joints[1:], dtype=np.float32),
            "root": np.asarray(mats.transl[1:], dtype=np.float32),
            "features": features.astype(np.float32, copy=False),
        }

    @staticmethod
    def _materialize_full_pose_arrays(
        features: np.ndarray,
        *,
        init_root_pos: np.ndarray,
        init_root_rot6d: np.ndarray,
        init_joints76_rot6d: np.ndarray,
        identity_coeffs: np.ndarray,
        device: str,
    ) -> dict[str, np.ndarray]:
        """Materialize the full SOMA pose and validate it through SOMA FK.

        UMR features describe transitions *after* the absolute anchor frame.
        The returned arrays therefore drop that anchor frame, exactly as the
        historical ``decode_to_joints_arrays`` path does.  ``rotmat77`` keeps
        all 77 pose channels, including head, eyes, toe bases and terminal
        joints that cannot be inferred from a joints22 projection.
        """
        ensure_tokenizer_on_path()
        from data.umr_schema import CanonicalAnchor  # noqa: WPS433
        from data.umr_to_soma77 import materialize_features_matrices  # noqa: WPS433
        from data.soma77_fk import soma77_joints_world_xyz_from_matrices  # noqa: WPS433

        feats = np.asarray(features, dtype=np.float32)
        identity = np.asarray(identity_coeffs, dtype=np.float32)
        if identity.ndim == 1:
            identity = identity.reshape(1, -1)
        if identity.ndim != 2 or identity.shape[0] != 1:
            raise ValueError(
                f"identity_coeffs must be [C] or [1,C], got {identity.shape}"
            )
        if feats.ndim != 2 or feats.shape[1] != 499:
            raise ValueError(f"features must be [T,499], got {feats.shape}")
        if feats.shape[0] == 0:
            return {
                "rotmat77": np.zeros((0, 77, 3, 3), dtype=np.float32),
                "transl": np.zeros((0, 3), dtype=np.float32),
                "joints77": np.zeros((0, 77, 3), dtype=np.float32),
                "identity_coeffs": identity.astype(np.float32, copy=False),
                "foot_contacts": np.zeros((0, 4), dtype=np.float32),
            }

        anchor = CanonicalAnchor(
            init_root_pos=np.asarray(init_root_pos, dtype=np.float32).reshape(3),
            init_root_rot6d=np.asarray(init_root_rot6d, dtype=np.float32).reshape(6),
            init_joints76_rot6d=np.asarray(init_joints76_rot6d, dtype=np.float32).reshape(76, 6),
        )
        mats = materialize_features_matrices(feats, anchor)
        joints = soma77_joints_world_xyz_from_matrices(
            mats.rotmat77, mats.transl, identity, device=device,
        )
        return {
            "rotmat77": np.asarray(mats.rotmat77[1:], dtype=np.float32),
            "transl": np.asarray(mats.transl[1:], dtype=np.float32),
            "joints77": np.asarray(joints[1:], dtype=np.float32),
            "identity_coeffs": identity.astype(np.float32, copy=False),
            "foot_contacts": np.asarray(mats.foot_contacts[1:], dtype=np.float32),
        }

    @torch.no_grad()
    def decode_to_full_pose_arrays(
        self,
        codes: np.ndarray | torch.Tensor,
        *,
        init_root_pos: np.ndarray,
        init_root_rot6d: np.ndarray,
        init_joints76_rot6d: np.ndarray,
        identity_coeffs: np.ndarray,
        device: str = "cpu",
    ) -> dict[str, np.ndarray]:
        """Decode packets into full SOMA local rotations and root translation.

        Returns ``rotmat77 [T,77,3,3]``, ``transl [T,3]``, FK-validated
        ``joints77 [T,77,3]``, clip-static ``identity_coeffs [1,C]``, and
        ``foot_contacts [T,4]``. The pose convention is relative to the SOMA
        joint-orient basis, i.e. callers must use
        ``SOMALayer.pose(..., absolute_pose=False)``.
        """
        return self._materialize_full_pose_arrays(
            self.decode(codes),
            init_root_pos=init_root_pos,
            init_root_rot6d=init_root_rot6d,
            init_joints76_rot6d=init_joints76_rot6d,
            identity_coeffs=identity_coeffs,
            device=device,
        )

    @torch.no_grad()
    def decode_to_full_pose_arrays_batch(
        self,
        codes_list: list[np.ndarray],
        anchors_list: list[dict[str, np.ndarray]],
        identities_list: list[np.ndarray],
        *,
        device: str = "cuda",
        batch_size: int = 64,
    ) -> list[dict[str, np.ndarray]]:
        """Exactly decode a batch of Semoco clips into full SOMA poses.

        Decoder batches are grouped by token count.  That avoids the
        edge-replicated padding approximation in :meth:`decode_batch`, so a
        result is numerically equivalent to :meth:`decode_to_full_pose_arrays`
        for the same packet sequence and anchor.
        """
        if len(codes_list) != len(anchors_list) or len(codes_list) != len(identities_list):
            raise ValueError("codes_list, anchors_list, and identities_list must have equal length")

        results: list[dict[str, np.ndarray] | None] = [None] * len(codes_list)
        by_length: dict[int, list[int]] = {}
        for index, codes in enumerate(codes_list):
            count = int(np.asarray(codes).shape[0])
            by_length.setdefault(count, []).append(index)

        for token_count, indices in by_length.items():
            if token_count == 0:
                for index in indices:
                    anchor = anchors_list[index]
                    results[index] = self._materialize_full_pose_arrays(
                        np.zeros((0, 499), dtype=np.float32),
                        init_root_pos=anchor["init_root_pos"],
                        init_root_rot6d=anchor["init_root_rot6d"],
                        init_joints76_rot6d=anchor["init_joints76_rot6d"],
                        identity_coeffs=identities_list[index],
                        device=device,
                    )
                continue
            for start in range(0, len(indices), max(1, int(batch_size))):
                group = indices[start : start + max(1, int(batch_size))]
                features_list = self.decode_batch([codes_list[index] for index in group])
                for index, features in zip(group, features_list, strict=True):
                    anchor = anchors_list[index]
                    results[index] = self._materialize_full_pose_arrays(
                        features,
                        init_root_pos=anchor["init_root_pos"],
                        init_root_rot6d=anchor["init_root_rot6d"],
                        init_joints76_rot6d=anchor["init_joints76_rot6d"],
                        identity_coeffs=identities_list[index],
                        device=device,
                    )

        if any(result is None for result in results):
            raise RuntimeError("full-pose batch decode did not produce every requested result")
        return [result for result in results if result is not None]

    @torch.no_grad()
    def decode_to_joints_arrays_batch(
        self,
        codes_list: list[np.ndarray],
        anchors_list: list[dict[str, np.ndarray]],
        identities_list: list[np.ndarray],
        *,
        device: str = "cuda",
        batch_size: int = 64,
    ) -> list[dict[str, np.ndarray]]:
        """Batch decode + FK for multiple clips, grouped by equal code length.

        Clips with identical ``code_len`` are batched together for the VQ decode
        step — since all clips in a group have the same token count, zero padding
        is needed and the decode is exact (no edge-replicate contamination).

        FK is run per-clip on *device* (default GPU) to avoid the CPU bottleneck.

        Returns one dict per clip (same keys as :meth:`decode_to_joints_arrays`).
        """
        ensure_tokenizer_on_path()
        import torch  # noqa: WPS433
        from data.umr_schema import CanonicalAnchor  # noqa: WPS433
        from data.umr_to_soma77 import materialize_features_matrices  # noqa: WPS433
        from data.soma77_fk import soma77_joints_world_xyz_from_matrices  # noqa: WPS433

        n = len(codes_list)
        results: list[dict[str, np.ndarray] | None] = [None] * n

        # Group by code_len for exact zero-padding batch decode
        by_len: dict[int, list[int]] = {}
        for i, c in enumerate(codes_list):
            tok_len = int(np.asarray(c).shape[0])
            if tok_len > 0:
                by_len.setdefault(tok_len, []).append(i)

        for tok_len, idxs in by_len.items():
            for start in range(0, len(idxs), batch_size):
                group = idxs[start : start + batch_size]
                group_codes = [codes_list[i] for i in group]

                # Batch decode: same length → zero padding → exact
                features_list = self.decode_batch(group_codes)  # list[[T_i, 499]]

                for gi, i in enumerate(group):
                    feats = features_list[gi]
                    if feats.shape[0] == 0:
                        continue

                    a = anchors_list[i]
                    ident = np.asarray(identities_list[i], dtype=np.float32)
                    if ident.ndim == 1:
                        ident = ident.reshape(1, -1)

                    anchor = CanonicalAnchor(
                        init_root_pos=np.asarray(a["init_root_pos"], dtype=np.float32).reshape(3),
                        init_root_rot6d=np.asarray(a["init_root_rot6d"], dtype=np.float32).reshape(6),
                        init_joints76_rot6d=np.asarray(a["init_joints76_rot6d"], dtype=np.float32).reshape(76, 6),
                    )
                    mats = materialize_features_matrices(feats, anchor)
                    joints = soma77_joints_world_xyz_from_matrices(
                        mats.rotmat77, mats.transl, ident, device=device,
                    )
                    results[i] = {
                        "joints77": np.asarray(joints[1:], dtype=np.float32),
                        "root": np.asarray(mats.transl[1:], dtype=np.float32),
                        "features": feats.astype(np.float32, copy=False),
                    }

        # Any clip with zero tokens → empty result
        for i in range(n):
            if results[i] is None:
                results[i] = {
                    "joints77": np.zeros((0, 77, 3), dtype=np.float32),
                    "root": np.zeros((0, 3), dtype=np.float32),
                    "features": np.zeros((0, 499), dtype=np.float32),
                }

        return results  # type: ignore[return-value]


    @torch.no_grad()
    def decode_to_mesh_arrays(
        self,
        codes: np.ndarray | torch.Tensor,
        *,
        init_root_pos: np.ndarray,
        init_root_rot6d: np.ndarray,
        init_joints76_rot6d: np.ndarray,
        identity_coeffs: np.ndarray,
        low_lod: bool = True,
        device: str = "cpu",
    ) -> dict[str, np.ndarray]:
        """Like :meth:`decode_to_joints_arrays` but also returns the SOMA-X body MESH.

        Runs the same materialize + ``SOMALayer.pose`` FK as the joints path, but
        keeps ``fk_out["vertices"]`` (which the joints-only path discards) and the
        constant face topology. ``low_lod``: True -> 4505 verts / 9006 faces,
        False -> 18056 verts / 36108 faces. Returns
        ``{vertices [T,V,3], faces [F,3], joints [T,77,3]}`` (leading anchor frame
        dropped, matching :meth:`decode_to_joints_arrays`).
        """
        ensure_tokenizer_on_path()
        import torch  # noqa: WPS433
        from data.umr_schema import CanonicalAnchor  # noqa: WPS433
        from data.umr_to_soma77 import materialize_features_matrices  # noqa: WPS433
        from data.soma77_fk import (  # noqa: WPS433
            DEFAULT_IDENTITY_MODEL, _identity_betas, cached_soma_layer, default_soma_data_root,
        )

        features = self.decode(codes)
        anchor = CanonicalAnchor(
            init_root_pos=np.asarray(init_root_pos, dtype=np.float32).reshape(3),
            init_root_rot6d=np.asarray(init_root_rot6d, dtype=np.float32).reshape(6),
            init_joints76_rot6d=np.asarray(init_joints76_rot6d, dtype=np.float32).reshape(76, 6),
        )
        identity = np.asarray(identity_coeffs, dtype=np.float32)
        if identity.ndim == 1:
            identity = identity.reshape(1, -1)
        mats = materialize_features_matrices(features, anchor)
        dev = torch.device(device if (torch.cuda.is_available() and "cuda" in str(device)) else "cpu")
        layer = cached_soma_layer(
            str(default_soma_data_root().expanduser().resolve()),
            identity_model_type=DEFAULT_IDENTITY_MODEL, device=str(dev), low_lod=bool(low_lod),
        )
        soma = layer.soma
        T = int(np.asarray(mats.rotmat77).shape[0])
        betas = _identity_betas(soma, DEFAULT_IDENTITY_MODEL, identity, T, dev)
        poses_t = torch.tensor(np.asarray(mats.rotmat77, dtype=np.float32), device=dev).contiguous()
        transl_t = torch.tensor(np.asarray(mats.transl, dtype=np.float32), device=dev)
        with layer.lock, torch.no_grad():
            soma.prepare_identity(betas, None, global_scale=1.0)
            fk_out = soma.pose(poses_t, transl=transl_t, pose2rot=False)
        verts = fk_out["vertices"].detach().cpu().numpy().astype(np.float32, copy=False)
        joints = fk_out["joints"].detach().cpu().numpy().astype(np.float32, copy=False)
        if joints.shape[1] == 78:
            joints = joints[:, 1:]
        faces = np.asarray(soma.faces.detach().cpu().numpy(), dtype=np.int32)
        return {"vertices": verts[1:], "faces": faces, "joints": joints[1:]}


def canonical_anchor(identity_dim: int = 10) -> dict[str, np.ndarray]:
    """A neutral decode seed: origin root, identity rotations (rest pose).

    ``init_root_rot6d`` / each joint's ``rot6d`` are the identity 6-D rotation
    ``[1,0,0,0,1,0]`` (first two columns of I). Used when generating from text
    only (no ground-truth frame-0 pose is available).
    """
    eye6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    return {
        "init_root_pos": np.zeros((3,), dtype=np.float32),
        "init_root_rot6d": eye6.copy(),
        "init_joints76_rot6d": np.tile(eye6, (76, 1)),
        "identity_coeffs": np.zeros((identity_dim,), dtype=np.float32),
    }


def decode_codes_to_joints(
    tokenizer: FrozenMotionTokenizer,
    codes: np.ndarray,
    rec_id: str,
    *,
    recordings_root: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Full decode path: codes -> ``{joints77 [T,77,3], root [T,3], features [T,499]}``.

    Uses the recording's ``umr499.npz`` anchor for the frame-0 seed.
    """
    from .local_uri import resolve_local_uri
    root = resolve_local_uri(recordings_root) if recordings_root else resolve_local_uri("local://recordings")
    anchor_npz = root / rec_id / "umr499.npz"
    return tokenizer.decode_to_joints(codes, anchor_npz=anchor_npz, device=device)


def soma_skeleton_edges() -> np.ndarray:
    """``[[parent, child], ...]`` for the 76 non-root SOMA77 joints (for viewers)."""
    ensure_tokenizer_on_path()
    from data.soma77_fk import soma77_edges  # noqa: WPS433

    return np.asarray(soma77_edges(), dtype=np.int64)
