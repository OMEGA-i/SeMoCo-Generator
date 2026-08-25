"""SeMoCo-Generator (our T2M model) inference adapter.

Emits native ``motion_codes``; decoding to ``soma77`` is the conversion graph's
job (using this model's tokenizer + canonical anchor, exposed via
:attr:`SemocoModel.tokenizer` / :attr:`SemocoModel.anchor`).
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ...dataset import T2MCodeDataset, collate_t2m
from ...paths import humanml3d_root
from ...tokenizer_bridge import FrozenMotionTokenizer, canonical_anchor
from ..rollout import generate_from_text, load_model
from ..schema import ModelInput, ModelOutput, MotionClip
from .base import MotionModel
from .registry import MODEL_SCHEMAS


class SemocoModel(MotionModel):
    schema = MODEL_SCHEMAS["semoco"]

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        tokenizer_checkpoint: str | Path,
        codes_root: str | Path | None = None,
        text_encoder: str | None = None,
        device: str = "cuda:0",
        fk_device: str = "cpu",
        max_tok: int = 125,
        eos_thresh: float = 1.01,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._checkpoint_path = str(checkpoint)
        self._tokenizer_path = str(tokenizer_checkpoint)
        self.model, self.checkpoint_meta = load_model(checkpoint, self.device)
        self.tokenizer = FrozenMotionTokenizer.load(tokenizer_checkpoint, device=self.device)

        # torch.compile disabled — CUDAGraphs cache a separate graph for each
        # KV-cache shape (1..125 tokens), consuming 19+ GiB GPU memory and
        # causing OOM on H100 80 GB with batch=128+CFG.
        # The unified-batch design (all clips share one global_max_tok) is the
        # primary throughput win; eager mode is stable and predictable.

        self.fps = float(self.tokenizer.spec.source_fps)
        self.fk_device = fk_device
        self.max_tok = int(max_tok)
        # The eval protocol forces decode length to match the GT-derived target
        # duration for every model, regardless of what a model would "naturally"
        # pick. Our own EOS head can trigger well before `target_tok` (observed
        # ~20% of SOMA clips stopping at roughly half the requested length),
        # which would not be comparable to other rows in the table:
        # text-conditioned retrieval especially penalizes a clip that only
        # covers the first half of the described action. Default
        # `eos_thresh > 1.0` is unreachable by a sigmoid probability, so decoding
        # always runs the full `target_tok` packets -- pass e.g. 0.5 explicitly to
        # get the model's own natural stopping behavior back (ablation / demo use).
        self.eos_thresh = float(eos_thresh)
        self.anchor = self._load_fallback_anchor(codes_root)
        self._text_encoder_key = (
            text_encoder
            or (self.checkpoint_meta.get("data_meta") or {}).get("encode_key")
            or self.checkpoint_meta.get("text_encoder_key")
            or "flan"
        )
        self.dataset = None
        self._live_text_encoder = None
        self._hml_anchors: dict[str, dict[str, np.ndarray]] | None = None
        if codes_root:
            self.dataset = T2MCodeDataset(
                codes_root, "test", text_encoder_key=self._text_encoder_key, max_motion_tok=self.max_tok,
            )

    def weight_signature(self) -> str:
        from .registry import weight_signature

        # Delegates to the free-function form (the single source of truth --
        # see its docstring) so this can never desync from what the CPU
        # aggregation pass recomputes without loading the model.
        return weight_signature(
            "semoco", checkpoint=self._checkpoint_path,
            tokenizer_checkpoint=self._tokenizer_path, max_tok=self.max_tok,
            eos_thresh=self.eos_thresh,
        )

    # ------------------------------------------------------------------
    def estimate_generation_memory(self, batch_size: int, max_tok: int, *, cfg_scale: float | None = 3.0) -> dict:
        """Return estimated memory footprint for a generation batch.

        Used by runners to pre-check GPU headroom before calling ``generate()``.
        With CFG (classifier-free guidance), the effective batch size doubles.
        """
        from ..resource_guard import estimate_kv_cache_bytes as _est

        eff_batch = int(batch_size) * (2 if (cfg_scale is not None and cfg_scale != 1.0) else 1)
        m = self.model.cfg
        kv_bytes = _est(
            num_heads=getattr(m, "num_kv_heads", getattr(m, "num_heads", 0)),
            head_dim=getattr(m, "head_dim", 0),
            num_layers=getattr(m, "num_layers", getattr(m, "n_layers", 0)),
            batch_size=eff_batch,
            max_seq_len=int(max_tok),
        )
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return {"kv_cache_bytes": kv_bytes, "param_bytes": param_bytes, "eff_batch": eff_batch}

    def _get_live_text_encoder(self):
        if self._live_text_encoder is None:
            from ...text.registry import get_encoder_cls

            cls = get_encoder_cls(self._text_encoder_key)
            self._live_text_encoder = cls.load(device=str(self.device))
            print(f"[semoco] live text encoder loaded: {self._text_encoder_key}", flush=True)
        return self._live_text_encoder

    def _text_batch(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dataset is not None:
            rows = {
                str(self.dataset.entry(i).get("caption", "")): i
                for i in range(len(self.dataset))
            }
            missing = [p for p in prompts if p not in rows]
            if not missing:
                batch = collate_t2m([self.dataset[rows[p]] for p in prompts])
                return batch["text_emb"], batch["text_valid"]
            print(
                f"[semoco] {len(missing)}/{len(prompts)} prompts missing from store; "
                f"falling back to live {self._text_encoder_key} encode",
                flush=True,
            )
        enc = self._get_live_text_encoder()
        emb, mask = enc.encode(list(prompts))
        return emb.to(self.device), mask.to(self.device)

    def _target_tok(self, duration_s: float) -> int:
        token_rate = float(self.fps) / 4.0
        target = int(round(float(duration_s) * token_rate))
        return max(1, min(int(self.max_tok), target))

    @staticmethod
    def _load_fallback_anchor(codes_root: str | Path | None) -> dict[str, np.ndarray]:
        """Load a dataset-mean anchor from *codes_root*, falling back to canonical.

        The dataset-mean anchor is precomputed from the T2M train split and stored
        as ``canonical_mean_anchor.npy`` in the code store.  It provides a much
        better first-frame pose than the identity canonical anchor: root Y ≈ 0.87 m
        (standing height) instead of 0 (ground level), and realistic joint rotations
        instead of all-identity.
        """
        if codes_root is not None:
            from ...local_uri import resolve_local_uri
            root = resolve_local_uri(str(codes_root))
            mean_path = root / "canonical_mean_anchor.npy"
            if mean_path.is_file():
                flat = np.load(mean_path).astype(np.float32)
                return {
                    "init_root_pos": flat[0:3].copy(),
                    "init_root_rot6d": flat[3:9].copy(),
                    "init_joints76_rot6d": flat[9:465].reshape(76, 6).copy(),
                    "identity_coeffs": np.zeros((10,), dtype=np.float32),
                }
        return canonical_anchor(identity_dim=10)

    def _load_hml_anchors(self) -> dict[str, dict[str, np.ndarray]] | None:
        """Load precomputed per-clip anchors for HumanML3D eval (SMPL-fitting based).

        Looks for ``hml_per_clip_anchors.pkl`` alongside the HumanML3D data root.
        The file is produced by ``python -m semoco_generator.tools.precompute_hml_anchors`` and provides
        100% anchor coverage for the HML test split.

        These are GT-derived anchors from the first frame of HML ground-truth
        joints.  This is the correct eval strategy: the model generates motion
        *codes* from text only (no anchor in the generation path), and the anchor
        is used solely as the FK initial condition — exactly as during training.
        """
        if self._hml_anchors is not None:
            return self._hml_anchors if len(self._hml_anchors) > 0 else None
        import pickle as _pickle
        candidates = [
            humanml3d_root() / "hml_per_clip_anchors.pkl",
        ]
        for p in candidates:
            path = Path(p)
            if path.is_file():
                with open(path, "rb") as f:
                    self._hml_anchors = _pickle.load(f)
                print(f"[semoco] loaded {len(self._hml_anchors)} precomputed HML anchors from {path}",
                      flush=True)
                return self._hml_anchors
        self._hml_anchors = {}
        return None

    def build_prompt_anchor_map(
        self, prompts: list[tuple[str, str, str]]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Build ``{prompt_id: anchor_dict}`` from per-clip anchors in the code store.

        *prompts* is a list of ``(prompt_id, rec_id, caption)`` tuples from the
        track inputs. Matching is multi-level across **all splits**
        (train + test + val) of the code store:

        1. **rec_id** (motion-level) — primary strategy.
        2. **caption exact** (clip-level) — first fallback.
        3. **caption case-insensitive** — second fallback.

        Unmatched prompts are silently omitted from the returned dict; the caller
        (``_resolve_anchor``) falls back to ``canonical_anchor``.

        The returned anchor dict has the same keys as :func:`canonical_anchor`:
        ``init_root_pos [3], init_root_rot6d [6], init_joints76_rot6d [76,6],
        identity_coeffs [identity_dim]``.
        """
        if self.dataset is None:
            return {}

        import json as _json

        # ---- Load anchor indices from ALL splits ----
        # Each split has its own anchors/identities arrays, so we store
        # (split_name, row) tuples instead of just row indices.
        rec_id_to_loc: dict[str, tuple[str, int]] = {}
        caption_to_loc: dict[str, tuple[str, int]] = {}
        caption_lower_to_loc: dict[str, tuple[str, int]] = {}

        for split in ("train", "test", "val"):
            index_path = self.dataset.root / f"{split}.index.json"
            if not index_path.is_file():
                continue
            entries = _json.loads(index_path.read_text())
            for entry in entries:
                row = int(entry.get("row", 0))
                rid = str(entry.get("rec_id", "")).strip()
                cap = str(entry.get("caption", "")).strip()
                if rid and rid not in rec_id_to_loc:
                    rec_id_to_loc[rid] = (split, row)
                if cap:
                    if cap not in caption_to_loc:
                        caption_to_loc[cap] = (split, row)
                    cap_lower = cap.lower()
                    if cap_lower not in caption_lower_to_loc:
                        caption_lower_to_loc[cap_lower] = (split, row)

        # ---- Load anchors/identities arrays lazily per split ----
        _split_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        def _get_arrays(split_name: str) -> tuple[np.ndarray, np.ndarray]:
            if split_name not in _split_arrays:
                anc = np.load(self.dataset.root / f"{split_name}.anchor.npy", mmap_mode="r")
                idn = np.load(self.dataset.root / f"{split_name}.identity.npy", mmap_mode="r")
                _split_arrays[split_name] = (anc, idn)
            return _split_arrays[split_name]

        result: dict[str, dict[str, np.ndarray]] = {}
        matched_by_rec_id = 0
        matched_by_exact_caption = 0
        matched_by_lower_caption = 0
        matched_by_hml = 0

        for prompt_id, rec_id, caption in prompts:
            loc: tuple[str, int] | None = None

            # Layer 1: rec_id match
            if rec_id:
                loc = rec_id_to_loc.get(rec_id.strip())
                if loc is not None:
                    matched_by_rec_id += 1

            # Layer 2: exact caption match
            if loc is None and caption:
                loc = caption_to_loc.get(caption.strip())
                if loc is not None:
                    matched_by_exact_caption += 1

            # Layer 3: case-insensitive caption match
            if loc is None and caption:
                loc = caption_lower_to_loc.get(caption.strip().lower())
                if loc is not None:
                    matched_by_lower_caption += 1

            if loc is None:
                # Layer 4: Precomputed HML GT anchors (SMPL-fitting based, 100% coverage)
                hml = self._load_hml_anchors()
                if hml is not None:
                    anchor = hml.get(prompt_id)
                    if anchor is not None:
                        matched_by_hml += 1
                        result[prompt_id] = {
                            k: v.copy() for k, v in anchor.items()
                        }
                continue

            split_name, row = loc
            anchors_arr, identities_arr = _get_arrays(split_name)
            flat = np.asarray(anchors_arr[row], dtype=np.float32)
            ident = np.asarray(identities_arr[row], dtype=np.float32)
            result[prompt_id] = {
                "init_root_pos": flat[0:3].copy(),
                "init_root_rot6d": flat[3:9].copy(),
                "init_joints76_rot6d": flat[9:465].reshape(76, 6).copy(),
                "identity_coeffs": ident.copy(),
            }

        total = len(prompts)
        matched = matched_by_rec_id + matched_by_exact_caption + matched_by_lower_caption + matched_by_hml
        unmatched = total - matched
        parts = [f"rec_id={matched_by_rec_id}", f"caption={matched_by_exact_caption}"]
        if matched_by_lower_caption:
            parts.append(f"lower={matched_by_lower_caption}")
        if matched_by_hml:
            parts.append(f"hml={matched_by_hml}")
        print(
            f"[semoco] anchor map: {matched}/{total} matched "
            f"({matched / total * 100:.1f}%)"
            + (f" — {', '.join(parts)}" if matched else "")
            + (f", {unmatched} using canonical fallback" if unmatched else ""),
            flush=True,
        )
        return result

    def _sample_generators(self, seed: int, count: int) -> list[torch.Generator]:
        dev = self.device if self.device.type == "cuda" else torch.device("cpu")
        gens: list[torch.Generator] = []
        for _ in range(count):
            gen = torch.Generator(device=dev)
            gen.manual_seed(int(seed))
            gens.append(gen)
        return gens

    # ------------------------------------------------------------------
    def generate(self, inputs: Sequence[ModelInput]) -> list[ModelOutput]:
        inputs = list(inputs)
        if not inputs:
            return []
        # Group by seed so each forward uses one deterministic seed.
        by_seed: dict[int, list[int]] = {}
        for idx, inp in enumerate(inputs):
            by_seed.setdefault(int(inp.seed), []).append(idx)

        results: list[ModelOutput | None] = [None] * len(inputs)
        for seed, idxs in by_seed.items():
            prompts = [inputs[i].text for i in idxs]
            # Batch text encode once for all samples in this seed.
            text_emb, text_valid = self._text_batch(prompts)
            # Compute per-sample target lengths and group only by cfg_scale.
            # Per-duration bucketing was the primary cause of 40% GPU util:
            # 128 clips spread across ~50 different target_tok values produced
            # batches of 1-3 samples each, where kernel-launch overhead dominates.
            # Now we batch ALL clips of the same cfg_scale together, generate to
            # the maximum target length, and trim each sample's output to its
            # individual target.  Mixed-attention masks keep samples independent
            # so short clips do not affect long ones.
            per_sample: list[tuple[int, int, float]] = []  # (local_idx, global_idx, eff_cfg)
            target_toks: dict[int, int] = {}               # global_idx -> target_tok
            for local, i in enumerate(idxs):
                inp = inputs[i]
                target_toks[i] = self._target_tok(inp.length.to_seconds(self.fps))
                eff_cfg = 3.0 if inp.cfg_scale is None else float(inp.cfg_scale)
                per_sample.append((local, i, eff_cfg))
            # Bucket by cfg_scale only (typically a single bucket).
            by_cfg: dict[float, list[tuple[int, int, int]]] = {}
            for local, i, eff_cfg in per_sample:
                by_cfg.setdefault(eff_cfg, []).append((local, i, target_toks[i]))
            for eff_cfg, members in by_cfg.items():
                local_idx = [local for local, _, _ in members]
                global_idx = [i for _, i, _ in members]
                member_targets = [tt for _, _, tt in members]
                global_max_tok = min(max(member_targets), int(self.max_tok))
                seqs = generate_from_text(
                    self.model,
                    text_emb[local_idx],
                    text_valid[local_idx],
                    max_tok=global_max_tok,
                    cfg_scale=eff_cfg,
                    eos_thresh=self.eos_thresh,
                    device=self.device,
                    generators=self._sample_generators(int(seed), len(members)),
                )
                for seq, i, tt in zip(seqs, global_idx, member_targets):
                    inp = inputs[i]
                    # Trim to the sample's individual target length.
                    codes = seq[:tt].cpu().numpy().astype(np.int64)
                    clip = None
                    if len(codes):
                        clip = MotionClip(rep="motion_codes", array=codes, fps=self.fps)
                    results[i] = ModelOutput(
                        model_id="semoco",
                        prompt_id=inp.prompt_id,
                        seed=int(seed),
                        native_motion=clip,
                        status="ok" if clip is not None else "failed",
                        error=None if clip is not None else "empty code sequence",
                        provenance={
                            "target_tok": tt,
                            "gen_tok": int(codes.shape[0]) if len(codes) else 0,
                            "max_tok": self.max_tok,
                            "global_max_tok": global_max_tok,
                            "eos_thresh": self.eos_thresh,
                            "effective_cfg": eff_cfg,
                            "anchor": "resolved_at_conversion",
                            "text_source": self._text_encoder_key,
                            "batch_size": len(members),
                        },
                    )
        return [r for r in results if r is not None]

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.anchor = None
        self._live_text_encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["SemocoModel"]
