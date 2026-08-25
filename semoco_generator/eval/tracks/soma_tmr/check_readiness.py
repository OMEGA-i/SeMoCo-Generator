"""Validate a T2M store (+ optional checkpoint alignment) for SOMA/TMR evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from ....local_uri import resolve_local_uri


def _default_tokenizer() -> Path | None:
    try:
        from ....paths import default_checkpoint

        return default_checkpoint()
    except Exception:  # noqa: BLE001
        return None


def check_readiness(
    codes_root: str | Path,
    split: str = "test",
    *,
    text_encoder: str | None = None,
    checkpoint: str | Path | None = None,
    tokenizer: str | Path | None = None,
) -> list[str]:
    root = resolve_local_uri(codes_root)
    missing: list[str] = []
    for name in (
        f"{split}.codes.npy",
        f"{split}.index.json",
        f"{split}.anchor.npy",
        f"{split}.identity.npy",
        f"{split}.meta.json",
    ):
        if not (root / name).is_file():
            missing.append(str(root / name))

    # Text embeddings: prefer suffixed, allow legacy unsuffixed.
    text_ok = False
    if text_encoder:
        text_ok = (root / f"{split}.text_emb.{text_encoder}.npy").is_file() or (
            root / f"{split}.text_emb.npy"
        ).is_file()
        if not text_ok:
            missing.append(str(root / f"{split}.text_emb.{text_encoder}.npy"))
    else:
        text_ok = any(
            (root / f"{split}.text_emb.{k}.npy").is_file() for k in ("flan", "siglip", "qwen3")
        ) or (root / f"{split}.text_emb.npy").is_file()
        if not text_ok:
            missing.append(str(root / f"{split}.text_emb.<flan|siglip|qwen3>.npy"))

    meta_path = root / f"{split}.meta.json"
    store_clip_dim = None
    store_tok = None
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            store_tok = meta.get("tokenizer_checkpoint")
        except json.JSONDecodeError:
            missing.append(f"{meta_path} (invalid json)")
    # Encoder-specific meta may carry clip_dim.
    for key in ((text_encoder,) if text_encoder else ("flan", "siglip", "qwen3", None)):
        suffix = f".{key}" if key else ""
        enc_meta = root / f"{split}.meta{suffix}.json"
        if enc_meta.is_file():
            try:
                store_clip_dim = json.loads(enc_meta.read_text()).get("clip_dim", store_clip_dim)
                break
            except json.JSONDecodeError:
                pass

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            missing.append(str(ckpt_path))
        else:
            try:
                import torch

                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                ckpt_key = (
                    (ckpt.get("data_meta") or {}).get("encode_key")
                    or ckpt.get("text_encoder_key")
                )
                if text_encoder and ckpt_key and ckpt_key != text_encoder:
                    missing.append(
                        f"text_encoder mismatch: store/arg={text_encoder} ckpt={ckpt_key}"
                    )
                mcfg = ckpt.get("model_config") or {}
                ckpt_dim = mcfg.get("clip_dim") or mcfg.get("text_dim")
                if store_clip_dim is not None and ckpt_dim is not None and int(store_clip_dim) != int(ckpt_dim):
                    missing.append(
                        f"clip_dim mismatch: store={store_clip_dim} ckpt={ckpt_dim}"
                    )
            except Exception as exc:  # noqa: BLE001
                missing.append(f"checkpoint unreadable: {exc}")

    # Mirror the runner's resolution order. The path recorded in the store meta
    # is whatever absolute path the export machine used, so it failing to exist
    # is only fatal when no other tokenizer is reachable.
    if not any(c and Path(c).is_file() for c in (tokenizer, _default_tokenizer(), store_tok)):
        missing.append(
            "no usable tokenizer checkpoint: pass --semoco-tokenizer or set "
            f"$SOMA_TOKENIZER_CHECKPOINT (store recorded: {store_tok})"
        )

    return missing


__all__ = ["check_readiness"]
