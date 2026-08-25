"""Autoregressive rollout for SeMoCo-Generator (offline + KV-cached online).

Loads a trained checkpoint and continues a motion-code prefix one packet at a
time, sampling each codebook with its own temperature / top-p. Returns the full
code sequence (prefix + generated) ready for ``tokenizer_bridge.decode_codes_to_joints``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..model import MotionGPT, MotionGPTConfig


@dataclass
class SamplingConfig:
    """Per-codebook sampling. Scalars broadcast to all codebooks."""

    temperature: list[float] | float = 0.9
    top_p: list[float] | float = 0.9
    top_k: list[int] | int = 0

    def temps(self, q: int) -> list[float]:
        t = self.temperature
        return [float(t)] * q if isinstance(t, (int, float)) else [float(x) for x in t]

    def tops(self, q: int) -> list[float]:
        p = self.top_p
        return [float(p)] * q if isinstance(p, (int, float)) else [float(x) for x in p]

    def topks(self, q: int) -> list[int]:
        k = self.top_k
        return [int(k)] * q if isinstance(k, int) else [int(x) for x in k]


def default_motion_sampling(q: int) -> SamplingConfig:
    """Qwen3-TTS-style split: warmer q0, cooler residual/detail codebooks."""
    temps = [0.9, 0.8, 0.8, 0.75] + [0.7] * max(0, q - 6) + [0.6, 0.55]
    temps = temps[:q] + [0.7] * max(0, q - len(temps))
    tops = [0.9] * q
    return SamplingConfig(temperature=temps[:q], top_p=tops, top_k=0)


def load_model(checkpoint: str | Path, device: str | torch.device = "cuda") -> tuple[MotionGPT, dict]:
    """Load a trained MotionGPT checkpoint onto ``device``."""
    if isinstance(device, str):
        dev = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    else:
        dev = device if torch.cuda.is_available() or device.type != "cuda" else torch.device("cpu")
    # Load weights straight onto the target device (eval path; no CPU staging).
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    mcfg = MotionGPTConfig(**ckpt["model_config"])
    model = MotionGPT(mcfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(dev)
    return model, ckpt


def _sample_one(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int = 0,
    *,
    generators: list[torch.Generator] | None = None,
) -> torch.Tensor:
    """``logits [B, V]`` -> sampled indices ``[B]`` (long).

    When ``generators`` is provided, sampling is performed row-by-row so each
    sample can own an independent RNG stream. This lets batched eval generation
    match per-prompt sampling semantics while still sharing the expensive model
    forward pass across prompts.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    if top_k > 0 and top_k < logits.shape[-1]:
        kth = torch.topk(logits, k=top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if generators is not None:
        if len(generators) != probs.shape[0]:
            raise ValueError(f"expected {probs.shape[0]} generators, got {len(generators)}")
        out: list[torch.Tensor] = []
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            keep = cdf - sorted_probs <= top_p
            sorted_probs = sorted_probs * keep
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            for i, gen in enumerate(generators):
                choice = torch.multinomial(sorted_probs[i], num_samples=1, generator=gen)
                out.append(sorted_idx[i].gather(0, choice))
        else:
            for i, gen in enumerate(generators):
                out.append(torch.multinomial(probs[i], num_samples=1, generator=gen))
        return torch.cat(out, dim=0)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf - sorted_probs <= top_p           # always keep top-1
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        choice = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idx.gather(-1, choice).squeeze(-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _sample_packet_from_hidden(
    model: MotionGPT,
    q0_logits: torch.Tensor,
    hidden: torch.Tensor,
    sampling: SamplingConfig,
    *,
    generators: list[torch.Generator] | None = None,
) -> torch.Tensor:
    """Sample one full packet with codebook-axis autoregression."""
    q = model.cfg.num_codebooks
    temps, tops, topks = sampling.temps(q), sampling.tops(q), sampling.topks(q)
    cols = [_sample_one(q0_logits[:, -1, :], temps[0], tops[0], topks[0], generators=generators)]
    for i in range(1, q):
        prefix = torch.stack(cols, dim=-1)
        logits_i = model.decode_packet(hidden[:, -1, :], prefix)
        cols.append(_sample_one(logits_i, temps[i], tops[i], topks[i], generators=generators))
    return torch.stack(cols, dim=-1)


@torch.no_grad()
def rollout(
    model: MotionGPT,
    prefix_codes: torch.Tensor,
    num_steps: int,
    *,
    sampling: SamplingConfig | None = None,
    device: str | torch.device | None = None,
    generators: list[torch.Generator] | None = None,
) -> torch.Tensor:
    """Continue ``prefix_codes [B, Tp, Q]`` for ``num_steps`` packets.

    Returns ``[B, Tp + num_steps, Q]`` (long), prefix included.
    """
    model.eval()
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    q = model.cfg.num_codebooks
    if sampling is None:
        sampling = default_motion_sampling(q)

    prefix = prefix_codes.to(dev).long()
    if prefix.dim() == 2:
        prefix = prefix.unsqueeze(0)
    B, Tp, Q = prefix.shape
    if Q != q:
        raise ValueError(f"prefix Q={Q} != model Q={q}")

    amp = torch.bfloat16 if dev.type == "cuda" else torch.float32
    caches = model.init_caches(B, dev, amp)
    # Prefill the prefix.
    with torch.autocast(device_type="cuda", dtype=amp, enabled=dev.type == "cuda"):
        logits, caches, hidden = model.forward_kv(prefix, caches, 0, return_hidden=True)
    nxt = _sample_packet_from_hidden(
        model, logits[0], hidden, sampling, generators=generators,
    ).unsqueeze(1)  # [B, 1, Q]

    generated = [prefix, nxt]
    pos = Tp
    for _ in range(num_steps - 1):
        with torch.autocast(device_type="cuda", dtype=amp, enabled=dev.type == "cuda"):
            logits, caches, hidden = model.forward_kv(nxt, caches, pos, return_hidden=True)
        nxt = _sample_packet_from_hidden(
            model, logits[0], hidden, sampling, generators=generators,
        ).unsqueeze(1)
        generated.append(nxt)
        pos += 1
    return torch.cat(generated, dim=1)  # [B, Tp + num_steps, Q]


# ---------------------------------------------------------------------------
# Text2Motion generation (text prefix prefill -> AR sample packets -> EOS)
# ---------------------------------------------------------------------------
_guide = MotionGPT.cfg_guide  # single source of truth for CFG logit combination


def _sample_packet_cfg(
    model: MotionGPT,
    hidden: torch.Tensor,      # [2B, 1, D] (rows 0:B cond, B:2B uncond) or [B,1,D]
    sampling: SamplingConfig,
    *,
    cfg_scale: float,
    n_cond: int,
    generators: list[torch.Generator] | None = None,
) -> torch.Tensor:
    """Sample one packet ``[B, Q]`` with codebook-axis AR + CFG on every codebook."""
    q = model.cfg.num_codebooks
    temps, tops, topks = sampling.temps(q), sampling.tops(q), sampling.topks(q)
    h = hidden[:, -1, :]                                       # [2B or B, D]
    use_cfg = cfg_scale != 1.0 and h.shape[0] == 2 * n_cond

    q0_all = model.decode_packet(h)                             # [*, V]
    if use_cfg:
        q0 = _guide(q0_all[:n_cond], q0_all[n_cond:], cfg_scale)
    else:
        q0 = q0_all[:n_cond]
    cols = [_sample_one(q0, temps[0], tops[0], topks[0], generators=generators)]      # [B]
    for i in range(1, q):
        prefix_cols = [c.repeat(2) if use_cfg else c for c in cols]
        prefix = torch.stack(prefix_cols, dim=-1)              # [* , i]
        logits_i = model.decode_packet(h, prefix)               # [*, V]
        gi = _guide(logits_i[:n_cond], logits_i[n_cond:], cfg_scale) if use_cfg else logits_i[:n_cond]
        cols.append(_sample_one(gi, temps[i], tops[i], topks[i], generators=generators))
    return torch.stack(cols, dim=-1)                           # [B, Q]


@torch.no_grad()
def generate_from_text_stream(
    model: MotionGPT,
    text_emb: torch.Tensor,
    text_valid: torch.Tensor,
    *,
    max_tok: int = 300,
    cfg_scale: float = 3.0,
    eos_thresh: float = 0.5,
    chunk: int = 4,
    sampling: SamplingConfig | None = None,
    device: str | torch.device | None = None,
    generators: list[torch.Generator] | None = None,
):
    """Streaming generation: yields the per-sample accumulated codes.

    Same mixed-attention prefill + AR + EOS as :func:`generate_from_text`, but
    ``yield``\\s the current ``list[[t, Q]]`` (per-sample, EOS-trimmed so far)
    every ``chunk`` packets, and once more at the end. Consume fully to get the
    complete result (that is exactly what :func:`generate_from_text` does).
    Used by the live viewer to decode + display motion as it is produced.
    """
    if not model.cfg.use_text:
        raise RuntimeError("generate_from_text requires cfg.use_text=True")
    model.eval()
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    q = model.cfg.num_codebooks
    if sampling is None:
        sampling = default_motion_sampling(q)

    text_emb = text_emb.to(dev).float()
    text_valid = text_valid.to(dev).bool()
    B, L, _ = text_emb.shape
    amp = torch.bfloat16 if dev.type == "cuda" else torch.float32
    use_cfg = cfg_scale != 1.0

    # Assemble prefill = [text (bidirectional), BOS]. For CFG, stack a null-text
    # branch of identical length so both branches batch together.
    text_x = model.project_text(text_emb)                                 # [B, L, D]
    if use_cfg:
        null = model.null_text.to(text_x.dtype).expand(B, L, -1)
        text_x = torch.cat([text_x, null], dim=0)                         # [2B, L, D]
        text_valid = torch.cat([text_valid, text_valid], dim=0)
    n_all = text_x.shape[0]
    bos = model.motion_bos.to(text_x.dtype).expand(n_all, 1, -1)
    prefill = torch.cat([text_x, bos], dim=1)                             # [*, L+1, D]

    role = torch.cat([
        torch.zeros(n_all, L, dtype=torch.long, device=dev),
        torch.ones(n_all, 1, dtype=torch.long, device=dev),               # BOS
    ], dim=1)
    key_valid = torch.cat([text_valid, torch.ones(n_all, 1, dtype=torch.bool, device=dev)], dim=1)
    mask = model.mixed_attn_mask(role, key_valid)

    caches = model.init_caches(n_all, dev, amp)
    with torch.autocast(device_type="cuda", dtype=amp, enabled=dev.type == "cuda"):
        caches, hidden = model.forward_embeds(prefill, attn_mask=mask, caches=caches,
                                              start_pos=0, run_decoder=False)
        # EOS at BOS position, then sample first packet.
        eos0 = torch.sigmoid(model.eos_head(hidden[:B, -1, :]).squeeze(-1))
        nxt = _sample_packet_cfg(
            model, hidden, sampling, cfg_scale=cfg_scale, n_cond=B, generators=generators,
        )  # [B, Q]

    seqs: list[list[torch.Tensor]] = [[] for _ in range(B)]
    finished = eos0 > eos_thresh                                          # [B] bool

    def _snapshot() -> list[torch.Tensor]:
        return [torch.stack(s, dim=0) if s else torch.zeros(0, q, dtype=torch.long, device=dev)
                for s in seqs]

    pos = L + 1
    for step in range(max_tok):
        for b in range(B):
            if not finished[b]:
                seqs[b].append(nxt[b])
        if bool(finished.all()):
            break
        feed = nxt.repeat(2, 1) if use_cfg else nxt                       # [*, Q]
        # Grow key_valid by one (the new motion token is always valid) and mask
        # padded text keys during this single-token decode step.
        key_valid = torch.cat([key_valid, torch.ones(n_all, 1, dtype=torch.bool, device=dev)], dim=1)
        step_mask = key_valid[:, None, None, :]                           # [*, 1, 1, total]
        with torch.autocast(device_type="cuda", dtype=amp, enabled=dev.type == "cuda"):
            feed_emb = model.embed(feed.unsqueeze(1))                     # [*, 1, D]
            caches, hidden = model.forward_embeds(feed_emb, attn_mask=step_mask,
                                                  caches=caches, start_pos=pos, run_decoder=False)
            eos = torch.sigmoid(model.eos_head(hidden[:B, -1, :]).squeeze(-1))
            nxt = _sample_packet_cfg(
                model, hidden, sampling, cfg_scale=cfg_scale, n_cond=B, generators=generators,
            )
        finished = finished | (eos > eos_thresh)
        pos += 1
        if (step + 1) % chunk == 0:
            yield _snapshot()
    yield _snapshot()


@torch.no_grad()
def generate_from_text(
    model: MotionGPT,
    text_emb: torch.Tensor,
    text_valid: torch.Tensor,
    *,
    max_tok: int = 300,
    cfg_scale: float = 3.0,
    eos_thresh: float = 0.5,
    sampling: SamplingConfig | None = None,
    device: str | torch.device | None = None,
    generators: list[torch.Generator] | None = None,
) -> list[torch.Tensor]:
    """Generate motion codes from text (mixed-attention prefix LM + EOS stop).

    ``text_emb``   ``[B, L, clip_dim]`` frozen Flan-T5 word embeddings.
    ``text_valid`` ``[B, L]`` bool.
    Returns a list of ``B`` tensors ``[T_b, Q]`` (per-sample, EOS-trimmed).
    Thin wrapper that consumes :func:`generate_from_text_stream` to the end.
    """
    out: list[torch.Tensor] | None = None
    for out in generate_from_text_stream(
        model, text_emb, text_valid, max_tok=max_tok, cfg_scale=cfg_scale,
        eos_thresh=eos_thresh, chunk=max(1, max_tok), sampling=sampling, device=device,
        generators=generators,
    ):
        pass
    if out is None:  # max_tok <= 0
        B = text_emb.shape[0]
        return [torch.zeros(0, model.cfg.num_codebooks, dtype=torch.long) for _ in range(B)]
    return out


# ---------------------------------------------------------------------------
# Online (streaming) prediction — stateful KV-cached generator
# ---------------------------------------------------------------------------
class OnlinePrediction:
    """Stateful single-stream packet generator with a persistent KV cache.

    Feed a prefix once via ``reset()``, then call ``step()`` to emit one packet
    at a time while reusing cached attention state.  Minimal v0 interface stub;
    shares sampling logic with the rest of this module.
    """

    def __init__(
        self,
        model,
        *,
        sampling: SamplingConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.q = model.cfg.num_codebooks
        self.sampling = sampling or default_motion_sampling(self.q)
        self._amp = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self._caches = None
        self._pos = 0

    @torch.no_grad()
    def reset(self, prefix_codes: torch.Tensor) -> torch.Tensor:
        """Prime the cache with a prefix; returns the first generated packet ``[B, Q]``."""
        prefix = prefix_codes.to(self.device).long()
        if prefix.dim() == 2:
            prefix = prefix.unsqueeze(0)
        B = prefix.shape[0]
        self._caches = self.model.init_caches(B, self.device, self._amp)
        with torch.autocast(device_type="cuda", dtype=self._amp, enabled=self.device.type == "cuda"):
            logits, self._caches, hidden = self.model.forward_kv(prefix, self._caches, 0, return_hidden=True)
        self._pos = prefix.shape[1]
        return _sample_packet_from_hidden(self.model, logits[0], hidden, self.sampling)

    @torch.no_grad()
    def step(self, last_packet: torch.Tensor) -> torch.Tensor:
        """Advance one packet. ``last_packet [B, Q]`` -> next packet ``[B, Q]``."""
        if self._caches is None:
            raise RuntimeError("call reset(prefix) before step()")
        nxt = last_packet.to(self.device).long()
        if nxt.dim() == 2:
            nxt = nxt.unsqueeze(1)
        with torch.autocast(device_type="cuda", dtype=self._amp, enabled=self.device.type == "cuda"):
            logits, self._caches, hidden = self.model.forward_kv(nxt, self._caches, self._pos, return_hidden=True)
        self._pos += 1
        return _sample_packet_from_hidden(self.model, logits[0], hidden, self.sampling)
