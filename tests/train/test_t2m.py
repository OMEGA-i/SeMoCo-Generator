"""Self-contained tests for the text2motion reform (no data / tokenizer needed).

Run::

    python -m pytest tests/test_t2m.py -q
    # or without pytest:
    python tests/test_t2m.py
"""

from __future__ import annotations

import torch

from semoco_generator.model import MotionGPT, MotionGPTConfig


def _small_cfg(use_text: bool) -> MotionGPTConfig:
    return MotionGPTConfig(
        num_codebooks=4, codebook_size=32,
        d_model=64, n_layers=2, n_heads=4, ffn_hidden=128,
        max_seq_len=512, code_pred_layers=2,
        use_text=use_text, clip_dim=16,
    )


def test_mixed_attn_mask_semantics():
    model = MotionGPT(_small_cfg(True))
    # 1 sample: 2 text (both valid) + BOS + 2 motion (all valid). S=5.
    role = torch.tensor([[0, 0, 1, 1, 1]])
    valid = torch.tensor([[True, True, True, True, True]])
    m = model.mixed_attn_mask(role, valid)[0, 0]  # [S, S]
    # text<->text bidirectional
    assert m[0, 1] and m[1, 0]
    # text must NOT attend motion
    assert not m[0, 2] and not m[1, 4]
    # motion attends all text
    assert m[2, 0] and m[4, 1]
    # motion causal among motion (BOS=2, m0=3, m1=4)
    assert m[3, 2] and m[4, 3] and m[4, 2]
    assert not m[2, 3] and not m[3, 4]
    # padded key masked out
    valid2 = torch.tensor([[True, False, True, True, True]])
    m2 = model.mixed_attn_mask(role, valid2)[0, 0]
    assert not m2[0, 1]      # key 1 is padding -> masked (off-diagonal)
    print("mixed_attn_mask semantics OK")


def test_t2m_train_step_and_backward():
    torch.manual_seed(0)
    cfg = _small_cfg(True)
    model = MotionGPT(cfg)
    B, L, Tm = 3, 5, 7
    text_emb = torch.randn(B, L, cfg.clip_dim)
    text_valid = torch.ones(B, L, dtype=torch.bool)
    text_valid[0, :2] = False   # left-padded example
    motion_codes = torch.randint(0, cfg.codebook_size, (B, Tm, cfg.num_codebooks))
    motion_valid = torch.ones(B, Tm, dtype=torch.bool)
    motion_valid[1, 4:] = False

    drop = torch.tensor([False, True, False])
    q_logits, eos_logits = model.t2m_train_step(text_emb, text_valid, motion_codes, motion_valid, drop_text=drop)
    assert len(q_logits) == cfg.num_codebooks
    assert q_logits[0].shape == (B, Tm, cfg.codebook_size)
    assert eos_logits.shape == (B, Tm + 1)

    loss, metrics = model.packet_ce_loss(
        model.cfg, q_logits, motion_codes, motion_valid
    )
    loss = loss + eos_logits.float().mean() * 0.0  # touch eos in graph
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if p.grad is not None]
    assert len(grads) > 0 and torch.isfinite(loss)
    # text_proj / null_text / motion_bos / eos_head must receive gradients.
    assert model.text_proj.weight.grad is not None
    assert model.motion_bos.grad is not None
    assert model.eos_head.weight.grad is not None
    print(f"t2m_train_step OK  loss={loss.item():.3f}")


def test_generate_from_text_shapes():
    from semoco_generator.eval.rollout import generate_from_text, SamplingConfig

    torch.manual_seed(0)
    cfg = _small_cfg(True)
    model = MotionGPT(cfg).eval()
    B, L = 2, 4
    text_emb = torch.randn(B, L, cfg.clip_dim)
    text_valid = torch.ones(B, L, dtype=torch.bool)
    text_valid[1, :1] = False
    seqs = generate_from_text(
        model, text_emb, text_valid, max_tok=12, cfg_scale=3.0,
        sampling=SamplingConfig(temperature=1.0, top_p=1.0), device="cpu",
    )
    assert len(seqs) == B
    for s in seqs:
        assert s.dim() == 2 and s.shape[-1] == cfg.num_codebooks
        assert s.shape[0] <= 12
    # cfg_scale=1 (no guidance) path
    seqs1 = generate_from_text(model, text_emb, text_valid, max_tok=8, cfg_scale=1.0, device="cpu")
    assert len(seqs1) == B
    # EOS effectively disabled (sigmoid < 2.0 always) -> loop must emit max_tok frames,
    # proving the prefill + KV-cache decode mechanics run end-to-end.
    seqs2 = generate_from_text(model, text_emb, text_valid, max_tok=10, cfg_scale=3.0,
                               eos_thresh=2.0, device="cpu")
    assert all(int(s.shape[0]) == 10 for s in seqs2), [int(s.shape[0]) for s in seqs2]
    print(f"generate_from_text OK  eos-on lens={[int(s.shape[0]) for s in seqs]}  "
          f"eos-off lens={[int(s.shape[0]) for s in seqs2]}")


def test_motion_only_regression():
    """The original motion-only path must still work unchanged."""
    torch.manual_seed(0)
    cfg = _small_cfg(False)
    model = MotionGPT(cfg)
    B, T = 2, 10
    codes = torch.randint(0, cfg.codebook_size, (B, T, cfg.num_codebooks))
    seg = torch.zeros(B, T, dtype=torch.long)
    pos = torch.arange(T).unsqueeze(0).expand(B, -1)
    logits = model.forward_packed(codes, codes, seg, pos)
    assert len(logits) == cfg.num_codebooks and logits[0].shape == (B, T, cfg.codebook_size)
    # KV-cache rollout path
    caches = model.init_caches(B, torch.device("cpu"), torch.float32)
    lg, caches, hidden = model.forward_kv(
        codes[:, :1], caches, 0, return_hidden=True
    )
    assert hidden.shape == (B, 1, cfg.d_model)
    print("motion-only regression OK")


def test_code_axis_causal_mask_polarity():
    """SDPA bool mask: True = allow. Code-axis must be lower-triangular.

    A ``triu`` mask (old True=blocked convention) becomes reverse-causal under
    current PyTorch SDPA and lets residual head q_k attend to emb(q_k) — the
    training target — collapsing q1..q{Q-2} accuracy to 1 within a few steps.
    """
    from semoco_generator.model.packet_decoder import PacketDecoder

    cfg = MotionGPTConfig(
        num_codebooks=8, codebook_size=64, d_model=64, n_layers=2, n_heads=4,
        ffn_hidden=128, max_seq_len=64, code_pred_layers=2,
    )
    dec = PacketDecoder(cfg)
    L = cfg.num_codebooks
    mask = dec._causal_mask(L, torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert mask.shape == (L, L)
    # Past + self allowed; future blocked.
    assert bool(mask[0, 0]) and not bool(mask[0, 1])
    assert bool(mask[-1].all())
    assert not bool(mask[1, 2])
    assert bool(mask[2, 0]) and bool(mask[2, 1]) and bool(mask[2, 2])
    print("code_axis_causal_mask polarity OK")


def test_code_axis_no_target_leak_short_train():
    """Regression: short teacher-forced train must NOT drive mid residuals to acc≈1.

    Under the reverse-causal leak, q1/q{Q-2} hit ~1.0 by step ~20 while q0 and
    q{Q-1} stay near chance. With a correct mask, mid residuals stay far below 1.
    """
    import torch.nn.functional as F
    from semoco_generator.model.packet_decoder import PacketDecoder

    torch.manual_seed(0)
    Q, V, D = 8, 64, 64
    cfg = MotionGPTConfig(
        num_codebooks=Q, codebook_size=V, d_model=D, n_layers=2, n_heads=4,
        ffn_hidden=128, max_seq_len=64, code_pred_layers=2,
    )
    dec = PacketDecoder(cfg).train()
    opt = torch.optim.AdamW(dec.parameters(), lr=3e-3)
    B, T = 16, 24

    def _accs():
        dec.eval()
        with torch.no_grad():
            h = torch.randn(B, T, D)
            t = torch.randint(0, V, (B, T, Q))
            logits = dec(h, target_codes=t)
            out = [(lg.argmax(-1) == t[..., i]).float().mean().item() for i, lg in enumerate(logits)]
        dec.train()
        return out

    for _ in range(30):
        h = torch.randn(B, T, D)
        t = torch.randint(0, V, (B, T, Q))
        logits = dec(h, target_codes=t)
        loss = sum(F.cross_entropy(lg.reshape(-1, V), t[..., i].reshape(-1)) for i, lg in enumerate(logits))
        opt.zero_grad()
        loss.backward()
        opt.step()

    accs = _accs()
    # Chance is ~1/V ≈ 0.016. Allow learning, but forbid the leak signature.
    assert accs[1] < 0.5, f"q1 acc={accs[1]:.3f} looks like target leak"
    assert accs[Q - 2] < 0.5, f"q{Q-2} acc={accs[Q-2]:.3f} looks like target leak"
    print(f"code_axis_no_target_leak OK  accs={[round(a, 3) for a in accs]}")


def test_greedy_codes_free_run():
    """Free-running codebook axis: shape OK; q0 matches TF q0 argmax."""
    from semoco_generator.model.packet_decoder import PacketDecoder

    torch.manual_seed(0)
    cfg = MotionGPTConfig(
        num_codebooks=4, codebook_size=32, d_model=64, n_layers=2, n_heads=4,
        ffn_hidden=128, max_seq_len=64, code_pred_layers=2,
    )
    dec = PacketDecoder(cfg).eval()
    B, T = 3, 5
    hidden = torch.randn(B, T, cfg.d_model)
    targets = torch.randint(0, cfg.codebook_size, (B, T, cfg.num_codebooks))
    with torch.no_grad():
        tf_logits = dec(hidden, target_codes=targets)
        roll = dec.greedy_codes(hidden)
    assert roll.shape == (B, T, cfg.num_codebooks)
    assert torch.equal(roll[..., 0], tf_logits[0].argmax(dim=-1))
    # Residuals free-run ≠ TF argmax in general (different prefixes).
    print(f"greedy_codes OK  roll_q0_match_tf={True}")


if __name__ == "__main__":
    test_mixed_attn_mask_semantics()
    test_t2m_train_step_and_backward()
    test_generate_from_text_shapes()
    test_motion_only_regression()
    test_code_axis_causal_mask_polarity()
    test_code_axis_no_target_leak_short_train()
    test_greedy_codes_free_run()
    print("\nALL T2M TESTS PASSED")
