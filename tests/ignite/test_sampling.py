import torch

from tokamak_foundation_model.ignite.sampling import SamplerConfig, apply_top_p


def test_temp_for_scalar_and_dict():
    assert SamplerConfig().temp_for("mhr") == 1.0
    s = SamplerConfig(temperature={"mhr": 0.7})
    assert s.temp_for("mhr") == 0.7
    assert s.temp_for("absent_modality") == 1.0     # falls back to 1.0


def test_apply_top_p_is_identity_when_none():
    p = torch.tensor([[0.5, 0.3, 0.2]])
    assert torch.equal(apply_top_p(p, None), p)


def test_apply_top_p_keeps_nucleus_and_renormalizes():
    p = torch.tensor([[0.6, 0.3, 0.08, 0.02]])
    out = apply_top_p(p, 0.9)
    assert out[0, 2] == 0 and out[0, 3] == 0        # tail dropped
    assert torch.isclose(out.sum(), torch.tensor(1.0))
    assert torch.isclose(out[0, 0] / out[0, 1], torch.tensor(2.0))  # ratios preserved


def test_apply_top_p_always_keeps_at_least_one_token():
    p = torch.tensor([[0.99, 0.01]])
    out = apply_top_p(p, 0.1)                        # threshold below the top prob
    assert (out > 0).sum() == 1 and torch.isclose(out.sum(), torch.tensor(1.0))


def test_rank_normalize_is_monotone_and_bounded():
    from tokamak_foundation_model.ignite.sampling import rank_normalize
    c = torch.tensor([[0.9, 0.1, 0.5]])
    r = rank_normalize(c)
    assert r.min() >= 0 and r.max() <= 1
    assert r[0, 0] > r[0, 2] > r[0, 1]              # order preserved


def test_rank_normalize_equalizes_scales_across_modalities():
    """A 64k-vocab modality has structurally smaller probabilities than a 1k one;
    rank normalization must make their confidences comparable."""
    from tokamak_foundation_model.ignite.sampling import rank_normalize
    small = torch.tensor([[0.002, 0.001, 0.004]])   # 64k-vocab scale
    large = torch.tensor([[0.20, 0.10, 0.40]])      # 1k-vocab scale
    assert torch.allclose(rank_normalize(small), rank_normalize(large))


def test_global_pool_defers_low_confidence_modality():
    """Validity smoke test: a pooled decode still produces a complete, in-vocab frame.

    Despite the name (kept from the plan), this asserts shape and code range only — it exercises
    the ``global_pool=True`` path end-to-end without crashing. The deferral BEHAVIOUR is covered
    by the direct ``_global_reveal`` tests below, which compare allocations against the fixed
    quota on hand-built confidences.
    """
    from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
    from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
    from tokamak_foundation_model.ignite.sampling import SamplerConfig

    cfg = DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 8, 5), ModalitySpec("b", "slowts", 8, 5)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=2,
        maskgit_decode_steps=4, actuator_dim=6)
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    past = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.k0_seed + 1, cfg.actuator_dim)
    out = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(2),
                            sampler=SamplerConfig(global_pool=True))
    for m in cfg.modalities:                         # still a complete, valid frame
        assert out[m.name].shape == (1, m.n_tok)
        assert (out[m.name] >= 0).all() and (out[m.name] < m.codebook_size).all()


def test_rank_normalize_breaks_ties_in_index_order():
    """Ties must resolve deterministically, not by whatever the sort backend happens to do.

    A stable ASCENDING sort places tied values in increasing index order, so among equal
    values the LOWER index takes the LOWER rank. Pinned exactly, because a non-stable sort
    would make reveal order device-dependent for equal-confidence tokens.
    """
    from tokamak_foundation_model.ignite.sampling import rank_normalize
    r = rank_normalize(torch.tensor([[0.5, 0.5, 0.1]]))
    assert torch.equal(r, torch.tensor([[0.5, 1.0, 0.0]]))
    assert r[0, 1] > r[0, 0]                         # tied pair: later index ranks higher
    # an entirely tied row still yields distinct, strictly index-ordered ranks (no rank reused)
    r4 = rank_normalize(torch.tensor([[0.2, 0.2, 0.2, 0.2]]))
    assert torch.equal(r4, torch.tensor([[0.0, 1 / 3, 2 / 3, 1.0]]))
    assert (r4[0, 1:] > r4[0, :-1]).all()


def _tiny_pool_model(n_a=6, n_b=4, steps=4, v_a=5, v_b=7):
    """Two-modality toy dynamics model, deterministically initialised."""
    from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
    from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
    cfg = DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", n_a, v_a), ModalitySpec("b", "slowts", n_b, v_b)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=2,
        maskgit_decode_steps=steps, actuator_dim=6)
    torch.manual_seed(0)
    return cfg, MaskGITDynamics(cfg).eval()


def test_global_pool_changes_the_decoded_frame_and_stays_valid():
    """The flag must actually be WIRED: same seed, same inputs, different committed codes.

    Guards the failure mode where ``global_pool=True`` is silently ignored — a shape/range-only
    check passes either way, so this compares against the default sampler for a fixed seed.
    """
    from tokamak_foundation_model.ignite.sampling import SamplerConfig
    cfg, mg = _tiny_pool_model()
    past = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok),
                                  generator=torch.Generator().manual_seed(4))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.k0_seed + 1, cfg.actuator_dim,
                      generator=torch.Generator().manual_seed(5))
    default = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(2))
    pooled = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(2),
                               sampler=SamplerConfig(global_pool=True))
    assert any(not torch.equal(default[n], pooled[n]) for n in default), \
        "global_pool=True produced the default frame — the flag is not wired through"
    for m in cfg.modalities:                          # still a complete, valid frame
        p = pooled[m.name]
        assert p.shape == (1, m.n_tok)
        assert (p >= 0).all() and (p < m.codebook_size).all()
        assert not (p == mg.backbone.tok.mask_ids[m.name]).any(), \
            "pooled decode left [MASK] ids behind — the schedule did not fully reveal"


def test_norm_log_confidence_levels_and_cross_vocab_comparability():
    """``1 + log(c)/log(V)``: 0 at uniform, 1 at certainty, and equal across vocab sizes.

    This is the property the global pool needs and pure rank normalization cannot give: a
    64k-vocab token and a 1k-vocab token that are equally confident RELATIVE to their own
    uniform baseline must score the same, even though their raw probabilities differ 8x.
    """
    from tokamak_foundation_model.ignite.sampling import norm_log_confidence
    for V in (1000, 65536):
        assert torch.allclose(norm_log_confidence(torch.tensor([1.0 / V]), V),
                              torch.tensor([0.0]), atol=1e-6)          # uniform -> 0
        assert float(norm_log_confidence(torch.tensor([0.999]), V)) > 0.999   # certain -> ~1
        assert float(norm_log_confidence(torch.tensor([1.0]), V)) == 1.0
    # same normalized level (halfway: c = V**-0.5) -> same score despite 8x raw-prob gap
    small_vocab = norm_log_confidence(torch.tensor([1000 ** -0.5]), 1000)
    large_vocab = norm_log_confidence(torch.tensor([65536 ** -0.5]), 65536)
    assert torch.allclose(small_vocab, large_vocab, atol=1e-6)
    assert torch.allclose(small_vocab, torch.tensor([0.5]), atol=1e-6)
    # monotone in c, and a zeroed probability stays finite rather than -inf
    c = torch.tensor([[0.0, 1e-9, 0.01, 0.5, 1.0]])
    out = norm_log_confidence(c, 1000)
    assert torch.isfinite(out).all()
    assert (out[0, 1:] > out[0, :-1]).all()
    # float16: the 1e-12 floor underflows to 0 in fp16, so the clamp must happen in fp32 or
    # log() returns -inf and the token sorts to the bottom of every pool forever.
    assert torch.isfinite(norm_log_confidence(torch.zeros(1, dtype=torch.float16), 1000)).all()
    assert torch.isfinite(norm_log_confidence(torch.zeros(1, dtype=torch.bfloat16), 65536)).all()


def test_global_pool_allocation_follows_modality_confidence():
    """The R1 lever: reveals follow WHICH MODALITY is confident, which quotas cannot do.

    Two 8-token modalities sharing a vocab. One is confident (c ~ 0.9), the other near its
    uniform baseline (c ~ 0.001 at V=1000). At this step the frame's budget is 5 reveals: the
    pool spends all 5 on the confident modality and lets the uncertain one defer entirely, and
    the allocation FLIPS when the confidences are swapped. The fixed quota is [2, 2] in both
    cases — blind to confidence by construction.

    Deterministic: hand-built confidences fed straight to the two policies, no model forward
    and no RNG.
    """
    from tokamak_foundation_model.ignite.maskgit import _cosine_keep_fractions
    cfg, mg = _tiny_pool_model(n_a=8, n_b=8, steps=4, v_a=1000, v_b=1000)
    names = [m.name for m in cfg.modalities]
    revealed = {m.name: torch.zeros(1, m.n_tok, dtype=torch.bool) for m in cfg.modalities}
    frac = _cosine_keep_fractions(cfg.maskgit_decode_steps)[1]
    total = cfg.tokens_per_frame
    budget = total - int(round(frac * total))
    assert budget == 5                                    # guards the fixture, not the policy

    confident = torch.linspace(0.85, 0.95, 8).unsqueeze(0)
    uncertain = torch.linspace(0.001, 0.002, 8).unsqueeze(0)   # ~1/V, i.e. no information
    a_hot = {"a": confident, "b": uncertain}
    b_hot = {"a": uncertain, "b": confident}

    def counts(take):
        return [int(take[n].sum()) for n in names]

    pool_a = counts(mg._global_reveal(a_hot, revealed, frac))
    pool_b = counts(mg._global_reveal(b_hot, revealed, frac))
    assert pool_a == [5, 0], f"confident modality should take the whole budget, got {pool_a}"
    assert pool_b == [0, 5], f"allocation must flip with the confidences, got {pool_b}"
    assert pool_a != pool_b                               # responsive, not fixed
    assert sum(pool_a) == sum(pool_b) == budget           # budget still honoured

    for conf in (a_hot, b_hot):                           # the quota ignores all of this
        q = mg._per_modality_reveal(conf, revealed, frac)
        assert [int(q[n].sum()) for n in names] == [2, 2]


def test_global_pool_scores_vocab_normalized_not_raw_probability():
    """A large-vocab modality must not be starved for having structurally smaller probabilities.

    Modality "a" has a 64k vocab and c = 0.01; "b" has a 1k vocab and c = 0.05 — five times the
    RAW probability. Relative to their own uniform baselines "a" is the more confident one
    (0.585 vs 0.566), so it must win the budget. Scoring raw probability would invert this.
    """
    from tokamak_foundation_model.ignite.maskgit import _cosine_keep_fractions
    from tokamak_foundation_model.ignite.sampling import norm_log_confidence
    cfg, mg = _tiny_pool_model(n_a=8, n_b=8, steps=4, v_a=65536, v_b=1000)
    names = [m.name for m in cfg.modalities]
    conf = {"a": torch.full((1, 8), 0.01), "b": torch.full((1, 8), 0.05)}
    assert conf["b"][0, 0] > conf["a"][0, 0]                        # b wins on raw probability
    assert float(norm_log_confidence(conf["a"][0, 0], 65536)) > \
        float(norm_log_confidence(conf["b"][0, 0], 1000))           # a wins once normalized
    revealed = {m.name: torch.zeros(1, m.n_tok, dtype=torch.bool) for m in cfg.modalities}
    frac = _cosine_keep_fractions(cfg.maskgit_decode_steps)[1]
    pool = mg._global_reveal(conf, revealed, frac)
    assert [int(pool[n].sum()) for n in names] == [5, 0], \
        "pool followed raw probability instead of vocab-normalized confidence"


def _tiny_rev():
    from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 6, 5), ModalitySpec("b", "slowts", 4, 5)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=2,
        maskgit_decode_steps=4, actuator_dim=6)


def test_revision_rounds_produce_a_valid_complete_frame():
    import torch
    from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
    from tokamak_foundation_model.ignite.sampling import SamplerConfig

    cfg = _tiny_rev()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    past = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.k0_seed + 1, cfg.actuator_dim)
    out = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(3),
                            sampler=SamplerConfig(revision_rounds=2, revision_frac=0.5))
    for m in cfg.modalities:
        assert out[m.name].shape == (1, m.n_tok)
        assert (out[m.name] >= 0).all() and (out[m.name] < m.codebook_size).all()


def test_revision_can_change_committed_tokens():
    """A revision round must actually be able to revise — otherwise it is a no-op."""
    import torch
    from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
    from tokamak_foundation_model.ignite.sampling import SamplerConfig

    cfg = _tiny_rev()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    past = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.k0_seed + 1, cfg.actuator_dim)
    base = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(3),
                             sampler=SamplerConfig())
    rev = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(3),
                            sampler=SamplerConfig(revision_rounds=3, revision_frac=0.9))
    assert any(not torch.equal(base[m.name], rev[m.name]) for m in cfg.modalities)
