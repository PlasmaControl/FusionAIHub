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
    """With a global pool, the confident modality reveals more tokens at step 1 than the
    uncertain one — impossible under per-modality fixed quotas."""
    import torch
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


def _tiny_pool_model(n_a=6, n_b=4, steps=4):
    """Two-modality toy dynamics model, deterministically initialised."""
    from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
    from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
    cfg = DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", n_a, 5), ModalitySpec("b", "slowts", n_b, 7)),
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


def test_global_pool_can_move_a_reveal_slot_between_modalities():
    """The pool budgets over the WHOLE frame, so it can shift a reveal slot BETWEEN modalities —
    something per-modality quotas structurally cannot do.

    Uses the production frame ratio (a 192-token spectro modality beside a 4-token slow-TS one)
    at decode step 0 of the real 10-step cosine schedule: the fixed quota gives both slots to
    spectro and starves slow-TS ([2, 0]), while the pool splits them ([1, 1]).

    Seed-free and exact — confidences are hand-built and no model forward runs. Verified
    invariant to confidence MAGNITUDE and to within-modality ordering.

    NOTE: this pins a *budgeting* difference, NOT confidence-based deferral. ``rank_normalize``
    maps every modality onto the same rank set, so the pooled allocation is currently
    independent of confidence magnitude; see the task-4 report. Kept deliberately narrow so it
    does not lock in that limitation.
    """
    from tokamak_foundation_model.ignite.maskgit import _cosine_keep_fractions
    cfg, mg = _tiny_pool_model(n_a=192, n_b=4, steps=10)
    names = [m.name for m in cfg.modalities]
    conf = {"a": torch.linspace(0.90, 0.99, 192).unsqueeze(0),     # confident modality
            "b": torch.linspace(0.001, 0.002, 4).unsqueeze(0)}     # uncertain modality
    revealed = {m.name: torch.zeros(1, m.n_tok, dtype=torch.bool) for m in cfg.modalities}
    frac = _cosine_keep_fractions(cfg.maskgit_decode_steps)[0]
    quota = mg._per_modality_reveal(conf, revealed, frac)
    pool = mg._global_reveal(conf, revealed, frac)
    assert [int(quota[n].sum()) for n in names] == [2, 0]
    assert [int(pool[n].sum()) for n in names] == [1, 1]
    # each policy honours its own budget definition: summed per-modality vs one frame-wide round
    assert sum(int(quota[n].sum()) for n in names) == sum(
        m.n_tok - int(round(frac * m.n_tok)) for m in cfg.modalities)
    total = cfg.tokens_per_frame
    assert sum(int(pool[n].sum()) for n in names) == total - int(round(frac * total))
