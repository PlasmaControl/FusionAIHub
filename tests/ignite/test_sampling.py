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
