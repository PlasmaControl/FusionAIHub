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


def _tiny_rev():
    from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 6, 5), ModalitySpec("b", "slowts", 4, 5)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=2,
        maskgit_decode_steps=4, actuator_dim=6)
