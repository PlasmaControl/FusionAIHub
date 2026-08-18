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
