"""The quantizer of the learned tokenizers and the token network (plan 1c)."""

import numpy as np
import pytest
import torch

from tokamak_foundation_model.ignite.mast.tokenizers.learned import identity, nets, quantizer, recode, store

META = {"signals": {"m": {"kind": "frame_dct", "domain": "linear", "n_streams": 3, "values_per_token": 1}}}


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def level_positions(x, n):
    pos = (quantizer.round_levels(x, n).detach().numpy() + 1) / 2 * (n - 1)
    assert np.abs(pos - np.rint(pos)).max() < 1e-4
    return np.unique(np.rint(pos))


def test_rounding_keeps_the_ends_and_passes_the_gradient():
    v = torch.tensor([-1.0, -0.99, 0.01, 0.99, 1.0], requires_grad=True)
    r = quantizer.round_levels(v, 8)
    assert torch.allclose(r, torch.tensor([-1.0, -1.0, 1 / 7, 1.0, 1.0]), atol=1e-6)
    r.sum().backward()
    assert torch.allclose(v.grad, torch.ones(5))
    x = torch.linspace(-1, 1, 1001)
    assert len(level_positions(x, 8)) == 8 and len(level_positions(x, 64)) == 64
    coarse = quantizer.round_levels(x, 8)
    assert torch.allclose(quantizer.round_levels(coarse, 64), coarse, atol=1e-6)   # the 8 levels are among the 64


def test_dropped_values_are_zero_and_the_rest_keep_their_precision():
    v = torch.tensor([[0.30, -0.52, 0.11, 0.90]] * 3)
    out = quantizer.apply(v, torch.tensor([2, 0, 1]), torch.tensor([2, 4, 3]))
    assert torch.allclose(out[0, :2], v[0, :2], atol=1e-7) and out[0, 2:].abs().sum() == 0        # unrounded, two kept
    assert torch.allclose(out[1], torch.tensor([3 / 7, -3 / 7, 1 / 7, 1.0]), atol=1e-6)           # 8 levels, all kept
    assert torch.allclose(out[2, :3], torch.tensor([19 / 63, -33 / 63, 7 / 63]), atol=1e-6) and out[2, 3] == 0


def test_the_draw_skips_rounding_for_half_and_keeps_all_values_for_more_than_half():
    precision, n_keep = quantizer.draw(20000, 8, gen())
    assert 0.48 < (precision == 2).float().mean() < 0.52 and 0.23 < (precision == 0).float().mean() < 0.27
    assert 0.54 < (n_keep == 8).float().mean() < 0.585 and n_keep.min() == 1 and n_keep.max() == 8
    again = quantizer.draw(20000, 8, gen())
    assert torch.equal(again[0], precision) and torch.equal(again[1], n_keep)


def test_the_net_gives_bounded_values_and_decodes_them():
    torch.manual_seed(0)
    net = nets.TokenNet(16, 8, 5, d_model=32, depth=1)
    e = 50.0 * torch.randn(64, 16)                                   # far outside unit scale
    feats, ident = torch.zeros(64, identity.N_FEAT), torch.arange(64) % 5
    net.eval()
    e_hat, v = net(e, feats, ident)
    assert v.shape == (64, 8) and v.abs().max() <= 1 and e_hat.shape == (64, 16) and torch.isfinite(e_hat).all()
    assert torch.allclose(net.decode(v, feats, ident), e_hat)        # eval mode: the values pass unchanged
    net.train()
    with pytest.raises(ValueError, match="generator"):
        net(e, feats, ident)
    a, _ = net(e, feats, ident, generator=gen(1))
    b, _ = net(e, feats, ident, generator=gen(1))
    assert torch.equal(a, b) and not torch.allclose(a, e_hat)


def test_identity_without_a_table_row_is_the_dropped_identity():
    torch.manual_seed(0)
    net = nets.TokenNet(16, 8, 5, d_model=32, depth=1).eval()
    with torch.no_grad():
        net.learned.weight.normal_()
    feats = torch.randn(4, identity.N_FEAT)
    known, unknown = torch.arange(4), torch.full((4,), -1)
    dropped = net.identity(feats, known, drop=torch.ones(4, dtype=torch.bool))
    assert torch.allclose(net.identity(feats, unknown), dropped)
    assert not torch.allclose(net.identity(feats, known), dropped)
    assert not torch.allclose(dropped[0], dropped[1])                # the table's features still tell streams apart
    control = nets.TokenNet(48, 24, 0, d_model=32, depth=1)
    assert control.identity(feats, unknown).abs().sum() == 0


def test_the_net_recoder_is_a_recoder_and_is_stored(tmp_path):
    torch.manual_seed(0)
    table = identity.StreamTable({"m": (0, 3)}, np.random.default_rng(0).standard_normal((3, identity.N_FEAT)).astype(np.float32))
    config = dict(d_in=16, n_values=8, n_streams=3, d_model=32, depth=1, p_drop_ident=0.1, concat=False, n_tok=1, n_val=8)
    net = nets.TokenNet(16, 8, 3, d_model=32, depth=1)
    r = nets.NetRecoder(net, table, concat=False, n_tok=1, n_val=8, batch=7)
    e = np.random.default_rng(1).standard_normal((20, 16)).astype(np.float32)
    ident = np.arange(20) % 3
    v = r.encode(e, ident)
    assert isinstance(r, recode.Recoder) and r.kind == "net" and (r.d_in, r.n_values) == (16, 8) and not net.training
    assert v.shape == (20, 8) and v.dtype == np.float32 and np.abs(v).max() <= 1 and r.decode(v, ident).shape == (20, 16)
    whole = nets.NetRecoder(net, table, concat=False, n_tok=1, n_val=8)
    assert np.allclose(whole.encode(e, ident), v, atol=1e-6)         # the batch size changes nothing
    nets.save_net(tmp_path / "best.pt", net, config)
    back = nets.load_net_recoder(tmp_path, table)
    assert np.allclose(back.encode(e, ident), v, atol=1e-6) and (back.concat, back.n_tok, back.n_val) == (False, 1, 8)
    recode.save_recoder(tmp_path, r, {"m": store.Norm(np.zeros((3, 16)), np.ones(3))}, table, META)
    loaded = recode.load_recoder(tmp_path)
    assert loaded.recoder.kind == "net" and np.allclose(loaded.recoder.decode(v, ident), r.decode(v, ident), atol=1e-6)
