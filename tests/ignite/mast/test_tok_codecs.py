"""Codecs without learned weights. This file grows in Tasks 4 and 5."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import transform
from tokamak_foundation_model.ignite.mast.tokenizers import codecs, grid

RNG = np.random.default_rng(0)


def stats(compress, c):
    return transform.ChannelStats(compress=compress, scale=np.full(c, 2.0), mean=np.full(c, 0.5), std=np.full(c, 1.5))


def test_the_first_value_is_the_frame_mean_and_values_are_nested():
    x = RNG.standard_normal((7, 25))
    v = codecs.dct_values(x, 25)
    assert np.allclose(v[:, 0], x.mean(-1)) and np.allclose((v ** 2).sum(-1), (x ** 2).mean(-1))
    assert np.allclose(codecs.dct_decode(v, 25), x)
    assert np.allclose(codecs.dct_decode(v[:, :1], 25), np.repeat(x.mean(-1, keepdims=True), 25, -1))
    err = [np.mean((codecs.dct_decode(v[:, :k], 25) - x) ** 2) for k in (1, 2, 4, 8, 16, 25)]
    assert all(a > b for a, b in zip(err, err[1:])) and err[-1] < 1e-20
    assert np.allclose(codecs.dct_values(x, 4), v[:, :4])                    # a prefix is the same values


def test_a_frame_of_one_sample_is_exact_with_one_value():
    v = codecs.dct_values(np.array([[3.5]]), 4)
    assert v.tolist() == [[3.5, 0.0, 0.0, 0.0]] and codecs.dct_decode(v, 1).tolist() == [[3.5]]


def test_a_partly_valid_frame_falls_back_to_the_mean_of_its_valid_samples():
    v = codecs.dct_values(np.array([[1.0, np.nan, 3.0, np.nan]]), 3)
    assert v.tolist() == [[2.0, 0.0, 0.0]] and np.allclose(codecs.dct_decode(v, 4), 2.0)


def test_a_frame_without_a_valid_sample_has_no_values():
    v = codecs.dct_values(np.full((1, 4), np.nan), 2)
    assert np.isnan(v).all() and np.isnan(codecs.dct_decode(v, 4)).all()


@pytest.mark.parametrize("compress,domain", [("none", "linear"), ("asinh", "linear"), ("asinh", "compressed")])
def test_frame_dct_with_every_coefficient_is_the_identity(compress, domain):
    c = codecs.FrameDCT("s", 0.00025, (3,), stats(compress, 3), k=20, domain=domain)
    x = RNG.standard_normal((4, 3, 100))                                     # 5 frames of 20 samples
    v = c.encode(x)
    assert v.shape == (4, 5, 3, 20) and (c.n_tok, c.n_val) == (3, 20)
    assert np.allclose(c.decode(v, 100), x) and np.allclose(c.roundtrip(x), x)


def test_frame_dct_with_one_value_holds_the_frame_mean_in_its_domain():
    st = stats("asinh", 2)
    x = np.abs(RNG.standard_normal((1, 2, 40))) * 10
    lin = codecs.FrameDCT("s", 0.00025, (2,), st, k=1, domain="linear").roundtrip(x)
    com = codecs.FrameDCT("s", 0.00025, (2,), st, k=1, domain="compressed").roundtrip(x)
    means = x.reshape(1, 2, 2, 20).mean(-1)
    assert np.allclose(lin.reshape(1, 2, 2, 20)[..., 0], means)              # the linear domain keeps the mean
    assert (com.reshape(1, 2, 2, 20)[..., 0] < means).all()                  # a mean on the asinh scale is low


def test_roundtrip_reads_through_the_grids():
    st = stats("none", 1)
    g = grid.GridSet(lo=np.array([[-1.0]]), hi=np.array([[1.0]]), width=np.array([[1.0]]), n_bins=3)
    c = codecs.FrameDCT("s", 0.005, (1,), st, k=1, grids=g)
    x = st.mean[0] + st.std[0] * np.array([[[0.2, 5.0, np.nan]]])            # linear units 0.2, 5.0, missing
    lin = lambda a: (a - st.mean[0]) / st.std[0]
    assert np.allclose(lin(c.roundtrip(x, "ranged"))[0, 0, :2], [0.2, 1.0])
    assert np.allclose(lin(c.roundtrip(x, "nearest"))[0, 0, :2], [0.0, 1.0])
    assert np.isnan(c.roundtrip(x, "ranged")[0, 0, 2])
    with pytest.raises(ValueError, match="grids"):
        codecs.FrameDCT("s", 0.005, (1,), st, k=1).roundtrip(x, "ranged")


def low_rank(n, d=30, seed=0):
    """Rows of dimension d that lie exactly in a 4-dimensional subspace about the mean 3."""
    rng = np.random.default_rng(seed)
    true = np.linalg.qr(rng.standard_normal((d, 4)))[0].T
    return 3.0 + (rng.standard_normal((n, 4)) * np.array([8.0, 4.0, 2.0, 1.0])) @ true, true


def linear_codec(n_values, per_token, **kw):
    x, _ = low_rank(4000)
    mean, basis, eig, _ = codecs.fit_pca(x, 6)
    st = transform.ChannelStats(compress="none", scale=np.ones(30), mean=np.zeros(30), std=np.ones(30))
    return codecs.LinearCodec("p", 0.005, (30,), st, mean, basis, eig, n_values, per_token, **kw), x


def test_fit_pca_finds_the_subspace_in_order_of_variance():
    x, true = low_rank(4000)
    mean, basis, eig, total = codecs.fit_pca(x, 6)
    assert mean.shape == (30,) and basis.shape == (6, 30) and eig.shape == (6,)
    assert np.allclose(mean, x.mean(0)) and np.allclose(basis @ basis.T, np.eye(6), atol=1e-8)
    assert np.allclose(eig[:4], [64.0, 16.0, 4.0, 1.0], rtol=0.15) and (np.abs(eig[4:]) < 1e-8).all()
    assert np.isclose(total, eig.sum()) and np.allclose(true @ basis[:4].T @ basis[:4], true, atol=1e-6)
    assert codecs.fit_pca(x, 96)[1].shape == (30, 30)                        # never more components than points


def test_fit_pca_uses_every_pair_of_valid_points():
    x, _ = low_rank(20000)
    holes = x.copy()
    holes[np.random.default_rng(1).random(x.shape) < 0.3] = np.nan
    holes[:, 29] = np.nan                                                    # a point that is never valid
    mean, basis, eig, _ = codecs.fit_pca(holes, 2)
    ref_mean, ref, _, _ = codecs.fit_pca(x[:, :29], 2)                       # the same 29 points without holes
    assert mean[29] == 0.0 and np.abs(basis[:, 29]).max() < 1e-12
    assert np.allclose(mean[:29], ref_mean, atol=0.1)
    assert np.linalg.norm(ref @ basis[:, :29].T @ basis[:, :29] - ref) < 0.1


def test_a_low_rank_profile_is_exact_with_enough_values_and_tokens_are_nested():
    c, x = linear_codec(4, 2)
    w = np.moveaxis(x[:12].reshape(3, 4, 30), 1, 2)                          # (N=3, D=30, T=4 frames)
    v = c.encode(w)
    assert v.shape == (3, 4, 2, 2) and (c.n_tok, c.n_val) == (2, 2)
    assert np.allclose(c.decode(v, 4), w, atol=1e-6)
    coarse = v.copy()
    coarse[:, :, 1] = 0.0                                                    # drop the second token
    assert np.allclose(c.decode(coarse, 4), linear_codec(2, 2)[0].roundtrip(w), atol=1e-6)


def test_zero_values_decode_to_the_mean_profile():
    c, x = linear_codec(0, 2)
    w = x[:2].T[None]                                                        # (1, 30, 2)
    assert c.encode(w).shape == (1, 2, 0, 2)
    assert np.allclose(c.roundtrip(w), np.repeat(c.profile_mean[None, :, None], 2, -1))


def test_a_profile_with_missing_points_is_encoded_from_its_valid_points():
    c, x = linear_codec(4, 2)
    w = x[:1].T[None].copy()                                                 # (1, 30, 1)
    whole = c.encode(w)
    w[0, ::3, 0] = np.nan                                                    # a third of the points missing
    v = c.encode(w)
    assert np.allclose(v, whole, atol=1e-5)                                  # exact low rank: 20 points are enough
    assert np.isfinite(c.decode(v, 1)).all()                                 # the decoder fills the missing points


def test_fewer_valid_points_than_components_still_gives_finite_values():
    c, x = linear_codec(4, 2, ridge=0.5)
    w = np.full((1, 30, 2), np.nan)
    w[0, :2, 0] = x[0, :2]                                                   # frame 0: two valid points; frame 1: none
    v = c.encode(w)
    assert np.isfinite(v[0, 0]).all() and np.isnan(v[0, 1]).all()
    out = c.decode(v, 2)
    assert np.isfinite(out[0, :, 0]).all() and np.isnan(out[0, :, 1]).all()


def test_a_map_is_flattened_row_major():
    st = transform.ChannelStats(compress="none", scale=np.ones((3, 4)), mean=np.zeros((3, 4)), std=np.ones((3, 4)))
    c = codecs.LinearCodec("m", 0.005, (3, 4), st, np.zeros(12), np.eye(12), np.ones(12), 12, 4)
    w = RNG.standard_normal((2, 3, 4, 5))
    v = c.encode(w)
    assert v.shape == (2, 5, 3, 4) and np.allclose(v[1, 2].ravel(), w[1, :, :, 2].ravel(), atol=1e-6)
    assert np.allclose(c.decode(v, 5), w, atol=1e-6)


def test_the_signal_domain_is_tokamarks_standardization():
    st = transform.ChannelStats(compress="none", scale=np.ones(30), mean=np.full(30, 5.0), std=np.linspace(1.0, 3.0, 30))
    c = codecs.LinearCodec("p", 0.005, (30,), st, np.zeros(30), np.eye(30), np.ones(30), 30, 30, domain="signal",
                           signal_mean=2.0, signal_std=4.0)
    w = RNG.standard_normal((2, 30, 1))
    assert np.allclose(c.to_domain(w), (w - 2.0) / 4.0)
    assert np.allclose(c.encode(w)[:, 0, 0, :], (w[:, :, 0] - 2.0) / 4.0, atol=1e-6)
    assert np.allclose(c.domain_from_model(transform.to_model(w[0], st)), (w[0] - 2.0) / 4.0)
    with pytest.raises(ValueError, match="multiple"):
        codecs.LinearCodec("p", 0.005, (30,), st, np.zeros(30), np.eye(30), np.ones(30), 10, 4)
