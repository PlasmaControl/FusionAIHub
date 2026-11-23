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
