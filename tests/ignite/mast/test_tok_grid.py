"""Ordered bins: two-hot targets, expected values, and the three read-outs."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast.tokenizers import grid


def one(lo=-10.0, hi=10.0, width=1.0, n_bins=5):
    return grid.GridSet(lo=np.array([[lo]]), hi=np.array([[hi]]), width=np.array([[width]]), n_bins=n_bins)


def test_centres_are_asinh_spaced_between_lo_and_hi():
    c = one().centres()
    assert c.shape == (1, 1, 5)
    assert np.allclose(c[0, 0], np.sinh(np.linspace(np.arcsinh(-10.0), np.arcsinh(10.0), 5)))
    assert np.isclose(c[0, 0, 0], -10.0) and np.isclose(c[0, 0, -1], 10.0) and (np.diff(c[0, 0]) > 0).all()


def test_two_hot_has_two_neighbours_and_its_mean_is_the_value():
    g = one()
    c = g.centres()[0, 0]
    v = np.array([[[c[1]]], [[0.5 * (c[1] + c[2])]], [[0.25 * c[3]]], [[10.0]]])        # (4, 1, 1)
    w = g.two_hot(v)
    assert w.shape == (4, 1, 1, 5) and np.allclose(w.sum(-1), 1.0) and ((w > 0).sum(-1) <= 2).all()
    assert np.allclose(w[0, 0, 0], [0, 1, 0, 0, 0]) and np.allclose(w[1, 0, 0], [0, 0.5, 0.5, 0, 0])
    assert np.allclose(w[2, 0, 0], [0, 0, 0.75, 0.25, 0])
    assert np.allclose(g.expected(w), v)


def test_a_value_outside_the_range_reads_as_the_outermost_centre():
    g = one()
    v = np.array([[[-50.0]], [[1e9]], [[np.nan]]])
    for mode in ("ranged", "nearest"):
        r = g.read(v, mode)
        assert np.allclose(r[:2, 0, 0], [-10.0, 10.0]) and np.isnan(r[2, 0, 0])
    assert g.index(v)[:, 0, 0].tolist() == [0, 4, -1]
    assert np.array_equal(g.read(v, "unquantized"), v, equal_nan=True)


def test_nearest_is_the_nearest_centre_and_ranged_is_exact_inside():
    g = one()
    c = g.centres()[0, 0]
    v = np.array([[[0.49 * (c[2] + c[3])]], [[0.51 * (c[2] + c[3])]]])
    assert np.allclose(g.read(v, "nearest")[:, 0, 0], [c[2], c[3]])
    assert np.allclose(g.read(v, "ranged"), v)


def test_a_prefix_of_the_values_is_read_with_the_first_grids():
    g = grid.GridSet(lo=np.array([[-1.0, -4.0, -9.0]]), hi=np.array([[1.0, 4.0, 9.0]]), width=np.ones((1, 3)), n_bins=9)
    assert np.allclose(g.read(np.array([[5.0, 5.0]]), "ranged"), [[1.0, 4.0]])


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unquantized"):
        one().read(np.zeros((1, 1)), "rounded")


def test_fit_ranges_covers_the_data_with_a_margin():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((5000, 2, 3))
    v[:, 1, 2] = 7.0                         # a constant position
    v[:, 0, 1] = np.nan                      # a position that was never seen
    v[0, 0, 0] = 40.0                        # an outlier: the range covers it, the width does not grow with it
    lo, hi, width = grid.fit_ranges(v)
    assert lo.shape == hi.shape == width.shape == (2, 3)
    span = 40.0 - np.nanmin(v[:, 0, 0])
    assert np.isclose(hi[0, 0], 40.0 + 0.05 * span) and np.isclose(lo[0, 0], np.nanmin(v[:, 0, 0]) - 0.05 * span)
    assert 0.8 < width[0, 0] < 1.2
    assert (lo[1, 2], hi[1, 2], width[1, 2]) == (6.0, 8.0, 1.0)
    assert np.isnan(lo[0, 1]) and np.isnan(hi[0, 1]) and np.isnan(width[0, 1])
