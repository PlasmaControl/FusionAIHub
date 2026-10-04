"""The BES channel geometry: channel flux and the block that covers the pedestal."""

from __future__ import annotations

import numpy as np

from labeler.confinement import bes_geometry as geo


def _flux(r_grid, z_grid, axis=(1.7, 0.0), scale=0.4):
    """psi(R, Z) = scale * rho^2 about ``axis`` as (nz, nr); psi_N = rho^2 / a^2."""
    rr, zz = np.meshgrid(r_grid, z_grid)
    return scale * ((rr - axis[0]) ** 2 + (zz - axis[1]) ** 2)


def test_psin_is_zero_on_axis_and_one_on_the_boundary_in_either_axis_order():
    r_grid = np.linspace(0.8, 2.6, 65)
    z_grid = np.linspace(-1.6, 1.6, 65)
    psi = _flux(r_grid, z_grid)  # (nz, nr)
    a = 0.55
    ssimag, ssibry = 0.0, 0.4 * a**2
    args = (
        r_grid,
        z_grid,
        np.array([ssimag]),
        np.array([ssibry]),
        np.array([1.7]),
        np.array([0.0]),
    )
    # channels at R = axis + 0.2 a and at the boundary (cm), Z = 0
    r_cm = np.array([(1.7 + 0.2 * a) * 100, (1.7 + a) * 100])
    z_cm = np.zeros(2)
    for flux in (psi[None], psi.T[None]):  # the record's axis order is not known
        psin, residual = geo.channel_psin(flux, *args, r_cm, z_cm)
        assert residual < 1e-3
        np.testing.assert_allclose(psin[0], [0.04, 1.0], atol=2e-3)


def test_median_psin_over_the_labelled_windows_only():
    times = np.array([0.0, 100.0, 200.0, 300.0])
    psin = np.array([[0.1, 0.9], [0.2, 0.9], [0.3, 0.9], [9.0, 9.0]])
    med = geo.shot_median_psin(times, psin, np.array([50.0, 150.0, 250.0]))
    np.testing.assert_allclose(med, [0.25, 0.9])
    assert np.isnan(geo.shot_median_psin(times, psin, np.array([500.0]))).all()


def _grid(rows):
    """64 channels from per-row psi_N ranges (8 columns across ``lo``..``hi``)."""
    return np.concatenate([np.linspace(lo, hi, 8) for lo, hi in rows])


def test_first_six_rows_stand_when_every_row_covers_the_pedestal():
    got = geo.choose_block(_grid([(0.8, 1.05)] * 8))
    assert got["start"] == 0 and got["rows_covering"] == 6 and got["reaches"]


def test_a_displaced_first_row_moves_the_block_down():
    rows = [(0.5, 0.75)] + [(0.8, 1.05)] * 7  # row 0 sits inboard of the pedestal
    got = geo.choose_block(_grid(rows))
    assert got["start"] == 1 and got["rows_covering"] == 6


def test_an_array_that_stops_short_of_the_separatrix_does_not_reach():
    got = geo.choose_block(_grid([(0.6, 0.93)] * 8))
    assert not got["reaches"] and got["outer_psin"] < geo.REACH_PSIN


def test_no_flux_gives_no_block():
    got = geo.choose_block(np.full(64, np.nan))
    assert got["start"] is None and not got["reaches"]
