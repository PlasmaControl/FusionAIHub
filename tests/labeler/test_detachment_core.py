"""The detachment bin grid, the Indicator contract and the TangTV gate."""

import numpy as np
import pytest

from labeler.events.detachment import core, tangtv


def test_edges_are_aligned_and_cover():
    edges = core.bin_edges(120.0, 330.0)
    assert edges[0] == 100.0 and edges[-1] >= 330.0
    assert np.allclose(np.diff(edges), core.BIN_MS)
    assert np.allclose(core.bin_centres(edges), edges[:-1] + core.BIN_MS / 2)


def test_median_ignores_a_spike_and_mean_does_not():
    edges = np.array([0.0, 50.0, 100.0])
    t = np.array([1.0, 10.0, 20.0, 30.0, 60.0])
    y = np.array([1.0, 1.0, 100.0, 1.0, 5.0])
    med, count = core.bin_median(t, y, edges)
    mean, _ = core.bin_mean(t, y, edges)
    assert med.tolist() == [1.0, 5.0] and count.tolist() == [4, 1]
    assert mean[0] == pytest.approx(25.75)


def test_keep_mask_and_min_count():
    edges = np.array([0.0, 50.0, 100.0])
    t = np.array([10.0, 20.0, 70.0])
    y = np.array([1.0, 3.0, 9.0])
    keep = np.array([True, False, True])
    med, count = core.bin_median(t, y, edges, keep=keep, min_count=2)
    assert np.isnan(med).all() and count.tolist() == [1, 1]
    mean, _ = core.bin_mean(t, y, edges, keep=keep)
    assert mean.tolist() == [1.0, 9.0]


def test_a_modulated_beam_has_zero_median_but_a_real_mean():
    edges = np.array([0.0, 50.0])
    t = np.arange(0.0, 50.0, 1.0)
    power = np.where(t % 2 == 0, 1.0, 0.0)
    power[::2] = 1.0
    power[1::2] = 0.0
    assert core.bin_median(t, power, edges)[0][0] in (0.0, 0.5)
    assert core.bin_mean(t, power, edges)[0][0] == pytest.approx(0.5)


def test_fraction_counts_flagged_samples():
    edges = np.array([0.0, 50.0, 100.0])
    t = np.array([5.0, 10.0, 60.0])
    flag = np.array([True, False, True])
    frac = core.bin_fraction(t, flag, edges)
    assert frac.tolist() == [0.5, 1.0]
    assert np.isnan(core.bin_fraction(np.array([]), np.array([]), edges)).all()


def test_assemble_forces_abstain_where_invalid():
    ind = core.assemble(
        "x",
        [0.1, 0.9],
        [True, False],
        ["", "why"],
        [core.ATTACHED, core.DETACHED],
    )
    assert ind.vote.tolist() == [core.ATTACHED, core.ABSTAIN]
    assert ind.reason.tolist() == ["", "why"]


def test_indicator_rejects_a_vote_on_an_invalid_bin():
    with pytest.raises(ValueError):
        core.Indicator(
            "x",
            np.zeros(1),
            np.zeros(1, bool),
            np.array([""], dtype=object),
            np.array([core.ATTACHED]),
        )


# ---- TangTV -----------------------------------------------------------------------


def test_front_dz_definition():
    # strike -1.25, X-point -1.0: a front at the strike point is 0, at the X-point 1.
    assert tangtv.front_dz(-1.25, -1.0, -1.25) == pytest.approx(0.0)
    assert tangtv.front_dz(-1.0, -1.0, -1.25) == pytest.approx(1.0)
    assert tangtv.front_dz(-1.125, -1.0, -1.25) == pytest.approx(0.5)
    assert np.isnan(tangtv.front_dz(-1.0, -1.0, -1.0))


def test_dz_votes():
    dz = np.array([0.0, 0.34, 0.4, 0.5, 0.9, 1.0, 1.2, np.nan])
    v = tangtv.dz_vote(dz).tolist()
    assert v == [1, 1, -1, 2, 2, 2, 3, -1]


def test_gate_accepts_the_shelf_and_rejects_the_floor():
    # shelf: outer strike at R 1.5, Z -1.25; floor: R 1.2, Z -1.363.
    valid, why = tangtv.shelf_gate(
        [1.5, 1.2, 1.5, -0.89, 1.5],
        [-1.25, -1.363, -1.25, -0.89, -1.25],
        [1.3, 1.3, 1.3, 1.3, 1.3],
        [-1.1, -1.1, 0.5, -1.1, -9.99],
    )
    assert valid.tolist() == [True, False, False, False, False]
    assert why.tolist() == [
        "",
        "strike_on_floor",
        "not_lower_null",
        "efit_missing",
        "efit_missing",
    ]


def test_ze_is_the_height_of_the_emission_outboard_of_the_x_point():
    radii = np.linspace(1.0, 1.7, 8)
    elevation = np.linspace(-1.4, -0.9, 6)
    frames = np.zeros((2, 6, 8), dtype=np.float32)
    frames[0, 1, 5] = 1.0  # row 1 outboard of rx
    frames[1, 4, 5] = 1.0  # row 4 outboard
    frames[1, 0, 1] = 5.0  # bright, but inboard of rx: ignored
    ze = tangtv.outer_leg_ze(frames, radii, elevation, np.array([1.3, 1.3]))
    assert ze == pytest.approx([elevation[1], elevation[4]])
    empty = tangtv.outer_leg_ze(np.zeros((1, 6, 8)), radii, elevation, np.array([1.3]))
    assert np.isnan(empty).all()


def make_run(gate_ok=True, ze=-1.1):
    edges = core.bin_edges(0.0, 200.0)
    ft = np.array([10.0, 30.0, 60.0, 80.0, 110.0, 130.0, 160.0, 180.0])
    et = np.array([0.0, 100.0, 200.0])
    rv = np.full(3, 1.5 if gate_ok else 1.2)
    zv = np.full(3, -1.25 if gate_ok else -1.363)
    rx = np.full(3, 1.3)
    zx = np.full(3, -1.0)
    return tangtv.tangtv_indicator(edges, ft, np.full(8, ze), et, rv, zv, rx, zx)


def test_indicator_votes_on_the_shelf():
    ind = make_run(True, -1.1)  # DZ = 1 - 0.1/0.25 = 0.6: detached
    assert ind.valid.all() and (ind.vote == core.DETACHED).all()
    assert ind.value == pytest.approx(np.full(4, 0.6))


def test_indicator_never_votes_on_the_floor():
    ind = make_run(False, -1.1)
    assert not ind.valid.any() and (ind.vote == core.ABSTAIN).all()
    assert set(ind.reason) == {"strike_on_floor"}
