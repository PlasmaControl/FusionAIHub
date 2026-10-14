"""The v5 RWM sensor: the n = 1 delta-Bp, its coil pickup and rwm-ours's columns."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.events import rwm
from labeler.rwm import bp
from labeler.rwm import features as f

T = np.arange(0.0, 2000.0, bp.STEP_MS)
PHI = np.deg2rad(list(bp.ANGLES_DEG.values()))


def probes(amplitude_g, phase_deg=40.0, offsets_g=(5.0, -12.0, 8.0, 20.0)):
    """The four probe differences, tesla, of an n = 1 field of `amplitude_g`
    (one value per `T`) at `phase_deg`, on static offsets."""
    phase = np.deg2rad(phase_deg)
    return {
        name: (T, (2 * amplitude_g * np.cos(phi - phase) + offset) / bp.T_TO_G)
        for (name, phi, offset) in zip(bp.ANGLES_DEG, PHI, offsets_g, strict=True)
    }


def burst(lo=1000.0, hi=1020.0, size=20.0):
    return np.where((T >= lo) & (T < hi), size, 0.0)


def test_n1_fit_recovers_amplitude_and_phase():
    phase = np.deg2rad(-70.0)
    Y = np.array([2 * 3.0 * np.cos(phi - phase) for phi in PHI])[:, None]
    amplitude, found = bp.n1_fit(Y)
    assert amplitude[0] == pytest.approx(3.0)
    assert found[0] == pytest.approx(-70.0)
    Y[1, 0] = np.nan
    assert np.isnan(bp.n1_fit(Y)[0][0])


def test_amplitude_removes_offsets_and_sees_a_short_burst():
    got = bp.amplitude(probes(burst()), pickup=False)
    t, a = got["t_ms"], got["amp_g"]
    assert np.nanmax(a[(t >= 1000) & (t < 1020)]) == pytest.approx(20.0, rel=0.02)
    # The static offsets are gone: quiet time reads zero.
    assert np.nanmax(np.abs(a[(t > 300) & (t < 900)])) < 1e-9
    assert got["phase_deg"][np.argmax(np.nan_to_num(a))] == pytest.approx(40.0)


def test_single_probes_stand_in_for_missing_differences():
    differences = probes(burst())
    singles = {}
    for name, (probe, partner) in bp.SINGLES.items():
        t, d = differences[name]
        # Two probes 180 deg apart sharing a 3 G common field: gain x their
        # difference is the MPID point.
        half = d / bp.DIFFERENCE_GAIN / 2
        singles[probe], singles[partner] = (t, 3e-4 + half), (t, 3e-4 - half)
    got = bp.amplitude(singles, pickup=False)
    want = bp.amplitude(differences, pickup=False)
    assert got["probes"] == "single probes" and want["probes"] == "MPID"
    np.testing.assert_allclose(got["amp_g"], want["amp_g"], atol=1e-9)
    # A cached MPID point wins over its singles; a lone single is not enough.
    mixed = {**singles, "mpid067u": differences["mpid067u"]}
    assert bp.probe_source(mixed) == "mixed"
    del singles["mpi66m340f"]
    assert bp.amplitude(singles, pickup=False) is None


def test_coil_pickup_is_regressed_out_and_a_plasma_burst_kept():
    rng = np.random.default_rng(0)
    current = 2000.0 * np.sin(2 * np.pi * T / 37.0) + 50 * rng.standard_normal(T.size)
    signals = probes(burst())
    coupling = (3.0, -1.0, 2.0, 0.5)  # G per kA, per probe
    for (name, (t, y)), k in zip(list(signals.items()), coupling, strict=True):
        signals[name] = (t, y + k * current / 1000.0 / bp.T_TO_G)
    signals["iu30"] = (T, current)
    raw = bp.amplitude(signals, pickup=False)
    clean = bp.amplitude(signals)
    quiet = (T > 300) & (T < 900)
    assert np.nanmax(raw["amp_g"][quiet]) > 1.0
    assert np.nanmax(clean["amp_g"][quiet]) < 0.05
    found = [c["coef_g_per_ka"][0] for c in clean["pickup"]["channels"]]
    # The 100 ms running medians of a probe and a coil differ near the burst.
    assert found == pytest.approx(coupling, abs=0.02)
    peak = np.nanmax(clean["amp_g"][(T >= 1000) & (T < 1020)])
    assert peak == pytest.approx(20.0, rel=0.05)


def test_the_causal_amplitude_never_reads_a_later_sample():
    early = bp.amplitude(probes(burst(1500, 1520)), causal=True, pickup=False)
    later = bp.amplitude(
        probes(burst(1500, 1520) + burst(1600, 1620)), causal=True, pickup=False
    )
    before = early["t_ms"] < 1600
    np.testing.assert_array_equal(early["amp_g"][before], later["amp_g"][before])
    centred = bp.amplitude(probes(burst(1500, 1520) + burst(1600, 1620)), pickup=False)
    assert not np.allclose(
        np.nan_to_num(centred["amp_g"][before]), np.nan_to_num(early["amp_g"][before])
    )


def test_bp_runs_bridge_short_dips_and_drop_short_runs():
    t = np.arange(0.0, 50.0, 0.5)
    above = ((t >= 10) & (t < 15)) | ((t >= 16) & (t < 20)) | ((t >= 30) & (t < 31))
    finite = np.ones(t.size, dtype=bool)
    # A 1 ms dip is bridged; a 1 ms run is too short.
    assert rwm.bp_runs(t, above, finite) == [(10.0, 19.5)]
    # A dip with a missing sample is not bridged: two runs, both long enough.
    finite[t == 15.5] = False
    assert rwm.bp_runs(t, above, finite) == [(10.0, 14.5), (16.0, 19.5)]


@pytest.mark.parametrize(
    "series,end,why,rise",
    [
        # Above the threshold from 2 ms after the onset to 30 ms: present to there.
        ((lambda t: np.where((t >= 902) & (t < 930), 20.0, 0.0)), 930.0, "bp_n1", 2.0),
        # Already up 15 ms before the onset (an earlier burst): still this onset's.
        (
            (lambda t: np.where((t >= 885) & (t < 925), 20.0, 0.0)),
            925.0,
            "bp_n1",
            -15.0,
        ),
        # A rise 30 ms late is another excursion: no response, one frame present.
        (
            (lambda t: np.where((t >= 930) & (t < 960), 20.0, 0.0)),
            910.0,
            "no bp_n1 response",
            None,
        ),
    ],
)
def test_bp_end_follows_delta_bp_back_under_the_threshold(series, end, why, rise):
    t = np.arange(0.0, 2000.0, 0.5)
    got = rwm.bp_end(900.0, series=(t, series(t), 5.0))
    assert got[:3] == (end, why, rise)
    if rise is None:
        assert got[4] == 900.0 + rwm.MAX_PRESENT_MS  # uncertain to the cap
    else:
        assert got[3] == 20.0 and got[4] is None


def test_q_surface_rho_takes_the_outermost_crossing():
    q = np.array([[1.5, 2.5, 1.9], [1.8, 2.6, 2.1], [2.2, 2.8, 1.8], [3.0, 3.0, 2.4]])
    rho = np.tile(np.array([0.0, 0.3, 0.6, 0.9])[:, None], (1, 3))
    got = f.q_surface_rho(q, rho)
    assert got[0] == pytest.approx(0.3 + 0.3 * 0.2 / 0.4)
    assert np.isnan(got[1])  # q above 2 everywhere: no surface
    assert got[2] == pytest.approx(0.6 + 0.3 * 0.2 / 0.6)  # outermost of three


def test_bp_columns_are_causal_and_nan_without_inputs():
    grid = np.arange(500.0, 1900.0, f.STEP_MS)
    empty = f.bp_columns({}, grid)
    assert set(empty) == set(f.BP_FEATURES)
    assert all(np.isnan(v).all() for v in empty.values())
    signals = probes(burst(1500, 1520))
    signals["iu30"] = (T, np.where(T >= 1200, 1000.0, 0.0))
    got = f.bp_columns(signals, grid)
    assert np.nanmax(got["bp_n1_max_g"][grid < 1500]) < 1e-6
    assert got["bp_n1_max_g"][grid == 1510][0] == pytest.approx(20.0, rel=0.05)
    assert got["icoil_dev_ka"][grid == 1200][0] == pytest.approx(1.0)
    assert np.isnan(got["rot_q2_khz"]).all()
    assert "bp_n1_g" in f.V5_FEATURES and f.V5_FEATURES[: len(f.FEATURES)] == f.FEATURES
