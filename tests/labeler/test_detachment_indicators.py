"""The Prad and Afrac indicators and the Langmuir sweep reader."""

import numpy as np
import pytest

from labeler.events.detachment import afrac, core, langmuir, prad, signals
from labeler.events.detachment import thresholds as th

EDGES = core.bin_edges(0.0, 5000.0)
N = len(EDGES) - 1


def series(value, dt=10.0, t1=5000.0):
    t = np.arange(0.0, t1, dt)
    return t, np.full(t.shape, value, dtype=float)


def clean_elm():
    t = np.arange(5001.0)
    return t, np.zeros(len(t), bool)


def test_prad_votes_follow_the_fraction():
    t, p_in = series(4e6)
    _, low = series(1.0e6)  # f = 0.25
    _, mid = series(1.8e6)  # f = 0.45
    _, high = series(2.4e6)  # f = 0.6
    for rad, want in ((low, core.ATTACHED), (mid, core.ABSTAIN), (high, core.DETACHED)):
        ind = prad.prad_indicator(EDGES, t, rad, t, p_in, *clean_elm())
        assert ind.valid.all() and (ind.vote == want).all()
    ind = prad.prad_indicator(EDGES, t, high, t, p_in, *clean_elm())
    assert ind.value == pytest.approx(np.full(N, 0.6))


def test_prad_reasons():
    t, rad = series(2e6)
    none = prad.prad_indicator(EDGES, None, None, t, rad)
    assert set(none.reason) == {"no_bolometer"} and not none.valid.any()
    no_power = prad.prad_indicator(EDGES, t, rad, None, None)
    assert set(no_power.reason) == {"no_input_power"}
    _, weak = series(0.2e6)
    low = prad.prad_indicator(EDGES, t, rad, t, weak, *clean_elm())
    assert set(low.reason) == {"low_power"}
    te, flag = clean_elm()
    flag[(te >= 1000) & (te < 1100)] = True
    _, p_in = series(4e6)
    elm = prad.prad_indicator(EDGES, t, rad, t, p_in, te, flag)
    bad = (EDGES[:-1] >= 1000) & (EDGES[:-1] < 1100)
    assert (elm.reason[bad] == "elm").all() and elm.valid[~bad].all()


def test_prad_never_votes_marfe():
    t, p_in = series(4e6)
    for f in (0.0, 0.2, 0.9, 5.0):
        ind = prad.prad_indicator(
            EDGES, t, np.full(t.shape, f * 4e6), t, p_in, *clean_elm()
        )
        assert core.MARFE not in set(ind.vote)


def sweeps(level):
    """Jsat per 1 ms sweep, one probe, with the attached level at `level`."""
    t = np.arange(0.5, 5000.0, 1.0)
    return t, np.asarray(level, dtype=float)


def test_afrac_attached_then_detached():
    t = np.arange(0.5, 5000.0, 1.0)
    # constant density and power; Jsat falls to a fifth in the second half.
    jsat = np.where(t < 2500, 1.0, 0.2)[None, :]
    tn, ne = series(1e14)
    tp, psol = series(4e6)
    ind = afrac.afrac_indicator(
        EDGES,
        t,
        jsat,
        tn,
        ne,
        tp,
        psol,
        elm_t_ms=clean_elm()[0],
        elm_flag=clean_elm()[1],
    )
    first, second = EDGES[:-1] < 2500, EDGES[:-1] >= 2500
    assert ind.valid.all()
    assert (ind.vote[first] == core.ATTACHED).all()
    assert (ind.vote[second] == core.DETACHED).all()
    assert ind.value[first] == pytest.approx(np.ones(first.sum()))


def test_afrac_model_takes_out_the_density_scaling():
    t = np.arange(0.5, 5000.0, 1.0)
    tn = np.arange(0.0, 5000.0, 10.0)
    ne = np.where(tn < 2500, 1e14, 2e14)
    # Jsat doubles-squared with the density: the model says that is still attached.
    jsat = np.where(t < 2500, 1.0, 4.0)[None, :]
    tp, psol = series(4e6)
    ind = afrac.afrac_indicator(
        EDGES,
        t,
        jsat,
        tn,
        ne,
        tp,
        psol,
        elm_t_ms=clean_elm()[0],
        elm_flag=clean_elm()[1],
    )
    assert (ind.vote == core.ATTACHED).all()


def test_afrac_peak_over_probes():
    t = np.arange(0.5, 1000.0, 1.0)
    jsat = np.vstack([np.where(t < 500, 1.0, 0.1), np.where(t < 500, 0.1, 1.0)])
    edges = core.bin_edges(0.0, 1000.0)
    peak, which = afrac.peak_jsat(edges, t, jsat)
    assert peak == pytest.approx(np.ones(len(edges) - 1))
    assert which[0] == 0 and which[-1] == 1


def test_afrac_invalid_without_inputs_or_reference():
    t = np.arange(0.5, 5000.0, 1.0)
    jsat = np.ones((1, t.size))
    tn, ne = series(1e14)
    tp, psol = series(4e6)
    assert set(afrac.afrac_indicator(EDGES, None, None, tn, ne, tp, psol).reason) == {
        "no_probes"
    }
    assert set(afrac.afrac_indicator(EDGES, t, jsat, None, None, tp, psol).reason) == {
        "no_density"
    }
    assert set(afrac.afrac_indicator(EDGES, t, jsat, tn, ne, None, None).reason) == {
        "no_power"
    }
    short = core.bin_edges(0.0, 1000.0)  # 1 s < AFRAC_MIN_MS
    ind = afrac.afrac_indicator(
        short,
        t,
        jsat,
        tn,
        ne,
        tp,
        psol,
        elm_t_ms=clean_elm()[0],
        elm_flag=clean_elm()[1],
    )
    assert set(ind.reason) == {"short_reference"} and not ind.valid.any()
    _, weak = series(0.1e6)
    low = afrac.afrac_indicator(
        EDGES,
        t,
        jsat,
        tn,
        ne,
        tp,
        weak,
        elm_t_ms=clean_elm()[0],
        elm_flag=clean_elm()[1],
    )
    assert set(low.reason) == {"low_power"}


def test_afrac_ramp_is_invalid():
    t = np.arange(0.5, 5000.0, 1.0)
    jsat = np.ones((1, t.size))
    tn, ne = series(1e14)
    tp, psol = series(4e6)
    ti = np.arange(0.0, 5000.0, 20.0)
    ip = np.where(ti < 1000, ti * 2e3, 2e6)  # 2 MA/s ramp in the first second
    ind = afrac.afrac_indicator(EDGES, t, jsat, tn, ne, tp, psol, ti, ip, *clean_elm())
    ramp = (EDGES[:-1] > 60) & (EDGES[1:] < 950)  # the record's edge reads 0
    assert (ind.reason[ramp] == "ramp").all()
    assert ind.valid[EDGES[:-1] > 1100].all()


def test_afrac_reads_the_inter_elm_level():
    # Every 10th ms-sweep is an ELM spike 10x the quiet level; masked, the bin's
    # median current is the quiet one and the shot reads attached throughout.
    t = np.arange(0.5, 5000.0, 1.0)
    jsat = np.ones((1, t.size))
    flag = np.zeros(t.size, dtype=bool)
    flag[::10] = True
    jsat[0, flag] = 10.0
    tn, ne = series(1e14)
    tp, psol = series(4e6)
    te = np.r_[t[0] - 0.5, t, t[-1] + 0.5]
    ef = np.r_[False, flag, False]
    ind = afrac.afrac_indicator(EDGES, t, jsat, tn, ne, tp, psol, None, None, te, ef)
    assert ind.valid.all()
    assert ind.value == pytest.approx(np.ones(N))


def test_pre_masked_jsat_does_not_drop_an_entire_bin_at_an_elm_centre():
    t = core.bin_centres(EDGES)
    te, flag = clean_elm()
    flag[te == 25] = True
    tn, ne = series(1e14)
    tp, psol = series(4e6)
    ind = afrac.afrac_indicator(
        EDGES,
        t,
        np.ones((1, len(t))),
        tn,
        ne,
        tp,
        psol,
        elm_t_ms=te,
        elm_flag=flag,
        pre_masked=True,
    )
    assert ind.valid.all()
    assert ind.value == pytest.approx(np.ones(N))


def test_elm_at_uses_the_nearest_sample():
    t = np.arange(0.0, 100.0, 10.0)
    flag = t == 30.0
    got = core.elm_at([29.0, 31.0, 55.0, 500.0], t, flag)
    assert got.tolist() == [True, True, False, True]
    assert core.elm_at([1.0, 2.0], None, None).all()


def test_power_and_current_abstain_without_elm_information():
    t, p = series(4e6)
    tn, ne = series(1e14)
    rad = prad.prad_indicator(EDGES, t, p / 2, t, p)
    jsat = afrac.afrac_indicator(EDGES, t, np.ones((1, len(t))), tn, ne, t, p)
    for indicator in (rad, jsat):
        assert not indicator.valid.any()
        assert set(indicator.reason) == {"elm_unknown"}
        assert (indicator.vote == core.ABSTAIN).all()


def test_elm_mask_preserves_inter_elm_time(monkeypatch):
    t = np.arange(0.0, 200.1, 0.1)
    y = np.ones((4, len(t)))
    y[:, (t >= 99.9) & (t <= 100.1)] = 10
    monkeypatch.setattr(signals, "corpus_group", lambda *a, **kw: (t, y))
    tm, flag = signals.elm_mask(1, cache={})
    assert flag[(tm >= 99.9) & (tm <= 100.1)].all()
    assert not flag[(tm < 97.8) | (tm > 102.2)].any()
    assert not flag[(tm >= 110) & (tm < 150)].any()


def test_elm_mask_uses_cached_filterscopes_when_corpus_is_absent(monkeypatch):
    t = np.arange(201.0)
    monkeypatch.setattr(signals, "corpus_group", lambda *a, **kw: None)
    got = signals.elm_mask(206879, cache={"fs02": (t, np.ones(len(t)))})
    assert got is not None and not got[1].any()


def test_elm_mask_uses_cache_when_corpus_filterscopes_are_dead(monkeypatch):
    t = np.arange(201.0)
    monkeypatch.setattr(
        signals, "corpus_group", lambda *a, **kw: (t, np.full((8, len(t)), np.nan))
    )
    got = signals.elm_mask(206879, cache={"fs02": (t, np.ones(len(t)))})
    assert got is not None and not got[1].any()


def test_greenwald_density_requires_confirmed_units():
    assert signals.density_line_si(np.array([2e14]), "m/cm3")[0] == 2e20
    assert signals.density_line_si(np.array([2e16]), "cm^-2")[0] == 2e20
    assert signals.density_line_si(np.array([2e14]), "") is None


def test_greenwald_fraction_uses_geometric_r0_not_unrelated_rout():
    edges = np.array([0.0, 50.0])
    cache = {
        name: (np.array([25.0]), np.array([value]))
        for name, value in {
            "density_v2_si": 2e20,
            "aminor": 0.5,
            "kappa": 2.0,
            "r0": 1.94,
            "rout": 0.1,
            "ipmeas": 1e6,
        }.items()
    }
    # Vertical chord length = 2 m; nbar=1e20, nG=4/pi * 1e20.
    assert signals.greenwald_fraction(edges, cache) == pytest.approx([np.pi / 4])
    del cache["density_v2_si"]
    assert np.isnan(signals.greenwald_fraction(edges, cache)).all()


def test_probe_flux_accepts_multislice_efit01_and_rejects_sparse_maps():
    maps = {
        "source": np.array("EFIT01"),
        "gtime_ms": np.array([0.0, 40.0]),
        "r": np.array([1.49, 1.51]),
        "z": np.array([-1.26, -1.24]),
        "ssimag": np.zeros(2),
        "ssibry": np.ones(2),
        "psirz": np.tile(np.array([[0.99, 1.06], [0.99, 1.06]]), (2, 1, 1)),
    }
    psi = signals.flux_at_positions(
        maps, np.array([10.0, 30.0, 500.0]), np.array([[1.5, -1.25]])
    )
    assert psi[0, :2] == pytest.approx([1.025, 1.025])
    assert np.isnan(psi[0, 2])
    maps["gtime_ms"] = np.array([0.0])
    assert np.isnan(
        signals.flux_at_positions(maps, np.array([10.0]), np.array([[1.5, -1.25]]))
    ).all()


# ---- Langmuir sweeps ----------------------------------------------------------------


def synthetic_sweeps(n_sweeps=40, n=500, isat=0.5, offset=0.1):
    """A 1 kHz sawtooth bias and the textbook I-V current on top of an offset."""
    phase = np.tile(np.linspace(-2.5, 0.5, n), n_sweeps)
    te = 0.25
    current = offset + isat * (1.0 - np.exp((phase + 0.1) / te)) * -1.0 + isat
    current = offset + isat * np.exp((phase + 0.1) / te) - isat
    return current * -1.0 + 2 * offset, phase


def test_sweep_jsat_recovers_the_ion_plateau_over_the_offset():
    current = np.full(40 * 500, 0.1)  # offset only, ion branch is +0.5 above it
    phase = np.tile(np.linspace(-2.5, 0.5, 500), 40)
    current = 0.1 + 0.5 * np.where(phase < -1.0, 1.0, np.exp(-(phase + 1.0) * 6))
    jsat, response = langmuir.sweep_jsat(current, phase, 500, 0.1)
    assert jsat == pytest.approx(np.full(40, 0.5), abs=1e-6)
    assert (response > langmuir.MIN_RESPONSE).all()


def test_sweep_jsat_flat_probe_and_no_sweep():
    phase = np.tile(np.linspace(-2.5, 0.5, 500), 4)
    jsat, response = langmuir.sweep_jsat(np.full(2000, 0.3), phase, 500, 0.3)
    assert jsat == pytest.approx(np.zeros(4)) and response.max() < 1e-9
    jsat, _ = langmuir.sweep_jsat(np.ones(1000), np.full(1000, 0.1), 500, 0.0)
    assert np.isnan(jsat).all()


def test_probe_pairs_are_current_then_voltage():
    assert langmuir.probe_pairs(6) == [(0, 1), (2, 3), (4, 5)]


# ---- signals -------------------------------------------------------------------------


def test_window_mean_and_empty_windows():
    t = np.arange(0.0, 100.0, 10.0)
    y = np.arange(10.0)
    m = signals.window_mean(t, y, np.array([50.0, 500.0]), 20.0)
    assert m[0] == pytest.approx(5.0) and np.isnan(m[1])


def test_input_power_gate_constants_are_ordered():
    assert th.AFRAC_DETACHED_MAX < th.AFRAC_ATTACHED_MIN
    assert th.PRAD_ATTACHED_MAX < th.PRAD_DETACHED_MIN
    assert th.DZ_ATTACHED_MAX < th.DZ_DETACHED_MIN < th.DZ_MARFE_MIN


def test_positioned_jsat_selects_sol_probe_instead_of_private_flux_probe():
    positions = np.array([[1.494, -1.25], [1.503, -1.25], [1.512, -1.25]])
    strike = np.array([[1.5, -1.25]])
    which, valid, reason = afrac.select_sol_probe(
        np.ones((3, 1)), positions, strike, np.array([[0.993], [1.02], [1.03]])
    )
    assert which.tolist() == [2]
    assert valid.tolist() == [True]
    assert reason.tolist() == [""]


def test_positioned_jsat_rejects_private_flux_uncertain_and_distant_probes():
    which, valid, reason = afrac.select_sol_probe(
        np.ones((1, 4)),
        np.array([[1.51, -1.25]]),
        np.array([[1.5, -1.25], [1.5, -1.25], [1.48, -1.25], [1.5, -1.25]]),
        np.array([[0.993, 1.005, 1.1, np.nan]]),
    )
    assert which.tolist() == [-1, -1, -1, -1]
    assert not valid.any()
    assert reason.tolist() == [
        "probe_not_sol",
        "probe_not_sol",
        "probe_far_from_strike",
        "probe_flux_unknown",
    ]
