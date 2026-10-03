"""The Prad and Afrac indicators and the Langmuir sweep reader."""

import numpy as np
import pytest

from labeler.events.detachment import afrac, core, langmuir, prad, signals
from labeler.events.detachment import thresholds as th

EDGES = core.bin_edges(0.0, 5000.0)
N = len(EDGES) - 1


def series(value, dt=10.0, t1=5000.0):
    # Cover the centered 250 ms normalization window at both label-grid edges.
    t = np.arange(-150.0, t1 + 150.0, dt)
    return t, np.full(t.shape, value, dtype=float)


def clean_elm():
    t = np.arange(-150.0, 5151.0)
    return t, np.zeros(len(t), bool)


def test_prad_votes_follow_the_fraction():
    t, p_in = series(4e6)
    _, low = series(1.0e6)  # f = 0.25
    _, mid = series(1.7e6)  # f = 0.425, between the cutoffs
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
    short = core.bin_edges(0.0, 1000.0)
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
    assert ind.valid.all()  # No unsourced three-second duration gate.
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


def test_power_window_preserves_nan_and_internal_gaps():
    t = np.arange(101.0)
    y = np.ones(len(t))
    y[(t >= 40) & (t <= 60)] = np.nan
    assert np.isnan(signals.window_mean(t, y, [50.0], 20.0)[0])
    keep = (t < 40) | (t > 60)
    assert np.isnan(signals.window_mean(t[keep], y[keep], [50.0], 50.0)[0])


def test_power_window_checks_missing_upper_endpoint_in_its_closed_mean():
    t = np.arange(101.0)
    y = t.copy()
    assert signals.window_mean(t, y, [50.0], 20.0)[0] == 50.0
    y[t == 60.0] = np.nan
    assert np.isnan(signals.window_mean(t, y, [50.0], 20.0)[0])


def test_heating_missing_beam_samples_are_not_zero(monkeypatch):
    t = np.arange(0.0, 401.0, 10.0)
    beam = np.full((2, len(t)), 1e6)
    beam[:, (t >= 180) & (t <= 220)] = np.nan
    monkeypatch.setattr(signals, "corpus_group", lambda *a, **kw: (t, beam))
    cache = {"poh": (t, np.full(len(t), 1e5)), "wmhd": (t, np.ones(len(t)))}
    _, heat, _ = signals.heating_power(1, cache)
    assert np.isnan(heat[t == 200]).all()
    assert heat[t == 100] == pytest.approx([2.1e6])


def test_elm_nan_gap_remains_unknown_without_poisoning_other_bins(monkeypatch):
    t = np.arange(0.0, 1000.1, 0.1)
    y = np.ones((4, len(t)))
    y[:, (t >= 100) & (t < 150)] = np.nan
    monkeypatch.setattr(signals, "corpus_group", lambda *a, **kw: (t, y))
    tm, flag = signals.elm_mask(1, cache={})
    known = core.elm_bin_known(np.arange(0.0, 301.0, 50.0), tm, flag)
    assert known.tolist() == [True, True, False, True, True, True]
    assert np.isnan(flag[(tm >= 100) & (tm < 150)]).all()
    assert core.elm_at([125.0], tm, flag).all()


def test_prad_uses_quarter_second_power_mean_at_beam_blip():
    t = np.arange(0.0, 1001.0)
    power = np.where((t >= 500) & (t < 550), 1.4e6, 1e6)
    edges = np.array([500.0, 550.0])
    ind = prad.prad_indicator(
        edges, t, np.full(len(t), 0.48e6), t, power, t, np.zeros(len(t))
    )
    assert ind.valid[0]
    assert 0.43 < ind.value[0] < 0.46
    assert ind.vote[0] == core.ABSTAIN


def test_prad_rejects_large_negative_offset_and_uncovered_power():
    t = np.arange(0.0, 1001.0)
    edges = np.array([500.0, 550.0])
    p = np.full(len(t), 1e6)
    ind = prad.prad_indicator(
        edges, t, np.full(len(t), -0.1e6), t, p, t, np.zeros(len(t))
    )
    assert not ind.valid[0] and ind.reason[0] == "negative_radiation"
    ind = prad.prad_indicator(
        edges, t, np.full(len(t), -0.01e6), t, p, t, np.zeros(len(t))
    )
    assert ind.valid[0] and ind.value[0] == 0.0
    p[(t >= 600) & (t <= 625)] = np.nan
    ind = prad.prad_indicator(
        edges, t, np.full(len(t), 0.5e6), t, p, t, np.zeros(len(t))
    )
    assert not ind.valid[0] and ind.reason[0] == "no_input_power"


def test_prad_smoothing_cannot_hide_negative_radiation_in_label_bin():
    t = np.arange(1001.0)
    radiation = np.where((t >= 500) & (t < 550), -0.1e6, 1e6)
    ind = prad.prad_indicator(
        np.array([500.0, 550.0]),
        t,
        radiation,
        t,
        np.full(len(t), 2e6),
        t,
        np.zeros(len(t)),
    )
    assert ind.value[0] > 0.0  # Positive smoothed value is not measurement validity.
    assert not ind.valid[0]
    assert ind.reason[0] == "negative_radiation"
    assert ind.vote[0] == core.ABSTAIN


def test_prad_averaging_window_requires_elm_availability_outside_native_bin():
    t = np.arange(0.0, 1001.0)
    radiation = np.where((t >= 500.0) & (t < 550.0), 0.3e6, 1.0e6)
    power = np.full(t.shape, 1.0e6)
    edges = np.array([500.0, 550.0])
    covered = prad.prad_indicator(edges, t, radiation, t, power, t, np.zeros(t.shape))
    assert covered.valid[0] and covered.vote[0] == core.DETACHED
    elm_t = np.arange(500.0, 551.0)
    unknown = prad.prad_indicator(
        edges, t, radiation, t, power, elm_t, np.zeros(elm_t.shape)
    )
    assert not unknown.valid[0]
    assert unknown.reason[0] == "elm_unknown"
    assert unknown.vote[0] == core.ABSTAIN


def test_input_power_gate_constants_are_ordered():
    assert th.AFRAC_DETACHED_MAX < th.AFRAC_ATTACHED_MIN
    assert th.PRAD_ATTACHED_MAX < th.PRAD_DETACHED_MIN
    assert th.PRAD_REL_ATTACHED_MAX < th.PRAD_REL_DETACHED_MIN
    assert th.DZ_ATTACHED_MAX < th.DZ_DETACHED_MIN < th.DZ_MARFE_MIN


def test_prad_anchor_constants_match_the_measured_record():
    import json
    from pathlib import Path

    path = (
        Path(__file__).parents[2] / "docs/labeler/results/detachment_prad_anchor.json"
    )
    measured = json.loads(path.read_text())["measured_anchor"]
    assert th.PRAD_ANCHOR_SHOT == json.loads(path.read_text())["shot"]
    assert th.PRAD_ANCHOR_P_IN_MW == pytest.approx(measured["p_in_mw"], abs=1e-3)
    assert th.PRAD_ANCHOR_ATTACHED_MW == pytest.approx(
        measured["attached_mw"], abs=1e-3
    )
    assert th.PRAD_ANCHOR_DETACHED_MW == pytest.approx(
        measured["detached_mw"], abs=1e-3
    )


def test_prad_cutoffs_sit_between_the_measured_anchor_values():
    lo, hi = th.prad_cutoffs()
    p_in = th.PRAD_ANCHOR_P_IN_MW
    attached = th.PRAD_ANCHOR_ATTACHED_MW / p_in
    detached = th.PRAD_ANCHOR_DETACHED_MW / p_in
    assert attached < lo < hi < detached
    assert 0.5 * (lo + hi) == pytest.approx(0.5 * (attached + detached))
    assert hi - lo == pytest.approx(2 * th.PRAD_BAND_MW / p_in)
    # a wider band moves both cutoffs outward, never across the midpoint
    wide_lo, wide_hi = th.prad_cutoffs(band_mw=0.2)
    assert wide_lo < lo < hi < wide_hi
    rel_lo, rel_hi = th.prad_relative_cutoffs()
    assert (
        1.0 < rel_lo < rel_hi < th.PRAD_ANCHOR_DETACHED_MW / th.PRAD_ANCHOR_ATTACHED_MW
    )


def test_prad_votes_use_the_cutoffs_passed_in():
    f = np.array([0.30, 0.42, 0.50, np.nan])
    default = prad.fdiv_vote(f)
    assert default.tolist() == [
        core.ATTACHED,
        core.ABSTAIN,
        core.DETACHED,
        core.ABSTAIN,
    ]
    shifted = prad.fdiv_vote(f, attached_max=0.45, detached_min=0.55)
    assert shifted.tolist() == [
        core.ATTACHED,
        core.ATTACHED,
        core.ABSTAIN,
        core.ABSTAIN,
    ]


def test_relative_fdiv_reads_the_change_from_the_shots_own_baseline():
    # a flat-top shot at 0.3 whose last third detaches to 0.45; the beam ramp-up
    # bins read 0.1 for want of power and are not the unseeded baseline
    f = np.r_[np.full(20, 0.10), np.full(60, 0.30), np.full(30, 0.45)]
    p_in = np.r_[np.full(20, 1.0e6), np.full(90, 4.0e6)]
    valid = np.ones(len(f), bool)
    ratio = prad.relative_fdiv(f, valid, p_in)
    assert ratio[20:80] == pytest.approx(np.ones(60))
    assert ratio[80:] == pytest.approx(np.full(30, 1.5))
    votes = prad.relative_vote(ratio)
    assert (votes[20:80] == core.ATTACHED).all() and (votes[80:] == core.DETACHED).all()
    # without the power guard the ramp-up sets the baseline and everything reads high
    assert np.nanmin(prad.relative_fdiv(f, valid)[20:]) == pytest.approx(3.0)


def test_relative_fdiv_needs_a_baseline():
    f = np.full(th.PRAD_BASELINE_MIN_BINS - 1, 0.3)
    assert np.isnan(prad.relative_fdiv(f, np.ones(len(f), bool))).all()
    f = np.full(60, 0.3)
    valid = np.zeros(60, bool)
    valid[:10] = True
    assert np.isnan(prad.relative_fdiv(f, valid)).all()
    ratio = prad.relative_fdiv(
        np.r_[np.full(50, 0.3), np.nan], np.r_[np.ones(50, bool), False]
    )
    assert np.isnan(ratio[-1]) and ratio[0] == pytest.approx(1.0)


#: 189057 at 3000 ms (EFIT02 slice): the outer strike point and the processed
#: Langmuir probes p12-p18 on the divertor shelf at Z = -1.249 m, with psiN read
#: off the multi-slice flux map and the measured median Jsat (A/cm^2). Flux
#: grows only ~0.4 per metre here, so psiN = 1.01 lies 2.3 cm from the strike.
SHELF_STRIKE = np.array([[1.4928, -1.2448]])
SHELF_R = [1.5027, 1.5316, 1.5591, 1.5870, 1.6148, 1.6430, 1.6702]
SHELF_POSITIONS = np.array([[r, -1.2494] for r in SHELF_R])
SHELF_PSI_N = np.array(
    [[1.0026], [1.0153], [1.0289], [1.0444], [1.0614], [1.0806], [1.1008]]
)
SHELF_JSAT = np.array([[5.847], [5.583], [3.353], [2.115], [1.207], [1.102], [0.859]])


def test_positioned_jsat_takes_the_peak_current_in_the_near_sol_window():
    which, valid, reason = afrac.select_sol_probe(
        SHELF_JSAT, SHELF_POSITIONS, SHELF_STRIKE, SHELF_PSI_N
    )
    # p12 sits 9.9 mm outboard at psiN 1.003: SOL side, so it is eligible and has
    # the peak; the former 1.01 flux cut and 2 cm cap rejected it
    assert (
        which.tolist() == [0] and valid.tolist() == [True] and reason.tolist() == [""]
    )


def test_positioned_jsat_follows_the_peak_outward_as_the_target_detaches():
    jsat = SHELF_JSAT.copy()
    jsat[0] = 0.8  # the probe nearest the strike point has lost its current
    which, valid, _ = afrac.select_sol_probe(
        jsat, SHELF_POSITIONS, SHELF_STRIKE, SHELF_PSI_N
    )
    assert which.tolist() == [1] and valid.tolist() == [True]
    # a larger current beyond the window (psiN > 1.05) is never selected
    jsat[4] = 40.0
    which, _, _ = afrac.select_sol_probe(
        jsat, SHELF_POSITIONS, SHELF_STRIKE, SHELF_PSI_N
    )
    assert which.tolist() == [1]


def test_positioned_jsat_rejections_name_the_reason():
    positions = np.array([[1.45, -1.25], [1.503, -1.25], [1.52, -1.25]])
    strike = np.tile([1.5, -1.25], (4, 1))
    psi_n = np.array(
        [
            [0.99, 1.02, np.nan, 1.08],  # probe 0: inboard of the strike point
            [1.002, 1.001, np.nan, 1.08],  # probe 1: only 3 mm outboard
            [1.02, 0.998, np.nan, 1.07],  # probe 2: 2 cm outboard
        ]
    )
    which, valid, reason = afrac.select_sol_probe(
        np.ones((3, 4)), positions, strike, psi_n
    )
    # bin 0 is valid on probe 2; bin 1 has probe 2 inside the separatrix; bin 2 has
    # no flux anywhere; bin 3 is outside the separatrix but beyond psiN 1.05
    assert which.tolist() == [2, -1, -1, -1]
    assert valid.tolist() == [True, False, False, False]
    assert reason.tolist() == [
        "",
        "probe_not_sol",
        "probe_flux_unknown",
        "probe_beyond_sol_window",
    ]


def test_positioned_jsat_eligible_probe_without_current():
    which, valid, reason = afrac.select_sol_probe(
        np.array([[np.nan, 0.0]]),
        np.array([[1.52, -1.25]]),
        np.tile([1.5, -1.25], (2, 1)),
        np.array([[1.02, 1.02]]),
    )
    assert which.tolist() == [-1, -1] and not valid.any()
    assert reason.tolist() == ["no_probe_samples", "no_probe_samples"]


def test_reported_probe_names_the_nearest_known_probe_on_invalid_bins():
    strike = np.array([[1.5, -1.25], [1.5, -1.25]])
    positions = np.array([[1.45, -1.25], [1.503, -1.25], [1.58, -1.25]])
    psi_n = np.array([[0.99, np.nan], [1.002, np.nan], [1.04, np.nan]])
    which, valid, reason = afrac.select_sol_probe(
        np.ones((3, 2)), positions, strike, psi_n
    )
    assert valid.tolist() == [True, False] and reason[1] == "probe_flux_unknown"
    reported = afrac.reported_probe(positions, strike, psi_n, which, valid)
    # bin 0 reports the chosen probe, bin 1 (no flux anywhere) reports none
    assert reported.tolist() == [which[0], -1]
    psi_n[:, 1] = [0.99, 1.002, 1.04]
    which, valid, reason = afrac.select_sol_probe(
        np.ones((3, 2)), positions, strike, psi_n
    )
    reported = afrac.reported_probe(positions, strike, psi_n, which, valid)
    assert reported[1] == which[1]


def test_beam_power_falls_back_to_the_bms_total(monkeypatch):
    monkeypatch.setattr(signals, "corpus_group", lambda *a, **k: None)
    t = np.arange(0.0, 1000.0, 1.0)
    cache = {"pinj_bms": (t, np.full(len(t), 4.0, dtype=np.float32))}
    got = signals.beam_power(0, cache, np.array([500.0]), 250.0)
    assert got == pytest.approx([4.0e6])
    assert signals.beam_power(0, {}, np.array([500.0]), 250.0) is None
    # the NB total in kW outranks the BMS record when both exist
    cache["pinj_total"] = (t, np.full(len(t), 3500.0, dtype=np.float32))
    assert signals.beam_power(0, cache, np.array([500.0]), 250.0) == pytest.approx(
        [3.5e6]
    )
