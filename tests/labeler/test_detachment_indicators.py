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


def afrac_run(jsat, psi=None, *, density=1e14, power=4e6, **kwargs):
    """`afrac_indicator` on per-bin probe currents (probe, bin) at constant ne, P."""
    jsat = np.atleast_2d(np.asarray(jsat, dtype=float))
    psi = np.full(jsat.shape, 1.0) if psi is None else np.asarray(psi, dtype=float)
    tn, ne = series(density) if np.isscalar(density) else density
    tp, psol = series(power)
    elm_t, elm_flag = clean_elm()
    kwargs.setdefault("elm_t_ms", elm_t)
    kwargs.setdefault("elm_flag", elm_flag)
    return afrac.afrac_indicator(EDGES, jsat, psi, tn, ne, tp, psol, **kwargs)


def test_afrac_attached_then_detached():
    # constant density and power; Jsat falls to a fifth in the second half
    jsat = np.where(core.bin_centres(EDGES) < 2500, 1.0, 0.2)
    ind, which, refs = afrac_run(jsat)
    first, second = EDGES[:-1] < 2500, EDGES[:-1] >= 2500
    assert ind.valid.all() and (which == 0).all()
    assert (ind.vote[first] == core.ATTACHED).all()
    assert (ind.vote[second] == core.DETACHED).all()
    assert ind.value[first] == pytest.approx(np.ones(first.sum()))
    assert refs == pytest.approx([1.0 / (1e14**2 * (4e6) ** (-3 / 7))])


def test_afrac_model_takes_out_the_density_scaling():
    tn = np.arange(-150.0, 5151.0, 10.0)
    ne = np.where(tn < 2500, 1e14, 2e14)
    # Jsat quadruples with the doubled density: the model says that is still attached
    jsat = np.where(core.bin_centres(EDGES) < 2500, 1.0, 4.0)
    ind, *_ = afrac_run(jsat, density=(tn, ne))
    assert (ind.vote == core.ATTACHED).all()


def test_afrac_peak_over_probes():
    t = np.arange(0.5, 1000.0, 1.0)
    jsat = np.vstack([np.where(t < 500, 1.0, 0.1), np.where(t < 500, 0.1, 1.0)])
    edges = core.bin_edges(0.0, 1000.0)
    peak, which = afrac.peak_jsat(edges, t, jsat)
    assert peak == pytest.approx(np.ones(len(edges) - 1))
    assert which[0] == 0 and which[-1] == 1


def test_afrac_does_not_depend_on_which_probe_is_read():
    # An attached plasma throughout; the strike point moves, so the probe nearest
    # the separatrix changes from p0 (a strong probe) to p1 (ten times weaker).
    # One shared reference would read the change as a detachment; each probe's own
    # reference reads it as attached on both sides.
    first = core.bin_centres(EDGES) < 2500
    jsat = np.vstack([np.where(first, 10.0, 9.0), np.full(N, 1.0)])
    psi = np.vstack([np.where(first, 1.003, 1.05), np.full(N, 1.006)])
    ind, which, refs = afrac_run(jsat, psi)
    assert ind.valid.all()
    assert (which[first] == 0).all() and (which[~first] == 1).all()
    assert (ind.vote == core.ATTACHED).all()
    assert ind.value[first] == pytest.approx(np.ones(first.sum()))
    assert ind.value[~first] == pytest.approx(np.ones((~first).sum()))
    assert refs[0] / refs[1] == pytest.approx(10.0)


def test_afrac_reads_the_probe_nearest_the_separatrix_inside_the_window():
    jsat = np.vstack([np.full(N, 1.0), np.full(N, 3.0), np.full(N, 9.0)])
    psi = np.vstack([np.full(N, 1.008), np.full(N, 0.996), np.full(N, 1.02)])
    ind, which, refs = afrac_run(jsat, psi)
    assert (which == 1).all() and ind.valid.all()
    # the probe at psiN 1.02 is outside the window: no reference, never read
    assert np.isnan(refs[2]) and np.isfinite(refs[:2]).all()


def test_afrac_l_mode_bins_abstain_and_stay_out_of_the_reference():
    centres = core.bin_centres(EDGES)
    lmode = (centres >= 1000) & (centres < 2000)
    jsat = np.where(lmode, 100.0, 1.0)
    ind, _, refs = afrac_run(jsat, lmode=lmode)
    assert (ind.reason[lmode] == "l_mode").all() and not ind.valid[lmode].any()
    assert (ind.vote[lmode] == core.ABSTAIN).all()
    assert ind.valid[~lmode].all()
    assert ind.value[~lmode] == pytest.approx(np.ones((~lmode).sum()))
    assert refs == pytest.approx([1.0 / (1e14**2 * (4e6) ** (-3 / 7))])


def test_afrac_probe_reasons():
    ok = np.ones(N)
    cases = {
        "probe_flux_unknown": (ok, np.full(N, np.nan)),
        "probe_off_separatrix": (ok, np.full(N, 1.05)),
        "no_probe_samples": (np.zeros(N), np.full(N, 1.0)),
    }
    for reason, (jsat, psi) in cases.items():
        ind, which, _ = afrac_run(jsat, psi)
        assert set(ind.reason) == {reason}, reason
        assert not ind.valid.any() and (which == -1).all()
    # a reference needs AFRAC_REFERENCE_MIN_MS of bins near the separatrix: 20 bins
    # on this 50 ms grid
    need = th.min_bins(th.AFRAC_REFERENCE_MIN_MS, 50.0)
    assert need == 20
    psi = np.full(N, 1.05)
    psi[: need - 1] = 1.0
    ind, *_ = afrac_run(ok, psi)
    assert "short_reference" in set(ind.reason) and not ind.valid.any()
    psi[:need] = 1.0
    ind, *_ = afrac_run(ok, psi)
    assert ind.valid[:need].all()


def test_afrac_power_on_the_efit_time_base_is_interpolated_to_the_bin():
    # P_SOL every 40 ms: a bin median would leave every other 20 ms bin without a
    # sample (reason no_power); the value at the bin centre does not
    width = 20.0
    edges = core.bin_edges(0.0, 5000.0, width)
    n = len(edges) - 1
    tn, ne = series(1e14)
    tp = np.arange(10.0, 5000.0, 40.0)
    psol = np.full(tp.shape, 4e6)
    elm_t, elm_flag = clean_elm()
    psi = np.full((1, n), 1.0)
    ind, *_ = afrac.afrac_indicator(
        edges,
        np.ones((1, n)),
        psi,
        tn,
        ne,
        tp,
        psol,
        elm_t_ms=elm_t,
        elm_flag=elm_flag,
    )
    inside = (core.bin_centres(edges) >= tp[0]) & (core.bin_centres(edges) <= tp[-1])
    assert "no_power" not in set(ind.reason[inside])
    assert ind.valid[inside & (core.bin_centres(edges) > 1200.0)].all()


def test_afrac_reference_length_is_a_duration_not_a_bin_count():
    # the same second of near-separatrix time defines a reference at any width
    tn, ne = series(1e14)
    tp, psol = series(4e6)
    elm_t, elm_flag = clean_elm()
    for width in (20.0, 50.0, 100.0):
        edges = core.bin_edges(0.0, 5000.0, width)
        n = len(edges) - 1
        need = th.min_bins(th.AFRAC_REFERENCE_MIN_MS, width)
        assert need * width == pytest.approx(1000.0)
        valid = []
        for near in (need - 1, need):
            psi = np.full((1, n), 1.05)
            psi[0, :near] = 1.0
            ind, *_ = afrac.afrac_indicator(
                edges,
                np.ones((1, n)),
                psi,
                tn,
                ne,
                tp,
                psol,
                elm_t_ms=elm_t,
                elm_flag=elm_flag,
            )
            valid.append(bool(ind.valid.any()))
        assert valid == [False, True]


def test_afrac_invalid_without_inputs():
    jsat = np.ones((1, N))
    psi = np.ones((1, N))
    tn, ne = series(1e14)
    tp, psol = series(4e6)
    assert set(
        afrac.afrac_indicator(EDGES, None, None, tn, ne, tp, psol)[0].reason
    ) == {"no_probes"}
    assert set(
        afrac.afrac_indicator(EDGES, jsat, psi, None, None, tp, psol)[0].reason
    ) == {"no_density"}
    assert set(
        afrac.afrac_indicator(EDGES, jsat, psi, tn, ne, None, None)[0].reason
    ) == {"no_power"}
    low, *_ = afrac_run(jsat, psi, power=0.1e6)
    assert set(low.reason) == {"low_power"}


def test_afrac_ramp_is_invalid():
    ti = np.arange(0.0, 5000.0, 20.0)
    ip = np.where(ti < 1000, ti * 2e3, 2e6)  # 2 MA/s ramp in the first second
    ind, *_ = afrac_run(np.ones(N), ip_t_ms=ti, ip_a=ip)
    ramp = (EDGES[:-1] > 60) & (EDGES[1:] < 950)  # the record's edge reads 0
    assert (ind.reason[ramp] == "ramp").all()
    assert ind.valid[EDGES[:-1] > 1100].all()


def test_afrac_elm_bins_are_invalid():
    te, flag = clean_elm()
    flag[(te >= 1000) & (te < 1050)] = True  # the whole bin is inside an ELM
    ind, *_ = afrac_run(np.ones(N), elm_t_ms=te, elm_flag=flag)
    assert ind.reason[20] == "elm" and not ind.valid[20]
    assert ind.valid[np.arange(N) != 20].all()


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
    one = np.ones((1, N))
    jsat, *_ = afrac.afrac_indicator(EDGES, one, one, tn, ne, t, p)
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
    need = th.min_bins(th.PRAD_BASELINE_MIN_MS, 50.0)
    assert need == 40
    f = np.full(need - 1, 0.3)
    assert np.isnan(prad.relative_fdiv(f, np.ones(len(f), bool))).all()
    # the baseline length is a duration: 100 bins at 20 ms, 20 at 100 ms
    for width, count in ((20.0, 100), (100.0, 20)):
        assert th.min_bins(th.PRAD_BASELINE_MIN_MS, width) == count
        short = np.full(count - 1, 0.3)
        long = np.full(count, 0.3)
        assert np.isnan(
            prad.relative_fdiv(short, np.ones(count - 1, bool), None, width)
        ).all()
        assert np.isfinite(
            prad.relative_fdiv(long, np.ones(count, bool), None, width)
        ).all()
    f = np.full(60, 0.3)
    valid = np.zeros(60, bool)
    valid[:10] = True
    assert np.isnan(prad.relative_fdiv(f, valid)).all()
    ratio = prad.relative_fdiv(
        np.r_[np.full(50, 0.3), np.nan], np.r_[np.ones(50, bool), False]
    )
    assert np.isnan(ratio[-1]) and ratio[0] == pytest.approx(1.0)


def test_exported_fdiv_votes_on_the_relative_ratio_and_needs_a_baseline():
    f = np.r_[np.full(60, 0.3), np.full(30, 0.45)]
    absolute = prad.fdiv_vote(f)
    valid = np.ones(len(f), bool)
    ind = core.assemble("prad", f, valid, np.full(len(f), ""), absolute)
    ratio = prad.relative_fdiv(f, valid)
    out = prad.with_relative_vote(ind, ratio)
    assert (out.vote[:60] == core.ATTACHED).all()
    assert (out.vote[60:] == core.DETACHED).all()
    # a shot without a baseline cannot vote: invalid, reason no_baseline
    short = prad.with_relative_vote(ind, np.full(len(f), np.nan))
    assert not short.valid.any() and set(short.reason) == {"no_baseline"}
    assert (short.vote == core.ABSTAIN).all()


def test_reported_probe_names_the_nearest_known_probe_on_invalid_bins():
    psi_n = np.array(
        [[0.99, np.nan, 1.07], [1.002, np.nan, 1.04], [1.04, np.nan, 1.02]]
    )
    which = np.array([1, -1, -1])
    valid = np.array([True, False, False])
    reported = afrac.reported_probe(psi_n, which, valid)
    # bin 0 reports the probe that was read, bin 1 (no flux anywhere) none, bin 2
    # the probe nearest the separatrix although it was too far to vote
    assert reported.tolist() == [1, -1, 2]


def test_afrac_window_is_one_decision_not_a_fitted_number():
    assert th.AFRAC_PSI_WINDOW == pytest.approx(0.01)
    assert th.AFRAC_REFERENCE_MIN_MS == pytest.approx(1000.0)
    assert th.min_bins(th.AFRAC_REFERENCE_MIN_MS, 50.0) == 20


def test_regime_prefers_the_table_then_the_hmode_detector(monkeypatch, tmp_path):
    import pandas as pd

    from labeler.events import spans

    monkeypatch.setenv("LABELER_ROOT", str(tmp_path))
    centres = np.array([100.0, 600.0, 1100.0, 1600.0])
    found = spans.Found(
        spans=((500.0, 1200.0, spans.PRESENT), (1200.0, 1700.0, spans.UNCERTAIN)),
        measured=((0.0, 1700.0),),
    )
    monkeypatch.setattr(spans, "detect_hmode", lambda shot, paths: found)
    got, source = signals.regime(7, centres)
    # L where the method measured outside its H-mode spans; an uncertain span and
    # time outside what it measured are unknown, not L
    assert got.tolist() == ["L", "H", "H", "unknown"] and source == "dalpha_detector"
    table = tmp_path / signals.REGIME_TABLE
    table.parent.mkdir(parents=True)
    rows = [(7, 5, 0, 900), (7, 1, 900, 2000), (8, 2, 0, 2000)]
    pd.DataFrame(rows, columns=["shot", "category", "t_start", "t_end"]).to_csv(
        table, index=False
    )
    monkeypatch.setattr(spans, "detect_hmode", lambda *a: pytest.fail("table first"))
    got, source = signals.regime(7, centres)
    assert got.tolist() == ["unknown", "unknown", "H", "H"] and source == "regime_table"
    got, source = signals.regime(8, centres)
    assert set(got) == {"L"} and source == "regime_table"


def test_regime_is_unknown_without_the_detectors_inputs(monkeypatch, tmp_path):
    from labeler.events import spans

    monkeypatch.setenv("LABELER_ROOT", str(tmp_path))

    def missing(shot, paths):
        raise KeyError("no co2")

    monkeypatch.setattr(spans, "detect_hmode", missing)
    got, source = signals.regime(7, np.array([100.0, 600.0]))
    assert got.tolist() == ["unknown", "unknown"] and source == "none"


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


def regime_inputs(n, *, share=0.0, power=1.2e6, known=True):
    unknown = np.full(n, "unknown")
    return {
        "regime": unknown,
        "source": np.full(n, "none"),
        "elm_known": np.full(n, known),
        "elm_share": np.full(n, share, dtype=float),
        "p_in_w": np.full(n, power, dtype=float),
        "width_ms": 50.0,
    }


def test_probable_l_is_elm_free_known_coverage_and_under_2_mw():
    n = 40  # 2 s of 50 ms bins, more than the 1 s the window needs
    regime, source = signals.probable_regimes(**regime_inputs(n))
    assert set(regime) == {"probable_L"} and set(source) == {"probable_L"}
    assert signals.GATED_REGIMES == ("L", "probable_L")
    # a power at or above the cut, or any ELM share, is not probable L
    regime, _ = signals.probable_regimes(**regime_inputs(n, power=2.5e6))
    assert set(regime) == {"unknown"}
    kw = regime_inputs(n)
    kw["elm_share"][7] = 0.2
    regime, source = signals.probable_regimes(**kw)
    assert set(regime) == {"probable_H"} and set(source) == {"probable_H"}
    # unknown ELM coverage is not "no ELMs"
    regime, _ = signals.probable_regimes(**regime_inputs(n, share=np.nan, known=False))
    assert set(regime) == {"unknown"}


def test_probable_regimes_leave_known_regimes_and_short_windows_alone():
    n = 40
    kw = regime_inputs(n)
    kw["regime"] = np.array(["H"] * 10 + ["L"] * 10 + ["unknown"] * 20)
    kw["source"] = np.array(["regime_table"] * 20 + ["none"] * 20)
    regime, source = signals.probable_regimes(**kw)
    assert regime[:20].tolist() == ["H"] * 10 + ["L"] * 10
    assert set(regime[20:]) == {"probable_L"}
    assert set(source[:20]) == {"regime_table"}
    # the window must last min_bins(1000 ms): 20 bins at 50 ms, 10 at 100 ms
    short = regime_inputs(19)
    assert set(signals.probable_regimes(**short)[0]) == {"unknown"}
    wide = {**regime_inputs(10), "width_ms": 100.0}
    assert set(signals.probable_regimes(**wide)[0]) == {"probable_L"}
    assert th.min_bins(th.PROBABLE_REGIME_MIN_MS, 20.0) == 50
