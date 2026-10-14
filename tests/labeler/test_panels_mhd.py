"""The taxonomy's panel work: short titles, the MHD pages, the markers, the ELM
rows reimplemented from Eldon's module and the H-mode edge Te row.

Everything runs on synthetic corpus files (`editor_tree`); no archive or fdp.
"""

from __future__ import annotations

import ast
import inspect
import re

import numpy as np
import pytest

from labeler import frames
from labeler.events import equilibrium, panels
from labeler.events.panels import _shared, ece_geometry, mhd, mhd_1_1
from labeler.events.panels import edge_localized_mode as elm
from labeler.events.panels import high_confinement_mode as hmode
from labeler.events.panels import neoclassical_tearing_mode as ntm
from labeler.events.verify import NoDataError, Panel
from labeler.features.store import FeatureArray, write_features
from labeler.frames import features as frames_features

from . import editor_tree as tree

SHOT = 201240


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    """No archive, no fdp: a signal a test did not write is missing."""
    nothing = lambda *args, **kwargs: ({}, {})
    monkeypatch.setattr(equilibrium.resolve_archive, "resolve", nothing)
    monkeypatch.setattr(equilibrium.resolve_fdp, "resolve", nothing)
    tree.no_fetch(monkeypatch)


def _caveat(panel) -> str:
    return (panel.metadata or {}).get("caveat", "")


def _by_title(built, title):
    [one] = [panel for panel in built if panel.title == title]
    return one


# --- every title is short -----------------------------------------------------

#: The calls whose positional argument at this index is a row title.
TITLE_ARGUMENT = {
    "Panel": 0,
    "_trace": 0,
    "power_panel": 0,
    "line_panel": 2,
    "_context_panel": 1,
    "scalar": 1,
    "Dalpha": 2,
}
MODULES = {**panels.BUILDERS, "_shared": _shared}


def _text(node):
    """A string literal, or an f-string with each field read as "00"; else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value if isinstance(v, ast.Constant) else "00" for v in node.values
        )
    return None


def title_strings(module) -> set[str]:
    """Every row title a module writes: `title=` keywords, the title argument of
    the calls above, `*TITLE*` constants and the second item of a row tuple."""
    found = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            found.update(_text(k.value) for k in node.keywords if k.arg == "title")
            at = TITLE_ARGUMENT.get(name)
            if at is not None and len(node.args) > at:
                found.add(_text(node.args[at]))
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any("TITLE" in n for n in names):
                found.update(
                    _text(c)
                    for c in ast.walk(node.value)
                    if isinstance(c, (ast.Constant, ast.JoinedStr))
                )
        elif isinstance(node, ast.Tuple) and len(node.elts) >= 3:
            first, second = _text(node.elts[0]), _text(node.elts[1])
            if first is not None and second is not None:
                found.add(second)
    return {t for t in found if t}


@pytest.mark.parametrize("name", sorted(MODULES))
def test_every_title_a_module_writes_fits_the_row(name):
    titles = title_strings(MODULES[name])
    too_long = sorted(t for t in titles if len(t) > _shared.TITLE_MAX)
    assert not too_long, f"{name}: titles over {_shared.TITLE_MAX} characters"


def test_the_scan_finds_the_titles_it_is_meant_to():
    every = set().union(*(title_strings(m) for m in MODULES.values()))
    assert {
        "ECE Te, ch 00-00",
        "Thomson core Te",
        "ECE side comparison",
        "D-alpha FS strike sum",
        "ELM jump-and-hold",
        "W_MHD",
        "edge Te (ECE)",
        "n=1..3 amplitude",
        "q_min, q95 (EFIT01)",
        "qmin (EFIT01 aeqdsk)",
        "TangTV front DZ",
    } <= every
    assert max(map(len, every)) <= _shared.TITLE_MAX


def _digits_off(text: str) -> str:
    return re.sub(r"\d", "0", text)


@pytest.mark.parametrize("name", sorted(frames.SPECS))
def test_every_frames_role_is_still_a_row_title_prefix(name):
    spec = frames.SPECS[name]
    titles = [_digits_off(t) for t in title_strings(panels.BUILDERS[spec.store_event])]
    for role in spec.roles:
        assert any(t.startswith(_digits_off(role.title)) for t in titles), role.title


def test_with_caveat_joins_notes_ends_each_with_a_stop_and_skips_empty_ones():
    base = Panel(title="x", x=np.arange(3.0), y=np.zeros((1, 3)))
    assert _shared.with_caveat(base, "", "  ") is base
    once = _shared.with_caveat(base, "One", "", "Two.")
    assert _caveat(once) == "One. Two."
    assert _caveat(_shared.with_caveat(once, "Three")) == "One. Two. Three."
    assert base.metadata is None or "caveat" not in base.metadata


# --- the toroidal-n score and the n=1..3 amplitude ---------------------------


def test_mode_numbers_scores_each_cell_by_how_well_the_phases_fit_its_n():
    phi = list(ntm.PROBES.values())
    rng = np.random.default_rng(1)
    spec = np.exp(1j * rng.uniform(0, 2 * np.pi, (len(phi), 4, 5)))
    angle = np.deg2rad(phi)
    spec[:, 1, 2] = np.exp(1j * 2 * angle)  # a perfect n = 2
    spec[:, 3, 0] = np.exp(-1j * 2 * angle)  # the opposite sense: n = -2
    modes = ntm.mode_numbers(spec, phi)
    again, score = ntm.mode_numbers(spec, phi, with_score=True)
    assert modes.dtype == again.dtype == np.int8
    assert score.dtype == np.float32 and score.shape == modes.shape
    np.testing.assert_array_equal(modes, again)
    assert (score >= 0).all() and (score <= 1 + 1e-6).all()
    assert (modes[1, 2], modes[3, 0]) == (2, -2)
    assert score[1, 2] == pytest.approx(1, abs=1e-5)
    assert score[3, 0] == pytest.approx(1, abs=1e-5)
    # Random phases fit no n well.
    assert np.median(score) < 0.8


def test_amplitude_by_n_takes_the_loudest_qualifying_cell_of_each_n_in_band():
    f_khz = [0.5, 5, 10, 25, 40]  # the first and last rows are out of the band
    db = np.array(
        [[50, 50, 50], [10, 20, 5], [30, 15, 25], [8, 40, 12], [99, 99, 99]], float
    )
    modes = np.array([[1, 1, 1], [1, 1, 2], [2, 2, 3], [1, 3, 3], [1, 1, 1]])
    score = np.full(db.shape, 0.9)
    score[1, 1] = 0.79  # the 20 dB cell of n = 1 fits too badly to count
    out = mhd.amplitude_by_n(db, modes, score, f_khz, min_fit=0.8)
    assert out.dtype == np.float32 and out.shape == (3, 3)
    nan = np.nan
    np.testing.assert_array_equal(
        out, np.array([[10, nan, nan], [30, 15, 5], [nan, 40, 25]], dtype=np.float32)
    )
    # Nothing in the band: all gaps.
    assert np.isnan(mhd.amplitude_by_n(db, modes, score, [0.1] * 5, min_fit=0.8)).all()
    # The default gate is 0.95: a 0.9 fit counts for nothing.
    assert mhd.MIN_FIT == 0.95
    assert np.isnan(mhd.amplitude_by_n(db, modes, score, f_khz)).all()


def test_coherent_keeps_only_the_cells_on_a_line_of_adjacent_columns():
    mask = np.zeros((10, 14), dtype=bool)
    mask[3, 1:6] = True  # a line five columns long
    mask[6, 8] = True  # a lone cell
    mask[8, 10:12] = True  # two columns: too short for three
    kept = mhd.coherent(mask)
    assert kept[3, 1:6].all() and kept.sum() == 5
    # A line that drifts one bin a column is still one (within +-1 bin) ...
    drift = np.zeros((8, 8), dtype=bool)
    drift[np.arange(1, 7), np.arange(1, 7)] = True
    assert mhd.coherent(drift)[np.arange(2, 6), np.arange(2, 6)].all()
    # ... the same line two bins a column is not.
    jumpy = np.zeros((9, 6), dtype=bool)
    jumpy[[0, 2, 4, 6], [0, 1, 2, 3]] = True
    assert not mhd.coherent(jumpy).any()


def test_null_angles_are_seeded_permutations_that_move_the_probes():
    phi = list(ntm.PROBES.values())
    draws = mhd.null_angles(phi)
    assert len(draws) == mhd.NULL_DRAWS
    for moved in draws:
        np.testing.assert_array_equal(np.sort(moved), np.sort(phi))
        shift = np.abs((moved - np.asarray(phi) + 180.0) % 360.0 - 180.0).mean()
        assert shift >= mhd.NULL_MIN_SHIFT_DEG
    again = mhd.null_angles(phi)
    assert all(np.array_equal(a, b) for a, b in zip(draws, again, strict=True))
    assert not np.array_equal(draws[0], draws[1])
    assert len(mhd.null_angles(phi, draws=5, seed=3)) == 5


def test_the_amplitude_threshold_is_the_larger_of_12_db_and_the_nulls():
    got = mhd.Amplitude(np.zeros((3, 4), np.float32), np.array([np.nan, 30.0, 5.0]), 0.97)
    np.testing.assert_array_equal(got.threshold_db, [12.0, 30.0, 12.0])


def _noise_shot(p, shot):
    """Gaussian noise on the 29 channels, the cohort window 50-150 ms."""
    t = tree.times(0.0, 250.0, 500_000)
    y = np.random.default_rng(23).standard_normal((29, len(t)))
    tree.write(p.corpus_file(shot), {"mirnov": (t, y)})
    tree.cohort(p, [tree.queue_row(shot, 0, window=(50, 150))])


def test_noise_alone_is_not_flagged_as_a_mode_and_the_gate_is_the_nulls(tmp_path):
    p = tree.paths(tmp_path)
    _noise_shot(p, SHOT)
    s = ntm.spectra(SHOT, paths=p)
    modes, score = ntm.mode_numbers(s.spec, list(ntm.PROBES.values()), with_score=True)
    got = mhd.amplitude(s, modes, score)
    assert got.gate >= mhd.MIN_FIT and got.db.shape == (3, len(s.t_ms))
    flagged = np.isfinite(got.db[:, s.plasma]) & (
        got.db[:, s.plasma] >= got.threshold_db[:, None]
    )
    assert flagged.mean(axis=1).max() < 0.05


def _mirnov_shot(p, shot, n):
    """Quiet noise, a 10 kHz line of toroidal number `n` from 70 to 130 ms, and a
    cohort window of 50-150 ms (the quiet outside it is the floor's)."""
    t = tree.times(0.0, 250.0, 500_000)
    y = np.random.default_rng(11 + n).standard_normal((29, len(t)))
    line = (t >= 70) & (t < 130)
    for row, phi in ntm.PROBES.items():
        y[row, line] += np.cos(2 * np.pi * 10.0 * t[line] + n * np.deg2rad(phi))
    y[:, (t < 50) | (t >= 150)] *= 0.01
    tree.write(p.corpus_file(shot), {"mirnov": (t, y)})
    tree.cohort(p, [tree.queue_row(shot, 0, window=(50, 150))])


@pytest.mark.parametrize("n", [1, 2, 3])
def test_a_mode_of_toroidal_number_n_lights_only_its_own_amplitude_row(tmp_path, n):
    p = tree.paths(tmp_path)
    _mirnov_shot(p, SHOT + n, n)
    amplitude = _by_title(panels.build("mhd_n1", SHOT + n, paths=p), mhd.AMPLITUDE_TITLE)
    inside = (amplitude.x > 75) & (amplitude.x < 125)
    medians = np.median(amplitude.y[:, inside], axis=1)
    assert medians[n - 1] > 20
    assert np.delete(medians, n - 1).max() < 10
    assert amplitude.legend == ["n=1", "n=2", "n=3"]
    assert amplitude.hlines and min(amplitude.hlines) >= mhd.THRESHOLD_DB
    assert amplitude.hlines == sorted(set(amplitude.hlines))
    assert "null-calibrated per shot" in _caveat(amplitude)
    assert "12 dB" in _caveat(amplitude) and "0.95" in _caveat(amplitude)


# --- q_min and q95 ------------------------------------------------------------


def _features(p, shot, **arrays):
    """Features-store scalars on a 5 ms grid to 250 ms; name -> values."""
    t_s = np.arange(0.0, 0.25, 0.005)
    stored = {
        name: FeatureArray(x=t_s, y=np.atleast_2d(np.asarray(y, float) + 0 * t_s), attrs={})
        for name, y in arrays.items()
    }
    write_features(p.features_file(shot), shot, stored, {})


def test_the_q_row_draws_q_min_and_q95_with_a_line_at_each_rational_surface(tmp_path):
    p = tree.paths(tmp_path)
    _features(p, SHOT, qmin=1.4, q95=4.5)
    [row] = mhd.q_panel(SHOT, paths=p)
    assert row.title == "q_min, q95 (EFIT01)" and row.legend == ["q_min", "q95"]
    assert row.hlines == pytest.approx([1.0, 4 / 3, 1.5, 2.0, 3.0])
    assert row.y.shape[0] == 2 and np.allclose(row.y[0], 1.4) and np.allclose(row.y[1], 4.5)
    assert "q = 1, 4/3, 3/2, 2 and 3" in _caveat(row) and "not available" not in _caveat(row)


def test_the_q_row_survives_a_missing_q95_and_says_so_but_not_a_missing_q(tmp_path):
    p = tree.paths(tmp_path)
    _features(p, SHOT, qmin=1.4)
    [row] = mhd.q_panel(SHOT, paths=p)
    assert row.legend == ["q_min"] and row.y.shape[0] == 1
    assert "q95 not available" in _caveat(row)
    with pytest.raises(NoDataError, match="no q_min or q95"):
        mhd.q_panel(SHOT + 9, paths=p)


def test_the_mhd_page_is_the_magnetics_the_amplitude_q_and_beta_n(tmp_path):
    p = tree.paths(tmp_path)
    assert panels.build("mhd_n1", SHOT, paths=p) == []
    _mirnov_shot(p, SHOT, 1)
    _features(p, SHOT, qmin=1.4, q95=4.5, betan=1.8)
    built = panels.build("mhd_n1", SHOT, paths=p)
    assert [b.title for b in built] == [
        "MPI66M322D power",
        "toroidal n, MPI66M",
        "n=1..3 amplitude",
        "q_min, q95 (EFIT01)",
        "beta_N",
    ] == list(mhd.TITLES)
    assert [b.kind for b in built][:2] == ["heatmap", "heatmap"]
    assert panels.BUILDERS["mhd_n1"] is mhd


# --- markers -------------------------------------------------------------------


def test_the_markers_registry_names_the_two_events_that_have_them():
    assert set(panels.MARKERS) == {"mhd_n1.m1", "edge_localized_mode"}
    assert panels.MARKERS["mhd_n1.m1"] is mhd_1_1.markers
    assert panels.MARKERS["edge_localized_mode"] is elm.markers
    assert panels.BUILDERS["mhd_n1.m1"] is mhd_1_1


def _elm_dalpha_shot(p, shot, elms=(50.0, 110.0, 170.0, 230.0), flash=None):
    """FS01-FS08 at 50 kHz, FS01 with an ELM-shaped burst at each of `elms` and,
    with `flash=(t_ms, size)`, a larger one (a disruption's)."""
    t = tree.times(0.0, 300.0, 50_000)
    rng = np.random.default_rng(5)
    fs = np.full((104, len(t)), np.nan)
    fs[:8] = 1.0 + 0.01 * rng.standard_normal((8, len(t)))
    bursts = [(e, 3.0) for e in elms] + ([flash] if flash else [])
    for e, size in bursts:
        after = t >= e
        fs[0, after] += size * (1 - np.exp(-(t[after] - e) / 0.1)) * np.exp(-(t[after] - e) / 2.0)
    tree.write(p.corpus_file(shot), {"filterscopes": (t, fs)})


def test_elm_markers_are_the_jump_and_hold_onsets_inside_the_view(tmp_path):
    p = tree.paths(tmp_path)
    _elm_dalpha_shot(p, SHOT)
    [marker] = elm.markers(SHOT, paths=p)
    assert marker["label"] == "ELM onset"
    assert all(isinstance(t, float) for t in marker["t_ms"])
    np.testing.assert_allclose(marker["t_ms"], [50, 110, 170, 230], atol=0.3)
    [inside] = elm.markers(SHOT, t_range=(100, 200), paths=p)
    np.testing.assert_allclose(inside["t_ms"], [110, 170], atol=0.3)
    assert elm.markers(SHOT, t_range=(280, 290), paths=p) == []
    assert elm.markers(SHOT + 1, paths=p) == []  # no D-alpha at all


def test_elm_markers_stand_inside_the_flat_top_and_leave_out_the_disruptions_flash(
    tmp_path,
):
    p = tree.paths(tmp_path)
    _elm_dalpha_shot(p, SHOT, flash=(275.0, 60.0))
    ts = tree.times(0.0, 300.0, 10_000)
    ip = np.where(ts < 270, 1.0e6, 1.0e6 * np.exp(-(ts - 270) / 1.0))
    tree.write(p.raw_cache / f"{SHOT}_processed.h5", {"ip": (ts, ip[None])})
    [marker] = elm.markers(SHOT, paths=p)
    # The flash (a 60-unit burst) neither sets the scale nor is an onset.
    np.testing.assert_allclose(marker["t_ms"], [50, 110, 170, 230], atol=0.3)


def _sawtooth_ece(p, shot):
    """A 48-channel radiometer at 100 kHz: the core channel (24) ramps and drops
    every 25 ms from 150 to 550 ms, the two off-axis channels do the opposite.
    The cohort window is 100-600 ms; the record is padded either side."""
    t = np.arange(0.0, 700.0, 0.01)
    rng = np.random.default_rng(3)
    ramp = np.zeros_like(t)
    on = (t >= 150) & (t < 550)
    ramp[on] = ((t[on] - 150) % 25) / 25 * 0.5
    core = 24
    y = 1.0 + 0.01 * rng.standard_normal((48, len(t)))
    y[core] = 5.0 + ramp + 0.01 * rng.standard_normal(len(t))
    radius = mhd_1_1.nominal_radius(mhd_1_1.ECE_GHZ, mhd_1_1.ECE_GHZ[core])
    for channel in mhd_1_1._off_axis(radius, core):
        y[channel] = 2.0 - ramp + 0.01 * rng.standard_normal(len(t))
    tree.write(p.corpus_file(shot), {"ece": (t, y)})
    tree.cohort(p, [tree.queue_row(shot, 0, window=(100, 600))])
    return np.arange(175.0, 551.0, 25.0)


def test_the_1_1_page_marks_every_crash_and_draws_no_crash_score(tmp_path):
    p = tree.paths(tmp_path)
    crashes = _sawtooth_ece(p, SHOT)
    [marker] = mhd_1_1.markers(SHOT, paths=p)
    assert marker["label"] == "crash" and len(marker["t_ms"]) == len(crashes)
    assert np.abs(np.array(marker["t_ms"]) - crashes).max() < 1.0
    [inside] = mhd_1_1.markers(SHOT, t_range=(300, 400), paths=p)
    assert all(300 <= t <= 400 for t in inside["t_ms"])
    built = panels.build("mhd_n1.m1", SHOT, paths=p)
    assert [(b.title, b.kind) for b in built] == [("ECE Te map", "heatmap")]
    assert not any("crash" in title.lower() for title in mhd_1_1.TITLES)
    assert mhd_1_1.markers(SHOT + 3, paths=p) == []  # no ECE
    # No Bt0 or axis in the store: the hottest channel, and the caveat says so.
    assert "the hottest" in _caveat(built[0]) and "channel 25" in _caveat(built[0])
    assert "no Bt0 or EFIT axis" in _caveat(built[0])


# --- the 1/1 core channel: Bt0 and the axis, not the hottest ---------------------


def _axis_features(p, shot, bt0, axis_m):
    """Bt and the EFIT axis R, flat, on a 5 ms grid to 700 ms (the cohort's window)."""
    t_s = np.arange(0.0, 0.7, 0.005)
    flat = lambda v: FeatureArray(x=t_s, y=np.full((1, len(t_s)), float(v)), attrs={})
    write_features(p.features_file(shot), shot, {"bt": flat(bt0), "r0": flat(axis_m)}, {})


def _bt_for(channel, axis_m):
    """The Bt0 that puts `channel`'s second-harmonic radius at `axis_m`."""
    return axis_m * mhd_1_1.ECE_GHZ[channel] / (2 * 27.992 * mhd_1_1.FIELD_REFERENCE_M)


def _never(*args):
    raise AssertionError("the hottest channel was read")


def test_the_core_is_the_channel_nearest_the_axis_at_the_windows_bt0(tmp_path):
    p = tree.paths(tmp_path)
    _axis_features(p, SHOT, _bt_for(30, 1.72), 1.72)
    got = mhd_1_1.choose(SHOT, (100.0, 600.0), paths=p, hottest_channel=_never)
    assert (got.core, got.how) == (30, "axis")
    assert got.bt0 == pytest.approx(_bt_for(30, 1.72)) and got.axis_m == pytest.approx(1.72)
    assert got.radius_m[30] == pytest.approx(1.72, abs=1e-6)
    # R = 2 x 27.992 GHz/T x Bt0 x 1.6955 m / f, falling as the frequency rises.
    expected = 2 * 27.992 * got.bt0 * 1.6955 / mhd_1_1.ECE_GHZ
    np.testing.assert_allclose(got.radius_m, expected, rtol=1e-9)
    assert (np.diff(got.radius_m[1:15]) < 0).all()
    assert "Bt0 = " in got.caveat() and "channel 31" in got.caveat()


def test_the_core_is_never_one_of_channels_38_to_40(tmp_path):
    p = tree.paths(tmp_path)
    assert mhd_1_1.EXCLUDED_CHANNELS == (37, 38, 39)
    # An axis where channel 39 (0-based 38) sits: the nearest usable one is chosen.
    _axis_features(p, SHOT, _bt_for(38, 1.60), 1.60)
    got = mhd_1_1.choose(SHOT, (100.0, 600.0), paths=p, hottest_channel=_never)
    assert got.how == "axis" and got.core == 36
    te = np.ones((40, 50))
    te[39], te[20] = 9.0, 5.0  # the band edge is the hottest
    assert mhd_1_1.hottest(te, np.arange(50.0), None) == 20
    # Off-axis channels skip them too.
    radius = np.linspace(2.4, 1.5, 40)
    radius[38] = radius[30] + 0.20
    assert 38 not in mhd_1_1._off_axis(radius, 30)


def test_a_calibrated_ece_psi_chooses_the_core_before_bt0_does(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    _axis_features(p, SHOT, _bt_for(30, 1.72), 1.72)
    psi = np.full((40, 6), 0.5)
    psi[22] = 0.05  # the channel nearest the axis
    psi[38] = 0.01  # a band-edge channel: never
    got = ece_geometry.Geometry(np.linspace(0.0, 700.0, 6), np.zeros((40, 6)), psi, np.ones(6))
    monkeypatch.setattr(ece_geometry, "load_geometry", lambda shot, paths=None: got)
    chosen = mhd_1_1.choose(SHOT, (100.0, 600.0), paths=p, hottest_channel=_never)
    assert (chosen.core, chosen.how) == (22, "psi")
    assert chosen.radius_m[30] == pytest.approx(1.72, abs=1e-6)  # still Bt0's radii
    assert "least calibrated ECE psi" in chosen.caveat()


def test_without_bt0_or_the_axis_the_hottest_channel_stands_at_the_nominal_axis(tmp_path):
    p = tree.paths(tmp_path)
    got = mhd_1_1.choose(SHOT, None, paths=p, hottest_channel=lambda: 7)
    assert (got.core, got.how) == (7, "hottest")
    assert got.radius_m[7] == pytest.approx(mhd_1_1.R_AXIS_M)
    assert np.isnan(got.bt0) and "no Bt0 or EFIT axis" in got.caveat()
    # An axis outside a tokamak's range is not believed either.
    _axis_features(p, SHOT, 2.0, 7.0)
    assert mhd_1_1.choose(SHOT, None, paths=p, hottest_channel=lambda: 7).how == "hottest"


def test_the_1_1_page_places_the_channels_by_bt0_and_the_axis_when_it_has_them(tmp_path):
    p = tree.paths(tmp_path)
    crashes = _sawtooth_ece(p, SHOT)
    _axis_features(p, SHOT, _bt_for(24, 1.75), 1.75)
    [heatmap] = panels.build("mhd_n1.m1", SHOT, paths=p)
    assert "Bt0 = " in _caveat(heatmap) and "channel 25" in _caveat(heatmap)
    assert "nearest the EFIT magnetic axis" in _caveat(heatmap)
    assert "channels 38-40 are never the core" in _caveat(heatmap)
    ghz = mhd_1_1.ECE_GHZ
    assert heatmap.y.min() == pytest.approx(1.75 * ghz[24] / ghz[:40].max(), abs=0.015)
    [marker] = mhd_1_1.markers(SHOT, paths=p)
    assert len(marker["t_ms"]) == len(crashes)
    assert np.abs(np.array(marker["t_ms"]) - crashes).max() < 1.0


# --- the ELM strike-point chord choice ---------------------------------------


def _strikes(n=100, *, ip=1e6, r_out=1.5, r_in=1.2, z_out=-1.2):
    return tuple(np.full(n, float(v)) if np.isscalar(v) else np.asarray(v, float)
                 for v in (ip, r_out, r_in, z_out))


def test_a_lower_strike_point_picks_the_chords_outside_it_and_inside_the_inner_one():
    choice = elm.select_chords(*_strikes())
    assert choice.names == ("FS00", "FS01", "FS07", "FS08", "FS06")
    assert (choice.r_out, choice.r_in, choice.z_avg) == pytest.approx((1.5, 1.2, -1.2))


def test_an_upper_strike_point_picks_among_the_upper_chords():
    choice = elm.select_chords(*_strikes(z_out=1.2))
    assert choice.names == ("FS13UP", "FS12UP", "FS11UP", "FS00UP", "FS03UP", "FS04UP")


def test_an_inner_strike_point_on_the_centre_post_is_moved_off_it():
    choice = elm.select_chords(*_strikes(r_in=1.0))
    assert choice.r_in == elm.CENTRE_POST_CLEARANCE
    assert "FS00" in choice.names  # R 1.015 m: inside the inner strike point now


def test_a_strike_point_counts_only_while_ip_is_high_and_the_inner_one_was_found():
    ip = np.r_[np.full(50, 1e6), np.full(50, 1e6), np.full(50, 1e4)]
    r_out = np.r_[np.full(50, 1.5), np.full(50, 1.9), np.full(50, 2.5)]
    r_in = np.r_[np.full(50, 1.2), np.zeros(50), np.full(50, 1.2)]  # 0: not found
    z_out = np.full(150, -1.2)
    choice = elm.select_chords(ip, r_out, r_in, z_out)
    assert choice.r_out == pytest.approx(1.5)  # not 1.9 (no inner), not 2.5 (no Ip)


def test_a_double_null_averages_over_the_side_the_strike_point_is_mostly_on():
    z_out = np.r_[np.full(60, -0.6), np.full(40, 0.6)]
    r_out = np.r_[np.full(60, 1.5), np.full(40, 1.9)]
    choice = elm.select_chords(np.full(100, 1e6), r_out, np.full(100, 1.2), z_out)
    assert choice.z_avg == pytest.approx(-0.6) and choice.r_out == pytest.approx(1.5)
    assert all(elm.CHORDS[name][1] < 0 for name in choice.names)


def test_efit_zero_padded_strike_points_are_missing_not_positions():
    # EFIT writes 0.0 where it found no strike point: the first 60 samples have none.
    r_out = np.r_[np.zeros(60), np.full(40, 1.5)]
    z_out = np.r_[np.zeros(60), np.full(40, -1.2)]
    choice = elm.select_chords(*_strikes(r_out=r_out, z_out=z_out))
    assert (choice.r_out, choice.r_in, choice.z_avg) == pytest.approx((1.5, 1.2, -1.2))
    assert choice.names == ("FS00", "FS01", "FS07", "FS08", "FS06")
    # The inner one lost too, at other times: only samples with both count.
    r_in = np.r_[np.full(80, 1.2), np.zeros(20)]
    choice = elm.select_chords(*_strikes(r_out=r_out, r_in=r_in, z_out=z_out))
    assert (choice.r_out, choice.r_in) == pytest.approx((1.5, 1.2))
    # Zero throughout is no strike point at all, not one at R = 0, Z = 0.
    zeros = np.zeros(100)
    with pytest.raises(ValueError, match="no strike point"):
        elm.select_chords(np.full(100, 1e6), zeros, zeros, zeros)
    with pytest.raises(ValueError, match="no strike point"):
        elm.select_chords(*_strikes(r_in=zeros))


def test_no_strike_point_raises_and_no_chord_left_gives_no_names():
    nan = np.full(10, np.nan)
    with pytest.raises(ValueError, match="no strike point"):
        elm.select_chords(nan, nan, nan, nan)
    assert elm.select_chords(*_strikes(), available=("FS02",)).names == ()


# --- the jump-and-hold detector ----------------------------------------------


def _elm_trace(elms, *, dt=0.02, flash=None, seed=5):
    """D-alpha at 1/dt kHz with an ELM-shaped burst (3 units) at each of `elms`,
    and, with `flash=(t_ms, size)`, a larger burst of a disruption."""
    t = np.arange(0.0, 300.0, dt)
    y = 1.0 + 0.01 * np.random.default_rng(seed).standard_normal(len(t))
    bursts = [(e, 3.0) for e in elms] + ([flash] if flash else [])
    for e, size in bursts:
        after = t >= e
        y[after] += size * (1 - np.exp(-(t[after] - e) / 0.1)) * np.exp(-(t[after] - e) / 2.0)
    return t, y


@pytest.mark.parametrize("dt", [0.02, 0.1])
def test_jump_and_hold_finds_each_elm_and_merges_a_pair_inside_a_millisecond(dt):
    t, y = _elm_trace([50.0, 90.0, 120.0, 120.6, 170.0, 230.0], dt=dt)
    jump = elm.jump_and_hold(t, y)
    assert jump.dtype == np.float32 and np.nanmax(jump) >= 1.0
    found = elm.onsets(t, jump)
    np.testing.assert_allclose(found, [50, 90, 120, 170, 230], atol=2 * dt + 0.1)


def test_ip_keeps_a_disruptions_flash_from_being_an_onset_and_from_setting_the_scale():
    t, y = _elm_trace([50.0, 120.0], flash=(275.0, 60.0))
    ip = np.where(t < 270, 1.0e6, 1.0e6 * np.exp(-(t - 270) / 1.0))
    # The scale is a percentile of the rises, which one flash cannot move (the
    # global maximum it replaces lost every ELM here), so the ELMs stand either way.
    for jump in (elm.jump_and_hold(t, y), elm.jump_and_hold(t, y, ip)):
        found = elm.onsets(t, jump)
        np.testing.assert_allclose(found[:2], [50, 120], atol=0.3)
        assert abs(found[-1] - 275) < 0.5  # and an unmasked flash is an onset
    steady = elm.onsets(t, elm.jump_and_hold(t, y, ip), keep=elm.flat_top(t, ip))
    np.testing.assert_allclose(steady, [50, 120], atol=0.3)  # the mask drops it


def test_the_scale_is_the_flat_tops_and_an_onset_outside_it_is_dropped():
    t, y = _elm_trace([50.0, 120.0, 170.0], flash=(275.0, 60.0))
    ip = np.where(t < 270, 1.0e6, 1.0e6 * np.exp(-(t - 270) / 1.0))
    keep = elm.flat_top(t, ip)
    assert keep[(t > 20) & (t < 255)].all() and not keep[t > 272].any()
    jump = elm.jump_and_hold(t, y, ip)
    np.testing.assert_array_equal(jump, elm.jump_and_hold(t, y, keep=keep))
    # The flash (20 ELMs' height) lies outside the mask the scale is read in.
    assert np.nanmax(jump[~keep]) > 5 * np.nanmax(jump[keep])
    np.testing.assert_allclose(elm.onsets(t, jump, keep=keep), [50, 120, 170], atol=0.3)
    assert abs(elm.onsets(t, jump)[-1] - 275) < 0.5  # no mask: the flash is one
    # Without Ip, or with no sample of it, the whole record is the flat top.
    assert elm.flat_top(t, np.full(len(t), np.nan)).all()


def test_a_dalpha_with_no_rise_cannot_be_normalised():
    t = np.arange(0.0, 50.0, 0.02)
    with pytest.raises(NoDataError, match="no rise"):
        elm.jump_and_hold(t, np.ones(len(t)))


def test_onsets_skip_a_start_inside_the_first_stretch_and_merge_close_ones():
    x = np.arange(10.0)
    jump = np.array([0.5, 0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5])
    np.testing.assert_array_equal(elm.onsets(x, jump, 0.12, 1.0), [4, 6, 9])
    np.testing.assert_array_equal(elm.onsets(x, jump, 0.12, 3.0), [4, 9])
    assert len(elm.onsets(x, np.full(10, 0.5))) == 0  # flagged throughout: no start


# --- the strike-point D-alpha end to end -------------------------------------


def _strike_shot(p, shot, *, z_out, with_upper=False, radii=(1.5, 1.2)):
    """FS01-FS08 holding 1..8 (FS06 dead), strike points R `radii` m, Ip steady."""
    t = tree.times(0.0, 300.0, 10_000)
    fs = np.full((104, len(t)), np.nan)
    for i in range(8):
        fs[i] = i + 1.0
    fs[5] = 0.0
    tree.write(p.corpus_file(shot), {"filterscopes": (t, fs)})
    ts = tree.times(0.0, 300.0, 1000)
    strike = np.stack([np.full(len(ts), v) for v in (*radii, z_out)])
    groups = {"strike_points": (ts, strike), "ip": (ts, np.full((1, len(ts)), 1e6))}
    if with_upper:
        groups["filterscopes_up"] = (ts, np.stack([np.full(len(ts), i + 1.0) for i in range(8)]))
    tree.write(p.raw_cache / f"{shot}_processed.h5", groups)


def test_lower_strike_sum_is_fs01_fs07_fs08_without_the_dead_fs06(tmp_path):
    p = tree.paths(tmp_path)
    _strike_shot(p, SHOT, z_out=-1.2)
    d = elm.strike_dalpha(SHOT, paths=p)
    assert d.title == "D-alpha FS strike sum" and d.legend == "strike sum"
    assert np.allclose(d.y, 1 + 7 + 8)
    assert d.caveat.startswith("FS01+FS07+FS08 summed") and "lower divertor" in d.caveat
    assert "Eldon" in d.caveat
    assert elm.dalpha(SHOT, paths=p).title == d.title


def test_upper_strike_sum_reads_the_fetched_upper_chords(tmp_path):
    p = tree.paths(tmp_path)
    _strike_shot(p, SHOT, z_out=+1.2, with_upper=True)
    d = elm.strike_dalpha(SHOT, paths=p)
    assert np.allclose(d.y, 1 + 2 + 3 + 4 + 7 + 8)
    assert d.caveat.startswith("FS13UP+FS12UP+FS11UP+FS00UP+FS03UP+FS04UP summed")
    assert "upper divertor" in d.caveat


def test_the_strike_sum_row_is_followed_by_the_single_chord_the_frames_model_reads(
    tmp_path,
):
    p = tree.paths(tmp_path)
    _strike_shot(p, SHOT, z_out=-1.2)
    built = elm.dalpha_panels(SHOT, paths=p)
    titles = [b.title for b in built]
    assert titles[0] == "D-alpha FS strike sum"
    assert re.fullmatch(r"D-alpha FS\d\d", titles[1]), titles
    assert "D-alpha PCPHD03" not in titles  # none cached: not drawn
    # The frames role and the paper roster take the chord, never the sum.
    assert [frames_features.title_matches(t, "D-alpha FS") for t in titles] == [
        False, True
    ]


def test_zero_padded_strike_points_fall_back_to_the_spans_filterscope(tmp_path):
    p = tree.paths(tmp_path)
    _strike_shot(p, SHOT, z_out=0.0, radii=(0.0, 0.0))
    d = elm.dalpha(SHOT, paths=p)
    assert re.fullmatch(r"D-alpha FS\d\d", d.title)
    assert d.caveat.startswith("The strike points could not be used (")
    assert "no strike point" in d.caveat
    assert [b.title for b in elm.dalpha_panels(SHOT, paths=p)] == [d.title]


def test_without_strike_points_the_row_is_the_spans_filterscope_and_says_why(tmp_path):
    p = tree.paths(tmp_path)
    _elm_dalpha_shot(p, SHOT)
    d = elm.dalpha(SHOT, paths=p)
    assert d.title == "D-alpha FS01" and d.legend == "FS01"
    assert d.caveat.startswith("The strike points could not be used (")
    with pytest.raises(NoDataError):
        elm.strike_dalpha(SHOT, paths=p)


# --- the H-mode edge Te row ----------------------------------------------------


def _radiometer(p, shot):
    """48 channels at 10 kHz, channel i holding i + 1 keV throughout."""
    t = tree.times(0.0, 100.0, 10_000)
    te = np.stack([np.full(len(t), i + 1.0) for i in range(48)])
    tree.write(p.corpus_file(shot), {"ece": (t, te)})


def test_edge_te_uses_the_outermost_channels_inside_the_plasma_when_geometry_is_cached(
    tmp_path,
):
    p = tree.paths(tmp_path)
    _radiometer(p, SHOT)
    rho = 1.11 - 0.02 * np.arange(40)  # channel 0 is outside the plasma
    t_s = np.array([0.0, 0.1, 0.2])
    arrays = {
        "ece_psi": FeatureArray(t_s, np.repeat((rho**2)[:, None], 3, axis=1), {}),
        "qpsi": FeatureArray(t_s, np.stack([0.8 + 0.8 * np.linspace(0, 1, 33)] * 3).T, {}),
    }
    write_features(p.features_file(SHOT), SHOT, arrays, {})
    [row] = hmode.edge_te_panel(SHOT, paths=p)
    assert row.title == "edge Te (ECE)" and row.legend == ["ch 6", "ch 7", "ch 8", "ch 9"]
    assert np.allclose(row.y[:, 0], [7, 8, 9, 10])
    assert "from the ECE geometry" in _caveat(row) and "1 ms mean" in _caveat(row)


def test_edge_te_without_geometry_takes_the_first_four_channels_and_says_so(tmp_path):
    p = tree.paths(tmp_path)
    _radiometer(p, SHOT)
    [row] = hmode.edge_te_panel(SHOT, paths=p)
    assert row.legend == ["ch 0", "ch 1", "ch 2", "ch 3"]
    assert np.allclose(row.y[:, 0], [1, 2, 3, 4])
    assert "No ECE geometry" in _caveat(row) and "outside the plasma" in _caveat(row)
    assert "1 ms mean" in _caveat(row)


def test_edge_te_falls_back_to_the_fetched_edge_group_and_then_to_nothing(tmp_path):
    p = tree.paths(tmp_path)
    t = tree.times(0.0, 100.0, 10_000)
    edge = np.stack([np.full(len(t), 10.0 + i) for i in range(8)])
    tree.write(p.raw_cache / f"{SHOT}_processed.h5", {"ece_edge": (t, edge)})
    [row] = hmode.edge_te_panel(SHOT, paths=p)
    assert np.allclose(row.y[:, 0], [10, 11, 12, 13])
    assert "No ECE geometry" in _caveat(row)
    with pytest.raises(NoDataError):
        hmode.edge_te_panel(SHOT + 5, paths=p)
    assert "edge Te (ECE)" in hmode.TITLES
