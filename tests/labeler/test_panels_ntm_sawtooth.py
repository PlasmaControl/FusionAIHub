"""The tearing-mode and sawtooth editors' panels, on synthetic corpus files."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from labeler.events import panels
from labeler.events.panels import _shared
from labeler.events.panels import neoclassical_tearing_mode as ntm
from labeler.events.panels import sawtooth_oscillation as saw
from labeler.events.review import panel_rows
from labeler.events.verify import NoDataError

from . import editor_tree as tree

SHOT = 201235


def _mirnov(p, n: int, *, khz: float = 10.0):
    """A mode travelling counter-clockwise from above, cos(wt + n phi), on the
    MPI66M probes from 50 to 150 ms, in noise."""
    t = tree.times(0.0, 200.0, 500_000)
    y = tree.noise(29, t)
    on = (t > 50) & (t < 150)
    for row, phi in ntm.PROBES.items():
        y[row, on] += np.cos(2 * np.pi * khz * t[on] + n * np.deg2rad(phi))
    tree.write(p.corpus_file(SHOT), {"mirnov": (t, y)})


@pytest.mark.parametrize("n", [1, -4, 5])
def test_every_cell_of_the_line_takes_its_mode_number(tmp_path, n):
    p = tree.paths(tmp_path)
    _mirnov(p, n)
    power, modes = panels.build("neoclassical_tearing_mode", SHOT, paths=p)
    assert power.title == "MPI66M322D power" and power.y[-1] <= 30
    line = np.argmin(np.abs(power.y - 10.0))
    inside = (power.x > 60) & (power.x < 140)
    for panel in (power, modes):
        assert panel.z[line, inside].mean() > panel.z[line, ~inside].mean() + 20
    assert (modes.ylabel, modes.zmin, modes.zmax) == ("kHz", *ntm.Z_DB)
    assert (modes.modes[line, inside] == n).all()
    assert set(np.unique(modes.modes)) <= set(ntm.N_COLOURS) == set(ntm.N_VALUES)


def test_ntm_rows_and_beta_n(tmp_path):
    p = tree.paths(tmp_path)
    _mirnov(p, 1)
    tree.features(p, SHOT, np.arange(0.0, 200.0, 20.0), np.linspace(1, 2, 10))
    titles = [x.title for x in panels.build("neoclassical_tearing_mode", SHOT, paths=p)]
    assert titles[-1] == "beta_N" and len(titles) == 3
    _grid, rows, _info = panel_rows.build("neoclassical_tearing_mode", SHOT, p)
    assert [row.y_units for row in rows] == ["kHz", "kHz", "β_N"]
    assert "modes" not in rows[0].meta()
    assert rows[1].meta()["modes"] == {
        "n": list(range(-4, 6)),
        "levels": 25,
        "colours": [ntm.N_COLOURS[n] for n in range(-4, 6)],
    }


def _plasma_mirnov(p, *, end_ms: float = 250.0, quiet_db: float = 40.0):
    """A plasma from 50 to 150 ms in a record to `end_ms`, the rest `quiet_db` down.

    In the plasma, noise and a steady 10 kHz line from 70 to 130 ms: 60 % of the
    plasma, so a floor over the plasma stays under it.
    """
    t = tree.times(0.0, end_ms, 500_000)
    y = tree.noise(max(ntm.PROBES) + 1, t)
    line = (t >= 70) & (t < 130)
    for row in ntm.PROBES:
        y[row, line] += 0.4 * np.cos(2 * np.pi * 10.0 * t[line])
    y[:, (t < 50) | (t >= 150)] *= 10 ** (-quiet_db / 20)
    tree.write(p.corpus_file(SHOT), {"mirnov": (t, y)})


def _window(p, source: str) -> None:
    """The shot's 50-150 ms plasma window in the cohort or the population."""
    rows = [tree.queue_row(SHOT + 1, 0)]
    if source == "cohort":
        rows.append(tree.queue_row(SHOT, 1, window=(50, 150)))
    tree.cohort(p, rows)
    if source == "population":
        p.catalog.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"shot": [SHOT], "window_start_ms": [50], "window_end_ms": [150]}
        ).to_csv(p.catalog / "population.csv", index=False)


def _assert_plasma_on_scale(power) -> None:
    """The plasma's noise floor near 0 dB, the line well above it, nothing pinned."""
    plasma = (power.x > 55) & (power.x < 145)
    on = (power.x > 75) & (power.x < 125)
    noise = np.abs(power.y - 10.0) > 1.0
    line = np.argmin(np.abs(power.y - 10.0))
    z = power.z[:, plasma]
    assert np.mean(z > ntm.Z_DB[1]) < 0.01, "the plasma is not saturated"
    assert abs(np.quantile(z[noise], ntm.FLOOR_QUANTILE)) < 1.5
    assert 0 < np.median(z[noise]) < 8
    assert power.z[line, on].mean() > np.median(z[noise]) + 10
    assert np.mean(power.z[line, on] > ntm.Z_DB[1]) < 0.5


@pytest.mark.parametrize("source", ["cohort", "population"])
def test_the_spectrogram_floor_is_the_plasma_windows(tmp_path, source):
    p = tree.paths(tmp_path)
    _plasma_mirnov(p)
    _window(p, source)
    power = panels.build("neoclassical_tearing_mode", SHOT, paths=p)[0]
    assert power.x[0] < 10 and power.x[-1] > 240, "the whole record is drawn"
    _assert_plasma_on_scale(power)
    quiet = power.x > 160
    assert power.z[:, quiet].mean() < -30, "off the plasma is 40 dB below it"


def test_without_a_window_the_floor_is_the_louder_half_of_the_record(tmp_path):
    p = tree.paths(tmp_path)
    _plasma_mirnov(p, end_ms=200.0)
    _window(p, "none")
    _assert_plasma_on_scale(panels.build("neoclassical_tearing_mode", SHOT, paths=p)[0])


def _crashes(t, period_ms, *, start=0.0, stop=np.inf, amp=0.05):
    """A sawtooth: a slow rise, then a drop in one sample, every `period_ms`."""
    on = (t >= start) & (t < stop)
    return np.where(on, amp * ((t - start) % period_ms) / period_ms, 0.0)


def _sawtooth(p, *, ece=True, sxr=True, te=False, moving=(3, 5, 20, 25), early=()):
    """ECE with a 3-sample spike at 40 ms, and an SXR fan whose bright core
    chords sit still while the dim `moving` ones crash every 50 ms; the `early`
    ones crash every 20 ms, only before 70 ms. `te`: Thomson's 44 core chords
    every 10 ms, hottest between chords 41 and 42, with failed fits (0 and
    below) on 40-42."""
    groups = {}
    if te:
        t = tree.times(0.0, 100.0, 100)
        chords = np.arange(44)[:, None]
        y = np.repeat(3000.0 * np.exp(-(((chords - 41.5) / 2.5) ** 2)), len(t), axis=1)
        y[40, 3], y[41, 5], y[42, 7] = 0.0, -12.0, 0.0
        groups["ts_core_temp"] = (t, y)
    if ece:
        t = tree.times(0.0, 100.0, 500_000)
        y = 1.0 + tree.noise(48, t, 0.01)
        y[:, 20_000:20_003] += 20.0
        groups["ece"] = (t, y)
    if sxr:
        t = tree.times(0.0, 400.0, 10_000)
        y = np.full((320, len(t)), np.nan)  # the SX90RM1F fan is dark
        first = dict(saw.SXR_ARRAYS)["SX90RP1F"]
        chords = np.arange(32)
        fan = np.exp(-(((chords - 10.3) / 3) ** 2))[:, None] + tree.noise(32, t, 1e-4)
        for c in moving:
            fan[c] += _crashes(t, 50.0)
        for c in early:
            fan[c] += _crashes(t, 20.0, stop=70.0, amp=0.2)
        y[first : first + 32] = fan
        groups["sxr"] = (t, y)
    tree.write(p.corpus_file(SHOT), groups)
    tree.cohort(p, [tree.queue_row(SHOT + 1, 0)])  # no window: the whole record


def test_sawteeth_draw_ece_and_the_sxr_chords_that_crash(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _sawtooth(p)
    built = panels.build("sawtooth_oscillation", SHOT, paths=p)
    assert [x.title for x in built][:4] == [
        f"ECE Te, ch {a}-{a + 3} (0.05 ms median)" for a in (20, 24, 28, 32)
    ]
    ece = built[0]
    assert ece.y.shape == (4, 50_000), "every 500 kHz sample, at its own time"
    assert np.nanmax(ece.y) < 1.1, "the spike at 40 ms is gone"
    sxr = built[4]
    assert sxr.title == "SXR SX90RP1F, the 4 chords with the most crash-like drops"
    assert sxr.legend == [f"SX90RP1F{c}" for c in ("04", "06", "21", "26")]


def test_the_te_row_is_thomsons_hottest_core_chords_in_kev(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _sawtooth(p, ece=False, te=True)
    te, sxr = panels.build("sawtooth_oscillation", SHOT, paths=p)
    assert te.title == "Te, Thomson core: the 4 hottest chords"
    assert te.ylabel == "keV"
    assert te.legend == ["chord 40", "chord 41", "chord 42", "chord 43"]
    assert te.y[1, 0] == pytest.approx(3.0 * np.exp(-0.04)) and te.y.shape == (4, 10)
    assert np.isnan(te.y[:3]).sum() == 3, "a failed fit is a gap, not a 0 keV dip"
    assert np.isnan(te.y[0, 3]) and np.isnan(te.y[1, 5]) and np.isnan(te.y[2, 7])
    assert sxr.title.startswith("SXR")


def test_the_te_chords_are_the_windows_hottest_and_clipped_only_above(
    tmp_path, monkeypatch
):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    t = tree.times(0.0, 100.0, 100)  # 0, 10, ... 90 ms
    chords = np.arange(44)[:, None]
    centre = np.where(t >= 60, 38.5, 20.0)  # the core is at 38-39 in the window
    y = 3000.0 * np.exp(-(((chords - centre) / 2.5) ** 2)) * (1 + 0.1 * np.sin(t))
    y[10] = 0.0
    y[10, 7] = 9000.0  # the hottest, on 1 of the window's 4 samples
    y[38, 1] = 50_000.0  # a bad fit, before the window
    tree.write(p.corpus_file(SHOT), {"ts_core_temp": (t, y)})
    tree.cohort(p, [tree.queue_row(SHOT, 0, window=(60, 90))])
    [te] = panels.build("sawtooth_oscillation", SHOT, paths=p)
    # The window's, not the record's (20's), a fixed 40-43, or a sparse chord's.
    assert te.legend == [f"chord {c}" for c in (37, 38, 39, 40)]
    assert te.title.endswith(saw.TE_CLIPPED)
    assert te.y[1, 1] == np.nanmax(te.y[1]) < 3.3, "the bad fit sits on the edge"
    assert te.y[1, 0] < 0.01, "a cooler Te is not raised to the window's range"
    [view] = panels.build("sawtooth_oscillation", SHOT, paths=p, t_range=(0, 45))
    assert view.legend == te.legend and view.x.max() <= 45, "not the view's chords"
    assert view.y[1, 1] == te.y[1, 1], "clipped as over the record"
    dark = SHOT + 2
    tree.write(p.corpus_file(dark), {"ts_core_temp": (t, np.zeros((44, len(t))))})
    with pytest.raises(NoDataError, match="no Thomson core chord"):
        saw.te_panels(dark, paths=p)


def test_despike_drops_a_spike_keeps_a_step_and_leaves_gaps_nan():
    # Bins of 10, 10, 2 and 2 samples, exact in binary.
    x = np.r_[np.arange(0.0, 10.0, 0.5), np.arange(10.0, 20.0, 2.5)]
    y = np.where(x < 10, 3.0, 2.0)[None, :]  # a drop at bin 2's edge
    y[0, 4:6] = 40.0  # a 2-sample spike in bin 0
    y[0, -2:] = np.nan  # bin 3, only gaps
    out = _shared.despike(x, y, 5.0)
    assert out.shape == y.shape
    assert np.array_equal(out[0, :-2], np.where(x[:-2] < 10, 3.0, 2.0))
    assert np.isnan(out[0, -2:]).all()
    assert _shared.despike([], np.empty((2, 0)), 5.0).shape == (2, 0)


def test_the_ece_rows_keep_the_review_grid(tmp_path, monkeypatch):
    """On the corpus's float32 seconds (4.0-4.1 s, where one median a bin at its
    samples' mean time spaced them 0.05001068 ms) the grid stays
    `FINEST_DT_MS` and every ECE column inside the record holds a sample."""
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    t_s = (tree.times(4000.0, 4100.0, 500_000) / 1000).astype(np.float32)
    p.corpus_file(SHOT).parent.mkdir(parents=True)
    with h5py.File(p.corpus_file(SHOT), "w") as f:
        f["ece/xdata"] = t_s
        f["ece/ydata"] = np.ones((36, len(t_s)), dtype=np.float32)
    tree.cohort(p, [tree.queue_row(SHOT + 1, 0)])
    grid, rows, _ = panel_rows.build("sawtooth_oscillation", SHOT, p)
    assert grid.dt_ms == panel_rows.FINEST_DT_MS
    centres = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    inside = (centres > 1000 * t_s[0]) & (centres < 1000 * t_s[-1])
    for row in rows:
        assert np.isfinite(row.values[..., inside]).all(), row.title


def test_the_sxr_chords_are_chosen_over_the_ip_flat_top(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _sawtooth(p, ece=False, moving=(3, 5, 20), early=(28,))
    tree.cohort(p, [tree.queue_row(SHOT, 0, window=(0, 400))])
    t = tree.times(0.0, 400.0, 1_000)
    ip = 1e6 * np.clip(np.minimum(t / 100, (400 - t) / 100), 0, 1)
    tree.write(p.raw_cache / f"{SHOT}_processed.h5", {"ip": (t, ip[None])})
    [sxr] = panels.build("sawtooth_oscillation", SHOT, paths=p, t_range=(0, 60))
    # 29 crashes most, before the flat-top; 11 is the brightest of the still.
    assert sxr.legend == [f"SX90RP1F{c}" for c in ("04", "06", "11", "21")]
    assert sxr.x.max() <= 60, "the view is sliced; the choice is not the view's"
    [whole] = panels.build("sawtooth_oscillation", SHOT, paths=tree.paths(tmp_path))
    assert whole.legend == sxr.legend, "the same chords in any view"


def test_a_shot_without_ece_or_sxr_draws_the_other(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _sawtooth(p, ece=False)
    [only] = panels.build("sawtooth_oscillation", SHOT, paths=p)
    assert only.title.startswith("SXR")
    other = tree.paths(tmp_path / "other")
    _sawtooth(other, sxr=False)
    assert len(panels.build("sawtooth_oscillation", SHOT, paths=other)) == 4
    none = tree.paths(tmp_path / "none")
    tree.write(none.corpus_file(SHOT), {"ip": ([0.0, 1.0], [[1.0, 1.0]])})
    with pytest.raises(NoDataError, match="no panels"):
        panel_rows.build("sawtooth_oscillation", SHOT, none)
