"""The ELM and H-mode editors' panels, on synthetic corpus files."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.events import panels, spans
from labeler.events.panels import _shared
from labeler.events.panels import edge_localized_mode as elm
from labeler.events.review import panel_rows
from labeler.events.review.rows import ImageRow, TraceRow

from . import editor_tree as tree

SHOT = 201234


def _elm_shot(p, *, pcphd03="varies", fs01=True):
    """CO2 with one burst, FS01-08 (FS01 dark unless `fs01`) and PCPHD03.

    `pcphd03` is "varies" (a baseline with ELM-like bursts), "flat" (a
    constant) or None (not in the cache).
    """
    fast = tree.times(0.0, 200.0, 500_000)
    co2 = tree.noise(4, fast)
    co2[0, (fast > 100) & (fast < 102)] *= 30  # one broadband burst
    slow = tree.times(0.0, 200.0, 10_000)
    fs = np.full((104, len(slow)), np.nan)
    fs[:8] = 1.0 + tree.noise(8, slow, 0.01)
    if not fs01:
        fs[0] = np.nan
    tree.write(p.corpus_file(SHOT), {"co2": (fast, co2), "filterscopes": (slow, fs)})
    t = tree.times(0.0, 200.0, 50_000)
    if pcphd03 == "varies":
        y = 0.05 + tree.noise(1, t, 0.002)[0]
        for peak in np.arange(40.0, 161.0, 20.0):
            y += 0.2 * np.exp(-0.5 * ((t - peak) / 0.5) ** 2)
    elif pcphd03 == "flat":
        y = np.full(len(t), 0.004) + tree.noise(1, t, 0.001)[0]
    if pcphd03 is not None:
        tree.write(p.raw_cache / f"{SHOT}_processed.h5", {"pcphd03": (t, y)})


def test_elm_draws_co2_power_to_125_khz_pcphd03_and_the_spans_filterscope(
    tmp_path, monkeypatch
):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _elm_shot(p)
    power, pcphd03, fs = panels.build("edge_localized_mode", SHOT, paths=p)
    assert (power.title, power.kind, power.ylabel) == ("CO2 R0 power", "heatmap", "kHz")
    assert power.y[0] == 0 and 124 < power.y[-1] <= 125
    assert (power.zmin, power.zmax) == _shared.Z_DB
    assert np.all(np.diff(power.x) > 0) and 0 < power.x[0] < power.x[-1] < 200
    burst = (power.x > 100) & (power.x < 102)
    assert power.z[:, burst].mean() > power.z[:, ~burst].mean() + 10, "the burst"
    assert pcphd03.title == "D-alpha PCPHD03" and pcphd03.y.shape == (1, 10_000)
    assert fs.title == "D-alpha FS01, the ELM spans' channel"
    assert fs.legend == ["FS01"] and fs.y.shape == (1, 2000)


def test_the_filterscope_is_the_one_the_spans_read(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _elm_shot(p, fs01=False)
    _t, y = spans.read(SHOT, "filterscopes", p, range(8))
    assert spans.dalpha_channel(y, SHOT) == 1
    *_, fs = panels.build("edge_localized_mode", SHOT, paths=p)
    assert fs.title == "D-alpha FS02, the ELM spans' channel" and fs.legend == ["FS02"]
    window = panels.build("edge_localized_mode", SHOT, t_range=(0, 1), paths=p)
    assert window[-1].legend == ["FS02"], "chosen over the record, not the view"


def test_a_flat_pcphd03_is_left_out_and_the_title_says_so(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _elm_shot(p, pcphd03="flat")
    _power, fs = panels.build("edge_localized_mode", SHOT, paths=p)
    assert fs.title == "D-alpha FS01, the ELM spans' channel (PCPHD03 flat, left out)"
    t = tree.times(0.0, 200.0, 50_000)
    ramp = np.linspace(0.0, 1.0, len(t))[None]
    assert elm.flat(t, ramp * 0.01, None) and not elm.flat(t, ramp * 0.03, None)
    assert elm.flat(t, ramp * 0.03, (0, 100)) is False
    assert elm.flat(t, ramp * 0.03, (0, 40)), "over the window only"


def test_elm_falls_back_to_fs01_and_leaves_out_what_is_missing(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tried = tree.no_fetch(monkeypatch)
    _elm_shot(p, pcphd03=None)
    power, dalpha = panels.build("edge_localized_mode", SHOT, paths=p)
    assert power.title == "CO2 R0 power"
    assert dalpha.title == "D-alpha FS01, the ELM spans' channel (PCPHD03 not found)"
    assert tried == [(SHOT, ["PCPHD03"])] * 2, "one try and its retry"
    other = SHOT + 1
    slow = tree.times(0.0, 200.0, 10_000)
    tree.write(p.corpus_file(other), {"filterscopes": (slow, np.ones((104, 2000)))})
    [only] = panels.build("edge_localized_mode", other, paths=p)
    assert only.title.startswith("D-alpha FS01"), "no CO2 anywhere: no spectrogram"


def test_elm_rows_are_an_image_and_two_traces(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _elm_shot(p)
    grid, rows, info = panel_rows.build("edge_localized_mode", SHOT, p)
    assert [type(row) for row in rows] == [ImageRow, TraceRow, TraceRow]
    assert (rows[0].z_lo, rows[0].z_hi) == _shared.Z_DB
    assert grid.dt_ms == panel_rows.FINEST_DT_MS and info["params"]


def test_a_late_spike_does_not_set_the_traces_range(tmp_path, monkeypatch):
    """PCPHD03's ELMs at 0.25 on 0.05, a 0.4 ms spike of 5 at the end."""
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _elm_shot(p, pcphd03=None)
    t = tree.times(0.0, 200.0, 50_000)
    y = 0.05 + tree.noise(1, t, 0.002)[0]
    for peak in np.arange(40.0, 161.0, 20.0):
        y += 0.2 * np.exp(-0.5 * ((t - peak) / 0.5) ** 2)
    y[(t > 195) & (t < 195.4)] = 5.0  # the end of the discharge
    tree.write(p.raw_cache / f"{SHOT}_processed.h5", {"pcphd03": (t, y)})
    _power, pcphd03, _fs = panels.build("edge_localized_mode", SHOT, paths=p)
    assert pcphd03.title == "D-alpha PCPHD03, clipped to its plasma range"
    top = pcphd03.y.max()
    assert top < 1.0, "the spike is clipped"
    elms = pcphd03.y[0, (pcphd03.x > 30) & (pcphd03.x < 170)]
    assert elms.max() == pytest.approx(y[(t > 30) & (t < 170)].max()), "ELMs whole"
    assert elms.max() > 0.4 * top, "the ELMs fill the row the page autoscales"
    _grid, rows, _ = panel_rows.build("edge_localized_mode", SHOT, p)
    assert np.nanmax(rows[1].values) == pytest.approx(top)
    [[low], [high]] = _shared.robust_limits(t, y[None], None)
    assert low < 0.05 and 0.25 <= high < 1.0


def test_the_co2_floor_is_the_plasma_windows(tmp_path, monkeypatch):
    """CO2 40 dB quieter outside the 50-150 ms window: the floor is inside it."""
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    fast = tree.times(0.0, 300.0, 500_000)
    co2 = tree.noise(4, fast)
    co2[:, (fast < 50) | (fast > 150)] *= 0.01
    tree.write(p.corpus_file(SHOT), {"co2": (fast, co2)})
    tree.cohort(p, [tree.queue_row(SHOT, 0, window=(50, 150))])
    [power] = elm.co2_panel(SHOT, paths=p)
    plasma = (power.x > 55) & (power.x < 145)
    assert abs(np.median(power.z[:, plasma])) < 1.5, "the plasma sits on the floor"
    assert power.z[:, power.x > 160].mean() < -30


def _hmode_shot(p, *, betan=True):
    slow = tree.times(0.0, 300.0, 10_000)
    fs = np.full((104, len(slow)), np.nan)
    fs[:3] = 2.0 - (slow > 150)  # D-alpha falls at 150 ms on three channels
    pinj = np.zeros((8, len(slow)))
    pinj[:2, slow > 50] = 2.5e6  # two beams, 5 MW
    fast = tree.times(0.0, 300.0, 500_000)
    co2 = np.ones((4, len(fast))) * (1 + (fast > 150))
    groups = {"filterscopes": (slow, fs), "pinj": (slow, pinj), "co2": (fast, co2)}
    tree.write(p.corpus_file(SHOT), groups)
    if betan:
        grid = np.arange(0.0, 300.0, 20.0)
        tree.features(p, SHOT, grid, 1.0 + grid / 300)


def test_hmode_draws_dalpha_density_beams_and_betan(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _hmode_shot(p)
    dalpha, density, beams, betan = panels.build("high_confinement_mode", SHOT, paths=p)
    assert dalpha.legend == ["FS01", "FS02", "FS03"] and dalpha.y.shape[0] == 3
    assert np.allclose(np.diff(density.x), 1.0) and density.y.shape == (1, 300)
    assert np.allclose(density.y[0, [10, 200]], [1.0, 2.0])
    assert beams.ylabel == "MW" and np.isclose(beams.y.max(), 5.0)
    assert np.allclose(betan.x, np.arange(0.0, 300.0, 20.0))
    window = panels.build("high_confinement_mode", SHOT, t_range=(100, 200), paths=p)
    assert all(100 <= panel.x.min() and panel.x.max() <= 200 for panel in window)


def test_hmode_without_features_still_draws_the_rest(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _hmode_shot(p, betan=False)
    titles = [x.title for x in panels.build("high_confinement_mode", SHOT, paths=p)]
    assert titles == [
        "D-alpha filterscopes",
        "density, CO2 R0 (1 ms mean)",
        "NBI power",
    ]


def test_bin_mean_times_are_the_samples_and_gaps_are_nan():
    x = np.arange(0.0, 4.0, 0.5)
    y = np.array([[1, 3, np.nan, np.nan, 5, 7, 1, 1]], dtype=float)
    centres, means = _shared.bin_mean(x, y, 1.0)
    assert np.allclose(centres, [0.25, 1.25, 2.25, 3.25])
    assert np.isnan(means[0, 1]) and np.allclose(means[0, [0, 2, 3]], [2, 6, 1])
    times, means = _shared.bin_mean([0.0, 0.5, 3.0], [[1.0, 3.0, 5.0]], 1.0)
    assert np.allclose(times, [0.25, 3.0]) and np.allclose(means, [[2.0, 5.0]])
