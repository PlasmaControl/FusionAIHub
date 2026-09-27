"""The ELM and H-mode editors' panels, on synthetic corpus files."""

from __future__ import annotations

import numpy as np

from labeler.events import panels
from labeler.events.panels import _shared
from labeler.events.review import panel_rows
from labeler.events.review.rows import ImageRow, TraceRow

from . import editor_tree as tree

SHOT = 201234


def _elm_shot(p, *, pcphd03=True):
    fast = tree.times(0.0, 200.0, 500_000)
    co2 = tree.noise(4, fast)
    co2[0, (fast > 100) & (fast < 102)] *= 30  # one broadband burst
    slow = tree.times(0.0, 200.0, 10_000)
    fs = np.full((104, len(slow)), np.nan)
    fs[:8] = 1.0 + tree.noise(8, slow, 0.01)
    tree.write(p.corpus_file(SHOT), {"co2": (fast, co2), "filterscopes": (slow, fs)})
    if pcphd03:
        tree.write(
            p.raw_cache / f"{SHOT}_processed.h5",
            {"pcphd03": (tree.times(0.0, 200.0, 50_000), np.ones(10_000))},
        )


def test_elm_draws_co2_power_to_125_khz_and_pcphd03(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _elm_shot(p)
    power, dalpha = panels.build("edge_localized_mode", SHOT, paths=p)
    assert (power.title, power.kind, power.ylabel) == ("CO2 R0 power", "heatmap", "kHz")
    assert power.y[0] == 0 and 124 < power.y[-1] <= 125
    assert (power.zmin, power.zmax) == _shared.Z_DB
    assert np.all(np.diff(power.x) > 0) and 0 < power.x[0] < power.x[-1] < 200
    burst = (power.x > 100) & (power.x < 102)
    assert power.z[:, burst].mean() > power.z[:, ~burst].mean() + 10, "the burst"
    assert dalpha.title == "D-alpha PCPHD03" and dalpha.y.shape == (1, 10_000)


def test_elm_falls_back_to_fs01_and_leaves_out_what_is_missing(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tried = tree.no_fetch(monkeypatch)
    _elm_shot(p, pcphd03=False)
    power, dalpha = panels.build("edge_localized_mode", SHOT, paths=p)
    assert power.title == "CO2 R0 power"
    assert dalpha.title == "D-alpha FS01 (PCPHD03 not found)"
    assert tried == [(SHOT, ["PCPHD03"])] * 2, "one try and its retry"
    other = SHOT + 1
    slow = tree.times(0.0, 200.0, 10_000)
    tree.write(p.corpus_file(other), {"filterscopes": (slow, np.ones((104, 2000)))})
    [only] = panels.build("edge_localized_mode", other, paths=p)
    assert only.title.startswith("D-alpha FS01"), "no CO2 anywhere: no spectrogram"


def test_elm_rows_are_an_image_and_a_trace(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.no_fetch(monkeypatch)
    _elm_shot(p)
    grid, rows, info = panel_rows.build("edge_localized_mode", SHOT, p)
    assert [type(row) for row in rows] == [ImageRow, TraceRow]
    assert (rows[0].z_lo, rows[0].z_hi) == _shared.Z_DB
    assert grid.dt_ms == panel_rows.FINEST_DT_MS and info["params"]


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
