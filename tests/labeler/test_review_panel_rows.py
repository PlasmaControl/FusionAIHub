"""Panels onto one grid: binning fine data, interpolating coarse data, scaling."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.events.review import panel_rows
from labeler.events.review.rows import Grid, pool
from labeler.events.verify import NoDataError, Panel


def _panels(monkeypatch, *built):
    monkeypatch.setattr(
        panel_rows.panels, "build", lambda event, shot, **kw: list(built)
    )


def test_fine_lines_are_min_max_binned_at_the_finest_grid(monkeypatch):
    x = np.array([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    _panels(monkeypatch, Panel("ne", x=x, y=np.array([[1.0, 5, 2, 4, 9, 3]])))
    grid, rows, info = panel_rows.build("detachment", 1, None)
    assert grid == Grid(-0.025, 0.05, 3)
    assert rows[0].values[0].tolist() == [[1, 2, 3]]
    assert rows[0].values[1].tolist() == [[5, 4, 9]]
    assert rows[0].legend == ["ch 0"]
    assert info == {"params": {"finest_dt_ms": 0.05, "percentiles": [1.0, 99.5]}}


def test_coarse_lines_are_interpolated_onto_the_grid():
    panel = Panel("ip", x=np.array([0.0, 0.1]), y=np.array([[0.0, 10.0]]))
    row = panel_rows._trace("p0", panel, np.array([0.0, 0.1]), Grid(-0.025, 0.05, 3))
    assert row.values.tolist() == [[[0, 5, 10]], [[0, 5, 10]]]


def test_indicator_steps_cover_full_bins_and_keep_isolated_valid_values():
    panel = Panel(
        "Afrac",
        x=np.array([25.0, 75, 125]),
        y=np.array([[0.2, np.nan, 0.8]]),
        metadata={
            "trace_style": "step",
            "bin_start_ms": [0.0, 50, 100],
            "bin_end_ms": [50.0, 100, 150],
        },
    )
    row = panel_rows._trace("p0", panel, panel.x, Grid(0, 10, 15))
    np.testing.assert_allclose(row.values[:, 0, :5], 0.2)
    assert np.isnan(row.values[:, 0, 5:10]).all()
    np.testing.assert_allclose(row.values[:, 0, 10:], 0.8)


def test_indicator_bins_finer_than_grid_keep_their_minimum_and_maximum():
    panel = Panel(
        "Afrac",
        x=np.array([2.0, 6, 10]),
        y=np.array([[np.nan, 0.8, 0.2]]),
        metadata={
            "trace_style": "step",
            "bin_start_ms": [0.0, 4, 8],
            "bin_end_ms": [4.0, 8, 12],
        },
    )
    row = panel_rows._trace("p0", panel, panel.x, Grid(0, 12, 1))
    np.testing.assert_allclose(row.values[:, 0, 0], [0.2, 0.8])


def test_a_heatmap_is_scaled_and_placed_and_empty_columns_are_zero(monkeypatch):
    heatmap = Panel("S", x=np.array([0.0, 1, 2]), y=np.array([100.0]), kind="heatmap",
                    z=np.array([[0.0, 0.5, 1.0]]), zmin=0, zmax=1, bands=[(80, 250)])
    line = Panel("L", x=np.array([0.0, 3.0]), y=np.array([[0.0, 1.0]]))
    _panels(monkeypatch, heatmap, line)
    grid, rows, _ = panel_rows.build("fishbone", 1, None)
    assert grid == Grid(-0.5, 1.0, 4)
    assert rows[0].values.tolist() == [[0, 128, 255, 0]]
    assert (rows[0].z_lo, rows[0].z_hi, rows[0].band) == (0.0, 1.0, (80, 250))
    assert [row.name for row in rows] == ["p0", "p1"]


def test_a_modes_heatmap_codes_each_cell_by_level_then_mode(monkeypatch):
    colours = {-1: "#00ffff", 1: "#ff0000", 2: "#00ff00"}
    heatmap = Panel(
        "n",
        x=np.array([0.0, 1, 2]),
        y=np.array([5.0, 10.0]),
        kind="heatmap",
        z=np.array([[0.0, 0.5, 1.0], [1.0, np.nan, 0.5]]),
        zmin=0,
        zmax=1,
        modes=np.array([[1, 2, -1], [2, 1, 1]]),
        mode_colours=colours,
    )
    _panels(monkeypatch, heatmap)
    _grid, rows, _ = panel_rows.build("neoclassical_tearing_mode", 1, None)
    # 85 levels of 3 modes: level 42 is half-way, 84 the top; nan is level 0.
    assert rows[0].values.tolist() == [
        [1, 42 * 3 + 2, 84 * 3],
        [84 * 3 + 2, 1, 42 * 3 + 1],
    ]
    assert rows[0].meta()["modes"] == {
        "n": [-1, 1, 2],
        "levels": 85,
        "colours": ["#00ffff", "#ff0000", "#00ff00"],
    }
    # A block's max is its brightest cell, in that cell's own mode.
    assert pool(rows[0].values, 8, "image").tolist() == [[84 * 3], [84 * 3 + 2]]


def test_heatmap_edges_become_centres():
    panel = Panel("S", x=np.array([0.0, 1, 2]), y=np.array([0.0, 0.5, 1.0]),
                  kind="heatmap", z=np.zeros((2, 2)))
    row = panel_rows._image("p0", panel, np.array([0.5, 1.5]), Grid(0.0, 1.0, 2))
    assert (row.y0, row.dy) == (0.25, 0.5)


def test_the_colour_limits_are_the_1st_and_99_5th_percentiles():
    assert panel_rows._limits(np.arange(1000.0)) == pytest.approx((9.99, 994.005))
    assert panel_rows._limits(np.full(4, 2.0)) == (2.0, 3.0)


def test_a_shot_with_no_panels_has_no_data(monkeypatch):
    _panels(monkeypatch)
    with pytest.raises(NoDataError, match="no panels for shot 7"):
        panel_rows.build("detachment", 7, None)
