"""Panels onto one grid: binning fine data, interpolating coarse data, scaling."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.events.review import panel_rows
from labeler.events.review.rows import Grid
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
