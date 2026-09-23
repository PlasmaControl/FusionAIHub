"""The row store: pooling, the pyramid, and reading a window."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from labeler.events.review import rows
from labeler.events.review.rows import Grid, ImageRow, TraceRow


def _store(path, n=10_000):
    image = ImageRow(
        "R0",
        "R0",
        np.tile(np.arange(n) % 256, (3, 1)).astype(np.uint8),
        y0=0.0,
        dy=1.0,
        y_units="kHz",
        z_lo=-3.0,
        z_hi=27.0,
        z_units="dB",
        band=(80.0, 250.0),
    )
    trace = TraceRow(
        "p1",
        "Density",
        np.stack([np.zeros((2, n)), np.ones((2, n))]).astype(np.float32),
        y_units="1e19 m^-3",
        legend=["a", "b"],
        hlines=[0.5],
    )
    rows.write(path, Grid(0.0, 1.0, n), [image, trace], event="alfven_eigenmode",
               shot=170790, params={"hop": 128})
    return path


@pytest.fixture
def store(tmp_path):
    return _store(tmp_path / "170790.h5")


def test_images_pool_by_max_and_the_last_block_pads_with_its_own_last_column():
    assert rows.pool(np.array([[1, 5, 2, 7, 3]]), 2, "image").tolist() == [[5, 7, 3]]


def test_traces_pool_min_and_max_and_skip_nan_unless_a_block_is_all_nan():
    values = np.array([[[1, np.nan, np.nan, 0]], [[2, np.nan, 5, np.nan]]])
    pooled = rows.pool(values, 2, "trace")
    assert pooled[0].tolist() == [[1, 0]]
    assert pooled[1].tolist() == [[2, 5]]
    assert np.isnan(rows.pool(np.full((2, 1, 2), np.nan), 2, "trace")).all()


def test_the_store_holds_a_pyramid_per_row(store):
    with h5py.File(store, "r") as f:
        shapes = [f["rows/R0"][str(level)].shape for level in rows.LEVELS]
        assert shapes == [(3, 10000), (3, 1250), (3, 157)]
        assert f["rows/p1/64"].shape == (2, 2, 157)
        assert f["rows/R0/1"].chunks == (3, 512)
        assert f["rows/R0/1"].compression == "gzip"
        assert f.attrs["event"] == "alfven_eigenmode" and f.attrs["shot"] == 170790
        assert json.loads(f.attrs["params"]) == {"hop": 128}


def test_meta_describes_the_grid_and_the_rows(store):
    meta = rows.meta(store)
    assert meta["grid"] == {"t0": 0.0, "dt": 1.0, "n": 10000}
    assert meta["t_range"] == [0.0, 10000.0]
    assert meta["rows"] == [
        {"name": "R0", "kind": "image", "title": "R0", "n_y": 3, "y0": 0.0, "dy": 1.0,
         "y_units": "kHz", "z_lo": -3.0, "z_hi": 27.0, "z_units": "dB",
         "band": [80.0, 250.0]},
        {"name": "p1", "kind": "trace", "title": "Density", "n_channels": 2,
         "y_units": "1e19 m^-3", "legend": ["a", "b"], "hlines": [0.5]},
    ]


def test_a_whole_record_read_takes_the_coarsest_level_that_fills_the_columns(store):
    data, grid = rows.read_window(store, 0, 10_000, 1000)
    assert grid == {"t0": 0.0, "t1": 10000.0, "n": 625}
    assert len(data) == 3 * 625 + 2 * 2 * 625 * 4


def test_a_zoomed_read_comes_from_the_finest_level(store):
    data, grid = rows.read_window(store, 100, 150, 1000)
    assert grid == {"t0": 100.0, "t1": 150.0, "n": 50}
    image = np.frombuffer(data[: 3 * 50], dtype=np.uint8).reshape(3, 50)
    assert image[0].tolist() == list(range(100, 150))


def test_a_read_past_the_start_is_clamped_to_the_record(store):
    _, grid = rows.read_window(store, -500, 50, 1000)
    assert grid == {"t0": 0.0, "t1": 50.0, "n": 50}


def test_a_read_outside_the_record_is_refused(store):
    with pytest.raises(ValueError, match="outside the record"):
        rows.read_window(store, 20_000, 30_000, 1000)


def test_a_read_with_fewer_columns_pools_on_the_fly(store):
    data, grid = rows.read_window(store, 0, 1000, 300)
    assert grid == {"t0": 0.0, "t1": 1000.0, "n": 250}
    image = np.frombuffer(data[: 3 * 250], dtype=np.uint8).reshape(3, 250)
    assert image[0, :3].tolist() == [3, 7, 11]


def test_a_read_across_a_chunk_boundary_is_seamless(store):
    data, _ = rows.read_window(store, 500, 530, 1000)
    image = np.frombuffer(data[: 3 * 30], dtype=np.uint8).reshape(3, 30)
    assert image[0].tolist() == [v % 256 for v in range(500, 530)]
    trace = np.frombuffer(data[3 * 30 :], dtype="<f4").reshape(2, 2, 30)
    assert (trace[0] == 0).all() and (trace[1] == 1).all()


def test_a_row_that_does_not_fit_the_grid_is_refused_and_nothing_is_left(tmp_path):
    image = ImageRow("R0", "R0", np.zeros((2, 5), np.uint8), y0=0, dy=1,
                     y_units="kHz", z_lo=0, z_hi=1, z_units="")
    with pytest.raises(ValueError, match="5 columns, the grid 6"):
        rows.write(tmp_path / "s.h5", Grid(0.0, 1.0, 6), [image])
    assert list(tmp_path.iterdir()) == []
