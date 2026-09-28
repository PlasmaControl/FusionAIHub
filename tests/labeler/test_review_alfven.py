"""The Heidbrink AE rows, on a synthetic CO2 record with a known mode and burst."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.config import Paths
from labeler.events import raw
from labeler.events.review import alfven

RATE = 5e6 / 3  # the CO2 digitiser, about 1.6667 MHz


def _record(seconds=0.4):
    """Unit noise on four chords; a 120 kHz mode on all four over 150-250 ms,
    and a 180 kHz burst on V2 alone over 50-120 ms."""
    n = int(seconds * RATE)
    t_s = np.arange(n) / RATE
    chords = np.random.default_rng(0).normal(0.0, 1.0, (4, n))
    mode = (t_s > 0.15) & (t_s < 0.25)
    chords[:, mode] += 0.5 * np.sin(2 * np.pi * 120e3 * t_s[mode])
    burst = (t_s > 0.05) & (t_s < 0.12)
    chords[2, burst] += 0.5 * np.sin(2 * np.pi * 180e3 * t_s[burst])
    return t_s * 1000, chords.astype(np.float32)


def _db(grid, row, t_ms, khz):
    """Mean dB over the columns centred strictly inside `t_ms`, at one frequency."""
    centres = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    inside = (centres > t_ms[0]) & (centres < t_ms[1])
    level = row.values[round((khz - row.y0) / row.dy), inside].mean()
    return level * (row.z_hi - row.z_lo) / 255 + row.z_lo


@pytest.fixture(scope="module")
def made():
    return alfven.spectrogram_rows(*_record())


def test_the_grid_is_the_heidbrink_stft(made):
    grid, rows = made
    assert grid.dt_ms == pytest.approx(0.256)
    assert [row.name for row in rows] == ["R0xV1", "R0xV2", "R0xV3"]
    for row in rows:
        assert row.values.shape == (257, grid.n) and row.values.dtype == np.uint8
        assert row.y0 == 0 and row.dy == pytest.approx(500 / 512)
        assert row.band == (80.0, 250.0) and (row.z_lo, row.z_hi) == (-3.0, 27.0)


def test_a_mode_on_every_chord_shows_on_every_row(made):
    grid, rows = made
    for row in rows:
        assert _db(grid, row, (160, 240), 120) >= 12, row.name


def test_a_burst_on_one_chord_stays_out_of_the_other_rows(made):
    grid, rows = made
    by_name = {row.name: row for row in rows}
    for name in ("R0xV1", "R0xV3"):
        assert _db(grid, by_name[name], (60, 110), 180) <= 3, name
    cross = by_name["R0xV2"]
    assert _db(grid, cross, (160, 240), 120) - _db(grid, cross, (60, 110), 180) >= 10


def test_build_reads_co2_through_the_raw_tiers_and_says_where_it_came_from(tmp_path):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    t_ms, chords = _record(0.05)
    raw.write_group(raw.cache_path(170790, paths=paths), "co2", t_ms, chords)
    grid, rows, info = alfven.build("alfven_eigenmode", 170790, paths)
    assert info["source"]["tier"] == "cache"
    assert info["source"]["path"] == str(tmp_path / "raw" / "170790_processed.h5")
    assert info["params"] == alfven.PARAMS
    assert len(rows) == 3 and rows[0].values.shape[1] == grid.n
