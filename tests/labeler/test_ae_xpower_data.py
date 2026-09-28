"""The AE model's frames: inputs from the review rows, targets, MHD flags, split."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.ae.xpower import data
from labeler.events.review import rows
from labeler.events.review.labels import normalise


def test_a_window_holds_the_whole_frames_inside_it():
    assert data.window_frames((0, 2000)) == (0, 200)
    assert data.window_frames((5, 2003)) == (1, 199)
    assert data.window_frames((-15, 20)) == (-1, 3)
    assert data.window_frames((3, 9)) == (1, 0)


def test_the_band_is_80_to_250_khz_of_the_page_rows():
    dy = 500 / 512  # kHz between the rows' 257 bins
    band = data.band_slice(0.0, dy, 257)
    assert (band.start, band.stop) == (82, 257)
    assert band.start * dy >= 80 > (band.start - 1) * dy
    with pytest.raises(ValueError, match="no bin"):
        data.band_slice(300.0, dy, 4)


def test_sub_frames_average_the_columns_centred_in_them():
    grid = rows.Grid(t0_ms=-1.0, dt_ms=0.5, n=100)  # columns centred -0.75 .. 48.75
    values = np.tile(np.arange(100, dtype=np.uint8), (2, 3, 1))
    x, observed = data.frame_inputs(values, grid, first=0, n=6)
    assert x.shape == (2, 3, 30)
    # Sub-frame 0, 0-2 ms, holds the columns centred 0.25, 0.75, 1.25, 1.75: 2..5.
    assert x[0, 0, 0] == pytest.approx(np.mean([2, 3, 4, 5]) / 255)
    assert x[1, 2, 23] == pytest.approx(np.mean([94, 95, 96, 97]) / 255)
    # The record ends at 49 ms: frame 4 is observed, frame 5 (50-60 ms) is not.
    assert observed.tolist() == [True, True, True, True, True, False]
    assert not x[..., 25:].any()


def test_targets_are_the_owners_frame_states():
    label = normalise((0, 100), [(20, 45, 1), (70, 80, 2)])
    states = data.targets(label, -2, 14)
    assert states.tolist() == [-1, -1, 0, 0, 1, 1, 1, 0, 0, 2, 0, 0, -1, -1]


def _clean_file(path, t_ms, clean, ann):
    np.savez(
        path,
        t_ms=t_ms,
        mask_clean=np.packbits(clean, axis=-1),
        ann=ann.astype(np.uint8),
    )


def test_mhd_frames_are_where_two_chords_carry_a_low_line_half_the_frame(tmp_path):
    t = -0.768 + 0.256 * np.arange(200)  # -0.77 .. 50.2 ms: frames 0-4 covered
    clean = np.zeros((4, 512, len(t)), dtype=bool)
    frame = np.floor(t / 10).astype(int)
    clean[:2, 30, frame == 1] = True  # two chords, a 15 kHz line: frame 1 is MHD
    clean[:1, 30, frame == 2] = True  # one chord only: not
    clean[:2, 200, frame == 3] = True  # two chords, ~98 kHz: AE band, not MHD
    both = frame == 4
    clean[:2, 40, np.flatnonzero(both)[:10]] = True  # two chords, a third of frame 4
    path = tmp_path / "170700_valid_clean.npz"
    _clean_file(path, t, clean, np.zeros(len(t)))
    assert data.clean_path(tmp_path, 170700) == path
    assert data.clean_path(tmp_path, 170701) is None
    mhd = data.mhd_frames(path, 0, 6)
    assert mhd.tolist() == [False, True, False, False, False, False]
    frames = data.tokeye_frames(path, 0, 6)
    assert frames["ae"][3] == 1.0 and frames["ae"][1] == 0.0
    assert frames["covered"].tolist() == [True] * 5 + [False]
    assert not data.mhd_frames(None, 0, 6).any()


def test_a_persistent_line_is_notched_before_it_can_flag_mhd(tmp_path):
    t = -0.768 + 0.256 * np.arange(400)
    clean = np.zeros((4, 512, len(t)), dtype=bool)
    clean[:, 50, :] = True  # lit the whole record: an instrument line, notched
    path = tmp_path / "170701_train_clean.npz"
    _clean_file(path, t, clean, np.zeros(len(t)))
    assert not data.mhd_frames(path, 0, 10).any()


def test_the_seldnet_split_is_read_from_the_mask_names(tmp_path):
    for name in (
        "170659_valid_clean.npz",
        "176060_train_clean.npz",
        "176060_train_probs.npz",
    ):
        (tmp_path / name).write_bytes(b"")
    assert data.seldnet_split(tmp_path) == {170659: "valid", 176060: "train"}


def test_the_split_holds_out_seldnets_validation_block_and_draws_val_by_seed():
    seldnet = {s: ("valid" if s < 1100 else "train") for s in range(1000, 1300)}
    reviewed = list(range(1000, 1290))
    split = data.make_split(reviewed, seldnet)
    assert split == data.make_split(reversed(reviewed), seldnet)
    assert {s for s, v in split.items() if v == "test"} == set(range(1000, 1100))
    assert sum(v == "val" for v in split.values()) == data.N_VAL
    assert data.make_split(reviewed, seldnet, seed=1) != split
    with pytest.raises(ValueError, match="no SELDNet split"):
        data.make_split([999], seldnet)


def _store(path, n=4000):
    """A review store like the AE page's: three cross-power rows, 257 bins."""
    grid = rows.Grid(t0_ms=-100.0, dt_ms=0.256, n=n)
    rng = np.random.default_rng(0)
    built = [
        rows.ImageRow(
            name,
            name,
            rng.integers(0, 50, (257, n), dtype=np.uint8),
            y0=0.0,
            dy=500 / 512,
            y_units="kHz",
            z_lo=-3.0,
            z_hi=27.0,
            z_units="dB",
            band=(80.0, 250.0),
        )
        for name in data.CROSS_ROWS
    ]
    rows.write(path, grid, built, event="alfven_eigenmode", shot=1)
    return grid


def test_a_reviewed_shot_loads_as_frames_with_context(tmp_path):
    store = tmp_path / "1.h5"
    _store(store)
    loaded = data.store_rows(store)
    grid, values, y0, _ = loaded
    assert values.shape == (3, 257, 4000) and grid.dt_ms == 0.256 and y0 == 0.0
    label = normalise((0, 500), [(100, 250, 1)])
    shot = data.load_shot(1, label, loaded, masks_dir=tmp_path, context=5)
    assert shot.first == -5 and shot.n == 60
    assert shot.x.shape == (3, 175, 300) and shot.x.dtype == np.float16
    assert shot.states[:5].tolist() == [-1] * 5 and shot.states[15] == 1
    assert shot.observed.all() and not shot.mhd.any()
    coarse = data.store_rows(store, level=8)[0]
    assert coarse.dt_ms == pytest.approx(0.256 * 8) and coarse.n == 500


def test_the_raw_route_gives_the_page_rows_and_a_tone_lands_in_the_band():
    fs, seconds = 500_000, 0.2
    t = np.arange(int(fs * seconds)) / fs
    rng = np.random.default_rng(1)
    tone = np.sin(2 * np.pi * 150e3 * t) * (t > 0.15)
    chords = rng.normal(0, 1, (4, t.size)) + 3 * tone
    grid, values, y0, dy = data.raw_rows(t * 1000, chords.astype(np.float32))
    assert values.shape[:2] == (3, 257) and grid.dt_ms == pytest.approx(0.256)
    band = data.band_slice(y0, dy, values.shape[1])
    x, _ = data.frame_inputs(values[:, band], grid, first=0, n=20)
    k = round((150 - band.start * dy) / dy)
    before, after = x[:, k, 20:70].mean(), x[:, k, 80:98].mean()
    assert after > before + 0.2
