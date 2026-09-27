"""Pseudo-masks: TokEye's coherent lines inside the owner's AE frames, 80-250 kHz."""

from __future__ import annotations

import numpy as np
import pandas as pd

from labeler.ae.seg import pseudo
from labeler.ae.xpower.data import clean_path, tokeye_clean
from labeler.events.review import labels
from labeler.events.review.rows import Grid

from . import ae_tree


def test_page_bin_k_takes_tokeye_bins_2k_minus_2_and_2k_minus_1():
    clean = np.zeros((4, 512, 3), dtype=bool)
    clean[:2, 299, 0] = True  # two chords: lit
    clean[:1, 198, 1] = True  # one chord: not
    clean[:, 511, 2] = True
    rows = pseudo.tokeye_rows(clean)
    assert rows.shape == (257, 3)
    assert np.flatnonzero(rows[:, 0]).tolist() == [150]
    assert not rows[:, 1].any()
    assert np.flatnonzero(rows[:, 2]).tolist() == [256]


def test_columns_are_ored_into_the_grid_and_covered_only_inside_the_record():
    grid = Grid(0.0, 2.0, 4)
    lit = np.array([[True, False, False, False, True, False]])
    t = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    assert pseudo.pool_columns(lit, t, grid).tolist() == [[True, False, True, False]]
    assert pseudo.covered_columns(t, grid).tolist() == [True, True, True, False]


def _mask(tmp_path, spans=None):
    paths = ae_tree.build(tmp_path, {101: "train"}, tokeye_dt=0.256)
    label = labels.read_saved(paths.label_tables / "alfven_eigenmode")[101]
    if spans is not None:
        label = labels.normalise((0, 2000), spans)
    return pseudo.make(paths, 101, label), paths


def test_the_owners_ae_frames_keep_tokeyes_line_and_absent_frames_are_background(
    tmp_path,
):
    pm, _ = _mask(tmp_path)
    t = pm.t0_ms + (np.arange(pm.mask.shape[1]) + 0.5) * pm.dt_ms
    ae = (t > 310) & (t < 890)
    mhd = (t > 1210) & (t < 1490)
    assert (pm.mask[150, ae] == 1).all()
    assert (pm.mask[[148, 149, 151, 152]][:, ae] == pseudo.IGNORE).all()
    assert (pm.mask[153:, ae] == 0).all() and (pm.mask[82:148, ae] == 0).all()
    assert (pm.mask[100, mhd] == 0).all(), "the MHD harmonic is a hard negative"
    assert (pm.mask[:82] == pseudo.IGNORE).all(), "below 80 kHz is not scored"
    assert (pm.mask[:, t > 2010] == pseudo.IGNORE).all(), "TokEye stops at 2 s"
    assert (pm.mask[82:, (t > 950) & (t < 1150)] == 0).all()
    assert pm.present_unlit == 0


def test_present_frames_tokeye_left_dark_and_specks_are_ignored(tmp_path):
    pm, _ = _mask(tmp_path, [(300, 900, 1), (1000, 1100, 1), (1200, 1500, 2)])
    t = pm.t0_ms + (np.arange(pm.mask.shape[1]) + 0.5) * pm.dt_ms
    dark = (t > 1000) & (t < 1100)
    assert (pm.mask[:, dark] == pseudo.IGNORE).all()
    assert pm.present_unlit == dark.sum()
    assert (pm.mask[:, (t > 1200) & (t < 1500)] == pseudo.IGNORE).all(), "uncertain"
    grid = Grid(pm.t0_ms, pm.dt_ms, pm.mask.shape[1])
    t_ms, clean, ann = tokeye_clean(clean_path(tmp_path / "root" / "ae" / "masks", 101))
    clean = clean.copy()
    clean[:, 399, (t_ms > 500) & (t_ms < 505)] = True  # a 2-column speck at 195 kHz
    label = labels.normalise((0, 2000), [(300, 900, 1)])
    speck = pseudo.build(101, label, grid, 257, 0.0, 500 / 512, (t_ms, clean, ann))
    cols = np.flatnonzero((t > 501) & (t < 506))  # the three columns it lights
    assert (speck.mask[200, cols] == pseudo.IGNORE).all()
    assert not (speck.mask[200] == 1).any()


def test_the_command_writes_a_mask_per_saved_shot_and_an_index(tmp_path, monkeypatch):
    paths = ae_tree.build(tmp_path, {101: "train", 102: "valid"}, tokeye_dt=0.256)
    ae_tree.env(monkeypatch, paths)
    assert pseudo.main([]) == 0
    out = pseudo.pseudo_dir(paths)
    loaded = pseudo.PseudoMask.load(out / "101.npz")
    assert loaded.mask.dtype == np.uint8 and loaded.mask.shape[0] == 257
    index = pd.read_csv(out / "index.csv")
    assert index.shot.tolist() == [101, 102]
    assert index.regions.tolist() == [1, 1]
    assert (index.ae_px > 250).all()
    assert (out / "meta.json").is_file()
