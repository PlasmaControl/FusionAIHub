"""pseudo-v2: TokEye's whole-shot masks inside the owner's frames over 0-250 kHz,
the MHD-line rules, and how their values are picked on SegNet v2's training shots."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from labeler.ae import seg, xpower
from labeler.ae.seg import mhdlines, pseudo
from labeler.ae.seg.mhdlines import Rules, ShotLines
from labeler.ae.seg.pseudo import IGNORE
from labeler.ae.xpower.data import BAND_KHZ, FULL_BAND_KHZ, N_VAL, clean_path
from labeler.config import Paths, sha256_of
from labeler.events.catalog.states import ABSENT, PRESENT
from labeler.events.review import labels
from labeler.scoring.frames import OUTSIDE

from . import ae_tree, round3_tree

POOL = [101, 102, 103, 104]
TEST = [111]
SPLITS = {**dict.fromkeys(POOL, "train"), **dict.fromkeys(TEST, "valid")}
RULES = Rules(run_ms=20.0, drift_khz=1.0, steady_ms=200.0, bright_u8=200)


def test_the_seg_versions_name_their_masks_band_labels_and_directories(tmp_path):
    v1, v2 = seg.SEG_VERSIONS["v1"], seg.SEG_VERSIONS["v2"]
    assert (v1.pseudo, v1.band_khz, v1.labels, v1.ae_version) == (
        "pseudo-v1",
        BAND_KHZ,
        None,
        "v1",
    )
    assert (v2.pseudo, v2.band_khz, v2.labels, v2.ae_version) == (
        "pseudo-v2",
        FULL_BAND_KHZ,
        "v3",
        "v3",
    )
    assert not v1.whole_window and v2.whole_window
    paths = Paths(root=tmp_path)
    masks = tmp_path / "segmentation" / "alfven_eigenmode"
    assert seg.pseudo_dir(paths) == seg.pseudo_dir(paths, "v1") == masks / "pseudo-v1"
    assert seg.pseudo_dir(paths, "v2") == masks / "pseudo-v2"
    assert seg.model_dir(paths) == tmp_path / "models" / "ae_seg" / "v1"
    assert seg.model_dir(paths, "v2") == tmp_path / "models" / "ae_seg" / "v2"
    assert seg.poi_dir(paths) == tmp_path / "poi" / "alfven_eigenmode" / "ae_seg-v1"
    assert seg.poi_dir(paths, "v2").name == "ae_seg-v2"
    assert (mhdlines.EIGHT == pseudo.EIGHT).all()
    assert pseudo.N_VAL == N_VAL == 16
    assert mhdlines.PAIRS[:4] == ((5.0, 50.0), (5.0, 100.0), (5.0, 200.0), (3.0, 50.0))
    assert len(mhdlines.PAIRS) == 12
    assert mhdlines.RUN_MS == (20.0, 50.0, 100.0, 200.0, 500.0)
    assert mhdlines.RULE_BELOW_KHZ == BAND_KHZ[0] == 80.0
    assert mhdlines.BRIGHT_PERCENTILE == 50


def test_the_absent_run_rule_takes_whole_regions_that_run_into_absent_time():
    lit = np.zeros((10, 20), dtype=bool)
    lit[2, 0:10] = True  # present 0-4, absent 5-9
    lit[3, 10] = True  # 8-connected to it: a sixth absent column
    lit[6, 0:5] = True  # present only
    absent = np.zeros(20, dtype=bool)
    absent[5:] = True
    got = mhdlines.absent_run(lit, absent, 2.0, 12.0)  # six columns of 2 ms
    assert np.flatnonzero(got[2]).tolist() == list(range(10)), "present part too"
    assert got[3, 10] and not got[6].any() and got.sum() == 11
    assert not mhdlines.absent_run(lit, absent, 2.0, 12.5).any()


def _chirp(lit, row0, cols, every, thick=1):
    """A line `thick` bins wide rising one bin every `every` columns over `cols`."""
    for c in cols:
        r = row0 + (c - cols[0]) // every
        lit[r : r + thick, c] = True


def test_the_steady_rule_takes_a_steady_low_line_not_a_chirp_or_a_high_line():
    lit = np.zeros((100, 120), dtype=bool)
    lit[20, 0:60] = True  # 20 kHz for 120 ms
    _chirp(lit, 30, range(60), 3)  # 30 to 49 kHz, 1 kHz every 6 ms
    lit[70, 0:60] = True  # 70 kHz: above the guard
    got = mhdlines.steady(lit, 0.0, 1.0, 2.0, 1.0, 100.0)
    assert np.array_equal(got, lit & (np.arange(100) == 20)[:, None])
    free = mhdlines.steady(lit, 0.0, 1.0, 2.0, 1.0, 100.0, guard_khz=None)
    assert np.flatnonzero(free.any(axis=1)).tolist() == [20, 70]
    assert not mhdlines.steady(lit, 0.0, 1.0, 2.0, 5.0, 50.0)[30:50].any(), "chirp"
    assert not mhdlines.steady(lit, 0.0, 1.0, 2.0, 5.0, 200.0).any(), "120 ms < 200"


def test_a_region_crossing_80_khz_is_mhd_like_below_it_and_ae_above_it():
    """The rules act on the regions lit below 80 kHz: a region crossing 80 kHz is
    cut there, and in a present frame its part above stays AE, as in pseudo-v1."""
    dy = 500 / 512  # bin 81 is 79.1 kHz, bin 82 80.1 kHz
    lit = np.zeros((257, 300), dtype=bool)
    lit[70:96, 40:140] = True  # 68-94 kHz, from absent time into present
    lit[30, 100:300] = True  # a steady line at 29 kHz, present only
    lit[30:96, 220] = True  # a streak from it up to 94 kHz
    lit[150, 100:300] = True  # AE at 146 kHz, so no present column is unlit
    chirp = np.zeros_like(lit)
    _chirp(chirp, 45, range(150, 200), 2)  # 44-67 kHz for 102 ms: AE
    lit |= chirp
    state = np.full(300, PRESENT)
    state[:100] = ABSENT
    lines = ShotLines(8, 0.0, 2.048, 0.0, dy, lit, state, np.zeros_like(lit, np.uint8))
    low = lit & (np.arange(257) < 82)[:, None]
    assert np.array_equal(mhdlines.below(lit, 0.0, dy), low)
    assert mhdlines.below(np.ones((100, 1)), 0.0, 1.0).sum() == 80, "80.0 is not below"
    assert np.array_equal(mhdlines.mhd_like(lines, RULES), low & ~chirp)
    m = pseudo.build_v2(lines, RULES).mask
    assert (m[82:96, 100:140] == 1).all() and (m[82:96, 220] == 1).all(), "above"
    assert (m[70:82, 100:140] == IGNORE).all() and (m[30:82, 220] == IGNORE).all()
    assert (m[30, 100:300] == IGNORE).all() and (m[chirp] == 1).all()
    assert (m[:, :100] == 0).all(), "absent: 0, the region's absent part too"
    # The rules on whole regions would have taken both parts above 80 kHz.
    whole = mhdlines.absent_run(lit, state == ABSENT, 2.048, 20.0)
    assert whole[82:96, 100:140].all()
    assert mhdlines.steady(lit, 0.0, dy, 2.048, 1.0, 200.0)[82:96, 220].all()


def _lines(lit, state, bright=None) -> ShotLines:
    """1 kHz bins from 0 kHz, 2 ms columns."""
    if bright is None:
        bright = np.zeros(lit.shape, dtype=np.uint8)
    return ShotLines(1, 0.0, 2.0, 0.0, 1.0, lit, state, bright)


def _picking() -> ShotLines:
    """Present columns 0-199, absent 200-399, 1 kHz bins to 250 kHz:
    - C, an AE chirp (2000 present pixels);
    - E, a chirp running 30 ms into absent time (60);
    - H, a harmonic at 230 kHz running 400 ms into it (80);
    - S, a steady line at 100 kHz for 120 ms (60);
    - D, a line drifting 1 kHz every 16 ms for 200 ms (100)."""
    lit = np.zeros((251, 400), dtype=bool)
    _chirp(lit, 140, range(100), 2, thick=20)  # C
    _chirp(lit, 90, range(170, 215), 2, thick=2)  # E
    lit[230, 120:400] = True  # H
    lit[100, 0:60] = True  # S
    _chirp(lit, 82, range(100), 8)  # D
    state = np.full(400, ABSENT)
    state[:200] = PRESENT
    return _lines(lit, state)


def test_the_rules_are_the_first_run_and_the_furthest_pair_within_5_percent():
    rules, report = mhdlines.pick([_picking()])
    assert report["lit_present_px"] == 2300  # 115 pixels may go
    assert report["absent_run_px"] == {
        "20": 140,
        "50": 80,
        "100": 80,
        "200": 80,
        "500": 0,
    }
    assert report["steady_px"] == {
        "5-50": 240,
        "5-100": 140,
        "5-200": 0,
        "3-50": 240,
        "3-100": 140,
        "3-200": 0,
        "2-50": 140,
        "2-100": 140,
        "2-200": 0,
        "1-50": 140,
        "1-100": 140,
        "1-200": 0,
    }
    # Nothing is lit below 80 kHz, so every pair takes nothing: PAIRS order.
    assert set(report["steady_take_px"].values()) == {0}
    assert rules == Rules(run_ms=50.0, drift_khz=5.0, steady_ms=200.0)
    assert report["run_ms_within"] and report["steady_within"]
    assert report["rule_shots"] == [1] and report["max_loss"] == 0.05
    assert report["rule_below_khz"] == 80.0
    bands = report["bands"]
    assert len(bands) == 13 and sum(b["lit_present_px"] for b in bands) == 2300
    assert bands[-1]["lo_khz"] == 240.0 and bands[-1]["hi_khz"] == 250.0
    assert bands[11] == {
        "lo_khz": 220.0,
        "hi_khz": 240.0,
        "lit_present_px": 80,
        "absent_run_px": 0,
        "steady_px": 0,
    }, "the rules act below 80 kHz only: H stays AE"


def test_the_steady_pair_is_the_one_within_the_budget_that_reaches_furthest():
    """(5, 200) is the first of PAIRS within the budget, but (2, 100) takes more
    below 80 kHz, where the rule acts, so it is the pair chosen."""
    lit = np.zeros((251, 400), dtype=bool)
    _chirp(lit, 140, range(100), 2, thick=20)  # AE, 2000 present pixels
    _chirp(lit, 110, range(100), 16, thick=2)  # AE rising 1 kHz every 32 ms (200)
    lit[20, 0:75] = True  # an MHD line at 20 kHz for 150 ms
    lit[40, 0:110] = True  # and one at 40 kHz for 220 ms
    state = np.full(400, ABSENT)
    state[:200] = PRESENT
    rules, report = mhdlines.pick([_lines(lit, state)])
    assert report["lit_present_px"] == 2200  # 110 pixels may go
    assert report["steady_px"] == {
        "5-50": 200,
        "5-100": 200,
        "5-200": 0,
        "3-50": 200,
        "3-100": 200,
        "3-200": 0,
        "2-50": 200,
        "2-100": 0,
        "2-200": 0,
        "1-50": 200,
        "1-100": 0,
        "1-200": 0,
    }
    assert report["steady_take_px"] == {
        f"{d:g}-{s:g}": 110 if s == 200 else 185 for d, s in mhdlines.PAIRS
    }
    assert rules == Rules(run_ms=20.0, drift_khz=2.0, steady_ms=100.0)
    assert report["run_ms_within"] and report["steady_within"]
    assert [b["steady_px"] for b in report["bands"][:4]] == [0, 75, 110, 0]


def test_rules_that_cost_too_much_fall_back_to_the_least_inclusive():
    lit = np.zeros((251, 600), dtype=bool)
    lit[150, 0:200] = True  # a steady AE line, present only
    lit[200, 0:600] = True  # a line running 800 ms into absent time
    state = np.full(600, ABSENT)
    state[:200] = PRESENT
    rules, report = mhdlines.pick([_lines(lit, state)])
    assert rules == Rules(run_ms=500.0, drift_khz=1.0, steady_ms=200.0)
    assert not report["run_ms_within"] and not report["steady_within"]
    assert report["absent_run_px"]["500"] == 200
    assert report["steady_px"]["1-200"] == 400
    run, pair = mhdlines.fallbacks(rules, report)
    assert run == (
        "no absent-run candidate is within the 5% budget (20 px); its fallback, "
        "500 ms, costs 200 of the 400 lit present-frame px at 80-250 kHz (50.0%), "
        "over the budget"
    )
    assert pair.startswith("no steady candidate") and "1 kHz over 200 ms" in pair
    assert "costs 400 of the 400" in pair and "(100.0%)" in pair
    said = mhdlines.rules_md(rules, report)
    assert "**Over the budget:** No absent-run candidate" in said
    assert ". No steady candidate" in said
    with pytest.raises(ValueError, match="no present pixel"):
        mhdlines.pick([_lines(np.zeros_like(lit), state)])


def test_the_bright_level_is_the_median_at_the_ae_pixels():
    bright = np.arange(100, dtype=np.uint8).reshape(10, 10)
    lit = np.zeros((10, 10), dtype=bool)
    lit[7, :5] = True  # lit, so not background
    lines = _lines(lit, np.full(10, PRESENT), bright)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5] = 1  # 50 .. 59
    mask[6] = IGNORE
    assert mhdlines.bright_level([mask], [lines]) == 54  # floor(54.5)
    assert mhdlines.bright_level([np.zeros_like(mask)], [lines]) is None
    # The background is rows 0-4 and 7-9 but the lit five: 25 of its 75 reach 54.
    share = mhdlines.bright_background_share([mask], [lines], 54)
    assert share == pytest.approx(1 / 3)
    assert mhdlines.bright_background_share([mask], [lines], None) is None
    values = np.zeros((3, 2, 2), dtype=np.uint8)
    values[1, 0, 0], values[2, 1, 1] = 7, 9
    assert mhdlines.brightest(values).tolist() == [[7, 0], [0, 9]]


def _v2_shot() -> tuple[ShotLines, np.ndarray]:
    """The page's grid, 300 columns of 2.048 ms: absent 0-99 and 250-279, present
    100-249, outside the window from 280. The shot and its AE pixels."""
    ae = np.zeros((257, 300), dtype=bool)
    _chirp(ae, 150, range(120, 220), 10, thick=2)  # 146-157 kHz
    lit = ae.copy()
    lit[20, 50:250] = True  # an MHD line at 19.5 kHz, from absent time
    lit[100, 50:180] = True  # its harmonic at 97.7 kHz, above 80 kHz
    lit[40, 100:250] = True  # a steady line at 39 kHz, 307 ms, present only
    lit[230, 150:152] = True  # a speck
    bright = np.zeros((257, 300), dtype=np.uint8)
    bright[180, 150:160] = 230  # bright where TokEye lit nothing
    bright[190, 150:160] = 100
    state = np.full(300, ABSENT)
    state[100:250] = PRESENT
    state[280:] = OUTSIDE
    return ShotLines(7, 0.0, 2.048, 0.0, 500 / 512, lit, state, bright), ae


def test_pseudo_v2_ignores_mhd_lines_in_present_frames_and_zeroes_absent_ones():
    lines, ae = _v2_shot()
    pm = pseudo.build_v2(lines, RULES)
    m = pm.mask
    assert (pm.shot, pm.t0_ms, pm.dt_ms, pm.y0_khz) == (7, 0.0, 2.048, 0.0)
    assert m.shape == (257, 300) and m.dtype == np.uint8
    assert (m[:, :100] == 0).all() and (m[:, 250:280] == 0).all(), "absent: 0"
    assert (m[:, 280:] == IGNORE).all(), "outside the window"
    assert (m[ae] == 1).all() and (m == 1).sum() == ae.sum() + 80 == 280
    assert (m[100, 100:180] == 1).all(), "above 80 kHz a lit present pixel is AE"
    for row in (20, 40):
        assert (m[row, 120:220] == IGNORE).all(), f"the MHD-like line at bin {row}"
    assert (m[230, 150:152] == IGNORE).all(), "the speck"
    assert (m[180, 150:160] == IGNORE).all() and (m[190, 150:160] == 0).all()
    assert m[[153, 154, 157, 158], 170].tolist() == [IGNORE] * 4, "the ring"
    assert m[[5, 150, 160, 250], 170].tolist() == [0] * 4, "0-250 kHz are scored"
    assert (m[:, 220:250] == IGNORE).all() and pm.present_unlit == 30
    loose = pseudo.build_v2(lines, Rules(run_ms=1000.0, drift_khz=1.0, steady_ms=1e4))
    for row in (20, 100, 40):
        assert (loose.mask[row, 120:180] == 1).all(), "without the rules: AE"
    assert (loose.mask[180, 150:160] == 0).all(), "no bright rule without bright_u8"


def _v2_tree(tmp_path, monkeypatch):
    """The whole-shot tree with TokEye's real spacing, v3's snapshot frozen, and
    one validation shot, so three of the four pool shots are rule shots."""
    paths = round3_tree.build(tmp_path, SPLITS, tokeye_dt=0.256)
    ae_tree.env(monkeypatch, paths)
    digest = ae_tree.snapshot(paths, monkeypatch, "v3")
    monkeypatch.setattr(pseudo, "N_VAL", 1)
    return paths, digest


def test_the_command_writes_pseudo_v2_from_v3s_snapshot_and_masks_full(
    tmp_path, monkeypatch, capsys
):
    paths, digest = _v2_tree(tmp_path, monkeypatch)
    live = labels.labels_path(xpower.event_dir(paths))
    live.write_text("shot,category,t_start,t_end,confidence\n")  # not read
    assert pseudo.main(["--version", "v2"]) == 0
    out = seg.pseudo_dir(paths, "v2")
    masks = xpower.tokeye_masks(paths, "v3")
    split = pseudo.v2_split(POOL + TEST, masks)
    rule = sorted(s for s, v in split.items() if v == "train")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["pseudo"] == "pseudo-v2" and meta["rule_shots"] == rule
    assert meta["split"] == {"train": 3, "val": 1, "test": 1} and len(rule) == 3
    assert meta["labels_sha256"] == digest and meta["failed"] == []
    for shot in POOL + TEST:
        file = clean_path(masks, shot)
        assert meta["tokeye_sha256"][str(shot)] == sha256_of(file)
    # The tree's AE line is flat, so every steady pair would take it: the fallback,
    # said out loud.
    record = json.loads((out / "rules.json").read_text())
    assert record["rules"] == meta["rules"]
    assert meta["rules"] == {
        "run_ms": 20.0,
        "drift_khz": 1.0,
        "steady_ms": 200.0,
        "guard_khz": 60.0,
        "bright_u8": 220,
    }
    assert record["run_ms_within"] and not record["steady_within"]
    assert record["absent_run_px"] == {f"{r:g}": 0 for r in mhdlines.RUN_MS}
    assert record["steady_px"]["1-200"] == record["lit_present_px"] > 0
    assert set(record["steady_take_px"].values()) == {0}, "no low line in present"
    said = capsys.readouterr()
    warnings = [line for line in said.err.splitlines() if "WARNING" in line]
    assert len(warnings) == 1 and "no steady candidate is within" in warnings[0]
    assert f"costs {record['lit_present_px']} of the" in warnings[0]
    summary = json.loads(said.out.splitlines()[-1])
    assert (summary["run_ms_within"], summary["steady_within"]) == (True, False)
    assert summary["bright_u8"] == 220 and summary["failed"] == []
    # No background pixel is as bright as the tree's AE line.
    assert summary["bright_background_share"] == record["bright_background_share"]
    assert record["bright_background_share"] == 0.0
    rules_md = (out / "rules.md").read_text()
    assert rules_md.startswith("# pseudo-v2's MHD-line rules")
    assert "**Over the budget:** No steady candidate" in rules_md
    assert "IGNOREs 0.0% of those masks' present-frame background" in rules_md
    index = pd.read_csv(out / "index.csv")
    assert index.shot.tolist() == POOL + TEST and index.regions.tolist() == [2] * 5
    pm = pseudo.PseudoMask.load(out / "101.npz")
    t = pm.t0_ms + (np.arange(pm.mask.shape[1]) + 0.5) * pm.dt_ms
    ae = ((t > 310) & (t < 890)) | ((t > 2210) & (t < 2590))
    assert (pm.mask[150, ae] == 1).all(), "both AE spans, the late one after 2 s"
    assert (pm.mask[:148, ae] == 0).all() and (pm.mask[153:, ae] == 0).all()
    absent = (t > 910) & (t < 2190)
    assert (pm.mask[:, absent] == 0).all(), "the MHD mode and below 80 kHz: 0"
    assert (pm.mask[:, t > 3010] == IGNORE).all() and pm.present_unlit == 0


def test_pseudo_v1_still_reads_the_live_labels_and_v1s_masks(tmp_path, monkeypatch):
    paths, _ = _v2_tree(tmp_path, monkeypatch)
    assert pseudo.main([]) == 0
    pm = pseudo.PseudoMask.load(seg.pseudo_dir(paths) / "101.npz")
    t = pm.t0_ms + (np.arange(pm.mask.shape[1]) + 0.5) * pm.dt_ms
    assert (pm.mask[:82] == IGNORE).all() and (pm.mask[:, t > 2010] == IGNORE).all()
    meta = json.loads((seg.pseudo_dir(paths) / "meta.json").read_text())
    assert meta["pseudo"] == "pseudo-v1" and "rules" not in meta
    live = labels.labels_path(xpower.event_dir(paths))
    assert meta["labels_sha256"] == sha256_of(live)
    assert not seg.pseudo_dir(paths, "v2").exists()


def _damage(file, how: str) -> None:
    """Replace a record with bytes that are not a mask, or cut its end off, as a
    copy that stopped short would (np.load raises BadZipFile on it)."""
    data = b"not a mask" if how == "garbage" else file.read_bytes()[:-200]
    file.write_bytes(data)


@pytest.mark.parametrize("how", ["garbage", "truncated"])
def test_a_rule_shot_that_cannot_be_read_stops_before_anything_is_written(
    tmp_path, monkeypatch, capsys, how
):
    paths, _ = _v2_tree(tmp_path, monkeypatch)
    masks = xpower.tokeye_masks(paths, "v3")
    split = pseudo.v2_split(POOL + TEST, masks)
    shot = min(s for s, v in split.items() if v == "train")
    _damage(clean_path(masks, shot), how)
    with pytest.raises(SystemExit) as error:
        pseudo.main(["--version", "v2"])
    assert error.value.code != 0
    assert f"rule shot {shot}" in capsys.readouterr().err
    out = seg.pseudo_dir(paths, "v2")
    assert not (out / "meta.json").exists() and not (out / "rules.json").exists()


def test_a_shot_whose_record_is_cut_short_fails_and_the_others_are_written(
    tmp_path, monkeypatch
):
    paths, _ = _v2_tree(tmp_path, monkeypatch)
    _damage(clean_path(xpower.tokeye_masks(paths, "v3"), TEST[0]), "truncated")
    assert pseudo.main(["--version", "v2"]) == 1  # the test shot: not a rule shot
    meta = json.loads((seg.pseudo_dir(paths, "v2") / "meta.json").read_text())
    assert meta["failed"] == TEST and meta["shots"] == len(POOL)
