"""below80: the owner-absent frames where v3 says AE off an MHD frame, or where
TokEye lights only below 80 kHz once pseudo-v2's steady lines are removed, or
below 80 kHz past pseudo-v3's per-line MHD markers."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from labeler.ae import seg, xpower
from labeler.ae.seg import markers, mhdlines
from labeler.ae.seg.mhdlines import Rules
from labeler.ae.xpower import below80, evaluate
from labeler.ae.xpower.data import clean_path
from labeler.config import Paths, sha256_of

from . import ae_tree, round3_tree

POOL = [101, 102, 103, 104]
TEST = [111]
SPLITS = {**dict.fromkeys(POOL, "train"), **dict.fromkeys(TEST, "valid")}
RULES = Rules(run_ms=20.0, drift_khz=1.0, steady_ms=200.0, bright_u8=220)
#: Frames (10 ms) where the stand-in model says 0.9: owner-present (40-49),
#: owner-absent (100-104, 175-177) and absent on the MHD mode (130-134).
FIRES = [*range(40, 50), *range(100, 105), *range(130, 135), *range(175, 178)]


def _fires(model, rows, first, n, *, band, context=20):
    frame = first + np.arange(n)
    prob = np.where(np.isin(frame, FIRES), 0.9, 0.1).astype(np.float32)
    return prob, np.ones(n, dtype=bool)


def _below_80(paths, shot) -> None:
    """Add to the shot's masks-full record a chirp from 61 to 78 kHz over
    1700-1800 ms and a steady 25 kHz line over 1900-2150 ms."""
    file = clean_path(xpower.tokeye_masks(paths, "v3"), shot)
    with np.load(file) as z:
        saved = {key: z[key] for key in z.files}
    t = saved["t_ms"]
    clean = np.unpackbits(saved["mask_clean"], axis=-1, count=len(t)).astype(bool)
    chirp = np.flatnonzero((t >= 1700) & (t < 1800))
    bins = (124 + (t[chirp] - 1700) * 0.36).astype(int)  # TokEye bins 124-159
    clean[:, bins, chirp] = True
    clean[:, 50, (t >= 1900) & (t < 2150)] = True  # 24.9 kHz for 250 ms
    saved["mask_clean"] = np.packbits(clean, axis=-1)
    np.savez(file, **saved)


def _crossing(paths, shot) -> None:
    """Add to the shot's masks-full record a steady 25 kHz line over 2640-2920 ms,
    a 10.7 kHz blip off it over 2860-2880 ms and again over 2900-2920 ms, and
    over 2900-2920 ms a bar joining the line to 81 kHz."""
    file = clean_path(xpower.tokeye_masks(paths, "v3"), shot)
    with np.load(file) as z:
        saved = {key: z[key] for key in z.files}
    t = saved["t_ms"]
    clean = np.unpackbits(saved["mask_clean"], axis=-1, count=len(t)).astype(bool)
    clean[:, 50, (t >= 2640) & (t < 2920)] = True  # page bin 26, 25.4 kHz
    clean[:, 20, (t >= 2860) & (t < 2880)] = True  # page bin 11, 10.7 kHz
    clean[:, 20, (t >= 2900) & (t < 2920)] = True
    clean[:, 50:166, (t >= 2900) & (t < 2920)] = True  # page bins 26-83, to 81 kHz
    saved["mask_clean"] = np.packbits(clean, axis=-1)
    np.savez(file, **saved)


def _tree(tmp_path, monkeypatch):
    """The whole-shot tree, v3's snapshot, its chosen model (111 its test shot)
    and the record of its one test, pseudo-v2's rules, shot 101's low lines, and
    the stand-in model."""
    paths = round3_tree.build(tmp_path, SPLITS, tokeye_dt=0.256)
    ae_tree.env(monkeypatch, paths)
    digest = ae_tree.snapshot(paths, monkeypatch, "v3")
    models = ae_tree.chosen(paths, {**dict.fromkeys(POOL, "train"), 111: "test"}, "v3")
    # below80 gives v3's test F1, so it comes after v3's one test: its record.
    meta = {"model_sha256": sha256_of(evaluate.chosen_model(models))}
    (models / "evaluation.json").write_text(json.dumps({"meta": meta}))
    rules = seg.pseudo_dir(paths, "v2") / "rules.json"
    rules.parent.mkdir(parents=True)
    rules.write_text(json.dumps({"rules": asdict(RULES)}))
    # The catalog's NTM table, which the per-line reason reads: no interval.
    ntm = paths.label_tables / markers.NTM_TABLE
    ntm.parent.mkdir(parents=True, exist_ok=True)
    ntm.write_text("shot,category,t_start,t_end,confidence\n")
    _below_80(paths, 101)
    monkeypatch.setattr(evaluate, "probabilities", _fires)
    return paths, digest


def test_the_table_its_reasons_and_its_place(tmp_path):
    assert below80.COLUMNS == (
        "shot",
        "split",
        "t_start",
        "t_end",
        "why",
        "f_lo_khz",
        "f_hi_khz",
        "p_max",
        "frames",
    )
    assert below80.REASONS == ("model", "tokeye_below80", "tokeye_lines")
    assert below80.FLOOR_KHZ == 80.0 and below80.VERSION == "v3"
    # The steady lines stop at pseudo-v2's own cut, which must be the floor.
    assert below80.FLOOR_KHZ == mhdlines.RULE_BELOW_KHZ
    suggestions = tmp_path / "suggestions" / "ae_xpower" / "v3"
    assert below80.below80_file(Paths(root=tmp_path)) == suggestions / "below80.csv"


def test_tokeye_below_80_takes_the_chirp_not_a_steady_line_or_a_harmonic(
    tmp_path, monkeypatch
):
    paths, _ = _tree(tmp_path, monkeypatch)
    _, saved = xpower.read_snapshot(paths, "v3")
    flag, f_lo, f_hi = below80.low_frames(paths, 101, saved[101], RULES, 0, 300)
    assert np.flatnonzero(flag).tolist() == list(range(170, 180))
    assert (f_lo[flag] > 60).all() and (f_hi[flag] < 80).all()
    assert (f_lo[flag] <= f_hi[flag]).all()
    assert np.isnan(f_lo[~flag]).all() and np.isnan(f_hi[~flag]).all()
    # The chirp's TokEye bins 124-159 are the page's bins 63-80 (page bin k takes
    # TokEye bins 2k - 2 and 2k - 1): 61.52 and 78.125 kHz at the store's y0 0, DY.
    assert f_lo[flag].min() == f_lo[170] == pytest.approx(63 * ae_tree.DY)
    assert f_hi[flag].max() == f_hi[179] == pytest.approx(80 * ae_tree.DY)
    loose = Rules(run_ms=20.0, drift_khz=1.0, steady_ms=1000.0)
    flag, _, _ = below80.low_frames(paths, 101, saved[101], loose, 0, 300)
    assert np.flatnonzero(flag).tolist() == [*range(170, 180), *range(190, 215)]
    flag, _, _ = below80.low_frames(paths, 111, saved[111], loose, 0, 300)
    assert not flag.any(), "the MHD mode's 98 kHz harmonic is above 80 kHz"


def test_a_steady_line_is_cut_at_80_khz_as_pseudo_v2_cuts_it(tmp_path, monkeypatch):
    """A steady line joined to pixels at or above 80 kHz is removed below 80 kHz
    only (`mhdlines.below`, as `mhdlines.mhd_like` finds pseudo-v2's lines): the
    pixels above stay, so the columns holding them do not flag, though a blip
    off the line is left below 80 kHz there. Where the line has no bar, its
    removal leaves the blip alone, and those frames flag."""
    paths, _ = _tree(tmp_path, monkeypatch)
    _crossing(paths, 102)
    _, saved = xpower.read_snapshot(paths, "v3")
    flag, _, _ = below80.low_frames(paths, 102, saved[102], RULES, 0, 300)
    assert np.flatnonzero(flag).tolist() == [286, 287], "not 290-291, the bar's"


def test_the_command_lists_owner_absent_frames_by_reason(tmp_path, monkeypatch):
    paths, digest = _tree(tmp_path, monkeypatch)
    assert below80.main(["--workers", "1"]) == 0
    file = below80.below80_file(paths)
    table = pd.read_csv(file)
    assert tuple(table.columns) == below80.COLUMNS
    got = {
        shot: [(r.t_start, r.t_end, r.why, r.frames) for r in rows.itertuples()]
        for shot, rows in table.groupby("shot")
    }
    # The chirp is no MHD line to pseudo-v3's markers either; the steady 25 kHz
    # line and the MHD mode are, and the mode's 98 kHz harmonic is above 80 kHz.
    lines = "tokeye_below80+tokeye_lines"
    assert got[101] == [
        (1000, 1050, "model", 5),
        (1700, 1750, lines, 5),
        (1750, 1780, f"model+{lines}", 3),
        (1780, 1800, lines, 2),
    ]
    for shot in (102, 103, 104, 111):
        assert got[shot] == [(1000, 1050, "model", 5), (1750, 1780, "model", 3)]
    assert set(table[table.shot == 101].split) == {"train"}
    assert set(table[table.shot == 111].split) == {"test"}
    one = table[table.shot == 101]
    assert one.p_max.tolist() == [0.9, 0.1, 0.9, 0.1]
    assert one.f_lo_khz.isna().tolist() == [True, False, False, False]
    assert (one.f_lo_khz.dropna() > 60).all() and (one.f_hi_khz.dropna() < 80).all()
    # The chirp's page bins 63 and 80: 61.52 and 78.125 kHz, a tie written 78.12.
    assert one.f_lo_khz.min() == pytest.approx(round(63 * ae_tree.DY, 2))
    assert one.f_hi_khz.max() == pytest.approx(round(80 * ae_tree.DY, 2))
    record = json.loads(file.with_suffix(".json").read_text())
    assert record["frames"] == {"model": 37, lines: 7, f"model+{lines}": 3}
    assert record["reasons"] == {"model": 40, "tokeye_below80": 10, "tokeye_lines": 10}
    assert record["tier"] == "suggestions" and record["labels_sha256"] == digest
    assert record["rules"] == asdict(RULES) and record["failed"] == []
    test = record["test"]
    assert test["shots"] == [111] and test["removed_frames"] == 8
    # 10 of 100 present frames found, 13 absent ones called: 8 of them flagged.
    assert test["f1_with"]["value"] == pytest.approx(20 / 123)
    assert test["f1_without"]["value"] == pytest.approx(20 / 115)
    report = file.with_suffix(".md").read_text()
    assert report.startswith("# ") and "suggestions" in report


def test_below80_needs_pseudo_v2s_rules(tmp_path, monkeypatch, capsys):
    paths, _ = _tree(tmp_path, monkeypatch)
    rules = seg.pseudo_dir(paths, "v2") / "rules.json"
    rules.unlink()
    with pytest.raises(SystemExit) as error:
        below80.main(["--workers", "1"])
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert str(rules) in stderr and "Traceback" not in stderr
    assert not below80.below80_file(paths).exists()


def _no_shot(*args):
    raise AssertionError("a shot ran before the command refused")


def test_below80_comes_after_v3s_one_test(tmp_path, monkeypatch, capsys):
    """below80.json gives v3's F1 on its test shots: before any shot, the command
    refuses while v3's `evaluation.json` is missing or scored another model."""
    paths, _ = _tree(tmp_path, monkeypatch)
    monkeypatch.setattr(below80, "run_all", _no_shot)
    evaluation = xpower.model_dir(paths, "v3") / "evaluation.json"

    def refused() -> str:
        with pytest.raises(SystemExit) as error:
            below80.main(["--workers", "1"])
        assert error.value.code != 0
        stderr = capsys.readouterr().err
        assert str(evaluation) in stderr and "Traceback" not in stderr
        return stderr

    evaluation.unlink()
    assert "no test yet" in refused()
    evaluation.write_text(json.dumps({"meta": {"model_sha256": "0" * 64}}))
    assert "another model" in refused()
    assert not below80.below80_file(paths).exists()


def test_a_failed_shot_is_named_and_left_out_of_the_f1s(tmp_path, monkeypatch):
    paths, _ = _tree(tmp_path, monkeypatch)
    clean_path(xpower.tokeye_masks(paths, "v3"), 111).unlink()  # the test shot
    assert below80.main(["--workers", "1"]) == 1
    file = below80.below80_file(paths)
    assert 111 not in set(pd.read_csv(file).shot)
    record = json.loads(file.with_suffix(".json").read_text())
    assert record["failed"] == [111] and record["shots"] == 4
    test = record["test"]
    assert test["shots"] == [] and test["f1_with"]["value"] is None
    report = file.with_suffix(".md").read_text()
    assert "Failed shots, left out of the table and of both F1s: 111." in report
    assert "would bias a later version's test on the same shots" in report
