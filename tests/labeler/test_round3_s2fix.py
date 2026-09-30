"""S2's fix: pseudo-v3's per-line MHD markers and their gate, SegNet v3 on v2's
split, the MHD-lines metric, SegNet v2's diagnosis, below80's per-line reason and
the paper's examples past 2 s."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from labeler.ae import seg, xpower
from labeler.ae.seg import diagnose, evaluate, markers, mhdlines, pseudo, train
from labeler.ae.seg.markers import GateBox, Markers
from labeler.ae.seg.mhdlines import ShotLines
from labeler.ae.seg.pseudo import IGNORE
from labeler.ae.xpower import below80
from labeler.ae.xpower.data import FULL_BAND_KHZ, clean_path
from labeler.config import Paths, sha256_of
from labeler.events.catalog.states import ABSENT, PRESENT
from labeler.paper import shots
from labeler.scoring.frames import OUTSIDE

from . import ae_tree, round3_tree
from .ae_tree import AE_MS, MHD_MS
from .test_ae_seg_train import Small
from .test_round3_below80 import _tree as _below80_tree

POOL = [101, 102, 103, 104]
TEST = [111]
SPLITS = {**dict.fromkeys(POOL, "train"), **dict.fromkeys(TEST, "valid")}
NTM_111_MS = (350.0, 500.0)  # a catalog NTM interval inside 111's AE span


def test_segnet_v3_is_gated_and_says_its_test_is_a_second_use(tmp_path):
    v1, v2, v3 = (seg.SEG_VERSIONS[v] for v in ("v1", "v2", "v3"))
    assert (v3.pseudo, v3.band_khz, v3.labels, v3.ae_version) == (
        "pseudo-v3",
        FULL_BAND_KHZ,
        "v3",
        "v3",
    )
    assert v3.whole_window and v3.gated and v3.test_of == "v2"
    assert not (v1.gated or v2.gated) and v1.test_of is v2.test_of is None
    assert seg.reuse_note("v1") is None and seg.reuse_note("v2", 60) is None
    note = seg.reuse_note("v3", 60)
    assert "SegNet v3's test is a second use of SegNet v2's 60 test shots" in note
    assert "models/ae_seg/v2/diagnosis.md" in note and "not an unbiased" in note
    assert note.endswith("Tier: suggestions.")
    paths = Paths(root=tmp_path)
    assert seg.pseudo_dir(paths, "v3").name == "pseudo-v3"
    assert seg.model_dir(paths, "v3") == tmp_path / "models" / "ae_seg" / "v3"
    fixed = Markers()
    assert (fixed.drift_khz, fixed.steady_ms, fixed.gap_ms) == (2.0, 100.0, 20.0)
    assert (fixed.guard_khz, fixed.harmonics, fixed.f0_min_khz) == (30.0, 5, 4.0)
    assert fixed.below_khz == mhdlines.RULE_BELOW_KHZ == 80.0
    assert fixed.bright_u8 is None
    gate = {box.shot for box in markers.GATE}
    assert gate == {176035, 176041, 176053} and not gate & {170720, 170790}


def _lines(lit, state, bright=None, shot=1) -> ShotLines:
    """1 kHz bins from 0 kHz, 2 ms columns."""
    if bright is None:
        bright = np.where(lit, 200, 0).astype(np.uint8)
    return ShotLines(shot, 0.0, 2.0, 0.0, 1.0, lit, state, bright)


def test_a_column_run_splits_at_a_brightness_minimum_into_two_segments():
    lit = np.zeros((30, 2), dtype=bool)
    lit[10:17, 0] = True  # two lines touching: bright at 11 and 15
    lit[20, 0] = True
    lit[10:12, 1] = True
    bright = np.zeros((30, 2), dtype=np.uint8)
    bright[10:17, 0] = [50, 200, 100, 20, 100, 200, 50]
    bright[10:12, 1] = [10, 10]  # a tie: the lower bin
    ids, ridge = markers.segments(lit, bright)
    assert len(set(ids[10:13, 0])) == len(set(ids[13:17, 0])) == 1
    assert len({ids[10, 0], ids[13, 0], ids[20, 0], ids[10, 1]}) == 4
    assert (ids[~lit] == 0).all() and (ids[lit] > 0).all()
    assert np.flatnonzero(ridge[:, 0]).tolist() == [11, 15, 20]
    assert np.flatnonzero(ridge[:, 1]).tolist() == [10]


def _marked() -> ShotLines:
    """Present, 1 kHz bins, 2 ms columns (a steady line needs 50 columns):
    - rows 20 (0-59) and 27 (0-29 and 38-69, a 16 ms gap): steady lines;
    - row 41 (0-59): 20 kHz's second harmonic, one bin off;
    - row 50 (0-59): steady, but above the 30 kHz guard, and no harmonic;
    - row 80 (0-59): 20 kHz's fourth harmonic, at the 80 kHz cut;
    - a chirp over 5-16 kHz, one bin every 5 columns over 60-119."""
    lit = np.zeros((120, 120), dtype=bool)
    lit[20, 0:60] = True
    lit[27, 0:30] = True
    lit[27, 38:70] = True
    lit[41, 0:60] = True
    lit[50, 0:60] = True
    lit[80, 0:60] = True
    for c in range(60, 120):
        lit[5 + (c - 60) // 5, c] = True
    return _lines(lit, np.full(120, PRESENT))


def test_the_markers_take_steady_lines_their_harmonics_and_ntm_lines_below_80():
    lines = _marked()
    lit = lines.lit

    def rows(got) -> list[int]:
        return np.flatnonzero(got.any(axis=1)).tolist()

    def steady(**change) -> list[int]:
        return rows(
            markers.mhd_lines(lines, replace(Markers(), **change), none)["steady"]
        )

    none = np.zeros(120, dtype=bool)
    got = markers.mhd_lines(lines, Markers(), none)
    assert np.array_equal(
        got["steady"], lit & np.isin(np.arange(120), [20, 27])[:, None]
    )
    assert np.array_equal(got["comb"], lit & (np.arange(120) == 41)[:, None])
    assert not got["ntm"].any(), "no NTM interval"
    assert np.array_equal(got["mhd"], got["steady"] | got["comb"])
    assert steady(gap_ms=0.0) == [20], "the gap is not bridged"
    assert steady(guard_khz=60.0) == [20, 27, 41, 50]
    uncut = markers.mhd_lines(lines, Markers(), none, below_khz=math.inf)
    assert rows(uncut["comb"]) == [41, 80] and rows(uncut["steady"]) == [20, 27]
    # Inside an NTM interval a line at 4-30 kHz is MHD, steady or not.
    ntm = markers.ntm_columns([(120.0, 240.0)], 0.0, 2.0, 120)
    assert np.flatnonzero(ntm).tolist() == list(range(60, 120))
    tearing = markers.mhd_lines(lines, Markers(), ntm)["ntm"]
    chirp = lit & (np.arange(120) < 17)[:, None]
    assert np.array_equal(tearing & chirp, chirp)
    assert not (tearing[:, :60]).any() and rows(tearing) == [*range(5, 17), 27]


def _v3_shot() -> tuple[ShotLines, np.ndarray, np.ndarray]:
    """The page's grid, 300 columns of 2.048 ms: absent 0-99 and 250-279, present
    100-249, outside the window from 280. The shot, its AE pixels (above 80 kHz,
    and a chirp at 44-53 kHz in present time), and its MHD lines: a steady line at
    19.5 kHz and its harmonic at 40 kHz over 50-249, from absent into present."""
    dy = 500 / 512
    ae = np.zeros((257, 300), dtype=bool)
    for c in range(120, 220):
        ae[150 + (c - 120) // 10 : 152 + (c - 120) // 10, c] = True  # 146-157 kHz
    for c in range(150, 230):
        ae[45 + (c - 150) // 8, c] = True  # 44-53 kHz, not steady
    mhd = np.zeros_like(ae)
    mhd[20, 50:250] = True
    mhd[41, 50:250] = True
    lit = ae | mhd
    for c in range(46):
        lit[50 + c // 4, c] = True  # an absent chirp at 49-60 kHz: not MHD
    lit[100, 50:100] = True  # 97.7 kHz, absent: above 80 kHz
    state = np.full(300, ABSENT)
    state[100:250] = PRESENT
    state[280:] = OUTSIDE
    bright = np.where(lit, 200, 0).astype(np.uint8)
    return ShotLines(7, 0.0, 2.048, 0.0, dy, lit, state, bright), ae, mhd


def test_pseudo_v3_ignores_absent_frames_below_80_khz_but_the_mhd_lines(tmp_path):
    lines, ae, mhd = _v3_shot()
    pm = pseudo.build_v3(lines, Markers(), np.zeros(300, dtype=bool))
    m = pm.mask
    absent, present = lines.state == ABSENT, lines.state == PRESENT
    low = np.arange(257) < 82  # bin 81 is 79.1 kHz, bin 82 80.1 kHz
    assert (m[:, 280:] == IGNORE).all(), "outside the window"
    assert (m[~low][:, absent] == 0).all(), "absent above 80 kHz: 0, 97.7 kHz too"
    zero = (m == 0) & low[:, None] & absent[None, :]
    assert np.array_equal(zero, mhd & absent[None, :]), "the MHD lines only"
    assert (m[low][:, absent] != 1).all() and (m[:82, :46] == IGNORE).all()
    assert (m[mhd & present[None, :]] == IGNORE).all()
    assert np.array_equal(pm.mhd, mhd & present[None, :]) and pm.mhd.sum() == 300
    assert (m[ae] == 1).all() and (m == 1).sum() == ae.sum()
    assert pm.present_unlit == 40, "100-119 and 230-249: no AE"
    # pseudo-v2 on the same shot: every absent pixel 0, and no `mhd` saved.
    v2 = pseudo.build_v2(lines, mhdlines.Rules(20.0, 2.0, 100.0))
    assert v2.mhd is None and (v2.mask[:, absent] == 0).all()
    v2.save(tmp_path / "v2.npz")
    pm.save(tmp_path / "v3.npz")
    with np.load(tmp_path / "v2.npz") as z:
        assert "mhd" not in z.files
    assert pseudo.PseudoMask.load(tmp_path / "v2.npz").mhd is None
    back = pseudo.PseudoMask.load(tmp_path / "v3.npz")
    assert np.array_equal(back.mhd, pm.mhd) and np.array_equal(back.mask, m)


def test_a_gate_box_passes_on_its_share_and_fails_off_the_rule_shots(monkeypatch):
    lit = np.zeros((30, 100), dtype=bool)
    lit[15] = True
    state = np.full(100, PRESENT)
    state[50:] = ABSENT
    mask = np.zeros((30, 100), dtype=np.uint8)
    mask[15, :25] = 1
    mask[15, 25:50] = IGNORE
    boxes = (
        GateBox(1, "half AE", (0.0, 100.0), (10.0, 20.0), PRESENT, 1, False, 0.5),
        GateBox(1, "not AE", (0.0, 100.0), (10.0, 20.0), PRESENT, 1, True, 0.05),
        GateBox(1, "hard negative", (0.0, 200.0), (10.0, 20.0), ABSENT, 0, False, 0.9),
        GateBox(1, "nothing lit", (0.0, 100.0), (20.0, 30.0), PRESENT, 1, True, 0.05),
        GateBox(2, "a val shot", (0.0, 100.0), (10.0, 20.0), PRESENT, 1, True, 0.05),
        GateBox(3, "unread", (0.0, 100.0), (10.0, 20.0), PRESENT, 1, True, 0.05),
    )
    monkeypatch.setattr(markers, "GATE", boxes)
    rows = markers.gate_rows({1: mask}, {1: _lines(lit, state)}, {1, 3})
    got = [(r["lit_px"], r["share"], r["passed"], r["why"]) for r in rows]
    assert got == [
        (50, 0.5, True, ""),
        (50, 0.5, False, ""),
        (50, 1.0, True, ""),
        (0, None, False, ""),
        (0, None, False, "not a rule shot"),
        (0, None, False, "not read"),
    ]
    assert rows[0]["t_ms"] == [0.0, 100.0] and rows[0]["says"] == "half AE"


def _ntm_table(paths, rows) -> str:
    """The catalog's NTM table in the tree (category 1: present); its sha256."""
    file = paths.label_tables / markers.NTM_TABLE
    file.parent.mkdir(parents=True, exist_ok=True)
    head = "shot,category,t_start,t_end,confidence"
    file.write_text("\n".join([head, *(f"{s},{c},{a},{b}," for s, c, a, b in rows)]))
    return sha256_of(file)


def _low_steady(paths, shot: int) -> None:
    """Light the shot's whole-shot TokEye mask at 19.5 kHz (TokEye bin 39, page
    bin 20) over `AE_MS` too: a steady MHD line in the owner's present frames."""
    file = clean_path(xpower.tokeye_masks(paths, "v3"), shot)
    with np.load(file) as z:
        saved = {key: z[key] for key in z.files}
    t = saved["t_ms"]
    lit = np.unpackbits(saved["mask_clean"], axis=-1, count=len(t)).astype(bool)
    lit[:, 39, (t >= AE_MS[0]) & (t < AE_MS[1])] = True
    saved["mask_clean"] = np.packbits(lit, axis=-1)
    np.savez(file, **saved)


def _split_file(file, split) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"{s},{v}\n" for s, v in sorted(split.items()))
    file.write_text("shot,split\n" + rows)


def _refused(command, argv, capsys) -> str:
    with pytest.raises(SystemExit) as error:
        command(argv)
    assert error.value.code != 0
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    return stderr


def test_pseudo_v3_is_gated_and_segnet_v3_trains_on_v2s_split(
    tmp_path, monkeypatch, capsys
):
    paths = round3_tree.build(tmp_path, SPLITS, tokeye_dt=0.256)
    for shot in POOL + TEST:
        _low_steady(paths, shot)
    ae_tree.env(monkeypatch, paths)
    digest = ae_tree.snapshot(paths, monkeypatch, "v3")
    monkeypatch.setattr(pseudo, "N_VAL", 1)
    ntm_sha256 = _ntm_table(paths, [(111, 1, *NTM_111_MS), (101, 2, 0.0, 3000.0)])
    out = seg.pseudo_dir(paths, "v3")
    rules_file = out / "rules.json"

    # The gate's own shots are no rule shots here: it fails and writes no mask.
    assert pseudo.main(["--version", "v3"]) == 2
    assert sorted(p.name for p in out.iterdir()) == ["rules.json", "rules.md"]
    record = json.loads(rules_file.read_text())
    assert record["gate_passed"] is False and record["within"] is True
    assert {r["why"] for r in record["gate"]} == {"not a rule shot"}
    assert record["cost_px"] == 0 < record["lit_present_px"]
    assert "**The gate: FAIL.** No mask was written." in (out / "rules.md").read_text()
    assert not pseudo.gate_passed(paths, "v3")
    capsys.readouterr()
    models = ae_tree.chosen(paths, {**dict.fromkeys(POOL, "train"), 111: "test"}, "v3")
    stderr = _refused(train.main, ["--version", "v3"], capsys)
    assert str(rules_file) in stderr and "gate has not passed" in stderr

    # A gate on a rule shot of this tree: the AE line stays AE, the steady
    # 19.5 kHz line is not AE in present frames and a hard negative in absent ones.
    split = pseudo.v2_split(POOL + TEST, xpower.tokeye_masks(paths, "v3"))
    rule = sorted(s for s, v in split.items() if v == "train")
    shot = rule[0]
    boxes = (
        GateBox(shot, "AE", AE_MS, (140.0, 150.0), PRESENT, 1, False, 0.9),
        GateBox(shot, "MHD", AE_MS, (15.0, 25.0), PRESENT, 1, True, 0.05),
        GateBox(shot, "MHD", MHD_MS, (15.0, 25.0), ABSENT, 0, False, 0.9),
    )
    monkeypatch.setattr(markers, "GATE", boxes)
    assert pseudo.main(["--version", "v3"]) == 0
    assert pseudo.gate_passed(paths, "v3")
    record = json.loads(rules_file.read_text())
    assert [r["passed"] for r in record["gate"]] == [True] * 3
    assert record["gate"][1]["share"] == 0.0 and record["gate"][2]["share"] == 1.0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["pseudo"] == "pseudo-v3" and meta["gate_passed"] is True
    assert meta["rule_shots"] == rule and meta["labels_sha256"] == digest
    assert meta["ntm_sha256"] == ntm_sha256 == record["ntm_sha256"]
    assert meta["rules"] == asdict(Markers(bright_u8=220)) == record["rules"]
    assert pd.read_csv(out / "index.csv").shot.tolist() == POOL + TEST
    pm = pseudo.PseudoMask.load(out / "101.npz")
    t = pm.t0_ms + (np.arange(pm.mask.shape[1]) + 0.5) * pm.dt_ms
    early = (t > 310) & (t < 890)
    assert (pm.mask[150, early | ((t > 2210) & (t < 2590))] == 1).all()
    assert (pm.mask[20, early] == IGNORE).all() and pm.mhd[20, early].all()
    assert np.flatnonzero(pm.mhd.any(axis=1)).tolist() == [20]
    absent = (t > 910) & (t < 2190)
    assert (pm.mask[82:, absent] == 0).all(), "above 80 kHz: 0"
    below = pm.mask[:82][:, absent]
    assert set(np.nonzero(below == 0)[0]) == {20}, "the MHD mode only"
    assert (pm.mask[20, (t > 1210) & (t < 1490)] == 0).all()
    assert (pm.mask[:82, (t > 910) & (t < 1190)] == IGNORE).all()

    # SegNet v3 trains only on SegNet v2's split, shot for shot.
    v2_split = seg.model_dir(paths, "v2") / "split.csv"
    stderr = _refused(train.main, ["--version", "v3"], capsys)
    assert str(v2_split) in stderr and "FileNotFoundError" in stderr
    _split_file(v2_split, {**split, shot: "val"})
    stderr = _refused(train.main, ["--version", "v3"], capsys)
    assert "SegNet v2's split differs from ours" in stderr
    assert not seg.model_dir(paths, "v3").exists()
    _split_file(v2_split, split)
    monkeypatch.setattr(train, "TrainConfig", Small)
    assert train.main(["--version", "v3", "--epochs", "1"]) == 0
    models_v3 = seg.model_dir(paths, "v3")
    _, blob = train.load(models_v3 / "model.pt")
    assert train.blob_version(blob) == "v3" and train.blob_band(blob) == FULL_BAND_KHZ
    assert train.read_split(models_v3 / "split.csv") == split
    trained = json.loads((models_v3 / "training.json").read_text())
    assert trained["test_reuse"] == seg.reuse_note("v3", 1)
    assert trained["pseudo"] == "pseudo-v3"
    assert models.is_dir()

    # Its one test states the second use and counts the MHD lines it cannot see.
    assert evaluate.main(["--version", "v3"]) == 0
    scored = json.loads((models_v3 / "evaluation.json").read_text())
    assert scored["meta"]["evaluation_inputs"]["ntm_sha256"] == ntm_sha256
    lines = scored["mhd_lines"]
    assert set(lines["shots"]) == {"111"}
    test_pm = pseudo.PseudoMask.load(out / "111.npz")
    assert lines["mhd_px"] == int(test_pm.mhd.sum()) > 0
    n = test_pm.mask.shape[1]
    ntm = markers.ntm_columns([NTM_111_MS], test_pm.t0_ms, test_pm.dt_ms, n)
    assert lines["ntm_cols"] == int(ntm.sum()) > 0 and lines["ntm_lit_px"] > 0
    assert set(evaluate.MHD_LINE_KEYS) <= set(lines["shots"]["111"])
    report = (models_v3 / "evaluation.md").read_text()
    assert evaluate.BY_CONSTRUCTION in report
    assert seg.reuse_note("v3", 1) in report
    assert "## MHD lines in present frames" in report

    # The side command: the same calls, broken down, and nothing re-selected.
    assert diagnose.main(["--version", "v3"]) == 0
    broken = json.loads((models_v3 / "diagnosis.json").read_text())
    assert broken["matches_evaluation"] is True
    assert broken["meta"]["side_command"] and broken["meta"]["tier"] == "suggestions"
    assert broken["mhd_lines"] == lines
    bands = ["all", "0-20", "20-40", "40-60", "60-80", "80-250"]
    assert list(broken["pixels"]) == bands
    assert set(broken["pixels"]["all"]) == {"whole", "0-2 s", "after 2 s"}
    assert set(broken["frames"]) == {"ae_seg", "ae_seg_80_250"}
    explained = (models_v3 / "diagnosis.md").read_text()
    assert evaluate.BY_CONSTRUCTION in explained and "cut to 80-250 kHz" in explained
    # A model other than the one scored is refused.
    (models_v3 / "split.csv").write_text("shot,split\n")
    stderr = _refused(diagnose.main, ["--version", "v3"], capsys)
    assert "not the file evaluation.json scored" in stderr


def test_the_mhd_lines_counts_and_their_summary():
    on = np.zeros((4, 6), dtype=bool)
    on[0, :3] = on[3, 4] = True
    mhd = np.zeros_like(on)
    mhd[0, 1:5] = True
    lit = np.zeros_like(on)
    lit[1, :] = lit[0, 2] = True
    low = np.array([True, True, True, False])
    ntm = np.array([0, 1, 1, 1, 1, 0], dtype=bool)
    window = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    got = evaluate.mhd_line_counts(on, mhd, lit, low, ntm, window)
    assert got == {
        "mhd_px": 4,
        "mhd_ae_px": 2,
        "ntm_cols": 3,
        "ntm_lit_px": 4,
        "ntm_ae_px": 2,
    }
    nothing = dict.fromkeys(evaluate.MHD_LINE_KEYS, 0)
    summary = evaluate.mhd_lines_summary({12: got, 3: nothing})
    assert summary["mhd_ae_share"] == 0.5 and summary["ntm_ae_share"] == 0.5
    assert list(summary["shots"]) == ["3", "12"]
    assert evaluate.mhd_lines_summary({3: nothing})["mhd_ae_share"] is None
    md = "\n".join(evaluate.mhd_lines_md(summary, "pseudo-v3"))
    assert "4 MHD-like pixels pseudo-v3 IGNORES in present frames: 2 (50.0%)" in md
    body = [line for line in md.splitlines() if line.startswith(("| 12 |", "| 3 |"))]
    assert [line.split()[1] for line in body] == ["12", "3"]


def _line_across_80_and_a_drift(paths, shot) -> None:
    """Add to the shot's masks-full record a bar over 68-88 kHz at 1700-1760 ms,
    and a line drifting from 10.7 to 20.5 kHz over 1900-2000 ms."""
    file = clean_path(xpower.tokeye_masks(paths, "v3"), shot)
    with np.load(file) as z:
        saved = {key: z[key] for key in z.files}
    t = saved["t_ms"]
    clean = np.unpackbits(saved["mask_clean"], axis=-1, count=len(t)).astype(bool)
    clean[:, 139:180, (t >= 1700) & (t < 1760)] = True  # page bins 70-90
    drift = np.flatnonzero((t >= 1900) & (t < 2000))
    clean[:, 2 * (11 + (t[drift] - 1900) // 10).astype(int) - 1, drift] = True
    saved["mask_clean"] = np.packbits(clean, axis=-1)
    np.savez(file, **saved)


def test_tokeye_lines_keeps_a_line_running_past_80_khz_not_an_ntm_line(
    tmp_path, monkeypatch
):
    """The per-line reason needs nothing clear at or above 80 kHz, as the owner
    saw only 80-250 kHz; the steady rule's reason does. Inside a catalog NTM
    interval a drifting line below 30 kHz is MHD, outside one it is not."""
    paths, _ = _below80_tree(tmp_path, monkeypatch)
    _line_across_80_and_a_drift(paths, 102)
    _, saved = xpower.read_snapshot(paths, "v3")
    label = saved[102]
    rules = mhdlines.Rules(run_ms=20.0, drift_khz=1.0, steady_ms=200.0)
    low, _, _ = below80.low_frames(paths, 102, label, rules, 0, 300)
    assert np.flatnonzero(low).tolist() == list(range(190, 200)), "the drift only"
    flag, f_lo, f_hi = below80.line_frames(paths, 102, label, 0, 300)
    assert np.flatnonzero(flag).tolist() == [*range(170, 176), *range(190, 200)]
    assert (f_hi[170:176] < 80).all() and f_lo[170] == pytest.approx(70 * ae_tree.DY)
    spans = [(1850.0, 2050.0)]
    flag, _, _ = below80.line_frames(paths, 102, label, 0, 300, spans=spans)
    assert np.flatnonzero(flag).tolist() == list(range(170, 176))
    # The MHD mode at 19.5 kHz (1200-1500 ms) is steady: never flagged.
    assert not flag[120:150].any()
    assert below80.why_name(1) == "model" and below80.why_name(6) == (
        "tokeye_below80+tokeye_lines"
    )


def test_below80_gives_each_reasons_recall_on_the_owner_cited_stretch(
    tmp_path, monkeypatch
):
    paths, _ = _below80_tree(tmp_path, monkeypatch)
    cited = ((101, 1700.0, 1800.0), (176041, 424.0, 1504.0))
    monkeypatch.setattr(below80, "OWNER_CITED", cited)
    assert below80.main(["--workers", "1", "--shots", "101"]) == 0
    file = below80.below80_file(paths)
    record = json.loads(file.with_suffix(".json").read_text())
    ours, theirs = record["owner_cited"]
    assert ours["frames"] == 10
    assert ours["listed"] == {
        "model": 3,
        "tokeye_below80": 10,
        "tokeye_lines": 10,
        "any": 10,
    }
    assert ours["recall"]["model"] == pytest.approx(0.3) and ours["recall"]["any"] == 1
    assert theirs == {
        "shot": 176041,
        "t_ms": [424.0, 1504.0],
        "frames": None,
        "listed": None,
        "recall": None,
    }
    assert record["markers"] == asdict(Markers())
    assert record["ntm_sha256"] == sha256_of(paths.label_tables / markers.NTM_TABLE)
    report = file.with_suffix(".md").read_text()
    assert "Recall on 101's owner-absent stretch at 1700-1800 ms" in report
    assert "10 of its 10 scored absent frames are listed (100%)" in report
    assert "model 3 (30%)" in report
    assert "176041's owner-absent stretch at 424-1504 ms" in report
    assert "the shot was not read." in report


def test_a_whole_window_versions_examples_hold_a_shot_past_2_s():
    f1 = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5}
    assert shots.pick_examples(f1) == [1, 3, 5]
    assert shots.pick_examples(f1, long=set()) == [1, 3, 5], "none has one"
    assert shots.pick_examples(f1, long={5}) == [1, 3, 5], "already held"
    assert shots.pick_examples(f1, long={4}) == [1, 4, 5]
    assert shots.pick_examples(f1, long={2, 4}) == [1, 2, 5], "ties: the better"
    assert shots.pick_examples({1: 0.9, 2: 0.8}, long={7}) == [1, 2]
    assert shots.LONG_MS == shots.SCORED_MS == 2000.0
    assert "past 2000 ms" in shots.pick_texts(None).examples
    assert shots.pick_texts().examples == shots.EXAMPLES_RULE
