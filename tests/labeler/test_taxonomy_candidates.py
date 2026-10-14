"""The taxonomy's candidate tables (`labeler.events.candidates`): the amplitude
threshold rule, the crash runs, the q rule and the ECE boxes, and the one CLI.

The rules run on synthetic inputs; a few end-to-end runs use the synthetic
Mirnov and ECE shots of `test_panels_mhd`. Nothing is fetched.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from labeler.events import candidates, equilibrium, spans, suggestions
from labeler.events.raw import NO_FETCH_ENV
from labeler.features.store import FeatureArray, write_features

from . import editor_tree as tree
from .test_panels_mhd import _mirnov_shot, _sawtooth_ece

SHOT = 201250


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    """No archive, no fdp: an input a test did not write is missing."""
    nothing = lambda *args, **kwargs: ({}, {})
    monkeypatch.setattr(equilibrium.resolve_archive, "resolve", nothing)
    monkeypatch.setattr(equilibrium.resolve_fdp, "resolve", nothing)
    tree.no_fetch(monkeypatch)


def _read(path) -> list[list]:
    return pd.read_csv(path, keep_default_na=False).values.tolist()


# --- the threshold rule --------------------------------------------------------


def test_a_run_needs_min_ms_above_the_threshold_and_a_gap_of_at_most_max_gap_to_join():
    t = np.arange(0.0, 400.0)  # 1 ms columns, each covering half a ms either side
    level = np.zeros(len(t))
    level[50:80] = 15  # 30 ms: kept
    level[110:120] = 15  # 10 ms, 30 ms from the last: too short
    level[150:165] = level[180:195] = 15  # 15 ms each, 15 ms apart: one 45 ms run
    level[230:255] = level[280:305] = 15  # 25 ms each, 25 ms apart: two runs
    assert candidates.threshold_runs(t, level) == [
        (49.5, 79.5),
        (149.5, 194.5),
        (229.5, 254.5),
        (279.5, 304.5),
    ]
    assert candidates.threshold_runs(t, level, threshold=16) == []
    assert candidates.threshold_runs(t, level, min_ms=40) == [(149.5, 194.5)]


def test_a_wider_gap_joins_more_and_a_gap_in_the_data_is_not_above():
    t = np.arange(0.0, 400.0)
    level = np.zeros(len(t))
    level[230:255] = level[280:305] = 15
    assert candidates.threshold_runs(t, level, max_gap_ms=30) == [(229.5, 304.5)]
    level[280:305] = 0
    level[240] = np.nan  # a missing column is not above: a 1 ms gap inside the run
    assert candidates.threshold_runs(t, level) == [(229.5, 254.5)]
    assert candidates.threshold_runs(t, level, max_gap_ms=0.5) == []  # two short runs
    assert candidates.threshold_runs(t, np.full(len(t), np.nan)) == []
    assert candidates.threshold_runs([0.0], [20.0]) == []


def test_the_defaults_are_the_documented_rule():
    assert (candidates.THRESHOLD_DB, candidates.MIN_MS, candidates.MAX_GAP_MS) == (
        12.0,
        20.0,
        20.0,
    )
    assert (candidates.TOO_MUCH, candidates.TOO_LITTLE) == (0.8, 0.01)
    assert (candidates.RETUNE_UP, candidates.RETUNE_DOWN) == (1.5, 0.5)
    assert candidates.CRASH_RUN == {
        "min_count": spans.MIN_RUN,
        "max_gap_ms": spans.SAWTOOTH_MAX_GAP_MS,
        "pad_ms": spans.PAD_MS,
    }


def test_crashes_make_a_run_only_in_threes_no_more_than_the_sawtooth_gap_apart():
    gap = spans.SAWTOOTH_MAX_GAP_MS
    runs = candidates.crash_spans([100, 100 + gap, 100 + 2 * gap, 10_000, 10_000 + gap])
    assert runs == [(95.0, 100 + 2 * gap + 5.0)]  # the pair at 10 000 is not a run
    assert candidates.crash_spans([100, 100 + gap + 1, 100 + 2 * gap + 2]) == []
    assert candidates.crash_spans([]) == []


def test_flagged_fraction_is_the_present_share_of_the_tiled_time():
    rows = [[1, 0, 0, 100, ""], [1, 1, 100, 160, ""], [1, 2, 160, 200, ""]]
    assert candidates.flagged_fraction(rows) == pytest.approx(60 / 200)
    assert candidates.flagged_fraction([]) == 0.0


# --- the n=1..3 amplitude rows and their retune --------------------------------


def _levels(*, crashes=None, loud=(100.0, 200.0), db=15.0, quiet=5.0, shot_window=(0.0, 1000.0)):
    """Amplitudes at 2 ms columns: n=1 at `db` inside `loud`, `quiet` elsewhere,
    n=2 and n=3 quiet throughout."""
    t = np.arange(0.0, 1000.0, 2.0)
    amplitude = np.full((3, len(t)), quiet)
    amplitude[0, (t >= loud[0]) & (t <= loud[1])] = db
    return candidates.Levels(t, amplitude, shot_window, "catalog", crashes)


def test_an_n_is_present_where_its_amplitude_holds_and_n1_adds_the_crash_runs():
    got = _levels(crashes=np.array([300.0, 350.0, 400.0]))
    rows = candidates.n_amplitude_rows(7, got, 1, 12.0, with_crashes=True)
    assert rows == [
        [7, 0, 0, 99, ""],
        [7, 1, 99, 201, ""],
        [7, 0, 201, 295, ""],
        [7, 1, 295, 405, ""],
        [7, 0, 405, 1000, ""],
    ]
    # n=2 is quiet and does not take the crashes; nor does a run of two.
    quiet = candidates.n_amplitude_rows(7, got, 2, 12.0, with_crashes=False)
    assert quiet == [[7, 0, 0, 1000, ""]]
    amplitude_only = [[7, 0, 0, 99, ""], [7, 1, 99, 201, ""], [7, 0, 201, 1000, ""]]
    two = _levels(crashes=np.array([300.0, 350.0]))
    assert candidates.n_amplitude_rows(7, two, 1, 12.0, with_crashes=True) == amplitude_only
    # A shot whose ECE could not be read has no crashes (its meta says why).
    unread = _levels(crashes=None)
    assert candidates.n_amplitude_rows(7, unread, 1, 12.0, with_crashes=True) == amplitude_only


def _fake_levels(monkeypatch, by_shot):
    monkeypatch.setattr(
        candidates, "levels", lambda shot, paths, *, crashes: by_shot[shot]
    )


def test_a_table_over_80_percent_flagged_is_retuned_once_upward(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    _fake_levels(monkeypatch, {1: _levels(loud=(0.0, 998.0), db=14.0)})
    loud = candidates.run_n_amplitude("mhd_n1", [1], p)
    assert loud.meta["flagged_fraction_first"] > candidates.TOO_MUCH
    assert loud.meta["retuned"] is True
    assert loud.meta["threshold_db"] == pytest.approx(18.0)  # 12 x 1.5
    assert loud.meta["first_threshold_db"] == 12.0
    assert loud.meta["flagged_fraction"] == 0.0  # at 18 dB nothing is above
    assert loud.rows == [[1, 0, 0, 1000, ""]]


def test_a_table_under_1_percent_flagged_is_retuned_once_downward(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    _fake_levels(monkeypatch, {1: _levels(quiet=7.0)})
    quiet = candidates.run_n_amplitude("mhd_n2", [1], p)
    assert quiet.meta["retuned"] is True
    assert quiet.meta["threshold_db"] == pytest.approx(6.0)  # 12 x 0.5
    assert quiet.meta["flagged_fraction_first"] == 0.0
    assert quiet.meta["flagged_fraction"] > 0.99  # 7 dB is above the lowered 6 dB
    assert "once" in quiet.meta["retune_rule"]


def test_an_explicit_threshold_is_never_retuned_and_a_shot_that_failed_is_skipped(
    tmp_path, monkeypatch
):
    p = tree.paths(tmp_path)

    def levels(shot, paths, *, crashes):
        if shot == 2:
            raise ValueError("no Mirnov record")
        return _levels(db=5.0)

    monkeypatch.setattr(candidates, "levels", levels)
    result = candidates.run_n_amplitude("mhd_n3", [1, 2], p, threshold=12.0)
    assert result.meta["retuned"] is False and result.meta["threshold_db"] == 12.0
    assert result.skipped == {"2": "ValueError: no Mirnov record"}
    assert sorted(result.per_shot) == ["1"]
    assert {row[0] for row in result.rows} == {1}


def test_a_shots_null_raises_its_threshold_and_the_meta_records_it_per_shot(
    tmp_path, monkeypatch
):
    p = tree.paths(tmp_path)
    plain = _levels(loud=(100.0, 200.0), db=15.0)
    nulled = _levels(loud=(100.0, 200.0), db=15.0)
    nulled.null_db, nulled.gate = np.array([20.0, np.nan, np.nan]), 0.97
    assert candidates.shot_threshold(plain, 1, 12.0) == 12.0
    assert candidates.shot_threshold(nulled, 1, 12.0) == 20.0
    assert candidates.shot_threshold(nulled, 2, 12.0) == 12.0  # no null for n = 2
    # 15 dB is a mode where there is no null, noise where the null reaches 20 dB.
    assert candidates.n_amplitude_rows(7, plain, 1, 12.0, with_crashes=False) == [
        [7, 0, 0, 99, ""], [7, 1, 99, 201, ""], [7, 0, 201, 1000, ""]
    ]
    assert candidates.n_amplitude_rows(7, nulled, 1, 12.0, with_crashes=False) == [
        [7, 0, 0, 1000, ""]
    ]
    _fake_levels(monkeypatch, {1: plain, 2: nulled})
    result = candidates.run_n_amplitude("mhd_n2", [1, 2], p, threshold=12.0)
    assert result.per_shot["1"]["threshold_db"] == 12.0
    assert "null_db" not in result.per_shot["1"] and "fit_gate" not in result.per_shot["1"]
    n1 = candidates.run_n_amplitude("mhd_n1", [1, 2], p, threshold=12.0)
    assert n1.per_shot["2"]["threshold_db"] == 20.0
    assert (n1.per_shot["2"]["null_db"], n1.per_shot["2"]["fit_gate"]) == (20.0, 0.97)
    assert "null-calibrated per shot" in n1.meta["rule"]["amplitude"]
    assert "per_shot[shot].threshold_db" in n1.meta["rule"]["threshold_per_shot"]


def test_mhd_n1_records_how_many_crashes_each_shot_has_or_why_it_has_none(
    tmp_path, monkeypatch
):
    p = tree.paths(tmp_path)
    got = _levels(crashes=np.array([300.0, 350.0, 400.0]))
    broken = _levels(crashes=None)
    broken.crash_error = "NoDataError: no ece"
    _fake_levels(monkeypatch, {1: got, 2: broken})
    result = candidates.run_n_amplitude("mhd_n1", [1, 2], p, threshold=12.0)
    assert result.per_shot["1"]["crashes"] == 3
    assert "crashes" not in result.per_shot["2"]
    assert result.per_shot["2"]["crash_error"] == "NoDataError: no ece"
    assert result.meta["rule"]["unions"].startswith("the 1/1 crash runs")


def test_n_amplitude_finds_the_line_of_a_synthetic_mirnov_shot(tmp_path):
    p = tree.paths(tmp_path)
    _mirnov_shot(p, SHOT, 2)  # n = 2 from 70 to 130 ms, cohort window 50-150 ms
    n2 = candidates.run("n_amplitude", "mhd_n2", p, shots=[SHOT], threshold=12.0)
    table = pd.read_csv(n2["table"], keep_default_na=False)
    present = table[table.category == 1]
    assert len(present) == 1
    assert 58 <= present.t_start.iloc[0] <= 78 and 122 <= present.t_end.iloc[0] <= 142
    assert (table.t_start.min(), table.t_end.max()) == (50, 150)  # the cohort window
    n1 = candidates.run("n_amplitude", "mhd_n1", p, shots=[SHOT], threshold=12.0)
    assert (pd.read_csv(n1["table"]).category == 1).sum() == 0  # n=1 is quiet
    assert "crash_error" in n1["per_shot"][str(SHOT)]  # no ECE: n=1 notes it


# --- the crash method ------------------------------------------------------------


def test_the_crash_method_runs_the_sawtooth_rule_over_the_ece_crash_times(tmp_path):
    p = tree.paths(tmp_path)
    crashes = _sawtooth_ece(p, SHOT)
    meta = candidates.run("crash", "mhd_n1.m1", p, shots=[SHOT, SHOT + 1])
    rows = _read(meta["table"])
    assert [r[1] for r in rows] == [0, 1, 0]
    assert abs(rows[1][2] - (crashes[0] - 5)) <= 2 and abs(rows[1][3] - (crashes[-1] + 5)) <= 2
    assert (rows[0][2], rows[2][3]) == (100, 600)  # the cohort window, tiled
    assert meta["per_shot"][str(SHOT)] == {"crashes": len(crashes), "runs": 1}
    assert str(SHOT + 1) in meta["skipped"]  # no window and no ECE
    assert meta["rule"]["min_count"] == spans.MIN_RUN


# --- the q rule -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "q_min", "m"),
    [
        (1, 1.6, 2),
        (1, 2.4, 3),
        (1, 3.5, 4),
        (1, 0.8, 2),  # m > n even where q_min is under 1
        (1, 2.0, 2),  # q_min exactly on a rational surface: that surface
        (2, 1.4, 3),
        (2, 1.5, 3),
        (3, 1.2, 4),
        (3, 1.0, 4),
    ],
)
def test_m_is_the_smallest_integer_above_n_with_m_over_n_at_least_q_min(n, q_min, m):
    assert candidates.choose_m(n, q_min) == m


@pytest.mark.parametrize(
    ("n", "q_min", "event"),
    [
        (1, 1.6, "mhd_n1.m2"),
        (1, 2.4, "mhd_n1.m3"),
        (2, 1.4, "mhd_n2.m3"),
        (3, 1.2, "mhd_n3.m4"),
        (1, 3.5, None),  # (4, 1) is not one of the taxonomy's children
        (2, 2.2, None),  # (5, 2)
        (3, 1.6, None),  # (5, 3)
    ],
)
def test_a_tearing_mode_belongs_to_the_child_its_m_and_n_name_or_to_none(n, q_min, event):
    assert candidates.child_event(n, q_min) == event
    assert set(candidates.Q_RULE_EVENTS.values()) == candidates.ALLOWED


def _tm_table(p):
    """The tearing-mode interval table of two shots (and the q_min of the first):
    q_min 1.4 before 3 s and 3.2 after it. Shot 101 has a whole interval but no q."""
    rows = [
        (100, 0, 0, 500, ""),
        (100, 1, 900, 900, '{"iscrowd": 0, "n": 1}'),  # a point marker, not an interval
        (100, 1, 1000, 2000, '{"iscrowd": 1, "n": 1}'),  # q 1.4: (2, 1)
        (100, 1, 1200, 1800, '{"iscrowd": 1, "n": 2}'),  # q 1.4: (3, 2)
        (100, 1, 3500, 4500, '{"iscrowd": 1, "n": 1}'),  # q 3.2: (4, 1), not a child
        (100, 3, 6000, 7000, ""),
        (101, 0, 0, 1000, ""),
        (101, 1, 200, 800, '{"iscrowd": 1, "n": 1}'),  # no q_min
    ]
    path = p.label_tables / candidates.TM_INTERVALS
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        rows, columns=["shot", "category", "t_start", "t_end", "attrs"]
    ).assign(confidence="").to_csv(path, index=False)
    t_s = np.arange(0.0, 8.0, 0.01)
    q = np.where(t_s < 3.0, 1.4, 3.2)
    write_features(
        p.features_file(100), 100, {"qmin": FeatureArray(t_s, q[None], {})}, {}
    )


def test_the_q_rule_keeps_an_interval_for_the_child_its_median_q_min_names(tmp_path):
    p = tree.paths(tmp_path)
    _tm_table(p)
    frame = candidates.tm_intervals(p)
    rows, info = candidates.q_rule_rows(100, frame, "mhd_n1.m2", p)
    assert rows == [
        [100, 0, 0, 1000, ""],
        [100, 1, 1000, 2000, ""],
        [100, 0, 2000, 6000, ""],
        [100, 3, 6000, 7000, ""],
    ]
    first, second, third = info["intervals"]
    assert (first["n"], first["m"], first["event"]) == (1, 2, "mhd_n1.m2")
    assert first["q_min"] == pytest.approx(1.4)
    assert (second["n"], second["m"], second["event"]) == (2, 3, "mhd_n2.m3")
    assert third["dropped"] == "(4, 1) is not in the taxonomy" and "event" not in third
    other, _ = candidates.q_rule_rows(100, frame, "mhd_n2.m3", p)
    assert other == [
        [100, 0, 0, 1200, ""],
        [100, 1, 1200, 1800, ""],
        [100, 0, 1800, 6000, ""],
        [100, 3, 6000, 7000, ""],
    ]
    none, no_q = candidates.q_rule_rows(101, frame, "mhd_n1.m2", p)
    assert none == [[101, 0, 0, 1000, ""]]
    assert no_q["intervals"][0]["dropped"] == "no q_min"


def test_the_q_rule_run_counts_what_it_kept_and_dropped_and_names_the_skipped(tmp_path):
    p = tree.paths(tmp_path)
    _tm_table(p)
    meta = candidates.run("q_rule", "mhd_n1.m2", p, shots=[100, 101, 102])
    assert meta["intervals_kept"] == 1
    assert meta["intervals_dropped"] == {"no q_min": 1, "not in the taxonomy": 1}
    assert meta["skipped"] == {"102": "not in the tearing-mode interval table"}
    assert meta["table"].endswith("suggestions/q_rule/v1/mhd_n1.m2_suggest_q_rule_v1.csv")
    assert meta["shots"] == 2 and meta["rule"]["event"] == "(m, n) = (2, 1)"
    assert sorted(map(tuple, meta["rule"]["kept_only_if"])) == sorted(candidates.ALLOWED)


# --- the ECE boxes ---------------------------------------------------------------


def _boxes(p, shot, lines):
    folder = p.label_tables / candidates.ECE_LABELS
    folder.mkdir(parents=True, exist_ok=True)
    text = "t1, t2, f1, f2, channel, type\n" + "\n".join(lines) + "\n"
    (folder / f"labels_{shot}.txt").write_text(text)


def test_ece_boxes_are_present_where_a_box_of_the_type_lies_and_uncertain_elsewhere(
    tmp_path,
):
    p = tree.paths(tmp_path)
    # Times in seconds, all exact in binary: 250-875 ms is where any box lies.
    _boxes(
        p,
        SHOT,
        [
            "0.25, 0.75, 60, 115, 2, TAE",
            "0.5, 0.875, 55, 125, 3, TAE",
            "0.375, 0.5, 55, 125, 3, RSAE",
        ],
    )
    _boxes(p, SHOT + 1, ["0.25, 0.5, 60, 115, 2, TAE"])
    tae = candidates.run("ece_boxes", "alfven_eigenmode.tae", p)
    assert _read(tae["table"]) == [
        [SHOT, 1, 250, 875, ""],
        [SHOT + 1, 1, 250, 500, ""],
    ]
    rsae = candidates.run("ece_boxes", "alfven_eigenmode.rsae", p)
    assert _read(rsae["table"]) == [
        [SHOT, 2, 250, 375, ""],
        [SHOT, 1, 375, 500, ""],
        [SHOT, 2, 500, 875, ""],
    ]
    assert rsae["skipped"] == {str(SHOT + 1): "no RSAE box"}
    assert "unknown, not absent" in rsae["rule"]["uncertain"]
    assert tae["per_shot"][str(SHOT)]["boxes"] == 2
    assert tae["per_shot"][str(SHOT)]["window_ms"] == [250.0, 875.0]


def test_ece_box_rows_never_call_a_time_absent():
    boxes = pd.DataFrame(
        {"t1": [0.25, 0.5], "t2": [0.375, 0.75], "type": ["TAE", "RSAE"]}
    )
    rows = candidates.ece_box_rows(5, boxes, "TAE")
    assert [r[1] for r in rows] == [1, 2]
    assert 0 not in {r[1] for r in rows}


# --- the table, its merge and the CLI --------------------------------------------


def test_a_table_sits_under_suggestions_by_method_version_and_event(tmp_path):
    p = tree.paths(tmp_path)
    path = candidates.table_path(p, "mhd_n1.m3", "q_rule", "v1")
    assert path == p.root / "suggestions" / "q_rule" / "v1" / "mhd_n1.m3_suggest_q_rule_v1.csv"
    assert path == suggestions.table_path(p, "mhd_n1.m3", "q_rule", "v1")


def test_a_run_replaces_the_rows_and_notes_of_the_shots_it_ran_and_keeps_the_rest(tmp_path):
    p = tree.paths(tmp_path)

    def result(shots, *, skipped=None, state=1):
        return candidates.Result(
            rows=[[s, state, 0, 100, ""] for s in shots],
            skipped=skipped or {},
            per_shot={str(s): {"state": state} for s in shots},
            meta={"rule": {"x": 1}},
        )

    first = result([1, 2], skipped={"3": "no ECE"}, state=1)
    candidates.write(p, "mhd_n1.m1", "crash", "v1", first, shots_run=[1, 2, 3])
    second = result([2, 3], state=0)
    meta = candidates.write(p, "mhd_n1.m1", "crash", "v1", second, shots_run=[2, 3])
    path = candidates.table_path(p, "mhd_n1.m1", "crash", "v1")
    assert _read(path) == [[1, 1, 0, 100, ""], [2, 0, 0, 100, ""], [3, 0, 0, 100, ""]]
    assert meta["skipped"] == {}  # shot 3 was rerun and read
    assert meta["per_shot"] == {
        "1": {"state": 1},
        "2": {"state": 0},
        "3": {"state": 0},
    }
    on_disk = json.loads(path.with_suffix(".meta.json").read_text())
    assert on_disk["event"] == "mhd_n1.m1" and on_disk["method"] == "crash"
    assert on_disk["categories"]["3"] == "not_observable"
    assert on_disk["rows"] == 3 and on_disk["shots"] == 3 and on_disk["git_sha"]


def test_a_method_and_event_that_do_not_go_together_are_refused(tmp_path):
    p = tree.paths(tmp_path)
    with pytest.raises(ValueError, match="q_rule is for"):
        candidates.run("q_rule", "mhd_n1.m1", p)
    assert set(candidates.EVENTS) == {
        "n_amplitude",
        "crash",
        "q_rule",
        "ece_boxes",
        "mhd_modes",
    }
    assert candidates.EVENTS["n_amplitude"] == ("mhd_n1", "mhd_n2", "mhd_n3")


def test_the_roster_is_the_parent_events_shots_file(tmp_path):
    p = tree.paths(tmp_path)
    (p.label_tables / "mhd_n1").mkdir(parents=True)
    pd.DataFrame({"shot": [30, 10, 20]}).to_csv(p.label_tables / "mhd_n1" / "shots.csv", index=False)
    assert candidates.roster(p, "mhd_n1.m2") == [10, 20, 30]


def test_the_cli_writes_one_table_and_fetches_nothing_unless_asked(
    tmp_path, monkeypatch, capsys
):
    p = tree.paths(tmp_path)
    tree.use_env(monkeypatch, p)
    monkeypatch.setenv(NO_FETCH_ENV, "1")  # what the CLI sets itself, restored after
    _tm_table(p)
    (p.label_tables / "mhd_n1").mkdir(parents=True)
    pd.DataFrame({"shot": [100, 101]}).to_csv(p.label_tables / "mhd_n1" / "shots.csv", index=False)
    assert candidates.main(["--method", "q_rule", "--event", "mhd_n1.m3", "--version", "v2"]) == 0
    out = capsys.readouterr().out
    assert '"event": "mhd_n1.m3"' in out and "skipped: 0" in out
    assert (p.root / "suggestions/q_rule/v2/mhd_n1.m3_suggest_q_rule_v2.csv").is_file()
    candidates.main(["--method", "q_rule", "--event", "mhd_n1.m2", "--limit", "1"])
    table = _read(p.root / "suggestions/q_rule/v1/mhd_n1.m2_suggest_q_rule_v1.csv")
    assert {row[0] for row in table} == {100}  # --limit 1: the first of the roster


@pytest.mark.parametrize(
    "argv",
    [
        ["--method", "q_rule", "--event", "mhd_n1.m1"],
        ["--method", "crash", "--event", "mhd_n1.m1", "--version", "1"],
        ["--method", "guess", "--event", "mhd_n1"],
    ],
)
def test_the_cli_refuses_a_wrong_event_version_or_method(argv):
    with pytest.raises(SystemExit) as exit_:
        candidates.main(argv)
    assert exit_.value.code == 2
