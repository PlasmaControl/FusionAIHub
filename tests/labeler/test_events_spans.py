"""Span suggestions for the ELM, H-mode and sawtooth editors."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from labeler.events import pipeline, spans, suggestions
from labeler.events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT, UNCERTAIN
from labeler.events.verify import NoDataError

from . import editor_tree as tree

SHOT = 198658


def test_runs_need_min_count_points_and_are_padded():
    times = [100, 120, 140, 900, 1000, 1150, 1500, 1510]
    got = spans.runs(times, max_gap_ms=200, min_count=3, pad_ms=5)
    assert got == [(95.0, 145.0), (895.0, 1155.0)]
    assert spans.runs([], max_gap_ms=200, min_count=3, pad_ms=5) == []


@pytest.mark.parametrize(
    ("marks", "measured", "expected"),
    [
        ([(300, True), (600, False)], [(0, 1000)], [(300, 600, PRESENT)]),
        ([(300, True)], [(0, 1000)], [(300, 1000, PRESENT)]),
        ([(200, False)], [(0, 1000)], [(0, 200, UNCERTAIN)]),
        (
            [(100, True), (400, False), (700, False)],
            [(0, 1000)],
            [(100, 400, PRESENT), (400, 700, UNCERTAIN)],
        ),
        ([(100, True), (200, True), (300, False)], [(0, 1000)], [(100, 300, PRESENT)]),
        (
            [(300, True), (550, True), (800, False)],
            [(0, 500), (600, 1000)],
            [(300, 500, PRESENT), (600, 800, UNCERTAIN)],
        ),
    ],
)
def test_hmode_spans(marks, measured, expected):
    assert spans.hmode_spans(marks, measured) == expected


def test_rows_are_not_observable_where_the_method_could_not_see():
    assert spans.shot_rows(SHOT, (0, 1000), None) == [
        [SHOT, NOT_OBSERVABLE, 0, 1000, ""]
    ]
    found = spans.Found(((100, 300, PRESENT), (850, 950, PRESENT)), ((50.4, 900.2),))
    got = [row[1:4] for row in spans.shot_rows(SHOT, (0, 1000), found)]
    assert got == [
        [NOT_OBSERVABLE, 0, 50],
        [ABSENT, 50, 100],
        [PRESENT, 100, 300],
        [ABSENT, 300, 850],
        [PRESENT, 850, 901],
        [NOT_OBSERVABLE, 901, 1000],
    ]


def test_the_tearing_mode_editor_gets_its_window_all_absent():
    method = spans.METHODS["neoclassical_tearing_mode"]
    assert (method.name, method.inputs) == ("window", ())
    found = method.detect(SHOT, None)
    assert spans.shot_rows(SHOT, (-20, 4000), found) == [[SHOT, ABSENT, -20, 4000, ""]]


def test_the_coverage_resolutions_are_the_pipelines():
    assert spans.ELM_MIN_GAP_S == pipeline.ELM_MIN_GAP_S
    assert spans.SAWTOOTH_MIN_GAP_S == pipeline.SAWTOOTH_MIN_GAP_S
    assert spans.LH_MIN_GAP_S == pipeline.LH_MIN_GAP_S


def _write_synth(p, s):
    tree.write(
        p.corpus_file(SHOT),
        {
            "filterscopes": (s["dalpha_t_s"] * 1000, s["dalpha_y"]),
            "pinj": (s["pinj_t_s"] * 1000, s["pinj_y"][None] * 1000),  # kW -> W
            "ece": (s["ece_t_s"] * 1000, s["ece_y"]),
        },
    )
    # The density as a fetch leaves it: in the raw cache, not the corpus.
    tree.write(
        p.raw_cache / f"{SHOT}_processed.h5",
        {"co2": (s["ne_t_s"] * 1000, s["ne_y"][None])},
    )


def test_hmode_and_sawteeth_from_the_synthetic_shot(tmp_path, synth_shot):
    p = tree.paths(tmp_path)
    _write_synth(p, synth_shot)
    hmode = spans.detect_hmode(SHOT, p)
    [(start, stop, state)] = hmode.spans
    assert state == PRESENT
    assert start == pytest.approx(300, abs=1) and stop == pytest.approx(600, abs=1)
    assert hmode.measured[0][0] <= 100 and hmode.measured[-1][1] >= 700
    saw = spans.detect_sawtooth(SHOT, p)
    crashes = np.asarray(synth_shot["crash_times_s"]) * 1000
    [(start, stop, state)] = saw.spans
    assert start == pytest.approx(crashes[0] - spans.PAD_MS, abs=1.5)
    assert stop == pytest.approx(crashes[-1] + spans.PAD_MS, abs=1.5)


def _elm_train(p):
    t = tree.times(0.0, 1000.0, 10_000)
    y = np.full((8, len(t)), np.nan)
    y[0] = 1.0 + tree.noise(1, t, 0.002)[0]
    for peak in [*np.arange(200.0, 401.0, 20.0), 700.0]:  # eleven ELMs, one alone
        y[0] += 1.5 * np.exp(-0.5 * ((t - peak) / 0.4) ** 2)
    tree.write(p.corpus_file(SHOT), {"filterscopes": (t, y)})


def test_elm_runs_from_a_dalpha_spike_train(tmp_path):
    p = tree.paths(tmp_path)
    _elm_train(p)
    found = spans.detect_elm(SHOT, p)  # no density or beams: no H-mode gate
    [(start, stop, state)] = found.spans
    assert state == PRESENT
    assert start == pytest.approx(195, abs=1) and stop == pytest.approx(405, abs=1)
    assert found.measured == ((0.0, pytest.approx(999.9)),)


def test_elms_the_hmode_method_saw_in_l_mode_are_dropped(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    _elm_train(p)
    hmode = spans.Found(((300.0, 1000.0, PRESENT),), ((0.0, 1000.0),))
    monkeypatch.setattr(spans, "detect_hmode", lambda shot, paths: hmode)
    [(start, stop, _)] = spans.detect_elm(SHOT, p).spans
    assert start == 300.0 and stop == pytest.approx(405, abs=1)


def test_minus():
    assert spans.minus([(0, 100)], [(50, 60), (10, 20)]) == [
        (0, 10),
        (20, 50),
        (60, 100),
    ]
    assert spans.minus([(0, 10), (20, 30)], [(-5, 25)]) == [(25, 30)]
    assert spans.minus([(0, 10)], []) == [(0, 10)]


def test_a_missing_input_is_a_no_data_error(tmp_path):
    p = tree.paths(tmp_path)
    tree.write(p.corpus_file(SHOT), {"ip": ([0.0, 1.0], [[1.0, 1.0]])})
    with pytest.raises(NoDataError, match="corpus or the raw cache"):
        spans.detect_hmode(SHOT, p)


@pytest.fixture
def cohort_env(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.use_env(monkeypatch, p)
    tree.cohort(
        p,
        [
            tree.queue_row(3, 3),
            tree.queue_row(1, 0, blind=True),
            tree.queue_row(2, 1),
            tree.queue_row(4, 2),
            tree.queue_row(5, 4, window=("", "")),
        ],
    )
    failing = {4}

    def detect(shot, paths):
        if shot in failing:
            raise NoDataError(f"shot {shot}: no 'filterscopes'")
        return spans.Found(((100, 200, PRESENT),), ((0.0, 800.0),))

    method = spans.Method("edge_localized_mode", "elm_clock", detect, ("x",))
    monkeypatch.setitem(spans.METHODS, "edge_localized_mode", method)
    return p, failing


def _run(capsys, *args):
    assert spans.main(["--event", "edge_localized_mode", *args]) == 0
    return json.loads(capsys.readouterr().out)


def test_the_queue_is_suggested_in_order_and_merged_by_shot(cohort_env, capsys):
    p, failing = cohort_env
    assert list(spans.queue(p).shot) == [2, 4, 3, 5]
    first = _run(capsys, "--limit", "2")
    assert (first["shots_run"], first["skipped_run"], first["shots"]) == (2, 1, 2)
    path = suggestions.table_path(p, "edge_localized_mode", "elm_clock", "v1")
    table = pd.read_csv(path)
    four = table[table.shot == 4]
    assert four.iloc[:, :4].values.tolist() == [[4, NOT_OBSERVABLE, 0, 1000]]
    assert four.confidence.isna().all()
    assert table[table.shot == 2].category.tolist() == [0, 1, 0, 3]
    everything = _run(capsys)
    assert (everything["shots_run"], everything["shots"]) == (1, 3), "2 and 4 are done"
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["skipped"] == {
        "4": "NoDataError: shot 4: no 'filterscopes'",
        "5": "no catalog window",
    }
    assert meta["method"] == "elm_clock" and meta["rule"]["min_run"] == spans.MIN_RUN
    failing.clear()
    again = _run(capsys, "--shots", "4", "--force")
    assert (again["shots_run"], again["skipped_run"]) == (1, 0)
    table = pd.read_csv(path)
    assert table[table.shot == 4].category.tolist() == [0, 1, 0, 3]
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["skipped"] == {"5": "no catalog window"}
    assert sorted(table.shot.unique()) == [2, 3, 4], "the blind shot is never suggested"


def test_a_blind_shot_cannot_be_named(cohort_env, capsys):
    with pytest.raises(SystemExit):
        spans.main(["--event", "edge_localized_mode", "--shots", "1"])
    assert "not in the cohort (or blind): [1]" in capsys.readouterr().err


def test_with_no_cohort_in_the_tables_the_checkouts_is_read(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    own = tmp_path / "checkout" / "data" / "events"
    monkeypatch.setattr(spans, "DEFAULT_LABEL_TABLES", own)
    assert spans.cohort_path(p) == own / "catalog" / "cohort.csv"
    tree.cohort(p, [tree.queue_row(2, 1)])
    assert spans.cohort_path(p) == p.label_tables / "catalog" / "cohort.csv"
