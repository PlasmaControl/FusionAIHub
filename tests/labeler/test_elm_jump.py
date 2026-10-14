"""The elm_jump method: ELMy spans from the page's jump-and-hold onsets."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from labeler.config import Paths
from labeler.events import elm_jump
from labeler.events.panels import edge_localized_mode as panel
from labeler.events.review import labels, silver


def test_each_onset_gets_the_largest_jump_just_after_it():
    x = np.arange(0.0, 10.0, 0.1)
    jump = np.zeros_like(x)
    jump[21], jump[25] = 0.4, 0.9
    jump[61] = np.nan
    sizes = elm_jump.onset_jumps(x, jump, [2.0, 6.0, 20.0], within_ms=1.0)
    assert sizes[0] == pytest.approx(0.9) and sizes[1] == 0.0 and np.isnan(sizes[2])


def test_runs_need_five_onsets_within_fifty_ms_and_are_padded():
    five = [100.0, 140.0, 180.0, 220.0, 260.0]
    assert elm_jump.elmy_runs(five) == [(95.0, 265.0)]
    assert elm_jump.elmy_runs(five[:4]) == []
    assert elm_jump.elmy_runs([100.0, 140.0, 180.0, 220.0, 271.0]) == []


def test_holes_are_the_frames_painted_l_or_qh():
    values = np.array([1, 2, 2, 1, 3, 0, -1, 2], dtype=np.int8)
    assert elm_jump.gate_holes(values, 100) == [
        (110.0, 130.0),
        (140.0, 150.0),
        (170.0, 180.0),
    ]


def test_the_gate_paints_by_precedence_and_never_reads_dalpha_lh(tmp_path, monkeypatch):
    paths = Paths(root=tmp_path, corpus=tmp_path, label_tables=tmp_path / "tables")

    def layer(kind, name, spans, unknown=()):
        # A regime table asserts only its spans: the rest of its window is unknown.
        label = labels.Label((0, 1000), spans)
        return silver.Layer(kind, f"{kind}:{name}", label, unknown)

    found = [
        layer("format", "high_confinement_mode/h", ((0, 500, 1),), ((500, 1000),)),
        layer("source", "dalpha_lh/v1/x", ((0, 1000, 2),)),
        layer(
            "suggest", "lmode_frames/v2/y", ((300, 800, 2),), ((0, 300), (800, 1000))
        ),
    ]
    monkeypatch.setattr(silver, "layers", lambda *args: found)
    sources, holes = elm_jump.gate(paths, 1)
    # The H save wins its 0-500 ms; the L draft fills 500-800; dalpha_lh is out.
    assert holes == [(500.0, 800.0)]
    assert sources == ["format:high_confinement_mode/h", "suggest:lmode_frames/v2/y"]
    monkeypatch.setattr(silver, "layers", lambda *args: found[1:2])
    assert elm_jump.gate(paths, 1) == ([], [])


def test_the_side_comes_from_the_strike_sum_chords():
    upper = SimpleNamespace(
        title=panel.DALPHA_STRIKE_TITLE, legend="", caveat="FS02UP+FS04UP summed (x)"
    )
    assert elm_jump.side_of(upper) == ("upper", ["FS02UP", "FS04UP"])
    lower = SimpleNamespace(
        title=panel.DALPHA_STRIKE_TITLE, legend="", caveat="FS04+FS05 summed"
    )
    assert elm_jump.side_of(lower) == ("lower", ["FS04", "FS05"])
    fallback = SimpleNamespace(title="D-alpha FS01", legend="FS01", caveat="no")
    assert elm_jump.side_of(fallback) == ("fallback", ["FS01"])


def test_a_table_drafted_under_other_rules_is_not_run_into(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("shot,category,t_start,t_end,confidence\n")
    path.with_suffix(".meta.json").write_text(
        json.dumps({"rule": {**elm_jump.RULE, "min_count": 3}})
    )
    with pytest.raises(elm_jump.RuleChanged):
        elm_jump.check_rule(path)
    path.with_suffix(".meta.json").write_text(json.dumps({"rule": elm_jump.RULE}))
    elm_jump.check_rule(path)


def test_the_gate_leaves_out_the_rows_a_table_took_from_dalpha_lh(
    tmp_path, monkeypatch
):
    # As regimes v1: curated shots open on curated intervals, any other on the
    # D-alpha L-H detector's rows, which the gate never reads.
    paths = Paths(root=tmp_path, corpus=tmp_path, label_tables=tmp_path / "tables")
    regimes = tmp_path / "suggestions/regimes/v1/confinement_suggest_regimes_v1.csv"
    regimes.parent.mkdir(parents=True)
    regimes.write_text(
        "shot,category,t_start,t_end,confidence\n1,1,0,1000,\n2,1,0,1000,\n"
    )
    curated = tmp_path / "merged_intervals.csv"
    curated.write_text("shot,t_start,t_end\n1,0,1000\n")
    detector = (
        tmp_path / "suggestions/dalpha_lh/v1/confinement_suggest_dalpha_lh_v1.csv"
    )
    sources = {"curated": {"table": str(curated)}, "detector": {"table": str(detector)}}
    regimes.with_suffix(".meta.json").write_text(json.dumps({"sources": sources}))
    assert elm_jump.dalpha_lh_shots(regimes) == {2}

    name = f"source:{regimes.with_suffix('').as_posix()}"
    found = [
        silver.Layer("source", name, labels.Label((0, 1000), ((0, 1000, 1),))),
        silver.Layer(
            "suggest", "suggest:lmode/v1/l", labels.Label((0, 1000), ((0, 1000, 2),))
        ),
    ]
    monkeypatch.setattr(silver, "layers", lambda *args: found)
    assert elm_jump.gate(paths, 1) == ([name], [])
    assert elm_jump.gate(paths, 2) == (["suggest:lmode/v1/l"], [(0.0, 1000.0)])


def test_run_writes_the_onsets_only_after_the_table_validated(tmp_path, monkeypatch):
    paths = Paths(root=tmp_path, corpus=tmp_path, label_tables=tmp_path / "tables")
    monkeypatch.setattr(
        elm_jump, "windows", lambda *args: ({1: (0.0, 1000.0)}, {1: "ip"}, {})
    )
    onsets = pd.DataFrame(
        {"shot": [1], "t_ms": [100.0], "jump": [0.5], "in_span": [False]}
    )
    for rows, fails in (([[1, 0, 1000, 0, ""]], True), ([[1, 0, 0, 1000, ""]], False)):
        shot = elm_jump.Shot(rows, onsets)
        monkeypatch.setattr(elm_jump, "draft", lambda *args, shot=shot: shot)
        if fails:  # t_end before t_start: the table is refused, and nothing written
            with pytest.raises(ValueError):
                elm_jump.run(paths, shots=[1], version="v9")
            assert not elm_jump.onsets_path(paths, "v9").exists()
            assert not elm_jump.table_path(paths, "v9").exists()
        else:
            assert elm_jump.run(paths, shots=[1], version="v9")["rows"] == 1
            assert pd.read_csv(elm_jump.onsets_path(paths, "v9")).t_ms.tolist() == [
                100.0
            ]
