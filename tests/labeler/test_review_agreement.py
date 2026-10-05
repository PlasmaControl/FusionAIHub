"""Agreement of saved reviews with the suggestions the page opened them on."""

from __future__ import annotations

import json

import pytest

from labeler.events import suggestions
from labeler.events.review import agreement, labels

from . import editor_tree as tree

EVENT = "high_confinement_mode"
SUGGESTED = [(100, 300, 1)]
SAVED = {
    1: SUGGESTED,  # confirmed
    2: [(200, 400, 1)],  # moved: 10 frames each way
    3: [*SUGGESTED, (500, 600, 2)],  # an uncertain stretch the method missed
    4: [(0, 50, 1)],  # saved from another table: not counted
}
OTHER = "high_confinement_mode_format_2026_v1.csv"


@pytest.mark.parametrize(
    "event", ["detachment", "confinement", "minimum_safety_factor"]
)
def test_multiclass_events_refuse_binary_agreement(event):
    with pytest.raises(ValueError, match="multiclass"):
        agreement.require_binary(event)


def _event(p):
    table = suggestions.table_path(p, EVENT, "dalpha_lh", "v1")
    rows = [
        row
        for shot in SAVED
        for row in suggestions.span_rows(shot, (0, 1000), SUGGESTED)
    ]
    suggestions.write_table(table, rows, {"method": "dalpha_lh"})
    event_dir = p.label_tables / EVENT
    labels.write_pointer(event_dir, table, method="dalpha_lh", version="v1")
    for shot, spans in SAVED.items():
        label = labels.normalise((0, 1000), spans)
        labels.save(event_dir, shot, label, source=OTHER if shot == 4 else table.name)
    return event_dir


def test_frames_are_pooled_over_the_shots_saved_from_the_table(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    got = agreement.agreement(_event(p))
    assert (got["shots"], got["unchanged"]) == (3, 1)
    assert got["excluded_saves"] == {OTHER: 1}, "shot 4 is in the table, saved off it"
    cells = {cell: got[cell] for cell in agreement.CELLS}
    assert cells == {"tp": 50, "fp": 10, "fn": 10, "tn": 220, "excluded": 10}
    assert got["precision"] == got["recall"] == 0.8333
    assert got["ready"] is False, "three shots are too few"
    monkeypatch.setattr(agreement, "MIN_SHOTS", 3)
    assert agreement.agreement(p.label_tables / EVENT)["ready"] is True
    monkeypatch.setattr(agreement, "MIN_SCORE", 0.9)
    assert agreement.agreement(p.label_tables / EVENT)["ready"] is False


def test_nothing_saved_is_not_ready(tmp_path, monkeypatch, capsys):
    p = tree.paths(tmp_path)
    tree.use_env(monkeypatch, p)
    assert agreement.main(["--event", EVENT]) == 0
    got = json.loads(capsys.readouterr().out)
    assert (got["shots"], got["precision"], got["ready"]) == (0, None, False)


def _table(p, version, spans):
    table = suggestions.table_path(p, EVENT, "dalpha_lh", version)
    suggestions.write_table(
        table, suggestions.span_rows(1, (0, 1000), spans), {"method": "dalpha_lh"}
    )
    return table


def test_a_save_counts_only_against_the_table_it_was_opened_on(tmp_path):
    p = tree.paths(tmp_path)
    event_dir = p.label_tables / EVENT
    first = _table(p, "v1", [(100, 300, 1)])
    labels.write_pointer(event_dir, first, method="dalpha_lh", version="v1")
    confirmed = labels.normalise((0, 1000), [(100, 300, 1)])
    labels.save(event_dir, 1, confirmed, source=first.name)
    before = agreement.agreement(event_dir)
    assert (before["shots"], before["unchanged"], before["tp"]) == (1, 1, 20)
    second = _table(p, "v2", [(500, 900, 1)])
    labels.write_pointer(event_dir, second, method="dalpha_lh", version="v2")
    moved = agreement.agreement(event_dir)
    assert moved["table"] == str(second.resolve())
    assert (moved["shots"], moved["tp"], moved["fp"], moved["precision"]) == (
        0,
        0,
        0,
        None,
    ), "the save was opened on v1's table, not v2's"
    assert moved["excluded_saves"] == {first.name: 1}
    labels.save(event_dir, 1, confirmed, source=second.name)  # reopened on v2
    again = agreement.agreement(event_dir)
    assert (again["shots"], again["tp"], again["fp"], again["fn"]) == (1, 0, 40, 20)
    assert again["excluded_saves"] == {}
