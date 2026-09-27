"""Agreement of saved reviews with the suggestions the page opened them on."""

from __future__ import annotations

import json

from labeler.events import suggestions
from labeler.events.review import agreement, labels

from . import editor_tree as tree

EVENT = "high_confinement_mode"
SUGGESTED = [(100, 300, 1)]
SAVED = {
    1: SUGGESTED,  # confirmed
    2: [(200, 400, 1)],  # moved: 10 frames each way
    3: [*SUGGESTED, (500, 600, 2)],  # an uncertain stretch the method missed
    4: [(0, 50, 1)],  # saved before the pointer: not from the table
}


def _event(p):
    table = suggestions.table_path(p, EVENT, "dalpha_lh", "v1")
    rows = [
        row
        for shot in (1, 2, 3)
        for row in suggestions.span_rows(shot, (0, 1000), SUGGESTED)
    ]
    suggestions.write_table(table, rows, {"method": "dalpha_lh"})
    event_dir = p.label_tables / EVENT
    labels.write_pointer(event_dir, table, method="dalpha_lh", version="v1")
    for shot, spans in SAVED.items():
        label = labels.normalise((0, 1000), spans)
        labels.save(event_dir, shot, label, source=table.name)
    return event_dir


def test_frames_are_pooled_over_the_shots_saved_from_the_table(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    got = agreement.agreement(_event(p))
    assert (got["shots"], got["unchanged"]) == (3, 1)
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
