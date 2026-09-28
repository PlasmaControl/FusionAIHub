"""The round-two editors' rosters: the cohort's queue, and the page's pointer."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from labeler.events import rosters, spans, suggestions
from labeler.events.review import cohort_rosters, labels

from . import editor_tree as tree

EVENT = "high_confinement_mode"
HEADER = "shot,tier,holdout,reviewers,verified_on,notes\n"
EXISTING = (
    HEADER + "1,gold,false,alice;bob,2026-01-01,EXAMPLE - replace with a real shot\n"
    "3,unverified,false,,,EXAMPLE - replace with a real shot\n"
    "178640,gold,true,,,\n"
    "178641,gold,false,,,\n"
    "201003,silver,false,nc,2026-09-20,kept\n"
)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    p = tree.paths(tmp_path)
    tree.use_env(monkeypatch, p)
    tree.cohort(
        p,
        [
            tree.queue_row(201004, 3),
            tree.queue_row(201002, 0, blind=True),
            tree.queue_row(201001, 1),
            tree.queue_row(201003, 2),
        ],
    )
    path = rosters.roster_path(EVENT, root=p.label_tables)
    path.parent.mkdir(parents=True)
    path.write_text(EXISTING)
    return p, path


def _run(capsys, *args):
    assert cohort_rosters.main(["--event", EVENT, *args]) == 0
    return json.loads(capsys.readouterr().out)


def test_the_roster_is_the_queue_then_the_rest_without_examples(cohort, capsys):
    _, path = cohort
    summary = _run(capsys)
    assert (summary["shots"], summary["queue"], summary["kept"]) == (5, 3, 2)
    assert path.read_text() == (
        HEADER + "201001,unverified,false,,,\n"
        "201003,silver,false,nc,2026-09-20,kept\n"
        "201004,unverified,false,,,\n"
        "178640,gold,true,,,\n"
        "178641,gold,false,,,\n"
    )
    before = path.read_bytes()
    _run(capsys)
    assert path.read_bytes() == before, "a rerun writes the same file"
    queue = labels.queue(path.parent, rosters.read_roster(path))
    order = [row["shot"] for row in queue["shots"]]
    assert order == [201001, 201003, 201004, 178640, 178641], "the page's order"


def test_write_roster_sorts_unless_told_to_keep_the_order(tmp_path):
    frame = cohort_rosters.cohort_roster(
        [3, 1, 2], pd.DataFrame(columns=HEADER[:-1].split(","))
    )
    rosters.write_roster(frame, tmp_path / "sorted.csv")
    rosters.write_roster(frame, tmp_path / "kept.csv", keep_order=True)
    assert list(rosters.read_roster(tmp_path / "sorted.csv").shot) == [1, 2, 3]
    assert list(rosters.read_roster(tmp_path / "kept.csv").shot) == [3, 1, 2]


def test_point_opens_the_page_on_the_spans_table(cohort, capsys):
    p, path = cohort
    with pytest.raises(SystemExit):
        cohort_rosters.main(["--event", EVENT, "--point"])
    assert "run `python -m labeler.events.spans" in capsys.readouterr().err
    assert path.read_text() == EXISTING, "nothing is written without the table"
    table = suggestions.table_path(p, EVENT, "dalpha_lh", spans.VERSION)
    suggestions.write_table(table, [[201001, 1, 0, 1000, ""]], {"method": "dalpha_lh"})
    summary = _run(capsys, "--point")
    assert summary["pointer"]["method"] == "dalpha_lh"
    assert summary["pointer"]["version"] == spans.VERSION
    assert labels.source_path(path.parent) == table.resolve()
    [only] = labels.read_source(path.parent).values()
    assert only.intervals == ((0, 1000, 1),)
