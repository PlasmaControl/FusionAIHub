"""The links table: verified links only, one row per (shot, paper)."""

from __future__ import annotations

import pandas as pd
import pytest

from labeler.events.databases import DatabaseError
from labeler.literature.papers import (
    PAPER_COLUMNS,
    Record,
    links,
    papers_frame,
    read_papers,
    validate_papers,
    write_papers,
)

PAPER = Record("osti", "2426954", "10.1063/5.0157020", "A title", 2023, "Phys. Plasmas")


def test_one_row_per_shot_and_exact_beats_range():
    text = "shots 189600-189603 ... later, discharge 189602 again, and shot 189601."
    rows = links(PAPER, text, {189601, 189602, 189603})
    assert [(r["shot"], r["match_type"]) for r in rows] == [
        (189601, "exact"),
        (189602, "exact"),
        (189603, "exact"),
    ]
    assert rows[0]["verified_by"] == "auto"
    assert rows[0]["record_id"] == "2426954" and rows[0]["year"] == 2023
    assert "shot 189601" in rows[0]["context"]


def test_a_shot_outside_the_set_is_not_linked():
    assert links(PAPER, "shot 189631", {189632}) == []


def test_the_table_round_trips_sorted(tmp_path):
    rows = links(PAPER, "shot 189700 and shot 189631", {189631, 189700})
    other = Record("osti", "13472", title="Report")  # no year, no doi
    rows += links(other, "DIII-D 189631", {189631})
    path = tmp_path / "papers.csv"
    write_papers(pd.DataFrame(rows, columns=list(PAPER_COLUMNS)), path)
    back = read_papers(path)
    assert back[["shot", "record_id"]].values.tolist() == [
        [189631, "13472"],
        [189631, "2426954"],
        [189700, "2426954"],
    ]
    assert back["year"].isna().tolist() == [True, False, False]
    assert back["doi"].tolist() == ["", "10.1063/5.0157020", "10.1063/5.0157020"]


@pytest.mark.parametrize(
    "change, problem",
    [
        ({"source": "scholar"}, "source must be one of"),
        ({"match_type": "fuzzy"}, "match_type must be one of"),
        ({"verified_by": "model"}, "verified_by must be one of"),
        ({"record_id": " "}, "names its record"),
        ({"shot": -3}, "nonnegative integer"),
    ],
)
def test_malformed_links_are_refused(change, problem):
    row = links(PAPER, "shot 189631", {189631})[0] | change
    with pytest.raises(DatabaseError, match=problem):
        validate_papers(pd.DataFrame([row], columns=list(PAPER_COLUMNS)))


def test_a_link_appears_once():
    row = links(PAPER, "shot 189631", {189631})[0]
    with pytest.raises(DatabaseError, match="appears twice"):
        papers_frame([row, row])
