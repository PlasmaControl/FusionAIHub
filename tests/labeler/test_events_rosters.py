"""The shots.csv review roster: its schema, and what a review does to it."""

from datetime import date

import pandas as pd
import pytest

from labeler.events.databases import DatabaseError
from labeler.events.rosters import (
    ROSTER_COLUMNS,
    read_roster,
    record_review,
    tier_for,
    validate_roster,
    write_roster,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=list(ROSTER_COLUMNS))


def test_tier_counts_reviewers_not_quality():
    assert tier_for([]) == "unverified"
    assert tier_for(["alice"]) == "silver"
    assert tier_for(["alice", "bob"]) == "gold"
    assert tier_for(["alice", "bob", "carol"]) == "gold"


def test_a_valid_roster_round_trips(tmp_path):
    path = tmp_path / "shots.csv"
    frame = _frame([
        [170815, "gold", "alice;bob", "2026-01-01", "retimed onset"],
        [178631, "silver", "alice", "2026-01-02", ""],
        [185945, "unverified", "", "", ""],
    ])
    write_roster(frame, path)
    got = read_roster(path)
    assert list(got.columns) == list(ROSTER_COLUMNS)
    assert got.shot.tolist() == [170815, 178631, 185945]
    assert got.tier.tolist() == ["gold", "silver", "unverified"]


def test_tier_must_agree_with_the_reviewer_count():
    with pytest.raises(DatabaseError, match="tier"):
        validate_roster(_frame([[1, "gold", "alice", "2026-01-01", ""]]))
    with pytest.raises(DatabaseError, match="tier"):
        validate_roster(_frame([[1, "silver", "alice;bob", "2026-01-01", ""]]))
    with pytest.raises(DatabaseError, match="tier"):
        validate_roster(_frame([[1, "unverified", "alice", "2026-01-01", ""]]))
    with pytest.raises(DatabaseError, match="tier"):
        validate_roster(_frame([[1, "silver", "", "", ""]]))


def test_an_unknown_tier_is_rejected():
    with pytest.raises(DatabaseError, match="tier"):
        validate_roster(_frame([[1, "bronze", "alice", "2026-01-01", ""]]))


def test_a_shot_appears_once():
    with pytest.raises(DatabaseError, match="duplicate shot"):
        validate_roster(_frame([
            [1, "silver", "alice", "2026-01-01", ""],
            [1, "silver", "bob", "2026-01-01", ""],
        ]))


def test_a_reviewer_appears_once_per_shot():
    with pytest.raises(DatabaseError, match="duplicate reviewer"):
        validate_roster(_frame([[1, "gold", "alice;alice", "2026-01-01", ""]]))


def test_a_verified_row_carries_a_date():
    with pytest.raises(DatabaseError, match="verified_on"):
        validate_roster(_frame([[1, "silver", "alice", "", ""]]))
    with pytest.raises(DatabaseError, match="verified_on"):
        validate_roster(_frame([[1, "unverified", "", "2026-01-01", ""]]))


def test_two_reviewers_promote_a_shot_to_gold(tmp_path):
    path = tmp_path / "shots.csv"
    write_roster(_frame([[170815, "unverified", "", "", ""]]), path)

    record_review(path, 170815, "alice", on=date(2026, 1, 1))
    row = read_roster(path).iloc[0]
    assert row.tier == "silver"
    assert row.reviewers == "alice"
    assert row.verified_on == "2026-01-01"

    record_review(path, 170815, "bob", on=date(2026, 1, 2))
    row = read_roster(path).iloc[0]
    assert row.tier == "gold"
    assert row.reviewers == "alice;bob"
    assert row.verified_on == "2026-01-02"


def test_the_same_reviewer_twice_redates_without_promoting(tmp_path):
    path = tmp_path / "shots.csv"
    write_roster(_frame([[170815, "silver", "alice", "2026-01-01", ""]]), path)
    record_review(path, 170815, "alice", on=date(2026, 3, 9))
    row = read_roster(path).iloc[0]
    assert row.tier == "silver"
    assert row.reviewers == "alice"
    assert row.verified_on == "2026-03-09"


def test_reviewing_an_unrostered_shot_adds_it(tmp_path):
    path = tmp_path / "shots.csv"
    write_roster(_frame([]), path)
    record_review(path, 199999, "alice", on=date(2026, 1, 1), notes="new")
    got = read_roster(path)
    assert got.shot.tolist() == [199999]
    assert got.tier.tolist() == ["silver"]
    assert got.notes.tolist() == ["new"]


def test_notes_stay_on_one_line(tmp_path):
    path = tmp_path / "shots.csv"
    write_roster(_frame([[1, "unverified", "", "", ""]]), path)
    with pytest.raises(DatabaseError, match="one line"):
        record_review(path, 1, "alice", on=date(2026, 1, 1), notes="two\nlines")


def test_every_category_has_a_valid_roster():
    from labeler.config import Paths
    from labeler.events.rosters import ROSTER_NAME, read_roster

    root = Paths.from_env().label_tables
    categories = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert len(categories) == 16
    for category in categories:
        path = root / category / ROSTER_NAME
        assert path.is_file(), f"{category} has no {ROSTER_NAME}"
        frame = read_roster(path)
        assert len(frame) == 3, f"{category} should ship three example rows"
        assert frame.tier.tolist() == ["gold", "silver", "unverified"]
