"""The shots.csv review roster: its schema, and what a review does to it."""

from datetime import date

import pandas as pd
import pytest

from labeler.events.databases import DatabaseError
from labeler.events.rosters import (
    ROSTER_COLUMNS,
    read_roster,
    record_review,
    validate_roster,
    write_roster,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=list(ROSTER_COLUMNS))


def test_a_valid_roster_round_trips(tmp_path):
    path = tmp_path / "shots.csv"
    frame = _frame([
        [170815, "gold", "false", "alice;bob", "2026-01-01", "retimed onset"],
        [178631, "silver", "true", "alice", "2026-01-02", ""],
        [185945, "unverified", "false", "", "", ""],
    ])
    write_roster(frame, path)
    got = read_roster(path)
    assert list(got.columns) == list(ROSTER_COLUMNS)
    assert got.shot.tolist() == [170815, 178631, 185945]
    assert got.tier.tolist() == ["gold", "silver", "unverified"]
    assert got.holdout.tolist() == ["false", "true", "false"]


def test_a_hand_set_tier_need_not_agree_with_the_reviewer_count():
    """`tier` is a curation call now, not a derived count.

    The user's own sawtooth roster is the motivating case: ten curated
    gold shots that nobody has reviewed yet.
    """
    got = validate_roster(
        _frame([[178642, "gold", "false", "", "", ""]])
    )
    assert got.iloc[0].tier == "gold"
    got = validate_roster(
        _frame([[1, "silver", "false", "alice;bob", "2026-01-01", ""]])
    )
    assert got.iloc[0].tier == "silver"


def test_an_unknown_tier_is_rejected():
    with pytest.raises(DatabaseError, match="tier"):
        validate_roster(_frame([[1, "bronze", "false", "alice", "2026-01-01", ""]]))


def test_holdout_is_required():
    with pytest.raises(DatabaseError, match="holdout"):
        validate_roster(_frame([[1, "unverified", "", "", "", ""]]))


def test_an_unknown_holdout_value_is_rejected():
    with pytest.raises(DatabaseError, match="holdout"):
        validate_roster(_frame([[1, "unverified", "yes", "", "", ""]]))


def test_holdout_true_and_false_are_both_accepted():
    got = validate_roster(_frame([
        [1, "unverified", "true", "", "", ""],
        [2, "unverified", "false", "", "", ""],
    ]))
    assert got.holdout.tolist() == ["true", "false"]


def test_a_shot_appears_once():
    with pytest.raises(DatabaseError, match="duplicate shot"):
        validate_roster(_frame([
            [1, "silver", "false", "alice", "2026-01-01", ""],
            [1, "silver", "false", "bob", "2026-01-01", ""],
        ]))


def test_a_reviewer_appears_once_per_shot():
    with pytest.raises(DatabaseError, match="duplicate reviewer"):
        validate_roster(_frame([[1, "gold", "false", "alice;alice", "2026-01-01", ""]]))


def test_a_verified_row_carries_a_date():
    with pytest.raises(DatabaseError, match="verified_on"):
        validate_roster(_frame([[1, "silver", "false", "alice", "", ""]]))
    with pytest.raises(DatabaseError, match="verified_on"):
        validate_roster(_frame([[1, "unverified", "false", "", "2026-01-01", ""]]))


def test_two_reviewers_both_land_in_reviewers_and_tier_is_left_alone(tmp_path):
    path = tmp_path / "shots.csv"
    write_roster(_frame([[170815, "unverified", "false", "", "", ""]]), path)

    record_review(path, 170815, "alice", on=date(2026, 1, 1))
    row = read_roster(path).iloc[0]
    assert row.tier == "unverified"
    assert row.reviewers == "alice"
    assert row.verified_on == "2026-01-01"

    record_review(path, 170815, "bob", on=date(2026, 1, 2))
    row = read_roster(path).iloc[0]
    assert row.tier == "unverified"
    assert row.reviewers == "alice;bob"
    assert row.verified_on == "2026-01-02"


def test_the_same_reviewer_twice_redates_without_promoting(tmp_path):
    path = tmp_path / "shots.csv"
    write_roster(
        _frame([[170815, "silver", "false", "alice", "2026-01-01", ""]]), path
    )
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
    assert got.tier.tolist() == ["unverified"]
    assert got.holdout.tolist() == ["false"]
    assert got.notes.tolist() == ["new"]


def test_notes_stay_on_one_line(tmp_path):
    path = tmp_path / "shots.csv"
    write_roster(_frame([[1, "unverified", "false", "", "", ""]]), path)
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
        assert len(frame), f"{category} roster is empty"
        if frame.shot.tolist() != [1, 2, 3]:
            # A curated category. `read_roster` has already validated it;
            # its shots are the curator's business, so pin nothing here -
            # a hardcoded count would fail the next time one is added.
            continue
        assert frame.tier.tolist() == ["gold", "silver", "unverified"]
