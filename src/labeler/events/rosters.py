"""The per-category review roster: who has looked at which shot.

`data/events/<category>/shots.csv` is a shot-level roster, not an interval
table. A shot enters it when somebody puts it up for review; the interval
tables under `format/` and `extend_*/` stay the record of what is labelled.

`tier` is a hand-set curation call, not a count: `validate_roster` only
checks that it is one of `unverified`, `silver`, `gold`, and nothing derives
it from the reviewer list. `holdout` is a required boolean reserving a shot
from training and tuning for final evaluation only. `reviewers` and
`verified_on` keep their own meaning — who pressed Verify and when — and
their own rule: blank iff there are no reviewers. Pressing Verify records a
reviewer and a date and does not touch `tier`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from ..config import Paths
from .databases import DatabaseError

ROSTER_COLUMNS = ("shot", "tier", "holdout", "reviewers", "verified_on", "notes")
TIERS = ("unverified", "silver", "gold")
HOLDOUT_VALUES = ("false", "true")
REVIEWER_SEPARATOR = ";"
ROSTER_NAME = "shots.csv"


def split_reviewers(value) -> list[str]:
    """Reviewer ids from one cell, in the order they reviewed."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part for part in str(value).split(REVIEWER_SEPARATOR) if part]


def roster_path(event: str, *, root: Path | None = None) -> Path:
    """The roster for one category, independently of the working directory."""
    root = Paths.from_env().label_tables if root is None else Path(root)
    return root / event / ROSTER_NAME


def validate_roster(frame: pd.DataFrame) -> pd.DataFrame:
    """Check the public roster schema; return it with `shot` as int64."""
    if tuple(frame.columns) != ROSTER_COLUMNS:
        raise DatabaseError(f"Expected columns {ROSTER_COLUMNS}")
    result = frame.copy()
    for column in ("tier", "holdout", "reviewers", "verified_on", "notes"):
        result[column] = result[column].fillna("").astype(str)
    shots = pd.to_numeric(result["shot"], errors="coerce")
    if not (shots.notna() & (shots % 1 == 0) & (shots >= 0)).all():
        raise DatabaseError("shot must be a nonnegative integer")
    result["shot"] = shots.astype("int64")
    duplicated = result["shot"][result["shot"].duplicated()]
    if len(duplicated):
        raise DatabaseError(f"duplicate shot {int(duplicated.iloc[0])}")
    for row in result.itertuples():
        reviewers = split_reviewers(row.reviewers)
        if len(set(reviewers)) != len(reviewers):
            raise DatabaseError(f"duplicate reviewer on shot {row.shot}")
        if row.tier not in TIERS:
            raise DatabaseError(f"tier {row.tier!r} is not one of {TIERS}")
        if row.holdout not in HOLDOUT_VALUES:
            raise DatabaseError(
                f"shot {row.shot}: holdout {row.holdout!r} is not one of "
                f"{HOLDOUT_VALUES}"
            )
        if bool(reviewers) != bool(row.verified_on):
            raise DatabaseError(
                f"shot {row.shot}: verified_on and reviewers must agree"
            )
        if "\n" in row.notes:
            raise DatabaseError(f"shot {row.shot}: notes must be one line")
    return result


def read_roster(path) -> pd.DataFrame:
    """Read and validate one roster."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not len(frame.columns):
        frame = pd.DataFrame(columns=list(ROSTER_COLUMNS))
    return validate_roster(frame)


def write_roster(frame: pd.DataFrame, path) -> None:
    """Validate, sort by shot, and write one roster."""
    validated = validate_roster(frame).sort_values("shot", ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(path, index=False)


def record_review(
    path,
    shot: int,
    reviewer: str,
    *,
    on: date | None = None,
    notes: str | None = None,
) -> pd.DataFrame:
    """Append one reviewer to a shot's row. `tier` is left untouched.

    A reviewer who has already reviewed this shot re-dates the row rather
    than appearing twice. A shot not yet in the roster is added, with
    `tier="unverified"` and `holdout="false"`. Returns the roster as written.
    """
    frame = read_roster(path)
    on = datetime.now(UTC).date() if on is None else on
    rows = frame.index[frame["shot"] == int(shot)]
    if len(rows):
        index = rows[0]
    else:
        index = len(frame)
        frame.loc[index] = [int(shot), "unverified", "false", "", "", ""]
    reviewers = split_reviewers(frame.at[index, "reviewers"])
    if reviewer not in reviewers:
        reviewers.append(reviewer)
    frame.at[index, "reviewers"] = REVIEWER_SEPARATOR.join(reviewers)
    frame.at[index, "verified_on"] = on.isoformat()
    if notes is not None:
        frame.at[index, "notes"] = notes
    write_roster(frame, path)
    return read_roster(path)
