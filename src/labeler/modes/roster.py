"""An event's `shots.csv` roster (`shot,tier,holdout,reviewers,verified_on,notes`), replaced.

The three MHD events share one list (v5's convention), so a proposed shot set replaces all
three the same way: the rows of the old roster whose shot is in the set stay as they are (a
`holdout` flag, a reviewer or a tier is never rewritten), the rows of every other shot go, and a
shot of the set the roster lacks is added as `unverified`, `holdout` false. An audited shot gets
the audited row whether or not it was listed (its `holdout` flag is kept when it was). A shot
with no label (the labeller failed on it) stays listed, unverified, with a note saying so.

Every function returns counts and never the shots: the old roster may hold held-out shots, which
nothing here prints or logs. `backup` copies a file before it is changed, and refuses a folder
inside a git working tree.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from labeler.config import atomic_path

COLUMNS = ["shot", "tier", "holdout", "reviewers", "verified_on", "notes"]
TIERS = ("unverified", "silver", "gold")


def read(path) -> pd.DataFrame:
    """The roster as text, every cell a string (a blank stays blank, a shot stays an int in text)."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"{path}: columns {list(frame.columns)}, expected {COLUMNS}")
    if frame.shot.duplicated().any():
        raise ValueError(f"{path}: a shot is listed twice")
    return frame


def audited_row(
    shot: int, tier: str, reviewers: str, verified_on: str, notes: str, holdout="false"
) -> dict:
    if tier not in TIERS:
        raise ValueError(f"tier {tier!r} is not one of {TIERS}")
    return {
        "shot": str(int(shot)),
        "tier": tier,
        "holdout": holdout,
        "reviewers": reviewers,
        "verified_on": verified_on,
        "notes": notes,
    }


def replace(
    old: pd.DataFrame,
    shots,
    audited: dict[int, dict] | None = None,
    notes: dict[int, str] | None = None,
):
    """`(new roster, counts)`: see the module docstring. `audited`: `{shot: audited_row(...)}`;
    `notes`: `{shot: note}` added to the note cell of a listed shot (after any note it has).

    counts: `old`, `removed`, `kept`, `added`, `audited` (rows set by `audited`, of which
    `audited_new` were not in the old roster), `new`."""
    audited, notes = audited or {}, notes or {}
    wanted = {int(s) for s in shots} | set(audited)
    have = old.shot.astype(int)
    keep = have.isin(wanted)
    rows = old[keep].copy()
    present = set(have[keep])
    new = [
        {
            "shot": str(s),
            "tier": "unverified",
            "holdout": "false",
            "reviewers": "",
            "verified_on": "",
            "notes": "",
        }
        for s in sorted(wanted - present)
    ]
    rows = pd.concat([rows, pd.DataFrame(new, columns=COLUMNS)], ignore_index=True)
    for shot, row in audited.items():
        at = rows.index[rows.shot.astype(int) == shot]
        was = rows.loc[at[0], "holdout"] if len(at) else None
        row = {**row, "holdout": was if was is not None else row["holdout"]}
        rows.loc[at[0], COLUMNS] = [row[c] for c in COLUMNS]
    for shot, note in notes.items():
        at = rows.index[rows.shot.astype(int) == shot]
        if len(at) and shot not in audited:
            was = rows.loc[at[0], "notes"]
            rows.loc[at[0], "notes"] = f"{was}; {note}" if was else note
    if have.is_monotonic_increasing:
        rows = rows.sort_values("shot", key=lambda s: s.astype(int), kind="stable")
    rows = rows.reset_index(drop=True)
    counts = {
        "old": len(old),
        "removed": int((~keep).sum()),
        "kept": int(keep.sum()),
        "added": len(new),
        "audited": len(audited),
        "audited_new": sum(1 for s in audited if s not in present),
        "noted": sum(
            1 for s in notes if s in set(rows.shot.astype(int)) and s not in audited
        ),
        "new": len(rows),
    }
    return rows, counts


def write(path, frame: pd.DataFrame) -> None:
    """The roster as the CSV every reader expects (the columns in order, LF, no index)."""
    with atomic_path(path) as tmp:
        frame[COLUMNS].to_csv(tmp, index=False, lineterminator="\n")


def backup(path, base, destination) -> Path:
    """Copy `path` to `destination/<path relative to base>` before it is changed.

    An existing copy is kept (the first backup is the original); a `destination` inside a git
    working tree is refused, since a roster may hold held-out shot numbers."""
    path, base, destination = (
        Path(path).resolve(),
        Path(base).resolve(),
        Path(destination).resolve(),
    )
    for folder in (destination, *destination.parents):
        if (folder / ".git").exists():
            raise ValueError(f"{destination} is inside the git working tree {folder}")
    target = destination / path.relative_to(base)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return target
