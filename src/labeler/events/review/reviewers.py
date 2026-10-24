"""The names the review page offers when it asks who is reviewing.

The page opens on this list: a reviewer picks their name, or adds it, and the
page records it with each save (`versions.clean_name`). A name is an
attribution, not a login: `reviewer` stays the login of the server's process.

The list is `<label tables>/reviewers.txt`, one name per line, which the owner
may edit, and every name already saved in an event's `review/` logs: a name is on
the list from its first save. Add Name appends a line to the file (the first one
writes it, with the names already saved) and never rewrites the others when the
process may write there (the owner's); a line that
is not a name is left in the file and off the list. In a shared folder another
account cannot write the file: its Add Name writes nothing, the name is kept by
its first save. The page shows the names unique regardless of case, and sorted
the same way.

`shot_reviewers` is the other list: everyone who saved a shot's label or its AE
mask decisions, read from the same logs, so a save adds its reviewer once.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ...config import atomic_path
from . import saves
from .labels import REVIEW
from .versions import clean_name

FILE = "reviewers.txt"
#: The review logs whose lines carry a `name`: labels and AE mask decisions.
LOGS = ("history.jsonl", "masks.jsonl")


def names_path(label_tables) -> Path:
    return Path(label_tables) / FILE


def _unique(names) -> list[str]:
    """First spelling of each name, case aside, sorted without regard to case."""
    kept: dict[str, str] = {}
    for name in names:
        kept.setdefault(name.casefold(), name)
    return sorted(kept.values(), key=str.casefold)


def _clean(names) -> list[str]:
    found = []
    for name in names:
        try:
            name = clean_name(name) if isinstance(name, str) else None
        except ValueError:
            continue
        if name:
            found.append(name)
    return found


def _log_lines(path: Path):
    """The log's entries; a line that does not parse is skipped."""
    if not path.is_file():
        return
    for line in saves.read_text(path).splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            yield entry


def saved_names(label_tables) -> list[str]:
    """Every name in the events' review logs (and their subcategories'); a line
    that does not parse is skipped."""
    found = []
    for log in LOGS:
        paths = [
            *Path(label_tables).glob(f"*/{REVIEW}/{log}"),
            *Path(label_tables).glob(f"*/subcategories/*/{REVIEW}/{log}"),
        ]
        for path in sorted(paths):
            found.extend(entry.get("name") for entry in _log_lines(path))
    return _unique(_clean(found))


def shot_reviewers(event_dir, shot: int) -> list[dict]:
    """Everyone who saved the shot under a name, first contributor first.

    One row per person. A save made without a name credits no one: the server's
    login keys the checker's integrity rules, it is not a reviewer. A name is the
    same person regardless of case; its first spelling is kept.
    """
    shot = int(shot)
    kept: dict[str, dict] = {}
    for log in LOGS:
        for entry in _log_lines(Path(event_dir) / REVIEW / log):
            try:
                if int(entry.get("shot")) != shot:
                    continue
            except (TypeError, ValueError):
                continue
            who = next(iter(_clean([entry.get("name")])), None)
            if who is None:
                continue
            at = str(entry.get("saved_at") or "")
            row = kept.setdefault(
                who.casefold(),
                {"name": who, "saves": 0, "first": at, "last": at},
            )
            row["saves"] += 1
            row["first"] = min(row["first"], at)
            row["last"] = max(row["last"], at)
    return sorted(kept.values(), key=lambda row: row["first"])


def read(label_tables) -> list[str]:
    """The names the page offers: the owner's file and every saved name."""
    path = names_path(label_tables)
    listed = (
        _clean(path.read_text(encoding="utf-8-sig").splitlines())
        if path.is_file() else []
    )
    return _unique([*listed, *saved_names(label_tables)])


def _may_write(path: Path) -> bool:
    return os.access(path if path.exists() else path.parent, os.W_OK)


def add(label_tables, name: str | None) -> tuple[list[str], str]:
    """`(names, name)` once `name` is on the list: as typed, or as already listed.

    Without write access to `reviewers.txt` (a contributor in a shared folder) the
    file is left alone and the name is on the list from its first save.
    """
    name = clean_name(name)
    if name is None:
        raise ValueError("a name is required")
    names = read(label_tables)
    listed = next((n for n in names if n.casefold() == name.casefold()), None)
    if listed is not None:
        return names, listed
    path = names_path(label_tables)
    if not _may_write(path):
        return _unique([*names, name]), name
    if path.is_file():
        text = path.read_text(encoding="utf-8-sig")
        text += "" if not text or text.endswith("\n") else "\n"
    else:
        text = "".join(f"{n}\n" for n in names)
    with atomic_path(path) as tmp:
        tmp.write_text(f"{text}{name}\n", encoding="utf-8")
    return read(label_tables), name
