"""The names the review page offers when it asks who is reviewing.

The page opens on this list: a reviewer picks their name, or adds it, and the
page records it with each save (`versions.clean_name`). A name is an
attribution, not a login: `reviewer` stays the login of the server's process.

The list is `<label tables>/reviewers.txt`, one name per line, which the owner
may edit. Until that file exists the list is every name already saved in an
event's `review/` logs, and the first name added writes the file with them.
Names are unique regardless of case, and sorted the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...config import atomic_path
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


def saved_names(label_tables) -> list[str]:
    """Every name in the events' review logs; a line that does not parse is skipped."""
    found = []
    for log in LOGS:
        for path in sorted(Path(label_tables).glob(f"*/{REVIEW}/{log}")):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict):
                    found.append(entry.get("name"))
    return _unique(_clean(found))


def read(label_tables) -> list[str]:
    """The names the page offers."""
    path = names_path(label_tables)
    if not path.is_file():
        return saved_names(label_tables)
    return _unique(_clean(path.read_text(encoding="utf-8").splitlines()))


def add(label_tables, name: str | None) -> tuple[list[str], str]:
    """`(names, name)` once `name` is on the list: as typed, or as already listed."""
    name = clean_name(name)
    if name is None:
        raise ValueError("a name is required")
    names = read(label_tables)
    listed = next((n for n in names if n.casefold() == name.casefold()), None)
    if listed is not None:
        return names, listed
    names = _unique([*names, name])
    with atomic_path(names_path(label_tables)) as tmp:
        tmp.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")
    return names, name
