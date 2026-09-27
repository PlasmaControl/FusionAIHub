"""Every saved version of a shot's label, and the name a reviewer types.

`review/history.jsonl` is append-only: one line per save, never rewritten. A
shot's versions are its lines in order, numbered from 1. Restoring a version
in the page loads it as a draft; saving that draft appends a new line, so no
version is ever lost and the file stays a complete record of who saved what.

Each line keeps two identities. `reviewer` is the login of the process that
served the page (`getpass.getuser()`): the checker's blind-read integrity
rules key on it, so it is never taken from the client. `name` is what the
reviewer typed in the page's name box, or null: an attribution, not an
authentication. Lines written before names existed have no `name` key and
read as null.
"""

from __future__ import annotations

import unicodedata

from . import labels

#: The longest name the page may send; longer is refused, not cut.
NAME_MAX = 64


def clean_name(name: str | None) -> str | None:
    """The typed name with its outer whitespace removed; None when empty.

    Refuses a name longer than `NAME_MAX` characters, Unicode Cc (C0 and C1
    controls, including newline, tab, NUL and NEL), Cs (lone surrogates, which
    UTF-8 cannot encode), line and paragraph separators U+2028 and U+2029,
    and bidi embedding, override and isolate controls U+202A-U+202E and
    U+2066-U+2069, which reorder the displayed line, the login included.
    Every other character is accepted, including ZWNJ (U+200C) and ZWJ
    (U+200D), so scripts that need joiners can keep them in a name.
    """
    if name is None:
        return None
    name = str(name).strip()
    if not name:
        return None
    if len(name) > NAME_MAX:
        raise ValueError(f"a name is at most {NAME_MAX} characters")
    if any(
        unicodedata.category(ch) in {"Cc", "Cs"}
        or "\u2028" <= ch <= "\u202e"
        or "\u2066" <= ch <= "\u2069"
        for ch in name
    ):
        raise ValueError("a name cannot hold control characters")
    return name


def shot_versions(event_dir, shot: int) -> list[dict]:
    """The shot's saves, oldest first: version, time, who, and what was saved."""
    shot = int(shot)
    found = []
    for entry in labels.read_history(event_dir):
        if int(entry["shot"]) != shot:
            continue
        found.append(
            {
                "version": len(found) + 1,
                "saved_at": entry["saved_at"],
                "reviewer": entry.get("reviewer"),
                "name": entry.get("name"),
                "window": list(entry["window"]),
                "intervals": [list(span) for span in entry["intervals"]],
                "source": entry.get("source"),
            }
        )
    return found
