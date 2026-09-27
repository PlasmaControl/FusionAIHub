"""Every saved version of a shot's label, and the name a reviewer types.

`review/history.jsonl` is append-only: one line per save, never rewritten. A
shot's versions are its lines in order, numbered from 1. Restoring a version
in the page loads it as a draft; a save appends a version and never rewrites
one, but a crash between writing the label and its history line can leave
the current label without its version line.

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
JOINERS = frozenset("\u200c\u200d")


def clean_name(name: str | None) -> str | None:
    """The typed name with its outer whitespace removed; None when empty.

    Refuses a name longer than `NAME_MAX` characters and every Unicode C
    character: Cc (controls), Cf (format), Cs (surrogates), Co (private use)
    and Cn (unassigned), except ZWNJ (U+200C) and ZWJ (U+200D), which pass
    because scripts need these joiners inside words. Also refuses line and
    paragraph separators U+2028 and U+2029 because they break a line.
    """
    if name is None:
        return None
    name = str(name).strip()
    if not name:
        return None
    if len(name) > NAME_MAX:
        raise ValueError(f"a name is at most {NAME_MAX} characters")
    if any(
        (unicodedata.category(ch).startswith("C") and ch not in JOINERS)
        or ch in {"\u2028", "\u2029"}
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
