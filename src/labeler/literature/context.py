"""Where a paper names a DIII-D shot, and whether the words around it say so.

A shot number counts when it lies within `REACH` characters of
"shot", "discharge", "#" or "DIII-D":

- `exact`: the number itself is written. A number in a run of numbers joined
  only by commas, semicolons, slashes, "&", "and", "or" or spaces counts when
  any number of the run is within reach, so a long list keeps its context.
- `range`: the number lies strictly inside a stated range within reach:
  `189600-189650`, `189600 to 189650`, `189600 through 189650`, or `189600-50`
  (a dash and 2 to 5 closing digits; that closing shot is `range` too). A range
  spans at most `RANGE_CAP` shots; a wider one is not expanded. Its written
  ends are `exact`.

A number counts only as DIII-D's. The machine a number (a run, a range) belongs
to is the nearest device name within reach before it, else the nearest after
it; with no name in reach it is the paper's machine. Counts over the whole
normalised text choose NSTX-U (including NSTX) when it outnumbers DIII-D and
at least ties LHD, or LHD when it outnumbers both; otherwise it is DIII-D.
Only NSTX-U and LHD change this default because their shot numbers overlap
the corpus's 185601-206000 range; the other machines' shot numbers do not.

Text is normalised first: dashes become "-", soft hyphens go, a word broken
across a line ("dis-\\ncharge") is joined, and runs of whitespace become one space.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Collection
from dataclasses import dataclass

REACH = 60
RANGE_CAP = 50
KEYWORDS = re.compile(r"\bshot|\bdischarge|#|DIII\s?-\s?D", re.IGNORECASE)
DIII_D = re.compile(r"DIII\s?-\s?D", re.IGNORECASE)
#: Other machines, whose own shot numbers can read as DIII-D's.
OTHER_DEVICES = re.compile(
    r"\b(?:NSTX(?:-U)?|EAST|KSTAR|JET|ASDEX|AUG|TCV|MAST(?:-U)?|JT-60(?:U|SA)?"
    r"|WEST|LHD|HL-2A|HL-3|C-Mod|TFTR|W7-X|ST40|COMPASS)\b"
)
NSTX_U = re.compile(r"\bNSTX(?:-U)?\b")
LHD = re.compile(r"\bLHD\b")

_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−﹣－"} | {"­": None, "＃": "#"})
_BROKEN_WORD = re.compile(r"([A-Za-z])-[ \t]*\n\s*([a-z])")
_TOKEN = re.compile(r"(?<![\d.])\d{6}(?!\d|\.\d)")
_RUN_GAP = re.compile(r"\s*(?:(?:[,;/&]|and|or)\s*)*", re.IGNORECASE)
_RANGE = re.compile(
    r"(?<![\d.])(?P<lo>\d{6})\s*(?P<sep>-|to|through|thru)\s*"
    r"(?P<tail>\d{2,6})(?!\d|\.\d)",
    re.IGNORECASE,
)


def normalise(text: str) -> str:
    text = _BROKEN_WORD.sub(r"\1\2", text.translate(_DASHES))
    return re.sub(r"\s+", " ", text)


@dataclass(frozen=True)
class Mention:
    """One shot number the text names in context, and the text around it."""

    shot: int
    match_type: str  # "exact" or "range"
    start: int  # offsets into the normalised text
    end: int
    context: str


def _reach(text: str):
    """`near(a, b)`: whether some keyword ends or starts within reach of [a, b)."""
    spans = [(m.start(), m.end()) for m in KEYWORDS.finditer(text)]
    starts = [a for a, _ in spans]
    ends = [b for _, b in spans]  # sorted too: matches never overlap

    def near(a: int, b: int) -> bool:
        i = bisect.bisect_right(starts, b + REACH)  # keywords starting in reach
        return i > 0 and ends[i - 1] >= a - REACH

    return near


def _devices(text: str):
    """`other(a, b)`: whether [a, b) belongs to a machine other than DIII-D."""
    diii_d = [(m.start(), m.end(), False) for m in DIII_D.finditer(text)]
    paper_other = max(len(NSTX_U.findall(text)), len(LHD.findall(text))) > len(diii_d)
    names = sorted(
        diii_d + [(m.start(), m.end(), True) for m in OTHER_DEVICES.finditer(text)]
    )  # names never overlap, so ends are sorted too
    starts = [start for start, _, _ in names]

    def other(a: int, b: int) -> bool:
        i = bisect.bisect_left(starts, a)  # names[:i] start before the number
        if i and names[i - 1][1] >= a - REACH:
            return names[i - 1][2]
        if i < len(names) and names[i][0] <= b + REACH:
            return names[i][2]
        return paper_other

    return other


def _runs(text: str) -> list[list[re.Match]]:
    runs: list[list[re.Match]] = []
    for m in _TOKEN.finditer(text):
        if runs and _RUN_GAP.fullmatch(text[runs[-1][-1].end() : m.start()]):
            runs[-1].append(m)
        else:
            runs.append([m])
    return runs


def _range(m: re.Match) -> range | None:
    """The shots strictly inside a range match (and an abbreviated end)."""
    lo, sep, tail = m.group("lo"), m.group("sep"), m.group("tail")
    if len(tail) == 6:
        hi, last = int(tail), int(tail) - 1
    elif sep == "-":
        hi = last = int(lo[: 6 - len(tail)] + tail)
    else:
        return None
    if not 0 < hi - int(lo) <= RANGE_CAP:
        return None
    return range(int(lo) + 1, last + 1)


def mentions(text: str, shots: Collection[int] | None = None) -> list[Mention]:
    """Every shot the text names in context, in text order; `shots` limits which."""
    text = normalise(text)
    near, other = _reach(text), _devices(text)

    def keep(shot: int) -> bool:
        return shots is None or shot in shots

    def mention(shot: int, kind: str, a: int, b: int) -> Mention:
        return Mention(shot, kind, a, b, text[max(0, a - REACH) : b + REACH])

    found = []
    for run in _runs(text):
        if other(run[0].start(), run[-1].end()):
            continue
        if any(near(m.start(), m.end()) for m in run):
            found += [
                mention(int(m.group()), "exact", m.start(), m.end())
                for m in run
                if keep(int(m.group()))
            ]
    for m in _RANGE.finditer(text):
        inside = _range(m)
        if inside is not None and near(*m.span()) and not other(*m.span()):
            found += [mention(s, "range", *m.span()) for s in inside if keep(s)]
    return sorted(found, key=lambda x: (x.start, x.shot, x.match_type))
