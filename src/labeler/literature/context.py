"""Where a paper names a DIII-D shot, and whether the words around it say so.

A shot number counts when it lies within `REACH` characters of
"shot", "discharge", "#" or "DIII-D":

- `exact`: the number itself is written. A number in a run of numbers joined
  only by commas, semicolons, slashes, "&", "and", "or" or spaces counts when
  any number of the run is within reach, so a long list keeps its context.
- A stated range within reach counts both written six-digit ends as `exact`
  and its unwritten numbers as `range`:
  `189600-189650`, `189600 to 189650`, `189600 through 189650`, or `189600-50`
  (a dash and 2 to 5 closing digits; that closing shot is `range` too). A range
  has ends differing by at most `RANGE_CAP`; a wider or reversed one is not a range:
  only its six-digit tokens can count, and an abbreviated end counts nothing.

Each context extends `REACH` characters either side of the span covering its
number (its whole range for a `range` mention) and the whole keyword match
nearest that number among those in reach of its run or range.

Numbers immediately after "%" are URL encoding, not shot tokens or range starts.
Round, equally spaced runs of at least three numbers are axis ticks, not shot
tokens or range starts, and cannot give the remaining run its context.
Numbers immediately after an ASCII letter and hyphen are identifiers, not shot
tokens or range starts.
Numbers after an article-numbering journal's name and volume are citation
identifiers, not shot tokens or range starts, even when the name is glued to a word.
Numbers followed within 40 characters by a comma and a listed country, without
intervening digits, are postal codes, not shot tokens or range starts.

A number counts only as DIII-D's. The machine a number (a run, a range) belongs
to is the nearest device name within reach before it, else the nearest after
it; with no name in reach it is the paper's machine. Counts over the whole
normalised text choose NSTX-U (including NSTX) when it outnumbers DIII-D and
at least ties LHD, or LHD when it outnumbers both; otherwise it is DIII-D.
Only NSTX-U and LHD change this default because their shot numbers overlap
the corpus's 185601-204999 range (`FIRST_SHOT` and `LAST_SHOT`); the other
machines' shot numbers do not.

Text is normalised first: dashes become "-", soft hyphens go, a word broken
across a line ("dis-\\ncharge") is joined, and runs of whitespace become one space.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Collection
from dataclasses import dataclass
from itertools import pairwise

REACH = 60
RANGE_CAP = 50
TICK_MIN_RUN = 3
TICK_MIN_STEP = 100
POSTAL_REACH = 40
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
_TOKEN = re.compile(r"(?<![\d.%])\d{6}(?!\d|\.\d)")
_RUN_GAP = re.compile(r"\s*(?:(?:[,;/&]|and|or)\s*)*", re.IGNORECASE)
_RANGE = re.compile(
    r"(?<![\d.%])(?P<lo>\d{6})\s*(?P<sep>-|to|through|thru)\s*"
    r"(?P<tail>\d{2,6})(?!\d|\.\d)",
    re.IGNORECASE,
)
_PHYSICAL_REVIEW = re.compile(
    r"(?:Phys\.|Physical)\s*(?:Rev\.|Review)\s*"
    r"(?:(?:Lett\.|Letters|[A-EX]\.?|Research|Applied|Fluids|Accel\.\s*Beams)\s*)?"
    r"\d{1,4}\s*,?\s*$"
)
_OTHER_JOURNALS = re.compile(
    r"(?:Appl\.\s*Phys\.\s*Lett\.|Applied\s*Physics\s*Letters"
    r"|J\.\s*Appl\.\s*Phys\.|Journal\s*of\s*Applied\s*Physics"
    r"|J\.\s*Chem\.\s*Phys\.|Journal\s*of\s*Chemical\s*Physics"
    r"|Phys\.\s*Plasmas|Physics\s*of\s*Plasmas"
    r"|Rev\.\s*Sci\.\s*Instrum\.|Review\s*of\s*Scientific\s*Instruments"
    r"|Nucl\.\s*Fusion|Nuclear\s*Fusion"
    r"|Plasma\s*Phys\.\s*Control\.\s*Fusion"
    r"|Plasma\s*Physics\s*and\s*Controlled\s*Fusion)\s*\d{1,4}\s*,?\s*$"
)
# Longest normalised prefix: "Plasma Physics and Controlled Fusion 1234 , "
# (44 characters); 64 keeps a 20-character margin.
_ARTICLE_SPAN = 64
_IDENTIFIER = re.compile(r"[A-Za-z]-$")
_POSTAL = re.compile(
    r"[^\d]*?,\s*(?:(?:P\.?\s*R\.?\s*)?China|Russia|Russian Federation"
    r"|India|Kazakhstan|Singapore)\b"
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
    """The keyword in reach of [a, b) nearest the mention's [start, end)."""
    spans = [(m.start(), m.end()) for m in KEYWORDS.finditer(text)]
    starts = [a for a, _ in spans]
    ends = [b for _, b in spans]  # sorted too: matches never overlap

    def near(a: int, b: int, start: int, end: int) -> tuple[int, int] | None:
        lo = bisect.bisect_left(ends, a - REACH)
        hi = bisect.bisect_right(starts, b + REACH)
        if lo >= hi:
            return None
        i = bisect.bisect_left(starts, start, lo, hi)
        candidates = spans[max(lo, i - 1) : min(hi, i + 1)]
        # An equally distant keyword before the mention wins the tie.
        return min(candidates, key=lambda k: max(start - k[1], k[0] - end, 0))

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


def _article_number(text: str, start: int) -> bool:
    """Whether a number follows an article-numbering journal and its volume."""
    before = max(0, start - _ARTICLE_SPAN)
    return bool(
        _PHYSICAL_REVIEW.search(text, before, start)
        or _OTHER_JOURNALS.search(text, before, start)
    )


def _not_token(text: str, m: re.Match) -> bool:
    # One extra character preserves the country's real word boundary at the edge.
    postal = _POSTAL.match(text, m.end(), m.end() + POSTAL_REACH + 1)
    return bool(
        _IDENTIFIER.search(text, max(0, m.start() - 2), m.start())
        or _article_number(text, m.start())
        or (postal and postal.end() <= m.end() + POSTAL_REACH)
    )


def _ticks(run: list[re.Match]) -> set[int]:
    """Offsets of round tick labels, including equal-step subruns."""
    values = [int(m.group()) for m in run]
    dropped = set()
    for i in range(len(run) - TICK_MIN_RUN + 1):
        part = values[i : i + TICK_MIN_RUN]
        step = part[1] - part[0]
        if (
            abs(step) >= TICK_MIN_STEP
            and all(b - a == step for a, b in pairwise(part))
            and all(n % step == 0 for n in part)
        ):
            dropped.update(m.start() for m in run[i : i + TICK_MIN_RUN])
    return dropped


def _runs(text: str) -> list[list[re.Match]]:
    runs: list[list[re.Match]] = []
    for m in _TOKEN.finditer(text):
        if _not_token(text, m):
            continue
        if runs and _RUN_GAP.fullmatch(text[runs[-1][-1].end() : m.start()]):
            runs[-1].append(m)
        else:
            runs.append([m])
    kept = []
    for run in runs:
        ticks = _ticks(run)
        kept.append([m for m in run if m.start() not in ticks])
    return [run for run in kept if run]


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

    def mention(shot: int, kind: str, a: int, b: int, keyword: tuple[int, int]):
        if keep(shot):
            left, right = min(a, keyword[0]), max(b, keyword[1])
            found[a, shot, kind] = Mention(
                shot, kind, a, b, text[max(0, left - REACH) : right + REACH]
            )

    found = {}
    runs = _runs(text)
    token_starts = {m.start() for run in runs for m in run}
    for run in runs:
        span = (run[0].start(), run[-1].end())
        if other(*span):
            continue
        for m in run:
            keyword = near(*span, *m.span())
            if keyword is not None:
                mention(int(m.group()), "exact", *m.span(), keyword)
    for m in _RANGE.finditer(text):
        if m.start() not in token_starts:
            continue
        inside = _range(m)
        keyword = near(*m.span(), *m.span())
        if inside is not None and keyword is not None and not other(*m.span()):
            for s in inside:
                mention(s, "range", *m.span(), keyword)
            for group in ("lo", "tail"):
                if len(m.group(group)) == 6 and m.start(group) in token_starts:
                    keyword = near(*m.span(), *m.span(group))
                    mention(int(m.group(group)), "exact", *m.span(group), keyword)
    return sorted(found.values(), key=lambda x: (x.start, x.shot, x.match_type))
