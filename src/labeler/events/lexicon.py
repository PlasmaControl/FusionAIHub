"""What the operators call each phenomenon, and how a phrase becomes a hit.

`lexicons.yaml` is the list of names - the single source of the round-1
phenomenon ids and their aliases, read by shot_design as well as by labeler
(plan 5.6) - and this module is its reader and its matcher. Nothing here
knows where text comes from: `hits` takes TEXT, not a shot or a path, so
the corpus can be swapped under it without touching a line of the matching
(`text_weak.py` is the half that reads files, and has been swapped once
already).

The matching is deliberately dumb - space-glued phrases, no stemming, no
model - because the failure mode of a clever matcher here is a confident
wrong claim, and the failure mode of this one is a missed mention that the
detectors were going to have to find anyway. See A7 "text word-boundary".
"""
from __future__ import annotations

import functools
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

#: Plan 5.6's round-1 ids, and the ids added since: `transient`, the three
#: `qmin_*` rule labels and the `fast_ion` topic. The lexicon may not name
#: anything else: a phenomenon id is a join key against labels, events and
#: shot_design's `phenomena.yaml`, and a typo that loads silently is a
#: phenomenon that quietly has no evidence.
#:
#: The three `qmin_*` ids are `events/heuristics.py`'s q-min regime bands
#: (task L-D2). They are SCENARIO ids rather than instability ids - what
#: the discharge was doing, not what went wrong in it - which is why their
#: aliases are so much sparser than an instability's: an operator writes
#: "hybrid" when the shot is one and otherwise writes nothing at all.
#:
#: `fast_ion` was added by the I11 review and is deliberately NOT an alias
#: list bolted onto `ae`. `ae` is one specific MHD mode: it has a detector
#: band (>= 40 kHz tracks), a label head, and an `--avoid phenomenon:ae`
#: path that is read as "the AE detector looked and saw nothing". Folding
#: `fida`, `beam ion`, `fast ion` and `energetic particle` into it would
#: make a diagnostic name and a transport topic resolve to a mode
#: observation, and every consumer that reads an `ae` hit as "this shot had
#: an Alfven eigenmode" would then be reading a topic match. `fast_ion` is
#: a TOPIC: no detector writes it, no model scores it, and shot_design's
#: registry gives it text evidence only.
PHENOMENON_IDS = (
    "ae", "eho", "elm", "transient", "tearing", "sawtooth", "fishbone", "qcm",
    "qh", "lh", "detachment", "pickup", "rwm",
    "qmin_hybrid", "qmin_elevated", "qmin_high", "fast_ion",
)

#: The alias lists themselves. Shipped beside this module because shot_design
#: reads it too - one file, two readers.
DEFAULT_LEXICON = Path(__file__).with_name("lexicons.yaml")

#: What one mention is worth, and so what a single mention is capped at.
#: Four independent mentions reach 1.0; nothing else about text does.
TEXT_ONLY_CEILING = 0.25

#: A `.` BETWEEN two digits is part of the number and not punctuation -
#: `1.3` is one token. Everywhere else a `.` is punctuation, so this is
#: written as "a dot with a non-digit on one side or the other" and reused
#: by both of the patterns below, which have to agree about it.
_DECIMAL_SAFE_DOT = r"(?<!\d)\.|\.(?!\d)"

#: What ends a sentence: `. ! ? ;` and the newline - except the `.` inside
#: a decimal number. Splitting "beams 1.3/2.4 MW" on both dots leaves the
#: middle piece "3/2", which IS the tearing alias, so a power pair read as
#: a mode number. Measured corpus-wide after the fix: 84 of the 22,950
#: bundles had a run-scope tearing hit made of nothing but this - "GasA
#: (1.3/2.5/4V)" and "probe scan width/height 0.6/1.2/1.8kA" are the two
#: phrasings - and they are gone. (None of the 84 is one of the 500
#: `recommender_v1` shots, so the shipped table below is unchanged.)
_TERMINATOR = re.compile(rf"(?:[!?;\n]|{_DECIMAL_SAFE_DOT})+")

#: Everything that is NOT part of a word. `\w` keeps letters, digits and
#: the underscore; the hyphen and the slash are kept too, so that
#: "elm-free" is one word and not an ELM and "2/1" is a mode number and not
#: a 2 and a 1. The underscore is kept for the same reason in reverse: the
#: session text is full of control-system parameter names, and
#: `RWM_GAINMULT` is an identifier rather than somebody saying an RWM
#: happened. The decimal point is kept on the same terms as the sentence
#: split, so that "q95=3.2 at 2.11 s" carries no "3/2" either.
_PUNCTUATION = re.compile(rf"(?:[^\w\-/.]|{_DECIMAL_SAFE_DOT})+")

#: Stripped from a token's ENDS, where they are punctuation after all - a
#: trailing dash, a bare slash between two spaces.
_IN_TOKEN = "-/"


class LexiconError(ValueError):
    """The lexicon file says something the matcher cannot use."""


@dataclass(frozen=True)
class Phenomenon:
    """One phenomenon's names, and the phrases that deny it."""

    id: str
    title: str
    aliases: tuple[str, ...]
    negatives: tuple[str, ...] = ()
    weight: float = 1.0


@dataclass(frozen=True)
class Lexicon:
    """Every phenomenon the text layer knows how to look for.

    Frozen all the way down - tuples of frozen `Phenomenon`s - so it is
    hashable, which is what lets `_matchers` cache on the lexicon itself
    rather than on an identity a caller would have to keep alive.
    """

    version: int
    phenomena: tuple[Phenomenon, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.phenomena)

    def __getitem__(self, phenomenon_id: str) -> Phenomenon:
        for p in self.phenomena:
            if p.id == phenomenon_id:
                return p
        raise KeyError(phenomenon_id)


@dataclass(frozen=True)
class Hit:
    """One phenomenon named in one sentence.

    `polarity` is `"neg"` when that sentence also carries one of the
    phenomenon's `negatives`: "no elms" names ELMs and denies them, and a
    counter that could not tell the two apart would rank the shots where
    somebody wrote that a phenomenon was ABSENT.
    """

    phenomenon: str
    alias: str
    sentence: str
    polarity: str


def load_lexicon(path=None) -> Lexicon:
    """Parse and check the lexicon; every problem is a `LexiconError`.

    Checked rather than trusted because this file is edited by hand, is
    read by two packages, and its mistakes are silent: an alias with a
    capital in it can never match (the matcher lowercases the text and not
    the alias), and an unknown id produces evidence rows that join to
    nothing.

    UTF-8 explicitly, not the process locale: `alfvén` is in the shipped
    file, and a bundle read under `LANG=C` would otherwise raise a
    `UnicodeDecodeError` from inside a function that promises `LexiconError`.
    """
    path = Path(DEFAULT_LEXICON if path is None else path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError) as exc:
        raise LexiconError(f"lexicon not readable: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise LexiconError(f"{path}: the lexicon is a mapping, not {type(raw)}")
    version = raw.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise LexiconError(f"{path}: `version` must be a positive integer")
    entries = raw.get("phenomena")
    if not isinstance(entries, Mapping) or not entries:
        raise LexiconError(f"{path}: `phenomena` must be a non-empty mapping")
    out = []
    for pid, body in entries.items():
        if pid not in PHENOMENON_IDS:
            raise LexiconError(
                f"{path}: {pid!r} is not one of the round-1 phenomena "
                f"{PHENOMENON_IDS}"
            )
        if not isinstance(body, Mapping):
            raise LexiconError(f"{path}: {pid} must be a mapping")
        title = body.get("title")
        if not isinstance(title, str) or not title.strip():
            raise LexiconError(f"{path}: {pid} needs a non-empty `title`")
        aliases = _phrases(path, pid, body.get("aliases"), "aliases")
        if not aliases:
            raise LexiconError(f"{path}: {pid} needs at least one of `aliases`")
        negatives = _phrases(path, pid, body.get("negatives") or [], "negatives")
        _check_reachable(path, pid, aliases, negatives)
        weight = body.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) \
                or not (math.isfinite(weight) and weight > 0.0):
            raise LexiconError(
                f"{path}: {pid} `weight` must be a finite positive number"
            )
        out.append(
            Phenomenon(id=pid, title=title, aliases=aliases,
                       negatives=negatives, weight=float(weight))
        )
    return Lexicon(version=int(version), phenomena=tuple(out))


def _phrases(path, pid: str, value, field: str) -> tuple[str, ...]:
    """A checked list of lowercase, non-empty, matchable phrases."""
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise LexiconError(f"{path}: {pid} `{field}` must be a list of phrases")
    seen: list[str] = []
    for phrase in value:
        if not isinstance(phrase, str) or not phrase.strip():
            raise LexiconError(f"{path}: {pid} `{field}` holds an empty phrase")
        if phrase != phrase.lower():
            raise LexiconError(
                f"{path}: {pid} `{field}` phrase {phrase!r} is not lowercase; "
                "the matcher lowercases the TEXT, so a capital never matches"
            )
        if _glue(phrase).strip() == "":
            raise LexiconError(
                f"{path}: {pid} `{field}` phrase {phrase!r} is all punctuation"
            )
        if phrase in seen:
            raise LexiconError(f"{path}: {pid} `{field}` repeats {phrase!r}")
        seen.append(phrase)
    return tuple(seen)


def _check_reachable(path, pid: str, aliases, negatives) -> None:
    """Refuse a negative that no alias of its own phenomenon can fire with.

    A negative does not produce evidence by itself: it marks a sentence
    that an ALIAS already matched as a statement of absence (see `hits`).
    So the matcher is run on each negative phrase alone, and a phrase that
    matches nothing is dead text - `"sawtooth-free"` is one token, so
    `" sawtooth "` is not inside it, and unless the hyphenated form is an
    alias too the phrase yields no evidence at all where `"elm-free"`
    yields evidence of absence. That asymmetry shipped once; it is checked
    here rather than left to a reviewer's eye.
    """
    glued = [_glue(a) for a in aliases]
    for negative in negatives:
        target = _glue(negative)
        if not any(a in target for a in glued):
            raise LexiconError(
                f"{path}: {pid} negative {negative!r} can never fire - no "
                f"alias of {pid} matches the phrase itself, so no sentence "
                "carrying it can be marked negative. Add the phrase to "
                "`aliases` too, the way `elm-free` is"
            )


def sentences(text: str) -> list[str]:
    """Lowercased, whitespace-collapsed sentences.

    Split on `. ! ? ;` and on newlines, because a logbook entry is as often
    a list of lines as it is prose. The sentence is the unit the matcher
    works in: it is the span over which a negation ("no elms") plausibly
    applies, and small enough that two phenomena named in one are named
    together. It is also the unit that keeps a phrase from spanning two
    logbook ENTRIES, which would attribute one person's word to another's.

    The cost of splitting on newlines is a phrase that a wrapped line broke
    in half: `"resistive wall\\nmode"` is not an `rwm` hit. It degrades
    gracefully wherever a shorter alias exists - `"edge harmonic\\noscillation"`
    is still an `eho` hit via `edge harmonic` - and that is part of why the
    lexicon carries the short spellings, but a long phrase with no short
    form is simply missed.

    A `.` between two digits is NOT a terminator (`_DECIMAL_SAFE_DOT`), so
    text whose only separator was a decimal point is ONE sentence. That is
    the rule behaving correctly - `"1.2"` was never the end of a sentence -
    but it is the one way it can LOWER a count rather than remove a false
    positive: two mentions split by `"... 4.5 ..."` now share a sentence
    and collapse into one `n_pos`, and a negative can now reach an alias it
    could not before. Measured over 500 shots: no movement.
    """
    return [
        s for s in (" ".join(part.split()).lower()
                    for part in _TERMINATOR.split(text))
        if s
    ]


def _glue(sentence: str) -> str:
    """The sentence as `" word word "`, for whole-phrase containment.

    Every character that is not part of a word becomes a space, EXCEPT a
    hyphen or a slash inside a token and a `.` between two digits:
    "elm-free" is one token, so is "2/1", and so is "1.3" - while a
    trailing dash, a comma, or the full stop that ends a sentence is not
    part of the word before it. Wrapping the result in spaces is what makes
    `" nt " in " we want "` false - the whole point of the exercise.

    The decimal rule is not a nicety. `.` was a separator everywhere until
    the L7-fix review, so "beams 1.3/2.4 MW" glued to " beams 1 3/2 4 mw "
    and a beam power pair produced the tearing alias "3/2".
    """
    flat = _PUNCTUATION.sub(" ", sentence.lower())
    tokens = (token.strip(_IN_TOKEN) for token in flat.split())
    return f" {' '.join(t for t in tokens if t)} "


@functools.lru_cache(maxsize=8)
def _matchers(lexicon: Lexicon):
    """Per phenomenon, its glued aliases longest first and its negatives.

    Longest first so that a sentence saying "edge harmonic oscillation" is
    one mention of the EHO and not also one of "edge harmonic": a lexicon
    that spells out its own acronym would otherwise count double, and the
    count is what the confidence is made of.

    Cached on the lexicon itself - it is frozen and hashable - because
    `hits` is called once per logbook entry, five entries a shot, and the
    glued lists do not change between calls. Small `maxsize`: a process
    uses one lexicon, or two while a test compares them.
    """
    out = []
    for p in lexicon.phenomena:
        glued = [(_glue(a), a) for a in p.aliases]
        glued.sort(key=lambda pair: (-len(pair[0]), pair[1]))
        out.append((p.id, glued, [_glue(n) for n in p.negatives]))
    return tuple(out)


def hits(text: str, lexicon: Lexicon) -> dict[str, list[Hit]]:
    """Every phenomenon named in `text`, by id, in the order it was named.

    At most one hit per phenomenon per sentence - `n_pos` counts MENTIONS,
    which is sentences, not the number of ways the lexicon happens to spell
    the thing. Phenomena with no hit are absent from the mapping rather
    than present with an empty list.
    """
    matchers = _matchers(lexicon)
    out: dict[str, list[Hit]] = {}
    for sentence in sentences(text):
        glued = _glue(sentence)
        for pid, aliases, negatives in matchers:
            found = next((a for g, a in aliases if g in glued), None)
            if found is None:
                continue
            polarity = "neg" if any(n in glued for n in negatives) else "pos"
            out.setdefault(pid, []).append(
                Hit(phenomenon=pid, alias=found, sentence=sentence,
                    polarity=polarity)
            )
    return out
