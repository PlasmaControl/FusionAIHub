"""What the operators wrote, as evidence - and only ever as weak evidence.

DIII-D's logbook is the only place some phenomena are ever named: nobody
runs a detector for a fishbone, but somebody typed "fishbones on this one"
at 3 pm. `lexicons.yaml` is the list of what they call each thing, and this
module is the matcher over the per-shot text bundles.

**Text is never a label by itself** (plan 2). A lexicon hit is a claim that
a human wrote a word, not that the phenomenon happened: the same sentence
may be a plan for the next shot, a complaint about a diagnostic, or a note
about a different shot entirely. So a hit's confidence is capped at
`TEXT_ONLY_CEILING` for a single mention and reaches 1.0 only after four,
and its place in the pipeline is to PRIORITISE - which chunks a human is
asked to annotate, which shots ideate ranks - never to decide.

Two scopes, and the difference matters. A bundle is a run's session text,
the marker line `## Shot-specific context (from summary.html)`, and then
the shot's own block. The session text is shared by every shot of the run,
so a sawtooth mentioned there is evidence about the RUN; attaching it to
this shot attaches it to the twenty others in the same session too
(Appendix C item 12). `weak_labels` will read either and says which in the
`scope` column; `text_events`, which writes rows against one shot, reads
the shot's own block and nothing else.

The matching is deliberately dumb - space-glued phrases, no stemming, no
model - because the failure mode of a clever matcher here is a confident
wrong claim, and the failure mode of this one is a missed mention that the
detectors were going to have to find anyway. See A7 "text word-boundary".
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ..config import Paths
from .schema import Event

#: Plan 5.6's round-1 ids. The lexicon may not name anything else: a
#: phenomenon id is a join key against labels, events and ideate's
#: `phenomena.yaml`, and a typo that loads silently is a phenomenon that
#: quietly has no evidence.
PHENOMENON_IDS = (
    "ae", "eho", "elm", "tearing", "sawtooth", "fishbone", "qcm", "qh",
    "lh", "detachment", "pickup", "rwm",
)

#: The alias lists themselves. Shipped beside this module because ideate
#: reads it too - one file, two readers.
LEXICON_PATH = Path(__file__).with_name("lexicons.yaml")

#: The line that separates a run's session text from this shot's own.
MARKER = "## Shot-specific context (from summary.html)"

#: The header of the `- KEY: value` block at the end of a shot's block.
TABLE_HEADER = "SHOT TABLE ROW"

#: What one mention is worth, and so what a single mention is capped at.
#: Four independent mentions reach 1.0; nothing else about text does.
TEXT_ONLY_CEILING = 0.25

#: Where a hit's text came from. `run` is the session text every shot of
#: the run shares; `shot` is this shot's own block.
SCOPES = ("shot", "run")

#: How much of the first positive sentence a row carries.
SNIPPET_CHARS = 200

#: Column order of a `weak_labels` frame.
COLUMNS = ("shot", "phenomenon", "n_pos", "n_neg", "scope", "snippet")

#: Dtype of every column; `object` is pandas' dtype for python strings.
DTYPES = {
    "shot": "int32",
    "phenomenon": "object",
    "n_pos": "int32",
    "n_neg": "int32",
    "scope": "object",
    "snippet": "object",
}

#: What ends a sentence: `. ! ? ;` and the newline.
_TERMINATOR = re.compile(r"[.!?;\n]+")

#: Everything that is NOT part of a word. `\w` keeps letters, digits and
#: the underscore; the hyphen and the slash are kept too, so that
#: "elm-free" is one word and not an ELM and "2/1" is a mode number and not
#: a 2 and a 1. The underscore is kept for the same reason in reverse: the
#: session text is full of control-system parameter names, and
#: `RWM_GAINMULT` is an identifier rather than somebody saying an RWM
#: happened.
_PUNCTUATION = re.compile(r"[^\w\-/]+")

#: Stripped from a token's ENDS, where they are punctuation after all - a
#: trailing dash, a bare slash between two spaces.
_IN_TOKEN = "-/"


class LexiconError(ValueError):
    """The lexicon file says something `text_weak` cannot use."""


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
    """Every phenomenon the text layer knows how to look for."""

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
    """
    path = Path(LEXICON_PATH if path is None else path)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except OSError as exc:
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


# --------------------------------------------------------- reading a bundle

def _read(shot: int, root=None) -> str:
    """One shot's whole text bundle, or "" where there is no file.

    Missing is ordinary: the corpus has text for about 13,000 of the 17,000
    shots, and a caller joining text to features must not have to care.
    """
    paths = Paths.from_env() if root is None else Paths(text_root=Path(root))
    path = paths.text_file(int(shot))
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _split(text: str) -> tuple[str, str]:
    """`(run context, shot block)` either side of `MARKER`.

    No marker means no shot block - not "the whole file is the shot's" -
    because the run's text is the thing that is always there. Both sides
    are stripped, so that a bundle which ENDS at the marker - the
    session-fallback case, where the summary had nothing for this shot -
    reads as an empty block rather than as a newline.
    """
    head, sep, tail = text.partition(MARKER)
    return (head.strip(), tail.strip()) if sep else (text.strip(), "")


def run_context(shot, *, root=None) -> str:
    """The session text of the run this shot belongs to."""
    return _split(_read(shot, root))[0]


def shot_block(shot, *, root=None) -> str:
    """This shot's own text: everything after the marker.

    `""` for a shot with no file, and for the session-fallback case where
    the summary carried nothing for the shot and the bundle ends at the
    marker.
    """
    return _split(_read(shot, root))[1]


def shot_table_row(text: str) -> dict[str, str]:
    """The `- KEY: value` block under `SHOT TABLE ROW`, as a mapping.

    Values are left as strings, and split on the FIRST colon: a
    `TIME-OF-SHOT` of `10:08` is one value and not a second key. `{}` where
    the block is missing, which is the session-fallback case and the
    "(Shot table key/value mapping not found.)" one.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(TABLE_HEADER):
            break
    else:
        return {}
    out: dict[str, str] = {}
    for line in lines[i + 1:]:
        if not line.strip():
            continue
        if not line.startswith("- "):
            break
        key, sep, value = line[2:].partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


# ------------------------------------------------------------ the matching

def sentences(text: str) -> list[str]:
    """Lowercased, whitespace-collapsed sentences.

    Split on `. ! ? ;` and on newlines, because a logbook entry is as often
    a list of lines as it is prose. The sentence is the unit the matcher
    works in: it is the span over which a negation ("no elms") plausibly
    applies, and small enough that two phenomena named in one are named
    together.
    """
    return [
        s for s in (" ".join(part.split()).lower()
                    for part in _TERMINATOR.split(text))
        if s
    ]


def _glue(sentence: str) -> str:
    """The sentence as `" word word "`, for whole-phrase containment.

    Every character that is not part of a word becomes a space, EXCEPT a
    hyphen or a slash inside a token: "elm-free" is one token and so is
    "2/1", while a trailing dash or a comma is not part of the word before
    it. Wrapping the result in spaces is what makes `" nt " in " we want "`
    false - the whole point of the exercise.
    """
    flat = _PUNCTUATION.sub(" ", sentence.lower())
    tokens = (token.strip(_IN_TOKEN) for token in flat.split())
    return f" {' '.join(t for t in tokens if t)} "


def _matchers(lexicon: Lexicon):
    """Per phenomenon, its glued aliases longest first and its negatives.

    Longest first so that a sentence saying "edge harmonic oscillation" is
    one mention of the EHO and not also one of "edge harmonic": a lexicon
    that spells out its own acronym would otherwise count double, and the
    count is what the confidence is made of.
    """
    out = []
    for p in lexicon.phenomena:
        glued = [(_glue(a), a) for a in p.aliases]
        glued.sort(key=lambda pair: (-len(pair[0]), pair[1]))
        out.append((p.id, glued, [_glue(n) for n in p.negatives]))
    return out


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


# -------------------------------------------------------------- the outputs

def _empty_frame() -> pd.DataFrame:
    """No hits, right columns, right dtypes."""
    return pd.DataFrame({name: pd.Series(dtype=DTYPES[name]) for name in COLUMNS})


def _counts(found: Mapping[str, list[Hit]]) -> list[tuple[str, int, int, str]]:
    """`(phenomenon, n_pos, n_neg, snippet)` per phenomenon, id order."""
    rows = []
    for pid in sorted(found):
        hs = found[pid]
        pos = [h for h in hs if h.polarity == "pos"]
        snippet = pos[0].sentence[:SNIPPET_CHARS] if pos else ""
        rows.append((pid, len(pos), len(hs) - len(pos), snippet))
    return rows


def weak_labels(shots: Iterable[int], lexicon: Lexicon, *, root=None,
                scope: str = "shot") -> pd.DataFrame:
    """One row per (shot, phenomenon) named in the shots' text.

    `scope="shot"` reads each shot's own block; `scope="run"` reads the
    session text it shares with the rest of its run, which is evidence
    about the run and is labelled as such - a consumer that treats a run
    row as a shot row has claimed the same sentence for every shot of the
    session.

    One file read per shot either way, and no row at all for a phenomenon
    nobody mentioned: an absent row is "not mentioned", which is not the
    same claim as `n_pos == 0`.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}; got {scope!r}")
    rows = []
    for shot in shots:
        run, block = _split(_read(shot, root))
        text = block if scope == "shot" else run
        for phenomenon, n_pos, n_neg, snippet in _counts(hits(text, lexicon)):
            rows.append(
                {
                    "shot": int(shot),
                    "phenomenon": phenomenon,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "scope": scope,
                    "snippet": snippet,
                }
            )
    if not rows:
        return _empty_frame()
    return (
        pd.DataFrame(rows, columns=list(COLUMNS))
        .sort_values(["shot", "phenomenon"], kind="stable")
        .reset_index(drop=True)
        .astype(DTYPES)
    )


def _pulse_length_s(table: Mapping[str, str]) -> float:
    """`PULSE-LENGTH` in seconds, or 0.0 where the table does not say.

    The table row is the only shot duration the text bundle carries, and it
    is a string somebody's HTML scraper produced: "(none)" and "" both
    happen. A text claim is about the whole shot, so a shot whose length is
    unknown gets the point event at 0 - `Event` requires finite times, and
    inventing a span would be inventing a claim about when.
    """
    try:
        value = float(table.get("PULSE-LENGTH", ""))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


def text_events(shot, lexicon: Lexicon, *, root=None) -> list[Event]:
    """One `evidence_kind="text"` event per phenomenon this shot's text names.

    The span is the whole shot - text says THAT, almost never when - so
    `[0, PULSE-LENGTH)` from the table row, and a point at 0 where the
    table does not give one. Coverage is the same span: the text was read
    for the whole shot, so "no `eho` row" means nobody wrote it, not that
    nobody looked.

    Only the shot's own block is read. The session text is the run's
    (Appendix C item 12); `weak_labels(..., scope="run")` is where it goes,
    and a consumer that wants to fold it in has to decide what a run-scope
    claim is worth for one of its twenty shots.
    """
    block = _split(_read(shot, root))[1]
    span = _pulse_length_s(shot_table_row(block))
    out = []
    for phenomenon, n_pos, n_neg, snippet in _counts(hits(block, lexicon)):
        if n_pos < 1:
            continue
        out.append(
            Event(
                shot=int(shot),
                source="text",
                evidence_kind="text",
                phenomenon=phenomenon,
                t0_s=0.0,
                t1_s=span,
                confidence=min(1.0, TEXT_ONLY_CEILING * n_pos),
                attrs={
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "snippet": snippet,
                    "scope": "shot",
                },
                t_cov0_s=0.0,
                t_cov1_s=span,
            )
        )
    return out
