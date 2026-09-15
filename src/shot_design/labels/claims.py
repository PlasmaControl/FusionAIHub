"""What the operators said about a phenomenon, as a claim -- never as a label.

Plan section 2: "Text is never a label by itself." A logbook sentence is evidence about a
phenomenon only once three things are recorded with it, and this module's job is those three:

* **polarity** `{pos, neg, uncertain}` -- "no EHO observed" is not weak evidence for an EHO, and a
  hedge ("maybe a tearing mode") is not an assertion. The negation and mitigation guards are
  `shotdb.text`'s, reused per match rather than copied: they were measured against the real
  53,179-record logbook (clause bound, 16-character window, the "no CAUSE = no EFFECT" chains
  chief operators write), and a second copy would drift from the first correction onwards.
* **temporality** `{observed, planned, historical}` -- a pre-shot entry's "plan to get QH next
  shot" is an intention, and "like shot 190090 the EHO" is about a different discharge. Both read
  as a report of this shot if nothing separates them. A logbook entry also carries whole *spans*
  about another discharge, headed by "Last shot:" or "Next shot:" on a line of their own, and
  those are dated by their heading rather than sentence by sentence: `shotdb.text`'s
  `other_shot_spans()` draws the boundary, the same one `verdict()` grades by.
* **scope** `{shot, run}` -- Appendix C item 12: 2,302 of 22,950 shots have no shot table row of
  their own and their bundle carries only the session's text. That text is a claim about the run.
  So is everything before the bundle's shot-specific marker, on every shot.

The phenomenon names and their aliases are labeler's `events/lexicons.yaml` where it exists, so
that the two packages agree on what "eho" means; until it lands from the labeler workstream the
fallback is the `themes:` block of `configs/shot_design/labels.yaml`, whose ids are curation themes
(`qh_mode`, `tearing_mhd`) rather than phenomena. `load_lexicon` records which file it read and
the join's manifest carries it, so a table can always be traced to the vocabulary that made it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .. import config
from ..shotdb import text as text_mod

#: `text_claims.parquet`, in this order (plan section 5.5).
CLAIMS_DTYPES: dict[str, str] = {
    "shot": "int32",
    "phenomenon": "object",
    "polarity": "object",
    "temporality": "object",
    "snippet": "object",
    "scope": "object",
}
CLAIMS_COLUMNS: tuple[str, ...] = tuple(CLAIMS_DTYPES)

POLARITIES = ("pos", "neg", "uncertain")
TEMPORALITIES = ("observed", "planned", "historical")
SCOPES = ("shot", "run")

#: How a span headed by another shot's marker is dated. `shotdb.text._OTHER_SHOT`'s four
#: backward markers put the text behind us; "next shot:" puts it ahead. Neither is `observed`:
#: nothing under either heading is a report of this discharge, which is the whole reason the span
#: is dated by its heading and not by whatever verbs its sentences happen to use.
_SPAN_TEMPORALITY: dict[str, str] = {
    "last": "historical", "previous": "historical", "prev": "historical", "prior": "historical",
    "next": "planned",
}

#: How long a stored sentence may be. The snippet is what a reader is shown next to a hit, not the
#: record: the bundle keeps the whole text.
SNIPPET_CHARS = 240

# A sentence ends at .!? followed by space, or at a newline: the logbook is half prose and half
# bulleted lines, and both have to break. Deliberately coarser than `text._CLAUSE_BREAK`, which
# splits *clauses* -- the temporality cue and the phenomenon it governs routinely sit in different
# clauses of one sentence ("Plan to get QH next shot").
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_WS = re.compile(r"[ \t]+")

# Planning cues: the plan's own list, plus the purpose clause the corpus writes an installed
# protection with ("... dud trip **to avoid** the tearing mode"). The purpose clause is here at
# sentence scope AND applied per match through `text.mitigated()` below, because the corpus writes
# it in both orders -- purpose first ("to avoid the tearing mode") and purpose last ("the tearing
# mode ... to avoid it") -- and only the second is a forward cue from the mention.
_PLANNED = re.compile(
    r"(?i)\b(?:plan(?:s|ned|ning)?|will|shall|going to|try(?:ing)? to|aim(?:s|ing)?|goal"
    r"|intend(?:s|ing)?|next shot|this shot:|request(?:s|ed|ing)?"
    r"|to (?:avoid|prevent|preclude|mitigate|protect against|guard against))\b"
)
# Back-references to another discharge, *inside* one sentence. `text.other_shot_spans()` handles
# the other shape, a heading over several lines of prose; here they are kept rather than deleted,
# because a claim about a previous shot is a real claim -- about a previous shot.
_HISTORICAL = re.compile(
    r"(?i)\b(?:like shot|as in|repeat(?:ed|ing)? (?:of|from)?|previous(?:ly)?|last shot"
    r"|prior shot|earlier shot|same as shot)\b"
)
# Hedges. A hedged mention is `uncertain`: the operator wrote down a suspicion, and promoting it
# to an assertion is exactly how a text hit becomes a fake label.
_HEDGE = re.compile(
    r"(?i)(?:\b(?:maybe|perhaps|possibly|probably|might|unclear|not sure|seems?|appears?"
    r"|apparently|suspect(?:ed)?|hard to tell|some evidence|looks? like|think|believe|likely"
    r"|presumably|possible)\b|\?)"
)


@dataclass(frozen=True)
class Lexicon:
    """Phenomenon ids to alias lists, and the file they came from."""

    source: Path
    aliases: dict[str, tuple[str, ...]]
    exclude: dict[str, tuple[str, ...]]
    patterns: dict[str, re.Pattern[str]]
    excludes: dict[str, re.Pattern[str] | None]


# ----------------------------------------------------------------------------------- lexicon


def labelmaker_lexicon_path() -> Path:
    """Where labeler's `events/lexicons.yaml` is, whether or not it exists yet."""
    from labeler import events as lm_events

    return Path(lm_events.__file__).resolve().parent / "lexicons.yaml"


def default_lexicon_path() -> Path:
    """labeler's lexicon if it is on this branch, else shot_design's own themes.

    Single source, one direction: shot_design reads labeler's aliases, never the other way round.
    """
    path = labelmaker_lexicon_path()
    return path if path.exists() else config.CONFIG_DIR / "labels.yaml"


def _alias_lists(doc: Mapping) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """The three shapes a lexicon file is written in, as `(aliases, exclude)`.

    * labeler's / `phenomena.yaml`'s: `phenomena: {id: {aliases: [...], exclude: [...]}}`,
    * the same mapping at the top level, or with a bare list instead of the `aliases:` key,
    * shot_design's `labels.yaml`: `themes: [{id: ..., keywords: [...]}]`.

    Tolerant on purpose: `events/lexicons.yaml` lands from the parallel labeler workstream and
    this side must read it the day it appears, not the day someone notices the key is spelled
    differently. A document that is none of these raises rather than yielding no phenomena, which
    would look exactly like "the operators said nothing".
    """
    aliases: dict[str, list[str]] = {}
    exclude: dict[str, list[str]] = {}
    if doc.get("themes"):
        for theme in doc["themes"]:
            aliases[str(theme["id"])] = [str(k) for k in theme.get("keywords") or []]
        return aliases, exclude
    block = doc.get("phenomena") or {
        k: v for k, v in doc.items() if isinstance(v, dict | list) and k != "version"
    }
    if not block:
        raise ValueError("lexicon has no `phenomena:`, no `themes:` and no top-level entries")
    for name, entry in block.items():
        if isinstance(entry, list):
            aliases[str(name)] = [str(a) for a in entry]
        elif isinstance(entry, dict):
            aliases[str(name)] = [str(a) for a in entry.get("aliases") or []]
            exclude[str(name)] = [str(a) for a in entry.get("exclude") or []]
        else:
            # ValueError, not TypeError, despite ruff's TRY004, for the reason
            # `labeler.models.registry.parse_card` gives at the same choice: `doc` came out of
            # a YAML file, so this is a malformed *data file*, not a caller passing the wrong
            # type, and the sibling branch above raises ValueError for the same category.
            raise ValueError(  # noqa: TRY004
                f"lexicon entry {name!r} is a {type(entry).__name__}, "
                "expected a list of aliases or a mapping with `aliases:`"
            )
    return aliases, exclude


def _pattern(words: Iterable[str]) -> re.Pattern[str] | None:
    """One alternation, longest alias first so `qh-mode` wins over `qh`.

    `(?<!\\w)`/`(?!\\w)` rather than `\\b`, because a few aliases are not words -- `labels.yaml`
    writes ` nt ` with its own spaces and `n=3` with punctuation, and `\\b` is defined against the
    character next to it rather than against the alias.
    """
    got = sorted({w.strip().lower() for w in words if w and w.strip()}, key=lambda w: (-len(w), w))
    if not got:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(w) for w in got) + r")(?!\w)", re.IGNORECASE)


def load_lexicon(path=None) -> Lexicon:
    """The alias lists, from `path` or from `default_lexicon_path()`."""
    path = Path(path) if path is not None else default_lexicon_path()
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    aliases, exclude = _alias_lists(doc)
    ordered = {
        name: tuple(sorted({a.strip().lower() for a in words if a.strip()},
                           key=lambda w: (-len(w), w)))
        for name, words in aliases.items()
    }
    return Lexicon(
        source=path,
        aliases=ordered,
        exclude={k: tuple(v) for k, v in exclude.items()},
        patterns={name: p for name, words in aliases.items() if (p := _pattern(words))},
        excludes={name: _pattern(words) for name, words in exclude.items()},
    )


# ------------------------------------------------------------------------------- the sources


def _bundle_path(text_root: Path, shot: int) -> Path:
    """The per-shot text bundle, at `paths.yaml`'s own layout under `text_root`."""
    return text_root / "shotsummary" / "processed" / "per_shot_txt" / f"shot_{shot}.txt"


def _log_texts(text_root: Path, shots: list[int]) -> dict[int, str]:
    """`shot -> log_text` for the wanted shots, in one pass over `sql/logs.jsonl`.

    Matched on the raw bytes with `text._SHOT_PREFIX`, which is how `build_logs_subset` reads the
    same 616 MB file: only the wanted lines are ever parsed.
    """
    path = text_root / "sql" / "logs.jsonl"
    if not path.exists():
        return {}
    wanted, out = set(shots), {}
    with open(path, "rb") as fh:
        for line in fh:
            m = text_mod._SHOT_PREFIX.match(line)
            if m and int(m.group(1)) in wanted:
                rec = json.loads(line)
                out[int(rec["shot"])] = text_mod._clean(rec.get("log_text")) or ""
    return out


def _dated_spans(body: str) -> list[tuple[str, str | None]]:
    """One logbook entry as `(text, temporality)` pieces.

    A span headed by another shot's marker carries that heading's date; everything else is undated
    here and each sentence says for itself. Without this, "Last shot:" followed by two lines of
    prose yields `observed` rows on *this* shot for what the previous one did -- the entry is this
    shot's, but those sentences are not.
    """
    spans = text_mod.other_shot_spans(body)
    if not spans:
        return [(body, None)]
    out: list[tuple[str, str | None]] = []
    cut = 0
    for start, end, which in spans:
        out.append((body[cut:start], None))
        out.append((body[start:end], _SPAN_TEMPORALITY.get(which, "historical")))
        cut = end
    out.append((body[cut:], None))
    return [(text, when) for text, when in out if text.strip()]


def _segments(
    text_root: Path, shot: int, logs: Mapping[int, str]
) -> list[tuple[str, str, str | None]]:
    """`(text, scope, temporality)` for everything written about one shot.

    `temporality` is the section's own, where the section has one, and None where the sentence has
    to say. `text.bundle_blocks` gives the three:

    * the **session** block -- the run day's summaries -- is a record of the run: run scope, and
      each sentence dated on its own words;
    * the **mini-proposal** is by construction a statement of intent, written before the run day:
      run scope, `planned`, whatever verbs it happens to use. Its sentences are the ones a
      consumer must not read as "this run saw an EHO";
    * the **shot-specific** block is this discharge's -- unless the summary page had no row for it
      and the block is the `SHOT_TABLE_MISSING` sentinel, in which case nothing in the bundle is
      the shot's and the whole thing is the run's (Appendix C item 12).

    The logbook entries are this shot's by definition, so they are shot scope -- but an entry is
    not uniformly *about* this shot: `_dated_spans` splits off the spans headed by another
    discharge's marker and dates those by the heading.
    """
    out: list[tuple[str, str, str | None]] = []
    path = _bundle_path(text_root, shot)
    if path.exists():
        session, planned, specific = text_mod.bundle_blocks(
            path.read_text(encoding="utf-8", errors="replace")
        )
        out.append((session, "run", None))
        out.append((planned, "run", "planned"))
        scope = "run" if text_mod.SHOT_TABLE_MISSING in specific else "shot"
        out.append((specific, scope, None))
    for entry in text_mod.parse_log_entries(logs.get(shot)):
        out.extend((body, "shot", when) for body, when in _dated_spans(entry.text))
    return out


# -------------------------------------------------------------------------------- the claims


def _snippet(sentence: str) -> str:
    got = _WS.sub(" ", sentence).strip()
    return got if len(got) <= SNIPPET_CHARS else got[: SNIPPET_CHARS - 1].rstrip() + "…"


def _polarity(sentence: str, start: int, end: int) -> str:
    """`neg` if the mention is negated or named as the thing a plan prevents, `uncertain` if
    hedged, else `pos`.

    A mitigation is a negation: "dud trip **to avoid** the tearing mode" says the mode is what the
    shot is arranged not to have. The two cues are the same statement written in the two orders the
    corpus uses -- purpose first, which `text.negated()` already catches because `avoid` is in its
    vocabulary, and purpose last, which is `text.mitigated()` -- so they must give the same
    polarity or one sentence would read differently from its own paraphrase. What keeps the pair
    out of "this shot had no tearing mode" is the `planned` temporality beside it, which both
    orders also agree on.

    Negation beats hedging: "no EHO, maybe next time" denies the EHO, it does not hedge it.
    """
    if text_mod.negated(sentence, start, end) or text_mod.mitigated(sentence, start, end):
        return "neg"
    if _HEDGE.search(sentence):
        return "uncertain"
    return "pos"


def _temporality(sentence: str, start: int, end: int) -> str:
    """`planned` beats `historical` beats `observed`. "Repeat last shot but plan to avoid the
    tearing mode" is a plan that happens to mention a previous shot, and a plan is the stronger
    statement about what the sentence is doing."""
    if _PLANNED.search(sentence) or text_mod.mitigated(sentence, start, end):
        return "planned"
    if _HISTORICAL.search(sentence):
        return "historical"
    return "observed"


def empty_claims() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in CLAIMS_DTYPES.items()})


def _frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return empty_claims()
    df = pd.DataFrame(rows, columns=list(CLAIMS_COLUMNS)).astype(CLAIMS_DTYPES)
    return df.drop_duplicates().reset_index(drop=True)


def text_claims(shots: Iterable[int], *, text_root, lexicon_path=None) -> pd.DataFrame:
    """One row per (sentence, phenomenon) the operators wrote about these shots.

    A shot with no bundle and no logbook record makes no rows: silence is not a claim, and a shot
    that nobody wrote about must not be readable as a shot where nothing happened.
    """
    text_root = Path(text_root)
    shots = sorted({int(s) for s in shots})
    if not shots or not text_root.exists():
        return empty_claims()
    lexicon = load_lexicon(lexicon_path)
    logs = _log_texts(text_root, shots)
    rows: list[dict] = []
    for shot in shots:
        for body, scope, section_temporality in _segments(text_root, shot, logs):
            for sentence in _SENTENCE.split(body):
                if not sentence.strip():
                    continue
                for name, pattern in lexicon.patterns.items():
                    veto = lexicon.excludes.get(name)
                    if veto is not None and veto.search(sentence):
                        continue
                    hit = pattern.search(sentence)
                    if hit is None:
                        continue
                    rows.append({
                        "shot": shot,
                        "phenomenon": name,
                        "polarity": _polarity(sentence, hit.start(), hit.end()),
                        "temporality": section_temporality
                        or _temporality(sentence, hit.start(), hit.end()),
                        "snippet": _snippet(sentence),
                        "scope": scope,
                    })
    return _frame(rows)


__all__ = [
    "CLAIMS_COLUMNS",
    "CLAIMS_DTYPES",
    "POLARITIES",
    "SCOPES",
    "SNIPPET_CHARS",
    "TEMPORALITIES",
    "Lexicon",
    "default_lexicon_path",
    "empty_claims",
    "labelmaker_lexicon_path",
    "load_lexicon",
    "text_claims",
]
