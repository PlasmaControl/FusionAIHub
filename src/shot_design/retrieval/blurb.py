"""The three-sentence summary that leads every result card.

Written once, offline, from the shot's own text (mini-proposal title and purpose, run title,
the shot brief, the operators' entries, the outcome), stored in shots.parquet, and gated: a
blurb may not state a number or a shot number that is absent from its source, write an unsupported
number word, introduce an abbreviation, contain a quotation mark, fail to contain exactly three
complete sentences, or exceed the word cap. It is a plain-language summary in the model's own words:
the blurb does not quote the operators --
the card's "more" section already shows a verbatim operator quote (`describe.best_quote`), so
faithfulness there is that function's job, not this one's. Fail the gate and the template
stands, marked so on the card. The page never waits on the model; `shot_design blurb` backfills.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel

from shot_design import config
from shot_design.retrieval import describe
from shot_design.schema import ShotRecord

_log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_NUMBER_WORDS = frozenset(
    {
        "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
        "ninety", "hundred", "thousand", "million", "dozen",
    }
)
_ACRONYM = re.compile(r"\b[A-Z]{2,6}s?\b")


def _system(max_words: int) -> str:
    """Prompt v6: v5's three-sentence contract plus verbatim abbreviations and digit quantities."""
    return (
        "You summarise DIII-D tokamak shots for physicists scanning a list of past shots to find "
        "one worth looking at. Write exactly three plain sentences: the first says what the "
        "experiment or this shot set out to do; the second says whether it succeeded "
        "(say when success is unknown); the third gives one interesting finding from the "
        "operator entries or shot brief. When nothing notable is recorded, the third sentence "
        "must be exactly: No notable findings were logged. "
        "Use only the text given. Do not invent or round numbers; prefer no numbers at all. "
        "Copy every abbreviation, acronym and symbol exactly as the text writes it (for example "
        "AE, EHO, QH, FPP, PCS, RMP, LM, li, V.s, betan). Never expand, translate or explain "
        "one, even when you think you know what it stands for, and never introduce an "
        "abbreviation the text does not use. Write a quantity with digits and the unit exactly "
        "as the text gives it, or leave it out. Never write a number as a word (not thirteen, "
        "not several hundred, not two-one). "
        "Do not use quotation "
        "marks -- write in your own words, never quote the operators. If the shot ended in a "
        "disruption, a fast current quench or was terminated early, the second sentence must say "
        "so plainly. No headings, no lists, no preamble, exactly three sentences, and never "
        f"more than {max_words} words."
    )


class Blurb(BaseModel):
    text: str
    source: Literal["llm", "template"]
    reason: str | None = None
    candidate: str | None = None  # kept for dry-run inspection, including a rejected reply


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def source_text(rec: ShotRecord) -> str:
    h = rec.human
    parts = [
        f"Shot {rec.shot}"
        + (f" ({rec.shot_date})" if rec.shot_date else "")
        + f", campaign {rec.campaign}."
    ]
    if h.mp_title:
        parts.append(f"Mini-proposal {h.mpid or ''}: {h.mp_title}".strip())
    if h.mp_purpose:
        parts.append("Purpose: " + _norm(h.mp_purpose)[:1200])
    if h.run_title:
        parts.append(f"Run {h.run_id or ''}: {h.run_title}".strip())
    if h.shot_brief:
        parts.append("Shot brief: " + _norm(h.shot_brief))
    quotes = [q for q in (describe.quotable(e) for e in h.log_entries) if q]
    if quotes:
        parts.append("Operator entries:\n" + "\n".join(f"- {q}" for q in quotes[:8]))
    if h.chief_operator_status:
        parts.append(f"Chief operator status: {h.chief_operator_status}")
    out = describe._outcome_line(rec)
    if out:
        parts.append(out)
    parts.append(f"Verdict: {h.verdict}.")
    return "\n".join(parts)


def fallback(rec: ShotRecord) -> str:
    lines = [describe._header(rec), describe._outcome_line(rec)]
    return " ".join(line for line in lines if line)


def gate(source: str, candidate: str, rec: ShotRecord, max_words: int) -> str | None:
    if not candidate.strip():
        return "empty"
    if len(candidate.split()) > max_words:
        return f"more than {max_words} words"
    if any(q in candidate for q in ('"', "“", "”")):
        return "quotation marks"
    src_nums, src_shots = describe.extract_facts(source)
    nums, shots = describe.extract_facts(candidate)
    # Shot references first: every shot number is also a number, so checking numbers first would
    # report an invented "161234" as an invented number and hide the more useful reason.
    if not set(shots) <= set(src_shots) | {str(rec.shot)}:
        return f"shot not in source: {sorted(set(shots) - set(src_shots))}"
    # extract_facts returns sets (each number counted once), so a multiset comparison here would
    # always agree with (or be stricter than) the set comparison for no extra benefit -- reduced
    # to the one check that matches what extract_facts actually hands back.
    if not set(nums) <= set(src_nums):
        return f"number not in source: {sorted(set(nums) - set(src_nums))}"
    candidate_number_words = {
        re.sub(r"^[\W_]+|[\W_]+$", "", token).lower()
        for token in re.split(r"[\s-]+", candidate)
    } & _NUMBER_WORDS
    unknown_number_words = {
        word for word in candidate_number_words
        if re.search(rf"\b{re.escape(word)}\b", source, flags=re.IGNORECASE) is None
    }
    if unknown_number_words:
        return f"number written as a word: {sorted(unknown_number_words)}"
    source_compact = source.replace("-", "").lower()
    acronym_texts = (candidate, candidate.replace("DIII-D", "DIII").replace("-", ""))
    candidate_acronyms = {
        token.removesuffix("s")
        for text in acronym_texts
        for token in _ACRONYM.findall(text)
    }
    unknown_acronyms = {
        token for token in candidate_acronyms
        if token != "DIII" and token.replace("-", "").lower() not in source_compact
    }
    if unknown_acronyms:
        return f"abbreviation not in source: {sorted(unknown_acronyms)}"
    # Count terminator groups followed by whitespace/end; decimal points stay inside words.
    # Unit abbreviations such as "kA." terminate a sentence here. No NLP dependency is needed.
    sentences = len(re.findall(r"[.!?]+(?=\s|$)", candidate))
    if sentences != 3 or not candidate.rstrip().endswith((".", "!", "?")):
        return f"expected exactly three complete sentences (found {sentences} terminators)"
    return None


def make(
    rec: ShotRecord, client, cfg: dict | None = None, *, cache: bool | None = None,
) -> Blurb:
    """The blurb for one record: the model's sentences when they pass the gate, else the template
    with the reason kept. `client` is an LLMClient or None.

    Nothing is cached here. The request is deterministic (temperature 0), so LLMClient.chat's own
    request cache already answers a rebuild without touching the model; a second cache keyed on
    (prompt version, model, source text) would additionally freeze the *gate's verdict*, so a
    tightened gate or a re-generated blurb could not change a stored answer without a hand-deleted
    directory.
    """
    bcfg = (cfg or getattr(client, "cfg", None) or config.load_yaml("llm.yaml"))["blurb"]
    if client is None or not client.available()[0]:
        return Blurb(text=fallback(rec), source="template", reason="no model")
    src = source_text(rec)
    max_words = int(bcfg["max_words"])
    try:
        reply = client.chat(
            # The prompt version rides in the prompt itself, so bumping it changes the request and
            # LLMClient's cache answers with a fresh call rather than last week's sentences.
            [
                {
                    "role": "system",
                    "content": f"{_system(max_words)}\n(prompt v{bcfg['prompt_version']})",
                },
                {"role": "user", "content": src},
            ],
            model=bcfg["model"],
            max_tokens=220,
            cache=cache,
        )
        candidate = _norm(reply.content)
        reason = gate(src, candidate, rec, max_words)
    except Exception as exc:  # noqa: BLE001 — unavailable mid-run, malformed reply: the template stands
        candidate, reason = "", f"model error: {type(exc).__name__}"
        _log.warning("blurb for shot %s fell back: %s", rec.shot, exc)
    if reason is None:
        return Blurb(text=candidate, source="llm", candidate=candidate)
    return Blurb(text=fallback(rec), source="template", reason=reason, candidate=candidate)
