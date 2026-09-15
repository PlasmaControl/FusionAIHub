"""The two-sentence summary that leads every result card.

Written once, offline, from the shot's own text (mini-proposal title and purpose, run title,
the shot brief, the operators' entries, the outcome), stored in shots.parquet, and gated: a
blurb may not state a number or a shot number that is absent from its source, may not contain a
quotation mark, and may not run longer than the configured word cap. It is a plain-language
summary in the model's own words, not a quotation: the blurb does not quote the operators --
the card's "more" section already shows a verbatim operator quote (`describe.best_quote`), so
faithfulness there is that function's job, not this one's. Fail the gate and the template
stands, marked so on the card. The page never waits on the model; `ideate blurb` backfills.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel

from ideate import config
from ideate.retrieval import describe
from ideate.schema import ShotRecord

_log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


def _system(max_words: int) -> str:
    """The system prompt: exactly two plain sentences, no quotes, no invented numbers.

    Owner's decision (2026-09-06): this is a find-your-shot summary, not a quoted excerpt --
    "i just want a 2 sentence summary of the shot. dont overcomplicate." So the prompt asks for
    two sentences only (what the experiment/shot set out to do, then how it went), in the
    model's own words, with no quotation marks at all -- a verbatim operator quote already has
    its own place on the card (`describe.best_quote`), so the blurb is not the place fabrication
    or splicing could hide behind a quote mark.
    """
    return (
        "You summarise DIII-D tokamak shots for physicists scanning a list of past shots to find "
        "one worth looking at. Write exactly two plain sentences: the first says what the "
        "experiment or this shot set out to do, the second says how it went. Use only the text "
        "given. Do not invent or round numbers; prefer no numbers at all. Do not use quotation "
        "marks -- write in your own words, never quote the operators. If the shot ended in a "
        "disruption, a fast current quench or was terminated early, the second sentence must say "
        "so plainly. No headings, no lists, no preamble, no more than two sentences, and never "
        f"more than {max_words} words."
    )


class Blurb(BaseModel):
    text: str
    source: Literal["llm", "template"]
    reason: str | None = None


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
    return None


def make(rec: ShotRecord, client, cfg: dict | None = None) -> Blurb:
    """The blurb for one record: the model's sentences when they pass the gate, else the template
    with the reason kept. `client` is an LLMClient or None.

    Nothing is cached here. The request is deterministic (temperature 0), so LLMClient.chat's own
    request cache already answers a rebuild without touching the model; a second cache keyed on
    (prompt version, model, source text) would additionally freeze the *gate's verdict*, so a
    tightened gate or a re-generated blurb could not change a stored answer without a hand-deleted
    directory.
    """
    bcfg = (cfg or config.load_yaml("llm.yaml"))["blurb"]
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
        )
        candidate = _norm(reply.content)
        reason = gate(src, candidate, rec, max_words)
    except Exception as exc:  # noqa: BLE001 — unavailable mid-run, malformed reply: the template stands
        candidate, reason = "", f"model error: {type(exc).__name__}"
        _log.warning("blurb for shot %s fell back: %s", rec.shot, exc)
    if reason is None:
        return Blurb(text=candidate, source="llm")
    return Blurb(text=fallback(rec), source="template", reason=reason)
