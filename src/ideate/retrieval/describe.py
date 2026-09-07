"""One paragraph per shot, written by a template, optionally polished by a local LLM.

The template is the source of truth. `describe` renders only fields the record actually has --
a missing number is an omitted clause, never the string "None" and never a plausible default --
and it is the ONE renderer: `ideate show` prints `segment_line`, `ideate export` stores
`describe`, and every `ideate query` result carries `describe`. Units and k/M prefixes come from
`rank.display`, which reads them from configs/ideate/, so a physicist reading a search result and a
physicist reading `ideate show` are reading the same numbers in the same units.

`polish` is the seam where a language model may rewrite that paragraph, and `check_facts` is the
gate it has to pass: every number and every shot number in the candidate must match the template
as a multiset, in both directions, or the template is returned unchanged -- a fluent paraphrase
that quietly changes 1.21 MA into 1.2 MA is worse than no paraphrase at all. The call goes
through `ideate.llm.client`, which opens no socket when `configs/ideate/llm.yaml` says
`provider: off` and none when no endpoint file has been published; with no model running the
template is simply the answer, silently.

**The operator quote is the fabrication risk in this file.** It is taken from
`HumanTier.log_entries` -- the output of `text.parse_log_entries`, one entry by one author at one
time -- and never from the concatenated log text, which would splice sentences written by
different people into a single quotation. `_quote` returns a whitespace-normalised *prefix of a
single entry* and nothing else; there is no path through this module that can join two entries.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import html
import math
import re
from collections import Counter

from .. import schema
from .rank import display, units

# Roles worth quoting, best first. PCS/RF/DIAGNOSTICS/BEAMS entries are settings dumps and
# tab-separated status tables (measured across the database: every RF entry is a gyrotron table),
# so they are not quoted at all rather than truncated into something that reads like a sentence.
QUOTE_ROLES = ("PHYSICS_OPERATOR", "SESSION_LEADER", "CHIEF_OPERATOR")
MAX_QUOTE = 140
_MIN_SENTENCE = 40  # do not cut to a sentence boundary shorter than this; word-cut instead

# The preshot template ("Preshot:\n<precomment>\n- - - -\nrequested ip: 0 MA, ...") is a form,
# not commentary, and its dashed divider and requested line are what identify it.
_NOT_COMMENTARY = ("\t", "requested ip:", "- - - -", "pcs changes:")
_WS = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")

# Pinned by the plan. Numbers are compared as a multiset, so 1.21 appearing twice in the template
# and once in a candidate is a mismatch; shot references are pulled out separately because a
# 6-digit shot number is a fact about *which discharge*, not a quantity, and losing one is the
# single worst thing a paraphrase can do.
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?")
# The ONE shot-number pattern: `ui/chat.py` marks unverified shots with this same object, so the
# fact check and the chat can never disagree about what looks like a shot. Both bands are needed
# -- the database spans 160904-204988, and a `1\d{5}`-only pattern was blind to the 24 shots of
# poc_v1 at or above 200000.
SHOT_REF = re.compile(r"\b[12]\d{5}\b")
_SHOT_REF = SHOT_REF  # the older private name, kept working for callers that already import it

REGIME_NAMES = {
    "L": "L-mode",
    "H": "H-mode",
    "QH": "QH-mode",
    "neg_tri": "negative triangularity",
}
END_REASONS = {
    "programmed_rampdown": "programmed ramp-down",
    "fast_current_quench": "fast current quench",
    "early_termination": "early termination",
    "no_plasma": "no plasma",
    "ip_signal_unusable": "Ip signal unusable",
}


def _unit(name: str) -> str:
    """The registry's unit for a quantity `display` cannot prefix, shortened to `[?]` when it is
    honestly unconfirmed: "9.82e+13 [?]" in a one-line description rather than the registry's
    full "[?] line-integrated (node declares V)", which `ideate show --full` still prints."""
    u = units().get(name, "")
    return "[?]" if u.startswith("[?]") else u


def _f(x: float | None) -> float | None:
    """None for anything that is not a real number -- NaN included, since a Parquet round trip
    turns an unrecorded scalar into NaN and "nan MW" on a demo screen is worse than silence."""
    if x is None:
        return None
    v = float(x)
    return None if math.isnan(v) else v


def _members_on(vals: dict[str, float | None], prefix: str, stat: str) -> list[str]:
    """Which members of an actuator system actually ran, in registry order."""
    out = []
    for key, v in vals.items():
        if not key.startswith(f"{prefix}_") or not key.endswith(f"_{stat}"):
            continue
        member = key[len(prefix) + 1 : -len(stat) - 1]
        if member != "total" and (_f(v) or 0.0) > 0:
            out.append(member)
    return out


def _system(vals: dict[str, float | None], prefix: str, label: str) -> str | None:
    """ "PNBI 8.2 MW (15L 15R 30L)", or "PECH off", or nothing at all.

    Three states, not two. `pech_total_mean` is None both when the system idled (features.py
    averages an actuator over its positive samples, and an idle system has none) and when nothing
    was recorded; `pech_total_on_frac` is 0.0 for the first and None for the second, which is the
    only thing that separates "the gyrotrons were off" from "we do not know".
    """
    total = _f(vals.get(f"{prefix}_total_mean"))
    if total is None:
        return f"{label} off" if _f(vals.get(f"{prefix}_total_on_frac")) == 0.0 else None
    on = _members_on(vals, prefix, "mean")
    return f"{label} {display(f'{prefix}_total_mean', total)}" + (
        f" ({' '.join(on)})" if on else ""
    )


def _provenance(rec: schema.ShotRecord, name: str) -> str:
    p = rec.derived_provenance.get(name)
    if p is None:
        return ""
    bits = [f"{p.tool}{p.version or ''}"] + (["assumed"] if p.assumed else [])
    return f" ({', '.join(bits)})"


def _header(rec: schema.ShotRecord) -> str:
    h = rec.human
    inner = [str(rec.shot_date) if rec.shot_date else "", f"run {h.run_id}" if h.run_id else ""]
    if h.mpid or h.mp_title:
        mp = f"MP {h.mpid}" if h.mpid else "MP"
        inner.append(f'{mp} "{h.mp_title}"' if h.mp_title else mp)
    kept = [b for b in inner if b]
    return f"Shot {rec.shot}" + (f" ({', '.join(kept)})" if kept else "") + "."


def segment_line(rec: schema.ShotRecord, segment: str) -> str | None:
    """One segment's headline numbers: "Flat top 0.53-5.20 s: Ip 985 kA, Bt 1.95 T, PNBI 8.21 MW
    (15L 15R ...), PECH off, betaN 2.28 (EFIT01, assumed), kappa 1.81, ...".

    Every number goes through `rank.display`, so the unit and its k/M prefix come from the
    registry and the magnitude -- the same call the query's `similar`/`differs` lines make. This
    used to hardcode `/1e6 -> MA` and `/1e6 -> MW` alongside a second copy in cli.py with
    different precisions (`q95 {:.1f}` here, `q95 {:.2f}` there), so `show` and `query` printed
    the same shot differently. `ideate show` prints this line as its headline.
    """
    seg = rec.segment(segment)  # type: ignore[arg-type]
    if seg is None:
        return None
    vals: dict[str, float | None] = {**seg.raw, **seg.derived}
    name = segment.replace("_", " ").capitalize()
    bits: list[str] = []
    for col, label in (("ip_mean", "Ip"), ("bt_mean", "Bt")):
        v = _f(vals.get(col))
        if v is not None:
            bits.append(f"{label} {display(col, v)}")
    for prefix, label in (("pnbi", "PNBI"), ("pech", "PECH")):
        s = _system(vals, prefix, label)
        if s:
            bits.append(s)
    for col, label in (("q95_mean", "q95"), ("betan_mean", "betaN")):
        v = _f(vals.get(col))
        if v is not None:
            bits.append(f"{label} {display(col, v)}" + _provenance(rec, col.removesuffix("_mean")))
    v = _f(vals.get("kappa_mean"))
    if v is not None:
        bits.append(f"kappa {display('kappa_mean', v)}")
    top, bot = _f(vals.get("tritop_mean")), _f(vals.get("tribot_mean"))
    if top is not None and bot is not None:
        bits.append(f"delta {(top + bot) / 2:.3g}")
    ne = _f(vals.get("ne_line_mean"))
    if ne is not None:
        # Not through `display`: ne_line's registry unit is the honest "[?] line-integrated (node
        # declares V)", which is right in `show --full` and too long for a one-line description.
        bits.append(f"ne_line {ne:.3g} {_unit('ne_line')}".rstrip())
    if not bits:
        return None
    return f"{name} {seg.t0_ms / 1000:.2f}-{seg.t1_ms / 1000:.2f} s: " + ", ".join(bits) + "."


def _labels_line(rec: schema.ShotRecord) -> str | None:
    bits = []
    if rec.labels.regime != "unknown":
        regime = REGIME_NAMES.get(rec.labels.regime, rec.labels.regime)
        src = rec.labels.regime_source
        bits.append(f"{regime} ({src})" if src != "none" else regime)
    bits.extend(sorted(rec.labels.operational))
    return f"Labels: {', '.join(bits)}." if bits else None


def _outcome_line(rec: schema.ShotRecord) -> str | None:
    o, bits = rec.outcome, []
    for label, hit, err in (
        ("Ip", o.ip_target_hit, o.ip_target_err),
        ("NBI", o.nbi_target_hit, o.nbi_target_err),
    ):
        if hit is None:
            continue
        word = "target hit" if hit else "missed target"
        pct = None if err is None else err * 100
        # An error that rounds to zero prints as "(0 %)", not "(-0 %)": the sign of a rounded-away
        # tenth of a percent is noise, and "-0" reads like a bug on a demo screen.
        err_text = "" if pct is None else f" ({pct:+.0f} %)" if round(pct) else " (0 %)"
        bits.append(f"{label} {word}{err_text}")
    if o.end_reason:
        end = END_REASONS.get(o.end_reason, o.end_reason.replace("_", " "))
        full = rec.segment("full")
        bits.append(end + (f" at {full.t1_ms / 1000:.1f} s" if full else ""))
    if o.fault_strings:
        bits.append("faults: " + ", ".join(o.fault_strings))
    return f"Outcome: {'; '.join(bits)}." if bits else None


def _shorten(text: str) -> str:
    """A verbatim prefix of `text`, at most MAX_QUOTE characters, cut at a sentence boundary
    where there is one. Never edits the words it keeps."""
    if len(text) <= MAX_QUOTE:
        return text
    ends = [m.end() for m in _SENTENCE_END.finditer(text) if m.end() <= MAX_QUOTE]
    if ends and ends[-1] >= _MIN_SENTENCE:
        return text[: ends[-1]]
    cut = text.rfind(" ", 0, MAX_QUOTE)
    return text[: cut if cut > 0 else MAX_QUOTE].rstrip() + " ..."


def quotable(entry: schema.LogEntry) -> str | None:
    """The display text of a logbook entry worth quoting, or None for one that is not.

    The one filter for every quote on a screen: `_quote` below and the query's `text_highlight`
    (rank._highlight) both go through here, so a `[PCS] PCS CHANGES: ...` settings dump or a
    tab-separated RF status table is quoted nowhere, and the two never disagree about what an
    operator "said". (rank._highlight used to scan every entry with no role filter: on the real
    database a query for "ECH gyrotron power" highlighted a PCS dump on 91 of 105 shots.)

    Whitespace is collapsed (a logbook entry is wrapped across lines) but no word is changed,
    dropped from the middle, or joined to another entry's. HTML entities are decoded for display
    only -- the logbook is scraped from HTML and 5 of the 105 built shots carry one, so shot
    161584's session leader renders as "didn't get ECH" rather than "didn&#39;t get ECH". This is
    deliberately NOT done in text.parse_log_entries: decoding there would change verdict() and
    fault_strings(), because "n&#39;t" becomes "n't", which is inside their negation vocabulary.
    That is a scored behaviour change needing its own measured pass (task-9-report.md, "Not
    done"); a display string has no such constraint.
    """
    if entry.role not in QUOTE_ROLES:
        return None
    low = entry.text.lower()
    if any(marker in low for marker in _NOT_COMMENTARY):
        return None
    return _WS.sub(" ", html.unescape(entry.text)).strip() or None


def best_quote(rec: schema.ShotRecord) -> tuple[schema.LogEntry, str] | None:
    """The most informative single logbook entry and its display text, or None.

    One entry, one author, one timestamp -- `quotable` decides which entries qualify and how
    their text is shown. Nothing here can join two entries. A "Postshot:" note is the session
    leader saying how the shot actually went, which is what a search result wants; anything else
    from the same role is preshot intent.
    """
    best: tuple[tuple[int, int, int], schema.LogEntry, str] | None = None
    for i, e in enumerate(rec.human.log_entries):
        text = quotable(e)
        if text is None:
            continue
        postshot = e.text.lower().lstrip().startswith("postshot")
        key = (QUOTE_ROLES.index(e.role), 0 if postshot else 1, i)
        if best is None or key < best[0]:
            best = (key, e, text)
    return None if best is None else (best[1], best[2])


def _quote(rec: schema.ShotRecord) -> str | None:
    """The most informative single logbook entry, quoted as a prefix of itself.

    One entry, one author, one timestamp -- `quotable` decides which entries qualify and how
    their text is shown; `_shorten` cuts a verbatim prefix. Nothing here can join two entries.
    """
    found = best_quote(rec)
    if found is None:
        return None
    entry, text = found
    who = " ".join(b for b in (entry.role, entry.author, entry.time) if b)
    return f'Operator: "{_shorten(text)}" ({who}).'


def describe(rec: schema.ShotRecord, segment: str = "flat_top") -> str:
    """The template. One line per fact group; missing facts are omitted, never rendered.

    `segment` names which segment's numbers to print and is not silently swapped for another:
    a result that says it is about the ramp-up must not quote flat-top numbers. A record with no
    such segment simply loses that line.
    """
    lines = [
        _header(rec),
        segment_line(rec, segment),
        _labels_line(rec),
        _outcome_line(rec),
        _quote(rec),
    ]
    return "\n".join(line for line in lines if line)


def extract_facts(text: str) -> tuple[set[str], set[str]]:
    """(numbers, shot references) as they are written -- the checkable content of a description."""
    return set(_NUM.findall(text)), set(SHOT_REF.findall(text))


def check_facts(template: str, candidate: str) -> bool:
    """True when `candidate` states exactly the template's numbers and shot references.

    Multisets, both directions: a paraphrase may not add a number, drop one, round one, or repeat
    one. `extract_facts` returns sets because that is the useful shape for a caller reporting
    *which* fact differs; the check itself counts, because "1.2 MA and 1.2 T" losing one of its
    two 1.2s is exactly the kind of damage a fluent rewrite does.
    """
    return all(
        Counter(pat.findall(template)) == Counter(pat.findall(candidate))
        for pat in (_NUM, SHOT_REF)
    )


def _client_for_polish():
    """The shared client, or None when the config says off. Separate so tests can swap it."""
    from ideate.llm.client import (
        LLMClient,  # optional at import time: describe is used by the CLI
    )

    client = LLMClient()
    return None if client.off else client


def polish(text: str) -> tuple[str, bool]:
    """(text, was_polished). The template survives unless a running model returns a rewrite that
    passes check_facts -- the same numbers and shot numbers, as multisets, in both directions.
    No model, no server, any error: the template comes back unchanged and nothing is logged."""
    client = _client_for_polish()
    if client is None or not client.available()[0]:
        return text, False
    try:
        reply = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the shot description below for fluency. Keep every number, unit "
                        "and shot number exactly as written. One paragraph. No new facts."
                    ),
                },
                {"role": "user", "content": text},
            ],
            model="fast",
            max_tokens=300,
        )
    except Exception:  # noqa: BLE001 — optional operation; preserve the fallback contract
        return text, False
    candidate = reply.content.strip()
    if candidate and check_facts(text, candidate):
        return candidate, True
    return text, False
