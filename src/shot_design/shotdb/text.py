"""Human tier: logbook records (sql/logs.jsonl), run metadata and miniproposal text
(shotsummary/raw), operator verdicts, and MiniLM text embeddings.

Data quirks: absent fields come back as JSON null, and _clean() also treats the literal string
"None" as missing -- that spelling is a quirk of the sibling per-run shot_<N>.json export and, as
re-measured in fix wave 1, appears in 0 values of logs.jsonl's 53,179 records and 0 values of the
3,061 shotsummary metadata.json files, so the branch never fires on what this module reads and
nothing should be built on it firing. "IP min" in chief-operator notes is the normal end of a
DIII-D discharge (Ip fell below the minimum), not a fault.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import os
import re
import types
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..config import Paths, load_paths
from ..schema import HumanTier, LogEntry

_log = logging.getLogger(__name__)

_HEADER = re.compile(
    r"^###\s*\[(?P<role>[A-Za-z_]+)\]\s+(?P<author>\S+)\s+(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$",
    re.MULTILINE,
)


# The logbook stores a little HTML inside log_text, and one construct of it is expensive: the
# mini-proposal link the autologger inserts, <a href="https://diii-d.gat.com/DIII-D/physics/
# miniprop/mp_list.php?mpid=..." target="_blank" >, is 53 tokens of MiniLM's 256-token window on
# its own and is identical across every shot of a run, so it is pure noise for retrieval. Measured
# on the full corpus: 20,645 <a>/</a>, 3,603 <b>/</b>, 20 <pre>, 14 <font> and 4 <del> tags over
# 10,821 of the 53,179 records. This is a whitelist of those five tag names and not a general
# <[^>]*> strip because operators write physics in angle brackets -- <ne>=4.2e13, <TINJ>~2.3,
# <taus>~44, "Cut early gas (0 <t < 200ms)", "Try for q95 < 3 hybrid" -- and a general strip eats
# all of it. Stripping these five changes verdict() for 0 and fault_strings() for 0 of the 10,821
# affected records (fix wave 1), so it is a pure token saving.
_MARKUP = re.compile(r"</?(?:a|b|pre|font|del)\b[^>]*>", re.IGNORECASE)


def parse_log_entries(log_text: str | None) -> list[LogEntry]:
    if not log_text:
        return []
    log_text = _MARKUP.sub(" ", log_text)
    heads = list(_HEADER.finditer(log_text))
    out: list[LogEntry] = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(log_text)
        body = log_text[h.end() : end].strip()
        out.append(
            LogEntry(role=h["role"].upper(), author=h["author"], time=h["ts"][11:16], text=body)
        )
    return out


# Real logs.jsonl corpus check (task-8-report.md): ~10,300 of 53,179 records carry this phrase,
# 99%+ as one comma-joined line ("...MA, btor: ... T, ..."), but a same-mini-proposal-template run
# of shots (e.g. 152721-152723) instead repeats "requested" on its own line per field with no
# comma at all ("requested ip: 1.1 MA\nrequested btor: 2.0 T\n..."). [,\s]* accepts either a comma
# or a bare newline as the separator, and the optional "requested " covers the per-field repeat.
_REQ = re.compile(
    r"requested ip:\s*(?P<ip>[-\d.]+)\s*MA[,\s]*(?:requested\s+)?btor:\s*(?P<bt>[-\d.]+)\s*T[,\s]*"
    r"(?:requested\s+)?pnbi:\s*(?P<pnbi>[-\d.]+)\s*MW[,\s]*(?:requested\s+)?pech:\s*(?P<pech>[-\d.]+)\s*MW"
    r"(?:[,\s]*(?:requested\s+)?density:\s*(?P<ne>[\d.eE+-]+))?",
    re.IGNORECASE,
)


def parse_requested(text: str | None) -> dict[str, float]:
    """The session leader's pre-shot 'requested ip/btor/pnbi/pech[/density]' line, if filled in."""
    m = _REQ.search(text or "")
    if not m:
        return {}
    vals = {
        "ip_MA": float(m["ip"]),
        "bt_T": float(m["bt"]),
        "pnbi_MW": float(m["pnbi"]),
        "pech_MW": float(m["pech"]),
    }
    if m["ne"]:
        vals["density_cm3"] = float(m["ne"])
    return {} if all(v == 0 for v in vals.values()) else vals  # all-zero = the unfilled template


# Chief-operator termination/fault vocabulary. "IP min" is deliberately absent (normal end of shot),
# but "IP min FIRST FAULT" (below) is not the same claim -- it says the PCS fault log actually
# logged this as the first-tripped fault, which is a real fault regardless of which system it names.
# Real corpus check: "pcs first fault" (the only literal form here before this task) shows up once
# in a dense sample where "ipmin first fault"/"ipdot first fault"/etc. show up 70+ times -- \w+
# generalizes to whichever system name precedes "first fault" (confirmed on shot 153002's
# "IPmin first fault", still normalizing to "pcs first fault" for the literal "PCS first fault" input).
#
# Fix wave 1 (task-8-report.md): that generalization also matched routine gyrotron status tables,
# e.g. shot 198785's RF entry "...Actual Pulse [ms]\tBlocks\tFirst Fault\nHAN\t1800\t2200\t106\t...",
# where "Blocks" and "First Fault" are two adjacent *column headers*, not a system name followed by
# a real fault report -- confirmed on the full corpus at 327 distinct shots, versus 35 for the real
# "pcs first fault" phrasing the generalization was made for. The distinguishing feature is the
# whitespace between the preceding word and "first fault": every real chief-operator sentence in
# the corpus separates them with exactly one space ("IPmin first fault", "PCS first fault @ 4.8
# sec"), while every table-header false positive uses a tab (the common case) or a run of
# padding spaces (shot 199969's "Blocks                          First Fault") to align columns --
# never a single space. Requiring a literal single space (not \s+) keeps every real prefix matching
# while dropping every table header seen in the corpus (also kills two much rarer false positives
# of the same shape: a "Type ... First fault type and time" table column, and a vertical
# "Blocks:\tPCS Mod\nFirst Fault: ..." field listing, both newline/tab-separated, shot 199967).
# Trade-off, measured and accepted: 9 shots phrase a real prefix across a line wrap ("...current
# limit\nfirst fault", shot 137209) and now produce no fault string for that mention either, since
# a wrapped newline isn't a single space -- versus the 327 shots of table header this removes.
# Every dominant single-space phrasing (pcs/ipmin/ipdot/dot/limit first fault) is unaffected.
_FAULT = re.compile(
    r"(dud trip|\w+ first fault|locked[- ]?mode|disrupt(?:ed|ion|s)?|fizzle[sd]?|no plasma|abort(?:ed)?"
    r"|vertical (?:instability|displacement)|\bvde\b|runaway)",
    re.IGNORECASE,
)
# Fault names that spell their own negation, so a *preceding* negation is not a negation of them
# (see the same carve-out for the verdict lexicon below): "no plasma" in shot 140183's "No Gas =
# No Plasma" is a real zero-plasma shot, not a cancelled one. Everything else here is a bare noun
# phrase and is negation-guarded.
_SELF_NEGATING_FAULT = re.compile(r"(?i)^no\b")


def fault_strings(*texts: str | None) -> list[str]:
    """Fault names actually asserted in the given text(s), de-duplicated in order of first mention.

    Fix wave 2 (task-8-report.md): a match whose own clause is negated is skipped, using the same
    clause-bounded backward window as _guarded() below (which is defined further down, with the
    verdict lexicon it was written for -- resolved at call time, so the order is fine). Without
    this, 543 of 53,176 corpus records reported a fault whose every mention in the record was
    explicitly *absent*: shot 150716's "No locked mode." returned ['locked mode'], shot 150736's
    "No DUD trip." returned ['dud trip'], shot 150286's "pellet no runaways" returned ['runaway'],
    shot 152790's "appears to have not generated a locked mode" returned ['locked mode']. Task 10
    uses this function as the primary failure detector for curation (the logbook's own shot_ok
    field is null in every record), so a false "this shot faulted" is a wrongly-labelled shot in
    the curated set.

    Task-10 fix wave 1: the two scope guards defined below -- _own_shot_text() and mitigated() --
    are applied here too, and for the same reason. Shot 175676's only fault language is a pre-shot
    plan ("Implement locked mode dud trip to avoid disruption") sitting under a "Last shot:"
    heading, and it shipped into the curated list as `fault: locked mode;dud trip` on a shot the
    chief operator called ok. A fault named inside a plan to avoid it, or inside a sentence
    grading a different shot, is not a fault this shot had. Note the mitigation guard is checked
    *before* the _SELF_NEGATING_FAULT carve-out, which only concerns backward negation.

    Deliberately *not* filtered here: routine ramp-down disruption language ("disrupts in rampdown
    as usual"). That is a different concern from negation -- the operator is asserting a real
    disruption, just an expected one -- and verdict() handles it separately via
    _ROUTINE_DISRUPT_CTX. fault_strings() reports "disrupted" whenever it is asserted.
    """
    seen: list[str] = []
    for t in texts:
        t = _own_shot_text(t)
        for m in _FAULT.finditer(t):
            if mitigated(t, m.start(), m.end()):
                continue
            if not _SELF_NEGATING_FAULT.match(m.group(1)):
                before, _ = _clause(t, m.start(), m.end())
                if _NEGATION.search(before):
                    continue
            s = re.sub(r"[- ]+", " ", m.group(1).lower())
            if s not in seen:
                seen.append(s)
    return seen


# Verdict lexicon ported from shotsearch/src/shotsearch/data/goodness.py (same author, MIT).
# Real corpus check (task-8-report.md): chief operators write both "bad shot" and the reversed
# "shot: bad"/"shot bad" (e.g. shot 160309's "Plasma shot: Bad. D2 opened up its breaker.", shot
# 153296's "Plasma Shot bad, ..."); without the reversed form, 160309 fell through to "unknown"
# despite an explicit chief-operator "Bad" -- the one free outcome label this project has for it.
# Anchored on "shot" immediately before "bad" (not a bare \bbad\b) so it does not fire on the real
# "Not bad." asides seen in the same corpus (e.g. shot 166632), which have no "shot" next to "bad".
#
# Fix wave 1 (task-8-report.md): these two alternatives were matched with a bare .search(),
# unguarded, so a direct negation of them was never checked -- e.g. shot 154383's "Result:  OK, not
# a bad shot." and shot 194237's "Not a bad shot, considering." both matched "bad shot" and read as
# "bad" despite the operator explicitly saying the opposite; shot 133446's "Not bad shot." is a
# third.
#
# Fix wave 2 (task-8-report.md): "lost it/the shot/vertically" joined them. It had been left
# unconditional on the theory that it, too, spelled its own negation -- it does not; it is a bare
# assertion. Measured on the full corpus: 163 occurrences, of which 3 are directly negated and
# *none* appear in the causal "no CAUSE = no EFFECT" chain the unconditional group exists to
# protect. Shot 204672's physics operator writes "this rampdown plasma is ELM-free, and disrupts at
# 8.1 sec. Not lost vertically." against a session leader's "Post: very good shot."; shot 204673 is
# the same session and sentence; shot 121329 writes "Doesn't look like a standard disruption, and
# it\nwasn't lost vertically." All three read "bad" before this wave.
_STRONG_NEG = re.compile(
    r"(?i)(?:bad shot|shot\s*[:,]?\s*bad\b|lost (?:it|the (?:shot|plasma|discharge)|vertically))"
)
# The alternatives below stay unconditional (a bare .search(), never run through _guarded()). Each
# one was re-measured against the full corpus for fix wave 2 -- the count of matches in
# VERDICT_ROLES text that a clause-bounded backward guard *would* cancel is given per phrase -- and
# each is kept unconditional for one of two evidenced reasons:
#  - Self-negating: the negation is inside the match, so a backward guard has nothing to find and
#    the question does not arise. "not (?:a |very )?good" (645 matches), "wasn't good" (3),
#    "not great" (74), "not successful" (17).
#  - Self-negating *and* a known false-cancellation risk: chief operators routinely write causal
#    "no CAUSE {so,=,-} no EFFECT" chains, where the first "no"/"didn't" has nothing to do with the
#    phrase that follows -- shot 140183's "No Gas = No Plasma", shot 136807's "No ECH so no plasma
#    at all", shot 131081's "the ecoil didn't run - no plasma". Guarding would wrongly cancel 4 of
#    342 "no plasma" matches, 1 of 102 "no breakdown", 1 of 495 "didn't run/breakdown", 2 of 162
#    "no good" -- every one a genuinely bad shot. No real double negative of any of them exists in
#    the corpus.
# The remaining bare-assertion alternatives are unconditional because the corpus contains no
# negated occurrence of them at all, so guarding would be a no-op with only downside risk:
# "bad data" 0 of 7, "aborted" 0 of 124, "lm kills" 0 of 51, "killed the shot/plasma/discharge"
# 0 of 62, "bummer" 0 of 916. "bad plasma" has 1 of 20, and it is a false cancellation, not a real
# one (shot 200786's "Han and Nasa did not fire due to bad plasma at 600 ms" -- the "did not fire"
# is the cause, the bad plasma is real), so it stays unconditional too.
#
# Known defect, deliberately left for its own measured pass (out of scope for fix wave 2): the
# "aborted?" alternative was meant to be "abort(?:ed)?" (as _FAULT above has it), so it matches
# "aborte"/"aborted" but never bare "abort"/"aborts"/"aborting" -- 591 real occurrences of those
# forms in the corpus are missed. Fixing it is a behavior change of its own size and direction
# (shot 152273's "IP min - Swing Probe abort at 10 sec" is not obviously a bad shot), so it needs
# its own measurement, not a drive-by edit here. This is a recorded bug, not intent.
_STRONG_NEG_UNCONDITIONAL = re.compile(
    r"(?i)(?:bummer|bad plasma|bad data|no good|not (?:a |very )?good"
    r"|was\s?n'?t good|not great|not successful|no plasma|aborted?|lm kills?"
    r"|killed the (?:shot|plasma|discharge)"
    r"|did ?n.?t (?:run|break ?down)|failed to (?:form|break ?down|sustain|run)|no break ?down)"
)
# Bare "mode" is deliberately absent here too (see the "clean" comment below): DIII-D shots
# constantly mention H-mode/L-mode/3-2-mode etc. (confirmed all over VERDICT_ROLES text in the
# real corpus), so only the qualified "locked mode" -- a genuine fault name, not a mode number --
# is matched.
_STRONG_NEG_KW = re.compile(r"(?i)\b(?:locked[- ]?modes?|vde)\b")
# "trip" alone (real example: shot 150555's chief-operator "B module 2 tripped early") missed every
# past/plural/progressive conjugation. The trailing \b still excludes real PCS parameter names built
# on the same root (TRIPLEVEL, TRIPOUT, TRIPOFF all seen in the corpus) since none of them continue
# with exactly "ped"/"s"/"ping" -- moot for verdict() anyway, which never sees PCS-role text.
#
# Fix wave 1 (task-8-report.md): this fed verdict() via a bare .findall(), unguarded, so an
# explicitly *absent* fault counted as evidence of one. Re-measured for fix wave 2 on the full
# corpus with the shipped guard (clause-truncated 16-char backward window, current _NEGATION):
# 491 of 8,242 "trip" matches in VERDICT_ROLES text are negation-preceded, e.g. "No trips.",
# "No trip, good 5s shot", "did not trip", "never tripped", "to avoid tripping". (Fix wave 1's
# comment claimed 226 here; no variant reproduces that -- see the fix-wave-2 report. For reference,
# the untruncated window gives 551, the pre-fix-wave-1 _NEGATION gives 368 untruncated / 318
# truncated, and a strict-adjacency regex gives 261.) The guard is also why _NEGATION below had to
# grow an "avoid" alternative: "avoided?" only ever matched "avoide"/"avoided", never bare "avoid",
# so "to avoid tripping" fell through unguarded.
_TRIP = re.compile(r"(?i)\btrip(?:ped|s|ping)?\b")
# Fix wave 2, second pass (task-8-report.md, "Fix wave 2" section): the 16-char cap above still
# misses a same-clause negation further back than that. Shot 199933's entire qualifying text is
# "CHange back to timeOnly / No longer get the V1 trip" (21 chars from "No" to "trip") and shot
# 207440's is "never got the first n=1 mode amplitude trip" (39 chars) -- both read "bad" when they
# should be "unknown". Re-measured on the full corpus: with the 16-char cap lifted (scanning back to
# the nearest _CLAUSE_BREAK with no length limit), 180 "trip" matches that the shipped guard leaves
# counted would newly find a same-clause negation. (A first pass by a reviewer had estimated 52; no
# variant tried here reproduces that count, so this comment records the re-measured figure and not
# the estimate -- see the fix-wave-2 report for the methodology.)
#
# Lifting the cap for all 180 trades one error for another, though: 15 of them are a *causal
# reassertion*, not an absence -- shot 157369's "Still no lower coils due to power supply trip" is
# the shape ("no/not X due to/because of Y trip"), where the trip is what caused X and must still
# count; spot-checked a majority of the 15 and every one read the same way. _TRIP_CAUSAL below
# detects that shape and keeps the trip real rather than cancelling it. Of the other 165, the large
# majority are a *plan* ("to avoid ... trip") that is correctly cancelled either way; a small,
# measured residual is not rescued by this or any other mechanism here -- see _guarded_trip()'s
# docstring for the two known cases and why they are being left as a documented limitation.
_TRIP_CAUSAL = re.compile(r"(?i)\b(?:due to|because of)\b")
# Fix wave 2 (task-8-report.md): "interlock" and "overcurrent" joined "trip" here. Fix wave 1 had
# left them in the unconditional group alongside "no beams"/"did not fire"/"did not get", but
# unlike those they are bare nouns with no negation of their own, and the corpus negates them
# constantly: 35 of 939 "overcurrent" matches and 7 of 285 "interlock" matches are negation-
# preceded, nearly all of them a *plan to avoid* the fault rather than a report of one ("Reduce AA9
# amplitude by 10% to avoid overcurrent" on shot 159328, "Hopefully 7B will not overcurrent" on
# 153766, "Tweak D1 supply control to avoid chopper overcurrent" on 154975, "repeat without 5B
# interlock" on 160476, "Making sure ECH has no interlock to prevent early injection" on 198591).
# ("trip" itself moved out to _TRIP above: its same-clause negations sit far enough back, and often
# behind a causal reassertion, that it needs the wider guard in _guarded_trip() below instead of
# this module's standard 16-char window.)
_WEAK_NEG = re.compile(r"(?i)\b(?:interlock|overcurrent)\b")
# Same self-negation reasoning as _STRONG_NEG_UNCONDITIONAL above, re-measured for fix wave 2:
# "no beams" (1 of 537 would be cancelled), "did not fire" (3 of 551), "did not get" (1 of 306),
# "failed to fire" (1 of 93) all already contain their negation and are vulnerable to an unrelated
# earlier negation bleeding in -- shot 153792's "Restore 153780, no I-coils No beams", shot 158561's
# "IU30 did not run (AAs did not get ready)", shot 169097's "30L was not ready and did not fire at
# all"; every one a genuine fault that guarding would wrongly cancel. "oops" (2 of 202) and
# "oopsie" (1 of 15) are checked the same way and stay unconditional for the same reason: both
# negated-looking occurrences are causal chains, not negations of the exclamation itself (shot
# 154876's "No AA (oops, not enabled)", shot 164570's "no pellet oops.", shot 183123's "No glow
# oopsie."). "fails? to fire" has 0 negated occurrences of its own. Only the two bare nouns left
# in _WEAK_NEG above -- interlock, overcurrent -- go through this module's standard _guarded()
# window; bare trip has its own, wider one, _guarded_trip() below.
_WEAK_NEG_UNCONDITIONAL = re.compile(
    r"(?i)\b(?:oops|oopsie|fails? to fire|failed to fire|did not fire|no beams|did not get)\b"
)
_DISRUPT = re.compile(r"(?i)\bdisrupt(?:ion|ed|s)?\b")
# "clean" is deliberately absent: this project already removed it once, from a different keyword
# list, for the same reason bare "mode" is absent above (false matches on plasma vocabulary -- see
# CLAUDE.md). Confirmed still live in this shotsearch-ported list against the real corpus: a
# word-bounded "clean" hit 190 times in a 20% sample of VERDICT_ROLES text, 125 of them a "clean
# up"/"clean-up" *maintenance* shot (often one with "requested ip: 0 MA", i.e. no plasma at all --
# e.g. shot 154319), which is not evidence the shot's physics went well.
#
# Fix wave 1 (task-8-report.md): this was the dangerous direction, with *no* guard at all -- a bare
# .findall() -- so a negated positive scored as a point *for* "good". _STRONG_NEG_UNCONDITIONAL only
# covers a few negated positives ("not good", "wasn't good", "not great", "not successful"); it has
# no alternative for "not ok"/"not okay"/"not fine"/"not nice"/"not excellent"/"not as planned", so
# those fell through to here and matched the bare positive word inside the negation. Real examples:
# shot 155244's session leader "Didnt help, Not ok." and shot 164109's physics operator "SPA3 is
# not OK, and hasn't been" (the record runs straight on into "It is producing about 30% more
# current than requested") -- a described hardware problem, scored as a success. Fixed by running
# the whole pattern through _guarded() rather than special-casing each phrase, so it covers every
# positive word uniformly instead of only the ones someone thought to enumerate. Unlike
# _STRONG_NEG_UNCONDITIONAL / _WEAK_NEG_UNCONDITIONAL above, none of these words are themselves a
# self-negation ("good"/"ok"/... never starts with "no"/"not"/"n't"), so there is no double-negative
# ambiguity for a backward guard to misread -- confirmed by hand-checking a random sample of 25 of
# the real cancellations this guard makes, every one a genuine "not good"/"no good"/"not as
# planned"-style negation of that specific word, not an unrelated earlier negative bleeding in.
# Re-measured for fix wave 2 on the full corpus: the shipped guard (clause-truncated 16-char
# backward window, current _NEGATION) cancels 1,588 of 91,839 _POS matches in VERDICT_ROLES text.
# (Fix wave 1's comment claimed 1,632; no variant reproduces that -- see the fix-wave-2 report. For
# reference, the untruncated window gives 2,006, the pre-fix-wave-1 _NEGATION gives 1,910
# untruncated / 1,519 truncated, and a strict-adjacency regex gives 948.)
_POS = re.compile(
    r"(?i)\b(?:good|ok|okay|nice|great|excellent|fine|successful|as (?:requested|planned)"
    r"|runs? through|ran through|runs? to rampdown|ran to|(?:to|until|reached) eof)\b"
)
# Negation vocabulary, expanded from the real corpus for fix wave 1 (task-8-report.md) now that
# _guarded() covers _STRONG_NEG/_POS/_WEAK_NEG too, not just _STRONG_NEG_KW/_DISRUPT:
#   - "avoided?"/"prevented?" were typo'd optional-suffix groups that never matched the corpus's
#     own dominant bare forms "avoid"/"prevent" ("to avoid tripping", "to prevent BT trip" -- 150+
#     hits between them); rewritten as a proper optional suffix so the bare verb matches too.
#   - "...n't" (didn't, doesn't, wasn't, hasn't, won't, ...) is a very common real negation this
#     list had no answer for at all (hundreds of "didn't trip"/"doesn't trip" hits). \w+n't is safe
#     to add unconditionally: no ordinary English word in this corpus's vocabulary contains an
#     apostrophe, so it cannot false-match plasma jargon the way bare "mode"/"clean" once did.
_NEGATION = re.compile(
    r"(?i)\b(?:no|not|without|avoid(?:s|ed|ing)?|prevent(?:s|ed|ing)?|free of|never|\w+n't)\b"
)
# DIII-D shots routinely end on a scheduled ramp-down that operators describe with the same verb
# as a genuine disruption ("disrupts in rampdown as usual"), so a nearby ramp-down mention reads
# the same as an explicit negation for this one word -- unlike locked-mode/VDE, which stay a fault
# regardless of when they happen. Deliberately searched over the plain (un-clause-truncated) 16-char
# window on both sides: operators punctuate straight through this phrasing ("Runs to start of
# rampdown; disrupts vertically", shot 150006; "Shot runs to ramp-down and disrupts early in
# ramp-down", shot 200590), so clause-truncating this particular window turns 8 correctly-excused
# routine ramp-downs into faults -- measured on the full corpus for fix wave 2 and rejected.
_ROUTINE_DISRUPT_CTX = re.compile(r"(?i)ramp[- ]?down")
# The one *following* cue that genuinely neutralizes a disruption mention: the passive "Disruption
# avoided/prevented/averted" report. See _guarded_disrupt() for why the forward window is this
# narrow instead of the full _NEGATION vocabulary.
_DISRUPT_AVERTED = re.compile(
    r"(?i)^\W*(?:was |were |is |are |been |successfully |completely |entirely )*"
    r"(?:avoid(?:ed)?|prevent(?:ed)?|averted)\b"
)
VERDICT_ROLES = ("SESSION_LEADER", "PHYSICS_OPERATOR", "CHIEF_OPERATOR")
# A negation immediately preceding an *earlier, unrelated* clause must not reach across it: "no
# disruption, ran through" (c.f. the real "No trip, good 5s shot") needs "no" to cancel only
# "disruption", not the independent "ran through" verdict a few words later. _clause() below stops
# its backward scan at the nearest one of these, so same-clause negation ("not a bad shot") still
# works exactly as before while a negation from a previous clause no longer bleeds forward. Includes
# "-": chief-operator shorthand uses a bare dash the same way prose uses a comma to join two short
# clauses (shot 163826's physics operator writes "I-coils do not trip - good", the sentence its
# entry actually ends on -- confirmed live during fix wave 1's own verification, where the "not"
# from the first clause was otherwise reaching across the dash and wrongly cancelling the second,
# unrelated "good"), and this corpus also uses longer "- - - -" runs
# purely as section dividers, which this only ever truncates *after* (never a problem to stop there).
# Deliberately *not* including "\n": chief-operator prose in this corpus line-wraps constantly
# without regard for sentence structure (confirmed on shot 164892's "the shot does not\ndevelop a
# locked mode", a mid-sentence wrap, not a new clause) -- treating a bare newline as a break wrongly
# un-cancelled that "not", flipping the shot to a false "bad" during fix wave 1's own verification.
_CLAUSE_BREAK = re.compile(r"[.,;:!?-]")


def _clause(
    text: str,
    start: int,
    end: int,
    *,
    before_window: int | None = 16,
    after_window: int | None = 16,
) -> tuple[str, str]:
    """The context on each side of text[start:end], each truncated at the nearest _CLAUSE_BREAK so
    neither side reaches into a neighbouring clause. `before_window` caps how far back the backward
    side looks before that truncation is applied; every caller but _guarded_trip() takes the
    default 16, unchanged from before this parameter existed. Pass None for a backward scan bounded
    only by the clause itself, however far back that is -- see _guarded_trip() for why bare "trip"
    needs that. `after_window` is the same knob forwards, and only mitigated() passes None for it:
    the purpose clause it looks for ("... dud trip to avoid disruption") routinely sits further
    ahead than 16 characters, while _DISRUPT_AVERTED keeps the original 16-char forward window it
    was measured on. Returns (before, after)."""
    if before_window is None:
        raw_before = text[:start]
    else:
        raw_before = text[max(0, start - before_window) : start]
    breaks = list(_CLAUSE_BREAK.finditer(raw_before))
    before = raw_before[breaks[-1].end() :] if breaks else raw_before
    after = text[end:] if after_window is None else text[end : end + after_window]
    brk = _CLAUSE_BREAK.search(after)
    if brk:
        after = after[: brk.start()]
    return before, after


# --------------------------------------------------------------------------------------------
# Scope guards: text inside a log entry that is not a report about *this* shot.
#
# Both of these were added in task-10 fix wave 1, after shot 175676 shipped into
# configs/shot_design/shot_lists/poc_v1.yaml labelled `verdict: bad, fault: locked mode;dud trip` when
# the operators had recorded the opposite. Its whole record (run 20180227) reads:
#   [CHIEF_OPERATOR]  "10:32 Plasma shot ok."
#   [SESSION_LEADER]  "Postshot: Density increased a little too high, but should still have
#                      usable data."
# and the only fault language anywhere in it is the physics operator's *pre-shot plan*:
#   "Last shot: Successful enough - late tearing mode kills us, but get 2/3 LBOs\ngood.\n\n
#    This shot:\n... - Implement locked mode dud trip to avoid disruption.\n..."
# -- "kills us" grades the *previous* shot, and "locked mode"/"dud trip" name a protection the
# operator is about to install *so that* a disruption does not happen. Neither is a report that
# this shot faulted.

# (a) Back-references. Operators structure a pre-shot entry as "Last shot: <how the previous one
# went>" / "This shot: <the plan>", and often close with "Next shot: <what to try after>". A span
# introduced by a marker naming a *different* shot is removed before any lexicon sees it, so its
# words vote on the shot they are actually about (i.e. on nothing here). Measured on the full
# corpus: 1,971 of 53,179 records carry one of these markers in VERDICT_ROLES text --
# "this" 1,591, "last" 1,086, "next" 465, "previous" 231, "prior" 1, and bare "prev" 0 (kept in
# the pattern only because it costs nothing). A colon is required: "better than last shot" (no
# colon) is a comparison inside a real report and must not blank the rest of the entry.
# The negative lookbehind is for "Repeat last shot: ..." / "Repeat previous shot: ...", which
# introduces *this* shot by naming what it copies, not a comment about the other one. 5 of the
# 1,318 back-reference markers in the corpus are that phrasing (shots 145809, 171645, 178751,
# 193022, 195173) and in every one the text that follows describes this shot.
_SHOT_SCOPE = re.compile(
    r"(?i)(?<!repeat )\b(?P<which>last|previous|prev|prior|next|this|current)\s+shots?\s*:"
)
_OTHER_SHOT = frozenset({"last", "previous", "prev", "prior", "next"})
# A dropped span ends at the next marker, at a blank line, at one of the logbook's own dashed
# divider lines (the session leader's pre-shot template writes "- - - - - - -" between the free
# text and the `requested ip:` line -- shot 199172 is the shape), or at an outcome heading.
# Whichever comes first.
#
# The outcome heading matters because "Next shot:" in particular is sometimes a heading for the
# shot being set up rather than the one after it -- shot 158368's session leader writes "Next
# Shot: Higher density scan. Target: ... Result: Density limit for setting of 6." and that
# Result is this shot's. Exactly 5 dropped spans in the corpus contain an outcome heading (shots
# 122156, 153006, 153011, 158368, 172545) and in all five the heading introduces a report about
# the shot the entry belongs to, so stopping there is right in every case seen.
_SCOPE_END = re.compile(
    r"(?i)\n[ \t]*(?:\n|(?:[-=_*][ \t]*){3,})|\b(?:post\s?-?\s?shot|results?|outcome)\s*:"
)


def other_shot_spans(t: str | None) -> list[tuple[int, int, str]]:
    """`(start, end, marker)` of every span of `t` that is about a *different* discharge.

    The span runs from the marker ("Last shot:", "Next shot:", ...) to whichever comes first of
    the next marker, a blank line, one of the logbook's dashed dividers and an outcome heading --
    the boundaries measured on the corpus in the comments above.

    Public because two callers need the same boundary for opposite reasons: `verdict()` deletes
    these spans before grading a shot (_own_shot_text below), and `labels.claims` keeps them and
    dates every claim inside one by its marker instead of `observed`. A previous shot's "strong
    EHO" is a real claim -- about the previous shot -- and one boundary rule has to serve both, or
    a span could be graded as another shot's and recorded as this one's.
    """
    t = t or ""
    marks = list(_SHOT_SCOPE.finditer(t))
    spans: list[tuple[int, int, str]] = []
    cut = 0
    for i, m in enumerate(marks):
        which = m["which"].lower()
        if which not in _OTHER_SHOT:
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(t)
        brk = _SCOPE_END.search(t, m.end(), end)
        if brk:
            end = brk.start()
        if m.start() < cut:  # already inside a span we took
            continue
        spans.append((m.start(), end, which))
        cut = end
    return spans


def _own_shot_text(t: str | None) -> str:
    """`t` with every span attributed to a different shot replaced by a clause break.

    The replacement is " . " rather than "" so the two surviving halves stay separate clauses --
    splicing them would let a negation on one side of the removed span reach across it, which is
    the same bug _CLAUSE_BREAK exists to prevent.
    """
    t = t or ""
    spans = other_shot_spans(t)
    if not spans:
        return t
    out, cut = [], 0
    for start, end, _which in spans:
        out.append(t[cut:start])
        cut = end
    out.append(t[cut:])
    return " . ".join(out)


# (b) Planned mitigations. A fault name followed, inside its own clause, by a purpose clause
# saying the plan is to *avoid* that fault is not a report of the fault. _NEGATION already covers
# the backward form ("to avoid tripping" cancels a following "trip"); this is the same cue in the
# other direction, which is how the corpus phrases an installed protection: the fault is named
# first and the purpose last ("Implement locked mode dud trip to avoid disruption").
#
# The forward window is clause-bounded but *not* capped at 16 characters, because the purpose
# clause routinely sits further ahead than that (in 175676, "to avoid" is 9 characters past the
# end of "dud trip" but 20 past the end of "locked mode"). This is deliberately much narrower
# than a forward _NEGATION scan, which was measured and rejected for _guarded_disrupt() -- an
# infinitive "to avoid/to prevent/..." is a statement of purpose, whereas a bare following "no"/
# "not"/"without" is usually about something else ("disrupts without locking").
# "to stop" is deliberately absent, unlike "to avoid"/"to prevent": it is the corpus's ordinary
# verb for a discharge ending, not a statement of purpose -- shot 126933's chief operator writes
# "E-coil volt sec trip caused plasma to stop at 3 sec", where the trip is exactly what happened.
_MITIGATION = re.compile(
    r"(?i)\bto\s+(?:avoid|prevent|preclude|mitigate|protect against|guard against)\b"
)


def negated(text: str, start: int, end: int) -> bool:
    """True if text[start:end] is preceded, within its own clause, by a negation cue.

    Public because it is not only the verdict's business: `labels.claims` asks the same question
    of one phenomenon mention to decide a claim's polarity, and the answer has to be the same one
    `_guarded()` gives -- the clause bound, the 16-character window and the negation vocabulary
    below were all measured against the real 53,179-record logbook, and a second copy of that
    judgement would drift from this one on the first correction.
    """
    return bool(_NEGATION.search(_clause(text, start, end)[0]))


def mitigated(text: str, start: int, end: int) -> bool:
    """True if text[start:end] is followed, within its own clause, by a purpose clause.

    Public for the same reason as negated(): "locked mode dud trip to avoid disruption" is a plan,
    not a report, whether the reader is grading the shot or recording a claim about tearing."""
    return bool(_MITIGATION.search(_clause(text, start, end, after_window=None)[1]))


def _guarded(pattern: re.Pattern[str], text: str, *, mitigation: bool = False) -> int:
    """Count matches not preceded, within their own clause, by a negation cue: 'no disruption' is
    not bad. A cancelled match contributes nothing at all -- it is dropped from the count, never
    flipped into a vote for the opposite polarity. So "not a bad shot" yields no signal from that
    phrase (the verdict falls back on whatever else is in the text) rather than becoming a vote for
    "good", and symmetrically a negated positive like "not ok" yields no signal rather than
    becoming a vote for "bad".

    `mitigation=True` additionally applies mitigated() -- the forward "to avoid/to prevent"
    purpose clause. It is passed only for the fault lexicons, never for _POS: a positive word is
    not neutralised by a plan ("good, to avoid ..." is still a good shot), and the whole point of
    the guard is that a fault named inside a plan to avoid it has not happened.
    """
    return sum(
        1
        for m in pattern.finditer(text)
        if not negated(text, m.start(), m.end())
        and not (mitigation and mitigated(text, m.start(), m.end()))
    )


def _guarded_trip(text: str) -> int:
    """Like _guarded(_TRIP, text), but the backward scan is not capped at 16 characters: it reaches
    all the way back to the previous clause break, because bare "trip"'s same-clause negation is
    often further back than that (see the comment on _TRIP above -- 180 such matches on the full
    corpus). A causal marker ("due to"/"because of") between the negation and "trip" overrides the
    cancellation, because the corpus's dominant far-same-clause shape is a causal reassertion ("no
    X due to Y trip"), not an absence -- 15 of the 180.

    Known, measured residual, left as a documented limitation rather than solved: 2 of the 180 (2 of
    8,242 "trip" matches overall) are a shape neither this nor any due-to/because-of check catches,
    and both still read as cancelled when the trip is real. Shot 164995's "Not enough to avoid the
    power supply trip" is a same-clause *double* negative (failed to avoid it = it happened); shot
    188761's "were not enough to go to the requested reference so it tripped soon after" reasserts
    causally with "so" rather than a backward "due to"/"because of". Extending _TRIP_CAUSAL to a
    forward "so" was tried and rejected: bare "so" is far too common and ambiguous elsewhere in this
    corpus's prose to add without its own full measured pass, which is out of scope here.
    """
    n = 0
    for m in _TRIP.finditer(text):
        if mitigated(text, m.start(), m.end()):
            continue
        before, _ = _clause(text, m.start(), m.end(), before_window=None)
        neg = _NEGATION.search(before)
        if neg and not _TRIP_CAUSAL.search(before[neg.end() :]):
            continue
        n += 1
    return n


def _guarded_disrupt(text: str) -> int:
    """Like _guarded, but a *following* cue can neutralize the word too, which a plain negation
    cannot do.

    Fix wave 2 (task-8-report.md): the backward negation scan now uses the same clause-bounded
    window as _guarded() -- fix wave 1 added clause truncation there and left this sibling on an
    unbounded 16-char window, so a negation about a different clause cancelled a real disruption:
    shot 150744's "No H-mode, disrupts.", shot 156476's "No H again. Disrupted again.", shot
    154876's "AAs didn't fire; not enabled. LM disruption at about 2.9 seconds.", shot 153122's
    "No ECH and no EFC. Plasma disrupted in Flattop." 25 verdict changes on the full corpus, all
    toward "bad" (good->mixed 14, mixed->bad 9, unknown->bad 2).

    The forward direction is deliberately *not* the full _NEGATION vocabulary, contrary to the
    symmetric design proposed for this fix wave: measured on the full corpus, a clause-bounded
    forward _NEGATION scan changes 8 more verdicts and 7 of the 8 are wrong, because "disrupt" is
    routinely followed by a negation about something else -- "disrupts without catching" (shot
    192196), "shot disrupted without rmp" (207647), "disrupts without locking" (149053), "disrupts
    but with n=2 not n=1 mode" (207245), "early disruption | never went into H-mode" (147204),
    "LM disruption ~ 4100ms Did not seem to get..." (131143), "preceding disruption and no
    prediction" (180807). Every one of those is a real disruption. What the symmetric design was
    for -- shot 159422's "Disruption avoided for the entire discharge" -- is instead handled by
    _DISRUPT_AVERTED, a narrow passive "avoided/prevented/averted" follower: it has exactly 3
    occurrences in the whole corpus (shots 159422, 198833, 199909) and all 3 are genuine.
    """
    n = 0
    for m in _DISRUPT.finditer(text):
        before, after = _clause(text, m.start(), m.end())
        if _NEGATION.search(before) or _DISRUPT_AVERTED.search(after):
            continue
        if mitigated(text, m.start(), m.end()):
            continue
        if _ROUTINE_DISRUPT_CTX.search(text[max(0, m.start() - 16) : m.end() + 16]):
            continue
        n += 1
    return n


def verdict(entries: list[LogEntry], extra: str = "") -> str:
    # _own_shot_text() is applied per entry, before the join: a "Last shot:" span that runs to the
    # end of its own entry must not swallow the next operator's entry as well.
    #
    # Entries are joined on " . ", not " " (task-10 fix wave 1). Two log entries are never one
    # clause -- they are usually two different operators -- so every guard in this module, forward
    # and backward, has to stop at the boundary. On a bare " " join it did not, and the corpus
    # shows all three failure modes it causes. Measured on the full corpus, the separator changes
    # 7 verdicts and 0 fault_strings results, and all 7 were hand-checked as fixes:
    #   * a negation reaching forward into the next operator's entry -- shot 149143's chief writes
    #     "11:08  Plasma Shot - IP Dot - No 30L" and its "No" was cancelling the session leader's
    #     own "bad shot; multiple beam failures." (unknown -> bad). Shots 180651, 204912, 166551
    #     and 145722 are the same shape.
    #   * a *phrase* spliced out of two authors' text: "... Same, bad" followed by "Plasma shot;
    #     short" matched _STRONG_NEG_UNCONDITIONAL's "bad plasma" on shot 152216, and shot
    #     178493's "doesn't look half-bad" + "Plasma shot  ok." did the same (bad -> good).
    #   * the forward window of mitigated() reaching a "to avoid" in the *next* entry, which is
    #     what made shot 208177's real "still tripped density limit" and shot 125470's "n=1 at
    #     2900 => dud trip" read as plans.
    text = (
        " . ".join(_own_shot_text(e.text) for e in entries if e.role in VERDICT_ROLES)
        + " . "
        + _own_shot_text(extra)
    )
    if (
        _guarded(_STRONG_NEG, text, mitigation=True)
        or _STRONG_NEG_UNCONDITIONAL.search(text)
        or _guarded(_STRONG_NEG_KW, text, mitigation=True)
    ):
        return "bad"
    pos = _guarded(_POS, text)
    neg = (
        _guarded_trip(text)
        + _guarded(_WEAK_NEG, text, mitigation=True)
        + len(_WEAK_NEG_UNCONDITIONAL.findall(text))
        + _guarded_disrupt(text)
    )
    if pos == 0 and neg == 0:
        return "unknown"
    if pos > neg:
        return "good"
    return "bad" if neg > pos else "mixed"


def parse_shot_range(s: str | None) -> tuple[int, int] | None:
    m = re.search(r"(\d{5,6})\D+(\d{5,6})", s or "")
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (min(a, b), max(a, b))


# ---------------------------------------------------------------- data access


def _clean(v):
    return None if v is None or v == "" or v == "None" else v


@functools.lru_cache(maxsize=4)
def _shot_index(path_str: str, key: tuple[int, int]) -> types.MappingProxyType[int, str]:
    d = json.loads(Path(path_str).read_text())
    return types.MappingProxyType({int(k): str(v) for k, v in d.items() if str(k).isdigit()})


def load_shot_index(paths: Paths) -> Mapping[int, str]:
    """shot -> run_id, from sql/index.json. This is the authoritative shot->run mapping; the
    per-shot text bundles are not (see mp_text).

    Read-only on purpose (fix wave 1, item 3): the result is the memoized dict itself, so a caller
    that popped a key from it would silently corrupt every later human_tier() in the process. The
    real index.json holds 53,176 entries and human_tier() looks a shot up in it once, so handing
    out a defensive copy would cost 0.66 ms per shot against human_tier's own 6.7 ms -- 10 % -- for
    a mapping whose values are all immutable strings. A MappingProxyType costs 0.1 us and refuses
    the mutation outright.

    Keyed on (path, size, mtime_ns) like the other two caches in this module, which it was not
    before (fix wave 1, item 7) -- index.json is rewritten wholesale when the text archive is
    refreshed, and there is no reason for this one cache to keep serving the old mapping.
    """
    p = paths.shot_index_json
    return _shot_index(str(p), _stat_key(p)) if p.exists() else {}


def subset_path(paths: Paths) -> Path:
    return paths.text_cache_dir / "logs_subset.jsonl"


# Every one of the 53,179 lines of the real logs.jsonl begins exactly '{"shot": <digits>,'
# (verified on the full 616 MB file: 0 non-matching lines), so the shot can be matched on the raw
# bytes and only the wanted lines ever go through json.loads. Those 53,179 lines carry 53,176
# distinct shots -- shots 193355 (3 lines) and 150000 (2) are repeated, and _read_subset() below
# keeps the last line for a repeated shot, which is the same record the whole-file readers used in
# tasks 7-8 ended up with.
_SHOT_PREFIX = re.compile(rb'^\{"shot":\s*(\d+)')


def _stat_key(p: Path) -> tuple[int, int]:
    """(size, mtime_ns) -- the cache key that makes memoizing a file read safe here.

    An append or a rewrite that changes the length changes the size, and any other rewrite changes
    the mtime at the filesystem's resolution. That resolution is not infinite: st_mtime_ns on this
    project's /tmp is only ~1 ms granular (3 of 5 consecutive same-size writes measured during fix
    wave 1 shared a timestamp), so the key is not a general-purpose guarantee. What makes it safe
    is the access pattern, not the filesystem: the subset cache only ever changes by whole lines
    (build_logs_subset rewrites it through a temp file and a rename, adding records and dropping a
    torn tail, so any change moves the size), and the mini-proposal PDFs live on a read-only
    archive.
    """
    st = p.stat()
    return (st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=2)
def _subset_records(path_str: str, key: tuple[int, int]) -> dict[int, dict]:
    """Parsed subset cache, memoized on (size, mtime_ns) -- see _read_subset for why.

    The returned dict is shared between callers; treat it (and the records in it) as read-only.
    load_log_record() deep-copies out of it for that reason.

    An undecodable LAST line is dropped with a warning rather than raised: it is what a build
    killed mid-append left behind (the cache used to be appended through a buffered handle, so a
    SIGKILL could cut the final record anywhere), and raising on it broke every later build and
    add until someone deleted the file by hand. The next build_logs_subset re-extracts that shot
    and rewrites the file whole. An undecodable line anywhere else is not that failure mode and is
    still an error.
    """
    out: dict[int, dict] = {}
    lines = [ln for ln in Path(path_str).read_bytes().split(b"\n") if ln.strip()]
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            if i == len(lines) - 1:
                _log.warning(
                    "%s: dropping an undecodable trailing line (%d bytes; a build was killed "
                    "mid-write) -- its shot will be re-extracted on the next build",
                    path_str,
                    len(line),
                )
                break
            raise ValueError(f"{path_str}: undecodable record on line {i + 1}") from e
        out[int(rec["shot"])] = rec
    return out


def _read_subset(paths: Paths) -> dict[int, dict]:
    """Deviation from the task-9 brief, made on a measurement (task-9-report.md): the brief
    re-read and re-parsed the whole subset on every load_log_record() call. Measured on the real
    626-record / 7.9 MB subset for this project's 936 staged shots, that is 23 ms per shot, i.e.
    ~22 s of pure re-parsing for one database build, growing quadratically with the shot list.
    Memoizing on (path, size, mtime_ns) makes it one parse and keeps append-then-read correct,
    which is the access pattern build_logs_subset() + human_tier() actually use."""
    p = subset_path(paths)
    if not p.exists():
        return {}
    return _subset_records(str(p), _stat_key(p))


def build_logs_subset(paths: Paths, shots: set[int]) -> int:
    """Stream the 616 MB logs.jsonl once and add the records for `shots` to the subset cache.

    The cache is rewritten whole -- the existing complete lines, then the new records -- into a
    sibling temp file that is fsynced and renamed over the old one, so a reader never sees a
    half-written file and a build killed at any point leaves either the old cache or the new one,
    never a torn tail. (Appending in place, as this used to, is exactly what produced the torn
    trailing line _subset_records now tolerates.) A line of the old file that lacks its newline is
    such a tail and is not carried over; the shot it belonged to is in `wanted` again.
    """
    have = set(_read_subset(paths))
    wanted = {s for s in shots if s not in have}
    if not wanted:
        return 0
    out = subset_path(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    n = 0
    try:
        with open(tmp, "wb") as dst:
            if out.exists():
                with open(out, "rb") as old:
                    for line in old:
                        if line.endswith(b"\n"):
                            dst.write(line)
            with open(paths.logs_jsonl, "rb") as src:
                for line in src:
                    m = _SHOT_PREFIX.match(line)
                    if m and int(m.group(1)) in wanted:
                        dst.write(line.rstrip(b"\n") + b"\n")
                        n += 1
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return n


def load_log_record(shot: int, paths: Paths) -> dict | None:
    """One logbook record, deep-copied out of the memoized subset.

    Fix wave 1, item 2: the shallow copy this used to return did not protect the cache. Real
    records carry two list-valued fields, `topics` (the roles that wrote entries, e.g.
    ['ANALYSIS', 'CHIEF_OPERATOR', 'PCS', ...]) and `keywords` (always []), and both lists were
    shared with the cached record, so `load_log_record(163112, p)["keywords"].append("POISON")`
    poisoned every later read of shot 163112 for the life of the process. Measured on the real
    626-record subset, deepcopy costs 11.9 us per record against human_tier's 6.7 ms per shot
    (0.2 %), so correctness is simply cheaper than the alternatives here.
    """
    rec = _read_subset(paths).get(shot)
    return copy.deepcopy(rec) if rec is not None else None


def run_metadata(run_id: str | None, paths: Paths) -> dict | None:
    if not run_id:
        return None
    p = paths.shotsummary_raw_dir / run_id / "metadata.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---- the per-shot bundle's own three blocks ---------------------------------------------------
#
# A bundle is `# DIII-D per-shot text bundle`, then three `## ` sections: the general session
# context (the run's header fields and a `METADATA (selected)` JSON object), the mini-proposal,
# and the shot-specific block scraped out of that run day's summary.html. The three helpers below
# read the third and the metadata of the first; `mp_text` above reads the second.
#
# Scope matters more than it looks. The general session context carries `- Run:`,
# `- Shot Range:`, `- Session Leader:` bullets in the same `- KEY: value` shape as the shot
# table row, so a `- (\w+): (.*)` sweep of the whole bundle silently mixes a RUN-level field into
# a per-shot fact. Every helper here is anchored on its own section for that reason.
SHOT_BLOCK_MARKER = "## Shot-specific context (from summary.html)"
SHOT_TABLE_HEADER = "SHOT TABLE ROW (name -> value)"
# What the tool writes when summary.html had no row for this shot -- the "session fallback" case:
# the bundle exists, but everything in it is the session's and none of it is the shot's.
SHOT_TABLE_MISSING = "(Shot table key/value mapping not found.)"
_METADATA = re.compile(r"^METADATA \(selected\)\s*\n(\{.*?\n\s*\})", re.MULTILINE | re.DOTALL)
_KV_LINE = re.compile(r"^- ([^:\n]+): ?(.*)$")


def bundle_blocks(bundle: str) -> tuple[str, str, str]:
    """The bundle's three sections verbatim: `(session, planned, shot_specific)`.

    Each may be empty -- a run with no mini-proposal has no planned block, and a bundle from
    before the summary page had a row for the shot has no shot-specific one -- and the three
    concatenate back to the whole bundle, so nothing is silently dropped by reading it this way.

    The split is worth having as one function rather than three searches at the call sites,
    because *which section a sentence came from is what the sentence means*: the same words are a
    record of the run in the first block, an intention in the second and a fact about this
    discharge in the third. `labels.claims` reads all three and scopes and dates them by which one
    they were in; `shot_block()` below is the third alone, which is what the older readers want.
    """
    mp = _BUNDLE_MP.search(bundle)
    i = bundle.find(SHOT_BLOCK_MARKER)
    end = len(bundle) if i < 0 else i
    if mp is None or mp.start() > end:
        return bundle[:end], "", bundle[end:]
    return bundle[: mp.start()], bundle[mp.start() : end], bundle[end:]


def shot_block(bundle: str) -> str:
    """The bundle's shot-specific block: everything after the marker, stripped.

    Empty when the bundle has no such section at all. Measured over the 13,106 bundles that are
    also in the FAITH corpus: this block runs 138-302 characters for a well-formed plasma shot
    (median 252) -- it is a table, not prose -- so a length threshold on it is a threshold on
    how many columns that run day's summary page filled in.
    """
    i = bundle.find(SHOT_BLOCK_MARKER)
    return bundle[i + len(SHOT_BLOCK_MARKER) :].strip() if i >= 0 else ""


def shot_table_row(bundle: str) -> dict[str, str]:
    """The `SHOT TABLE ROW (name -> value)` block as `{KEY: value}`, in the file's own order.

    Empty for the session-fallback case and for a bundle with no shot-specific block: an absent
    row is absent evidence, and this returns nothing rather than something a caller could read as
    a measurement. Keys are exactly as the summary page spelled them (`IP-(MA)`,
    `PBEAM-MAX-(MW)`), because the page is what the next campaign will change.

    Only the run of `- ` lines that follows the header is read. The block continues into
    `IMPORTANT <pre> BLOCKS` on some shots, and PCS-change lines there are not table columns.
    """
    block = shot_block(bundle)
    i = block.find(SHOT_TABLE_HEADER)
    if i < 0:
        return {}
    out: dict[str, str] = {}
    for line in block[i + len(SHOT_TABLE_HEADER) :].splitlines():
        if not line.strip():
            if out:  # a blank line ends the row; leading blanks before it do not
                break
            continue
        m = _KV_LINE.match(line)
        if not m:
            break
        out[m.group(1).strip()] = m.group(2).strip()
    return out


def mp_subjects(bundle: str) -> tuple[str, ...]:
    """Every `Subject:` line of the bundle's mini-proposal block, in file order, deduplicated.

    Anchored on the planned block for the reason `bundle_blocks` exists: "Subject" appears in the
    logbook text too, and a run-level intention read out of an operator's entry is not the
    experiment's subject.

    ALL of the lines, not the first. A real block is three renderings of one mini-proposal -- the
    PDF text, the markdown export and the HTML page -- and the two that carry a `Subject:` line
    disagree: measured over 400 random bundles, 194 have two subject lines and 106 have one, and
    the PDF's is routinely both truncated at the page's line break and mangled by the font
    mapping (`Diagno'tic checkout fo' QH/WPQH-Mode`), while the markdown's `**Subject**:` is
    clean. Which of the two is readable is a property of that run day's PDF, so a caller that
    matches keywords wants both; `_subject` (first match only) stays what `mp_text` uses, where a
    single title is what is being served.
    """
    _, planned, _ = bundle_blocks(bundle)
    if not planned:
        return ()
    out: list[str] = []
    for m in _SUBJECT.finditer(planned):
        got = re.sub(r"\s+", " ", m.group(1)).strip()
        if got and got not in out:
            out.append(got)
    return tuple(out)


def session_metadata(bundle: str) -> dict:
    """The `METADATA (selected)` JSON of the general session context (`run_id`, `shot_range`,
    `title`), or `{}`.

    The tool writes it with `json.dumps(indent=2)` and then collapses every run of spaces, so
    what is on disk is one-space-indented JSON; `json.loads` does not care, and nothing here may
    depend on the indentation. A block that does not parse is `{}` rather than an exception --
    this is scraped text, and one malformed run day may not stop a corpus-wide selection.
    """
    m = _METADATA.search(bundle)
    if not m:
        return {}
    try:
        got = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    return got if isinstance(got, dict) else {}


_BUNDLE_MP = re.compile(r"^## Planned context \(mini-proposal\)\s*$", re.MULTILINE)
# The bundle's own header line, used only to check that the bundle is about the run we asked for
# (see mp_text). It sits in the first ~40 bytes of every real bundle that has one.
_BUNDLE_RUN = re.compile(r"^RUN_ID:\s*(\S+)", re.MULTILINE)
_SUBJECT = re.compile(r"Subject\*{0,2}:\s*(.+)")
_PURPOSE = re.compile(r"(?:Purpose|Goal|Objective)[^\n]*\n", re.IGNORECASE)


# A symbol-font glyph a PDF maps outside Unicode comes back from pypdf as an UNPAIRED surrogate,
# which is a str Python will hold but cannot encode: pydantic's model_dump_json raises
# PydanticSerializationError on it, and it does so in records_to_tables -- after every shot in the
# build has already been read. Measured on configs/shot_design/shot_lists/poc_v1.yaml: 8 of its 200
# shots, from 4 run days (20170307, 20220630A, 20240626B, 20240821A), carry them in mp_purpose;
# nothing else in the human tier does, and sql/logs.jsonl has none in any of its 53,179 records.
# U+FFFD is the honest replacement -- "a character was here that could not be decoded" -- and it
# changes no classification: _prose_ratio already dropped these tokens, since neither a lone
# surrogate nor U+FFFD is str.isalpha().
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


@functools.lru_cache(maxsize=128)
def _pdf_text(path_str: str, key: tuple[int, int], pages: int = 4) -> str:
    """First `pages` pages of a mini-proposal PDF, memoized on (path, size, mtime_ns).

    Second deviation from the task-9 brief, same measurement (task-9-report.md): mp_text() is
    per shot but a mini-proposal is per run, so the brief re-extracted the same PDF once per shot.
    This project's 936 staged shots resolve to 36 runs (25 with a PDF), and the re-extraction was
    44 s of the build; memoized it is 1.9 s.
    """
    try:
        from pypdf import PdfReader

        raw = "\n".join((p.extract_text() or "") for p in PdfReader(path_str).pages[:pages])
    except Exception:  # noqa: BLE001 — a corrupt PDF must not abort a shot's build
        return ""
    return _LONE_SURROGATE.sub("\ufffd", raw)


# ---- is a PDF extraction actually this experiment's purpose? (fix wave 1, item 1) -------------
#
# len(txt) > 200 was the only gate on the pdf tier, so mp_text() returned source="pdf" with text
# that is not a purpose at all. All 1,661 miniproposal.pdf files under shotsummary/raw were
# extracted and classified for this fix wave (task-9-report.md, "Fix wave 1"): 1,375 give a real
# Purpose section and 270 do not -- 112 "5. Resources / Machine Setup" form dumps, 58 broken font
# encodings, 43 "This is a test PDF document" placeholders, 56 first-3,000-chars-of-page-1 with no
# Purpose heading at all, and 1 table-of-contents line. Real cases: shot 194802's run yields the
# Acrobat placeholder, 162163's is symbol soup ("!<forall>#<exists>%&..."), 172187's purpose is the
# bibliography, 200057's is "goal': Thi' expe'imen" add'e''e'..." and 178661's starts at
# "3. Experimental Method". This is correction C1's defect -- text labelled as this shot's purpose
# when it is not -- reached through the PDF tier instead of the bundle tier.
_PDF_PLACEHOLDER = re.compile(r"this is a test pdf document", re.IGNORECASE)
# A word that survived text extraction intact: ASCII letters, with digits/apostrophe/hyphen allowed
# inside (n=1, ITER's, off-axis). Deliberately *not* "is this character non-ASCII" -- the reviewer's
# suggested <90 %-ASCII rule flags run 20120813, whose purpose is clean English about "high
# <beta>N (>4), high q min (>2) plasmas", purely for its Greek letters, and misses the 2022-2025
# smart-quote corruption ("The primary objec<ldquo>ive of <ldquo>his experimen<ldquo>...") which
# stays above 90 % ASCII. Counting whole words instead separates the two cleanly.
_PLAIN_WORD = re.compile(r"^[A-Za-z][A-Za-z0-9'-]*$")
_PROSE_MIN = 0.75


def _prose_ratio(s: str) -> float:
    """Fraction of the word-like tokens of `s` that came out of the PDF as plain ASCII words.

    Measured over the first 800 chars of every mini-proposal purpose (1,645 PDFs with >200 chars
    of text): the 1,376 with a real Purpose section sit at a median 0.98 and a 1st percentile of
    0.55, the 58 broken-encoding ones at a median 0.38 with 14 of them yielding no countable words
    at all. 0.75 is inside that gap; every run it rejects from the good group was verified by hand
    to be genuinely corrupted text.
    """
    toks = [t.strip(".,;:()[]{}\"'*\u2022\u00b7-\u2013\u2014") for t in s.split()]
    toks = [t for t in toks if len(t) >= 2 and any(c.isalpha() for c in t)]
    if len(toks) < 12:  # too little running text to judge -- and too little to be a purpose
        return 0.0
    return sum(1 for t in toks if _PLAIN_WORD.match(t)) / len(toks)


def _pdf_usable(txt: str, purpose: str) -> bool:
    """Whether a mini-proposal PDF's extracted text may be served as this run's purpose.

    Three signals, each measured on all 1,661 real PDFs (task-9-report.md):
      * the Acrobat "test PDF document" placeholder (43 runs, all unusable);
      * no Purpose/Goal/Objective heading anywhere in the first four pages, in which case _purpose()
        falls back to the top of page 1 -- the cover sheet or the "Resources / Machine Setup" form
        (168 runs, all unusable);
      * the extracted purpose is not plain-ASCII prose (56 runs, broken font encodings).
    `_subject(txt) is None` is deliberately *not* a fourth rule: it is true for 332 runs but for
    only 270 unusable ones, and after the three rules above it rejects a further 107 runs whose
    purpose is perfectly good text that simply has no "Subject:" line (e.g. run 20170621A,
    "Goals: As this experiment is part of the DIII-D National Campaign..."). human_tier() already
    falls back to run_title for the title in that case.
    """
    if _PDF_PLACEHOLDER.search(txt[:2000]):
        return False
    if not _PURPOSE.search(txt):
        return False
    return _prose_ratio(purpose[:800]) >= _PROSE_MIN


def _subject(text: str) -> str | None:
    m = _SUBJECT.search(text)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def _purpose(text: str, max_chars: int) -> str:
    m = _PURPOSE.search(text)
    start = m.start() if m else 0
    return re.sub(r"[ \t]+", " ", text[start : start + max_chars]).strip()


def mp_text(
    run_id: str | None, shot: int, paths: Paths, max_chars: int = 3000
) -> tuple[str | None, str | None, str]:
    """(title, purpose, source): per-shot bundle (has the MP PDF text) > miniproposal.pdf > md header.

    The bundle tier is only trusted when the bundle's own RUN_ID header equals `run_id` (which the
    caller resolves from sql/index.json, the authoritative mapping). Real corpus check
    (task-9-report.md): shotsummary/processed/per_shot_txt/shot_<N>.txt appears to have been written
    once per shot number *mentioned anywhere* in a run's text, so a shot quoted as a comparison in a
    later run got a bundle carrying that later run's mini-proposal. Of the 22,950 bundles, 10,545
    agree with index.json, 1,876 disagree outright, and 10,529 are for shots index.json does not
    know at all. Worked example: shot_161405.txt (a 2015 shot, index.json run 20150122B) carries
    "RUN_ID: 20240530" and run 20240530's material -- shot_range 198801 - 198817, "RT torbeam
    commissioning and KSTAR shape development". Every one of the 7 staged shots that has a bundle is
    such a mismatch, so for this project's shot list the bundle tier yields nothing and
    miniproposal.pdf is the load-bearing one. A bundle with no RUN_ID header, or a `run_id` of None,
    is equally unverifiable and is skipped the same way.
    """
    bundle = paths.per_shot_txt_dir / f"shot_{shot}.txt"
    if bundle.exists():
        txt = bundle.read_text(errors="replace")
        run_m = _BUNDLE_RUN.search(txt)
        m = _BUNDLE_MP.search(txt)
        if m and run_id and run_m and run_m.group(1) == run_id:
            body = txt[m.end() :]
            nxt = re.search(r"^## ", body, re.MULTILINE)
            body = (body[: nxt.start()] if nxt else body).strip()
            if len(body) > 200:
                return _subject(body), _purpose(body, max_chars), "bundle"
    if run_id:
        pdf = paths.shotsummary_raw_dir / run_id / "miniproposal.pdf"
        if pdf.exists():
            txt = _pdf_text(str(pdf), _stat_key(pdf))
            purpose = _purpose(txt, max_chars)
            if len(txt) > 200 and _pdf_usable(txt, purpose):
                return _subject(txt), purpose, "pdf"
        md = paths.shotsummary_raw_dir / run_id / "miniproposal.md"
        if md.exists():
            title = _subject(md.read_text(errors="replace"))
            if title:
                return title, None, "md_header"
    return None, None, "none"


def human_tier(shot: int, paths: Paths) -> HumanTier:
    rec = load_log_record(shot, paths) or {}
    run_id = _clean(rec.get("run")) or load_shot_index(paths).get(shot)
    meta = run_metadata(run_id, paths) or {}
    entries = parse_log_entries(_clean(rec.get("log_text")))
    title, purpose, source = mp_text(run_id, shot, paths)
    chief = next((e.text for e in entries if e.role == "CHIEF_OPERATOR"), None)
    # Real corpus check (task-9-report.md): `refshot` is null in every one of the 53,176 records of
    # logs.jsonl, so this parse never fires in production. Kept because the field exists in the
    # schema and a future export may populate it -- do not build anything on it being present.
    refshot = _clean(rec.get("refshot"))
    return HumanTier(
        mpid=_clean(rec.get("mpid")) or _clean(meta.get("mpid")),
        mp_title=title or _clean(rec.get("run_title")) or _clean(meta.get("title")),
        mp_purpose=purpose,
        mp_text_source=source,
        run_id=run_id,
        run_title=_clean(rec.get("run_title")) or _clean(meta.get("title")),
        session_leaders=str(meta.get("session_leaders") or "").split(),
        shot_brief=_clean(rec.get("shot_brief")),
        precomment=_clean(rec.get("precomment")),
        log_entries=entries,
        chief_operator_status=chief[:200] if chief else None,
        # Also null in every record of the real corpus (task-9-report.md), along with shot_ok,
        # chief_operator, plasma_shot and refshot -- five fields, not the six the task-9 brief's
        # correction C2 listed. `keywords` is the sixth name on that list and it is *not* null: it
        # is present as an always-empty list [] in all 53,179 records (re-measured in fix wave 1),
        # which is a different thing to plan around. The chief operator's assessment lives only
        # inside log_text, which is where chief_operator_status above and verdict() below get it.
        quality_comment=_clean(rec.get("quality_comment")),
        requested=parse_requested(_clean(rec.get("log_text"))),
        refshot=int(refshot) if refshot is not None and str(refshot).isdigit() else None,
        mp_step=_clean(rec.get("mp_step")),
        verdict=verdict(
            entries,
            " ".join(
                filter(None, [_clean(rec.get("quality_comment")), _clean(rec.get("shot_brief"))])
            ),
        ),
    )


# ---------------------------------------------------------------- text composition + embeddings


def compose_texts(h: HumanTier, max_log_chars: int = 2000) -> tuple[str, str]:
    """text_mp = experiment intent; text_log = what this shot's operators wrote.

    text_log opens with a head block -- the chief operator's status, the session leader's
    precomment and the requested-parameter line -- and then the log entries, VERDICT_ROLES first.
    The head block is a median 30 tokens and fits inside MiniLM's 256-token window for 626 of the
    626 staged shots that have text; head + the first verdict-role entry fits for 621 of 626. What
    truncation loses is the tail of the DIAGNOSTICS/PCS/ANALYSIS entries, which is the intended
    trade. HumanTier.verdict is computed on the whole log_entries list, never on this string.

    `shot_brief` used to be the first thing in the head and is deliberately gone (fix wave 1, item
    5). It is "autoload system <timestamp>" in all 52,871 non-null records of the corpus -- the
    same 13 tokens for every shot ever taken, with a per-shot timestamp that is actively harmful as
    an embedding feature. The old docstring said the brief went first "so the informative part
    survives"; it had it backwards about which part is informative.

    Measured on the 626 staged shots with text, with the model's own tokenizer: text_log went from
    a median 469 tokens / 89 % of shots over the 256-token limit / a median 55 % of the tokens
    reaching the encoder, to a median 412 / 85 % / 62 %. Three sources: the brief (13 tokens on
    every shot), the logbook's HTML (53 for the mini-proposal anchor alone, which landed inside the
    window for 229 of the 626), and the log entry that merely repeats the chief-operator line
    already in the head (571 of 626). The per-shot saving is a median 28 tokens, less than the ~75
    those three cost, because text_log is also capped at max_log_chars: removing text from the
    front lets more tail entries in under the cap, and 19 shots come out slightly longer. What
    improves either way is what the encoder sees, which is the point.

    The precomment is *contained in* a session-leader entry for 221 of 626 shots rather than equal
    to one, so that redundancy is left alone: dropping the head copy would push it behind however
    many entries sort ahead of that one, and dropping the entry would lose the rest of what the
    session leader wrote.

    The char caps are deliberately much larger than 256 tokens' worth (~830 chars of this text):
    text_mp/text_log are also the human-readable description fields, not only encoder input.
    """
    text_mp = " . ".join(dict.fromkeys(filter(None, [h.mp_title, h.run_title, h.mp_purpose])))
    head: list[str] = [p for p in (h.chief_operator_status, h.precomment) if p]
    if h.requested:
        r = h.requested
        line = (
            f"requested Ip {r.get('ip_MA', 0):g} MA, Bt {r.get('bt_T', 0):g} T, "
            f"PNBI {r.get('pnbi_MW', 0):g} MW, PECH {r.get('pech_MW', 0):g} MW"
        )
        # parse_requested's optional fifth field. None of the 936 staged shots has it, which is why
        # it went unnoticed, but 64 of the 4,015 corpus records with a requested block do (shots
        # 171042-171045 among them) and dropping it silently lost a parsed number.
        if "density_cm3" in r:
            line += f", ne {r['density_cm3']:g}e13 cm^-3"
        head.append(line)
    ordered = sorted(h.log_entries, key=lambda e: 0 if e.role in VERDICT_ROLES else 1)
    # An entry the head already reproduces *verbatim* is dropped, which is the <=200-char
    # chief-operator entry that chief_operator_status was taken from. A longer one is kept, since
    # the head only carries its first 200 chars -- deduplicating on a prefix would lose the rest.
    seen = set(head)
    parts = head + [f"[{e.role}] {e.text}" for e in ordered if e.text and e.text not in seen]
    return text_mp, " \n".join(parts)[:max_log_chars]


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # beat 6 larger encoders on this operator text (shotsearch)


@functools.lru_cache(maxsize=1)
def _load_model(name: str | None = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name or load_paths().sentence_transformers_model, device="cpu")


EMBED_DIM = 384  # all-MiniLM-L6-v2's width; only used when there is nothing to embed, see below


def embed_texts(texts: list[str]) -> np.ndarray:
    """Unit-norm embeddings, one row per text; an empty or whitespace-only text gives a zero row.

    The width follows whatever EMBED_MODEL produces rather than a hardcoded 384, so swapping the
    model does not silently truncate or pad. EMBED_DIM is the fallback for the one case where the
    width cannot be observed -- every text empty, so the model is never loaded -- and has to be
    changed alongside EMBED_MODEL.
    """
    idx = [i for i, t in enumerate(texts) if t and t.strip()]
    if not idx:
        return np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    vecs = np.asarray(
        _load_model().encode(
            [texts[i] for i in idx],
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
        ),
        dtype=np.float32,
    )
    out = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
    out[idx] = vecs
    return out
