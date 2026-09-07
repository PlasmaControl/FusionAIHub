"""The `recommender_v1` development universe: 500 shots chosen from corpus ∩ text (plan §5.7).

Everything in iterations 0-3 is developed, trained and evaluated on this list, so the list is a
prior on every result that follows it. The brief the user gave for it was "pretty diverse and ones
that actually are doing something, not empty", and §5.7 turns that into four eligibility rules and
a diversity quota. This module is the rules; `cli.cmd_corpus` is the I/O around them.

**Eligibility** (all four, per §5.7), evaluated per shot with the reason for every rejection kept:

* **(a) the shot table row** -- `SHOT_TYPE == plasma`, `IP-(MA) >= 0.5`, `PULSE-LENGTH >= 2 s`, and
  heating (`PBEAM-MAX >= 1 MW` or `PECH-MAX > 0`). A missing field is a rejection, not a pass: the
  2021-2025 corpus has plasma shots whose row carries six columns and no heating at all, and
  reading absence as "unconstrained" would admit them on evidence that is not there.
* **(b) the shot's own text** -- the shot-specific block must be the shot's, not the session's
  (`session_fallback`), long enough to carry a table (`shot_text`), and the run title must not name
  a machine activity (`title`).
* **(c) the census** -- `mhr`, `ece` and `filterscopes` all present with >= 2 s of coverage. This is
  the "not empty" half of the user's sentence, and it is read off the fresh census
  (`corpus_coverage.parquet`), never off a group's channel count.
* **(d) Ip flat-top >= 1 s** -- see `PROXY_RAMP_S` for which of the two sources answers this and
  why the proxy is the one that answered it for `recommender_v1`.

**Diversity** is `diversify()`: caps (<= 3 per run day, <= 5 per mini-proposal), a floor per lexicon
theme, floors for the three sparse groups, a ceiling on any one year, and a preference for shots
that already have labelmaker features or IGNITE frame codes -- capped so they cannot dominate.

Two deviations from a literal reading of §5.7, both measured, both reported by `summarize()`:

1. **`MIN_SHOT_CHARS` is 200, not the 300 the plan names.** §5.7 was written before the bundles
   were measured. The shot-specific block of a bundle that passes rule (a) runs 138-302 characters
   (median 252, 99th percentile 281) over the 5,761 such shots in corpus ∩ text: it is a table of
   at most fourteen columns, not prose. At 300 the rule admits 5 shots of 5,761; the blocks it
   would admit *elsewhere* in the corpus are the multi-kilobyte `IMPORTANT <pre> BLOCKS` PCS
   dumps, which is the opposite of the physics content the clause is reaching for. 200 keeps the
   clause doing what it was written to do -- reject a bundle that carries no table of its own --
   and `--min-shot-chars` re-opens the number.
2. **`mpid` does not come from the text.** No `mpid` appears in any of the 300 sampled bundles'
   `METADATA (selected)`; the logbook (`sql/logs.jsonl`) has one for every shot, so the CLI takes
   it from there and the <= 5 cap is honoured. When it cannot be had, `mpid is None` and those
   shots are not capped against each other (see `diversify`).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from . import text

__all__ = [
    "Candidate",
    "Quotas",
    "ShotFacts",
    "assign_theme",
    "diversify",
    "document",
    "eligible",
    "flattop_from_ip",
    "flattop_proxy",
    "format_summary",
    "lexicon_themes",
    "mpid_index",
    "parse_facts",
    "replace",
    "spans",
    "summarize",
    "with_flattop",
]

# ------------------------------------------------------------------------------ the rule's constants

RULE = "plan §5.7"

MIN_IP_MA = 0.5
MIN_PULSE_LENGTH_S = 2.0
MIN_PBEAM_MW = 1.0
MIN_FLATTOP_S = 1.0
MIN_SHOT_CHARS = 200  # see the module docstring, deviation 1
CENSUS_GROUPS = ("mhr", "ece", "filterscopes")
MIN_GROUP_SPAN_S = 2.0
# The groups whose presence the list is required to spread over. They are the sparse ones -- co2
# 43 %, bes 38 %, tangtv 44 % of the corpus -- so without a floor a greedy fill would take the
# shots that have them only by accident.
DIVERSITY_GROUPS = ("co2", "bes", "tangtv")

# `\b` on each alternative, so "Steps toward a stationary hybrid" is not a PS test and "Detachment
# ... IRTV heat flux control test" is. That second one is a real title in the pool and a real
# false positive of this rule: 108 of the 5,572 shots that pass (a)+(c) are dropped by it, and a
# handful of those are physics runs whose title happens to contain "test". The rule is §5.7's and
# is applied as written; `hand_review: add:` in the generated YAML is where a reviewer puts one
# back.
TITLE_EXCLUDE = re.compile(r"(?i)\b(?:test|ps|bcoil|abort|mag\s+cal)\b")

# Ip flat-top, measured: the longest contiguous run of |Ip| at or above this fraction of the 95th
# percentile of |Ip|. The 95th percentile rather than the max because the current overshoot at the
# end of the ramp-up is a single sample on a 25 ms grid, and dividing by it would shorten every
# flat-top by the length of that overshoot.
FLATTOP_FRACTION = 0.8

# The pulse-length proxy for flat-top, and the whole reason it exists.
#
# §5.7 wants Ip flat-top from EFIT/Ip. There are two sources: labelmaker's feature files
# (`<shot>_features.h5`, group `ip`) exist for 526 shots, and fdp can produce the rest -- but only
# if the shortlist is small enough to be worth a per-shot fdp call. It was not: after rules
# (a)-(c) the pool is 5,572 shots, six times the ~900 the brief set as the ceiling for that path.
# So the proxy answers rule (d) for the shots with no feature file, and `flattop_source` records
# which of the two answered for every shot in the list.
#
# The constant is measured, not assumed. Over the 319 shots that have BOTH a labelmaker `ip`
# feature and a `PULSE-LENGTH` in their bundle, `PULSE-LENGTH - flattop_from_ip(ip)` has median
# 1.276 s (IQR 1.06-1.74) -- the ramp-up plus the ramp-down. At 1.27 the proxy agrees with the
# measured flat-top on the >= 1 s question for 316 of those 319 shots (99.1 %), and all three
# disagreements are in the safe direction: the proxy rejects a shot whose real flat-top clears 1 s,
# never the other way round. A selection rule that errs by dropping a good shot costs a shot; one
# that errs by admitting a bad one costs a training set.
PROXY_RAMP_S = 1.27


# ------------------------------------------------------------------------------ what a shot is


@dataclass(frozen=True)
class ShotFacts:
    """Everything the eligibility rules read about one shot, out of its text bundle.

    `flattop_s`/`flattop_source` start as the pulse-length proxy and are replaced by
    `with_flattop` when a measured Ip trace is available. Every numeric field is `None` when the
    shot table row did not carry it -- never 0.0, which the rules would read as a measurement.
    """

    shot: int
    run_id: str | None
    mpid: str | None
    year: int | None
    title: str | None
    shot_type: str | None
    ip_ma: float | None
    pulse_length_s: float | None
    pbeam_max_mw: float | None
    pech_max_mw: float | None
    shot_chars: int
    has_shot_table: bool
    flattop_s: float
    flattop_source: str


@dataclass(frozen=True)
class Candidate:
    """An eligible shot, with the facts `diversify` allocates on. `reason` is filled in by
    `diversify` with the quota that admitted it."""

    shot: int
    run_id: str | None
    mpid: str | None
    year: int | None
    theme: str | None
    has_co2: bool
    has_bes: bool
    has_tangtv: bool
    flattop_s: float
    flattop_source: str
    preferred: bool
    reason: str = ""


@dataclass(frozen=True)
class Quotas:
    """The diversity budget. `n` is the target size; everything else is a cap or a floor.

    `per_theme`, `group_min` and `preferred_cap` are floors and ceilings on *phases* of the fill,
    not post-hoc filters: a floor that cannot be met takes what the pool has and says so in the
    summary, and a ceiling stops a phase rather than removing anything already selected.
    """

    n: int = 500
    per_run: int = 3
    per_mpid: int = 5
    per_theme: int = 20
    max_year_frac: float = 0.40
    group_min: Mapping[str, int] = field(
        default_factory=lambda: {"co2": 50, "bes": 50, "tangtv": 30}
    )
    preferred_cap: int = 150
    # Excluded from the theme quotas by §5.7: a floor of 20 on "startup/checkout/calibration" is a
    # floor on the shots the list exists to avoid. It is still a theme, and shots carrying it can
    # still arrive through the general fill -- where they compete round-robin with the other
    # thirteen instead of taking the 23 % of the pool they hold.
    exclude_theme: str = "startup_checkout"


# ------------------------------------------------------------------------------ parsing a bundle


def _num(v: str | None) -> float | None:
    """A shot table value as a float, or None.

    `str.replace(" ", "")` because the summary page's HTML-to-text conversion splits some
    exponents: real rows carry `KAPPA: 1.9 E15`. A value that is not a number at all (an empty
    cell, a stray word) is None -- absent, which every rule treats as a rejection.
    """
    if v is None:
        return None
    try:
        return float(str(v).replace(" ", ""))
    except ValueError:
        return None


def _year(run_id: str | None) -> int | None:
    """The campaign year out of a run id (`20220301`, `20220630A`). None if it is not a run id."""
    if not run_id or len(run_id) < 4 or not run_id[:4].isdigit():
        return None
    return int(run_id[:4])


def parse_facts(shot: int, bundle: str, *, mpid: str | None = None) -> ShotFacts:
    """One shot's facts out of its `shot_<N>.txt`. Pure -- no filesystem, no census."""
    row = text.shot_table_row(bundle)
    meta = text.session_metadata(bundle)
    run_m = text._BUNDLE_RUN.search(bundle)
    run_id = run_m.group(1) if run_m else (meta.get("run_id") or None)
    pulse = _num(row.get("PULSE-LENGTH"))
    return ShotFacts(
        shot=int(shot),
        run_id=run_id,
        mpid=mpid,
        year=_year(run_id),
        title=(meta.get("title") or None),
        shot_type=(row.get("SHOT_TYPE") or None),
        ip_ma=_num(row.get("IP-(MA)")),
        pulse_length_s=pulse,
        pbeam_max_mw=_num(row.get("PBEAM-MAX-(MW)")),
        pech_max_mw=_num(row.get("PECH-MAX-(MW)")),
        shot_chars=len(text.shot_block(bundle)),
        has_shot_table=bool(row),
        flattop_s=flattop_proxy(pulse),
        flattop_source="pulse_length_proxy",
    )


def with_flattop(facts: ShotFacts, flattop_s: float, source: str) -> ShotFacts:
    """`facts` with a measured flat-top in place of the proxy, and the source that measured it."""
    return replace(facts, flattop_s=float(flattop_s), flattop_source=source)


# ------------------------------------------------------------------------------ rule (d) sources


def flattop_from_ip(t_s, ip) -> float:
    """The longest contiguous stretch, in seconds, of `|ip|` at or above `FLATTOP_FRACTION` of its
    95th percentile. NaN when the trace carries nothing to measure.

    Sign-blind because DIII-D runs both polarities and the shot table's `IP-(MA)` is a magnitude.
    Non-finite samples are dropped rather than treated as low current, which would cut a flat-top
    in two at every dropout.
    """
    t = np.asarray(t_s, dtype=float).ravel()
    y = np.abs(np.asarray(ip, dtype=float).ravel())
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    if t.size < 3:
        return float("nan")
    plateau = float(np.percentile(y, 95))
    if not math.isfinite(plateau) or plateau <= 0.0:
        return float("nan")
    above = y >= FLATTOP_FRACTION * plateau
    best, i = 0.0, 0
    while i < above.size:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < above.size and above[j + 1]:
            j += 1
        best = max(best, float(t[j] - t[i]))
        i = j + 1
    return best


def flattop_proxy(pulse_length_s: float | None) -> float:
    """`PULSE-LENGTH` minus the measured ramp time (`PROXY_RAMP_S`), floored at zero.

    NaN when there is no pulse length -- a shot with no pulse length has already failed rule (a),
    and a zero here would be a flat-top measurement of a shot nobody measured.
    """
    if pulse_length_s is None or not math.isfinite(float(pulse_length_s)):
        return float("nan")
    return max(0.0, float(pulse_length_s) - PROXY_RAMP_S)


# ------------------------------------------------------------------------------ the census side


def spans(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    """`{shot: {group: span_s}}` for the groups the census says actually recorded.

    Absent from the mapping means the census has no PRESENT row for it: a placeholder, an
    unopenable file, or a shot the census never saw. All three are "this diagnostic did not record
    for this shot", which is what rule (c) asks; the three are told apart in the census table
    itself, not here.
    """
    present = df.loc[df["present"] & (df["group"] != "")]
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for shot, group, t0, t1 in zip(
        present["shot"], present["group"], present["t0_s"], present["t1_s"], strict=True
    ):
        span = float(t1) - float(t0)
        if math.isfinite(span):
            key = str(group)
            out[int(shot)][key] = max(out[int(shot)].get(key, span), span)
    return dict(out)


# ------------------------------------------------------------------------------ eligibility


def eligible(
    facts: ShotFacts,
    group_spans: Mapping[str, float],
    *,
    min_shot_chars: int = MIN_SHOT_CHARS,
) -> tuple[bool, tuple[str, ...]]:
    """`(ok, reasons)` for one shot against §5.7's rules (a)-(d).

    EVERY failing clause is reported, not the first: the summary block counts how often each rule
    fired, and a short-circuit would attribute every rejection to whichever rule happens to be
    evaluated first. `reasons` is empty exactly when `ok`.
    """
    bad: list[str] = []

    # (a) the shot table row
    if (facts.shot_type or "").strip().lower() != "plasma":
        bad.append("shot_type")
    if facts.ip_ma is None or facts.ip_ma < MIN_IP_MA:
        bad.append("ip")
    if facts.pulse_length_s is None or facts.pulse_length_s < MIN_PULSE_LENGTH_S:
        bad.append("pulse_length")
    beam = facts.pbeam_max_mw is not None and facts.pbeam_max_mw >= MIN_PBEAM_MW
    ech = facts.pech_max_mw is not None and facts.pech_max_mw > 0.0
    if not (beam or ech):
        bad.append("heating")

    # (b) the shot's own text
    if not facts.has_shot_table:
        bad.append("session_fallback")
    if facts.shot_chars < min_shot_chars:
        bad.append("shot_text")
    if facts.title and TITLE_EXCLUDE.search(facts.title):
        bad.append("title")

    # (c) the census
    for group in CENSUS_GROUPS:
        if group_spans.get(group, 0.0) < MIN_GROUP_SPAN_S:
            bad.append(f"census_{group}")

    # (d) Ip flat-top
    # `not (x >= y)` and not `x < y`: flattop_s is NaN for a shot with no pulse length at all,
    # and NaN < 1.0 is False -- which would let exactly the shots with no measurement through.
    if not (facts.flattop_s >= MIN_FLATTOP_S):
        bad.append("flattop")

    return (not bad), tuple(bad)


# ------------------------------------------------------------------------------ themes


def lexicon_themes() -> list[dict]:
    """The 14 keyword themes of configs/ideate/labels.yaml, in file order (first match wins)."""
    return list(config.load_yaml("labels.yaml").get("themes") or [])


def assign_theme(title: str | None, themes: Sequence[Mapping] | None = None) -> str | None:
    """The first lexicon theme whose keywords appear in `title`, or None.

    The title is lowercased and given a space at each end, which is what makes labels.yaml's
    `" nt "` and `"rt "` anchor on word boundaries -- the same normalisation the label rules use,
    so a shot's theme here and its theme there cannot disagree.
    """
    if not title:
        return None
    hay = " " + re.sub(r"\s+", " ", str(title).lower()).strip() + " "
    for entry in themes if themes is not None else lexicon_themes():
        if any(str(k).lower() in hay for k in entry.get("keywords", ())):
            return str(entry["id"])
    return None


# ------------------------------------------------------------------------------ diversification


class _Allocator:
    """The caps, and the one place a candidate is accepted.

    A candidate is admitted only if it clears every cap at once, and admitting it updates every
    counter -- so the phases below cannot each keep their own idea of how full a run day is.
    """

    def __init__(self, quotas: Quotas) -> None:
        self.q = quotas
        self.year_cap = max(1, int(quotas.n * quotas.max_year_frac))
        self.taken: dict[int, Candidate] = {}
        self._runs: Counter[str] = Counter()
        self._mpids: Counter[str] = Counter()
        self._years: Counter[int] = Counter()
        self._preferred = 0

    def full(self) -> bool:
        return len(self.taken) >= self.q.n

    def add(self, c: Candidate, reason: str) -> bool:
        if c.shot in self.taken or self.full():
            return False
        if c.run_id is not None and self._runs[c.run_id] >= self.q.per_run:
            return False
        # A candidate with no mpid is not capped: the alternative is one bucket holding every shot
        # whose logbook record has no mini-proposal, which would cap them at five between them --
        # a cap on the absence of a field rather than on concentration in an experiment.
        if c.mpid is not None and self._mpids[c.mpid] >= self.q.per_mpid:
            return False
        if c.year is not None and self._years[c.year] >= self.year_cap:
            return False
        # The preferred cap is global, not a budget for phase 1. §5.7 caps these shots so they
        # cannot dominate the list, and a cap that only bound the phase that seeks them out would
        # be undone by the general fill picking up the other 376 of them.
        if c.preferred and self._preferred >= self.q.preferred_cap:
            return False
        self.taken[c.shot] = replace(c, reason=reason)
        self._preferred += bool(c.preferred)
        if c.run_id is not None:
            self._runs[c.run_id] += 1
        if c.mpid is not None:
            self._mpids[c.mpid] += 1
        if c.year is not None:
            self._years[c.year] += 1
        return True


def _shuffled(candidates: Iterable[Candidate], seed: int) -> list[Candidate]:
    """`candidates` in a deterministic pseudo-random order.

    Sorted by shot FIRST, so the order depends on the seed and the membership of the pool and not
    on the order the caller's filesystem happened to hand the bundles over.
    """
    out = sorted(candidates, key=lambda c: c.shot)
    random.Random(seed).shuffle(out)
    return out


def _by_year(candidates: Sequence[Candidate], seed: int) -> Iterator[Candidate]:
    """`candidates` round-robin over their years, each year's shots in seeded order.

    This is what spreads the list over the campaign: taking a bucket in flat seeded order gives
    back the pool's own year distribution (2022 is 31 % of it), while cycling the years gives each
    campaign year an equal claim on the bucket until it runs out.
    """
    buckets: dict[int | None, list[Candidate]] = defaultdict(list)
    for c in _shuffled(candidates, seed):
        buckets[c.year].append(c)
    # Within a year, a preferred shot goes first. This is §5.7's "prefer the labelmaker-featured
    # and IGNITE frame-code shots when they qualify" applied where it belongs -- as a tie-break
    # inside every phase, so a shot whose features already exist is the one a theme quota takes
    # when it has a free choice, rather than as a phase that spends run-day slots ahead of the
    # floors. The global `preferred_cap` still bounds the total; a stable sort keeps the seeded
    # order inside each of the two halves.
    for bucket in buckets.values():
        bucket.sort(key=lambda c: not c.preferred)
    order = sorted(buckets, key=lambda y: (y is None, y or 0))
    while any(buckets[y] for y in order):
        for y in order:
            if buckets[y]:
                yield buckets[y].pop(0)


def _round_robin(streams: Mapping[object, Iterator[Candidate]]) -> Iterator[Candidate]:
    """One item from each stream in key order, until every stream is exhausted."""
    live = {k: streams[k] for k in sorted(streams, key=lambda k: (k is None, str(k)))}
    while live:
        for key in list(live):
            try:
                yield next(live[key])
            except StopIteration:
                del live[key]


def diversify(
    candidates: Sequence[Candidate], quotas: Quotas, seed: int
) -> list[Candidate]:
    """Choose `quotas.n` of `candidates`, deterministically given `seed`. Returned in shot order.

    Four phases, each drawing from the same pool through the same `_Allocator`, so a shot admitted
    by an earlier phase is simply not available to a later one and every cap is global:

    1. **theme floors** -- each lexicon theme except `exclude_theme` up to `per_theme`. A theme the
       pool cannot fill takes what there is; `summarize` reports the shortfall and how many
       candidates that theme had, because the two ways to fall short look identical in the count.
    2. **group floors** -- `co2`, `bes`, `tangtv` up to `group_min`, from the candidates that carry
       the group.
    3. **preferred** -- shots that already have labelmaker features or IGNITE frame codes, up to
       `preferred_cap`. AFTER the floors, deliberately. §5.7 states the theme and group quotas as
       floors (">= 20 per theme where available") and the preferred list as a preference with a
       ceiling ("prefer ... capped at 150"); running the preference first spends run-day slots that
       a floor then cannot get back, and measured on the real pool it costs three themes their
       quota. A preference that is satisfied last is still satisfied; a floor that is missed is
       a hole in the list.
    4. **fill** -- round-robin over themes, and within a theme round-robin over years, until `n`.

    Every phase walks its candidates in `_by_year` order, so the year spread is not something the
    last phase has to repair.
    """
    alloc = _Allocator(quotas)
    pool = sorted(candidates, key=lambda c: c.shot)

    if quotas.per_theme > 0:
        themes = sorted({c.theme for c in pool} - {None, quotas.exclude_theme}, key=str)
        for theme in themes:
            got = sum(1 for c in alloc.taken.values() if c.theme == theme)
            for c in _by_year([x for x in pool if x.theme == theme], seed):
                if got >= quotas.per_theme or alloc.full():
                    break
                if alloc.add(c, f"theme:{theme}"):
                    got += 1

    for group, floor in sorted(quotas.group_min.items()):
        attr = f"has_{group}"
        got = sum(1 for c in alloc.taken.values() if getattr(c, attr, False))
        for c in _by_year([x for x in pool if getattr(x, attr, False)], seed):
            if got >= floor or alloc.full():
                break
            if alloc.add(c, f"group:{group}"):
                got += 1

    if quotas.preferred_cap > 0:
        preferred = sum(1 for c in alloc.taken.values() if c.preferred)
        for c in _by_year([c for c in pool if c.preferred], seed):
            if preferred >= quotas.preferred_cap or alloc.full():
                break
            preferred += alloc.add(c, "preferred")

    streams = {
        theme: _by_year([c for c in pool if c.theme == theme], seed)
        for theme in {c.theme for c in pool}
    }
    for c in _round_robin(streams):
        if alloc.full():
            break
        alloc.add(c, "fill")

    return sorted(alloc.taken.values(), key=lambda c: c.shot)


# ------------------------------------------------------------------------------ the report


def summarize(
    *,
    n_candidates: int,
    reasons: Counter,
    selected: Sequence[Candidate],
    quotas: Quotas,
    candidates: Sequence[Candidate] | None = None,
) -> dict:
    """The summary block: how big the pool was, what rejected the rest, and what came out.

    `reasons` counts rule failures over the whole pool and a shot may appear in several of its
    entries -- `eligible` reports every clause that fired, so these are counts of REJECTIONS BY
    RULE and not a partition of the rejected shots.
    """
    themes = Counter(c.theme for c in selected)
    have = Counter(c.theme for c in candidates or ())
    unmet = {}
    for entry in lexicon_themes():
        tid = str(entry["id"])
        if tid == quotas.exclude_theme:
            continue
        got = themes.get(tid, 0)
        if got < quotas.per_theme:
            # `available` is the count of ELIGIBLE candidates carrying the theme. A quota can miss
            # for two reasons that look identical in `got` -- the pool has no more shots of that
            # theme, or the <= 3 per run day and <= 5 per mini-proposal caps ran out first, which
            # is what happens to a theme that lives on two run days -- and the reader has to be
            # able to tell them apart without re-running the selection.
            unmet[tid] = {"want": quotas.per_theme, "got": got, "available": int(have.get(tid, 0))}
    return {
        "n_eligible": int(n_candidates),
        "n_selected": len(selected),
        "failed": {k: int(v) for k, v in sorted(reasons.items())},
        "theme": {(k or "none"): int(v) for k, v in sorted(themes.items(), key=lambda kv: str(kv[0]))},
        "theme_quota_unmet": unmet,
        "year": {int(k): int(v) for k, v in sorted(Counter(c.year for c in selected).items()) if k},
        "groups": {g: int(sum(getattr(c, f"has_{g}") for c in selected)) for g in DIVERSITY_GROUPS},
        "preferred": int(sum(c.preferred for c in selected)),
        "reason": {k: int(v) for k, v in sorted(Counter(c.reason for c in selected).items())},
        "flattop_source": {
            k: int(v) for k, v in sorted(Counter(c.flattop_source for c in selected).items())
        },
        "runs": len({c.run_id for c in selected}),
        "mpids": len({c.mpid for c in selected if c.mpid}),
    }


def document(
    selected: Sequence[Candidate],
    summary: Mapping,
    *,
    name: str,
    seed: int,
    n: int,
    rule: str = RULE,
) -> dict:
    """The shot-list YAML document, in `config.load_shot_list`'s format.

    `hand_review` is always written empty: it is the human gate on a generated list, and the
    generator's job is to leave the block there, not to fill it in.
    """
    return {
        "name": name,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "rule": rule,
        "seed": int(seed),
        "n": int(n),
        "summary": dict(summary),
        "hand_review": {"drop": [], "add": []},
        "shots": [
            {
                "shot": int(c.shot),
                "run_id": c.run_id,
                "mpid": c.mpid,
                "year": c.year,
                "theme": c.theme,
                "reason": c.reason,
                "has_co2": bool(c.has_co2),
                "has_bes": bool(c.has_bes),
                "has_tangtv": bool(c.has_tangtv),
                "flattop_s": round(float(c.flattop_s), 3),
                "flattop_source": c.flattop_source,
            }
            for c in selected
        ],
    }


def format_summary(summary: Mapping) -> str:
    """The summary block as the text `ideate corpus select` prints."""
    head = (
        f"{summary['n_eligible']:,} eligible -> {summary['n_selected']:,} selected"
        f"  ({summary['runs']} run days, {summary['mpids']} mini-proposals,"
        f" {summary['preferred']} preferred)"
    )
    lines = [
        head,
        "",
        "rejected by rule (a shot may fail several):",
    ]
    lines += [f"  {k:<22}{v:>8,}" for k, v in summary["failed"].items()]
    for title, key in (("theme", "theme"), ("year", "year"), ("group", "groups")):
        lines += ["", f"{title}:"]
        lines += [f"  {k!s:<22}{v:>8,}" for k, v in summary[key].items()]
    if summary["theme_quota_unmet"]:
        lines += ["", "theme quotas not met (got / want, and how many eligible shots there were):"]
        lines += [
            f"  {k:<22}{v['got']:>8,} of {v['want']:<6}{v.get('available', '?'):>8,} eligible"
            for k, v in summary["theme_quota_unmet"].items()
        ]
    lines += ["", "admitted by:"]
    lines += [f"  {k:<22}{v:>8,}" for k, v in summary["reason"].items()]
    lines += ["", "flat-top source:"]
    lines += [f"  {k:<22}{v:>8,}" for k, v in summary["flattop_source"].items()]
    return "\n".join(lines)


# ------------------------------------------------------------------------------ loading the pool


def read_bundles(text_dir: Path, shots: Iterable[int]) -> dict[int, str]:
    """`{shot: bundle text}` for the shots that have a `shot_<N>.txt`."""
    text_dir = Path(text_dir)
    out = {}
    for shot in shots:
        p = text_dir / f"shot_{int(shot)}.txt"
        if p.exists():
            out[int(shot)] = p.read_text(errors="replace")
    return out


def preferred_shots(*, features_dir: Path | None, frame_codes_dir: Path | None) -> set[int]:
    """Shots whose labelmaker features or IGNITE frame codes already exist.

    §5.7 prefers them because they are the shots a training run can use today: their features are
    computed and their codes are encoded, so admitting them costs nothing that has not been paid.
    """
    out: set[int] = set()
    if features_dir and Path(features_dir).is_dir():
        for p in Path(features_dir).glob("*_features.h5"):
            stem = p.stem.split("_")[0]
            if stem.isdigit():
                out.add(int(stem))
    if frame_codes_dir and Path(frame_codes_dir).is_dir():
        for p in Path(frame_codes_dir).glob("*.pt"):
            if p.stem.isdigit():
                out.add(int(p.stem))
    return out


def mpid_index(logs_jsonl: Path | None, shots: Iterable[int]) -> dict[int, str | None]:
    """`{shot: mpid}` for `shots`, streamed once out of the logbook JSONL.

    §5.7 caps the list at five shots per mini-proposal, and the per-shot bundles carry no `mpid`
    at all -- 0 of 300 sampled `METADATA (selected)` blocks have one -- so without the logbook
    that cap would simply be skipped. `sql/logs.jsonl` has an `mpid` for every record.

    Streamed on the raw bytes with `text._SHOT_PREFIX`, the same trick `text.build_logs_subset`
    uses and for the same measured reason: every one of the 53,179 lines of the real 616 MB file
    begins `{"shot": <digits>,`, so only the wanted lines are ever handed to `json.loads`. Unlike
    `build_logs_subset` this writes no cache -- a selection run reads the file once and wants the
    one field, not a 60 MB subset of full records on disk.

    A shot with no record, or a record whose `mpid` is null, comes back None -- which `diversify`
    reads as "not capped against the others", never as a shared bucket.
    """
    want = {int(s) for s in shots}
    out: dict[int, str | None] = dict.fromkeys(sorted(want))
    if not logs_jsonl or not Path(logs_jsonl).exists():
        return out
    with open(logs_jsonl, "rb") as f:
        for line in f:
            m = text._SHOT_PREFIX.match(line)
            if not m or int(m.group(1)) not in want:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a torn tail line is not a reason to lose the rest
                continue
            out[int(m.group(1))] = text._clean(rec.get("mpid"))
    return out


def measured_flattop(shot: int, features_dir: Path | None) -> float | None:
    """Ip flat-top from labelmaker's `<shot>_features.h5`, or None when there is no such file.

    Read through h5py directly rather than through `labelmaker.features.store` so that `ideate`
    does not import `labelmaker` for one array; the layout (`ip/xdata`, `ip/ydata`) is the one
    that module writes and the one the corpus uses everywhere else.
    """
    if not features_dir:
        return None
    path = Path(features_dir) / f"{int(shot)}_features.h5"
    if not path.exists():
        return None
    import h5py

    try:
        with h5py.File(path, "r", locking=False) as f:
            if "ip" not in f or "xdata" not in f["ip"] or "ydata" not in f["ip"]:
                return None
            got = flattop_from_ip(f["ip"]["xdata"][:], f["ip"]["ydata"][:])
    except (OSError, KeyError, ValueError):
        return None
    return None if not math.isfinite(got) else float(got)
