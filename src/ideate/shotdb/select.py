"""The `recommender_v1` development universe: 500 shots chosen from corpus ∩ text (plan §5.7).

Everything in iterations 0-3 is developed, trained and evaluated on this list, so the list is a
prior on every result that follows it. The brief the user gave for it was "pretty diverse and ones
that actually are doing something, not empty", and §5.7 turns that into four eligibility rules and
a diversity quota. This module is the rules; `cli.cmd_corpus` is the I/O around them.

**Eligibility** (all four, per §5.7), evaluated per shot with the reason for every rejection kept:

* **(a) the shot table row** -- `SHOT_TYPE == plasma`, `abs(IP-(MA)) >= 0.5`, `PULSE-LENGTH >= 2 s`,
  and heating (`PBEAM-MAX >= 1 MW` or `PECH-MAX > 0`). A missing field is a rejection, not a pass:
  the 2021-2025 corpus has plasma shots whose row carries six columns and no heating at all, and
  reading absence as "unconstrained" would admit them on evidence that is not there.
* **(b) the shot's own text** -- the shot-specific block must be the shot's, not the session's
  (`session_fallback`), long enough to carry a table (`shot_text`), and the run title must not name
  a machine activity (`title`).
* **(c) the census** -- `mhr`, `ece` and `filterscopes` all present with >= 2 s of coverage. This is
  the "not empty" half of the user's sentence, and it is read off the fresh census
  (`corpus_coverage.parquet`), never off a group's channel count.
* **(d) Ip flat-top >= 1 s** -- in TWO passes. The first estimates it from `PULSE-LENGTH` for
  every shot of the 13,313-bundle pool (`PROXY_RAMP_S`, and the measured Ip trace wherever a
  labelmaker feature file already exists); the second (`verify_flattop`, `--verify-flattop`) runs
  over the 500 SELECTED shots only, where the per-shot cost of a real measurement is affordable,
  replaces every proxy with the measured number it can, drops what the measurement rejects and
  fills the hole from the same theme. A selected shot with no feature file is `pending`, not
  `pulse_length_proxy`: it is listed for labelmaker's features stage, and the run that finishes
  it is `--finalize --from-list <the committed yaml>` (`reverify_flattop`), which re-measures
  EXACTLY the committed shots and never re-selects -- a plain re-run sees a bigger feature store
  than the selection did and returns a different list, so the loop would never converge.

**Amendment (2026-09-07 review).** Four readings of §5.7 are this module's, not the plan text's,
and each is recorded here and in the generated YAML's `rule:` string because the plan is frozen:

* `abs(IP-(MA)) >= 0.5`, not the signed value. DIII-D runs both polarities and the logbook records
  a reversed-Ip discharge as a negative `IP-(MA)`; the signed comparison was a direction filter
  nobody wrote, and it rejected all 663 reversed-Ip plasma shots of the pool -- among them all 254
  whose run title matches the `qh_mode` lexicon, over 13 run days, which is why v1 of the list had
  zero `qh_mode` shots. Eligible pool 5,299 -> 5,812; `qh_mode` 185 -> 206 eligible.
* `MIN_SHOT_CHARS` 200, not 300 (deviation 1 below).
* Themes are assigned physics-first and matched against the mini-proposal subject as well as the
  run-day title (`assign_theme`).
* Rule (d) is the two-pass verification above.

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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
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
    "candidates_from_rows",
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
    "reverify_flattop",
    "spans",
    "store_fingerprint",
    "summarize",
    "verify_flattop",
    "with_flattop",
]

# ------------------------------------------------------------------------------ the rule's constants

RULE = (
    "plan §5.7 as amended 2026-09-07: abs(Ip), ≥200 shot chars, physics-first themes, "
    "two-pass flat-top"
)

# The seed the committed `recommender_v1` was drawn with. Named here rather than only in the CLI's
# `--seed` default because `--from-list` takes the seed off the document it is re-verifying, and
# the two have to be the same number when neither the caller nor the document says otherwise.
DEFAULT_SEED = 20260907

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

# The one theme that is a statement about the MACHINE and not about the physics, and the only one
# §5.7 excludes from the quotas. `assign_theme` gives it last (see there); `Quotas.exclude_theme`
# keeps it out of the floors. It is deliberately NOT moved to the end of `configs/ideate/labels.yaml`
# instead: `retrieval.scenarios.themes_of` returns every matching theme and `labels.claims` keys a
# dict by id, so neither depends on the order -- but `labels.yaml` is also read by the parallel
# labelmaker workstream, and a selection rule is the wrong place from which to renumber a shared
# lexicon.
FALLBACK_THEME = "startup_checkout"

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

# What answered rule (d) for a shot. `pending` is not a measurement and not an estimate: it is a
# selected shot whose flat-top nobody has measured yet, carrying the proxy number as its estimate
# and waiting for labelmaker's features stage. A list that still has any is refused by
# `--finalize` and by `corpus select --verify-flattop` unless `--allow-pending` is given.
FLATTOP_PROXY = "pulse_length_proxy"
FLATTOP_MEASURED = "features_ip"
FLATTOP_PENDING = "pending"


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
    mp_subject: str | None = None

    @property
    def ip_sign(self) -> int | None:
        """+1 forward, -1 reversed, None when the row carried no current at all.

        Rule (a) reads `abs(ip_ma)`, so the polarity does not decide anything -- but it decides a
        great deal about the plasma, and 663 of the pool's shots have it negative. Carrying it
        into the YAML is what lets a reader see that a counter-Ip shot is one, instead of finding
        out from the traces.
        """
        return _sign(self.ip_ma)


def _sign(x: float | None) -> int | None:
    if x is None or not math.isfinite(float(x)) or float(x) == 0.0:
        return None
    return 1 if float(x) > 0 else -1


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
    ip_sign: int | None = None
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
    exclude_theme: str = FALLBACK_THEME


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
        flattop_source=FLATTOP_PROXY,
        # Every subject line of the mini-proposal block, joined -- see `text.mp_subjects` for why
        # there are two of them and why both are kept. `assign_theme` matches on it alongside the
        # run-day title, which is what labels.yaml's header has always promised.
        mp_subject=" | ".join(text.mp_subjects(bundle)) or None,
    )


def with_flattop(facts: ShotFacts, flattop_s: float, source: str) -> ShotFacts:
    """`facts` with a measured flat-top in place of the proxy, and the source that measured it."""
    return replace(facts, flattop_s=float(flattop_s), flattop_source=source)


# ------------------------------------------------------------------------------ rule (d) sources


def flattop_from_ip(t_s, ip) -> float:
    """The longest contiguous stretch, in seconds, of `|ip|` at or above `FLATTOP_FRACTION` of its
    95th percentile. NaN when the trace carries nothing to measure.

    Sign-blind because DIII-D runs both polarities: `|Ip|` is what "flat-top" is about, and a
    reversed-Ip trace is otherwise indistinguishable from one that never got off zero. (The shot
    table's `IP-(MA)` is signed for the same reason, which rule (a) reads with `abs` -- this
    docstring used to claim it was a magnitude, and that claim is what the signed comparison in
    `eligible` was resting on.) Non-finite samples are dropped rather than treated as low current,
    which would cut a flat-top in two at every dropout.
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
    # `abs`: the logbook's `IP-(MA)` is SIGNED and DIII-D runs both polarities. `>= 0.5` on the
    # signed number reads as "at least 0.5 MA and forward-going", which is a physics filter the
    # plan never asked for and which rejected all 663 reversed-Ip plasma shots of the pool --
    # including all 254 whose title matches the `qh_mode` lexicon, over 13 run days, which is the
    # theme §5.7 most wanted. The polarity is kept (`ShotFacts.ip_sign`) rather than tested.
    if facts.ip_ma is None or abs(facts.ip_ma) < MIN_IP_MA:
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


def assign_theme(
    title: str | None,
    themes: Sequence[Mapping] | None = None,
    *,
    subject: str | None = None,
    fallback: str = FALLBACK_THEME,
) -> str | None:
    """The first PHYSICS lexicon theme whose keywords appear in the run title or the mini-proposal
    subject; `fallback` (`startup_checkout`) only when none does; None when nothing matches.

    Two things here are the 2026-09-07 amendment, both because a theme is what §5.7's quotas
    allocate on and a shot filed under the wrong one is a shot the quota that wanted it never saw:

    * **Physics themes are tried first.** `labels.yaml` lists `startup_checkout` first and the
      assignment was first-match, so every run day whose title also said "checkout",
      "calibration" or "commissioning" was filed as machine time -- 521 of the 1,394
      checkout-titled eligible shots also match a physics theme, and `startup_checkout` is the
      one theme §5.7 excludes from the quotas. "Divertor diagnostic checkout for QH/WPQH-Mode" is
      a real run day, and it is a QH-mode run day. The fallback is still assigned when nothing
      physical matches, so the summary can still count genuine machine time.
    * **The mini-proposal subject is matched too**, which is what `labels.yaml`'s own header has
      said since it was written ("matched against run title + MP title") and what was never
      implemented. The subject is the experiment's words; the run-day title is the session
      leader's shorthand for the day, and for 294 of the 1,084 eligible shots the title alone
      leaves without a theme, the subject supplies one.

    Everything is lowercased and given a space at each end, which is what makes labels.yaml's
    `" nt "` and `"rt "` anchor on word boundaries -- the same normalisation the label rules use,
    so a shot's theme here and its theme there cannot disagree. Title and subject are joined by
    `" | "` so no keyword can straddle the two.
    """
    parts = [str(p) for p in (title, subject) if p]
    if not parts:
        return None
    hay = " " + re.sub(r"\s+", " ", " | ".join(parts).lower()).strip() + " "
    entries = list(themes if themes is not None else lexicon_themes())
    physics = [e for e in entries if str(e.get("id")) != fallback]
    for entry in physics + [e for e in entries if str(e.get("id")) == fallback]:
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

    def add(self, c: Candidate, reason: str, *, force: bool = False) -> bool:
        """Admit `c` if every cap allows it. `force` admits it anyway and still counts it.

        `force` is for `reverify_flattop`, and for nothing else: the shots of an already committed
        list were allocated once, by the run that made the list and against the pool it saw, and
        re-imposing the caps on them at re-verification time would let the second invocation
        quietly shrink the list. They are still COUNTED, so a replacement drawn afterwards sees
        the run day, mini-proposal and year they occupy.
        """
        if c.shot in self.taken:
            return False
        if force:
            self._count(c, reason)
            return True
        if self.full():
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
        self._count(c, reason)
        return True

    def _count(self, c: Candidate, reason: str) -> None:
        self.taken[c.shot] = replace(c, reason=reason)
        self._preferred += bool(c.preferred)
        if c.run_id is not None:
            self._runs[c.run_id] += 1
        if c.mpid is not None:
            self._mpids[c.mpid] += 1
        if c.year is not None:
            self._years[c.year] += 1


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


# ------------------------------------------------------- rule (d), pass two: the verification


def verify_flattop(
    selected: Sequence[Candidate],
    candidates: Sequence[Candidate],
    quotas: Quotas,
    seed: int,
    measure: Callable[[int], float | None],
) -> tuple[list[Candidate], list[dict]]:
    """Re-run rule (d) on the SELECTED shots with a measured Ip trace, and refill what it drops.

    `measure(shot)` returns the measured flat-top in seconds, or None when there is nothing to
    measure it from (no feature file yet). Returns `(selected, replacements)`, the list in shot
    order and one replacement record per shot the measurement dropped.

    This is the pass that makes rule (d) bite. The brief allows the expensive source only when the
    shortlist is small; the whole pool is 5,812 shots and far past that, but the LIST is 500, so
    the measurement is affordable exactly here -- after the quotas, on the shots that will actually
    be used. Three outcomes per shot:

    * measured and >= `MIN_FLATTOP_S` -- kept, `flattop_source="features_ip"`, and the estimate is
      replaced by the number;
    * measured and short -- DROPPED. The proxy said this shot had a flat-top and the Ip trace says
      it did not, which is the case rule (d) exists for. A shot from the SAME theme takes its
      place, through the same `_Allocator` rebuilt from the survivors, so every cap still holds
      globally and a theme floor that was met before the verification is still met after it. Same
      theme first and then any theme, because a hole in the list is worse than a hole in one
      quota; `summary.replacements` records which happened.
    * not measurable -- kept and marked `pending`. The number stays the proxy estimate (it is the
      best estimate there is) but the SOURCE says nobody has measured it, which is the difference
      the CLI's `--allow-pending` gate is about.

    A replacement is itself measured before it is admitted, so the verification cannot fill a hole
    with a shot that would have failed the same check.
    """
    kept, dropped, _pending = _measured(selected, measure)
    return _refill(kept, dropped, candidates, quotas, seed, measure)


def reverify_flattop(
    existing: Sequence[Candidate],
    candidates: Sequence[Candidate],
    quotas: Quotas,
    seed: int,
    measure: Callable[[int], float | None],
) -> tuple[list[Candidate], list[dict], list[int]]:
    """Re-run rule (d) on an ALREADY COMMITTED list. Returns `(list, replacements, pending)`.

    This is the second invocation (`corpus select --finalize --from-list <yaml>`), and it exists
    because the first one changes the world it ran in. `--verify-flattop` writes a pending file;
    labelmaker's features stage runs over it; and the store that comes back is BIGGER than the one
    the selection saw -- which moves `eligible` (a measured flat-top replaces the proxy for those
    shots) and moves the `preferred` tie-break in `diversify`. Re-running the whole selection on
    the new store therefore returns a DIFFERENT list -- measured on the real corpus: 78 of 500
    shots changed after 17 new feature files, and 350 shots newly pending after the full store --
    so the features-then-finalize loop would never converge, and the list committed to git would
    never be the list that was verified.

    So the committed list is the eligibility snapshot and is not re-selected. Exactly the listed
    shots are re-measured; each one is kept (measured >= `MIN_FLATTOP_S`), dropped, or reported
    `pending` because nothing can measure it yet. The kept shots go back through the allocator
    with `force=True` -- their caps were satisfied once, when the list was made -- and only the
    replacements for the dropped ones are allocated against those counts, same theme first, the
    replacement itself measured. `pending` is returned rather than raised on: the caller decides
    whether an unfinished list may be written, and it needs the shot numbers to rewrite the
    features work order for exactly the shots still waiting.
    """
    kept, dropped, pending = _measured(existing, measure)
    final, replacements = _refill(
        kept, dropped, candidates, quotas, seed, measure, force_kept=True
    )
    return final, replacements, pending


def _measured(
    selected: Sequence[Candidate], measure: Callable[[int], float | None]
) -> tuple[list[Candidate], list[tuple[Candidate, float]], list[int]]:
    """`selected` split three ways by the measurement: kept, `(dropped, its flat-top)`, pending.

    A pending shot is KEPT, carrying its proxy estimate and a `flattop_source` that says nobody
    has measured it; its shot number is also returned, which is what the work order is written
    from.
    """
    kept: list[Candidate] = []
    dropped: list[tuple[Candidate, float]] = []
    pending: list[int] = []
    for c in sorted(selected, key=lambda c: c.shot):
        got = measure(c.shot)
        if got is None:
            kept.append(replace(c, flattop_source=FLATTOP_PENDING))
            pending.append(int(c.shot))
        elif got >= MIN_FLATTOP_S:
            kept.append(replace(c, flattop_s=float(got), flattop_source=FLATTOP_MEASURED))
        else:
            dropped.append((c, float(got)))
    return kept, dropped, pending


def _refill(
    kept: Sequence[Candidate],
    dropped: Sequence[tuple[Candidate, float]],
    candidates: Sequence[Candidate],
    quotas: Quotas,
    seed: int,
    measure: Callable[[int], float | None],
    *,
    force_kept: bool = False,
) -> tuple[list[Candidate], list[dict]]:
    """Fill each hole `dropped` left, from `candidates`, same theme first. Shared by both passes."""
    alloc = _Allocator(quotas)
    for c in kept:
        alloc.add(c, c.reason, force=force_kept)
    used = {c.shot for c in kept} | {c.shot for c, _ in dropped}
    spare = [c for c in candidates if c.shot not in used]
    verdicts: dict[int, float | None] = {}
    replacements: list[dict] = []
    for c, got in dropped:
        same = [x for x in spare if x.theme == c.theme]
        other = [x for x in spare if x.theme != c.theme]
        chosen = None
        for cand in list(_by_year(same, seed)) + list(_by_year(other, seed)):
            if cand.shot not in verdicts:  # measured once, however many holes it is offered for
                verdicts[cand.shot] = measure(cand.shot)
            got_c = verdicts[cand.shot]
            if got_c is not None and got_c < MIN_FLATTOP_S:
                continue
            fixed = (
                replace(cand, flattop_source=FLATTOP_PENDING)
                if got_c is None
                else replace(cand, flattop_s=float(got_c), flattop_source=FLATTOP_MEASURED)
            )
            if alloc.add(fixed, f"replaces:{c.shot}"):
                chosen = alloc.taken[cand.shot]
                spare = [x for x in spare if x.shot != cand.shot]
                break
        replacements.append(
            {
                "dropped": int(c.shot),
                "dropped_flattop_s": round(got, 3),
                "theme": c.theme,
                "replacement": None if chosen is None else int(chosen.shot),
                "replacement_theme": None if chosen is None else chosen.theme,
                "replacement_flattop_source": None if chosen is None else chosen.flattop_source,
            }
        )
    return sorted(alloc.taken.values(), key=lambda c: c.shot), replacements


# ------------------------------------------------------------------------------ the report


def summarize(
    *,
    n_candidates: int,
    reasons: Counter,
    selected: Sequence[Candidate],
    quotas: Quotas,
    candidates: Sequence[Candidate] | None = None,
    replacements: Sequence[Mapping] | None = None,
    finalized: bool = False,
    n_verified: int | None = None,
    store: Mapping | None = None,
) -> dict:
    """The summary block: how big the pool was, what rejected the rest, and what came out.

    `reasons` counts rule failures over the whole pool and a shot may appear in several of its
    entries -- `eligible` reports every clause that fired, so these are counts of REJECTIONS BY
    RULE and not a partition of the rejected shots.

    `finalized`, `n_verified`, `n_dropped` and `store` are the provenance of the SECOND pass. A
    list written by `--verify-flattop --allow-pending` and one written by `--finalize` are the
    same file under the same name with different standing, and the difference has to be readable
    off the document rather than off whoever remembers which command was run. `store` is
    `store_fingerprint()`: how many feature files existed and how new the newest was, which is
    what says WHICH store verified these rows -- the store grows between the two invocations, by
    design, and two runs a day apart are not the same verification.
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
        # The polarity histogram. Rule (a) does not test it, so the list's counter-Ip content is a
        # fact about the list a reader has to be able to see without opening 500 traces.
        "ip_sign": {
            ("+1" if k == 1 else "-1" if k == -1 else "unknown"): int(v)
            for k, v in sorted(
                Counter(c.ip_sign for c in selected).items(), key=lambda kv: (kv[0] is None, kv[0])
            )
        },
        "replacements": [dict(r) for r in (replacements or ())],
        "runs": len({c.run_id for c in selected}),
        "mpids": len({c.mpid for c in selected if c.mpid}),
        "finalized": bool(finalized),
        # Counted from the rows when the caller does not say, so the number can never disagree
        # with the `flattop_source` histogram above it.
        "n_verified": int(
            n_verified
            if n_verified is not None
            else sum(c.flattop_source == FLATTOP_MEASURED for c in selected)
        ),
        "n_dropped": len(replacements or ()),
        "n_featured": None if store is None else int(store.get("n_featured", 0)),
        "n_frame_codes": None if store is None else int(store.get("n_frame_codes", 0)),
        # The fingerprint, flat so it can be diffed between two runs. `n_files` repeats
        # `n_featured` deliberately: the pair (count, newest mtime) is the identity of the store
        # and belongs together, and `n_featured` alone is what a reader greps for.
        "feature_store": {
            "n_files": None if store is None else int(store.get("n_featured", 0)),
            "max_mtime": None if store is None else store.get("max_mtime"),
        },
    }


def document(
    selected: Sequence[Candidate],
    summary: Mapping,
    *,
    name: str,
    seed: int,
    n: int,
    rule: str = RULE,
    hand_review: Mapping | None = None,
) -> dict:
    """The shot-list YAML document, in `config.load_shot_list`'s format.

    `hand_review` is written empty for a fresh list: it is the human gate on a generated list, and
    the generator's job is to leave the block there, not to fill it in. A re-verification
    (`--from-list`) passes the block of the list it verified, because resetting a reviewer's
    decisions to empty is a silent loss of exactly the judgement the block exists to record.
    """
    return {
        "name": name,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "rule": rule,
        "seed": int(seed),
        "n": int(n),
        "summary": dict(summary),
        "hand_review": (
            {"drop": [], "add": []}
            if hand_review is None
            else {"drop": list(hand_review.get("drop") or []), "add": list(hand_review.get("add") or [])}
        ),
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
                # +1 / -1 / None. Rule (a) reads |Ip|, so a reversed-Ip shot is admitted on its
                # merits -- and a reader of the list can see which shots those are from here.
                "ip_sign": None if c.ip_sign is None else int(c.ip_sign),
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
    if summary.get("ip_sign"):
        lines += ["", "Ip polarity:"]
        lines += [f"  {k:<22}{v:>8,}" for k, v in summary["ip_sign"].items()]
    store = summary.get("feature_store") or {}
    n_files = store.get("n_files")
    fingerprint = (
        f"  {'feature store':<22}{n_files if n_files is not None else '?':>8} files, "
        f"newest {store.get('max_mtime') or '?'}"
    )
    lines += ["", "verification:"]
    lines += [
        f"  {'finalized':<22}{str(bool(summary.get('finalized'))).lower():>8}",
        f"  {'verified':<22}{summary.get('n_verified', 0):>8,}",
        f"  {'dropped':<22}{summary.get('n_dropped', 0):>8,}",
        fingerprint,
    ]
    repl = summary.get("replacements") or []
    lines += ["", f"flat-top verification dropped {len(repl)} shot(s):"]
    lines += [
        f"  {r['dropped']} ({r['dropped_flattop_s']} s, {r['theme'] or 'no theme'})"
        f" -> {r['replacement'] or 'no replacement available'}"
        for r in repl
    ] or ["  (none)"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------ loading the pool


def read_bundles(text_dir: Path, shots: Iterable[int]) -> dict[int, str]:
    """`{shot: bundle text}` for the shots that have a `shot_<N>.txt`."""
    text_dir = Path(text_dir)
    out = {}
    for shot in shots:
        p = text_dir / f"shot_{int(shot)}.txt"
        if p.exists():
            # Both arguments spelled out, and they do different jobs. `encoding=` pins the
            # decoding: without it the file reads under the process locale, so the same bundle
            # decodes differently on a machine with LANG unset than on one without.
            # `errors="replace"` then handles what is genuinely there -- the corpus is scraped
            # HTML and some bundles carry bytes that are not valid UTF-8 -- and it is safe only
            # BECAUSE the encoding is pinned; on the locale path it would have been papering over
            # a decoding bug instead. A bundle is read for a shot table and a title, and one
            # replacement character costs neither.
            out[int(shot)] = p.read_text(encoding="utf-8", errors="replace")
    return out


def candidates_from_rows(
    rows: Iterable[Mapping], *, preferred: Iterable[int] = ()
) -> list[Candidate]:
    """The `shots:` rows of a committed list, back as `Candidate`s, in shot order.

    The row is the whole record: run day, mini-proposal, year, theme, the group flags, the
    polarity, the flat-top and what measured it, and the quota that admitted the shot. Rebuilding
    them from the document rather than re-deriving them from the corpus is the point of
    `--from-list` -- the committed list is the eligibility snapshot, and re-deriving would be
    re-selecting under another name.

    `preferred` is the ONE fact the document does not carry, because it is not a property of the
    shot: it is "this shot already had labelmaker features when the list was made", and by the
    time a list is finalized the features stage has run over the list itself, so it is true of
    nearly every row. It is passed in from today's store and used for one thing only -- counting
    the preferred cap against any REPLACEMENT drawn now (the kept rows are forced in, caps and
    all), which is the same reading the cap has everywhere else: a ceiling on how many
    already-featured shots a fill may take.
    """
    want = {int(s) for s in preferred}
    out = []
    for row in rows:
        shot = int(row["shot"])
        out.append(
            Candidate(
                shot=shot,
                run_id=row.get("run_id"),
                mpid=row.get("mpid"),
                year=None if row.get("year") is None else int(row["year"]),
                theme=row.get("theme"),
                has_co2=bool(row.get("has_co2")),
                has_bes=bool(row.get("has_bes")),
                has_tangtv=bool(row.get("has_tangtv")),
                flattop_s=float(row.get("flattop_s") or 0.0),
                flattop_source=str(row.get("flattop_source") or FLATTOP_PROXY),
                preferred=shot in want,
                ip_sign=None if row.get("ip_sign") is None else int(row["ip_sign"]),
                reason=str(row.get("reason") or ""),
            )
        )
    return sorted(out, key=lambda c: c.shot)


def store_fingerprint(
    features_dir: Path | None, frame_codes_dir: Path | None = None
) -> dict:
    """Which feature store a run saw: `{n_featured, n_frame_codes, max_mtime}`.

    Two invocations of `corpus select` write the same file under the same name and mean different
    things, and what changed between them is this store -- the features stage ran over the pending
    list. The count and the newest mtime are cheap, need no file opened, and are enough to tell a
    verification done before that job from one done after it.
    """
    n_features, newest = 0, None
    if features_dir and Path(features_dir).is_dir():
        for p in Path(features_dir).glob("*_features.h5"):
            n_features += 1
            mtime = p.stat().st_mtime
            newest = mtime if newest is None else max(newest, mtime)
    n_codes = 0
    if frame_codes_dir and Path(frame_codes_dir).is_dir():
        n_codes = sum(1 for p in Path(frame_codes_dir).glob("*.pt") if p.stem.isdigit())
    return {
        "n_featured": n_features,
        "n_frame_codes": n_codes,
        "max_mtime": None
        if newest is None
        else dt.datetime.fromtimestamp(newest, dt.UTC).isoformat(timespec="seconds"),
    }


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
