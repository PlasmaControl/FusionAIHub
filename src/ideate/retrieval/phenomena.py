"""Find the shots where a phenomenon actually happened -- and say which kind of evidence says so.

Four things can claim that shot 185946 had an edge harmonic oscillation, and they are not
interchangeable:

* a **detector** saw it -- a `tokeye_track` row in the 2-20 kHz band with a harmonic comb. This
  is the only class that is an OBSERVATION, and it is the only one that can rule the phenomenon
  *out*, and then only inside the window the diagnostic actually covered;
* a **label model** scored it -- `labels_wide.max_valid` for a detection head. A model's opinion
  about the present, which is not the same as a measurement of it;
* a **forecast** put a probability on it starting soon -- an `evidence_kind="forecast"` row, a
  risk curve crossed at a threshold. A statement about the future, never about the past
  (plan §2, §7, Appendix C item 7). It lives in its own field, `PhenomenonHit.forecasts`, and it
  never merges into `intervals`;
* an **operator wrote it down** -- a `text_claims` row. Never a label by itself (plan §2): a
  hit resting on nothing but text is capped at labelmaker's `TEXT_ONLY_CEILING` and carries the
  caveat "TEXT ONLY".

So the ranking is a class order first and arithmetic second. `locate` sorts by evidence class
before it sorts by score, and a shot with one observed interval outranks a shot with a hundred
forecast rows however the weights are set. The weights, in `configs/ideate/retrieval.yaml`, are
not fitted -- there is no labelled set of "shots with an EHO" to fit them against.

**Absence is not evidence.** Every route out of this module distinguishes "the diagnostic looked
and saw nothing" from "nobody looked": `coverage` is the union of the coverage windows of the
detectors this phenomenon reads, and a hit with `coverage is None` says so in its caveats. That
is also what `avoid` turns on -- `--avoid phenomenon:elm` drops the shots whose ELM detector fired
and KEEPS, with a caveat, the shots where no ELM detector ran at all. Dropping those would be
reading a gap in the data as a physics result.

**The vocabulary is labelmaker's.** The phenomenon ids and their aliases come from
`src/labelmaker/events/lexicons.yaml`, which is the single source of both (plan §5.6), and
`resolve` matches text by calling labelmaker's own `hits()` rather than by carrying a second
tokenizer that would drift from the first correction onwards. `configs/ideate/phenomena.yaml`
adds only the scoring machinery around those ids, and an `aliases:` key in it is an error.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from labelmaker.events import lexicon as lexicon_mod

from .. import config
from ..schema import EventRef, Interval, PhenomenonHit
from . import describe as describe_mod

#: The registry file, under `configs/ideate/`.
CONFIG_NAME = "phenomena.yaml"

#: The `retrieval.yaml` block `locate` reads its weights from.
RETRIEVAL_BLOCK = "phenomenon"

#: Fallback weights, used only for a key the config does not set. The config is the source; these
#: exist so that adding a term to the score does not need a config migration to be readable.
DEFAULT_WEIGHTS: dict[str, float] = {"label": 1.0, "event": 1.0, "text": 0.5, "database": 1.0}
DEFAULT_SATURATION_N = 3.0

#: Corpus groups a `requires_group:` entry may name -- the `has_*` columns of shots.parquet. A
#: typo here is a requirement no shot can ever satisfy, so it is checked rather than trusted.
CORPUS_GROUPS: tuple[str, ...] = ("bes", "co2", "ece", "filterscopes", "mhr", "mirnov")

#: The four states `PhenomenonHit.coverage_state` can be in, worst-informed first. They are not
#: degrees of one thing: only `observed` is an answer to "was it there?", and the other three are
#: three different ways of saying the database cannot tell you. The names and the order are
#: `ideate.mcp.tools.EVENT_STATES`', so a `get_events` reply and a phenomenon hit describe the
#: same shot with the same word.
#:
#:   unindexed    nothing in the pipeline looks for this phenomenon at all (`rwm`, `detachment`)
#:   unprocessed  its detectors exist, and none of them has run on this shot
#:   uncovered    they ran, but not over the window the question is about
#:   observed     they covered (part of) the window; `coverage` is that intersection
COVERAGE_STATES: tuple[str, ...] = ("unindexed", "unprocessed", "uncovered", "observed")

#: Which `evidence_kind` values are OBSERVATIONS. An allow-list, not "anything that is not a
#: forecast": labelmaker's `EVIDENCE_KINDS` also holds `text`, `model`, `human` and `database`,
#: and one `source="tokeye_track", evidence_kind="model"` row would otherwise be reported as a
#: diagnostic sighting by the same code path that reports a detector's. The registry's source
#: pins keep those rows out today; this keeps them out when a source starts writing both.
OBSERVED_KINDS = frozenset({"detector", "heuristic"})
FORECAST_KIND = "forecast"

#: How good a detection label has to be before it is EVIDENCE rather than a number. Below it the
#: probability is still reported in `label_evidence` -- it is a measurement of the model, and
#: suppressing it would be its own kind of lie -- but it does not put the shot in the LABELLED
#: tier and it does not make the shot a hit. Measured on the 500-shot database: a quarter of the
#: `tm_prob` rows that "have label evidence" are at or below 0.010, i.e. the model saying no.
#: 0.5 is a declared default and not a fitted operating point; a `thr:` published on a registry
#: entry (from a model card) overrides it for that label, which is what `thr` is for.
DEFAULT_LABEL_FLOOR = 0.5

#: The tier order, in one sentence, so `retrieval.yaml` and `docs/IDEATE.md` cannot state it
#: differently from the code that sorts by it. A test asserts both files contain this string.
RANKING_SENTENCE = (
    "an observed hit outranks a label-only hit, which outranks a forecast-only hit, which "
    "outranks a curated-list hit, which outranks a text-only hit"
)

#: What `PhenomenonHit.actuators_at_onset` reports. **Segment statistics, not onset values**:
#: these are the segment's own summary columns, so what the field says is "this is what the
#: machine was doing during the segment the phenomenon was found in", not "this is what it was
#: doing at t0". A per-onset lookup needs the raw waveform, which the DB does not carry; when it
#: does, this constant and the field's meaning change together. `irmp` is a peak and the other
#: three are means because peak is the only statistic the registry stores for the I-coils -- the
#: keys are the column names, so the field cannot misdescribe itself.
ACTUATOR_COLUMNS: tuple[str, ...] = (
    "pnbi_total_mean", "pech_total_mean", "gas_total_mean", "irmp_total_peak",
)

# ------------------------------------------------------------------------- the caveat vocabulary
#
# One string per thing a reader has to know before believing a hit. They are constants because
# callers key on them -- the MCP layer and the CLI both surface them verbatim -- and because a
# caveat that is spelled two ways is a caveat nobody can filter on.

#: `unindexed`: nothing in the pipeline looks for this phenomenon, so nothing ever could have.
NO_COVERAGE = "no diagnostic coverage recorded; absence is not evidence"
#: `unprocessed`: the detectors exist and none of them has run on this shot. `.format(title=...)`
COVERAGE_UNPROCESSED = "no detector for {title} has run on this shot; absence is not evidence"
#: `uncovered`: they ran, elsewhere in the record. `.format(title=..., segment=...)`
COVERAGE_OUTSIDE_WINDOW = (
    "the {title} detectors ran on this shot but not over the {segment} window; absence there is "
    "unmeasured, not established"
)
#: `observed`, but only over part of the window. `.format(title=..., segment=..., covered=...)`
COVERAGE_PARTIAL = (
    "the {title} detectors covered only {covered} of the {segment} window; absence outside that "
    "is unmeasured"
)
#: The covered stretches are disjoint. `.format(title=..., segment=..., n=...)`
COVERAGE_GAPS = (
    "the {title} coverage of the {segment} window has {n} gap(s): `coverage` is the hull of "
    "`coverage_windows`, which is the union"
)
#: The detector that fired is not a detector for this CLASS. Attached by an `events:` rule that
#: declares `caveat:`, to every hit whose observed intervals came through that rule.
TRANSIENT_NOT_CLASSIFIED = (
    "observed via tokeye_transient, a class-agnostic transient detector: a sawtooth crash or a "
    "disruption precursor is transient too, so this is an ELM-LIKE TRANSIENT and not a "
    "classified ELM"
)
#: Rows that matched a rule but are of a kind that is neither observation nor forecast.
#: `.format(n=..., kinds=...)`
UNCLASSIFIED_KIND = (
    "{n} row(s) of evidence_kind {kinds} match this phenomenon's rules and are counted as "
    "neither observation nor forecast"
)
#: The label model ran and scored below its floor. `.format(key=..., p=..., floor=...)`
LABEL_BELOW_FLOOR = (
    "{key} scored {p:.3f}, below the {floor:.2f} evidence floor: reported, not counted as evidence"
)
#: The quote beside the hit is the shot's best entry, not a sentence about this phenomenon.
#: `.format(title=...)`
QUOTE_UNRELATED = (
    "the quote is this shot's most informative logbook entry and does not mention {title}"
)
#: Printed once, under `resolved:`, when the database holds no observation at all.
#: `.format(n=...)`
ALL_FORECASTS = (
    "this database holds {n} event row(s), all of them forecasts: no observed-class hit is "
    "possible yet"
)
#: Nothing but operator text supports this hit. Its score is capped at `TEXT_ONLY_CEILING`.
TEXT_ONLY = "TEXT ONLY"
#: Ranked on forecasts: the best evidence class present is a model's claim about the future.
FORECAST_ONLY = (
    "ranked on forecasts: the strongest evidence here is a model's estimate of what was about "
    "to happen, not an observation"
)
#: Ranked on labels: the best evidence class present is a model's claim about the present. The
#: number is in the string because "the model ran" is not the same claim as "the model said 0.9",
#: and a caveat that hides the score lets a 0.51 read like a sighting. `.format(p=...)`
LABEL_ONLY = (
    "ranked on model labels: the strongest evidence here is a model's score ({p:.3f}), not a "
    "diagnostic"
)
#: Ranked on a curated human list, and nothing else says a word.
DATABASE_ONLY = (
    "ranked on a curated human list: no detector, model or logbook claims this shot"
)
NO_OBSERVED = "no observed evidence: no detector claims this phenomenon on this shot"
NO_LABEL_MODEL = "no label evidence: no model in the registry emits a label for this phenomenon"
LABEL_UNAVAILABLE = "label not run on this shot, or no valid samples: unavailable, not 0"
NO_TEXT = "no operator text names this phenomenon on this shot"
RUN_SCOPE_TEXT = "text evidence is run scope: a session-wide sentence, not this shot's logbook"

#: `.format(title=...)` -- the operators wrote that the phenomenon was ABSENT.
NEGATIVE_CLAIM = "operator log says NOT {title}"
#: `.format(segment=...)` -- the shot has no such segment, so the whole record was searched.
NO_SEGMENT = "no {segment} segment on this shot; the whole record was searched"
#: `.format(token=..., title=...)` -- why an `avoid` token did not remove this shot. One per
#: coverage state, because "nobody has ever looked for this", "the detector has not run on this
#: shot" and "the detector ran, but not over this window" are three different answers, and none
#: of them is the fourth one -- a detector that covered the window and found nothing, which is
#: the only real negative and the only case that carries no caveat at all.
AVOID_NO_COVERAGE = (
    "kept despite --avoid {token}: nothing looked for {title} on this shot, so its absence "
    "is unmeasured, not established"
)
AVOID_UNPROCESSED = (
    "kept despite --avoid {token}: no detector for {title} has run on this shot, so its absence "
    "is unmeasured, not established"
)
AVOID_UNCOVERED = (
    "kept despite --avoid {token}: the {title} detectors ran on this shot but not over the "
    "window searched, so its absence there is unmeasured, not established"
)
AVOID_PARTIAL = (
    "kept despite --avoid {token}: the {title} detectors covered only part of the window "
    "searched, so outside that its absence is unmeasured"
)
AVOID_COVERAGE_CAVEATS: dict[str, str] = {
    "unindexed": AVOID_NO_COVERAGE,
    "unprocessed": AVOID_UNPROCESSED,
    "uncovered": AVOID_UNCOVERED,
    "partial": AVOID_PARTIAL,
}
#: `locate(..., notes=[])` -- what an `avoid` filter did, which no surviving hit can carry
#: because the shots it speaks about are the ones that are gone. `.format(token=, n=, title=)`
AVOID_DROPPED = "--avoid {token}: dropped {n} shot(s) with observed {title} evidence"
#: `.format(n=..., caveat=...)` -- and what that observed evidence actually was.
AVOID_DROPPED_CAVEATED = "{n} of those drops rest on evidence that carries: {caveat}"
#: `.format(n=..., limit=...)` -- events that could not be shown to clear `--min-confidence`.
DROPPED_UNSCORED = (
    "{n} event(s) not shown: the source recorded no confidence, so they cannot be shown to "
    "reach --min-confidence {limit}"
)


#: The caveats an `events:` rule may attach, by the name the registry calls them. A rule whose
#: detector does not detect the CLASS it is registered under says so on every hit it produces,
#: and the registry names the string rather than restating it, so the config cannot invent a
#: caveat the callers who key on these constants have never seen.
EVENT_CAVEATS: dict[str, str] = {"transient_not_classified": TRANSIENT_NOT_CLASSIFIED}


class PhenomenaError(ValueError):
    """The phenomenon registry says something `locate` cannot use."""


# ------------------------------------------------------------------------------- registry types


@dataclass(frozen=True)
class LabelRef:
    """One `labels_wide` series read as evidence for a phenomenon.

    `thr` is this label's operating point IF a model card has published one, and it is the floor
    the probability has to clear to be EVIDENCE: below it the number is still reported in
    `label_evidence` and still caveated, but it does not lift the shot into the LABELLED tier.
    None -- which is every entry today, because `labels_wide.thr` is NaN for both detection
    labels -- falls back to `DEFAULT_LABEL_FLOOR`, so a label is never thresholded against a
    number nobody declared, and `thr:` is never a key that silently does nothing.
    """

    slug: str
    label: str
    thr: float | None = None
    weight: float = 1.0

    @property
    def key(self) -> str:
        return f"{self.slug}/{self.label}"


@dataclass(frozen=True)
class EventRule:
    """Which `events.parquet` rows are OBSERVED evidence for a phenomenon.

    `phenomenon` is the string the SOURCE writes, which is deliberately not ideate's id:
    `tokeye_track` writes `coherent_mode` for every track it finds, and what makes one of them an
    EHO rather than a tearing mode is the band and the harmonic count, not the detector's label.


    `weight` is how much one matching row is worth in the event term, and it is not always 1:
    `tokeye_transient` writes `phenomenon="elm"` for any burst, and labelmaker's own module says
    a sawtooth crash and a disruption precursor are transient too, so a row from it is weaker
    evidence of an ELM than an `ece_sawtooth` crash is of a sawtooth. `caveat` names the string
    every hit built on this rule carries for the same reason; the two go together.

    `max_bandwidth_khz` bounds the track's width. `band_khz` alone matches on the centroid, and
    a track spanning 0.5-248 kHz has a centroid somewhere: on the three real labelmaker shots
    5-12 % of the rows the tearing rule matched are that wide. The rule of thumb the registry
    follows is that a track has to FIT INSIDE the band it is claimed to be in.
    """

    source: str
    phenomenon: str | None = None
    band_khz: tuple[float | None, float | None] | None = None
    max_bandwidth_khz: float | None = None
    min_harmonics: int | None = None
    chirp_sign: int | None = None
    pickup: bool | None = None
    weight: float = 1.0
    caveat: str | None = None

    def matches(self, row: Mapping, attrs: Mapping) -> bool:
        """Does this row satisfy the rule? Frequency is read from `attrs.f_centroid_khz`."""
        if str(row["source"]) != self.source:
            return False
        if self.phenomenon is not None and str(row["phenomenon"]) != self.phenomenon:
            return False
        if self.band_khz is not None:
            f = _centroid_khz(row, attrs)
            if f is None:
                return False
            lo, hi = self.band_khz
            if (lo is not None and f < lo) or (hi is not None and f > hi):
                return False
        if self.max_bandwidth_khz is not None:
            bw = _bandwidth_khz(row, attrs)
            # A row that records NO width is not rejected for not recording one: "we cannot show
            # this is a smear" is not "this is a smear". Every real `tokeye_track` row carries
            # `attrs.bandwidth_khz`, so the predicate bites where it was measured to be needed.
            if bw is not None and bw > self.max_bandwidth_khz:
                return False
        if self.min_harmonics is not None:
            n = attrs.get("n_harmonics")
            if not isinstance(n, int | float) or int(n) < self.min_harmonics:
                return False
        if self.chirp_sign is not None:
            c = attrs.get("chirp_khz_per_ms")
            if not isinstance(c, int | float) or not math.isfinite(float(c)):
                return False
            if (float(c) < 0) != (self.chirp_sign < 0) or float(c) == 0.0:
                return False
        return self.pickup is None or bool(attrs.get("pickup")) is self.pickup


@dataclass(frozen=True)
class Phenomenon:
    """One registry entry: labelmaker's names, and ideate's evidence rules around them."""

    id: str
    title: str
    aliases: tuple[str, ...]
    exclude: tuple[str, ...]  # labelmaker calls these `negatives`; same idea, its name
    labels: tuple[LabelRef, ...] = ()
    events: tuple[EventRule, ...] = ()
    forecasts: tuple[str, ...] = ()
    #: Sources whose rows establish that someone LOOKED for this phenomenon, beyond its own
    #: detectors. Declared and not inferred: a detector that finds nothing writes no rows, so on
    #: a quiet shot the only record that the mhr data was read for ELMs at all is `elm_clock`'s
    #: `elm_free` interval -- a different source from the one that would have reported an ELM.
    #: Inferring this from the `diag` string instead would have let any row off any diagnostic
    #: donate coverage to a phenomenon nothing detects (`rwm`, `detachment`), which would turn
    #: "nobody has ever looked for this" into "we looked and it was not there".
    coverage_sources: tuple[str, ...] = ()
    band_khz: tuple[float | None, float | None] | None = None
    diags: tuple[str, ...] = ()
    requires_group: tuple[str, ...] = ()
    prior: float = 0.0
    database: Mapping[str, object] | None = None

    @property
    def sources(self) -> tuple[str, ...]:
        """The detectors that can claim this phenomenon."""
        return tuple(dict.fromkeys(r.source for r in self.events))

    @property
    def covering_sources(self) -> tuple[str, ...]:
        """Every source whose rows say someone looked. Empty when nothing detects this at all,
        which is exactly the case whose coverage must stay unknown."""
        return tuple(dict.fromkeys((*self.sources, *self.coverage_sources)))


@dataclass(frozen=True)
class Evidence:
    """Everything one shot's tables say about one phenomenon, by class, before it is scored."""

    shot: int
    phenomenon: str
    window: tuple[float, float] | None
    intervals: tuple[Interval, ...] = ()
    refs: tuple[EventRef, ...] = ()
    forecasts: tuple[Interval, ...] = ()
    label_evidence: dict[str, float | None] = field(default_factory=dict)
    max_label_p: float | None = None
    text_hits: int = 0
    text_snippets: tuple[str, ...] = ()
    n_negative_claims: int = 0
    #: The intersection of the coverage union with `window`, as a hull. None unless
    #: `coverage_state == "observed"`: a window somebody looked at is not coverage of the window
    #: somebody ASKED about, and reporting it as though it were is how a ramp-up-only detector
    #: pass becomes a clean flat-top negative.
    coverage: tuple[float, float] | None = None
    #: The same intersection as the real union, gaps and all. `coverage` is its hull.
    coverage_windows: tuple[tuple[float, float], ...] = ()
    coverage_state: str = "unindexed"
    coverage_partial: bool = False
    #: The event term's input: the sum of the WEIGHTS of the rules the intervals matched, which
    #: is `len(intervals)` only when every rule weighs 1.
    event_weight: float = 0.0
    in_database: bool = False
    caveats: tuple[str, ...] = ()

    @property
    def total_duration_s(self) -> float:
        return float(sum(iv.t1_s - iv.t0_s for iv in self.intervals))


# ------------------------------------------------------------------------------- the registry


@functools.lru_cache(maxsize=2)
def _lexicon(path: str | None = None) -> lexicon_mod.Lexicon:
    """labelmaker's parsed lexicon: the ids, the aliases and the negatives, from its own reader."""
    return lexicon_mod.load_lexicon(path)


# path -> (size, mtime_ns, registry). Same contract as `config.load_yaml`: an edited file is
# picked up on the next call with no restart.
_CACHE: dict[str, tuple[int, int, dict[str, Phenomenon]]] = {}


def registry(path: Path | str | None = None) -> dict[str, Phenomenon]:
    """The validated registry, by phenomenon id, with labelmaker's aliases merged in.

    Validated rather than trusted, because every mistake this file can make is silent: an id that
    is not one of the round-1 twelve joins to no label, no event and no claim, and a re-stated
    alias list is a second vocabulary that drifts from labelmaker's on the first correction.
    """
    p = Path(path) if path is not None else config.CONFIG_DIR / CONFIG_NAME
    try:
        st = p.stat()
    except OSError as exc:
        raise PhenomenaError(f"no phenomenon registry at {p}: {exc}") from exc
    hit = _CACHE.get(str(p))
    if hit is None or hit[0] != st.st_size or hit[1] != st.st_mtime_ns:
        try:
            doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise PhenomenaError(f"{p}: not readable as YAML: {exc}") from exc
        hit = _CACHE[str(p)] = (st.st_size, st.st_mtime_ns, _build(doc, p))
    return dict(hit[2])


def titles() -> dict[str, str]:
    """Phenomenon id -> display title, for the "try one of these" line."""
    return {pid: ph.title for pid, ph in registry().items()}


def _build(doc: Mapping, source: Path) -> dict[str, Phenomenon]:
    if not isinstance(doc, Mapping):
        raise PhenomenaError(f"{source}: the registry is a mapping, not {type(doc).__name__}")
    entries = doc.get("phenomena")
    if not isinstance(entries, Mapping) or not entries:
        raise PhenomenaError(f"{source}: `phenomena` must be a non-empty mapping")
    lex = _lexicon()
    want, got = set(lexicon_mod.PHENOMENON_IDS), set(entries)
    if want != got:
        missing, extra = sorted(want - got), sorted(got - want)
        raise PhenomenaError(
            f"{source}: the registry must name exactly labelmaker's round-1 phenomena. "
            f"missing {missing}, unknown {extra}"
        )
    out: dict[str, Phenomenon] = {}
    for pid, body in entries.items():
        if not isinstance(body, Mapping):
            raise PhenomenaError(f"{source}: {pid} must be a mapping")
        for banned in ("aliases", "exclude", "negatives"):
            if banned in body:
                raise PhenomenaError(
                    f"{source}: {pid} carries `{banned}:`. The alias lists live in labelmaker's "
                    f"{lex_path()} and are read from there -- one file, two readers (plan §5.6). "
                    "Add the phrase there instead."
                )
        title = body.get("title")
        if not isinstance(title, str) or not title.strip():
            raise PhenomenaError(f"{source}: {pid} needs a non-empty `title`")
        prior = body.get("prior", 0.0)
        if not isinstance(prior, int | float) or isinstance(prior, bool) or not 0.0 <= prior <= 1.0:
            raise PhenomenaError(f"{source}: {pid} `prior` must be a number in [0, 1]")
        groups = tuple(str(g) for g in body.get("requires_group") or ())
        unknown = [g for g in groups if g not in CORPUS_GROUPS]
        if unknown:
            raise PhenomenaError(
                f"{source}: {pid} requires_group {unknown} -- not corpus groups {CORPUS_GROUPS}"
            )
        entry = lex[pid]
        out[pid] = Phenomenon(
            id=pid,
            title=title.strip(),
            aliases=entry.aliases,
            exclude=entry.negatives,
            labels=_labels(source, pid, body.get("labels")),
            events=_events(source, pid, body.get("events")),
            forecasts=tuple(str(s) for s in body.get("forecasts") or ()),
            coverage_sources=tuple(
                str(x) for x in _seq(source, pid, body.get("coverage_sources"), "coverage_sources")
            ),
            band_khz=_band(source, pid, body.get("band_khz")),
            diags=tuple(str(d) for d in body.get("diags") or ()),
            requires_group=groups,
            prior=float(prior),
            database=dict(body["database"]) if body.get("database") else None,
        )
    return out


def lex_path() -> Path:
    """Where the alias lists live. labelmaker's file, never a copy of it."""
    return Path(lexicon_mod.DEFAULT_LEXICON)


def _seq(source: Path, pid: str, value, field_name: str) -> list:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise PhenomenaError(f"{source}: {pid} `{field_name}` must be a list")
    return list(value)


def _forecast_keys() -> frozenset[str]:
    """`<slug>/<label>` of every series `labels.yaml` declares to be a RISK, i.e. a forecast."""
    try:
        from ..labels.join import forecast_rules

        return frozenset(r.key for r in forecast_rules())
    except (OSError, ValueError, KeyError):  # a labels.yaml that will not parse is its own error
        return frozenset()


def _labels(source: Path, pid: str, value) -> tuple[LabelRef, ...]:
    out = []
    forecasts = _forecast_keys()
    for item in _seq(source, pid, value, "labels"):
        if not isinstance(item, Mapping) or not item.get("slug") or not item.get("label"):
            raise PhenomenaError(f"{source}: {pid} `labels` entries need `slug` and `label`")
        thr = item.get("thr")
        ref = LabelRef(
            slug=str(item["slug"]),
            label=str(item["label"]),
            thr=None if thr is None else float(thr),
            weight=float(item.get("weight", 1.0)),
        )
        if ref.key in forecasts:
            raise PhenomenaError(
                f"{source}: {pid} lists {ref.key} under `labels:`, but labels.yaml declares it a "
                "forecast. A risk curve is a claim about the FUTURE and reaches a hit through "
                "`forecasts:`; scoring it as present-tense label evidence is the tier order's "
                "whole point (plan §2, §7)."
            )
        out.append(ref)
    return tuple(out)


def _events(source: Path, pid: str, value) -> tuple[EventRule, ...]:
    out = []
    for item in _seq(source, pid, value, "events"):
        if not isinstance(item, Mapping) or not item.get("source"):
            raise PhenomenaError(f"{source}: {pid} `events` entries need a `source`")
        unknown = set(item) - {
            "source", "phenomenon", "band_khz", "max_bandwidth_khz", "attrs", "weight", "caveat",
        }
        if unknown:
            raise PhenomenaError(f"{source}: {pid} `events` entry has unknown key(s) {sorted(unknown)}")
        attrs = item.get("attrs") or {}
        if not isinstance(attrs, Mapping):
            raise PhenomenaError(f"{source}: {pid} `events.attrs` must be a mapping")
        unknown = set(attrs) - {"min_harmonics", "chirp_sign", "pickup"}
        if unknown:
            raise PhenomenaError(f"{source}: {pid} `events.attrs` has unknown key(s) {sorted(unknown)}")
        caveat = item.get("caveat")
        if caveat is not None and str(caveat) not in EVENT_CAVEATS:
            raise PhenomenaError(
                f"{source}: {pid} `events.caveat` {caveat!r} is not one of "
                f"{sorted(EVENT_CAVEATS)}. The strings are constants in "
                "`ideate.retrieval.phenomena` because callers key on them."
            )
        weight = float(item.get("weight", 1.0))
        if not 0.0 < weight <= 1.0:
            raise PhenomenaError(
                f"{source}: {pid} `events.weight` must be in (0, 1]; got {weight}"
            )
        max_bw = item.get("max_bandwidth_khz")
        if max_bw is not None and float(max_bw) <= 0.0:
            raise PhenomenaError(f"{source}: {pid} `events.max_bandwidth_khz` must be positive")
        out.append(
            EventRule(
                source=str(item["source"]),
                phenomenon=None if item.get("phenomenon") is None else str(item["phenomenon"]),
                band_khz=_band(source, pid, item.get("band_khz")),
                max_bandwidth_khz=None if max_bw is None else float(max_bw),
                weight=weight,
                caveat=None if caveat is None else EVENT_CAVEATS[str(caveat)],
                min_harmonics=(
                    None if attrs.get("min_harmonics") is None else int(attrs["min_harmonics"])
                ),
                chirp_sign=None if attrs.get("chirp_sign") is None else int(attrs["chirp_sign"]),
                pickup=None if attrs.get("pickup") is None else bool(attrs["pickup"]),
            )
        )
    return tuple(out)


def _band(source: Path, pid: str, value) -> tuple[float | None, float | None] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise PhenomenaError(f"{source}: {pid} `band_khz` must be a [lo, hi] pair, either nullable")
    lo, hi = (None if v is None else float(v) for v in value)
    if lo is not None and hi is not None and hi < lo:
        raise PhenomenaError(f"{source}: {pid} `band_khz` is inverted: [{lo}, {hi}]")
    return (lo, hi)


# ------------------------------------------------------------------------------------- resolve


def resolve(text: str | None) -> list[tuple[str, float]]:
    """`"edge harmonic oscillation"` -> `[("eho", 1.0)]`. Longest alias wins; a denial vetoes.

    Matching is labelmaker's `hits()` and not a second tokenizer: it is the function that knows a
    `.` between two digits is part of the number, that `elm-free` is one token so " elm " is not
    inside it, and that a sentence is the span a negation applies over. A phenomenon whose every
    mention in `text` is a denial ("no EHO this shot") resolves to nothing rather than to itself,
    which is what `exclude` means.

    The weights are a rank, not a probability: 1.0 for the phenomenon whose longest alias matched,
    then 1/2, 1/3 ... for the others named alongside it. Returns `[]` for empty text and for text
    that names nothing -- which the caller must report as "nothing resolved", never as "no shot
    has one".
    """
    if not text or not text.strip():
        return []
    found = lexicon_mod.hits(text, _lexicon())
    known = set(registry())
    scored: list[tuple[str, int, int]] = []
    for pid, hs in found.items():
        if pid not in known:
            continue
        positive = [h for h in hs if h.polarity == "pos"]
        if not positive:  # every mention denied it
            continue
        scored.append((pid, max(len(h.alias) for h in positive), len(positive)))
    scored.sort(key=lambda t: (-t[1], -t[2], t[0]))
    return [(pid, 1.0 / (i + 1)) for i, (pid, _, _) in enumerate(scored)]


# ------------------------------------------------------------------------------------ evidence


def _phenomenon(ph: str | Phenomenon) -> Phenomenon:
    if isinstance(ph, Phenomenon):
        return ph
    reg = registry()
    if ph not in reg:
        raise PhenomenaError(f"unknown phenomenon {ph!r}; the registry has {sorted(reg)}")
    return reg[ph]


def _attrs(value) -> dict:
    """One event row's `attrs`, decoded. A row whose attrs will not parse has no descriptors --
    which fails every descriptor predicate, rather than passing them all by accident."""
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value:
        return {}
    try:
        out = json.loads(value)
    except ValueError:
        return {}
    return out if isinstance(out, dict) else {}


def _f(value) -> float | None:
    """A real number, or None. NaN is None: a Parquet round trip turns "not recorded" into NaN,
    and a caller that reads that as 0.0 has invented a measurement."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _centroid_khz(row: Mapping, attrs: Mapping) -> float | None:
    """Where the mode sits, in kHz. `attrs.f_centroid_khz` is the probability-weighted centre of
    a track and is what a band means; `f0_khz`/`f1_khz` is the track's full extent, which on a
    real row runs 0.5-248 kHz and would put one track in every band at once. The extent's midpoint
    is the fallback for a source that records no centroid."""
    f = _f(attrs.get("f_centroid_khz"))
    if f is not None:
        return f
    f0, f1 = _f(row.get("f0_khz")), _f(row.get("f1_khz"))
    if f0 is None or f1 is None:
        return f0 if f1 is None else f1
    return 0.5 * (f0 + f1)


def _bandwidth_khz(row: Mapping, attrs: Mapping) -> float | None:
    """How WIDE the track is, in kHz. `attrs.bandwidth_khz` is recorded on every real
    `tokeye_track` row (0 missing of 1,147 on the three labelmaker shots); the `f0-f1` extent is
    the fallback. None when neither is recorded -- and None never rejects a row."""
    bw = _f(attrs.get("bandwidth_khz"))
    if bw is not None:
        return abs(bw)
    f0, f1 = _f(row.get("f0_khz")), _f(row.get("f1_khz"))
    return None if f0 is None or f1 is None else abs(f1 - f0)


def _window(db, shot: int, segment: str) -> tuple[float, float] | None:
    """The segment's `[t0, t1]` in seconds, or None when the shot has no such segment."""
    seg_id = f"{shot}:{segment}"
    index = getattr(db.segments, "index", None)
    if index is None or seg_id not in index:
        return None
    row = db.segments.loc[seg_id]
    t0, t1 = _f(row.get("t0_ms")), _f(row.get("t1_ms"))
    return None if t0 is None or t1 is None else (t0 / 1000.0, t1 / 1000.0)


def _overlaps(row: Mapping, window: tuple[float, float] | None) -> bool:
    """Half-open overlap, with a point event (t1 == t0) still inside its own instant."""
    if window is None:
        return True
    t0, t1 = _f(row.get("t0_s")), _f(row.get("t1_s"))
    if t0 is None or t1 is None:
        return False
    return t0 <= window[1] and t1 >= window[0]


def _interval(row: Mapping, attrs: Mapping) -> Interval:
    return Interval(
        t0_s=float(row["t0_s"]),
        t1_s=float(row["t1_s"]),
        f0_khz=_f(row.get("f0_khz")),
        f1_khz=_f(row.get("f1_khz")),
        source=str(row["source"]),
        evidence_kind=str(row["evidence_kind"]),
        confidence=_f(row.get("confidence")),
        event_id=str(row["event_id"]),
    )


def _rows(frame: pd.DataFrame, shot: int) -> list[dict]:
    if frame is None or len(frame) == 0 or "shot" not in frame.columns:
        return []
    return frame.loc[frame["shot"] == shot].to_dict("records")


# path -> ((size, mtime_ns), frame) for `db/event_sources.parquet`, which `locate` would
# otherwise re-read once per candidate shot.
_SOURCES_CACHE: dict[str, tuple[tuple[int, int], pd.DataFrame]] = {}


def _sources_frame(path: Path) -> pd.DataFrame | None:
    try:
        st = path.stat()
    except OSError:
        return None
    stamp = (st.st_size, st.st_mtime_ns)
    hit = _SOURCES_CACHE.get(str(path))
    if hit is None or hit[0] != stamp:
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):  # a broken coverage table must not lose the events
            return None
        _SOURCES_CACHE[str(path)] = hit = (stamp, frame)
    return hit[1]


def _event_sources(db) -> pd.DataFrame | None:
    """labelmaker's source contract for this database, or None when it predates it.

    `db/event_sources.parquet` (`ideate.labels.event_sources`) is one row per source that ran or
    was deliberately skipped, and it is the only record of a detector that ran and emitted
    NOTHING -- which is the coverage that matters most, because it is the only thing that can
    turn "no rows" into a negative. Read through `getattr` and a file probe rather than required,
    so this module works unchanged on a database built before the contract, which is every
    database today.
    """
    frame = getattr(db, "event_sources", None)
    if frame is None:
        db_dir = getattr(db, "db_dir", None)
        frame = None if db_dir is None else _sources_frame(Path(db_dir) / "event_sources.parquet")
    if frame is None or len(frame) == 0:
        return None
    return frame if {"shot", "source", "status"} <= set(frame.columns) else None


def _looked_windows(db, shot: int, ph: Phenomenon) -> list[tuple[float, float]]:
    """Every stretch in which somebody looked for `ph` on this shot. Unmerged, unclipped."""
    covering = set(ph.covering_sources)
    if not covering:
        return []
    sources = _event_sources(db)
    if sources is not None:
        rows = [
            r
            for r in sources.loc[sources["shot"] == shot].to_dict("records")
            if str(r.get("source")) in covering and str(r.get("status")) == "ok"
        ]
    else:
        # No source table: fall back to the coverage columns of the rows the covering sources
        # did write. A forecast row's validity window is not a diagnostic's coverage, and
        # neither is a `text` or `model` row's, so only the observed kinds donate.
        rows = [
            r
            for r in _rows(getattr(db, "events", None), shot)
            if str(r.get("source")) in covering
            and str(r.get("evidence_kind")) in OBSERVED_KINDS
        ]
    out = []
    for r in rows:
        # Old elm_clock rows came from magnetics masks, including persisted
        # source keys a rerun leaves behind. They did not inspect D-alpha.
        if ph.id == "elm" and r.get("diag") != "filterscopes":
            continue
        a, b = _f(r.get("t_cov0_s")), _f(r.get("t_cov1_s"))
        if a is not None and b is not None and b >= a:
            out.append((a, b))
    return out


def _merge(windows: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """The union, as disjoint windows in time order. Two passes over the same stretch are one."""
    out: list[tuple[float, float]] = []
    for a, b in sorted(windows):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _clip(
    windows: Iterable[tuple[float, float]], window: tuple[float, float] | None
) -> list[tuple[float, float]]:
    """The part of each window inside `window`. A window that only TOUCHES the edge covers
    nothing of it, so the overlap has to be strictly positive."""
    if window is None:
        return list(windows)
    lo, hi = window
    out = []
    for a, b in windows:
        a2, b2 = max(a, lo), min(b, hi)
        if b2 > a2:
            out.append((a2, b2))
    return out


#: Slack on the two coverage/window comparisons, in seconds. A detector whose pass ends 1 ns
#: before the flat top does is not a partially covered shot.
_EPS = 1e-9


def _coverage_for(db, shot: int, ph: Phenomenon, window, segment: str):
    """`(state, hull, windows, partial, caveats)` -- which of `COVERAGE_STATES` this shot is in.

    The window filter is applied to the COVERAGE, not only to the events. That is the whole
    point: a pass that read the ramp-up says nothing about the flat top, and a `coverage` that
    reported it anyway would suppress the caveat, satisfy `--avoid` as an established negative,
    and turn a gap in the data into a physics result.
    """
    title = ph.title
    if not ph.covering_sources:
        return "unindexed", None, (), False, [NO_COVERAGE]
    raw = _merge(_looked_windows(db, shot, ph))
    if not raw:
        return "unprocessed", None, (), False, [COVERAGE_UNPROCESSED.format(title=title)]
    windows = tuple(_clip(raw, window))
    if not windows:
        return (
            "uncovered", None, (), False,
            [COVERAGE_OUTSIDE_WINDOW.format(title=title, segment=segment)],
        )
    hull = (windows[0][0], windows[-1][1])
    caveats: list[str] = []
    if len(windows) > 1:
        caveats.append(COVERAGE_GAPS.format(title=title, segment=segment, n=len(windows) - 1))
    partial = window is not None and (
        len(windows) > 1 or hull[0] > window[0] + _EPS or hull[1] < window[1] - _EPS
    )
    if partial:
        covered = ", ".join(f"{a:.3f}-{b:.3f} s" for a, b in windows)
        caveats.append(COVERAGE_PARTIAL.format(title=title, segment=segment, covered=covered))
    return "observed", hull, windows, partial, caveats


def _keeps_confidence(row: Mapping, limit: float) -> bool | None:
    """True/False for a scored row, None for one whose source recorded no confidence."""
    c = _f(row.get("confidence"))
    if c is None:
        return None
    return c >= limit


def evidence(
    shot: int,
    ph: str | Phenomenon,
    db,
    segment: str = "flat_top",
    *,
    min_confidence: float = 0.0,
    label_floor: float | None = None,
) -> Evidence:
    """Every class of evidence for one phenomenon on one shot, kept apart and never merged.

    `segment` is the window the observed intervals and the forecasts are clipped to; a shot with
    no such segment is searched whole and says so in a caveat, because silently widening the
    window would make a ramp-down ELM answer a question about the flat top.

    `min_confidence` drops events below the bar. A row whose source recorded NO confidence is
    kept when the bar is 0 and dropped above it -- it cannot be shown to clear a bar it was never
    scored against -- and the count of those is reported as a caveat rather than vanishing.

    `label_floor` is how high a detection label has to score to be evidence; None reads
    `retrieval.yaml`. See `DEFAULT_LABEL_FLOOR`: the model running is not the same fact as the
    model saying yes.
    """
    ph = _phenomenon(ph)
    if label_floor is None:
        label_floor = _config()[2]
    window = _window(db, shot, segment)
    caveats: list[str] = []
    if window is None:
        caveats.append(NO_SEGMENT.format(segment=segment))

    rows = _rows(getattr(db, "events", None), shot)
    intervals: list[Interval] = []
    refs: list[EventRef] = []
    forecasts: list[Interval] = []
    rule_caveats: list[str] = []
    event_weight = 0.0
    n_unscored = 0
    other_kinds: dict[str, int] = {}
    for row in rows:
        attrs = _attrs(row.get("attrs"))
        kind = str(row.get("evidence_kind"))
        if not _overlaps(row, window):
            continue
        rule = None
        if kind == FORECAST_KIND:
            if str(row.get("source")) not in ph.forecasts or str(row.get("phenomenon")) != ph.id:
                continue
        else:
            rule = next((r for r in ph.events if r.matches(row, attrs)), None)
            if rule is None:
                continue
            if kind not in OBSERVED_KINDS:
                # It matched the rule and it is not a forecast, and it is still not an
                # observation: `text` is somebody typing and `model` is a model scoring. Counted
                # in a caveat rather than dropped silently, because a row nobody sees is a row
                # nobody can correct the registry with.
                other_kinds[kind] = other_kinds.get(kind, 0) + 1
                continue
        keep = _keeps_confidence(row, min_confidence)
        if keep is False:
            continue
        if keep is None and min_confidence > 0.0:
            n_unscored += 1
            continue
        if rule is None:
            forecasts.append(_interval(row, attrs))
        else:
            intervals.append(_interval(row, attrs))
            event_weight += rule.weight
            if rule.caveat is not None:
                rule_caveats.append(rule.caveat)
            refs.append(
                EventRef(
                    shot=int(row["shot"]),
                    event_id=str(row["event_id"]),
                    source=str(row["source"]),
                    phenomenon=str(row["phenomenon"]),
                    t0_s=float(row["t0_s"]),
                    t1_s=float(row["t1_s"]),
                )
            )
    if n_unscored:
        caveats.append(DROPPED_UNSCORED.format(n=n_unscored, limit=min_confidence))
    if other_kinds:
        caveats.append(
            UNCLASSIFIED_KIND.format(
                n=sum(other_kinds.values()), kinds=", ".join(sorted(other_kinds))
            )
        )
    caveats.extend(dict.fromkeys(rule_caveats))
    state, coverage, cov_windows, partial, cov_caveats = _coverage_for(
        db, shot, ph, window, segment
    )
    caveats.extend(cov_caveats)
    if not intervals:
        caveats.append(NO_OBSERVED)

    label_evidence, max_p, label_caveats = _labels_for(db, shot, ph, label_floor)
    caveats.extend(label_caveats)

    hits, snippets, n_neg, text_caveats = _text_for(db, shot, ph)
    caveats.extend(text_caveats)

    return Evidence(
        shot=int(shot),
        phenomenon=ph.id,
        window=window,
        intervals=tuple(sorted(intervals, key=lambda iv: (iv.t0_s, iv.event_id))),
        refs=tuple(sorted(refs, key=lambda r: (r.t0_s, r.event_id))),
        forecasts=tuple(sorted(forecasts, key=lambda iv: (iv.t0_s, iv.event_id))),
        label_evidence=label_evidence,
        max_label_p=max_p,
        text_hits=hits,
        text_snippets=snippets,
        n_negative_claims=n_neg,
        coverage=coverage,
        coverage_windows=tuple(cov_windows),
        coverage_state=state,
        coverage_partial=partial,
        event_weight=event_weight,
        in_database=_in_database(ph, shot),
        caveats=tuple(dict.fromkeys(caveats)),
    )


def _labels_for(db, shot: int, ph: Phenomenon, floor: float):
    """`{"<slug>/<label>.max_valid": p | None, ...}`, the best p, and the caveats.

    None, never 0.0, for a label that was not run on this shot or whose every sample was invalid:
    `n_valid == 0` means the model emitted nothing it stood behind, and a 0.0 there reads as "the
    model looked and said no" (plan §2, "unavailable" is never rendered as 0).

    A probability BELOW the label's floor (`LabelRef.thr`, else `floor`) is reported and does not
    count: "the model ran" is not evidence, and on the real database a quarter of the tearing
    label's scored shots are at or under 0.010. The distinction between "unavailable" and "scored
    low" survives: the first is None in `label_evidence`, the second is the number itself.
    """
    if not ph.labels:
        return {}, None, [NO_LABEL_MODEL]
    frame = getattr(db, "labels_wide", None)
    rows = _rows(frame, shot)
    by_key = {f"{r['slug']}/{r['label']}": r for r in rows}
    out: dict[str, float | None] = {}
    best: float | None = None
    low: list[str] = []
    unavailable = False
    for ref in ph.labels:
        row = by_key.get(ref.key)
        n_valid = None if row is None else _f(row.get("n_valid"))
        if row is None or not n_valid:
            out[f"{ref.key}.max_valid"] = None
            out[f"{ref.key}.frac_above"] = None
            unavailable = True
            continue
        p = _f(row.get("max_valid"))
        out[f"{ref.key}.max_valid"] = p
        out[f"{ref.key}.frac_above"] = _f(row.get("frac_above"))
        if p is None:
            continue
        bar = floor if ref.thr is None else ref.thr
        if p < bar:
            low.append(LABEL_BELOW_FLOOR.format(key=ref.key, p=p, floor=bar))
            continue
        weighted = p * ref.weight
        best = weighted if best is None else max(best, weighted)
    caveats = ([LABEL_UNAVAILABLE] if unavailable else []) + low
    return out, best, caveats


#: How many `text_claims` snippets a hit carries. The table is the record; this is a reading aid.
MAX_SNIPPETS = 3


def _text_for(db, shot: int, ph: Phenomenon):
    """Positive, observed claims: their count, up to `MAX_SNIPPETS` snippets, and the caveats.

    Only `polarity == "pos"` and `temporality == "observed"` counts: a plan ("try for QH next
    shot") and a back-reference ("like shot 190090") are claims about something else. A negative
    claim is not subtracted -- it is a caveat, in the operators' own frame.
    """
    rows = [r for r in _rows(getattr(db, "text_claims", None), shot)
            if str(r.get("phenomenon")) == ph.id]
    positive = [r for r in rows
                if str(r.get("polarity")) == "pos" and str(r.get("temporality")) == "observed"]
    negative = [r for r in rows if str(r.get("polarity")) == "neg"]
    caveats: list[str] = []
    if negative:
        caveats.append(NEGATIVE_CLAIM.format(title=ph.title))
    if not positive:
        caveats.append(NO_TEXT)
    elif all(str(r.get("scope")) == "run" for r in positive):
        # A run-scope sentence is shared by every shot of the session -- an experiment title, a
        # control-system note. Measured over the 500-shot database: elm and rwm fire on 99.4 % of
        # shots at run scope and 11.0 %/1.0 % at shot scope. It separates almost nothing.
        caveats.append(RUN_SCOPE_TEXT)
    ordered = sorted(positive, key=lambda r: (str(r.get("scope")) != "shot",))
    snippets = tuple(str(r.get("snippet", "")) for r in ordered[:MAX_SNIPPETS] if r.get("snippet"))
    return len(positive), snippets, len(negative), caveats


@functools.lru_cache(maxsize=4)
def _database_shots(path_key: str) -> frozenset[int]:
    """The shots a curated list names, via `paths.yaml`'s key for it. Empty when it is not here."""
    # `_qh_shots` is the ONE reader of that CSV (BOM and all); a second copy here would be a
    # second answer to "which shots does the QH database name".
    from ..shotdb.build import _qh_shots

    try:
        path = getattr(config.load_paths(), path_key, None)
    except (OSError, ValueError):
        return frozenset()
    return frozenset() if path is None else _qh_shots(str(path))


@functools.lru_cache(maxsize=4)
def _table_shots(stems: tuple[str, ...]) -> frozenset[int]:
    """The shots labelmaker's curated tables name, by manifest stem.

    The second way a `database:` block can point at a list, and the one new tables use:
    `data/events/tables.yaml` already declares where the CSV is and what its columns mean, so
    the registry names the STEM and labelmaker resolves it. A stem the manifest does not know,
    or a manifest that cannot be read at all, is an empty set and not an exception -- the
    tables are optional data and a database built without them must still rank.
    """
    from labelmaker.events import databases as label_tables

    try:
        specs = {spec.stem: spec for spec in label_tables.load_manifest()}
    except (OSError, ValueError):
        return frozenset()
    out: set[int] = set()
    for stem in stems:
        spec = specs.get(str(stem))
        if spec is None:
            continue
        try:
            out |= label_tables.shots(spec)
        except (OSError, ValueError):
            continue
    return frozenset(out)


def _in_database(ph: Phenomenon, shot: int) -> bool:
    """Does a curated list name this shot? Membership, and never an observation.

    `_tier` puts DATABASE below FORECAST, so a shot whose only evidence is a curated listing
    can never come back in the observed class, and `DATABASE_ONLY` is the caveat it carries.
    That is the same rule labelmaker writes on the rows themselves (NaN coverage, NaN
    confidence): a list names a shot, it does not measure one.
    """
    if ph.database is None:
        return False
    key = ph.database.get("path_key")
    if key:
        return int(shot) in _database_shots(str(key))
    tables = ph.database.get("tables") or ()
    return bool(tables) and int(shot) in _table_shots(
        tuple(str(t) for t in tables)
    )


# -------------------------------------------------------------------------------------- locate

#: Evidence classes, best first. `locate` sorts by this BEFORE it sorts by score, so no weighting
#: of the arithmetic can put a forecast above an observation (plan §7). Between the two middle
#: tiers: a detection label is a model's claim about the present and a forecast is a model's claim
#: about the future, and the present-tense one is the closer answer to "did this happen".
OBSERVED, LABELLED, FORECAST, DATABASE, TEXTUAL = 4, 3, 2, 1, 0


def _tier(ev: Evidence) -> int:
    if ev.intervals:
        return OBSERVED
    if ev.max_label_p is not None:
        return LABELLED
    if ev.forecasts:
        return FORECAST
    if ev.in_database:
        return DATABASE
    return TEXTUAL


def has_evidence(ev: Evidence) -> bool:
    """Is there anything at all to show? A shot none of the four classes says a word about is
    not a weak hit, it is a non-answer, and returning it at score 0 with a "TEXT ONLY" caveat
    would attach a claim to a shot nothing claimed anything about."""
    return bool(
        ev.intervals or ev.forecasts or ev.text_hits or ev.in_database
        or ev.max_label_p is not None
    )


def _config() -> tuple[dict[str, float], float, float]:
    block = config.load_yaml("retrieval.yaml").get(RETRIEVAL_BLOCK) or {}
    weights = {**DEFAULT_WEIGHTS, **{k: float(v) for k, v in (block.get("weights") or {}).items()}}
    return (
        weights,
        float(block.get("saturation_n", DEFAULT_SATURATION_N)),
        float(block.get("label_floor", DEFAULT_LABEL_FLOOR)),
    )


def sat(n: float, saturation_n: float = DEFAULT_SATURATION_N) -> float:
    """`1 - exp(-n / n0)`: three detections is already "this shot has them", and three hundred is
    not a hundred times the evidence that it does.

    `n` is the summed rule WEIGHT, not the row count (`Evidence.event_weight`), so a detector
    that does not detect the class it is registered under contributes less than one row's worth.
    The exact curve is pinned by a test written in `math.exp`, not in this function: it is the
    one piece of arithmetic here that encodes a physics judgement, and a stub that returned 1.0
    for any positive count would otherwise pass the suite.
    """
    return 0.0 if n <= 0 else float(1.0 - math.exp(-n / saturation_n))


def score(ev: Evidence, weights: Mapping[str, float] | None = None, saturation_n: float | None = None) -> float:
    """The four-term score. See `configs/ideate/retrieval.yaml`'s `phenomenon:` block.

    Text alone is capped at labelmaker's `TEXT_ONLY_CEILING` -- four independent mentions and no
    more, which is what one mention is worth times four -- because a shot whose only evidence is
    that somebody typed the word is not a shot where the thing was measured.
    """
    if weights is None or saturation_n is None:
        cfg_weights, cfg_sat, _floor = _config()
        weights = cfg_weights if weights is None else weights
        saturation_n = cfg_sat if saturation_n is None else saturation_n
    total = 0.0
    if ev.max_label_p is not None:
        total += weights.get("label", 1.0) * ev.max_label_p
    if ev.intervals:
        total += weights.get("event", 1.0) * sat(ev.event_weight, saturation_n)
    if ev.text_hits:
        total += weights.get("text", 0.5) * math.tanh(ev.text_hits / 2.0)
    if ev.in_database:
        total += weights.get("database", 1.0)
    if _tier(ev) == TEXTUAL:
        total = min(total, lexicon_mod.TEXT_ONLY_CEILING)
    return float(total)


def _avoid_ids(avoid: Iterable[str]) -> list[str]:
    """`"phenomenon:elm"` or `"elm"` -> `"elm"`. Anything else is an error, not a silent no-op."""
    reg, out = registry(), []
    for token in avoid or ():
        pid = str(token).split(":", 1)[1] if str(token).startswith("phenomenon:") else str(token)
        if pid not in reg:
            raise PhenomenaError(
                f"--avoid {token!r}: {pid!r} is not a phenomenon; the registry has {sorted(reg)}"
            )
        out.append(pid)
    # De-duplicated: `--avoid elm --avoid phenomenon:elm` is one constraint, and evaluating it
    # twice per candidate would double the work and print the caveat twice.
    return list(dict.fromkeys(out))


def _candidates(db, segment: str, constraints) -> list[int]:
    mask = db.mask(segment, constraints)
    shots = db.segments.loc[mask, "shot"] if mask.any() else db.segments.loc[[], "shot"]
    return sorted({int(s) for s in shots})


def locate(
    ph: str | Phenomenon,
    db,
    n: int = 20,
    *,
    segment: str = "flat_top",
    constraints=None,
    min_confidence: float = 0.0,
    avoid: Iterable[str] = (),
    notes: list[str] | None = None,
) -> list[PhenomenonHit]:
    """The shots where `ph` happened, best evidence first.

    Ordering is `(evidence class, score)`: observed, then label, then forecast, then curated
    database, then text-only, and only inside a class does the score decide. That is not a tie-break convenience -- it is
    plan §7's rule, and it is what stops a shot with a hundred forecast rows from outranking the
    one shot a detector actually saw the mode on.

    `avoid` takes `phenomenon:<id>` tokens. A shot with OBSERVED evidence of the avoided
    phenomenon inside the window is dropped. A shot where nothing looked for it is KEPT, with a
    caveat saying so: no data is not a negative, and silently dropping those would turn a gap in
    the diagnostic coverage into a physics claim. The only case that carries NO caveat is the
    real negative -- a detector that covered the whole window and found nothing; partial cover,
    cover elsewhere in the record, a detector that never ran and a phenomenon nothing detects
    are four different sentences, and `AVOID_COVERAGE_CAVEATS` has one for each.

    `notes` is filled, when a list is passed, with what the run did to shots that are NOT in the
    result: how many `avoid` dropped and on what kind of evidence. A dropped shot cannot carry a
    caveat, so without this the strongest claim the filter makes is the one nothing reports.
    """
    ph = _phenomenon(ph)
    weights, saturation_n, floor = _config()
    avoid_ids = _avoid_ids(avoid)
    cache: dict[tuple[int, str], Evidence] = {}

    def ev_for(shot: int, entry: Phenomenon) -> Evidence:
        key = (int(shot), entry.id)
        if key not in cache:
            cache[key] = evidence(
                shot, entry, db, segment, min_confidence=min_confidence, label_floor=floor
            )
        return cache[key]

    n_dropped: dict[str, int] = dict.fromkeys(avoid_ids, 0)
    drop_caveats: dict[str, dict[str, int]] = {pid: {} for pid in avoid_ids}
    scored: list[tuple[int, float, Evidence, list[str]]] = []
    for shot in _candidates(db, segment, constraints):
        ev = ev_for(shot, ph)
        extra: list[str] = []
        dropped = False
        for other in avoid_ids:
            other_ph = registry()[other]
            oev = ev_for(shot, other_ph)
            if oev.intervals:
                dropped = True
                n_dropped[other] += 1
                for caveat in EVENT_CAVEATS.values():
                    if caveat in oev.caveats:
                        drop_caveats[other][caveat] = drop_caveats[other].get(caveat, 0) + 1
                break
            state = (
                "partial"
                if oev.coverage_state == "observed" and oev.coverage_partial
                else oev.coverage_state
            )
            template = AVOID_COVERAGE_CAVEATS.get(state)
            if template is not None:
                extra.append(
                    template.format(token=f"phenomenon:{other}", title=other_ph.title)
                )
        if dropped or not has_evidence(ev):
            continue
        scored.append((_tier(ev), score(ev, weights, saturation_n), ev, extra))
    if notes is not None:
        for pid, count in n_dropped.items():
            if not count:
                continue
            notes.append(
                AVOID_DROPPED.format(
                    token=f"phenomenon:{pid}", n=count, title=registry()[pid].title
                )
            )
            for caveat, n_caveated in sorted(drop_caveats[pid].items()):
                notes.append(AVOID_DROPPED_CAVEATED.format(n=n_caveated, caveat=caveat))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2].shot))
    return [
        _hit(ev, tier, value, extra, db, segment) for tier, value, ev, extra in scored[: max(int(n), 0)]
    ]


def _hit(ev: Evidence, tier: int, value: float, extra: list[str], db, segment: str) -> PhenomenonHit:
    caveats = list(ev.caveats) + extra
    if tier == TEXTUAL:
        caveats.insert(0, TEXT_ONLY)
    elif tier == FORECAST:
        caveats.insert(0, FORECAST_ONLY)
    elif tier == LABELLED:
        caveats.insert(0, LABEL_ONLY.format(p=ev.max_label_p))
    elif tier == DATABASE:
        caveats.insert(0, DATABASE_ONLY)
    quote, role, quote_caveat = _quote(ev, db)
    if quote_caveat is not None:
        caveats.append(quote_caveat)
    row = _shot_row(db, ev.shot)
    return PhenomenonHit(
        shot=ev.shot,
        phenomenon=ev.phenomenon,
        score=value,
        intervals=list(ev.intervals),
        total_duration_s=ev.total_duration_s,
        coverage=ev.coverage,
        coverage_windows=[tuple(w) for w in ev.coverage_windows],
        coverage_state=ev.coverage_state,
        quote=quote,
        quote_role=role,
        text_snippets=list(ev.text_snippets),
        actuators_at_onset=_actuators(db, ev.shot, segment),
        label_evidence=dict(ev.label_evidence),
        forecasts=list(ev.forecasts),
        caveats=list(dict.fromkeys(caveats)),
        run_id=None if row is None else _str(row.get("run_id")),
        mp_title=None if row is None else _str(row.get("mp_title")),
    )


def _str(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    return text or None


def _shot_row(db, shot: int) -> dict | None:
    shots = getattr(db, "shots", None)
    if shots is None or shot not in shots.index:
        return None
    return shots.loc[shot].to_dict()


def _mentions(text: str, pid: str) -> bool:
    """Does this sentence NAME the phenomenon? labelmaker's matcher, so the quote picker and
    `resolve` agree about what counts as a mention, including its denials."""
    return any(h.polarity == "pos" for h in lexicon_mod.hits(text, _lexicon()).get(pid, ()))


def _quote(ev: Evidence, db) -> tuple[str | None, str | None, str | None]:
    """One logbook entry, or one claim snippet: the text, its role, and a caveat when it needs one.

    `describe.best_quote` returns a whole single `LogEntry` -- one author, one timestamp -- and
    nothing here joins two of them; splicing two operators' sentences into one quotation is the
    fabrication this module is most able to commit. The fallback is a `text_claims` snippet,
    flagged `quote_role="claim_snippet"` so a reader is never told an extract from a sentence is
    a logbook entry.

    An entry that MENTIONS the phenomenon is preferred over the shot's best entry, and when there
    is none the hit says so: a quotation printed beside a hit reads as the reason for the hit, and
    on the real database the top EHO hit was quoting a dud trip on the locked-mode detector.
    """
    try:
        rec = db.get(ev.shot)
    except (KeyError, ValueError, OSError):
        rec = None
    caveat = None
    found = None
    if rec is not None:
        found = describe_mod.best_quote(rec, where=lambda text: _mentions(text, ev.phenomenon))
        if found is None:
            found = describe_mod.best_quote(rec)
            if found is not None:
                caveat = QUOTE_UNRELATED.format(title=_phenomenon(ev.phenomenon).title)
    if found is not None:
        entry, text = found
        return text, entry.role, caveat
    if ev.text_snippets:
        # A claim snippet is a sentence about this phenomenon by construction.
        return ev.text_snippets[0], "claim_snippet", None
    return None, None, None


def _actuators(db, shot: int, segment: str) -> dict[str, float | None]:
    """`ACTUATOR_COLUMNS` for the shot's segment. Read that constant's docstring before reading
    the numbers: they are the SEGMENT's statistics, not the values at the phenomenon's onset."""
    seg_id = f"{shot}:{segment}"
    out: dict[str, float | None] = dict.fromkeys(ACTUATOR_COLUMNS)
    segments = getattr(db, "segments", None)
    if segments is None or seg_id not in segments.index:
        return out
    row = segments.loc[seg_id]
    for col in ACTUATOR_COLUMNS:
        out[col] = _f(row[col]) if col in segments.columns else None
    return out


__all__ = [
    "CORPUS_GROUPS",
    "COVERAGE_STATES",
    "EventRule",
    "Evidence",
    "LabelRef",
    "PhenomenaError",
    "Phenomenon",
    "evidence",
    "has_evidence",
    "locate",
    "registry",
    "resolve",
    "score",
    "titles",
]
