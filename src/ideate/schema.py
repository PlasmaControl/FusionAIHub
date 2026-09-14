"""All Pydantic models: the per-shot record (three tiers + provenance), the query state,
sessions and the two plain-file interop formats. No I/O, no numpy here.

Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from itertools import pairwise
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

Status = Literal["present", "unavailable", "pending", "not_installed"]
SegName = Literal["full", "ramp_up", "flat_top", "ramp_down"]
Regime = Literal["L", "H", "neg_tri", "QH", "unknown"]


class Provenance(BaseModel):
    tool: str  # "EFIT", "d3d_fusion_data", "toksearch", "QH_Database", "ideate"
    version: str | None = None  # "01" (EFIT01), "kinetic", ...
    tree: str | None = None  # MDSplus tree, e.g. "efit01"
    run_id: str | None = None  # CAKE/OMFIT run id when applicable
    ts: datetime | None = None
    assumed: bool = False  # staged d3d_fusion_data files do not record the EFIT run


class Segment(BaseModel):
    name: SegName
    t0_ms: float
    t1_ms: float
    raw: dict[str, float | None] = Field(default_factory=dict)  # e.g. ip_mean, pnbi_15L_peak
    derived: dict[str, float | None] = Field(
        default_factory=dict
    )  # e.g. betan_mean, q95_min (EFIT)


class LogEntry(BaseModel):
    role: str  # SESSION_LEADER | PHYSICS_OPERATOR | CHIEF_OPERATOR | PCS | ...
    author: str | None = None
    time: str | None = None  # "HH:MM"
    text: str


class HumanTier(BaseModel):
    mpid: str | None = None
    mp_title: str | None = None
    mp_purpose: str | None = None
    mp_text_source: Literal["bundle", "pdf", "md_header", "none"] = "none"
    run_id: str | None = None
    run_title: str | None = None
    session_leaders: list[str] = Field(default_factory=list)
    shot_brief: str | None = None
    precomment: str | None = None
    log_entries: list[LogEntry] = Field(default_factory=list)
    chief_operator_status: str | None = None
    quality_comment: str | None = None
    requested: dict[str, float] = Field(
        default_factory=dict
    )  # parsed "requested ip/btor/pnbi/pech"
    refshot: int | None = None
    mp_step: str | None = None
    verdict: Literal["good", "bad", "mixed", "unknown"] = "unknown"


class Labels(BaseModel):
    regime: Regime = "unknown"
    regime_source: Literal["database", "text", "heuristic", "none"] = "none"
    operational: set[str] = Field(default_factory=set)
    cluster: int | None = None

    @field_serializer("operational")
    def _sorted(self, v: set[str]) -> list[str]:
        return sorted(v)


class Outcome(BaseModel):
    ip_target_err: float | None = None  # (measured - target) / target over the flat top
    ip_target_hit: bool | None = None  # |err| <= 0.15
    nbi_target_err: float | None = None
    nbi_target_hit: bool | None = None  # None when the BMSPINJ unit (MW vs kW) is ambiguous
    nbi_target_unit: str | None = None
    flat_top_ms: float | None = None
    ended_early: bool | None = None  # Ip collapsed while the PCS target was still high
    fast_quench: bool | None = None  # Ip fell from >50% to <10% of flat top within 50 ms
    end_reason: str | None = None
    fault_strings: list[str] = Field(default_factory=list)


class ShotRecord(BaseModel):
    schema_version: str = "1"
    shot: int
    shot_date: date | None = None
    campaign: str
    segments: list[Segment]
    derived_provenance: dict[str, Provenance] = Field(default_factory=dict)
    raw_sources: dict[str, Literal["staged", "fetched", "corpus", "labelmaker"]] = Field(
        default_factory=dict
    )
    # Which raw layer answered, per signal, when that layer has more than one source of its own
    # ("labelmaker:archive", "labelmaker:fdp", "corpus"). Empty on a legacy build, where
    # `raw_sources` is already the whole answer.
    feature_resolvers: dict[str, str] = Field(default_factory=dict)
    # The raw groups the reader saw for this shot -- a coverage fact about the shot, not about a
    # signal, and the one `has_<group>`/`groups_filled` in shots.parquet are derived from.
    raw_groups: list[str] = Field(default_factory=list)
    reader: str = "legacy"  # which reader built this record
    has_frame_codes: bool = False  # <data_root>/frame_codes/<shot>.pt exists
    human: HumanTier = Field(default_factory=HumanTier)
    labels: Labels = Field(default_factory=Labels)
    outcome: Outcome = Field(default_factory=Outcome)
    coverage: dict[str, Status] = Field(default_factory=dict)
    # Why a signal is `unavailable`, for the reasons that are not the ordinary "DIII-D did not
    # record it": an address naming a channel the file does not have, a group stored 3-D. Only
    # the reader that can say (the corpus one) fills it in, and only for the signals it applies
    # to -- an empty dict is the normal case, not a gap.
    coverage_reasons: dict[str, str] = Field(default_factory=dict)
    built_at: datetime
    builder_sha: str

    def segment(self, name: SegName) -> Segment | None:
        return next((s for s in self.segments if s.name == name), None)


class Range(BaseModel):
    lo: float | None = None
    hi: float | None = None


class QueryState(BaseModel):
    text: str | None = None
    negatives: list[str] = Field(default_factory=list)
    ref_shot: int | None = None
    ref_window_ms: tuple[float, float] | None = None
    segment: SegName = "flat_top"
    constraints: dict[str, Range] = Field(default_factory=dict)
    actuators: dict[str, float] = Field(default_factory=dict)  # {"nbi.total": 5e6, "ech.LUKE": 8e5}
    require_labels: set[str] = Field(default_factory=set)
    avoid_labels: set[str] = Field(default_factory=set)
    exclude_shots: set[int] = Field(default_factory=set)
    exclude_runs: set[str] = Field(default_factory=set)
    channel_weights: dict[str, float] | None = None
    prefer_outcome: Literal["any", "success"] = "any"
    n: int = 10


class Flag(BaseModel):
    rule_id: str
    severity: Literal["info", "warn", "error"]
    message: str
    value: float | None = None
    limit: float | None = None
    source: str = "config"


Seed = Literal["reference", "median", "edited"]


class Vertex(BaseModel):
    """A breakpoint of a piecewise-linear actuator waveform: PCS's Vertex(x, y) shape."""

    t_s: float  # seconds from shot time 0 (the IGNITE/PCS convention)
    y: float  # in the actuator's units (configs/ideate/actuators.yaml)


class ActuatorWaveform(BaseModel):
    key: str  # a QueryState.actuators key: "nbi.total", "ech.LUKE"
    units: str | None = None
    description: str = ""
    vertices: list[Vertex]
    seed: Seed = "reference"
    source_shots: list[int] = Field(default_factory=list)

    @field_validator("vertices")
    @classmethod
    def _sorted_unique_finite(cls, v: list[Vertex]) -> list[Vertex]:
        if not v:
            raise ValueError("a waveform needs at least one vertex")
        if any(not (math.isfinite(p.t_s) and math.isfinite(p.y)) for p in v):
            raise ValueError("vertex times and values must be finite")
        v = sorted(v, key=lambda p: p.t_s)
        for a, b in pairwise(v):
            if a.t_s == b.t_s:
                raise ValueError(f"two vertices at t = {a.t_s} s")
        return v


class ActuationSet(BaseModel):
    """What the Actuation tab edits and `POST /api/actuation` stores. `id`, `created` and `flags`
    are the server's: the page sends them null/empty and gets the stored set back."""

    id: str | None = None
    created: datetime | None = None
    source_shot: int | None = None
    segment: SegName = "flat_top"
    query: QueryState = Field(default_factory=QueryState)
    waveforms: dict[str, ActuatorWaveform] = Field(default_factory=dict)
    gas_species: dict[str, str] = Field(
        default_factory=dict
    )  # valve id -> species, e.g. {"GASA": "D2"}
    flags: list[Flag] = Field(default_factory=list)
    notes: str = ""
    schema_version: Literal["1"] = "1"

    @model_validator(mode="after")
    def _keys_match(self) -> ActuationSet:
        for k, w in self.waveforms.items():
            if w.key != k:
                raise ValueError(f"waveform stored under {k!r} says key={w.key!r}")
        return self


class Explanation(BaseModel):
    matched_constraints: list[str] = Field(default_factory=list)
    top_similar: list[tuple[str, float, float]] = Field(default_factory=list)
    top_different: list[tuple[str, float, float]] = Field(default_factory=list)
    channel_ranks: dict[str, int | None] = Field(default_factory=dict)
    text_highlight: str | None = None


class ResultItem(BaseModel):
    id: str  # "161234:flat_top"
    shot: int
    segment: SegName
    score: float
    description: str
    caveats: list[str] = Field(default_factory=list)
    polished: bool = False
    blurb: str | None = None  # the offline two-sentence summary from shots.parquet
    explanation: Explanation = Field(default_factory=Explanation)
    flags: list[Flag] = Field(default_factory=list)
    labels: Labels = Field(default_factory=Labels)
    outcome: Outcome = Field(default_factory=Outcome)
    run_id: str | None = None


class Interval(BaseModel):
    """One stretch of one shot that one source claims a phenomenon occupied.

    `evidence_kind` is labelmaker's (`detector`, `heuristic`, `forecast`, `text`, ...) and is what
    separates the two lists a `PhenomenonHit` keeps: an observation and a forecast are both
    intervals and are never the same claim. A point event -- an ELM, an L-H transition -- has
    `t1_s == t0_s`; a spectrogram track carries its frequency extent and a transient carries none.
    `confidence` is None where the source recorded none, never 0.0: "nobody scored it" and "scored
    it zero" are different facts (plan §2).
    """

    t0_s: float
    t1_s: float
    f0_khz: float | None = None
    f1_khz: float | None = None
    source: str
    evidence_kind: str
    confidence: float | None = None
    event_id: str


class EventRef(BaseModel):
    """The identity of one `events.parquet` row: enough to fetch it again, and no interpretation.

    Returned beside the intervals so a caller that wants the whole row -- the attrs the detector
    wrote, its coverage window, the run that produced it -- goes back to the table rather than
    being handed a summary that quietly became the record.
    """

    shot: int
    event_id: str
    source: str
    phenomenon: str  # the SOURCE's own string ("coherent_mode", "elm_free"), not ideate's id
    t0_s: float
    t1_s: float


class PhenomenonHit(BaseModel):
    """One shot's evidence for one phenomenon, with every class of it kept apart.

    The separation is the point. `intervals` is what a diagnostic showed; `forecasts` is what a
    model estimated was about to happen, and it never merges into the first (plan §2, §7).
    `label_evidence` holds probabilities from `labels_wide` and uses None -- never 0.0 -- for a
    label that was not run or had no valid samples. `text_snippets` is what the operators wrote,
    which is a claim and not a label by itself, so a hit resting on nothing else is capped and
    carries the caveat "TEXT ONLY". `caveats` is not decoration: every hit missing an evidence
    class says so there, and a reader who drops them is reading a different claim.
    """

    shot: int
    phenomenon: str
    score: float
    intervals: list[Interval] = Field(default_factory=list)
    total_duration_s: float = 0.0
    # The part of the searched window the phenomenon's detectors actually covered, as a hull.
    # None whenever `coverage_state` is not "observed": a detector that read the ramp-up has not
    # looked at the flat top, and reporting its window here would read as "we looked and it was
    # not there" about a stretch nobody read.
    coverage: tuple[float, float] | None = None
    # The same intersection as the real union, gaps and all; `coverage` is its hull.
    coverage_windows: list[tuple[float, float]] = Field(default_factory=list)
    # Which of the four states the coverage is in -- see `phenomena.COVERAGE_STATES`. Only
    # "observed" makes an empty `intervals` a negative; "unindexed" (nothing looks for this at
    # all), "unprocessed" (its detectors have not run here) and "uncovered" (they ran elsewhere
    # in the record) are three different ways of saying the database cannot tell you.
    coverage_state: Literal["unindexed", "unprocessed", "uncovered", "observed"] | None = None
    quote: str | None = None
    quote_role: str | None = None
    text_snippets: list[str] = Field(default_factory=list)
    actuators_at_onset: dict[str, float | None] = Field(default_factory=dict)
    label_evidence: dict[str, float | None] = Field(default_factory=dict)
    forecasts: list[Interval] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    run_id: str | None = None
    mp_title: str | None = None


class Session(BaseModel):
    session_id: str
    created: datetime
    parent_id: str | None = None
    query: QueryState
    results: list[ResultItem] = Field(default_factory=list)
    notes: str = ""
    db_manifest_sha: str


class ShotSummary(BaseModel):
    """Flat interop export of one shot (JSON/Parquet), importable by OMFIT/notebooks."""

    shot: int
    shot_date: date | None
    campaign: str
    mpid: str | None
    mp_title: str | None
    run_id: str | None
    flat_top: dict[str, float | None]
    labels: Labels
    outcome: Outcome
    description: str
    provenance: dict[str, Provenance]
    coverage_fraction: float


class CandidateActuation(BaseModel):
    """A proposed shot's settings; keys as in QueryState.actuators or PCS feature names."""

    source_shot: int | None = None
    segment: SegName = "flat_top"
    actuators: dict[str, float]
    notes: str = ""


def to_summary(rec: ShotRecord, description: str = "") -> ShotSummary:
    ft = rec.segment("flat_top") or rec.segment("full")
    scalars = {**(ft.raw if ft else {}), **(ft.derived if ft else {})}
    n_present = sum(1 for s in rec.coverage.values() if s == "present")
    return ShotSummary(
        shot=rec.shot,
        shot_date=rec.shot_date,
        campaign=rec.campaign,
        mpid=rec.human.mpid,
        mp_title=rec.human.mp_title,
        run_id=rec.human.run_id,
        flat_top=scalars,
        labels=rec.labels,
        outcome=rec.outcome,
        description=description,
        provenance=rec.derived_provenance,
        coverage_fraction=(n_present / len(rec.coverage)) if rec.coverage else 0.0,
    )


# ------------------------------------------------------------------------------ evaluation
#
# The evaluation harness's own records (task I11). They are here with everything else rather
# than in `eval/` because a report that is written to JSON, read back by a later run and
# diffed against it is an interop format, and this module is where those live.


class EvalPrompt(BaseModel):
    """One row of `configs/ideate/evalsets/reference_shot_prompts.csv`.

    `expect_phenomena` is what a DIII-D physicist typing this sentence MEANS, written down when
    the prompt was authored and never afterwards. It is deliberately not "what `resolve` returns":
    the gap between the two is the measurement.
    """

    prompt_id: str
    category: str
    prompt: str
    expect_phenomena: list[str] = Field(default_factory=list)
    expect_constraints: dict[str, Range] = Field(default_factory=dict)
    expect_segment: SegName = "flat_top"
    hand_graded: bool = False
    notes: str = ""


class PromptOutcome(BaseModel):
    """What one prompt produced. Every field is a measurement, none is a verdict."""

    prompt_id: str
    category: str
    n_results: int
    candidates: int  # rows that survived the hard filter
    segment_rows: int  # rows of that segment in the split at all
    channels: dict[str, int] = Field(default_factory=dict)  # per channel, candidates returned
    resolved: list[str] = Field(default_factory=list)  # what `phenomena.resolve` made of the text
    expected_resolved: bool | None = None  # None when the prompt expects no phenomenon
    shots: list[int] = Field(default_factory=list)  # the top-n shots, in rank order
    run_ids: list[str] = Field(default_factory=list)  # distinct run days among them
    duplicate_shots: int = 0  # positions beyond the first held by an already-listed shot
    proxy_hit: bool | None = None  # hand-graded prompts only; see EvalReport.proxy
    error: str | None = None


class CategoryStats(BaseModel):
    category: str
    n_prompts: int
    coverage: float
    n_with_expectation: int
    resolution: float | None = None  # None when no prompt of the category expects a phenomenon
    # Averaged over the prompts that ANSWERED; None when none did. A prompt that returned
    # nothing has no top-10 to be diverse, and dividing by it would make the number smaller than
    # any top-10 ever showed.
    mean_run_diversity: float | None = None


class ProxyGrade(BaseModel):
    """The machine-checkable stand-in for the 20 hand grades.

    IT IS NOT THE HAND GRADE. It asks one narrow question -- does any shot in the top 5 carry
    evidence of every phenomenon the prompt expects (or satisfy every constraint it states) --
    and a prompt can pass it with five useless shots or fail it while returning the five a
    physicist would have picked. The human rubric is in the evalset's `notes` column and only a
    human closes it.
    """

    caveat: str = (
        "PROXY, NOT A HUMAN GRADE: expected phenomena present in the top-5 evidence "
        "(or expected constraints satisfied). It does not judge whether the shots are the "
        "right ones; the rubric in the evalset's `notes` column does, and only a human can."
    )
    n_prompts: int = 0
    n_checkable: int = 0  # the rest state no machine-checkable expectation
    n_hit: int = 0
    fraction: float | None = None


class EvalReport(BaseModel):
    """The evalset run against one database, on one side of the frozen split."""

    evalset: str
    evalset_sha256: str
    split: Literal["dev", "eval", "all"]
    split_shots: int
    n_prompts: int
    n_results_total: int
    coverage: float  # prompts with >= 1 result
    hard_filter_survival: float  # mean candidates / segment rows
    channel_participation: dict[str, float]  # per channel, fraction of prompts it contributed to
    resolution_overall: float | None
    categories: list[CategoryStats] = Field(default_factory=list)
    mean_run_diversity: float | None = None  # distinct run days per ANSWERED prompt's top-10
    duplicate_rate: float = 0.0  # duplicated positions / results returned
    proxy: ProxyGrade = Field(default_factory=ProxyGrade)
    outcomes: list[PromptOutcome] = Field(default_factory=list)
    db_dir: str = ""
    db_manifest_sha: str | None = None
    n_errors: int = 0
    generated_at: datetime | None = None


class LatencyRow(BaseModel):
    """One timed operation. `budget_s` is the plan's Appendix B number, or None where the plan
    sets none -- and a row with no budget gets no verdict rather than a free PASS.

    `run_medians` is one median per SEPARATE block of `repeats` samples. It exists because a
    single block on a shared login node measures the node: the I11 review re-ran this harness and
    got `search_no_text` at 126, 156 and 235 ms against a 200 ms budget, and `search_text`
    anywhere from 174 ms to 2.3 s. A row that straddles its budget across those blocks has no
    verdict to give, and says `load-dependent` instead of picking one.
    """

    name: str
    what: str
    repeats: int
    median_s: float
    p95_s: float
    budget_s: float | None = None
    run_medians: list[float] = Field(default_factory=list)

    @property
    def verdict(self) -> str:
        """PASS / FAIL only where EVERY run agrees; `load-dependent` where they do not."""
        if self.budget_s is None:
            return "n/a"
        runs = self.run_medians or [self.median_s]
        if all(m <= self.budget_s for m in runs):
            return "PASS"
        if all(m > self.budget_s for m in runs):
            return "FAIL"
        return "load-dependent"


class LatencyReport(BaseModel):
    db_dir: str
    n_shots: int
    n_segment_rows: int
    repeats: int
    runs: int = 1
    # What actually decides these numbers on a shared node, recorded so a later run is
    # comparable with this one rather than merely printed beside it.
    load_avg: tuple[float, float, float] | None = None
    cpu_count: int | None = None
    torch_threads: int | None = None
    warm: bool = True
    rows: list[LatencyRow] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    generated_at: datetime | None = None


class RecallReport(BaseModel):
    """`locate()`'s recall against a human-annotated sheet, `split=test` rows only.

    `n_labelled` counts only `y`/`n` rows: a `?` or an empty cell is an annotator who did not
    answer, and counting those as negatives would manufacture a recall number out of unread
    windows.
    """

    phenomenon: str
    sheet: str
    n_rows: int
    n_test_rows: int
    n_labelled: int
    n_positive: int
    n_negative: int
    n_rows_not_in_db: int = 0  # annotated windows whose shot this database does not hold
    true_positive: int = 0
    false_negative: int = 0
    false_positive: int = 0
    true_negative: int = 0
    recall: float | None = None
    specificity: float | None = None
    n_shots: int = 0
    caveats: list[str] = Field(default_factory=list)
    generated_at: datetime | None = None
