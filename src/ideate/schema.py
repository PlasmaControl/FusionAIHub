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
    raw_sources: dict[str, Literal["staged", "fetched"]] = Field(default_factory=dict)
    human: HumanTier = Field(default_factory=HumanTier)
    labels: Labels = Field(default_factory=Labels)
    outcome: Outcome = Field(default_factory=Outcome)
    coverage: dict[str, Status] = Field(default_factory=dict)
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
    y: float  # in the actuator's units (configs/actuators.yaml)


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
    polished: bool = False
    blurb: str | None = None  # the offline two-sentence summary from shots.parquet
    explanation: Explanation = Field(default_factory=Explanation)
    flags: list[Flag] = Field(default_factory=list)
    labels: Labels = Field(default_factory=Labels)
    outcome: Outcome = Field(default_factory=Outcome)
    run_id: str | None = None


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
