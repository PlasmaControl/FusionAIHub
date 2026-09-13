"""A shot's masks and events, summarised on a sliding window.

This is the bridge between what the detectors found and what a classifier
can be trained on. `masks.py` leaves a shot as packed boolean maps and a
per-column activity trace; `tracks.py`, `transients.py` and `heuristics.py`
leave it as event rows. A prior scores a WINDOW - "EHO at 2-20 kHz for more
than 100 ms while the ELMs are away" - so both have to be reduced onto one
grid of windows first, and `FEATURE_NAMES` is that reduction, frozen.

**0.34 s every 0.17 s.** The window is long enough to hold the shortest
phenomenon round 1 asks about (a 100 ms EHO, a 20-200 ms sawtooth train)
and short enough that a 5 s shot is ~30 windows, which is what makes a
2,000-shot candidate pool searchable. The stride is half the width, so
every instant is in exactly two windows and a mode near a window edge is
still whole in its neighbour.

**Diagnostics only - with one indirect bit, named rather than denied.**
Plan section 7. No feature here READS an actuator or an EFIT array: not
`pinj`, not `ech`, not the RMP coils, not `betan`, and `windows.py`
imports nothing from `features/`, so it cannot reach them even by
accident. What is here is what a diagnostic saw: the coherent mask, the
transient trace, the band power, the tracks, and the ELM/sawtooth/L-H/
pickup event families. The annotator picks candidate windows with actuator
conditions in mind, so a classifier trained on those conditions would be
scored against its own selection - the circularity the plan rules out.

**"What a diagnostic saw" is a POLICY, not an adjective.** An events file
holds every kind of claim about a shot side by side - a detector's, a
heuristic's, a forecast's, an operator's logbook line - and the row that
says which is `evidence_kind`, with `source` saying who. Selecting a
family by `phenomenon` alone therefore counts a shift-leader's
"updated ELM detector tuning" as an ELM at `t = 0` and a model's
`label_forecast` as a crash that happened: BOTH were reproduced on real
records by the iteration-0 critic, and both are what `DIAGNOSTIC_EVIDENCE`
and `FAMILY_SOURCES` now forbid. Every row reaching a feature must have
`evidence_kind` in `DIAGNOSTIC_EVIDENCE` (`detector` or `heuristic`) AND
a `source` that `FAMILY_SOURCES` allows for its own phenomenon;
`EventTable.from_frame` applies that filter once, before any clustering,
union or overlap, so nothing downstream can reach a row it excluded.
`text`, `forecast`, `human`, `database` and `model` rows are not dropped
from the FILE - they are what retrieval and the text channel are for -
they are simply not evidence about what a diagnostic showed, and this
module computes nothing else.

The exception is `lh_recent`. It counts `lh_transition` events, and
`heuristics.lh_transitions` is GATED on NBI power (`LH_MIN_PINJ_KW`): it
returns `[]` without beams and drops any candidate whose preceding window
is under the gate. So `lh_recent == 1` entails `pinj` above that gate -
one indirect bit, inherited from the detector rather than read here, on a
plasma-state label whose evidence (D-alpha, density) is diagnostic and
whose NBI test is a false-positive filter, not the signal. Anything else
built on heuristic events (a `qh_proxy`, say) inherits the same caveat and
must say so here.
`test_no_feature_carries_actuator_or_equilibrium_context` checks the
NAMES, which is necessary and not sufficient: this paragraph is the audit.

**One physical event, however many blocks wrote it.** `transients.py`
emits its rows per `(diag, channel, pass)`, so a shot whose ELM clock ran
on two D-alpha channels has every crash AND every ELM-free interval in the
table twice; summing them made `elm_free_frac` 2.0 and doubled
`elm_rate_hz`. Two policies, applied once when `EventTable` parses the
frame: INTERVAL phenomena are UNIONED per phenomenon before any overlap is
measured, and every fraction is clamped to [0, 1]; POINT phenomena (`elm`,
`sawtooth`, `lh_transition`) are CLUSTERED within `DEDUP_S` and counted as
clusters, so `elm_rate_hz`, `n_sawtooth`, `time_since_last_elm_s` and
`sawtooth_period_ms` are about physical events rather than rows. This
makes a duplicated table safe; it does not make two channels' disagreement
go away, so the pipeline should still prefer ONE reference D-alpha channel
for the ELM clock.

**Three diagnostics, twelve features each.** `mhr` (magnetics), `co2`
(interferometer) and `ece` (electron temperature) are the three that carry
round 1's channels. A diagnostic a shot does not have contributes twelve
zeros and a `cov_frac` of zero, which is distinguishable from "present and
quiet" precisely because `cov_frac` is a feature.

Everything here is numpy on stored arrays: no U-Net, no torch, no corpus
read. A window's features are a function of `<shot>_masks.npz` and
`<shot>_events.parquet` and of nothing else.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..ae.labels import N_BINS
from .heuristics import (
    LH_PHENOMENON,
    LH_SOURCE,
    SAWTOOTH_PHENOMENON,
    SAWTOOTH_SOURCE,
)
from .masks import (
    N_BANDS,
    freq_axis_khz,
    list_blocks,
    read_mask,
    unpack,
)
from .schema import read_events
from .tracks import PHENOMENON as TRACK_PHENOMENON
from .tracks import PICKUP_PHENOMENON
from .tracks import SOURCE as TRACK_SOURCE
from .transients import FREE_PHENOMENON
from .transients import FREE_SOURCE as ELM_FREE_SOURCE
from .transients import PHENOMENON as ELM_PHENOMENON
from .transients import SOURCE as ELM_SOURCE

#: The window, and how far it slides. Half-overlapping on purpose.
WINDOW_S = 0.34
STRIDE_S = 0.17

#: The three diagnostics the 36 per-diagnostic features are computed for,
#: in the order they appear in `FEATURE_NAMES`.
DIAGS = ("mhr", "co2", "ece")

#: The pass band each of `band_logpow_{lo,mid,hi}` averages over, in kHz.
#: `lo` and `mid` are read off the ZOOM pass (0.12207 kHz/bin), where the
#: mode structure below 60 kHz is resolved; `hi` off the WIDE pass, which is
#: the only one that reaches 250 kHz at all.
BAND_LO_KHZ = (2.0, 20.0)
BAND_MID_KHZ = (20.0, 60.0)
BAND_HI_KHZ = (100.0, 250.0)

# `_band_logpow` is 16 bands of 32 bins, and a band belongs to a pass band
# when its CENTRE bin does. On the 500 kHz record that resolves to:
#
# * zoom, 0.12207 kHz/bin, band `j` centred on 2.014 + 3.906`j` kHz:
#   `lo` = bands 0-4 (2.0-17.6 kHz), `mid` = bands 5-14 (21.6-56.7 kHz);
# * wide, 0.48828 kHz/bin, band `j` centred on 8.06 + 15.625`j` kHz:
#   `hi` = bands 6-15 (101.8-242.4 kHz).
#
# Computed from the block's own `fs_hz`/`decim` rather than hard-coded, so
# a channel digitised at another rate lands in the bands its frequencies
# actually are; the numbers above are what the production rate gives, and
# `test_the_band_indices_are_the_documented_ones` pins them.
#
# A band is the mean of 32 bins, so `lo` includes the 0-2 kHz corner of
# band 0 and `hi` stops at the top bin: these are broad power summaries,
# not filters, and nothing downstream reads a band edge as a cut-off.

#: The evidence kinds a row must carry before any of the 46 features may
#: be computed from it. A `forecast` is a model's estimate of what was
#: ABOUT to happen, a `text` row is an operator's mention of a phenomenon
#: (which may be a mention of the DETECTOR - "updated ELM detector tuning"
#: - rather than of an ELM), and `human`, `database` and `model` rows are
#: somebody else's claim, restated. None of them is what a diagnostic
#: showed on this shot, which is the only thing this module reduces.
DIAGNOSTIC_EVIDENCE: tuple[str, ...] = ("detector", "heuristic")

#: phenomenon -> the sources whose rows may become that family's features.
#: The evidence kind is necessary and not sufficient: a `heuristic` row
#: claiming `elm` from some source other than the ELM clock's own detector
#: is a claim this module has no calibration for, and a phenomenon absent
#: from this map reaches no feature at all.
#:
#: `elm` comes from `tokeye_transient` and NOT from `elm_clock`, which is
#: the derived source: the clock's own rows are the `elm_free` intervals it
#: computes FROM those ELMs (`transients.transients_to_events` writes both,
#: one source each, and `docs/LABELMAKER.md` tabulates them). Keeping the
#: two apart is what lets a consumer trust one and not the other.
FAMILY_SOURCES: dict[str, tuple[str, ...]] = {
    TRACK_PHENOMENON: (TRACK_SOURCE,),
    PICKUP_PHENOMENON: (TRACK_SOURCE,),
    ELM_PHENOMENON: (ELM_SOURCE,),
    FREE_PHENOMENON: (ELM_FREE_SOURCE,),
    SAWTOOTH_PHENOMENON: (SAWTOOTH_SOURCE,),
    LH_PHENOMENON: (LH_SOURCE,),
}

#: Point detections this close together are one physical event seen by
#: more than one block. 2 ms is well under the shortest interval round 1
#: cares about - an ELM train at 200 Hz is 5 ms apart, a sawtooth train
#: 20-200 ms - and well over the jitter between two channels' views of the
#: same crash, which is a column or two of a 0.256 ms grid.
DEDUP_S = 0.002

#: A window is valid when at least one diagnostic covers this much of it.
COVERAGE_MIN_FRAC = 0.5

#: `time_since_last_elm_s` is capped here: past a second the exact age of
#: the last ELM says nothing a classifier can use, and an uncapped value
#: would make an ELM-free shot's feature scale with the shot length.
ELM_AGE_CAP_S = 1.0

#: Sawteeth this far either side of the window centre are what
#: `sawtooth_period_ms` takes the median interval of - a crash train is a
#: property of the phase, not of the 0.34 s window, and two crashes 76 ms
#: apart may straddle the window edge.
SAWTOOTH_HALF_S = 0.5

#: `lh_recent` looks back this far from the window centre, `(centre - w,
#: centre]`: an L-H transition is a point event, and what matters to a
#: window is whether the plasma has just entered H-mode.
LH_RECENT_S = 0.5

#: The twelve features every diagnostic gets, in order, prefixed `{diag}_`.
PER_DIAG_SUFFIXES = (
    "coh_lit_frac_wide",
    "coh_lit_frac_zoom",
    "tra_col_act_mean_wide",
    "tra_col_act_max_wide",
    "band_logpow_lo",
    "band_logpow_mid",
    "band_logpow_hi",
    "n_tracks",
    "track_f_centroid_khz",
    "track_abs_chirp_max",
    "track_n_harmonics_max",
    "track_duration_max_ms",
)

#: The seven shot-wide features, plus one coverage fraction per diagnostic.
GLOBAL_NAMES = (
    "elm_rate_hz",
    "elm_free_frac",
    "time_since_last_elm_s",
    "n_sawtooth",
    "sawtooth_period_ms",
    "lh_recent",
    "pickup_flag",
)

#: FROZEN. A trained model card records these names and the GBDT runner
#: refuses a bundle whose feature names differ, so an insertion anywhere but
#: the end silently re-labels every column of every model already fitted.
#: Append only, and only with a retrain.
FEATURE_NAMES: tuple[str, ...] = (
    *(f"{diag}_{suffix}" for diag in DIAGS for suffix in PER_DIAG_SUFFIXES),
    *GLOBAL_NAMES,
    *(f"cov_frac_{diag}" for diag in DIAGS),
)

N_FEATURES = len(FEATURE_NAMES)


# ------------------------------------------------------------------- grid

def window_grid(
    t_cov0_s: float,
    t_cov1_s: float,
    *,
    width_s: float = WINDOW_S,
    stride_s: float = STRIDE_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(centres, starts, ends)` tiling `[t_cov0_s, t_cov1_s)`.

    Full windows every `stride_s`, then ONE truncated window carrying the
    remainder, so the end of a record is summarised rather than dropped -
    a shot's last 200 ms is where a disruption precursor lives. A coverage
    shorter than one width is that single truncated window; an empty or
    backwards coverage is no windows at all.

    A window's centre is the midpoint of what it actually spans, which for
    the truncated one is not `start + width/2`.
    """
    t0, t1 = float(t_cov0_s), float(t_cov1_s)
    width, stride = float(width_s), float(stride_s)
    if not (width > 0.0 and stride > 0.0):
        raise ValueError(f"width and stride must be positive: {width}, {stride}")
    empty = np.zeros(0, dtype=np.float64)
    if not t1 > t0:
        return empty, empty, empty
    span = t1 - t0
    # 1e-9 s absorbs the float error in a coverage that is an exact multiple
    # of the stride; it is a nanosecond against a 0.256 ms column.
    n_full = 1 + int((span - width) / stride + 1e-9) if span >= width else 0
    starts = t0 + stride * np.arange(n_full, dtype=np.float64)
    last_end = starts[-1] + width if n_full else t0
    if last_end < t1 - 1e-9:
        starts = np.append(starts, starts[-1] + stride if n_full else t0)
    ends = np.minimum(starts + width, t1)
    return 0.5 * (starts + ends), starts, ends


# ----------------------------------------------------------- stored blocks

@dataclass(frozen=True)
class WindowBlock:
    """One `(diag, channel, pass)` block, reduced to what a window needs.

    The coherent mask is 512 x T booleans and only ever enters a feature as
    "how many pixels of these columns are lit", so it is unpacked once and
    summed down to `coh_lit_per_col` here rather than kept: a shot's eleven
    channels on two passes are 180 MB of mask and 1.4 MB of column sums.
    """

    diag: str
    channel: int
    pass_name: str
    t_s: np.ndarray              # (T,) float64, column centre times
    coh_lit_per_col: np.ndarray  # (T,) int64, lit rows per column
    n_rows: int
    col_act: np.ndarray          # (T,) float32, the stored `_col_act`
    band_logpow: np.ndarray      # (N_BANDS, T) float32
    fs_hz: float
    decim: int

    def columns_in(self, t0: float, t1: float) -> slice:
        """The columns whose centres lie in `[t0, t1)`."""
        lo = int(np.searchsorted(self.t_s, t0, side="left"))
        hi = int(np.searchsorted(self.t_s, t1, side="left"))
        return slice(lo, hi)


def band_edges_khz(fs_hz: float, decim: int, n_bands: int = N_BANDS) -> np.ndarray:
    """`(n_bands, 2)` kHz: the first and last bin centre of each band."""
    freq = freq_axis_khz(float(fs_hz), int(decim)).reshape(n_bands, -1)
    return np.stack([freq[:, 0], freq[:, -1]], axis=1)


def band_indices(
    fs_hz: float, decim: int, lo_khz: float, hi_khz: float,
    n_bands: int = N_BANDS,
) -> np.ndarray:
    """Which of the stored bands fall inside `[lo_khz, hi_khz]`.

    By band CENTRE, so a band is in exactly one pass band and no band is
    split. See the module's band note for the production numbers.
    """
    centre = band_edges_khz(fs_hz, decim, n_bands).mean(axis=1)
    return np.flatnonzero((centre >= float(lo_khz)) & (centre <= float(hi_khz)))


def blocks_from_masks(masks_path) -> dict[str, list[WindowBlock]]:
    """`<shot>_masks.npz` -> its `DIAGS` blocks, grouped by diagnostic.

    A block on any other diagnostic is SKIPPED without being read: nothing
    in `FEATURE_NAMES` names it, and reading it means unpacking a 512-row
    boolean map - 8.4 MB for a 16,391-column wide pass - only to throw it
    away. A caller that wants to inspect a whole masks file has
    `masks.list_blocks`/`masks.read_mask` for that.
    """
    masks_path = Path(masks_path)
    if not masks_path.exists():
        raise FileNotFoundError(f"no masks file: {masks_path}")
    out: dict[str, list[WindowBlock]] = {}
    for prefix in list_blocks(masks_path):
        meta = read_mask(masks_path, f"{prefix}_meta")
        if str(meta["diag"]) not in DIAGS:
            continue
        n_cols = int(meta["n_cols"])
        packed = read_mask(masks_path, f"{prefix}_coh_packed")
        coh = unpack(packed, (N_BINS, n_cols))
        block = WindowBlock(
            diag=str(meta["diag"]),
            channel=int(meta["channel"]),
            pass_name=str(meta["pass_name"]),
            t_s=np.asarray(read_mask(masks_path, f"{prefix}_t_s"), dtype=np.float64),
            coh_lit_per_col=coh.sum(axis=0).astype(np.int64),
            n_rows=int(coh.shape[0]),
            col_act=np.asarray(
                read_mask(masks_path, f"{prefix}_col_act"), dtype=np.float32
            ),
            band_logpow=np.asarray(
                read_mask(masks_path, f"{prefix}_band_logpow"), dtype=np.float32
            ),
            fs_hz=float(meta["fs_hz"]),
            decim=int(meta["decim"]),
        )
        out.setdefault(block.diag, []).append(block)
    return out


def coverage_from_blocks(
    blocks: Mapping[str, Sequence[WindowBlock]],
) -> dict[str, tuple[float, float]]:
    """diag -> `(t0_s, t1_s)`, the span of its columns.

    The mask's own column grid is the authority on what was looked at: an
    event row's `t_cov` says the same thing, but only where that detector
    wrote a row, and a diagnostic that produced no events at all still has
    coverage. A diagnostic with no block is simply absent from the dict.
    """
    cov: dict[str, tuple[float, float]] = {}
    for diag, blks in blocks.items():
        spans = [(b.t_s[0], b.t_s[-1]) for b in blks if b.t_s.size]
        if spans:
            cov[diag] = (
                float(min(s[0] for s in spans)), float(max(s[1] for s in spans))
            )
    return cov


# ------------------------------------------------------------------ events

def _merge_intervals(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """`(n, 2)` DISJOINT intervals covering the union of `[t0, t1)`.

    Within one detector block the intervals of a phenomenon are already
    disjoint - `transients.elm_free_intervals` returns maximal ones - but
    the block is `(diag, channel, pass)`, so two channels write two full
    copies of the same quiet stretch. A fraction computed by summing those
    is not a fraction; the union is.

    Zero-length and non-finite rows are dropped: a point event carries
    `t1 == t0` and covers no time, and a NaN bound is no interval at all.
    Touching intervals (`a == previous end`) merge, because a union
    measured in seconds cannot tell them apart.
    """
    lo = np.asarray(t0, dtype=np.float64).ravel()
    hi = np.asarray(t1, dtype=np.float64).ravel()
    if lo.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    keep = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    lo, hi = lo[keep], hi[keep]
    order = np.argsort(lo, kind="stable")
    out: list[list[float]] = []
    for i in order:
        a, b = float(lo[i]), float(hi[i])
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def _cluster_times(t_s: np.ndarray, tol_s: float = DEDUP_S) -> np.ndarray:
    """Sorted times, ONE per cluster of detections within `tol_s`.

    A cluster is a run of times whose consecutive gaps are all `<= tol_s`,
    and its time is the mean of the run. The run is what makes N channels'
    views of one crash collapse to one event however they are staggered;
    it also means a genuine train faster than `1 / DEDUP_S` would chain,
    which is why `DEDUP_S` is an order of magnitude under the fastest
    interval round 1 asks about rather than a nice round tolerance.
    """
    t = np.sort(np.asarray(t_s, dtype=np.float64).ravel())
    t = t[np.isfinite(t)]
    if t.size == 0:
        return t
    starts = np.empty(t.size, dtype=bool)
    starts[0] = True
    np.greater(np.diff(t), float(tol_s), out=starts[1:])
    idx = np.flatnonzero(starts)
    sizes = np.diff(np.append(idx, t.size))
    return np.add.reduceat(t, idx) / sizes


def diagnostic_mask(events: pd.DataFrame) -> np.ndarray:
    """Which rows of `events` may be reduced into `FEATURE_NAMES`.

    Both halves of the policy, applied together: the row's `evidence_kind`
    must be in `DIAGNOSTIC_EVIDENCE`, and its `source` must be one
    `FAMILY_SOURCES` allows for its own `phenomenon`. A phenomenon this
    module computes no feature for - `hl_transition`, `nbi_on`, `qh` -
    fails the second test and is excluded, which is the same answer for a
    different reason and is why the mask is computed from the pair rather
    than from a list of families.

    Returns a plain boolean array rather than a pandas mask so that the
    caller can index either a frame or the arrays it came from, and so that
    an empty frame - the common case for a shot nothing has run on - costs
    nothing.
    """
    n = len(events)
    if n == 0:
        return np.zeros(0, dtype=bool)
    kind = events["evidence_kind"].to_numpy(dtype=object)
    source = events["source"].to_numpy(dtype=object)
    phenomenon = events["phenomenon"].to_numpy(dtype=object)
    allowed = DIAGNOSTIC_EVIDENCE
    return np.fromiter(
        (
            k in allowed and s in FAMILY_SOURCES.get(p, ())
            for k, s, p in zip(kind, source, phenomenon, strict=True)
        ),
        dtype=bool,
        count=n,
    )


def diagnostic_rows(events: pd.DataFrame) -> pd.DataFrame:
    """`events` narrowed to what a diagnostic showed on this shot.

    The one gate between an events file and the 46 features. A consumer
    that wants the text or the forecasts has the whole frame and the
    `evidence_kind` column; a consumer computing a diagnostic feature goes
    through here, so that "which rows count" is one decision in one place
    rather than a `phenomenon` test repeated per family.
    """
    return events.loc[diagnostic_mask(events)]


def _attr(raw: str, key: str) -> float:
    """One numeric field of a stored `attrs` JSON string, or NaN."""
    try:
        value = json.loads(raw).get(key)
    except (TypeError, ValueError):
        return float("nan")
    return float("nan") if value is None else float(value)


@dataclass(frozen=True)
class EventTable:
    """One shot's event rows, split into the families a window asks about.

    Parsed once and passed to every window: `attrs` is a JSON string per
    row, and re-parsing a shot's tracks for each of thirty windows is the
    one thing in here that would actually cost something.

    This is also where the per-block duplicates go: `elm_free` reaches
    `elm_free` UNIONED (`_merge_intervals`) and the three point families
    reach their arrays CLUSTERED (`_cluster_times`), so every window sees
    physical events and the window code never has to know that the rows
    came from more than one `(diag, channel, pass)`.
    """

    track_diag: np.ndarray
    track_t0: np.ndarray
    track_t1: np.ndarray
    track_conf: np.ndarray
    track_f_khz: np.ndarray
    track_chirp: np.ndarray
    track_harmonics: np.ndarray
    track_duration_ms: np.ndarray
    track_pickup: np.ndarray
    pickup_t0: np.ndarray
    pickup_t1: np.ndarray
    elm_s: np.ndarray
    elm_free: np.ndarray
    sawtooth_s: np.ndarray
    lh_s: np.ndarray

    @classmethod
    def of(cls, events) -> EventTable:
        """Accept either a parsed table or the frame it came from."""
        return events if isinstance(events, cls) else cls.from_frame(events)

    @classmethod
    def from_frame(cls, events: pd.DataFrame) -> EventTable:
        # ONE filter, here, before anything is clustered, unioned or
        # counted: every array below is built off `df`, so a row the
        # evidence policy excludes is unreachable from every feature
        # rather than excluded family by family.
        df = diagnostic_rows(events)
        tracks = df[df["source"] == TRACK_SOURCE]
        attrs = tracks["attrs"].to_numpy()

        def field(key: str) -> np.ndarray:
            return np.array([_attr(a, key) for a in attrs], dtype=np.float64)

        pickup = df[df["phenomenon"] == PICKUP_PHENOMENON]
        free = df.loc[
            df["phenomenon"] == FREE_PHENOMENON, ["t0_s", "t1_s"]
        ].to_numpy(dtype=np.float64).reshape(-1, 2)
        points = {
            key: _cluster_times(
                df.loc[df["phenomenon"] == key, "t0_s"].to_numpy(dtype=np.float64)
            )
            for key in (ELM_PHENOMENON, SAWTOOTH_PHENOMENON, LH_PHENOMENON)
        }
        return cls(
            track_diag=tracks["diag"].to_numpy(dtype=object),
            track_t0=tracks["t0_s"].to_numpy(dtype=np.float64),
            track_t1=tracks["t1_s"].to_numpy(dtype=np.float64),
            track_conf=tracks["confidence"].to_numpy(dtype=np.float64),
            track_f_khz=field("f_centroid_khz"),
            track_chirp=field("chirp_khz_per_ms"),
            track_harmonics=field("n_harmonics"),
            track_duration_ms=field("duration_ms"),
            # `tracks.as_attrs` writes every `Track` field, `pickup` among
            # them; NaN (the key absent) is not a pickup row.
            track_pickup=field("pickup"),
            pickup_t0=pickup["t0_s"].to_numpy(dtype=np.float64),
            pickup_t1=pickup["t1_s"].to_numpy(dtype=np.float64),
            elm_s=points[ELM_PHENOMENON],
            elm_free=_merge_intervals(free[:, 0], free[:, 1]),
            sawtooth_s=points[SAWTOOTH_PHENOMENON],
            lh_s=points[LH_PHENOMENON],
        )


def _overlaps(t0: np.ndarray, t1: np.ndarray, start: float, end: float) -> np.ndarray:
    """Rows whose extent meets the half-open window `[start, end)`.

    Both are half-open, as everything in this package is, so an interval
    that ENDS at `start` does not overlap - it is over by the time the
    window begins - and one that starts at `end` belongs to the next
    window. A point event (`t1 == t0`) would fail that test at its own
    start, so it is asked the containment question instead: `start <= t0 <
    end`. Every instant is therefore in exactly one window of a
    non-overlapping tiling, and in exactly two of this one.
    """
    point = t1 <= t0
    return np.where(
        point, (t0 >= start) & (t0 < end), (t0 < end) & (t1 > start)
    )


def _overlap_s(t0: np.ndarray, t1: np.ndarray, start: float, end: float) -> float:
    """Seconds the intervals `[t0, t1)` spend inside `[start, end)`.

    This SUMS, which is the union only when the intervals are disjoint -
    so every interval family reaches it through `_merge_intervals`, once,
    when `EventTable` parses the frame. Within one detector block they are
    disjoint already (`elm_free_intervals` returns maximal intervals);
    across two channels' blocks they are not, which is what made
    `elm_free_frac` 2.0.
    """
    if t0.size == 0:
        return 0.0
    lo = np.maximum(t0, start)
    hi = np.minimum(t1, end)
    return float(np.maximum(hi - lo, 0.0).sum())


def _max_or_zero(values: np.ndarray) -> float:
    """The largest finite value, or 0 when there is none."""
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else 0.0


# ---------------------------------------------------------------- features

def _cov_frac(
    cov: Mapping[str, tuple[float, float]], diag: str, start: float, end: float
) -> float:
    """How much of `[start, end)` the diagnostic was looking at, in [0, 1].

    Zero for a diagnostic the shot does not have at all, which is what makes
    "nobody looked" distinguishable from "present and quiet" - the whole
    reason the three `cov_frac_*` are features rather than a dropped row.
    """
    span = cov.get(diag)
    width = end - start
    if span is None or width <= 0.0 or not np.isfinite(span).all():
        return 0.0
    inside = min(float(span[1]), end) - max(float(span[0]), start)
    return max(0.0, inside) / width


def _diag_features(
    diag: str,
    blocks: Mapping[str, Sequence[WindowBlock]],
    table: EventTable,
    cov: Mapping[str, tuple[float, float]],
    start: float,
    end: float,
) -> list[float]:
    """The twelve `{diag}_*` values, in `PER_DIAG_SUFFIXES` order."""
    if _cov_frac(cov, diag, start, end) <= 0.0:
        # The diagnostic is absent, or looked nowhere near this window.
        # Twelve zeros, and `cov_frac_{diag}` = 0 beside them says which.
        return [0.0] * len(PER_DIAG_SUFFIXES)

    blks = list(blocks.get(diag, ()) or ())
    by_pass = {
        name: [b for b in blks if b.pass_name == name] for name in ("wide", "zoom")
    }

    def lit_fraction(pass_name: str) -> float:
        # Recomputed from the unpacked coherent mask, not from the stored
        # `_row_lit`: that summary is over the WHOLE record, and a window
        # asks what was lit here. Pooled over the diagnostic's channels -
        # lit pixels over the total pixels of the columns in the window -
        # so a channel whose record stops early contributes what it has
        # rather than dragging the fraction down with absent columns.
        lit = 0
        total = 0
        for b in by_pass[pass_name]:
            cols = b.columns_in(start, end)
            n = cols.stop - cols.start
            if n <= 0:
                continue
            lit += int(b.coh_lit_per_col[cols].sum())
            total += n * b.n_rows
        return lit / total if total else 0.0

    # The transient trace is the wide pass' business: an ELM is broadband,
    # and the zoom pass throws away everything above 62 kHz.
    act = [b.col_act[b.columns_in(start, end)] for b in by_pass["wide"]]
    act_all = np.concatenate([a for a in act if a.size] or [
        np.zeros(0, dtype=np.float32)
    ])

    def band_power(pass_name: str, band: tuple[float, float]) -> float:
        # The mean over the pass band's bands and the window's columns, then
        # the mean over the diagnostic's channels - so two channels of `mhr`
        # weigh the same however many columns each of them has.
        means: list[float] = []
        for b in by_pass[pass_name]:
            cols = b.columns_in(start, end)
            idx = band_indices(b.fs_hz, b.decim, band[0], band[1])
            if cols.stop - cols.start <= 0 or idx.size == 0:
                continue
            means.append(float(b.band_logpow[idx, cols].mean()))
        return float(np.mean(means)) if means else 0.0

    # A pickup row IS a `tokeye_track` row (`tracks.tracks_to_events` only
    # changes its `phenomenon`), and a receiver line spans the whole record
    # at a fixed low frequency with high confidence: counted as a track it
    # would pin `track_f_centroid_khz` near the receiver in EVERY window of
    # an affected shot. It is excluded from all five track statistics here;
    # `pickup_flag` is where a receiver line is reported, and the test that
    # pins this is `test_a_pickup_row_is_not_counted_among_the_tracks`.
    on = (
        _overlaps(table.track_t0, table.track_t1, start, end)
        & (table.track_diag == diag)
        & ~(table.track_pickup > 0.0)
    )
    f_khz = table.track_f_khz[on]
    conf = table.track_conf[on]
    centroid = 0.0
    if f_khz.size:
        good = np.isfinite(f_khz)
        weight = np.where(np.isfinite(conf) & (conf > 0.0), conf, 0.0)[good]
        values = f_khz[good]
        if values.size:
            centroid = float(
                np.average(values, weights=weight) if weight.sum() > 0.0
                else values.mean()
            )

    return [
        lit_fraction("wide"),
        lit_fraction("zoom"),
        float(act_all.mean()) if act_all.size else 0.0,
        float(act_all.max()) if act_all.size else 0.0,
        band_power("zoom", BAND_LO_KHZ),
        band_power("zoom", BAND_MID_KHZ),
        band_power("wide", BAND_HI_KHZ),
        float(int(on.sum())),
        centroid,
        _max_or_zero(np.abs(table.track_chirp[on])),
        _max_or_zero(table.track_harmonics[on]),
        _max_or_zero(table.track_duration_ms[on]),
    ]


def _global_features(table: EventTable, start: float, end: float) -> list[float]:
    """The seven shot-wide values, in `GLOBAL_NAMES` order."""
    width = end - start
    centre = 0.5 * (start + end)

    elm = table.elm_s
    n_elm = int(((elm >= start) & (elm < end)).sum())
    before = elm[elm <= centre]
    age = ELM_AGE_CAP_S if before.size == 0 else min(
        ELM_AGE_CAP_S, float(centre - before[-1])
    )

    saw = table.sawtooth_s
    near = saw[
        (saw >= centre - SAWTOOTH_HALF_S) & (saw <= centre + SAWTOOTH_HALF_S)
    ]
    period_ms = float(np.median(np.diff(near)) * 1e3) if near.size >= 2 else 0.0

    # `lh_recent` is the one feature carrying an indirect actuator bit.
    # Nothing is read from `pinj` here - but `heuristics.lh_transitions`
    # returns `[]` without beams and drops every candidate whose preceding
    # window is under `LH_MIN_PINJ_KW`, so a 1 in this column entails NBI
    # above that gate. Inherited from the detector, stated in the module
    # docstring, and tolerated because the transition's own evidence
    # (D-alpha, density) is diagnostic and the NBI test is a
    # false-positive filter rather than the signal.
    lh = table.lh_s
    recent = bool(((lh > centre - LH_RECENT_S) & (lh <= centre)).any())

    pickup = bool(
        _overlaps(table.pickup_t0, table.pickup_t1, start, end).any()
    )
    # `elm_free` is already the UNION of what every block wrote, so the
    # clamp is belt and braces - and it is here because a feature named as
    # a fraction leaving [0, 1] on some shots and not others is exactly the
    # kind of defect a GBDT absorbs silently.
    free_s = _overlap_s(table.elm_free[:, 0], table.elm_free[:, 1], start, end)
    return [
        n_elm / width if width > 0.0 else 0.0,
        min(1.0, free_s / width) if width > 0.0 else 0.0,
        age,
        float(int(((saw >= start) & (saw < end)).sum())),
        period_ms,
        float(recent),
        float(pickup),
    ]


def window_features(
    window,
    *,
    blocks: Mapping[str, Sequence[WindowBlock]],
    events,
    cov: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    """One window -> the 46 features, in `FEATURE_NAMES` order.

    `window` is `(start_s, end_s)`, half-open; every feature that needs an
    instant (`time_since_last_elm_s`, `sawtooth_period_ms`, `lh_recent`)
    uses the MIDPOINT of what the window actually spans, which for the
    truncated last window of a shot is not `start + WINDOW_S / 2`.

    `events` is either the shot's frame or an `EventTable` already parsed
    from one; `blocks` and `cov` are `blocks_from_masks` and
    `coverage_from_blocks`, or a hand-built equivalent.
    """
    start, end = float(window[0]), float(window[1])
    table = EventTable.of(events)
    values: list[float] = []
    for diag in DIAGS:
        values.extend(_diag_features(diag, blocks, table, cov, start, end))
    values.extend(_global_features(table, start, end))
    values.extend(_cov_frac(cov, diag, start, end) for diag in DIAGS)
    out = np.asarray(values, dtype=np.float32)
    if out.size != N_FEATURES:
        raise AssertionError(f"{out.size} features, expected {N_FEATURES}")
    return out


def shot_window_features(shot: int, paths):
    """One shot -> `(centres (T,), X (46, T), valid (T,))`.

    Reads `paths.masks_file(shot)` and `paths.events_file(shot)` and nothing
    else. Either one missing raises `FileNotFoundError` NAMING the file: a
    shot the mask job has not reached is the common case, and the resolver
    turns this into a recorded per-shot miss rather than a failed run.

    `valid[t]` is "at least one of mhr/co2/ece covered at least
    `COVERAGE_MIN_FRAC` of this window". The features of an invalid window
    are still computed - they are zeros and whatever the events say - but
    nothing should be trained or scored on them, which is why the mask is
    returned beside them rather than folded into the values here.
    """
    masks_path = Path(paths.masks_file(int(shot)))
    events_path = Path(paths.events_file(int(shot)))
    if not masks_path.exists():
        raise FileNotFoundError(f"no masks file for shot {shot}: {masks_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"no events file for shot {shot}: {events_path}")
    blocks = blocks_from_masks(masks_path)
    cov = coverage_from_blocks(blocks)
    table = EventTable.from_frame(read_events(events_path))
    spans = [cov[d] for d in DIAGS if d in cov]
    if not spans:
        empty = np.zeros(0, dtype=np.float64)
        return empty, np.zeros((N_FEATURES, 0), dtype=np.float32), empty.astype(bool)
    centres, starts, ends = window_grid(
        min(s[0] for s in spans), max(s[1] for s in spans)
    )
    x = np.zeros((N_FEATURES, centres.size), dtype=np.float32)
    for i, (t0, t1) in enumerate(zip(starts, ends, strict=True)):
        x[:, i] = window_features(
            (t0, t1), blocks=blocks, events=table, cov=cov
        )
    # The threshold is applied to the float64 fractions, not to the stored
    # float32 column: a coverage a few nanoseconds under a half rounds UP
    # to exactly 0.5 in float32 and would call an uncovered window valid.
    cov_frac = np.array(
        [
            [_cov_frac(cov, d, t0, t1)
             for t0, t1 in zip(starts, ends, strict=True)]
            for d in DIAGS
        ],
        dtype=np.float64,
    ).reshape(len(DIAGS), centres.size)
    valid = cov_frac.max(axis=0, initial=0.0) >= COVERAGE_MIN_FRAC
    return centres, x, np.asarray(valid, dtype=bool)
