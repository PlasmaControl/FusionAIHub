"""A coherent mask -> boxes in time and frequency, with descriptors.

`masks.py` says WHERE the network saw coherent activity. This says what a
lit region IS, in the only vocabulary that survives a shot without a human
in it: a box in time and frequency, a chirp rate, a bandwidth, a duty cycle,
a list of harmonic partners. Nothing here names a mode. A `coherent_mode`
event is a class-agnostic claim - "something coherent was here, at this
frequency, chirping this fast" - and the classifier that turns a list of
those into "EHO" or "fishbone" is a later, separate, supervised step. Naming
a mode from an unlabelled mask would be inventing a label, and the point of
the events table is that every row says who claims it.

**Components, then merge, because a mode blinks.** `scipy.ndimage.label`
with 8-connectivity turns the mask into blobs; a real mode is many of them,
because the network's probability dips under `PROB_THRESHOLD` wherever an
ELM or a sawtooth crosses it. `merge` puts blobs back together when they
overlap in frequency and are close in time. The geometry is ported from
`raddet_map_probe.py::merge_boxes_time` in the TokEye probe - the box
arithmetic that was measured against radar detections there - and kept
twice: `_merge_literal` is that function, transliterated onto half-open
extents and used by nothing but the test that pins it, and `merge` is the
production path, which sorts by start column and breaks out of its scan the
moment the next candidate is too far away to reach. The thresholds are OURS,
not raddet's: `MIN_AREA` 200 pixels, `MERGE_GAP_COLS` 40 columns (41 ms of
the zoom pass, longer than the ELM cycle a mode blinks with), `FREQ_OVERLAP`
0.5.

**`pickup` is a descriptor, not a filter.** A row lit for 80% of the record
is a receiver line - `masks.py` measured them on the pilot shot, co2 at
1.7-2.3 kHz and ece at 8.4-9.0 kHz - and it comes out of here as an event
with `phenomenon="pickup"` rather than being deleted. A consumer that wants
plasma drops them in one predicate; a consumer looking for the channel whose
pickup moved has them.

**The descriptors are the contract.** The annotation priors and the window
features read `Track` by field name, so a field's DEFINITION is as pinned as
its name: `f_centroid_khz` is probability-weighted over the track's pixels,
the chirp is a least-squares fit of the per-column centroid against time in
milliseconds, `bandwidth_khz` is the MEDIAN per-column extent (a mean would
be set by the one column an ELM smeared), `duty` is the fraction of the
track's columns that are lit at all.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from scipy import ndimage

from ..ae.labels import N_BINS
from .masks import freq_axis_khz, read_mask, unpack
from .schema import Event

#: Pixels a blob needs before it is a component at all. Below this it is
#: the salt the threshold leaves behind, and merging it can only bridge two
#: tracks that have nothing to do with each other.
MIN_AREA = 200
#: Columns two blobs of one mode may be apart. 40 columns is 41 ms of the
#: zoom pass and 10 ms of the wide one - longer than the ELM cycle that
#: makes a mode blink, shorter than the pause between two bursts.
MERGE_GAP_COLS = 40
#: Fraction of the SMALLER frequency extent two blobs must share.
FREQ_OVERLAP = 0.5
#: Percentile of the lit probabilities an event's confidence is. Not the
#: maximum: one hot pixel should not make a faint track certain.
CONF_PCT = 95
#: A row lit for this fraction of the record is receiver pickup.
PICKUP_ROW_FRACTION = 0.8
#: Multiples `harmonics` looks for, and the relative tolerance it allows.
#: Fifth harmonics of a 5 kHz EHO are still inside the zoom band; a sixth
#: is rarely above the noise.
HARMONIC_ORDERS = (2, 3, 4, 5)
HARMONIC_TOL = 0.06
#: Seconds two tracks must coexist before one can be the other's harmonic.
MIN_OVERLAP_S = 0.02
#: Box IoU two channels' tracks need before they are called the same event.
IOU_MIN = 0.2

#: Who claims these events, and what they are claimed to be.
SOURCE = "tokeye_track"
PHENOMENON = "coherent_mode"
PICKUP_PHENOMENON = "pickup"


@dataclass(eq=False)
class Component:
    """One 8-connected blob of the coherent mask.

    Extents are half-open, like every slice in this package: rows
    `[row0, row1)`, columns `[col0, col1)`. `rows`/`cols` are the blob's own
    pixels, kept rather than recomputed because `descriptors` weights them
    by probability and by power and a bounding box cannot say where they are.

    `eq=False` on purpose: a component holds arrays, so a generated
    `__eq__` would compare them elementwise and raise on the truth value of
    the result. Identity is also what `merge`'s test wants - the fast path
    must return THE components it was given, in the same groups.
    """

    row0: int
    row1: int
    col0: int
    col1: int
    n_pix: int
    rows: np.ndarray
    cols: np.ndarray
    label: int


@dataclass(frozen=True)
class Track:
    """One merged region of the coherent mask, described.

    Times are seconds of the shot's clock and are the CENTRES of the first
    and last lit column, so a one-column track is a point event. Frequencies
    are the centres of the first and last lit row. Everything is a python
    scalar: a Track goes into an event's `attrs` as JSON, and a numpy float
    does not serialise.
    """

    t0_s: float
    t1_s: float
    f0_khz: float
    f1_khz: float
    f_centroid_khz: float
    chirp_khz_per_ms: float
    chirp_r2: float
    bandwidth_khz: float
    duration_ms: float
    duty: float
    mean_prob: float
    conf: float
    n_pix: int
    n_components: int
    row_lit_fraction: float
    pickup: bool


# -------------------------------------------------------------- components

def components(coh_mask, *, min_area: int = MIN_AREA) -> list[Component]:
    """`(512, T)` coherent mask -> its blobs, smallest ones dropped.

    8-connectivity, because a mode that drifts a row per column is one mode.
    `min_area` is applied HERE, before anything is merged: a blob too small
    to be a track must not be able to bridge two that are.

    Returned in `scipy.ndimage.label`'s order, which is the raster order of
    each blob's first pixel.
    """
    mask = np.asarray(coh_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != N_BINS:
        raise ValueError(f"expected a ({N_BINS}, T) mask, got {mask.shape}")
    labels = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))[0]
    out: list[Component] = []
    for label, window in enumerate(ndimage.find_objects(labels), start=1):
        if window is None:
            continue
        blob = labels[window] == label
        n_pix = int(blob.sum())
        if n_pix < int(min_area):
            continue
        rows, cols = np.nonzero(blob)
        out.append(
            Component(
                row0=int(window[0].start), row1=int(window[0].stop),
                col0=int(window[1].start), col1=int(window[1].stop),
                n_pix=n_pix,
                rows=(rows + window[0].start).astype(np.int64),
                cols=(cols + window[1].start).astype(np.int64),
                label=label,
            )
        )
    return out


# ------------------------------------------------------------------- merge

def _sorted(comps: Sequence[Component]) -> list[Component]:
    """Components by start column, then by the rest of the box.

    Ascending `col0` is what makes the fast path's early break legitimate,
    and the remaining keys only make the order total, so that two runs on
    one mask group the same way.
    """
    return sorted(comps, key=lambda c: (c.col0, c.row0, c.col1, c.row1, c.label))


def _merge_literal(
    comps: Sequence[Component],
    *,
    max_gap: int = MERGE_GAP_COLS,
    freq_overlap: float = FREQ_OVERLAP,
) -> list[list[Component]]:
    """The reference merge, transliterated. Used by the test that pins it.

    Ported from `raddet_map_probe.py::merge_boxes_time(comps, max_gap,
    freq_overlap)` in the TokEye probe. Components are put in `_sorted`
    order first, exactly as `merge` does - the predicate is applied against
    a growing box, so the seed order is part of the answer and the two
    would not otherwise be comparable. Then, repeatedly, every box that is
    still free is taken as a seed and every other free box that shares at least
    `freq_overlap` of the SMALLER frequency extent and is within `max_gap`
    columns is absorbed into it, until a whole pass absorbs nothing. The
    predicate is applied against the GROWING box, not pairwise, so the
    result is not the transitive closure of the pairwise relation and the
    order matters - which is why this is kept, exactly, beside the fast one.

    The reference's boxes are inclusive (`ov = min - max + 1`); a Component
    is half-open, so every `+ 1` in it becomes nothing here.
    """
    items = [[c.row0, c.row1, c.col0, c.col1, [c]] for c in _sorted(comps)]
    changed = True
    while changed:
        changed = False
        out: list[list] = []
        used = [False] * len(items)
        for i in range(len(items)):
            if used[i]:
                continue
            used[i] = True
            y0, y1, x0, x1, members = items[i]
            members = list(members)
            merged_any = True
            while merged_any:
                merged_any = False
                for j in range(len(items)):
                    if used[j]:
                        continue
                    b0, b1, a0, a1, other = items[j]
                    ov = min(y1, b1) - max(y0, b0)
                    m = min(y1 - y0, b1 - b0)
                    if m <= 0 or ov / m < freq_overlap:
                        continue
                    if max(a0 - x1, x0 - a1) > max_gap:
                        continue
                    x0, x1 = min(x0, a0), max(x1, a1)
                    y0, y1 = min(y0, b0), max(y1, b1)
                    members.extend(other)
                    used[j] = True
                    merged_any = True
                    changed = True
            out.append([y0, y1, x0, x1, members])
        items = out
    return [item[4] for item in items]


def merge(
    comps: Sequence[Component],
    *,
    max_gap: int = MERGE_GAP_COLS,
    freq_overlap: float = FREQ_OVERLAP,
) -> list[list[Component]]:
    """Components -> tracks: one list of components per track.

    `_merge_literal`'s partition, reached without its scan. Components are
    sorted by start column, and the inner scan stops at the first candidate
    that starts more than `max_gap` columns after the running box ends -
    every later one starts later still. A mask of 16,391 columns can hold
    hundreds of blobs, of which only a handful are ever within reach of any
    one of them.

    Because a seed is always the first free box in ascending `col0`, and
    everything it absorbs starts at or after it, a track's start column is
    its seed's: the groups come back in ascending start time, and stay
    sorted between passes, which is what keeps the break legitimate.
    """
    items = [[c.row0, c.row1, c.col0, c.col1, [c]] for c in _sorted(comps)]
    changed = True
    while changed:
        changed = False
        out: list[list] = []
        used = [False] * len(items)
        for i in range(len(items)):
            if used[i]:
                continue
            used[i] = True
            y0, y1, x0, x1, members = items[i]
            members = list(members)
            merged_any = True
            while merged_any:
                merged_any = False
                for j in range(len(items)):
                    b0, b1, a0, a1, other = items[j]
                    if a0 - x1 > max_gap:
                        break                      # and so does every later j
                    if used[j]:
                        continue
                    ov = min(y1, b1) - max(y0, b0)
                    m = min(y1 - y0, b1 - b0)
                    if m <= 0 or ov / m < freq_overlap:
                        continue
                    if x0 - a1 > max_gap:
                        continue
                    x0, x1 = min(x0, a0), max(x1, a1)
                    y0, y1 = min(y0, b0), max(y1, b1)
                    members.extend(other)
                    used[j] = True
                    merged_any = True
                    changed = True
            out.append([y0, y1, x0, x1, members])
        items = out
    return [item[4] for item in items]


# ------------------------------------------------------------- descriptors

def _chirp(t_ms: np.ndarray, f_khz: np.ndarray) -> tuple[float, float]:
    """Least-squares slope of frequency against time, and its R^2.

    `(0.0, 0.0)` for a track too short to have a slope, and an R^2 of zero
    for one too short to have a fit (fewer than three columns: two points
    lie on their own line exactly, and calling that a perfect fit would let
    a two-column blob outrank a real chirp). A centroid that does not move
    also scores zero - there is no variance for a line to explain, and
    "flat" is what `chirp_khz_per_ms == 0` already says.
    """
    if t_ms.size < 2:
        return 0.0, 0.0
    dt = t_ms - t_ms.mean()
    df = f_khz - f_khz.mean()
    sxx = float(dt @ dt)
    if sxx <= 0.0:
        return 0.0, 0.0
    slope = float(dt @ df) / sxx
    ss_tot = float(df @ df)
    if t_ms.size < 3 or ss_tot <= 0.0:
        return slope, 0.0
    resid = df - slope * dt
    return slope, max(0.0, 1.0 - float(resid @ resid) / ss_tot)


def descriptors(
    group: Sequence[Component],
    *,
    prob,
    raw_logpow,
    freq_khz,
    t_s,
) -> Track:
    """One track's components -> its `Track`.

    `prob` is the coherent probability map the mask came from, `raw_logpow`
    the pre-standardisation log-power under it (`masks.unstandardise`, or
    `masks.band_logpow`'s bands stretched back to 512 rows when all that
    survives is the stored block). The two weight different things, on purpose:

    * `f_centroid_khz` is weighted by PROBABILITY - it answers "where is
      this detection", and a detection is what the network says it is;
    * the per-column centroid the chirp is fitted to is weighted by POWER -
      it answers "where inside the lit band is the mode", and the network
      lights a band whole while the power picks out the ridge in it.

    Columns with no power to weight by fall back to the geometric centre of
    what is lit, so a track always has a chirp rather than a NaN. Every
    field is finite: a Track is stored as an event's `attrs`, and JSON has
    no NaN.
    """
    group = list(group)
    if not group:
        raise ValueError("a track needs at least one component")
    prob = np.asarray(prob, dtype=np.float32)
    power = np.asarray(raw_logpow, dtype=np.float32)
    freq = np.asarray(freq_khz, dtype=np.float64)
    times = np.asarray(t_s, dtype=np.float64)
    if prob.ndim != 2 or prob.shape[0] != N_BINS:
        raise ValueError(f"expected a ({N_BINS}, T) probability map, got {prob.shape}")
    if power.shape != prob.shape:
        raise ValueError(f"raw_logpow is {power.shape}, prob is {prob.shape}")
    if freq.shape != (prob.shape[0],) or times.shape != (prob.shape[1],):
        raise ValueError(
            f"axes {freq.shape}, {times.shape} do not describe a {prob.shape} map"
        )

    rows = np.concatenate([c.rows for c in group])
    cols = np.concatenate([c.cols for c in group])
    row0 = min(c.row0 for c in group)
    row1 = max(c.row1 for c in group)
    col0 = min(c.col0 for c in group)
    col1 = max(c.col1 for c in group)
    span = col1 - col0
    f_pix = freq[rows]

    p = prob[rows, cols].astype(np.float64)
    p_sum = float(p.sum())
    centroid = float(p @ f_pix / p_sum) if p_sum > 0.0 else float(f_pix.mean())

    # Per column: the power-weighted centroid, the extent, and whether it is
    # lit at all. `bincount` rather than a loop - a track can be sixteen
    # thousand columns wide.
    idx = cols - col0
    w = np.maximum(power[rows, cols].astype(np.float64), 0.0)
    n_lit = np.bincount(idx, minlength=span)
    w_sum = np.bincount(idx, weights=w, minlength=span)
    wf_sum = np.bincount(idx, weights=w * f_pix, minlength=span)
    f_sum = np.bincount(idx, weights=f_pix, minlength=span)
    lit = n_lit > 0
    weighted = w_sum > 0.0
    per_col = np.zeros(span, dtype=np.float64)
    per_col[weighted] = wf_sum[weighted] / w_sum[weighted]
    flat = lit & ~weighted
    per_col[flat] = f_sum[flat] / n_lit[flat]

    # The frequency axis is monotone, so the extreme rows are the extreme
    # frequencies and the extent can be taken on row indices.
    hi = np.full(span, -1, dtype=np.int64)
    lo = np.full(span, N_BINS, dtype=np.int64)
    np.maximum.at(hi, idx, rows)
    np.minimum.at(lo, idx, rows)
    bandwidth = float(np.median(freq[hi[lit]] - freq[lo[lit]]))

    t0_s = float(times[col0])
    t1_s = float(times[col1 - 1])
    chirp, r2 = _chirp((times[col0:col1][lit] - t0_s) * 1e3, per_col[lit])

    per_row = np.bincount(rows - row0, minlength=row1 - row0)
    row_lit_fraction = float(per_row.max() / times.size)

    return Track(
        t0_s=t0_s,
        t1_s=t1_s,
        f0_khz=float(freq[row0]),
        f1_khz=float(freq[row1 - 1]),
        f_centroid_khz=centroid,
        chirp_khz_per_ms=chirp,
        chirp_r2=r2,
        bandwidth_khz=bandwidth,
        duration_ms=(t1_s - t0_s) * 1e3,
        duty=float(lit.sum() / span),
        mean_prob=float(p.mean()),
        conf=float(np.percentile(p, CONF_PCT)),
        n_pix=int(rows.size),
        n_components=len(group),
        row_lit_fraction=row_lit_fraction,
        pickup=bool(row_lit_fraction >= PICKUP_ROW_FRACTION),
    )


# ---------------------------------------------------- harmonics, coincidence

def _overlap_s(a: Track, b: Track) -> float:
    """Seconds two tracks are both on the spectrogram."""
    return min(a.t1_s, b.t1_s) - max(a.t0_s, b.t0_s)


def harmonics(
    tracks: Sequence[Track],
    *,
    tol: float = HARMONIC_TOL,
    min_overlap_s: float = MIN_OVERLAP_S,
) -> dict[int, list[int]]:
    """Track index -> the indices whose centroid is a multiple of its own.

    A mode with harmonics is a different animal from one without - an EHO is
    a comb, a fishbone is not - and this is the cheapest evidence of it that
    an unlabelled mask offers. Only `HARMONIC_ORDERS` count, the tolerance
    is relative (a 6% window at 8 kHz is 0.5 kHz, about a mode's own
    bandwidth), and the two have to be on the spectrogram at the same time
    for at least `min_overlap_s`: a 16 kHz burst an hour after an 8 kHz one
    is not its second harmonic.

    Every index is a key, so a caller can index without a `get`.
    """
    out: dict[int, list[int]] = {i: [] for i in range(len(tracks))}
    for i, a in enumerate(tracks):
        if not a.f_centroid_khz > 0.0:
            continue
        for j, b in enumerate(tracks):
            if i == j or _overlap_s(a, b) < min_overlap_s:
                continue
            if any(
                abs(b.f_centroid_khz - n * a.f_centroid_khz)
                <= tol * n * a.f_centroid_khz
                for n in HARMONIC_ORDERS
            ):
                out[i].append(j)
    return out


def n_harmonics(
    tracks: Sequence[Track],
    *,
    tol: float = HARMONIC_TOL,
    min_overlap_s: float = MIN_OVERLAP_S,
) -> list[int]:
    """How many harmonic partners each track has, in track order."""
    found = harmonics(tracks, tol=tol, min_overlap_s=min_overlap_s)
    return [len(found[i]) for i in range(len(tracks))]


def _span_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    """Overlap over union of two 1-D extents; 0.0 for extents that touch.

    What `_iou` degrades to when one axis has no extent to share.
    """
    inter = min(a1, b1) - max(a0, b0)
    union = max(a1, b1) - min(a0, b0)
    if inter <= 0.0 or union <= 0.0:
        return 0.0
    return inter / union


def _iou(a: Track, b: Track) -> float:
    """Intersection over union of two tracks' time-frequency boxes.

    A one-column or a one-row track has NO AREA, and two of them have no
    union either - which an area ratio cannot score at all. Scoring that
    case 1.0, as this did, made every pair of degenerate boxes that so much
    as touched a coincidence, whatever `iou_min` said: two one-row tracks
    sharing 1 ms of 100 were grouped, and so were two point tracks on
    adjacent bands that share nothing.

    So a zero union falls back to the axis that still has extent: the boxes
    must COINCIDE on the degenerate axis, and the score is the overlap
    fraction of the other one. When both axes are degenerate - two pixels,
    or a row crossing a column - there is no such axis, and only identical
    boxes match.
    """
    dt = min(a.t1_s, b.t1_s) - max(a.t0_s, b.t0_s)
    df = min(a.f1_khz, b.f1_khz) - max(a.f0_khz, b.f0_khz)
    if dt < 0.0 or df < 0.0:
        return 0.0
    inter = dt * df
    union = (
        (a.t1_s - a.t0_s) * (a.f1_khz - a.f0_khz)
        + (b.t1_s - b.t0_s) * (b.f1_khz - b.f0_khz)
        - inter
    )
    if union > 0.0:
        return inter / union
    # Zero union: both boxes are degenerate, and on the same axes - a box
    # with area cannot make the union zero.
    flat_t = a.t0_s == a.t1_s or b.t0_s == b.t1_s
    flat_f = a.f0_khz == a.f1_khz or b.f0_khz == b.f1_khz
    if flat_t and flat_f:
        return float(
            (a.t0_s, a.t1_s, a.f0_khz, a.f1_khz)
            == (b.t0_s, b.t1_s, b.f0_khz, b.f1_khz)
        )
    if flat_t:
        if (a.t0_s, a.t1_s) != (b.t0_s, b.t1_s):
            return 0.0
        return _span_iou(a.f0_khz, a.f1_khz, b.f0_khz, b.f1_khz)
    if (a.f0_khz, a.f1_khz) != (b.f0_khz, b.f1_khz):
        return 0.0
    return _span_iou(a.t0_s, a.t1_s, b.t0_s, b.t1_s)


def cooccurrence(
    by_channel: Mapping[str, Sequence[Track]],
    *,
    iou_min: float = IOU_MIN,
) -> list[set[tuple[str, int]]]:
    """Groups of `(channel_key, track_idx)` that are the same event.

    A mode on the magnetics and on the ECE at the same time and frequency is
    one mode seen twice, and that is evidence about what it is: a mode that
    shows on every diagnostic is global, one that shows only on the edge ECE
    is not. Only pairs on DIFFERENT channels are compared - one channel
    seeing its own mode twice is one detector, not a coincidence - and the
    grouping is the transitive closure, so a mode that drifts across three
    channels' boxes comes back as one group.

    Singletons are left out: a track nobody else saw is not a coincidence,
    and it is already in the events table on its own.
    """
    keys: list[tuple[str, int]] = []
    items: list[Track] = []
    for channel in by_channel:
        for i, track in enumerate(by_channel[channel]):
            keys.append((channel, i))
            items.append(track)
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if keys[a][0] == keys[b][0]:
                continue
            if _iou(items[a], items[b]) >= iou_min:
                parent[find(a)] = find(b)

    groups: dict[int, set[tuple[str, int]]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(find(i), set()).add(key)
    return [g for _, g in sorted(groups.items(), key=lambda kv: min(kv[1]))
            if len(g) > 1]


# ------------------------------------------------------------------ events

def as_attrs(track: Track) -> dict[str, Any]:
    """A Track as a JSON-safe dict: python scalars, no NaN.

    A non-finite field becomes `None`, which is what JSON has for "not
    known": a track read back from a packed mask has no probabilities
    behind it, so its `conf` and `mean_prob` are NaN, and `schema` refuses
    a bare `NaN` in an attrs string on purpose.
    """
    out: dict[str, Any] = {}
    for name, value in asdict(track).items():
        if isinstance(value, bool):
            out[name] = bool(value)
        elif isinstance(value, (int, np.integer)):
            out[name] = int(value)
        else:
            number = float(value)
            out[name] = number if math.isfinite(number) else None
    return out


def tracks_to_events(
    tracks: Sequence[Track],
    *,
    shot: int,
    diag: str,
    channel: int,
    pass_name: str,
    t_cov: tuple[float, float],
    unet_sha256: str,
) -> list[Event]:
    """Tracks -> event rows, one per track.

    `phenomenon` is `coherent_mode` for everything except a pickup line,
    which is `pickup`: nothing here knows what a mode IS, and a row that
    said so would be a claim no evidence in this module supports. What the
    descriptors know goes into `attrs`, whole, beside the checkpoint that
    produced the mask - an event that cannot say which weights found it
    cannot be re-checked when the weights change.
    """
    counts = n_harmonics(tracks)
    out: list[Event] = []
    for i, track in enumerate(tracks):
        attrs = as_attrs(track)
        attrs["n_harmonics"] = int(counts[i])
        attrs["unet_sha256"] = str(unet_sha256)
        out.append(
            Event(
                shot=int(shot),
                source=SOURCE,
                evidence_kind="detector",
                phenomenon=PICKUP_PHENOMENON if track.pickup else PHENOMENON,
                t0_s=track.t0_s,
                t1_s=track.t1_s,
                f0_khz=track.f0_khz,
                f1_khz=track.f1_khz,
                confidence=track.conf,
                diag=str(diag),
                channel=int(channel),
                pass_name=str(pass_name),
                attrs=attrs,
                t_cov0_s=float(t_cov[0]),
                t_cov1_s=float(t_cov[1]),
            )
        )
    return out


def tracks_for_block(
    block_prefix: str,
    masks_path,
    *,
    min_area: int = MIN_AREA,
    max_gap: int = MERGE_GAP_COLS,
    freq_overlap: float = FREQ_OVERLAP,
) -> list[Track]:
    """One stored `(diag, channel, pass)` block -> its tracks.

    The whole path in one call, for the script that walks a masks file.
    What the file kept is a BOOLEAN mask and a 16-band power summary, so
    `prob` here is 1 on every lit pixel - which is a geometry, not a
    probability. `mean_prob` and `conf` therefore come back NaN, not 1.0:
    the file has no probabilities in it, and a track whose confidence is
    unknown must not outrank one whose confidence was measured. The
    confidence that means anything is the one computed while the
    probabilities were still in memory. The bands are stretched back over
    their 32 rows each, which is enough to weight a centroid inside a band
    and not enough to place a mode.
    """
    meta = read_mask(masks_path, f"{block_prefix}_meta")
    n_cols = int(meta["n_cols"])
    mask = unpack(
        read_mask(masks_path, f"{block_prefix}_coh_packed"), (N_BINS, n_cols)
    )
    bands = np.asarray(
        read_mask(masks_path, f"{block_prefix}_band_logpow"), dtype=np.float32
    )
    t_s = np.asarray(read_mask(masks_path, f"{block_prefix}_t_s"), dtype=np.float64)
    freq_khz = freq_axis_khz(float(meta["fs_hz"]), int(meta["decim"]))
    prob = mask.astype(np.float32)
    raw_logpow = np.repeat(bands, N_BINS // bands.shape[0], axis=0)
    groups = merge(
        components(mask, min_area=min_area),
        max_gap=max_gap,
        freq_overlap=freq_overlap,
    )
    return [
        replace(
            descriptors(
                group, prob=prob, raw_logpow=raw_logpow,
                freq_khz=freq_khz, t_s=t_s,
            ),
            mean_prob=math.nan,
            conf=math.nan,
        )
        for group in groups
    ]
