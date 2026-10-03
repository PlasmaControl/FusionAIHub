"""TokEye's mode mask -> a filtered mask -> components -> each tagged by a label.

The interpreter figure (`scripts/labeler/paper/fig_interpreter_tokeye.py`) shows
how a label can be made from the network's mask without a human in the loop.
This module is that chain, on one `(512, T)` mask of one pass (`masks`' wide or
zoom pass), in four steps:

1. `mode_mask`: the coherent channel at `PROB_THRESHOLD` or above, and not the
   transient channel: a mode, not an ELM or a sawtooth burst crossing it.
2. `bridge_pickup` and `clean`: a row lit for over 80% of the record is a
   persistent-line candidate, not a physical identification of pickup. It is
   dropped where a row above and below it is not lit, so a mode
   crossing it stays whole. Then `skimage.morphology.remove_small_objects` and
   `remove_small_holes`.
3. `tracks.components`: the 8-connected blobs of what is left.
4. `tag_blobs` / `tag_mask`: intersect component pixels with PRESENT label
   times and the fixed event band. NTM requires dominant measured n=1 or n=2.
   Sawtooth crashes are point events, never rotating-mode tags.

The tag says that a mode and a label coincide in time and band: it is the
label's, not a second classification of the mode.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..ae.labels import PROB_THRESHOLD
from ..events.tracks import Component, components

AE = "alfven_eigenmode"
NTM = "neoclassical_tearing_mode"
SAWTOOTH = "sawtooth_oscillation"
#: The band (kHz, lower edge included, upper edge excluded) each event's tag
#: contains every highlighted pixel: AE >=60 kHz, NTM/sawtooth <60 kHz.
SPLIT_KHZ = 60.0
BANDS = {
    AE: (SPLIT_KHZ, math.inf),
    NTM: (0.0, SPLIT_KHZ),
    SAWTOOTH: (0.0, SPLIT_KHZ),
}
#: Temporal persistence alone does not establish receiver pickup. The old
#: 0.4 cutoff removed real n=2 rows at 15 and ~30 kHz; retain those rows.
PERSISTENT_ROW_SHARE = 0.8
#: `remove_small_objects`' size, pixels, by pass: the zoom pass has four times
#: the wide pass's columns per ms, so its blobs are larger.
MIN_SIZE = {"zoom": 40, "wide": 30}
#: `remove_small_holes`' area, pixels.
HOLE_AREA = 10


def mode_mask(
    coherent: np.ndarray, transient: np.ndarray, threshold: float = PROB_THRESHOLD
) -> np.ndarray:
    """The pixels the coherent channel lights and the transient one does not."""
    return (np.asarray(coherent) >= threshold) & ~(np.asarray(transient) >= threshold)


def bridge_pickup(
    lit: np.ndarray, row_share: np.ndarray, share: float = PERSISTENT_ROW_SHARE
) -> np.ndarray:
    """`lit` with each pickup row (`row_share` above `share`) set to what lies
    on both sides of it: lit only where the nearest rows above and below that
    are not pickup are both lit, so a line of pickup goes and a mode that
    crosses it stays one."""
    lit = np.asarray(lit, bool)
    pickup = np.asarray(row_share) > share
    out = lit.copy()
    rows = np.flatnonzero(pickup)
    if not rows.size:
        return out
    for run in np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1):
        below, above = run[0] - 1, run[-1] + 1
        if below < 0 or above >= len(lit):
            out[run] = False
        else:
            out[run] = lit[below] & lit[above]
    return out


def clean(lit: np.ndarray, min_size: int, hole_area: int = HOLE_AREA) -> np.ndarray:
    """`lit` without objects under `min_size` pixels (8-connected), and
    without holes under `hole_area` pixels. scikit-image is read here, not at
    import, so the TokEye cache step can run where it is not installed."""
    from skimage.morphology import remove_small_holes, remove_small_objects

    kept = remove_small_objects(
        np.asarray(lit, bool), min_size=min_size, connectivity=2
    )
    return remove_small_holes(kept, area_threshold=hole_area, connectivity=2)


def filtered(
    coherent: np.ndarray,
    transient: np.ndarray,
    row_share: np.ndarray,
    min_size: int,
    threshold: float = PROB_THRESHOLD,
) -> np.ndarray:
    """The chain's steps 1 and 2 in one: `mode_mask`, `bridge_pickup`, `clean`."""
    lit = mode_mask(coherent, transient, threshold)
    return clean(bridge_pickup(lit, row_share), min_size)


@dataclass(frozen=True)
class Blob:
    """A component, in the shot's units: its time span (ms), its frequency
    range and centroid (kHz), its pixels and the events that tag it."""

    component: Component
    t0_ms: float
    t1_ms: float
    f0_khz: float
    f1_khz: float
    f_khz: float
    tags: tuple[str, ...] = ()
    dominant_n: int | None = None

    @property
    def n_pix(self) -> int:
        return self.component.n_pix


def union(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """`spans` (start, end) merged where they touch or overlap, in order."""
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def overlap_ms(a: float, b: float, spans: Sequence[tuple[float, float]]) -> float:
    """The length of `[a, b]` that the (disjoint) `spans` cover."""
    return math.fsum(max(0.0, min(b, e) - max(a, s)) for s, e in spans)


def blobs(
    lit: np.ndarray, t_ms: np.ndarray, f_khz: np.ndarray, min_area: int = 1
) -> list[Blob]:
    """The components of `lit` (rows at `f_khz`, columns at `t_ms`, both
    increasing) in the shot's units, untagged. A column's time is its centre,
    so a blob one column wide has zero length."""
    t_ms, f_khz = np.asarray(t_ms, float), np.asarray(f_khz, float)
    out = []
    for c in components(lit, min_area=min_area):
        out.append(
            Blob(
                c,
                float(t_ms[c.col0]),
                float(t_ms[c.col1 - 1]),
                float(f_khz[c.row0]),
                float(f_khz[c.row1 - 1]),
                float(f_khz[c.rows].mean()),
            )
        )
    return out


def present_columns(t_ms: np.ndarray, spans: Sequence[tuple[float, float]]):
    """Half-open PRESENT spans on a time grid; absent gaps stay false."""
    t = np.asarray(t_ms)
    keep = np.zeros(t.shape, bool)
    for a, b in spans:
        keep |= (t >= a) & (t < b)
    return keep


def tag_blobs(
    found: Sequence[Blob],
    spans: Mapping[str, Sequence[tuple[float, float]]],
    t_ms: np.ndarray,
    f_khz: np.ndarray,
    n_map: np.ndarray | None = None,
    bands: Mapping[str, tuple[float, float]] = BANDS,
) -> list[Blob]:
    """Tag components with any eligible pixels; drawing must use `tag_mask`.

    NTM requires the most common finite n among a component's measured pixels
    to be 1 or 2. Unknown n (including outside the n map's support) never
    supplies evidence. Sawtooth is deliberately excluded from mode tags.
    """
    columns = {event: present_columns(t_ms, s) for event, s in spans.items()}
    out = []
    for blob in found:
        rr, cc = blob.component.rows, blob.component.cols
        dominant = None
        if n_map is not None:
            values = n_map[rr, cc]
            values = values[np.isfinite(values)].astype(int)
            if values.size:
                ns, counts = np.unique(values, return_counts=True)
                if np.count_nonzero(counts == counts.max()) == 1:
                    dominant = int(ns[np.argmax(counts)])
        tags = []
        for event, (lo, hi) in bands.items():
            if event not in columns or event == SAWTOOTH:
                continue
            if event == NTM and dominant not in (1, 2):
                continue
            eligible = columns[event][cc] & (f_khz[rr] >= lo) & (f_khz[rr] < hi)
            if eligible.any():
                tags.append(event)
        out.append(
            Blob(
                blob.component,
                blob.t0_ms,
                blob.t1_ms,
                blob.f0_khz,
                blob.f1_khz,
                blob.f_khz,
                tuple(tags),
                dominant,
            )
        )
    return out


def tag_mask(found, event, shape, t_ms, f_khz, spans) -> np.ndarray:
    """Only tagged component pixels inside PRESENT times AND the event band.

    A component crossing 60 kHz is split at pixel level; a component crossing
    an absent interval is split in time, regardless of its overall coverage.
    The 55 kHz display fold has no role in this decision.
    """
    out = np.zeros(shape, bool)
    for blob in found:
        if event in blob.tags:
            out[blob.component.rows, blob.component.cols] = True
    lo, hi = BANDS[event]
    out &= present_columns(t_ms, spans)[None, :]
    out &= ((f_khz >= lo) & (f_khz < hi))[:, None]
    return out


def sample_n_map(codes, ns, map_t, map_f, t_ms, f_khz) -> np.ndarray:
    """Decode the nearest n-map cell on a spectrogram grid, NaN outside support.

    Codes carry brightness in their quotient and n in their remainder. Level
    zero is no measured mode. No extrapolation beyond the map's cell edges.
    """
    map_t, map_f = np.asarray(map_t), np.asarray(map_f)
    out = np.full((len(f_khz), len(t_ms)), np.nan, dtype=np.float32)
    if len(map_t) < 2 or len(map_f) < 2:
        return out
    dt, df = np.median(np.diff(map_t)), np.median(np.diff(map_f))
    tc = (t_ms >= map_t[0] - dt / 2) & (t_ms < map_t[-1] + dt / 2)
    fr = (f_khz >= map_f[0] - df / 2) & (f_khz < map_f[-1] + df / 2)
    ci = np.clip(np.rint((t_ms[tc] - map_t[0]) / dt).astype(int), 0, len(map_t) - 1)
    ri = np.clip(np.rint((f_khz[fr] - map_f[0]) / df).astype(int), 0, len(map_f) - 1)
    chosen = np.asarray(codes)[np.ix_(ri, ci)].astype(int)
    decoded = np.asarray(ns)[chosen % len(ns)].astype(np.float32)
    decoded[chosen < len(ns)] = np.nan
    out[np.ix_(fr, tc)] = decoded
    return out
