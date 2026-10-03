"""TokEye's mode mask -> a filtered mask -> components -> each tagged by a label.

The interpreter figure (`scripts/labeler/paper/fig_interpreter_tokeye.py`) shows
how a label can be made from the network's mask without a human in the loop.
This module is that chain, on one `(512, T)` mask of one pass (`masks`' wide or
zoom pass), in four steps:

1. `mode_mask`: the coherent channel at `PROB_THRESHOLD` or above, and not the
   transient channel: a mode, not an ELM or a sawtooth burst crossing it.
2. `bridge_pickup` and `clean`: a row lit for most of the record is receiver
   pickup, a line at one frequency (`tracks.PICKUP_ROW_FRACTION` is the event
   layer's 0.8; the figure's looser `PICKUP_SHARE` catches the lines of a short
   window); it is dropped where a row above and below it is not lit, so a mode
   crossing it stays whole. Then `skimage.morphology.remove_small_objects` and
   `remove_small_holes`.
3. `tracks.components`: the 8-connected blobs of what is left.
4. `tag_components`: a blob is tagged by each event whose label is present over
   at least `COVER` of its time and whose band holds its frequency centroid:
   AE above `SPLIT_KHZ`, the tearing mode and the sawtooth below it.

The tag says that a mode and a label coincide in time and band: it is the
label's, not a second classification of the mode. Two events in one band (a
tearing mode and a sawtooth, both labelled present) both tag the blob.
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
#: needs the blob's frequency centroid in: AEs above 60 kHz, the tearing mode
#: and the sawtooth below it.
SPLIT_KHZ = 60.0
BANDS = {
    AE: (SPLIT_KHZ, math.inf),
    NTM: (0.0, SPLIT_KHZ),
    SAWTOOTH: (0.0, SPLIT_KHZ),
}
#: The share of a blob's time a label must be present over to tag it.
COVER = 0.5
#: A row lit for more than this share of the record is pickup (`bridge_pickup`).
PICKUP_SHARE = 0.4
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
    lit: np.ndarray, row_share: np.ndarray, share: float = PICKUP_SHARE
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


def tag_blobs(
    found: Sequence[Blob],
    spans: Mapping[str, Sequence[tuple[float, float]]],
    bands: Mapping[str, tuple[float, float]] = BANDS,
    cover: float = COVER,
) -> list[Blob]:
    """`found` with each blob's tags: the events (by `bands`' order) whose
    present `spans` (ms) cover at least `cover` of the blob's time and whose
    band holds its frequency centroid. A blob of one column is covered when its
    time lies in a span."""
    merged = {event: union(s) for event, s in spans.items()}
    out = []
    for blob in found:
        tags = []
        for event, (lo, hi) in bands.items():
            if event not in merged or not lo <= blob.f_khz < hi:
                continue
            length = blob.t1_ms - blob.t0_ms
            if length > 0:
                share = overlap_ms(blob.t0_ms, blob.t1_ms, merged[event]) / length
            else:
                share = float(
                    overlap_ms(blob.t0_ms, blob.t0_ms + 1e-9, merged[event]) > 0
                )
            if share >= cover:
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
            )
        )
    return out
