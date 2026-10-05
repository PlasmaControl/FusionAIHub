r"""fig_interpreter (the teaser): raw signals -> TokEye-processed -> labelled.

    PYTHONPATH=src pixi run --frozen -e labelmaker \
        python scripts/labeler/paper/fig_interpreter_tokeye.py \
        [--shot 199563] [--tmin MS --tmax MS] [--out DIR]

One non-blind cohort shot over a few seconds, in three groups on one time axis:

- **raw**: the Mirnov probe's spectrogram (MPI66M322D, the wide pass) on one
  linear 0-250 kHz axis, one colour scale, D-alpha and NBI power;
- **processed**: the same spectrogram through TokEye (the network's mode mask,
  `labeler.paper.mode_tags`: transient burst removed, pickup removed,
  `skimage.morphology.remove_small_objects` and small-hole filling, connected
  components) on the same linear axis (the zoom pass below `ZOOM_TOP_KHZ`, the
  wide pass above), the toroidal-n view (the review page's n map, gated by the
  same mask) in the mask below 30 kHz. Highlights intersect PRESENT label
  times with AE >=60 kHz or NTM <=30 kHz; NTM requires dominant and pixel n=1 or 2.
  Optional ECE-supported, ELM-vetoed crash candidates appear on a thin strip. D-alpha
  carries the ELM label's span and the D-alpha peaks in
  it, and the confinement classes shade it;
- **labels**: one track per event on the shot's time axis, from the best tier
  that holds the shot (`labeler.paper.label_figure.TRACKS`): present, absent or
  blank (unassessed / unobservable).

TokEye runs once per shot and window, on the CPU (or `--device cuda` in a CUDA
build: then `--cache-only` there, the drawing needs scikit-image), and is cached
under `--cache`. Writes `fig_interpreter.pdf` and `.png` (150 dpi) and the
record `fig_interpreter.json` beside them; `--record` copies the record where
the repo keeps it. A tag says that a mode and a label coincide in time and band,
not that the mode is that event.

CONFINEMENT_RUN_DIR defaults to runs/labeler/confinement/v1 in the checkout
containing LABELER_LABEL_TABLES. Missing required curated/roster/fallback tables
are errors. The confinement row is the saved four-class review, else the curated
regimes, else the released confine-ours roster, else the D-alpha L-H table, never
the H-mode frame model. --sawtooth-source accepts read-only physics
JSONs, a JSON directory or cohort CSV shards; --sawtooth-evidence supplies full
JSON evidence for the reduced cohort CSV schema. Expert sawtooth intervals take
precedence over physics states; candidate ticks use separate physics evidence.
--ae-labels accepts isolated
paper-model inference predictions. The catalog sawtooth frame model is not shown.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path as PlotPath
from scipy import ndimage, signal

from labeler.config import Paths, atomic_path, git_sha, sha256_of
from labeler.events import masks, unet
from labeler.events.catalog.cohort import read_cohort
from labeler.events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT, UNCERTAIN
from labeler.events.spans import cohort_path
from labeler.events.verify import (
    NoDataError,
    corpus_path,
    corpus_signal,
)
from labeler.paper import PAGE_IN, figure_sources, mode_tags, roster, style
from labeler.paper import label_figure as lf

STEM = "fig_interpreter"
HEIGHT_IN = 5.6
DPI_PNG = 150
DPI_PDF = 300
#: The window's margin the TokEye cache keeps beyond the figure's, ms.
CACHE_MARGIN_MS = 100.0
#: Each shot's window, ms, where `--tmin`/`--tmax` are not given (the cohort
#: window otherwise): chosen on the shot's own labels, see the report.
PRESETS = {
    186636: (1300.0, 3900.0),
    201973: (1600.0, 3350.0),
    199563: (700.0, 5800.0),
    201978: (1500.0, 3300.0),
    203187: (1700.0, 3150.0),
}
#: The AE/NTM split (mode_tags.SPLIT_KHZ): AE is tinted at and above it, NTM and
#: sawtooth below it. The frequency axis itself is linear and unbroken.
SPLIT_KHZ = mode_tags.SPLIT_KHZ
#: The tag analysis reads the zoom pass (0.12 kHz/bin; its decimation filter rolls
#: off from about 50 kHz, Nyquist 62.5 kHz) below `SPLIT_KHZ`, the wide pass
#: (0.49 kHz/bin, to 250 kHz) above it.
#: The mask is drawn from the zoom pass up to here; its top few kHz are the
#: decimation filter's roll-off (a dark strip), so the wide pass fills the rest.
ZOOM_TOP_KHZ = 50.0
#: Both spectrograms: one linear axis, these ticks.
FREQ_TICKS_KHZ = [0, 50, 100, 150, 200, 250]
TOP_KHZ = 250.0
N_VIEW_KHZ = 30.0  # the n map's band
#: The columns an image is pooled to for the page (a 3 s wide pass has 12,000).
IMAGE_COLUMNS = 3000
#: The ELM peaks: D-alpha less its running median, over this many MADs.
ELM_WINDOW = 25
ELM_MADS = 4.0
#: Okabe-Ito blue, bluish-green and yellow; other n is purple, at full colour.
N_COLOURS = {1: "#0072B2", 2: "#009E73", 3: "#F0E442"}
N_OTHER = "#8C6BB1"
EVENT_COLOURS = {
    mode_tags.AE: "#CC79A7",
    mode_tags.NTM: "#E69F00",
    mode_tags.SAWTOOTH: "#D55E00",
    "edge_localized_mode": "#333333",
}
EVENT_NAMES = {
    mode_tags.AE: "AE",
    mode_tags.NTM: "NTM",
    mode_tags.SAWTOOTH: "sawtooth",
    "edge_localized_mode": "ELMs",
}
TIER_NAMES = {lf.SILVER: "expert", lf.LEGACY: "imported", lf.GENERATED: "detector"}
#: The confinement classes' colours, as the paper's other figures draw them; class 5
#: (uncertain) is hatched, never coloured.
CLASS_COLOURS = {
    1: lf.COLOURS["H-mode"],
    2: lf.COLOURS["L-mode"],
    3: lf.COLOURS["QH-mode"],
    4: lf.COLOURS["WPQH-mode"],
}
#: Legend order and compact names of the classes (L, H, QH, WPQH).
CLASS_LEGEND = {2: "L", 1: "H", 3: "QH", 4: "WPQH"}
ABSENT_GREY = "#e4e4e4"
NTM_CONTOUR_COLOUR = EVENT_COLOURS[mode_tags.NTM]
NTM_CONTOUR_LW = 0.9
#: The thin dashed outline of the rest of a tagged component the detector clip cuts.
NTM_DASHED_LW = 0.7
NTM_DASHED_STYLE = (0, (3, 1.6))
#: Outlines enclosing fewer print pixels (150 dpi) than this are not drawn.
NTM_OUTLINE_MIN_PX = 100
HEADINGS = {
    "h_raw": ("Raw signals",),
    "h_proc": (
        "TokEye-processed: modes overlapping event labels",
        ("mask → remove small objects / fill holes → components → time/band tags"),
    ),
    "h_lab": ("Labels; source at right",),
}
INK = "#222222"
BAR = (0.12, 0.76)
FONT = 7


# ---------------------------------------------------------------------- TokEye


def tokeye_file(paths: Paths, shot: int, t0: float, t1: float, cache: Path) -> Path:
    return cache / f"tokeye_{shot}_{int(t0)}_{int(t1)}.npz"


def run_tokeye(paths: Paths, shot: int, t0: float, t1: float, device: str) -> dict:
    """TokEye's wide and zoom passes over `shot`'s probe record, cropped to
    `t0`-`t1` ms: `{pass}_{raw,coh,tra}` as `(512, T)` float16 (the
    spectrogram, the coherent and the transient channel), the `_t_ms` and
    `_f_khz` axes and `_row_lit`, the share of the whole record each row's
    coherent mask lights (pickup shows as a row lit for most of it)."""
    import torch

    if device == "cpu":
        torch.set_num_threads(8)
    model = unet.load_unet(roster.tokeye_file(paths), device=device)
    y, fs, t0_s, _ = masks.read_waveform(
        paths.corpus_file(shot), roster.GATE_GROUP, roster.GATE_ROW
    )
    out: dict = {}
    for name, decim in (("wide", 1), ("zoom", masks.ZOOM_DECIM)):
        spec, meta = masks.prep(y, fs_hz=fs, decim=decim)
        probs = masks.infer(model, spec, device, batch=8, amp=False)
        t_ms = masks.col_times_s(spec.shape[1], fs, decim, t0_s) * 1000
        keep = (t_ms >= t0) & (t_ms <= t1)
        out[f"{name}_raw"] = masks.unstandardise(spec[:, keep], meta).astype(np.float16)
        out[f"{name}_coh"] = probs[0][:, keep].astype(np.float16)
        out[f"{name}_tra"] = probs[1][:, keep].astype(np.float16)
        out[f"{name}_t_ms"] = t_ms[keep]
        out[f"{name}_f_khz"] = masks.freq_axis_khz(fs, decim)
        out[f"{name}_row_lit"] = (probs[0] >= mode_tags.PROB_THRESHOLD).mean(axis=1)
    return out


def tokeye_passes(
    paths: Paths, shot: int, t0: float, t1: float, cache: Path, device: str
) -> tuple[dict, Path]:
    """Reuse arrays only with matching checkpoint, waveform and preprocessing."""
    path = tokeye_file(paths, shot, t0, t1, cache)
    sha = sha256_of(roster.tokeye_file(paths))
    fingerprints = json.dumps(
        figure_sources.tokeye_fingerprints(
            paths,
            shot,
            roster.GATE_GROUP,
            roster.GATE_ROW,
            inspect.getsource(run_tokeye),
        ),
        sort_keys=True,
    )

    def matches(z):
        return (
            "fingerprints" in z.files
            and str(z["checkpoint_sha256"]) == sha
            and str(z["fingerprints"]) == fingerprints
        )

    if path.is_file():
        with np.load(path) as z:
            if matches(z):
                return {k: z[k] for k in z.files}, path
    # A shorter display can use the same full-record inference and row shares.
    # Reuse its immutable superset cache rather than changing model inputs.
    for existing in sorted(cache.glob(f"tokeye_{shot}_*.npz")):
        _, _, a, b = existing.stem.split("_")
        if float(a) <= t0 and float(b) >= t1:
            with np.load(existing) as z:
                if matches(z):
                    return {k: z[k] for k in z.files}, existing
    out = run_tokeye(paths, shot, t0, t1, device)
    out["checkpoint_sha256"] = np.array(sha)
    out["fingerprints"] = np.array(fingerprints)
    cache.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp, open(tmp, "wb") as handle:
        np.savez(handle, **out)
    return out, path


# ---------------------------------------------------------------------- labels


def ae_rows_raw(paths: Paths, candidate: lf.Candidate) -> tuple[lf.Row, ...]:
    """The AE frame model over the shot's CO2 in the raw cache (a shot the
    corpus holds no CO2 for, `label_figure.ae_rows`' corpus only)."""
    from labeler.ae.xpower.data import raw_rows, window_frames
    from labeler.ae.xpower.extend import frame_states
    from labeler.ae.xpower.train import probabilities
    from labeler.events import suggestions
    from labeler.paper.shots import Model

    model = Model.load(lf.ae_model_file(paths), {})
    co2 = corpus_signal(candidate.shot, "co2", corpus=paths.raw_cache)
    rows = raw_rows(co2.x, co2.y)
    first, n = window_frames(candidate.window)
    prob, observed = probabilities(
        model.net, rows, first, n, band=model.blob["band_khz"]
    )
    states = frame_states(prob, observed, model.threshold)
    table = suggestions.frame_rows(candidate.shot, first, states)
    return tuple(lf.Row(float(a), float(b), int(c)) for _, c, a, b, _ in table)


def shot_tracks(
    paths: Paths,
    candidate: lf.Candidate,
    sawtooth_labels: Path | None = None,
    ae_labels: Path | None = None,
    show_sawtooth: bool = False,
) -> tuple[tuple[lf.Track, ...], Path | None]:
    """The catalog's tracks on `candidate`: `label_figure`'s, the AE generated
    from the corpus CO2, else from the raw cache; and the CO2 file the
    generated AE track ran over (None where the AE track is not generated)."""
    chosen_ae = figure_sources.reviewed_or_stored_ae(paths, candidate.shot, ae_labels)
    confinement = figure_sources.confinement_track(paths, candidate.shot)
    specs = tuple(
        s
        for s in lf.TRACKS
        if s.key not in ("confinement", mode_tags.SAWTOOTH)
        and (chosen_ae is None or s.key != mode_tags.AE)
    )
    read = lf.read_sources(paths, specs)
    read = lf.generate(paths, [candidate], read, specs)
    selected = {t.spec.key: t for t in lf.tracks_of(candidate.shot, read, specs)}
    selected["confinement"] = confinement
    if show_sawtooth:
        selected[mode_tags.SAWTOOTH] = figure_sources.sawtooth_track(
            paths, candidate.shot, sawtooth_labels, []
        )
    if chosen_ae is not None:
        selected[mode_tags.AE] = chosen_ae
    tracks = tuple(selected[s.key] for s in lf.TRACKS if s.key in selected)
    co2 = None
    out = []
    for track in tracks:
        if (
            track.spec.key == "edge_localized_mode"
            and track.source
            and track.source.tier == lf.SILVER
        ):
            track = replace(
                track,
                rows=tuple(
                    replace(r, category=figure_sources.elm_category(r))
                    for r in track.rows
                ),
            )
        if track.spec.key == mode_tags.AE and track.source is None:
            source = replace(
                lf.TRACKS[0].sources[1],
                what=lf.TRACKS[0].sources[1].what + ", the raw cache's CO2",
                locate=lf.ae_model_file,
            )
            try:
                rows = ae_rows_raw(paths, candidate)
            except NoDataError:
                rows = ()
            if rows:
                track = lf.Track(track.spec, source, lf.ae_model_file(paths), rows)
                co2 = corpus_path(candidate.shot, corpus=paths.raw_cache)
        elif (
            track.spec.key == mode_tags.AE
            and track.source is not None
            and track.source.run is not None
        ):
            co2 = paths.corpus_file(candidate.shot)
        out.append(track)
    return tuple(out), co2


def present_spans(track: lf.Track) -> list[tuple[float, float]]:
    return [(r.t_start, r.t_end) for r in track.rows if r.category == PRESENT]


# ---------------------------------------------------------------------- images


def pool(a: np.ndarray, k: int, how: str = "mean") -> np.ndarray:
    """`a`'s columns in blocks of `k`, their mean or max (the ragged end dropped)."""
    n = a.shape[-1] // k * k
    b = a[..., :n].reshape(*a.shape[:-1], n // k, k)
    return b.mean(-1) if how == "mean" else b.max(-1)


def pooled_axis(t: np.ndarray, k: int) -> np.ndarray:
    n = len(t) // k * k
    return t[:n].reshape(-1, k).mean(-1)


class Band:
    """One pass's rows `lo`-`hi` kHz over the window: the raw spectrogram, the
    filtered mask and the blobs, pooled to the page's columns."""

    def __init__(self, z: dict, name: str, lo: float, hi: float, t0: float, t1: float):
        t, f = z[f"{name}_t_ms"], z[f"{name}_f_khz"]
        cols = (t >= t0) & (t <= t1)
        rows = (f >= lo) & (f < hi)
        self.name = name
        self.t, self.f = t[cols], f[rows]
        self.rows = rows
        self.raw = z[f"{name}_raw"][rows][:, cols].astype(np.float32)
        # the mask over every row, so a blob at the band's edge is whole; then cut
        lit = mode_tags.filtered(
            z[f"{name}_coh"].astype(np.float32),
            z[f"{name}_tra"].astype(np.float32),
            z[f"{name}_row_lit"],
            mode_tags.MIN_SIZE[name],
        )
        self.lit_all = lit
        self.lit = lit[rows][:, cols]
        self.lit_full = lit[:, cols]  # 512 rows: `tracks.components` wants them all
        self.row0 = int(np.flatnonzero(rows)[0])
        self.all_t = z[f"{name}_t_ms"]
        self.all_f = f
        self.cols = cols
        self.edges = None  # exact (lo, hi) kHz to stretch the image over, if set
        self.k = max(1, len(self.t) // IMAGE_COLUMNS)
        lo_v, hi_v = np.percentile(self.raw, (3, 99.8))
        self.norm = np.clip((self.raw - lo_v) / (hi_v - lo_v), 0, 1)

    def extent(self) -> tuple[float, float, float, float]:
        dt = float(np.median(np.diff(self.t)))
        df = float(np.median(np.diff(self.f)))
        lo, hi = self.edges or (self.f[0] - df / 2, self.f[-1] + df / 2)
        return (self.t[0] - dt / 2, self.t[-1] + dt / 2, lo, hi)

    def blobs(self, spans: dict, n_read=None) -> list[mode_tags.Blob]:
        """The blobs of this band's mask, tagged by `spans`' events."""
        found = mode_tags.blobs(self.lit_full, self.t, self.all_f)
        # Keep whole components for dominant-n evidence across the 60 kHz AE/NTM split.
        found = [b for b in found if (self.rows[b.component.rows]).any()]
        n = None
        if n_read is not None:
            f = n_read.meta["y0"] + np.arange(n_read.meta["n_y"]) * n_read.meta["dy"]
            n = mode_tags.sample_n_map(
                n_read.values,
                n_read.meta["modes"]["n"],
                n_read.centres,
                f,
                self.t,
                self.all_f,
            )
        self.n_map = n
        if n is not None:
            n[self.all_f > N_VIEW_KHZ] = np.nan
        bands = {
            e: (max(lo, self.f[0]), min(hi, self.f[-1] + 1e-6))
            for e, (lo, hi) in mode_tags.BANDS.items()
        }
        return mode_tags.tag_blobs(found, spans, self.t, self.all_f, n, bands)

    def tag_image(self, blobs: list[mode_tags.Blob], event: str, spans) -> np.ndarray:
        """`event`'s tagged blobs as a boolean image like `lit`."""
        return mode_tags.tag_mask(
            blobs,
            event,
            self.lit_full.shape,
            self.t,
            self.all_f,
            spans,
            n_map=self.n_map,
        )[self.rows]


# ----------------------------------------------------------------------- draw


def style_axes(ax, bottom: bool = False) -> None:
    ax.tick_params(length=2, pad=1.5, labelsize=FONT)
    ax.tick_params(labelbottom=bottom, bottom=bottom)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_raw(ax, band: Band, cmap: str = "viridis") -> None:
    img = pool(band.norm, band.k)
    ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=band.extent(),
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )


def draw_processed(ax, band: Band) -> None:
    """White coherent mask: white means n was not measured."""
    rgb = np.zeros((*band.lit.shape, 3), np.float32)
    rgb[band.lit] = 1.0
    k = band.k
    n = rgb.shape[1] // k * k
    img = rgb[:, :n].reshape(rgb.shape[0], n // k, k, 3).max(2)
    ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=band.extent(),
        interpolation="nearest",
    )


def event_clip(ax, spans, event):
    """A compound path: PRESENT times intersected with the fixed event band.

    The clip also cuts the half-cell/line-width at a boundary, so pooling,
    stroke widths and native pixel extents cannot bleed into an absent span.
    """
    lo, hi = mode_tags.BANDS[event]
    lo, hi = max(lo, ax.get_ylim()[0]), min(hi, ax.get_ylim()[1])
    rects = []
    for a, b in spans:
        a, b = max(a, ax.get_xlim()[0]), min(b, ax.get_xlim()[1])
        if b > a and hi > lo:
            rects.append(
                PlotPath(
                    [(a, lo), (b, lo), (b, hi), (a, hi), (a, lo)],
                    [1, 2, 2, 2, 79],
                )
            )
    return PlotPath.make_compound_path(*rects) if rects else PlotPath(np.empty((0, 2)))


def display_support(ax, band, mask):
    """`mask` max-pooled to the print pixels of `ax` (so narrow ridges survive),
    holes filled, with the pooling indices, the labelled regions and their sizes."""
    pos = ax.get_position()
    # The band's share of the panel's linear frequency axis.
    visible_share = (band.f[-1] - band.f[0]) / np.diff(ax.get_ylim())[0]
    nf = min(
        len(band.f), max(1, round(pos.height * HEIGHT_IN * DPI_PNG * visible_share))
    )
    nt = min(len(band.t), max(1, round(pos.width * PAGE_IN * DPI_PNG)))
    ri = np.linspace(0, mask.shape[0], nf, endpoint=False).astype(int)
    ci = np.linspace(0, mask.shape[1], nt, endpoint=False).astype(int)
    display = np.maximum.reduceat(np.maximum.reduceat(mask, ri, axis=0), ci, axis=1)
    display = ndimage.binary_fill_holes(display)
    regions, count = ndimage.label(display, structure=np.ones((3, 3)))
    sizes = np.bincount(regions.ravel(), minlength=count + 1)
    return display, ri, ci, regions, sizes


def contour_of(ax, band, display, **style_kw):
    """The outer contour of a print-pixel `display` mask over `band`'s extent."""
    nf, nt = display.shape
    x0, x1, y0, y1 = band.extent()
    # Pad with zero so contours also close at the view's edges.
    x = x0 + (np.arange(-1, nt + 1) + 0.5) * (x1 - x0) / nt
    y = y0 + (np.arange(-1, nf + 1) + 0.5) * (y1 - y0) / nf
    return ax.contour(
        x,
        y,
        np.pad(display, 1),
        levels=[0.5],
        colors=NTM_CONTOUR_COLOUR,
        **style_kw,
    )


def project(ax, band, mask, event, spans, edge=False, record=None, full=None):
    """Opaque event tint or a solid outer contour around pooled support.

    A contour is drawn only around regions of at least `NTM_OUTLINE_MIN_PX`
    print pixels (display only: tags, counts and the tag arrays are unchanged);
    `record` receives the rule and how many regions it left unoutlined. `full`
    (edge only) is the support of the whole tagged components, whatever the
    time: where it reaches beyond `mask` (the detector-positive clip cuts a
    component) a thin dashed contour of the rest is drawn under the solid one.
    """
    if not mask.any():
        return mask
    if edge:
        display, ri, ci, regions, sizes = display_support(ax, band, mask)
        small = np.flatnonzero(sizes < NTM_OUTLINE_MIN_PX)
        small = small[small > 0]
        if record is not None:
            record.update(
                min_px=NTM_OUTLINE_MIN_PX,
                regions=int(len(sizes) - 1),
                omitted_fragments=int(small.size),
                omitted_sizes_px=sorted(int(sizes[i]) for i in small),
            )
        display &= ~np.isin(regions, small)
        if full is not None and (full & ~mask).any():
            whole, _, _, whole_regions, whole_sizes = display_support(ax, band, full)
            keep = np.flatnonzero(whole_sizes >= NTM_OUTLINE_MIN_PX)
            keep = keep[keep > 0]
            whole &= np.isin(whole_regions, keep)
            beyond = [int(k) for k in keep if (whole_regions == k)[~display].any()]
            if record is not None:
                record["dashed_regions"] = len(beyond)
            if beyond:
                contour_of(
                    ax,
                    band,
                    np.isin(whole_regions, beyond),
                    linewidths=NTM_DASHED_LW,
                    linestyles=[NTM_DASHED_STYLE],
                    zorder=4.5,
                )
        elif record is not None:
            record["dashed_regions"] = 0
        artist = contour_of(
            ax,
            band,
            display,
            linewidths=NTM_CONTOUR_LW,
            linestyles="-",
            zorder=5,
        )
        artist.set_clip_path(event_clip(ax, spans, event), ax.transData)
        # Audit the measured support of the outlined region; a contour stroke
        # borders that region and does not assign n to its surrounding pixels.
        keep_px = display[
            (np.searchsorted(ri, np.arange(mask.shape[0]), side="right") - 1).clip(0)
        ][:, (np.searchsorted(ci, np.arange(mask.shape[1]), side="right") - 1).clip(0)]
        return mask & ~ndimage.binary_erosion(mask) & keep_px
    shown = mask
    rgba = np.zeros((*shown.shape, 4), np.float32)
    rgba[..., :3] = to_rgb(EVENT_COLOURS[event])
    rgba[..., 3] = shown
    artist = ax.imshow(
        rgba,
        origin="lower",
        aspect="auto",
        extent=band.extent(),
        interpolation="nearest",
        zorder=4,
    )
    artist.set_clip_path(event_clip(ax, spans, event), ax.transData)
    return shown


def chip(ax, x, y, text, colour, **kw):
    """A label with a coloured background; returns the text."""
    return ax.text(
        x,
        y,
        text,
        fontsize=FONT,
        color="black",
        bbox={"boxstyle": "round,pad=0.12", "fc": colour, "ec": "none", "alpha": 0.95},
        zorder=8,
        **kw,
    )


def clear_of(fixed, moved: list) -> None:
    """Shift each text of `moved` that `fixed` covers to start where `fixed`
    ends (left-aligned texts of one axes, in data x)."""
    fixed.figure.draw_without_rendering()
    box = fixed.get_window_extent()
    inv = fixed.axes.transData.inverted()
    for text in moved:
        if box.overlaps(text.get_window_extent()):
            text.set_x(inv.transform((box.x1, box.y0))[0] + 15.0)


def trace(ax, read, colour=INK, lw=0.5):
    low, high = read.values
    for c in range(len(low)):
        ax.fill_between(read.centres, low[c], high[c], color=colour, lw=lw)


def elm_peaks(read, spans) -> np.ndarray:
    """D-alpha's spikes inside the ELM label's `spans`: the times (ms) of the
    maxima above the running median by `ELM_MADS` MADs."""
    x = read.values[1][0].astype(float)
    resid = x - ndimage.median_filter(x, size=ELM_WINDOW, mode="nearest")
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    peaks, _ = signal.find_peaks(resid, height=ELM_MADS * mad, distance=3)
    t = read.centres[peaks]
    inside = np.zeros(len(t), bool)
    for a, b in spans:
        inside |= (t >= a) & (t < b)
    return t[inside]


def track_bars(ax, track: lf.Track, colour: str, regimes=None, bar=BAR) -> None:
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for side in ("left", "top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    if track.source is None:
        return
    for r in track.rows:
        span = [(r.t_start, r.t_end - r.t_start)]
        if r.category == ABSENT:
            ax.broken_barh(span, bar, facecolor=ABSENT_GREY, lw=0)
            continue
        if regimes is not None:
            face = regimes.get(r.category, "#888888")
            hatch = "//////" if r.category == 5 else None
        else:
            face = colour
            hatch = None
            if r.category == UNCERTAIN:
                hatch = "//////"
            elif r.category == NOT_OBSERVABLE:
                continue  # blank: no assessment; hatching only means uncertain
        ax.broken_barh(
            span,
            bar,
            facecolor=face if hatch is None else "white",
            edgecolor=INK if hatch else face,
            hatch=hatch,
            lw=0.4 if hatch else 0,
        )
        if r.category in (PRESENT, UNCERTAIN) and r.crowd == 1 and regimes is None:
            ax.plot(
                [r.t_start, r.t_end],
                [bar[0] + bar[1] / 2] * 2,
                color=colour if hatch else "white",
                lw=0.4,
                marker="o",
                ms=1.8,
                markeredgecolor=colour,
                markerfacecolor="white",
                zorder=6,
            )


def draw_frequency_panels(ax, raw: Band, zoom: Band, top: Band) -> None:
    """The raw spectrogram and the TokEye mask, each on one linear 0-250 kHz axis.

    The raw panel is the wide pass alone on one colour scale. The mask panel takes
    the zoom pass below `ZOOM_TOP_KHZ` and the wide pass above it (`top`, drawn
    over the zoom pass's decimation roll-off); no axis break or stretching."""
    for name in ("raw", "pr"):
        ax[name].set_ylim(0, TOP_KHZ)
        ax[name].set_yticks(FREQ_TICKS_KHZ)
        ax[name].set_ylabel("kHz", labelpad=2)
    draw_raw(ax["raw"], raw)
    draw_processed(ax["pr"], zoom)
    draw_processed(ax["pr"], top)


def draw_legends(
    fig,
    ax,
    display_tracks,
    by_key,
    projected,
    crashes,
    n_handles,
    visible,
    peaks,
    t0,
    t1,
    ntm_dashed=False,
):
    """Source-aware signal/event keys and aligned label-state keys."""
    da = ax["da_pr"]
    elm_key = "edge_localized_mode"
    pos = ax["h_proc"].get_position()
    event_handles = [
        Patch(fc="white", ec=INK, lw=0.5, label="TokEye mask,\nn not measured")
    ]
    if projected["wide"][mode_tags.AE].any():
        event_handles.append(
            Patch(
                fc=EVENT_COLOURS[mode_tags.AE],
                label="AE (detector-positive\ntime; mask ≥60 kHz)",
            )
        )
    if projected["zoom"][mode_tags.NTM].any():
        ntm_label = (
            "NTM candidate\nsuggestions"
            if by_key[mode_tags.NTM].source.tier == lf.GENERATED
            else "NTM labels"
        )
        event_handles.append(
            (
                Patch(fc="black", label=f"{ntm_label}\n(n=1 or 2, ≤30 kHz)"),
                Line2D([], [], color=NTM_CONTOUR_COLOUR, ls="-", lw=NTM_CONTOUR_LW),
            )
        )
        if ntm_dashed:
            event_handles.append(
                Line2D(
                    [],
                    [],
                    color=NTM_CONTOUR_COLOUR,
                    ls=NTM_DASHED_STYLE,
                    lw=NTM_DASHED_LW,
                    label="rest of NTM component",
                )
            )
    legend_options = {
        "loc": "upper left",
        "ncols": 1,
        "frameon": False,
        "fontsize": FONT,
        "handlelength": 1.0,
        "handletextpad": 0.3,
        "borderpad": 0,
        "borderaxespad": 0,
        "labelspacing": 0.3,
    }
    fig.legend(
        handles=event_handles,
        labels=[
            h[0].get_label() if isinstance(h, tuple) else h.get_label()
            for h in event_handles
        ],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        bbox_to_anchor=(0.792, pos.y1 - 0.012),
        **legend_options,
    )
    if n_handles:
        fig.legend(
            handles=n_handles,
            title="toroidal mode number n\n(Mirnov array)",
            title_fontsize=FONT,
            bbox_to_anchor=(0.792, ax["pr"].get_position().y0),
            **{
                **legend_options,
                "loc": "lower left",
                "ncols": 2,
                "columnspacing": 0.4,
                "handlelength": 0.7,
                "labelspacing": 0.25,
            },
        )
    elm_handles = []
    if visible:
        elm_handles.append(Patch(fc="none", ec=INK, lw=0.8, label="ELM intervals"))
    if len(peaks):
        elm_handles.append(
            Line2D([], [], color=INK, marker="v", ls="", ms=3, label="D-alpha peaks")
        )
    if elm_handles:
        fig.legend(
            handles=elm_handles,
            bbox_to_anchor=(0.792, da.get_position().y1),
            **legend_options,
        )

    colours = [
        EVENT_COLOURS[t.spec.key]
        for t in display_tracks
        if t.spec.key in EVENT_COLOURS and figure_sources.has_present_time(t, (t0, t1))
    ]
    display_states = [
        r for t in display_tracks for r in figure_sources.state_intervals(t, (t0, t1))
    ]
    display_keys = []
    handles = []
    if colours:
        handles.append(tuple(Patch(fc=c, lw=0) for c in colours))
        display_keys.append("present")
    if any(r["state"] == "uncertain" for r in display_states):
        handles.append(Patch(fc="white", ec=INK, hatch="//////", lw=0.4))
        display_keys.append("uncertain")
    if any(r["state"] == "absent" for r in display_states):
        handles.append(Patch(fc=ABSENT_GREY, lw=0))
        display_keys.append("absent")
    has_blank = any(
        t.source is None
        or sum(
            r["end_ms"] - r["start_ms"]
            for r in figure_sources.state_intervals(t, (t0, t1))
            if r["state"] != "unassessed"
        )
        < t1 - t0 - 1e-6
        for t in display_tracks
    )
    if has_blank:
        handles.append(Patch(fc="white", ec="#999999", lw=0.5))
        display_keys.append("blank")
    labels = [
        "blank: unassessed / unobservable" if k == "blank" else k for k in display_keys
    ]
    # The confinement row's class colours, only those the row draws in the window.
    drawn_classes = {
        r.category
        for r in by_key["confinement"].rows
        if r.category in CLASS_LEGEND and r.t_end > t0 and r.t_start < t1
    }
    for c, name in CLASS_LEGEND.items():
        if c in drawn_classes:
            handles.append(Patch(fc=CLASS_COLOURS[c], lw=0))
            labels.append(name)
    fig.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        loc="lower center", ncols=len(handles), frameon=False, fontsize=FONT,
        bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=1.8,
        handletextpad=0.5,
    )  # fmt: skip
    expert_crowd = (
        by_key[elm_key].source is not None
        and (by_key[elm_key].source.tier == lf.SILVER)
        and any(
            r.crowd == 1
            and r.category in (PRESENT, UNCERTAIN)
            and r.t_start < t1
            and r.t_end > t0
            for r in by_key[elm_key].rows
        )
    )
    if expert_crowd:
        fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=INK,
                    lw=3,
                    marker="o",
                    ms=3,
                    markerfacecolor="white",
                    label="expert ELM interval (one span for many ELMs)",
                )
            ],
            loc="lower center",
            frameon=False,
            fontsize=FONT,
            bbox_to_anchor=(0.5, 0.034),
        )
    return colours, display_keys


def draw(
    paths: Paths,
    candidate: lf.Candidate,
    tracks: tuple[lf.Track, ...],
    z: dict,
    t0: float,
    t1: float,
    stem: Path,
    png_dpi: int = DPI_PNG,
    crash_source: Path | None = None,
    crash_evidence: Path | None = None,
    annotations: dict | None = None,
) -> dict:
    """The figure for `candidate` over `t0`-`t1` ms; returns what the record holds."""
    by_key = {t.spec.key: t for t in tracks}
    # the review stores: the n map, D-alpha, NBI
    _, drawn, stores = lf.signals(paths, candidate.shot, t0 - 10, t1 + 10)
    n_sig, da_sig, nbi_sig = drawn[0], drawn[1], drawn[2]
    all_elm_peaks = (
        elm_peaks(da_sig.rows[0], [(t0 - 10, t1 + 10)]) if da_sig.rows else None
    )
    crashes, crash_record = figure_sources.crash_times(
        paths,
        candidate.shot,
        None,
        crash_source,
        elm_times=all_elm_peaks,
        evidence=crash_evidence,
    )
    crashes = crashes[(crashes >= t0) & (crashes < t1)]
    show_sawtooth = mode_tags.SAWTOOTH in by_key
    if not show_sawtooth:
        crashes = np.array([])
        saw = None
    elif by_key[mode_tags.SAWTOOTH].source is not None:
        saw = by_key[mode_tags.SAWTOOTH]
    elif crash_source is not None:
        saw = figure_sources.sawtooth_track(
            paths, candidate.shot, crash_source, crashes
        )
    else:
        spec = by_key[mode_tags.SAWTOOTH].spec
        src = lf.Source(
            lf.GENERATED,
            "local ECE-supported crash candidate detector",
            lambda p: p.corpus_file(candidate.shot),
        )
        saw = lf.Track(
            spec,
            src,
            paths.corpus_file(candidate.shot),
            tuple(lf.Row(float(t - 0.5), float(t + 0.5), PRESENT) for t in crashes),
        )
    tracks = tuple(saw if t.spec.key == mode_tags.SAWTOOTH else t for t in tracks)
    if saw is not None:
        by_key[mode_tags.SAWTOOTH] = saw
    show_sawtooth = saw is not None
    saw_guard = None
    if saw is not None and saw.file and Path(saw.file).suffix == ".json":
        saw_guard = json.loads(Path(saw.file).read_text()).get("density_guard")
    display_saw, saw_changes = (
        figure_sources.sawtooth_display(saw, (t0, t1))
        if saw is not None
        else (None, [])
    )
    display_tracks = tuple(
        display_saw if t.spec.key == mode_tags.SAWTOOTH else t
        for t in tracks
        if t.spec.key != mode_tags.SAWTOOTH or show_sawtooth
    )

    low = Band(z, "zoom", 0.0, SPLIT_KHZ, t0, t1)
    high = Band(z, "wide", SPLIT_KHZ, TOP_KHZ + 1, t0, t1)
    # What is drawn: the wide pass on one colour scale over 0-250 kHz, and the mask
    # from the zoom pass below ZOOM_TOP_KHZ and the wide pass above it.
    raw_view = Band(z, "wide", 0.0, TOP_KHZ + 1, t0, t1)
    mask_top = Band(z, "wide", ZOOM_TOP_KHZ, TOP_KHZ + 1, t0, t1)
    spans = {e: present_spans(by_key[e]) for e in (mode_tags.AE, mode_tags.NTM)}
    n_original = n_sig.rows[0] if n_sig.rows else None
    blobs_low = low.blobs(spans, n_original)
    blobs_high = high.blobs(spans, n_original)
    projected = {
        band.name: {
            e: band.tag_image(bs, e, spans[e]) for e in (mode_tags.AE, mode_tags.NTM)
        }
        for band, bs in ((low, blobs_low), (high, blobs_high))
    }
    everywhere = [(-math.inf, math.inf)]
    whole_ntm = {
        band.name: band.tag_image(bs, mode_tags.NTM, everywhere)
        for band, bs in ((low, blobs_low), (high, blobs_high))
    }
    harmonic = figure_sources.harmonic_support(
        low.n_map[low.rows] if low.n_map is not None else None,
        low.lit & mode_tags.present_columns(low.t, spans[mode_tags.NTM])[None, :],
        low.t,
        low.f,
    )

    harmonic3 = figure_sources.harmonic_support(
        low.n_map[low.rows] if low.n_map is not None else None,
        low.lit & mode_tags.present_columns(low.t, spans[mode_tags.NTM])[None, :],
        low.t,
        low.f,
        order=3,
    )

    # the n map, gated by the same filtered mask on the zoom pass
    gate = roster.Gate(
        low.lit_all,
        low.all_f,
        low.all_t,
        {"mask": "TokEye zoom pass, filtered", "threshold": mode_tags.PROB_THRESHOLD},
    )
    n_read, n_kept = roster.gated(n_sig.rows[0], gate) if n_sig.rows else (None, None)

    layout = {
        "h_raw": 0.32, "raw": 3.6,
        "g1": 0.1, "da_raw": 0.38,
        "g2": 0.2, "nbi": 0.38, "h_proc": 0.52,
        "pr": 3.6,
        "crashes": 0.24 if len(crashes) else 0.001,
        "g3": 0.1, "da_pr": 0.52, "h_lab": 0.36,
    }  # fmt: skip
    names = [*layout, *[f"track{i}" for i in range(len(display_tracks))]]
    heights = [
        *layout.values(),
        *[0.20 for t in display_tracks],
    ]
    with style():
        fig = Figure(figsize=(PAGE_IN, HEIGHT_IN))
        gs = fig.add_gridspec(
            len(heights),
            1,
            height_ratios=heights,
            hspace=0.0,
            left=0.14,
            right=0.78,
            top=0.985,
            bottom=0.095,
        )
        ax = {n: fig.add_subplot(gs[i]) for i, n in enumerate(names)}
        for n in ("h_raw", "g1", "g2", "h_proc", "g3", "h_lab"):
            ax[n].set_visible(False)
        track_axes = [ax[f"track{i}"] for i in range(len(display_tracks))]
        for n in (
            "raw",
            "da_raw",
            "nbi",
            "pr",
            "da_pr",
            "crashes",
        ):
            ax[n].set_xlim(t0, t1)
            style_axes(ax[n])
        for a in track_axes:
            a.set_xlim(t0, t1)
            style_axes(a)

        # ---- raw
        draw_frequency_panels(ax, raw_view, low, mask_top)
        ax["raw"].text(
            1.02,
            0.97,
            "Mirnov\nmagnetics",
            transform=ax["raw"].transAxes,
            fontsize=FONT,
            ha="left",
            va="top",
            color=INK,
        )
        if da_sig.rows:
            trace(ax["da_raw"], da_sig.rows[0])
        ax["da_raw"].set_ylabel(
            "D-alpha\n(a.u.)", rotation=0, ha="right", va="center", labelpad=3
        )
        ax["da_raw"].set_yticks([])
        ax["da_raw"].set_ylim(bottom=0)
        ax["da_raw"].tick_params(bottom=False)
        if nbi_sig.rows:
            trace(ax["nbi"], nbi_sig.rows[0], colour="#555555")
        ax["nbi"].set_ylabel(
            "NBI power\n(MW)", rotation=0, ha="right", va="center", labelpad=3
        )
        ax["nbi"].set_ylim(bottom=0)

        # ---- processed: binary coherent mask, with measured n below 30 kHz
        keys = []
        if n_read is not None:
            names = n_read.meta["modes"]["n"]
            keys = [f"n={names[i]}" for i in draw_n_view(ax["pr"], n_read)]
        n_handles = n_key(n_read) if n_read is not None else []
        rendered_ntm = {}
        ntm_outline_regions = {}
        for band in (high, low):
            project(
                ax["pr"],
                band,
                projected[band.name][mode_tags.AE],
                mode_tags.AE,
                spans[mode_tags.AE],
            )
            rendered_ntm[band.name] = project(
                ax["pr"],
                band,
                projected[band.name][mode_tags.NTM],
                mode_tags.NTM,
                spans[mode_tags.NTM],
                edge=True,
                record=ntm_outline_regions.setdefault(band.name, {}),
                full=whole_ntm[band.name],
            )
        ae_annotation = (annotations or {}).get("ae_label")
        strip = ax["crashes"]
        strip.set_facecolor("#222222")
        strip.set_ylim(0, 1)
        strip.set_yticks([])
        strip.tick_params(bottom=False)
        for side in strip.spines.values():
            side.set_visible(False)
        strip.vlines(
            crashes,
            0.08,
            0.92,
            color=EVENT_COLOURS[mode_tags.SAWTOOTH],
            lw=0.7,
            linestyle="dotted",
        )
        strip.set_ylabel(
            "ECE candidates", rotation=0, ha="right", va="center", labelpad=3
        )
        if len(crashes):
            strip.text(
                0.015,
                0.5,
                "ECE-supported crash candidates",
                transform=strip.transAxes,
                color="white",
                fontsize=FONT,
                ha="left",
                va="center",
            )
        if not len(crashes):
            strip.set_visible(False)

        # D-alpha: the ELM label's span and spikes, the confinement regimes
        elm_key = "edge_localized_mode"
        elm_spans = present_spans(by_key[elm_key])
        largest_dalpha_peak_ms = None
        first_large_peak_before_expert_ms = None
        expert_elm_start_ms = None
        if da_sig.rows:
            read = da_sig.rows[0]
            cols = np.flatnonzero((read.centres >= t0) & (read.centres < t1))
            if len(cols):
                i = cols[np.argmax(read.values[1][0, cols])]
                largest_dalpha_peak_ms = float(read.centres[i])
                starts = [a for a, b in elm_spans if a < t1 and b > t0]
                expert_elm_start_ms = min(starts) if starts else None
                if (
                    starts
                    and by_key[elm_key].source.tier == lf.SILVER
                    and largest_dalpha_peak_ms < min(starts)
                ):
                    first_large_peak_before_expert_ms = (
                        min(starts) - largest_dalpha_peak_ms
                    )
        uncertain_elm = [
            (r.t_start, r.t_end)
            for r in by_key[elm_key].rows
            if r.category == UNCERTAIN
        ]
        peaks = np.array([])
        visible = []
        elm_boxes = []
        peak_artist = None
        elm_chip = None
        da = ax["da_pr"]
        if da_sig.rows:
            trace(da, da_sig.rows[0])
            top = float(np.max(da_sig.rows[0].values[1]))
            da.set_ylim(0, top * 3.0)
            peaks = elm_peaks(da_sig.rows[0], elm_spans)
            if len(peaks):
                (peak_artist,) = da.plot(
                    peaks,
                    np.full(len(peaks), top),
                    "v",
                    color=EVENT_COLOURS[elm_key],
                    ms=2.2,
                    mew=0,
                )
            for a, b in elm_spans:
                box = Rectangle(
                    (a, 0),
                    b - a,
                    top * 1.45,
                    fill=False,
                    ec=EVENT_COLOURS[elm_key],
                    lw=0.8,
                )
                da.add_patch(box)
                elm_boxes.append(box)
            for a, b in uncertain_elm:
                box = Rectangle(
                    (a, 0),
                    b - a,
                    top * 1.45,
                    fc="#eeeeee",
                    ec="#aaaaaa",
                    hatch="////",
                    lw=0.5,
                    alpha=0.55,
                    zorder=0.5,
                )
                da.add_patch(box)
                elm_boxes.append(box)
            visible = [
                (max(a, t0), min(b, t1)) for a, b in elm_spans if a < t1 and b > t0
            ]
            if not visible:
                visible = [
                    (max(a, t0), min(b, t1))
                    for a, b in uncertain_elm
                    if a < t1 and b > t0
                ]
            if visible:
                a, _ = max(visible, key=lambda s: s[1] - s[0])
                elm_chip = da.text(
                    a + 20, top * 1.55, "ELMs", fontsize=FONT,
                    va="bottom", ha="left", color=INK,
                )  # fmt: skip
        da.set_yticks([])
        da.set_ylabel(
            "D-alpha\n(a.u.)", rotation=0, ha="right", va="center", labelpad=3
        )
        regime_names = {1: "H-mode", 2: "L-mode", 3: "QH-mode", 4: "WPQH-mode"}
        shown = set()
        lmode_inferred = False
        regime_texts = []
        regime_regions = {}
        for r in by_key["confinement"].rows:
            category = r.category
            label = regime_names.get(category)
            # The L-H detector's pre-transition interval is explicitly a low
            # confinement cue; retain its binary H-mode categories in the track.
            if (
                by_key["confinement"].spec.title
                == figure_sources.BINARY_CONFINEMENT_TITLE
                and category == ABSENT
            ):
                later_h = any(
                    h.category == PRESENT and h.t_start >= r.t_end - 1e-6
                    for h in by_key["confinement"].rows
                )
                if later_h:
                    category = 2
                    label = "L (inferred)"
                    lmode_inferred |= r.t_end > t0 and r.t_start < t1
            if category in regime_names:
                da.axvspan(r.t_start, r.t_end, color=CLASS_COLOURS[category],
                           alpha=0.2, lw=0, zorder=0)  # fmt: skip
                if r.t_end > t0 and r.t_start < t1:
                    shown.add(category)
                    a, b = max(r.t_start, t0), min(r.t_end, t1)
                    regime_regions.setdefault(category, []).append((a, b, label))
        for regions in regime_regions.values():
            # Adjacent source rows shade one continuous region; merge only
            # their label-placement bounds, preserving all original rows.
            continuous = []
            for a, b, label in sorted(regions):
                if continuous and a <= continuous[-1][1] + 1e-6:
                    continuous[-1][1] = max(continuous[-1][1], b)
                else:
                    continuous.append([a, b, label])
            a, b, label = max(continuous, key=lambda r: r[1] - r[0])
            text = da.text(
                (a + b) / 2, 0.98, label,
                transform=da.get_xaxis_transform(), fontsize=FONT,
                color="#444444", va="top", ha="center",
            )  # fmt: skip
            fig.draw_without_rendering()
            region_width = (
                da.transData.transform((b, 0))[0] - (da.transData.transform((a, 0))[0])
            )
            if text.get_window_extent().width + 4 * fig.dpi / 72 > region_width:
                text.set_text(label.replace(" (inferred)", "\ninferred"))
            regime_texts.append((text, a, b))
            if elm_chip is not None:
                clear_of(text, [elm_chip])

        # ---- label tracks
        track_source_texts = []
        titles = {
            mode_tags.AE: "AE", mode_tags.NTM: "NTM", mode_tags.SAWTOOTH: "sawtooth",
            elm_key: "ELMs", "confinement": by_key["confinement"].spec.title,
        }  # fmt: skip
        for a, track in zip(track_axes, display_tracks, strict=True):
            key = track.spec.key
            track_bars(a, track, EVENT_COLOURS.get(key, "#888888"),
                       CLASS_COLOURS if key == "confinement" else None)  # fmt: skip
            a.set_ylabel(titles[key], rotation=0, ha="right", va="center", labelpad=3)
            tier = "" if track.source is None else TIER_NAMES[track.source.tier]
            if key == "confinement" and track.source is not None:
                tier = figure_sources.confinement_row_source(track, (t0, t1)) or tier
            if key == mode_tags.NTM and tier == "detector":
                tier = "detector (suggestions)"
            if (
                key == mode_tags.SAWTOOTH
                and track.source is not None
                and track.source.what.startswith("physics")
            ):
                tier = figure_sources.sawtooth_row_source(
                    figure_sources.state_intervals(track, (t0, t1)), saw_guard
                )
            # Extra lines hang below the row; the first stays on its centre.
            first_line_px = 0.0
            if "\n" in tier:
                probe = a.text(0, 0, tier.split("\n")[0], fontsize=FONT)
                first_line_px = probe.get_window_extent().height
                probe.remove()
            row_px = a.get_window_extent().height
            source_text = a.text(
                1.008, 0.5 + 0.5 * first_line_px / row_px,
                tier, transform=a.transAxes, fontsize=FONT, linespacing=1.0,
                va="top" if "\n" in tier else "center", ha="left", color="#444444",
            )  # fmt: skip
            track_source_texts.append((key, source_text))
        track_axes[-1].tick_params(labelbottom=True, bottom=True)
        track_axes[-1].spines["bottom"].set_visible(True)
        track_axes[-1].set_xlabel("time (ms)", labelpad=1)

        # group headings
        for name, lines in HEADINGS.items():
            pos = ax[name].get_position()
            for i, text in enumerate(lines):
                bold = "bold" if text == lines[0] else "normal"
                y = pos.y1 - (i + 0.5) * pos.height / len(lines)
                fig.text(pos.x0, y, text, fontsize=FONT, fontweight=bold,
                         va="center", ha="left")  # fmt: skip

        colours, display_keys = draw_legends(
            fig,
            ax,
            display_tracks,
            by_key,
            projected,
            crashes,
            n_handles,
            visible,
            peaks,
            t0,
            t1,
            ntm_dashed=any(
                r.get("dashed_regions", 0) for r in ntm_outline_regions.values()
            ),
        )
        fig.draw_without_rendering()
        ae_label = None
        ae_chip = None
        if ae_annotation and projected["wide"][mode_tags.AE].any():
            ae_label = next(
                text
                for key in fig.legends
                for text in key.texts
                if text.get_text().startswith("AE (detector-positive")
            )
            ae_chip = ax["pr"].text(
                ae_annotation["time_ms"],
                ae_annotation["frequency_khz"],
                "AE",
                fontsize=FONT,
                color="black",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "fc": EVENT_COLOURS[mode_tags.AE],
                    "ec": "white",
                    "lw": 0.5,
                },
                zorder=9,
            )
            fig.draw_without_rendering()
        n_legend = next(
            (
                key
                for key in fig.legends
                if key.get_title().get_text().startswith("toroidal mode number")
            ),
            None,
        )
        layout_record = {
            "frequency_axis": {
                "scale": "linear",
                "band_khz": list(ax["pr"].get_ylim()),
                "scale_breaks_khz": [],
                "ticks_khz": ax["pr"].get_yticks().tolist(),
                "raw_spectrogram": "wide pass, one colour scale, 0-250 kHz",
                "mask_zoom_pass_khz": [0.0, ZOOM_TOP_KHZ],
                "mask_wide_pass_khz": [ZOOM_TOP_KHZ, TOP_KHZ],
                "ae_ntm_split_khz": SPLIT_KHZ,
                "n_view_top_khz": N_VIEW_KHZ,
            },
            "n_view_height_in": ax["pr"].get_position().height
            * HEIGHT_IN
            * N_VIEW_KHZ
            / TOP_KHZ,
            "ece_candidate_key_placement": "in strip" if len(crashes) else None,
            "frequency_panels": {
                name: {
                    "band_khz": list(ax[name].get_ylim()),
                    "height_in": ax[name].get_position().height * HEIGHT_IN,
                    "bounds": list(ax[name].get_position().extents),
                    "ticks_khz": ax[name].get_yticks().tolist(),
                }
                for name in ("raw", "pr")
            },
            "frequency_tick_bounds": {
                prefix: [
                    {
                        "text": text.get_text(),
                        "bounds": list(
                            text.get_window_extent()
                            .transformed(fig.transFigure.inverted())
                            .extents
                        ),
                    }
                    for text in ax[prefix].get_yticklabels()
                ]
                for prefix in ("raw", "pr")
            },
            "regime_text_bounds": [
                {
                    "text": text.get_text(),
                    "bounds": list(
                        text.get_window_extent()
                        .transformed(fig.transFigure.inverted())
                        .extents
                    ),
                    "region_x_bounds": [
                        (da.transData.transform((x, 0))[0] - fig.bbox.x0)
                        / fig.bbox.width
                        for x in (a, b)
                    ],
                }
                for text, a, b in regime_texts
            ],
            "track_source_text_bounds": [
                {
                    "track": key,
                    "text": text.get_text(),
                    "bounds": list(
                        text.get_window_extent()
                        .transformed(fig.transFigure.inverted())
                        .extents
                    ),
                }
                for key, text in track_source_texts
            ],
            "ntm_key_black_swatch": bool(projected["zoom"][mode_tags.NTM].any()),
            "n_key_bounds": None
            if n_legend is None
            else list(
                n_legend.get_window_extent()
                .transformed(fig.transFigure.inverted())
                .extents
            ),
            "ae_in_panel_label": None
            if ae_chip is None
            else {
                "text": ae_chip.get_text(),
                "anchor_ms_khz": [
                    ae_annotation["time_ms"],
                    ae_annotation["frequency_khz"],
                ],
                "bounds": list(
                    ae_chip.get_bbox_patch()
                    .get_window_extent()
                    .transformed(fig.transFigure.inverted())
                    .extents
                ),
            },
            "ae_margin_label": None
            if ae_label is None
            else {
                "text": ae_label.get_text(),
                "bounds": list(
                    ae_label.get_window_extent()
                    .transformed(fig.transFigure.inverted())
                    .extents
                ),
            },
            "elm_box_bounds": [
                list(
                    box.get_window_extent()
                    .transformed(fig.transFigure.inverted())
                    .extents
                )
                for box in elm_boxes
                if box.get_x() < t1 and box.get_x() + box.get_width() > t0
            ],
            "dalpha_peak_box_gap_pt": None
            if peak_artist is None
            else (
                da.transData.transform((t0, top * 1.45))[1]
                - da.transData.transform((t0, top))[1]
            )
            * 72
            / fig.dpi
            - peak_artist.get_markersize() / 2
            - 0.4,
            "legend_labels": [t.get_text() for key in fig.legends for t in key.texts],
            "present_chip_colours": colours,
            "heading_and_legend_text_bounds": [
                {
                    "text": text.get_text(),
                    "font_pt": text.get_fontsize(),
                    "bounds": [
                        text.get_window_extent().x0 / fig.bbox.width,
                        text.get_window_extent().y0 / fig.bbox.height,
                        text.get_window_extent().x1 / fig.bbox.width,
                        text.get_window_extent().y1 / fig.bbox.height,
                    ],
                }
                for text in [
                    *fig.texts,
                    *[text for key in fig.legends for text in key.texts],
                    *[
                        key.get_title()
                        for key in fig.legends
                        if key.get_title().get_text()
                    ],
                ]
            ],
        }
        paths_out = save_figure(fig, stem, png_dpi)

    tags_count = {
        e: sum(e in b.tags for b in blobs_high + blobs_low)
        for e in (mode_tags.AE, mode_tags.NTM, mode_tags.SAWTOOTH)
    }
    track_records = {t.spec.key: track_record(t, (t0, t1)) for t in tracks}
    if saw is not None:
        track_records[mode_tags.SAWTOOTH]["source_files"] = [
            {"path": str(saw.file), "sha256": sha256_of(saw.file)}
        ]
        track_records[mode_tags.SAWTOOTH]["display_intervals_ms"] = (
            figure_sources.state_intervals(display_saw, (t0, t1))
            if show_sawtooth
            else []
        )
        track_records[mode_tags.SAWTOOTH]["display_merge"] = {
            "minimum_duration_ms": figure_sources.SAWTOOTH_DISPLAY_MIN_MS,
            "rule": "exact source states clipped to view; no smoothing; "
            "crash ticks unchanged",
            "changes": saw_changes if show_sawtooth else [],
        }
    late_absent = np.zeros(len(high.t), bool)
    active_ends = [b for a, b in spans[mode_tags.AE] if a < t1 and b > t0]
    if active_ends:
        late_absent = high.t >= max(active_ends)
        late_absent &= mode_tags.present_columns(
            high.t,
            [
                (r.t_start, r.t_end)
                for r in by_key[mode_tags.AE].rows
                if r.category == ABSENT
            ],
        )
    late = figure_sources.late_untagged_lines(high.lit, high.t, high.f, late_absent)
    return {
        "tracks": track_records,
        "figure": [str(p) for p in paths_out],
        "figure_sha256": {str(p): sha256_of(p) for p in paths_out},
        "blobs": {
            "wide_above_split": len(blobs_high),
            "zoom_below_split": len(blobs_low),
            "tagged": tags_count,
            "untagged": sum(not b.tags for b in blobs_high + blobs_low),
        },
        "elm_peaks_in_label": len(peaks),
        "elm_peak_times_ms": peaks.tolist(),
        "first_large_peak_before_expert_ms": first_large_peak_before_expert_ms,
        "largest_dalpha_peak_ms": largest_dalpha_peak_ms,
        "expert_elm_start_ms": expert_elm_start_ms,
        "lmode_inferred": lmode_inferred,
        "elm_hmode_conflicts_ms": [
            [max(a, r.t_start, t0), min(b, r.t_end, t1)]
            for a, b in elm_spans
            for r in by_key["confinement"].rows
            if by_key["confinement"].spec.title
            == figure_sources.BINARY_CONFINEMENT_TITLE
            and r.category == ABSENT
            and min(b, r.t_end, t1) > max(a, r.t_start, t0)
        ],
        "elm_qh_overlaps_ms": figure_sources.elm_qh_overlaps(
            elm_spans,
            by_key["confinement"].rows,
            (t0, t1),
            by_key["confinement"].spec.title == figure_sources.CONFINEMENT_TITLE,
        ),
        "ae_physical_review_caveat": (annotations or {}).get(
            "ae_physical_review_caveat"
        ),
        "elm_uncertain_spans_ms": uncertain_elm,
        "elm_crowd_spans_ms": [
            {"span_ms": [r.t_start, r.t_end], "category": r.category}
            for r in by_key[elm_key].rows
            if r.crowd == 1
            and r.category in (PRESENT, UNCERTAIN)
            and r.t_end > t0
            and r.t_start < t1
        ],
        "sawtooth_strip_shown": bool(len(crashes)),
        "sawtooth_track_shown": show_sawtooth,
        "sawtooth_visibility_rule": "show every selected track, including uncertain "
        "and unassessed states",
        "display_state_keys": display_keys,
        "layout": layout_record,
        "catalog_sawtooth_frame_model_shown": False,
        "late_untagged_high_frequency": late,
        "n2_harmonic_consistent": figure_sources.harmonic_consistent(harmonic),
        "n3_harmonic_consistent": figure_sources.harmonic_consistent(harmonic3),
        "harmonic_support": harmonic,
        "harmonic3_support": harmonic3,
        "persistent_line_rows": {
            name: int(
                (
                    np.asarray(z[f"{name}_row_lit"]) > mode_tags.PERSISTENT_ROW_SHARE
                ).sum()
            )
            for name in ("wide", "zoom")
        },
        "ntm_outline_display": {
            "rule": "display only: an outlined region needs at least min_px "
            "print pixels (150 dpi, after max pooling and hole filling); "
            "smaller regions are left unoutlined; tags and counts unchanged",
            "min_px": NTM_OUTLINE_MIN_PX,
            "omitted_fragments": sum(
                r.get("omitted_fragments", 0) for r in ntm_outline_regions.values()
            ),
            "dashed_rule": "a thin dashed outline of the rest of a tagged "
            "component's measured n=1/2 support (all time, NTM band) where the "
            "detector-positive clip cuts it; display only",
            "dashed_regions": sum(
                r.get("dashed_regions", 0) for r in ntm_outline_regions.values()
            ),
            "bands": ntm_outline_regions,
        },
        "n3_components_unoutlined": sum(b.dominant_n == 3 for b in blobs_low),
        "n3_outline_rule": "both dominant n and outlined support must be n=1 or 2",
        "n_map": None
        if n_read is None
        else {
            "kept_share": n_kept,
            "keys": [*keys, "other n"],
            "support_khz": list(n_read.extent[2:]),
            "outside_support": "unknown; cannot supply dominant n evidence",
        },
        "ntm_dominant_n": {
            str(n): sum(
                mode_tags.NTM in b.tags and b.dominant_n == n
                for b in blobs_high + blobs_low
            )
            for n in (1, 2)
        },
        "ae_boxes_ms_khz": [],
        "sawtooth_crashes": {**crash_record, "drawn_times_ms": crashes.tolist()},
        "projection_audit": {
            band.name: {
                e: figure_sources.projection_audit(
                    projected[band.name][e],
                    band.t,
                    band.f,
                    [
                        (r.t_start, r.t_end)
                        for r in by_key[e].rows
                        if r.category == PRESENT
                    ],
                    mode_tags.BANDS[e],
                )
                for e in (mode_tags.AE, mode_tags.NTM)
            }
            for band in (low, high)
        },
        "ntm_measured_pixel_audit": {
            band.name: {
                "outside_measured_n": int(
                    (
                        rendered_ntm[band.name] & ~np.isfinite(band.n_map[band.rows])
                    ).sum()
                )
                if band.n_map is not None
                else int(rendered_ntm[band.name].sum()),
                "above_n_view_band": int(
                    rendered_ntm[band.name][band.f > N_VIEW_KHZ].sum()
                ),
                "rendered_outline_pixels": int(rendered_ntm[band.name].sum()),
                "measured_n3_outline_pixels": int(
                    (rendered_ntm[band.name] & (band.n_map[band.rows] == 3)).sum()
                )
                if band.n_map is not None
                else 0,
            }
            for band in (low, high)
        },
        "regimes_shown": sorted(regime_names[c] for c in shown),
        "stores": {
            e: None if p is None else {"path": str(p), "sha256": sha256_of(p)}
            for e, p in stores.items()
        },
    }


def draw_n_view(ax, read) -> list[int]:
    """Full-colour hues over 0-30 kHz; brightness never encodes n."""
    modes = read.meta["modes"]
    palette = np.array([to_rgb(N_COLOURS.get(n, N_OTHER)) for n in modes["n"]])
    codes = read.values.astype(np.int64)
    t0, t1, f0, f1 = read.extent
    keep = round((min(f1, N_VIEW_KHZ) - f0) / (f1 - f0) * codes.shape[0])
    lit = codes[:keep] >= len(modes["n"])  # level 0: no mode, left clear
    rgb = palette[codes[:keep] % len(palette)]
    rgba = np.concatenate([rgb, lit[..., None]], axis=-1)
    ax.imshow(
        rgba,
        origin="lower",
        aspect="auto",
        extent=(t0, t1, f0, f0 + (f1 - f0) * keep / codes.shape[0]),
        interpolation="nearest",
        zorder=2,
    )
    return n_seen(read)


def n_seen(read) -> list[int]:
    """Every keyed n drawn, however small its share; all others are purple."""
    modes = read.meta["modes"]
    k = len(modes["n"])
    codes = read.values.astype(np.int64)
    lit = codes[codes >= k] % k
    seen = set(lit.tolist())
    return [i for i, n in enumerate(modes["n"]) if n in N_COLOURS and i in seen]


def n_key(read) -> list[Patch]:
    modes = read.meta["modes"]
    return [
        Patch(fc=N_COLOURS[modes["n"][i]], lw=0, label=f"n={modes['n'][i]}")
        for i in n_seen(read)
    ] + [Patch(fc=N_OTHER, lw=0, label="other n")]


def save_figure(fig: Figure, stem: Path, png_dpi: int = DPI_PNG) -> list[Path]:
    out = []
    for suffix, dpi in ((".pdf", DPI_PDF), (".png", png_dpi)):
        path = Path(stem).with_suffix(suffix)
        extra = {"metadata": {"CreationDate": None}} if suffix == ".pdf" else {}
        with atomic_path(path) as tmp:
            fig.savefig(tmp, format=suffix[1:], dpi=dpi, **extra)
        out.append(path)
    return out


# ------------------------------------------------------------------------ main


def track_record(track: lf.Track, window=None) -> dict | None:
    if track.source is None:
        return None
    present = present_spans(track)
    metadata = Path(track.file).with_suffix(".meta.json")
    meta = json.loads(metadata.read_text()) if metadata.is_file() else {}
    model = meta.get("method", track.source.what)
    segments = getattr(track, "segments", None)
    if segments is not None:
        model = meta.get("model", model)
    physics = (
        json.loads(Path(track.file).read_text())
        if track.spec.key == mode_tags.SAWTOOTH and Path(track.file).suffix == ".json"
        else {}
    )
    state_rows = figure_sources.state_intervals(track, window or (-math.inf, math.inf))
    ae_threshold = figure_sources.AE_THRESHOLD
    ae_bin_ms = (
        25.0
        if track.spec.key == mode_tags.AE and (track.source.what.startswith("ae-ours"))
        else None
    )
    if track.spec.key == mode_tags.AE and track.source.run is not None:
        from labeler.paper.shots import Model
        from labeler.scoring.frames import FRAME_MS

        model = "CO2 xpower frame model (80-250 kHz)"
        ae_threshold = Model.load(Path(track.file), {}).threshold
        ae_bin_ms = FRAME_MS
    performance = None
    if track.spec.key == mode_tags.NTM and track.source.tier == lf.GENERATED:
        evaluation = Path(meta["evaluation"])
        evaluated = json.loads(evaluation.read_text())
        performance = {
            **evaluated["scores"][meta["method"]],
            "evaluation": str(evaluation),
            "evaluation_sha256": sha256_of(evaluation),
            "bin_ms": evaluated["bin_ms"],
            "pool": evaluated["pool"],
            "bar_criteria": evaluated["bar_criteria"],
        }
    return {
        "tier": track.source.tier,
        "performance": performance,
        "title": track.spec.title,
        "what": track.source.what,
        "path": str(track.file),
        "sha256": sha256_of(track.file) if Path(track.file).is_file() else None,
        "rows": len(track.rows),
        "model": model,
        "decision_threshold": (
            {
                mode_tags.AE: ae_threshold,
                mode_tags.NTM: figure_sources.NTM_THRESHOLD,
                mode_tags.SAWTOOTH: figure_sources.SAWTOOTH_THRESHOLD,
            }.get(track.spec.key)
            if track.source.tier == lf.GENERATED
            else None
        ),
        "crowd_semantics": (
            "iscrowd=1 identifies the expert crowd lane; category 1 stays present, "
            "category 2 stays uncertain (hatching with a crowd marker)"
            if track.spec.key == "edge_localized_mode"
            else None
        ),
        "primary_bars": meta.get("bar"),
        "temporal_bin_ms": ae_bin_ms,
        "metadata": str(metadata) if metadata.is_file() else None,
        "metadata_sha256": sha256_of(metadata) if metadata.is_file() else None,
        "states": track.spec.states,
        "state_intervals_ms": state_rows,
        "density_guard": physics.get("density_guard"),
        "present_spans_ms": [[round(a, 1), round(b, 1)] for a, b in present],
        **(
            {}
            if segments is None
            else {
                "segments": list(segments),
                "roster": {
                    key: meta.get(key)
                    for key in (
                        "producer",
                        "status",
                        "made_at",
                        "git_sha",
                        "model",
                        "segmentation",
                        "categories",
                        "columns_meaning",
                    )
                },
            }
        ),
    }


def draft_caption(shot: int, records, drawn) -> str:
    text = figure_sources.caption(shot, records, drawn)
    text = re.sub(r"\bn=([0-9/]+)", r"$n=\1$", text)
    text = text.replace("mode number n ", "mode number $n$ ")
    text = text.replace("≥", r"$\geq$").replace("≤", r"$\leq$")
    text = text.replace("–", "--")
    text = text.replace("→", r"$\rightarrow$")
    text = text.replace("<60", "$<60$")
    label = f"fig:interpreter-{shot}"
    return f"\\caption{{{text}}}\n\\label{{{label}}}\n"


def main(argv=None) -> int:
    import torch

    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shot", type=int, default=199563)
    parser.add_argument(
        "--tmin", "--start", type=float, help="ms; supply both bounds or neither"
    )
    parser.add_argument("--tmax", "--end", type=float)
    parser.add_argument("--out", type=Path, help="folder; default round4/fig1b")
    parser.add_argument(
        "--cache", type=Path, help="TokEye cache; default round4/fig1b/cache"
    )
    parser.add_argument("--device", default="cpu", help="TokEye's device")
    parser.add_argument("--cache-only", action="store_true", help="TokEye, no figure")
    parser.add_argument("--png-dpi", type=int, default=DPI_PNG, help="the PNG's dpi")
    parser.add_argument("--record", type=Path, help="also write the record here")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("docs/labeler/fig1_annotations.json"),
        help="shot-specific presentation and review annotations (JSON keyed by shot)",
    )
    parser.add_argument(
        "--sawtooth-crashes",
        "--sawtooth-source",
        dest="sawtooth_crashes",
        type=Path,
        help="read-only physics shot JSON, JSON directory, cohort shards or crash CSV",
    )
    parser.add_argument(
        "--sawtooth-labels",
        type=Path,
        help="same physics source as crashes (legacy alias; cannot replace "
        "independently)",
    )
    parser.add_argument("--ae-labels", type=Path, help="isolated ae-ours inference CSV")
    parser.add_argument(
        "--show-sawtooth",
        action="store_true",
        help="show local crash evidence even without a supplied physics source",
    )
    parser.add_argument(
        "--sawtooth-evidence",
        type=Path,
        help="full physics JSON evidence paired with cohort CSVs",
    )
    args = parser.parse_args(argv)
    if (args.tmin is None) != (args.tmax is None):
        parser.error("supply both --tmin/--start and --tmax/--end, or neither")
    if args.tmin is not None and not (
        math.isfinite(args.tmin) and math.isfinite(args.tmax) and args.tmin < args.tmax
    ):
        parser.error("time bounds must be finite and start must precede end")
    annotations = json.loads(args.annotations.read_text()).get(str(args.shot), {})
    clean_head = not subprocess.check_output(["git", "status", "--porcelain"]).strip()

    paths = Paths.from_env()
    out = args.out or paths.root / "round4" / "fig1b"
    cache = args.cache or paths.root / "round4" / "fig1b" / "cache"
    found = [c for c in lf.candidates(paths) if c.shot == args.shot]
    if not found:
        raise SystemExit(f"{args.shot}: not a non-blind cohort shot")
    (candidate,) = found
    cohort = read_cohort(cohort_path(paths))
    split = str(cohort.loc[cohort["shot"] == args.shot, "split"].iloc[0])
    if split == "test":
        raise SystemExit(f"{args.shot}: a test shot, not drawn")
    if args.tmin is None or args.tmax is None:
        default = PRESETS.get(
            args.shot, (float(candidate.window[0]), float(candidate.window[1]))
        )
        t0, t1 = default
    else:
        t0, t1 = args.tmin, args.tmax
    z, cache_file = tokeye_passes(
        paths, args.shot, t0 - CACHE_MARGIN_MS, t1 + CACHE_MARGIN_MS, cache, args.device
    )
    print(f"TokEye: {cache_file}")
    if args.cache_only:
        return 0
    saw_source = args.sawtooth_crashes or args.sawtooth_labels
    if (
        args.sawtooth_crashes
        and args.sawtooth_labels
        and args.sawtooth_crashes.resolve() != args.sawtooth_labels.resolve()
    ):
        raise SystemExit("sawtooth track and crashes must use the same source")
    tracks, co2 = shot_tracks(
        paths,
        candidate,
        saw_source,
        args.ae_labels,
        args.show_sawtooth or saw_source is not None,
    )
    out.mkdir(parents=True, exist_ok=True)
    drawn = draw(
        paths,
        candidate,
        tracks,
        z,
        t0,
        t1,
        out / STEM,
        args.png_dpi,
        saw_source,
        args.sawtooth_evidence,
        annotations,
    )
    records = drawn.pop("tracks")
    detector_training = {}
    for key in (mode_tags.AE, mode_tags.NTM):
        track = records.get(key) or {}
        if track.get("tier") != lf.GENERATED:
            continue
        if key == mode_tags.AE and track["what"].startswith("ae-ours"):
            training_file = Path(
                "src/labeler/models/d3d_ae_activity_seldnet/training_shots.txt"
            )
            trained_shots = [int(s) for s in training_file.read_text().split()]
            weights = json.loads(Path(track["metadata"]).read_text())["checkpoint"]
        elif key == mode_tags.AE:
            # The frame detector's own split: the shots it trained on.
            from labeler.ae.xpower.train import read_split

            weights = Path(track["path"])
            training_file = weights.parent / "split.csv"
            trained_shots = sorted(
                s for s, v in read_split(training_file).items() if v == "train"
            )
        else:
            meta = json.loads(Path(track["metadata"]).read_text())
            weights = meta.get("model") or meta["weights"]
            training_file = Path(weights).parent / "training.json"
            trained_shots = json.loads(training_file.read_text())["shots"]["train"]
        detector_training[key] = {
            "training_source": str(training_file),
            "training_source_sha256": sha256_of(training_file),
            "training_shots": trained_shots,
            "training_shot_count": len(trained_shots),
            "figure_shot_in_training": args.shot in trained_shots,
            "checkpoint": str(weights),
            "checkpoint_sha256": sha256_of(Path(weights)),
        }
    caption = draft_caption(args.shot, records, drawn)
    caption_file = out / "caption.tex"
    with atomic_path(caption_file) as tmp:
        Path(tmp).write_text(caption)
    appendix = figure_sources.appendix_notes(
        args.shot, records, drawn, detector_training
    )
    appendix_file = out / "appendix.txt"
    with atomic_path(appendix_file) as tmp:
        Path(tmp).write_text(appendix + "\n")
    checkpoint = roster.tokeye_file(paths)
    record = {
        "shot": args.shot,
        "year": candidate.year,
        "window_ms": [t0, t1],
        "split": split,
        "publication_suitability": annotations.get("publication_suitability", {}),
        "annotations": {
            "path": str(args.annotations),
            "sha256": sha256_of(args.annotations),
            "shot": annotations,
        },
        "detector_training": detector_training,
        "decision_thresholds": {
            "ae": figure_sources.AE_THRESHOLD,
            "ntm": figure_sources.NTM_THRESHOLD,
            "sawtooth": figure_sources.SAWTOOTH_THRESHOLD,
            "tokeye": mode_tags.PROB_THRESHOLD,
            "sawtooth_elm_veto_ms": figure_sources.ELM_VETO_MS,
            "ece_crash_match_ms": figure_sources.ECE_MATCH_MS,
        },
        "ae_threshold_scope": "AE threshold is the paper ae-ours operating point; "
        "earlier frame fallback uses its checkpoint threshold in tracks",
        "sawtooth_decision_rule": "core ECE drop with spatial inversion/heat pulse; "
        "only accepted physics states; explicit confidence >=0.6 if provided; "
        "deterministic accepted physics classes need no invented probability; "
        "veto inclusive +/-5ms D-alpha peaks or recorded ELM coincidence",
        "print_layout": {
            "width_in": PAGE_IN,
            "height_in": HEIGHT_IN,
            "minimum_font_pt": FONT,
            "include_at": "textwidth",
        },
        "tokeye": {
            "checkpoint": str(checkpoint),
            "sha256": sha256_of(checkpoint),
            "probe": f"{roster.GATE_GROUP} row {roster.GATE_ROW} ({roster.GATE_TITLE})",
            "threshold": mode_tags.PROB_THRESHOLD,
            "cache": str(cache_file),
            "cache_sha256": sha256_of(cache_file),
            "fingerprints": json.loads(str(z["fingerprints"])),
            "device": args.device,
            "corpus": str(paths.corpus_file(args.shot)),
        },
        "filter": {
            "mask": "coherent >= threshold and not transient >= threshold",
            "persistent_line_candidate_row_share": mode_tags.PERSISTENT_ROW_SHARE,
            "min_size_px": mode_tags.MIN_SIZE,
            "hole_area_px": mode_tags.HOLE_AREA,
            "components": "scipy.ndimage.label, 8-connected (3x3 structure)",
            "tag_rule": "component AND PRESENT time pixels AND event band; "
            "NTM dominant measured n in {1,2} AND pixel n in {1,2} <=30 kHz; "
            "sawtooth crash ticks, no mode tag",
            "ae_highlight_rule": "pixel tint only; no component bounding boxes",
            "outline_display_rule": "max pool measured n=1 or 2 support at "
            "150-dpi axes resolution; fill holes; solid orange #E69F00 outer "
            "contour, 0.9 pt; clip to raw time/band spans; display only: regions "
            f"under {NTM_OUTLINE_MIN_PX} print pixels are not outlined, tags "
            "unchanged; pixel audit covers outlined region support, not contour "
            "stroke",
            "projection_audit_rule": "enumerate projected coordinates against "
            "raw, end-exclusive intervals; independent of present_columns",
            "n_palette": N_COLOURS,
            "n_brightness_rule": "constant full colour for every measured n pixel",
            "raw_normalisation": "3rd/99.8th percentiles of the whole 0-250 kHz window, "
            "one colour scale",
            "unkeyed_n_colour": N_OTHER,
            "bands_khz": {
                k: [lo, None if math.isinf(hi) else hi]
                for k, (lo, hi) in mode_tags.BANDS.items()
            },
            "split_khz": SPLIT_KHZ,
            "frequency_axis": "linear 0-250 kHz, no scale breaks",
            "elm_peaks": f"D-alpha less a {ELM_WINDOW}-sample running median, "
            f"over {ELM_MADS} MADs",
        },
        "tracks": records,
        "ae_co2": None
        if co2 is None
        else {"path": str(co2), "sha256": sha256_of(co2) if co2.is_file() else None},
        "ae_ours_lookup": {
            "path": str(paths.labels_file(args.shot)),
            "exists": paths.labels_file(args.shot).is_file(),
            "supplied_predictions": str(args.ae_labels) if args.ae_labels else None,
            "selected_source_path": records[mode_tags.AE]["path"]
            if records[mode_tags.AE] is not None
            else None,
            "selection_order": [
                "expert review",
                "supplied ae-ours",
                "stored ae-ours",
                "interferometer frame fallback",
            ],
            "selected": any(
                t.spec.key == mode_tags.AE
                and t.source
                and t.source.what.startswith("ae-ours")
                for t in tracks
            ),
        },
        "caption": {"path": str(caption_file), "sha256": sha256_of(caption_file)},
        "appendix": {"path": str(appendix_file), "sha256": sha256_of(appendix_file)},
        "drawn": drawn,
        "git": git_sha(),
        "render_started_from_clean_head": clean_head,
        "render_code_sha256": {
            p: sha256_of(Path(p))
            for p in (
                "scripts/labeler/paper/fig_interpreter_tokeye.py",
                "src/labeler/paper/figure_sources.py",
                "src/labeler/paper/mode_tags.py",
            )
        },
    }
    with atomic_path(out / f"{STEM}.json") as tmp:
        Path(tmp).write_text(json.dumps(record, indent=1) + "\n")
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(args.record) as tmp:
            Path(tmp).write_text(json.dumps(record, indent=1) + "\n")
        with atomic_path(args.record.with_suffix(".caption.tex")) as tmp:
            Path(tmp).write_text(caption)
        with atomic_path(args.record.with_suffix(".appendix.txt")) as tmp:
            Path(tmp).write_text(appendix + "\n")
    print(json.dumps(drawn["blobs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
