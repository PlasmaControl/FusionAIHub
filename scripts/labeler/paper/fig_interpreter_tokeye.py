r"""fig_interpreter (the teaser): raw signals -> TokEye-processed -> labelled.

    PYTHONPATH=src pixi run --frozen -e labelmaker \
        python scripts/labeler/paper/fig_interpreter_tokeye.py \
        [--shot 201978] [--tmin MS --tmax MS] [--out DIR]

One non-blind cohort shot over a few seconds, in three groups on one time axis:

- **raw**: the Mirnov probe's spectrogram (MPI66M322D, 0-250 kHz, split at
  `FOLD_KHZ`: the zoom pass below, the wide pass above), D-alpha and NBI power;
- **processed**: the same spectrogram through TokEye (the network's mode mask,
  `labeler.paper.mode_tags`: transient burst removed, pickup removed,
  `skimage.morphology.remove_small_objects` and small-hole filling, connected
  components), the toroidal-n view (the review page's n map, gated by the same
  mask) in place of the 0-30 kHz band. Highlights intersect PRESENT label
  times with AE >=60 kHz or NTM <60 kHz; NTM requires dominant measured n=1/2.
  Optional ECE-verified, ELM-vetoed sawtooth crashes appear on a thin strip. D-alpha
  carries the ELM label's span and the D-alpha peaks in
  it, and the confinement regimes shade it;
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
containing LABELER_LABEL_TABLES. Missing required curated/fallback tables are
errors. Curated four-class regimes win; otherwise the D-alpha L-H table is
used, never the H-mode frame model. --sawtooth-source accepts read-only physics
JSONs, a JSON directory or cohort CSV shards; --sawtooth-evidence supplies full
JSON evidence for the reduced cohort CSV schema. Track and ticks share the
same source when --show-sawtooth is enabled. --ae-labels accepts isolated
paper-model inference predictions. The primary omits sawtooth (ECE density guard).
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
HEIGHT_IN = 5.5
DPI_PNG = 150
DPI_PDF = 300
#: The window's margin the TokEye cache keeps beyond the figure's, ms.
CACHE_MARGIN_MS = 100.0
#: Each shot's window, ms, where `--tmin`/`--tmax` are not given (the cohort
#: window otherwise): chosen on the shot's own labels, see the report.
PRESETS = {
    186636: (1300.0, 3900.0),
    201973: (1600.0, 3350.0),
    201978: (1500.0, 3300.0),
    203187: (1700.0, 3150.0),
}
#: The spectrogram is folded here: below it the zoom pass (0.12 kHz/bin, reliable to
#: about 55 kHz), above it the wide pass (0.49 kHz/bin, to 250 kHz).
FOLD_KHZ = 55.0
TOP_KHZ = 250.0
N_VIEW_KHZ = 30.0  # the n map's band
#: The columns an image is pooled to for the page (a 3 s wide pass has 12,000).
IMAGE_COLUMNS = 3000
#: The ELM peaks: D-alpha less its running median, over this many MADs.
ELM_WINDOW = 25
ELM_MADS = 4.0
#: Blue, cyan and blue-green (no event pink/orange/yellow); other n are grey.
N_COLOURS = {1: "#0072B2", 2: "#56B4E9", 3: "#009E73"}
N_OTHER = "#8C6BB1"
EVENT_COLOURS = {
    mode_tags.AE: "#CC79A7",
    mode_tags.NTM: "#E69F00",
    mode_tags.SAWTOOTH: "#F0E442",
    "edge_localized_mode": "#333333",
}
EVENT_NAMES = {
    mode_tags.AE: "AE",
    mode_tags.NTM: "NTM",
    mode_tags.SAWTOOTH: "sawtooth",
    "edge_localized_mode": "ELMs",
}
TIER_NAMES = {lf.SILVER: "expert", lf.LEGACY: "legacy", lf.GENERATED: "detector"}
#: The confinement regimes, in grey: darker is better confined.
REGIME_GREYS = {1: "#3c3c3c", 2: "#bdbdbd", 3: "#6e6e6e", 4: "#8a8a8a", 5: "#d0d0d0"}
ABSENT_GREY = "#e4e4e4"
HEADINGS = {
    "h_raw": ("Raw signals: magnetics spectrogram (Mirnov probe)",),
    "h_proc": (
        "TokEye-processed: label–mode coincidence; toroidal n below 30 kHz",
        (
            "mask → remove small objects / fill holes → components → "
            "clip to label times and band"
        ),
    ),
    "h_lab": ("Labels and suggestions; source at right",),
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
    """`run_tokeye`'s arrays, from `cache` where the shot's file there was made
    with the pinned checkpoint, else made and kept."""
    path = tokeye_file(paths, shot, t0, t1, cache)
    sha = sha256_of(roster.tokeye_file(paths))
    if path.is_file():
        with np.load(path) as z:
            if str(z["checkpoint_sha256"]) == sha:
                return {k: z[k] for k in z.files}, path
    # A shorter display can use the same full-record inference and row shares.
    # Reuse its immutable superset cache rather than changing model inputs.
    for existing in sorted(cache.glob(f"tokeye_{shot}_*.npz")):
        _, _, a, b = existing.stem.split("_")
        if float(a) <= t0 and float(b) >= t1:
            with np.load(existing) as z:
                if str(z["checkpoint_sha256"]) == sha:
                    return {k: z[k] for k in z.files}, existing
    out = run_tokeye(paths, shot, t0, t1, device)
    out["checkpoint_sha256"] = np.array(sha)
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
    if show_sawtooth and sawtooth_labels is not None:
        selected[mode_tags.SAWTOOTH] = figure_sources.sawtooth_track(
            paths, candidate.shot, sawtooth_labels, []
        )
    elif show_sawtooth:
        saw_spec = next(s for s in lf.TRACKS if s.key == mode_tags.SAWTOOTH)
        selected[mode_tags.SAWTOOTH] = lf.Track(saw_spec, None, None, ())
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
        self.k = max(1, len(self.t) // IMAGE_COLUMNS)
        lo_v, hi_v = np.percentile(self.raw, (3, 99.8))
        self.norm = np.clip((self.raw - lo_v) / (hi_v - lo_v), 0, 1)

    def extent(self) -> tuple[float, float, float, float]:
        dt = float(np.median(np.diff(self.t)))
        df = float(np.median(np.diff(self.f)))
        return (
            self.t[0] - dt / 2,
            self.t[-1] + dt / 2,
            self.f[0] - df / 2,
            self.f[-1] + df / 2,
        )

    def blobs(self, spans: dict, n_read=None) -> list[mode_tags.Blob]:
        """The blobs of this band's mask, tagged by `spans`' events."""
        found = mode_tags.blobs(self.lit_full, self.t, self.all_f)
        # Keep whole components for dominant-n evidence across the display fold.
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


def project(ax, band, mask, event, spans, edge=False):
    """Tint or a two-pixel outline pooled at the 150-dpi print resolution."""
    if not mask.any():
        return mask
    if edge:
        pos = ax.get_position()
        nf = min(len(band.f), max(1, round(pos.height * HEIGHT_IN * DPI_PNG)))
        nt = min(len(band.t), max(1, round(pos.width * PAGE_IN * DPI_PNG)))
        # Max pooling preserves narrow ridges when reduced to print pixels.
        ri = np.linspace(0, mask.shape[0], nf, endpoint=False).astype(int)
        ci = np.linspace(0, mask.shape[1], nt, endpoint=False).astype(int)
        display = np.maximum.reduceat(np.maximum.reduceat(mask, ri, axis=0), ci, axis=1)
        edge_mask = display & ~ndimage.binary_erosion(display)
        outline = ndimage.binary_dilation(edge_mask) & display
        # Expand the print-grid inner edge back to native pixels, then intersect
        # the measured-pixel mask. No stroke may invent n outside its support.
        row_bin = np.searchsorted(ri, np.arange(mask.shape[0]), side="right") - 1
        col_bin = np.searchsorted(ci, np.arange(mask.shape[1]), side="right") - 1
        shown = outline[np.ix_(row_bin, col_bin)] & mask
    else:
        shown = mask
    rgba = np.zeros((*shown.shape, 4), np.float32)
    rgba[..., :3] = to_rgb(EVENT_COLOURS[event])
    rgba[..., 3] = shown * (1.0 if edge else 0.65)
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


def leader(ax, text, xy, position, colour):
    ax.annotate(
        text,
        xy=xy,
        xytext=position,
        textcoords="axes fraction",
        fontsize=FONT,
        color="white",
        ha="left",
        va="center",
        bbox={"fc": "black", "ec": "none", "alpha": 0.8, "pad": 1.2},
        arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.8},
        zorder=9,
        annotation_clip=False,
    )


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
            hatch = "////" if r.category == 5 else None
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
            edgecolor=face,
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
    elif crash_source is not None:
        saw = figure_sources.sawtooth_track(
            paths, candidate.shot, crash_source, crashes
        )
    else:
        spec = by_key[mode_tags.SAWTOOTH].spec
        src = lf.Source(
            lf.GENERATED,
            "local ECE-verified crash detector",
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

    low = Band(z, "zoom", 0.0, FOLD_KHZ, t0, t1)
    high = Band(z, "wide", FOLD_KHZ, TOP_KHZ + 1, t0, t1)
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
    harmonic = figure_sources.harmonic_support(
        low.n_map[low.rows] if low.n_map is not None else None,
        low.lit & mode_tags.present_columns(low.t, spans[mode_tags.NTM])[None, :],
        low.t,
        low.f,
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
        "h_raw": 0.32, "raw_hi": 1.0, "raw_lo": 0.8, "g1": 0.1, "da_raw": 0.42,
        "g2": 0.07, "nbi": 0.42, "h_proc": 1.21 if len(crashes) else 0.94,
        "pr_hi": 1.0, "pr_lo": 0.8,
        "crashes": 0.16 if len(crashes) else 0.001,
        "g3": 0.1, "da_pr": 0.62, "h_lab": 0.40,
    }  # fmt: skip
    names = [*layout, *[f"track{i}" for i in range(len(tracks))]]
    heights = [
        *layout.values(),
        *[0.18 for t in tracks],
    ]
    with style():
        fig = Figure(figsize=(PAGE_IN, HEIGHT_IN))
        gs = fig.add_gridspec(
            len(heights),
            1,
            height_ratios=heights,
            hspace=0.0,
            left=0.14,
            right=0.895,
            top=0.985,
            bottom=0.12,
        )
        ax = {n: fig.add_subplot(gs[i]) for i, n in enumerate(names)}
        for n in ("h_raw", "g1", "g2", "h_proc", "g3", "h_lab"):
            ax[n].set_visible(False)
        track_axes = [ax[f"track{i}"] for i in range(len(tracks))]
        for n in (
            "raw_hi",
            "raw_lo",
            "da_raw",
            "nbi",
            "pr_hi",
            "pr_lo",
            "da_pr",
            "crashes",
        ):
            ax[n].set_xlim(t0, t1)
            style_axes(ax[n])
        for a in track_axes:
            a.set_xlim(t0, t1)
            style_axes(a)

        # ---- raw
        for hi, lo, band in (("raw_hi", "raw_lo", None), ("pr_hi", "pr_lo", None)):
            ax[hi].set_ylim(FOLD_KHZ, TOP_KHZ)
            ax[lo].set_ylim(0, FOLD_KHZ)
            ax[hi].set_yticks([100, 150, 200, 250])
            ax[lo].set_yticks([0, 20, 40, 55])
            ax[hi].spines["bottom"].set_visible(False)
            ax[hi].tick_params(bottom=False)
            ax[hi].set_ylabel("kHz", labelpad=2)
            ax[lo].set_ylabel("kHz", labelpad=2)
            # Break markers on both edges make the change of scale explicit.
            for x in (0, 1):
                for panel, y in ((hi, 0), (lo, 1)):
                    ax[panel].plot(
                        [x - 0.007, x + 0.007],
                        [y - 0.035, y + 0.035],
                        transform=ax[panel].transAxes,
                        color=INK,
                        lw=0.7,
                        clip_on=False,
                        zorder=10,
                    )
        draw_raw(ax["raw_hi"], high)
        draw_raw(ax["raw_lo"], low)
        label_box = {"fc": "black", "alpha": 0.5, "ec": "none", "pad": 1.0}
        for name, text in (("raw_lo", "0-55 kHz, finer resolution"),):
            ax[name].text(
                0.006, 0.94, text, transform=ax[name].transAxes, color="white",
                fontsize=FONT, va="top", ha="left", bbox=label_box,
            )  # fmt: skip
        if da_sig.rows:
            trace(ax["da_raw"], da_sig.rows[0])
        ax["da_raw"].set_ylabel(
            "D-alpha", rotation=0, ha="right", va="center", labelpad=3
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

        # ---- processed: the mask times the spectrogram, the n view below 30 kHz
        draw_processed(ax["pr_hi"], high)
        draw_processed(ax["pr_lo"], low)
        keys = []
        if n_read is not None:
            names = n_read.meta["modes"]["n"]
            keys = [f"n={names[i]}" for i in draw_n_view(ax["pr_lo"], n_read)]
        n_handles = n_key(n_read) if n_read is not None else []
        rendered_ntm = {}
        for panel, band in (("pr_hi", high), ("pr_lo", low)):
            project(
                ax[panel],
                band,
                projected[band.name][mode_tags.AE],
                mode_tags.AE,
                spans[mode_tags.AE],
            )
            rendered_ntm[band.name] = project(
                ax[panel],
                band,
                projected[band.name][mode_tags.NTM],
                mode_tags.NTM,
                spans[mode_tags.NTM],
                edge=True,
            )
        # Leaders identify representative structures, never claim seeding.
        ae_components, _ = ndimage.label(projected["wide"][mode_tags.AE])
        areas = np.bincount(ae_components.ravel())
        if len(areas) > 1:
            rr, cc = np.nonzero(ae_components == (np.argmax(areas[1:]) + 1))
            i = len(rr) // 2
            leader(
                ax["pr_hi"],
                "AE prediction"
                if by_key[mode_tags.AE].source.tier == lf.GENERATED
                else "AE interval",
                (high.t[cc[i]], high.f[rr[i]]),
                (0.55, 0.88),
                EVENT_COLOURS[mode_tags.AE],
            )
        ntm_n1 = [b for b in blobs_low if mode_tags.NTM in b.tags and b.dominant_n == 1]
        if ntm_n1:
            b = max(ntm_n1, key=lambda b: b.n_pix)
            rr, cc = b.component.rows, b.component.cols
            keep = mode_tags.present_columns(low.t[cc], spans[mode_tags.NTM])
            if low.n_map is not None:
                keep &= low.n_map[rr, cc] == 1
            if keep.any():
                i = np.flatnonzero(keep)[len(np.flatnonzero(keep)) // 2]
                leader(
                    ax["pr_lo"],
                    "n=1 mode; NTM prediction"
                    if by_key[mode_tags.NTM].source.tier == lf.GENERATED
                    else "n=1 mode; NTM label",
                    (low.t[cc[i]], low.all_f[rr[i]]),
                    (0.53, 0.88),
                    EVENT_COLOURS[mode_tags.NTM],
                )
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
        strip.set_ylabel("ECE crashes", rotation=0, ha="right", va="center", labelpad=3)
        if not len(crashes):
            strip.set_visible(False)

        # D-alpha: the ELM label's span and spikes, the confinement regimes
        elm_key = "edge_localized_mode"
        elm_spans = present_spans(by_key[elm_key])
        uncertain_elm = [
            (r.t_start, r.t_end)
            for r in by_key[elm_key].rows
            if r.category == UNCERTAIN
        ]
        peaks = np.array([])
        elm_chip = None
        da = ax["da_pr"]
        if da_sig.rows:
            trace(da, da_sig.rows[0])
            top = float(np.max(da_sig.rows[0].values[1]))
            da.set_ylim(0, top * 1.65)
            peaks = elm_peaks(da_sig.rows[0], elm_spans)
            if len(peaks):
                da.plot(peaks, np.full(len(peaks), top * 1.14), "v",
                        color=EVENT_COLOURS[elm_key], ms=2.2, mew=0)  # fmt: skip
            for a, b in elm_spans:
                da.add_patch(
                    Rectangle(
                        (a, 0),
                        b - a,
                        top * 1.25,
                        fill=False,
                        ec=EVENT_COLOURS[elm_key],
                        lw=0.8,
                    )
                )
            for a, b in uncertain_elm:
                da.add_patch(
                    Rectangle(
                        (a, 0),
                        b - a,
                        top * 1.25,
                        fc="#eeeeee",
                        ec="#aaaaaa",
                        hatch="////",
                        lw=0.5,
                        alpha=0.55,
                        zorder=0.5,
                    )
                )
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
                    a + 20, top * 1.30, "ELMs", fontsize=FONT,
                    va="bottom", ha="left", color=INK,
                )  # fmt: skip
        da.set_yticks([])
        da.set_ylabel("D-alpha", rotation=0, ha="right", va="center", labelpad=3)
        regime_names = {1: "H-mode", 2: "L-mode", 3: "QH-mode", 4: "WPQH-mode"}
        shown = set()
        regime_texts = []
        for r in by_key["confinement"].rows:
            if r.category in regime_names:
                da.axvspan(r.t_start, r.t_end, color=REGIME_GREYS[r.category],
                           alpha=0.2, lw=0, zorder=0)  # fmt: skip
                if r.t_end > t0 and r.t_start < t1 and r.category not in shown:
                    shown.add(r.category)
                    regime_texts.append(
                        da.text(max(r.t_start, t0) + 20, 0.94, regime_names[r.category],
                                transform=da.get_xaxis_transform(), fontsize=FONT,
                                color="#444444", va="top", ha="left")
                    )  # fmt: skip
        if elm_chip is not None and regime_texts:
            clear_of(elm_chip, regime_texts)

        # ---- label tracks
        titles = {
            mode_tags.AE: "AE", mode_tags.NTM: "NTM", mode_tags.SAWTOOTH: "sawtooth",
            elm_key: "ELMs", "confinement": by_key["confinement"].spec.title,
        }  # fmt: skip
        for a, track in zip(track_axes, tracks, strict=True):
            key = track.spec.key
            track_bars(a, track, EVENT_COLOURS.get(key, "#888888"),
                       REGIME_GREYS if key == "confinement" else None)  # fmt: skip
            a.set_ylabel(titles[key], rotation=0, ha="right", va="center", labelpad=3)
            tier = "" if track.source is None else TIER_NAMES[track.source.tier]
            if key == mode_tags.NTM and tier == "detector":
                tier = "unverified"  # The caption explains the magnetic suggestion.
            a.text(1.008, 0.5, tier, transform=a.transAxes, fontsize=FONT,
                   va="center", ha="left", color="#444444")  # fmt: skip
        track_axes[-1].tick_params(labelbottom=True, bottom=True)
        track_axes[-1].spines["bottom"].set_visible(True)
        track_axes[-1].set_xlabel("time (ms)", labelpad=1)

        # group headings
        for name, lines in HEADINGS.items():
            pos = ax[name].get_position()
            for i, text in enumerate(lines):
                bold = "bold" if text == lines[0] else "normal"
                y = pos.y1 - 0.028 - i * 0.024
                fig.text(pos.x0, y, text, fontsize=FONT, fontweight=bold,
                         va="bottom", ha="left")  # fmt: skip

        pos = ax["h_proc"].get_position()
        event_handles = [
            Patch(fc=EVENT_COLOURS[mode_tags.AE], alpha=0.65, label="AE ≥60 kHz"),
            Line2D(
                [], [], color=EVENT_COLOURS[mode_tags.NTM], ls="-", label="NTM <60 kHz"
            ),
            Patch(fc="none", ec=INK, lw=0.8, label="ELM intervals"),
            Line2D(
                [],
                [],
                color=INK,
                marker="v",
                ls="",
                ms=3,
                label="D-alpha peaks",
            ),
        ]
        mask_handle = Patch(
            fc="white", ec=INK, lw=0.5, label="TokEye mask, n not measured"
        )
        fig.legend(
            handles=[mask_handle, *n_handles],
            loc="lower left",
            ncols=1 + len(n_handles),
            frameon=False,
            fontsize=FONT,
            handlelength=1.0,
            columnspacing=0.9,
            handletextpad=0.3,
            borderpad=0,
            borderaxespad=0,
            bbox_to_anchor=(pos.x0, pos.y0 + (0.058 if len(crashes) else 0.030)),
        )
        fig.legend(
            handles=event_handles,
            loc="lower left",
            ncols=len(event_handles),
            frameon=False,
            fontsize=FONT,
            handlelength=1.1,
            columnspacing=0.9,
            handletextpad=0.3,
            borderaxespad=0,
            bbox_to_anchor=(pos.x0, pos.y0 + (0.030 if len(crashes) else 0.004)),
        )
        if len(crashes):
            fig.legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        color=EVENT_COLOURS[mode_tags.SAWTOOTH],
                        ls=":",
                        label="sawtooth crashes (ECE-verified)",
                    )
                ],
                loc="lower left",
                frameon=False,
                fontsize=FONT,
                borderaxespad=0,
                bbox_to_anchor=(pos.x0, pos.y0 + 0.004),
            )
        colours = [EVENT_COLOURS[e] for e in (mode_tags.AE, mode_tags.NTM)]
        if show_sawtooth:
            colours.append(EVENT_COLOURS[mode_tags.SAWTOOTH])
        handles = [
            tuple(Patch(fc=c, lw=0) for c in (*colours, INK)),
            Patch(fc="white", ec=INK, hatch="//////", lw=0.4),
            Patch(fc=ABSENT_GREY, lw=0),
            Patch(fc="white", ec="#999999", lw=0.5),
        ]  # fmt: skip
        fig.legend(
            handles,
            ["present", "uncertain", "absent",
             "blank: unassessed / unobservable"],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
            loc="lower center", ncols=4, frameon=False, fontsize=FONT,
            bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=2.5,
        )  # fmt: skip
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
                    label="expert crowd interval",
                )
            ],
            loc="lower center",
            frameon=False,
            fontsize=FONT,
            bbox_to_anchor=(0.5, 0.034),
        )
        paths_out = save_figure(fig, stem, png_dpi)

    tags_count = {
        e: sum(e in b.tags for b in blobs_high + blobs_low)
        for e in (mode_tags.AE, mode_tags.NTM, mode_tags.SAWTOOTH)
    }
    track_records = {t.spec.key: track_record(t) for t in tracks}
    if show_sawtooth:
        track_records[mode_tags.SAWTOOTH]["source_files"] = crash_record["files"]
    if show_sawtooth and crash_source is not None:
        track_records[mode_tags.SAWTOOTH]["path"] = str(crash_source)
        track_records[mode_tags.SAWTOOTH]["sha256"] = crash_record.get("sha256")
    return {
        "tracks": track_records,
        "figure": [str(p) for p in paths_out],
        "blobs": {
            "wide_above_fold": len(blobs_high),
            "zoom_below_fold": len(blobs_low),
            "tagged": tags_count,
            "untagged": sum(not b.tags for b in blobs_high + blobs_low),
        },
        "elm_peaks_in_label": len(peaks),
        "elm_peak_times_ms": peaks.tolist(),
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
        "sawtooth_omission_reason": (
            "ECE density guard"
            if candidate.shot == 201978 and not show_sawtooth
            else "optional source track omitted"
            if not show_sawtooth
            else None
        ),
        "view_description": (
            "follow the AE cascade, H-mode transition, ELM onset and low-frequency ridges"
            if candidate.shot == 201978
            else "follow aligned signals, modes and intervals"
        ),
        "n2_harmonic_consistent": harmonic["support_ms"]
        >= harmonic["minimum_support_ms"],
        "harmonic_support": harmonic,
        "n3_components_unoutlined": sum(b.dominant_n == 3 for b in blobs_low),
        "n3_outline_rule": "dominant n=3 does not meet the NTM n=1/2 rule",
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
    """Accessible remap over 0-30 kHz; every unkeyed n is grey."""
    modes = read.meta["modes"]
    palette = np.array([to_rgb(N_COLOURS.get(n, N_OTHER)) for n in modes["n"]])
    codes = read.values.astype(np.int64)
    t0, t1, f0, f1 = read.extent
    keep = round((min(f1, N_VIEW_KHZ) - f0) / (f1 - f0) * codes.shape[0])
    lit = codes[:keep] >= len(modes["n"])  # level 0: no mode, left clear
    brightness = 0.3 + 0.7 * (codes[:keep] // len(palette)) / (modes["levels"] - 1)
    rgb = palette[codes[:keep] % len(palette)] * brightness[..., None]
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
    """Every coloured n drawn, however small its share; all others are grey."""
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
    ] + [Line2D([], [], color=N_OTHER, marker="D", ls="", ms=4, label="other n")]


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


def track_record(track: lf.Track) -> dict | None:
    if track.source is None:
        return None
    present = present_spans(track)
    metadata = Path(track.file).with_suffix(".meta.json")
    meta = json.loads(metadata.read_text()) if metadata.is_file() else {}
    model = meta.get("method", track.source.what)
    ae_threshold = figure_sources.AE_THRESHOLD
    if track.spec.key == mode_tags.AE and track.source.run is not None:
        from labeler.paper.shots import Model

        model = "CO2 xpower frame model (80-250 kHz)"
        ae_threshold = Model.load(Path(track.file), {}).threshold
    return {
        "tier": track.source.tier,
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
        "metadata": str(metadata) if metadata.is_file() else None,
        "metadata_sha256": sha256_of(metadata) if metadata.is_file() else None,
        "states": track.spec.states,
        "present_spans_ms": [[round(a, 1), round(b, 1)] for a, b in present],
    }


def draft_caption(shot: int, records, drawn) -> str:
    text = figure_sources.caption(shot, records, drawn)
    text = re.sub(r"\bn=([0-9/]+)", r"$n=\1$", text)
    text = text.replace("≥", r"$\geq$").replace("<60", "$<60$")
    label = "fig:interpreter" if shot == 201978 else f"fig:interpreter-{shot}"
    return f"\\caption{{{text}}}\n\\label{{{label}}}\n"


def main(argv=None) -> int:
    import torch

    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shot", type=int, default=201978)
    parser.add_argument("--tmin", type=float, help="ms; the shot's preset window")
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--out", type=Path, help="folder; default round4/fig1")
    parser.add_argument("--cache", type=Path, help="TokEye cache; default <out>/cache")
    parser.add_argument("--device", default="cpu", help="TokEye's device")
    parser.add_argument("--cache-only", action="store_true", help="TokEye, no figure")
    parser.add_argument("--png-dpi", type=int, default=DPI_PNG, help="the PNG's dpi")
    parser.add_argument("--record", type=Path, help="also write the record here")
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
        help="same physics source as crashes (legacy alias; cannot replace independently)",
    )
    parser.add_argument("--ae-labels", type=Path, help="isolated ae-ours inference CSV")
    parser.add_argument(
        "--show-sawtooth",
        action="store_true",
        help="show physics categories and verified ticks for a suitable later shot",
    )
    parser.add_argument(
        "--sawtooth-evidence",
        type=Path,
        help="full physics JSON evidence paired with cohort CSVs",
    )
    args = parser.parse_args(argv)

    paths = Paths.from_env()
    out = args.out or paths.root / "round4" / "fig1"
    cache = args.cache or paths.root / "round4" / "fig1" / "cache"
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
        paths, candidate, saw_source, args.ae_labels, args.show_sawtooth
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
    )
    records = drawn.pop("tracks")
    caption = draft_caption(args.shot, records, drawn)
    caption_file = out / "caption.tex"
    with atomic_path(caption_file) as tmp:
        Path(tmp).write_text(caption)
    checkpoint = roster.tokeye_file(paths)
    record = {
        "shot": args.shot,
        "year": candidate.year,
        "window_ms": [t0, t1],
        "split": split,
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
            "NTM dominant measured n in {1,2} AND measured n pixel <=30 kHz; "
            "sawtooth crash ticks, no mode tag",
            "ae_highlight_rule": "pixel tint only; no component bounding boxes",
            "outline_display_rule": "max pool at 150-dpi axes resolution; "
            "erode for edge, expand one print pixel inward; re-intersect native "
            "measured-pixel mask; clip to raw time/band spans",
            "projection_audit_rule": "enumerate projected coordinates against "
            "raw, end-exclusive intervals; independent of present_columns",
            "n_palette": N_COLOURS,
            "unkeyed_n_colour": N_OTHER,
            "bands_khz": {
                k: [lo, None if math.isinf(hi) else hi]
                for k, (lo, hi) in mode_tags.BANDS.items()
            },
            "fold_khz": FOLD_KHZ,
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
        "drawn": drawn,
        "git": git_sha(),
    }
    with atomic_path(out / f"{STEM}.json") as tmp:
        Path(tmp).write_text(json.dumps(record, indent=1) + "\n")
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(args.record) as tmp:
            Path(tmp).write_text(json.dumps(record, indent=1) + "\n")
        with atomic_path(args.record.with_suffix(".caption.tex")) as tmp:
            Path(tmp).write_text(caption)
    print(json.dumps(drawn["blobs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
