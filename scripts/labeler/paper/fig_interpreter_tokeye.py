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
  mask) in place of the 0-30 kHz band, and each component tagged by the event
  whose label is present over it: AEs above 60 kHz, the tearing mode and the
  sawtooth below. D-alpha carries the ELM label's span and the D-alpha peaks in
  it, and the confinement regimes shade it;
- **labels**: one track per event on the shot's time axis, from the best tier
  that holds the shot (`labeler.paper.label_figure.TRACKS`): present, absent or
  blank (never assessed).

TokEye runs once per shot and window, on the CPU (or `--device cuda` in a CUDA
build: then `--cache-only` there, the drawing needs scikit-image), and is cached
under `--cache`. Writes `fig_interpreter.pdf` and `.png` (150 dpi) and the
record `fig_interpreter.json` beside them; `--record` copies the record where
the repo keeps it. A tag says that a mode and a label coincide in time and band,
not that the mode is that event.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch, Rectangle
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
    mode_palette,
)
from labeler.paper import PAGE_IN, mode_tags, roster, style
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
    201978: (1400.0, 4900.0),
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
#: A run of a label's blobs this close in time is one box, ms.
BOX_GAP_MS = 200.0
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
    "h_raw": ("Raw signals",),
    "h_proc": (
        "TokEye-processed, and tagged by the labels",
        (
            "TokEye mask → remove small objects, fill holes → connected components → "
            "tagged by the event label over them"
        ),
    ),
    "h_lab": ("Labels; source at right: expert review, legacy table or detector",),
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
    paths: Paths, candidate: lf.Candidate
) -> tuple[tuple[lf.Track, ...], Path | None]:
    """The catalog's tracks on `candidate`: `label_figure`'s, the AE generated
    from the corpus CO2, else from the raw cache; and the CO2 file the
    generated AE track ran over (None where the AE track is not generated)."""
    read = lf.read_sources(paths)
    read = lf.generate(paths, [candidate], read)  # a shot without corpus CO2: none
    tracks = lf.tracks_of(candidate.shot, read)
    co2 = None
    out = []
    for track in tracks:
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


def labelled_spans(track: lf.Track) -> list[tuple[float, float]]:
    return [(r.t_start, r.t_end) for r in track.rows if r.category in (1, 2)]


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
        lit[~rows] = False
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

    def blobs(self, spans: dict) -> list[mode_tags.Blob]:
        """The blobs of this band's mask, tagged by `spans`' events."""
        found = mode_tags.blobs(self.lit_full, self.t, self.all_f)
        return mode_tags.tag_blobs(found, spans)

    def tag_image(self, blobs: list[mode_tags.Blob], event: str) -> np.ndarray:
        """`event`'s tagged blobs as a boolean image like `lit`."""
        img = np.zeros(self.lit.shape, bool)
        for b in blobs:
            if event in b.tags:
                img[b.component.rows - self.row0, b.component.cols] = True
        return img


# ----------------------------------------------------------------------- draw


def style_axes(ax, bottom: bool = False) -> None:
    ax.tick_params(length=2, pad=1.5, labelsize=FONT)
    ax.tick_params(labelbottom=bottom)
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


def draw_processed(ax, band: Band, tagged: list[mode_tags.Blob], fill=()) -> None:
    """The mask times the spectrogram in grey; the blobs `fill` names the tags
    of are tinted in their event's colour (the first of `fill` that tags one)."""
    base = 0.25 + 0.75 * band.norm
    rgb = np.zeros((*band.lit.shape, 3), np.float32)
    rgb[band.lit] = base[band.lit][:, None]
    for b in tagged:
        for event in fill:
            if event in b.tags:
                rr, cc = b.component.rows - band.row0, b.component.cols
                tint = np.array(to_rgb(EVENT_COLOURS[event]), np.float32)
                rgb[rr, cc] = tint * (0.4 + 0.6 * base[rr, cc][:, None])
                break
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


def outline(ax, band: Band, mask: np.ndarray, colour: str, grow: int, lw=0.7) -> None:
    """The edge of `mask` (a boolean image like the band's), grown `grow` pixels."""
    if not mask.any():
        return
    grown = ndimage.binary_dilation(mask, iterations=grow)
    pooled = pool(grown.astype(np.float32), band.k, "max")
    t = pooled_axis(band.t, band.k)
    ax.contour(t, band.f, pooled, levels=[0.5], colors=[colour], linewidths=lw)


def merged_boxes(blobs: list[mode_tags.Blob], event: str) -> list[tuple]:
    """`event`'s blobs in runs closer than `BOX_GAP_MS`: each run's
    `(t0, t1, f0, f1)`."""
    mine = sorted((b for b in blobs if event in b.tags), key=lambda b: b.t0_ms)
    runs: list[list] = []
    for b in mine:
        if runs and b.t0_ms - runs[-1][1] <= BOX_GAP_MS:
            runs[-1] = [
                runs[-1][0],
                max(runs[-1][1], b.t1_ms),
                min(runs[-1][2], b.f0_khz),
                max(runs[-1][3], b.f1_khz),
            ]
        else:
            runs.append([b.t0_ms, b.t1_ms, b.f0_khz, b.f1_khz])
    return [tuple(r) for r in runs]


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


def draw_boxes(ax, boxes, event, ylim, chip_y) -> None:
    """Dashed boxes round `boxes` (padded, kept in `ylim`) and one chip naming
    `event` at the left of the largest, at height `chip_y`."""
    colour = EVENT_COLOURS[event]
    for t0, t1, f0, f1 in boxes:
        lo, hi = max(f0 - 2.0, ylim[0] + 0.5), min(f1 + 2.0, ylim[1] - 0.5)
        ax.add_patch(
            Rectangle(
                (t0 - 8.0, lo),
                t1 - t0 + 16.0,
                hi - lo,
                fill=False,
                ec=colour,
                lw=0.9,
                ls=(0, (3, 1.5)),
                zorder=5,
            )
        )
    if boxes:
        t0, _, _, _ = max(boxes, key=lambda b: (b[1] - b[0]) * (b[3] - b[2]))
        chip(ax, t0 - 8.0, chip_y, EVENT_NAMES[event], colour, ha="left", va="center")


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
        inside |= (t >= a) & (t <= b)
    return t[inside]


def track_bars(ax, track: lf.Track, colour: str, regimes=None) -> None:
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for side in ("left", "top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    if track.source is None:
        return
    for r in track.rows:
        span = [(r.t_start, r.t_end - r.t_start)]
        if r.category == ABSENT:
            ax.broken_barh(span, BAR, facecolor=ABSENT_GREY, lw=0)
            continue
        if regimes is not None:
            face = regimes.get(r.category, "#888888")
            hatch = "////" if r.category == 5 else None
        else:
            face = colour
            hatch = None
            if r.category == UNCERTAIN or r.crowd == 1:
                hatch = "//////"
            elif r.category == NOT_OBSERVABLE:
                hatch = "xxxx"
        ax.broken_barh(
            span,
            BAR,
            facecolor=face if hatch is None else "white",
            edgecolor=face,
            hatch=hatch,
            lw=0.4 if hatch else 0,
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
) -> dict:
    """The figure for `candidate` over `t0`-`t1` ms; returns what the record holds."""
    by_key = {t.spec.key: t for t in tracks}
    # the review stores: the n map, D-alpha, NBI
    _, drawn, stores = lf.signals(paths, candidate.shot, t0, t1)
    n_sig, da_sig, nbi_sig = drawn[0], drawn[1], drawn[2]

    low = Band(z, "zoom", 0.0, FOLD_KHZ, t0, t1)
    high = Band(z, "wide", FOLD_KHZ, TOP_KHZ + 1, t0, t1)
    spans_low = {
        e: present_spans(by_key[e]) for e in (mode_tags.NTM, mode_tags.SAWTOOTH)
    }
    spans_high = {mode_tags.AE: present_spans(by_key[mode_tags.AE])}
    blobs_low = low.blobs(spans_low)
    blobs_high = high.blobs(spans_high)

    # the n map, gated by the same filtered mask on the zoom pass
    gate = roster.Gate(
        low.lit_all,
        low.all_f,
        low.all_t,
        {"mask": "TokEye zoom pass, filtered", "threshold": mode_tags.PROB_THRESHOLD},
    )
    n_read, n_kept = roster.gated(n_sig.rows[0], gate) if n_sig.rows else (None, None)

    layout = {
        "h_raw": 0.2, "raw_hi": 1.0, "raw_lo": 0.8, "g1": 0.1, "da_raw": 0.42,
        "g2": 0.07, "nbi": 0.42, "h_proc": 0.56, "pr_hi": 1.0, "pr_lo": 0.8,
        "g3": 0.1, "da_pr": 0.55, "h_lab": 0.2,
    }  # fmt: skip
    names = [*layout, *[f"track{i}" for i in range(len(tracks))]]
    heights = [*layout.values(), *[0.16] * len(tracks)]
    with style():
        fig = Figure(figsize=(PAGE_IN, HEIGHT_IN))
        gs = fig.add_gridspec(
            len(heights),
            1,
            height_ratios=heights,
            hspace=0.0,
            left=0.115,
            right=0.915,
            top=0.985,
            bottom=0.085,
        )
        ax = {n: fig.add_subplot(gs[i]) for i, n in enumerate(names)}
        for n in ("h_raw", "g1", "g2", "h_proc", "g3", "h_lab"):
            ax[n].set_visible(False)
        track_axes = [ax[f"track{i}"] for i in range(len(tracks))]
        for n in ("raw_hi", "raw_lo", "da_raw", "nbi", "pr_hi", "pr_lo", "da_pr"):
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
            ax[lo].set_yticks([0, 20, 40])
            ax[hi].spines["bottom"].set_visible(False)
            ax[hi].tick_params(bottom=False)
            ax[hi].set_ylabel("kHz", labelpad=2)
            ax[lo].set_ylabel("kHz", labelpad=2)
        draw_raw(ax["raw_hi"], high)
        draw_raw(ax["raw_lo"], low)
        label_box = {"fc": "black", "alpha": 0.5, "ec": "none", "pad": 1.0}
        for name, text in (
            ("raw_hi", "magnetics spectrogram (Mirnov probe)"),
            ("raw_lo", "same, 0-55 kHz expanded"),
        ):
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
        draw_processed(ax["pr_hi"], high, blobs_high, fill=(mode_tags.AE,))
        draw_processed(ax["pr_lo"], low, blobs_low)
        wash = [
            Patch(fc=EVENT_COLOURS[e], lw=0, label=EVENT_NAMES[e])
            for e in (mode_tags.NTM, mode_tags.SAWTOOTH)
            if any(e in b.tags for b in blobs_low)
        ]
        for e, grow in ((mode_tags.SAWTOOTH, 3), (mode_tags.NTM, 1)):
            outline(ax["pr_lo"], low, low.tag_image(blobs_low, e),
                    EVENT_COLOURS[e], grow, lw=0.5)  # fmt: skip
        if wash:
            key = ax["pr_lo"].legend(
                handles=wash, loc="upper left", ncols=len(wash), frameon=True,
                facecolor="black", edgecolor="none", framealpha=0.6, fontsize=FONT,
                handlelength=1.0, columnspacing=1.0, borderaxespad=0.2,
                labelcolor="white", title="label over the mode", title_fontsize=FONT,
                alignment="left", labelspacing=0.2,
            )  # fmt: skip
            key.get_title().set_color("white")
        keys = []
        if n_read is not None:
            names = n_read.meta["modes"]["n"]
            keys = [f"n={names[i]}" for i in draw_n_view(ax["pr_lo"], n_read)]
        n_handles = n_key(n_read) if n_read is not None else []
        # the tags: the AE blobs tinted and boxed; a wash behind the others
        draw_boxes(
            ax["pr_hi"], merged_boxes(blobs_high, mode_tags.AE), mode_tags.AE,
            (FOLD_KHZ, TOP_KHZ), TOP_KHZ - 12,
        )  # fmt: skip

        # D-alpha: the ELM label's span and spikes, the confinement regimes
        elm_key = "edge_localized_mode"
        elm_spans = labelled_spans(by_key[elm_key])
        peaks = np.array([])
        da = ax["da_pr"]
        if da_sig.rows:
            trace(da, da_sig.rows[0])
            top = float(np.max(da_sig.rows[0].values[1]))
            da.set_ylim(0, top * 1.35)
            peaks = elm_peaks(da_sig.rows[0], elm_spans)
            if len(peaks):
                da.plot(peaks, np.full(len(peaks), top * 1.14), "v",
                        color=EVENT_COLOURS[elm_key], ms=2.2, mew=0)  # fmt: skip
            for a, b in elm_spans:
                da.add_patch(
                    Rectangle((a, 0), b - a, top * 1.25, fill=False,
                              ec=EVENT_COLOURS[elm_key], lw=0.9, ls=(0, (3, 1.5)))
                )  # fmt: skip
            if elm_spans:
                a, b = max(elm_spans, key=lambda s: s[1] - s[0])
                elm_chip = chip(da, min(b, t1) - 20, top * 1.3, "ELMs", "#cccccc",
                                ha="right", va="top")  # fmt: skip
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
        if elm_spans and regime_texts:
            clear_of(elm_chip, regime_texts)

        # ---- label tracks
        titles = {
            mode_tags.AE: "AE", mode_tags.NTM: "NTM", mode_tags.SAWTOOTH: "sawtooth",
            elm_key: "ELMs", "confinement": "confinement",
        }  # fmt: skip
        for a, track in zip(track_axes, tracks, strict=True):
            key = track.spec.key
            track_bars(a, track, EVENT_COLOURS.get(key, "#888888"),
                       REGIME_GREYS if key == "confinement" else None)  # fmt: skip
            a.set_ylabel(titles[key], rotation=0, ha="right", va="center", labelpad=3)
            tier = "" if track.source is None else TIER_NAMES[track.source.tier]
            a.text(1.008, 0.5, tier, transform=a.transAxes, fontsize=FONT,
                   va="center", ha="left", color="#444444")  # fmt: skip
        track_axes[-1].tick_params(labelbottom=True, bottom=True)
        track_axes[-1].spines["bottom"].set_visible(True)
        track_axes[-1].set_xlabel("time (ms)", labelpad=1)

        # group headings
        for name, lines in HEADINGS.items():
            pos = ax[name].get_position()
            for i, text in enumerate(reversed(lines)):
                bold = "bold" if text == lines[0] else "normal"
                y = pos.y0 + pos.height * 0.12 + i * 0.019
                fig.text(pos.x0, y, text, fontsize=FONT, fontweight=bold,
                         va="bottom", ha="left")  # fmt: skip

        if n_handles:
            pos = ax["h_proc"].get_position()
            fig.legend(
                [Patch(fc="none", lw=0), *n_handles],
                ["toroidal n, below 30 kHz:", *[h.get_label() for h in n_handles]],
                loc="lower right", ncols=len(n_handles) + 1, frameon=False,
                fontsize=FONT, handlelength=0.9, columnspacing=0.9,
                handletextpad=0.4, borderpad=0, borderaxespad=0,
                bbox_to_anchor=(0.915, pos.y0 + pos.height * 0.12 + 0.019 - 0.004),
            )  # fmt: skip
        colours = [
            EVENT_COLOURS[e] for e in (mode_tags.AE, mode_tags.NTM, mode_tags.SAWTOOTH)
        ]
        handles = [
            tuple(Patch(fc=c, lw=0) for c in (*colours, INK)),
            Patch(fc="white", ec=INK, hatch="//////", lw=0.4),
            Patch(fc=ABSENT_GREY, lw=0),
            Patch(fc="white", ec="#999999", lw=0.5),
        ]  # fmt: skip
        fig.legend(
            handles,
            ["present", "uncertain, or an ELMing period", "absent",
             "blank: never assessed"],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
            loc="lower center", ncols=4, frameon=False, fontsize=FONT,
            bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=3.2,
        )  # fmt: skip
        paths_out = save_figure(fig, stem, png_dpi)

    tags_count = {
        e: sum(e in b.tags for b in blobs_high + blobs_low)
        for e in (mode_tags.AE, mode_tags.NTM, mode_tags.SAWTOOTH)
    }
    return {
        "figure": [str(p) for p in paths_out],
        "blobs": {
            "wide_above_fold": len(blobs_high),
            "zoom_below_fold": len(blobs_low),
            "tagged": tags_count,
            "untagged": sum(not b.tags for b in blobs_high + blobs_low),
        },
        "elm_peaks_in_label": len(peaks),
        "n_map": None if n_read is None else {"kept_share": n_kept, "keys": keys},
        "regimes_shown": sorted(regime_names[c] for c in shown),
        "stores": {
            e: None if p is None else {"path": str(p), "sha256": sha256_of(p)}
            for e, p in stores.items()
        },
    }


def draw_n_view(ax, read) -> list[int]:
    """The gated n map over 0-30 kHz, in its palette, in place of the band; the
    indices of the n it keys (`n_seen`)."""
    modes = read.meta["modes"]
    colours = dict(zip(modes["n"], modes["colours"], strict=True))
    palette = np.array([to_rgb(c) for c in mode_palette(colours)])
    codes = np.minimum(read.values.astype(np.int64), len(palette) - 1)
    t0, t1, f0, f1 = read.extent
    keep = round((min(f1, N_VIEW_KHZ) - f0) / (f1 - f0) * codes.shape[0])
    lit = codes[:keep] >= len(modes["n"])  # level 0: no mode, left clear
    rgba = np.concatenate([palette[codes[:keep]], lit[..., None]], axis=-1)
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
    """The indices of the n a map lights in `roster.KEY_MIN_SHARE` of its lit
    pixels (the page's key rule)."""
    modes = read.meta["modes"]
    k = len(modes["n"])
    codes = read.values.astype(np.int64)
    lit = codes[codes >= k] % k
    share = np.bincount(lit, minlength=k) / max(lit.size, 1)
    return [int(i) for i in np.flatnonzero(share >= roster.KEY_MIN_SHARE)]


def n_key(read) -> list[Patch]:
    modes = read.meta["modes"]
    return [
        Patch(fc=modes["colours"][i], lw=0, label=f"n={modes['n'][i]}")
        for i in n_seen(read)
    ]


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
    return {
        "tier": track.source.tier,
        "what": track.source.what,
        "path": str(track.file),
        "sha256": sha256_of(track.file) if Path(track.file).is_file() else None,
        "rows": len(track.rows),
        "present_spans_ms": [[round(a, 1), round(b, 1)] for a, b in present],
    }


def main(argv=None) -> int:
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
    tracks, co2 = shot_tracks(paths, candidate)
    out.mkdir(parents=True, exist_ok=True)
    drawn = draw(paths, candidate, tracks, z, t0, t1, out / STEM, args.png_dpi)
    checkpoint = roster.tokeye_file(paths)
    record = {
        "shot": args.shot,
        "year": candidate.year,
        "window_ms": [t0, t1],
        "split": split,
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
            "pickup_row_share": mode_tags.PICKUP_SHARE,
            "min_size_px": mode_tags.MIN_SIZE,
            "hole_area_px": mode_tags.HOLE_AREA,
            "components": "skimage connectivity 2 (8-connected)",
            "tag_cover": mode_tags.COVER,
            "bands_khz": {
                k: [lo, None if math.isinf(hi) else hi]
                for k, (lo, hi) in mode_tags.BANDS.items()
            },
            "fold_khz": FOLD_KHZ,
            "elm_peaks": f"D-alpha less a {ELM_WINDOW}-sample running median, "
            f"over {ELM_MADS} MADs",
        },
        "tracks": {t.spec.key: track_record(t) for t in tracks},
        "ae_co2": None
        if co2 is None
        else {"path": str(co2), "sha256": sha256_of(co2) if co2.is_file() else None},
        "drawn": drawn,
        "git": git_sha(),
    }
    with atomic_path(out / f"{STEM}.json") as tmp:
        Path(tmp).write_text(json.dumps(record, indent=1) + "\n")
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(args.record) as tmp:
            Path(tmp).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(drawn["blobs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
