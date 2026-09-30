"""The paper's interpreter figure, `fig_interpreter`: one roster shot's suggestions.

The paper's build draws it (`paper.build`); this CLI draws a preview of it
into a scratch build:

    PYTHONPATH=src pixi run -e labelmaker python -m labeler.paper.roster \\
        --out DIR [--shot SHOT] [--version V] [--seg-version V] \\
        [--tables-version V]

The AE test shots carry none of the other events' groups, so no figure built
on them can show the other phenomena. This one draws a non-blind roster shot
of the frozen cohort with at least 2 s of corpus CO2 (`candidates`: 153 of its
450 non-blind shots):

- AE from the frame model (`AE_VERSION`) and SegNet's mask (`SEG_VERSION`),
  both run here over the shot's corpus CO2 rows (`data.raw_rows`), as the AE
  extension runs them; the picture is those rows pooled to SegNet's store level
  (`shots.PICTURE_LEVEL`), and the mask lies on its pixels. The mask is every
  pixel of the whole picture at or above SegNet's threshold, on every
  frequency row it has (`seg_band`, the picture's own band), not only its
  blob's AE band (the owner, 2026-09-29 21:05: "segmentation on on the full
  spectrogram instead of ae"), so its key is `shots.ROSTER_MASK_LABEL`,
  "segmentation"; the paper's own figures keep "segmentation: AE";
- NTM, H-mode, ELMing and sawteeth from their frame models' suggestion tables
  (`TABLES`, `TABLE_VERSION`, which is `frames.VERSION`, F1): a track says
  `NO_TABLE` where a table does not exist and `NOT_APPLIED` where the shot is
  not in it; the bars are `shots.state_bars`.

**The signals** (F10) lie between the CO2 spectrogram, on top, and the tracks,
at the bottom: the rows the frame models read (`frames.SPECS[m].roles`), from
the shot's review stores (`paths.spectrogram_file(spec.store_event, shot)`),
each matched by its role's title prefix (`frames.features.find_role`), one
panel each in `PANELS`' order. The NTM's Mirnov spectrogram (`MPI66M322D
power`) is drawn as the CO2 one is (`shots.show_image`); its n map (`toroidal
n, MPI66M probes`, `verify.mode_bytes`' codes) in the page's colours
(`verify.mode_palette`), a colour per n, with a small key of the n in view.
One D-alpha panel serves ELMs and H-mode: the ELM model's filterscope, the
channel its spans read (`D-alpha FS01` on shot 199563), labelled by its
title; neither `D-alpha PCPHD03` nor H-mode's pooled filterscopes row is
drawn. H-mode's NBI power; the sawteeth's four ECE Te rows in one panel, a
colour per row with a small key, and its SXR chords in their own. A trace is
drawn as each column's minimum-to-maximum band, as the store keeps it. Each row
is read over the figure's time range at a store level with a column for about
each pixel across the page (`PAGE_COLUMNS`; `rows.read_window`, which pools
with `rows.pool`), on the one time axis the panels share. A y-label is the
signal's name with its row's units. A store or row that is not there gives
a panel that says so (`shots.NO_DATA`, "no SXR data"); nothing is fetched.

**What it is not.** Everything on it is a suggestion: the title says so
(`TITLE`), every track's key is `shots.SUGGESTED`'s (the mask's is
`shots.ROSTER_MASK_LABEL`), no track is the owner's, and its record says
`"tier": "suggestions"`. It is never a reviewed label. It is the paper's own
interpreter figure: `paper.build` draws it into the paper's build as
`fig_interpreter` and records it in the manifest under `interpreter`. The CLI
is for previews: it writes `fig_interpreter` and `roster.json` into a scratch
build only, and refuses the paper's own `paper/` and anything inside it
(`ScratchOnly`) before it reads anything.

**The shot** is `PICK_RULE`'s (`pick`), or the one `--shot` names (`NAMED`),
which must be a candidate (`named`; else `NotACandidate`). Its record
(`record`: `roster.json` here, the build manifest's `interpreter`) holds it,
the rule, the number of candidates, each model, table and store it read, by
path and sha256 (a table or store that does not exist as null), the band the
mask was drawn over (`seg_band_khz`, null without SegNet), and the corpus
file by path alone: it is GBs, read in slices, never hashed. Every other file
is read once, through a `snapshot.Snapshot`, so each sha256 is of the bytes
drawn.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import frames
from ..ae import seg as ae_seg
from ..ae import xpower
from ..ae.seg import train as seg_train
from ..ae.seg.poi import ae_pixels
from ..ae.xpower.data import raw_rows, targets, window_frames
from ..ae.xpower.evaluate import chosen_model
from ..ae.xpower.extend import MIN_CO2_S, frame_states
from ..ae.xpower.train import probabilities
from ..config import Paths, atomic_path, git_sha
from ..events import suggestions
from ..events.catalog.cohort import read_cohort
from ..events.catalog.states import PRESENT
from ..events.review import labels
from ..events.review import rows as store_rows
from ..events.review.labels import Label
from ..events.review.rows import Grid, pool
from ..events.spans import cohort_path
from ..events.verify import corpus_signal, mode_palette
from ..frames.features import find_role
from ..scoring.frames import FRAME_MS
from . import (
    AE,
    FONT_PT,
    LOGIN_THREADS,
    ORDER,
    PAGE_IN,
    paper_dir,
    save,
    shots,
    style,
    title,
)
from .shots import (
    MARGIN_MS,
    NO_DATA,
    NOT_APPLIED,
    PICTURE_LEVEL,
    ROSTER_MASK_LABEL,
    _legend,
    _mask,
    _spectrogram,
    show_image,
    state_bars,
    text_track,
)
from .snapshot import Snapshot

#: The other paper phenomena, in `ORDER`'s order, and each one's frame model
#: (a `labeler.frames.SPECS` key), whose suggestion table the track reads.
TABLES = {
    "neoclassical_tearing_mode": "ntm_frames",
    "high_confinement_mode": "hmode_frames",
    "edge_localized_mode": "elm_frames",
    "sawtooth_oscillation": "sawtooth_frames",
}
TABLE_VERSION = frames.VERSION
#: The AE frame model: ae_xpower v2, the paper's (its best scorer, the build's
#: `--version v2`). The build draws the figure with its own `--version`.
AE_VERSION = "v2"
#: SegNet v3, trained over 0-250 kHz, for the figure's full-band mask (the
#: owner, 2026-09-29 21:05): v1 and v2, trained on the AE band, give nothing
#: usable below it (on shot 199563 v1 fills 0-42 kHz over the whole shot and v2
#: misses the 25 kHz tearing mode), and v3 draws the MHD lines too, as a
#: segmentation of the full spectrogram, not of AE, should. The build's
#: `--interpreter-seg-version` defaults to it.
SEG_VERSION = "v3"
STEM = "fig_interpreter"
MANIFEST = "roster.json"
PICK_RULE = (
    "the non-blind roster shot with at least 2 s of corpus CO2 on which the most "
    "of the four frame models suggest a present span, then the most present time "
    "over the four, then the lower shot number"
)
NAMED = "named by --shot"
NO_TABLE = "no table"
TITLE = "shot {shot} ({year}): the models' suggestions, not reviewed"
TIER = "suggestions"
#: About a column per pixel across the page, at `paper.save`'s 300 dpi.
PAGE_COLUMNS = round(PAGE_IN * 300)
TRACE_LW = 0.3
#: The n map's key names an n with at least this share of its lit pixels: the
#: map's scattered noise pixels light every n, and a key of all of them
#: overflows the panel (shot 199563 lit ten).
KEY_MIN_SHARE = 0.05
#: A row per colour: the ECE groups', the SXR chords'.
ROW_COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")


@dataclass(frozen=True)
class SignalPanel:
    """A signal panel (F10): the rows of frame model `method`'s roles named
    `role`, drawn as `kind` ("image", "modes" or "trace"). Its y-label is
    `name` with the row's units when `units` (`name` None: the row's title
    before its first comma, as "D-alpha FS01"); `title` names it in
    `NO_DATA`'s "no <title> data"."""

    title: str
    method: str
    role: str
    kind: str
    name: str | None
    units: bool = True
    height: float = 0.9

    @property
    def roles(self) -> tuple:
        return tuple(r for r in frames.SPECS[self.method].roles if r.name == self.role)

    @property
    def event(self) -> str:
        return frames.SPECS[self.method].store_event


PANELS = (
    SignalPanel("MPI66M322D power", "ntm_frames", "power", "image", "Mirnov", height=2),
    SignalPanel("toroidal n", "ntm_frames", "modes", "modes", "n", height=1.6),
    SignalPanel("D-alpha FS", "elm_frames", "dalpha", "trace", None, units=False),
    SignalPanel("NBI power", "hmode_frames", "nbi", "trace", "NBI"),
    SignalPanel("ECE Te", "sawtooth_frames", "ece", "trace", "ECE Te", height=1.2),
    SignalPanel("SXR", "sawtooth_frames", "sxr", "trace", "SXR"),
)


@dataclass(frozen=True)
class Read:
    """One store row over `t0`-`t1` ms: its description (`rows.meta`'s) and
    values, an image's `(n_y, k)` bytes or a trace's `(2, C, k)` minima and
    maxima."""

    meta: dict
    values: np.ndarray
    t0: float
    t1: float

    @property
    def centres(self) -> np.ndarray:
        k = self.values.shape[-1]
        return self.t0 + (np.arange(k) + 0.5) * (self.t1 - self.t0) / k

    @property
    def extent(self) -> tuple[float, float, float, float]:
        n_y, y0, dy = self.meta["n_y"], self.meta["y0"], self.meta["dy"]
        return (self.t0, self.t1, y0 - dy / 2, y0 + (n_y - 0.5) * dy)


@dataclass(frozen=True)
class Signal:
    """A drawn signal panel: its y-label and rows, or the `text` it says."""

    panel: SignalPanel
    label: str
    rows: tuple[Read, ...] = ()
    text: str | None = None


class ScratchOnly(ValueError):
    """The CLI's preview goes into a scratch build, never the paper's own."""


class NotACandidate(ValueError):
    """A named shot that is not a roster candidate."""


@dataclass(frozen=True)
class Candidate:
    shot: int
    year: int
    window: tuple[int, int]  # the cohort's window_start_ms, window_end_ms


@dataclass(frozen=True)
class RosterShot:
    """The drawn shot. `prob` is the frame model's P(AE) per 10 ms frame from
    `first`; the picture (`grid`, `image`, `y0`, `dy`) and SegNet's `mask` are
    the corpus rows pooled to `PICTURE_LEVEL`; `tracks` holds, per `ORDER`
    category, the suggested state per frame, or the text `NO_TABLE` or
    `NOT_APPLIED`; `signals` the signal panels (F10), `stores` each store
    event's file, None where there is none, and `seg_band` the band, kHz, the
    mask is over: the picture's own, every row (None: SegNet not run)."""

    shot: int
    year: int
    window: tuple[int, int]
    grid: Grid
    image: np.ndarray  # (n_y, grid.n) uint8: the first cross-power row, pooled
    y0: float
    dy: float
    first: int
    prob: np.ndarray
    threshold: float
    mask: np.ndarray | None  # (n_y, grid.n) bool: SegNet, every row; None: not run
    tracks: dict
    signals: tuple[Signal, ...] = ()
    stores: dict = field(default_factory=dict)
    seg_band: tuple[float, float] | None = None

    @property
    def edges(self) -> np.ndarray:
        return (self.first + np.arange(len(self.prob) + 1)) * FRAME_MS


def candidates(paths: Paths, snap: Snapshot | None = None) -> list[Candidate]:
    """The frozen cohort's non-blind shots with at least `MIN_CO2_S` of corpus
    CO2, by shot; the cohort read through `snap` (pinned as "cohort") if given."""
    path = cohort_path(paths)
    cohort = read_cohort(path if snap is None else snap.copy("cohort", path))
    keep = cohort[~cohort["blind"] & (cohort["span_co2_s"] >= MIN_CO2_S)]
    columns = ("shot", "year", "window_start_ms", "window_end_ms")
    found = [
        Candidate(int(shot), int(year), (int(lo), int(hi)))
        for shot, year, lo, hi in zip(*(keep[c] for c in columns), strict=True)
    ]
    return sorted(found, key=lambda c: c.shot)


def table_file(paths: Paths, category: str, version: str = TABLE_VERSION) -> Path:
    """The suggestion table `category`'s frame model wrote."""
    return suggestions.table_path(paths, category, TABLES[category], version)


def table_key(method: str) -> str:
    """Frame model `method`'s suggestion table's key in a snapshot."""
    return f"frames_table_{method}"


def store_key(event: str, shot: int) -> str:
    """`shot`'s `event` review store's key in a snapshot."""
    return f"store_{event}_{shot}"


def read_tables(
    paths: Paths, version: str = TABLE_VERSION, snap: Snapshot | None = None
) -> dict[str, dict[int, Label] | None]:
    """Each frame model's table as labels by shot (`labels.read_labels`: its
    window from the first to the last row, the absent rows dropped), in
    `TABLES`' order; None where the table does not exist. Read through `snap`
    (`table_key`) if given."""
    out = {}
    for category, method in TABLES.items():
        file = table_file(paths, category, version)
        if not file.is_file():
            out[category] = None
        elif snap is None:
            out[category] = labels.read_labels(file)
        else:
            out[category] = snap.labels(table_key(method), file)
    return out


def present_ms(label: Label | None) -> int:
    """The present spans' total time, ms; 0 for no label."""
    if label is None:
        return 0
    return int(sum(b - a for a, b, c in label.intervals if c == PRESENT))


def pick(
    found: list[Candidate], tables: Mapping[str, Mapping[int, Label] | None]
) -> Candidate:
    """`PICK_RULE`'s shot among `found`: the most of the four tables with some
    present time on it, then the most present time over the four, then the
    lower shot number. A missing table (None) counts 0."""
    if not found:
        raise ValueError("no candidate shot to pick from")

    def key(c: Candidate) -> tuple[int, int, int]:
        times = [present_ms((tables.get(t) or {}).get(c.shot)) for t in TABLES]
        return (-sum(t > 0 for t in times), -sum(times), c.shot)

    return min(found, key=key)


def named(found: list[Candidate], shot: int | None) -> Candidate | None:
    """The candidate `shot` names; None for no shot, and `NotACandidate` for a
    shot that is not one."""
    if shot is None:
        return None
    for candidate in found:
        if candidate.shot == shot:
            return candidate
    raise NotACandidate(f"{shot}: not a non-blind roster shot with corpus CO2")


def choose(
    found: list[Candidate],
    tables: Mapping[str, Mapping[int, Label] | None],
    chosen: Candidate | None,
) -> tuple[Candidate, str]:
    """The shot to draw and its rule: `chosen` (`NAMED`), else `pick`'s
    (`PICK_RULE`)."""
    if chosen is not None:
        return chosen, NAMED
    return pick(found, tables), PICK_RULE


def _source(store: Path | bytes):
    """A store for h5py: its path, or a file over the bytes a snapshot read."""
    return io.BytesIO(store) if isinstance(store, bytes) else store


def read_row(store: Path | bytes, row: dict, t0: float, t1: float) -> Read | None:
    """The store row `row` (a `rows.meta` description) over `t0`-`t1` ms, at
    about `PAGE_COLUMNS` columns (`rows.read_window`); None where the store's
    record does not reach that range. `store` is a path or the file's bytes."""
    described = store_rows.meta(_source(store))
    start, end = described["t_range"]
    t0, t1 = max(t0, start), min(t1, end)
    if t1 <= t0:
        return None
    hide = {r["name"] for r in described["rows"]} - {row["name"]}
    data, grid = store_rows.read_window(_source(store), t0, t1, PAGE_COLUMNS, hide=hide)
    if row["kind"] == "image":
        values = np.frombuffer(data, np.uint8).reshape(row["n_y"], grid["n"])
    else:
        shape = (2, row["n_channels"], grid["n"])
        values = np.frombuffer(data, "<f4").reshape(shape)
    return Read(row, values, grid["t0"], grid["t1"])


def _label(panel: SignalPanel, found: list[Read]) -> str:
    if panel.name is None:
        head = found[0].meta["title"].split(",")[0] if found else panel.title
        return head.replace(" ", "\n", 1)
    units = found[0].meta.get("y_units", "") if found and panel.units else ""
    return f"{panel.name}\n({units})" if units else panel.name


def signals(
    paths: Paths, shot: int, t0: float, t1: float, snap: Snapshot | None = None
) -> tuple[tuple[Signal, ...], dict]:
    """`PANELS`' signals of `shot` over `t0`-`t1` ms from its review stores,
    and each store event's file (None where there is none). Each store is read
    once through `snap` (`store_key`) if given. Never fetches."""
    stores = {}
    for panel in PANELS:
        path = paths.spectrogram_file(panel.event, shot)
        stores.setdefault(panel.event, path if path.is_file() else None)
    read_from = {
        event: path if snap is None else snap.read(store_key(event, shot), path)
        for event, path in stores.items()
        if path is not None
    }
    out = []
    for panel in PANELS:
        store = read_from.get(panel.event)
        described = [] if store is None else store_rows.meta(_source(store))["rows"]
        found = []
        for role in panel.roles:
            row = find_role(described, role)
            read = None if row is None else read_row(store, row, t0, t1)
            if read is not None:
                found.append(read)
        text = None if found else NO_DATA.format(panel.title)
        out.append(Signal(panel, _label(panel, found), tuple(found), text))
    return tuple(out), stores


def whole_band(y0: float, dy: float, n_y: int) -> tuple[float, float]:
    """The picture's own band, kHz: from its lowest row's lower edge to its
    highest row's upper edge, so every row's centre lies in it."""
    return (float(y0 - dy / 2), float(y0 + (n_y - 0.5) * dy))


def roster_shot(
    paths: Paths,
    candidate: Candidate,
    *,
    model: shots.Model,
    segmentation: shots.Segmentation | None = None,
    tables: Mapping[str, Mapping[int, Label] | None],
    snap: Snapshot | None = None,
) -> RosterShot:
    """`candidate` with the frame model `model`, and SegNet `segmentation` if
    given, run over its corpus CO2 rows (read in place: `NoDataError` without
    them), its tracks from `tables` (`read_tables`) and its signal panels from
    its review stores, read through `snap` if given (`signals`)."""
    co2 = corpus_signal(candidate.shot, "co2", corpus=paths.corpus)
    rows = raw_rows(co2.x, co2.y)
    first, n = window_frames(candidate.window)
    prob, observed = probabilities(
        model.net, rows, first, n, band=model.blob["band_khz"]
    )
    threshold = model.threshold
    grid, values, y0, dy = rows
    pooled = pool(values, PICTURE_LEVEL, "image")
    picture = Grid(grid.t0_ms, grid.dt_ms * PICTURE_LEVEL, -(-grid.n // PICTURE_LEVEL))
    mask = seg_band = None
    if segmentation is not None:
        seg_band = whole_band(y0, dy, pooled.shape[1])
        mask = ae_pixels(
            seg_train.predict(segmentation.net, pooled),
            segmentation.threshold,
            y0,
            dy,
            band=seg_band,
        )
    edges = (first + np.array([0, n])) * FRAME_MS
    drawn, stores = signals(
        paths, candidate.shot, edges[0] - MARGIN_MS, edges[1] + MARGIN_MS, snap
    )
    tracks: dict = {AE: frame_states(prob, observed, threshold)}
    for category in TABLES:
        table = tables.get(category)
        if table is None:
            tracks[category] = NO_TABLE
        elif candidate.shot not in table:
            tracks[category] = NOT_APPLIED
        else:
            tracks[category] = targets(table[candidate.shot], first, n)
    return RosterShot(
        shot=candidate.shot,
        year=candidate.year,
        window=candidate.window,
        grid=picture,
        image=pooled[0],
        y0=y0,
        dy=dy,
        first=first,
        prob=prob,
        threshold=threshold,
        mask=mask,
        tracks=tracks,
        signals=drawn,
        stores=stores,
        seg_band=seg_band,
    )


def _key(ax, handles, labelcolor=None) -> None:
    """A small key inside a signal panel, kept out of the figure's legend."""
    if handles:
        ax.legend(
            handles=handles,
            loc="upper right",
            ncols=len(handles),
            fontsize=FONT_PT - 2,
            frameon=False,
            handlelength=0.8,
            columnspacing=0.8,
            borderaxespad=0.1,
            labelcolor=labelcolor,
        )


def _modes(ax, read: Read) -> None:
    """The n map in the page's colours, a colour per n, keyed by the n in view
    (`KEY_MIN_SHARE` of the lit pixels), in white on the map's black."""
    modes = read.meta["modes"]
    colours = dict(zip(modes["n"], modes["colours"], strict=True))
    palette = np.array([to_rgb(c) for c in mode_palette(colours)])
    codes = np.minimum(read.values.astype(np.int64), len(palette) - 1)
    ax.imshow(
        palette[codes],
        origin="lower",
        aspect="auto",
        extent=read.extent,
        interpolation="nearest",
    )
    k = len(modes["n"])
    lit = codes[codes >= k] % k  # level 0 is black, no mode
    share = np.bincount(lit, minlength=k) / max(lit.size, 1)
    seen = np.flatnonzero(share >= KEY_MIN_SHARE)
    handles = [
        Patch(color=modes["colours"][i], label=f"n={modes['n'][i]}") for i in seen
    ]
    _key(ax, handles, labelcolor="white")


def _traces(ax, sig: Signal) -> None:
    """Each channel's minimum-to-maximum band: a colour per row when there are
    several (the ECE groups, keyed), else a colour per channel."""
    handles = []
    several = len(sig.rows) > 1
    for i, read in enumerate(sig.rows):
        low, high = read.values
        for c in range(len(low)):
            colour = ROW_COLOURS[(i if several else c) % len(ROW_COLOURS)]
            ax.fill_between(
                read.centres,
                low[c],
                high[c],
                color=colour,
                lw=TRACE_LW,
                label="_signal",
            )
        if several:
            name = read.meta["title"].split(",", 1)[-1].split("(")[0].strip()
            colour = ROW_COLOURS[i % len(ROW_COLOURS)]
            handles.append(Line2D([], [], color=colour, lw=1.0, label=name))
    _key(ax, handles)


def _signal(ax, sig: Signal) -> None:
    ax.set_ylabel(sig.label)
    if sig.text is not None:
        ax.set_yticks([])
        text_track(ax, sig.text)
        return
    kind = sig.panel.kind
    if kind == "trace":
        _traces(ax, sig)
        return
    (read,) = sig.rows
    top = read.extent[3]
    if kind == "image":
        show_image(ax, read.values, read.extent, top, sig.label)
    else:
        _modes(ax, read)
        ax.set_ylim(0, top)


def draw(s: RosterShot, stem: Path) -> Figure:
    """The spectrogram with SegNet's mask, the signal panels (F10), then one
    track per phenomenon, each the models' suggestions."""
    t0, t1 = s.edges[0] - MARGIN_MS, s.edges[-1] + MARGIN_MS
    heights = [4, *(g.panel.height for g in s.signals), *[0.45] * len(ORDER)]
    with style():
        fig = Figure(
            figsize=(PAGE_IN, 3.4 + 0.45 * math.fsum(heights[1 : -len(ORDER)])),
            layout="constrained",
        )
        spec, *axes = fig.subplots(
            len(heights), 1, sharex=True, gridspec_kw={"height_ratios": heights}
        )
        panels, tracks = axes[: len(s.signals)], axes[len(s.signals) :]
        _spectrogram(spec, s)
        spec.set_xlim(t0, t1)
        _mask(spec, s, ROSTER_MASK_LABEL)
        spec.set_title(TITLE.format(shot=s.shot, year=s.year))
        for ax, sig in zip(panels, s.signals, strict=True):
            _signal(ax, sig)
        for ax, category in zip(tracks, ORDER, strict=True):
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_ylabel(title(category), rotation=0, ha="right", va="center")
            track = s.tracks[category]
            if isinstance(track, str):
                text_track(ax, track)
            else:
                state_bars(ax, s, track)
        for ax in (spec, *panels, *tracks[:-1]):
            ax.tick_params(labelbottom=False)
        for ax in tracks[:-1]:
            ax.tick_params(bottom=False)
        tracks[-1].set_xlabel("time (ms)")
        _legend(fig)
        save(fig, stem)
    return fig


def pinned(snap: Snapshot, key: str) -> dict | None:
    """`key`'s pin in `snap`, its path and sha256; None where it was not read."""
    if key not in snap.pinned:
        return None
    path, sha = snap.pinned[key]
    return {"path": str(path), "sha256": sha}


def record(
    s: RosterShot,
    snap: Snapshot,
    paths: Paths,
    *,
    rule: str,
    candidates: int,
    ae_version: str,
    seg_version: str,
    tables_version: str = TABLE_VERSION,
    ae_key: str = "ae_model",
    seg_key: str = "seg_model",
) -> dict:
    """What the figure of `s` was drawn from: its shot and the `rule` that
    picked it among `candidates`, each model, table and store `snap` read for
    it, by path and sha256 (null where one was not there), the band its mask
    is over, and the corpus file by path alone (never hashed: GBs, read in
    slices); `roster.json`'s, less its commit, and the build manifest's
    `interpreter`."""
    return {
        "shot": s.shot,
        "year": s.year,
        "window": list(s.window),
        "pick_rule": rule,
        "candidates": candidates,
        "ae_version": ae_version,
        "ae_model": pinned(snap, ae_key),
        "seg_version": seg_version,
        "seg_model": pinned(snap, seg_key),
        "seg_band_khz": None if s.seg_band is None else list(s.seg_band),
        "tables_version": tables_version,
        "tables": {
            method: pinned(snap, table_key(method)) for method in TABLES.values()
        },
        "stores": {event: pinned(snap, store_key(event, s.shot)) for event in s.stores},
        "corpus": str(paths.corpus_file(s.shot)),
        "tier": TIER,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out", type=Path, required=True, help="a scratch build, never paper/"
    )
    parser.add_argument(
        "--shot", type=int, help=f"a roster candidate (default: {PICK_RULE})"
    )
    parser.add_argument(
        "--version",
        default=AE_VERSION,
        help=f"the AE frame model's version (default {AE_VERSION})",
    )
    parser.add_argument(
        "--seg-version",
        default=SEG_VERSION,
        help=f"the segmentation's version (default {SEG_VERSION})",
    )
    parser.add_argument(
        "--tables-version",
        default=TABLE_VERSION,
        help=f"the frame models' tables' version (default {TABLE_VERSION})",
    )
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    paper = paper_dir(paths).resolve()
    if args.out.resolve().is_relative_to(paper):
        raise ScratchOnly(
            f"{args.out}: the roster CLI draws previews into a scratch build, never "
            f"{paper} or inside it; the paper's fig_interpreter is paper.build's"
        )
    torch.set_num_threads(LOGIN_THREADS)  # one shot, on the login node
    with tempfile.TemporaryDirectory(prefix="roster-inputs-") as scratch:
        snap = Snapshot(Path(scratch))
        found = candidates(paths, snap)
        chosen = named(found, args.shot)
        tables = read_tables(paths, args.tables_version, snap)
        candidate, rule = choose(found, tables, chosen)
        model = snap.model(
            "ae_model", chosen_model(xpower.model_dir(paths, args.version)), {}
        )
        seg_file = ae_seg.model_dir(paths, args.seg_version) / "model.pt"
        segmentation = (
            snap.segmentation("seg_model", seg_file) if seg_file.is_file() else None
        )
        s = roster_shot(
            paths,
            candidate,
            model=model,
            segmentation=segmentation,
            tables=tables,
            snap=snap,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        draw(s, args.out / STEM)
        drawn_from = record(
            s,
            snap,
            paths,
            rule=rule,
            candidates=len(found),
            ae_version=args.version,
            seg_version=args.seg_version,
            tables_version=args.tables_version,
        )
    manifest = args.out / MANIFEST
    with atomic_path(manifest) as tmp:
        tmp.write_text(json.dumps(drawn_from | {"git": git_sha()}, indent=1) + "\n")
    print(json.dumps({"out": str(args.out), "shot": s.shot, "manifest": str(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
