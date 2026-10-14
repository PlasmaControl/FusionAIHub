r"""fig_interpreter (the teaser): raw signals -> TokEye-processed -> labelled.

    PYTHONPATH=src pixi run --frozen -e labeler \
        python scripts/labeler/paper/fig_interpreter_tokeye.py \
        [--shot 199597] [--tmin MS --tmax MS] [--out DIR] [--rows EVENT=CSV ...]

One non-blind cohort shot over a few seconds, in three groups on one time axis:

- **raw**: the CO2 interferometer's R0 x V1 cross-power spectrogram (`SOURCE_ROW`,
  `PARTNER_ROW`, the wide pass) on one
  linear 0-150 kHz axis, one colour scale, D-alpha and NBI power;
- **processed**: the same spectrogram through TokEye (the network's mode mask,
  `labeler.paper.mode_tags`: transient burst removed, pickup removed,
  `skimage.morphology.remove_small_objects` and small-hole filling, connected
  components) on the same linear axis (the zoom pass below `ZOOM_TOP_KHZ`, the
  wide pass above), the toroidal-n view (the review page's n map, gated by the
  same mask) in the mask below 30 kHz. Highlights intersect PRESENT label
  times with AE >=60 kHz or NTM <=30 kHz; NTM requires dominant and pixel n=1 or 2.
  Panel (b) has no key beside it: a chip in each event's colour names AE, TM,
  transient (when drawn) and each n colour in place (`label_events`), read from the
  panel's own pixels, and the panels run to the right margin (`RIGHT`).
  Optional ECE-supported, ELM-vetoed crash candidates appear on a thin strip. D-alpha
  carries the ELM label's span and the D-alpha peaks in
  it, and the confinement classes shade it;
- **labels**: one track per event on the shot's time axis: present, absent or
  blank (unassessed / unobservable), adjacent rows of one state drawn as one run,
  the confinement runs named in place (L, H, QH, WPQH). There is no key below the
  panels. Each event's rows are `--rows EVENT=CSV` where given, else its review
  page's source table (`data/events/<event>/review/source.json`) where that holds
  the shot, else the tiers below (`labeler.paper.label_figure.TRACKS`). A table
  whose metadata names a model (threshold, evaluation, checkpoint) is that
  detector's output and keeps its threshold, held-out score and training check; a
  curated or imported table is the legacy tier; any other table holds labels. A
  gold evaluation shot never draws an expert review: its tiers skip that one.

Panel (a) is drawn in `RAW_CMAP`. Panel (b) draws no transient overlay: the owner
turned it off on 2026-10-14 as too distracting (`SHOW_TRANSIENT`, False; the red
lines over the spectrogram), so the default figure, its record, caption draft and
appendix say nothing of red transient lines. The transient channel still decides
the mask (the coherent mask excludes it) and `--transient` draws the overlay again:
the wide pass's transient channel over the whole 0-150 kHz band, at its own display
cut (`TRANSIENT_THRESHOLD`, 0.35; `--transient-cut`, 0 < cut <= 1), max-pooled to
the PNG's print pixels. The coherent mask and its tags stay at `TOKEYE_THRESHOLD`.
Uncertain time is hatched with explicit line segments (`hatch_lines`), which every
PDF viewer draws alike.

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
`--rows` and the source tables come before all of these.
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import math
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import rgb_to_hsv, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path as PlotPath
from scipy import ndimage, signal

from labeler.config import Paths, atomic_path, git_sha, sha256_of
from labeler.events import equilibrium, masks, unet
from labeler.events.catalog.cohort import read_cohort
from labeler.events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT, UNCERTAIN
from labeler.events.review import labels
from labeler.events.review.gold import is_gold
from labeler.events.spans import cohort_path
from labeler.events.verify import (
    NoDataError,
    corpus_path,
    corpus_signal,
)
from labeler.paper import PAGE_IN, figure_sources, mode_tags, roster, style, theme
from labeler.paper import label_figure as lf

STEM = "fig_interpreter"
HEIGHT_IN = 5.43
#: The page's margins above the panels and below them (the time axis and its label;
#: there is no key below), inches.
TOP_IN = 0.086
BOTTOM_IN = 0.36
DPI_PNG = 150
DPI_PDF = 300
#: The window's margin the TokEye cache keeps beyond the figure's, ms.
CACHE_MARGIN_MS = 100.0
#: The `split` a shot in no cohort split carries, and its default window, ms.
OUTSIDE_COHORT = "outside_cohort"
OUTSIDE_WINDOW = (8, 7000)

#: Each shot's window, ms, where `--tmin`/`--tmax` are not given (the cohort
#: window otherwise): chosen on the shot's own labels, see the report.
PRESETS = {
    199597: (100.0, 6200.0),
    186636: (1300.0, 3900.0),
    201973: (1600.0, 3350.0),
    199563: (300.0, 5800.0),
    201978: (1500.0, 3300.0),
    203187: (1700.0, 3150.0),
}
#: The AE/NTM split (mode_tags.SPLIT_KHZ): AE is tinted at and above it, NTM and
#: sawtooth below it. The frequency axis itself is linear and unbroken.
SPLIT_KHZ = mode_tags.SPLIT_KHZ
#: The tag analysis reads the zoom pass (0.12 kHz/bin; its decimation filter rolls
#: off from about 50 kHz, Nyquist 62.5 kHz) below `SPLIT_KHZ`, the wide pass
#: (0.49 kHz/bin, drawn to `TOP_KHZ`) above it.
#: The mask is drawn from the zoom pass up to here; its top few kHz are the
#: decimation filter's roll-off (a dark strip), so the wide pass fills the rest.
ZOOM_TOP_KHZ = 50.0
#: Both spectrograms: one linear axis, these ticks.
FREQ_TICKS_KHZ = [0, 50, 100, 150]
TOP_KHZ = 150.0
N_VIEW_KHZ = 30.0  # the n map's band
#: The columns an image is pooled to for the page (a 3 s wide pass has 12,000).
IMAGE_COLUMNS = 3000
#: The ELM peaks: D-alpha less its running median, over this many MADs.
ELM_WINDOW = 25
ELM_MADS = 4.0
#: The paper theme (`labeler.paper.theme`): n = 1 sea green, n = 2 sky (coral is the
#: TM outline's), n = 3 saffron; other n is purple, at full colour.
N_COLOURS = {1: theme.SEA_GREEN, 2: theme.SKY, 3: theme.SAFFRON}
N_OTHER = "#8C6BB1"
EVENT_COLOURS = {
    mode_tags.AE: theme.PLUM,
    mode_tags.NTM: theme.CORAL,
    mode_tags.SAWTOOTH: theme.SLATE,
    "edge_localized_mode": "#333333",
}
EVENT_NAMES = {
    mode_tags.AE: "AE",
    mode_tags.NTM: "TM",
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
#: The classes' short names, written on their runs in (f) and their shading in (c).
CLASS_LEGEND = {2: "L", 1: "H", 3: "QH", 4: "WPQH"}
ABSENT_GREY = "#e4e4e4"
#: Uncertain time: white, hatched in the event's colour with thin lines (an
#: uncertain confinement class in `UNCERTAIN_CLASS_COLOUR`). The hatch is drawn as
#: explicit line segments (`hatch_lines`), not a PDF hatch pattern, which poppler
#: renders faint and grey: "/" lines `UNCERTAIN_HATCH_STEP_PT` apart along the row
#: (the spacing matplotlib's "////" has), `UNCERTAIN_HATCH_LW` pt wide.
UNCERTAIN_HATCH_STEP_PT = 6.0
UNCERTAIN_HATCH_LW = 0.5
#: A clipped hatch line shorter than this, pt, is dropped: a box corner's nub
#: would print as a stub. A run too narrow to keep any line (under about 1.4 pt)
#: gets one vertical line instead, so it never reads as blank (unassessed).
UNCERTAIN_HATCH_MIN_PT = 2.0
UNCERTAIN_CLASS_COLOUR = "#888888"
#: A class name is written on its run only where it fits with this much room, pt.
RUN_TEXT_PAD_PT = 2.0
NTM_CONTOUR_COLOUR = EVENT_COLOURS[mode_tags.NTM]
NTM_CONTOUR_LW = 0.9
#: The thin dashed outline of the rest of a tagged component the detector clip cuts.
NTM_DASHED_LW = 0.7
NTM_DASHED_STYLE = (0, (3, 1.6))
#: Outlines enclosing fewer print pixels (150 dpi) than this are not drawn.
NTM_OUTLINE_MIN_PX = 100
#: The ELM interval boxes, in coral so they read against the D-alpha trace.
ELM_BOX_COLOUR = theme.CORAL
#: The boxes' name, written once over the first box.
ELM_BOX_TEXT = "ELMs"
#: The probability TokEye's coherent channel is cut at in this figure, for the mask,
#: its tags and the transients it leaves out (the network's operating point is
#: `mode_tags.PROB_THRESHOLD`, 0.2; the figure raises it to show less
#: low-confidence noise).
TOKEYE_THRESHOLD = 0.45
#: Whether panel (b) draws the transient overlay at all. OFF: the owner turned it off
#: on 2026-10-14 ("the red lines are actually too distracting ... just remove the
#: transient portion from spectrogram"). The code path stays behind this constant and
#: `--transient`, with `TRANSIENT_THRESHOLD` as its cut, so it can come back; the
#: default render has no red lines, no "transient" chip and no sentence about them.
SHOW_TRANSIENT = False
#: The transient overlay's own display cut (used only where the overlay is drawn),
#: between the network's operating point (0.2, where the bursts crowd the panel) and
#: the mask's 0.45 (where few show); the owner chose 0.35 on 2026-10-14, the day he
#: turned the overlay off (`SHOW_TRANSIENT`) as too distracting. Display only,
#: nothing is tagged from it. Its colour. The overlay is the wide pass's transient
#: channel over the whole band (`transient_overlay`): the zoom pass (1 ms columns)
#: showed almost no transient below `ZOOM_TOP_KHZ`, so the red lines stopped at the
#: pass seam.
TRANSIENT_THRESHOLD = 0.35
TRANSIENT_COLOUR = "#E8262B"
TRANSIENT_PASS = "wide"
#: Panel (b)'s layers, bottom to top: the black face of the axes, the red transient
#: overlay (`TRANSIENT_Z`; only with `SHOW_TRANSIENT`), the white mask (`MASK_Z`, transparent where unlit), the
#: n view (2), the AE tint (4), the TM outlines (4.5, 5), leaders (7) and chips (8).
#: The mask lies over the red, so at any resolution no lit mask pixel turns red,
#: however the red is pooled to print pixels.
PANEL_FACE = "black"
TRANSIENT_Z = 0.1
MASK_Z = 0.2
#: Panel (b)'s named classes are drawn by artists whose gid is this prefix and the
#: class ("AE", "TM", "transient"; "n" for the n view, one image of every n colour):
#: `drawn_classes` renders each alone, so a chip names a class only where that
#: class's own artists draw it, never on a blend of two colours (`label_events`).
DRAWN_GID = "panel-b:"
#: A pixel is a class's where its artists alone cover at least this share of it,
#: and (for the n view) its colour is within `DRAWN_RGB_TOL` of the class's.
DRAWN_MIN_ALPHA = 0.5
DRAWN_RGB_TOL = 0.03
#: Panel (b)'s named classes and their colours (a chip is its class's colour).
N_CLASS_COLOURS = {**{f"n={n}": c for n, c in N_COLOURS.items()}, "other n": N_OTHER}
CLASS_PALETTE = {
    EVENT_NAMES[mode_tags.AE]: EVENT_COLOURS[mode_tags.AE],
    "transient": TRANSIENT_COLOUR,
    EVENT_NAMES[mode_tags.NTM]: EVENT_COLOURS[mode_tags.NTM],
    **N_CLASS_COLOURS,
}
#: Panel (a)'s colour map.
RAW_CMAP = "gist_heat"
#: The share of the D-alpha panel's height the ELM boxes span; the regime names sit above.
ELM_BOX_TOP = 0.62
#: D-alpha's panel runs to this multiple of its own maximum (room for the regime names).
DALPHA_HEADROOM = 1.38
#: The signal TokEye segments and the raw panel draws: the cross-power spectrogram
#: of the CO2 interferometer chords R0 and V1 (corpus `co2` rows 0 and 1, 500 kHz),
#: the AE review page's rows, in which a mode both chords see stands out of the noise
#: each sees alone. The n colouring and the NTM tags still read the Mirnov array,
#: gated by this mask.
SOURCE_GROUP = "co2"
SOURCE_ROW = 0
PARTNER_ROW = 1
SOURCE_TITLE = "CO2 R0 x V1 cross-power"
INK = "#222222"
BAR = (0.12, 0.76)
FONT = 7
#: The gridspec's right edge: the panels run to it, nothing sits beside them.
RIGHT = 0.98
#: Panel (b) names its events in place with chips (`label_events`), measured in print
#: pixels (the 150-dpi PNG): the largest gap between a chip and the pixels it names
#: with no leader line, the margin kept to the panel edge, to other events' pixels
#: and to other chips, the dilation that merges one event's specks into one region
#: (AE is the sparsest), and the smallest region that is named (below it the colour
#: is specks; the n=3 and other-n colours show only as 1-3 pixel scraps here);
#: colours are told apart by hue, within `LABEL_HUE_TOL` degrees, and a pixel names
#: a class only where that class's own artists draw it (`drawn_classes`). Chips keep
#: `LABEL_CHIP_GAP_PX` from one another and leader lines `LABEL_LEADER_CLEAR_PX`.
LABEL_NEAR_PX = 8
LABEL_MARGIN_PX = 4
LABEL_CHIP_GAP_PX = 6
LABEL_LEADER_CLEAR_PX = 2
LABEL_MERGE_PX = {"AE": 10}
LABEL_MERGE_DEFAULT_PX = 3
LABEL_MIN_REGION_PX = 30
LABEL_HUE_TOL = 6.0
#: What covering a white (coherent-mask) pixel costs a chip, in px of distance from
#: the event it names, per unit of the chip's area that is white.
LABEL_WHITE_COST = 30.0
LABEL_LEADER_LW = 0.5
#: A leader is stroked in the panel's black this far beyond each side, so where it
#: crosses a red transient line no print pixel mixes the two (a blue leader over
#: red would blend to a purple the AE audit reads as pink): over a pixel, at least.
LABEL_LEADER_HALO_PT = 0.6


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
    y, fs, t0_s, t1_s = masks.read_waveform(
        paths.corpus_file(shot), SOURCE_GROUP, SOURCE_ROW
    )
    y2, fs2, t0_2, t1_2 = masks.read_waveform(
        paths.corpus_file(shot), SOURCE_GROUP, PARTNER_ROW
    )
    if (y.shape, fs, t0_s, t1_s) != (y2.shape, fs2, t0_2, t1_2):
        raise ValueError(f"shot {shot}: the two CO2 chords are not on one clock")
    out: dict = {}
    for name, decim in (("wide", 1), ("zoom", masks.ZOOM_DECIM)):
        spec, meta = masks.prep_cross(y, y2, fs_hz=fs, decim=decim)
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
            SOURCE_GROUP,
            SOURCE_ROW,
            inspect.getsource(run_tokeye),
            partner_row=PARTNER_ROW,
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


#: The events whose rows a table can supply, by track key (the event's folder under
#: the label tables): `--rows EVENT=CSV`, else the review page's source table.
ROW_EVENTS = (
    mode_tags.AE,
    mode_tags.NTM,
    "confinement",
    "edge_localized_mode",
    mode_tags.SAWTOOTH,
)
SUPPLIED = "--rows"
POINTER = "review source.json"
#: Metadata keys that name a model behind a table's rows: its rows are that
#: detector's output at a threshold (`table_kind`).
DETECTOR_KEYS = ("threshold", "evaluation", "model", "checkpoint")


@dataclass(frozen=True)
class TableTrack(lf.Track):
    """A track read whole from one table: `table` says which (`how`, `SUPPLIED`
    or `POINTER`; `path`; the table's `method` and `version`; its `kind`,
    `table_kind`)."""

    table: dict | None = None


def parse_rows(items) -> dict[str, Path]:
    """`--rows EVENT=CSV` items as {event: table}: each a known event, once."""
    found: dict[str, Path] = {}
    for item in items or ():
        event, sep, path = item.partition("=")
        if not sep or event not in ROW_EVENTS:
            raise SystemExit(
                f"--rows {item}: want EVENT=CSV, EVENT one of {', '.join(ROW_EVENTS)}"
            )
        if event in found:
            raise SystemExit(f"--rows: {event} is given twice")
        if not Path(path).is_file():
            raise SystemExit(f"--rows {item}: no such file")
        found[event] = Path(path)
    return found


def table_rows(path: Path) -> dict[int, tuple[lf.Row, ...]]:
    """A table's rows by shot: a format table, or a merged regime table (its
    `regimes` letters, `label_figure.REGIME_CODES`)."""
    with open(path) as handle:
        header = handle.readline().strip().split(",")
    return lf.read_rows(path, regimes="regimes" in header)


def table_kind(path: Path, meta: dict, shot: int) -> str:
    """What a table's rows on `shot` are, from its metadata: "detector" where it
    names a model, threshold or held-out evaluation (`DETECTOR_KEYS`: a model's
    output at a threshold, with that model's caveats); "imported" for an event's
    `format/` table; "curated" where its curated source (`sources.curated.table`,
    as the merged regime table's) holds the shot; else "labels", a rule's."""
    if any(meta.get(k) is not None for k in DETECTOR_KEYS):
        return "detector"
    if path.parent.name == "format":
        return "imported"
    curated = ((meta.get("sources") or {}).get("curated") or {}).get("table")
    if curated and Path(curated).is_file() and shot in table_rows(Path(curated)):
        return "curated"
    return "labels"


def no_q1_rows(path: Path, meta: dict, shot: int) -> list[list[float]] | None:
    """A q = 1-gated table's no-q=1 rows on `shot`: its absent rows with a blank
    confidence (the gate's, not the physics rule's), as [start, end] ms; None
    where the table has no such gate (its metadata names no `rule.gate.no_q1`)."""
    if not ((meta.get("rule") or {}).get("gate") or {}).get("no_q1"):
        return None
    with open(path) as handle:
        header = handle.readline().strip().split(",")
    if "confidence" not in header:
        return None
    frame = pd.read_csv(path, index_col=False)
    gate = frame[
        (frame["shot"] == shot)
        & (frame["category"] == ABSENT)
        & frame["confidence"].isna()
    ]
    return [[float(a), float(b)] for a, b in zip(gate["t_start"], gate["t_end"])]


def table_track(
    paths: Paths, shot: int, key: str, supplied: Path | None = None
) -> TableTrack | None:
    """`key`'s rows on `shot` from `supplied` (`--rows`), else from the review
    page's source table (`review/source.json`); the table's `.meta.json`, where
    there is one, gives its method, version and kind (`table_kind`: a detector's
    rows keep its threshold, a curated or imported table is the legacy tier).
    None where the event has no pointer or its table holds no row for the shot;
    a supplied table without the shot is an error."""
    spec = next(s for s in lf.TRACKS if s.key == key)
    if supplied is not None:
        path, how = Path(supplied), SUPPLIED
    else:
        pointer = labels.pointer_path(paths.label_tables / key)
        if not pointer.is_file():
            return None
        entry = json.loads(pointer.read_text())
        path, how = Path(entry["table"]), POINTER
        if not path.is_file():
            raise FileNotFoundError(f"{pointer} names {path}, which does not exist")
    meta_file = path.with_suffix(".meta.json")
    meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
    if supplied is None:
        meta = {**meta, "method": entry["method"], "version": entry["version"]}
    method, version = meta.get("method"), meta.get("version")
    rows = table_rows(path).get(shot)
    if not rows:
        if supplied is not None:
            raise SystemExit(f"--rows {key}={path}: no row for shot {shot}")
        return None
    kind = table_kind(path, meta, shot)
    name = " ".join(x for x in (method, version) if x) or path.name
    if kind == "detector":
        threshold = meta.get("threshold")
        what = name + (f", p>={threshold:g}" if threshold is not None else "")
    else:
        what = f"{name} {figure_sources.TABLE_KINDS[kind]}"
    tier = lf.LEGACY if kind in ("curated", "imported") else lf.GENERATED
    source = lf.Source(tier, f"{what} ({how})", lambda p, t=path: t)
    table = {
        "how": how,
        "path": str(path),
        "method": method,
        "version": version,
        "kind": kind,
    }
    gate = no_q1_rows(path, meta, shot) if key == mode_tags.SAWTOOTH else None
    if gate is not None:
        # the no-q=1-negatives view: the gate's absent time, and the shot's total
        # as the table's metadata states it
        table["no_q1_rows_ms"] = gate
        table["no_q1_ms"] = ((meta.get("per_shot") or {}).get(str(shot)) or {}).get(
            "no_q1_ms"
        )
    return TableTrack(spec, source, path, rows, table)


def shot_tracks(
    paths: Paths,
    candidate: lf.Candidate,
    sawtooth_labels: Path | None = None,
    ae_labels: Path | None = None,
    show_sawtooth: bool = False,
    outside_rows: Path | None = None,
    rows: dict[str, Path] | None = None,
) -> tuple[tuple[lf.Track, ...], Path | None]:
    """The catalog's tracks on `candidate`: each event's table (`rows`, else its
    review source table, `table_track`) where one holds the shot, else the
    catalog's (`catalog_tracks`); and the CO2 file a generated AE track ran over
    (None where the AE track is not generated). A table's sawtooth row is drawn
    whether or not `show_sawtooth` asks for one. A gold evaluation shot draws no
    expert review: the catalog skips that tier for it, and an expert row that
    still reaches it is an error."""
    rows = rows or {}
    tables = {
        key: track
        for key in ROW_EVENTS
        if (track := table_track(paths, candidate.shot, key, rows.get(key)))
    }
    gold = is_gold(paths.label_tables, candidate.shot)
    tracks, co2 = catalog_tracks(
        paths,
        candidate,
        sawtooth_labels,
        ae_labels,
        show_sawtooth,
        outside_rows,
        skip=frozenset(tables),
        expert=not gold,
    )
    out = [*tracks, *tables.values()]
    expert = [t.spec.key for t in out if t.source and t.source.tier == lf.SILVER]
    if expert and gold:
        raise SystemExit(
            f"{candidate.shot} is a gold evaluation shot: the {', '.join(expert)} "
            "row would draw the expert review; give it with --rows EVENT=CSV"
        )
    order = {s.key: i for i, s in enumerate(lf.TRACKS)}
    return tuple(sorted(out, key=lambda t: order[t.spec.key])), co2


def catalog_tracks(
    paths: Paths,
    candidate: lf.Candidate,
    sawtooth_labels: Path | None = None,
    ae_labels: Path | None = None,
    show_sawtooth: bool = False,
    outside_rows: Path | None = None,
    skip: frozenset = frozenset(),
    expert: bool = True,
) -> tuple[tuple[lf.Track, ...], Path | None]:
    """The tracks of the events not in `skip` (`shot_tracks`' tables):
    `label_figure`'s, the AE generated from the corpus CO2, else from the raw
    cache; and the CO2 file the generated AE track ran over (None where the AE
    track is not generated). Not `expert` (a gold evaluation shot): no event's
    expert review is read, each falls through to its next tier."""
    chosen_ae = (
        None
        if mode_tags.AE in skip
        else figure_sources.reviewed_or_stored_ae(
            paths, candidate.shot, ae_labels, expert=expert
        )
    )
    specs = tuple(
        s
        if expert
        else replace(s, sources=tuple(x for x in s.sources if x.tier != lf.SILVER))
        for s in lf.TRACKS
        if s.key not in ("confinement", mode_tags.SAWTOOTH, *skip)
        and (chosen_ae is None or s.key != mode_tags.AE)
    )
    read = lf.read_sources(paths, specs)
    read = lf.generate(paths, [candidate], read, specs)
    selected = {t.spec.key: t for t in lf.tracks_of(candidate.shot, read, specs)}
    if "confinement" not in skip:
        selected["confinement"] = figure_sources.confinement_track(
            paths, candidate.shot, expert=expert
        )
    for key in (mode_tags.NTM, "edge_localized_mode"):
        # A shot in no cohort split has no row in the frame models' merged tables:
        # their rows for it come from a single-shot run saved beside its metadata.
        table = None if outside_rows is None else Path(outside_rows) / f"{key}.csv"
        if key in skip or table is None or not table.is_file():
            continue
        if selected[key].source is None:
            held = lf.read_rows(table).get(candidate.shot)
            if held:
                what = selected[key].spec.sources[-1].what
                source = lf.Source(
                    lf.GENERATED,
                    f"{what}, run on this shot alone",
                    lambda p, t=table: t,
                )
                selected[key] = lf.Track(selected[key].spec, source, table, held)
    if show_sawtooth and mode_tags.SAWTOOTH not in skip:
        selected[mode_tags.SAWTOOTH] = figure_sources.sawtooth_track(
            paths, candidate.shot, sawtooth_labels, [], expert=expert
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

    def __init__(
        self,
        z: dict,
        name: str,
        lo: float,
        hi: float,
        t0: float,
        t1: float,
        transient_cut: float = TRANSIENT_THRESHOLD,
    ):
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
            threshold=TOKEYE_THRESHOLD,
        )
        # The transient channel at its own display cut, after the same small-object
        # removal; drawn only where the mask is not lit (the mask is drawn in full).
        tra = mode_tags.clean(
            z[f"{name}_tra"].astype(np.float32) >= transient_cut,
            mode_tags.MIN_SIZE[name],
        )
        tra &= ~lit
        self.tra = tra[rows][:, cols]
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


def draw_raw(ax, band: Band, cmap: str = RAW_CMAP) -> None:
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


def draw_processed(ax, band: Band, below: float | None = None):
    """White coherent mask: white means n was not measured. Transparent where the
    mask is unlit, so the panel's black face shows there, and the red transient
    overlay beneath it (`MASK_Z` over `TRANSIENT_Z`; drawn only with `SHOW_TRANSIENT`)
    never covers a lit pixel.
    `below` (kHz) clips the image to the frequencies under it, where the pass
    above takes over. Returns the image."""
    lit = pool(band.lit.astype(np.float32), band.k, "max")
    rgba = np.ones((*lit.shape, 4), np.float32)
    rgba[..., 3] = lit
    x0, x1, y0, _ = extent = band.extent()
    image = ax.imshow(
        rgba,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        zorder=MASK_Z,
    )
    if below is not None:
        image.set_clip_path(
            Rectangle((x0, y0), x1 - x0, below - y0, transform=ax.transData)
        )
    return image


def print_pool(ax, band: Band, mask: np.ndarray) -> np.ndarray:
    """`mask` max-pooled to the print pixels (`DPI_PNG`) its share of `ax` covers,
    so a line one column wide survives at the page's size."""
    pos = ax.get_position()
    share = (band.f[-1] - band.f[0]) / np.diff(ax.get_ylim())[0]
    nf = min(mask.shape[0], max(1, round(pos.height * HEIGHT_IN * DPI_PNG * share)))
    nt = min(mask.shape[1], max(1, round(pos.width * PAGE_IN * DPI_PNG)))
    ri = np.linspace(0, mask.shape[0], nf, endpoint=False).astype(int)
    ci = np.linspace(0, mask.shape[1], nt, endpoint=False).astype(int)
    return np.maximum.reduceat(np.maximum.reduceat(mask, ri, axis=0), ci, axis=1)


def on_grid(mask, t_src, f_src, t_dst, f_dst) -> np.ndarray:
    """`mask` (rows `f_src`, columns `t_src`) on the grid `f_dst` x `t_dst`: a cell
    is lit where a source row inside its frequency cell (else the nearest source
    row) is lit in the source column whose time is nearest; cells outside the
    source's band are not lit."""
    df = float(np.median(np.diff(f_dst))) if len(f_dst) > 1 else 0.0
    lo = np.searchsorted(f_src, f_dst - df / 2)
    hi = np.searchsorted(f_src, f_dst + df / 2)
    nearest = np.abs(f_src[None, :] - f_dst[:, None]).argmin(axis=1)
    lo, hi = np.where(hi > lo, lo, nearest), np.where(hi > lo, hi, nearest + 1)
    cum = np.vstack([np.zeros((1, mask.shape[1]), np.int64), np.cumsum(mask, 0)])
    rows = cum[hi] - cum[lo] > 0
    df_src = float(np.median(np.diff(f_src))) if len(f_src) > 1 else 0.0
    rows[(f_dst < f_src[0] - df_src / 2) | (f_dst > f_src[-1] + df_src / 2)] = False
    i = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    cols = np.where(t_dst - t_src[i - 1] <= t_src[i] - t_dst, i - 1, i)
    return rows[:, cols]


def transient_overlay(wide: Band, zoom: Band) -> np.ndarray:
    """The red overlay of (b) on `wide`'s grid, over its whole band: the wide pass's
    transient pixels (`Band.tra`: at the display cut, small objects removed, where
    the wide pass's mask is not lit) less where (b) draws the zoom pass's mask
    instead (below `ZOOM_TOP_KHZ`), so red never covers a drawn mask pixel."""
    tra = wide.tra.copy()
    below = wide.f < ZOOM_TOP_KHZ
    tra[below] &= ~on_grid(zoom.lit, zoom.t, zoom.f, wide.t, wide.f[below])
    return tra


def draw_transient(ax, band: Band, tra: np.ndarray) -> int:
    """`tra` (on `band`'s grid) in red beneath the mask (`TRANSIENT_Z`; transparent
    elsewhere), max-pooled to print pixels and drawn without interpolation: a print
    pixel that pooling turns red under a lit mask pixel stays white. Returns the
    print-pixel columns drawn red."""
    if not tra.any():
        return 0
    pooled = print_pool(ax, band, tra)
    rgba = np.zeros((*pooled.shape, 4), np.float32)
    rgba[pooled] = (*to_rgb(TRANSIENT_COLOUR), 1.0)
    ax.imshow(
        rgba,
        origin="lower",
        aspect="auto",
        extent=band.extent(),
        interpolation="none",
        zorder=TRANSIENT_Z,
        gid=DRAWN_GID + "transient",
    )
    return int(pooled.any(axis=0).sum())


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
                ).set_gid(DRAWN_GID + EVENT_NAMES[event])
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
        artist.set_gid(DRAWN_GID + EVENT_NAMES[event])
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
        gid=DRAWN_GID + EVENT_NAMES[event],
    )
    artist.set_clip_path(event_clip(ax, spans, event), ax.transData)
    return shown


def chip(ax, x, y, text, colour, text_colour="black", alpha=0.95, **kw):
    """A label with a coloured background; returns the text."""
    return ax.text(
        x,
        y,
        text,
        fontsize=FONT,
        color=text_colour,
        bbox={"boxstyle": "round,pad=0.12", "fc": colour, "ec": "none", "alpha": alpha},
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


def display_runs(rows) -> list[lf.Row]:
    """`rows` in time order, each run of adjacent rows (one ending where the next
    starts) of one category and crowd flag drawn as one row; display only."""
    runs: list[lf.Row] = []
    for r in sorted(rows, key=lambda r: (r.t_start, r.t_end)):
        last = runs[-1] if runs else None
        if (
            last is not None
            and (last.category, last.crowd) == (r.category, r.crowd)
            and r.t_start <= last.t_end + 1e-6
        ):
            runs[-1] = replace(last, t_end=max(last.t_end, r.t_end))
        else:
            runs.append(r)
    return runs


def hatch_segments(box, step: float) -> list[tuple[tuple, tuple]]:
    """Slash lines (slope 1) across `box` (x0, y0, x1, y1, display pixels), one where
    x - y is a multiple of `step` pixels on the page (so adjacent boxes' lines meet),
    each clipped to the box: [((xa, ya), (xb, yb)), ...]."""
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0 or step <= 0:
        return []
    out = []
    for c in np.arange(math.ceil((x0 - y1) / step) * step, x1 - y0, step):
        a, b = max(x0, y0 + c), min(x1, y1 + c)
        if b > a:
            out.append(((a, a - c), (b, b - c)))
    return out


def hatch_lines(ax, x0, x1, y0, y1, colour, transform=None, zorder=2.5):
    """Hatch the box `x0`-`x1` by `y0`-`y1` (in `transform`'s coordinates, the
    axes' data by default, x clipped to the view) with explicit "/" line segments
    `UNCERTAIN_HATCH_STEP_PT` apart, `UNCERTAIN_HATCH_LW` wide, in `colour`: every
    PDF viewer draws them as Agg does (poppler draws a PDF hatch pattern faint).
    A corner piece shorter than `UNCERTAIN_HATCH_MIN_PT` is dropped; a box too
    narrow to keep any piece gets one vertical line at its middle instead, so
    uncertain time never prints blank. Returns the LineCollection (None for a box
    out of view or empty)."""
    transform = transform or ax.transData
    lo, hi = ax.get_xlim()
    x0, x1 = max(x0, lo), min(x1, hi)
    if x1 <= x0:
        return None
    (px0, py0), (px1, py1) = transform.transform([(x0, y0), (x1, y1)])
    if py1 <= py0:
        return None
    step = UNCERTAIN_HATCH_STEP_PT * ax.figure.dpi / 72
    shortest = UNCERTAIN_HATCH_MIN_PT * ax.figure.dpi / 72
    segments = [
        (a, b)
        for a, b in hatch_segments((px0, py0, px1, py1), step)
        if math.hypot(b[0] - a[0], b[1] - a[1]) >= shortest
    ]
    if not segments:
        middle = (px0 + px1) / 2
        segments = [((middle, py0), (middle, py1))]
    inverse = transform.inverted()
    lines = LineCollection(
        [inverse.transform(np.asarray(s, float)) for s in segments],
        colors=colour,
        linewidths=UNCERTAIN_HATCH_LW,
        capstyle="butt",
        transform=transform,
        zorder=zorder,
    )
    ax.add_collection(lines, autolim=False)
    return lines


def track_bars(ax, track: lf.Track, colour: str, regimes=None, bar=BAR) -> list:
    """`track`'s runs (`display_runs`) as bars; returns the runs drawn."""
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for side in ("left", "top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    if track.source is None:
        return []
    runs = display_runs(track.rows)
    for r in runs:
        span = [(r.t_start, r.t_end - r.t_start)]
        if r.category == ABSENT:
            ax.broken_barh(span, bar, facecolor=ABSENT_GREY, lw=0)
            continue
        if regimes is not None:
            face = regimes.get(r.category, "#888888")
            uncertain = r.category == 5
            hatch_colour = UNCERTAIN_CLASS_COLOUR
        else:
            face = colour
            uncertain = r.category == UNCERTAIN
            hatch_colour = colour
            if r.category == NOT_OBSERVABLE:
                continue  # blank: no assessment; hatching only means uncertain
        if uncertain:
            # white under explicit hatch lines (`hatch_lines`), no edge
            ax.broken_barh(span, bar, facecolor="white", lw=0)
            hatch_lines(ax, r.t_start, r.t_end, bar[0], bar[0] + bar[1], hatch_colour)
        else:
            ax.broken_barh(span, bar, facecolor=face, edgecolor=face, lw=0)
        if r.category in (PRESENT, UNCERTAIN) and r.crowd == 1 and regimes is None:
            ax.plot(
                [r.t_start, r.t_end],
                [bar[0] + bar[1] / 2] * 2,
                color=colour if uncertain else "white",
                lw=0.4,
                marker="o",
                ms=1.8,
                markeredgecolor=colour,
                markerfacecolor="white",
                zorder=6,
            )
    return runs


def ink_on(colour: str) -> str:
    """Text that reads on `colour`: white on a dark fill, `INK` on a light one."""
    r, g, b = to_rgb(colour)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else INK


def name_runs(ax, runs, t0: float, t1: float, bar=BAR) -> list[dict]:
    """Write each confinement class's short name (`CLASS_LEGEND`) on its runs in
    view where it fits with `RUN_TEXT_PAD_PT` to spare; each run's text and
    bounds (None where it does not fit), for the record."""
    fig = ax.figure
    out = []
    for r in runs:
        a, b = max(r.t_start, t0), min(r.t_end, t1)
        if r.category not in CLASS_LEGEND or b <= a:
            continue
        text = ax.text(
            (a + b) / 2, bar[0] + bar[1] / 2, CLASS_LEGEND[r.category],
            fontsize=FONT, ha="center", va="center_baseline",
            color=ink_on(CLASS_COLOURS[r.category]), zorder=7,
        )  # fmt: skip
        fig.draw_without_rendering()
        width = ax.transData.transform((b, 0))[0] - ax.transData.transform((a, 0))[0]
        box = text.get_window_extent()
        fits = box.width + 2 * RUN_TEXT_PAD_PT * fig.dpi / 72 <= width
        if not fits:
            text.remove()
        out.append(
            {
                "text": CLASS_LEGEND[r.category],
                "category": r.category,
                "run_ms": [a, b],
                "font_pt": FONT,
                "bounds": list(box.transformed(fig.transFigure.inverted()).extents)
                if fits
                else None,
            }
        )
    return out


def draw_frequency_panels(
    ax, raw: Band, zoom: Band, top: Band, transient: bool = SHOW_TRANSIENT
) -> dict[str, int]:
    """The raw spectrogram and the TokEye mask, each on one linear 0-150 kHz axis.

    The raw panel is the wide pass alone on one colour scale. The mask panel takes
    the zoom pass below `ZOOM_TOP_KHZ` and the wide pass above it (`top`, drawn
    in place of the zoom pass's decimation roll-off); no axis break or stretching.
    With `transient` (off by default, `SHOW_TRANSIENT`) the wide pass's transient
    channel is drawn in red across the whole band (`transient_overlay`), beneath
    the mask, on the panel's black face; returns the print-pixel columns it draws
    red, by pass (`draw_transient`), and nothing (an empty dict) without it."""
    for name in ("raw", "pr"):
        ax[name].set_ylim(0, TOP_KHZ)
        ax[name].set_yticks(FREQ_TICKS_KHZ)
        ax[name].set_ylabel("kHz", labelpad=2)
    draw_raw(ax["raw"], raw)
    ax["pr"].set_facecolor(PANEL_FACE)
    red = (
        {TRANSIENT_PASS: draw_transient(ax["pr"], raw, transient_overlay(raw, zoom))}
        if transient
        else {}
    )
    draw_processed(ax["pr"], zoom, below=top.extent()[2])
    draw_processed(ax["pr"], top)
    return red


def panel_raster(fig, ax, only=None):
    """The axes' interior as the 150-dpi PNG draws it: RGB floats, rows top to
    bottom, with the interior's pixel box in the whole image. With `only` (a list
    of `ax`'s artists), those artists alone on a transparent page: RGBA floats
    (straight alpha), every other artist hidden for the render and then restored."""
    hidden = []
    if only is not None:
        keep = {id(a) for a in only}
        for artist in fig.findobj(include_self=False):
            if artist is ax or id(artist) in keep or not artist.get_visible():
                continue
            artist.set_visible(False)
            hidden.append(artist)
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=DPI_PNG, transparent=only is not None)
    finally:
        for artist in hidden:
            artist.set_visible(True)
    buf.seek(0)
    img = mpimg.imread(buf, format="png")
    img = img[..., :3] if only is None else img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = ax.get_position().extents
    box = (round((1 - y1) * h), round((1 - y0) * h), round(x0 * w), round(x1 * w))
    return img[box[0] : box[1], box[2] : box[3]], box, (h, w)


def drawn_classes(fig, ax) -> dict[str, np.ndarray]:
    """Where each of panel (b)'s named classes is drawn, in `panel_raster`'s pixels:
    its own artists (gid `DRAWN_GID` + class) rendered alone, the pixels they cover
    at least `DRAWN_MIN_ALPHA`; the n view (one image of every n) split by colour,
    within `DRAWN_RGB_TOL`. A blend of two classes' colours where they meet (an
    anti-aliased TM outline over an n=1 ridge has n=3's hue) is neither's unless
    one of them draws it."""
    groups: dict[str, list] = {}
    for artist in ax.get_children():
        gid = artist.get_gid() or ""
        if gid.startswith(DRAWN_GID):
            groups.setdefault(gid.removeprefix(DRAWN_GID), []).append(artist)
    shape = None
    out = {}
    for group, artists in groups.items():
        rgba, _, _ = panel_raster(fig, ax, only=artists)
        shape = rgba.shape[:2]
        cover = rgba[..., 3] >= DRAWN_MIN_ALPHA
        if group != "n":
            out[group] = cover
            continue
        for name, colour in N_CLASS_COLOURS.items():
            near = np.abs(rgba[..., :3] - np.array(to_rgb(colour))).max(axis=-1)
            out[name] = cover & (near <= DRAWN_RGB_TOL)
    if shape is not None:
        empty = np.zeros(shape, bool)
        for name in CLASS_PALETTE:
            out.setdefault(name, empty)
    return out


def event_classes(rgb) -> dict[str, np.ndarray]:
    """The pixels of each coloured event in a raster, by hue (the mask is grey)."""
    hsv = rgb_to_hsv(rgb)
    strong = (hsv[..., 1] > 0.3) & (hsv[..., 2] > 0.2)
    hue = hsv[..., 0] * 360.0
    out = {}
    for name, colour in CLASS_PALETTE.items():
        centre = rgb_to_hsv(np.array(to_rgb(colour)))[0] * 360.0
        delta = np.abs((hue - centre + 180.0) % 360.0 - 180.0)
        out[name] = strong & (delta <= LABEL_HUE_TOL)
    return out


def largest_region(mask: np.ndarray, merge_px: int):
    """The pixels of `mask` in its largest region (specks within `merge_px` of each
    other are one region), and their count; None if `mask` is empty."""
    if not mask.any():
        return None
    structure = np.ones((3, 3), bool)
    grown = ndimage.binary_dilation(mask, structure, iterations=merge_px)
    regions, count = ndimage.label(grown, structure)
    sizes = np.bincount(regions[mask], minlength=count + 1)
    sizes[0] = 0
    best = int(sizes.argmax())
    return mask & (regions == best), int(sizes[best])


def regions_at_least(mask: np.ndarray, merge_px: int, min_px: int) -> np.ndarray:
    """The pixels of `mask` in every region (merged as in `largest_region`) of at
    least `min_px` pixels."""
    structure = np.ones((3, 3), bool)
    grown = ndimage.binary_dilation(mask, structure, iterations=merge_px)
    regions, count = ndimage.label(grown, structure)
    sizes = np.bincount(regions[mask], minlength=count + 1)
    sizes[0] = 0
    return mask & (sizes >= min_px)[regions]


def text_colour_on(colour: str) -> str:
    """Black or white, whichever reads better on `colour` (WCAG contrast)."""

    def luminance(rgb):
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    lum = luminance(to_rgb(colour))
    return "black" if (lum + 0.05) / 0.05 >= 1.05 / (lum + 0.05) else "white"


def segment_pixels(a, b) -> list[tuple[int, int]]:
    """The (row, col) pixels along the segment `a` to `b`."""
    n = max(2, math.ceil(2 * math.hypot(b[0] - a[0], b[1] - a[1])))
    t = np.linspace(0.0, 1.0, n)
    rows = np.rint(a[0] + t * (b[0] - a[0])).astype(int)
    cols = np.rint(a[1] + t * (b[1] - a[1])).astype(int)
    return list(dict.fromkeys(zip(rows.tolist(), cols.tolist(), strict=True)))


def find_chip_spot(target, size, blocked, white, avoid, floor, near_only):
    """Where a chip of `size` (rows, cols) goes beside the pixels `target`.

    A candidate keeps clear of `blocked` and of the panel edge, and, when `floor` is
    given, sits wholly above that row (the crowded 0-30 kHz band below it). Cost is
    its distance to `target` plus `LABEL_WHITE_COST` times its share of white mask
    pixels; with `near_only` it must lie within `LABEL_NEAR_PX` of `target`, else a
    farther one needs a straight leader to its nearest target pixel that crosses
    nothing in `avoid`. Returns (top, left, gap_px, leader) or None; the leader is
    (from, to) pixels, or None when the chip is near enough to need none."""
    bh, bw = size
    h, w = target.shape
    dist, (near_r, near_c) = ndimage.distance_transform_edt(
        ~target, return_indices=True
    )
    walls = blocked.copy()
    edge = LABEL_MARGIN_PX + 1
    walls[:edge] = walls[-edge:] = True
    walls[:, :edge] = walls[:, -edge:] = True
    hit = ndimage.maximum_filter(walls, size=(bh, bw), mode="constant", cval=True)
    gap = ndimage.minimum_filter(dist, size=(bh, bw), mode="constant", cval=np.inf)
    shade = ndimage.uniform_filter(
        white.astype(float), size=(bh, bw), mode="constant", cval=0.0
    )
    rows, cols = np.mgrid[0:h, 0:w]
    top = rows - bh // 2
    ok = ~hit & np.isfinite(gap)
    if floor is not None:
        ok &= top + bh - 1 <= floor - LABEL_MARGIN_PX
    if near_only:
        ok &= gap <= LABEL_NEAR_PX
    rr, cc = np.nonzero(target)
    centre = (rr.mean(), cc.mean())
    cost = gap + LABEL_WHITE_COST * shade
    cost = cost + 0.02 * np.hypot(rows - centre[0], cols - centre[1])
    cost = np.where(ok, cost, np.inf)
    order = np.argsort(cost, axis=None, kind="stable")
    for index in order[: min(int(ok.sum()), 4000)]:
        r, c = divmod(int(index), w)
        t, left = r - bh // 2, c - bw // 2
        box = dist[t : t + bh, left : left + bw]
        qr, qc = np.unravel_index(int(box.argmin()), box.shape)
        q = (t + int(qr), left + int(qc))
        p = (int(near_r[q]), int(near_c[q]))
        if gap[r, c] <= LABEL_NEAR_PX:
            return t, left, float(gap[r, c]), None
        if near_only:
            continue
        crossed = [
            pt
            for pt in segment_pixels(q, p)
            if avoid[pt]
            and not target[pt]
            and max(abs(pt[0] - p[0]), abs(pt[1] - p[1])) > 1
        ]
        if not crossed:
            return t, left, float(gap[r, c]), (q, p)
    return None


def label_events(fig, pr, window, transient_drawn: bool) -> dict:
    """Name panel (b)'s events in place: a chip (the event's colour, black or white
    text) beside the largest region of each class, read from the panel's own
    pixels so it follows what is drawn. A class's pixels are those of its hue that
    its own artists draw (`drawn_classes`): a blend where two colours meet names
    neither, so a chip names a class only where at least `LABEL_MIN_REGION_PX`
    print pixels of it are drawn; a class with no such region is left unnamed, and
    recorded. A chip never covers another chip, a leader, or a pixel of any event
    colour but the transient's; the AE and transient chips cover no transient pixel
    either, while the TM and n chips may cover a few of the one-pixel transient
    lines (costed like white mask pixels; the record counts them). AE and transient
    chips touch their region's neighbourhood; TM and the n lines, whose band is
    crowded, sit just above 30 kHz with a short leader line to the nearest pixel of
    their region."""
    drawn = drawn_classes(fig, pr)
    rgb, (r0, _, c0, _), (height, width) = panel_raster(fig, pr)
    h, w = rgb.shape[:2]
    # by hue, what any event colour paints (blends too): what a chip keeps clear of
    classes = event_classes(rgb)
    # what names a class: its hue where its own artists draw it
    named = {
        k: v & drawn[k] if k in drawn else np.zeros_like(v) for k, v in classes.items()
    }
    hsv = rgb_to_hsv(rgb)
    white = (hsv[..., 1] < 0.2) & (hsv[..., 2] > 0.7)
    structure = np.ones((3, 3), bool)
    any_event = np.any(list(classes.values()), axis=0)
    blocked = ndimage.binary_dilation(any_event, structure, iterations=LABEL_MARGIN_PX)
    # The TM and n chips sit above the crowded 0-30 kHz band, where the transient
    # lines (display only, one print pixel wide) run through the whole band: those
    # chips may cover a few of them, at the white-pixel cost, rather than sit far
    # away on a long leader; they keep clear of every other event.
    transient = classes["transient"]
    solid = any_event & ~transient
    blocked_solid = ndimage.binary_dilation(
        solid, structure, iterations=LABEL_MARGIN_PX
    )
    lo, hi = pr.get_ylim()
    floor = round(h * (hi - N_VIEW_KHZ) / (hi - lo))
    t0, t1 = window

    def khz(row):
        return hi - (hi - lo) * row / h

    def ms(col):
        return t0 + (t1 - t0) * col / w

    # (text, colour class, chip colour, near-only): placed in this order; the
    # transient chip last, as its lines run through most of the panel and their
    # chip fits beside any of them, the crowded band's chips beside few places
    order = [
        ("AE", "AE", EVENT_COLOURS[mode_tags.AE], True),
        ("TM", "TM", EVENT_COLOURS[mode_tags.NTM], False),
        *[(f"n={n}", f"n={n}", c, False) for n, c in N_COLOURS.items()],
        ("other n", "other n", N_OTHER, False),
        ("transient", "transient", TRANSIENT_COLOUR, True),
    ]
    regions, skipped, wanted = {}, {}, []
    for text, key, colour, near_only in order:
        found = largest_region(
            named[key], LABEL_MERGE_PX.get(key, LABEL_MERGE_DEFAULT_PX)
        )
        regions[text] = {
            "pixels": int(named[key].sum()),
            "hue_pixels": int(classes[key].sum()),
            "largest_region_px": 0 if found is None else found[1],
            "minimum_px": LABEL_MIN_REGION_PX,
        }
        if found is None:
            continue
        if found[1] < LABEL_MIN_REGION_PX:
            skipped[text] = (
                f"largest region {found[1]} px < {LABEL_MIN_REGION_PX} px (specks)"
            )
        elif text == "transient" and not transient_drawn:
            skipped[text] = "transient channel not drawn"
        elif text == "transient":
            # the transient lines run through the whole band: the chip may sit
            # beside any line of a named size, not only the densest cluster
            merge = LABEL_MERGE_PX.get(key, LABEL_MERGE_DEFAULT_PX)
            target = regions_at_least(named[key], merge, LABEL_MIN_REGION_PX)
            wanted.append((text, key, colour, near_only, target))
        else:
            wanted.append((text, key, colour, near_only, found[0]))
    chips = {}
    for text, _, colour, _, _ in wanted:
        chips[text] = chip(
            pr,
            0.5,
            0.5,
            text,
            colour,
            text_colour=text_colour_on(colour),
            alpha=1.0,
            transform=pr.transAxes,
            ha="center",
            va="center",
        )
    fig.draw_without_rendering()
    scale = DPI_PNG / fig.dpi

    def size_of(text):
        extent = chips[text].get_bbox_patch().get_window_extent()
        return math.ceil(extent.height * scale), math.ceil(extent.width * scale)

    # a leader may cross the transient lines (display only, one print pixel wide,
    # and at TokEye's operating point they run through the whole band) and the TM
    # outlines (they enclose the n=1 and n=2 pixels the n chips name, so a leader
    # from above must cross one); a chip still sits clear of both
    crossable = classes["transient"] | classes[EVENT_NAMES[mode_tags.NTM]]
    chip_area = np.zeros((h, w), bool)
    leader_area = np.zeros((h, w), bool)
    placed = []
    for text, key, colour, near_only, target in wanted:
        spot = find_chip_spot(
            target,
            size_of(text),
            (blocked if near_only else blocked_solid)
            | ndimage.binary_dilation(
                chip_area, structure, iterations=LABEL_CHIP_GAP_PX
            )
            | ndimage.binary_dilation(
                leader_area, structure, iterations=LABEL_LEADER_CLEAR_PX
            ),
            white if near_only else white | transient,
            ndimage.binary_dilation(any_event & ~target & ~crossable, structure)
            | chip_area
            | leader_area,
            None if near_only else floor,
            near_only,
        )
        if spot is None:
            chips.pop(text).remove()
            skipped[text] = "no free position beside its region"
            continue
        top, left, gap, leader = spot
        bh, bw = size_of(text)
        chip_area[top : top + bh, left : left + bw] = True
        chips[text].set_position(((left + bw / 2) / w, 1 - (top + bh / 2) / h))
        if leader is not None:
            (qr, qc), (tr, tc) = leader
            pr.add_artist(
                Line2D(
                    [(qc + 0.5) / w, (tc + 0.5) / w],
                    [1 - (qr + 0.5) / h, 1 - (tr + 0.5) / h],
                    transform=pr.transAxes,
                    color=colour,
                    lw=LABEL_LEADER_LW,
                    solid_capstyle="butt",
                    zorder=7,
                    path_effects=[
                        patheffects.withStroke(
                            linewidth=LABEL_LEADER_LW + 2 * LABEL_LEADER_HALO_PT,
                            foreground=PANEL_FACE,
                            capstyle="butt",
                        )
                    ],
                )
            )
            for point in segment_pixels(*leader):
                leader_area[point] = True
        rows_, cols_ = np.nonzero(target)
        placed.append((text, key, colour, gap, leader, target, rows_, cols_))
    fig.draw_without_rendering()
    inverse = fig.transFigure.inverted()
    records = []
    for text, key, colour, gap, leader, target, rows_, cols_ in placed:
        extent = chips[text].get_bbox_patch().get_window_extent().transformed(inverse)
        # the chip's box in the panel's pixels, rounded outward, against the raster
        # as it was before any chip: no event pixel may lie under it
        x0, y0, x1, y1 = extent.extents
        rows = slice(
            max(0, math.floor((1 - y1) * height) - r0),
            math.ceil((1 - y0) * height) - r0,
        )
        cols = slice(max(0, math.floor(x0 * width) - c0), math.ceil(x1 * width) - c0)
        records.append(
            {
                "text": text,
                "colour_class": key,
                "chip_colour": colour,
                "text_colour": text_colour_on(colour),
                "font_pt": FONT,
                "bounds": [x0, y0, x1, y1],
                "region_px": int(target.sum()),
                "region_ms": [ms(float(cols_.min())), ms(float(cols_.max() + 1))],
                "region_khz": [khz(float(rows_.max() + 1)), khz(float(rows_.min()))],
                "gap_px": gap,
                "leader": None
                if leader is None
                else {
                    "from": [(leader[0][1] + 0.5) / w, 1 - (leader[0][0] + 0.5) / h],
                    "to": [(leader[1][1] + 0.5) / w, 1 - (leader[1][0] + 0.5) / h],
                    "axes_fraction": True,
                },
                "covered_event_pixels": int(solid[rows, cols].sum()),
                "covered_transient_pixels": int(transient[rows, cols].sum()),
                "covered_white_share": float(white[rows, cols].mean()),
            }
        )
    return {
        "placed": records,
        "skipped": skipped,
        "regions": regions,
        "rule": "chips read from the panel's own pixels before they are drawn: each "
        "class's largest region of pixels of its hue that its own artists draw "
        "(rendered alone, covering at least half a pixel; a blend where two colours "
        "meet names neither), merged within "
        f"{LABEL_MERGE_DEFAULT_PX} px (AE {LABEL_MERGE_PX['AE']} px), named when it "
        f"has at least {LABEL_MIN_REGION_PX} print pixels; AE and transient chips "
        f"within {LABEL_NEAR_PX} px of the region (the transient chip beside any "
        "of its regions of that size), TM and n chips above the "
        f"{N_VIEW_KHZ:g} kHz band with a leader line; no chip on an event pixel, "
        "another chip or a leader (the TM and n chips may cover transient lines, "
        "costed like white pixels); a leader crosses no event pixel but the "
        "transient lines and the TM outlines",
    }


def display_states(display_tracks, t0, t1):
    """What the label rows draw, for the record: the present colours and which of
    present, uncertain, absent and blank (unassessed) time shows. The figure has
    no key for them: the draft caption's (f) sentence and the appendix name each
    state shown (`figure_sources.row_key`: colour present, grey absent, hatched
    uncertain, blank unassessed)."""
    colours = [
        EVENT_COLOURS[t.spec.key]
        for t in display_tracks
        if t.spec.key in EVENT_COLOURS and figure_sources.has_present_time(t, (t0, t1))
    ]
    display_states = [
        r for t in display_tracks for r in figure_sources.state_intervals(t, (t0, t1))
    ]
    display_keys = []
    if colours:
        display_keys.append("present")
    if any(r["state"] == "uncertain" for r in display_states):
        display_keys.append("uncertain")
    if any(r["state"] == "absent" for r in display_states):
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
        display_keys.append("blank")
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
    transient_cut: float = TRANSIENT_THRESHOLD,
    transient: bool = SHOW_TRANSIENT,
) -> dict:
    """The figure for `candidate` over `t0`-`t1` ms; returns what the record holds.
    Panel (b) draws the transient overlay, cut at `transient_cut`, only with
    `transient` (off by default, `SHOW_TRANSIENT`)."""
    if not 0 < transient_cut <= 1:
        raise ValueError(f"transient cut {transient_cut}: want 0 < cut <= 1")
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

    low = Band(z, "zoom", 0.0, SPLIT_KHZ, t0, t1, transient_cut)
    high = Band(z, "wide", SPLIT_KHZ, TOP_KHZ + 1, t0, t1, transient_cut)
    # What is drawn: the wide pass on one colour scale over 0-150 kHz, and the mask
    # from the zoom pass below ZOOM_TOP_KHZ and the wide pass above it.
    raw_view = Band(z, "wide", 0.0, TOP_KHZ + 1, t0, t1, transient_cut)
    mask_top = Band(z, "wide", ZOOM_TOP_KHZ, TOP_KHZ + 1, t0, t1, transient_cut)
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
        {"mask": "TokEye zoom pass, filtered", "threshold": TOKEYE_THRESHOLD},
    )
    n_read, n_kept = roster.gated(n_sig.rows[0], gate) if n_sig.rows else (None, None)

    # Heights in inches, top to bottom: (a) raw, (b) processed, (c) D-alpha with the
    # ELM and confinement labels, (d) NBI power, (e) beta_N, (f) the label rows.
    layout = {
        "raw": 1.35, "g0": 0.07, "pr": 1.35, "g1": 0.08, "da_pr": 0.4,
        "g2": 0.06, "nbi": 0.4, "g3": 0.06, "bn": 0.4, "g4": 0.1,
    }  # fmt: skip
    names = [*layout, *[f"track{i}" for i in range(len(display_tracks))]]
    heights = [
        *layout.values(),
        *[0.19 for t in display_tracks],
    ]
    with style():
        fig = Figure(figsize=(PAGE_IN, HEIGHT_IN))
        gs = fig.add_gridspec(
            len(heights),
            1,
            height_ratios=heights,
            hspace=0.0,
            left=0.14,
            right=RIGHT,
            top=1 - TOP_IN / HEIGHT_IN,
            bottom=BOTTOM_IN / HEIGHT_IN,
        )
        ax = {n: fig.add_subplot(gs[i]) for i, n in enumerate(names)}
        for n in ("g0", "g1", "g2", "g3", "g4"):
            ax[n].set_visible(False)
        track_axes = [ax[f"track{i}"] for i in range(len(display_tracks))]
        for n in (
            "raw",
            "nbi",
            "bn",
            "pr",
            "da_pr",
        ):
            ax[n].set_xlim(t0, t1)
            style_axes(ax[n])
        for a in track_axes:
            a.set_xlim(t0, t1)
            style_axes(a)

        # ---- raw
        transient_columns = draw_frequency_panels(
            ax, raw_view, low, mask_top, transient
        )
        transient_shown = any(transient_columns.values())
        if nbi_sig.rows:
            trace(ax["nbi"], nbi_sig.rows[0], colour="#555555")
        ax["nbi"].set_ylabel(
            "NBI\n(MW)", rotation=0, ha="right", va="center", labelpad=3
        )
        ax["nbi"].set_ylim(bottom=0)
        beta = equilibrium.signal(candidate.shot, "betan", paths)
        beta_ms = np.ravel(beta.x) * 1000.0
        beta_y = np.ravel(beta.y)
        beta_in = (beta_ms >= t0) & (beta_ms <= t1)
        ax["bn"].plot(beta_ms[beta_in], beta_y[beta_in], color="#555555", lw=0.8)
        ax["bn"].set_ylabel(
            r"$\beta_N$", rotation=0, ha="right", va="center", labelpad=3
        )
        ax["bn"].set_ylim(bottom=0)

        # ---- processed: binary coherent mask, with measured n below 30 kHz
        keys = []
        if n_read is not None:
            names = n_read.meta["modes"]["n"]
            keys = [f"n={names[i]}" for i in draw_n_view(ax["pr"], n_read)]
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
        da = ax["da_pr"]
        if da_sig.rows:
            trace(da, da_sig.rows[0])
            top = float(np.max(da_sig.rows[0].values[1]))
            da.set_ylim(0, top * DALPHA_HEADROOM)
            # Red boxes span the lower part of the panel; the regime names above
            # them stay clear of the boxes. Drawn in axes height so the scale is
            # D-alpha's own.
            box_transform = da.get_xaxis_transform()
            peaks = elm_peaks(da_sig.rows[0], elm_spans)
            for a, b in elm_spans:
                box = Rectangle(
                    (a, 0),
                    b - a,
                    ELM_BOX_TOP,
                    fill=False,
                    ec=ELM_BOX_COLOUR,
                    lw=0.8,
                    transform=box_transform,
                )
                da.add_patch(box)
                elm_boxes.append(box)
            # name the boxes once, over the first one's top-left corner (owner,
            # 2026-10-14: say what the coral box is)
            shown_elm = [a for a, b in elm_spans if a < t1 and b > t0]
            if shown_elm:
                da.annotate(
                    ELM_BOX_TEXT,
                    xy=(max(min(shown_elm), t0), ELM_BOX_TOP),
                    xycoords=box_transform,
                    xytext=(2, 1),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=FONT,
                    color=ELM_BOX_COLOUR,
                )
            for a, b in uncertain_elm:
                box = Rectangle(
                    (a, 0),
                    b - a,
                    ELM_BOX_TOP,
                    fc="#eeeeee",
                    ec="#aaaaaa",
                    lw=0.5,
                    alpha=0.55,
                    zorder=0.5,
                    transform=box_transform,
                )
                da.add_patch(box)
                elm_boxes.append(box)
                # explicit hatch lines, as the label rows' (`hatch_lines`)
                hatch_lines(da, a, b, 0, ELM_BOX_TOP, "#aaaaaa", box_transform, 0.6)
            visible = [
                (max(a, t0), min(b, t1)) for a, b in elm_spans if a < t1 and b > t0
            ]
            if not visible:
                visible = [
                    (max(a, t0), min(b, t1))
                    for a, b in uncertain_elm
                    if a < t1 and b > t0
                ]
        da.set_yticks([])
        da.set_ylabel(
            "D-alpha\n(a.u.)", rotation=0, ha="right", va="center", labelpad=3
        )
        regime_names = CLASS_LEGEND  # the names (f) writes on its runs
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
        for category, regions in regime_regions.items():
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
            pad_px = 4 * fig.dpi / 72
            if text.get_window_extent().width + pad_px > region_width:
                text.set_text(label.replace(" (inferred)", "\ninferred"))
            regime_texts.append((text, a, b))

        # ---- label tracks
        titles = {
            mode_tags.AE: "AE", mode_tags.NTM: "TM", mode_tags.SAWTOOTH: "sawtooth",
            elm_key: "ELMs", "confinement": by_key["confinement"].spec.title,
        }  # fmt: skip
        drawn_runs = {}
        run_texts = []
        for a, track in zip(track_axes, display_tracks, strict=True):
            key = track.spec.key
            drawn_runs[key] = track_bars(
                a, track, EVENT_COLOURS.get(key, "#888888"),
                CLASS_COLOURS if key == "confinement" else None,
            )  # fmt: skip
            if key == "confinement":
                run_texts = name_runs(a, drawn_runs[key], t0, t1)
            a.set_ylabel(titles[key], rotation=0, ha="right", va="center", labelpad=3)
        track_axes[-1].tick_params(labelbottom=True, bottom=True)
        track_axes[-1].spines["bottom"].set_visible(True)
        track_axes[-1].set_xlabel("time (ms)", labelpad=1)

        # panel letters, in the left margin at each panel's top; the five label rows
        # are one panel
        letters = {
            "raw": "(a)", "pr": "(b)", "da_pr": "(c)", "nbi": "(d)",
            "bn": "(e)", "track0": "(f)",
        }  # fmt: skip
        for name, letter in letters.items():
            pos = ax[name].get_position()
            fig.text(0.012, pos.y1, letter, fontsize=FONT, fontweight="bold",
                     va="top", ha="left")  # fmt: skip

        colours, display_keys = display_states(display_tracks, t0, t1)
        # Panel (b)'s events are named in place, once everything else is drawn.
        panel_labels = label_events(fig, ax["pr"], (t0, t1), transient_shown)
        fig.draw_without_rendering()
        layout_record = {
            "frequency_axis": {
                "scale": "linear",
                "band_khz": list(ax["pr"].get_ylim()),
                "scale_breaks_khz": [],
                "ticks_khz": ax["pr"].get_yticks().tolist(),
                "raw_spectrogram": "CO2 R0 x V1 cross-power, wide pass, one colour scale, 0-150 kHz",
                "mask_zoom_pass_khz": [0.0, ZOOM_TOP_KHZ],
                "mask_wide_pass_khz": [ZOOM_TOP_KHZ, TOP_KHZ],
                "transient_pass_khz": {TRANSIENT_PASS: [0.0, TOP_KHZ]}
                if transient
                else {},
                "ae_ntm_split_khz": SPLIT_KHZ,
                "n_view_top_khz": N_VIEW_KHZ,
            },
            "n_view_height_in": ax["pr"].get_position().height
            * HEIGHT_IN
            * N_VIEW_KHZ
            / TOP_KHZ,
            "ece_candidate_key_placement": None,
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
                for key, text in []
            ],
            "panel_b_labels": panel_labels["placed"],
            "panel_b_labels_skipped": panel_labels["skipped"],
            "panel_b_label_rule": panel_labels["rule"],
            "panel_b_label_regions": panel_labels["regions"],
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
            "confinement_run_texts": run_texts,
            "label_row_runs": {key: len(runs) for key, runs in drawn_runs.items()},
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
        "beta_n": {
            "store": beta.attrs.get("store"),
            "resolver": beta.attrs.get("resolver"),
            "samples_drawn": int(beta_in.sum()),
        },
        "tokeye_threshold": TOKEYE_THRESHOLD,
        # the overlay is off by default (owner, 2026-10-14: too distracting): the
        # record says so, with no cut or pass where nothing is drawn
        "tokeye_transient_overlay": transient,
        "tokeye_transient_drawn": transient_shown,
        "tokeye_transient_threshold": transient_cut if transient else None,
        "tokeye_transient_pass": TRANSIENT_PASS if transient else None,
        "tokeye_transient_print_columns": transient_columns,
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
        "sawtooth_strip_shown": False,
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
            "component's measured n=1/2 support (all time, TM band) where the "
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
        "sawtooth_crashes": {
            **crash_record,
            # The ECE strip is no longer drawn; the candidates stay in the record.
            "drawn_times_ms": [],
            "candidate_times_ms": crashes.tolist(),
        },
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
        "regimes_shown": sorted(lf.REGIMES[c] for c in shown),
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
        gid=DRAWN_GID + "n",
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
    # A table's rows (`TableTrack`) take their threshold and held-out score from
    # the table's own metadata: a detector's table names them (`table_kind`), a
    # table of labels has neither.
    table = getattr(track, "table", None)
    performance = None
    if (
        track.spec.key == mode_tags.NTM
        and track.source.tier == lf.GENERATED
        and (table is None or "evaluation" in meta)
    ):
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
            meta.get("threshold")
            if table is not None
            else {
                mode_tags.AE: ae_threshold,
                mode_tags.NTM: meta.get("threshold", figure_sources.NTM_THRESHOLD),
                mode_tags.SAWTOOTH: figure_sources.SAWTOOTH_THRESHOLD,
            }.get(track.spec.key)
            if track.source.tier == lf.GENERATED
            else None
        ),
        "source_table": table,
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
        **(
            {}
            if (table or {}).get("no_q1_rows_ms") is None
            else {
                "no_q1_intervals_ms": figure_sources.clip_spans(
                    table["no_q1_rows_ms"], window or (-math.inf, math.inf)
                )
            }
        ),
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


def display_cut(text: str) -> float:
    """A `--transient-cut`: a probability above 0 and at most 1 (0 would paint
    every pixel the mask leaves unlit)."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"{text}: want 0 < cut <= 1")
    return value


def main(argv=None) -> int:
    import torch

    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shot", type=int, default=199597)
    parser.add_argument(
        "--tmin", "--start", type=float, help="ms; supply both bounds or neither"
    )
    parser.add_argument("--tmax", "--end", type=float)
    parser.add_argument("--out", type=Path, help="folder; default round4/fig1d")
    parser.add_argument(
        "--cache", type=Path, help="TokEye cache; default round4/fig1d/cache"
    )
    parser.add_argument("--device", default="cpu", help="TokEye's device")
    parser.add_argument("--cache-only", action="store_true", help="TokEye, no figure")
    parser.add_argument("--png-dpi", type=int, default=DPI_PNG, help="the PNG's dpi")
    parser.add_argument("--record", type=Path, help="also write the record here")
    parser.add_argument(
        "--outside-cohort",
        action="store_true",
        help="draw a shot that is in no cohort split (a detector-only figure)",
    )
    parser.add_argument("--year", type=int, default=2024, help="its run year")
    parser.add_argument(
        "--outside-rows",
        type=Path,
        help="directory of <event>.csv/.meta.json single-shot frame-model rows "
        "for TM and ELMs, for a shot outside the cohort",
    )
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
    parser.add_argument(
        "--rows",
        action="append",
        metavar="EVENT=CSV",
        help="an event's rows from this label table, before its review source "
        f"table and the tiers (repeatable; EVENT one of {', '.join(ROW_EVENTS)})",
    )
    parser.add_argument(
        "--transient",
        action="store_true",
        default=SHOW_TRANSIENT,
        help="draw the transient overlay (red) in panel (b); off by default, the "
        "owner turned it off on 2026-10-14 as too distracting",
    )
    parser.add_argument(
        "--transient-cut",
        type=display_cut,
        default=TRANSIENT_THRESHOLD,
        help="the transient overlay's display cut, 0 < cut <= 1, used with "
        f"--transient (the mask stays at {TOKEYE_THRESHOLD:g})",
    )
    args = parser.parse_args(argv)
    supplied = parse_rows(args.rows)
    if (args.tmin is None) != (args.tmax is None):
        parser.error("supply both --tmin/--start and --tmax/--end, or neither")
    if args.tmin is not None and not (
        math.isfinite(args.tmin) and math.isfinite(args.tmax) and args.tmin < args.tmax
    ):
        parser.error("time bounds must be finite and start must precede end")
    annotations = json.loads(args.annotations.read_text()).get(str(args.shot), {})
    clean_head = not subprocess.check_output(["git", "status", "--porcelain"]).strip()

    paths = Paths.from_env()
    out = args.out or paths.root / "round4" / "fig1d"
    cache = args.cache or paths.root / "round4" / "fig1d" / "cache"
    found = [c for c in lf.candidates(paths) if c.shot == args.shot]
    cohort = read_cohort(cohort_path(paths))
    if found:
        (candidate,) = found
        split = str(cohort.loc[cohort["shot"] == args.shot, "split"].iloc[0])
    elif args.outside_cohort and not (cohort["shot"] == args.shot).any():
        # A shot in no split at all: no detector trained or was scored on it.
        candidate = lf.Candidate(args.shot, args.year, OUTSIDE_WINDOW)
        split = OUTSIDE_COHORT
    else:
        raise SystemExit(f"{args.shot}: not a non-blind cohort shot")
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
        args.outside_rows,
        supplied,
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
        args.transient_cut,
        args.transient,
    )
    records = drawn.pop("tracks")
    detector_training = {}
    for key in (mode_tags.AE, mode_tags.NTM):
        track = records.get(key) or {}
        if not figure_sources.is_detector(track):
            continue  # labels, not a detector of ours: nothing it trained on
        if key == mode_tags.AE and track["what"].startswith("ae-ours"):
            training_file = Path(
                "src/labeler/models/d3d_ae_activity_seldnet/training_shots.txt"
            )
            trained_shots = [int(s) for s in training_file.read_text().split()]
            weights = json.loads(Path(track["metadata"]).read_text())["checkpoint"]
        elif key == mode_tags.AE:
            # The frame detector's own split: the shots it trained on. A
            # detector's table names its checkpoint in its metadata.
            from labeler.ae.xpower.train import read_split

            weights = Path(track["path"])
            if track.get("source_table"):
                meta = json.loads(Path(track["metadata"]).read_text())
                weights = Path(meta.get("model") or meta["checkpoint"])
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
        "outside_cohort": (
            None
            if split != OUTSIDE_COHORT
            else {
                "year": args.year,
                "rows_dir": None
                if args.outside_rows is None
                else str(args.outside_rows),
            }
        ),
        "publication_suitability": annotations.get("publication_suitability", {}),
        "gold_shot": is_gold(paths.label_tables, args.shot),
        "rows_supplied": {key: str(path) for key, path in supplied.items()},
        "source_tables": {
            key: {**t["source_table"], "sha256": t["sha256"]}
            for key, t in records.items()
            if t is not None and t.get("source_table")
        },
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
            "tokeye": TOKEYE_THRESHOLD,
            "tokeye_transient_display": args.transient_cut if args.transient else None,
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
            "probe": f"{SOURCE_GROUP} rows {SOURCE_ROW} and {PARTNER_ROW} ({SOURCE_TITLE})",
            "threshold": TOKEYE_THRESHOLD,
            "transient_overlay": args.transient,
            "transient_display_threshold": args.transient_cut
            if args.transient
            else None,
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
            "TM dominant measured n in {1,2} AND pixel n in {1,2} <=30 kHz; "
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
            "raw_normalisation": "3rd/99.8th percentiles of the whole 0-150 kHz window, "
            "one colour scale",
            "raw_colormap": RAW_CMAP,
            "transient_display_rule": (
                f"the {TRANSIENT_PASS} pass's transient over "
                f"the whole 0-{TOP_KHZ:g} kHz band, >= its display cut, small "
                "objects removed as the mask's, where neither the wide pass's mask "
                f"nor the drawn zoom-pass mask (below {ZOOM_TOP_KHZ:g} kHz) is "
                f"lit; max pooled to {DPI_PNG}-dpi print pixels, drawn with no "
                "interpolation; display only"
                if args.transient
                else "not drawn: the owner turned the overlay off on 2026-10-14 "
                "(too distracting); the transient channel still keeps its pixels "
                "out of the coherent mask"
            ),
            "label_row_display_rule": "adjacent rows of one category and crowd flag "
            "drawn as one run; confinement class names written on runs they fit; "
            "uncertain time white, hatched in the event's colour with explicit "
            f"line segments ({UNCERTAIN_HATCH_STEP_PT:g} pt apart, "
            f"{UNCERTAIN_HATCH_LW:g} pt), not a PDF hatch pattern; no key",
            "unkeyed_n_colour": N_OTHER,
            "bands_khz": {
                k: [lo, None if math.isinf(hi) else hi]
                for k, (lo, hi) in mode_tags.BANDS.items()
            },
            "split_khz": SPLIT_KHZ,
            "frequency_axis": "linear 0-150 kHz, no scale breaks",
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
