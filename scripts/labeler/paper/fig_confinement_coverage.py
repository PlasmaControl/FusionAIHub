r"""fig_confinement_coverage: every curated-label shot, its L / H / QH / WPQH segments,
whether BES and the beam gating are available, and the regime time by year.

    PYTHONPATH=src pixi run --frozen --no-install -e labelmaker \
        python scripts/labeler/paper/fig_confinement_coverage.py \
        [--out PATH.pdf] [--png PATH.png] [--table PATH.csv]

One row per shot of the curated confinement intervals outside the cohort's blind test
split, in shot-number order and in three panels side by side. Each row has two strips:
the labelled regimes (L, H, QH, WPQH) above, and below them, in light grey, the time
when the 150L beam is at or above 700 kW and the 150R beam at or below 200 kW, the gate
the BES classifier of Gill et al. (2024) is trained under (blank where the shot has no
beam record). Two squares left of a row say which BES record the shot has: the left one
is black where the corpus holds the BES at 500 kHz, the right one mid grey where the
native 1 MHz BES was fetched (every corpus shot is also fetched natively, so a corpus
row shows both squares and a fetched-only row the grey one alone). The bars
underneath add the labelled regime time of each year (left; a shot's year is the year
its EFIT01 reconstruction was inserted into MDSplus, from ``round4/conf/dates.csv``, see
``confinement_shot_dates_fetch.py``; shot-number bands of 5 000 when that file is
missing) and of each 0.2 s of shot time (right, aligned with the time axes above). The
time axes end at the last labelled time of the data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.confinement import bes_protocol as bp
from labeler.confinement import bes_windows as bw
from labeler.confinement import zerod

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
WORK = LABELER / "round4/conf"
DEFAULT_OUT = WORK / "fig_confinement_coverage.pdf"
DATES = WORK / "dates.csv"
CORPUS = zerod.CORPUS
PAGE_IN = 6.75  # ICML \textwidth
FONT_PT = 7
# Okabe-Ito, one colour per regime; greys for the availability marks: the beam gate is a
# light strip along the time axis, the BES marks are dark squares left of the row.
COLOURS = {"L": "#0072B2", "H": "#E69F00", "QH": "#009E73", "WP": "#CC79A7"}
NAMES = {"L": "L", "H": "H", "QH": "QH", "WP": "WPQH"}
INK = "#333333"
GATE = "#cfcfcf"
BES_CORPUS = "#111111"
BES_NATIVE = "#6b6b6b"
PANELS = 3
FIG_H_IN = 9.0
BAND = 5000
BEAM_END_S = 10.0
BEAM_GRID_MS = 5.0
TIME_BIN_S = 0.2
STYLE = {
    "font.size": FONT_PT,
    "axes.labelsize": FONT_PT,
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "legend.fontsize": FONT_PT,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def corpus_has_bes(shot: int) -> bool:
    import h5py

    path = CORPUS / f"{shot}_processed.h5"
    if not path.exists():
        return False
    with h5py.File(path, "r") as f:
        return "bes" in f and f["bes/ydata"].shape[-1] > 1


def beam_valid_runs(shot: int) -> list[tuple[float, float]] | None:
    """Runs (start, length) in s where the BES beam gate holds; None with no record."""
    got = zerod.read_beams(shot, LABELER / "raw", WORK / "zerod")
    if got is None:
        return None
    t_ms, rows = got
    grid = np.arange(0.0, BEAM_END_S * 1000.0, BEAM_GRID_MS)
    left = np.interp(grid, t_ms, np.nan_to_num(rows[0]), left=0.0, right=0.0)
    right = np.interp(grid, t_ms, np.nan_to_num(rows[1]), left=0.0, right=0.0)
    ok = (left >= bp.GATE_LEFT_W) & (right <= bp.GATE_RIGHT_W)
    edges = np.flatnonzero(np.diff(np.r_[False, ok, False].astype(np.int8)))
    return [
        (grid[a] / 1000.0, (grid[b - 1] - grid[a] + BEAM_GRID_MS) / 1000.0)
        for a, b in zip(edges[::2], edges[1::2], strict=True)
    ]


def collect() -> list[dict]:
    """Per shot: its regime segments, beam-valid runs and BES flags."""
    iv = bw.curated_intervals()
    year = (
        pd.read_csv(DATES, keep_default_na=False).set_index("shot").year
        if DATES.exists()
        else pd.Series(dtype=float)
    )
    native = {
        int(f.stem) for f in (WORK / "bes1mhz").glob("*.npz") if "tmp" not in f.name
    }
    rows = []
    for shot, g in iv.groupby("shot"):
        shot = int(shot)
        rows.append(
            {
                "shot": shot,
                "segments": [
                    (r.regimes, r.t_start / 1000.0, (r.t_end - r.t_start) / 1000.0)
                    for r in g.itertuples()
                ],
                "beam": beam_valid_runs(shot),
                "bes_corpus": corpus_has_bes(shot),
                "bes_native": shot in native,
                "year": int(year[shot]) if shot in year.index else None,
            }
        )
    return rows


def t_max(rows: list[dict]) -> float:
    """The last labelled time of the data, rounded up to half a second."""
    end = max(start + length for r in rows for _, start, length in r["segments"])
    return math.ceil(end * 2) / 2


def draw_panel(ax, rows: list[dict], n_rows: int, t_end: float) -> None:
    """``rows`` top to bottom on an axis that is ``n_rows`` rows tall."""
    ax.set_xlim(-0.9, t_end)
    ax.set_ylim(n_rows - 0.5, -0.5)
    for i, row in enumerate(rows):
        if row["beam"]:
            ax.broken_barh(row["beam"], (i + 0.16, 0.34), facecolors=GATE, linewidth=0)
        for regime, start, length in row["segments"]:
            ax.broken_barh(
                [(start, length)],
                (i - 0.5, 0.64),
                facecolors=COLOURS[regime],
                linewidth=0,
            )
        ax.broken_barh(
            [(-0.88, 0.36)],
            (i - 0.5, 0.9),
            facecolors=BES_CORPUS if row["bes_corpus"] else "none",
            linewidth=0,
        )
        ax.broken_barh(
            [(-0.48, 0.36)],
            (i - 0.5, 0.9),
            facecolors=BES_NATIVE if row["bes_native"] else "none",
            linewidth=0,
        )
    ticks = list(range(0, len(rows), 20))
    ax.set_yticks(ticks, [str(rows[i]["shot"]) for i in ticks])
    ax.set_xticks(np.arange(0, t_end + 0.01, 1.0))
    ax.set_xlabel("time in shot (s)")
    ax.tick_params(length=2, pad=1.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def group_table(rows: list[dict], by_year: bool) -> list[dict]:
    """Labelled seconds per regime, and shots, in each year (or, without dates, each
    band of ``BAND`` shot numbers)."""
    groups: dict[int, dict] = {}
    for row in rows:
        key = row["year"] if by_year else row["shot"] // BAND * BAND
        entry = groups.setdefault(
            key, {"band": key, "shots": 0, **{c: 0.0 for c in COLOURS}}
        )
        entry["shots"] += 1
        for regime, _, length in row["segments"]:
            if regime in COLOURS:
                entry[regime] += length
    return [groups[k] for k in sorted(groups)]


def draw_groups(ax, groups: list[dict], by_year: bool) -> None:
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for regime, colour in COLOURS.items():
        h = np.array([g[regime] for g in groups])
        ax.bar(x, h, 0.78, bottom=bottom, color=colour, linewidth=0)
        bottom += h
    for xi, g, top in zip(x, groups, bottom, strict=True):
        ax.text(
            xi,
            top + 4,
            str(g["shots"]),
            ha="center",
            va="bottom",
            fontsize=FONT_PT,
            color=INK,
        )
    ax.set_xticks(
        x, [str(g["band"]) if by_year else f"{g['band'] // 1000}" for g in groups]
    )
    ax.set_xlim(-0.7, len(groups) - 0.3)
    ax.set_ylim(0, bottom.max() * 1.16)
    ax.set_xlabel(
        "year of the shot (EFIT01 insertion; shots above)"
        if by_year
        else "shot number / 1000 (bands of 5 000; shots above)"
    )
    ax.set_ylabel("labelled time (s)")
    ax.tick_params(length=2, pad=1.5)


def time_table(rows: list[dict], t_end: float) -> np.ndarray:
    """Labelled shot-seconds per regime in each ``TIME_BIN_S`` bin of shot time,
    ``(len(COLOURS), n_bins)``."""
    edges = np.arange(0.0, t_end + TIME_BIN_S, TIME_BIN_S)
    out = np.zeros((len(COLOURS), len(edges) - 1))
    for row in rows:
        for regime, start, length in row["segments"]:
            lo = np.clip(edges[:-1], start, start + length)
            hi = np.clip(edges[1:], start, start + length)
            out[list(COLOURS).index(regime)] += hi - lo
    return out


def draw_time(ax, rows: list[dict], t_end: float) -> None:
    table = time_table(rows, t_end)
    edges = np.arange(0.0, t_end + TIME_BIN_S, TIME_BIN_S)
    bottom = np.zeros(table.shape[1])
    for (_, colour), h in zip(COLOURS.items(), table, strict=True):
        ax.bar(
            edges[:-1],
            h,
            TIME_BIN_S,
            bottom=bottom,
            align="edge",
            color=colour,
            linewidth=0,
        )
        bottom += h
    ax.set_xlim(-0.9, t_end)
    ax.set_xticks(np.arange(0, t_end + 0.01, 1.0))
    ax.set_xlabel("time in shot (s)")
    ax.set_ylabel("labelled time, all shots (s)")
    ax.tick_params(length=2, pad=1.5)


def make_figure(rows: list[dict]) -> Figure:
    by_year = all(r["year"] is not None for r in rows)
    t_end = t_max(rows)
    per = math.ceil(len(rows) / PANELS)
    fig = Figure(figsize=(PAGE_IN, FIG_H_IN), dpi=150)
    width, gap, x0 = 0.265, 0.05, 0.085
    for i in range(PANELS):
        ax = fig.add_axes((x0 + i * (width + gap), 0.17, width, 0.78))
        draw_panel(ax, rows[i * per : (i + 1) * per], per, t_end)
    groups = fig.add_axes((x0, 0.045, 2 * width - 0.02, 0.085))
    draw_groups(groups, group_table(rows, by_year), by_year)
    marginal = fig.add_axes((x0 + 2 * (width + gap), 0.045, width, 0.085))
    draw_time(marginal, rows, t_end)
    handles = [Patch(color=c, label=NAMES[r]) for r, c in COLOURS.items()] + [
        Patch(color=GATE, label="beam gate"),
        Patch(color=BES_CORPUS, label="BES, 500 kHz corpus"),
        Patch(color=BES_NATIVE, label="BES, 1 MHz fetched"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.997),
        ncol=7,
        frameon=False,
        handlelength=1.1,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--png", type=Path, default=None)
    ap.add_argument("--table", type=Path, default=None)
    args = ap.parse_args(argv)
    rows = collect()
    with matplotlib.rc_context(STYLE):
        fig = make_figure(rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out)
        png = args.png or args.out.with_suffix(".png")
        fig.savefig(png, dpi=150)
    by_year = all(r["year"] is not None for r in rows)
    bands = group_table(rows, by_year)
    summary = {
        "shots": len(rows),
        "grouped_by": "year (EFIT01 insertion time)" if by_year else "shot-number band",
        "time_axis_end_s": t_max(rows),
        "beam_record_missing": [r["shot"] for r in rows if r["beam"] is None],
        "bes_corpus": sum(r["bes_corpus"] for r in rows),
        "bes_native": sum(r["bes_native"] for r in rows),
        "bands": bands,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "bands"}))
    if args.table:
        with open(args.table, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bands[0]))
            w.writeheader()
            w.writerows(bands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
