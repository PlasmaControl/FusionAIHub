r"""fig_confinement_coverage: every curated-label shot, its L / H / QH / WPQH segments,
whether BES and the beam gating are available, and the regime time by shot-number band.

    PYTHONPATH=src pixi run --frozen --no-install -e labelmaker \
        python scripts/labeler/paper/fig_confinement_coverage.py \
        [--out PATH.pdf] [--png PATH.png] [--table PATH.csv]

One row per shot of the curated confinement intervals outside the cohort's blind test
split, in shot-number order and in two panels side by side. Each row has two strips: the
labelled regimes (L, H, QH, WPQH) above, and below them the time when the 150L beam is
at or above 700 kW and the 150R beam at or below 200 kW, the gate the BES classifier of
Gill et al. (2024) is trained under (blank where the shot has no beam record). The two
squares left of a row say whether the corpus holds the shot's BES (500 kHz) and whether
the native 1 MHz BES was fetched. The bars underneath add the labelled regime time of
each band of 5 000 shot numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
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
CORPUS = zerod.CORPUS
PAGE_IN = 6.75  # ICML \textwidth
FONT_PT = 7
# Okabe-Ito, one colour per regime; greys for the two availability strips.
COLOURS = {"L": "#0072B2", "H": "#E69F00", "QH": "#009E73", "WP": "#CC79A7"}
NAMES = {"L": "L", "H": "H", "QH": "QH", "WP": "WPQH"}
INK = "#333333"
BES_CORPUS = "#111111"
BES_NATIVE = "#9a9a9a"
T_MAX_S = 8.4
BAND = 5000
BEAM_GRID_MS = 5.0
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
    grid = np.arange(0.0, T_MAX_S * 1000.0, BEAM_GRID_MS)
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
            }
        )
    return rows


def draw_panel(ax, rows: list[dict], n_rows: int) -> None:
    """``rows`` top to bottom on an axis that is ``n_rows`` rows tall."""
    ax.set_xlim(-0.75, T_MAX_S)
    ax.set_ylim(n_rows - 0.5, -0.5)
    for i, row in enumerate(rows):
        for regime, start, length in row["segments"]:
            ax.broken_barh(
                [(start, length)],
                (i - 0.5, 0.64),
                facecolors=COLOURS[regime],
                linewidth=0,
            )
        if row["beam"]:
            ax.broken_barh(row["beam"], (i + 0.2, 0.3), facecolors=INK, linewidth=0)
        ax.broken_barh(
            [(-0.72, 0.3)],
            (i - 0.5, 1.0),
            facecolors=BES_CORPUS if row["bes_corpus"] else "none",
            linewidth=0,
        )
        ax.broken_barh(
            [(-0.38, 0.3)],
            (i - 0.5, 1.0),
            facecolors=BES_NATIVE if row["bes_native"] else "none",
            linewidth=0,
        )
    ticks = list(range(0, len(rows), 25))
    ax.set_yticks(ticks, [str(rows[i]["shot"]) for i in ticks])
    ax.set_xticks(np.arange(0, 9, 2))
    ax.set_xlabel("time in shot (s)")
    ax.tick_params(length=2, pad=1.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def band_table(rows: list[dict]) -> list[dict]:
    """Labelled seconds per regime, and shots, in each band of ``BAND`` shot numbers."""
    bands: dict[int, dict] = {}
    for row in rows:
        b = row["shot"] // BAND * BAND
        entry = bands.setdefault(
            b, {"band": b, "shots": 0, **{c: 0.0 for c in COLOURS}}
        )
        entry["shots"] += 1
        for regime, _, length in row["segments"]:
            if regime in COLOURS:
                entry[regime] += length
    return [bands[b] for b in sorted(bands)]


def draw_bands(ax, bands: list[dict]) -> None:
    x = np.arange(len(bands))
    bottom = np.zeros(len(bands))
    for regime, colour in COLOURS.items():
        h = np.array([b[regime] for b in bands])
        ax.bar(x, h, 0.78, bottom=bottom, color=colour, linewidth=0)
        bottom += h
    for xi, b, top in zip(x, bands, bottom, strict=True):
        ax.text(
            xi,
            top + 4,
            str(b["shots"]),
            ha="center",
            va="bottom",
            fontsize=FONT_PT,
            color=INK,
        )
    ax.set_xticks(x, [f"{b['band'] // 1000}" for b in bands])
    ax.set_xlim(-0.7, len(bands) - 0.3)
    ax.set_ylim(0, bottom.max() * 1.16)
    ax.set_xlabel("shot number / 1000 (band of 5 000; count of shots above each bar)")
    ax.set_ylabel("labelled time (s)")
    ax.tick_params(length=2, pad=1.5)


def make_figure(rows: list[dict]) -> Figure:
    half = (len(rows) + 1) // 2
    fig = Figure(figsize=(PAGE_IN, 9.0), dpi=150)
    left = fig.add_axes((0.065, 0.19, 0.43, 0.77))
    right = fig.add_axes((0.55, 0.19, 0.43, 0.77))
    draw_panel(left, rows[:half], half)
    draw_panel(right, rows[half:], half)  # same row pitch as the left panel
    bands = fig.add_axes((0.065, 0.045, 0.915, 0.09))
    draw_bands(bands, band_table(rows))
    handles = [Patch(color=c, label=NAMES[r]) for r, c in COLOURS.items()] + [
        Patch(color=INK, label="beam gate"),
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
    bands = band_table(rows)
    summary = {
        "shots": len(rows),
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
