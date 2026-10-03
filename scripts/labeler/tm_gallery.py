#!/usr/bin/env python
"""A gallery of the tearing-mode interval labels: spectrogram and n = 1 / n = 2 RMS.

Each shot is three rows on one time axis: a strip with its intervals (one colour per
toroidal number, a triangle at each onset, a hatch where the interval ended by locking),
the 0-50 kHz MHR spectrogram (corpus `mhr` row 0, in dB above each frequency's
own floor over the plasma), and the n = 1 and n = 2 RMS (log gauss) with the onset
threshold and, per interval, the release level it was cut at. MHR often covers only
part of a pulse; its uncovered times are grey. `--diagnostic mirnov` gives a second
gallery with the longer MPI66M322D record (corpus `mirnov` row 15).

    PYTHONPATH=$PWD/src pixi run --frozen --no-install -e labelmaker python \\
        scripts/labeler/tm_gallery.py --n 12 --seed 3 --out <stem>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from labeler.events.interval_tables import parse_attrs
from labeler.events.panels._shared import above_floor_db, finite, plasma_columns, stft
from labeler.events.verify import NoDataError, corpus_signal
from labeler.tearing import rule

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
SIGNALS = LABELER / "round4/tm/signals"
LABELS = (
    REPO / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"
)
FULL = LABELER / "round4/tm/labels/tm_intervals_full_cohort.csv"
COHORT = REPO / "data/events/catalog/cohort.csv"
#: Okabe-Ito: n = 1 vermillion, n = 2 blue; the traces use the same colours.
COLOUR = {1: "#D55E00", 2: "#0072B2"}
PROBE_ROW = 15
Z_DB = (-3.0, 42.0)
FONT = 7.5


def intervals_of(frame: pd.DataFrame):
    """`(spans, onsets)` of one shot's rows: `[(t0, t1, n, locked)]`, `[(t, n)]`."""
    spans, onsets = [], []
    for row in frame[frame.category == 1].itertuples(index=False):
        attrs = parse_attrs(row.attrs)
        if row.t_end > row.t_start:
            spans.append(
                (row.t_start, row.t_end, int(attrs["n"]), bool(attrs.get("locked")))
            )
        else:
            onsets.append((row.t_start, int(attrs["n"])))
    return spans, onsets


def spectrogram(shot: int, window, diagnostic="mhr"):
    """`(t_s, f_khz, db)` of the probe's 0-50 kHz power over the window, or None."""
    try:
        array = corpus_signal(
            shot,
            diagnostic,
            channels=[PROBE_ROW if diagnostic == "mirnov" else 0],
            t_range=window,
        )
    except (NoDataError, KeyError, OSError):
        return None
    t_ms, f_hz, spec = stft(
        array.x, finite(array.y[0]), rate_hz=100_000, nperseg=1024, hop=256
    )
    power = np.abs(spec) ** 2
    columns = plasma_columns(t_ms, power.sum(axis=0), window)
    return t_ms / 1000.0, f_hz / 1000.0, above_floor_db(power, 0.2, columns=columns)


def draw_shot(fig, grid, shot, window, signals, frame, thresholds, diagnostic):
    """The three rows of one shot in its cell of the figure."""
    sub = grid.subgridspec(3, 1, height_ratios=[0.35, 2.2, 1.6], hspace=0.08)
    strip, spec_ax, rms_ax = (fig.add_subplot(sub[i]) for i in range(3))
    w0, w1 = window[0] / 1000.0, window[1] / 1000.0
    spans, onsets = intervals_of(frame)
    for t0, t1, n, locked in spans:
        strip.axvspan(
            t0 / 1000,
            t1 / 1000,
            ymin=0.12 + 0.4 * (n == 2),
            ymax=0.52 + 0.4 * (n == 2),
            color=COLOUR[n],
            lw=0,
            hatch="////" if locked else None,
        )
        rms_ax.axvspan(t0 / 1000, t1 / 1000, color=COLOUR[n], alpha=0.12, lw=0)
    for t, n in onsets:
        strip.plot(
            [t / 1000],
            [0.5 + 0.4 * (n == 2) - 0.2],
            marker="v",
            ms=3.5,
            color="k",
            ls="",
        )
    strip.set_xlim(w0, w1)
    strip.set_ylim(0, 1)
    strip.set_yticks([])
    strip.set_xticks([])
    for side in ("top", "right", "left"):
        strip.spines[side].set_visible(False)
    strip.set_title(f"{shot}", fontsize=FONT, loc="left", pad=2)
    got = spectrogram(shot, window, diagnostic)
    spec_ax.set_facecolor("0.85")
    if got is None:
        spec_ax.text(
            0.5,
            0.5,
            f"no {diagnostic.upper()} record",
            ha="center",
            va="center",
            transform=spec_ax.transAxes,
            fontsize=FONT,
        )
        spec_ax.set_xlim(w0, w1)
    else:
        t, f, db = got
        spec_ax.pcolormesh(
            t,
            f,
            db,
            vmin=Z_DB[0],
            vmax=Z_DB[1],
            cmap="viridis",
            shading="auto",
            rasterized=True,
        )
        spec_ax.set_xlim(w0, w1)
    spec_ax.set_ylim(0, 50)
    for t0, t1, n, _ in spans:
        spec_ax.axvspan(t0 / 1000, t1 / 1000, color=COLOUR[n], alpha=0.12, lw=0)
        for edge in (t0, t1):
            spec_ax.axvline(edge / 1000, color=COLOUR[n], lw=0.6, ls="--")
    spec_ax.set_ylabel("kHz", fontsize=FONT, labelpad=1)
    spec_ax.set_xticklabels([])
    t_ms, n1, n2 = signals
    for n, y in ((1, n1), (2, n2)):
        rms_ax.plot(t_ms / 1000, np.clip(y, 0.05, None), color=COLOUR[n], lw=0.6)
    for rule_ in rule.RULES:
        rms_ax.axhline(rule_.onset_g, color=COLOUR[rule_.n], lw=0.6, ls=":")
    for item in thresholds:
        rms_ax.plot(
            [item.start_ms / 1000, item.end_ms / 1000],
            [item.release_g] * 2,
            color=COLOUR[item.n],
            lw=1.0,
            ls="--",
        )
    rms_ax.set_yscale("log")
    rms_ax.set_ylim(0.05, 150)
    rms_ax.set_xlim(w0, w1)
    rms_ax.set_ylabel("RMS (G)", fontsize=FONT, labelpad=1)
    rms_ax.set_xlabel("time (s)", fontsize=FONT, labelpad=1)
    for ax in (spec_ax, rms_ax):
        ax.tick_params(labelsize=FONT, length=2, pad=1)
    return None if got is None else [float(got[0][0]), float(got[0][-1])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument(
        "--full",
        type=Path,
        default=FULL,
        help="the per-interval table with the release levels",
    )
    ap.add_argument("--signals-dir", type=Path, default=SIGNALS)
    ap.add_argument("--out", type=Path, required=True, help="path without suffix")
    ap.add_argument(
        "--with-mode", action="store_true", help="draw only shots that have an interval"
    )
    ap.add_argument(
        "--shots", type=int, nargs="+", help="these shots instead of a draw"
    )
    ap.add_argument("--columns", type=int, default=3)
    ap.add_argument("--diagnostic", choices=("mhr", "mirnov"), default="mhr")
    args = ap.parse_args(argv)

    cohort = pd.read_csv(COHORT).set_index("shot")
    table = pd.read_csv(args.labels)
    full = pd.read_csv(args.full)
    labelled = sorted(
        set(table.shot) & {int(p.stem) for p in args.signals_dir.glob("*.npz")}
    )
    # The rule is frozen on the training and validation shots; a drawn shot is never a
    # test shot.
    eligible = (
        [s for s in labelled if cohort.loc[s, "split"] != "test"]
        if set(labelled) <= set(cohort.index)
        else labelled
    )
    if args.with_mode:
        has = set(table[(table.category == 1) & (table.t_end > table.t_start)].shot)
        eligible = [s for s in eligible if s in has]
    rng = np.random.default_rng(args.seed)
    shots = args.shots or sorted(
        rng.choice(eligible, size=min(args.n, len(eligible)), replace=False).tolist()
    )
    if any(s not in eligible for s in shots):
        raise SystemExit("gallery shots must be eligible development shots")
    columns = args.columns
    rows = int(np.ceil(len(shots) / columns))
    plt.rcParams.update({"font.size": FONT, "axes.linewidth": 0.5})
    fig = plt.figure(figsize=(7.3, 1.75 * rows + 0.25))
    grid = fig.add_gridspec(
        rows,
        columns,
        hspace=0.45,
        wspace=0.28,
        left=0.06,
        right=0.995,
        top=0.975,
        bottom=0.09 if rows < 4 else 0.065,
    )
    coverage = {}
    for k, shot in enumerate(shots):
        row = cohort.loc[shot]
        window = (float(row.window_start_ms), float(row.window_end_ms))
        with np.load(args.signals_dir / f"{shot}.npz") as npz:
            signals = (npz["t_ms"], npz["n1rms"], npz["n2rms"])
        frame = table[table.shot == shot]
        items = [
            rule.Interval(
                n=int(r.n),
                start_ms=r.t_start,
                end_ms=r.t_end,
                peak_g=r.peak_g,
                peak_ms=r.peak_ms,
                release_g=r.release_g,
            )
            for r in full[full.shot == shot].itertuples(index=False)
        ]
        coverage[shot] = draw_shot(
            fig,
            grid[k // columns, k % columns],
            shot,
            window,
            signals,
            frame,
            items,
            args.diagnostic,
        )
    handles = [
        plt.Line2D([], [], color=COLOUR[1], lw=1.5, label="n = 1"),
        plt.Line2D([], [], color=COLOUR[2], lw=1.5, label="n = 2"),
        plt.Line2D([], [], color="k", marker="v", ls="", ms=3.5, label="onset"),
        plt.Line2D([], [], color="0.3", ls=":", lw=0.8, label="onset level"),
        plt.Line2D([], [], color="0.3", ls="--", lw=1.0, label="release level"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=FONT,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=150)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "made_by": "scripts/labeler/tm_gallery.py",
                "diagnostic": args.diagnostic,
                "seed": args.seed,
                "shots": shots,
                "split": "development (train and validation); blind test excluded",
                "spectrogram_coverage_s": coverage,
                "labels": str(args.labels),
                "frequency_khz": [0, 50],
                "dpi": 150,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{args.out}: shots {shots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
