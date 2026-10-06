#!/usr/bin/env python
"""A gallery of strong rotating n=1/n=2 modes (tearing-mode proxies).

Each shot is three rows on one time axis: a strip with its intervals (one colour per
toroidal number, a triangle at each onset) and, behind them, the uncertain time (one
flat grey with no outlines: uncertain rows that overlap are drawn once, so a darker or
boxed patch never means "more uncertain"; hatched where the uncertain time is a locked
phase: slashes where a step of the radial field confirms the lock of a mode, crosses
where the field steps in flat-top time with no mode seen (`locked_unseeded`), dots
where the mode collapsed or locked but the field did not confirm it),
the 0-50 kHz MHR spectrogram (corpus `mhr` row 2, in dB above each frequency's
own floor over the plasma), and the n = 1 and n = 2 RMS (log gauss) with the onset
threshold and, per interval, the release level it was cut at. Both panels shade each
interval in its colour; where an n = 1 and an n = 2 interval overlap the shadings
blend into a light grey, named in the legend ("n = 1 and 2 overlap"). MHR often
covers only part of a pulse; its uncovered times are grey. `--diagnostic mirnov`
gives a second gallery with the longer MPI66M322D record (corpus `mirnov` row 15).

    PYTHONPATH=$PWD/src pixi run --frozen --no-install -e labelmaker python \\
        scripts/labeler/tm_gallery.py --n 12 --seed 3 --out <stem>
"""

from __future__ import annotations

import argparse
import hashlib
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
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from labeler.config import git_sha
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
MHR_ROW = 2
Z_DB = (-3.0, 42.0)
FONT = 7.5
#: Uncertain rows made by the lock evidence, and how they are hatched: a step of the
#: radial field confirms the lock of a mode (slashes), the field steps in flat-top time
#: with no mode seen (crosses), or the mode collapsed or locked unconfirmed (dots).
LOCK_HATCH = {
    "confirmed_locked_phase": "////",
    "locked_unseeded": "xxxx",
    "post_collapse_lock_unknown": "..",
    "rotation_after_lock_unassessed": "..",
}


def overlap_colour(alpha: float = 0.12):
    """The colour where the n = 1 and n = 2 shadings overlap, over a white panel.

    Each interval is a translucent span in its own colour (`alpha`), so two overlapping
    ones blend into a light grey; the legend names it so it is not read as uncertain.
    """
    rgb = np.ones(3)
    for n in (1, 2):
        rgb = alpha * np.array(matplotlib.colors.to_rgb(COLOUR[n])) + (1 - alpha) * rgb
    return tuple(float(v) for v in rgb)


def has_overlap(frame: pd.DataFrame) -> bool:
    """Whether one shot has an n = 1 and an n = 2 interval that overlap in time."""
    spans = intervals_of(frame)[0]
    return any(
        a[2] != b[2] and a[0] < b[1] and b[0] < a[1]
        for i, a in enumerate(spans)
        for b in spans[i + 1 :]
    )


def intervals_of(frame: pd.DataFrame):
    """Spans `(t0,t1,n,locked,candidate)` and onset points `(t,n)` of one shot."""
    spans, onsets = [], []
    for row in frame[frame.category == 1].itertuples(index=False):
        attrs = parse_attrs(row.attrs)
        if row.t_end > row.t_start:
            spans.append(
                (
                    row.t_start,
                    row.t_end,
                    int(attrs["n"]),
                    bool(attrs.get("locked")),
                    bool(attrs.get("locked_candidate")),
                )
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
            channels=[PROBE_ROW if diagnostic == "mirnov" else MHR_ROW],
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
    sub = grid.subgridspec(3, 1, height_ratios=[0.35, 2.2, 2.6], hspace=0.08)
    strip, spec_ax, rms_ax = (fig.add_subplot(sub[i]) for i in range(3))
    w0, w1 = window[0] / 1000.0, window[1] / 1000.0
    spans, onsets = intervals_of(frame)
    for t0, t1, n, _, _ in spans:
        strip.axvspan(
            t0 / 1000,
            t1 / 1000,
            ymin=0.12 + 0.4 * (n == 2),
            ymax=0.52 + 0.4 * (n == 2),
            facecolor=COLOUR[n],
            edgecolor=COLOUR[n],
            lw=0,
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
    uncertain = frame[(frame.category == 2) & (frame.t_end > frame.t_start)]
    for row in uncertain.itertuples(index=False):
        reason = parse_attrs(row.attrs).get("reason")
        # Flat and outline-free, so overlapping uncertain rows merge into one grey and
        # never stack into a darker or boxed patch; the locked phases are the
        # uncertain rows the lock evidence made, hatched on top of it.
        strip.axvspan(
            row.t_start / 1000,
            row.t_end / 1000,
            facecolor="0.8",
            edgecolor="0.25",
            lw=0,
            hatch=LOCK_HATCH.get(reason),
            zorder=0 if reason not in LOCK_HATCH else 0.5,
        )
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
    for t0, t1, n, _, _ in spans:
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
    rms_ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 6, 12, 100]))
    rms_ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    rms_ax.yaxis.set_minor_formatter(NullFormatter())
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
    ap.add_argument(
        "--lock-example",
        type=Path,
        help="rule-diagnostics JSON: also draw the cohort lock with the largest "
        "radial-field step among the confirmed n = 1 locks (the fixed rule that picks "
        "a locking example)",
    )
    ap.add_argument("--columns", type=int, default=3)
    ap.add_argument(
        "--width", type=float, default=7.3, help="publication width in inches"
    )
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
    lock_rule = None
    if args.lock_example:
        best = json.loads(args.lock_example.read_text())["lock_steps"]["cohort"][
            "largest_step"
        ]
        if best["shot"] not in shots:
            shots = [*shots, int(best["shot"])]
        lock_rule = (
            f"the locking example is shot {best['shot']}, the confirmed cohort n = 1 "
            f"lock with the largest radial-field step ({best['step']:.1f} native "
            f"units: {best['before']:.1f} before, {best['after']:.1f} after)"
        )
    if any(s not in eligible for s in shots):
        raise SystemExit("gallery shots must be eligible development shots")
    columns = args.columns
    rows = int(np.ceil(len(shots) / columns))
    plt.rcParams.update({"font.size": FONT, "axes.linewidth": 0.5})
    # The log-scale separation between 6 and 12 G needs enough vertical space for
    # both tick labels at the specified publication font size.
    column = args.width <= 3.25
    fig = plt.figure(figsize=(args.width, 3.2 * rows + (1.0 if column else 0.4)))
    grid = fig.add_gridspec(
        rows,
        columns,
        hspace=0.25,
        wspace=0.28,
        left=0.15 if args.width <= 3.25 else 0.06,
        right=0.995,
        top=0.975,
        bottom=(0.28 if rows == 1 else 0.18)
        if column
        else (0.09 if rows < 4 else 0.065),
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
        plt.Line2D(
            [], [], color=COLOUR[1], ls=":", lw=0.8, label="n = 1 start level (12 G)"
        ),
        plt.Line2D(
            [], [], color=COLOUR[2], ls=":", lw=0.8, label="n = 2 start level (6 G)"
        ),
        plt.Line2D([], [], color=COLOUR[1], ls="--", lw=1.0, label="n = 1 end level"),
        plt.Line2D([], [], color=COLOUR[2], ls="--", lw=1.0, label="n = 2 end level"),
        Patch(
            facecolor="0.8",
            edgecolor="none",
            label="uncertain",
        ),
    ]
    if any(has_overlap(table[table.shot == shot]) for shot in shots):
        handles.append(
            Patch(
                facecolor=overlap_colour(),
                edgecolor="0.5",
                lw=0.4,
                label="n = 1 and 2 overlap",
            )
        )
    selected_spans = intervals_of(table[table.shot.isin(shots)])[0]
    reasons = {
        parse_attrs(a).get("reason")
        for a in table[table.shot.isin(shots) & (table.category == 2)]["attrs"]
    }
    for kind, hatch, label in (
        (
            {"confirmed_locked_phase"},
            "////",
            "locked (confirmed)",
        ),
        (
            {"locked_unseeded"},
            "xxxx",
            "field step, no mode seen",
        ),
        (
            {"post_collapse_lock_unknown", "rotation_after_lock_unassessed"},
            "..",
            "lock unconfirmed",
        ),
    ):
        if reasons & kind:
            handles.append(
                Patch(facecolor="0.8", edgecolor="0.25", hatch=hatch, label=label)
            )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2 if args.width <= 3.25 else 5,
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
                "git_sha": git_sha(),
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                "full_labels_sha256": hashlib.sha256(
                    args.full.read_bytes()
                ).hexdigest(),
                "png_sha256": hashlib.sha256(
                    args.out.with_suffix(".png").read_bytes()
                ).hexdigest(),
                "pdf_sha256": hashlib.sha256(
                    args.out.with_suffix(".pdf").read_bytes()
                ).hexdigest(),
                "diagnostic": args.diagnostic,
                "diagnostic_row": PROBE_ROW if args.diagnostic == "mirnov" else MHR_ROW,
                "seed": args.seed,
                "shots": shots,
                "split": "development (train and validation); blind test excluded",
                "spectrogram_coverage_s": coverage,
                "labels": str(args.labels),
                "plotted_intervals": selected_spans,
                "uncertain_rows": [
                    {
                        "shot": int(r.shot),
                        "t_start": r.t_start,
                        "t_end": r.t_end,
                        **parse_attrs(r.attrs),
                    }
                    for r in table[
                        (table.shot.isin(shots)) & (table.category == 2)
                    ].itertuples(index=False)
                ],
                "frequency_khz": [0, 50],
                "dpi": 150,
                "width_inches": args.width,
                "font_pt": FONT,
                "lock_example_rule": lock_rule,
                "use": "column example"
                if args.width <= 3.25
                else "supplementary audit gallery",
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
