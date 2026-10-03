#!/usr/bin/env python
"""Example figure: `elm-ours` on two reviewed shots, beside ELM-O and the ELM clock.

    python scripts/labeler/elm_example_figure.py --run cv2 [--out-dir DIR]

Two shots of the 73 with BES are shown, picked by a rule, not by eye: of the shots
whose scored bins are 10 to 90 % present, the ones at the 75th and the 25th percentile
of the per-shot F1 of `elm-ours` (a typical and a weaker shot). Each panel shows the
filterscope D-alpha (FS02, log level), the reviewed spans (crowd, single ELM, absent),
the out-of-fold event probability of `elm-ours` with its fold's threshold, and the
spans ELM-O and the ELM clock detect, over a window of up to 1.5 s from 100 ms before
the first present span. Writes `fig_elm_examples.pdf`, `.png` (150 dpi) and
`fig_elm_examples.json` (the shots, the rule, their F1) under
`$LABELER_ROOT/round4/elm/figures/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from labeler.config import Paths, git_sha
from labeler.elm import compare, inputs, methods, train

WINDOW_MS = 1500.0
LEAD_MS = 100.0
#: Okabe-Ito colours: crowd, single ELM, absent, event probability, ELM-O, clock.
CROWD, SINGLE, ABSENT = "#E69F00", "#0072B2", "#999999"
PROB, ELMO, CLOCK = "#009E73", "#CC79A7", "#56B4E9"


def f1_of(truth: np.ndarray, call: np.ndarray) -> float:
    tp = float(np.sum(truth & call))
    fp = float(np.sum(~truth & call))
    fn = float(np.sum(truth & ~call))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")


def pick_shots(data, sets, oof) -> tuple[list[int], dict]:
    """The 75th and 25th percentile shots by per-shot F1 among the informative ones."""
    bes = sets["bes73"]
    rows = []
    for s in bes.shots:
        part = methods.trace_part(
            data[s].spans,
            s,
            bes.bins[s],
            bes.cover[s],
            oof.trace(s)[0],
            oof.threshold[s],
        )
        if len(part.truth) and 0.1 <= part.truth.mean() <= 0.9:
            rows.append((f1_of(part.truth, part.call), s))
    rows.sort()
    n = len(rows)
    lo, hi = rows[int(0.25 * (n - 1))], rows[round(0.75 * (n - 1))]
    info = {
        "rule": "bes73 shots with 10-90 % present bins, ordered by per-shot F1 of "
        "elm-ours; the shots at the 75th and 25th percentile ranks",
        "candidates": n,
        "per_shot_f1": {str(s): f for f, s in rows},
    }
    return [hi[1], lo[1]], info


def window_of(spans, cov0, cov1) -> tuple[float, float]:
    present = spans[spans.kind.isin(["individual", "crowd"])]
    t0 = float(present.t_start.min()) - LEAD_MS if len(present) else float(cov0.min())
    t0 = max(t0, float(cov0.min()))
    return t0, min(t0 + WINDOW_MS, float(cov1.max()))


def shade(ax, spans, t0, t1) -> None:
    for r in spans.itertuples():
        if r.t_end < t0 or r.t_start > t1:
            continue
        colour = {"crowd": CROWD, "individual": SINGLE, "absent": ABSENT}.get(r.kind)
        if colour is None:
            continue
        ax.axvspan(
            max(r.t_start, t0),
            min(r.t_end, t1),
            color=colour,
            alpha=0.5 if r.kind == "individual" else 0.25,
            lw=0,
        )


def draw_shot(axes, shot, data, sets, oof, elmo, clock, panel) -> None:
    d = data[shot]
    t0, t1 = window_of(d.spans, d.cov0, d.cov1)
    x = d.x[inputs.CHANNELS.index("fs02")]
    t = inputs.GRID0_MS + 0.1 * (np.arange(len(x)) + 0.5)
    sel = (t >= t0) & (t <= t1)
    event = oof.trace(shot)[0]
    tt = inputs.GRID0_MS + np.arange(len(event)) + 0.5
    sl = (tt >= t0) & (tt <= t1)
    ax, bx = axes
    for a in axes:
        shade(a, d.spans, t0, t1)
    ax.plot(t[sel], x[sel], color="#222222", lw=0.6)
    ax.set_ylabel("D-alpha FS02\n(log level)", fontsize=8)
    ax.set_title(f"({panel}) shot {shot}", fontsize=9, loc="left")
    bx.plot(tt[sl], event[sl], color=PROB, lw=1.0, label="elm-ours")
    bx.axhline(oof.threshold[shot], color=PROB, lw=0.8, ls="--")
    bx.set_ylim(-0.02, 1.02)
    bx.set_ylabel("event\nprobability", fontsize=8)
    for y, spans, colour in (
        (0.93, elmo.get(shot), ELMO),
        (0.85, clock.get(shot), CLOCK),
    ):
        if spans is None:
            continue
        for r in spans.itertuples():
            if r.t_end_ms < t0 or r.t_start_ms > t1:
                continue
            if r.t_end_ms - r.t_start_ms < 4.0:  # a single ELM: draw a tick
                mid = 0.5 * (r.t_start_ms + r.t_end_ms)
                bx.plot([mid, mid], [y - 0.035, y + 0.035], color=colour, lw=1.2)
            else:
                bx.plot(
                    [max(r.t_start_ms, t0), min(r.t_end_ms, t1)],
                    [y, y],
                    color=colour,
                    lw=2.5,
                    solid_capstyle="butt",
                )
    bx.set_xlim(t0, t1)
    bx.set_xlabel("time in shot (ms)", fontsize=8)
    for a in axes:
        a.tick_params(labelsize=8)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    out_dir = args.out_dir or paths.root / "round4" / "elm" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = train.load(paths)
    oof = methods.Oof(paths.root / "round4" / "elm" / "cv" / args.run)
    sets = compare.load_sets(paths, data)
    elmo, clock = compare.load_detected(paths)
    shots, info = pick_shots(data, sets, oof)

    fig, axes = plt.subplots(
        2, 2, figsize=(7.0, 4.0), sharex="col", gridspec_kw={"height_ratios": [1, 1]}
    )
    for i, shot in enumerate(shots):
        draw_shot(axes[:, i], shot, data, sets, oof, elmo, clock, "ab"[i])
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CROWD, alpha=0.25, lw=0),
        plt.Rectangle((0, 0), 1, 1, color=SINGLE, alpha=0.5, lw=0),
        plt.Rectangle((0, 0), 1, 1, color=ABSENT, alpha=0.25, lw=0),
        plt.Line2D([0], [0], color=PROB, lw=1.0),
        plt.Line2D([0], [0], color=PROB, lw=0.8, ls="--"),
        plt.Line2D([0], [0], color=ELMO, lw=1.2),
        plt.Line2D([0], [0], color=CLOCK, lw=2.5),
    ]
    names = [
        "reviewed: crowd",
        "reviewed: single ELM",
        "reviewed: absent",
        "elm-ours",
        "its threshold",
        "elm-elmo ELMs (ticks)",
        "elm-clock spans",
    ]
    fig.legend(
        handles,
        names,
        loc="lower center",
        ncol=4,
        fontsize=7,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.4,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_elm_examples.{ext}", dpi=150)
    info.update(
        {
            "git": git_sha(),
            "run": args.run,
            "shots": [int(s) for s in shots],
            "figure": str(out_dir / "fig_elm_examples.pdf"),
        }
    )
    (out_dir / "fig_elm_examples.json").write_text(json.dumps(info, indent=1))
    print("shots", shots, {s: round(info["per_shot_f1"][str(s)], 3) for s in shots})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
