#!/usr/bin/env python
"""Example figure: `elm-ours` on two reviewed shots, beside ELM-O and the ELM clock.

    python scripts/labeler/elm_example_figure.py --run cv2 [--out-dir DIR]

Two shots of the 73 with BES are shown, picked by a rule, not by eye: of the shots
whose scored bins are 10 to 90 % present, the ones at the 75th and the 25th percentile
of the per-shot F1 of `elm-ours` (a typical and a weaker shot). Each panel shows the
filterscope D-alpha (FS02, log10 a.u.), the reviewed spans (crowd, non-crowd
present, absent),
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

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import compare, inputs, methods, prepare, train

WINDOW_MS = 1500.0
LEAD_MS = 100.0
#: Okabe-Ito colours: crowd, non-crowd, absent, probability, ELM-O, clock.
CROWD, NON_CROWD, ABSENT = "#E69F00", "#CC79A7", "#999999"
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
    present = spans[spans.kind.isin(["non_crowd", "crowd"])]
    t0 = float(present.t_start.min()) - LEAD_MS if len(present) else float(cov0.min())
    t0 = max(t0, float(cov0.min()))
    return t0, min(t0 + WINDOW_MS, float(cov1.max()))


def shade(ax, spans, t0, t1) -> None:
    for r in spans.itertuples():
        if r.t_end < t0 or r.t_start > t1:
            continue
        colour = {"crowd": CROWD, "non_crowd": NON_CROWD, "absent": ABSENT}.get(
            r.kind
        )
        if colour is None:
            continue
        ax.axvspan(
            max(r.t_start, t0),
            min(r.t_end, t1),
            color=colour,
            alpha=0.5 if r.kind == "non_crowd" else 0.25,
            lw=0,
        )


def draw_shot(axes, shot, data, sets, oof, elmo, clock, panel) -> dict:
    d = data[shot]
    t0, t1 = window_of(d.spans, d.cov0, d.cov1)
    x = d.x[inputs.CHANNELS.index("fs02")] * inputs.FS_SCALE + inputs.FS_CENTRE
    x = np.where(d.x[inputs.VALID] > 0, x, np.nan)
    t = inputs.GRID0_MS + inputs.DT_MS * (np.arange(len(x)) + 0.5)
    sel = (t >= t0) & (t <= t1)
    event = oof.trace(shot)[0]
    tt = inputs.GRID0_MS + np.arange(len(event)) + 0.5
    sl = (tt >= t0) & (tt <= t1)
    ax, bx = axes
    for a in axes:
        shade(a, d.spans, t0, t1)
    ax.plot(t[sel], x[sel], color="#222222", lw=0.6)
    ax.set_ylabel("D-alpha FS02\n(log10 a.u.)", fontsize=8)
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
            if r.t_end_ms - r.t_start_ms < 4.0:  # short detection: draw a tick
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
    info = {
        "panel": panel,
        "shot": int(shot),
        "window_ms": [t0, t1],
        "dalpha_limits_log10_au": list(ax.get_ylim()),
        "probability_limits": list(bx.get_ylim()),
        "fold": int(oof.fold_of[shot]),
        "inner_validation_threshold": float(oof.threshold[shot]),
    }
    if shot == 200427 and t0 <= 2690.0 and t1 >= 2760.0:
        text = "Reviewed non-crowd present span\n2690–2760 ms; D-alpha drop"
        ax.annotate(
            text,
            xy=(2760.0, float(np.interp(2760.0, t, x))),
            xytext=(0.24, 0.08),
            textcoords="axes fraction",
            fontsize=7,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": NON_CROWD, "lw": 0.7},
        )
        info["annotation"] = {
            "span_ms": [2690.0, 2760.0],
            "text": text.replace("\n", "; "),
            "interpretation": "Reviewed present span at a D-alpha drop; neither "
            "an independently verified isolated ELM nor a verified transition.",
        }
    return info


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
        2, 2, figsize=(7.0, 4.6), sharex="col", gridspec_kw={"height_ratios": [1, 1]}
    )
    panels = []
    for i, shot in enumerate(shots):
        panels.append(
            draw_shot(axes[:, i], shot, data, sets, oof, elmo, clock, "ab"[i])
        )
        fold = oof.fold_of[shot]
        panels[-1]["records"] = {
            role: {"path": str(path), "sha256": sha256_of(path)}
            for role, path in {
                "input": prepare.inputs_dir(paths) / f"{shot}.npy",
                "prediction": oof.dir / "pred" / f"{shot}.npz",
                "fold_threshold": oof.dir / f"fold{fold}" / "fold.json",
            }.items()
        }
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CROWD, alpha=0.25, lw=0),
        plt.Rectangle((0, 0), 1, 1, color=NON_CROWD, alpha=0.5, lw=0),
        plt.Rectangle((0, 0), 1, 1, color=ABSENT, alpha=0.25, lw=0),
        plt.Line2D([0], [0], color=PROB, lw=1.0),
        plt.Line2D([0], [0], color=PROB, lw=0.8, ls="--"),
        plt.Line2D([0], [0], color=ELMO, lw=1.2),
        plt.Line2D([0], [0], color=CLOCK, lw=2.5),
    ]
    names = [
        "reviewed: crowd",
        "reviewed: non-crowd present spans",
        "reviewed: absent",
        "elm-ours probability",
        "CV threshold",
        "ELM-O detections (ticks)",
        "elm-clock present spans",
    ]
    fig.legend(
        handles[:3],
        names[:3],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=3,
        fontsize=7.5,
        frameon=False,
        columnspacing=1.5,
        handlelength=1.4,
    )
    fig.legend(
        handles[3:],
        names[3:],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=4,
        fontsize=7.5,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.4,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_elm_examples.{ext}", dpi=150)
    info.update(
        {
            "git": git_sha(),
            "source_sha256": sha256_of(Path(__file__)),
            "run_record_sha256": sha256_of(oof.dir / "run.json"),
            "run": args.run,
            "shots": [int(s) for s in shots],
            "figure": str(out_dir / "fig_elm_examples.pdf"),
            "png": str(out_dir / "fig_elm_examples.png"),
            "placement": "two-column, 7-inch width; do not shrink to one column",
            "size_inches": [7.0, 4.6],
            "minimum_font_pt_at_placement": 7.0,
            "png_dpi": 150,
            "panels": panels,
            "source_records": {
                str(s): inputs.read_metadata(prepare.signals_dir(paths) / f"{s}.npz")
                for s in shots
            },
            "plotted_heads": ["occupancy"],
            "axes": {
                "x": "time in shot (ms)",
                "dalpha": "FS02 log10 a.u.; source physical units unconfirmed",
                "inverse_input_transform": "fs02 * FS_SCALE + FS_CENTRE",
                "FS_SCALE": inputs.FS_SCALE,
                "FS_CENTRE": inputs.FS_CENTRE,
                "floor_au": inputs.FS_FLOOR,
                "reduction": "maximum per 0.1 ms cell; missing cells omitted",
                "probability": "out-of-fold elm-ours event probability per 1 ms",
            },
            "window_rule": {"duration_ms": WINDOW_MS, "lead_ms": LEAD_MS},
            "caption": (
                f"Reviewed spans and detector outputs on shots {shots[0]} and "
                f"{shots[1]}, selected by a fixed rule from shots with BES. "
                "Among shots with 10–90% present 50 ms bins, panels "
                "(a) and (b) show the 75th and 25th percentile ranks of per-shot "
                "out-of-fold elm-ours F1. Top: FS02 D-alpha on a log10 a.u. axis, "
                "with the model-input normalization reversed; source physical "
                "units are unconfirmed. Shading marks reviewed crowd, non-crowd "
                "present and absent spans. Non-crowd spans are not verified "
                "isolated ELMs. Bottom: out-of-fold elm-ours event probability "
                "and the threshold selected on inner-validation shots. "
                f"Panel (a) uses fold {panels[0]['fold']} and threshold "
                f"{panels[0]['inner_validation_threshold']:.3f}; panel (b) uses "
                f"fold {panels[1]['fold']} and threshold "
                f"{panels[1]['inner_validation_threshold']:.3f}. They differ "
                "because each held-out fold has its own inner-validation set. "
                "Also shown are ELM-O "
                "detections (short detections drawn as ticks), and the original "
                "elm-clock present spans. The review began from that clock and "
                "is not independent of it. Each window begins 100 ms before the "
                "first reviewed present span, clipped to input coverage, and "
                "lasts up to 1500 ms. The annotated 2690–2760 ms span on shot "
                "200427 coincides with a D-alpha drop; no ELM or confinement "
                "transition is independently established. Place at 7-inch "
                "two-column width to preserve text of at least 7 pt."
            ),
        }
    )
    (out_dir / "fig_elm_examples.json").write_text(json.dumps(info, indent=1))
    print("shots", shots, {s: round(info["per_shot_f1"][str(s)], 3) for s in shots})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
