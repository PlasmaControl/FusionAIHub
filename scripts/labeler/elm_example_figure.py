#!/usr/bin/env python
"""Example figure: `elm-ours` on two reviewed shots, beside elm-elmo and the ELM clock.

    python scripts/labeler/elm_example_figure.py --run cv2 [--out-dir DIR]

Two shots of the 73 with BES are shown, chosen by a fixed rule with no manual
substitution: of the shots whose scored bins are 10 to 90 % present, panel (a) is at
the 75th and panel (b) at the 25th percentile of per-shot F1 of `elm-ours`.
Each panel shows normalized filterscope D-alpha (FS02) as log10(S/S0), with S0 the
preprocessing centre defined in the caption, the reviewed spans (crowd, non-crowd
present, absent), the out-of-fold event probability of `elm-ours` with its fold's
threshold, and the spans elm-elmo and the ELM clock detect, over a window of up to
1.5 s from 100 ms before the first present span. Writes `fig_elm_examples.pdf`,
`.png` (150 dpi), `fig_elm_examples_caption.tex` (the caption, which defines S0 and
the selection rule) and `fig_elm_examples.json` (the shots, the rule, their F1)
under `$LABELER_ROOT/round4/elm/figures/`.
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
from labeler.elm import compare, inputs, labels, methods, prepare, train

WINDOW_MS = 1500.0
LEAD_MS = 100.0
#: Okabe-Ito colours: crowd, non-crowd, absent, probability, elm-elmo, clock.
CROWD, NON_CROWD, ABSENT = "#E69F00", "#CC79A7", "#999999"
PROB, ELMO, CLOCK = "#009E73", "#D55E00", "#56B4E9"


#: Reviewed non-crowd present spans that sit in a high-recycling phase, where ELM
#: identity is ambiguous: the span and the time from which FS02-04 D-alpha rises. The
#: caption carries the note only when the shot is shown and the data confirm both the
#: span and a rise of at least `RISE_MIN_LOG10` against the 200 ms before the rise.
HIGH_RECYCLING = {200427: {"span_ms": (2689.0, 2765.0), "rise_ms": 2430.0}}
RISE_MIN_LOG10 = 0.3
RISE_BASELINE_MS = 200.0


def f1_of(truth: np.ndarray, call: np.ndarray) -> float:
    tp = float(np.sum(truth & call))
    fp = float(np.sum(~truth & call))
    fn = float(np.sum(truth & ~call))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")


def pick_shots(data, sets, oof) -> tuple[list[int], dict]:
    """The 75th and 25th percentile shots by per-shot F1 among the informative ones.

    A shot is a candidate when it is a bes73 shot whose scored bins are 10 to 90 %
    present. The rule is applied mechanically; no shot is replaced by hand.
    """
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
    low_rank, high_rank = int(0.25 * (n - 1)), round(0.75 * (n - 1))
    lo, hi = rows[low_rank], rows[high_rank]
    info = {
        "rule": "bes73 shots with 10-90 % present scored bins, ordered by per-shot "
        "F1 of elm-ours; panel a at the 75th percentile rank and panel b at the "
        "25th percentile rank; no substitution",
        "ranks": {"a": high_rank, "b": low_rank},
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
        colour = {"crowd": CROWD, "non_crowd": NON_CROWD, "absent": ABSENT}.get(r.kind)
        if colour is None:
            continue
        ax.axvspan(
            max(r.t_start, t0),
            min(r.t_end, t1),
            color=colour,
            alpha=0.5 if r.kind == "non_crowd" else 0.25,
            lw=0,
        )


def scored_window_audit(shot, spans, bins, event, threshold, elmo, t0, t1):
    """Record exact scored bins, independent of the longer displayed trace window."""
    scores = labels.bin_scores(event, bins)
    calls = scores >= threshold
    detected = elmo.get(shot)
    elmo_calls = (
        labels.hard_hits(
            detected.t_start_ms.to_numpy(float),
            detected.t_end_ms.to_numpy(float),
            bins,
        )
        if detected is not None
        else np.zeros(len(bins.t0), dtype=bool)
    )
    keep = (bins.t0 < t1) & (bins.t0 + labels.BIN_MS > t0)
    parts = [
        {
            "span_ms": [float(start), float(start + labels.BIN_MS)],
            "truth": int(truth),
            "kind": kind,
            "ours_bin_mean": float(value),
            "ours_call": bool(ours_call),
            "elmo_call": bool(elmo_call),
        }
        for start, truth, kind, value, ours_call, elmo_call in zip(
            bins.t0[keep],
            bins.truth[keep],
            bins.kind[keep],
            scores[keep],
            calls[keep],
            elmo_calls[keep],
            strict=True,
        )
    ]
    selected_spans = spans[(spans.t_start < t1) & (spans.t_end > t0)]
    display_edges = (
        np.arange(np.floor(t0 / labels.BIN_MS), np.ceil(t1 / labels.BIN_MS))
        * labels.BIN_MS
    )
    return {
        "interval_ms": [t0, t1],
        "scoring_set": "bes73 before DSM-specific restrictions",
        "bin_rule": "50 ms bins wholly inside one reviewed scored span and "
        "elm-elmo analysed coverage; the span must have at least half its time covered",
        "reviewed_spans": [
            {
                "span_ms": [float(r.t_start), float(r.t_end)],
                "kind": r.kind,
            }
            for r in selected_spans.itertuples()
        ],
        "scored_bins": parts,
        "excluded_display_grid_bins_ms": [
            [float(start), float(start + labels.BIN_MS)]
            for start in display_edges
            if not np.isclose(bins.t0, start).any()
        ],
        "ours_false_negative_bins": sum(
            row["truth"] == 1 and not row["ours_call"] for row in parts
        ),
        "elmo_positive_bins": sum(row["elmo_call"] for row in parts),
    }


def recycling_note(shot, d) -> dict | None:
    """The high-recycling note for a shown shot, with the FS02-04 levels behind it."""
    spec = HIGH_RECYCLING.get(shot)
    if spec is None:
        return None
    start, stop = spec["span_ms"]
    rise = spec["rise_ms"]
    spans = d.spans[d.spans.kind == "non_crowd"]
    if not (
        ((spans.t_start - start).abs() <= 1) & ((spans.t_end - stop).abs() <= 1)
    ).any():
        return None
    t = inputs.GRID0_MS + inputs.DT_MS * (np.arange(d.x.shape[1]) + 0.5)
    valid = d.x[inputs.VALID] > 0
    before = valid & (t >= rise - RISE_BASELINE_MS) & (t < rise)
    inside = valid & (t >= start) & (t < stop)
    levels = {}
    for channel in ("fs02", "fs03", "fs04"):
        x = d.x[inputs.CHANNELS.index(channel)] * inputs.FS_SCALE
        levels[channel] = {
            "median_log10_before_rise": float(np.median(x[before])),
            "median_log10_in_span": float(np.median(x[inside])),
        }
    if not all(
        v["median_log10_in_span"] - v["median_log10_before_rise"] >= RISE_MIN_LOG10
        for v in levels.values()
    ):
        return None
    return {
        "span_ms": [start, stop],
        "rise_ms": rise,
        "baseline_ms": RISE_BASELINE_MS,
        "minimum_rise_log10": RISE_MIN_LOG10,
        "levels": levels,
    }


def draw_shot(axes, shot, data, sets, oof, elmo, clock, panel) -> dict:
    d = data[shot]
    t0, t1 = window_of(d.spans, d.cov0, d.cov1)
    # The existing input is (log10(max(FS02, floor)) - centre) / scale.
    # Removing only scale plots log10(max(FS02, floor) / 10**centre), without
    # assigning a physical unit to the source ordinate.
    x = d.x[inputs.CHANNELS.index("fs02")] * inputs.FS_SCALE
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
    ax.set_ylabel("FS02 D-alpha\n" + r"$\log_{10}(S/S_0)$", fontsize=8)
    ax.set_title(f"({panel}) shot {shot}", fontsize=9, loc="left")
    bx.plot(tt[sl], event[sl], color=PROB, lw=1.0, label="elm-ours")
    bx.axhline(oof.threshold[shot], color=PROB, lw=0.8, ls="--")
    bx.set_ylim(-0.02, 1.02)
    bx.set_ylabel("ELMy-occupancy\nprobability", fontsize=8)
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
        "dalpha_limits_log10_normalized": list(ax.get_ylim()),
        "probability_limits": list(bx.get_ylim()),
        "fold": int(oof.fold_of[shot]),
        "inner_validation_threshold": float(oof.threshold[shot]),
        "high_recycling": recycling_note(shot, d),
        "displayed_scoring_audit": scored_window_audit(
            shot,
            d.spans,
            sets["bes73"].bins[shot],
            event,
            oof.threshold[shot],
            elmo,
            t0,
            t1,
        ),
    }
    return info


def disagreement_bins(audit: dict) -> list[dict]:
    """Scored crowd bins that `elm-ours` calls absent and elm-elmo calls present."""
    return [
        r
        for r in audit["scored_bins"]
        if r["truth"] == 1 and not r["ours_call"] and r["elmo_call"]
    ]


def disagreement_sentence(panel: dict) -> str:
    """The disagreement with elm-elmo in the panel where it is largest (>= 3 bins)."""
    audit = panel["displayed_scoring_audit"]
    bins = disagreement_bins(audit)
    first, last = bins[0]["span_ms"][0], bins[-1]["span_ms"][1]
    text = (
        f"In panel ({panel['panel']}) the low elm-ours output from {first:g} to "
        f"{last:g} ms covers {len(bins)} scored 50 ms crowd bins that elm-ours "
        "calls absent and elm-elmo calls present."
    )
    excluded = audit["excluded_display_grid_bins_ms"]
    starts = [r["span_ms"][0] for r in audit["reviewed_spans"] if r["kind"] == "crowd"]
    straddle = [
        (a, b) for a, b in excluded if any(a < t < b for t in starts) and b <= first
    ]
    if straddle:
        a, b = straddle[-1]
        start = next(t for t in starts if a < t < b)
        text += (
            f" The {a:g}--{b:g} ms bin straddles the reviewed crowd start at "
            f"{start:g} ms and is not scored."
        )
    return text


def recycling_sentence(panels) -> str:
    """One sentence per shown panel whose reviewed non-crowd span is in a high-
    recycling phase (see `HIGH_RECYCLING`)."""
    out = []
    for panel in panels:
        note = panel.get("high_recycling")
        if note:
            start, stop = note["span_ms"]
            out.append(
                f"In panel ({panel['panel']}) the reviewed non-crowd present span at "
                f"{start:g}--{stop:g} ms lies in a high-recycling phase (FS02--04 "
                f"D-alpha rises from {note['rise_ms']:g} ms), so ELM identity there "
                "is ambiguous."
            )
    return " ".join(out)


def caption_text(shots, info, panels, disagreement) -> str:
    """The figure caption in LaTeX; it states the selection rule and defines S0."""
    first, second = panels
    return (
        f"Reviewed spans and detector outputs on shots {shots[0]} and {shots[1]}. "
        "Shots are chosen by a fixed rule, with no substitution: among BES shots "
        "with 10--90\\% present scored 50 ms bins "
        f"({info['candidates']} candidates), panel (a) is at the 75th and panel (b) "
        "at the 25th percentile of per-shot out-of-fold elm-ours F1. "
        "Top: FS02 D-alpha as $\\log_{10}(S/S_0)$, where "
        f"$S_0=10^{{{inputs.FS_CENTRE:g}}}$ ph/(sr\\,cm$^2$\\,s), the corpus "
        "filterscope unit, is the preprocessing centre (input floor "
        f"$10^{{{np.log10(inputs.FS_FLOOR):g}}}$ in the same unit); the ratio is "
        "dimensionless, and the stored input records keep no unit metadata. "
        "Shading: reviewed crowd, non-crowd present and absent spans. "
        "Bottom: out-of-fold ELMy-occupancy probability with its fold's threshold "
        f"(a: fold {first['fold']}, {first['inner_validation_threshold']:.3f}; b: "
        f"fold {second['fold']}, {second['inner_validation_threshold']:.3f}; each "
        "held-out fold has its own inner-validation shots), elm-elmo detections "
        "(short ones as ticks) and the elm-clock present spans, which seeded the "
        "review and are not independent of it. "
        + (disagreement_sentence(disagreement) + " " if disagreement else "")
        + (recycling_sentence(panels) + " " if recycling_sentence(panels) else "")
        + "Windows start 100 ms before the first reviewed present span, clipped "
        "to input coverage, and last up to 1500 ms."
    )


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
        "elm-elmo detections (ticks)",
        "elm-clock present spans",
    ]
    visible_kinds = set()
    for panel in panels:
        shot = panel["shot"]
        start, stop = panel["window_ms"]
        spans = data[shot].spans
        visible_kinds.update(
            spans.loc[(spans.t_start < stop) & (spans.t_end > start), "kind"]
        )
    shading = [
        i
        for i, kind in enumerate(("crowd", "non_crowd", "absent"))
        if kind in visible_kinds
    ]
    fig.legend(
        [handles[i] for i in shading],
        [names[i] for i in shading],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=len(shading),
        fontsize=7.5,
        frameon=False,
        columnspacing=1.5,
        handlelength=1.4,
    )
    fig.legend(
        handles[3:],
        names[3:],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        fontsize=7.5,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.4,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_elm_examples.{ext}", dpi=150)
    counts = [len(disagreement_bins(p["displayed_scoring_audit"])) for p in panels]
    disagreement = panels[int(np.argmax(counts))] if max(counts) >= 3 else None
    caption = caption_text(shots, info, panels, disagreement)
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
            "evaluation_sources": {
                role: {"path": str(path), "sha256": sha256_of(path)}
                for role, path in {
                    "reviewed_spans": prepare.review_csv(paths),
                    "elmo_coverage": (
                        paths.root / compare.ELMO_DIR / "review_coverage.csv"
                    ),
                    "elmo_detections": (
                        paths.root / compare.ELMO_DIR / "review_elms.csv"
                    ),
                }.items()
            },
            "plotted_heads": ["occupancy"],
            "axes": {
                "x": "time in shot (ms)",
                "dalpha": "log10(max(FS02, FS_FLOOR) / S0), dimensionless ratio; S0 "
                "and the floor are in the corpus filterscope unit ph/(sr cm2 s) "
                "(src/labeler/events/raw.py); the stored input records keep no unit "
                "metadata",
                "input_transform_to_plot": "fs02 * FS_SCALE",
                "FS_SCALE": inputs.FS_SCALE,
                "FS_CENTRE": inputs.FS_CENTRE,
                "S0_native_ordinate": 10.0**inputs.FS_CENTRE,
                "S0_definition": "10**FS_CENTRE, the existing preprocessing centre",
                "floor_native_ordinate": inputs.FS_FLOOR,
                "reduction": "maximum per 0.1 ms cell; missing cells omitted",
                "probability": "out-of-fold ELMy-occupancy probability per 1 ms",
            },
            "window_rule": {"duration_ms": WINDOW_MS, "lead_ms": LEAD_MS},
            "caption": caption,
        }
    )
    (out_dir / "fig_elm_examples_caption.tex").write_text(caption + "\n")
    (out_dir / "fig_elm_examples.json").write_text(json.dumps(info, indent=1))
    print("shots", shots, {s: round(info["per_shot_f1"][str(s)], 3) for s in shots})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
