"""Render train-shot sawtooth overviews and 60 ms crash/profile panels.

All plots use the frozen detector records and cached, native-rate-antialiased
signals. Gallery selection reuses the original random train-shot sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sawtooth_physics import REPO, REVIEW, SEED, WORK, local_scalar, save_json

from labeler.config import Paths
from labeler.sawtooth.physics import Rule

BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
GRAY = "#999999"


def mean_trace(values):
    valid = np.isfinite(values)
    return np.divide(
        np.where(valid, values, 0).sum(axis=0),
        valid.sum(axis=0),
        out=np.full(values.shape[1], np.nan),
        where=valid.sum(axis=0) > 0,
    )


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2], strict=True)


def state_spans(record):
    """Read either explicit four-state intervals or state-bearing train spans."""
    spans = record.get("states", record.get("state_intervals", []))
    if not spans:
        spans = record.get("intervals", [])
    for span in spans:
        attrs = span.get("attrs", {})
        state = span.get("state", attrs.get("state", "present"))
        yield float(span["start_s"]), float(span["end_s"]), state


def shade_states(axis, t, observable, record):
    for lo, hi, state in state_spans(record):
        if state in ("present", "uncertain", "unassessed"):
            color = {"present": GREEN, "uncertain": ORANGE, "unassessed": GRAY}[state]
            axis.axvspan(lo * 1000, hi * 1000, color=color, alpha=0.12, lw=0)
    dt = np.median(np.diff(t))
    for start, stop in runs(~observable):
        axis.axvspan(
            t[start] * 1000,
            (t[stop - 1] + dt) * 1000,
            color=GRAY,
            alpha=0.22,
            lw=0,
        )


def crash_state(crash):
    attrs = crash.get("attrs", {})
    return crash.get("state", attrs.get("state", "present"))


def state_at(time, record, t, observable):
    index = min(np.searchsorted(t, time), len(t) - 1)
    if not observable[index]:
        return "unassessed"
    for start, end, state in state_spans(record):
        if start <= time < end:
            return state
    return "absent"


def marker_color(crash):
    return ORANGE if crash_state(crash) == "uncertain" else GREEN


def trace_legend():
    return [
        Line2D([], [], color=BLUE, lw=1, label="Core ECE proxy (20–27)"),
        Line2D([], [], color=VERMILION, lw=1, label="Outer ECE proxy (8–15)"),
        Line2D(
            [],
            [],
            color=GREEN,
            lw=1,
            marker="o",
            markersize=3,
            label="Present crash candidate",
        ),
        Line2D(
            [],
            [],
            color=ORANGE,
            lw=1,
            marker="o",
            markersize=3,
            label="Uncertain crash candidate",
        ),
        Patch(facecolor=GREEN, alpha=0.12, label="Present bin span"),
        Patch(facecolor=ORANGE, alpha=0.12, label="Uncertain bin span"),
        Patch(facecolor=GRAY, alpha=0.3, label="Unassessed: core ECE invalid"),
    ]


def overview(shot, record, t, y, observable, destination):
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.4), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.08, top=0.70, hspace=0.28)
    fig.suptitle(f"DIII-D {shot}: whole-shot overview", y=0.99, fontsize=8)
    qreference = 1 + Rule().qmin_margin
    handles = trace_legend() + [
        Line2D([], [], color="0.25", lw=0.7, label="EFIT01 q-min (magnetics only)"),
        Line2D(
            [],
            [],
            color="0.25",
            ls=":",
            lw=0.7,
            label=f"q = {qreference:g}: soft conflict reference",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=1,
        frameon=False,
        handlelength=1.5,
        labelspacing=0.15,
    )
    stride = max(1, len(t) // 4500)
    for rows, color in ((slice(20, 28), BLUE), (slice(8, 16), VERMILION)):
        trace = mean_trace(y[rows])
        axes[0].plot(t[::stride] * 1000, trace[::stride], color=color, lw=0.6)
    for axis in axes:
        shade_states(axis, t, observable, record)
        axis.grid(axis="y", color="0.9", lw=0.4)
    crashes = record.get("crashes", [])
    for crash in crashes:
        axes[0].axvline(
            crash["time_s"] * 1000, color=marker_color(crash), lw=0.4, alpha=0.35
        )
    axes[0].set_ylabel("ECE Te (keV)")
    for state in ("present", "uncertain"):
        picks = [r for r in crashes if crash_state(r) == state]
        axes[1].scatter(
            [r["time_s"] * 1000 for r in picks],
            [r["attrs"].get("inversion_channel", np.nan) for r in picks],
            s=5,
            color=GREEN if state == "present" else ORANGE,
        )
    axes[1].set_ylabel("Inversion channel")
    axes[1].set_ylim(-0.5, 47.5)
    axes[1].set_yticks([0, 16, 32, 47])
    axes[1].set_title("Channel boundary; calibrated radius unavailable", fontsize=7)
    qmin = local_scalar(shot, "qmin", Paths.from_env())
    if qmin is not None:
        axes[2].plot(qmin[0] * 1000, qmin[1], color="0.25", lw=0.6)
    axes[2].axhline(qreference, color="0.25", ls=":", lw=0.7)
    axes[2].set_ylabel("EFIT01 q-min")
    qlimit = 2.5
    if qmin is not None:
        visible_q = np.asarray(qmin[1])[
            (qmin[0] >= t[0]) & (qmin[0] <= t[-1]) & np.isfinite(qmin[1])
        ]
        if len(visible_q):
            qlimit = max(qlimit, float(visible_q.max()) * 1.03)
    axes[2].set_ylim(0, qlimit)
    axes[2].set_xlim(float(t[0] * 1000), float(t[-1] * 1000))
    axes[2].set_xlabel("Time (ms)")
    return save_figure(fig, destination / f"shot_{shot}_overview")


def choose_crash(record, t, y, observable):
    crashes = record.get("crashes", [])
    # Prefer the middle uncertain train to expose unresolved q/ECE conflicts.
    uncertain = [r for r in crashes if crash_state(r) == "uncertain"]
    choices = uncertain or crashes
    if choices:
        return choices[len(choices) // 2], "Detected crash"
    core = mean_trace(y[20:28])
    finite = np.isfinite(core) & observable
    derivative = np.diff(core, prepend=np.nan)
    valid = finite & np.isfinite(derivative)
    if valid.any():
        index = int(np.argmin(np.where(valid, derivative, np.inf)))
    elif observable.any():
        indices = np.flatnonzero(observable)
        index = int(indices[len(indices) // 2])
    else:
        index = len(t) // 2
    return {"time_s": float(t[index]), "attrs": {}}, "Diagnostic window; no label"


def crash_window(shot, record, t, y, observable, destination):
    crash, selection = choose_crash(record, t, y, observable)
    time = crash["time_s"]
    state = (
        crash_state(crash)
        if selection == "Detected crash"
        else state_at(time, record, t, observable)
    )
    near = (t >= time - 0.03) & (t <= time + 0.03)
    pre = (t >= time - 0.002) & (t < time - 0.0005)
    post = (t >= time + 0.0005) & (t < time + 0.002)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        before = np.nanmedian(y[:, pre], axis=1)
        after = np.nanmedian(y[:, post], axis=1)
    central = crash["attrs"].get("central_channel")
    if central is None:
        usable = np.where(np.isfinite(before[20:36]), before[20:36], -np.inf)
        central = int(np.argmax(usable) + 20)
    gain_start = crash["attrs"].get("rise_start", 8)
    gain_stop = crash["attrs"].get("rise_stop", 16)
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.1))
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.10, top=0.66, hspace=1.20)
    state_kind = (
        "Crash candidate state" if selection == "Detected crash" else "Bin state"
    )
    fig.suptitle(
        f"DIII-D {shot}: {selection.lower()} at {time * 1000:.1f} ms\n"
        f"{state_kind}: {state}",
        y=0.99,
        fontsize=8,
    )
    handles = trace_legend()
    handles[0] = Line2D(
        [], [], color=BLUE, lw=1, label=f"Central ECE proxy (channel {central})"
    )
    gain_label = (
        f"Gain-block ECE mean ({gain_start}–{gain_stop - 1})"
        if "rise_start" in crash["attrs"]
        else f"Outer ECE proxy mean ({gain_start}–{gain_stop - 1})"
    )
    handles[1] = Line2D(
        [],
        [],
        color=VERMILION,
        lw=1,
        label=gain_label,
    )
    if selection != "Detected crash":
        handles += [
            Line2D([], [], color="0.25", lw=0.8, label="Diagnostic-window centre")
        ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        frameon=False,
        ncol=1,
        labelspacing=0.15,
    )
    axes[0].plot(t[near] * 1000, y[central, near], color=BLUE, lw=0.8)
    axes[0].plot(
        t[near] * 1000,
        mean_trace(y[gain_start:gain_stop])[near],
        color=VERMILION,
        lw=0.8,
    )
    shade_states(axes[0], t, observable, record)
    axes[0].axvline(
        time * 1000,
        color=marker_color(crash) if selection == "Detected crash" else "0.25",
        lw=0.8,
    )
    axes[0].set_xlim(max(t[0], time - 0.03) * 1000, min(t[-1], time + 0.03) * 1000)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("ECE Te (keV)")
    channels = np.arange(len(y))
    axes[1].plot(channels, before, color=BLUE, lw=0.8, label="Before (−2 to −0.5 ms)")
    axes[1].plot(
        channels, after, color=VERMILION, lw=0.8, label="After (+0.5 to +2 ms)"
    )
    inversion = crash["attrs"].get("inversion_channel")
    if inversion is not None:
        axes[1].axvline(
            inversion,
            color=marker_color(crash),
            ls=":",
            lw=0.8,
            label="Inversion boundary",
        )
    axes[1].legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.05), frameon=False, ncol=1
    )
    axes[1].set_xlim(-0.5, 47.5)
    axes[1].set_xlabel("ECE channel (spatial mapping unavailable)")
    axes[1].set_ylabel("ECE Te (keV)")
    for axis in axes:
        axis.grid(axis="y", color="0.9", lw=0.4)
    return save_figure(fig, destination / f"shot_{shot}_crash"), {
        "shot": shot,
        "center_s": time,
        "selection": selection,
        "state": state,
        "bin_state": state_at(time, record, t, observable),
        "window_ms": 60,
        "inversion_channel": inversion,
        "central_channel": central,
        "gain_block_channels": [gain_start, gain_stop],
        "central_relative_drop": crash["attrs"].get("central_relative_drop"),
        "uncertainty_reasons": crash["attrs"].get("uncertainty_reasons", []),
    }


def save_figure(fig, stem):
    paths = []
    for extension in ("pdf", "png"):
        path = stem.with_suffix("." + extension)
        fig.savefig(path, dpi=150)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument(
        "--output", type=Path, default=REPO / "outputs/labeler/sawtooth/fix"
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPO / "outputs/labeler/sawtooth/fix/gallery.json",
    )
    parser.add_argument(
        "--confirm-inspection",
        action="store_true",
        help="Record image-viewer inspection after every generated PNG was viewed",
    )
    args = parser.parse_args()
    if args.confirm_inspection:
        path = args.output / "gallery.json"
        record = json.loads(path.read_text())
        paths = [Path(value) for value in record["figures"] if value.endswith(".png")]
        record["png_inspection"] = {
            "complete": True,
            "method": "Each PNG viewed with the agent image viewer",
            "sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
        }
        save_json(path, record)
        return
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    expert = set(pd.read_csv(REVIEW).shot)
    train = set(cohort.loc[cohort.split == "train", "shot"]) - expert
    previous = json.loads(args.selection.read_text()) if args.selection.exists() else {}
    shots = previous.get("shots", [])
    if len(shots) != 12 or not set(shots) <= train:
        eligible = sorted(
            s for s in train if (args.work / "signals" / f"{s}.npz").exists()
        )
        shots = (
            np.random.default_rng(SEED)
            .choice(eligible, size=12, replace=False)
            .tolist()
        )
    destination = args.work / "gallery"
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    figures, windows = [], []
    for shot in shots:
        record = json.loads((args.work / "shots" / f"{shot}.json").read_text())
        with np.load(args.work / "signals" / f"{shot}.npz") as data:
            t, y = data["t"], data["y"].astype(float)
            observable = data["observable"].astype(bool)
        figures.extend(overview(shot, record, t, y, observable, destination))
        crash_figures, window = crash_window(
            shot, record, t, y, observable, destination
        )
        figures.extend(crash_figures)
        windows.append(window)
    save_json(
        args.output / "gallery.json",
        {
            "seed": SEED,
            "shots": shots,
            "sampling": (
                "Original uniform random usable nonexpert train-shot sample retained"
            ),
            "figures": figures,
            "crash_windows": windows,
            "geometry_note": (
                "Verified spatial pairing unavailable; channel order is not radius."
            ),
            "q_note": "EFIT01 q-min is a soft conflict flag; it never defines absence.",
            "marker_note": (
                "Crash markers show diagnostic candidate state; shaded spans show "
                "canonical bin state. Crash points and sustained-presence spans "
                "have different temporal support."
            ),
            "png_inspection": "pending",
        },
    )


if __name__ == "__main__":
    main()
