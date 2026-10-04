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
PURPLE = "#CC79A7"
ORIGINAL_GALLERY_SHOTS = [
    186532,
    196405,
    192148,
    204261,
    203563,
    202206,
    201978,
    196020,
    192149,
    191326,
    195064,
    193024,
]


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
    colors = {
        "present": GREEN,
        "uncertain": ORANGE,
        "unassessed": GRAY,
        "absent_q_prior": PURPLE,
    }
    for lo, hi, state in state_spans(record):
        if state in colors:
            axis.axvspan(lo * 1000, hi * 1000, color=colors[state], alpha=0.12, lw=0)
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
    return "uncertain"


def marker_color(crash):
    return ORANGE if crash_state(crash) == "uncertain" else GREEN


def proxy_groups(record):
    geometry = record.get("core_geometry", {})
    return (
        geometry.get("core_channels", list(range(20, 36))),
        geometry.get("outer_channels", list(range(8, 16))),
    )


def trace_legend(record, window=None):
    core, outer = proxy_groups(record)
    handles = [
        Line2D([], [], color=color, lw=1, label=group_label(record, channels, name))
        for channels, color, name in (
            (core, BLUE, "Core"),
            (outer, VERMILION, "Outer LFS"),
        )
        if channels
    ]
    return handles + state_legend(record, window)


def state_legend(record, window=None):
    handles = []
    shown = {
        state
        for lo, hi, state in state_spans(record)
        if window is None or (lo < window[1] and hi > window[0])
    }
    for state, color, label in (
        ("present", GREEN, "Present"),
        ("absent", "#FFFFFF", "Absent (ECE-tested)"),
        ("absent_q_prior", PURPLE, "Q-prior only"),
        ("uncertain", ORANGE, "Uncertain"),
        ("unassessed", GRAY, "Unassessed"),
    ):
        if state in shown:
            handles.append(
                Patch(facecolor=color, edgecolor="0.7", alpha=0.25, label=label)
            )
    return handles


def group_label(record, channels, name):
    if not channels:
        return f"{name}: unavailable"
    rho = record.get("core_geometry", {}).get("nominal_rho_median", [])
    values = [rho[c] for c in channels if c < len(rho) and rho[c] is not None]
    if values:
        lo, hi = min(values), max(values)
        value = f"{lo:.2f}" if hi - lo < 0.02 else f"{lo:.2f}–{hi:.2f}"
        return f"{name}: nominal ρ={value}"
    return f"{name}: ch {min(channels)}–{max(channels)}"


def overview(shot, record, t, y, observable, destination):
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.4), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.08, top=0.79, hspace=0.28)
    fig.suptitle(f"DIII-D {shot}: whole-shot overview", y=0.99, fontsize=8)
    qreference = 1 + record.get("rule", {}).get("qmin_margin", Rule().qmin_margin)
    qmin = local_scalar(shot, "qmin", Paths.from_env())
    handles = trace_legend(record) + [
        Line2D(
            [],
            [],
            color="0.25",
            ls=":",
            lw=0.7,
            label=f"q = {qreference:g}: conflict",
        ),
    ]
    if qmin is not None:
        handles.append(Line2D([], [], color="0.25", lw=0.7, label="EFIT01 q-min"))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        labelspacing=0.15,
        columnspacing=0.8,
    )
    stride = max(1, len(t) // 4500)
    core, outer = proxy_groups(record)
    for rows, color in ((core, BLUE), (outer, VERMILION)):
        trace = mean_trace(y[rows])
        axes[0].plot(t[::stride] * 1000, trace[::stride], color=color, lw=0.6)
    if not outer:
        axes[0].text(
            0.02,
            0.93,
            "Nominal outer LFS group unavailable",
            transform=axes[0].transAxes,
            va="top",
            fontsize=7,
        )
    for axis in axes:
        shade_states(axis, t, observable, record)
        axis.grid(axis="y", color="0.9", lw=0.4)
    crashes = record.get("crashes", [])
    for crash in crashes:
        axes[0].axvline(
            crash["time_s"] * 1000, color=marker_color(crash), lw=0.4, alpha=0.35
        )
    axes[0].set_ylabel("ECE Te (keV)")
    calibrated = any(r["attrs"].get("inversion_rho") is not None for r in crashes)
    rho_key = "inversion_rho" if calibrated else "inversion_nominal_rho"
    q1_key = "q1_rho" if calibrated else "q1_nominal_rho"
    use_rho = any(r["attrs"].get(rho_key) is not None for r in crashes)
    for state in ("present", "uncertain"):
        picks = [r for r in crashes if crash_state(r) == state]
        axes[1].scatter(
            [r["time_s"] * 1000 for r in picks],
            [
                r["attrs"].get(rho_key if use_rho else "inversion_channel")
                if r["attrs"].get(rho_key if use_rho else "inversion_channel")
                is not None
                else np.nan
                for r in picks
            ],
            s=5,
            color=GREEN if state == "present" else ORANGE,
        )
    if use_rho:
        q1 = [r for r in crashes if r["attrs"].get(q1_key) is not None]
        if q1:
            axes[1].scatter(
                [r["time_s"] * 1000 for r in q1],
                [r["attrs"][q1_key] for r in q1],
                marker="x",
                s=6,
                color="0.25",
                label="EFIT q = 1",
            )
            axes[1].legend(loc="upper right", frameon=False)
        axes[1].set_ylabel("Inversion ρ" if calibrated else "Nominal inversion ρ")
        axes[1].set_ylim(0, 1)
    else:
        axes[1].set_ylabel("Inversion channel")
        axes[1].set_ylim(-0.5, 47.5)
        axes[1].set_yticks([0, 16, 32, 47])
        axes[1].set_title(
            "No crash candidates"
            if not crashes
            else "Nominal inversion radius unavailable",
            fontsize=7,
        )
    if qmin is not None:
        axes[2].plot(qmin[0] * 1000, qmin[1], color="0.25", lw=0.6)
    axes[2].axhline(qreference, color="0.25", ls=":", lw=0.7)
    axes[2].set_ylabel("EFIT01 q-min")
    axes[2].set_ylim(0, 4)
    axes[2].set_xlim(float(t[0] * 1000), float(t[-1] * 1000))
    axes[2].set_xlabel("Time (ms)")
    return save_figure(fig, destination / f"shot_{shot}_overview")


def choose_crash(record, t, y, observable):
    crashes = record.get("crashes", [])
    # Fixed rule, no visual selection: the accepted crash with the largest
    # normalized inversion amplitude a_norm; present crashes first, and an
    # uncertain crash only when the shot has no present one.
    for wanted in ("present", "uncertain"):
        pool = [
            r
            for r in crashes
            if crash_state(r) == wanted and r.get("attrs", {}).get("a_norm") is not None
        ]
        if pool:
            best = max(pool, key=lambda r: (r["attrs"]["a_norm"], -r["time_s"]))
            return best, "Detected crash"
    if crashes:
        return crashes[len(crashes) // 2], "Detected crash"
    core_channels, _ = proxy_groups(record)
    core = mean_trace(y[core_channels])
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
        core_channels, _ = proxy_groups(record)
        usable = np.where(
            np.isfinite(before[core_channels]), before[core_channels], -np.inf
        )
        central = int(core_channels[np.argmax(usable)])
    gain_start = crash["attrs"].get("rise_start")
    gain_stop = crash["attrs"].get("rise_stop")
    gain_channels = (
        list(range(gain_start, gain_stop))
        if gain_start is not None and gain_stop is not None
        else proxy_groups(record)[1]
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.3))
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.10, top=0.77, hspace=0.75)
    state_kind = (
        "Crash candidate state" if selection == "Detected crash" else "Bin state"
    )
    fig.suptitle(
        f"DIII-D {shot}: {selection.lower()} at {time * 1000:.1f} ms\n"
        f"{state_kind}: {state}",
        y=0.99,
        fontsize=8,
    )
    handles = [
        Line2D(
            [], [], color=BLUE, lw=1, label=group_label(record, [central], "Central")
        )
    ]
    if gain_channels:
        handles.append(
            Line2D(
                [],
                [],
                color=VERMILION,
                lw=1,
                label=group_label(record, gain_channels, "Gain block"),
            )
        )
    handles += state_legend(record, window=(time - 0.03, time + 0.03))
    if selection != "Detected crash":
        handles += [Line2D([], [], color="0.25", lw=0.8, label="Window centre")]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        frameon=False,
        ncol=2,
        labelspacing=0.15,
        columnspacing=0.6,
    )
    axes[0].plot(t[near] * 1000, y[central, near], color=BLUE, lw=0.8)
    axes[0].plot(
        t[near] * 1000,
        mean_trace(y[gain_channels])[near],
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
        loc="center",
        bbox_to_anchor=(0.72, 0.40),
        bbox_transform=fig.transFigure,
        frameon=False,
        ncol=1,
    )
    axes[1].set_xlim(-0.5, 47.5)
    axes[1].set_xlabel("ECE channel")
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
        "outer_trace_channels": gain_channels,
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


def failed_support_panels(shot, record, destination):
    """Retain sampled shots without inventing a usable core or crash label."""
    import h5py

    from labeler.sawtooth.preprocessing import sample_native

    source = Paths.from_env().corpus_file(shot)
    with h5py.File(source, "r", locking=False) as file:
        t, y = sample_native(file["ece"])
    lo, hi = record["window_s"]
    keep = (t >= lo) & (t <= hi)
    t, y = t[keep], y[:, keep]
    trace = mean_trace(y[:40])
    stride = max(1, len(t) // 4500)
    error = record["error"].split(": ", 1)[-1]
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.3), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.10, top=0.82, hspace=0.25)
    fig.suptitle(f"DIII-D {shot}: sensor support diagnostic", y=0.98, fontsize=8)
    fig.text(0.5, 0.92, "Unassessed: usable core unavailable", ha="center", fontsize=7)
    axes[0].plot(t[::stride] * 1000, trace[::stride], color=BLUE, lw=0.6)
    axes[0].set_ylabel("Raw ECE mean (keV)")
    axes[0].text(
        0.02,
        0.95,
        "Channels 0–39; no spatial mask",
        transform=axes[0].transAxes,
        va="top",
        fontsize=7,
    )
    qmin = local_scalar(shot, "qmin", Paths.from_env())
    if qmin is not None:
        axes[1].plot(qmin[0] * 1000, qmin[1], color="0.25", lw=0.6)
    axes[1].set(
        xlabel="Time (ms)",
        ylabel="EFIT01 q-min",
        ylim=(0, 4),
        xlim=(lo * 1000, hi * 1000),
    )
    for axis in axes:
        axis.grid(axis="y", color="0.9", lw=0.4)
    figures = save_figure(fig, destination / f"shot_{shot}_overview")
    center = float((lo + hi) / 2)
    near = (t >= center - 0.03) & (t <= center + 0.03)
    before = (t >= center - 0.002) & (t < center - 0.0005)
    after = (t >= center + 0.0005) & (t < center + 0.002)
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.3))
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.10, top=0.80, hspace=0.65)
    fig.suptitle(f"DIII-D {shot}: sensor diagnostic; no label", y=0.98, fontsize=8)
    fig.text(0.5, 0.92, "Unassessed: usable core unavailable", ha="center", fontsize=7)
    axes[0].plot(t[near] * 1000, trace[near], color=BLUE, lw=0.8)
    axes[0].set(xlabel="Time (ms)", ylabel="Raw ECE mean (keV)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pre, post = (
            np.nanmedian(y[:, before], axis=1),
            np.nanmedian(y[:, after], axis=1),
        )
    axes[1].plot(
        np.arange(len(y)), pre, color=BLUE, lw=0.8, label="Before (−2 to −0.5 ms)"
    )
    axes[1].plot(
        np.arange(len(y)), post, color=VERMILION, lw=0.8, label="After (+0.5 to +2 ms)"
    )
    axes[1].legend(
        loc="center",
        bbox_to_anchor=(0.72, 0.40),
        bbox_transform=fig.transFigure,
        frameon=False,
    )
    axes[1].set(xlabel="ECE channel", ylabel="Raw ECE Te (keV)", xlim=(-0.5, 47.5))
    for axis in axes:
        axis.grid(axis="y", color="0.9", lw=0.4)
    figures.extend(save_figure(fig, destination / f"shot_{shot}_crash"))
    return figures, {
        "shot": shot,
        "selection": "Sensor diagnostic; no label",
        "state": "unassessed",
        "bin_state": "unassessed",
        "center_s": center,
        "window_ms": 60,
        "reader_or_support_error": error,
        "raw_signal_source": str(source),
        "raw_signal_processing": "Native ECE FIR antialias to 10kHz; no spatial mask",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument(
        "--output", type=Path, default=REPO / "outputs/labeler/sawtooth/fix4"
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPO / "outputs/labeler/sawtooth/fix4/gallery.json",
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
        if any(
            hashlib.sha256(Path(source).read_bytes()).hexdigest() != value
            for source, value in record["source_sha256"].items()
        ):
            raise ValueError("gallery inputs changed; regenerate and inspect again")
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
    shots = previous.get("shots", ORIGINAL_GALLERY_SHOTS)
    if shots != ORIGINAL_GALLERY_SHOTS or not set(shots) <= train:
        raise ValueError(
            "retain the original 12 nonexpert train shots without resampling"
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
    figures, windows, sources = [], [], []
    for shot in shots:
        record = json.loads((args.work / "shots" / f"{shot}.json").read_text())
        sources.append(args.work / "shots" / f"{shot}.json")
        if "error" in record:
            panels, window = failed_support_panels(shot, record, destination)
            figures.extend(panels)
            windows.append(window)
            sources.append(Paths.from_env().corpus_file(shot))
            continue
        sources.append(args.work / "signals" / f"{shot}.npz")
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
                "Original uniform random usable nonexpert train-shot sample retained; "
                "current support failures remain represented as unassessed"
            ),
            "figures": figures,
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources
            },
            "crash_windows": windows,
            "crash_selection_rule": (
                "per shot, the accepted crash with the largest normalized inversion "
                "amplitude a_norm (present crashes first; an uncertain crash only "
                "when the shot has no present one); ties go to the earlier crash"
            ),
            "geometry_note": (
                "Nominal geometric rho=abs(R-axis)/(LCFS_outer_R-axis), not "
                "calibrated flux. Outer proxy is on the low-field side."
            ),
            "q_note": (
                "EFIT01 q-min is magnetics-only. Soft conflict and the q-prior "
                "state follow the frozen rule; display range 0–4."
            ),
            "marker_note": (
                "Crash markers show diagnostic candidate state; shaded spans show "
                "canonical bin state. Crash points and sustained-presence spans "
                "have different temporal support. Purple spans are q-prior only: "
                "EFIT01 q_min >= 1.5 with no ECE absence test, exported as "
                "uncertain."
            ),
            "png_inspection": "pending",
        },
    )


if __name__ == "__main__":
    main()
