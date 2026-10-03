"""Export real camera frames and producer detachment timelines for the paper.

The 6.75-inch width is intended for two ICML columns without further scaling.
All text is at least 7 pt; only the camera pixels are rasterized in the PDF.
Inputs are read-only: primary labels come from labels_bins.csv.gz, quantitative
traces and vote validity from producer NPZ bins, and frames from an existing
review store (or the corpus when that store is absent). No browser is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import pairwise
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

from labeler.config import git_dirty, git_sha, sha256_of
from labeler.events.review import recipe

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text

REPO = Path(__file__).resolve().parents[2]
LABELER_ROOT = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
PRODUCER = Path("/scratch/gpfs/nc1514/FusionAIHub-r4-detach")
STATE_NAMES = {
    0: "unassessed",
    1: "attached",
    2: "detached",
    3: "MARFE",
    4: "uncertain",
}
COLORS = {0: "#ffffff", 1: "#0e8a8c", 2: "#0072b2", 3: "#cc79a7", 4: "#d55e00"}
INDICATORS = ("afrac", "prad", "tangtv")
WIDTH_IN = 6.75
HEIGHT_IN = 5.6
MIN_FONT_PT = 7.0


def file_record(path: Path, *, hash_file: bool = True) -> dict:
    """Record exact small sources; corpus provenance uses selected pixel hashes."""
    record = {"path": str(path.resolve()), "bytes": path.stat().st_size}
    if hash_file:
        record["sha256"] = sha256_of(path)
    return record


def read_inputs(bins_path: Path, labels_path: Path, cohort_path: Path, shot: int):
    cohort = pd.read_csv(cohort_path)
    cohort_shot = cohort[cohort.shot == shot]
    if not cohort_shot.empty and cohort_shot.split.iloc[0] not in ("train", "val"):
        raise ValueError("paper example cannot be a blind test cohort shot")
    with np.load(bins_path, allow_pickle=False) as data:
        bins = {key: data[key].copy() for key in data.files}
    starts = np.asarray(bins["start_ms"], dtype=float)
    if len(starts) < 2 or not np.all(np.diff(starts) == 50.0):
        raise ValueError("expected the producer's contiguous 50 ms bin grid")
    labels = pd.read_csv(labels_path, low_memory=False)
    labels = labels[labels.shot == shot].sort_values("start_ms")
    if labels.start_ms.duplicated().any():
        raise ValueError("duplicate producer labels for the example")
    if labels.split.astype(str).str.lower().eq("test").any() or (
        "holdout" in labels
        and labels.holdout.astype(str).str.lower().isin(["1", "true", "yes"]).any()
    ):
        raise ValueError("producer blind test shot cannot be a paper example")
    if (
        not labels.empty
        and not cohort_shot.empty
        and set(labels.split) != {str(cohort_shot.split.iloc[0])}
    ):
        raise ValueError("producer and cohort split disagree")
    assessed = np.sum([bins[f"{name}_valid"] for name in INDICATORS], axis=0) >= 2
    indexed = labels.set_index("start_ms")
    # The producer table contains assessed bins only. Its absent rows stay unknown.
    if not labels.empty and set(indexed.index) != set(starts[assessed]):
        raise ValueError("label rows disagree with the producer validity mask")
    for name in INDICATORS:
        if labels.empty:
            break
        for suffix in ("valid", "vote"):
            field = f"{name}_{suffix}"
            if not np.array_equal(
                indexed.loc[starts[assessed], field], bins[field][assessed]
            ):
                raise ValueError(
                    f"producer bins and full label table disagree: {field}"
                )
        field = f"{name}_value"
        if not np.allclose(
            indexed.loc[starts[assessed], field],
            bins[field][assessed],
            rtol=1e-5,
            atol=1e-7,
            equal_nan=True,
        ):
            raise ValueError(f"producer bins and full label table disagree: {field}")
    if not labels.empty and not np.array_equal(
        indexed.loc[starts[assessed], "tangtv_source"],
        bins["tangtv_source"][assessed],
    ):
        raise ValueError("producer bins and labels disagree on TangTV provenance")
    aligned = indexed.reindex(starts)
    bins["state_lm"] = aligned.state_lm.fillna(0).to_numpy(dtype=np.int8)
    bins["state_rule"] = aligned.state_rule.fillna(0).to_numpy(dtype=np.int8)
    bins["confidence"] = aligned.confidence.to_numpy(dtype=float)
    bins["assessed"] = assessed
    bins["label_available"] = not labels.empty
    if not set(bins["state_lm"]) <= set(STATE_NAMES):
        raise ValueError("producer has an unsupported label state")
    for name in INDICATORS:
        valid, vote = bins[f"{name}_valid"], bins[f"{name}_vote"]
        if np.any(~valid & (vote != -1)) or not set(vote) <= {-1, 1, 2, 3}:
            raise ValueError(f"invalid producer vote coding: {name}")
    return bins, str(
        cohort_shot.split.iloc[0]
    ) if not cohort_shot.empty else "producer_external"


def camera_clock(store_path: Path, corpus_path: Path, channel: int):
    if store_path.is_file():
        with h5py.File(store_path, "r") as source:
            key = f"videos/tangtv/{channel}"
            if key in source:
                group = source[key]
                return group["times_ms"][:], "review_store"
    with h5py.File(corpus_path, "r") as source:
        group = source["tangtv"]
        times = group["xdata"][:].astype(float) * 1000
        if len(times) < 2 or group["ydata"].ndim != 4:
            raise ValueError("example has no real TangTV movie")
        if channel < 0 or channel >= group["ydata"].shape[0]:
            raise ValueError("camera channel is out of range")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("camera timestamps must be finite and strictly increasing")
    return times, "corpus"


def choose_times(
    bins: dict, frame_times: np.ndarray, *, inversion_only=False
) -> list[float]:
    """One frame in each present certain state, from its longest camera-covered run."""
    starts, states = bins["start_ms"], bins["state_lm"]
    bin_index = np.searchsorted(starts, frame_times, side="right") - 1
    inside = (bin_index >= 0) & (frame_times < starts[-1] + 50)
    safe_index = np.clip(bin_index, 0, len(starts) - 1)
    chosen = []
    for state in (1, 2, 3, 4):
        runs = []
        eligible = states == state
        if inversion_only:
            eligible &= bins["tangtv_valid"] & (bins["tangtv_source"] == "inversion")
        boundaries = np.flatnonzero(np.diff(np.r_[False, eligible, False]))
        for lo, hi in zip(boundaries[::2], boundaries[1::2], strict=True):
            available = np.flatnonzero(inside & (safe_index >= lo) & (safe_index < hi))
            if len(available):
                midpoint = (starts[lo] + starts[hi - 1] + 50) / 2
                nearest = available[np.argmin(abs(frame_times[available] - midpoint))]
                runs.append((hi - lo, -lo, float(frame_times[nearest])))
        if runs:
            chosen.append(max(runs)[2])
    if not chosen and not inversion_only:
        covered = frame_times[inside]
        chosen = (
            [float(covered[i]) for i in np.linspace(0, len(covered) - 1, 3).astype(int)]
            if len(covered)
            else []
        )
    if not chosen:
        raise ValueError(
            "no camera frames coincide with the requested producer evidence"
        )
    return sorted(chosen[:3])


def inversion_example(args):
    """Audit queued train/val examples; never select from the blind test split."""
    roster = args.store_root.parents[1] / "tables/detachment/shots.csv"
    queued = set(pd.read_csv(roster).shot) if roster.is_file() else {args.shot}
    cohort = pd.read_csv(args.cohort)
    audit, candidates = [], []
    blind = set(cohort.loc[cohort.split.eq("test"), "shot"])
    for shot in sorted(queued - blind):
        path = args.producer_root / f"bins/{shot}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as source:
            count = int(
                np.sum(
                    source["tangtv_valid"] & (source["tangtv_source"] == "inversion")
                )
            )
        if not count:
            continue
        entry = {"shot": int(shot), "valid_inversion_bins": count}
        try:
            bins, _ = read_inputs(
                path, args.producer_root / "labels_bins.csv.gz", args.cohort, int(shot)
            )
            times, _ = camera_clock(
                args.store_root / f"{shot}.h5",
                args.corpus_root / f"{shot}_processed.h5",
                args.channel,
            )
            selected = choose_times(bins, times, inversion_only=True)
            entry["label_states_with_inversion_frames"] = len(selected)
            if len(selected) >= 2:
                candidates.append((len(selected), count, -int(shot), selected))
        except (ValueError, OSError, KeyError) as error:
            entry["excluded_reason"] = str(error)
        audit.append(entry)
    if candidates:
        _, _, negative_shot, times = max(candidates)
        return -negative_shot, times, audit
    return args.shot, None, audit


def read_frames(store_path, corpus_path, channel, times, requested, source_kind):
    indices = [int(np.argmin(abs(times - t))) for t in requested]
    frames, records = [], []
    with h5py.File(corpus_path, "r") as corpus:
        data = corpus["tangtv/ydata"]
        corpus_times = corpus["tangtv/xdata"][:].astype(float) * 1000
        if source_kind == "review_store":
            with h5py.File(store_path, "r") as store:
                group = store[f"videos/tangtv/{channel}"]
                lo, hi = (float(group.attrs[key]) for key in ("z_lo", "z_hi"))
                source_indices = group["source_indices"][:]
                for index in indices:
                    frames.append(group["frames"][index])
                native_indices = [int(source_indices[index]) for index in indices]
        else:
            native_indices = indices
            # One fixed scale across the discharge, using deterministic pixel samples.
            samples = []
            for index in range(len(times)):
                pixels = np.asarray(data[channel, index], dtype=float).ravel()[::256]
                samples.append(pixels[np.isfinite(pixels)])
            lo, hi = np.percentile(np.concatenate(samples), [1.0, 99.5])
            hi = max(hi, lo + 1)
            for index in indices:
                raw = np.asarray(data[channel, index], dtype=float)
                pixels = (np.nan_to_num(raw, nan=lo) - lo) * 255 / (hi - lo)
                frames.append(np.clip(np.rint(pixels), 0, 255).astype(np.uint8))
        for index, native_index, frame, target in zip(
            indices, native_indices, frames, requested, strict=True
        ):
            raw = np.asarray(data[channel, native_index], dtype=np.float32)
            if abs(corpus_times[native_index] - times[index]) > 1e-6:
                raise ValueError("preview and corpus timestamps disagree")
            expected = np.clip(
                np.rint(
                    (
                        np.nan_to_num(raw, nan=lo, posinf=hi, neginf=lo).astype(float)
                        - lo
                    )
                    * (255 / (hi - lo))
                ),
                0,
                255,
            ).astype(np.uint8)
            if not np.array_equal(frame, expected):
                raise ValueError(
                    "preview pixels disagree with the recorded corpus scale"
                )
            records.append(
                {
                    "requested_ms": float(target),
                    "displayed_ms": float(times[index]),
                    "preview_index": index if source_kind == "review_store" else None,
                    "corpus_index": native_index,
                    "raw_pixel_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                    "display_pixel_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                    "pixel_match_to_corpus": True,
                }
            )
    return frames, records, [float(lo), float(hi)]


def stripe(ax, starts, codes, y, *, valid=None):
    for start, code, index in zip(starts, codes, range(len(starts)), strict=True):
        invalid = valid is not None and not valid[index]
        abstain = valid is not None and valid[index] and code == -1
        ax.add_patch(
            Rectangle(
                (start, y - 0.34),
                50,
                0.68,
                facecolor="#d3d8d5"
                if invalid or abstain
                else COLORS.get(int(code), "white"),
                edgecolor="#69736e" if invalid else "none",
                linewidth=0.25 if invalid else 0,
                hatch="///" if invalid else None,
            )
        )


def plot_figure(bins, frames, frame_records, shot, channel, out, interpretation):
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "hatch.linewidth": 0.35,
        }
    )
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), facecolor="white")
    starts = bins["start_ms"]
    letters = list("ABC"[: len(frames)])
    fig.text(
        0.04,
        0.986,
        (
            f"Shot {shot} · TangTV channel 2: lower divertor, PERP STANDARD"
            if channel == 2
            else f"Shot {shot} · TangTV channel {channel}; indicator uses channel 2"
        ),
        fontsize=8,
        va="top",
    )
    for i, (frame, record, letter) in enumerate(
        zip(frames, frame_records, letters, strict=True)
    ):
        frame_width = 0.91 / len(frames)
        ax = fig.add_axes([0.04 + i * frame_width, 0.815, frame_width - 0.015, 0.12])
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_axis_off()
        time = record["displayed_ms"]
        state = bins["state_lm"][np.searchsorted(starts, time, side="right") - 1]
        record["producer_state"] = int(state)
        ax.set_title(f"{letter}  {time:.1f} ms", fontsize=8, pad=3, loc="left")
        ax.text(
            0,
            -0.05,
            f"Producer label: {STATE_NAMES[int(state)] if bins['label_available'] else 'not published'}",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
        )
    fig.text(
        0.04,
        0.77,
        "Corpus frames: 50 Hz linear blends; same fixed grayscale scale throughout.",
        fontsize=7,
    )
    lane = fig.add_axes([0.20, 0.545, 0.76, 0.19])
    names = [
        "Producer label\n(unverified)",
        "Afrac vote",
        "f_div vote",
        "TangTV vote",
    ]
    stripe(lane, starts, bins["state_lm"], 3)
    for y, name in zip((2, 1, 0), INDICATORS, strict=True):
        stripe(lane, starts, bins[f"{name}_vote"], y, valid=bins[f"{name}_valid"])
    lane.set_yticks([3, 2, 1, 0], names)
    lane.set_ylim(-0.5, 3.6)
    lane.tick_params(axis="both", length=0, labelbottom=False)
    timeline_axes = [lane]
    trace_specs = [
        ("afrac", 0.365, "Afrac"),
        ("prad", 0.26, "f_div"),
        ("tangtv", 0.152, "TangTV DZ"),
    ]
    sources = set(bins["tangtv_source"][bins["tangtv_valid"]])
    source_text = {
        frozenset(): "TangTV DZ: no valid front-height bins",
        frozenset({"surrogate"}): "TangTV DZ: ridge model estimate from raw frames",
        frozenset({"inversion"}): "TangTV DZ: tomographic inversion",
    }.get(
        frozenset(sources),
        "TangTV DZ: inversion (solid) / ridge model estimate (dashed)",
    )
    for name, bottom, ylabel in trace_specs:
        ax = fig.add_axes([0.20, bottom, 0.76, 0.085], sharex=lane)
        value = np.where(bins[f"{name}_valid"], bins[f"{name}_value"], np.nan)
        edges = np.r_[starts, starts[-1] + 50]
        if name == "tangtv":
            for source, style in (("inversion", "-"), ("surrogate", "--")):
                y = np.where(bins["tangtv_source"] == source, value, np.nan)
                ax.stairs(
                    y,
                    edges,
                    baseline=None,
                    color="#222222",
                    linewidth=0.8,
                    linestyle=style,
                )
        else:
            ax.stairs(value, edges, baseline=None, color="#222222", linewidth=0.8)
        for threshold in recipe.guides(interpretation["record"], name):
            ax.axhline(threshold, color="#777777", linestyle=":", linewidth=0.6)
        if not np.isfinite(value).any():
            ax.set_yticks([])
            ax.text(
                0.5,
                0.5,
                "No valid bins",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="#69736e",
            )
        if name == "prad":
            above = np.isfinite(value) & (value > 1)
            ax.plot(
                starts[above] + 25, value[above], "o", color="#d55e00", markersize=3
            )
        ax.set_ylabel(ylabel, rotation=0, ha="right", va="center", labelpad=15)
        ax.yaxis.label.set_fontsize(7)
        ax.tick_params(axis="both", length=2, labelbottom=name == "tangtv")
        ax.spines[["top", "right"]].set_visible(False)
        timeline_axes.append(ax)
    lane.set_xlim(starts[0], starts[-1] + 50)
    for ax in timeline_axes:
        for record, letter in zip(frame_records, letters, strict=True):
            ax.axvline(record["displayed_ms"], color="#000000", linewidth=0.7)
            if ax is lane:
                ax.text(
                    record["displayed_ms"],
                    3.55,
                    letter,
                    ha="center",
                    fontsize=7,
                    va="bottom",
                )
    timeline_axes[-1].set_xlabel("Time (ms)", labelpad=2)
    legend = [
        Patch(facecolor=COLORS[state], label=STATE_NAMES[state])
        for state in (1, 2, 3, 4)
    ]
    legend.extend(
        [
            Patch(
                facecolor="#d3d8d5",
                label="valid abstention",
            ),
            Patch(
                facecolor="#d3d8d5",
                edgecolor="#69736e",
                hatch="///",
                label="invalid",
            ),
            Patch(facecolor="white", edgecolor="#cccccc", label="unassessed"),
        ]
    )
    legend_artist = fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.535),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handlelength=1.7,
        handleheight=0.85,
    )
    fig.text(
        0.04,
        0.078,
        "f_div = Prad,div / P_in; orange points >1: check heating-power denominator.",
        fontsize=7,
    )
    fig.text(0.04, 0.055, source_text + "; producer validity gates apply.", fontsize=7)
    fig.text(
        0.04,
        0.032,
        "Unverified producer labels; raw views do not visibly separate states; no expert verification.",
        fontsize=7,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_trace_gap = (
        (
            legend_artist.get_window_extent(renderer).y0
            - timeline_axes[1].get_tightbbox(renderer).y1
        )
        * 72
        / fig.dpi
    )
    if legend_trace_gap < 4:
        raise ValueError(
            "legend must be separated from the Afrac trace by at least 4 pt"
        )
    trace_gaps = [
        (upper.get_tightbbox(renderer).y0 - lower.get_tightbbox(renderer).y1)
        * 72
        / fig.dpi
        for upper, lower in pairwise(timeline_axes[1:])
    ]
    if min(trace_gaps) < 2:
        raise ValueError(
            f"trace labels must be separated by at least 2 pt; measured {trace_gaps}"
        )
    minimum_font = min(
        item.get_fontsize() for item in fig.findobj(Text) if item.get_text()
    )
    if minimum_font < MIN_FONT_PT:
        raise ValueError(f"figure font below paper minimum: {minimum_font}")
    base = out / f"detachment_review_{shot}_paper"
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return {
        "pdf": str(base.with_suffix(".pdf")),
        "png": str(base.with_suffix(".png")),
        "width_inches": WIDTH_IN,
        "height_inches": HEIGHT_IN,
        "intended_use": "ICML two-column width (6.75 inches), without further scaling",
        "minimum_font_pt": minimum_font,
        "legend_to_afrac_gap_pt": legend_trace_gap,
        "minimum_gap_between_traces_pt": min(trace_gaps),
        "state_colors": COLORS,
        "png_dpi": 150,
        "pdf_vector_elements": "all labels, traces, guides, vote/label strips and cursors",
        "pdf_raster_elements": "real corpus camera frames only",
        "tangtv_display": source_text,
        "caption": (
            "Real corpus views and producer suggestions, not expert state examples. "
            "The raw view does not visibly separate the states. "
            "f_div = Prad,div / P_in; orange points flag f_div > 1, a possible "
            "heating-power denominator problem. Grey votes abstain; hatching "
            "marks invalid votes; blank label bins are unassessed."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=190212)
    parser.add_argument("--out", type=Path, default=LABELER_ROOT / "round4/detach-ui")
    parser.add_argument(
        "--producer-root", type=Path, default=LABELER_ROOT / "round4/detach"
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=LABELER_ROOT / "round4/detach-ui/spectrograms/detachment",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("/scratch/gpfs/EKOLEMEN/foundation_model"),
    )
    parser.add_argument(
        "--cohort", type=Path, default=REPO / "data/events/catalog/cohort.csv"
    )
    parser.add_argument("--producer-worktree", type=Path, default=PRODUCER)
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--prefer-inversion", action="store_true")
    parser.add_argument(
        "--times-ms",
        type=float,
        nargs="+",
        help="one to three requested frame times; nearest real frames are shown",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        help="additional small JSON source record, for committing",
    )
    args = parser.parse_args()
    if args.times_ms is not None and not 1 <= len(args.times_ms) <= 3:
        parser.error("--times-ms takes one to three times")
    args.out.mkdir(parents=True, exist_ok=True)
    inversion_audit = []
    if args.prefer_inversion and args.times_ms is None:
        args.shot, args.times_ms, inversion_audit = inversion_example(args)
    producer_method = args.producer_worktree / "docs/labeler/detachment.md"
    bins_path = args.producer_root / "bins" / f"{args.shot}.npz"
    labels_path = args.producer_root / "labels_bins.csv.gz"
    interpretation = recipe.load(labels_path)
    store_path = args.store_root / f"{args.shot}.h5"
    corpus_path = args.corpus_root / f"{args.shot}_processed.h5"
    bins, split = read_inputs(bins_path, labels_path, args.cohort, args.shot)
    times, source_kind = camera_clock(store_path, corpus_path, args.channel)
    requested = args.times_ms or choose_times(bins, times)
    if not all(bins["start_ms"][0] <= t < bins["start_ms"][-1] + 50 for t in requested):
        raise ValueError("requested frame times must lie inside the producer bin grid")
    frames, frame_records, scale = read_frames(
        store_path, corpus_path, args.channel, times, requested, source_kind
    )
    if not all(
        bins["start_ms"][0] <= f["displayed_ms"] < bins["start_ms"][-1] + 50
        for f in frame_records
    ):
        raise ValueError("nearest camera frame is outside the producer bin grid")
    for frame in frame_records:
        index = (
            np.searchsorted(bins["start_ms"], frame["displayed_ms"], side="right") - 1
        )
        frame["bin_start_ms"] = float(bins["start_ms"][index])
        frame["confidence"] = (
            float(bins["confidence"][index]) if bins["assessed"][index] else None
        )
        frame["indicators"] = {}
        for name in INDICATORS:
            value = float(bins[f"{name}_value"][index])
            frame["indicators"][name] = {
                "value": value if np.isfinite(value) else None,
                "valid": bool(bins[f"{name}_valid"][index]),
                "reason": str(bins[f"{name}_reason"][index]),
                "vote": int(bins[f"{name}_vote"][index]),
            }
        frame["indicators"]["tangtv"]["source"] = str(bins["tangtv_source"][index])
    figure = plot_figure(
        bins, frames, frame_records, args.shot, args.channel, args.out, interpretation
    )
    script = Path(__file__).resolve()
    record = {
        "script": file_record(script),
        "git_sha": git_sha(full=True),
        "git_dirty": git_dirty(),
        "command": ["python", str(script.relative_to(REPO)), *os.sys.argv[1:]],
        "shot": args.shot,
        "inversion_example_audit": inversion_audit,
        "split": split,
        "sources": {
            "cohort": file_record(args.cohort),
            "bins": file_record(bins_path),
            "primary_labels": file_record(labels_path),
            "corpus": file_record(corpus_path, hash_file=False),
            "camera_store": file_record(store_path)
            if source_kind == "review_store"
            else None,
            "producer_thresholds": file_record(
                args.producer_worktree / "src/labeler/events/detachment/thresholds.py"
            ),
            "producer_handoff": file_record(args.producer_root / "HANDOFF.md"),
            "producer_method": file_record(producer_method),
        },
        "camera": {
            "source_kind": source_kind,
            "group": "tangtv",
            "channel": args.channel,
            "view": "LODIV_240RM1:PERP:STANDARD"
            if args.channel == 2
            else f"channel {args.channel}",
            "grayscale_limits": scale,
            "scale_percentiles": [1.0, 99.5],
            "sampling": "50 Hz corpus linear blends; review-store previews at most 20 fps",
            "selection": "explicit requested times or selected inversion states"
            if args.times_ms
            else "middle of longest camera-covered run of each present label state (up to three); unpublished shots use three coverage times",
            "frames": frame_records,
        },
        "definitions": {
            "label_states": STATE_NAMES,
            "recipe": interpretation,
            "unassessed": "absent row in primary label table; never assumed attached",
            "missing_indicator": "valid=False; vote=-1; reason from producer",
            "valid_abstention": "valid=True; vote=-1; producer reason/evidence gates apply",
            "prad": "f_div = Prad,div / P_in",
            "surrogate": "ridge model estimate of front height from raw camera frames, not a tomographic measurement",
        },
        "bin_width_ms": 50,
        "total_bins": len(bins["start_ms"]),
        "label_available": bins["label_available"],
        "assessed_bins": int(bins["assessed"].sum()),
        "f_div_above_one_bins": [
            {"start_ms": float(t), "end_ms": float(t + 50), "value": float(v)}
            for t, v, valid in zip(
                bins["start_ms"], bins["prad_value"], bins["prad_valid"], strict=True
            )
            if valid and v > 1
        ],
        "label_counts": {
            STATE_NAMES[k]: int(np.count_nonzero(bins["state_lm"] == k))
            for k in STATE_NAMES
        },
        "indicator_counts": {
            name: {
                "valid": int(bins[f"{name}_valid"].sum()),
                "missing": int((~bins[f"{name}_valid"]).sum()),
                "valid_abstentions": int(
                    np.sum(bins[f"{name}_valid"] & (bins[f"{name}_vote"] == -1))
                ),
                "votes": {
                    STATE_NAMES[k]: int(np.sum(bins[f"{name}_vote"] == k))
                    for k in (1, 2, 3)
                },
            }
            for name in INDICATORS
        },
        "figure": figure,
        "network_fetch": False,
        "source_writes": False,
        "verification": {
            "all_selected_frames_match_corpus": all(
                f["pixel_match_to_corpus"] for f in frame_records
            )
        },
    }
    for kind in ("pdf", "png"):
        record["figure"][f"{kind}_sha256"] = sha256_of(Path(figure[kind]))
    evidence = args.out / f"detachment_review_{args.shot}_paper.json"
    evidence.write_text(json.dumps(record, indent=2) + "\n")
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"evidence": str(evidence), "figure": figure}))


if __name__ == "__main__":
    main()
