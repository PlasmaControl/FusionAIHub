#!/usr/bin/env python
"""Appendix camera/inversion views and a separate indicator timeline.

    python scripts/labeler/detach_figure.py --shot 189057 \\
        --out-dir $LABELER_ROOT/round4/detach/figure

Both figures require full-textwidth placement at 6.75 inches (minimum 7 pt),
with vector PDF plus 150-dpi PNG. Three columns
show attached/detached fronts and the highest front; headers give the actual
observed label. Inversions include EFIT flux surfaces and the g-file LIM wall.
The bolometer row shows parked raw-voltage chord profiles; no spatial inversion
or fabricated chord geometry is implied.
The recorded IRTV heat-flux attempt returned NODATA. Every number comes from
the exported labels and parked signals; omissions are recorded in figure.json.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from detach_json import dumps

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from labeler.events.detachment import core, thresholds

#: Okabe-Ito: attached blue, detached orange, MARFE reddish purple, uncertain grey.
STATE_COLOUR = {1: "#0072B2", 2: "#E69F00", 3: "#CC79A7", 4: "#999999"}
STATE_NAME = {1: "attached", 2: "detached", 3: "MARFE", 4: "uncertain"}
INK = "#222222"
#: Top of the plotted inversion window (m): the divertor view, not the whole frame.
VIEW_ZMAX = -0.85


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def load_labels() -> pd.DataFrame:
    return pd.read_csv(root() / "labels_bins.csv.gz")


def pick_times(frame: pd.DataFrame, shot: int) -> dict[int, float]:
    """One bin centre per observed state, chosen near its median time.

    State 3 falls back to the deepest detached bin (largest DZ) when the shot has
    no MARFE bin, so the third column is always populated.
    """
    group = frame[
        (frame.shot == shot)
        & frame.tangtv_valid
        & (frame.aux_rvsod >= thresholds.SHELF_WALL_R)
        & (abs(frame.aux_zvsod - thresholds.SHELF_Z) <= thresholds.SHELF_Z_TOL)
    ].copy()
    out: dict[int, float] = {}
    for state in (1, 2, 3):
        rows = group[group.state_lm == state]
        if not len(rows):
            rows = group[group.tangtv_vote == state]
        if len(rows):
            centre = rows.start_ms.median()
            best = rows.iloc[np.argmin(abs(rows.start_ms - centre))]
            out[state] = float(best.start_ms) + core.BIN_MS / 2
    if 3 not in out:
        rows = group
        if len(rows):
            best = rows.sort_values("tangtv_value", ascending=False).iloc[0]
            out[3] = float(best.start_ms) + core.BIN_MS / 2
    return out


def choose_shot(frame: pd.DataFrame) -> int:
    """Prefer both certain states; otherwise show an honest front-vote example."""
    candidates, front_examples = [], []
    for shot, group in frame.groupby("shot"):
        upper = group[
            group.tangtv_valid
            & (group.aux_rvsod >= thresholds.SHELF_WALL_R)
            & (abs(group.aux_zvsod - thresholds.SHELF_Z) <= thresholds.SHELF_Z_TOL)
        ]
        states = set(upper.state_lm)
        if not (root() / "inversions" / f"{shot}.npz").is_file():
            continue
        power = upper.aux_p_in_w.median()
        if {1, 2}.issubset(states):
            candidates.append((3 in states, float(power), len(upper), int(shot)))
        counts = upper.tangtv_vote.value_counts()
        if {1, 2}.issubset(counts.index):
            support = int(upper.state_lm.isin((1, 2, 3)).sum())
            balance = min(int(counts[1]), int(counts[2]))
            front_examples.append((support * balance, balance, float(power), int(shot)))
    if candidates:
        return max(candidates)[-1]
    if front_examples:
        return max(front_examples)[-1]
    raise SystemExit("no upper-shelf shot has attached and detached front votes")


def bolo_profiles(shot: int, times: dict[int, float]):
    """Median raw chord voltage over 50 ms, relative to the pre-plasma baseline.

    The corpus modalities.yaml identifies BOL_L01_V..L24_V then U01_V..U24_V.
    These are voltages, not calibrated radiation or a bolometer inversion.
    """
    import h5py

    path = Path("/scratch/gpfs/EKOLEMEN/foundation_model") / f"{shot}_processed.h5"
    if not path.is_file():
        return {}, "no corpus file"
    out = {}
    with h5py.File(path, "r") as handle:
        if "bolo" not in handle:
            return {}, "no bolo group"
        group = handle["bolo"]
        t = group["xdata"][:] * 1000
        data = group["ydata"]
        if len(t) < 2 or data.shape[0] != 48:
            return {}, "bolo stub or unexpected channel count"
        stop = int(np.searchsorted(t, 0))
        if stop < 10:
            return {}, "no pre-plasma voltage baseline"
        baseline = np.nanmedian(data[:, :stop], axis=1)
        for role, time in times.items():
            lo, hi = np.searchsorted(t, [time - 25, time + 25])
            if hi > lo:
                out[role] = np.nanmedian(data[:, lo:hi], axis=1) - baseline
    return out, str(path)


def read_video(shot: int):
    """`(t_ms, frames)` of the raw TangTV video of a shot, from its plasma_tv .sav."""
    from scipy.io import readsav

    source = Path(str(np.load(root() / "inversions" / f"{shot}.npz")["source"]))
    for path in (source, source.with_name(source.stem + "_raw.sav")):
        if not path.is_file():
            continue
        record = readsav(str(path))["emission_structure"][0]
        if "VID" in record.dtype.names:
            return np.asarray(record["VID_TIMES"], float), np.asarray(record["VID"])
    return None


def efit_slice(shot: int, t_ms: float):
    """Normalised flux on the (R, Z) grid and the boundary at the nearest EFIT time."""
    path = root() / "efit" / f"{shot}.npz"
    if not path.is_file():
        return None
    with np.load(path) as f:
        k = int(np.argmin(np.abs(f["gtime_ms"] - t_ms)))
        if abs(f["gtime_ms"][k] - t_ms) > 40:
            return None
        psi, axis, edge = f["psirz"][k], float(f["ssimag"][k]), float(f["ssibry"][k])
        return {
            "r": f["r"],
            "z": f["z"],
            "psin": (psi - axis) / (edge - axis),
            "rb": f["rbbbs"][k],
            "zb": f["zbbbs"][k],
            "t_ms": float(f["gtime_ms"][k]),
            "source": str(f["source"]),
            "lim": f.get("lim", None),
        }


def style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def strip(ax, start_ms, state, label) -> None:
    """A colour strip of states over time (0 = nothing drawn)."""
    for value, colour in STATE_COLOUR.items():
        hit = state == value
        if hit.any():
            ax.broken_barh(
                [(s, core.BIN_MS) for s in start_ms[hit]],
                (0, 1),
                facecolors=colour,
                linewidth=0,
            )
    ax.set_ylim(0, 1)
    ax.set_facecolor("#F3F3F3")
    if not np.isin(state, (1, 2, 3, 4)).any():
        ax.text(
            0.5,
            0.5,
            "no valid vote",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7,
        )
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=4, fontsize=8.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)


def timeline(fig, spec, group: pd.DataFrame, times: dict[int, float]) -> None:
    sub = spec.subgridspec(
        7, 1, height_ratios=[0.35, 0.35, 0.35, 0.35, 1.4, 1.4, 1.4], hspace=0.28
    )
    axes = [fig.add_subplot(sub[i]) for i in range(7)]
    start = group.start_ms.to_numpy()
    votes = {
        "label": group.state_lm.to_numpy(),
        "Jsat ratio": np.where(group.afrac_valid, group.afrac_vote, 0),
        "Prad,div": np.where(group.prad_valid, group.prad_vote, 0),
        "TangTV": np.where(group.tangtv_valid, group.tangtv_vote, 0),
    }
    for ax, (name, state) in zip(axes[:4], votes.items(), strict=True):
        strip(ax, start, state, name)
    centre = start + core.BIN_MS / 2
    dz = group.tangtv_value.to_numpy(float)
    series = (
        (
            "Jsat ratio",
            np.where(group.afrac_valid, group.afrac_value, np.nan),
            (
                (thresholds.AFRAC_ATTACHED_MIN, "--"),
                (thresholds.AFRAC_DETACHED_MAX, ":"),
            ),
        ),
        (
            r"$f_{\mathrm{div}}$",
            np.where(group.prad_valid, group.prad_value, np.nan),
            (
                (thresholds.PRAD_ATTACHED_MAX, "--"),
                (thresholds.PRAD_DETACHED_MIN, ":"),
            ),
        ),
        (
            "DZ",
            np.where(group.tangtv_valid, dz, np.nan),
            (
                (thresholds.DZ_ATTACHED_MAX, "--"),
                (thresholds.DZ_DETACHED_MIN, ":"),
                (thresholds.DZ_MARFE_MIN, "-."),
            ),
        ),
    )
    for ax, (name, y, lines) in zip(axes[4:], series, strict=True):
        ax.plot(centre, y, color=INK, lw=0.9, marker=".", ms=2.5)
        for level, dash in lines:
            ax.axhline(level, color="#666666", lw=0.6, ls=dash)
        ax.set_ylabel(name)
        if name == "Jsat ratio":
            ax.set_yscale("log")
        if not np.isfinite(y).any():
            ax.text(
                0.5, 0.55, "no valid measurement", transform=ax.transAxes, ha="center"
            )
        ax.grid(axis="y", color="0.9", lw=0.4)
    axes[5].set_ylim(bottom=0)
    axes[6].set_ylim(bottom=min(0, np.nanmin(dz) if np.isfinite(dz).any() else 0))
    lo, hi = start.min(), start.max() + core.BIN_MS
    for ax in axes:
        ax.set_xlim(lo, hi)
        for t in times.values():
            observed = int(group.iloc[np.argmin(np.abs(centre - t))].state_lm)
            ax.axvline(t, color=STATE_COLOUR[observed], lw=1.0, alpha=0.9)
    for letter, time in zip("abc", times.values(), strict=False):
        axes[0].text(time, 1.2, letter, ha="center", va="bottom", fontsize=8)
    for ax in axes[4:6]:
        ax.tick_params(axis="x", labelbottom=False)
    for ax in axes:
        ax.yaxis.set_label_coords(-0.11, 0.5)
    axes[6].set_xlabel("time (ms)")


def summary(ax, frame: pd.DataFrame, shot: int) -> dict:
    """Per-shot fractions of assessed bins by state, one column per shot."""
    assessed = frame[frame.assessed & frame.state_lm.isin((1, 2, 3, 4))]
    table = (
        assessed.groupby(["shot", "state_lm"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[1, 2, 3, 4], fill_value=0)
    )
    table = table[table.sum(axis=1) >= 10]
    share = table.div(table.sum(axis=1), axis=0)
    order = share.sort_values([1, 2, 3, 4], ascending=False).index
    share = share.loc[order]
    x = np.arange(len(share))
    bottom = np.zeros(len(share))
    for state in (1, 2, 3, 4):
        ax.bar(
            x,
            share[state].to_numpy(),
            bottom=bottom,
            width=1.0,
            color=STATE_COLOUR[state],
            linewidth=0,
            label=STATE_NAME[state],
        )
        bottom += share[state].to_numpy()
    if shot in share.index:
        ax.axvline(
            list(share.index).index(shot), color=INK, lw=0.9, label="example shot"
        )
    ax.set_xlim(-0.5, len(share) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(
        f"{len(share)} assessed shots (at least 10 bins), ordered by state mix"
    )
    ax.set_ylabel("share of bins")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#F3F3F3", edgecolor="#DDDDDD"))
    labels.append("unassessed / no vote")
    ax.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.65),
        frameon=False,
    )
    return {
        "population": "Assessed shots with at least 10 assessed bins each",
        "n_shots": len(share),
        "ordered_shots": [int(value) for value in share.index],
        "state_bin_counts_by_shot": {
            str(key): {
                STATE_NAME[state]: int(table.loc[key, state]) for state in (1, 2, 3, 4)
            }
            for key in share.index
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--times", type=float, nargs=3, default=None)
    parser.add_argument("--out-dir", default=str(root() / "figure"))
    args = parser.parse_args()
    style()
    frame = load_labels()
    if args.shot is None:
        args.shot = choose_shot(frame)
    group = frame[frame.shot == args.shot].sort_values("start_ms")
    if group.empty:
        raise SystemExit(f"shot {args.shot} is not in labels_bins.csv.gz")
    upper = group[
        group.tangtv_valid
        & (group.aux_rvsod >= thresholds.SHELF_WALL_R)
        & (abs(group.aux_zvsod - thresholds.SHELF_Z) <= thresholds.SHELF_Z_TOL)
    ]
    upper_counts = {
        "n_valid_measurement_bins": len(upper),
        "vote_bins": {
            STATE_NAME[state]: int((upper.tangtv_vote == state).sum())
            for state in (1, 2, 3)
        },
        "certain_state_bins": {
            STATE_NAME[state]: int((upper.state_lm == state).sum())
            for state in (1, 2, 3)
        },
    }
    times = pick_times(frame, args.shot)
    if args.times:
        times = dict(zip((1, 2, 3), args.times, strict=True))
    if len(times) != 3:
        raise SystemExit("the selected shot needs an upper-shelf frame for all columns")
    inv = np.load(root() / "inversions" / f"{args.shot}.npz")
    video = read_video(args.shot)
    profiles, bolo_source = bolo_profiles(args.shot, times)
    fig = plt.figure(figsize=(6.75, 5.05), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        3,
        height_ratios=[1, 1.15, 0.9],
        left=0.095,
        right=0.965,
        top=0.925,
        bottom=0.135,
        hspace=0.4,
        wspace=0.42,
    )
    notes = []
    selections = {}
    profile_values = (
        np.concatenate(list(profiles.values())) if profiles else np.array([])
    )
    ylim = (
        (float(np.nanmin(profile_values)), float(np.nanmax(profile_values)))
        if len(profile_values) and np.isfinite(profile_values).any()
        else (-1, 1)
    )
    padding = max((ylim[1] - ylim[0]) * 0.08, 0.005)
    for col, state in enumerate((1, 2, 3)):
        if state not in times:
            continue
        t = times[state]
        label = STATE_NAME[state]
        selected = group.iloc[int(np.argmin(np.abs(group.start_ms + 25 - t)))]
        if not bool(selected.tangtv_valid) or not (
            selected.aux_rvsod >= thresholds.SHELF_WALL_R
            and abs(selected.aux_zvsod - thresholds.SHELF_Z) <= thresholds.SHELF_Z_TOL
        ):
            raise SystemExit(f"selected time {t} ms is not a valid upper-shelf frame")
        if int(selected.state_lm) != state:
            label = "highest front" if state == 3 else STATE_NAME[state] + " front"
            label += "\nlabel: " + STATE_NAME[int(selected.state_lm)]
        # raw frame
        ax = fig.add_subplot(outer[0, col])
        ax.set_title(f"({chr(97 + col)}) {label}, {t:.0f} ms")
        if video is not None:
            vt, vid = video
            k = int(np.argmin(np.abs(vt - t)))
            top = float(np.percentile(vid[k], 99.5))
            ax.imshow(vid[k], cmap="gray", vmin=0, vmax=max(top, 1.0), aspect="auto")
            if col == 0:
                ax.set_ylabel("raw TangTV")
        else:
            ax.text(
                0.5, 0.5, "raw frame unavailable", transform=ax.transAxes, ha="center"
            )
        ax.set_xticks([])
        ax.set_yticks([])
        # inversion with EFIT
        ax = fig.add_subplot(outer[1, col])
        j = int(np.argmin(np.abs(inv["times_ms"] - t)))
        radii, elev = inv["radii"], inv["elevation"]
        shown = inv["frames"][j].astype(float)
        ax.imshow(
            shown,
            origin="lower",
            extent=(radii[0], radii[-1], elev[0], elev[-1]),
            cmap="magma",
            vmin=0,
            vmax=float(np.percentile(shown, 99.8)),
            aspect="equal",
        )
        efit = efit_slice(args.shot, t)
        row = group.iloc[int(np.argmin(np.abs(group.start_ms + 25 - t)))]
        if efit is not None:
            ax.contour(
                efit["r"],
                efit["z"],
                efit["psin"],
                levels=[1.0],
                colors="#56B4E9",
                linewidths=1.0,
            )
            ax.contour(
                efit["r"],
                efit["z"],
                efit["psin"],
                levels=[0.98, 1.02, 1.05, 1.1],
                colors="#56B4E9",
                linewidths=0.4,
            )
            if efit["lim"] is not None:
                lim = efit["lim"]
                ax.plot(lim[:, 0], lim[:, 1], color="#56B4E9", lw=1.0)
        else:
            notes.append("EFIT flux map not parked")
        if np.isfinite(row.aux_rxpt1):
            ax.plot(row.aux_rxpt1, row.aux_zxpt1, "x", color="white", ms=4)
        if np.isfinite(row.aux_rvsod):
            ax.plot(row.aux_rvsod, row.aux_zvsod, "o", mfc="none", mec="white", ms=4)
        if np.isfinite(row.tangtv_value) and np.isfinite(row.aux_zxpt1):
            ze = row.aux_zxpt1 - (1 - row.tangtv_value) * (
                row.aux_zxpt1 - row.aux_zvsod
            )
            ax.axhline(ze, color="#E69F00", lw=0.8, ls="--")
        ax.set_xlim(max(radii[0], 1.0), min(radii[-1], 1.8))
        ax.set_ylim(elev[0], VIEW_ZMAX)
        ax.set_xticks([1.0, 1.2, 1.4, 1.6])
        ax.set_xlabel("R (m)")
        if col == 0:
            ax.set_ylabel("Z (m)")
        ax.set_title(f"C-III inversion ({inv['times_ms'][j]:.0f} ms)")
        selections[state] = {
            "bin_centre_ms": t,
            "bin_start_ms": float(selected.start_ms),
            "state": STATE_NAME[int(selected.state_lm)],
            "tangtv_dz": float(selected.tangtv_value),
            "prad_div_fraction": float(selected.prad_value),
            "strike_r_m": float(selected.aux_rvsod),
            "strike_z_m": float(selected.aux_zvsod),
            "inversion_time_ms": float(inv["times_ms"][j]),
            "raw_time_ms": float(vt[k]) if video is not None else None,
            "efit_time_ms": efit["t_ms"] if efit is not None else None,
            "efit_source": efit["source"] if efit is not None else None,
        }
        # Chord index is the only honest coordinate without endpoint calibration.
        ax = fig.add_subplot(outer[2, col])
        if state in profiles:
            lower, upper = profiles[state][:24], profiles[state][24:]
            ax.plot(np.arange(1, 25), lower, color="#0072B2", lw=0.8, label="lower fan")
            ax.plot(np.arange(1, 25), upper, color="#D55E00", lw=0.8, label="upper fan")
            ax.set_ylim(ylim[0] - padding, ylim[1] + padding)
            selections[state]["bolo_delta_v_by_channel"] = profiles[state].tolist()
        else:
            ax.text(0.5, 0.5, "no usable chords", transform=ax.transAxes, ha="center")
            notes.append("bolo profile unavailable: " + bolo_source)
        ax.set_xlim(1, 24)
        ax.set_xticks([1, 8, 16, 24])
        ax.set_xlabel("chord index within fan")
        if col == 0:
            ax.set_ylabel("bolometer\n" + r"$\Delta V$ (V)")
            ax.legend(loc="best", frameon=False, fontsize=7)
    fig.legend(
        [
            Line2D([], [], color="#56B4E9", lw=1),
            Line2D([], [], color="0.3", marker="x", ls="none", ms=4),
            Line2D([], [], color="0.3", marker="o", mfc="none", ls="none", ms=4),
            Line2D([], [], color="#E69F00", ls="--", lw=0.8),
        ],
        [r"EFIT $\psi_N$ and wall", "X-point", "outer strike point", "C-III height"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_detachment_views.pdf")
    fig.savefig(out / "fig_detachment_views.png", dpi=150)
    plt.close(fig)
    fig = plt.figure(figsize=(6.75, 5.2))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[4, 0.8],
        left=0.18,
        right=0.985,
        top=0.955,
        bottom=0.17,
        hspace=0.25,
    )
    timeline(fig, grid[0], group, times)
    summary_record = summary(fig.add_subplot(grid[1]), frame, args.shot)
    fig.savefig(out / "fig_detachment_timeline.pdf")
    fig.savefig(out / "fig_detachment_timeline.png", dpi=150)
    plt.close(fig)
    (out / "figure.json").write_text(
        dumps(
            {
                "shot": args.shot,
                "labels_source": str(root() / "labels_bins.csv.gz"),
                "labels_sha256": hashlib.sha256(
                    (root() / "labels_bins.csv.gz").read_bytes()
                ).hexdigest(),
                "chosen_times_ms": times,
                "timeline_markers": (
                    "Vertical lines a, b, c mark the selected times of the "
                    "corresponding camera, inversion and bolometer-profile columns."
                ),
                "example_upper_shelf_counts": upper_counts,
                "example_selection": (
                    "Prefer a shot with both certain attached and detached labels. "
                    "If unavailable, select an upper-shelf front-vote transition "
                    "with consensus support; every column displays its actual "
                    "exported label."
                ),
                "selection_values": selections,
                "multi_shot_summary": summary_record,
                "observed_states": {
                    k: STATE_NAME[
                        int(
                            group.iloc[
                                np.argmin(np.abs(group.start_ms + 25 - t))
                            ].state_lm
                        )
                    ]
                    for k, t in times.items()
                },
                "column_roles": {
                    1: "attached front",
                    2: "detached front",
                    3: "MARFE if confirmed, otherwise highest valid upper-shelf front",
                },
                "width_in": 6.75,
                "min_font_pt": 7,
                "placement": (
                    "Full textwidth at 6.75 inches. Do not shrink to a single "
                    "column; minimum 7 pt is guaranteed only at the declared size."
                ),
                "bolometer_row": (
                    "48 raw-voltage channels, 50 ms median minus per-channel "
                    "pre-plasma median; lower and upper fans plotted by chord "
                    "index, without invented spatial geometry or radiation "
                    "calibration."
                ),
                "bolometer_source": bolo_source,
                "bolometer_channel_order": (
                    "modalities.yaml: BOL_L01_V..L24_V, BOL_U01_V..U24_V"
                ),
                "timeline_values": (
                    "Jsat ratio, voted Prad,div/P_in with 0.36/0.50 thresholds, "
                    "and voted DZ with 0.35/0.50/1.20 thresholds. Pale strips are "
                    "unassessed or have no valid vote (invalid measurements or "
                    "abstention); darker grey means assessed uncertainty."
                ),
                "irtv_row": "omitted: IRTV HEATFLUX attempt on 189057 returned NODATA; no heat-flux record obtained for the selected shot",
                "raw_overlay": "no camera projection calibration available",
                "efit_source": "EFIT02, or explicitly recorded EFIT01 fallback",
                "notes": notes,
            },
            indent=1,
        )
    )
    for note in sorted(set(notes)):
        print("note:", note)
    print("chosen times (ms), front examples:", times)
    print("wrote", out / "fig_detachment_views.pdf")


if __name__ == "__main__":
    main()
