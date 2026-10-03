#!/usr/bin/env python
"""Appendix camera/inversion views and a separate indicator timeline.

    python scripts/labeler/detach_figure.py --shot 189057 \\
        --out-dir $LABELER_ROOT/round4/detach/figure

Both figures are 6.75 inches wide, vector PDF plus 150-dpi PNG. Three columns
show attached/detached fronts and the highest front; headers give the actual
observed label. Inversions include EFIT flux surfaces and the g-file LIM wall.
The bolometer row is omitted because no usable chord geometry was obtained.
The recorded IRTV heat-flux attempt returned NODATA. Every number comes from
the exported labels and parked signals; omissions are recorded in figure.json.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

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
    """One bin centre (ms) per state: the most confident bin with a TangTV frame.

    State 3 falls back to the deepest detached bin (largest DZ) when the shot has
    no MARFE bin, so the third column is always populated.
    """
    group = frame[(frame.shot == shot) & frame.tangtv_valid].copy()
    out: dict[int, float] = {}
    for state in (1, 2, 3):
        rows = group[group.state_lm == state]
        if not len(rows):
            rows = group[group.tangtv_vote == state]
        if len(rows):
            best = rows.sort_values("confidence", ascending=False).iloc[0]
            out[state] = float(best.start_ms) + core.BIN_MS / 2
    if 3 not in out:
        rows = group
        if len(rows):
            best = rows.sort_values("tangtv_value", ascending=False).iloc[0]
            out[3] = float(best.start_ms) + core.BIN_MS / 2
    return out


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
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=4, fontsize=8.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)


def timeline(fig, spec, group: pd.DataFrame, times: dict[int, float]) -> None:
    sub = spec.subgridspec(7, 1, height_ratios=[0.5, 0.5, 0.5, 0.5, 1.4, 1.4, 1.4])
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
    p_in = group.aux_p_in_w.to_numpy(float)
    zx = group.aux_zxpt1.to_numpy(float)
    zs = group.aux_zvsod.to_numpy(float)
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
            "Prad,div\n(MW)",
            np.where(group.prad_valid, group.prad_value * p_in / 1e6, np.nan),
            (),
        ),
        (
            "ZE (m)",
            np.where(group.tangtv_valid, zx - (1.0 - dz) * (zx - zs), np.nan),
            (),
        ),
    )
    for ax, (name, y, lines) in zip(axes[4:], series, strict=True):
        ax.plot(centre, y, color=INK, lw=0.9, marker=".", ms=2.5)
        for level, dash in lines:
            ax.axhline(level, color="#666666", lw=0.6, ls=dash)
        ax.set_ylabel(name)
        if name == "Jsat ratio":
            ax.set_yscale("log")
    ref = group.aux_zvsod.to_numpy(float)
    ok = np.isfinite(ref) & group.tangtv_valid.to_numpy()
    if ok.any():
        axes[6].plot(
            centre[ok], ref[ok], color="#666666", lw=0.7, ls="--", label="strike point"
        )
        axes[6].plot(
            centre[ok], zx[ok], color="#666666", lw=0.7, ls=":", label="X-point"
        )
        axes[6].legend(loc="lower left", frameon=False)
    lo, hi = start.min(), start.max() + core.BIN_MS
    for ax in axes:
        ax.set_xlim(lo, hi)
        for t in times.values():
            observed = int(group.iloc[np.argmin(np.abs(centre - t))].state_lm)
            ax.axvline(t, color=STATE_COLOUR[observed], lw=1.0, alpha=0.9)
    for ax in axes[4:6]:
        ax.tick_params(axis="x", labelbottom=False)
    for ax in axes:
        ax.yaxis.set_label_coords(-0.075, 0.5)
    axes[6].set_xlabel("time (ms)")


def summary(ax, frame: pd.DataFrame, shot: int) -> None:
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
        ax.axvline(list(share.index).index(shot), color=INK, lw=0.7)
    ax.set_xlim(-0.5, len(share) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"{len(share)} labelled shots, ordered by state mix")
    ax.set_ylabel("share of bins")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.42), frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shot", type=int, default=189057)
    parser.add_argument("--times", type=float, nargs=3, default=None)
    parser.add_argument("--out-dir", default=str(root() / "figure"))
    args = parser.parse_args()
    style()
    frame = load_labels()
    group = frame[frame.shot == args.shot].sort_values("start_ms")
    if group.empty:
        raise SystemExit(f"shot {args.shot} is not in labels_bins.csv.gz")
    times = pick_times(frame, args.shot)
    if args.times:
        times = dict(zip((1, 2, 3), args.times, strict=True))
    inv = np.load(root() / "inversions" / f"{args.shot}.npz")
    video = read_video(args.shot)
    fig = plt.figure(figsize=(6.75, 4.2), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.8, 1.7],
        left=0.09,
        right=0.99,
        top=0.81,
        bottom=0.13,
        hspace=0.35,
        wspace=0.34,
    )
    notes = []
    for col, state in enumerate((1, 2, 3)):
        if state not in times:
            continue
        t = times[state]
        label = STATE_NAME[state]
        selected = group.iloc[int(np.argmin(np.abs(group.start_ms + 25 - t)))]
        if int(selected.state_lm) != state:
            label = "highest front" if state == 3 else STATE_NAME[state] + " front"
            label += "\nlabel: " + STATE_NAME[int(selected.state_lm)]
        # raw frame
        ax = fig.add_subplot(outer[0, col])
        if video is not None:
            vt, vid = video
            k = int(np.argmin(np.abs(vt - t)))
            top = float(np.percentile(vid[k], 99.5))
            ax.imshow(vid[k], cmap="gray", vmin=0, vmax=max(top, 1.0), aspect="auto")
            ax.set_title(f"{label}, t = {t:.0f} ms\nraw frame ({vt[k]:.0f} ms)")
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
            ax.axhline(ze, color="#F0E442", lw=0.8, ls="--")
        ax.set_xlim(max(radii[0], 1.0), min(radii[-1], 1.8))
        ax.set_ylim(elev[0], VIEW_ZMAX)
        ax.set_xlabel("R (m)")
        if col == 0:
            ax.set_ylabel("Z (m)")
        ax.set_title(f"inversion ({inv['times_ms'][j]:.0f} ms)")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_detachment_views.pdf")
    fig.savefig(out / "fig_detachment_views.png", dpi=150)
    plt.close(fig)
    fig = plt.figure(figsize=(6.75, 5.4))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[3.5, 1],
        left=0.18,
        right=0.985,
        top=0.98,
        bottom=0.13,
        hspace=0.55,
    )
    timeline(fig, grid[0], group, times)
    summary(fig.add_subplot(grid[1]), frame, args.shot)
    fig.savefig(out / "fig_detachment_timeline.pdf")
    fig.savefig(out / "fig_detachment_timeline.png", dpi=150)
    plt.close(fig)
    import json

    (out / "figure.json").write_text(
        json.dumps(
            {
                "shot": args.shot,
                "chosen_times_ms": times,
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
                    3: "highest front; not a MARFE claim",
                },
                "width_in": 6.75,
                "min_font_pt": 7,
                "bolometer_row": "omitted: no chord endpoints/calibration in BOLOM node survey or local plasma_tv resources",
                "irtv_row": "omitted: IRTV HEATFLUX node on 189057 has no data; corpus is a stub",
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
