#!/usr/bin/env python
"""Appendix views of the detachment states on one transition shot, an indicator
timeline and the MARFE witness.

    python scripts/labeler/detach_figure.py --shot 201081 \
        --out-dir $LABELER_ROOT/round4/detach/figure

The views figure shows the attached, detached and re-attached phases of one shot in
time order. Each column is a sustained interval (a run of at least `MIN_RUN_BINS`
consecutive 50 ms bins of one TangTV vote, the longest such run before, at and
after the detached phase), never a single bin, and is titled by that TangTV vote
with the exported label of its bins in brackets: the label can be `certain`,
`tangtv_only` (silver) or uncertain, and the column says which. The raw TangTV frame
and the C-III inversion with EFIT flux surfaces at the interval centre, the
front-height line at
the interval's median DZ, and under them the PRAD_DIVL and PRAD_TOT traces with the
three intervals marked. A 2D bolometer emissivity does not exist in the corpus or in
the BOLOM tree (`detachment_bolometer_availability.json`), so the radiation row is
the divertor and total radiated-power traces. The timeline adds the label and
the Afrac and TangTV votes, f_div (a corroborator, not a vote) and DZ with their
cutoffs and, where fetched, the divertor Thomson Te at the SOL chords. Both figures use the full 6.75 inch text width
(minimum 7 pt); vector PDF plus 150-dpi PNG. Every number comes from the exported
labels and parked signals; omissions are recorded in figure.json.
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
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from labeler.events.detachment import core, thresholds

#: Okabe-Ito: attached blue, detached orange, MARFE reddish purple, uncertain grey.
STATE_COLOUR = {
    1: "#0072B2",
    2: "#E69F00",
    3: "#CC79A7",
    4: "#999999",
    # silver (tangtv_only) tints of attached and detached
    11: "#8EC1E3",
    12: "#F4D58D",
}
STATE_NAME = {1: "attached", 2: "detached", 3: "MARFE", 4: "uncertain"}
SILVER_OFFSET = 10
INK = "#222222"
#: Top of the plotted inversion window (m): the divertor view, not the whole frame.
VIEW_ZMAX = -0.85
#: A sustained interval: consecutive bins of one state (250 ms is about 7 frames).
MIN_RUN_BINS = 5
#: Published Te-cliff times of the transition shot (Chen 2026, from the digest).
PUBLISHED_CLIFFS_MS = {201081: (2650.0, 4450.0)}


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def load_labels() -> pd.DataFrame:
    return pd.read_csv(root() / "labels_bins.csv.gz")


def upper_shelf(group: pd.DataFrame) -> pd.DataFrame:
    return group[
        group.tangtv_valid
        & (group.aux_rvsod >= thresholds.SHELF_WALL_R)
        & (abs(group.aux_zvsod - thresholds.SHELF_Z) <= thresholds.SHELF_Z_TOL)
    ]


def runs(group: pd.DataFrame, column: str, state: int) -> list[dict]:
    """Maximal runs of consecutive 50 ms bins whose `column` equals `state`."""
    group = group.sort_values("start_ms")
    hit = (group[column] == state).to_numpy()
    start = group.start_ms.to_numpy(float)
    out, k = [], 0
    while k < len(group):
        if not hit[k]:
            k += 1
            continue
        e = k
        while (
            e + 1 < len(group) and hit[e + 1] and start[e + 1] - start[e] == core.BIN_MS
        ):
            e += 1
        out.append(
            {
                "state": state,
                "start_ms": float(start[k]),
                "end_ms": float(start[e] + core.BIN_MS),
                "n_bins": e - k + 1,
                "basis": column,
            }
        )
        k = e + 1
    return [r for r in out if r["n_bins"] >= MIN_RUN_BINS]


def longest(candidates: list[dict]) -> dict | None:
    return max(candidates, key=lambda r: r["n_bins"]) if candidates else None


def label_composition(group: pd.DataFrame, run: dict) -> dict:
    """Exported label of the bins of a TangTV-vote run: certain, tangtv_only or not."""
    inside = group[
        (group.start_ms >= run["start_ms"]) & (group.start_ms < run["end_ms"])
    ]
    labelled = inside.state_rule.eq(run["state"])
    return {
        "bins": len(inside),
        "certain": int((labelled & inside.tier.eq("certain")).sum()),
        "tangtv_only": int((labelled & inside.tier.eq("tangtv_only")).sum()),
        "uncertain": int(inside.state_rule.eq(core.UNCERTAIN).sum()),
        "other_tiers": {
            str(k): int(v)
            for k, v in inside.loc[inside.state_rule.eq(core.UNCERTAIN), "tier"]
            .value_counts()
            .items()
        },
    }


def label_text(composition: dict, state: int) -> str:
    """The bracket of a column title: what the exported label of its bins is."""
    n, name = composition["bins"], STATE_NAME[state]
    if composition["certain"] == n:
        return f"label: certain {name}"
    if composition["tangtv_only"] == n:
        return f"label: {name}, TangTV only"
    if composition["uncertain"] == n:
        return "label: uncertain"
    return (
        f"label: {composition['certain']} certain, {composition['tangtv_only']} "
        f"TangTV only,\n{composition['uncertain']} uncertain of {n} bins"
    )


def sustained_intervals(group: pd.DataFrame) -> list[dict]:
    """The attached, detached and re-attached TangTV-vote intervals in time order.

    The detached interval is the longest run of TangTV-detached upper-shelf bins.
    The attached intervals are the longest TangTV-attached run ending before it and
    the longest one starting after it. The runs follow the TangTV vote, not the
    exported label (the label of the same bins can be certain, silver or uncertain);
    each interval records the composition of its exported label.
    """
    upper = upper_shelf(group)

    def best(state, keep):
        return longest([r for r in runs(upper, "tangtv_vote", state) if keep(r)])

    detached = best(core.DETACHED, lambda r: True)
    if detached is None:
        return []
    before = best(core.ATTACHED, lambda r: r["end_ms"] <= detached["start_ms"])
    after = best(core.ATTACHED, lambda r: r["start_ms"] >= detached["end_ms"])
    chosen = [r for r in (before, detached, after) if r]
    for r in chosen:
        r["centre_ms"] = 0.5 * (r["start_ms"] + r["end_ms"])
        r["label_composition"] = label_composition(group, r)
    return sorted(chosen, key=lambda r: r["start_ms"])


def choose_shot(frame: pd.DataFrame) -> int:
    """The transition shot: 201081 when assessed, otherwise the shot with the
    longest attached-detached-attached sequence of sustained TangTV-vote intervals."""
    if (frame.shot == 201081).any() and len(
        sustained_intervals(frame[frame.shot == 201081])
    ) == 3:
        return 201081
    best = (0, 0, 0)
    for shot, group in frame.groupby("shot"):
        if not (root() / "inversions" / f"{shot}.npz").is_file():
            continue
        found = sustained_intervals(group)
        score = (len(found), sum(r["n_bins"] for r in found), int(shot))
        best = max(best, score)
    if not best[0]:
        raise SystemExit("no shot has a sustained detached interval with inversions")
    return best[2]


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
    if not np.isin(state, (1, 2, 3, 4, 11, 12)).any():
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


def timeline(fig, spec, group: pd.DataFrame, intervals: list[dict], te, cliffs) -> None:
    """Votes, f_div, DZ and (when fetched) the SOL divertor Te against time."""
    n_rows = 6 if te is not None else 5
    ratios = [0.35, 0.35, 0.35, 1.4, 1.4] + ([1.4] if te is not None else [])
    sub = spec.subgridspec(n_rows, 1, height_ratios=ratios, hspace=0.28)
    axes = [fig.add_subplot(sub[i]) for i in range(n_rows)]
    start = group.start_ms.to_numpy()
    silver = group.tier.eq("tangtv_only").to_numpy() & group.state_rule.isin((1, 2))
    votes = {
        "label": group.state_rule.to_numpy() + SILVER_OFFSET * silver,
        "Afrac": np.where(group.afrac_valid, group.afrac_vote, 0),
        "TangTV": np.where(group.tangtv_valid, group.tangtv_vote, 0),
    }
    for ax, (name, state) in zip(axes[:3], votes.items(), strict=True):
        strip(ax, start, state, name)
    centre = start + core.BIN_MS / 2
    dz = group.tangtv_value.to_numpy(float)
    series = [
        (
            "$f_{\\mathrm{div}}$ / base.\n(not a vote)",
            np.where(group.prad_valid, group.prad_rel_value, np.nan),
            (
                (thresholds.PRAD_REL_ATTACHED_MAX, "--"),
                (thresholds.PRAD_REL_DETACHED_MIN, ":"),
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
    ]
    if te is not None:
        series.append(("$T_e$ (eV)", te, ((5.0, ":"), (10.0, "--"))))
    for ax, (name, y, lines) in zip(axes[3:], series, strict=True):
        ax.plot(centre, y, color=INK, lw=0.9, marker=".", ms=2.5)
        for level, dash in lines:
            ax.axhline(level, color="#666666", lw=0.6, ls=dash)
        ax.set_ylabel(name)
        if not np.isfinite(y).any():
            ax.text(
                0.5, 0.55, "no valid measurement", transform=ax.transAxes, ha="center"
            )
        ax.grid(axis="y", color="0.9", lw=0.4)
    axes[3].set_ylim(bottom=0)
    axes[4].set_ylim(bottom=min(0, np.nanmin(dz) if np.isfinite(dz).any() else 0))
    if te is not None:
        axes[5].set_yscale("log")
        axes[5].set_ylim(0.5, 60)
    lo, hi = start.min(), start.max() + core.BIN_MS
    for ax in axes:
        ax.set_xlim(lo, hi)
        for r in intervals if ax in axes[3:] else ():
            ax.axvspan(
                r["start_ms"],
                r["end_ms"],
                color=STATE_COLOUR[r["state"]],
                alpha=0.14,
                lw=0,
            )
        for t in cliffs:
            ax.axvline(t, color=INK, lw=0.8, ls=(0, (1, 1.5)))
    for letter, r in zip("abc", intervals, strict=False):
        axes[0].text(r["centre_ms"], 1.2, letter, ha="center", va="bottom", fontsize=8)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    for ax in axes:
        ax.yaxis.set_label_coords(-0.11, 0.5)
    axes[-1].set_xlabel("time (ms)")


#: Summary-bar categories: (legend name, colour key).
SUMMARY_CLASSES = (
    ("attached, certain", 1),
    ("attached, TangTV only", 1 + SILVER_OFFSET),
    ("detached, certain", 2),
    ("detached, TangTV only", 2 + SILVER_OFFSET),
    ("uncertain", 4),
)


def label_class(frame: pd.DataFrame) -> pd.Series:
    """The summary class of every bin: the colour key of its exported label."""
    silver = frame.tier.eq("tangtv_only") & frame.state_rule.isin((1, 2))
    return frame.state_rule + SILVER_OFFSET * silver


def summary(ax, frame: pd.DataFrame, shot: int) -> dict:
    """Per-shot fractions of assessed bins by exported label and tier, one bar per
    shot, the shot number under every bar."""
    assessed = frame[frame.assessed & frame.state_rule.isin((1, 2, 3, 4))]
    keys = [key for _, key in SUMMARY_CLASSES]
    table = (
        assessed.assign(klass=label_class(assessed))
        .groupby(["shot", "klass"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=keys, fill_value=0)
    )
    table = table[table.sum(axis=1) >= 10]
    share = table.div(table.sum(axis=1), axis=0)
    certain = share[1] + share[2]
    order = share.assign(_c=certain).sort_values(["_c", 1, 2, 11, 12], ascending=False)
    share = share.loc[order.index]
    x = np.arange(len(share))
    bottom = np.zeros(len(share))
    for name, key in SUMMARY_CLASSES:
        ax.bar(
            x,
            share[key].to_numpy(),
            bottom=bottom,
            width=1.0,
            color=STATE_COLOUR[key],
            linewidth=0,
            label=name,
        )
        bottom += share[key].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in share.index], rotation=90, fontsize=7)
    ax.tick_params(axis="x", length=2, pad=1)
    if shot in share.index:
        ax.axvline(
            list(share.index).index(shot),
            color=INK,
            lw=0.9,
            ls=(0, (1, 1.5)),
            label="example shot",
        )
    ax.set_xlim(-0.5, len(share) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(
        "shot (assessed bins at least 10), ordered by certain share", labelpad=2
    )
    ax.set_ylabel("share of bins")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -1.0),
        frameon=False,
    )
    return {
        "population": "Assessed shots with at least 10 assessed bins each",
        "n_shots": len(share),
        "ordered_shots": [int(value) for value in share.index],
        "classes": [name for name, _ in SUMMARY_CLASSES],
        "label_class_bin_counts_by_shot": {
            str(key): {name: int(table.loc[key, k]) for name, k in SUMMARY_CLASSES}
            for key in share.index
        },
    }


def mark_geometry(ax, row: pd.Series) -> dict:
    """X-point (cross) and outer strike point (ring), white with a dark halo so they
    read on the bright emission and on the dark background; None where not finite."""
    halo = [pe.withStroke(linewidth=2.2, foreground="#000000")]
    if np.isfinite(row.aux_rxpt1) and np.isfinite(row.aux_zxpt1):
        ax.plot(
            row.aux_rxpt1,
            row.aux_zxpt1,
            "x",
            color="white",
            ms=6,
            mew=1.4,
            path_effects=halo,
            zorder=6,
        )
    if np.isfinite(row.aux_rvsod) and np.isfinite(row.aux_zvsod):
        ax.plot(
            row.aux_rvsod,
            row.aux_zvsod,
            "o",
            mfc="none",
            mec="white",
            ms=5,
            mew=1.2,
            path_effects=halo,
            zorder=6,
        )
    return {
        "x_point_rz_m": [float(row.aux_rxpt1), float(row.aux_zxpt1)],
        "strike_point_rz_m": [float(row.aux_rvsod), float(row.aux_zvsod)],
    }


def witness_inversion(ax, shot: int, time: float, row: pd.Series) -> dict:
    """Show a parked inversion with the same EFIT contours as the main appendix."""
    path = root() / "inversions" / f"{shot}.npz"
    with np.load(path) as inv:
        index = int(np.argmin(abs(inv["times_ms"] - time)))
        radii, elev = inv["radii"], inv["elevation"]
        shown = inv["frames"][index].astype(float)
        ax.imshow(
            shown,
            origin="lower",
            extent=(radii[0], radii[-1], elev[0], elev[-1]),
            cmap="magma",
            vmin=0,
            vmax=max(float(np.percentile(shown, 99.8)), 0.001),
            aspect="equal",
        )
        actual = float(inv["times_ms"][index])
        ax.set_xlim(max(radii[0], 1.0), min(radii[-1], 1.8))
        ax.set_ylim(elev[0], VIEW_ZMAX)
    efit = efit_slice(shot, time)
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
            ax.plot(efit["lim"][:, 0], efit["lim"][:, 1], color="#56B4E9", lw=1)
    markers = mark_geometry(ax, row)
    if np.isfinite(row.tangtv_value) and np.isfinite(row.aux_zxpt1):
        height = row.aux_zxpt1 - (1 - row.tangtv_value) * (
            row.aux_zxpt1 - row.aux_zvsod
        )
        ax.axhline(height, color="#E69F00", lw=0.8, ls="--")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.text(
        0.02,
        0.03,
        f"inversion {actual:.0f} ms",
        color="white",
        transform=ax.transAxes,
        fontsize=7,
    )
    return {
        "inversion_source": str(path),
        "inversion_time_ms": actual,
        "efit_time_ms": efit["t_ms"] if efit else None,
        "efit_source": efit["source"] if efit else None,
        "flux_contours": [0.98, 1.0, 1.02, 1.05, 1.1] if efit else [],
        **markers,
    }


PUBLISHED_MARFE_ONSET_MS = 3705.0


def marfe_witness(frame: pd.DataFrame, out: Path) -> dict:
    """Juxtapose a TangTV-high candidate and the published MARFE onset of 199166.

    There is no certain MARFE (the fG >= 0.8 density cue has no literature source,
    `detach_label.py`), so both columns show `candidate_marfe` evidence at most. The
    published onset column reports the recall of that onset (0 of 1) in plain text.
    """
    rows = frame[(frame.shot == 199172) & (frame.tier == "candidate_marfe")]
    selection = "median-time candidate_marfe bin of 199172"
    if rows.empty:
        rows = frame[(frame.shot == 199172) & frame.tangtv_vote.eq(core.MARFE)]
        selection = "median-time TangTV MARFE vote of 199172; exported tier shown"
    if rows.empty:
        return {"status": "unavailable", "reason": "199172 has no MARFE candidate"}
    candidate = rows.iloc[int(np.argmin(abs(rows.start_ms - rows.start_ms.median())))]
    published = frame[frame.shot == 199166].sort_values("start_ms")
    if published.empty:
        return {"status": "unavailable", "reason": "199166 has no exported bins"}
    onset = PUBLISHED_MARFE_ONSET_MS
    missed = published.iloc[
        int(np.argmin(abs(published.start_ms + core.BIN_MS / 2 - onset)))
    ]
    first_candidate = published[published.tier == "candidate_marfe"]
    in_bin = published[
        (published.start_ms <= onset) & (onset < published.start_ms + core.BIN_MS)
    ]
    selections = [
        (
            199172,
            float(candidate.start_ms + core.BIN_MS / 2),
            candidate,
            "199172: candidate MARFE (no certain MARFE)",
        ),
        (199166, onset, missed, "199166: published MARFE onset, 3705 ms"),
    ]
    fig = plt.figure(figsize=(6.75, 4.1))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.1, 1.05],
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.12,
        hspace=0.44,
        wspace=0.28,
    )
    records = {}
    for col, (shot, time, row, title) in enumerate(selections):
        ax = fig.add_subplot(grid[0, col])
        provenance = witness_inversion(ax, shot, time, row)
        ax.set_title(title, fontsize=8)
        state = STATE_NAME[int(row.state_rule)]
        spatial = bool(row.tangtv_marfe_spatial)
        cue = bool(row.tangtv_marfe_second_cue)
        frame_ms = provenance["inversion_time_ms"]
        lines = [
            (
                f"Frame shown: {frame_ms:.0f} ms; label bin "
                f"{row.start_ms:.0f}\u2013{row.start_ms + core.BIN_MS:.0f} ms"
            ),
            f"Exported label: {state}; tier: {row.tier}",
            f"DZ = {row.tangtv_value:.3f}; $f_G$ = {row.aux_greenwald_fraction:.3f}",
            (
                f"Inside-separatrix cue: {'passes' if spatial else 'fails'}; "
                f"density cue: {'passes' if cue else 'fails'}"
            ),
        ]
        if shot == 199166:
            lines.append(
                "MARFE recall at the published onset: 0 of 1\n"
                "(tier candidate_marfe; MARFE is never a state)"
            )
        text_ax = fig.add_subplot(grid[1, col])
        text_ax.axis("off")
        text_ax.text(
            0,
            1,
            "\n".join(lines),
            transform=text_ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            linespacing=1.5,
        )
        records[str(shot)] = {
            **provenance,
            "shot": shot,
            "requested_time_ms": time,
            "bin_start_ms": float(row.start_ms),
            "bin_end_ms": float(row.start_ms + core.BIN_MS),
            "exported_state": state,
            "label_tier": str(row.tier),
            "tangtv_tier": str(row.tangtv_tier),
            "tangtv_vote": int(row.tangtv_vote),
            "prad_vote": int(row.prad_vote),
            "dz": float(row.tangtv_value),
            "prad_div_fraction": float(row.prad_value),
            "greenwald_fraction": float(row.aux_greenwald_fraction),
            "persistent_height_candidate": bool(row.tangtv_marfe_candidate),
            "inside_separatrix_cue": spatial,
            "density_cue": cue,
        }
    fig.legend(
        [
            Line2D([], [], color="#56B4E9", lw=1),
            Line2D([], [], color="0.3", marker="x", ls="none", ms=5),
            Line2D([], [], color="0.3", marker="o", mfc="none", ls="none", ms=5),
            Line2D([], [], color="#E69F00", ls="--", lw=0.8),
        ],
        [r"EFIT $\psi_N$ and wall", "X-point", "outer strike point", "C-III height"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
    )
    fig.savefig(
        out / "fig_detachment_marfe_witness.pdf", metadata={"CreationDate": None}
    )
    fig.savefig(out / "fig_detachment_marfe_witness.png", dpi=150)
    plt.close(fig)
    states_at_onset = {
        "bin_start_ms": float(in_bin.start_ms.iloc[0]) if len(in_bin) else None,
        "exported_state": STATE_NAME[int(in_bin.state_rule.iloc[0])]
        if len(in_bin)
        else None,
        "tier": str(in_bin.tier.iloc[0]) if len(in_bin) else None,
    }
    record = {
        "status": "available",
        "scope": "MARFE evidence on one shot and the recall of one published onset; "
        "no independent benchmark",
        "independent_benchmark": "unavailable",
        "certain_marfe_exported": False,
        "certain_marfe_reason": (
            "no literature source for the density cue (fG >= 0.8): Dong 2025 gives "
            "fG >~ 0.5 on HL-3 with a core-point density and core-Te condition, "
            "not this cue"
        ),
        "candidate_selection": selection,
        "published_onset_source": (
            ".tmp/label_papers/Chen_2026_Nucl._Fusion_66_036014.md"
        ),
        "published_onset_ms": onset,
        "published_onset_recall": {
            "recalled": 0,
            "of": 1,
            "meaning": "no bin of 199166 is exported as MARFE (no shot has one)",
            "label_at_onset": states_at_onset,
            "first_candidate_marfe_bin_ms": (
                float(first_candidate.start_ms.min()) if len(first_candidate) else None
            ),
            "candidate_marfe_bins_199166": len(first_candidate),
        },
        "thresholds": {
            "dz_candidate_min": thresholds.DZ_MARFE_MIN,
            "greenwald_cue_min": thresholds.GREENWALD_CUE_MIN,
            "persistence_min_bins": thresholds.MARFE_MIN_BINS,
            "prad_div_and_afrac": "not MARFE corroborators",
        },
        "width_in": 6.75,
        "height_in": 4.1,
        "min_font_pt": 7,
        "selections": records,
        "flux_legend": "blue: EFIT psi_N/wall; white x: X-point; white o: "
        "outer strike point; dashed orange: extracted C-III front height",
        "labels_source": str(root() / "labels_bins.csv.gz"),
        "labels_sha256": hashlib.sha256(
            (root() / "labels_bins.csv.gz").read_bytes()
        ).hexdigest(),
    }
    (out / "marfe_witness.json").write_text(dumps(record, indent=1))
    return record


def draw_inversion(ax, shot, group, run, inv, efit_t) -> dict:
    """The inversion nearest the interval centre with EFIT, X-point, strike point
    and the front-height line at the interval's median DZ."""
    j = int(np.argmin(np.abs(inv["times_ms"] - run["centre_ms"])))
    radii, elev = inv["radii"], inv["elevation"]
    shown = inv["frames"][j].astype(float)
    ax.imshow(
        shown,
        origin="lower",
        extent=(radii[0], radii[-1], elev[0], elev[-1]),
        cmap="magma",
        vmin=0,
        vmax=max(float(np.percentile(shown, 99.8)), 0.001),
        aspect="equal",
    )
    efit = efit_slice(shot, efit_t)
    inside = group[
        (group.start_ms >= run["start_ms"]) & (group.start_ms < run["end_ms"])
    ]
    row = inside.iloc[
        np.argmin(np.abs(inside.start_ms + core.BIN_MS / 2 - run["centre_ms"]))
    ]
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
            ax.plot(efit["lim"][:, 0], efit["lim"][:, 1], color="#56B4E9", lw=1.0)
    markers = mark_geometry(ax, row)
    dz = float(np.nanmedian(inside.tangtv_value))
    if np.isfinite(dz) and np.isfinite(row.aux_zxpt1):
        height = row.aux_zxpt1 - (1 - dz) * (row.aux_zxpt1 - row.aux_zvsod)
        ax.axhline(height, color="#E69F00", lw=0.8, ls="--")
    ax.set_xlim(max(radii[0], 1.0), min(radii[-1], 1.8))
    ax.set_ylim(elev[0], VIEW_ZMAX)
    ax.set_xticks([1.0, 1.2, 1.4, 1.6])
    ax.set_xlabel("R (m)")
    return {
        "inversion_time_ms": float(inv["times_ms"][j]),
        "efit_time_ms": efit["t_ms"] if efit else None,
        "efit_source": efit["source"] if efit else None,
        "median_dz": dz,
        "strike_r_m": float(row.aux_rvsod),
        "strike_z_m": float(row.aux_zvsod),
        **markers,
    }


def views_figure(shot, group, intervals, out) -> dict:
    inv = np.load(root() / "inversions" / f"{shot}.npz")
    video = read_video(shot)
    fig = plt.figure(figsize=(6.75, 6.0), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        3,
        height_ratios=[1, 1.15, 0.95],
        left=0.095,
        right=0.965,
        top=0.93,
        bottom=0.17,
        hspace=0.46,
        wspace=0.42,
    )
    selections = {}
    for col, run in enumerate(intervals):
        state = run["state"]
        span = f"{run['start_ms']:.0f}–{run['end_ms']:.0f} ms"
        ax = fig.add_subplot(outer[0, col])
        bracket = label_text(run["label_composition"], state)
        ax.set_title(
            f"({chr(97 + col)}) TangTV {STATE_NAME[state]}, {span}\n({bracket})",
            fontsize=7.5,
        )
        raw_time = None
        if video is not None:
            vt, vid = video
            k = int(np.argmin(np.abs(vt - run["centre_ms"])))
            raw_time = float(vt[k])
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
        ax = fig.add_subplot(outer[1, col])
        info = draw_inversion(ax, shot, group, run, inv, run["centre_ms"])
        if col == 0:
            ax.set_ylabel("Z (m)")
        ax.set_title(f"C-III inversion ({info['inversion_time_ms']:.0f} ms)")
        selections[chr(97 + col)] = {
            **run,
            "state": STATE_NAME[state],
            "tangtv_vote": STATE_NAME[state],
            "exported_label": bracket,
            "raw_time_ms": raw_time,
            **info,
        }
    ax = fig.add_subplot(outer[2, :])
    centre = group.start_ms.to_numpy(float) + core.BIN_MS / 2
    divl = group.aux_prad_divl_w.to_numpy(float) / 1e6
    total = group.aux_prad_tot_w.to_numpy(float) / 1e6
    ax.plot(centre, total, color="#0072B2", lw=1.0, label=r"$P_{\mathrm{rad,tot}}$")
    ax.plot(centre, divl, color="#D55E00", lw=1.0, label=r"$P_{\mathrm{rad,div,L}}$")
    for col, run in enumerate(intervals):
        ax.axvspan(
            run["start_ms"],
            run["end_ms"],
            color=STATE_COLOUR[run["state"]],
            alpha=0.18,
            lw=0,
        )
        ax.text(
            run["centre_ms"],
            1.02,
            chr(97 + col),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    cliffs = PUBLISHED_CLIFFS_MS.get(shot, ())
    for t in cliffs:
        ax.axvline(t, color=INK, lw=0.8, ls=(0, (1, 1.5)))
    ax.set_xlim(centre.min(), centre.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("radiated power (MW)")
    ax.grid(axis="y", color="0.9", lw=0.4)
    handles, labels = ax.get_legend_handles_labels()
    if cliffs:
        handles.append(Line2D([], [], color=INK, lw=0.8, ls=(0, (1, 1.5))))
        labels.append("Te cliffs (Chen 2026)")
    handles += [
        plt.Rectangle((0, 0), 1, 1, color=STATE_COLOUR[s], alpha=0.35) for s in (1, 2)
    ]
    labels += ["TangTV-attached interval", "TangTV-detached interval"]
    ax.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        frameon=False,
    )
    fig.legend(
        [
            Line2D([], [], color="#56B4E9", lw=1),
            Line2D([], [], color="0.3", marker="x", ls="none", ms=4),
            Line2D([], [], color="0.3", marker="o", mfc="none", ls="none", ms=4),
            Line2D([], [], color="#E69F00", ls="--", lw=0.8),
        ],
        [r"EFIT $\psi_N$ and wall", "X-point", "outer strike point", "C-III height"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
    )
    fig.savefig(out / "fig_detachment_views.pdf", metadata={"CreationDate": None})
    fig.savefig(out / "fig_detachment_views.png", dpi=150)
    plt.close(fig)
    return selections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--out-dir", default=str(root() / "figure"))
    args = parser.parse_args()
    style()
    frame = load_labels()
    if args.shot is None:
        args.shot = choose_shot(frame)
    group = frame[frame.shot == args.shot].sort_values("start_ms")
    if group.empty:
        raise SystemExit(f"shot {args.shot} is not in labels_bins.csv.gz")
    intervals = sustained_intervals(group)
    if not intervals:
        raise SystemExit("the shot has no sustained detached interval")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selections = views_figure(args.shot, group, intervals, out)
    from detach_te_check import attach_te

    with_te = attach_te(group.reset_index(drop=True))
    te = with_te.te_ev.to_numpy(float) if np.isfinite(with_te.te_ev).any() else None
    cliffs = PUBLISHED_CLIFFS_MS.get(args.shot, ())
    fig = plt.figure(figsize=(6.75, 6.9))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[4.6, 0.9],
        left=0.18,
        right=0.985,
        top=0.96,
        bottom=0.2,
        hspace=0.3,
    )
    timeline(fig, grid[0], group, intervals, te, cliffs)
    summary_record = summary(fig.add_subplot(grid[1]), frame, args.shot)
    fig.savefig(out / "fig_detachment_timeline.pdf", metadata={"CreationDate": None})
    fig.savefig(out / "fig_detachment_timeline.png", dpi=150)
    plt.close(fig)
    witness_record = marfe_witness(frame, out)
    labels_path = root() / "labels_bins.csv.gz"
    (out / "figure.json").write_text(
        dumps(
            {
                "scope": "exploratory coverage, indicator agreement and views; "
                "no independent benchmark",
                "independent_benchmark": "unavailable",
                "shot": args.shot,
                "labels_source": str(labels_path),
                "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                "min_run_bins": MIN_RUN_BINS,
                "intervals": selections,
                "interval_rule": (
                    "longest run of consecutive TangTV-detached upper-shelf bins; "
                    "longest TangTV-attached run before and after it. The columns "
                    "follow the TangTV vote, not the exported label; each column "
                    "title carries the exported label of its bins in brackets "
                    "(certain, TangTV only (silver) or uncertain)"
                ),
                "caption_note": (
                    "Columns are titled by the TangTV vote with the exported label of "
                    "the same bins in brackets; a column can be a TangTV vote the "
                    "label calls uncertain. The summary bars below the timeline use "
                    "the exported label and tier (silver = TangTV only)."
                ),
                "published_te_cliffs_ms": list(cliffs),
                "published_te_cliffs_source": ".tmp/label_papers/"
                "Chen_2026_Nucl._Fusion_66_036014.md (shot 201081)",
                "multi_shot_summary": summary_record,
                "marfe_witness": witness_record,
                "width_in": 6.75,
                "min_font_pt": 7,
                "placement": (
                    "Full textwidth at 6.75 inches. Do not shrink to a single "
                    "column; minimum 7 pt is guaranteed only at the declared size."
                ),
                "radiation_row": (
                    "PRAD_DIVL (lower-divertor) and PRAD_TOT traces, 250 ms "
                    "inter-ELM means as used by f_div. A 2D bolometer emissivity "
                    "to overlay with EFIT does not exist: see "
                    "docs/labeler/results/detachment_bolometer_availability.json"
                ),
                "timeline_rows": (
                    "label (silver = TangTV only), Afrac and TangTV vote strips; "
                    "f_div over its shot baseline (a corroborator, not a vote) and "
                    "DZ with the cutoffs (dashed attached, dotted detached, "
                    "dash-dot MARFE); "
                    "divertor Thomson Te at the SOL chords near the target with the "
                    "5 and 10 eV bands when fetched (log axis)"
                ),
                "timeline_te": "shown"
                if te is not None
                else "no Thomson Te for this shot",
                "irtv_row": "omitted: the IRTV HEATFLUX attempt returned NODATA",
                "raw_overlay": "no camera projection calibration available",
                "efit_source": "EFIT02, or explicitly recorded EFIT01 fallback",
            },
            indent=1,
        )
    )
    print("intervals:", [(r["state"], r["start_ms"], r["end_ms"]) for r in intervals])
    print("wrote", out / "fig_detachment_views.pdf")


if __name__ == "__main__":
    main()
