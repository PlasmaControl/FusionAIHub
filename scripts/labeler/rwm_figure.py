#!/usr/bin/env python
"""Plot held-out rwm-brf scores and beta_N/li near onsets on eight fixed examples.

Choose first, middle and last n=1 Hanson shots in each campaign by shot number,
without inspecting scores. Add the first matched comparison shot for each campaign's
first Hanson example, centred on that Hanson's onset time (a reference, not an onset
on the unlabelled shot). Write vector PDF, 150-dpi PNG and source rows outside git.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from labeler.rwm.records import load_evaluation

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "labeler" / "rwm"
SCORE_COLOR, BETA_COLOR = "#0072B2", "#D55E00"
WINDOW_MS = (-800.0, 600.0)
#: A pre-onset beta_N/l_i below this share of its earlier maximum counts as a collapse.
COLLAPSE_FRACTION = 0.75


def collapse_counts(rows, slices):
    """Eligible n=1 onsets whose window beta_N/l_i is below 0.75 of the earlier peak.

    For each merged n=1 onset (`onset_physics` rows) the value is the first cached
    10 ms slice in [o-20 ms, o), the same snapshot the onset tables use; the
    earlier peak is the largest finite beta_N/l_i of that shot's cached slices before
    the snapshot. An onset with either missing is not eligible.
    """
    by_shot = {int(s): g for s, g in slices.groupby("shot")}
    counts = {}
    for row in rows:
        value, shot = row["betan_over_li"], int(row["shot"])
        cell = counts.setdefault(
            str(row["campaign"]), {"eligible": 0, "below": 0, "below_shots": []}
        )
        prior = by_shot[shot]
        prior = prior.betan_over_li[prior.t_ms < row["sample_ms"]].dropna()
        if value is None or not len(prior):
            continue
        cell["eligible"] += 1
        if value < COLLAPSE_FRACTION * prior.max():
            cell["below"] += 1
            cell["below_shots"].append(shot)
    return counts


def main():
    record = load_evaluation(OUT / "evaluation.json")
    paths_root = Path(os.environ["LABELER_ROOT"])
    out_dir = paths_root / "round4" / "rwm"
    prediction_path = Path(record["configs"]["rwm-brf"]["predictions"])
    scores = pd.read_parquet(prediction_path)
    onsets = pd.read_csv(out_dir / "growth_onsets.csv")
    onsets = onsets[onsets.ntor == 1]
    roster = pd.read_csv(out_dir / "shots.csv")
    examples, first_by_campaign = [], {}
    for year in (2014, 2018):
        members = sorted(
            set(
                scores.loc[
                    (scores.campaign == year) & (scores.role == "hanson"), "shot"
                ]
            )
            & set(onsets.shot)
        )
        chosen = [members[0], members[len(members) // 2], members[-1]]
        first_by_campaign[year] = chosen[0]
        for shot in chosen:
            event_times = sorted(onsets.loc[onsets.shot == shot, "onset_ms"])
            examples.append(
                {
                    "shot": int(shot),
                    "campaign": year,
                    "role": "hanson",
                    "centre_ms": float(event_times[0]),
                    "target_onsets_ms": event_times,
                }
            )
    for year, partner in first_by_campaign.items():
        matched = roster[roster.selected & (roster.matched_to == partner)]
        shot = int(matched.shot.min())
        centre = float(onsets.loc[onsets.shot == partner, "onset_ms"].min())
        examples.append(
            {
                "shot": shot,
                "campaign": year,
                "role": "comparison",
                "matched_to": int(partner),
                "centre_ms": centre,
                "target_onsets_ms": [],
            }
        )
    # Rows compare campaigns, with the unlabelled examples together on the last row.
    examples = [examples[i] for i in (0, 3, 1, 4, 2, 5, 6, 7)]
    frames = []
    for example in examples:
        rows = scores[scores.shot == example["shot"]].sort_values("t_ms").copy()
        rows["relative_ms"] = rows.t_ms - example["centre_ms"]
        rows = rows[rows.relative_ms.between(*WINDOW_MS)]
        frames.append(rows)
        example["displayed_slices"] = len(rows)
    finite_ratios = np.concatenate(
        [r.betan_over_li.dropna().to_numpy() for r in frames]
    )
    ratio_max = max(8.0, 2.0 * np.ceil(finite_ratios.max() / 2.0))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )
    fig, axes = plt.subplots(4, 2, figsize=(3.25, 5.6), sharex=True, sharey=True)
    for index, (ax, example, rows) in enumerate(zip(axes.ravel(), examples, frames)):
        right = ax.twinx()
        x = rows.relative_ms.to_numpy() / 1000.0
        right.plot(x, rows.betan_over_li, color=BETA_COLOR, lw=0.9, ls="--")
        ax.plot(x, rows.score, color=SCORE_COLOR, lw=1.0)
        for onset in example["target_onsets_ms"]:
            relative = (onset - example["centre_ms"]) / 1000.0
            if WINDOW_MS[0] / 1000 <= relative <= WINDOW_MS[1] / 1000:
                ax.axvspan(relative - 0.1, relative, color="#999999", alpha=0.15)
                ax.axvline(relative, color="#222222", lw=0.7)
        right.axhline(4, color=BETA_COLOR, alpha=0.4, lw=0.5, ls=":")
        if example["role"] == "comparison":
            ax.text(
                0.04,
                0.93,
                f"{example['shot']} (unlabelled)",
                transform=ax.transAxes,
                va="top",
                fontsize=7.0,
            )
        else:
            ax.text(
                0.04,
                0.93,
                str(example["shot"]),
                transform=ax.transAxes,
                va="top",
                fontsize=7.5,
            )
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(np.array(WINDOW_MS) / 1000)
        ax.set_xticks([-0.6, 0, 0.6])
        right.set_ylim(0, ratio_max)
        right.set_yticks([0, 4, ratio_max])
        right.tick_params(axis="y", colors=BETA_COLOR, length=2, pad=1)
        ax.tick_params(length=2, pad=1)
        if index % 2 == 0:
            right.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        right.spines["top"].set_visible(False)
        ax.grid(axis="y", lw=0.35, color="#dddddd")
    fig.subplots_adjust(
        left=0.13, right=0.86, bottom=0.11, top=0.89, wspace=0.20, hspace=0.23
    )
    fig.text(0.015, 0.52, "Held-out rwm-brf score", rotation=90, va="center")
    fig.text(0.945, 0.52, r"$\beta_N/l_i$", rotation=90, va="center", color=BETA_COLOR)
    fig.text(
        0.5, 0.035, "Time from onset / matched reference (s)", ha="center", fontsize=7.0
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=SCORE_COLOR, lw=1, label="rwm-brf"),
            Line2D([0], [0], color=BETA_COLOR, lw=1, ls="--", label=r"$\beta_N/l_i$"),
            Line2D([0], [0], color="#222222", lw=0.7, label="n=1 onset"),
            Patch(facecolor="#999999", alpha=0.15, label="100 ms forecast band"),
            Line2D(
                [0],
                [0],
                color=BETA_COLOR,
                alpha=0.4,
                lw=0.5,
                ls=":",
                label=r"no-wall proxy $\beta_N/l_i=4$",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.5,
    )
    stem = out_dir / "rwm_onset_scores"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    plt.close(fig)
    source_rows = out_dir / "rwm_figure_rows.csv"
    pd.concat(frames).to_csv(source_rows, index=False)
    first_onset = next(
        row for row in record["onset_physics"]["rows"] if row["shot"] == 156785
    )
    collapse = collapse_counts(
        record["onset_physics"]["rows"],
        pd.read_parquet(
            out_dir / "slices.parquet", columns=["shot", "t_ms", "betan_over_li"]
        ),
    )
    early, late = collapse["2014"], collapse["2018"]
    metadata = {
        "script": "scripts/labeler/rwm_figure.py",
        "model": "rwm-brf",
        "prediction_path": str(prediction_path),
        "fold_seed": 0,
        "selection": "first/middle/last n=1 Hanson shot in each campaign; first matched comparisons",
        "examples": examples,
        "window_ms": list(WINDOW_MS),
        "ratio_ylim": [0, ratio_max],
        "score_is_calibrated_probability": False,
        "dpi": 150,
        "pdf": str(stem.with_suffix(".pdf")),
        "png": str(stem.with_suffix(".png")),
        "source_rows_csv": str(source_rows),
        "onset_physics_source": (
            "outputs/labeler/rwm/evaluation.json#/onset_physics/rows"
        ),
        "first_panel_onset_physics": first_onset,
        "collapse_check": {
            "definition": (
                "n=1 onsets whose first cached slice in [o-20 ms, o) has beta_N/l_i "
                f"below {COLLAPSE_FRACTION} of the largest finite beta_N/l_i of that "
                "shot's cached slices before it"
            ),
            "fraction": COLLAPSE_FRACTION,
            "by_campaign": collapse,
            "source": "outputs/labeler/rwm/evaluation.json#/onset_physics/rows",
        },
        "caption": (
            "Held-out scores of the equilibrium-scalar timing baseline rwm-brf (blue) "
            "and βN/li (orange) on shots chosen by number, not by score. Vertical lines "
            "mark n=1 onsets; grey spans are 100 ms forecast targets. Bottom panels "
            "are unlabelled comparisons centred on matched Hanson onsets. The dotted "
            "βN/li=4 proxy uses the right axis, aligning with score 0.5. Panels "
            "156796 and 158022 show βN/li collapses well before the onset, which is "
            "atypical: in the window just before the onset only "
            f"{early['below']} of {early['eligible']} eligible 2014 onsets "
            f"({', '.join(map(str, early['below_shots']))}) and "
            f"{late['below']} of {late['eligible']} in 2018 have βN/li below "
            f"{COLLAPSE_FRACTION:g} of its earlier maximum. The "
            "score is not a precursor indicator: on 156785 it rises after the onset "
            "and on 176077 it spikes at the onset. Neither physical duration nor "
            "warning skill is established."
        ),
    }
    metadata["caption_words"] = len(metadata["caption"].split())
    (OUT / "figure.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    main()
