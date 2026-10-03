#!/usr/bin/env python
"""Write detachment F1 sources and a final-column-width panel.

Local single indicators and shot-held-out detach-victor are scored against the
unverified consensus on non-test shots. Published settings remain unavailable.
Coverage has separate measurement, valid-measurement, vote and consensus counts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from detach_json import dumps
from detach_label import cohort_split, load_all
from detach_ours import BOOTSTRAPS, with_ci

from labeler.events.detachment import core

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs" / "labeler" / "results"
F1_JSON = REPO / "docs" / "labeler" / "figure2_detach.json"
LF_NAMES = ("afrac", "prad", "tangtv")
MODEL_NAMES = {
    "afrac": "detach-jsat-proxy",
    "prad": "detach-prad-proxy",
    "tangtv": "detach-tangtv-local",
}
SETTINGS = {
    "afrac": "Uncalibrated local Jsat ratio; attached-current fit unavailable",
    "prad": "Local Prad,div/P_in thresholds 0.36 / 0.50",
    "tangtv": "Upper-shelf C-III front with local transition band, quality "
    "and MARFE gates",
}
PUBLISHED = {
    "afrac": {
        "model": "detach-afrac",
        "paper": "Eldon 2021/2022",
        "setting": "Fitted attached pre-puff current reference, separate L/H",
        "reason": "No independently fitted attached-current reference",
    },
    "prad": {
        "model": "detach-prad",
        "paper": "Eldon 2019",
        "setting": "Calibrated lower-divertor radiation sensor",
        "reason": "Published measurement has no universal state threshold; "
        "local thresholds are a proxy setting",
    },
    "tangtv": {
        "model": "detach-tangtv",
        "paper": "Chen 2026",
        "setting": "Published upper-shelf C-III SSA/DZ front setting",
        "reason": "SSA extraction partially reused; local quality, transition "
        "and MARFE gates modify the published setting",
    },
    "victor": {
        "model": "detach-victor",
        "paper": "Victor and Scotti 2024",
        "setting": "Published binary hand-labelled divertor-camera CNN",
        "reason": "Local CNN uses three consensus targets and different "
        "camera data; published hand-labelled setting unavailable",
    },
}


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def population(frame: pd.DataFrame, keep) -> dict:
    selected = frame.loc[np.asarray(keep, bool)]
    return {
        "bins": len(selected),
        "shots": int(selected.shot.nunique()),
        "shot_ids": sorted(int(s) for s in selected.shot.unique()),
    }


def f1_entry(truth, pred, shot, rng, *, source, key, setting) -> dict:
    scores = with_ci(truth, pred, shot, rng)
    f1 = scores["macro_f1"]
    return {
        "status": "available" if len(truth) else "unavailable",
        "f1": f1["value"],
        "ci95": {"f1": f1["ci95"]},
        "valid_replicates": {"f1": f1["valid_replicates"]},
        "replicates": BOOTSTRAPS,
        "n_bins": len(truth),
        "n_shots": len(np.unique(shot)),
        "shot_ids": sorted(int(s) for s in np.unique(shot)),
        "state_counts": {
            core.STATE_NAMES[k + 1]: int(np.sum(truth == k)) for k in range(3)
        },
        "per_state": {
            core.STATE_NAMES[k + 1]: scores["f1_" + core.STATE_NAMES[k + 1]]
            for k in range(3)
        },
        "setting": setting,
        "source": source,
        "key": key,
    }


def build() -> dict:
    labels = pd.read_csv(root() / "labels_bins.csv.gz")
    non_test = labels.split.ne("test").to_numpy()
    certain = labels.state_lm.isin(core.VOTE_STATES).to_numpy()
    rng = np.random.default_rng(7)
    local = {}
    for name in LF_NAMES:
        cast = (
            labels[f"{name}_valid"].to_numpy(bool)
            & labels[f"{name}_vote"].isin(core.VOTE_STATES).to_numpy()
        )
        keep = non_test & certain & cast
        local[MODEL_NAMES[name]] = f1_entry(
            labels.state_lm.to_numpy(int)[keep] - 1,
            labels[f"{name}_vote"].to_numpy(int)[keep] - 1,
            labels.shot.to_numpy()[keep],
            rng,
            source=str(root() / "labels_bins.csv.gz"),
            key=f"{name}_vote versus state_lm; valid votes, certain, split!=test",
            setting=SETTINGS[name],
        )
        local[MODEL_NAMES[name]]["reference_certain_population"] = population(
            labels, non_test & certain
        )
    predictions = root() / "victor" / "predictions.npz"
    if predictions.is_file():
        with np.load(predictions) as data:
            rows = labels.set_index(["shot", "start_ms"]).loc[
                list(zip(data["shot"], data["start_ms"], strict=True))
            ]
            keep = (
                rows.split.ne("test").to_numpy()
                & rows.state_lm.isin(core.VOTE_STATES).to_numpy()
                & np.isfinite(data["prob"]).all(axis=1)
            )
            target = np.where(
                rows.state_lm.isin(core.VOTE_STATES), rows.state_lm - 1, -1
            )
            if not np.array_equal(target, data["y"]):
                raise ValueError("Victor predictions are stale; rerun prep and train")
            local["detach-victor"] = f1_entry(
                target[keep],
                data["prob"][keep].argmax(axis=1),
                data["shot"][keep],
                rng,
                source=str(predictions),
                key="shot-held-out prob argmax versus state_lm; certain, split!=test",
                setting="Local camera CNN, 5-fold shot-grouped CV; non-test "
                "training and held-out predictions",
            )
    else:
        local["detach-victor"] = f1_entry(
            np.array([], int),
            np.array([], int),
            np.array([], int),
            rng,
            source=str(predictions),
            key="unavailable predictions",
            setting="Local camera CNN; predictions unavailable",
        )
    legacy = {
        p["model"]: {
            **p,
            "status": "unavailable",
            "reproduced": False,
            "reproduction_status": "partial" if name == "tangtv" else "unavailable",
            "f1": None,
            "ci95": {"f1": [None, None]},
            "valid_replicates": {"f1": 0},
            "replicates": BOOTSTRAPS,
        }
        for name, p in PUBLISHED.items()
    }
    result = {
        "task": "detachment",
        "metric": "macro F1 over reference states present in each population",
        "bin_ms": core.BIN_MS,
        "reference": "unverified compatible diagnostic consensus; certain "
        "state_lm (primary rule alias); non-test shots",
        "interpretation": "Diagnostic agreement only. Single indicators share "
        "votes with consensus; camera CNN shares its diagnostic. No independent "
        "physical validation or confirmed MARFE accuracy.",
        "bootstrap": {"unit": "shot", "replicates": BOOTSTRAPS, "ci": 0.95, "seed": 7},
        "results": {"legacy": legacy, "Tokamak-SI": local},
        "rows": [
            {
                "panel": "detachment",
                "series": model,
                "group": setting,
                "metric": "macro F1",
                "value": entry["f1"],
                "ci_lo": entry["ci95"]["f1"][0],
                "ci_hi": entry["ci95"]["f1"][1],
                "source": "docs/labeler/figure2_detach.json",
                "key": f"results.{setting}.{model}.f1",
            }
            for setting, models in (("legacy", legacy), ("Tokamak-SI", local))
            for model, entry in models.items()
        ],
    }
    F1_JSON.write_text(dumps(result, indent=1))
    coverage(labels, certain)
    return result


def coverage(labels: pd.DataFrame, certain: np.ndarray) -> None:
    bins = load_all(root() / "bins")
    bins["split"] = bins.shot.map(cohort_split(bins.shot.unique()))
    state = labels.set_index(["shot", "start_ms"]).state_lm
    bins["consensus"] = state.reindex(
        pd.MultiIndex.from_frame(bins[["shot", "start_ms"]])
    ).to_numpy()
    table = []
    for name in LF_NAMES:
        valid = bins[f"{name}_valid"].to_numpy(bool)
        cast = valid & bins[f"{name}_vote"].isin(core.VOTE_STATES).to_numpy()
        table.append(
            {
                "indicator": name,
                "setting": SETTINGS[name],
                "population": "all extracted shot/bin grids, every split; "
                "counts are not performance scores",
                "population_counts": population(bins, np.ones(len(bins), bool)),
                "measurement": population(
                    bins, np.isfinite(bins[f"{name}_value"].to_numpy(float))
                ),
                "valid_measurement": population(bins, valid),
                "vote": population(bins, cast),
                "consensus": population(
                    bins, cast & bins.consensus.isin(core.VOTE_STATES).to_numpy()
                ),
                "published_setting_reproduced": False,
                "reproduction_status": "partial" if name == "tangtv" else "proxy",
            }
        )
    paper = {
        "task": "detachment",
        "bin_ms": core.BIN_MS,
        "reference": "unverified diagnostic consensus; no expert truth",
        "legacy": [
            {
                "indicator": f"{p['model']} ({p['paper']})",
                "setting": p["setting"],
                "reproduced": False,
                "reproduction_status": "partial" if name == "tangtv" else "unavailable",
                "reason": p["reason"],
                "coverage_kind": "published setting unavailable",
                "coverage": None,
            }
            for name, p in PUBLISHED.items()
            if name != "victor"
        ],
        "Tokamak-SI": {
            "coverage_kind": "certain compatible diagnostic consensus; unverified",
            "setting": "TangTV-supported compatible rule with geometry, "
            "measurement-quality and MARFE gates; posterior diagnostic only",
            "assessed": population(labels, np.ones(len(labels), bool)),
            "certain": population(labels, certain),
            "state_counts": {
                core.STATE_NAMES[k]: int(np.sum(labels.state_lm == k))
                for k in (1, 2, 3, 4)
            },
            "tier_counts": {
                str(k): int(v) for k, v in labels.tier.value_counts().items()
            },
        },
        "coverage_table": table,
        "coverage_definitions": {
            "measurement": "finite scalar measurement, before validity gates",
            "valid_measurement": "measurement passes indicator validity gates",
            "vote": "valid attached/detached/MARFE vote; excludes abstentions",
            "consensus": "vote bin also has a certain exported consensus state",
        },
        "local_proxies": {
            "jsat": table[0]["valid_measurement"],
            "prad": table[1]["valid_measurement"],
        },
        "sources": [str(root() / "bins"), str(root() / "labels_bins.csv.gz")],
        "performance_source": "../figure2_detach.json",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "detachment_figure2.json").write_text(dumps(paper, indent=1))


def draw(data: dict, out: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    models = (*MODEL_NAMES.values(), "detach-victor")
    fig, ax = plt.subplots(figsize=(3.25, 2.65))
    for i, model in enumerate(models):
        entry = data["results"]["Tokamak-SI"][model]
        value, ci = entry["f1"], entry["ci95"]["f1"]
        if value is None or not np.isfinite(value):
            ax.text(i, 0.5, "unavailable", rotation=90, ha="center", fontsize=7)
            continue
        ax.bar(i, value, color="#0072B2" if i == 3 else "#E69F00", width=0.65)
        top = value
        if all(v is not None and np.isfinite(v) for v in ci):
            ax.errorbar(
                i,
                value,
                yerr=[[max(value - ci[0], 0)], [max(ci[1] - value, 0)]],
                fmt="none",
                color="#333333",
                lw=0.7,
                capsize=2,
            )
            top = max(top, ci[1])
        ax.text(i, top + 0.025, f"{value:.2f}", ha="center", fontsize=7)
    ax.set_xticks(
        range(4), ["Jsat\nproxy", "Prad\nproxy", "C-III\nfront", "Victor\nCNN"]
    )
    ax.set_xlim(-0.7, 3.7)
    ax.set_ylim(0, 1.13)
    ax.set_yticks((0, 0.5, 1))
    ax.set_ylabel("Macro F1 vs consensus")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    fig.text(
        0.54,
        0.015,
        "Diagnostic agreement; published settings unavailable",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.23)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_detachment_figure2.pdf", metadata={"CreationDate": None})
    fig.savefig(out / "fig_detachment_figure2.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=root() / "figure")
    args = parser.parse_args()
    data = build()
    draw(data, args.out_dir)
    print("wrote", F1_JSON, "and", args.out_dir / "fig_detachment_figure2.pdf")


if __name__ == "__main__":
    main()
