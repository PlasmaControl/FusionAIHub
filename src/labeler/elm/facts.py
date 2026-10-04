"""Numbers quoted in the ELM captions, notes and documents, read from the JSON records.

`load` reads the evaluation records under `outputs/labeler/elm/`; `facts` reduces them
to the figures the generators (`scripts/labeler/elm_paper_tables.py`,
`scripts/labeler/elm_protocol.py`) print. No caption or document types a number: a
re-run of an evaluation changes the text with it. Records that a later step produces
(`dsm/baseline_seeds.json`, `ours/run_day_cv.json`, ...) are optional; `facts` then
leaves their entries `None` and the generators drop the sentence that needs them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parents[3] / "outputs" / "labeler" / "elm"

REQUIRED = {
    "ours": "ours/evaluation.json",
    "dsm": "dsm/evaluation.json",
    "swap": "swap/evaluation.json",
    "feature": "ours/feature_only.json",
    "strata": "ours/annotation_strata.json",
    "offsets": "review_start_offsets.json",
    "density": "density_units.json",
    "filterscope": "filterscope_metadata.json",
    "history": "training_history.json",
    "native_detection": "dsm/native_detection.json",
    "rejection": "ours/rejection_sensitivity.json",
    "seeds": "ours/seed_repeats.json",
}
OPTIONAL = {
    "native": "dsm/native_evaluation.json",
    "smith": "smith/evaluation.json",
    "moved": "ours/moved_boundary_stratum.json",
    "tiled": "ours/tiled_inference.json",
    "run_day": "ours/run_day_cv.json",
    "dsm_seeds": "dsm/baseline_seeds.json",
    "density_range": "density_range.json",
    "photodiode": "dsm/swap_photodiode_agreement.json",
}
ELMO = "elm-elmo"
OURS = "elm-ours"
CLOCK = "elm-clock"
DSM_DETECT = "elm-dsm-detect"
DSM_NATIVE = "elm-dsm-native-detect"
OURS_NATIVE = "elm-ours-native-folds"
METRIC_KEYS = ("auroc", "auprc", "f1")


def paths(outputs: Path = OUTPUTS) -> dict[str, Path]:
    """Every record path, required and optional, that exists."""
    wanted = {**REQUIRED, **OPTIONAL}
    return {k: outputs / v for k, v in wanted.items() if (outputs / v).exists()}


def load(outputs: Path = OUTPUTS) -> dict:
    """The JSON records by key; a missing required one is an error."""
    found = paths(outputs)
    missing = [k for k in REQUIRED if k not in found]
    if missing:
        raise FileNotFoundError(f"missing ELM records: {missing}")
    return {k: json.loads(p.read_text()) for k, p in found.items()}


def fmt(value: float, ci=None, digits: int = 3, signed: bool = False) -> str:
    """`0.298` or `0.298 [0.132, 0.549]`; `+` signs for differences."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    sign = "+" if signed else ""
    out = f"{value:{sign}.{digits}f}"
    if ci is not None and not math.isnan(ci[0]):
        out += f" [{ci[0]:{sign}.{digits}f}, {ci[1]:{sign}.{digits}f}]"
    return out


def point(res: dict, key: str) -> tuple[float, list | None]:
    """A method result's point value and its interval."""
    return res["point"][key], res.get("ci95", {}).get(key)


def paired(row: dict) -> tuple[float, list]:
    return row["value"], row["ci95"]


def delta_text(rows: dict) -> str:
    """`AUROC +0.023 [-0.014, +0.067], ...` from `{metric: (value, interval)}`."""
    return ", ".join(
        f"{m.upper()} {fmt(v, ci, 3, signed=True)}" for m, (v, ci) in rows.items()
    )


def spread(values) -> dict:
    values = [float(v) for v in values]
    mean = sum(values) / len(values)
    sd = (
        math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        if len(values) > 1
        else float("nan")
    )
    return {"mean": mean, "min": min(values), "max": max(values), "sd": sd}


def facts(records: dict) -> dict:
    """The figures the captions, notes and documents quote."""
    ours, dsm = records["ours"], records["dsm"]
    sets = {tag: ours["sets"][tag] for tag in ("all119", "bes73")}
    common = {tag: dsm["sets"][tag] for tag in ("all119", "bes73")}
    out: dict = {
        "bins": {t: s["bins"] for t, s in sets.items()},
        "shots": {t: s["n_shots"] for t, s in sets.items()},
        "common_bins": {t: s["bins"] for t, s in common.items()},
        "common_shots": {t: s["n_shots"] for t, s in common.items()},
    }
    own = sets["all119"]["methods"][OURS]
    counts = own["counts"]
    out["crowd_share"] = counts["crowd_bins"] / (counts["tp"] + counts["fn"])
    shares = records["strata"]["clock_boundary_identity"]["shares"]
    out["clock_share"] = {
        "start": shares["start_within_1ms"],
        "end": shares["end_within_1ms"],
        "both": shares["both_within_1ms"],
        "crowd_spans": records["strata"]["clock_boundary_identity"]["crowd_spans"],
    }
    audit = records["history"]["same_day_outer_fold_audit"]
    out["review_run_days"] = {
        "days": audit["run_days"],
        "crossing": audit["run_days_spanning_multiple_folds"],
    }
    if "smith" in records:
        sm = records["smith"]["onset_run_day_audit"]
        out["smith_run_days"] = {
            "days": sm["run_days"],
            "crossing": sm["run_days_spanning_multiple_folds"],
        }
        head = records["smith"]["methods"]["elm-ours-onset"]["events"]["2"]
        win = records["smith"]["onset_window_audit"]
        out["smith_head"] = {
            "recall": head["point"]["recall"],
            "max_error_ms": win["max_absolute_error_ms"],
            "correct_cell": win["correct_1ms_cell_fraction"],
            "out_of_window": win["out_of_window_firings"],
            "windows": records["smith"]["counts"]["hand_onsets"],
            "shots": records["smith"]["shots"],
            "elmo_region_recall": records["smith"]["reimplemented_elmo_overlap"][
                "point"
            ]["recall"],
        }
    alarm = {}
    for tag in ("all119", "bes73"):
        for name, res in sets[tag]["methods"].items():
            if "absent_span_alarm_rate" in res["point"]:
                alarm[f"{tag}/{name}"] = {
                    "span_alarm": point(res, "absent_span_alarm_rate"),
                    "span_alarm_guard25": point(res, "absent_span_alarm_rate_guard25"),
                    "bin_fpr": point(res, "false_alarm_bin_rate"),
                }
    out["alarm"] = alarm
    out["paired_primary"] = {
        m: paired(sets["bes73"]["paired"][f"{OURS} - {ELMO}: {m}"])
        for m in ("auroc", "auprc", "f1")
    }
    out["paired_common_elmo"] = {
        m: paired(common["bes73"]["paired"][f"{OURS} - {ELMO}: {m}"])
        for m in ("auroc", "auprc", "f1")
    }
    out["paired_common_dsm"] = {
        tag: {
            m: paired(common[tag]["paired"][f"{OURS} - {DSM_DETECT}: {m}"])
            for m in ("auroc", "auprc", "f1")
        }
        for tag in ("all119", "bes73")
    }
    folds = records["history"]["runs"][records["history"]["reported_run"]][
        "completed_folds"
    ]
    out["ours_folds"] = [
        {"fold": f["fold"], "epoch": f["best"]["epoch"], "threshold": f["threshold"]}
        for f in folds
    ]
    ranges = records["seeds"]["seed_ranges"]
    out["seed_range"] = {
        tag: {
            m: ranges[tag][m]["original_and_three_repeats"]
            for m in ("auroc", "auprc", "f1")
        }
        for tag in ("all119", "bes73")
    }
    out["epochs"] = ours["config"]["epochs"]
    out["replicates"] = sets["bes73"]["paired"][f"{OURS} - {ELMO}: auroc"]["replicates"]
    mode = sets["all119"]["annotation_modes"]["non_crowd_only"]
    mres = mode["methods"][OURS]
    out["non_crowd_only"] = {
        "shots": mode["n_shots"],
        "bins": mode["bins"],
        "precision": point(mres, "precision"),
        "recall": point(mres, "recall"),
        "f1": point(mres, "f1"),
    }
    detect = dsm["detectors"][DSM_DETECT]["folds"]
    out["dsm_folds"] = [
        {"fold": r["fold"], "epoch": r["best_epoch"], "threshold": r["threshold"]}
        for r in detect
    ]
    native = records["native_detection"]
    out["native_detection"] = {
        "shots_at_least_90": native["shots_at_least_90_percent"],
        "panel_shots": native["sets"]["all119"]["n_shots"],
        "epochs": native["recipe"]["epochs"],
        "bes_panel_shots": native["sets"]["bes73"]["n_shots"],
        "folds": [
            {
                "fold": f["fold"],
                "train": len(f["train"]),
                "inner_val": len(f["inner_val"]),
                "test": len(f["test"]),
                "threshold": f["threshold"],
            }
            for f in native["folds"]
        ],
        "has_ours_native_folds": OURS_NATIVE in native["sets"]["all119"]["methods"],
        "ours_native_thresholds": native.get("ours_on_native_folds", {}).get(
            "thresholds"
        ),
    }
    out["ours_training_sizes"] = [
        {
            "train": len(f["train"]),
            "inner_val": len(f["inner_val"]),
            "test": len(f["test"]),
        }
        for f in folds_history(records)
    ]
    offsets = records["offsets"]
    quart = offsets["offset_ms"]
    out["offsets"] = {
        "starts": offsets["n_starts"],
        "shots": offsets["n_shots_with_non_crowd_spans"],
        "matched": offsets["n_matched"],
        "matched_shots": offsets["n_matched_shots"],
        "unmatched": offsets["n_unmatched"],
        "median": round(quart["median"]),
        "p25": round(quart["p25"]),
        "p75": round(quart["p75"]),
    }
    out["moved"] = None
    if "moved" in records:
        bes = records["moved"]["sets"]["bes73"]
        stratum = bes["strata"]["moved_spans"]
        out["moved"] = {
            "present_spans": stratum["present_spans"],
            "absent_spans": stratum["absent_spans"],
            "bins": stratum["bins"],
            "positive_bins": stratum["positive_bins"],
            "shots": stratum["shots_with_bins"],
            "positive_shots": stratum["shots_with_positive_bins"],
            "all_present_spans": bes["present_spans_with_bins"],
            "moved_present_spans": bes["present_spans_moved"],
            "methods": stratum["methods"],
            "paired": stratum["paired_ours_minus"],
        }
    out["density"] = density_facts(records.get("density_range"))
    out["photodiode"] = photodiode_facts(records.get("photodiode"))
    out["tiled"] = tiled_facts(records.get("tiled"))
    out["run_day"] = run_day_facts(records.get("run_day"))
    out["dsm_baselines"] = dsm_baseline_facts(records.get("dsm_seeds"))
    return out


def folds_history(records: dict) -> list[dict]:
    """The reported run's completed folds, each with its shot counts."""
    history = records["history"]
    return history["runs"][history["reported_run"]]["completed_folds"]


def density_facts(record: dict | None) -> dict | None:
    """Numerical range of the fast-density input, from `density_range.json`."""
    if not record:
        return None
    return {
        "median": record["median"],
        "p05": record["p05"],
        "p95": record["p95"],
        "upper_clip_share": record["share_at_upper_clip"],
        "clip_range": record["clip_range"],
        "window_ms": record["window_ms"],
        "shots": record["shots"],
        "chords": record["chords"],
        "chords_zeroed": record["chords_zeroed_or_empty"],
    }


def photodiode_facts(record: dict | None) -> dict | None:
    """Upstream export against a fresh fetch of the swap shots' PCPHD02/03."""
    if not record:
        return None
    summary = record["summary"]
    return {
        "records": summary["records"],
        "identical": summary["identical_samples"],
        "max_value_difference": summary["max_abs_value_difference"],
        "shots": len(record["shots"]),
    }


def tiled_facts(record: dict | None) -> dict | None:
    """Whole-shot against tiled inference, all119 AUROC, from `tiled_inference.json`."""
    if not record:
        return None
    body = record["sets"]["all119"]
    whole, tiled = (body["methods"][n] for n in body["methods"])
    change = body["paired_whole_minus_tiled"]["auroc"]
    return {
        "tile_ms": record["tile_ms"],
        "whole_auroc": whole["point"]["auroc"],
        "tiled_auroc": tiled["point"]["auroc"],
        "whole_minus_tiled": (change["value"], change["ci95"]),
        "serving_threshold": record["serving_threshold"],
        "fold_thresholds": list(record["fold_thresholds"].values()),
    }


def run_day_facts(record: dict | None) -> dict | None:
    """Run-day-grouped CV figures for `elm-ours`, from `run_day_cv.json`."""
    if not record:
        return None
    out = {
        "days": record["run_days"],
        "crossing": record["run_days_spanning_multiple_folds"],
        "folds": record["folds"],
    }
    for tag, body in record["sets"].items():
        res = body["methods"][OURS]
        out[tag] = {m: point(res, m) for m in ("auroc", "auprc", "f1")}
        out[tag]["headline"] = body["headline_elm_ours"]
    out["auroc"] = out["all119"]["auroc"]
    return out


def dsm_baseline_facts(record: dict | None) -> dict | None:
    """Mean and range over the post-warm-up DSM detection repeats (all119 panel)."""
    if not record:
        return None
    out = {"rule": record["protocol"]["rule"], "seeds": record["protocol"]["seeds"]}
    for key, name, label in (
        ("reduced_60_input", DSM_DETECT, "reduced"),
        ("native_124_input", DSM_NATIVE, "native"),
    ):
        if key not in record:
            continue
        agg = record[key]["aggregate"]
        row = {"n_shots": {t: agg[t]["n_shots"] for t in agg}}
        for tag in ("all119", "bes73"):
            if tag in agg and name in agg[tag]["methods"]:
                row[tag] = {
                    m: {
                        k: agg[tag]["methods"][name][m][k]
                        for k in ("mean", "min", "max")
                    }
                    for m in ("auroc", "auprc", "f1")
                }
        folds = [f for seed in record[key]["folds_by_seed"] for f in seed["folds"]]
        epochs = [f["selected_epoch"] for f in folds]
        thresholds = [f["threshold"] for f in folds]
        short = key.split("_")[0]
        row["total_epochs"] = record["protocol"][f"{short}_epochs"]
        row["warmup_epochs"] = record["protocol"][f"{short}_warmup_epochs"]
        row["epochs"] = (min(epochs), max(epochs))
        row["thresholds"] = (min(thresholds), max(thresholds))
        row["reported"] = {
            tag: {m: agg[tag]["reported_single_run"][name][m] for m in METRIC_KEYS}
            for tag in ("all119", "bes73")
            if tag in agg and name in agg[tag]["methods"]
        }
        row["auroc"] = row["all119"]["auroc"]
        row["n_shots"] = row["n_shots"]["all119"]
        row["bins"] = agg["all119"]["bins"]
        out[label] = row
    return out
