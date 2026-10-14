"""R3: rwm-brf with the ONSBRADIAL features, and an onset-detection benchmark.

Round five. ``rwm_sensor_separation.py`` found ONSBRADIAL's onset jump above the
0.65 bar against matched comparison shots, so its features join the forest here.
Four causal columns come from |ONSBRADIAL| binned to 1 ms (2018 shots only; the
2014 shots have no record and are imputed with the training-fold median like any
missing input): the 5 ms trailing mean, the 20 ms trailing maximum, the 20 ms
trailing log slope (as N1RMS's) and ``onsb_jump``, ln of the 5 ms mean over the
median 5 ms mean in [t - 120, t - 20 ms].

Forecast: one reference-split nested CV of ``rwm-brf-onsb`` (the reference
columns plus these four) on the round-4 slices, with the reference's outer shot
folds, seeds and alarm protocol, paired against the saved ``rwm-brf`` predictions
(as ``rwm_rotation_ablation.py``). ``evaluation.json`` is not touched.

Detection: a slice is positive within 20 ms either side of a merged n = 1 Hanson
onset; it is excluded from 20 ms after an onset to that onset's measured end, or
to the end of its uncertain stretch where N1RMS did not respond (``rwm_onsets``
``ENDS_VERSION`` meta: present or uncertain, not a negative), and around n = 2
onsets; other
Hanson slices are negatives. The forests (with and without the sensor columns)
train on the Hanson slices of the same outer folds, no inner tuning; the rules
read one column. AUROC and AUPRC pool the held-out Hanson slices, with 95%
campaign-stratified shot-bootstrap intervals (1,000 draws), overall and by
campaign; comparison-shot slices are scored as assumed negatives in a
sensitivity only.

    pixi run --frozen -e labeler python scripts/labeler/rwm_sensor_model.py

Writes ``outputs/labeler/rwm/sensor_model.json`` (details under
``$LABELER_ROOT/round5/rwm/``) and the predictions beside them.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import rwm_evaluate as baseline

from labeler.config import Paths, git_sha
from labeler.events import suggestions
from labeler.rwm import evaluate as ev
from labeler.rwm import features, labels, metrics
from labeler.rwm.records import load_evaluation, write_evaluation
from labeler.timebase import decimate_to_step

OUT = baseline.REPO / "outputs/labeler/rwm/sensor_model.json"
SENSOR = ("onsb_g", "onsb_max_g", "onsb_growth_per_s", "onsb_jump")
BASELINE_MS = (-120.0, -20.0)
DETECT_MS = 20.0
#: The review source whose onset ends bound the detection exclusion (v2 bounds
#: the N1RMS rise; v1's unbounded ends could come from a later excursion).
ENDS_VERSION = "v2"
DETECTORS = {
    "rwm-det-brf": {"kind": "brf", "columns": baseline.ALL},
    "rwm-det-brf-onsb": {"kind": "brf", "columns": (*baseline.ALL, *SENSOR)},
    "rwm-det-rule-n1rms-max": {"kind": "rule", "column": "n1rms_max_g"},
    "rwm-det-rule-onsb-max": {"kind": "rule", "column": "onsb_max_g"},
    "rwm-det-rule-onsb-jump": {"kind": "rule", "column": "onsb_jump"},
    "rwm-det-rule-elapsed-time": {"kind": "rule", "column": features.TIME_COLUMN},
}


def sensor_columns(slices: pd.DataFrame, directory) -> pd.DataFrame:
    """The four ONSBRADIAL columns on the slice grid; NaN without a record."""
    out = pd.DataFrame(np.nan, index=slices.index, columns=list(SENSOR))
    for shot, index in slices.groupby("shot").indices.items():
        path = directory / f"{int(shot)}.npz"
        if not path.is_file():
            continue
        with np.load(path) as npz:
            t, y = decimate_to_step(npz["t_ms"], np.abs(npz["onsbradial"]), 1.0)
        y = y[0]
        grid = slices.t_ms.to_numpy()[index]
        mean = features.trailing_mean(t, y, grid, features.RMS_WINDOW_MS)
        smooth = features.trailing_mean(t, y, t, features.RMS_WINDOW_MS)
        base = np.full(len(grid), np.nan)
        for i, at in enumerate(grid):
            before = (
                (t >= at + BASELINE_MS[0])
                & (t < at + BASELINE_MS[1])
                & np.isfinite(smooth)
            )
            if before.any():
                base[i] = np.median(smooth[before])
        with np.errstate(invalid="ignore", divide="ignore"):
            jump = np.where((base > 0) & (mean > 0), np.log(mean / base), np.nan)
        out.iloc[index, 0] = mean
        out.iloc[index, 1] = features.trailing_max(
            t, y, grid, features.GROWTH_WINDOW_MS
        )
        out.iloc[index, 2] = features.trailing_log_slope(
            t, y, grid, features.GROWTH_WINDOW_MS, features.LOG_FLOOR_G
        )
        out.iloc[index, 3] = jump
    return out


def measured_ends(paths: Paths) -> dict[int, list[tuple[float, float]]]:
    """`{shot: [(onset, end)]}` from the `rwm_onsets` `ENDS_VERSION` table's meta,
    the end being where the onset's present or uncertain stretch ends."""
    table = suggestions.table_path(
        paths, "resistive_wall_mode", "rwm_onsets", ENDS_VERSION
    )
    meta = json.loads(table.with_suffix(".meta.json").read_text())
    return {
        int(shot): [
            (float(p["t_ms"]), float(p.get("uncertain_to_ms", p["end_ms"])))
            for p in info.get("onsets", [])
            if "end_ms" in p
        ]
        for shot, info in meta["per_shot"].items()
    }


def detection_labels(table, target, other, ends) -> np.ndarray:
    """1 within DETECT_MS of a target onset, -1 in its measured aftermath or near
    an n = 2 onset, 0 otherwise on Hanson shots; comparison shots UNLABELLED."""
    label = np.full(len(table), labels.UNLABELLED, dtype=np.int8)
    for shot, index in table.groupby("shot").indices.items():
        if table.role.iloc[index[0]] != "hanson":
            continue
        t = table.t_ms.to_numpy()[index]
        value = np.zeros(len(t), dtype=np.int8)
        for onset, end in ends.get(int(shot), []):
            value[(t > onset + DETECT_MS) & (t < end)] = labels.EXCLUDED
        for onset in other.get(int(shot), []):
            value[np.abs(t - onset) <= DETECT_MS] = labels.EXCLUDED
        for onset in target.get(int(shot), []):
            value[np.abs(t - onset) <= DETECT_MS] = labels.POSITIVE
        if not target.get(int(shot)):
            value[:] = labels.EXCLUDED
        label[index] = value
    return label


def detection_scores(table, folds) -> pd.DataFrame:
    """Held-out scores of every detector on the supplied outer shot folds."""
    scores = pd.DataFrame(index=table.index)
    for name, config in DETECTORS.items():
        column = np.full(len(table), np.nan)
        for number, held in enumerate(folds):
            test = table.shot.isin(held).to_numpy()
            if config["kind"] == "brf":
                model = ev.Brf(
                    config["columns"],
                    seed=baseline.SEED + 1000 * (number + 1) + 999,
                    **baseline.FOREST,
                ).fit(table[~test])
            else:
                model = ev.Rule(config["column"])
            column[test] = model.score(table[test])
        scores[name] = column
    return scores


def _pooled(rows_by_shot, shots, name):
    score = np.concatenate([rows_by_shot[s][name] for s in shots])
    label = np.concatenate([rows_by_shot[s]["label"] for s in shots])
    return metrics.auroc(score, label), metrics.auprc(score, label)


def detection_metrics(frame, replicates) -> dict:
    """Pooled AUROC/AUPRC with campaign-stratified shot-bootstrap intervals, and the
    with-sensor minus without-sensor forest difference on the same draws."""
    rows_by_shot = {
        int(shot): {
            "label": (part.label == labels.POSITIVE).to_numpy(),
            **{name: part[name].to_numpy() for name in DETECTORS},
        }
        for shot, part in frame.groupby("shot")
    }
    strata = frame.drop_duplicates("shot").set_index("shot").campaign
    shots = sorted(rows_by_shot)
    rng = np.random.default_rng(baseline.SEED)
    draws = []
    for _ in range(replicates):
        draws.append(
            [
                int(s)
                for c in sorted(strata.unique())
                for s in rng.choice(
                    strata.index[strata == c].to_numpy(), (strata == c).sum()
                )
            ]
        )
    out = {}
    for name in (*DETECTORS, "difference"):
        if name == "difference":
            values = [
                np.subtract(
                    _pooled(rows_by_shot, sample, "rwm-det-brf-onsb"),
                    _pooled(rows_by_shot, sample, "rwm-det-brf"),
                )
                for sample in [shots, *draws]
            ]
            key = "rwm-det-brf-onsb minus rwm-det-brf"
        else:
            values = [_pooled(rows_by_shot, sample, name) for sample in [shots, *draws]]
            key = name
        values = np.asarray(values, dtype=float)
        out[key] = {
            metric: {
                "estimate": float(values[0, i]),
                "low": float(np.nanpercentile(values[1:, i], 2.5)),
                "high": float(np.nanpercentile(values[1:, i], 97.5)),
            }
            for i, metric in enumerate(("auroc", "auprc"))
        }
    return {
        "shots": len(shots),
        "positive_slices": int(sum(r["label"].sum() for r in rows_by_shot.values())),
        "slices": int(sum(len(r["label"]) for r in rows_by_shot.values())),
        "metrics": out,
    }


def _interval(metric) -> str:
    return f"{metric['estimate']:.3f} [{metric['low']:.3f}, {metric['high']:.3f}]"


def readme_lines(record: dict) -> list[str]:
    """The README Models lines of `sensor_model.json` (`rwm_tables.py` adds them)."""
    forecast, detection = record["forecast"], record["detection"]["hanson"]
    change = forecast["paired_change"]
    found = rule = detection["metrics"]
    rule_2018 = record["detection"]["hanson_2018"]["metrics"]
    base = detection["positive_slices"] / detection["slices"]
    lines = [
        (
            f"- rwm-brf-onsb | 2026_10_14 | Phase-controlled AUROC: "
            f"{_interval(forecast['phase_controlled_auroc'])} | AUROC: "
            f"{_interval(forecast['metrics']['slice_auroc'])} (primary; "
            f"phase-confounded) | AUPRC: {_interval(forecast['metrics']['slice_auprc'])}"
            f" | F1: {_interval(forecast['metrics']['slice_f1'])} | Limitation: "
            "rwm-brf plus four ONSBRADIAL columns, which exist on the 2018 shots only "
            "(2014 imputed); paired change in phase-controlled AUROC "
            f"{_interval(change['phase_controlled_auroc'])}, slice AUROC "
            f"{_interval(change['slice_auroc'])}, F1 {_interval(change['slice_f1'])}: "
            "the sensor adds no measurable forecast skill"
        )
    ]
    for name, clause in (
        (
            "rwm-det-brf",
            (
                "detection benchmark, onset ±20 ms on held-out Hanson slices (the "
                "reference folds, no tuning); single columns: N1RMS 20 ms maximum "
                f"{rule['rwm-det-rule-n1rms-max']['auroc']['estimate']:.3f}, elapsed "
                f"time {rule['rwm-det-rule-elapsed-time']['auroc']['estimate']:.3f}; "
                "the ranking may draw on post-onset βN and N1RMS changes, none "
                "RWM-specific"
            ),
        ),
        (
            "rwm-det-brf-onsb",
            (
                "rwm-det-brf plus the ONSBRADIAL columns; difference "
                f"{_interval(found['rwm-det-brf-onsb minus rwm-det-brf']['auroc'])}; "
                "ONSBRADIAL's 20 ms maximum alone "
                f"{rule_2018['rwm-det-rule-onsb-max']['auroc']['estimate']:.3f} on the "
                "2018 shots"
            ),
        ),
    ):
        lines.append(
            f"- {name} | 2026_10_14 | Detection AUROC: "
            f"{_interval(found[name]['auroc'])} | AUPRC: "
            f"{_interval(found[name]['auprc'])} (positive base rate {base:.3f}) | "
            f"Limitation: {clause}"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    start = time.monotonic()
    paths = Paths.from_env()
    work = paths.root / "round5/rwm"
    work.mkdir(parents=True, exist_ok=True)
    baseline._init(paths.root / "round4/rwm/slices.parquet", args.replicates)
    target, other = baseline._STATE["onsets"], baseline._STATE["other"]
    every = {
        s: sorted(target.get(s, []) + other.get(s, []))
        for s in set(target) | set(other)
    }
    reference = load_evaluation(baseline.REPO / "outputs/labeler/rwm/evaluation.json")[
        "configs"
    ]["rwm-brf"]
    original, original_alarms, _ = baseline.replay(reference, target, every)
    original_groups = ev.shot_records(original, original_alarms, target)
    folds = [
        part.shot.unique().tolist() for _, part in original.groupby("fold", sort=True)
    ]
    slices = baseline._STATE["slices"].copy()
    slices[list(SENSOR)] = sensor_columns(slices, work / "sensor/onsbradial")
    coverage = {
        str(c): int(
            slices[(slices.campaign == c)]
            .groupby("shot")
            .onsb_g.apply(lambda v: v.notna().any())
            .sum()
        )
        for c in (2014, 2018)
    }

    # Forecast: the reference forest plus the sensor columns, paired.
    table = ev.relabel(slices, target, other, labels.HORIZON_MS)
    config = {"kind": "brf", "columns": [*baseline.ALL, *SENSOR], "comparison": False}
    oof, alarms, rules = ev.cross_validate(
        table,
        baseline.factory(config),
        target,
        explanation_onsets=every,
        seed=baseline.SEED,
        outer_shot_folds=folds,
    )
    predictions = work / "predictions_rwm-brf-onsb_seed0.parquet"
    oof.to_parquet(predictions, index=False)
    ordered = original.sort_values(["shot", "t_ms"])
    fitted = oof.sort_values(["shot", "t_ms"])
    for column in ("shot", "t_ms", "fold", "label", "label_broad"):
        if not np.array_equal(ordered[column], fitted[column]):
            raise ValueError(f"the sensor fit differs from the reference in {column}")
    print("completed the forecast CV", flush=True)
    summary, groups = baseline.summarise(oof, alarms, target, rules)
    paired = metrics.paired_bootstrap(
        groups,
        original_groups,
        ev.statistic,
        replicates=args.replicates,
        seed=baseline.SEED,
        method="basic",
    )
    paired["phase_controlled_auroc"] = ev.phase_controlled_bootstrap(
        groups, original_groups, replicates=args.replicates, seed=baseline.SEED
    )

    # Detection: onset +-20 ms on the same outer folds.
    detect = slices.copy()
    detect["label"] = detection_labels(detect, target, other, measured_ends(paths))
    scored = pd.concat([detect, detection_scores(detect, folds)], axis=1)
    scored.drop(columns=list(features.FEATURES), errors="ignore").to_parquet(
        work / "predictions_rwm-det_seed0.parquet", index=False
    )
    hanson = scored[
        (scored.role == "hanson")
        & scored.label.isin([labels.POSITIVE, labels.NEGATIVE])
    ]
    with_comparison = pd.concat(
        [hanson, scored[scored.role == "comparison"].assign(label=labels.NEGATIVE)]
    )
    detection = {
        "hanson": detection_metrics(hanson, args.replicates),
        **{
            f"hanson_{c}": detection_metrics(
                hanson[hanson.campaign == c], args.replicates
            )
            for c in (2014, 2018)
        },
        "with_comparison_negatives": detection_metrics(
            with_comparison, args.replicates
        ),
        "with_comparison_negatives_2018": detection_metrics(
            with_comparison[with_comparison.campaign == 2018], args.replicates
        ),
    }
    record = {
        "script": "scripts/labeler/rwm_sensor_model.py",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "status": "completed",
        "sensor": {
            "point": "ONSBRADIAL",
            "columns": list(SENSOR),
            "shots_with_record_by_campaign": coverage,
            "separation": "outputs/labeler/rwm/sensor_separation.json",
        },
        "forecast": {
            "model": "rwm-brf-onsb",
            "options": config,
            "predictions": str(predictions),
            "reference_predictions": reference["predictions"],
            "protocol": {
                "outer_folds": "the reference's outer shot folds",
                "inner_folds": ev.INNER_FOLDS,
                "fold_seed": baseline.SEED,
                "forest": baseline.FOREST,
                "bootstrap_replicates": args.replicates,
                "paired_direction": "with ONSBRADIAL minus the reference forest",
                "intervals": "95% basic paired shot-bootstrap, fixed predictions",
            },
            "reference_metrics": reference["metrics"],
            "reference_phase_controlled_auroc": reference.get("phase_controlled_auroc"),
            "paired_change": paired,
            **summary,
        },
        "detection": {
            "definition": {
                "positive": f"|t - o| <= {DETECT_MS:g} ms, merged n=1 Hanson onsets",
                "excluded": f"o + 20 ms to the rwm_onsets {ENDS_VERSION} measured "
                "end, or its uncertain stretch's end; "
                "|t - o| <= 20 ms around n=2 onsets; n=2-only shots",
                "negative": "other Hanson slices in the 0.5 peak-Ip window",
                "folds": "the reference's outer shot folds, no inner tuning",
                "intervals": "95% percentile, campaign-stratified shot bootstrap",
            },
            "detectors": DETECTORS,
            **detection,
        },
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    write_evaluation(baseline.clean(record), OUT, work / "sensor_model_details.json")
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
