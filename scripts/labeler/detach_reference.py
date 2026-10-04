#!/usr/bin/env python
"""Score a frozen, small literature reference and a separate manual-front check.

Reads the current bins and labels under $LABELER_ROOT/round4/detach. Published
state statements are kept in detachment_reference_sources.json; transition-only
statements are never expanded into synthetic ground-truth intervals. Missing
shots, invalid indicators and uncertain consensus labels remain abstentions in
the reference denominator. The owner-selected front points check DZ only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from detach_json import dumps

from labeler.events.detachment import core, thresholds

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs" / "labeler" / "results"
SOURCE = RESULTS / "detachment_reference_sources.json"
STATES = {"attached": 1, "detached": 2, "marfe": 3}
NAMES = {0: "abstain", **{v: k for k, v in STATES.items()}}
METHODS = ("consensus", "afrac", "prad", "tangtv")


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def fingerprint(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def metrics(table: np.ndarray, reference_classes=None) -> dict:
    """A 3 x 4 table: truth rows, predictions 1..3 plus abstention column."""
    n_ref = float(table.sum())
    cast = table[:, :3]
    n_vote = float(cast.sum())
    n_correct = float(np.trace(cast))
    if reference_classes is None:
        reference_classes = tuple(k for k in range(3) if table[k].sum())
    f1 = {}
    for k, name in enumerate(STATES):
        denominator = table[k].sum() + cast[:, k].sum()
        f1[name] = float(2 * cast[k, k] / denominator) if table[k].sum() else None
    supported = [f1[list(STATES)[k]] for k in reference_classes]
    expected = float(cast.sum(axis=1) @ cast.sum(axis=0)) / n_vote**2 if n_vote else 1
    kappa = (
        float((n_correct / n_vote - expected) / (1 - expected))
        if n_vote and expected < 1 - 1e-12
        else None
    )
    binary = np.array(
        [
            [table[0, 0], table[0, 1:3].sum(), table[0, 3]],
            [table[1:, 0].sum(), table[1:, 1:3].sum(), table[1:, 3].sum()],
        ]
    )
    binary_f1 = {}
    for k, name in enumerate(("attached", "not_attached")):
        denominator = binary[k].sum() + binary[:, k].sum()
        binary_f1[name] = (
            float(2 * binary[k, k] / denominator) if binary[k].sum() else None
        )
    return {
        "n_reference_points": int(n_ref),
        "n_cast_votes": int(n_vote),
        "n_abstentions": int(n_ref - n_vote),
        "vote_coverage": n_vote / n_ref if n_ref else None,
        "strict_accuracy_all_references": n_correct / n_ref if n_ref else None,
        "agreement_on_cast_votes": n_correct / n_vote if n_vote else None,
        "kappa_on_cast_votes": kappa,
        "f1_with_abstentions_as_false_negatives": f1,
        "macro_f1_supported_states": float(np.mean(supported))
        if supported and all(v is not None for v in supported)
        else None,
        "confusion_truth_by_prediction": table.astype(int).tolist(),
        "binary_attached_vs_not_attached": {
            "strict_accuracy_all_references": float(np.trace(binary[:, :2]) / n_ref)
            if n_ref
            else None,
            "f1_with_abstentions_as_false_negatives": binary_f1,
            "confusion_truth_by_prediction": binary.astype(int).tolist(),
        },
    }


def bootstrap(tables: np.ndarray, seed: int, replicates: int) -> dict:
    """Resample entire shots, retaining missing predictions in each denominator."""
    rng = np.random.default_rng(seed)
    total = tables.sum(axis=0)
    reference_classes = tuple(k for k in range(3) if total[k].sum())
    values = {
        name: []
        for name in (
            "vote_coverage",
            "strict_accuracy_all_references",
            "agreement_on_cast_votes",
            "kappa_on_cast_votes",
            "macro_f1_supported_states",
            "f1_attached",
            "f1_detached",
            "f1_marfe",
            "binary_f1_attached",
            "binary_f1_not_attached",
        )
    }
    if len(tables) >= 2:
        for _ in range(replicates):
            resampled = tables[rng.integers(len(tables), size=len(tables))].sum(axis=0)
            score = metrics(resampled, reference_classes)
            score.update(
                {
                    f"f1_{key}": value
                    for key, value in score[
                        "f1_with_abstentions_as_false_negatives"
                    ].items()
                }
            )
            score.update(
                {
                    f"binary_f1_{key}": value
                    for key, value in score["binary_attached_vs_not_attached"][
                        "f1_with_abstentions_as_false_negatives"
                    ].items()
                }
            )
            for name, items in values.items():
                value = score[name]
                if value is not None and np.isfinite(value):
                    items.append(value)
    return {
        "unit": "shot",
        "replicates": replicates if len(tables) >= 2 else 0,
        "seed": seed,
        "n_shots": len(tables),
        "reference_classes": [list(STATES)[k] for k in reference_classes],
        "missing_class_support": "class F1 undefined; macro requires every frozen class",
        "warning": (
            f"Only {len(tables)} reference shots; intervals are descriptive and "
            "cannot support population-accuracy claims."
        ),
        "ci95": {
            name: {
                "interval": np.quantile(items, [0.025, 0.975]).tolist()
                if items
                else None,
                "valid_replicates": len(items),
            }
            for name, items in values.items()
        },
    }


def published_points(source: dict, bins_dir: Path, labels: pd.DataFrame) -> list:
    rows = []
    for record in source["records"]:
        if not record["score_primary"]:
            continue
        shot, time = record["shot"], record["time_ms"]
        path = bins_dir / f"{shot}.npz"
        item = {
            "reference_id": record["id"],
            "shot": shot,
            "reference_time_ms": time,
            "truth": STATES[record["state"]],
            "truth_name": record["state"],
            "source": record["source"],
            "digest_lines": record["digest_lines"],
            "primary_state": "unassessed",
            "primary_state_code": 0,
            "prediction": {},
        }
        index = None
        bins = {}
        if path.is_file():
            with np.load(path) as f:
                bins = {key: f[key] for key in f.files}
            width = (
                float(np.median(np.diff(bins["start_ms"])))
                if len(bins["start_ms"]) > 1
                else core.BIN_MS
            )
            hits = np.flatnonzero(
                (bins["start_ms"] <= time) & (time < bins["start_ms"] + width)
            )
            if len(hits):
                index = int(hits[0])
                item["start_ms"] = float(bins["start_ms"][index])
                item["bin_ms"] = width
                item["bins_source"] = fingerprint(path)
                item["gate_diagnostics"] = {
                    key: bins[key][index].item() if key in bins else None
                    for key in (
                        "aux_greenwald_fraction",
                        "greenwald_source",
                        "tangtv_source",
                        "tangtv_efit_source",
                        "tangtv_marfe_efit_source",
                        "tangtv_marfe_candidate",
                        "tangtv_marfe_spatial",
                        "tangtv_marfe_second_cue",
                        "tangtv_tier",
                        "aux_elm_known",
                        "aux_elm_share",
                        "afrac_method",
                    )
                }
                item["gate_diagnostics"]["consensus_tier"] = "bin_not_assessed"
        for method in METHODS:
            value, valid, reason, vote = None, False, "no_containing_bin", 0
            if index is not None and method != "consensus":
                valid = bool(bins[f"{method}_valid"][index])
                raw_vote = int(bins[f"{method}_vote"][index])
                vote = raw_vote if valid and raw_vote in (1, 2, 3) else 0
                value = float(bins[f"{method}_value"][index])
                reason = str(bins[f"{method}_reason"][index])
                if valid and vote == 0:
                    reason = "valid_measurement_abstains"
            elif index is not None:
                match = labels[
                    (labels.shot == shot)
                    & np.isclose(
                        labels.start_ms.to_numpy(float),
                        item["start_ms"],
                        rtol=0,
                        atol=1e-5,
                    )
                ]
                reason = "bin_not_assessed"
                if len(match):
                    state = int(match.iloc[0].state_rule)
                    item["primary_state_code"] = state
                    item["primary_state"] = {
                        0: "unassessed",
                        1: "attached",
                        2: "detached",
                        3: "marfe",
                        4: "uncertain",
                    }.get(state, "unrecognized")
                    item["gate_diagnostics"]["consensus_tier"] = str(
                        match.iloc[0].get("tier", "unrecorded")
                    )
                    valid = state in (1, 2, 3)
                    vote = state if valid else 0
                    reason = "" if valid else "uncertain_consensus"
            item["prediction"][method] = {
                "vote": vote,
                "vote_name": NAMES[vote],
                "measurement_valid": valid,
                "value": value,
                "reason": reason,
            }
        rows.append(item)
    return rows


def state_scores(rows: list, replicates: int) -> dict:
    shots = sorted({item["shot"] for item in rows})
    scores = {}
    for method in METHODS:
        tables = np.zeros((len(shots), 3, 4))
        for item in rows:
            vote = item["prediction"][method]["vote"]
            tables[
                shots.index(item["shot"]), item["truth"] - 1, vote - 1 if vote else 3
            ] += 1
        scores[method] = {
            "overall": metrics(tables.sum(axis=0)),
            "by_shot": {
                str(shot): metrics(table)
                for shot, table in zip(shots, tables, strict=True)
            },
            "shot_bootstrap": bootstrap(tables, 42, replicates),
        }
        for key, subset in [
            ("overall", rows),
            *[
                (str(shot), [row for row in rows if row["shot"] == shot])
                for shot in shots
            ],
        ]:
            summary = (
                scores[method]["overall"]
                if key == "overall"
                else scores[method]["by_shot"][key]
            )
            n_valid = sum(
                item["prediction"][method]["measurement_valid"] for item in subset
            )
            summary["n_valid_measurements"] = n_valid
            summary["measurement_coverage"] = n_valid / len(subset) if subset else None
    return scores


def manual_front_check(source: dict, bins_dir: Path) -> dict:
    """Compare binned manual DZ to automatic DZ, without deriving state truth."""
    directory = Path(source["manual_front_check"]["directory"])
    records, skipped, provenance = [], [], []
    for path in sorted(directory.glob("*.pkl")):
        match = re.search(r"_(\d{6})$", path.stem)
        if match is None:
            skipped.append({"path": str(path), "reason": "no_shot_in_filename"})
            continue
        shot = int(match.group(1))
        inv_path, bins_path = (
            root() / "inversions" / f"{shot}.npz",
            bins_dir / f"{shot}.npz",
        )
        if not inv_path.is_file() or not bins_path.is_file():
            skipped.append({"shot": shot, "reason": "missing_inversion_or_bins"})
            continue
        # This is a known owner-owned local annotation file, never an upload.
        with path.open("rb") as f:
            manual_z = np.asarray(pickle.load(f), float)
        with np.load(inv_path) as f:
            times = f["times_ms"]
            inversion_source = Path(str(f["source"]))
        with np.load(bins_path) as f:
            bins = {key: f[key] for key in f.files}
        if manual_z.ndim != 1 or len(manual_z) != len(times):
            skipped.append({"shot": shot, "reason": "annotation_frame_count_mismatch"})
            continue
        if inversion_source.stem.removesuffix("_raw") != path.stem:
            skipped.append(
                {"shot": shot, "reason": "annotation_camera_source_mismatch"}
            )
            continue
        provenance.append(
            {
                "shot": shot,
                "manual": fingerprint(path),
                "inversion": fingerprint(inv_path),
                "inversion_source_sav": str(inversion_source),
                "bins": fingerprint(bins_path),
                "n_manual_frame_points": len(manual_z),
                "time_assignment": (
                    "By inversion-frame index; exact equal frame counts required."
                ),
            }
        )
        starts = bins["start_ms"]
        width = float(np.median(np.diff(starts))) if len(starts) > 1 else core.BIN_MS
        for k, start in enumerate(starts):
            hit = (times >= start) & (times < start + width) & np.isfinite(manual_z)
            if not hit.any():
                continue
            zx, zs = float(bins["aux_zxpt1"][k]), float(bins["aux_zvsod"][k])
            ze = float(np.median(manual_z[hit]))
            manual_dz = (
                1 - (zx - ze) / (zx - zs) if zx - zs >= thresholds.MIN_LEG_M else np.nan
            )
            auto_dz = float(bins["tangtv_value"][k])
            valid = bool(bins["tangtv_valid"][k] and np.isfinite(manual_dz + auto_dz))
            records.append(
                {
                    "shot": shot,
                    "start_ms": float(start),
                    "n_manual_frame_points": int(hit.sum()),
                    "manual_ze_m": ze,
                    "manual_dz": manual_dz,
                    "automatic_dz": auto_dz,
                    "paired_valid": valid,
                    "automatic_reason": str(bins["tangtv_reason"][k]),
                }
            )
    paired = [row for row in records if row["paired_valid"]]
    error = np.asarray([row["automatic_dz"] - row["manual_dz"] for row in paired])
    return {
        "purpose": (
            "Numerical DZ check only; these annotations are not independent "
            "state truth."
        ),
        "provenance": provenance,
        "n_annotation_files": len(provenance),
        "n_annotated_bins": len(records),
        "n_paired_valid_bins": len(paired),
        "paired_valid_fraction": len(paired) / len(records) if records else None,
        "n_paired_shots": len({row["shot"] for row in paired}),
        "mae_dz": float(np.mean(np.abs(error))) if len(error) else None,
        "rmse_dz": float(np.sqrt(np.mean(error**2))) if len(error) else None,
        "bias_dz": float(np.mean(error)) if len(error) else None,
        "ci95": None,
        "ci_reason": (
            "Only one annotated shot; a shot-bootstrap interval would be degenerate."
        ),
        "records": records,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sources", type=Path, default=SOURCE)
    parser.add_argument("--bins-dir", type=Path, default=root() / "bins")
    parser.add_argument("--labels", type=Path, default=root() / "labels_bins.csv.gz")
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "detachment_reference.json"
    )
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    source = json.loads(args.sources.read_text())
    labels = pd.read_csv(args.labels)
    rows = published_points(source, args.bins_dir, labels)
    cohorts = pd.read_csv(REPO / "data" / "events" / "catalog" / "cohort.csv")
    split = dict(zip(cohorts.shot, cohorts.split, strict=True))
    for item in rows:
        item["cohort_split"] = split.get(item["shot"], "external_development")
    primary = [record for record in source["records"] if record["score_primary"]]
    record = {
        "reference_kind": (
            "Published state statements external to the computed consensus; "
            "literature-source overlap with threshold motivation disclosed."
        ),
        "limitations": source["scope"],
        "sources_record": fingerprint(args.sources),
        "digest_provenance": {
            name: {**item, **fingerprint(Path(item["digest"]))}
            for name, item in source["sources"].items()
        },
        "labels_record": fingerprint(args.labels),
        "bin_directory": str(args.bins_dir),
        "matching": (
            "Only the half-open bin [start_ms, start_ms+bin_ms) containing each "
            "published instant; no nearest-bin fallback or duration inference."
        ),
        "consensus_column": "state_rule",
        "n_primary_reference_points": len(primary),
        "n_primary_reference_shots": len({item["shot"] for item in primary}),
        "primary_shots": sorted({item["shot"] for item in primary}),
        "n_excluded_ambiguous_statements": len(source["records"]) - len(primary),
        "confusion_row_names": list(STATES),
        "confusion_column_names": [*STATES, "abstain"],
        "metrics_policy": (
            "Coverage uses every primary point including unavailable shots. "
            "F1 treats abstention as a false negative. Kappa is defined only on "
            "cast votes and is null for a degenerate one-class table."
        ),
        "rows": rows,
        "scores": state_scores(rows, args.replicates),
        "manual_front_check": manual_front_check(source, args.bins_dir),
        "thresholds": {
            name: getattr(thresholds, name)
            for name in (
                "AFRAC_ATTACHED_MIN",
                "AFRAC_DETACHED_MAX",
                "PRAD_ATTACHED_MAX",
                "PRAD_DETACHED_MIN",
                "DZ_ATTACHED_MAX",
                "DZ_DETACHED_MIN",
                "DZ_MARFE_MIN",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dumps(record, indent=2) + "\n")
    csv_rows = [
        {
            "reference_id": item["reference_id"],
            "shot": item["shot"],
            "time_ms": item["reference_time_ms"],
            "truth": item["truth_name"],
            "primary_state": item["primary_state"],
            "cohort_split": item["cohort_split"],
            "bin_start_ms": item.get("start_ms"),
            **{name: item["prediction"][name]["vote_name"] for name in METHODS},
        }
        for item in rows
    ]
    pd.DataFrame(csv_rows).to_csv(args.output.with_suffix(".csv"), index=False)
    print(
        dumps(
            {
                "output": str(args.output),
                "reference_points": len(rows),
                "reference_shots": record["primary_shots"],
                "consensus": record["scores"]["consensus"]["overall"],
                "manual_DZ_paired_bins": record["manual_front_check"][
                    "n_paired_valid_bins"
                ],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
