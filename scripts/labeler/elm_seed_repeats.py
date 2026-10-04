#!/usr/bin/env python
"""Repeat ELM CV training seeds on fixed partitions and record shot-bootstrap CIs.

Run ``--train`` with the approved CUDA interpreter, then ``--aggregate`` through
the labelmaker pixi environment. Each repeat keeps cv2's outer and inner shot
partitions and changes only initialisation, dropout, crop and augmentation seeds.
Per-run predictions, evaluations and logs stay under LABELER_ROOT/round4/elm.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from labeler.config import Paths, sha256_of
from labeler.elm import train

REPO = Path(__file__).resolve().parents[2]
METRICS = ("auroc", "auprc", "f1", "precision", "recall")
DEFAULT_SEEDS = (20261004, 20261005, 20261006)
EVALUATION_SOURCES = (
    "scripts/labeler/elm_ours_evaluate.py",
    "src/labeler/config.py",
    *(
        f"src/labeler/elm/{name}.py"
        for name in (
            "compare",
            "inputs",
            "labels",
            "methods",
            "score",
            "train",
            "prepare",
            "clock_onsets",
            "onset",
            "swap",
        )
    ),
)


def read_record(path):
    return json.loads(path.read_text())


def folds_from(run_dir):
    return [read_record(run_dir / f"fold{k}" / "fold.json") for k in range(5)]


def check_partitions(source_dir, repeat_dir):
    source = read_record(source_dir / "run.json")
    repeat = read_record(repeat_dir / "run.json")
    if source["folds"] != repeat["folds"]:
        raise ValueError(f"changed outer shot partitions: {repeat_dir}")
    if {k: v for k, v in source["config"].items() if k != "seed"} != {
        k: v for k, v in repeat["config"].items() if k != "seed"
    }:
        raise ValueError(f"changed training hyperparameters: {repeat_dir}")
    first_data, next_data = (
        record["provenance"]["data"] for record in (source, repeat)
    )
    for key in ("reviewed_labels", "cohort"):
        if first_data[key]["sha256"] != next_data[key]["sha256"]:
            raise ValueError(f"changed {key} content: {repeat_dir}")
    if {row["shot"]: row["sha256"] for row in first_data["inputs"]} != {
        row["shot"]: row["sha256"] for row in next_data["inputs"]
    }:
        raise ValueError(f"changed prepared-input content: {repeat_dir}")
    for first, second in zip(folds_from(source_dir), folds_from(repeat_dir)):
        for key in ("train", "inner_val", "test"):
            if first[key] != second[key]:
                raise ValueError(f"changed {key} shots: {repeat_dir}")


def train_repeats(args, root, source_dir, source):
    config = source["config"]
    # The public CLI exposes the cv2 settings that changed from defaults; refuse
    # another protocol rather than silently substituting unexposed defaults.
    default = train.Config()
    for key in ("weight_decay", "dropout", "crop_ms", "level_shift"):
        if config[key] != getattr(default, key):
            raise ValueError(f"source {key} is not supported by the training CLI")
    inner = len(folds_from(source_dir)[0]["inner_val"])
    for seed in args.seeds:
        name = f"{args.source_run}_seed{seed}"
        repeat_dir = root / "cv" / name
        log_dir = root / "seed_repeats"
        log_dir.mkdir(parents=True, exist_ok=True)
        if (repeat_dir / "run.json").exists():
            record = read_record(repeat_dir / "run.json")
            if len(record["fold_records"]) == 5:
                if record["config"]["seed"] != seed:
                    raise ValueError(f"changed training seed: {repeat_dir}")
                check_partitions(source_dir, repeat_dir)
                continue
            raise ValueError(f"incomplete existing repeat: {repeat_dir}")
        command = [
            sys.executable,
            "-u",
            "-m",
            "labeler.elm.train",
            "--run",
            name,
            "--folds",
            "5",
            "--inner-val",
            str(inner),
            "--epochs",
            str(config["epochs"]),
            "--iters",
            str(config["iters"]),
            "--batch",
            str(config["batch"]),
            "--lr",
            str(config["lr"]),
            "--seed",
            str(source.get("partition_seed", config["seed"])),
            "--training-seed",
            str(seed),
            "--device",
            "cuda",
        ]
        print(f"training {name}; log={log_dir / (name + '.log')}", flush=True)
        with (log_dir / f"{name}.log").open("w") as log:
            subprocess.run(
                command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        check_partitions(source_dir, repeat_dir)
        subprocess.run(
            [
                "bash",
                "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/tmpsweep.sh",
            ],
            check=True,
        )


def evaluation_of(root, source_dir, name):
    repeat_dir = root / "cv" / name
    check_partitions(source_dir, repeat_dir)
    eval_dir = root / "seed_repeats" / name
    evaluation_path = eval_dir / "evaluation.json"
    metadata_path = eval_dir / "evaluation_provenance.json"
    metadata = {
        "run_record_sha256": sha256_of(repeat_dir / "run.json"),
        "sources": {path: sha256_of(REPO / path) for path in EVALUATION_SOURCES},
    }
    if not (
        evaluation_path.exists()
        and metadata_path.exists()
        and read_record(metadata_path) == metadata
    ):
        subprocess.run(
            [
                sys.executable,
                "scripts/labeler/elm_ours_evaluate.py",
                "--run",
                name,
                "--out-dir",
                str(eval_dir),
            ],
            cwd=REPO,
            check=True,
        )
        metadata_path.write_text(json.dumps(metadata, indent=1) + "\n")
    return read_record(evaluation_path), eval_dir


def aggregate(args, root, source_dir):
    results = []
    set_definitions = None
    for name in run_names(args):
        repeat_dir = root / "cv" / name
        evaluation, eval_dir = evaluation_of(root, source_dir, name)
        run = read_record(repeat_dir / "run.json")
        definitions = {
            key: {field: value[field] for field in ("shots", "n_shots", "bins")}
            for key, value in evaluation["sets"].items()
        }
        if set_definitions is not None and definitions != set_definitions:
            raise ValueError(f"changed evaluation shots or bins: {name}")
        set_definitions = definitions
        fold_records = folds_from(repeat_dir)
        selection = []
        for fold in fold_records:
            selection.append(
                {
                    "fold": fold["fold"],
                    "best": fold["best"],
                    "nonfinite_validation_epochs": {
                        metric: sum(
                            not np.isfinite(row[metric]) for row in fold["history"]
                        )
                        for metric in (
                            "val_auprc",
                            "val_auroc",
                            "val_f1",
                            "val_threshold",
                        )
                    },
                    "validation_history": [
                        {
                            field: row[field]
                            for field in (
                                "epoch",
                                "val_auprc",
                                "val_auroc",
                                "val_f1",
                                "val_threshold",
                            )
                        }
                        for row in fold["history"]
                    ],
                }
            )
        results.append(
            {
                "run": name,
                "training_seed": run["config"]["seed"],
                "is_new_repeat": name != args.source_run,
                "run_record": str(repeat_dir / "run.json"),
                "run_record_sha256": sha256_of(repeat_dir / "run.json"),
                "training_provenance_mode": run["provenance"]["mode"],
                "training_code": run["provenance"]["code"],
                "evaluation": str(eval_dir / "evaluation.json"),
                "evaluation_sha256": sha256_of(eval_dir / "evaluation.json"),
                "evaluation_sources": read_record(
                    eval_dir / "evaluation_provenance.json"
                )["sources"],
                "selected_epochs": [
                    row["best"]["epoch"] for row in run["fold_records"]
                ],
                "fold_selection": selection,
                "training_seconds": sum(
                    epoch["seconds"]
                    for fold in fold_records
                    for epoch in fold["history"]
                ),
                "event_thresholds": evaluation["event_thresholds"],
                "sets": {
                    key: {
                        "n_shots": value["n_shots"],
                        "bins": value["bins"],
                        "point": {
                            m: value["methods"]["elm-ours"]["point"][m] for m in METRICS
                        },
                        "ci95": {
                            m: value["methods"]["elm-ours"]["ci95"][m] for m in METRICS
                        },
                    }
                    for key, value in evaluation["sets"].items()
                },
            }
        )
    source = read_record(source_dir / "run.json")
    shot_lists = folds_from(source_dir)
    all_shots = {shot for fold in source["folds"] for shot in fold}
    with (Paths.from_env().catalog / "cohort.csv").open() as handle:
        cohort_splits = Counter(
            row["split"]
            for row in csv.DictReader(handle)
            if int(row["shot"]) in all_shots
        )
    if sum(cohort_splits.values()) != len(all_shots) or cohort_splits["test"]:
        raise ValueError("reviewed shots missing from cohort or include blind test")
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_script": str(Path(__file__).relative_to(REPO)),
        "source_script_sha256": sha256_of(Path(__file__)),
        "protocol": "Three new training seeds; identical outer and inner shot "
        "partitions and cv2 hyperparameters. Thresholds and best epochs "
        "are selected separately on each fixed inner-validation split.",
        "config": {k: v for k, v in source["config"].items() if k != "seed"},
        "partition_seed": source.get("partition_seed", source["config"]["seed"]),
        "folds": [
            {"fold": k, **{key: row[key] for key in ("train", "inner_val", "test")}}
            for k, row in enumerate(shot_lists)
        ],
        "cohort_test_shots_used": cohort_splits["test"],
        "cohort_split_counts": dict(cohort_splits),
        "input_content_matches_source": True,
        "sets": set_definitions,
        "ci_method": "95% percentile shot bootstrap, 1000 replicates per run; "
        "seed spread is reported separately and is not a CI.",
        "results": results,
        "seed_ranges": {},
    }
    for key in results[0]["sets"]:
        record["seed_ranges"][key] = {}
        for metric in METRICS:
            values = np.array([r["sets"][key]["point"][metric] for r in results])
            new = values[1:]
            record["seed_ranges"][key][metric] = {
                "three_new_seeds": {
                    "min": float(new.min()),
                    "max": float(new.max()),
                    "mean": float(new.mean()),
                    "sample_sd": float(new.std(ddof=1)),
                },
                "original_and_three_repeats": {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                },
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(args.out)
    print(json.dumps(record["seed_ranges"], indent=1))


def run_names(args):
    return (args.source_run, *(f"{args.source_run}_seed{seed}" for seed in args.seeds))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-run", default="cv2")
    parser.add_argument("--seeds", type=int, nargs=3, default=DEFAULT_SEEDS)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--evaluate-available",
        action="store_true",
        help="score completed runs while later repeats train",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO / "outputs/labeler/elm/ours/seed_repeats.json"
    )
    args = parser.parse_args(argv)
    if not (args.train or args.aggregate or args.evaluate_available):
        parser.error("choose --train, --aggregate and/or --evaluate-available")
    if len(set(args.seeds)) != 3:
        parser.error("three distinct new seeds are required")
    root = Paths.from_env().root / "round4" / "elm"
    source_dir = root / "cv" / args.source_run
    source = read_record(source_dir / "run.json")
    if source["config"]["seed"] in args.seeds:
        parser.error("repeat seeds must differ from the source training seed")
    if args.train:
        train_repeats(args, root, source_dir, source)
    if args.evaluate_available:
        for name in run_names(args):
            path = root / "cv" / name / "run.json"
            if path.exists() and len(read_record(path)["fold_records"]) == 5:
                evaluation_of(root, source_dir, name)
    if args.aggregate:
        aggregate(args, root, source_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
