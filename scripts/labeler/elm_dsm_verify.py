#!/usr/bin/env python
"""Verify repaired DSM OOF scores from checkpoints and optimizer-only statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import compare, dsm, train

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/dsm/reproducibility.json"


def main():
    paths = Paths.from_env()
    work = paths.root / "round4/elm/dsm"
    fits = json.loads((work / "fits.json").read_text())
    metadata = fits["detection_input_repair"]
    detector = fits["detectors"][compare.NAME["detect"]]
    data = train.load(paths)
    spans = {s: value.spans for s, value in data.items()}
    raw = {
        s: dsm.load_rows(s, Path(metadata["raw_row_store"]) / f"{s}.npz") for s in data
    }
    if any(row is None for row in raw.values()):
        raise ValueError("a repaired raw row artifact is missing")
    torch.set_num_threads(4)
    with np.load(detector["scores"]) as saved:
        scores = {int(k[1:]): saved[k] for k in saved.files}
    thresholds = json.loads((work / "thresholds.json").read_text())[
        compare.NAME["detect"]
    ]
    results = []
    for fold in detector["folds"]:
        norm = dsm.fit_detection_normalization(raw, spans, fold["train_shot_ids"])
        disjoint = not (
            set(fold["train_shot_ids"])
            & (set(fold["inner_val_shot_ids"]) | set(fold["test_shot_ids"]))
        )
        model = dsm.Detector(dropout=fits["detector_config"]["dropout"])
        model.load_state_dict(torch.load(fold["checkpoint"], weights_only=True))
        errors = {}
        for shot in fold["test_shot_ids"]:
            normalized = dsm.normalize_detection_rows(raw[shot], norm)
            actual = dsm.predict(model, normalized.x)
            errors[str(shot)] = float(np.max(np.abs(actual - scores[shot])))
        row_digests = all(
            dsm.rows_digest(raw[int(shot)]) == digest
            for shot, digest in fold["raw_rows_sha256"].items()
        )
        result = {
            "fold": fold["fold"],
            "test_shots": fold["test_shot_ids"],
            "normalization_recomputed_identically": norm == fold["normalization"],
            "normalization_excludes_validation_and_test": disjoint,
            "all_raw_row_digests_match": row_digests,
            "checkpoint_hash_matches": sha256_of(fold["checkpoint"])
            == fold["checkpoint_sha256"],
            "thresholds_match": all(
                float(thresholds[str(s)]) == fold["threshold"]
                for s in fold["test_shot_ids"]
            ),
            "maximum_score_error": max(errors.values()),
            "per_shot_maximum_score_error": errors,
        }
        result["passed"] = (
            all(
                result[k]
                for k in (
                    "normalization_recomputed_identically",
                    "normalization_excludes_validation_and_test",
                    "all_raw_row_digests_match",
                    "checkpoint_hash_matches",
                    "thresholds_match",
                )
            )
            and result["maximum_score_error"] <= 1e-7
        )
        results.append(result)
    record = {
        "git": git_sha(full=True),
        "script_sha256": sha256_of(__file__),
        "verification_scope": "Recompute optimizer-training-only normalization and "
        "checkpoint forward passes on every outer-test shot; compare with the "
        "serialized score arrays consumed by the new evaluation. No retraining "
        "or repeated bootstrap. Historical input-poor rescore proof superseded.",
        "fit_record": str(work / "fits.json"),
        "fit_record_sha256": sha256_of(work / "fits.json"),
        "scores": detector["scores"],
        "score_sha256": sha256_of(detector["scores"]),
        "folds": results,
        "passed": all(row["passed"] for row in results),
    }
    OUT.write_text(json.dumps(record, indent=1))
    print(
        json.dumps(
            {
                "passed": record["passed"],
                "folds": len(results),
                "maximum_score_error": max(r["maximum_score_error"] for r in results),
            }
        )
    )
    if not record["passed"]:
        raise RuntimeError("repaired DSM checkpoint verification failed")


if __name__ == "__main__":
    main()
