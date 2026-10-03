#!/usr/bin/env python
"""Evaluate one smoothed-selection CV sensitivity without replacing frozen cv2."""

from __future__ import annotations

import json
import re
from pathlib import Path

from elm_ours_evaluate import main as evaluate

from labeler.config import Paths, git_sha, sha256_of

REPO = Path(__file__).resolve().parents[2]


def main():
    root = Paths.from_env().root / "round4/elm/cv"
    first = root / "cv2"
    next_run = root / "cv2-smoothed"
    source = json.loads((first / "run.json").read_text())
    target = json.loads((next_run / "run.json").read_text())
    assert source["folds"] == target["folds"]
    assert target["config"]["selection"] == "trailing3"
    assert source["config"] == {
        k: v for k, v in target["config"].items() if k != "selection"
    }
    selections = []
    for k in range(5):
        old = json.loads((first / f"fold{k}/fold.json").read_text())
        new = json.loads((next_run / f"fold{k}/fold.json").read_text())
        for role in ("train", "inner_val", "test"):
            assert old[role] == new[role]
        selections.append(
            {
                "fold": k,
                "old_epoch": old["best"]["epoch"],
                "smoothed_epoch": new["best"]["epoch"],
                "smoothed_criterion": new["best"]["selection_score"],
                "old_threshold": old["threshold"],
                "smoothed_threshold": new["threshold"],
            }
        )
    out = next_run / "evaluation"
    evaluate(["--run", "cv2-smoothed", "--out-dir", str(out)])
    metrics = json.loads((out / "evaluation.json").read_text())
    primary = json.loads(
        (REPO / "outputs/labeler/elm/ours/evaluation.json").read_text()
    )
    record = {
        "git": git_sha(),
        "role": "development sensitivity; primary cv2 and frozen Smith "
        "ensemble unchanged; no selection based on these held-out scores",
        "selection": "trailing 3-epoch mean inner-validation AUPRC, each epoch "
        "after ceil(25*0.15)=4 warm-up epochs; retain current endpoint weights; "
        "threshold uses its inner-validation F1 optimum",
        "recipe_difference": ["selection"],
        "folds_and_inner_partitions_identical": True,
        "selections": selections,
        "sets": metrics["sets"],
        "primary_sets": primary["sets"],
        "sources": {
            str(p): sha256_of(p)
            for p in (
                first / "run.json",
                next_run / "run.json",
                out / "evaluation.json",
            )
        },
    }
    gpu_log = root.parent / "smoothed_train.log"
    log = gpu_log.read_text()
    peak = re.search(r"max_allocated_bytes (\d+)", log)
    record["gpu"] = {
        "visible_device": 1,
        "allocator_cap_fraction": 0.28,
        "max_allocated_bytes": int(peak[1]),
        "log": str(gpu_log),
        "log_sha256": sha256_of(gpu_log),
    }
    path = REPO / "outputs/labeler/elm/ours/smoothed_selection.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print("sensitivity", path)


if __name__ == "__main__":
    main()
