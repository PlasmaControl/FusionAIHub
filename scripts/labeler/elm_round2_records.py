#!/usr/bin/env python
"""Archive superseded ELM records and verify retained primary scientific values."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from labeler.config import Paths, git_sha, sha256_of

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs/labeler/elm"
BASE = "eb4e3630"


def same(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-14) or (
        math.isnan(a) and math.isnan(b)
    )


def main():
    archive = Paths.from_env().root / "round4/elm/archive/fix_round2"
    archive.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUTS / "round2_records.json"
    previous = json.loads(result_path.read_text()) if result_path.exists() else {}
    archived = previous.get("archived_records", {})
    duplicate_names = (
        "evaluation_before_phase_fix.json",
        "evaluation_trained.json",
        "evaluation_trained_before_phase_fix.json",
        "prefetch_evaluation.json",
        "prefetch_evaluation_before_phase_fix.json",
        "reproducibility_before_phase_fix.json",
        "own_target_correction.json",
    )
    files = [OUTPUTS / "dsm" / name for name in duplicate_names]
    files += [
        OUTPUTS / "swap/prefetch_evaluation.json",
        OUTPUTS / "ours/prefix_evaluation.json",
    ]
    for path in files:
        if not path.exists():
            continue
        target = archive / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256_of(path)
        if target.exists() and sha256_of(target) != digest:
            raise ValueError(f"archive would overwrite different bytes: {target}")
        shutil.move(str(path), str(target))
        archived[str(path.relative_to(REPO))] = {
            "path": str(target),
            "sha256": digest,
        }
    checks = {}
    for component in ("ours", "dsm"):
        path = f"outputs/labeler/elm/{component}/evaluation.json"
        original = json.loads(
            subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=REPO)
        )
        current = json.loads((REPO / path).read_text())
        for tag, subset in original["sets"].items():
            checks[f"{component}.{tag}.bins"] = (
                subset["bins"] == current["sets"][tag]["bins"]
            )
            for method, res in subset["methods"].items():
                new = current["sets"][tag]["methods"][method]
                for metric, value in res["point"].items():
                    key = f"{component}.{tag}.{method}.{metric}"
                    checks[key] = same(value, new["point"][metric])
                    if metric in res["ci95"]:
                        checks[key + ".ci95"] = all(
                            same(a, b)
                            for a, b in zip(
                                res["ci95"][metric], new["ci95"][metric], strict=True
                            )
                        )
    record = {
        "git": git_sha(full=True),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_revision": BASE,
        "archived_records": archived,
        "canonical_dsm_record": "outputs/labeler/elm/dsm/evaluation.json",
        "canonical_scope": "One current consolidated evaluation of all DSM variants; "
        "superseded snapshots remain outside git with hashes.",
        "retained_bin_metrics_and_intervals": checks,
        "all_retained_values_unchanged": all(checks.values()),
    }
    result_path.write_text(json.dumps(record, indent=1))
    print(
        "archived", len(archived), "retained checks", len(checks), all(checks.values())
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
