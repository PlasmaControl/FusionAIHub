#!/usr/bin/env python
"""Refresh current ELM record audit revisions without rewriting training history."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from labeler.config import Paths, git_sha, sha256_of

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm"
CURRENT = (
    "ours/evaluation.json",
    "ours/feature_only.json",
    "ours/smoothed_selection.json",
    "smith/evaluation.json",
    "dsm/evaluation.json",
    "dsm/native_evaluation.json",
    "dsm/detection_input_audit.json",
    "dsm/detection_input_comparison.json",
    "dsm/detection_input_fetch.json",
    "dsm/density_source_units.json",
    "dsm/reproducibility.json",
    "swap/evaluation.json",
    "ours/annotation_strata.json",
    "filterscope_metadata.json",
    "density_units.json",
)


def main():
    revision = git_sha(full=True)
    code_paths = sorted((REPO / "src/labeler/elm").glob("*.py"))
    code_paths += sorted((REPO / "scripts/labeler").glob("elm_*.py"))
    hashes = {str(p.relative_to(REPO)): sha256_of(p) for p in code_paths}
    audit = {
        "git": revision,
        "created": datetime.now(UTC).isoformat(),
        "meaning": "git identifies current record audit/evaluation code, not "
        "a retroactive claim about original training; embedded training-time "
        "and retrospective source revisions/hashes are retained",
        "code_sha256": hashes,
        "records": {},
    }
    for name in CURRENT:
        path = OUT / name
        if not path.exists():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text())
        old = record.get("git")
        if old and old != revision:
            record.setdefault("generation_git", old)
        record["git"] = revision
        record["audit_git_meaning"] = (
            "current code/record audit; historical training provenance remains "
            "embedded and is not rewritten"
        )
        if "source_evaluation" in record:
            record["source_evaluation_sha256"] = sha256_of(
                Path(record["source_evaluation"])
            )
        if name == "ours/annotation_strata.json":
            sensitivity = record["checkpoint_selection_audit"][
                "smoothed_selection_sensitivity"
            ]
            sensitivity["evaluation_record_sha256"] = sha256_of(
                Path(sensitivity["evaluation_record"])
            )
        path.write_text(json.dumps(record, indent=1) + "\n")
        audit["records"][name] = {"sha256": sha256_of(path), "git": revision}
    figure = Paths.from_env().root / "round4/elm/figures/fig_elm_examples.json"
    record = json.loads(figure.read_text())
    record["git"] = revision
    record["visually_inspected"] = True
    record["inspection"] = (
        "Panel b shot 203941 shows burst activity; labels are occupancy, "
        "not independently adjudicated physical events. At 7-inch width axes "
        "and legend are readable; the former drop-shaped panel is replaced."
    )
    record["png_sha256"] = sha256_of(Path(record["png"]))
    record["pdf_sha256"] = sha256_of(Path(record["figure"]))
    figure.write_text(json.dumps(record, indent=1) + "\n")
    audit["figure"] = {"path": str(figure), "sha256": sha256_of(figure)}
    (OUT / "provenance.json").write_text(json.dumps(audit, indent=1) + "\n")
    print("current audit revision", revision)


if __name__ == "__main__":
    main()
