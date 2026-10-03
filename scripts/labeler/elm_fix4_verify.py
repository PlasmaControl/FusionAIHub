#!/usr/bin/env python
"""Record covering tests, scoped lint and artifact checks for ELM fix round four."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from labeler.config import git_sha, sha256_of

REPO = Path(__file__).resolve().parents[2]
BASE = "8e166437"
OUT = REPO / "outputs/labeler/elm"
TESTS = (
    "tests/labeler/test_elm_audit.py",
    "tests/labeler/test_elm_bootstrap.py",
    "tests/labeler/test_elm_coverage.py",
    "tests/labeler/test_elm_dsm_adapter.py",
    "tests/labeler/test_elm_dsm_fix.py",
    "tests/labeler/test_elm_dsm_swap.py",
    "tests/labeler/test_elm_feature.py",
    "tests/labeler/test_elm_nonfinite.py",
    "tests/labeler/test_elm_ours.py",
    "tests/labeler/test_elm_tables.py",
    "tests/labeler/test_elm_train_stability.py",
    "tests/test_elm_smith.py",
)
PIXI = [
    "pixi",
    "run",
    "--frozen",
    "--no-install",
    "--manifest-path",
    "/scratch/gpfs/nc1514/FusionAIHub/pyproject.toml",
    "-e",
    "labelmaker",
]


def run(command):
    started = time.monotonic()
    result = subprocess.run(
        command, cwd=REPO, text=True, capture_output=True, check=False
    )
    return {
        "command": command,
        "exit_code": result.returncode,
        "seconds": time.monotonic() - started,
        "output_tail": (result.stdout + result.stderr)[-12000:],
    }


def main():
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE], cwd=REPO, text=True
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=REPO, text=True
    ).splitlines()
    python = sorted({p for p in changed + untracked if p.endswith(".py")})
    new = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=A", BASE], cwd=REPO, text=True
    ).splitlines()
    new_python = sorted({p for p in new + untracked if p.endswith(".py")})
    steps = {
        "tests": run(
            [
                "bash",
                "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh",
                str(REPO),
                *TESTS,
                "-q",
                "-p",
                "no:cacheprovider",
            ]
        ),
        "ruff": run([*PIXI, "ruff", "check", *python]),
        "format_new": run([*PIXI, "ruff", "format", "--check", *new_python]),
    }
    records = {}
    for name in (
        "ours/evaluation.json",
        "ours/feature_only.json",
        "ours/smoothed_selection.json",
        "ours/annotation_strata.json",
        "dsm/evaluation.json",
        "dsm/native_evaluation.json",
        "dsm/detection_input_audit.json",
        "dsm/detection_input_comparison.json",
        "dsm/reproducibility.json",
        "smith/evaluation.json",
        "swap/evaluation.json",
        "tables.json",
        "protocol.json",
        "table_render_verification.json",
    ):
        path = OUT / name
        records[name] = {"sha256": sha256_of(path), "bytes": path.stat().st_size}
    renders = json.loads((OUT / "table_render_verification.json").read_text())
    all_tex = list(OUT.rglob("*.tex"))
    assertions = {
        "canonical_records_under_2MB": all(
            r["bytes"] <= 2_000_000 for r in records.values()
        ),
        "two_main_panels": (OUT / "table_elm_benchmark.tex")
        .read_text()
        .count("common bins")
        == 2,
        "all_table_pages_viewed": renders["all_pages_viewed"],
        "no_latex_warnings": renders["warning_count"] == 0,
        "every_caption_has_domain_note": all(
            "WPQH phases" in p.read_text()
            for p in all_tex
            if "\\caption{" in p.read_text()
        ),
        "no_exact_export_paper_panel": "exact native exports"
        not in (OUT / "appendix/table_elm_dsm_native.tex").read_text(),
        "independent_smith_zero_shot_overlap": json.loads(
            (OUT / "smith/evaluation.json").read_text()
        )["protocol"]["independence"]["shot_overlap"]
        == [],
        "independent_smith_zero_run_day_overlap": json.loads(
            (OUT / "smith/evaluation.json").read_text()
        )["protocol"]["independence"]["run_day_overlap"]
        == [],
    }
    record = {
        "git": git_sha(full=True),
        "base": BASE,
        "checks": steps,
        "assertions": assertions,
        "records": records,
        "script_sha256": sha256_of(__file__),
    }
    record["passed"] = all(s["exit_code"] == 0 for s in steps.values()) and all(
        assertions.values()
    )
    (OUT / "fix_round4_verification.json").write_text(
        json.dumps(record, indent=1) + "\n"
    )
    for key, step in steps.items():
        print(key, step["exit_code"], step["output_tail"], flush=True)
    print("assertions", assertions)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
