#!/usr/bin/env python
"""Record covering-test/Ruff evidence and unchanged primary ELM benchmark metrics."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from labeler.config import git_sha

REPO = Path(__file__).resolve().parents[2]
BASE = "eb4e3630"
OUT = REPO / "outputs/labeler/elm/fix_round2_verification.json"
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
TESTS = [
    f"tests/labeler/test_elm_{name}.py"
    for name in (
        "ours",
        "dsm_swap",
        "audit",
        "provenance",
        "dsm_fix",
        "clock_onsets",
        "inputs",
        "input_metadata",
        "nonfinite",
        "dsm_adapter",
        "train_stability",
    )
]


def changed_files(added_only=False):
    command = ["git", "diff", BASE, "--name-only"]
    if added_only:
        command.append("--diff-filter=A")
    command += ["--", "*.py"]
    files = subprocess.check_output(command, cwd=REPO, text=True).splitlines()
    files += subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "*.py"],
        cwd=REPO,
        text=True,
    ).splitlines()
    return sorted(set(files) | {"scripts/labeler/elm_fix_verify.py"})


def main():
    commands = {
        "tests": [
            "bash",
            "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh",
            str(REPO),
            *TESTS,
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        "ruff": [*PIXI, "ruff", "check", *changed_files()],
        "new_file_format": [*PIXI, "ruff", "format", "--check", *changed_files(True)],
    }
    record = {"git": git_sha(full=True), "created": datetime.now(UTC).isoformat()}
    for name, command in commands.items():
        result = subprocess.run(
            command, cwd=REPO, text=True, capture_output=True, check=False
        )
        output = result.stdout + result.stderr
        record[name] = {
            "command": command,
            "exit_code": result.returncode,
            "output": output,
        }
        print(name, result.returncode, output[-1200:], flush=True)
    path = "outputs/labeler/elm/ours/evaluation.json"
    old = json.loads(
        subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=REPO)
    )
    new = json.loads((REPO / path).read_text())
    unchanged = {}
    for tag, subset in old["sets"].items():
        unchanged[tag] = subset["bins"] == new["sets"][tag]["bins"]
        for method, result in subset["methods"].items():
            for metric, value in result["point"].items():
                renamed = metric.replace(
                    "individual_span_recall", "non_crowd_span_touch_recall"
                )
                current = new["sets"][tag]["methods"][method]["point"][renamed]
                unchanged[f"{tag}.{method}.{renamed}"] = math.isclose(
                    value, current, rel_tol=0, abs_tol=1e-14
                ) or (math.isnan(value) and math.isnan(current))
                if metric in result["ci95"]:
                    unchanged[f"{tag}.{method}.{renamed}.ci95"] = all(
                        math.isclose(a, b, rel_tol=0, abs_tol=1e-14)
                        or (math.isnan(a) and math.isnan(b))
                        for a, b in zip(
                            result["ci95"][metric],
                            new["sets"][tag]["methods"][method]["ci95"][renamed],
                            strict=True,
                        )
                    )
    record["primary_benchmark_unchanged"] = {
        "base_revision": BASE,
        "checks": unchanged,
        "absolute_tolerance": 1e-14,
        "all_equivalent_within_tolerance": all(unchanged.values()),
    }
    readme = "data/events/edge_localized_mode/README.md"
    before = subprocess.check_output(
        ["git", "show", f"{BASE}:{readme}"], cwd=REPO, text=True
    )
    stable_before = [line for line in before.splitlines() if "**stable**:" in line]
    stable_after = [
        line
        for line in (REPO / readme).read_text().splitlines()
        if "**stable**:" in line
    ]
    aliases = {"d3d_elm_time_to_event_dsm": "elm-dsm"}
    canonical_before = [
        line.replace(old, new) for line in stable_before for old, new in aliases.items()
    ]
    record["readme_stable_pointer"] = {
        "before": stable_before,
        "after": stable_after,
        "same_adapter_with_user_facing_alias": all(
            "elm-dsm" in line for line in canonical_before + stable_after
        )
        and bool(stable_before)
        and bool(stable_after),
    }
    record["passed"] = (
        all(record[k]["exit_code"] == 0 for k in commands)
        and all(unchanged.values())
        and record["readme_stable_pointer"]["same_adapter_with_user_facing_alias"]
    )
    OUT.write_text(json.dumps(record, indent=1))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
