#!/usr/bin/env python
"""Record scoped verification and unchanged ELM occupancy bin metrics for fix 3."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from labeler.config import git_sha, sha256_of

REPO = Path(__file__).resolve().parents[2]
BASE = "aa4ccbf2"
OUT = REPO / "outputs/labeler/elm/fix_round3_verification.json"
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
        "coverage",
        "dsm_swap",
        "audit",
        "dsm_fix",
        "dsm_adapter",
        "tables",
        "train_stability",
        "input_metadata",
        "provenance",
    )
]


def changed_python(added_only=False):
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
    return sorted(set(files))


def artifact_checks():
    checks = {}
    protocol = json.loads((REPO / "outputs/labeler/elm/protocol.json").read_text())
    for name, entry in protocol["sources"].items():
        checks[f"protocol.source.{name}"] = (
            sha256_of(REPO / entry["path"]) == entry["sha256"]
        )
    for path, expected in protocol["artifacts"].items():
        checks[f"protocol.artifact.{path}"] = sha256_of(REPO / path) == expected
    out = REPO / "outputs/labeler/elm"
    tables = json.loads((out / "tables.json").read_text())
    for name, entry in tables["sources"].items():
        checks[f"tables.source.{name}"] = sha256_of(entry["path"]) == entry["sha256"]
    for name, expected in tables["tables"].items():
        paths = list(out.rglob(name))
        checks[f"table.{name}"] = len(paths) == 1 and sha256_of(paths[0]) == expected
    notes = tables["shared_notes"]
    checks["tables.shared_notes"] = sha256_of(notes["path"]) == notes["sha256"]
    renders = json.loads((out / "table_render_verification.json").read_text())
    checks["renders.compiled"] = renders["compile_failures"] == 0
    checks["renders.viewed"] = renders["all_pages_viewed"]
    for entry in renders["artifacts"]:
        checks[f"render.viewed.{entry['source']['path']}"] = entry["viewed"]
        for artifact in (entry["source"], entry["pdf"], *entry["pngs"]):
            checks[f"render.hash.{artifact['path']}"] = (
                sha256_of(artifact["path"]) == artifact["sha256"]
            )
    script = renders["script"]
    checks["renders.script"] = sha256_of(script["path"]) == script["sha256"]
    return checks


def main():
    native_test = REPO / "tests/labeler/test_elm_dsm_native.py"
    tests = TESTS + (
        [str(native_test.relative_to(REPO))] if native_test.exists() else []
    )
    commands = {
        "tests": [
            "bash",
            "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh",
            str(REPO),
            *tests,
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        "ruff": [*PIXI, "ruff", "check", *changed_python()],
        "new_file_format": [*PIXI, "ruff", "format", "--check", *changed_python(True)],
    }
    record = {"created": datetime.now(UTC).isoformat(), "git": git_sha(full=True)}
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
        print(name, result.returncode, output[-2000:], flush=True)
    source = "outputs/labeler/elm/ours/evaluation.json"
    before = json.loads(subprocess.check_output(["git", "show", f"{BASE}:{source}"]))
    current = json.loads((REPO / source).read_text())
    checks = {}
    for panel, old in before["sets"].items():
        new = current["sets"][panel]
        checks[f"{panel}.bins"] = old["bins"] == new["bins"]
        for name, result in old["methods"].items():
            for metric in ("auroc", "auprc", "precision", "recall", "f1"):
                if metric not in result["point"]:
                    continue
                checks[f"{panel}.{name}.{metric}"] = math.isclose(
                    result["point"][metric],
                    new["methods"][name]["point"][metric],
                    rel_tol=0,
                    abs_tol=1e-14,
                )
                checks[f"{panel}.{name}.{metric}.ci95"] = all(
                    math.isclose(a, b, rel_tol=0, abs_tol=1e-14)
                    for a, b in zip(
                        result["ci95"][metric],
                        new["methods"][name]["ci95"][metric],
                        strict=True,
                    )
                )
    record["primary_bin_metrics_unchanged"] = {
        "base_revision": BASE,
        "checks": checks,
        "all_unchanged": all(checks.values()),
    }
    artifacts = artifact_checks()
    record["artifact_hashes_current"] = {
        "checks": artifacts,
        "all_current": all(artifacts.values()),
    }
    print("artifact hashes current", all(artifacts.values()), flush=True)
    record["sources"] = {
        path: sha256_of(REPO / path)
        for path in (
            source,
            "outputs/labeler/elm/dsm/evaluation.json",
            "outputs/labeler/elm/swap/evaluation.json",
            "outputs/labeler/elm/tables.json",
            "outputs/labeler/elm/ours/annotation_strata.json",
            "outputs/labeler/elm/filterscope_metadata.json",
            "outputs/labeler/elm/dsm/native_evaluation.json",
            "outputs/labeler/elm/dsm/native_fetch.json",
            "outputs/labeler/elm/table_render_verification.json",
        )
    }
    record["passed"] = (
        all(checks.values())
        and all(artifacts.values())
        and all(record[key]["exit_code"] == 0 for key in commands)
    )
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
