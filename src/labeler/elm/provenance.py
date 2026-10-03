"""Content provenance and compatible continuation of ELM cross-validation runs.

Retrospective records describe files at the time of the audit. Their hashes do
not establish that those files were unchanged since training.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ..config import Paths, sha256_of
from . import prepare

SOURCE_FILES = (
    "src/labeler/config.py",
    "src/labeler/env.py",
    *(
        f"src/labeler/elm/{name}.py"
        for name in ("inputs", "labels", "net", "onset", "prepare", "score", "train")
    ),
)


def observed_at() -> str:
    return datetime.now(UTC).isoformat()


def file_record(path: Path) -> dict:
    """Identify a file by absolute path, size, modification time and SHA256."""
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_of(path),
    }


def code_record(revision: str | None = None, evidence: str | None = None) -> dict:
    """Hash the live training sources, or source blobs at a retrospective commit."""
    repo = Path(__file__).resolve().parents[3]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, timeout=10)

    resolved = git("rev-parse", f"{revision or 'HEAD'}^{{commit}}").decode().strip()
    sources, changed = [], []
    for name in SOURCE_FILES:
        committed = git("show", f"{resolved}:{name}")
        current = (repo / name).read_bytes()
        content = committed if revision else current
        sources.append({"path": name, "sha256": hashlib.sha256(content).hexdigest()})
        if current != committed:
            changed.append(name)
    return {
        "revision": resolved,
        "source_origin": "git_commit" if revision else "training_worktree",
        "sources": sources,
        "worktree_differs_from_revision": changed,
        "revision_is_reconstructed": revision is not None,
        "evidence": evidence,
    }


def data_record(paths: Paths, shots: list[int]) -> dict:
    """The reviewed labels, cohort split table and prepared inputs actually read."""
    return {
        "reviewed_labels": file_record(prepare.review_csv(paths)),
        "cohort": file_record(paths.catalog / "cohort.csv"),
        "inputs": [
            {"shot": shot, **file_record(prepare.inputs_dir(paths) / f"{shot}.npy")}
            for shot in sorted(shots)
        ],
    }


def fold_artifacts(out: Path, fold: int, shots: list[int]) -> dict:
    """A completed fold's checkpoint, detailed record and held-out predictions."""
    return {
        "checkpoints": [file_record(out / f"fold{fold}" / "model.pt")],
        "fold_record": file_record(out / f"fold{fold}" / "fold.json"),
        "predictions": [
            {"shot": shot, **file_record(out / "pred" / f"{shot}.npz")}
            for shot in sorted(shots)
        ],
    }


def _content(value):
    """Compare content while allowing relocated files and observation timestamps."""
    if isinstance(value, dict):
        return {
            k: _content(v)
            for k, v in value.items()
            if k not in ("path", "mtime_ns", "observed_at")
        }
    if isinstance(value, list):
        return [_content(v) for v in value]
    return value


def continue_record(out: Path, record: dict) -> dict:
    """Retain completed folds only when splits, settings and input bytes match.

    Older records lack `inner_val`; recover it from their detailed fold records.
    A legacy record without data/code hashes can be merged on its recorded
    settings, but is explicitly identified as lacking contemporaneous provenance.
    """
    path = out / "run.json"
    if not path.exists():
        if list(out.glob("fold*/model.pt")):
            raise ValueError(
                "existing checkpoints have no run.json; use a new run name"
            )
        return record
    previous = json.loads(path.read_text())
    for key in ("config", "folds", "shots", "parameters"):
        if previous.get(key) != record[key]:
            raise ValueError(f"incompatible existing run {key}; use a new run name")
    inner = previous.get("inner_val")
    if inner is None:
        counts = {
            len(
                json.loads((out / f"fold{r['fold']}" / "fold.json").read_text())[
                    "inner_val"
                ]
            )
            for r in previous["fold_records"]
        }
        if counts != {record["inner_val"]}:
            raise ValueError("incompatible existing run inner_val; use a new run name")
    elif inner != record["inner_val"]:
        raise ValueError("incompatible existing run inner_val; use a new run name")
    prior_prov = previous.get("provenance")
    if prior_prov:
        for key in ("data",):
            if _content(prior_prov[key]) != _content(record["provenance"][key]):
                raise ValueError(f"incompatible existing run {key}; use a new run name")
        if prior_prov["code"]["sources"] != record["provenance"]["code"]["sources"]:
            raise ValueError("incompatible existing run code; use a new run name")
        record["provenance"] = prior_prov
    else:
        record["legacy_folds_without_provenance"] = [
            r["fold"] for r in previous["fold_records"]
        ]
    for row in previous["fold_records"]:
        k = row["fold"]
        if k < 0 or k >= len(record["folds"]) or row["test"] != record["folds"][k]:
            raise ValueError("incompatible existing fold record; use a new run name")
    record["fold_records"] = previous["fold_records"]
    return record


def add_fold(record: dict, row: dict) -> None:
    """Replace one fold without discarding any other completed fold."""
    rows = {r["fold"]: r for r in record["fold_records"]}
    rows[row["fold"]] = row
    record["fold_records"] = [rows[k] for k in sorted(rows)]


def backfill(out: Path, paths: Paths, revision: str, evidence: str) -> dict:
    """Populate an existing record without altering fits, thresholds or folds."""
    path = out / "run.json"
    record = json.loads(path.read_text())
    prior = record.get("provenance", {})
    original = prior.get("original_run_record", file_record(path))
    shots = sorted(s for fold in record["folds"] for s in fold)
    code = code_record(revision, evidence)
    record.setdefault("git_recorded_at_training", record["git"])
    record["git"] = code["revision"]
    record["provenance"] = {
        "mode": "retrospective",
        "observed_at": observed_at(),
        "caveat": (
            "Producing code reconstructed from the committed training implementation; "
            "no contemporaneous working-tree snapshot was saved. Artifact and data "
            "hashes describe bytes observed during this audit, not training-time hashes."
        ),
        "code": code,
        "data": data_record(paths, shots),
        "original_run_record": original,
    }
    for row in record["fold_records"]:
        row["artifacts"] = fold_artifacts(out, row["fold"], row["test"])
    path.write_text(json.dumps(record, indent=1) + "\n")
    return record
