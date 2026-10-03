"""Small evaluation summaries with verifiable external shot-level details."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DETAIL_KEYS = ("per_shot", "shots", "alarm_shots")
SUMMARY_MAX_BYTES = 500_000


def evaluation_details_path(
    path: Path, artifact_dir: Path, *, canonical_summary: Path
) -> Path:
    """Keep the canonical archive name and isolate custom output destinations."""
    suffix = ""
    if path.resolve() != canonical_summary.resolve():
        destination = str(path.resolve()).encode()
        suffix = "_" + hashlib.sha256(destination).hexdigest()[:16]
    return artifact_dir / f"{path.stem}{suffix}_details.json"


def _summary(value):
    if isinstance(value, dict):
        return {
            key: _summary(item)
            for key, item in value.items()
            if not (key in DETAIL_KEYS and isinstance(item, (list, dict)))
        }
    if isinstance(value, list):
        return [_summary(item) for item in value]
    return value


def load_evaluation(path: Path, *, details: bool = False) -> dict:
    """Read summary metrics, or verify and load the complete external record.

    Older records without a pointer are already complete and remain readable.
    The summary alone suffices for tables, figures and saved-prediction replay.
    """
    record = json.loads(path.read_text())
    pointer = record.get("external_details")
    if details and pointer is not None:
        payload = Path(pointer["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != pointer["sha256"]:
            raise ValueError("evaluation details SHA-256 does not match the summary")
        return json.loads(payload)
    return record


def write_evaluation(record: dict, path: Path, details_path: Path) -> None:
    """Preserve the full record externally and write a summary below 0.5 MB."""
    payload = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
    summary = _summary(record)
    summary["external_details"] = {
        "path": str(details_path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "omitted_keys": list(DETAIL_KEYS),
        "contents": "complete evaluation, including nested per-shot records",
        "reader": "labeler.rwm.records.load_evaluation(path, details=True)",
    }
    small = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode()
    if len(small) >= SUMMARY_MAX_BYTES:
        raise ValueError("evaluation summary must remain below 0.5 MB")
    if path.resolve() == details_path.resolve():
        raise ValueError("summary and external details need distinct paths")
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(small)
