"""Freeze producer-owned interpretation records without local science defaults."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ...config import sha256_of


def sources(labels_path):
    root = Path(labels_path).parent
    worktree = Path(__file__).resolve().parents[4].with_name("FusionAIHub-r4-detach")
    return {
        "recipe": Path(
            os.environ.get(
                "LABELER_DETACHMENT_RECIPE", str(root / "review_recipe.json")
            )
        ),
        "label_metadata": root / "labels_rule.meta.json",
        "method_record": Path(
            os.environ.get(
                "LABELER_DETACHMENT_METHOD",
                str(worktree / "docs/labeler/detachment.md"),
            )
        ),
    }


def load(labels_path):
    """Prefer a full recipe record; legacy delivery retains published method text.

    The fallback is the producer's own method document plus exported metadata,
    never a UI-authored threshold or evidence gate. Missing records stay explicit.
    """
    paths = sources(labels_path)
    result = {"record": {}, "documentation": "", "sources": {}}
    for key, path in paths.items():
        result["sources"][key] = {
            "path": str(path),
            "sha256": sha256_of(path) if path.is_file() else None,
        }
    try:
        if paths["recipe"].is_file():
            result["record"] = json.loads(paths["recipe"].read_text())
        else:
            if paths["label_metadata"].is_file():
                metadata = json.loads(paths["label_metadata"].read_text())
                result["record"] = metadata.get("recipe", metadata)
            if paths["method_record"].is_file():
                result["documentation"] = (
                    paths["method_record"]
                    .read_text()
                    .split("## Evaluation", 1)[0]
                    .strip()
                )
        if not isinstance(result["record"], dict):
            raise TypeError("producer recipe must be a JSON object")
    except (OSError, ValueError, TypeError) as error:
        result.update(record={}, documentation="", reason=str(error))
    if not result["record"] and not result["documentation"]:
        result.setdefault("reason", "Producer interpretation record unavailable")
    return result


def guides(record, indicator):
    """Optional structured thresholds from the producer, never UI constants."""
    values = record.get("thresholds", {}).get(indicator, {})
    return sorted(
        {
            float(v)
            for v in values.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    )
