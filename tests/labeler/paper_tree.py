"""Evaluation records for the `labeler.paper` tests, shaped as the runs write them."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from labeler.ae import xpower
from labeler.config import Paths
from labeler.paper import scores

from . import ae_tree


def temporary_paths(tmp_path: Path, monkeypatch) -> Paths:
    """A temporary Paths (root, label tables, corpus) under `tmp_path`, laid
    out as `ae_tree.build` lays its tree, and set in the environment, so a test
    whose code reads no Paths still has one set before the call."""
    paths = Paths(
        root=tmp_path / "root",
        label_tables=tmp_path / "events",
        corpus=tmp_path / "corpus",
    )
    ae_tree.env(monkeypatch, paths)
    return paths


def est(value, low=None, high=None) -> dict:
    """A `stats.Estimate.as_json`."""
    return {"value": value, "low": low, "high": high, "undefined_replicates": 0}


def ae_evaluation() -> dict:
    """`models/ae_xpower/v1/evaluation.json`: method i scores 0.9 - 0.1 i."""
    methods = {}
    for i, m in enumerate(scores.AE_NAMES):
        base = round(0.9 - 0.1 * i, 2)
        methods[m] = {
            "precision": est(base, base - 0.05, base + 0.03),
            "recall": est(base - 0.02, base - 0.07, base + 0.01),
            "f1": est(base - 0.01, base - 0.06, base + 0.02),
            "fp_rate_mhd": est(0.02 * (i + 1), 0.01 * (i + 1), 0.03 * (i + 1)),
            "fp_rate_other": est(0.01 * (i + 1)),
        }
    return {
        "meta": {"candidate": "band80-mhd3", "threshold": 0.42, "tier": "suggestions"},
        "bar": {"A1": True, "A2": False, "A3": True, "all": False},
        "bar_thresholds": {
            "f1": 0.9,
            "precision": 0.75,
            "recall": 0.75,
            "f1_vs_seldnet_low": -0.03,
            "mhd_fp_rate": 0.05,
        },
        "differences": {
            "f1_minus_seldnet": est(0.1, -0.02, 0.2),
            "mhd_fp_minus_seldnet": est(-0.02, -0.05, 0.01),
            "f1_minus_always": est(0.5, 0.4, 0.6),
        },
        "methods": methods,
        "frames": {
            "shots": 40,
            "scored": 9000,
            "present": 2500,
            "mhd_absent": 600,
            "shots_with_mhd_absent": 12,
        },
    }


def seg_evaluation() -> dict:
    """`models/ae_seg/v1/evaluation.json`: method i scores 0.8 - 0.1 i."""
    methods = {}
    for i, m in enumerate(scores.SEG_NAMES):
        base = round(0.8 - 0.1 * i, 2)
        methods[m] = {
            metric: est(base, base - 0.04, base + 0.04) for metric in scores.SEG_METRICS
        }
        methods[m]["fp_rate_mhd"] = est(0.01 * (i + 1), 0.0, 0.02 * (i + 1))
    return {
        "meta": {"threshold": 0.5, "tier": "suggestions"},
        "bar": {"G1": True, "G2": True, "G3": True, "all": True},
        "bar_thresholds": {
            "dice": 0.75,
            "dice_low": 0.65,
            "frame_precision": 0.9,
            "mhd_fp_rate": 0.05,
        },
        "differences": {
            "dice_minus_recipe": est(0.1, 0.05, 0.15),
            "frame_f1_minus_recipe": est(-0.0003, -0.008, 0.007),
        },
        "methods": methods,
        "counts": {
            "shots": 40,
            "ae_pixels": 123456,
            "scored_pixels": 7654321,
            "frames": 9000,
            "present_frames": 2500,
            "mhd_absent_frames": 600,
        },
    }


MIN_PT = 6  # D38 asks 7 pt; nothing in a figure is smaller than 6


def small_text(fig) -> list[tuple[str, float]]:
    """Every visible, non-empty text of a drawn figure below `MIN_PT`."""
    from matplotlib.text import Text

    return [
        (t.get_text(), t.get_fontsize())
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT
    ]


def scored_labels(paths: Paths, version: str = "v1") -> Path:
    """The chosen model's copy of the labels it was trained and scored on (D18)."""
    models = xpower.model_dir(paths, version)
    candidate = json.loads((models / "chosen.json").read_text())["candidate"]
    return models / candidate / "review" / "labels.csv"


def record_labels(paths: Paths, version: str = "v1") -> str:
    """Name the model's copy in its `evaluation.json`, as `evaluate` does."""
    sha = hashlib.sha256(scored_labels(paths, version).read_bytes()).hexdigest()
    evaluation = xpower.model_dir(paths, version) / "evaluation.json"
    record = json.loads(evaluation.read_text())
    record["meta"]["labels_sha256"] = sha
    evaluation.write_text(json.dumps(record))
    return sha


def as_version(paths: Paths, version: str = "v2", *, keep: bool = False) -> Path:
    """The v1 frame-model records at `version`, laid out as v2 will be:
    `models/ae_xpower/<version>/<candidate>/{model.pt, split.csv,
    review/labels.csv}`, `chosen.json` and `evaluation.json`, naming the copy.
    v1 is moved, leaving nothing, unless `keep`. Its models directory."""
    old, new = xpower.model_dir(paths), xpower.model_dir(paths, version)
    (shutil.copytree if keep else shutil.move)(old, new)
    record_labels(paths, version)
    return new


POI_CSV = (
    "shot,region,t_start_ms,t_end_ms,f_lo_khz,f_hi_khz,pixels,in_scored_window\n"
    "102,1,300,900,140,152,40,True\n"
)


def seg_record(paths: Paths, seg_version: str = "v1", *, frame: str = "v1") -> str:
    """The segmentation at `seg_version`, as `labeler.ae.seg` leaves it: its
    `evaluation.json` naming its own copy of the labels (taken from the frame
    model `frame`'s) and that model, the copy, and points of interest for 102.
    The copy's sha256."""
    seg = paths.root / "models" / "ae_seg" / seg_version
    copy = seg / "review" / "labels.csv"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(scored_labels(paths, frame).read_bytes())
    sha = hashlib.sha256(copy.read_bytes()).hexdigest()
    record = seg_evaluation()
    record["meta"]["inputs"] = {"labels_sha256": sha}
    model = xpower.model_dir(paths, frame) / "band80-mhd3" / "model.pt"
    record["meta"]["ae_model"] = str(model)
    (seg / "evaluation.json").write_text(json.dumps(record))
    poi = paths.root / "poi" / "alfven_eigenmode" / f"ae_seg-{seg_version}" / "poi.csv"
    poi.parent.mkdir(parents=True)
    poi.write_text(POI_CSV)
    return sha
