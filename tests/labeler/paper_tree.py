"""Evaluation records for the `labeler.paper` tests, shaped as the runs write them."""

from __future__ import annotations

from labeler.paper import scores


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
        "bar_thresholds": {"f1": 0.9, "mhd_fp_rate": 0.05},
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
