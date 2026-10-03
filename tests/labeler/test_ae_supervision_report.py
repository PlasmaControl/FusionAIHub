"""The swap report reads paired differences in the direction its prose states."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def report(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/labeler"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("ae_supervision_swap_report")


def score(mean, shot_ci):
    return {"mean": mean, "ci95": [mean - 1, mean + 1], "ci95_shot": shot_ci}


def test_pair_flips_a_stored_difference_and_its_interval(report):
    summary = {
        "paired_differences": {
            "a minus b": {"auroc": score(0.2, [0.1, 0.4])},
        }
    }
    same = report.pair(summary, "a", "b")["auroc"]
    flipped = report.pair(summary, "b", "a")["auroc"]
    assert same["mean"] == 0.2
    assert flipped["mean"] == -0.2
    assert flipped["ci95_shot"] == [-0.4, -0.1]
    assert flipped["ci95"] == [-1.2, 0.8]


def test_verdict_uses_the_shot_interval_only(report):
    assert report.verdict(score(0.3, [0.1, 0.5])) == "leads"
    assert report.verdict(score(-0.3, [-0.5, -0.1])) == "trails"
    assert report.verdict(score(0.0, [-0.1, 0.1])) == "is not resolved from"


def test_clock_profile_names_the_peak_and_the_edges(report):
    prior = [0.0] * 200
    prior[50] = 0.8
    text = report.profile(prior)
    assert "peaks at 505 ms (0.80)" in text
    assert "0.00 over its first and last 100 ms" in text


def test_table_cells_never_hold_both_an_sd_and_an_interval(report):
    seed = {"value": 0.5, "ci95": [0.4, 0.6]}
    mean = {"mean": 0.5, "sd": 0.02, "ci95": [0.4, 0.6], "ci95_shot": [0.45, 0.55]}
    assert "[" in report.point_ci(seed) and report.PM not in report.point_ci(seed)
    assert report.PM in report.mean_sd(mean) and "[" not in report.mean_sd(mean)
