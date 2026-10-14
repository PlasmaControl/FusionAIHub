"""The tearing benchmark gives the DSM no published operating point."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/labeler"


@pytest.fixture
def tm_benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("tm_benchmark")


def test_dsm_risk_03_is_an_appendix_fixed_threshold_not_a_published_one(tm_benchmark):
    titles, order = tm_benchmark.SETTING_TITLE, tm_benchmark.SETTING_ORDER
    assert set(titles) == set(order)
    assert order.index("Legacy, fixed") > order.index("Legacy, tuned")
    heading = titles["Legacy, fixed"]
    assert "published" not in heading and "not an operating point" in heading
    note = tm_benchmark.DSM_RISK_03_NOTE
    assert "upstream helper default" in note and "never reaches it" in note
    assert "no published operating point" in note


def test_the_dsm_threshold_sources_name_the_helper_default(tm_benchmark):
    from tm_prior_published import THRESHOLD_SOURCE

    for text in (THRESHOLD_SOURCE, tm_benchmark.DSM_RISK_03_MEANING):
        assert "metrics_helpers.py:85" in text
        assert "0.323" in text
    assert "no published operating point" in THRESHOLD_SOURCE
