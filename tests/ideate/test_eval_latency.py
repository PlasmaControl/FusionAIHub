"""The latency table: that it times what it says it times, and that a row with no budget gets no
verdict.

The numbers themselves are not asserted here -- a wall-clock assertion on a shared login node is
a flaky test, not a measurement. What is asserted is the shape of the measurement: the warmup is
discarded, the repeats are the repeats, the budgets are the plan's, and a FAIL is reachable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ideate.eval import latency as lat
from ideate.schema import LatencyRow
from ideate.shotdb import store

from .conftest import shot_record, write_db


@pytest.fixture(autouse=True)
def _no_minilm(monkeypatch):
    from ideate.shotdb import text as text_mod

    monkeypatch.setattr(
        text_mod, "embed_texts", lambda texts: np.tile([1.0, 0.0, 0.0], (len(texts), 1))
    )


@pytest.fixture
def db_dir(tmp_path: Path) -> Path:
    d = tmp_path / "db"
    d.mkdir()
    write_db(
        d,
        [
            shot_record(300, "r1", 1.0e6, 5.0e6, "QH-mode with an EHO"),
            shot_record(301, "r1", 1.1e6, 5.5e6, "ELM suppression with n=3 RMP"),
            shot_record(302, "r2", 1.2e6, 6.0e6, "tearing mode locked"),
        ],
    )
    return d


# ------------------------------------------------------------------------------- the timing


def test_the_warmup_calls_are_thrown_away():
    calls = []
    samples = lat.time_it(lambda: calls.append(1), repeats=4, warmup=2)
    assert len(calls) == 6  # 2 discarded + 4 timed
    assert len(samples) == 4


def test_every_sample_is_a_positive_wall_clock_duration():
    samples = lat.time_it(lambda: sum(range(1000)), repeats=5, warmup=0)
    assert len(samples) == 5
    assert all(s >= 0.0 for s in samples)


def test_measure_times_all_five_operations_with_the_plans_budgets(db_dir):
    report = lat.measure(db_dir, repeats=2)
    assert [r.name for r in report.rows] == [
        "load", "search_text", "search_no_text", "phenomenon_locate", "describe"
    ]
    assert {r.name: r.budget_s for r in report.rows} == {
        "load": 3.0,
        "search_text": 0.4,
        "search_no_text": 0.2,
        "phenomenon_locate": 0.3,
        "describe": None,
    }
    assert all(r.repeats == 2 for r in report.rows)


def test_the_budgets_are_the_ones_appendix_b_states():
    """Pinned as literals so a later 'adjustment' to make a row pass is a visible diff here."""
    assert lat.BUDGETS_S == {
        "load": 3.0,
        "search_text": 0.4,
        "search_no_text": 0.2,
        "phenomenon_locate": 0.3,
        "describe": None,
    }


def test_p95_is_never_below_the_median(db_dir):
    report = lat.measure(db_dir, repeats=3)
    assert all(r.p95_s >= r.median_s for r in report.rows)


def test_the_report_says_it_is_warm_and_what_it_timed(db_dir):
    report = lat.measure(db_dir, repeats=2, text="QH-mode", phenomenon="tearing")
    assert report.warm is True
    joined = " ".join(report.notes)
    assert "warm" in joined and "'QH-mode'" in joined and "'tearing'" in joined
    assert report.n_shots == 3 and report.n_segment_rows == 6


def test_an_empty_database_is_refused_rather_than_timed(tmp_path):
    d = tmp_path / "db"
    d.mkdir()
    write_db(d, [])
    with pytest.raises(ValueError, match="no shots"):
        lat.measure(d)


def test_a_database_whose_shots_lack_the_segment_says_so_instead_of_timing_an_empty_search(
    db_dir,
):
    report = lat.measure(db_dir, repeats=1, segment="ramp_down")
    assert any("no shot has a ramp_down segment" in n for n in report.notes)


# ------------------------------------------------------------------------------ the verdict


def test_a_row_under_its_budget_passes_and_one_over_it_fails():
    under = LatencyRow(name="x", what="", repeats=20, median_s=0.1, p95_s=0.2, budget_s=0.4)
    over = LatencyRow(name="x", what="", repeats=20, median_s=0.5, p95_s=0.9, budget_s=0.4)
    assert under.verdict == "PASS"
    assert over.verdict == "FAIL"


def test_a_row_with_no_budget_gets_no_verdict_rather_than_a_free_pass():
    """`describe` has no Appendix B number. A table of passes, one of which was never tested
    against anything, is worse than an honest gap."""
    row = LatencyRow(name="describe", what="", repeats=20, median_s=0.9, p95_s=1.0)
    assert row.verdict == "n/a"


def test_the_verdict_is_read_off_the_median_not_the_p95():
    row = LatencyRow(name="x", what="", repeats=20, median_s=0.3, p95_s=9.0, budget_s=0.4)
    assert row.verdict == "PASS"


# ----------------------------------------------------------------------------- the markdown


def test_the_markdown_is_the_budget_table_with_a_verdict_per_row(db_dir):
    md = lat.markdown(lat.measure(db_dir, repeats=2))
    assert "| operation | what | median | p95 | budget | verdict |" in md
    assert "`search_text`" in md and "400.0 ms" in md
    assert "| — | n/a |" in md  # the describe row: no budget, no verdict
    assert "warm" in md
