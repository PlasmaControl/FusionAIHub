"""`labels.event_sources`: the contract that lets "nobody looked" be told from "nothing happened".

labelmaker writes `events/<shot>_sources.parquet`; ideate reads it. That side of the work is
being done in labelmaker, so everything here is built against the CONTRACT with synthetic
fixtures -- which is also what the contract is for: two codebases writing and reading the same
file need one definition of it, and `write_sources` is it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ideate.labels import event_sources as es


def _rows(shot: int) -> list[dict]:
    return [
        es.source_row(shot, "tokeye_track", t_cov0_s=0.0, t_cov1_s=6.0, n_events=3,
                      diag="mhr", channel=0, pass_name="wide"),
        es.source_row(shot, "ece_sawtooth", t_cov0_s=1.0, t_cov1_s=5.0, n_events=0, diag="ece"),
        es.source_row(shot, "dalpha_lh", status="skipped", reason="no d_alpha on this shot"),
    ]


def test_the_contract_is_exactly_these_columns_in_this_order(tmp_path):
    """A column added on one side and not the other is a file neither can read. The order and the
    dtypes are the contract, not an implementation detail of whoever writes it first."""
    assert es.SOURCES_COLUMNS == (
        "shot", "source", "status", "reason", "t_cov0_s", "t_cov1_s", "n_events",
        "diag", "channel", "pass_name", "run_id", "git_sha", "written_at",
    )
    assert es.SOURCES_DTYPES["shot"] == "int32"
    assert es.SOURCES_DTYPES["t_cov0_s"] == es.SOURCES_DTYPES["t_cov1_s"] == "float64"
    assert es.SOURCES_DTYPES["n_events"] == "int32"
    assert es.SOURCES_DTYPES["channel"] == "int16"
    assert es.STATUSES == ("ok", "skipped", "error")

    path = es.write_sources(tmp_path / "198658_sources.parquet", _rows(198658))
    got = es.read_sources(path)
    assert list(got.columns) == list(es.SOURCES_COLUMNS)
    assert {c: str(d) for c, d in got.dtypes.items()} == dict(es.SOURCES_DTYPES)
    assert len(got) == 3


def test_a_missing_file_reads_as_an_empty_typed_frame_not_an_error(tmp_path):
    """`events/` does not exist until the mask job has run, and a shot nobody has processed is
    the state this whole table exists to make visible. It cannot be an exception."""
    got = es.read_sources(es.sources_file(tmp_path, 198658))
    assert got.empty
    assert list(got.columns) == list(es.SOURCES_COLUMNS)


def test_a_row_with_an_unknown_status_or_a_missing_column_is_refused(tmp_path):
    with pytest.raises(ValueError, match="status"):
        es.source_row(1, "x", status="maybe")
    good = es.source_row(1, "x")
    with pytest.raises(ValueError, match="missing"):
        es.write_sources(tmp_path / "a.parquet", [{k: v for k, v in good.items() if k != "diag"}])
    with pytest.raises(ValueError, match="unknown columns"):
        es.write_sources(tmp_path / "b.parquet", [{**good, "surprise": 1}])
    with pytest.raises(ValueError, match="status must be one of"):
        es.write_sources(tmp_path / "c.parquet", [{**good, "status": "fine"}])


def test_the_union_takes_only_the_shots_whose_file_exists(tmp_path):
    """A shot with no file contributes NO ROWS. A row of zeros would say "we looked and saw
    nothing", which is the confusion the table is for."""
    events = tmp_path / "events"
    es.write_sources(es.sources_file(events, 198658), _rows(198658))
    es.write_sources(es.sources_file(events, 190090), _rows(190090))

    got = es.sources_union([198658, 190090, 204346], events_dir=events)
    assert sorted(set(got["shot"])) == [190090, 198658]
    assert len(got) == 6
    assert list(got.columns) == list(es.SOURCES_COLUMNS)
    assert es.sources_union([204346], events_dir=events).empty


def test_the_summary_counts_each_status_and_says_whether_anything_completed(tmp_path):
    events = tmp_path / "events"
    es.write_sources(es.sources_file(events, 198658), _rows(198658))
    es.write_sources(
        es.sources_file(events, 190090),
        [es.source_row(190090, "tokeye_track", status="error", reason="mhr read failed")],
    )
    union = es.sources_union([198658, 190090], events_dir=events)

    assert es.shot_summary(union, 198658) == {
        "n_sources": 3, "n_sources_ok": 2, "n_sources_skipped": 1, "n_sources_error": 0,
        "n_sources_unknown_coverage": 0, "has_observed_products": True,
    }
    failed = es.shot_summary(union, 190090)
    assert failed["n_sources_error"] == 1
    assert failed["has_observed_products"] is False, "a source that crashed observed nothing"
    absent = es.shot_summary(union, 204346)
    assert absent["n_sources"] == 0 and absent["has_observed_products"] is False


def test_a_shot_whose_every_source_was_skipped_has_no_observed_product(tmp_path):
    union = es.sources_union([1], events_dir=tmp_path)
    assert union.empty
    skipped = pd.DataFrame([es.source_row(1, "a", status="skipped", reason="no diag")])
    skipped = skipped[list(es.SOURCES_COLUMNS)].astype(es.SOURCES_DTYPES)
    assert es.shot_summary(skipped, 1)["has_observed_products"] is False


def test_coverage_comes_from_the_ok_rows_only(tmp_path):
    """A skipped source has no coverage, and an errored one has none either -- reading a span off
    a row that did not complete is how a window nobody examined becomes "covered"."""
    rows = [
        es.source_row(1, "a", t_cov0_s=1.0, t_cov1_s=4.0),
        es.source_row(1, "b", t_cov0_s=3.0, t_cov1_s=6.0),
        es.source_row(1, "c", status="skipped", reason="x", t_cov0_s=-10.0, t_cov1_s=94.0),
    ]
    df = pd.DataFrame(rows)[list(es.SOURCES_COLUMNS)].astype(es.SOURCES_DTYPES)
    assert es.coverage_span(df) == (1.0, 6.0)
    assert es.covers(df, 2.0, 3.0) is True
    assert es.covers(df, 0.0, 0.5) is False
    assert es.covers(df, 20.0, 30.0) is False
    assert es.covers(df, None, None) is True
    assert es.coverage_span(es.empty_sources()) is None
    assert es.covers(es.empty_sources(), 1.0, 2.0) is None


def test_a_source_with_no_recorded_coverage_does_not_pretend_to_have_any():
    rows = [es.source_row(1, "a", n_events=0)]  # t_cov defaults to NaN
    df = pd.DataFrame(rows)[list(es.SOURCES_COLUMNS)].astype(es.SOURCES_DTYPES)
    assert np.isnan(df["t_cov0_s"].iloc[0])
    assert es.coverage_span(df) is None
    assert es.covers(df, 1.0, 2.0) is None


def test_a_text_source_that_ran_is_not_an_observation_and_covers_nothing():
    """`text` runs the lexicon over the shot's logbook entries. That it RAN says nothing about
    what any diagnostic showed -- the policy `labelmaker.events.windows.DIAGNOSTIC_EVIDENCE`
    states for rows, applied to the source that writes them -- so a shot whose only completed
    source is `text` has no observed product, and the shot span its row carries covers no
    window. Otherwise a logbook-only shot would come back `observed`: "0 detections inside
    coverage", from a detector that never ran."""
    assert es.NON_DIAGNOSTIC_SOURCES == ("text",)
    rows = [
        es.source_row(1, "text", t_cov0_s=0.0, t_cov1_s=6.5, n_events=2),
        es.source_row(1, "tokeye_track", status="skipped", reason="group absent", diag="mhr"),
    ]
    df = pd.DataFrame(rows)[list(es.SOURCES_COLUMNS)].astype(es.SOURCES_DTYPES)
    summary = es.shot_summary(df, 1)
    assert summary["n_sources_ok"] == 1                 # it did run, and the count says so
    assert summary["has_observed_products"] is False    # but it observed no diagnostic
    assert es.coverage_span(df) is None
    assert es.covers(df, 1.0, 2.0) is None

    # Beside a detector, the detector's span is the coverage - not the text's shot span.
    both = pd.concat([
        df, pd.DataFrame([es.source_row(1, "ece_sawtooth", t_cov0_s=1.0, t_cov1_s=4.0, diag="ece")]),
    ])[list(es.SOURCES_COLUMNS)].astype(es.SOURCES_DTYPES)
    assert es.shot_summary(both, 1)["has_observed_products"] is True
    assert es.coverage_span(both) == (1.0, 4.0)
    assert es.covers(both, 5.0, 6.0) is False


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[list(es.SOURCES_COLUMNS)].astype(es.SOURCES_DTYPES)


def test_an_ok_source_whose_coverage_is_unknown_can_neither_cover_nor_uncover(tmp_path):
    """THE CORNER. Real shot 198658 wrote `actuator/ech_power_total` as `ok` with NaN coverage:
    the detector RAN and nothing records over what span. Such a row is not a skipped source (it
    did run, and `has_observed_products` says so), and it is not coverage either -- it cannot
    make a window observed, because nothing in it says the window was looked at."""
    rows = [es.source_row(1, "actuator", diag="ech_power_total", n_events=0)]  # t_cov NaN
    df = _df(rows)

    assert es.shot_summary(df, 1)["has_observed_products"] is True, "it ran"
    assert es.shot_summary(df, 1)["n_sources_unknown_coverage"] == 1
    assert [r["source"] for r in es.unknown_coverage_rows(df).to_dict("records")] == ["actuator"]
    assert es.coverage_span(df) is None
    assert es.covers(df, 3.0, 4.0) is None, "unknown coverage is not 'no', and not 'yes'"
    assert es.observing_rows(df, 3.0, 4.0).empty


def test_an_unknown_coverage_row_beside_a_covering_one_is_still_listed(tmp_path):
    """The caveat has to name it even when the shot does have real coverage: "ech_power_total ran"
    is part of what the reply means, and a reader counting sources would otherwise count it."""
    df = _df([
        es.source_row(1, "actuator", diag="ech_power_total", n_events=0),
        es.source_row(1, "ece_sawtooth", diag="ece", t_cov0_s=1.0, t_cov1_s=4.0, n_events=0),
    ])
    assert es.coverage_span(df) == (1.0, 4.0)
    assert es.covers(df, 2.0, 3.0) is True
    assert es.covers(df, 8.0, 9.0) is False
    assert len(es.unknown_coverage_rows(df)) == 1
    assert [r["source"] for r in es.observing_rows(df, 2.0, 3.0).to_dict("records")] \
        == ["ece_sawtooth"]


def test_a_window_in_the_gap_between_two_sources_is_not_covered_by_their_hull():
    """Coverage is per source, not the outer hull of all of them: two passes over [1,2] and [5,6]
    have not looked at 3-4 s, and answering from the hull is the same borrowed-span defect the
    per-source table exists to fix. `retrieval.phenomena` clips per source for the same reason."""
    df = _df([
        es.source_row(1, "a", t_cov0_s=1.0, t_cov1_s=2.0),
        es.source_row(1, "b", t_cov0_s=5.0, t_cov1_s=6.0),
    ])
    assert es.coverage_span(df) == (1.0, 6.0)  # the hull is still reported, labelled as the hull
    assert es.covers(df, 3.0, 4.0) is False
    assert es.covers(df, 1.5, 3.0) is True
    assert es.observing_rows(df, 3.0, 4.0).empty
