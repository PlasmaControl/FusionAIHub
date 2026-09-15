"""An event is validated on construction and survives parquet unchanged.

The events file is the one place where a detector, a heuristic, a forecast
and a human annotation meet, so the schema is strict about what an event may
say: a finite time extent, a known evidence kind, a confidence that is a
probability, and a horizon only where "forecast" makes one meaningful.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from labeler.events.schema import (
    COLUMNS,
    DTYPES,
    EVIDENCE_KINDS,
    KNOWN_SOURCES,
    PASS_NAMES,
    SOURCE_COLUMNS,
    SOURCE_DTYPES,
    SOURCE_STATUSES,
    Event,
    index_rows,
    intervals,
    read_events,
    read_sources,
    write_events,
    write_sources,
)

SHOT = 190000


def _event(**kw) -> Event:
    base = {"shot": SHOT, "source": "tokeye_track", "phenomenon": "eho",
            "t0_s": 1.0, "t1_s": 1.5}
    return Event(**{**base, **kw})


def test_vocabularies_are_the_documented_ones():
    assert EVIDENCE_KINDS == (
        "detector", "heuristic", "forecast", "text", "human", "database", "model",
    )
    assert PASS_NAMES == ("", "wide", "zoom")
    assert KNOWN_SOURCES == (
        "tokeye_track", "tokeye_transient", "ece_sawtooth", "dalpha_lh",
        "elm_clock", "actuator", "qh_proxy", "text", "model", "label_forecast",
        "database", "qmin_rule",
    )


def test_a_plain_event_keeps_its_defaults():
    e = _event()
    assert e.evidence_kind == "detector" and e.pass_name == "" and e.channel == -1
    assert e.diag == "" and e.attrs == {}
    for field in (e.f0_khz, e.f1_khz, e.confidence, e.horizon_s,
                  e.t_cov0_s, e.t_cov1_s):
        assert np.isnan(field)


def test_a_point_event_is_allowed():
    e = _event(t0_s=2.0, t1_s=2.0)          # half-open, zero width
    assert e.t1_s == e.t0_s


@pytest.mark.parametrize("kw", [
    {"t0_s": float("nan")},
    {"t1_s": float("nan")},
    {"t0_s": float("inf")},
    {"t1_s": -float("inf")},
])
def test_non_finite_times_are_rejected(kw):
    with pytest.raises(ValueError, match="finite"):
        _event(**kw)


def test_a_backwards_interval_is_rejected():
    # Pin the message, not merely the field name: "t1_s" alone also matches
    # the non-finite complaint, so the loose form passed for the wrong reason.
    with pytest.raises(ValueError, match="t1_s must not precede t0_s"):
        _event(t0_s=1.5, t1_s=1.0)


def test_an_unknown_evidence_kind_is_rejected():
    with pytest.raises(ValueError, match="evidence_kind"):
        _event(evidence_kind="guess")


def test_an_unknown_pass_name_is_rejected():
    with pytest.raises(ValueError, match="pass_name"):
        _event(pass_name="medium")


@pytest.mark.parametrize("kw", [{"source": ""}, {"phenomenon": ""}])
def test_an_empty_source_or_phenomenon_is_rejected(kw):
    with pytest.raises(ValueError, match="empty"):
        _event(**kw)


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_a_confidence_outside_the_unit_interval_is_rejected(confidence):
    with pytest.raises(ValueError, match="confidence"):
        _event(confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_a_confidence_inside_the_unit_interval_is_kept(confidence):
    assert _event(confidence=confidence).confidence == confidence


def test_a_backwards_frequency_band_is_rejected():
    with pytest.raises(ValueError, match="f1_khz"):
        _event(f0_khz=20.0, f1_khz=5.0)


def test_half_a_frequency_band_is_allowed():
    # A heuristic that knows only one edge of a band still has a usable
    # event: the ordering check applies to a band, not to one finite edge.
    e = _event(f0_khz=20.0)
    assert e.f0_khz == 20.0 and np.isnan(e.f1_khz)
    assert np.isnan(_event(f1_khz=5.0).f0_khz)


def test_a_horizon_without_a_forecast_is_rejected():
    with pytest.raises(ValueError, match="horizon_s"):
        _event(horizon_s=0.05)


def test_a_forecast_without_a_horizon_is_rejected():
    with pytest.raises(ValueError, match="horizon_s"):
        _event(evidence_kind="forecast")


def test_a_forecast_with_a_horizon_is_accepted():
    e = _event(evidence_kind="forecast", horizon_s=0.05)
    assert e.horizon_s == 0.05


def test_unserialisable_attrs_are_rejected():
    with pytest.raises(ValueError, match="attrs"):
        _event(attrs={"blob": object()})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_attrs_are_rejected(value):
    # `json.dumps` writes bare `NaN`/`Infinity` by default, which is not JSON
    # and which no strict reader will take back. An attribute that cannot
    # round-trip is a bug where it was computed, not where it is read.
    with pytest.raises(ValueError, match="attrs"):
        _event(attrs={"chirp": value})


def test_event_ids_number_per_source_in_time_order(tmp_path):
    path = tmp_path / f"{SHOT}_events.parquet"
    events = [
        _event(source="tokeye_track", t0_s=3.0, t1_s=3.2),
        _event(source="ece_sawtooth", t0_s=2.5, t1_s=2.5, evidence_kind="heuristic"),
        _event(source="tokeye_track", t0_s=1.0, t1_s=1.2),
        _event(source="tokeye_track", t0_s=2.0, t1_s=2.1),
        _event(source="ece_sawtooth", t0_s=0.5, t1_s=0.5, evidence_kind="heuristic"),
    ]
    df = write_events(path, SHOT, events, run_id="r")
    by_id = dict(zip(df["event_id"], df["t0_s"], strict=True))
    assert by_id[f"{SHOT}-tokeye_track-00000"] == 1.0
    assert by_id[f"{SHOT}-tokeye_track-00001"] == 2.0
    assert by_id[f"{SHOT}-tokeye_track-00002"] == 3.0
    assert by_id[f"{SHOT}-ece_sawtooth-00000"] == 0.5
    assert by_id[f"{SHOT}-ece_sawtooth-00001"] == 2.5


def test_columns_and_dtypes_survive_a_round_trip(tmp_path):
    path = tmp_path / f"{SHOT}_events.parquet"
    written = write_events(
        path, SHOT,
        [_event(f0_khz=2.0, f1_khz=20.0, confidence=0.75, diag="mhr", channel=4,
                pass_name="zoom", attrs={"n_pix": 812, "chirp": -0.78},
                t_cov0_s=0.0, t_cov1_s=6.0),
         _event(source="label_forecast", phenomenon="disruption", t0_s=4.0,
                t1_s=4.0, evidence_kind="forecast", horizon_s=0.05)],
        run_id="run-test",
    )
    expected = {name: np.dtype(DTYPES[name]) for name in COLUMNS}
    assert written.dtypes.to_dict() == expected
    got = read_events(path)
    assert list(got.columns) == list(COLUMNS)
    assert got.dtypes.to_dict() == expected
    pd.testing.assert_frame_equal(got, written)
    row = got[got["source"] == "tokeye_track"].iloc[0]
    assert row["shot"] == SHOT and row["channel"] == 4 and row["pass_name"] == "zoom"
    assert row["attrs"] == '{"chirp": -0.78, "n_pix": 812}'
    assert row["run_id"] == "run-test" and row["git_sha"]
    stamped = datetime.fromisoformat(row["written_at"])
    assert stamped.tzinfo is not None and stamped.microsecond == 0
    # A forecast keeps its horizon, and the fields it never set stay NaN
    # through parquet - in float32 as in float64, so a consumer can tell
    # "no band" from a band at 0 kHz.
    fc = got[got["source"] == "label_forecast"].iloc[0]
    assert fc["evidence_kind"] == "forecast"
    assert fc["horizon_s"] == pytest.approx(0.05)
    for name in ("f0_khz", "f1_khz", "t_cov0_s", "t_cov1_s"):
        assert np.isnan(fc[name]), name


def test_a_write_leaves_no_temp_file(tmp_path):
    path = tmp_path / f"{SHOT}_events.parquet"
    write_events(path, SHOT, [_event()], run_id="r")
    assert list(tmp_path.iterdir()) == [path]


def test_merge_replaces_the_written_source_and_keeps_the_others(tmp_path):
    path = tmp_path / f"{SHOT}_events.parquet"
    write_events(path, SHOT, [
        _event(source="tokeye_track", t0_s=1.0, t1_s=1.5),
        _event(source="ece_sawtooth", phenomenon="sawtooth", t0_s=2.0, t1_s=2.0,
               evidence_kind="heuristic"),
    ], run_id="r1")
    write_events(path, SHOT, [
        _event(source="tokeye_track", t0_s=4.0, t1_s=4.5),
        _event(source="tokeye_track", t0_s=5.0, t1_s=5.5),
    ], run_id="r2")
    got = read_events(path)
    assert len(got) == 3
    assert got.dtypes.to_dict() == {n: np.dtype(DTYPES[n]) for n in COLUMNS}
    tracks = got[got["source"] == "tokeye_track"]
    assert tracks["t0_s"].tolist() == [4.0, 5.0]
    assert tracks["run_id"].tolist() == ["r2", "r2"]
    saw = got[got["source"] == "ece_sawtooth"]
    assert saw["t0_s"].tolist() == [2.0] and saw["run_id"].tolist() == ["r1"]


def test_an_explicit_source_list_clears_that_source(tmp_path):
    path = tmp_path / f"{SHOT}_events.parquet"
    write_events(path, SHOT, [
        _event(source="tokeye_track"),
        _event(source="qh_proxy", phenomenon="qh", evidence_kind="heuristic"),
    ], run_id="r1")
    got = write_events(path, SHOT, [], run_id="r2", sources=["tokeye_track"])
    assert got["source"].tolist() == ["qh_proxy"]
    assert read_events(path)["source"].tolist() == ["qh_proxy"]


def test_clearing_the_last_source_leaves_a_readable_empty_file(tmp_path):
    # "the detector ran and found nothing" is a file with no rows, not a
    # missing file, and it must still read back with the right dtypes.
    path = tmp_path / f"{SHOT}_events.parquet"
    write_events(path, SHOT, [_event()], run_id="r1")
    write_events(path, SHOT, [], run_id="r2", sources=["tokeye_track"])
    got = read_events(path)
    assert path.exists() and len(got) == 0
    assert got.dtypes.to_dict() == {n: np.dtype(DTYPES[n]) for n in COLUMNS}
    assert index_rows(path) == []


def test_merge_false_overwrites_the_file(tmp_path):
    path = tmp_path / f"{SHOT}_events.parquet"
    write_events(path, SHOT, [
        _event(source="tokeye_track"),
        _event(source="qh_proxy", phenomenon="qh", evidence_kind="heuristic"),
    ], run_id="r1")
    write_events(path, SHOT, [_event(source="elm_clock", phenomenon="elm",
                                     evidence_kind="heuristic")],
                 run_id="r2", merge=False)
    assert read_events(path)["source"].tolist() == ["elm_clock"]


def test_events_from_another_shot_are_refused(tmp_path):
    path = tmp_path / f"{SHOT}_events.parquet"
    with pytest.raises(ValueError, match="shot"):
        write_events(path, SHOT, [_event(), _event(shot=190001)], run_id="r")
    assert not path.exists()


def test_read_filters_by_source_and_phenomenon(tmp_path):
    path = tmp_path / f"{SHOT}_events.parquet"
    write_events(path, SHOT, [
        _event(source="tokeye_track", phenomenon="eho", t0_s=1.0, t1_s=1.5),
        _event(source="tokeye_track", phenomenon="fishbone", t0_s=2.0, t1_s=2.1),
        _event(source="qh_proxy", phenomenon="qh", t0_s=3.0, t1_s=3.5,
               evidence_kind="heuristic"),
    ], run_id="r")
    assert len(read_events(path, source="tokeye_track")) == 2
    assert read_events(path, phenomenon="qh")["source"].tolist() == ["qh_proxy"]
    both = read_events(path, source="tokeye_track", phenomenon="eho")
    assert both["t0_s"].tolist() == [1.0]
    assert both.index.tolist() == [0]                 # re-indexed after filtering
    assert read_events(path, source="nobody").empty


def test_a_missing_file_reads_as_an_empty_typed_frame(tmp_path):
    got = read_events(tmp_path / "no_such_events.parquet")
    assert list(got.columns) == list(COLUMNS)
    assert got.dtypes.to_dict() == {n: np.dtype(DTYPES[n]) for n in COLUMNS}
    assert len(got) == 0
    assert index_rows(tmp_path / "no_such_events.parquet") == []


def test_index_rows_summarise_each_source_and_phenomenon(tmp_path):
    path = tmp_path / f"{SHOT}_events.parquet"
    write_events(path, SHOT, [
        _event(phenomenon="eho", t0_s=1.0, t1_s=1.5, confidence=0.4),
        _event(phenomenon="eho", t0_s=3.0, t1_s=3.25, confidence=0.9),
        _event(phenomenon="fishbone", t0_s=2.0, t1_s=2.1, confidence=0.5),
    ], run_id="run-test")
    rows = {(r["source"], r["phenomenon"]): r for r in index_rows(path)}
    assert set(rows) == {("tokeye_track", "eho"), ("tokeye_track", "fishbone")}
    eho = rows[("tokeye_track", "eho")]
    assert eho["shot"] == SHOT and eho["n_events"] == 2
    assert eho["t0_min_s"] == 1.0 and eho["t1_max_s"] == 3.25
    assert eho["confidence_max"] == pytest.approx(0.9)
    assert eho["evidence_kind"] == "detector" and eho["run_id"] == "run-test"
    assert eho["written_at"]
    assert rows[("tokeye_track", "fishbone")]["n_events"] == 1


def test_index_rows_feed_a_keyed_events_index(tmp_path):
    from labeler.labels.store import append_index

    path = tmp_path / f"{SHOT}_events.parquet"
    write_events(path, SHOT, [_event(), _event(phenomenon="qcm", t0_s=2.0,
                                               t1_s=2.5)], run_id="r1")
    index = tmp_path / "events_index.parquet"
    keys = ["shot", "source", "phenomenon"]
    append_index(index, index_rows(path), keys=keys)
    write_events(path, SHOT, [_event()], run_id="r2")     # one event now
    append_index(index, index_rows(path), keys=keys)
    df = pd.read_parquet(index)
    assert len(df) == 2                                   # eho replaced, qcm kept
    assert df.set_index("phenomenon")["n_events"].to_dict() == {"eho": 1, "qcm": 1}
    assert df.set_index("phenomenon")["run_id"].to_dict() == {"eho": "r2",
                                                              "qcm": "r1"}


def test_intervals_sort_by_start_time():
    events = [
        _event(phenomenon="eho", t0_s=3.0, t1_s=3.5),
        _event(phenomenon="qcm", t0_s=0.5, t1_s=0.8),
        _event(phenomenon="eho", t0_s=1.0, t1_s=1.5),
    ]
    got = intervals(events, "eho")
    assert got.dtype == np.float64 and got.shape == (2, 2)
    np.testing.assert_array_equal(got, [[1.0, 1.5], [3.0, 3.5]])


def test_intervals_accept_a_frame_and_return_an_empty_pair_array(tmp_path):
    path = tmp_path / f"{SHOT}_events.parquet"
    write_events(path, SHOT, [_event(phenomenon="eho", t0_s=2.0, t1_s=2.5),
                              _event(phenomenon="eho", t0_s=1.0, t1_s=1.5)],
                 run_id="r")
    df = read_events(path)
    np.testing.assert_array_equal(intervals(df, "eho"), [[1.0, 1.5], [2.0, 2.5]])
    for empty in (intervals(df, "sawtooth"), intervals([], "eho")):
        assert empty.shape == (0, 2) and empty.dtype == np.float64


# ----------------------------------------- the per-source completion record


def _source(**kw) -> dict:
    base = {"source": "tokeye_track", "status": "ok", "reason": "",
            "t_cov0_s": 0.0, "t_cov1_s": 6.0, "n_events": 3,
            "diag": "mhr", "channel": 4, "pass_name": "wide"}
    return {**base, **kw}


def test_the_sources_contract_is_the_documented_columns_and_dtypes():
    # CONTRACT: shot_design's consumer is built against this list. A column may
    # be appended; none may be renamed, reordered or dropped.
    assert SOURCE_COLUMNS == (
        "shot", "source", "status", "reason", "t_cov0_s", "t_cov1_s",
        "n_events", "diag", "channel", "pass_name", "run_id", "git_sha",
        "written_at", "intervals", "min_gap_s",
    )
    assert SOURCE_STATUSES == ("ok", "skipped", "error")
    assert [SOURCE_DTYPES[c] for c in SOURCE_COLUMNS] == [
        "int32", "object", "object", "object", "float64", "float64",
        "int32", "object", "int16", "object", "object", "object", "object",
        "object", "float64",
    ]


def test_a_source_that_ran_and_found_nothing_is_a_row(tmp_path):
    # The whole point of the file: an events file holds no row for a
    # detector that found nothing, and this is what tells that from a
    # detector that never ran.
    path = tmp_path / f"{SHOT}_sources.parquet"
    out = write_sources(
        path, SHOT,
        [_source(diag="mhr", channel=0, n_events=0),
         _source(diag="mhr", channel=4, n_events=7)],
        run_id="r1",
    )
    assert list(out["n_events"]) == [0, 7]
    assert set(out["status"]) == {"ok"}
    back = read_sources(path)
    assert list(back.columns) == list(SOURCE_COLUMNS)
    assert back.dtypes.to_dict() == {
        c: np.dtype(SOURCE_DTYPES[c]) for c in SOURCE_COLUMNS
    }
    quiet = back[back["channel"] == 0].iloc[0]
    assert (quiet["n_events"], quiet["status"], quiet["reason"]) == (0, "ok", "")
    assert (quiet["t_cov0_s"], quiet["t_cov1_s"]) == (0.0, 6.0)
    assert quiet["run_id"] == "r1" and quiet["git_sha"]
    datetime.fromisoformat(str(quiet["written_at"]))


def test_a_skipped_step_is_a_row_carrying_its_reason(tmp_path):
    path = tmp_path / f"{SHOT}_sources.parquet"
    out = write_sources(
        path, SHOT,
        [_source(source="ece_sawtooth", diag="ece", channel=-1,
                 pass_name="", status="skipped", reason="KeyError: no group 'ece'",
                 t_cov0_s=float("nan"), t_cov1_s=float("nan"), n_events=0)],
        run_id="r1",
    )
    row = out.iloc[0]
    assert row["status"] == "skipped"
    assert row["reason"] == "KeyError: no group 'ece'"
    # Unknown, not zero: "nobody looked" is a third answer.
    assert np.isnan(row["t_cov0_s"]) and np.isnan(row["t_cov1_s"])


def test_a_merge_replaces_one_key_and_keeps_the_others(tmp_path):
    path = tmp_path / f"{SHOT}_sources.parquet"
    write_sources(path, SHOT, [
        _source(diag="mhr", channel=0, n_events=1),
        _source(diag="mhr", channel=4, n_events=2),
        _source(source="ece_sawtooth", diag="ece", channel=-1, pass_name="",
                n_events=47),
    ], run_id="r1")
    out = write_sources(
        path, SHOT, [_source(diag="mhr", channel=4, n_events=99)],
        run_id="r2",
    )
    got = {
        (r["source"], r["diag"], r["channel"]): (r["n_events"], r["run_id"])
        for _, r in out.iterrows()
    }
    assert got == {
        ("tokeye_track", "mhr", 0): (1, "r1"),
        ("tokeye_track", "mhr", 4): (99, "r2"),
        ("ece_sawtooth", "ece", -1): (47, "r1"),
    }


def test_the_same_key_may_not_be_written_twice_in_one_call(tmp_path):
    with pytest.raises(ValueError, match="two records for one"):
        write_sources(tmp_path / "s.parquet", SHOT,
                      [_source(), _source(n_events=9)], run_id="r")


def test_read_sources_of_a_missing_file_is_an_empty_typed_frame(tmp_path):
    got = read_sources(tmp_path / "nothing.parquet")
    assert got.empty and list(got.columns) == list(SOURCE_COLUMNS)
    assert got.dtypes.to_dict() == {
        c: np.dtype(SOURCE_DTYPES[c]) for c in SOURCE_COLUMNS
    }
    assert read_sources(tmp_path / "nothing.parquet", source="text").empty


@pytest.mark.parametrize("bad,match", [
    ({"status": "partial"}, "status"),
    ({"source": ""}, "source must not be empty"),
    ({"reason": "why"}, "an ok source carries no reason"),
    ({"status": "skipped", "reason": ""}, "must say why"),
    ({"pass_name": "medium"}, "pass_name"),
    ({"n_events": -1}, "n_events"),
    ({"t_cov0_s": 3.0, "t_cov1_s": 1.0}, "t_cov1_s must not precede"),
])
def test_a_record_that_cannot_be_true_is_refused(tmp_path, bad, match):
    with pytest.raises(ValueError, match=match):
        write_sources(tmp_path / "s.parquet", SHOT, [_source(**bad)],
                      run_id="r")
