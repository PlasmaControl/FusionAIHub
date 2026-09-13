"""The MCP server: the three tool functions, and one real stdio roundtrip.

Two levels, deliberately. The tool FUNCTIONS are tested directly against the synthetic database
(`conftest.ideate_db`) because that is where the behaviour is -- what a caveat says, what a
missing table returns, that a forecast never lands in `events`. The SERVER is tested once, over
a real `python -m ideate.mcp` subprocess, because the things a wrapper gets wrong are invisible
in-process: a tool whose signature will not turn into a JSON schema, a module that writes to
stdout and corrupts the transport, an entry point that does not exist.

Nothing here reaches the network or the real store: `IDEATE_DATA_ROOT` points at `tmp_path`
throughout, including in the subprocess's environment.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ideate.mcp import server as server_mod
from ideate.mcp import tools

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_cached_db():
    """Each test loads its own database. The server caches one per directory on purpose."""
    tools.reset_cache()
    yield
    tools.reset_cache()


# ------------------------------------------------------------------------------ events fixture


def _event(shot: int, phenomenon: str, t0: float, t1: float, **over) -> dict:
    from labelmaker.events import schema as events_schema

    row = {
        "shot": shot,
        "event_id": over.pop("event_id", f"{shot}-x-00001"),
        "source": over.pop("source", "tokeye_track"),
        "evidence_kind": over.pop("evidence_kind", "detector"),
        "phenomenon": phenomenon,
        "t0_s": t0,
        "t1_s": t1,
        "f0_khz": np.nan,
        "f1_khz": np.nan,
        "confidence": 0.9,
        "horizon_s": np.nan,
        "diag": "mhr",
        "channel": 0,
        "pass_name": "wide",
        "attrs": json.dumps(over.pop("attrs", {"amp": 1.0}), sort_keys=True),
        "t_cov0_s": 0.0,
        "t_cov1_s": 6.0,
        "run_id": "r1",
        "git_sha": "abc",
        "written_at": "2026-09-07T00:00:00+00:00",
    }
    row.update(over)
    assert set(row) == set(events_schema.COLUMNS)
    return row


def write_events(db_dir: Path, rows: list[dict]) -> None:
    from labelmaker.events import schema as events_schema

    df = pd.DataFrame(rows, columns=list(events_schema.COLUMNS)).astype(events_schema.DTYPES)
    df.to_parquet(db_dir / "events.parquet", index=False)


# ------------------------------------------------------------------------------- search_shots


def test_search_shots_ranks_and_always_carries_caveats(ideate_db):
    got = tools.search_shots(ref_shot=100, n=3)
    assert "error" not in got
    assert isinstance(got["caveats"], list)
    assert got["n"] == len(got["results"]) <= 3
    assert 100 not in [r["shot"] for r in got["results"]]  # a shot is not its own neighbour
    assert {"shot", "segment", "score", "description"} <= set(got["results"][0])


def test_search_shots_with_nothing_to_search_on_says_so_rather_than_returning_nothing(ideate_db):
    """A model that gets `[]` will try again with different words. It has to be told that no
    CHANNEL fired -- that the query carried no text, no reference and no constraint -- because
    rephrasing cannot fix that and the next call would be wasted."""
    got = tools.search_shots()
    assert got["results"] == [] and got["n"] == 0
    assert any("no channel" in c for c in got["caveats"])


def test_search_shots_takes_constraints_as_a_mapping_or_a_pair(ideate_db):
    a = tools.search_shots(ref_shot=100, constraints={"ip_mean": {"lo": 1.3e6}}, n=5)
    b = tools.search_shots(ref_shot=100, constraints={"ip_mean": [1.3e6, None]}, n=5)
    # The same two shots in the same order: the two spellings are one constraint, not two.
    assert [r["shot"] for r in a["results"]] == [r["shot"] for r in b["results"]]
    assert sorted(r["shot"] for r in a["results"]) == [200, 201]  # 100/101 are under 1.3 MA


def test_search_shots_reports_an_unknown_constraint_column_as_an_error_not_an_exception(
    ideate_db,
):
    got = tools.search_shots(ref_shot=100, constraints={"nope_mean": {"lo": 1.0}})
    assert "nope_mean" in got["error"] and isinstance(got["caveats"], list)


def test_search_shots_filters_on_labels(ideate_db):
    got = tools.search_shots(ref_shot=100, require_labels=["L"], avoid_labels=["dud"], n=5)
    assert [r["shot"] for r in got["results"]] == [200]


def test_a_reference_shot_the_database_does_not_hold_is_an_error_with_the_hint(ideate_db):
    got = tools.search_shots(ref_shot=999999)
    assert "999999" in got["error"] and "ideate add" in got["error"]


def test_an_unconstrained_search_does_not_pass_over_the_table_again_for_its_report(
    ideate_db, monkeypatch
):
    """`search_report` recomputed the hard filter (and two more column passes) purely to report
    the candidate count, on top of the pass every channel makes for itself. A query that
    constrains NOTHING admits every row of its segment, which is one comparison on one column --
    so the tool adds no pass of its own, and the count it reports is the same number."""
    from ideate.retrieval import channels
    from ideate.retrieval import rank as rank_mod
    from ideate.shotdb.store import ShotDB

    real, calls = channels.hard_filter, []

    def counting(q, db):
        calls.append(q.segment)
        return real(q, db)

    monkeypatch.setattr(channels, "hard_filter", counting)
    monkeypatch.setattr(rank_mod, "hard_filter", counting)
    got = tools.search_shots(ref_shot=100, n=3)
    by_the_tool = len(calls)

    calls.clear()
    db = ShotDB.load(ideate_db / "db")
    rank_mod.search(
        __import__("ideate.schema", fromlist=["QueryState"]).QueryState(
            ref_shot=100, segment="flat_top", n=3
        ),
        db,
    )
    assert by_the_tool == len(calls)  # the tool itself filters not once more than the search does
    assert got["query"]["candidates"] == 4  # every flat_top row: nothing was filtered away


# ------------------------------------------------------------------------------ describe_shot


def test_describe_shot_returns_the_record_and_the_prose(ideate_db):
    got = tools.describe_shot(100)
    assert got["shot"] == 100 and got["segment"] == "flat_top"
    assert got["record"]["shot"] == 100 and got["record"]["human"]["run_id"] == "r1"
    assert isinstance(got["description"], str) and got["description"]
    assert got["caveats"] == []


def test_describe_shot_accepts_the_segment_name_a_model_is_likely_to_type(ideate_db):
    """`flattop` is what a model writes; `flat_top` is what the schema calls it. Taking the
    first and saying so beats an error the model cannot act on."""
    got = tools.describe_shot(100, segment="flattop")
    assert got["segment"] == "flat_top"
    assert any("flat_top" in c for c in got["caveats"])


def test_an_unknown_segment_name_is_an_error_that_lists_the_ones_there_are(ideate_db):
    got = tools.describe_shot(100, segment="middle")
    assert "middle" in got["error"] and "flat_top" in got["error"]


def test_describe_shot_on_a_shot_that_is_not_there_is_an_error_dict(ideate_db):
    got = tools.describe_shot(999999)
    assert "999999" in got["error"] and got["caveats"] == []


def test_describe_shot_says_which_device_encoded_the_frame_codes(ideate_db, monkeypatch):
    """The four-key cache payload records no device, and the codes are not bit-identical across
    devices or across BLAS thread counts. A description that says "this shot is encoded" without
    saying how is the state the encode product shipped in."""
    from ideate import config
    from ideate.design import provenance

    codes = Path(config.load_paths().data_root) / "frame_codes"
    codes.mkdir(parents=True, exist_ok=True)
    (codes / "100.pt").write_bytes(b"")
    provenance.write_sidecar(
        codes, 100, provenance.build_sidecar(100, device="cuda", input_file=None, bundle=None)
    )
    (codes / "101.pt").write_bytes(b"")  # encoded, but nobody recorded how

    got = tools.describe_shot(100)
    assert got["frame_codes"]["present"] is True
    assert got["frame_codes"]["device"] == "cuda"
    assert not [c for c in got["caveats"] if "provenance sidecar" in c]

    unknown = tools.describe_shot(101)
    assert unknown["frame_codes"]["present"] is True
    assert unknown["frame_codes"]["device"] is None
    assert any("no provenance sidecar" in c for c in unknown["caveats"])

    none_at_all = tools.describe_shot(200)
    assert none_at_all["frame_codes"] == {"present": False, "device": None, "path": None}


def test_a_missing_database_is_the_error_the_cli_prints(tmp_path, monkeypatch):
    monkeypatch.setenv("IDEATE_DATA_ROOT", str(tmp_path / "empty"))
    monkeypatch.delenv("IDEATE_PATHS", raising=False)
    got = tools.describe_shot(100)
    assert "no database" in got["error"] and "ideate build" in got["error"]
    assert got["caveats"] == []


# --------------------------------------------------------------------------------- get_events


def write_sources(db_dir: Path, rows: list[dict]) -> None:
    """`db/event_sources.parquet` -- what `ideate labels join` ingests from labelmaker."""
    from ideate.labels import event_sources as es

    es.write_sources(db_dir / "event_sources.parquet", rows)


def _source(shot: int, name: str = "tokeye_track", **over) -> dict:
    from ideate.labels import event_sources as es

    over.setdefault("t_cov0_s", 0.0)
    over.setdefault("t_cov1_s", 6.0)
    return es.source_row(shot, name, **over)


def test_get_events_without_a_database_is_the_error_the_cli_prints(tmp_path, monkeypatch):
    """The tool now needs the shot index to tell an unindexed shot from an unexamined one, so a
    missing database is the same error `describe_shot` and `search_shots` give."""
    monkeypatch.setenv("IDEATE_DATA_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv("IDEATE_PATHS", raising=False)
    got = tools.get_events(shot=1)
    assert "error" in got and "ideate build" in got["error"]


def test_get_events_without_an_events_table_says_the_join_has_not_run(ideate_db):
    got = tools.get_events(shot=100)
    assert got["events"] == [] and got["n"] == 0 and got["forecasts"] == []
    assert "no events table yet (labelmaker events not joined)" in got["caveats"]


# --------------------------------------------------------- the four states of an empty answer


def test_a_shot_the_database_does_not_hold_is_unindexed_not_quiet(ideate_db):
    """THE DEFECT. `get_events(198658)` returned `{"n": 0, "caveats": []}` for a shot that is not
    in the 500-shot database at all, while `describe_shot(198658)` correctly said so. An
    assistant reading the two together learns that the shot is in the database and was quiet."""
    write_events(ideate_db / "db", [_event(100, "tearing", 1.0, 2.0)])
    got = tools.get_events(198658)
    assert got["status"] == "unindexed"
    assert "error" in got and "not in the database" in got["error"]


def test_an_indexed_shot_nobody_processed_is_unprocessed(ideate_db):
    """No source rows and no non-forecast events: nobody ran a detector over this shot, so its
    empty event list is not a report that the shot was quiet."""
    write_events(
        ideate_db / "db",
        [
            _event(
                100, "disruption", 3.0, 3.0, event_id="100-f-00001",
                source="label_forecast", evidence_kind="forecast", horizon_s=0.2,
            )
        ],
    )
    got = tools.get_events(100)
    assert got["status"] == "unprocessed"
    assert got["n"] == 0 and got["n_forecasts"] == 1
    assert any("Absence is not evidence" in c for c in got["caveats"])
    assert got["coverage"]["n_sources_ok"] == 0
    assert got["coverage"]["has_observed_products"] is False


def test_a_shot_where_only_the_text_ran_is_unprocessed_not_observed(ideate_db):
    """labelmaker records `text` as a source that RAN (over the shot's own span). A lexicon hit is
    not a detector, so a shot with a logbook and no detector run is `unprocessed` - its mention
    is in `text_mentions` and its empty `events` is not "0 detections inside coverage"."""
    write_sources(ideate_db / "db", [_source(100, "text", t_cov0_s=0.0, t_cov1_s=6.5, n_events=1)])
    write_events(
        ideate_db / "db",
        [_event(100, "elm", 0.0, 0.0, source="text", evidence_kind="text", diag="", channel=-1,
                pass_name="")],
    )
    got = tools.get_events(100)
    assert got["status"] == "unprocessed"
    assert got["n"] == 0 and got["n_text_mentions"] == 1
    assert got["coverage"]["has_observed_products"] is False
    assert got["coverage"]["t_cov0_s"] is None
    assert any("Absence is not evidence" in c for c in got["caveats"])


def test_a_window_outside_every_sources_coverage_is_uncovered_and_names_the_span(ideate_db):
    write_sources(ideate_db / "db", [_source(100, t_cov0_s=1.0, t_cov1_s=4.0)])
    write_events(ideate_db / "db", [_event(100, "tearing", 1.5, 2.0)])

    got = tools.get_events(100, t0_s=8.0, t1_s=9.0)
    assert got["status"] == "uncovered"
    assert got["n"] == 0
    assert any("outside every source's coverage" in c and "1.0 to 4.0" in c for c in got["caveats"])
    assert got["coverage"]["t_cov0_s"] == 1.0 and got["coverage"]["t_cov1_s"] == 4.0

    inside = tools.get_events(100, t0_s=1.5, t1_s=2.5)
    assert inside["status"] == "observed" and inside["n"] == 1


def test_a_source_that_ran_and_saw_nothing_says_so_in_as_many_words(ideate_db):
    """The state the whole contract exists for: somebody looked, over a known span, and there was
    nothing to see. That is an observation, and it must not read like an unexamined shot."""
    write_sources(
        ideate_db / "db",
        [_source(100, "tokeye_track", n_events=0), _source(100, "ece_sawtooth", n_events=0)],
    )
    got = tools.get_events(100)
    assert got["status"] == "observed"
    assert got["n"] == 0
    assert got["coverage"]["n_sources_ok"] == 2
    assert any("0 detections inside their coverage" in c for c in got["caveats"])


def test_a_source_that_failed_is_reported_rather_than_counted_as_coverage(ideate_db):
    write_sources(
        ideate_db / "db",
        [
            _source(100, "tokeye_track", n_events=0),
            _source(100, "ece_sawtooth", status="error", reason="ece read failed",
                    t_cov0_s=float("nan"), t_cov1_s=float("nan")),
            _source(100, "dalpha_lh", status="skipped", reason="no d_alpha on this shot",
                    t_cov0_s=float("nan"), t_cov1_s=float("nan")),
        ],
    )
    got = tools.get_events(100)
    assert got["coverage"]["n_sources_ok"] == 1
    assert got["coverage"]["n_sources_error"] == 1
    assert got["coverage"]["n_sources_skipped"] == 1
    assert any("FAILED on shot 100" in c for c in got["caveats"])
    reasons = {s["source"]: s["reason"] for s in got["coverage"]["sources"]}
    assert reasons["dalpha_lh"] == "no d_alpha on this shot"


def test_an_ok_source_whose_coverage_is_unknown_never_makes_a_window_observed(ideate_db):
    """THE CORNER, from real shot 198658: `actuator/ech_power_total` is `ok` with NaN coverage --
    it ran, and nothing records over what span. The reply used to say `observed` and "0 detections
    inside their coverage" while its own `coverage` block said the coverage was None. A source
    that cannot say what it looked at can neither cover nor un-cover the window, so the window is
    NOT observed; it is `uncovered`, and the caveat names the source rather than implying it saw
    nothing."""
    write_sources(
        ideate_db / "db",
        [_source(100, "actuator", diag="ech_power_total", n_events=0,
                 t_cov0_s=float("nan"), t_cov1_s=float("nan"))],
    )
    write_events(ideate_db / "db", [])

    got = tools.get_events(100, t0_s=3.0, t1_s=4.0)
    assert got["status"] == "uncovered"
    assert got["coverage"]["has_observed_products"] is True, "it did run: this is not unprocessed"
    assert got["coverage"]["t_cov0_s"] is None
    assert got["coverage"]["n_sources_unknown_coverage"] == 1
    assert any("actuator" in c and "coverage unknown" in c for c in got["caveats"])
    assert not any("0 detections inside their coverage" in c for c in got["caveats"])

    whole_shot = tools.get_events(100)
    assert whole_shot["status"] == "uncovered", "no window is not a covered window"


def test_the_unknown_coverage_caveat_is_there_even_beside_real_coverage(ideate_db):
    """A shot with one honest span and one unknown one is observed over the span -- and the
    reader still has to be told that one source's coverage is unrecorded, because "2 sources ran"
    would otherwise read as two sources having looked at the window."""
    write_sources(
        ideate_db / "db",
        [
            _source(100, "actuator", diag="ech_power_total", n_events=0,
                    t_cov0_s=float("nan"), t_cov1_s=float("nan")),
            _source(100, "ece_sawtooth", diag="ece", t_cov0_s=0.0, t_cov1_s=6.0, n_events=0),
        ],
    )
    write_events(ideate_db / "db", [])

    got = tools.get_events(100, t0_s=1.0, t1_s=2.0)
    assert got["status"] == "observed"
    assert got["coverage"]["n_sources_unknown_coverage"] == 1
    assert any("actuator" in c and "coverage unknown" in c for c in got["caveats"])
    assert any("1 source(s) ran" in c and "0 detections" in c for c in got["caveats"])


def test_the_no_detection_caveat_counts_only_what_covered_the_window(ideate_db):
    """`n_sources_ok` is a count of what RAN -- the logbook lexicon and an unknown-coverage
    actuator included. Putting it in a sentence about detections INSIDE COVERAGE inflates the
    observation: on real 198658 it would have read "24 sources ran ... 0 detections"."""
    write_sources(
        ideate_db / "db",
        [
            _source(100, "text", t_cov0_s=0.0, t_cov1_s=6.5, n_events=1),
            _source(100, "tokeye_track", t_cov0_s=0.0, t_cov1_s=6.0, n_events=0),
            _source(100, "ece_sawtooth", diag="ece", t_cov0_s=5.0, t_cov1_s=6.0, n_events=0),
        ],
    )
    write_events(ideate_db / "db", [])

    got = tools.get_events(100, t0_s=1.0, t1_s=2.0)
    assert got["status"] == "observed"
    assert got["coverage"]["n_sources_ok"] == 3
    assert any("1 source(s) ran" in c and "0 detections inside their coverage" in c
               for c in got["caveats"]), got["caveats"]


def test_a_curated_database_listing_never_makes_a_shot_observed(ideate_db):
    """A `database:<stem>` source is a published table of shots, not a detector, and a
    `evidence_kind="database"` row is its entry. Neither is a diagnostic having looked, so a shot
    whose only completed source is a curated list is `unprocessed`: nobody ran a detector over
    it, and its empty `events` is not an observation of nothing."""
    write_sources(
        ideate_db / "db",
        [_source(100, "database:rwm_database", n_events=1,
                 t_cov0_s=float("nan"), t_cov1_s=float("nan"))],
    )
    write_events(
        ideate_db / "db",
        [_event(100, "rwm", 2.0, 2.0, event_id="100-d-00001", source="database:rwm_database",
                evidence_kind="database", diag="", channel=-1, pass_name="",
                t_cov0_s=float("nan"), t_cov1_s=float("nan"))],
    )

    got = tools.get_events(100)
    assert got["status"] == "unprocessed"
    assert got["coverage"]["has_observed_products"] is False
    assert got["coverage"]["n_sources_unknown_coverage"] == 0
    assert any("Absence is not evidence" in c for c in got["caveats"])
    assert not any("0 detections inside their coverage" in c for c in got["caveats"])


def test_a_reversed_or_non_finite_window_is_an_error_not_a_silent_empty(ideate_db):
    """A reversed window used to come back as a successful empty result, which reads exactly like
    "nothing happened in that interval" -- for an interval that does not exist."""
    write_events(ideate_db / "db", [_event(100, "tearing", 1.0, 2.0)])

    reversed_ = tools.get_events(100, t0_s=5.0, t1_s=1.0)
    assert "error" in reversed_ and "reversed" in reversed_["error"]
    assert any("not a report that the window was quiet" in c for c in reversed_["caveats"])

    assert "error" in tools.get_events(100, t0_s=0.0, t1_s=0.0)
    nan = tools.get_events(100, t0_s=float("nan"), t1_s=2.0)
    assert "error" in nan and "not finite" in nan["error"]
    assert "error" in tools.get_events(100, t1_s=float("inf"))


def test_a_text_row_is_a_lexicon_hit_and_never_lands_in_events(ideate_db):
    """The API documentation claimed every non-forecast row describes what a diagnostic showed.
    A text row describes what somebody WROTE, which is not the same claim and not the same
    evidence."""
    write_sources(ideate_db / "db", [_source(100)])
    write_events(
        ideate_db / "db",
        [
            _event(100, "tearing", 1.0, 2.0, event_id="100-x-00001"),
            _event(100, "tearing", 1.0, 2.0, event_id="100-t-00001",
                   source="text", evidence_kind="text"),
        ],
    )
    got = tools.get_events(100)
    assert [e["event_id"] for e in got["events"]] == ["100-x-00001"]
    assert [e["event_id"] for e in got["text_mentions"]] == ["100-t-00001"]
    assert got["n"] == 1 and got["n_text_mentions"] == 1
    assert any("not an assertion that the phenomenon occurred" in c for c in got["caveats"])


def test_get_events_filters_by_shot_and_decodes_attrs(ideate_db):
    write_events(
        ideate_db / "db",
        [
            _event(100, "tearing", 1.0, 2.0, attrs={"m": 2, "n": 1}),
            _event(101, "elm", 1.0, 1.0, event_id="101-x-00001"),
        ],
    )
    got = tools.get_events(shot=100)
    assert got["n"] == 1 and [e["shot"] for e in got["events"]] == [100]
    assert got["events"][0]["attrs"] == {"m": 2, "n": 1}  # decoded, not the JSON string


def test_get_events_filters_by_phenomenon_and_by_time_overlap(ideate_db):
    write_events(
        ideate_db / "db",
        [
            _event(100, "tearing", 1.0, 2.0, event_id="100-x-00001"),
            _event(100, "tearing", 4.0, 5.0, event_id="100-x-00002"),
            _event(100, "elm", 1.5, 1.5, event_id="100-y-00001"),
        ],
    )
    assert [e["event_id"] for e in tools.get_events(100, phenomenon="tearing")["events"]] == [
        "100-x-00001",
        "100-x-00002",
    ]
    # Overlap, not containment: an event straddling the window edge is in the window.
    window = tools.get_events(100, t0_s=1.8, t1_s=4.2)["events"]
    assert [e["event_id"] for e in window] == ["100-x-00001", "100-x-00002"]
    # ... and a POINT EVENT (t1 == t0, an L-H transition) is in a window whose edge touches it.
    # The window itself may not be a point -- `t0_s < t1_s` is validated, because a zero-width
    # window and a reversed one are the same typo and neither can be answered honestly.
    assert [e["event_id"] for e in tools.get_events(100, t0_s=1.5, t1_s=2.5)["events"]] == [
        "100-x-00001",
        "100-y-00001",
    ]


def test_a_forecast_is_never_returned_as_an_observed_event(ideate_db):
    """The single most important thing this tool does. `label_forecast` rows are a MODEL's claim
    about what was about to happen, computed by the join out of the labels; a detector row is
    somebody's claim about what a diagnostic showed. Handing an assistant one table with both in
    it is how "shot 100 disrupted at 3.2 s" gets written from a risk curve."""
    write_events(
        ideate_db / "db",
        [
            _event(100, "tearing", 1.0, 2.0, event_id="100-x-00001"),
            _event(
                100, "disruption", 3.0, 3.0, event_id="100-f-00001",
                source="label_forecast", evidence_kind="forecast", horizon_s=0.2,
            ),
        ],
    )
    got = tools.get_events(100)
    assert [e["event_id"] for e in got["events"]] == ["100-x-00001"]
    assert [e["event_id"] for e in got["forecasts"]] == ["100-f-00001"]
    assert got["n"] == 1 and got["n_forecasts"] == 1
    assert any("forecast" in c for c in got["caveats"])


def test_a_shot_with_no_events_in_the_table_is_not_an_error(ideate_db):
    write_events(ideate_db / "db", [_event(100, "tearing", 1.0, 2.0)])
    got = tools.get_events(101)
    assert got["events"] == [] and got["n"] == 0 and "error" not in got
    assert got["status"] == "unprocessed"


def test_an_unreadable_events_table_is_an_error_dict_not_an_exception(ideate_db):
    (ideate_db / "db" / "events.parquet").write_bytes(b"not parquet")
    got = tools.get_events(100)
    assert "error" in got and isinstance(got["caveats"], list)


def test_a_row_with_no_recorded_time_is_counted_out_of_a_window_not_dropped_in_silence(ideate_db):
    """A NaN compares False against both bounds, so a row whose times were never recorded
    vanishes from a windowed call looking exactly like a row that did not overlap. "We do not
    know when this happened" is not "this did not happen then", and the difference is the whole
    reason `search_shots` carries `nan_excluded`; this is the same hole in the other tool."""
    write_events(
        ideate_db / "db",
        [
            _event(100, "tearing", 1.0, 2.0, event_id="100-x-00001"),
            _event(100, "tearing", np.nan, np.nan, event_id="100-x-00002"),
        ],
    )
    got = tools.get_events(100, t0_s=0.0, t1_s=5.0)
    assert [e["event_id"] for e in got["events"]] == ["100-x-00001"]
    assert got["nan_excluded"] == 1
    assert any("no recorded time" in c for c in got["caveats"])
    # No window: the timeless row is not excluded from anything, and says so with a zero.
    whole = tools.get_events(100)
    assert whole["n"] == 2 and whole["nan_excluded"] == 0
    assert not any("no recorded time" in c for c in whole["caveats"])


def test_a_missing_value_of_any_pandas_flavour_becomes_null(ideate_db):
    """`_event_row` mapped non-finite FLOATS to null. A `pd.NaT` or a `pd.NA` -- what a datetime
    or a nullable-integer column hands back, which the events schema will grow one day -- went
    through untouched and would raise inside the JSON encoder, i.e. inside the transport."""
    row = tools._event_row(
        {"t0_s": np.nan, "written_at": pd.NaT, "channel": pd.NA, "phenomenon": "tearing"}
    )
    assert row == {"t0_s": None, "written_at": None, "channel": None, "phenomenon": "tearing"}
    json.dumps(row)  # the point: it survives the encoder


# ------------------------------------------------------------------------------- the registry


def test_every_registered_tool_is_a_documented_function_with_annotated_arguments():
    """The docstring and the annotations ARE the tool schema -- an assistant sees nothing else.
    An unannotated argument becomes an untyped schema slot the model has to guess at."""
    assert [fn.__name__ for fn in server_mod.TOOLS] == [
        "search_shots",
        "describe_shot",
        "get_events",
    ]
    for fn in server_mod.TOOLS:
        assert (fn.__doc__ or "").strip(), fn.__name__
        sig = inspect.signature(fn)
        for name, p in sig.parameters.items():
            assert p.annotation is not inspect.Parameter.empty, f"{fn.__name__}.{name}"


def _registered(name):
    """The tool as the SERVER holds it -- i.e. wrapped in whatever guards registration adds."""
    return {fn.__name__: fn for fn in server_mod.TOOLS}[name]


def test_a_registered_tool_that_raises_anything_at_all_answers_with_an_error_dict():
    """The module's headline promise, for the exception nobody anticipated. Each tool catches a
    NAMED set; anything outside it reached the framework, which answered the model
    `is_error=True, "Error executing tool describe_shot"` -- no `caveats` key, no sentence, and
    nothing to act on. The guard has to sit at registration so the plain function keeps raising
    for its own tests, and has to use `functools.wraps` so the signature and docstring -- which
    ARE the tool schema -- survive it."""
    from ideate.mcp.tools import never_raises

    @never_raises
    def boom(shot: int, segment: str = "flat_top") -> dict:
        """Raise something nobody caught."""
        raise RuntimeError("the parquet writer was mid-flight")

    got = boom(shot=1)
    assert got["error"] == "RuntimeError: the parquet writer was mid-flight"
    assert isinstance(got["caveats"], list)
    assert boom.__name__ == "boom" and boom.__doc__.startswith("Raise something")
    # `inspect.signature` follows `__wrapped__`, so the schema mcp builds is the plain
    # function's: the same parameters, in order, with their defaults. (The annotations are
    # strings here only because this test module imports `annotations` from `__future__`.)
    params = inspect.signature(boom).parameters
    assert list(params) == ["shot", "segment"] and params["segment"].default == "flat_top"


def test_a_half_published_database_is_an_error_dict_naming_the_rebuild(tmp_path, monkeypatch):
    """`db/manifest.json` present, the tables not yet: `ideate build` publishes per file, so a
    build in flight IS this state. `_db()`'s existence check passes and `ShotDB.load` then
    raises on `shots.parquet` -- which is the unanticipated exception, on a state the database
    is really in."""
    root = tmp_path / "half"
    (root / "db").mkdir(parents=True)
    (root / "db" / "manifest.json").write_text(json.dumps({"n_shots": 4}), encoding="utf-8")
    monkeypatch.setenv("IDEATE_DATA_ROOT", str(root))
    monkeypatch.delenv("IDEATE_PATHS", raising=False)
    got = _registered("describe_shot")(shot=100)
    assert "FileNotFoundError" in got["error"] and "shots.parquet" in got["error"]
    assert any("rebuilt" in c and "ideate build" in c for c in got["caveats"])


def test_an_events_table_with_the_wrong_columns_is_an_error_dict_not_a_key_error(ideate_db):
    """`pd.read_parquet` is inside the try; `df["shot"]` is not. A readable table written by
    something else -- an older schema, another tool's parquet -- raises `KeyError` from a line
    no `except` covers."""
    pd.DataFrame({"shot_number": [100], "t_start": [1.0]}).to_parquet(
        ideate_db / "db" / "events.parquet"
    )
    got = _registered("get_events")(shot=100)
    assert "KeyError" in got["error"] and "shot" in got["error"]
    assert isinstance(got["caveats"], list)


def test_the_server_registers_the_three_tools_and_the_manifest_resource(ideate_db):
    from mcp.client import Client

    async def go():
        async with Client(server_mod.build_server()) as client:
            tool_list = await client.list_tools()
            resources = await client.list_resources()
            manifest = await client.read_resource("ideate://manifest")
            return tool_list, resources, manifest

    tool_list, resources, manifest = asyncio.run(go())
    assert [t.name for t in tool_list.tools] == ["search_shots", "describe_shot", "get_events"]
    for t in tool_list.tools:
        assert t.description and t.input_schema["type"] == "object"
    assert [str(r.uri) for r in resources.resources] == ["ideate://manifest"]
    assert json.loads(manifest.contents[0].text)["reader"] == "test"
    # The guard at registration must not eat the schema: a `*args` wrapper that did would leave
    # every tool with an empty property set and the model guessing at argument names.
    by_name = {t.name: t for t in tool_list.tools}
    assert set(by_name["describe_shot"].input_schema["properties"]) == {"shot", "segment"}
    assert by_name["describe_shot"].input_schema["required"] == ["shot"]
    assert "logbook" in by_name["describe_shot"].description


def test_the_manifest_resource_carries_the_missing_database_error(tmp_path, monkeypatch):
    from mcp.client import Client

    monkeypatch.setenv("IDEATE_DATA_ROOT", str(tmp_path / "empty"))
    monkeypatch.delenv("IDEATE_PATHS", raising=False)

    async def go():
        async with Client(server_mod.build_server()) as client:
            return await client.read_resource("ideate://manifest")

    doc = json.loads(asyncio.run(go()).contents[0].text)
    assert "no database" in doc["error"]


# -------------------------------------------------------------------- the stdio roundtrip


def test_a_stdio_client_can_list_the_tools_and_call_one(tmp_path):
    """`python -m ideate.mcp` in a real subprocess, spoken to over stdin/stdout.

    The data root is the state the I7 review reproduced this failure on: `db/manifest.json`
    published and the tables not (a build writes per file, so a build in flight IS this). Two
    calls, because the two things that can go wrong here are different. `get_events` is the
    anticipated failure the tool words itself; `describe_shot` walks into `ShotDB.load`, which
    raises `FileNotFoundError` on `shots.parquet` -- and before the catch-all that came back as
    `is_error=True` with the string "Error executing tool describe_shot": no `caveats` key, no
    sentence, nothing the model could do next. Asserting `is_error is False` HERE, over the real
    transport, is the only place that distinction is visible at all.

    The guard is `asyncio.wait_for`: a server that hangs on start or never answers must fail
    this test in a minute rather than wedge the suite. A subprocess that cannot start at all
    (no interpreter, no package) is a skip, not a failure -- that is an environment fact.
    """
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters, get_default_environment

    root = tmp_path / "root"
    (root / "db").mkdir(parents=True)
    (root / "db" / "manifest.json").write_text(json.dumps({"n_shots": 4}), encoding="utf-8")
    env = get_default_environment()
    env.update(IDEATE_DATA_ROOT=str(root), HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    env.pop("IDEATE_PATHS", None)
    # THIS checkout's src first, exactly as the suite is run. Without it the subprocess imports
    # whichever `ideate` is installed in the environment -- the main checkout -- and the test
    # silently exercises somebody else's code, which is how a worktree passes a test it breaks.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "ideate.mcp"], cwd=str(REPO), env=env
    )

    async def go():
        async with Client(params) as client:
            tool_list = await client.list_tools()
            events = await client.call_tool("get_events", {"shot": 1})
            half = await client.call_tool("describe_shot", {"shot": 100})
            try:
                malformed = await client.call_tool("describe_shot", {"shot": "one hundred"})
            except Exception as exc:  # noqa: BLE001 - what the framework does IS the finding
                malformed = exc
            return tool_list, events, half, malformed

    try:
        tool_list, result, half, malformed = asyncio.run(asyncio.wait_for(go(), timeout=90))
    except (FileNotFoundError, PermissionError) as exc:  # pragma: no cover - environment
        pytest.skip(f"cannot spawn the server subprocess: {exc}")
    except TimeoutError:  # pragma: no cover - a hung server
        pytest.fail("the stdio server did not answer within 90 s")

    assert [t.name for t in tool_list.tools] == ["search_shots", "describe_shot", "get_events"]
    # Both tools now walk into `ShotDB.load` -- `get_events` needs the shot index to tell an
    # unindexed shot from an unexamined one -- and both come back as the SAME application error
    # rather than the framework's bare "Error executing tool <name>".
    for outcome in (result, half):
        assert not outcome.is_error
        doc = json.loads(outcome.content[0].text)
        assert "FileNotFoundError" in doc["error"] and "shots.parquet" in doc["error"]
        assert any("rebuilt" in c and "ideate build" in c for c in doc["caveats"])

    # ... and the documented LIMIT of that promise, exercised rather than asserted in prose: an
    # argument of the wrong TYPE never reaches the function, so it cannot carry `caveats`. The
    # framework rejects it -- as a raised error or as `is_error` -- and `INSTRUCTIONS` says so.
    if isinstance(malformed, Exception):
        assert "shot" in str(malformed) or "valid" in str(malformed).lower()
    else:
        assert malformed.is_error or "caveats" not in json.loads(malformed.content[0].text)


def test_the_instructions_do_not_promise_a_caveat_the_transport_cannot_deliver(ideate_db):
    """The server told models "every reply carries caveats". Application errors do; a call whose
    ARGUMENTS fail the tool schema is rejected by the framework before the function runs, and
    comes back with no `caveats` key at all. Documenting the limit is the fix that was chosen
    over normalising it -- normalising would mean loosening every annotation to `str | int`, and
    the annotations ARE the schema an assistant reads."""
    text = " ".join(server_mod.INSTRUCTIONS.split())
    assert "Every reply the tools THEMSELVES produce carries `caveats`" in text
    assert "does NOT cover a call whose arguments do not match the tool's schema" in text
    assert "no `caveats` field" in text
    # The four states and the three lists, named where a model will actually read them.
    for word in tools.EVENT_STATES:
        assert word in text
    for word in ("events", "text_mentions", "forecasts", "status"):
        assert word in text

    # ... and the promise that IS made holds for a failure nobody anticipated.
    guarded = server_mod.never_raises(
        lambda shot: (_ for _ in ()).throw(RuntimeError("something nobody wrote an except for"))
    )
    assert "caveats" in guarded(shot=100)


def test_the_project_mcp_config_points_at_this_server():
    """`.mcp.json` is what makes `claude` in this checkout see the server at all. It is the one
    file nothing else in the suite would notice going stale."""
    cfg = json.loads((REPO / ".mcp.json").read_text(encoding="utf-8"))
    entry = cfg["mcpServers"]["ideate"]
    assert entry["args"][-2:] == ["-m", "ideate.mcp"]
    assert Path(entry["cwd"]).resolve() == REPO
    assert os.path.isabs(entry["cwd"])
