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


def test_a_missing_database_is_the_error_the_cli_prints(tmp_path, monkeypatch):
    monkeypatch.setenv("IDEATE_DATA_ROOT", str(tmp_path / "empty"))
    monkeypatch.delenv("IDEATE_PATHS", raising=False)
    got = tools.describe_shot(100)
    assert "no database" in got["error"] and "ideate build" in got["error"]
    assert got["caveats"] == []


# --------------------------------------------------------------------------------- get_events


def test_get_events_without_a_table_says_the_join_has_not_run(tmp_path, monkeypatch):
    monkeypatch.setenv("IDEATE_DATA_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv("IDEATE_PATHS", raising=False)
    got = tools.get_events(shot=1)
    assert got["events"] == [] and got["n"] == 0 and got["forecasts"] == []
    assert got["caveats"] == ["no events table yet (labelmaker events not joined)"]


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
    assert [e["event_id"] for e in tools.get_events(100, t0_s=1.5, t1_s=1.5)["events"]] == [
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
    got = tools.get_events(999999)
    assert got["events"] == [] and got["n"] == 0 and "error" not in got


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
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "ideate.mcp"], cwd=str(REPO), env=env
    )

    async def go():
        async with Client(params) as client:
            tool_list = await client.list_tools()
            events = await client.call_tool("get_events", {"shot": 1})
            half = await client.call_tool("describe_shot", {"shot": 100})
            return tool_list, events, half

    try:
        tool_list, result, half = asyncio.run(asyncio.wait_for(go(), timeout=90))
    except (FileNotFoundError, PermissionError) as exc:  # pragma: no cover - environment
        pytest.skip(f"cannot spawn the server subprocess: {exc}")
    except TimeoutError:  # pragma: no cover - a hung server
        pytest.fail("the stdio server did not answer within 90 s")

    assert [t.name for t in tool_list.tools] == ["search_shots", "describe_shot", "get_events"]
    assert not result.is_error
    doc = json.loads(result.content[0].text)
    assert doc["events"] == []
    assert doc["caveats"] == ["no events table yet (labelmaker events not joined)"]

    assert not half.is_error  # NOT the framework's "Error executing tool describe_shot"
    doc = json.loads(half.content[0].text)
    assert "FileNotFoundError" in doc["error"] and "shots.parquet" in doc["error"]
    assert any("rebuilt" in c and "ideate build" in c for c in doc["caveats"])


def test_the_project_mcp_config_points_at_this_server():
    """`.mcp.json` is what makes `claude` in this checkout see the server at all. It is the one
    file nothing else in the suite would notice going stale."""
    cfg = json.loads((REPO / ".mcp.json").read_text(encoding="utf-8"))
    entry = cfg["mcpServers"]["ideate"]
    assert entry["args"][-2:] == ["-m", "ideate.mcp"]
    assert Path(entry["cwd"]).resolve() == REPO
    assert os.path.isabs(entry["cwd"])
