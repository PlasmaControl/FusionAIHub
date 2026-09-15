"""Retrieval and MCP use the same shot, source and window coverage contract."""

import numpy as np
import pandas as pd
import pytest

from shot_design.labels import event_sources as es
from shot_design.mcp import tools
from shot_design.retrieval import phenomena as ph
from shot_design.shotdb.store import ShotDB


@pytest.fixture
def gap_chain(ideate_db, tmp_path, monkeypatch):
    from shot_design import cli
    from labeler.config import Paths
    from labeler.events import pipeline

    from ..labeler.coverage_fixture import gapped_filterscopes
    from .conftest import shot_record, write_db

    paths = Paths(root=tmp_path / "products", corpus=tmp_path / "corpus")
    monkeypatch.setenv("LABELMAKER_ROOT", str(paths.root))
    expected = gapped_filterscopes(paths.corpus)
    paths.labels.mkdir(parents=True)
    result = pipeline.process_shot(198658, paths, model=None, passes=("wide",))
    assert not result.error
    records = [shot_record(s, "run", 1e6, 5e6, "ELMs and sawteeth")
               for s in (198658, 101, 200, 201)]
    # Named segments give evidence/locate/describe the SAME windows as MCP.
    from shot_design.schema import Segment
    records[0].segments = []
    for name, a, b in (("flat_top", 1.2, 1.8), ("ramp_up", .5, 1.5), ("ramp_down", 2.5, 3.0)):
        records[0].segments.append(Segment(name=name, t0_ms=a * 1000, t1_ms=b * 1000))
    write_db(ideate_db / "db", records)
    assert cli.main([
        "labels", "join", "--shots", "198658", "--labeler-root", str(paths.root),
        "--db", str(ideate_db / "db"), "--no-text",
    ]) == 0
    tools.reset_cache()
    yield ideate_db, expected
    tools.reset_cache()


@pytest.mark.parametrize("segment,a,b,status,partial", [
    ("flat_top", 1.2, 1.8, "uncovered", False),
    ("ramp_up", .5, 1.5, "observed", True),
    ("ramp_down", 2.5, 3.0, "observed", False),
])
def test_real_pipeline_join_and_all_consumers_preserve_the_gap(
    gap_chain, segment, a, b, status, partial,
):
    import json

    from shot_design.retrieval.describe import describe

    root, expected = gap_chain
    db = ShotDB.load(root / "db")
    row = db.event_sources.query("source == 'elm_clock'").iloc[0]
    result = tools.get_events(198658, "elm", a, b)
    assert result["status"] == status
    assert json.loads(row.intervals) == [list(i) for i in expected]
    assert result["coverage_partial"] is partial
    ev = ph.evidence(198658, "elm", db, segment=segment)
    assert ev.coverage_state == status and ev.coverage_partial is partial
    direct = ph._coverage_for(db, 198658, ph.registry()["elm"], (a, b), segment)
    assert direct[0] == status and direct[3] is partial
    description = describe(db.get(198658), segment=segment, db=db)
    assert f"coverage: {status}" in description
    hits = ph.locate("elm", db, segment=segment)
    if status == "uncovered":
        assert result["n"] == 0
        assert not any("This IS an observation" in c for c in result["caveats"])
        assert any("covered intervals" in c for c in result["caveats"])
        assert "covered intervals" in description
        assert not any(h.shot == 198658 and h.evidence.intervals for h in hits)
    if partial:
        assert any("covered only" in c for c in result["caveats"])
        assert "covered only" in description
    print(json.dumps({"window": [a, b], "status": status, "coverage_partial": partial,
                      "caveats": result["caveats"]}))


def test_real_stdio_mcp_preserves_the_pipeline_dropout(gap_chain):
    import asyncio
    import json
    import os
    import sys
    from pathlib import Path

    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters, get_default_environment

    root, _ = gap_chain
    repo = Path(__file__).resolve().parents[2]
    env = get_default_environment()
    env.update(IDEATE_DATA_ROOT=str(root), LABELMAKER_ROOT=str(root.parent / "products"),
               HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(repo / "src"), env.get("PYTHONPATH", "")]))
    env.pop("IDEATE_PATHS", None)
    params = StdioServerParameters(command=sys.executable, args=["-m", "shot_design.mcp"],
                                   cwd=str(repo), env=env)

    async def go():
        async with Client(params) as client:
            answers = []
            for a, b in ((1.2, 1.8), (.5, 1.5), (2.5, 3.0)):
                result = await client.call_tool("get_events", {
                    "shot": 198658, "phenomenon": "elm", "t0_s": a, "t1_s": b,
                })
                assert not result.is_error
                answers.append(json.loads(result.content[0].text))
            return answers

    results = asyncio.run(asyncio.wait_for(go(), timeout=90))
    assert [r["status"] for r in results] == ["uncovered", "observed", "observed"]
    assert [r["coverage_partial"] for r in results] == [False, True, False]
    assert any("covered intervals" in c for c in results[0]["caveats"])
    assert any("covered only" in c for c in results[1]["caveats"])


def test_new_event_fallback_cannot_reintroduce_the_pipeline_hull(gap_chain):
    root, _ = gap_chain
    (root / "db/event_sources.parquet").unlink()
    tools.reset_cache()
    result = tools.get_events(198658, "elm", 1.2, 1.8)
    assert result["status"] == "uncovered"
    assert not any("older writer" in c for c in result["caveats"])
    db = ShotDB.load(root / "db")
    assert ph.evidence(198658, "elm", db).coverage_state == "uncovered"


def test_whole_record_searches_agree_that_interior_gaps_are_partial(gap_chain):
    root, _ = gap_chain
    db = ShotDB.load(root / "db")
    result = tools.get_events(198658, "elm")
    ev = ph.evidence(198658, "elm", db, segment="full")  # absent: searches whole record
    assert result["status"] == ev.coverage_state == "observed"
    assert result["coverage_partial"] is ev.coverage_partial is True


@pytest.mark.parametrize("bounds,state,partial,gaps", [
    ((1.2, 1.8), "uncovered", False, 0),
    ((.5, 1.5), "observed", True, 0),
    ((.5, 3.0), "observed", True, 1),
    ((2.5, 3.0), "observed", False, 0),
    ((None, None), "observed", True, 1),
    ((.5, None), "observed", True, 1),
    ((None, 3.0), "observed", True, 1),
    ((2.5, None), "observed", False, 0),
    ((None, .5), "observed", False, 0),
    ((7., None), "uncovered", False, 0),
    ((None, -.1), "uncovered", False, 0),
])
def test_mcp_and_retrieval_share_window_decisions_and_qualifications(
    gap_chain, bounds, state, partial, gaps,
):
    root, _ = gap_chain
    db = ShotDB.load(root / "db")
    result = tools.get_events(198658, "elm", *bounds)
    direct = ph._coverage_for(db, 198658, ph.registry()["elm"],
                              None if bounds == (None, None) else bounds, "requested")
    assert result["status"] == direct[0] == state
    assert result["coverage_partial"] is direct[3] is partial
    assert result["coverage_windows"] == [list(w) for w in direct[2]]
    for caveats in (result["caveats"], direct[4]):
        assert sum("gap(s)" in c for c in caveats) == bool(gaps)
        assert sum("covered only" in c for c in caveats) == partial
    def qualifications(cs):
        return [c for c in cs if "gap(s)" in c or "covered only" in c]

    assert qualifications(result["caveats"]) == qualifications(direct[4])


def test_gap_qualification_does_not_confuse_source_and_requested_hulls(gap_chain):
    _, expected = gap_chain
    result = tools.get_events(198658, "elm", .5, 3.0)
    assert result["coverage"]["t_cov0_s"] == expected[0][0]
    assert result["coverage"]["t_cov1_s"] == expected[-1][1]
    assert result["coverage_windows"] == [[.5, expected[0][1]], [expected[1][0], 3.0]]
    caveat, = [c for c in result["caveats"] if "gap(s)" in c]
    assert "`coverage` is the hull of `coverage_windows`" not in caveat
    assert "disjoint" in caveat and "coverage_windows" in caveat


def test_mcp_evidence_and_describe_call_the_same_window_function(gap_chain, monkeypatch):
    from shot_design.retrieval.describe import describe

    root, _ = gap_chain
    db = ShotDB.load(root / "db")
    real = ph.coverage_for_sources
    calls = []

    def tracked(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(ph, "coverage_for_sources", tracked)
    tools.get_events(198658, "elm", .5, 1.5)
    assert len(calls) == 1
    calls.clear()
    ph.evidence(198658, "elm", db, segment="ramp_up")
    assert len(calls) == 1
    calls.clear()
    describe(db.get(198658), segment="ramp_up", db=db)
    assert calls, "describe must reach the shared decision through evidence"


def test_text_only_empty_coverage_is_unprocessed_and_never_unknown(ideate_db):
    from shot_design.retrieval.describe import describe

    es.write_sources(ideate_db / "db/event_sources.parquet", [
        es.source_row(100, "text", intervals="[]", min_gap_s=0, n_events=1),
    ])
    tools.reset_cache()
    db = ShotDB.load(ideate_db / "db")
    summary = es.shot_summary(db.coverage_sources, 100)
    assert summary["n_sources_ok"] == 1
    assert summary["n_sources_unknown_coverage"] == 0
    assert summary["has_observed_products"] is False
    for bounds in ((None, None), (1.2, 1.8)):
        assert es.coverage_state(db.coverage_sources, *bounds) == "unprocessed"
        assert es.unknown_coverage_rows(db.coverage_sources).empty
        result = tools.get_events(100, t0_s=bounds[0], t1_s=bounds[1])
        assert result["status"] == "unprocessed"
        assert result["coverage"]["n_sources_unknown_coverage"] == 0
        assert result["coverage_windows"] == []
        assert not result["coverage_partial"]
        assert not any("coverage unknown" in c or "nobody looked there" in c
                       for c in result["caveats"])
    ev = ph.evidence(100, "elm", db)
    assert ev.coverage_state == "unprocessed"
    assert not any("coverage unknown" in c for c in ev.caveats)
    description = describe(db.get(100), db=db)
    assert "coverage: uncovered" not in description
    assert "coverage unknown" not in description
    tools.reset_cache()


@pytest.mark.parametrize("legacy_events", [False, True])
def test_older_source_and_event_hulls_are_disclosed_by_all_readers(ideate_db, legacy_events):
    from shot_design.retrieval.describe import describe

    from .test_phenomena import _db_with, _elm_clock

    _db_with(ideate_db, [_elm_clock(100, "old", (0, 6))])
    if not legacy_events:
        old = pd.DataFrame([es.source_row(100, "elm_clock", diag="filterscopes",
                                          t_cov0_s=0, t_cov1_s=6)])
        old.drop(columns=["intervals", "min_gap_s"]).to_parquet(
            ideate_db / "db/event_sources.parquet", index=False)
    tools.reset_cache()
    db = ShotDB.load(ideate_db / "db")
    result = tools.get_events(100, "elm", 1.2, 1.8)
    ev = ph.evidence(100, "elm", db)
    assert result["status"] == ev.coverage_state == "observed"
    for caveats in (result["caveats"], ev.caveats):
        assert any(es.LEGACY_HULL_CAVEAT in c for c in caveats)
    rec = db.get(100)
    rec.human.log_entries[0].text = "ELMs"
    assert es.LEGACY_HULL_CAVEAT in describe(rec, db=db)
    tools.reset_cache()


@pytest.mark.parametrize('filtered', [True, False])
@pytest.mark.parametrize(
    'shot,pid,status,span,expected',
    [
        (999, 'elm', 'ok', (0, 6), 'unindexed'),
        (999, 'rwm', 'ok', (0, 6), 'unindexed'),
        (100, 'rwm', 'ok', (0, 6), 'unprocessed'),
        (100, 'elm', 'ok', (0, 6), 'observed'),
        (100, 'elm', 'skipped', (np.nan, np.nan), 'unprocessed'),
        (100, 'elm', 'ok', (0, .9), 'uncovered'),
        (100, 'elm', 'ok', (np.nan, np.nan), 'uncovered'),
        (100, 'elm', None, (np.nan, np.nan), 'unprocessed'),
        (100, 'elm', 'ok', (0, 1), 'observed'),
        (100, 'elm', 'ok', (np.inf, np.inf), 'uncovered'),
    ],
)
def test_both_tools_agree_on_the_report_tables_and_coverage_edges(
    ideate_db, shot, pid, status, span, expected, filtered,
):
    rows = [] if status is None else [es.source_row(
        100, 'elm_clock', diag='filterscopes', status=status,
        t_cov0_s=span[0], t_cov1_s=span[1],
    )]
    es.write_sources(ideate_db / 'db/event_sources.parquet', rows)
    tools.reset_cache()
    db = ShotDB.load(ideate_db / 'db')
    retrieval = ph.evidence(shot, pid, db)
    mcp = tools.get_events(shot, pid if filtered else None, 1, 5)
    if filtered:
        assert retrieval.coverage_state == mcp['status'] == expected
    else:
        # An unfiltered indexed RWM row may use elm_clock; RWM itself still cannot.
        unfiltered_expected = 'observed' if shot == 100 and pid == 'rwm' else expected
        assert retrieval.coverage_state == expected
        assert mcp['status'] == unfiltered_expected
    if filtered and shot == 100 and pid == 'rwm':
        caveat = 'no detector registered for rwm; text/database evidence only'
        assert caveat in retrieval.caveats and caveat in mcp['caveats']
    if status == 'ok' and not np.isfinite(span).all():
        assert any('coverage unknown' in c for c in retrieval.caveats)
        assert any('coverage unknown' in c for c in mcp['caveats'])
    tools.reset_cache()


def test_unknown_phenomenon_and_registered_no_detector_are_distinct(ideate_db):
    tools.reset_cache()

    unknown = tools.get_events(100, 'zzz_not_a_phenomenon')
    known = tools.get_events(100, 'rwm')
    assert unknown['status'] == known['status'] == 'unprocessed'
    assert any('unknown phenomenon id' in c and 'zzz_not_a_phenomenon' in c
               for c in unknown['caveats'])
    assert not any('no detector registered' in c for c in unknown['caveats'])
    assert 'no detector registered for rwm; text/database evidence only' in known['caveats']
    tools.reset_cache()


def test_former_caps_cannot_borrow_the_long_gas_span_or_invent_qh_completion(ideate_db):
    es.write_sources(ideate_db / "db/event_sources.parquet", [
        es.source_row(100, "ece_sawtooth", diag="ece", t_cov0_s=-.05, t_cov1_s=6.143,
                      intervals="[[-0.05, 6.143]]", min_gap_s=.008),
        es.source_row(100, "actuator", diag="gas_a", t_cov0_s=-10, t_cov1_s=94.9,
                      intervals="[[-10, 94.9]]", min_gap_s=.02),
        es.source_row(100, "qh_proxy", status="skipped", reason="no Ip flat-top"),
    ])
    tools.reset_cache()
    db = ShotDB.load(ideate_db / "db")
    for pid, window, status in (("sawtooth", (14, 15), "uncovered"),
                                ("qh", (3, 4), "unprocessed")):
        result = tools.get_events(100, pid, *window)
        assert result["status"] == status and result["n"] == 0
        assert ph._coverage_for(db, 100, ph.registry()[pid], window, "test")[0] == status
        assert not any(r["source"] == "actuator" for r in result["coverage"]["sources"])
    tools.reset_cache()


def test_an_unrelated_detector_only_covers_an_unfiltered_mcp_call(ideate_db):
    es.write_sources(ideate_db / 'db/event_sources.parquet', [
        es.source_row(100, 'ece_sawtooth', t_cov0_s=0, t_cov1_s=6),
    ])
    tools.reset_cache()
    assert tools.get_events(100, t0_s=1, t1_s=5)['status'] == 'observed'
    assert tools.get_events(100, 'elm', 1, 5)['status'] == 'unprocessed'
    assert ph.evidence(100, 'elm', ShotDB.load(ideate_db / 'db')).coverage_state == 'unprocessed'
    tools.reset_cache()


def test_sources_are_an_optional_typed_table_loaded_once(ideate_db, monkeypatch):
    db_dir = ideate_db / 'db'
    absent = ShotDB.load(db_dir)
    pd.testing.assert_frame_equal(absent.event_sources, es.empty_sources())
    assert absent.load_errors == {}
    es.write_sources(db_dir / 'event_sources.parquet', [
        es.source_row(100, 'elm_clock', diag='filterscopes', t_cov0_s=0, t_cov1_s=6),
    ])
    db = ShotDB.load(db_dir)
    assert len(db.event_sources) == 1

    def no_read(*args, **kwargs):
        pytest.fail('coverage must read the loaded table, not reopen parquet')

    monkeypatch.setattr(pd, 'read_parquet', no_read)
    assert ph.evidence(100, 'elm', db).coverage_state == 'observed'


def test_a_broken_sources_table_is_reported_by_both_readers(ideate_db):
    (ideate_db / 'db/event_sources.parquet').write_text('torn parquet')
    db = ShotDB.load(ideate_db / 'db')
    assert set(db.load_errors) == {'event_sources'}
    assert db.event_sources.empty
    tools.reset_cache()
    for caveats in (ph.evidence(100, 'elm', db).caveats, tools.get_events(100)['caveats']):
        assert any('event_sources' in c and 'could not read' in c for c in caveats)
    tools.reset_cache()


def test_legacy_event_coverage_agrees_and_counts_one_source_once(ideate_db):
    from .test_phenomena import _db_with, _elm_clock

    db = _db_with(ideate_db, [
        _elm_clock(100, 'quiet-a', (0, 6)),
        _elm_clock(100, 'quiet-b', (0, 6)),
    ])
    tools.reset_cache()
    result = tools.get_events(100, 'elm', 1, 5)
    assert result['status'] == ph.evidence(100, 'elm', db).coverage_state == 'observed'
    assert result['coverage']['n_sources_ok'] == 1
    tools.reset_cache()


def test_no_registered_detector_caveat_does_not_deny_returned_event_rows(ideate_db):
    from .test_phenomena import _db_with, _event

    _db_with(ideate_db, [_event(100, 'foreign-rwm', phenomenon='rwm')])
    tools.reset_cache()
    result = tools.get_events(100, 'rwm', 1, 5)
    assert result['status'] == 'unprocessed' and result['n'] == 1
    assert not any('has no detector or heuristic rows' in c for c in result['caveats'])
    assert 'no detector registered for rwm; text/database evidence only' in result['caveats']
    tools.reset_cache()
