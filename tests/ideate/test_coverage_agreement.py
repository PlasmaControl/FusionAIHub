"""Retrieval and MCP use the same shot, source and window coverage contract."""

import numpy as np
import pandas as pd
import pytest

from ideate.labels import event_sources as es
from ideate.mcp import tools
from ideate.retrieval import phenomena as ph
from ideate.shotdb.store import ShotDB


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
    ideate_db, shot, pid, status, span, expected,
):
    rows = [] if status is None else [es.source_row(
        100, 'elm_clock', status=status, t_cov0_s=span[0], t_cov1_s=span[1],
    )]
    es.write_sources(ideate_db / 'db/event_sources.parquet', rows)
    tools.reset_cache()
    db = ShotDB.load(ideate_db / 'db')
    retrieval = ph.evidence(shot, pid, db)
    mcp = tools.get_events(shot, pid, 1, 5)
    assert retrieval.coverage_state == mcp['status'] == expected
    if shot == 100 and pid == 'rwm':
        caveat = 'no detector registered for rwm; text/database evidence only'
        assert caveat in retrieval.caveats and caveat in mcp['caveats']
    if status == 'ok' and not np.isfinite(span).all():
        assert any('coverage unknown' in c for c in retrieval.caveats)
        assert any('coverage unknown' in c for c in mcp['caveats'])
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
        es.source_row(100, 'elm_clock', t_cov0_s=0, t_cov1_s=6),
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
