"""Hard filters distinguish a covered negative from an unexamined segment."""

import numpy as np
import pytest

from ideate import cli
from ideate.labels import event_sources as es
from ideate.mcp import tools
from ideate.retrieval import phenomena as ph
from ideate.retrieval import rank
from ideate.schema import QueryState

from .test_phenomena import _claim, _db_with, _event, _label_row


def test_tokens_are_segment_scoped_and_only_observations_make_phenomenon_tokens(ideate_db):
    db = _db_with(ideate_db, [
        _event(100, 'early', t0_s=.2, t1_s=.4),
        _event(101, 'forecast', source='label_forecast', phenomenon='tearing',
               evidence_kind='forecast'),
        _event(200, 'observed'),
    ], claims=[_claim(201, 'tearing')])
    mask = db.mask('flat_top', require_labels=['phenomenon:tearing'])
    assert db.segments.loc[mask, 'shot'].tolist() == [200]
    mask = db.mask('ramp_up', require_labels=['phenomenon:tearing'])
    assert db.segments.loc[mask, 'shot'].tolist() == [100]
    for token in ('source:detector', 'source:tokeye_track'):
        mask = db.mask('flat_top', require_labels=[token])
        assert db.segments.loc[mask, 'shot'].tolist() == [200]
    mask = db.mask('flat_top', require_labels=['source:forecast'])
    assert db.segments.loc[mask, 'shot'].tolist() == [101]


def test_label_tokens_require_valid_probability_above_the_operating_point(ideate_db):
    db = _db_with(ideate_db, [], labels=[
        _label_row(100, 'tm_prob', max_valid=.8, thr=.7),
        _label_row(101, 'tm_prob', max_valid=.6, thr=.7),
        _label_row(200, 'tm_prob', max_valid=.9, n_valid=0, thr=.7),
        _label_row(201, 'tm_prob', max_valid=.6),  # registered detection floor .5
    ])
    mask = db.mask('flat_top', require_labels=['label:d3d_tearing_onset_cnn1d/tm_prob'])
    assert db.segments.loc[mask, 'shot'].tolist() == [100, 201]


def _avoid_db(ideate_db):
    es.write_sources(ideate_db / 'db/event_sources.parquet', [
        es.source_row(100, 'elm_clock', t_cov0_s=0, t_cov1_s=6),
        es.source_row(101, 'elm_clock', t_cov0_s=0, t_cov1_s=.9),
        es.source_row(201, 'elm_clock', t_cov0_s=np.nan, t_cov1_s=np.nan),
    ])
    return _db_with(ideate_db, [])


def test_avoid_requires_relevant_coverage_and_reports_each_excluded_state(ideate_db):
    db = _avoid_db(ideate_db)
    q = QueryState(text='plasma', avoid_labels={'phenomenon:elm'})
    mask = db.mask(q.segment, avoid_labels=q.avoid_labels)
    assert db.segments.loc[mask, 'shot'].tolist() == [100]
    report = rank.search_report(q, db)
    assert report['candidates'] == 1
    assert any('unprocessed' in c and '1' in c for c in report['caveats'])
    assert any('uncovered' in c and '2' in c for c in report['caveats'])
    assert any('coverage unknown' in c for c in report['caveats'])


def test_avoid_excludes_observed_events_and_never_borrows_another_detector(ideate_db):
    es.write_sources(ideate_db / 'db/event_sources.parquet', [
        es.source_row(100, 'elm_clock', t_cov0_s=0, t_cov1_s=6),
        es.source_row(101, 'ece_sawtooth', t_cov0_s=0, t_cov1_s=6),
    ])
    db = _db_with(ideate_db, [_event(
        100, 'elm', source='tokeye_transient', phenomenon='elm', t0_s=2, t1_s=2,
    )])
    assert not db.mask('flat_top', avoid_labels=['phenomenon:elm']).any()
    assert not db.mask('flat_top', avoid_labels=['phenomenon:rwm']).any()
    q = QueryState(text='plasma', avoid_labels={'phenomenon:rwm'})
    assert any('no detector registered for rwm' in c for c in rank.search_report(q, db)['caveats'])


def test_token_sets_and_coverage_are_cached_across_channel_masks(ideate_db, monkeypatch):
    db = _avoid_db(ideate_db)
    first = db.mask('flat_top', avoid_labels=['phenomenon:elm'])
    tokens = db._label_tokens

    def no_evidence(*args, **kwargs):
        pytest.fail('a second mask must reuse precomputed evidence')

    monkeypatch.setattr(ph, 'evidence', no_evidence)
    np.testing.assert_array_equal(first, db.mask('flat_top', avoid_labels=['phenomenon:elm']))
    assert db._label_tokens is tokens


def test_query_and_mcp_surface_coverage_exclusions_even_with_no_results(ideate_db, capsys):
    _avoid_db(ideate_db)
    tools.reset_cache()
    reply = tools.search_shots(ref_shot=100, avoid_labels=['phenomenon:rwm'])
    assert reply['results'] == []
    assert any('unprocessed' in c for c in reply['caveats'])
    cli.main(['query', '--ref', '100', '--avoid', 'phenomenon:rwm', '--json'])
    assert 'no detector registered for rwm' in capsys.readouterr().out
    tools.reset_cache()
