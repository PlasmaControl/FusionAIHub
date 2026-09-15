"""Hard filters distinguish a covered negative from an unexamined segment."""

import numpy as np
import pytest

from shot_design import cli
from shot_design.labels import event_sources as es
from shot_design.mcp import tools
from shot_design.retrieval import phenomena as ph
from shot_design.retrieval import rank
from shot_design.schema import QueryState

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
        es.source_row(100, 'elm_clock', diag='filterscopes', t_cov0_s=0, t_cov1_s=6),
        es.source_row(101, 'elm_clock', diag='filterscopes', t_cov0_s=0, t_cov1_s=.9),
        es.source_row(201, 'elm_clock', diag='filterscopes',
                      t_cov0_s=np.nan, t_cov1_s=np.nan),
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
        es.source_row(100, 'elm_clock', diag='filterscopes', t_cov0_s=0, t_cov1_s=6),
        es.source_row(101, 'ece_sawtooth', t_cov0_s=0, t_cov1_s=6),
    ])
    db = _db_with(ideate_db, [_event(
        100, 'elm', source='elm_clock', phenomenon='elm', diag='filterscopes',
        t0_s=2, t1_s=2,
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


def test_observed_tokens_keep_transient_and_elm_provenance_distinct(ideate_db):
    es.write_sources(ideate_db / 'db/event_sources.parquet', [
        es.source_row(101, 'elm_clock', diag='filterscopes', t_cov0_s=0, t_cov1_s=6),
    ])
    db = _db_with(ideate_db, [
        _event(100, 'transient', source='tokeye_transient', phenomenon='transient'),
        _event(101, 'elm', source='elm_clock', phenomenon='elm', diag='filterscopes'),
    ])

    assert db.segments.loc[
        db.mask('flat_top', require_labels=['phenomenon:transient']), 'shot'
    ].tolist() == [100]
    assert db.segments.loc[
        db.mask('flat_top', require_labels=['phenomenon:elm']), 'shot'
    ].tolist() == [101]
    assert db.segments.loc[
        db.mask('flat_top', require_labels=['source:tokeye_transient']), 'shot'
    ].tolist() == [100]
    assert db.segments.loc[
        db.mask('flat_top', require_labels=['source:elm_clock']), 'shot'
    ].tolist() == [101]
    assert not ph.evidence(100, 'elm', db).intervals
    elm = ph.evidence(101, 'elm', db)
    assert [iv.source for iv in elm.intervals] == ['elm_clock']
    assert ph.TRANSIENT_NOT_CLASSIFIED not in elm.caveats
    elm_report = rank.search_report(QueryState(avoid_labels={'phenomenon:elm'}), db)
    assert any('dropped 1' in c and 'observed' in c for c in elm_report['caveats'])
    assert not any(ph.TRANSIENT_NOT_CLASSIFIED in c for c in elm_report['caveats'])
    transient_report = rank.search_report(
        QueryState(avoid_labels={'phenomenon:transient'}), db
    )
    assert any('dropped 1' in c and 'observed' in c for c in transient_report['caveats'])
    assert not any(ph.TRANSIENT_NOT_CLASSIFIED in c for c in transient_report['caveats'])


def test_require_rejects_unknown_phenomena_before_building_tokens(ideate_db):
    db = _db_with(ideate_db, [])
    with pytest.raises(ph.PhenomenaError, match='sawtoot.*nearest.*sawtooth'):
        db.mask('flat_top', require_labels=['phenomenon:sawtoot'])
    assert '_label_tokens' not in db.__dict__
    assert not db.mask('flat_top', require_labels=['phenomenon:sawtooth']).any()


def test_mcp_require_returns_an_error_with_the_bad_id_and_suggestions(ideate_db):
    _db_with(ideate_db, [])
    tools.reset_cache()
    reply = tools.search_shots(ref_shot=100, require_labels=['phenomenon:sawtoot'])
    assert 'sawtoot' in reply.get('error', '')
    assert 'nearest' in reply['error'] and 'sawtooth' in reply['error']
    assert any('sawtoot' in c for c in reply['caveats'])
    tools.reset_cache()


@pytest.mark.parametrize('spans,eligible', [
    ([(1, 1.02)], False),
    ([(1, 2.96)], False),
    ([(1, 3)], True),
    ([(1, 4)], True),
    ([(0, 1)], False),
    ([(1, 1.5), (4.5, 5)], False),  # the hull hides a three-second gap
    ([(1, 2.5), (1.5, 3)], True),  # union: exactly half, not the sum
    ([(1, 2), (1.5, 2.5)], False),  # double counting would admit this
])
def test_avoid_requires_half_the_segment_measured_without_changing_its_state(
    ideate_db, spans, eligible,
):
    es.write_sources(ideate_db / 'db/event_sources.parquet', [
        es.source_row(100, 'elm_clock', diag='filterscopes', t_cov0_s=a, t_cov1_s=b)
        for a, b in spans
    ])
    db = _db_with(ideate_db, [])
    assert ph.evidence(100, 'elm', db).coverage_state == 'observed'
    mask = db.mask('flat_top', avoid_labels=['phenomenon:elm'])
    assert db.segments.loc[mask, 'shot'].tolist() == ([100] if eligible else [])


def test_retained_partial_negative_carries_its_fraction_on_the_shot(ideate_db, capsys):
    es.write_sources(ideate_db / 'db/event_sources.parquet', [
        es.source_row(101, 'elm_clock', diag='filterscopes', t_cov0_s=1, t_cov1_s=3),
        es.source_row(201, 'elm_clock', diag='filterscopes', t_cov0_s=0, t_cov1_s=6),
    ])
    db = _db_with(ideate_db, [])
    query = QueryState(ref_shot=100, avoid_labels={'phenomenon:elm'})
    result = rank.search(query, db)
    entries = {item.shot: item.model_dump() for item in result.items}
    assert set(entries) == {101, 201}
    assert any('50.0%' in c and 'phenomenon:elm' in c for c in entries[101].get('caveats', []))
    assert entries[201]['caveats'] == []
    tools.reset_cache()
    reply = tools.search_shots(ref_shot=100, avoid_labels=['phenomenon:elm'])
    item = next(r for r in reply['results'] if r['shot'] == 101)
    assert any('50.0%' in c for c in item['caveats'])
    cli.main(['query', '--ref', '100', '--avoid', 'phenomenon:elm'])
    assert '50.0%' in capsys.readouterr().out
    tools.reset_cache()


@pytest.mark.parametrize('option', ['--require', '--avoid'])
def test_cli_phenomenon_errors_do_not_offer_irrelevant_column_advice(ideate_db, capsys, option):
    assert cli.main(['query', '--ref', '100', option, 'phenomenon:sawtoot']) == 2
    error = capsys.readouterr().err
    assert 'sawtoot' in error and 'nearest' in error
    assert 'Columns are' not in error


def test_avoid_coverage_runs_only_once_per_mcp_search(ideate_db, monkeypatch):
    db = _avoid_db(ideate_db)
    calls = []
    original = type(db)._avoid_coverage

    def counted(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(db), '_avoid_coverage', counted)
    tools.reset_cache()
    reply = tools.search_shots(ref_shot=101, avoid_labels=['phenomenon:elm'])
    assert [item['shot'] for item in reply['results']] == [100]
    assert any('uncovered' in c for c in reply['caveats'])
    assert len(calls) == 1
    tools.reset_cache()
