"""A detector's legacy diagnostic establishes neither coverage nor a phenomenon hit."""

import pytest

from shot_design.labels import event_sources as es
from shot_design.mcp import tools
from shot_design.retrieval import phenomena as ph
from shot_design.shotdb import store

from .test_phenomena import _claim, _db_with, _event, _registry_file


@pytest.fixture(autouse=True)
def _reset_mcp():
    tools.reset_cache()
    yield
    tools.reset_cache()


@pytest.mark.parametrize('value', [
    'filterscopes', 42, False, None, {}, ['filterscopes', 7],
    ['filterscopes', None], [['filterscopes']],
])
def test_coverage_diags_rejects_non_lists_and_non_strings(tmp_path, value):
    def change(doc):
        doc['phenomena']['elm']['coverage_diags'] = value

    with pytest.raises(ph.PhenomenaError, match='elm.*coverage_diags'):
        ph.registry(_registry_file(tmp_path, change))


def test_coverage_diags_parses_and_governs_another_registered_detector(ideate_db, tmp_path):
    def change(doc):
        doc['phenomena']['sawtooth']['coverage_diags'] = ['mhr', 'filterscopes']

    entry = ph.registry(_registry_file(tmp_path, change))['sawtooth']
    assert entry.coverage_diags == ('mhr', 'filterscopes')
    db = _db_with(ideate_db, [
        _event(100, 'allowed', source='ece_sawtooth', phenomenon='sawtooth', diag='mhr'),
        _event(101, 'rejected', source='ece_sawtooth', phenomenon='sawtooth', diag='ece'),
    ])
    accepted = ph.evidence(100, entry, db)
    assert [iv.event_id for iv in accepted.intervals] == ['allowed']
    assert accepted.coverage_state == 'observed'
    rejected = ph.evidence(101, entry, db)
    assert rejected.intervals == ()
    assert rejected.coverage_state == 'unprocessed'


def _clock_coverage_db(ideate_db, diag, source_table):
    events = [] if source_table else [_event(
        100, 'clock-free', source='elm_clock', phenomenon='elm_free',
        evidence_kind='heuristic', diag=diag, t0_s=0, t1_s=6,
    )]
    _db_with(ideate_db, events, claims=[_claim(100, 'elm')])
    if source_table:
        es.write_sources(ideate_db / 'db/event_sources.parquet', [
            es.source_row(100, 'elm_clock', diag=diag, t_cov0_s=0, t_cov1_s=6),
        ])
    return store.ShotDB.load(ideate_db / 'db')


@pytest.mark.parametrize('source_table', [False, True])
@pytest.mark.parametrize('diag,state', [('mhr', 'unprocessed'), ('filterscopes', 'observed')])
def test_coverage_diags_controls_locate_coverage(ideate_db, diag, state, source_table):
    db = _clock_coverage_db(ideate_db, diag, source_table)
    ev = ph.evidence(100, 'elm', db)
    assert ev.coverage_state == state
    assert ev.coverage == ((1.0, 5.0) if state == 'observed' else None)
    assert ev.intervals == ()
    (hit,) = ph.locate('elm', db)
    assert hit.shot == 100
    assert hit.coverage_state == state
    assert hit.intervals == []


@pytest.mark.parametrize('source_table', [False, True])
@pytest.mark.parametrize('diag,state', [('mhr', 'unprocessed'), ('filterscopes', 'observed')])
def test_coverage_diags_controls_mcp_coverage(ideate_db, diag, state, source_table):
    _clock_coverage_db(ideate_db, diag, source_table)
    reply = tools.get_events(100, phenomenon='elm', t0_s=1, t1_s=5)
    assert reply['status'] == state
    assert reply['events'] == [] and reply['n'] == 0
    if diag == 'mhr':
        assert reply['coverage']['sources'] == []
        assert reply['coverage']['n_sources_ok'] == 0
        assert any('1 coverage row' in c and 'elm_clock/mhr' in c for c in reply['caveats'])
    else:
        assert reply['coverage']['n_sources_ok'] == 1
        assert reply['coverage']['sources'][0]['diag'] == 'filterscopes'
        assert not any('excluded' in c for c in reply['caveats'])


def _clock_hits_db(ideate_db, source_table):
    db = _db_with(ideate_db, [
        _event(100, 'legacy-1', source='elm_clock', phenomenon='elm', diag='mhr'),
        _event(100, 'legacy-2', source='elm_clock', phenomenon='elm', diag='mhr'),
        _event(101, 'dalpha', source='elm_clock', phenomenon='elm', diag='filterscopes'),
        # Non-covering claims must survive a diagnostic restriction on the detector.
        _event(100, 'written', source='text', phenomenon='elm', evidence_kind='text', diag=''),
        _event(100, 'predicted', source='label_forecast', phenomenon='elm',
               evidence_kind='forecast', diag=''),
    ])
    if source_table:
        es.write_sources(ideate_db / 'db/event_sources.parquet', [
            es.source_row(shot, 'elm_clock', diag='filterscopes', t_cov0_s=0, t_cov1_s=6)
            for shot in (100, 101)
        ])
        db = store.ShotDB.load(ideate_db / 'db')
    return db


@pytest.mark.parametrize('source_table', [False, True])
def test_coverage_diags_rejects_legacy_hits_in_evidence(ideate_db, source_table):
    db = _clock_hits_db(ideate_db, source_table)
    rejected = ph.evidence(100, 'elm', db)
    assert rejected.intervals == ()
    assert rejected.refs == ()
    assert rejected.event_weight == 0.0
    assert [iv.event_id for iv in rejected.forecasts] == ['predicted']
    accepted = ph.evidence(101, 'elm', db)
    assert [iv.event_id for iv in accepted.intervals] == ['dalpha']
    assert [ref.event_id for ref in accepted.refs] == ['dalpha']
    assert accepted.event_weight == 1.0
    hits = ph.locate('elm', db)
    assert [hit.shot for hit in hits if hit.intervals] == [101]


@pytest.mark.parametrize('source_table', [False, True])
def test_coverage_diags_rejects_legacy_hits_in_mcp_with_a_count(ideate_db, source_table):
    _clock_hits_db(ideate_db, source_table)
    rejected = tools.get_events(100, phenomenon='elm')
    assert rejected['events'] == [] and rejected['n'] == 0
    assert rejected['status'] == ('observed' if source_table else 'unprocessed')
    assert any('2 event row' in c and 'elm_clock/mhr' in c for c in rejected['caveats'])
    assert [row['event_id'] for row in rejected['text_mentions']] == ['written']
    assert [row['event_id'] for row in rejected['forecasts']] == ['predicted']
    accepted = tools.get_events(101, phenomenon='elm')
    assert accepted['status'] == 'observed'
    assert accepted['n'] == 1
    assert [row['event_id'] for row in accepted['events']] == ['dalpha']
    # Unfiltered event browsing still exposes the original diagnostic and provenance.
    raw = tools.get_events(100)
    assert [row['event_id'] for row in raw['events']] == ['legacy-1', 'legacy-2']


@pytest.mark.parametrize('source_table', [False, True])
@pytest.mark.parametrize('diag', ['mhr', 'ece', 'filterscopes', ''])
def test_empty_coverage_diags_leaves_sawtooth_rows_accepted(ideate_db, diag, source_table):
    db = _db_with(ideate_db, [_event(
        100, 'saw', source='ece_sawtooth', phenomenon='sawtooth', diag=diag,
    )])
    if source_table:
        es.write_sources(ideate_db / 'db/event_sources.parquet', [
            es.source_row(100, 'ece_sawtooth', diag=diag, t_cov0_s=0, t_cov1_s=6),
        ])
        db = store.ShotDB.load(ideate_db / 'db')
    assert ph.registry()['sawtooth'].coverage_diags == ()
    ev = ph.evidence(100, 'sawtooth', db)
    assert ev.coverage_state == 'observed'
    assert [iv.event_id for iv in ev.intervals] == ['saw']
    reply = tools.get_events(100, phenomenon='sawtooth')
    assert reply['status'] == 'observed'
    assert reply['n'] == 1
    assert reply['events'][0]['diag'] == diag
    assert not any('excluded' in c for c in reply['caveats'])


def test_an_ineligible_diagnostic_cannot_fill_an_eligible_sources_gap(ideate_db):
    es.write_sources(ideate_db / 'db/event_sources.parquet', [
        es.source_row(100, 'elm_clock', diag='filterscopes', t_cov0_s=0, t_cov1_s=6,
                      intervals='[[0, 0.9], [5.1, 6]]', min_gap_s=.003),
        es.source_row(100, 'elm_clock', diag='mhr', t_cov0_s=0, t_cov1_s=6,
                      intervals='[[0, 6]]', min_gap_s=.001),
    ])
    reply = tools.get_events(100, 'elm', 1, 5)
    assert reply['status'] == 'uncovered'
    db = store.ShotDB.load(ideate_db / 'db')
    assert ph.evidence(100, 'elm', db).coverage_state == 'uncovered'
