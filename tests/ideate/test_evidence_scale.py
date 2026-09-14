"""Loaded detector evidence must scale by rows, not shots times full-table scans."""

import gc
from time import perf_counter

import pandas as pd
import pytest

from ideate.labels import event_sources as es
from ideate.retrieval import channels
from ideate.retrieval import phenomena as ph
from ideate.schema import QueryState
from ideate.shotdb.store import ShotDB

from .conftest import shot_record, write_db
from .test_phenomena import _claim, _event, _label_row, _write_tables


def write_scale_db(root, n=2000):
    """Two segments, several sources, forecasts, labels and text on every shot."""
    root.mkdir(parents=True, exist_ok=True)
    records, events, sources, labels, claims = [], [], [], [], []
    for i in range(n):
        shot = 200000 + i
        records.append(shot_record(shot, f'r{i // 20}', 1e6 + i, 5e6, 'sawtooth'))
        events.extend([
            _event(shot, f'{shot}-saw', source='ece_sawtooth', phenomenon='sawtooth'),
            _event(shot, f'{shot}-track'),
            _event(shot, f'{shot}-risk', source='label_forecast',
                   phenomenon='tearing', evidence_kind='forecast'),
        ])
        sources.extend(es.source_row(shot, source, t_cov0_s=0, t_cov1_s=6)
                       for source in ('ece_sawtooth', 'elm_clock', 'tokeye_track'))
        labels.append(_label_row(shot, 'tm_prob'))
        claims.append(_claim(shot, 'tearing'))
    write_db(root, records)
    _write_tables(root, events, labels, claims)
    es.write_sources(root / 'event_sources.parquet', sources)


@pytest.fixture(scope='module')
def scale_db_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp('evidence-scale') / 'db'
    write_scale_db(root)
    return root


@pytest.mark.parametrize('operation,budget', [('tokens', 3), ('avoid', .5), ('channel', .5)])
def test_loaded_evidence_time_budget(scale_db_dir, operation, budget):
    db = ShotDB.load(scale_db_dir)
    # Configuration parsing is process startup; evidence/index construction stays timed.
    ph.registry()
    _ = db._phenomenon_config
    ph.resolve('sawtooth')
    # Isolate each fresh-snapshot measurement from garbage left by earlier suite tests.
    # Collection remains enabled while timing this operation's own index/evidence allocations.
    gc.collect()
    start = perf_counter()
    if operation == 'tokens':
        got = db._label_tokens
        assert len(got) == 4000
        assert sum('phenomenon:sawtooth' in tokens for tokens in got) == 2000
    elif operation == 'avoid':
        got = db._avoid_coverage('flat_top', ['phenomenon:elm'])
        assert got.sum() == 4000  # no ELMs; all flat tops have finite coverage
    else:
        got = channels.CHANNELS['phenomenon'](QueryState(text='sawtooth'), db)
        assert len(got) == channels.CANDIDATES
        assert got[0][0] == '200000:flat_top'
    elapsed = perf_counter() - start
    print(f'{operation}: {elapsed:.6f} s (2000 shots)')
    assert elapsed < budget


def test_evidence_queries_never_rescan_full_frames(scale_db_dir, monkeypatch):
    db = ShotDB.load(scale_db_dir)
    frames = (db.events, db.event_sources, db.labels_wide, db.text_claims)
    scans = [0]
    original = pd.DataFrame.__getitem__
    original_records = pd.DataFrame.to_dict

    def count_scan(frame, key):
        if any(frame is full for full in frames):
            scans[0] += 1
        return original(frame, key)

    monkeypatch.setattr(pd.DataFrame, '__getitem__', count_scan)

    def count_records(frame, *args, **kwargs):
        if any(frame is full for full in frames):
            scans[0] += 1
        return original_records(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, 'to_dict', count_records)
    first = ph.evidence(200000, 'tearing', db)
    assert len(es.for_shot(db, 200000)) == 3
    assert len(first.intervals) == len(first.forecasts) == first.text_hits == 1
    initial = scans[0]
    for shot in range(200001, 200021):
        ev = ph.evidence(shot, 'tearing', db)
        assert len(ev.intervals) == len(ev.forecasts) == ev.text_hits == 1
        assert ev.coverage_state == 'observed'
        assert len(es.for_shot(db, shot)) == 3
    assert scans[0] == initial, 'per-shot queries must not scan a full evidence frame'


def test_evidence_cache_is_bounded_and_eviction_preserves_results(scale_db_dir):
    db = ShotDB.load(scale_db_dir)
    first = db.phenomenon_evidence(200000, 'tearing', 'flat_top')
    for shot in range(200000, 202000):
        for pid in ('tearing', 'sawtooth', 'elm'):
            db.phenomenon_evidence(shot, pid, 'flat_top')
    assert len(db._phenomenon_evidence) <= 4096
    assert db.phenomenon_evidence(200000, 'tearing', 'flat_top') == first
