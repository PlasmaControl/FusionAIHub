"""A phenomenon channel votes only for resolved, evidenced, eligible segments."""

import math

import pytest

from ideate.retrieval import channels, rank
from ideate.schema import QueryState, Range

from .test_phenomena import _claim, _db_with, _event, _label_row


def test_the_channel_uses_the_existing_score_and_tier_order(ideate_db):
    db = _db_with(ideate_db, [
        _event(200, 'seen'),
        _event(101, 'risk', source='label_forecast', evidence_kind='forecast', phenomenon='tearing'),
    ], labels=[_label_row(100, 'tm_prob', max_valid=.8)], claims=[
        _claim(201, 'tearing'), _claim(200, 'tearing'),
    ])
    q = QueryState(text='tearing mode')
    got = channels.CHANNELS['phenomenon'](q, db)
    assert [sid for sid, _ in got] == [
        '200:flat_top', '100:flat_top', '101:flat_top', '201:flat_top',
    ]
    assert got[0][1] == pytest.approx(1 - math.exp(-1 / 3) + .5 * math.tanh(.5))
    assert got[1][1] == pytest.approx(.8)
    assert got[2][1] == 0  # class order keeps a forecast above the text-only score
    assert rank.load_cfg()['weights']['phenomenon'] == 1.2


def test_no_resolved_phenomenon_is_exactly_no_vote_in_rrf(ideate_db, monkeypatch):
    db = _db_with(ideate_db, [_event(100, 'seen')])
    from ideate.shotdb import text

    monkeypatch.setattr(text, 'embed_texts', lambda texts: db.emb['text_log'][:1])
    q = QueryState(text='plasma current', ref_shot=101)
    assert channels.CHANNELS['phenomenon'](q, db) == []
    with_channel = rank.search(q, db)
    monkeypatch.delitem(channels.CHANNELS, 'phenomenon')
    without = rank.search(q, db)
    assert with_channel.items == without.items
    assert {k: v for k, v in with_channel.rankings.items() if k != 'phenomenon'} == without.rankings


def test_the_channel_applies_the_full_hard_filter_before_its_candidate_limit(ideate_db):
    db = _db_with(ideate_db, [_event(shot, str(shot)) for shot in (100, 101, 200, 201)])
    q = QueryState(text='tearing mode', constraints={'ip_mean': Range(lo=1.3e6)},
                   avoid_labels={'dud'}, exclude_shots={201})
    assert [sid for sid, _ in channels.CHANNELS['phenomenon'](q, db)] == ['200:flat_top']


def test_resolution_denials_and_text_negatives_do_not_create_positive_hits(ideate_db):
    db = _db_with(ideate_db, [_event(100, 'seen')])
    assert channels.CHANNELS['phenomenon'](QueryState(text='no tearing mode'), db) == []
    assert channels.CHANNELS['phenomenon'](QueryState(text='plasma without tearing'), db) == []
