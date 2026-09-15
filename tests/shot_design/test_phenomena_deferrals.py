"""The deferred corpus predicate, raw label probability, and relevant quote excerpt."""

from dataclasses import replace

import pytest

from shot_design import cli
from shot_design.retrieval import phenomena as ph

from .test_phenomena import _claim, _db_with, _event, _label_row


def test_missing_required_groups_are_caveated_when_the_corpus_was_inventoried(ideate_db):
    db = _db_with(ideate_db, [_event(100, 'seen')])
    db.shots.loc[100, 'reader'] = 'corpus'
    db.shots.loc[100, 'has_mhr'] = False
    ev = ph.evidence(100, 'tearing', db)
    assert ev.intervals  # missing inventory cannot erase an actual detector result
    assert 'required corpus group mhr is absent for tearing' in ev.caveats
    db.shots.loc[100, 'has_mhr'] = True
    assert not any('required corpus group' in c for c in ph.evidence(100, 'tearing', db).caveats)


def test_legacy_empty_inventory_and_unknown_flags_do_not_assert_missing_groups(ideate_db):
    db = _db_with(ideate_db, [_event(100, 'seen')])
    assert not any('required corpus group' in c for c in ph.evidence(100, 'tearing', db).caveats)
    db.shots = db.shots.drop(columns=['has_mhr'])
    db.shots.loc[100, 'reader'] = 'corpus'
    assert not any('required corpus group' in c for c in ph.evidence(100, 'tearing', db).caveats)


def test_label_only_displays_the_raw_probability_of_the_weighted_winning_label(ideate_db):
    db = _db_with(ideate_db, [], labels=[_label_row(100, 'tm_prob', max_valid=.8)])
    entry = ph.registry()['tearing']
    entry = replace(entry, labels=(replace(entry.labels[0], weight=.25),))
    hit = ph.locate(entry, db)[0]
    assert hit.score == pytest.approx(.2)
    assert ph.LABEL_ONLY.format(p=.8) in hit.caveats
    assert ph.LABEL_ONLY.format(p=.2) not in hit.caveats


def test_a_narrow_quote_keeps_the_late_phenomenon_mention_and_single_entry(ideate_db, capsys):
    db = _db_with(ideate_db, [], claims=[_claim(100, 'tearing')])
    rec = db.get(100)
    text = 'Repeated density scan. ' * 12 + 'A clear tearing mode locked late in the shot.'
    rec.human.log_entries[0].text = text
    db.shots.loc[100, 'record_json'] = rec.model_dump_json()
    hit = ph.locate('tearing', db)[0]
    assert hit.quote == text.strip()
    excerpt = ph.shorten_quote(hit.quote, 'tearing', 60)
    assert 'tearing mode' in excerpt
    assert excerpt.strip(' .') in text
    assert len(excerpt) <= 60
    cli._phenomenon_table([hit], 'tearing')
    table_line = next(line for line in capsys.readouterr().out.splitlines() if line.lstrip().startswith('100'))
    assert 'tearing mode' in table_line


def test_excerpt_centres_on_a_mention_inside_one_long_sentence():
    text = 'repeat density settings ' * 30 + 'tearing mode locked while ramping down ' * 10
    excerpt = ph.shorten_quote(text, 'tearing', 60)
    assert 'tearing mode' in excerpt
    assert excerpt.strip(' .') in text
    assert len(excerpt) <= 60


@pytest.mark.parametrize('intervals,template', [
    ('[[0, 0.9], [5.1, 6]]', ph.AVOID_UNCOVERED),
    ('[[0, 2], [3, 6]]', ph.AVOID_PARTIAL),
])
def test_avoid_cannot_turn_an_interior_gap_into_a_negative(ideate_db, intervals, template):
    from shot_design.labels import event_sources as es
    from shot_design.shotdb.store import ShotDB

    _db_with(ideate_db, [], claims=[_claim(100, 'tearing')])
    es.write_sources(ideate_db / 'db/event_sources.parquet', [
        es.source_row(100, 'elm_clock', diag='filterscopes', t_cov0_s=0, t_cov1_s=6,
                      intervals=intervals, min_gap_s=.003),
    ])
    (hit,) = ph.locate('tearing', ShotDB.load(ideate_db / 'db'), avoid=['elm'])
    assert hit.shot == 100
    assert template.format(token='phenomenon:elm', title=ph.registry()['elm'].title,
                           segment='flat_top') in hit.caveats
