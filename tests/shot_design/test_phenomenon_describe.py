"""Descriptions keep observations, forecasts and operator quotations apart."""

from shot_design import cli
from shot_design.mcp import tools
from shot_design.retrieval import describe
from shot_design.retrieval import phenomena as ph

from .test_phenomena import _claim, _db_with, _event


def test_a_phenomenon_line_distinguishes_observed_and_forecast_ranges(ideate_db):
    db = _db_with(ideate_db, [
        _event(100, 'seen'),
        _event(100, 'risk', source='label_forecast', evidence_kind='forecast',
               phenomenon='tearing', t0_s=3, t1_s=4),
    ])
    line = describe._phenomenon_line(ph.evidence(100, 'tearing', db), db.get(100))
    assert 'observed 2.000-2.500 s' in line
    assert 'forecast 3.000-4.000 s' in line
    assert 'coverage: observed' in line
    assert '\n' not in line


def test_text_only_line_never_quotes_a_claim_snippet_or_joined_log(ideate_db, monkeypatch):
    db = _db_with(ideate_db, [], claims=[_claim(100, 'tearing', snippet='not a log entry')])
    rec = db.get(100)
    rec.human.log_entries[0].text = 'Repeated density. ' * 15 + 'A tearing mode locked late.'
    calls = []
    real = describe.best_quote

    def best_quote(record, where=None):
        calls.append(record.shot)
        return real(record, where)

    monkeypatch.setattr(describe, 'best_quote', best_quote)
    line = describe._phenomenon_line(ph.evidence(100, 'tearing', db), rec)
    assert 'text-only' in line and 'coverage: unprocessed' in line
    assert 'tearing mode locked' in line
    assert 'not a log entry' not in line
    assert calls == [100]
    quote = line.split('"')[1]
    assert quote.strip(' .') in rec.human.log_entries[0].text


def test_a_forecast_only_description_does_not_claim_an_observation(ideate_db):
    db = _db_with(ideate_db, [_event(
        100, 'risk', source='label_forecast', evidence_kind='forecast', phenomenon='tearing',
    )])
    line = describe._phenomenon_line(ph.evidence(100, 'tearing', db), db.get(100))
    assert 'forecast 2.000-2.500 s' in line and 'coverage: unprocessed' in line
    assert 'observed ' not in line


def test_mcp_and_cli_describe_include_each_evidenced_phenomenon_once(ideate_db, capsys):
    _db_with(ideate_db, [_event(100, 'seen')], claims=[_claim(100, 'rwm')])
    tools.reset_cache()
    response = tools.describe_shot(100)
    prose = response['description']
    assert prose.count('Tearing mode (tearing):') == 1
    assert prose.count('Resistive wall mode (rwm):') == 1
    assert 'no detector registered for rwm; text/database evidence only' in prose
    assert cli.main(['describe', '100']) == 0
    assert capsys.readouterr().out.strip() == prose
    tools.reset_cache()


def test_description_retains_run_scope_and_operator_denial_caveats(ideate_db):
    db = _db_with(ideate_db, [], claims=[
        _claim(100, 'tearing', scope='run'),
        _claim(100, 'tearing', polarity='neg'),
    ])
    line = describe._phenomenon_line(ph.evidence(100, 'tearing', db), db.get(100))
    assert ph.RUN_SCOPE_TEXT in line
    assert ph.NEGATIVE_CLAIM.format(title='Tearing mode') in line


def test_description_reports_when_a_missing_segment_widens_the_search(ideate_db):
    db = _db_with(ideate_db, [_event(100, 'seen')])
    ev = ph.evidence(100, 'tearing', db, segment='ramp_down')
    assert ph.NO_SEGMENT.format(segment='ramp_down') in describe._phenomenon_line(ev, db.get(100))
