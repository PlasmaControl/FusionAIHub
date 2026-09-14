"""The documented search spelling and coverage exclusions are usable from the CLI."""

import json

from ideate import cli
from ideate.mcp import tools
from ideate.shotdb import text

from .test_phenomena import _db_with


def test_positional_query_text_matches_the_existing_text_option(ideate_db, monkeypatch, capsys):
    db = _db_with(ideate_db, [])
    monkeypatch.setattr(text, 'embed_texts', lambda texts: db.emb['text_log'][:1])
    assert cli.main(['query', 'edge harmonic oscillation', '--n', '5', '--json']) == 0
    positional = json.loads(capsys.readouterr().out)
    assert cli.main(['query', '--text', 'edge harmonic oscillation', '--n', '5', '--json']) == 0
    assert json.loads(capsys.readouterr().out) == positional


def test_a_filter_with_zero_covered_candidates_is_a_successful_empty_search(ideate_db, capsys):
    assert cli.main(['query', '--ref', '100', '--avoid', 'phenomenon:rwm', '--json']) == 0
    reply = json.loads(capsys.readouterr().out)
    assert reply['results'] == []
    assert any('unprocessed' in c for c in reply['caveats'])
    assert cli.main(['query', '--ref', '100', '--avoid', 'phenomenon:rwm']) == 0
    output = capsys.readouterr()
    assert 'nothing passed the filters' in output.err
    assert 'no channel had anything to search on' not in output.err
    tools.reset_cache()
    reply = tools.search_shots(ref_shot=100, avoid_labels=['phenomenon:rwm'])
    assert not any('no channel had anything to search on' in c for c in reply['caveats'])
    tools.reset_cache()
