"""Scenarios: search hits grouped by MP, run, then theme; no language model involved.

Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import pytest

from shot_design import config
from shot_design.retrieval import scenarios
from shot_design.shotdb import build, store

from .test_build_store import stub_embeddings  # noqa: F401  (fixture)

THEMES = [
    {"id": "qh_mode", "keywords": ["qh-mode", "qh mode"]},
    {"id": "control", "keywords": ["control", "controller"]},
    {"id": "neg_tri", "keywords": ["negative triangularity"]},
]


def test_theme_counts_match_titles_case_insensitively():
    titles = ["Access and control of QH-mode edge", "QH MODE at low torque", None, "Pedestal"]
    assert scenarios.theme_counts(titles, THEMES) == {"qh_mode": 2, "control": 1, "neg_tri": 0}


def test_theme_label_is_human_readable():
    assert scenarios.theme_label("detachment_divertor") == "detachment / divertor"
    assert scenarios.theme_label("qh_mode") == "QH mode"


def test_group_key_prefers_mpid_then_run_then_theme():
    assert scenarios.group_key("2014-21-20", "20150120", ["qh_mode"]) == ("mp", "2014-21-20")
    assert scenarios.group_key(None, "20150120", ["qh_mode"]) == ("run", "20150120")
    assert scenarios.group_key(None, None, ["qh_mode", "control"]) == ("theme", "qh_mode")
    assert scenarios.group_key(None, None, []) == ("theme", "other")


def _scenario(*, n_hits: int, n_shots: int, best_rank: int) -> scenarios.Scenario:
    return scenarios.Scenario(
        kind="mp",
        key="dummy",
        goal="dummy goal",
        shots=list(range(n_shots)),
        representative_shot=0,
        n_shots=n_shots,
        n_hits=n_hits,
        n_success=0,
        best_rank=best_rank,
    )


def test_order_ranks_by_hits_then_shots_then_best_rank():
    few_hits_big_group = _scenario(n_hits=1, n_shots=6, best_rank=40)
    many_hits_small_group = _scenario(n_hits=4, n_shots=4, best_rank=2)
    many_hits_bigger_group = _scenario(n_hits=4, n_shots=5, best_rank=7)
    ordered = scenarios._order([few_hits_big_group, many_hits_small_group, many_hits_bigger_group])
    assert ordered == [many_hits_bigger_group, many_hits_small_group, few_hits_big_group]


@pytest.fixture
def db(paths, staged_shot_a, staged_shot_b, text_fixtures, stub_embeddings):  # noqa: F811
    build.build(
        [staged_shot_a, staged_shot_b], paths, build.load_build_cfg(), workers=1, encode=False
    )
    return store.ShotDB.load(paths.db_dir)


def test_propose_groups_the_two_fixture_shots_into_one_mp_scenario(
    db, staged_shot_a, staged_shot_b
):
    out = scenarios.propose_scenarios("QH-mode at low torque", db)
    assert out.query_text == "QH-mode at low torque"
    assert out.n_hits == 2
    assert len(out.scenarios) == 1
    sc = out.scenarios[0]
    assert sc.kind == "mp" and sc.key == "2014-21-20"
    assert sorted(sc.shots) == sorted([staged_shot_a, staged_shot_b])
    assert sc.representative_shot in sc.shots
    assert sc.n_shots == 2
    assert sc.n_hits == 2
    assert 0 <= sc.n_success <= 2
    assert sc.goal  # MP title / run title, never empty for these fixtures
    assert sc.quote is not None  # a quote is one entry's text, verbatim
    rec = db.get(sc.quote_shot)
    assert any(sc.quote in (e.text.replace("\n", " ")) for e in rec.human.log_entries)


def test_a_scenario_counts_the_whole_mp_not_only_the_surfaced_hits(
    db, staged_shot_a, staged_shot_b
):
    """The two fixture shots share mpid 2014-21-20; a one-hit search must still report both."""
    out = scenarios.propose_scenarios("QH-mode at low torque", db, n_hits=1)
    assert out.n_hits == 1  # one hit was searched for and one shot was grouped
    assert len(out.scenarios) == 1
    sc = out.scenarios[0]
    assert sc.kind == "mp" and sc.key == "2014-21-20"
    assert sc.n_hits == 1
    assert sc.n_shots == 2
    assert sorted(sc.shots) == sorted([staged_shot_a, staged_shot_b])
    assert sc.shots[0] == sc.representative_shot  # the hit leads, the rest of the MP follows
    assert sum(sc.verdicts.values()) == 2  # verdicts cover the whole MP too


def test_propose_with_a_theme_and_no_text_uses_the_theme_keywords(db):
    out = scenarios.propose_scenarios(None, db, theme="qh_mode")
    assert out.query_text.startswith("qh-mode")
    assert out.theme == "qh_mode"


def test_propose_rejects_an_unknown_theme(db):
    with pytest.raises(ValueError, match="unknown theme"):
        scenarios.propose_scenarios(None, db, theme="not_a_theme")


def test_propose_with_nothing_to_search_returns_empty(db):
    out = scenarios.propose_scenarios(None, db)
    assert out.scenarios == [] and out.n_hits == 0


def test_ui_yaml_has_the_landing_block():
    cfg = config.load_yaml("ui.yaml")
    assert cfg["landing"] == {"n_scenarios": 6, "n_hits": 50, "prompt": "What do you want to test?"}
