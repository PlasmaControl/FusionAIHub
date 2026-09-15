"""The suggested configuration (shot_design.retrieval.suggest): what the successful matches ran, as a
median with a quartile range per actuator key, each value traceable to its shots.

Fixture facts (conftest): shot 900001 has beams 15L 2.0 MW, 30L 1.5 MW, 33L 1.0 MW and gas A;
shot 900002 has no beams and no gas. Both build with verdict "unknown" and no Ip-target miss.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import pytest

from shot_design.retrieval import suggest as sg
from shot_design.shotdb import build, store

from .test_build_store import stub_embeddings  # noqa: F401  (fixture)


@pytest.fixture
def db(paths, staged_shot_a, staged_shot_b, text_fixtures, stub_embeddings):  # noqa: F811
    build.build(
        [staged_shot_a, staged_shot_b], paths, build.load_build_cfg(), workers=1, encode=False
    )
    return store.ShotDB.load(paths.db_dir)


def test_suggest_medians_over_shots_where_the_actuator_was_on(db, staged_shot_a, staged_shot_b):
    ids = [f"{staged_shot_a}:flat_top", f"{staged_shot_b}:flat_top"]
    s = sg.suggest(ids, db, k=10, successful_only=False)
    assert s.segment == "flat_top" and s.n_considered == 2 and s.n_used == 2
    assert s.used_shots == [staged_shot_a, staged_shot_b]
    by = {k.key: k for k in s.keys}
    assert by["nbi.15L"].column == "pnbi_15L_peak" and by["nbi.15L"].units == "W"
    assert by["nbi.15L"].n_on == 1 and by["nbi.15L"].shots == [staged_shot_a]
    assert abs(by["nbi.15L"].median - 2.0e6) < 1.0e3
    assert abs(by["nbi.total"].median - 4.5e6) < 1.0e4  # only 900001 ran beams
    assert "nbi.15R" not in by  # never on in either shot
    assert by["gas.GASA"].n_on == 1
    assert s.candidate.segment == "flat_top"
    assert abs(s.candidate.actuators["nbi.total"] - 4.5e6) < 1.0e4
    assert set(s.candidate.actuators) == set(by)


def test_suggest_drops_unsuccessful_shots_and_says_why(db, staged_shot_a, staged_shot_b):
    db.shots.loc[staged_shot_a, "verdict"] = "bad"
    db.shots.loc[staged_shot_b, "ip_target_hit"] = False
    ids = [f"{staged_shot_a}:flat_top", f"{staged_shot_b}:flat_top"]
    s = sg.suggest(ids, db)
    assert s.n_used == 0 and s.keys == []
    assert s.dropped == {staged_shot_a: "verdict bad", staged_shot_b: "missed Ip target"}
    assert s.candidate.actuators == {}


def test_suggest_respects_k_and_ignores_unknown_ids(db, staged_shot_a, staged_shot_b):
    ids = ["123456:flat_top", f"{staged_shot_a}:flat_top", f"{staged_shot_b}:flat_top"]
    s = sg.suggest(ids, db, k=1, successful_only=False)
    assert s.n_considered == 1 and s.used_shots == [staged_shot_a]


def test_suggest_quartiles_bracket_the_median(db, staged_shot_a, staged_shot_b):
    s = sg.suggest(
        [f"{staged_shot_a}:flat_top", f"{staged_shot_b}:flat_top"], db, successful_only=False
    )
    for k in s.keys:
        assert k.q25 <= k.median <= k.q75


def test_suggest_empty_input(db):
    s = sg.suggest([], db)
    assert s.n_considered == 0 and s.keys == [] and s.segment == "flat_top"


def test_suggest_counts_a_shot_once_across_its_segments(db, staged_shot_a, staged_shot_b):
    """Two segments of one shot are one shot: the k-cut, used_shots and n_on are per shot."""
    ids = [f"{staged_shot_a}:flat_top", f"{staged_shot_a}:full", f"{staged_shot_b}:flat_top"]
    s = sg.suggest(ids, db, k=10, successful_only=False)
    assert s.n_considered == 2 and s.n_used == 2
    assert s.used_shots == [staged_shot_a, staged_shot_b]
    by = {k.key: k for k in s.keys}
    assert by["nbi.15L"].n_on == 1 and by["nbi.15L"].shots == [staged_shot_a]
    assert s.segment == "flat_top"  # the first-ranked segment of 900001, not its "full" row
    assert sg.suggest(ids, db, k=1, successful_only=False).used_shots == [staged_shot_a]
