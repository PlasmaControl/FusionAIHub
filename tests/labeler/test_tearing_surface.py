"""Which poloidal number the EFIT safety factor can give a labelled mode."""

from __future__ import annotations

import numpy as np

from labeler.events.catalog.states import attr_problems
from labeler.events.interval_tables import parse_attrs
from labeler.tearing import rule, surface

RHO = np.linspace(0.0, 1.0, 33)


def profile(q0, q95):
    """A monotonic q from `q0` on axis to `q95` at rho 0.95 (and above)."""
    return q0 + (q95 - q0) * np.clip(RHO / 0.95, 0, 1) ** 2


def test_candidates_are_the_rational_surfaces_above_q_of_one_inside_the_profile():
    # q from 1.2 to 2.8: 2/1 (q = 2) is the only surface for n = 1
    assert surface.candidate_m(1, profile(1.2, 2.8), RHO) == [2]
    # n = 2 sees q = 1.5, 2, 2.5 (m = 3, 4, 5)
    assert surface.candidate_m(2, profile(1.2, 2.8), RHO) == [3, 4, 5]
    # a wider profile has more
    assert surface.candidate_m(1, profile(1.1, 5.2), RHO) == [2, 3, 4, 5]
    # q <= 1 gives no tearing surface; nothing finite gives none
    assert surface.candidate_m(1, profile(0.9, 1.0), RHO) == []
    assert surface.candidate_m(1, np.full(33, np.nan), RHO) == []


def test_the_edge_where_q_diverges_is_not_read():
    q = profile(1.2, 2.8)
    q[-2:] = 40.0  # beyond rho 0.95: near the separatrix
    assert surface.candidate_m(1, q, RHO) == [2]


def test_q_alone_cannot_identify_m_even_with_only_one_candidate():
    t = np.arange(0.0, 3000.0, 25.0)
    q = np.stack([profile(1.2, 2.8)] * len(t), axis=1)
    assert surface.supported_m(1, t, q, RHO, 500.0, 1500.0) is None
    assert surface.supported_m(2, t, q, RHO, 500.0, 1500.0) is None
    wide = np.stack([profile(1.1, 5.2)] * len(t), axis=1)
    assert surface.supported_m(1, t, wide, RHO, 500.0, 1500.0) is None
    # an interval with no q sample in it has no m
    assert surface.supported_m(1, t, q, RHO, 10_000.0, 11_000.0) is None


def test_m_uses_q_at_the_island_radius_rather_than_all_candidate_surfaces():
    t = np.array([0.0, 25.0, 50.0])
    rho = np.array([0.0, 0.5, 0.9, 1.0])
    q = np.tile(np.array([1.2, 1.5, 3.0, 10.0])[:, None], (1, 3))
    assert surface.supported_m(2, t, q, rho, 0, 50, island_rho=0.5) == 3
    assert surface.supported_m(1, t, q, rho, 0, 50, island_rho=0.5) is None
    assert surface.supported_m(2, t, q, rho, 0, 50, island_rho=0.98) is None
    assert surface.supported_m(2, t, q, rho, 100, 200, island_rho=0.5) is None


def test_m_is_empty_without_finite_q_at_an_observed_island_radius():
    t = np.array([0.0, 25.0])
    rho = np.array([0.0, 0.5, 0.9])
    q = np.full((3, 2), np.nan)
    assert surface.supported_m(2, t, q, rho, 0, 25, island_rho=0.5) is None


def test_an_m_hook_returning_none_leaves_both_onset_and_span_unattributed():
    t = np.arange(3000.0)
    n1 = np.full(t.shape, 0.2)
    n1[1000:1500] = 30.0
    label = rule.label_shot(
        1, t, n1, None, (5.0, 2900.0),
        coherent={1: np.ones(t.shape, bool)},
        m_of=lambda n, start, end: None,
    )
    present = rule.shot_table(label).query("category == 1")
    assert len(present) == 2
    attrs = [parse_attrs(a) for a in present["attrs"].tolist()]
    assert {a["iscrowd"] for a in attrs} == {0, 1}
    assert all("m" not in a and "efit_tree" not in a for a in attrs)


def test_a_label_carries_m_and_its_efit_tree_when_the_hook_gives_one():
    t = np.arange(0.0, 3000.0, 1.0)
    n1 = np.full(t.shape, 0.2)
    n1[1000:1500] = 30.0
    seen = []

    def m_of(n, start, end):
        seen.append((n, round(start), round(end)))
        return 2

    label = rule.label_shot(
        1, t, n1, None, (5.0, 2900.0), start_ms=100.0,
        coherent={1: np.ones(t.shape, bool)}, m_of=m_of,
    )
    assert seen and seen[0][0] == 1
    (item,) = label.intervals
    assert item.m == 2
    table = rule.shot_table(label)
    span = table[(table.category == 1) & (table.t_end > table.t_start)]
    attrs = parse_attrs(span["attrs"].iloc[0])
    assert attrs["m"] == 2 and attrs["efit_tree"] == "efit01" and attrs["n"] == 1
    assert not attr_problems(rule.CATEGORY, attrs)
    assert rule.intervals_frame([label]).m.tolist() == [2]
    # no hook, no m
    bare = rule.label_shot(
        1, t, n1, None, (5.0, 2900.0), start_ms=100.0,
        coherent={1: np.ones(t.shape, bool)},
    )
    assert bare.intervals[0].m is None
    assert "m" not in parse_attrs(
        rule.shot_table(bare)
        .query("category == 1 and t_end > t_start")["attrs"]
        .iloc[0]
    )
