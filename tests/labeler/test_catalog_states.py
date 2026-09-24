"""The catalog's four states and what each of its six phenomena may record."""

from __future__ import annotations

import pytest

from labeler.events.catalog.states import PHENOMENA, STATE_NAMES, attr_problems


def test_four_states_and_six_phenomena():
    assert STATE_NAMES == {
        0: "absent",
        1: "present",
        2: "uncertain",
        3: "not_observable",
    }
    assert sorted(PHENOMENA) == [
        "alfven_eigenmode",
        "disruption",
        "edge_localized_mode",
        "high_confinement_mode",
        "neoclassical_tearing_mode",
        "sawtooth_oscillation",
    ]
    assert {c: p.points for c, p in PHENOMENA.items() if p.points} == {
        "edge_localized_mode": ("elm",),
        "sawtooth_oscillation": ("crash",),
        "disruption": ("t_D", "t80", "t20"),
    }


@pytest.mark.parametrize(
    "category, attrs",
    [
        (
            "neoclassical_tearing_mode",
            {"m": 2, "n": 1, "efit_tree": "efit01", "seed": "none", "locked": False},
        ),
        ("neoclassical_tearing_mode", {"override": "q_unreliable"}),
        ("sawtooth_oscillation", {"period_ms": 12, "inversion_radius_m": 0.21}),
        ("disruption", {"intentional": True, "phase": "rampdown"}),
        ("alfven_eigenmode", {"type": "RSAE", "reason": "co2 chords saturated"}),
        ("high_confinement_mode", {}),
    ],
)
def test_attributes_that_fit(category, attrs):
    assert attr_problems(category, attrs) == []


@pytest.mark.parametrize(
    "category, attrs, problem",
    [
        ("alfven_eigenmode", {"m": 2}, "'m' is not an attribute of alfven_eigenmode"),
        (
            "alfven_eigenmode",
            {"type": "BAE"},
            "type='BAE' is not one of ['RSAE', 'TAE', 'other']",
        ),
        ("neoclassical_tearing_mode", {"m": 2.0}, "m=2.0 is not an integer"),
        ("neoclassical_tearing_mode", {"m": True}, "m=True is not an integer"),
        ("neoclassical_tearing_mode", {"locked": 1}, "locked=1 is not true or false"),
        (
            "sawtooth_oscillation",
            {"period_ms": float("nan")},
            "period_ms=nan is not a finite number",
        ),
        (
            "sawtooth_oscillation",
            {"period_ms": False},
            "period_ms=False is not a finite number",
        ),
        ("disruption", {"reason": "  "}, "reason='  ' is not text"),
    ],
)
def test_attributes_that_do_not(category, attrs, problem):
    assert attr_problems(category, attrs) == [problem]
