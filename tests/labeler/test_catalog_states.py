"""The catalog's four states and what each of its six phenomena may record."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from labeler.events.catalog.states import (
    COMMON_ATTRS,
    PHENOMENA,
    STATE_NAMES,
    attr_problems,
)


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


def test_an_int_too_large_for_a_double_is_a_problem_not_a_crash():
    # Not a whole message comparison: the value is a 401-digit number.
    problems = attr_problems("sawtooth_oscillation", {"period_ms": 10**400})
    assert len(problems) == 1
    assert problems[0].endswith("is not a finite number")


@pytest.mark.parametrize(
    "category, key, value, problem",
    [
        ("neoclassical_tearing_mode", "m", 1, "m=1 is below 2"),
        ("neoclassical_tearing_mode", "m", -3, "m=-3 is below 2"),
        ("neoclassical_tearing_mode", "m", 2, None),
        ("neoclassical_tearing_mode", "n", 0, "n=0 is below 1"),
        ("neoclassical_tearing_mode", "n", 1, None),
        ("edge_localized_mode", "frequency_hz", 0, "frequency_hz=0 must be above 0"),
        (
            "edge_localized_mode",
            "frequency_hz",
            -50,
            "frequency_hz=-50 must be above 0",
        ),
        ("edge_localized_mode", "frequency_hz", 0.01, None),
        ("sawtooth_oscillation", "period_ms", 0, "period_ms=0 must be above 0"),
        ("sawtooth_oscillation", "period_ms", 0.01, None),
        (
            "sawtooth_oscillation",
            "inversion_channel",
            0,
            "inversion_channel=0 is below 1",
        ),
        ("sawtooth_oscillation", "inversion_channel", 1, None),
        ("sawtooth_oscillation", "inversion_channel", 48, None),
        (
            "sawtooth_oscillation",
            "inversion_channel",
            49,
            "inversion_channel=49 is above 48",
        ),
        (
            "sawtooth_oscillation",
            "inversion_radius_m",
            0,
            "inversion_radius_m=0 must be above 0",
        ),
        ("sawtooth_oscillation", "inversion_radius_m", 0.01, None),
    ],
)
def test_physical_attribute_bounds(category, key, value, problem):
    assert attr_problems(category, {key: value}) == (
        [] if problem is None else [problem]
    )


@pytest.mark.parametrize(
    "category, key, kind",
    [
        ("neoclassical_tearing_mode", "m", "an integer"),
        ("neoclassical_tearing_mode", "n", "an integer"),
        ("edge_localized_mode", "frequency_hz", "a finite number"),
        ("sawtooth_oscillation", "period_ms", "a finite number"),
        ("sawtooth_oscillation", "inversion_channel", "an integer"),
        ("sawtooth_oscillation", "inversion_radius_m", "a finite number"),
    ],
)
@pytest.mark.parametrize("value", [True, "2"])
def test_bounded_attributes_still_refuse_bools_and_wrong_types(
    category, key, kind, value
):
    assert attr_problems(category, {key: value}) == [f"{key}={value!r} is not {kind}"]


def test_always_observable_phenomena_are_recorded_in_the_vocabulary():
    assert {key for key, p in PHENOMENA.items() if p.observable_always} == {
        "neoclassical_tearing_mode",
        "disruption",
    }


def test_data_guide_phenomena_table_agrees_with_the_vocabulary():
    guide = (Path(__file__).resolve().parents[2] / "data/events/README.md").read_text()
    match = re.search(r'<table id="catalog-phenomena">.*?</table>', guide, re.DOTALL)
    assert match, "The data guide must list each phenomenon's attributes and points"
    table = ET.fromstring(match.group())
    documented = {}
    for row in table.findall("tbody/tr"):
        category_cell, attrs_cell, points_cell, observable_cell = row.findall("td")
        category = category_cell.findtext("code")
        assert category not in documented, f"Repeated phenomenon: {category}"
        documented[category] = (
            {
                key.text: " ".join(key.tail.strip(":; \n").split())
                for key in attrs_cell.findall("code")
            },
            tuple(kind.text for kind in points_cell.findall("code")),
            "".join(observable_cell.itertext()).strip(),
        )
    assert set(documented) == set(PHENOMENA)
    for category, phenomenon in PHENOMENA.items():
        attrs, points, not_observable = documented[category]
        assert set(attrs) == set(COMMON_ATTRS) | set(phenomenon.attrs), category
        for key, rule in {**COMMON_ATTRS, **phenomenon.attrs}.items():
            description = attrs[key]
            context = f"{category}.{key}: {description}"
            if isinstance(rule, frozenset):
                assert description.startswith("string, "), context
                assert set(description.removeprefix("string, ").split(" / ")) == rule
                continue
            expected = {
                int: "integer",
                float: "number",
                bool: "boolean, true / false",
                str: "nonblank string",
            }[rule]
            if key in phenomenon.lower_bounds:
                value, inclusive = phenomenon.lower_bounds[key]
                expected += f" {'≥' if inclusive else '>'} {value:g}"
            if key in phenomenon.upper_bounds:
                value, inclusive = phenomenon.upper_bounds[key]
                expected += f" and {'≤' if inclusive else '<'} {value:g}"
            if key in phenomenon.units:
                expected += f", {phenomenon.units[key]}"
            assert description == expected, context
        assert points == phenomenon.points, category
        assert not_observable == ("No" if phenomenon.observable_always else "Yes")
