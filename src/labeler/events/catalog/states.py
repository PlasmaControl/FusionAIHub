"""The catalog's four states, and what each of its six phenomena records.

In a catalog table the `category` column is the state. Time a table has no row
for was not assessed: a fifth answer, never the same as absent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

ABSENT, PRESENT, UNCERTAIN, NOT_OBSERVABLE = 0, 1, 2, 3
STATE_NAMES = {
    ABSENT: "absent",
    PRESENT: "present",
    UNCERTAIN: "uncertain",
    NOT_OBSERVABLE: "not_observable",
}


@dataclass(frozen=True)
class Phenomenon:
    """A catalog phenomenon: its category directory, attributes and point kinds.

    An attribute allows a type (`int`, `float`, `bool`, `str`) or a set of words.
    """

    category: str
    name: str
    attrs: Mapping[str, type | frozenset] = field(default_factory=dict)
    points: tuple[str, ...] = ()


def _words(*words: str) -> frozenset:
    return frozenset(words)


#: Any span may say why a stretch was not observable, in the reader's words.
COMMON_ATTRS = {"reason": str}

PHENOMENA = {
    p.category: p
    for p in (
        Phenomenon("alfven_eigenmode", "AE", {"type": _words("RSAE", "TAE", "other")}),
        Phenomenon(
            "neoclassical_tearing_mode",
            "NTM",
            {
                "m": int,
                "n": int,
                "efit_tree": _words("efit01", "efit02"),
                "seed": _words("sawtooth", "elm", "fishbone", "none"),
                "confinement": _words("L", "H"),
                "locked": bool,
                "override": _words(
                    "island_not_resolved",
                    "q_unreliable",
                    "classical_tm",
                    "not_tearing_mode",
                ),
                "other_mhd": _words("m1", "classical_tm", "fishbone", "eho", "kink"),
            },
        ),
        Phenomenon(
            "high_confinement_mode",
            "H-mode",
            {"variant": _words("standard", "QH", "other")},
        ),
        Phenomenon(
            "edge_localized_mode", "ELMing", {"frequency_hz": float}, points=("elm",)
        ),
        Phenomenon(
            "sawtooth_oscillation",
            "sawteeth",
            {"period_ms": float, "inversion_channel": int, "inversion_radius_m": float},
            points=("crash",),
        ),
        Phenomenon(
            "disruption",
            "disruption",
            {"intentional": bool, "phase": _words("flattop", "rampdown")},
            points=("t_D", "t80", "t20"),
        ),
    )
}


def attr_problems(category: str, attrs: Mapping) -> list[str]:
    """What is wrong with one row's `attrs` for `category`; empty if nothing."""
    allowed = {**COMMON_ATTRS, **PHENOMENA[category].attrs}
    problems = []
    for key, value in attrs.items():
        rule = allowed.get(key)
        if rule is None:
            problems.append(f"{key!r} is not an attribute of {category}")
        elif not _fits(value, rule):
            problems.append(f"{key}={value!r} is not {_describe(rule)}")
    return problems


def _fits(value, rule) -> bool:
    if isinstance(rule, frozenset):
        return isinstance(value, str) and value in rule
    if isinstance(value, bool) or rule is bool:
        return isinstance(value, bool) and rule is bool
    if rule is float:
        return isinstance(value, int | float) and math.isfinite(value)
    if rule is str:
        return isinstance(value, str) and bool(value.strip())
    return isinstance(value, rule)


_KINDS = {
    int: "an integer",
    float: "a finite number",
    bool: "true or false",
    str: "text",
}


def _describe(rule) -> str:
    if isinstance(rule, frozenset):
        return f"one of {sorted(rule)}"
    return _KINDS[rule]
