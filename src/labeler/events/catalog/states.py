"""The catalog's four states, and what each of its six phenomena records.

In a catalog table the `category` column is the state. Time a table has no row
for was not assessed: a fifth answer, never the same as absent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..channels import N_ECE_CHANNELS

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
    `lower_bounds` and `upper_bounds` map numeric attributes to `(bound, inclusive)`.
    `units` gives the physical unit where one applies.
    """

    category: str
    name: str
    attrs: Mapping[str, type | frozenset] = field(default_factory=dict)
    points: tuple[str, ...] = ()
    lower_bounds: Mapping[str, tuple[float, bool]] = field(default_factory=dict)
    observable_always: bool = False
    upper_bounds: Mapping[str, tuple[float, bool]] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)


def _words(*words: str) -> frozenset:
    return frozenset(words)


#: Any span may say why a stretch was not observable, in the reader's words.
COMMON_ATTRS = {"reason": str, "iscrowd": int}

PHENOMENA = {
    p.category: p
    for p in (
        Phenomenon("alfven_eigenmode", "AE", {"type": _words("RSAE", "TAE", "other")}),
        Phenomenon(
            "neoclassical_tearing_mode",
            "TM",
            {
                "m": int,
                "n": int,
                "efit_tree": _words("efit01", "efit02"),
                "seed": _words("sawtooth", "elm", "fishbone", "none"),
                "confinement": _words("L", "H"),
                "locked": bool,
                "locked_candidate": bool,
                "locked_known": bool,
                "lock_time_ms": float,
                "lock_candidates_ms": list,
                "onset_window_ms": list,
                "ended": _words("decay", "plasma_end", "locked", "unknown"),
                "override": _words(
                    "island_not_resolved",
                    "q_unreliable",
                    "classical_tm",
                    "not_tearing_mode",
                ),
                "other_mhd": _words("m1", "classical_tm", "fishbone", "eho", "kink"),
            },
            lower_bounds={"m": (2, True), "n": (1, True)},
            observable_always=True,  # Magnetics are an inclusion rule.
        ),
        Phenomenon(
            "high_confinement_mode",
            "H-mode",
            {"variant": _words("standard", "QH", "other")},
        ),
        Phenomenon(
            "edge_localized_mode",
            "ELMing",
            {"frequency_hz": float},
            points=("elm",),
            lower_bounds={"frequency_hz": (0, False)},
            units={"frequency_hz": "Hz"},
        ),
        Phenomenon(
            "sawtooth_oscillation",
            "sawtooth",
            {"period_ms": float, "inversion_channel": int, "inversion_radius_m": float},
            points=("crash",),
            lower_bounds={
                "period_ms": (0, False),
                # TECEF channel 1-48 = corpus ECE row + 1. Panel titles and
                # heuristic inversion_channel_lo/_stop use zero-based rows.
                # The owner must define which side of the inversion before use.
                "inversion_channel": (1, True),
                "inversion_radius_m": (0, False),
            },
            upper_bounds={"inversion_channel": (N_ECE_CHANNELS, True)},
            units={"period_ms": "ms", "inversion_radius_m": "m"},
        ),
        Phenomenon(
            "disruption",
            "disruption",
            {"intentional": bool, "phase": _words("flattop", "rampdown")},
            points=("t_D", "t80", "t20"),
            observable_always=True,  # Ip always exists.
        ),
    )
}


def attr_problems(category: str, attrs: Mapping) -> list[str]:
    """What is wrong with one row's `attrs` for `category`; empty if nothing."""
    phenomenon = PHENOMENA[category]
    allowed = {**COMMON_ATTRS, **phenomenon.attrs}
    problems = []
    for key, value in attrs.items():
        if key == "iscrowd":
            if not isinstance(value, int) or value not in (0, 1):
                problems.append(
                    f"iscrowd={value!r} must be 0 (individual) or 1 (group)"
                )
            continue
        rule = allowed.get(key)
        if rule is None:
            problems.append(f"{key!r} is not an attribute of {category}")
        elif not _fits(value, rule):
            problems.append(f"{key}={value!r} is not {_describe(rule)}")
        else:
            if key in phenomenon.lower_bounds:
                minimum, inclusive = phenomenon.lower_bounds[key]
                if inclusive and value < minimum:
                    problems.append(f"{key}={value!r} is below {minimum:g}")
                elif not inclusive and value <= minimum:
                    problems.append(f"{key}={value!r} must be above {minimum:g}")
            if key in phenomenon.upper_bounds:
                maximum, inclusive = phenomenon.upper_bounds[key]
                if inclusive and value > maximum:
                    problems.append(f"{key}={value!r} is above {maximum:g}")
                elif not inclusive and value >= maximum:
                    problems.append(f"{key}={value!r} must be below {maximum:g}")
    return problems


def _fits(value, rule) -> bool:
    if isinstance(rule, frozenset):
        return isinstance(value, str) and value in rule
    if isinstance(value, bool) or rule is bool:
        return isinstance(value, bool) and rule is bool
    if rule is float:
        try:
            return isinstance(value, int | float) and math.isfinite(value)
        except OverflowError:  # an int too large for a double
            return False
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
