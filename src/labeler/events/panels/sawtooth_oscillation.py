"""Sawteeth, as the inversion of adjacent ECE channels across the q = 1 surface,
and the soft X-ray core chords that drop at each crash.

A shot without ECE, or without SXR, gets the other's rows alone.
"""

from __future__ import annotations

import numpy as np

from ..raw import raw_signal
from ..verify import NoDataError, Panel
from ._shared import optional

#: Four rows of four ADJACENT channels covering 20-35, sixteen in all. The
#: flip a sawtooth crash makes is a RELATIVE thing - inner channels drop as
#: outer ones rise - so channels are overplotted in adjacent groups rather
#: than drawn one per panel.
CHANNEL_ROWS = (
    (20, 21, 22, 23),
    (24, 25, 26, 27),
    (28, 29, 30, 31),
    (32, 33, 34, 35),
)
#: The SXR fans tried in order, by their first row in the corpus's 320 (32
#: chords each). The first with `MIN_CHORDS` chords finite over at least half
#: the record is drawn: its `BRIGHTEST` brightest, the chords through the core.
SXR_ARRAYS = (
    ("SX90RM1F", 192),
    ("SX90RP1F", 256),
    ("SX90RM1S", 224),
    ("SX90RP1S", 288),
)
CHORDS = 32
MIN_CHORDS = 8
BRIGHTEST = 4


def ece_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    built = []
    for row in CHANNEL_ROWS:
        array = raw_signal(
            int(shot), "ece", channels=list(row), t_range=t_range, paths=paths
        )
        built.append(
            Panel(
                title=f"ECE ch {row[0]}-{row[-1]}",
                x=array.x,
                y=array.y,
                ylabel="keV",
                legend=[f"ch {c}" for c in row],
            )
        )
    return built


def sxr_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    for name, first in SXR_ARRAYS:
        rows = list(range(first, first + CHORDS))
        array = raw_signal(
            int(shot), "sxr", channels=rows, t_range=t_range, paths=paths
        )
        lit = np.isfinite(array.y).mean(axis=1) >= 0.5
        if lit.sum() < MIN_CHORDS:
            continue
        level = np.full(CHORDS, -np.inf)
        level[lit] = np.nanmedian(array.y[lit], axis=1)
        top = np.sort(np.argsort(-level, kind="stable")[:BRIGHTEST])
        return [
            Panel(
                title=f"SXR {name}, the {BRIGHTEST} brightest chords",
                x=array.x,
                y=array.y[top],
                legend=[f"{name}{c + 1:02d}" for c in top],
            )
        ]
    raise NoDataError(f"shot {int(shot)}: no SXR fan has {MIN_CHORDS} finite chords")


def panels(shot, *, t_range=None, paths=None):
    kwargs = {"t_range": t_range, "paths": paths}
    return optional("ECE", shot, lambda: ece_panels(shot, **kwargs)) + optional(
        "SXR", shot, lambda: sxr_panels(shot, **kwargs)
    )
