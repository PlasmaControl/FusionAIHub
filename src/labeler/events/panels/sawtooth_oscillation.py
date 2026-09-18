"""Sawteeth, as the inversion of adjacent ECE channels across the q = 1 surface."""

from __future__ import annotations

from ..raw import raw_signal
from ..verify import Panel

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

GUIDANCE = (
    "<b>What you are looking for:</b> a sawtooth ramp on the inner channels "
    "that collapses abruptly while the outer channels jump up at the same "
    "instant - the inversion across the q = 1 surface. A rise or fall that "
    "moves every channel the same way is not a sawtooth."
    "<br><br>A shot outside the corpus is fetched live, 48 ECE channels over "
    "MDSplus, which is slow the first time and cached afterwards."
)


def panels(shot, *, t_range=None, paths=None):
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
