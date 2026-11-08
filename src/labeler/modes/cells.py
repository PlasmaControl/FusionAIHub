"""The events' whole-ms states for one shot: `mhd_n<K>` and its `m<m>` subcategories.

The states, the span padding (+-2 ms around each 4 ms frame), the merge across gaps of up to
20 ms, the 20 ms minimum and every not-observable rule are those of `labeler.mhd.rows` for the
v5 suggestion tables (0 absent, 1 present, 2 uncertain, 3 not observable); this module only
feeds them the tracks the owner's rules leave (`labeler.modes.tracks`) and takes the
subcategories from the classes the labelled modes carry instead of the five v5 fixed. Two views:

- the suggestion view (`strict` false): the v5 semantics. `mhd_n<K>` is present over every
  non-line track of that n, labelled or not (an n-only abstention still shows an n mode); a
  subcategory is present over its labelled tracks and uncertain over the abstentions whose
  90% credible set holds its class.
- the label view (`strict` true): only labelled modes. `mhd_n<K>` is present over the
  labelled tracks of that n alone, and a subcategory only over its own labelled tracks.

The crash table and the q = 1 gate of v5 (`rows.shot_cells(crash=, no_q1=)`) are FusionAIHub
inputs, not omnimode's, and are not applied: a sawtooth shows only as its labelled 1/1 tracks.
"""

from __future__ import annotations

from labeler.mhd import pipeline, rows

from . import tracks as T

PARENTS = rows.PARENTS
PRESENT, UNCERTAIN, ABSENT, UNSEEN = (
    rows.PRESENT,
    rows.UNCERTAIN,
    rows.ABSENT,
    rows.UNSEEN,
)


def children(names) -> dict[str, str]:
    """`{event: class}` for class names such as `"2/1"`, by n then m: `{"mhd_n1.m2": "2/1"}`."""
    ordered = sorted(set(names), key=lambda name: T.parse_class(name)[::-1])
    return {T.event_of(name): name for name in ordered}


def parent_of(event: str) -> str:
    """`mhd_n1` for `mhd_n1.m2` (and for `mhd_n1` itself)."""
    return event.split(".")[0]


def shot_cells(
    got: pipeline.Shot, kids: dict[str, str], *, strict: bool = False
) -> dict[str, rows.Cells]:
    """Every event's cells for the shot: the three parents and the subcategories `kids`.

    `got` is a shot read by `tracks.read_shot` and passed through `tracks.apply_rules` (the
    caller owns that step so it can count what was dropped). `strict`: the label view, see
    the module docstring."""
    if strict:
        got = T.labelled(got)
    out = {event: rows.parent_cells(got, n) for event, n in PARENTS.items()}
    for event, name in kids.items():
        out[event] = rows.child_cells(got, out[parent_of(event)], name)
    return out


def present_runs(cells: rows.Cells) -> list[tuple[int, int, float]]:
    """`(start_ms, stop_ms, confidence)` of each run of present cells, half-open on the whole-ms
    grid. The confidence is the largest the run holds, NaN when it holds none."""
    import numpy as np

    on = np.r_[False, cells.state == PRESENT, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(on))
    out = []
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        got = cells.conf[a:b]
        finite = got[np.isfinite(got)]
        out.append(
            (
                cells.lo + int(a),
                cells.lo + int(b),
                float(finite.max()) if len(finite) else float("nan"),
            )
        )
    return out
