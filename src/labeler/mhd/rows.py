"""The eight MHD events' suggestion rows from a shot's labelled tracks (decisions M1-M6,
EQ2 and round 5's P2).

Cells are whole ms over the shot's window, states 0 absent, 1 present, 2 uncertain, 3 not
observable. A track covers +-`PAD_MS` around each of its frames; the union of the tracks
that count is merged across gaps of up to `MERGE_MS` and spans shorter than `MIN_MS` are
dropped (`spans`).

- Parent `mhd_nK`: present over every track with that n except fixed-frequency lines
  (harmonics, non-resonant and out-of-scope modes included); confidence the largest track
  coherence (median n-fit score) in the span. Time outside the Mirnov record's finite
  stretch, and its NaN gaps, are not observable; a shot whose Mirnov cannot be read (no
  record, no `mirnov` group or no finite sample in the window, fewer than four live
  toroidal probes) is not observable for all eight events. `mhd_n1` is also present over
  the crash table's present spans (a q-gated ECE sawtooth crash is 1/1 internal-kink
  evidence, so n = 1), whatever the Mirnov shows there; no confidence on that time. The
  union takes the same span rule (`merged_union`): absent gaps of up to `MERGE_MS` between
  its present runs are present (no confidence), and a run shorter than `MIN_MS` with its
  merges is dropped unless it holds track time (the tracks' spans passed the rule already).
- Child m/n: inside the parent's present time only. Not observable when the shot has no
  equilibrium tree at all, where no slice lies within 30 ms, and over tracks the model
  could not run on; uncertain over tracks whose status is uncertain, no rotation or q_min
  above and whose 90% credible set holds the class (confidence its posterior); present over
  tracks with status m/n and that label (confidence P(resonant)). On an EFIT01 shot
  (magnetics-only: its q_min reads high) a q_min-above track is uncertain for its candidate
  class too, never absent. Outside the parent's present time a child takes the parent's
  state (absent or not observable).
- `mhd_n1.m1` is the resonant 1/1 tracks united with the crash table's present spans inside
  the parent's present time (only a crash run the union's span rule dropped is left out);
  a resonant 1/1 track's cells keep their confidence where a crash covers them too. Time
  with no q = 1 surface (`labeler.sawtooth.qgate`) is absent, except where m1 is not
  observable.
- The crash is 1/1 evidence only: 2/1 and 3/1 are set against the n = 1 tracks' own parent
  cells, so the crash table leaves them as they are (absent where the Mirnov saw no n = 1
  track, not observable where it saw nothing).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from .equilibrium import TRUSTED
from .pipeline import UNSURE, Shot, eq_cells, eq_times, tree

PAD_MS, MERGE_MS, MIN_MS = 2.0, 20.0, 20.0
PARENTS = {"mhd_n1": 1, "mhd_n2": 2, "mhd_n3": 3}
CHILDREN = {
    "mhd_n1.m1": "1/1",
    "mhd_n1.m2": "2/1",
    "mhd_n1.m3": "3/1",
    "mhd_n2.m3": "3/2",
    "mhd_n3.m4": "4/3",
}
EVENTS = (*PARENTS, *CHILDREN)
ABSENT, PRESENT, UNCERTAIN, UNSEEN = 0, 1, 2, 3


def spans(t_ms, value, *, pad=PAD_MS, merge=MERGE_MS, min_ms=MIN_MS):
    """`(start, stop, max value)` of the frames `t_ms` padded by `pad`, merged across gaps of
    up to `merge`, kept when at least `min_ms` long."""
    t = np.asarray(t_ms, float)
    v = np.asarray(value, float)
    if not len(t):
        return []
    o = np.argsort(t, kind="stable")
    out = []
    for a, b, x in zip(t[o] - pad, t[o] + pad, v[o]):
        if out and a - out[-1][1] <= merge:
            out[-1][1] = max(out[-1][1], b)
            out[-1][2] = np.fmax(out[-1][2], x)
        else:
            out.append([a, b, x])
    return [(a, b, x) for a, b, x in out if b - a >= min_ms]


class Cells:
    """Whole-ms cells over [lo, hi) with a state and a confidence."""

    def __init__(self, window, state=ABSENT):
        self.lo, self.hi = int(np.floor(window[0])), int(np.ceil(window[1]))
        n = max(0, self.hi - self.lo)
        self.state = np.full(n, state, np.int64)
        self.conf = np.full(n, np.nan)

    def index(self, a, b):
        start = max(int(np.floor(a)), self.lo) - self.lo
        stop = min(int(np.ceil(b)), self.hi) - self.lo
        return max(start, 0), max(stop, 0)

    def paint(self, a, b, state, conf=np.nan, where=None):
        i, j = self.index(a, b)
        if j <= i:
            return
        sel = np.zeros(len(self.state), bool)
        sel[i:j] = True
        if where is not None:
            sel &= where
        self.state[sel] = state
        self.conf[sel] = conf

    def mask(self, a, b):
        sel = np.zeros(len(self.state), bool)
        i, j = self.index(a, b)
        sel[i:j] = True
        return sel

    def rows(self, shot) -> list[list]:
        if not len(self.state):
            return []
        conf = np.where(
            np.isin(self.state, (PRESENT, UNCERTAIN)), np.round(self.conf, 3), -1
        )
        conf = np.nan_to_num(conf, nan=-1)
        key = self.state * 10_000 + np.round(conf * 1000).astype(np.int64)
        edges = np.flatnonzero(np.diff(key)) + 1
        starts, stops = np.r_[0, edges], np.r_[edges, len(key)]
        return [
            [
                int(shot),
                int(self.state[a]),
                self.lo + int(a),
                self.lo + int(b),
                "" if conf[a] < 0 else float(conf[a]),
            ]
            for a, b in zip(starts, stops)
        ]

    def seconds(self) -> dict[str, float]:
        names = ("absent", "present", "uncertain", "not_observable")
        return {
            names[s]: round(float((self.state == s).sum()) / 1000.0, 3)
            for s in range(4)
        }


def _frames_of(got: Shot, ids):
    fr = got.frames
    return fr[fr.family_id.isin(list(ids))]


def _track_spans(got: Shot, sel, value_col: str | None = None, values=None):
    """Spans over the frames of the tracks `sel` (a boolean mask over got.tracks)."""
    T = got.tracks[sel]
    if not len(T):
        return []
    if values is None:
        values = T[value_col].astype(float).values
    val = dict(zip(T.family_id, values))
    fr = _frames_of(got, T.family_id)
    return spans(fr.t_ms.values, fr.family_id.map(val).values)


def parent_cells(got: Shot, n: int, *, crash=()) -> Cells:
    """The parent n's cells; `crash` (n = 1): present spans united with the tracks'."""
    c = Cells(got.info["window"], UNSEEN)
    if got.mirnov:
        cover = got.info.get("mirnov_ms") or got.info["window"]
        c.paint(cover[0], cover[1], ABSENT)
        T = got.tracks
        if len(T):
            sel = (T.n.astype(int) == n) & ~T.line.astype(bool)
            for a, b, x in _track_spans(got, sel.values, "coh"):
                c.paint(a, b, PRESENT, x)
        for a, b in got.info.get("mirnov_gaps", ()):  # also where a span merges across
            c.paint(a, b, UNSEEN)
    if len(crash):
        tracks = c.state == PRESENT
        for a, b in crash:  # seen on ECE, so present over Mirnov gaps too
            c.paint(a, b, PRESENT, np.nan, where=c.state != PRESENT)
        merged_union(c, keep=tracks, base=_seen(got).state)
    return c


def _seen(got: Shot) -> Cells:
    """A parent's cells before any track: absent over the Mirnov record's finite stretch,
    not observable outside it, over its NaN gaps and on a shot without a readable record."""
    c = Cells(got.info["window"], UNSEEN)
    if got.mirnov:
        cover = got.info.get("mirnov_ms") or got.info["window"]
        c.paint(cover[0], cover[1], ABSENT)
        for a, b in got.info.get("mirnov_gaps", ()):
            c.paint(a, b, UNSEEN)
    return c


def _present_runs(state) -> list[tuple[int, int]]:
    """`(start, stop)` cell indices of the present runs."""
    on = np.r_[False, np.asarray(state) == PRESENT, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(on))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def merged_union(c: Cells, *, keep, base) -> None:
    """The span rule (`spans`) on a union of present spans, in place: present runs at most
    `MERGE_MS` apart are one span, whose absent cells between them become present (no
    confidence; not observable cells stay so, as a Mirnov gap inside a track's span does);
    a span shorter than `MIN_MS` goes back to `base` unless it holds a cell of `keep`."""
    runs = _present_runs(c.state)
    groups: list[list[tuple[int, int]]] = []
    for a, b in runs:
        if groups and a - groups[-1][-1][1] <= MERGE_MS:
            groups[-1].append((a, b))
        else:
            groups.append([(a, b)])
    for group in groups:
        lo, hi = group[0][0], group[-1][1]
        if hi - lo < MIN_MS and not np.asarray(keep)[lo:hi].any():
            drop = np.zeros(len(c.state), bool)
            drop[lo:hi] = c.state[lo:hi] == PRESENT
            c.state[drop] = np.asarray(base)[drop]
            c.conf[drop] = np.nan
            continue
        for (_, b), (a, _) in pairwise(group):
            gap = np.zeros(len(c.state), bool)
            gap[b:a] = c.state[b:a] == ABSENT
            c.state[gap] = PRESENT
            c.conf[gap] = np.nan


def child_cells(got: Shot, parent: Cells, name: str, *, extra_present=(), no_q1=()):
    """The child m/n's cells inside `parent`'s present time (module docstring)."""
    n = int(name.split("/")[1])
    c = Cells(got.info["window"])
    c.state[:] = np.where(parent.state == UNSEEN, UNSEEN, ABSENT)
    inside = parent.state == PRESENT
    if inside.any():
        T = got.tracks
        eq_tree = tree(got.info)
        if eq_tree is None:
            c.state[inside] = UNSEEN
        else:
            mid = eq_cells(eq_times(got.info), c.lo, c.hi)
            c.state[inside & ~mid] = UNSEEN
            if len(T):
                same_n = (T.n.astype(int) == n).values
                failed = (
                    same_n
                    & T.why.astype(str)
                    .str.startswith(("model failed", "no views"))
                    .values
                )
                for a, b, _ in _track_spans(got, failed, values=np.zeros(failed.sum())):
                    c.paint(a, b, UNSEEN, where=inside)
                col = f"p_{name}"
                unsure = (
                    same_n
                    & T.status.isin(UNSURE).values
                    & T.set90.astype(str)
                    .str.split("|")
                    .apply(lambda s: name in s)
                    .values
                )
                if eq_tree not in TRUSTED:  # EFIT01's q_min reads high: never absent
                    unsure |= (
                        same_n
                        & (T.status == "q_min above").values
                        & (T.cand == name).values
                    )
                for a, b, x in _track_spans(got, unsure, col):
                    c.paint(a, b, UNCERTAIN, x, where=inside)
                res = same_n & (T.status == "m/n").values & (T.label == name).values
                for a, b, x in _track_spans(got, res, "p_res"):
                    c.paint(a, b, PRESENT, x, where=inside)
        for a, b in extra_present:  # a resonant 1/1 track's cells keep their confidence
            c.paint(a, b, PRESENT, np.nan, where=inside & (c.state != PRESENT))
    for a, b in no_q1:  # no q = 1 surface, no 1/1 mode; not observable stays so
        c.paint(a, b, ABSENT, where=c.state != UNSEEN)
    return c


def shot_cells(got: Shot, *, crash=(), no_q1=None) -> dict[str, Cells]:
    """Every event's cells for the shot. `crash`: the crash table's present spans for
    mhd_n1 and mhd_n1.m1; `no_q1`: the q gate's no-q=1 spans (None: gate not applied)."""
    out = {
        e: parent_cells(got, n, crash=crash if n == 1 else ())
        for e, n in PARENTS.items()
    }
    tracks_n1 = parent_cells(got, 1) if len(crash) else out["mhd_n1"]
    for event, name in CHILDREN.items():
        parent = out[event.split(".")[0]]
        if event == "mhd_n1.m1":
            out[event] = child_cells(
                got, parent, name, extra_present=crash, no_q1=no_q1 or ()
            )
        elif event.startswith("mhd_n1."):  # the crash is 1/1 evidence only
            out[event] = child_cells(got, tracks_n1, name)
        else:
            out[event] = child_cells(got, parent, name)
    return out


def clipped_ms(parent: Cells, extra) -> float:
    """How much of the `extra` spans lies outside `parent`'s present time (ms); `parent`
    may be any event's cells (the crash's time outside mhd_n1, or outside m1)."""
    total = 0.0
    for a, b in extra:
        sel = parent.mask(a, b)
        total += float((sel & (parent.state != PRESENT)).sum())
    return total


def crash_only_ms(got: Shot, crash) -> float:
    """How much of mhd_n1's present time comes from `crash` alone, the union's merged gaps
    included (ms)."""
    tracks_only = parent_cells(got, 1)
    with_crash = parent_cells(got, 1, crash=crash)
    return float(((with_crash.state == PRESENT) & (tracks_only.state != PRESENT)).sum())
