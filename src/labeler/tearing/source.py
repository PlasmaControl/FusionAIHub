"""The tearing-mode review source `tm_mhd`: the RMS rule's labels joined with the MHD mode
tracks (round 5, decisions S5 as amended on 2026-10-14, S6 and EQ4).

Cells are whole ms over the shot's window, states 0 absent, 1 present, 2 uncertain,
3 not observable, each decided at the cell's centre.

- `rule_cells`: the rule table's span rows (`extend_tm_interval`); where rows overlap, not
  observable wins over present, present over uncertain, uncertain over absent (an n = 2
  sub-seed line does not undo an n = 1 interval). Onset points (zero length) are dropped.
  `locked_cells`: the rule's uncertain rows whose reason is a locked phase
  (`LOCKED_REASONS`); a locked island does not rotate, so the rotating-mode tracks say
  nothing about it.
- `track_classes`: each labelled track of `labeler.mhd` (`mhd_modes`' cache) is
  `resonant` (status m/n with a tearing label of `TM_LABELS`: 2/1, 3/1, 3/2, 4/3, and 4/2,
  an island at q = 2), `q1` (status m/n with m == n: 1/1 and its k/k harmonics),
  `not_resonant`, or `unsure`. Unsure is a track whose status is uncertain, no rotation or
  q_min above; a track the model could not judge (no equilibrium, no slice within 30 ms,
  no views, model failed); a track with status m/n and another m != n label (4/1, 5/2,
  5/3, 7/3: outside the named classes, the resonance law was not fit on them;
  `unsure:label` with `detail`); and an n >= 2 track that would be resonant but sits at n
  times a simultaneous 1/1 track's frequency (`q1_harmonics`: an unlocked harmonic of the
  1/1, or a mode that cannot be told from one; `unsure:harmonic`). Lines and harmonics take
  no class: a harmonic follows its fundamental.
- `merge`: present is the rule's present time united with the resonant spans (frames
  +-`PAD_MS`, gaps up to `MERGE_MS` joined, spans of at least `MIN_MS`). Rule-uncertain time
  becomes absent where the MHD observed it (`observed_cells`: the Mirnov record, an
  equilibrium tree with a slice within 30 ms, a ZIPFIT rotation profile within 60 ms) and
  only `q1` or `not_resonant` tracks cover it, except in a locked phase; time with an
  `unsure` track, a resonant track too short for a span, or no track at all stays
  uncertain. Rule absent and not observable stay as they are (unless a resonant span makes
  them present). Where the shot has no rule label, the MHD alone decides: present on the
  resonant spans, absent where it observed and found no `unsure` or resonant track, not
  observable where the Mirnov was never read (`read_cells`, as the MHD events do: decision
  M6), uncertain elsewhere.
- `resolved_bins`: the benchmark's `mhd_resolved` view, the rule's bin targets with the
  rule-uncertain bins taking this table's present or absent.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import scoring
from .rule import LOCK_REASONS

ABSENT, PRESENT, UNCERTAIN, NOT_OBSERVABLE = 0, 1, 2, 3
NO_RULE = -1
#: Overlapping rule rows: the first of these wins.
PRECEDENCE = (NOT_OBSERVABLE, PRESENT, UNCERTAIN, ABSENT)
PAD_MS, MERGE_MS, MIN_MS = 2.0, 50.0, 50.0
EQ_MAX_DT, ROT_MAX_DT = 30.0, 60.0
UNSURE_STATUSES = ("uncertain", "no rotation", "q_min above")
UNJUDGED = ("no equilibrium", "no views", "model failed")
CLASSES = ("resonant", "q1", "not_resonant", "unsure")
#: `track_classes(..., detail=True)`'s two further kinds of unsure track.
DETAILS = ("unsure:label", "unsure:harmonic")
#: Decision S5 as amended: the m/n whose resonant tracks make a tearing mode present.
TM_LABELS = ("2/1", "3/1", "3/2", "4/3", "4/2")
#: The rule's uncertain reasons for a locked phase, which the MHD never makes absent.
LOCKED_REASONS = LOCK_REASONS
#: `labeler.mhd.rules.harmonic_lock`'s frequency tolerance, |k f_1 - f| <= max(kHz, rel f).
HARM_KHZ, HARM_REL = 0.3, 0.02
#: An n >= 2 track is a 1/1 harmonic when at least this share of its frames is at n f_1.
HARM_SHARE = 0.5


def rule_cells(rows: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    """The rule's state per ms of [lo, hi), `NO_RULE` where no span row covers it."""
    mid = np.arange(lo, hi) + 0.5
    out = np.full(len(mid), NO_RULE, np.int64)
    spans = rows[rows.t_end > rows.t_start]
    for state in reversed(PRECEDENCE):
        kept = spans[spans.category.astype(int) == state]
        for a, b in zip(kept.t_start.astype(float), kept.t_end.astype(float)):
            out[(mid >= a) & (mid < b)] = state
    return out


def reasons(rows: pd.DataFrame) -> np.ndarray:
    """Each rule row's `attrs` reason, "" without one."""
    if "attrs" not in rows:
        return np.full(len(rows), "", object)
    out = []
    for value in rows["attrs"]:
        if isinstance(value, str) and value.strip():
            value = json.loads(value)
        out.append(str(value.get("reason", "")) if isinstance(value, dict) else "")
    return np.array(out, object)


def locked_cells(rows: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    """Per ms of [lo, hi): an uncertain rule row with a locked-phase reason
    (`LOCKED_REASONS`) covers it."""
    mid = np.arange(lo, hi) + 0.5
    out = np.zeros(len(mid), bool)
    keep = (
        (rows.t_end > rows.t_start).values
        & (rows.category.astype(int) == UNCERTAIN).values
        & np.isin(reasons(rows), LOCKED_REASONS)
    )
    for a, b in zip(rows.t_start[keep].astype(float), rows.t_end[keep].astype(float)):
        out |= (mid >= a) & (mid < b)
    return out


def q1_harmonics(tracks: pd.DataFrame, frames: pd.DataFrame) -> np.ndarray:
    """Per track: n >= 2, and at least `HARM_SHARE` of its frames lie at n times the
    frequency of a resonant 1/1 track's frame at the same time (within `HARM_KHZ` or
    `HARM_REL`, the harmonic lock's tolerance)."""
    out = np.zeros(len(tracks), bool)
    if not len(tracks) or not len(frames):
        return out
    n = tracks.n.astype(int).values
    one = (
        (tracks.label.astype(str) == "1/1") & (tracks.status.astype(str) == "m/n")
    ).values
    if not one.any() or not (n >= 2).any():
        return out
    base = frames[frames.family_id.isin(set(tracks.family_id[one]))]
    base = base[["t_ms", "f0"]].rename(columns={"f0": "f_1"})
    fr = frames[frames.family_id.isin(set(tracks.family_id[n >= 2]))]
    fr = fr[["family_id", "t_ms", "f0"]].assign(row=np.arange(len(fr)))
    pair = fr.merge(base, on="t_ms")
    k = pair.family_id.map(dict(zip(tracks.family_id, n))).values
    f = pair.f0.astype(float).values
    hit = np.abs(k * pair.f_1.astype(float).values - f) <= np.maximum(
        HARM_KHZ, HARM_REL * f
    )
    rows = np.zeros(len(fr), bool)
    rows[pair.row.values[hit]] = True
    share = pd.Series(rows).groupby(fr.family_id.values).mean()
    return tracks.family_id.isin(set(share.index[share >= HARM_SHARE])).values


def track_classes(tracks: pd.DataFrame, frames=None, *, detail=False) -> np.ndarray:
    """Each track's class (module docstring), "" for lines, harmonics and the rest.
    Without `frames` no track is tested as a 1/1 harmonic; with `detail` an unsure track
    for its label or as a 1/1 harmonic reads `unsure:label` or `unsure:harmonic`."""
    out = np.full(len(tracks), "", object)
    if not len(tracks):
        return out
    modelled = tracks.modelled.astype(bool).values
    status = tracks.status.astype(str).values
    label = tracks.label.astype(str).values
    why = tracks.why.astype(str)
    same = np.array([_m_equals_n(v) for v in label])
    named = np.isin(label, TM_LABELS)
    resonant = modelled & (status == "m/n")
    out[resonant & ~same & named] = "resonant"
    out[resonant & ~same & ~named] = "unsure:label"
    out[resonant & same] = "q1"
    out[modelled & (status == "not resonant")] = "not_resonant"
    out[modelled & np.isin(status, UNSURE_STATUSES)] = "unsure"
    out[~modelled & why.str.startswith(UNJUDGED).values] = "unsure"
    if frames is not None:
        out[(out == "resonant") & q1_harmonics(tracks, frames)] = "unsure:harmonic"
    if not detail:
        out[np.isin(out, DETAILS)] = "unsure"
    return out


def _m_equals_n(label: str) -> bool:
    m, _, n = label.partition("/")
    return bool(n) and int(m) == int(n)


def spans(t_ms, value, *, pad=PAD_MS, merge=MERGE_MS, min_ms=0.0):
    """`(start, stop, max value)` of frames `t_ms` padded by `pad`, joined across gaps of
    up to `merge`, kept when at least `min_ms` long (`labeler.mhd.rows.spans`' rule)."""
    t = np.asarray(t_ms, float)
    v = np.asarray(value, float)
    out: list[list[float]] = []
    for i in np.argsort(t, kind="stable"):
        a, b = t[i] - pad, t[i] + pad
        if out and a - out[-1][1] <= merge:
            out[-1][1] = max(out[-1][1], b)
            out[-1][2] = np.fmax(out[-1][2], v[i])
        else:
            out.append([a, b, v[i]])
    return [(a, b, x) for a, b, x in out if b - a >= min_ms]


def _paint(lo, hi, found, conf=None):
    mid = np.arange(lo, hi) + 0.5
    mask = np.zeros(len(mid), bool)
    for a, b, x in found:
        sel = (mid >= a) & (mid < b)
        mask |= sel
        if conf is not None:
            conf[sel] = np.fmax(conf[sel], x)
    return mask


def track_masks(tracks, frames, lo: int, hi: int) -> dict[str, np.ndarray]:
    """Per ms of [lo, hi): `present` (resonant spans), `conf` (their largest P(resonant)),
    `seen` (q1 or not_resonant cover), `unsure` (every unsure track's cover and resonant
    cover outside the present spans) and `cls` (the cover of each class and of the two
    `DETAILS`, which the class `unsure` there leaves out, for the report)."""
    cls = track_classes(tracks, frames, detail=True)
    n = max(0, hi - lo)
    conf = np.full(n, np.nan)
    cover = {}
    present = np.zeros(n, bool)
    for name in CLASSES + DETAILS:
        sel = cls == name
        ids = set(tracks.family_id[sel]) if len(tracks) else set()
        fr = frames[frames.family_id.isin(ids)] if ids else frames.iloc[:0]
        cover[name] = _paint(lo, hi, spans(fr.t_ms, np.zeros(len(fr))))
        if name == "resonant" and ids:
            p_res = dict(zip(tracks.family_id[sel], tracks.p_res[sel].astype(float)))
            found = spans(fr.t_ms, fr.family_id.map(p_res), min_ms=MIN_MS)
            present = _paint(lo, hi, found, conf)
    unsure = cover["unsure"] | cover["unsure:label"] | cover["unsure:harmonic"]
    return {
        "present": present,
        "conf": np.where(present, conf, np.nan),
        "seen": cover["q1"] | cover["not_resonant"],
        "unsure": unsure | (cover["resonant"] & ~present),
        "cls": cover,
    }


def near_cells(times, lo: int, hi: int, max_dt: float) -> np.ndarray:
    """Per ms of [lo, hi): a time of `times` lies within `max_dt` of the cell's centre."""
    mid = np.arange(lo, hi) + 0.5
    t = np.sort(np.asarray(times, float))
    t = t[np.isfinite(t)]
    if not len(t) or not len(mid):
        return np.zeros(len(mid), bool)
    j = np.clip(np.searchsorted(t, mid), 1, max(1, len(t) - 1))
    near = np.minimum(
        np.abs(t[j - 1] - mid), np.abs(t[np.minimum(j, len(t) - 1)] - mid)
    )
    return near <= max_dt


def read_cells(info: dict, mirnov: bool, lo: int, hi: int) -> np.ndarray:
    """Per ms of [lo, hi): the Mirnov record was read there, as the MHD events count it
    (`labeler.mhd.rows`): a readable record, inside its finite stretch, outside its NaN
    gaps."""
    mid = np.arange(lo, hi) + 0.5
    if not mirnov:
        return np.zeros(len(mid), bool)
    a, b = info.get("mirnov_ms") or info["window"]
    out = (mid >= a) & (mid < b)
    for a, b in info.get("mirnov_gaps", ()):
        out &= ~((mid >= a) & (mid < b))
    return out


def observed_cells(info: dict, mirnov: bool, tree, rot_times, lo: int, hi: int):
    """Decision EQ4, per ms of [lo, hi): the Mirnov record was read there
    (`read_cells`), the shot has an equilibrium tree with a slice within `EQ_MAX_DT`, and
    a rotation profile lies within `ROT_MAX_DT`."""
    if tree is None:
        return np.zeros(max(0, hi - lo), bool)
    eq_t = info.get("eq_t", info.get("efit02_t", []))
    return (
        read_cells(info, mirnov, lo, hi)
        & near_cells(eq_t, lo, hi, EQ_MAX_DT)
        & near_cells(rot_times, lo, hi, ROT_MAX_DT)
    )


def merge(rule, observed, present, seen, unsure, *, locked=None, read=None):
    """The source's state per ms (module docstring). `rule` holds `NO_RULE` where the rule
    has no label; `observed`, `present`, `seen` and `unsure` are `observed_cells`' and
    `track_masks`' masks; `locked` (`locked_cells`) is never made absent; `read`
    (`read_cells`) marks where the Mirnov was read, and a cell without a rule label
    outside it is not observable (None: everywhere, no locked phase)."""
    rule = np.asarray(rule, np.int64)
    locked = np.zeros(len(rule), bool) if locked is None else np.asarray(locked, bool)
    read = np.ones(len(rule), bool) if read is None else np.asarray(read, bool)
    alone = np.where(observed & ~unsure, ABSENT, UNCERTAIN)
    out = np.where(rule == NO_RULE, alone, rule)
    out[(rule == UNCERTAIN) & observed & seen & ~unsure & ~locked] = ABSENT
    out[present] = PRESENT
    out[(rule == NO_RULE) & ~read] = NOT_OBSERVABLE
    return out


def table_rows(shot: int, lo: int, state, conf) -> list[list]:
    """Rows `[shot, category, t_start, t_end, confidence]` of the runs of equal state and
    confidence (3 decimals; blank where there is none)."""
    state = np.asarray(state, np.int64)
    if not len(state):
        return []
    c = np.round(np.asarray(conf, float), 3)
    key = state * 10_000 + np.nan_to_num(c * 1000, nan=-1).astype(np.int64)
    edges = np.flatnonzero(np.diff(key)) + 1
    starts, stops = np.r_[0, edges], np.r_[edges, len(key)]
    return [
        [
            int(shot),
            int(state[a]),
            lo + int(a),
            lo + int(b),
            "" if np.isnan(c[a]) else float(c[a]),
        ]
        for a, b in zip(starts, stops)
    ]


def resolved_bins(rule_rows, source_rows, centres) -> tuple[np.ndarray, np.ndarray]:
    """`(y, valid)` of the `mhd_resolved` view over bin `centres`: the rule's targets
    (`scoring.label_bins`), except that a bin the rule leaves uncertain (not one it calls
    not observable) is positive where the source table is present there and negative where
    it is absent; elsewhere it stays unscored."""
    y, valid = scoring.label_bins(rule_rows, centres)
    span = rule_rows[rule_rows.t_end > rule_rows.t_start]

    def of(frame, category):
        kept = frame[frame.category.astype(int) == category]
        return scoring._inside(centres, zip(kept.t_start, kept.t_end, strict=True))

    open_ = of(span, UNCERTAIN) & ~of(span, NOT_OBSERVABLE)
    present = open_ & of(source_rows, PRESENT)
    absent = open_ & of(source_rows, ABSENT) & ~present
    y = y.copy()
    y[present], y[absent] = 1, 0
    return y, valid | present | absent
