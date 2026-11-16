"""Find measured shots that match conditions on actuators and plasma quantities.

Every tokenized shot is in the index (:mod:`.index`) with 50 ms series of

=================  ===============================================================
``pnbi_MW``        total neutral beam power
``torque_Nm``      net beam torque (+ co-current)
``counter_MW``     beam power injected counter-current
``ech_MW``         total ECH power
``gas_TorrL_s``    total gas flow
``rot_core_kms``   core toroidal rotation |v|, 90th percentile over CER T02, T04-T08
``ti_core_keV``    core T_i, 90th percentile over the same chords
``te_core_keV``    core T_e (90th percentile over the core Thomson chords)
``ne_core_1e19``   core n_e (75th percentile over the core Thomson chords)
=================  ===============================================================

The plasma quantities are decoded from the shot's own tokens by the frozen codecs (what
IGNITE sees), NaN where the diagnostic was not recorded, and NaN after the plasma end.
Conditions are ``"<series> <op> <value>"`` with op in ``< <= > >=``; a shot matches
when all of them hold together for at least ``min_duration_s`` inside ``window``.
Scalar filters use ``shot``, ``ip_peak_MA`` and ``t_end``.
"""

from __future__ import annotations

import operator
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import machine as mc
from .index import SERIES

_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
_COND = re.compile(r"^\s*([a-zA-Z_0-9]+)\s*(<=|>=|<|>)\s*(-?[0-9.eE+-]+)\s*$")
SCALARS = ("shot", "ip_peak_MA", "t_end")


@lru_cache(maxsize=1)
def _index() -> Dict[str, np.ndarray]:
    path = mc.INDEX_DIR / "shots.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; build it with "
                                "`python -m tokamak_foundation_model.ignite.designer.index`")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _parse(conditions: Sequence[str]):
    out = []
    for c in conditions:
        m = _COND.match(c)
        if not m:
            raise ValueError(f"condition {c!r} is not '<name> <op> <value>'")
        name, op, value = m.groups()
        if name not in SERIES and name not in SCALARS:
            raise ValueError(f"unknown quantity {name!r}; series: {SERIES}, scalars: {SCALARS}")
        out.append((name, _OPS[op], float(value), c))
    return out


def series(shot: int) -> Dict[str, np.ndarray]:
    """All index series of one shot, with ``t_s`` (frame start times)."""
    z = _index()
    i = int(np.searchsorted(z["shot"], shot))
    if i >= len(z["shot"]) or z["shot"][i] != shot:
        raise KeyError(f"shot {shot} is not in the index")
    a, b = z["offsets"][i], z["offsets"][i + 1]
    out = {k: z[f"series__{k}"][a:b].astype(np.float32) for k in SERIES}
    out["t_s"] = mc.time_of(np.arange(b - a))
    return out


def _runs(mask: np.ndarray):
    """(start, length) of the longest run of True."""
    best = (0, 0)
    start = None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1]:
                best = (start, i - start)
            start = None
    return best


def search(conditions: Sequence[str] = (), window=(2.0, 6.0), min_duration_s: float = 0.5,
           near: Optional[int] = None, exclude: Sequence[int] = (), limit: int = 20) -> dict:
    """Shots in which all ``conditions`` hold together for ``min_duration_s`` in ``window``.

    Ranked by the length of the matching interval, or, with ``near`` (a shot number), by
    how close the shot's series are to that shot's over ``window`` (mean absolute
    difference in units of each series' spread across all shots, over the series both
    shots recorded). Returns the matches with their matching interval and the series
    means over it.
    """
    z = _index()
    conds = _parse(conditions)
    t0, t1 = window
    n_need = max(1, int(round(min_duration_s / mc.FRAME_S)))
    spread = {k: float(np.nanstd(z[f"series__{k}"].astype(np.float32))) or 1.0 for k in SERIES}
    ref = None
    if near is not None:
        r = series(near)
        w = (r["t_s"] >= t0) & (r["t_s"] < t1)
        ref = {k: float(np.nanmean(r[k][w])) if np.isfinite(r[k][w]).any() else np.nan
               for k in SERIES}
    hits = []
    excluded = set(int(x) for x in exclude)
    for i, shot in enumerate(z["shot"]):
        shot = int(shot)
        if shot in excluded:
            continue
        scal = {"shot": shot, "ip_peak_MA": float(z["ip_peak_MA"][i]), "t_end": float(z["t_end"][i])}
        if any(name in SCALARS and not op(scal[name], v) for name, op, v, _ in conds):
            continue
        a, b = z["offsets"][i], z["offsets"][i + 1]
        t = mc.time_of(np.arange(b - a))
        w = (t >= t0) & (t < t1)
        ok = w.copy()
        for name, op, v, _ in conds:
            if name in SCALARS:
                continue
            x = z[f"series__{name}"][a:b].astype(np.float32)
            with np.errstate(invalid="ignore"):
                ok &= op(x, v)  # NaN (not recorded, after the plasma) never matches
        start, length = _runs(ok)
        if length < n_need:
            continue
        sel = slice(a + start, a + start + length)
        means = {}
        for k in SERIES:
            x = z[f"series__{k}"][sel].astype(np.float32)
            means[k] = round(float(np.nanmean(x)), 3) if np.isfinite(x).any() else None
        hit = {"shot": shot, "token_set": str(z["token_set"][i]),
               "match_s": [round(float(t[start]), 2), round(float(t[start] + length * mc.FRAME_S), 2)],
               "duration_s": round(length * mc.FRAME_S, 2), "means": means,
               "ip_peak_MA": round(scal["ip_peak_MA"], 3), "plasma_end_s": round(scal["t_end"], 2)}
        if ref is not None:
            mine = {k: z[f"series__{k}"][a:b].astype(np.float32)[w] for k in SERIES}
            d = [abs(np.nanmean(mine[k]) - ref[k]) / spread[k] for k in SERIES
                 if np.isfinite(ref[k]) and np.isfinite(mine[k]).any()]
            hit["distance"] = round(float(np.mean(d)), 3) if d else None
        hits.append(hit)
    if ref is not None:
        hits.sort(key=lambda h: (h.get("distance") is None, h.get("distance") or 0))
    else:
        hits.sort(key=lambda h: -h["duration_s"])
    return {"conditions": list(conditions), "window_s": list(window),
            "min_duration_s": min_duration_s, "near": near, "n_matches": len(hits),
            "matches": hits[:limit],
            "caveat": "plasma quantities are codec decodes of the tokens (what IGNITE sees), "
                      "reduced to core values by fixed chord choices; check a precedent's "
                      "raw data before relying on it"}
