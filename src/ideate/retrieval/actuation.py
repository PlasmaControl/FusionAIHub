"""Actuator waveforms: seed breakpoints from a shot's own trace or the median of several, check
them against the limit rules, save and reload them under the data root.

A seed is the decimated trace the plots use (`read_traces`) simplified with
Ramer-Douglas-Peucker. The distance is VERTICAL (value minus the chord at that time), not the
perpendicular of the textbook version: time and the actuator's units are different quantities,
so "within tolerance of the trace" means the reconstructed waveform is never more than `tol` off
the measured value at any sample. Endpoints are always kept and no point is ever dropped; the
vertex cap is met by raising the tolerance, bisected onto the budget (`_fit_cap`).


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise

import numpy as np

from .. import config
from ..config import Paths, SignalSpec
from ..flags import rules
from ..schema import ActuationSet, ActuatorWaveform, Flag, Vertex
from ..shotdb import features, legacy_raw


@dataclass
class Trace:
    name: str  # what was asked for: "ip", "nbi.total", "raw:magnetics/mpi66m307d"
    label: str
    units: str | None
    t_ms: list[float]
    y: list[float | None]  # None where the raw value is not finite (JSON has no NaN)
    status: str  # present | unavailable | pending | not_installed | unknown_signal | no_samples_in_window
    source: str | None = None  # staged | fetched
    members: list[str] = field(default_factory=list)  # for a total: the members summed


def decimate(t: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """At most about 2n points: the index of each bucket's min and max, in time order.

    Returns the inputs unchanged (same objects) when they already fit. A bucket that is all NaN
    keeps its first sample so the gap still shows on the plot.
    """
    if n < 2 or t.size <= 2 * n:
        return t, y
    edges = np.linspace(0, t.size, n + 1).astype(int)
    keep: set[int] = set()
    for a, b in pairwise(edges):
        if b <= a:
            continue
        seg = y[a:b]
        if not np.isfinite(seg).any():
            keep.add(int(a))
            continue
        keep.add(a + int(np.nanargmin(seg)))
        keep.add(a + int(np.nanargmax(seg)))
    idx = np.fromiter(sorted(keep), dtype=int)
    return t[idx], y[idx]


def _clip(t: np.ndarray, y: np.ndarray, t0: float | None, t1: float | None):
    if t0 is None and t1 is None:
        return t, y
    m = np.ones(t.size, dtype=bool)
    if t0 is not None:
        m &= t >= t0
    if t1 is not None:
        m &= t <= t1
    return t[m], y[m]


def _json_values(y: np.ndarray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in y]


def _trace(
    name: str,
    label: str,
    units: str | None,
    t: np.ndarray,
    y: np.ndarray,
    status: str,
    source: str | None,
    n: int,
    t0: float | None,
    t1: float | None,
    members: list[str] | None = None,
) -> Trace:
    t, y = _clip(t, y.astype(np.float64, copy=False), t0, t1)
    if status == "present" and t.size == 0:
        # The signal is in the file; the requested window just does not overlap what it recorded.
        # Saying "present" with empty arrays reads on the page as "recorded nothing", which is a
        # statement about the diagnostic rather than about the window the user asked for.
        status = "no_samples_in_window"
    t, y = decimate(t, y, n)
    return Trace(name, label, units, t.tolist(), _json_values(y), status, source, members or [])


def _empty_trace(name: str, label: str, units: str | None, status: str) -> Trace:
    return Trace(name, label, units, [], [], status)


def _specs(shot: int) -> dict[str, SignalSpec]:
    return {s.name: s for s in config.expand_registry(shot, include_not_installed=True)}


def _total(
    shot: int,
    system: str,
    sysdef: config.SystemSpec,
    specs: dict[str, SignalSpec],
    paths: Paths,
    n: int,
    t0: float | None,
    t1: float | None,
) -> Trace:
    """Sum of the installed members, computed by `features.system_totals` -- the same function the
    database build uses for `{prefix}_total_{stat}`.

    Delegating rather than re-summing here is the point: a second implementation would let the
    plotted `nbi.total` and the stored `pnbi_total_peak` disagree about the same shot. It also
    carries over that function's rule that a member's NaN is "not recorded", not "off", so a
    sample no member recorded stays NaN (None in the JSON) instead of being reported as zero.
    """
    name = f"{system}.total"
    signals: dict[str, legacy_raw.Signal | None] = {}
    for m in sysdef.members:
        key = f"{sysdef.prefix}_{m}"
        spec = specs.get(key)
        if spec is None:
            continue
        sig = legacy_raw.read_signal(shot, spec, paths)
        # A (1,1) NaN placeholder group decodes to a one-sample signal; it is not a time grid to
        # sum onto, so it is dropped here rather than becoming system_totals' reference member.
        if sig is not None and sig.t_ms.size > 1:
            signals[key] = sig
    if not signals:
        return _empty_trace(name, name, sysdef.units, "unavailable")
    totals = features.system_totals(signals, {system: sysdef})
    total = totals[f"{sysdef.prefix}_total"]
    members = [str(specs[k].member) for k in signals]
    return _trace(
        name, name, sysdef.units, total.t_ms, total.y, "present", total.source, n, t0, t1, members
    )


def read_traces(
    shot: int,
    names: list[str],
    paths: Paths,
    n: int = 1500,
    t0: float | None = None,
    t1: float | None = None,
) -> list[Trace]:
    """One Trace per requested name, in order; a name that cannot be served still comes back, with
    a status, so the page can say why a panel is empty instead of showing nothing."""
    specs = _specs(shot)
    systems = config.actuator_systems(shot)
    out: list[Trace] = []
    for name in names:
        if name.startswith("raw:"):
            group, _, col = name[4:].partition("/")
            spec = SignalSpec(name=name, group=group, col=col)
            sig = legacy_raw.read_signal(shot, spec, paths) if group and col else None
            label = f"{group}/{col}"
            if sig is None:
                out.append(_empty_trace(name, label, None, "unavailable"))
            else:
                out.append(
                    _trace(
                        name, label, sig.units, sig.t_ms, sig.y, "present", sig.source, n, t0, t1
                    )
                )
        elif name.endswith(".total") and name[:-6] in systems:
            out.append(_total(shot, name[:-6], systems[name[:-6]], specs, paths, n, t0, t1))
        elif "." in name and name.split(".", 1)[0] in systems:
            system, member = name.split(".", 1)
            sysdef = systems[system]
            # `specs` was built with include_not_installed=True, so a key present there but with
            # installed=False is a real member just not fitted on this shot; a key absent
            # altogether is not a member of the system at all (falls to unknown_signal below).
            spec = specs.get(f"{sysdef.prefix}_{member}")
            if spec is None:
                out.append(_empty_trace(name, name, None, "unknown_signal"))
                continue
            if not spec.installed:
                out.append(_empty_trace(name, name, spec.units, "not_installed"))
                continue
            sig = legacy_raw.read_signal(shot, spec, paths)
            if sig is None:
                out.append(_empty_trace(name, name, spec.units, legacy_raw.signal_status(shot, spec, paths)))
            else:
                out.append(
                    _trace(
                        name,
                        name,
                        sig.units or spec.units,
                        sig.t_ms,
                        sig.y,
                        "present",
                        sig.source,
                        n,
                        t0,
                        t1,
                    )
                )
        elif name in specs:
            spec = specs[name]
            if not spec.installed:
                out.append(_empty_trace(name, name, spec.units, "not_installed"))
                continue
            sig = legacy_raw.read_signal(shot, spec, paths)
            if sig is None:
                out.append(_empty_trace(name, name, spec.units, legacy_raw.signal_status(shot, spec, paths)))
            else:
                out.append(
                    _trace(
                        name,
                        name,
                        sig.units or spec.units,
                        sig.t_ms,
                        sig.y,
                        "present",
                        sig.source,
                        n,
                        t0,
                        t1,
                    )
                )
        else:
            out.append(_empty_trace(name, name, None, "unknown_signal"))
    return out


Reader = Callable[[int, list[str]], list[Trace]]
ID_RE = re.compile(r"^\d{8}T\d{6}\.\d{6}Z-\d+$")
_EMPTY = [Vertex(t_s=0.0, y=0.0), Vertex(t_s=1.0, y=0.0)]  # an absent or empty trace


def _empty() -> list[Vertex]:
    """A fresh pair of zero vertices. Copies, never `_EMPTY` itself: pydantic does not revalidate
    a model instance, so handing the module-level objects out would let one edited seed change
    every other absent actuator in the process."""
    return [v.model_copy() for v in _EMPTY]


def load_cfg() -> dict:
    return config.load_yaml("ui.yaml")["actuation"]


# ------------------------------------------------------------------------------ simplify


def rdp(t: np.ndarray, y: np.ndarray, tol: float) -> np.ndarray:
    """Indices to keep (sorted); the first and last always. Iterative, vertical distance."""
    n = t.size
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        chord = y[a] + (y[b] - y[a]) * (t[a + 1 : b] - t[a]) / (t[b] - t[a])
        d = np.abs(y[a + 1 : b] - chord)
        i = int(np.argmax(d))
        if d[i] > tol:
            k = a + 1 + i
            keep[k] = True
            stack.append((a, k))
            stack.append((k, b))
    return np.flatnonzero(keep)


def _fit_cap(t: np.ndarray, y: np.ndarray, tol: float, cap: int, span: float) -> np.ndarray:
    """The vertex set of the largest count that is still <= `cap`, found by raising the tolerance.

    Doubling alone overshoots: RDP's count falls in rungs, and a doubling can step straight over
    the rung that would have spent the budget. Measured on a 3-period sine (600 samples, 0.03 of
    the span): 28 -> 20 -> 13 -> 8 vertices, so a cap of 12 was met with 8 when the 12-vertex rung
    sits at tol = 2.5e5, between the last two doublings. So: double only to BRACKET the answer
    (`lo` over the cap, `hi` under it), then bisect that bracket -- 40 rounds, or until it is
    narrower than 1e-6 of the span -- and keep the largest vertex set seen that fits. Points are
    never dropped to meet the cap and the endpoints are never touched; only the tolerance moves.

    A rung can still be far below the cap when no set of that size exists: the same sine at 6
    periods has 12 interior extrema, its rungs are 14 and then 2, and no 12-vertex piecewise-linear
    fit can separate 12 extrema with 11 segments anyway. There the answer is the endpoint chord.
    """
    lo, hi, best = tol, (tol if tol > 0 else span * 0.01 or 1.0), None
    for _ in range(64):  # the chord of the endpoints always fits, so this terminates
        hi *= 2
        cand = rdp(t, y, hi)
        if cand.size <= cap:
            best = cand
            break
        lo = hi
    if best is None:  # unreachable for finite data; never return an over-cap set silently
        return rdp(t, y, hi)
    for _ in range(40):
        if hi - lo <= 1e-6 * span:
            break
        mid = (lo + hi) / 2
        cand = rdp(t, y, mid)
        if cand.size <= cap:
            hi = mid
            best = cand if cand.size > best.size else best
        else:
            lo = mid
    return best


def simplify(t_s: np.ndarray, y: np.ndarray, cfg: dict) -> list[Vertex]:
    """Breakpoints for a sampled trace: finite samples only, sorted, unique in time, RDP at
    `rdp_tol_frac` of the peak-to-peak range, at most `max_vertices` by raising the tolerance
    (`_fit_cap` bisects it onto the budget)."""
    t_s = np.asarray(t_s, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t_s) & np.isfinite(y)
    t_s, y = t_s[m], y[m]
    if t_s.size == 0:
        return _empty()
    order = np.argsort(t_s, kind="stable")
    t_s, y = t_s[order], y[order]
    _, first = np.unique(t_s, return_index=True)
    t_s, y = t_s[first], y[first]
    if t_s.size == 1:
        return [Vertex(t_s=float(t_s[0]), y=float(y[0]))]
    span = float(np.max(y) - np.min(y))
    tol = float(cfg["rdp_tol_frac"]) * span
    cap = int(cfg["max_vertices"])
    idx = rdp(t_s, y, tol)
    if idx.size > cap:
        idx = _fit_cap(t_s, y, tol, cap, span)
    return [Vertex(t_s=float(t_s[i]), y=float(y[i])) for i in idx]


def values_at(vertices: list[Vertex], t_s: np.ndarray) -> np.ndarray:
    """The waveform at `t_s`: linear between vertices, held at the ends (np.interp's rule)."""
    xs = np.array([v.t_s for v in vertices], dtype=float)
    ys = np.array([v.y for v in vertices], dtype=float)
    return np.interp(np.asarray(t_s, dtype=float), xs, ys)


# ------------------------------------------------------------------------------ seeds


def descriptions() -> dict[str, str]:
    """`nbi.total` / `ech.LUKE` -> the one-line `description:` from
    configs/ideate/actuators.yaml (the system's line for a total); keys without one are absent."""
    out: dict[str, str] = {}
    for name, sysdef in config.load_yaml("actuators.yaml")["systems"].items():
        if sysdef.get("description"):
            out[f"{name}.total"] = str(sysdef["description"])
        for m in sysdef["members"]:
            if m.get("description"):
                out[f"{name}.{m['id']}"] = str(m["description"])
    return out


def gas_species() -> list[str]:
    """The species a gas valve may be set to feed: `species:` under `gas:` in actuators.yaml,
    first entry the default. A selection, not a waveform."""
    return [str(s) for s in config.load_yaml("actuators.yaml")["systems"]["gas"]["species"]]


def check_gas_species(mapping: dict[str, str]) -> None:
    """Raise ValueError for a valve that is not a gas member or a species not in the vocabulary."""
    valves = {str(m["id"]) for m in config.load_yaml("actuators.yaml")["systems"]["gas"]["members"]}
    allowed = gas_species()
    for valve, species in mapping.items():
        if valve not in valves:
            raise ValueError(f"{valve!r} is not a gas valve (one of {', '.join(sorted(valves))})")
        if species not in allowed:
            raise ValueError(f"unknown gas species {species!r} (one of {', '.join(allowed)})")


def _arrays(tr: Trace) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(tr.t_ms, dtype=float) / 1000.0
    y = np.array([np.nan if v is None else v for v in tr.y], dtype=float)
    return t, y


def _read(reader: Reader | None, paths: Paths, n: int) -> Reader:
    return reader or (lambda shot, names: read_traces(shot, names, paths, n=n))


def seed_reference(
    shot: int, key: str, paths: Paths, cfg: dict, n: int = 1500, reader: Reader | None = None
) -> ActuatorWaveform:
    tr = _read(reader, paths, n)(shot, [key])[0]
    present = tr.status == "present" and len(tr.t_ms) > 0
    vertices = simplify(*_arrays(tr), cfg) if present else _empty()
    return ActuatorWaveform(
        key=key,
        units=tr.units,
        description=descriptions().get(key, ""),
        vertices=vertices,
        seed="reference",
        # Empty when nothing was read, exactly as `seed_median` does it: a beam that exists but was
        # never fired is a flat zero trace too, and the page must not call that "no data".
        source_shots=[shot] if present else [],
    )


def median_grid(
    series: list[tuple[np.ndarray, np.ndarray]], step_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Point-wise median of several (t, y) series on a common grid over the union of their
    extents. Each series is interpolated only inside its own extent and only through its finite
    samples, so a NaN or a shorter record does not vote at that time; a grid point nobody covers
    is NaN (simplify drops it). The order of a series is not trusted: the extent comes from
    min/max and each series is sorted before `np.interp`, which needs increasing x."""
    if not series:
        raise ValueError("median_grid needs at least one series")
    lo = min(float(np.min(t)) for t, _ in series)
    hi = max(float(np.max(t)) for t, _ in series)
    grid = np.arange(lo, hi + step_s / 2, step_s)
    rows = []
    for t, y in series:
        m = np.isfinite(y)
        t, y = t[m], y[m]
        order = np.argsort(t, kind="stable")
        t, y = t[order], y[order]
        if t.size >= 2:
            rows.append(np.interp(grid, t, y, left=np.nan, right=np.nan))
        elif t.size == 1:
            rows.append(np.where(np.isclose(grid, t[0]), y[0], np.nan))
        else:
            rows.append(np.full(grid.size, np.nan))
    with warnings.catch_warnings():
        # An all-NaN column is "nobody has data here", not a mistake; every other RuntimeWarning
        # from the median still surfaces.
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        med = np.nanmedian(np.vstack(rows), axis=0)
    return grid, med


def seed_median(
    shots: list[int], key: str, paths: Paths, cfg: dict, n: int = 1500, reader: Reader | None = None
) -> ActuatorWaveform:
    """`median_grid` of the shots' traces on a `grid_ms` grid, then simplified like a reference.
    `source_shots` lists the shots whose trace was present with at least two finite samples."""
    read = _read(reader, paths, n)
    traces = [(s, read(s, [key])[0]) for s in shots]
    units = next((tr.units for _, tr in traces if tr.units), None)
    present = []
    for s, tr in traces:
        if tr.status != "present" or len(tr.t_ms) < 2:
            continue
        t, y = _arrays(tr)
        m = np.isfinite(y)
        if m.sum() >= 2:
            present.append((s, t[m], y[m]))
    if not present:
        return ActuatorWaveform(
            key=key, units=units, description=descriptions().get(key, ""),
            vertices=_empty(), seed="median", source_shots=[],
        )  # fmt: skip
    grid, med = median_grid([(t, y) for _, t, y in present], float(cfg["grid_ms"]) / 1000.0)
    return ActuatorWaveform(
        key=key,
        units=units,
        description=descriptions().get(key, ""),
        vertices=simplify(grid, med, cfg),
        seed="median",
        source_shots=[s for s, _, _ in present],
    )


# ------------------------------------------------------------------------------ flags


def peak(w: ActuatorWaveform) -> float:
    """The SIGNED maximum of the drawn vertices, matching the database's signed `{prefix}_*_peak`
    columns (a 95th percentile of the raw signed samples) that the limit rules were fitted
    against, so the two sides of a comparison mean the same thing. The known limitation: a coil
    driven negative peaks near zero and never reaches the I-/C-coil rules. Deliberate, not an
    oversight -- an amplitude notion would have to change the database columns too."""
    return max(v.y for v in w.vertices)


def evaluate(w: ActuatorWaveform, key: str, rule_cfg: dict) -> list[Flag]:
    """The limit rules on the waveform's peak. Only violations: a rule that could not run on a
    lone actuator says so with value=None, and ten such lines under every plot would bury the one
    that matters."""
    return [f for f in rules.evaluate_flags({key: peak(w)}, None, rule_cfg) if f.value is not None]


def check(wfs: dict[str, ActuatorWaveform], rule_cfg: dict) -> dict[str, list[Flag]]:
    return {k: evaluate(w, k, rule_cfg) for k, w in wfs.items()}


# ------------------------------------------------------------------------------ storage


def new_id(source_shot: int | None, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return f"{now.strftime('%Y%m%dT%H%M%S.%f')}Z-{source_shot or 0}"


def save(aset: ActuationSet, paths: Paths, rule_cfg: dict | None = None) -> ActuationSet:
    """Validate the keys and the gas species, assign id and created, re-evaluate the flags (the
    stored flags are the server's, never the page's), write `<actuations_dir>/<id>.json`
    atomically, return what was stored. Every check runs before the write: a rejected set leaves
    no file behind."""
    rule_cfg = rule_cfg or rules.load_rules()
    # The same key vocabulary the seed route enforces: a set whose keys are not actuators cannot be
    # re-seeded, re-checked or read back by the page, so it is refused here rather than stored.
    unknown = sorted(set(aset.waveforms) - set(rules.actuator_columns()))
    if unknown:
        raise ValueError(f"unknown actuator key(s): {', '.join(unknown)}")
    check_gas_species(aset.gas_species)  # ValueError -> the route's 400
    now = datetime.now(UTC)
    flags = [f for k, w in aset.waveforms.items() for f in evaluate(w, k, rule_cfg)]
    stored = aset.model_copy(
        update={"id": new_id(aset.source_shot, now), "created": now, "flags": flags}
    )
    paths.actuations_dir.mkdir(parents=True, exist_ok=True)
    final = paths.actuations_dir / f"{stored.id}.json"
    tmp = final.with_suffix(".json.part")
    tmp.write_text(stored.model_dump_json(indent=1), encoding="utf-8")
    os.replace(tmp, final)
    return stored


def list_sets(paths: Paths) -> list[dict]:
    """`{id, created, source_shot, n_waveforms}` newest first (ids sort by their timestamp)."""
    out: list[dict] = []
    if not paths.actuations_dir.exists():
        return out
    for p in sorted(paths.actuations_dir.glob("*.json"), reverse=True):
        if not ID_RE.fullmatch(p.stem):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append(
            {
                "id": p.stem,
                "created": d.get("created"),
                "source_shot": d.get("source_shot"),
                "n_waveforms": len(d.get("waveforms") or {}),
            }
        )
    return out


def load(aid: str, paths: Paths) -> ActuationSet | None:
    if not ID_RE.fullmatch(aid):
        raise ValueError(f"not an actuation id: {aid!r}")
    p = paths.actuations_dir / f"{aid}.json"
    if not p.exists():
        return None
    return ActuationSet.model_validate_json(p.read_text(encoding="utf-8"))
