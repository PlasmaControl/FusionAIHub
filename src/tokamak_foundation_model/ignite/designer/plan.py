"""Actuator plans: a reference shot plus edit operations, built in physical units.

A plan is plain JSON, so an agent can write it::

    {"shot": 204011, "goal": "...", "rationale": "...",
     "ops": [{"op": "balance_torque", "t": 3.5, "off": ["NB15L", "NB30R"],
              "counter": ["NB21L"], "target_Nm": 0, "ramp_s": 1.0}]}

Operations (times in s on the 50 ms grid, values in display units: MW, kV, N m,
Torr L/s, kA). Each takes ``t`` (start), optional ``t_end`` (default: end of the
shot's frames) and optional ``ramp_s``: the channels then move linearly from their
current to their new values over ``ramp_s``, from the first frame that changes, like
IGNITE's ``linear:`` mode.

``beams``           switch sources ``off``, or run them at ``power_MW`` ({source: MW}),
                    optionally with ``kv``, ``torque_per_MW`` or ``direction`` ("co" or
                    "counter") per source. Torque and voltage follow the power (see
                    :func:`_couple_beams`). A source switched on follows the power of the
                    sources the op leaves alone (``hold``: "follow" default, "beam_phase",
                    "window"; see :func:`_op_beams`).
``balance_torque``  switch ``off`` sources off and size the ``counter`` sources, frame by
                    frame, so the net beam torque is ``target_Nm`` (IGNITE's ``balance:``
                    mode for ``target_Nm`` 0).
``scale``           multiply ``channels`` by ``factor``.
``set``             hold ``channels`` at ``value``.
``waveform``        draw ``channels`` through ``points`` [[t, value], ...] (linear between
                    points; outside the first and last point the channel is unchanged).
``shift``           delay (``dt`` > 0) or advance the measured waveform of ``channels``.
``copy_from``       take ``groups`` (default all) from another tokenized ``shot``.

``channels`` is ``"group"`` or ``"group:name1,name2"`` with the names of
:data:`machine.CHANNEL_NAMES` (e.g. ``"pinj:NB15L,NB15R"``, ``"gas_flow:GASA"``,
``"ech_power"``). Couplings are enforced while building: beam torque and voltage follow
beam power (torque and voltage cannot be edited on their own), the raw gas valve signal
follows the gas flow, and the RMP coil currents that appear in both ``rmp`` and ``i_coil``
change together.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import machine as mc
from . import shots

_P = mc.columns("pinj")
_V = mc.columns("beam_voltage")
_T = mc.columns("tinj")
_GF = mc.columns("gas_flow")
_GR = mc.columns("gas_raw")
# Below this a beam counts as off (W).
_BEAM_OFF_W = 5e4


@dataclass
class Built:
    """A built plan: the edited actuators and what was changed."""

    plan: dict
    shot: shots.Shot
    act: np.ndarray  # (F, 88) physical, edited
    notes: List[str] = field(default_factory=list)

    @property
    def measured(self) -> np.ndarray:
        return self.shot.act

    @property
    def changed(self) -> np.ndarray:
        """(F, 88) bool: channels and frames the plan changed."""
        return self.act != self.measured

    def first_change_s(self) -> Optional[float]:
        rows = np.nonzero(self.changed.any(1))[0]
        return None if not len(rows) else round(float(mc.time_of(rows[0])), 3)

    def normalised(self) -> np.ndarray:
        """IGNITE input (F, 88): the measured normalised actuators, with the changed
        entries replaced by their normalised plan values."""
        mu, sd = mc.act_stats()
        out = self.shot.act_norm.copy()
        varies = sd > 0
        edit = self.changed & varies[None, :]
        safe = np.where(varies, sd, 1.0)
        out[edit] = ((self.act - mu) / safe)[edit]
        return out

    def summary(self, t0: float = 2.0, t1: Optional[float] = None) -> dict:
        """Measured vs planned means of the derived totals and every changed channel."""
        s = self.shot
        t1 = t1 or (s.t_end or float(mc.time_of(s.n_frames)))
        first = self.first_change_s()
        out = {"shot": s.shot, "first_change_s": first, "notes": self.notes, "totals": {},
               "changed_channels": {}}
        windows = {"before_change": (t0, first or t1), "after_change": (first or t1, t1)}
        for label, (a, b) in windows.items():
            k = slice(max(0, mc.frame_of(_grid(a))), min(s.n_frames, mc.frame_of(_grid(b))))
            if k.stop <= k.start:
                continue
            out["totals"][label] = {
                "window_s": [round(a, 3), round(b, 3)],
                **{f"{name}_measured": round(float(fn(self.measured[k])), 3)
                   for name, fn in _TOTALS.items()},
                **{f"{name}_plan": round(float(fn(self.act[k])), 3)
                   for name, fn in _TOTALS.items()},
            }
        step = 5
        k_end = min(s.n_frames, mc.N_FRAMES)
        out["trace_0p25s"] = {"t_s": [round(float(x), 2) for x in mc.time_of(np.arange(mc.K0, k_end, step))]}
        for name, fn in _TOTALS.items():
            for label, a in (("measured", self.measured), ("plan", self.act)):
                out["trace_0p25s"][f"{name}_{label}"] = [
                    round(float(fn(a[k:k + step])), 2) for k in range(mc.K0, k_end, step)]
        ch = self.changed.any(0)
        for c in np.nonzero(ch)[0]:
            group = next(g for g in mc.GROUPS if mc.columns(g).start <= c < mc.columns(g).stop)
            name = mc.CHANNEL_NAMES[group][c - mc.columns(group).start]
            unit, scale = mc.DISPLAY[group]
            rows = self.changed[:, c]
            out["changed_channels"][f"{group}:{name}"] = {
                "unit": unit,
                "from_s": round(float(mc.time_of(np.nonzero(rows)[0][0])), 3),
                "to_s": round(float(mc.time_of(np.nonzero(rows)[0][-1] + 1)), 3),
                "measured_mean_over_changed_frames": round(float(self.measured[rows, c].mean() * scale), 3),
                "plan_mean_over_changed_frames": round(float(self.act[rows, c].mean() * scale), 3),
            }
        return out


def _grid(t: float) -> float:
    return round((t - mc.T0) / mc.FRAME_S) * mc.FRAME_S + mc.T0


_TOTALS = {
    "nbi_power_MW": lambda a: (a[:, _P] * 1e-6).sum(1).mean(),
    "nbi_torque_Nm": lambda a: a[:, _T].sum(1).mean(),
    "ech_power_MW": lambda a: (a[:, mc.columns("ech_power")] * 1e-6).sum(1).mean(),
    "gas_TorrL_s": lambda a: a[:, _GF].sum(1).mean(),
}


# --------------------------------------------------------------------------- build
def build(plan: dict) -> Built:
    """Apply a plan's operations, in order, to its reference shot's measured actuators."""
    if isinstance(plan, (str, Path)):
        plan = json.loads(Path(plan).read_text())
    s = shots.load(int(plan["shot"]))
    built = Built(plan=plan, shot=s, act=s.act.copy())
    for i, op in enumerate(plan.get("ops", [])):
        kind = op.get("op")
        if kind not in _OPS:
            raise ValueError(f"op {i}: unknown op {kind!r}; known: {sorted(_OPS)}")
        before = built.act.copy()
        frames = _frames(built, op)
        target = before.copy()
        _OPS[kind](built, target, frames, op)
        ramp = float(op.get("ramp_s", 0) or 0)
        if ramp > 0:
            target = _ramp(before, target, frames.start, ramp)
        built.act = target.astype(np.float32)
    return built


def plasma_end_frame(s: shots.Shot) -> int:
    """First frame after the plasma end (the whole record when it is unknown)."""
    if not s.t_end:
        return s.n_frames
    return min(s.n_frames, int(np.ceil((s.t_end - mc.T0) / mc.FRAME_S - 1e-9)))


def _frames(built: Built, op: dict) -> slice:
    """Frames an op acts on: ``t`` to ``t_end``. Without ``t_end``: to the plasma end
    (``copy_from``: the whole record, like IGNITE's ``donor:`` mode); ``"end"``: the whole
    record."""
    s = built.shot
    k0 = mc.frame_of(op["t"]) if "t" in op else mc.K0
    end = op.get("t_end")
    if end == "end" or (end is None and op.get("op") == "copy_from"):
        k1 = s.n_frames
    elif end is None:
        k1 = max(k0 + 1, plasma_end_frame(s))
    else:
        k1 = mc.frame_of(end)
        if k1 < plasma_end_frame(s) - 1:
            built.notes.append(
                f"{op.get('op')} ends at {end} s, before the plasma end ({s.t_end:.2f} s): "
                "from then on the measured actuators return")
    if not 0 <= k0 < k1 <= s.n_frames:
        raise ValueError(f"{op.get('op')}: frames {k0}-{k1} outside the shot's 0-{s.n_frames}")
    return slice(k0, k1)


def _ramp(before: np.ndarray, target: np.ndarray, k0: int, ramp_s: float) -> np.ndarray:
    """Move linearly from ``before`` to ``target`` over ``ramp_s``, starting at the first
    frame (at or after ``k0``) where they differ, as IGNITE's ``linear:`` mode."""
    diff = np.nonzero((target != before).any(1))[0]
    diff = diff[diff >= k0]
    if not len(diff):
        return target
    first = int(diff[0])
    n = max(1, int(round(ramp_s / mc.FRAME_S)))
    w = np.clip(np.arange(before.shape[0] - first, dtype=np.float32) / n, 0, 1)[:, None]
    out = before.copy()
    out[first:] = before[first:] + w * (target[first:] - before[first:])
    return out


def _cols(spec: str) -> List[int]:
    cols = mc.resolve(spec)
    for c in cols:
        if _T.start <= c < _T.stop or _V.start <= c < _V.stop:
            raise ValueError(
                f"{spec}: beam torque and voltage cannot be edited on their own; "
                "edit the beam power (pinj, beams or balance_torque) and they follow"
            )
        if _GR.start <= c < _GR.stop:
            raise ValueError(f"{spec}: edit gas_flow; the raw valve signal follows it")
    return sorted({a for c in cols for a in mc.alias_columns(c)})


def _scale_of(col: int) -> float:
    group = next(g for g in mc.GROUPS if mc.columns(g).start <= col < mc.columns(g).stop)
    return mc.DISPLAY[group][1]


def _after_edit(built: Built, before: np.ndarray, target: np.ndarray, cols: Sequence[int]):
    """Apply the couplings to channels an op edited directly."""
    beams = [c - _P.start for c in cols if _P.start <= c < _P.stop]
    if beams:
        _couple_beams(built, before, target, beams)
    valves = [c - _GF.start for c in cols if _GF.start <= c < _GF.stop]
    if valves:
        _couple_gas(built, before, target, valves)


def torque_per_mw(s: shots.Shot, beam: str, direction: Optional[str] = None) -> float:
    """Torque per MW of a source (N m / MW, signed: + co-current, - counter-current).

    Measured in this shot if the source ran. Otherwise the median over the era's shots
    in which it injected in ``direction`` ("co" or "counter"); without ``direction`` the
    era must have used the source in one direction only.
    """
    i = mc.BEAMS.index(beam)
    p = s.act[:, _P.start + i]
    on = p > 3e5
    if on.sum() >= 3:
        rate = float(np.median(s.act[on, _T.start + i] / p[on] * 1e6))
        if direction and (rate > 0) != (direction == "co"):
            raise ValueError(f"{beam} injected {'co' if rate > 0 else 'counter'}-current in "
                             f"{s.shot}; it cannot be run {direction}-current in this plan")
        return rate
    era = mc.era_limits(s.shot)
    lim = era["beams"][beam]
    if direction is None:
        seen = [d for d in ("co", "counter") if lim[f"n_shots_{d}"]]
        if len(seen) != 1:
            raise ValueError(
                f"{beam} ran {lim['n_shots_co']} shots co- and {lim['n_shots_counter']} "
                f"counter-current in shots {era['shots'][0]}-{era['shots'][1]}: give its "
                "direction ('co' or 'counter')")
        direction = seen[0]
    rate = lim.get(f"torque_per_MW_{direction}")
    if rate is None:
        raise ValueError(f"{beam} never injected {direction}-current in shots "
                         f"{era['shots'][0]}-{era['shots'][1]}")
    return float(rate)


def typical_kv(s: shots.Shot, beam: str) -> float:
    i = mc.BEAMS.index(beam)
    v = s.act[:, _V.start + i]
    on = s.act[:, _P.start + i] > 3e5
    if on.sum() >= 3:
        return float(np.median(v[on]) * 1e-3)
    return float(mc.era_limits(s.shot)["beams"][beam]["kv_median"])


def _couple_beams(built: Built, before, target, beams: Sequence[int], kv=None, rate=None,
                  direction=None):
    """Torque and voltage of edited beams follow their power, frame by frame.

    A source that was on keeps its measured torque per MW and voltage (torque scales with
    power); a source switched on gets the torque per MW of :func:`torque_per_mw` and the
    voltage ``kv`` (default: :func:`typical_kv`); a source switched off has zero torque
    and voltage.
    """
    s = built.shot
    for i in beams:
        name = mc.BEAMS[i]
        p0, p1 = before[:, _P.start + i], target[:, _P.start + i]
        t0, v0 = before[:, _T.start + i], before[:, _V.start + i]
        was_on = p0 > _BEAM_OFF_W
        changed = p1 != p0
        switched_on = (~was_on & (p1 > _BEAM_OFF_W) & changed).any()
        r = (rate or {}).get(name)
        if r is None:
            r = torque_per_mw(s, name, (direction or {}).get(name)) if switched_on else 0.0
        r = float(r)
        k = (kv or {}).get(name)
        k = (typical_kv(s, name) if switched_on else 0.0) if k is None else float(k)
        tq = np.where(was_on, t0 * np.divide(p1, p0, out=np.zeros_like(p1), where=was_on),
                      r * p1 * 1e-6)
        volt = np.where(was_on, v0, k * 1e3)
        on = p1 > _BEAM_OFF_W
        # Frames where the source was and stays off keep their measured (noise) values.
        stays_off = ~was_on & ~on
        edit = changed & ~stays_off
        target[edit, _T.start + i] = np.where(on, tq, 0.0)[edit]
        target[edit, _V.start + i] = np.where(on, volt, 0.0)[edit]
        if (~was_on & on & changed).any():
            built.notes.append(f"{name} switched on: {r:+.3f} N m/MW, {k:.0f} kV")


def _couple_gas(built: Built, before, target, valves: Sequence[int]):
    """The raw valve signal follows the flow, through a linear fit of raw on flow over the
    shot's measured frames (or in proportion when the fit is poor)."""
    s = built.shot
    for i in valves:
        f0, f1 = before[:, _GF.start + i], target[:, _GF.start + i]
        r0 = before[:, _GR.start + i]
        changed = f1 != f0
        mf, mr = s.act[:, _GF.start + i], s.act[:, _GR.start + i]
        ok = np.isfinite(mf) & np.isfinite(mr)
        fit = None
        if ok.sum() > 10 and np.std(mf[ok]) > 0:
            b, a = np.polyfit(mf[ok], mr[ok], 1)
            resid = mr[ok] - (a + b * mf[ok])
            if np.std(mr[ok]) > 0 and 1 - resid.var() / mr[ok].var() > 0.9:
                fit = (a, b)
        if fit is not None:
            target[changed, _GR.start + i] = (fit[0] + fit[1] * f1)[changed]
        else:
            ratio = np.divide(f1, f0, out=np.ones_like(f1), where=np.abs(f0) > 1e-6)
            target[changed, _GR.start + i] = (r0 * ratio)[changed]
            built.notes.append(f"{mc.CHANNEL_NAMES['gas_flow'][i]}: raw valve signal scaled with the flow "
                               "(no linear raw-flow relation in this shot)")


# --------------------------------------------------------------------------- ops
def _op_scale(built, target, k, op):
    cols = _cols(op["channels"])
    before = target.copy()
    target[k, cols] = target[k, cols] * float(op["factor"])
    _after_edit(built, before, target, cols)


def _op_set(built, target, k, op):
    cols = _cols(op["channels"])
    before = target.copy()
    for c in cols:
        target[k, c] = float(op["value"]) / _scale_of(c)
    _after_edit(built, before, target, cols)


def _op_waveform(built, target, k, op):
    cols = _cols(op["channels"])
    pts = np.asarray(op["points"], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2 or (np.diff(pts[:, 0]) <= 0).any():
        raise ValueError("waveform points must be [[t, value], ...] with increasing t")
    before = target.copy()
    times = built.shot.times
    inside = (times >= pts[0, 0] - 1e-9) & (times < pts[-1, 0] - 1e-9)
    # Frame means of the piecewise-linear waveform (midpoint of each 50 ms frame).
    values = np.interp(times + mc.FRAME_S / 2, pts[:, 0], pts[:, 1])
    for c in cols:
        target[inside, c] = values[inside] / _scale_of(c)
    _after_edit(built, before, target, cols)


def _op_shift(built, target, k, op):
    cols = _cols(op["channels"])
    n = int(round(float(op["dt"]) / mc.FRAME_S))
    before = target.copy()
    src = np.clip(np.arange(k.start, k.stop) - n, 0, target.shape[0] - 1)
    target[k.start:k.stop, cols] = before[src][:, cols]
    _after_edit(built, before, target, cols)


def _op_copy_from(built, target, k, op):
    donor = shots.load(int(op["shot"]))
    groups = op.get("groups") or list(mc.GROUPS)
    if any(g in groups for g in ("pinj", "beam_voltage", "tinj")):
        groups = sorted(set(groups) | {"pinj", "beam_voltage", "tinj"})
    if "gas_flow" in groups or "gas_raw" in groups:
        groups = sorted(set(groups) | {"gas_flow", "gas_raw"})
    if "rmp" in groups or "i_coil" in groups:
        groups = sorted(set(groups) | {"rmp", "i_coil"})
    if donor.n_frames < k.stop:
        raise ValueError(f"copy_from {donor.shot}: {donor.n_frames} frames < {k.stop}")
    for g in groups:
        cols = mc.columns(g)
        target[k, cols] = donor.act[k, cols]
    built.notes.append(f"copied {', '.join(groups)} from {donor.shot}")


def _op_beams(built, target, k, op):
    before = target.copy()
    edited = []
    for name in op.get("off", []):
        i = mc.BEAMS.index(name)
        target[k, _P.start + i] = 0.0
        edited.append(i)
    # A source given a power where it was off follows the sources this op leaves as they
    # are (all sources if it leaves none): "follow" (default) scales its power with their
    # total power relative to its median over the op window (capped at 1), so it ramps
    # down and stops with them; "beam_phase" holds it constant from their first to their
    # last frame on; "window" holds it over the whole op window.
    hold = op.get("hold", "follow")
    if hold not in ("follow", "beam_phase", "window"):
        raise ValueError(f"beams hold {hold!r}: use 'follow', 'beam_phase' or 'window'")
    touched = set(op.get("off", [])) | set(op.get("power_MW") or {})
    kept = [_P.start + i for i, b in enumerate(mc.BEAMS) if b not in touched]
    p_kept = (before[:, kept] * 1e-6).sum(1)
    if not (p_kept[k] > 0.3).any():
        p_kept = (built.shot.act[:, _P] * 1e-6).sum(1)
    on_meas = np.nonzero(p_kept > 0.3)[0]
    phase = np.zeros(target.shape[0], bool)
    if len(on_meas):
        phase[on_meas[0]:on_meas[-1] + 1] = True
    in_k = np.zeros(target.shape[0], bool)
    in_k[k] = True
    ref = np.median(p_kept[in_k & phase]) if (in_k & phase).any() else 0.0
    follow = np.clip(p_kept / ref, 0.0, 1.0) if ref > 0 else phase.astype(float)
    for name, mw in (op.get("power_MW") or {}).items():
        i = mc.BEAMS.index(name)
        rows = in_k.copy()
        was_off = before[:, _P.start + i] <= _BEAM_OFF_W
        value = np.full(target.shape[0], float(mw) * 1e6)
        if hold != "window":
            rows &= ~was_off | phase
            if hold == "follow":
                value = np.where(was_off, value * follow, value)
            if (in_k & was_off & ~phase).any():
                built.notes.append(
                    f"{name}: on only while the kept beams are on, until "
                    f"{mc.time_of(on_meas[-1] + 1):.2f} s"
                    + (", in proportion to their power" if hold == "follow" else ""))
        target[rows, _P.start + i] = value[rows]
        edited.append(i)
    _couple_beams(built, before, target, sorted(set(edited)), kv=op.get("kv"),
                  rate=op.get("torque_per_MW"), direction=op.get("direction"))


def _op_balance_torque(built, target, k, op):
    """Switch ``off`` sources off and size the ``counter`` sources so that, frame by frame,
    the net beam torque is ``target_Nm``; the other sources keep their recorded waveforms.

    Each counter source has its own counter-current torque per MW (:func:`torque_per_mw`).
    ``split`` "equal_power" (default) runs them at equal power; "fill" runs the source with
    the most torque per MW up to its limit first, the next one up to its limit, and so on
    (the last source takes the rest). ``limit_kind`` picks the limit: "db" (default, the
    most the source delivered in at least three shots of the database), "era" (the most
    in the reference shot's 2000-shot block) or "p95" (what 95 % of that block's shots
    reached). ``torque_per_MW`` (one
    number) overrides the rates; with equal power this is IGNITE's ``balance:`` mode. ``kv`` defaults to the sources' typical
    voltage.
    """
    off = [mc.BEAMS.index(b) for b in op.get("off", [])]
    counter = [mc.BEAMS.index(b) for b in op["counter"]]
    s = built.shot
    if op.get("torque_per_MW") is not None:
        rates = [-abs(float(op["torque_per_MW"]))] * len(counter)
    else:
        rates = [torque_per_mw(s, mc.BEAMS[i], "counter") for i in counter]
    if any(r >= 0 for r in rates):
        raise ValueError(f"counter sources {op['counter']} inject co-current here "
                         f"(torque per MW {rates}); they cannot cancel torque")
    kv = float(op.get("kv") or np.mean([typical_kv(s, mc.BEAMS[i]) for i in counter]))
    goal = float(op.get("target_Nm", 0.0))
    kept = [i for i in range(len(mc.BEAMS)) if i not in off and i not in counter]
    need = np.clip(target[k, _T][:, kept].sum(1) - goal, 0.0, None)
    for i in off:
        target[k, _P.start + i] = 0.0
        target[k, _V.start + i] = 0.0
        target[k, _T.start + i] = 0.0
    split = op.get("split", "equal_power")
    if split == "fill":
        # Largest torque per MW first, each source up to its limit; what is left goes to
        # the last source (validation flags it if too much).
        era = mc.era_limits(s.shot)
        key = _limit_key(op.get("limit_kind", "db"))
        order = sorted(range(len(counter)), key=lambda j: -abs(rates[j]))
        left = need.copy()
        power, torque = [None] * len(counter), [None] * len(counter)
        for n_done, j in enumerate(order):
            cap = era["beams"][mc.BEAMS[counter[j]]][key]
            p = left / abs(rates[j]) if n_done == len(order) - 1 else \
                np.minimum(left / abs(rates[j]), cap)
            power[j], torque[j] = p * 1e6, rates[j] * p
            left = left - abs(rates[j]) * p
    elif split != "equal_power":
        raise ValueError(f"split {split!r}: use 'equal_power' or 'fill'")
    elif len(set(rates)) == 1:
        # Equal power and equal torque coincide; same arithmetic as balance: in eval.
        share = need / len(counter)
        power = [share / abs(rates[0]) * 1e6] * len(counter)
        torque = [-share] * len(counter)
    else:
        p = need / sum(abs(r) for r in rates)
        power = [p * 1e6] * len(counter)
        torque = [r * p for r in rates]
    for j, i in enumerate(counter):
        target[k, _T.start + i] = torque[j]
        target[k, _P.start + i] = power[j]
        pw = np.asarray(power[j])
        target[k, _V.start + i] = kv * 1e3 * (pw > 0)
    built.notes.append("counter sources " + ", ".join(
        f"{mc.BEAMS[i]} {r:+.3f} N m/MW" for i, r in zip(counter, rates)) + f", {kv:.0f} kV")


_LIMIT_KEYS = {"db": "power_MW_db", "era": "power_MW_max", "p95": "power_MW_p95_of_shots"}


def _limit_key(limit_kind: str) -> str:
    if limit_kind not in _LIMIT_KEYS:
        raise ValueError(f"limit_kind {limit_kind!r}: use one of {sorted(_LIMIT_KEYS)}")
    return _LIMIT_KEYS[limit_kind]


_OPS = {"scale": _op_scale, "set": _op_set, "waveform": _op_waveform, "shift": _op_shift,
        "copy_from": _op_copy_from, "beams": _op_beams, "balance_torque": _op_balance_torque}


# --------------------------------------------------------------------------- beam options
def beam_options(shot: int, t: float, target_Nm: float = 0.0,
                 counter: Sequence[str] = (), keep_on: Sequence[str] = (),
                 t_end: Optional[float] = None, split: str = "fill", limit_kind: str = "db",
                 torque_per_MW: Optional[float] = None, kv: Optional[float] = None) -> List[dict]:
    """Every way to reach ``target_Nm`` by switching co sources off and sizing ``counter``.

    Enumerates each subset of the sources running at ``t`` (except ``counter`` and
    ``keep_on``) to switch off and builds the ``balance_torque`` op for it (``split`` as
    there; "fill" uses the counter sources' headroom best, up to the limit of
    ``limit_kind``). Reports each counter source's power (mean and maximum over
    ``window_s``, from ``t`` to the plasma end or ``t_end``), the planned and measured
    total beam power (means over the same window), and
    whether every counter source stays within its limit (``limit_kind``: "db", the most
    it delivered in any shot of the database; "era", the most in the reference shot's
    2000-shot block; "p95", what 95 % of that block's shots reached). For a set that is
    not feasible with ``keep_on`` at their measured power, it also gives the largest
    fraction of that power (beam modulation) at which it is, with the ops for it, and
    ``at_counter_limit``: the net torque left with ``keep_on`` at their measured power and
    the counter sources at their limit (the target is then missed by that much). Sorted:
    feasible sets by total beam power (highest first), then the others by that fraction
    (highest first).
    """
    s = shots.load(shot)
    k0 = mc.frame_of(t)
    k1 = mc.frame_of(t_end) if t_end else s.n_frames
    k1 = min(k1, plasma_end_frame(s))
    window = slice(k0, k1)
    running = [b for i, b in enumerate(mc.BEAMS)
               if (s.act[window, _P.start + i] > 3e5).mean() > 0.5]
    free = [b for b in running if b not in counter and b not in keep_on]
    era = mc.era_limits(shot)
    if not counter:
        raise ValueError("give the counter sources (`counter`)")
    key = _limit_key(limit_kind)
    limit = {c: era["beams"][c][key] for c in counter}
    p_meas = (s.act[window, _P] * 1e-6).sum(1).mean()
    window_s = [round(float(mc.time_of(k0)), 3), round(float(mc.time_of(k1)), 3)]

    def ops_for(off, fraction):
        bal = {"op": "balance_torque", "t": t, "off": list(off), "counter": list(counter),
               "target_Nm": target_Nm, "split": split, "limit_kind": limit_kind}
        if t_end:
            bal["t_end"] = t_end
        if torque_per_MW is not None:
            bal["torque_per_MW"] = torque_per_MW
        if kv is not None:
            bal["kv"] = kv
        ops = [bal]
        if fraction < 1 and keep_on:
            sc = {"op": "scale", "channels": "pinj:" + ",".join(keep_on), "factor": round(fraction, 3),
                  "t": t}
            if t_end:
                sc["t_end"] = t_end
            ops = [sc, bal]
        return ops

    def evaluate(off, fraction):
        b = build({"shot": shot, "ops": ops_for(off, fraction)})
        pc = {c: b.act[window, _P.start + mc.BEAMS.index(c)] * 1e-6 for c in counter}
        ok = all(pc[c].max() <= limit[c] * 1.0001 for c in counter)
        return b, pc, ok

    out = []
    for n in range(len(free) + 1):
        for off in itertools.combinations(free, n):
            b, pc, ok = evaluate(off, 1.0)
            row = {
                "off": list(off),
                "on": [x for x in running if x not in off] + list(counter),
                "counter_MW_mean": {c: round(float(v.mean()), 3) for c, v in pc.items()},
                "counter_MW_max": {c: round(float(v.max()), 3) for c, v in pc.items()},
                "counter_limit_MW": limit,
                # Means over window_s (t to the plasma end or t_end), the same window
                # for every number of the row.
                "window_s": window_s,
                "total_power_MW": round(float((b.act[window, _P] * 1e-6).sum(1).mean()), 3),
                "measured_power_MW": round(float(p_meas), 3),
                "net_torque_Nm": round(float(b.act[window, _T].sum(1).mean()), 3),
                "keeps": list(keep_on),
                "feasible": bool(ok),
                "ops": ops_for(off, 1.0),
            }
            if not ok and keep_on:
                lo, hi = 0.0, 1.0
                if evaluate(off, 0.0)[2]:
                    for _ in range(14):
                        mid = (lo + hi) / 2
                        lo, hi = (mid, hi) if evaluate(off, mid)[2] else (lo, mid)
                    f = float(np.floor(lo * 1000) / 1000)
                    bf, _, _ = evaluate(off, f)
                    row["keep_on_max_fraction"] = f
                    row["at_max_fraction"] = {
                        "total_power_MW": round(float((bf.act[window, _P] * 1e-6).sum(1).mean()), 3),
                        "ops": ops_for(off, f)}
                else:
                    row["keep_on_max_fraction"] = None
                lim_op = {"op": "beams", "t": t, "off": list(off),
                          "power_MW": {c: limit[c] for c in counter},
                          "direction": {c: "counter" for c in counter}}
                if t_end:
                    lim_op["t_end"] = t_end
                if torque_per_MW is not None:
                    lim_op["torque_per_MW"] = torque_per_MW
                if kv is not None:
                    lim_op["kv"] = kv
                bl = build({"shot": shot, "ops": [lim_op]})
                row["at_counter_limit"] = {
                    "net_torque_Nm": round(float(bl.act[window, _T].sum(1).mean()), 3),
                    "total_power_MW": round(float((bl.act[window, _P] * 1e-6).sum(1).mean()), 3),
                    "ops": [lim_op]}
            out.append(row)

    def rank(r):
        # Feasible sets by beam power; the others by how much of the kept beams' power
        # survives, then by beam power at that fraction.
        if r["feasible"]:
            return (0, -r["total_power_MW"], 0.0)
        f = r.get("keep_on_max_fraction")
        if f is None:
            return (2, 0.0, -r["total_power_MW"])
        return (1, -f, -r["at_max_fraction"]["total_power_MW"])

    out.sort(key=rank)
    return out


def save(plan: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=1))
    return path
