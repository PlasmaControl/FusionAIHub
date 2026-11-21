"""Checks of a built plan against the database and IGNITE's input contract.

Each finding is ``{"level": "error" | "warning" | "info", "check": ..., "message": ...}``.
A plan with an error must not be simulated. Feasible means within what the database shows
DIII-D has run (:func:`machine.database_limits`): a channel above the most it delivered in
at least three tokenized shots (2 % tolerance), a beam, gyrotron or valve switched on that ran in no
shot of the reference shot's 2000-shot block, or a total (beam power, counter-current
beam power, ECH power, |net beam torque|) above the largest any shot reached, is an
error. Above the block's own maximum is information (check ``above_era``; see
:func:`margins`). A net beam torque of the opposite sign to the measured one is a
warning. No limit is written in the code.
"""

from __future__ import annotations

from typing import List

import numpy as np

from . import machine as mc
from .plan import Built

_TOL = 1.02
# Measured traces dip slightly below zero around switch-offs; only planned values clearly
# below zero are errors (MW, MW, Torr L/s).
_NEG = {"pinj": 0.01, "ech_power": 0.005, "gas_flow": 0.05}


def check(b: Built) -> List[dict]:
    s = b.shot
    out: List[dict] = []

    def add(level, check, message):
        out.append({"level": level, "check": check, "message": message})

    changed = b.changed
    rows = np.nonzero(changed.any(1))[0]
    if not len(rows):
        add("info", "no_change", "the plan equals the measured actuators")
        return out
    first, last = int(rows[0]), int(rows[-1])
    if first < mc.K0:
        add("error", "seed", f"the plan changes the actuators at {mc.time_of(first):.2f} s, "
            f"inside IGNITE's measured initial state (1-{mc.time_of(mc.K0):.1f} s)")
    if first >= mc.N_FRAMES:
        add("error", "window", f"the first change ({mc.time_of(first):.2f} s) is after "
            f"IGNITE's prediction window ends ({mc.time_of(mc.N_FRAMES):.1f} s)")
    elif last >= mc.N_FRAMES:
        add("info", "window", f"changes after {mc.time_of(mc.N_FRAMES):.1f} s are not simulated")
    if s.t_end and mc.time_of(first) >= s.t_end:
        add("warning", "plasma_end", f"the first change is after the plasma ended ({s.t_end:.2f} s)")

    era = mc.era_limits(s.shot)
    tag = f"shots {era['shots'][0]}-{era['shots'][1]}"
    a, m = b.act, b.measured
    P, V, T = mc.columns("pinj"), mc.columns("beam_voltage"), mc.columns("tinj")

    def db_text(d):
        return f"{d['max_db']:.2f} in shots {', '.join(map(str, d['max_db_shots']))}"

    for i, name in enumerate(mc.BEAMS):
        cols = [P.start + i, V.start + i, T.start + i]
        if not changed[:, cols].any():
            continue
        lim = era["beams"][name]
        p = a[:, P.start + i] * 1e-6
        edited = changed[:, P.start + i]
        if (p[edited] < -_NEG["pinj"]).any():
            add("error", "beam_power", f"{name} power below zero")
        top = p[edited].max() if edited.any() else 0.0
        if top > lim["power_MW_db"] * _TOL:
            add("error", "beam_power", f"{name} at {top:.2f} MW is above the most it delivered "
                f"in at least {mc.DB_MIN_SHOTS} shots of the database ({lim['power_MW_db']:.2f} MW in "
                f"shots {', '.join(map(str, lim['power_MW_db_shots']))})")
        elif top > lim["power_MW_max"] * _TOL:
            add("info", "above_era", f"{name} at {top:.2f} MW is above the most it delivered "
                f"in {tag} ({lim['power_MW_max']:.2f} MW) but within the database "
                f"({lim['power_MW_db']:.2f} MW)")
        on_now = p > 0.05
        on_before = m[:, P.start + i] * 1e-6 > 0.05
        if (on_now & ~on_before).any() and not lim["n_shots_used"]:
            add("error", "beam_available", f"{name} ran in no shot of {tag}")
        planned = on_now & changed[:, cols].any(1)
        kv = a[planned, V.start + i] * 1e-3
        if len(kv) and lim["kv_db"] and kv.max() > lim["kv_db"] * _TOL:
            add("error", "beam_voltage", f"{name} at {kv.max():.0f} kV is above the highest "
                f"voltage it ran at in the database ({lim['kv_db']:.0f} kV)")
    E = mc.columns("ech_power")
    for i, name in enumerate(mc.CHANNEL_NAMES["ech_power"]):
        c = E.start + i
        if not changed[:, c].any():
            continue
        lim = era["ech_power"][name]
        p = a[:, c] * 1e-6
        if (p[changed[:, c]] < -_NEG["ech_power"]).any():
            add("error", "ech_power", f"gyrotron {name} power below zero")
        if p.max() > lim["max_db"] * _TOL:
            add("error", "ech_power", f"gyrotron {name} at {p.max():.2f} MW is above the most "
                f"it delivered in at least {mc.DB_MIN_SHOTS} shots of the database "
                f"({db_text(lim)})")
        elif p.max() > lim["max"] * _TOL:
            add("info", "above_era", f"gyrotron {name} at {p.max():.2f} MW is above the most "
                f"it delivered in {tag} ({lim['max']:.2f} MW) but within the database")
        if (p > 0.02).any() and not lim["n_shots_used"]:
            add("error", "ech_available", f"gyrotron {name} ran in no shot of {tag}")
    for group, unit, scale in (("gas_flow", "Torr L/s", 1.0), ("rmp", "kA", 1e-3),
                               ("i_coil", "kA", 1e-3)):
        C = mc.columns(group)
        for i, name in enumerate(mc.CHANNEL_NAMES[group]):
            c = C.start + i
            if not changed[:, c].any():
                continue
            v = a[:, c] * scale
            lim = era[group][name]
            if group == "gas_flow" and (v[changed[:, c]] < -_NEG["gas_flow"]).any():
                add("error", group, f"{name} flow below zero")
            if np.abs(v).max() > lim["max_db"] * _TOL:
                add("error", group, f"{group} {name} at {np.abs(v).max():.2f} {unit} is above "
                    f"the most it delivered in at least {mc.DB_MIN_SHOTS} shots of the database "
                    f"({db_text(lim)})")
            elif np.abs(v).max() > lim["max"] * _TOL:
                add("info", "above_era", f"{group} {name} at {np.abs(v).max():.2f} {unit} is above "
                    f"{tag}'s maximum ({lim['max']:.2f}) but within the database")
            if (np.abs(v) > 0.1 * max(lim["max_db"], 1e-9)).any() and not lim["n_shots_used"] \
                    and (changed[:, c] & (np.abs(m[:, c] * scale) < 1e-9)).any():
                add("error", f"{group}_available", f"{name} ran in no shot of {tag}")
    # Totals per frame against the largest any shot reached (each shot's 99th percentile).
    w = slice(mc.K0, min(mc.N_FRAMES, len(a)))
    tq = a[w, T]
    totals = {"pnbi_MW": (a[w, P] * 1e-6).sum(1),
              "counter_MW": np.where(tq < -0.05, a[w, P] * 1e-6, 0.0).sum(1),
              "ech_MW": (a[w, E] * 1e-6).sum(1),
              "torque_Nm": np.abs(tq.sum(1))}
    rows_w = changed[w].any(1)
    for key, x in totals.items():
        db = era["totals_db"][key]
        x = x[rows_w]
        if len(x) and x.max() > db["max_db"] * _TOL:
            add("error", "total", f"{key} reaches {x.max():.2f}, above the most at least "
                f"{mc.DB_MIN_SHOTS} shots in the database reached ({db_text(db)})")
    # Net beam torque reversed against the measurement in changed frames.
    tq_plan, tq_meas = a[w, T].sum(1), m[w, T].sum(1)
    flip = rows_w & (np.sign(tq_plan) != np.sign(tq_meas)) & (np.abs(tq_plan) > 0.1) \
        & (np.abs(tq_meas) > 0.1)
    if flip.any():
        j = np.nonzero(flip)[0]
        add("warning", "torque_reversal", f"the net beam torque has the opposite sign to the "
            f"measured one in {len(j)} frames ({mc.time_of(mc.K0 + j[0]):.2f}-"
            f"{mc.time_of(mc.K0 + j[-1] + 1):.2f} s, down to "
            f"{tq_plan[j].min() if tq_meas[j[0]] > 0 else tq_plan[j].max():+.2f} N m)")
    for note in b.notes:
        add("info", "build", note)
    return out


def ok(findings: List[dict]) -> bool:
    return not any(f["level"] == "error" for f in findings)


def margins(b: Built) -> dict:
    """For every changed beam source: its planned maximum against the most it delivered
    in the reference shot's block (era) and in the database (MW)."""
    era = mc.era_limits(b.shot.shot)
    P = mc.columns("pinj")
    out = {}
    for i, name in enumerate(mc.BEAMS):
        edited = b.changed[:, P.start + i]
        if not edited.any():
            continue
        top = float(b.act[edited, P.start + i].max() * 1e-6)
        lim = era["beams"][name]
        out[name] = {"plan_max_MW": round(top, 3),
                     "era_max_MW": round(float(lim["power_MW_max"]), 3),
                     "db_max_MW": round(float(lim["power_MW_db"]), 3),
                     "within_era": top <= lim["power_MW_max"] * _TOL,
                     "within_db": top <= lim["power_MW_db"] * _TOL}
    return out
