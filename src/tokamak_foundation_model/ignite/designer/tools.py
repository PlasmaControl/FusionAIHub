"""The designer's tools: one function per tool, JSON in and JSON out.

The CLI (:mod:`.__main__`), the MCP server (:mod:`.mcp_server`) and the notebook all call
these. :data:`TOOLS` holds each tool's description and input schema for agents.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import logbook as lb
from . import machine as mc
from . import plan as pl
from . import precedents as pr
from . import shots
from . import simulate as sim
from . import validate


def machine(shot: Optional[int] = None) -> dict:
    """How to work, IGNITE's actuator channels (with their signal paths) and what the
    database shows for the era of ``shot``."""
    out = {"knowledge": mc.knowledge(),
           "signals": {g: list(v) for g, v in mc.SIGNALS.items()},
           "frame_grid": {"t0_s": mc.T0, "dt_s": mc.FRAME_S, "seed_s": [mc.T0, mc.time_of(mc.K0)],
                          "prediction_s": [mc.time_of(mc.K0), mc.time_of(mc.N_FRAMES)]},
           "channels": {g: list(v) for g, v in mc.CHANNEL_NAMES.items()},
           "units": {g: mc.DISPLAY[g][0] for g in mc.GROUPS},
           "limits_method": mc.limits()["method"]}
    if shot is not None:
        out["era_limits"] = mc.era_limits(int(shot))
    return out


def describe_shot(shot: int, t0: float = 2.0, t1: Optional[float] = None) -> dict:
    out = shots.describe(int(shot), t0, t1)
    try:
        s = pr.series(int(shot))
        w = (s["t_s"] >= t0) & (s["t_s"] < (t1 or 99))
        out["plasma_means"] = {k: (round(float(np.nanmean(s[k][w])), 3)
                                   if np.isfinite(s[k][w]).any() else None)
                               for k in pr.SERIES}
        step = 5
        out["plasma_waveforms_0p25s"] = {
            k: [None if not np.isfinite(v) else round(float(v), 2)
                for v in np.nanmean(_pad(s[k][w], step).reshape(-1, step), axis=1)]
            for k in ("rot_core_kms", "ti_core_keV", "te_core_keV", "ne_core_1e19")}
    except KeyError:
        pass
    return out


def _pad(x, step):
    n = (-len(x)) % step
    return np.concatenate([x, np.full(n, np.nan, np.float32)])


def search_precedents(conditions=(), window=(2.0, 6.0), min_duration_s: float = 0.5,
                      near: Optional[int] = None, exclude=(), limit: int = 20) -> dict:
    with np.errstate(all="ignore"):
        return pr.search(conditions, tuple(window), min_duration_s, near, exclude, limit)


def beam_options(shot: int, t: float, counter, target_Nm: float = 0.0, keep_on=(),
                 t_end: Optional[float] = None, split: str = "fill",
                 limit_kind: str = "db", limit: int = 12) -> dict:
    opts = pl.beam_options(int(shot), float(t), float(target_Nm), tuple(counter),
                           tuple(keep_on), t_end, split, limit_kind)
    return {"n_options": len(opts), "n_feasible": sum(o["feasible"] for o in opts),
            "options": opts[:limit]}


def check_plan(plan: dict) -> dict:
    """Build and validate a plan; returns the findings and measured-vs-plan summary."""
    b = pl.build(plan)
    findings = validate.check(b)
    return {"ok": validate.ok(findings),
            "within_era": not any(f["check"] == "above_era" for f in findings),
            "findings": findings, "beam_margins": validate.margins(b),
            "summary": b.summary()}


def export_plan(plan: dict, name: Optional[str] = None,
                seeds=(1, 2, 3, 4)) -> dict:
    """Write the plan's IGNITE input to a run directory and return the rollout command.
    Does not submit anything."""
    return sim.run(plan, name=name, seeds=tuple(seeds), go=False)


def simulate(run_dir: str, seeds=(1, 2, 3, 4), confirm: bool = False) -> dict:
    """Submit the IGNITE rollouts of an exported plan (one GPU job). Needs
    ``confirm=true``: rollouts cost GPU time and the user must have approved them."""
    if not confirm:
        return {"submitted": False, "command": shlex.join(sim.command(run_dir, tuple(seeds))),
                "message": "not submitted: pass confirm=true once the user approved it"}
    return {"submitted": True, "job": sim.submit(run_dir, tuple(seeds)), "run_dir": run_dir}


def simulation_status(run_dir: str) -> dict:
    return sim.status(run_dir)


def simulation_results(run_dir: str) -> dict:
    """Core plasma quantities after the first change: measured, replay and plan (seed
    means), and the plan-minus-replay change paired by seed."""
    r = sim.results(run_dir)
    plan = json.loads((Path(run_dir) / "plan.json").read_text())
    b = pl.build(plan)
    first = b.first_change_s() or mc.time_of(r["k0"])
    t = r["t_s"]
    s = shots.load(int(plan["shot"]))
    end = min(t[-1] + mc.FRAME_S, s.t_end or 99)
    w = (t >= first + 0.5) & (t < end)  # from 0.5 s after the change to the plasma end
    out = {"window_s": [round(float(first + 0.5), 2), round(float(end), 2)], "seeds": r["seeds"],
           "quantities": {}}
    for q, meas in r["measured"].items():
        rep, pln = r["replay"].get(q), r["plan"].get(q)
        if rep is None or pln is None:
            continue
        with np.errstate(all="ignore"):
            d = np.nanmean(pln[:, w], 1) - np.nanmean(rep[:, w], 1)
            out["quantities"][q] = {
                "measured": round(float(np.nanmean(meas[w])), 3),
                "replay": round(float(np.nanmean(rep[:, w])), 3),
                "plan": round(float(np.nanmean(pln[:, w])), 3),
                "plan_minus_replay_per_seed": [round(float(x), 3) for x in d],
            }
    return out


def logbook(shot: int, max_chars: int = 6000) -> dict:
    """What the DIII-D logbook says about one shot and its run."""
    return lb.shot(int(shot), max_chars=int(max_chars))


def search_logbook(query: str, conditions=(), window=(2.0, 6.0), min_duration_s: float = 0.5,
                   shot_range=None, tokenized_only: bool = True, limit: int = 20) -> dict:
    """Logbook full-text search, optionally restricted to shots that also meet measured
    ``conditions`` (as in search_precedents)."""
    keep = None
    if conditions:
        res = pr.search(list(conditions), tuple(window), float(min_duration_s), limit=100000)
        keep = [m["shot"] for m in res["matches"]]
    out = lb.search(query, shots=keep, shot_range=shot_range, tokenized_only=tokenized_only,
                    limit=int(limit))
    if conditions:
        out["conditions"] = list(conditions)
        out["n_measured_matches"] = len(keep)
    return out


def search_logbook_runs(query: str, limit: int = 10) -> dict:
    """Experiments (runs) whose title or run summary match ``query``."""
    return lb.search_runs(query, limit=int(limit))


TOOLS: Dict[str, Dict[str, Any]] = {
    "machine": {
        "fn": machine,
        "description": "Read first. How to work; IGNITE's actuator channels with their "
                       "MDSplus/PTDATA signal paths, the frame grid, and what the database "
                       "shows for the era of a shot (per channel: whether it ran, its maxima, "
                       "torque per MW by direction; the most it delivered in any shot).",
        "schema": {"type": "object", "properties": {"shot": {"type": "integer"}}},
    },
    "describe_shot": {
        "fn": describe_shot,
        "description": "Measured actuators of a tokenized shot (per beam source, gyrotron, "
                       "gas valve and coil, with each beam's torque per MW and direction) and "
                       "its core plasma quantities, over [t0, t1) s.",
        "schema": {"type": "object", "required": ["shot"], "properties": {
            "shot": {"type": "integer"}, "t0": {"type": "number"}, "t1": {"type": "number"}}},
    },
    "search_precedents": {
        "fn": search_precedents,
        "description": "Measured shots (all tokenized shots) in which every condition "
                       "'<series> <op> <value>' holds for min_duration_s inside window. Series: "
                       + ", ".join(pr.SERIES) + "; scalars: shot, ip_peak_MA, t_end. "
                       "'near' ranks by similarity to a shot.",
        "schema": {"type": "object", "properties": {
            "conditions": {"type": "array", "items": {"type": "string"}},
            "window": {"type": "array", "items": {"type": "number"}},
            "min_duration_s": {"type": "number"}, "near": {"type": "integer"},
            "exclude": {"type": "array", "items": {"type": "integer"}},
            "limit": {"type": "integer"}}},
    },
    "logbook": {
        "fn": logbook,
        "description": "What the DIII-D logbook holds on one shot: run and experiment title, "
                       "mini-proposal, configuration, pre-shot comment, every log entry by role "
                       "(session leader, physics operator, RF, PCS changes, diagnostics), the "
                       "summary of its run and the run's shot numbers. Use it to read what a "
                       "shot (reference or precedent) was meant to do and what happened.",
        "schema": {"type": "object", "required": ["shot"], "properties": {
            "shot": {"type": "integer"}, "max_chars": {"type": "integer"}}},
    },
    "search_logbook": {
        "fn": search_logbook,
        "description": "Full-text search (BM25) over the logbook of 53,000 shots: run titles, "
                       "pre-shot comments and log entries; returns shots with a snippet. Words "
                       "are ANDed, \"quoted phrases\" stay phrases, word* is a prefix, OR/NEAR "
                       "pass through. Operators write the same thing many ways: try variants "
                       "with OR. Optional 'conditions' "
                       "(as in search_precedents) keep only shots that also meet them; "
                       "tokenized_only (default true) keeps shots IGNITE can start from.",
        "schema": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string"},
            "conditions": {"type": "array", "items": {"type": "string"}},
            "window": {"type": "array", "items": {"type": "number"}},
            "min_duration_s": {"type": "number"},
            "shot_range": {"type": "array", "items": {"type": "integer"}},
            "tokenized_only": {"type": "boolean"}, "limit": {"type": "integer"}}},
    },
    "search_logbook_runs": {
        "fn": search_logbook_runs,
        "description": "Experiments (runs) whose title or run summary match a query, with "
                       "their shot range and how many of their shots are tokenized.",
        "schema": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}}},
    },
    "beam_options": {
        "fn": beam_options,
        "description": "Every beam set that reaches target_Nm net torque from time t by "
                       "switching running sources off and sizing the counter sources, with "
                       "the counter power needed, total power and whether all sources stay "
                       "within what they delivered in the database; for infeasible sets, the largest "
                       "fraction of the keep_on beams' power (modulation) that makes them "
                       "feasible, with ready ops, and the net torque left with keep_on at full "
                       "power and the counter sources at their limit (at_counter_limit). "
                       "counter (required) are the sources that inject counter-current; "
                       "keep_on (default none) are sources never switched off; split "
                       "'fill' (default) or 'equal_power'; limit_kind 'db' (default: the "
                       "most the source delivered in any shot of the database), 'era' (the "
                       "most in the reference shot's 2000-shot block) or 'p95' (what 95 % "
                       "of that block's shots reached).",
        "schema": {"type": "object", "required": ["shot", "t", "counter"], "properties": {
            "shot": {"type": "integer"}, "t": {"type": "number"},
            "target_Nm": {"type": "number"},
            "counter": {"type": "array", "items": {"type": "string"}},
            "keep_on": {"type": "array", "items": {"type": "string"}},
            "t_end": {"type": "number"}, "split": {"type": "string"},
            "limit_kind": {"type": "string"},
            "limit": {"type": "integer"}}},
    },
    "check_plan": {
        "fn": check_plan,
        "description": "Build a plan {shot, goal, rationale, ops:[...]} and validate it "
                       "against the database. ok: no errors (every channel within what "
                       "DIII-D delivered in the database); within_era: also within the "
                       "reference shot's 2000-shot block; beam_margins: each changed "
                       "source's planned maximum against both. Op reference: see the "
                       "'plan_ops' section of the machine knowledge.",
        "schema": {"type": "object", "required": ["plan"],
                   "properties": {"plan": {"type": "object"}}},
    },
    "export_plan": {
        "fn": export_plan,
        "description": "Write a valid plan's IGNITE input to a run directory and return the "
                       "rollout command. Submits nothing.",
        "schema": {"type": "object", "required": ["plan"], "properties": {
            "plan": {"type": "object"}, "name": {"type": "string"},
            "seeds": {"type": "array", "items": {"type": "integer"}}}},
    },
    "simulate": {
        "fn": simulate,
        "description": "Submit IGNITE rollouts (plan and replay, paired by seed) of an "
                       "exported run directory. Only with confirm=true after the user "
                       "approved GPU use.",
        "schema": {"type": "object", "required": ["run_dir"], "properties": {
            "run_dir": {"type": "string"},
            "seeds": {"type": "array", "items": {"type": "integer"}},
            "confirm": {"type": "boolean"}}},
    },
    "simulation_status": {
        "fn": simulation_status,
        "description": "Job state and finished seeds of a run directory.",
        "schema": {"type": "object", "required": ["run_dir"],
                   "properties": {"run_dir": {"type": "string"}}},
    },
    "simulation_results": {
        "fn": simulation_results,
        "description": "Core rotation, T_i, T_e and n_e from 0.5 s after the change to the "
                       "plasma end: measured, replay, plan, and plan minus replay per seed.",
        "schema": {"type": "object", "required": ["run_dir"],
                   "properties": {"run_dir": {"type": "string"}}},
    },
}


def call(name: str, args: Optional[dict] = None) -> Any:
    if name not in TOOLS:
        raise KeyError(f"unknown tool {name!r}; tools: {sorted(TOOLS)}")
    return TOOLS[name]["fn"](**(args or {}))


def to_json(x) -> str:
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    return json.dumps(x, indent=1, default=default)
