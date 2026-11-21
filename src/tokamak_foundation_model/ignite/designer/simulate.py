"""Export a built plan to IGNITE, submit the rollouts and read the results back.

A run directory holds

``plan.json``     the plan as written by the agent
``plan.npz``      ``act_norm`` (IGNITE's input, every cache frame), ``act`` (physical),
                  ``measured`` (physical) and ``changed``; the eval reads it through
                  ``--actuator_mode file:<run>/plan.npz``
``act_global/``   links to the shot's actuator entry and the training normalisation
``seed<N>/``      one IGNITE rollout per seed. Each also holds the same seed's rollout
                  under the measured actuators (``real__`` keys), so plan and replay
                  differ only by the actuators.

Submission goes through ``scripts/slurm_stellarai/run_shot.sh`` (one GPU, one job for
all seeds). Nothing is submitted unless :func:`submit` is called.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from . import machine as mc
from . import validate
from .plan import Built, build

REPO = Path(__file__).resolve().parents[4]
RUN_SHOT = REPO / "scripts" / "slurm_stellarai" / "run_shot.sh"
RUNS_DIR = Path(os.environ.get("IGNITE_DESIGNER_RUNS", mc.MODELS / "designer_runs"))
CKPT = mc.MODELS / "ignite_prod_v4" / "runs" / "soup_lu18_h16_30" / "dynamics_step30000.pt"
STEP = 30000
PLASMA = ("cer_rot", "cer_ti", "ts_core_temp", "ts_core_density")


def export(b: Built, run_dir) -> Path:
    """Write the plan and IGNITE's input for it; refuses a plan with validation errors."""
    findings = validate.check(b)
    if not validate.ok(findings):
        raise ValueError("plan has validation errors: " + "; ".join(
            f["message"] for f in findings if f["level"] == "error"))
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "plan.json").write_text(json.dumps(b.plan, indent=1))
    (run / "validation.json").write_text(json.dumps(findings, indent=1))
    np.savez_compressed(run / "plan.npz", act_norm=b.normalised().astype(np.float32),
                        act=b.act, measured=b.measured, changed=b.changed,
                        shot=b.shot.shot, t0=mc.T0, frame_s=mc.FRAME_S)
    links = run / "act_global"
    links.mkdir(exist_ok=True)
    for name, target in ((f"{b.shot.shot}.pt", b.shot.token_set.act_global / f"{b.shot.shot}.pt"),
                         ("act_global_stats.npz", mc.ACT_STATS)):
        link = links / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    return run


def command(run_dir, seeds: Sequence[int] = (1, 2, 3, 4), ckpt=CKPT,
            account: str = "kolemen", time: str = "02:00:00") -> list:
    """The sbatch command that rolls out the plan (and the replay) for ``seeds``."""
    run = Path(run_dir).resolve()
    plan = json.loads((run / "plan.json").read_text())
    shot = int(plan["shot"])
    ts = next(t for t in mc.TOKEN_SETS if (t.frame_codes / f"{shot}.pt").exists())
    env = ",".join([
        "ALL", f"SHOT={shot}", f"SCENARIO=file:{run}/plan.npz",
        f"SEEDS={' '.join(str(s) for s in seeds)}", f"OUT_DIR={run}",
        f"CACHE_DIR={ts.frame_codes}", f"ACT_GLOBAL={run}/act_global", f"CKPT={ckpt}",
    ])
    return ["sbatch", f"--account={account}", f"--time={time}", "--job-name=ignite_design",
            f"--output={run}/slurm_%j.out", f"--export={env}", str(RUN_SHOT)]


def submit(run_dir, seeds: Sequence[int] = (1, 2, 3, 4), **kw) -> str:
    """Submit the rollouts; returns the job id (also written to ``jobs.json``)."""
    cmd = command(run_dir, seeds, **kw)
    # sbatch --export splits on commas; SEEDS holds spaces only, paths hold none.
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    job = out.strip().split()[-1]
    jobs = Path(run_dir) / "jobs.json"
    previous = json.loads(jobs.read_text()) if jobs.exists() else []
    jobs.write_text(json.dumps(previous + [{"job": job, "seeds": list(seeds), "cmd": cmd}],
                               indent=1))
    return job


def status(run_dir) -> dict:
    """Job state and which seeds have finished."""
    run = Path(run_dir)
    jobs = json.loads((run / "jobs.json").read_text()) if (run / "jobs.json").exists() else []
    states = {}
    for j in jobs:
        q = subprocess.run(["sacct", "-n", "-X", "-j", j["job"], "-o", "State"],
                           capture_output=True, text=True).stdout.split()
        states[j["job"]] = q[0] if q else "UNKNOWN"
    plan = json.loads((run / "plan.json").read_text())
    done = sorted(int(p.parent.name[4:]) for p in run.glob(f"seed*/codes_{plan['shot']}_step*.npz"))
    return {"jobs": states, "seeds_done": done}


def results(run_dir, device: str = "cpu") -> Dict:
    """Decode the core plasma quantities of every finished seed: measured, replay
    (measured actuators) and plan, in physical units, frame by frame.

    Returns ``{"t_s", "k0", "measured": {q: (F,)}, "replay": {q: (S, F)}, "plan": {q: (S, F)},
    "seeds", "act_measured", "act_plan"}`` with q in ``rot_core_kms``, ``ti_core_keV``,
    ``te_core_keV``, ``ne_core_1e19`` (the reductions of :mod:`.index`).
    """
    import torch

    from .. import eval_dynamics as ev
    from .. import train_dynamics as td
    from .index import CODEC_TMPL, CER_CORE

    run = Path(run_dir)
    plan = json.loads((run / "plan.json").read_text())
    shot = int(plan["shot"])
    archives = sorted(run.glob(f"seed*/codes_{shot}_step*.npz"))
    if not archives:
        raise FileNotFoundError(f"no finished rollout in {run}")
    codecs = td.load_frozen_codecs(list(PLASMA), repo=REPO, tmpl=CODEC_TMPL)
    for c in codecs.values():
        c[0].to(device)
    conv = ev.profile_converters(codecs, PLASMA)
    names = {"cer_rot": ("rot_core_kms", 1.0), "cer_ti": ("ti_core_keV", 1e-3),
             "ts_core_temp": ("te_core_keV", 1e-3), "ts_core_density": ("ne_core_1e19", 1e-19)}

    def core(name, x):
        with np.errstate(all="ignore"):
            prof = np.nanmean(conv[name](x), axis=-1)
            if name.startswith("cer"):
                v = np.nanpercentile(np.abs(prof[:, list(CER_CORE)]), 90, axis=1)
            elif name == "ts_core_temp":
                v = np.nanpercentile(prof, 90, axis=1)
            else:
                v = np.nanpercentile(prof, 75, axis=1)
        return v * names[name][1]

    out = {"measured": {}, "replay": {}, "plan": {}, "seeds": []}
    for path in archives:
        z = np.load(path, allow_pickle=True)
        F, K0 = int(z["F"]), int(z["K0"])
        out.update(k0=K0, t_s=mc.time_of(np.arange(F)))
        out["seeds"].append(int(path.parent.name[4:]))
        for name in PLASMA:
            key = names[name][0]
            codec = codecs[name][0]
            gt = z[f"gt__{name}"][:F]
            if (gt == gt[:1]).all():
                continue  # not recorded in this shot
            if key not in out["measured"]:
                out["measured"][key] = core(name, ev.decode_flat_chunked(
                    codec, ev.archived(z, f"gt__{name}")[:F], torch.device(device)))
            for arm, prefix in (("plan", "pred"), ("replay", "real")):
                if f"{prefix}__{name}" in z.files:
                    x = ev.decode_flat_chunked(codec, ev.archived(z, f"{prefix}__{name}")[:F],
                                               torch.device(device))
                    out[arm].setdefault(key, []).append(core(name, x))
    for arm in ("plan", "replay"):
        out[arm] = {k: np.stack(v) for k, v in out[arm].items()}
    p = np.load(run / "plan.npz")
    out["act_measured"], out["act_plan"] = p["measured"][:F], p["act"][:F]
    return out


def run(plan: dict, name: Optional[str] = None, seeds=(1, 2, 3, 4), go: bool = False) -> dict:
    """Build, validate and export a plan; submit only when ``go`` is True."""
    b = build(plan)
    digest = hashlib.sha1(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:8]
    run_dir = RUNS_DIR / (name or f"{plan['shot']}_{digest}")
    export(b, run_dir)
    out = {"run_dir": str(run_dir), "command": shlex.join(command(run_dir, seeds))}
    if go:
        out["job"] = submit(run_dir, seeds)
    return out
