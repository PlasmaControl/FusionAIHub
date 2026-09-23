"""Write a simulation's ``simulation.h5``, ``metrics.json``, ``report.md`` and panels.

All four come from one `Ensemble`, its decoded features and the run's metadata, so every
number in the report can be recomputed from the h5 file.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import score
from .core import Ensemble
from .decode import feature_name

SCHEMA = "shot-design-simulation-metrics-v1"
# Okabe-Ito blue and vermillion; the measurement is black
_COLORS = {"real": "#0072B2", "proposed": "#D55E00"}
_REDUCTION = (
    "features/<m> are per-frame reductions of the codec's decoded output: spectro -> mean z "
    "over 10-60 kHz for mhr/mirnov and over the full band otherwise, video -> frame mean "
    "(F, 1), slowts/fastts -> mean over the intra-frame time axis. Arms are (M, F, C), the "
    "measurement (gt) is (F, C)."
)


def metrics(ens: Ensemble, feats: dict, families: dict[str, str], meta: dict) -> dict:
    """The ``metrics.json`` document."""
    mods: dict[str, dict] = {m: {"held": True} for m in ens.held}
    for m, per in feats.items():
        mods[m] = {
            "family": families[m],
            "feature": feature_name(m, families[m]),
            **score.modality_scores(per["gt"], {a: per[a] for a in ens.arms}, ens.k0),
        }
    return {
        "schema": SCHEMA,
        "members": int(meta["members"]),
        "k0": ens.k0,
        "n_predict": int(meta["n_predict"]),
        "frame_s": float(meta["frame_s"]),
        "t0_s": float(meta["t0_s"]),
        "decode_steps": int(meta["decode_steps"]),
        "temperature": float(meta["temperature"]),
        "held": sorted(ens.held),
        "modalities": dict(sorted(mods.items())),
    }


def write(
    out_dir: Path,
    ens: Ensemble,
    feats: dict,
    families: dict[str, str],
    meta: dict,
    actuators: dict[str, torch.Tensor] | None = None,
) -> dict:
    """Write the four outputs under ``out_dir`` and return the metrics document.

    ``feats`` is `decode.decode_ensemble`'s output, ``families`` maps each modality to
    its codec family, and ``meta`` must carry members, n_predict, frame_s, t0_s (the shot
    time of the first predicted frame), decode_steps and temperature.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = metrics(ens, feats, families, meta)
    with h5py.File(out_dir / "simulation.h5", "w") as f:
        for arm, tokens in {"gt": ens.gt, **ens.arms}.items():
            for m, t in tokens.items():
                f.create_dataset(f"tokens/{arm}/{m}", data=t.numpy().astype(np.int16))
        for m, per in feats.items():
            for label, arr in per.items():
                f.create_dataset(f"features/{m}/{label}", data=arr.astype(np.float32))
        for arm, a in (actuators or {}).items():
            f.create_dataset(f"actuators/{arm}", data=a.float().numpy())
        f.attrs.update(meta)
        f.attrs["held"] = sorted(ens.held)
        f.attrs["seeds"] = json.dumps(ens.seeds)
        f.attrs["batch"] = ens.batch
        f.attrs["reduction"] = _REDUCTION
    text = json.dumps(doc, indent=1, allow_nan=False)
    (out_dir / "metrics.json").write_text(text + "\n")
    for m, per in feats.items():
        _panel(out_dir / "panels" / f"{m}.png", m, doc, per)
    (out_dir / "report.md").write_text(_markdown(doc, meta))
    return doc


def _panel(path: Path, name: str, doc: dict, per: dict[str, np.ndarray]) -> None:
    """Shot time on x; each arm's ensemble mean and 10-90 % band; the measurement in
    black. A feature with several channels is drawn as its channel mean."""
    t = doc["t0_s"] + (np.arange(per["gt"].shape[0]) - doc["k0"]) * doc["frame_s"]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for arm, color in _COLORS.items():
        if arm in per:
            series = per[arm].mean(axis=-1)  # (M, F)
            lo, hi = np.percentile(series, [10, 90], axis=0)
            ax.fill_between(t, lo, hi, color=color, alpha=0.2, linewidth=0)
            ax.plot(t, series.mean(axis=0), color=color, linewidth=1.4, label=arm)
    ax.plot(t, per["gt"].mean(axis=-1), color="black", linewidth=1.0, label="measured")
    ax.axvline(doc["t0_s"], color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("shot time (s)")
    ax.set_ylabel("codec z")
    ax.set_title(f"{name}, {doc['modalities'][name]['feature']}", fontsize=10)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _markdown(doc: dict, meta: dict) -> str:
    scored = {m: e for m, e in doc["modalities"].items() if not e.get("held")}
    lines = [
        "# Simulation",
        "",
        (
            f"{doc['members']} rollouts per arm from {doc['k0']} measured frames, "
            f"{doc['n_predict']} frames predicted from {doc['t0_s']:.2f} s. IGNITE "
            f"{meta.get('codec_generation', '?')} step {meta.get('dynamics_step', '?')}, "
            f"{doc['decode_steps']} decode passes, temperature {doc['temperature']:g}."
        ),
        "",
        "| signal | skill vs persistence | spread / error | edit effect / noise |",
        "|---|---|---|---|",
    ]
    for m, e in scored.items():
        resolved = " (resolved)" if e.get("resolved") else ""
        lines.append(
            f"| {m}, {e['feature']} | {_fmt(e['skill'])} | {_fmt(e['spread_error'])} "
            f"| {_fmt(e.get('effect_to_noise'))}{resolved} |"
        )
    lines += [
        "",
        (
            "Skill is 1 - CRPS / CRPS(persistence): above 0 the rollouts beat holding the "
            "last measured frame. Spread / error is 1 for a calibrated ensemble, below 1 "
            "for an overconfident one. An edit is resolved when it moves the ensemble mean "
            f"at least {score.RESOLVED:g} times as far as rerunning the real actuators on "
            "fresh random numbers does."
        ),
    ]
    worse = [m for m, e in scored.items() if e["skill"] is not None and e["skill"] < 0]
    if worse:
        lines += ["", f"Worse than persistence: {', '.join(worse)}."]
    if doc["held"]:
        lines += ["", f"Absent from the seed, held at the placeholder: {', '.join(doc['held'])}."]
    return "\n".join(lines) + "\n"
