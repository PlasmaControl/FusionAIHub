"""Write ``simulation.h5``, per-modality panels, and the honest ``report.md``.

The three artifacts share one source of truth (`arms` + `decoded` + `meta`) so the
numbers in the table match the arrays a reader can pull out of the h5 file. The
report is deliberately "honest": IGNITE v4's dynamics checkpoint is an early one
(see `QUALITATIVE_SENTENCE`), so whenever the rollout does WORSE than the
do-nothing persistence baseline for any modality (`skill < 0`), the report says so
in plain language instead of letting a reader infer it from a table of numbers.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 (must follow matplotlib.use("Agg"))
import numpy as np
import torch

from .core import SimulationArms

QUALITATIVE_SENTENCE = (
    "IGNITE v4 dynamics (step {step}) is an early checkpoint; "
    "treat results as qualitative."
)

# Discloses the per-modality reductions `decode.decode_modalities` applies before a
# decoded array reaches this file -- most pressingly that video is NOT raw frames.
_REDUCTION_NOTE = (
    "decoded/<m> arrays are per-frame reductions of the codec's decoded output, "
    "not raw decoder output: spectro -> band-power mean|value| over frequency "
    "(10-60 kHz for mhr/mirnov, full band otherwise), video -> per-frame "
    "channel/space/time mean (F, 1) (raw frames are not stored), slowts/fastts -> "
    "mean over the intra-frame time axis (F, C)."
)

_ARM_COLORS = {"gt": "black", "real": "tab:blue", "proposed": "tab:orange"}


def frac_static(
    proposed: dict[str, torch.Tensor], seed_frames: int
) -> dict[str, float]:
    """Frame-to-frame staticness of the PROPOSED rollout's predicted region.

    The fraction of tokens at predicted frame f (``seed_frames <= f < F``)
    equal to the token at the SAME position in frame f-1. 1.0 = the rollout
    froze solid from the seed onward (every predicted frame repeats its
    predecessor); 0.0 = every predicted frame changed at every token
    position. Defined on ``proposed`` alone -- it describes what the
    design's own rollout did, not a comparison against ``real`` or ``gt``.
    """
    out: dict[str, float] = {}
    for m, codes in proposed.items():
        f_total = codes.shape[0]
        if seed_frames <= 0 or seed_frames >= f_total:
            out[m] = float("nan")
            continue
        pred = codes[seed_frames:]
        prev = codes[seed_frames - 1 : -1]
        out[m] = (pred == prev).float().mean().item()
    return out


def _to_numpy(x, dtype) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _write_panel(path: Path, per_arm: dict[str, np.ndarray], seed_frames: int) -> None:
    """Three-line (gt, real, proposed) plot of a decoded channel mean over time."""
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for arm_name in ("gt", "real", "proposed"):
        arr = per_arm.get(arm_name)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim > 1 and arr.shape[1] > 1:
            series = arr.mean(axis=1)
        else:
            series = arr.reshape(-1)
        ax.plot(series, label=arm_name, color=_ARM_COLORS.get(arm_name))
    ax.axvline(
        seed_frames - 0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="seed/predict boundary",
    )
    ax.set_xlabel("frame")
    ax.set_ylabel("decoded value (channel mean)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _render_report(arms: SimulationArms, frac: dict[str, float], meta: dict) -> str:
    lines = [
        "# Simulation report",
        "",
        "| modality | frac_static | token_acc | persistence_acc | skill "
        "| divergence_vs_real |",
        "|---|---|---|---|---|---|",
    ]
    any_negative_skill = False
    for m in sorted(arms.token_accuracy):
        token_acc = arms.token_accuracy[m]
        persistence_acc = arms.persistence_accuracy[m]
        skill = token_acc - persistence_acc
        any_negative_skill = any_negative_skill or skill < 0
        fs = frac.get(m, float("nan"))
        divergence = arms.divergence_vs_real.get(m, float("nan"))
        lines.append(
            f"| {m} | {fs:.2f} | {token_acc:.2f} | {persistence_acc:.2f} | "
            f"{skill:.2f} | {divergence:.2f} |"
        )
    lines.append("")
    if any_negative_skill:
        lines.append(QUALITATIVE_SENTENCE.format(step=meta.get("dynamics_step")))
    return "\n".join(lines) + "\n"


def write(
    out_dir: Path,
    arms: SimulationArms,
    decoded: dict[str, dict[str, np.ndarray]],
    meta: dict,
    actuators: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write ``simulation.h5``, ``panels/<m>.png`` and ``report.md`` under ``out_dir``.

    ``actuators``, when given, supplies the ``real``/``proposed`` actuator
    trajectories for the ``actuators/{real,proposed}`` h5 group (D3 has
    these; the brief's own report table test does not, hence the keyword
    defaulting to None and the group being skipped then). Returns
    ``out_dir`` -- the root all three artifacts are written under.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_dir / "simulation.h5", "w") as f:
        arms_by_label = (
            ("real", arms.real), ("proposed", arms.proposed), ("gt", arms.gt),
        )
        for arm_name, codes in arms_by_label:
            grp = f.create_group(f"tokens/{arm_name}")
            for m, t in codes.items():
                grp.create_dataset(m, data=_to_numpy(t, np.int64))
        for m, per_arm in decoded.items():
            grp = f.create_group(f"decoded/{m}")
            for arm_name, arr in per_arm.items():
                grp.create_dataset(arm_name, data=_to_numpy(arr, np.float32))
        if actuators is not None:
            grp = f.create_group("actuators")
            for arm_name in ("real", "proposed"):
                if arm_name in actuators:
                    data = _to_numpy(actuators[arm_name], np.float32)
                    grp.create_dataset(arm_name, data=data)
        for k, v in meta.items():
            f.attrs[k] = v
        f.attrs["reduction"] = _REDUCTION_NOTE

    frac = frac_static(arms.proposed, arms.seed_frames)

    panel_dir = out_dir / "panels"
    for m, per_arm in decoded.items():
        _write_panel(panel_dir / f"{m}.png", per_arm, arms.seed_frames)

    (out_dir / "report.md").write_text(_render_report(arms, frac, meta))
    return out_dir
