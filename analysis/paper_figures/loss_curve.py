"""Loss curve of the 48-layer run for the paper, on one fixed set of windows.

The run's own ``loss_history.jsonl`` cannot be plotted as one curve: its validation
windows changed twice (step 4000: plasma-end table and re-encoded CO2; step 5100:
201081/201085 moved between sets), and its training loss is one batch every 50 steps.
``score`` re-scores every saved snapshot (``dynamics_step<N>.pt``) on the same 256
windows from the discharges the run trains on and 256 from the others, with the
training run's own validation code (``Trainer._validate``: masked cross-entropy at the
run's mask schedule, and next-frame cross-entropy from a real 20-frame history), and
writes one JSON line per snapshot. Forward passes only; no sampled rollouts. ``plot``
draws the per-token curves in the paper style.

Score (one GPU, ~10 GB per checkpoint)::

    python analysis/paper_figures/loss_curve.py score
Plot (login node)::

    python analysis/paper_figures/loss_curve.py plot
"""
import json
import os
from pathlib import Path
import re
import shutil
import sys

M = Path("/scratch/gpfs/EKOLEMEN/ps9551/ignite/models")
RUN = Path(os.environ.get("LOSS_RUN", M / "ignite_prod_v4/runs/ss75g_l48_spec3_h16"))
CACHE = M / "ignite_prod_v4/frame_codes"
WORK = Path(os.environ.get("LOSS_WORK", "/scratch/gpfs/EKOLEMEN/ps9551/ignite/paper_figures/loss_l48"))
OUT = Path(__file__).resolve().parent / os.environ.get("LOSS_FIG_DIR", "figures_l48_s8600")
LIVE_END = M / "ignite_prod_v4/tend_ip_fallback.json"
# Tokens per frame of each modality (1209 in all); the per-token loss weights them so.
TOKENS = {"ece": 192, "bes": 192, "mhr": 192, "co2": 192, "mirnov": 192,
          "tangtv_lower": 108, "tangtv_upper": 108, "filterscopes": 5}
N_WINDOWS = 256


def per_token(per_modality):
    n = {k: TOKENS.get(k, 4) for k in per_modality}
    return sum(per_modality[k] * n[k] for k in n) / sum(n.values())


def score():
    import torch
    from tokamak_foundation_model.ignite import train_dynamics as td

    WORK.mkdir(parents=True, exist_ok=True)
    # The run's frozen split; an empty out_dir, so nothing is resumed or written there.
    shutil.copy(RUN / "split_shots.json", WORK / "split_shots.json")
    args = td.build_arg_parser().parse_args([
        "--cache_dir", str(CACHE), "--out_dir", str(WORK), "--depth", "48",
        "--d_model", "1024", "--n_heads", "16", "--k0_seed", "20", "--n_predict", "80",
        "--batch_size", "4", "--live_end_json", str(LIVE_END), "--window_sample", "live",
        "--val_window_sample", "first", "--val_windows_per_shot", "10",
        "--val_windows", str(N_WINDOWS), "--grad_ckpt", "0"])
    tr = td.Trainer(CACHE, WORK, td.TrainConfig.from_cli(args))
    tr._setup_model_config()
    tr._setup_split()
    tr.live_end = td._load_live_end(str(LIVE_END))
    tr.use_live = True
    tr.t0_start = td._manifest_t0(CACHE)
    tr.use_presence, tr.presence = False, None
    tr.model = td.MaskGITDynamics(tr.model_cfg).to(tr.device)

    def windows(shots):
        ds = tr._dataset(shots, windows_per_shot=10, window_sample=td.WindowSample.FIRST)
        pos = torch.linspace(0, len(ds) - 1, steps=N_WINDOWS).tolist()
        items = [ds[int(i)] for i in pos]
        return [td._collate_frames(items[s:s + 4]) for s in range(0, len(items), 4)]

    sets = {"train": windows(tr.train_shots), "other": windows(tr.val_shots)}
    out = WORK / "scores.jsonl"
    done = {json.loads(x)["step"] for x in out.read_text().splitlines()} if out.exists() else set()
    ckpts = sorted(RUN.glob("dynamics_step*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    for path in ckpts:
        step = int(re.findall(r"\d+", path.name)[0])
        if step in done:
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        tr.model.load_state_dict(state["model"])
        del state
        row = {"step": step}
        for name, batches in sets.items():
            tr.val_batches = batches
            masked = tr._validate()
            nxt = tr._validate(mask_ratio=1.0, history_frames=tr.model_cfg.k0_seed)
            row[name] = {"masked": per_token(masked.per_modality),
                         "next_frame": per_token(nxt.per_modality),
                         "masked_permod": masked.per_modality,
                         "next_frame_permod": nxt.per_modality}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(step, {k: round(row[k]["masked"], 4) for k in sets},
              {k: round(row[k]["next_frame"], 4) for k in sets}, flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_figures as mf

    mf.style()
    rows = sorted((json.loads(x) for x in (WORK / "scores.jsonl").read_text().splitlines()),
                  key=lambda r: r["step"])
    hist = [json.loads(x) for x in (RUN / "loss_history.jsonl").read_text().splitlines()]
    task = [(r["step"], r["val_task"]) for r in hist if r.get("val_task") is not None]
    step = [r["step"] / 1000 for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(mf.WIDTH, 2.4), gridspec_kw={"wspace": 0.32})
    colors = {"train": "0.25", "other": mf.PALETTE["real"]}
    labels = {"train": "discharges trained on", "other": "other discharges"}
    import math
    for key in ("train", "other"):
        a.plot(step, [math.exp(r[key]["masked"]) for r in rows], color=colors[key], lw=1.3,
               marker="o", ms=3, label=labels[key])
    a.set_xlabel("training step (thousands)")
    # exp(cross-entropy): the number of equally likely codes, of 1000, the model is left
    # to choose from per token (perplexity).
    a.set_ylabel("perplexity (codes of 1000)")
    a.set_ylim(0, None)
    a.legend(frameon=False, loc="upper right")
    s, v = zip(*task)
    b.plot([x / 1000 for x in s], v, color=mf.PALETTE["real"], lw=1.3, marker="o", ms=3)
    b.axhline(0, color="0.6", lw=0.6)
    b.set_xlabel("training step (thousands)")
    b.set_ylabel("rollout skill")
    for ax in (a, b):
        ax.set_xlim(0, max(max(step), s[-1] / 1000) + 0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "loss_curve.png", bbox_inches="tight")
    fig.savefig(OUT / "loss_curve.pdf", bbox_inches="tight")
    print("wrote", OUT / "loss_curve.pdf")




def plot_error():
    """Training curve in physical units: median relative error of the core T_e, T_i, n_e
    and |v_tor| of one sampled rollout per discharge (the scorecard's, measured
    actuators, 2 s to the plasma end) against the measurement, at every snapshot, on the
    same discharges. Data: ``claude_scratch/loss_curve/err_steps.py`` (from
    ``summary_performance.py reduce`` of each ``scorecard_<step>/val32``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_figures as mf
    import summary_performance as sp

    mf.style()
    js = json.loads((WORK / "err_vs_step.json").read_text())
    steps = [s / 1000 for s in js["steps"]]
    colors = ("#1b7f79", "#c1440e", "#5d6b12", "#7b3294")  # as summary_performance C
    fig, ax = plt.subplots(figsize=(mf.WIDTH * 0.55, 2.3))
    for (n, (lab, _u, _s)), col in zip(sp.PROFILES.items(), colors):
        ax.plot(steps, [js["err"][str(s)][n] for s in js["steps"]], color=col, lw=1.3,
                marker="o", ms=3, label=lab)
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("median relative error (%)")
    ax.set_ylim(0, None)
    ax.set_xlim(0, max(steps) + 0.3)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "training_curve.png", bbox_inches="tight")
    fig.savefig(OUT / "training_curve.pdf", bbox_inches="tight")
    print("wrote", OUT / "training_curve.pdf")


if __name__ == "__main__":
    {"score": score, "plot": plot, "plot_error": plot_error}[sys.argv[1]]()
