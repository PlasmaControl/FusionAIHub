"""Paper-quality loss-curve figures for the IGNITE backbone scaling study.

Re-renders from the DURABLE loss records — no training needed:
  * per-cell metrics json ``e2e_overfit_200729.json`` -> ``loss_curve`` (per-step; written by
    tests/ignite/test_e2e_overfit_realshot.py for every run from 2026-08-07 on)
  * fallback: the run logs' ``[e2e] backbone step N masked_ce=X`` lines (50-step samples;
    the only record for the pre-json cells — logs are append-only, so this always works)
  * (production runs: ``loss_history.jsonl`` has the same role; point --cell at it if needed)

Outputs (PDF vector + PNG preview, overwritten in place in the scaling dir):
  fig_scaling_curves.{pdf,png}   1x3 small multiples (one panel per width; color = depth)
  fig_scaling_summary.{pdf,png}  final masked CE vs relative compute (log-log)

Design: categorical hues from the validated palette (CVD-checked), color follows DEPTH
consistently across panels, one y-scale everywhere, recessive grid, ink-colored labels.

--session renders the 2026-08-16 OVERFIT / REPLAY figure set instead of the scaling grid, from
``loss_history.jsonl`` plus the recorded rollout metrics. Same durable-record principle: no
training or inference is re-run, so the figures can be regenerated any time.

Usage:  python analysis/render_scaling_loss_figs.py [--dir eval_runs/ignite_e2e_scaling_200729]
        python analysis/render_scaling_loss_figs.py --session [--out eval_runs/session_figs]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

WIDTHS = (256, 512, 1024)
DEPTHS = (4, 8, 16)
# validated categorical palette (dataviz reference, light surface) — color follows DEPTH
DEPTH_COLOR = {4: "#2a78d6", 8: "#eb6834", 16: "#1baf7a"}
INK, INK2 = "#1a1a19", "#5f5e56"

# legacy cells whose curves live only in logs (runs before the json loss_curve existed)
LOG_FALLBACK = {
    "d256x4": "e2e_scale_d256x4_5189391.out",
    "d256x8": "e2e_scale_d256x8_5189393.out",
    "d512x4": "e2e_scale_d512x4_5189392.out",
    "d512x8": "e2e_scale_d512x8_5189644.out",
    "d256x16": "e2e_scale_d256x16_5190810.out",
}
_PAT = re.compile(r"backbone step\s+(\d+)\s+masked_ce=([\d.]+)")


def _bin50(steps: np.ndarray, vals: np.ndarray):
    """Mean-bin a per-step curve to 50-step resolution (comparable with log-sampled cells)."""
    if len(steps) < 2 or (steps[1] - steps[0]) >= 50:
        return steps, vals
    edges = np.arange(0, steps.max() + 50, 50)
    idx = np.digitize(steps, edges)
    out_s, out_v = [], []
    for b in np.unique(idx):
        m = idx == b
        out_s.append(steps[m].mean())
        out_v.append(vals[m].mean())
    return np.asarray(out_s), np.asarray(out_v)


def load_cell(scaling_dir: Path, width: int, depth: int):
    """-> (steps, ce, final_ce) or None. Prefers the json per-step curve; falls back to logs."""
    cell = f"d{width}x{depth}"
    j = scaling_dir / cell / "e2e_overfit_200729.json"
    if j.exists():
        m = json.loads(j.read_text())
        if m.get("loss_curve"):
            v = np.asarray(m["loss_curve"], dtype=float)
            s = np.arange(1, len(v) + 1, dtype=float)
            s, v = _bin50(s, v)
            return s, v, float(m["maskgit_ce_end"])
    lf = LOG_FALLBACK.get(cell)
    if lf and (REPO / "logs" / lf).exists():
        txt = (REPO / "logs" / lf).read_text(errors="ignore")
        pairs = [(int(a), float(b)) for a, b in _PAT.findall(txt)]
        if len(pairs) >= 3:
            s = np.asarray([p[0] for p in pairs], dtype=float)
            v = np.asarray([p[1] for p in pairs], dtype=float)
            final = float(v[-3:].mean())
            jm = scaling_dir / cell / "e2e_overfit_200729.json"
            if jm.exists():
                final = float(json.loads(jm.read_text()).get("maskgit_ce_end", final))
            return s, v, final
    return None


MODELS = Path("/lustre/orion/fus187/proj-shared/models")
# (label, run dir, marginal ln(vocab) mean, colour)
SESSION_RUNS = [
    ("LR 3e-4  (production)", "ignite_bpfull/runs/overfit_const_d256", 2.0794, "#c2352b"),
    ("LR 3e-4  cosine decay", "ignite_bpfull/runs/overfit_proof_d256", 2.0794, "#e0a13a"),
    ("LR 1e-3", "ignite_bpfull/runs/overfit_lr1e3_d256", 2.0794, "#2a78d6"),
    ("LR 3e-3", "ignite_bpfull/runs/overfit_lr3e3_d256", 2.0794, "#1baf7a"),
]
SESSION_SETS = [
    ("bpfull  2 spectro, 1280 tok", "ignite_bpfull/runs/overfit_lr1e3_d256", 2.0794, "#2a78d6"),
    ("bpspectro  5 spectro, 4000 tok", "ignite_bpspectro/runs/overfit_fullcov_d256", 2.0794, "#1baf7a"),
    ("production  14 diag, 1593 tok", "ignite_production/runs/overfit_fullcov_d256", 8.096, "#7a4fbf"),
]
# rollout replay measured on the memorised shot 201056 (bp_eval2), same model/shot/decoder
REPLAY = [(0.7748, 0.442, 0.267, "weak fit"), (0.043, 0.807, 0.838, "memorised"),
          (0.0045, 0.842, 0.890, "best")]
PERMOD_14 = [("ece", 0.285), ("mhr", 0.199), ("bes", 0.124), ("tangtv_lower", 0.079),
             ("co2", 0.079), ("tangtv_upper", 0.077), ("ts_core_density", 0.076),
             ("ts_core_temp", 0.068), ("ts_tang_density", 0.050), ("ts_tang_temp", 0.048),
             ("cer_ti", 0.044), ("cer_rot", 0.039), ("filterscopes", 0.033), ("mse", 0.030)]


def _hist(run):
    p = MODELS / run / "loss_history.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open()]
    tr = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    if len(tr) < 4:
        return None
    s = np.array([a for a, _ in tr]); v = np.array([b for _, b in tr])
    k = max(len(v) // 40, 3)                      # smooth: reveal-ladder noise swings ~0.5
    ker = np.ones(k) / k
    return s[k - 1:], np.convolve(v, ker, mode="valid")


def render_session(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 2.9))

    ax = axes[0]                                   # LR ladder
    for lab, run, _m, c in SESSION_RUNS:
        g = _hist(run)
        if g: ax.plot(g[0], g[1], color=c, lw=1.4, label=lab)
    ax.axhline(2.0794, color=INK2, ls=":", lw=0.9)
    ax.text(0.98, 2.0794, " marginal ln8", color=INK2, va="bottom", ha="right",
            transform=ax.get_yaxis_transform(), fontsize=6.5)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("training CE (nats)")
    ax.set_title("a  learning rate is the limiter\n(one shot, d256xL4)", loc="left")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]                                   # encodings
    for lab, run, marg, c in SESSION_SETS:
        g = _hist(run)
        if g: ax.plot(g[0], g[1] / marg, color=c, lw=1.4, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("CE / marginal")
    ax.set_title("b  both encodings memorise\n(normalised by marginal)", loc="left")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[2]                                   # replay vs fit
    ce = [r[0] for r in REPLAY]
    ax.plot(ce, [r[1] for r in REPLAY], "o-", color="#2a78d6", lw=1.4, label="token acc")
    ax.plot(ce, [r[2] for r in REPLAY], "s-", color="#1baf7a", lw=1.4, label="band-power r")
    ax.axhline(0.369, color="#c2352b", ls="--", lw=1.0)
    ax.text(0.02, 0.369, " persistence", color="#c2352b", va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=6.5)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_ylim(0, 1)
    ax.set_xlabel("training CE (better ->)"); ax.set_ylabel("rollout quality, co2")
    ax.set_title("c  replay tracks fit quality\n(80-frame rollout, shot 201056)", loc="left")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[3]                                   # per-modality
    names = [n for n, _ in PERMOD_14][::-1]; vals = [v for _, v in PERMOD_14][::-1]
    ax.barh(range(len(names)), vals, color="#7a4fbf", height=0.7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6)
    ax.axvline(1.0, color=INK2, ls=":", lw=0.9)
    ax.set_xlabel("CE / ln(vocab)"); ax.set_xlim(0, 0.32)
    ax.set_title("d  all 14 diagnostics fit\n(memorised shot, gen condition)", loc="left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_overfit_session.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig_overfit_session.pdf/.png")


# ---- production-model characterisation, all measured 2026-08-16 --------------------------- #
# CE at ratio 1.0 as REAL history is withdrawn (permod_ce PM_HISTORY_FRAMES) = compounding cost
COMPOUND = {"frames": [99, 90, 50, 20],
            "co2": [1.1966, 1.2331, 1.3211, 1.4053],
            "mhr": [1.7714, 1.7683, 1.7749, 1.8126]}
# matched 1-step condition, bpspectro step-700 ckpt vs model-free baselines
SCORE5 = [("mirnov", 1.3453, 1.1624, 1.0131), ("co2", 1.4350, 1.2591, 1.2110),
          ("bes", 1.2809, 1.0889, 1.3066), ("ece", 1.7578, 1.6726, 1.5340),
          ("mhr", 1.8891, 1.8384, 1.7321)]                      # name, bigram, table3, model
# bp_predss: the objective arm — gen_ce barely moves while masked_ce falls fast
PREDSS = {"step": [250, 500, 750, 1000], "masked": [1.6329, 1.4773, 1.3749, 1.2140],
          "gen": [1.7818, 1.7542, 1.7481, 1.7166]}


def render_production(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))

    ax = axes[0]                                   # compounding
    for m, c in (("co2", "#2a78d6"), ("mhr", "#eb6834")):
        ax.plot(COMPOUND["frames"], COMPOUND[m], "o-", color=c, lw=1.4, label=m)
    ax.invert_xaxis()
    ax.set_xlabel("frames of REAL history"); ax.set_ylabel("CE at ratio 1.0 (nats)")
    ax.set_title("a  compounding cost\n(bpfull d1024xL16, step 4000)", loc="left")
    ax.legend(frameon=False)

    ax = axes[1]                                   # matched-condition scoreboard
    n = np.arange(len(SCORE5)); w = 0.27
    ax.barh(n + w, [r[1] for r in SCORE5], w, color="#c2352b", label="bigram")
    ax.barh(n, [r[2] for r in SCORE5], w, color="#e0a13a", label="3-frame table")
    ax.barh(n - w, [r[3] for r in SCORE5], w, color="#1baf7a", label="model")
    ax.axvline(2.0794, color=INK2, ls=":", lw=0.9)
    ax.text(2.0794, len(SCORE5) - 0.4, " marginal", color=INK2, fontsize=6.5, va="top")
    ax.set_yticks(n); ax.set_yticklabels([r[0] for r in SCORE5])
    ax.set_xlabel("CE, matched 1-step (nats)"); ax.set_xlim(0, 2.25)
    ax.set_title("b  model beats both baselines\non 4/5 diagnostics", loc="left")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[2]                                   # the objective arm
    ax.plot(PREDSS["step"], PREDSS["masked"], "o-", color="#2a78d6", lw=1.4,
            label="masked_ce (infilling)")
    ax.plot(PREDSS["step"], PREDSS["gen"], "s-", color="#c2352b", lw=1.4,
            label="gen_ce (generation)")
    ax.set_xlabel("step"); ax.set_ylabel("val CE (nats)")
    ax.set_title("c  infilling improves,\ngeneration barely does", loc="left")
    ax.legend(frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_production_characterisation.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig_production_characterisation.pdf/.png")


# --- memorisation-capacity runs: train/val vs EPOCH (windows/batch differs per N) ------------
# (label, run, windows, effective batch, colour) — epoch = step * batch / windows, and the
# effective batch is NOT constant across runs (ACCUM_STEPS changes it), so it must travel with
# each entry or an accumulation arm lands at the wrong epoch on the x-axis.
MEMCAP = [("N=1  d256xL4 (memorised)", "ignite_bpfull/runs/overfit_lr1e3_d256",  140,  8, "#2a78d6"),
          ("N=2  batch 64",            "ignite_bpfull/runs/memcap_N2_acc8",      280, 64, "#111111"),
          ("N=2  stride 10",           "ignite_bpfull/runs/memcap_N2_stride10",   28,  8, "#b02f8a"),
          ("N=4  d256xL4",             "ignite_bpfull/runs/memcap_N4",           560,  8, "#eb6834"),
          ("N=8  d256xL4 (stalled)",   "ignite_bpfull/runs/memcap_N8",          1120,  8, "#1baf7a")]



def _series(run, windows, batch=8):
    p = MODELS / run / "loss_history.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open()]
    spe = windows / batch                       # steps per epoch
    tr = [(r["step"] / spe, r["loss"]) for r in rows if "loss" in r]
    va = [(r["step"] / spe, r["val_loss"], r.get("gen_loss"))
          for r in rows if "val_loss" in r]
    if len(tr) < 2:
        return None
    e = np.array([a for a, _ in tr]); v = np.array([b for _, b in tr])
    # Rolling-mean window must leave a DRAWABLE line. A fixed k=5 on a 5-point series collapses
    # to one value and matplotlib renders nothing — a young arm silently vanishes from the plot.
    # Cap k so at least 4 smoothed points survive, and fall back to raw for very short series.
    k = max(1, min(len(v) // 30, max(len(v) - 3, 1)))
    sm = np.convolve(v, np.ones(k) / k, mode="valid") if k > 1 else v
    return e, v, e[k - 1:], sm, va


def render_memcap(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))

    ax = axes[0]
    for lab, run, w, bs, c in MEMCAP:
        g = _series(run, w, bs)
        if not g:
            continue
        e, raw, es, sm, _ = g
        ax.plot(e, raw, color=c, lw=0.5, alpha=0.25)          # raw points, honest spread
        ax.plot(es, sm, color=c, lw=1.6, label=lab)
    ax.axhline(2.0794, color=INK2, ls=":", lw=0.9)
    ax.text(0.99, 2.0794, " marginal", color=INK2, va="bottom", ha="right",
            transform=ax.get_yaxis_transform(), fontsize=6.5)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlabel("epoch  (= step x batch / windows)"); ax.set_ylabel("TRAIN CE (nats)")
    ax.set_title("a  memorisation vs epochs\nd256xL4, LR 1e-3 const, wd=0, gen_mask=1.0", loc="left")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    for lab, run, w, bs, c in MEMCAP:
        g = _series(run, w, bs)
        if not g:
            continue
        _e, _raw, es, sm, va = g
        ax.plot(es, sm, color=c, lw=1.6, label=f"{lab} train")
        if va:
            ax.plot([a for a, _, _ in va], [b for _, b, _ in va], color=c, lw=1.2,
                    ls="--", marker="o", ms=3, label=f"{lab} val")
    ax.axhline(2.0794, color=INK2, ls=":", lw=0.9)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("CE (nats)")
    ax.set_title("b  train (solid) vs val (dashed)\ngap = overfit; both high = underfit", loc="left")
    ax.legend(frameon=False, loc="lower left", ncol=2)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_memcap_curves.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig_memcap_curves.pdf/.png")


# --- stride-10 LR scan: separate figure, STEP axis (all arms share 28 windows, so steps are
# directly comparable and no epoch conversion is needed) --------------------------------------
LRSCAN = [("LR 3e-4",  "ignite_bpfull/runs/stride10_s10lr3e4",  "#2a78d6"),
          ("LR 1e-3",  "ignite_bpfull/runs/memcap_N2_stride10", "#1baf7a"),
          ("LR 3e-3",  "ignite_bpfull/runs/stride10_s10lr3e3",  "#e0a13a"),
          ("LR 1e-2",  "ignite_bpfull/runs/stride10_s10lr1e2",  "#c2352b")]


def render_lrscan(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for lab, run, c in LRSCAN:
        f = MODELS / run / "loss_history.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.open()]
        tr = [(r["step"], r["loss"]) for r in rows if "loss" in r]
        va = [(r["step"], r["val_loss"], r.get("gen_loss")) for r in rows if "val_loss" in r]
        if len(tr) < 2:
            continue
        st = np.array([a for a, _ in tr]); v = np.array([b for _, b in tr])
        k = max(1, min(len(v) // 20, max(len(v) - 3, 1)))
        sm = np.convolve(v, np.ones(k) / k, mode="valid") if k > 1 else v
        axes[0].plot(st, v, color=c, lw=0.5, alpha=0.25)
        axes[0].plot(st[k - 1:], sm, color=c, lw=1.6, label=lab)
        if va:
            axes[1].plot([a for a, _, _ in va], [b for _, b, _ in va], color=c, lw=1.4,
                         marker="o", ms=3, label=f"{lab} masked")
            axes[1].plot([a for a, _, _ in va], [g for _, _, g in va], color=c, lw=1.2,
                         ls="--", label=f"{lab} gen")
    for ax, ttl, yl in ((axes[0], "a  stride-10 LR scan, TRAIN\nN=2, 28 windows, batch 8, beta2=0.9",
                         "train CE (nats)"),
                        (axes[1], "b  held-out (solid masked, dashed gen)", "val CE (nats)")):
        ax.axhline(2.0794, color=INK2, ls=":", lw=0.9)
        ax.set_xlabel("optimizer step"); ax.set_ylabel(yl)
        ax.set_title(ttl, loc="left"); ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_stride10_lrscan.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig_stride10_lrscan.pdf/.png")


def render_runs(specs, out: Path, marginal: float = None) -> None:
    """THE DEFAULT VERDICT FIGURE: train AND both fixed-protocol val metrics, one axis per run.

    Every wrong call this project has made came from reading ONE curve. The training loss under
    `gen_mask_p>0` resamples its own task each step (random split point + random ladder), so it
    swings ~0.4 nats and cannot be read for trend; the val metrics are fixed-protocol and CAN.
    Plotting them apart let N=1 look like a triumph (train 1.06 -> 0.0068) while its held-out
    gen_ce went 2.14 -> 18.28. Here they always share an axis, and the best-val step is marked,
    so a turnaround is visible the moment it happens instead of eight hours later.
    """
    out.mkdir(parents=True, exist_ok=True)
    n = len(specs)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.1), squeeze=False)
    for ax, (lab, run) in zip(axes[0], specs):
        p = (MODELS / run / "loss_history.jsonl") if not Path(run).is_absolute() \
            else Path(run) / "loss_history.jsonl"
        if not p.exists():
            ax.set_title(f"{lab}\n(no loss_history)", loc="left"); continue
        rows = [json.loads(l) for l in p.open()]
        tr = np.array([(r["step"], r["loss"]) for r in rows if "loss" in r], dtype=float)
        va = [(r["step"], r.get("val_loss"), r.get("gen_loss")) for r in rows if "val_loss" in r]
        if tr.size:
            k = max(1, min(len(tr) // 25, max(len(tr) - 3, 1)))
            sm = np.convolve(tr[:, 1], np.ones(k) / k, mode="valid") if k > 1 else tr[:, 1]
            ax.plot(tr[:, 0], tr[:, 1], color="#2a78d6", lw=0.4, alpha=0.25)
            ax.plot(tr[k - 1:, 0], sm, color="#2a78d6", lw=1.6, label="train")
        if va:
            vs = [a for a, _, _ in va]
            vm = [b for _, b, _ in va]
            ax.plot(vs, vm, color="#eb6834", lw=1.5, marker="o", ms=3, label="val masked")
            g = [(a, c) for a, _, c in va if c is not None]
            if g:
                ax.plot([a for a, _ in g], [c for _, c in g], color="#b02f8a", lw=1.4,
                        ls="--", marker="s", ms=2.5, label="val gen")
            bi = int(np.argmin(vm))
            ax.axvline(vs[bi], color="#111111", ls=":", lw=1.0)
            ax.annotate(f"best val {vm[bi]:.3f}\n@step {vs[bi]}", (vs[bi], vm[bi]),
                        textcoords="offset points", xytext=(6, 8), fontsize=6.5)
        if marginal:
            ax.axhline(marginal, color=INK2, ls=":", lw=0.9)
        ax.set_yscale("log"); ax.set_xlabel("optimizer step"); ax.set_ylabel("CE (nats)")
        ax.set_title(lab, loc="left"); ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_runs.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig_runs.pdf/.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval_runs/ignite_e2e_scaling_200729")
    ap.add_argument("--runs", default=None,
                    help='verdict figure: "label=run_path[,label=run_path...]" (run_path is '
                         'relative to the models root, or absolute). Plots train + both '
                         'fixed-protocol val metrics on one axis and marks the best-val step.')
    ap.add_argument("--marginal", type=float, default=None,
                    help="draw a marginal/uniform-guess reference line at this CE")
    ap.add_argument("--lrscan", action="store_true",
                    help="render the stride-10 LR scan as its own figure (step axis)")
    ap.add_argument("--memcap", action="store_true",
                    help="render the memorisation-capacity train/val curves vs EPOCH")
    ap.add_argument("--session", action="store_true",
                    help="render the overfit/replay figure set instead of the scaling grid")
    ap.add_argument("--out", default="eval_runs/session_figs")
    args = ap.parse_args()
    sdir = (REPO / args.dir) if not Path(args.dir).is_absolute() else Path(args.dir)

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7, "axes.edgecolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # embed TrueType (journal requirement)
    })

    if args.runs:
        # rsplit, not split: labels legitimately contain "=" ("N=1 (overfit)=path"), while run
        # paths never do. Splitting on the FIRST "=" silently truncated every label to "N" and
        # turned the path into garbage, so the panels rendered empty.
        specs = [(s.rsplit("=", 1)[0], s.rsplit("=", 1)[1]) if "=" in s else (s, s)
                 for s in args.runs.split(",")]
        render_runs(specs, (REPO / args.out) if not Path(args.out).is_absolute()
                    else Path(args.out), marginal=args.marginal)
        return
    if args.lrscan:
        render_lrscan((REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out))
        return
    if args.memcap:
        render_memcap((REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out))
        return
    if args.session:
        o = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
        render_session(o); render_production(o)
        return

    cells = {}
    for w in WIDTHS:
        for d in DEPTHS:
            got = load_cell(sdir, w, d)
            if got:
                cells[(w, d)] = got
    if not cells:
        raise SystemExit(f"no loss records found under {sdir}")

    # ---- Fig 1: training curves, small multiples by width, color = depth ------------------- #
    ylo = min(v.min() for _s, v, _f in cells.values()) * 0.8
    yhi = max(v.max() for _s, v, _f in cells.values()) * 1.15
    fig, axes = plt.subplots(1, len(WIDTHS), figsize=(7.0, 2.4), sharey=True)
    for ax, w in zip(axes, WIDTHS):
        for d in DEPTHS:
            if (w, d) not in cells:
                continue
            s, v, f = cells[(w, d)]
            ax.plot(s, v, color=DEPTH_COLOR[d], lw=1.4, solid_capstyle="round",
                    label=f"depth {d}")
            ax.annotate(f"{f:.2f}", (s[-1], v[-1]), xytext=(3, 0),
                        textcoords="offset points", fontsize=6.5, color=INK, va="center")
        ax.set_yscale("log")
        ax.set_ylim(ylo, yhi)
        ax.set_title(f"$d_\\mathrm{{model}}$ = {w}")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25, which="both", lw=0.4)
    axes[0].set_ylabel("masked-token CE")
    handles = [plt.Line2D([], [], color=DEPTH_COLOR[d], lw=1.4, label=f"depth {d}")
               for d in DEPTHS]
    axes[-1].legend(handles=handles, frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(sdir / f"fig_scaling_curves.{ext}", dpi=300)
    plt.close(fig)

    # ---- Fig 2: final CE vs relative compute (log-log) ------------------------------------- #
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for d in DEPTHS:
        xs, ys, labs = [], [], []
        for w in WIDTHS:
            if (w, d) not in cells:
                continue
            xs.append((w / 256) ** 2 * (d / 4))
            ys.append(cells[(w, d)][2])
            labs.append(f"d{w}")
        if not xs:
            continue
        ax.plot(xs, ys, "o-", color=DEPTH_COLOR[d], lw=1.2, ms=4,
                mec="white", mew=0.6, label=f"depth {d}")
        for x, y, t in zip(xs, ys, labs):
            ax.annotate(t, (x, y), xytext=(4, 3), textcoords="offset points",
                        fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relative compute  $(d/256)^2 \\times (\\mathrm{depth}/4)$")
    ax.set_ylabel("final masked-token CE")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(sdir / f"fig_scaling_summary.{ext}", dpi=300)
    plt.close(fig)
    print(f"rendered {len(cells)} cells -> {sdir}/fig_scaling_curves.[pdf|png], "
          f"fig_scaling_summary.[pdf|png]")


if __name__ == "__main__":
    main()
