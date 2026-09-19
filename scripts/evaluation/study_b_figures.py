"""Render Study B figures from the actuator-scan npz/json outputs.

Pure CPU / matplotlib — run any time after ``study_b_actuator_scan.py``
(any subset of its outputs; missing files are skipped with a warning).

Inputs (in ``<out-root>/study_b/``):
- ``b1_sensitivity.npz``   S (n_act, 2, n_diag) = |Δpred|/σ_d per ptype
- ``b1_sign_matrix.json``  physical-readout signs vs textbook expectations
- ``b2_dose_response.npz`` per-window Δreadout keyed ``act|scale|readout``
- ``b3_actuator_swap.npz`` cos-sims keyed ``{lat,pred}_{swap,natural}|name``

Figures → ``<out-root>/figures/study_b/``, tables → ``<out-root>/tables/``.

    python scripts/evaluation/study_b_figures.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))

from tfm_eval.plotting import (  # noqa: E402
    MODALITY_COLORS,
    OKABE_ITO,
    SEQ_CMAP,
    save_fig,
    set_style,
    signed_heatmap,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-root",
        default=str(_HERE.parents[2] / "data" / "outputs" / "eval_suite"
                    / "e2e_stage1_best"),
    )
    return ap.parse_args()


def fig_b1_sensitivity(z, fig_dir):
    """Two-panel actuator x diagnostic sensitivity heatmap (|Δz|/σ_d)."""
    S = z["S"]
    act = [str(s) for s in z["act_names"]]
    diag = [str(s) for s in z["diag_names"]]
    ptypes = [str(s) for s in z["ptypes"]]
    nv = z["n_valid_windows"]
    rows = [f"{a} (n={n})" for a, n in zip(act, nv)]

    fig, axes = plt.subplots(
        1, len(ptypes), figsize=(7.0 * len(ptypes), 0.45 * len(act) + 1.8),
        constrained_layout=True,
    )
    vmax = np.nanpercentile(S, 99)
    for pi, (ax, pt) in enumerate(zip(np.atleast_1d(axes), ptypes)):
        im = ax.imshow(S[:, pi, :], cmap=SEQ_CMAP, vmin=0, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(len(diag)), diag, rotation=45, ha="right")
        ax.set_yticks(range(len(rows)), rows if pi == 0 else [""] * len(rows))
        ax.set_title(f"perturbation: {pt}")
        for i in range(len(act)):
            for j in range(len(diag)):
                v = S[i, pi, j]
                if np.isfinite(v):
                    lum = v / max(vmax, 1e-9)
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=6,
                            color="black" if lum > 0.6 else "white")
        fig.colorbar(im, ax=ax, shrink=0.85,
                     label=r"mean $|\Delta$pred$|$ / $\sigma_d$")
    fig.suptitle("B1 — actuator sensitivity of one-step predictions")
    save_fig(fig, fig_dir / "b1_sensitivity_heatmap")


def fig_b1_signs(rows, fig_dir, tab_dir):
    """Signed directional-consistency heatmap + csv/tex of scored cells."""
    rows_m = [r for r in rows if r["ptype"] == "mult_20pct"]
    acts = sorted({r["actuator"] for r in rows_m})
    reads = sorted({r["readout"] for r in rows_m})
    M = np.full((len(acts), len(reads)), np.nan)
    annot = np.full((len(acts), len(reads)), "", dtype=object)
    for r in rows_m:
        i, j = acts.index(r["actuator"]), reads.index(r["readout"])
        mv, cons = r["mean_delta"], r["sign_consistency"]
        if np.isfinite(mv) and np.isfinite(cons):
            # signed directional consistency in [-1, 1]: +1 = every window
            # moves the readout up under the actuator, -1 = down.
            M[i, j] = np.sign(mv) * (2.0 * cons - 1.0)
        if r["agrees"] is not None:
            annot[i, j] = "✓" if r["agrees"] else "✗"
    fig = signed_heatmap(
        M, acts, reads, annot=annot,
        title="B1 — response direction vs physics expectation "
              "(✓/✗ = scored cells, ±20% actuator scan)",
        cbar_label="signed directional consistency",
    )
    save_fig(fig, fig_dir / "b1_sign_matrix")

    scored = [r for r in rows_m if r["agrees"] is not None]
    hdr = ["actuator", "readout", "mean_delta", "sign_consistency",
           "expected_sign", "agrees"]
    with open(tab_dir / "b1_sign_matrix.csv", "w") as f:
        f.write(",".join(hdr) + "\n")
        for r in scored:
            f.write(f"{r['actuator']},{r['readout']},{r['mean_delta']:.4g},"
                    f"{r['sign_consistency']:.3f},{r['expected_sign']:+d},"
                    f"{r['agrees']}\n")
    n_ok = sum(r["agrees"] for r in scored)
    lines = [r"\begin{tabular}{llrrcc}", r"\toprule",
             "actuator & readout & $\\overline{\\Delta}$ & consistency & "
             "expected & agrees \\\\", r"\midrule"]
    for r in scored:
        lines.append(
            f"{r['actuator'].replace('_', ' ')} & "
            f"{r['readout'].replace('_', ' ')} & {r['mean_delta']:.3g} & "
            f"{r['sign_consistency']:.2f} & {r['expected_sign']:+d} & "
            f"{'yes' if r['agrees'] else 'no'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              f"% sign agreement {n_ok}/{len(scored)} (mult_20pct)"]
    (tab_dir / "b1_sign_matrix.tex").write_text("\n".join(lines) + "\n")
    print(f"B1 sign agreement: {n_ok}/{len(scored)} scored cells")


def fig_b2_dose(z, fig_dir):
    """Dose-response grid: rows = actuators, cols = readouts."""
    scales = z["scales"]
    acts = [str(s) for s in z["dose_actuators"]]
    reads = [str(s) for s in z["readouts"]]
    fig, axes = plt.subplots(
        len(acts), len(reads), figsize=(2.1 * len(reads), 1.9 * len(acts)),
        sharex=True, constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    c = OKABE_ITO[0]
    for i, a in enumerate(acts):
        for j, r in enumerate(reads):
            ax = axes[i, j]
            med, lo, hi = [], [], []
            for s in scales:
                v = z[f"{a}|{s}|{r}"]
                v = v[np.isfinite(v)]
                if v.size:
                    q = np.percentile(v, [25, 50, 75])
                    lo.append(q[0]); med.append(q[1]); hi.append(q[2])
                else:
                    lo.append(np.nan); med.append(np.nan); hi.append(np.nan)
            ax.axhline(0.0, color="0.75", lw=0.8, zorder=1)
            ax.axvline(1.0, color="0.85", lw=0.8, zorder=1)
            ax.fill_between(scales, lo, hi, color=c, alpha=0.25, lw=0)
            ax.plot(scales, med, color=c, lw=1.6, marker="o", ms=3)
            if i == 0:
                ax.set_title(r.replace("_", " "), fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{a.replace('_', ' ')}\n" r"$\Delta$ readout",
                              fontsize=8)
            if i == len(acts) - 1:
                ax.set_xlabel("actuator scale")
            ax.tick_params(labelsize=7)
    fig.suptitle("B2 — dose response of one-step predictions "
                 "(median ± IQR, physical units, Δ vs unscaled)")
    save_fig(fig, fig_dir / "b2_dose_response")


def fig_b3_swap(z, fig_dir, tab_dir):
    """Actuator-swap vs natural-variability cos-sim, dot + IQR per name."""
    names = sorted({k.split("|", 1)[1] for k in z.files if "|" in k})
    # order: predictions first (the conditioning verdict), latents after
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.32 * len(names) + 1.6),
                             sharey=True, constrained_layout=True)
    med_rows = {}
    for ax, kind, title in (
        (axes[0], "pred", "decoded predictions"),
        (axes[1], "lat", "pooled backbone latents"),
    ):
        ys, labels = [], []
        labeled = False
        for yi, name in enumerate(names):
            ks, kn = f"{kind}_swap|{name}", f"{kind}_natural|{name}"
            if ks not in z.files:
                continue
            for key, color, dy, lab in (
                (ks, OKABE_ITO[0], +0.12, "swapped actuators"),
                (kn, "0.55", -0.12, "natural cross-window"),
            ):
                v = z[key]
                v = v[np.isfinite(v)]
                q = np.percentile(v, [25, 50, 75])
                ax.plot(q[1], yi + dy, "o", ms=4, color=color, zorder=3,
                        label=None if labeled else lab)
                ax.plot([q[0], q[2]], [yi + dy] * 2, color=color, lw=1.4,
                        zorder=2)
                med_rows[f"{key}"] = float(q[1])
            labeled = True
            ys.append(yi); labels.append(name)
        ax.set_yticks(ys, labels)
        ax.set_xlabel("cosine similarity to unperturbed pass")
        ax.set_title(title, fontsize=10)
        ax.axvline(1.0, color="0.85", lw=0.8)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("B3 — swapping another window's actuator trajectory: "
                 "swap≈1 while natural≪1 ⇒ weak actuator conditioning")
    save_fig(fig, fig_dir / "b3_actuator_swap")
    (tab_dir / "b3_swap_medians.json").write_text(
        json.dumps(med_rows, indent=1))


def fig_b3_layerwise(z, fig_dir):
    """Propagation profile per zeroed actuator + head-impact heatmap."""
    rows = [str(s) for s in z["row_names"]]
    diag = [str(s) for s in z["diag_names"]]
    rel = z["layer_rel"]          # (R, L) diag-token slice
    act_rel = z["act_layer_rel"]  # (R, L) actuator-token slice
    head = z["head_rel"]          # (R, D)
    L = rel.shape[1]
    xs = np.arange(L)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), constrained_layout=True,
        width_ratios=[1.0, 1.15],
    )
    # rank actuators (non-ALL) by final-layer influence; color the top 4
    order = sorted(range(1, len(rows)), key=lambda r: -rel[r, -1])
    for r in order[4:]:
        ax1.plot(xs, rel[r], color="0.8", lw=0.9, zorder=1)
    labels = [(rel[0, -1], "ALL zeroed", "black", True)]
    for ci, r in enumerate(order[:4]):
        ax1.plot(xs, rel[r], color=OKABE_ITO[ci], lw=1.6, zorder=3)
        labels.append((rel[r, -1], rows[r], OKABE_ITO[ci], False))
    ax1.plot(xs, rel[0], color="black", lw=2.2, zorder=4)
    # de-collide direct labels: enforce a minimum gap in log10 space
    labels.sort(key=lambda t: t[0])
    ys = np.log10([max(v, 1e-12) for v, *_ in labels])
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.09)
    for (v, text, color, bold), ly in zip(labels, ys):
        ax1.annotate(text, (xs[-1], 10 ** ly), xytext=(4, 0),
                     textcoords="offset points", fontsize=8, color=color,
                     va="center",
                     fontweight="bold" if bold else "normal")
    ax1.plot(xs, act_rel[0], color="0.4", lw=1.4, ls="--", zorder=2)
    ax1.annotate("actuator tokens (ALL)", (xs[L // 3], act_rel[0, L // 3]),
                 xytext=(0, 6), textcoords="offset points", fontsize=7,
                 color="0.4")
    ax1.set_yscale("log")
    ax1.set_xlabel("backbone layer (0 = post-embed, last = final norm)")
    ax1.set_ylabel("relative L2 of diagnostic tokens vs baseline")
    ax1.set_title("propagation of zeroed-actuator information", fontsize=10)
    ax1.set_xlim(0, L * 1.22)  # room for direct labels
    if len(rows) - 1 > 4:
        ax1.text(0.02, 0.02, f"gray: remaining {len(rows) - 1 - 4} actuators",
                 transform=ax1.transAxes, fontsize=7, color="0.5")

    im = ax2.imshow(head, cmap=SEQ_CMAP, aspect="auto", vmin=0)
    ax2.set_xticks(range(len(diag)), diag, rotation=45, ha="right")
    ax2.set_yticks(range(len(rows)), rows)
    vmax = float(np.nanmax(head)) or 1.0
    for i in range(len(rows)):
        for j in range(len(diag)):
            ax2.text(j, i, f"{head[i, j]:.1%}", ha="center", va="center",
                     fontsize=6,
                     color="black" if head[i, j] / vmax > 0.6 else "white")
    fig.colorbar(im, ax=ax2, shrink=0.85,
                 label="head-output relative L2 vs baseline")
    ax2.set_title("impact on decoded predictions", fontsize=10)
    fig.suptitle(f"B3 — layerwise actuator propagation "
                 f"(n={int(z['n_windows'])} windows)")
    save_fig(fig, fig_dir / "b3_layerwise")


def fig_b4_physics(z, meta, fig_dir):
    """Physics-signature rollout fans: rows = cases, cols = readouts."""
    from matplotlib import colormaps

    cases = meta["cases"]
    dt = float(z["dt_s"])
    ncols = max(len(c["readouts"]) for c in cases)
    fig, axes = plt.subplots(
        len(cases), ncols, figsize=(3.1 * ncols, 2.5 * len(cases)),
        constrained_layout=True, squeeze=False,
    )
    cmap = colormaps["viridis"]
    for ci, case in enumerate(cases):
        scales = case["scales"]
        for ri, r in enumerate(case["readouts"]):
            ax = axes[ci, ri]
            t = (np.arange(z[f"BASE|{r}"].shape[0]) + 1) * dt
            gt = np.nanmedian(z[f"GT|{r}"], axis=1)
            ax.plot(t, gt, color="0.35", ls=":", lw=1.3,
                    label="GT (actual actuation)")
            for scale in scales:
                key = f"{case['case']}|{scale}|{r}"
                if key not in z.files:
                    continue
                med = np.nanmedian(z[key], axis=1)
                color = ("black" if scale == 1.0 else
                         cmap(0.15 + 0.7 * scales.index(scale)
                              / max(len(scales) - 1, 1)))
                ax.plot(t, med, color=color,
                        lw=2.0 if scale == 1.0 else 1.4,
                        label=f"x{scale}")
            ax.set_title(f"{case['case']}: {r.replace('_', ' ')}",
                         fontsize=9)
            ax.text(0.02, 0.98, case["expect"].get(r, ""),
                    transform=ax.transAxes, fontsize=6.5, va="top",
                    color="0.3", wrap=True)
            if r == "d_alpha":
                rates = []
                for scale in scales:
                    k = f"{case['case']}|{scale}|elm_rate"
                    if k in z.files and z[k].size:
                        rates.append(
                            f"x{scale}:{np.nanmedian(z[k]):.0f}")
                if rates:
                    ax.text(0.02, 0.02, "pred peak rate [Hz] "
                            + " ".join(rates), transform=ax.transAxes,
                            fontsize=6, va="bottom", color="0.3")
            if ci == len(cases) - 1:
                ax.set_xlabel("rollout time [s]")
            if ri == 0:
                ax.set_ylabel("median readout (phys.)", fontsize=8)
                # scale sets differ per case -> one legend per row
                ax.legend(fontsize=6, loc="best", framealpha=0.75,
                          borderpad=0.3, labelspacing=0.25)
            ax.tick_params(labelsize=7)
        for ri in range(len(case["readouts"]), ncols):
            axes[ci, ri].set_visible(False)
    fig.suptitle(
        f"B4 — textbook actuator signatures under sustained scaled "
        f"actuation ({int(z['n_windows'])} windows, K rollout)")
    save_fig(fig, fig_dir / "b4_physics_sweeps")


def main() -> int:
    args = parse_args()
    root = Path(args.out_root)
    sb = root / "study_b"
    fig_dir = root / "figures" / "study_b"
    tab_dir = root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    made = 0
    p = sb / "b1_sensitivity.npz"
    if p.exists():
        fig_b1_sensitivity(np.load(p), fig_dir)
        made += 1
    else:
        print(f"skip B1 heatmap — {p} missing")
    p = sb / "b1_sign_matrix.json"
    if p.exists():
        fig_b1_signs(json.loads(p.read_text()), fig_dir, tab_dir)
        made += 1
    else:
        print(f"skip B1 signs — {p} missing")
    p = sb / "b2_dose_response.npz"
    if p.exists():
        fig_b2_dose(np.load(p), fig_dir)
        made += 1
    else:
        print(f"skip B2 — {p} missing")
    p = sb / "b3_actuator_swap.npz"
    if p.exists():
        fig_b3_swap(np.load(p), fig_dir, tab_dir)
        made += 1
    else:
        print(f"skip B3 swap — {p} missing")
    p = sb / "b3_layerwise.npz"
    if p.exists():
        fig_b3_layerwise(np.load(p), fig_dir)
        made += 1
    else:
        print(f"skip B3 layerwise — {p} missing")
    p = sb / "b4_physics_sweeps.npz"
    pm = sb / "b4_meta.json"
    if p.exists() and pm.exists():
        fig_b4_physics(np.load(p), json.loads(pm.read_text()), fig_dir)
        made += 1
    else:
        print(f"skip B4 — {p} missing")
    print(f"{made}/6 Study B figure groups rendered → {fig_dir}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
