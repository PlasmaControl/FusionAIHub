"""Figures 01-07 for outputs/labelmaker/presentation_continued, both tearing models.

A copy of `../../presentation/scripts/make_presentation.py` parametrised by
`--dsm-slug` and `--out-dir`, so the seven Phase 2 figures can be re-rendered
against the retrained survival checkpoint with the CNN panels unchanged. The
original is left untouched.

Reads pool_rows.npz (built by pool_rows.py over the 500-shot pool for the same
slug) plus the validation JSON, and the label files for the two example shots.

    python make_presentation.py [--dsm-slug <slug>] [--subset {all,held_out,in_training}]
                                [--out-dir <dir>] [--rows pool_rows.npz]

`--subset` restricts the whole pool to the shots the survival model was or was
not trained on; see the original for what that does and does not mean.
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from labeler.config import Paths
from labeler.labels.store import read_label
from labeler.validate import binary_metrics

SCR = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--dsm-slug", default="d3d_tearing_time_to_event_dsm_continued")
ap.add_argument("--subset", choices=("all", "held_out", "in_training"), default="all",
                help="which shots of the pool to draw, by the survival model's training set")
ap.add_argument("--out-dir", default=str(SCR.parent))
ap.add_argument("--rows", default=str(SCR / "pool_rows.npz"))
args = ap.parse_args()
SUBSET = args.subset
OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
paths = Paths.from_env()
CNN, DSM = "d3d_tearing_onset_cnn1d", args.dsm_slug
# The CNN reconstruction panel reads the CNN's own validation report; it is the
# same file whichever survival checkpoint is being drawn.
VAL = paths.validation / CNN

d = np.load(args.rows)
RISKS = ("tm_risk_250ms", "tm_risk_500ms", "tm_risk_1s")
HOR = dict(zip(RISKS, d["horizons"]))

# ---- palette (dataviz reference instance, light mode) ----
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ARCH, RECON, VIOLET, GREEN = "#2a78d6", "#eb6834", "#4a3aa7", "#1f8a5a"
GRAY = MUTED
DSM_C = {"tm_risk_250ms": "#7fb2ec", "tm_risk_500ms": GREEN, "tm_risk_1s": VIOLET}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.titlepad": 9,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9.5, "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.9,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
})
RING = dict(markeredgecolor=SURF, markeredgewidth=1.4)
LBL_A, LBL_R = "archived (training) inputs", "labelmaker reconstruction"

# ---- the subset: a whole number of shots, by the survival model's training set ----
TRAIN = d["in_training"].astype(bool)
KEEP = {"all": np.ones(TRAIN.size, bool), "held_out": ~TRAIN, "in_training": TRAIN}[SUBSET]
SUBSET_NOTE = {
    "all": "every aligned shot of the pool, in-sample and held-out together",
    "held_out": "only the shots the survival model was NOT trained on",
    "in_training": "only the shots the survival model WAS trained on",
}[SUBSET]

# ---- CNN rows: published (valid) rows only ----
v = d["valid"].astype(bool) & KEEP
y = d["truth_tm"].astype(bool)[v]
pa, pr = d["p_arch"][v], d["p_recon"][v]
ba, br, bt = d["b_arch"][v], d["b_recon"][v], d["truth_bn"][v]
N, NPOS = y.size, int(y.sum())
BASE = NPOS / N
NSHOTS = int(np.unique(d["shot"][KEEP]).size)

# ---- DSM rows: valid, pre-onset, on shots that have an onset ----
dv = d["dsm_valid"].astype(bool) & KEEP
tto = d["time_to_onset"]
has_onset = d["has_onset"].astype(bool)
# Two different questions, and they must not be mixed:
#  ALL   - every aligned shot, pre-onset rows; a quiet shot contributes
#          negatives only. "Which shots and times are heading for a mode?"
#  ONSET - only the shots that do have an onset. "Given this shot gets one,
#          when?" - much harder, and the positive rate is far higher.
pre_all = dv & (~has_onset | (np.isfinite(tto) & (tto > 1e-9)))
pre_onset = dv & has_onset & np.isfinite(tto) & (tto > 1e-9)


def dsm_rows(mask):
    out = {}
    for r in RISKS:
        keep = mask & np.isfinite(d[r])
        within = np.zeros(keep.sum(), bool)
        tt = tto[keep]
        good = np.isfinite(tt)
        within[good] = tt[good] <= HOR[r] + 1e-9
        out[r] = (d[r][keep], within)
    return out


DSM_ALL, DSM_ONSET = dsm_rows(pre_all), dsm_rows(pre_onset)
DSM_ROWS = DSM_ALL


def roc(p, t):
    o = np.argsort(-p, kind="mergesort"); tt = t[o]
    return np.r_[0, np.cumsum(~tt) / (~tt).sum()], np.r_[0, np.cumsum(tt) / tt.sum()]


def pr_curve(p, t):
    o = np.argsort(-p, kind="mergesort"); tt = t[o]; k = np.arange(1, tt.size + 1)
    return np.cumsum(tt) / tt.sum(), np.cumsum(tt) / k


def at_threshold(p, t, th):
    pred = p >= th; tp = (pred & t).sum(); fp = (pred & ~t).sum(); fn = (~pred & t).sum()
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0
    return prec, rec, f1, (fp / (~t).sum() if (~t).any() else np.nan), pred.mean()


def footer(fig, text):
    """Wrapped, so `bbox="tight"` does not stretch the canvas to fit one line."""
    width = int(fig.get_size_inches()[0] * 15)
    fig.text(0.01, -0.01, "\n".join(textwrap.wrap(" ".join(text.split()), width)),
             ha="left", va="top", fontsize=7.8, color=MUTED)


SUB = (f"500-shot pool: {NSHOTS} shots aligned to their archived training rows, "
       f"{N:,} published CNN rows ({NPOS:,} tearing-positive, {BASE:.1%}); "
       f"survival model {DSM}; subset {SUBSET} - {SUBSET_NOTE}")

# ---------- 1. ROC + PR, both models ----------
fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
(a1, a2), (a3, a4) = axes
for p, c, name in ((pa, ARCH, LBL_A), (pr, RECON, LBL_R)):
    fpr, tpr = roc(p, y)
    a1.plot(fpr, tpr, color=c, label=f"{name} - AUROC {binary_metrics(p, y)['auroc']:.3f}")
    prec, rec, _, fp5, _ = at_threshold(p, y, 0.5)
    a1.plot(fp5, rec, "o", color=c, ms=7, **RING)
    r, q = pr_curve(p, y); a2.plot(r, q, color=c, label=name); a2.plot(rec, prec, "o", color=c, ms=7, **RING)
a1.plot([0, 1], [0, 1], color=AXIS, lw=0.9, zorder=0)
a1.set(xlabel="false-positive rate", ylabel="recall", xlim=(0, 1), ylim=(0, 1.02))
a1.set_title("CNN tm_prob - is a mode present at t+25 ms")
a1.legend(loc="lower right")
a2.axhline(BASE, color=AXIS, lw=0.9, zorder=0)
a2.text(0.99, BASE + 0.02, f"base rate {BASE:.3f}", ha="right", fontsize=8, color=INK2)
a2.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02))
a2.set_title("CNN tm_prob, precision-recall (dots: threshold 0.5)")
a2.legend(loc="upper right")
for r in RISKS:
    p, t = DSM_ROWS[r]
    fpr, tpr = roc(p, t)
    a3.plot(fpr, tpr, color=DSM_C[r], label=f"{r} - AUROC {binary_metrics(p, t.astype(float))['auroc']:.3f}")
    rr, qq = pr_curve(p, t)
    a4.plot(rr, qq, color=DSM_C[r], label=f"{r} - {t.mean():.1%} of rows positive")
a3.plot([0, 1], [0, 1], color=AXIS, lw=0.9, zorder=0)
a3.set(xlabel="false-positive rate", ylabel="recall", xlim=(0, 1), ylim=(0, 1.02))
a3.set_title("Survival risk - will a mode appear within the horizon")
a3.legend(loc="lower right")
a4.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02))
a4.set_title("Survival risk, precision-recall")
a4.legend(loc="upper right")
fig.suptitle("Both tearing models against the archived truth", x=0.01, ha="left", fontsize=13, fontweight="semibold")
footer(fig, SUB + f". The survival panels use every aligned shot's pre-onset rows "
       f"({int(pre_all.sum()):,} rows): once a mode is present, 'will one appear' is no longer the "
       "question, and a shot that never gets one contributes negatives. Restricted to the "
       f"{int(np.isfinite(d['per_shot'][:, 6]).sum())} shots that DO get one ({int(pre_onset.sum()):,} rows), "
       "AUROC falls to "
       + ", ".join(f"{binary_metrics(*[DSM_ONSET[r][0], DSM_ONSET[r][1].astype(float)])['auroc']:.2f}" for r in RISKS)
       + " - telling which shots are heading for a mode is easier than timing the one that comes. "
       "The two halves answer different questions and their base rates differ.")
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "01_roc_pr.png"); plt.close(fig)

# ---------- 2. F1 against threshold ----------
ths = np.linspace(0.02, 0.98, 97)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
best = {}
for p, c, name in ((pa, ARCH, LBL_A), (pr, RECON, LBL_R)):
    f1 = np.array([at_threshold(p, y, th)[2] for th in ths]); i = int(np.nanargmax(f1))
    best[name] = (f1[i], ths[i])
    a1.plot(ths, f1, color=c, label=f"{name} - best {f1[i]:.2f} at {ths[i]:.2f}")
    a1.plot(ths[i], f1[i], "o", color=c, ms=7, **RING)
    a1.plot(0.5, at_threshold(p, y, 0.5)[2], "s", color=c, ms=6, **RING)
for r in RISKS:
    p, t = DSM_ROWS[r]
    f1 = np.array([at_threshold(p, t, th)[2] for th in ths]); i = int(np.nanargmax(f1))
    a2.plot(ths, f1, color=DSM_C[r], label=f"{r} - best {f1[i]:.2f} at {ths[i]:.2f}")
    a2.plot(ths[i], f1[i], "o", color=DSM_C[r], ms=7, **RING)
for ax, ttl in ((a1, "CNN tm_prob (squares: the reported threshold 0.5)"),
                (a2, "Survival risk at three horizons")):
    ax.axvline(0.5, color=INK2, lw=0.9, ls=(0, (2, 3)), zorder=0)
    ax.set(xlim=(0, 1), ylim=(0, 0.75), xlabel="decision threshold")
    ax.set_title(ttl); ax.legend(loc="upper right")
a1.set(ylabel="F1")
fig.suptitle("The reported threshold is not the best threshold for either model", x=0.01, ha="left", fontsize=13, fontweight="semibold")
footer(fig, SUB + ". The CNN was trained oversampled and class-weighted (mse_bin_os_w), so at 0.5 it "
       f"flags {at_threshold(pa, y, 0.5)[4]:.0%} of rows against a {BASE:.1%} base rate. The survival "
       "risk is a probability of onset within the horizon over every aligned shot's pre-onset rows, on a "
       "scale of its own; both need a chosen operating point, which `analyze` takes per label in its config.")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "02_f1_vs_threshold.png"); plt.close(fig)

# ---------- 3. calibration ----------
edges = np.linspace(0, 1, 11)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.9))
series = [(pa, y, ARCH, f"CNN, {LBL_A}"), (pr, y, RECON, f"CNN, {LBL_R}"),
          (DSM_ALL["tm_risk_1s"][0], DSM_ALL["tm_risk_1s"][1], VIOLET,
           "survival tm_risk_1s, every aligned shot"),
          (DSM_ONSET["tm_risk_1s"][0], DSM_ONSET["tm_risk_1s"][1], GREEN,
           "survival tm_risk_1s, only shots with an onset")]
for p, t, c, name in series:
    which = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    mp = np.array([p[which == b].mean() if (which == b).any() else np.nan for b in range(10)])
    ob = np.array([t[which == b].mean() if (which == b).any() else np.nan for b in range(10)])
    a1.plot(mp, ob, "-o", color=c, ms=7, label=f"{name} - ECE {binary_metrics(p, np.asarray(t, float))['ece']:.3f}", **RING)
    a2.hist(p, bins=edges, histtype="step", color=c, lw=1.9, label=name)
a1.plot([0, 1], [0, 1], color=AXIS, lw=0.9, zorder=0)
a1.set(xlabel="mean predicted probability in bin", ylabel="observed frequency of the event", xlim=(0, 1), ylim=(0, 1))
a1.set_title("Reliability - each label against its own event and row set")
a1.legend(loc="upper left")
a2.set_yscale("log"); a2.set(xlabel="predicted probability", ylabel="rows (log)", xlim=(0, 1))
a2.set_title("Where the probabilities land")
a2.legend(loc="upper right")
fig.suptitle("Calibration depends on which question is asked", x=0.01, ha="left", fontsize=13, fontweight="semibold")
footer(fig, SUB + ". The CNN is over-confident on its own inputs and less so through the reconstruction. "
       "The survival risk is well calibrated over every aligned shot, where an onset within 1 s is rare "
       f"({DSM_ALL['tm_risk_1s'][1].mean():.1%} of rows) - and badly UNDER-confident once the row set is "
       f"narrowed to shots that do get one ({DSM_ONSET['tm_risk_1s'][1].mean():.1%} positive): its scale is "
       "tuned to the population, not to a shot already known to be heading for a mode.")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "03_calibration.png"); plt.close(fig)

# Figure 04 is pooled over every aligned shot in the CNN's validation report,
# which has no held-out / in-training split of its own. Rather than publish it
# inside a subset folder under a population it does not have, it is drawn for
# `--subset all` alone.
if SUBSET == "all":
    # ---------- 4. reconstruction fidelity ----------
    rec = json.loads((VAL / "reconstruction.json").read_text())["per_feature"]
    groups = [
        ("ZIPFIT fits standing in for the pipeline's mtanh / csaps fits",
         ["cer_rot_csaps_1d", "thomson_temp_mtanh_1d", "thomson_density_mtanh_1d"]),
        ("offline EFIT01 standing in for real-time EFITRT1",
         ["1/qpsi_EFITRT1", "kappa_EFITRT1", "R0_EFITRT1"]),
        ("PTDATA through fdp", ["ip", "bt"]),
        ("served by the archive - the training rows themselves",
         ["pres_EFIT01", "ech_pwr_total", "EC.RHO_ECH", "gapin_EFIT01", "tribot_EFIT01",
          "tritop_EFIT01", "tinj", "pinj"]),
    ]
    UNPRICED = ["qmin", "li", "aminor", "volume", "pcbcoil", "ti_zipfit"]
    FLOOR = 1e-6
    fig, ax = plt.subplots(figsize=(11.5, 8))
    ypos, labels, ticks = 0, [], []
    for title, names in groups:
        ypos += 0.9
        ax.text(FLOOR * 0.55, ypos + len(names) - 0.35, title, fontsize=9, color=INK2, va="center", ha="left", clip_on=False)
        ypos -= 0.2
        for n in names:
            vv = rec[n]["median_rel"]; exact = vv == 0.0
            ax.barh(ypos, FLOOR * 1.6 if exact else max(vv, FLOOR), left=FLOOR, height=0.6,
                    color=GRAY if exact else ARCH, alpha=0.55 if exact else 1.0, edgecolor=SURF, linewidth=1.4)
            txt = "exact (0)" if exact else f"{vv:.1e}   corr {rec[n]['corr']:.2f}   {rec[n]['n_shots']} shots"
            ax.text((FLOOR + (FLOOR * 1.6 if exact else max(vv, FLOOR))) * 1.25, ypos, txt, va="center", fontsize=8.3, color=INK2)
            labels.append(n); ticks.append(ypos); ypos += 1
    ypos += 0.9
    ax.text(FLOOR * 0.55, ypos + len(UNPRICED) - 0.35,
            "used only by the survival model - no archived column to price against",
            fontsize=9, color=INK2, va="center", ha="left", clip_on=False)
    ypos -= 0.2
    for n in UNPRICED:
        ax.barh(ypos, FLOOR * 1.6, left=FLOOR, height=0.6, color=SURF, edgecolor=AXIS, linewidth=1.0, hatch="///")
        ax.text(FLOOR * 3.0, ypos, "unpriced - the survival model's training rows are not on disk",
                va="center", fontsize=8.3, color=MUTED)
        labels.append(n); ticks.append(ypos); ypos += 1
    ax.set_xscale("log"); ax.set_xlim(FLOOR, 1.0); ax.set_ylim(-0.6, ypos + 0.4)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_xlabel("median relative difference to the model's own training input (log)")
    ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("Reconstruction fidelity, feature by feature - the profile substitutions are the cost")
    ax.legend(handles=[Patch(color=ARCH, label="measured substitution"),
                       Patch(color=GRAY, alpha=0.55, label="bit-identical to the training input"),
                       Patch(facecolor=SURF, edgecolor=AXIS, hatch="///", label="no truth to compare against")],
              loc="upper right")
    footer(fig, f"{NSHOTS} aligned shots of the 500-shot pool. Pooled median of |ours - archived| / |archived| at the aligned rows.")
    fig.tight_layout(); fig.savefig(OUT / "04_reconstruction_fidelity.png"); plt.close(fig)

# ---------- 5. example shots ----------
def spans(mask, x):
    mask = np.asarray(mask, bool); out, start = [], None
    for i, on in enumerate(mask):
        if on and start is None: start = x[i]
        if start is not None and (not on or i == mask.size - 1):
            out.append((start, x[i] if on else x[i - 1])); start = None
    return out


def example_shot(shot, tag, title, note):
    sel = d["shot"] == shot
    t_m = d["t"][sel]; tm_m = d["truth_tm"][sel].astype(bool)
    onset = float(t_m[np.argmax(tm_m)]) if tm_m.any() else None
    lp = paths.labels_file(shot)
    lab = read_label(lp, CNN, "tm_prob"); t_all, p_pub = lab.x, lab.y[0]
    v_pub = read_label(lp, CNN, "tm_prob_valid").y[0].astype(bool)
    b_pub = read_label(lp, CNN, "betan").y[0]
    risk = read_label(lp, DSM, "tm_risk_1s").y[0]
    rv = read_label(lp, DSM, "tm_risk_1s_valid").y[0].astype(bool)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11.5, 9.6), sharex=True,
                                        gridspec_kw={"height_ratios": [1.25, 1, 1]})
    for ax in (ax1, ax2, ax3):
        for a, b in spans(tm_m, t_m):
            ax.axvspan(a, b, color=VIOLET, alpha=0.14, lw=0, zorder=0)
        if onset is not None:
            ax.axvline(onset, color=VIOLET, lw=1.5, zorder=1)
    for ax, mask in ((ax1, ~v_pub), (ax2, ~rv), (ax3, ~v_pub)):
        for a, b in spans(mask, t_all):
            ax.axvspan(a, b, color=GRAY, alpha=0.13, lw=0, zorder=0)
    ax1.plot(t_all, p_pub, color=RECON, label="published tm_prob (reconstruction)")
    ax1.plot(t_m, d["p_arch"][sel], "o", color=ARCH, ms=5, label="tm_prob from the archived training inputs", **RING)
    ax1.axhline(0.5, color=INK2, lw=0.9, ls=(0, (2, 3)), zorder=0)
    ax1.set(ylabel="tm_prob", ylim=(-0.03, 1.05))
    ax2.plot(t_all, risk, color=RECON, label="published tm_risk_1s (survival model)")
    ax2.axhline(0.2, color=INK2, lw=0.9, ls=(0, (2, 3)), zorder=0)
    ax2.set(ylabel="tm_risk_1s (onset within 1 s)", ylim=(-0.03, 1.05))
    ax3.plot(t_all, b_pub, color=RECON, label="published betan (reconstruction)")
    ax3.plot(t_m, d["b_arch"][sel], "o", color=ARCH, ms=5, label="betan from archived inputs", **RING)
    ax3.plot(t_m, d["truth_bn"][sel], color=INK, lw=1.1, ls=(0, (5, 2)), label="archived truth")
    ax3.set(ylabel="betan", xlabel="time (s)")
    for ax in (ax1, ax2, ax3):
        h, _ = ax.get_legend_handles_labels()
        h += [Patch(color=VIOLET, alpha=0.3, label="archived label: mode present"),
              Patch(color=GRAY, alpha=0.3, label="_valid = 0 (input missing or out of domain)")]
        ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8,
                  ncol=3, borderaxespad=0.2)
    ax3.set_xlim(t_all[0], t_all[-1])
    fig.suptitle(f"Shot {shot} - {title}", x=0.01, ha="left", fontsize=13, fontweight="semibold")
    # The example shots are picked from the drawn subset (`lead_times` filters
    # by KEEP), so a held-out rendering picks held-out example shots and the
    # file names differ between subsets. Say so on the figure.
    footer(fig, f"{note} Subset {SUBSET}: {SUBSET_NOTE}.")
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.subplots_adjust(hspace=0.44); fig.savefig(OUT / f"05{tag}_example_shot_{shot}.png"); plt.close(fig)


# lead time per shot for each model: first crossing minus archived onset
def lead_times(prob_key, thresh, valid_key):
    out = {}
    vv = d[valid_key].astype(bool) & KEEP
    for shot in np.unique(d["shot"][KEEP]):
        sel = (d["shot"] == shot) & KEEP
        tm = d["truth_tm"][sel].astype(bool)
        if not tm.any():
            continue
        onset = float(d["t"][sel][np.argmax(tm)])
        p, tt, ok = d[prob_key][sel], d["t"][sel], vv[sel]
        cross = np.flatnonzero(ok & np.isfinite(p) & (p >= thresh))
        out[int(shot)] = (onset - float(tt[cross[0]])) if cross.size else None
    return out


lead_cnn = lead_times("p_recon", 0.5, "valid")
lead_dsm = lead_times("tm_risk_1s", 0.2, "dsm_valid")
early = sorted((v, s) for s, v in lead_cnn.items() if v is not None and v > 0.3)
late = sorted((v, s) for s, v in lead_cnn.items() if v is not None and v < -0.3)
ex1 = early[len(early) // 2][1] if early else int(d["per_shot"][0, 0])
ex2 = late[len(late) // 2][1] if late else int(d["per_shot"][1, 0])
print("examples:", ex1, lead_cnn[ex1], "|", ex2, lead_cnn[ex2])
example_shot(ex1, "a", "the CNN calls the mode before the archived label does",
             "Violet: the archive's own tearing label and the first row it calls a mode. Matched rows exist only "
             "where the upstream training filter kept the row. The label at t uses inputs averaged over [t-50 ms, t].")
example_shot(ex2, "b", "the CNN calls the mode only after it is already there",
             "Same overlays. A late crossing is not necessarily a model error: the archived label is itself a "
             "25 ms series whose onset timing has not been audited.")

# ---------- 6. does the risk anticipate the onset ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5))
rng = np.random.default_rng(0); order = rng.permutation(N)
neg, pos = order[~y[order]], order[y[order]]
a1.scatter(pa[neg], pr[neg], s=6, color=GRAY, alpha=0.3, lw=0, label=f"no mode ({(~y).sum():,} rows)")
a1.scatter(pa[pos], pr[pos], s=8, color=VIOLET, alpha=0.5, lw=0, label=f"mode present ({NPOS:,} rows)")
a1.plot([0, 1], [0, 1], color=AXIS, lw=0.9, zorder=0)
a1.set(xlabel="tm_prob from archived inputs", ylabel="tm_prob from the reconstruction", xlim=(0, 1), ylim=(0, 1))
a1.set_title("What the reconstruction does to the CNN, row by row")
a1.legend(loc="upper left", frameon=True, facecolor=SURF, edgecolor="none", framealpha=0.9)
bins = np.arange(0.0, 3.01, 0.25)
mid = (bins[:-1] + bins[1:]) / 2
keep = pre_onset & np.isfinite(d["tm_risk_1s"])
tt, rk = tto[keep], d["tm_risk_1s"][keep]
pc = d["p_recon"][keep]
which = np.digitize(tt, bins) - 1
for vals, c, name in ((rk, VIOLET, "survival tm_risk_1s"), (pc, RECON, "CNN tm_prob")):
    med = np.array([np.median(vals[which == b]) if (which == b).any() else np.nan for b in range(len(mid))])
    q1 = np.array([np.percentile(vals[which == b], 25) if (which == b).any() else np.nan for b in range(len(mid))])
    q3 = np.array([np.percentile(vals[which == b], 75) if (which == b).any() else np.nan for b in range(len(mid))])
    a2.plot(mid, med, "-o", color=c, ms=6, label=name, **RING)
    a2.fill_between(mid, q1, q3, color=c, alpha=0.15, lw=0)
a2.invert_xaxis()
a2.set(xlabel="time until the archived onset (s)", ylabel="predicted value", ylim=(0, 1))
a2.set_title("Both rise as the onset approaches (median, quartile band)")
a2.legend(loc="upper left")
fig.suptitle("Row by row: the reconstruction penalty, and whether the risk anticipates", x=0.01, ha="left", fontsize=13, fontweight="semibold")
footer(fig, SUB + f". Right: {int(keep.sum()):,} pre-onset valid rows of the {int(np.isfinite(d['per_shot'][:, 6]).sum())} shots that have an onset, binned by time until it.")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "06_prediction_shift.png"); plt.close(fig)

# ---------- 7. coverage and lead time ----------
ps = d["per_shot"][{"all": slice(None), "held_out": d["per_shot"][:, 8] < 0.5,
                    "in_training": d["per_shot"][:, 8] > 0.5}[SUBSET]]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
b = np.linspace(0, 1, 21)
a1.hist(ps[:, 4], bins=b, histtype="step", color=ARCH, lw=1.9, label=f"tearing CNN - median {np.median(ps[:, 4]):.0%}")
a1.hist(ps[:, 5], bins=b, histtype="step", color=VIOLET, lw=1.9, label=f"survival model - median {np.median(ps[:, 5]):.0%}")
a1.set(xlabel="fraction of the 240 timesteps published as valid", ylabel="shots", xlim=(0, 1))
a1.set_title("Published coverage per shot")
a1.legend(loc="upper center")
lb = np.arange(-3.0, 3.01, 0.25)
lc = np.array([v for v in lead_cnn.values() if v is not None])
ld = np.array([v for v in lead_dsm.values() if v is not None])
a2.hist(lc, bins=lb, histtype="step", color=ARCH, lw=1.9,
        label=f"CNN tm_prob at 0.5 - median {np.median(lc):+.2f} s ({lc.size} shots)")
a2.hist(ld, bins=lb, histtype="step", color=VIOLET, lw=1.9,
        label=f"survival risk at 0.2 - median {np.median(ld):+.2f} s ({ld.size} shots)")
a2.axvline(0.0, color=INK2, lw=1.0, ls=(0, (2, 3)))
top = max(np.histogram(lc, bins=lb)[0].max(), np.histogram(ld, bins=lb)[0].max())
a2.set_ylim(0, top * 1.45)
a2.text(0.12, top * 0.62, "fires before\nthe archived onset ->", fontsize=8, color=INK2, va="top")
a2.set(xlabel="lead time: archived onset minus first threshold crossing (s)", ylabel="shots")
a2.set_title("How early each model calls the onset")
a2.legend(loc="upper right")
fig.suptitle("Coverage, and lead time on the shots that have an onset", x=0.01, ha="left", fontsize=13, fontweight="semibold")
footer(fig, f"{NSHOTS} aligned shots; lead time on the {int(np.isfinite(ps[:, 6]).sum())} with an archived onset, "
       "counting only shots whose label crosses its threshold at all. Positive means the model called it early.")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "07_coverage_and_lead_time.png"); plt.close(fig)

summary = {
    "dsm_slug": DSM, "subset": SUBSET, "subset_note": SUBSET_NOTE,
    "n_shots_in_training": int((d["per_shot"][:, 8] > 0.5).sum()),
    "n_shots": NSHOTS, "n_rows_cnn_valid": int(N), "base_rate": float(BASE),
    "cnn_auroc_archived": binary_metrics(pa, y)["auroc"],
    "cnn_auroc_reconstructed": binary_metrics(pr, y)["auroc"],
    "cnn_best_f1": {k: [float(v[0]), float(v[1])] for k, v in best.items()},
    "dsm": {r: {"auroc": binary_metrics(*[DSM_ROWS[r][0], DSM_ROWS[r][1].astype(float)])["auroc"],
                "positive_rate": float(DSM_ROWS[r][1].mean()), "n": int(DSM_ROWS[r][0].size)}
            for r in RISKS},
    "lead_time_median_s": {"cnn": float(np.median(lc)), "dsm": float(np.median(ld))},
    "lead_time_fires_early_fraction": {"cnn": float((lc > 0).mean()), "dsm": float((ld > 0).mean())},
    "coverage_median": {"cnn": float(np.median(ps[:, 4])), "dsm": float(np.median(ps[:, 5]))},
    "examples": {"early": int(ex1), "late": int(ex2)},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=1))
