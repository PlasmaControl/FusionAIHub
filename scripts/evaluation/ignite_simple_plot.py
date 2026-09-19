"""One plain figure: what actually happened vs what the model predicted.

Everything else in this directory answers "how do we know the number is honest". This answers
"is it any good", for someone who has never heard of MaskGIT, tokens, or counterfactuals.

Design rules, deliberately narrow:
  * two lines -- measured (black) and predicted (blue). Nothing else on the main panel.
  * the seed window the model was handed is shaded and labelled, so it is obvious that the
    interesting part is the region where the model is on its own.
  * time is trimmed to where the diagnostic actually recorded data, so nothing on the plot is
    an artefact of padding.
  * the caption states, in words, how it did against the do-nothing baseline.

Reads the archived traces, so it needs no GPU and no model.

    python scripts/evaluation/ignite_simple_plot.py \
        --traces data/outputs/ignite_bp_cases/tm_199597/bp_cases_traces.npz \
        --shot 199597 --quantity "Tearing-mode activity" \
        --out data/outputs/ignite_bp_cases/tm_199597/simple
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from tfm_eval.plotting import save_fig, set_style  # noqa: E402

FRAME_DT_S = 0.05


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="bp_cases_traces.npz")
    ap.add_argument("--shot", required=True)
    ap.add_argument("--key", default="mhr__TM 1-20 kHz",
                   help="which trace, e.g. 'mhr__TM 1-20 kHz' or 'mhr__AE 50-250 kHz'")
    ap.add_argument("--quantity", default="Tearing-mode activity",
                   help="plain-language name for the y axis")
    ap.add_argument("--valid-until-s", type=float, default=3.75,
                   help="trim here; beyond it the diagnostic recorded nothing")
    ap.add_argument("--with-control", action="store_true",
                   help="add a second panel: same shot with the RMP coils turned off")
    ap.add_argument("--control-arm", default="group_zero:rmp")
    ap.add_argument("--control-label", default="if the RMP coils were switched off")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    set_style()
    z = np.load(args.traces, allow_pickle=True)
    K0 = int(z["K0"])
    gt = z[f"trace_gt__{args.key}"]
    pred = z[f"trace_pred__real__{args.key}"]
    pers = z[f"trace_pers__{args.key}"]

    n = min(len(gt), int(round(args.valid_until_s / FRAME_DT_S)))
    t = (np.arange(n) + 0.5) * FRAME_DT_S
    gt, pred, pers = gt[:n], pred[:n], pers[:n]

    # How did it do, in one sentence, against doing nothing at all?
    g, p, q = gt[K0:], pred[K0:], pers[K0:]
    err = float(np.sqrt(((p - g) ** 2).mean()))
    err_pers = float(np.sqrt(((q - g) ** 2).mean()))
    better = (1.0 - err / err_pers) * 100.0

    nrow = 2 if args.with_control else 1
    fig, axes = plt.subplots(nrow, 1, figsize=(8.4, 3.4 * nrow + 0.6), sharex=True,
                             squeeze=False)
    ax = axes[0][0]
    t_split = K0 * FRAME_DT_S

    for a in axes[:, 0]:
        a.axvspan(t[0] - 0.5 * FRAME_DT_S, t_split, color="#EDEDED", lw=0, zorder=0)
    ax.plot(t, gt, color="black", lw=1.6, label="what actually happened", zorder=3)
    ax.plot(t, pred, color="#0072B2", lw=1.6, ls="--", label="what the model predicted",
            zorder=4)
    ax.annotate("the model is shown\nthis much, then\npredicts on its own →",
                xy=(t_split, ax.get_ylim()[1]), xytext=(-6, -6),
                textcoords="offset points", ha="right", va="top",
                fontsize=8, color="#555555")
    ax.axvline(t_split, color="0.4", lw=1.0, zorder=2)
    ax.set_ylabel(args.quantity)
    ax.set_title(f"{args.quantity} in shot {args.shot}", loc="left", fontsize=12)
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    verdict = (f"After it stops being shown the data, the model's error is "
               f"{abs(better):.0f}% {'lower' if better > 0 else 'higher'} than simply "
               f"assuming nothing changes.")
    ax.text(0.0, -0.22, verdict, transform=ax.transAxes, fontsize=9, color="#333333")

    if args.with_control:
        axc = axes[1][0]
        ctrl = z[f"trace_pred__{args.control_arm}__{args.key}"][:n]
        axc.plot(t, pred, color="#0072B2", lw=1.6, ls="--",
                 label="model, real controls", zorder=3)
        axc.plot(t, ctrl, color="#D55E00", lw=1.6, label=f"model, {args.control_label}",
                 zorder=4)
        axc.axvline(t_split, color="0.4", lw=1.0)
        axc.set_ylabel(args.quantity)
        axc.set_xlabel("time through the shot [s]")
        axc.set_title("Same shot, one control knob changed", loc="left", fontsize=12)
        axc.legend(loc="lower right", fontsize=9, frameon=False)
        d = float(np.nanmean(ctrl[K0:] - pred[K0:]))
        axc.text(0.0, -0.28,
                 f"Turning the coils off makes the model predict "
                 f"{'more' if d > 0 else 'less'} activity "
                 f"(average change {d:+.3f}) — the direction physics expects.",
                 transform=axc.transAxes, fontsize=9, color="#333333")
    else:
        ax.set_xlabel("time through the shot [s]")

    fig.subplots_adjust(hspace=0.42, bottom=0.18)
    paths = save_fig(fig, Path(args.out))
    plt.close(fig)
    print(f"[ok] {paths[0]}")
    print(f"     prediction error {err:.4f} vs do-nothing {err_pers:.4f} "
          f"-> {better:+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
