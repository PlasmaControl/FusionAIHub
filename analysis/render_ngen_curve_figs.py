"""Paper-quality generalization-curve figures for the IGNITE N-shots probe.

Renders from the DURABLE probe records (no re-training):
  * runs/N*/loss_history.jsonl                        -> final validation masked-CE per N
  * eval_runs/ignite_ngen_probe/N*/eval_metrics.json  -> held-out rollout metrics
    (within-campaign val tail, median over 16 shots)
  * eval_runs/ignite_ngen_probe/N*/shot200729/eval_metrics.json -> the cross-campaign
    benchmark shot (different era than the 190000-193005 training range)

Outputs (PDF vector + PNG, overwritten in place in eval_runs/ignite_ngen_probe/):
  fig_ngen_curve.{pdf,png}  two panels: (a) val masked-CE vs N with the chance reference;
                            (b) rollout nRMSE vs training FRAMES on the held-out
                            validation shots (persistence drawn as a dotted reference). The cross-campaign shot 200729 is EXCLUDED by default
                            (user 2026-08-09); pass --cross to add it for diagnostics.

Usage:  python analysis/render_ngen_curve_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODELS = Path("/lustre/orion/fus187/proj-shared/models")
# Two probes. v1 = WITHIN-campaign (train shots drawn from one era) — the arms that showed
# clean data scaling but collapsed on a different campaign. v2 = RANDOM over ALL campaigns,
# which is what production does; v2 is the representative study and the default here.
PROBES = {
    "v1": dict(runs=MODELS / "ignite_ngen_probe/runs", ev=REPO / "eval_runs/ignite_ngen_probe",
               cross=True, label="within-campaign"),
    "v2": dict(runs=MODELS / "ignite_ngen_probe_rand/runs",
               ev=REPO / "eval_runs/ignite_ngen_probe_rand",
               cross=False, label="random over all campaigns"),
}
NS = (10, 100, 1000)
INK, INK2 = "#1a1a19", "#5f5e56"
C_SLOW, C_VID = "#2a78d6", "#eb6834"       # validated categorical palette
CHANCE_CE = 8.1                             # mean ln(vocab) over the 1593-token frame


def _final_val_ce(runs: Path, n: int) -> float:
    # Match on the PARSED key, never the raw text: the best-checkpoint logger emits
    # {"step": .., "best_val_loss": ..} records whose text also contains "val_loss".
    vals = [d["val_loss"]
            for d in (json.loads(l) for l in
                      (runs / f"N{n}/loss_history.jsonl").read_text().splitlines() if l.strip())
            if "val_loss" in d]
    return vals[-1]


def _rollout(ev: Path, n: int, sub: str = ""):
    """(slow, video, slow_pers, video_pers) token accuracy — median over the eval shots.

    Persistence (freeze the last seed frame) is absent from probe-v1 records, which predate
    the baseline; those entries come back None and the reference lines are simply not drawn.
    """
    m = json.loads((ev / f"N{n}{sub}/eval_metrics.json").read_text())
    ps = m["per_shot"]
    slow_keys = tuple(k for k in m["mean_token_accuracy"]
                      if k.startswith(("ts_", "cer", "mse")))
    vid_keys = ("tangtv_lower", "tangtv_upper")

    def dynamic(s: str, k: str) -> bool:
        """Is modality k actually CHANGING in shot s over the rollout window?

        A missing/parked diagnostic encodes to a frozen code sequence, which hands BOTH the
        model and persistence a free 1.000 and compresses the measurable margin toward zero.
        9/16 shots in the v2 eval set have frozen video. Such (shot, modality) pairs are
        dropped per-modality — a shot may be dynamic in Thomson and static in video.
        """
        pf = ps[s].get("token_accuracy_persistence")
        return pf is None or pf[k] < 0.999

    def med(keys, field):
        if not all(field in ps[s] for s in ps):
            return None
        vals = []
        for s in ps:
            # nRMSE exists only for EVAL_MODALITIES (the 11 with a decoder in the eval), while
            # token accuracy covers all 14 — so intersect with the field's OWN keys rather
            # than assuming the two dicts have the same modality set.
            live = [k for k in keys if k in ps[s][field] and dynamic(s, k)]
            if live:
                v = [ps[s][field][k] for k in live]
                v = [x for x in v if np.isfinite(x)]
                if v:
                    vals.append(np.mean(v))
        return float(np.median(vals)) if vals else None
    # Panel (b) plots nRMSE (user 2026-08-12): token accuracy measures agreement in CODE
    # space, which says nothing about the size of the physical error. nRMSE is in the decoded
    # signal's own units and is what a reader can judge. Static screening matters MORE here,
    # not less: a frozen diagnostic has a constant GT, so its nRMSE is exactly 0 — a perfect
    # score for predicting nothing, which would drag the median down hard.
    return (med(slow_keys, "nrmse"), med(vid_keys, "nrmse"),
            med(slow_keys, "nrmse_persistence"), med(vid_keys, "nrmse_persistence"))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", action="store_true",
                    help="also draw the cross-campaign benchmark shot (off by default)")
    ap.add_argument("--probe", choices=sorted(PROBES), default="v2",
                    help="v2 (default) = random over ALL campaigns, the representative "
                         "study; v1 = the older within-campaign probe")
    args = ap.parse_args()
    cfg = PROBES[args.probe]
    RUNS, EV = cfg["runs"], cfg["ev"]
    if args.cross and not cfg["cross"]:
        raise SystemExit(f"--cross has no meaning for probe {args.probe} "
                         "(it samples all campaigns; there is no held-out era shot)")
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    # X AXIS = TRAINING FRAMES, not shots (user 2026-08-12). Shots are an arbitrary unit —
    # a shot is only as informative as it is long — so the data-scaling claim belongs on the
    # amount of DATA. Frames/shot is read from the cache the probe actually trained on rather
    # than assumed. (Windows would double-count: a stride-1 window per frame overlaps 59/60
    # with its neighbour, so frames is the honest measure of distinct data.)
    import torch
    _fc = Path(str(RUNS).replace("/runs", "/frame_codes"))
    _f = sorted(p for p in _fc.glob("*.pt") if not p.stem.startswith("_"))
    frames_per_shot = int(torch.load(_f[0], map_location="cpu")["n_frames"]) if _f else 219
    XS = [n * frames_per_shot for n in NS]

    ce = [_final_val_ce(RUNS, n) for n in NS]
    within = [_rollout(EV, n) for n in NS]
    cross = [_rollout(EV, n, "/shot200729") for n in NS] if cfg["cross"] else None

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # (a) validation masked-CE vs N
    ax_a.plot(XS, ce, "o-", color=INK, lw=1.3, ms=4, mec="white", mew=0.6)
    ax_a.axhline(CHANCE_CE, color=INK2, ls=":", lw=0.9)
    ax_a.annotate("chance", (XS[0], CHANCE_CE), xytext=(0, 4),
                  textcoords="offset points", fontsize=6.5, color=INK2)
    for n, v in zip(XS, ce):
        ax_a.annotate(f"{v:.2f}", (n, v), xytext=(5, 3), textcoords="offset points",
                      fontsize=6.5, color=INK2)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("training frames")
    ax_a.set_ylabel("validation masked-token CE")
    ax_a.set_title("(a) held-out masked prediction", fontsize=8)
    ax_a.grid(alpha=0.25, which="both", lw=0.4)

    # (b) rollout token accuracy vs N (held-out validation shots). The cross-campaign
    # benchmark shot is EXCLUDED by default (user 2026-08-09: not representative — the
    # probe trained on one campaign; production trains on all); --cross re-adds it.
    series = [(within, "-", None)]
    if args.cross:
        series.append((cross, "--", "200729"))
    for vals, ls, _tag in series:
        ax_b.plot(XS, [v[0] for v in vals], ls, color=C_SLOW, lw=1.3, marker="o",
                  ms=3.5, mec="white", mew=0.5)
        ax_b.plot(XS, [v[1] for v in vals], ls, color=C_VID, lw=1.3, marker="o",
                  ms=3.5, mec="white", mew=0.5)
    # Persistence reference (freeze the last seed frame). It depends only on the eval shots
    # and K0, not on the model, so it is one flat line per family; a curve is only meaningful
    # as the MARGIN above it. Absent for probe v1, whose records predate the baseline.
    pers = {}
    for idx, col, tag in ((2, C_SLOW, "slow-TS"), (3, C_VID, "video")):
        vals = [v[idx] for v in within if v[idx] is not None]
        if vals:
            pers[tag] = float(np.mean(vals))
            ax_b.axhline(pers[tag], color=col, ls=":", lw=1.0, alpha=0.85)
    handles = [plt.Line2D([], [], color=C_SLOW, lw=1.3, label="slow-TS"),
               plt.Line2D([], [], color=C_VID, lw=1.3, label="video")]
    if pers:
        handles.append(plt.Line2D([], [], color=INK2, lw=1.0, ls=":", label="persistence"))
    if args.cross:
        handles += [plt.Line2D([], [], color=INK, lw=1.1, ls="-", label="held-out"),
                    plt.Line2D([], [], color=INK, lw=1.1, ls="--", label="shot 200729")]
    ax_b.legend(handles=handles, frameon=False, ncol=2, loc="upper left",
                columnspacing=0.9, handlelength=1.6)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("training frames")
    ax_b.set_ylabel("rollout nRMSE")
    ax_b.set_title("(b) 2 s autoregressive rollout", fontsize=8)
    ax_b.grid(alpha=0.25, which="both", lw=0.4)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(EV / f"fig_ngen_curve.{ext}", dpi=300)
    plt.close(fig)
    print(f"NGEN_CURVE_OK [{args.probe}: {cfg['label']}] -> {EV}/fig_ngen_curve.[pdf|png]")
    print("  val CE:", dict(zip(NS, [round(v, 3) for v in ce])))
    print(f"  frames/shot {frames_per_shot} -> x = {XS}")
    print("  held-out nRMSE (slowts, video):",
          {n: (round(v[0], 3) if v[0] else None, round(v[1], 3) if v[1] else None)
           for n, v in zip(NS, within)})
    if pers:
        print("  persistence:", {k: round(v, 3) for k, v in pers.items()})
        print("  NOTE: static (shot, modality) pairs dropped — frozen diagnostics give both "
              "model and persistence a free 1.000")
    else:
        print("  WARNING: no persistence field in these records (probe v1 predates it), so "
              "STATIC diagnostics could NOT be screened out — curves here are contaminated "
              "by shots whose modality never changes. Compare magnitudes with v2 only.")
    if cross:
        print("  200729 (slowts, video):",
              {n: (round(v[0], 3), round(v[1], 3)) for n, v in zip(NS, cross)})


if __name__ == "__main__":
    main()
