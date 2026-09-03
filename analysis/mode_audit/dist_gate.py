"""3e — DISTRIBUTIONAL GATE: does the world model FORECAST modes?

The audit proved exact-code CE collapses at LOW loss (argmax capture 0.00), so
"prediction loss ~ 0" is the WRONG target. The right success criterion is
DISTRIBUTIONAL: the model's SAMPLED renders must (1) fire the mode detector at
~GT rate, (2) at the RIGHT frequency, (3) with matching band-power. Codec- and
model-agnostic: operates on decoded (pred, gt) spectrogram batches (N,C,F,T),
where pred[i] is the model's forecast of the window gt[i] actually is.

Reuses the audit's band-prominence detector (identical constants to gate.py).

Metrics (per modality):
  fire_cut          : P75 of GT best-channel band-prominence (fixes the GT-active set).
  fire_recall       : on GT-active windows, frac where PRED also fires  (>=0.5 = the
                      Branch-3 kill-criterion gate: model fires >=50% of GT rate = NOT collapsed).
  fire_rate_gt/pred : frac of ALL windows that fire (collapse if pred<<gt).
  freq_in_tol       : on both-fire windows, frac with |peakbin_pred - peakbin_gt| <= TOL.
  bandpower_pearson : median Pearson r of the 5-40 kHz band-profile (pred vs gt), GT-active.
  persistence_pred  : if `consecutive`, frac of adjacent PRED windows whose fired
                      peak-freq agrees within TOL (samples persist, not speckle).

PASS (pre-registered): fire_recall>=0.5 AND freq_in_tol>=0.7 AND bandpower_pearson>=0.5.

CLI:  python dist_gate.py pred.pt gt.pt [--consecutive]
      (each .pt = a (N,C,F,T) float tensor; aligned index-for-index.)
"""
import argparse
import json
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

FS, NFFT = 500_000.0, 1024
DF = FS / NFFT / 1e3
MODE_LO, MODE_HI = int(round(5.0 / DF)), int(round(40.0 / DF))  # 5-40 kHz band
TOL_BINS = int(round(1.0 / DF))                                 # +-1 kHz freq tolerance

GATE = {"fire_recall": 0.50, "freq_in_tol": 0.70, "bandpower_pearson": 0.50}


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def band_prom(x_ch):
    """x_ch (F,T) -> (prominence_profile over band, global peak bin, peak value)."""
    prof = np.abs(x_ch[MODE_LO:MODE_HI]).mean(1)
    pd = prof - gaussian_filter1d(prof, 6.0)
    return pd, MODE_LO + int(np.argmax(pd)), float(pd.max())


def win_P(x):
    """x (C,F,T) -> best-channel band-peak prominence (the window 'fire strength')."""
    return max(band_prom(x[c])[2] for c in range(x.shape[0]))


def strong_ch(x):
    return int(np.argmax([band_prom(x[c])[2] for c in range(x.shape[0])]))


def band_profile(x_ch):
    prof = np.abs(x_ch[MODE_LO:MODE_HI]).mean(1)
    return prof - gaussian_filter1d(prof, 6.0)


def distributional_gate(pred, gt, fire_pct=75.0, tol_bins=TOL_BINS, consecutive=False):
    """pred,gt : (N,C,F,T). Returns the metric dict + PASS booleans."""
    pred = _to_np(pred)
    gt = _to_np(gt)
    assert pred.shape == gt.shape, f"pred {pred.shape} != gt {gt.shape}"
    N = pred.shape[0]
    gtP = np.array([win_P(gt[i]) for i in range(N)])
    prP = np.array([win_P(pred[i]) for i in range(N)])
    cut = float(np.percentile(gtP, fire_pct))
    gt_active = gtP >= cut
    pred_fire = prP >= cut
    n_act = int(gt_active.sum())

    fire_recall = float(np.mean(pred_fire[gt_active])) if n_act else float("nan")
    both = gt_active & pred_fire

    # freq-in-tol on both-fire windows, compared on GT's strongest channel
    ft = []
    for i in np.where(both)[0]:
        ch = strong_ch(gt[i])
        _, f_gt, _ = band_prom(gt[i, ch])
        _, f_pr, _ = band_prom(pred[i, ch])
        ft.append(abs(f_pr - f_gt) <= tol_bins)
    freq_in_tol = float(np.mean(ft)) if ft else float("nan")

    # band-power profile correlation on GT-active windows (GT strong channel)
    bp = []
    for i in np.where(gt_active)[0]:
        ch = strong_ch(gt[i])
        a = band_profile(gt[i, ch])
        b = band_profile(pred[i, ch])
        if a.std() > 1e-9 and b.std() > 1e-9:
            bp.append(float(np.corrcoef(a, b)[0, 1]))
    bandpower_pearson = float(np.median(bp)) if bp else float("nan")

    res = {
        "n_windows": N,
        "n_gt_active": n_act,
        "fire_cut": cut,
        "fire_rate_gt": float(np.mean(gt_active)),
        "fire_rate_pred": float(np.mean(pred_fire)),
        "fire_recall": fire_recall,
        "freq_in_tol": freq_in_tol,
        "bandpower_pearson": bandpower_pearson,
        "tol_bins": tol_bins,
    }

    if consecutive:  # do fired PRED modes persist window-to-window (not speckle)?
        agree = []
        for i in range(N - 1):
            if pred_fire[i] and pred_fire[i + 1]:
                ch = strong_ch(pred[i])
                _, f0, _ = band_prom(pred[i, ch])
                _, f1, _ = band_prom(pred[i + 1, ch])
                agree.append(abs(f1 - f0) <= tol_bins)
        res["persistence_pred"] = float(np.mean(agree)) if agree else float("nan")

    res["checks"] = {k: (res[k] >= v) for k, v in GATE.items()}
    res["PASS"] = bool(all(res["checks"].values()))
    return res


def _summary(res):
    return (f"fire_recall={res['fire_recall']:.3f} (gt_rate={res['fire_rate_gt']:.2f} "
            f"pred_rate={res['fire_rate_pred']:.2f}) | freq_in_tol={res['freq_in_tol']:.3f} "
            f"| bandpower_r={res['bandpower_pearson']:.3f}"
            + (f" | persist={res.get('persistence_pred', float('nan')):.3f}" if "persistence_pred" in res else "")
            + f"  ==> {'PASS' if res['PASS'] else 'FAIL'}")


if __name__ == "__main__":
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("pred"); ap.add_argument("gt")
    ap.add_argument("--consecutive", action="store_true")
    ap.add_argument("--fire_pct", type=float, default=75.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pred = torch.load(a.pred, map_location="cpu")
    gt = torch.load(a.gt, map_location="cpu")
    res = distributional_gate(pred, gt, fire_pct=a.fire_pct, consecutive=a.consecutive)
    print("[dist_gate] " + _summary(res))
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
