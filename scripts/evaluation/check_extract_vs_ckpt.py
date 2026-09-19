"""Cross-check extract_latents output against ckpt['metrics'].

Pools per-window (mae, weight) from every npz in a latents directory —
exactly the trainer's masked-MAE aggregation — and prints model/copy MAE
beside the training-time values stored in the checkpoint. Magnitudes should
agree to within the window-mix difference (our 0.25 s stride, t ≥ 1 s vs
training's full 0.01 s-stride val sweep); order-of-magnitude disagreement
means a semantics bug, and the sweep must not proceed.

Usage::

    python scripts/evaluation/check_extract_vs_ckpt.py \
        data/outputs/eval_suite/e2e_stage1_best/latents/val
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))

from tfm_eval.ckpt import DEFAULT_CKPT, load_ckpt  # noqa: E402


def main() -> int:
    lat_dir = Path(sys.argv[1])
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CKPT
    files = sorted(lat_dir.glob("*.npz"))
    if not files:
        print(f"no npz files in {lat_dir}")
        return 1

    num = den = cnum = cden = None
    diag_names = None
    n_windows = 0
    for f in files:
        z = np.load(f, allow_pickle=False)
        if diag_names is None:
            diag_names = [str(s) for s in z["diag_names"]]
            num = np.zeros(len(diag_names)); den = np.zeros(len(diag_names))
            cnum = np.zeros(len(diag_names)); cden = np.zeros(len(diag_names))
        mae, w = z["mae"], z["mae_w"]
        cm, cw = z["copy_mae"], z["copy_mae_w"]
        ok = np.isfinite(mae) & (w > 0)
        cok = np.isfinite(cm) & (cw > 0)
        num += np.where(ok, mae * w, 0).sum(0)
        den += np.where(ok, w, 0).sum(0)
        cnum += np.where(cok, cm * cw, 0).sum(0)
        cden += np.where(cok, cw, 0).sum(0)
        n_windows += mae.shape[0]

    ck_metrics = load_ckpt(ckpt_path).get("metrics", {}) or {}
    print(f"{len(files)} shots, {n_windows} windows from {lat_dir}\n")
    print(f"{'modality':>22s} {'mae(sweep)':>11s} {'mae(ckpt)':>10s} "
          f"{'copy(sweep)':>12s} {'copy(ckpt)':>11s} {'ratio(sweep)':>13s} "
          f"{'ratio(ckpt)':>12s}")
    # Absolute MAE shifts with shot/window mix, but two invariants hold if
    # the metric semantics match training: (1) the model/copy ratio stays
    # close to the checkpoint's, (2) model and copy MAE scale by the SAME
    # per-modality factor (activity level moves both together).
    worst_ratio_dev = 0.0
    worst_factor_skew = 1.0
    for j, name in enumerate(diag_names):
        mae = num[j] / den[j] if den[j] > 0 else float("nan")
        cmae = cnum[j] / cden[j] if cden[j] > 0 else float("nan")
        ck = ck_metrics.get(name, {})
        ck_m, ck_c = ck.get("model_mae", float("nan")), ck.get(
            "copy_mae", float("nan"))
        ratio = mae / cmae if cmae and np.isfinite(cmae) else float("nan")
        ck_r = ck_m / ck_c if ck_c else float("nan")
        if all(np.isfinite(v) and v > 0 for v in (mae, cmae, ck_m, ck_c)):
            worst_ratio_dev = max(worst_ratio_dev, abs(ratio - ck_r))
            skew = (mae / ck_m) / (cmae / ck_c)
            worst_factor_skew = max(worst_factor_skew,
                                    max(skew, 1.0 / skew))
        print(f"{name:>22s} {mae:11.4f} {ck_m:10.4f} {cmae:12.4f} "
              f"{ck_c:11.4f} {ratio:13.3f} {ck_r:12.3f}")
    ok = worst_ratio_dev < 0.2 and worst_factor_skew < 1.6
    print(f"\nworst |ratio - ckpt ratio| = {worst_ratio_dev:.3f} (<0.2), "
          f"worst model/copy factor skew = {worst_factor_skew:.2f}x (<1.6)"
          f" → {'PASS — semantics match training' if ok else 'FAIL — investigate'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
