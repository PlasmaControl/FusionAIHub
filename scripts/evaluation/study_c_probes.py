"""Study C: downstream-task linear probes on frozen world-model latents.

Fits linear probes on the pooled latent caches written by
``extract_latents.py``: TRAIN-split windows fit the probe, ALL VAL-split
windows score it, with GroupKFold-by-shot hyperparameter selection so no
shot leaks across CV folds (``tfm_eval.probing``). Labels come from the
``study_c_make_labels.py`` window CSVs, joined on ``(shot, t_start_s)``.

Targets: ``elm_next`` (ELM in the next 50 ms window; AUROC/AP),
``mirnov_bp`` / ``mhr_bp`` (concurrent log10 magnetics band power; R2 and
Spearman), and optionally ``log10_n1rms_now`` (``--n1rms-csv-stem``).

Steps (select with ``--only``):
  * ``headline``   — diag_global probe per target vs persistence /
    prevalence / physics-scalar baselines; ROC+PR figure (c0) and
    ``tables/c_probe_metrics.json``.
  * ``slices``     — ridge probe per pooled slice x target
    (``tables/c_slice_sweep.csv`` + figure c1).
  * ``efficiency`` — probe score vs number of labeled train shots
    (``tables/c_sample_efficiency.npz`` + figure c2).
  * ``timeline``   — hero-shot D-alpha + probe P(elm_next) timeline (c3).

CPU-only; never launches GPU work. Run from the repo root on a login node
(~45-60 min and ~8 GB RAM at real-data defaults; the slice sweep dominates):
    python scripts/evaluation/study_c_probes.py
Smoke test (2 cached shots; numbers are garbage, only artifacts matter):
    python scripts/evaluation/study_c_probes.py --smoke
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from tfm_eval.plotting import (  # noqa: E402
    OKABE_ITO,
    efficiency_curve,
    save_fig,
    set_style,
)
from tfm_eval.probing import (  # noqa: E402
    join_window_labels,
    load_latent_cache,
    logistic_probe,
    ridge_probe,
    rows_for_shots,
    subsample_shots,
)

_OUT_DEFAULT = (_HERE.parents[2] / "data" / "outputs" / "eval_suite"
                / "e2e_stage1_best")
_SMOKE_ROOT = _OUT_DEFAULT.parent / "e2e_stage1_best_smoke"

STEPS = ("headline", "slices", "efficiency", "timeline")
HEADLINE_KEY = "diag_global"
WINDOW_S = 0.05  # label window length (s) — study_c_make_labels grid
LABEL_COLS = ["elm_now", "elm_next", "elm_in_range", "dalpha_max_now",
              "mirnov_bp", "mhr_bp"]
METRIC_COLS = ("auroc", "ap", "r2", "spearman", "best_alpha", "cv_score")
TARGET_TITLES = {
    "elm_next": "ELM in next window",
    "mirnov_bp": "Mirnov band power",
    "mhr_bp": "MHR band power",
    "log10_n1rms_now": "n=1 mode amplitude",
}
REAL_NS = (5, 10, 25, 50, 100, 200, 400)
SMOKE_NS = (1, 2)
SMOKE_TRAIN_SHOT = 191599  # ELMy — fits the smoke probe
SMOKE_EVAL_SHOT = 204510
RIDGE_SPLITS = 3  # 3 CV folds + refit = ~4 SVDs per ridge_probe
MIN_ROWS = 5  # skip probes with fewer usable rows on either side
GRAY = "#7F7F7F"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-latents",
                    default=str(_OUT_DEFAULT / "latents" / "val"))
    ap.add_argument("--train-latents",
                    default=str(_OUT_DEFAULT / "latents" / "train"))
    ap.add_argument("--labels-dir", default=str(_OUT_DEFAULT / "labels"))
    ap.add_argument("--out-root", default=str(_OUT_DEFAULT))
    ap.add_argument(
        "--n1rms-csv-stem", default=None,
        help="stem of <labels-dir>/<stem>_{split}.csv (e.g. n1rms_windows); "
             "adds the log10_n1rms_now regression target",
    )
    ap.add_argument(
        "--max-train-rows", type=int, default=20000,
        help="seeded row cap applied after shot selection (bounds SVD cost)",
    )
    ap.add_argument("--seeds", type=int, default=5,
                    help="efficiency-curve repeats per train-set size")
    ap.add_argument("--smoke", action="store_true",
                    help="2-shot smoke cache: garbage numbers, all artifacts")
    ap.add_argument("--only", default=",".join(STEPS),
                    help=f"comma list from {{{','.join(STEPS)}}}")
    return ap.parse_args()


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_HERE.parents[2]), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _plain(obj):
    """Recursively coerce numpy scalars so json holds plain floats/ints."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def _subset_cache(cache, keep):
    """Row-subset every per-window array of a load_latent_cache() dict."""
    out = dict(cache)
    for k in ("shot", "t_start_s", "local_idx", "mae", "mae_w", "copy_mae",
              "copy_mae_w", "diag_valid", "act_valid", "physics"):
        out[k] = cache[k][keep]
    out["pooled"] = {k: v[keep] for k, v in cache["pooled"].items()}
    out["n_shots"] = int(np.unique(out["shot"]).size)
    return out


def control_features(physics):
    """24-dim physics-scalar control: NaN->0 values + finite indicators."""
    finite = np.isfinite(physics)
    vals = np.where(finite, physics, 0.0).astype(np.float64)
    return np.concatenate([vals, finite.astype(np.float64)], axis=1)


def cap_rows(idx, cap, seed):
    """Seeded subsample of row indices down to ``cap`` (order preserved)."""
    if cap and idx.size > cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    return idx


def binary_scores(y_true, y_score):
    if np.unique(y_true).size < 2:
        return {"auroc": float("nan"), "ap": float("nan")}
    return {"auroc": float(roc_auc_score(y_true, y_score)),
            "ap": float(average_precision_score(y_true, y_score))}


def logistic_proba(X_tr, y_tr, X_te, C, max_iter=2000):
    """P(y=1) on X_te from a refit that mirrors probing.logistic_probe's
    final model exactly (same standardization, same C, deterministic)."""
    from sklearn.linear_model import LogisticRegression

    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit((X_tr - mu) / sd, y_tr)
    return clf.predict_proba((X_te - mu) / sd)[:, 1]


def main() -> int:
    args = parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    steps = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise SystemExit(f"--only: unknown step(s) {sorted(unknown)}")

    if args.smoke:
        args.val_latents = args.train_latents = str(
            _SMOKE_ROOT / "latents" / "val")
        if args.out_root == str(_OUT_DEFAULT):
            args.out_root = str(_SMOKE_ROOT)
        args.seeds = 2
    ns_grid = SMOKE_NS if args.smoke else REAL_NS
    cs_grid = (0.1,) if args.smoke else (0.01, 0.1, 1.0)

    out_root = Path(args.out_root)
    tables_dir = out_root / "tables"
    fig_dir = out_root / "figures" / "study_c"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = Path(args.labels_dir)
    set_style()

    # ------------------------------------------------------------- caches
    pooled_keys = None if "slices" in steps else [HEADLINE_KEY]
    if args.smoke:  # one cache dir, split by shot: 191599 fits, 204510 evals
        cache = load_latent_cache(args.val_latents, pooled_keys=pooled_keys)
        tr_cache = _subset_cache(cache, cache["shot"] == SMOKE_TRAIN_SHOT)
        va_cache = _subset_cache(cache, cache["shot"] == SMOKE_EVAL_SHOT)
        del cache
    else:
        tr_cache = load_latent_cache(args.train_latents,
                                     pooled_keys=pooled_keys)
        va_cache = load_latent_cache(args.val_latents,
                                     pooled_keys=pooled_keys)
    log(f"caches: train {tr_cache['shot'].size} windows / "
        f"{tr_cache['n_shots']} shots; val {va_cache['shot'].size} windows "
        f"/ {va_cache['n_shots']} shots")

    # ------------------------------------------------------------- labels
    def load_labels(cache, split):
        # Both smoke shots come from the val split, so smoke joins the val
        # CSV on both sides of the probe.
        csv_split = "val" if args.smoke else split
        vals, joined = join_window_labels(
            cache, labels_dir / f"windows_{csv_split}.csv", LABEL_COLS)
        vals["_joined"] = joined
        if args.n1rms_csv_stem:
            p = labels_dir / f"{args.n1rms_csv_stem}_{csv_split}.csv"
            if p.exists():
                n1, _ = join_window_labels(
                    cache, p, ["log10_n1rms_now", "n1rms_present"])
                vals.update(n1)
            else:
                log(f"WARNING: {p} missing — n1rms target dropped")
        return vals

    tr_lab = load_labels(tr_cache, "train")
    va_lab = load_labels(va_cache, "val")
    log(f"labels joined: train {int(tr_lab['_joined'].sum())}, "
        f"val {int(va_lab['_joined'].sum())} windows")

    # ------------------------------------------------------------ targets
    targets = [
        {"name": "elm_next", "cls": True, "gate": "elm_in_range"},
        {"name": "mirnov_bp", "cls": False, "gate": None},
        {"name": "mhr_bp", "cls": False, "gate": None},
    ]
    if args.n1rms_csv_stem and all(
            "log10_n1rms_now" in lab for lab in (tr_lab, va_lab)):
        targets.append({"name": "log10_n1rms_now", "cls": False,
                        "gate": "n1rms_present"})

    def label_mask(lab, spec):
        m = lab["_joined"] & np.isfinite(lab[spec["name"]])
        if spec["gate"]:
            m &= lab[spec["gate"]] == 1  # NaN (unjoined) compares False
        return m

    base_tr = {s["name"]: label_mask(tr_lab, s) for s in targets}
    base_va = {s["name"]: label_mask(va_lab, s) for s in targets}
    for s in targets:
        log(f"rows[{s['name']}]: train {int(base_tr[s['name']].sum())}, "
            f"val {int(base_va[s['name']].sum())}")

    ctrl_tr = control_features(tr_cache["physics"])
    ctrl_va = control_features(va_cache["physics"])

    _fin_tr, _fin_va = {}, {}

    def finite_tr(key):
        if key not in _fin_tr:
            _fin_tr[key] = np.isfinite(tr_cache["pooled"][key]).all(axis=1)
        return _fin_tr[key]

    def finite_va(key):
        if key not in _fin_va:
            _fin_va[key] = np.isfinite(va_cache["pooled"][key]).all(axis=1)
        return _fin_va[key]

    def probe_indices(name, key=None, diag_j=None, seed=0):
        """(capped train row idx, full val row idx) for target/slice."""
        m_tr, m_va = base_tr[name].copy(), base_va[name].copy()
        if key is not None:
            m_tr &= finite_tr(key)
            m_va &= finite_va(key)
        if diag_j is not None:  # per-modality validity, train AND eval
            m_tr &= tr_cache["diag_valid"][:, diag_j] == 1
            m_va &= va_cache["diag_valid"][:, diag_j] == 1
        tr_idx = cap_rows(np.flatnonzero(m_tr), args.max_train_rows, seed)
        return tr_idx, np.flatnonzero(m_va)

    def persistence_elm():
        """elm_now as the elm_next score — metric only, nothing fitted."""
        m = base_va["elm_next"]
        return binary_scores(va_lab["elm_next"][m], va_lab["elm_now"][m])

    # ----------------------------------------------------------- headline
    hl = {}  # target -> probe results / baselines / refit val scores

    def ensure_headline():
        if hl:
            return hl
        for spec in targets:
            name, cls = spec["name"], spec["cls"]
            tr_idx, va_idx = probe_indices(name, key=HEADLINE_KEY)
            entry = {"tr_idx": tr_idx, "va_idx": va_idx, "probe": None}
            hl[name] = entry
            if tr_idx.size < MIN_ROWS or va_idx.size < MIN_ROWS:
                log(f"headline[{name}]: too few rows "
                    f"({tr_idx.size}/{va_idx.size}) — skipped")
                continue
            X_tr = tr_cache["pooled"][HEADLINE_KEY][tr_idx]
            X_va = va_cache["pooled"][HEADLINE_KEY][va_idx]
            y_tr, y_va = tr_lab[name][tr_idx], va_lab[name][va_idx]
            g_tr = tr_cache["shot"][tr_idx]
            if cls:
                if np.unique(y_tr).size < 2:
                    log(f"headline[{name}]: single train class — skipped")
                    continue
                res = logistic_probe(X_tr, y_tr, g_tr, X_va, y_va,
                                     Cs=cs_grid)
                # deterministic refit at best_C to expose predict_proba for
                # the ROC/PR figure and the hero timeline
                entry["scores_va"] = logistic_proba(X_tr, y_tr, X_va,
                                                    res["best_C"])
                ctl = logistic_probe(ctrl_tr[tr_idx], y_tr, g_tr,
                                     ctrl_va[va_idx], y_va, Cs=cs_grid)
                entry["ctl_scores_va"] = logistic_proba(
                    ctrl_tr[tr_idx], y_tr, ctrl_va[va_idx], ctl["best_C"])
                entry["y_va"] = y_va
                entry["elm_now_va"] = va_lab["elm_now"][va_idx]
                entry["prevalence"] = float(y_va.mean())
                entry["baselines"] = {
                    "persistence_elm_now": persistence_elm(),
                    "prevalence_ap": float(y_va.mean()),
                    "physics_control": ctl,
                }
            else:
                res = ridge_probe(X_tr, y_tr, g_tr, X_va, y_va,
                                  n_splits=RIDGE_SPLITS)
                # No persistence baseline for the concurrent regression
                # targets: "persistence" would be the measured label itself
                # at inference time, which the probe never sees.
                ctl = ridge_probe(ctrl_tr[tr_idx], y_tr, g_tr,
                                  ctrl_va[va_idx], y_va,
                                  n_splits=RIDGE_SPLITS)
                entry["baselines"] = {"physics_control": ctl}
            entry["probe"] = res
            key_metric = "auroc" if cls else "r2"
            log(f"headline[{name}]: {key_metric}={res[key_metric]:.4f} "
                f"(n_tr={tr_idx.size}, n_va={va_idx.size})")
        return hl

    def fig_headline_roc_pr():
        e = hl.get("elm_next")
        if not e or e["probe"] is None:
            log("c0 ROC/PR skipped — no elm_next headline probe")
            return
        y, s, cs = e["y_va"], e["scores_va"], e["ctl_scores_va"]
        pn = e["elm_now_va"]
        pers = e["baselines"]["persistence_elm_now"]
        ctl = e["baselines"]["physics_control"]
        tpr_p = float(pn[y == 1].mean()) if (y == 1).any() else np.nan
        fpr_p = float(pn[y == 0].mean()) if (y == 0).any() else np.nan

        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(6.6, 2.9))
        fpr, tpr, _ = roc_curve(y, s)
        ax_roc.plot(fpr, tpr, color=OKABE_ITO[0], lw=1.5,
                    label=f"latent probe ({e['probe']['auroc']:.3f})")
        cf, ct, _ = roc_curve(y, cs)
        ax_roc.plot(cf, ct, color=OKABE_ITO[2], lw=1.5,
                    label=f"physics scalars ({ctl['auroc']:.3f})")
        ax_roc.plot([0, 1], [0, 1], color=GRAY, lw=0.8, linestyle=":")
        ax_roc.plot([fpr_p], [tpr_p], marker="x", markersize=5.0,
                    color="#000000", linestyle="none",
                    label=f"persistence ({pers['auroc']:.3f})")
        ax_roc.set_xlabel("false-positive rate")
        ax_roc.set_ylabel("true-positive rate")
        ax_roc.set_title("ROC — ELM in next window (AUROC)")
        ax_roc.legend(loc="lower right")

        prec, rec, _ = precision_recall_curve(y, s)
        ax_pr.plot(rec, prec, color=OKABE_ITO[0], lw=1.5,
                   label=f"latent probe ({e['probe']['ap']:.3f})")
        cprec, crec, _ = precision_recall_curve(y, cs)
        ax_pr.plot(crec, cprec, color=OKABE_ITO[2], lw=1.5,
                   label=f"physics scalars ({ctl['ap']:.3f})")
        ax_pr.axhline(e["prevalence"], color=GRAY, lw=0.8, linestyle=":",
                      label=f"prevalence ({e['prevalence']:.3f})")
        if (pn == 1).any():
            ax_pr.plot([tpr_p], [float(y[pn == 1].mean())], marker="x",
                       markersize=5.0, color="#000000", linestyle="none",
                       label=f"persistence ({pers['ap']:.3f})")
        ax_pr.set_xlabel("recall")
        ax_pr.set_ylabel("precision")
        ax_pr.set_title("precision-recall (AP)")
        ax_pr.legend(loc="best")
        for ax in (ax_roc, ax_pr):
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
        save_fig(fig, fig_dir / "c0_elm_next_roc_pr")
        plt.close(fig)
        log("wrote figures/study_c/c0_elm_next_roc_pr.{png,pdf}")

    # ------------------------------------------------------- slice sweep
    def step_slices():
        keys = list(tr_cache["pooled"].keys())  # 23, cache order
        diag_names = tr_cache["diag_names"]
        scores = {}
        rows_csv = []
        for spec in targets:
            name, cls = spec["name"], spec["cls"]
            metric = "auroc" if cls else "r2"
            for key in keys:
                diag_j = (diag_names.index(key)
                          if key in diag_names else None)
                tr_idx, va_idx = probe_indices(name, key=key, diag_j=diag_j)
                res = None
                ok = tr_idx.size >= MIN_ROWS and va_idx.size >= MIN_ROWS
                if ok and cls and np.unique(tr_lab[name][tr_idx]).size < 2:
                    ok = False
                if ok:
                    res = ridge_probe(
                        tr_cache["pooled"][key][tr_idx],
                        tr_lab[name][tr_idx], tr_cache["shot"][tr_idx],
                        va_cache["pooled"][key][va_idx],
                        va_lab[name][va_idx],
                        classification=cls, n_splits=RIDGE_SPLITS)
                scores[(name, key)] = res
                row = {"feature": key, "target": name,
                       "kind": "classification" if cls else "regression",
                       "n_train": int(tr_idx.size),
                       "n_test": int(va_idx.size)}
                if res:
                    row.update({k: f"{res[k]:.6g}" for k in METRIC_COLS
                                if k in res})
                rows_csv.append(row)
                shown = res[metric] if res else float("nan")
                log(f"slice[{name}][{key}]: {metric}={shown:.4f} "
                    f"(n_tr={tr_idx.size}, n_va={va_idx.size})")

        csv_path = tables_dir / "c_slice_sweep.csv"
        fields = ["feature", "target", "kind", *METRIC_COLS,
                  "n_train", "n_test"]
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows_csv)
        log(f"wrote {csv_path.relative_to(out_root)}")

        # Baseline vlines on the full (unrestricted) eval rows: persistence
        # for the ELM panel, physics-scalar control for the regressions.
        def control_vline(name, cls):
            if cls:
                return persistence_elm()["auroc"], "persistence (elm_now)"
            tr_idx, va_idx = probe_indices(name)
            if tr_idx.size < MIN_ROWS or va_idx.size < MIN_ROWS:
                return float("nan"), "physics-scalar control"
            res = ridge_probe(ctrl_tr[tr_idx], tr_lab[name][tr_idx],
                              tr_cache["shot"][tr_idx], ctrl_va[va_idx],
                              va_lab[name][va_idx],
                              n_splits=RIDGE_SPLITS)
            return res["r2"], "physics-scalar control"

        fig, axs = plt.subplots(
            1, len(targets),
            figsize=(3.1 * len(targets), 0.30 * len(keys) + 1.2))
        for ax, spec in zip(np.atleast_1d(axs), targets):
            name, cls = spec["name"], spec["cls"]
            metric = "auroc" if cls else "r2"
            vals = np.array([
                scores[(name, k)][metric] if scores[(name, k)]
                else np.nan for k in keys])
            order = np.argsort(np.nan_to_num(vals, nan=-np.inf))
            v, labs = vals[order], [keys[i] for i in order]
            ax.barh(np.arange(len(labs)),
                    np.where(np.isfinite(v), v, 0.0),
                    color=OKABE_ITO[0], height=0.72)
            ax.set_yticks(range(len(labs)), labels=labs, fontsize=6.8)
            base, base_lab = control_vline(name, cls)
            if np.isfinite(base):
                ax.axvline(base, color=GRAY, lw=1.0, linestyle=(0, (2, 2)),
                           label=base_lab, zorder=2.6)
                ax.legend(loc="lower right", fontsize=6.5)
            finite = v[np.isfinite(v)]
            if finite.size:
                lo = min(float(finite.min()), 0.0)
                hi = max(float(finite.max()),
                         base if np.isfinite(base) else 0.0)
                lo = max(lo, -1.0)
                pad = 0.05 * max(hi - lo, 0.1)
                ax.set_xlim(lo - pad, hi + pad)
            ax.set_xlabel("AUROC (ridge score)" if cls else r"R$^2$")
            ax.set_title(TARGET_TITLES.get(name, name))
        save_fig(fig, fig_dir / "c1_slice_sweep")
        plt.close(fig)
        log("wrote figures/study_c/c1_slice_sweep.{png,pdf}")

    # ------------------------------------------------- sample efficiency
    def step_efficiency():
        eff = [s for s in targets if s["name"] in ("elm_next", "mirnov_bp")]
        ns = np.asarray(ns_grid, dtype=int)
        payload = {"ns": ns, "seeds": np.arange(args.seeds)}
        panels = []
        for spec in eff:
            name, cls = spec["name"], spec["cls"]
            metric = "auroc" if cls else "r2"
            _, va_idx = probe_indices(name, key=HEADLINE_KEY)
            if va_idx.size < MIN_ROWS:
                log(f"efficiency[{name}]: too few eval rows — skipped")
                continue
            X_va = va_cache["pooled"][HEADLINE_KEY][va_idx]
            y_va = va_lab[name][va_idx]
            c_va = ctrl_va[va_idx]
            m_full = base_tr[name] & finite_tr(HEADLINE_KEY)
            pool = np.unique(tr_cache["shot"][m_full])
            lat = np.full((ns.size, args.seeds), np.nan)
            ctl = np.full((ns.size, args.seeds), np.nan)
            for i, n in enumerate(ns):
                for si in range(args.seeds):
                    subset = subsample_shots(pool, int(n), seed=si)
                    m = m_full & rows_for_shots(tr_cache["shot"], subset)
                    tr_idx = cap_rows(np.flatnonzero(m),
                                      args.max_train_rows, seed=si)
                    if tr_idx.size < MIN_ROWS:
                        continue
                    y_tr = tr_lab[name][tr_idx]
                    if cls and np.unique(y_tr).size < 2:
                        continue
                    g_tr = tr_cache["shot"][tr_idx]
                    r = ridge_probe(
                        tr_cache["pooled"][HEADLINE_KEY][tr_idx], y_tr,
                        g_tr, X_va, y_va, classification=cls,
                        n_splits=RIDGE_SPLITS)
                    lat[i, si] = r[metric]
                    r = ridge_probe(ctrl_tr[tr_idx], y_tr, g_tr, c_va,
                                    y_va, classification=cls,
                                    n_splits=RIDGE_SPLITS)
                    ctl[i, si] = r[metric]
                log(f"efficiency[{name}] n={int(n)}: latent "
                    f"{np.nanmean(lat[i]):.4f}, control "
                    f"{np.nanmean(ctl[i]):.4f} ({metric})")
            payload[f"{name}_latent"] = lat
            payload[f"{name}_control"] = ctl
            payload[f"{name}_metric"] = np.array(metric)
            panels.append((name, metric, lat, ctl))

        pers_auroc = persistence_elm()["auroc"]
        payload["elm_next_persistence_auroc"] = np.array(pers_auroc)
        npz_path = tables_dir / "c_sample_efficiency.npz"
        np.savez(npz_path, **payload)
        log(f"wrote {npz_path.relative_to(out_root)}")
        if not panels:
            return

        fig, axs = plt.subplots(1, len(panels),
                                figsize=(3.5 * len(panels), 2.7))
        for ax, (name, metric, lat, ctl) in zip(np.atleast_1d(axs), panels):
            if name == "elm_next" and np.isfinite(pers_auroc):
                ax.axhline(pers_auroc, color=GRAY, lw=1.0,
                           linestyle=(0, (2, 2)),
                           label="persistence (elm_now)")
            curves = {}
            for lab, arr in (("latent probe", lat),
                             ("physics scalars", ctl)):
                mean = np.nanmean(arr, axis=1)
                sd = np.nanstd(arr, axis=1)
                curves[lab] = (mean, mean - sd, mean + sd)
            efficiency_curve(ax, ns, curves, xlabel="labeled train shots",
                             ylabel={"auroc": "AUROC",
                                     "r2": r"R$^2$"}[metric])
            ax.set_title(TARGET_TITLES.get(name, name))
        save_fig(fig, fig_dir / "c2_sample_efficiency")
        plt.close(fig)
        log("wrote figures/study_c/c2_sample_efficiency.{png,pdf}")

    # ------------------------------------------------------ hero timeline
    def step_timeline():
        ensure_headline()
        e = hl.get("elm_next")
        if not e or e["probe"] is None:
            log("c3 timeline skipped — no elm_next headline model")
            return
        va_idx, y, p = e["va_idx"], e["y_va"], e["scores_va"]
        shots = va_cache["shot"][va_idx]
        t = va_cache["t_start_s"][va_idx].astype(np.float64)
        pos_shots, counts = np.unique(shots[y == 1], return_counts=True)
        if pos_shots.size == 0:
            log("c3 timeline skipped — no positive eval windows")
            return
        hero = int(pos_shots[np.argmax(counts)])
        e["hero_shot"] = hero
        sel = shots == hero
        order = np.argsort(t[sel])
        th, ph, yh = t[sel][order], p[sel][order], y[sel][order]
        m_top = ((va_cache["shot"] == hero) & va_lab["_joined"]
                 & np.isfinite(va_lab["dalpha_max_now"]))
        tt = va_cache["t_start_s"][m_top].astype(np.float64)
        dv = va_lab["dalpha_max_now"][m_top]
        o2 = np.argsort(tt)

        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True,
                                       figsize=(6.8, 3.6))
        ax0.plot(tt[o2], dv[o2], color="#000000", lw=1.2, marker=".",
                 markersize=3.0)
        ax0.set_yscale("log")
        ax0.set_ylabel("D-alpha window max")
        ax0.set_title(f"shot {hero} — ELM-next probe timeline")
        first = True
        for tw, yw in zip(th, yh):
            if yw == 1:  # shade the window whose NEXT 50 ms holds an ELM
                ax1.axvspan(tw, tw + WINDOW_S, color=OKABE_ITO[1],
                            alpha=0.15, linewidth=0,
                            label="elm_next = 1" if first else None)
                first = False
        ax1.plot(th, ph, color=OKABE_ITO[0], lw=1.5, marker="o",
                 markersize=3.0, label="probe P(elm_next)")
        ax1.set_ylim(-0.02, 1.02)
        ax1.set_ylabel("P(ELM in next window)")
        ax1.set_xlabel("time (s)")
        ax1.legend(loc="best")
        save_fig(fig, fig_dir / "c3_elm_timeline")
        plt.close(fig)
        log(f"wrote figures/study_c/c3_elm_timeline.{{png,pdf}} "
            f"(shot {hero}, {int(counts.max())} positive windows)")

    # ---------------------------------------------------------- dispatch
    if "headline" in steps:
        log("step: headline")
        ensure_headline()
        fig_headline_roc_pr()
    if "slices" in steps:
        log("step: slices")
        step_slices()
    if "efficiency" in steps:
        log("step: efficiency")
        step_efficiency()
    if "timeline" in steps:
        log("step: timeline")
        step_timeline()

    # --------------------------------------------------------------- json
    report = {
        "config": {
            "args": vars(args),
            "git_sha": _git_sha(),
            "date": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "steps": steps,
            "targets": [s["name"] for s in targets],
        },
        "targets": {},
    }
    for spec in targets:
        e = hl.get(spec["name"])
        if not e:
            continue
        entry = {
            "kind": "classification" if spec["cls"] else "regression",
            "headline_feature": HEADLINE_KEY,
            "headline": e["probe"],
            "baselines": e.get("baselines", {}),
            "n_rows": {"train": int(e["tr_idx"].size),
                       "test": int(e["va_idx"].size)},
        }
        if "prevalence" in e:
            entry["prevalence"] = e["prevalence"]
        if "hero_shot" in e:
            entry["timeline_shot"] = e["hero_shot"]
        report["targets"][spec["name"]] = entry
    json_path = tables_dir / "c_probe_metrics.json"
    json_path.write_text(json.dumps(_plain(report), indent=1))
    log(f"wrote {json_path.relative_to(out_root)}")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
