"""Shared probing utilities for Studies A8 and C.

Three layers:

1. **Latent-cache loading** — reads the per-shot npz files written by
   ``extract_latents.py`` (schema: ``pooled`` fp16 (n, 23, 1024) with
   ``pooled_names``; ``mae``/``mae_w``/``copy_mae``/``copy_mae_w`` (n, 12);
   ``diag_valid``/``act_valid``; ``physics`` (n, 12) with alphabetical
   ``physics_names``; ``t_start_s``; shot number = filename stem).
2. **Label joining** — joins Study C window CSVs on the campaign-wide key
   ``(shot, t_start_s rounded to 2 dp)``.
3. **Probes** — a fast closed-form ridge probe (economy SVD once per fold,
   then the whole alpha path for free) used for BOTH regression targets
   (R²/Spearman) and binary targets scored by the continuous output
   (AUROC/AP) — the standard "linear probe" of the representation-learning
   literature — plus an sklearn logistic probe for headline classification
   numbers. Hyperparameters are always selected with GroupKFold over shots
   so no shot leaks between CV folds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

DEFAULT_ALPHAS = np.logspace(-3, 5, 9)


# ---------------------------------------------------------------- loading


def load_latent_cache(lat_dir, pooled_keys=None, max_shots=None):
    """Concatenate per-shot npz caches into flat arrays.

    Returns a dict with per-window arrays ``shot`` (int64), ``t_start_s``,
    ``local_idx``, ``mae``/``mae_w``/``copy_mae``/``copy_mae_w`` (N, 12),
    ``diag_valid`` (N, 12), ``act_valid`` (N, 9), ``physics`` (N, 12),
    ``pooled`` = {key: (N, 1024) float32}, and the name lists. ``pooled_keys``
    limits which of the 23 slices are materialized (None = all).
    """
    files = sorted(Path(lat_dir).glob("*.npz"))
    if max_shots:
        files = files[:max_shots]
    if not files:
        raise FileNotFoundError(f"no npz caches in {lat_dir}")

    names = {}
    cols = {k: [] for k in ("shot", "t_start_s", "local_idx", "mae", "mae_w",
                            "copy_mae", "copy_mae_w", "diag_valid",
                            "act_valid", "physics")}
    pooled_cols = None
    for f in files:
        z = np.load(f, allow_pickle=False)
        if not names:
            names = {k: [str(s) for s in z[k]] for k in
                     ("pooled_names", "diag_names", "act_names",
                      "physics_names")}
            keys = list(names["pooled_names"]) if pooled_keys is None \
                else list(pooled_keys)
            missing = set(keys) - set(names["pooled_names"])
            if missing:
                raise KeyError(f"pooled keys not in cache: {sorted(missing)}")
            idx = [names["pooled_names"].index(k) for k in keys]
            pooled_cols = {k: [] for k in keys}
        n = z["local_idx"].shape[0]
        cols["shot"].append(np.full(n, int(f.stem), dtype=np.int64))
        for k in ("t_start_s", "local_idx", "mae", "mae_w", "copy_mae",
                  "copy_mae_w", "diag_valid", "act_valid", "physics"):
            cols[k].append(z[k])
        pooled = z["pooled"]
        for k, j in zip(pooled_cols, idx):
            pooled_cols[k].append(pooled[:, j].astype(np.float32))

    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["pooled"] = {k: np.concatenate(v) for k, v in pooled_cols.items()}
    out.update(names)
    out["n_shots"] = len(files)
    return out


def join_window_labels(cache, labels_csv, cols):
    """Join label columns onto cache rows by ``(shot, t_start_s)``.

    Returns ``(values, joined)``: ``values[col]`` is (N,) float64 with NaN
    where unjoined or missing; ``joined`` is the (N,) bool join-hit mask.
    """
    import csv as _csv

    table = {}
    with open(labels_csv) as f:
        for row in _csv.DictReader(f):
            key = (int(row["shot"]), f"{float(row['t_start_s']):.2f}")
            table[key] = row

    n = cache["shot"].shape[0]
    values = {c: np.full(n, np.nan) for c in cols}
    joined = np.zeros(n, dtype=bool)
    for i in range(n):
        row = table.get((int(cache["shot"][i]),
                         f"{float(cache['t_start_s'][i]):.2f}"))
        if row is None:
            continue
        joined[i] = True
        for c in cols:
            try:
                values[c][i] = float(row[c])
            except (ValueError, KeyError):
                pass
    return values, joined


# ----------------------------------------------------------------- probes


def _standardize(X_tr, X_te):
    mu = X_tr.mean(0)
    sd = X_tr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X_tr - mu) / sd, (X_te - mu) / sd


def _ridge_path_predict(X_tr, y_tr, X_te, alphas):
    """Centered ridge via one economy SVD; returns (n_alphas, n_te) preds."""
    y_mu = y_tr.mean()
    U, s, Vt = np.linalg.svd(X_tr, full_matrices=False)
    Uty = U.T @ (y_tr - y_mu)
    XteV = X_te @ Vt.T
    preds = np.empty((len(alphas), X_te.shape[0]))
    for a, alpha in enumerate(alphas):
        coef_v = (s / (s**2 + alpha)) * Uty  # coefficients in V-basis
        preds[a] = XteV @ coef_v + y_mu
    return preds


def _score(y_true, y_score, classification):
    if classification:
        if len(np.unique(y_true)) < 2:
            return {"auroc": np.nan, "ap": np.nan}
        return {"auroc": roc_auc_score(y_true, y_score),
                "ap": average_precision_score(y_true, y_score)}
    ss_res = np.sum((y_true - y_score) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rho = spearmanr(y_true, y_score).statistic if y_true.size > 2 else np.nan
    return {"r2": r2, "spearman": rho}


def ridge_probe(X_tr, y_tr, groups_tr, X_te, y_te, *, classification=False,
                alphas=DEFAULT_ALPHAS, n_splits=5):
    """Grouped-CV ridge probe: pick alpha by GroupKFold over shots, refit
    on all of train, score on test. Binary targets (``classification=True``)
    are fit as 0/1 regression and scored by ranking (AUROC selects alpha).

    Returns test metrics plus ``best_alpha``, ``cv_score``, ``n_train``,
    ``n_test`` (all plain floats/ints, json-ready).
    """
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    y_te = np.asarray(y_te, dtype=np.float64)
    key = "auroc" if classification else "r2"

    n_splits = min(n_splits, len(np.unique(groups_tr)))
    if n_splits >= 2:
        fold_scores = np.zeros((n_splits, len(alphas)))
        gkf = GroupKFold(n_splits=n_splits)
        for k, (ia, ib) in enumerate(gkf.split(X_tr, y_tr, groups_tr)):
            Xa, Xb = _standardize(X_tr[ia], X_tr[ib])
            preds = _ridge_path_predict(Xa, y_tr[ia], Xb, alphas)
            for a in range(len(alphas)):
                fold_scores[k, a] = _score(y_tr[ib], preds[a],
                                           classification).get(key, np.nan)
        mean_scores = np.nanmean(fold_scores, axis=0)
        best = int(np.nanargmax(mean_scores))
    else:  # single train shot: grouped CV impossible, take mid-grid alpha
        mean_scores = np.full(len(alphas), np.nan)
        best = len(alphas) // 2

    Xa, Xb = _standardize(X_tr, X_te)
    pred = _ridge_path_predict(Xa, y_tr, Xb, alphas[best:best + 1])[0]
    out = _score(y_te, pred, classification)
    out = {k: float(v) for k, v in out.items()}
    out.update(best_alpha=float(alphas[best]),
               cv_score=float(mean_scores[best]),
               n_train=int(len(y_tr)), n_test=int(len(y_te)))
    return out


def logistic_probe(X_tr, y_tr, groups_tr, X_te, y_te,
                   Cs=(0.01, 0.1, 1.0), n_splits=3, max_iter=2000):
    """sklearn logistic probe (headline classification numbers only —
    ~10-30 s/fit at 30k x 1024, so keep the C grid short)."""
    from sklearn.linear_model import LogisticRegression

    n_splits = min(n_splits, len(np.unique(groups_tr)))
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        cv = np.zeros((n_splits, len(Cs)))
        for k, (ia, ib) in enumerate(gkf.split(X_tr, y_tr, groups_tr)):
            Xa, Xb = _standardize(X_tr[ia], X_tr[ib])
            for c, C in enumerate(Cs):
                clf = LogisticRegression(C=C, max_iter=max_iter)
                clf.fit(Xa, y_tr[ia])
                cv[k, c] = _score(y_tr[ib], clf.decision_function(Xb),
                                  True)["auroc"]
        best = int(np.nanargmax(np.nanmean(cv, axis=0)))
    else:  # single train shot: no grouped CV, take mid-grid C
        best = len(Cs) // 2
    Xa, Xb = _standardize(X_tr, X_te)
    clf = LogisticRegression(C=Cs[best], max_iter=max_iter)
    clf.fit(Xa, y_tr)
    out = _score(y_te, clf.decision_function(Xb), True)
    out.update(best_C=float(Cs[best]), n_train=int(len(y_tr)),
               n_test=int(len(y_te)))
    return {k: float(v) for k, v in out.items()}


def subsample_shots(shots, n, seed):
    """Deterministic n-shot subset of the unique shots in ``shots``."""
    uniq = np.unique(shots)
    rng = np.random.default_rng(seed)
    if n >= uniq.size:
        return set(uniq.tolist())
    return set(rng.choice(uniq, size=n, replace=False).tolist())


def rows_for_shots(shot_col, shot_subset):
    return np.isin(shot_col, np.fromiter(shot_subset, dtype=np.int64))
