"""A8: latent -> physics linear probes — what do the latents encode?

Ridge probes (``tfm_eval.probing.ridge_probe``: GroupKFold-over-shots
alpha selection, fit on the TRAIN cache, scored on the held-out VAL
cache) map every pooled latent slice to 7 diagnostic physics scalars in
PHYSICAL units: ne_core, te_core, ne_tang, te_tang, ti, rotation and
d_alpha. d_alpha spans decades, so it is probed as log10(d_alpha) on
rows with d_alpha > 0 (column named ``log10_d_alpha`` in all outputs).

Features (15): the 12 per-diagnostic pooled slices (rows additionally
restricted to windows where that diagnostic is valid, in both splits),
``diag_global``, ``act_global``, and a non-latent control
``act_scalars`` — 5 operating-point actuator scalars (NaN -> 0) plus
their 5 finite-indicator columns. Latent features are only interesting
where they beat that 10-dim control.

Outputs: ``tables/a8_probe_r2.csv`` (feature x target test-R² matrix),
``tables/a8_details.json`` (per-cell r2/spearman/best_alpha/n),
``figures/study_a/a8_probe_heatmap`` (signed heatmap, display clipped to
[-0.05, 1]).

CPU-only, ~45-60 min at defaults (105 ridge cells, 4 SVDs each).
``--smoke`` splits the 2-shot smoke cache into a fake train (shot
191599) / val (shot 204510) pair — numbers are garbage, only code paths
and file production are being tested.

Run from the repo root (login node is fine, CPU-only):
    python scripts/evaluation/study_a_probes.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from tfm_eval.plotting import save_fig, set_style, signed_heatmap  # noqa: E402
from tfm_eval.probing import load_latent_cache, ridge_probe  # noqa: E402

_EVAL_ROOT = _HERE.parents[2] / "data" / "outputs" / "eval_suite"
_SMOKE_LATENTS = _EVAL_ROOT / "e2e_stage1_best_smoke" / "latents" / "val"
ANALYSES = ("a8",)
SEED = 42
MIN_TRAIN_ROWS = 20
MIN_TEST_ROWS = 10

#: probe targets, physics-matrix column names (d_alpha probed as log10)
TARGETS = ["ne_core", "te_core", "ne_tang", "te_tang", "ti", "rotation",
           "d_alpha"]
#: operating-point scalars forming the non-latent "act_scalars" control
ACT_SCALAR_COLS = ["beam_voltage", "ech_power_total", "gas_flow_total",
                   "pin_total", "rmp_abs"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--val-latents",
        default=str(_EVAL_ROOT / "e2e_stage1_best" / "latents" / "val"),
    )
    ap.add_argument(
        "--train-latents",
        default=str(_EVAL_ROOT / "e2e_stage1_best" / "latents" / "train"),
    )
    ap.add_argument("--out-root", default=str(_EVAL_ROOT / "e2e_stage1_best"))
    ap.add_argument("--max-train-rows", type=int, default=15000,
                    help="seeded per-fit cap on the ridge training rows")
    ap.add_argument("--n-splits", type=int, default=3,
                    help="GroupKFold splits (over shots) for alpha selection")
    ap.add_argument("--smoke", action="store_true",
                    help="2-shot smoke cache, split by shot (code-path test)")
    ap.add_argument("--only", default=",".join(ANALYSES),
                    help="comma-list subset of {a8}")
    args = ap.parse_args()
    if args.smoke:
        args.val_latents = str(_SMOKE_LATENTS)
        args.train_latents = str(_SMOKE_LATENTS)
        args.out_root = str(_EVAL_ROOT / "e2e_stage1_best_smoke")
        args.max_train_rows = 2000
    return args


# ------------------------------------------------------------------ helpers


def _peek_diag_names(lat_dir):
    files = sorted(Path(lat_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no npz caches in {lat_dir}")
    with np.load(files[0]) as z:
        return [str(s) for s in z["diag_names"]]


def _mask_cache(cache, m):
    """Row-masked copy of a loaded cache (name lists pass through)."""
    out = {}
    for k, v in cache.items():
        if k == "pooled":
            out[k] = {kk: vv[m] for kk, vv in v.items()}
        elif isinstance(v, np.ndarray) and v.ndim >= 1 \
                and v.shape[0] == m.size:
            out[k] = v[m]
        else:
            out[k] = v
    return out


def _split_cache_by_shot(cache):
    """Smoke-only fake split: first sorted shot -> train, rest -> val."""
    shots = np.unique(cache["shot"])
    tr = cache["shot"] == shots[0]
    return _mask_cache(cache, tr), _mask_cache(cache, ~tr)


def _target(cache, name):
    """(N,) float64 target; d_alpha becomes log10 on rows with d_alpha>0."""
    y = cache["physics"][:, cache["physics_names"].index(name)].astype(
        np.float64)
    if name == "d_alpha":
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.where(y > 0, np.log10(np.maximum(y, 1e-300)), np.nan)
    return y


def _feature_matrix(cache, feat):
    """(X, valid_row_mask or None) for one feature name."""
    if feat == "act_scalars":
        cols = [cache["physics_names"].index(n) for n in ACT_SCALAR_COLS]
        raw = cache["physics"][:, cols].astype(np.float64)
        finite = np.isfinite(raw)
        X = np.concatenate([np.where(finite, raw, 0.0),
                            finite.astype(np.float64)], axis=1)
        return X, None
    X = cache["pooled"][feat]
    if feat in cache["diag_names"]:
        j = cache["diag_names"].index(feat)
        return X, cache["diag_valid"][:, j] > 0
    return X, None


# --------------------------------------------------------------------- A8


def run_a8(val, train, args, fig_dir, tab_dir, t0):
    features = (["diag_global"] + list(val["diag_names"])
                + ["act_global", "act_scalars"])
    target_labels = ["log10_d_alpha" if t == "d_alpha" else t
                     for t in TARGETS]
    r2 = np.full((len(features), len(TARGETS)), np.nan)
    details = {f: {} for f in features}

    y_tr_all = {t: _target(train, t) for t in TARGETS}
    y_te_all = {t: _target(val, t) for t in TARGETS}
    for fi, feat in enumerate(features):
        X_tr, valid_tr = _feature_matrix(train, feat)
        X_te, valid_te = _feature_matrix(val, feat)
        for ti, (tgt, label) in enumerate(zip(TARGETS, target_labels)):
            rows_tr = np.isfinite(y_tr_all[tgt])
            rows_te = np.isfinite(y_te_all[tgt])
            if valid_tr is not None:  # diag slice valid in BOTH splits
                rows_tr &= valid_tr
                rows_te &= valid_te
            idx_tr = np.flatnonzero(rows_tr)
            idx_te = np.flatnonzero(rows_te)
            if idx_tr.size < MIN_TRAIN_ROWS or idx_te.size < MIN_TEST_ROWS:
                details[feat][label] = {
                    "skipped": f"too few rows (n_train={idx_tr.size}, "
                               f"n_test={idx_te.size})"}
                print(f"[{time.time()-t0:7.1f}s] {feat}×{label}  SKIPPED "
                      f"(n_tr={idx_tr.size}, n_te={idx_te.size})",
                      flush=True)
                continue
            if idx_tr.size > args.max_train_rows:
                rng = np.random.default_rng([SEED, fi, ti])
                idx_tr = np.sort(rng.choice(idx_tr, size=args.max_train_rows,
                                            replace=False))
            res = ridge_probe(
                X_tr[idx_tr], y_tr_all[tgt][idx_tr], train["shot"][idx_tr],
                X_te[idx_te], y_te_all[tgt][idx_te],
                n_splits=args.n_splits,
            )
            r2[fi, ti] = res["r2"]
            details[feat][label] = res
            print(f"[{time.time()-t0:7.1f}s] {feat}×{label}  "
                  f"r2={res['r2']:+.3f} rho={res['spearman']:+.3f} "
                  f"alpha={res['best_alpha']:g} "
                  f"(n_tr={res['n_train']}, n_te={res['n_test']})",
                  flush=True)

    df = pd.DataFrame(r2, index=pd.Index(features, name="feature"),
                      columns=target_labels)
    df.to_csv(tab_dir / "a8_probe_r2.csv")
    payload = {
        "protocol": "ridge_probe fit on train cache, scored on val cache; "
                    "alpha by GroupKFold over shots",
        "notes": {
            "d_alpha": "probed as log10(d_alpha) on rows with d_alpha > 0",
            "act_scalars": "non-latent control: "
                           + ", ".join(ACT_SCALAR_COLS)
                           + " with NaN->0 + 5 finite-indicator columns",
            "diag_slices": "rows restricted to diag_valid == 1 for that "
                           "modality in both splits",
        },
        "n_splits": args.n_splits,
        "max_train_rows": args.max_train_rows,
        "smoke": bool(args.smoke),
        "cells": details,
    }
    (tab_dir / "a8_details.json").write_text(json.dumps(payload, indent=1))

    disp = np.clip(r2, -0.05, 1.0)
    annot = np.array([[f"{v:.2f}" if np.isfinite(v) else ""
                       for v in row] for row in r2], dtype=object)
    fig = signed_heatmap(
        disp, features, target_labels, annot=annot,
        title="A8 linear probes: pooled latents -> physics scalars",
        cbar_label="test R²", figsize=(7.2, 7.0),
    )
    paths = save_fig(fig, fig_dir / "a8_probe_heatmap")
    print(f"[{time.time()-t0:7.1f}s] [a8] wrote "
          f"{tab_dir / 'a8_probe_r2.csv'}, "
          f"{tab_dir / 'a8_details.json'}, {paths[0]}", flush=True)
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"), flush=True)


# ---------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    t0 = time.time()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    unknown = only - set(ANALYSES)
    if unknown:
        raise SystemExit(f"unknown --only entries: {sorted(unknown)} "
                         f"(choose from {list(ANALYSES)})")
    fig_dir = Path(args.out_root) / "figures" / "study_a"
    tab_dir = Path(args.out_root) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    pooled_keys = _peek_diag_names(args.val_latents) + ["diag_global",
                                                        "act_global"]
    val = load_latent_cache(args.val_latents, pooled_keys=pooled_keys)
    print(f"[{time.time()-t0:7.1f}s] val cache: {val['n_shots']} shots / "
          f"{val['shot'].size} windows from {args.val_latents}", flush=True)
    if args.smoke and args.train_latents == args.val_latents:
        train, val = _split_cache_by_shot(val)
        print(f"[{time.time()-t0:7.1f}s] smoke: split the val cache by shot "
              f"— fake train = shot {int(np.unique(train['shot'])[0])} "
              f"({train['shot'].size} windows), val = "
              f"{np.unique(val['shot']).tolist()} ({val['shot'].size} "
              "windows); numbers are garbage", flush=True)
    else:
        if not sorted(Path(args.train_latents).glob("*.npz")):
            print(f"ERROR: no train cache at {args.train_latents} — the A8 "
                  "probe protocol (fit train -> eval val) needs it; rerun "
                  "once the train extraction lands", flush=True)
            return 1
        train = load_latent_cache(args.train_latents,
                                  pooled_keys=pooled_keys)
        print(f"[{time.time()-t0:7.1f}s] train cache: {train['n_shots']} "
              f"shots / {train['shot'].size} windows from "
              f"{args.train_latents}", flush=True)

    if "a8" in only:
        run_a8(val, train, args, fig_dir, tab_dir, t0)
    print(f"[{time.time()-t0:7.1f}s] DONE — figures in {fig_dir}, tables in "
          f"{tab_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
