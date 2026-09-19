"""A1/A2/A5/A6/A7: representation quality from cached window latents.

CPU-only driver over the per-shot latent caches written by
``extract_latents.py`` (no model forwards, no GPU, no sbatch needed):

* **A1** — trainer-faithful pooled MAE table (model vs persistence copy;
  val vs train vs checkpoint-logged) + a printed overfit/underfit verdict.
* **A2** — per-shot model-vs-copy MAE scatter, one panel per diagnostic.
* **A5** — 2-D PCA/UMAP embeddings of the ``diag_global`` latent colored
  by split, window time, core density and input power.
* **A6** — memorization/retrieval: cosine NN distance of val windows to
  the train set vs to other val shots, difficulty-vs-novelty correlation,
  and a same-shot 1-NN retrieval control (requires the train cache).
* **A7** — eigenspectrum of all 23 pooled latent slices (participation
  ratio, effective rank, top-10 variance share).

A missing/empty ``--train-latents`` directory degrades gracefully (the
val cache lands first in production): A1 drops the train columns, A2/A5
run val-only, A6 is skipped. ``--smoke`` points both splits at the 2-shot
smoke cache (A1 sees the same cache as train and val; A6 uses it as a
fake train set) — numbers are garbage, only code paths and file
production are being tested.

Run from the repo root (login node is fine, CPU-only):
    python scripts/evaluation/study_a_latent_analysis.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tfm_eval.ckpt import DEFAULT_CKPT, load_ckpt  # noqa: E402
from tfm_eval.metrics import aggregate_masked_mae  # noqa: E402
from tfm_eval.plotting import OKABE_ITO, SEQ_CMAP, save_fig, set_style  # noqa: E402
from tfm_eval.probing import load_latent_cache  # noqa: E402

_EVAL_ROOT = _HERE.parents[2] / "data" / "outputs" / "eval_suite"
_SMOKE_LATENTS = _EVAL_ROOT / "e2e_stage1_best_smoke" / "latents" / "val"
ANALYSES = ("a1", "a2", "a5", "a6", "a7")
SEED = 42
VAL_COLOR = OKABE_ITO[0]    # blue
TRAIN_COLOR = OKABE_ITO[3]  # orange
NAN_GRAY = "#D9D9D9"


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
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT,
                    help="only its training-time metrics dict is read (A1)")
    ap.add_argument("--max-points", type=int, default=20000,
                    help="row cap for embeddings / NN queries / eigenspectra")
    ap.add_argument("--smoke", action="store_true",
                    help="2-shot smoke cache as both splits (code-path test)")
    ap.add_argument("--only", default=",".join(ANALYSES),
                    help="comma-list subset of {a1,a2,a5,a6,a7}")
    args = ap.parse_args()
    if args.smoke:
        args.val_latents = str(_SMOKE_LATENTS)
        args.train_latents = str(_SMOKE_LATENTS)
        args.out_root = str(_EVAL_ROOT / "e2e_stage1_best_smoke")
        args.max_points = 2000
    return args


# ------------------------------------------------------------------ helpers


def _pool_mae(mae_col, w_col) -> float:
    """Trainer-faithful pooled MAE of one (window,) column pair.

    ``metrics.aggregate_masked_mae`` is torch-only, so numpy columns are
    wrapped here once instead of at every call site.
    """
    return aggregate_masked_mae(torch.as_tensor(np.ascontiguousarray(mae_col)),
                                torch.as_tensor(np.ascontiguousarray(w_col)))


def _cache_mae_columns(cache):
    """(model, copy, ratio) pooled per modality, each (12,) with NaN holes."""
    n_mod = len(cache["diag_names"])
    model = np.array([_pool_mae(cache["mae"][:, j], cache["mae_w"][:, j])
                      for j in range(n_mod)])
    copy = np.array([_pool_mae(cache["copy_mae"][:, j],
                               cache["copy_mae_w"][:, j])
                     for j in range(n_mod)])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = model / copy
    ratio[~np.isfinite(ratio)] = np.nan
    return model, copy, ratio


def _phys_col(cache, name):
    return cache["physics"][:, cache["physics_names"].index(name)].astype(
        np.float64)


def _subsample_idx(n, cap, seed):
    """Sorted random row subset (identity when n <= cap)."""
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False))


def _l2norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# ----------------------------------------------------------------------- A1


def _ckpt_columns(path, diag_names, t0):
    """(mae, copy, ratio) from the checkpoint's training-time metrics dict,
    or None when the checkpoint / dict is unavailable (table then omits the
    ckpt columns — the analysis must not die on a missing 16 GB file)."""
    try:
        print(f"[{time.time()-t0:7.1f}s] [a1] reading ckpt metrics from "
              f"{path} (mmap, ~1 min)", flush=True)
        metrics = load_ckpt(path).get("metrics") or {}
        if not metrics:
            raise KeyError("checkpoint has no 'metrics' dict")
        mae = np.array([float(metrics.get(n, {}).get("model_mae", np.nan))
                        for n in diag_names])
        copy = np.array([float(metrics.get(n, {}).get("copy_mae", np.nan))
                         for n in diag_names])
    except Exception as exc:  # degrade, never crash on the optional column
        print(f"WARNING: [a1] checkpoint metrics unavailable ({exc}) — "
              "table omits ckpt columns", flush=True)
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = mae / copy
    ratio[~np.isfinite(ratio)] = np.nan
    return mae, copy, ratio


def _write_booktabs(df, path):
    esc_cols = [c.replace("_", "\\_") for c in df.columns]
    lines = ["\\begin{tabular}{l" + "r" * len(df.columns) + "}", "\\toprule",
             "modality & " + " & ".join(esc_cols) + " \\\\", "\\midrule"]
    for name, row in df.iterrows():
        esc = str(name).replace("_", "\\_")
        cells = " & ".join("--" if not np.isfinite(v) else f"{v:.4f}"
                           for v in row)
        lines.append(f"{esc} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    Path(path).write_text("\n".join(lines))


def run_a1(val, train, args, tab_dir, t0):
    """Pooled MAE table (val/train/ckpt) + overfit-vs-underfit verdict."""
    diag_names = val["diag_names"]
    df = pd.DataFrame(index=pd.Index(diag_names, name="modality"))
    v_mae, v_copy, v_ratio = _cache_mae_columns(val)
    df["val_mae"], df["val_copy"], df["val_ratio"] = v_mae, v_copy, v_ratio
    t_ratio = None
    if train is not None:
        t_mae, t_copy, t_ratio = _cache_mae_columns(train)
        df["train_mae"], df["train_copy"] = t_mae, t_copy
        df["train_ratio"] = t_ratio
    ck = _ckpt_columns(args.checkpoint, diag_names, t0)
    if ck is not None:
        df["ckpt_val_mae"], df["ckpt_val_copy"], df["ckpt_ratio"] = ck

    df.to_csv(tab_dir / "a1_mae_table.csv")
    _write_booktabs(df, tab_dir / "a1_mae_table.tex")
    print(f"[{time.time()-t0:7.1f}s] [a1] pooled masked MAE "
          "(z-units; ratio = model/copy):", flush=True)
    print(df.to_string(float_format=lambda v: f"{v:.4f}"), flush=True)

    lines = []
    if t_ratio is not None:
        both = np.isfinite(v_ratio) & np.isfinite(t_ratio)
        gap = float(np.mean(np.abs(v_ratio[both] - t_ratio[both])))
        mv = float(np.mean(v_ratio[both]))
        mt = float(np.mean(t_ratio[both]))
        lines.append(
            f"A1 verdict — over {int(both.sum())} modalities with finite "
            f"ratios: mean val_ratio = {mv:.3f}, mean train_ratio = {mt:.3f},"
            f" mean |val_ratio - train_ratio| = {gap:.3f}.")
        lines.append("Observed per-modality ratios (model/copy):")
        lines.append(f"  {'modality':22s} {'val_ratio':>9s} {'train_ratio':>11s}"
                     f" {'val-train':>9s}")
        for j, name in enumerate(diag_names):
            vr = f"{v_ratio[j]:9.3f}" if np.isfinite(v_ratio[j]) else \
                f"{'--':>9s}"
            tr = f"{t_ratio[j]:11.3f}" if np.isfinite(t_ratio[j]) else \
                f"{'--':>11s}"
            dd = v_ratio[j] - t_ratio[j]
            df_ = f"{dd:+9.3f}" if np.isfinite(dd) else f"{'--':>9s}"
            lines.append(f"  {name:22s} {vr} {tr} {df_}")
        if mt < mv and gap > 0.05:
            lines.append(
                "train_ratio sits well below val_ratio: the generative-"
                "overfitting signature — the model beats persistence far "
                "more on shots it has seen than on held-out shots.")
        elif gap <= 0.05:
            lines.append(
                "train and val ratios are near-equal (gap <= 0.05): no "
                "memorization signature; the model is underfitting / "
                "capacity- or data-limited rather than overfit.")
        else:
            lines.append(
                "ratios differ but not in the train << val overfitting "
                "direction; inspect the per-modality rows before "
                "concluding.")
    else:
        vals = ", ".join(f"{n}={r:.3f}" for n, r in zip(diag_names, v_ratio)
                         if np.isfinite(r))
        lines.append(
            "A1 verdict (degraded) — train cache unavailable, so the "
            "overfit-vs-underfit gap cannot be measured yet. "
            f"Val ratios (model/copy): {vals}.")
    if ck is not None and t_ratio is not None:
        both = np.isfinite(v_mae) & np.isfinite(ck[0])
        if both.any():
            dev = float(np.mean(np.abs(v_mae[both] - ck[0][both])))
            lines.append(
                f"Consistency check: mean |val_mae - ckpt_val_mae| = "
                f"{dev:.4f} (extraction sweep vs training-time logging).")
    if args.smoke:
        lines.append("(smoke mode: train == val cache — this verdict is a "
                     "code-path test only.)")
    verdict = "\n".join(lines)
    (tab_dir / "a1_verdict.txt").write_text(verdict + "\n")
    print(verdict, flush=True)


# ----------------------------------------------------------------------- A2


def _per_shot_mae(cache):
    """(shots, model (S, 12), copy (S, 12)) pooled per shot & modality."""
    order = np.argsort(cache["shot"], kind="stable")
    sorted_shot = cache["shot"][order]
    shots, starts = np.unique(sorted_shot, return_index=True)
    bounds = np.append(starts, sorted_shot.size)
    n_mod = len(cache["diag_names"])
    model = np.full((shots.size, n_mod), np.nan)
    copy = np.full((shots.size, n_mod), np.nan)
    for i in range(shots.size):
        rows = order[bounds[i]:bounds[i + 1]]
        for j in range(n_mod):
            model[i, j] = _pool_mae(cache["mae"][rows, j],
                                    cache["mae_w"][rows, j])
            copy[i, j] = _pool_mae(cache["copy_mae"][rows, j],
                                   cache["copy_mae_w"][rows, j])
    return shots, model, copy


def run_a2(val, train, fig_dir, t0):
    """12-panel log-log per-shot scatter: model MAE vs persistence MAE."""
    set_style()
    _, mv, cv = _per_shot_mae(val)
    mt = ct = None
    if train is not None:
        _, mt, ct = _per_shot_mae(train)
    fig, axs = plt.subplots(3, 4, figsize=(9.6, 7.4))
    for j, name in enumerate(val["diag_names"]):
        ax = axs.flat[j]
        ok_v = (np.isfinite(mv[:, j]) & np.isfinite(cv[:, j])
                & (mv[:, j] > 0) & (cv[:, j] > 0))
        pts = [cv[ok_v, j], mv[ok_v, j]]
        if mt is not None:
            ok_t = (np.isfinite(mt[:, j]) & np.isfinite(ct[:, j])
                    & (mt[:, j] > 0) & (ct[:, j] > 0))
            if ok_t.any():
                ax.scatter(ct[ok_t, j], mt[ok_t, j], s=9, alpha=0.5,
                           color=TRAIN_COLOR, linewidths=0, label="train",
                           zorder=2.1)
                pts += [ct[ok_t, j], mt[ok_t, j]]
        if ok_v.any():
            ax.scatter(cv[ok_v, j], mv[ok_v, j], s=9, alpha=0.5,
                       color=VAL_COLOR, linewidths=0, label="val",
                       zorder=2.3)
        allpts = np.concatenate([p for p in pts if p.size]) if any(
            p.size for p in pts) else np.array([])
        if allpts.size:
            lo, hi = 0.8 * allpts.min(), 1.25 * allpts.max()
            ax.plot([lo, hi], [lo, hi], color="#999999", linestyle="--",
                    linewidth=0.8, zorder=1.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            frac = float(np.mean(mv[ok_v, j] < cv[ok_v, j])) if ok_v.any() \
                else float("nan")
            sub = (f"{frac:.0%} of val shots below y=x" if np.isfinite(frac)
                   else "no valid val shots")
            ax.set_title(f"{name}\n{sub}", fontsize=7.5)
        else:
            ax.text(0.5, 0.5, "no valid windows", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7.5, color="#666666")
            ax.set_title(name, fontsize=7.5)
        if j == 0 and mt is not None:
            ax.legend(loc="lower right", fontsize=6.5, markerscale=1.6)
    fig.supxlabel("persistence-copy MAE per shot (z-units)")
    fig.supylabel("model MAE per shot (z-units)")
    paths = save_fig(fig, fig_dir / "a2_shotwise_scatter")
    plt.close(fig)
    print(f"[{time.time()-t0:7.1f}s] [a2] wrote {paths[0]}", flush=True)


# ----------------------------------------------------------------------- A5


def _draw_embedding(emb, split, t_start, log_ne, pin, method, out_stem,
                    subtitle=""):
    """2x2 panels of one 2-D embedding: split / time / density / power."""
    set_style()
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.8))
    ax = axs[0, 0]
    if split is not None:
        for lab, code, color in (("train", 1, TRAIN_COLOR),
                                 ("val", 0, VAL_COLOR)):
            m = split == code
            ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.5, color=color,
                       linewidths=0, label=lab)
        ax.legend(loc="best", markerscale=2.5)
        ax.set_title("split")
    else:
        ax.text(0.5, 0.5, "train cache absent —\nsplit panel skipped",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="#666666")
        ax.set_title("split (skipped)")
    panels = [
        (axs[0, 1], t_start, "window start", "window start (s)"),
        (axs[1, 0], log_ne, "log10 ne_core", "log10 ne_core (phys. units)"),
        (axs[1, 1], pin, "pin_total", "pin_total (phys. units)"),
    ]
    for ax, c, title, cbar_label in panels:
        finite = np.isfinite(c)
        ax.set_title(title)
        if not finite.any():
            ax.text(0.5, 0.5, "all values NaN", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="#666666")
            continue
        vmin, vmax = np.percentile(c[finite], [2.0, 98.0])
        if not vmin < vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        if (~finite).any():
            ax.scatter(emb[~finite, 0], emb[~finite, 1], s=4, alpha=0.4,
                       color=NAN_GRAY, linewidths=0)
            ax.annotate("gray = NaN", xy=(0.02, 0.02),
                        xycoords="axes fraction", fontsize=6.5,
                        color="#666666")
        sc = ax.scatter(emb[finite, 0], emb[finite, 1], s=4, alpha=0.6,
                        c=c[finite], cmap=SEQ_CMAP, vmin=vmin, vmax=vmax,
                        linewidths=0)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(cbar_label)
    for ax in axs.flat:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax in axs[1]:
        ax.set_xlabel(f"{method} 1 (unitless)")
    for ax in axs[:, 0]:
        ax.set_ylabel(f"{method} 2 (unitless)")
    fig.suptitle(f"A5 — diag_global embedding, {method}{subtitle}")
    paths = save_fig(fig, out_stem)
    plt.close(fig)
    return paths


def run_a5(val, train, args, fig_dir, t0):
    """PCA + UMAP embeddings of the pooled diag_global latent."""
    from sklearn.decomposition import PCA

    iv = _subsample_idx(val["shot"].size, args.max_points, SEED)
    X = val["pooled"]["diag_global"][iv]
    t_start = val["t_start_s"][iv].astype(np.float64)
    ne = _phys_col(val, "ne_core")[iv]
    pin = _phys_col(val, "pin_total")[iv]
    split = None
    if train is not None:
        it = _subsample_idx(train["shot"].size, args.max_points, SEED + 1)
        X = np.vstack([X, train["pooled"]["diag_global"][it]])
        t_start = np.concatenate([t_start,
                                  train["t_start_s"][it].astype(np.float64)])
        ne = np.concatenate([ne, _phys_col(train, "ne_core")[it]])
        pin = np.concatenate([pin, _phys_col(train, "pin_total")[it]])
        split = np.r_[np.zeros(iv.size, dtype=int),
                      np.ones(it.size, dtype=int)]
    with np.errstate(invalid="ignore", divide="ignore"):
        log_ne = np.log10(ne)
    log_ne[~np.isfinite(log_ne)] = np.nan
    print(f"[{time.time()-t0:7.1f}s] [a5] embedding {X.shape[0]} windows "
          f"(val {iv.size}" + ("" if split is None else
                               f" + train {int(split.sum())}") + ")",
          flush=True)

    pca = PCA(n_components=2, random_state=SEED)
    emb = pca.fit_transform(X.astype(np.float64))
    evr = pca.explained_variance_ratio_
    _draw_embedding(emb, split, t_start, log_ne, pin, "PC",
                    fig_dir / "a5_embedding_pca",
                    subtitle=f" ({evr[0]:.0%}+{evr[1]:.0%} var)")
    print(f"[{time.time()-t0:7.1f}s] [a5] PCA done "
          f"(explained var {evr[0]:.2f}/{evr[1]:.2f})", flush=True)

    try:
        import umap  # lazy: slow numba import, optional dependency

        emb_u = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                          random_state=SEED).fit_transform(X)
        _draw_embedding(np.asarray(emb_u), split, t_start, log_ne, pin,
                        "UMAP", fig_dir / "a5_embedding_umap")
        print(f"[{time.time()-t0:7.1f}s] [a5] UMAP done", flush=True)
    except Exception as exc:  # UMAP optional — fall back to PCA-only
        print(f"WARNING: [a5] UMAP import/fit failed ({exc}) — PCA-only",
              flush=True)


# ----------------------------------------------------------------------- A6


def run_a6(val, train, args, fig_dir, tab_dir, t0):
    """Memorization: NN distance to train vs to other val shots + control."""
    from scipy.stats import spearmanr
    from sklearn.neighbors import NearestNeighbors

    set_style()
    Xv = _l2norm(val["pooled"]["diag_global"])
    Xt = _l2norm(train["pooled"]["diag_global"])
    n_val = Xv.shape[0]
    iv = _subsample_idx(n_val, args.max_points, SEED)
    Xq = Xv[iv]
    shots_q = val["shot"][iv]
    if args.smoke:
        print(f"[{time.time()-t0:7.1f}s] [a6] smoke: val cache doubles as a "
              "FAKE train set — distances are meaningless", flush=True)

    nn_tr = NearestNeighbors(n_neighbors=1, metric="cosine",
                             algorithm="brute").fit(Xt)
    d_train = nn_tr.kneighbors(Xq)[0][:, 0]
    print(f"[{time.time()-t0:7.1f}s] [a6] d_train done "
          f"({iv.size} queries vs {Xt.shape[0]} train windows)", flush=True)

    # NN among val windows of OTHER shots: one brute query per query-shot
    # against the complement — exact, and the same total flops as a single
    # all-pairs pass.
    d_val_other = np.full(iv.size, np.nan)
    for s in np.unique(shots_q):
        other = val["shot"] != s
        if not other.any():
            continue
        qm = shots_q == s
        nn_o = NearestNeighbors(n_neighbors=1, metric="cosine",
                                algorithm="brute").fit(Xv[other])
        d_val_other[qm] = nn_o.kneighbors(Xq[qm])[0][:, 0]
    print(f"[{time.time()-t0:7.1f}s] [a6] d_val_other done", flush=True)

    # per-window mean z-MAE over finite & diag_valid modalities
    mae = val["mae"][iv]
    ok = np.isfinite(mae) & (val["diag_valid"][iv] > 0)
    cnt = ok.sum(1)
    mean_mae = np.where(cnt > 0,
                        np.where(ok, mae, 0.0).sum(1) / np.maximum(cnt, 1),
                        np.nan)

    # shot-retrieval control: 1-NN among all other val windows (same-shot
    # allowed, self excluded via the global row index)
    nn_v = NearestNeighbors(n_neighbors=2, metric="cosine",
                            algorithm="brute").fit(Xv)
    _, idx2 = nn_v.kneighbors(Xq)
    nn_idx = np.where(idx2[:, 0] == iv, idx2[:, 1], idx2[:, 0])
    same_shot = val["shot"][nn_idx] == shots_q
    top1_acc = float(same_shot.mean())
    n_per_query = np.array([(val["shot"] == s).sum() for s in shots_q],
                           dtype=np.float64)
    chance = float(np.mean((n_per_query - 1) / max(n_val - 1, 1)))

    both = np.isfinite(d_train) & np.isfinite(d_val_other)
    frac_mem = float(np.mean(d_train[both] < d_val_other[both])) \
        if both.any() else float("nan")
    pair = np.isfinite(d_train) & np.isfinite(mean_mae)
    if pair.sum() > 2 and np.unique(d_train[pair]).size > 1:
        sp = spearmanr(d_train[pair], mean_mae[pair])
        rho, pval = float(sp.statistic), float(sp.pvalue)
    else:
        rho = pval = float("nan")

    fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.3))
    ax = axs[0]
    hi = 1.05 * max(float(np.nanmax(d_train[both])) if both.any() else 1.0,
                    float(np.nanmax(d_val_other[both])) if both.any() else 1.0,
                    1e-3)
    if both.sum() > 4000:
        hb = ax.hexbin(d_train[both], d_val_other[both], gridsize=45,
                       cmap=SEQ_CMAP, mincnt=1,
                       extent=(0.0, hi, 0.0, hi))
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("windows")
    else:
        ax.scatter(d_train[both], d_val_other[both], s=8, alpha=0.5,
                   color=VAL_COLOR, linewidths=0)
    ax.plot([0.0, hi], [0.0, hi], color="#999999", linestyle="--",
            linewidth=0.8, zorder=1.5)
    ax.set_xlim(0.0, hi)
    ax.set_ylim(0.0, hi)
    ax.set_xlabel("cosine NN distance to train")
    ax.set_ylabel("cosine NN distance to other val shots")
    ax.set_title(f"{frac_mem:.1%} of val windows closer\nto train than to "
                 "other val shots", fontsize=8)

    ax = axs[1]
    bins = np.linspace(0.0, hi, 40)
    ax.hist(d_train[np.isfinite(d_train)], bins=bins, alpha=0.5,
            color=TRAIN_COLOR, label="to train")
    ax.hist(d_val_other[np.isfinite(d_val_other)], bins=bins, alpha=0.5,
            color=VAL_COLOR, label="to other val shots")
    ax.set_xlabel("cosine NN distance")
    ax.set_ylabel("windows")
    ax.set_title("NN distance distributions", fontsize=8)
    ax.legend(loc="best")

    ax = axs[2]
    ax.scatter(d_train[pair], mean_mae[pair], s=8, alpha=0.5,
               color=VAL_COLOR, linewidths=0)
    ax.set_xlabel("cosine NN distance to train")
    ax.set_ylabel("window mean z-MAE")
    rho_txt = f"{rho:.2f}" if np.isfinite(rho) else "n/a"
    ax.set_title(f"difficulty vs novelty\nSpearman rho = {rho_txt}",
                 fontsize=8)
    paths = save_fig(fig, fig_dir / "a6_memorization")
    plt.close(fig)

    def _dist_stats(d):
        d = d[np.isfinite(d)]
        if not d.size:
            return {"n": 0}
        deciles = np.percentile(d, np.arange(0, 101, 10))
        return {"n": int(d.size), "mean": float(d.mean()),
                "median": float(np.median(d)),
                "p05": float(np.percentile(d, 5)),
                "p95": float(np.percentile(d, 95)),
                "deciles_p0_to_p100": [float(v) for v in deciles],
                "hist_counts": np.histogram(d, bins=bins)[0].tolist()}

    payload = {
        "n_val_windows": int(n_val),
        "n_val_queries": int(iv.size),
        "n_train_windows": int(Xt.shape[0]),
        "frac_d_train_lt_d_val_other": frac_mem,
        "hist_bin_edges": [float(v) for v in bins],
        "d_train": _dist_stats(d_train),
        "d_val_other": _dist_stats(d_val_other),
        "spearman_d_train_vs_window_mae": {"rho": rho, "p": pval,
                                           "n": int(pair.sum())},
        "shot_retrieval": {"top1_same_shot_acc": top1_acc,
                           "chance": chance, "n_queries": int(iv.size)},
        "smoke_fake_train": bool(args.smoke),
    }
    (tab_dir / "a6_memorization.json").write_text(
        json.dumps(payload, indent=1))
    print(f"[{time.time()-t0:7.1f}s] [a6] frac(d_train < d_val_other) = "
          f"{frac_mem:.3f}; retrieval top-1 same-shot = {top1_acc:.3f} "
          f"(chance {chance:.3f}); wrote {paths[0]}", flush=True)


# ----------------------------------------------------------------------- A7


def run_a7(val, args, fig_dir, tab_dir, t0):
    """Eigenspectrum of each pooled slice: PR, effective rank, var_top10."""
    set_style()
    diag_names = val["diag_names"]
    act_names = val["act_names"]
    rows = []
    scree = None
    for key in val["pooled_names"]:
        if key in diag_names:
            m = val["diag_valid"][:, diag_names.index(key)] > 0
            idx = np.flatnonzero(m)
        else:
            idx = np.arange(val["shot"].size)
        n_rows = int(idx.size)
        if n_rows < 50:
            rows.append({"slice": key, "n_rows": n_rows, "PR": np.nan,
                         "eff_rank": np.nan, "var_top10": np.nan,
                         "n_dims": 1024})
            print(f"[{time.time()-t0:7.1f}s] [a7] {key:22s} n={n_rows:6d} "
                  "< 50 rows — NaN", flush=True)
            continue
        if n_rows > args.max_points:
            rng = np.random.default_rng(SEED)
            idx = np.sort(rng.choice(idx, size=args.max_points,
                                     replace=False))
        X = val["pooled"][key][idx].astype(np.float64)
        Xc = X - X.mean(0)
        lam = np.linalg.svd(Xc, compute_uv=False) ** 2
        tot = lam.sum()
        p = lam / tot
        pr = float(tot ** 2 / (lam ** 2).sum())
        pos = p[p > 0]
        eff_rank = float(np.exp(-(pos * np.log(pos)).sum()))
        var10 = float(p[:10].sum())
        rows.append({"slice": key, "n_rows": int(idx.size), "PR": pr,
                     "eff_rank": eff_rank, "var_top10": var10,
                     "n_dims": 1024})
        if key == "diag_global":
            scree = p
        print(f"[{time.time()-t0:7.1f}s] [a7] {key:22s} n={idx.size:6d} "
              f"PR={pr:7.1f} eff_rank={eff_rank:7.1f} top10={var10:.3f}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "a7_spectrum.csv", index=False)

    from matplotlib.patches import Patch

    group_color = {"diagnostic": OKABE_ITO[0], "actuator": OKABE_ITO[3],
                   "global": OKABE_ITO[2]}
    fig = plt.figure(figsize=(9.0, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.15))
    ax1 = fig.add_subplot(gs[0, 0])
    if scree is not None:
        ranks = np.arange(1, scree.size + 1)
        ax1.loglog(ranks, scree, color=OKABE_ITO[0], linewidth=1.5,
                   label="diag_global")
        ax1.loglog(ranks, scree[0] * ranks ** -1.0, color="#999999",
                   linestyle="--", linewidth=0.8, label="slope -1")
        ax1.legend(loc="best")
    else:
        ax1.text(0.5, 0.5, "diag_global spectrum unavailable", ha="center",
                 va="center", transform=ax1.transAxes, fontsize=8)
    ax1.set_xlabel("PC rank")
    ax1.set_ylabel("eigenvalue fraction")
    ax1.set_title("diag_global scree")

    ax2 = fig.add_subplot(gs[0, 1])
    ok = df[np.isfinite(df["PR"])].sort_values("PR")
    y = np.arange(len(ok))
    groups = ["diagnostic" if s in diag_names else
              "actuator" if s in act_names else "global"
              for s in ok["slice"]]
    ax2.barh(y, ok["PR"], color=[group_color[g] for g in groups],
             height=0.75)
    ax2.set_yticks(y, labels=ok["slice"], fontsize=6.5)
    for yi, (pr, er) in enumerate(zip(ok["PR"], ok["eff_rank"])):
        ax2.annotate(f"ER={er:.0f}", xy=(pr, yi), xytext=(3, 0),
                     textcoords="offset points", va="center", fontsize=6.0,
                     color="#333333")
    ax2.set_xlabel("participation ratio (of 1024 dims)")
    ax2.set_title("dimensionality per pooled slice")
    ax2.margins(x=0.12)
    present = [g for g in group_color if g in groups]
    ax2.legend(handles=[Patch(facecolor=group_color[g], label=g)
                        for g in present], loc="lower right", fontsize=6.5)
    paths = save_fig(fig, fig_dir / "a7_eigenspectrum")
    plt.close(fig)
    print(f"[{time.time()-t0:7.1f}s] [a7] wrote {paths[0]} and "
          f"{tab_dir / 'a7_spectrum.csv'}", flush=True)


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

    need_global = bool(only & {"a5", "a6"})
    val_keys = None if "a7" in only else \
        (["diag_global"] if need_global else [])
    val = load_latent_cache(args.val_latents, pooled_keys=val_keys)
    print(f"[{time.time()-t0:7.1f}s] val cache: {val['n_shots']} shots / "
          f"{val['shot'].size} windows from {args.val_latents}", flush=True)

    train = None
    if sorted(Path(args.train_latents).glob("*.npz")):
        train_keys = ["diag_global"] if need_global else []
        train = load_latent_cache(args.train_latents, pooled_keys=train_keys)
        print(f"[{time.time()-t0:7.1f}s] train cache: {train['n_shots']} "
              f"shots / {train['shot'].size} windows from "
              f"{args.train_latents}", flush=True)
    else:
        print(f"WARNING: no train cache at {args.train_latents} — A1 drops "
              "train columns, A2/A5 run val-only, A6 is skipped", flush=True)

    if "a1" in only:
        run_a1(val, train, args, tab_dir, t0)
    if "a2" in only:
        run_a2(val, train, fig_dir, t0)
    if "a5" in only:
        run_a5(val, train, args, fig_dir, t0)
    if "a6" in only:
        if train is None:
            print(f"[{time.time()-t0:7.1f}s] [a6] SKIPPED — requires the "
                  "train cache", flush=True)
        else:
            run_a6(val, train, args, fig_dir, tab_dir, t0)
    if "a7" in only:
        run_a7(val, args, fig_dir, tab_dir, t0)
    print(f"[{time.time()-t0:7.1f}s] DONE — figures in {fig_dir}, tables in "
          f"{tab_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
