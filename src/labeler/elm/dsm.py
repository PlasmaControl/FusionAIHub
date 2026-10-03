"""`elm-dsm`: the lab's ELM time-to-event model on the reviewed shots, as a detector.

The model is Deep Survival Machines (auton-survival, LogNormal, three components, a
128-unit ReLU6 embedding with dropout 0.2) fitted by the labeler on the ELM survival
rows of the `wpqh_elm_hiro` project: 60 non-BES columns on a 25 ms grid, risk of an
ELM within 5, 10, 20 and 50 ms. Three things are done with it here.

* **Own target** (`legacy_own_target`): the limited-input refit on its own split's
  test rows, the time to the next ELM, AUROC at each horizon with 95 % shot-bootstrap
  intervals over the split's own shots.
* **Survival refit on the reviewed bins**: the model served from the corpus as
  `labeler.models.d3d_elm_time_to_event_dsm` serves it (`shot_rows`), its 50 ms risk at
  a bin's start read as the bin's score: "will an ELM start in the next 50 ms",
  asked before the bin begins.
* **Detection** (`Detector`, `fit_fold`): the same inputs and embedding with a single
  logit in place of the survival heads, trained with cross-entropy to say whether the
  50 ms ending at a row's time stamp is present (ELMy) or absent in the review: the
  objective changed from "time to the next ELM" to "is an ELM going on now", the
  horizon 0 case. Trained by shot-grouped cross-validation on the same folds as
  `elm-ours`; the threshold is chosen on each fold's inner-validation shots.

**Rows.** Each row of the 25 ms grid summarises the 50 ms before its time stamp (a
`[t - 50, t)` mean), so the row at a bin's end summarises the bin and the row at its
start the bin before it.

This is **DSM refit, limited inputs (60 of the original 124)**, not the original
124-input checkpoint. Missing `ip` and `bt` can be fetched into the isolated round-four
store by `elm_dsm_fetch.py`; no corpus or production feature file is changed. Remaining
missing columns, including the two photodiodes (`pcphd02/03`) on every shot, are filled
at the training mean. Inputs outside the refit's row filter are clipped rather than
dropped; the evaluation records missingness, filter failures and the risk scale.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ..config import Paths, sha256_of
from ..features import namespace as ns
from ..features import resolve_archive, resolve_corpus
from ..features.store import FeatureArray
from ..models import elm_inputs
from ..models.d3d_elm_time_to_event_dsm import spec
from ..models.runners import dsm_pickle
from . import labels, methods, score

SLUG = spec.SLUG
SPLIT_PKL = Path("/projects/EKOLEMEN/wpqh_elm_hiro/data/train_test_split_model10.pkl")
HORIZONS_MS = spec.HORIZONS_MS
T_OFFSET_MS = spec.T_OFFSET_MS
N_COLUMNS = len(spec.COLUMNS)
DISPLAY_NAME = "DSM refit, limited inputs (60 of the original 124)"
ROWS_SCHEMA = 2
ROW_T_MS = ns.GRID_S * 1000.0
WINDOW_MS = methods.WINDOW_MS


# ------------------------------------------------------------ own target


def auroc_by_shot(
    risk: np.ndarray, case: np.ndarray, keep: np.ndarray, shot: np.ndarray, n_bins=2000
):
    """Per-shot case and control histograms of `risk`, for a fast shot bootstrap.

    The risk is cut at quantiles of the kept rows; the AUROC of pooled histograms
    (ties within a cut at half credit) differs from the exact rank AUROC by less than
    the cut width, which `legacy_own_target` records.
    """
    r = risk[keep]
    edges = np.quantile(r, np.linspace(0, 1, n_bins + 1)[1:-1])
    cut = np.searchsorted(edges, r)
    shots, inv = np.unique(shot[keep], return_inverse=True)
    pos = np.zeros((len(shots), n_bins))
    neg = np.zeros((len(shots), n_bins))
    c = case[keep]
    np.add.at(pos, (inv[c], cut[c]), 1)
    np.add.at(neg, (inv[~c], cut[~c]), 1)
    return shots, pos, neg


def hist_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC of pooled case and control histograms, ties at half credit."""
    p, n = pos.sum(axis=0), neg.sum(axis=0)
    below = np.cumsum(n) - n
    num = (p * (below + 0.5 * n)).sum()
    return float(num / (p.sum() * n.sum())) if p.sum() and n.sum() else float("nan")


def legacy_own_target(model_dir: Path, boot_draws: np.ndarray, split_pkl=SPLIT_PKL):
    """The limited-input refit on its own test rows: AUROC per horizon, with intervals.

    Rows are upstream's 1 ms rows. A case at horizon `h` is a row whose next ELM is
    within `h` ms, a control one whose next ELM is later than `h`; a row censored
    inside `h` is neither (it is unknown whether an ELM came). The shots are the
    split's own shot ids; intervals are percentile intervals over shot draws.
    """
    with open(split_pkl, "rb") as fh:
        d = pickle.load(fh)
    cols = list(elm_inputs.column_indices("no_bes"))
    x = np.asarray(d["test_final_x_normalized"], dtype=np.float64)[:, cols]
    e = np.asarray(d["test_final_e"], dtype=float).ravel() == 1.0
    t = np.asarray(d["test_final_t"], dtype=float).ravel()
    shot = np.asarray(d["test_final_shots_list"]).ravel()
    graph = dsm_pickle.load_dsm(model_dir / spec.ARTIFACTS[0])
    surv = dsm_pickle.survival(graph, x, [h + T_OFFSET_MS for h in HORIZONS_MS])
    n_shots = int(np.unique(shot).size)
    out: dict = {
        "display_name": DISPLAY_NAME,
        "split": str(split_pkl),
        "rows": len(t),
        "shots": n_shots,
        "split_shots": {
            "train": sorted(int(v) for v in np.unique(d["train_final_shots_list"])),
            "test": sorted(int(v) for v in np.unique(shot)),
        },
        "event_rate": float(e.mean()),
        "horizons": {},
    }
    for j, h in enumerate(HORIZONS_MS):
        risk = 1.0 - surv[:, j]
        case = e & (t <= h)
        control = t > h
        keep = case | control
        exact = score.roc_auc(case[keep], risk[keep])
        ap = score.average_precision(case[keep], risk[keep])
        shots, pos, neg = auroc_by_shot(risk, case, keep, shot)
        point = hist_auroc(pos, neg)
        reps = np.array(
            [hist_auroc(pos[d_], neg[d_]) for d_ in _resample(len(shots), boot_draws)]
        )
        out["horizons"][f"h{int(h)}ms"] = {
            "horizon_ms": h,
            "queried_at_ms": h + T_OFFSET_MS,
            "cases": int(case.sum()),
            "controls": int(control.sum()),
            "censored_within_h": int((~e & (t <= h)).sum()),
            "auroc": exact,
            "auroc_histogram": point,
            "auroc_ci95": score._ci(reps),
            "auprc": ap,
            "prevalence": float(case[keep].mean()),
        }
    return out


def _resample(n: int, draws: np.ndarray | None):
    """Draws over `n` shots: the supplied ones when they have the right width."""
    if draws is not None and draws.shape[1] == n:
        return draws
    return score.draws(n)


# --------------------------------------------------- reviewed shots as rows


@dataclass
class Rows:
    """One shot's DSM inputs on the 25 ms grid."""

    shot: int
    x: np.ndarray  # (240, 60) float32, normalised, mean-filled
    usable: np.ndarray  # (240,) bool: the ECE record covers the row's 50 ms window
    in_filter: np.ndarray  # (240,) bool: upstream's |z| <= 10 row filter passes
    missing: tuple[str, ...]  # canonical features nothing served
    resolvers: dict[str, str]
    filled: tuple[str, ...]  # model columns held at the training mean
    source_signature: str = ""  # source files, normalisation and serving schema


def fetched_features_dir(paths: Paths) -> Path:
    """The isolated Ip/Bt fetch store; never the production feature store."""
    return paths.root / "round4" / "elm" / "dsm" / "fetched_features"


def fetched_features(paths: Paths, shot: int) -> dict[str, FeatureArray]:
    path = fetched_features_dir(paths) / f"{shot}.npz"
    if not path.exists():
        return {}
    arrays = {}
    with np.load(path, allow_pickle=False) as z:
        for name in ("ip", "bt"):
            if f"{name}_x" in z.files and f"{name}_y" in z.files:
                arrays[name] = FeatureArray(
                    x=z[f"{name}_x"],
                    y=z[f"{name}_y"],
                    attrs={
                        **json.loads(str(z[f"{name}_attrs"])),
                        # Keep the fdp sampling convention in InputSpec.build.
                        "resolver": "fdp",
                        "fetch_cache": str(path),
                    },
                )
    return arrays


def source_signature(paths: Paths, shot: int, norm: dict) -> str:
    """Cache key that changes when Ip/Bt is fetched, inputs change, or norms change.

    Large read-only source files use metadata; the small fetched feature file uses a
    content hash. The schema version invalidates the old cache without deleting it.
    """
    files = [paths.corpus_file(shot), *resolve_archive.ARCHIVE_FILES]
    records = []
    for path in files:
        st = path.stat() if path.exists() else None
        records.append((str(path), None if st is None else (st.st_size, st.st_mtime_ns)))
    fetched = fetched_features_dir(paths) / f"{shot}.npz"
    key = {
        "schema": ROWS_SCHEMA,
        "shot": shot,
        "files": records,
        "fetch_sha256": sha256_of(fetched) if fetched.exists() else None,
        "norm": norm,
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def rows_digest(rows: Rows) -> str:
    """Content identity of the rows used by a fit, independent of cache timestamps."""
    h = hashlib.sha256()
    for arr in (rows.x, rows.usable, rows.in_filter):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(json.dumps([rows.missing, rows.filled, rows.resolvers]).encode())
    return h.hexdigest()


def risk_quantiles(risk: np.ndarray) -> dict:
    """Risk scale in the selected rows/bins, including finite-value counts."""
    v = np.asarray(risk).ravel()
    finite = v[np.isfinite(v)]
    quantiles = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    return {
        "n": int(v.size),
        "nonfinite": int(v.size - finite.size),
        "quantiles": {
            f"{q:g}": float(a)
            for q, a in zip(
                quantiles,
                np.quantile(finite, quantiles) if finite.size else [float("nan")] * 9,
                strict=True,
            )
        },
    }


def row_diagnostics(rows: dict[int, Rows], risk: dict[int, np.ndarray]) -> dict:
    """Missingness and serving-filter failures, with an explicit usable-row base."""
    usable = sum(int(r.usable.sum()) for r in rows.values())
    outside = sum(int((r.usable & ~r.in_filter).sum()) for r in rows.values())
    names = sorted({n for r in rows.values() for n in r.missing})
    return {
        "display_name": DISPLAY_NAME,
        "input_columns": N_COLUMNS,
        "original_input_columns": 124,
        "usable_rows": usable,
        "outside_training_filter_usable_rows": outside,
        "outside_training_filter_usable_row_share": outside / usable
        if usable
        else float("nan"),
        "training_filter": (
            "all measured non-photodiode columns |z| <= 10 and measured CO2 in "
            "[0, 1e15]; filter failures retained, normalized inputs clipped to [-10, 10]"
        ),
        "missing_features": {
            n: {
                "n_shots": sum(n in r.missing for r in rows.values()),
                "shots": sorted(s for s, r in rows.items() if n in r.missing),
            }
            for n in names
        },
        "always_mean_filled_columns": list(spec.ALWAYS_MEAN_FILLED),
        "risk_quantiles_usable_rows": {
            f"h{int(h)}ms": risk_quantiles(
                np.concatenate([risk[s][r.usable, j] for s, r in rows.items()])
            )
            for j, h in enumerate(HORIZONS_MS)
        },
    }


def shot_features(paths: Paths, shot: int, names=None):
    """The canonical features the model reads, from the archive then the corpus."""
    names = spec.INPUT_SPEC.canonical_names if names is None else names
    arrays: dict = {}
    for source in ("archive", "corpus"):
        want = [n for n in names if source in ns.by_name(n).sources and n not in arrays]
        if not want:
            continue
        if source == "archive":
            got, _ = resolve_archive.resolve(shot, want)
        else:
            got, _ = resolve_corpus.resolve(shot, want, corpus=paths.corpus)
        arrays.update(got)
    for name, arr in fetched_features(paths, shot).items():
        if name in names:
            arrays.setdefault(name, arr)
    return arrays


def shot_rows(paths: Paths, shot: int, norm: dict) -> Rows:
    """`Rows` of one shot, built as the adapter builds them."""
    arrays = shot_features(paths, shot)
    built = spec.INPUT_SPEC.build(arrays, ns.GRID_S)
    x, in_filter = spec.preprocess(built, norm)
    # Upstream dropped training rows with any |z| > 10; many rows of these shots are
    # that far out (a different era's ECE and actuator levels), and a benchmark that
    # drops them would score a different set of bins. The inputs are clipped to the
    # filter's limit instead and `in_filter` records which rows it would have dropped.
    x = np.clip(x, -spec.Z_LIMIT, spec.Z_LIMIT)
    ece = arrays.get("ece")
    usable = np.ones(len(ns.GRID_S), dtype=bool)
    if ece is None:
        usable[:] = False
    else:
        lo, hi = float(ece.x[0]), float(ece.x[-1])
        # the row's window is [t - 50 ms, t): it must lie inside the ECE record
        usable &= (ns.GRID_S - 0.05 >= lo) & (ns.GRID_S <= hi)
    filled = list(spec.ALWAYS_MEAN_FILLED)
    for f in spec.INPUT_SPEC.fields:
        if f.canonical in built.missing:
            if f.kind == "profile":
                filled += [
                    f"ece_slow_channel_{k + 1}{elm_inputs.SUFFIX}" for k in range(48)
                ]
            else:
                filled.append(f.model_name)
    return Rows(
        shot,
        x.astype(np.float32),
        usable,
        in_filter,
        tuple(built.missing),
        dict(built.resolvers),
        tuple(filled),
        source_signature(paths, shot, norm),
    )


def save_rows(rows: Rows, path: Path) -> None:
    np.savez_compressed(
        path,
        x=rows.x,
        usable=rows.usable,
        in_filter=rows.in_filter,
        missing=np.array(rows.missing, dtype=str),
        filled=np.array(rows.filled, dtype=str),
        resolvers=json.dumps(rows.resolvers),
        source_signature=rows.source_signature,
    )


def load_rows(shot: int, path: Path, signature: str | None = None) -> Rows | None:
    """Cached rows, or None when absent, old, or built from different sources."""
    if not Path(path).exists():
        return None
    with np.load(path) as z:
        if "resolvers" not in z.files:
            return None
        saved = str(z["source_signature"]) if "source_signature" in z.files else ""
        if signature is not None and saved != signature:
            return None
        return Rows(
            shot,
            z["x"],
            z["usable"],
            z["in_filter"],
            tuple(str(v) for v in z["missing"]),
            json.loads(str(z["resolvers"])),
            tuple(str(v) for v in z["filled"]),
            saved,
        )


def cached_rows(paths: Paths, shot: int, norm: dict, cache_dir: Path) -> Rows:
    """The shot's `Rows`, read from `cache_dir` or built from the corpus and saved."""
    path = Path(cache_dir) / f"{shot}.npz"
    rows = load_rows(shot, path, source_signature(paths, shot, norm))
    if rows is None:
        rows = shot_rows(paths, shot, norm)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        save_rows(rows, path)
    return rows


def row_index(bins: labels.Bins, lag_rows: int = 0) -> np.ndarray:
    """Grid index of the row `lag_rows` after each bin's end row (see `row_part`)."""
    end = bins.t0 + WINDOW_MS + lag_rows * methods.ROW_MS
    return np.rint((end - ROW_T_MS[0]) / methods.ROW_MS).astype(int)


def bins_with_rows(bins: labels.Bins, usable: np.ndarray, lags=(0,)) -> labels.Bins:
    """The bins for which every row in `lags` exists and is usable."""
    keep = np.ones(len(bins.t0), dtype=bool)
    for lag in lags:
        k = row_index(bins, lag)
        ok = (k >= 0) & (k < len(ROW_T_MS))
        ok[ok] = usable[k[ok]]
        keep &= ok
    return methods.restrict_bins(bins, keep)


def usable_cover(usable: np.ndarray, cover: pd.DataFrame) -> pd.DataFrame:
    """`cover` cut to the time the usable rows summarise: the union of `[t - 50, t)`."""
    t = ROW_T_MS[usable]
    u0, u1 = labels.merge_intervals(t - WINDOW_MS, t) if len(t) else ([], [])
    c0, c1 = labels.merge_intervals(
        cover.t_start_ms.to_numpy(float), cover.t_end_ms.to_numpy(float)
    )
    return methods.cover_frame(*methods.intersect(c0, c1, u0, u1))


def published_risk(graph: dsm_pickle.DsmGraph, x: np.ndarray) -> np.ndarray:
    """`(rows, 4)` risk of an ELM within 5, 10, 20, 50 ms: `1 - S(h + 1)`."""
    surv = dsm_pickle.survival(graph, x, [h + T_OFFSET_MS for h in HORIZONS_MS])
    return 1.0 - surv


def window_labels(spans: pd.DataFrame) -> np.ndarray:
    """The review's call on the 50 ms ending at each grid row: 1, 0 or -1.

    A row is labelled when the window `[t - 50, t)` lies wholly inside one absent,
    individual or crowd span (the scored-bin rule at every row, not only at the bins'
    own offsets), -1 otherwise.
    """
    out = np.full(len(ROW_T_MS), -1, dtype=np.int8)
    start = ROW_T_MS - WINDOW_MS
    for r in spans.itertuples():
        if r.kind not in labels.SCORED_KINDS:
            continue
        inside = (start >= r.t_start - 1e-9) & (ROW_T_MS <= r.t_end + 1e-9)
        out[inside] = 0 if r.kind == "absent" else 1
    return out


# -------------------------------------------------------------- detector


class Detector(nn.Module):
    """The DSM embedding (Linear without bias, ReLU6, dropout) and one logit."""

    def __init__(self, n_in: int = N_COLUMNS, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Linear(n_in, hidden, bias=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(torch.clamp(self.embedding(x), 0.0, 6.0))
        return self.head(h).squeeze(-1)

    def load_published(self, graph: dsm_pickle.DsmGraph) -> None:
        """Start from the limited-input refit's embedding weights."""
        w = torch.as_tensor(graph.embedding[0], dtype=torch.float32)
        with torch.no_grad():
            self.embedding.weight.copy_(w)


@dataclass(frozen=True)
class FitConfig:
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 512
    dropout: float = 0.2
    seed: int = 20261003


def rows_for(rows: Rows, spans: pd.DataFrame):
    """Training rows of one shot: `(x, y)` of the usable, labelled grid rows."""
    y = window_labels(spans)
    keep = rows.usable & (y >= 0)
    return rows.x[keep], y[keep].astype(np.float32)


def bin_end_scores(rows_score: np.ndarray, bins: labels.Bins) -> np.ndarray:
    """Score of the row summarising each bin (the row at its end)."""
    return rows_score[row_index(bins)]


@torch.no_grad()
def predict(model: Detector, x: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(x))).numpy()


def fit_fold(rows, spans, bins, train, val, cfg: FitConfig, init=None, log=None):
    """Train a `Detector` on `train` shots; keep the best inner-val AUPRC epoch.

    `rows`, `spans` and `bins` are dicts by shot. Returns the best state, the
    F1-maximising threshold on the inner-validation bins at that epoch, and the history.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    xs, ys = zip(*(rows_for(rows[s], spans[s]) for s in train))
    x, y = np.concatenate(xs), np.concatenate(ys)
    model = Detector(dropout=cfg.dropout)
    if init is not None:
        model.load_published(init)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    best, best_state, best_thr, history = -1.0, None, 0.5, []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(x))
        losses = []
        for i in range(0, len(x), cfg.batch):
            idx = order[i : i + cfg.batch]
            loss = F.binary_cross_entropy_with_logits(
                model(torch.from_numpy(x[idx])), torch.from_numpy(y[idx])
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        truth, sc = [], []
        for s in val:
            b = bins_with_rows(bins[s], rows[s].usable)
            truth.append(b.truth)
            sc.append(bin_end_scores(predict(model, rows[s].x), b))
        truth, sc = np.concatenate(truth), np.concatenate(sc)
        ap = score.average_precision(truth, sc)
        thr, f1 = score.best_threshold(truth, sc)
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "val_auprc": ap,
                "val_f1": f1,
            }
        )
        if log:
            log(history[-1])
        if ap > best:
            best, best_thr = ap, thr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    return best_state, best_thr, history


def load_norm(paths: Paths) -> dict:
    return json.loads((paths.models / SLUG / spec.ARTIFACTS[1]).read_text())
