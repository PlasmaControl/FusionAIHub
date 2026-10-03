"""`elm-dsm`: the lab's ELM time-to-event model on the reviewed shots, as a detector.

The model is Deep Survival Machines (auton-survival, LogNormal, three components, a
128-unit ReLU6 embedding with dropout 0.2) fitted by the labeler on the ELM survival
rows of the `wpqh_elm_hiro` project: 60 non-BES columns trained on 1 ms rows,
served as 50 ms means on a 25 ms grid. Three things are done with it here.

* **Own target** (`legacy_own_target`): the limited-input refit on its original
  early-stopping validation rows (upstream calls them test), the time to the next ELM,
  AUROC at each horizon with 95 % physical-shot-bootstrap intervals. Upstream split
  identifiers name phases, and phases of one physical shot can occur on both sides.
* **elm-dsm refit on the reviewed bins**: the model served from the corpus as
  `labeler.models.d3d_elm_time_to_event_dsm` serves it (`shot_rows`), its 50 ms risk at
  a bin's start read as the bin's score. This is an offline risk score: centered
  NBI smoothing incorporates a row 25 ms later, so it is not a causal forecast.
* **elm-dsm (60-input 1×128 refit, detection)** (`Detector`, `fit_fold`): a
  reduced-input adaptation with a single 128-unit embedding and one logit,
  trained with cross-entropy to say whether the
  50 ms ending at a row's time stamp is present (ELMy) or absent in the review: the
  target is occupancy. The native source instead used 124 inputs, [100, 1000]
  embedding layers and 1 ms rows; this adaptation is not its objective-only
  retrain. Trained by shot-grouped cross-validation on the same folds as
  `elm-ours`; the threshold is chosen on each fold's inner-validation shots.

**Rows.** Each row of the 25 ms grid summarises the 50 ms before its time stamp (a
`[t - 50, t)` mean), so the row at a bin's end summarises the bin and the row at its
start the bin before it.

This is **elm-dsm refit, 60 of the original 124 inputs**, not the original
124-input checkpoint. Missing `ip` and `bt` can be fetched into the isolated round-four
store by `elm_dsm_fetch.py`; no corpus or production feature file is changed. Remaining
missing columns, including the two photodiodes (`pcphd02/03`) in the survival and
historical serving paths, are filled at the training mean. Repaired detection rows
use measured PCPHD02/03, or the recorded FS02/03 substitutes, and repaired density.
Inputs outside the refit's row filter are clipped rather than
dropped; the evaluation records missingness, filter failures and the risk scale.

Historical variants (elm-dsm refit and detectors reusing source statistics or weights)
use upstream feature means and standard deviations computed before its split.
They inherit feature-statistics exposure to blind-cohort shots 190532 and 190646;
this is not reviewed-label leakage. The confirmatory detection fit uses raw rows,
training-partition statistics and independent random weights, with no source reuse.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, replace
from functools import lru_cache
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
from ..timebase import window_mean
from . import labels, methods, score

SLUG = spec.SLUG
SPLIT_PKL = Path("/projects/EKOLEMEN/wpqh_elm_hiro/data/train_test_split_model10.pkl")
HORIZONS_MS = spec.HORIZONS_MS
T_OFFSET_MS = spec.T_OFFSET_MS
N_COLUMNS = len(spec.COLUMNS)
DISPLAY_NAME = "elm-dsm refit"
SERVING = {
    "interpretation": "offline risk score, not a causal forecast",
    "source_refit_trained_row_ms": 1.0,
    "detection_trained_window_ms": spec.SERVING_WINDOW_MS,
    "training_note": (
        "survival refit trained on source 1 ms rows; detection variants refitted "
        "on reviewed 50 ms-mean rows"
    ),
    "serving_window_ms": spec.SERVING_WINDOW_MS,
    "serving_grid_ms": spec.DT_S * 1000.0,
    "nbi_boxcar_ms": spec.NBI_BOXCAR_MS,
    "nbi_lookahead_ms": spec.NBI_LOOKAHEAD_MS,
    "dalpha_input": "Survival refit and historical detectors mean-fill PCPHD02/03; "
    "the source-isolated reduced-input detector uses repaired measured photodiodes "
    "or recorded FS02/03 substitutes (see detection_input_repair)",
    "native_input_columns": 124,
    "native_embedding_layers": [100, 1000],
    "native_trained_row_ms": 1.0,
    "refit_input_columns": N_COLUMNS,
    "refit_embedding_layers": [128],
}
PREPROCESSING_EXPOSURE = {
    "applies_to": [
        "elm-dsm refit",
        "elm-dsm (source statistics, detection)",
        "elm-dsm (source weights and statistics, detection)",
    ],
    "scope": "upstream feature means/std computed before source split",
    "normalization_physical_shots": len(spec.NORMALIZATION_SHOTS),
    "blind_cohort_shots": spec.MEMBERSHIP["blind_cohort_normalization_shots"],
    "source": spec.MEMBERSHIP["normalization_source"],
    "role": "feature-statistics exposure, independent of reviewed-label CV",
    "decision": "retain as supplemental; confirmatory detection refits preprocessing",
}
ROWS_SCHEMA = 2
ROW_T_MS = ns.GRID_S * 1000.0
WINDOW_MS = methods.WINDOW_MS


# ------------------------------------------------------------ own target


def physical_shot_ids(phase_ids) -> np.ndarray:
    """Decode upstream ``<shot>_<phase>`` IDs and their old numeric rendering.

    The upstream data-processing notebook stores strings such as ``190643_0``.
    Python's ``int`` accepts underscores and renders that string as ``1906430``;
    this older evaluation representation must therefore be divided by ten. Already
    physical six-digit IDs are retained. Do not treat phases as independent shots.
    """
    out = []
    for value in np.asarray(phase_ids).ravel():
        text = str(value)
        if "_" in text:
            shot, phase = text.rsplit("_", 1)
            if not shot.isdigit() or not phase.isdigit():
                raise ValueError(f"invalid upstream phase identifier: {value!r}")
            number = int(shot)
        else:
            number = int(value)
            if 1_000_000 <= number < 10_000_000:
                number //= 10
        if not 100_000 <= number < 1_000_000:
            raise ValueError(f"invalid physical shot identifier: {value!r}")
        out.append(number)
    return np.asarray(out, dtype=np.int64)


def split_identity(train_phase_ids, test_phase_ids) -> dict:
    """Preserve phase IDs while exposing physical membership and phase split leakage."""
    phases = {
        "train": np.unique(train_phase_ids),
        "test": np.unique(test_phase_ids),
    }
    shots = {
        k: sorted(int(s) for s in np.unique(physical_shot_ids(v)))
        for k, v in phases.items()
    }
    return {
        "phase_id_encoding": (
            "upstream <physical shot>_<phase>; old int(string) rendering is "
            "physical shot * 10 + phase for single-digit phases"
        ),
        "split_phase_ids": {k: [str(v) for v in ids] for k, ids in phases.items()},
        "split_phase_records": {k: len(v) for k, v in phases.items()},
        "split_shots": shots,
        "split_physical_shot_counts": {k: len(v) for k, v in shots.items()},
        "physical_shots_in_both_split_sides": sorted(
            set(shots["train"]) & set(shots["test"])
        ),
        "selection_role": "early-stopping validation (upstream key: test)",
        "selection_note": (
            "The original test rows were passed as val_data for early stopping; "
            "own-target metrics are validation evidence, not untouched held-out evidence."
        ),
    }


def split_overlap(identity: dict, reviewed_shots, cohort: pd.DataFrame) -> dict:
    """Review and fixed-cohort membership of the original refit's physical shots."""
    split_shots = identity["split_shots"]
    review = {k: sorted(set(reviewed_shots) & set(v)) for k, v in split_shots.items()}
    return {
        "reviewed_shots_in_published_split": {k: len(v) for k, v in review.items()},
        "reviewed_shot_ids_in_published_split": review,
        "cohort_physical_shot_overlap": {
            k: {
                split: sorted(
                    set(v) & set(map(int, cohort.loc[cohort.split == split, "shot"]))
                )
                for split in ("train", "val", "test")
            }
            for k, v in split_shots.items()
        },
        "overlap_scope": (
            "original refit training and early-stopping validation, separately from "
            "reviewed-label detector CV, which excludes cohort test shots"
        ),
    }


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
    """Refit on its early-stopping validation rows, with physical-shot intervals.

    Rows are upstream's 1 ms rows. A case at horizon `h` is a row whose next ELM is
    within `h` ms, a control one whose next ELM is later than `h`; a row censored
    inside `h` is neither (it is unknown whether an ELM came). The shots are the
    split's own phase ids; intervals group all phases of each physical shot together.
    """
    with open(split_pkl, "rb") as fh:
        d = pickle.load(fh)
    cols = list(elm_inputs.column_indices("no_bes"))
    x = np.asarray(d["test_final_x_normalized"], dtype=np.float64)[:, cols]
    e = np.asarray(d["test_final_e"], dtype=float).ravel() == 1.0
    t = np.asarray(d["test_final_t"], dtype=float).ravel()
    phase = np.asarray(d["test_final_shots_list"]).ravel()
    shot = physical_shot_ids(phase)
    graph = dsm_pickle.load_dsm(model_dir / spec.ARTIFACTS[0])
    surv = dsm_pickle.survival(graph, x, [h + T_OFFSET_MS for h in HORIZONS_MS])
    n_shots = int(np.unique(shot).size)
    out: dict = {
        "display_name": DISPLAY_NAME,
        "split": str(split_pkl),
        "rows": len(t),
        "shots": n_shots,
        "phase_records": len(np.unique(phase)),
        "bootstrap_unit": "physical shot (all phase records grouped)",
        **split_identity(d["train_final_shots_list"], phase),
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
        bootstrap = score.bootstrap_summary(
            {"auroc": reps},
            n_shots=len(shots),
            positive_shots=int((pos.sum(axis=1) > 0).sum()),
        )
        out["horizons"][f"h{int(h)}ms"] = {
            "horizon_ms": h,
            "queried_at_ms": h + T_OFFSET_MS,
            "cases": int(case.sum()),
            "controls": int(control.sum()),
            "censored_within_h": int((~e & (t <= h)).sum()),
            "auroc": exact,
            "auroc_histogram": point,
            "auroc_ci95": bootstrap["ci95"]["auroc"],
            "bootstrap_draw_counts": bootstrap["bootstrap_draw_counts"],
            "descriptive_only": bootstrap["descriptive_only"],
            "positive_shots": bootstrap["positive_shots"],
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
    x: np.ndarray  # (240, 60): raw or normalised, according to source_signature
    usable: np.ndarray  # full ECE window, plus measured repaired-input coverage
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


def source_signature(paths: Paths, shot: int, norm: dict | None) -> str:
    """Cache key that changes when Ip/Bt is fetched, inputs change, or norms change.

    Large read-only source files use metadata; the small fetched feature file uses a
    content hash. The schema version invalidates the old cache without deleting it.
    """
    files = [paths.corpus_file(shot), *resolve_archive.ARCHIVE_FILES]
    records = []
    for path in files:
        st = path.stat() if path.exists() else None
        records.append(
            (str(path), None if st is None else (st.st_size, st.st_mtime_ns))
        )
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
        "serving": SERVING,
        "preprocessing_exposure": PREPROCESSING_EXPOSURE,
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


def shot_rows(paths: Paths, shot: int, norm: dict | None) -> Rows:
    """Adapter rows, or unnormalized, unclipped rows when ``norm`` is None.

    Raw rows use only fixed input conversion/smoothing. Identity constants do not
    import any source statistics; missing columns remain flagged for fold fitting.
    """
    arrays = shot_features(paths, shot)
    built = spec.INPUT_SPEC.build(arrays, ns.GRID_S)
    identity = {"mean": [0.0] * N_COLUMNS, "std": [1.0] * N_COLUMNS}
    x, in_filter = spec.preprocess(built, identity if norm is None else norm)
    # Upstream dropped training rows with any |z| > 10; many rows of these shots are
    # that far out (a different era's ECE and actuator levels), and a benchmark that
    # drops them would score a different set of bins. The inputs are clipped to the
    # filter's limit instead and `in_filter` records which rows it would have dropped.
    if norm is None:
        in_filter = np.ones(len(x), dtype=bool)
    else:
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
        x.astype(np.float64 if norm is None else np.float32),
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


def cached_rows(paths: Paths, shot: int, norm: dict | None, cache_dir: Path) -> Rows:
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


def future_review_targets(spans, times, horizon_ms: float, *, onsets=False):
    """Horizon-specific review target in ``(t, t+h]``, unknown outside coverage.

    Occupancy asks whether any present interval intersects the future window;
    onset asks whether a non-crowd start is strictly after t and at or before t+h.
    Every window must be fully reviewed. Crowd time is unknown for onset targets.
    Span starts are annotation boundaries, not independently verified ELM onsets.
    """
    times = np.asarray(times, dtype=float)
    if horizon_ms <= 0:
        raise ValueError("forecast horizon must be positive")
    kinds = ("absent", "non_crowd") if onsets else labels.SCORED_KINDS
    known = spans[spans.kind.isin(kinds)]
    a, b = labels.merge_intervals(known.t_start, known.t_end, tol=1e-9)
    out = np.full(len(times), -1, dtype=np.int8)
    for lo, hi in zip(a, b, strict=True):
        out[(times >= lo) & (times + horizon_ms <= hi)] = 0
    for row in known[known.kind != "absent"].itertuples():
        hit = (row.t_start > times) if onsets else (row.t_end > times)
        hit &= row.t_start <= times + horizon_ms
        out[(out >= 0) & hit] = 1
    return out


@lru_cache(maxsize=1)
def upstream_photodiodes():
    """Read the immutable source photodiode records once per process."""
    path = Path("/projects/EKOLEMEN/wpqh_elm_hiro/data/dalpha_wpqh.pkl")
    with path.open("rb") as handle:
        return pickle.load(handle)


def repair_detection_inputs(paths: Paths, rows: Rows) -> Rows:
    """Supply measured D-alpha and fast V2/V3 only for isolated detector refits.

    Keep historical survival serving unchanged. PCPHD02/03 are preferred; FS02/03
    50 ms means are explicitly named substitutes where photodiodes are missing.
    DENV2F/3F means retain their native ordinate. A live source-unit probe reports
    V, unlike the cm^-2 training CO2 columns; numerical agreement is audited by
    elm_dsm_input_audit.py. No physical conversion is guessed. Fold normalization
    fits measured values directly. Missing windows narrow usable rows.
    """
    from . import inputs, prepare

    path = prepare.signals_dir(paths) / f"{rows.shot}.npz"
    with np.load(path) as record:
        tf, ti = record["t_fs_ms"], record["t_int_ms"]
        fs, density = record["filterscopes"], record["interferometer"]
    x = rows.x.copy()
    usable = rows.usable.copy()
    filled, missing = set(rows.filled), set(rows.missing)
    resolvers = dict(rows.resolvers)
    index = {name: i for i, name in enumerate(spec.COLUMNS)}
    sources = [(path, sha256_of(path))]
    for j, name in enumerate(("pcphd02", "pcphd03")):
        cache = (
            paths.root
            / "round4/elm/dsm/native_photodiodes"
            / (f"{rows.shot}_{name}.npz")
        )
        if cache.exists():
            with np.load(cache) as photo:
                times, values = photo["x"], photo["y"].ravel()
            source = f"{name.upper()} fetched photodiode"
            sources.append((cache, sha256_of(cache)))
        else:
            photo = upstream_photodiodes().get(str(rows.shot), {}).get(name, {})
            if np.asarray(photo.get("data", [])).size > 2:
                times, values = photo["times"], photo["data"]
                source = f"{name.upper()} upstream photodiode"
            else:
                times, values = tf, fs[j]
                source = f"FS{j + 2:02d} substitute for {name.upper()}"
        value = window_mean(times, np.ravel(values), ROW_T_MS - WINDOW_MS, WINDOW_MS)
        col = name + elm_inputs.SUFFIX
        x[:, index[col]] = np.nan_to_num(value)
        usable &= np.isfinite(value)
        filled.discard(col)
        resolvers[name] = source + "; 50 ms mean"
    for j, chord in enumerate(("v2", "v3")):
        value = window_mean(ti, density[j], ROW_T_MS - WINDOW_MS, WINDOW_MS)
        col = f"co2_density_slow_{chord}{elm_inputs.SUFFIX}"
        if np.nanmedian(np.abs(density[j])) > inputs.BAD_DENSITY:
            # Same fixed failed-digitiser screen as elm-ours; it is not tuned
            # against labels and the omitted column is explicit in the audit.
            x[:, index[col]] = 0.0
            filled.add(col)
            missing.add(f"co2_{chord}")
            resolvers[f"co2_{chord}"] = f"DENV{j + 2}F rejected: failed digitiser"
            continue
        x[:, index[col]] = np.nan_to_num(value)
        usable &= np.isfinite(value)
        filled.discard(col)
        missing.discard(f"co2_{chord}")
        resolvers[f"co2_{chord}"] = f"DENV{j + 2}F; 50 ms mean; native ordinate"
    signature = hashlib.sha256(
        json.dumps(
            [rows.source_signature, [(str(p), s) for p, s in sources], resolvers],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return replace(
        rows,
        x=x,
        usable=usable,
        missing=tuple(sorted(missing)),
        filled=tuple(sorted(filled)),
        resolvers=resolvers,
        source_signature=signature,
    )


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
    device: str = "cpu"
    #: `raw` keeps the best inner-validation AUPRC epoch (the recipe of the reported
    #: row); `post_warmup_trailing3` keeps the epoch ending the best three-epoch mean
    #: AUPRC window that lies wholly at or after epoch `warmup`.
    selection: str = "raw"
    warmup: int = 0


def selection_criterion(history: list[dict], rule: str, warmup: int) -> float | None:
    """The checkpoint criterion after the last epoch in `history`, or None."""
    if rule == "raw":
        return history[-1]["val_auprc"]
    if rule != "post_warmup_trailing3":
        raise ValueError(f"unknown checkpoint selection rule: {rule}")
    if len(history) < warmup + 3:
        return None
    values = [row["val_auprc"] for row in history[-3:]]
    return float(np.mean(values)) if np.isfinite(values).all() else None


def fit_detection_normalization(rows, spans, train) -> dict:
    """Fit measured-column statistics on usable labeled optimizer-training rows.

    Inner-validation and outer-test shots never enter these statistics. Missing
    columns contribute no values; columns with no measurements or zero variance
    get unit scale. Each fold records its exact fitting shots and row counts.
    """
    values = [[] for _ in spec.COLUMNS]
    for shot in train:
        row = rows[shot]
        keep = row.usable & (window_labels(spans[shot]) >= 0)
        filled = set(row.filled)
        for j, name in enumerate(spec.COLUMNS):
            if name not in filled:
                v = row.x[keep, j]
                values[j].append(v[np.isfinite(v)])
    mean, std, count = [], [], []
    for parts in values:
        v = np.concatenate(parts) if parts else np.array([], dtype=float)
        count.append(len(v))
        mean.append(float(v.mean()) if len(v) else 0.0)
        scale = float(v.std()) if len(v) else 0.0
        std.append(scale if scale > 0 else 1.0)
    return {
        "columns": list(spec.COLUMNS),
        "mean": mean,
        "std": std,
        "measured_rows_per_column": count,
        "fit_shots": sorted(map(int, train)),
        "scope": "usable labeled optimizer-training rows; inner validation excluded",
        "source_parameters_reused": False,
    }


def normalize_detection_rows(rows: Rows, norm: dict) -> Rows:
    """Mean-fill and clip using only the supplied detection fold's statistics."""
    if list(norm["columns"]) != list(spec.COLUMNS):
        raise ValueError("detection normalization column order does not match inputs")
    x = (rows.x - np.asarray(norm["mean"])) / np.asarray(norm["std"])
    filled = [j for j, name in enumerate(spec.COLUMNS) if name in rows.filled]
    x[:, filled] = 0.0
    in_filter = np.isfinite(x).all(axis=1) & (np.abs(x) <= spec.Z_LIMIT).all(axis=1)
    x = np.clip(np.nan_to_num(x), -spec.Z_LIMIT, spec.Z_LIMIT).astype(np.float32)
    return replace(rows, x=x, in_filter=in_filter)


def historical_detector_sources(detectors: dict) -> dict[str, str]:
    """Map retained historical artifacts without reclassifying a later clean fit.

    The first migration predates the explicit exposed name. Every subsequent fit
    must prefer the already retained artifact; a fresh store has no supplemental
    historical rows to preserve.
    """
    out = {}
    exposed = "elm-dsm-detect-exposed"
    if exposed in detectors:
        out[exposed] = exposed
    elif (
        "elm-dsm-detect" in detectors
        and detectors["elm-dsm-detect"].get("source_normalization_reused") is not False
    ):
        out[exposed] = "elm-dsm-detect"
    if "elm-dsm-detect-init" in detectors:
        out["elm-dsm-detect-init"] = "elm-dsm-detect-init"
    return out


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
    device = next(model.parameters()).device
    return torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()


def fit_fold(rows, spans, bins, train, val, cfg: FitConfig, init=None, log=None):
    """Train a `Detector` on `train` shots; keep the best inner-val AUPRC epoch.

    `rows`, `spans` and `bins` are dicts by shot. Returns the best state, the
    F1-maximising threshold on the inner-validation bins at that epoch, and the history.
    """
    if set(train) & set(val):
        raise ValueError("optimizer-training and inner-validation shots overlap")
    if init is not None:
        raise ValueError("confirmatory detection cannot reuse source model parameters")
    norm = fit_detection_normalization(rows, spans, train)
    rows = {s: normalize_detection_rows(r, norm) for s, r in rows.items()}
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    xs, ys = zip(*(rows_for(rows[s], spans[s]) for s in train))
    x, y = np.concatenate(xs), np.concatenate(ys)
    model = Detector(dropout=cfg.dropout).to(cfg.device)
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
                model(torch.from_numpy(x[idx]).to(cfg.device)),
                torch.from_numpy(y[idx]).to(cfg.device),
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
        criterion = selection_criterion(history, cfg.selection, cfg.warmup)
        if cfg.selection != "raw":
            history[-1]["selection_score"] = criterion
        if log:
            log(history[-1])
        if criterion is not None and criterion > best:
            best, best_thr = criterion, thr
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    if best_state is None:
        raise ValueError("no finite checkpoint-selection score; cannot select a fit")
    return best_state, best_thr, history


def load_norm(paths: Paths) -> dict:
    return json.loads((paths.models / SLUG / spec.ARTIFACTS[1]).read_text())
