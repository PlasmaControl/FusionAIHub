#!/usr/bin/env python
"""Audit and score the original 124-input DSM on its native 1 ms inputs.

The exact-export panel never reconstructs, clips or fills inputs. A separate
reconstructed-input panel uses original diagnostic records and the source's
6000-bin sample-count averaging. Source-export agreement is reported explicitly.
All results are continuous-score AUROCs, with physical-shot bootstrap intervals.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np
import torch

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import dsm, labels, prepare, score
from labeler.models import elm_inputs
from labeler.models.runners import dsm_pickle

SOURCE = Path("/projects/EKOLEMEN/wpqh_elm_hiro")
H5 = Path("/scratch/gpfs/EKOLEMEN/hackathon/raw_h5_files")
COLUMNS = elm_inputs.CURRENT_DIAGNOSTIC_ORDER
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/dsm/native_evaluation.json"
GROUP_COLUMNS = {
    "ip": [COLUMNS[0]],
    "mag_pcb_coil": [COLUMNS[1]],
    "gas": [COLUMNS[2]],
    "p_inj": [COLUMNS[3]],
    "t_inj": [COLUMNS[4]],
    "ech": [COLUMNS[5]],
    "ece_slow": [n for n in COLUMNS if n.startswith("ece_")],
    "co2_density_slow": [n for n in COLUMNS if n.startswith("co2_")],
    "bes_slow": [n for n in COLUMNS if n.startswith("bes_")],
}


def selected_graph():
    """Read the actual native setting selected for the shipped wpqh1 graphs."""
    path = SOURCE / "hiro_scripts/models/model9.pkl"
    with path.open("rb") as fh:
        payload = dsm_pickle.RestrictedUnpickler(fh).load()
    tm = dsm_pickle._find_torch_model(payload[1][0])
    f64 = lambda t: t.detach().cpu().double().numpy()
    graph = dsm_pickle.DsmGraph(
        k=int(tm.k),
        dist=tm.dist,
        temp=float(tm.temp),
        embedding=tuple(f64(w.weight) for w in tm.embedding if hasattr(w, "weight")),
        gate=f64(tm.gate["1"][0].weight),
        scaleg=(f64(tm.scaleg["1"][0].weight), f64(tm.scaleg["1"][0].bias)),
        shapeg=(f64(tm.shapeg["1"][0].weight), f64(tm.shapeg["1"][0].bias)),
        shape=f64(tm.shape["1"]),
        scale=f64(tm.scale["1"]),
    )
    checks = {}
    for part, expected in (
        ("embedding", list(graph.embedding)),
        ("shapeg", [graph.shapeg[0]]),
        ("scaleg", [graph.scaleg[0]]),
        ("gate", [graph.gate]),
    ):
        path = SOURCE / f"hiro_scripts/wpqh1_{part}.h5"
        weights = []
        with h5py.File(path) as h:

            def collect(name, obj, weights=weights):
                if isinstance(obj, h5py.Dataset) and name.endswith("kernel:0"):
                    weights.append(np.asarray(obj).T)

            h.visititems(collect)
        error = max(
            float(np.max(np.abs(a - b))) for a, b in zip(weights, expected, strict=True)
        )
        if error > 1e-6:
            raise ValueError(f"native setting does not match {part} export: {error}")
        checks[part] = {
            "path": str(path),
            "sha256": sha256_of(path),
            "max_abs_weight_error": error,
        }
    return graph, {
        "checkpoint": str(SOURCE / "hiro_scripts/models/model9.pkl"),
        "checkpoint_sha256": sha256_of(SOURCE / "hiro_scripts/models/model9.pkl"),
        "selected_parameter_index": 1,
        "parameters": payload[1][3],
        "shipped_export_weight_checks": checks,
        "activation": "original PyTorch ReLU6; Keras conversion uses unbounded ReLU",
    }


def risk(graph, x):
    return np.concatenate(
        [dsm.published_risk(graph, x[i : i + 4096]) for i in range(0, len(x), 4096)]
    )


def areas_by_shot(truth, values, shots):
    """Exact rank point plus quantile-histogram shot CI, with error recorded."""
    truth = np.asarray(truth, dtype=bool)
    ids, pos, neg = dsm.auroc_by_shot(values, truth, np.ones(len(truth), bool), shots)
    draws = score.draws(len(ids))
    counts = np.stack([np.bincount(d, minlength=len(ids)) for d in draws])
    bp, bn = counts @ pos, counts @ neg
    denominator = bp.sum(axis=1) * bn.sum(axis=1)
    numerator = (bp * (np.cumsum(bn, axis=1) - 0.5 * bn)).sum(axis=1)
    reps = np.divide(
        numerator, denominator, out=np.full(len(draws), np.nan), where=denominator > 0
    )
    point = score.roc_auc(truth, values)
    return {
        "auroc": point,
        "auroc_ci95": score._ci(reps),
        "auprc": score.average_precision(truth, values),
        "rows": len(truth),
        "cases": int(truth.sum()),
        "controls": int((~truth).sum()),
        "shots": list(map(int, ids)),
        "n_shots": len(ids),
        "bootstrap_replicates": len(draws),
        "bootstrap_unit": "physical shot",
        "ci_histogram_bins": pos.shape[1],
        "auroc_histogram_error": dsm.hist_auroc(pos, neg) - point,
    }


def photo_record(photodiodes, paths, shot, name):
    value = photodiodes.get(str(shot), {}).get(name, {})
    if np.asarray(value.get("data", [])).size > 2:
        return np.ravel(value["times"]), np.ravel(value["data"]), "upstream pickle"
    path = paths.root / "round4/elm/dsm/native_photodiodes" / f"{shot}_{name}.npz"
    if path.exists():
        with np.load(path) as z:
            return z["x"], z["y"].ravel(), str(path)
    return None


def audit(shots, photodiodes, paths, export_shots):
    records = {}
    for shot in shots:
        path = H5 / f"{shot}_slow.h5"
        missing, groups = [], {}
        h = h5py.File(path) if path.exists() else None
        try:
            for group, names in GROUP_COLUMNS.items():
                shape = (
                    list(h[group]["block0_values"].shape)
                    if h is not None and group in h
                    else []
                )
                groups[group] = shape
                if np.prod(shape) <= 1 or not shape:
                    missing.extend(names)
                elif len(names) > 1 and shape[-1] < len(names):
                    missing.extend(names[shape[-1] :])
        finally:
            if h is not None:
                h.close()
        photo_sources = {}
        for name in ("pcphd02", "pcphd03"):
            got = photo_record(photodiodes, paths, shot, name)
            if got is None:
                missing.append(name + elm_inputs.SUFFIX)
            else:
                photo_sources[name] = {"source": got[2], "samples": len(got[0])}
        records[str(shot)] = {
            "original_h5": str(path),
            "original_h5_exists": path.exists(),
            "group_shapes": groups,
            "photodiodes": photo_sources,
            "exact_native_export_available": shot in export_shots,
            "missing_native_export_columns": []
            if shot in export_shots
            else list(COLUMNS),
            "missing_reconstruction_columns": sorted(missing),
            "reconstructable": not missing,
            "blocked_reason": None
            if not missing
            else "original diagnostic records missing; adapter 50 ms means cannot recover 1 ms inputs",
        }
    return records


def source_average(times, values):
    """Reproduce source count-based 6000-bin means, without interpolation or fills."""
    times, values = np.asarray(times), np.asarray(values)
    left, right = np.argmax(times >= 0), np.argmax(times >= 5999)
    values = values[left:right]
    edges = (np.arange(6001) * len(values) / 6000).astype(int)
    out = np.full((6000, values.shape[1]), np.nan)
    for i, (a, b) in enumerate(pairwise(edges)):
        if b > a:
            out[i] = values[a:b].mean(axis=0)
    return out


def reconstruct(shot, photodiodes, paths, norm):
    raw = np.full((6000, 124), np.nan)
    index = {n: i for i, n in enumerate(COLUMNS)}
    with h5py.File(H5 / f"{shot}_slow.h5") as h:
        for group, names in GROUP_COLUMNS.items():
            g = h[group]
            value = np.asarray(g["block0_values"])
            if group == "mag_pcb_coil":
                value = 1.69861e-5 * value[:, :1]
            elif group in ("p_inj", "t_inj"):
                value = value.reshape(len(value), -1, 8).sum(axis=2)[:, :1]
            elif group == "ech":
                value = value[:, 1:2]
            elif group == "co2_density_slow":
                chords = [s.decode() for s in np.asarray(g["block0_items"])]
                value = value[:, [chords.index(c) for c in ("r0", "v1", "v2", "v3")]]
            elif len(names) == 1:
                value = value[:, :1]
            raw[:, [index[n] for n in names]] = source_average(g["axis1"][:], value)
    for name in ("pcphd02", "pcphd03"):
        times, values, _ = photo_record(photodiodes, paths, shot, name)
        raw[:, index[name + elm_inputs.SUFFIX]] = source_average(
            times, values[:, None]
        )[:, 0]
    mean = np.array([norm[n][0] for n in COLUMNS])
    std = np.array([norm[n][1] for n in COLUMNS])
    x = (raw - mean) / std
    # compiled_model9 smooths normalized NBI columns, preserving zero padding in
    # normalized units. Smoothing raw values before normalization changes edges.
    for j in (3, 4):
        x[:, j] = np.convolve(x[:, j], np.ones(100) / 100, "same")
    return raw, x


def review_targets(spans, times):
    truth = np.full(len(times), -1, dtype=np.int8)
    for r in spans.itertuples():
        if r.kind in labels.SCORED_KINDS:
            inside = (times >= r.t_start) & (times + 1 <= r.t_end)
            truth[inside] = int(r.kind != "absent")
    return truth


def review_summary(parts):
    truth, values, shots = zip(*parts)
    return {
        f"h{int(h)}ms": areas_by_shot(
            np.concatenate(truth), np.concatenate(values)[:, j], np.concatenate(shots)
        )
        for j, h in enumerate(dsm.HORIZONS_MS)
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--skip-own-target", action="store_true")
    args = ap.parse_args()
    paths = Paths.from_env()
    torch.set_num_threads(4)
    graph, checkpoint = selected_graph()
    with (SOURCE / "data/compiled_model9.pkl").open("rb") as fh:
        source = pickle.load(fh)
    with (SOURCE / "data/dalpha_wpqh.pkl").open("rb") as fh:
        photodiodes = pickle.load(fh)
    review = labels.review_table(prepare.review_csv(paths))
    shots = sorted(map(int, review.shot.unique()))
    source_shots = dsm.physical_shot_ids(source["final_shots_list"])
    export_shots = sorted(set(shots) & set(map(int, source_shots)))
    coverage = audit(shots, photodiodes, paths, export_shots)
    phases = np.unique(source["final_shots_list"])
    cutoff = int(
        np.argmax(source["final_shots_list"] == phases[int(0.9 * len(phases))])
    )
    source_train = sorted(set(map(int, source_shots[cutoff:])))
    source_val = sorted(set(map(int, source_shots[:cutoff])))
    record = {
        "git": git_sha(full=True),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "script_sha256": sha256_of(__file__),
        "checkpoint": checkpoint,
        "input_columns": list(COLUMNS),
        "n_input_columns": 124,
        "grid_ms": 1.0,
        "nbi_boxcar_ms": 100.0,
        "nbi_lookahead_ms": 50.0,
        "source_native_export": str(SOURCE / "data/compiled_model9.pkl"),
        "source_native_export_sha256": sha256_of(SOURCE / "data/compiled_model9.pkl"),
        "source_split": {
            "optimizer_train_shots": source_train,
            "early_stopping_validation_shots": source_val,
            "train_rows": len(source_shots) - cutoff,
            "validation_rows": cutoff,
            "implementation": "train_elm_model.py fits last 10% phases, validates first 90%; historical swapped sides preserved",
        },
        "exposure": {
            "normalization_shots": sorted(set(map(int, source_shots))),
            "reviewed_optimizer_train": sorted(set(shots) & set(source_train)),
            "reviewed_checkpoint_selection": sorted(set(shots) & set(source_val)),
            "reviewed_normalization": export_shots,
            "interpretation": "source-exposed historical native checkpoint; not confirmatory held-out evidence",
        },
        "coverage": coverage,
        "reviewed_exact_export": {},
        "reviewed_reconstructed": {},
        "reconstruction_export_agreement": {},
    }
    parts = []
    for shot in export_shots:
        keep = source_shots == shot
        times = source["final_times_list"][keep]
        truth = review_targets(review[review.shot == shot], times)
        x = source["final_x_normalized"][keep]
        ok = truth >= 0
        coverage[str(shot)]["exact_export_reviewed_rows"] = int(ok.sum())
        if not ok.any():
            coverage[str(shot)]["exact_export_blocked_reason"] = (
                "source-export phase rows do not overlap a scored expert-review span"
            )
            continue
        values = risk(graph, x[ok])
        parts.append((truth[ok], values, np.full(ok.sum(), shot)))
        print("exact export", shot, ok.sum(), flush=True)
    record["reviewed_exact_export"] = {
        "method": "elm-dsm native exact export",
        "source_exposed": True,
        "threshold": "none; continuous-score metrics only",
        "horizons": review_summary(parts),
    }
    reconstructed = []
    for shot in shots:
        if not coverage[str(shot)]["reconstructable"]:
            continue
        raw, x = reconstruct(shot, photodiodes, paths, source["normalizations"])
        truth = review_targets(review[review.shot == shot], np.arange(6000))
        checked = [i for i, n in enumerate(COLUMNS) if not n.startswith("pcphd")]
        co2 = [i for i, n in enumerate(COLUMNS) if n.startswith("co2_")]
        ok = (
            (truth >= 0)
            & np.isfinite(x).all(axis=1)
            & (np.abs(x[:, checked]) <= 10).all(axis=1)
        )
        ok &= ((raw[:, co2] >= 0) & (raw[:, co2] <= 1e15)).all(axis=1)
        if not ok.any():
            coverage[str(shot)]["blocked_reason"] = (
                "no finite reviewed rows within native normalization/filter domain"
            )
            continue
        values = risk(graph, x[ok])
        reconstructed.append((truth[ok], values, np.full(ok.sum(), shot)))
        coverage[str(shot)]["reconstructed_reviewed_rows"] = int(ok.sum())
        if shot in export_shots:
            keep = source_shots == shot
            times = np.asarray(source["final_times_list"][keep], int)
            difference = x[times] - source["final_x_normalized"][keep]
            record["reconstruction_export_agreement"][str(shot)] = {
                n: {
                    "median_abs_normalized_difference": float(
                        np.nanmedian(np.abs(difference[:, j]))
                    ),
                    "max_abs_normalized_difference": float(
                        np.nanmax(np.abs(difference[:, j]))
                    ),
                }
                for j, n in enumerate(COLUMNS)
            }
        print("reconstructed", shot, ok.sum(), flush=True)
    record["reviewed_reconstructed"] = {
        "method": "elm-dsm native reconstructed inputs",
        "threshold": "none; continuous-score metrics only",
        "role": "separate sensitivity panel; exact upstream exports are the faithful comparison",
        "protocol": "original H5 columns and PCPHD02/03; source count-based 6000-bin means; original normalization then 100-sample within-shot NBI smoothing; no mean fill or clipping",
        "caveat": "source NBI smoothing concatenated filtered phase rows across boundaries; serving reconstructs within-shot rows before native domain filtering, so agreement audit must be read with this panel",
        "horizons": review_summary(reconstructed) if reconstructed else {},
    }
    if not args.skip_own_target:
        values = risk(graph, source["final_x_normalized"][:cutoff])
        t, e = source["final_t"][:cutoff], source["final_e"][:cutoff] == 1
        own = {}
        for j, h in enumerate(dsm.HORIZONS_MS):
            cases = e & (t <= h)
            keep = cases | (t > h)
            own[f"h{int(h)}ms"] = areas_by_shot(
                cases[keep], values[keep, j], source_shots[:cutoff][keep]
            )
        record["own_target"] = {
            "role": "original checkpoint early-stopping validation; historical reversed source split",
            "horizons": own,
            "t_offset_ms": 1.0,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
