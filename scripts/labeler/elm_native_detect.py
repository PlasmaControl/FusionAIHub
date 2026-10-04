#!/usr/bin/env python
"""Audit native 1 ms DSM inputs and refit [100,1000] for reviewed occupancy.

No fetching or source statistics/weights. Freeze this recipe before inspecting
outer predictions; it remains developmental CV on previously reviewed shots.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import elm_dsm_native as native
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import compare, dsm, methods, score, swap, train

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/dsm/native_detection.json"
NAME = "elm-dsm-native-detect"
OURS_NATIVE = "elm-ours-native-folds"


def mean_cells(times, values, starts):
    """Timestamp-aware 1 ms means; empty/nonfinite columns remain missing."""
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    lo = np.searchsorted(times, starts)
    hi = np.searchsorted(times, starts + 1)
    out = np.full((len(starts), values.shape[1]), np.nan)
    for j in range(values.shape[1]):
        good = np.isfinite(values[:, j])
        sums = np.r_[0.0, np.cumsum(np.where(good, values[:, j], 0.0))]
        counts = np.r_[0, np.cumsum(good)]
        n = counts[hi] - counts[lo]
        np.divide(sums[hi] - sums[lo], n, out=out[:, j], where=n > 0)
    return out


def read_raw(shot, paths, photos, n_ms):
    starts = np.arange(n_ms, dtype=float) - 50
    raw = np.full((n_ms, 124), np.nan)
    index = {n: i for i, n in enumerate(native.COLUMNS)}
    path = native.H5 / f"{shot}_slow.h5"
    if path.exists():
        with h5py.File(path) as h:
            for group, names in native.GROUP_COLUMNS.items():
                if group not in h:
                    continue
                g = h[group]
                value = np.asarray(g["block0_values"])
                if value.size <= 1:
                    continue
                if group == "mag_pcb_coil":
                    value = 1.69861e-5 * value[:, :1]
                elif group in ("p_inj", "t_inj"):
                    value = value.reshape(len(value), -1, 8).sum(axis=2)[:, :1]
                elif group == "ech":
                    value = value[:, 1:2]
                elif group == "co2_density_slow":
                    chords = [s.decode() for s in g["block0_items"][:]]
                    value = value[
                        :, [chords.index(c) for c in ("r0", "v1", "v2", "v3")]
                    ]
                elif len(names) == 1:
                    value = value[:, :1]
                for j in range(0, min(len(names), value.shape[1]), 8):
                    cols = names[j : j + 8]
                    raw[:, [index[n] for n in cols]] = mean_cells(
                        g["axis1"][:], value[:, j : j + len(cols)], starts
                    )
    for name in ("pcphd02", "pcphd03"):
        got = native.photo_record(photos, paths, shot, name)
        if got is not None:
            t, v, _ = got
            raw[:, index[name + "_downsampled"]] = mean_cells(t, v, starts)[:, 0]
    return raw.astype(np.float32)


@torch.no_grad()
def predict(model, x):
    model.eval()
    device = next(model.parameters()).device
    return np.concatenate(
        [
            torch.sigmoid(model(torch.from_numpy(x[i : i + 4096]).to(device)))
            .cpu()
            .numpy()
            .ravel()
            for i in range(0, len(x), 4096)
        ]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    previous = json.loads(OUT.read_text()) if args.rescore else None
    paths = Paths.from_env()
    torch.set_num_threads(4)
    data = train.load(paths)
    sets = compare.load_sets(paths, data)
    oof = methods.Oof(paths.root / "round4/elm/cv/cv2")
    # elm-ours refitted on this comparator's own train/inner-validation shots
    # (scripts/labeler/elm_native_ours.py); absent until that run exists
    own_dir = paths.root / "round4/elm/cv/native37"
    oof_own = methods.Oof(own_dir) if (own_dir / "run.json").exists() else None
    work = paths.root / "round4/elm/dsm/native_detection"
    work.mkdir(parents=True, exist_ok=True)
    with (native.SOURCE / "data/dalpha_wpqh.pkl").open("rb") as fh:
        photos = pickle.load(fh)
    available, raw, bins = {}, {}, {}
    for s, d in data.items():
        if (
            args.rescore
            and previous["coverage"][str(s)]["complete_124_input_bins"] == 0
        ):
            available[str(s)] = previous["coverage"][str(s)]
            continue
        x = (
            np.load(work / f"{s}_raw.npy")
            if args.rescore
            else read_raw(s, paths, photos, d.n_ms)
        )
        ix = (d.bins.t0.astype(int) + 50)[:, None] + np.arange(50)
        cells = x[ix]
        # A column exists at 1 ms only if finite throughout at least one scored bin.
        present = np.isfinite(cells).all(axis=1)
        best = int(present.sum(axis=1).max(initial=0))
        missing = [native.COLUMNS[j] for j in range(124) if not present[:, j].any()]
        keep = np.isfinite(cells).all(axis=(1, 2))
        available[str(s)] = {
            "max_measured_columns_per_bin": best,
            "columns_with_1ms_support": int(present.any(axis=0).sum()),
            "missing_columns": missing,
            "complete_124_input_bins": int(keep.sum()),
            "at_least_112_input_bins": int((present.sum(axis=1) >= 112).sum()),
        }
        if keep.any():
            raw[s] = x
            bins[s] = methods.restrict_bins(d.bins, keep)
            if not args.rescore:
                np.save(work / f"{s}_raw.npy", x)
        print(
            "audit", s, best, "columns;", int(keep.sum()), "complete bins", flush=True
        )
    gate = sum(r["max_measured_columns_per_bin"] >= 112 for r in available.values())
    record = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "script_sha256": sha256_of(Path(__file__)),
        "coverage": available,
        "shots_at_least_90_percent": gate,
        "input_count_distribution": dict(
            Counter(r["max_measured_columns_per_bin"] for r in available.values())
        ),
        "missing_column_shot_counts": dict(
            Counter(n for r in available.values() for n in r["missing_columns"])
        ),
        "gate": ">=112/124 inputs at 1 ms on >=30 reviewed shots",
        "support_rule": "Training and comparison use only complete 124-input "
        "50 ms primary interior bins; all 50 native 1 ms rows finite.",
        "cohort_test_shots_used": 0,
        "recipe": {
            "layers": [100, 1000],
            "activation": "ReLU6",
            "dropout": 0.2,
            "epochs": args.epochs,
            "batch": 2048,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "seed": score.SEED,
            "normalization": "optimizer-training 1 ms rows only; no source "
            "weights/stats; clip standardized inputs +/-10",
            "nbi": "100-row centered mean in normalized units within shot; "
            "source concatenated-phase smoothing not reused",
            "selection": "inner-val bin AUPRC; bin F1 threshold",
            "target": "reviewed occupancy at 1 ms; average 50 row probabilities "
            "for a scored 50 ms bin",
        },
        "folds": [],
        "sets": {},
    }
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    if gate < 30:
        print("Native gate not met", flush=True)
        return
    if len(raw) < 30:
        raise RuntimeError("Gate met but fewer than 30 complete-input shots")
    predictions, thresholds = {}, {}
    for fold in oof.record["fold_records"]:
        k = fold["fold"]
        original = json.loads((oof.dir / f"fold{k}/fold.json").read_text())
        tr, va, te = (
            [s for s in original[key] if s in raw]
            for key in ("train", "inner_val", "test")
        )
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
        indices = {
            s: (bins[s].t0.astype(int) + 50)[:, None] + np.arange(50) for s in raw
        }
        tx = np.concatenate([raw[s][indices[s].ravel()] for s in tr])
        mean, std = tx.mean(axis=0, dtype=np.float64), tx.std(axis=0, dtype=np.float64)
        std[std < 1e-12] = 1.0
        normalized = {}
        for s, x in raw.items():
            z = np.nan_to_num((x - mean) / std, nan=0.0)
            for j in (3, 4):
                z[:, j] = np.convolve(z[:, j], np.ones(100) / 100, "same")
            normalized[s] = np.clip(z, -10, 10).astype(np.float32)
        x = np.concatenate([normalized[s][indices[s].ravel()] for s in tr])
        y = np.concatenate([np.repeat(bins[s].truth, 50) for s in tr]).astype(
            np.float32
        )
        torch.manual_seed(score.SEED + k)
        rng = np.random.default_rng(score.SEED + k)
        model = nn.Sequential(
            nn.Linear(124, 100, bias=False),
            nn.ReLU6(),
            nn.Dropout(0.2),
            nn.Linear(100, 1000, bias=False),
            nn.ReLU6(),
            nn.Dropout(0.2),
            nn.Linear(1000, 1),
        )
        model.to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        best, state, threshold, history = -1.0, None, 0.5, []
        if args.rescore:
            saved = torch.load(
                work / f"fold{k}.pt", weights_only=False, map_location="cpu"
            )
            state = saved["state"]
            old_fold = next(r for r in previous["folds"] if r["fold"] == k)
            threshold, history = old_fold["threshold"], old_fold["history"]
        for epoch in range(0 if args.rescore else args.epochs):
            model.train()
            order = rng.permutation(len(x))
            losses = []
            for i in range(0, len(x), 2048):
                take = order[i : i + 2048]
                loss = F.binary_cross_entropy_with_logits(
                    model(torch.from_numpy(x[take]).to(args.device)).ravel(),
                    torch.from_numpy(y[take]).to(args.device),
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            truth = np.concatenate([bins[s].truth for s in va])
            values = np.concatenate(
                [predict(model, normalized[s])[indices[s]].mean(1) for s in va]
            )
            ap = score.average_precision(truth, values)
            thr, f1 = score.best_threshold(truth, values)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "val_auprc": ap,
                    "val_f1": f1,
                    "threshold": thr,
                }
            )
            if ap > best:
                best, threshold = ap, thr
                state = {
                    n: v.detach().cpu().clone() for n, v in model.state_dict().items()
                }
            print("fold", k, "epoch", epoch, "val AUPRC", round(ap, 4), flush=True)
        model.load_state_dict(state)
        if not args.rescore:
            torch.save({"state": state, "mean": mean, "std": std}, work / f"fold{k}.pt")
        for s in te:
            predictions[s] = predict(model, normalized[s])
            thresholds[s] = threshold
            np.savez_compressed(work / f"{s}_pred.npz", p=predictions[s])
        record["folds"].append(
            {
                "fold": k,
                "train": tr,
                "inner_val": va,
                "test": te,
                "threshold": threshold,
                "best_epoch": int(np.argmax([r["val_auprc"] for r in history])),
                "history": history,
            }
        )
        OUT.write_text(json.dumps(record, indent=1) + "\n")
    dsmwork = paths.root / "round4/elm/dsm"
    with np.load(dsmwork / "scores_elm-dsm-detect.npz") as z:
        reduced = {int(n[1:]): z[n] for n in z}
    reduced_thr = json.loads((dsmwork / "thresholds.json").read_text())[
        "elm-dsm-detect"
    ]
    reduced_rows = {
        s: dsm.load_rows(s, dsmwork / f"repaired_raw_rows/{s}.npz") for s in raw
    }
    elmo, clock = compare.load_detected(paths)
    sweep = compare.load_elmo_sweep(paths)
    for tag, panel in sets.items():
        parts = {NAME: [], "elm-ours": [], "elm-dsm-detect": [], "elm-clock": []}
        if oof_own is not None:
            parts[OURS_NATIVE] = []
        if panel.has_elmo:
            parts["elm-elmo"] = []
        used = {}
        for s in panel.shots:
            if s not in predictions:
                continue
            b0 = panel.bins[s]
            keep = np.isin(b0.t0, bins[s].t0)
            row_ix = dsm.row_index(b0)
            in_range = (row_ix >= 0) & (row_ix < len(reduced_rows[s].usable))
            good = np.zeros(len(row_ix), bool)
            good[in_range] = reduced_rows[s].usable[row_ix[in_range]]
            keep &= good
            b = methods.restrict_bins(b0, keep)
            if not len(b.t0):
                continue
            used[str(s)] = b.t0.tolist()
            ix = (b.t0.astype(int) + 50)[:, None] + np.arange(50)
            v = predictions[s][ix].mean(1)
            parts[NAME].append(
                score.ShotScore(s, b.truth, b.kind, v >= thresholds[s], v)
            )
            parts["elm-ours"].append(
                methods.trace_part(
                    data[s].spans,
                    s,
                    b,
                    panel.cover[s],
                    oof.trace(s)[0],
                    oof.threshold[s],
                )
            )
            if oof_own is not None:
                parts[OURS_NATIVE].append(
                    methods.trace_part(
                        data[s].spans,
                        s,
                        b,
                        panel.cover[s],
                        oof_own.trace(s)[0],
                        oof_own.threshold[s],
                    )
                )
            v = reduced[s][dsm.row_index(b)]
            parts["elm-dsm-detect"].append(
                score.ShotScore(s, b.truth, b.kind, v >= reduced_thr[str(s)], v)
            )
            parts["elm-clock"].append(
                methods.span_part(
                    data[s].spans,
                    s,
                    b,
                    panel.cover[s],
                    clock.get(s, methods.span_frame([], [])),
                )
            )
            if panel.has_elmo:
                p = methods.span_part(
                    data[s].spans,
                    s,
                    b,
                    panel.cover[s],
                    elmo.get(s, methods.span_frame([], [])),
                )
                p.score = swap.sweep_bin_scores(sweep[sweep.shot == s], b)
                parts["elm-elmo"].append(p)
        summary = methods.summarise_methods(parts, score.draws(len(used)), "elm-ours")
        summary.update(
            {
                "shots": list(map(int, used)),
                "n_shots": len(used),
                "bins": sum(len(v) for v in used.values()),
                "bin_starts": used,
            }
        )
        record["sets"][tag] = summary
    if oof_own is not None:
        record["ours_on_native_folds"] = {
            "method": OURS_NATIVE,
            "script": "scripts/labeler/elm_native_ours.py",
            "run": str(own_dir),
            "run_json_sha256": sha256_of(own_dir / "run.json"),
            "git": oof_own.record["git"],
            "config": oof_own.record["config"],
            "thresholds": [f["threshold"] for f in oof_own.record["fold_records"]],
            "rule": "same network, recipe and per-fold seeds as the headline "
            "elm-ours; trained on this comparator's train shots, inner-validation "
            "shots select the checkpoint and threshold",
        }
    record["checkpoints"] = {p.name: sha256_of(p) for p in work.glob("fold*.pt")}
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    print({k: (r["n_shots"], r["bins"]) for k, r in record["sets"].items()}, flush=True)


if __name__ == "__main__":
    main()
