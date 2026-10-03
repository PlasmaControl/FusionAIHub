#!/usr/bin/env python
"""Refit both DSM detection baselines with post-warm-up selection and three seeds.

    python scripts/labeler/elm_dsm_baseline_seeds.py [--variant reduced native]

The reported 60-input 1x128 detector kept the best single inner-validation AUPRC
epoch (zero-based epochs 35, 31, 2, 24, 0 of 40) and the native 124-input
[100, 1000] refit was one fixed 25-epoch run. Two folds of the first stopped before
the model had learned anything transferable, so this script repeats both with the
same data, folds, inner-validation shots, normalization and optimizer, changing only

* the checkpoint rule: the epoch ending the best three-epoch mean inner-validation
  AUPRC, a window that must lie wholly at or after epoch `ceil(0.15 * epochs)`
  (the warm-up fraction `elm-ours` uses), and the threshold at that epoch; and
* the seed: three new training seeds (fold `k` uses `seed + 100 k` for the
  reduced detector and `seed + k` for the native one, as the reported fits do).

Scoring is that of the reported rows: the reduced detector on the secondary
common bins (all119 and bes73), the native one on the 37-shot complete-input panel,
both beside `elm-ours`, shared bootstrap draws, paired differences. The results are
lower bounds on DSM detection skill under this recipe; they are not a search for the
best DSM. Output: `outputs/labeler/elm/dsm/baseline_seeds.json`; fitted scores and
checkpoints stay under `$LABELER_ROOT/round4/elm/dsm/baseline_seeds/`.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import elm_dsm_evaluate as dsm_eval
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import compare, dsm, methods, score, swap, train

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/dsm/baseline_seeds.json"
SEEDS = (20261104, 20261105, 20261106)
RULE = "post_warmup_trailing3"
WARMUP_FRACTION = 0.15
REDUCED_EPOCHS = 40
NATIVE_EPOCHS = 25
METRICS = ("auroc", "auprc", "f1")
RAW_ROWS = "repaired_raw_rows"
NATIVE_NAME = "elm-dsm-native-detect"


def warmup_epochs(epochs: int) -> int:
    return math.ceil(WARMUP_FRACTION * epochs)


def spread(values) -> dict:
    v = np.asarray(values, float)
    return {
        "values": [float(x) for x in v],
        "mean": float(v.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
        "sample_sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
    }


def fold_shots(oof, k):
    return json.loads((oof.dir / f"fold{k}" / "fold.json").read_text())


# ----------------------------------------------------------------- 60-input


def fit_reduced(rows, data, oof, seed, device):
    """The reduced detector over the five outer folds: held-out scores, records."""
    spans = {s: d.spans for s, d in data.items()}
    bins = {s: d.bins for s, d in data.items()}
    scores, thresholds, records = {}, {}, []
    for k in range(len(oof.record["folds"])):
        info = fold_shots(oof, k)
        cfg = dsm.FitConfig(
            epochs=REDUCED_EPOCHS,
            seed=seed + 100 * k,
            device=device,
            selection=RULE,
            warmup=warmup_epochs(REDUCED_EPOCHS),
        )
        norm = dsm.fit_detection_normalization(rows, spans, info["train"])
        normalized = {s: dsm.normalize_detection_rows(r, norm) for s, r in rows.items()}
        state, thr, history = dsm.fit_fold(
            rows, spans, bins, info["train"], info["inner_val"], cfg
        )
        model = dsm.Detector(dropout=cfg.dropout).to(device)
        model.load_state_dict(state)
        for s in info["test"]:
            scores[s] = dsm.predict(model, normalized[s].x)
            thresholds[s] = float(thr)
        chosen = max(
            (h for h in history if h["selection_score"] is not None),
            key=lambda h: h["selection_score"],
        )
        records.append(
            {
                "fold": k,
                "seed": cfg.seed,
                "selected_epoch": chosen["epoch"],
                "selection_score": chosen["selection_score"],
                "threshold": float(thr),
                "inner_val_auprc_at_epoch": chosen["val_auprc"],
                "val_auprc_by_epoch": [h["val_auprc"] for h in history],
                "train_shots": len(info["train"]),
                "inner_val_shots": len(info["inner_val"]),
                "test_shot_ids": info["test"],
            }
        )
        print(
            f"reduced seed {seed} fold {k}: epoch {chosen['epoch']}, "
            f"threshold {thr:.4f}",
            flush=True,
        )
    return scores, thresholds, records


def common_summary(sdef, data, oof, dscores, elmo, clock, sweep, name):
    """`elm-ours` and `name` on the common bins of one set, with paired differences."""
    parts, bins_of = compare.common_parts(sdef, data, oof, dscores, elmo, clock, sweep)
    keep = {key: parts[key] for key in (compare.NAME["ours"], name)}
    summary = methods.summarise_methods(
        keep, score.draws(len(sdef.shots)), compare.NAME["ours"]
    )
    return {
        "n_shots": len(sdef.shots),
        "bins": int(sum(len(b.t0) for b in bins_of.values())),
        "methods": {
            key: {
                "point": summary["methods"][key]["point"],
                "ci95": summary["methods"][key]["ci95"],
            }
            for key in keep
        },
        "paired": {
            m: summary["paired"][f"{compare.NAME['ours']} - {name}: {m}"]
            for m in METRICS
        },
    }


# ------------------------------------------------------------------- native


def native_inputs(paths, data, work, coverage):
    """Cached 1 ms input arrays and complete-input bins of the supported shots."""
    raw, bins = {}, {}
    for s, d in data.items():
        if coverage[str(s)]["complete_124_input_bins"] == 0:
            continue
        x = np.load(work / f"{s}_raw.npy")
        index = (d.bins.t0.astype(int) + 50)[:, None] + np.arange(50)
        keep = np.isfinite(x[index]).all(axis=(1, 2))
        if keep.any():
            raw[s], bins[s] = x, methods.restrict_bins(d.bins, keep)
    return raw, bins


@torch.no_grad()
def native_predict(model, x):
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


def fit_native(raw, bins, oof, seed, device, epochs):
    """The [100, 1000] detector over the folds; the recipe of `elm_native_detect`."""
    predictions, thresholds, records = {}, {}, []
    rule_warmup = warmup_epochs(epochs)
    indices = {s: (bins[s].t0.astype(int) + 50)[:, None] + np.arange(50) for s in raw}
    for k in range(len(oof.record["folds"])):
        original = fold_shots(oof, k)
        tr, va, te = (
            [s for s in original[key] if s in raw]
            for key in ("train", "inner_val", "test")
        )
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
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
        torch.manual_seed(seed + k)
        rng = np.random.default_rng(seed + k)
        model = nn.Sequential(
            nn.Linear(124, 100, bias=False),
            nn.ReLU6(),
            nn.Dropout(0.2),
            nn.Linear(100, 1000, bias=False),
            nn.ReLU6(),
            nn.Dropout(0.2),
            nn.Linear(1000, 1),
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        best, state, threshold, history = -1.0, None, 0.5, []
        for epoch in range(epochs):
            model.train()
            order = rng.permutation(len(x))
            losses = []
            for i in range(0, len(x), 2048):
                take = order[i : i + 2048]
                loss = F.binary_cross_entropy_with_logits(
                    model(torch.from_numpy(x[take]).to(device)).ravel(),
                    torch.from_numpy(y[take]).to(device),
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            truth = np.concatenate([bins[s].truth for s in va])
            values = np.concatenate(
                [native_predict(model, normalized[s])[indices[s]].mean(1) for s in va]
            )
            ap = score.average_precision(truth, values)
            thr, f1 = score.best_threshold(truth, values)
            history.append(
                {"epoch": epoch, "loss": float(np.mean(losses)), "val_auprc": ap}
            )
            criterion = dsm.selection_criterion(history, RULE, rule_warmup)
            history[-1]["selection_score"] = criterion
            if criterion is not None and criterion > best:
                best, threshold = criterion, thr
                state = {
                    n: v.detach().cpu().clone() for n, v in model.state_dict().items()
                }
                chosen = {"epoch": epoch, "val_auprc": ap, "val_f1": f1}
        if state is None:
            raise ValueError("no eligible native checkpoint")
        model.load_state_dict(state)
        for s in te:
            predictions[s] = native_predict(model, normalized[s])
            thresholds[s] = float(threshold)
        records.append(
            {
                "fold": k,
                "seed": seed + k,
                "selected_epoch": chosen["epoch"],
                "selection_score": best,
                "threshold": float(threshold),
                "inner_val_auprc_at_epoch": chosen["val_auprc"],
                "val_auprc_by_epoch": [h["val_auprc"] for h in history],
                "train_shots": len(tr),
                "inner_val_shots": len(va),
                "test_shot_ids": te,
            }
        )
        print(
            f"native seed {seed} fold {k}: epoch {chosen['epoch']}, "
            f"threshold {threshold:.4f}",
            flush=True,
        )
    return predictions, thresholds, records, indices, bins


def native_panels(
    sets,
    data,
    oof,
    raw,
    bins,
    predictions,
    thresholds,
    indices,
    reduced,
    reduced_thr,
    reduced_rows,
    elmo,
    clock,
    sweep,
):
    """The 37-shot panels: native, reduced, `elm-ours` and ELM-O on identical bins."""
    out = {}
    for tag, panel in sets.items():
        parts = {key: [] for key in (NATIVE_NAME, "elm-ours", "elm-dsm-detect")}
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
            b = methods.restrict_bins(b0, keep & good)
            if not len(b.t0):
                continue
            used[s] = b
            ix = (b.t0.astype(int) + 50)[:, None] + np.arange(50)
            v = predictions[s][ix].mean(1)
            parts[NATIVE_NAME].append(
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
            v = reduced[s][dsm.row_index(b)]
            parts["elm-dsm-detect"].append(
                score.ShotScore(s, b.truth, b.kind, v >= reduced_thr[s], v)
            )
            if panel.has_elmo:
                e = elmo.get(s, methods.span_frame([], []))
                p = methods.span_part(data[s].spans, s, b, panel.cover[s], e)
                p.score = swap.sweep_bin_scores(sweep[sweep.shot == s], b)
                parts["elm-elmo"].append(p)
        summary = methods.summarise_methods(parts, score.draws(len(used)), "elm-ours")
        out[tag] = {
            "n_shots": len(used),
            "bins": int(sum(len(b.t0) for b in used.values())),
            "shots": sorted(int(s) for s in used),
            "methods": {
                key: {
                    "point": summary["methods"][key]["point"],
                    "ci95": summary["methods"][key]["ci95"],
                }
                for key in parts
            },
            "paired": {
                f"{key}: {m}": summary["paired"][f"elm-ours - {key}: {m}"]
                for key in parts
                if key != "elm-ours"
                for m in METRICS
                if f"elm-ours - {key}: {m}" in summary["paired"]
            },
        }
    return out


# --------------------------------------------------------------- aggregation


def aggregate(seed_results, reported):
    """Mean and spread over seeds of each method's metrics and paired differences."""
    out = {}
    first = seed_results[0]
    for panel, body in first.items():
        out[panel] = {"n_shots": body["n_shots"], "bins": body["bins"], "methods": {}}
        for name in body["methods"]:
            out[panel]["methods"][name] = {
                m: spread([r[panel]["methods"][name]["point"][m] for r in seed_results])
                for m in METRICS
            }
        out[panel]["paired_elm_ours_minus"] = {}
        for key in body["paired"]:
            out[panel]["paired_elm_ours_minus"][key] = {
                "value": spread(
                    [r[panel]["paired"][key]["value"] for r in seed_results]
                ),
                "ci95_by_seed": [r[panel]["paired"][key]["ci95"] for r in seed_results],
            }
        out[panel]["reported_single_run"] = reported.get(panel)
    return out


def reduced_reported(paths):
    """The reported single-run reduced rows, read from the evaluation record."""
    record = json.loads((REPO / "outputs/labeler/elm/dsm/evaluation.json").read_text())
    return {
        tag: {
            "n_shots": record["sets"][tag]["n_shots"],
            "bins": record["sets"][tag]["bins"],
            "elm-dsm-detect": {
                m: record["sets"][tag]["methods"]["elm-dsm-detect"]["point"][m]
                for m in METRICS
            },
            "fold_epochs": [
                r["best_epoch"] for r in record["detectors"]["elm-dsm-detect"]["folds"]
            ],
            "fold_thresholds": [
                r["threshold"] for r in record["detectors"]["elm-dsm-detect"]["folds"]
            ],
        }
        for tag in ("all119", "bes73")
    }


def native_reported():
    record = json.loads(
        (REPO / "outputs/labeler/elm/dsm/native_detection.json").read_text()
    )
    return {
        tag: {
            "n_shots": record["sets"][tag]["n_shots"],
            "bins": record["sets"][tag]["bins"],
            NATIVE_NAME: {
                m: record["sets"][tag]["methods"][NATIVE_NAME]["point"][m]
                for m in METRICS
            },
            "fold_epochs": [r["best_epoch"] for r in record["folds"]],
            "fold_thresholds": [r["threshold"] for r in record["folds"]],
        }
        for tag in ("all119", "bes73")
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--variant",
        nargs="+",
        choices=("reduced", "native"),
        default=["reduced", "native"],
    )
    ap.add_argument("--seeds", type=int, nargs=3, default=SEEDS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if score.SEED in args.seeds:
        ap.error("new seeds must differ from the reported fits' seed")
    torch.set_num_threads(4)
    paths = Paths.from_env()
    work = paths.root / "round4/elm/dsm"
    store = work / "baseline_seeds"
    store.mkdir(parents=True, exist_ok=True)
    data = train.load(paths)
    oof = methods.Oof(paths.root / "round4/elm/cv/cv2")
    sets = compare.load_sets(paths, data)
    elmo, clock = compare.load_detected(paths)
    sweep = compare.load_elmo_sweep(paths)
    shots = sorted(data)
    norm = dsm.load_norm(paths)
    survival_rows = dsm_eval.load_rows(paths, shots, work, norm)
    repaired = {s: dsm.load_rows(s, work / RAW_ROWS / f"{s}.npz") for s in shots}
    if any(r is None for r in repaired.values()):
        raise ValueError("saved repaired detection rows are required")
    names = (compare.NAME["detect"], compare.NAME["exposed"], compare.NAME["init"])
    base = compare.DsmScores.load(
        work, survival_rows, variants=names, detection_rows=repaired
    )
    record = {
        "git": git_sha(full=True),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "script_sha256": sha256_of(Path(__file__)),
        "protocol": {
            "selection_rule": RULE,
            "rule": "epoch ending the best three-epoch mean inner-validation "
            "AUPRC; the window lies wholly at or after the warm-up epoch; the "
            "threshold is the inner-validation F1 maximiser at that epoch",
            "warmup_fraction": WARMUP_FRACTION,
            "reduced_epochs": REDUCED_EPOCHS,
            "reduced_warmup_epochs": warmup_epochs(REDUCED_EPOCHS),
            "native_epochs": NATIVE_EPOCHS,
            "native_warmup_epochs": warmup_epochs(NATIVE_EPOCHS),
            "seeds": list(args.seeds),
            "reported_fit_seed": score.SEED,
            "unchanged": "data, outer folds, inner-validation shots, training-only "
            "normalization, architecture, optimizer, bins and scoring",
            "interpretation": "lower bounds on DSM detection skill under this "
            "recipe; no hyper-parameter search",
        },
        "cohort_test_shots_used": 0,
        "folds": oof.record["folds"],
    }
    reduced_by_seed, reduced_scores = [], {}
    if "reduced" in args.variant or "native" in args.variant:
        for seed in args.seeds:
            sc, thr, folds = fit_reduced(repaired, data, oof, seed, args.device)
            reduced_scores[seed] = (sc, thr)
            np.savez_compressed(
                store / f"reduced_seed{seed}_scores.npz",
                **{f"s{s}": v.astype(np.float32) for s, v in sc.items()},
            )
            reduced_by_seed.append({"seed": seed, "folds": folds})
    if "reduced" in args.variant:
        per_seed = []
        for seed in args.seeds:
            sc, thr = reduced_scores[seed]
            dscores = replace(
                base,
                scores={**base.scores, compare.NAME["detect"]: sc},
                threshold={**base.threshold, compare.NAME["detect"]: thr},
            )
            per_seed.append(
                {
                    tag: common_summary(
                        sdef,
                        data,
                        oof,
                        dscores,
                        elmo,
                        clock,
                        sweep,
                        compare.NAME["detect"],
                    )
                    for tag, sdef in sets.items()
                }
            )
        record["reduced_60_input"] = {
            "folds_by_seed": reduced_by_seed,
            "per_seed": per_seed,
            "aggregate": aggregate(per_seed, reduced_reported(paths)),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1) + "\n")
    if "native" in args.variant:
        native_dir = work / "native_detection"
        coverage = json.loads(
            (REPO / "outputs/labeler/elm/dsm/native_detection.json").read_text()
        )["coverage"]
        raw, bins = native_inputs(paths, data, native_dir, coverage)
        per_seed, folds_by_seed = [], []
        for i, seed in enumerate(args.seeds):
            preds, thr, folds, indices, bins = fit_native(
                raw, bins, oof, seed, args.device, NATIVE_EPOCHS
            )
            folds_by_seed.append({"seed": seed, "folds": folds})
            sc, rthr = reduced_scores[seed]
            per_seed.append(
                native_panels(
                    sets,
                    data,
                    oof,
                    raw,
                    bins,
                    preds,
                    thr,
                    indices,
                    sc,
                    rthr,
                    repaired,
                    elmo,
                    clock,
                    sweep,
                )
            )
        record["native_124_input"] = {
            "supported_shots": sorted(int(s) for s in raw),
            "folds_by_seed": folds_by_seed,
            "per_seed": per_seed,
            "aggregate": aggregate(per_seed, native_reported()),
            "reduced_comparator": "the seed-matched reduced detector above",
        }
        args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
