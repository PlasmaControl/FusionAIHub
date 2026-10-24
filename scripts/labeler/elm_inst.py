#!/usr/bin/env python
"""Train and score `elm-inst`: ELM onsets at 0.1 ms beside ELMy occupancy.

    python scripts/labeler/elm_inst.py prepare
    python scripts/labeler/elm_inst.py train --fold K --device cuda:0
    python scripts/labeler/elm_inst.py predict --device cuda:0
    python scripts/labeler/elm_inst.py evaluate

The model and its targets are `labeler.elm.inst`. Each of the five folds trains on
two sources at once and never sees its test shots of either:

* the 119 reviewed shots, split exactly as run `cv2` split them (its `fold.json`:
  `train`, `inner_val`, `test`), for occupancy, absent time and clock pseudo-onsets;
* Smith's 2,316 hand-labelled windows on 211 shots, split as the Smith onset run
  split them (run-day folds; its `fold.json`), for onsets.

Selection: each epoch scores the inner-validation shots of both sources; the kept
epoch maximises the trailing three-epoch mean of (occupancy AUPRC on the reviewed
50 ms bins + onset F1) / 2, where the onset F1 counts Smith's onsets found within
2 ms, unmatched detections in his windows, and every detection in guarded reviewed
absent time as false. The onset and occupancy thresholds are the inner-validation
F1-maximising ones. Nothing here reads the DIII-D challenge key, which stays held
out (`elm_challenge_evaluate.py` scores the served model on it).

`predict` writes every shot's out-of-fold traces and the five-model mean on the
challenge's 24 shots; `evaluate` writes the developmental record
`outputs/labeler/elm/inst/evaluation.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import (
    challenge,
    inputs,
    inst,
    labels,
    net,
    onset,
    prepare,
    score,
    train,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/inst/evaluation.json"
RUN = "inst1"
REVIEW_RUN = "round4/elm/cv/cv2"
SMITH_RUN = "round4/elm/smith/cv"
FOLDS = 5
SEED = 20261008
TOL_MS = 2.0
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
EPOCHS = 40
ITERS = 100
BATCH = 16  # half reviewed crops, half Smith crops
CROP_MS = 2048
REPLICATES = 1000
KEY_SHOTS_FROM = "edge_localized_mode/raw/elm_challenge_key_b71f991.json"


def work(paths: Paths) -> Path:
    return prepare.work_dir(paths) / "inst" / RUN


def smith_dir(paths: Paths) -> Path:
    return paths.root / "round4/elm/smith"


# ------------------------------------------------------------------ prepare


def run_prepare(paths: Paths, args) -> None:
    """Clock pseudo-onsets (its half-prominence span starts) on every reviewed shot."""
    table = labels.review_table(prepare.review_csv(paths))
    shots = sorted(int(s) for s in table.shot.unique())
    train.check_no_holdout(shots, paths)
    out, failed = {}, {}
    for shot in shots:
        try:
            spans, info = challenge.clock_spans(shot, paths)
            out[str(shot)] = {
                "spans": np.asarray(spans).round(4).tolist(),
                "filterscope": info["filterscope"],
                "tier": info["tier"],
            }
        except Exception as error:  # noqa: BLE001 - recorded, never hidden
            failed[str(shot)] = f"{type(error).__name__}: {str(error)[:160]}"
    work(paths).mkdir(parents=True, exist_ok=True)
    (work(paths) / "pseudo.json").write_text(
        json.dumps(
            {
                "git": git_sha(),
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "rule": "challenge.clock_spans: the ELM clock's peaks on the first "
                "finite filterscope, each the span of its half-prominence width",
                "shots": out,
                "failed": failed,
            }
        )
        + "\n"
    )
    print(f"pseudo-onsets on {len(out)} shots, {len(failed)} failed", flush=True)


# --------------------------------------------------------------------- data


class Shot:
    """One shot's input and targets: `x (C, cells)`, `state (n_ms,)` (review only),
    instance `heat/known/weight (cells,)`, and its evaluation fields."""

    def __init__(self, shot, x, state, target, source, **extra):
        self.shot, self.x, self.state, self.source = shot, x, state, source
        valid = x[inputs.VALID] > 0
        self.heat = target.heat
        self.known = target.known & valid
        self.weight = target.weight
        self.n_ms = x.shape[1] // inputs.CELLS_PER_MS
        self.__dict__.update(extra)


def load_review(paths: Paths, pseudo: dict) -> dict[int, Shot]:
    out = {}
    for shot, d in train.load(paths).items():
        starts = np.asarray(pseudo.get(str(shot), {}).get("spans", []), float)
        starts = starts.reshape(-1, 2)
        target = inst.review_target(d.spans, starts[:, 0], d.x.shape[1])
        out[shot] = Shot(
            shot,
            d.x,
            d.dense.state,
            target,
            "review",
            spans=d.spans,
            bins=d.bins,
            window=d.dense.window,
            pseudo=starts,
        )
    return out


def load_smith(paths: Paths) -> dict[int, Shot]:
    out = {}
    for path in sorted((smith_dir(paths) / "targets").glob("*.npz")):
        shot = int(path.stem)
        z = np.load(path)
        x = np.load(smith_dir(paths) / "inputs" / f"{shot}.npy")
        n_ms = x.shape[1] // inputs.CELLS_PER_MS
        target = inst.smith_target(
            z["truth"], z["window_t0"], z["window_t1"], x.shape[1]
        )
        out[shot] = Shot(
            shot,
            x,
            np.full(n_ms, labels.IGNORE, np.int8),
            target,
            "smith",
            truth=z["truth"].astype(float),
            window_t0=z["window_t0"].astype(float),
            window_t1=z["window_t1"].astype(float),
        )
    return out


def folds_of(paths: Paths) -> list[dict]:
    """Fold k's review and Smith `train`, `inner_val` and `test` shots."""
    out = []
    for k in range(FOLDS):
        r = json.loads((paths.root / REVIEW_RUN / f"fold{k}/fold.json").read_text())
        s = json.loads((paths.root / SMITH_RUN / f"fold{k}/fold.json").read_text())
        out.append(
            {
                part: {"review": sorted(r[part]), "smith": sorted(s[part])}
                for part in ("train", "inner_val", "test")
            }
        )
    return out


# ------------------------------------------------------------------ batches


def crop(rng, d: Shot, crop_ms: int, cfg) -> tuple[np.ndarray, ...]:
    """One augmented crop placed over the shot's labelled time."""
    if d.source == "review":
        lo, hi = (w - inputs.GRID0_MS for w in d.window)
        first = rng.uniform(lo - crop_ms / 2, max(lo, hi - crop_ms / 2))
    else:
        i = int(rng.integers(len(d.truth)))
        centre = d.truth[i] - inputs.GRID0_MS
        first = centre - rng.uniform(crop_ms / 4, 3 * crop_ms / 4)
    a = int(np.clip(first, 0, max(0, d.n_ms - crop_ms)))
    m = min(a + crop_ms, d.n_ms) - a
    c, k = inputs.CELLS_PER_MS, crop_ms * inputs.CELLS_PER_MS
    x = np.zeros((inputs.N_CHANNELS, k), np.float32)
    state = np.full(crop_ms, labels.IGNORE, np.int8)
    heat = np.zeros(k, np.float32)
    known = np.zeros(k, bool)
    weight = np.zeros(k, np.float32)
    x[:, : m * c] = d.x[:, a * c : (a + m) * c]
    state[:m] = d.state[a : a + m]
    heat[: m * c] = d.heat[a * c : (a + m) * c]
    known[: m * c] = d.known[a * c : (a + m) * c]
    weight[: m * c] = d.weight[a * c : (a + m) * c]
    train.augment(rng, x, cfg)
    return x, state, heat, known, weight


def batch_of(rng, data, review, smith, cfg, device):
    half = cfg.batch // 2
    picks = [data[int(s)] for s in rng.choice(review, half)]
    picks += [data[int(s)] for s in rng.choice(smith, cfg.batch - half)]
    items = [crop(rng, d, cfg.crop_ms, cfg) for d in picks]
    x, state, heat, known, weight = (np.stack(v) for v in zip(*items, strict=True))
    state = state.astype(np.int64)
    return tuple(
        torch.from_numpy(a).to(device) for a in (x, state, heat, known, weight)
    )


def loss_of(model, x, state, heat, known, weight):
    occ, ins = model(x)
    usable = train.ms_valid(x) & (state >= 0)
    ev = F.binary_cross_entropy_with_logits(occ, (state > 0).float(), reduction="none")
    ev = (ev * usable).sum() / usable.sum().clamp(min=1)
    on = inst.focal_loss(ins, heat, known, weight)
    return ev + on, float(ev.detach()), float(on.detach())


# --------------------------------------------------------------- validation


def guarded_absent(spans: pd.DataFrame) -> list[tuple[float, float]]:
    g = inst.ABSENT_GUARD_MS
    return [
        (r.t_start + g, r.t_end - g)
        for r in spans.itertuples()
        if r.kind == "absent" and r.t_end - r.t_start > 2 * g
    ]


def present_spans(spans: pd.DataFrame) -> list[tuple[float, float]]:
    return [
        (r.t_start, r.t_end)
        for r in spans.itertuples()
        if r.kind in ("crowd", "non_crowd")
    ]


def in_any(t: np.ndarray, intervals) -> np.ndarray:
    out = np.zeros(len(t), bool)
    for a, b in intervals:
        out |= (t >= a) & (t < b)
    return out


def smith_counts(found, d: Shot, tol=TOL_MS) -> tuple[int, int, int, list[float]]:
    """Onsets found within `tol` of Smith's, and unmatched detections in his windows."""
    windows = list(zip(d.window_t0, d.window_t1, strict=True))
    found = np.asarray(found, float)
    found = found[in_any(found, windows)]
    return inst.match(found, d.truth, tol)


def absent_alarms(found, d: Shot) -> int:
    return int(in_any(np.asarray(found, float), guarded_absent(d.spans)).sum())


def onset_f1(candidates, data, thr) -> tuple[float, dict]:
    """Onset F1 at `thr` from each shot's `(peak times, peak heights)` at the lowest
    threshold: greedy suppression keeps a peak only for a higher one, so the peaks at
    a higher threshold are exactly those of these that reach it."""
    tp = fp = fn = 0
    for s, (t, h) in candidates.items():
        d = data[s]
        found = t[h >= thr]
        if d.source == "smith":
            a, b, c, _ = smith_counts(found, d)
            tp, fp, fn = tp + a, fp + b, fn + c
        else:
            fp += absent_alarms(found, d)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return f1, {"tp": tp, "fp": fp, "fn": fn}


def validate(model, data, review_val, smith_val, device):
    traces = {s: inst.predict(model, data[s].x, device) for s in review_val + smith_val}
    truth = np.concatenate([data[s].bins.truth for s in review_val])
    sc = np.concatenate(
        [labels.bin_scores(traces[s][0], data[s].bins) for s in review_val]
    )
    ap = score.average_precision(truth, sc)
    occ_thr, occ_f1 = score.best_threshold(truth, sc)
    candidates = {}
    for s, (_, heat) in traces.items():
        cells = inst.peaks(heat, float(GRID[0]))
        candidates[s] = (inst.time_of(cells), heat[cells])
    best = (-1.0, float(GRID[-1]), {})
    for thr in GRID:
        f1, counts = onset_f1(candidates, data, float(thr))
        if f1 > best[0]:
            best = (f1, float(thr), counts)
    return {
        "val_auprc": ap,
        "val_auroc": score.roc_auc(truth, sc),
        "val_occ_f1": occ_f1,
        "occ_threshold": occ_thr,
        "val_onset_f1": best[0],
        "onset_threshold": best[1],
        "onset_counts": best[2],
    }


# -------------------------------------------------------------------- train


def run_train(paths: Paths, args) -> None:
    pseudo = json.loads((work(paths) / "pseudo.json").read_text())["shots"]
    data = {**load_review(paths, pseudo), **load_smith(paths)}
    fold = folds_of(paths)[args.fold]
    fit_r, fit_s = fold["train"]["review"], fold["train"]["smith"]
    val_r, val_s = fold["inner_val"]["review"], fold["inner_val"]["smith"]
    test = set(fold["test"]["review"]) | set(fold["test"]["smith"])
    if test & (set(fit_r) | set(fit_s) | set(val_r) | set(val_s)):
        raise ValueError("a test shot is in training or inner validation")
    cfg = train.Config(
        epochs=args.epochs,
        iters=args.iters,
        batch=BATCH,
        crop_ms=CROP_MS,
        seed=SEED + args.fold,
    )
    device = torch.device(args.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = inst.InstNet(dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * cfg.iters, pct_start=0.15
    )
    warmup = int(np.ceil(cfg.epochs * 0.15))
    history, best, best_state, best_row = [], -1.0, None, None
    for epoch in range(cfg.epochs):
        model.train()
        t0, parts = time.time(), []
        for _ in range(cfg.iters):
            loss, ev, on = loss_of(
                model, *batch_of(rng, data, fit_r, fit_s, cfg, device)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            parts.append((float(loss.detach()), ev, on))
        row = {
            "epoch": epoch,
            "loss": float(np.mean([p[0] for p in parts])),
            "occupancy_loss": float(np.mean([p[1] for p in parts])),
            "instance_loss": float(np.mean([p[2] for p in parts])),
            **validate(model, data, val_r, val_s, device),
            "seconds": time.time() - t0,
        }
        row["score"] = 0.5 * (row["val_auprc"] + row["val_onset_f1"])
        history.append(row)
        if epoch >= warmup + 2:
            row["selection"] = float(np.mean([h["score"] for h in history[-3:]]))
            if row["selection"] > best:
                best = row["selection"]
                best_row = row
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        print(
            json.dumps(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in row.items()
                }
            ),
            flush=True,
        )
    out = work(paths) / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "model.pt")
    durations = np.concatenate(
        [
            np.diff(data[s].pseudo, axis=1)[:, 0][
                in_any(data[s].pseudo[:, 0], present_spans(data[s].spans))
            ]
            for s in fit_r
        ]
    )
    (out / "fold.json").write_text(
        json.dumps(
            {
                "fold": args.fold,
                "git": git_sha(),
                "config": {
                    **cfg.__dict__,
                    "selection": "trailing3 of (AUPRC + onset F1) / 2",
                },
                "shots": fold,
                "best": best_row,
                "occ_threshold": best_row["occ_threshold"],
                "onset_threshold": best_row["onset_threshold"],
                "event_duration_ms": float(np.median(durations)),
                "event_duration_rule": "median clock half-prominence span inside the "
                "training shots' reviewed present spans",
                "history": history,
                "checkpoint_sha256": sha256_of(out / "model.pt"),
                "parameters": net.n_parameters(model),
            },
            indent=1,
        )
        + "\n"
    )


# ------------------------------------------------------------------ predict


def models_of(directory: Path, device):
    models, records = [], []
    for k in range(FOLDS):
        model = inst.InstNet().to(device)
        model.load_state_dict(
            torch.load(
                directory / f"fold{k}/model.pt", map_location=device, weights_only=True
            )
        )
        models.append(model)
        records.append(json.loads((directory / f"fold{k}/fold.json").read_text()))
    return models, records


def key_shots(paths: Paths) -> list[int]:
    from labeler.events.source_formatters import read_elm_challenge

    return sorted(
        c["shot"] for c in read_elm_challenge(paths.label_tables / KEY_SHOTS_FROM)
    )


def run_predict(paths: Paths, args) -> None:
    """Out-of-fold traces of every reviewed and Smith shot, the five-model mean on the
    challenge's shots, and the Smith onset ensemble on the reviewed shots."""
    device = torch.device(args.device)
    directory = work(paths)
    models, records = models_of(directory, device)
    for sub in ("pred", "key", "smith_head_review"):
        (directory / sub).mkdir(exist_ok=True)
    for k, rec in enumerate(records):
        for source, shots in rec["shots"]["test"].items():
            base = (
                prepare.inputs_dir(paths)
                if source == "review"
                else smith_dir(paths) / "inputs"
            )
            for s in shots:
                occ, ins = inst.predict(models[k], np.load(base / f"{s}.npy"), device)
                np.savez_compressed(
                    directory / "pred" / f"{s}.npz",
                    occ=occ,
                    inst=ins.astype(np.float16),
                )
    signals = prepare.signals_dir(paths)
    for s in key_shots(paths):
        x = inputs.read_channels(signals / f"{s}.npz")
        traces = [inst.predict(m, x, device) for m in models]
        np.savez_compressed(
            directory / "key" / f"{s}.npz",
            occ=np.mean([t[0] for t in traces], axis=0),
            inst=np.mean([t[1] for t in traces], axis=0).astype(np.float32),
            valid=x[inputs.VALID] > 0,
        )
    # the Smith onset head (trained on Smith only) on the reviewed shots, served as on
    # the key: five-model mean, mean fold threshold
    smith_models, thresholds = [], []
    for k in range(FOLDS):
        m = net.ElmUNet().to(device)
        m.load_state_dict(
            torch.load(
                paths.root / SMITH_RUN / f"fold{k}/model.pt",
                map_location=device,
                weights_only=True,
            )
        )
        smith_models.append(m)
        thresholds.append(
            json.loads((paths.root / SMITH_RUN / f"fold{k}/fold.json").read_text())[
                "threshold"
            ]
        )
    for s in sorted(int(p.stem) for p in prepare.inputs_dir(paths).glob("*.npy")):
        x = np.load(prepare.inputs_dir(paths) / f"{s}.npy")
        p = np.mean([train.predict(m, x, device)[1] for m in smith_models], axis=0)
        np.savez_compressed(
            directory / "smith_head_review" / f"{s}.npz",
            onset=p,
            threshold=np.mean(thresholds),
        )
    print("predicted", flush=True)


# ----------------------------------------------------------------- evaluate


def bootstrap(per_shot: dict, fn, rng) -> list[float]:
    shots = sorted(per_shot)
    vals = []
    for _ in range(REPLICATES):
        pick = rng.integers(0, len(shots), len(shots))
        vals.append(fn([per_shot[shots[i]] for i in pick]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def f1_of(rows) -> float:
    tp = sum(r[0] for r in rows)
    fp = sum(r[1] for r in rows)
    fn = sum(r[2] for r in rows)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def counts_summary(per_shot: dict, rng) -> dict:
    rows = list(per_shot.values())
    tp, fp, fn = (int(sum(r[i] for r in rows)) for i in range(3))
    offsets = [o for r in rows for o in r[3]]
    return {
        "shots": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "f1": f1_of(rows),
        "f1_ci95": bootstrap(per_shot, f1_of, rng),
        "median_abs_offset_ms": float(np.median(np.abs(offsets))) if offsets else None,
    }


def alarm_summary(per_shot: dict) -> dict:
    """Detections in guarded reviewed absent time: count, per 10 s, spans with any."""
    n = sum(r["n"] for r in per_shot.values())
    secs = sum(r["seconds"] for r in per_shot.values())
    spans = sum(r["spans"] for r in per_shot.values())
    hit = sum(r["spans_alarmed"] for r in per_shot.values())
    return {
        "shots": len(per_shot),
        "absent_seconds": secs,
        "detections": int(n),
        "per_10_s": 10 * n / max(secs, 1e-9),
        "absent_spans": int(spans),
        "spans_with_a_detection": int(hit),
        "fraction_of_spans": hit / max(spans, 1),
    }


def alarms_of(found, d: Shot) -> dict:
    guarded = guarded_absent(d.spans)
    found = np.asarray(found, float)
    return {
        "n": int(in_any(found, guarded).sum()),
        "seconds": sum(b - a for a, b in guarded) / 1000,
        "spans": len(guarded),
        "spans_alarmed": sum(
            bool(((found >= a) & (found < b)).any()) for a, b in guarded
        ),
    }


def run_evaluate(paths: Paths, args) -> None:
    directory = work(paths)
    pseudo = json.loads((directory / "pseudo.json").read_text())["shots"]
    review = load_review(paths, pseudo)
    smith = load_smith(paths)
    records = [
        json.loads((directory / f"fold{k}/fold.json").read_text()) for k in range(FOLDS)
    ]
    fold_of = {
        s: k
        for k, r in enumerate(records)
        for src in r["shots"]["test"].values()
        for s in src
    }
    cv2 = [
        json.loads((paths.root / REVIEW_RUN / f"fold{k}/fold.json").read_text())
        for k in range(FOLDS)
    ]
    smith_folds = [
        json.loads((paths.root / SMITH_RUN / f"fold{k}/fold.json").read_text())
        for k in range(FOLDS)
    ]
    rng = np.random.default_rng(SEED)

    def trace(s):
        z = np.load(directory / "pred" / f"{s}.npz")
        return z["occ"], z["inst"].astype(np.float32)

    # occupancy on the reviewed bins, beside cv2's out-of-fold event head
    occ = {"elm-inst": {}, "elm-ours (cv2)": {}}
    for s, d in review.items():
        k = fold_of[s]
        o, _ = trace(s)
        sc = labels.bin_scores(o, d.bins)
        occ["elm-inst"][s] = (d.bins.truth, sc, records[k]["occ_threshold"])
        p = np.load(paths.root / REVIEW_RUN / "pred" / f"{s}.npz")["p"][0]
        occ["elm-ours (cv2)"][s] = (
            d.bins.truth,
            labels.bin_scores(p, d.bins),
            cv2[k]["threshold"],
        )

    def occ_metrics(rows):
        t = np.concatenate([r[0] for r in rows])
        sc = np.concatenate([r[1] for r in rows])
        call = np.concatenate([r[1] >= r[2] for r in rows])
        tp, fp, fn = (
            int((call & (t == 1)).sum()),
            int((call & (t == 0)).sum()),
            int((~call & (t == 1)).sum()),
        )
        return {
            "auroc": score.roc_auc(t, sc),
            "auprc": score.average_precision(t, sc),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "bins": int(t.size),
        }

    occupancy = {}
    for name, per in occ.items():
        m = occ_metrics(list(per.values()))
        m["auroc_ci95"] = bootstrap(per, lambda rows: occ_metrics(rows)["auroc"], rng)
        m["f1_ci95"] = bootstrap(per, lambda rows: occ_metrics(rows)["f1"], rng)
        occupancy[name] = m

    # onsets: Smith's windows (out of fold) and alarms in reviewed absent time
    decodings = {"ungated": None, "gated": "occ"}
    smith_view, alarm_view = {}, {}
    for name, gate in decodings.items():
        sm, al = {}, {}
        for s, d in {**smith, **review}.items():
            k = fold_of[s]
            o, h = trace(s)
            g = records[k]["occ_threshold"] if gate else None
            found = inst.onsets(
                h, records[k]["onset_threshold"], o if gate else None, g
            )
            if d.source == "smith":
                sm[s] = smith_counts(found, d)
            else:
                al[s] = alarms_of(found, d)
        smith_view[f"elm-inst ({name})"] = counts_summary(sm, rng)
        alarm_view[f"elm-inst ({name})"] = alarm_summary(al)
    # the Smith onset head, out of fold on Smith, its five-model mean on the review
    sm = {}
    for s, d in smith.items():
        k = next(i for i, f in enumerate(smith_folds) if s in f["test"])
        p = np.load(paths.root / SMITH_RUN / "pred" / f"{s}.npz")["onset"]
        found = inputs.GRID0_MS + onset.peaks(p, smith_folds[k]["threshold"]) + 0.5
        sm[s] = smith_counts(found, d)
    smith_view["elm-ours-onset-smith"] = counts_summary(sm, rng)
    al, al_clock, al_cv2 = {}, {}, {}
    for s, d in review.items():
        z = np.load(directory / "smith_head_review" / f"{s}.npz")
        al[s] = alarms_of(
            inputs.GRID0_MS + onset.peaks(z["onset"], float(z["threshold"])) + 0.5, d
        )
        al_clock[s] = alarms_of(d.pseudo[:, 0], d)
        k = fold_of[s]
        p = np.load(paths.root / REVIEW_RUN / "pred" / f"{s}.npz")["p"][1]
        al_cv2[s] = alarms_of(
            inputs.GRID0_MS + onset.peaks(p, cv2[k]["onset_threshold"]) + 0.5, d
        )
    alarm_view["elm-ours-onset-smith"] = alarm_summary(al)
    alarm_view["elm-clock"] = alarm_summary(al_clock)
    alarm_view["elm-ours-onset-review (cv2)"] = alarm_summary(al_cv2)

    # detections inside reviewed present spans (no instance truth there; descriptive)
    present = {}
    for name, gate in decodings.items():
        n = 0
        for s, d in review.items():
            k = fold_of[s]
            o, h = trace(s)
            found = inst.onsets(
                h,
                records[k]["onset_threshold"],
                o if gate else None,
                records[k]["occ_threshold"] if gate else None,
            )
            n += int(in_any(found, present_spans(d.spans)).sum())
        present[f"elm-inst ({name})"] = n
    present["elm-clock"] = int(
        sum(
            in_any(d.pseudo[:, 0], present_spans(d.spans)).sum()
            for d in review.values()
        )
    )

    # the decoding served on the challenge key is fixed here, on development data only
    def dev_f1(name):
        c = smith_view[f"elm-inst ({name})"]
        a = alarm_view[f"elm-inst ({name})"]["detections"]
        return 2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"] + a, 1)

    primary = max(decodings, key=dev_f1)
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_sha(full=True),
        "run": str(directory),
        "scripts_sha256": {
            p.name: sha256_of(p)
            for p in (Path(__file__), REPO / "src/labeler/elm/inst.py")
        },
        "protocol": "Developmental out-of-fold scores: review shots split as cv2, Smith "
        "shots as the Smith onset run (run-day folds); each fold model never saw its "
        "test shots of either source. The DIII-D challenge key is not read here.",
        "fold_thresholds": {
            "onset": [r["onset_threshold"] for r in records],
            "occupancy": [r["occ_threshold"] for r in records],
            "event_duration_ms": [r["event_duration_ms"] for r in records],
            "selected_epochs": [r["best"]["epoch"] for r in records],
        },
        "occupancy_review_bins": occupancy,
        "onsets_smith_windows_2ms": smith_view,
        "alarms_in_guarded_review_absent_time": alarm_view,
        "detections_in_review_present_spans": present,
        "dev_onset_f1_with_absent_alarms": {n: dev_f1(n) for n in decodings},
        "primary_decoding": primary,
        "primary_rule": "the decoding with the higher development onset F1, counting "
        "Smith-window matches and reviewed absent-time detections as false; fixed "
        "before the challenge key is scored",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1, default=float) + "\n")
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "occupancy_review_bins",
                    "onsets_smith_windows_2ms",
                    "alarms_in_guarded_review_absent_time",
                    "detections_in_review_present_spans",
                    "dev_onset_f1_with_absent_alarms",
                    "primary_decoding",
                )
            },
            indent=1,
            default=float,
        )
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("prepare", "train", "predict", "evaluate"))
    ap.add_argument("--fold", type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--iters", type=int, default=ITERS)
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    {
        "prepare": run_prepare,
        "train": run_train,
        "predict": run_predict,
        "evaluate": run_evaluate,
    }[args.stage](paths, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
