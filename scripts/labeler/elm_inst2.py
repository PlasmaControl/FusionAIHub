#!/usr/bin/env python
"""Train and score `elm-inst2`: `elm-inst` with reviewed ELM trains and a spike loss.

    python scripts/labeler/elm_inst2.py prepare                  # inputs, onsets, folds
    python scripts/labeler/elm_inst2.py train --fold K --device cuda:0
    python scripts/labeler/elm_inst2.py predict --device cuda:0
    python scripts/labeler/elm_inst2.py evaluate                 # development record
    python scripts/labeler/elm_inst2.py key                      # the challenge key, once

The network is `inst.InstNet`, unchanged; the new targets and the spike loss are
`labeler.elm.spikes`. Each of the five folds trains on three sources and never sees
its test shots of any:

* the 119 reviewed shots and Smith's 2,316 windows, split and targeted exactly as for
  `elm-inst` (`elm_inst.py`: occupancy, guarded absent time and the clock's
  pseudo-onsets at half weight; Smith's onsets);
* Ricardo Shousha's review of 2026-10-23 (pin `2026-10-23`) on his shots that are on
  no hold-out list and have an input: occupancy from his spans and rows (absent rows
  under `SHORT_ABSENT_MS`, the gaps between his 10 ms rows, are ignored there), his
  guarded absent time, and the `elm_jump` v2 onsets inside his rows, merged
  (`spikes.accepted_onsets`), at full weight. His shots are dealt into five folds by
  run day; fold k tests on group k and validates on group k + 1.

The spike loss is asked at Smith's onsets and at his. Epoch selection is `elm-inst`'s:
the trailing three-epoch mean of (occupancy AUPRC + onset F1) / 2 on inner
validation, where the onset F1 also counts his inner-validation onsets
(`spikes.judge`). Each fold records its inner-validation maximum of `spikes.merit_gain`
(the challenge merit function's published costs, every ELM found priced large) and its
F1 maximum. `evaluate` fixes the two decodings the key is scored with before the key is
read: `elm-inst2` at the grid threshold where the pooled out-of-fold counts save the
most by the conservative rule (`rule_gains`), `elm-inst2-alt` at the mean fold
threshold. Every path that trains refuses the challenge key's shots
(`train.check_no_holdout`); `key` first rebuilds `elm-inst`'s graded files through the
same decoding, then scores both once, and a lock under the data root refuses a second
run from any checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import (
    challenge,
    heldout,
    inputs,
    inst,
    labels,
    net,
    prepare,
    score,
    spikes,
    train,
)
from labeler.events import shotlists
from labeler.events.source_formatters import read_elm_challenge
from labeler.scoring import events as boundary

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import elm_inst as base  # the sys.path insert above makes this work

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/inst2/evaluation.json"
KEY_OUT = REPO / "outputs/labeler/elm/inst2/key.json"
RUN = "inst2"
METHOD = "elm-inst2"
ALT = f"{METHOD}-alt"
COMMENT = (
    "FusionAIHub elm-inst2: peaks of the five fold models' mean 0.1 ms onset heatmap "
    "(trained with reviewed ELM trains and a spike loss), 3 ms apart, each event the "
    "training shots' median clock event duration long"
)
PIN = "2026-10-23"
#: The 120 shots reviewed before him (the 119 scored and 189093, which has no input).
BEFORE_PIN = "2026-10-17"
JUMP = "suggestions/elm_jump/v2/edge_localized_mode_suggest_elm_jump_v2_onsets.csv"
SHORT_ABSENT_MS = 50.0
BATCH = 24  # a third reviewed crops, a third Smith's, a third his
#: His crops are the batch's last third. The focal loss is normalised by the onsets it
#: holds, and his crops hold hundreds each, so it is taken over his crops and over the
#: rest apart: the reviewed absent time keeps the weight it had for `elm-inst`.
HIS_FROM = 2 * (BATCH // 3)
KEY = "edge_localized_mode/raw/elm_challenge_key_b71f991.json"
CLONE = "benchmarks/elm/challenge/elm_detection_challenge"
CLONE_COMMIT = "b71f991"
SUBMISSIONS = "benchmarks/elm/challenge/submissions"
#: The merit scores `elm_challenge_evaluate.py` recorded for `elm-inst` and the others.
STORED_MERIT = "benchmarks/elm/challenge/merit_scores.json"
LOCK = "key.lock"  # in the run folder: written before any elm-inst2 file is graded
SCORES = "key_merit_scores.json"
TRADEOFF_COUNTS = (
    "smith_tp",
    "smith_fp",
    "smith_fn",
    "review_alarms",
    "his_tp_outside_crowds",
    "his_tp_in_crowds",
    "his_fn",
    "his_fp",
    "his_empty",
    "his_other",
)
MERIT_KEYS = (
    "total",
    "timing_mismatch",
    "false_positive",
    "overcounting",
    "false_negative",
    "incorrect_classification",
    "real_events",
    "detected_events",
)
BUSY_EVENTS = 10
TOLERANCES_MS = (1.0, 2.0, 5.0)
GAP_EDGES_MS = (0.0, 10.0, 20.0, 50.0, 1e12)
NEAR_MS = 20.0


def work(paths: Paths) -> Path:
    return prepare.work_dir(paths) / "inst" / RUN


def his_dir(paths: Paths) -> Path:
    return work(paths) / "shousha"


# ------------------------------------------------------------------ prepare


def his_shots(paths: Paths) -> tuple[list[int], dict[int, str]]:
    """His reviewed shots that may train a model, and why each other one may not."""
    pin = prepare.pins_dir(paths) / PIN
    his = set(heldout.last_saves(pin / "history.jsonl"))
    before = set(labels.review_table(prepare.review_csv(paths, BEFORE_PIN)).shot)
    shots, left = [], {}
    for s in sorted(his - {int(x) for x in before}):
        if shotlists.held_out(paths, s):
            left[s] = "held out (the challenge key)"
        else:
            shots.append(s)
    return shots, left


def run_prepare(paths: Paths, args) -> None:
    """His shots' inputs (into this run's folder, never `round4/elm/inputs`), their
    merged onsets, and the run-day folds."""
    pin_sums = heldout.verify_pin(prepare.pins_dir(paths) / PIN)
    shots, left = his_shots(paths)
    train.check_no_holdout(shots, paths)
    train.check_no_test(shots, pd.read_csv(paths.catalog / "cohort.csv"))
    table = labels.review_table(prepare.review_csv(paths, PIN))
    jump = pd.read_csv(paths.root / JUMP)
    out = his_dir(paths) / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    onsets, counts, ready = {}, {}, []
    for s in shots:
        source = prepare.signals_dir(paths) / f"{s}.npz"
        if not source.is_file():
            left[s] = "no fetched input"
            continue
        try:
            x = inputs.read_channels(source)
        except Exception as error:  # noqa: BLE001 - recorded, never hidden
            left[s] = f"{type(error).__name__}: {str(error)[:160]}"
            continue
        np.save(out / f"{s}.npy", x)
        kept, n = spikes.accepted_onsets(
            table[table.shot == s], jump[jump.shot == s].t_ms
        )
        onsets[str(s)] = kept.round(3).tolist()
        counts[str(s)] = {**n, "valid_fraction": float((x[inputs.VALID] > 0).mean())}
        ready.append(s)
    days = heldout.run_days(paths, ready)
    groups = {s: days.get(s, f"shot {s}") for s in ready}
    folds = train.deal_group_folds(
        dict.fromkeys(ready), groups, k=base.FOLDS, seed=base.SEED
    )
    total = {
        k: int(sum(c[k] for c in counts.values()))
        for k in ("rows", "rows_without_onset", "onsets_in_rows", "kept")
    }
    record = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "pin": PIN,
        "pin_sha256": pin_sums,
        "jump_table": JUMP,
        "jump_table_sha256": sha256_of(paths.root / JUMP),
        "merge_ms": spikes.MERGE_MS,
        "shots": ready,
        "left_out": {str(s): why for s, why in sorted(left.items())},
        "run_days": {str(s): groups[s] for s in ready},
        "folds": folds,
        "totals": total,
        "onsets": onsets,
        "counts": counts,
    }
    (his_dir(paths) / "prepared.json").write_text(json.dumps(record, indent=1) + "\n")
    print(
        f"{len(ready)} shots ready, {len(left)} left out; {total}; fold sizes "
        f"{[len(f) for f in folds]}",
        flush=True,
    )


# --------------------------------------------------------------------- data


def prepared(paths: Paths) -> dict:
    return json.loads((his_dir(paths) / "prepared.json").read_text())


def occupancy_spans(spans: pd.DataFrame) -> pd.DataFrame:
    """His spans for the occupancy target: absent rows shorter than `SHORT_ABSENT_MS`
    (the gaps between his 10 ms rows) dropped, so that time is ignored."""
    short = (spans.kind == "absent") & (spans.t_end - spans.t_start < SHORT_ABSENT_MS)
    return spans[~short]


def load_his(paths: Paths) -> dict[int, base.Shot]:
    rec = prepared(paths)
    train.check_no_holdout(rec["shots"], paths)
    table = labels.review_table(prepare.review_csv(paths, PIN))
    out = {}
    for s in rec["shots"]:
        x = np.load(his_dir(paths) / "inputs" / f"{s}.npy")
        n_ms = x.shape[1] // inputs.CELLS_PER_MS
        spans = table[table.shot == s].reset_index(drop=True)
        onsets = np.asarray(rec["onsets"][str(s)], dtype=float)
        dense = labels.dense(occupancy_spans(spans), n_ms)
        d = base.Shot(
            s,
            x,
            dense.state,
            spikes.review_onset_target(spans, onsets, x.shape[1]),
            "shousha",
            spans=spans,
            onsets=onsets,
            window=dense.window,
        )
        d.anchor = spikes.anchors(onsets, x.shape[1]) & (x[inputs.VALID] > 0)
        out[s] = d
    return out


def load_all(paths: Paths) -> dict[int, base.Shot]:
    """The reviewed and Smith shots as `elm-inst` reads them, and his; every shot gets
    its spike-loss anchors (none on the reviewed shots, whose ELMs are not marked)."""
    pseudo = json.loads((base.work(paths) / "pseudo.json").read_text())["shots"]
    review = base.load_review(paths, pseudo)
    smith = base.load_smith(paths)
    for d in review.values():
        d.anchor = np.zeros(d.x.shape[1], dtype=bool)
    for d in smith.values():
        d.anchor = spikes.anchors(d.truth, d.x.shape[1]) & (d.x[inputs.VALID] > 0)
    his = load_his(paths)
    if (set(review) | set(smith)) & set(his):
        raise ValueError("his shots overlap the reviewed or Smith shots")
    return {**review, **smith, **his}


def folds_of(paths: Paths) -> list[dict]:
    """`elm-inst`'s folds of the reviewed and Smith shots, with his: fold k tests on his
    run-day group k and validates on group k + 1."""
    groups = prepared(paths)["folds"]
    out = []
    for k, fold in enumerate(base.folds_of(paths)):
        test, val = sorted(groups[k]), sorted(groups[(k + 1) % base.FOLDS])
        fit = sorted(s for g in groups for s in g if s not in set(test) | set(val))
        fold = {part: dict(v) for part, v in fold.items()}
        fold["train"]["shousha"] = fit
        fold["inner_val"]["shousha"] = val
        fold["test"]["shousha"] = test
        out.append(fold)
    return out


# ------------------------------------------------------------------ batches


def crop(rng, d: base.Shot, crop_ms: int, cfg) -> tuple[np.ndarray, ...]:
    """`elm_inst.crop` with the spike anchors; his crops fall over his window."""
    if d.source in ("review", "shousha"):
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
    anchor = np.zeros(k, bool)
    x[:, : m * c] = d.x[:, a * c : (a + m) * c]
    state[:m] = d.state[a : a + m]
    heat[: m * c] = d.heat[a * c : (a + m) * c]
    known[: m * c] = d.known[a * c : (a + m) * c]
    weight[: m * c] = d.weight[a * c : (a + m) * c]
    anchor[: m * c] = d.anchor[a * c : (a + m) * c]
    train.augment(rng, x, cfg)
    return x, state, heat, known, weight, anchor


def batch_of(rng, data, fit: dict[str, list[int]], cfg, device):
    if cfg.batch != BATCH:
        raise ValueError(f"batch {cfg.batch}: loss_of splits a batch of {BATCH}")
    third = HIS_FROM // 2
    picks = [data[int(s)] for s in rng.choice(fit["review"], third)]
    picks += [data[int(s)] for s in rng.choice(fit["smith"], third)]
    picks += [data[int(s)] for s in rng.choice(fit["shousha"], cfg.batch - HIS_FROM)]
    items = [crop(rng, d, cfg.crop_ms, cfg) for d in picks]
    x, state, heat, known, weight, anchor = (
        np.stack(v) for v in zip(*items, strict=True)
    )
    state = state.astype(np.int64)
    return tuple(
        torch.from_numpy(a).to(device) for a in (x, state, heat, known, weight, anchor)
    )


def loss_of(model, x, state, heat, known, weight, anchor):
    occ, ins = model(x)
    usable = train.ms_valid(x) & (state >= 0)
    ev = F.binary_cross_entropy_with_logits(occ, (state > 0).float(), reduction="none")
    ev = (ev * usable).sum() / usable.sum().clamp(min=1)
    h = HIS_FROM
    on = inst.focal_loss(ins[:h], heat[:h], known[:h], weight[:h]) + inst.focal_loss(
        ins[h:], heat[h:], known[h:], weight[h:]
    )
    sp = spikes.spike_loss(ins, anchor)
    loss = ev + on + spikes.SPIKE_WEIGHT * sp
    return loss, float(ev.detach()), float(on.detach()), float(sp.detach())


# --------------------------------------------------------------- validation


def candidates_of(heat: np.ndarray, min_sep_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Peak times and heights at the grid's lowest threshold: a peak at a higher
    threshold is exactly one of these that reaches it (`elm_inst.onset_f1`)."""
    cells = inst.peaks(heat, float(base.GRID[0]), min_sep_ms)
    return inst.time_of(cells), heat[cells]


def counts_at(d: base.Shot, found) -> tuple[int, int, int]:
    """`(tp, fp, fn)` of one shot's detections, by its source's truth."""
    if d.source == "smith":
        tp, fp, fn, _ = base.smith_counts(found, d)
        return tp, fp, fn
    if d.source == "review":
        return 0, base.absent_alarms(found, d), 0
    j = spikes.judge(found, d.onsets, d.spans)
    return j["tp"], j["fp"], j["fn"]


def threshold_rows(candidates, data, duration_ms: float) -> list[dict]:
    rows = []
    for thr in base.GRID:
        tp = fp = fn = 0
        for s, (t, h) in candidates.items():
            a, b, c = counts_at(data[s], t[h >= thr])
            tp, fp, fn = tp + a, fp + b, fn + c
        rows.append(
            {
                "threshold": float(thr),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "merit_gain": spikes.merit_gain(tp, fp, duration_ms),
            }
        )
    return rows


def validate(model, data, val: dict[str, list[int]], device, duration_ms: float):
    shots = val["review"] + val["smith"] + val["shousha"]
    traces = {s: inst.predict(model, data[s].x, device) for s in shots}
    truth = np.concatenate([data[s].bins.truth for s in val["review"]])
    sc = np.concatenate(
        [labels.bin_scores(traces[s][0], data[s].bins) for s in val["review"]]
    )
    occ_thr, occ_f1 = score.best_threshold(truth, sc)
    candidates = {s: candidates_of(traces[s][1], spikes.MIN_SEP_MS) for s in shots}
    rows = threshold_rows(candidates, data, duration_ms)
    by_f1 = max(rows, key=lambda r: r["f1"])
    by_merit = max(rows, key=lambda r: r["merit_gain"])
    return {
        "val_auprc": score.average_precision(truth, sc),
        "val_auroc": score.roc_auc(truth, sc),
        "val_occ_f1": occ_f1,
        "occ_threshold": occ_thr,
        "val_onset_f1": by_f1["f1"],
        "f1_threshold": by_f1["threshold"],
        "merit_threshold": by_merit["threshold"],
        "val_merit_gain": by_merit["merit_gain"],
        "onset_counts": by_merit,
    }


def event_duration(data, shots) -> float:
    """`elm-inst`'s event length: the median clock half-prominence span inside the
    training shots' reviewed present spans."""
    return float(
        np.median(
            np.concatenate(
                [
                    np.diff(data[s].pseudo, axis=1)[:, 0][
                        base.in_any(
                            data[s].pseudo[:, 0], base.present_spans(data[s].spans)
                        )
                    ]
                    for s in shots
                ]
            )
        )
    )


# -------------------------------------------------------------------- train


def run_train(paths: Paths, args) -> None:
    data = load_all(paths)
    fold = folds_of(paths)[args.fold]
    fit, val, test = fold["train"], fold["inner_val"], fold["test"]
    seen = {s for part in (fit, val) for v in part.values() for s in v}
    if seen & {s for v in test.values() for s in v}:
        raise ValueError("a test shot is in training or inner validation")
    train.check_no_holdout(sorted(seen), paths)
    cfg = train.Config(
        epochs=args.epochs,
        iters=args.iters,
        batch=BATCH,
        crop_ms=base.CROP_MS,
        seed=base.SEED + args.fold,
    )
    duration = event_duration(data, fit["review"])
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
            loss, ev, on, sp = loss_of(model, *batch_of(rng, data, fit, cfg, device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            parts.append((float(loss.detach()), ev, on, sp))
        t1 = time.time()
        row = {
            "epoch": epoch,
            "loss": float(np.mean([p[0] for p in parts])),
            "occupancy_loss": float(np.mean([p[1] for p in parts])),
            "instance_loss": float(np.mean([p[2] for p in parts])),
            "spike_loss": float(np.mean([p[3] for p in parts])),
            **validate(model, data, val, device, duration),
            "train_seconds": t1 - t0,
            "seconds": time.time() - t0,
        }
        row["score"] = 0.5 * (row["val_auprc"] + row["val_onset_f1"])
        history.append(row)
        if epoch >= min(warmup + 2, cfg.epochs - 1):
            row["selection"] = float(np.mean([h["score"] for h in history[-3:]]))
            if row["selection"] > best:
                best, best_row = row["selection"], row
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
    (out / "fold.json").write_text(
        json.dumps(
            {
                "fold": args.fold,
                "git": git_sha(),
                "config": {
                    **cfg.__dict__,
                    "selection": "trailing3 of (AUPRC + onset F1) / 2",
                    "threshold": "inner-validation maximum of spikes.merit_gain",
                    "min_sep_ms": spikes.MIN_SEP_MS,
                    "spike_weight": spikes.SPIKE_WEIGHT,
                    "merge_ms": spikes.MERGE_MS,
                },
                "shots": fold,
                "best": best_row,
                "occ_threshold": best_row["occ_threshold"],
                "onset_threshold": best_row["merit_threshold"],
                "f1_threshold": best_row["f1_threshold"],
                "event_duration_ms": duration,
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


def run_predict(paths: Paths, args) -> None:
    """Out-of-fold traces of every shot of the three sources, the five-model mean on
    the challenge's shots, and `elm-inst`'s five-model mean on his shots (it never saw
    them), for the development comparison."""
    device = torch.device(args.device)
    directory = work(paths)
    models, records = base.models_of(directory, device)
    for sub in ("pred", "key", "inst1_his"):
        (directory / sub).mkdir(exist_ok=True)
    where = {
        "review": prepare.inputs_dir(paths),
        "smith": base.smith_dir(paths) / "inputs",
        "shousha": his_dir(paths) / "inputs",
    }
    for k, rec in enumerate(records):
        for source, shots in rec["shots"]["test"].items():
            for s in shots:
                occ, ins = inst.predict(
                    models[k], np.load(where[source] / f"{s}.npy"), device
                )
                np.savez_compressed(
                    directory / "pred" / f"{s}.npz",
                    occ=occ,
                    inst=ins.astype(np.float16),
                )
    signals = prepare.signals_dir(paths)
    for s in base.key_shots(paths):
        x = inputs.read_channels(signals / f"{s}.npz")
        traces = [inst.predict(m, x, device) for m in models]
        np.savez_compressed(
            directory / "key" / f"{s}.npz",
            occ=np.mean([t[0] for t in traces], axis=0),
            inst=np.mean([t[1] for t in traces], axis=0).astype(np.float32),
            valid=x[inputs.VALID] > 0,
        )
    old, _ = base.models_of(base.work(paths), device)
    for s in prepared(paths)["shots"]:
        x = np.load(where["shousha"] / f"{s}.npy")
        traces = [inst.predict(m, x, device) for m in old]
        np.savez_compressed(
            directory / "inst1_his" / f"{s}.npz",
            occ=np.mean([t[0] for t in traces], axis=0),
            inst=np.mean([t[1] for t in traces], axis=0).astype(np.float32),
        )
    print("predicted", flush=True)


# ----------------------------------------------------------------- evaluate


def his_summary(per_shot: dict, gaps: dict, guarded_s: float, rng) -> dict:
    """Pooled `spikes.judge` counts on his shots, recall by the distance to the nearest
    other kept onset, and shot-bootstrap intervals for recall and F1."""
    rows = list(per_shot.values())
    tot = {
        k: int(sum(r[k] for r in rows))
        for k in ("tp", "fn", "fp", "fp_absent", "fp_near", "empty", "other")
    }

    def recall(rs):
        tp = sum(r["tp"] for r in rs)
        return tp / max(tp + sum(r["fn"] for r in rs), 1)

    def f1(rs):
        tp = sum(r["tp"] for r in rs)
        return 2 * tp / max(2 * tp + sum(r["fp"] + r["fn"] for r in rs), 1)

    by_gap = []
    for lo, hi in pairwise(GAP_EDGES_MS):
        n = hit = 0
        for s, r in per_shot.items():
            inside = (gaps[s] >= lo) & (gaps[s] < hi)
            n += int(inside.sum())
            hit += int((r["matched_onsets"] & inside).sum())
        by_gap.append(
            {
                "from_ms": lo,
                "to_ms": hi,
                "onsets": n,
                "found": hit,
                "recall": hit / max(n, 1),
            }
        )
    found = tot["tp"] + tot["fp"] + tot["empty"] + tot["other"]
    return {
        "shots": len(rows),
        **tot,
        "detections": found,
        "recall": recall(rows),
        "recall_ci95": base.bootstrap(per_shot, recall, rng),
        "judged_precision": tot["tp"] / max(tot["tp"] + tot["fp"], 1),
        "f1_judged": f1(rows),
        "f1_judged_ci95": base.bootstrap(per_shot, f1, rng),
        "absent_alarms_per_10_s": 10 * tot["fp_absent"] / max(guarded_s, 1e-9),
        "by_distance_to_nearest_other_onset_ms": by_gap,
    }


def nearest_gap(onsets: np.ndarray) -> np.ndarray:
    return challenge.nearest_other_gap_ms(onsets)[1]


def occ_metrics(rows) -> dict:
    t = np.concatenate([r[0] for r in rows])
    sc = np.concatenate([r[1] for r in rows])
    call = np.concatenate([r[1] >= r[2] for r in rows])
    tp = int((call & (t == 1)).sum())
    fp = int((call & (t == 0)).sum())
    fn = int((~call & (t == 1)).sum())
    return {
        "auroc": score.roc_auc(t, sc),
        "auprc": score.average_precision(t, sc),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "bins": int(t.size),
    }


def rule_gains(n: dict, duration_ms: float, reported: int = 5) -> dict:
    """What one threshold's pooled counts `n` (`tradeoff`) would save on the merit
    function's published costs with every detection reported as type `reported`
    (`spikes.prices`), by two rules. `as_trained` is the folds' own: every ELM found
    is large and only judged detections are false. `conservative`: an ELM of his found
    inside a crowd is small, and his detections in a present 10 ms cell where he kept
    no onset (`empty`) are false too."""
    tp = n["smith_tp"] + n["his_tp_outside_crowds"]
    fp = n["smith_fp"] + n["review_alarms"] + n["his_fp"]
    return {
        "gain_as_trained": spikes.merit_gain(
            tp + n["his_tp_in_crowds"], fp, duration_ms, reported=reported
        ),
        "gain_conservative": spikes.merit_gain(
            tp,
            fp + n["his_empty"],
            duration_ms,
            tp_small=n["his_tp_in_crowds"],
            reported=reported,
        ),
    }


def tradeoff(data, candidates: dict, duration_ms: float) -> list[dict]:
    """Counts at each grid threshold, pooled over the shots of `candidates`
    (`candidates_of`) by source, and their `rule_gains`."""
    crowd = {
        s: spikes.in_crowd(d.onsets, d.spans)
        for s, d in data.items()
        if d.source == "shousha"
    }
    rows = []
    for thr in base.GRID:
        n = dict.fromkeys(TRADEOFF_COUNTS, 0)
        for s, (t, h) in candidates.items():
            d, found = data[s], t[h >= thr]
            if d.source == "smith":
                tp, fp, fn, _ = base.smith_counts(found, d)
                n["smith_tp"] += tp
                n["smith_fp"] += fp
                n["smith_fn"] += fn
            elif d.source == "review":
                n["review_alarms"] += base.absent_alarms(found, d)
            else:
                j = spikes.judge(found, d.onsets, d.spans)
                n["his_tp_in_crowds"] += int((j["matched_onsets"] & crowd[s]).sum())
                n["his_tp_outside_crowds"] += int(
                    (j["matched_onsets"] & ~crowd[s]).sum()
                )
                for k in ("fn", "fp", "empty", "other"):
                    n[f"his_{k}"] += j[k]
        rows.append({"threshold": float(thr), **n, **rule_gains(n, duration_ms)})
    return rows


def run_evaluate(paths: Paths, args) -> None:
    """The development record: `elm-inst2` out of fold beside `elm-inst`, the pooled
    trade-off over the threshold grid, and the decodings the key is scored with."""
    directory = work(paths)
    if (directory / LOCK).exists():
        raise SystemExit(
            f"{directory / LOCK} exists: the key was scored with the development "
            "record as it is"
        )
    rec = prepared(paths)
    data = load_all(paths)
    records = [
        json.loads((directory / f"fold{k}/fold.json").read_text())
        for k in range(base.FOLDS)
    ]
    old = [
        json.loads((base.work(paths) / f"fold{k}/fold.json").read_text())
        for k in range(base.FOLDS)
    ]
    for k, r in enumerate(records):
        if sha256_of(directory / f"fold{k}/model.pt") != r["checkpoint_sha256"]:
            raise ValueError(f"fold{k}/model.pt is not the checkpoint fold.json names")
    fold_of = {
        s: k
        for k, r in enumerate(records)
        for v in r["shots"]["test"].values()
        for s in v
    }
    rng = np.random.default_rng(base.SEED)
    pred2, pred1 = directory / "pred", base.work(paths) / "pred"
    his1 = directory / "inst1_his"

    def heat_of(run_dir, s):
        return np.load(run_dir / f"{s}.npz")["inst"].astype(np.float32)

    def candidates(run_dir, his_dir, sep):
        return {
            s: candidates_of(
                heat_of(his_dir if d.source == "shousha" else run_dir, s), sep
            )
            for s, d in data.items()
        }

    duration = float(np.mean([r["event_duration_ms"] for r in records]))
    inst1 = {
        "onset_threshold": float(np.mean([r["onset_threshold"] for r in old])),
        "min_sep_ms": inst.MIN_SEP_MS,
        "event_duration_ms": float(np.mean([r["event_duration_ms"] for r in old])),
        "gate": None,
    }
    table = tradeoff(data, candidates(pred2, pred2, spikes.MIN_SEP_MS), duration)
    table1 = tradeoff(
        data, candidates(pred1, his1, inst.MIN_SEP_MS), inst1["event_duration_ms"]
    )
    best = max(table, key=lambda r: (r["gain_conservative"], r["threshold"]))
    served = {
        "onset_threshold": best["threshold"],
        "min_sep_ms": spikes.MIN_SEP_MS,
        "event_duration_ms": duration,
        "gate": None,
        "rule": "peaks of the five fold models' mean heatmap at the grid threshold "
        "where the out-of-fold counts of the three sources, pooled, save the most by "
        "the conservative rule (rule_gains; the higher threshold on a tie), "
        f"{spikes.MIN_SEP_MS:g} ms apart, ungated; each event the mean fold event "
        "duration long",
    }
    alt = {
        **served,
        "onset_threshold": float(np.mean([r["onset_threshold"] for r in records])),
        "rule": "as served, at the mean of the fold thresholds (each fold's "
        "inner-validation maximum of spikes.merit_gain, every ELM found priced large)",
    }

    # his shots judged (`spikes.judge`), Smith's windows matched and the reviewed absent
    # time's alarms counted, out of fold; elm-inst on his shots is its five-model mean
    # (it never saw them), elsewhere its own out-of-fold models
    decodings = {
        "elm-inst2 (fold thresholds)": lambda s, d: (
            pred2,
            records[fold_of[s]]["onset_threshold"],
            spikes.MIN_SEP_MS,
        ),
        "elm-inst2 (served)": lambda s, d: (
            pred2,
            served["onset_threshold"],
            spikes.MIN_SEP_MS,
        ),
        ALT: lambda s, d: (pred2, alt["onset_threshold"], spikes.MIN_SEP_MS),
        "elm-inst": lambda s, d: (
            (his1, inst1["onset_threshold"], inst.MIN_SEP_MS)
            if d.source == "shousha"
            else (pred1, old[fold_of[s]]["onset_threshold"], inst.MIN_SEP_MS)
        ),
    }
    his = {s: d for s, d in data.items() if d.source == "shousha"}
    gaps = {s: nearest_gap(d.onsets) for s, d in his.items()}
    guarded_s = (
        sum(b - a for d in his.values() for a, b in base.guarded_absent(d.spans)) / 1000
    )
    his_view, smith_view, alarm_view = {}, {}, {}
    for name, how in decodings.items():
        judged, matched, alarms = {}, {}, {}
        for s, d in data.items():
            run_dir, thr, sep = how(s, d)
            found = inst.time_of(inst.peaks(heat_of(run_dir, s), thr, sep))
            if d.source == "shousha":
                judged[s] = spikes.judge(found, d.onsets, d.spans)
            elif d.source == "smith":
                matched[s] = base.smith_counts(found, d)
            else:
                alarms[s] = base.alarms_of(found, d)
        his_view[name] = his_summary(judged, gaps, guarded_s, rng)
        smith_view[name] = base.counts_summary(matched, rng)
        alarm_view[name] = base.alarm_summary(alarms)
    occupancy = {}
    for name, run_dir, recs in (
        ("elm-inst2", pred2, records),
        ("elm-inst", pred1, old),
    ):
        per = {
            s: (
                d.bins.truth,
                labels.bin_scores(np.load(run_dir / f"{s}.npz")["occ"], d.bins),
                recs[fold_of[s]]["occ_threshold"],
            )
            for s, d in data.items()
            if d.source == "review"
        }
        m = occ_metrics(list(per.values()))
        m["auroc_ci95"] = base.bootstrap(
            per, lambda rows: occ_metrics(rows)["auroc"], rng
        )
        m["f1_ci95"] = base.bootstrap(per, lambda rows: occ_metrics(rows)["f1"], rng)
        occupancy[name] = m

    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_sha(full=True),
        "run": str(directory),
        "scripts_sha256": {
            p.name: sha256_of(p)
            for p in (
                Path(__file__),
                HERE / "elm_inst.py",
                REPO / "src/labeler/elm/spikes.py",
                REPO / "src/labeler/elm/inst.py",
            )
        },
        "protocol": "Development scores, out of fold: reviewed shots split as cv2, Smith "
        "shots as the Smith onset run, his shots by run day; no fold model saw its test "
        "shots of any source. His onsets are the elm_jump v2 onsets inside his accepted "
        "rows, merged at spikes.MERGE_MS: agreement with them is agreement with a "
        "reviewed elm_jump, not with hand timing. The decodings the key is scored with "
        "are fixed here: elm-inst2 (primary) at the conservative rule's maximum over "
        "the pooled out-of-fold counts, elm-inst2-alt at the mean fold threshold. "
        "Single out-of-fold models are scored here, never the served five-model mean "
        "(elm-inst on his shots is its five-model mean), and the bootstrap intervals "
        "of two methods are not paired. The challenge key is not read here; its "
        "shots' traces are only hashed.",
        "his_shots": {
            "pin": rec["pin"],
            "pin_sha256": rec["pin_sha256"],
            "jump_table_sha256": rec["jump_table_sha256"],
            "shots": len(rec["shots"]),
            "left_out": rec["left_out"],
            "totals": rec["totals"],
            "folds": rec["folds"],
        },
        "fold_thresholds": {
            "onset_merit": [r["onset_threshold"] for r in records],
            "onset_f1": [r["f1_threshold"] for r in records],
            "occupancy": [r["occ_threshold"] for r in records],
            "event_duration_ms": [r["event_duration_ms"] for r in records],
            "selected_epochs": [r["best"]["epoch"] for r in records],
        },
        "primary": METHOD,
        "served": served,
        "alt": alt,
        "elm_inst_served": inst1,
        "tradeoff_pooled_out_of_fold": {
            "rules": "gain_as_trained = spikes.merit_gain(smith_tp + his tp, smith_fp "
            "+ review_alarms + his_fp); gain_conservative = spikes.merit_gain(smith_tp "
            "+ his_tp_outside_crowds, smith_fp + review_alarms + his_fp + his_empty, "
            "tp_small=his_tp_in_crowds); each with the method's mean fold event "
            "duration",
            "elm-inst2": table,
            "elm-inst": table1,
        },
        "his_onsets_2ms": his_view,
        "onsets_smith_windows_2ms": smith_view,
        "alarms_in_guarded_review_absent_time": alarm_view,
        "occupancy_review_bins": occupancy,
        "hashes": {
            "checkpoints": {
                f"fold{k}": r["checkpoint_sha256"] for k, r in enumerate(records)
            },
            "key_traces": {
                p.stem: sha256_of(p) for p in sorted((directory / "key").glob("*.npz"))
            },
            "elm_inst_key_traces": {
                p.stem: sha256_of(p)
                for p in sorted((base.work(paths) / "key").glob("*.npz"))
            },
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1, default=float) + "\n")
    for r in table:
        print(json.dumps({k: round(v, 1) for k, v in r.items()}), flush=True)
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "served",
                    "alt",
                    "his_onsets_2ms",
                    "onsets_smith_windows_2ms",
                    "alarms_in_guarded_review_absent_time",
                    "occupancy_review_bins",
                )
            },
            indent=1,
            default=float,
        )
    )


# ---------------------------------------------------------------------- key


def covered(starts, stops, times) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    ok = np.zeros(len(t), dtype=bool)
    for a, b in zip(starts, stops, strict=True):
        ok |= (t >= a) & (t <= b)
    return ok


def shot_of(label: str) -> int:
    """The shot of a merit-function case label `DIII-D#<shot>: <description>`."""
    return int(label.split("#", 1)[1].split(":", 1)[0])


def window_of(case: dict) -> tuple[float, float]:
    return case["tstart"], case["tend"]


def decode(z: dict, served: dict, window) -> np.ndarray:
    """One key shot's spans from its five-model mean traces `z`: the heatmap's peaks at
    `served`'s threshold, `min_sep_ms` apart, inside the valid input and the case
    `window`, each `event_duration_ms` long; ungated, as `elm_challenge_evaluate`
    decoded `elm-inst`."""
    c0, c1 = inputs.valid_intervals(z["valid"])
    t = inst.time_of(
        inst.peaks(z["inst"], served["onset_threshold"], served["min_sep_ms"])
    )
    t = t[covered(c0, c1, t) & (t >= window[0]) & (t <= window[1])]
    d = served["event_duration_ms"]
    return np.column_stack([t, t + d]) if len(t) else np.zeros((0, 2))


def traces_of(directory: Path, shots, sums: dict) -> dict:
    """The key shots' five-model mean traces in `directory`, each the file whose sha256
    the development record holds."""
    out = {}
    for s in shots:
        path = directory / f"{s}.npz"
        if sha256_of(path) != sums.get(str(s)):
            raise SystemExit(f"{path} is not the trace the development record hashed")
        z = np.load(path)
        out[s] = {"inst": z["inst"], "valid": z["valid"]}
    return out


def write_submissions(
    folder: Path, found: dict, cases: dict, method: str, comment: str
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.json"):
        old.unlink()
    for s, spans in found.items():
        record = challenge.submission(
            s, spans, window_of(cases[s]), method=method, comment=comment
        )
        (folder / f"DIII-D_{s}.json").write_text(json.dumps(record) + "\n")


def grade(paths: Paths, folders: dict[str, Path], out: Path) -> dict:
    """`folders` graded by the challenge's merit function at `CLONE_COMMIT`, through
    `elm_challenge_merit.py` in its own venv, into `out`."""
    command = [
        str(paths.root / "envs/elm_challenge/bin/python"),
        str(REPO / "scripts/labeler/elm_challenge_merit.py"),
        "--clone",
        str(paths.root / CLONE),
        "--key",
        str(paths.label_tables / KEY),
        "--out",
        str(out),
    ]
    for name, folder in folders.items():
        command += ["--folder", f"{name}={folder}"]
    subprocess.run(command, check=True)
    merit = json.loads(out.read_text())
    if not merit["clone_commit"].startswith(CLONE_COMMIT):
        raise SystemExit(f"the clone is at {merit['clone_commit']}, not {CLONE_COMMIT}")
    return merit


def claim(lock: Path, note: dict) -> None:
    """Create `lock`, or refuse when it exists. It lives in the run folder under the
    data root, so no checkout can score the key a second time."""
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        raise SystemExit(f"{lock} exists: the key is scored once") from None
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(note) + "\n")


def reproduce(paths: Paths, dev: dict, cases: dict, scratch: Path):
    """`elm-inst`'s graded submissions rebuilt from its key traces through `decode`:
    their events must be the graded files' and the merit function must give their
    totals and no detection's again, or `decode` is not the path `elm-inst` was scored
    by. Returns the rebuilt spans, the merit record and the check; nothing of
    `elm-inst2` is graded here."""
    stored = paths.root / SUBMISSIONS
    traces = traces_of(
        base.work(paths) / "key", sorted(cases), dev["hashes"]["elm_inst_key_traces"]
    )
    found = {
        s: decode(z, dev["elm_inst_served"], window_of(cases[s]))
        for s, z in traces.items()
    }
    folder = scratch / "elm-inst"
    write_submissions(folder, found, cases, "elm-inst", "elm-inst rebuilt as a check")
    for s in sorted(cases):
        mine = json.loads((folder / f"DIII-D_{s}.json").read_text())
        graded = json.loads((stored / "elm-inst" / f"DIII-D_{s}.json").read_text())
        differ = [
            k for k in ("time", "event_flag", "event_type") if mine[k] != graded[k]
        ]
        if differ:
            raise SystemExit(
                f"{s}: the rebuilt elm-inst {differ} differ from the graded file"
            )
    merit = grade(
        paths,
        {"elm-inst": folder, "no-detection": stored / "no-detection"},
        scratch / "merit.json",
    )
    published = json.loads((paths.root / STORED_MERIT).read_text())
    totals = {}
    for name in ("elm-inst", "no-detection"):
        now = float(merit["folders"][name]["total"]["total"])
        then = float(published["folders"][name]["total"]["total"])
        if abs(now - then) > 1e-6:
            raise SystemExit(
                f"{name}: merit {now} now against {then} in {STORED_MERIT}"
            )
        totals[name] = now
    return (
        found,
        merit,
        {
            "elm_inst_files_identical": True,
            "totals": totals,
            "clone_commit": merit["clone_commit"],
        },
    )


def signed_offsets(case: dict, spans, tol_ms: float) -> list[float]:
    """Detection start minus key onset (ms) of each pair `challenge.onset_match` makes."""
    onsets, _ = challenge.key_onsets(case)
    found = np.asarray(spans, dtype=float).reshape(-1, 2)[:, 0]
    found = found[(found >= case["tstart"]) & (found <= case["tend"])]
    m = boundary.match(onsets, found, tol_ms)
    return (m.estimate[m.pairs[:, 1]] - m.reference[m.pairs[:, 0]]).tolist()


def onset_summary(cases: dict, found: dict, shots, tol_ms: float) -> dict:
    """Detections against the key's ELM onsets over `shots`, pooled
    (`challenge.onset_match`)."""
    rows = [challenge.onset_match(cases[s], found[s], tol_ms) for s in shots]
    tp, fp, fn, excluded = (
        sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "excluded")
    )
    signed = [d for s in shots for d in signed_offsets(cases[s], found[s], tol_ms)]
    return {
        "n_shots": len(shots),
        **challenge.bin_metrics_from_counts(tp, fp, fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "excluded_beside_uncertain_events": excluded,
        "median_signed_offset_ms": float(np.median(signed)) if signed else None,
        "median_abs_offset_ms": float(np.median(np.abs(signed))) if signed else None,
    }


def crowding_of(cases: dict, found: dict, shots) -> list[dict]:
    """The key's ELMs by the distance to their nearest other, and how many have a
    detection start within 2 ms (not one to one)."""
    gaps, offsets = [], []
    for s in shots:
        onsets, _ = challenge.key_onsets(cases[s])
        onsets, gap = challenge.nearest_other_gap_ms(onsets)
        gaps.append(gap)
        offsets.append(
            challenge.nearest_signed_offsets_ms(onsets, found[s][:, 0], NEAR_MS)
        )
    gap, offset = np.concatenate(gaps), np.concatenate(offsets)
    hit = np.abs(offset) <= TOLERANCES_MS[1]
    return [
        {
            "from_ms": lo,
            "to_ms": hi,
            "key_elms": int(((gap >= lo) & (gap < hi)).sum()),
            "detection_within_2_ms": int((hit & (gap >= lo) & (gap < hi)).sum()),
        }
        for lo, hi in pairwise(GAP_EDGES_MS)
    ]


def scopes_of(paths: Paths, cases: dict) -> dict[str, set]:
    shots = set(cases)
    samples = {
        json.loads(p.read_text())["shot"]
        for p in (paths.root / CLONE / "samples").glob("*.json")
    }
    busy = {s for s in shots if len(cases[s]["events"]) >= BUSY_EVENTS}
    return {
        "all_graded_cases": shots,
        "on_the_samples_shots": samples & shots,
        f"windows_with_{BUSY_EVENTS}_or_more_key_events": busy,
        f"windows_with_fewer_than_{BUSY_EVENTS}_key_events": shots - busy,
    }


def views(cases: dict, found: dict, merit: dict, scopes: dict) -> dict:
    """Merit sums by scope for the folders graded in `merit`; onset matches at each
    tolerance and by crowding for the methods decoded in `found`."""
    merit_view = {}
    for name, scores in merit["folders"].items():
        per_case = {
            shot_of(v["label"]): v
            for c, v in scores.items()
            if c not in ("total", "total_not_counting_missing") and not v["missing"]
        }
        merit_view[name] = {
            scope: {
                "n_cases": len(want & set(per_case)),
                **{
                    k: float(sum(per_case[s][k] for s in want if s in per_case))
                    for k in MERIT_KEYS
                },
            }
            for scope, want in scopes.items()
        }
        merit_view[name]["per_case"] = {
            str(s): {k: float(v[k]) for k in MERIT_KEYS}
            for s, v in sorted(per_case.items())
        }
    shots = sorted(scopes["all_graded_cases"])
    return {
        "merit": merit_view,
        "onsets": {
            method: {
                scope: {
                    f"{tol:g} ms": onset_summary(cases, spans, sorted(want), tol)
                    for tol in TOLERANCES_MS
                }
                for scope, want in scopes.items()
            }
            for method, spans in found.items()
        },
        "onset_by_crowding": {
            method: crowding_of(cases, spans, shots) for method, spans in found.items()
        },
        "detections": {
            method: {str(s): len(v) for s, v in spans.items()}
            for method, spans in found.items()
        },
    }


def shared_run_days(paths: Paths, key_shots) -> dict:
    """The key shots whose run day a shot that trained the five models shares, with
    those shots by source: such a key shot is not independent of the training data."""
    trained: dict[str, set[int]] = {}
    for k in range(base.FOLDS):
        fold = json.loads((work(paths) / f"fold{k}/fold.json").read_text())["shots"]
        for part in fold.values():
            for source, shots in part.items():
                trained.setdefault(source, set()).update(int(s) for s in shots)
    days = heldout.run_days(paths, set(key_shots).union(*trained.values()))
    shared = {}
    for s in key_shots:
        if s not in days:
            continue
        same = {
            source: sorted(t for t in shots if days.get(t) == days[s])
            for source, shots in trained.items()
        }
        same = {source: v for source, v in same.items() if v}
        if same:
            shared[str(s)] = {"run_day": days[s], "training_shots": same}
    return {
        "shared": shared,
        "key_shots_without_a_run_day": [s for s in key_shots if s not in days],
    }


def run_key(paths: Paths, args) -> None:
    """The challenge key, scored once. `elm-inst`'s graded files are rebuilt through
    `decode` first (`reproduce`) and every view is drawn for it, so a failure costs no
    scoring; then the lock, and `elm-inst2` and `elm-inst2-alt` with the decodings the
    development record fixed, graded beside `elm-inst` and no detection."""
    directory = work(paths)
    lock, scores_file = directory / LOCK, directory / SCORES
    for path in (KEY_OUT, lock, scores_file):
        if path.exists():
            raise SystemExit(
                f"{path} exists: the key is scored once (delete it by hand)"
            )
    dev = json.loads(OUT.read_text())
    cases = {c["shot"]: c for c in read_elm_challenge(paths.label_tables / KEY)}
    shots = sorted(cases)
    scopes = scopes_of(paths, cases)
    run_days = shared_run_days(paths, shots)
    found_inst1, merit_inst1, check = reproduce(
        paths, dev, cases, directory / "key_check"
    )
    # every view once on elm-inst alone: a failure here costs no scoring
    views(cases, {"elm-inst": found_inst1}, merit_inst1, scopes)
    traces = traces_of(directory / "key", shots, dev["hashes"]["key_traces"])
    claim(
        lock,
        {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "git": git_sha(full=True),
            "job": os.environ.get("SLURM_JOB_ID"),
            "development_record_sha256": sha256_of(OUT),
        },
    )
    submissions = paths.root / SUBMISSIONS
    decodings = {METHOD: dev["served"], ALT: dev["alt"]}
    found = {}
    for method, served in decodings.items():
        found[method] = {
            s: decode(traces[s], served, window_of(cases[s])) for s in shots
        }
        write_submissions(
            submissions / method,
            found[method],
            cases,
            method,
            f"{COMMENT}; threshold {served['onset_threshold']:g}",
        )
    found["elm-inst"] = found_inst1
    merit = grade(
        paths,
        {m: submissions / m for m in (METHOD, ALT, "elm-inst", "no-detection")},
        scores_file,
    )
    out = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_sha(full=True),
        "clone_commit": merit["clone_commit"],
        "key_sha256": merit["key_sha256"],
        "development_record_sha256": sha256_of(OUT),
        "statement": "scored with the challenge merit function at commit b71f991, once; "
        "not an official leaderboard result. elm-inst2 is the primary, elm-inst2-alt "
        "the alternative, both declared in the development record before the key was "
        "read. Sums are over graded cases only.",
        "primary": METHOD,
        "decodings": {**decodings, "elm-inst": dev["elm_inst_served"]},
        "reproduction_check": check,
        "run_days": run_days,
        **views(cases, found, merit, scopes),
    }
    text = json.dumps(out, indent=1, default=float) + "\n"
    KEY_OUT.parent.mkdir(parents=True, exist_ok=True)
    KEY_OUT.write_text(text)
    (directory / "key_record.json").write_text(text)
    print(
        json.dumps(
            {n: {sc: v[sc]["total"] for sc in scopes} for n, v in out["merit"].items()},
            indent=1,
        )
    )
    print(
        json.dumps(
            {m: v["all_graded_cases"] for m, v in out["onsets"].items()},
            indent=1,
            default=float,
        )
    )
    print(json.dumps(out["onset_by_crowding"], indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("prepare", "train", "predict", "evaluate", "key"))
    ap.add_argument("--fold", type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=base.EPOCHS)
    ap.add_argument("--iters", type=int, default=base.ITERS)
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    {
        "prepare": run_prepare,
        "train": run_train,
        "predict": run_predict,
        "evaluate": run_evaluate,
        "key": run_key,
    }[args.stage](paths, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
