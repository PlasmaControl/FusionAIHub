"""saw-ours v2 (fix6): the learned ECE picker combined with the derivative picker.

fix5's saw-ours lost to saw-derivative on presence F1 in the figure's view
(`no_q1_negatives`), and the fix5 diagnosis gave three reasons: every operating
point was selected on the primary view, whose negatives (ECE-tested quiet time)
both models separate almost perfectly, so the selected thresholds said nothing
about the hard negatives the figure's view adds (ECE edges in time with no q = 1
surface); the network never saw those negatives in training; and its 100 ms
window misses long-period trains whose next crash is out of view.

Variants (`VARIANTS`), all on the fix5 folds, inner splits and labels:
- f5: the fix5 network, unchanged;
- a: refit with the no-q = 1 negatives in the supervision mask;
- b: a plus the derivative picker's z-trace and edge score as two input channels;
- c: b with a 400 ms window (100 ms margin at inference).
Each variant's presence may be ANDed with the derivative edge score
(`labeler.sawtooth.combine`). Its threshold and z, and the crash threshold, are
selected per fold on the inner-selection shots in the no-q = 1 view. The winner
is the variant with the best pooled inner-selection presence F1; it is reported
as saw-ours. The derivative alone is re-selected the same way as an ablation.
An output q = 1 gate is a sensitivity only: it zeroes exactly the time the view
adds as negatives, so it is never a candidate for the winner.

Stages: fit (a, b, c), infer, calibrate, select, evaluate. Nothing is written
under the fix5 work tree or outputs; the record goes to outputs/.../fix6.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sawtooth_benchmark as sb
from sawtooth_physics import REPO, REVIEW, SEED, save_json

from labeler.config import Paths
from labeler.sawtooth import combine
from labeler.sawtooth.metrics import (
    aggregate,
    bin_times,
    binary_cells,
    event_cells,
    paired_bootstrap,
    point_metrics,
    presence_from_cells,
    score_histogram,
    spans_at,
)
from labeler.sawtooth.models import PhasePicker

ROOT = Paths.from_env().root
FIX5_WORK = ROOT / "round4/hl3/work"
FIX5_OUTPUT = REPO / "outputs/labeler/sawtooth/fix5"
OUTPUT = REPO / "outputs/labeler/sawtooth/fix6"
WORK = ROOT / "round4/hl3/work6"
VARIANTS = {
    "f5": {
        "train": False,
        "supervision": "primary",
        "derivative_inputs": False,
        "width": 1000,
        "margin": 100,
    },
    "a": {
        "train": True,
        "supervision": "no_q1",
        "derivative_inputs": False,
        "width": 1000,
        "margin": 100,
    },
    "b": {
        "train": True,
        "supervision": "no_q1",
        "derivative_inputs": True,
        "width": 1000,
        "margin": 100,
    },
    "c": {
        "train": True,
        "supervision": "no_q1",
        "derivative_inputs": True,
        "width": 4000,
        "margin": 1000,
    },
}
DESCRIPTION = {
    "f5": "fix5 network unchanged (48 ECE channels, 100 ms window, primary-view "
    "supervision); operating point and derivative AND re-selected",
    "a": "refit with observable no-q = 1 time in the supervision mask (the "
    "review source's absent definition); 48 ECE channels, 100 ms window",
    "b": "a plus two derivative input channels: the axis-channel z-trace and the "
    "+/-125 ms edge score, log1p-compressed",
    "c": "b with a 400 ms window (100 ms inference margin) for long-period trains",
}
Z_GRID = (0, *sb.DERIVATIVE_Z)
POLICY = (
    "fix6: observable-only inputs; supervision as the variant states; "
    "balanced crash-centred; operating points on inner selection in the "
    "no-q = 1 view"
)
SELECTION = (
    "per fold, on the inner-selection shots only, in the no-q = 1 view "
    "(assessed bins plus observable no-q = 1 time, labels unchanged): presence "
    "threshold and AND z by presence F1 on 2 ms bins (ties: larger threshold, "
    "then smaller z); crash threshold and whether picks need presence by crash "
    "F1 at +/-2 ms (ties: larger threshold)"
)
CIRCULARITY = (
    "the derivative picker reads the same axis-region ECE edges the physics-rule "
    "labels are built from (present = periodic train of core edges; tested "
    "absence = no qualified core edge within +/-375 ms), so it is close to the "
    "labelling rule itself (inner-selection primary-view presence F1 0.997-1.0) "
    "and any model that combines with it inherits that partial circularity"
)
GATE_CAVEAT = (
    "the q-gated rows zero presence and drop picks where labeler.sawtooth.qgate "
    "calls no_q1, which is exactly the time the no-q = 1 view adds as negatives, "
    "so those negatives become true negatives by construction and the view "
    "reduces to the primary view; the gated rows are a sensitivity, never the "
    "drawn model"
)


# ---------------------------------------------------------------- inputs


def split_manifest():
    """The fix5 split, read (never rebuilt: `sb.manifest` writes to fix5)."""
    split = json.loads((FIX5_WORK / "split_manifest.json").read_text())
    installed = json.loads((FIX5_OUTPUT / "split_manifest.json").read_text())
    for key in ("folds", "training_cohort", "fixed_validation_supported"):
        if split[key] != installed[key]:
            raise ValueError(f"fix5 split manifests disagree on {key}")
    split["folds"] = {int(k): v for k, v in split["folds"].items()}
    return split


def fold_shots(split, fold):
    seed = SEED + fold
    outer = [s for s in split["training_cohort"] if split["folds"][s] != fold]
    heldout = [s for s in split["training_cohort"] if split["folds"][s] == fold]
    training, selection = sb.inner_split(outer, seed=seed)
    sb.guard_partition(split, training, selection, heldout)
    return training, selection, heldout


def derivative_parts(signal):
    """Observable derivative peaks (s), their z, and the native edge score."""
    peaks, amplitudes = sb.derivative_candidates(signal)
    t = np.asarray(signal["t"], float)
    observable = sb.masks_at(signal, t[peaks])[0]
    times, amps = t[peaks][observable], amplitudes[observable]
    return times, amps, combine.edge_score(t, times, amps)


def model_inputs(signal, spec):
    values = sb.input_values(signal, "saw-ours")
    if not spec["derivative_inputs"]:
        return values
    z, _ = combine.derivative_trace(signal)
    _, _, edge = derivative_parts(signal)
    extra = combine.input_channels(z, edge)
    extra[:, ~np.asarray(signal["observable"], bool)] = np.nan
    return np.concatenate([values, extra])


def view(shot, signal, supervision):
    if supervision == "primary":
        return signal
    return sb.with_q_gate(signal, sb.q_gate(shot, signal))


# ---------------------------------------------------------------- fitting


def windows(work, shots, spec, boundary, *, per_shot):
    """`sb.windows` for the PhaseNet picker, with the variant's inputs, width and
    supervision mask; the same per-shot random generator and sampling policy."""
    width = spec["width"]
    xs, ys, owners = [], [], []
    for shot in shots:
        data = sb.load_signal(work, shot)
        t = data["t"]
        if len(t) < width:
            continue
        x = model_inputs(data, spec)
        rec = sb.record(work, shot)
        pick, present, classes = sb.targets(t, rec, boundary)
        supervised = view(shot, data, spec["supervision"])
        observable, assessed = sb.masks_at(supervised, t)
        rng = np.random.default_rng(SEED + shot)
        centers = sb.assessed_points(
            supervised, [r["time_s"] for r in sb.present_rows(rec, "crashes")]
        )
        starts = sb.sample_window_starts(
            t, classes, centers, width, rng, count=per_shot
        )
        for start in starts:
            mask = assessed[start : start + width]
            if observable[start : start + width].mean() < 0.8 or not mask.any():
                continue
            window = x[:, start : start + width]
            with np.errstate(over="ignore"):
                stored = window.astype(np.float16)
            if (np.isinf(stored) & np.isfinite(window)).any():
                raise ValueError(f"shot {shot}: an input overflows float16")
            xs.append(stored)
            ys.append(
                np.stack(
                    [pick[start : start + width], present[start : start + width], mask]
                ).astype(np.float16)
            )
            owners.append(shot)
    if not xs:
        raise ValueError(f"no supervised windows on {shots}")
    return np.stack(xs), np.stack(ys), np.array(owners)


def fit(args, variant, fold, split, device):
    spec = VARIANTS[variant]
    out = args.work / "models" / variant / f"fold_{fold}"
    if (out / "fit.json").exists():
        print(f"resume: {variant} fold {fold} fitted", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    training, selection, heldout = fold_shots(split, fold)
    boundary = sb.period_boundary(args.work, training)
    begun = time.monotonic()
    x, y, owners = windows(args.work, training, spec, boundary, per_shot=64)
    vx, vy, _ = windows(args.work, selection, spec, boundary, per_shot=32)
    mean, std = sb.normalizer(x)
    seed = SEED + fold
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PhasePicker(channels=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    history, best, best_epoch, stale, epoch, best_state = [], np.inf, -1, 0, 0, None
    batch = 32
    while stale < args.patience and epoch < args.max_epochs:
        epoch += 1
        model.train()
        losses = []
        for _ in range(args.steps):
            selected = rng.integers(0, len(x), size=batch)
            logits = model(
                torch.from_numpy(sb.normalize(x[selected], mean, std)).to(device)
            )
            value = sb.loss(
                "saw-ours",
                logits,
                torch.from_numpy(y[selected].astype(np.float32)).to(device),
            )
            optimizer.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(value.item())
        model.eval()
        vloss, vweights = [], []
        with torch.no_grad():
            for start in range(0, len(vx), batch):
                yy = vy[start : start + batch].astype(np.float32)
                logits = model(
                    torch.from_numpy(
                        sb.normalize(vx[start : start + batch], mean, std)
                    ).to(device)
                )
                vloss.append(
                    sb.loss("saw-ours", logits, torch.from_numpy(yy).to(device)).item()
                )
                vweights.append(float(yy[:, 2].sum()))
        validation = float(np.average(vloss, weights=vweights))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": validation,
            }
        )
        print(
            f"{variant} fold {fold} epoch {epoch}: train {np.mean(losses):.4f} val {validation:.4f}",
            flush=True,
        )
        if not np.isfinite(validation):
            raise ValueError("nonfinite inner selection loss")
        if validation < best - args.min_delta:
            best, best_epoch, stale = validation, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stale += 1
        scheduler.step()
    torch.save(
        {
            "state_dict": best_state,
            "mean": mean,
            "std": std,
            "channels": x.shape[1],
            "variant": variant,
            "spec": spec,
            "boundary_ms": boundary,
            "training_shots": training,
            "selection_shots": selection,
        },
        out / "checkpoint.pt",
    )
    save_json(
        out / "fit.json",
        {
            "variant": variant,
            "spec": spec,
            "description": DESCRIPTION[variant],
            "policy": POLICY,
            "fold": fold,
            "seed": seed,
            "training_shots": training,
            "selection_shots": selection,
            "heldout_shots": heldout,
            "training_windows": len(owners),
            "selection_windows": len(vx),
            "input_channels": int(x.shape[1]),
            "history": history,
            "best_epoch": best_epoch,
            "best_inner_loss": best,
            "epochs_completed": epoch,
            "stopping_reason": "inner selection loss patience exhausted"
            if stale >= args.patience
            else "maximum epoch budget reached",
            "optimizer": "Adam lr 0.001 wd 0.0001, ExponentialLR 0.97, batch 32, "
            f"{args.steps} steps/epoch, patience {args.patience}, min delta {args.min_delta}",
            "seconds": round(time.monotonic() - begun, 1),
            "device": str(device),
            "threads": torch.get_num_threads(),
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    )


# ---------------------------------------------------------------- inference


def network(variant, fold, device):
    if variant == "f5":
        path = FIX5_WORK / "models/saw-ours" / f"fold_{fold}" / "checkpoint.pt"
    else:
        path = WORK / "models" / variant / f"fold_{fold}" / "checkpoint.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = PhasePicker(channels=state.get("channels", 48)).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state["mean"], state["std"]


def infer(model, signal, spec, mean, std, device):
    """`sb.infer`'s PhaseNet branch with the variant's inputs, width and margin."""
    t = signal["t"]
    values = model_inputs(signal, spec)
    width, margin = spec["width"], spec["margin"]
    hop = width - 2 * margin
    starts = np.arange(0, len(t), hop)
    probability = np.zeros((2, len(t)), dtype=np.float32)
    count = np.zeros(len(t), dtype=np.float32)
    batch = 32 if width <= 1000 else 8
    with torch.no_grad():
        for offset in range(0, len(starts), batch):
            selected = starts[offset : offset + batch]
            chunks = [
                values[
                    :, np.clip(np.arange(s - margin, s - margin + width), 0, len(t) - 1)
                ]
                for s in selected
            ]
            logits = model(
                torch.from_numpy(sb.normalize(np.stack(chunks), mean, std)).to(device)
            )
            output = (
                torch.stack(
                    [logits[:, :2].softmax(dim=1)[:, 1], logits[:, 2].sigmoid()], dim=1
                )
                .cpu()
                .numpy()
            )
            for start, p in zip(selected, output, strict=True):
                end = min(len(t), start + hop)
                probability[:, start:end] += p[:, margin : margin + end - start]
                count[start:end] += 1
    probability /= np.maximum(count, 1)
    return {"t": t, "crash": probability[0], "presence": probability[1]}


def prediction_path(variant, fold, shot):
    if variant == "f5":
        stored = FIX5_WORK / "predictions/saw-ours" / f"fold_{fold}" / f"{shot}.npz"
        if stored.exists():
            return stored
    return WORK / "predictions" / variant / f"fold_{fold}" / f"{shot}.npz"


def run_inference(args, variant, fold, split, device):
    spec = VARIANTS[variant]
    _, selection, heldout = fold_shots(split, fold)
    shots = (
        selection
        + heldout
        + split["expert_shots"]
        + split["fixed_validation_supported"]
    )
    model, mean, std = network(variant, fold, device)
    for shot in shots:
        path = prediction_path(variant, fold, shot)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        signal = sb.load_signal(args.work, shot)
        if variant == "f5":
            raw = sb.infer(model, "saw-ours", signal, mean, std)
            raw = {k: raw[k] for k in ("t", "crash", "presence")}
        else:
            raw = infer(model, signal, spec, mean, std, device)
        np.savez_compressed(path, **raw)
    print(f"inferred {variant} fold {fold}: {len(shots)} shots", flush=True)


def load_raw(variant, fold, shot):
    with np.load(prediction_path(variant, fold, shot)) as data:
        return {k: data[k] for k in ("t", "crash", "presence")}


# ---------------------------------------------------------------- operating points


def presence_score(raw, edge, z, no_q1=None):
    score = combine.combined_presence(raw["presence"], edge, z)
    return score if no_q1 is None else combine.gated_presence(score, no_q1)


def model_picks(
    signal, crash, threshold, presence=None, presence_threshold=None, no_q1=None
):
    peaks, _ = find_peaks(crash, height=threshold, distance=50)
    if presence is not None:
        peaks = peaks[presence[peaks] >= presence_threshold]
    if no_q1 is not None:
        peaks = peaks[~no_q1[peaks]]
    return sb.observable_points(signal, signal["t"][peaks])


def derivative_picks(signal, times, amps, z, no_q1=None):
    keep = amps >= z
    picked = times[keep]
    if no_q1 is not None and len(picked):
        index = np.clip(np.searchsorted(signal["t"], picked), 0, len(signal["t"]) - 1)
        picked = picked[~no_q1[index]]
    return picked


class Shot:
    """One shot's signal, record, gate and derivative parts, read once."""

    def __init__(self, work, shot):
        self.shot = shot
        self.signal = sb.load_signal(work, shot)
        self.rec = sb.record(work, shot)
        self.gate = sb.q_gate(shot, self.signal)
        self.no_q1 = np.asarray(self.gate.no_q1, bool)
        self.times, self.amps, self.edge = derivative_parts(self.signal)
        self.view = sb.with_q_gate(self.signal, self.gate)
        t = self.signal["t"]
        self.bins = bin_times((t[0], t[-1]))
        self.valid = sb.masks_at(self.view, self.bins)[1]
        spans = [
            (r["start_s"], r["end_s"]) for r in sb.present_rows(self.rec, "intervals")
        ]
        self.truth = spans_at(self.bins, spans)[self.valid]
        self.reference = sb.assessed_points(
            self.view, [r["time_s"] for r in sb.present_rows(self.rec, "crashes")]
        )

    def views(self):
        """The fix5 scoring views of this shot, built once."""
        if not hasattr(self, "_views"):
            self._views = {
                "primary": self.signal,
                "old_negatives_sensitivity": sb.with_q_prior(self.signal),
                "no_q1_negatives": self.view,
                **{
                    f"edge_{c:g}_ms": sb.with_edge_context(self.signal, self.rec, c)
                    for c in sb.EDGE_CONTEXTS_MS
                },
            }
        return self._views

    def at_bins(self, score):
        return np.interp(self.bins, self.signal["t"], score)[self.valid]

    def crash_cells(self, picks):
        return event_cells(self.reference, sb.assessed_points(self.view, picks), 2)


def best(options, key):
    return max(options, key=key)


def calibrate_model(shots, raws, *, gated, z_grid):
    """Presence (threshold, z) then crash (threshold, picks need presence) on the
    inner-selection shots in the no-q = 1 view."""
    options = []
    for z in z_grid:
        scored = [
            (
                s.truth,
                s.at_bins(
                    presence_score(raws[s.shot], s.edge, z, s.no_q1 if gated else None)
                ),
            )
            for s in shots
        ]
        for threshold in sb.THRESHOLDS:
            cells = sum(
                (binary_cells(tr, p >= threshold) for tr, p in scored),
                start=np.zeros(4),
            )
            options.append(
                {
                    "threshold": float(threshold),
                    "z": z,
                    "cells": cells.astype(int).tolist(),
                    **point_metrics(cells[:3]),
                }
            )
    chosen = best(
        options, lambda r: (-1 if r["f1"] is None else r["f1"], r["threshold"], -r["z"])
    )
    crash_options = []
    for need_presence in (False, True):
        for threshold in sb.THRESHOLDS:
            cells = np.zeros(3)
            for s in shots:
                raw = raws[s.shot]
                presence = presence_score(
                    raw, s.edge, chosen["z"], s.no_q1 if gated else None
                )
                cells += s.crash_cells(
                    model_picks(
                        s.signal,
                        raw["crash"],
                        threshold,
                        presence if need_presence else None,
                        chosen["threshold"],
                        s.no_q1 if gated else None,
                    )
                )
            crash_options.append(
                {
                    "threshold": float(threshold),
                    "need_presence": need_presence,
                    "cells": cells.astype(int).tolist(),
                    **point_metrics(cells),
                }
            )
    crash = best(
        crash_options,
        lambda r: (
            -1 if r["f1"] is None else r["f1"],
            r["threshold"],
            not r["need_presence"],
        ),
    )
    return {
        "presence": chosen,
        "crash": crash,
        "presence_options": options,
        "crash_options": crash_options,
    }


def calibrate_derivative(shots, *, gated):
    options, crash_options = [], []
    for z in sb.DERIVATIVE_Z:
        cells = sum(
            (
                binary_cells(
                    s.truth,
                    s.at_bins(
                        presence_score(
                            {"presence": np.ones(len(s.signal["t"]), np.float32)},
                            s.edge,
                            z,
                            s.no_q1 if gated else None,
                        )
                    )
                    >= 0.5,
                )
                for s in shots
            ),
            start=np.zeros(4),
        )
        options.append(
            {"z": z, "cells": cells.astype(int).tolist(), **point_metrics(cells[:3])}
        )
        crash = sum(
            (
                s.crash_cells(
                    derivative_picks(
                        s.signal, s.times, s.amps, z, s.no_q1 if gated else None
                    )
                )
                for s in shots
            ),
            start=np.zeros(3),
        )
        crash_options.append(
            {"z": z, "cells": crash.astype(int).tolist(), **point_metrics(crash)}
        )
    key = lambda r: (-1 if r["f1"] is None else r["f1"], r["z"])
    return {
        "presence": best(options, key),
        "crash": best(crash_options, key),
        "presence_options": options,
        "crash_options": crash_options,
    }


def calibrate(args, split):
    for fold in range(3):
        _, selection, _ = fold_shots(split, fold)
        sb.guard_partition(split, [], selection, [])
        shots = [Shot(args.work, shot) for shot in selection]
        out = args.work / "calibration" / f"fold_{fold}"
        out.mkdir(parents=True, exist_ok=True)
        result = {
            "fold": fold,
            "selection_shots": selection,
            "selection": SELECTION,
            "derivative": calibrate_derivative(shots, gated=False),
            "derivative_q_gated": calibrate_derivative(shots, gated=True),
        }
        for variant in args.variants:
            raws = {s.shot: load_raw(variant, fold, s.shot) for s in shots}
            result[variant] = {
                "combined": calibrate_model(shots, raws, gated=False, z_grid=Z_GRID),
                "network_only": calibrate_model(shots, raws, gated=False, z_grid=(0,)),
                "combined_q_gated": calibrate_model(
                    shots, raws, gated=True, z_grid=Z_GRID
                ),
            }
            p = result[variant]["combined"]["presence"]
            print(
                f"fold {fold} {variant}: t {p['threshold']} z {p['z']} inner F1 {p['f1']:.4f}",
                flush=True,
            )
        save_json(out / "calibration.json", result)


def select(args):
    folds = [
        json.loads((args.work / f"calibration/fold_{f}/calibration.json").read_text())
        for f in range(3)
    ]
    table = {}
    for variant in args.variants:
        row = {}
        for kind in ("combined", "network_only", "combined_q_gated"):
            cells = np.sum(
                [f[variant][kind]["presence"]["cells"] for f in folds], axis=0
            )
            crash = np.sum([f[variant][kind]["crash"]["cells"] for f in folds], axis=0)
            row[kind] = {
                "presence_cells": cells.tolist(),
                "presence": point_metrics(cells[:3]),
                "crash_cells": crash.tolist(),
                "crash": point_metrics(crash),
                "by_fold": [
                    {
                        k: f[variant][kind]["presence"][k]
                        for k in ("threshold", "z", "f1")
                    }
                    for f in folds
                ],
            }
        table[variant] = row
    derivative = {}
    for kind in ("derivative", "derivative_q_gated"):
        cells = np.sum([f[kind]["presence"]["cells"] for f in folds], axis=0)
        derivative[kind] = {
            "presence_cells": cells.tolist(),
            "presence": point_metrics(cells[:3]),
            "by_fold": [
                {k: f[kind]["presence"][k] for k in ("z", "f1")} for f in folds
            ],
        }
    candidates = [v for v in args.variants if v in VARIANTS]
    winner = max(candidates, key=lambda v: table[v]["combined"]["presence"]["f1"])
    result = {
        "criterion": "pooled inner-selection presence F1 in the no-q = 1 view, "
        "each fold at its own selected operating point; candidates exclude the "
        "q-gated rows (" + GATE_CAVEAT + ")",
        "variants": table,
        "derivative": derivative,
        "winner": winner,
    }
    save_json(args.work / "selection.json", result)
    print(
        json.dumps(
            {v: table[v]["combined"]["presence"]["f1"] for v in table}
            | {"winner": winner}
        ),
        flush=True,
    )


# ---------------------------------------------------------------- evaluation


def entries(args, split):
    """name -> how to predict OOF / held-out shots, with frozen operating points."""
    folds = [
        json.loads((args.work / f"calibration/fold_{f}/calibration.json").read_text())
        for f in range(3)
    ]
    winner = json.loads((args.work / "selection.json").read_text())["winner"]
    fix5 = [
        json.loads((FIX5_WORK / f"models/saw-ours/fold_{f}/complete.json").read_text())
        for f in range(3)
    ]
    out = {"saw-ours": ("model", winner, "combined", False)}
    out["saw-ours-fix5"] = ("fix5", None, None, False)
    for variant in args.variants:
        out[f"saw-ours-v2-{variant}"] = ("model", variant, "combined", False)
    out[f"saw-ours-v2-{winner}-network-only"] = ("model", winner, "network_only", False)
    out[f"saw-ours-v2-{winner}-q-gated"] = ("model", winner, "combined_q_gated", True)
    out["saw-derivative"] = ("derivative-fix5", None, None, False)
    out["saw-derivative-noq1-selected"] = ("derivative", None, "derivative", False)
    out["saw-derivative-noq1-selected-q-gated"] = (
        "derivative",
        None,
        "derivative_q_gated",
        True,
    )
    out["saw-always-present"] = ("always", None, None, False)
    return out, folds, fix5, winner


def operating(kind, variant, key, folds, fix5, fold):
    """(presence threshold, presence z, crash threshold, need presence, crash z)."""
    if kind == "fix5":
        s = fix5[fold]
        return (
            s["presence_threshold"],
            0,
            s["selected_crash_threshold"]["threshold"],
            False,
            None,
        )
    if kind == "derivative-fix5":
        d = fix5[fold]["derivative_baseline"]
        return 0.5, d["presence_z"], None, False, d["z"]
    if kind == "derivative":
        d = folds[fold][key]
        return 0.5, d["presence"]["z"], None, False, d["crash"]["z"]
    if kind == "always":
        return 0.5, 0, None, False, None
    c = folds[fold][variant][key]
    return (
        c["presence"]["threshold"],
        c["presence"]["z"],
        c["crash"]["threshold"],
        c["crash"]["need_presence"],
        None,
    )


def predict(kind, variant, gated, ops, s, raws):
    """A scoring-ready prediction from per-fold raws (one fold OOF, three for an
    ensemble) and per-fold operating points (means for an ensemble)."""
    signal = s.signal
    t = signal["t"]
    no_q1 = s.no_q1 if gated else None
    base = {k: signal[k] for k in ("t", "observable", "assessed")}
    if kind == "always":
        return {
            **base,
            "presence": np.ones(len(t), np.float32),
            "picks": np.array([]),
        }, 0.5
    if kind.startswith("derivative"):
        presence_z = float(np.mean([o[1] for o in ops]))
        crash_z = float(np.mean([o[4] for o in ops]))
        presence = presence_score(
            {"presence": np.ones(len(t), np.float32)}, s.edge, presence_z, no_q1
        )
        return {
            **base,
            "presence": presence,
            "picks": derivative_picks(signal, s.times, s.amps, crash_z, no_q1),
        }, 0.5
    presences = [
        presence_score(raw, s.edge, op[1], no_q1)
        for raw, op in zip(raws, ops, strict=True)
    ]
    presence = np.mean(presences, axis=0).astype(np.float32)
    crash = np.mean([raw["crash"] for raw in raws], axis=0)
    threshold = float(np.mean([o[0] for o in ops]))
    crash_threshold = float(np.mean([o[2] for o in ops]))
    need = sum(o[3] for o in ops) * 2 > len(ops)
    picks = model_picks(
        signal, crash, crash_threshold, presence if need else None, threshold, no_q1
    )
    return {**base, "presence": presence, "picks": picks}, threshold


def oof(name, spec, s, split, folds, fix5):
    kind, variant, key, gated = spec
    fold = split["folds"][s.shot]
    ops = [operating(kind, variant, key, folds, fix5, fold)]
    raws = []
    if kind == "fix5":
        with np.load(FIX5_WORK / f"predictions/saw-ours/fold_{fold}/{s.shot}.npz") as d:
            pred = {k: d[k] for k in d.files}
        return pred, ops[0][0]
    if kind == "derivative-fix5":
        d = fix5[fold]["derivative_baseline"]
        return sb.baseline_prediction(
            "saw-derivative", s.signal, d["z"], d["presence_z"]
        ), 0.5
    if kind == "always":
        return sb.baseline_prediction("saw-always-present", s.signal, 0), 0.5
    if kind == "model":
        raws = [load_raw(variant, fold, s.shot)]
    return predict(kind, variant, gated, ops, s, raws)


def heldout(name, spec, s, folds, fix5, work):
    kind, variant, key, gated = spec
    if kind == "fix5":
        prediction, _, threshold = sb.heldout_prediction(
            FIX5_WORK, "saw-ours", s.shot, fix5
        )
        return prediction, threshold
    if kind == "derivative-fix5":
        prediction, _, threshold = sb.heldout_prediction(
            FIX5_WORK, "saw-derivative", s.shot, fix5
        )
        return prediction, threshold
    ops = [operating(kind, variant, key, folds, fix5, f) for f in range(3)]
    raws = [load_raw(variant, f, s.shot) for f in range(3)] if kind == "model" else []
    return predict(kind, variant, gated, ops, s, raws)


def score_entry(name, spec, args, split, folds, fix5, cache):
    rows = {
        v: {1: [], 2: []}
        for v in (
            "primary",
            "old_negatives_sensitivity",
            "no_q1_negatives",
            *[f"edge_{c:g}_ms" for c in sb.EDGE_CONTEXTS_MS],
        )
    }
    by_shot = {v: [] for v in rows}
    fixed_rows, fold_rows, gates = {1: [], 2: []}, {f: [] for f in range(3)}, []
    for shot in split["training_cohort"]:
        s = cache(shot)
        pred, threshold = oof(name, spec, s, split, folds, fix5)
        for v, signal in s.views().items():
            shot_rows, result = sb.score_prediction(
                shot, name, pred, signal, s.rec, threshold
            )
            result["fold"] = split["folds"][shot]
            for tol in (1, 2):
                rows[v][tol].append(shot_rows[tol])
            by_shot[v].append(result)
            if v == "primary":
                primary_rows = shot_rows
        fixed, _ = sb.score_prediction(
            shot, name, pred, s.signal, s.rec, sb.FIXED_PRESENCE_THRESHOLD
        )
        for tol in (1, 2):
            fixed_rows[tol].append(fixed[tol])
        fold_rows[split["folds"][shot]].append((primary_rows[2], fixed[2]))
        gates.append(sb.gate_record(shot, s.gate, s.signal))
    score = {
        f"crash_tolerance_{tol}ms": aggregate(rows["primary"][tol]) for tol in (1, 2)
    }
    by_primary = by_shot["primary"] + split["unavailable_training"]
    by_primary.sort(key=lambda r: r["shot"])
    score["assessment_totals"] = sb.assessment_totals(by_primary)
    score["coverage"] = sb.coverage(by_primary)
    score["old_negatives_sensitivity"] = {
        "definition": sb.OLD_NEGATIVES,
        **{
            f"crash_tolerance_{tol}ms": aggregate(
                rows["old_negatives_sensitivity"][tol]
            )
            for tol in (1, 2)
        },
        "assessment_totals": sb.assessment_totals(by_shot["old_negatives_sensitivity"]),
    }
    score["no_q1_negatives"] = {
        "definition": sb.NO_Q1_NEGATIVES,
        **{
            f"crash_tolerance_{tol}ms": aggregate(rows["no_q1_negatives"][tol])
            for tol in (1, 2)
        },
        "assessment_totals": sb.assessment_totals(by_shot["no_q1_negatives"]),
        "gate_shots": sb.gate_totals(gates),
        "added_seconds_by_tree": sb.gate_seconds(gates),
        "equilibria": sb.gate_equilibria(gates),
    }
    score["edge_context_sensitivity"] = {
        "definition": sb.EDGE_CONTEXT_SENSITIVITY,
        "contexts_ms": list(sb.EDGE_CONTEXTS_MS),
        **{
            f"{c:g}_ms": {
                "isolated_edge_context_ms": c,
                **{
                    f"crash_tolerance_{tol}ms": aggregate(rows[f"edge_{c:g}_ms"][tol])
                    for tol in (1, 2)
                },
                "assessment_totals": sb.assessment_totals(by_shot[f"edge_{c:g}_ms"]),
                "negative_bins": sum(
                    r["presence"]["negative_bins"]
                    for r in by_shot[f"edge_{c:g}_ms"]
                    if r["presence"] is not None
                ),
                "positive_bins": sum(
                    r["presence"]["positive_bins"]
                    for r in by_shot[f"edge_{c:g}_ms"]
                    if r["presence"] is not None
                ),
            }
            for c in sb.EDGE_CONTEXTS_MS
        },
    }
    kind = spec[0]
    summaries = [
        {"presence_threshold": operating(kind, spec[1], spec[2], folds, fix5, f)[0]}
        for f in range(3)
    ]
    score["presence_fixed_threshold"] = sb.presence_fixed_threshold(
        fold_rows,
        fixed_rows[2],
        summaries,
        "saw-derivative" if kind.startswith("derivative") or kind == "always" else name,
    )
    score["operating_points"] = {
        "by_fold": [
            dict(
                zip(
                    (
                        "presence_threshold",
                        "presence_z",
                        "crash_threshold",
                        "crash_picks_need_presence",
                        "crash_z",
                    ),
                    operating(kind, spec[1], spec[2], folds, fix5, f),
                    strict=True,
                ),
                fold=f,
            )
            for f in range(3)
        ],
    }
    score["by_shot"] = by_primary
    # Fixed validation: the frozen three-fold ensemble, means of inner selections.
    fixed = {
        v: {1: [], 2: []}
        for v in ("primary", "old_negatives_sensitivity", "no_q1_negatives")
    }
    fixed_by_shot = {v: [] for v in fixed}
    fixed_gates = []
    for shot in split["fixed_validation_nonexpert"]:
        if shot not in split["fixed_validation_supported"]:
            fixed_by_shot["primary"].append(sb.unavailable_record(args.work, shot))
            continue
        s = cache(shot)
        pred, threshold = heldout(name, spec, s, folds, fix5, args.work)
        for v, signal in list(s.views().items())[:3]:
            shot_rows, result = sb.score_prediction(
                shot, name, pred, signal, s.rec, threshold
            )
            fixed_by_shot[v].append(result)
            for tol in (1, 2):
                fixed[v][tol].append(shot_rows[tol])
        fixed_gates.append(sb.gate_record(shot, s.gate, s.signal))
    score["fixed_validation"] = {
        **{
            f"crash_tolerance_{tol}ms": aggregate(fixed["primary"][tol])
            for tol in (1, 2)
        },
        "assessment_totals": sb.assessment_totals(fixed_by_shot["primary"]),
        "coverage": sb.coverage(fixed_by_shot["primary"]),
        "old_negatives_sensitivity": {
            "definition": sb.OLD_NEGATIVES,
            **{
                f"crash_tolerance_{tol}ms": aggregate(
                    fixed["old_negatives_sensitivity"][tol]
                )
                for tol in (1, 2)
            },
            "assessment_totals": sb.assessment_totals(
                fixed_by_shot["old_negatives_sensitivity"]
            ),
        },
        "no_q1_negatives": {
            "definition": sb.NO_Q1_NEGATIVES,
            **{
                f"crash_tolerance_{tol}ms": aggregate(fixed["no_q1_negatives"][tol])
                for tol in (1, 2)
            },
            "assessment_totals": sb.assessment_totals(fixed_by_shot["no_q1_negatives"]),
            "gate_shots": sb.gate_totals(fixed_gates),
            "added_seconds_by_tree": sb.gate_seconds(fixed_gates),
            "equilibria": sb.gate_equilibria(fixed_gates),
        },
        "score_label": "conditional agreement with the physics rule on assessed bins",
        "protocol": "second held-out set: nonexpert fixed validation; three frozen fold "
        "ensembles (fold presence scores averaged); means of inner-selected operating points",
    }
    score["expert"] = expert(name, spec, args, folds, fix5, split, cache)
    paired = {
        "out_of_fold": {v: rows[v] for v in ("primary", "no_q1_negatives")},
        "fixed_validation": {v: fixed[v] for v in ("primary", "no_q1_negatives")},
    }
    return score, paired


def expert(name, spec, args, folds, fix5, split, cache):
    """The fix5 expert block: reviewed spans, exploratory, no bootstrap."""
    review = pd.read_csv(REVIEW)
    hists, cells_all, rows = [], [], []
    threshold = None
    for shot in split["expert_shots"]:
        s = cache(shot)
        pred, threshold = heldout(name, spec, s, folds, fix5, args.work)
        recs = review[review.shot == shot]
        positive = [
            (r.t_start / 1000, r.t_end / 1000)
            for r in recs.itertuples()
            if r.category == 1
        ]
        known = [
            (r.t_start / 1000, r.t_end / 1000)
            for r in recs.itertuples()
            if r.category in (0, 1)
        ]
        t = pred["t"]
        bins = bin_times(
            (max(t[0], recs.t_start.min() / 1000), min(t[-1], recs.t_end.max() / 1000))
        )
        observable, _ = sb.masks_at(s.signal, bins)
        valid = spans_at(bins, known) & observable
        truth = spans_at(bins, positive)[valid]
        probability = np.interp(bins, t, pred["presence"])[valid]
        hist = score_histogram(truth, probability)
        cells = binary_cells(truth, probability >= threshold)
        hists.append(hist)
        cells_all.append(cells)
        rows.append(
            {
                "shot": shot,
                "presence": presence_from_cells(hist, cells, threshold),
                "presence_cells": cells.astype(int).tolist(),
                "assessed_bins": int(valid.sum()),
            }
        )
    return {
        "shots": len(rows),
        "by_shot": rows,
        "presence": presence_from_cells(
            np.sum(hists, axis=0), np.sum(cells_all, axis=0), threshold
        ),
        "presence_threshold": threshold,
        "note": "three-fold ensemble on the reviewed spans (review/labels.csv as regridded "
        "to 10 ms after the fix5 run); exploratory, anchored to prior rule suggestions",
    }


def evaluate(args):
    split = split_manifest()
    named, folds, fix5, winner = entries(args, split)
    shots = {}

    def cache(shot):
        if shot not in shots:
            shots[shot] = Shot(args.work, shot)
        return shots[shot]

    mismatch = 0
    for shot in split["training_cohort"]:
        s = cache(shot)
        z = fix5[split["folds"][shot]]["derivative_baseline"]["presence_z"]
        reference = sb.derivative_prediction(s.signal, z)["presence"]
        mismatch += int(((s.edge >= z) != (reference >= 0.5)).sum())
    print(
        f"edge score vs derivative presence: {mismatch} native samples differ",
        flush=True,
    )
    results, paired_rows = {}, {}
    for name, spec in named.items():
        begun = time.monotonic()
        results[name], paired_rows[name] = score_entry(
            name, spec, args, split, folds, fix5, cache
        )
        print(
            f"scored {name} in {time.monotonic() - begun:.0f} s: no-q1 F1 "
            f"{results[name]['no_q1_negatives']['crash_tolerance_2ms']['presence']['f1']:.4f}",
            flush=True,
        )
    comparators = ("saw-ours-fix5", "saw-derivative", "saw-derivative-noq1-selected")
    for name, result in results.items():
        for group in ("out_of_fold", "fixed_validation"):
            target = result if group == "out_of_fold" else result["fixed_validation"]
            target["paired"] = {}
            for v in ("primary", "no_q1_negatives"):
                target["paired"][v] = {
                    other: {
                        f"crash_tolerance_{tol}ms": paired_bootstrap(
                            paired_rows[name][group][v][tol],
                            paired_rows[other][group][v][tol],
                            direction=f"{name} minus {other}",
                        )
                        for tol in (1, 2)
                    }
                    for other in comparators
                    if other != name
                    and (name == "saw-ours" or other == "saw-derivative")
                }
    installed = json.loads((FIX5_OUTPUT / "benchmark.json").read_text())
    output = {
        "legacy": installed["legacy"],  # the HL-3 paper's stated values, as fix5
        "protocol": {
            **installed["protocol"],
            "fix6": {
                "model": "saw-ours (fix6) is saw-ours v2: "
                + DESCRIPTION[winner]
                + "; presence is ANDed with the derivative edge score at the inner-selected z",
                "winner": winner,
                "variants": {v: DESCRIPTION[v] for v in args.variants},
                "selection": SELECTION,
                "winner_selection": "pooled inner-selection no-q = 1 presence F1; q-gated rows excluded",
                "interval_threshold": "selected per fold on inner selection shots in the "
                "no-q = 1 view (fix5 selected on the primary view); same grid plus z in "
                f"{list(Z_GRID)}",
                "circularity": CIRCULARITY,
                "q_gate": GATE_CAVEAT,
                "unchanged": "folds, inner splits, labels (saw_physics v4 rows behind the "
                "fix5 shot records, crash v3), gate, bins, scoring functions and bootstrap "
                "(sawtooth_benchmark.score_prediction / aggregate / paired_bootstrap)",
                "saw-ours-fix5": "fix5 saw-ours, rescored here from its stored predictions "
                "and thresholds; saw-ours-fix5, saw-derivative and saw-always-present equal "
                "outputs/labeler/sawtooth/fix5 in every scored number except the expert "
                "block (review spans regridded since); the no-q = 1 equilibria list one "
                "more EFIT02ER file than fix5 (fetched since), which changes no gate "
                "decision on the scored bins",
                "saw-derivative": "the fix5 derivative picker (presence_z on primary-view "
                "inner F1), a component of saw-ours and an ablation, no longer drawn alone",
                "saw-derivative-noq1-selected": "the same picker with presence and crash z "
                "selected on inner no-q = 1 F1, the derivative alone under saw-ours' selection",
                "copied_from_fix5": "saw-hl3-full and saw-hl3-ece are copied verbatim from "
                "outputs/labeler/sawtooth/fix5/benchmark.json (not rescored)",
                "work": str(args.work),
                "edge_score_check": f"{mismatch} OOF native samples where edge score >= "
                "presence_z differs from the fix5 derivative presence",
            },
        },
        "selection": json.loads((args.work / "selection.json").read_text()),
        "Tokamak-SI": {},
    }
    detail = args.work / "benchmark_detail.json"
    for name, result in results.items():
        output["Tokamak-SI"][name] = {
            **{k: v for k, v in result.items() if k != "by_shot"},
            "by_shot_count": len(result["by_shot"]),
            "out_of_fold_shots": result["coverage"]["supported_shot_ids"],
            "by_shot_details": str(detail),
        }
    for name in ("saw-hl3-full", "saw-hl3-ece"):
        output["Tokamak-SI"][name] = {
            **installed["Tokamak-SI"][name],
            "provenance": "copied from fix5 benchmark.json",
        }
    save_json(detail, {"Tokamak-SI": results})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT / "benchmark.json", output)
    for variant in args.variants:
        for fold in range(3):
            fit_json = args.work / "models" / variant / f"fold_{fold}" / "fit.json"
            record = {
                "variant": variant,
                "fold": fold,
                "calibration": {
                    k: folds[fold][variant][k]["presence"]
                    | {"crash": folds[fold][variant][k]["crash"]}
                    for k in ("combined", "network_only", "combined_q_gated")
                },
            }
            if fit_json.exists():
                record["fit"] = json.loads(fit_json.read_text())
            save_json(OUTPUT / f"saw-ours-v2-{variant}_fold_{fold}.json", record)


# ---------------------------------------------------------------- main


def setup(args):
    args.work.mkdir(parents=True, exist_ok=True)
    for name in ("shots", "signals", "freeze.json"):
        link = args.work / name
        if not link.exists():
            link.symlink_to((FIX5_WORK / name).resolve())
    save_json(
        args.work / "split_manifest.json",
        {**split_manifest(), "source": str(FIX5_WORK / "split_manifest.json")},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["setup", "fit", "infer", "calibrate", "select", "evaluate"]
    )
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2]
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    if args.stage == "setup":
        return setup(args)
    split = split_manifest()
    if args.stage in ("fit", "infer"):
        torch.set_num_threads(args.threads)
        # The labeler environment's torch is CPU-only; the picker is small enough
        # for CPU fitting (fix5 fitted it on a V100 through another environment).
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.28)
        for variant in args.variants:
            for fold in args.folds:
                if args.stage == "fit" and VARIANTS[variant]["train"]:
                    fit(args, variant, fold, split, device)
                elif args.stage == "infer":
                    run_inference(args, variant, fold, split, device)
        return None
    return {"calibrate": calibrate, "select": select, "evaluate": evaluate}[args.stage](
        *((args, split) if args.stage == "calibrate" else (args,))
    )


if __name__ == "__main__":
    main()
