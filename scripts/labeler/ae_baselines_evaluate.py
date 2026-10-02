#!/usr/bin/env python
"""Benchmark the older CO2 Alfven-eigenmode detectors against the SELDnet.

The older detectors are Alvin Garcia's (UCI): a Keras LSTM (3 x LSTM(64), dropout
0.5, 3 x Dense(128), five sigmoid outputs eae/tae/rsae/bae/lfm) and a PyRCN
two-layer echo state network ("RCN"). Each comes on two inputs, one CO2 chord's
spectrogram (``spec``) or the cross-power of a chord pair (``xpow``), 20-250 kHz.
None is run here: the LSTM needs TensorFlow 2.1 and the ESNs scikit-learn 0.23
pickles, and no environment of ours has either. What is read is what they saved,
their predictions on their own validation shots (``results_dp/{spec,xpow}/``), the
only shots they did not train on.

**Shots.** The SELDnet's 60 validation shots that are also in that validation set
(19) are the fair set: neither detector trained on them. The other 41 were in the
older detectors' training set, and no predictions were saved for them.

**Frames.** The owner's 10 ms frames of 0-2 s (``labeler.scoring.frames``). The
truth is the owner-reviewed AE table (``review/labels.csv``, the snapshot the
ae_xpower v2 test used), frames called present or absent; a second truth, the
Heidbrink hand annotation (``annotated``, which under-counts: an unannotated frame
is not an absent one), is reported beside it. Each detector's output is put on that
grid: the SELDnet's 0.256 ms frame probabilities are averaged in each frame (hard
call: half the columns >= 0.5, as in the ae_xpower test); an older detector's value
is that of the bin nearest the frame's centre (RCN bins are 14.1 ms, LSTM bins
70.2-70.8 ms). AE activity of an older detector is its largest output over eae, tae,
rsae and bae (LFM is not AE here, as in ours).

**Scores.** AUROC and AUPRC of the continuous output, frame precision, recall and F1
at the detector's own operating point (SELDnet 0.5; RCN 0.10 and LSTM 0.15, the
thresholds in Alvin's notebooks), and the best F1 over all thresholds on these
frames (optimistic, tuned on the test frames). 95 % intervals are bootstrapped over
shots (2000 replicates). An older detector is scored on each of a shot's inputs (4
chords or 10 chord pairs, each a sample, as their own evaluation does: ``each``) and
with those averaged (``mean``).

    python scripts/labeler/ae_baselines_evaluate.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
for extra in (str(REPO / "src"), str(Path(__file__).resolve().parent)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from ae_train import DEFAULT_DATASET

from labeler.ae.xpower.data import targets
from labeler.events.catalog.states import ABSENT, PRESENT
from labeler.events.review import labels as review_labels
from labeler.scoring.frames import FRAME_MS

LABELER = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
DF2 = Path("/projects/EKOLEMEN/agarcia/df2")
REVIEW = LABELER / "models/ae_xpower/v2/review/labels.csv"
SELDNET_PROBS = LABELER / "benchmarks/ae/seldnet_threeway_sce_valid_probs.npz"
DEFAULT_OUT = REPO / "outputs/labeler/ae/baselines"
RECORD_MS = 2000.0
SELD_FRAMES = 7820
N_FRAMES = int(RECORD_MS / FRAME_MS)
VARIANTS = ("spec", "xpow")
MODELS = ("lstm", "rcn")
#: Their output order is eae, tae, rsae, bae, lfm; the first four are AE.
AE_CLASSES = slice(0, 4)
THRESHOLDS = {"seldnet": 0.5, "rcn": 0.10, "lstm": 0.15}
REPLICATES = 2000
SEED = 20260923
KEYS = ("auroc", "auprc", "precision", "recall", "f1", "best_f1")


def auroc(score: np.ndarray, truth: np.ndarray) -> float:
    """Rank AUROC with ties averaged; NaN if one class is missing."""
    pos = int(truth.sum())
    neg = truth.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return float((rankdata(score)[truth].sum() - pos * (pos + 1) / 2) / (pos * neg))


def pr_curve(score: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall at every distinct threshold of `score`."""
    order = np.argsort(-score, kind="mergesort")
    hit = truth[order].astype(float)
    ranked = score[order]
    last = np.r_[ranked[1:] != ranked[:-1], True]
    tp = np.cumsum(hit)[last]
    taken = (np.arange(hit.size) + 1)[last]
    return tp / taken, tp / max(hit.sum(), 1.0)


def metrics(score: np.ndarray, hard: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """AUROC, AUPRC (average precision), P/R/F1 of `hard`, and the best F1 over thresholds."""
    tp = int((hard & truth).sum())
    precision = tp / hard.sum() if hard.sum() else float("nan")
    recall = tp / truth.sum() if truth.sum() else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    out = {"auroc": auroc(score, truth), "precision": precision, "recall": recall, "f1": f1}
    if truth.any():
        p, r = pr_curve(score, truth)
        out["auprc"] = float(np.sum(np.diff(np.r_[0.0, r]) * p))
        out["best_f1"] = float((2 * p * r / np.maximum(p + r, 1e-12)).max())
    else:
        out["auprc"] = out["best_f1"] = float("nan")
    return {k: float(out[k]) for k in KEYS}


def bootstrap(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, list]:
    """95 % intervals of `metrics` over shots; `parts` is one (score, hard, truth) per shot."""
    rng = np.random.default_rng(SEED)
    draws: dict[str, list[float]] = {k: [] for k in KEYS}
    for _ in range(REPLICATES):
        pick = rng.integers(0, len(parts), len(parts))
        got = metrics(*(np.concatenate([parts[i][j] for i in pick]) for j in range(3)))
        for k in KEYS:
            draws[k].append(got[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def frame_index(n: int) -> np.ndarray:
    """The 10 ms frame each of `n` SELDnet columns (0.256 ms apart) falls in."""
    centre = (np.arange(n) + 0.5) * RECORD_MS / SELD_FRAMES
    return np.minimum((centre // FRAME_MS).astype(int), N_FRAMES - 1)


def frame_mean(values: np.ndarray) -> np.ndarray:
    k = frame_index(values.size)
    return np.bincount(k, weights=values, minlength=N_FRAMES) / np.maximum(np.bincount(k, minlength=N_FRAMES), 1)


def nearest_bin(t_bins: np.ndarray) -> np.ndarray:
    """For each 10 ms frame the index of the detector bin whose centre is nearest."""
    centre = (np.arange(N_FRAMES) + 0.5) * FRAME_MS
    return np.abs(centre[:, None] - t_bins[None, :]).argmin(axis=1)


def load_older(df2: Path) -> dict:
    """Saved validation predictions of both detectors on both inputs, and each shot's rows."""
    names_spec = [str(n) for n in np.load(df2 / "shots_val_7525.npy", allow_pickle=True)]
    pairs = np.load(df2 / "xpow_shots_chords.npy", allow_pickle=True)
    perm = np.load(df2 / "irand_7525_xpow.npy")
    # The cross-power rows follow the last len(rows) of that permutation (checked below).
    time_rcn = {
        "spec": np.load(df2 / "spec_dp_8_4/spec_time.npy"),
        "xpow": np.load(df2 / "xpow_time_8_4.txt", allow_pickle=True),  # an .npy, named .txt
    }
    out: dict = {"trained_on": {int(str(n)[:6]) for n in np.load(df2 / "shots_train_7525.npy", allow_pickle=True)}}
    for variant in VARIANTS:
        base = df2 / "results_dp" / variant
        lstm = np.load(base / "lstm/y_pred_rebin.npy")
        rcn = np.load(base / "rcn/y_pred.npy")
        names = names_spec if variant == "spec" else [str(n) for n in pairs[perm[len(pairs) - len(lstm):]]]
        rows: dict[int, list[int]] = {}
        for i, name in enumerate(names):
            rows.setdefault(int(name[:6]), []).append(i)
        out[variant] = {
            "rows": rows,
            "lstm": lstm,
            "lstm_t": np.load(base / "lstm/time_rebin.npy"),
            "rcn": rcn,
            # the cross-power ESN writes 141 bins of the 142; the lag scan puts bin i on bin i
            "rcn_t": time_rcn[variant][: rcn.shape[1]],
            "truth": np.load(base / "lstm/y_val_rebin.npy"),
        }
    return out


def check_order(v: dict) -> dict[str, float]:
    """Check that the saved arrays follow the order the rows were named in.

    The LSTM's saved truth is the same on every row of a shot (labels are per shot), so
    it is wrong if the names group rows wrongly. The RCN's output should agree with the
    LSTM's more on the same row than on another shot's row.
    """
    spread = max(float(np.abs(v["truth"][r] - v["truth"][r[0]]).max()) for r in v["rows"].values())
    lstm = v["lstm"]
    rcn = np.zeros_like(lstm)
    for k, c in enumerate(v["lstm_t"]):
        rcn[:, k, :] = v["rcn"][:, np.abs(v["rcn_t"] - c) <= 35.0, :].mean(axis=1)

    def unit(a: np.ndarray) -> np.ndarray:
        a = a.reshape(len(a), -1)
        a = a - a.mean(axis=1, keepdims=True)
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

    p, q = unit(lstm), unit(rcn)
    shuffled = np.random.default_rng(SEED).permutation(len(q))
    return {
        "lstm_truth_spread_within_shot": spread,
        "rcn_vs_lstm_r_same_row": float((p * q).sum(axis=1).mean()),
        "rcn_vs_lstm_r_shuffled_rows": float((p * q[shuffled]).sum(axis=1).mean()),
    }


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def table(block: dict, names: dict[str, str]) -> list[str]:
    def cell(m: dict, key: str) -> str:
        ci = (m.get("ci95") or {}).get(key)
        return f"{m[key]:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")

    lines = [
        "| Method | Samples | AUROC | AUPRC | Precision | Recall | F1 | Best F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for method, m in block.items():
        lines.append(
            f"| {names.get(method, method)} | {m['n_samples']} | "
            + " | ".join(cell(m, k) for k in ("auroc", "auprc", "precision", "recall", "f1"))
            + f" | {m['best_f1']:.3f} |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--review", type=Path, default=REVIEW)
    ap.add_argument("--seldnet-probs", type=Path, default=SELDNET_PROBS)
    ap.add_argument("--df2", type=Path, default=DF2)
    args = ap.parse_args(argv)

    older = load_older(args.df2)
    owner = review_labels.read_labels(args.review)
    probs = np.load(args.seldnet_probs)
    valid: list[int] = []
    annotated: dict[int, np.ndarray] = {}
    for path in sorted(args.dataset_dir.glob("*.npz")):
        with np.load(path) as z:
            if str(z["split"]) == "valid":
                valid.append(int(path.stem.split("_")[0]))
                annotated[valid[-1]] = z["annotated"].astype(float)
    if set(older["spec"]["rows"]) != set(older["xpow"]["rows"]):
        raise SystemExit("the spec and xpow validation sets differ")
    clean = [s for s in valid if s in older["spec"]["rows"]]
    leaked = [s for s in valid if s in older["trained_on"]]
    print(f"{len(valid)} valid shots: {len(clean)} fair, {len(leaked)} in the older detectors' training set")
    order = {v: check_order(older[v]) for v in VARIANTS}
    print("order check", json.dumps(order, indent=1))

    bins = {(v, m): nearest_bin(older[v][f"{m}_t"]) for v in VARIANTS for m in MODELS}
    seld = {s: (frame_mean(probs[f"{s}_valid"].astype(np.float64)), frame_mean((probs[f"{s}_valid"] >= 0.5).astype(float))) for s in valid}
    truths: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for shot in valid:
        state = targets(owner[shot], 0, N_FRAMES)
        truths[shot] = {
            "reviewed": (np.isin(state, (ABSENT, PRESENT)), state == PRESENT),
            "annotated": (np.ones(N_FRAMES, bool), frame_mean(annotated[shot]) >= 0.5),
        }

    def outputs(method: str, shot: int) -> tuple[np.ndarray, np.ndarray]:
        """(score, hard), each (samples, 200 frames), of `method` on `shot`."""
        if method == "always":
            return np.ones((1, N_FRAMES)), np.ones((1, N_FRAMES), bool)
        if method == "seldnet":
            mean, share = seld[shot]
            return mean[None, :], (share >= 0.5)[None, :]
        model, variant, mode = method.split("_")
        v = older[variant]
        picks = v[model][v["rows"][shot]][:, :, AE_CLASSES].max(axis=2)
        score = (picks if mode == "each" else picks.mean(axis=0, keepdims=True))[:, bins[variant, model]]
        return score, score >= THRESHOLDS[model]

    def parts_of(method: str, shots: list[int], truth_name: str):
        parts = []
        for shot in shots:
            scored, truth = truths[shot][truth_name]
            score, hard = outputs(method, shot)
            parts.append((score[:, scored].ravel(), hard[:, scored].ravel(), np.tile(truth[scored], len(score))))
        return parts

    methods = ["seldnet"] + [f"{m}_{v}_{mode}" for v in VARIANTS for m in MODELS for mode in ("each", "mean")] + ["always"]
    result: dict = {}
    for truth_name in ("reviewed", "annotated"):
        block: dict = {}
        for method in methods:
            parts = parts_of(method, clean, truth_name)
            score, hard, truth = (np.concatenate([p[j] for p in parts]) for j in range(3))
            block[method] = {
                **metrics(score, hard, truth),
                "ci95": None if method == "always" else bootstrap(parts),
                "n_samples": int(truth.size),
                "truth_fraction": float(truth.mean()),
            }
        # the SELDnet on all 60 validation shots, for reference (41 of them are older-train shots)
        score, hard, truth = (np.concatenate([p[j] for p in parts_of("seldnet", valid, truth_name)]) for j in range(3))
        block["seldnet_all_60"] = {**metrics(score, hard, truth), "n_samples": int(truth.size), "truth_fraction": float(truth.mean())}
        result[truth_name] = block

    args.out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_sha(),
        "shots": {"fair": clean, "older_trained_on": leaked, "valid": len(valid)},
        "thresholds": THRESHOLDS,
        "order_check": order,
        "frame_ms": FRAME_MS,
        "bootstrap": {"replicates": REPLICATES, "seed": SEED},
        "inputs": {
            "review": str(args.review),
            "seldnet_probs": str(args.seldnet_probs),
            "older_predictions": str(args.df2 / "results_dp"),
        },
        "results": result,
    }
    (args.out_dir / "evaluation.json").write_text(json.dumps(record, indent=1) + "\n")

    names = {
        "seldnet": "SELDnet (d3d_ae_activity_seldnet)",
        "always": "every frame present",
        "seldnet_all_60": "SELDnet, all 60 validation shots (reference)",
    }
    for v, what in (("spec", "chord"), ("xpow", "chord pair")):
        for m, label in (("lstm", "LSTM"), ("rcn", "RCN")):
            names[f"{m}_{v}_each"] = f"{label} {v}, each {what}"
            names[f"{m}_{v}_mean"] = f"{label} {v}, {what}s averaged"
    for truth_name, block in result.items():
        print(f"\n== truth: {truth_name}")
        for line in table(block, names):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
