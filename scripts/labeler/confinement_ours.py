#!/usr/bin/env python
"""confine-ours: a 1D U-Net labelling L / H / QH / WPQH from 0D signals.

Reads the 1 ms grids of ``labeler.confinement.zerod`` (D-alpha, line density, beta_N,
W_MHD, injected power), trains on the curated intervals outside the cohort's blind test
split with shot-grouped 5-fold cross-validation, scores every curated shot on a model
that never saw it, and applies the five models to the roster.

Stages (``--help`` of each)::

    build     the 1 ms grids of the roster and curated shots (CPU)
    train     5 folds on the GPU: models, and the predictions of the held-out shots
    score     per-bin and per-shot (segmental F1, edit) scores, and the comparison with
              confine-cnn on the shots both saw
              -> outputs/labeler/confinement/ours/scores.json
    apply     segment every roster shot
              -> data/events/confinement/extend_confine_ours/

Large files (grids, models, probabilities) live under ``$LABELER_ROOT/round4/conf``;
only the small CSV and JSON records are written into the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.confinement import bes_protocol as bp
from labeler.confinement import bes_windows as bw
from labeler.confinement import scoring, segments, zerod

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
WORK = LABELER / "round4/conf"
RAW = LABELER / "raw"
ZEROD = WORK / "zerod"
GRIDS = WORK / "zerod_grid"
#: Models and held-out probabilities by the grouping that dealt the folds: whole run
#: days (the headline) or single shots (the first version's folds, kept for the
#: comparison).
OURS_DIRS = {"run_day": WORK / "ours_runday", "shot": WORK / "ours"}
SCORE_FILES = {"run_day": "scores.json", "shot": "scores_random_shots.json"}
DATES = WORK / "dates.csv"
CORPUS_BES = WORK / "bes500k"
DALPHA_TABLE = LABELER / "suggestions/dalpha_lh/v1/confinement_suggest_dalpha_lh_v1.csv"
OUT = REPO / "outputs/labeler/confinement/ours"
EXTEND = REPO / "data/events/confinement/extend_confine_ours"
ROSTER = Path(
    os.environ.get(
        "CONFINEMENT_ROSTER",
        "/scratch/gpfs/nc1514/FusionAIHub/data/events/confinement/shots.csv",
    )
)
SEED = 20261001
FOLDS = 5
REPLICATES = 1000
#: Class index (L, H, QH, WP) to the review's category (1 high, 2 low, 3 qh, 4 wpqh).
CATEGORY = {0: 2, 1: 1, 2: 3, 3: 4}
MODE_BINS = 21
MIN_SEGMENT_MS = 20
BEAM_ON_W = 2e5
#: Gaps in the beam-on mask up to this long are bridged: a modulated beam (blips of
#: 10 ms) leaves the plasma in a regime the whole time.
BEAM_GAP_MS = 50
#: A roster segment whose mean class probability is below this floor is category 5
#: (uncertain), whatever class it was.
CONFIDENCE_FLOOR = 0.7
UNCERTAIN = 5
#: Categories whose labels on shots outside the curated set go to the unreviewed tier:
#: the curated set excludes ELM-free and other quiescent H-modes that the network
#: could read as QH or WPQH.
UNREVIEWED_CATEGORIES = (3, 4)


def wanted_shots() -> list[int]:
    curated = set(bw.curated_intervals().shot)
    return sorted(int(s) for s in set(pd.read_csv(ROSTER).shot) | curated)


def _build_one(shot: int, force: bool) -> dict:
    out = GRIDS / f"{shot}.npz"
    if out.exists() and not force:
        return {"shot": shot, "status": "exists"}
    result = zerod.assemble(shot, RAW, ZEROD)
    if result is None:
        return {"shot": shot, "status": "no D-alpha"}
    zerod.save_grid(out, result)
    return {
        "shot": shot,
        "status": "built",
        "lacks": result["lacks"],
        "bins": result["grid"].shape[1],
    }


def build(args: argparse.Namespace) -> None:
    GRIDS.mkdir(parents=True, exist_ok=True)
    shots = wanted_shots()
    if args.fetched_only:
        shots = [s for s in shots if (ZEROD / f"{s}.npz").exists()]
    if args.limit:
        shots = shots[: args.limit]
    with ProcessPoolExecutor(args.workers) as pool:
        log = list(pool.map(_build_one, shots, [args.force] * len(shots)))
    frame = pd.DataFrame(log)
    print(frame.status.value_counts().to_string())
    lacks = frame[frame.status == "built"].lacks.explode().value_counts()
    print(
        "channels missing on built shots:\n",
        lacks.to_string() if len(lacks) else "none",
    )
    (WORK / "logs").mkdir(parents=True, exist_ok=True)
    frame.to_csv(WORK / "logs/zerod_grid_build.csv", index=False)


def load_curated() -> tuple[
    list[int], list[np.ndarray], list[np.ndarray], pd.DataFrame
]:
    """Shots with a grid and labelled bins: their ids, network inputs and per-bin
    labels."""
    intervals = bw.curated_intervals()
    ids, xs, ys = [], [], []
    for shot, g in intervals.groupby("shot"):
        path = GRIDS / f"{shot}.npz"
        if not path.exists():
            continue
        grid, _ = zerod.load_grid(path)
        y = zerod.interval_labels(g, grid.shape[1])
        if (y >= 0).sum() == 0:
            continue
        ids.append(int(shot))
        xs.append(zerod.to_input(grid))
        ys.append(y)
    return ids, xs, ys, intervals


def ours_dir(grouping: str) -> Path:
    return OURS_DIRS[grouping]


def shot_groups(ids: list[int], grouping: str) -> dict[int, int]:
    """The group each shot's fold is dealt by: its run day, or itself."""
    if grouping == "shot":
        return {s: s for s in ids}
    if not DATES.exists():
        raise SystemExit(
            f"{DATES} is missing: run scripts/labeler/confinement_shot_dates_fetch.py"
        )
    days = bp.run_day_groups(pd.read_csv(DATES, keep_default_na=False))
    return {s: days[s] for s in ids}


def fold_roles(
    ids: list[int], ys: list[np.ndarray], grouping: str = "run_day"
) -> tuple[dict[int, int], dict[int, str], dict[int, int]]:
    """Each shot's test fold, its class signature and its group; whole groups (run
    days) go to one fold under ``run_day``, single shots under ``shot``."""
    signature = {
        s: "".join(zerod.CLASSES[c] + "," for c in sorted(set(y[y >= 0].tolist())))
        for s, y in zip(ids, ys, strict=True)
    }
    groups = shot_groups(ids, grouping)
    rng = np.random.default_rng(SEED)
    if grouping == "shot":
        test = bp.deal(ids, signature, FOLDS, rng)
    else:
        test = bp.deal_groups(ids, groups, signature, FOLDS, rng)
    return test, signature, groups


def neighbour_pairs(test_of: dict[int, int], within: int = 2) -> dict:
    """Pairs of scored shots at most ``within`` shot numbers apart, and how many of them
    fall in different folds (a model that trained on one sees a near-copy of the
    other)."""
    shots = sorted(test_of)
    pairs = [
        (a, b) for i, a in enumerate(shots) for b in shots[i + 1 :] if b - a <= within
    ]
    return {
        "within_shot_numbers": within,
        "pairs": len(pairs),
        "in_different_folds": sum(test_of[a] != test_of[b] for a, b in pairs),
    }


def train(args: argparse.Namespace) -> None:
    import torch

    from labeler.confinement import unet

    directory = ours_dir(args.grouping)
    directory.mkdir(parents=True, exist_ok=True)
    ids, xs, ys, _ = load_curated()
    test_of, signature, groups = fold_roles(ids, ys, args.grouping)
    device = torch.device(args.device)
    cfg = unet.UNetConfig(steps=args.steps) if args.steps else unet.UNetConfig()
    pos = {s: i for i, s in enumerate(ids)}
    corpus = unet.Corpus(xs, ys, device, cfg.window)
    print(
        f"{len(ids)} shots, {sum(int((y >= 0).sum()) for y in ys)} labelled bins",
        flush=True,
    )
    for fold in range(FOLDS):
        if args.folds is not None and fold not in args.folds:
            continue
        target = directory / f"fold{fold}_pred.npz"
        if target.exists() and not args.force:
            continue
        rest = [s for s in ids if test_of[s] != fold]
        rng = np.random.default_rng(SEED + 1 + fold)
        if args.grouping == "shot":
            val_of = bp.deal(rest, signature, 6, rng)
        else:
            val_of = bp.deal_groups(rest, groups, signature, 6, rng)
        val = [s for s in rest if val_of[s] == 0]
        fit = [s for s in rest if val_of[s] != 0]
        held = [s for s in ids if test_of[s] == fold]
        run_cfg = unet.UNetConfig(**{**cfg.as_dict(), "seed": SEED + fold})
        print(
            f"fold {fold}: {len(fit)} train, {len(val)} validation, "
            f"{len(held)} test shots",
            flush=True,
        )
        model, record = unet.train(
            corpus,
            np.array([pos[s] for s in fit]),
            [xs[pos[s]] for s in val],
            [ys[pos[s]] for s in val],
            run_cfg,
            device,
            log=lambda m, fold=fold: print(f"fold {fold} {m}", flush=True),
        )
        torch.save(model.state_dict(), directory / f"fold{fold}_model.pt")
        probs = {
            f"s{s}": unet.predict_shot(model, xs[pos[s]], device).astype(np.float16)
            for s in held
        }
        np.savez(target, **probs)
        record.update(
            {
                "fold": fold,
                "grouping": args.grouping,
                "train_shots_list": fit,
                "val_shots_list": val,
                "test_shots_list": held,
            }
        )
        (directory / f"fold{fold}_training.json").write_text(json.dumps(record))
        print(
            f"fold {fold} done: best val macro-F1 "
            f"{record['best_val_macro_f1']:.4f} ({record['seconds']} s)",
            flush=True,
        )


def held_out_probs(grouping: str) -> dict[int, np.ndarray]:
    directory = ours_dir(grouping)
    out: dict[int, np.ndarray] = {}
    for fold in range(FOLDS):
        path = directory / f"fold{fold}_pred.npz"
        if path.exists():
            with np.load(path) as d:
                out.update({int(k[1:]): d[k].astype(np.float32) for k in d.files})
    return out


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def per_bin_scores(
    probs: dict[int, np.ndarray],
    ys: dict[int, np.ndarray],
    keep: dict[int, np.ndarray] | None = None,
) -> dict:
    """Macro-F1 (shot bootstrap), per-class F1, AUROC and AUPRC, labelled bins."""
    guess, truth, shot, soft = [], [], [], []
    for s, p in probs.items():
        y = ys[s]
        sel = y >= 0
        if keep is not None:
            sel &= keep[s]
        guess.append(p.argmax(0)[sel])
        truth.append(y[sel].astype(np.int64))
        shot.append(np.full(sel.sum(), s))
        soft.append(p.T[sel])
    guess, truth, shot, soft = (np.concatenate(a) for a in (guess, truth, shot, soft))
    out = bp.summarise(guess, truth, shot, None, replicates=REPLICATES)
    out["rank"] = scoring.one_vs_rest(soft, truth)
    return out


def corpus_bes_shots() -> set[int]:
    """The shots the corpus holds BES for (the BES ablation's 117 corpus shots)."""
    return {int(f.stem) for f in CORPUS_BES.glob("*.npz") if "tmp" not in f.name}


def by_population(probs: dict[int, np.ndarray], ys: dict[int, np.ndarray]) -> dict:
    """Per-bin scores on the corpus shots (2021-2023, 500 kHz BES in the corpus) and on
    the other shots, the split the BES ablation found its shot-held-out deficit on."""
    corpus = corpus_bes_shots()
    out = {}
    for name, keep in (
        ("corpus_shots", lambda s: s in corpus),
        ("other_shots", lambda s: s not in corpus),
    ):
        sub = {s: p for s, p in probs.items() if keep(s)}
        if sub:
            scored = per_bin_scores(sub, ys)
            out[name] = {k: scored[k] for k in ("windows", "shots", "macro_f1", "ci95")}
            out[name]["classes"] = scored["classes"]
    return out


def score(args: argparse.Namespace) -> None:
    ids, _, ys_list, _ = load_curated()
    ys = dict(zip(ids, ys_list, strict=True))
    test_of, _, groups = fold_roles(ids, ys_list, args.grouping)
    probs = held_out_probs(args.grouping)
    probs = {s: p for s, p in probs.items() if s in ys}
    margin = args.margin_ms
    away = {}
    for s in probs:
        y = ys[s]
        near = np.zeros(len(y), dtype=bool)
        for c in np.flatnonzero(np.diff(y.astype(np.int16)) != 0) + 1:
            near[max(0, c - int(margin)) : c + int(margin)] = True
        away[s] = ~near
    result: dict = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": "confine-ours",
        "folds": FOLDS,
        "grouping": args.grouping,
        "groups": len(set(groups.values())),
        "shots_per_fold": np.bincount(list(test_of.values()), minlength=FOLDS).tolist(),
        "neighbour_pairs": neighbour_pairs(test_of),
        "shots_scored": len(probs),
        "shots_without_prediction": sorted(set(ids) - set(probs)),
        "per_bin_all_labelled": per_bin_scores(probs, ys),
        f"per_bin_away_from_edges_{int(margin)}ms": per_bin_scores(probs, ys, away),
        "per_bin_by_population": by_population(probs, ys),
    }
    per_shot = [
        segments.shot_counts(ys[s], p.argmax(0)) for s, p in sorted(probs.items())
    ]
    result["segmental"] = segments.summarise(per_shot, replicates=REPLICATES)
    smooth = [
        segments.shot_counts(ys[s], mode_filter(p.argmax(0), MODE_BINS))
        for s, p in sorted(probs.items())
    ]
    result["segmental_mode_filtered"] = {
        "mode_bins": MODE_BINS,
        **segments.summarise(smooth, replicates=REPLICATES),
    }
    result["comparison_with_confine_cnn"] = {
        Path(d).name: compare_with_cnn(probs, ys, Path(d)) for d in args.cnn_predictions
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / SCORE_FILES[args.grouping]).write_text(json.dumps(result, indent=1))
    pb = result["per_bin_all_labelled"]
    print(f"{len(probs)} shots: macro-F1 {pb['macro_f1']:.3f} {pb['ci95']['macro_f1']}")
    seg = result["segmental"]
    print(
        "segmental:",
        {k: round(seg[k]["value"], 3) for k in ("f1_10", "f1_25", "f1_50", "edit")},
    )


def mode_filter(labels: np.ndarray, width: int) -> np.ndarray:
    """The most common class in a centred window of ``width`` bins."""
    counts = np.stack(
        [
            np.convolve((labels == c).astype(np.float32), np.ones(width), mode="same")
            for c in range(len(zerod.CLASSES))
        ]
    )
    return counts.argmax(0)


def compare_with_cnn(
    probs: dict[int, np.ndarray], ys: dict[int, np.ndarray], path: Path
) -> dict:
    """confine-ours against the BES classifier on the BES windows both scored.

    Each BES window carries its centre time and interval label; confine-ours is read at
    the bin holding that time. Both are held out by shot. Two populations: every window
    the BES row predicted, and the paper-criteria windows (gated, away from interval
    ends).
    """
    if not path.exists():
        return {"status": f"{path} does not exist"}
    pred = pd.concat(
        [pd.read_csv(f) for f in sorted(path.glob("*_predictions.csv"))],
        ignore_index=True,
    )
    pred = pred[pred.shot.isin(probs)].copy()
    ours = np.zeros((len(pred), 4), dtype=np.float32)
    for s, g in pred.groupby("shot"):
        p = probs[int(s)]
        bins = np.clip(np.floor(g.center_ms.to_numpy()).astype(int), 0, p.shape[1] - 1)
        ours[pred.index.get_indexer(g.index)] = p[:, bins].T
    cnn = pred[[f"p_{c}" for c in bp.CLASSES]].to_numpy()
    truth = pred.label.to_numpy()
    shots = pred.shot.to_numpy()
    out: dict = {
        "predictions": str(path),
        "windows": len(pred),
        "shots": int(pred.shot.nunique()),
    }
    for name, mask in (
        ("all_windows", np.ones(len(pred), dtype=bool)),
        ("score_ok_windows", pred.score_ok.to_numpy().astype(bool)),
    ):
        _, by_cnn = bp.confusion_by_shot(cnn.argmax(1), truth, shots, mask)
        _, by_ours = bp.confusion_by_shot(ours.argmax(1), truth, shots, mask)
        out[name] = {
            "confine-cnn": bp.summarise(
                cnn.argmax(1), truth, shots, mask, replicates=REPLICATES
            ),
            "confine-ours": bp.summarise(
                ours.argmax(1), truth, shots, mask, replicates=REPLICATES
            ),
            "macro_f1_ours_minus_cnn": bp.paired_difference(
                by_ours, by_cnn, replicates=REPLICATES
            ),
        }
    return out


def mean_models(device, grouping: str):
    import torch

    from labeler.confinement import unet

    directory = ours_dir(grouping)
    models = []
    for fold in range(FOLDS):
        path = directory / f"fold{fold}_model.pt"
        if path.exists():
            m = unet.UNet1d().to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)
    return models


def bridge_gaps(active: np.ndarray, gap: int) -> np.ndarray:
    """``active`` with every run of False of at most ``gap`` bins that lies between two
    runs of True set to True."""
    out = active.copy()
    edges = np.flatnonzero(np.diff(np.r_[False, active, False].astype(np.int8)))
    # edges alternate: a run starts, ends, the next starts, ...
    for end, start in zip(edges[1:-1:2], edges[2::2], strict=False):
        if start - end <= gap:
            out[end:start] = True
    return out


def segment_shot(
    prob: np.ndarray, active: np.ndarray
) -> list[tuple[int, int, int, float]]:
    """Runs of one class within the active bins: ``(class, first bin, last bin + 1,
    confidence)``."""
    labels = mode_filter(prob.argmax(0), MODE_BINS)
    labels = np.where(active, labels, -1)
    runs = segments.segments(labels)
    merged: list[list[int]] = []
    for cls, lo, hi in runs:
        if (
            merged
            and merged[-1][0] == cls
            and merged[-1][2] == lo
            or hi - lo < MIN_SEGMENT_MS
            and merged
            and merged[-1][2] == lo
        ):
            merged[-1][2] = hi
        else:
            merged.append([cls, lo, hi])
    out = []
    for cls, lo, hi in merged:
        if hi - lo >= MIN_SEGMENT_MS:
            out.append((cls, lo, hi, float(prob[cls, lo:hi].mean())))
    return out


ROSTER_COLUMNS = [
    "shot",
    "category",
    "t_start",
    "t_end",
    "confidence",
    "predicted",
    "source",
    "tier",
    "extrapolated",
]
INDEX_OF = {cat: idx for idx, cat in CATEGORY.items()}


def roster_rows(
    shot: int,
    found: list[tuple[int, int, int, float]],
    *,
    curated: bool,
    last_curated: int,
) -> list[dict]:
    """The roster rows of one shot's segments.

    ``category`` is the segment's class, or 5 (uncertain) when its confidence is under
    ``CONFIDENCE_FLOOR``; ``predicted`` keeps the class either way. A curated shot's
    segments come from the fold model that never saw it (``held_out``), any other
    shot's from the five-model mean (``ensemble``). A QH or WPQH segment on a shot
    outside the curated set is on the ``unreviewed`` tier, and a shot past the last
    curated one is ``extrapolated``.
    """
    rows = []
    for cls, lo, hi, conf in found:
        predicted = CATEGORY[cls]
        unreviewed = not curated and predicted in UNREVIEWED_CATEGORIES
        rows.append(
            {
                "shot": shot,
                "category": predicted if conf >= CONFIDENCE_FLOOR else UNCERTAIN,
                "t_start": float(lo),
                "t_end": float(hi),
                "confidence": round(conf, 3),
                "predicted": predicted,
                "source": "held_out" if curated else "ensemble",
                "tier": "unreviewed" if unreviewed else "model",
                "extrapolated": bool(shot > last_curated),
            }
        )
    return rows


def _mask(spans, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for lo, hi in spans:
        out[max(int(lo), 0) : max(int(hi), 0)] = True
    return out


def roster_audit(
    frame: pd.DataFrame,
    curated: set[int],
    intervals: pd.DataFrame,
    dalpha: pd.DataFrame | None,
    last_curated: int,
) -> dict:
    """What the roster labels are worth, from the roster and the curated intervals.

    * how much of a curated shot's roster label time lies inside a curated interval,
      and inside one of the same class (the rest is time nobody curated);
    * the prevalence of each class on the shots outside the curated set, and how much
      of it clears the confidence floor;
    * for those shots, how the network's H and L time sits against the D-alpha H-mode
      table (high stretches, uncertain stretches, the rest).
    """
    end = int(frame.t_end.max()) + 1 if len(frame) else 1
    cur_all = {int(s): g for s, g in intervals.groupby("shot")}
    names = {1: "H", 2: "L", 3: "QH", 4: "WPQH"}
    ms = {c: 0.0 for c in names}
    inside = dict(ms)
    same = dict(ms)
    for shot, g in frame[frame.shot.isin(curated)].groupby("shot"):
        iv = cur_all.get(int(shot))
        if iv is None:
            continue
        anywhere = _mask(zip(iv.t_start, iv.t_end, strict=True), end)
        by_class = {
            c: _mask(
                zip(iv.t_start[iv.label == i], iv.t_end[iv.label == i], strict=True),
                end,
            )
            for c, i in INDEX_OF.items()
        }
        for row in g.itertuples():
            lo, hi = int(row.t_start), int(row.t_end)
            ms[row.predicted] += hi - lo
            inside[row.predicted] += anywhere[lo:hi].sum()
            same[row.predicted] += by_class[row.predicted][lo:hi].sum()
    total = sum(ms.values())
    out: dict = {
        "confidence_floor": CONFIDENCE_FLOOR,
        "last_curated_shot": int(last_curated),
        "curated_shots": {
            "shots": int(frame.shot[frame.shot.isin(curated)].nunique()),
            "label_seconds": total / 1000,
            "share_inside_a_curated_interval": sum(inside.values()) / total
            if total
            else None,
            "share_inside_a_curated_interval_of_the_same_class": sum(same.values())
            / total
            if total
            else None,
            "by_class": {
                names[c]: {
                    "label_seconds": ms[c] / 1000,
                    "share_inside_a_curated_interval": inside[c] / ms[c]
                    if ms[c]
                    else None,
                    "share_inside_one_of_the_same_class": same[c] / ms[c]
                    if ms[c]
                    else None,
                }
                for c in names
            },
        },
    }
    other = frame[~frame.shot.isin(curated)].copy()
    other["seconds"] = (other.t_end - other.t_start) / 1000
    prevalence = {}
    for cat, name in names.items():
        g = other[other.predicted == cat]
        per_shot = g.groupby("shot").seconds.sum()
        prevalence[name] = {
            "segments": len(g),
            "segments_at_or_above_floor": int((g.confidence >= CONFIDENCE_FLOOR).sum()),
            "shots": int(g.shot.nunique()),
            "shots_with_1_s_or_more": int((per_shot >= 1).sum()),
            "seconds": float(g.seconds.sum()),
            "mean_confidence": float(g.confidence.mean()) if len(g) else None,
        }
    out["non_curated_shots"] = {
        "shots": int(other.shot.nunique()),
        "shots_past_the_last_curated_shot": int(
            other.shot[other.shot > last_curated].nunique()
        ),
        "by_predicted_class": prevalence,
        "segments_made_uncertain_by_the_floor": int(
            (other.category == UNCERTAIN).sum()
        ),
        "unreviewed_tier_segments": int((other.tier == "unreviewed").sum()),
    }
    if dalpha is not None:
        table = {int(s): g for s, g in dalpha.groupby("shot")}
        agree = {name: {"high": 0.0, "uncertain": 0.0, "other": 0.0} for name in "HL"}
        shots_used = set()
        for shot, g in other[other.predicted.isin((1, 2))].groupby("shot"):
            if int(shot) not in table:
                continue
            shots_used.add(int(shot))
            t = table[int(shot)]
            n = int(max(g.t_end.max(), t.t_end.max())) + 1
            high = _mask(t.loc[t.category == 1, ["t_start", "t_end"]].to_numpy(), n)
            unsure = _mask(t.loc[t.category == 5, ["t_start", "t_end"]].to_numpy(), n)
            for row in g.itertuples():
                lo, hi = int(row.t_start), int(row.t_end)
                k = np.sum(high[lo:hi])
                u = np.sum(unsure[lo:hi] & ~high[lo:hi])
                d = agree["H" if row.predicted == 1 else "L"]
                d["high"] += k
                d["uncertain"] += u
                d["other"] += (hi - lo) - k - u
        out["dalpha_cross_check"] = {
            "table": str(DALPHA_TABLE),
            "shots": len(shots_used),
            "note": "D-alpha H-mode table: high, uncertain, or unmarked (no H-mode "
            "detected or not observable; L-mode is never established by it)",
            **{
                f"{name}_seconds": {k: v / 1000 for k, v in d.items()}
                | {
                    "share_in_dalpha_high": d["high"] / max(sum(d.values()), 1.0),
                }
                for name, d in agree.items()
            },
        }
    return out


def apply(args: argparse.Namespace) -> None:
    import torch

    from labeler.confinement import unet

    device = torch.device(args.device)
    models = mean_models(device, args.grouping)
    if len(models) < FOLDS:
        raise SystemExit(f"{len(models)} of {FOLDS} fold models found")
    ids, _, ys, _ = load_curated()
    test_of, _, _ = fold_roles(ids, ys, args.grouping)
    intervals = bw.curated_intervals()
    last_curated = int(intervals.shot.max())
    roster = sorted(int(s) for s in pd.read_csv(ROSTER).shot)
    rows, missing, stats = [], {}, {"curated_held_out": 0, "ensemble": 0}
    all_probs = {}
    for shot in roster:
        path = GRIDS / f"{shot}.npz"
        if not path.exists():
            missing[shot] = "no D-alpha record"
            continue
        grid, _ = zerod.load_grid(path)
        power = grid[zerod.CHANNELS.index("pinj")] * zerod.SCALES["pinj"]
        if not np.isfinite(power).any():
            missing[shot] = "no beam power record"
            continue
        x = zerod.to_input(grid)
        if shot in test_of:
            probs = unet.predict_shot(models[test_of[shot]], x, device)
            stats["curated_held_out"] += 1
        else:
            probs = np.mean([unet.predict_shot(m, x, device) for m in models], axis=0)
            stats["ensemble"] += 1
        active = bridge_gaps(np.nan_to_num(power, nan=0.0) >= BEAM_ON_W, BEAM_GAP_MS)
        if active.sum() < MIN_SEGMENT_MS:
            missing[shot] = "beam power never reaches the threshold"
            continue
        all_probs[f"s{shot}"] = probs.astype(np.float16)
        found = segment_shot(probs, active)
        if not found:
            missing[shot] = (
                f"no segment of {MIN_SEGMENT_MS} ms or more while the beam is on"
            )
        rows += roster_rows(
            shot, found, curated=shot in test_of, last_curated=last_curated
        )
    frame = pd.DataFrame(rows, columns=ROSTER_COLUMNS)
    EXTEND.mkdir(parents=True, exist_ok=True)
    frame.to_csv(EXTEND / "roster.csv", index=False)
    directory = ours_dir(args.grouping)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "roster_probs.npz", **all_probs)
    dalpha = pd.read_csv(DALPHA_TABLE) if DALPHA_TABLE.exists() else None
    audit = roster_audit(frame, set(test_of), intervals, dalpha, last_curated)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "roster_audit.json").write_text(
        json.dumps(
            {
                "git": git_sha(),
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "grouping": args.grouping,
                **audit,
            },
            indent=1,
        )
    )
    names = {
        1: "high (H)",
        2: "low (L)",
        3: "qh (QH)",
        4: "wpqh (WPQH)",
        5: "uncertain (segment below the confidence floor)",
    }
    meta = {
        "category": "confinement",
        "categories": {str(k): v for k, v in names.items()},
        "columns": list(frame.columns),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "made_by": "scripts/labeler/confinement_ours.py apply",
        "git_sha": git_sha(),
        "producer": "confine-ours",
        "status": "experimental",
        "model": "1D U-Net (U-Time style) over D-alpha, line density, beta_N, W_MHD "
        "and injected power; 5 folds, each holding whole run days out; "
        "docs/labeler/confinement_ours.md",
        "roster": str(ROSTER),
        "segmentation": {
            "bins_ms": 1.0,
            "active": f"total beam power >= {BEAM_ON_W:.0f} W, gaps of at most "
            f"{BEAM_GAP_MS} ms bridged",
            "mode_filter_bins": MODE_BINS,
            "min_segment_ms": MIN_SEGMENT_MS,
            "confidence": "mean probability of the segment's class over its bins",
            "confidence_floor": f"a segment under {CONFIDENCE_FLOOR} is category "
            f"{UNCERTAIN} (uncertain); `predicted` keeps its class",
        },
        "columns_meaning": {
            "category": "1 high, 2 low, 3 qh, 4 wpqh, 5 uncertain (under the floor)",
            "predicted": "the segment's class, before the confidence floor",
            "source": "held_out: a curated shot, read by the fold model that never "
            "saw it; ensemble: any other shot, the mean of the five fold models",
            "tier": "unreviewed: a QH or WPQH segment on a shot outside the curated "
            "set (the curated set leaves out ELM-free and other quiescent H-modes "
            "the network could read as QH); model: the rest",
            "extrapolated": f"true for a shot past the last curated one "
            f"({last_curated}); the network never saw those campaigns",
        },
        "shots_in_roster": len(roster),
        "shots_labelled": int(frame.shot.nunique()),
        "shots_missing": {
            "count": len(missing),
            "reasons": {str(k): v for k, v in sorted(missing.items())},
        },
        "prediction_source": {
            "curated shots": "the fold model that never saw the shot",
            "other shots": "mean of the five fold models",
            **stats,
        },
        "audit": "outputs/labeler/confinement/ours/roster_audit.json",
        "curated_interval_coverage": audit["curated_shots"][
            "share_inside_a_curated_interval"
        ],
        "coverage": "Active (beam-on) bins only; other time is not assessed, not low. "
        "On curated shots only that share of the label time lies inside a curated "
        "interval; the rest is the network's reading of time nobody curated.",
    }
    (EXTEND / "roster.meta.json").write_text(json.dumps(meta, indent=1))
    print(
        f"{len(frame)} segments on {frame.shot.nunique()} of {len(roster)} roster "
        f"shots; {len(missing)} missing; {int((frame.category == UNCERTAIN).sum())} "
        f"below the floor; curated-interval coverage "
        f"{meta['curated_interval_coverage']:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--force", action="store_true")
    b.add_argument(
        "--fetched-only",
        action="store_true",
        help="only shots whose 0D fetch has landed",
    )
    b.add_argument("--limit", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--grouping", choices=tuple(OURS_DIRS), default="run_day")
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--folds", type=int, nargs="+", default=None)
    t.add_argument("--steps", type=int, default=0)
    t.add_argument("--force", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--grouping", choices=tuple(OURS_DIRS), default="run_day")
    s.add_argument("--margin-ms", type=float, default=20.0)
    s.add_argument(
        "--cnn-predictions",
        nargs="*",
        default=[],
        help="directories of confine-cnn ablation rows to compare with",
    )
    a = sub.add_parser("apply")
    a.add_argument("--grouping", choices=tuple(OURS_DIRS), default="run_day")
    a.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    {"build": build, "train": train, "score": score, "apply": apply}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
