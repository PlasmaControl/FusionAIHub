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
    protocol  confine-ours on exactly confine-cnn's comparison-table protocol: its
              142 test shots, its scored windows pooled over the five draws, its
              labels, macro F1, AUROC and AUPRC with shot-bootstrap intervals, and
              the paired difference
              -> outputs/labeler/confinement/ours/protocol_<variant>.json
    select    the inner-validation scores of the variants, which chose between them
              -> outputs/labeler/confinement/ours/selection.json
    apply     segment every roster shot
              -> <events>/confinement/extend_confine_ours/

``--variant`` names the recipe (``VARIANTS``): the 2026-10-03 networks (``run_day``,
the roster labels' and the default), the first version's folds by shot (``shot``)
and the 2026-10-14 variants scored on the protocol.

Large files (grids, models, probabilities) live under ``$LABELER_ROOT/round4/conf``
and ``$LABELER_ROOT/round5/conf``; only the small CSV and JSON records are written
into the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths
from labeler.confinement import bes_protocol as bp
from labeler.confinement import bes_windows as bw
from labeler.confinement import scoring, segments, zerod

EVENTS = Paths.from_env().label_tables

LABELER = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler"))
WORK = LABELER / "round4/conf"
RAW = LABELER / "raw"
ZEROD = WORK / "zerod"
GRIDS = WORK / "zerod_grid"
ROUND5 = LABELER / "round5/conf"


@dataclass(frozen=True)
class Variant:
    """One recipe of confine-ours: where its networks and held-out probabilities
    live, how its folds are dealt (whole run days or single shots), what it reads
    (the counter beams' power as a further channel; the D-alpha channels, or zeros
    in their place for the twin that runs on a shot with no D-alpha record) and how
    many networks, trained from different seeds on the same split, each fold
    averages."""

    directory: Path
    grouping: str = "run_day"
    counter_beams: bool = False
    d_alpha: bool = True
    seeds: int = 1
    scores: str = ""

    @property
    def channels(self) -> int:
        return zerod.N_INPUT + (2 if self.counter_beams else 0)


VARIANTS = {
    # the 2026-10-03 networks: the roster labels and the headline 0.885
    "run_day": Variant(WORK / "ours_runday", scores="scores.json"),
    # the first version's folds by single shot, kept for the leakage comparison
    "shot": Variant(WORK / "ours", grouping="shot", scores="scores_random_shots.json"),
    # 2026-10-14, on the 416 curated shots that have D-alpha (15 more fetched): the
    # same recipe, and the same with the counter beams' power added; inner
    # validation (`select`) kept the first, then three seeds per fold of it
    "base": Variant(ROUND5 / "ours_base"),
    "beams": Variant(ROUND5 / "ours_beams", counter_beams=True),
    "base3": Variant(ROUND5 / "ours_base", seeds=3),
    # the D-alpha-free twin of `base3` (one seed), for the protocol's three shots
    # with no D-alpha record anywhere
    "base_no_da": Variant(ROUND5 / "ours_base_no_da", d_alpha=False),
}
#: Models and held-out probabilities by the grouping that dealt the folds: whole run
#: days (the headline) or single shots (the first version's folds, kept for the
#: comparison).
OURS_DIRS = {
    "run_day": VARIANTS["run_day"].directory,
    "shot": VARIANTS["shot"].directory,
}
SCORE_FILES = {"run_day": "scores.json", "shot": "scores_random_shots.json"}
COUNTER = ROUND5 / "counter_beams"
DATES = WORK / "dates.csv"
CORPUS_BES = WORK / "bes500k"
DALPHA_TABLE = LABELER / "suggestions/dalpha_lh/v1/confinement_suggest_dalpha_lh_v1.csv"
OUT = REPO / "outputs/labeler/confinement/ours"
EXTEND = EVENTS / "confinement/extend_confine_ours"
ROSTER = Path(
    os.environ.get(
        "CONFINEMENT_ROSTER",
        str(EVENTS / "confinement/shots.csv"),
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
#: the curated set may not contain ELM-free and other quiescent H-modes that the
#: network could read as QH or WPQH (unverified).
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


def counter_channel(shot: int, n: int) -> np.ndarray:
    """The counter beams' power on a shot's ``n`` bins (``zerod.beam_group``), cached
    under ``COUNTER``."""
    path = COUNTER / f"{shot}.npy"
    if path.exists():
        got = np.load(path)
        if len(got) == n:
            return got
    got = zerod.beam_group(shot, n, RAW, ZEROD).astype(np.float32)
    COUNTER.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{shot}.{os.getpid()}.tmp.npy")
    np.save(tmp, got)
    tmp.replace(path)
    return got


def network_input(grid: np.ndarray, shot: int, variant: Variant) -> np.ndarray:
    """A shot's network input under ``variant``: ``zerod.to_input``, with the D-alpha
    channels zeroed for the D-alpha-free twin and the counter beams' power (clipped
    as the other channels) and its mask appended where the variant reads it."""
    x = zerod.to_input(grid)
    if not variant.d_alpha:
        x[: zerod.CHANNELS.index("dens")] = 0.0
    if variant.counter_beams:
        counter = counter_channel(shot, grid.shape[1])
        value = np.clip(np.nan_to_num(counter, nan=0.0), -1.0, 8.0)
        mask = np.isfinite(counter).astype(np.float32)
        x = np.concatenate([x, value[None], mask[None]])
    return x.astype(np.float32)


def load_curated(
    variant: Variant | None = None,
) -> tuple[list[int], list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    """Shots with a grid and labelled bins: their ids, network inputs (under
    ``variant``, the 2026-10-03 inputs by default) and per-bin labels."""
    variant = variant or VARIANTS["run_day"]
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
        xs.append(network_input(grid, int(shot), variant))
        ys.append(y)
    return ids, xs, ys, intervals


def ours_dir(grouping: str) -> Path:
    return VARIANTS[grouping].directory


def run_days() -> dict[int, int]:
    """Every dated curated shot's run-day group (``bp.run_day_groups``)."""
    if not DATES.exists():
        raise SystemExit(
            f"{DATES} is missing: run scripts/labeler/confinement_shot_dates_fetch.py"
        )
    return bp.run_day_groups(pd.read_csv(DATES, keep_default_na=False))


def shot_groups(ids: list[int], grouping: str) -> dict[int, int]:
    """The group each shot's fold is dealt by: its run day, or itself."""
    if grouping == "shot":
        return {s: s for s in ids}
    days = run_days()
    return {s: days[s] for s in ids}


def deal_folds(
    ids: list[int], ys: list[np.ndarray], grouping: str = "run_day"
) -> tuple[dict[int, int], dict[int, str], dict[int, int]]:
    """Each shot's test fold, dealt afresh, its class signature and its group; whole
    groups (run days) go to one fold under ``run_day``, single shots under
    ``shot``."""
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


def fold_record(variant: Variant, fold: int) -> dict | None:
    """A fold's training record (its first network's), or None."""
    path = variant.directory / f"fold{fold}_training.json"
    return json.loads(path.read_text()) if path.exists() else None


def recorded_test_folds(variant: Variant) -> dict[int, int] | None:
    """The test fold of every shot a variant's networks were scored on, from their
    training records (``test_shots_list``); None unless every fold has one."""
    out: dict[int, int] = {}
    for fold in range(FOLDS):
        record = fold_record(variant, fold)
        if record is None:
            return None
        out.update({int(s): fold for s in record["test_shots_list"]})
    return out


def fold_roles(
    ids: list[int], ys: list[np.ndarray], grouping: str = "run_day"
) -> tuple[dict[int, int], dict[int, str], dict[int, int]]:
    """Each shot's test fold, its class signature and its group, for the variant
    named ``grouping`` (``run_day``: whole run days to a fold; ``shot``: single
    shots). Once the variant's networks are trained, the test folds are the ones
    they were trained with (their records), so that a shot added to the curated set
    later neither moves the others nor is taken for a shot a fold held out: it is
    left out of the map, as a shot no network was tested on. Before that they are
    dealt afresh (``deal_folds``)."""
    variant = VARIANTS[grouping]
    test, signature, groups = deal_folds(ids, ys, variant.grouping)
    recorded = recorded_test_folds(variant)
    if recorded is not None:
        test = {s: recorded[s] for s in ids if s in recorded}
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


def seed_stem(fold: int, seed: int) -> str:
    """The file stem of a fold's network from seed index ``seed``; the first keeps
    the 2026-10-03 names."""
    return f"fold{fold}" if seed == 0 else f"fold{fold}_s{seed}"


def train(args: argparse.Namespace) -> None:
    import torch

    from labeler.confinement import unet

    variant = VARIANTS[args.variant]
    directory = variant.directory
    directory.mkdir(parents=True, exist_ok=True)
    ids, xs, ys, _ = load_curated(variant)
    test_of, signature, groups = deal_folds(ids, ys, variant.grouping)
    device = torch.device(args.device)
    cfg = unet.UNetConfig(steps=args.steps) if args.steps else unet.UNetConfig()
    pos = {s: i for i, s in enumerate(ids)}
    corpus = unet.Corpus(xs, ys, device, cfg.window)
    print(
        f"{args.variant}: {len(ids)} shots, {xs[0].shape[0]} input channels, "
        f"{sum(int((y >= 0).sum()) for y in ys)} labelled bins",
        flush=True,
    )
    seeds = range(variant.seeds) if args.seeds is None else args.seeds
    for fold in range(FOLDS):
        if args.folds is not None and fold not in args.folds:
            continue
        rest = [s for s in ids if test_of[s] != fold]
        rng = np.random.default_rng(SEED + 1 + fold)
        if variant.grouping == "shot":
            val_of = bp.deal(rest, signature, 6, rng)
        else:
            val_of = bp.deal_groups(rest, groups, signature, 6, rng)
        val = [s for s in rest if val_of[s] == 0]
        fit = [s for s in rest if val_of[s] != 0]
        held = [s for s in ids if test_of[s] == fold]
        for k in seeds:
            stem = seed_stem(fold, k)
            target = directory / f"{stem}_pred.npz"
            if target.exists() and not args.force:
                continue
            first = fold_record(variant, fold)
            if k > 0 and first is not None and first["test_shots_list"] != held:
                raise SystemExit(
                    f"fold {fold}: the shots differ from its first network's; "
                    "retrain every seed"
                )
            # the first seed of a fold keeps the 2026-10-03 seed
            run_cfg = unet.UNetConfig(
                **{**cfg.as_dict(), "seed": SEED + fold + 1000 * k}
            )
            print(
                f"fold {fold} seed {k}: {len(fit)} train, {len(val)} validation, "
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
                log=lambda m, fold=fold, k=k: print(f"fold {fold}/{k} {m}", flush=True),
            )
            torch.save(model.state_dict(), directory / f"{stem}_model.pt")
            for name, shots in (("val", val), ("pred", held)):
                np.savez(
                    directory / f"{stem}_{name}.npz",
                    **{
                        f"s{s}": unet.predict_shot(model, xs[pos[s]], device).astype(
                            np.float16
                        )
                        for s in shots
                    },
                )
            record.update(
                {
                    "fold": fold,
                    "seed_index": k,
                    "variant": args.variant,
                    "grouping": variant.grouping,
                    "counter_beams": variant.counter_beams,
                    "d_alpha": variant.d_alpha,
                    "input_channels": int(xs[0].shape[0]),
                    "train_shots_list": fit,
                    "val_shots_list": val,
                    "test_shots_list": held,
                }
            )
            (directory / f"{stem}_training.json").write_text(json.dumps(record))
            print(
                f"fold {fold} seed {k} done: best val macro-F1 "
                f"{record['best_val_macro_f1']:.4f} ({record['seconds']} s)",
                flush=True,
            )


def _mean_npz(paths: list[Path]) -> dict[int, np.ndarray] | None:
    """The per-shot mean of the probabilities in ``paths``; None if one is missing."""
    if not paths or not all(p.exists() for p in paths):
        return None
    parts = []
    for path in paths:
        with np.load(path) as d:
            parts.append({int(k[1:]): d[k].astype(np.float32) for k in d.files})
    return {s: np.mean([p[s] for p in parts], axis=0) for s in parts[0]}


def held_out_probs(grouping: str) -> dict[int, np.ndarray]:
    """The held-out class probabilities of every shot the variant's folds tested,
    each fold's networks (``Variant.seeds``) averaged; a fold missing one is left
    out."""
    variant = VARIANTS[grouping]
    out: dict[int, np.ndarray] = {}
    for fold in range(FOLDS):
        got = _mean_npz(
            [
                variant.directory / f"{seed_stem(fold, k)}_pred.npz"
                for k in range(variant.seeds)
            ]
        )
        if got is not None:
            out.update(got)
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
    variant = VARIANTS[args.variant]
    ids, _, ys_list, _ = load_curated()
    ys = dict(zip(ids, ys_list, strict=True))
    test_of, _, groups = fold_roles(ids, ys_list, args.variant)
    probs = held_out_probs(args.variant)
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
        "variant": args.variant,
        "folds": FOLDS,
        "grouping": variant.grouping,
        "groups": len({groups[s] for s in test_of}),
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
    name = variant.scores or f"scores_{args.variant}.json"
    (OUT / name).write_text(json.dumps(result, indent=1))
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


#: confine-cnn's comparison-table row: the paper's selection, architecture, training
#: and split, the test windows of its five overlapping by-shot draws pooled (142
#: distinct test shots). `protocol` scores confine-ours on exactly its windows.
CNN_ROW = "full_cum_abcdrgef"
CNN_ROW_DIR = WORK / "ablation" / CNN_ROW
CNN_RECORD = REPO / "outputs/labeler/confinement/bes/ablation_rows" / f"{CNN_ROW}.json"


def cnn_windows(directory: Path = CNN_ROW_DIR) -> pd.DataFrame:
    """confine-cnn's scored windows of a split row: each draw's test windows that its
    gate and margins keep (``score_ok``), a shot tested by several draws once per
    draw, as the row's record pools them."""
    files = sorted(directory.glob("split*_predictions.csv"))
    if not files:
        raise SystemExit(f"no split predictions in {directory}")
    pred = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return pred[pred.score_ok.astype(bool)].reset_index(drop=True)


def at_windows(probs: dict[int, np.ndarray], windows: pd.DataFrame) -> np.ndarray:
    """confine-ours's class probabilities at each window, read at the 1 ms bin that
    holds the window's centre; NaN rows on a shot it has no probabilities for."""
    out = np.full((len(windows), len(bp.CLASSES)), np.nan, dtype=np.float32)
    centre = windows.center_ms.to_numpy()
    for s, idx in windows.groupby("shot").indices.items():
        p = probs.get(int(s))
        if p is None:
            continue
        bins = np.clip(np.floor(centre[idx]).astype(int), 0, p.shape[1] - 1)
        out[idx] = p[:, bins].T
    return out


def protocol_bins(ys: dict[int, np.ndarray], windows: pd.DataFrame) -> dict:
    """confine-ours's own labelled 1 ms bins on the protocol's shots against the
    bins the protocol scores (those that hold a scored window's centre): how many of
    ours it drops, and how often a window's label is ours at its bin."""
    labelled = scored = checked = agree = 0
    centre = windows.center_ms.to_numpy()
    label = windows.label.to_numpy()
    shots = 0
    for s, idx in windows.groupby("shot").indices.items():
        y = ys.get(int(s))
        if y is None:
            continue
        shots += 1
        bins = np.floor(centre[idx]).astype(int)
        inside = (bins >= 0) & (bins < len(y))
        hit = np.zeros(len(y), dtype=bool)
        hit[bins[inside]] = True
        labelled += int((y >= 0).sum())
        scored += int((hit & (y >= 0)).sum())
        checked += int(inside.sum())
        agree += int((y[bins[inside]] == label[idx][inside]).sum())
    return {
        "shots": shots,
        "labelled_bins": labelled,
        "scored_bins": scored,
        "dropped_bins": labelled - scored,
        "share_dropped": (labelled - scored) / labelled if labelled else None,
        "windows_checked": checked,
        "windows_whose_label_is_ours": agree,
    }


def protocol_scores(
    windows: pd.DataFrame, ours: np.ndarray, *, replicates: int = REPLICATES
) -> dict:
    """Both models on the protocol's windows that confine-ours has probabilities for
    (every one when it covers them all): macro and per-class F1 (the windows pooled,
    as the row's record pools them) and macro AUROC and AUPRC, each with its 95 %
    shot bootstrap, and the paired bootstrap of ours minus the CNN (the same shots
    drawn for both)."""
    keep = np.isfinite(ours).all(axis=1)
    truth = windows.label.to_numpy()[keep]
    shots = windows.shot.to_numpy()[keep]
    cnn = windows[[f"p_{c}" for c in bp.CLASSES]].to_numpy()[keep]
    mine = ours[keep]
    out: dict = {"windows": int(keep.sum()), "shots": int(np.unique(shots).size)}
    by_shot = {}
    rank_diff = scoring.paired_rank_difference(
        mine, cnn, truth, shots, replicates=replicates
    )
    for name, p, side in (("confine-cnn", cnn, "b"), ("confine-ours", mine, "a")):
        entry = bp.summarise(p.argmax(1), truth, shots, None, replicates=replicates)
        rank = scoring.one_vs_rest(p, truth)
        rank["ci95"] = {
            key: rank_diff[key][f"{side}_ci95"] for key in ("auroc", "auprc")
        }
        entry["ranking"] = rank
        out[name] = entry
        by_shot[name] = bp.confusion_by_shot(p.argmax(1), truth, shots)[1]
    out["ours_minus_cnn"] = {
        "macro_f1": bp.paired_difference(
            by_shot["confine-ours"], by_shot["confine-cnn"], replicates=replicates
        ),
        **{
            key: {k: v for k, v in rank_diff[key].items() if not k.endswith("_ci95")}
            for key in ("auroc", "auprc")
        },
        "replicates": replicates,
    }
    return out


def clean_folds(variant: Variant, day: int, days: dict[int, int]) -> list[int]:
    """The folds of a variant none of whose networks trained or validated on a shot
    of run day ``day``."""
    out = []
    for fold in range(FOLDS):
        record = fold_record(variant, fold)
        seen = set(record["train_shots_list"]) | set(record["val_shots_list"])
        if not any(days.get(int(s)) == day for s in seen):
            out.append(fold)
    return out


def protocol_probs(
    name: str, twin: str | None, shots: list[int], device
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """confine-ours's probabilities on ``shots``, each from networks that never
    trained or validated on it or on its run day: the variant's fold that tested it
    (``held_out``); for a shot no fold tested but with a grid, the mean of the
    variant's folds that never saw its run day (``clean_folds``); for a shot with no
    D-alpha record, the same from the D-alpha-free ``twin`` (``no_d_alpha_twin``).
    Returns the probabilities and where each came from."""
    held = held_out_probs(name)
    days = run_days()
    probs, source, nets = {}, {}, {}
    for shot in shots:
        if shot in held:
            probs[shot], source[shot] = held[shot], "held_out"
            continue
        path = GRIDS / f"{shot}.npz"
        if path.exists():
            chosen, grid = name, zerod.load_grid(path)[0]
        elif twin is not None:
            got = zerod.assemble(shot, RAW, ZEROD, require_d_alpha=False)
            if got is None:
                continue
            chosen, grid = twin, got["grid"]
        else:
            continue
        if chosen not in nets:
            nets[chosen] = fold_networks(device, chosen)
            if len(nets[chosen]) < FOLDS:
                raise SystemExit(f"{chosen}: not every fold has its networks")
        use = VARIANTS[chosen]
        folds = clean_folds(use, days[shot], days)
        if not folds:
            continue
        x = network_input(grid, shot, use)
        probs[shot] = np.mean(
            [predict_mean(nets[chosen][f], x, device) for f in folds], axis=0
        )
        source[shot] = "clean_folds" if chosen == name else "no_d_alpha_twin"
    return probs, source


def protocol(args: argparse.Namespace) -> None:
    import torch

    device = torch.device(args.device)
    windows = cnn_windows()
    record = json.loads(CNN_RECORD.read_text())
    ids, _, ys_list, _ = load_curated()
    ys = dict(zip(ids, ys_list, strict=True))
    shots = sorted(int(s) for s in windows.shot.unique())
    probs, source = protocol_probs(args.variant, args.twin, shots, device)
    ours = at_windows(probs, windows)
    scored = protocol_scores(windows, ours, replicates=args.replicates)
    own = record["own_population"]
    cnn_all = bp.summarise(
        windows[[f"p_{c}" for c in bp.CLASSES]].to_numpy().argmax(1),
        windows.label.to_numpy(),
        windows.shot.to_numpy(),
        None,
        replicates=args.replicates,
    )
    if (cnn_all["windows"], cnn_all["shots"]) != (own["windows"], own["shots"]) or abs(
        cnn_all["macro_f1"] - own["macro_f1"]
    ) > 1e-9:
        raise SystemExit(
            f"the windows give confine-cnn {cnn_all['macro_f1']:.4f} on "
            f"{cnn_all['windows']} windows, the record {own['macro_f1']:.4f} on "
            f"{own['windows']}"
        )
    counts = pd.Series(source).value_counts().to_dict()
    result = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": "confine-ours",
        "variant": args.variant,
        "twin": args.twin,
        "recipe": {
            "grouping": VARIANTS[args.variant].grouping,
            "counter_beams": VARIANTS[args.variant].counter_beams,
            "seeds": VARIANTS[args.variant].seeds,
        },
        "protocol": {
            "cnn_row": CNN_ROW,
            "cnn_record": str(CNN_RECORD.relative_to(REPO)),
            "windows": "confine-cnn's own population: the test windows of its five "
            "by-shot draws that its gate and margins keep, a shot once per draw "
            "that tests it; labels are the windows' own; confine-ours is read at the "
            "1 ms bin that holds each window's centre",
            "cnn_windows": own["windows"],
            "cnn_shots": own["shots"],
            "cnn_macro_f1_record": own["macro_f1"],
        },
        "ours_sources": {
            "counts": counts,
            "shots": {
                k: sorted(s for s, v in source.items() if v == k) for k in counts
            },
            "without_probabilities": sorted(set(shots) - set(probs)),
        },
        "bins": protocol_bins(ys, windows),
        **scored,
    }
    if scored["windows"] < len(windows):
        result["confine-cnn_all_windows"] = cnn_all
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"protocol_{args.variant}.json"
    out.write_text(json.dumps(result, indent=1))
    for name in ("confine-cnn", "confine-ours"):
        e = scored[name]
        print(
            f"{name}: macro F1 {e['macro_f1']:.3f} {np.round(e['ci95']['macro_f1'], 3)}"
            f" AUROC {e['ranking']['auroc']['macro']:.3f} AUPRC "
            f"{e['ranking']['auprc']['macro']:.3f} ({e['shots']} shots, "
            f"{e['windows']} windows)"
        )
    d = scored["ours_minus_cnn"]["macro_f1"]
    print(f"ours - cnn: {d['difference']:+.3f} {np.round(d['ci95'], 3)}; wrote {out}")


def select(args: argparse.Namespace) -> None:
    """Inner validation of each variant: per fold, the macro F1 of the fold's
    networks averaged, on the run days that validated it (never its test days),
    over every validation shot and over those outside the protocol's test shots."""
    ids, _, ys_list, _ = load_curated()
    ys = dict(zip(ids, ys_list, strict=True))
    outside = set(ids) - {int(s) for s in cnn_windows().shot.unique()}
    result = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "metric": "per-bin macro F1 of the inner-validation shots' labelled bins, "
        "the fold's networks' probabilities averaged; folds pooled and per fold",
        "variants": {},
    }
    for name in args.variants:
        variant = VARIANTS[name]
        per_fold, pooled = [], {"all": [], "outside_protocol": []}
        for fold in range(FOLDS):
            got = _mean_npz(
                [
                    variant.directory / f"{seed_stem(fold, k)}_val.npz"
                    for k in range(variant.seeds)
                ]
            )
            if got is None:
                per_fold = None
                break
            entry = {}
            for key, keep in (("all", got), ("outside_protocol", outside)):
                k = len(bp.CLASSES)
                conf = np.zeros((k, k), dtype=np.int64)
                for s, p in got.items():
                    if s not in keep:
                        continue
                    y = ys[s]
                    sel = y >= 0
                    flat = y[sel].astype(np.int64) * k + p.argmax(0)[sel]
                    conf += np.bincount(flat, minlength=k * k).reshape(k, k)
                entry[key] = bp.macro_f1(conf)
                pooled[key].append(conf)
            per_fold.append(entry)
        if per_fold is None:
            result["variants"][name] = {"status": "not every fold has its networks"}
            continue
        result["variants"][name] = {
            "seeds": variant.seeds,
            "counter_beams": variant.counter_beams,
            "d_alpha": variant.d_alpha,
            "per_fold": per_fold,
            **{
                f"pooled_{k}": bp.macro_f1(np.sum(v, axis=0)) for k, v in pooled.items()
            },
            **{
                f"mean_{k}": float(np.mean([f[k] for f in per_fold]))
                for k in ("all", "outside_protocol")
            },
        }
        r = result["variants"][name]
        print(
            f"{name}: inner validation pooled {r['pooled_all']:.4f} "
            f"(outside the protocol's shots {r['pooled_outside_protocol']:.4f}); "
            f"per-fold mean {r['mean_all']:.4f}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(result, indent=1))


def mean_models(device, grouping: str):
    """The first network of each fold of the variant named ``grouping`` (the
    2026-10-03 variants have one per fold)."""
    return [nets[0] for nets in fold_networks(device, grouping)]


def fold_networks(device, name: str) -> list[list]:
    """Each fold's networks of a variant, its ``seeds`` of them, in fold order; a fold
    missing one is left out."""
    import torch

    from labeler.confinement import unet

    variant = VARIANTS[name]
    out = []
    for fold in range(FOLDS):
        paths = [
            variant.directory / f"{seed_stem(fold, k)}_model.pt"
            for k in range(variant.seeds)
        ]
        if not all(p.exists() for p in paths):
            continue
        nets = []
        for path in paths:
            m = unet.UNet1d(c_in=variant.channels).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            nets.append(m)
        out.append(nets)
    return out


def predict_mean(nets, x: np.ndarray, device) -> np.ndarray:
    """The mean class probabilities ``(4, n)`` of ``nets`` on one shot's input."""
    from labeler.confinement import unet

    return np.mean([unet.predict_shot(m, x, device) for m in nets], axis=0)


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

#: The roster columns beyond the five interval columns; in an extend table they go
#: in the `attrs` cell, because the review page's table reader refuses any other
#: column (it skips `roster.csv` for that reason).
ATTR_COLUMNS = ["predicted", "source", "tier", "extrapolated"]
#: The roster in that layout, which the review page reads; `roster.csv` itself
#: stays as it is because the paper figures read its nine columns and pin its hash.
REVIEW_TABLE = "roster_review"


def to_extend_table(frame: pd.DataFrame) -> pd.DataFrame:
    """The roster-format rows as an extend table: shot, category, t_start, t_end,
    confidence and an `attrs` JSON object with the other four columns."""
    keep = list(frame.columns[:5])
    attrs = [
        json.dumps(
            {
                "predicted": int(row.predicted),
                "source": str(row.source),
                "tier": str(row.tier),
                "extrapolated": bool(row.extrapolated),
            },
            sort_keys=True,
        )
        for row in frame.itertuples()
    ]
    return frame[keep].assign(attrs=attrs)


def write_review_table(directory: Path = EXTEND) -> pd.DataFrame:
    """Write `roster_review.csv` and its meta from the `roster.csv` and
    `roster.meta.json` in `directory`; the roster files are only read."""
    frame = pd.read_csv(directory / "roster.csv")
    table = to_extend_table(frame)
    table.to_csv(directory / f"{REVIEW_TABLE}.csv", index=False)
    meta = json.loads((directory / "roster.meta.json").read_text())
    meta["columns"] = list(table.columns)
    meta["attrs_keys"] = ATTR_COLUMNS
    meta["derived_from"] = "roster.csv (the same rows; roster.csv is unchanged)"
    meta["made_by"] = "scripts/labeler/confinement_ours.py review"
    meta["columns_meaning"] = {
        "attrs": "a JSON object with the roster's four extra columns, predicted, "
        "source, tier and extrapolated (the review page's table reader accepts no "
        "column beyond `attrs`, so roster.csv's own extra columns make it skip that "
        "table); the keys are explained below",
        **{
            key: text for key, text in meta["columns_meaning"].items() if key != "attrs"
        },
    }
    (directory / f"{REVIEW_TABLE}.meta.json").write_text(json.dumps(meta, indent=1))
    return table


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

    device = torch.device(args.device)
    variant = VARIANTS[args.variant]
    models = fold_networks(device, args.variant)
    if len(models) < FOLDS:
        raise SystemExit(f"{len(models)} of {FOLDS} fold models found")
    ids, _, ys, _ = load_curated()
    test_of, _, _ = fold_roles(ids, ys, args.variant)
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
        x = network_input(grid, shot, variant)
        if shot in test_of:
            probs = predict_mean(models[test_of[shot]], x, device)
            stats["curated_held_out"] += 1
        else:
            probs = np.mean([predict_mean(m, x, device) for m in models], axis=0)
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
    directory = variant.directory
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
                "grouping": variant.grouping,
                "variant": args.variant,
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
            "set (the curated set may not contain ELM-free and other quiescent "
            "H-modes the network could read as QH (unverified)); model: the rest",
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
    write_review_table()
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

    def variant_arg(parser) -> None:
        # --grouping is the 2026-10-03 name of the two variants it had
        parser.add_argument(
            "--variant",
            "--grouping",
            dest="variant",
            choices=tuple(VARIANTS),
            default="run_day",
        )

    t = sub.add_parser("train")
    variant_arg(t)
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--folds", type=int, nargs="+", default=None)
    t.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="seed indices to train (default: all of the variant's)",
    )
    t.add_argument("--steps", type=int, default=0)
    t.add_argument("--force", action="store_true")
    s = sub.add_parser("score")
    variant_arg(s)
    s.add_argument("--margin-ms", type=float, default=20.0)
    s.add_argument(
        "--cnn-predictions",
        nargs="*",
        default=[],
        help="directories of confine-cnn ablation rows to compare with",
    )
    pr = sub.add_parser("protocol")
    variant_arg(pr)
    pr.add_argument(
        "--twin",
        choices=tuple(VARIANTS),
        default=None,
        help="the D-alpha-free variant for the protocol's shots with no D-alpha",
    )
    pr.add_argument("--device", default="cpu")
    pr.add_argument("--replicates", type=int, default=REPLICATES)
    se = sub.add_parser("select")
    se.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), required=True)
    a = sub.add_parser("apply")
    variant_arg(a)
    a.add_argument("--device", default="cuda:0")
    sub.add_parser(
        "review",
        help="write roster_review.csv, the roster in the layout the review page "
        "reads, from the roster.csv already there",
    )
    args = ap.parse_args(argv)
    if args.stage == "review":
        print(f"{len(write_review_table())} rows written to {REVIEW_TABLE}.csv")
        return 0
    {
        "build": build,
        "train": train,
        "score": score,
        "protocol": protocol,
        "select": select,
        "apply": apply,
    }[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
