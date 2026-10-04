#!/usr/bin/env python
"""Close the gap between the retrained BES confinement classifier and the published one.

Gill et al. (2024) report macro F1 0.94; the first retrain on our labels
(``confinement_bes_benchmark.py``) scored 0.678 on shots it never saw. This script
reruns the classifier one protocol difference at a time and cumulatively, scoring every
row the same way (per window, shot-bootstrap intervals), to attribute the gap:

  a  beam gating: 150L >= 700 kW, 150R <= 200 kW (400 kW for WPQH in training)
  b  transition exclusion: 20 ms inside each interval end, 100 ms after an L-mode
     interval
  c  the paper's optimiser: Adam, weight decay 1e-3, learning rates 1e-3 / 1e-5
  d  the BES layout: ``dead`` (the first rows' check that the block's channels carry a
     signal) or ``geometry`` (the 6 x 8 block chosen from the channels' (R, Z)
     positions against the EFIT flux, shots whose array does not reach the separatrix
     dropped; ``labeler.confinement.bes_geometry``)
  r  the paper's rows (first six of the 8 x 8 array) rather than rows 1-6
  g  the native 1 MHz sampling (a window is 1.02 ms, not 2.05 ms)
  e  every curated shot with BES (fetched), not just those in the corpus
  f  the paper's split by discharge, stratified by dominant regime, 72.5 / 15 / 12.5 %

The ROWS table below names each configuration. ``full_*`` rows train as the paper did
(no padding in the convolution, 227,644 parameters; 60,000 steps with no early stopping;
the checkpoint with the best validation F1) and choose the block by geometry. Stages::

    consolidate --data 500k|1m     join per-shot feature files into one dataset
    geometry                       per-shot 6 x 8 block from the fetched BES geometry
    run ROW [ROW ...]              train a row's folds (GPU), write predictions
    summarize                      score every finished row -> ablation.json and one
                                   ablation_rows/<row>.json each
    populations                    every row on the corpus and the fetched-only shots,
                                   paired factor steps, label fragmentation, class mix
    table                          the record as Markdown tables

Every score is per window and carries a 95 % interval bootstrapped over shots (1000
replicates). A row is scored on its own population (the windows it kept) and on the
paper-criteria population (gated, 20 ms inside its interval, 100 ms after L-mode) for
comparison. The cohort's blind test shots are never loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.confinement import bes_protocol as bp
from labeler.confinement import bes_windows as bw
from labeler.confinement import scoring

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
WORK = LABELER / "round4/conf"
DATASETS = WORK / "datasets"
RUNS = WORK / "ablation"
GEOMETRY = WORK / "geometry"
DEFAULT_OUT = REPO / "outputs/labeler/confinement/bes"
REPLICATES = 1000
SEED = 20261001
REPEATS = 5
TRANSITION_MS = 20.0
BUILDUP_MS = 100.0
#: Margins (ms inside an interval end) at which a row's own population is re-scored.
MARGINS_MS = (0.0, 20.0, 50.0, 100.0, 200.0)


@dataclass(frozen=True)
class Row:
    """One configuration of the pipeline."""

    name: str
    label: str
    data: str = "500k"  # sampling rate of the features: 500k (corpus) or 1m (native)
    # corpus: the shots the corpus holds BES for (every one of them is also fetched at
    # 1 MHz, so the 500 kHz and 1 MHz rows train on the same shots); all: every fetched
    # shot
    shots: str = "corpus"
    # rows [lo, hi) of the 8 x 8 array: the first retrain used rows 1-6
    rows: tuple[int, int] = (1, 7)
    gate: bool = False
    transition_ms: float = 0.0
    buildup_ms: float = 0.0
    # "": no check; "dead": drop shots with a dead channel in the block; "geometry":
    # the block chosen per shot from the channels' positions, shots whose array does
    # not reach the separatrix dropped
    layout: str = ""
    optimiser: str = "ours"
    # "padded": the first retrain's convolution and early stopping on validation loss;
    # "paper": no padding (227,644 parameters), 60,000 steps, best validation-F1
    # checkpoint
    arch: str = "padded"
    # cv5: shot-grouped 5-fold; paper: stratified 72.5 / 15 / 12.5 % splits by shot;
    # blocks: the same fractions drawn over blocks of 50 windows (0.2 s at 500 kHz,
    # 0.1 s at 1 MHz), so a test window has training windows of its own shot beside it
    # (a leaky diagnostic)
    protocol: str = "cv5"


def _chain() -> dict[str, Row]:
    """Rows: the baseline, each factor alone, and the factors added cumulatively."""
    base = Row("base", "first retrain (shot-grouped 5-fold, no gating)")
    alone = {
        "a": replace(base, name="a", label="+ beam gating", gate=True),
        "b": replace(
            base,
            name="b",
            label="+ transition exclusion",
            transition_ms=TRANSITION_MS,
            buildup_ms=BUILDUP_MS,
        ),
        "c": replace(base, name="c", label="+ paper optimiser", optimiser="paper"),
        "d": replace(base, name="d", label="+ channel check", layout="dead"),
        "r": replace(base, name="r", label="+ paper rows (first six)", rows=(0, 6)),
        "g": replace(base, name="g", label="+ native 1 MHz", data="1m"),
        "f": replace(base, name="f", label="+ paper split protocol", protocol="paper"),
        # the extra shots exist only as native 1 MHz fetches, so e is tested together
        # with g (compare ``only_g``)
        "ge": replace(
            base,
            name="ge",
            label="+ native 1 MHz and every fetched shot",
            data="1m",
            shots="all",
        ),
    }
    rows = {
        "base": base,
        **{f"only_{k}": replace(v, name=f"only_{k}") for k, v in alone.items()},
    }
    steps = [
        ("a", {"gate": True}, "+ beam gating"),
        (
            "ab",
            {"transition_ms": TRANSITION_MS, "buildup_ms": BUILDUP_MS},
            "+ transition exclusion",
        ),
        ("abc", {"optimiser": "paper"}, "+ paper optimiser"),
        ("abcd", {"layout": "dead"}, "+ channel check"),
        ("abcdr", {"rows": (0, 6)}, "+ paper rows"),
        ("abcdrg", {"data": "1m"}, "+ native 1 MHz"),
        ("abcdrge", {"shots": "all"}, "+ every fetched shot"),
        ("abcdrgef", {"protocol": "paper"}, "+ paper split protocol"),
    ]
    cur = base
    for name, change, label in steps:
        cur = replace(cur, name=f"cum_{name}", label=label, **change)
        rows[cur.name] = cur
    # the paper's split on the corpus shots at 500 kHz: a protocol-matched number that
    # needs no further fetch
    rows["cum_abcdrf"] = replace(
        rows["cum_abcdr"],
        name="cum_abcdrf",
        label="+ paper split protocol (500 kHz, corpus shots)",
        protocol="paper",
    )
    # the same pipeline with a within-shot split: how far the score rises when test
    # windows share shots with training windows (a leaky diagnostic, not a benchmark)
    rows["leak_abcdr"] = replace(
        rows["cum_abcdr"],
        name="leak_abcdr",
        label="within-shot block split, paper criteria (500 kHz; leaky diagnostic)",
        protocol="blocks",
    )
    # and with every fetched shot at 1 MHz: does the within-shot split still reach the
    # published score when the pipeline has the data it was built for
    rows["leak_abcdrge"] = replace(
        rows["cum_abcdrge"],
        name="leak_abcdrge",
        label="within-shot block split, paper criteria, every shot at 1 MHz (leaky)",
        protocol="blocks",
    )
    # the paper's training as well: no padding in the convolution, 60,000 steps with no
    # early stopping, best validation-F1 checkpoint; and (d) the block chosen by
    # geometry
    full = {"arch": "paper", "layout": "geometry"}
    rows["full_cum_abcdrge"] = replace(
        rows["cum_abcdrge"],
        name="full_cum_abcdrge",
        label="paper training and geometry block, every fetched shot, 5-fold by shot",
        **full,
    )
    rows["full_cum_abcdrgef"] = replace(
        rows["cum_abcdrgef"],
        name="full_cum_abcdrgef",
        label="paper training, geometry block and split: the protocol row",
        **full,
    )
    rows["full_leak_abcdrge"] = replace(
        rows["leak_abcdrge"],
        name="full_leak_abcdrge",
        label="paper training and geometry block, within-shot block split (leaky)",
        **full,
    )
    # factors a, b, c and r added singly to ``only_ge`` (all 444 shots at 1 MHz), where
    # every fold has the classes the single-factor rows on the 117 corpus shots lack
    for key, change, label in (
        ("a", {"gate": True}, "+ beam gating"),
        (
            "b",
            {"transition_ms": TRANSITION_MS, "buildup_ms": BUILDUP_MS},
            "+ transition exclusion",
        ),
        ("c", {"optimiser": "paper"}, "+ paper optimiser"),
        ("r", {"rows": (0, 6)}, "+ paper rows (first six)"),
    ):
        rows[f"ge_{key}"] = replace(
            rows["only_ge"],
            name=f"ge_{key}",
            label=f"every fetched shot at 1 MHz {label}",
            **change,
        )
    rows["full_only_c"] = replace(
        rows["only_c"],
        name="full_only_c",
        label="paper optimiser, architecture and training length (117 shots)",
        arch="paper",
    )
    return rows


ROWS = _chain()
#: The paper's population, for the common scoring of every row.
PAPER_CRITERIA = {
    "gate": True,
    "transition_ms": TRANSITION_MS,
    "buildup_ms": BUILDUP_MS,
}


def dataset_dir(kind: str) -> Path:
    return DATASETS / kind


def consolidate(args: argparse.Namespace) -> None:
    source = WORK / ("bes1mhz" if args.data == "1m" else "bes500k")
    table = bw.consolidate(source, dataset_dir(args.data), bw.curated_intervals())
    print(
        f"{args.data}: {len(table)} windows on {table.shot.nunique()} shots -> "
        f"{dataset_dir(args.data)}"
    )


def fetched_shots(directory: Path) -> set[int]:
    return {int(f.stem) for f in directory.glob("*.npz") if "tmp" not in f.name}


def load(row: Row):
    """Window table, memory-mapped features, channel power and the row's index into the
    dataset, all restricted to the row's shot set."""
    table, feats, power = bw.open_dataset(dataset_dir(row.data))
    if row.shots == "corpus":
        allowed = fetched_shots(WORK / "bes500k")
        keep = table.shot.isin(allowed).to_numpy()
        index = np.flatnonzero(keep)
        table = table.iloc[index].reset_index(drop=True)
        power = power[index]
        index_map = index
    else:
        index_map = np.arange(len(table))
    return table, feats, power, index_map


def load_blocks() -> pd.DataFrame:
    """The per-shot block table of the ``geometry`` stage (indexed by shot)."""
    return pd.read_csv(GEOMETRY / "blocks.csv").set_index("shot")


def plan(row: Row, table: pd.DataFrame, power: np.ndarray):
    """Masks, the per-fold roles of a row and its per-window block start rows (or
    None when the block is the fixed rows of the row)."""
    train_ok, score_ok = bp.window_masks(
        table, gate=row.gate, transition_ms=row.transition_ms, buildup_ms=row.buildup_ms
    )
    starts = None
    if row.layout == "dead":
        ch = slice(row.rows[0] * 8, row.rows[1] * 8)
        live = bp.live_shots(table, power, ch)
        ok = table.shot.isin(live).to_numpy()
        train_ok, score_ok = train_ok & ok, score_ok & ok
    elif row.layout == "geometry":
        blocks = load_blocks()
        usable = blocks.index[blocks.reaches & blocks.start.notna()]
        ok = table.shot.isin(usable).to_numpy()
        train_ok, score_ok = train_ok & ok, score_ok & ok
        starts = table.shot.map(blocks.start).fillna(0).to_numpy().astype(np.int64)
    if row.protocol == "cv5":
        roles = [bp.cv_roles(table, f) for f in range(5)]
        names = [f"fold{f}" for f in range(5)]
    elif row.protocol == "blocks":
        roles = [bp.block_roles(table, SEED + r) for r in range(REPEATS)]
        names = [f"split{r}" for r in range(REPEATS)]
    else:
        dom = bp.dominant_regime(table, train_ok & score_ok)
        roles = [bp.paper_roles(table, dom, SEED + r) for r in range(REPEATS)]
        names = [f"split{r}" for r in range(REPEATS)]
    return train_ok, score_ok, roles, names, starts


def training_config(row: Row):
    """The row's optimiser and, for ``arch == "paper"``, the paper's architecture and
    training length."""
    from labeler.confinement import bes_cnn as cnn

    cfg = cnn.PAPER if row.optimiser == "paper" else cnn.OURS
    if row.arch == "paper":
        cfg = replace(cfg, padding=cnn.PAPER_PADDING, early_stop=False)
    return cfg


def run_row(row: Row, args: argparse.Namespace) -> None:
    import torch

    from labeler.confinement import bes_cnn as cnn
    from labeler.confinement import bes_features as bf

    started = time.time()
    table, feats, power, index_map = load(row)
    labels = table.label.to_numpy().astype(np.int64)
    train_ok, score_ok, roles, names, starts = plan(row, table, power)
    out = (args.runs_dir or RUNS) / row.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "row.json").write_text(json.dumps(asdict(row), indent=1))
    cfg = training_config(row)
    if args.steps:
        cfg = replace(cfg, steps=args.steps)
    device = torch.device(args.device)
    for i, (role, name) in enumerate(zip(roles, names, strict=True)):
        if args.folds is not None and i not in args.folds:
            continue
        target = out / f"{name}_predictions.csv"
        if target.exists() and not args.force:
            continue
        train = np.flatnonzero((role == 0) & train_ok)
        val = np.flatnonzero((role == 1) & score_ok)
        test = np.flatnonzero(role == 2)
        if starts is None:
            block_power = power[train][:, row.rows[0] * 8 : row.rows[1] * 8]
        else:
            block_power = cnn.take_rows(
                power[train], starts[train], row.rows[1] - row.rows[0], 1
            )
        offset = bf.standardising_offset(block_power)
        # only the windows this split reads; ids are the table's positions, as the
        # splits use them
        needed = np.union1d(np.union1d(train, val), test)
        data = cnn.Features(
            feats,
            index_map[needed],
            row.rows,
            offset,
            device,
            gpu_gb=args.gpu_gb,
            ids=needed,
            starts=None if starts is None else starts[needed],
        )
        run_cfg = replace(cfg, seed=SEED + i)
        print(
            f"{row.name} {name}: {len(train)} train, {len(val)} val, "
            f"{len(test)} test windows on {table.shot[test].nunique()} shots",
            flush=True,
        )
        model, record = cnn.train(
            data,
            labels,
            train,
            val,
            run_cfg,
            device,
            log=lambda m, tag=f"{row.name} {name}": print(f"{tag} {m}", flush=True),
        )
        probs = cnn.predict(model, data, test)
        frame = table.iloc[test][
            ["shot", "start_ms", "center_ms", "label", "interval"]
        ].copy()
        frame["split"] = name
        frame["train_ok"] = train_ok[test]
        frame["score_ok"] = score_ok[test]
        for j, c in enumerate(bp.CLASSES):
            frame[f"p_{c}"] = probs[:, j]
        frame.to_csv(target, index=False)
        record["test_windows"] = len(test)
        record["test_shots"] = int(table.shot[test].nunique())
        (out / f"{name}_training.json").write_text(json.dumps(record))
        print(
            f"{row.name} {name} done: best step {record['best_step']}, "
            f"val macro-F1 {record['best_val_macro_f1']:.4f} ({record['seconds']} s)",
            flush=True,
        )
    print(f"{row.name}: {time.time() - started:.0f} s", flush=True)


def geometry(args: argparse.Namespace) -> None:
    """Per-shot 6 x 8 block of the 8 x 8 array, from the fetched channel positions.

    Reads ``$WORK/geometry/<shot>.npz`` (``confinement_bes_geometry_fetch.py``) for
    every shot of the 1 MHz dataset, takes each channel's median normalised flux over
    the shot's labelled windows and applies ``bes_geometry.choose_block``. Writes
    ``blocks.csv`` (here and in the output directory) and ``geometry.json``.
    """
    from labeler.confinement import bes_geometry as geo

    table = pd.read_csv(
        dataset_dir("1m") / "windows.csv", usecols=["shot", "center_ms"]
    )
    records = []
    for shot, group in table.groupby("shot"):
        rec = {"shot": int(shot), "windows": len(group), "status": "ok"}
        path = GEOMETRY / f"{shot}.npz"
        if not path.exists():
            records.append({**rec, "status": "no_geometry_file", "reaches": False})
            continue
        with np.load(path) as d:
            r_cm, z_cm = d["r_cm"], d["z_cm"]
            tree, times, psin = str(d["tree"]), d["gtime_ms"], d["psin"]
            rec["axis_residual"] = float(d["axis_residual"])
        row_r = r_cm.reshape(8, 8).mean(axis=1)
        rec["rows_displaced"] = int((np.abs(row_r - np.median(row_r)) > 1.0).sum())
        rec["r_min_cm"], rec["r_max_cm"] = float(r_cm.min()), float(r_cm.max())
        rec["z_span_cm"] = float(z_cm.max() - z_cm.min())
        rec["tree"] = tree
        if psin.shape[0] == 0:
            records.append({**rec, "status": "no_efit", "reaches": False})
            continue
        median = geo.shot_median_psin(times, psin, group.center_ms.to_numpy())
        records.append({**rec, **geo.choose_block(median)})
    frame = pd.DataFrame(records)
    GEOMETRY.mkdir(parents=True, exist_ok=True)
    frame.to_csv(GEOMETRY / "blocks.csv", index=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(args.out_dir / "geometry_blocks.csv", index=False)
    ok = frame[frame.reaches.astype(bool)]
    summary = {
        "criteria": {
            "pedestal_band_psin": list(geo.PEDESTAL_BAND),
            "row_min_channels_in_band": geo.ROW_MIN_CHANNELS,
            "block_rows": geo.BLOCK_ROWS,
            "reach_psin": geo.REACH_PSIN,
        },
        "shots": len(frame),
        "kept": len(ok),
        "dropped": {
            "no_geometry_file": int((frame.status == "no_geometry_file").sum()),
            "no_efit": int((frame.status == "no_efit").sum()),
            "array_does_not_reach_separatrix": int(
                ((frame.status == "ok") & ~frame.reaches.astype(bool)).sum()
            ),
        },
        "dropped_shots": [
            int(s) for s in frame.shot[~frame.reaches.astype(bool)].tolist()
        ],
        "block_start_counts": {
            str(int(k)): int(v) for k, v in ok.start.value_counts().sort_index().items()
        },
        "rows_covering_counts": {
            str(int(k)): int(v)
            for k, v in ok.rows_covering.value_counts().sort_index().items()
        },
        "shots_with_displaced_rows": int((frame.rows_displaced.fillna(0) > 0).sum()),
        "windows_kept": int(ok.windows.sum()),
        "windows_total": int(frame.windows.sum()),
        "outer_psin_median": float(ok.outer_psin.median()),
        "channels_in_band_median": float(ok.channels_in_band.median()),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_sha(),
    }
    (args.out_dir / "geometry.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "dropped_shots"}))


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


def row_names(row: Row) -> list[str]:
    """The fold or split names of a row's runs."""
    return (
        [f"fold{f}" for f in range(5)]
        if row.protocol == "cv5"
        else [f"split{r}" for r in range(REPEATS)]
    )


def row_predictions(row: Row):
    """A finished row's out-of-sample windows: the table of predictions, the argmax
    classes, the truth, the shot of each window, the row's own-population mask and the
    paper-criteria mask, and each window's depth inside its interval (ms); None if a
    fold is missing."""
    out = RUNS / row.name
    files = [out / f"{n}_predictions.csv" for n in row_names(row)]
    if not all(f.exists() for f in files):
        return None
    pred = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    probs = pred[[f"p_{c}" for c in bp.CLASSES]].to_numpy()
    own = pred.score_ok.to_numpy()
    table, _, _, _ = load(row)
    _, crit_score = bp.window_masks(table, **PAPER_CRITERIA)
    key = pd.MultiIndex.from_arrays([table.shot, table.start_ms.round(3)])
    crit = pd.Series(crit_score, index=key)
    pkey = pd.MultiIndex.from_arrays([pred.shot, pred.start_ms.round(3)])
    on_paper = crit.reindex(pkey).fillna(False).to_numpy().astype(bool)
    end = table.start_ms + table.dt_ms * 1024
    inside = pd.Series(
        np.minimum(table.start_ms - table.t_start, table.t_end - end).to_numpy(),
        index=key,
    )
    depth = inside.reindex(pkey).to_numpy()
    if row.layout == "geometry":
        blocks = load_blocks()
        usable = blocks.index[blocks.reaches & blocks.start.notna()]
        on_paper &= pred.shot.isin(usable).to_numpy()
    return pred, probs, own, on_paper, depth


def score_row(row: Row, rank: bool = False) -> dict | None:
    """Both scorings of a finished row; None if a fold is missing."""
    out = RUNS / row.name
    names = row_names(row)
    got = row_predictions(row)
    if got is None:
        return None
    pred, probs, own, on_paper, depth = got
    guess = probs.argmax(1)
    truth = pred.label.to_numpy()
    shots = pred.shot.to_numpy()
    result = {
        "row": asdict(row),
        "windows_predicted": len(pred),
        "own_population": bp.summarise(guess, truth, shots, own, replicates=REPLICATES),
        "paper_criteria_population": bp.summarise(
            guess, truth, shots, on_paper, replicates=REPLICATES
        ),
        "all_windows": bp.summarise(guess, truth, shots, None, replicates=REPLICATES),
        # the own population with the windows nearest an interval end dropped
        "margin_sensitivity": {
            f"{int(m)}": {
                k: v
                for k, v in bp.summarise(
                    guess, truth, shots, own & (depth >= m), replicates=REPLICATES
                ).items()
                if k in ("windows", "shots", "macro_f1", "ci95")
            }
            for m in MARGINS_MS
        },
        "training": [
            {
                k: v
                for k, v in json.loads((out / f"{n}_training.json").read_text()).items()
                if k != "history"
            }
            for n in names
        ],
    }
    if rank:
        result["ranking"] = {
            "population": "own",
            **scoring.rank_with_ci(probs[own], truth[own], shots[own], replicates=1000),
        }
    if row.protocol != "cv5":
        per = []
        for n in names:
            sub = pred[pred.split == n]
            p = sub[[f"p_{c}" for c in bp.CLASSES]].to_numpy().argmax(1)
            conf = bp.confusion_by_shot(
                p, sub.label.to_numpy(), sub.shot.to_numpy(), sub.score_ok.to_numpy()
            )[1].sum(0)
            per.append(bp.macro_f1(conf))
        result["per_split_macro_f1"] = per
    return result


def summarize(args: argparse.Namespace) -> None:
    results = {}
    for name, row in ROWS.items():
        res = score_row(row, rank=name in args.rank)
        if res is not None:
            results[name] = res
    record = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "replicates": REPLICATES,
        "paper_criteria": PAPER_CRITERIA,
        "rows": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ablation.json").write_text(json.dumps(record, indent=1))
    per_row = args.out_dir / "ablation_rows"
    per_row.mkdir(exist_ok=True)
    for name, res in results.items():
        (per_row / f"{name}.json").write_text(
            json.dumps(
                {
                    "git": record["git"],
                    "created": record["created"],
                    "replicates": REPLICATES,
                    "paper_criteria": PAPER_CRITERIA,
                    "row": asdict(ROWS[name]),
                    **res,
                },
                indent=1,
            )
        )
    for name, res in results.items():
        own, crit = res["own_population"], res["paper_criteria_population"]
        print(
            f"{name:14s} own {own['macro_f1']:.3f} [{own['ci95']['macro_f1'][0]:.2f}, "
            f"{own['ci95']['macro_f1'][1]:.2f}] ({own['shots']} shots)  "
            f"paper-criteria {crit['macro_f1']:.3f} ({crit['shots']} shots)"
        )


def _ci(entry: dict) -> str:
    lo, hi = entry["ci95"]["macro_f1"]
    return f"{entry['macro_f1']:.3f} [{lo:.2f}, {hi:.2f}]"


#: Factor steps, as (name, row, the row it is paired with): each is scored on the shots
#: both rows score, with the shot bootstrap of the paired difference.
STEPS = (
    ("a gate", "only_a", "base"),
    ("b margins", "only_b", "base"),
    ("c optimiser", "only_c", "base"),
    ("d channel check", "only_d", "base"),
    ("r rows", "only_r", "base"),
    ("g 1 MHz", "only_g", "base"),
    ("f paper split", "only_f", "base"),
    ("e shots (with g)", "only_ge", "only_g"),
    ("a gate, 444 shots", "ge_a", "only_ge"),
    ("b margins, 444 shots", "ge_b", "only_ge"),
    ("c optimiser, 444 shots", "ge_c", "only_ge"),
    ("r rows, 444 shots", "ge_r", "only_ge"),
    ("cum a gate", "cum_a", "base"),
    ("cum b margins", "cum_ab", "cum_a"),
    ("cum c optimiser", "cum_abc", "cum_ab"),
    ("cum d channel check", "cum_abcd", "cum_abc"),
    ("cum r rows", "cum_abcdr", "cum_abcd"),
    ("cum g 1 MHz", "cum_abcdrg", "cum_abcdr"),
    ("cum e shots", "cum_abcdrge", "cum_abcdrg"),
    ("cum f paper split", "cum_abcdrgef", "cum_abcdrge"),
    ("paper training + geometry (5-fold)", "full_cum_abcdrge", "cum_abcdrge"),
    ("paper training + geometry (paper split)", "full_cum_abcdrgef", "cum_abcdrgef"),
    ("f under paper training", "full_cum_abcdrgef", "full_cum_abcdrge"),
    ("paper training alone (optimiser row)", "full_only_c", "only_c"),
    ("within-shot split, early-stopped", "leak_abcdrge", "cum_abcdrgef"),
    ("within-shot split, paper training", "full_leak_abcdrge", "full_cum_abcdrgef"),
)
#: Rows scored by the year of the shot (when the shot dates are on disk).
YEAR_ROWS = ("only_ge", "cum_abcdrgef", "full_cum_abcdrge", "full_cum_abcdrgef")
DATES = WORK / "dates.csv"


def _by_shot(pred, probs, own, mask=None):
    keep = own if mask is None else own & mask
    return bp.confusion_by_shot(
        probs.argmax(1), pred.label.to_numpy(), pred.shot.to_numpy(), keep
    )


def _score(ids, conf):
    """Macro/per-class F1 and bootstrap interval of per-shot confusion matrices."""
    total = conf.sum(axis=0)
    f1 = bp.f1_from_conf(total)
    return {
        "shots": len(ids),
        "windows": int(total.sum()),
        "macro_f1": bp.macro_f1(total) if len(ids) else None,
        "f1": {
            c: (None if not np.isfinite(f1[i]) else float(f1[i]))
            for i, c in enumerate(bp.CLASSES)
        },
        "ci95_macro_f1": bp.bootstrap(conf, replicates=REPLICATES)["macro_f1"]
        if len(ids)
        else None,
        "shots_per_class": {
            c: int((conf[:, i].sum(axis=1) > 0).sum()) for i, c in enumerate(bp.CLASSES)
        },
    }


def fragmentation(corpus: set[int], fetched: set[int]) -> dict:
    """Interval lengths per class: the corpus shots against the fetched-only ones."""
    iv = bw.curated_intervals()
    iv = iv[iv.shot.isin(fetched)].copy()
    iv["length_ms"] = iv.t_end - iv.t_start
    iv["population"] = np.where(iv.shot.isin(corpus), "corpus", "other")
    out: dict = {}
    for pop, g in [*iv.groupby("population"), ("all", iv)]:
        out[pop] = {
            "shots": int(g.shot.nunique()),
            "intervals": len(g),
            "intervals_per_shot_median": float(g.groupby("shot").size().median()),
            "classes": {
                bw.CLASSES[k]: {
                    "intervals": len(c),
                    "median_ms": float(c.length_ms.median()),
                    "q25_ms": float(c.length_ms.quantile(0.25)),
                    "q75_ms": float(c.length_ms.quantile(0.75)),
                    "share_under_100_ms": float((c.length_ms < 100).mean()),
                    "labelled_s": float(c.length_ms.sum() / 1000.0),
                }
                for k, c in g.groupby("label")
            },
        }
    return out


def populations(args: argparse.Namespace) -> None:
    """Every finished row on the corpus shots and on the fetched-only shots, the paired
    factor steps, the interval fragmentation of the labels, the class-mix reweighting of
    the protocol rows and, with the shot dates on disk, the protocol rows by year."""
    corpus = fetched_shots(WORK / "bes500k")
    fetched = fetched_shots(WORK / "bes1mhz")
    rows, cache = {}, {}
    for name, row in ROWS.items():
        got = row_predictions(row)
        if got is None:
            continue
        cache[name] = got
        pred, probs, own, _, _ = got
        in_corpus = pred.shot.isin(corpus).to_numpy()
        rows[name] = {
            "own": _score(*_by_shot(pred, probs, own)),
            "corpus_shots": _score(*_by_shot(pred, probs, own, in_corpus)),
            "other_shots": _score(*_by_shot(pred, probs, own, ~in_corpus)),
        }
    steps = []
    for label, name, ref in STEPS:
        if name not in cache or ref not in cache:
            continue
        pa, qa, oa, _, _ = cache[name]
        pb, qb, ob, _, _ = cache[ref]
        ids_a, conf_a = _by_shot(pa, qa, oa)
        ids_b, conf_b = _by_shot(pb, qb, ob)
        entry = {"step": label, "row": name, "versus": ref}
        for pop, keep in (
            ("common_shots", lambda i: np.ones(len(i), bool)),
            ("corpus_shots", lambda i: np.isin(i, list(corpus))),
            ("other_shots", lambda i: ~np.isin(i, list(corpus))),
        ):
            common = np.intersect1d(ids_a, ids_b)
            common = common[keep(common)]
            if len(common) < 5:
                entry[pop] = {"shots": len(common)}
                continue
            ca = conf_a[np.searchsorted(ids_a, common)]
            cb = conf_b[np.searchsorted(ids_b, common)]
            entry[pop] = {
                "shots": len(common),
                "row_macro_f1": bp.macro_f1(ca.sum(axis=0)),
                "versus_macro_f1": bp.macro_f1(cb.sum(axis=0)),
                **bp.paired_difference(ca, cb, replicates=REPLICATES),
            }
        steps.append(entry)
    mix = {}
    for name in ("cum_abcdrgef", "full_cum_abcdrgef"):
        if name in cache:
            pred, probs, own, _, _ = cache[name]
            _, conf = _by_shot(pred, probs, own)
            total = conf.sum(axis=0)
            re = bp.reweight_to_mix(total)
            mix[name] = {
                "as_scored": bp.macro_f1(total),
                "paper_test_mix": list(bp.PAPER_TEST_MIX),
                "reweighted_macro_f1": bp.macro_f1(re),
                "reweighted_f1": {
                    c: float(v) for c, v in zip(bp.CLASSES, bp.f1_from_conf(re))
                },
            }
    years = {}
    if DATES.exists():
        year = pd.read_csv(DATES).set_index("shot").year
        for name in YEAR_ROWS:
            if name not in cache:
                continue
            pred, probs, own, _, _ = cache[name]
            y = pred.shot.map(year).to_numpy()
            years[name] = {
                str(int(v)): _score(*_by_shot(pred, probs, own, y == v))
                for v in sorted(pd.unique(y[~pd.isna(y)]))
            }
    record = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "replicates": REPLICATES,
        "populations": {
            "corpus": f"{len(corpus & fetched)} shots the corpus holds BES for",
            "other": f"{len(fetched - corpus)} fetched-only shots",
        },
        "rows": rows,
        "paired_steps": steps,
        "fragmentation": fragmentation(corpus, fetched),
        "class_mix": mix,
        "by_year": years,
        "year_source": str(DATES) if DATES.exists() else None,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "populations.json").write_text(json.dumps(record, indent=1))
    for name, r in rows.items():
        c, o = r["corpus_shots"], r["other_shots"]
        fmt = lambda e: (
            f"{e['macro_f1']:.3f} ({e['shots']} shots)" if e["macro_f1"] else "-"
        )
        print(f"{name:20s} corpus {fmt(c):22s} other {fmt(o)}")


def _support(labels: np.ndarray, shots: np.ndarray) -> dict:
    """Windows and shots per class of a set of windows."""
    return {
        c: {
            "windows": int((labels == i).sum()),
            "shots": len(np.unique(shots[labels == i])),
        }
        for i, c in enumerate(bp.CLASSES)
    }


#: A validation set with fewer shots than this in a class picks its checkpoint on that
#: class from almost nothing.
THIN_SHOTS = 3


def validation_audit(args: argparse.Namespace) -> None:
    """Per row and split: the classes the validation set holds (the checkpoint is the
    best validation macro-F1, a mean over the classes present), the test shots per
    class and the training run's best step. Marks a split degenerate when a class is
    absent from validation and thin when a class has under three validation shots; a
    row with such a split is underpowered for ranking single factors."""
    rows = {}
    for name, row in ROWS.items():
        done = [
            n for n in row_names(row) if (RUNS / name / f"{n}_training.json").exists()
        ]
        if not done:
            continue
        table, _, power, _ = load(row)
        _, score_ok, roles, names, _ = plan(row, table, power)
        labels, shot = table.label.to_numpy(), table.shot.to_numpy()
        splits = {}
        for role, split in zip(roles, names, strict=True):
            record_path = RUNS / name / f"{split}_training.json"
            if not record_path.exists():
                continue
            record = json.loads(record_path.read_text())
            val = (role == 1) & score_ok
            test = (role == 2) & score_ok
            support = _support(labels[val], shot[val])
            history = [h["val_macro_f1"] for h in record["history"]]
            splits[split] = {
                "val_shots": len(np.unique(shot[val])),
                "val": support,
                "classes_missing": [c for c, v in support.items() if v["windows"] == 0],
                "classes_thin": [
                    c for c, v in support.items() if 0 < v["shots"] < THIN_SHOTS
                ],
                "test": _support(labels[test], shot[test]),
                "steps_run": record["steps_run"],
                "best_step": record["best_step"],
                "best_val_macro_f1": record["best_val_macro_f1"],
                "val_macro_f1_spread": float(max(history) - min(history)),
            }
        degenerate = [k for k, v in splits.items() if v["classes_missing"]]
        thin = [
            k for k, v in splits.items() if v["classes_thin"] and k not in degenerate
        ]
        rows[name] = {
            "shots": row.shots,
            "splits": splits,
            "degenerate_splits": degenerate,
            "thin_splits": thin,
            "underpowered": bool(degenerate or thin),
            "wpqh_test_shots": [v["test"]["WP"]["shots"] for v in splits.values()],
        }
    record = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "thin_shots_below": THIN_SHOTS,
        "paper_wpqh_test_shots": 7,
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "validation_audit.json").write_text(json.dumps(record, indent=1))
    for name, r in rows.items():
        print(
            f"{name:20s} degenerate {r['degenerate_splits']} thin {r['thin_splits']} "
            f"WPQH test shots {r['wpqh_test_shots']}"
        )


def markdown(args: argparse.Namespace) -> None:
    """The ablation record as Markdown tables (for docs/labeler/confinement_*.md)."""
    record = json.loads((args.out_dir / "ablation.json").read_text())
    rows = record["rows"]
    print(
        "| Row | Configuration | Shots | Windows | Macro F1 [95 % CI], own windows | "
        "Macro F1 on the paper's windows | F1 L / H / QH / WPQH (own) | "
        "Shots L / H / QH / WPQH (own) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name, res in rows.items():
        own, crit = res["own_population"], res["paper_criteria_population"]
        per = " / ".join(
            f"{own['classes'][c]['f1']:.2f}" if own["classes"][c]["f1"] else "-"
            for c in bp.CLASSES
        )
        n_shots = " / ".join(str(own["classes"][c]["shots"]) for c in bp.CLASSES)
        print(
            f"| `{name}` | {res['row']['label']} | {own['shots']} | {own['windows']:,} "
            f"| {_ci(own)} | {_ci(crit)} | {per} | {n_shots} |"
        )
    print()
    print("| Row | " + " | ".join(f"{int(m)} ms" for m in MARGINS_MS) + " |")
    print("|---|" + "---|" * len(MARGINS_MS))
    for name, res in rows.items():
        cells = [_ci(res["margin_sensitivity"][f"{int(m)}"]) for m in MARGINS_MS]
        print(f"| `{name}` | " + " | ".join(cells) + " |")
    for name, res in rows.items():
        if "ranking" in res:
            r = res["ranking"]
            print(
                f"\n`{name}`: macro AUROC {r['auroc']['macro']:.3f} "
                f"[{r['ci95']['auroc'][0]:.3f}, {r['ci95']['auroc'][1]:.3f}], "
                f"macro AUPRC {r['auprc']['macro']:.3f} "
                f"[{r['ci95']['auprc'][0]:.3f}, {r['ci95']['auprc'][1]:.3f}]"
            )


def _cell(entry: dict | None) -> str:
    """``0.703 [0.59, 0.79] (142)`` for a scored population, ``-`` when it is empty."""
    if not entry or entry.get("macro_f1") is None:
        return "-"
    lo, hi = entry["ci95_macro_f1"]
    return f"{entry['macro_f1']:.3f} [{lo:.2f}, {hi:.2f}] ({entry['shots']})"


def _diff(entry: dict) -> str:
    """``+0.034 [-0.02, +0.09] (35)`` for a paired step on one population."""
    if "difference" not in entry:
        return f"- ({entry['shots']})"
    lo, hi = entry["ci95"]
    return f"{entry['difference']:+.3f} [{lo:+.2f}, {hi:+.2f}] ({entry['shots']})"


def markdown_populations(args: argparse.Namespace) -> None:
    """``populations.json`` and ``validation_audit.json`` as Markdown tables."""
    pop = json.loads((args.out_dir / "populations.json").read_text())
    print("| Row | All scored shots | Corpus shots | Fetched-only shots |")
    print("|---|---|---|---|")
    for name, r in pop["rows"].items():
        print(
            f"| `{name}` | {_cell(r['own'])} | {_cell(r['corpus_shots'])} | "
            f"{_cell(r['other_shots'])} |"
        )
    print()
    print("| Step | Row against its reference | Common shots | Corpus shots | Other |")
    print("|---|---|---|---|---|")
    for e in pop["paired_steps"]:
        print(
            f"| {e['step']} | `{e['row']}` - `{e['versus']}` | "
            f"{_diff(e['common_shots'])} | {_diff(e['corpus_shots'])} | "
            f"{_diff(e['other_shots'])} |"
        )
    print()
    print(
        "| Class | Corpus: intervals | median (IQR), ms | under 100 ms | "
        "Fetched-only: intervals | median (IQR), ms | under 100 ms |"
    )
    print("|---|---|---|---|---|---|---|")
    frag = pop["fragmentation"]
    for c in bp.CLASSES:
        cells = []
        for who in ("corpus", "other"):
            e = frag[who]["classes"][c]
            cells += [
                f"{e['intervals']}",
                f"{e['median_ms']:.0f} ({e['q25_ms']:.0f}-{e['q75_ms']:.0f})",
                f"{100 * e['share_under_100_ms']:.0f} %",
            ]
        print(f"| {c} | " + " | ".join(cells) + " |")
    print()
    for name, m in pop["class_mix"].items():
        per = " / ".join(f"{m['reweighted_f1'][c]:.2f}" for c in bp.CLASSES)
        print(
            f"`{name}` as scored {m['as_scored']:.3f}; reweighted to the paper's "
            f"test mix {m['paper_test_mix']} labelled s per class: "
            f"{m['reweighted_macro_f1']:.3f} (F1 {per})"
        )
    audit_path = args.out_dir / "validation_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        print()
        print("| Row | Degenerate splits | Thin splits | WPQH test shots per split |")
        print("|---|---|---|---|")
        for name, r in audit["rows"].items():
            print(
                f"| `{name}` | {', '.join(r['degenerate_splits']) or '-'} | "
                f"{', '.join(r['thin_splits']) or '-'} | {r['wpqh_test_shots']} |"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    sub = ap.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("consolidate")
    c.add_argument("--data", choices=("500k", "1m"), required=True)
    r = sub.add_parser("run")
    r.add_argument("rows", nargs="+", choices=sorted(ROWS))
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--folds", type=int, nargs="+", default=None)
    r.add_argument("--force", action="store_true")
    r.add_argument(
        "--steps", type=int, default=None, help="cap the steps (smoke tests)"
    )
    r.add_argument("--runs-dir", type=Path, default=None, help="write predictions here")
    r.add_argument(
        "--gpu-gb",
        type=float,
        default=4.0,
        help="keep the split's features on the GPU when they take at most this much",
    )
    sub.add_parser("geometry", help="per-shot block from the fetched BES geometry")
    sub.add_parser("populations", help="rows on fixed populations, paired steps")
    sub.add_parser("audit", help="validation sets and test shots per class and split")
    sub.add_parser("table", help="print ablation.json as Markdown")
    sub.add_parser("tables", help="print populations.json and the audit as Markdown")
    s = sub.add_parser("summarize")
    s.add_argument(
        "--rank",
        nargs="*",
        default=["cum_abcdr", "cum_abcdrf", "cum_abcdrgef"],
        help="rows that also get AUROC / AUPRC with bootstrap intervals",
    )
    args = ap.parse_args(argv)
    if args.stage == "consolidate":
        consolidate(args)
    elif args.stage == "geometry":
        geometry(args)
    elif args.stage == "populations":
        populations(args)
    elif args.stage == "audit":
        validation_audit(args)
    elif args.stage == "table":
        markdown(args)
    elif args.stage == "tables":
        markdown_populations(args)
    elif args.stage == "run":
        for name in args.rows:
            run_row(ROWS[name], args)
    else:
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
