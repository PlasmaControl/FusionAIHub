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
  d  a per-shot check that the channels of the block carry a signal
  r  the paper's rows (first six of the 8 x 8 array) rather than rows 1-6
  g  the native 1 MHz sampling (a window is 1.02 ms, not 2.05 ms)
  e  every curated shot with BES (fetched), not just those in the corpus
  f  the paper's split by discharge, stratified by dominant regime, 72.5 / 15 / 12.5 %

The ROWS table below names each configuration. Stages::

    consolidate --data 500k|1m     join per-shot feature files into one dataset
    run ROW [ROW ...]              train a row's folds (GPU), write predictions
    summarize                      score every finished row -> ablation.json and one
                                   ablation_rows/<row>.json each
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
    layout: bool = False
    optimiser: str = "ours"
    # cv5: shot-grouped 5-fold; paper: stratified 72.5 / 15 / 12.5 % splits by shot;
    # blocks: the same fractions drawn over 0.2 s blocks of windows, so a test window
    # has training windows of its own shot beside it (a leaky diagnostic)
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
        "d": replace(base, name="d", label="+ channel check", layout=True),
        "r": replace(base, name="r", label="+ paper rows (first six)", rows=(0, 6)),
        "g": replace(base, name="g", label="+ native 1 MHz", data="1m"),
        "f": replace(base, name="f", label="+ paper split protocol", protocol="paper"),
        # the extra shots exist only as native 1 MHz fetches, so e is tested together with g
        # (compare ``only_g``)
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
        ("abcd", {"layout": True}, "+ channel check"),
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
    # the same pipeline with a within-shot split: how far the score rises when test windows
    # share shots with training windows (a leaky diagnostic, not a benchmark number)
    rows["leak_abcdr"] = replace(
        rows["cum_abcdr"],
        name="leak_abcdr",
        label="within-shot block split, paper criteria (500 kHz; leaky diagnostic)",
        protocol="blocks",
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


def plan(row: Row, table: pd.DataFrame, power: np.ndarray):
    """Masks and the per-fold roles of a row."""
    train_ok, score_ok = bp.window_masks(
        table, gate=row.gate, transition_ms=row.transition_ms, buildup_ms=row.buildup_ms
    )
    if row.layout:
        ch = slice(row.rows[0] * 8, row.rows[1] * 8)
        live = bp.live_shots(table, power, ch)
        ok = table.shot.isin(live).to_numpy()
        train_ok, score_ok = train_ok & ok, score_ok & ok
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
    return train_ok, score_ok, roles, names


def run_row(row: Row, args: argparse.Namespace) -> None:
    import torch

    from labeler.confinement import bes_cnn as cnn
    from labeler.confinement import bes_features as bf

    started = time.time()
    table, feats, power, index_map = load(row)
    labels = table.label.to_numpy().astype(np.int64)
    train_ok, score_ok, roles, names = plan(row, table, power)
    out = (args.runs_dir or RUNS) / row.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "row.json").write_text(json.dumps(asdict(row), indent=1))
    cfg = cnn.PAPER if row.optimiser == "paper" else cnn.OURS
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
        offset = bf.standardising_offset(
            power[train][:, row.rows[0] * 8 : row.rows[1] * 8]
        )
        data = cnn.Features(
            feats,
            index_map,
            row.rows,
            offset,
            device,
            ids=np.arange(len(index_map)),  # the table's positions, as the splits use
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


def score_row(row: Row, rank: bool = False) -> dict | None:
    """Both scorings of a finished row; None if a fold is missing."""
    out = RUNS / row.name
    names = (
        [f"fold{f}" for f in range(5)]
        if row.protocol == "cv5"
        else [f"split{r}" for r in range(REPEATS)]
    )
    files = [out / f"{n}_predictions.csv" for n in names]
    if not all(f.exists() for f in files):
        return None
    pred = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    probs = pred[[f"p_{c}" for c in bp.CLASSES]].to_numpy()
    guess = probs.argmax(1)
    truth = pred.label.to_numpy()
    shots = pred.shot.to_numpy()
    # Repeated splits: a shot's predictions from different splits stay together in the
    # bootstrap because they share the shot id.
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
    sub.add_parser("table", help="print ablation.json as Markdown")
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
    elif args.stage == "table":
        markdown(args)
    elif args.stage == "run":
        for name in args.rows:
            run_row(ROWS[name], args)
    else:
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
