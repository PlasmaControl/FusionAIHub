#!/usr/bin/env python
"""Score the RWM baselines by nested shot-grouped cross-validation with shot bootstraps.

Reads the slice table written by ``rwm_build.py`` and scores each configuration of
`CONFIGS`: `rwm-brf` (Piccione et al. 2022's balanced random forest, adapted), `rwm-nnpu`
(the same features under a non-negative positive-unlabelled risk), the beta_N / l_i
ranking and the `rwm_candidates` screen as rule baselines, feature-group ablations, and the
horizon and prior sensitivities. Every number carries a 95 % interval from 1000 shot
bootstraps (`labeler.rwm.metrics.shot_bootstrap`, Hanson and comparison shots resampled
as separate strata); the thresholds are chosen on inner out-of-fold scores of the
training shots only. Writes `outputs/labeler/rwm/evaluation.json`.

    python scripts/labeler/rwm_evaluate.py [--replicates 1000] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths
from labeler.events import rwm
from labeler.rwm import evaluate as ev
from labeler.rwm import features, labels, metrics

#: The locked-mode detector (DUSBRADIAL) is exactly zero on every 2014 shot and nonzero on
#: the 2018 ones, so as an input it tags the campaign; it is fetched and stored but kept
#: out of the model, and `rwm-brf-with-locked-mode` shows what it does when it is let in.
ALL = tuple(f for f in features.FEATURES if f != "lock_v")
EQUILIBRIUM = (
    "betan",
    "li",
    "q95",
    "qmin",
    "wmhd_mj",
    "betan_over_li",
    "betan_minus_4li",
    "ip_ma",
)
MAGNETICS = ("n1rms_g", "n1rms_max_g", "n1rms_growth_per_s", "n2rms_g")
#: Small and regularised: a few hundred positive slices from 48 events overfit a wide
#: network (train AUROC 0.98 against 0.7 held out in a trial fold). The logistic loss
#: replaces Kiryo et al.'s sigmoid loss, which saturates at a class prior of 0.006.
NETWORK = {
    "hidden": (32, 32),
    "epochs": 15,
    "batch_p": 64,
    "batch_u": 512,
    "weight_decay": 1e-3,
    "loss": "logistic",
}
FOREST = {"n_estimators": 300, "max_depth": 8, "min_leaf": 5}
SEED = 0

#: name -> (model, horizon_ms, options). `training` and `target` are written to the JSON.
CONFIGS = {
    "rwm-brf": {"kind": "brf", "columns": ALL, "comparison": True, "horizon": 100.0},
    "rwm-brf-hanson-only": {
        "kind": "brf",
        "columns": ALL,
        "comparison": False,
        "horizon": 100.0,
    },
    "rwm-brf-equilibrium-only": {
        "kind": "brf",
        "columns": EQUILIBRIUM,
        "comparison": True,
        "horizon": 100.0,
    },
    "rwm-brf-magnetics-only": {
        "kind": "brf",
        "columns": MAGNETICS,
        "comparison": True,
        "horizon": 100.0,
    },
    "rwm-brf-no-rotation": {
        "kind": "brf",
        "columns": tuple(c for c in ALL if not c.startswith("rot_")),
        "comparison": True,
        "horizon": 100.0,
    },
    "rwm-brf-with-locked-mode": {
        "kind": "brf",
        "columns": tuple(features.FEATURES),
        "comparison": True,
        "horizon": 100.0,
    },
    "rwm-brf-horizon-50": {
        "kind": "brf",
        "columns": ALL,
        "comparison": True,
        "horizon": 50.0,
    },
    "rwm-brf-horizon-200": {
        "kind": "brf",
        "columns": ALL,
        "comparison": True,
        "horizon": 200.0,
    },
    "rwm-brf-all-modes": {
        "kind": "brf",
        "columns": ALL,
        "comparison": True,
        "horizon": 100.0,
        "all_modes": True,
    },
    "rwm-nnpu": {"kind": "nnpu", "columns": ALL, "horizon": 100.0, "prior_scale": 1.0},
    "rwm-nnpu-prior-x0.5": {
        "kind": "nnpu",
        "columns": ALL,
        "horizon": 100.0,
        "prior_scale": 0.5,
    },
    "rwm-nnpu-prior-x2": {
        "kind": "nnpu",
        "columns": ALL,
        "horizon": 100.0,
        "prior_scale": 2.0,
    },
    "rule-betan-over-li": {"kind": "rule", "column": "betan_over_li", "horizon": 100.0},
    "rule-rwm-candidates": {
        "kind": "rule",
        "column": "rule_candidate",
        "binary": True,
        "horizon": 100.0,
    },
}
#: Fold seeds for the split-sensitivity rerun of the headline configuration.
SPLIT_SEEDS = (1, 2, 3, 4)


def clean(value):
    """The value with NaN and infinity as None, which JSON can hold."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def factory(config):
    kind = config["kind"]
    if kind == "brf":
        return lambda seed: ev.Brf(
            config["columns"], use_comparison=config["comparison"], seed=seed, **FOREST
        )
    if kind == "nnpu":
        return lambda seed: ev.Nnpu(
            config["columns"], prior_scale=config["prior_scale"], seed=seed, **NETWORK
        )
    return lambda seed: ev.Rule(config["column"], binary=config.get("binary", False))


_STATE: dict = {}


def _init(slices_path, replicates):
    import torch

    torch.set_num_threads(1)
    paths = Paths.from_env()
    table = rwm.onset_table(paths)
    onsets, other = {}, {}
    for row in table.itertuples():
        (onsets if row.ntor == 1 else other).setdefault(int(row.shot), []).append(
            float(row.t_ms)
        )
    onsets = {s: labels.merge_close(v) for s, v in onsets.items()}
    other = {s: labels.merge_close(v) for s, v in other.items()}
    _STATE.update(
        slices=pd.read_parquet(slices_path),
        onsets=onsets,
        other=other,
        replicates=replicates,
    )


def run_config(args):
    name, config, seed = args
    slices, onsets, other = _STATE["slices"], _STATE["onsets"], _STATE["other"]
    target = (
        {
            s: sorted(onsets.get(s, []) + other.get(s, []))
            for s in set(onsets) | set(other)
        }
        if config.get("all_modes")
        else onsets
    )
    excluded = {} if config.get("all_modes") else other
    table = ev.relabel(slices, target, excluded, config["horizon"])
    every = {
        s: sorted(onsets.get(s, []) + other.get(s, []))
        for s in set(onsets) | set(other)
    }
    oof, alarms, rules = ev.cross_validate(table, factory(config), every, seed=seed)
    groups = ev.shot_records(oof, alarms, target, every)
    result = {
        "model": config["kind"],
        "horizon_ms": config["horizon"],
        "options": {k: v for k, v in config.items() if k not in ("kind", "horizon")},
        "fold_seed": seed,
        "counts": ev.counts(groups),
        "rules_by_fold": rules,
    }
    if seed == SEED:
        result["metrics"] = metrics.shot_bootstrap(
            groups, ev.statistic, replicates=_STATE["replicates"], seed=SEED
        )
        for year in sorted({r["campaign"] for g in groups.values() for r in g}):
            part = {
                k: [r for r in v if r["campaign"] == year] for k, v in groups.items()
            }
            result.setdefault("by_campaign", {})[str(year)] = {
                "counts": ev.counts(part),
                "metrics": ev.statistic(part),
            }
    else:
        result["metrics"] = ev.statistic(groups)
    return name, seed, clean(json.loads(json.dumps(result, default=float)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slices", type=Path)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", nargs="*", help="configuration names to run")
    parser.add_argument(
        "--out", type=Path, default=REPO / "outputs/labeler/rwm/evaluation.json"
    )
    args = parser.parse_args()
    paths = Paths.from_env()
    slices_path = args.slices or paths.root / "round4" / "rwm" / "slices.parquet"
    names = args.only or list(CONFIGS)
    jobs = [(n, CONFIGS[n], SEED) for n in names]
    if "rwm-brf" in names:
        jobs += [("rwm-brf", CONFIGS["rwm-brf"], s) for s in SPLIT_SEEDS]
    with Pool(
        args.workers, initializer=_init, initargs=(slices_path, args.replicates)
    ) as pool:
        done = pool.map(run_config, jobs, chunksize=1)
    results: dict = {}
    if args.only and args.out.is_file():
        results = json.loads(args.out.read_text()).get("configs", {})
    for name, seed, result in done:
        if seed == SEED:
            results[name] = result
        else:
            results[name].setdefault("split_seeds", {})[str(seed)] = result["metrics"]
    _init(slices_path, 1)
    slices = _STATE["slices"]
    record = {
        "script": "scripts/labeler/rwm_evaluate.py",
        "protocol": {
            "outer_folds": ev.OUTER_FOLDS,
            "inner_folds": ev.INNER_FOLDS,
            "fold_seed": SEED,
            "bootstrap_replicates": args.replicates,
            "forest": FOREST,
            "network": {
                k: list(v) if isinstance(v, tuple) else v for k, v in NETWORK.items()
            },
            "alarm_grid": {
                "high_quantiles": ev.HIGH_QUANTILES,
                "low_fractions": ev.LOW_FRACTIONS,
                "hold_ms": ev.HOLD_MS,
            },
            "step_ms": features.STEP_MS,
            "split": "shot-grouped; shots.json cohort_overlap counts the shots in the frozen cohort",
        },
        "configs": results,
        "single_feature_auroc": ev.single_feature_auroc(
            ev.relabel(slices, _STATE["onsets"], _STATE["other"], 100.0)
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(clean(record), indent=2, default=float, allow_nan=False) + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
