"""Census the q-min regime rule over a shot list, and pin what it found.

`events.heuristics.qmin_regimes` is a threshold rule with two knobs that
decide almost everything it says - the Ip flat-top gate and the 500 ms
minimum band - and neither can be justified from a synthetic trace. This
script runs the rule over a real shot list through the features store and
writes what it found to
`tests/labelmaker/data/qmin_regimes_recommender_v1.json`, which
`tests/labelmaker/test_events_heuristics.py` reads on every run.

The number that matters most is the one for the variant that is NOT
shipped. `qmin > 0.95` for 500 ms anywhere in the record - the same rule
with the flat-top gate removed - fires on 497 of the 500 `recommender_v1`
shots, because the current ramp takes every discharge through every band on
its way up and back down. A gate that turns a 99.4% label into a 54% one is
not a detail of the rule, it IS the rule, and a comment saying so would not
fail when somebody removed it.

    PYTHONPATH=$PWD/src \\
        pixi run --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \\
        -e labelmaker python scripts/labelmaker/qmin_regime_census.py \\
        --shot-file /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/recommender_v1.txt \\
        --write

Without `--write` it prints and compares against the committed record,
exiting 1 on any difference - which is the mode to run after touching a
threshold, since the committed thresholds are checked against the module's
by the test suite and a change to either makes the record stale.

The store is READ-ONLY here: nothing in this script writes outside the
repository's own `tests/labelmaker/data`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from labelmaker.config import Paths
from labelmaker.events import heuristics
from labelmaker.features import store as feature_store

#: Where the committed record lives, and what the test reads.
RECORD = REPO / "tests" / "labelmaker" / "data" / "qmin_regimes_recommender_v1.json"

#: The shot list the shipped record was measured over.
DEFAULT_SHOT_FILE = Path(
    "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/recommender_v1.txt"
)


def _runs(mask) -> list[tuple[int, int]]:
    idx = np.flatnonzero(
        np.diff(np.concatenate([[0], np.asarray(mask).view(np.int8), [0]]))
    )
    return list(zip(idx[::2].tolist(), idx[1::2].tolist(), strict=True))


def ungated_fires(t_s, y) -> bool:
    """The rule WITHOUT the flat-top gate: `q > 0.95` for `QMIN_MIN_MS`.

    One band and not three, because that is the claim the gate exists to
    refute - "this shot spent half a second above the hybrid threshold" -
    and a shot wandering between 1.2 and 2.5 satisfies it without holding
    any single band for half a second.
    """
    t = np.asarray(t_s, dtype=np.float64).ravel()
    q = np.asarray(y, dtype=np.float64).ravel()
    with np.errstate(invalid="ignore"):
        mask = np.isfinite(q) & (q > heuristics.QMIN_HYBRID)
    least_s = heuristics.QMIN_MIN_MS * 1e-3
    return any(float(t[b - 1] - t[a]) >= least_s for a, b in _runs(mask))


def census(shots, paths: Paths) -> dict:
    """Run the rule, gated and ungated, over `shots`. Reads only."""
    bands = [name for name, _lo, _hi in heuristics.QMIN_BANDS]
    gated = dict.fromkeys(bands, 0)
    ungated_by_band = dict.fromkeys(bands, 0)
    n_ungated = 0
    missing = {"features": [], "ip": [], "qmin": [], "flattop": []}
    per_shot: dict[str, list[str]] = {}
    started = time.monotonic()
    for shot in shots:
        path = paths.features_file(shot)
        if not path.exists():
            missing["features"].append(int(shot))
            continue
        try:
            qmin = feature_store.read_feature(path, "qmin")
        except KeyError:
            missing["qmin"].append(int(shot))
            continue
        if ungated_fires(qmin.x, qmin.y[0]):
            n_ungated += 1
        loose = heuristics.qmin_regimes(
            qmin.x, qmin.y[0], (float(qmin.x[0]), float(qmin.x[-1])), shot=shot,
        )
        for name in {e.phenomenon for e in loose}:
            ungated_by_band[name] += 1
        try:
            ip = feature_store.read_feature(path, "ip")
        except KeyError:
            missing["ip"].append(int(shot))
            continue
        flattop = heuristics.ip_flattop(ip.x, ip.y[0])
        if not np.isfinite(flattop).all():
            missing["flattop"].append(int(shot))
            continue
        found = sorted({
            e.phenomenon
            for e in heuristics.qmin_regimes(qmin.x, qmin.y[0], flattop,
                                             shot=shot)
        })
        per_shot[str(int(shot))] = found
        for name in found:
            gated[name] += 1
    n = len(shots)
    return {
        "note": (
            "Written by scripts/labelmaker/qmin_regime_census.py. The "
            "`ungated` block is the rule with the Ip flat-top gate REMOVED "
            "and is the justification for having the gate at all."
        ),
        "shot_file": "",
        "n_shots": n,
        "features_root": "",
        "thresholds": {
            "bands": [[name, lo, hi] for name, lo, hi in heuristics.QMIN_BANDS],
            "min_ms": heuristics.QMIN_MIN_MS,
            "flattop_frac": heuristics.FLATTOP_FRAC,
            "efit": heuristics.QMIN_EFIT,
        },
        "gated": dict(sorted(gated.items())),
        "ungated": {
            "any_band": n_ungated,
            "fraction": round(n_ungated / n, 4) if n else 0.0,
            "by_band": dict(sorted(ungated_by_band.items())),
        },
        "skipped": {k: v for k, v in missing.items() if v},
        "n_skipped": {k: len(v) for k, v in missing.items()},
        "seconds": round(time.monotonic() - started, 1),
        "per_shot": per_shot,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shot-file", type=Path, default=DEFAULT_SHOT_FILE)
    parser.add_argument("--root", type=Path, default=None,
                        help="labelmaker root holding features/ (default: "
                             "LABELMAKER_ROOT)")
    parser.add_argument("--write", action="store_true",
                        help=f"rewrite {RECORD.name} instead of comparing")
    args = parser.parse_args(argv)

    paths = Paths.from_env()
    if args.root:
        paths = Paths(root=args.root, corpus=paths.corpus,
                      text_root=paths.text_root, logs_jsonl=paths.logs_jsonl)
    shots = [int(s) for s in args.shot_file.read_text().split()]
    record = census(shots, paths)
    record["shot_file"] = str(args.shot_file)
    record["features_root"] = str(paths.features)
    print(json.dumps({k: v for k, v in record.items() if k != "per_shot"},
                     indent=2))
    if args.write:
        RECORD.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"wrote {RECORD}")
        return 0
    if not RECORD.exists():
        print(f"no committed record at {RECORD}; run with --write",
              file=sys.stderr)
        return 1
    old = json.loads(RECORD.read_text())
    drift = [
        key for key in ("gated", "ungated", "thresholds", "per_shot",
                        "n_shots", "n_skipped")
        if old.get(key) != record.get(key)
    ]
    if drift:
        print(f"the committed record disagrees on {drift}", file=sys.stderr)
        return 1
    print("the committed record matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
