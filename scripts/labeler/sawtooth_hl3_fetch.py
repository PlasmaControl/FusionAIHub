"""Fetch the two HL-3 full-input signals the local store lacks.

Run on the login node under the fdp wrapper, at most three workers, one second
between fetches, and stop at the first authentication error (the owner's login has
lapsed; the run then continues with what is on disk):

    LABELER_RAW_CACHE=$LABELER_ROOT/round4/hl3/raw \\
    pixi run --frozen -e labelmaker fdp run python \\
        scripts/labeler/sawtooth_hl3_fetch.py --workers 3 --pace 1

Only the shots the benchmark uses are fetched (its train folds, the supported
fixed-validation shots and the three reviewed shots); the blind test split is never
read. Two signals, each only where the local store has none:

* stored energy: EFIT01 WMHD (joules, 20 ms cadence) through the repo's `wmhd`
  feature route, parked in the raw-cache layout under `LABELER_RAW_CACHE`
  (`round4/hl3/raw`, never the shared cache or the corpus);
* beam power on the shots whose corpus `pinj` group is a stub (the per-beam MDSplus
  nodes hold nothing there): PTDATA `BMSPINJ`, a total in MW stored as volts, as the
  detachment stream measured (its calibration against summed beams is within 3 %),
  parked as a 1 ms block mean in `round4/hl3/bms/<shot>.npz` (seconds, watts). Shots
  already holding it in the detachment stream's cache are not fetched again.

The ledger is `outputs/labeler/sawtooth/fix5/hl3_full_fetch.json`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
from detach_fetch import block_mean, corpus_is_stub, is_auth_error
from sawtooth_physics import OUTPUT, REPO, save_json

from labeler.config import Paths
from labeler.events import equilibrium
from labeler.events.verify import NoDataError
from labeler.features import resolve_fdp

ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/hl3"
BMS = ROOT / "bms"
STOP = ROOT / "fetch_auth_stop"
LOG = ROOT / "fetch_log.jsonl"
DETACH_CACHE = Path(os.environ["LABELER_ROOT"]) / "round4/detach/cache"
BMS_BLOCK_MS = 1.0
MAX_CONSECUTIVE_UNKNOWN = 3
#: (shot, name) pairs an earlier call found absent upstream: not asked for again.
ABSENT: set = set()


def population():
    """The shots the benchmark reads: train folds, supported validation, reviewed."""
    path = REPO / "outputs/labeler/sawtooth/fix5/split_manifest.json"
    split = json.loads(path.read_text())
    shots = set(split["training_cohort"]) | set(split["fixed_validation_supported"])
    shots |= set(split["expert_shots"])
    forbidden = set(split["blind_test_excluded"])
    if shots & forbidden:
        raise ValueError("the blind test split must never be fetched")
    return sorted(shots)


def has_bms_locally(shot):
    path = DETACH_CACHE / f"{shot}.npz"
    if not path.is_file():
        return False
    with np.load(path) as data:
        return "pinj_bms__y" in data.files


def needs(shot, paths):
    """What is missing locally for this shot: a subset of ('wmhd', 'bms')."""
    out = []
    try:
        equilibrium.signal(shot, "wmhd", paths, fetch=False)
    except NoDataError:
        out.append("wmhd")
    if (
        corpus_is_stub(shot, "pinj")
        and not has_bms_locally(shot)
        and not (BMS / f"{shot}.npz").is_file()
    ):
        out.append("bms")
    return [name for name in out if (shot, name) not in ABSENT]


def fetch_bms(shot):
    record = resolve_fdp._fetch_ptdata("BMSPINJ", shot)
    t = np.asarray(record["times"], dtype=float)
    y = np.asarray(record["data"], dtype=float)
    if len(t) < 2 or len(t) != len(y) or not np.isfinite(y).any():
        raise ValueError("BMSPINJ holds no usable series")
    t, y = block_mean(t, y, BMS_BLOCK_MS)
    BMS.mkdir(parents=True, exist_ok=True)
    temporary = BMS / f".{shot}.tmp.npz"
    np.savez(
        temporary, t_s=(t / 1000).astype(np.float64), w=(y * 1e6).astype(np.float32)
    )
    temporary.replace(BMS / f"{shot}.npz")
    return f"ok {len(y)} blocks, max {np.nanmax(y):.2f} MW"


def work_one(args):
    shot, pace = args
    row = {"shot": shot}
    if STOP.exists():
        return {**row, "status": "skipped_after_auth_stop", "auth": True}
    paths = Paths.from_env()
    for name in needs(shot, paths):
        try:
            if name == "wmhd":
                array = equilibrium.signal(shot, "wmhd", paths, fetch=True)
                row[name] = f"ok {len(array.x)} samples"
            else:
                row[name] = fetch_bms(shot)
        except Exception as error:  # noqa: BLE001  an absent node is data
            text = f"{type(error).__name__}: {error}"[:240]
            row[name] = "error " + text
            if is_auth_error(text):
                STOP.touch()
                row["auth"] = True
                return row
            row["unknown_error"] = not isinstance(error, (NoDataError, ValueError))
        time.sleep(pace)
    row.setdefault(
        "status", "done" if any(k in row for k in ("wmhd", "bms")) else "local"
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("at most three workers")
    if os.environ.get("LABELER_RAW_CACHE") != str(ROOT / "raw"):
        parser.error(f"set LABELER_RAW_CACHE={ROOT / 'raw'}")
    ROOT.mkdir(parents=True, exist_ok=True)
    shots = population()
    paths = Paths.from_env()
    if LOG.is_file():
        for line in LOG.read_text().splitlines():
            row = json.loads(line)
            for name in ("wmhd", "bms"):
                if row.get(name, "").startswith(
                    ("error NoDataError", "error ValueError")
                ):
                    ABSENT.add((row["shot"], name))
    pending = [shot for shot in shots if needs(shot, paths)]
    if args.limit:
        pending = pending[: args.limit]
    rows, unknown_streak = [], 0
    # toksearch's ptserver reader is not fork-safe: fork the pool before any fetch.
    context = multiprocessing.get_context("fork")
    with context.Pool(args.workers) as pool:
        for row in pool.imap_unordered(work_one, [(s, args.pace) for s in pending]):
            rows.append(row)
            with LOG.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if row.get("auth"):
                print("authentication error: stopping all fetching", flush=True)
                pool.terminate()
                break
            unknown_streak = unknown_streak + 1 if row.get("unknown_error") else 0
            if unknown_streak >= MAX_CONSECUTIVE_UNKNOWN:
                print("three unexplained failures in a row: stopping", flush=True)
                pool.terminate()
                break
    # The ledger is the whole run history: the last row per shot over every call.
    history = {}
    for line in LOG.read_text().splitlines():
        row = json.loads(line)
        history[row["shot"]] = row
    still = [shot for shot in shots if needs(shot, paths)]
    counts = {}
    for row in history.values():
        for key in ("wmhd", "bms"):
            if key in row:
                state = row[key].split()[0]
                counts[f"{key}_{state}"] = counts.get(f"{key}_{state}", 0) + 1
    summary = {
        "shots_considered": len(shots),
        "shots_fetched_or_tried": len(history),
        "still_missing_locally": len(still),
        "still_missing_shots": still,
        "blind_test_fetched": False,
        "workers": args.workers,
        "pace_s": args.pace,
        "authentication_stop": any(r.get("auth") for r in history.values()),
        "counts": counts,
        "rows": [history[shot] for shot in sorted(history)],
    }
    save_json(OUTPUT / "hl3_full_fetch.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    return 3 if summary["authentication_stop"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
