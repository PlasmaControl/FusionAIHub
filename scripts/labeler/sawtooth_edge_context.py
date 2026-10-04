"""TRAIN-only derivation of the isolated-edge veto length of the quiet-core test.

``isolated_edge_context_ms`` is how far around an isolated qualified core edge
the tested-absence label is withheld. The fix-round-4 value (the 5.15 ms frame
holdoff) was chosen on an exploration of 23 shots, three of which were val
(blind-queue) shots, and its provenance record was empty. This script derives
the value again from TRAIN cohort shots alone, by one stated rule, and writes
``edge_context_derivation.json`` for the packaged freeze and the setup script to
agree with.

Stages:
  run     label the TRAIN cohort shots into a separate work directory (the
          tested-absence mask with only the frame-holdoff veto, ``absent_holdoff``,
          is stored per shot; any longer veto follows from it exactly);
  derive  tested-absent seconds, retained fraction and nearby isolated edges for
          each candidate length, and the chosen length.

Rule. Candidates are the frame holdoff (5.15 ms), 50 ms and the full +/-375 ms
absence-test context. The chosen length is the longest candidate that keeps at
least half of the TRAIN tested-absent time left by the frame holdoff: a longer
veto is the more conservative quiet-core definition, and the limit stops it
removing most of the negatives. The threshold of one half is a stated trade-off,
not a quantity the data fixes; the sensitivity table below shows every
candidate, and the benchmark repeats it on the scored negatives.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sawtooth_physics import FS, OUTPUT, REPO, WORK, process_shot, save_json

from labeler.sawtooth.metrics import spans_at
from labeler.sawtooth.physics import apply_edge_context

DERIVATION_WORK = WORK.parent / "fix5_edge_context_derivation"
CANDIDATES_MS = (5.15, 50.0, 375.0)
MINIMUM_RETAINED = 0.5
REACH_MS = 375.0
CRITERION = (
    "longest candidate isolated-edge veto (5.15 ms frame holdoff, 50 ms, 375 ms "
    "full absence-test context) that keeps at least half of the TRAIN tested-absent "
    "time left by the frame holdoff; TRAIN cohort shots only, no val or test shot"
)
# The shots of the fix-round-4 exploration that set the earlier value. The
# earlier freeze record listed none and called it a 17-shot TRAIN draw plus six
# train/val shots. This is that set of 23: a seeded draw of 17 TRAIN shots plus
# 186532, 192148, 203563, 186636, 190637, 189324 and 191384 (one of the seven is
# also in the draw).
PREVIOUS_EXPLORATION_SHOTS = (
    185838, 186532, 186636, 186981, 189324, 189541, 190602, 190637, 191384,
    191448, 191519, 192148, 194410, 195187, 195910, 198899, 200704, 201991,
    202028, 203136, 203174, 203530, 203563,
)  # fmt: skip


def split_map():
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    return {int(s): str(p) for s, p in zip(cohort.shot, cohort.split, strict=True)}


def run(args):
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    train = cohort[cohort.split == "train"]
    jobs = [
        (
            int(r.shot),
            str(args.work),
            True,
            (r.window_start_ms / 1000, r.window_end_ms / 1000),
        )
        for r in train.itertuples()
    ]
    if args.reverse:
        # A second worker starting from the far end meets the first one; cached
        # shots return at once, so the overlap costs nothing.
        jobs.reverse()
    begun = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(pool.map(process_shot, jobs, chunksize=1)):
            if i % 50 == 0:
                print(
                    f"{i + 1}/{len(jobs)} shot {record['shot']} "
                    f"error {record.get('error', '')}",
                    flush=True,
                )
    print(json.dumps({"shots": len(jobs), "seconds": time.monotonic() - begun}))


def tested_absence(signal, record, context_ms):
    """Tested-absent mask under one veto length, as the state partition builds it."""
    t = signal["t"]
    edges = record["absence_diagnostics"]["core_relaxation_test"][
        "ambiguous_edge_times_s"
    ]
    absent = apply_edge_context(t, signal["absent_holdoff"], edges, context_ms)
    present = spans_at(
        t,
        [
            (r["start_s"], r["end_s"])
            for r in record["states"]
            if r["state"] == "present"
        ],
    )
    doubt = spans_at(
        t, [(r["start_s"], r["end_s"]) for r in record["uncertain_intervals"]]
    )
    return np.asarray(signal["observable"], dtype=bool) & ~doubt & ~present & absent


def edges_near(t, edges, absent, reach_s):
    """Edges with at least one tested-absent sample within ``reach_s``."""
    csum = np.r_[0, np.cumsum(absent)]
    count = 0
    for edge in edges:
        lo, hi = np.searchsorted(t, [edge - reach_s, edge + reach_s])
        count += bool(csum[min(len(t), hi + 1)] - csum[lo])
    return count


def choose_context(rows, minimum_retained=MINIMUM_RETAINED):
    """Longest candidate keeping at least ``minimum_retained`` of the shortest's time."""
    rows = sorted(rows, key=lambda row: row["context_ms"])
    base = rows[0]["tested_absent_s"]
    for row in rows:
        row["retained_fraction"] = row["tested_absent_s"] / base if base else None
        row["passes"] = bool(base) and row["retained_fraction"] >= minimum_retained
    passing = [row["context_ms"] for row in rows if row["passes"]]
    if not passing:
        raise ValueError("no candidate keeps the required tested-absent time")
    return max(passing)


def derive(args):
    splits = split_map()
    shots = sorted(int(p.stem) for p in (args.work / "shots").glob("*.json"))
    outside = [s for s in shots if splits.get(s) != "train"]
    if outside:
        raise ValueError(f"derivation work holds non-TRAIN shots: {outside[:5]}")
    totals = {c: {"seconds": 0.0, "near": 0, "shots": 0} for c in CANDIDATES_MS}
    used, edge_total = [], 0
    for shot in shots:
        record = json.loads((args.work / "shots" / f"{shot}.json").read_text())
        path = args.work / "signals" / f"{shot}.npz"
        if "error" in record or not path.exists():
            continue
        with np.load(path) as arrays:
            signal = {k: arrays[k] for k in ("t", "observable", "absent_holdoff")}
        edges = record["absence_diagnostics"]["core_relaxation_test"][
            "ambiguous_edge_times_s"
        ]
        used.append(shot)
        edge_total += len(edges)
        for context in CANDIDATES_MS:
            absent = tested_absence(signal, record, context)
            totals[context]["seconds"] += float(absent.sum()) / FS
            totals[context]["near"] += edges_near(
                signal["t"], edges, absent, REACH_MS / 1000
            )
            totals[context]["shots"] += bool(absent.any())
    rows = [
        {
            "context_ms": context,
            "tested_absent_s": totals[context]["seconds"],
            "shots_with_tested_absence": totals[context]["shots"],
            "isolated_edges_within_375ms_of_tested_absence": totals[context]["near"],
        }
        for context in CANDIDATES_MS
    ]
    chosen = choose_context(rows)
    exploration = [
        {"shot": s, "split": splits.get(s)} for s in PREVIOUS_EXPLORATION_SHOTS
    ]
    record = {
        "criterion": CRITERION,
        "minimum_retained_fraction": MINIMUM_RETAINED,
        "candidates_ms": list(CANDIDATES_MS),
        "chosen_ms": chosen,
        "rows": rows,
        "isolated_qualified_edges_total": edge_total,
        "derivation_split": "train",
        "derivation_shots": used,
        "derivation_shot_count": len(used),
        "val_or_test_shots_read": 0,
        "previous_exploration": {
            "note": (
                "fix-round-4 value 5.15 ms came from this exploration of 23 shots; "
                "three were val (blind-queue) shots, so the value is re-derived "
                "above on TRAIN shots only and the earlier choice is not carried "
                "forward"
            ),
            "shots": exploration,
            "split_counts": {
                split: sum(row["split"] == split for row in exploration)
                for split in sorted({row["split"] for row in exploration})
            },
        },
        "work": str(args.work),
        "trade_off": (
            "the retained-fraction limit of one half is a stated trade-off, not "
            "independent of the outcome it protects (the number of negatives); "
            "the candidate rows are the full sensitivity on TRAIN"
        ),
    }
    save_json(OUTPUT / "edge_context_derivation.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "derivation_shots"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "derive"])
    parser.add_argument("--work", type=Path, default=DERIVATION_WORK)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    {"run": run, "derive": derive}[args.stage](args)


if __name__ == "__main__":
    main()
