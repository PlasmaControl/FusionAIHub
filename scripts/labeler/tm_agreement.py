#!/usr/bin/env python
"""How the whole-interval tearing-mode labels agree with the lab's onset labels.

Two references, each an onset per shot:

* ``survival``: ``raw/tm_labels.h5`` (the survival-model labels, 20 ms; the first
  sample above zero is the onset: n = 1 RMS above 12 G for 50 ms in H-mode, onset at a
  tenth of the peak). Needs nothing but the file.
* ``seo``: Seo's tearing archive (``tm_label`` rows, 25 ms), placed in time by
  labeler's own row match (``validate.archived_truth``), which needs the shot's feature
  file under the labeler root named by ``LABELER_ROOT``; a shot the match rejects is
  left out and counted.

Writes ``<out>/agreement_<reference>_<set>.json`` and a per-onset / per-interval CSV.

    PYTHONPATH=$PWD/src pixi run --frozen --no-install -e labelmaker python \\
        scripts/labeler/tm_agreement.py --reference survival
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths, git_sha
from labeler.tearing import agreement, rule
from labeler.tearing.agreement import Reference

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
OUT_ROOT = Path(os.environ.get("TM_OUT_ROOT", LABELER / "round4/tm"))
CATALOG = REPO / "data/events/catalog"
SURVIVAL_H5 = REPO / "data/events/neoclassical_tearing_mode/raw/tm_labels.h5"
MAIN_CHECKOUT = Path("/scratch/gpfs/nc1514/FusionAIHub")


def survival_references(shots) -> tuple[list[Reference], list[int]]:
    """`(references, absent)`: the survival label's onset per shot it holds."""
    path = (
        SURVIVAL_H5
        if SURVIVAL_H5.is_file()
        else (MAIN_CHECKOUT / "data/events/neoclassical_tearing_mode/raw/tm_labels.h5")
    )
    refs, absent = [], []
    with h5py.File(path, "r") as f:
        for shot in shots:
            key = str(int(shot))
            if key not in f:
                absent.append(int(shot))
                continue
            label = f[key]["label"][:]
            time = f[key]["time"][:].astype(float)
            on = np.flatnonzero(label > 0)
            onset = float(time[on[0]]) if on.size else None
            step = float(np.median(np.diff(time))) if len(time) > 1 else 20.0
            refs.append(
                Reference(int(shot), onset, float(time[0]), float(time[-1] + step))
            )
    return refs, absent


def seo_references(shots, paths: Paths) -> tuple[list[Reference], dict]:
    """`(references, skipped)`: Seo's first tearing row per shot the match places."""
    from labeler.validate import archived_truth

    refs, skipped = [], {}
    for shot in shots:
        truth = archived_truth(int(shot), paths)
        if not truth.get("available"):
            skipped[int(shot)] = truth.get("reason", "unavailable")
            continue
        t_ms = np.asarray(truth["t"], dtype=float) * 1000.0
        onset = None if truth["onset_s"] is None else float(truth["onset_s"]) * 1000.0
        refs.append(Reference(int(shot), onset, float(t_ms.min()), float(t_ms.max())))
    return refs, skipped


def miss_reasons(onsets: pd.DataFrame, intervals: pd.DataFrame, table, signals: Path):
    """The `agreement.REASONS` entry for each missed onset, as a list in row order."""
    windows = table.set_index("shot")[["window_start_ms", "window_end_ms"]]
    starts = intervals[intervals.n == 1].groupby("shot").t_start.apply(list).to_dict()
    start_file = signals.parent / "labels/plasma_start_cohort.json"
    plasma_starts = json.loads(start_file.read_text()) if start_file.is_file() else {}
    out = []
    for row in onsets.itertuples(index=False):
        if row.matched:
            out.append("")
            continue
        path = signals / f"{int(row.shot)}.npz"
        if not path.is_file():
            out.append("no_record")
            continue
        with np.load(path) as npz:
            t_ms, n1 = npz["t_ms"], npz["n1rms"]
        win = windows.loc[int(row.shot)]
        window = (float(win.window_start_ms), float(win.window_end_ms))
        plasma_start = plasma_starts.get(str(int(row.shot)), {}).get(
            "start_ms", window[0]
        )
        if window[0] <= row.onset_ms < plasma_start:
            out.append("excluded_ramp_up")
            continue
        reason = agreement.miss_reason(
            t_ms,
            n1,
            row.onset_ms,
            (plasma_start, window[1]),
            starts.get(int(row.shot), []),
        )
        if reason == "short_burst" and any(
            item.end_ms >= row.onset_ms - agreement.TOLERANCE_MS
            and item.start_ms <= row.onset_ms + 300.0
            for item in rule.mode_intervals(
                t_ms, n1, rule.N1_RULE, (plasma_start, window[1])
            )
        ):
            reason = "coherent_line_not_supported"
        out.append(reason)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", choices=("survival", "seo"), required=True)
    ap.add_argument(
        "--from", dest="source", choices=("cohort", "population"), default="cohort"
    )
    ap.add_argument(
        "--intervals",
        type=Path,
        default=None,
        help="the per-interval table (default: round4/tm/labels)",
    )
    ap.add_argument(
        "--exclude-test",
        action="store_true",
        help="leave out the cohort's blind test shots",
    )
    ap.add_argument("--tol-ms", type=float, default=agreement.TOLERANCE_MS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--features-root", type=Path, default=OUT_ROOT / "lroot")
    args = ap.parse_args(argv)

    full = args.intervals or (
        OUT_ROOT / "labels" / f"tm_intervals_full_{args.source}.csv"
    )
    intervals = pd.read_csv(full)
    cohort = pd.read_csv(CATALOG / "cohort.csv")
    table = pd.read_csv(CATALOG / f"{args.source}.csv")
    shots = [int(s) for s in table.shot]
    test = set(cohort[cohort.split == "test"].shot)
    shots = [s for s in shots if s not in test]
    args.exclude_test = True
    if args.reference == "survival":
        refs, absent = survival_references(shots)
        notes = {"not_in_reference": len(absent)}
    else:
        refs, skipped = seo_references(
            shots, replace(Paths.from_env(), root=args.features_root)
        )
        notes = {
            "skipped": len(skipped),
            "skipped_reasons_first_10": dict(list(skipped.items())[:10]),
        }
    # Intervals can only be compared where the labels were made.
    labelled = set(intervals.shot) | set(table.shot)
    refs = [r for r in refs if r.shot in labelled]
    both, strict = {}, {}
    for name, ns in (("n1", (1,)), ("any_n", None)):
        onsets, compared = agreement.compare_onsets(
            refs, intervals, tol_ms=args.tol_ms, ns=ns
        )
        both[name] = agreement.summarize(onsets, compared, refs)
        strict_onsets, strict_intervals = agreement.compare_onsets(
            refs, intervals, tol_ms=0.0, ns=ns
        )
        strict[name] = agreement.summarize(strict_onsets, strict_intervals, refs)
        if name == "n1" and len(onsets):
            onsets["reason"] = miss_reasons(
                onsets, intervals, table, OUT_ROOT / "signals"
            )
            both[name]["missed_reasons"] = {
                key: int((onsets.reason == key).sum())
                for key in (
                    *agreement.REASONS,
                    "no_record",
                    "excluded_ramp_up",
                    "coherent_line_not_supported",
                )
            }
        stem = f"{args.reference}_{args.source}_{name}{args.tag}"
        out = OUT_ROOT / "agreement"
        out.mkdir(parents=True, exist_ok=True)
        onsets.to_csv(out / f"onsets_{stem}.csv", index=False)
        compared.to_csv(out / f"intervals_{stem}.csv", index=False)
    record = {
        "reference": args.reference,
        "shots_set": args.source,
        "exclude_test": bool(args.exclude_test),
        "tolerance_ms": args.tol_ms,
        "n_shots_considered": len(shots),
        "git_sha": git_sha(),
        "intervals_table": str(full),
        "intervals_sha256": hashlib.sha256(full.read_bytes()).hexdigest(),
        "caveat": "Survival agreement shares N1RMS, 12 G / 50 ms and "
        "10%-of-peak onset with this rule; it is near-circular and does not "
        "independently validate tearing islands."
        if args.reference == "survival"
        else "Seo labels are growth-phase labels with limited temporal coverage.",
        "unmatched_interval_breakdown": {
            "definition": "n1 intervals on selected shots: compare only starts "
            "inside available archive coverage; other spans are not comparable",
            "all_n1_intervals": int(
                intervals[intervals.shot.isin(shots) & intervals.n.eq(1)].shape[0]
            ),
            "outside_reference_coverage_or_no_reference": int(
                intervals[intervals.shot.isin(shots) & intervals.n.eq(1)].shape[0]
            )
            - both["n1"]["compared_intervals"],
            "reference_mode_shots": both["n1"].get(
                "intervals_without_an_onset_on_reference_mode_shots", 0
            ),
            "reference_quiet_shots": both["n1"].get(
                "intervals_without_an_onset_on_reference_quiet_shots", 0
            ),
        },
        **notes,
        "agreement": both,
        "agreement_strict": strict,
        "agreement_strict_tolerance_ms": 0.0,
    }
    path = (
        OUT_ROOT
        / "agreement"
        / f"agreement_{args.reference}_{args.source}{args.tag}.json"
    )
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(both["n1"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
