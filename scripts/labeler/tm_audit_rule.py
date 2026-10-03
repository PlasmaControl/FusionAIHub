#!/usr/bin/env python
"""Audit continuous seed holds and interval changes after a frozen TM rule fix.

Change counts compare (shot,n,start,end) rounded to 1 microsecond. Attribute-only
changes are counted separately. Changed original intervals include removed and
modified spans; new non-overlapping spans are additionally reported. An independent
seed audit uses the raw and 5 ms-median RMS, never joining above-threshold runs or
acquisition gaps. The selected cohort review shots also list their final categories.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from labeler.config import git_sha
from labeler.tearing.magfeatures import FEATURE_NAMES

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
OUT = ROOT / "round4/tm"
REVIEWED = (185953, 194410, 186561, 190790, 195040, 196494, 187072)


def longest_run(t, mask):
    """Largest continuous true duration, including one median-step sample width."""
    if len(t) < 2:
        return 0.0
    dt = float(np.median(np.diff(t)))
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    longest = 0.0
    for a, b in zip(
        np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
    ):
        cuts = np.r_[a, np.flatnonzero(np.diff(t[a:b]) > dt * 1.5) + a + 1, b]
        for lo, hi in pairwise(cuts):
            longest = max(longest, float(t[hi - 1] - t[lo] + dt))
    return longest


def key(row):
    return (int(row.shot), int(row.n), round(row.t_start, 6), round(row.t_end, 6))


def changes(old, new):
    old_keys = {key(r): r for r in old.itertuples(index=False)}
    new_keys = {key(r): r for r in new.itertuples(index=False)}
    changed = []
    removed, modified = 0, 0
    for k, row in old_keys.items():
        if k in new_keys:
            continue
        overlaps = new[
            (new.shot == row.shot)
            & (new.n == row.n)
            & (new.t_start < row.t_end)
            & (new.t_end > row.t_start)
        ]
        state = "removed" if overlaps.empty else "modified"
        removed += state == "removed"
        modified += state == "modified"
        changed.append(
            {
                "old": list(k),
                "change": state,
                "overlapping_new": [
                    list(key(r)) for r in overlaps.itertuples(index=False)
                ],
            }
        )
    added = []
    for k, row in new_keys.items():
        overlaps = old[
            (old.shot == row.shot)
            & (old.n == row.n)
            & (old.t_start < row.t_end)
            & (old.t_end > row.t_start)
        ]
        if overlaps.empty:
            added.append(list(k))
    common_cols = sorted(
        set(old.columns)
        & set(new.columns) - {"shot", "n", "t_start", "t_end", "duration_ms"}
    )
    attr_changed = 0
    for k in set(old_keys) & set(new_keys):
        attr_changed += any(
            str(getattr(old_keys[k], col)) != str(getattr(new_keys[k], col))
            for col in common_cols
        )
    return {
        "n_old": len(old),
        "n_new": len(new),
        "n_exact_geometry_unchanged": len(set(old_keys) & set(new_keys)),
        "n_old_geometry_changed": len(changed),
        "n_old_removed": removed,
        "n_old_modified": modified,
        "n_new_nonoverlapping_added": len(added),
        "n_unchanged_geometry_attributes_changed": attr_changed,
        "changed_old_intervals": changed,
        "added_intervals": added,
    }


def seed_audit(frame, signals):
    details, missing = [], []
    for shot, rows in frame.groupby("shot"):
        path = signals / f"{int(shot)}.npz"
        if not path.is_file():
            missing.append(int(shot))
            continue
        with np.load(path) as data:
            t = data["t_ms"]
            dt = float(np.median(np.diff(t)))
            width = max(1, round(5 / dt)) | 1
            for row in rows.itertuples(index=False):
                y = data[f"n{int(row.n)}rms"]
                smoothed = median_filter(y, width)
                # Rule endpoints are inclusive t[b-1], not half-open bin edges.
                # Include the final sample so an exact 50-sample seed is 50 ms.
                tolerance = max(1e-7, abs(dt) * 1e-6)
                inside = (t >= row.t_start - tolerance) & (t <= row.t_end + tolerance)
                seed = 12.0 if row.n == 1 else 6.0
                raw = longest_run(
                    t[inside], (y[inside] >= seed) & np.isfinite(y[inside])
                )
                smooth = longest_run(
                    t[inside],
                    (smoothed[inside] >= seed) & np.isfinite(smoothed[inside]),
                )
                details.append(
                    {
                        "shot": int(shot),
                        "n": int(row.n),
                        "t_start": row.t_start,
                        "t_end": row.t_end,
                        "raw_longest_seed_ms": raw,
                        "median5ms_longest_seed_ms": smooth,
                        "raw_seed_failure": raw < 50 - 1e-6,
                        "median_seed_failure": smooth < 50 - 1e-6,
                    }
                )
    n = len(details)
    return {
        "n_intervals_audited": n,
        "missing_signal_shots": missing,
        "raw_failure_count": sum(r["raw_seed_failure"] for r in details),
        "median_failure_count": sum(r["median_seed_failure"] for r in details),
        "raw_failure_fraction": sum(r["raw_seed_failure"] for r in details) / n
        if n
        else None,
        "median_failure_fraction": sum(r["median_seed_failure"] for r in details) / n
        if n
        else None,
        "by_n": {
            str(mode): {
                "n_intervals": len(rows),
                "raw_failure_count": sum(r["raw_seed_failure"] for r in rows),
                "median_failure_count": sum(r["median_seed_failure"] for r in rows),
            }
            for mode in (1, 2)
            if (rows := [r for r in details if r["n"] == mode])
        },
        "details": details,
    }


def locking_audit(frame, frequencies):
    """Separate missing frequency records from unknown locked-mode confirmation."""
    unknown, candidates = [], []
    for shot, rows in frame.groupby("shot"):
        path = frequencies / f"{int(shot)}.npz"
        frequency_known = {1: False, 2: False}
        if path.is_file():
            with np.load(path) as data:
                frequency_known = {
                    n: np.isfinite(data[f"n{n}freq"]).any() for n in (1, 2)
                }
        for row in rows.itertuples(index=False):
            common = {
                "shot": int(shot),
                "n": int(row.n),
                "t_start": row.t_start,
                "t_end": row.t_end,
                "ended": row.ended,
                "locked_known": bool(getattr(row, "locked_known", False)),
            }
            if not frequency_known[int(row.n)]:
                unknown.append(common)
            if bool(getattr(row, "locked_candidate", False)):
                times = getattr(row, "lock_candidates_ms", "()")
                times = ast.literal_eval(times) if isinstance(times, str) else times
                candidates.append(
                    {
                        **common,
                        "lock_time_ms": row.lock_time_ms,
                        "lock_candidates_ms": list(times),
                    }
                )
    known = frame.get("locked_known", pd.Series(False, index=frame.index)).astype(bool)
    candidate = frame.get("locked_candidate", pd.Series(False, index=frame.index))
    return {
        "n_confirmed_locked": int(frame.locked.sum()),
        "n_locked_candidates": int(candidate.sum()),
        "n_lock_confirmation_unknown": int((~known).sum()),
        "lock_confirmation_unknown_shots": sorted(
            frame.loc[~known, "shot"].unique().astype(int).tolist()
        ),
        "n_intervals_without_frequency_record": len(unknown),
        "frequency_unknown_shots": sorted({r["shot"] for r in unknown}),
        "frequency_unknown_rows": unknown,
        "n_unknown_frequency_incorrect_end": sum(
            r["ended"] != "unknown" for r in unknown
        ),
        "n_unknown_frequency_incorrect_known_flag": sum(
            r["locked_known"] for r in unknown
        ),
        "intervals_by_end": {
            str(k): int(v) for k, v in frame.ended.value_counts().items()
        },
        "candidate_rows": candidates,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--original-dir", type=Path, default=OUT / "fix1_original")
    parser.add_argument("--new-dir", type=Path, default=OUT / "labels")
    parser.add_argument("--signals-dir", type=Path, default=OUT / "signals")
    parser.add_argument("--freq-dir", type=Path, default=OUT / "signals_freq")
    parser.add_argument("--mag-dir", type=Path, default=OUT / "magfeatures")
    parser.add_argument(
        "--catalog-labels",
        type=Path,
        default=REPO
        / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    record = {
        "made_by": "scripts/labeler/tm_audit_rule.py",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "policy": __doc__,
        "original_dir": str(args.original_dir),
        "new_dir": str(args.new_dir),
        "source_sha256": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                REPO / "src/labeler/tearing/rule.py",
                REPO / "scripts/labeler/tm_label.py",
                Path(__file__),
            )
        },
        "label_sha256": {},
    }
    for name in ("cohort", "population"):
        filename = f"tm_intervals_full_{name}.csv"
        record["label_sha256"][name] = {
            "original": hashlib.sha256(
                (args.original_dir / filename).read_bytes()
            ).hexdigest(),
            "regenerated": hashlib.sha256(
                (args.new_dir / filename).read_bytes()
            ).hexdigest(),
        }
        old = pd.read_csv(args.original_dir / filename)
        new = pd.read_csv(args.new_dir / filename)
        record[name] = {
            "changes": changes(old, new),
            "before_seed_audit": seed_audit(old, args.signals_dir),
            "after_seed_audit": seed_audit(new, args.signals_dir),
            "locking_audit": locking_audit(new, args.freq_dir),
        }
    table = pd.read_csv(args.catalog_labels)
    record["reviewed_shots"] = {
        str(shot): json.loads(
            table[(table.shot == shot) & (table.category.isin([1, 2]))].to_json(
                orient="records"
            )
        )
        for shot in REVIEWED
    }
    # Fixed review examples distinguish the two early broadband pulses from the
    # later rotating coherent line, independently of the regenerated label state.
    with np.load(args.mag_dir / "190790.npz") as data:
        t, features = data["centres_ms"], data["features"]
    with np.load(args.freq_dir / "190790.npz") as data:
        ft, frequency = data["t_ms"], data["n1freq"]
    fit = features[:, FEATURE_NAMES.index("fit1")]
    prominence = features[:, FEATURE_NAMES.index("line_prominence_db")]
    line = features[:, FEATURE_NAMES.index("line_khz")]
    record["physics_cases_190790"] = []
    for start, end in ((1676, 1752), (2238, 2411), (3163, 5764)):
        use = (t >= start) & (t < end)
        coherent = use & (fit >= 0.9) & (prominence >= 10) & (line > 0) & (line <= 30)
        f_use = (ft >= start) & (ft < end) & np.isfinite(frequency)
        record["physics_cases_190790"].append(
            {
                "t_start": start,
                "t_end": end,
                "n_mirnov_bins": int(use.sum()),
                "n_coherent_bins": int(coherent.sum()),
                "coherent_fraction": float(coherent.sum() / use.sum()),
                "continuous_coherent_ms": longest_run(t, coherent),
                "phase_fit_quantiles": {
                    str(q): float(np.quantile(fit[use], q)) for q in (0.1, 0.5, 0.9)
                },
                "prominence_quantiles_db": {
                    str(q): float(np.quantile(prominence[use], q))
                    for q in (0.1, 0.5, 0.9)
                },
                "n_frequency_bins": int(f_use.sum()),
                "frequency_rotating_below30_fraction": float(
                    ((frequency[f_use] > 0) & (frequency[f_use] <= 30)).mean()
                )
                if f_use.any()
                else None,
                "frequency_quantiles_khz": {
                    str(q): float(np.quantile(frequency[f_use], q))
                    for q in (0.1, 0.5, 0.9)
                }
                if f_use.any()
                else {},
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    "changes": {
                        k: v
                        for k, v in record[name]["changes"].items()
                        if not isinstance(v, list)
                    },
                    "after_seed_audit": {
                        k: v
                        for k, v in record[name]["after_seed_audit"].items()
                        if k != "details"
                    },
                    "locking_audit": {
                        k: v
                        for k, v in record[name]["locking_audit"].items()
                        if not isinstance(v, list)
                    },
                }
                for name in ("cohort", "population")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
