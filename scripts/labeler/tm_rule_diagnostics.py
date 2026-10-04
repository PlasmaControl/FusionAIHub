#!/usr/bin/env python
"""What the third label round changed, measured on the development cohort.

Four numbers the documentation quotes, each before (the previous label round's table,
snapshotted under ``round4/tm/fix2_original``) and after (the current tables):

* the uncertain fraction of observable time, pooled and for the median shot;
* the n = 2 seeds each cut removes: the harmonic veto (n2/n1) and the frequency cap
  (30 kHz for every n before, 30 kHz times n now), counted by relabelling every shot
  with the cut on and off;
* the quiet-time false-confirmation rate of the radial-field lock test: pseudo-intervals
  placed in time the previous labels call absent, tested with the absolute 5 rule
  (before) and the rule that needs a rise of 5 over the 200 ms before the pseudo-onset
  (after);
* the lock counts: intervals ending in a confirmed lock, uncertain time that is
  `locked_unseeded`, and rejected candidates that end in a lock.

Only development shots are read. Writes ``<sources>/rule_diagnostics_fix3.json``.

    PYTHONPATH=$PWD/src LABELER_NO_FETCH=1 pixi run --frozen --no-install -e labelmaker \\
        python scripts/labeler/tm_fix3_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "src", Path(__file__).resolve().parent):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import tm_label

from labeler.config import git_sha
from labeler.events.interval_tables import parse_attrs
from labeler.tearing import rule, scoring

ROOT = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
OUT = ROOT / "round4/tm"
BEFORE = OUT / "fix2_original"
CATALOG = REPO / "data/events/catalog"
SOURCES = REPO / "data/events/neoclassical_tearing_mode/benchmark/sources"
NEW_TABLE = (
    REPO / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"
)
#: Quiet-time pseudo-intervals: the mode "begins" at a drawn time, its transition is
#: `GAP_MS` later, and the field is read in -5 to +100 ms of that.
GAP_MS = 300.0
DRAWS_PER_SHOT = 20
NO_CAP = 1.0e9


def seconds_by_state(table: pd.DataFrame, window, start=None) -> dict:
    """Seconds per state in `window` (from `start` on, if given); uncertain beats present."""
    w0, w1 = (float(window[0]), float(window[1]))
    if start is not None:
        w0 = max(w0, float(start))
    busy: list = []
    out = {}
    for key, category in (
        ("not_observable", 3),
        ("uncertain", 2),
        ("present", 1),
        ("absent", 0),
    ):
        spans = rule._union(
            [
                (max(w0, float(r.t_start)), min(w1, float(r.t_end)))
                for r in table[table.category.eq(category)].itertuples()
                if r.t_end > r.t_start and r.t_end > w0 and r.t_start < w1
            ]
        )
        out[key] = sum(b - a for a, b in rule._minus(spans, busy)) / 1000.0
        busy = rule._union(busy + spans)
    return out


def uncertain_fractions(
    table: pd.DataFrame, cohort: pd.DataFrame, starts: dict
) -> dict:
    """Pooled and median-shot uncertain fraction of observable time, window and flat-top."""
    rows = {"window": [], "flat_top": []}
    for shot, group in table.groupby("shot"):
        window = (
            cohort.loc[shot, "window_start_ms"],
            cohort.loc[shot, "window_end_ms"],
        )
        start = starts.get(str(shot), {}).get("start_ms", window[0])
        for key, begin in (("window", None), ("flat_top", start)):
            rows[key].append(seconds_by_state(group, window, begin))
    result = {}
    for key, items in rows.items():
        frame = pd.DataFrame(items)
        seen = frame.absent + frame.present + frame.uncertain
        fraction = frame.uncertain / seen.where(seen > 0)
        result[key] = {
            "shots": len(frame),
            "uncertain_s": float(frame.uncertain.sum()),
            "observable_s": float(seen.sum()),
            "pooled_fraction": float(frame.uncertain.sum() / seen.sum()),
            "median_shot_fraction": float(fraction.median()),
            "shots_over_50_percent": int((fraction > 0.5).sum()),
            "shots_over_90_percent": int((fraction > 0.9).sum()),
        }
    return result


def lock_counts(table: pd.DataFrame, intervals: pd.DataFrame) -> dict:
    """Intervals by end, and uncertain rows and seconds by lock reason."""
    reasons = {}
    uncertain = table[table.category.eq(2) & (table.t_end > table.t_start)]
    for row in uncertain.itertuples():
        reason = parse_attrs(row.attrs).get("reason", "ramp_up")
        entry = reasons.setdefault(reason, {"rows": 0, "seconds": 0.0, "shots": set()})
        entry["rows"] += 1
        entry["seconds"] += float(row.t_end - row.t_start) / 1000.0
        entry["shots"].add(int(row.shot))
    return {
        "intervals": len(intervals),
        "intervals_by_end": {
            str(k): int(v) for k, v in intervals.ended.value_counts().items()
        },
        "uncertain_by_reason": {
            k: {"rows": v["rows"], "seconds": v["seconds"], "shots": len(v["shots"])}
            for k, v in sorted(reasons.items())
        },
    }


def relabel(shot, row, start, rules, cap_khz):
    """The shot's label under `rules` and `cap_khz`, without EFIT m."""
    record = tm_label.load_signals(shot, tm_label.SIGNALS)
    if record is None:
        return None
    t_ms, n1, n2 = record
    t_ms, n1, _ = rule.uniform(t_ms, n1)
    n2 = rule.uniform(record[0], n2)[1]
    coherent, weak, locks, seed, weak_release, screened = tm_label.line_evidence(
        shot, t_ms, tm_label.FREQUENCIES, cap_khz=cap_khz
    )
    window = (float(row.window_start_ms), float(row.window_end_ms))
    amplitude = None
    lock_path = tm_label.LOCK_SIGNALS / f"{shot}.npz"
    if rule.valid_lock_shot(shot) and lock_path.is_file():
        with np.load(lock_path) as z:
            amplitude = {1: scoring.align_scores(z["t_ms"], z["bradial"], t_ms)}
    return rule.label_shot(
        shot,
        t_ms,
        n1,
        n2,
        window,
        start,
        lock_ms=locks,
        coherent=coherent,
        seed_coherent=seed,
        weak_coherent=weak,
        weak_release_coherent=weak_release,
        screened=screened,
        lock_amplitude=amplitude,
        rules=rules,
    )


def n2_cuts(cohort: pd.DataFrame, starts: dict, new_ratio: float) -> dict:
    """n = 2 intervals kept with each cut on and off, before and after this round."""
    base = replace(rule.N2_RULE, harmonic_ratio=0.57)
    variants = {
        "before_cap30_veto0.57": (base, {2: 30.0}),
        "before_cap30_no_veto": (replace(base, harmonic_ratio=None), {2: 30.0}),
        "before_no_cap_veto0.57": (base, {2: NO_CAP}),
        "after_cap60_veto": (replace(base, harmonic_ratio=new_ratio), {2: 60.0}),
        "after_cap60_no_veto": (replace(base, harmonic_ratio=None), {2: 60.0}),
        "after_no_cap_veto": (replace(base, harmonic_ratio=new_ratio), {2: NO_CAP}),
        "cap30_veto_new": (replace(base, harmonic_ratio=new_ratio), {2: 30.0}),
    }
    kept = {name: 0 for name in variants}
    shots_with = {name: set() for name in variants}
    dev = cohort[cohort.split != "test"]
    for row in dev.itertuples():
        shot = int(row.shot)
        start = starts.get(str(shot), {}).get("start_ms", row.window_start_ms)
        for name, (n2_rule, cap) in variants.items():
            label = relabel(shot, row, start, (rule.N1_RULE, n2_rule), cap)
            if label is None:
                break
            n = sum(item.n == 2 for item in label.intervals)
            kept[name] += n
            if n:
                shots_with[name].add(shot)
    kept_n = {k: int(v) for k, v in kept.items()}
    return {
        "n2_intervals_kept": kept_n,
        "shots_with_n2_interval": {k: len(v) for k, v in shots_with.items()},
        "removed_by_harmonic_veto": {
            "before": kept_n["before_cap30_no_veto"] - kept_n["before_cap30_veto0.57"],
            "after": kept_n["after_cap60_no_veto"] - kept_n["after_cap60_veto"],
        },
        "removed_by_frequency_cap": {
            "before": kept_n["before_no_cap_veto0.57"]
            - kept_n["before_cap30_veto0.57"],
            "after": kept_n["after_no_cap_veto"] - kept_n["after_cap60_veto"],
        },
        "harmonic_ratio_before": 0.57,
        "harmonic_ratio_after": new_ratio,
        "note": "The Mirnov features stop at 30 kHz, so the 60 kHz n = 2 cap acts "
        "through N2FREQ only.",
    }


def _qualified(mask, dt, hold_ms):
    """Runs of `mask` at least `hold_ms` long, as `(start, stop)` sample indices."""
    return [
        (a, b)
        for a, b in zip(*rule._runs(mask), strict=True)
        if (b - a) * dt >= hold_ms - 1e-9
    ]


def _survivors(runs, kept, dt, hold_ms):
    """How many of the seed `runs` still hold a `hold_ms` run once `kept` is applied."""
    return sum(bool(_qualified(kept[a:b], dt, hold_ms)) for a, b in runs)


def n2_seed_cuts(cohort: pd.DataFrame, starts: dict, new_ratio: float) -> dict:
    """n = 2 seeds (50 ms above 6 G, raw and 5 ms median) each cut removes.

    A seed is a qualified run on the RMS alone, inside the flat-top; a cut removes it
    when no 50 ms stretch of it is left with the cut applied, each cut alone and both
    together, with the previous settings (veto 0.57, cap 30 kHz) and the current ones
    (veto `new_ratio`, cap 60 kHz).
    """
    mr = rule.N2_RULE
    totals = {
        "seeds": 0,
        "shots": set(),
        "veto_before": 0,
        "veto_after": 0,
        "cap_before": 0,
        "cap_after": 0,
        "both_before": 0,
        "both_after": 0,
    }
    for row in cohort[cohort.split != "test"].itertuples():
        shot = int(row.shot)
        record = tm_label.load_signals(shot, tm_label.SIGNALS)
        if record is None:
            continue
        t, n1, dt = rule.uniform(record[0], record[1])
        n2 = rule.uniform(record[0], record[2])[1]
        start = starts.get(str(shot), {}).get("start_ms", row.window_start_ms)
        window = (t >= start) & (t <= row.window_end_ms)
        strong = (
            np.isfinite(n2)
            & window
            & (n2 > mr.onset_g)
            & (rule.smoothed(n2, dt, 5.0) > mr.onset_g)
        )
        runs = _qualified(strong, dt, mr.hold_ms)
        if not runs:
            continue
        totals["seeds"] += len(runs)
        totals["shots"].add(shot)
        support = {}
        for name, cap in (("before", 30.0), ("after", 60.0)):
            seed = tm_label.line_evidence(
                shot, t, tm_label.FREQUENCIES, cap_khz={2: cap}
            )[3]
            support[name] = np.asarray(seed[2], bool)
        veto = {
            "before": np.nan_to_num(n2) > 0.57 * np.nan_to_num(n1),
            "after": np.nan_to_num(n2) > new_ratio * np.nan_to_num(n1),
        }
        for name in ("before", "after"):
            lost = {
                "veto": len(runs) - _survivors(runs, veto[name], dt, mr.hold_ms),
                "cap": len(runs) - _survivors(runs, support[name], dt, mr.hold_ms),
                "both": len(runs)
                - _survivors(runs, veto[name] & support[name], dt, mr.hold_ms),
            }
            for key, value in lost.items():
                totals[f"{key}_{name}"] += value
    totals["shots"] = len(totals["shots"])
    totals["definition"] = (
        "seeds are runs of at least 50 ms with raw and 5 ms median n = 2 RMS above 6 G "
        "in the flat-top; a cut removes a seed when no 50 ms stretch of it survives; "
        "'both' applies the veto and the coherent-frequency cap together"
    )
    return totals


def false_confirmation(cohort: pd.DataFrame, before_table: pd.DataFrame) -> dict:
    """Quiet-time false-confirmation rate of the absolute and the relative lock test."""
    rng = np.random.default_rng(0)
    counts = {"draws": 0, "absolute": 0, "relative": 0, "shots": 0}
    offset = {"draws": 0, "absolute": 0, "relative": 0, "shots": 0}
    for row in cohort[cohort.split != "test"].itertuples():
        shot = int(row.shot)
        path = tm_label.LOCK_SIGNALS / f"{shot}.npz"
        sig = tm_label.SIGNALS / f"{shot}.npz"
        if not (rule.valid_lock_shot(shot) and path.is_file() and sig.is_file()):
            continue
        with np.load(sig) as z:
            t, _, dt = rule.uniform(z["t_ms"], z["n1rms"])
        with np.load(path) as z:
            amp = np.abs(scoring.align_scores(z["t_ms"], z["bradial"], t))
        quiet = before_table[(before_table.shot == shot) & (before_table.category == 0)]
        spans = [
            (float(r.t_start), float(r.t_end))
            for r in quiet.itertuples()
            if r.t_end - r.t_start >= 200.0 + GAP_MS + 200.0
        ]
        if not spans:
            continue
        counts["shots"] += 1
        base_high = False
        for _ in range(DRAWS_PER_SHOT):
            a, b = spans[rng.integers(len(spans))]
            onset = rng.uniform(a + 200.0, b - GAP_MS - 100.0)
            time = onset + GAP_MS
            absolute = rule.lock_confirmation(amp, t, dt, rule.LOCK_RISE, time)
            base = rule.lock_baseline(amp, t, dt, onset, a)
            relative = (
                None
                if base is None
                else rule.lock_confirmation(amp, t, dt, base + rule.LOCK_RISE, time)
            )
            for bucket in (counts,) + (
                (offset,) if base is not None and base >= 3.0 else ()
            ):
                bucket["draws"] += 1
                bucket["absolute"] += absolute is not None
                bucket["relative"] += relative is not None
            base_high |= base is not None and base >= 3.0
        offset["shots"] += int(base_high)
    return {
        "definition": "pseudo-intervals begin at a drawn time in a stretch the "
        "previous labels call absent (at least 700 ms long, "
        f"{DRAWS_PER_SHOT} draws per shot, seed 0); the test reads -5 to +100 ms "
        f"around {GAP_MS:.0f} ms after the pseudo-onset for 20 ms at the level",
        "all": {
            **counts,
            "absolute_rate": counts["absolute"] / max(counts["draws"], 1),
            "relative_rate": counts["relative"] / max(counts["draws"], 1),
        },
        "draws_with_baseline_at_least_3": {
            **offset,
            "absolute_rate": offset["absolute"] / max(offset["draws"], 1),
            "relative_rate": offset["relative"] / max(offset["draws"], 1),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=SOURCES / "rule_diagnostics_fix3.json")
    ap.add_argument("--skip-n2", action="store_true")
    ap.add_argument("--harmonic-ratio", type=float, default=rule.N2_RULE.harmonic_ratio)
    args = ap.parse_args(argv)
    cohort = pd.read_csv(CATALOG / "cohort.csv").set_index("shot", drop=False)
    before = pd.read_csv(BEFORE / "tm_interval.csv")
    after = pd.read_csv(NEW_TABLE)
    blind = set(cohort[cohort.split == "test"].shot)
    assert not (set(before.shot) | set(after.shot)) & blind
    starts_before = json.loads((BEFORE / "plasma_start_cohort.json").read_text())
    starts_after = json.loads((OUT / "labels/plasma_start_cohort.json").read_text())
    full_before = pd.read_csv(BEFORE / "tm_intervals_full_cohort.csv")
    full_after = pd.read_csv(OUT / "labels/tm_intervals_full_cohort.csv")
    record = {
        "made_by": "scripts/labeler/tm_fix3_diagnostics.py",
        "git_sha": git_sha(),
        "split": "development shots only; blind test shots never read",
        "uncertain_fraction": {
            "before": uncertain_fractions(before, cohort, starts_before),
            "after": uncertain_fractions(after, cohort, starts_after),
        },
        "locking": {
            "before": lock_counts(before, full_before),
            "after": lock_counts(after, full_after),
        },
        "false_confirmation": false_confirmation(cohort, before),
    }
    dev = cohort[cohort.split != "test"]
    have = [
        int(s) for s in dev.shot if (tm_label.LOCK_SIGNALS / f"{int(s)}.npz").is_file()
    ]
    record["lock_records"] = {
        "dev_shots": len(dev),
        "dev_shots_with_record": len(have),
        "dev_shots_without_record": len(dev) - len(have),
    }
    if not args.skip_n2:
        record["n2_cuts"] = n2_cuts(cohort, starts_after, args.harmonic_ratio)
        record["n2_seed_cuts"] = n2_seed_cuts(cohort, starts_after, args.harmonic_ratio)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "locking"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
