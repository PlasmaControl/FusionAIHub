#!/usr/bin/env python
"""What the fourth label round changed, measured on the development cohort.

Numbers the documentation quotes, each before (the previous label round's table,
snapshotted under ``round4/tm/fix3_original``) and after (the current tables):

* the uncertain fraction of observable time, pooled and for the median shot;
* the n = 2 seeds the harmonic veto and the frequency cap each remove: the veto at the
  previous level (0.72) and at the current one (0.57, calibrated on the bins that fit
  n = 2), counted by relabelling every shot with the cut on and off;
* the quiet-time false-confirmation rate of the radial-field lock test: pseudo-locks
  placed in time the previous labels call absent, tested with the previous round's rule
  (the baseline read before the pseudo-onset, a level held 20 ms) and with the current
  one (a step at the time itself), at lags of 300, 1000 and 2000 ms between the
  pseudo-onset and the pseudo-lock and at lags drawn from the interval durations of
  the current cohort labels;
* the lock counts and the size of the step at every confirmed lock, and the cohort lock
  with the largest step (the column example's locking shot).

Only development shots are read. Writes ``<sources>/rule_diagnostics_fix4.json``.

    PYTHONPATH=$PWD/src LABELER_NO_FETCH=1 pixi run --frozen --no-install -e labelmaker \\
        python scripts/labeler/tm_rule_diagnostics.py
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
BEFORE = OUT / "fix3_original"
CATALOG = REPO / "data/events/catalog"
SOURCES = REPO / "data/events/neoclassical_tearing_mode/benchmark/sources"
NEW_TABLE = (
    REPO / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"
)
#: Quiet-time pseudo-locks: the mode "begins" at a drawn time and the lock is tested
#: `lag` later; the lags are these, then a draw from the cohort's interval durations.
LAGS_MS = (300.0, 1000.0, 2000.0)
DRAWS_PER_SHOT = 20
#: A pseudo-lock needs this much quiet time before the pseudo-onset's baseline reach
#: and after the lock (ms): the 200 ms baseline and the 120 ms step window.
MARGIN_BEFORE_MS, MARGIN_AFTER_MS = 200.0, 120.0
PREVIOUS_RATIO = 0.72
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


def n2_cuts(cohort: pd.DataFrame, starts: dict, ratio: float) -> dict:
    """n = 2 intervals kept with each cut on and off, at the previous and current veto."""
    base = rule.N2_RULE
    cap = {2: 60.0}
    variants = {
        "previous_veto": (replace(base, harmonic_ratio=PREVIOUS_RATIO), cap),
        "no_veto": (replace(base, harmonic_ratio=None), cap),
        "current_veto": (replace(base, harmonic_ratio=ratio), cap),
        "current_veto_no_cap": (replace(base, harmonic_ratio=ratio), {2: NO_CAP}),
    }
    kept = {name: 0 for name in variants}
    shots_with = {name: set() for name in variants}
    for row in cohort[cohort.split != "test"].itertuples():
        shot = int(row.shot)
        start = starts.get(str(shot), {}).get("start_ms", row.window_start_ms)
        for name, (n2_rule, caps) in variants.items():
            label = relabel(shot, row, start, (rule.N1_RULE, n2_rule), caps)
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
            "previous_level": kept_n["no_veto"] - kept_n["previous_veto"],
            "current_level": kept_n["no_veto"] - kept_n["current_veto"],
        },
        "removed_by_frequency_cap": kept_n["current_veto_no_cap"]
        - kept_n["current_veto"],
        "harmonic_ratio_previous": PREVIOUS_RATIO,
        "harmonic_ratio_current": ratio,
        "note": "The Mirnov features stop at 30 kHz, so the 60 kHz n = 2 cap acts "
        "through N2FREQ only. Counts are n = 2 intervals after the whole rule.",
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


def n2_seed_cuts(cohort: pd.DataFrame, starts: dict, ratio: float) -> dict:
    """n = 2 seeds (50 ms above 6 G, raw and 5 ms median) each cut removes.

    A seed is a qualified run on the RMS alone, inside the flat-top; a cut removes it
    when no 50 ms stretch of it is left with the cut applied: the harmonic veto alone
    at the previous level and at `ratio`, the coherent-frequency cap alone (60 kHz),
    and the veto (at `ratio`) and the cap together.
    """
    mr = rule.N2_RULE
    totals = {
        "seeds": 0,
        "shots": set(),
        "veto_previous": 0,
        "veto_current": 0,
        "cap": 0,
        "both_previous": 0,
        "both_current": 0,
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
        seed = tm_label.line_evidence(shot, t, tm_label.FREQUENCIES, cap_khz={2: 60.0})[
            3
        ]
        support = np.asarray(seed[2], bool)
        hold = mr.hold_ms
        for name, level in (("previous", PREVIOUS_RATIO), ("current", ratio)):
            veto = np.nan_to_num(n2) > level * np.nan_to_num(n1)
            totals[f"veto_{name}"] += len(runs) - _survivors(runs, veto, dt, hold)
            totals[f"both_{name}"] += len(runs) - _survivors(
                runs, veto & support, dt, hold
            )
        totals["cap"] += len(runs) - _survivors(runs, support, dt, hold)
    totals["shots"] = len(totals["shots"])
    totals["definition"] = (
        "seeds are runs of at least 50 ms with raw and 5 ms median n = 2 RMS above 6 G "
        "in the flat-top; a cut removes a seed when no 50 ms stretch of it survives; "
        "'both' applies the veto and the coherent-frequency cap together"
    )
    return totals


def previous_confirmation(amp, t, dt, onset, time, floor_ms):
    """The previous round's lock test: a level held 20 ms near `time`, True or False.

    The baseline is the median over the 200 ms before `onset` (the whole flat-top
    from `floor_ms` if under half of that window is measured); the field must stay at
    or above baseline + 5 for 20 ms somewhere in -5 to +100 ms of `time`.
    """
    window = (t >= onset - 200.0) & (t < onset) & np.isfinite(amp)
    if window.sum() >= max(1, int(0.5 * 200.0 / dt)):
        base = float(np.median(amp[window]))
    else:
        measured = np.isfinite(amp) & (t >= floor_ms)
        if not measured.any():
            return False
        base = float(np.median(amp[measured]))
    nearby = (t >= time - 5.0) & (t <= time + 100.0)
    return any(
        (hi - lo) * dt >= 20.0 - 1e-9
        for lo, hi in zip(
            *rule._runs(nearby & (amp >= base + rule.LOCK_RISE)), strict=True
        )
    )


def false_confirmation(
    cohort: pd.DataFrame, before_table: pd.DataFrame, durations
) -> dict:
    """Quiet-time false-confirmation rate of the previous and the current lock test.

    A pseudo-onset is drawn in a stretch the previous labels call absent, the
    pseudo-lock `lag` later, and both tests are asked about the same pseudo-lock. The
    stretch must reach `MARGIN_BEFORE_MS` before the pseudo-onset and `MARGIN_AFTER_MS`
    past the pseudo-lock. Lags are 300, 1000 and 2000 ms, then each draw's lag is one
    of `durations` (the cohort's interval durations) that fits its stretch.
    """
    rng = np.random.default_rng(0)
    durations = np.sort(np.asarray(durations, float))
    records = []
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
            if r.t_end - r.t_start >= MARGIN_BEFORE_MS + MARGIN_AFTER_MS
        ]
        if spans:
            records.append((shot, t, dt, amp, spans))
    out = {}
    for name, lags in [(f"lag_{int(lag)}_ms", lag) for lag in LAGS_MS] + [
        ("lag_from_interval_durations", None)
    ]:
        counts = {"draws": 0, "previous": 0, "current": 0, "shots": 0}
        used = []
        for shot, t, dt, amp, spans in records:
            made = 0
            for _ in range(DRAWS_PER_SHOT):
                a, b = spans[rng.integers(len(spans))]
                room = b - a - MARGIN_BEFORE_MS - MARGIN_AFTER_MS
                if lags is None:
                    fits = durations[durations <= room]
                    if not fits.size:
                        continue
                    lag = float(fits[rng.integers(len(fits))])
                else:
                    lag = lags
                    if lag > room:
                        continue
                onset = rng.uniform(a + MARGIN_BEFORE_MS, b - MARGIN_AFTER_MS - lag)
                time = onset + lag
                counts["draws"] += 1
                counts["previous"] += previous_confirmation(amp, t, dt, onset, time, a)
                counts["current"] += (
                    rule.lock_confirmation(amp, t, dt, time, floor_ms=a)[0] is not None
                )
                used.append(lag)
                made += 1
            counts["shots"] += bool(made)
        counts["previous_rate"] = counts["previous"] / max(counts["draws"], 1)
        counts["current_rate"] = counts["current"] / max(counts["draws"], 1)
        if lags is None:
            counts["lag_quantiles_ms"] = {
                str(q): float(np.percentile(used, q)) for q in (10, 25, 50, 75, 90)
            }
        out[name] = counts
    return {
        "definition": "pseudo-onsets are drawn in stretches the previous labels call "
        f"absent (at least {MARGIN_BEFORE_MS + MARGIN_AFTER_MS:.0f} ms long, up to "
        f"{DRAWS_PER_SHOT} draws per shot and lag, seed 0); the pseudo-lock is `lag` "
        "after the pseudo-onset; the previous rule reads the baseline before the "
        "pseudo-onset and asks for a level held 20 ms in -5 to +100 ms of the "
        "pseudo-lock, the current rule asks for a step at the pseudo-lock (median "
        "over 20 to 120 ms after, 5 above the median over 200 to 20 ms before; the "
        "baseline is not read before the stretch's start). The last row draws each "
        "lag from the current cohort's interval durations that fits the stretch.",
        "interval_duration_quantiles_ms": {
            str(q): float(np.percentile(durations, q)) for q in (10, 25, 50, 75, 90)
        },
        "n_interval_durations": len(durations),
        "rows": out,
    }


def lock_steps(full: pd.DataFrame) -> dict:
    """Size of the radial-field step at every confirmed lock, and the largest cohort one.

    Each confirmed lock's step is `lock_step` at its lock time: the median over 20 to
    120 ms after minus the median over 200 to 20 ms before.
    """
    rows = []
    for item in full[
        full.locked.fillna(False).isin((True, "True", "true"))
    ].itertuples():
        shot = int(item.shot)
        with np.load(tm_label.SIGNALS / f"{shot}.npz") as z:
            t, _, dt = rule.uniform(z["t_ms"], z["n1rms"])
        with np.load(tm_label.LOCK_SIGNALS / f"{shot}.npz") as z:
            amp = np.abs(scoring.align_scores(z["t_ms"], z["bradial"], t))
        got = rule.lock_step(amp, t, dt, float(item.lock_time_ms))
        if got is None:
            continue
        rows.append(
            {
                "shot": shot,
                "n": int(item.n),
                "t_start_ms": float(item.t_start),
                "lock_time_ms": float(item.lock_time_ms),
                "before": got[0],
                "after": got[1],
                "step": got[1] - got[0],
            }
        )
    steps = np.array([r["step"] for r in rows])
    best = max(rows, key=lambda r: r["step"]) if rows else None
    return {
        "n_confirmed_locks": int(
            (full.locked.fillna(False).isin((True, "True", "true"))).sum()
        ),
        "n_with_a_judged_step": len(rows),
        "step_min": float(steps.min()) if len(steps) else None,
        "step_median": float(np.median(steps)) if len(steps) else None,
        "step_max": float(steps.max()) if len(steps) else None,
        "largest_step": best,
        "rows": sorted(rows, key=lambda r: -r["step"]),
    }


def previous_onset_window() -> dict:
    """The previous round's "inside the onset window" figures, from its snapshot.

    That round flagged a matched reference onset as inside the window when it lay
    between the window start and the interval's END, which the matching itself almost
    guarantees; the flag is now defined against the interval's start.
    """
    out = {}
    for ref in ("seo", "survival"):
        record = json.loads((BEFORE / f"agreement_{ref}_cohort_dev.json").read_text())
        n1 = record["agreement"]["n1"]
        fraction = n1["error_ms"]["reference_inside_onset_window_fraction"]
        out[ref] = {
            "matched": n1["matched"],
            "inside_previous_definition": round(fraction * n1["matched"]),
            "fraction_previous_definition": fraction,
            "median_error_ms": n1["error_ms"]["median"],
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=SOURCES / "rule_diagnostics_fix4.json")
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
    full_population = pd.read_csv(OUT / "labels/tm_intervals_full_population.csv")
    record = {
        "made_by": "scripts/labeler/tm_rule_diagnostics.py",
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
        "lock_steps": {
            "cohort": lock_steps(full_after),
            "population": lock_steps(full_population),
        },
        "false_confirmation": false_confirmation(
            cohort, before, full_after.duration_ms.to_numpy(float)
        ),
        "onset_window_flag_before": previous_onset_window(),
        "duplicate_rows": {
            "before": int(before.duplicated().sum()),
            "after": int(after.duplicated().sum()),
        },
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
    print(
        json.dumps(
            {k: v for k, v in record.items() if k not in ("locking", "lock_steps")},
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
