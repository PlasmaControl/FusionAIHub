#!/usr/bin/env python
"""Write the whole-interval tearing-mode labels for the cohort (and the population).

Reads the fetched n = 1 / n = 2 RMS records (`tm_fetch.py`), applies the frozen rule
(`labeler.tearing.rule`) inside each shot's catalog window, and writes

* ``data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv`` (+ meta):
  the cohort's rows in the catalog's interval schema, small enough to commit;
* ``$LABELER_ROOT/round4/tm/labels/`` the same for the population, and one row per
  interval with the amplitudes the catalog schema has no place for.

Run from the worktree::

    PYTHONPATH=$PWD/src pixi run --frozen --no-install -e labelmaker python \\
        scripts/labeler/tm_label.py --from cohort
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths, git_sha
from labeler.events import spans
from labeler.events.interval_tables import parse_attrs, write_interval_table
from labeler.tearing import rule, scoring, surface

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
OUT_ROOT = LABELER / "round4/tm"
SIGNALS = OUT_ROOT / "signals"
FREQUENCIES = OUT_ROOT / "signals_freq"
MAGFEATURES = OUT_ROOT / "magfeatures"
LOCK_SIGNALS = OUT_ROOT / "signals_lock"
CATALOG = REPO / "data/events/catalog"
COHORT_OUT = REPO / "data/events/neoclassical_tearing_mode/extend_tm_interval"


def load_signals(shot: int, directory: Path):
    """`(t_ms, n1rms, n2rms)` of a fetched shot, or None."""
    path = directory / f"{shot}.npz"
    if not path.is_file():
        return None
    with np.load(path) as npz:
        return npz["t_ms"], npz["n1rms"], npz["n2rms"]


def lock_times(shot: int, directory: Path):
    """`{n: times}` the toroidal modes' frequency fell to zero, or None: no record."""
    path = directory / f"{shot}.npz"
    if not path.is_file():
        return None
    with np.load(path) as npz:
        return {
            1: rule.frequency_locks(npz["t_ms"], npz["n1freq"]),
            2: rule.frequency_locks(npz["t_ms"], npz["n2freq"]),
        }


def surface_hook(shot: int, roots):
    """`m_of(n, start_ms, end_ms)` from the shot's EFIT q in its features, or None.

    `roots` are the `Paths` to look in, in order; the first holding the shot's feature
    file with a `qpsi` record is used. No ECE island radius has been resolved by
    this pipeline, so `supported_m` leaves m empty even for a unique candidate.
    """
    for paths in roots:
        path = paths.features_file(shot)
        if not path.is_file():
            continue
        with h5py.File(path, "r") as f:
            if "qpsi" not in f:
                continue
            t_ms = f["qpsi/xdata"][:] * 1000.0
            q, rho = f["qpsi/ydata"][:], f["qpsi/rho"][:]
        return lambda n, start, end: surface.supported_m(n, t_ms, q, rho, start, end)
    return None


#: Weak-track amplitude floors: the frozen development quiet-time p95 of the Mirnov
#: coherent amplitude (log10, `calibration_dev_fix1.json`).
WEAK_FLOOR = {1: -0.2435681, 2: -0.6028450}
#: The cap on a rotating line's frequency per toroidal number (kHz): n times 30.
LINE_CAP_KHZ = {n: rule.LINE_KHZ_PER_N * n for n in (1, 2)}


def line_evidence(shot, t_ms, freq_dir, mag_dir=MAGFEATURES, cap_khz=None):
    """Coherent/weak-line masks on the uniform RMS grid, and frequency drops.

    N<n>FREQ is the processed n-resolved line frequency. Where missing, use the
    Mirnov phase fit >=0.9, prominence >=10 dB and coherent amplitude above the
    development-only quiet p95 (`WEAK_FLOOR`). The Mirnov line frequency is searched
    only over 1-30 kHz (`magfeatures.LINE_KHZ`), so no range test is repeated here. An
    available N<n>FREQ above `cap_khz[n]` (default 30 kHz times n) vetoes the Mirnov
    fallback for that n; the Mirnov features themselves stop at 30 kHz, so the n = 2
    cap of 60 kHz acts through N2FREQ alone. The weak track is released at the weak
    floor itself, the level that opened it. No blind test calibration is read.
    """
    cap = {**LINE_CAP_KHZ, **(cap_khz or {})}
    t = np.asarray(t_ms)
    dt = float(np.median(np.diff(t)))
    mirnov = {1: np.zeros(t.shape, bool), 2: np.zeros(t.shape, bool)}
    weak_mirnov = {1: np.zeros(t.shape, bool), 2: np.zeros(t.shape, bool)}
    seed_support = {1: np.zeros(t.shape, bool), 2: np.zeros(t.shape, bool)}
    screened = {1: np.zeros(t.shape, bool), 2: np.zeros(t.shape, bool)}
    path = mag_dir / f"{shot}.npz"
    if path.is_file():
        with np.load(path) as z:
            centres, features = z["centres_ms"], z["features"]
            names = list(z["names"])
        index = np.searchsorted(centres + 5.0, t)
        inside = (index < len(centres)) & (t >= centres[0] - 5.0)
        for n, floor in WEAK_FLOOR.items():
            screened[n][inside] = np.isfinite(features[index[inside]]).all(axis=1)
            amplitude = np.max(
                features[
                    :, [i for i, name in enumerate(names) if name.startswith(f"a{n}_")]
                ],
                axis=1,
            )
            frequency = features[:, names.index("line_khz")]
            prominent = features[:, names.index("line_prominence_db")] >= 10.0
            mask = (features[:, names.index(f"fit{n}")] >= 0.9) & prominent
            mask &= amplitude > floor
            mirnov[n][inside] = mask[index[inside]]
            # a<n> is already restricted to cells best-fitting n with >=0.9
            # coherence. A different, stronger line can reduce fit<n> at the
            # overall peak without erasing this weaker n-resolved line.
            weak_mask = (amplitude > floor) & prominent
            weak_mirnov[n][inside] = weak_mask[index[inside]]
            seed_mask = mask & rule.coherent_frequency(frequency, 10.0)
            seed_support[n][inside] = seed_mask[index[inside]]
    support = {n: mask.copy() for n, mask in mirnov.items()}
    locks = None
    path = None if freq_dir is None else freq_dir / f"{shot}.npz"
    if path is not None and path.is_file():
        locks = {}
        with np.load(path) as z:
            for n in (1, 2):
                freq = z[f"n{n}freq"]
                aligned = scoring.align_scores(z["t_ms"], freq, t)
                stable = rule.coherent_frequency(aligned, dt, max_khz=cap[n])
                support[n] |= stable
                screened[n] |= np.isfinite(aligned)
                # Available processed frequency settles the rotating seed:
                # a coherent rapid sweep or stationary pulse is only a candidate.
                seed_support[n] = np.where(
                    np.isfinite(aligned),
                    rule.coherent_frequency(aligned, dt, max_khz=cap[n]),
                    seed_support[n],
                )
                support[n][aligned > cap[n]] = False
                mirnov[n][aligned > cap[n]] = False
                if np.isfinite(freq).any():
                    locks[n] = rule.frequency_locks(z["t_ms"], freq)
    # The weak track opens above the floor and is released at the same floor.
    return support, weak_mirnov, locks, seed_support, weak_mirnov, screened


def shot_label(
    shot: int,
    window,
    paths: Paths,
    directory: Path,
    freq_dir=None,
    q_roots=(),
    *,
    rules=rule.RULES,
    cap_khz=None,
):
    """`(label, how, locks_known)` of one shot, or None: its record was not fetched."""
    record = load_signals(shot, directory)
    if record is None:
        return None
    t_ms, n1, n2 = record
    start, how = spans.plasma_start(shot, paths, window)
    t_ms, n1, _ = rule.uniform(t_ms, n1)
    _, n2, _ = rule.uniform(record[0], n2)
    coherent, weak, locks, seed, weak_release, screened = line_evidence(
        shot, t_ms, freq_dir, cap_khz=cap_khz
    )
    amplitude = None
    lock_path = LOCK_SIGNALS / f"{shot}.npz"
    if rule.valid_lock_shot(shot) and lock_path.is_file():
        with np.load(lock_path) as z:
            amplitude = {1: scoring.align_scores(z["t_ms"], z["bradial"], t_ms)}
    label = rule.label_shot(
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
        m_of=surface_hook(shot, q_roots or (paths,)),
    )
    return label, how, locks is not None and all(i.n in locks for i in label.intervals)


def table_for(
    shots: pd.DataFrame, paths: Paths, directory: Path, freq_dir=None, q_roots=()
):
    """`(rows, intervals, labels, missing, starts, unlocked)` over the shots.

    `unlocked` lists the shots with an interval and no frequency record, whose
    `locked` is therefore unknown (left unset).
    """
    labels, tables, missing, starts, unlocked = [], [], [], {}, []
    for row in shots.itertuples(index=False):
        window = (float(row.window_start_ms), float(row.window_end_ms))
        made = shot_label(int(row.shot), window, paths, directory, freq_dir, q_roots)
        if made is None:
            missing.append(int(row.shot))
            continue
        label, how, known = made
        labels.append(label)
        starts[int(row.shot)] = {"start_ms": label.start_ms, "from": how}
        if label.intervals and not known and freq_dir is not None:
            unlocked.append(int(row.shot))
        tables.append(rule.shot_table(label))
    rows = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return rows, rule.intervals_frame(labels), labels, missing, starts, unlocked


def counts_of(frame, intervals) -> dict:
    """What the table holds: intervals by n and end reason, locked, m, onset points."""
    span = frame[(frame.category == 1) & (frame.t_end > frame.t_start)]
    point = frame[(frame.category == 1) & (frame.t_end == frame.t_start)]
    return {
        "n_intervals": len(span),
        "n_onset_points": len(point),
        "intervals_by_n": {
            str(k): int(v) for k, v in intervals.n.value_counts().sort_index().items()
        },
        "intervals_by_end": {
            str(k): int(v) for k, v in intervals.ended.value_counts().items()
        },
        "n_locked": int(intervals.locked.sum()),
        "n_locked_candidates": int(intervals.locked_candidate.sum()),
        "n_locked_known": int(intervals.locked_known.sum()),
        "n_uncertain_rows": int(frame.category.eq(2).sum()),
        "uncertain_rows_by_reason": {
            str(k): int(v)
            for k, v in frame[frame.category == 2]["attrs"]
            .map(lambda a: parse_attrs(a).get("reason", "ramp_up"))
            .value_counts()
            .items()
        },
        "n_onset_windows": int(intervals.onset_window_start_ms.notna().sum()),
        "n_onset_windows_at_most_5_ms": int(
            (
                (intervals.t_start - intervals.onset_window_start_ms)
                <= rule.ONSET_WINDOW_DEGENERATE_MS + rule.ONSET_WINDOW_EPSILON_MS
            ).sum()
        ),
        "n_onset_rows_flagged_degenerate": int(
            point["attrs"]
            .map(lambda a: bool(parse_attrs(a).get("onset_window_degenerate")))
            .sum()
        ),
        "n_with_m": int(intervals.m.notna().sum()),
        "n_without_observed_onset": int((~intervals.onset_seen.astype(bool)).sum()),
        "shots_with_uncertain": int(frame[frame.category == 2].shot.nunique()),
        "shots_with_not_observable": int(frame[frame.category == 3].shot.nunique()),
        "median_duration_ms": float(intervals.duration_ms.median())
        if len(intervals)
        else None,
    }


def meta_for(which, frame, labels, missing, unlocked, shots, rules, extra=None):
    present = frame[(frame.category == 1) & (frame.t_end > frame.t_start)]
    return {
        "category": rule.CATEGORY,
        "table_kind": "intervals",
        "made_by": "scripts/labeler/tm_label.py",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "rule_sha256": hashlib.sha256(
            (REPO / "src/labeler/tearing/rule.py").read_bytes()
        ).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shots": which,
        "n_requested_shots": len(shots),
        "n_labelled_shots": len(labels),
        "n_missing_signal_shots": len(missing),
        "n_shots_with_a_mode": int(present.shot.nunique()),
        "signals": "\\MHD::N1RMS, \\MHD::N2RMS (tree mhd, gauss, 1 kHz), fetched",
        "rule": {f"n{r.n}": asdict(r) for r in rules},
        "rule_doc": "docs/labeler/tearing_detection.md; labeler.tearing.rule",
        "crowd": "span rows iscrowd 1 (the whole interval), onset point rows "
        "iscrowd 0; t_start == t_end marks a point",
        "absent": "the rest of the shot's catalog window; ramp-up uncertain where the "
        "rule fires in it; unsupported seeds and >=100 ms weak coherent lines "
        "uncertain; every acquisition gap is not observable",
        "plasma_start": "labeler.events.spans.plasma_start: the rule runs from the "
        "time Ip reaches its flat-top fraction; each shot's start is in "
        "$LABELER_ROOT/round4/tm/labels/plasma_start_<set>.json",
        "missing_signal_shots": missing,
        "locked": {
            "signal": "\\MHD::N1FREQ, \\MHD::N2FREQ (kHz); PTDATA DUSBRADIAL "
            "(native ptdata units, treated as gauss by disruption-py)",
            "rule": "Frequency <=1 kHz for 20 ms after >=1.5 kHz is only a "
            "locked_candidate, storing earliest lock_time_ms and every in-span "
            "drop in lock_candidates_ms (plus <=100 ms after the RMS end). "
            "An abrupt raw >seed to <release collapse within <=5 ms never "
            "counts as decay: unknown unless |DUSBRADIAL| steps at a candidate "
            "time: its median over the 20 to 120 ms after the time is >=5 above "
            "its median over the 200 to 20 ms before it (each window needs 50 ms "
            "measured, the baseline is not read before the plasma start). The "
            "candidate times are the frequency drops at least 50 ms after the "
            "seed starts, the collapse, every interval's end and the end of each "
            "rejected candidate. "
            "Post-lock time is uncertain until |DUSBRADIAL| stays <5 above that "
            "local baseline for 200 ms (shorter dips are no release) or the "
            "discharge ends; an absent diagnostic leaves an unconfirmed collapse's "
            "tail uncertain. A step of 5 followed by 100 ms above the quiet level "
            "in flat-top time no interval or lock tail covers is uncertain "
            "`locked_unseeded`. Without frequency: ended=unknown and "
            "locked_known=false per row.",
            "locked_known": "true when a lock was confirmed (then locked is true "
            "too); false only says none was confirmed: no radial-field record, no "
            "step, or n = 2, which has no confirmation",
            "intervals_without_a_frequency_record_shots": unlocked,
            "unconfirmed_lock_shots": sorted(
                label.shot
                for label in labels
                if any(not item.locked_known for item in label.intervals)
            ),
        },
        "coherent_line": "Continuous >=50 ms seed crossing before merging, "
        "n-resolved frequency 1.5 kHz to 30 kHz times n or Mirnov phase fit>=0.9, "
        "prominence>=10 dB and amplitude above development-only quiet p95; local "
        "50 ms frequency p90-p10 width <= max(2 kHz, 25% median), >=80% span "
        "support. Unsupported seeds and sustained >=100 ms weak coherent activity "
        "are uncertain, excluded from detector training/scoring.",
        "weak_track": "Continuous >=100 ms n-resolved Mirnov amplitude above "
        "the frozen development quiet p95 establishes uncertainty, extended "
        "along the coherent line down to that same amplitude floor; <=50 ms "
        "evidence interruptions can be joined, acquisition gaps cannot. The "
        "screen runs over the whole catalog window, ramp-up included.",
        "onset_window_ms": "An onset point carries [start of the preceding same-n "
        "weak track, interval start] in ms where such a track leads into the "
        "interval; the onset itself stays at the interval start (where the RMS "
        "crossed a tenth of the peak). A window of at most 5 ms is one the weak "
        "track opened at the interval start: the onset row carries "
        "onset_window_degenerate true and the window is counted in "
        "counts.n_onset_windows_at_most_5_ms, not widened.",
        "calibration": "benchmark/sources/calibration_dev_fix1.json (weak floors, "
        "weak RMS thresholds) and calibration_dev_fix4.json (harmonic ratio)",
        "screening_missing": "All requested shots exclude blind cohort IDs before "
        "input reads; unavailable line screening at RMS above frozen weak_g "
        "is uncertain even without positive coherent-line evidence.",
        "test_exposure_correction": "An earlier harmonic ratio was read off a "
        "blind-test shot; it was replaced by a development-only calibration.",
        "m": {
            "signal": "qpsi_EFIT01 (the shot's feature file), offline EFIT01",
            "rule": "labeler.tearing.surface.supported_m requires an independently "
            "observed island radius to evaluate m = n*q there. No ECE island "
            "radius is resolved here, so m is empty; q alone is insufficient.",
        },
        **(extra or {}),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--from", dest="source", choices=("cohort", "population"), default="cohort"
    )
    ap.add_argument("--signals-dir", type=Path, default=SIGNALS)
    ap.add_argument("--freq-dir", type=Path, default=FREQUENCIES)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--features-root",
        type=Path,
        nargs="*",
        default=[OUT_ROOT / "lroot", LABELER],
        help="labeler roots whose feature files give EFIT q (first match wins)",
    )
    args = ap.parse_args(argv)

    paths = Paths.from_env()
    table = pd.read_csv(CATALOG / f"{args.source}.csv")
    blind = set(pd.read_csv(CATALOG / "cohort.csv").query("split == 'test'").shot)
    table = table[~table.shot.isin(blind)]
    shots = table[["shot", "window_start_ms", "window_end_ms"]]
    q_roots = [replace(paths, root=root) for root in args.features_root]
    rows, intervals, labels, missing, starts, unlocked = table_for(
        shots, paths, args.signals_dir, args.freq_dir, q_roots
    )
    out_dir = args.out_dir or (
        COHORT_OUT if args.source == "cohort" else OUT_ROOT / "labels"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_for(
        args.source,
        rows,
        labels,
        missing,
        unlocked,
        shots,
        rule.RULES,
        {"counts": counts_of(rows, intervals)},
    )
    name = (
        "tm_interval.csv" if args.source == "cohort" else "tm_interval_population.csv"
    )
    write_interval_table(rows, out_dir / name, meta)
    big = OUT_ROOT / "labels"
    big.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(big / f"tm_intervals_full_{args.source}.csv", index=False)
    (big / f"plasma_start_{args.source}.json").write_text(
        json.dumps(starts, indent=0) + "\n", encoding="utf-8"
    )
    print(
        f"{args.source}: {len(labels)} shots labelled, {len(missing)} without a "
        f"record, {len(intervals)} intervals on {intervals.shot.nunique()} shots"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
