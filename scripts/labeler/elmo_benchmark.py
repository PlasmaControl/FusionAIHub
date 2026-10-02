#!/usr/bin/env python
"""Benchmark the ELM-O detector (O'Shea et al. 2023) on Smith's labelled ELMs and on our ELM labels.

ELM-O is rule-based. Two interferometer chords (``denv2f``, ``denv3f``), three
filterscopes (FS02-FS04) and the mean of the 64 BES channels each vote on where an ELM
is; an ELM is where all three kinds agree. The detector here is written from the paper's
description and the notes in docs/labeler/elm_benchmark_elmo.md (the public code has no
licence, so none of it is copied); the check against that code is recorded in the same
page.

**Detector** (``detect``). Every signal is brought to the BES sample rate by spectrum-
preserving up-sampling (DCT, zero-padded, inverse DCT, in pieces of at most 200 ms). The
BES trace is the mean of the 64 channels, the last 32 doubled (they have half the range);
it is a candidate wherever it exceeds ``t`` (1 V). Each other signal is differenced and
made absolute; it is a candidate wherever that exceeds its ``eta`` quantile over the
window (0.997). Candidate masks of the interferometers and filterscopes are widened by
100 microseconds either side; an interferometer vote is either chord, a filterscope
vote is two of three; an ELM is where all three kinds vote, gaps shorter than 100
microseconds closed. Sample counts follow the BES rate, so the 500 kS/s corpus BES
uses half the samples of the 1 MS/s that Smith's windows have.

**Smith's windows** (``smith``). The two files under /projects/EKOLEMEN/dsmith/data hold
2,489 windows of 64-channel BES (1 MS/s), each with one hand-marked ELM region. The
interferometer and filterscope records for those shots are fetched by
``scripts/labeler/elmo_fetch.py``. The paper's rule scores a window: true positive when
a detected span overlaps the marked region, false negative when none does, and each
detected span that overlaps nothing marked is a false positive. The paper's table (its
"AUC" equals precision x recall) is scanned over ``t`` and ``eta``. Windows of the
second file whose marked region overlaps one already taken are dropped as duplicates.

**Our review labels** (``review``). The shots of
``data/events/edge_localized_mode/review/labels.csv`` that the corpus holds BES for are
run in consecutive 200 ms chunks (the paper's window) over the reviewed time. Spans of
the review are scored in 50 ms bins that lie wholly inside one labelled span (a span of
category 1 is present, category 0 absent; uncertain and not-observable time is left
out), per detected ELM (precision), per individual span (hit) and per crowd span
(coverage), against the ``elm_clock`` suggestion the reviewers started from.

    python scripts/labeler/elmo_benchmark.py smith
    python scripts/labeler/elmo_benchmark.py review
    python scripts/labeler/elmo_benchmark.py evaluate

Needs numpy, scipy, pandas and h5py (``-e labelmaker``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.fft import dct, idct
from scipy.ndimage import maximum_filter1d

REPO = Path(__file__).resolve().parents[2]
LABELER = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
SMITH_DIR = Path("/projects/EKOLEMEN/dsmith/data")
SMITH_FILES = ("labeled-elm-events.hdf5", "labeled_elm_events_long_windows_20220921.hdf5")
REVIEW = REPO / "data/events/edge_localized_mode/review/labels.csv"
ELM_CLOCK = LABELER / "suggestions/elm_clock/v1/edge_localized_mode_suggest_elm_clock_v1.csv"
DEFAULT_WORK = LABELER / "benchmarks/elm/elmo"
DEFAULT_OUT = REPO / "outputs/labeler/elm/elmo"

ETA = 0.997
ETA_CODE = 0.995
THRESHOLD = 1.0
BLUR_US = 100.0
GAP_US = 100.0
CHUNK_MS = 200.0
THRESHOLDS = (0.5, 1.0, 2.0, 5.0)
ETAS = np.round(np.concatenate((np.arange(0.2, 0.951, 0.05), np.arange(0.96, 0.9995, 0.001))), 4)
#: The paper's Table II: t -> (eta, precision, recall, "AUC").
PAPER = {
    0.5: (0.998, 0.996, 0.974, 0.970),
    1.0: (0.997, 0.995, 0.976, 0.971),
    2.0: (0.997, 0.997, 0.975, 0.972),
    5.0: (0.2, 0.998, 0.961, 0.959),
}
BIN_MS = 50.0
#: An absent bin this close to a present span is "near" it: label boundaries are fuzzy there.
NEAR_MS = 500.0
VARIANTS = (("paper", ETA, True), ("code_eta", ETA_CODE, True), ("no_bes", ETA, False))
REPLICATES = 1000
BOOT_SEED = 20261001
DETECTION_KEYS = ("present", "absent", "other", "absent_within_50ms", "absent_within_500ms")


# ---------------------------------------------------------------- the detector


def lengthen(x: np.ndarray, length: int) -> np.ndarray:
    """Spectrum-preserving up-sampling of ``x`` to ``length`` samples (DCT, zero pad, inverse)."""
    spectrum = np.zeros(length)
    spectrum[: x.shape[0]] = dct(x, norm="ortho")
    return idct(spectrum, norm="ortho") * np.sqrt(length / x.shape[0])


def upsample(x: np.ndarray, length: int, chunk: int) -> np.ndarray:
    """``x`` up-sampled to about ``length`` samples in equal pieces of at most ``chunk`` samples.

    The pieces are ``length // n`` samples each, so the result can be a point or two
    shorter than ``length`` when ``length`` needs more than one piece.
    """
    n = -(-length // chunk)
    out_len = length // n
    in_len = x.shape[0] // n
    out = np.empty(n * out_len)
    for i in range(n):
        out[i * out_len : (i + 1) * out_len] = lengthen(x[i * in_len : (i + 1) * in_len], out_len)
    return out


class Trace:
    """What the detector reads: the BES trace and the differenced signals on one sample grid."""

    def __init__(self, t0_ms: float, dt_ms: float, bes: np.ndarray, interferometer: np.ndarray,
                 filterscopes: np.ndarray, chunk: int):
        """``bes`` is the 64-channel mean on the grid; the other signals are the raw records
        of the same time span, any sample rate, up-sampled here."""
        up = [upsample(np.asarray(row, dtype=float), bes.shape[0], chunk)
              for row in (*interferometer, *filterscopes)]
        size = min([bes.shape[0], *(u.shape[0] for u in up)])
        self.t0_ms, self.dt_ms = t0_ms, dt_ms
        self.diff = np.abs(np.diff(np.vstack([u[:size] for u in up]), axis=1))
        self.bes = bes[: size - 1]
        self.fs03 = up[3][: size - 1]
        self.half = max(1, round(BLUR_US / (dt_ms * 1000.0)))
        self.gap = max(1, round(GAP_US / (dt_ms * 1000.0)))


def dilate(mask: np.ndarray, half: int) -> np.ndarray:
    """``mask`` widened by ``half`` samples either side."""
    return maximum_filter1d(mask.astype(np.uint8), size=2 * half + 1, mode="constant", cval=0).astype(bool)


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Starts and (exclusive) stops of the runs of True in ``mask``."""
    edge = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return edge[0::2], edge[1::2]


def bridge(starts: np.ndarray, stops: np.ndarray, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Join runs separated by at most ``gap`` samples."""
    if starts.size == 0:
        return starts, stops
    cut = np.flatnonzero(starts[1:] - stops[:-1] > gap)
    return (np.concatenate(([starts[0]], starts[1:][cut])),
            np.concatenate((stops[:-1][cut], [stops[-1]])))


def votes(trace: Trace, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interferometer and filterscope votes for each value of ``thresholds`` (the 5 quantiles)."""
    candidates = trace.diff > thresholds[:, None]
    wide = [dilate(row, trace.half) for row in candidates]
    interferometer = wide[0] | wide[1]
    filterscopes = (wide[2].astype(np.int8) + wide[3] + wide[4]) > 1
    return interferometer, filterscopes


def spans_of(trace: Trace, interferometer: np.ndarray, filterscopes: np.ndarray,
             bes: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """ELM spans (sample indices, stop exclusive): all kinds of vote, small gaps closed."""
    elm = interferometer & filterscopes
    if bes is not None:
        elm = elm & bes
    return bridge(*runs(elm), trace.gap)


def detect(trace: Trace, eta: float = ETA, threshold: float = THRESHOLD,
           use_bes: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """ELM-O on one window: the spans of the ELMs it finds, as sample indices."""
    interferometer, filterscopes = votes(trace, np.quantile(trace.diff, eta, axis=1))
    return spans_of(trace, interferometer, filterscopes, trace.bes > threshold if use_bes else None)


def peaks(trace: Trace, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Sample index of the FS03 maximum inside each span."""
    return np.array([s + int(np.argmax(trace.fs03[s:e])) for s, e in zip(starts, stops)], dtype=int)


def bes_mean(signals: np.ndarray) -> np.ndarray:
    """The 64-channel BES mean the detector reads (the last 32 channels doubled)."""
    y = np.array(signals, dtype=np.float64)
    y[32:] *= 2.0
    return np.nan_to_num(y.mean(axis=0))


def source_slice(t_ms: np.ndarray, y: np.ndarray, t0_ms: float, t1_ms: float) -> np.ndarray | None:
    """The samples of record ``y`` (channels x time) in [t0, t1); None when it covers too little."""
    inside = (t_ms >= t0_ms) & (t_ms < t1_ms)
    if inside.sum() < 8:
        return None
    step = float(np.median(np.diff(t_ms[inside])))
    if inside.sum() < 0.9 * (t1_ms - t0_ms) / step:
        return None
    return np.nan_to_num(y[:, inside])


# ---------------------------------------------------------------- Smith's windows


def smith_index() -> pd.DataFrame:
    """One row per hand-labelled window of Smith's files, with the marked ELM region."""
    rows = []
    for rank, name in enumerate(SMITH_FILES):
        with h5py.File(SMITH_DIR / name, "r") as f:
            for key, group in f.items():
                t = group["time"][()]
                mark = np.flatnonzero(np.diff(np.concatenate(([0], group["labels"][()].astype(np.int8), [0]))))
                if mark.size != 2:
                    continue
                rows.append({
                    "file": name, "rank": rank, "group": key, "shot": int(group.attrs["shot"]),
                    "n": t.size, "t0_ms": float(t[0]), "dt_ms": float((t[-1] - t[0]) / (t.size - 1)),
                    "label_start": int(mark[0]), "label_stop": int(mark[1]),
                })
    d = pd.DataFrame(rows)
    d["label_t0_ms"] = d.t0_ms + d.label_start * d.dt_ms
    d["label_t1_ms"] = d.t0_ms + d.label_stop * d.dt_ms
    return d


def drop_duplicates(d: pd.DataFrame) -> pd.DataFrame:
    """Keep the first file's windows and, of the second's, those whose marked ELM is not yet taken."""
    keep = []
    for _, group in d.groupby("shot"):
        taken: list[tuple[float, float]] = []
        for index, row in group.sort_values(["rank", "label_t0_ms"]).iterrows():
            if any(row.label_t0_ms < b and row.label_t1_ms > a for a, b in taken):
                continue
            taken.append((row.label_t0_ms, row.label_t1_ms))
            keep.append(index)
    return d.loc[sorted(keep)].reset_index(drop=True)


def count(starts: np.ndarray, stops: np.ndarray, lo: int, hi: int) -> tuple[int, int, int]:
    """(tp, fp, fn) of one window by the paper's rule against the marked samples [lo, hi)."""
    overlap = (starts < hi) & (stops > lo)
    hit = bool(overlap.any())
    return int(hit), int((~overlap).sum()), int(not hit)


def smith_shot(job: tuple) -> tuple[list[dict], np.ndarray]:
    """Score every window of one shot at the published setting and over the whole (t, eta) grid."""
    shot, rows, signals_dir = job
    results, sweeps = [], []
    record = signals_dir / f"{shot}.npz"
    data = dict(np.load(record)) if record.exists() else None
    files: dict[str, h5py.File] = {}
    try:
        for row in rows:
            out = {"file": row["file"], "group": row["group"], "shot": shot, "status": "ok",
                   "label_start": row["label_start"], "label_stop": row["label_stop"]}
            sweep = np.zeros((3, len(THRESHOLDS), len(ETAS)), dtype=np.int16)
            if data is None:
                out["status"] = "no_signals"
                results.append(out)
                sweeps.append(sweep)
                continue
            if row["file"] not in files:
                files[row["file"]] = h5py.File(SMITH_DIR / row["file"], "r")
            group = files[row["file"]][row["group"]]
            t = group["time"][()]
            bes = bes_mean(group["signals"][()])
            t_end = row["t0_ms"] + row["n"] * row["dt_ms"]
            interferometer = source_slice(data["t_int_ms"], data["interferometer"], row["t0_ms"], t_end)
            filterscopes = source_slice(data["t_fs_ms"], data["filterscopes"], row["t0_ms"], t_end)
            if interferometer is None or filterscopes is None:
                out["status"] = "short_signals"
                results.append(out)
                sweeps.append(sweep)
                continue
            trace = Trace(float(t[0]), row["dt_ms"], bes, interferometer, filterscopes,
                          chunk=round(CHUNK_MS / row["dt_ms"]))
            lo, hi = row["label_start"], min(row["label_stop"], trace.bes.size)
            quantiles = np.quantile(trace.diff, ETAS, axis=1)
            for j in range(len(ETAS)):
                interferometer_vote, filterscope_vote = votes(trace, quantiles[j])
                if np.isclose(ETAS[j], ETA):
                    published = (interferometer_vote, filterscope_vote)
                for i, threshold in enumerate(THRESHOLDS):
                    starts, stops = spans_of(trace, interferometer_vote, filterscope_vote, trace.bes > threshold)
                    sweep[:, i, j] = count(starts, stops, lo, hi)
            starts, stops = spans_of(trace, *published, trace.bes > THRESHOLD)
            window = slice(lo, hi)
            at_bes = lo + int(np.argmax(trace.bes[window]))
            at_fs = lo + int(np.argmax(trace.fs03[window]))
            out.update({
                "t0_ms": row["t0_ms"], "label_t0_ms": row["label_t0_ms"], "label_t1_ms": row["label_t1_ms"],
                "n": row["n"], "spans": int(starts.size),
                "interferometer_vote": bool(published[0][window].any()),
                "filterscope_vote": bool(published[1][window].any()),
                "bes_vote": bool((trace.bes[window] > THRESHOLD).any()),
                "bes_max": float(trace.bes[window].max()),
                "lag_us": (at_fs - at_bes) * row["dt_ms"] * 1000.0,
            })
            results.append(out)
            sweeps.append(sweep)
    finally:
        for f in files.values():
            f.close()
    return results, np.stack(sweeps) if sweeps else np.zeros((0, 3, len(THRESHOLDS), len(ETAS)), dtype=np.int16)


def run_smith(args: argparse.Namespace) -> None:
    work = args.work
    index = drop_duplicates(smith_index())
    if args.shots:
        index = index[index.shot.isin(args.shots)].reset_index(drop=True)
    print(f"{len(index)} windows, {index.shot.nunique()} shots", flush=True)
    signals = args.signals or work / "signals"
    jobs = [(int(shot), g.to_dict("records"), signals) for shot, g in index.groupby("shot")]
    results: dict[int, tuple[list[dict], np.ndarray]] = {}
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job, out in zip(jobs, pool.map(smith_shot, jobs)):
            results[job[0]] = out
            print(f"{job[0]} {len(job[1])} windows {time.time() - started:.0f}s", flush=True)
    records = [r for job in jobs for r in results[job[0]][0]]
    sweeps = np.concatenate([results[job[0]][1] for job in jobs])
    frame = pd.DataFrame(records)
    frame.to_csv(work / "smith_windows.csv", index=False)
    np.savez_compressed(work / "smith_sweep.npz", counts=sweeps, thresholds=THRESHOLDS, etas=ETAS)
    print(frame.status.value_counts().to_dict())


# ---------------------------------------------------------------- the review labels


def bes_samples(shot: int) -> int:
    """Samples in a shot's corpus BES record; shots without BES hold a one-sample stub."""
    try:
        with h5py.File(CORPUS / f"{shot}_processed.h5", "r") as f:
            return int(f["bes/xdata"].shape[0])
    except (OSError, KeyError):
        return 0


def review_shot(job: tuple) -> dict:
    """ELM-O over the reviewed time of one shot, in 200 ms chunks, for each variant."""
    shot, window, signals_dir = job
    data = dict(np.load(signals_dir / f"{shot}.npz"))
    rows, cover, lags = [], [], []
    skipped = 0
    with h5py.File(CORPUS / f"{shot}_processed.h5", "r") as f:
        x = f["bes/xdata"]
        n, x0, x1 = x.shape[0], float(x[0]), float(x[-1])
        dt_ms = (x1 - x0) / (n - 1) * 1000.0
        t0_ms = x0 * 1000.0
        lo = int(np.ceil((max(window[0], t0_ms) - t0_ms) / dt_ms))
        hi = int(np.floor((min(window[1], t0_ms + (n - 1) * dt_ms) - t0_ms) / dt_ms))
        step = round(CHUNK_MS / dt_ms)
        ydata = f["bes/ydata"]
        for a in range(lo, hi, step):
            b = min(a + step, hi)
            if b - a < step // 10:
                continue
            c0, c1 = t0_ms + a * dt_ms, t0_ms + b * dt_ms
            interferometer = source_slice(data["t_int_ms"], data["interferometer"], c0, c1)
            filterscopes = source_slice(data["t_fs_ms"], data["filterscopes"], c0, c1)
            if interferometer is None or filterscopes is None:
                skipped += 1
                continue
            trace = Trace(c0, dt_ms, bes_mean(ydata[:, a:b]), interferometer, filterscopes, chunk=b - a)
            cover.append((c0, c1))
            for name, eta, use_bes in VARIANTS:
                starts, stops = detect(trace, eta, THRESHOLD, use_bes)
                peak = peaks(trace, starts, stops)
                for s, e, p in zip(starts, stops, peak):
                    rows.append((shot, name, c0 + s * dt_ms, c0 + e * dt_ms, c0 + p * dt_ms))
                    if name == "paper":
                        lags.append((p - (s + int(np.argmax(trace.bes[s:e])))) * dt_ms * 1000.0)
    return {"shot": shot, "elms": rows, "cover": cover, "skipped": skipped, "dt_ms": dt_ms,
            "lag_us": float(np.median(lags)) if lags else float("nan")}


def review_labels() -> pd.DataFrame:
    """The review spans with a ``kind`` column: absent, individual, crowd, uncertain, not observable."""
    d = pd.read_csv(REVIEW)
    crowd = d["attrs"].map(lambda a: json.loads(a).get("iscrowd", 0) if isinstance(a, str) else 0)
    d["kind"] = np.select(
        [d.category == 0, (d.category == 1) & (crowd == 1), d.category == 1, d.category == 2],
        ["absent", "crowd", "individual", "uncertain"], default="not_observable")
    return d


def run_review(args: argparse.Namespace) -> None:
    work = args.work
    labels = review_labels()
    shots = sorted(labels.shot.unique())
    if args.shots:
        shots = [s for s in shots if s in set(args.shots)]
    windows = labels.groupby("shot").agg(lo=("t_start", "min"), hi=("t_end", "max"))
    signals = args.signals or work / "signals"
    fetched = [s for s in shots if (signals / f"{s}.npz").exists()]
    ready = [s for s in fetched if bes_samples(s) > 1]
    print(f"{len(ready)} of {len(shots)} shots have the fetched signals and BES in the corpus "
          f"({len(shots) - len(fetched)} not fetched, {len(fetched) - len(ready)} without BES)", flush=True)
    jobs = [(int(s), (float(windows.lo[s]), float(windows.hi[s])), signals) for s in ready]
    elms, cover, summary = [], [], []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job, out in zip(jobs, pool.map(review_shot, jobs)):
            elms += out["elms"]
            cover += [(job[0], c0, c1) for c0, c1 in out["cover"]]
            paper = sum(1 for r in out["elms"] if r[1] == "paper")
            summary.append({"shot": job[0], "chunks": len(out["cover"]), "skipped_chunks": out["skipped"],
                            "analysed_ms": sum(c1 - c0 for c0, c1 in out["cover"]), "elms": paper,
                            "median_lag_us": out["lag_us"], "dt_ms": out["dt_ms"]})
            print(f"{job[0]} {paper} ELMs {time.time() - started:.0f}s", flush=True)
    pd.DataFrame(elms, columns=["shot", "variant", "t_start_ms", "t_end_ms", "peak_ms"]).to_csv(
        work / "review_elms.csv", index=False)
    pd.DataFrame(cover, columns=["shot", "t_start_ms", "t_end_ms"]).to_csv(work / "review_coverage.csv", index=False)
    pd.DataFrame(summary).to_csv(work / "review_shots.csv", index=False)


# ---------------------------------------------------------------- scoring


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def prf(tp: float, fp: float, fn: float) -> dict:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p + r else float("nan")
    return {"precision": p, "recall": r, "f1": f1}


def ci(values: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def smith_results(work: pd.DataFrame, counts: np.ndarray) -> dict:
    """The paper's scoring over Smith's windows: published setting, Table II, curves, failures."""
    ok = (work.status == "ok").to_numpy()
    shots = work.shot.to_numpy()
    i1, j = THRESHOLDS.index(THRESHOLD), int(np.argmin(np.abs(ETAS - ETA)))

    def score(sel: np.ndarray, i: int = i1, jj: int = j) -> dict:
        tp, fp, fn = (int(counts[sel, k, i, jj].sum()) for k in range(3))
        return {"windows": int(sel.sum()), "tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)}

    out: dict = {"windows": len(work), "status": work.status.value_counts().to_dict(),
                 "shots": int(work.shot.nunique()), "scored_windows": int(ok.sum()),
                 "published_setting": {"threshold": THRESHOLD, "eta": ETA, **score(ok)}}
    out["published_setting"]["precision_x_recall"] = (
        out["published_setting"]["precision"] * out["published_setting"]["recall"])
    out["by_file"] = {name: score(ok & (work.file == name).to_numpy()) for name in SMITH_FILES}
    out["table"] = {}
    for i, t in enumerate(THRESHOLDS):
        rows = [(score(ok, i, jj), float(ETAS[jj])) for jj in range(len(ETAS))]
        best, eta = max(rows, key=lambda r: (r[0]["precision"] * r[0]["recall"]) if r[0]["tp"] else -1)
        out["table"][str(t)] = {"eta": eta, **best, "precision_x_recall": best["precision"] * best["recall"],
                                "paper": dict(zip(("eta", "precision", "recall", "auc"), PAPER[t]))}
    curve = [score(ok, i1, jj) for jj in range(len(ETAS))]
    out["curve_t1"] = [{"eta": float(e), "precision": c["precision"], "recall": c["recall"]}
                       for e, c in zip(ETAS, curve)]
    # bootstrap over shots at the published setting
    unique = np.unique(shots[ok])
    per_shot = np.array([[counts[ok & (shots == s), k, i1, j].sum() for k in range(3)] for s in unique])
    rng = np.random.default_rng(BOOT_SEED)
    boot = []
    for _ in range(REPLICATES):
        tp, fp, fn = per_shot[rng.integers(0, len(unique), len(unique))].sum(axis=0)
        boot.append((tp / (tp + fp), tp / (tp + fn)))
    boot = np.array(boot)
    out["published_setting"]["ci95"] = {"precision": ci(boot[:, 0]), "recall": ci(boot[:, 1]),
                                        "precision_x_recall": ci(boot[:, 0] * boot[:, 1])}
    # what the false negatives lack
    miss = ok & (counts[:, 2, i1, j] > 0)
    out["false_negatives"] = {
        "windows": int(miss.sum()),
        "no_interferometer_vote": int((miss & ~work.interferometer_vote.fillna(False).to_numpy(bool)).sum()),
        "no_filterscope_vote": int((miss & ~work.filterscope_vote.fillna(False).to_numpy(bool)).sum()),
        "no_bes_vote": int((miss & ~work.bes_vote.fillna(False).to_numpy(bool)).sum()),
        "bes_max_median": float(work.bes_max[miss].median()) if miss.any() else None,
        "shots_with_any": int(work.shot[miss].nunique()),
    }
    per_shot_miss = pd.DataFrame({"shot": shots[ok], "miss": miss[ok]}).groupby("shot").miss.agg(["sum", "size"])
    out["false_negatives"]["shots_with_half_or_more"] = int((per_shot_miss["sum"] >= 0.5 * per_shot_miss["size"]).sum())
    interferometer_vote = work.interferometer_vote.fillna(False).to_numpy(bool)
    filterscope_vote = work.filterscope_vote.fillna(False).to_numpy(bool)
    out["false_negatives"]["top_shots"] = {
        int(s): {"missed": int(r["sum"]), "windows": int(r["size"]),
                 "with_interferometer_vote": int((miss & (shots == s) & interferometer_vote).sum()),
                 "with_filterscope_vote": int((miss & (shots == s) & filterscope_vote).sum())}
        for s, r in per_shot_miss.sort_values("sum", ascending=False).head(5).iterrows()}
    lag = work.lag_us.abs()
    out["lag_us"] = {"median_abs_all": float(lag[ok].median()), "median_abs_false_negatives": float(lag[miss].median()) if miss.any() else None,
                     "windows_over_1ms": int((lag[ok] > 1000).sum())}
    return out


def merged(cover: pd.DataFrame, tol: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """The analysed time of one shot as separate intervals: chunks that touch are joined."""
    starts: list[float] = []
    stops: list[float] = []
    for s, e in zip(*(cover.sort_values("t_start_ms")[c] for c in ("t_start_ms", "t_end_ms"))):
        if stops and s <= stops[-1] + tol:
            stops[-1] = max(stops[-1], e)
        else:
            starts.append(s)
            stops.append(e)
    return np.array(starts), np.array(stops)


def bin_table(spans: pd.DataFrame, cover: pd.DataFrame, shot_labels: pd.DataFrame) -> dict:
    """One shot's counts against ``spans`` (start/stop columns in ms, sorted, not overlapping).

    Bins of 50 ms wholly inside one labelled span and inside the analysed time give the
    bin counts; a span with at least half its length analysed gives the span counts.
    """
    starts, stops = spans.t_start_ms.to_numpy(float), spans.t_end_ms.to_numpy(float)
    cov0, cov1 = merged(cover)

    def any_in(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        if not len(starts):
            return np.zeros(len(lo), dtype=bool)
        i = np.minimum(np.searchsorted(stops, lo, side="right"), len(starts) - 1)
        return (stops[i] > lo) & (starts[i] < hi)

    out = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "fp_near": 0, "tn_near": 0, "crowd_bins": 0, "crowd_hit": 0,
           "individual_bins": 0, "individual_hit": 0, "individual_spans": 0, "individual_span_hit": 0,
           "absent_spans": 0, "absent_span_alarm": 0, "crowd_spans": 0, "crowd_coverage": []}
    present = shot_labels[shot_labels.kind.isin(("individual", "crowd"))]
    present_start, present_stop = present.t_start.to_numpy(float), present.t_end.to_numpy(float)
    for row in shot_labels.itertuples():
        if row.kind not in ("absent", "individual", "crowd") or row.t_end <= row.t_start:
            continue
        analysed = np.clip(np.minimum(cov1, row.t_end) - np.maximum(cov0, row.t_start), 0, None).sum()
        if analysed < 0.5 * (row.t_end - row.t_start):
            continue
        edges = np.arange(int(np.ceil(row.t_start / BIN_MS)), int(np.floor(row.t_end / BIN_MS))) * BIN_MS
        k = np.searchsorted(cov0, edges, side="right") - 1
        edges = edges[(k >= 0) & (cov1[np.maximum(k, 0)] >= edges + BIN_MS)]
        hit = any_in(edges, edges + BIN_MS)
        span_hit = bool(any_in(np.array([float(row.t_start)]), np.array([float(row.t_end)]))[0])
        if row.kind == "absent":
            if present_start.size and edges.size:
                gap = np.maximum(np.maximum(present_start[None, :] - (edges[:, None] + BIN_MS),
                                            edges[:, None] - present_stop[None, :]), 0).min(axis=1)
                near = gap <= NEAR_MS
            else:
                near = np.zeros(edges.size, dtype=bool)
            out["fp"] += int(hit.sum())
            out["tn"] += int((~hit).sum())
            out["fp_near"] += int((hit & near).sum())
            out["tn_near"] += int((~hit & near).sum())
            out["absent_spans"] += 1
            out["absent_span_alarm"] += int(span_hit)
            continue
        out["tp"] += int(hit.sum())
        out["fn"] += int((~hit).sum())
        if row.kind == "crowd":
            out["crowd_bins"] += int(edges.size)
            out["crowd_hit"] += int(hit.sum())
            out["crowd_spans"] += 1
            if edges.size:
                out["crowd_coverage"].append(float(hit.mean()))
        else:
            out["individual_bins"] += int(edges.size)
            out["individual_hit"] += int(hit.sum())
            out["individual_spans"] += 1
            out["individual_span_hit"] += int(span_hit)
    return out


def detections(peak_ms: np.ndarray, shot_labels: pd.DataFrame) -> dict:
    """ELMs counted by the kind of reviewed span their FS03 peak falls in."""
    labelled = shot_labels.sort_values("t_start")
    first, last = labelled.t_start.to_numpy(float), labelled.t_end.to_numpy(float)
    kinds = labelled.kind.to_numpy()
    i = np.maximum(np.searchsorted(first, peak_ms, side="right") - 1, 0)
    kind = np.where((peak_ms >= first[i]) & (peak_ms < last[i]), kinds[i], "none")
    present = int(np.isin(kind, ("individual", "crowd")).sum())
    absent = int((kind == "absent").sum())
    # absent-span ELMs close to a present span: label edges are fuzzy there
    marked = labelled[labelled.kind.isin(("individual", "crowd"))]
    gap = np.full(absent, np.inf)
    if absent and len(marked):
        at = peak_ms[kind == "absent"][:, None]
        gap = np.maximum(np.maximum(marked.t_start.to_numpy(float)[None, :] - at,
                                    at - marked.t_end.to_numpy(float)[None, :]), 0).min(axis=1)
    return {"present": present, "absent": absent, "other": int(kind.size - present - absent),
            "absent_within_50ms": int((gap <= 50).sum()), "absent_within_500ms": int((gap <= NEAR_MS).sum())}


def span_rates(labels: pd.DataFrame, elms: pd.DataFrame, cover: pd.DataFrame) -> pd.DataFrame:
    """Per absent, individual or crowd span of the review: the ELM-O ELMs (paper variant) peaking in it."""
    rows = []
    found = elms[elms.variant == "paper"]
    spans = labels[labels.kind.isin(("absent", "individual", "crowd")) & labels.shot.isin(cover.shot.unique())]
    for shot, group in spans.groupby("shot"):
        peak = found[found.shot == shot].peak_ms.to_numpy(float)
        cov0, cov1 = merged(cover[cover.shot == shot])
        for r in group.itertuples():
            analysed = float(np.clip(np.minimum(cov1, r.t_end) - np.maximum(cov0, r.t_start), 0, None).sum())
            if analysed <= 0 or analysed < 0.5 * (r.t_end - r.t_start):
                continue
            n = int(((peak >= r.t_start) & (peak < r.t_end)).sum())
            rows.append({"shot": int(shot), "kind": r.kind, "t_start": int(r.t_start), "t_end": int(r.t_end),
                         "analysed_ms": analysed, "elms": n, "per_second": n / (analysed / 1000.0)})
    return pd.DataFrame(rows).sort_values("elms", ascending=False).reset_index(drop=True)


def sum_counts(rows: list[dict], keys: tuple[str, ...]) -> dict:
    return {k: sum(r[k] for r in rows) for k in keys}


def review_results(work: Path) -> dict:
    labels = review_labels()
    elms = pd.read_csv(work / "review_elms.csv")
    cover = pd.read_csv(work / "review_coverage.csv")
    shots_table = pd.read_csv(work / "review_shots.csv")
    clock = pd.read_csv(ELM_CLOCK)
    clock = clock[clock.category == 1].rename(columns={"t_start": "t_start_ms", "t_end": "t_end_ms"})
    shots = sorted(cover.shot.unique())
    keys = ("tp", "fp", "fn", "tn", "fp_near", "tn_near", "crowd_bins", "crowd_hit", "individual_bins", "individual_hit",
            "individual_spans", "individual_span_hit", "absent_spans", "absent_span_alarm", "crowd_spans")
    per: dict[str, dict[int, dict]] = {}
    det: dict[str, dict[int, dict]] = {}
    for variant in [v[0] for v in VARIANTS] + ["elm_clock"]:
        per[variant], det[variant] = {}, {}
        for shot in shots:
            sl = labels[labels.shot == shot]
            cv = cover[cover.shot == shot]
            if variant == "elm_clock":
                spans = clock[clock.shot == shot].sort_values("t_start_ms")
            else:
                spans = elms[(elms.shot == shot) & (elms.variant == variant)].sort_values("t_start_ms")
                det[variant][shot] = detections(spans.peak_ms.to_numpy(float), sl)
            per[variant][shot] = bin_table(spans, cv, sl)
    rates = span_rates(labels, elms, cover)
    alarms = rates[rates.kind == "absent"]
    alarms.drop(columns="kind").to_csv(work / "absent_with_elms.csv", index=False)
    by_kind = {k: {"spans": len(g), "elms": int(g.elms.sum()), "median_ms": float(g.analysed_ms.median()),
                   "per_second_quartiles": [float(v) for v in g.per_second.quantile([0.25, 0.5, 0.75])]}
               for k, g in rates.groupby("kind")}
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.integers(0, len(shots), len(shots)) for _ in range(REPLICATES)]
    out: dict = {"absent_spans": {
        "spans": len(alarms), "with_any_elm": int((alarms.elms > 0).sum()),
        "with_5_or_more": int((alarms.elms >= 5).sum()), "elms_in_them": int(alarms.elms.sum()),
        "shots_with_5_or_more": int(alarms.shot[alarms.elms >= 5].nunique()),
        "top": alarms.drop(columns="kind").head(8).to_dict("records")}, "span_rates": by_kind, "shots": len(shots), "shots_in_labels": int(labels.shot.nunique()),
                 "analysed_hours": float(cover.eval("t_end_ms - t_start_ms").sum() / 3.6e6),
                 "skipped_chunks": int(shots_table.skipped_chunks.sum()),
                 "median_lag_us": float(shots_table.median_lag_us.median()),
                 "shots_with_lag_over_1ms": int((shots_table.median_lag_us.abs() > 1000).sum())}
    for variant in per:
        rows = [per[variant][s] for s in shots]
        total = sum_counts(rows, keys)

        def metrics(sel: list[dict]) -> dict:
            s = sum_counts(sel, keys)
            return {
                **prf(s["tp"], s["fp"], s["fn"]),
                "false_alarm_bin_rate": s["fp"] / (s["fp"] + s["tn"]) if s["fp"] + s["tn"] else float("nan"),
                "false_alarm_bin_rate_near": s["fp_near"] / (s["fp_near"] + s["tn_near"])
                if s["fp_near"] + s["tn_near"] else float("nan"),
                "false_alarm_bin_rate_far": (s["fp"] - s["fp_near"]) / (s["fp"] + s["tn"] - s["fp_near"] - s["tn_near"])
                if s["fp"] + s["tn"] - s["fp_near"] - s["tn_near"] else float("nan"),
                "crowd_bin_recall": s["crowd_hit"] / s["crowd_bins"] if s["crowd_bins"] else float("nan"),
                "individual_span_recall": s["individual_span_hit"] / s["individual_spans"] if s["individual_spans"] else float("nan"),
                "absent_span_alarm_rate": s["absent_span_alarm"] / s["absent_spans"] if s["absent_spans"] else float("nan"),
            }

        res = {"counts": total, **metrics(rows)}
        boot = [metrics([rows[i] for i in d]) for d in draws]
        res["ci95"] = {k: ci(np.array([b[k] for b in boot])) for k in boot[0]}
        coverage = [c for r in rows for c in r["crowd_coverage"]]
        res["crowd_span_coverage_median"] = float(np.median(coverage)) if coverage else None
        if det[variant]:
            d = sum_counts([det[variant][s] for s in shots], DETECTION_KEYS)
            res["detections"] = {**d, "precision": d["present"] / (d["present"] + d["absent"])
                                 if d["present"] + d["absent"] else float("nan")}
            boots = []
            for dr in draws:
                dd = sum_counts([det[variant][shots[i]] for i in dr], ("present", "absent"))
                boots.append(dd["present"] / max(dd["present"] + dd["absent"], 1))
            res["detections"]["precision_ci95"] = ci(np.array(boots))
        out[variant] = res
    return out


def run_evaluate(args: argparse.Namespace) -> None:
    work, out_dir = args.work, args.out
    record: dict = {"git": git_sha(), "created": datetime.now(UTC).isoformat(timespec="seconds"),
                    "paper": {"table": {str(t): dict(zip(("eta", "precision", "recall", "auc"), v)) for t, v in PAPER.items()}},
                    "parameters": {"eta": ETA, "eta_code_default": ETA_CODE, "bes_threshold_V": THRESHOLD,
                                   "blur_us": BLUR_US, "gap_us": GAP_US, "chunk_ms": CHUNK_MS}}
    windows = work / "smith_windows.csv"
    if windows.exists():
        record["smith"] = smith_results(pd.read_csv(windows), np.load(work / "smith_sweep.npz")["counts"])
    if (work / "review_elms.csv").exists():
        record["review"] = review_results(work)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k in ("git", "created")}))
    if "smith" in record:
        print("smith published setting:", json.dumps({k: v for k, v in record["smith"]["published_setting"].items()}))
    if "review" in record:
        for variant in [v[0] for v in VARIANTS] + ["elm_clock"]:
            r = record["review"][variant]
            print(variant, {k: round(r[k], 3) for k in ("precision", "recall", "f1", "false_alarm_bin_rate",
                                                       "crowd_bin_recall", "individual_span_recall")})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name, fn in (("smith", run_smith), ("review", run_review), ("evaluate", run_evaluate)):
        p = sub.add_parser(name)
        p.add_argument("--work", type=Path, default=DEFAULT_WORK)
        p.add_argument("--out", type=Path, default=DEFAULT_OUT)
        p.add_argument("--signals", type=Path, help="fetched records (default: <work>/signals)")
        p.add_argument("--workers", type=int, default=16)
        p.add_argument("--shots", type=int, nargs="+", help="only these shots")
        p.set_defaults(func=fn)
    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
