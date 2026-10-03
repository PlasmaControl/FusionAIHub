"""Generate additive sawtooth labels from read-only stores and inspect their scope.

Run ``labels`` on the cohort, then ``labels --population`` (resumes shot files),
and ``validate``. No network resolver is called. Large signals/labels/figures are
written only to --work; cohort CSV shards and evaluation JSON go to the worktree.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from labeler.config import Paths
from labeler.events import equilibrium, schema
from labeler.events.panels import ece_geometry
from labeler.events.verify import NoDataError
from labeler.sawtooth.metrics import (
    aggregate,
    event_cells,
    score_histogram,
    spans_at,
)
from labeler.sawtooth.physics import Rule, detect, noise_calibration

REPO = Path(__file__).resolve().parents[2]
WORK = Paths.from_env().root / "round4/saw"
SEED = 20261003
FS = 10000
REVIEW = Paths.from_env().label_tables / "sawtooth_oscillation/review/labels.csv"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sampled(group, *, rows=None, fs=FS):
    """Read at at least 50 kHz then FIR-decimate to 10 kHz.

    ECE's 500 kHz archive is first sampled at 50 kHz to bound IO and memory.
    That first sampling step has no antialias filter: above-25-kHz noise may
    alias. The second step is antialiased. This limitation is recorded in docs.
    Low-cadence auxiliary signals are interpolated, without claiming new bandwidth.
    """
    x, y = group["xdata"], group["ydata"]
    if len(x) < 32 or y.ndim != 2 or y.shape[1] != len(x):
        raise ValueError("absent waveform or incompatible clock")
    dt = (float(x[-1]) - float(x[0])) / (len(x) - 1)
    if dt <= 0:
        raise ValueError("nonincreasing clock")
    stride = max(1, round(1 / dt / (5 * fs)))
    tx = np.asarray(x[::stride], dtype=float)
    # Verify the sampled clock; detector rejects gaps rather than bridging them.
    if not np.allclose(np.diff(tx), dt * stride, rtol=0.05, atol=1e-7):
        raise ValueError("nonuniform waveform")
    values = np.asarray(
        y[slice(None) if rows is None else rows, ::stride], dtype=np.float32
    )
    factor = max(1, round(1 / (dt * stride * fs)))
    if factor > 1:
        # Keep missing supports missing after the FIR filter, not zero-filled labels.
        valid = np.isfinite(values)
        values = resample_poly(np.where(valid, values, 0), 1, factor, axis=1)
        support = resample_poly(valid.astype(np.float32), 1, factor, axis=1)
        values[support < 0.999] = np.nan
        tx = tx[::factor][: values.shape[1]]
    return tx, values


def interpolate(t, tx, y):
    return np.stack([np.interp(t, tx, row, left=np.nan, right=np.nan) for row in y])


def local_scalar(shot, name, paths):
    try:
        array = equilibrium.signal(shot, name, paths, fetch=False)
        return array.x, array.y[0]
    except (NoDataError, ValueError):
        return None


def process_shot(job):
    shot, work, keep_signal, window = job
    work, paths = Path(work), Paths.from_env()
    record = work / "shots" / f"{shot}.json"
    signal = work / "signals" / f"{shot}.npz"
    if record.exists() and (not keep_signal or signal.exists()):
        return json.loads(record.read_text())
    started = time.monotonic()
    result = {"shot": int(shot), "crashes": [], "intervals": []}
    try:
        with h5py.File(paths.corpus_file(shot), "r", locking=False) as file:
            if "ece" not in file:
                raise ValueError("no ECE group")
            t, y = sampled(file["ece"])
            y[(y < 0) | (y > 100)] = np.nan
            if len(y) != 48:
                raise ValueError(f"ECE array has {len(y)} channels, expected 48")
            if window is None:
                ip = local_scalar(shot, "ip", paths)
                if ip is not None:
                    active = np.abs(ip[1]) > 0.1 * np.nanmax(np.abs(ip[1]))
                    if active.any():
                        window = (float(ip[0][active][0]), float(ip[0][active][-1]))
                if window is None:
                    window = (max(0.05, float(t[0])), float(t[-1]))
            keep = (t >= window[0]) & (t <= window[1])
            t, y = t[keep], y[:, keep]
            if len(t) < 1000 or np.isfinite(y).mean() < 0.1:
                raise ValueError("insufficient finite ECE in analysis window")
            result["window_s"] = [float(t[0]), float(t[-1])]
            result["window_source"] = (
                "cohort" if keep_signal else "local Ip or ECE extent"
            )
            sxr, dalpha = None, None
            result["sxr_channels"] = []
            if "sxr" in file and file["sxr/ydata"].shape[-1] > 32:
                # First lit fan among the four fans used by the established detector.
                for first in (0, 32, 160, 192):
                    ts, ys = sampled(file["sxr"], rows=slice(first, first + 32))
                    lit = np.isfinite(ys).mean(axis=1) > 0.75
                    if lit.sum() < 4:
                        continue
                    selected = np.flatnonzero(lit)
                    selected = selected[np.argsort(-np.nanmedian(ys[lit], axis=1))[:8]]
                    sxr = interpolate(t, ts, ys[selected])
                    result["sxr_channels"] = (first + selected).tolist()
                    break
            if "filterscopes" in file and file["filterscopes/ydata"].shape[-1] > 32:
                td, yd = sampled(file["filterscopes"], rows=slice(0, 8))
                finite = np.isfinite(yd)
                counts = finite.sum(axis=0)
                mean = np.divide(
                    np.where(finite, yd, 0).sum(axis=0),
                    counts,
                    out=np.full(len(td), np.nan),
                    where=counts > 0,
                )
                dalpha = td, mean
            baseline = np.zeros((4, len(t)), dtype=np.float32)
            for i, rows in enumerate((slice(20, 28), slice(8, 16))):
                v = y[rows]
                counts = np.isfinite(v).sum(axis=0)
                baseline[i] = np.divide(
                    np.where(np.isfinite(v), v, 0).sum(axis=0),
                    counts,
                    out=np.full(len(t), np.nan),
                    where=counts > 0,
                )
            result["mirnov_available"] = False
            if keep_signal and "mirnov" in file and file["mirnov/ydata"].shape[-1] > 32:
                tm, ym = sampled(file["mirnov"], rows=slice(0, 2))
                baseline[2] = interpolate(t, tm, ym).mean(axis=0)
                result["mirnov_available"] = bool(np.isfinite(baseline[2]).mean() > 0.5)
        ip, qmin = local_scalar(shot, "ip", paths), local_scalar(shot, "qmin", paths)
        if ip is not None:
            baseline[3] = np.interp(t, ip[0], ip[1], left=np.nan, right=np.nan) / 1e6
        result["ip_available"] = ip is not None
        result["qmin_available"] = qmin is not None
        try:
            geometry = ece_geometry.load_geometry(shot, paths)
            result["geometry"] = (
                "calibrated_psi" if geometry.psi is not None else "q_only"
            )
        except NoDataError:
            geometry = None
            result["geometry"] = "unavailable"
        detected = detect(
            t,
            y,
            shot=shot,
            qmin=qmin,
            geometry=geometry,
            dalpha=dalpha,
            sxr=sxr,
            core_channels=range(20, 36),
            ip=ip,
        )
        result.update(
            candidates=detected.candidates,
            rejected=detected.rejected,
            crashes=[
                {"time_s": e.t0_s, "confidence": e.confidence, "attrs": e.attrs}
                for e in detected.crashes
            ],
            intervals=[
                {"start_s": e.t0_s, "end_s": e.t1_s, "attrs": e.attrs}
                for e in detected.intervals
            ],
        )
        if keep_signal:
            signal.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(signal, t=t, y=y.astype(np.float16), baseline=baseline)
    except (OSError, KeyError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    save_json(record, result)
    return result


def export_csv(records, destination):
    destination.mkdir(parents=True, exist_ok=True)
    columns = ["shot", "category", "t_start", "t_end", "crowd", "confidence", "attrs"]
    part, bytes_used, handle = 0, 0, None
    try:
        for record in records:
            rows = [
                (r["time_s"], r["time_s"], False, r["confidence"], r["attrs"])
                for r in record["crashes"]
            ]
            rows += [
                (r["start_s"], r["end_s"], True, 1.0, r["attrs"])
                for r in record["intervals"]
            ]
            for start, end, crowd, confidence, attrs in rows:
                if handle is None or bytes_used > 1_400_000:
                    if handle is not None:
                        handle.close()
                    handle = (destination / f"labels-{part:03d}.csv").open(
                        "w", newline=""
                    )
                    writer = csv.writer(handle)
                    writer.writerow(columns)
                    part, bytes_used = part + 1, 0
                text = json.dumps(attrs, allow_nan=False, separators=(",", ":"))
                writer.writerow(
                    [
                        record["shot"],
                        1,
                        round(start * 1000, 4),
                        round(end * 1000, 4),
                        crowd,
                        confidence,
                        text,
                    ]
                )
                bytes_used += len(text) * 1.2 + 100
    finally:
        if handle is not None:
            handle.close()


def labels(args):
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    reviewed = pd.read_csv(REVIEW)
    shots = (
        sorted(
            int(p.name.split("_")[0])
            for p in Paths.from_env().corpus.glob("*_processed.h5")
        )
        if args.population
        else sorted(set(cohort.shot) | set(reviewed.shot))
    )
    if args.shots:
        shots = sorted(set(args.shots))
    windows = {
        int(r.shot): (r.window_start_ms / 1000, r.window_end_ms / 1000)
        for r in cohort.itertuples()
    }
    jobs = [
        (s, str(args.work), s in windows or s in set(reviewed.shot), windows.get(s))
        for s in shots
    ]
    begun = time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(pool.map(process_shot, jobs, chunksize=1)):
            records.append(record)
            if i % 50 == 0:
                print(
                    f"{i + 1}/{len(jobs)} shot {record['shot']} crashes {len(record['crashes'])} error {record.get('error', '')}",
                    flush=True,
                )
    summary = {
        "scope": "population" if args.population else "cohort_plus_review",
        "requested_shots": shots,
        "requested_count": len(shots),
        "processed_shots": [r["shot"] for r in records if "error" not in r],
        "errors": {r["shot"]: r["error"] for r in records if "error" in r},
        "crashes": sum(len(r["crashes"]) for r in records),
        "intervals": sum(len(r["intervals"]) for r in records),
        "positive_shots": sum(bool(r["intervals"]) for r in records),
        "qmin_shots": sum(r.get("qmin_available", False) for r in records),
        "geometry_counts": dict(
            Counter(r.get("geometry", "not_processed") for r in records)
        ),
        "sxr_shots": sum(bool(r.get("sxr_channels")) for r in records),
        "rejected": dict(
            sum((Counter(r.get("rejected", {})) for r in records), Counter())
        ),
        "split_counts": cohort.split.value_counts().to_dict(),
        "rule": asdict(Rule()),
        "seconds": round(time.monotonic() - begun, 1),
        "source_records": str(args.work / "shots"),
        "signals": str(args.work / "signals"),
    }
    scope = "population" if args.population else "cohort"
    save_json(args.work / f"{scope}_labels.json", summary)
    save_json(REPO / f"outputs/labeler/sawtooth/{scope}_labels.json", summary)
    if not args.population:
        export_csv(
            records, REPO / "data/events/sawtooth_oscillation/extend_saw_physics"
        )
        save_json(
            REPO / "outputs/labeler/sawtooth/noise_calibration.json",
            noise_calibration(),
        )
    else:
        export_csv(records, args.work / "population_labels")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if not isinstance(v, list)},
            allow_nan=False,
        ),
        flush=True,
    )


def records_at(work, shots):
    return [
        json.loads((work / "shots" / f"{int(s)}.json").read_text())
        for s in shots
        if (work / "shots" / f"{int(s)}.json").exists()
    ]


def validate(args):
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    review = pd.read_csv(REVIEW)
    reviewed = []
    for record in records_at(args.work, sorted(review.shot.unique())):
        rows = review[review.shot == record["shot"]]
        if "window_s" not in record:
            continue
        lo, hi = record["window_s"]
        t = np.arange(
            max(lo, rows.t_start.min() / 1000) + 0.001,
            min(hi, rows.t_end.max() / 1000),
            0.002,
        )
        positive = [
            (r.t_start / 1000, r.t_end / 1000)
            for r in rows.itertuples()
            if r.category == 1
        ]
        known = [
            (r.t_start / 1000, r.t_end / 1000)
            for r in rows.itertuples()
            if r.category in (0, 1)
        ]
        mask = spans_at(t, known)
        truth = spans_at(t, positive)[mask]
        found = spans_at(t, [(r["start_s"], r["end_s"]) for r in record["intervals"]])[
            mask
        ]
        times = np.array([r["time_s"] for r in record["crashes"]])
        support = spans_at(times, positive)
        covered = spans_at(times, known)
        overlap = [
            any(max(a, s["start_s"]) < min(b, s["end_s"]) for s in record["intervals"])
            for a, b in positive
        ]
        reviewed.append(
            {
                "shot": record["shot"],
                "cells": np.array([support.sum(), (covered & ~support).sum(), 0]),
                "histogram": score_histogram(truth, found.astype(float)),
                "expert_spans": len(positive),
                "overlapped_spans": sum(overlap),
                "supported_crashes": int(support.sum()),
                "assessed_crashes": int(covered.sum()),
            }
        )
    expert = aggregate(reviewed)
    expert["crash_recall"] = None
    expert["crash_f1"] = None
    expert["crash"] = {
        "span_supported_pick_fraction": sum(r["supported_crashes"] for r in reviewed)
        / max(1, sum(r["assessed_crashes"] for r in reviewed)),
        "supported_picks": sum(r["supported_crashes"] for r in reviewed),
        "assessed_picks": sum(r["assessed_crashes"] for r in reviewed),
    }
    expert["ci95"] = {
        k: v for k, v in expert["ci95"].items() if not k.startswith("crash_")
    }
    expert["span_overlap_recall"] = sum(r["overlapped_spans"] for r in reviewed) / max(
        1, sum(r["expert_spans"] for r in reviewed)
    )
    expert["by_shot"] = [
        {k: v for k, v in r.items() if k not in ("cells", "histogram")}
        for r in reviewed
    ]
    expert["limitation"] = (
        "Review provides spans, not crash times: crash precision/recall cannot be measured. Supported-pick fraction is not crash precision. Categories >=2 abstain."
    )
    agreement = []
    missing = []
    for record in records_at(args.work, cohort.shot):
        if "window_s" not in record:
            continue
        path = Paths.from_env().events_file(record["shot"])
        if not path.exists():
            missing.append(record["shot"])
            continue
        old = schema.read_events(path, source="ece_sawtooth")
        lo, hi = record["window_s"]
        reference = sorted(
            e.t0_s for e in old.itertuples() if lo <= e.t0_s <= hi and e.t0_s == e.t1_s
        )
        estimate = [r["time_s"] for r in record["crashes"]]
        bins = np.arange(lo + 0.001, hi, 0.002)
        # Old crash times imply trains under the SAME span construction.
        from labeler.sawtooth.physics import trains

        old_spans = [
            (reference[a], reference[b - 1]) for a, b in trains(sorted(reference))
        ]
        interval_spans = [(r["start_s"], r["end_s"]) for r in record["intervals"]]
        agreement.append(
            {
                "shot": record["shot"],
                "cells": event_cells(reference, estimate),
                "histogram": score_histogram(
                    spans_at(bins, old_spans),
                    spans_at(bins, interval_spans).astype(float),
                ),
            }
        )
    output = {
        "rule": asdict(Rule()),
        "expert": expert,
        "legacy_agreement": aggregate(agreement),
        "missing_legacy_shots": missing,
        "review_csv": str(REVIEW),
        "legacy_source": "read-only production ece_sawtooth event rows (ECE/SXR union)",
        "bin_ms": 2,
        "crash_tolerance_ms": 2,
        "rule_frozen_before_validation": True,
    }
    save_json(REPO / "outputs/labeler/sawtooth/validation.json", output)
    print(
        json.dumps(
            {
                "expert": expert["presence"],
                "legacy_agreement": output["legacy_agreement"]["crash"],
            }
        ),
        flush=True,
    )


def gallery(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    reviewed = set(pd.read_csv(REVIEW).shot)
    eligible = sorted(
        s
        for s in cohort[cohort.split == "train"].shot
        if s not in reviewed and (args.work / "signals" / f"{s}.npz").exists()
    )
    shots = (
        np.random.default_rng(SEED).choice(eligible, size=12, replace=False).tolist()
    )
    out = args.work / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    figures = []
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    for shot, record in zip(shots, records_at(args.work, shots), strict=True):
        data = np.load(args.work / "signals" / f"{shot}.npz")
        t, y = data["t"], data["y"].astype(float)
        fig, axes = plt.subplots(
            2, 1, figsize=(7, 3.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
        for rows, color, name in (
            (slice(20, 28), "#0072B2", "ECE core proxy"),
            (slice(8, 16), "#D55E00", "ECE outer proxy"),
        ):
            values = y[rows]
            valid = np.isfinite(values)
            trace = np.divide(
                np.where(valid, values, 0).sum(axis=0),
                valid.sum(axis=0),
                out=np.full(len(t), np.nan),
                where=valid.sum(axis=0) > 0,
            )
            axes[0].plot(
                t[::10] * 1000,
                gaussian_filter1d(np.nan_to_num(trace), 5)[::10],
                color=color,
                lw=0.7,
                label=name,
            )
        for r in record["crashes"]:
            axes[0].axvline(r["time_s"] * 1000, color="#009E73", lw=0.5, alpha=0.6)
        for r in record["intervals"]:
            axes[0].axvspan(
                r["start_s"] * 1000, r["end_s"] * 1000, color="#009E73", alpha=0.12
            )
        radii = [r["attrs"]["inversion_rho"] for r in record["crashes"]]
        if radii and all(r is not None for r in radii):
            axes[1].scatter(
                [r["time_s"] * 1000 for r in record["crashes"]],
                radii,
                s=6,
                label="inversion rho",
            )
            axes[1].scatter(
                [r["time_s"] * 1000 for r in record["crashes"]],
                [r["attrs"]["q1_rho"] for r in record["crashes"]],
                s=6,
                label="q = 1 rho",
            )
            axes[1].set_ylabel("Normalized radius")
        else:
            axes[1].scatter(
                [r["time_s"] * 1000 for r in record["crashes"]],
                [r["attrs"]["inversion_channel"] for r in record["crashes"]],
                s=6,
                color="#009E73",
                label="Inversion channel",
            )
            axes[1].set_ylabel("ECE channel")
            axes[1].text(
                0.02,
                0.90,
                "ECE radius and q = 1 surface unavailable",
                transform=axes[1].transAxes,
                fontsize=7,
            )
        axes[0].set_ylabel("ECE Te (keV)")
        axes[0].text(0.02, 0.93, f"Shot {shot}", transform=axes[0].transAxes)
        axes[0].legend(loc="upper right", frameon=False)
        axes[1].set_xlabel("Time (ms)")
        fig.tight_layout()
        for extension in ("pdf", "png"):
            path = out / f"shot_{shot}.{extension}"
            fig.savefig(path, dpi=150)
            figures.append(str(path))
        plt.close(fig)
    save_json(
        REPO / "outputs/labeler/sawtooth/gallery.json",
        {
            "seed": SEED,
            "shots": shots,
            "sampling": "12 uniform random usable nonreview train shots, no replacement",
            "figures": figures,
            "geometry_note": "Channels shown when calibrated radii are absent; no q=1 radius inferred.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["labels", "validate", "gallery"])
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--population", action="store_true")
    parser.add_argument("--shots", nargs="+", type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    if args.stage == "labels":
        labels(args)
    elif args.stage == "validate":
        validate(args)
    else:
        gallery(args)


if __name__ == "__main__":
    main()
