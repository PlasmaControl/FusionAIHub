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
from labeler.events import equilibrium
from labeler.events.panels import ece_geometry
from labeler.events.verify import NoDataError
from labeler.sawtooth.physics import Rule, detect, noise_calibration
from labeler.sawtooth.preprocessing import mask_spans, sample_native, state_spans

REPO = Path(__file__).resolve().parents[2]
WORK = Paths.from_env().root / "round4/saw/fix"
OUTPUT = REPO / "outputs/labeler/sawtooth/fix"
SEED = 20261003
FS = 10000
REVIEW = Paths.from_env().label_tables / "sawtooth_oscillation/review/labels.csv"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def legacy_sampled(group, *, rows=None, fs=FS):
    """Original aliased reader, retained ONLY for train-shot impact measurement.

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


sampled = sample_native


def frozen_rule(work):
    path = Path(work) / "freeze.json"
    if not path.exists():
        raise ValueError(f"Missing TRAIN-only freeze: {path}; run sawtooth_freeze.py")
    return Rule(**json.loads(path.read_text())["rule"])


def mean_finite(values):
    finite = np.isfinite(values)
    return np.divide(
        np.where(finite, values, 0).sum(axis=0),
        finite.sum(axis=0),
        out=np.full(values.shape[1], np.nan),
        where=finite.sum(axis=0) > 0,
    )


def density_support(file, t, shot, paths, rule):
    """Conservative second-harmonic X-mode cutoff proxy, never a radius claim."""
    support = np.ones(len(t), dtype=bool)
    info = {"status": "density_unavailable", "cutoff_proxy": True}
    if "ts_core_density" not in file:
        return support, info
    g = file["ts_core_density"]
    tx, y = np.asarray(g["xdata"]), np.asarray(g["ydata"])
    if len(tx) < 2:
        return support, info
    y[y <= 0] = np.nan
    good = np.isfinite(y).any(axis=0)
    ne = np.full(len(tx), np.nan)
    ne[good] = np.nanquantile(y[:, good], 0.9, axis=0)
    density = ece_geometry.align_q(t, tx, ne[None])[0]
    bt = local_scalar(shot, "bt", paths)
    if bt is None:
        # Read the local scalar archive without any remote resolver or writes.
        from labeler.features import resolve_archive

        arrays, _ = resolve_archive.resolve(shot, ["bt"])
        if "bt" in arrays:
            a = equilibrium.canonical(arrays["bt"], "bt")
            bt = a.x, a.y[0]
    if bt is not None:
        b = ece_geometry.align_q(t, bt[0], np.atleast_2d(bt[1]))[0]
        cutoff = 0.9 * 2 * (27.992e9 * np.abs(b) / 8.98) ** 2
        info["status"] = "Thomson_90percentile_and_local_bt"
    else:
        # A train-frozen conservative high-density guard when Bt is unavailable.
        cutoff = np.full(len(t), 8e19)
        info["status"] = "Thomson_90percentile_fixed_density_guard_bt_missing"
    high = np.isfinite(density) & (density >= cutoff)
    support[high] = False
    info.update(
        high_density_samples=int(high.sum()),
        density_samples=int(np.isfinite(density).sum()),
        fixed_density_guard_m3=8e19,
        margin=0.9,
    )
    return support, info


def interpolate(t, tx, y):
    return np.stack([np.interp(t, tx, row, left=np.nan, right=np.nan) for row in y])


def local_scalar(shot, name, paths):
    try:
        array = equilibrium.signal(shot, name, paths, fetch=False)
        return array.x, array.y[0]
    except (NoDataError, ValueError):
        return None


def local_q(shot, paths):
    """Prefer explicitly MSE-constrained local equilibrium, never infer from MSE data."""
    from labeler.events import raw
    from labeler.features.store import read_feature

    for path in (
        paths.features_file(shot),
        paths.corpus_file(shot),
        raw.cache_path(shot, paths=paths),
    ):
        for name in ("qmin_mse", "qmin_efit02", "qpsi_mse", "qpsi_efit02", "qpsi"):
            try:
                array = read_feature(path, name)
            except (OSError, KeyError):
                continue
            attrs = array.attrs
            constrained = (
                "mse" in name
                or str(attrs.get("mse_constrained", "")).lower() in ("true", "1", "yes")
                or "mse" in str(attrs.get("constraints", "")).lower()
            )
            if not constrained or len(array.x) < 2:
                continue
            y = np.asarray(array.y, dtype=float)
            valid = np.isfinite(y) & (y > 0)
            q = np.min(np.where(valid, y, np.inf), axis=0)
            q[~np.isfinite(q)] = np.nan
            if np.isfinite(q).any():
                return (array.x, q), f"MSE-constrained:{name}"
    q = local_scalar(shot, "qmin", paths)
    return q, "EFIT01" if q is not None else "unavailable"


def process_shot(job):
    shot, work, keep_signal, window, *options = job
    work, paths = Path(work), Paths.from_env()
    rule = frozen_rule(work)
    record = work / "shots" / f"{shot}.json"
    signal = work / "signals" / f"{shot}.npz"
    if record.exists() and (not keep_signal or signal.exists()):
        previous = json.loads(record.read_text())
        if "error" not in previous:
            return previous
    started = time.monotonic()
    result = {
        "shot": int(shot),
        "crashes": [],
        "intervals": [],
        "uncertain_intervals": [],
        "rule": asdict(rule),
    }
    try:
        with h5py.File(paths.corpus_file(shot), "r", locking=False) as file:
            if "ece" not in file:
                raise ValueError("no ECE group")
            reader = legacy_sampled if options and options[0] == "aliased" else sampled
            t, y = reader(file["ece"])
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
            sxr, dalpha, neutron, mirnov = None, None, None, None
            result["sxr_channels"] = []
            if keep_signal and "sxr" in file and file["sxr/ydata"].shape[-1] > 32:
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
            baseline = np.full((4, len(t)), np.nan, dtype=np.float32)
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
            if "mirnov" in file and file["mirnov/ydata"].shape[-1] > 32:
                try:
                    tm, ym = sampled(file["mirnov"], rows=slice(0, 2))
                    baseline[2] = mean_finite(interpolate(t, tm, ym))
                    result["mirnov_available"] = bool(
                        np.isfinite(baseline[2]).mean() > 0.5
                    )
                    tm, ym = sampled(file["mirnov"], rows=slice(0, 2), rms=True)
                    mirnov = tm, mean_finite(ym)
                except ValueError as error:
                    result["mirnov_error"] = str(error)
            if "neutron_rate" in file and file["neutron_rate/ydata"].shape[-1] > 32:
                try:
                    tn, yn = sampled(file["neutron_rate"], rows=slice(0, 1))
                    neutron = tn, mean_finite(yn)
                except ValueError as error:
                    result["neutron_error"] = str(error)
            result["neutron_available"] = neutron is not None
            observable, result["density_guard"] = density_support(
                file, t, shot, paths, rule
            )
        ip = local_scalar(shot, "ip", paths)
        qmin, q_source = local_q(shot, paths)
        if ip is not None:
            baseline[3] = np.interp(t, ip[0], ip[1], left=np.nan, right=np.nan) / 1e6
        result["ip_available"] = ip is not None
        result["qmin_available"] = qmin is not None
        result["q_source"] = q_source
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
            rule=rule,
            qmin=qmin,
            q_source=q_source,
            geometry=geometry,
            dalpha=dalpha,
            sxr=sxr,
            core_channels=range(20, 36),
            ip=ip,
            observability=observable,
            neutron=neutron,
            mirnov=mirnov,
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
            uncertain_intervals=[
                {"start_s": e.t0_s, "end_s": e.t1_s, "attrs": e.attrs}
                for e in detected.uncertain_intervals
            ],
        )
        states, assessed = state_spans(
            t,
            detected.observable,
            [(r["start_s"], r["end_s"]) for r in result["intervals"]],
            [(r["start_s"], r["end_s"]) for r in result["uncertain_intervals"]],
        )
        result.update(
            states=states,
            observable_spans=mask_spans(t, detected.observable),
            assessed_spans=mask_spans(t, assessed),
        )
        result["state_seconds"] = {
            state: sum(r["end_s"] - r["start_s"] for r in states if r["state"] == state)
            for state in ("present", "absent", "uncertain", "unassessed")
        }
        if keep_signal:
            signal.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                signal,
                t=t,
                y=y.astype(np.float16),
                baseline=baseline,
                observable=detected.observable,
                assessed=assessed,
            )
    except (OSError, KeyError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    save_json(record, result)
    return result


def export_csv(records, destination, *, compact=False, prefix="labels"):
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
            rows += [
                (r["start_s"], r["end_s"], True, 1.0, {"state": r["state"]})
                for r in record.get("states", [])
            ]
            for start, end, crowd, confidence, attrs in rows:
                if compact:
                    attrs = {
                        key: attrs[key]
                        for key in (
                            "inversion_rho",
                            "inversion_channel",
                            "period_ms",
                            "state",
                            "central_relative_drop",
                            "q_conflict",
                        )
                        if key in attrs
                    }
                if handle is None or bytes_used > 1_400_000:
                    if handle is not None:
                        handle.close()
                    handle = (destination / f"{prefix}-{part:03d}.csv").open(
                        "w", newline=""
                    )
                    writer = csv.writer(handle)
                    writer.writerow(columns)
                    part, bytes_used = part + 1, 0
                text = json.dumps(attrs, allow_nan=False, separators=(",", ":"))
                writer.writerow(
                    [
                        record["shot"],
                        {
                            "present": 1,
                            "absent": 0,
                            "uncertain": 2,
                            "unassessed": 3,
                        }.get(attrs.get("state", "present"), 1),
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
    # Remove only obsolete shards made by this writer, after successful export.
    for path in destination.glob(f"{prefix}-*.csv"):
        if int(path.stem.split("-")[-1]) >= part:
            path.unlink()


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
    if args.shards > 1:
        shots = [s for i, s in enumerate(shots) if i % args.shards == args.shard]
    windows = {
        int(r.shot): (r.window_start_ms / 1000, r.window_end_ms / 1000)
        for r in cohort.itertuples()
    }
    if args.population and args.skip_cohort:
        shots = [s for s in shots if s not in windows and s not in set(reviewed.shot)]
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
        "crash_state_counts": dict(
            Counter(
                point["attrs"]["state"]
                for record in records
                for point in record["crashes"]
            )
        ),
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
        "rule": asdict(frozen_rule(args.work)),
        "adaptations": {
            "core_proxy_channels": list(range(20, 36)),
            "minimum_ip_ma_when_measured": 0.3,
            "ece_valid_range_kev": [0, 100],
            "model_and_detector_sample_rate_hz": FS,
            "native_rate_antialiasing": "FIR polyphase, chunks with full halos",
            "radius_units": "sqrt(normalized poloidal flux), only if calibrated",
            "sxr_use": "coverage recorded only; no corroboration claim",
        },
        "seconds": round(time.monotonic() - begun, 1),
        "source_records": str(args.work / "shots"),
        "signals": str(args.work / "signals"),
    }
    summary["state_seconds"] = {
        state: sum(r.get("state_seconds", {}).get(state, 0) for r in records)
        for state in ("present", "absent", "uncertain", "unassessed")
    }
    summary["uncertain_intervals"] = sum(
        len(r.get("uncertain_intervals", [])) for r in records
    )
    summary["neutron_shots"] = sum(r.get("neutron_available", False) for r in records)
    summary["mirnov_shots"] = sum(r.get("mirnov_available", False) for r in records)
    summary["q_sources"] = dict(
        Counter(r.get("q_source", "not_processed") for r in records)
    )
    scope = "population" if args.population else "cohort"
    if args.shards > 1:
        scope += f"_shard_{args.shard}"
    save_json(args.work / f"{scope}_labels.json", summary)
    if args.shards == 1:
        save_json(OUTPUT / f"{scope}_labels.json", summary)
    if not args.population:
        export_csv(
            records,
            REPO / "data/events/sawtooth_oscillation/extend_saw_physics",
            compact=True,
            prefix="cohort",
        )
        save_json(
            OUTPUT / "noise_calibration.json",
            noise_calibration(frozen_rule(args.work)),
        )
        export_csv(records, args.work / "labels", compact=True, prefix="cohort")
    else:
        if args.shards == 1:
            export_csv(records, args.work / "labels", compact=True, prefix="population")
            export_csv(
                records,
                REPO / "data/events/sawtooth_oscillation/extend_saw_physics",
                compact=True,
                prefix="population",
            )
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
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/labeler/sawtooth_fix_validation.py"),
            "validate",
            "--work",
            str(args.work),
        ],
        check=True,
    )


def gallery(args):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/labeler/sawtooth_gallery.py"),
            "--work",
            str(args.work),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["labels", "validate", "gallery"])
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--population", action="store_true")
    parser.add_argument("--skip-cohort", action="store_true")
    parser.add_argument("--shots", nargs="+", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    if not 0 <= args.shard < args.shards or args.shards > 1 and not args.population:
        parser.error("shard must lie in 0..shards-1; sharding is population-only")
    if args.stage == "labels":
        labels(args)
    elif args.stage == "validate":
        validate(args)
    else:
        gallery(args)


if __name__ == "__main__":
    main()
