"""Generate additive sawtooth labels from read-only stores and inspect their scope.

Run ``labels`` on the cohort, then ``labels --population`` (resumes shot files),
and ``validate``. No network resolver is called. Large signals/labels/figures are
written only to --work; cohort CSV shards and evaluation JSON go to the worktree.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
from labeler.sawtooth.geometry import (
    load_radius_geometry,
    nominal_frequencies,
    radius_evidence,
    select_core,
)
from labeler.sawtooth.physics import Rule, detect, noise_calibration
from labeler.sawtooth.preprocessing import mask_spans, sample_native, state_spans

REPO = Path(__file__).resolve().parents[2]
WORK = Paths.from_env().root / "round4/saw/fix2"
OUTPUT = REPO / "outputs/labeler/sawtooth/fix2"
READER_POLICY = (
    "native_FIR_positive_absence_dynamic_core_perchannel_relaxation_phase_guard"
)
ECE_GEOMETRY_ARCHIVE = Path(os.environ.get(
    "LABELER_ECE_GEOMETRY_ROOT", str(REPO.parent / "omnimode/data")
))
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
    """Prefer explicitly MSE-constrained equilibrium, not MSE data presence."""
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


def enrich_radius_evidence(detected, radius, flux):
    """Retain nominal R metadata without turning it into calibrated rho."""
    for event in (
        detected.crashes + detected.intervals + detected.uncertain_intervals
    ):
        channel = event.attrs.get("inversion_channel")
        if channel is None:
            continue
        event.attrs.update(radius_evidence(
            radius, (event.t0_s + event.t1_s) / 2, channel, flux
        ))


def process_shot(job):
    shot, work, keep_signal, window, *options = job
    work, paths = Path(work), Paths.from_env()
    rule = frozen_rule(work)
    record = work / "shots" / f"{shot}.json"
    signal = work / "signals" / f"{shot}.npz"
    if record.exists() and (not keep_signal or signal.exists()):
        previous = json.loads(record.read_text())
        if previous.get("rule") != asdict(rule):
            raise ValueError(f"cached rule differs from freeze for shot {shot}")
        if previous.get("reader_policy") != READER_POLICY:
            raise ValueError(f"cached reader policy differs for shot {shot}")
        if "error" not in previous:
            return previous
    started = time.monotonic()
    result = {
        "shot": int(shot),
        "crashes": [],
        "intervals": [],
        "uncertain_intervals": [],
        "rule": asdict(rule),
        "reader_policy": READER_POLICY,
        "validation_status": "unvalidated_research_labels_owner_away",
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
            sxr, dalpha, neutron, mirnov, nbi = None, None, None, None, None
            radius, result["radius_geometry"] = load_radius_geometry(
                shot, t, len(y), paths, archive_root=ECE_GEOMETRY_ARCHIVE
            )
            core = select_core(
                y,
                radius=radius,
                minimum_channels=rule.minimum_channels,
                maximum_to_core=rule.maximum_channel_to_core,
            )
            result["core_geometry"] = core.info
            y[~core.physical_channels] = np.nan
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
            for i, rows in enumerate((core.core_channels, core.outer_channels)):
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
            if "pinj" in file and file["pinj/ydata"].shape[-1] > 32:
                try:
                    tb, yb = sampled(file["pinj"])
                    finite = np.isfinite(yb) & (yb >= 0)
                    # Native corpus pinj is W per beam (resolve_corpus.py),
                    # unlike canonical pinj_total, whose units are kW.
                    power = np.where(finite, yb, 0).sum(axis=0)
                    power[~finite.all(axis=0)] = np.nan
                    nbi = tb, power
                    result["nbi_source"] = f"{file.filename}:/pinj/ydata (W)"
                except ValueError as error:
                    result["nbi_error"] = str(error)
            result["nbi_available"] = nbi is not None
            observable, result["density_guard"] = density_support(
                file, t, shot, paths, rule
            )
        ip = local_scalar(shot, "ip", paths)
        if nbi is None:
            try:
                power = equilibrium.signal(shot, "pinj_total", paths, fetch=False)
                nbi = power.x, power.y[0] * 1000
                result["nbi_source"] = {
                    "store": power.attrs.get("store"),
                    "locator": power.attrs.get("locator"),
                    "canonical_units": "kW converted to W",
                }
                result["nbi_available"] = True
            except (NoDataError, ValueError):
                pass
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
            core_channels=core.core_channels,
            ip=ip,
            observability=observable,
            neutron=neutron,
            mirnov=mirnov,
            nbi=nbi,
        )
        enrich_radius_evidence(detected, radius, geometry)
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
            absence_diagnostics=detected.absence_diagnostics,
            assessment_policy="positive_absence_evidence_unresolved_is_uncertain",
        )
        states, assessed = state_spans(
            t,
            detected.observable,
            [(r["start_s"], r["end_s"]) for r in result["intervals"]],
            [(r["start_s"], r["end_s"]) for r in result["uncertain_intervals"]],
            absent=detected.absent_mask,
        )
        result.update(
            states=states,
            observable_spans=mask_spans(t, detected.observable),
            assessed_spans=mask_spans(t, assessed),
            absent_evidence_spans=mask_spans(t, detected.absent_mask),
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
                absent_evidence=detected.absent_mask,
                core_channels=core.core_channels,
                outer_channels=core.outer_channels,
            )
    except (OSError, KeyError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    save_json(record, result)
    return result


def export_rows(record):
    """Export the state partition without conflicting broad train annotations."""
    states = record.get("states", [])
    starts = np.array([span["start_s"] for span in states])
    for point in record["crashes"]:
        attrs = dict(point["attrs"])
        if states:
            index = int(np.searchsorted(starts, point["time_s"], side="right") - 1)
            state = (
                states[index]["state"]
                if index >= 0 and point["time_s"] < states[index]["end_s"]
                else "unassessed"
            )
            if state in ("uncertain", "unassessed") and attrs["state"] != state:
                attrs["candidate_state"] = attrs["state"]
                attrs["state"] = state
                attrs["uncertainty_reasons"] = sorted(
                    set(
                        attrs.get("uncertainty_reasons", [])
                        + ["overlapping_assessment_uncertainty"]
                    )
                )
        yield (
            point["time_s"],
            point["time_s"],
            False,
            point["confidence"],
            attrs,
        )
    if states:
        for span in states:
            if span["state"] != "present":
                yield (
                    span["start_s"], span["end_s"], True, 1.0,
                    {"state": span["state"]},
                )
                continue
            # The assessment partition can split a train. Intersect it with
            # original trains so each exported positive span retains metadata.
            covered = False
            for train in record.get("intervals", []):
                start = max(span["start_s"], train["start_s"])
                end = min(span["end_s"], train["end_s"])
                if end <= start:
                    continue
                attrs = dict(train["attrs"], state="present")
                for key in ("inversion_rho", "inversion_R_m", "q1_R_m"):
                    attrs.setdefault(key, None)
                yield start, end, True, 1.0, attrs
                covered = True
            if not covered:
                raise ValueError("present assessment span has no original train")
    else:
        for span in record["intervals"]:
            yield span["start_s"], span["end_s"], True, 1.0, span["attrs"]


def export_csv(records, destination, *, compact=False, prefix="labels"):
    destination.mkdir(parents=True, exist_ok=True)
    columns = ["shot", "category", "t_start", "t_end", "crowd", "confidence", "attrs"]
    part, bytes_used, handle = 0, 0, None
    written = set()
    try:
        for record in records:
            for start, end, crowd, confidence, attrs in export_rows(record):
                if compact:
                    attrs = {
                        key: attrs[key]
                        for key in (
                            "inversion_rho",
                            "inversion_R_m",
                            "q1_R_m",
                            "q1_rho",
                            "q1_radius_difference_m",
                            "inversion_channel",
                            "train_id",
                            "train_ids",
                            "period_ms",
                            "state",
                            "central_relative_drop",
                            "q_conflict",
                            "candidate_state",
                            "uncertainty_reasons",
                        )
                        if key in attrs
                    }
                if handle is None or bytes_used > 1_400_000:
                    if handle is not None:
                        handle.close()
                    path = destination / f"{prefix}-{part:03d}.csv"
                    written.add(path)
                    handle = path.open("w", newline="")
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
    for path in destination.glob(f"{prefix}-[0-9][0-9][0-9].csv"):
        if path not in written:
            path.unlink()
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
    if args.records_only:
        records = records_at(args.work, shots)
        if len(records) != len(shots):
            raise ValueError("records-only consolidation requires every requested shot")
        expected_rule = asdict(frozen_rule(args.work))
        for record in records:
            if record["rule"] != expected_rule:
                raise ValueError(f"cached rule mismatch for shot {record['shot']}")
            if record.get("reader_policy") != READER_POLICY:
                raise ValueError(f"cached reader mismatch for shot {record['shot']}")
            if "error" not in record and not record.get("assessment_policy"):
                raise ValueError(f"pending mask repair for shot {record['shot']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, record in enumerate(pool.map(process_shot, jobs, chunksize=1)):
                records.append(record)
                if i % 50 == 0:
                    print(
                        f"{i + 1}/{len(jobs)} shot {record['shot']} "
                        f"crashes {len(record['crashes'])} "
                        f"error {record.get('error', '')}",
                        flush=True,
                    )
    summary = {
        "scope": "population" if args.population else "cohort_plus_review",
        "cached_records_only": args.records_only,
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
        "exported_point_state_counts": dict(
            Counter(
                attrs["state"]
                for record in records
                for _, _, crowd, _, attrs in export_rows(record)
                if not crowd
            )
        ),
        "intervals": sum(len(r["intervals"]) for r in records),
        "positive_shots": sum(bool(r["intervals"]) for r in records),
        "qmin_shots": sum(r.get("qmin_available", False) for r in records),
        "geometry_counts": dict(
            Counter(r.get("geometry", "not_processed") for r in records)
        ),
        "radius_geometry_counts": dict(
            Counter(
                r.get("radius_geometry", {}).get("status", "not_processed")
                for r in records
            )
        ),
        "core_geometry_counts": dict(
            Counter(
                r.get("core_geometry", {}).get("status", "not_processed")
                for r in records
            )
        ),
        "physical_channel_exclusions": dict(
            Counter(
                str(channel) for r in records
                for channel in r.get("core_geometry", {}).get(
                    "implausible_channels", []
                )
            )
        ),
        "validation_status": "unvalidated_research_labels_owner_away",
        "sxr_shots": sum(bool(r.get("sxr_channels")) for r in records),
        "rejected": dict(
            sum((Counter(r.get("rejected", {})) for r in records), Counter())
        ),
        "split_counts": cohort.split.value_counts().to_dict(),
        "rule": asdict(frozen_rule(args.work)),
        "adaptations": {
            "core_proxy_channels": "per-shot hottest physically screened neighborhood",
            "terminal_channels_40_47": "excluded from uncalibrated core selection",
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
    summary["nbi_shots"] = sum(r.get("nbi_available", False) for r in records)
    summary["q_sources"] = dict(
        Counter(r.get("q_source", "not_processed") for r in records)
    )
    radius_comparisons = [
        point["attrs"]["q1_radius_difference_m"]
        for record in records for point in record["crashes"]
        if point["attrs"].get("q1_radius_difference_m") is not None
    ]
    summary["q1_major_radius_comparison"] = {
        "validation_status": "unvalidated_nominal_second_harmonic_EFIT_comparison",
        "comparable_crashes": len(radius_comparisons),
        "shots": [
            record["shot"] for record in records
            if any(point["attrs"].get("q1_radius_difference_m") is not None
                   for point in record["crashes"])
        ],
        "difference_definition": "nominal inversion R minus same-branch EFIT q=1 R",
        "median_difference_m": (
            float(np.median(radius_comparisons)) if radius_comparisons else None
        ),
        "maximum_absolute_difference_m": (
            float(np.max(np.abs(radius_comparisons))) if radius_comparisons else None
        ),
    }
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


def geometry_file_metadata(path):
    """One independent metadata-only read for the bounded audit workers."""
    candidates = {
        "ece_psi", "ece_q", "ece_frequency", "ece_frequencies", "ece_freq",
        "ece_frequency_hz", "ece_frequency_ghz", "qpsi", "psirz",
        "rgrid", "zgrid", "bcentr", "rcentr", "rmaxis", "r0", "bt",
    }
    result = {"file": str(path), "opened": False}
    try:
        with h5py.File(path, "r", locking=False) as file:
            result.update(opened=True, available=sorted(candidates & set(file)))
            if "ece" not in file:
                return result
            group = file["ece"]
            n = group["ydata"].shape[0]
            frequency, source = nominal_frequencies(file, n)
            result.update(
                ece=True, keys=list(group), attr_keys=list(group.attrs),
                geometry=sorted((set(file) & candidates) - {"bt", "r0"}),
                frequency=(
                    {"source": source, "frequency_hz": frequency.tolist()}
                    if frequency is not None else None
                ),
                example={
                    "file": str(path), "channel_count": int(n),
                    "ece_datasets": list(group),
                    "ece_attrs": {k: str(v) for k, v in group.attrs.items()},
                    "dataset_attrs": {
                        k: {a: str(v) for a, v in group[k].attrs.items()}
                        for k in group if isinstance(group[k], h5py.Dataset)
                    },
                },
            )
    except (OSError, KeyError, ValueError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def geometry_audit(args):
    """Inventory local HDF5 metadata without reading waveforms or fetching."""
    paths = Paths.from_env()
    stores = {
        "corpus": (paths.corpus, "*_processed.h5"),
        "features": (paths.features, "*_features.h5"),
        "raw_cache": (paths.raw_cache, "*_processed.h5"),
    }
    report = {
        "method": "read-only HDF5 group/dataset names and attributes; no waveforms",
        "workers": args.workers,
        "validation_status": "calibration_inventory_only_owner_away",
        "stores": {}, "frequency_sources": [], "geometry_sources": [],
        "example_ece_metadata": [], "errors": {},
    }
    if args.records_only:
        report = json.loads((args.work / "geometry_metadata_audit.json").read_text())
        report["metadata_records_reused"] = True
    for label, (root, pattern) in stores.items():
        files = sorted(
            p for p in root.glob(pattern)
            if p.is_file() and p.name.split("_")[0].isdigit()
        )
        if args.shots:
            wanted = set(args.shots)
            files = [p for p in files if int(p.name.split("_")[0]) in wanted]
        if args.records_only:
            previous = report["stores"][label]
            if len(files) != previous["files_requested"]:
                raise ValueError(f"metadata inventory changed for {label}")
            previous["shots"] = [int(p.name.split("_")[0]) for p in files]
            continue
        counts = Counter()
        keys, attrs, available = Counter(), Counter(), Counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for item in pool.map(geometry_file_metadata, files, chunksize=8):
                counts["files_requested"] += 1
                counts["files_opened"] += int(item["opened"])
                available.update(item.get("available", []))
                if "error" in item:
                    report["errors"][item["file"]] = item["error"]
                if item.get("ece"):
                    counts["ece_groups"] += 1
                    keys.update(item["keys"])
                    attrs.update(item["attr_keys"])
                    if item["frequency"] is not None:
                        report["frequency_sources"].append(item["frequency"])
                    if item["geometry"]:
                        report["geometry_sources"].append({
                            "file": item["file"], "groups": item["geometry"],
                        })
                    if len(report["example_ece_metadata"]) < 6:
                        report["example_ece_metadata"].append(item["example"])
                if counts["files_requested"] % 2000 == 0:
                    print(
                        f"{label}: {counts['files_requested']}/{len(files)}", flush=True
                    )
        report["stores"][label] = {
            "root": str(root), "glob": pattern, **dict(counts),
            "shots": [int(p.name.split("_")[0]) for p in files],
            "ece_dataset_keys": dict(keys), "ece_attribute_keys": dict(attrs),
            "potential_geometry_groups": dict(available),
        }
    report["frequency_metadata_count"] = len(report["frequency_sources"])
    archive_sources, frequency_vectors = [], []
    for path in sorted((ECE_GEOMETRY_ARCHIVE / "raw").glob("*.h5")):
        try:
            with h5py.File(path, "r", locking=False) as file:
                if "ecegeom/FREQ" not in file:
                    continue
                frequency = np.asarray(file["ecegeom/FREQ"], dtype=float).ravel()
                expected = {f"ECEVS{i + 1:02d}" for i in range(len(frequency))}
                identities = expected <= set(file.get("ece", {}))
                frequency_vectors.append(frequency)
                archive_sources.append({
                    "shot": int(path.stem),
                    "source": f"{path}:/ecegeom/FREQ",
                    "file_mtime_unix_s": path.stat().st_mtime,
                    "frequency_ghz": frequency.tolist(),
                    "stored_units": str(file["ecegeom/FREQ"].attrs.get("units")),
                    "source_tree": str(file["ecegeom"].attrs.get("source")),
                    "ecevs_identifiers_match_channel_order": identities,
                    "efit_groups": list(file.get("eq", {})),
                })
        except (OSError, KeyError, ValueError) as error:
            report["errors"][str(path)] = f"{type(error).__name__}: {error}"
    report["external_archive"] = {
        "root": str(ECE_GEOMETRY_ARCHIVE),
        "frequency_metadata_count": len(archive_sources),
        "sources": archive_sources,
        "documentation": [
            str(REPO.parent / "fdp/scripts/omnimode.py"),
            str(REPO.parent / "fdp/scripts/geometry.py"),
            str(REPO.parent / "omnimode/src/omnimode/modefit/ece_fwd.py"),
        ],
        "unit_contract": "ecegeom/FREQ GHz; original stored units blank",
        "scope": "same-shot configuration only; no transfer across shots",
    }
    report["external_archive"]["ece_node_provenance"] = {
        "tree": "ELECTRONS", "frequency_node": r"\ECE::TOP.SETUP.FREQ",
        "sightline_height_node": r"\ECE::TOP.SETUP.ECEZH",
        "reference_source_script": str(REPO.parent / "fdp/scripts/omnimode.py"),
        "corpus_order_source": str(
            REPO / "scripts/data_fetching_omega/config_atlas.yaml"
        ),
        "channel_join": "zero-based corpus row i joins TECEF{i+1:02d}/ECEVS{i+1:02d}",
    }
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    archive_shots = {source["shot"] for source in archive_sources}
    report["external_archive"].update(
        cohort_source=str(REPO / "data/events/catalog/cohort.csv"),
        cohort_overlap=sorted(archive_shots & set(cohort.shot)),
        corpus_overlap=sorted(
            archive_shots & set(report["stores"]["corpus"].get("shots", [
                int(p.name.split("_")[0])
                for p in paths.corpus.glob("*_processed.h5")
                if p.name.split("_")[0].isdigit()
            ]))
        ),
    )
    if frequency_vectors and len({len(v) for v in frequency_vectors}) == 1:
        values = np.stack(frequency_vectors)
        spread = np.ptp(values, axis=0)
        report["external_archive"].update(
            channel_frequency_range_ghz=spread.tolist(),
            invariant_channel_indices=np.flatnonzero(spread == 0).tolist(),
            universal_grid_available=bool((spread == 0).all()),
        )
    report["conclusion"] = (
        "no RF frequencies in corpus/features/raw_cache ECE metadata; "
        "same-shot external setup records are used when available; "
        "frequency grids vary, so other shots retain unknown radii"
        if not report["frequency_sources"] and archive_sources
        else "no channel RF-frequency calibration found in inventoried metadata; "
        "major radii and q=1 major-radius comparisons unavailable"
        if not report["frequency_sources"]
        else (
            "RF-frequency sources present; field reference and EFIT axis "
            "still required"
        )
    )
    save_json(OUTPUT / "geometry_metadata_audit.json", report)
    save_json(args.work / "geometry_metadata_audit.json", report)
    print(json.dumps({
        label: {key: value for key, value in store.items() if key != "shots"}
        for label, store in report["stores"].items()
    }), flush=True)


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
    parser.add_argument(
        "stage", choices=["labels", "validate", "gallery", "geometry-audit"]
    )
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--population", action="store_true")
    parser.add_argument("--skip-cohort", action="store_true")
    parser.add_argument("--records-only", action="store_true")
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
    elif args.stage == "geometry-audit":
        geometry_audit(args)
    else:
        gallery(args)


if __name__ == "__main__":
    main()
