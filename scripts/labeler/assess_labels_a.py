"""Reproduce L-A's read-only census and CPU ECE check; artifacts stay in /tmp.

Run with the checkout's src on PYTHONPATH. This reads existing products only;
it never calls a resolver, inference runner, or data-root writer.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from labeler.config import Paths, git_sha, sha256_of
from labeler.events import coverage, heuristics, schema


def grouped(frame, keys):
    return (
        frame.groupby(keys, dropna=False)
        .agg(rows=("shot", "size"), shots=("shot", "nunique"))
        .reset_index().to_dict("records")
    )


def census(paths, db, shots):
    labels = pd.read_parquet(db / "labels_wide.parquet")
    labels = labels[labels.shot.isin(shots)].copy()
    events = pd.read_parquet(db / "events.parquet")
    events = events[events.shot.isin(shots)]
    sources = pd.read_parquet(db / "event_sources.parquet")
    sources = sources[sources.shot.isin(shots)]
    labels["has_valid"] = labels.n_valid > 0
    labels["above_half"] = labels.max_valid > 0.5
    summary = labels.groupby(["slug", "label"]).agg(
        rows=("shot", "size"), shots=("shot", "nunique"),
        valid_shots=("has_valid", "sum"), valid_samples=("n_valid", "sum"),
        above_half_shots=("above_half", "sum"),
    ).reset_index().to_dict("records")
    inventory, feature_names, ece_metadata, label_groups = [], Counter(), [], Counter()
    for file in sorted(paths.events.glob("*_events.parquet")):
        frame = pd.read_parquet(file)
        inventory.append({
            "file": file.name, "sha256": sha256_of(file),
            "pool_rows": int(frame.shot.isin(shots).sum()),
            "counts": grouped(frame, ["shot", "evidence_kind", "source", "phenomenon"]),
        })
    for shot in shots:
        with h5py.File(paths.features_file(shot), "r", locking=False) as f:
            feature_names.update(f.keys())
            if "ece" in f:
                ece_metadata.append({
                    "shot": shot, "datasets": sorted(f["ece"].keys()),
                    "attrs": {k: str(v) for k, v in f["ece"].attrs.items()},
                })
        with h5py.File(paths.labels_file(shot), "r", locking=False) as f:
            label_groups.update(f.keys())
    groups = pd.read_parquet(db / "corpus_coverage.parquet")
    presence = []
    for population, frame in [("corpus", groups), ("recommender_v1", groups[groups.shot.isin(shots)])]:
        for group in ["ece", "filterscopes", "mirnov", "mhr", "co2", "bes"]:
            rows = frame[frame.group == group]
            presence.append({
                "population": population, "group": group, "shots": len(rows),
                "present": int(rows.present.sum()),
                "channels": sorted(map(int, rows.n_channels.unique())),
            })
    return {
        "shots": shots, "git_sha": git_sha(), "labels_wide_rows": len(labels),
        "labels_wide_shots": int(labels.shot.nunique()), "labels": summary,
        "events": grouped(events, ["evidence_kind", "source", "phenomenon"]),
        "event_sources_rows": len(sources),
        "observed_shots": int(events[events.evidence_kind.isin(["detector", "heuristic"])].shot.nunique()),
        "forecast_shots": int(events[events.evidence_kind == "forecast"].shot.nunique()),
        "labelmaker_event_files": inventory,
        "labelmaker_sources_files": len(list(paths.events.glob("*_sources.parquet"))),
        "label_file_model_shots": dict(label_groups), "feature_group_shots": dict(feature_names),
        "ece_feature_metadata": ece_metadata, "corpus_groups": presence,
        "input_sha256": {name: sha256_of(db / name) for name in [
            "labels_wide.parquet", "events.parquet", "event_sources.parquet",
            "corpus_coverage.parquet",
        ]},
    }


def check_sawteeth(paths, shots, out):
    # Matplotlib writes its cache only below the output directory, too.
    import os

    os.environ["MPLCONFIGDIR"] = str(out / "mpl-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for shot in shots:
        start = time.monotonic()
        with h5py.File(paths.corpus_file(shot), "r", locking=False) as f:
            t = np.asarray(f["ece/xdata"][:], dtype=np.float64)
            y = np.asarray(f["ece/ydata"][:], dtype=np.float32)
        cov = coverage.finite_span(t, y)
        begin = time.monotonic()
        events = heuristics.sawtooth_events(y, t, shot=shot, t_cov=cov)
        detector_s = time.monotonic() - begin
        with h5py.File(paths.features_file(shot), "r", locking=False) as f:
            ip_t = np.asarray(f["ip/xdata"][:], dtype=np.float64)
            ip = np.asarray(f["ip/ydata"][:], dtype=np.float64).reshape(-1)
            assert f["ip"].attrs["units"] == "A"
        crash_ip = np.interp([e.t0_s for e in events], ip_t, ip,
                             left=np.nan, right=np.nan)
        # A diagnostic sanity check, not a new acceptance gate or human truth.
        outside_pulse = int((np.isfinite(crash_ip) & (abs(crash_ip) < 100_000)).sum())
        schema.write_events(out / f"{shot}_sawtooth.parquet", shot, events,
                            run_id="task-LA-cpu", merge=False)
        summary = heuristics.sawtooth_summary(events)
        edges = Counter(e.attrs["inversion_channel_stop"] for e in events)
        # The plot is a signal-level sanity check, not an independent annotation.
        env, et = heuristics.envelope(y, t)
        del y
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
        axes[0].plot(et, env[10], lw=0.5, label="ECE channel 10 envelope")
        for e in events:
            axes[0].axvline(e.t0_s, color="tab:red", alpha=0.25, lw=0.5)
        axes[0].set(xlabel="Time (s)", ylabel="ECE (V)",
                    title=f"{shot}: {len(events)} heuristic crashes; no human truth")
        axes[0].legend()
        # Three evenly spaced accepted crashes show the radial sign reversal.
        examples = np.unique(np.linspace(0, len(events) - 1, min(3, len(events)), dtype=int))
        for i in examples:
            event = events[i]
            j = int(np.argmin(abs(et - event.t0_s)))
            steps = heuristics._crash_step(
                env, j, env_ms=heuristics.ENV_MS,
                gap_ms=heuristics.STEP_GAP_MS, span_ms=heuristics.STEP_SPAN_MS,
            )
            axes[1].plot(np.arange(48), steps, label=f"{event.t0_s:.3f} s")
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].set(xlabel="ECE channel (zero based; not radius)",
                    ylabel="After − before (V)", title="Accepted crash step profiles")
        if len(examples):
            axes[1].legend()
        fig.savefig(out / f"{shot}_sawtooth.png", dpi=130)
        plt.close(fig)
        row = {
            "shot": shot, **summary, "coverage_s": cov,
            "inversion_stop_counts": dict(sorted(edges.items())),
            "inversion_radius": None, "detector_s": detector_s,
            "ip_check_threshold_a": 100_000,
            "crashes_below_ip_threshold": outside_pulse,
            "crashes_with_unknown_ip": int((~np.isfinite(crash_ip)).sum()),
            "elapsed_s": time.monotonic() - start,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {"git_sha": git_sha(), "device": "cpu", "shots": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["census", "sawtooth"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shots", nargs="+", type=int)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(Path("/tmp")):
        parser.error("all assessment outputs must be under /tmp")
    out.mkdir(parents=True, exist_ok=True)
    paths = Paths.from_env()
    pool_file = paths.root / "recommender_v1.txt"
    pool = sorted(set(map(int, pool_file.read_text().split())))
    assert len(pool) == 500
    if args.mode == "census":
        result = census(paths, paths.root.parent / "ideate/db", pool)
    else:
        shots = args.shots or [pool[i] for i in np.linspace(0, 499, 10, dtype=int)]
        outside = sorted(set(shots) - set(pool))
        if outside:
            parser.error(f"shots outside recommender_v1: {outside}")
        result = check_sawteeth(paths, shots, out)
    result["pool_sha256"] = sha256_of(pool_file)
    (out / f"{args.mode}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(out / f"{args.mode}.json")


if __name__ == "__main__":
    main()
