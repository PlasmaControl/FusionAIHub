"""Native-sample diagnostic summaries; raw stores are opened read-only.

Each row summarizes only its 50 ms window. No interpolation, temporal decimation,
source identity, shot identifier or absolute clock enters the feature schema.
Raw Thomson channels have no radial coordinates: these are channel summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..config import Paths
from .labels import MODES, digest

# deterministic physical channel subsampling, fixed before test evaluation
DIAGNOSTICS = {
    "dalpha": ("filterscopes", "first8", True),
    "bes": ("bes", "spread8", True),
    "mirnov": ("mirnov", "spread4", True),
    "thomson_density": ("ts_core_density", "spread8", False),
    "thomson_temp": ("ts_core_temp", "spread8", False),
    "co2": ("co2", "all", False),
    "nbi": ("pinj", "all", False),
    "betan": ("betan", "all", False),
    "betap": ("betap", "all", False),
    "wmhd": ("wmhd", "all", False),
    "qmin": ("qmin", "all", False),
}
STATS = ("mean", "median", "channel_spread", "rms_fluctuation", "iqr",
         "relative_fluctuation", "diff_rms", "slope", "crest",
         "power_1_10khz", "power_10_50khz", "power_50_250khz")


def feature_columns():
    return [f"{name}__{stat}" for name in DIAGNOSTICS for stat in (*STATS, "present")]


def physical_eligibility(frame, columns):
    physical = [c for c in columns if not c.endswith("__present")]
    return np.isfinite(frame[physical].to_numpy(dtype=float)).any(axis=1)


def _missing():
    return {**dict.fromkeys(STATS, float("nan")), "present": 0.}


def summarize_window(t, y, start_s, end_s, *, fluctuation=False, cadence=None):
    """Summarize native samples, rejecting gaps and stale/singleton records.

    Sampling cadence must be <= half a bin. Each boundary can be up to one
    native cadence from an observation. Interior gaps >1.5 cadences reject the
    entire diagnostic. A channel is used only if every selected sample is finite.
    Periodograms use all native samples in non-overlapping Hann segments <=4096;
    incomplete final segments are omitted from spectra, never stride-decimated.
    Band powers are fractions of total positive-frequency fluctuation power.
    """
    result = _missing()
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all():
        return result
    if y.ndim == 1:
        y = y[None, :]
    if y.ndim != 2 or y.shape[1] != len(t) or (np.diff(t) <= 0).any():
        return result
    dt = float(np.median(np.diff(t))) if cadence is None else float(cadence)
    if not 0 < dt <= (end_s - start_s) / 2:
        return result
    mask = (t >= start_s - 1e-12) & (t < end_s - 1e-12)
    tw, yw = t[mask], y[:, mask]
    if len(tw) < 2 or tw[0] > start_s + dt * 1.01 \
            or tw[-1] + dt < end_s - dt * .01 \
            or (np.diff(tw) > dt * 1.5).any():
        return result
    yw = yw[np.isfinite(yw).all(axis=1)]
    if not len(yw):
        return result
    means = yw.mean(axis=1)
    centered = yw - means[:, None]
    rms = float(np.sqrt(np.mean(centered ** 2)))
    time_centered = tw - tw.mean()
    result.update(mean=float(np.mean(yw)), median=float(np.median(yw)),
                  channel_spread=float(np.std(means)), rms_fluctuation=rms,
                  iqr=float(np.quantile(yw, .75) - np.quantile(yw, .25)),
                  relative_fluctuation=rms / (float(np.mean(np.abs(yw))) + 1e-30),
                  diff_rms=float(np.sqrt(np.mean(np.diff(yw, axis=1) ** 2))),
                  slope=float(np.mean(centered @ time_centered /
                                      np.sum(time_centered ** 2))),
                  crest=float(np.quantile(np.abs(centered), .99) / (rms + 1e-30)),
                  present=1.)
    if fluctuation and len(tw) >= 64 \
            and np.max(np.abs(np.diff(tw) / dt - 1)) < .05:
        size = min(4096, len(tw))
        blocks = len(tw) // size
        segments = yw[:, :blocks * size].reshape(len(yw), blocks, size)
        segments = segments - segments.mean(axis=-1, keepdims=True)
        power = np.mean(np.abs(np.fft.rfft(segments * np.hanning(size))) ** 2,
                        axis=(0, 1))
        frequency = np.fft.rfftfreq(size, dt)
        total = power[frequency > 0].sum()
        if total > 0:
            for stat, lower, upper in (("power_1_10khz", 1000, 10000),
                                      ("power_10_50khz", 10000, 50000),
                                      ("power_50_250khz", 50000, 250000)):
                if frequency[-1] >= upper - 1 / (size * dt):
                    result[stat] = float(power[(frequency >= lower) &
                                              (frequency < upper)].sum() / total)
    return result


def derive_regimes(targets, merged):
    """Return L,H,QH,WP indices only for full bins with one exact subtype."""
    result = np.full(len(targets), -1, dtype=int)
    groups = dict(tuple(merged.groupby("shot")))
    for i, row in enumerate(targets.itertuples(index=False)):
        group = groups.get(row.shot)
        if group is None:
            continue
        spans = group[(group.t_start < row.t_end) & (group.t_end > row.t_start)]
        if len(spans) and set(spans.regimes) <= set(MODES) \
                and spans.regimes.nunique() == 1:
            overlap = (np.minimum(spans.t_end, row.t_end) -
                       np.maximum(spans.t_start, row.t_start)).sum()
            if abs(overlap - (row.t_end - row.t_start)) < 1e-6:
                result[i] = MODES.index(spans.regimes.iloc[0])
    return result


def _channels(policy, n):
    if policy == "first8":
        return np.arange(min(8, n))
    if policy.startswith("spread"):
        return np.unique(np.linspace(0, n - 1, min(int(policy[6:]), n)).astype(int))
    return np.arange(n)


def extract_shot(shot, targets, *, corpus, cache=None, equilibrium=None, clips=None):
    """Slice only labelled windows and selected channels, without live fetching."""
    frame = targets.reset_index(drop=True).copy()
    frame = pd.concat([frame, pd.DataFrame(
        {column: np.full(len(frame), 0. if column.endswith("__present") else np.nan)
         for column in feature_columns()})], axis=1)
    audit = {"shot": int(shot), "files": [], "diagnostics": {}, "errors": []}
    consumed = hashlib.sha256()
    with ExitStack() as stack:
        files = []
        candidates = [Path(corpus) / f"{shot}_processed.h5"]
        if cache:
            candidates.append(Path(cache) / f"{shot}_processed.h5")
        if equilibrium:
            candidates.append(Path(equilibrium) / f"{shot}_features.h5")
        for path in candidates:
            if path.exists():
                try:
                    files.append((path, stack.enter_context(h5py.File(path, "r"))))
                    st = path.stat()
                    audit["files"].append({"path": str(path), "bytes": st.st_size,
                                           "mtime_ns": st.st_mtime_ns})
                except OSError as exc:
                    audit["errors"].append(f"{path}: {exc}")
        clip_files = []
        if clips:
            for path in sorted(Path(clips).glob(f"bes_signals_{shot}at*.hdf5")):
                try:
                    clip_files.append((path, stack.enter_context(h5py.File(path, "r"))))
                    audit["files"].append({"path": str(path), "sha256": digest(path)})
                except OSError as exc:
                    audit["errors"].append(f"{path}: {exc}")
        for name, (group_name, policy, fluctuation) in DIAGNOSTICS.items():
            sources = [(path, f[group_name]["xdata"], f[group_name]["ydata"], 1.)
                       for path, f in files if group_name in f
                       and isinstance(f[group_name], h5py.Group)
                       and {"xdata", "ydata"}.issubset(f[group_name])]
            if name == "bes":
                sources += [(path, f["time"], f["signals"], .001)
                            for path, f in clip_files if "signals" in f and "time" in f]
            support = 0
            channel_audits = []
            for path, xt, yd, scale in sources:
                if len(xt.shape) != 1 or xt.shape[0] <= 1:
                    continue
                axis = np.asarray(xt, dtype=float) * scale
                if not np.isfinite(axis).all() or (np.diff(axis) <= 0).any():
                    audit["errors"].append(f"{path}/{group_name}: invalid time")
                    continue
                dt = float(np.median(np.diff(axis)))
                if yd.ndim not in (1, 2) or yd.shape[-1] != len(axis):
                    audit["errors"].append(f"{path}/{group_name}: invalid shape")
                    continue
                channels = _channels(policy, yd.shape[0] if yd.ndim == 2 else 1)
                channel_audits.append({"path": str(path), "channels_zero_based":
                                       channels.tolist(), "native_cadence_s": dt})
                for i, row in enumerate(frame.itertuples(index=False)):
                    if frame.loc[i, f"{name}__present"]:
                        continue
                    start, stop = row.t_start / 1000, row.t_end / 1000
                    lo, hi = np.searchsorted(axis, [start - 1e-12, stop - 1e-12])
                    if hi - lo < 2:
                        continue
                    y = np.asarray(yd[channels, lo:hi] if yd.ndim == 2 else yd[lo:hi],
                                   dtype=float)
                    # NBI is measured total power, not a mean over beam channels.
                    if name == "nbi" and y.ndim == 2:
                        y = np.where(np.isfinite(y).all(axis=0), y.sum(axis=0), np.nan)[None]
                    tw = axis[lo:hi]
                    consumed.update(name.encode())
                    consumed.update(tw.tobytes())
                    consumed.update(y.tobytes())
                    values = summarize_window(tw, y, start, stop,
                                              fluctuation=fluctuation, cadence=dt)
                    for stat, value in values.items():
                        frame.loc[i, f"{name}__{stat}"] = value
                    support += int(values["present"])
            audit["diagnostics"][name] = {"usable_bins": support, "sources": channel_audits}
    audit["consumed_input_sha256"] = consumed.hexdigest()
    return frame, audit


def _counts(frame):
    return {"shots": int(frame.shot.nunique()), "bins": len(frame),
            "seconds": float(((frame.t_end - frame.t_start) / 1000).sum()),
            "L_bins": int((frame.label == 0).sum()),
            "H_bins": int((frame.label == 1).sum()),
            "L_shots": int(frame.loc[frame.label == 0, "shot"].nunique()),
            "H_shots": int(frame.loc[frame.label == 1, "shot"].nunique())}


def extract(labels_dir, out, *, corpus, cache=None, equilibrium=None, clips=None):
    labels_dir, out = Path(labels_dir).resolve(), Path(out).resolve()
    for root in (corpus, cache, equilibrium, clips):
        if root and (out == Path(root).resolve() or Path(root).resolve() in out.parents):
            raise ValueError("Outputs must be outside raw diagnostic stores")
    targets = pd.read_csv(labels_dir / "targets.csv")
    targets = targets[targets.label >= 0].copy()
    targets["regime_label"] = derive_regimes(targets, pd.read_csv(labels_dir / "merged_intervals.csv"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "features").mkdir(exist_ok=True)
    frames, audits = [], []
    total = targets.shot.nunique()
    for k, (shot, rows) in enumerate(targets.groupby("shot", sort=True), 1):
        frame, audit = extract_shot(shot, rows, corpus=corpus, cache=cache,
                                    equilibrium=equilibrium, clips=clips)
        frame.to_csv(out / "features" / f"{shot}.csv", index=False)
        frames.append(frame)
        audits.append(audit)
        if k % 10 == 0 or audit["files"]:
            print(f"features {k}/{total}: shot {shot}, {int(physical_eligibility(frame, feature_columns()).sum())}/{len(frame)} bins", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(out / "features" / "all.csv", index=False)
    eligible = physical_eligibility(frame, feature_columns())
    summary = {
        "feature_schema": feature_columns(), "diagnostics": DIAGNOSTICS,
        "methodology": summarize_window.__doc__,
        "feature_semantics": "Within-bin native observations; NBI sum over beams; Thomson channel summaries without radial-coordinate or pedestal claims; no source, shot or absolute time features.",
        "hash_policy": "Consumed native times and selected physical values are SHA256 hashed per shot, including rejected windows; full raw files use immutable path/size/mtime metadata. Clip files also have full-file SHA256. Generated feature/label/split files are fully hashed.",
        "input_hashes": {name: digest(labels_dir / name) for name in
                         ("labels.json", "targets.csv", "split.csv", "merged_intervals.csv")},
        "feature_sha256": digest(out / "features" / "all.csv"),
        "code_sha256": digest(Path(__file__)),
        "annotated_support": {s: _counts(g) for s, g in frame.groupby("split")},
        "eligible_support": {s: _counts(g) for s, g in frame.loc[eligible].groupby("split")},
        "diagnostic_support": {name: {s: _counts(g) for s, g in
                                frame.loc[frame[f"{name}__present"] == 1].groupby("split")}
                               for name in DIAGNOSTICS},
        "excluded_all_physical_missing": {s: _counts(g) for s, g in frame.loc[~eligible].groupby("split")},
        "subtype_support": {s: {MODES[i]: _counts(g[g.regime_label == i]) for i in range(4)}
                            for s, g in frame.loc[eligible].groupby("split")},
        "shots": audits,
    }
    (out / "features.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    paths = Paths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=paths.root / "confinement/v1")
    parser.add_argument("--out", type=Path, default=paths.root / "confinement/v1")
    parser.add_argument("--corpus", type=Path, default=paths.corpus)
    parser.add_argument("--cache", type=Path, default=paths.raw_cache)
    parser.add_argument("--equilibrium", type=Path, default=paths.features)
    parser.add_argument("--clips", type=Path, default=paths.label_tables / "confinement/raw/confinement_data")
    args = parser.parse_args(argv)
    result = extract(args.labels, args.out, corpus=args.corpus, cache=args.cache,
                     equilibrium=args.equilibrium, clips=args.clips)
    print(json.dumps(result["eligible_support"], indent=2))


if __name__ == "__main__":
    main()
