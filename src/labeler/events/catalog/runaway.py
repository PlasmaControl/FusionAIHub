"""D2e: measure cold runaway-electron plateaus before drawing the population.

    pixi run -e labelmaker python -m labeler.events.catalog.runaway

Reads the pool, Ip log and corpus through the package's configured paths. Only
`data/events/catalog/{runaway.csv,runaway.meta.json}` is written by default.
Every shot passing rules 1-4 is measured over its assessed Ip window, including
both endpoints. A profile sample has at least ceil(n_channels / 2) finite,
positive ts_core_temp channels. Rule 5 excludes a median Thomson channel-p90
below 60 eV only when the window has at least one profile sample. The median
still uses all samples with any valid channel. `n_thomson` counts those samples;
`n_profile` counts profile samples. `no_thomson` means no usable Thomson profile
in the window: blank Te and retained. Beam power and neutron rates corroborate
the marks when the cohort consumes them; they never mark or unmark a shot.

The profile condition tests that core Thomson produced a profile somewhere in
the window (204081's never did), not that it resolved the plateau itself. On a
runaway plateau, cold plasma after the disruption leaves most channels without
a fit: plateau samples are sparse and carry the median. Corroboration guards
against a thermal shot whose Thomson works early, then falls to a few low
channels: a mark requires high neutron rates with low or absent beam power.

The corpus digest binds each file's three diagnostic groups (their presence,
shapes, and complete xdata/ydata arrays, encoded as little-endian float64), then
hashes sorted '<filename> <digest>\\n' records. Unused multi-GB waveforms are not
read or hashed. This records exactly the corpus content consumed by this scan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import time
from datetime import UTC, datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ...config import NamedBytes, Paths, atomic_path, git_dirty, git_sha
from . import population as pop
from . import window
from .check import CatalogError
from .points import validate_csv_fields

# The geometric middle of the gap between the warmest runaway shot and the
# coldest usable non-runaway shot over the assessed windows.
RUNAWAY_TE_EV = 60.0
RUNAWAY_MIN_NEUTRONS = 1e15  # per second
RUNAWAY_MAX_BEAM_KW = 1000.0
COLUMNS = (
    "shot",
    "te_p90_ev",
    "n_thomson",
    "n_profile",
    "pinj_kw",
    "neutron_rate_mean",
    "runaway",
)
GROUPS = ("ts_core_temp", "pinj", "neutron_rate")
STATISTIC = (
    "Median over assessed-window samples of the 90th percentile of finite, "
    "positive ts_core_temp channels, in eV; samples with no valid channel omitted. "
    "A profile sample has at least ceil(n_channels / 2) finite, positive channels; "
    "n_profile counts profile samples and n_thomson counts samples with any valid "
    "channel. Te uses all such samples when n_profile > 0. "
    "no_thomson means no usable Thomson profile in the window (n_profile == 0): "
    "blank Te and retained."
)


def definition() -> dict:
    return {
        "threshold_ev": RUNAWAY_TE_EV,
        "statistic": STATISTIC,
        "comparison": "te_p90_ev < threshold_ev",
        "no_thomson": "retain",
    }


def assess(path: Path, shot: int, start_ms: int, end_ms: int) -> tuple[dict, str]:
    """One shot's evidence and the digest of its diagnostic input arrays.

    Corpus xdata is normally seconds, converted to ms as in corpus_signal.
    Millisecond records are also accepted: axes reaching 100 are already ms,
    matching the judge's corpus scan convention. Absent-signal sentinels have
    at most one time sample. No empty/all-invalid sample emits a numpy warning.
    """
    digest = hashlib.sha256()
    values = {}
    try:
        with h5py.File(path, "r") as f:
            for name in GROUPS:
                digest.update((name + "\n").encode())
                if name not in f:
                    digest.update(b"absent\n")
                    values[name] = np.empty((0, 0))
                    continue
                group = f[name]
                x = np.asarray(group["xdata"], dtype="<f8")
                y = np.asarray(group["ydata"], dtype="<f8")
                for array in (x, y):
                    digest.update((str(array.shape) + "\n").encode())
                    digest.update(array.tobytes())
                if y.ndim != 2 or x.ndim != 1:
                    raise ValueError(f"{name}: expected time and channels by time")
                if y.shape[1] <= 1:
                    values[name] = np.empty((y.shape[0], 0))
                    continue
                if y.shape[1] != x.size or not np.isfinite(x).all():
                    raise ValueError(f"{name}: invalid time axis")
                if np.max(x) < 100:
                    x = x * 1000.0
                values[name] = y[:, (x >= start_ms) & (x <= end_ms)]
    except (OSError, KeyError, ValueError) as error:
        raise CatalogError(f"{path}: {error}") from error
    te = values["ts_core_temp"]
    positive = np.isfinite(te) & (te > 0)
    valid = positive.any(axis=0)
    n = int(valid.sum())
    n_profile = int((valid & (positive.sum(axis=0) >= (te.shape[0] + 1) // 2)).sum())
    te_p90 = (
        float(
            np.median(
                np.nanpercentile(
                    np.where(positive[:, valid], te[:, valid], np.nan), 90, axis=0
                )
            )
        )
        if n_profile
        else None
    )
    pinj = values["pinj"]
    finite = np.isfinite(pinj)
    totals = np.where(finite, pinj, 0).sum(axis=0)[finite.any(axis=0)]
    pinj_kw = float(np.mean(totals) / 1000.0) if totals.size else None
    neutron = []
    for channel in values["neutron_rate"]:
        good = channel[np.isfinite(channel)]
        neutron.append(str(float(np.mean(good))) if good.size else "")
    return {
        "shot": int(shot),
        "te_p90_ev": te_p90,
        "n_thomson": n,
        "n_profile": n_profile,
        "pinj_kw": pinj_kw,
        "neutron_rate_mean": ";".join(neutron),
        "runaway": te_p90 is not None and te_p90 < RUNAWAY_TE_EV,
    }, digest.hexdigest()


def read_runaway(path) -> pd.DataFrame:
    """Refuse malformed or internally inconsistent evidence before drawing."""
    try:
        validate_csv_fields(path)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if tuple(frame.columns) != COLUMNS:
            raise ValueError(f"expected columns {COLUMNS}")
        for field in ("shot", "n_thomson", "n_profile"):
            if not frame[field].str.fullmatch(r"[0-9]+").all():
                raise ValueError(f"{field} must be a nonnegative integer")
            frame[field] = frame[field].astype("int64")
        if frame.shot.duplicated().any():
            raise ValueError("duplicate shot")
        for field in ("te_p90_ev", "pinj_kw"):
            raw = frame[field]
            frame[field] = pd.to_numeric(raw.replace("", np.nan), errors="raise")
            if (~np.isfinite(frame[field]) & raw.ne("")).any():
                raise ValueError(f"{field} must be finite or blank")
        if not frame.runaway.str.lower().isin(("true", "false")).all():
            raise ValueError("runaway must be true/false")
        frame["runaway"] = frame.runaway.str.lower().eq("true")
        if (frame.n_profile > frame.n_thomson).any():
            raise ValueError("n_profile must not exceed n_thomson")
        if (frame.te_p90_ev.isna() != frame.n_profile.eq(0)).any():
            raise ValueError("te_p90_ev must be blank exactly when n_profile is zero")
        if (frame.te_p90_ev.dropna() <= 0).any():
            raise ValueError("te_p90_ev must be positive")
        if (frame.runaway != frame.te_p90_ev.lt(RUNAWAY_TE_EV)).any():
            raise ValueError("runaway disagrees with the Thomson threshold")
        for text in frame.neutron_rate_mean:
            if any(
                token and not np.isfinite(float(token)) for token in text.split(";")
            ):
                raise ValueError(
                    "neutron_rate_mean must contain finite means or blanks"
                )
    except (OSError, ValueError, OverflowError, pd.errors.ParserError) as error:
        raise CatalogError(f"{path}: {error}") from error
    return frame


def corroborated(evidence: pd.DataFrame) -> pd.Series:
    """Check neutron/beam support for each row without changing its Thomson mark."""
    neutrons = evidence.neutron_rate_mean.map(
        lambda text: max(
            (float(token) for token in text.split(";") if token), default=0.0
        )
    )
    beams = pd.to_numeric(evidence.pinj_kw).fillna(0.0)
    return neutrons.ge(RUNAWAY_MIN_NEUTRONS) & beams.lt(RUNAWAY_MAX_BEAM_KW)


def apply_rule(frame: pd.DataFrame, evidence: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Apply rule 5 to the same complete rule-4 set used by the scan."""
    eligible = frame.reasons.eq("")
    missing = sorted(set(frame.loc[eligible, "shot"]) - set(evidence.shot))
    if missing:
        raise CatalogError(
            f"{path}: {len(missing)} rule-4 shots missing: {missing[:5]}"
        )
    unsupported = sorted(
        evidence.loc[evidence.runaway & ~corroborated(evidence), "shot"]
    )
    if unsupported:
        raise CatalogError(
            f"{path}: marked shots not corroborated by neutrons and beams: "
            f"{unsupported}"
        )
    marked = set(evidence.loc[evidence.runaway, "shot"])
    result = frame.copy()
    result.loc[eligible & result.shot.isin(marked), "reasons"] = "runaway_plateau"
    return result


def _one(row, *, corpus):
    shot, start, end = row
    return assess(corpus / f"{shot}_processed.h5", shot, start, end)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m labeler.events.catalog.runaway")
    parser.add_argument(
        "--pool", type=Path, help="default: $LABELER_ROOT/catalog/pool.csv"
    )
    parser.add_argument(
        "--ip-log", type=Path, help="default: $LABELER_ROOT/catalog/ip.jsonl"
    )
    parser.add_argument("--out", type=Path, help="default: <label tables>/catalog")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args(argv)
    try:
        return _run(args, argv, Paths.from_env())
    except (CatalogError, OSError, pd.errors.ParserError) as error:
        parser.error(str(error))


def _run(args, argv, paths) -> int:
    started = time.monotonic()
    pool_path = args.pool or paths.catalog / "pool.csv"
    log_path = args.ip_log or paths.catalog / "ip.jsonl"
    out = args.out or paths.label_tables / "catalog"
    pool_data, log_data = pool_path.read_bytes(), log_path.read_bytes()
    frame = pop.population(
        pop.read_pool(NamedBytes(pool_data, pool_path)),
        window.read_log(NamedBytes(log_data, log_path)),
    )
    selected = frame.loc[
        frame.reasons.eq(""), ["shot", "window_start_ms", "window_end_ms"]
    ].sort_values("shot")
    rows = list(selected.itertuples(index=False, name=None))
    code = {"git_sha": git_sha(full=True), "git_dirty": git_dirty()}
    measure = partial(_one, corpus=paths.corpus)
    if args.workers == 1:
        measured = list(map(measure, rows))
    else:
        with Pool(args.workers) as pool:
            measured = pool.map(measure, rows, chunksize=8)
    result = pd.DataFrame([r for r, _ in measured], columns=COLUMNS)
    counts = {
        "shots": len(result),
        "marked": int(result.runaway.sum()),
        "no_thomson": int(result.n_profile.eq(0).sum()),
    }
    files = [
        {"path": f"{row['shot']}_processed.h5", "sha256": digest}
        for row, digest in measured
    ]
    file_bytes = "".join(f"{f['path']} {f['sha256']}\n" for f in files).encode()
    saved = result.copy()
    saved["runaway"] = saved.runaway.map({True: "true", False: "false"})
    data = saved.to_csv(index=False).encode()
    meta = {
        **code,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": shlex.join(
            ["python", "-m", "labeler.events.catalog.runaway", *argv]
        ),
        **definition(),
        "inputs": {
            "pool": {
                "path": str(pool_path),
                "sha256": hashlib.sha256(pool_data).hexdigest(),
            },
            "ip_log": {
                "path": str(log_path),
                "sha256": hashlib.sha256(log_data).hexdigest(),
                "version": window.LOG_VERSION,
            },
            "corpus_files": {
                "directory": str(paths.corpus),
                "count": len(files),
                "sha256": hashlib.sha256(file_bytes).hexdigest(),
                "digest_definition": "sha256 of sorted '<filename> <digest>\\n'; "
                "each digest hashes group name + newline, then 'absent\\n' or "
                "each xdata/ydata shape (Python tuple string) + newline "
                "and C-order little-endian float64 bytes",
                "groups": list(GROUPS),
                "files": files,
            },
        },
        "counts": counts,
        "outputs": {"runaway.csv": hashlib.sha256(data).hexdigest()},
    }
    # Validate our serialization before replacing either committed input.
    read_runaway(NamedBytes(data, out / "runaway.csv"))
    for name, content in (
        ("runaway.csv", data),
        ("runaway.meta.json", (json.dumps(meta, indent=1) + "\n").encode()),
    ):
        with atomic_path(out / name) as tmp:
            tmp.write_bytes(content)
    nearest = result.loc[~result.runaway & result.te_p90_ev.notna()].nsmallest(
        10, "te_p90_ev"
    )
    print(
        json.dumps(
            {
                **counts,
                "marked_shots": result.loc[result.runaway, "shot"].tolist(),
                "nearest_unmarked": nearest[["shot", "te_p90_ev"]].to_dict("records"),
                "elapsed_s": round(time.monotonic() - started, 3),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
