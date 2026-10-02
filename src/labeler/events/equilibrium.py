"""Equilibrium review signals and finite sample runs.

Drafts read local stores only. Panels may resolve an archive column or fetch a
canonical feature, caching it under the raw-cache root, never in the corpus.
"""

from __future__ import annotations

import numpy as np

from ..config import Paths
from ..features import namespace, resolve_archive, resolve_fdp
from ..features.store import FeatureArray, read_feature, write_features
from . import coverage, heuristics, raw
from .verify import NoDataError, Panel

MIN_MS = 500.0
BETAP_THRESHOLD = 1.0
QMIN_THRESHOLDS = (heuristics.QMIN_HYBRID, heuristics.QMIN_ELEVATED,
                   heuristics.QMIN_HIGH)
QMIN_UNCERTAIN = 5
QMIN_NOT_OBSERVABLE = 6
RULE = {
    "min_ms": MIN_MS,
    "ip_fraction": heuristics.FLATTOP_FRAC,
    "equilibrium": "efit01",
    "coverage": "finite scalar samples inside the longest Ip flat-top",
    "sample_bounds": "first and last samples; gaps and short bands stay unknown",
    "archive_time": "25 ms stamp lag removed; values retain the 50 ms boxcar",
}


def canonical(array: FeatureArray, name: str) -> FeatureArray:
    """Align archive sample centres and exclude invalid physical scalar values."""
    x, y, attrs = array.x, array.y, dict(array.attrs)
    if np.asarray(x).ndim != 1 or not np.isfinite(x).all() or (
        len(x) > 1 and (np.diff(x) <= 0).any()
    ):
        raise ValueError(f"{name} times must be finite and strictly increasing")
    if attrs.get("resolver") == "archive" and "archive_lag_corrected_s" not in attrs:
        x = x - namespace.STEP_S
        attrs.update(archive_lag_corrected_s=str(namespace.STEP_S),
                     sample_window_s=str(2 * namespace.STEP_S))
    positive = name in ("qmin", "li")
    nonnegative = name in ("betap", "betan", "n1rms", "n2rms", "wmhd")
    if (positive or nonnegative or name == "ip") and y.shape[0] != 1:
        raise ValueError(f"{name} must be one scalar trace")
    if positive or nonnegative:
        valid = y > 0 if positive else y >= 0
        y = np.where(valid & np.isfinite(y), y, np.nan)
    return FeatureArray(x=x, y=y, attrs=attrs)


def signal(shot: int, name: str, paths: Paths, *, fetch=False) -> FeatureArray:
    """Seconds and canonical values, preferring features to corpus and raw cache."""
    clock_problem = ""
    for path in (paths.features_file(shot), paths.corpus_file(shot),
                 raw.cache_path(shot, paths=paths)):
        try:
            array = read_feature(path, name)
        except (OSError, KeyError):
            continue
        array = canonical(array, name)
        if len(array.x) > 1 and np.isfinite(array.y).any():
            # An untagged legacy raw cache may use archive or native times.
            # It is unavailable until preparation refreshes its source clock.
            if path == raw.cache_path(shot, paths=paths) and (
                name in ("betap", "betan", "ip", "qpsi")
                and array.attrs.get("resolver") not in ("archive", "fdp")
            ):
                clock_problem = "; ambiguous legacy clock; refresh with preparation"
                continue
            return FeatureArray(array.x, array.y, {"store": str(path), **array.attrs})
    if fetch and not raw.fetching_disabled():
        spec = namespace.by_name(name)
        for source, resolver in (("archive", resolve_archive), ("fdp", resolve_fdp)):
            if source not in spec.sources:
                continue
            arrays, missing = resolver.resolve(int(shot), [name])
            if name not in arrays:
                continue
            array = canonical(arrays[name], name)
            if len(array.x) < 2 or not np.isfinite(array.y).any():
                continue
            path = raw.cache_path(shot, paths=paths)
            # Use the raw writer's thread lock while preserving both native
            # clock precision and resolver/locator attrs in the feature layout.
            with raw._lock_for(path.resolve()):
                write_features(path, int(shot), {name: array}, {})
            return array
        raise NoDataError(f"shot {shot}: no {name}: "
                          f"{missing.get(name, 'no samples')}{clock_problem}")
    raise NoDataError(f"shot {shot}: no {name} in features, corpus or raw cache"
                      f"{clock_problem}")


def sample_runs(t_ms, y) -> tuple[tuple[float, float], ...]:
    """Finite runs; NaNs and a gap exceeding 1.5 native steps split them."""
    t = np.asarray(t_ms, dtype=np.float64).ravel()
    values = heuristics._one_trace("scalar", y)
    if len(t) != len(values) or not np.isfinite(t).all():
        raise ValueError("a scalar needs one finite time per sample")
    if len(t) < 2:
        return ()
    steps = np.diff(t)
    if (steps <= 0).any():
        raise ValueError("scalar times must be strictly increasing")
    index = np.flatnonzero(np.isfinite(values))
    if not len(index):
        return ()
    cuts = np.flatnonzero((np.diff(index) > 1)
                          | (np.diff(t[index]) > 1.5 * np.median(steps))) + 1
    groups = np.split(index, cuts)
    return tuple((float(t[g[0]]), float(t[g[-1]])) for g in groups if len(g) > 1)


def gated(shot: int, name: str, paths: Paths, window):
    """The scalar and the overlap of its finite record with the Ip flat-top."""
    scalar = signal(shot, name, paths)
    ip = signal(shot, "ip", paths)
    ip_t = ip.x * 1000
    level = np.abs(heuristics._one_trace("ip", ip.y))
    peak = np.max(level[np.isfinite(level)]) if np.isfinite(level).any() else 0
    above = np.where(level > heuristics.FLATTOP_FRAC * peak, level, np.nan)
    runs = sample_runs(ip_t, above)
    if not runs:
        raise NoDataError(f"shot {shot}: no Ip flat-top")
    flattop = max(runs, key=lambda span: span[1] - span[0])
    measured = coverage.intersect_intervals(
        sample_runs(scalar.x * 1000, scalar.y), (flattop,),
    )
    return scalar, measured, {name: dict(scalar.attrs), "ip": dict(ip.attrs)}


def _bands(array, measured, classify, *, uncertain: int, low_is_absent=False):
    t = array.x * 1000
    values = heuristics._one_trace("scalar", array.y)
    classes = classify(values)
    spans = []
    for state in sorted(set(classes[np.isfinite(values)])):
        selected = np.where(classes == state, values, np.nan)
        for a, b in coverage.intersect_intervals(sample_runs(t, selected), measured):
            # Float32 legacy cache clocks differ by microseconds, far below
            # an EFIT slice; exact 500 ms runs should survive a round trip.
            category = int(state) if (
                (low_is_absent and state == 0) or b - a >= MIN_MS - 1e-3
            ) else uncertain
            spans.append((a, b, category))
    return tuple(sorted(spans))


def qmin(shot: int, paths: Paths, window=None):
    from .spans import Found

    array, measured, inputs = gated(shot, "qmin", paths, window)
    spans = _bands(array, measured,
                   lambda q: np.digitize(q, QMIN_THRESHOLDS, right=True) + 1,
                   uncertain=QMIN_UNCERTAIN)
    return Found(spans, measured, {"equilibrium": "efit01",
                                  "inputs": inputs})


def betap(shot: int, paths: Paths, window=None):
    from .spans import Found

    array, measured, inputs = gated(shot, "betap", paths, window)
    spans = _bands(array, measured, lambda beta: (beta > BETAP_THRESHOLD).astype(int),
                   uncertain=2, low_is_absent=True)
    return Found(spans, measured, {"equilibrium": "efit01",
                                  "inputs": inputs})


def line_panel(shot, name, title, *, paths, t_range=None, ylabel="", hlines=()):
    """A canonical scalar in milliseconds, with optional threshold lines."""
    try:
        array = signal(int(shot), name, paths, fetch=True)
    except ValueError as error:
        raise NoDataError(f"shot {shot}: malformed {name}: {error}") from error
    x, y = plot_arrays(array)
    if t_range is not None:
        keep = (x >= t_range[0]) & (x <= t_range[1])
        x, y = x[keep], y[:, keep]
    return [Panel(title=title, x=x, y=y, ylabel=ylabel, hlines=list(hlines))]


def plot_arrays(array: FeatureArray):
    """Millisecond arrays with explicit NaNs at missing-time stretches."""
    x, y = array.x * 1000, array.y
    if len(x) > 2:
        steps = np.diff(x)
        holes = np.flatnonzero(steps > 1.5 * np.median(steps))
        if len(holes):
            at = holes + 1
            x = np.insert(x, at, (x[holes] + x[holes + 1]) / 2)
            y = np.insert(y, at, np.nan, axis=1)
    return x, y


def _prepare(job):
    shot, names, paths = job
    missing = {}
    for name in names:
        try:
            signal(shot, name, paths, fetch=True)
        except (NoDataError, KeyError, OSError, ValueError) as error:
            missing[name] = str(error)
    return shot, missing


def main(argv=None) -> int:
    """Prefetch the canonical inputs to one editor into its local raw cache."""
    import argparse
    import json
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from . import spans

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--event", required=True, choices=(
        "minimum_safety_factor", "poloidal_beta", "resistive_wall_mode"))
    parser.add_argument("--shots", type=int, nargs="+")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.limit < 0:
        parser.error("workers must be positive and limit nonnegative")
    paths = Paths.from_env()
    targets = spans.targets(args.event, paths, "cohort")
    if args.shots:
        unknown = set(args.shots) - set(targets.shot)
        if unknown:
            parser.error(f"not in the review queue (or blind): {sorted(unknown)}")
        targets = targets[targets.shot.isin(args.shots)]
    if args.limit:
        targets = targets.head(args.limit)
    names = spans.METHODS[args.event].inputs
    jobs = [(int(shot), names, paths) for shot in targets.shot]
    print(f"{args.event}: preparing {len(jobs)} shots", flush=True)
    if args.workers == 1:
        results = [_prepare(job) for job in jobs]
    else:
        # Fork before any fdp fetch: toksearch's PTDATA reader is not fork-safe.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_prepare, job) for job in jobs]
            results = []
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 25 == 0:
                    print(f"{len(results)}/{len(jobs)} prepared", flush=True)
    missing = {str(shot): why for shot, why in sorted(results) if why}
    print(json.dumps({"event": args.event, "shots": len(results), "missing": missing}))
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
