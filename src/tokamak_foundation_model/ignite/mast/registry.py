"""The channel registry: one row per channel of the 39 signals the 14 TokaMark tasks use.

The registry is the readable record of what each channel is (units, IMAS path, position, family,
the roles it plays in the tasks) and of how it is standardized (the last four columns repeat the
numbers of ``stats.npz``, which is what the code reads; `stats_mismatches` compares the two, and
`cache.build` refuses to reuse a ``registry.csv`` that disagrees with ``stats.npz`` or `SAT_ABS`).
`zarr`, `tokamark` and `MAST_tools` are imported inside the functions that use them.

An empty cell means "not found", never zero: units, IMAS path, channel name, r and z are
copied from the Zarr attributes and coordinates where the store has them. Signals that
TokaMark's loader derives (the two Mirnov spectrograms, whose units differ from the stored raw
signal) have no units or IMAS path here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import transform
from .paths import prepare_write_root
from .tasks import ALL_TASKS, SPECTROGRAM_SIGNALS, SignalSpec, load_task
from .transform import ChannelStats

REGISTRY_COLUMNS = ("device", "signal", "channel", "channel_name", "family", "dt", "units", "imas_path",
                    "r", "z", "tm_mean", "tm_std", "roles", "sat_abs", "valid_fraction", "dead_fraction",
                    "compress", "scale", "mean", "std")
_STRING_COLUMNS = ("device", "signal", "channel_name", "family", "units", "imas_path", "roles", "compress")
_FLOAT_COLUMNS = ("dt", "r", "z", "tm_mean", "tm_std", "sat_abs", "valid_fraction", "dead_fraction",
                  "scale", "mean", "std")
_ROLE_CODES = (("inputs", "inp"), ("actuators", "act"), ("outputs", "out"))
_LISTED = 20                    # signals named in a printed summary

# ADC range of plus or minus 5 V (data profile of 2026-11-21): the largest value a channel
# can record, a little below 5 V. Empty for every other signal.
SAT_ABS = {"spectrometer_visible-filter_spectrometer_dalpha_voltage": 4.99}


# ------------------------------------------------------------------------------------ signals

def _collect() -> tuple[dict[str, SignalSpec], dict[str, set[str]]]:
    specs: dict[str, SignalSpec] = {}
    roles: dict[str, set[str]] = {}
    for task_name in ALL_TASKS:
        task = load_task(task_name)
        for field, code in _ROLE_CODES:
            for s in getattr(task, field):
                prev = specs.setdefault(s.name, s)
                if prev != s:
                    raise ValueError(f"{s.name} differs between tasks (seen {prev}, then {s} in {task_name})")
                roles.setdefault(s.name, set()).add(code)
    return specs, roles


def all_signals() -> list[SignalSpec]:
    """The distinct signals over the 14 tasks and all three roles (39), sorted by name."""
    specs, _ = _collect()
    return [specs[k] for k in sorted(specs)]


def _roles() -> dict[str, str]:
    """``"inp+out"``: the sorted set of roles of each signal over the 14 tasks."""
    _, roles = _collect()
    return {k: "+".join(sorted(v)) for k, v in roles.items()}


def family_of(s: SignalSpec) -> str:
    """``"slow" | "high_rate" | "profile" | "map"``, by rule.

    Two non-time dimensions at 5 ms is a map (psi). Thomson profiles and the boundary
    (``lcfs_r``, ``lcfs_z``) are profiles. 0.02 ms and the two spectrograms are high rate.
    Everything else is slow.
    """
    if len(s.values_shape) == 2 and np.isclose(s.dt, 0.005):
        return "map"
    if s.name.startswith("thomson_scattering-") or s.name in ("equilibrium-lcfs_r", "equilibrium-lcfs_z"):
        return "profile"
    if np.isclose(s.dt, 2e-5) or s.name in SPECTROGRAM_SIGNALS:
        return "high_rate"
    return "slow"


# ------------------------------------------------------------------- attributes and coordinates

_TIME_DIMS = ("time",)


def _is_time(dim: str) -> bool:
    return dim.startswith(_TIME_DIMS)        # "time", "time_saddle", "time_mirnov"


def _read_attributes(zarr_root: Path, specs: list[SignalSpec], sample_shots: list[int]) -> dict[str, dict]:
    """``{name: {"units", "imas_path", "channel_names", "r", "z"}}`` as far as the store offers them.

    A source can be missing from a shot, so each signal is looked for across ``sample_shots``
    until a shot has it. Coordinates are used only where they line up with the signal as
    TokaMark's loader returns it (same number of non-time dimensions, same lengths); derived
    signals, whose stored array is something else, get none.
    """
    import zarr

    found: dict[str, dict] = {}
    todo = {s.name: s for s in specs if s.name not in SPECTROGRAM_SIGNALS}
    n_wanted, unopened, tried = len(todo), 0, 0
    for shot in sample_shots:
        if not todo:
            break
        tried += 1
        try:
            group = zarr.open_group(str(Path(zarr_root) / f"{shot}.zarr"), mode="r")
        except Exception:      # a shot directory that is not there or cannot be opened: try the next
            unopened += 1
            continue
        for name in list(todo):
            spec = todo[name]
            source, signal = name.split("-", 1)
            if source not in group or signal not in group[source]:
                continue
            src, arr = group[source], group[source][signal]
            attrs = arr.attrs.asdict()
            rec = {"units": str(attrs.get("units", "") or ""), "imas_path": str(attrs.get("imas", "") or ""),
                   "channel_names": None, "r": None, "z": None}
            dims = [d for d in attrs.get("_ARRAY_DIMENSIONS", []) if not _is_time(d)]
            shape = tuple(spec.values_shape)
            if len(dims) == len(shape):
                for axis, dim in enumerate(dims):
                    if dim not in src or src[dim].ndim != 1 or src[dim].shape[0] != shape[axis]:
                        continue
                    coord = src[dim][:]
                    if dim == "major_radius":
                        rec["r"] = (axis, np.asarray(coord, dtype=np.float64))
                    elif dim == "z":
                        rec["z"] = (axis, np.asarray(coord, dtype=np.float64))
                    elif len(shape) == 1 and coord.dtype.kind in "USO":
                        rec["channel_names"] = [str(c) for c in coord]
            found[name] = rec
            del todo[name]
    # Said, like the validity pass says its counts: a signal no sample shot has leaves empty units and
    # IMAS cells ("not found"), and a shot that cannot be opened was skipped without a word.
    print(f"registry: attributes read for {len(found)} of {n_wanted} signals "
          f"({unopened} of {tried} sample shots could not be opened)")
    if todo:
        print(f"  no attributes found for: {', '.join(sorted(todo)[:_LISTED])}")
    return found


# ------------------------------------------------------------------------------ validity

def _validity_shot(args: tuple) -> tuple[int, dict | None, str | None]:
    """Per channel of one shot: the finite fraction of its samples and whether it is dead.

    Dead: no finite sample (the cache's definition of a dead channel: a channel held at one value
    is valid and only flagged there, and counts as live here). A signal the shot does not have, or
    has with another shape, is ``None``; `_measure_validity` then counts every channel of it as
    dead, which the cache does not (it records such a shot as ``absent``, not as dead channels),
    so ``dead_fraction`` is "no finite sample, or the shot lacks the signal".
    """
    shot, signals = args
    try:
        item = transform.load_shot(shot, signals)
    except Exception as e:
        return shot, None, f"{type(e).__name__}: {e}"
    out: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for spec in signals:
        v = transform._channels(item, spec)
        if v is None:
            out[spec.name] = None
            continue
        fin = np.isfinite(v)
        cnt = fin.sum(axis=1)
        out[spec.name] = (cnt / v.shape[1], cnt == 0)
    return shot, out, None


def _measure_validity(signals: list[SignalSpec], sample_shots: list[int], workers: int) -> dict[str, tuple]:
    """``{name: (valid_fraction (C,), dead_fraction (C,))}`` over the shots that could be read.

    A shot that lacks the signal counts as dead (it has no finite sample) and as 0 valid. This
    merges two states that the cache keeps apart (a dead channel of a present signal, and an
    ``absent`` signal), and the column is left as it is: read it as "dead or absent". A
    shot that cannot be read at all is left out of the denominator and counted in the printed
    summary. With no readable shot the fractions are NaN, not zero.
    """
    n_ch = {s.name: int(np.prod(s.values_shape)) for s in signals}
    valid = {k: np.zeros(c) for k, c in n_ch.items()}
    dead = {k: np.zeros(c) for k, c in n_ch.items()}
    failed: list[tuple[int, str]] = []
    n_ok = 0
    for shot, out, err in transform.map_shots(_validity_shot, sample_shots, (signals,), workers):
        if out is None:
            failed.append((shot, err))
            continue
        n_ok += 1
        for name, r in out.items():
            if r is None:
                dead[name] += 1
            else:
                valid[name] += r[0]
                dead[name] += r[1]
    print(f"registry: validity measured on {n_ok} of {len(sample_shots)} shots, "
          f"{len(failed)} shots failed to load")
    for shot, err in failed[:3]:
        print(f"  shot {shot}: {err[:200]}")
    denom = n_ok if n_ok else np.nan
    return {k: (valid[k] / denom, dead[k] / denom) for k in n_ch}


# ------------------------------------------------------------------------------------ the table

def _signal_rows(spec: SignalSpec, role: str, attrs: dict, validity: tuple) -> pd.DataFrame:
    shape = tuple(spec.values_shape)
    n = int(np.prod(shape))
    idx = np.stack(np.unravel_index(np.arange(n), shape), axis=1)         # (n, ndim), C order
    names = attrs.get("channel_names")
    if len(shape) == 2:
        channel_name = [f"{i},{j}" for i, j in idx]
    elif names is not None:
        channel_name = names
    else:
        channel_name = [""] * n

    def coordinate(key: str) -> np.ndarray:
        got = attrs.get(key)
        if got is None:
            return np.full(n, np.nan)
        axis, values = got
        return values[idx[:, axis]]

    return pd.DataFrame({
        "device": "MAST", "signal": spec.name, "channel": np.arange(n), "channel_name": channel_name,
        "family": family_of(spec), "dt": spec.dt, "units": attrs.get("units", ""),
        "imas_path": attrs.get("imas_path", ""), "r": coordinate("r"), "z": coordinate("z"),
        "tm_mean": spec.mean, "tm_std": spec.std, "roles": role,
        "sat_abs": SAT_ABS.get(spec.name, np.nan), "valid_fraction": validity[0],
        "dead_fraction": validity[1], "compress": "", "scale": np.nan, "mean": np.nan, "std": np.nan,
    })[list(REGISTRY_COLUMNS)]


def build_registry(zarr_root: Path, sample_shots: list[int], *, workers: int = 1) -> pd.DataFrame:
    """The registry of the 39 signals, the last four columns (compress, scale, mean, std) empty.

    ``sample_shots`` is used exactly as given (the caller passes training shots). Units, IMAS
    paths, channel names, r and z come from the Zarr store at ``zarr_root``; ``valid_fraction``
    and ``dead_fraction`` (the fraction of sampled shots in which the channel has no finite sample,
    or the shot lacks the signal) are measured through `transform.load_shot`, i.e. on the data the
    model sees (outliers replaced, derived signals included). ``workers`` is a process count for
    that pass. `attach_stats` fills the rest.
    """
    sample_shots = [int(s) for s in sample_shots]
    signals = all_signals()
    roles = _roles()
    attrs = _read_attributes(Path(zarr_root), signals, sample_shots)
    validity = _measure_validity(signals, sample_shots, workers)
    frames = [_signal_rows(s, roles[s.name], attrs.get(s.name, {}), validity[s.name]) for s in signals]
    df = pd.concat(frames, ignore_index=True)
    return df


def attach_stats(df: pd.DataFrame, stats: dict[str, ChannelStats]) -> pd.DataFrame:
    """A copy of ``df`` with compress, scale, mean and std filled from ``stats``.

    Row ``channel`` c of a signal takes the c-th element of the C-order flattened arrays.
    Signals without statistics stay empty. A signal whose rows do not match the size of its
    arrays raises ``ValueError``.
    """
    out = df.copy()
    out["compress"] = out["compress"].astype(object)
    for name, st in stats.items():
        rows = out.index[out["signal"] == name]
        arrays = {k: np.asarray(getattr(st, k), dtype=np.float64).ravel() for k in ("scale", "mean", "std")}
        n = arrays["mean"].size
        channels = out.loc[rows, "channel"].to_numpy()
        if len(rows) != n or (len(rows) and (channels.min() < 0 or channels.max() >= n)):
            raise ValueError(f"{name}: the registry has {len(rows)} rows but the statistics have {n} channels")
        for k, a in arrays.items():
            out.loc[rows, k] = a[channels]
        out.loc[rows, "compress"] = st.compress
    return out


_STATS_COLUMNS = ("compress", "scale", "mean", "std")
_MISMATCH_RTOL = 1e-9       # the CSV holds shortest-repr floats; pandas' fast parser may differ in the last digit


def stats_mismatches(df: pd.DataFrame, stats: dict[str, ChannelStats]) -> list[str]:
    """What in the registry ``df`` disagrees with ``stats`` and `SAT_ABS`; an empty list when it agrees.

    The registry repeats the numbers of ``stats.npz`` (compress, scale, mean, std) and holds the
    saturation limits of `SAT_ABS` (``sat_abs``). `cache.build` reuses a ``registry.csv`` that
    exists, and the shot files are written from ``stats.npz`` and from that file's ``sat_abs``, so
    a registry left over from other statistics would put other limits into the files while it
    still shows the old constants. Floats are compared to a relative 1e-9 (NaN equals NaN).
    """
    try:
        expected = attach_stats(df, stats)
    except ValueError as e:
        return [f"the registry does not fit the statistics: {e}"]
    problems = []

    def describe(column: str, bad: np.ndarray) -> str:
        first = df.loc[bad].iloc[0]
        return f"{column} differs in {int(bad.sum())} rows (first: {first['signal']} channel {first['channel']})"

    bad = (df["compress"].astype(str).to_numpy() != expected["compress"].astype(str).to_numpy())
    if bad.any():
        problems.append(describe("compress", bad))
    want_sat = df["signal"].map(lambda n: SAT_ABS.get(n, np.nan)).to_numpy(dtype=np.float64)
    for column, want in (*((c, expected[c].to_numpy(dtype=np.float64)) for c in _STATS_COLUMNS[1:]),
                         ("sat_abs", want_sat)):
        have = df[column].to_numpy(dtype=np.float64)
        same = np.isclose(have, want, rtol=_MISMATCH_RTOL, atol=0.0, equal_nan=True)
        if not same.all():
            problems.append(describe(column, ~same))
    return problems


def reconciled(df: pd.DataFrame, stats: dict[str, ChannelStats]) -> pd.DataFrame:
    """A copy of ``df`` with the statistics columns from ``stats`` and ``sat_abs`` from `SAT_ABS`.

    Raises ``ValueError`` where `attach_stats` does (a signal whose rows do not fit its arrays):
    that registry has to be measured again, not patched.
    """
    out = attach_stats(df, stats)
    out["sat_abs"] = out["signal"].map(lambda n: SAT_ABS.get(n, np.nan)).to_numpy(dtype=np.float64)
    return out


def save_registry(df: pd.DataFrame, path: Path) -> None:
    """CSV, empty cells for not found. Written under a root that is not the repository."""
    path = Path(path)
    prepare_write_root(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    df[list(REGISTRY_COLUMNS)].to_csv(tmp, index=False)
    tmp.replace(path)


def load_registry(path: Path) -> pd.DataFrame:
    """Read `save_registry`'s CSV: empty text columns stay ``""``, empty numbers are NaN."""
    dtype = {c: str for c in _STRING_COLUMNS} | {"channel": np.int64} | {c: np.float64 for c in _FLOAT_COLUMNS}
    df = pd.read_csv(path, dtype=dtype, keep_default_na=False,
                     na_values={c: [""] for c in _FLOAT_COLUMNS})
    if tuple(df.columns) != REGISTRY_COLUMNS:
        raise ValueError(f"{path} does not have the registry columns")
    return df
