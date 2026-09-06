"""Reading and writing `<shot>_features.h5`.

Same layout as the corpus itself - one group per quantity, `xdata` seconds,
`ydata` (C, T) - so anything that can read a corpus file can read a feature
file. A merge rewrites the whole file and renames it into place; a killed run
therefore never leaves a half-written feature file behind.

Most files are a few MB. One feature is not: `co2` is `(4, ~4.5e6)` float32,
72 MB, which is why `ydata` above `CHUNK_THRESHOLD` is written chunked and
read back in float32 rather than promoted to float64. A merge on such a shot
rewrites those 72 MB, which is the price of the atomic-rename guarantee.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

from .. import __version__
from ..config import atomic_path, git_sha
from . import namespace as ns

MISSING_ATTR = "missing"


@dataclass(frozen=True)
class FeatureArray:
    """One resolved feature: seconds on `x`, `(C, T)` values on `y`."""

    x: np.ndarray
    y: np.ndarray
    attrs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if np.asarray(self.y).ndim != 2:
            raise ValueError(f"ydata must be (C, T), got {np.shape(self.y)}")
        if np.shape(self.y)[-1] != np.shape(self.x)[-1]:
            raise ValueError(f"x {np.shape(self.x)} and y {np.shape(self.y)} disagree")


#: A `ydata` at least this large is stored chunked rather than contiguous.
#: The only feature that reaches it is `co2` - `(4, ~4.5e6)` float32, 72 MB -
#: and chunking is what lets h5py write and read it in pieces instead of
#: materialising the whole dataset at once. Deliberately no compression: the
#: record is broadband interferometer noise, which gzip barely shrinks while
#: costing minutes per shot.
CHUNK_THRESHOLD = 1 << 20


def _dataset_kwargs(y: np.ndarray) -> dict:
    """`create_dataset` options for one `ydata`, chunked when it is large."""
    if y.size < CHUNK_THRESHOLD:
        return {}
    # One chunk per channel, ~1 M samples wide: a whole-channel read (what
    # the AE adapter does) touches contiguous chunks, and a time-slice read
    # touches one chunk per channel.
    width = min(y.shape[-1], 1 << 20)
    return {"chunks": (1,) * (y.ndim - 1) + (width,)}


def _read_group(g, *, float32: bool = False) -> FeatureArray:
    """One stored group back into a FeatureArray.

    float64 by default, matching every consumer that samples onto the 25 ms
    grid. `float32` keeps a waveform in the dtype it was stored in: `co2` is
    `(4, ~4.5e6)`, which float64 would double to 144 MB per shot for no
    precision that was ever measured - the corpus itself is float32.
    """
    return FeatureArray(
        x=np.asarray(g["xdata"], dtype=np.float64),
        y=np.asarray(g["ydata"], dtype=np.float32 if float32 else np.float64),
        attrs={k: str(v) for k, v in g.attrs.items()},
    )


def _is_waveform(name: str) -> bool:
    try:
        return ns.by_name(name).kind == "waveform"
    except KeyError:
        return False


def _load_all(path: Path) -> tuple[dict[str, FeatureArray], dict[str, str]]:
    if not Path(path).exists():
        return {}, {}
    with h5py.File(path, "r") as f:
        arrays = {
            name: _read_group(f[name], float32=_is_waveform(name)) for name in f
        }
        missing = json.loads(f.attrs.get(MISSING_ATTR, "{}"))
    return arrays, missing


def write_features(
    path,
    shot: int,
    arrays: dict[str, FeatureArray],
    missing: dict[str, str],
    *,
    merge: bool = True,
) -> None:
    """Write a feature file atomically.

    With `merge`, groups already in the file are kept and any name now
    resolved is dropped from the recorded misses, so a rerun that fetches
    one more feature does not throw away the previous ones.

    A feature carrying fewer than two samples is moved into `missing`
    instead of being stored; see the comment below. The caller's `arrays`
    and `missing` dicts are never mutated.
    """
    path = Path(path)
    arrays, missing = dict(arrays), dict(missing)
    if merge:
        kept, kept_missing = _load_all(path)
        kept.update(arrays)
        kept_missing.update(missing)
        for name in arrays:
            kept_missing.pop(name, None)
        arrays, missing = kept, kept_missing
    # A group with fewer than two samples is the corpus' "signal absent"
    # sentinel, and these files share the corpus layout - so a *resolved*
    # one-sample group would read as absent to any consumer applying the
    # corpus rule, which `catalog.available_groups` does. It is reachable:
    # `decimate_to_step` on a degenerate time axis returns one sample.
    #
    # Such a feature is demoted into `missing` rather than persisted, and
    # rather than raised on. Raising would cost the shot every other feature
    # resolved in the same call, against this pipeline's rule that one bad
    # feature never costs a shot the rest; `missing` is the channel built to
    # carry exactly this. Demotion also heals a file that somehow already
    # holds a short group, which a raise would have made unwritable forever.
    for name in [n for n, a in arrays.items() if a.y.shape[-1] < 2]:
        missing[name] = f"OneSampleAmbiguous({arrays[name].y.shape[-1]})"
        del arrays[name]
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with atomic_path(path) as tmp, h5py.File(tmp, "w") as f:
        f.attrs["shot"] = int(shot)
        f.attrs["labelmaker_version"] = __version__
        f.attrs["git_sha"] = git_sha()
        f.attrs["written_at"] = now
        f.attrs[MISSING_ATTR] = json.dumps(missing, sort_keys=True)
        for name, arr in sorted(arrays.items()):
            g = f.create_group(name)
            g.create_dataset("xdata", data=np.asarray(arr.x, dtype=np.float64))
            y = np.asarray(arr.y, dtype=np.float32)
            g.create_dataset("ydata", data=y, **_dataset_kwargs(y))
            for k, v in arr.attrs.items():
                g.attrs[k] = v
            if "fetched_at" not in g.attrs:
                g.attrs["fetched_at"] = now
            g.attrs["complete"] = 1
            try:
                spec = ns.by_name(name)
            except KeyError:
                continue
            g.attrs["units"] = spec.units
            if spec.kind == "profile" and np.shape(arr.y)[0] == ns.RHO_GRID.size:
                g.create_dataset("rho", data=ns.RHO_GRID)


def read_feature(path, name: str) -> FeatureArray:
    """One feature out of a feature file, or KeyError."""
    with h5py.File(path, "r") as f:
        if name not in f:
            raise KeyError(f"{name} not in {path}")
        return _read_group(f[name], float32=_is_waveform(name))


def present(path) -> set[str]:
    """Feature names stored in the file (empty if the file does not exist)."""
    if not Path(path).exists():
        return set()
    with h5py.File(path, "r") as f:
        return set(f.keys())


def resolvers(path) -> dict[str, str]:
    """Stored feature name -> the source that actually produced it.

    The authoritative per-shot provenance: `namespace.sources` says only
    where a feature *could* come from, and a run resolves cheapest-source
    -first per feature, so one shot's file can legitimately carry a mix.
    That mix matters - see `resolve_archive`'s docstring for the 25 ms row
    lag the archive carries relative to the corpus and fdp - so the runner
    reports it per shot rather than leaving it to be discovered.
    """
    if not Path(path).exists():
        return {}
    with h5py.File(path, "r") as f:
        return {name: str(f[name].attrs.get("resolver", "unknown")) for name in f}


def missing_names(path) -> dict[str, str]:
    """Feature name -> cause, for everything a run tried and could not get."""
    if not Path(path).exists():
        return {}
    with h5py.File(path, "r") as f:
        return json.loads(f.attrs.get(MISSING_ATTR, "{}"))


#: Miss causes worth another attempt on a later run. Everything else - an
#: absent node, a shape mismatch, a genuinely one-sample record - is a
#: property of the data and will fail again identically, so a rerun skips it.
#:
#: The last three are properties of the *process*, not of the data, and they
#: are here because of a MEASURED trap. `resolve_fdp.available()` only checks
#: that toksearch imports, which it does in this environment - so a run
#: launched WITHOUT the `fdp run` wrapper does not report
#: `ToksearchUnavailable`; it reaches the fetch and fails per signal, with
#: `PtDataError` for every PTDATA point and `TreeFOPENR` for every MDSplus
#: node. Measured on shot 189382: without the wrapper `{'kappa':
#: 'TreeFOPENR', 'ne_zipfit': 'TreeFOPENR', 'ip': 'PtDataError'}`, and with
#: it all three fetch (bt and ip likewise on 190347). Calling those permanent
#: would let ONE mis-launched bulk run write "no fdp source" across the whole
#: corpus - including the 2,192 of 5,000 archive shots that have no `ip`/`bt`
#: column and depend on fdp entirely - and have every later, correctly
#: launched run skip them for good.
#:
#: `TreeFOPENR` is "could not open the tree", a whole-tree condition worth
#: one more open. The genuinely permanent MDSplus case, a node that does not
#: exist (`TreeNNF`), is deliberately NOT here.
#:
#: The accepted cost, stated so it is not rediscovered: where a tree really
#: is unopenable for a shot, that feature is refetched on every run and the
#: shot never reports `ok`. MEASURED in Task 12 - efit01 was `TreeFOPENR` on
#: 1 of 120 random overlap shots (190830). One failed open per run per such
#: shot is the cheaper error: the alternative is unrecoverable without
#: `--force`, which also discards the genuine permanent misses. `OSError`
#: carries the same trade-off for the 1.3-3.7% of truncated corpus files.
TRANSIENT_CAUSES = (
    "TimeoutError",
    "OSError",
    "ConnectionError",
    "StageTimeout",
    "ToksearchUnavailable",
    "PtDataError",
    "TreeFOPENR",
)


def is_transient(cause: str) -> bool:
    """True when a recorded miss is worth retrying on a later run.

    Matched as a substring, which is what makes a cause recorded across
    several sources work: `features_for_shot` joins one feature's causes
    with commas, so a feature the archive genuinely does not carry and fdp
    merely timed out on reads `"archive:KeyError,fdp:TimeoutError"`. That
    counts as transient - one of the two sources is worth another attempt,
    and retrying the pair costs one fetch while not retrying it loses the
    feature permanently.
    """
    return any(t in cause for t in TRANSIENT_CAUSES)


def permanent_names(path) -> set[str]:
    """Names whose recorded miss will fail again identically."""
    return {n for n, c in missing_names(path).items() if not is_transient(c)}


def is_complete(path, names, *, retry_transient: bool = True) -> bool:
    """True when every requested name is stored or permanently missed.

    A transient miss (a timeout, a dropped connection) does not count as
    known: the next run should try it again. Pass `retry_transient=False`
    to treat any recorded miss as final.
    """
    if not Path(path).exists():
        return False
    misses = missing_names(path)
    if retry_transient:
        misses = {n: c for n, c in misses.items() if not is_transient(c)}
    return set(names) <= (present(path) | set(misses))
