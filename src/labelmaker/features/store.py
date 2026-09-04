"""Reading and writing `<shot>_features.h5`.

Same layout as the corpus itself - one group per quantity, `xdata` seconds,
`ydata` (C, T) - so anything that can read a corpus file can read a feature
file. Files are small (a few MB), so a merge rewrites the whole file and
renames it into place; a killed run therefore never leaves a half-written
feature file behind.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .. import __version__
from ..config import git_sha
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


def _read_group(g) -> FeatureArray:
    """One stored group back into a FeatureArray, in float64."""
    return FeatureArray(
        x=np.asarray(g["xdata"], dtype=np.float64),
        y=np.asarray(g["ydata"], dtype=np.float64),
        attrs={k: str(v) for k, v in g.attrs.items()},
    )


def _load_all(path: Path) -> tuple[dict[str, FeatureArray], dict[str, str]]:
    if not Path(path).exists():
        return {}, {}
    with h5py.File(path, "r") as f:
        arrays = {name: _read_group(f[name]) for name in f}
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
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if merge:
        kept, kept_missing = _load_all(path)
        kept.update(arrays)
        kept_missing.update(missing)
        for name in arrays:
            kept_missing.pop(name, None)
        arrays, missing = kept, kept_missing
    # A group with fewer than two samples is the corpus' "signal absent"
    # sentinel, and these files share the corpus layout. Absence here travels
    # in the `missing` dict instead, so writing a *resolved* one-sample group
    # would read as absent to any consumer applying the corpus rule - which
    # `catalog.available_groups` does. Refuse it: a resolver that got one
    # sample should record a miss, not persist an ambiguity.
    too_short = {n: a.y.shape[-1] for n, a in arrays.items() if a.y.shape[-1] < 2}
    if too_short:
        raise ValueError(
            "refusing to write features with fewer than 2 samples, which the "
            f"corpus layout reads as absent: {too_short}"
        )
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["shot"] = int(shot)
        f.attrs["labelmaker_version"] = __version__
        f.attrs["git_sha"] = git_sha()
        f.attrs["written_at"] = now
        f.attrs[MISSING_ATTR] = json.dumps(missing, sort_keys=True)
        for name, arr in sorted(arrays.items()):
            g = f.create_group(name)
            g.create_dataset("xdata", data=np.asarray(arr.x, dtype=np.float64))
            g.create_dataset("ydata", data=np.asarray(arr.y, dtype=np.float32))
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
    tmp.replace(path)


def read_feature(path, name: str) -> FeatureArray:
    """One feature out of a feature file, or KeyError."""
    with h5py.File(path, "r") as f:
        if name not in f:
            raise KeyError(f"{name} not in {path}")
        return _read_group(f[name])


def present(path) -> set[str]:
    """Feature names stored in the file (empty if the file does not exist)."""
    if not Path(path).exists():
        return set()
    with h5py.File(path, "r") as f:
        return set(f.keys())


def missing_names(path) -> dict[str, str]:
    """Feature name -> cause, for everything a run tried and could not get."""
    if not Path(path).exists():
        return {}
    with h5py.File(path, "r") as f:
        return json.loads(f.attrs.get(MISSING_ATTR, "{}"))


def is_complete(path, names) -> bool:
    """True when every requested name is either stored or a recorded miss."""
    if not Path(path).exists():
        return False
    return set(names) <= (present(path) | set(missing_names(path)))
