"""Features from the 25 ms archive store.

`/projects/EKOLEMEN/profile_predictor/DATA/new_h5_data/example_191450_183224.h5`
holds 5,000 shots (183224..191450) as one group per shot, 209 columns each,
scalars `(240,)` and profiles `(240, 33)` on an implicit 25 ms grid. It is
the store the Phase 1 model's training features were built from: twelve of
its columns are bit-identical to that model's training inputs (see the
plan's Deviation 6), which is why this resolver comes first in every
feature's `sources`.

It covers only 21% of the corpus (3,621 of 16,909 shots), so it is the
proof-of-concept path, not the scaling path. `resolve_fdp` is the latter.

Availability varies per shot: some groups are missing individual columns, so
a missing column is recorded as a per-feature miss rather than raised.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

from . import namespace as ns
from .store import FeatureArray

SOURCE = "archive"

ARCHIVE_FILES: tuple[Path, ...] = (
    Path(
        "/projects/EKOLEMEN/profile_predictor/DATA/new_h5_data/"
        "example_191450_183224.h5"
    ),
)


@lru_cache(maxsize=4)
def shot_index(files: tuple[Path, ...] = ARCHIVE_FILES) -> dict[int, Path]:
    """shot -> the archive file holding it.

    Metadata only: listing the group names of a 5.8 GB file reads no arrays.
    Cached because a bulk run asks for it once per worker.
    """
    index: dict[int, Path] = {}
    for path in files:
        if not Path(path).exists():
            continue
        with h5py.File(path, "r") as f:
            for key in f:
                if key.isdigit():
                    index.setdefault(int(key), Path(path))
    return index


def resolve(
    shot: int,
    names: Sequence[str],
    *,
    files: tuple[Path, ...] = ARCHIVE_FILES,
) -> tuple[dict[str, FeatureArray], dict[str, str]]:
    """Read the requested canonical features for one shot.

    Returns `(arrays, missing)`; `missing` maps a feature name to a short
    cause so a run records why a shot is incomplete instead of failing.
    """
    specs = [ns.by_name(n) for n in names]
    for spec in specs:
        spec.locator_for(SOURCE)  # KeyError names the feature and the source
    path = shot_index(tuple(files)).get(int(shot))
    if path is None:
        return {}, {n: "ShotNotInArchive" for n in names}
    arrays: dict[str, FeatureArray] = {}
    missing: dict[str, str] = {}
    with h5py.File(path, "r") as f:
        group = f[str(int(shot))]
        for spec in specs:
            locator = spec.locator_for(SOURCE)
            if locator not in group:
                missing[spec.name] = "KeyError"
                continue
            raw = np.asarray(group[locator], dtype=np.float64)
            raw = np.squeeze(raw)
            if spec.kind == "profile":
                if raw.ndim != 2:
                    missing[spec.name] = "ShapeError"
                    continue
                y = raw.T                      # (n_rho, T)
                n = y.shape[1]
            else:
                if raw.ndim != 1:
                    missing[spec.name] = "ShapeError"
                    continue
                y = raw[None, :]
                n = y.shape[1]
            arrays[spec.name] = FeatureArray(
                x=ns.STEP_S * np.arange(n, dtype=np.float64),
                y=y,
                attrs={
                    "resolver": SOURCE,
                    "locator": locator,
                    "archive_file": str(path),
                    "n_rows": str(n),
                },
            )
    return arrays, missing
