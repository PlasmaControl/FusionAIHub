"""Features from the 25 ms archive store.

`/projects/EKOLEMEN/profile_predictor/DATA/new_h5_data/example_191450_183224.h5`
holds 5,000 shots (183224..191450) as one group per shot, 209 columns each,
scalars `(240,)` and profiles `(240, 33)` on an implicit 25 ms grid. It is
the store the Phase 1 model's training features were built from: twelve of
its columns are bit-identical to that model's training inputs (see the
plan's Deviation 6), which is why this resolver comes first in every
feature's `sources`.

It covers only 19% of the corpus (3,246 of 16,909 shots - the intersection,
not the 3,621 corpus shots that merely fall inside its shot-number span), so
it is the
proof-of-concept path, not the scaling path. `resolve_fdp` is the latter.

Availability varies per shot: some groups are missing individual columns, so
a missing column is recorded as a per-feature miss rather than raised.

OPEN, and MEASURED exactly - the time axis this module assigns is one step
late. `resolve` stamps row k at `STEP_S * k`, but row k is the mean of the
raw signal over the 50 ms interval `[25*(k-2), 25*k]` ms, i.e. a 50 ms
boxcar CENTRED on `25*(k-1)` ms. Established against PTDATA `ip` over 8
random overlap shots by scanning offsets from -50 to +20 ms against nearest
-sample and 12.5/25/50 ms window means: the 50 ms window at -25 ms gives a
median relative error of 3.9e-08 - float32 round-trip precision - while
every other combination is 1e-4 or worse, and nearest-sample at zero offset
(what a naive reading assumes) is 5.8e-03.

Two consequences, which is why this is documented rather than silently
corrected:

1. For a pure-archive build the VALUES are right. The model's training rows
   came from these same rows - matching x0.npy on the five bit-identical
   columns resolves 9,805 of 9,805 rows uniquely - and upstream's own t and
   t+dt were row offsets too, so the lag structure the model sees is the one
   it trained on. Only the timestamps written onto the labels are 25 ms late.
2. In a MIXED build it is a real misalignment. A feature served here sits
   25 ms - one whole `dt` - away from the same nominal time served by the
   corpus or fdp resolvers, which do carry true time axes. That is the
   scaling path, so it has to be settled before archive and non-archive
   features are combined in one row.

Correcting it would move every archive-derived output, including the
measurements the model card and the ECH validity rule already quote, so
Task 15's fix is deliberately NOT here: `resolve_archive`'s own axis is
untouched. Instead, `models.base.InputSpec.build` reconciles the offset at
the point model inputs are assembled - a corpus- or fdp-served field is
windowed into the archive's own 50 ms boxcar ending at `t`
(`ARCHIVE_WINDOW_S`, keyed on resolver via `ns.SAMPLING_BY_SOURCE`), so an
archive-served field and a non-archive one in the same row refer to the
same physical interval once `build()` is done with them, without moving any
number this module or the model card already publishes. See
`models.base.ARCHIVE_WINDOW_S`'s docstring and the Task 15 report for the
measurement that this convention wins by one to two orders of magnitude
over every alternative tried. It also explains a set of inflated agreement
figures that were in `namespace.py` before Task 12 re-measured them (r0
8.8e-3 -> 7.9e-4, kappa 3.1e-3 -> 1.2e-3): they were taken without the lag
correction.
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
                if raw.shape[1] != ns.RHO_GRID.size:
                    # The store's convention is (T, n_rho). A column stored
                    # time-first would transpose into something whose x and y
                    # lengths still agree, so nothing downstream would catch
                    # it until `build` failed on a broadcast, far from here.
                    missing[spec.name] = (
                        f"RadialAxisMismatch({raw.shape[1]}!={ns.RHO_GRID.size})"
                    )
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
