"""The one feature labelmaker computes for itself.

Every other resolver fetches: the archive store, the corpus, MDSplus. This
one reads two files the package wrote earlier in the same pipeline -
`masks/<shot>_masks.npz` from the U-Net run and
`events/<shot>_events.parquet` from the detectors and heuristics - and
reduces them with `events.windows` into `phenomenon_window_features`, the
46 diagnostics-only numbers a phenomenon prior scores and a GBDT is trained
on.

Two consequences of it being derived rather than fetched:

* `available()` takes the shot, unlike `resolve_fdp.available()`, which
  asks only whether toksearch imports. Whether this source can serve a shot
  is a per-shot fact - the mask job is a 2,000-shot SLURM array and reaches
  shots one at a time - so the question cannot be answered for the source
  as a whole;
* a missing file is a per-shot MISS, not a raise. A shot the mask job has
  not reached yet is the normal case for most of a campaign, and
  `run.features_for_shot` records `FileNotFoundError` against the feature
  and moves on, exactly as it does for a corpus file that is not there.

**The validity mask is expressed as NaN**, in all 46 channels of a window
no diagnostic covered at least half of. A resolver cannot narrow
`BuiltInputs.valid` - it does not see it, and `models.base` builds that
mask a stage later - so what it can do is the same thing `resolve_corpus`
does for an all-NaN channel sum: refuse to fabricate a zero. The AE adapter
is the precedent at the other end, narrowing `built.valid` from its own
frame counts inside `predict`; the GBDT runner narrows it from these NaNs
the same way. A zero here would be a claim - "nothing coherent, no ELMs,
no tracks" - about a window nobody looked at.

The timebase is SECONDS, like every other `FeatureArray` in the package
(`store.FeatureArray`: "seconds on `x`") and like `labels/store.py`'s
`xdata`. See the task L8 report: the brief asked for milliseconds, and
milliseconds here would be a silent factor of 1,000 wherever a window
centre is compared against `namespace.GRID_S` or a label's own time axis.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..events import windows
from . import namespace as ns
from .store import FeatureArray

SOURCE = "events"

#: The one canonical feature this source serves.
NAME = "phenomenon_window_features"


def available(paths, shot: int) -> bool:
    """True when both files this resolver reduces exist for the shot."""
    return (
        Path(paths.masks_file(int(shot))).exists()
        and Path(paths.events_file(int(shot))).exists()
    )


def resolve(
    shot: int,
    names: Sequence[str],
    *,
    paths,
) -> tuple[dict[str, FeatureArray], dict[str, str]]:
    """Reduce one shot's masks and events into the requested features.

    Returns `(arrays, missing)` like every other resolver; `missing` maps a
    feature name to a short cause so a run records why a shot is incomplete
    instead of failing.
    """
    # KeyError here names the feature and the source, and is raised before
    # any file is touched: asking the events source for `ip` is a
    # programming error, not a per-shot gap.
    locators = {n: ns.by_name(n).locator_for(SOURCE) for n in names}
    if not available(paths, shot):
        return {}, dict.fromkeys(names, "FileNotFoundError")
    try:
        centres, x, valid = windows.shot_window_features(shot, paths)
    except FileNotFoundError:
        # Raced with a run that removed a file between the check and here.
        return {}, dict.fromkeys(names, "FileNotFoundError")
    if centres.size == 0:
        # A masks file with no mhr/co2/ece block, or blocks spanning no
        # time: there is no window grid to put anything on.
        return {}, dict.fromkeys(names, f"NoWindows({centres.size})")
    y = np.array(x, dtype=np.float32, copy=True)
    y[:, ~valid] = np.nan
    masks_file = Path(paths.masks_file(int(shot)))
    events_file = Path(paths.events_file(int(shot)))
    arrays = {
        name: FeatureArray(
            x=np.asarray(centres, dtype=np.float64),
            y=y,
            attrs={
                "resolver": SOURCE,
                "locator": locators[name],
                "masks_file": str(masks_file),
                "events_file": str(events_file),
                "n_channels": str(windows.N_FEATURES),
                "n_windows": str(int(centres.size)),
                "n_valid_windows": str(int(valid.sum())),
                "window_s": str(windows.WINDOW_S),
                "stride_s": str(windows.STRIDE_S),
            },
        )
        for name in names
    }
    return arrays, {}
