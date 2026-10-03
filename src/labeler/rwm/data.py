"""Read a shot's stored signals and turn them into labelled slice tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Paths
from ..events import equilibrium
from ..events.verify import NoDataError
from ..features import namespace
from . import features, labels

#: What a shot's file must hold for its slices to be usable at all.
REQUIRED = ("ip", "betan", "li", "n1rms")
OPTIONAL = ("q95", "qmin", "wmhd", "n2rms", "rot_zipfit", "dusbradial")


def load_signals(shot: int, paths: Paths) -> dict:
    """`{name: (t_ms, y)}` from the features, corpus or raw cache; never fetches.

    A name with no usable record is left out of the result. A shot missing any of
    `REQUIRED` raises `NoDataError`.
    """
    out: dict = {}
    for name in REQUIRED + OPTIONAL:
        try:
            array = equilibrium.signal(int(shot), name, paths)
        except (NoDataError, KeyError, OSError, ValueError):
            if name in REQUIRED:
                raise NoDataError(f"shot {shot}: no {name}") from None
            continue
        y = np.asarray(array.y, dtype=float)
        out[name] = (
            np.asarray(array.x, dtype=float) * 1000.0,
            y if y.shape[0] > 1 else y[0],
        )
    if "rot_zipfit" in out:
        out["rho_grid"] = namespace.RHO_GRID
    return out


def shot_table(
    shot: int,
    paths: Paths,
    onsets_ms=(),
    *,
    other_onsets_ms=(),
    hanson: bool = False,
    step_ms: float = features.STEP_MS,
) -> pd.DataFrame:
    """Trailing calculations on one shot's offline inputs, with forecast labels.

    `label` is `labels.slice_labels` over the listed onsets, with negatives assumed
    absent on Hanson shots. Complete reviewed coverage has not been established.
    On a comparison shot every slice is `UNLABELLED` (-2).
    `other_onsets_ms` (the n = 2 onsets, for an n = 1 target) leave
    their slices out of both classes.
    """
    signals = load_signals(shot, paths)
    table = features.slice_table(signals, step_ms=step_ms)
    table.insert(0, "shot", int(shot))
    if hanson:
        table["label"] = labels.slice_labels(
            table.t_ms.to_numpy(), onsets_ms, other_onsets_ms=other_onsets_ms
        )
    else:
        table["label"] = np.int8(labels.UNLABELLED)
    return table
