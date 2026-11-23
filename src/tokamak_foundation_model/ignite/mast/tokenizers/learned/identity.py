"""What a stream is, as a feature vector the recoders can condition on (plan 1c).

A stream is one token position of a signal: a channel of a `frame_dct` signal, a (probe, band) of a
`band_dct` spectrogram, the whole profile or map of a `linear` signal. The features come from the
cache's registry (family, sampling interval, compression, unit, place in the array, position of the
probe), so a recoder trained on many signals can tell its streams apart, and a stream it has never
seen still has features. Everything is float32; a missing number is 0 with its flag column 0.

Columns of a feature vector (``N_FEAT = 18``):

====== =========================================================================================
0 - 3  family as one-hot, in the order slow, high_rate, profile, map
4      ``log10(dt) + 3``, `dt` the sampling interval in seconds
5      1 if the signal's ``compress`` is ``"asinh"``
6 - 10 unit as one-hot over `UNITS`
11     1 if the unit is none of `UNITS`
12     the stream's place in its array, ``i / (n - 1)`` over the channels (probes for ``band_dct``);
       0 for a single one
13     1 if the channel's ``r`` and ``z`` are both finite
14, 15 ``r`` and ``z`` (0 if not finite)
16     1 for ``band_dct``
17     the band index divided by `N_BANDS`
====== =========================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..codecs import BAND_BINS

N_FEAT = 18
UNITS = ("T", "V", "A", "Wb", "W")
N_BANDS = 8                                             # bands of a spectrogram (256 bins of 32)
FAMILIES = ("slow", "high_rate", "profile", "map")
KINDS = ("frame_dct", "band_dct", "linear")


def _base(rows: pd.DataFrame) -> np.ndarray:
    """Columns 0 to 11 of each registry row: `rows` (n rows) to float32 ``(n, 12)``, family to unit."""
    n = len(rows)
    out = np.zeros((n, 12), dtype=np.float32)
    family = rows["family"].to_numpy()
    for j, name in enumerate(FAMILIES):
        out[:, j] = family == name
    unknown = sorted(set(family) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown family {unknown}; one of {FAMILIES}")
    dt = rows["dt"].to_numpy(dtype=np.float64)
    if not (np.isfinite(dt).all() and (dt > 0).all()):
        raise ValueError("the sampling interval dt must be positive and finite in every row")
    out[:, 4] = np.log10(dt) + 3.0
    out[:, 5] = rows["compress"].to_numpy() == "asinh"
    units = rows["units"].to_numpy()
    for j, name in enumerate(UNITS):
        out[:, 6 + j] = units == name
    out[:, 11] = ~np.isin(units, UNITS)
    return out


def _position(rows: pd.DataFrame) -> np.ndarray:
    """Columns 13 to 15 of each registry row: `rows` (n rows) to float32 ``(n, 3)``: flag, ``r``, ``z`` (0 if not finite)."""
    r = rows["r"].to_numpy(dtype=np.float64)
    z = rows["z"].to_numpy(dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(z)
    return np.stack([ok, np.where(ok, r, 0.0), np.where(ok, z, 0.0)], axis=1).astype(np.float32)


def _place(i: np.ndarray, n: int) -> np.ndarray:
    """Place ``i / (n - 1)`` of index `i` (int array) of `n` in an array, 0 for a single one: float32 like `i`."""
    return (i / (n - 1) if n > 1 else np.zeros(len(i))).astype(np.float32)


def stream_features(rows: pd.DataFrame, kind: str, n_streams: int) -> np.ndarray:
    """The registry rows of one signal to the features of its streams: float32 ``(n_streams, N_FEAT)``.

    `rows` are the registry rows of the signal (any order; sorted by ``channel`` here); `kind` is the
    signal's 1b codec kind. For ``frame_dct`` a stream is a channel: `n_streams` must equal the number
    of rows. For ``band_dct`` the rows are probes times bins and a stream is (probe, band): `n_streams`
    is ``P * N_BANDS`` and the stream ``p * N_BANDS + b`` takes the probe's row of its first bin. For
    ``linear`` there is one stream and columns 12 to 15 are 0. Another `n_streams` raises `ValueError`.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown codec kind {kind!r}; one of {KINDS}")
    rows = rows.sort_values("channel", kind="stable")
    n_rows = len(rows)
    if n_rows == 0:
        raise ValueError("a signal needs at least one registry row")
    out = np.zeros((n_streams, N_FEAT), dtype=np.float32)
    if kind == "frame_dct":
        if n_streams != n_rows:
            raise ValueError(f"a frame_dct signal has one stream per channel: {n_rows} rows, n_streams={n_streams}")
        out[:, :12] = _base(rows)
        out[:, 12] = _place(np.arange(n_rows), n_rows)
        out[:, 13:16] = _position(rows)
    elif kind == "band_dct":
        per_probe = N_BANDS * BAND_BINS
        if n_streams % N_BANDS or n_rows != n_streams // N_BANDS * per_probe:
            raise ValueError(f"a band_dct signal has probes x {per_probe} rows and probes x {N_BANDS} streams: "
                             f"{n_rows} rows, n_streams={n_streams}")
        n_probes = n_streams // N_BANDS
        probe, band = np.divmod(np.arange(n_streams), N_BANDS)
        first = rows.iloc[probe * per_probe + band * BAND_BINS]          # the stream's first bin
        out[:, :12] = _base(first)
        out[:, 12] = _place(probe, n_probes)
        out[:, 13:16] = _position(rows.iloc[probe * per_probe])          # the probe's position
        out[:, 16] = 1.0
        out[:, 17] = band / N_BANDS
    else:
        if n_streams != 1:
            raise ValueError(f"a linear signal is one stream, n_streams={n_streams}")
        out[:, :12] = _base(rows.iloc[:1])
    return out


@dataclass(frozen=True)
class StreamTable:
    """The streams of several signals in one table.

    `signals` maps a signal name to ``(first row, number of streams)`` and `features` is float32
    ``(n, N_FEAT)``, one row per stream, in the order of the signals.
    """

    signals: dict[str, tuple[int, int]]
    features: np.ndarray

    def __len__(self) -> int:
        """Number of streams ``n``."""
        return int(self.features.shape[0])

    def index_of(self, name: str, n_streams: int) -> np.ndarray:
        """Rows of the signal `name` in the table: int64 ``(n_streams,)``; all -1 if the table does not hold it.

        `ValueError` if it holds the signal with another number of streams than `n_streams`.
        """
        if name not in self.signals:
            return np.full(n_streams, -1, dtype=np.int64)
        first, n = self.signals[name]
        if n != n_streams:
            raise ValueError(f"the table holds {name!r} with {n} streams, not {n_streams}")
        return np.arange(first, first + n, dtype=np.int64)


def build_table(registry: pd.DataFrame, signals: Sequence[tuple[str, str, int]]) -> StreamTable:
    """The table of `signals`, ``(name, kind, n_streams)`` in order, from the registry rows of each.

    `ValueError` names a signal without registry rows (or one listed twice).
    """
    index: dict[str, tuple[int, int]] = {}
    blocks = []
    start = 0
    for name, kind, n in signals:
        if name in index:
            raise ValueError(f"signal {name!r} is listed twice")
        rows = registry[registry["signal"] == name]
        if rows.empty:
            raise ValueError(f"the registry has no rows for signal {name!r}")
        blocks.append(stream_features(rows, kind, n))
        index[name] = (start, n)
        start += n
    features = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, N_FEAT), dtype=np.float32)
    return StreamTable(index, features)
