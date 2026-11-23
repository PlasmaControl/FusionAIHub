"""The codecs of an assignment that names a recoder (plan 1c).

A setting whose codec is `RECODED` or `RECODED_CONCAT` and whose `source` names a recoder directory
becomes a `recode.RecodedCodec`; every other setting is 1b's. 1b's `ladder.values_per_frame` counts both
names without reading the recoder. `sources_sha256` is the part of a stored score's identity that
says which recoder weights were behind it.

Units: "model units" are the cache's, "front values" are the front codec's, "example units" are front
values after `store.Norm`, and "recoder values" are what a recoder returns; the grids of a recoded
codec are in recoder values. Nothing here imports torch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...tasks import SignalSpec
from ...transform import ChannelStats
from .. import assign, codecs, fit as fit_module
from ..grid import GridSet
from . import recode, store

RECODED = "recoded"                   # a per-stream recoder
RECODED_CONCAT = "recoded_concat"     # a recoder of all streams of a frame at once
_NET_LO, _NET_HI, _NET_WIDTH = -1.0, 1.0, 1000.0     # a network's values lie in (-1, 1); width >> span: even bins


def _grids(codec: recode.RecodedCodec, got: recode.Loaded, name: str, n_bins: int) -> GridSet:
    """The grids of a recoded codec's ``(n_tok, n_val)`` positions, in recoder values.

    A linear recoder's: its stored ranges of the signal, ``(rows, n_values)`` each, with each row's first
    ``tokens_per_stream * n_val`` values regrouped into tokens of `n_val` (`ValueError` if the ranges are
    missing or do not fit the recoder). A network's: the even grid from -1 to 1.
    """
    shape = (codec.n_tok, codec.n_val)
    if got.recoder.kind == "net":
        return GridSet(np.full(shape, _NET_LO), np.full(shape, _NET_HI), np.full(shape, _NET_WIDTH), n_bins)
    if name not in got.ranges:
        raise ValueError(f"the recoder has no stored ranges for signal {name!r}; it has {sorted(got.ranges)}")
    rows = 1 if got.recoder.concat else codec.n_streams
    cut = codec.tokens_per_stream * codec.n_val
    parts = []
    for a in got.ranges[name]:
        a = np.asarray(a, dtype=np.float64)
        if a.shape != (rows, got.recoder.n_values):
            raise ValueError(f"ranges of {name!r} must be ({rows}, {got.recoder.n_values}), got {a.shape}")
        parts.append(a[:, :cut].reshape(shape))
    return GridSet(*parts, n_bins=n_bins)


def build_codec(spec: SignalSpec, stats: ChannelStats, fit: fit_module.SignalFit, setting: assign.Setting,
                recoders_root: Path, loaded: dict[str, recode.Loaded] | None = None,
                device: str = "cpu") -> codecs.Codec:
    """The codec of `setting` for a signal: 1b's `fit.build_codec`, unless `setting.codec` names a recoder.

    `stats` and `fit` are the signal's (1b's), in 1b's units. A recoded setting reads the recoder of
    ``recoders_root / setting.source`` (kept in `loaded` under the source when a dict is given), builds
    the front end from the kind, domain and values per token that the recoder recorded for the signal,
    keeps ``setting.k`` values, and takes grids with ``setting.n_bins`` bins from the stored ranges (the
    even grid from -1 to 1 for a network). `ValueError` names "concat" if the codec name and the
    recoder's `concat` disagree, the signal if the recoder does not list it, "domain" if
    ``setting.domain`` is not the recorded one.
    """
    if setting.codec not in (RECODED, RECODED_CONCAT):
        return fit_module.build_codec(spec, stats, fit, setting)
    if not setting.source:
        raise ValueError(f"{spec.name}: a {setting.codec!r} setting needs a source (the recoder directory)")
    loaded = {} if loaded is None else loaded
    if setting.source not in loaded:
        loaded[setting.source] = recode.load_recoder(Path(recoders_root) / setting.source, device)
    got = loaded[setting.source]
    recoder = got.recoder
    if (setting.codec == RECODED_CONCAT) != recoder.concat:
        raise ValueError(f"{spec.name}: codec {setting.codec!r} does not match recoder {setting.source!r}, "
                         f"whose concat is {recoder.concat}")
    info = got.meta["signals"].get(spec.name)
    if info is None or spec.name not in got.norms:
        raise ValueError(f"recoder {setting.source!r} does not list signal {spec.name!r}; "
                         f"it lists {sorted(got.meta['signals'])}")
    if setting.domain != info["domain"]:
        raise ValueError(f"{spec.name}: setting domain {setting.domain!r} is not the domain {info['domain']!r} "
                         f"that recoder {setting.source!r} was fitted in")
    front = store.front_codec(spec, stats, fit, info["kind"], info["domain"], int(info["values_per_token"]))
    s, _ = store.stream_shape(front)
    if s != int(info["n_streams"]):
        raise ValueError(f"{spec.name}: the front has {s} streams, recoder {setting.source!r} recorded "
                         f"{info['n_streams']}")
    args = (front, got.norms[spec.name], recoder, got.table.index_of(spec.name, s), setting.k)
    bare = recode.RecodedCodec(*args)
    return recode.RecodedCodec(*args, grids=_grids(bare, got, spec.name, setting.n_bins))


def build_codecs(signals: Sequence[SignalSpec], stats: Mapping[str, ChannelStats],
                 fits: Mapping[str, fit_module.SignalFit], assignment: Mapping[str, assign.Setting],
                 recoders_root: Path, device: str = "cpu") -> dict[str, codecs.Codec]:
    """`build_codec` for each of `signals`, with its stats, fit and setting looked up by name.

    Each recoder directory is read once for all the signals that name it.
    """
    loaded: dict[str, recode.Loaded] = {}
    return {s.name: build_codec(s, stats[s.name], fits[s.name], assignment[s.name], recoders_root, loaded, device)
            for s in signals}


def sources(assignment: Mapping[str, assign.Setting]) -> list[str]:
    """The distinct non-empty `source` names of an assignment, sorted."""
    return sorted({s.source for s in assignment.values() if s.source})


def sources_sha256(assignment: Mapping[str, assign.Setting], recoders_root: Path) -> str:
    """Hex sha256 over each source name and its `recode.recoder_sha256`, in that order, sources sorted.

    `FileNotFoundError` for a source with no saved recoder under `recoders_root`.
    """
    h = hashlib.sha256()
    for name in sources(assignment):
        h.update(name.encode())
        h.update(recode.recoder_sha256(Path(recoders_root) / name).encode())
    return h.hexdigest()
