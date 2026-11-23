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
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...tasks import SignalSpec
from ...transform import ChannelStats
from .. import assign, codecs, fit as fit_module
from ..grid import GridSet
from . import identity, recode, store

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


def _base_values(setting: assign.Setting, info: Mapping[str, object]) -> int:
    """Values per frame of the base setting of a signal whose front the recoder recorded as `info`.

    ``frame_dct``: ``S * K`` (S the recorded streams, one per channel). ``band_dct``: ``P * sum(k)`` with
    ``P = S / 8`` probes. ``linear``: ``k[0]``. An integer count (no unit).
    """
    kind, s = info["kind"], int(info["n_streams"])
    if kind == "frame_dct":
        return s * int(setting.k[0])
    if kind == "band_dct":
        return s // identity.N_BANDS * int(sum(setting.k))
    return int(setting.k[0])


def recode_assignment(base: Mapping[str, assign.Setting], loaded: recode.Loaded, source: str, *,
                      scale: float = 1.0, signals: Sequence[str] | None = None) -> dict[str, assign.Setting]:
    """`base` with the signals of a recoder (or the listed ones among them) behind that recoder.

    `loaded` is the recoder read from ``<recoders>/<source>``. Each chosen signal gets
    ``Setting(RECODED or RECODED_CONCAT, k, base domain, base n_bins, source)``. A per-stream recoder keeps
    what the base kept per stream: ``(K,)`` of a `frame_dct`, the eight numbers of a `band_dct`,
    ``(n_values,)`` of a `linear`. A concatenated recoder keeps ``k = (m,)``: `m` is the base's values per
    frame for the signal (S times K; P times ``sum(k)``) times `scale`, rounded down to a multiple of the
    recoder's `n_val` and at least one token. Signals not chosen keep their base setting.

    `ValueError` naming "scale" for a per-stream recoder with ``scale != 1``; "domain" if the base domain is
    not the one the recoder was fitted in; "recoded" if the base setting already names a recoder; "holds" if
    the recoder has fewer values than `k` asks for; "codec" if the base is another kind of codec than the
    recoder's front, or has no setting for a signal; "multiple" if a multi-token recoder is asked for a
    number of values that is not a whole number of tokens.
    """
    r = loaded.recoder
    concat = bool(r.concat)
    if not scale > 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if scale != 1.0 and not concat:
        raise ValueError(f"scale {scale} cannot apply to {source!r}: a per-stream recoder keeps the base's values "
                         "per stream, and only a concatenated recoder is sized by a scale")
    out = dict(base)
    for name, info in loaded.meta["signals"].items():
        if signals is not None and name not in signals:
            continue
        if name not in base:
            raise ValueError(f"codec: the base assignment has no setting for {name!r}, which recoder {source!r} lists")
        b = base[name]
        if b.codec in (RECODED, RECODED_CONCAT):
            raise ValueError(f"{name}: the base setting is already recoded (by {b.source!r}); a signal has one recoder")
        if b.domain != info["domain"]:
            raise ValueError(f"{name}: the base domain {b.domain!r} is not the domain {info['domain']!r} "
                             f"that recoder {source!r} was fitted in")
        if b.codec != info["kind"]:
            raise ValueError(f"{name}: the base codec is {b.codec!r} but recoder {source!r} was fitted behind a "
                             f"{info['kind']!r} codec")
        if concat:
            values = _base_values(b, info)
            m = max(r.n_val, int(math.floor(values * scale + 1e-9)) // r.n_val * r.n_val)
            k = (m,)
        else:
            k = tuple(b.k[:1]) if info["kind"] == "linear" else tuple(b.k)
            if r.n_tok > 1 and (len(set(k)) != 1 or k[0] % r.n_val):
                raise ValueError(f"{name}: recoder {source!r} has tokens of {r.n_val} values, so it needs one "
                                 f"number of values that is a multiple of {r.n_val}, got {list(k)}")
        if max(k) > r.n_values:
            raise ValueError(f"{name}: K {list(k)} asks for more values than recoder {source!r} holds ({r.n_values})")
        out[name] = assign.Setting(RECODED_CONCAT if concat else RECODED, k, b.domain, b.n_bins, source)
    return out
