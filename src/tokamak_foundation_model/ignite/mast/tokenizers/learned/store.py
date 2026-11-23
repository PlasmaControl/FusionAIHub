"""The examples of the learned tokenizers: front values as streams, their store and their scale (plan 1c).

The front end of a learned tokenizer is the top rung of a 1b codec (`front_setting`). Its values
``(F, n_tok, n_val)`` are regrouped into streams ``(F, S, D)`` (`to_streams`): the tokens of a
`frame_dct` or `band_dct` codec are already streams of ``D = n_val`` values; a `linear` codec's tokens
are all one stream of ``D = n_tok * n_val`` values. A frame of a stream is one example. Missing data
is NaN and stays NaN; a stream without a token in a frame has NaN there, never 0.

Units. "Model units" are the cache's. "Front values" are what the front codec's ``encode_domain``
returns, in the codec's domain. "Example units" are front values after `Norm`: each stream centred
per value and divided by one scale per stream.

Nothing here imports torch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ... import cache, transform
from ...paths import prepare_write_root
from ...tasks import SignalSpec
from .. import assign, codecs, fit as fit_module
from ..frames import frame_len

D_FRONT = {"frame_dct": 16, "band_dct": 32, "linear": 96}      # values per stream at the top rung
_MAX_LINEAR = 96                                                  # principal-component values of the top rung
_CHUNK = 1 << 16                                                  # examples per pass of `fit_norm`
_STORE_JSON = "store.json"


# ------------------------------------------------------------------------------------- the front end

def front_setting(spec: SignalSpec, kind: str, domain: str, values_per_token: int = 1) -> assign.Setting:
    """The top rung of the fixed codec of `kind` for a signal: the setting of its front end.

    ``frame_dct``: ``min(16, frame_len(dt))`` values per channel in `domain`. ``band_dct``: 32 values in
    each of the 8 bands, domain ``"band"``. ``linear``: the largest multiple of `values_per_token` that
    is at most ``min(96, prod(values_shape))`` principal-component values, in tokens of
    `values_per_token`, in `domain`.
    """
    if kind == "frame_dct":
        return assign.Setting("frame_dct", (min(D_FRONT["frame_dct"], frame_len(spec.dt)),), domain)
    if kind == "band_dct":
        return assign.Setting("band_dct", (D_FRONT["band_dct"],) * 8, "band")
    if kind == "linear":
        top = min(_MAX_LINEAR, int(np.prod(spec.values_shape, dtype=np.int64)))
        n = top // values_per_token * values_per_token
        if n < 1:
            raise ValueError(f"{spec.name}: {top} values cannot fill one token of {values_per_token}")
        return assign.Setting("linear", (n, values_per_token), domain)
    raise ValueError(f"unknown codec kind {kind!r}; one of {assign.CODECS}")


def front_codec(spec: SignalSpec, stats: transform.ChannelStats, fit: fit_module.SignalFit, kind: str,
                domain: str, values_per_token: int = 1) -> codecs.Codec:
    """The 1b codec at the top rung (`front_setting`) built from the signal's `fit` and its `stats`."""
    return fit_module.build_codec(spec, stats, fit, front_setting(spec, kind, domain, values_per_token))


def stream_shape(front: codecs.Codec) -> tuple[int, int]:
    """``(S, D)``, the streams of a frame and the values of each, of a front codec."""
    if isinstance(front, codecs.LinearCodec):
        return 1, front.n_tok * front.n_val
    if isinstance(front, (codecs.FrameDCT, codecs.BandDCT)):
        return front.n_tok, front.n_val
    raise ValueError(f"{type(front).__name__} is not a front end of a learned tokenizer")


def to_streams(front: codecs.Codec, values: np.ndarray) -> np.ndarray:
    """Front values ``(..., n_tok, n_val)`` to streams ``(..., S, D)`` (same units, NaN kept).

    The identity for `FrameDCT` and `BandDCT`; for a `LinearCodec` all tokens become one stream.
    """
    s, d = stream_shape(front)
    values = np.asarray(values)
    if values.shape[-2:] != (front.n_tok, front.n_val):
        raise ValueError(f"front values must end in ({front.n_tok}, {front.n_val}), got {values.shape}")
    return values.reshape(values.shape[:-2] + (s, d))


def from_streams(front: codecs.Codec, streams: np.ndarray) -> np.ndarray:
    """Streams ``(..., S, D)`` back to front values ``(..., n_tok, n_val)`` (the inverse of `to_streams`)."""
    s, d = stream_shape(front)
    streams = np.asarray(streams)
    if streams.shape[-2:] != (s, d):
        raise ValueError(f"streams must end in ({s}, {d}), got {streams.shape}")
    return streams.reshape(streams.shape[:-2] + (front.n_tok, front.n_val))


def stream_values(front: codecs.Codec, z_model: np.ndarray) -> np.ndarray:
    """One shot's cached values to streams: model units ``(*values_shape, T)`` to float32 ``(F, S, D)`` front values.

    NaN for a stream without a token in a frame. A series (`FrameDCT`) keeps only whole frames,
    ``F = T // frame_len(dt)``; a spectrogram or a profile or map keeps every frame the codec makes.
    """
    z = np.asarray(z_model)
    want = tuple(front.values_shape)
    if z.ndim != len(want) + 1 or z.shape[:-1] != want:
        raise ValueError(f"{front.name}: a shot's values must be {want} + (T,), got {z.shape}")
    v = front.encode_domain(front.domain_from_model(z[None]))[0]                  # (F, n_tok, n_val)
    if isinstance(front, codecs.FrameDCT):
        v = v[: z.shape[-1] // frame_len(front.dt)]
    return to_streams(front, v).astype(np.float32)


# ------------------------------------------------------------------------------------- the store

@dataclass
class Store:
    """The examples of one signal over some shots.

    `values` is float32 ``(M, S, D)`` front values, NaN where a stream has no token in the frame;
    `shots` is int64 ``(M,)``, the shot of each frame; `n_skipped` counts the shots without the signal.
    """

    name: str
    values: np.ndarray
    shots: np.ndarray
    n_skipped: int = 0


def _stack(name: str, front: codecs.Codec, per_shot: Iterable[tuple[int, np.ndarray | None]]) -> Store:
    """A `Store` from per-shot stream values: ``(shot, (F, S, D) float32 front values or None)``."""
    s, d = stream_shape(front)
    blocks, shots, skipped = [], [], 0
    for shot, v in per_shot:
        if v is None:
            skipped += 1
            continue
        if v.shape[1:] != (s, d):
            raise ValueError(f"{name}: shot {shot} has streams {v.shape[1:]}, expected {(s, d)}")
        blocks.append(v)
        shots.append(np.full(len(v), shot, dtype=np.int64))
    if blocks:
        return Store(name, np.concatenate(blocks, axis=0), np.concatenate(shots), skipped)
    return Store(name, np.zeros((0, s, d), dtype=np.float32), np.zeros(0, dtype=np.int64), skipped)


def collect(name: str, front: codecs.Codec, per_shot: Iterable[tuple[int, np.ndarray | None]]) -> Store:
    """The store of one signal: `per_shot` yields ``(shot, z_model)``, `z_model` float32 ``(*values_shape, T)``
    model units, or None for a shot without the signal (skipped and counted in `n_skipped`).
    """
    return _stack(name, front, ((shot, None if z is None else stream_values(front, z)) for shot, z in per_shot))


def _shot_streams(job: tuple) -> dict[str, np.ndarray | None]:
    """``(shot, root, stats_sha256, fronts)`` to ``{signal: (F, S, D) float32 front values or None}`` of one shot.

    Module-level so `transform.map_shots` can pickle it for its spawned workers.
    """
    shot, root, sha, fronts = job
    got = cache.read_shot(Path(root) / "shots" / f"{shot}.h5", list(fronts), stats_sha256=sha)
    return {n: None if c is None else stream_values(fronts[n], c.values) for n, c in got.items()}


def build_store(cache_root: Path, shots: Sequence[int], fronts: Mapping[str, codecs.Codec], *,
                workers: int) -> dict[str, Store]:
    """The stores of the signals of `fronts` over `shots`, read from ``cache_root/shots/<shot>.h5``.

    Each file is read with the ``stats_sha256`` of ``cache_root/manifest.json`` (a file made from other
    statistics raises `ValueError`) and encoded in one of `workers` processes. The cached values are
    float32 model units; the stores hold float32 front values. A shot without a signal is counted in
    that store's `n_skipped`.
    """
    root = Path(cache_root)
    sha = json.loads((root / "manifest.json").read_text())["stats_sha256"]
    fronts = dict(fronts)
    per_shot: dict[str, list[tuple[int, np.ndarray | None]]] = {n: [] for n in fronts}
    for shot, got in zip(shots, transform.map_shots(_shot_streams, shots, (str(root), sha, fronts), workers)):
        for n, v in got.items():
            per_shot[n].append((int(shot), v))
    return {n: _stack(n, fronts[n], per_shot[n]) for n in fronts}


def save_store(stores: Mapping[str, Store], out_dir: Path, meta: dict) -> None:
    """Write one ``<signal>.npz`` per store and ``store.json`` (`meta`, which must be JSON, and the shapes).

    `store.json` is written last, so a directory with it is complete. `FileExistsError` if `out_dir`
    already holds one; an existing store is never overwritten.
    """
    out = prepare_write_root(Path(out_dir))
    if (out / _STORE_JSON).exists():
        raise FileExistsError(f"{out / _STORE_JSON} exists; a store is written once")
    info = {}
    for name, s in stores.items():
        tmp = out / f"{name}.npz.tmp"
        with open(tmp, "wb") as f:
            np.savez(f, values=s.values, shots=s.shots, n_skipped=np.int64(s.n_skipped))
        os.replace(tmp, out / f"{name}.npz")
        info[name] = {"shape": list(s.values.shape), "n_skipped": int(s.n_skipped)}
    tmp = out / (_STORE_JSON + ".tmp")
    tmp.write_text(json.dumps({"meta": meta, "signals": info}, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out / _STORE_JSON)


def load_store(out_dir: Path, names: Sequence[str] | None = None) -> tuple[dict[str, Store], dict]:
    """Read what `save_store` wrote: ``({signal: Store}, meta)``; `names` limits the signals (None: all).

    `KeyError` names a signal the store does not hold.
    """
    out = Path(out_dir)
    doc = json.loads((out / _STORE_JSON).read_text())
    names = list(doc["signals"]) if names is None else list(names)
    unknown = [n for n in names if n not in doc["signals"]]
    if unknown:
        raise KeyError(f"{out} does not hold {unknown}; it has {sorted(doc['signals'])}")
    stores = {}
    for n in names:
        with np.load(out / f"{n}.npz", allow_pickle=False) as z:
            stores[n] = Store(n, z["values"], z["shots"], int(z["n_skipped"]))
    return stores, doc["meta"]


# ------------------------------------------------------------------------------------- the scale

@dataclass(frozen=True)
class Norm:
    """Per-stream standardization of front values into example units.

    `center` is ``(S, D)`` (the mean of each stream's value) and `scale` is ``(S,)`` (one scale per
    stream, so the shape of a stream's values over D is kept).
    """

    center: np.ndarray
    scale: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Front values ``(..., S, D)`` to example units ``(values - center) / scale[:, None]``; NaN kept."""
        values = np.asarray(values)
        dt = values.dtype if values.dtype.kind == "f" else np.dtype(np.float32)
        return (values - self.center.astype(dt)) / self.scale.astype(dt)[:, None]

    def invert(self, e: np.ndarray) -> np.ndarray:
        """Example units ``(..., S, D)`` back to front values (the inverse of `apply`); NaN kept."""
        e = np.asarray(e)
        dt = e.dtype if e.dtype.kind == "f" else np.dtype(np.float32)
        return e * self.scale.astype(dt)[:, None] + self.center.astype(dt)


def fit_norm(values: np.ndarray) -> Norm:
    """The `Norm` of a store's front values ``(M, S, D)`` (float32, NaN for no token).

    `center` is the mean over the finite examples of each ``(stream, value)``. `scale` is the root of the
    mean of ``(values - center) ** 2`` over the finite examples and the D values of the stream. A stream
    without a finite example gets `center` 0 and `scale` 1; a stream whose scale is 0 gets `scale` 1.
    Accumulated in float64 over chunks of examples.
    """
    v = np.asarray(values)
    if v.ndim != 3:
        raise ValueError(f"values must be (M, S, D), got {v.shape}")
    _, s, d = v.shape
    total = np.zeros((s, d))
    count = np.zeros((s, d))
    for i in range(0, len(v), _CHUNK):
        x = v[i:i + _CHUNK].astype(np.float64)
        fin = np.isfinite(x)
        total += np.where(fin, x, 0.0).sum(0)
        count += fin.sum(0)
    center = np.divide(total, count, out=np.zeros((s, d)), where=count > 0)
    sq = np.zeros(s)
    for i in range(0, len(v), _CHUNK):
        x = v[i:i + _CHUNK].astype(np.float64)
        sq += np.where(np.isfinite(x), (x - center) ** 2, 0.0).sum((0, 2))
    n = count.sum(1)
    scale = np.sqrt(np.divide(sq, n, out=np.ones(s), where=n > 0))
    scale[~(scale > 0)] = 1.0
    return Norm(center, scale)
