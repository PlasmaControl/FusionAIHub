"""A recoder behind the top rung of a fixed codec, and the codec that makes of the two one codec (plan 1c).

A *recoder* maps examples (the streams of `store`, in example units) to a few ordered values and back.
Its values are ordered, so a prefix with the rest set to 0 decodes to a coarser example. A *per-stream*
recoder maps one stream's ``D`` values; a *concatenated* one maps all ``S`` streams of a frame at once
(``d_in = S * D``). `LinearRecoder` is a rotation fitted by principal components; a network
(`nets.NetRecoder`, task 5) is the other kind.

`RecodedCodec` puts a recoder behind a 1b front codec (`store.front_codec`) and is itself a 1b `Codec`:
1b's floor, ladder, gate and instruments score it unchanged. A stream without a token in a frame has no
example and gets NaN values; a missing stream is never turned into data, except by a concatenated
recoder, which fills it with 0 in example units (the stream's mean) so that the other streams keep
their tokens. A frame in which every stream is missing is NaN for both.

Units. "Model units" are the cache's, "front values" are what the front codec's ``encode_domain``
returns, "example units" are front values after `store.Norm`, "recoder values" are what a recoder
returns (``encode``). A recoder is called on finite examples only; rows are put back in place after.

Nothing here imports torch; `load_recoder` imports `nets` only for a recoder of kind ``"net"``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...paths import prepare_write_root
from .. import codecs
from ..codecs import Codec
from ..grid import GridSet
from . import store
from .identity import StreamTable
from .store import Norm

RECODER_JSON = "recoder.json"
_FILES = ("basis.npz", "best.pt", "norm.npz", "ranges.npz", "recoder.json", "table.npz")   # the order of the hash


# ------------------------------------------------------------------------------------- the recoders

class Recoder:
    """The base of the recoders. A subclass sets the attributes and supplies `encode` and `decode`.

    Attributes: `kind` ``"linear"`` or ``"net"``; `concat` whether it takes all streams of a frame at
    once; `d_in` the values of one example (``D`` per-stream, ``S * D`` concatenated); `n_tok` and
    `n_val` the tokens of the recoder values and the values of each, so ``n_values = n_tok * n_val``.
    """

    kind: str
    concat: bool
    d_in: int
    n_tok: int
    n_val: int

    @property
    def n_values(self) -> int:
        """Recoder values per example, ``n_tok * n_val``."""
        return self.n_tok * self.n_val

    def encode(self, e: np.ndarray, ident: np.ndarray) -> np.ndarray:
        """Examples `e` ``(M, d_in)`` finite, example units, to recoder values ``(M, n_values)`` float32.

        `ident` is int64 ``(M,)``, each example's row in the recoder's `StreamTable` (-1: no identity).
        """
        raise NotImplementedError

    def decode(self, values: np.ndarray, ident: np.ndarray) -> np.ndarray:
        """Recoder values ``(M, n_values)`` to examples ``(M, d_in)``, example units (`ident` as in `encode`)."""
        raise NotImplementedError


class LinearRecoder(Recoder):
    """A rotation onto fitted directions: ``values = (e - mean) @ basis``, ``e = values @ basis.T + mean``.

    `mean` is ``(d_in,)`` and `basis` ``(d_in, n_values)`` with orthonormal columns in order of falling
    variance (example units), so a prefix of the values with the rest 0 is the best low-rank
    description. `n_val` None: one token of ``n_values`` values; otherwise `n_values` is cut into tokens
    of `n_val` values (`ValueError` if it is not a multiple). `ident` is not used.
    """

    kind = "linear"

    def __init__(self, mean: np.ndarray, basis: np.ndarray, *, concat: bool = False,
                 n_val: int | None = None) -> None:
        mean = np.asarray(mean, dtype=np.float64)
        basis = np.asarray(basis, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[1] < 1 or mean.shape != (basis.shape[0],):
            raise ValueError(f"mean must be (d_in,) and basis (d_in, n_values >= 1), got {mean.shape} and {basis.shape}")
        n_values = basis.shape[1]
        n_val = n_values if n_val is None else int(n_val)
        if n_val < 1 or n_values % n_val:
            raise ValueError(f"n_values={n_values} must be a multiple of n_val={n_val}")
        self.mean = mean
        self.basis = basis
        self.concat = bool(concat)
        self.d_in = int(basis.shape[0])
        self.n_tok = n_values // n_val
        self.n_val = n_val

    def encode(self, e: np.ndarray, ident: np.ndarray) -> np.ndarray:
        """Examples ``(M, d_in)``, example units, to values ``(M, n_values)`` float32; `ident` is not used."""
        return ((np.asarray(e, dtype=np.float64) - self.mean) @ self.basis).astype(np.float32)

    def decode(self, values: np.ndarray, ident: np.ndarray) -> np.ndarray:
        """Values ``(M, n_values)`` to examples ``(M, d_in)``, example units, float64; `ident` is not used."""
        return np.asarray(values, dtype=np.float64) @ self.basis.T + self.mean


def fit_linear(e: np.ndarray, n_values: int, *, concat: bool = False, n_val: int | None = None) -> LinearRecoder:
    """The `LinearRecoder` of the first `n_values` principal components of `e` ``(N, d_in)``, example units.

    NaN entries of `e` (a concatenated example has them when a stream is missing) are used pairwise
    (`codecs.fit_pca`). `ValueError` if `n_values` is not a multiple of `n_val` or is more than `d_in`.
    """
    e = np.asarray(e)
    if e.ndim != 2:
        raise ValueError(f"e must be (N, d_in), got {e.shape}")
    if n_val is not None and (n_val < 1 or n_values % n_val):
        raise ValueError(f"n_values={n_values} must be a multiple of n_val={n_val}")
    if not 1 <= n_values <= e.shape[1]:
        raise ValueError(f"n_values must be in 1..{e.shape[1]} (d_in), got {n_values}")
    mean, rows, _, _ = codecs.fit_pca(e, n_values)                 # rows (n_values, d_in)
    return LinearRecoder(mean, rows.T, concat=concat, n_val=n_val)


# ------------------------------------------------------------------------------------- the codec

class RecodedCodec(Codec):
    """A 1b `Codec` whose values are a recoder's, behind the top rung of a front codec.

    `front` is the 1b codec at the top rung (`store.front_codec`); `norm` takes its front values to
    example units; `recoder` takes examples to recoder values; `ident` is int64 ``(S,)``, the streams'
    rows in the recoder's table (a concatenated recoder is called with -1). `name`, `dt`,
    `values_shape`, `stats`, `domain`, `signal_mean`, `signal_std` and the three domain functions are
    the front's.

    `n_keep` is the number of recoder values kept: one number, or for a band front with a per-stream
    recoder one per band (stream ``s`` keeps ``n_keep[s % n_bands]``; one number is used for every
    band). With a one-token recoder the layout is ``n_val = max(n_keep)`` and a stream's values from
    its ``n_keep`` on are 0 (1b's convention for bands); with a multi-token recoder `n_keep` is one
    number and a multiple of ``recoder.n_val`` (`ValueError`), ``n_val = recoder.n_val`` and a stream
    keeps ``n_keep // n_val`` tokens. ``n_tok`` is S times the tokens of a stream per-stream, and the
    tokens of the one example concatenated. `k` is the public tuple of what is kept (1b's gate reads
    it for a codec of domain ``"band"``): ``n_keep`` per band for a band front with a per-stream
    recoder, else ``tuple(n_keep)``. `values_per_frame` is the number of kept values of a frame over
    the whole signal.
    """

    def __init__(self, front: Codec, norm: Norm, recoder: Recoder, ident: np.ndarray, n_keep: Sequence[int],
                 grids: GridSet | None = None) -> None:
        s, d = store.stream_shape(front)
        concat = bool(recoder.concat)
        if recoder.d_in != (s * d if concat else d):
            raise ValueError(f"{front.name}: the recoder takes {recoder.d_in} values per example, but the front has "
                             f"{s} streams of {d} values ({'concatenated' if concat else 'per stream'})")
        ident = np.asarray(ident, dtype=np.int64)
        if ident.shape != (s,):
            raise ValueError(f"ident must be ({s},), one row per stream, got {ident.shape}")
        if norm.center.shape != (s, d) or norm.scale.shape != (s,):
            raise ValueError(f"norm must be center ({s}, {d}) and scale ({s},), got {norm.center.shape} and {norm.scale.shape}")
        keep = [int(v) for v in n_keep]
        per_band = isinstance(front, codecs.BandDCT) and not concat
        if per_band:
            if len(keep) == 1:
                keep = keep * front.n_bands
            if len(keep) != front.n_bands:
                raise ValueError(f"{front.name}: n_keep needs one number, or one for each of the {front.n_bands} "
                                 f"bands, got {list(n_keep)}")
        elif len(keep) != 1:
            raise ValueError(f"{front.name}: n_keep is one number, got {list(n_keep)}")
        if min(keep) < 0 or max(keep) < 1 or max(keep) > recoder.n_values:
            raise ValueError(f"{front.name}: n_keep must be in 0..{recoder.n_values} (the recoder's values) with at "
                             f"least one above 0, got {keep}")
        if recoder.n_tok == 1:
            t, n_val = 1, max(keep)
        else:
            if len(set(keep)) != 1 or keep[0] % recoder.n_val:
                raise ValueError(f"{front.name}: n_keep must be one number that is a multiple of the recoder's "
                                 f"n_val={recoder.n_val}, got {keep}")
            t, n_val = keep[0] // recoder.n_val, recoder.n_val
        n_tok = t if concat else s * t
        if grids is not None and (grids.n_tok, grids.n_val) != (n_tok, n_val):
            raise ValueError(f"grids must be ({n_tok}, {n_val}), got ({grids.n_tok}, {grids.n_val})")
        super().__init__(front.name, front.dt, front.values_shape, front.stats, n_tok, n_val, front.domain, grids,
                         front.signal_mean, front.signal_std)
        self.front = front
        self.norm = norm
        self.recoder = recoder
        self.ident = ident
        self.k = tuple(keep)
        self.n_keep = self.k
        self.tokens_per_stream = t
        self.n_streams = s
        self._used = t * n_val                                       # recoder values fed to or from the codec
        stream_keep = np.tile(keep, s // len(keep)) if per_band else np.full(s, keep[0])
        self._keep = np.arange(n_val)[None, :] < stream_keep[:, None] if recoder.n_tok == 1 and not concat \
            else np.ones((n_tok, n_val), dtype=bool)                 # (n_tok, n_val)
        self._uneven = not bool(self._keep.all())
        self._values_per_frame = keep[0] if concat else int(stream_keep.sum())

    @property
    def values_per_frame(self) -> int:
        """Kept values per frame over the whole signal, an integer count (no unit)."""
        return self._values_per_frame

    # -- domain: the front's
    def to_domain(self, x: np.ndarray) -> np.ndarray:
        """Physical ``(..., *values_shape, T)`` to the front's domain."""
        return self.front.to_domain(x)

    def from_domain(self, u: np.ndarray) -> np.ndarray:
        """The front's domain to physical ``(..., *values_shape, T)``."""
        return self.front.from_domain(u)

    def domain_from_model(self, z: np.ndarray) -> np.ndarray:
        """Model units ``(..., *values_shape, T)`` to the front's domain."""
        return self.front.domain_from_model(z)

    # -- values
    def read(self, values: np.ndarray, mode: str) -> np.ndarray:
        """The grid read of `mode` on recoder values ``(N, F, n_tok, n_val)``, same shape.

        When streams keep different numbers of values the positions from a stream's `n_keep` on are set
        back to 0 after the read, as `BandDCT.read` does: a quantized read-out never decodes energy that
        the setting dropped. NaN stays NaN.
        """
        out = super().read(values, mode)
        if self._uneven:
            out = np.where(self._keep[:, : out.shape[-1]] | np.isnan(out), out, 0.0)
        return out

    def data_mask(self, x: np.ndarray) -> np.ndarray:
        """Where a sample's frame has data: `x` physical ``(N, *values_shape, T)`` -> bool, same shape.

        Per-stream recoder: the front's mask, so a sample has data when its channel (or its probe and
        band, or its profile) has a valid sample in the frame. Concatenated recoder: a frame has data
        when any stream of the signal has a valid sample in it, and the answer holds for every channel
        of the frame, since the frame's tokens fill a missing stream; only a frame in which every
        stream is missing is masked.
        """
        m = self.front.data_mask(x)
        if not self.recoder.concat:
            return m
        n, t = m.shape[0], m.shape[-1]
        frame = m.reshape(n, -1, t).any(1)                          # (N, T)
        return np.broadcast_to(frame.reshape((n,) + (1,) * len(self.values_shape) + (t,)), m.shape)

    def encode_domain(self, u: np.ndarray) -> np.ndarray:
        """The front's domain ``(N, *values_shape, T)`` to recoder values ``(N, F, n_tok, n_val)`` float64.

        Front values to streams, to example units, through the recoder. Per-stream: a stream without a
        token in a frame (a NaN example) gets NaN values and no other stream is touched. Concatenated: a
        missing stream is 0 in example units, so the frame gets its tokens; a frame with every stream
        missing is NaN.
        """
        e = self.norm.apply(store.to_streams(self.front, self.front.encode_domain(u)))     # (N, F, S, D)
        n, f, s, d = e.shape
        finite = np.isfinite(e).all(-1)                              # (N, F, S)
        if self.recoder.concat:
            frame = finite.any(-1)                                   # (N, F)
            rows = np.where(finite[..., None], e, 0.0)[frame].reshape(-1, s * d)
            out = np.full((n, f, self._used), np.nan)
            if len(rows):
                out[frame] = self.recoder.encode(rows, np.full(len(rows), -1, dtype=np.int64))[:, : self._used]
        else:
            rows = e[finite]                                         # (M, D)
            out = np.full((n, f, s, self._used), np.nan)
            if len(rows):
                out[finite] = self.recoder.encode(rows, np.broadcast_to(self.ident, finite.shape)[finite])[:, : self._used]
        out = out.reshape(n, f, self.n_tok, self.n_val)
        if self._uneven:
            out = np.where(self._keep | np.isnan(out), out, 0.0)
        return out

    def decode_domain(self, values: np.ndarray, n: int) -> np.ndarray:
        """Recoder values ``(N, F, n_tok, k)`` to the front's domain ``(N, *values_shape, n)``.

        `k` is at most `n_val` (exactly `n_val` with a multi-token recoder). The values are padded with
        zeros to the recoder's size, decoded to example units, taken out of example units and decoded
        by the front. A NaN value makes its example NaN, and a NaN example decodes to NaN.
        """
        v = np.asarray(values, dtype=np.float64)
        if v.ndim != 4 or v.shape[2] != self.n_tok or v.shape[3] > self.n_val \
                or (self.recoder.n_tok > 1 and v.shape[3] != self.n_val):
            raise ValueError(f"values must be (N, F, {self.n_tok}, k <= {self.n_val}), got {v.shape}")
        n_ex, f = v.shape[:2]
        s, d = store.stream_shape(self.front)
        concat = self.recoder.concat
        if concat:
            flat = v.reshape(n_ex, f, -1)                            # (N, F, t * k)
        else:
            flat = v.reshape(n_ex, f, s, -1)                         # (N, F, S, t * k)
        ok = np.isfinite(flat).all(-1)                               # (N, F) or (N, F, S)
        ident = np.full(int(ok.sum()), -1, dtype=np.int64) if concat else np.broadcast_to(self.ident, ok.shape)[ok]
        rows = np.zeros((int(ok.sum()), self.recoder.n_values))
        rows[:, : flat.shape[-1]] = flat[ok]
        e = np.full(ok.shape + ((s * d,) if concat else (d,)), np.nan)
        if len(rows):
            e[ok] = self.recoder.decode(rows, ident)
        e = e.reshape(n_ex, f, s, d)
        return self.front.decode_domain(store.from_streams(self.front, self.norm.invert(e)), n)


# ------------------------------------------------------------------------------------- save and load

@dataclass
class Loaded:
    """What `load_recoder` reads: the `recoder`, the `norms` of each signal it was fitted on, the stream
    `table`, the `meta` dict as saved (``meta["signals"][name]`` has ``kind``, ``domain``, ``n_streams`` and
    ``values_per_token``) and `ranges`, per signal ``(lo, hi, width)`` of the recoder values, each
    ``(S, n_values)`` per-stream and ``(1, n_values)`` concatenated; ``{}`` if none were saved."""

    recoder: Recoder
    norms: dict[str, Norm]
    table: StreamTable
    meta: dict
    ranges: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def save_recoder(out_dir: Path, recoder: Recoder, norms: Mapping[str, Norm], table: StreamTable, meta: dict,
                 ranges: Mapping[str, tuple] | None = None) -> None:
    """Write a recoder into `out_dir`: `recoder.json`, `norm.npz`, `table.npz`, `ranges.npz` if given, `basis.npz` if linear.

    `norms[name]` is the `Norm` of each signal, front values to example units. `table` is the stream
    table (its `features` ``(n, N_FEAT)`` float32 go to `table.npz`, its signals into `recoder.json`).
    `meta` is JSON and holds ``signals: {name: {kind, domain, n_streams, values_per_token}}``
    (`ValueError` without it). `ranges[name]` is ``(lo, hi, width)`` of the recoder values of that
    signal in recoder-value units, each ``(S, n_values)`` per-stream and ``(1, n_values)`` concatenated.
    A network's `best.pt` is put beside these by the trainer. `recoder.json` is written last, so a
    directory with it is complete; `FileExistsError` if `out_dir` holds one (never overwritten).
    """
    if "signals" not in meta:
        raise ValueError("meta must hold 'signals': {name: {kind, domain, n_streams, values_per_token}}")
    out = prepare_write_root(Path(out_dir))
    if (out / RECODER_JSON).exists():
        raise FileExistsError(f"{out / RECODER_JSON} exists; a recoder is written once")
    arrays = {}
    for name, norm in norms.items():
        arrays[f"{name}:center"], arrays[f"{name}:scale"] = norm.center, norm.scale
    _write_npz(out / "norm.npz", **arrays)
    _write_npz(out / "table.npz", features=np.asarray(table.features, dtype=np.float32))
    if ranges:
        _write_npz(out / "ranges.npz", **{f"{name}:{part}": np.asarray(a) for name, r in ranges.items()
                                          for part, a in zip(("lo", "hi", "width"), r)})
    if isinstance(recoder, LinearRecoder):
        _write_npz(out / "basis.npz", mean=recoder.mean, basis=recoder.basis)
    doc = {"kind": recoder.kind, "concat": bool(recoder.concat), "d_in": int(recoder.d_in),
           "n_tok": int(recoder.n_tok), "n_val": int(recoder.n_val),
           "signals": {name: [int(a), int(b)] for name, (a, b) in table.signals.items()}, "meta": meta}
    tmp = out / (RECODER_JSON + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, out / RECODER_JSON)


def load_recoder(directory: Path, device: str = "cpu") -> Loaded:
    """Read what `save_recoder` wrote (and, for a network, the trainer's `best.pt`) into a `Loaded`.

    A recoder of kind ``"net"`` is built by `nets.load_net_recoder(directory, table, device)`, imported
    only then. `ValueError` if a linear recoder's saved shapes disagree with `recoder.json`.
    """
    d = Path(directory)
    doc = json.loads((d / RECODER_JSON).read_text())
    with np.load(d / "table.npz", allow_pickle=False) as z:
        table = StreamTable({k: (int(a), int(b)) for k, (a, b) in doc["signals"].items()}, z["features"])
    norms: dict[str, Norm] = {}
    with np.load(d / "norm.npz", allow_pickle=False) as z:
        for key in z.files:
            name, part = key.rsplit(":", 1)
            if part == "center":
                norms[name] = Norm(z[key], z[f"{name}:scale"])
    ranges: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if (d / "ranges.npz").exists():
        with np.load(d / "ranges.npz", allow_pickle=False) as z:
            for key in z.files:
                name, part = key.rsplit(":", 1)
                if part == "lo":
                    ranges[name] = (z[key], z[f"{name}:hi"], z[f"{name}:width"])
    if doc["kind"] == "linear":
        with np.load(d / "basis.npz", allow_pickle=False) as z:
            recoder: Recoder = LinearRecoder(z["mean"], z["basis"], concat=doc["concat"], n_val=doc["n_val"])
        if (recoder.d_in, recoder.n_tok, recoder.n_val) != (doc["d_in"], doc["n_tok"], doc["n_val"]):
            raise ValueError(f"{d}: basis.npz is ({recoder.d_in}, {recoder.n_tok}, {recoder.n_val}) but recoder.json "
                             f"says ({doc['d_in']}, {doc['n_tok']}, {doc['n_val']}) for (d_in, n_tok, n_val)")
    elif doc["kind"] == "net":
        from .nets import load_net_recoder                          # torch is imported only for a network
        recoder = load_net_recoder(d, table, device)
    else:
        raise ValueError(f"{d}: unknown recoder kind {doc['kind']!r}")
    return Loaded(recoder, norms, table, doc["meta"], ranges)


def recoder_sha256(directory: Path) -> str:
    """Hex sha256 over the bytes of the files of a recoder directory that exist, in the order of `_FILES`
    (``basis.npz``, ``best.pt``, ``norm.npz``, ``ranges.npz``, ``recoder.json``, ``table.npz``).

    `FileNotFoundError` if the directory has no `recoder.json`: an incomplete or absent recoder has no hash.
    """
    d = Path(directory)
    if not (d / RECODER_JSON).is_file():
        raise FileNotFoundError(f"{d} holds no {RECODER_JSON}; it is not a saved recoder")
    h = hashlib.sha256()
    for name in _FILES:
        if (d / name).is_file():
            h.update((d / name).read_bytes())
    return h.hexdigest()
