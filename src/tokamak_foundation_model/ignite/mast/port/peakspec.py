"""Peak + baseline codec for the two MAST Mirnov spectrograms (dev-peter's peak-and-baseline advice).

Why. The log-power band-DCT codec (`logspec.LogBandDCT`) keeps the modes TokEye sees (mask F1 0.93
against the truth), but IGNITE's rollouts on it lose them: mask F1 0.14, no better than holding the
last seed frame, at half the codec's peak contrast (2.7 against 6.1 dB), for run 5 and run 6 alike
(`docs/models/ignite-mast-spectrogram-quality.md`, measured 2026-11-24). A spectral line is spread
over many of an array's 432 cosine coefficients, and a model that samples each coefficient on its
own washes the line out. Here a line is one group of three values (where, how high, how wide), so a
sampled peak always draws a sharp line.

What. Per probe and 5 ms frame, in centred log10 power ``u = log10(max(P, floor)) - log_mean`` (the
per-bin floor and mean of a `logspec` fit):

    S          the frame's spectrum: the mean of u over the frame's columns, per bin (all bins, DC
               included). A probe-frame with fewer than 90% finite bins has no data; the other
               missing bins count as 0 (the mean);
    baseline   the first `m` orthonormal DCT-II coefficients of S over the bins, fitted robustly: a
               second pass replaces the bins where the first residual tops `threshold` by the first
               baseline, so a strong line does not lift its surroundings;
    peaks      the bins fall into `n_bands` bands of equal width; band b holds at most one peak, the
               highest local maximum of R = S - baseline (the baseline as the grids will decode it)
               inside the band and above `threshold`, as (offset of the peak from the band's first
               bin, sub-bin by a parabola through three bins; height above the baseline; Gaussian
               sigma from the width at half height, `SIGMA_MIN` .. `SIGMA_MAX` bins). An empty band
               codes ``(-0.5, 0, SIGMA_MIN)``, the lowest centre of each of its three grids.

Slot b always means "band b": a mode drifting in frequency stays in its slot while it stays in its
band, so holding a token holds a mode. Decoding: baseline + the sum over bands of
``h exp(-(f - f_b)^2 / (2 s_b^2))``, the same spectrum for each column of the frame (as LogBandDCT).

Token layout: ``n_tok = P`` (one token a probe), ``n_val = m + 3 n_bands``, the values
``[c_0 .. c_{m-1}, off_0, h_0, s_0, ..., off_{B-1}, h_{B-1}, s_{B-1}]``. Every position is live.

Measured 2026-11-24 with the prototype (``spec_peak_proto3.py``, 36 validation probes classed ok by
the TokEye judge; TokEye mask F1 against the 1.024 ms truth, one pixel of slack, and peak contrast,
the median over columns and 32-bin bands of p95 - median of 10 log10 P), mode-rich probes:

                                               OMV (truth 7.8 dB)    CC (truth 7.2 dB)    values a probe
    LogBandDCT (current)                       0.925  5.3 dB         0.882  4.7 dB        144
    m 24, 32 bands, robust baseline (default)  0.867  5.4 dB         0.843  4.8 dB        120
    m 16, 32 bands                             0.859  4.9 dB         0.833  4.6 dB        112
    m 16, 24 bands                             0.851  5.0 dB         0.821  4.7 dB         88
    m 16, 16 bands                             0.820  5.1 dB         0.799  4.7 dB         64

The prototype dropped the DC bin; the codec keeps it in the baseline. The fit (`fit_peakspec`)
writes ``logspec_fit.{npz,json}`` with ``"codec": "peakspec"``, so a frame-code manifest records it
in ``logspec_fit_path`` / ``logspec_fit_sha256`` and `logspec.build_logspec_codecs` builds these
codecs from it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import scipy.fft

from .. import cache, paths, tasks, transform
from ..tokenizers import codecs as tok_codecs
from ..tokenizers import frames
from ..tokenizers.grid import GridSet, fit_ranges

CODEC_KIND = "peakspec"
FIT_VERSION = 1
PEAKSPEC_M: int = 24
PEAKSPEC_N_BANDS: int = 32
PEAKSPEC_THRESHOLD: float = 0.1  # log10 power above the baseline (1 dB); the prototype's value
PEAKSPEC_N_BINS: int = 64
SIGMA_MIN: float = 0.5
SIGMA_MAX: float = 6.0
MIN_FINITE_BINS: float = 0.9
HEIGHT_WIDTH: float = 0.3  # asinh width of the height grid: about 0.01 (0.1 dB) steps near 0
SIGMA_WIDTH: float = 1.0
_WIDTH_SCAN: int = 16  # bins searched on each side of a peak for its half height
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


class PeakSpec(tok_codecs.Codec):
    """Baseline + one peak a band, per probe and frame, in centred log10 power (module docstring)."""

    def __init__(
        self,
        name: str,
        dt: float,
        values_shape: tuple[int, int],
        stats: transform.ChannelStats,
        floor: np.ndarray,
        log_mean: np.ndarray,
        m: int = PEAKSPEC_M,
        n_bands: int = PEAKSPEC_N_BANDS,
        threshold: float = PEAKSPEC_THRESHOLD,
        grids: GridSet | None = None,
    ) -> None:
        values_shape = tuple(int(v) for v in values_shape)
        if len(values_shape) != 2:
            raise ValueError(f"{name}: values_shape must be (probes, n_bins), got {values_shape}")
        p, nb = values_shape
        m, n_bands = int(m), int(n_bands)
        if n_bands < 1 or nb % n_bands:
            raise ValueError(f"{name}: n_bands={n_bands} must divide n_bins={nb}")
        if not 1 <= m <= nb:
            raise ValueError(f"{name}: m={m} must be in 1..{nb}")
        if not float(threshold) > 0.0:
            raise ValueError(f"{name}: threshold must be positive, got {threshold}")
        super().__init__(
            name, dt, values_shape, stats, n_tok=p, n_val=m + 3 * n_bands, domain="signal",
            grids=grids,
        )
        floor_arr = np.asarray(floor, dtype=np.float64)
        log_mean_arr = np.asarray(log_mean, dtype=np.float64)
        if floor_arr.shape != values_shape or log_mean_arr.shape != values_shape:
            raise ValueError(
                f"{name}: floor and log_mean must have shape {values_shape}, got "
                f"{floor_arr.shape} and {log_mean_arr.shape}"
            )
        if not (np.isfinite(floor_arr).all() and (floor_arr > 0.0).all()):
            raise ValueError(f"{name}: floor must be finite and strictly positive")
        if not np.isfinite(log_mean_arr).all():
            raise ValueError(f"{name}: log_mean must be finite")
        if grids is not None and (grids.n_tok, grids.n_val) != (self.n_tok, self.n_val):
            raise ValueError(
                f"{name}: grids are {grids.n_tok}x{grids.n_val}, codec is {self.n_tok}x{self.n_val}"
            )
        self.floor = floor_arr
        self.log_mean = log_mean_arr
        self.m = m
        self.n_bands = n_bands
        self.band_w = nb // n_bands
        self.threshold = float(threshold)
        self.band_start = np.arange(n_bands, dtype=np.float64) * self.band_w
        self._centres = grids.centres() if grids is not None else None

    # -- domain: centred log10 power (as LogBandDCT)
    def to_domain(self, x: np.ndarray) -> np.ndarray:
        """Physical power ``(..., P, n_bins, T)`` to centred log10 power (NaN kept)."""
        arr = np.asarray(x, dtype=np.float64)
        return np.log10(np.maximum(arr, self.floor[..., None])) - self.log_mean[..., None]

    def from_domain(self, u: np.ndarray) -> np.ndarray:
        """Centred log10 power ``(..., P, n_bins, T)`` to physical power (NaN kept)."""
        return np.power(10.0, np.asarray(u, dtype=np.float64) + self.log_mean[..., None])

    def domain_from_model(self, z: np.ndarray) -> np.ndarray:
        """Model units (the cache's) ``(..., P, n_bins, T)`` to centred log10 power."""
        phys = self._per_signal(lambda a: transform.to_physical(a, self.stats), z)
        return self.to_domain(phys)

    # -- spectra per frame
    def frame_spectra(self, u: np.ndarray, fi: np.ndarray, n_frames: int) -> np.ndarray:
        """Domain samples ``(P, n_bins, M)`` on frames `fi` -> spectra ``(n_frames, P, n_bins)``.

        The mean over each frame's columns, per bin, of the finite values; a probe-frame with fewer
        than `MIN_FINITE_BINS` of its bins finite is NaN, its other empty bins are 0.
        """
        u = np.asarray(u, dtype=np.float64)
        fi = np.asarray(fi, dtype=np.int64)
        p, nb = self.values_shape
        if u.shape[:2] != (p, nb) or u.shape[-1] != fi.size:
            raise ValueError(f"u must be ({p}, {nb}, {fi.size}), got {u.shape}")
        sel = (fi >= 0) & (fi < n_frames)
        ok = np.isfinite(u[..., sel])
        vals = np.moveaxis(np.where(ok, u[..., sel], 0.0), -1, 0)  # (M', P, nb)
        cnt_in = np.moveaxis(ok, -1, 0).astype(np.float64)
        sums = np.zeros((n_frames, p, nb))
        cnt = np.zeros((n_frames, p, nb))
        np.add.at(sums, fi[sel], vals)
        np.add.at(cnt, fi[sel], cnt_in)
        mean = np.divide(sums, cnt, out=np.zeros_like(sums), where=cnt > 0)
        bad = (cnt > 0).mean(-1) < MIN_FINITE_BINS
        mean[bad] = np.nan
        return mean

    def _baseline(self, c: np.ndarray) -> np.ndarray:
        """Coefficients ``(..., k)`` (k <= n_bins) -> baseline ``(..., n_bins)``; NaN rows stay NaN."""
        nb = self.values_shape[1]
        c = np.asarray(c, dtype=np.float64)
        full = np.zeros(c.shape[:-1] + (nb,))
        full[..., : c.shape[-1]] = c
        return scipy.fft.idct(full, type=2, norm="ortho", axis=-1)

    def _coefs(self, x: np.ndarray) -> np.ndarray:
        return scipy.fft.dct(x, type=2, norm="ortho", axis=-1)[..., : self.m]

    def _quantize_baseline(self, c: np.ndarray, tok: np.ndarray) -> np.ndarray:
        """Baseline coefficients ``(K, m)`` of rows whose probe is `tok` ``(K,)`` -> nearest grid centres."""
        if self.grids is None:
            return c
        arr = np.full((c.shape[0], self.n_tok, self.n_val), np.nan)
        arr[np.arange(c.shape[0]), tok, : self.m] = c
        idx = self.grids.index(arr)[np.arange(c.shape[0]), tok, : self.m]  # (K, m)
        cen = self._centres[tok[:, None], np.arange(self.m)[None, :], np.maximum(idx, 0)]
        return np.where(idx < 0, np.nan, cen)

    def encode_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """Spectra ``(F, P, n_bins)`` (domain) -> values ``(F, P, n_val)``; NaN for a NaN probe-frame.

        The baseline values are returned unquantized; the peaks are found against the baseline
        the grids decode (when the codec has grids), so a frame's peaks sit on the baseline drawn.
        """
        s = np.asarray(spectra, dtype=np.float64)
        f_count, p, nb = s.shape
        out = np.full((f_count, p, self.n_val), np.nan)
        rows = np.isfinite(s).all(-1)
        if not rows.any():
            return out
        x = s[rows]  # (K, nb)
        tok = np.nonzero(rows)[1]
        k_rows = x.shape[0]
        thr = self.threshold
        c = self._coefs(x)
        b1 = self._baseline(c)
        c = self._coefs(np.where(x - b1 > thr, b1, x))  # the robust second pass
        r = x - self._baseline(self._quantize_baseline(c, tok))

        # local maxima of the residual above the threshold (an edge bin counts its one neighbour)
        neg = np.full((k_rows, 1), -np.inf)
        rp = np.concatenate([neg, r, neg], axis=1)
        is_max = (r > rp[:, :-2]) & (r >= rp[:, 2:]) & (r > thr)
        rm = np.where(is_max, r, -np.inf).reshape(k_rows, self.n_bands, self.band_w)
        j_in = np.argmax(rm, axis=-1)  # (K, B)
        has = np.isfinite(np.take_along_axis(rm, j_in[..., None], -1)[..., 0])
        j = j_in + np.arange(self.n_bands)[None, :] * self.band_w
        ri = np.arange(k_rows)[:, None]
        re = np.concatenate([r[:, :1], r, r[:, -1:]], axis=1)  # edge-padded for the parabola
        a, y, z = re[ri, j], re[ri, j + 1], re[ri, j + 2]
        den = a - 2.0 * y + z
        d = np.divide(0.5 * (a - z), den, out=np.zeros_like(den), where=den < 0)
        d = np.clip(d, -0.5, 0.5)
        h = y - 0.25 * (a - z) * d
        half = 0.5 * h
        left = np.zeros(j.shape, dtype=np.int64)
        right = np.zeros(j.shape, dtype=np.int64)
        go_l = np.ones(j.shape, dtype=bool)
        go_r = np.ones(j.shape, dtype=bool)
        for k in range(1, _WIDTH_SCAN + 1):
            jl, jr = j - k, j + k
            go_l &= (jl >= 0) & (r[ri, np.clip(jl, 0, nb - 1)] > half)
            go_r &= (jr < nb) & (r[ri, np.clip(jr, 0, nb - 1)] > half)
            left += go_l
            right += go_r
        # the half-height crossings, linear between the last bin above half and the next one
        # (half a bin out when the scan runs off the spectrum or past _WIDTH_SCAN)
        def _cross(k_in: np.ndarray, step: int) -> np.ndarray:
            jin = j + step * k_in
            jout = jin + step
            inside = (jout >= 0) & (jout < nb)
            r_in = r[ri, np.clip(jin, 0, nb - 1)]
            r_out = r[ri, np.clip(jout, 0, nb - 1)]
            drop = r_in - r_out
            frac = np.divide(r_in - half, drop, out=np.full(drop.shape, 0.5), where=inside & (drop > 0))
            return jin + step * np.clip(frac, 0.0, 1.0)

        fwhm = _cross(right, 1) - _cross(left, -1)
        sigma = np.clip(fwhm / 2.355, SIGMA_MIN, SIGMA_MAX)
        off = j_in + d
        peaks = np.stack(
            [np.where(has, off, -0.5), np.where(has, h, 0.0), np.where(has, sigma, SIGMA_MIN)],
            axis=-1,
        )  # (K, B, 3)
        vals = np.concatenate([c, peaks.reshape(k_rows, 3 * self.n_bands)], axis=1)
        out[rows] = vals
        return out

    def render_spectra(self, values: np.ndarray) -> np.ndarray:
        """Values ``(F, P, n_val)`` -> spectra ``(F, P, n_bins)`` (domain).

        A probe-frame with a NaN baseline value is NaN; a peak with a NaN or non-positive height
        draws nothing.
        """
        v = np.asarray(values, dtype=np.float64)
        if v.shape[-1] != self.n_val:
            raise ValueError(f"values need {self.n_val} values a token, got {v.shape}")
        s = self._baseline(v[..., : self.m])
        s[np.isnan(v[..., : self.m]).any(-1)] = np.nan
        pk = v[..., self.m:].reshape(v.shape[:-1] + (self.n_bands, 3))
        x = np.arange(self.values_shape[1], dtype=np.float64)
        for b in range(self.n_bands):
            f = self.band_start[b] + pk[..., b, 0]
            h = pk[..., b, 1]
            sg = np.maximum(pk[..., b, 2], 0.3)
            draw = np.isfinite(f) & np.isfinite(h) & np.isfinite(sg) & (h > 0.0)
            if not draw.any():
                continue
            g = np.exp(-0.5 * ((x - np.where(draw, f, 0.0)[..., None]) / np.where(draw, sg, 1.0)[..., None]) ** 2)
            s = s + np.where(draw, h, 0.0)[..., None] * g
        return s

    # -- encode / decode on a frame assignment (the frame cache's path)
    def encode_aligned(self, u: np.ndarray, fi: np.ndarray, n_frames: int) -> np.ndarray:
        """Domain samples ``(P, n_bins, M)`` on frames `fi` -> values ``(n_frames, P, n_val)``."""
        return self.encode_spectra(self.frame_spectra(u, fi, n_frames))

    def render_domain(self, values: np.ndarray, fi: np.ndarray, n_frames: int) -> np.ndarray:
        """Values ``(n_frames, P, n_val)`` -> domain ``(P, n_bins, M)`` on frames `fi` (NaN off them)."""
        v = np.asarray(values, dtype=np.float64)
        fi = np.asarray(fi, dtype=np.int64)
        spec = self.render_spectra(v[:n_frames])
        p, nb = self.values_shape
        out = np.full((p, nb, fi.size), np.nan)
        ok = (fi >= 0) & (fi < spec.shape[0])
        out[..., ok] = np.moveaxis(spec[fi[ok]], 0, -1)
        return out

    def render(self, values: np.ndarray, fi: np.ndarray, n_frames: int) -> np.ndarray:
        """Values ``(n_frames, P, n_val)`` -> physical power ``(P, n_bins, M)`` on frames `fi`."""
        return self.from_domain(self.render_domain(values, fi, n_frames))

    # -- the Codec interface (whole frames of a contiguous signal)
    def encode_domain(self, u: np.ndarray) -> np.ndarray:
        """Domain ``(N, P, n_bins, T)`` -> values ``(N, F, P, n_val)``."""
        u = np.asarray(u, dtype=np.float64)
        fi = frames.frame_index(u.shape[-1], self.dt)
        n_frames = int(fi.max()) + 1
        return np.stack([self.encode_aligned(u[i], fi, n_frames) for i in range(u.shape[0])])

    def decode_domain(self, values: np.ndarray, n: int) -> np.ndarray:
        """Values ``(N, F, P, n_val)`` -> domain ``(N, P, n_bins, n)``."""
        v = np.asarray(values, dtype=np.float64)
        if v.ndim != 4 or v.shape[2] != self.n_tok:
            raise ValueError(f"values must be (N, F, {self.n_tok}, {self.n_val}), got {v.shape}")
        fi = frames.frame_index(int(n), self.dt)
        n_frames = int(fi.max()) + 1
        if v.shape[1] < n_frames:
            raise ValueError(f"{n} samples need {n_frames} frames, values have {v.shape[1]}")
        return np.stack([self.render_domain(v[i], fi, v.shape[1]) for i in range(v.shape[0])])

    def data_mask(self, x: np.ndarray) -> np.ndarray:
        """A probe has data in a frame when one of its bins at one of the frame's columns is finite."""
        valid = np.isfinite(np.asarray(x, dtype=np.float64)).any(-2)  # (N, P, T)
        has = tok_codecs._frame_any(valid, self.dt)
        return np.broadcast_to(has[..., None, :], np.asarray(x).shape).copy()


# ------------------------------------------------------------------------------------- the fit

def _utc_now() -> str:
    return subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_npz(path: Path, entries: Mapping[str, np.ndarray]) -> None:
    """Deterministic, atomic npz (sorted keys, fixed time stamps, stored)."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
            for key in sorted(entries):
                buf = io.BytesIO()
                np.lib.format.write_array(buf, np.asarray(entries[key]), allow_pickle=False)
                zi = zipfile.ZipInfo(f"{key}.npy", date_time=_FIXED_TIME)
                zi.compress_type = zipfile.ZIP_STORED
                zi.external_attr = 0o600 << 16
                zf.writestr(zi, buf.getvalue())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def peak_grids(
    baseline_values: np.ndarray,
    heights: np.ndarray,
    m: int,
    n_bands: int,
    band_w: int,
    *,
    n_bins: int = PEAKSPEC_N_BINS,
    margin: float = 0.05,
    height_hi_pct: float = 99.9,
) -> GridSet:
    """The grids of a `PeakSpec` from fit data.

    `baseline_values` ``(N, P, m)`` (NaN missing): `fit_ranges` per probe and coefficient (asinh
    width from the MAD). Peaks, the same for every probe: offset ``-0.5 .. band_w - 0.5`` linear
    (sub-bin steps); height ``0 .. p<height_hi_pct>`` of `heights` (the heights of found peaks),
    asinh width `HEIGHT_WIDTH`; sigma `SIGMA_MIN` .. `SIGMA_MAX`, asinh width `SIGMA_WIDTH`.
    """
    bv = np.asarray(baseline_values, dtype=np.float64)
    p = bv.shape[1]
    lo_b, hi_b, w_b = fit_ranges(bv, margin=margin)
    if not (np.isfinite(lo_b).all() and np.isfinite(hi_b).all()):
        raise ValueError("a baseline coefficient was never seen in the fit data")
    h = np.asarray(heights, dtype=np.float64)
    h = h[np.isfinite(h) & (h > 0)]
    if h.size == 0:
        raise ValueError("no peak heights in the fit data")
    h_hi = float(np.percentile(h, height_hi_pct))
    n_val = m + 3 * n_bands
    lo = np.zeros((p, n_val))
    hi = np.ones((p, n_val))
    width = np.ones((p, n_val))
    lo[:, :m], hi[:, :m], width[:, :m] = lo_b, hi_b, w_b
    for b in range(n_bands):
        o = m + 3 * b
        lo[:, o], hi[:, o], width[:, o] = -0.5, band_w - 0.5, 1e3
        lo[:, o + 1], hi[:, o + 1], width[:, o + 1] = 0.0, h_hi, HEIGHT_WIDTH
        lo[:, o + 2], hi[:, o + 2], width[:, o + 2] = SIGMA_MIN, SIGMA_MAX, SIGMA_WIDTH
    return GridSet(lo, hi, width, n_bins=int(n_bins))


def _read_frames_worker(job: tuple) -> tuple[str, int, np.ndarray | None, np.ndarray | None]:
    """One (shot, signal): the unquantized values of its ok probes over the plasma frames."""
    import torch

    from . import framecache

    (shot, sig_name, probes, cache_root, frames_dir, stats_sha, st, codec_args) = job
    entry = torch.load(Path(frames_dir) / f"{shot}.pt", weights_only=False)
    if not entry["present"].get(sig_name, False):
        return shot, 0, None, None
    cs = cache.read_shot(
        Path(cache_root) / "shots" / f"{shot}.h5", [sig_name], stats_sha256=stats_sha
    ).get(sig_name)
    if cs is None or cs.values.shape[-1] == 0:
        return shot, 0, None, None
    codec = PeakSpec(*codec_args)
    t0, nf = float(entry["t0"]), int(entry["n_frames"])
    fi = framecache.sample_frame_index(cs.values.shape[-1], cs.t0, cs.dt, t0)
    keep = (fi >= 0) & (fi < nf)
    if not keep.any():
        return shot, 0, None, None
    u = codec.to_domain(transform.to_physical(cs.values[..., keep], st))
    vals = codec.encode_aligned(u, fi[keep], nf)[:, list(probes)]  # (F, P', n_val)
    vals = vals[np.isfinite(vals).all(-1).all(-1)]
    if vals.shape[0] == 0:
        return shot, 0, None, None
    pk = vals[..., codec.m:].reshape(vals.shape[:2] + (codec.n_bands, 3))
    return shot, int(vals.shape[0]), vals[..., : codec.m], pk[..., 1]


def fit_peakspec(
    cache_root: Path | str,
    frames_dir: Path | str,
    logspec_fit_dir: Path | str,
    keep_dir: Path | str,
    out_dir: Path | str,
    *,
    signal_names: Sequence[str] = tasks.SPECTROGRAM_SIGNALS,
    n_shots: int = 300,
    rich_share: float = 0.5,
    m: int = PEAKSPEC_M,
    n_bands: int = PEAKSPEC_N_BANDS,
    threshold: float = PEAKSPEC_THRESHOLD,
    n_bins: int = PEAKSPEC_N_BINS,
    margin: float = 0.05,
    seed: int = 0,
    workers: int = 16,
) -> dict:
    """Fit the grids on training shots the TokEye judge keeps; write ``logspec_fit.{npz,json}``.

    dev-peter's advice was to fit a spectrogram tokenizer on shots with enough mode content. Per
    signal, `n_shots` training (shot, signal) pairs kept by the judge (``keep_dir/signals.parquet``):
    the ``rich_share`` with the most mode activity (``mode_mean``), the rest drawn at random from the
    other kept pairs; only the probes classed ok (``probes.parquet``) contribute. The floor and
    log mean come from the `logspec` fit at `logspec_fit_dir` (copied, so the fit stands alone).
    `frames_dir` is any port frame-code cache: its ``t0`` / ``n_frames`` give the plasma frames.
    """
    import concurrent.futures
    import multiprocessing

    import pandas as pd

    cache_root, frames_dir = Path(cache_root), Path(frames_dir)
    logspec_fit_dir, keep_dir = Path(logspec_fit_dir), Path(keep_dir)
    out_dir = paths.prepare_write_root(Path(out_dir))
    stats_sha = str(json.loads((cache_root / "manifest.json").read_text())["stats_sha256"])
    if transform.stats_sha256(cache_root / "stats.npz") != stats_sha:
        raise ValueError(f"{cache_root}/stats.npz sha256 disagrees with manifest.json")
    ls_meta = json.loads((logspec_fit_dir / "logspec_fit.json").read_text())
    if str(ls_meta.get("stats_sha256")) != stats_sha:
        raise ValueError(f"logspec fit {logspec_fit_dir} was made with other stats")
    if ls_meta.get("codec", "band_dct") != "band_dct":
        raise ValueError(f"{logspec_fit_dir} is not a band-DCT logspec fit")
    stats = transform.load_stats(cache_root / "stats.npz")
    signals = pd.read_parquet(keep_dir / "signals.parquet")
    probes = pd.read_parquet(keep_dir / "probes.parquet")
    from .specquality import SHORT

    rng = np.random.default_rng(seed)
    entries: dict[str, np.ndarray] = {}
    sig_meta: dict[str, dict] = {}
    with np.load(logspec_fit_dir / "logspec_fit.npz", allow_pickle=False) as z:
        ls = {n: (np.asarray(z[f"{n}::floor"]), np.asarray(z[f"{n}::log_mean"])) for n in signal_names}
    for name in signal_names:
        short = SHORT[name]
        floor, log_mean = ls[name]
        values_shape = (int(floor.shape[0]), int(floor.shape[1]))
        dt = float(ls_meta["signals"][name]["dt"])
        kept = signals[(signals.sig == short) & (signals.split == "train") & signals.keep]
        kept = kept.sort_values(["mode_mean", "shot"], ascending=[False, True], na_position="last")
        n_rich = min(int(round(n_shots * rich_share)), len(kept))
        rich = kept.iloc[:n_rich]
        rest = kept.iloc[n_rich:]
        n_rand = min(n_shots - n_rich, len(rest))
        rand = rest.iloc[np.sort(rng.choice(len(rest), size=n_rand, replace=False))] if n_rand else rest.iloc[:0]
        pick = [str(s) for s in pd.concat([rich, rand]).shot.tolist()]
        ok = probes[(probes.sig == short) & (probes.cls == "ok")]
        ok_probes = ok.groupby("shot").probe.apply(lambda s: sorted(int(v) for v in s)).to_dict()
        codec_args = (name, dt, values_shape, stats[name], floor, log_mean, m, n_bands, threshold, None)
        jobs = [
            (s, name, ok_probes.get(int(s), ok_probes.get(s, [])), str(cache_root), str(frames_dir),
             stats_sha, stats[name], codec_args)
            for s in pick
        ]
        jobs = [j for j in jobs if j[2]]
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max(1, min(workers, len(jobs))), mp_context=ctx) as ex:
            results = list(ex.map(_read_frames_worker, jobs))
        used = [r for r in results if r[2] is not None]
        if not used:
            raise ValueError(f"{name}: no fit data in {len(jobs)} shots")
        # every probe's baseline values feed every probe's grid only through its own column; a probe
        # missing from a shot contributes NaN there
        base = []
        heights = []
        for (shot, _n, bv, hv), job in zip(results, jobs):
            if bv is None:
                continue
            full = np.full((bv.shape[0], values_shape[0], m), np.nan)
            full[:, job[2]] = bv
            base.append(full)
            heights.append(hv.reshape(-1))
        base_all = np.concatenate(base)
        h_all = np.concatenate(heights)
        band_w = values_shape[1] // n_bands
        grids = peak_grids(base_all, h_all, m, n_bands, band_w, n_bins=n_bins, margin=margin)
        PeakSpec(name, dt, values_shape, stats[name], floor, log_mean, m, n_bands, threshold, grids)
        entries[f"{name}::floor"] = floor.astype(np.float64)
        entries[f"{name}::log_mean"] = log_mean.astype(np.float64)
        entries[f"{name}::lo"] = grids.lo
        entries[f"{name}::hi"] = grids.hi
        entries[f"{name}::width"] = grids.width
        occupied = float((h_all > 0).mean())
        sig_meta[name] = {
            "dt": dt,
            "values_shape": list(values_shape),
            "shots": [r[0] for r in used],
            "n_shots": len(used),
            "n_rich_picked": int(n_rich),
            "n_frames": int(base_all.shape[0]),
            "band_occupancy": occupied,
            "height_p50": float(np.median(h_all[h_all > 0])) if (h_all > 0).any() else None,
            "height_hi": float(grids.hi[0, m + 1]),
        }
    _write_npz(out_dir / "logspec_fit.npz", entries)
    meta = {
        "codec": CODEC_KIND,
        "version": FIT_VERSION,
        "n_bins": int(n_bins),
        "m": int(m),
        "n_bands": int(n_bands),
        "threshold": float(threshold),
        "margin": float(margin),
        "stats_sha256": stats_sha,
        "cache_v1_root": str(cache_root.resolve()),
        "frames_dir": str(frames_dir.resolve()),
        "keep_dir": str(keep_dir.resolve()),
        "logspec_fit_source": str(logspec_fit_dir.resolve()),
        "logspec_fit_source_sha256": _sha256(logspec_fit_dir / "logspec_fit.npz"),
        "n_shots_target": int(n_shots),
        "rich_share": float(rich_share),
        "seed": int(seed),
        "created_utc": _utc_now(),
        "signals": sig_meta,
    }
    tmp = out_dir / f"logspec_fit.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, out_dir / "logspec_fit.json")
    meta["logspec_fit_sha256"] = _sha256(out_dir / "logspec_fit.npz")
    return meta


def build_peakspec_codecs(
    fit_dir: Path | str,
    stats: Mapping[str, transform.ChannelStats],
) -> dict[str, PeakSpec]:
    """Load a peakspec ``<fit_dir>/logspec_fit.{json,npz}`` into ``{name: PeakSpec}``."""
    fit_path = Path(fit_dir)
    meta = json.loads((fit_path / "logspec_fit.json").read_text())
    if meta.get("codec") != CODEC_KIND:
        raise ValueError(f"{fit_path} is not a peakspec fit (codec={meta.get('codec')!r})")
    out: dict[str, PeakSpec] = {}
    with np.load(fit_path / "logspec_fit.npz", allow_pickle=False) as z:
        for name, info in meta["signals"].items():
            if name not in stats:
                raise KeyError(f"signal {name!r} missing from stats")
            floor = np.asarray(z[f"{name}::floor"], dtype=np.float64)
            grids = GridSet(
                np.asarray(z[f"{name}::lo"], dtype=np.float64),
                np.asarray(z[f"{name}::hi"], dtype=np.float64),
                np.asarray(z[f"{name}::width"], dtype=np.float64),
                n_bins=int(meta["n_bins"]),
            )
            out[name] = PeakSpec(
                name, float(info["dt"]), (int(floor.shape[0]), int(floor.shape[1])), stats[name],
                floor, np.asarray(z[f"{name}::log_mean"], dtype=np.float64),
                int(meta["m"]), int(meta["n_bands"]), float(meta["threshold"]), grids,
            )
    return out
