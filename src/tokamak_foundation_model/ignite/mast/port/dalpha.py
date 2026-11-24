"""D-alpha peak + level codec for the MAST port: a frame's level plus its largest spikes.

The v1b assignment codes D-alpha (3 channels, 50 kHz, 250 samples per 5 ms frame) with
``frame_dct`` k=2: each frame's mean and slope. That smooths every ELM spike away, and in a
rollout each frame's predicted slope makes the decoded trace jagged. dev-peter's advice for
the D3D codecs was a combination of peak and baseline tokens. `PeakLevel` does that per
channel and frame, with ``n_val = 1 + 2 * n_peaks`` values:

    level      the frame mean of the signal with its coded spikes removed (V),
               decoded by linear interpolation between frame centres (no steps, no slope);
    h_i, t_i   height (V above the level curve) and time (fraction of the frame) of the
               i-th largest spike, i = 1 .. n_peaks; h = 0, t = 0 when the frame has no
               spike above the fitted threshold.

Each spike decodes as ``h * template`` at its time; the template is the mean normalized
spike shape fitted on training shots and may run into the next frame. The encoder works on
the quantized level curve and the quantized spikes the decoder will draw, frame by frame.

Measured 2026-11-24 on 62 validation shots (NRMSE = RMSE / global std 0.464 V over plasma
samples, median over shots; prototype ``dalpha_proto.py``):

    current frame_dct k=2 (6 tokens)    0.145      level + 1 spike  (6 tokens)   0.144
    level only (3 tokens)               0.182      level + 2 spikes (15 tokens)  0.131
                                                   level + 3 spikes (21 tokens)  0.119

The fit (`fit_dalpha`) stores the template, the spike threshold and the grids in
``dalpha_fit.npz``; a frame-code manifest records its path and sha256 like the log-power
Mirnov fit (``logspec``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .. import cache, paths, transform
from ..tokenizers import codecs as tok_codecs
from ..tokenizers import frames
from ..tokenizers.grid import GridSet

SIGNAL = "spectrometer_visible-filter_spectrometer_dalpha_voltage"
FIT_VERSION = 1


def _frame_reduce(fn, x: np.ndarray, frame_of: list[np.ndarray]) -> np.ndarray:
    """``fn`` (a NaN-ignoring reduction) of `x` over each frame's samples; NaN for a frame with no finite one."""
    out = np.full(len(frame_of), np.nan)
    for f, idx in enumerate(frame_of):
        seg = x[idx]
        ok = np.isfinite(seg)
        if ok.any():
            out[f] = fn(seg[ok])
    return out


class PeakLevel(tok_codecs.Codec):
    """Level + ``n_peaks`` spikes per channel and frame (module docstring). Physical units (V)."""

    def __init__(
        self,
        name: str,
        dt: float,
        values_shape: tuple[int, ...],
        stats: transform.ChannelStats,
        template: np.ndarray,
        pre: int,
        threshold: float,
        n_peaks: int,
        grids: GridSet,
        local_k: float = 5.0,
    ) -> None:
        if len(values_shape) != 1:
            raise ValueError(f"PeakLevel needs (channels,) values, got {values_shape}")
        super().__init__(
            name, dt, values_shape, stats, n_tok=int(values_shape[0]),
            n_val=1 + 2 * int(n_peaks), domain="signal", grids=grids,
            signal_mean=0.0, signal_std=1.0,
        )
        tmpl = np.asarray(template, dtype=np.float64)
        if tmpl.ndim != 1 or not (0 <= int(pre) < tmpl.size) or abs(tmpl[int(pre)] - 1.0) > 1e-6:
            raise ValueError("template must be 1-D with value 1 at index pre")
        self.template = tmpl
        self.pre = int(pre)
        self.threshold = float(threshold)
        self.local_k = float(local_k)
        self.n_peaks = int(n_peaks)
        if grids.n_tok != self.n_tok or grids.n_val != self.n_val:
            raise ValueError(f"grids are {grids.n_tok}x{grids.n_val}, codec is {self.n_tok}x{self.n_val}")
        self._centres = grids.centres()

    # -- quantization helpers
    def _q(self, values, tok: int, val: int) -> np.ndarray:
        """The grid centres nearest to `values` of token `tok`, value `val` (NaN stays NaN), same shape."""
        v = np.asarray(values, dtype=np.float64)
        arr = np.full(v.shape + (self.n_tok, self.n_val), np.nan)
        arr[..., tok, val] = v
        i = self.grids.index(arr)[..., tok, val]
        return np.where(i < 0, np.nan, self._centres[tok, val, np.maximum(i, 0)])

    # -- frame geometry
    @staticmethod
    def _frames(fi: np.ndarray, n_frames: int) -> tuple[list[np.ndarray], np.ndarray]:
        """Sample indices of each frame and each frame's centre (NaN for an empty frame)."""
        order = [np.flatnonzero(fi == f) for f in range(n_frames)]
        centre = np.array([idx.mean() if idx.size else np.nan for idx in order])
        return order, centre

    def _level_curve(self, levels: np.ndarray, centre: np.ndarray, frame_of: list[np.ndarray],
                     n_samp: int) -> np.ndarray:
        """Linear interpolation of per-frame levels between frame centres; NaN in frames without a level."""
        ok = np.isfinite(levels) & np.isfinite(centre)
        if not ok.any():
            return np.full(n_samp, np.nan)
        out = np.interp(np.arange(n_samp), centre[ok], levels[ok])
        for f, idx in enumerate(frame_of):
            if idx.size and not np.isfinite(levels[f]):
                out[idx] = np.nan
        return out

    def _add_spike(self, comp: np.ndarray, at: int, h: float) -> None:
        lo = max(0, at - self.pre)
        hi = min(comp.size, at - self.pre + self.template.size)
        if hi > lo:
            comp[lo:hi] += h * self.template[lo - (at - self.pre): hi - (at - self.pre)]

    def _spikes(self, x: np.ndarray, base: np.ndarray, frame_of: list[np.ndarray], c: int,
                ) -> tuple[np.ndarray, np.ndarray]:
        """Quantized spike values ``(n_frames, 2 * n_peaks)`` and their drawn component ``(M,)``."""
        n_frames = len(frame_of)
        vals = np.zeros((n_frames, 2 * self.n_peaks))
        comp = np.zeros(x.size)
        for f, idx in enumerate(frame_of):
            if idx.size == 0:
                continue
            seg = x[idx] - base[idx]
            if not np.isfinite(seg).any():
                vals[f] = np.nan
                continue
            n = idx.size
            # the threshold is the fitted one or `local_k` x this frame's robust noise, whichever is
            # larger: in a channel whose fast fluctuations top the fitted threshold every frame, a noise
            # maximum drawn as an ELM-shaped spike adds error instead of removing it (shot 22038, ch 0).
            # The noise is read from sample-to-sample differences (MAD / sqrt 2), which a slow drift of
            # the signal against the level curve does not inflate and sparse spikes barely move.
            d = np.diff(x[idx])
            d = d[np.isfinite(d)]
            sigma = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0) if d.size else 0.0
            thr = max(self.threshold, self.local_k * sigma)
            # a spike's time is quantized to about (n - 1) / (n_bins - 1) samples, so the residual of a
            # placed spike can top the threshold next to it; the next spike is searched outside that guard
            guard = int(np.ceil((n - 1) / max(self.grids.n_bins - 1, 1))) + 1
            blocked = np.zeros(n, dtype=bool)
            for p in range(self.n_peaks):
                r = np.where(blocked, np.nan, seg - comp[idx])
                if not np.isfinite(r).any():
                    break
                j = int(np.nanargmax(r))
                h = float(r[j])
                if not h > thr:
                    break
                hq = float(self._q(h, c, 1 + 2 * p))
                tq = float(self._q(j / max(n - 1, 1), c, 2 + 2 * p))
                jq = int(round(tq * max(n - 1, 1)))
                vals[f, 2 * p], vals[f, 2 * p + 1] = h, j / max(n - 1, 1)
                self._add_spike(comp, int(idx[jq]), hq)
                blocked[max(0, min(j, jq) - guard): max(j, jq) + guard + 1] = True
        return vals, comp

    # -- encode / decode on a frame assignment
    def encode_aligned(self, x: np.ndarray, fi: np.ndarray, n_frames: int) -> np.ndarray:
        """Physical ``(C, M)`` samples on frames `fi` -> values ``(n_frames, C, n_val)`` (NaN: no data)."""
        x = np.asarray(x, dtype=np.float64)
        frame_of, centre = self._frames(np.asarray(fi), n_frames)
        out = np.full((n_frames, self.n_tok, self.n_val), np.nan)
        for c in range(self.n_tok):
            xc = x[c]
            med = _frame_reduce(np.median, xc, frame_of)
            base0 = self._level_curve(self._q(med, c, 0), centre, frame_of, xc.size)
            _, comp0 = self._spikes(xc, base0, frame_of, c)
            lev = _frame_reduce(np.mean, xc - comp0, frame_of)
            levq = self._q(lev, c, 0)
            base1 = self._level_curve(levq, centre, frame_of, xc.size)
            spk, _ = self._spikes(xc, base1, frame_of, c)
            out[:, c, 0] = lev
            out[:, c, 1:] = spk
            out[~np.isfinite(lev), c, 1:] = np.nan
        return out

    def render(self, values: np.ndarray, fi: np.ndarray, n_frames: int) -> np.ndarray:
        """Values ``(n_frames, C, n_val)`` (grid centres) -> physical ``(C, M)`` on frames `fi`."""
        v = np.asarray(values, dtype=np.float64)
        fi = np.asarray(fi)
        frame_of, centre = self._frames(fi, n_frames)
        out = np.full((self.n_tok, fi.size), np.nan)
        for c in range(self.n_tok):
            base = self._level_curve(v[:, c, 0], centre, frame_of, fi.size)
            comp = np.zeros(fi.size)
            for f, idx in enumerate(frame_of):
                if idx.size == 0:
                    continue
                n = idx.size
                for p in range(self.n_peaks):
                    h, t = v[f, c, 1 + 2 * p], v[f, c, 2 + 2 * p]
                    if not (np.isfinite(h) and np.isfinite(t)) or h <= 0.0:
                        continue
                    jq = int(round(min(max(t, 0.0), 1.0) * max(n - 1, 1)))
                    self._add_spike(comp, int(idx[jq]), h)
            out[c] = base + comp
        return out

    # -- the Codec interface (whole frames of a contiguous signal)
    def encode_domain(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        fi = frames.frame_index(u.shape[-1], self.dt)
        n_frames = int(fi.max()) + 1
        return np.stack([self.encode_aligned(u[k], fi, n_frames) for k in range(u.shape[0])])

    def decode_domain(self, values: np.ndarray, n: int) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        fi = frames.frame_index(int(n), self.dt)
        if v.shape[-1] < self.n_val:
            pad = np.zeros(v.shape[:-1] + (self.n_val - v.shape[-1],))
            v = np.concatenate([v, pad], axis=-1)
        return np.stack([self.render(v[k], fi, v.shape[1]) for k in range(v.shape[0])])

    def data_mask(self, x: np.ndarray) -> np.ndarray:
        return tok_codecs._frame_any(np.isfinite(np.asarray(x)), self.dt)


# ------------------------------------------------------------------------------------- the fit

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dalpha_fit_sha256(fit_dir: Path | str) -> str:
    """sha256 hex digest of ``<fit_dir>/dalpha_fit.npz`` (it holds every number the codec uses)."""
    return _sha256(Path(fit_dir) / "dalpha_fit.npz")


def _read_physical(shot: str, cache_root: Path, frames_dir: Path, stats_sha: str,
                   st: transform.ChannelStats) -> tuple[np.ndarray, np.ndarray, int] | None:
    """A shot's D-alpha over its frame-cache plasma frames: physical ``(C, M)``, frame ids, n_frames."""
    import torch

    from . import framecache

    entry = torch.load(Path(frames_dir) / f"{shot}.pt", weights_only=False)
    if not entry["present"].get(SIGNAL, False):
        return None
    cs = cache.read_shot(Path(cache_root) / "shots" / f"{shot}.h5", [SIGNAL], stats_sha256=stats_sha).get(SIGNAL)
    if cs is None or cs.values.shape[-1] == 0:
        return None
    t0, nf = float(entry["t0"]), int(entry["n_frames"])
    fi = framecache.sample_frame_index(cs.values.shape[-1], cs.t0, cs.dt, t0)
    keep = (fi >= 0) & (fi < nf)
    if not keep.any():
        return None
    return transform.to_physical(cs.values, st)[:, keep].astype(np.float64), fi[keep], nf


def fit_dalpha(
    cache_root: Path | str,
    frames_dir: Path | str,
    train_shots: Sequence[str],
    out_dir: Path | str,
    *,
    n_shots: int = 400,
    n_peaks: int = 2,
    n_bins: int = 64,
    pre: int = 10,
    post: int = 90,
    level_width: float = 0.2,
    local_k: float = 5.0,
    seed: int = 0,
) -> dict:
    """Fit the template, spike threshold and grids on ``n_shots`` training shots; write the fit.

    `frames_dir` is any port frame-code cache: its per-shot ``t0``/``n_frames`` give the plasma
    frames. Threshold = 5 x the robust noise level (1.4826 x MAD) of the signal around its
    per-frame median curve; template = mean of the spikes above 20 x that level, normalized to
    1 at the peak, offset removed, clipped at 0. Grids (asinh centres, `GridSet`): level
    p0.05 .. max of the frame medians with width `level_width` (linear below it, log-like
    above: 0.2 V gives steps of about 13 mV at 0.1 V, 63 mV at 1 V and 0.25 V at 4 V); spike height
    0 .. p99.9 of the spikes above threshold with width 0.01 V; time 0 .. 1, uniform.
    """
    cache_root, frames_dir = Path(cache_root), Path(frames_dir)
    out_dir = paths.prepare_write_root(Path(out_dir))
    stats_sha = str(json.loads((cache_root / "manifest.json").read_text())["stats_sha256"])
    if transform.stats_sha256(cache_root / "stats.npz") != stats_sha:
        raise ValueError(f"{cache_root}/stats.npz sha256 disagrees with manifest.json")
    stats = transform.load_stats(cache_root / "stats.npz")
    st = stats[SIGNAL]
    rng = np.random.default_rng(seed)
    pool = sorted({str(s) for s in train_shots}, key=int)
    pick = rng.choice(pool, size=min(n_shots, len(pool)), replace=False).tolist()
    data = [d for d in (_read_physical(s, cache_root, frames_dir, stats_sha, st) for s in pick) if d is not None]
    if not data:
        raise ValueError("no training shot has D-alpha")
    n_ch = data[0][0].shape[0]
    meds, resid = [], []
    for x, fi, nf in data:
        frame_of = [np.flatnonzero(fi == f) for f in range(nf)]
        centre = np.array([i.mean() if i.size else np.nan for i in frame_of])
        for c in range(n_ch):
            m = _frame_reduce(np.median, x[c], frame_of)
            meds.append(m[np.isfinite(m)])
            ok = np.isfinite(m) & np.isfinite(centre)
            if ok.sum() > 1:
                resid.append(x[c] - np.interp(np.arange(x.shape[1]), centre[ok], m[ok]))
    allm = np.concatenate(meds)
    r = np.concatenate(resid)
    r = r[np.isfinite(r)]
    sigma = float(1.4826 * np.median(np.abs(r - np.median(r))))
    threshold = 5.0 * sigma
    # template from large isolated spikes
    acc, n_tmpl, heights = np.zeros(pre + post), 0, []
    for x, fi, nf in data:
        frame_of = [np.flatnonzero(fi == f) for f in range(nf)]
        centre = np.array([i.mean() if i.size else np.nan for i in frame_of])
        for c in range(n_ch):
            m = _frame_reduce(np.median, x[c], frame_of)
            ok = np.isfinite(m) & np.isfinite(centre)
            if ok.sum() < 2:
                continue
            rr = x[c] - np.interp(np.arange(x.shape[1]), centre[ok], m[ok])
            for idx in frame_of:
                seg = rr[idx]
                if idx.size == 0 or not np.isfinite(seg).any():
                    continue
                j = int(np.nanargmax(seg))
                h = float(seg[j])
                if h > threshold:
                    heights.append(h)
                a = int(idx[j])
                if h > 20 * sigma and a - pre >= 0 and a + post <= rr.size and np.isfinite(rr[a - pre: a + post]).all():
                    acc += rr[a - pre: a + post] / h
                    n_tmpl += 1
    if n_tmpl < 20:
        raise ValueError(f"only {n_tmpl} spikes for the template")
    tmpl = acc / n_tmpl
    tmpl = np.clip(tmpl - np.median(tmpl[-15:]), 0.0, None)
    tmpl = tmpl / tmpl[pre]
    n_val = 1 + 2 * n_peaks
    lo = np.zeros((n_ch, n_val))
    hi = np.ones((n_ch, n_val))
    width = np.full((n_ch, n_val), 100.0)
    # the level's top is the largest frame median: D-alpha saturates near 5 V and a saturated plateau
    # must be a level, not a train of spikes (p99.95 was 3.99 V; shot 22038 sat at 5 V)
    lo[:, 0], hi[:, 0], width[:, 0] = np.percentile(allm, 0.05), float(np.max(allm)), level_width
    h_hi = float(np.percentile(heights, 99.9)) if heights else 1.0
    for p in range(n_peaks):
        lo[:, 1 + 2 * p], hi[:, 1 + 2 * p], width[:, 1 + 2 * p] = 0.0, h_hi, 0.01
    np.savez(
        out_dir / "dalpha_fit.npz", template=tmpl, pre=np.int64(pre), threshold=np.float64(threshold),
        local_k=np.float64(local_k),
        n_peaks=np.int64(n_peaks), n_bins=np.int64(n_bins), lo=lo, hi=hi, width=width,
    )
    meta = {
        "version": FIT_VERSION, "signal": SIGNAL, "stats_sha256": stats_sha,
        "frames_dir": str(frames_dir), "n_shots_used": len(data), "shots": [str(s) for s in pick],
        "noise_sigma": sigma, "threshold": threshold, "local_k": local_k, "n_template_spikes": n_tmpl,
        "height_p999": h_hi, "level_range": [float(lo[0, 0]), float(hi[0, 0])],
        "n_peaks": n_peaks, "n_bins": n_bins, "pre": pre, "post": post, "level_width": level_width,
        "npz_sha256": dalpha_fit_sha256(out_dir),
    }
    (out_dir / "dalpha_fit.json").write_text(json.dumps(meta, indent=2))
    return meta


def build_dalpha_codec(fit_dir: Path | str, stats: Mapping[str, transform.ChannelStats],
                       dt: float) -> PeakLevel:
    """Load ``<fit_dir>/dalpha_fit.npz`` into a `PeakLevel` codec for `SIGNAL`."""
    with np.load(Path(fit_dir) / "dalpha_fit.npz", allow_pickle=False) as z:
        lo, hi, width = (np.asarray(z[k], dtype=np.float64) for k in ("lo", "hi", "width"))
        grids = GridSet(lo, hi, width, n_bins=int(z["n_bins"]))
        return PeakLevel(
            SIGNAL, float(dt), (int(lo.shape[0]),), stats[SIGNAL], np.asarray(z["template"]),
            int(z["pre"]), float(z["threshold"]), int(z["n_peaks"]), grids, float(z["local_k"]),
        )


def resolve_manifest_dalpha_fit(manifest: Mapping) -> Path | None:
    """The manifest's ``dalpha_fit_path`` after checking ``dalpha_fit_sha256``, or None."""
    raw_path = manifest.get("dalpha_fit_path")
    expected = manifest.get("dalpha_fit_sha256")
    if raw_path is None:
        if expected is not None:
            raise ValueError(f"manifest has dalpha_fit_sha256={expected!r} but no dalpha_fit_path")
        return None
    if not expected:
        raise ValueError(f"manifest sets dalpha_fit_path={raw_path!r} without dalpha_fit_sha256")
    path = Path(str(raw_path))
    actual = dalpha_fit_sha256(path)
    if actual != str(expected):
        raise ValueError(f"dalpha_fit_sha256 mismatch for {path}: manifest {expected}, disk {actual}")
    return path
