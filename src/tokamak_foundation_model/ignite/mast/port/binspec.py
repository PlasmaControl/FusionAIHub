"""Per-bin log-power codec for the two MAST Mirnov spectrograms: one token per (probe, frequency bin).

Why. The band-DCT codec (`logspec.LogBandDCT`) keeps the modes TokEye sees (mask F1 0.93 against the
truth), but IGNITE's rollouts on it come out as a mosaic of 31 kHz blocks whose brightness jumps frame
to frame. Measured 2026-11-25 on 200 validation shots (frame cache `frame_codes_v1b_logspec_cam_live_da`,
no model): the mean frame-to-frame change of a code divided by its change against a random frame of
the same shot is 0.24 for a band's mean (DCT level 0) in mode-active frames (TokEye mode share >= 0.05)
but 0.78 to 0.99 for the fine levels j >= 4, which are 78% of an array's 432 tokens. A line that
persists in frequency is a pattern over many cosine coefficients whose quantized codes jump with
sub-bin shifts and speckle, so the previous frame tells a model almost nothing about them, and
sampling them independently draws blocks. On the same frames, codes of each bin's own centred log
power persist: 0.31 for every bin of a mode-active OMV frame at 128 bins (2 kHz), 0.16 for the line
pixels (the frame's top 5%); CC 0.36 over all frames (job tmp ``truth_persist.py``,
``bin_persist.py``). A sampling error in a bin reads as speckle, not as a block.

What. Per probe and 5 ms frame, in centred log10 power ``u = log10(max(P, floor)) - log_mean`` (the
per-bin floor and mean of a `logspec` fit, as `LogBandDCT` and `peakspec.PeakSpec`):

    S        the frame's spectrum: the mean of u over the frame's columns, per bin (`PeakSpec`'s
             `frame_spectra`: a probe-frame with fewer than 90% finite bins has no data, its other
             empty bins count as 0, the mean);
    values   S averaged over `pool` neighbouring bins: ``n_bins / pool`` values a probe.

Decoding repeats each value over its `pool` bins, the same spectrum for each column of the frame.
Token layout: ``n_tok = P`` (one token a probe), ``n_val = n_bins / pool``, every position live, each
quantized on its own grid (`fit_ranges` over the fit data: range and asinh width per probe and bin).
With 256 bins and pool 2 an array has 3 x 128 = 384 positions (the band-DCT codec 432).

`BinSpec` subclasses `PeakSpec` for the shared domain, frame and render plumbing (the frame cache and
the rollout code dispatch on ``isinstance(codec, PeakSpec)``); only the spectrum <-> values step
differs. The fit (`fit_binspec`) writes ``logspec_fit.{npz,json}`` with ``"codec": "binspec"``, so a
frame-code manifest records it in ``logspec_fit_path`` / ``logspec_fit_sha256`` and
`logspec.build_logspec_codecs` builds these codecs from it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .. import cache, paths, tasks, transform
from ..tokenizers import codecs as tok_codecs
from ..tokenizers.grid import GridSet, fit_ranges
from .peakspec import PeakSpec, _sha256, _utc_now, _write_npz

CODEC_KIND = "binspec"
FIT_VERSION = 1
BINSPEC_POOL: int = 2
BINSPEC_N_BINS: int = 64  # grid levels a position (the missing code comes on top)


class BinSpec(PeakSpec):
    """Each probe's frame spectrum, averaged over `pool` neighbouring bins (module docstring)."""

    def __init__(
        self,
        name: str,
        dt: float,
        values_shape: tuple[int, int],
        stats: transform.ChannelStats,
        floor: np.ndarray,
        log_mean: np.ndarray,
        pool: int = BINSPEC_POOL,
        grids: GridSet | None = None,
    ) -> None:
        values_shape = tuple(int(v) for v in values_shape)
        if len(values_shape) != 2:
            raise ValueError(f"{name}: values_shape must be (probes, n_bins), got {values_shape}")
        p, nb = values_shape
        pool = int(pool)
        if pool < 1 or nb % pool:
            raise ValueError(f"{name}: pool={pool} must divide n_bins={nb}")
        tok_codecs.Codec.__init__(
            self, name, dt, values_shape, stats, n_tok=p, n_val=nb // pool, domain="signal",
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
        self.pool = pool
        self._centres = grids.centres() if grids is not None else None

    def encode_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """Spectra ``(F, P, n_bins)`` (domain) -> values ``(F, P, n_bins / pool)``; NaN rows stay NaN."""
        s = np.asarray(spectra, dtype=np.float64)
        f_count, p, nb = s.shape
        if (p, nb) != self.values_shape:
            raise ValueError(f"spectra must be (F, {self.values_shape[0]}, {self.values_shape[1]}), got {s.shape}")
        out = s.reshape(f_count, p, self.n_val, self.pool).mean(-1)
        out[~np.isfinite(s).all(-1)] = np.nan
        return out

    def render_spectra(self, values: np.ndarray) -> np.ndarray:
        """Values ``(F, P, n_val)`` -> spectra ``(F, P, n_bins)`` (domain); a probe-frame with a NaN is NaN."""
        v = np.asarray(values, dtype=np.float64)
        if v.shape[-1] != self.n_val:
            raise ValueError(f"values need {self.n_val} values a token, got {v.shape}")
        s = np.repeat(v, self.pool, axis=-1)
        s[np.isnan(v).any(-1)] = np.nan
        return s


# ------------------------------------------------------------------------------------- the fit

def _read_values_worker(job: tuple) -> tuple[str, int, np.ndarray | None]:
    """One (shot, signal): the unquantized values of its ok probes over the plasma frames."""
    import torch

    from . import framecache

    (shot, sig_name, probes, cache_root, frames_dir, stats_sha, st, codec_args) = job
    entry = torch.load(Path(frames_dir) / f"{shot}.pt", weights_only=False)
    if not entry["present"].get(sig_name, False):
        return shot, 0, None
    cs = cache.read_shot(
        Path(cache_root) / "shots" / f"{shot}.h5", [sig_name], stats_sha256=stats_sha
    ).get(sig_name)
    if cs is None or cs.values.shape[-1] == 0:
        return shot, 0, None
    codec = BinSpec(*codec_args)
    t0, nf = float(entry["t0"]), int(entry["n_frames"])
    fi = framecache.sample_frame_index(cs.values.shape[-1], cs.t0, cs.dt, t0)
    keep = (fi >= 0) & (fi < nf)
    if not keep.any():
        return shot, 0, None
    u = codec.to_domain(transform.to_physical(cs.values[..., keep], st))
    vals = codec.encode_aligned(u, fi[keep], nf)[:, list(probes)]  # (F, P', n_val)
    vals = vals[np.isfinite(vals).all(-1).all(-1)]
    if vals.shape[0] == 0:
        return shot, 0, None
    return shot, int(vals.shape[0]), vals


def fit_binspec(
    cache_root: Path | str,
    frames_dir: Path | str,
    logspec_fit_dir: Path | str,
    keep_dir: Path | str,
    out_dir: Path | str,
    *,
    signal_names: Sequence[str] = tasks.SPECTROGRAM_SIGNALS,
    n_shots: int = 300,
    rich_share: float = 0.5,
    pool: int = BINSPEC_POOL,
    n_bins: int = BINSPEC_N_BINS,
    margin: float = 0.05,
    seed: int = 0,
    workers: int = 16,
) -> dict:
    """Fit the grids on training shots the TokEye judge keeps; write ``logspec_fit.{npz,json}``.

    The shot choice is `peakspec.fit_peakspec`'s (dev-peter: fit a spectrogram tokenizer on shots
    with enough mode content): per signal, `n_shots` training (shot, signal) pairs kept by the judge,
    the ``rich_share`` with the most mode activity and the rest at random, only probes classed ok.
    The floor and log mean are copied from the band-DCT `logspec` fit at `logspec_fit_dir`.
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
        codec_args = (name, dt, values_shape, stats[name], floor, log_mean, pool, None)
        jobs = [
            (s, name, ok_probes.get(int(s), ok_probes.get(s, [])), str(cache_root), str(frames_dir),
             stats_sha, stats[name], codec_args)
            for s in pick
        ]
        jobs = [j for j in jobs if j[2]]
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max(1, min(workers, len(jobs))), mp_context=ctx) as ex:
            results = list(ex.map(_read_values_worker, jobs))
        used = [r for r in results if r[2] is not None]
        if not used:
            raise ValueError(f"{name}: no fit data in {len(jobs)} shots")
        n_val = values_shape[1] // pool
        vals = []
        for (_shot, _n, v), job in zip(results, jobs):
            if v is None:
                continue
            full = np.full((v.shape[0], values_shape[0], n_val), np.nan)
            full[:, job[2]] = v  # a probe missing from a shot contributes NaN to its column
            vals.append(full)
        all_v = np.concatenate(vals)
        lo, hi, width = fit_ranges(all_v, margin=margin)
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            raise ValueError(f"{name}: a (probe, bin) was never seen in the fit data")
        grids = GridSet(lo, hi, width, n_bins=int(n_bins))
        BinSpec(name, dt, values_shape, stats[name], floor, log_mean, pool, grids)
        entries[f"{name}::floor"] = floor.astype(np.float64)
        entries[f"{name}::log_mean"] = log_mean.astype(np.float64)
        entries[f"{name}::lo"] = grids.lo
        entries[f"{name}::hi"] = grids.hi
        entries[f"{name}::width"] = grids.width
        sig_meta[name] = {
            "dt": dt,
            "values_shape": list(values_shape),
            "shots": [r[0] for r in used],
            "n_shots": len(used),
            "n_rich_picked": int(n_rich),
            "n_frames": int(all_v.shape[0]),
            "lo_median": float(np.median(lo)),
            "hi_median": float(np.median(hi)),
            "width_median": float(np.median(width)),
        }
    _write_npz(out_dir / "logspec_fit.npz", entries)
    meta = {
        "codec": CODEC_KIND,
        "version": FIT_VERSION,
        "n_bins": int(n_bins),
        "pool": int(pool),
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


def build_binspec_codecs(
    fit_dir: Path | str,
    stats: Mapping[str, transform.ChannelStats],
) -> dict[str, BinSpec]:
    """Load a binspec ``<fit_dir>/logspec_fit.{json,npz}`` into ``{name: BinSpec}``."""
    fit_path = Path(fit_dir)
    meta = json.loads((fit_path / "logspec_fit.json").read_text())
    if meta.get("codec") != CODEC_KIND:
        raise ValueError(f"{fit_path} is not a binspec fit (codec={meta.get('codec')!r})")
    out: dict[str, BinSpec] = {}
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
            out[name] = BinSpec(
                name, float(info["dt"]), (int(floor.shape[0]), int(floor.shape[1])), stats[name],
                floor, np.asarray(z[f"{name}::log_mean"], dtype=np.float64), int(meta["pool"]), grids,
            )
    return out
