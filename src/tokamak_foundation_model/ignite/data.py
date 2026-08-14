"""IGNITE Phase-A data glue: log-power STFT + raw δ-shift consistency pairs.

This module is the **data-side** contract for the statistics-first spectrogram codec
(see ``docs/IGNITE_DESIGN.md`` §4.2). It provides two things:

1. :func:`log_power_stft` — the codec **input target**: a full-resolution log-power
   spectrogram computed with exactly the STFT grid of the existing data pipeline
   (``n_fft=STFT_N_FFT``, ``hop=STFT_HOP``, Hann window, DC bin dropped), then cropped /
   padded to ``(cfg.freq_bins, cfg.time_frames)``.

2. :func:`shift_pair_windows` — the **nuisance pair** for the shift-consistency loss:
   given a *raw* shot signal and a start time ``t0``, it takes the window
   ``[t0, t0 + CHUNK_S]`` and the δ-shifted window ``[t0 + δ, t0 + CHUNK_S + δ]``
   (δ = ``delta_ms · STFT_FS / 1000`` samples), STFTs both to log-power, and returns two
   ``(C, F, T)`` tensors that share ~all modes (freq statistic) but differ in sub-window
   phase realization. ``delta_ms`` is drawn from ``cfg.consistency_delta_ms`` when not given.

Reuse boundary (§7): this is DATA code. It reproduces the STFT recipe of
``data_loader.py`` (which it does NOT modify or import model logic from) and depends only
on ``config.py``, ``torch`` and ``numpy``. No FAITH *model* code is used.

STFT recipe mirrored from ``TokamakSingleFileDataset._compute_stft`` /
``_load_signal_raw`` in ``src/tokamak_foundation_model/data/data_loader.py``:

    spec = torch.stft(signal, n_fft, hop_length, window=hann_window(n_fft),
                       return_complex=True)
    magnitude = torch.abs(spec)[:, 1:, :]   # drop DC (bin 0) -> n_fft//2 freq bins

Here we additionally square the magnitude (power) and take ``log10(power + eps)``, matching
the design's *log-power* target (§4.2). ``center=True`` (torch default) is kept, so with a
window of ``W`` samples the number of STFT frames is ``W // STFT_HOP + 1``.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch

from .config import (
    CHUNK_S,
    FASTTS_FS,
    STFT_FS,
    STFT_HOP,
    STFT_N_FFT,
    FastTSCodecConfig,
    SpectroCodecConfig,
)

ArrayLike = Union[torch.Tensor, np.ndarray]

# Floor added inside the log to keep log-power finite where the STFT magnitude is ~0
# (silent bins / padded tail). log10(0 + eps) is finite and large-negative.
_LOG_EPS: float = 1e-10
_LOG_FLOOR: float = float(np.log10(_LOG_EPS))  # -10.0: log-power of silence
_LOG_CEIL: float = 20.0  # sane upper bound (real log-power maxes ~8); guards pathological bins


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #
def _as_tensor(x: ArrayLike) -> torch.Tensor:
    """Accept numpy or torch; return a float32 torch tensor (no copy if already so)."""
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, dtype=torch.float32)
    if isinstance(x, torch.Tensor):
        return x.to(torch.float32) if x.dtype != torch.float32 else x
    raise TypeError(f"expected torch.Tensor or np.ndarray, got {type(x)!r}")


def _crop_pad_freq_time(spec: torch.Tensor, freq_bins: int, time_frames: int) -> torch.Tensor:
    """Crop or right-pad the (..., F, T) spectrogram to (..., freq_bins, time_frames).

    Frequency is cropped from the low end (bins are already DC-dropped and ordered
    low→high; the codec models ``freq_bins`` starting at the lowest retained bin).
    Time is cropped/padded on the right (later frames), consistent with the data
    loader's right-padding of short windows. Padding uses the log-eps floor value so
    padded regions look like silence, not zeros in log-power space.
    """
    *lead, F, T = spec.shape

    # frequency: crop or pad on the high-freq (right) end of the freq axis
    if F >= freq_bins:
        spec = spec[..., :freq_bins, :]
    else:
        pad_f = freq_bins - F
        pad = spec.new_full((*lead, pad_f, spec.shape[-1]), float(np.log10(_LOG_EPS)))
        spec = torch.cat([spec, pad], dim=-2)

    # time: crop or right-pad
    T = spec.shape[-1]
    if T >= time_frames:
        spec = spec[..., :time_frames]
    else:
        pad_t = time_frames - T
        pad = spec.new_full((*spec.shape[:-1], pad_t), float(np.log10(_LOG_EPS)))
        spec = torch.cat([spec, pad], dim=-1)

    return spec


def _stft_log_power(sig: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Log-power STFT of a (..., W) signal → (..., n_fft//2, n_frames).

    Mirrors data_loader._compute_stft: Hann window, ``return_complex=True``,
    ``center=True`` (torch default), DC bin dropped. torch.stft accepts (W,) or
    (batch, W); flatten leading dims to a single batch axis, apply, then restore.
    """
    *lead, W = sig.shape
    flat = sig.reshape(-1, W)  # (N, W)
    spec = torch.stft(
        flat,
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP,
        window=window,
        return_complex=True,
    )  # (N, n_fft//2+1, n_frames)
    mag = torch.abs(spec)[:, 1:, :]  # drop DC -> (N, n_fft//2, n_frames)
    log_power = torch.log10(mag.pow(2) + _LOG_EPS)
    # Safety net: kill any residual non-finite + clamp to a sane band so a
    # pathological window can never inject inf/nan into the codec loss.
    log_power = torch.nan_to_num(log_power, nan=_LOG_FLOOR, posinf=_LOG_CEIL, neginf=_LOG_FLOOR)
    log_power = log_power.clamp(_LOG_FLOOR, _LOG_CEIL)
    F, n_frames = log_power.shape[-2], log_power.shape[-1]
    return log_power.reshape(*lead, F, n_frames)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def log_power_stft(raw: ArrayLike, cfg: SpectroCodecConfig) -> torch.Tensor:
    """Codec input target: log-power STFT of raw windows.

    Parameters
    ----------
    raw : (B, C, W) tensor or ndarray
        Raw multi-channel windows at ``STFT_FS`` (``W = cfg.window_samples`` for a
        full 50 ms window, but any ``W`` is accepted — the result is cropped/padded
        to ``cfg.time_frames``).
    cfg : SpectroCodecConfig
        Provides ``freq_bins`` and ``time_frames`` for the crop/pad.

    Returns
    -------
    (B, C, cfg.freq_bins, cfg.time_frames) float32 tensor
        Log-power spectrogram, DC-dropped, cropped/right-padded in time and cropped
        (or eps-padded) in frequency.
    """
    x = _as_tensor(raw)
    if x.dim() != 3:
        raise ValueError(f"log_power_stft expects (B, C, W); got shape {tuple(x.shape)}")
    # Sanitize raw: some modalities (notably the CO2 interferometer, present in older
    # shots via a PTDATA fallback) carry sentinel/garbage samples — non-finite or
    # ~float32-max (3.4e38) — that overflow the STFT to inf and collapse the codec to
    # one code (co2 gate env_corr=NaN). Real physical raw is bounded (<<1e20); map
    # non-finite / absurd-magnitude samples to 0 (→ log-eps floor after STFT). No-op
    # for the clean modalities (ece/bes/mhr are all finite, |raw|<<1e20).
    x = torch.where(torch.isfinite(x) & (x.abs() < 1e20), x, torch.zeros_like(x))
    # RAW per-channel standardization for modalities whose raw is NOT O(1) (co2 ~1e13). This MUST
    # happen BEFORE the STFT: it is a per-channel linear rescale, so log10(mag^2) shifts by the
    # per-channel constant -2*log10(raw_std) — moving co2's log-power from ~24 (100% clipped at the
    # _LOG_CEIL=20 ceiling -> flat plate) down into the un-clipped [-10, 20] band, while preserving
    # ALL spectral structure (a constant offset changes neither per-freq nor per-time variation).
    # raw_mean/std are (C,) from the FM's preprocessing_stats[modality]['raw']. No-op when unset
    # (ece/bes/mhr are O(1) raw -> byte-identical).
    if getattr(cfg, "input_standardize", False) and getattr(cfg, "raw_mean", None) is not None:
        rm = torch.as_tensor(cfg.raw_mean, dtype=x.dtype, device=x.device)  # (C,)
        rs = torch.as_tensor(cfg.raw_std, dtype=x.dtype, device=x.device)   # (C,)
        x = (x - rm[None, :, None]) / rs[None, :, None]
    window = torch.hann_window(STFT_N_FFT, dtype=x.dtype, device=x.device)
    spec = _stft_log_power(x, window)  # (B, C, n_fft//2, n_frames)
    spec = _crop_pad_freq_time(spec, cfg.freq_bins, cfg.time_frames)  # (B, C, F, T)
    # Per-freq log-power z-standardization for THIN modalities (co2): subtract the per-(channel,
    # freq) mean and divide by the per-freq std (clamped to a floor so near-constant "noise" bins
    # are not blown up) so a constant can no longer minimize recon-MAE (the mean-collapse fix).
    # Stats live in the codec's OWN log_power_stft space (C, F). No-op when unset (byte-identical).
    if getattr(cfg, "logpow_standardize", False) and getattr(cfg, "logpow_freq_mean", None) is not None:
        fm = torch.as_tensor(cfg.logpow_freq_mean, dtype=spec.dtype, device=spec.device)  # (C, F)
        fs = torch.as_tensor(cfg.logpow_freq_std, dtype=spec.dtype, device=spec.device)   # (C, F)
        fs = fs.clamp_min(float(getattr(cfg, "logpow_std_floor", 0.25)))
        spec = (spec - fm[..., None]) / fs[..., None]  # (C,F,1) broadcasts over (..., C, F, T)
    # Per-window instance z-score (mean~0, std~1 PER (window, channel)) on the log-power input.
    # ROOT-CAUSE FIX for co2 encoder death: co2's log-power carries a large DC offset (window mean
    # ~-9.9 vs ece's ~-1.2); the linear/token layers amplify that offset until the FSQ tanh bound
    # SATURATES (100% of dims pinned to the grid corners -> zero gradient -> encoder outputs a
    # constant -> 1 code). Centering + unit-scaling each (window, channel) removes the offset so the
    # pre-tanh values stay O(1) like ece's (which is why ece never saturated). No-op when unset
    # (ece/bes/mhr byte-identical). Applied LAST so it also neutralizes any residual offset the
    # raw-std / per-freq paths leave behind.
    if getattr(cfg, "input_instance_norm", False):
        mu = spec.mean(dim=(-2, -1), keepdim=True)               # per (..., C): over (F, T)
        sd = spec.std(dim=(-2, -1), keepdim=True)
        q = float(getattr(cfg, "instance_norm_quantize", 0.0))
        if q > 0.0:
            # SHIFT-ROBUST variant: piecewise-constant stats. sd is snapped to a log2
            # grid (multiplicative bins of 2^q) and mu to a grid of (q * snapped sd), so
            # the δ-shift realization jitter in the window stats almost never changes
            # the applied normalization — codes stop flipping with the stats (the
            # measured instance-norm stability tax, retrain v3 2026-08-02).
            sd = torch.exp2(torch.round(torch.log2(sd + 1e-5) / q) * q)
            mu = torch.round(mu / (q * sd)) * (q * sd)
            spec = (spec - mu) / sd
        else:
            spec = (spec - mu) / (sd + 1e-5)
    return spec


def raw_pair_windows(
    raw_shot: ArrayLike,
    t0: float,
    cfg: SpectroCodecConfig,
    delta_ms: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the two RAW windows of a δ-shift consistency pair (no STFT).

    Window A = ``[t0, t0 + CHUNK_S]``; window B = ``[t0 + δ, t0 + CHUNK_S + δ]`` where
    ``δ = round(delta_ms · STFT_FS / 1000)`` samples. Both windows are
    ``cfg.window_samples`` long.

    Parameters
    ----------
    raw_shot : (C, W_full) tensor or ndarray
        A raw single-shot signal at ``STFT_FS``.
    t0 : float
        Window start time in seconds (from the start of ``raw_shot``).
    cfg : SpectroCodecConfig
    delta_ms : float
        Sub-window shift in milliseconds (≥ 0).

    Returns
    -------
    (win_a, win_b) : each (C, cfg.window_samples) float32 tensor

    Raises
    ------
    ValueError
        If ``delta_ms < 0`` or the shifted window ``[t0+δ, t0+CHUNK_S+δ]`` overruns
        the available samples of ``raw_shot``.
    """
    if delta_ms < 0.0:
        raise ValueError(f"delta_ms must be >= 0, got {delta_ms}")
    x = _as_tensor(raw_shot)
    if x.dim() != 2:
        raise ValueError(f"raw_shot must be (C, W_full); got shape {tuple(x.shape)}")

    W = cfg.window_samples
    start_a = round(t0 * STFT_FS)
    shift = round(delta_ms * STFT_FS / 1000.0)
    start_b = start_a + shift

    total = x.shape[-1]
    if start_a < 0:
        raise ValueError(f"t0={t0}s maps to negative sample index {start_a}")
    if start_b + W > total:
        raise ValueError(
            f"shifted window [start={start_b}, end={start_b + W}) overruns shot of "
            f"{total} samples (t0={t0}s, delta_ms={delta_ms}, W={W})"
        )

    win_a = x[..., start_a : start_a + W]
    win_b = x[..., start_b : start_b + W]
    return win_a, win_b


def shift_pair_windows(
    raw_shot: ArrayLike,
    t0: float,
    cfg: SpectroCodecConfig,
    delta_ms: Union[float, None] = None,
    *,
    return_delta: bool = False,
    seed: Union[int, None] = None,
):
    """Build a δ-shift consistency PAIR of log-power spectrograms from a raw shot.

    See ``docs/IGNITE_DESIGN.md`` §4.2. The two windows share ~all mode content
    (frequency statistic) but differ in sub-window phase realization, so
    ``‖enc(spec_a) − enc(spec_b)‖²`` trains shift-invariant (statistics-first) features.

    Parameters
    ----------
    raw_shot : (C, W_full) tensor or ndarray
    t0 : float
        Window start time in seconds.
    cfg : SpectroCodecConfig
    delta_ms : float or None
        Shift in ms. If ``None`` it is drawn uniformly from
        ``cfg.consistency_delta_ms`` (``U[lo, hi]``).
    return_delta : bool, keyword-only
        If True, also return the (sampled) ``delta_ms`` as a third element.
    seed : int or None, keyword-only
        Seeds the δ draw (only used when ``delta_ms is None``) for reproducibility.

    Returns
    -------
    (spec_a, spec_b)                          if ``return_delta`` is False
    (spec_a, spec_b, delta_ms)                if ``return_delta`` is True
        ``spec_*`` are ``(cfg.channels_or_actual, cfg.freq_bins, cfg.time_frames)``
        log-power tensors. (Channel count follows ``raw_shot``'s leading dim.)
    """
    if delta_ms is None:
        lo, hi = cfg.consistency_delta_ms
        gen = np.random.default_rng(seed)
        delta_ms = float(gen.uniform(lo, hi))

    win_a, win_b = raw_pair_windows(raw_shot, t0=t0, cfg=cfg, delta_ms=delta_ms)

    # log_power_stft wants (B, C, W); add a batch axis, run, drop it.
    spec_a = log_power_stft(win_a.unsqueeze(0), cfg)[0]  # (C, F, T)
    spec_b = log_power_stft(win_b.unsqueeze(0), cfg)[0]  # (C, F, T)

    if return_delta:
        return spec_a, spec_b, delta_ms
    return spec_a, spec_b


# =========================================================================== #
# Fast-TS (filterscopes) — ELM ACTIVITY ENVELOPE transform + δ-shift pair.
# =========================================================================== #
# Design (docs/IGNITE_DESIGN.md §4.3): the fast-TS codec's statistic is the ELM ACTIVITY
# ENVELOPE (rate / amplitude), NOT the raw spike waveform. `elm_envelope` maps a raw
# filterscope window (C, W) -> a coarse (C, E) envelope by:
#   1. detrend      — subtract a slow moving-mean baseline (removes DC / slow drift),
#   2. rectify      — square the detrended signal (energy),
#   3. RMS-pool     — mean over non-overlapping `pool`-sample bins, then sqrt (per-bin RMS),
#   4. compress     — log1p(env / eps) so the dynamic range of ELM bursts is bounded.
# The sub-bin spike TIMING (which sample within a `pool`-bin a spike lands on) is discarded —
# that is the realization nuisance, the fast-TS analogue of STFT phase. The per-bin burst
# amplitude and the WHEN of a burst at bin resolution are kept — that is the statistic.
#
# Sanitation mirrors `log_power_stft`: filterscopes can carry non-finite / absurd sentinel
# samples; they are mapped to 0 (→ a quiet envelope) so a pathological window can never inject
# inf/nan into the codec loss.
_FASTTS_ENV_FLOOR: float = 0.0        # log1p(0) = 0: a silent (no-activity) envelope
_FASTTS_ENV_CEIL: float = 30.0        # sane upper bound on log1p(env/eps); guards pathological bins


def _moving_mean(x: torch.Tensor, win: int) -> torch.Tensor:
    """Centered moving-mean along the last axis via a length-`win` box filter (reflect-pad).

    ``x`` is (..., W); returns the same shape. ``win <= 1`` is the identity. Used to estimate
    the slow baseline that is subtracted before rectification (the detrend step). Reflect
    padding keeps the ends from being pulled toward zero.
    """
    if win <= 1:
        return x
    *lead, W = x.shape
    flat = x.reshape(-1, 1, W)                                   # (N, 1, W)
    pad = win // 2
    # reflect padding requires pad < W; clamp for tiny test windows.
    pad = min(pad, max(0, W - 1))
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (pad, pad), mode="reflect")
    kernel = torch.ones(1, 1, win, dtype=flat.dtype, device=flat.device) / float(win)
    out = torch.nn.functional.conv1d(flat, kernel)              # 'valid' conv over padded input
    # conv1d over the reflect-padded signal returns W - win + 1 + 2*pad; crop/pad to W.
    out = out[..., :W] if out.shape[-1] >= W else torch.nn.functional.pad(
        out, (0, W - out.shape[-1]), mode="replicate"
    )
    return out.reshape(*lead, W)


def _standardize_channels(x: torch.Tensor, cfg: FastTSCodecConfig) -> torch.Tensor:
    """Per-channel standardize a (B, C, W) raw window using ``cfg.channel_mean/std``.

    The SCALE FIX for the fast-TS ELM envelope. Mirrors ``data_loader._apply_preprocessing``'s
    ``method="standardize"`` branch EXACTLY: ``(x - mean) / std.clamp(min=1e-3)`` with the SAME
    per-channel raw mean/std the FM model consumes for the ``filterscopes`` modality. GLOBAL /
    per-channel (broadcast over the sample axis) — NOT per-window — so the relative ELM activity
    LEVEL is preserved (quiet windows stay small, active windows stay large). ``cfg.channel_mean``
    or ``cfg.channel_std`` being ``None`` is the identity (no-op, byte-identical to the pre-fix
    path). ``channel_std`` / ``channel_mean`` must have length ``C``.
    """
    if cfg.channel_mean is None or cfg.channel_std is None:
        return x
    C = x.shape[-2]
    mean = torch.as_tensor(cfg.channel_mean, dtype=x.dtype, device=x.device)
    std = torch.as_tensor(cfg.channel_std, dtype=x.dtype, device=x.device)
    if mean.numel() != C or std.numel() != C:
        raise ValueError(
            f"elm_envelope channel stats length {mean.numel()}/{std.numel()} != C={C}"
        )
    mean = mean.reshape(1, C, 1)
    std = std.reshape(1, C, 1).clamp(min=1e-3)  # matches data_loader std.clamp(min=1e-3)
    return (x - mean) / std


def elm_envelope(raw: ArrayLike, cfg: FastTSCodecConfig) -> torch.Tensor:
    """ELM activity envelope of raw filterscope windows — the codec input/target (§4.3).

    Parameters
    ----------
    raw : (B, C, W) tensor or ndarray
        Raw multi-channel filterscope windows at ``FASTTS_FS``. ``W`` need not equal
        ``cfg.env_bins * cfg.pool`` — the pooling crops the trailing remainder so any ``W``
        that is at least ``cfg.pool`` samples is accepted, and the result is cropped / zero
        (silence-floor) padded in the envelope-bin axis to exactly ``cfg.env_bins``.
    cfg : FastTSCodecConfig
        Provides ``pool`` (RMS bin size), ``baseline_win`` (detrend window), ``env_eps``
        (log1p reference) and ``env_bins`` (output length).

    Returns
    -------
    (B, C, cfg.env_bins) float32 tensor
        The log1p-compressed per-bin RMS envelope, non-finite-sanitized and clamped to a
        sane band.
    """
    x = _as_tensor(raw)
    if x.dim() != 3:
        raise ValueError(f"elm_envelope expects (B, C, W); got shape {tuple(x.shape)}")
    # Sanitize raw (mirror log_power_stft): map non-finite / absurd-magnitude samples to 0 so
    # a sentinel/garbage window becomes a quiet envelope instead of inf/nan. Done BEFORE
    # standardization so absurd sentinels can't poison the standardized signal.
    x = torch.where(torch.isfinite(x) & (x.abs() < 1e20), x, torch.zeros_like(x))

    # 0. SCALE FIX — per-channel standardization (see FastTSCodecConfig.channel_mean/std).
    # The raw filterscopes signal is UNSTANDARDIZED (per-channel std ~1e16); without this the
    # log1p(rms/env_eps) compression pins EVERY window at the clamp ceiling -> a constant
    # envelope -> codec collapse. Mirror the data_loader's method="standardize" exactly:
    # x <- (x - mean) / std, per channel, with the SAME per-channel raw mean/std the FM model
    # consumes (std clamped at 1e-3 like data_loader._apply_preprocessing). GLOBAL/per-channel,
    # NOT per-window, so the relative ELM activity LEVEL is preserved. None => no-op (identity),
    # byte-identical to the pre-fix path (synthetic tests / stat-less callers unaffected).
    x = _standardize_channels(x, cfg)

    # 1. detrend: subtract the slow moving-mean baseline.
    if cfg.baseline_win and cfg.baseline_win > 1:
        x = x - _moving_mean(x, int(cfg.baseline_win))

    # 2. rectify (energy).
    energy = x.pow(2)

    # 3. RMS-pool over non-overlapping `pool`-sample bins (crop the trailing remainder).
    pool = int(cfg.pool)
    B, C, W = energy.shape
    n_bins = W // pool
    if n_bins < 1:
        raise ValueError(f"elm_envelope: window W={W} shorter than pool={pool}")
    energy = energy[..., : n_bins * pool].reshape(B, C, n_bins, pool)
    rms = energy.mean(dim=-1).clamp_min(0.0).sqrt()             # (B, C, n_bins) per-bin RMS

    # 4. compress: log1p(rms / eps).
    env = torch.log1p(rms / float(cfg.env_eps))

    # sanitize + clamp, then crop / silence-floor-pad to cfg.env_bins.
    env = torch.nan_to_num(env, nan=_FASTTS_ENV_FLOOR,
                           posinf=_FASTTS_ENV_CEIL, neginf=_FASTTS_ENV_FLOOR)
    env = env.clamp(_FASTTS_ENV_FLOOR, _FASTTS_ENV_CEIL)
    return _crop_pad_env(env, cfg.env_bins)


def _crop_pad_env(env: torch.Tensor, env_bins: int) -> torch.Tensor:
    """Crop or right-pad the (..., E) envelope to (..., env_bins) with the silence floor."""
    E = env.shape[-1]
    if E == env_bins:
        return env
    if E > env_bins:
        return env[..., :env_bins]
    pad = env.new_full((*env.shape[:-1], env_bins - E), _FASTTS_ENV_FLOOR)
    return torch.cat([env, pad], dim=-1)


def fastts_raw_pair_windows(
    raw_shot: ArrayLike,
    t0: float,
    cfg: FastTSCodecConfig,
    delta_ms: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the two RAW filterscope windows of a δ-shift consistency pair (no envelope).

    The fast-TS analogue of :func:`raw_pair_windows`: window A = ``[t0, t0 + CHUNK_S]``;
    window B = ``[t0 + δ, t0 + CHUNK_S + δ]`` where ``δ = round(delta_ms · FASTTS_FS / 1000)``
    samples (at the filterscope rate, NOT the spectro STFT rate). Both windows are
    ``cfg.window_samples`` long.
    """
    if delta_ms < 0.0:
        raise ValueError(f"delta_ms must be >= 0, got {delta_ms}")
    x = _as_tensor(raw_shot)
    if x.dim() != 2:
        raise ValueError(f"raw_shot must be (C, W_full); got shape {tuple(x.shape)}")

    W = cfg.window_samples
    start_a = round(t0 * FASTTS_FS)
    shift = round(delta_ms * FASTTS_FS / 1000.0)
    start_b = start_a + shift

    total = x.shape[-1]
    if start_a < 0:
        raise ValueError(f"t0={t0}s maps to negative sample index {start_a}")
    if start_b + W > total:
        raise ValueError(
            f"shifted window [start={start_b}, end={start_b + W}) overruns shot of "
            f"{total} samples (t0={t0}s, delta_ms={delta_ms}, W={W})"
        )
    return x[..., start_a : start_a + W], x[..., start_b : start_b + W]


def fastts_shift_pair_windows(
    raw_shot: ArrayLike,
    t0: float,
    cfg: FastTSCodecConfig,
    delta_ms: Union[float, None] = None,
    *,
    return_delta: bool = False,
    seed: Union[int, None] = None,
):
    """Build a δ-shift consistency PAIR of ELM envelopes from a raw filterscope shot.

    The fast-TS analogue of :func:`shift_pair_windows`. The two windows share ~all ELM
    activity (the envelope statistic) but differ in sub-bin spike TIMING (the realization),
    so ``‖enc(env_a) − enc(env_b)‖²`` on the PRE-FSQ features trains timing-invariant
    (statistics-first) features. δ ~ ``U`` over ``cfg.consistency_delta_ms`` when not given.

    Returns
    -------
    (env_a, env_b)               if ``return_delta`` is False
    (env_a, env_b, delta_ms)     if ``return_delta`` is True
        ``env_*`` are ``(C, cfg.env_bins)`` envelope tensors. (Channel count follows
        ``raw_shot``'s leading dim.)
    """
    if delta_ms is None:
        lo, hi = cfg.consistency_delta_ms
        gen = np.random.default_rng(seed)
        delta_ms = float(gen.uniform(lo, hi))

    win_a, win_b = fastts_raw_pair_windows(raw_shot, t0=t0, cfg=cfg, delta_ms=delta_ms)
    env_a = elm_envelope(win_a.unsqueeze(0), cfg)[0]  # (C, E)
    env_b = elm_envelope(win_b.unsqueeze(0), cfg)[0]  # (C, E)

    if return_delta:
        return env_a, env_b, delta_ms
    return env_a, env_b


# --------------------------------------------------------------------------- #
# Integration note — pulling raw windows from the real data loader
# --------------------------------------------------------------------------- #
#
# The synthetic tests above do NOT touch HDF5. For real data, the raw window that
# feeds shift_pair_windows / log_power_stft comes from the EXISTING pipeline in
# ``src/tokamak_foundation_model/data/data_loader.py`` — reused, never modified:
#
#   ds = TokamakSingleFileDataset(hdf5_path, ...)          # opens one shot
#   cfg_sig = <the SignalConfig for the modality, e.g. the ece entry in ds's configs>
#   raw, valid_len, nan_mask = ds._load_signal_raw(         # (C, T) at cfg_sig.target_fs
#       ds.h5_file, cfg_sig, t_start=t0, t_end=t0 + CHUNK_S + delta_s)
#
# ``_load_signal_raw`` already resamples to ``target_fs`` (500 kHz for the spectro
# modalities, == STFT_FS) and zero-pads/NaN-fills, so its (C, T) output is exactly the
# raw window this module STFTs. To build a δ-pair, request the *extended* span
# ``[t0, t0 + CHUNK_S + delta_s]`` once, then slice the two sub-windows here (or call
# raw_pair_windows on that span with the t0 rebased to 0).
#
# STATUS — raw-window access is STRAIGHTFORWARD, with two caveats flagged for the
# codec-training glue (NOT blockers, and NOT changes to the loader):
#   1. ``_load_signal_raw`` is a "private" (underscore) method and takes an OPEN h5py
#      handle (``ds.h5_file``) plus a per-modality ``SignalConfig``; there is no public
#      "give me raw samples for modality X over [a, b]" entry point. It works today but
#      is an internal API, so pin the call site.
#   2. The higher-level ``__getitem__`` returns the *processed* STFT (magnitude,
#      log10(x+1), standardized) — NOT the raw signal and NOT log-POWER. So the codec
#      target must be recomputed via ``log_power_stft`` from the raw window; do not try
#      to reuse ``__getitem__``'s spectro output.
