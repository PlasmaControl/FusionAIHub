"""Codec data transforms: log-power STFT and raw δ-shift consistency pairs.

Data-side helpers for the spectrogram codec (``docs/IGNITE_DESIGN.md`` §4.2):

1. :func:`log_power_stft`, the codec input and target: a full-resolution log-power
   spectrogram on the STFT grid of the existing data pipeline (``n_fft=STFT_N_FFT``,
   ``hop=STFT_HOP``, Hann window, DC bin dropped), cropped or padded to
   ``(cfg.freq_bins, cfg.time_frames)``.
2. :func:`shift_pair_windows`, the pair for the shift-consistency loss: from a raw shot
   signal and a start time ``t0`` it takes the window ``[t0, t0 + CHUNK_S]`` and the
   window shifted by δ (``delta_ms * STFT_FS / 1000`` samples), and returns both as
   ``(C, F, T)`` log-power tensors. They share almost all modes but differ in sub-window
   phase realization. ``delta_ms`` is drawn from ``cfg.consistency_delta_ms`` if not
   given.

The module also has the spectro validity mask and the fast-TS (filterscope) window
transforms. It depends only on ``config.py``, ``torch`` and ``numpy`` (§7).

The STFT follows ``TokamakSingleFileDataset._compute_stft`` / ``_load_signal_raw`` in
``src/tokamak_foundation_model/data/data_loader.py``::

    spec = torch.stft(signal, n_fft, hop_length, window=hann_window(n_fft),
                       return_complex=True)
    magnitude = torch.abs(spec)[:, 1:, :]   # drop DC (bin 0) -> n_fft//2 freq bins

Here the magnitude is also squared and ``log10(power + eps)`` taken, the log-power
target of §4.2. ``center=True`` (the torch default) is kept, so a window of ``W``
samples gives ``W // STFT_HOP + 1`` frames.
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

# Floor inside the log, so silent bins and padding give a finite, large negative value.
_LOG_EPS: float = 1e-10
_LOG_FLOOR: float = float(np.log10(_LOG_EPS))  # -10.0: log-power of silence
_LOG_CEIL: float = 20.0  # real log-power peaks near 8; guards against pathological bins


# Internal helpers
def _as_tensor(x: ArrayLike) -> torch.Tensor:
    """Return a float32 torch tensor from numpy or torch (no copy if already one)."""
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, dtype=torch.float32)
    if isinstance(x, torch.Tensor):
        return x.to(torch.float32) if x.dtype != torch.float32 else x
    raise TypeError(f"expected torch.Tensor or np.ndarray, got {type(x)!r}")


def _crop_pad_freq_time(spec: torch.Tensor, freq_bins: int, time_frames: int) -> torch.Tensor:
    """Crop or right-pad a (..., F, T) spectrogram to (..., freq_bins, time_frames).

    Bins are DC-dropped and ordered low to high, so frequency keeps the lowest
    ``freq_bins``. Time is cropped or padded on the right, like the data loader pads
    short windows. Padding uses the log-eps floor, so padded regions look like silence.
    """
    *lead, F, T = spec.shape

    # Frequency: crop or pad at the high end.
    if F >= freq_bins:
        spec = spec[..., :freq_bins, :]
    else:
        pad_f = freq_bins - F
        pad = spec.new_full((*lead, pad_f, spec.shape[-1]), float(np.log10(_LOG_EPS)))
        spec = torch.cat([spec, pad], dim=-2)

    # Time: crop or right-pad.
    T = spec.shape[-1]
    if T >= time_frames:
        spec = spec[..., :time_frames]
    else:
        pad_t = time_frames - T
        pad = spec.new_full((*spec.shape[:-1], pad_t), float(np.log10(_LOG_EPS)))
        spec = torch.cat([spec, pad], dim=-1)

    return spec


def _stft_log_power(
    sig: torch.Tensor,
    window: torch.Tensor,
    n_fft: int = STFT_N_FFT,
    hop_length: int = STFT_HOP,
) -> torch.Tensor:
    """Log-power STFT of a (..., W) signal -> (..., n_fft//2, n_frames).

    As in data_loader._compute_stft: Hann window, ``return_complex=True``,
    ``center=True``, DC bin dropped. Leading dims are flattened into one batch axis for
    torch.stft and restored afterwards.

    ``n_fft`` and ``hop_length`` default to the module globals; :func:`log_power_stft`
    passes the per-codec ``cfg.stft_n_fft`` / ``cfg.stft_hop``.
    """
    *lead, W = sig.shape
    flat = sig.reshape(-1, W)  # (N, W)
    spec = torch.stft(
        flat,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )  # (N, n_fft//2+1, n_frames)
    mag = torch.abs(spec)[:, 1:, :]  # drop DC -> (N, n_fft//2, n_frames)
    log_power = torch.log10(mag.pow(2) + _LOG_EPS)
    # Replace any non-finite value and clamp, so a pathological window cannot put
    # inf or nan into the codec loss.
    log_power = torch.nan_to_num(log_power, nan=_LOG_FLOOR, posinf=_LOG_CEIL, neginf=_LOG_FLOOR)
    log_power = log_power.clamp(_LOG_FLOOR, _LOG_CEIL)
    F, n_frames = log_power.shape[-2], log_power.shape[-1]
    return log_power.reshape(*lead, F, n_frames)


# Public API
def log_power_stft(raw: ArrayLike, cfg: SpectroCodecConfig) -> torch.Tensor:
    """Codec input target: log-power STFT of raw windows.

    Parameters
    ----------
    raw : (B, C, W) tensor or ndarray
        Raw multi-channel windows at ``STFT_FS``. ``W = cfg.window_samples`` is a full
        50 ms window, but any ``W`` works; the result is cropped or padded to
        ``cfg.time_frames``.
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
    # Some modalities (notably CO2 in older shots, read through the PTDATA fallback)
    # contain non-finite or ~float32-max (3.4e38) sentinel samples. They overflow the STFT
    # to inf and collapsed the codec to one code (co2 gate env_corr=NaN). Real raw data is
    # far below 1e20, so such samples are set to 0 (the log-eps floor after the STFT).
    x = torch.where(torch.isfinite(x) & (x.abs() < 1e20), x, torch.zeros_like(x))
    # Per-channel standardization of the raw signal, for modalities whose raw is not
    # O(1) (co2 ~1e13). It has to come before the STFT: a per-channel linear rescale shifts
    # log10(mag^2) by -2*log10(raw_std) per channel, moving co2's log-power from ~24
    # (entirely clipped at _LOG_CEIL = 20, a flat plate) into [-10, 20] without changing
    # any spectral structure. raw_mean/std are (C,) from the foundation model's
    # preprocessing_stats[modality]['raw']. Skipped when unset (ece/bes/mhr).
    if getattr(cfg, "input_standardize", False) and getattr(cfg, "raw_mean", None) is not None:
        rm = torch.as_tensor(cfg.raw_mean, dtype=x.dtype, device=x.device)  # (C,)
        rs = torch.as_tensor(cfg.raw_std, dtype=x.dtype, device=x.device)   # (C,)
        x = (x - rm[None, :, None]) / rs[None, :, None]
    # Per-codec STFT grid. Old pickled configs lack the fields and keep the global
    # 1024/256 grid.
    n_fft = int(getattr(cfg, "stft_n_fft", STFT_N_FFT))
    hop = int(getattr(cfg, "stft_hop", STFT_HOP))
    window = torch.hann_window(n_fft, dtype=x.dtype, device=x.device)
    spec = _stft_log_power(x, window, n_fft=n_fft, hop_length=hop)  # (B, C, n_fft//2, n_frames)
    spec = _crop_pad_freq_time(spec, cfg.freq_bins, cfg.time_frames)  # (B, C, F, T)
    # Per-frequency z-score of the log-power for thin modalities (co2): subtract the
    # per-(channel, frequency) mean and divide by the std, floored so near-constant noise
    # bins are not blown up. Then a constant no longer minimizes the reconstruction MAE.
    # The stats are (C, F), computed in this function's own output space.
    if getattr(cfg, "logpow_standardize", False) and getattr(cfg, "logpow_freq_mean", None) is not None:
        fm = torch.as_tensor(cfg.logpow_freq_mean, dtype=spec.dtype, device=spec.device)  # (C, F)
        fs = torch.as_tensor(cfg.logpow_freq_std, dtype=spec.dtype, device=spec.device)   # (C, F)
        fs = fs.clamp_min(float(getattr(cfg, "logpow_std_floor", 0.25)))
        spec = (spec - fm[..., None]) / fs[..., None]  # (C,F,1) broadcasts over (..., C, F, T)
    # Per-(window, channel) z-score of the log-power input. co2's log-power has a large
    # DC offset (window mean ~-9.9 against ~-1.2 for ece); the linear layers amplify it
    # until the FSQ tanh saturates (every dimension at a grid corner, zero gradient, a
    # constant encoder, one code). Centering and scaling each (window, channel) keeps the
    # pre-tanh values O(1), as for ece. Applied after the other normalizations so it also
    # removes any offset they leave.
    if getattr(cfg, "input_instance_norm", False):
        mu = spec.mean(dim=(-2, -1), keepdim=True)               # per (..., C): over (F, T)
        sd = spec.std(dim=(-2, -1), keepdim=True)
        q = float(getattr(cfg, "instance_norm_quantize", 0.0))
        if q > 0.0:
            # Shift-robust variant with piecewise-constant stats: sd is snapped to a log2
            # grid (factors of 2^q) and mu to a grid of q * sd, so the δ-shift jitter of
            # the window stats almost never changes the normalization and the codes do
            # not flip with it.
            sd = torch.exp2(torch.round(torch.log2(sd + 1e-5) / q) * q)
            mu = torch.round(mu / (q * sd)) * (q * sd)
            spec = (spec - mu) / sd
        else:
            spec = (spec - mu) / (sd + 1e-5)
    # Band-power pooling, last: mean-pools freq_bins into cfg.band_pool equal bands, in
    # the order analysis/_specport_plateau.py --band_pool measured (after crop, per-
    # frequency z-score and instance norm). Pooling earlier would invalidate the (C, F)
    # z-score stats, which are defined at full width. Skipped when band_pool <= 0.
    _bp = int(getattr(cfg, "band_pool", 0) or 0)
    if _bp > 0:
        F = spec.shape[-2]
        if F % _bp:
            raise ValueError(f"band_pool {_bp} must divide freq_bins {F}")
        spec = spec.reshape(*spec.shape[:-2], _bp, F // _bp, spec.shape[-1]).mean(dim=-2)
    return spec


def spectro_frame_mask(
    raw: ArrayLike,
    nan_mask: ArrayLike,
    cfg: SpectroCodecConfig,
) -> torch.Tensor:
    """Per-(channel, STFT frame) validity mask of one raw window, ``(C, cfg.time_frames)``.

    The spectro counterpart of the video ``channel_valid`` flag and the slow-TS ``valid``
    mask: 1.0 where the (channel, frame) is real data, 0.0 where it is fill.

    Parameters
    ----------
    raw : (C, W)
        The raw window as ``TokamakH5Dataset._load_signal_raw`` returns it (at
        ``STFT_FS``, NaN replaced by 0, out-of-range positions zero-padded).
    nan_mask : (C, W)
        The loader's mask, 1.0 where the HDF5 value was NaN.
    cfg : SpectroCodecConfig
        Gives the per-codec STFT grid (``stft_n_fft`` / ``stft_hop``, defaulting to the
        module globals) and ``time_frames``.

    Returns
    -------
    (C, cfg.time_frames) float32 tensor of 1.0 / 0.0.

    Notes
    -----
    The loader marks absence in two ways, so there are two rules:

    1. NaN: the diagnostic recorded, but the value is NaN. Projected to STFT frames with
       the production rule (``data_loader._raw_to_frame_mask``): a frame is invalid if
       any sample in its ``n_fft``-wide support is, computed as a ``max_pool1d`` of the
       invalid indicator with ``kernel=n_fft, stride=hop, padding=n_fft//2``, the
       support ``torch.stft(center=True)`` reads. Using the same rule keeps the codec's
       notion of missing identical to the foundation model's.
    2. All-zero support: the diagnostic did not record. ``_load_signal_raw`` zero-fills
       an absent or empty HDF5 group, a zero-slab channel and any part of the window
       outside ``[xdata[0], xdata[-1]]``, without a NaN flag, so rule 1 calls them valid.
       A real 500 kHz digitizer trace is never exactly 0.0 over a whole 1024-sample
       support, so ``max|raw| == 0`` over the support detects these cases without false
       positives. This is the slow-TS ``zero_is_missing`` policy at frame level, so a
       single zero-crossing sample cannot trigger it.

    A frame is valid when neither rule fires and its raw support is finite. The time
    crop and pad follow :func:`_crop_pad_freq_time`; padded frames are invalid.
    """
    x = _as_tensor(raw)
    m = _as_tensor(nan_mask)
    if x.dim() != 2:
        raise ValueError(f"spectro_frame_mask expects raw (C, W); got {tuple(x.shape)}")
    if m.shape != x.shape:
        raise ValueError(
            f"spectro_frame_mask: nan_mask {tuple(m.shape)} != raw {tuple(x.shape)}"
        )
    n_fft = int(getattr(cfg, "stft_n_fft", STFT_N_FFT))
    hop = int(getattr(cfg, "stft_hop", STFT_HOP))

    invalid = ((m >= 0.5) | ~torch.isfinite(x)).to(torch.float32).unsqueeze(0)  # (1, C, W)
    # Rule 1: production _raw_to_frame_mask (max over each frame's n_fft support).
    inv_f = torch.nn.functional.max_pool1d(
        invalid, kernel_size=n_fft, stride=hop, padding=n_fft // 2
    ).squeeze(0)                                                    # (C, n_frames)
    # Rule 2: the same support on |raw|; an all-zero support is fill.
    absx = torch.nan_to_num(x.abs(), nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(0)
    max_f = torch.nn.functional.max_pool1d(
        absx, kernel_size=n_fft, stride=hop, padding=n_fft // 2
    ).squeeze(0)                                                    # (C, n_frames)
    valid = (inv_f < 0.5) & (max_f > 0.0)

    # Time crop or right-pad to cfg.time_frames; padded frames are invalid.
    C, n_frames = valid.shape
    T = int(cfg.time_frames)
    if n_frames >= T:
        valid = valid[:, :T]
    else:
        valid = torch.cat([valid, valid.new_zeros((C, T - n_frames))], dim=1)
    return valid.to(torch.float32)


def raw_pair_windows(
    raw_shot: ArrayLike,
    t0: float,
    cfg: SpectroCodecConfig,
    delta_ms: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the two raw windows of a δ-shift consistency pair (no STFT).

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
    """Build a δ-shift consistency pair of log-power spectrograms from a raw shot.

    See ``docs/IGNITE_DESIGN.md`` §4.2. The two windows share almost all mode content
    but differ in sub-window phase realization, so ``|enc(spec_a) - enc(spec_b)|^2``
    trains shift-invariant features.

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
        If True, also return the (possibly sampled) ``delta_ms``.
    seed : int or None, keyword-only
        Seeds the δ draw (only used when ``delta_ms is None``).

    Returns
    -------
    (spec_a, spec_b)                          if ``return_delta`` is False
    (spec_a, spec_b, delta_ms)                if ``return_delta`` is True
        ``spec_*`` are ``(C, cfg.freq_bins, cfg.time_frames)`` log-power tensors, with
        ``C`` taken from ``raw_shot``.
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


# Fast-TS (filterscopes) window transforms and δ-shift pairs.
#
# Two codec targets, chosen by cfg.target. "envelope" (the default) is the ELM activity
# envelope from elm_envelope: detrend, rectify, RMS-pool, log1p. "raw" (fastts_raw_window)
# keeps the 10 kHz samples themselves, sanitized and standardized per channel, (B, C, W)
# in and out.
#
# Both use the per-channel standardization (x - mean) / std that the data loader also
# applies to filterscopes (SignalConfig preprocess method="standardize"). It is affine per
# channel, so it loses no information; it brings the ~1e15-magnitude raw signal to O(1),
# and the gate's nRMSE (divided by each (window, channel) std) does not change under it.


def _standardize_channels(x: torch.Tensor, cfg: FastTSCodecConfig) -> torch.Tensor:
    """Standardize a (B, C, W) raw window per channel with ``cfg.channel_mean/std``.

    Same as the ``method="standardize"`` branch of ``data_loader._apply_preprocessing``:
    ``(x - mean) / std.clamp(min=1e-3)`` with the per-channel raw mean and std the
    foundation model uses for ``filterscopes``. The stats are global per channel, not
    per window, so the relative ELM activity level is kept (quiet windows stay small).
    If either stat is ``None`` this is the identity. Both must have length ``C``.
    """
    if cfg.channel_mean is None or cfg.channel_std is None:
        return x
    C = x.shape[-2]
    mean = torch.as_tensor(cfg.channel_mean, dtype=x.dtype, device=x.device)
    std = torch.as_tensor(cfg.channel_std, dtype=x.dtype, device=x.device)
    if mean.numel() != C or std.numel() != C:
        raise ValueError(
            f"fastts channel stats length {mean.numel()}/{std.numel()} != C={C}"
        )
    mean = mean.reshape(1, C, 1)
    std = std.reshape(1, C, 1).clamp(min=1e-3)  # matches data_loader std.clamp(min=1e-3)
    return (x - mean) / std


_FASTTS_ENV_FLOOR: float = 0.0        # log1p(0) = 0: a silent (no-activity) envelope
_FASTTS_ENV_CEIL: float = 30.0        # upper bound on log1p(env/eps)


def _moving_mean(x: torch.Tensor, win: int) -> torch.Tensor:
    """Centered moving mean along the last axis, box filter of length ``win``.

    ``x`` is (..., W); returns the same shape; ``win <= 1`` is the identity. Gives the
    slow baseline removed before rectification. Reflect padding keeps the ends from
    being pulled toward zero.
    """
    if win <= 1:
        return x
    *lead, W = x.shape
    flat = x.reshape(-1, 1, W)                                   # (N, 1, W)
    pad = win // 2
    # Reflect padding needs pad < W; clamp for tiny test windows.
    pad = min(pad, max(0, W - 1))
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (pad, pad), mode="reflect")
    kernel = torch.ones(1, 1, win, dtype=flat.dtype, device=flat.device) / float(win)
    out = torch.nn.functional.conv1d(flat, kernel)              # valid conv on padded input
    # The padded conv gives W - win + 1 + 2*pad samples; crop or pad to W.
    out = out[..., :W] if out.shape[-1] >= W else torch.nn.functional.pad(
        out, (0, W - out.shape[-1]), mode="replicate"
    )
    return out.reshape(*lead, W)


def elm_envelope(raw: ArrayLike, cfg: FastTSCodecConfig) -> torch.Tensor:
    """ELM activity envelope of raw filterscope windows, the envelope codec target (§4.3).

    Parameters
    ----------
    raw : (B, C, W) tensor or ndarray
        Raw multi-channel filterscope windows at ``FASTTS_FS``. Any ``W`` of at least
        ``cfg.pool`` samples works: the pooling drops the trailing remainder, and the
        result is cropped or padded with the silence floor to ``cfg.env_bins``.
    cfg : FastTSCodecConfig
        Provides ``pool`` (RMS bin size), ``baseline_win`` (detrend window), ``env_eps``
        (log1p reference) and ``env_bins`` (output length).

    Returns
    -------
    (B, C, cfg.env_bins) float32 tensor
        The log1p-compressed per-bin RMS envelope, with non-finite values replaced and
        clamped.
    """
    x = _as_tensor(raw)
    if x.dim() != 3:
        raise ValueError(f"elm_envelope expects (B, C, W); got shape {tuple(x.shape)}")
    # As in log_power_stft, set non-finite and absurd samples to 0, before the
    # standardization so sentinels cannot spoil it.
    x = torch.where(torch.isfinite(x) & (x.abs() < 1e20), x, torch.zeros_like(x))

    # 0. Per-channel standardization (FastTSCodecConfig.channel_mean/std). The raw signal
    # has per-channel std ~1e16; without this, log1p(rms/env_eps) sits at the clamp ceiling
    # in every window, the envelope is constant and the codec collapses.
    x = _standardize_channels(x, cfg)

    # 1. Detrend: subtract the slow moving-mean baseline.
    if cfg.baseline_win and cfg.baseline_win > 1:
        x = x - _moving_mean(x, int(cfg.baseline_win))

    # 2. Rectify.
    energy = x.pow(2)

    # 3. RMS-pool over non-overlapping `pool`-sample bins, dropping the remainder.
    pool = int(cfg.pool)
    B, C, W = energy.shape
    n_bins = W // pool
    if n_bins < 1:
        raise ValueError(f"elm_envelope: window W={W} shorter than pool={pool}")
    energy = energy[..., : n_bins * pool].reshape(B, C, n_bins, pool)
    rms = energy.mean(dim=-1).clamp_min(0.0).sqrt()             # (B, C, n_bins) per-bin RMS

    # 4. Compress: log1p(rms / eps).
    env = torch.log1p(rms / float(cfg.env_eps))

    # Sanitize and clamp, then crop or pad to cfg.env_bins.
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



def fastts_raw_window(raw: ArrayLike, cfg: FastTSCodecConfig) -> torch.Tensor:
    """Raw filterscope samples, sanitized and standardized per channel; the raw codec target.

    Unlike :func:`elm_envelope` it does not summarize: all ``W`` samples per channel are
    kept.

    Parameters
    ----------
    raw : (B, C, W_in) tensor or ndarray
        Raw multi-channel filterscope windows at ``FASTTS_FS``, cropped or zero-padded
        to ``cfg.window`` samples.
    cfg : FastTSCodecConfig
        Gives ``channel_mean`` / ``channel_std`` (``None`` means no standardization)
        and ``window``, the output length.

    Returns
    -------
    (B, C, cfg.window) float32 tensor
        Sanitized, standardized raw samples. Non-finite and absurd sentinel samples are
        set to 0 before standardization, as in ``log_power_stft``.
    """
    x = _as_tensor(raw)
    if x.dim() != 3:
        raise ValueError(f"fastts_raw_window expects (B, C, W); got shape {tuple(x.shape)}")
    # Set non-finite and absurd samples to 0 before standardizing, so sentinels cannot
    # spoil the standardized signal.
    x = torch.where(torch.isfinite(x) & (x.abs() < 1e20), x, torch.zeros_like(x))
    # Per-channel standardization is the only transform; it is affine.
    x = _standardize_channels(x, cfg)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return _crop_pad_window(x.to(torch.float32), int(cfg.window))


def _crop_pad_window(x: torch.Tensor, window: int) -> torch.Tensor:
    """Crop or right-pad the (..., W_in) raw window to (..., window) with zeros."""
    W = x.shape[-1]
    if W == window:
        return x
    if W > window:
        return x[..., :window]
    pad = x.new_zeros((*x.shape[:-1], window - W))
    return torch.cat([x, pad], dim=-1)


def fastts_raw_pair_windows(
    raw_shot: ArrayLike,
    t0: float,
    cfg: FastTSCodecConfig,
    delta_ms: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the two raw filterscope windows of a δ-shift consistency pair.

    Fast-TS version of :func:`raw_pair_windows`: window A is ``[t0, t0 + CHUNK_S]``,
    window B is shifted by ``δ = round(delta_ms * FASTTS_FS / 1000)`` samples, at the
    filterscope rate rather than the STFT rate. Both are ``cfg.window_samples`` long.
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
    """Build a δ-shift pair of filterscope windows (envelope or raw, per ``cfg``).

    Window A starts at ``t0``; window B is the same window shifted by δ. For the envelope
    codec both carry the same statistic and feed a consistency loss. For the raw codec
    they do not (a shift moves every sample), so ``FastTSCodecConfig.consistency_weight``
    defaults to 0 and the pair only lets the gate report ``stability``, how much the codes
    move under a shift. That number is expected to be low; ``spike.gate_score`` ignores
    it.

    Returns
    -------
    (win_a, win_b)               if ``return_delta`` is False
    (win_a, win_b, delta_ms)     if ``return_delta`` is True
        ``win_*`` are ``(C, E)`` envelopes or ``(C, cfg.window)`` raw windows.
    """
    if delta_ms is None:
        lo, hi = cfg.consistency_delta_ms
        gen = np.random.default_rng(seed)
        delta_ms = float(gen.uniform(lo, hi))

    win_a, win_b = fastts_raw_pair_windows(raw_shot, t0=t0, cfg=cfg, delta_ms=delta_ms)
    # Envelope mode (the default) or standardized raw samples. cfg.is_raw is False for
    # old pickled configs, which have no `target` field.
    fn = fastts_raw_window if cfg.is_raw else (lambda w, c: elm_envelope(w, c))
    win_a = fn(win_a.unsqueeze(0), cfg)[0]
    win_b = fn(win_b.unsqueeze(0), cfg)[0]

    if return_delta:
        return win_a, win_b, delta_ms
    return win_a, win_b


# Getting raw windows from the real data loader
#
# The synthetic tests do not touch HDF5. For real data, the raw window for
# shift_pair_windows / log_power_stft comes from the existing pipeline in
# ``src/tokamak_foundation_model/data/data_loader.py``, used unchanged:
#
#   ds = TokamakSingleFileDataset(hdf5_path, ...)          # opens one shot
#   cfg_sig = <the SignalConfig for the modality, e.g. the ece entry in ds's configs>
#   raw, valid_len, nan_mask = ds._load_signal_raw(         # (C, T) at cfg_sig.target_fs
#       ds.h5_file, cfg_sig, t_start=t0, t_end=t0 + CHUNK_S + delta_s)
#
# ``_load_signal_raw`` already resamples to ``target_fs`` (500 kHz for the spectro
# modalities, equal to STFT_FS) and zero-pads / NaN-fills, so its (C, T) output is the
# raw window this module transforms. For a δ-pair, read the extended span
# ``[t0, t0 + CHUNK_S + delta_s]`` once and slice both sub-windows from it (or call
# raw_pair_windows on that span with t0 rebased to 0).
#
# Two caveats:
#   1. ``_load_signal_raw`` is a private method taking an open h5py handle
#      (``ds.h5_file``) and a per-modality ``SignalConfig``; there is no public entry
#      point for raw samples of a modality over [a, b], so keep the call site pinned.
#   2. ``__getitem__`` returns the processed STFT (magnitude, log10(x+1), standardized),
#      not the raw signal and not log-power, so the codec target has to be computed with
#      ``log_power_stft`` from the raw window.
