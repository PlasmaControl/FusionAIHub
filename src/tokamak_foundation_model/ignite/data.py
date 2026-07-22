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
    STFT_FS,
    STFT_HOP,
    STFT_N_FFT,
    SpectroCodecConfig,
)

ArrayLike = Union[torch.Tensor, np.ndarray]

# Floor added inside the log to keep log-power finite where the STFT magnitude is ~0
# (silent bins / padded tail). log10(0 + eps) is finite and large-negative.
_LOG_EPS: float = 1e-10


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
    window = torch.hann_window(STFT_N_FFT, dtype=x.dtype, device=x.device)
    spec = _stft_log_power(x, window)  # (B, C, n_fft//2, n_frames)
    return _crop_pad_freq_time(spec, cfg.freq_bins, cfg.time_frames)


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
