"""Decode a paired rollout's tokens into comparable, physically-meaningful units.

``core.run_paired`` scores rollouts in TOKEN space (fraction of matching codes) because
that is what the dynamics model actually predicts. The report needs to SHOW the
rollout too, and a token id carries no magnitude -- decoding through the modality's
own frozen Phase-A codec (`shotdb.ignite.load_codecs`) puts real/proposed/gt back in
the codec's normalized reconstruction space, which is directly comparable across the
three arms (they share the same codec) even though it is not the original physical
unit (`eval_dynamics.decode_all`'s docstring: "no denormalization is applied (nor
needed)" for the same reason).

Three codec families decode to three different tensor shapes, and each is reduced to
a single `(F, C)` (or `(F, 1)` for video) series per arm for the panel/report -- a
per-frame scalar per channel is what a line plot and a markdown table both want:

  * spectro   -> `(F, C, freq_bins, time_frames)`, reduced to mean |value| in a
                 frequency band (`band_power`); `mhr`/`mirnov` use the 10-60 kHz band
                 the brief names, every other spectro modality gets the full band
                 (no named band exists for it).
  * video     -> `(F, C, Tv, H, W)`, reduced to one per-frame scalar (channel/space/
                 time mean) since a rollout-quality panel needs a trend line, not a
                 movie.
  * slowts /
    fastts    -> `(F, C, Tt)`, reduced by averaging the intra-frame time axis.
"""

from __future__ import annotations

import numpy as np
import torch

from tokamak_foundation_model.ignite import eval_dynamics
from tokamak_foundation_model.ignite.config import STFT_FS

from .core import ARM_LABELS, SimulationArms

# Named in the D2 brief as the two modalities whose panel/report band is 10-60 kHz.
_BANDED_SPECTRO = ("mhr", "mirnov")
_DEFAULT_BAND_KHZ = (10.0, 60.0)


def freq_axis_khz(cfg) -> np.ndarray | None:
    """A spectro codec's per-bin center frequency (kHz), low->high, or None.

    ``ignite.data.log_power_stft`` builds the codec's input by STFT-ing at
    `STFT_FS` (500 kHz), dropping the DC bin (bin 0), then keeping the LOW
    `cfg.freq_bins` of the remaining bins unchanged (`_crop_pad_freq_time`:
    "cropped from the low end ... ordered low->high"). So array index ``i``
    (0-indexed) is raw STFT bin ``i + 1``, at frequency
    ``(i + 1) * STFT_FS / cfg.stft_n_fft``.

    That 1:1 index<->bin mapping breaks once ``cfg.band_pool > 0``
    (`log_power_stft` mean-pools ``freq_bins`` STFT bins into ``band_pool``
    unevenly-sized bands as the LAST step) -- there is no way to recover which
    physical band a pooled bin covers from ``cfg`` alone, so this returns None
    and callers fall back to a full-band reduction.
    """
    n_fft = getattr(cfg, "stft_n_fft", None)
    freq_bins = getattr(cfg, "freq_bins", None)
    band_pool = getattr(cfg, "band_pool", 0)
    if not n_fft or not freq_bins or band_pool:
        return None
    bin_hz = STFT_FS / float(n_fft)
    idx = np.arange(1, int(freq_bins) + 1, dtype=np.float64)
    return idx * bin_hz / 1000.0


def band_power(
    dec: np.ndarray,
    freq_khz: np.ndarray | None,
    band: tuple[float, float] | None,
) -> np.ndarray:
    """``(F, C, Fr, Tb)`` decoded spectrogram -> ``(F, C)`` mean |value| in a band.

    Falls back to the FULL frequency axis (every bin, i.e. an ordinary
    band-power mean) when ``freq_khz`` is None (see :func:`freq_axis_khz`) or
    ``band`` is None, or when a given band selects no bin at all (a
    too-coarse ``freq_khz`` for the requested band).
    """
    if freq_khz is None or band is None:
        mask = np.ones(dec.shape[2], dtype=bool)
    else:
        lo, hi = band
        mask = (freq_khz >= lo) & (freq_khz <= hi)
        if not mask.any():
            mask = np.ones(dec.shape[2], dtype=bool)
    sub = dec[:, :, mask, :]
    return np.mean(np.abs(sub), axis=(2, 3)).astype(np.float32)


def _reduce_video(dec: np.ndarray) -> np.ndarray:
    """``(F, ...)`` decoded video -> ``(F, 1)`` per-frame mean.

    Raw decoded video frames are the largest arrays in this pipeline and only
    a trend matters for the panel/report, so they are never stored past this
    reduction (see ``report.write``'s ``reduction`` h5 attr, which discloses
    this).
    """
    f = dec.shape[0]
    return dec.reshape(f, -1).mean(axis=1, keepdims=True).astype(np.float32)


def _reduce_series(dec: np.ndarray) -> np.ndarray:
    """``(F, C, Tt)`` slow/fast time series -> ``(F, C)`` intra-frame mean."""
    if dec.ndim <= 2:
        return dec.astype(np.float32)
    return dec.mean(axis=tuple(range(2, dec.ndim))).astype(np.float32)


def _device_of(codec: torch.nn.Module, arms: SimulationArms) -> torch.device:
    """Where to run the decode -- the CODEC's device, not the arms' tensors' device.

    ``load_codecs`` places each codec on whatever device the caller asked for
    independently of where a caller's token tensors happen to live (e.g. D4 on
    Frontier runs `--device cuda` for the codec while arms built earlier may
    still be on cpu); `decode_flat_chunked` moves `flat` `.to(device)` before
    calling `codec.decode`, so `device` must be the codec's own device or that
    call raises a device-mismatch RuntimeError. Falls back to the arms' own
    tensor device only for a codec with no parameters (untestable edge case,
    kept for robustness).
    """
    try:
        return next(codec.parameters()).device
    except StopIteration:
        pass
    for arm in (arms.real, arms.proposed, arms.gt):
        for t in arm.values():
            return t.device
    return torch.device("cpu")


def decode_modalities(
    codecs: dict[str, tuple], arms: SimulationArms, names: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    """Decode real/proposed/gt codes for each of ``names`` through frozen codecs.

    Returns ``{m: {"real": (F, C), "proposed": (F, C), "gt": (F, C)}}``
    (video: ``(F, 1)``). A name absent from ``codecs`` (no frozen codec
    loaded for it -- placeholder or simply not requested) is skipped
    entirely, matching ``eval_dynamics.decode_all``'s convention.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        entry = codecs.get(name)
        if entry is None:
            continue
        codec, cfg, family = entry
        device = _device_of(codec, arms)
        per_arm: dict[str, np.ndarray] = {}
        arms_by_label = ((label, getattr(arms, label)) for label in ARM_LABELS)
        for arm_name, codes in arms_by_label:
            flat = codes.get(name)
            if flat is None:
                continue
            raw = eval_dynamics.decode_flat_chunked(codec, flat, device)
            if family == "spectro":
                band = _DEFAULT_BAND_KHZ if name in _BANDED_SPECTRO else None
                per_arm[arm_name] = band_power(raw, freq_axis_khz(cfg), band)
            elif family == "video":
                per_arm[arm_name] = _reduce_video(raw)
            else:  # slowts, fastts
                per_arm[arm_name] = _reduce_series(raw)
        out[name] = per_arm
    return out
