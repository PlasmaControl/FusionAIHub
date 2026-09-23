"""Decode rollout tokens into features that can be scored and plotted.

The dynamics model predicts codec tokens, and a token id carries no magnitude. Decoding
through each modality's own frozen codec (`shotdb.ignite.load_codecs`) puts every arm and
the measurement in the codec's normalised reconstruction space, where they compare directly
(`eval_dynamics.decode_all`: no denormalisation is applied, nor needed). Each family is then
reduced to one ``(F, C)`` feature per rollout:

  * spectro   -> mean z in a band (`band_power`): 10-60 kHz for mhr and mirnov, the full
                 band otherwise.
  * video     -> one frame mean, ``(F, 1)``.
  * slowts /
    fastts    -> the mean over the intra-frame time axis.
"""

from __future__ import annotations

import numpy as np
import torch

from tokamak_foundation_model.ignite import eval_dynamics
from tokamak_foundation_model.ignite.config import STFT_FS, STFT_N_FFT

from .core import Ensemble

# The two modalities whose band is 10-60 kHz (the Alfven-eigenmode range).
_BANDED_SPECTRO = ("mhr", "mirnov")
_DEFAULT_BAND_KHZ = (10.0, 60.0)
_CHUNK = 16  # frames per codec call


def freq_axis_khz(cfg) -> np.ndarray:
    """The centre frequency (kHz) of each row a spectro codec decodes, low to high.

    ``ignite.data.log_power_stft`` drops the DC bin and keeps the low ``cfg.freq_bins``
    bins, so row ``i`` is STFT bin ``i + 1``, at ``(i + 1) * STFT_FS / n_fft``. With
    ``cfg.band_pool > 0`` it then mean-pools them into ``band_pool`` equal bands, and each
    band sits at the mean frequency of its bins.
    """
    n_fft = int(getattr(cfg, "stft_n_fft", STFT_N_FFT))
    hz = np.arange(1, int(cfg.freq_bins) + 1) * STFT_FS / n_fft
    pool = int(getattr(cfg, "band_pool", 0) or 0)
    if pool > 0:
        hz = hz.reshape(pool, -1).mean(axis=1)
    return hz / 1000.0


def band_power(
    dec: np.ndarray, freq_khz: np.ndarray, band: tuple[float, float] | None
) -> np.ndarray:
    """``(F, C, Fr, Tb)`` decoded spectrogram -> ``(F, C)`` mean z in ``band`` (kHz).

    The full band (an ordinary signed mean) when ``band`` is None or holds no row.
    """
    mask = np.ones(dec.shape[2], dtype=bool)
    if band is not None:
        inside = (freq_khz >= band[0]) & (freq_khz <= band[1])
        mask = inside if inside.any() else mask
    return dec[:, :, mask, :].mean(axis=(2, 3)).astype(np.float32)


def _reduce_video(dec: np.ndarray) -> np.ndarray:
    """``(F, ...)`` decoded video -> ``(F, 1)`` per-frame mean."""
    f = dec.shape[0]
    return dec.reshape(f, -1).mean(axis=1, keepdims=True).astype(np.float32)


def _reduce_series(dec: np.ndarray) -> np.ndarray:
    """``(F, C, Tt)`` slow/fast time series -> ``(F, C)`` intra-frame mean."""
    if dec.ndim <= 2:
        return dec.astype(np.float32)
    return dec.mean(axis=tuple(range(2, dec.ndim))).astype(np.float32)


def feature_name(name: str, family: str) -> str:
    """What `features` reduces a modality to, in words."""
    if family == "spectro":
        return "10-60 kHz band power" if name in _BANDED_SPECTRO else "band power"
    return "frame mean" if family == "video" else "intra-frame mean"


@torch.no_grad()
def features(codec, cfg, family: str, name: str, codes: torch.Tensor) -> np.ndarray:
    """Codes ``(..., F, n_tok)`` -> the decoded feature ``(..., F, C)``.

    Each chunk is reduced as soon as it is decoded, so a rollout's full spectrogram (5 MiB a
    frame for mirnov) is never held at once. The decode runs on the codec's device.
    """
    device = next(codec.parameters()).device
    if family == "spectro":
        axis = freq_axis_khz(cfg)
        band = _DEFAULT_BAND_KHZ if name in _BANDED_SPECTRO else None

        def reduce(dec):
            return band_power(dec, axis, band)
    elif family == "video":
        reduce = _reduce_video
    else:
        reduce = _reduce_series
    flat = codes.reshape(-1, codes.shape[-1])
    parts = [
        reduce(eval_dynamics.decode_flat(codec, flat[i : i + _CHUNK].to(device).long())
               .float().cpu().numpy())
        for i in range(0, flat.shape[0], _CHUNK)
    ]
    out = np.concatenate(parts)
    return out.reshape(*codes.shape[:-1], out.shape[-1])


def decode_ensemble(codecs: dict, ens: Ensemble) -> dict[str, dict[str, np.ndarray]]:
    """``{m: {"gt": (F, C), arm: (M, F, C)}}`` for each modality with a codec, held ones
    excepted (a placeholder has nothing to score)."""
    out = {}
    for name, (codec, cfg, family) in codecs.items():
        if name in ens.held or name not in ens.gt:
            continue
        out[name] = {"gt": features(codec, cfg, family, name, ens.gt[name])} | {
            arm: features(codec, cfg, family, name, tokens[name])
            for arm, tokens in ens.arms.items()
        }
    return out
