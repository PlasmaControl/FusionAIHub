"""IGNITE Phase-A gate spike: a small training and gate probe for the spectro codec.

The minimal end-to-end run of a Phase-A codec against the gate
(``docs/IGNITE_DESIGN.md`` §4.4):

1. Train a :class:`~ignite.codec.SpectroCodec` with GAN alternation (a generator
   step on the codec, a discriminator step on an external
   :class:`~ignite.discriminator.FreqAwarePatchGAN`, both Adam).
2. Every ``eval_every`` steps, compute the gate on held-out batches: stability
   (codes of ``x`` against its δ-shifted pair), persistence and forecastability
   (codes over consecutive frames) and decode fidelity (recon against target).
3. Log one line per evaluation and return the final gate dict.

With ``--synthetic`` everything runs on CPU on sum-of-sinusoid signals fed through
``data.py``, so the smoke test needs no HDF5. :func:`real_ece_batches` and
:func:`real_frame_sequence` read real ece windows through the HDF5 loader. The
DDP codec trainer (``train_codec.py``) reuses the gate, step, guard and checkpoint
helpers defined here.

Following §7 this uses only ``torch`` and sibling ``ignite`` modules, no FAITH model
code. Forecastability uses the gate's default code-derived transition split; there
is no physics transition detector here.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

from . import data, gate
from .codec import SpectroCodec
from .config import CHUNK_S, SpectroCodecConfig
from .discriminator import FreqAwarePatchGAN
from .losses import discriminator_loss


# Synthetic data
def _sum_of_sinusoids(
    n_samples: int,
    freqs_hz: Iterable[float],
    fs: float,
    noise_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """One channel of raw signal, shape (n_samples,): a sum of sinusoids plus noise.

    Phases are drawn per call, so a δ-shift changes the sub-window realization while
    the frequencies (the statistic) stay the same.
    """
    t = torch.arange(n_samples, dtype=torch.float32) / fs
    x = torch.zeros(n_samples, dtype=torch.float32)
    for f in freqs_hz:
        phase = 2.0 * math.pi * float(torch.rand((), generator=generator))
        x = x + torch.sin(2.0 * math.pi * float(f) * t + phase)
    if noise_std > 0.0:
        x = x + noise_std * torch.randn(n_samples, generator=generator)
    return x


def _random_shot(
    cfg: SpectroCodecConfig,
    freqs_hz: List[float],
    span_windows: int,
    noise_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """A raw (C, W_full) shot long enough to cut δ-shift pairs from.

    Each channel is an independent realization of the same mode set: the frequency
    statistic is shared across channels and only the phases differ.
    """
    w_full = cfg.window_samples * span_windows
    chans = [
        _sum_of_sinusoids(w_full, freqs_hz, data.STFT_FS, noise_std, generator)
        for _ in range(cfg.channels)
    ]
    return torch.stack(chans, dim=0)  # (C, W_full)


def synthetic_batches(
    cfg: SpectroCodecConfig,
    n_batches: int,
    batch_size: int,
    device: str | torch.device = "cpu",
    *,
    delta_ms: Optional[float] = None,
    noise_std: float = 0.01,
    base_freqs_hz: Optional[List[float]] = None,
    seed: int = 0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Build ``n_batches`` of δ-shift consistency pairs ``(spec_a, spec_b)``.

    Each element is a pair of log-power spectrogram batches of shape
    ``(batch_size, C, freq_bins, time_frames)``, made from synthetic sum-of-sinusoid
    signals via :func:`data.shift_pair_windows`.

    ``spec_a`` is the window ``[t0, t0 + CHUNK_S]`` and ``spec_b`` the δ-shifted window
    ``[t0 + δ, t0 + CHUNK_S + δ]`` of the same shot: the same modes, a different
    sub-window realization. δ is ``delta_ms`` if given, else the midpoint of
    ``cfg.consistency_delta_ms``, which keeps the smoke test deterministic.

    Notes
    -----
    Each sample in a batch has a slightly different mode set and start time, so the
    batch is not degenerate. The list is materialized, so callers can iterate it
    again as held-out data.
    """
    gen = torch.Generator().manual_seed(seed)
    if base_freqs_hz is None:
        # A few plausible "mode" frequencies well inside the 0..fs/2 = 250 kHz band.
        base_freqs_hz = [40e3, 90e3, 150e3]
    # deterministic default δ = midpoint of the config range (smoke-test friendly)
    default_delta = 0.5 * (cfg.consistency_delta_ms[0] + cfg.consistency_delta_ms[1])

    # Enough raw span to hold [t0, t0 + CHUNK_S + δ] with room for varied t0.
    span_windows = 4

    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n_batches):
        a_list: List[torch.Tensor] = []
        b_list: List[torch.Tensor] = []
        for _ in range(batch_size):
            # jitter the mode set per-sample so the batch has diversity
            jitter = float(torch.rand((), generator=gen)) * 5e3
            freqs = [f + jitter for f in base_freqs_hz]
            shot = _random_shot(cfg, freqs, span_windows, noise_std, gen)
            # t0 leaves headroom before the end for the δ-shifted window
            t0 = data.CHUNK_S * (0.5 + float(torch.rand((), generator=gen)))
            d = default_delta if delta_ms is None else delta_ms
            a, b = data.shift_pair_windows(shot, t0=t0, cfg=cfg, delta_ms=d)
            a_list.append(a)
            b_list.append(b)
        spec_a = torch.stack(a_list, dim=0).to(device)
        spec_b = torch.stack(b_list, dim=0).to(device)
        batches.append((spec_a, spec_b))
    return batches


def _consecutive_frame_sequence(
    cfg: SpectroCodecConfig,
    batch_size: int,
    n_frames: int,
    device: str | torch.device = "cpu",
    *,
    noise_std: float = 0.01,
    seed: int = 123,
) -> torch.Tensor:
    """A batch of consecutive frames for persistence and forecastability.

    Returns ``(batch_size, n_frames, C, freq_bins, time_frames)``. Frames are
    consecutive 50 ms windows of a synthetic shot whose mode frequencies drift slowly,
    a predictable transition on which the Markov probe should beat persistence
    (§4.4), while the shared structure keeps persistence high.
    """
    gen = torch.Generator().manual_seed(seed)
    base_freqs = [40e3, 90e3, 150e3]
    drift_per_frame = 1.5e3  # slow, predictable upward drift

    seqs: List[torch.Tensor] = []
    for _ in range(batch_size):
        frames: List[torch.Tensor] = []
        for fi in range(n_frames):
            freqs = [f + fi * drift_per_frame for f in base_freqs]
            # one full window's worth of raw samples per frame
            chans = [
                _sum_of_sinusoids(cfg.window_samples, freqs, data.STFT_FS, noise_std, gen)
                for _ in range(cfg.channels)
            ]
            raw = torch.stack(chans, dim=0).unsqueeze(0)  # (1, C, W)
            spec = data.log_power_stft(raw, cfg)[0]  # (C, F, T)
            frames.append(spec)
        seqs.append(torch.stack(frames, dim=0))  # (n_frames, C, F, T)
    return torch.stack(seqs, dim=0).to(device)  # (B, n_frames, C, F, T)


# Real-data path (reads the existing HDF5 loader; data_loader.py is not modified)
#
# - The loader class is ``TokamakH5Dataset`` (data_loader.py), one instance per shot.
# - ``ds._load_signal_raw(ds.h5_file, cfg_sig, t_start, t_end)`` returns
#   ``(raw (C, T), valid_len, nan_mask (C, T))``, already resampled to
#   ``cfg_sig.target_fs`` (500 kHz for ece, equal to STFT_FS) and NaN/zero-filled.
#   It is an internal API that needs an open h5py handle, so ``ds._open_hdf5()`` is
#   called first.
# - For ece ``channels_to_use=slice(0, 40)``, so the raw output has 40 channels.
# - The loader zero-pads windows past the end of the data and reports in
#   ``valid_len`` how many target-rate samples are real; windows are checked on it.

# Default directory of the preprocessed shot HDF5 files (a Frontier path).
DEFAULT_DATA_DIR = "/lustre/orion/fus187/proj-shared/foundation_model"
_ECE_SIGNAL_NAME = "ece"


def discover_shots(
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    limit: Optional[int] = None,
) -> List[str]:
    """Enumerate shot ids from ``{shot}_processed.h5`` files in ``data_dir``.

    Returns the sorted list of shot-id strings (e.g. ``"200729"``). ``limit`` caps the
    count (first-N after sort) for quick probes.
    """
    d = Path(data_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"data_dir not found: {d}")
    shots = sorted(p.name[: -len("_processed.h5")] for p in d.glob("*_processed.h5"))
    if limit is not None:
        shots = shots[:limit]
    return shots


def _shot_path(shot: Union[str, int], data_dir: Union[str, Path]) -> Path:
    return Path(data_dir) / f"{shot}_processed.h5"


def _open_ece_dataset(hdf5_path: Union[str, Path]):
    """Open one shot's HDF5 via TokamakH5Dataset and return (ds, ece SignalConfig).

    Imported lazily so importing ``ignite.spike`` for the synthetic CPU smoke test
    never pulls in the heavier data-loader stack.
    """
    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

    ds = TokamakH5Dataset(
        hdf5_path,
        input_signals=[_ECE_SIGNAL_NAME],
        target_signals=[_ECE_SIGNAL_NAME],
    )
    ds._open_hdf5()
    ece_cfg = next(c for c in ds.signal_configs if c.name == _ECE_SIGNAL_NAME)
    return ds, ece_cfg


def _num_ece_channels(ece_cfg) -> int:
    """Channels actually loaded for ece after ``channels_to_use`` (== 40)."""
    if ece_cfg.channels_to_use is not None:
        return len(range(*ece_cfg.channels_to_use.indices(ece_cfg.num_channels)))
    return ece_cfg.num_channels


def _load_raw_span(
    ds,
    ece_cfg,
    t0: float,
    span_s: float,
    *,
    min_std: float,
) -> Optional[torch.Tensor]:
    """Load a raw ece window ``[t0, t0 + span_s]`` at STFT_FS; None if unusable.

    Returns None when the window runs past the real (non-padded) data, i.e.
    ``valid_len`` does not cover the span, when it holds non-finite values, or when
    it is near-flat (``std < min_std``).
    """
    t_end = t0 + span_s
    raw, valid_len, _nan = ds._load_signal_raw(ds.h5_file, ece_cfg, t_start=t0, t_end=t_end)
    need = round(span_s * ece_cfg.target_fs)
    if valid_len < need:
        return None  # window runs into padded / missing tail
    if not torch.isfinite(raw).all():
        return None
    if float(raw.std()) < min_std:
        return None  # all-zero / flat channel window
    return raw  # (C, T) float32 at STFT_FS


def real_ece_batches(
    cfg: SpectroCodecConfig,
    n_batches: int,
    batch_size: int,
    shots: Optional[Sequence[Union[str, int]]] = None,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    device: Union[str, torch.device] = "cpu",
    *,
    delta_ms: Optional[float] = None,
    t0_start: float = 1.0,
    t0_step: float = CHUNK_S,
    min_std: float = 1e-3,
    seed: int = 0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Real ece δ-shift consistency pairs from the existing HDF5 loader.

    Same structure as :func:`synthetic_batches`: a list of ``(spec_a, spec_b)``
    log-power spectrogram batches, each ``(batch_size, C, cfg.freq_bins,
    cfg.time_frames)``, where ``spec_b`` is the δ-shifted pair of ``spec_a`` (via
    :func:`data.shift_pair_windows`) and C is the ece channel count (40).

    Sets ``cfg.channels`` to the loaded ece channel count, so a codec and
    discriminator built from ``cfg`` match the data.

    Parameters
    ----------
    cfg : SpectroCodecConfig
        ``freq_bins`` / ``time_frames`` for the crop-pad, ``consistency_delta_ms`` for
        the δ draw when ``delta_ms`` is None. ``cfg.channels`` is set to 40.
    n_batches, batch_size : int
        Number of batches and pairs per batch.
    shots : sequence of shot ids, optional
        Shots to take windows from. Defaults to 8 shots found in ``data_dir``, with
        200729 first if present.
    data_dir : path
        Directory of ``{shot}_processed.h5`` files.
    device : str or torch.device
    delta_ms : float or None
        Sub-window shift in ms; None draws one per pair from
        ``cfg.consistency_delta_ms``.
    t0_start : float
        First window start in s. The default 1.0 s skips the ramp-up.
    t0_step : float
        Step between windows within a shot in s. The default ``CHUNK_S`` (50 ms) makes
        windows non-overlapping.
    min_std : float
        Minimum raw std for a window to count as non-degenerate.
    seed : int
        Seeds the per-pair δ draw.

    Returns
    -------
    list of (spec_a, spec_b)
        Each ``(batch_size, 40, cfg.freq_bins, cfg.time_frames)``, float32 on
        ``device``.

    Raises
    ------
    RuntimeError
        If there are not enough valid windows for ``n_batches * batch_size`` pairs.
    """
    channels = _fill_and_get_channels(cfg)
    n_needed = n_batches * batch_size

    # δ span we must have room for: A ends at CHUNK_S, B ends at CHUNK_S + δ.
    lo, hi = cfg.consistency_delta_ms
    max_delta_ms = hi if delta_ms is None else float(delta_ms)
    span_s = CHUNK_S + max_delta_ms / 1000.0

    shot_list = _resolve_shots(shots, data_dir)
    gen = torch.Generator().manual_seed(seed)

    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for shot in shot_list:
        if len(pairs) >= n_needed:
            break
        path = _shot_path(shot, data_dir)
        if not path.exists():
            continue
        ds, ece_cfg = _open_ece_dataset(path)
        assert _num_ece_channels(ece_cfg) == channels, "ece channel count changed"
        try:
            # walk windows within the shot's real duration
            end_time = min(ds.duration, _real_end_s(ds, ece_cfg))
            t0 = t0_start
            while t0 + span_s <= end_time and len(pairs) < n_needed:
                raw = _load_raw_span(ds, ece_cfg, t0, span_s, min_std=min_std)
                if raw is not None:
                    d = (
                        float(lo + (hi - lo) * torch.rand((), generator=gen))
                        if delta_ms is None
                        else float(delta_ms)
                    )
                    # rebase t0 to 0.0: `raw` already starts at the window's t0.
                    spec_a, spec_b = data.shift_pair_windows(
                        raw, t0=0.0, cfg=cfg, delta_ms=d
                    )
                    pairs.append((spec_a, spec_b))
                t0 += t0_step
        finally:
            _close_dataset(ds)

    if len(pairs) < n_needed:
        raise RuntimeError(
            f"real_ece_batches: found only {len(pairs)} valid windows across "
            f"{len(shot_list)} shots; need {n_needed} "
            f"(n_batches={n_batches} * batch_size={batch_size}). "
            "Add more shots, lower min_std, or shorten the span."
        )

    # pack into batches of (batch_size, C, F, T)
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for bi in range(n_batches):
        chunk = pairs[bi * batch_size : (bi + 1) * batch_size]
        spec_a = torch.stack([a for a, _ in chunk], dim=0).to(device)
        spec_b = torch.stack([b for _, b in chunk], dim=0).to(device)
        batches.append((spec_a, spec_b))
    return batches


def real_frame_sequence(
    cfg: SpectroCodecConfig,
    batch_size: int,
    n_frames: int,
    shots: Optional[Sequence[Union[str, int]]] = None,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    device: Union[str, torch.device] = "cpu",
    *,
    t0_start: float = 1.0,
    min_std: float = 1e-3,
) -> torch.Tensor:
    """Consecutive-frame ece sequences for persistence and forecastability.

    Returns ``(batch_size, n_frames, 40, cfg.freq_bins, cfg.time_frames)``: per
    sample, ``n_frames`` consecutive 50 ms (``CHUNK_S``) windows of a real shot, each
    turned into log-power by :func:`data.log_power_stft`. The trainer encodes this into
    the ``(B, n_frames, n_tok, fsq_dim)`` codes that ``gate.forecastability`` and
    ``gate.persistence`` take.

    A sequence is kept only if all its windows are valid. Shots are scanned with the
    start time stepping by ``n_frames * CHUNK_S``, so sequences do not overlap.

    Like :func:`real_ece_batches`, sets ``cfg.channels`` to 40.
    """
    channels = _fill_and_get_channels(cfg)
    seq_span_s = n_frames * CHUNK_S
    shot_list = _resolve_shots(shots, data_dir)

    seqs: List[torch.Tensor] = []
    for shot in shot_list:
        if len(seqs) >= batch_size:
            break
        path = _shot_path(shot, data_dir)
        if not path.exists():
            continue
        ds, ece_cfg = _open_ece_dataset(path)
        assert _num_ece_channels(ece_cfg) == channels, "ece channel count changed"
        try:
            end_time = min(ds.duration, _real_end_s(ds, ece_cfg))
            t0 = t0_start
            while t0 + seq_span_s <= end_time and len(seqs) < batch_size:
                frames: List[torch.Tensor] = []
                ok = True
                for fi in range(n_frames):
                    raw = _load_raw_span(
                        ds, ece_cfg, t0 + fi * CHUNK_S, CHUNK_S, min_std=min_std
                    )
                    if raw is None:
                        ok = False
                        break
                    spec = data.log_power_stft(raw.unsqueeze(0), cfg)[0]  # (C, F, T)
                    frames.append(spec)
                if ok:
                    seqs.append(torch.stack(frames, dim=0))  # (n_frames, C, F, T)
                t0 += seq_span_s
        finally:
            _close_dataset(ds)

    if len(seqs) < batch_size:
        raise RuntimeError(
            f"real_frame_sequence: found only {len(seqs)} valid {n_frames}-frame "
            f"sequences across {len(shot_list)} shots; need {batch_size}."
        )
    return torch.stack(seqs, dim=0).to(device)  # (B, n_frames, C, F, T)


# Helpers shared by the two real-data builders
def _fill_and_get_channels(cfg: SpectroCodecConfig) -> int:
    """Open a throwaway ece config to learn the real channel count, set cfg.channels."""
    channels = _ECE_CHANNELS_CACHE.get("n")
    if channels is None:
        from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

        ece_cfg = next(
            c for c in TokamakH5Dataset.SIGNAL_CONFIGS if c.name == _ECE_SIGNAL_NAME
        )
        channels = _num_ece_channels(ece_cfg)
        _ECE_CHANNELS_CACHE["n"] = channels
    cfg.channels = channels
    return channels


_ECE_CHANNELS_CACHE: Dict[str, int] = {}


def _resolve_shots(
    shots: Optional[Sequence[Union[str, int]]], data_dir: Union[str, Path]
) -> List[str]:
    if shots is not None:
        return [str(s) for s in shots]
    discovered = discover_shots(data_dir)
    # prefer 200729 (known-good, has modes) first, then fill from the front
    ordered: List[str] = []
    if "200729" in discovered:
        ordered.append("200729")
    for s in discovered:
        if s not in ordered:
            ordered.append(s)
        if len(ordered) >= 8:
            break
    return ordered


def _real_end_s(ds, ece_cfg) -> float:
    """Real (non-padded) ece data end time in seconds, from the shot's xdata."""
    for key_path in ece_cfg.hdf5_keys:
        grp = ds.h5_file
        ok = True
        for part in key_path.split("/"):
            if part in grp:
                grp = grp[part]
            else:
                ok = False
                break
        if ok and "xdata" in grp:
            return float(grp["xdata"][-1])
    return ds.duration


def _close_dataset(ds) -> None:
    h5 = getattr(ds, "h5_file", None)
    if h5 is not None:
        try:
            h5.close()
        except Exception:
            pass
        ds.h5_file = None


# Best-checkpoint selection (composite gate score) and an optional EMA of the codec

# Default reconstruction floor for disqualification; mirrors
# ``SpectroCodecConfig.gate_recon_floor``. Used when ``gate_score`` gets no explicit
# floor (e.g. in unit tests); callers with a cfg pass ``cfg.gate_recon_floor``.
DEFAULT_RECON_FLOOR: float = 0.2


def gate_score(gate_dict: Dict[str, object], recon_floor: float = DEFAULT_RECON_FLOOR,
               hard_min_codes: int = 8) -> float:
    """Composite selection score for a gate evaluation; higher is better.

    The score judges reconstruction, forecastability and utilization, not per-dim
    codebook entropy. A codec that reconstructs well but has one dead FSQ dimension
    (common with large codebooks: ``min_dim_entropy = 0``) is a good codec and stays
    selectable.

    The score is ``-inf`` (disqualified) only when reconstruction fails, i.e.
    ``decode["envelope_corr"]`` is NaN (a one-code codec has a flat envelope) or
    below ``recon_floor``, or when fewer than ``hard_min_codes`` distinct codes are
    used. The ``collapsed`` flag is not a hard gate, since it rejected codecs that
    reconstructed well (envelope_corr about 0.84); utilization enters only as a soft
    reward.

    Otherwise the score is a weighted sum of sub-metrics clamped to [0, 1]::

        recon   = clamp(envelope_corr, 0, 1)
        f1      = clamp(peak_f1, 0, 1)
        fcast   = clamp(margin_transition / MARGIN_SCALE, 0, 1)
        util    = 0.5 * clamp(min_dim_entropy, 0, 1)
                + 0.5 * clamp(frac_of_observable, 0, 1)

        score   = W_RECON * recon + W_F1 * f1 + W_FCAST * fcast + W_UTIL * util

    with ``(W_RECON, W_F1, W_FCAST, W_UTIL) = (1.0, 1.0, 1.0, 0.5)`` and
    ``MARGIN_SCALE = 0.1``: a per-position mean log-likelihood margin of 0.1 is a
    strong probe win and earns the full forecast point. Revisit the weights if one
    term starts to dominate selection.

    Parameters
    ----------
    gate_dict : dict
        As produced by :func:`compute_gate` (keys ``forecastability``, ``decode``,
        ``utilization``).
    recon_floor : float
        ``envelope_corr`` disqualification floor. Callers with a cfg pass
        ``cfg.gate_recon_floor``.
    hard_min_codes : int
        Minimum number of distinct codes; fewer disqualifies the evaluation.
    """
    fc = gate_dict["forecastability"]  # type: ignore[index]
    dec = gate_dict["decode"]  # type: ignore[index]
    ut = gate_dict["utilization"]  # type: ignore[index]

    # Hard disqualification: reconstruction failed
    env_corr = float(dec["envelope_corr"])  # type: ignore[index]
    if math.isnan(env_corr) or env_corr < recon_floor:
        return float("-inf")

    # Hard disqualification: terminal code collapse. A collapsing codec can keep
    # reconstructing through the decoder's position embeddings alone (co2 at about 25
    # codes still decoded at corr 0.78), so the recon floor never fires. The floor is
    # on the absolute distinct-code count, because the 0.02 fraction flag is wrong for
    # 64k codebooks (2% = 1280 codes would reject every healthy fsq6 checkpoint). The
    # default 8 is below every healthy codec seen (the smallest used 10) and above the
    # terminal one-code state. It applies only when the utilization dict carries
    # ``n_distinct_codes``.
    n_codes = ut.get("n_distinct_codes") if isinstance(ut, dict) else None  # type: ignore[union-attr]
    if n_codes is not None and int(n_codes) < int(hard_min_codes):
        return float("-inf")

    # Soft sub-metrics, clamped to [0, 1]
    def _clamp01(v: float) -> float:
        if math.isnan(v):
            return 0.0
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    MARGIN_SCALE = 0.1  # margin_transition ~0.1 => a full point of forecast reward
    W_RECON, W_F1, W_FCAST, W_UTIL = 1.0, 1.0, 1.0, 0.5

    recon = _clamp01(env_corr)
    f1 = _clamp01(float(dec["peak_f1"]))  # type: ignore[index]
    fcast = _clamp01(float(fc["margin_transition"]) / MARGIN_SCALE)  # type: ignore[index]
    # Utilization is a soft reward: mean of the per-dim entropy and the observable
    # fraction. Missing keys count as 0.
    min_dim_entropy = _clamp01(float(ut.get("min_dim_entropy", 0.0)))  # type: ignore[union-attr]
    frac_obs = _clamp01(float(ut.get("frac_of_observable", 0.0)))  # type: ignore[union-attr]
    util = 0.5 * min_dim_entropy + 0.5 * frac_obs

    return W_RECON * recon + W_F1 * f1 + W_FCAST * fcast + W_UTIL * util


class _EMA:
    """Exponential moving average of a module's state_dict.

    Float tensors are averaged; non-float buffers (e.g. ``num_batches_tracked``) are
    copied. ``state_dict`` returns a copy of the shadow, usable directly with
    :meth:`SpectroCodec.load_state_dict`.
    """

    def __init__(self, module: torch.nn.Module, decay: float) -> None:
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"_EMA: decay must be in [0, 1); got {decay}")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, p in module.state_dict().items():
            self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        d = self.decay
        for name, p in module.state_dict().items():
            s = self.shadow.get(name)
            if s is None:
                self.shadow[name] = p.detach().clone()
                continue
            if p.is_floating_point():
                # shadow = d * shadow + (1 - d) * live
                s.mul_(d).add_(p.detach(), alpha=1.0 - d)
            else:
                # int buffers (e.g. num_batches_tracked): track the live value directly.
                s.copy_(p.detach())

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: t.detach().clone() for name, t in self.shadow.items()}


# Gate evaluation
@torch.no_grad()
def compute_gate(
    codec: SpectroCodec,
    eval_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    frame_seq: torch.Tensor,
    cfg: SpectroCodecConfig,
) -> Dict[str, object]:
    """Compute the §4.4 gate quantities on held-out data.

    Parameters
    ----------
    codec : SpectroCodec
        The codec being trained.
    eval_pairs : list of (spec_a, spec_b) or (spec_a, spec_b, mask)
        Held-out δ-shift pairs, used for stability (codes of a against b) and decode
        fidelity (recon of a against a). The optional third element is the (B, C, T)
        per-(channel, STFT frame) validity mask of ``spec_a``
        (``data.spectro_frame_mask``, when ``cfg.mask_missing`` is set). It is passed
        to ``gate.decode_fidelity``, so the reconstruction statistics and baselines
        use real data only.
    frame_seq : tensor of shape (B, n_frames, C, F, T)
        Consecutive frames, used for persistence (frame 0 against 1) and
        forecastability (the whole code sequence).
    cfg : SpectroCodecConfig

    Returns
    -------
    dict
        ``stability`` and ``persistence`` (floats), ``forecastability``, ``decode``
        and ``utilization`` (dicts), and the pass flags ``pass_stability``,
        ``pass_persistence`` and ``pass_utilization``.
    """
    was_training = codec.training
    codec.eval()

    # Stability and decode fidelity on the held-out δ-pairs
    stab_vals: List[float] = []
    dec_corr: List[float] = []
    dec_f1: List[float] = []
    dec_sharp: List[float] = []
    # Patch-lattice keys, for the recon and the ground-truth control (``target_*``).
    # gate_score does not read them.
    _lat_keys = (
        "patch_lattice_ratio", "lattice_energy_frac", "seam_ratio_freq", "seam_ratio_time",
    )
    # Full-spectrogram keys (gate.full_spectro_metrics and its trivial baselines), the
    # only decode keys that compare the whole arrays. gate_score does not read them
    # either, so checkpoint selection does not depend on them.
    _full_keys = ("spec_nrmse", "spec_corr2d", "spec_valid_frac",
                  "spec_nrmse_band", "spec_corr2d_band")
    _base_keys = tuple(
        f"base_{b}_{m}"
        for b in ("self", "tmean", "cfmean", "wcmean")
        for m in ("spec_nrmse", "spec_corr2d", "spec_nrmse_band", "spec_corr2d_band")
    )
    dec_lat: Dict[str, List[float]] = {}
    for _pair in eval_pairs:
        spec_a, spec_b = _pair[0], _pair[1]
        pair_mask = _pair[2] if len(_pair) > 2 else None
        out_a = codec.forward(spec_a)
        _, codes_b = codec.quantize(codec.encode(spec_b))
        stab_vals.append(gate.stability(out_a["codes"], codes_b))
        dm = gate.decode_fidelity(
            out_a["recon"], spec_a, patch_f=cfg.patch_f, patch_t=cfg.patch_t,
            full_spec=True, mask=pair_mask,
        )
        dec_corr.append(dm["envelope_corr"])
        dec_f1.append(dm["peak_f1"])
        dec_sharp.append(dm["sharpness"])
        for k in _lat_keys:
            for kk in (k, f"target_{k}"):
                if kk in dm:
                    dec_lat.setdefault(kk, []).append(float(dm[kk]))
        for kk in _full_keys + _base_keys:
            if kk in dm:
                dec_lat.setdefault(kk, []).append(float(dm[kk]))

    stability_val = float(sum(stab_vals) / len(stab_vals))
    decode = {
        "envelope_corr": float(sum(dec_corr) / len(dec_corr)),
        "peak_f1": float(sum(dec_f1) / len(dec_f1)),
        "sharpness": float(sum(dec_sharp) / len(dec_sharp)),
    }
    for k, v in dec_lat.items():
        decode[k] = float(sum(v) / len(v))

    # Persistence and forecastability on the consecutive-frame sequence
    B, n_frames = frame_seq.shape[0], frame_seq.shape[1]
    # encode every frame -> (B, n_frames, n_tok, fsq_dim) codes
    flat = frame_seq.reshape(B * n_frames, *frame_seq.shape[2:])  # (B*Fr, C, F, T)
    _, codes_flat = codec.quantize(codec.encode(flat))
    codes_seq = codes_flat.reshape(B, n_frames, cfg.n_tok, cfg.fsq_dim)

    # persistence over the first consecutive step (frame 0 -> frame 1)
    persistence_val = gate.persistence(codes_seq[:, 0], codes_seq[:, 1])

    # Forecastability over the full sequence, with the default code-derived split
    forecast = gate.forecastability(codes_seq, transition_mask=None)

    # codebook utilization / collapse detection over the full eval code set (all frames).
    util = gate.utilization(codes_seq, cfg=cfg)

    if was_training:
        codec.train()

    return {
        "stability": stability_val,
        "persistence": float(persistence_val),
        "forecastability": forecast,
        "decode": decode,
        "utilization": util,
        "pass_stability": bool(stability_val >= cfg.gate_stability),
        "pass_persistence": bool(persistence_val >= cfg.gate_persistence),
        "pass_utilization": bool(not util["collapsed"]),
    }


def _fmt_gate(step: int, g: Dict[str, object]) -> str:
    fc = g["forecastability"]  # type: ignore[index]
    dec = g["decode"]  # type: ignore[index]
    ut = g["utilization"]  # type: ignore[index]
    return (
        f"[spike step {step:4d}] "
        f"stab={g['stability']:.3f}({'ok' if g['pass_stability'] else 'x'}) "
        f"persist={g['persistence']:.3f}({'ok' if g['pass_persistence'] else 'x'}) "
        f"util[frac={ut['frac_codes_used']:.4f} "  # type: ignore[index]
        # .get: synthetic or older gate dicts may lack the count and entropy fields.
        f"ncode={ut.get('n_distinct_codes', '?')}/{ut.get('codebook_size', '?')} "  # type: ignore[union-attr]
        f"eff={ut.get('effective_codes', float('nan')):.1f} "  # type: ignore[union-attr]
        f"minH={ut['min_dim_entropy']:.3f}]"  # type: ignore[index]
        f"({'ok' if g['pass_utilization'] else 'COLLAPSED'}) "
        f"forecast_margin_tr={fc['margin_transition']:+.4f}"  # type: ignore[index]
        f"(beats={fc['beats_persistence']}) "  # type: ignore[index]
        f"decode[corr={dec['envelope_corr']:.3f} "  # type: ignore[index]
        f"f1={dec['peak_f1']:.3f} "  # type: ignore[index]
        f"sharp={dec['sharpness']:.3f}"  # type: ignore[index]
        # Patch-lattice ratio, recon / ground-truth control. Synthetic or older gate
        # dicts carry no patch size, hence .get.
        + (f" lattice={dec.get('patch_lattice_ratio', float('nan')):.2f}"  # type: ignore[union-attr]
           f"/gt{dec.get('target_patch_lattice_ratio', float('nan')):.2f}"  # type: ignore[union-attr]
           if "patch_lattice_ratio" in dec else "")  # type: ignore[operator]
        # Full-spectrogram recon beside the time-averaged env_corr; /tm is the
        # time-mean-envelope baseline in the same units.
        + (f" nrmse={dec.get('spec_nrmse', float('nan')):.3f}"  # type: ignore[union-attr]
           f"/tm{dec.get('base_tmean_spec_nrmse', float('nan')):.3f}"  # type: ignore[union-attr]
           f" corr2d={dec.get('spec_corr2d', float('nan')):.3f}"  # type: ignore[union-attr]
           f"/tm{dec.get('base_tmean_spec_corr2d', float('nan')):.3f}"  # type: ignore[union-attr]
           if "spec_corr2d" in dec else "")  # type: ignore[operator]
        + "]"
    )


# Divergence guard shared by every adversarial codec train step, safe under DDP

# Skipped optimizer steps in this process, across all codecs. Rank 0 logs it so a
# firing guard shows in the .err log. Reset with :func:`reset_skipped_steps`.
_SKIPPED_STEPS: int = 0


def reset_skipped_steps() -> None:
    """Zero the process-wide skipped-optimizer-step counter (call once at trainer start)."""
    global _SKIPPED_STEPS
    _SKIPPED_STEPS = 0


def skipped_steps() -> int:
    """Return the process-wide count of optimizer steps skipped by the divergence guard."""
    return _SKIPPED_STEPS


def is_step_diverged(*losses: torch.Tensor) -> bool:
    """Whether every rank should skip this optimizer step.

    A collapsing codec can drive the adaptive adversarial weight and the generator
    loss to Inf/NaN on one rank. If that rank corrupts its parameters while the
    others step, the ranks desync and the next collective hits the NCCL watchdog
    (SIGTERM, exit 143). The decision is therefore made identical on all ranks: the
    local flag is 1 if any of ``losses`` is non-finite, and under DDP it is
    all-reduced with MAX, so if any rank is bad all ranks return True. Without DDP
    the local flag is returned.

    This gates only ``optimizer.step()``. The caller must still run ``backward()``
    (and so the gradient all-reduce) on every rank before calling it, so the
    collectives stay symmetric. On True the caller also calls
    :func:`note_skipped_step`.
    """
    local_bad = 0.0
    for loss in losses:
        if loss is None:
            continue
        if not torch.isfinite(loss).all():
            local_bad = 1.0
            break

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        flag = torch.tensor(
            [local_bad],
            device=losses[0].device if losses and losses[0] is not None else "cpu",
            dtype=torch.float32,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item() > 0.0)

    return local_bad > 0.0


def note_skipped_step() -> None:
    """Increment the process-wide skipped-optimizer-step counter (called on a skip)."""
    global _SKIPPED_STEPS
    _SKIPPED_STEPS += 1


# One generator/discriminator step, shared by run_spike and the streaming DDP
# trainer in train_codec.py
def codec_train_step(
    codec: "SpectroCodec",
    disc: "FreqAwarePatchGAN",
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    spec_a: torch.Tensor,
    spec_b: torch.Tensor,
    cfg: SpectroCodecConfig,
    step: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """One generator plus discriminator step on a δ-shift pair.

    Shared by :func:`run_spike` (fixed pre-built pool, one GPU) and the streaming DDP
    trainer (:mod:`train_codec`), so both use the same per-step math:

    1. Generator (codec) step: ``codec.generator_losses(spec_a, spec_b, disc, cfg,
       step)``, backward on ``total``, ``opt_g.step()``.
    2. Discriminator step: a fresh, detached recon of ``spec_a`` scored against the
       real ``spec_a`` by :func:`losses.discriminator_loss`, ``opt_d.step()``.

    ``codec`` and ``disc`` may be raw modules or DDP-wrapped. ``generator_losses`` is
    a ``SpectroCodec`` method; under DDP the trainer calls ``codec(spec_a)`` through
    the wrapper inside ``generator_losses`` so DDP sees the forward (see
    :mod:`train_codec`). On one GPU both are raw modules.

    Parameters
    ----------
    codec, disc : nn.Module
        The codec (generator) and the external discriminator, raw or DDP-wrapped.
    opt_g, opt_d : torch.optim.Optimizer
        Their Adam optimizers.
    spec_a, spec_b : tensor of shape (B, C, F, T)
        δ-shift pair, already on the target device.
    cfg : SpectroCodecConfig
    step : int
        Global step (drives ``cfg.adv_warmup_steps``).

    Returns
    -------
    g_terms : dict
        From :meth:`SpectroCodec.generator_losses` (``total``, ``adversarial``,
        ``pixel``, ``consistency``, ``entropy``, ``adaptive_weight``, ``recon``,
        ``codes``, ...).
    d_loss : tensor
        Scalar discriminator loss. Both outputs are live tensors; detach them for
        logging.
    """
    codec.train()

    # Generator (codec) step
    opt_g.zero_grad(set_to_none=True)
    g_terms = codec.generator_losses(spec_a, spec_b, disc, cfg, step=step)
    g_terms["total"].backward()
    # backward and its gradient all-reduce already ran on every rank; only the step
    # is skipped, on all ranks alike, so a collapsing codec cannot desync DDP.
    if is_step_diverged(g_terms["total"]):
        note_skipped_step()
    else:
        opt_g.step()

    # Discriminator step on a fresh recon, detached from the codec graph
    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        recon = codec.forward(spec_a)["recon"]
    d_loss = discriminator_loss(disc, spec_a, recon, cfg)
    d_loss.backward()
    if is_step_diverged(d_loss):
        note_skipped_step()
    else:
        opt_d.step()

    return g_terms, d_loss


# The spike
def run_spike(
    cfg: SpectroCodecConfig,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    steps: int,
    device: str | torch.device = "cpu",
    eval_every: int = 1,
    *,
    lr: float = 1e-3,
    disc_lr: Optional[float] = None,
    eval_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    frame_seq: Optional[torch.Tensor] = None,
    eval_batch_size: int = 2,
    eval_frames: int = 4,
    log_fn: Optional[Callable[[str], None]] = print,
    seed: int = 0,
    resume_state: Optional[Dict[str, object]] = None,
    on_eval: Optional[Callable[[int, "SpectroCodec", torch.optim.Optimizer,
                                 torch.optim.Optimizer, "FreqAwarePatchGAN",
                                 Dict[str, object]], None]] = None,
    ema: bool = False,
    ema_decay: float = 0.999,
    on_best: Optional[Callable[[int, "SpectroCodec", "FreqAwarePatchGAN",
                                 Dict[str, object]], None]] = None,
    on_ema: Optional[Callable[[int, Dict[str, torch.Tensor],
                               Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    """Train the codec with GAN alternation on ``batches``; gate every ``eval_every``.

    Parameters
    ----------
    cfg : SpectroCodecConfig
    batches : list of (spec_a, spec_b)
        δ-shift training pairs (see :func:`synthetic_batches`), cycled if ``steps``
        exceeds ``len(batches)``. ``spec_a`` is the recon target and the code source;
        ``spec_b`` is its δ-shifted pair for the consistency term.
    steps : int
        Number of generator/discriminator steps.
    device : str or torch.device
    eval_every : int
        Compute and log the gate every this many steps, and after the last step.
    lr, disc_lr : float
        Adam learning rates of the codec and the discriminator (``disc_lr``
        defaults to ``lr``).
    eval_pairs : list, optional
        Held-out δ-pairs for the gate; built synthetically if None.
    frame_seq : tensor, optional
        Held-out consecutive-frame sequence for persistence and forecastability;
        built synthetically if None.
    eval_batch_size, eval_frames : int
        Sizes of the synthetic held-out data.
    log_fn : callable(str) or None
        Per-evaluation logger (default ``print``); None silences it.
    seed : int
    resume_state : dict or None
        Checkpoint written by the CLI ``main`` to warm-start from: ``codec``,
        ``disc``, ``opt_g`` and ``opt_d`` state_dicts and ``step``, the number of
        steps already done. Training then runs ``steps`` more steps, and the global
        step numbers continue from there.
    on_eval : callable(step, codec, opt_g, opt_d, disc, gate_dict) or None
        Called at every gate evaluation so the CLI can write ``gate_<step>.json`` and
        ``codec_last.pt``. ``step`` is the global step; ``gate_dict`` also carries
        ``score`` (:func:`gate_score`) and ``is_best``.
    ema : bool
        Keep an exponential moving average of the codec parameters, updated every
        step, and pass it to ``on_ema`` at each evaluation.
    ema_decay : float
        EMA decay in [0, 1).
    on_best : callable(step, codec, disc, gate_dict) or None
        Called only when :func:`gate_score` strictly improves on the best so far (a
        ``-inf`` score never becomes best), so the CLI can write ``codec_best.pt``.
    on_ema : callable(step, ema_state_dict, gate_dict) or None
        Called at each evaluation when ``ema`` is set, so the CLI can write
        ``codec_ema.pt``.

    Returns
    -------
    dict
        The last gate dict from :func:`compute_gate`, plus ``steps`` (steps run),
        ``global_step`` (``start_step + steps``), and ``best_score`` / ``best_step``
        (``-inf`` / None if every evaluation failed reconstruction).
    """
    if steps < 1:
        raise ValueError("run_spike: steps must be >= 1")
    if not batches:
        raise ValueError("run_spike: need at least one training batch")

    torch.manual_seed(seed)
    device = torch.device(device)

    codec = SpectroCodec(cfg).to(device)
    disc = FreqAwarePatchGAN(cfg).to(device)

    opt_g = torch.optim.Adam(codec.parameters(), lr=lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=disc_lr if disc_lr is not None else lr)

    # Optional warm start
    start_step = 0
    if resume_state is not None:
        codec.load_state_dict(resume_state["codec"])
        if resume_state.get("disc") is not None:
            disc.load_state_dict(resume_state["disc"])
        if resume_state.get("opt_g") is not None:
            opt_g.load_state_dict(resume_state["opt_g"])
        if resume_state.get("opt_d") is not None:
            opt_d.load_state_dict(resume_state["opt_d"])
        start_step = int(resume_state.get("step", 0))

    # EMA shadow of the codec params, built after any warm-start load
    ema_shadow: Optional[_EMA] = _EMA(codec, ema_decay) if ema else None

    # best-checkpoint tracking (composite gate score; higher is better).
    best_score = float("-inf")
    best_step: Optional[int] = None

    # Held-out eval data (never trained on), moved to the device
    if eval_pairs is None:
        eval_pairs = synthetic_batches(
            cfg, n_batches=1, batch_size=eval_batch_size, device=device, seed=seed + 777
        )
    else:
        eval_pairs = [(a.to(device), b.to(device)) for a, b in eval_pairs]
    if frame_seq is None:
        frame_seq = _consecutive_frame_sequence(
            cfg, batch_size=eval_batch_size, n_frames=eval_frames, device=device,
            seed=seed + 999,
        )
    else:
        frame_seq = frame_seq.to(device)

    final_gate: Dict[str, object] = {}
    for local_step in range(steps):
        step = start_step + local_step
        spec_a, spec_b = batches[local_step % len(batches)]
        spec_a = spec_a.to(device)
        spec_b = spec_b.to(device)

        # One generator + discriminator step
        g_terms, d_loss = codec_train_step(
            codec, disc, opt_g, opt_d, spec_a, spec_b, cfg, step=step
        )

        # EMA update, every step
        if ema_shadow is not None:
            ema_shadow.update(codec)

        is_last = local_step == steps - 1
        if (step % eval_every == 0) or is_last:
            g = compute_gate(codec, eval_pairs, frame_seq, cfg)
            g["g_total"] = float(g_terms["total"].detach())
            g["d_loss"] = float(d_loss.detach())
            g["adaptive_weight"] = float(g_terms["adaptive_weight"])
            g["step"] = step

            # Composite selection score; -inf only if reconstruction fails (see
            # gate_score).
            score = gate_score(g, recon_floor=cfg.gate_recon_floor, hard_min_codes=getattr(cfg, "gate_hard_min_codes", 8))
            g["score"] = score
            is_best = score > best_score
            g["is_best"] = bool(is_best)
            if is_best:
                best_score = score
                best_step = step

            if log_fn is not None:
                log_fn(
                    _fmt_gate(step, g)
                    + f" g_total={g['g_total']:+.4f} d_loss={g['d_loss']:.4f}"
                    + f" adv_lam={g['adaptive_weight']:.4g}"
                    + f" score={score:+.4f}"
                )
            if on_eval is not None:
                on_eval(step, codec, opt_g, opt_d, disc, g)
            if is_best and on_best is not None:
                on_best(step, codec, disc, g)
                if log_fn is not None:
                    log_fn(f"[spike] new BEST score={score:+.4f} @ step {step}")
            if ema_shadow is not None and on_ema is not None:
                on_ema(step, ema_shadow.state_dict(), g)
            final_gate = g

    final_gate["steps"] = steps
    final_gate["global_step"] = start_step + steps
    final_gate["best_score"] = best_score
    final_gate["best_step"] = best_step
    return final_gate


# CLI entry point
def _jsonable(obj: object) -> object:
    """Recursively coerce a gate dict into plain JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    return str(obj)


def _write_gate_json(out_dir: Path, step: int, gate_dict: Dict[str, object]) -> Path:
    path = out_dir / f"gate_{step}.json"
    with open(path, "w") as fh:
        json.dump(_jsonable(gate_dict), fh, indent=2, sort_keys=True)
    return path


def _save_checkpoint(
    out_dir: Path,
    step: int,
    codec: SpectroCodec,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    disc: FreqAwarePatchGAN,
    cfg: SpectroCodecConfig,
) -> Path:
    """Write ``codec_last.pt`` (codec, disc, both optimizers, step) for ``--resume``.

    ``step`` is the 0-indexed current step. The checkpoint stores the number of
    completed steps (``step + 1``), which is what :func:`run_spike` expects in
    ``resume_state["step"]``, so resumed step numbers continue without a gap.
    """
    path = out_dir / "codec_last.pt"
    tmp = out_dir / "codec_last.pt.tmp"
    torch.save(
        {
            "codec": codec.state_dict(),
            "disc": disc.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "step": int(step) + 1,
            "cfg": cfg,
        },
        tmp,
    )
    tmp.replace(path)  # atomic-ish: avoid torn reads on resume
    return path


def resume_best_score(out_path) -> float:
    """Initial ``best_score`` for a trainer loop.

    The score stored in ``out_path/codec_best.pt``, else ``-inf``. Starting each
    resumed leg at ``-inf`` would let it overwrite ``codec_best.pt`` with a leg-local
    best that is worse than the stored one. Reading the score back from the file
    needs no change to the checkpoint format.
    """
    if out_path is None:
        return float("-inf")
    p = Path(out_path) / "codec_best.pt"
    if not p.exists():
        return float("-inf")
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        return float(ck.get("score", float("-inf")))
    except Exception:
        return float("-inf")


def _save_best_checkpoint(
    out_dir: Path,
    step: int,
    score: float,
    codec: SpectroCodec,
    disc: FreqAwarePatchGAN,
    cfg: SpectroCodecConfig,
    gate_dict: Dict[str, object],
) -> Path:
    """Write ``codec_best.pt``, the codec at the best composite-gate evaluation.

    Stores the codec and disc state, ``step`` (number of completed steps, as in
    :func:`_save_checkpoint`), ``score`` and the full ``gate`` dict, so the file
    describes itself. Written via a temporary file and replace.
    """
    path = out_dir / "codec_best.pt"
    tmp = out_dir / "codec_best.pt.tmp"
    torch.save(
        {
            "codec": codec.state_dict(),
            "disc": disc.state_dict(),
            "step": int(step) + 1,
            "score": float(score),
            "gate": _jsonable(gate_dict),
            "cfg": cfg,
        },
        tmp,
    )
    tmp.replace(path)
    return path


def _save_ema_checkpoint(
    out_dir: Path,
    step: int,
    ema_state: Dict[str, torch.Tensor],
    cfg: SpectroCodecConfig,
) -> Path:
    """Write ``codec_ema.pt`` — the EMA-averaged codec weights (drop-in state_dict)."""
    path = out_dir / "codec_ema.pt"
    tmp = out_dir / "codec_ema.pt.tmp"
    torch.save(
        {"codec": ema_state, "step": int(step) + 1, "cfg": cfg},
        tmp,
    )
    tmp.replace(path)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tokamak_foundation_model.ignite.spike",
        description="IGNITE Phase-A gate spike: train a spectrogram codec and gate it.",
    )
    p.add_argument("--n_shots", type=int, default=40,
                   help="Number of shots to auto-discover for the data pool.")
    p.add_argument("--shots", type=str, default=None,
                   help="Optional explicit comma-separated shot list (overrides --n_shots).")
    p.add_argument("--n_batches", type=int, default=400,
                   help="Pool size: number of (spec_a, spec_b) batches to pre-build.")
    p.add_argument("--batch_size", type=int, default=8, help="Pairs per batch.")
    p.add_argument("--steps", type=int, default=20000, help="Training/gate steps.")
    p.add_argument("--eval_every", type=int, default=1000,
                   help="Compute + persist the gate every this many steps.")
    p.add_argument("--lr", type=float, default=1e-3, help="Codec (generator) Adam LR.")
    p.add_argument("--disc_lr", type=float, default=None,
                   help="Discriminator Adam LR (defaults to --lr).")
    p.add_argument("--n_frames", type=int, default=6,
                   help="Consecutive frames per persistence/forecastability sequence.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory for gate_<step>.json / codec_last.pt / summary.json.")
    p.add_argument("--device", type=str, default="cuda",
                   help="torch device (default cuda; the smoke test uses cpu).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None,
                   help="Optional codec_last.pt to warm-start codec/opt/step from.")
    p.add_argument("--eval_n_batches", type=int, default=8,
                   help="Held-out eval-pair batches sliced from the tail of the pool.")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                   help="Directory of {shot}_processed.h5 files.")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic sum-of-sinusoid data instead of real HDF5 "
                        "(smoke tests and dry runs).")
    p.add_argument("--entropy_weight", type=float, default=None,
                   help="Override cfg.entropy_weight (anti-collapse regularizer strength).")
    p.add_argument("--consistency_weight", type=float, default=None,
                   help="Override cfg.consistency_weight (shift-invariance strength).")
    p.add_argument("--ema", action="store_true",
                   help="Maintain an EMA shadow of the codec params; write codec_ema.pt "
                        "at every eval. Default off.")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay in [0, 1) when --ema is set (default 0.999).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Build the pool once, run the spike, and persist gate/ckpt/summary to ``--out_dir``.

    Returns the final gate dict (also written to ``summary.json``).
    """
    args = build_arg_parser().parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ece has 40 channels. real_ece_batches sets cfg.channels from the data anyway;
    # setting it here makes the synthetic path match.
    cfg = SpectroCodecConfig(channels=40)
    if args.entropy_weight is not None:
        cfg.entropy_weight = float(args.entropy_weight)
    if args.consistency_weight is not None:
        cfg.consistency_weight = float(args.consistency_weight)

    shots: Optional[List[str]] = None
    if args.shots is not None:
        shots = [s.strip() for s in args.shots.split(",") if s.strip()]

    # Build the pool once: train batches, held-out eval slice and frame sequence
    print(f"[spike-cli] building pool: synthetic={args.synthetic} "
          f"n_batches={args.n_batches} batch_size={args.batch_size} "
          f"n_frames={args.n_frames} device={device}", flush=True)
    t_pool = time.time()

    total_batches = args.n_batches + args.eval_n_batches
    if args.synthetic:
        pool = synthetic_batches(
            cfg, n_batches=total_batches, batch_size=args.batch_size,
            device="cpu", seed=args.seed,
        )
        frame_seq = _consecutive_frame_sequence(
            cfg, batch_size=max(2, args.batch_size // 2), n_frames=args.n_frames,
            device="cpu", seed=args.seed + 999,
        )
    else:
        n_shots = args.n_shots if shots is None else len(shots)
        pool_shots = shots if shots is not None else discover_shots(
            args.data_dir, limit=n_shots
        )
        pool = real_ece_batches(
            cfg, n_batches=total_batches, batch_size=args.batch_size,
            shots=pool_shots, data_dir=args.data_dir, device="cpu", seed=args.seed,
        )
        frame_seq = real_frame_sequence(
            cfg, batch_size=max(2, args.batch_size // 2), n_frames=args.n_frames,
            shots=pool_shots, data_dir=args.data_dir, device="cpu",
        )

    # held-out eval pairs = tail slice of the pool (never trained on)
    train_batches = pool[: args.n_batches]
    eval_pairs = pool[args.n_batches :]
    print(f"[spike-cli] pool built in {time.time() - t_pool:.1f}s "
          f"(train={len(train_batches)} eval={len(eval_pairs)} "
          f"frame_seq={tuple(frame_seq.shape)} channels={cfg.channels})", flush=True)

    resume_state = None
    if args.resume is not None:
        print(f"[spike-cli] resuming from {args.resume}", flush=True)
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)

    # Write the gate JSON and the rolling checkpoint at every evaluation
    def on_eval(step, codec, opt_g, opt_d, disc, gate_dict):
        gp = _write_gate_json(out_dir, step, gate_dict)
        cp = _save_checkpoint(out_dir, step, codec, opt_g, opt_d, disc, cfg)
        print(f"[spike-cli] step {step}: wrote {gp.name} + {cp.name}", flush=True)

    # Write codec_best.pt when the composite score improves
    def on_best(step, codec, disc, gate_dict):
        bp = _save_best_checkpoint(
            out_dir, step, float(gate_dict["score"]), codec, disc, cfg, gate_dict
        )
        print(f"[spike-cli] step {step}: NEW BEST score={gate_dict['score']:+.4f} "
              f"-> wrote {bp.name}", flush=True)

    # Optional: write the EMA weights to codec_ema.pt at every evaluation
    def on_ema(step, ema_state, gate_dict):
        ep = _save_ema_checkpoint(out_dir, step, ema_state, cfg)
        print(f"[spike-cli] step {step}: wrote {ep.name} (EMA)", flush=True)

    disc_lr = args.disc_lr
    t_run = time.time()
    final_gate = run_spike(
        cfg,
        train_batches,
        steps=args.steps,
        device=device,
        eval_every=args.eval_every,
        lr=args.lr,
        disc_lr=disc_lr,
        eval_pairs=eval_pairs,
        frame_seq=frame_seq,
        seed=args.seed,
        resume_state=resume_state,
        on_eval=on_eval,
        ema=args.ema,
        ema_decay=args.ema_decay,
        on_best=on_best,
        on_ema=on_ema if args.ema else None,
    )
    final_gate["wall_s"] = time.time() - t_run

    # Summary
    summary = dict(final_gate)
    summary["config"] = {
        "n_shots": args.n_shots,
        "shots": shots,
        "n_batches": args.n_batches,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "lr": args.lr,
        "disc_lr": args.disc_lr,
        "n_frames": args.n_frames,
        "device": str(device),
        "seed": args.seed,
        "synthetic": args.synthetic,
        "resumed_from": args.resume,
        "channels": cfg.channels,
        "ema": args.ema,
        "ema_decay": args.ema_decay,
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(_jsonable(summary), fh, indent=2, sort_keys=True)
    print(f"[spike-cli] DONE steps={final_gate.get('steps')} "
          f"global_step={final_gate.get('global_step')} "
          f"stability={final_gate.get('stability'):.3f} "
          f"persistence={final_gate.get('persistence'):.3f} "
          f"best_score={final_gate.get('best_score')} "
          f"best_step={final_gate.get('best_step')} "
          f"-> {out_dir}/summary.json", flush=True)
    return final_gate


if __name__ == "__main__":
    main()
