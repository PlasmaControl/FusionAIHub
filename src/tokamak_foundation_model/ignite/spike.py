"""IGNITE Phase-A **gate spike** — a small, parametrizable training + gate probe.

This is the minimal end-to-end exercise of the Phase-A codec against the oracle gate
(``docs/IGNITE_DESIGN.md`` §4.4). It:

  1. trains a :class:`~ignite.codec.SpectroCodec` with a standard GAN alternation
     (generator step on the codec, discriminator step on an external
     :class:`~ignite.discriminator.FreqAwarePatchGAN`, both Adam);
  2. every ``eval_every`` steps computes the gate on held-out batches — **stability**
     (codes of ``x`` vs its δ-shifted pair), **persistence** + **forecastability** (codes
     over consecutive world-model frames), and **decode_fidelity** (recon vs target);
  3. logs a compact per-eval line and returns the final gate dict.

Everything runs on CPU with **synthetic** data (sum-of-sinusoid raw signals fed through
``data.py``) so the smoke test needs no real HDF5. The real-data entry point
(:func:`real_ece_batches`) is a clearly-marked TODO stub for the parent to wire.

Reuse boundary (§7): only ``torch`` + sibling ``ignite`` modules
(``codec``, ``discriminator``, ``losses``, ``gate``, ``data``, ``config``). No FAITH
model code.

Gate stratification note (§4.4): forecastability uses the gate's DEFAULT code-derived
transition split (``transition_mask=None``) — we do **not** build a physics detector here.
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


# ------------------------------------------------------------------------------------- #
# synthetic data
# ------------------------------------------------------------------------------------- #
def _sum_of_sinusoids(
    n_samples: int,
    freqs_hz: Iterable[float],
    fs: float,
    noise_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """A single-channel (n_samples,) raw signal = sum of sinusoids (+ optional noise).

    Phases are random (per call) so that a δ-shift genuinely changes the sub-window
    realization while the *frequencies* (the statistic) are unchanged.
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
    """A raw (C, W_full) shot long enough to slice δ-shift pairs from.

    Each channel is an independent sum-of-sinusoid realization of the SAME mode set
    (so the freq statistic is shared across channels, only the phase realization differs).
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
    """Produce ``n_batches`` of δ-shift consistency pairs ``(spec_a, spec_b)``.

    Each element is a pair of log-power spectrogram batches of shape
    ``(batch_size, C, freq_bins, time_frames)`` built entirely from synthetic
    sum-of-sinusoid raw signals via :func:`data.shift_pair_windows` — NO real data.

    ``spec_a`` is the window ``[t0, t0+CHUNK_S]`` and ``spec_b`` the δ-shifted window
    ``[t0+δ, t0+CHUNK_S+δ]`` of the same synthetic shot: same modes (statistic), different
    sub-window realization. ``δ`` defaults to the midpoint of ``cfg.consistency_delta_ms``
    (deterministic for the smoke test) unless ``delta_ms`` is given or ``None`` forces a
    per-pair draw from the config range.

    Notes
    -----
    * Each sample within a batch uses a slightly different mode set / start time so the
      batch is not degenerate.
    * The returned list is materialized (small, CPU-friendly) so callers can re-iterate it
      as held-out eval data.
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
    """A batch of consecutive world-model frames for persistence / forecastability.

    Returns ``(batch_size, n_frames, C, freq_bins, time_frames)``. Consecutive frames come
    from consecutive 50 ms windows of a synthetic shot whose mode frequencies **drift
    slowly** across frames — a predictable transition the Markov probe should beat
    persistence on (§4.4), while the shared structure keeps persistence meaningfully high.
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


# ------------------------------------------------------------------------------------- #
# real-data path (wires the existing HDF5 loader, read-only)
# ------------------------------------------------------------------------------------- #
#
# Integration facts (verified against real 200729 ece; see data.py integration note):
#   * The loader class is ``TokamakH5Dataset`` (data_loader.py). One instance == one shot.
#   * ``ds._load_signal_raw(ds.h5_file, cfg_sig, t_start, t_end) -> (raw (C,T), valid_len,
#     nan_mask (C,T))`` returns raw samples ALREADY resampled to ``cfg_sig.target_fs``
#     (500 kHz for ece == STFT_FS) and NaN/zero-filled. It's an internal (underscore) API
#     that needs an OPEN h5py handle — we call ``ds._open_hdf5()`` first and pin the call.
#   * For ece, ``channels_to_use=slice(0,40)`` so the raw output has C == 40 channels.
#   * The loader zero-PADS windows past the real data end and reports ``valid_len`` = the
#     number of real (non-padded) target-rate samples in the window. We guard on that.
#
# We do NOT modify data_loader.py. We only read ``_load_signal_raw`` + the ece SignalConfig.

# Default location of the preprocessed shot HDF5 files (see MEMORY / project notes).
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
    """Load a raw ece window ``[t0, t0 + span_s]`` at STFT_FS; None if it fails the guard.

    Guards (skip → None):
      * the window overruns the real (non-padded) data: ``valid_len`` must cover the whole
        span in target-rate samples;
      * the window is degenerate (all-NaN / all-zero / near-flat): ``std < min_std``.
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
    """Real ECE δ-shift consistency pairs from the existing HDF5 loader.

    Same structure as :func:`synthetic_batches`: a list of ``(spec_a, spec_b)`` log-power
    spectrogram batches, each ``(batch_size, C, cfg.freq_bins, cfg.time_frames)``, where
    ``spec_b`` is the δ-shifted nuisance pair of ``spec_a`` (via
    :func:`data.shift_pair_windows`). ``C`` is the real ece channel count (40).

    **Side effect (intended):** ``cfg.channels`` is set to the ece channel count actually
    loaded (40), so the codec / discriminator built from ``cfg`` match the data.

    Parameters
    ----------
    cfg : SpectroCodecConfig
        ``freq_bins`` / ``time_frames`` for the crop-pad; ``consistency_delta_ms`` for the
        δ draw when ``delta_ms is None``. ``cfg.channels`` is OVERWRITTEN to 40.
    n_batches, batch_size : int
        Number of held-out/train batches and pairs per batch.
    shots : sequence of shot ids, optional
        Shots to draw windows from. Defaults to the first 8 auto-discovered shots in
        ``data_dir`` (200729 is included explicitly first if present).
    data_dir : path
        Directory of ``{shot}_processed.h5`` files. Default = the training data dir.
    device : str | torch.device
    delta_ms : float or None
        Sub-window shift (ms). ``None`` → per-pair draw from ``cfg.consistency_delta_ms``.
    t0_start : float
        First window start time (s). Default 1.0 s (skips ramp-up per the data-windowing
        convention).
    t0_step : float
        Advance between successive windows within a shot (s). Default = ``CHUNK_S`` (50 ms)
        so windows are non-overlapping.
    min_std : float
        Minimum raw std for a window to count as non-degenerate.
    seed : int
        Seeds the per-pair δ draw for reproducibility.

    Returns
    -------
    list of (spec_a, spec_b), each ``(batch_size, C=40, cfg.freq_bins, cfg.time_frames)``
    float32 on ``device``.

    Raises
    ------
    RuntimeError
        If not enough valid windows can be found to fill ``n_batches * batch_size`` pairs.
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
    """Consecutive-frame ece sequence for persistence / forecastability.

    Returns ``(batch_size, n_frames, C=40, cfg.freq_bins, cfg.time_frames)``: for each
    sample, ``n_frames`` CONSECUTIVE 50 ms (== ``CHUNK_S``) windows of a real shot, each
    STFT'd to log-power via :func:`data.log_power_stft`. This is the material the trainer
    encodes into the ``(B, n_frames, n_tok, fsq_dim)`` code sequence that
    ``gate.forecastability`` / ``gate.persistence`` consume.

    A sequence sample is accepted only if ALL ``n_frames`` consecutive windows are valid
    (real, non-degenerate). Sequences are gathered by scanning shots and stepping the start
    time by ``n_frames * CHUNK_S`` (non-overlapping sequence blocks).

    **Side effect (intended):** sets ``cfg.channels`` to 40 (as :func:`real_ece_batches`).
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


# --- small helpers shared by the two real-data builders ------------------------------- #
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


# ------------------------------------------------------------------------------------- #
# best-checkpoint selection (composite gate score) + optional EMA of codec params
# ------------------------------------------------------------------------------------- #
# Default reconstruction floor for gate-score disqualification. Mirrors
# ``SpectroCodecConfig.gate_recon_floor``; used when ``gate_score`` is called without an
# explicit floor (e.g. minimal unit-test gate dicts). Callers with a cfg in scope pass
# ``cfg.gate_recon_floor`` so the config remains the single source of truth.
DEFAULT_RECON_FLOOR: float = 0.2


def gate_score(gate_dict: Dict[str, object], recon_floor: float = DEFAULT_RECON_FLOOR) -> float:
    """Composite selection score for a gate eval; higher is better.

    The score judges **reconstruction + forecastability + utilization** — NOT raw per-dim
    codebook entropy. A well-reconstructing codec that happens to have one dead FSQ dim (the
    classic large-codebook artifact: ``min_dim_entropy = 0``) is a GOOD codec and must be
    selectable; only a codec that genuinely fails to reconstruct is disqualified.

    HARD DISQUALIFICATION (``score = -inf``) applies ONLY when reconstruction genuinely
    fails — ``decode["envelope_corr"]`` is NaN (e.g. co2's dead 1-code codec, whose flat
    envelope makes the correlation undefined) or below ``recon_floor``. The dim-entropy /
    ``collapsed`` flag is NO LONGER a hard gate (that wrongly rejected ece/bes at
    env_corr≈0.84); it survives only as a soft utilization reward below.

    For codecs above the recon floor the finite score is a documented weighted sum of
    clamped, [0, 1]-normalized sub-metrics::

        recon   = clamp(envelope_corr, 0, 1)                     # reconstruction quality
        f1      = clamp(peak_f1, 0, 1)                           # mode-overlap quality
        fcast   = clamp(margin_transition / MARGIN_SCALE, 0, 1)  # forecastability (skill)
        util    = 0.5 * clamp(min_dim_entropy, 0, 1)
                + 0.5 * clamp(frac_of_observable, 0, 1)          # SOFT utilization reward

        score   = W_RECON * recon + W_F1 * f1 + W_FCAST * fcast + W_UTIL * util

    with weights ``(W_RECON, W_F1, W_FCAST, W_UTIL) = (1.0, 1.0, 1.0, 0.5)`` and
    ``MARGIN_SCALE = 0.1`` (a per-position mean log-lik margin of ~0.1 is a strong probe
    win, so it maps to a full point of forecast reward). All four terms are in a comparable
    [0, ~1] range so the weighted sum is a reasonable single-number proxy. Among
    reconstructing codecs, the best-utilized + most-forecastable wins — so after
    right-sizing the codebook the well-utilized codec is preferred. Revisit the weighting if
    one term starts to dominate selection in practice.

    Args:
        gate_dict:   a gate dict as produced by :func:`compute_gate` (keys ``forecastability``,
                     ``decode``, ``utilization``).
        recon_floor: envelope_corr disqualification floor. Defaults to
                     :data:`DEFAULT_RECON_FLOOR`; callers with a cfg pass
                     ``cfg.gate_recon_floor``.
    """
    fc = gate_dict["forecastability"]  # type: ignore[index]
    dec = gate_dict["decode"]  # type: ignore[index]
    ut = gate_dict["utilization"]  # type: ignore[index]

    # --- hard disqualification: reconstruction genuinely failed --------------------------
    env_corr = float(dec["envelope_corr"])  # type: ignore[index]
    if math.isnan(env_corr) or env_corr < recon_floor:
        return float("-inf")

    # --- soft, clamped/normalized sub-metrics (all in ~[0, 1]) ---------------------------
    def _clamp01(v: float) -> float:
        if math.isnan(v):
            return 0.0
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    MARGIN_SCALE = 0.1  # margin_transition ~0.1 => a full point of forecast reward
    W_RECON, W_F1, W_FCAST, W_UTIL = 1.0, 1.0, 1.0, 0.5

    recon = _clamp01(env_corr)
    f1 = _clamp01(float(dec["peak_f1"]))  # type: ignore[index]
    fcast = _clamp01(float(fc["margin_transition"]) / MARGIN_SCALE)  # type: ignore[index]
    # utilization is a SOFT reward (never a hard gate): mean of per-dim entropy + the
    # eval-size-relative observable fraction. Missing keys default to 0 (no reward).
    min_dim_entropy = _clamp01(float(ut.get("min_dim_entropy", 0.0)))  # type: ignore[union-attr]
    frac_obs = _clamp01(float(ut.get("frac_of_observable", 0.0)))  # type: ignore[union-attr]
    util = 0.5 * min_dim_entropy + 0.5 * frac_obs

    return W_RECON * recon + W_F1 * f1 + W_FCAST * fcast + W_UTIL * util


class _EMA:
    """Cheap exponential-moving-average shadow of a module's parameters.

    Maintains a detached CPU/-device shadow ``dict[name -> tensor]`` of the codec's params
    (float tensors EMA'd; non-float buffers/params copied as-is). ``update`` is O(#params)
    per step; ``state_dict`` returns the shadow merged over the live state so it is a
    drop-in weight set for :meth:`SpectroCodec.load_state_dict`.
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


# ------------------------------------------------------------------------------------- #
# gate evaluation
# ------------------------------------------------------------------------------------- #
@torch.no_grad()
def compute_gate(
    codec: SpectroCodec,
    eval_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    frame_seq: torch.Tensor,
    cfg: SpectroCodecConfig,
) -> Dict[str, object]:
    """Compute the four §4.4 gate quantities on held-out data.

    Parameters
    ----------
    codec : SpectroCodec
        The (partially) trained codec.
    eval_pairs : list of (spec_a, spec_b)
        Held-out δ-shift pairs; used for **stability** (codes of a vs b) and
        **decode_fidelity** (recon of a vs a).
    frame_seq : (B, n_frames, C, F, T)
        Consecutive world-model frames; used for **persistence** (frame t vs t+1) and
        **forecastability** (whole sequence of codes).
    cfg : SpectroCodecConfig

    Returns
    -------
    dict with keys ``stability`` (float), ``persistence`` (float),
    ``forecastability`` (dict), ``decode`` (dict), ``utilization`` (dict — codebook
    usage / collapse detector), plus the mandate booleans ``pass_stability`` /
    ``pass_persistence`` / ``pass_utilization``.
    """
    was_training = codec.training
    codec.eval()

    # --- stability + decode on the held-out δ-pairs -----------------------------------
    stab_vals: List[float] = []
    dec_corr: List[float] = []
    dec_f1: List[float] = []
    dec_sharp: List[float] = []
    for spec_a, spec_b in eval_pairs:
        out_a = codec.forward(spec_a)
        _, codes_b = codec.quantize(codec.encode(spec_b))
        stab_vals.append(gate.stability(out_a["codes"], codes_b))
        dm = gate.decode_fidelity(out_a["recon"], spec_a)
        dec_corr.append(dm["envelope_corr"])
        dec_f1.append(dm["peak_f1"])
        dec_sharp.append(dm["sharpness"])

    stability_val = float(sum(stab_vals) / len(stab_vals))
    decode = {
        "envelope_corr": float(sum(dec_corr) / len(dec_corr)),
        "peak_f1": float(sum(dec_f1) / len(dec_f1)),
        "sharpness": float(sum(dec_sharp) / len(dec_sharp)),
    }

    # --- persistence + forecastability on the consecutive-frame sequence --------------
    B, n_frames = frame_seq.shape[0], frame_seq.shape[1]
    # encode every frame -> (B, n_frames, n_tok, fsq_dim) codes
    flat = frame_seq.reshape(B * n_frames, *frame_seq.shape[2:])  # (B*Fr, C, F, T)
    _, codes_flat = codec.quantize(codec.encode(flat))
    codes_seq = codes_flat.reshape(B, n_frames, cfg.n_tok, cfg.fsq_dim)

    # persistence over the first consecutive step (frame 0 -> frame 1)
    persistence_val = gate.persistence(codes_seq[:, 0], codes_seq[:, 1])

    # forecastability over the full sequence; DEFAULT code-derived transition split
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
        f"minH={ut['min_dim_entropy']:.3f}]"  # type: ignore[index]
        f"({'ok' if g['pass_utilization'] else 'COLLAPSED'}) "
        f"forecast_margin_tr={fc['margin_transition']:+.4f}"  # type: ignore[index]
        f"(beats={fc['beats_persistence']}) "  # type: ignore[index]
        f"decode[corr={dec['envelope_corr']:.3f} "  # type: ignore[index]
        f"f1={dec['peak_f1']:.3f} "  # type: ignore[index]
        f"sharp={dec['sharpness']:.3f}]"  # type: ignore[index]
    )


# ------------------------------------------------------------------------------------- #
# DDP-safe divergence guard (shared by every adversarial codec train step)
# ------------------------------------------------------------------------------------- #
# Running count of skipped optimizer steps (across ALL codecs in this process). Rank-0 logs
# it so a firing guard is visible in the .err trace. Reset with :func:`reset_skipped_steps`.
_SKIPPED_STEPS: int = 0


def reset_skipped_steps() -> None:
    """Zero the process-wide skipped-optimizer-step counter (call once at trainer start)."""
    global _SKIPPED_STEPS
    _SKIPPED_STEPS = 0


def skipped_steps() -> int:
    """Return the process-wide count of optimizer steps skipped by the divergence guard."""
    return _SKIPPED_STEPS


def is_step_diverged(*losses: torch.Tensor) -> bool:
    """Collective-safe test: should EVERY rank skip this optimizer step?

    A codec that collapses (co2 / video, 2026-07) drives the adaptive adversarial weight and
    the generator loss to Inf/NaN on ONE rank; if that rank silently corrupts its params while
    the others step cleanly, the ranks desync and the next collective hits the NCCL watchdog ->
    SIGTERM (exit 143). This guard makes the skip decision IDENTICAL on all ranks:

    * local bad-flag = 1.0 if ANY of ``losses`` is non-finite on THIS rank, else 0.0;
    * under DDP (``dist.is_initialized()`` and ``world_size > 1``): ``all_reduce(MAX)`` the flag
      so every rank sees the SAME value — if ANY rank is bad, ALL ranks return True and skip;
    * non-distributed (``world_size == 1``): just the local flag.

    IMPORTANT: this only gates ``optimizer.step()``. The caller must still run ``backward()``
    (and thus the gradient all-reduce) UNIFORMLY on every rank before calling this; only the
    parameter update is conditionally + uniformly skipped, so the collective stays symmetric.

    When True, the caller also bumps the skipped-step counter via :func:`note_skipped_step`.
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


# ------------------------------------------------------------------------------------- #
# shared per-step generator/discriminator alternation (reused by run_spike AND the
# streaming DDP trainer in train_codec.py)
# ------------------------------------------------------------------------------------- #
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
    """ONE generator+discriminator alternation step on a δ-shift pair.

    This is the SINGLE source of truth for the per-step training math shared by
    :func:`run_spike` (fixed pre-built pool, 1-GPU) and the streaming DDP trainer
    (:mod:`train_codec`). It performs, in order:

      1. **generator (codec) step** — ``codec.generator_losses(spec_a, spec_b, disc, cfg,
         step)``; backprop ``total`` into the codec; ``opt_g.step()``;
      2. **discriminator step** — a fresh, *detached* recon of ``spec_a``, scored against
         the real ``spec_a`` via :func:`losses.discriminator_loss`; ``opt_d.step()``.

    ``codec`` / ``disc`` may be raw modules OR DDP-wrapped: ``generator_losses`` is a method
    on ``SpectroCodec``, so under DDP pass the *unwrapped* codec here for the generator loss
    but the *wrapped* module for the forward that DDP must hook. The DDP trainer handles that
    by calling ``codec(spec_a)`` through the wrapper inside ``generator_losses`` — see
    :mod:`train_codec` for how it composes this. For the 1-GPU path both are raw modules.

    Parameters
    ----------
    codec, disc : the codec (generator) and external discriminator (raw or DDP-wrapped).
    opt_g, opt_d : their Adam optimizers.
    spec_a, spec_b : (B, C, F, T) δ-shift pair already on the target device.
    cfg : SpectroCodecConfig.
    step : int — global step (drives ``cfg.adv_warmup_steps``).

    Returns
    -------
    (g_terms, d_loss)
        ``g_terms`` is the dict from :meth:`SpectroCodec.generator_losses` (keys include
        ``total`` / ``adversarial`` / ``pixel`` / ``consistency`` / ``entropy`` /
        ``adaptive_weight`` / ``recon`` / ``codes``); ``d_loss`` is the scalar discriminator
        loss tensor. Both are LIVE tensors for the caller to ``.detach()`` when logging.
    """
    codec.train()

    # ---- generator (codec) step ----
    opt_g.zero_grad(set_to_none=True)
    g_terms = codec.generator_losses(spec_a, spec_b, disc, cfg, step=step)
    g_terms["total"].backward()
    # DDP-safe divergence guard: backward (+ its grad all-reduce) already ran uniformly; only
    # opt_g.step() is conditionally + UNIFORMLY skipped so a collapsing codec can't desync DDP.
    if is_step_diverged(g_terms["total"]):
        note_skipped_step()
    else:
        opt_g.step()

    # ---- discriminator step (fresh recon, detached from the codec graph) ----
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


# ------------------------------------------------------------------------------------- #
# the spike
# ------------------------------------------------------------------------------------- #
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
    """Train the codec with GAN alternation on ``batches``; gate every ``eval_every`` steps.

    Parameters
    ----------
    cfg : SpectroCodecConfig
    batches : list of (spec_a, spec_b)
        δ-shift training pairs (see :func:`synthetic_batches`). Cycled if ``steps`` exceeds
        ``len(batches)``. ``spec_a`` is both the recon target and the code source; ``spec_b``
        is its δ-shifted nuisance pair for the consistency term.
    steps : int
        Number of generator/discriminator alternation steps.
    device : str | torch.device
    eval_every : int
        Compute + log the gate every this many steps (and always after the last step).
    lr, disc_lr : float
        Adam learning rates for the codec (generator) and discriminator
        (``disc_lr`` defaults to ``lr``).
    eval_pairs : optional held-out δ-pairs for the gate. Built synthetically if None.
    frame_seq : optional held-out consecutive-frame sequence for persistence /
        forecastability. Built synthetically if None.
    eval_batch_size, eval_frames : sizes for the auto-built held-out eval data.
    log_fn : callable(str) or None
        Per-eval logger (default ``print``); pass ``None`` to silence.
    resume_state : dict or None
        Optional checkpoint (as written by the CLI ``main``) to warm-start from. Keys used:
        ``codec`` / ``disc`` / ``opt_g`` / ``opt_d`` state_dicts and ``step`` (the number of
        steps ALREADY completed). Training then runs for ``steps`` MORE steps, and the
        global step numbers logged / passed to ``on_eval`` continue from ``resume_state``.
        This is a warm-start hook only — it does not change the training math.
    on_eval : callable(step, codec, opt_g, opt_d, disc, gate_dict) or None
        Called at every gate evaluation (and after the last step) so the CLI can persist a
        ``gate_<step>.json`` and a ``codec_last.pt`` checkpoint. ``step`` is the global step.
        The ``gate_dict`` handed in additionally carries a ``score`` key (the composite
        :func:`gate_score`, ``-inf`` only when reconstruction fails) and an ``is_best`` flag.
    ema : bool
        If True, maintain an exponential-moving-average shadow of the CODEC parameters
        (updated every training step) and expose it to ``on_ema`` at each eval. Cheap: a
        detached shadow param dict. Default off.
    ema_decay : float
        EMA decay in [0, 1); default 0.999.
    on_best : callable(step, codec, disc, gate_dict) or None
        Called ONLY when the composite :func:`gate_score` strictly improves on the best seen
        so far (a recon-failed eval, score ``-inf``, never becomes best). Lets the CLI persist
        a ``codec_best.pt`` frozen at the best eval. ``gate_dict`` carries ``score``/``is_best``.
    on_ema : callable(step, ema_state_dict, gate_dict) or None
        Called at each eval when ``ema`` is True, with the current EMA weight state_dict, so
        the CLI can persist ``codec_ema.pt``.

    Returns
    -------
    The final gate dict from :func:`compute_gate` (keys: ``stability``, ``persistence``,
    ``forecastability``, ``decode``, ``utilization``, ``pass_stability``,
    ``pass_persistence``, ``pass_utilization``), with an extra ``steps`` key recording how
    many steps were run and a ``global_step`` key giving the last global step number
    (== ``start_step + steps``). Also carries ``best_score`` / ``best_step`` (the composite
    score and global step of the best eval, ``None``/``-inf`` if every eval failed recon).
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

    # optional warm-start (additive; does not alter the training math below)
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

    # optional EMA shadow of the codec params (built AFTER any warm-start load).
    ema_shadow: Optional[_EMA] = _EMA(codec, ema_decay) if ema else None

    # best-checkpoint tracking (composite gate score; higher is better).
    best_score = float("-inf")
    best_step: Optional[int] = None

    # held-out eval data (never trained on) — moved to device to keep run_spike device-correct
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

        # ---- one generator+discriminator alternation step (shared per-step math) ----
        g_terms, d_loss = codec_train_step(
            codec, disc, opt_g, opt_d, spec_a, spec_b, cfg, step=step
        )

        # ---- EMA shadow update (every step; cheap) ----
        if ema_shadow is not None:
            ema_shadow.update(codec)

        is_last = local_step == steps - 1
        if (step % eval_every == 0) or is_last:
            g = compute_gate(codec, eval_pairs, frame_seq, cfg)
            g["g_total"] = float(g_terms["total"].detach())
            g["d_loss"] = float(d_loss.detach())
            g["adaptive_weight"] = float(g_terms["adaptive_weight"])
            g["step"] = step

            # composite selection score (-inf ONLY if reconstruction fails; utilization is
            # a soft reward, not a hard gate — see gate_score docstring).
            score = gate_score(g, recon_floor=cfg.gate_recon_floor)
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


# ------------------------------------------------------------------------------------- #
# CLI entry point (Phase-A gate-spike run)
# ------------------------------------------------------------------------------------- #
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
    """Write ``codec_last.pt`` (codec + disc + both optimizers + step) for ``--resume``.

    ``step`` here is the 0-indexed current step; the checkpoint records the COUNT of
    completed steps (``step + 1``) so that :func:`run_spike`'s ``resume_state["step"]``
    (= steps already completed) makes the resumed global step numbers continue seamlessly.
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


def _save_best_checkpoint(
    out_dir: Path,
    step: int,
    score: float,
    codec: SpectroCodec,
    disc: FreqAwarePatchGAN,
    cfg: SpectroCodecConfig,
    gate_dict: Dict[str, object],
) -> Path:
    """Write ``codec_best.pt`` — the codec frozen at the best composite-gate eval.

    Records the codec + disc state, the ``step`` (COUNT of completed steps, matching
    :func:`_save_checkpoint`), the composite ``score``, and the full ``gate`` dict so the
    frozen checkpoint is self-describing. Written atomically (tmp + replace).
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
        description="IGNITE Phase-A gate spike (1-GPU broad/firm run).",
    )
    p.add_argument("--n_shots", type=int, default=40,
                   help="Number of shots to auto-discover for the data pool (broad).")
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
                   help="Use synthetic sum-of-sinusoid data (NO real HDF5); for smoke tests "
                        "and dry-runs.")
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

    # ece codec config: real ece is 40 channels. real_ece_batches would overwrite
    # cfg.channels to the loaded count anyway; set it here so the synthetic path matches.
    cfg = SpectroCodecConfig(channels=40)
    if args.entropy_weight is not None:
        cfg.entropy_weight = float(args.entropy_weight)
    if args.consistency_weight is not None:
        cfg.consistency_weight = float(args.consistency_weight)

    shots: Optional[List[str]] = None
    if args.shots is not None:
        shots = [s.strip() for s in args.shots.split(",") if s.strip()]

    # ---- build the pool ONCE (train batches + held-out eval slice + frame sequence) ----
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

    # ---- persistence callback: write gate JSON + rolling checkpoint at every eval ----
    def on_eval(step, codec, opt_g, opt_d, disc, gate_dict):
        gp = _write_gate_json(out_dir, step, gate_dict)
        cp = _save_checkpoint(out_dir, step, codec, opt_g, opt_d, disc, cfg)
        print(f"[spike-cli] step {step}: wrote {gp.name} + {cp.name}", flush=True)

    # ---- best-by-gate callback: freeze codec_best.pt when the composite score improves ----
    def on_best(step, codec, disc, gate_dict):
        bp = _save_best_checkpoint(
            out_dir, step, float(gate_dict["score"]), codec, disc, cfg, gate_dict
        )
        print(f"[spike-cli] step {step}: NEW BEST score={gate_dict['score']:+.4f} "
              f"-> wrote {bp.name}", flush=True)

    # ---- optional EMA callback: write codec_ema.pt (EMA weights) at every eval ----
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

    # ---- summary ----
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
