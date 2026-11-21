"""IGNITE Phase-A streaming DDP codec trainer.

The spike (:mod:`ignite.spike`) builds a fixed batch pool with
``spike.real_ece_batches``, which opens shots one after another (~19 min for 40 shots)
and tops out around 40 shots. Production codecs need to see thousands of shots, so this
module streams them through
:class:`~tokamak_foundation_model.data.multi_file_dataset.TokamakMultiFileDataset`, the
class the foundation-model trainer uses under DDP. It already provides the global index
to ``(file_idx, chunk_idx)`` map, a per-worker LRU cache of HDF5 handles that pickles
correctly into DataLoader workers, a length-cache sidecar and ``make_dataloader``.

Design
------
* Data: :class:`CodecPairDataset` subclasses ``TokamakMultiFileDataset`` and overrides
  only the per-item hook (:meth:`_getitem_standard`), returning the codec's
  ``(spec_a, spec_b)`` δ-shift log-power pair for one spectro modality instead of the
  multi-modal training dict. The parent's ``__getitem__`` still maps the index and sets
  ``self.h5_file`` before calling the hook, so file-handle safety with
  ``num_workers > 0`` is inherited. The video, slow-TS datasets follow the same pattern.
* Training: the per-step math is :func:`ignite.spike.codec_train_step` (generator and
  discriminator alternation). The codec is wrapped in DDP through
  :class:`_GenLossAdapter`, whose ``forward`` calls ``generator_losses``, so DDP's
  reducer sees the generator loss; the discriminator has its own optimizer and DDP
  wrapper.
* Gate and best checkpoint: a held-out eval set (disjoint shots) is scored with the §4.4
  gate (:func:`ignite.spike.compute_gate`) and the best checkpoint is chosen by
  :func:`ignite.spike.gate_score`, as in the spike. Only rank 0 logs and writes.

Reuse boundary (§7): ``torch``, sibling ``ignite`` modules and the read-only data
pipeline (``data.data_loader`` / ``data.multi_file_dataset``). No FAITH model code.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from . import data, gate, spike
from .codec import SpectroCodec
from .config import (
    CHUNK_S,
    SLOWTS_FS,
    STFT_FS,
    SLOWTS_PREPROCESS_METHOD,
    SLOWTS_SIGNALS as _SLOWTS_SIGNALS,
    SlowTSCodecConfig,
    SpectroCodecConfig,
    VideoCodecConfig,
    slowts_patch_for,
)
from .discriminator import FreqAwarePatchGAN
from .slow_ts_codec import SlowTSCodec
from .video_codec import VideoCodec
from .video_discriminator import FramePatchGAN

# Spectro modalities. All are apply_stft=True with target_fs=500e3 (== STFT_FS) in
# TokamakH5Dataset.SIGNAL_CONFIGS, so their raw windows STFT onto the grid
# data.log_power_stft expects. Channel counts come from the loader's SignalConfig at
# runtime.
# mirnov (29 magnetic channels) is also the STFT front end the band-power tokeniser
# reads through; its loading/STFT parameters come from its SIGNAL_CONFIGS entry.
SPECTRO_MODALITIES = ("ece", "co2", "bes", "mhr", "mirnov")

# Video (tangtv) modalities, one codec per divertor. Each is a MovieConfig in
# TokamakH5Dataset.MOVIE_CONFIGS: tangtv_lower keeps raw camera channels [0, 2] (LODIV
# PAR-int and PERP), tangtv_upper keeps [4, 6] (UPDIV0 PERP and PAR). Both have 2
# channels after channels_to_use (read at runtime).
VIDEO_MODALITIES = ("tangtv_lower", "tangtv_upper")

# Slow-TS modalities: the 7 kinetic-profile signals (Thomson / CER / MSE), non-STFT
# SignalConfigs at target_fs=100 Hz. Thomson uses zero_is_missing, CER/MSE an explicit
# NaN mask (see config.SLOWTS_* and data_loader). One codec per signal, each frozen on
# its own (§4.3).
SLOWTS_MODALITIES = tuple(_SLOWTS_SIGNALS)

# Fast-TS (filterscopes) modality, trained by fastts_train; main() dispatches
# --modality filterscopes there. Must equal fastts_train.FASTTS_MODALITY; defined here
# rather than imported because fastts_train imports from this module.
FASTTS_MODALITY = "filterscopes"

DEFAULT_DATA_DIR = spike.DEFAULT_DATA_DIR


# Per-modality anti-collapse overrides
#
# Applied to the cfg in main() after channel-count sizing, only for codecs that
# collapsed (to one code from step 0: frac_codes_used=0.001, minH~0, env_corr~0/nan).
# Unlisted modalities (ece, bes, cer_ti, cer_rot, ts_core_temp) keep the config
# defaults, i.e. active_bias 0 (no stratification).
#
# Thresholds come from the fraction of active windows over a spread of real shots:
#   * co2 (spectro): only ~33% of windows have co2 data, and the log-power windows have
#     a median std of ~0.024 (5-95 pct 0.009-0.318) against ~0.85 for ece/bes/mhr, so
#     co2 is mostly a floor at ~-10. min_activity=0.10 picks roughly the top 10-20% most
#     structured windows. co2 also had adversarial sharpness spikes (13-509), so it also
#     gets the adversarial warmup and a lower adversarial_weight.
#   * filterscopes (fast-TS): ~76% of envelope windows saturate the log1p ceiling to a
#     constant (std 0) and ~21% are strongly structured (env std >= 2.4);
#     min_activity=0.5 separates them. The saturation is a scale artifact (raw counts
#     ~1e13-1e15), which stratification does not fix.
#   * ts_core_density (slow-TS, masked): bimodal present fraction, median ~0.07 but ~34%
#     of windows >= 0.75. min_activity=0.5 (present fraction) favours the ~38% of
#     well-observed windows.
#   * tangtv_lower (video): not floor-dominated (every clip has frame std >= ~3.8). It
#     reached env_corr 0.15 and then fell back with adversarial sharpness spikes, an
#     adversarial collapse, so it gets no activity stratification, only the adversarial
#     warmup and lower weight.
#
# adv_warmup_steps and adversarial_weight are read by generator_losses (spectro and
# video), so setting them here only affects the codecs listed.
_activity_overrides: Dict[str, Dict[str, float]] = {
    # spectro
    "co2": {"min_activity": 0.10, "active_bias": 0.5,
            "adv_warmup_steps": 1500, "adversarial_weight": 0.5},
    # mhr (spectro): adversarial knobs only, like co2's; mhr is O(1) and not
    # floor-dominated, so no activity bias.
    "mhr": {"adv_warmup_steps": 1500, "adversarial_weight": 0.5},
    # mirnov (spectro, 29 magnetic channels): with no entry, and again with mhr's
    # adversarial-only knobs, it collapsed by step 1000. Its data is clean (no dead or
    # flat channels, per-freq std median ~1.07) but its per-freq mean reaches -9.996,
    # the same large DC offset as co2, so it gets co2's full recipe.
    "mirnov": {"min_activity": 0.10, "active_bias": 0.5,
               "adv_warmup_steps": 1500, "adversarial_weight": 0.5},
    # video (adversarial instability only; no activity bias)
    "tangtv_lower": {"adv_warmup_steps": 1500, "adversarial_weight": 0.5},
    # slow-TS (masked; activity = present fraction). No discriminator, so no adv knobs.
    "ts_core_density": {"min_activity": 0.5, "active_bias": 0.5},
    # mse (Motional Stark Effect, slow-TS) depends on the neutral beams: ~75% of windows
    # are <10% present (NBI off), median present fraction 0.000. Without stratification
    # the codec collapses to one code and its decode correlation oscillates. Same
    # stratification as ts_core_density (favours the ~25% well-observed windows); see
    # also the non-finite channel masking in _fit_window.
    "mse": {"min_activity": 0.5, "active_bias": 0.5},
    # ts_tangential_* (slow-TS): the same bimodal present fraction as ts_core_density
    # (over 12 shots: median 0.067, 59% of windows <= 0.1 present, 34.75% >= 0.5;
    # density and temperature share the mask). Unstratified, ts_tangential_density
    # degraded over an 80k-step run (decode corr 0.30 -> 0.03, distinct codes 11 -> 5)
    # as mostly-missing windows took over the batches. Slow-TS stratification only works
    # since the redraw bound fix in _chunks_in_current_file.
    "ts_tangential_density": {"min_activity": 0.5, "active_bias": 0.5},
    "ts_tangential_temp": {"min_activity": 0.5, "active_bias": 0.5},
    # fast-TS (envelope std). After the scale fix the envelope is rich (median
    # within-window std 0.648, 91% of windows structured), yet the codec still pinned to
    # one code, even with entropy_weight=5.0. It was the only collapse-prone codec at
    # adversarial_weight 1.0 with no warmup, so the discriminator pushed a fresh encoder
    # into collapse from step 0; the fix is the same warmup and halved weight as the
    # others.
    #
    # apply_activity_overrides skips this entry for the raw-sample codec (cfg.is_raw):
    # 0.5 is a threshold on envelope std (median 0.648), while raw windows have a median
    # std of ~0.007 in standardised units, so it would reject nearly every window; and
    # adversarial_weight 0.5 would re-enable the GAN the raw codec has off by default.
    "filterscopes": {"min_activity": 0.5, "active_bias": 0.5,
                     "adv_warmup_steps": 1500, "adversarial_weight": 0.5},
}

# Modalities that need whole-shot presence filtering. A diagnostic recorded on only some
# shots stores a length-1 (C, 1) placeholder on the others; it floors to silence, swamps
# the batch and collapses the FSQ to one code. Within-shot stratification cannot help a
# shot with no active window, so such shots are dropped up front with
# filter_signal_present_files. Only co2 qualifies: absent on the older shot range
# (placeholder (4, 1)), present on the newer one ((4, ~4.5M)). ece/bes/mhr are present
# broadly. Slow-TS beam gaps (cer/mse) are handled by masking instead.
_PRESENCE_FILTER_SIGNALS: frozenset = frozenset({"co2"})

# Spectro modalities that get raw per-channel input standardisation. For thin modalities
# the log-power window is a large near-constant plate (co2 ~20 per freq, across-window
# std ~0.005 for most bins) with signal in only a few bins. MAE on that plate is
# minimised by a constant, so the FSQ collapses to one code whatever the token count
# (192 and 16 tokens both collapsed). Rescaling the input lifts the informative bins to
# O(1). co2 keeps the 192-token resolution: the input scale, not the token count, is
# what matters. Not needed for ece/bes/mhr, which are O(1) already.
_SPECTRO_STANDARDIZE_SIGNALS: frozenset = frozenset({"co2"})


def apply_spectro_standardization(cfg, modality: str, stats_path=None, log_fn=None) -> None:
    """Turn on raw per-channel input standardisation on ``cfg`` in place.

    No-op for modalities not in :data:`_SPECTRO_STANDARDIZE_SIGNALS`. Loads the FM's
    ``preprocessing_stats[modality]['raw']`` per-channel (C,) mean/std (raw units, so no
    log-unit mismatch) and sets ``cfg.input_standardize``, ``cfg.raw_mean`` and
    ``cfg.raw_std`` (saved in the checkpoint). ``log_power_stft`` z-scores the raw
    signal per channel before the STFT, which moves co2's log-power out of the _LOG_CEIL
    clip. Must run before the codec and dataset are built. Missing or degenerate stats
    leave standardisation off with a warning.
    """
    if modality not in _SPECTRO_STANDARDIZE_SIGNALS:
        return
    import numpy as _np
    from .fastts_train import DEFAULT_STATS_PATH
    path = stats_path or DEFAULT_STATS_PATH
    if not path or not Path(path).exists():
        if log_fn is not None:
            log_fn(f"[train_codec] WARN {modality}: stats {path!r} missing; RAW standardization OFF")
        return
    st = torch.load(path, map_location="cpu", weights_only=False)
    raw = st.get(modality, {}).get("raw") if isinstance(st, dict) else None
    if raw is None:
        if log_fn is not None:
            log_fn(f"[train_codec] WARN {modality}: no ['raw'] stats; RAW standardization OFF")
        return
    mean = _np.asarray(raw["mean"], dtype=_np.float32)   # (C,)
    std = _np.asarray(raw["std"], dtype=_np.float32)
    # A non-finite or zero-std channel would break the z-score: use centre 0, scale 1.
    mean = _np.where(_np.isfinite(mean), mean, 0.0)
    std = _np.where(_np.isfinite(std) & (std > 0), std, 1.0)
    if mean.shape[0] != cfg.channels:
        raise ValueError(
            f"{modality} raw stats C={mean.shape[0]} != cfg.channels={cfg.channels}; stale {path}"
        )
    cfg.input_standardize = True
    cfg.raw_mean = mean.tolist()
    cfg.raw_std = std.tolist()
    if log_fn is not None:
        log_fn(
            f"[train_codec] RAW input standardization ON for {modality}: per-channel z-score "
            f"(raw mean~{float(mean.mean()):.2e}, std~{float(std.mean()):.2e}) "
            f"-> log-power lands un-clipped in [-10, 20]"
        )


def apply_activity_overrides(cfg, modality: str, log_fn=None) -> None:
    """Apply the per-modality anti-collapse overrides to ``cfg`` in place.

    No-op for modalities not in :data:`_activity_overrides`. Each field is set only if
    ``cfg`` has it (video and slow-TS lack some adversarial knobs).
    """
    ov = _activity_overrides.get(modality)
    if not ov:
        return
    # The fast-TS entry is calibrated for the envelope codec only (see the table).
    # getattr so a config pickled before `target` existed takes the envelope branch.
    if modality == FASTTS_MODALITY and getattr(cfg, "is_raw", False):
        if log_fn is not None:
            log_fn(f"[train_codec] anti-collapse overrides for {modality}: SKIPPED "
                   f"(raw-sample codec; the entry is envelope-calibrated)")
        return
    applied = {}
    for k, v in ov.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
            applied[k] = v
    if log_fn is not None and applied:
        log_fn(f"[train_codec] anti-collapse overrides for {modality}: {applied}")


def compute_logpow_stats(
    modality: str,
    shots: Sequence[Union[str, int]],
    out_path: Union[str, Path],
    data_dir: Union[str, Path] = None,
    windows_per_shot: int = 4,
    seed: int = 0,
    log_fn=None,
    stft_n_fft: Optional[int] = None,
    stft_hop: Optional[int] = None,
    freq_bins: Optional[int] = None,
    time_frames: Optional[int] = None,
) -> dict:
    """Dataset-level per-(channel, freq) log-power mean/std for ``--logpow_stats_path``.

    The stats are computed in the space the codec sees, i.e. after
    ``apply_spectro_standardization``'s raw z-score for the modalities in
    ``_SPECTRO_STANDARDIZE_SIGNALS`` (co2). Without it co2's log-power is all clipped at
    the +20 ceiling and the stats degenerate (mean 20, std 0); the old
    ``codec_co2_perfreq_stats.pt`` on foundation_model_meta is like that, so regenerate
    it with this function.

    ``mean`` is pooled over all sampled windows. ``std`` is the within-shot scale,
    sqrt(mean over shots of each shot's variance about its own mean): the pooled std is
    dominated by differences between shots (~5x the within-shot scale for ece/mhr), and
    dividing by it hid each shot's structure from the loss, so ece/mhr fell back to one
    code. ``windows_per_shot`` windows are taken evenly spread within each shot
    (deterministic; no cache files are read or written). Writes ``{mean, std, std_kind,
    n_windows, n_shots, modality, space}`` and the STFT geometry to ``out_path``,
    moving an existing file to ``.bak`` first.
    """
    cfg = SpectroCodecConfig()
    cfg.channels = modality_channels(modality)
    # STFT geometry (None keeps the default 1024/256/512/96). The (C, F) stats are only
    # valid on the grid they were computed on.
    for _name, _val in (("stft_n_fft", stft_n_fft), ("stft_hop", stft_hop),
                        ("freq_bins", freq_bins), ("time_frames", time_frames)):
        if _val is not None:
            setattr(cfg, _name, int(_val))
    apply_spectro_standardization(cfg, modality, log_fn=log_fn)
    ddir = data_dir if data_dir is not None else DEFAULT_DATA_DIR

    s1 = torch.zeros(cfg.channels, cfg.freq_bins, dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    within_var = torch.zeros_like(s1)
    frames = 0
    n_windows_used = 0
    n_shots_used = 0
    for shot in shots:
        try:
            ds = CodecPairDataset(modality, [shot], cfg, data_dir=ddir, seed=seed)
        except ValueError:
            continue                                    # missing file -> skip
        n = len(ds)
        if n == 0:
            continue
        k = min(int(windows_per_shot), n)
        idxs = sorted({round(i * (n - 1) / max(1, k - 1)) for i in range(k)})
        x = torch.cat([ds[i][0].to(torch.float64) for i in idxs], dim=-1)  # (C, F, k*T)
        s1 += x.sum(dim=-1)
        s2 += (x ** 2).sum(dim=-1)
        frames += x.shape[-1]
        n_windows_used += len(idxs)
        within_var += x.var(dim=-1, unbiased=False)     # per-shot variance, own mean
        n_shots_used += 1
    if n_shots_used == 0:
        raise RuntimeError(f"compute_logpow_stats: no usable shots for {modality}")
    mean = s1 / frames
    std = (within_var / n_shots_used).clamp_min(0.0).sqrt()
    out = {
        "mean": mean.float().tolist(),
        "std": std.float().tolist(),
        "std_kind": "within_shot",
        "n_windows": n_windows_used,
        "n_shots": n_shots_used,
        "modality": modality,
        "space": ("codec_log_power_stft__post_raw_std"
                  if cfg.input_standardize else "codec_log_power_stft"),
        # The STFT grid these stats belong to, since loading 512-bin stats into a
        # 256-bin codec would be a silent error.
        "stft_n_fft": int(cfg.stft_n_fft),
        "stft_hop": int(cfg.stft_hop),
        "freq_bins": int(cfg.freq_bins),
        "time_frames": int(cfg.time_frames),
    }
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        out_p.replace(out_p.with_suffix(out_p.suffix + ".bak"))
    torch.save(out, out_p)
    if log_fn is not None:
        log_fn(
            f"[compute_logpow_stats] {modality}: {n_windows_used} windows over "
            f"{n_shots_used} shots -> {out_p} (space={out['space']}, std_kind=within_shot, "
            f"mean med={float(mean.median()):.3f}, std med={float(std.median()):.4f})"
        )
    return out


def _import_multifile():
    """Import the production multi-file dataset and samplers.

    Kept in a function so the import sits in one place; it runs at module import below.
    """
    from tokamak_foundation_model.data.multi_file_dataset import (
        DistributedTwoLevelSampler,
        TokamakMultiFileDataset,
        TwoLevelSampler,
    )

    return TokamakMultiFileDataset, TwoLevelSampler, DistributedTwoLevelSampler


# Bind the dataset and samplers at import; CodecPairDataset subclasses
# TokamakMultiFileDataset directly.
(
    TokamakMultiFileDataset,
    TwoLevelSampler,
    DistributedTwoLevelSampler,
) = _import_multifile()


# Channel counts (read from the loader config)
def _channels_after_selection(num_channels: int, channels_to_use) -> int:
    """Channel count after applying a ``channels_to_use`` slice / index-list / None."""
    if channels_to_use is None:
        return num_channels
    if isinstance(channels_to_use, slice):
        return len(range(*channels_to_use.indices(num_channels)))
    return len(list(channels_to_use))


def modality_channels(modality: str) -> int:
    """Channels loaded for ``modality`` after ``channels_to_use``.

    Reads the class-level ``TokamakH5Dataset.SIGNAL_CONFIGS`` (spectro, slow-TS) or
    ``MOVIE_CONFIGS`` (video). Like ``spike._num_ece_channels`` for any codec modality.
    """
    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

    if modality == FASTTS_MODALITY:
        # Fast-TS uses the fast-TS trainer's helper (imported lazily: fastts_train
        # imports from this module).
        from .fastts_train import fastts_channels
        return fastts_channels(modality)

    if modality in VIDEO_MODALITIES:
        mv = next(
            (m for m in TokamakH5Dataset.MOVIE_CONFIGS if m.name == modality), None
        )
        if mv is None:
            raise ValueError(f"unknown video modality {modality!r}; not in MOVIE_CONFIGS")
        return mv.channels

    cfg_sig = next(
        (c for c in TokamakH5Dataset.SIGNAL_CONFIGS if c.name == modality), None
    )
    if cfg_sig is None:
        raise ValueError(f"unknown modality {modality!r}; not in SIGNAL_CONFIGS")
    # Slow-TS signals are raw (non-STFT), spectro signals are STFT; any other non-STFT
    # signal is not a codec modality.
    if not cfg_sig.apply_stft and modality not in SLOWTS_MODALITIES:
        raise ValueError(
            f"modality {modality!r} is not a spectro (apply_stft) or slow-TS signal"
        )
    return _channels_after_selection(cfg_sig.num_channels, cfg_sig.channels_to_use)


def video_divertor(modality: str) -> str:
    """Map a video modality name to its divertor selector ('lower' / 'upper')."""
    if modality == "tangtv_lower":
        return "lower"
    if modality == "tangtv_upper":
        return "upper"
    raise ValueError(f"{modality!r} is not a video modality {VIDEO_MODALITIES}")


# Per-channel tangtv liveness for every shot (scripts/data_preparation/
# scan_video_channels.py): {shot_int: [7 flags]}, a flag is 1 when that camera's middle
# frame is finite. The video presence filter needs only this one torch.load and no HDF5
# scan (a cold scan at job time hits the NCCL watchdog).
VIDEO_LIVENESS_CACHE = Path(
    "/lustre/orion/fus187/proj-shared/foundation_model_meta/video_channel_liveness.pt"
)


# Per-channel spectro liveness (scan_video_channels.py --family spectro), used the same
# way as VIDEO_LIVENESS_CACHE.
# {shot_int: {modality: {"n", "t0", "t1", "live" [C flags], "live_frac" [C floats]}}}.
SPECTRO_LIVENESS_CACHE = Path(
    "/lustre/orion/fus187/proj-shared/foundation_model_meta/spectro_channel_liveness.pt"
)


def derive_presence_lengths_sidecar(
    modality: str,
    lengths_cache_dir: Union[str, Path],
    *,
    shots: Sequence[Union[str, int]],
    data_dir: Union[str, Path],
    require_all: bool = False,
    log_fn=None,
) -> Optional[Path]:
    """Write ``codec_<mod>_presence_lengths.pt`` by subsetting the unfiltered sidecar.

    ``TokamakMultiFileDataset._load_or_compute_lengths`` reuses a sidecar only when its
    stored path list matches, so a presence-filtered shot list misses
    ``codec_<mod>_lengths.pt`` and every run cold-scans every shot at ~0.77 s per file
    (~98 min for ece's 7616 shots, most of a 2 h leg).

    No scan is needed: a file's chunk count depends only on its duration and the window
    geometry, and the filtered list is a subset of the unfiltered one, so the lengths
    can be copied.

    Writes nothing and returns None if any required path is missing from the unfiltered
    cache, since a partly wrong lengths cache would silently mis-index windows.
    """
    cdir = Path(lengths_cache_dir)
    src = cdir / f"codec_{modality}_lengths.pt"
    dst = cdir / f"codec_{modality}_presence_lengths.pt"
    if not src.exists():
        if log_fn:
            log_fn(f"[train_codec] no unfiltered sidecar at {src}; cannot derive {dst.name}")
        return None
    cache = torch.load(str(src), map_location="cpu", weights_only=False)
    have = dict(zip(cache["paths"], cache["lengths"]))
    want = [str(q) for q in _shot_paths(shots, data_dir)]
    missing = [q for q in want if q not in have]
    if missing:
        if log_fn:
            log_fn(f"[train_codec] REFUSING to write {dst.name}: {len(missing)} of "
                   f"{len(want)} paths are absent from {src.name} (first: "
                   f"{Path(missing[0]).name}). Let the dataset scan instead.")
        return None
    tmp = Path(f"{dst}.tmp.{os.getpid()}")
    torch.save({"paths": want, "lengths": [have[q] for q in want]}, tmp)
    tmp.replace(dst)
    if log_fn:
        log_fn(f"[train_codec] wrote {dst} ({len(want)} shots) by subsetting {src.name} "
               f"-- no HDF5 scan")
    return dst


def spectro_live_shots(
    modality: str,
    shots: Sequence[Union[str, int]],
    *,
    require_all: bool = False,
    liveness_path: Union[str, Path] = SPECTRO_LIVENESS_CACHE,
    log_fn=None,
) -> List[str]:
    """Keep only shots whose channels for this spectro modality recorded data.

    A shot whose HDF5 group is an empty ``(C, 1)`` stub never yields a real window:
    every draw falls through ``_draw_valid_pair``'s redraws to the eps-floor fallback
    pair, a constant plate that is then reconstructed and shown to the discriminator as
    real. (co2 also has the whole-shot filter in ``_PRESENCE_FILTER_SIGNALS``.)

    ``require_all`` also drops shots with any dead channel. Order is kept. If the
    liveness cache is missing, the input list is returned with a warning.
    """
    path = Path(liveness_path)
    if not path.exists():
        if log_fn is not None:
            log_fn(f"[train_codec] WARNING: spectro liveness cache {path} missing -> NO "
                   f"per-channel presence filter applied for {modality}")
        return [str(s) for s in shots]
    live = torch.load(str(path), map_location="cpu", weights_only=False)
    keep: List[str] = []
    for sh in shots:
        rec = live.get(int(sh)) if str(sh).lstrip("-").isdigit() else None
        flags = (rec or {}).get(modality, {}).get("live")
        if not flags:
            continue
        n = int(sum(flags))
        if (n == len(flags)) if require_all else (n >= 1):
            keep.append(str(sh))
    if log_fn is not None:
        log_fn(f"[train_codec] {modality} spectro presence filter (require_all={require_all}): "
               f"kept {len(keep)}/{len(list(shots))} shots")
    return keep


def video_channels_of(modality: str) -> List[int]:
    """Raw tangtv channel indices this divertor codec uses (lower [0,2], upper [4,6]).

    Read from ``MOVIE_CONFIGS[modality].channels_to_use``, so the presence filter always
    matches what the loader slices.
    """
    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

    mv = next((m for m in TokamakH5Dataset.MOVIE_CONFIGS if m.name == modality), None)
    if mv is None:
        raise ValueError(f"unknown video modality {modality!r}; not in MOVIE_CONFIGS")
    sel = mv.channels_to_use
    if sel is None:
        return list(range(mv.channels))
    if isinstance(sel, slice):
        return list(range(mv.channels))[sel]
    return [int(c) for c in sel]


def video_live_shots(
    modality: str,
    shots: Sequence[Union[str, int]],
    *,
    require_all: bool = False,
    liveness_path: Union[str, Path] = VIDEO_LIVENESS_CACHE,
    log_fn=None,
) -> List[str]:
    """Keep only shots whose cameras for this divertor were recording.

    Over all 8753 shots:

        lower (ch 0, 2)   both 2640 (30.16%)   one 1623 (18.54%)   none 4490 (51.30%)
        upper (ch 4, 6)   both 1820 (20.79%)   one  930 (10.62%)   none 6003 (68.58%)

    A shot without a live camera is an all-NaN slab that the loader fills with zeros,
    and ``VideoCodecPairDataset._draw_valid_clip`` cannot redraw out of it, so it ends
    at the all-zero fallback clip. Unfiltered, half the lower stream and two thirds of
    the upper stream are constant zero.

    ``require_all`` also drops shots with only one live camera (leaving 2640 / 1820
    shots). Order is kept. If the liveness cache is missing, the input list is returned
    with a warning.
    """
    path = Path(liveness_path)
    if not path.exists():
        if log_fn is not None:
            log_fn(f"[train_codec] WARNING: video liveness cache {path} missing -> NO presence "
                   f"filter applied for {modality}")
        return [str(s) for s in shots]
    live = torch.load(str(path), map_location="cpu", weights_only=False)
    chs = video_channels_of(modality)
    keep: List[str] = []
    for sh in shots:
        flags = live.get(int(sh)) if str(sh).lstrip("-").isdigit() else None
        if flags is None:
            continue
        n = sum(int(flags[c]) for c in chs if c < len(flags))
        if (n == len(chs)) if require_all else (n >= 1):
            keep.append(str(sh))
    if log_fn is not None:
        log_fn(f"[train_codec] {modality} video presence filter (channels {chs}, "
               f"require_all={require_all}): kept {len(keep)}/{len(list(shots))} shots")
    return keep


# Spectro δ-shift-pair dataset
def _shot_paths(shots: Sequence[Union[str, int]], data_dir: Union[str, Path]) -> List[Path]:
    """Resolve ``shots`` to existing ``{shot}_processed.h5`` paths under ``data_dir``.

    Missing files are dropped; order is kept, so every rank gets the same file list.
    """
    d = Path(data_dir)
    paths: List[Path] = []
    for s in shots:
        p = d / f"{s}_processed.h5"
        if p.exists():
            paths.append(p)
    return paths


# Activity-stratified redraw, shared by all four codec datasets. Biases the batch toward
# active windows (~50% instead of the natural 35-48%) without dropping quiet ones: a
# below-threshold draw is still accepted as the fallback, so the codec still learns a
# code for quiet windows.
def _stratified_draw(
    idx: int,
    draw_valid,          # callable(chunk_idx) -> item  (redraws degenerate windows)
    activity_of,         # callable(item) -> float      (per-modality activity score)
    n_chunks,            # callable() -> int            (redraw bound, best effort)
    *,
    min_activity: float,
    active_bias: float,
    max_tries: int,
    seed: int,
):
    """Draw one item for window ``idx`` with an activity-stratified redraw.

    With ``active_bias <= 0`` this returns ``draw_valid(idx)`` and uses no extra RNG.

    With ``active_bias > 0``: draw the base item and keep it if it is active
    (``activity_of(item) >= min_activity``). Otherwise, with probability
    ``active_bias``, try up to ``max_tries`` other chunks and return the first active
    one. If none is active (or the coin says no), return the base item, so quiet windows
    are down-weighted, never dropped. The RNG is seeded from ``idx``, so each item is
    deterministic across workers and resumes.
    """
    base = draw_valid(idx)
    if active_bias <= 0.0:
        return base                                   # off: no extra RNG
    if float(activity_of(base)) >= min_activity:
        return base                                   # already active
    gen = torch.Generator().manual_seed(int(seed) + 90_001 * int(idx) + 13)
    if float(torch.rand((), generator=gen)) >= active_bias:
        return base                                   # coin says keep the quiet window
    n = max(1, int(n_chunks()))
    for _ in range(int(max_tries)):
        alt = int(torch.randint(0, n, (1,), generator=gen).item())
        cand = draw_valid(alt)
        if float(activity_of(cand)) >= min_activity:
            return cand                               # found an active window
    return base                                       # no active redraw found


def _chunks_in_current_file(ds, chunk_idx: int) -> int:
    """Redraw bound for hooks whose ``alt`` index is a within-shot chunk index.

    ``_build_window`` / ``_build_clip`` use ``alt`` as an index into the shot on
    ``ds.h5_file`` (``t_start = warmup + alt * step``), so the bound must be this file's
    chunk count, which the parent ``__getitem__`` stores as ``_cur_file_n_chunks``. A
    global bound would put almost every candidate outside the shot and make the redraws
    silent no-ops. (The spectro and fast-TS datasets use ``_chunks_in_current_shot``.)
    Falls back to ``chunk_idx + 1``.
    """
    n = int(getattr(ds, "_cur_file_n_chunks", 0))
    return n if n > 0 else int(chunk_idx) + 1


class CodecPairDataset(TokamakMultiFileDataset):
    """``(spec_a, spec_b)`` δ-shift log-power pairs for one spectro modality.

    A thin subclass of :class:`TokamakMultiFileDataset` that overrides only
    :meth:`_getitem_standard`. For a global ``idx`` the parent's ``__getitem__``
    binary-searches the cumulative lengths for ``file_idx`` and ``chunk_idx``, sets
    ``self.h5_file = self._get_file_handle(file_idx)`` (per-worker LRU handle) and calls
    ``self._getitem_standard(chunk_idx)``. The index map, file handles and worker safety
    are therefore inherited, and the override sees the open handle for the item's shot
    and the within-shot window index.

    Windows start at ``t_start = warmup_s + chunk_idx * step_size_s``, as in the parent.
    The dataset is built with ``chunk_duration_s = CHUNK_S + span_tail_s`` (the full
    δ-extended span), so the parent's length formula
    (``floor((duration - chunk_duration_s) / step_size_s) + 1``) already keeps
    ``[t_start, t_start + CHUNK_S + δ_max]`` inside the shot. ``step_size_s = CHUNK_S``
    keeps windows 50 ms apart.

    Each item loads the raw extended window with
    :meth:`TokamakH5Dataset._load_signal_raw` (resampled to ``target_fs == STFT_FS``,
    zero/NaN padded) and builds the pair with :func:`ignite.data.shift_pair_windows`
    (raw δ shift, then STFT to log-power). Degenerate windows (NaN, near flat, past the
    data end) are rejected as in the spike's ``_load_raw_span`` and redrawn from another
    chunk.

    Parameters
    ----------
    modality : str
        One of :data:`SPECTRO_MODALITIES`.
    shots : sequence of shot ids (str/int)
        Shots to stream windows from, resolved to ``{shot}_processed.h5`` under
        ``data_dir``; missing files are dropped.
    cfg : SpectroCodecConfig
        Provides ``freq_bins`` / ``time_frames`` / ``consistency_delta_ms``. Not
        modified; the caller sets ``cfg.channels`` (see :func:`modality_channels`).
    data_dir : path
        Directory of ``{shot}_processed.h5`` files.
    delta_ms_range : (lo, hi) or None
        δ range in ms; defaults to ``cfg.consistency_delta_ms``.
    t0_start : float
        Earliest window start (s), passed to the parent as ``warmup_s`` (default 1.0,
        skipping the ramp-up).
    min_std : float
        Minimum raw std for a window to count as non-degenerate.
    seed : int
        Base RNG seed, combined with ``idx`` so the δ draw and redraws are deterministic
        per item.
    lengths_cache_path : path or None
        Parent's length-cache sidecar.
    max_open_files : int
        Parent's per-worker file-handle limit.
    max_tries : int
        Redraws before falling back to the eps-floor pair.
    emit_mask : bool
        Return ``(spec_a, spec_b, mask)`` instead of a 2-tuple (see ``__init__``).
    """

    def __init__(
        self,
        modality: str,
        shots: Sequence[Union[str, int]],
        cfg: SpectroCodecConfig,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        *,
        delta_ms_range: Optional[Tuple[float, float]] = None,
        t0_start: float = 1.0,
        min_std: float = 1e-3,
        seed: int = 0,
        lengths_cache_path: Optional[Union[str, Path]] = None,
        max_open_files: int = 512,
        max_tries: int = 8,
        emit_mask: bool = False,
    ) -> None:
        if modality not in SPECTRO_MODALITIES:
            raise ValueError(f"modality {modality!r} not in {SPECTRO_MODALITIES}")

        self.modality = modality
        self.codec_cfg = cfg
        self.delta_ms_range = delta_ms_range or tuple(cfg.consistency_delta_ms)
        self.min_std = float(min_std)
        self.pair_seed = int(seed)
        self.max_tries = int(max_tries)
        # Activity-stratified sampling (0 = off).
        self.min_activity = float(getattr(cfg, "min_activity", 0.0))
        self.active_bias = float(getattr(cfg, "active_bias", 0.0))
        # Missing-data handling (see SpectroCodecConfig.mask_missing). Whether items
        # carry a mask is a constructor argument, not a cfg field: the trainer passes
        # emit_mask=cfg.mask_missing, while every other user (audit and figure scripts,
        # train_dynamics._single_shot_dataset for the frame-code cache,
        # compute_logpow_stats) builds this dataset from a checkpoint's cfg and unpacks
        # a 2-tuple. Keying it on the cfg would change their items as soon as a masked
        # codec is loaded.
        self.emit_mask = bool(emit_mask)
        self.require_live_channels = bool(getattr(cfg, "require_live_channels", False))
        # Reject fully missing windows only when the mask is actually returned.
        self.mask_missing = self.emit_mask

        # Span the pair needs: A covers [t0, t0 + CHUNK_S], B is shifted by up to δ_max.
        # The parent reserves the whole span as chunk_duration_s, while step_size_s
        # stays CHUNK_S so windows advance 50 ms at a time.
        self.span_tail_s = float(self.delta_ms_range[1]) / 1000.0
        span_s = CHUNK_S + self.span_tail_s

        paths = _shot_paths(shots, data_dir)
        if not paths:
            raise ValueError(
                f"CodecPairDataset: no {{shot}}_processed.h5 files found under {data_dir} "
                f"for the {len(list(shots))} requested shots."
            )

        # chunk_duration_s = full δ span so windows fit; step_size_s = CHUNK_S;
        # warmup_s = t0_start.
        super().__init__(
            hdf5_paths=paths,
            chunk_duration_s=span_s,
            step_size_s=CHUNK_S,
            warmup_s=float(t0_start),
            input_signals=[modality],
            target_signals=[modality],
            prediction_mode=False,
            lengths_cache_path=lengths_cache_path,
            max_open_files=max_open_files,
        )
        # Our modality's SignalConfig, for _load_signal_raw and the shot-length lookup.
        self._cfg_sig = next(c for c in self.signal_configs if c.name == modality)

    # Per-item hook: δ-pair
    def _getitem_standard(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Return one ``(spec_a, spec_b)`` δ-shift log-power pair for window ``idx``.

        Called by the parent's ``__getitem__`` after it mapped the global index and set
        ``self.h5_file`` to this shot's handle; ``idx`` is the within-shot
        ``chunk_idx``.

        With ``cfg.active_bias > 0`` the draw (:func:`_stratified_draw`) is biased
        toward log-power windows with std at least ``cfg.min_activity`` (co2 is mostly
        at the floor). With ``active_bias == 0`` (ece/bes/mhr) it is the plain build
        below.
        """
        item = _stratified_draw(
            idx, self._draw_valid_pair, lambda p: float(p[0].std()),
            lambda: self._chunks_in_current_shot(idx),
            min_activity=self.min_activity, active_bias=self.active_bias,
            max_tries=self.max_tries, seed=self.pair_seed,
        )
        # 2-tuple unless the constructor asked for the mask; then the third element is
        # the (C, cfg.time_frames) per-(channel, STFT frame) validity mask of spec_a.
        return item if self.emit_mask else (item[0], item[1])

    def _draw_valid_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the δ-pair for window ``idx``, redrawing degenerate windows."""
        pair = self._build_pair(idx)
        if pair is not None:
            return pair
        # Degenerate window (padded tail, flat, NaN): redraw from other chunks of this
        # shot.
        n_chunks = self._chunks_in_current_shot(idx)
        gen = torch.Generator().manual_seed(self.pair_seed + 100_003 * int(idx) + 7)
        for _ in range(self.max_tries):
            alt = int(torch.randint(0, max(1, n_chunks), (1,), generator=gen).item())
            pair = self._build_pair(alt)
            if pair is not None:
                return pair
        # Last resort: an eps-floor pair (finite) with an all-invalid mask. Not rare: a
        # shot whose HDF5 group is an empty (C, 1) stub (all of a 60-shot 190xxx sample
        # for bes and co2) can never give a real window, so every draw from it ends
        # here.
        C = self._num_channels()
        floor = float(math.log10(1e-10))
        z = torch.full((C, self.codec_cfg.freq_bins, self.codec_cfg.time_frames), floor)
        return z, z.clone(), torch.zeros((C, self.codec_cfg.time_frames))

    # Helpers
    def _build_pair(self, chunk_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Build the δ-pair for a within-shot ``chunk_idx``, or None if degenerate.

        The extended span comes from the parent's ``_load_signal_raw`` (at ``target_fs
        == STFT_FS``); the checks follow :func:`ignite.spike._load_raw_span`.
        """
        step = getattr(self, "step_size_s", self.chunk_duration_s)
        warmup = getattr(self, "warmup_s", 0.0)
        t_start = warmup + chunk_idx * step          # parent's windowing formula
        span_s = CHUNK_S + self.span_tail_s
        t_end = t_start + span_s

        raw, valid_len, nan_mask = self._load_signal_raw(
            self.h5_file, self._cfg_sig, t_start, t_end
        )
        need = round(span_s * self._cfg_sig.target_fs)
        if valid_len < need:
            return None                              # runs into padded / missing tail
        if not torch.isfinite(raw).all():
            return None
        if float(raw.std()) < self.min_std:
            return None                              # all-zero / flat window
        # The std check is over all channels, so a 40-channel ece window with 12 zero
        # channels still passes; that is what the per-channel mask below is for.

        lo, hi = self.delta_ms_range
        gen = torch.Generator().manual_seed(self.pair_seed + 1_000_003 * int(chunk_idx) + 1)
        d = float(lo + (hi - lo) * float(torch.rand((), generator=gen)))
        # raw starts at t_start, so t0 = 0.0 for the pair slicer.
        spec_a, spec_b = data.shift_pair_windows(
            raw, t0=0.0, cfg=self.codec_cfg, delta_ms=d
        )
        # Per-(channel, STFT frame) validity of window A (the reconstruction target),
        # from the raw samples A is computed from: the first cfg.window_samples samples.
        mask = data.spectro_frame_mask(
            raw[:, : self.codec_cfg.window_samples],
            nan_mask[:, : self.codec_cfg.window_samples],
            self.codec_cfg,
        )
        # require_live_channels (off by default): reject, and so redraw, any window with
        # a dead channel.
        if self.require_live_channels and float(mask.min()) <= 0.0:
            return None
        if self.mask_missing and float(mask.sum()) <= 0.0:
            return None                              # nothing real in this window
        return spec_a, spec_b, mask

    def _chunks_in_current_shot(self, chunk_idx: int) -> int:
        """Number of chunks in the shot on ``self.h5_file`` (best effort).

        Only bounds the redraw range. Falls back to ``chunk_idx + 1`` if the shot length
        cannot be read.
        """
        try:
            xd = None
            for key_path in self._cfg_sig.hdf5_keys:
                grp = self.h5_file
                ok = True
                for part in key_path.split("/"):
                    if part in grp:
                        grp = grp[part]
                    else:
                        ok = False
                        break
                if ok and "xdata" in grp:
                    xd = grp["xdata"]
                    break
            if xd is None or xd.shape[0] < 2:
                return chunk_idx + 1
            real_end = float(xd[-1])
            step = getattr(self, "step_size_s", self.chunk_duration_s)
            warmup = getattr(self, "warmup_s", 0.0)
            span_s = CHUNK_S + self.span_tail_s
            n = int(math.floor((real_end - warmup - span_s) / step)) + 1
            return max(1, n)
        except Exception:
            return chunk_idx + 1

    def _num_channels(self) -> int:
        cfg_sig = self._cfg_sig
        if cfg_sig.channels_to_use is not None:
            return len(range(*cfg_sig.channels_to_use.indices(cfg_sig.num_channels)))
        return cfg_sig.num_channels


def _pair_collate(batch):
    """Stack ``(spec_a, spec_b[, mask])`` items into ``(B, C, F, T)`` batches.

    Returns a 2-tuple for unmasked items and ``(a, b, (B, C, T) mask)`` when the dataset
    uses ``cfg.mask_missing``.
    """
    spec_a = torch.stack([it[0] for it in batch], dim=0)
    spec_b = torch.stack([it[1] for it in batch], dim=0)
    if len(batch[0]) < 3:
        return spec_a, spec_b
    return spec_a, spec_b, torch.stack([it[2] for it in batch], dim=0)


def make_codec_loader(
    dataset: CodecPairDataset,
    *,
    batch_size: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Build a δ-pair DataLoader over a :class:`CodecPairDataset`.

    Samplers as in the main foundation-model trainer:

    * DDP (``world_size > 1``): :class:`DistributedTwoLevelSampler`, which shards by
      file and iterates each file in order, keeping the per-worker file-handle cache
      warm.
    * single process: :class:`TwoLevelSampler`, as ``make_dataloader`` uses.

    The collate stacks ``(spec_a, spec_b)`` items into ``(B, C, F, T)`` batches instead
    of the parent's multi-modal ``collate_fn``.
    """
    if world_size > 1:
        sampler: torch.utils.data.Sampler = DistributedTwoLevelSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed,
        )
    else:
        sampler = TwoLevelSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_pair_collate,
        pin_memory=False,           # pairs are small; pinning adds little and copies RAM
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,             # equal per-rank step count under DDP
    )


def _epoch_cycler(loader: DataLoader):
    """Yield batches from ``loader`` forever, starting a new epoch when it is exhausted.

    The loader covers one finite pass over the windows, while training runs a fixed
    number of steps. ``set_epoch`` is called on the sampler each epoch so DDP shuffling
    changes between epochs.
    """
    sampler = getattr(loader, "sampler", None)
    epoch = 0
    while True:
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


# Video (tangtv) codec: dataset, loader and gate, following the spectro code above
class VideoCodecPairDataset(TokamakMultiFileDataset):
    """tangtv frame windows ``(frames (C,T,H,W), frame_mask)`` for one divertor.

    A thin subclass of :class:`TokamakMultiFileDataset`,
    like :class:`CodecPairDataset`; only :meth:`_getitem_standard` is overridden.

    The video codec has no δ-shift pair (docs/IGNITE_DESIGN.md §4.3; video has no
    STFT-phase nuisance), so an item is the frames plus a validity mask, not a pair.
    ``chunk_idx`` is the within-shot window index. Windows use the parent's formula
    (``t_start = warmup_s + chunk_idx * step_size_s``) and
    :meth:`TokamakH5Dataset._load_movie_raw` (resampled to ``target_fps``, zero where
    the camera was off or NaN).

    The mask is ``(T,)`` all ones by default, or ``(C, T)`` per (channel, frame) with
    ``cfg.mask_missing``.

    Parameters
    ----------
    modality : str
        One of :data:`VIDEO_MODALITIES` (``tangtv_lower`` / ``tangtv_upper``).
    shots, cfg, data_dir, seed, lengths_cache_path, max_open_files, max_tries, t0_start
        As for :class:`CodecPairDataset` (``cfg`` is a :class:`VideoCodecConfig`, not
        modified).
    min_std : float
        Clips with a lower std (flat, dead camera) are rejected.
    """

    def __init__(
        self,
        modality: str,
        shots: Sequence[Union[str, int]],
        cfg: VideoCodecConfig,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        *,
        t0_start: float = 1.0,
        min_std: float = 1e-6,
        seed: int = 0,
        lengths_cache_path: Optional[Union[str, Path]] = None,
        max_open_files: int = 512,
        max_tries: int = 8,
    ) -> None:
        if modality not in VIDEO_MODALITIES:
            raise ValueError(f"modality {modality!r} not in {VIDEO_MODALITIES}")

        self.modality = modality
        self.codec_cfg = cfg
        self.min_std = float(min_std)
        self.pair_seed = int(seed)
        self.max_tries = int(max_tries)
        # Activity-stratified sampling (0 = off).
        self.min_activity = float(getattr(cfg, "min_activity", 0.0))
        self.active_bias = float(getattr(cfg, "active_bias", 0.0))
        # Missing-data handling (both off by default).
        self.mask_missing = bool(getattr(cfg, "mask_missing", False))
        self.require_live_channels = bool(getattr(cfg, "require_live_channels", False))

        paths = _shot_paths(shots, data_dir)
        if not paths:
            raise ValueError(
                f"VideoCodecPairDataset: no {{shot}}_processed.h5 files found under "
                f"{data_dir} for the {len(list(shots))} requested shots."
            )

        # A window is one non-overlapping CHUNK_S (50 ms) clip starting after t0_start.
        # The movie is requested by name, so only this divertor's channels are loaded.
        super().__init__(
            hdf5_paths=paths,
            chunk_duration_s=CHUNK_S,
            step_size_s=CHUNK_S,
            warmup_s=float(t0_start),
            input_signals=[modality],
            target_signals=[modality],
            prediction_mode=False,
            lengths_cache_path=lengths_cache_path,
            max_open_files=max_open_files,
        )
        self._cfg_movie = next(c for c in self.movie_configs if c.name == modality)

    # Per-item hook: frame window
    def _getitem_standard(self, idx: int):  # type: ignore[override]
        """Return one ``(frames (C,T,H,W), frame_mask)`` for window ``idx``.

        Called by the parent's ``__getitem__`` after it mapped the global index and set
        ``self.h5_file``; ``idx`` is the within-shot ``chunk_idx``.

        With ``cfg.active_bias > 0`` the draw is biased toward clips with frame std at
        least ``cfg.min_activity``. Training leaves ``active_bias == 0`` for tangtv (its
        collapse was adversarial, not caused by empty windows).
        """
        return _stratified_draw(
            idx, self._draw_valid_clip, lambda c: float(c[0].std()),
            lambda: _chunks_in_current_file(self, idx),
            min_activity=self.min_activity, active_bias=self.active_bias,
            max_tries=self.max_tries, seed=self.pair_seed,
        )

    def _draw_valid_clip(self, idx: int):
        """Build the clip for window ``idx``, redrawing degenerate windows."""
        clip = self._build_clip(idx)
        if clip is not None:
            return clip
        # Degenerate window (dead camera, flat): redraw from other chunks of this shot
        # (alt is a within-shot index, see _chunks_in_current_file).
        n_chunks = max(1, _chunks_in_current_file(self, idx))
        gen = torch.Generator().manual_seed(self.pair_seed + 100_003 * int(idx) + 7)
        for _ in range(self.max_tries):
            alt = int(torch.randint(0, n_chunks, (1,), generator=gen).item())
            clip = self._build_clip(alt)
            if clip is not None:
                return clip
        # Last resort (whole shot degenerate): an all-zero clip with an all-invalid
        # mask.
        cfg = self.codec_cfg
        z = torch.zeros((cfg.channels, cfg.frames, cfg.height, cfg.width))
        m = (torch.zeros((cfg.channels, cfg.frames), dtype=torch.float32)
             if self.mask_missing else torch.zeros(cfg.frames, dtype=torch.float32))
        return z, m

    # Helpers
    def _build_clip(self, chunk_idx: int):
        """Build ``(frames, frame_mask)`` for a within-shot ``chunk_idx``, or None.

        Frames come from the parent's ``_load_movie_raw`` (resampled to ``target_fps``,
        NaN filled with 0), which also returns a per-channel liveness flag. Returns None
        for non-finite or flat clips and clips without a live camera.
        """
        cfg = self.codec_cfg
        step = getattr(self, "step_size_s", self.chunk_duration_s)
        warmup = getattr(self, "warmup_s", 0.0)
        t_start = warmup + chunk_idx * step          # parent's windowing formula
        t_end = t_start + CHUNK_S

        frames, channel_valid = self._load_movie_raw(
            self.h5_file, self._cfg_movie, t_start, t_end
        )
        # frames: (C, T, H, W) at target_fps. Fit T to cfg.frames in case of rounding.
        frames = self._fit_frames(frames)
        if not torch.isfinite(frames).all():
            return None
        if float(frames.std()) < self.min_std:
            return None                              # dead camera / flat window
        # channel_valid is (C,) bool; the loader zero-fills cameras that were off.
        any_live = bool(channel_valid.any().item())
        if not any_live:
            return None
        # require_live_channels rejects (and so redraws) any clip with a dead camera.
        # Even among shots with a live camera, 19.0% (lower) / 16.9% (upper) of the
        # channel slots are zero (see VideoCodecConfig.mask_missing).
        if self.require_live_channels and not bool(channel_valid.all().item()):
            return None
        if not self.mask_missing:
            # Default: an all-ones per-frame mask, which excludes nothing.
            frame_mask = torch.ones(cfg.frames, dtype=torch.float32)
            return frames, frame_mask
        # Per (channel, frame) validity: the loader flagged the camera live for this
        # window and the frame is finite. channel_valid is shorter than cfg.channels
        # only in tests with a reduced grid, so pad it.
        cv = channel_valid.to(torch.float32).reshape(-1)
        if cv.numel() < cfg.channels:
            cv = torch.cat([cv, torch.zeros(cfg.channels - cv.numel())])
        mask = cv[: cfg.channels, None].expand(cfg.channels, cfg.frames).clone()
        finite = torch.isfinite(frames).all(dim=-1).all(dim=-1)      # (C, T)
        mask = mask * finite.to(torch.float32)
        if float(mask.sum()) <= 0.0:
            return None
        return frames, mask

    def _fit_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Crop/pad the loaded ``(C, T, H, W)`` to ``(cfg.channels, cfg.frames, H, W)``.

        ``_load_movie_raw`` already resamples to the ``MovieConfig`` size (120x360 for
        tangtv, as in ``VideoCodecConfig``) and a 50 ms window is already ``cfg.frames``
        long. This makes ``cfg`` authoritative anyway: time can be off by a frame from
        rounding (cropped, or padded by repeating the last frame), and H/W are resized
        trilinearly only if they differ (lets tests use a smaller grid).
        """
        cfg = self.codec_cfg
        C, T, H, W = frames.shape
        # Spatial fit (normally a no-op).
        if (H, W) != (cfg.height, cfg.width):
            frames = torch.nn.functional.interpolate(
                frames.unsqueeze(0), size=(T, cfg.height, cfg.width),
                mode="trilinear", align_corners=False,
            ).squeeze(0)
            C, T = frames.shape[0], frames.shape[1]
        # Time fit.
        if T > cfg.frames:
            frames = frames[:, : cfg.frames]
        elif T < cfg.frames:
            if T == 0:
                return torch.zeros((cfg.channels, cfg.frames, cfg.height, cfg.width))
            pad = frames[:, -1:].expand(C, cfg.frames - T, cfg.height, cfg.width)
            frames = torch.cat([frames, pad], dim=1)
        return frames


def _video_collate(batch):
    """Stack ``(frames, frame_mask)`` items into ``((B,C,T,H,W), (B,T))`` batched tensors."""
    frames = torch.stack([f for f, _ in batch], dim=0)
    masks = torch.stack([m for _, m in batch], dim=0)
    return frames, masks


def make_video_loader(
    dataset: VideoCodecPairDataset,
    *,
    batch_size: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Build a frame-window DataLoader over a :class:`VideoCodecPairDataset`.

    Same samplers as :func:`make_codec_loader`; only the collate differs
    (``(frames, mask)`` instead of ``(spec_a, spec_b)``).
    """
    if world_size > 1:
        sampler: torch.utils.data.Sampler = DistributedTwoLevelSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed,
        )
    else:
        sampler = TwoLevelSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_video_collate,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )


# Slow-TS (Thomson / CER / MSE) codec: dataset, loader and gate. The lightest family
# (§4.3): masked reconstruction and entropy, no δ-pair, no discriminator.
def slowts_codec_cfg(signal: str, channels: int) -> SlowTSCodecConfig:
    """Build a :class:`SlowTSCodecConfig` for ``signal`` with the loader's channels.

    ``channels`` is the number of profile positions loaded (:func:`modality_channels`).
    Patch sizes come from :func:`slowts_patch_for` (4 radial-zone tokens per window,
    ``patch_c = ceil(channels / 4)``). The trainer, CLI and tests all build the cfg
    here.
    """
    patch_c, patch_t = slowts_patch_for(channels)
    return SlowTSCodecConfig(
        signal=signal, channels=channels, patch_c=patch_c, patch_t=patch_t, n_zones=4
    )


def load_slowts_channel_stats(
    signal: str,
    stats_path: Optional[Union[str, Path]] = None,
) -> Tuple[str, List[float], List[float]]:
    """Preprocessing method and per-channel mean/std for ``signal``.

    The slow-TS version of :func:`fastts_train.load_fastts_channel_stats`. Reads the
    ``preprocessing_stats.pt`` the data loader uses and returns
    ``(method, mean_list, std_list)``:

    * ``method`` is the FM's preprocessing for this signal
      (:data:`config.SLOWTS_PREPROCESS_METHOD`, read from the loader's
      ``SignalConfig.preprocess.method``): ``"log_standardize"`` for the 4 Thomson
      signals, ``"standardize"`` for cer_ti/cer_rot/mse.
    * ``mean`` / ``std`` come from the sub-dict the loader uses for that method
      (``'log'`` for ``log_standardize``, ``'raw'`` for ``standardize``), as in
      ``data_loader._update_preprocessing_stats``. Non-finite values become mean 0 / std
      1. None of the slow-TS signals use ``channels_to_use``, so no slicing is needed.

    Raises if the signal or its stats are missing, so a wrong ``--stats_path`` fails
    instead of silently skipping the scaling. ``stats_path=None`` uses
    ``fastts_train.DEFAULT_STATS_PATH``.
    """
    import numpy as np

    from .fastts_train import DEFAULT_STATS_PATH

    if signal not in SLOWTS_MODALITIES:
        raise ValueError(f"signal {signal!r} not in {SLOWTS_MODALITIES}")
    method = SLOWTS_PREPROCESS_METHOD[signal]
    _LOG_METHODS = {"log_standardize", "log_normalize"}
    sub_key = "log" if method in _LOG_METHODS else "raw"

    path = DEFAULT_STATS_PATH if stats_path is None else stats_path
    stats = torch.load(str(path), map_location="cpu", weights_only=False)
    if signal not in stats:
        raise KeyError(f"{signal!r} not in preprocessing_stats at {path}")
    entry = stats[signal]
    if "raw" in entry or "log" in entry:
        sub = entry.get(sub_key, {})
    else:
        sub = entry  # legacy flat format
    if "mean" not in sub or "std" not in sub:
        raise KeyError(f"{signal!r} {sub_key!r} stats missing mean/std at {path}")

    mean = np.asarray(sub["mean"], dtype=np.float64)
    std = np.asarray(sub["std"], dtype=np.float64)
    mean[~np.isfinite(mean)] = 0.0   # as _update_preprocessing_stats (NaN and inf)
    std[~np.isfinite(std)] = 1.0
    return method, mean.astype(np.float32).tolist(), std.astype(np.float32).tolist()


class SlowTSCodecPairDataset(TokamakMultiFileDataset):
    """Slow-TS windows ``(signal (C,T), mask (C,T))`` for one Thomson/CER/MSE signal.

    A thin subclass of :class:`TokamakMultiFileDataset`, like :class:`CodecPairDataset`
    and :class:`VideoCodecPairDataset`. The parent provides the index map, the
    per-worker LRU file handles, the length cache and worker plumbing; only
    :meth:`_getitem_standard` is overridden.

    The slow-TS codec has no δ-shift pair (§4.3) and no adversarial term, so an item is
    a window ``(C, T)`` plus a validity mask ``(C, T)``, not a pair. ``chunk_idx`` is
    the within-shot window index and ``self.h5_file`` this worker's handle. Windows use
    the parent's formula (``t_start = warmup_s + chunk_idx * step_size_s``) and its
    :meth:`TokamakH5Dataset._load_signal_raw` (resampled to 100 Hz, NaN filled with 0,
    raw NaN mask returned).

    Mask
    ----
    ``_load_signal_raw`` returns ``(raw (C,T), valid_len, nan_mask (C,T))`` with
    ``nan_mask`` 1.0 where the raw value was NaN. The validity mask (1 = real) follows
    ``TokamakH5Dataset._process_signal`` for the signal:

    * ``zero_is_missing`` signals (Thomson): valid = (raw != 0) and (nan_mask == 0).
    * otherwise (CER / MSE): valid = (nan_mask == 0).

    so the masked reconstruction never trains toward the fill value of a missing sample.

    Parameters
    ----------
    signal : str
        One of :data:`SLOWTS_MODALITIES`.
    shots, cfg, data_dir, seed, lengths_cache_path, max_open_files, max_tries, t0_start
        As for :class:`CodecPairDataset` (``cfg`` is a :class:`SlowTSCodecConfig`, not
        modified).
    """

    def __init__(
        self,
        signal: str,
        shots: Sequence[Union[str, int]],
        cfg: SlowTSCodecConfig,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        *,
        t0_start: float = 1.0,
        seed: int = 0,
        lengths_cache_path: Optional[Union[str, Path]] = None,
        max_open_files: int = 512,
        max_tries: int = 8,
    ) -> None:
        if signal not in SLOWTS_MODALITIES:
            raise ValueError(f"signal {signal!r} not in {SLOWTS_MODALITIES}")

        self.modality = signal
        self.codec_cfg = cfg
        self.item_seed = int(seed)
        self.max_tries = int(max_tries)
        # Activity-stratified sampling (0 = off). For slow-TS, activity is the window's
        # present fraction.
        self.min_activity = float(getattr(cfg, "min_activity", 0.0))
        self.active_bias = float(getattr(cfg, "active_bias", 0.0))

        paths = _shot_paths(shots, data_dir)
        if not paths:
            raise ValueError(
                f"SlowTSCodecPairDataset: no {{shot}}_processed.h5 files found under "
                f"{data_dir} for the {len(list(shots))} requested shots."
            )

        # A window is one non-overlapping CHUNK_S (50 ms) clip starting after t0_start.
        # The parent's length formula floor((duration - chunk)/step) + 1 keeps every
        # window inside the shot's data.
        super().__init__(
            hdf5_paths=paths,
            chunk_duration_s=CHUNK_S,
            step_size_s=CHUNK_S,
            warmup_s=float(t0_start),
            input_signals=[signal],
            target_signals=[signal],
            prediction_mode=False,
            lengths_cache_path=lengths_cache_path,
            max_open_files=max_open_files,
        )
        self._cfg_sig = next(c for c in self.signal_configs if c.name == signal)

    # Per-item hook: window and validity mask
    def _getitem_standard(self, idx: int):  # type: ignore[override]
        """Return one ``(signal (C,T), mask (C,T))`` for window ``idx``.

        Called by the parent's ``__getitem__`` after it mapped the global index and set
        ``self.h5_file``; ``idx`` is the within-shot ``chunk_idx``.

        With ``cfg.active_bias > 0`` the draw is biased toward windows whose present
        fraction (``mask.mean()``) is at least ``cfg.min_activity`` (ts_core_density is
        mostly empty). With ``active_bias == 0`` (default) it is the plain build below.
        """
        return _stratified_draw(
            idx, self._draw_valid_window, lambda it: float(it[1].mean()),
            lambda: _chunks_in_current_file(self, idx),
            min_activity=self.min_activity, active_bias=self.active_bias,
            max_tries=self.max_tries, seed=self.item_seed,
        )

    def _draw_valid_window(self, idx: int):
        """Build the window for ``idx``, redrawing degenerate windows."""
        item = self._build_window(idx)
        if item is not None:
            return item
        # Degenerate window (all missing / padded tail): redraw from other chunks of
        # this shot (alt is a within-shot index, see _chunks_in_current_file).
        n_chunks = max(1, _chunks_in_current_file(self, idx))
        gen = torch.Generator().manual_seed(self.item_seed + 100_003 * int(idx) + 7)
        for _ in range(self.max_tries):
            alt = int(torch.randint(0, n_chunks, (1,), generator=gen).item())
            item = self._build_window(alt)
            if item is not None:
                return item
        # Last resort (whole shot missing): an all-zero window with an all-invalid mask,
        # shaped to padded_channels so batches collate.
        cfg = self.codec_cfg
        z = torch.zeros((cfg.padded_channels, cfg.time_steps))
        m = torch.zeros((cfg.padded_channels, cfg.time_steps), dtype=torch.float32)
        return z, m

    # Helpers
    def _build_window(self, chunk_idx: int):
        """Build ``(signal (C,T), mask (C,T))`` for within-shot ``chunk_idx``, or None.

        Uses the parent's windowing and ``_load_signal_raw`` (100 Hz). Windows that run
        into the padded tail, are non-finite, or have no valid sample at all are
        rejected.
        """
        cfg = self.codec_cfg
        step = getattr(self, "step_size_s", self.chunk_duration_s)
        warmup = getattr(self, "warmup_s", 0.0)
        t_start = warmup + chunk_idx * step          # parent's windowing formula
        t_end = t_start + CHUNK_S

        raw, valid_len, nan_mask = self._load_signal_raw(
            self.h5_file, self._cfg_sig, t_start, t_end
        )
        need = round(CHUNK_S * self._cfg_sig.target_fs)
        if valid_len < need:
            return None                              # runs into padded / missing tail
        signal, mask = self._fit_window(raw, nan_mask)
        if not torch.isfinite(signal).all():
            return None
        if float(mask.sum()) <= 0.0:
            return None                              # whole window missing
        return signal, mask

    def _fit_window(self, raw: torch.Tensor, nan_mask: torch.Tensor):
        """Fit ``raw`` to ``(cfg.padded_channels, cfg.time_steps)`` and build the mask.

        ``_load_signal_raw`` gives ``(C, T)`` with T == cfg.time_steps for a 50 ms
        window at 100 Hz; this makes ``cfg`` authoritative for the shape. The real
        ``cfg.channels`` positions are masked and standardised first (so mask and
        per-channel stats stay aligned), then padded to ``cfg.padded_channels = n_zones
        * patch_c`` with the pad rows marked missing. The mask follows the class
        docstring.
        """
        cfg = self.codec_cfg
        C, T = raw.shape
        # Time fit (usually a no-op).
        if T > cfg.time_steps:
            raw = raw[:, : cfg.time_steps]
            nan_mask = nan_mask[:, : cfg.time_steps]
        elif T < cfg.time_steps:
            if T == 0:
                z = torch.zeros((cfg.padded_channels, cfg.time_steps))
                return z, torch.zeros_like(z)
            pad_r = raw[:, -1:].expand(C, cfg.time_steps - T)
            pad_m = nan_mask[:, -1:].expand(C, cfg.time_steps - T)
            raw = torch.cat([raw, pad_r], dim=1)
            nan_mask = torch.cat([nan_mask, pad_m], dim=1)
        # Channel fit to the real profile count (the loader already gives cfg.channels).
        # Zone padding comes after standardisation, so stats and mask stay aligned.
        if C != cfg.channels:
            raw = raw[: cfg.channels]
            nan_mask = nan_mask[: cfg.channels]
            if raw.shape[0] < cfg.channels:
                pad = torch.zeros((cfg.channels - raw.shape[0], cfg.time_steps))
                raw = torch.cat([raw, pad], dim=0)
                nan_mask = torch.cat([nan_mask, torch.ones_like(pad)], dim=0)

        # Validity mask, 1 = real sample, following _process_signal. It is built from
        # the raw window: standardising moves a real 0 off zero and a missing 0 to a
        # non-zero value, so zero_is_missing must be checked on raw values.
        valid = nan_mask < 0.5                       # nan_mask 1.0 means NaN
        if cfg.zero_is_missing:                      # Thomson: 0 means missing
            valid = valid & (raw != 0.0)
        # Non-finite raw values are masked like missing ones (excluded from the loss and
        # zeroed in the input). mse has ~2 of 69 channels that are always inf/nan;
        # rejecting whole windows for them dropped most mse windows and let a few
        # through that broke the reconstruction.
        valid = valid & torch.isfinite(raw)

        # Standardise as the FM does (SlowTSCodecConfig.channel_mean/std and
        # SLOWTS_PREPROCESS_METHOD), after the mask and before zone padding (the stats
        # have cfg.channels entries). Identity when the cfg has no stats.
        signal = self._standardize(raw)
        # Non-finite raw values standardise to inf/nan; set them to 0 (the standardised
        # mean) so ``signal * valid`` below cannot produce inf*0 = NaN. They are already
        # masked invalid.
        signal = torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        valid = valid.to(torch.float32)

        # After standardisation a missing position (raw 0) becomes (pp(0) - mean)/std, a
        # large negative value (~-25 for ts_core_density). The loss already excludes it,
        # but the encoder still sees it, and for a mostly-missing signal like
        # ts_core_density that value dominates the input and collapses the codec. So
        # zero the missing positions in the input (0 is the standardised mean).
        if self._standardize_active():
            signal = signal * valid

        # Pad the profile to cfg.padded_channels = n_zones * patch_c so it splits into
        # exactly n_zones zones (n_zones tokens). Pad rows are masked and zero in the
        # input, like a missing position.
        signal, valid = self._pad_zone_tail(signal, valid)
        return signal, valid

    def _pad_zone_tail(self, signal: torch.Tensor, valid: torch.Tensor):
        """Pad ``(C, T)`` to ``(padded_channels, T)`` with zero, masked-invalid rows.

        ``signal`` / ``valid`` are the real ``cfg.channels`` profile, already
        standardised and masked. The added rows look like missing positions to the
        encoder and the loss. No-op when no padding is needed.
        """
        cfg = self.codec_cfg
        pad_rows = cfg.padded_channels - signal.shape[0]
        if pad_rows <= 0:
            return signal, valid
        T = signal.shape[1]
        sig_pad = signal.new_zeros((pad_rows, T))
        val_pad = valid.new_zeros((pad_rows, T))
        signal = torch.cat([signal, sig_pad], dim=0)
        valid = torch.cat([valid, val_pad], dim=0)
        return signal, valid

    def _standardize_active(self) -> bool:
        """True if :meth:`_standardize` rescales (stats present and a real method).

        The input zeroing of missing positions only applies when standardising: without
        it the raw fill is already 0 (Thomson) and multiplying by the mask would change
        the input of callers without stats.
        """
        cfg = self.codec_cfg
        method = getattr(cfg, "preprocess_method", None)
        return (
            method not in (None, "none")
            and cfg.channel_mean is not None
            and cfg.channel_std is not None
        )

    def _standardize(self, raw: torch.Tensor) -> torch.Tensor:
        """Standardise ``raw`` (C,T) per channel like ``data_loader._apply_preprocessing``.

        Follows the loader's ``"standardize"`` / ``"log_standardize"`` branches (same
        clip/log/(x-mean)/std, same ``std.clamp(min=1e-3)``, log- or raw-space stats),
        so the codec sees the same O(1) input as the FM; raw Thomson density is ~1e19.
        The stats are global per channel, not per window, so the profile level is kept.

        Identity when ``cfg.channel_mean`` / ``cfg.channel_std`` is None or
        ``preprocess_method`` is None or ``"none"``. Stats whose length is not ``C``
        raise.
        """
        cfg = self.codec_cfg
        method = getattr(cfg, "preprocess_method", None)
        if method in (None, "none") or cfg.channel_mean is None or cfg.channel_std is None:
            return raw
        if method not in ("standardize", "log_standardize"):
            raise ValueError(
                f"SlowTSCodecPairDataset._standardize: unsupported preprocess_method "
                f"{method!r} (expected 'standardize' / 'log_standardize' / 'none' / None)"
            )
        C = raw.shape[0]
        mean = torch.as_tensor(cfg.channel_mean, dtype=raw.dtype)
        std = torch.as_tensor(cfg.channel_std, dtype=raw.dtype)
        if mean.numel() != C or std.numel() != C:
            raise ValueError(
                f"SlowTSCodecPairDataset._standardize: channel stats length "
                f"{mean.numel()}/{std.numel()} != C={C} for signal {cfg.signal!r}"
            )
        x = raw
        if method == "log_standardize":
            # As in data_loader: clip(x, min=-0.99) + 1, then log10; standardised with
            # the 'log' sub-dict stats.
            x = torch.log10(x.clamp(min=-0.99) + 1.0)
        mean = mean.reshape(C, 1)
        std = std.reshape(C, 1).clamp(min=1e-3)      # as in data_loader
        return (x - mean) / std


def _slowts_collate(batch):
    """Stack ``(signal, mask)`` items into ``((B,C,T), (B,C,T))`` batched tensors."""
    sig = torch.stack([s for s, _ in batch], dim=0)
    masks = torch.stack([m for _, m in batch], dim=0)
    return sig, masks


def make_slowts_loader(
    dataset: SlowTSCodecPairDataset,
    *,
    batch_size: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Build a slow-TS window DataLoader over a :class:`SlowTSCodecPairDataset`.

    Same samplers as :func:`make_codec_loader` (``DistributedTwoLevelSampler`` under
    DDP, ``TwoLevelSampler`` otherwise); only the collate differs (``(signal, mask)``).
    """
    if world_size > 1:
        sampler: torch.utils.data.Sampler = DistributedTwoLevelSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed,
        )
    else:
        sampler = TwoLevelSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_slowts_collate,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )


# Slow-TS stability nuisance and gate (the slow-TS version of spike.compute_gate)
def _nanmean(vals: Sequence[float]) -> float:
    """Mean of ``vals`` ignoring NaNs (NaN if all are NaN)."""
    good = [float(v) for v in vals if float(v) == float(v)]
    return float(sum(good) / len(good)) if good else float("nan")


def slowts_nuisance(x: torch.Tensor, *, jitter: float = 0.02, seed: int = 0) -> torch.Tensor:
    """Nuisance transform of a slow-TS window for the stability gate.

    Slow-TS, like video, has no STFT-phase pair, so stability (codes unchanged under a
    realisation-only change) is tested with a per-window multiplicative amplitude jitter
    of a few percent. The profile shape and its time trend are kept, so a
    statistics-first codec should give ``x`` and ``slowts_nuisance(x)`` nearly the same
    codes. ``x`` is ``(B, C, T)``.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scale = 1.0 + jitter * (2.0 * torch.rand((), generator=gen).item() - 1.0)
    return x * scale


@torch.no_grad()
def slowts_compute_gate(
    codec: SlowTSCodec,
    eval_windows: List[torch.Tensor],
    eval_masks: List[torch.Tensor],
    frame_seq: torch.Tensor,
    cfg: SlowTSCodecConfig,
) -> Dict[str, object]:
    """Slow-TS version of :func:`spike.compute_gate` (§4.4 gate, masked recon).

    Parameters
    ----------
    codec : SlowTSCodec
    eval_windows : list of (B, C, T)
        Held-out windows, for stability (codes of the window vs :func:`slowts_nuisance`)
        and decode fidelity (masked reconstruction vs the window).
    eval_masks : list of (B, C, T)
        Validity masks matching ``eval_windows``, used to mask decode fidelity.
    frame_seq : (B, n_windows, C, T)
        Consecutive 50 ms windows, for persistence (window t vs t+1) and forecastability
        (the whole code sequence).
    cfg : SlowTSCodecConfig

    Returns
    -------
    dict
        The same keys as :func:`spike.compute_gate`, so :func:`spike.gate_score` and the
        trainer use it unchanged.
    """
    was_training = codec.training
    codec.eval()

    stab_vals: List[float] = []
    dec_corr: List[float] = []
    dec_f1: List[float] = []
    dec_sharp: List[float] = []
    full_terms: List[Dict[str, float]] = []
    for wi, x in enumerate(eval_windows):
        out = codec.forward(x)
        nuis = slowts_nuisance(x, seed=wi)
        _, codes_n = codec.quantize(codec.encode(nuis))
        stab_vals.append(gate.stability(out["codes"], codes_n))
        m = eval_masks[wi] if wi < len(eval_masks) else None
        dm = gate.slowts_decode_fidelity(out["recon"], x, mask=m)
        dec_corr.append(dm["envelope_corr"])
        dec_f1.append(dm["peak_f1"])
        dec_sharp.append(dm["sharpness"])
        # Full-window reconstruction metrics and trivial baselines in the same units.
        # envelope_corr drops the time axis, so it reads 1.0 for a prediction with no
        # temporal structure; slowts_nrmse does not. Logged only: spike.gate_score does
        # not use them, so checkpoint selection is unchanged.
        full_terms.append(gate.full_slowts_metrics(out["recon"], x, mask=m))
        full_terms[-1].update(gate.trivial_slowts_baselines(x, mask=m))

    stability_val = float(sum(stab_vals) / len(stab_vals))
    decode = {
        "envelope_corr": float(sum(dec_corr) / len(dec_corr)),
        "peak_f1": float(sum(dec_f1) / len(dec_f1)),
        "sharpness": float(sum(dec_sharp) / len(dec_sharp)),
    }
    full = {k: _nanmean([f[k] for f in full_terms]) for k in full_terms[0]}

    B, n_win = frame_seq.shape[0], frame_seq.shape[1]
    flat = frame_seq.reshape(B * n_win, *frame_seq.shape[2:])   # (B*n_win, C, T)
    _, codes_flat = codec.quantize(codec.encode(flat))
    codes_seq = codes_flat.reshape(B, n_win, cfg.n_tok, cfg.fsq_dim)

    persistence_val = gate.persistence(codes_seq[:, 0], codes_seq[:, 1])
    forecast = gate.forecastability(codes_seq, transition_mask=None)
    util = gate.utilization(codes_seq, cfg=cfg)

    if was_training:
        codec.train()

    return {
        "stability": stability_val,
        "persistence": float(persistence_val),
        "forecastability": forecast,
        "decode": decode,
        "full": full,
        "utilization": util,
        "pass_stability": bool(stability_val >= cfg.gate_stability),
        "pass_persistence": bool(persistence_val >= cfg.gate_persistence),
        "pass_utilization": bool(not util["collapsed"]),
    }


# Slow-TS DDP generator-loss adapter and train steps (no δ-pair, no discriminator)
class _SlowTSGenLossAdapter(torch.nn.Module):
    """Wrapper whose ``forward`` calls ``SlowTSCodec.generator_losses``.

    Same purpose as :class:`_GenLossAdapter`. ``self.codec`` is the underlying
    :class:`SlowTSCodec`. The slow-TS signature has no ``x_shift`` and no ``disc``
    (masked reconstruction and entropy only).
    """

    def __init__(self, codec: SlowTSCodec) -> None:
        super().__init__()
        self.codec = codec

    def forward(
        self,
        x: torch.Tensor,
        cfg: SlowTSCodecConfig,
        step: int,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.codec.generator_losses(x, cfg, step=step, mask=mask)


def slowts_codec_train_step(
    codec: SlowTSCodec,
    opt_g: torch.optim.Optimizer,
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    cfg: SlowTSCodecConfig,
    step: int,
) -> Dict[str, torch.Tensor]:
    """One codec training step on a slow-TS window (single GPU, no DDP).

    No discriminator: a single generator step on the masked reconstruction and entropy
    loss. Used by tests and by single-process runs.
    """
    codec.train()
    opt_g.zero_grad(set_to_none=True)
    g_terms = codec.generator_losses(x, cfg, step=step, mask=mask)
    g_terms["total"].backward()
    # Divergence guard, as for the other codecs.
    if spike.is_step_diverged(g_terms["total"]):
        spike.note_skipped_step()
    else:
        opt_g.step()
    return g_terms


def _ddp_slowts_train_step(
    gen_module: torch.nn.Module,        # DDP-wrapped _SlowTSGenLossAdapter (or bare)
    codec: SlowTSCodec,                 # underlying raw codec (unused; for symmetry)
    opt_g: torch.optim.Optimizer,
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    cfg: SlowTSCodecConfig,
    step: int,
) -> Dict[str, torch.Tensor]:
    """DDP version of :func:`slowts_codec_train_step` (G forward goes through DDP)."""
    opt_g.zero_grad(set_to_none=True)
    g_terms = gen_module(x, cfg, step, mask)
    g_terms["total"].backward()
    # Divergence guard: uniform backward, then skip opt_g.step() on all ranks if any
    # rank's loss is non-finite.
    if spike.is_step_diverged(g_terms["total"]):
        spike.note_skipped_step()
    else:
        opt_g.step()
    return g_terms


# Held-out streamed slow-TS eval set (windows, masks and a consecutive-window sequence)
def _stream_slowts_eval_data(
    signal: str,
    eval_shots: Sequence[Union[str, int]],
    cfg: SlowTSCodecConfig,
    *,
    data_dir: Union[str, Path],
    eval_batches: int,
    eval_batch_size: int,
    eval_frames: int,
    device: torch.device,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """Build a small held-out slow-TS gate set (like :func:`_stream_video_eval_data`).

    Returns ``(eval_windows, eval_masks, frame_seq)`` for :func:`slowts_compute_gate`:
    lists of ``(B, C, T)`` window and mask batches (stability and decode fidelity), and
    a ``(B, n_windows, C, T)`` run of consecutive 50 ms windows (persistence and
    forecastability), with ``eval_frames`` windows. Every rank builds the same set; only
    rank 0 computes the gate.
    """
    ds = SlowTSCodecPairDataset(signal, eval_shots, cfg, data_dir=data_dir, seed=seed)
    n_win = eval_batches * eval_batch_size
    if len(ds) < n_win:
        raise RuntimeError(
            f"_stream_slowts_eval_data: only {len(ds)} eval windows across "
            f"{len(list(eval_shots))} eval shots; need {n_win} "
            f"(eval_batches={eval_batches} * eval_batch_size={eval_batch_size})."
        )
    flat = [ds[i] for i in range(n_win)]
    eval_windows: List[torch.Tensor] = []
    eval_masks: List[torch.Tensor] = []
    for bi in range(eval_batches):
        chunk = flat[bi * eval_batch_size : (bi + 1) * eval_batch_size]
        eval_windows.append(torch.stack([c[0] for c in chunk], dim=0).to(device))
        eval_masks.append(torch.stack([c[1] for c in chunk], dim=0).to(device))

    bsz = max(2, eval_batch_size // 2)
    n_windows = eval_frames
    n_avail = len(ds)
    seqs: List[torch.Tensor] = []
    b = 0
    while len(seqs) < bsz and (b + 1) * n_windows <= n_avail:
        block = [ds[b * n_windows + k][0] for k in range(n_windows)]
        seqs.append(torch.stack(block, dim=0))  # (n_windows, C, T)
        b += 1
    if len(seqs) < bsz:
        raise RuntimeError(
            f"_stream_slowts_eval_data: only {len(seqs)} consecutive {n_windows}-window "
            f"sequences across eval shots; need {bsz}. Add more eval shots."
        )
    frame_seq = torch.stack(seqs, dim=0).to(device)  # (bsz, n_windows, C, T)
    return eval_windows, eval_masks, frame_seq


# Video stability nuisance and gate (the video version of spike.compute_gate)
def video_nuisance(clip: torch.Tensor, *, bright: float = 0.02, offset: float = 0.02,
                   seed: int = 0) -> torch.Tensor:
    """Nuisance transform of a video clip for the stability gate.

    Video has no STFT-phase pair. The divertor camera is fixed, so a spatial shift is
    not a physical nuisance; the realisation noise is camera gain and brightness. The
    transform is a per-clip multiplicative gain plus a small additive offset on the
    standardised frames. Where the light sits and its relative intensity are kept, so a
    statistics-first codec should give ``clip`` and ``video_nuisance(clip)`` nearly the
    same codes.

    ``clip`` is ``(B, C, T, H, W)``; returns the same shape.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Per-clip multiplicative gain.
    gain = 1.0 + bright * (2.0 * torch.rand((), generator=gen).item() - 1.0)
    # Per-clip additive offset on the standardised frames.
    bias = offset * (2.0 * torch.rand((), generator=gen).item() - 1.0)
    return clip * gain + bias


@torch.no_grad()
def video_compute_gate(
    codec: VideoCodec,
    eval_clips: List[torch.Tensor],
    frame_seq: torch.Tensor,
    cfg: VideoCodecConfig,
    eval_masks: Optional[List[torch.Tensor]] = None,
) -> Dict[str, object]:
    """Video version of :func:`spike.compute_gate` (§4.4 gate on frame codes and recon).

    Parameters
    ----------
    codec : VideoCodec
    eval_clips : list of (B, C, T, H, W)
        Held-out clips, for stability (codes of the clip vs :func:`video_nuisance`) and
        decode fidelity (reconstruction vs clip).
    frame_seq : (B, n_windows, C, T, H, W)
        Consecutive 50 ms windows (one world-model frame each), for persistence (window
        t vs t+1) and forecastability (the whole code sequence).
    cfg : VideoCodecConfig
    eval_masks : list of tensor, optional
        Channel masks per clip batch, applied to the full-array metrics.

    Returns
    -------
    dict
        The same keys as :func:`spike.compute_gate` (``stability``, ``persistence``,
        ``forecastability``, ``decode``, ``utilization``, the ``pass_*`` flags), so
        :func:`spike.gate_score` and the trainer use it unchanged, plus ``full`` and
        ``rate``.
    """
    was_training = codec.training
    codec.eval()

    stab_vals: List[float] = []
    dec_corr: List[float] = []
    dec_f1: List[float] = []
    dec_sharp: List[float] = []
    vfull_rows: List[Dict[str, float]] = []
    vlat_rows: List[float] = []
    vgtlat_rows: List[float] = []
    for clip in eval_clips:
        out = codec.forward(clip)
        nuis = video_nuisance(clip, seed=len(stab_vals))
        _, codes_n = codec.quantize(codec.encode(nuis))
        stab_vals.append(gate.stability(out["codes"], codes_n))
        # recon is in the standardised frame space (VideoCodec.standardize_input), so
        # compare it with out["x_std"], not the raw clip.
        dm = gate.video_decode_fidelity(out["recon"], out["x_std"])
        dec_corr.append(dm["envelope_corr"])
        dec_f1.append(dm["peak_f1"])
        dec_sharp.append(dm["sharpness"])
        # Full-array masked metrics (nRMSE and the amplitude ratio nRMSE cannot show)
        # and the patch-lattice (checkerboard) ratio next to its GT value.
        # spike.gate_score reads none of them, so checkpoint selection is unchanged.
        _vm = None if eval_masks is None else eval_masks[len(vfull_rows)]
        vfull_rows.append(gate.full_video_metrics(out["recon"], out["x_std"], mask=_vm))
        vlat_rows.append(gate.video_patch_lattice(
            out["recon"], cfg.patch_h, cfg.patch_w, mask=_vm)["patch_lattice_ratio"])
        vgtlat_rows.append(gate.video_patch_lattice(
            out["x_std"], cfg.patch_h, cfg.patch_w, mask=_vm)["patch_lattice_ratio"])

    stability_val = float(sum(stab_vals) / len(stab_vals))
    decode = {
        "envelope_corr": float(sum(dec_corr) / len(dec_corr)),
        "peak_f1": float(sum(dec_f1) / len(dec_f1)),
        "sharpness": float(sum(dec_sharp) / len(dec_sharp)),
        # Patch lattice (checkerboard) next to the sharpness it can pass for.
        # spike.gate_score reads only envelope_corr / peak_f1 here, so selection is
        # unchanged. They sit in ``decode`` because spike._fmt_gate prints
        # ``lattice=<recon>/gt<target>`` when patch_lattice_ratio is present, which puts
        # the artifact in the training log: an adversarial term once raised video
        # sharpness 0.157 -> 0.378 and the log could not tell texture from checkerboard.
        # Always read it against the GT value, since a blur also lowers the lattice.
        "patch_lattice_ratio": (float(sum(vlat_rows) / len(vlat_rows))
                                if vlat_rows else float("nan")),
        "target_patch_lattice_ratio": (float(sum(vgtlat_rows) / len(vgtlat_rows))
                                       if vgtlat_rows else float("nan")),
    }
    import math as _vmath

    def _vavg(key):
        vals = [r[key] for r in vfull_rows if not _vmath.isnan(r[key])]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    vfull = {k: _vavg(k) for k in vfull_rows[0]} if vfull_rows else {}
    vfull["patch_lattice_ratio"] = (
        float(sum(vlat_rows) / len(vlat_rows)) if vlat_rows else float("nan"))
    vfull["gt_patch_lattice_ratio"] = (
        float(sum(vgtlat_rows) / len(vgtlat_rows)) if vgtlat_rows else float("nan"))

    B, n_win = frame_seq.shape[0], frame_seq.shape[1]
    flat = frame_seq.reshape(B * n_win, *frame_seq.shape[2:])   # (B*n_win, C, T, H, W)
    _, codes_flat = codec.quantize(codec.encode(flat))
    codes_seq = codes_flat.reshape(B, n_win, cfg.n_tok, cfg.fsq_dim)

    persistence_val = gate.persistence(codes_seq[:, 0], codes_seq[:, 1])
    forecast = gate.forecastability(codes_seq, transition_mask=None)
    util = gate.utilization(codes_seq, cfg=cfg)

    if was_training:
        codec.train()

    return {
        "stability": stability_val,
        "persistence": float(persistence_val),
        "forecastability": forecast,
        "decode": decode,
        "utilization": util,
        # Logged only; not read by spike.gate_score.
        "full": vfull,
        "rate": gate.code_rate_bits(codes_seq, cfg.codebook_size, n_tok=cfg.n_tok),
        "pass_stability": bool(stability_val >= cfg.gate_stability),
        "pass_persistence": bool(persistence_val >= cfg.gate_persistence),
        "pass_utilization": bool(not util["collapsed"]),
    }


# Checkpoint grafting
#
# Prefixes of parameters that are an exact identity at initialisation, so grafting them
# onto a checkpoint that predates them leaves step 0 unchanged: the refinement head's
# last conv is zero-init (nets.SpectroDecoder) and decoder_noise's per-channel scale is
# zeros.
_GRAFTABLE_PREFIXES = ("decoder.refine.", "decoder.noise_scale", "noise_scale")

# Keys whose shape depends only on len(cfg.fsq_levels), re-initialised when --fsq_levels
# changes the codebook dimensionality: the FSQ bottleneck's projections (d_model <->
# fsq_dim). Nothing else in the codec depends on fsq_dim, so every other weight is still
# taken from the parent checkpoint. A vocabulary change at the same fsq_dim (e.g.
# [8,5,5,5] -> [4,4,4,4]) keeps all shapes and never reaches this path.
_FSQ_DIM_DEPENDENT_KEYS = (
    "quantizer.fsq.project_in.weight", "quantizer.fsq.project_in.bias",
    "quantizer.fsq.project_out.weight", "quantizer.fsq.project_out.bias",
    "quantizer.pre_quant_levels.weight", "quantizer.pre_quant_levels.bias",
)


def _drop_fsq_dim_mismatches(codec, state, log_fn=None):
    """Return ``state`` without FSQ projection keys whose shape differs from ``codec``.

    Only keys in :data:`_FSQ_DIM_DEPENDENT_KEYS` are dropped, and only on a real shape
    difference, so an ordinary resume drops nothing and a mismatch anywhere else still
    raises in :func:`_load_codec_state_allow_graft`.
    """
    own = codec.state_dict()
    dropped = []
    for k in _FSQ_DIM_DEPENDENT_KEYS:
        if k in state and k in own and tuple(state[k].shape) != tuple(own[k].shape):
            dropped.append((k, tuple(state[k].shape), tuple(own[k].shape)))
    if not dropped:
        return state, []
    state = {k: v for k, v in state.items() if k not in {d[0] for d in dropped}}
    if log_fn is not None:
        log_fn(f"[train_codec] VOCAB RE-SIZE on resume: re-initializing {len(dropped)} FSQ "
               f"projection tensor(s) whose shape depends on len(fsq_levels) "
               f"({[(k, a, b) for k, a, b in dropped]}). Every other weight is grafted from the "
               f"parent checkpoint. This is NOT step-0-identical to the parent.")
    return state, [d[0] for d in dropped]


def _load_codec_state_allow_graft(codec, state, log_fn=None) -> None:
    """``codec.load_state_dict(state)`` that also allows adding identity-at-init modules.

    Loading stays strict otherwise: an architecture flag that silently fails to load has
    killed jobs before. Only keys under :data:`_GRAFTABLE_PREFIXES` (zero-init
    identities) may be missing; anything else raises and names the keys.

    This lets a healthy, high-utilisation codec gain the dilated refine head instead of
    training the head from scratch (which ends at ~103/1000 codes).
    """
    # Changing the number of FSQ levels changes the FSQ projections' shapes; drop those
    # so they are re-initialised instead of raising a size-mismatch error.
    state, _vocab_reinit = _drop_fsq_dim_mismatches(codec, state, log_fn=log_fn)
    missing, unexpected = codec.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(
            f"[train_codec] resume checkpoint has {len(unexpected)} key(s) the model does not: "
            f"{sorted(unexpected)[:8]} - architecture mismatch, refusing to load silently."
        )
    bad = [k for k in missing
           if not k.startswith(_GRAFTABLE_PREFIXES) and k not in _vocab_reinit]
    if bad:
        raise SystemExit(
            f"[train_codec] resume checkpoint is MISSING {len(bad)} non-graftable key(s): "
            f"{sorted(bad)[:8]} - refusing to train a partially-initialized codec."
        )
    if missing and log_fn is not None:
        log_fn(f"[train_codec] GRAFTED {len(missing)} newly-initialized key(s) onto the resumed "
               f"codec ({sorted({k.split('.')[1] if k.startswith('decoder.') else k.split('.')[0] for k in missing})}); "
               f"they are zero-init identities, so step 0 is unchanged.")
    return list(missing)


def _load_opt_state_allow_graft(opt, opt_state, model, grafted, log_fn=None) -> None:
    """Restore Adam moments onto a model that gained parameters since the checkpoint.

    ``Optimizer.load_state_dict`` matches parameters by position, so it raises when the
    counts differ, and grafted parameters sit in the middle of ``model.parameters()``
    (the refine head is registered inside the decoder), so restoring by index would pair
    moments with the wrong tensors. Instead remap by name: saved slot j belongs to the
    j-th non-grafted parameter. Grafted parameters start with zero moments, which is
    right for zero-init identities. With nothing grafted the state is loaded as is.
    """
    names = [n for n, _ in model.named_parameters()]
    saved_groups = opt_state.get("param_groups") or []
    n_saved = sum(len(g.get("params", [])) for g in saved_groups)
    if not grafted and n_saved == len(names):
        opt.load_state_dict(opt_state)
        return
    old_names = [n for n in names if n not in set(grafted)]
    if n_saved != len(old_names) or len(saved_groups) != 1:
        raise SystemExit(
            f"[train_codec] cannot align optimizer state: checkpoint has {n_saved} params in "
            f"{len(saved_groups)} group(s), model has {len(names)} ({len(old_names)} shared). "
            f"Refusing to restore misaligned Adam moments."
        )
    idx_of = {n: i for i, n in enumerate(names)}
    remap = {j: idx_of[nm] for j, nm in enumerate(old_names)}
    src = opt_state.get("state", {})
    new_state = {remap[int(j)]: v for j, v in src.items() if int(j) in remap}
    group = dict(saved_groups[0]); group["params"] = list(range(len(names)))
    opt.load_state_dict({"state": new_state, "param_groups": [group]})
    if log_fn is not None:
        log_fn(f"[train_codec] optimizer state remapped by NAME: {len(new_state)} of "
               f"{len(names)} params kept their Adam moments, {len(grafted)} grafted param(s) "
               f"start fresh.")


def _repin_lr_after_resume(opts, lrs, resumed: bool, log_fn=None, tag: str = "") -> None:
    """Set the requested --lr again after ``opt.load_state_dict`` restored the old one.

    ``Optimizer.load_state_dict`` restores ``param_groups`` including ``lr``, so a
    resumed run asking for another learning rate would silently keep the checkpoint's
    rate (``train_dynamics.py`` has the same guard). No-op when the rate is unchanged.
    """
    if not resumed:
        return
    for opt, lr in zip(opts, lrs):
        if opt is None or lr is None:
            continue
        prev = float(opt.param_groups[0]["lr"])
        for g in opt.param_groups:
            g["lr"] = lr
            if "initial_lr" in g:
                g["initial_lr"] = lr
        if log_fn is not None and abs(prev - lr) > 1e-12:
            log_fn(f"[train_codec] LR OVERRIDE on resume{tag}: checkpoint carried "
                   f"{prev:.3e}, requested {lr:.3e} -> using {lr:.3e}")


def _apply_lr_decay(opts, base_lrs, step: int, cfg) -> float:
    """Exponential LR decay as in the spectral-codec paper (arXiv 2406.05298 section 4).

    ``lr(step) = base_lr * gamma ** (step / every)`` with ``gamma = cfg.lr_decay_gamma``
    and ``every = cfg.lr_decay_every`` (paper: 0.998 per 1,000 steps). It uses the
    absolute global step, so a resumed run gets the same rate as an uninterrupted one
    and it works with :func:`_repin_lr_after_resume`, which resets ``base_lr`` before
    the loop.

    ``gamma == 1.0`` (the default) writes nothing. Returns the generator LR for this
    step.
    """
    gamma = float(getattr(cfg, "lr_decay_gamma", 1.0))
    if gamma == 1.0:
        return float(base_lrs[0]) if base_lrs and base_lrs[0] is not None else float("nan")
    every = max(1, int(getattr(cfg, "lr_decay_every", 1000)))
    scale = gamma ** (step / every)
    cur = float("nan")
    for i, (opt, base) in enumerate(zip(opts, base_lrs)):
        if opt is None or base is None:
            continue
        lr = float(base) * scale
        for g in opt.param_groups:
            g["lr"] = lr
        if i == 0:
            cur = lr
    return cur


class _GenLossAdapter(torch.nn.Module):
    """Wrapper whose ``forward`` calls ``SpectroCodec.generator_losses``.

    DDP only hooks the parameters used in the wrapped module's ``forward``, but the
    generator step is ``codec.generator_losses(...)``, not ``codec.forward``. Wrapping
    this adapter in DDP makes every codec parameter used there part of the gradient
    all-reduce.

    ``self.codec`` is the underlying :class:`SpectroCodec`, used directly (outside the
    DDP graph) for the detached discriminator reconstruction, the gate, EMA and
    ``state_dict``.
    """

    def __init__(self, codec: SpectroCodec) -> None:
        super().__init__()
        self.codec = codec

    def forward(
        self,
        spec_a: torch.Tensor,
        spec_b: torch.Tensor,
        disc: torch.nn.Module,
        cfg: SpectroCodecConfig,
        step: int,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.codec.generator_losses(
            spec_a, spec_b, disc, cfg, step=step, frame_mask=frame_mask
        )


def _ddp_codec_train_step(
    gen_module: torch.nn.Module,        # DDP-wrapped _GenLossAdapter (or bare adapter)
    codec: SpectroCodec,                # the underlying raw codec
    disc: torch.nn.Module,              # DDP-wrapped discriminator (or bare)
    disc_raw: FreqAwarePatchGAN,        # the underlying raw discriminator
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    spec_a: torch.Tensor,
    spec_b: torch.Tensor,
    cfg: SpectroCodecConfig,
    step: int,
    frame_mask: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """DDP version of :func:`spike.codec_train_step`.

    Same math, but the generator forward goes through the DDP-wrapped adapter (so codec
    gradients are all-reduced) and the discriminator forward through its DDP wrapper.
    Inside ``generator_losses`` the discriminator is ``disc_raw`` (no D gradient during
    the G step, as in the spike). The detached reconstruction for the D step runs on the
    raw codec under ``no_grad``.
    """
    codec.train()
    disc_raw.train()

    # Generator (codec) step through the DDP-wrapped adapter
    opt_g.zero_grad(set_to_none=True)
    g_terms = gen_module(spec_a, spec_b, disc_raw, cfg, step, frame_mask)
    g_terms["total"].backward()
    # Divergence guard: backward and its all-reduce ran on every rank, and opt_g.step()
    # is skipped on all ranks if any rank's loss is non-finite.
    if spike.is_step_diverged(g_terms["total"]):
        spike.note_skipped_step()
    else:
        opt_g.step()

    # Discriminator step on a fresh detached reconstruction. The paper updates D once
    # every two steps; cfg.disc_update_every defaults to 1 (every step). The check uses
    # the global step, so it agrees across ranks and across resumes and no rank skips
    # the all-reduce alone.
    every = int(getattr(cfg, "disc_update_every", 1))
    if every > 1 and (step % every) != 0:
        # Still report the discriminator loss (no grad), but take no step. Use disc_raw,
        # not the DDP wrapper: a DDP forward without a matching backward makes the next
        # iteration raise "Expected to have finished reduction in the prior iteration".
        with torch.no_grad():
            recon = codec.forward(spec_a)["recon"]
            d_loss = _spectro_discriminator_loss(disc_raw, spec_a, recon, cfg, frame_mask)
        return g_terms, d_loss

    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        recon = codec.forward(spec_a)["recon"]
    d_loss = _spectro_discriminator_loss(disc, spec_a, recon, cfg, frame_mask)
    d_loss.backward()
    if spike.is_step_diverged(d_loss):
        spike.note_skipped_step()
    else:
        opt_d.step()

    return g_terms, d_loss


def _spectro_discriminator_loss(
    disc: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    cfg: SpectroCodecConfig,
    frame_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """:func:`losses.discriminator_loss` with the missing windows removed on both sides.

    Without a mask, D learns to call a constant log-eps plate (a shot whose HDF5 group
    is an empty stub) or a zero channel "real", which rewards a mean-collapsed
    generator. The selector is the generator's own (``SpectroCodec._valid_windows``), so
    both halves of the GAN agree on what is real. With ``frame_mask`` None, or a mask
    that keeps every window, the tensors are passed through unchanged.
    """
    from .losses import discriminator_loss, time_smooth

    # D's real sample must be the target the generator is trained on
    # (cfg.target_time_smooth). Showing D the raw window against a smoothed target would
    # make the speckle the cue that separates real from fake, which the generator cannot
    # reproduce from 192 codes. time_smooth returns ``real`` itself for K <= 1.
    real = time_smooth(real, int(getattr(cfg, "target_time_smooth", 0) or 0))
    win = SpectroCodec._valid_windows(frame_mask, real.shape)
    if win is not None:
        real, fake = real[win], fake[win]
    return discriminator_loss(disc, real, fake, cfg)


# Video DDP generator-loss adapter and train steps (as for spectro, without x_shift)
class _VideoGenLossAdapter(torch.nn.Module):
    """Wrapper whose ``forward`` calls ``VideoCodec.generator_losses``.

    Same purpose as :class:`_GenLossAdapter`. ``self.codec`` is the underlying
    :class:`VideoCodec`. The video signature has no ``x_shift`` (no δ-pair).
    """

    def __init__(self, codec: VideoCodec) -> None:
        super().__init__()
        self.codec = codec

    def forward(
        self,
        x: torch.Tensor,
        disc: torch.nn.Module,
        cfg: VideoCodecConfig,
        step: int,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.codec.generator_losses(x, disc, cfg, step=step, frame_mask=frame_mask)


def video_codec_train_step(
    codec: VideoCodec,
    disc: FramePatchGAN,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    frames: torch.Tensor,
    frame_mask: Optional[torch.Tensor],
    cfg: VideoCodecConfig,
    step: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """One generator + discriminator step on a video clip (single GPU, no DDP).

    The video version of :func:`spike.codec_train_step`: a generator step on
    ``generator_losses``, then a discriminator step on a detached reconstruction, with
    no δ-shift pair. Used by tests and by single-process runs.
    """
    codec.train()

    opt_g.zero_grad(set_to_none=True)
    g_terms = codec.generator_losses(frames, disc, cfg, step=step, frame_mask=frame_mask)
    g_terms["total"].backward()
    # Divergence guard.
    if spike.is_step_diverged(g_terms["total"]):
        spike.note_skipped_step()
    else:
        opt_g.step()

    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        out = codec.forward(frames)
        recon, real = out["recon"], out["x_std"]     # both in standardised frame space
    if _disc_step_due(cfg, step):
        d_loss = _video_discriminator_loss(disc, real, recon, cfg, frame_mask)
        d_loss.backward()
        if spike.is_step_diverged(d_loss):
            spike.note_skipped_step()
        else:
            opt_d.step()
    else:
        with torch.no_grad():
            d_loss = _video_discriminator_loss(disc, real, recon, cfg, frame_mask)

    return g_terms, d_loss


def _disc_step_due(cfg: VideoCodecConfig, step: int) -> bool:
    """True when the discriminator should be updated on ``step``.

    ``cfg.disc_update_every`` defaults to 1 (every step); the paper uses 2. The d_loss
    value is still computed and logged on skipped steps; only backward and step are
    skipped.
    """
    every = int(getattr(cfg, "disc_update_every", 1) or 1)
    return every <= 1 or (int(step) % every == 0)


def _ddp_video_train_step(
    gen_module: torch.nn.Module,        # DDP-wrapped _VideoGenLossAdapter (or bare)
    codec: VideoCodec,                  # underlying raw codec
    disc: torch.nn.Module,              # DDP-wrapped discriminator (or bare)
    disc_raw: FramePatchGAN,            # underlying raw discriminator
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    frames: torch.Tensor,
    frame_mask: Optional[torch.Tensor],
    cfg: VideoCodecConfig,
    step: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """DDP version of :func:`video_codec_train_step` (G forward goes through DDP)."""
    codec.train()
    disc_raw.train()

    opt_g.zero_grad(set_to_none=True)
    g_terms = gen_module(frames, disc_raw, cfg, step, frame_mask)
    g_terms["total"].backward()
    # Divergence guard: uniform backward, then skip opt_g.step() on all ranks if any
    # rank's loss is non-finite.
    if spike.is_step_diverged(g_terms["total"]):
        spike.note_skipped_step()
    else:
        opt_g.step()

    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        out = codec.forward(frames)
        recon, real = out["recon"], out["x_std"]     # both in standardised frame space
    if _disc_step_due(cfg, step):
        d_loss = _video_discriminator_loss(disc, real, recon, cfg, frame_mask)
        d_loss.backward()
        if spike.is_step_diverged(d_loss):
            spike.note_skipped_step()
        else:
            opt_d.step()
    else:
        # Skipped D step: log the value through the raw module under no_grad. A DDP
        # forward without a matching backward breaks (or hangs) the next iteration.
        with torch.no_grad():
            d_loss = _video_discriminator_loss(disc_raw, real, recon, cfg, frame_mask)

    return g_terms, d_loss


def _video_discriminator_loss(
    disc: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    cfg: VideoCodecConfig,
    frame_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hinge discriminator loss for the frame PatchGAN.

    Same math as :func:`losses.discriminator_loss` (``E[relu(1-D(real))] +
    E[relu(1+D(fake))]`` over the per-frame score maps), kept separate because that one
    is typed to ``SpectroCodecConfig``.

    ``frame_mask`` drops frames whose cameras were off from both the real and fake side,
    with the generator's selector (:meth:`VideoCodec._disc_frame_selector`). Without it
    D learns to call a constant-zero plate real, and 51.3% (lower) / 68.6% (upper) of
    streamed windows are exactly that. ``None`` or an all-valid mask changes nothing.
    """
    from .losses import _as_score_list, _hinge_fake, _hinge_real

    keep = VideoCodec._disc_frame_selector(frame_mask, real.shape)
    if keep is not None:
        real = VideoCodec._select_frames(real, keep)
        fake = VideoCodec._select_frames(fake, keep)
    real_maps = _as_score_list(disc(real))
    fake_maps = _as_score_list(disc(fake.detach() if fake.requires_grad else fake))
    return _hinge_real(real_maps) + _hinge_fake(fake_maps)


# Held-out streamed eval set (disjoint shots) for the gate
def _stream_eval_data(
    modality: str,
    eval_shots: Sequence[Union[str, int]],
    cfg: SpectroCodecConfig,
    *,
    data_dir: Union[str, Path],
    eval_batches: int,
    eval_batch_size: int,
    eval_frames: int,
    device: torch.device,
    seed: int,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Build a small held-out gate set by streaming ``eval_shots`` on this rank.

    Returns ``(eval_pairs, frame_seq)`` as :func:`spike.compute_gate` expects:
    ``eval_pairs`` is a list of ``(spec_a, spec_b)`` batches (plus a mask when
    ``cfg.mask_missing``), ``frame_seq`` a ``(B, n_frames, C, F, T)`` sequence of
    consecutive frames. The set is built once, so gate scores are comparable across
    evals, and every rank builds the same one (only rank 0 computes the gate).
    """
    # δ-pairs from the training dataset class, single process, taken in index order.
    ds = CodecPairDataset(modality, eval_shots, cfg, data_dir=data_dir, seed=seed,
                          emit_mask=bool(getattr(cfg, "mask_missing", False)))
    # The held-out set must not be activity-stratified. min_activity / active_bias are a
    # training-time sampling choice; on the gate set they would bias the metrics toward
    # active windows (only co2 and mirnov use them), and _stratified_draw's redraws make
    # this single-process loop slow enough to use up a whole 2 h leg on mirnov. Zero
    # them on this dataset instance only (not on the cfg, which the codec and checkpoint
    # share), so _stratified_draw returns its first draw.
    ds.min_activity, ds.active_bias = 0.0, 0.0
    n_pairs = eval_batches * eval_batch_size
    n_avail = len(ds)
    if n_avail < n_pairs:
        raise RuntimeError(
            f"_stream_eval_data: only {n_avail} eval windows across {len(list(eval_shots))} "
            f"eval shots; need {n_pairs} (eval_batches={eval_batches} * "
            f"eval_batch_size={eval_batch_size}). Add more eval shots."
        )
    flat_pairs = [ds[i] for i in range(n_pairs)]
    eval_pairs: List[Tuple[torch.Tensor, ...]] = []
    for bi in range(eval_batches):
        chunk = flat_pairs[bi * eval_batch_size : (bi + 1) * eval_batch_size]
        a = torch.stack([p[0] for p in chunk], dim=0).to(device)
        b = torch.stack([p[1] for p in chunk], dim=0).to(device)
        # A mask is present only with cfg.mask_missing; spike.compute_gate passes it to
        # gate.decode_fidelity so the held-out metrics use real data only.
        if len(chunk[0]) > 2:
            eval_pairs.append(
                (a, b, torch.stack([p[2] for p in chunk], dim=0).to(device))
            )
        else:
            eval_pairs.append((a, b))

    # Consecutive-frame sequence for persistence / forecastability.
    frame_seq = _stream_frame_sequence(
        modality, eval_shots, cfg,
        data_dir=data_dir, batch_size=max(2, eval_batch_size // 2),
        n_frames=eval_frames, device=device, seed=seed + 999,
    )
    return eval_pairs, frame_seq


def _stream_frame_sequence(
    modality: str,
    shots: Sequence[Union[str, int]],
    cfg: SpectroCodecConfig,
    *,
    data_dir: Union[str, Path],
    batch_size: int,
    n_frames: int,
    device: torch.device,
    seed: int,
    t0_start: float = 1.0,
    min_std: float = 1e-3,
) -> torch.Tensor:
    """Consecutive-frame sequence for the gate (like spike.real_frame_sequence).

    Scans ``shots`` for runs of ``n_frames`` consecutive valid 50 ms windows and turns
    each into log-power with ``data.log_power_stft``. Single process.
    """
    seq_span_s = n_frames * CHUNK_S
    gen = torch.Generator().manual_seed(seed)
    shot_list = [str(s) for s in shots]

    seqs: List[torch.Tensor] = []
    for shot in shot_list:
        if len(seqs) >= batch_size:
            break
        path = Path(data_dir) / f"{shot}_processed.h5"
        if not path.exists():
            continue
        # Generic per-modality open (spike._open_ece_dataset is ece-only).
        from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

        ds = TokamakH5Dataset(path, input_signals=[modality], target_signals=[modality])
        ds._open_hdf5()
        cfg_sig = next(c for c in ds.signal_configs if c.name == modality)
        try:
            end_time = min(ds.duration, spike._real_end_s(ds, cfg_sig))
            t0 = t0_start
            while t0 + seq_span_s <= end_time and len(seqs) < batch_size:
                frames: List[torch.Tensor] = []
                ok = True
                for fi in range(n_frames):
                    raw = spike._load_raw_span(
                        ds, cfg_sig, t0 + fi * CHUNK_S, CHUNK_S, min_std=min_std
                    )
                    if raw is None:
                        ok = False
                        break
                    frames.append(data.log_power_stft(raw.unsqueeze(0), cfg)[0])
                if ok:
                    seqs.append(torch.stack(frames, dim=0))
                t0 += seq_span_s
        finally:
            spike._close_dataset(ds)

    if len(seqs) < batch_size:
        raise RuntimeError(
            f"_stream_frame_sequence: only {len(seqs)} valid {n_frames}-frame sequences "
            f"across {len(shot_list)} eval shots; need {batch_size}. Add more eval shots."
        )
    return torch.stack(seqs, dim=0).to(device)


# Held-out streamed video eval set (clips and a consecutive-window sequence)
def _stream_video_eval_data(
    modality: str,
    eval_shots: Sequence[Union[str, int]],
    cfg: VideoCodecConfig,
    *,
    data_dir: Union[str, Path],
    eval_batches: int,
    eval_batch_size: int,
    eval_frames: int,
    device: torch.device,
    seed: int,
) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
    """Build a small held-out video gate set (like :func:`_stream_eval_data`).

    Returns ``(eval_clips, frame_seq, eval_masks)`` for :func:`video_compute_gate`:
    ``eval_clips`` is a list of ``(B, C, T, H, W)`` clip batches (stability and decode
    fidelity), ``frame_seq`` a ``(B, n_windows, C, T, H, W)`` run of consecutive 50 ms
    windows (persistence and forecastability), ``eval_masks`` the per-batch channel
    masks. ``eval_frames`` is the number of consecutive windows (one 50 ms clip each).
    Every rank builds the same set; only rank 0 computes the gate.
    """
    ds = VideoCodecPairDataset(modality, eval_shots, cfg, data_dir=data_dir, seed=seed)
    n_clips = eval_batches * eval_batch_size
    if len(ds) < n_clips:
        raise RuntimeError(
            f"_stream_video_eval_data: only {len(ds)} eval windows across "
            f"{len(list(eval_shots))} eval shots; need {n_clips} "
            f"(eval_batches={eval_batches} * eval_batch_size={eval_batch_size})."
        )
    # Keep the masks too, so the gate is not computed over zero-filled dead cameras
    # (51.3% of lower / 68.6% of upper tangtv shots). They are all ones unless
    # cfg.mask_missing is on.
    items = [ds[i] for i in range(n_clips)]
    flat = [it[0] for it in items]
    flat_m = [it[1] for it in items]
    eval_clips: List[torch.Tensor] = []
    eval_masks: List[torch.Tensor] = []
    for bi in range(eval_batches):
        chunk = flat[bi * eval_batch_size : (bi + 1) * eval_batch_size]
        eval_clips.append(torch.stack(chunk, dim=0).to(device))
        eval_masks.append(
            torch.stack(flat_m[bi * eval_batch_size : (bi + 1) * eval_batch_size], dim=0)
            .to(device))

    # Consecutive-window sequence: eval_frames back-to-back 50 ms clips per sample.
    bsz = max(2, eval_batch_size // 2)
    n_windows = eval_frames
    n_avail = len(ds)
    seqs: List[torch.Tensor] = []
    # Take blocks of n_windows consecutive dataset indices (windows are the
    # non-overlapping 50 ms cores, in shot order).
    b = 0
    while len(seqs) < bsz and (b + 1) * n_windows <= n_avail:
        block = [ds[b * n_windows + k][0] for k in range(n_windows)]
        seqs.append(torch.stack(block, dim=0))  # (n_windows, C, T, H, W)
        b += 1
    if len(seqs) < bsz:
        raise RuntimeError(
            f"_stream_video_eval_data: only {len(seqs)} consecutive {n_windows}-window "
            f"sequences across eval shots; need {bsz}. Add more eval shots."
        )
    frame_seq = torch.stack(seqs, dim=0).to(device)  # (bsz, n_windows, C, T, H, W)
    return eval_clips, frame_seq, eval_masks


# DDP init from env vars (same contract as utils.distributed / _srun_rank_wrapper.sh)
class _DDPState:
    """Minimal DDP context from the env vars set by ``_srun_rank_wrapper.sh``.

    Reads ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``, uses NCCL and one GPU per rank
    (when each task sees a single GPU, the device index is 0). Single process when
    ``WORLD_SIZE`` is unset or 1 (the CPU smoke path).
    """

    def __init__(self, backend: str = "nccl", timeout_minutes: Optional[float] = None) -> None:
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        if ws > 1:
            self.rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = ws
            visible = torch.cuda.device_count()
            self.device_index = self.local_rank if visible > 1 else 0
            if visible > 1 and self.device_index >= visible:
                raise RuntimeError(
                    f"rank {self.rank}: LOCAL_RANK={self.local_rank} but only {visible} GPUs "
                    f"visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}); "
                    "bind one GPU per task (srun --gpu-bind, see SRUN_GPU_BIND)"
                )
            self.distributed = True
            if not dist.is_initialized():
                # timeout_minutes: how long a collective may wait for a lagging rank
                # (torch's NCCL default is 10 min); None keeps the default.
                timeout = (
                    None if timeout_minutes is None
                    else datetime.timedelta(minutes=timeout_minutes)
                )
                dist.init_process_group(
                    backend, rank=self.rank, world_size=self.world_size, timeout=timeout
                )
            if torch.cuda.is_available():
                torch.cuda.set_device(self.device_index)
        else:
            self.rank, self.local_rank, self.world_size = 0, 0, 1
            self.device_index = 0
            self.distributed = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", self.device_index)
        return torch.device("cpu")

    def wrap(self, module: torch.nn.Module) -> torch.nn.Module:
        if not self.distributed:
            return module
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(
            module,
            device_ids=[self.device_index] if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def shutdown(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


# Spectro trainer
def train_codec(
    cfg: SpectroCodecConfig,
    modality: str,
    train_shots: Sequence[Union[str, int]],
    eval_shots: Sequence[Union[str, int]],
    *,
    steps: int,
    eval_every: int,
    batch_size: int,
    num_workers: int,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    lr: float = 1e-3,
    disc_lr: Optional[float] = None,
    ema: bool = False,
    ema_decay: float = 0.999,
    eval_batches: int = 8,
    eval_batch_size: int = 4,
    eval_frames: int = 6,
    out_dir: Optional[Union[str, Path]] = None,
    seed: int = 0,
    ddp: Optional[_DDPState] = None,
    resume_state: Optional[Dict[str, object]] = None,
    lengths_cache_path: Optional[Union[str, Path]] = None,
    log_fn=print,
) -> Dict[str, object]:
    """Streaming DDP codec training loop for one spectro modality.

    Streams batches through :func:`_ddp_codec_train_step` (the spike's per-step math),
    runs the gate on a held-out streamed eval set every ``eval_every`` steps and keeps
    the best checkpoint by :func:`spike.gate_score`. Rank 0 logs and writes
    ``codec_last.pt``, ``codec_best.pt``, optionally ``codec_ema.pt``, and
    ``gate_<step>.json``.

    Returns the final gate dict with ``steps`` / ``global_step`` / ``best_score`` /
    ``best_step`` added, like :func:`spike.run_spike`.
    """
    if steps < 1:
        raise ValueError("train_codec: steps must be >= 1")
    ddp = ddp if ddp is not None else _DDPState()
    device = ddp.device
    torch.manual_seed(seed + ddp.rank)

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None and ddp.is_main:
        out_path.mkdir(parents=True, exist_ok=True)

    # Models
    codec = SpectroCodec(cfg).to(device)
    # Discriminator family; the default "patch" is FreqAwarePatchGAN.
    if str(getattr(cfg, "discriminator", "patch")) == "multiscale":
        from .discriminator import MultiScaleSpectroGAN
        disc_raw = MultiScaleSpectroGAN(cfg).to(device)
    else:
        disc_raw = FreqAwarePatchGAN(cfg).to(device)

    start_step = 0
    if resume_state is not None:
        _grafted = _load_codec_state_allow_graft(
            codec, resume_state["codec"], log_fn=(print if ddp.is_main else None)) or []
        if resume_state.get("disc") is not None:
            disc_raw.load_state_dict(resume_state["disc"])
        start_step = int(resume_state.get("step", 0))

    # Adam betas from the cfg (defaults 0.9/0.999, as in torch). The spectral-codec
    # paper (arXiv 2406.05298 section 4) uses LR 2e-4, beta1 0.8, beta2 0.99.
    _betas = (float(getattr(cfg, "adam_beta1", 0.9)), float(getattr(cfg, "adam_beta2", 0.999)))
    opt_g = torch.optim.Adam(codec.parameters(), lr=lr, betas=_betas)
    opt_d = torch.optim.Adam(
        disc_raw.parameters(), lr=disc_lr if disc_lr is not None else lr, betas=_betas
    )
    if resume_state is not None:
        if resume_state.get("opt_g") is not None:
            _load_opt_state_allow_graft(
                opt_g, resume_state["opt_g"], codec, _grafted,
                log_fn=(print if ddp.is_main else None))
        if resume_state.get("opt_d") is not None:
            opt_d.load_state_dict(resume_state["opt_d"])
    _repin_lr_after_resume(
        [opt_g, opt_d], [lr, disc_lr if disc_lr is not None else lr],
        resume_state is not None, log_fn=(print if ddp.is_main else None))

    gen_module = ddp.wrap(_GenLossAdapter(codec))
    disc = ddp.wrap(disc_raw)

    ema_shadow = spike._EMA(codec, ema_decay) if ema else None

    # Held-out gate set (same on every rank; the gate runs on rank 0)
    eval_pairs, frame_seq = _stream_eval_data(
        modality, eval_shots, cfg,
        data_dir=data_dir, eval_batches=eval_batches, eval_batch_size=eval_batch_size,
        eval_frames=eval_frames, device=device, seed=seed + 777,
    )

    # Train loader
    train_ds = CodecPairDataset(
        modality, train_shots, cfg,
        data_dir=data_dir, seed=seed, lengths_cache_path=lengths_cache_path,
        emit_mask=bool(getattr(cfg, "mask_missing", False)),
    )
    loader = make_codec_loader(
        train_ds,
        batch_size=batch_size, num_workers=num_workers,
        rank=ddp.rank, world_size=ddp.world_size, seed=seed, shuffle=True,
    )
    # The loader is a finite map-style epoch; cycle it to cover the fixed step budget.
    stream = _epoch_cycler(loader)

    # Start best-tracking from an existing codec_best.pt, so a resumed leg cannot
    # overwrite the overall best with a worse one of its own.
    best_score = spike.resume_best_score(out_path)
    best_step: Optional[int] = None
    final_gate: Dict[str, object] = {}
    spike.reset_skipped_steps()  # divergence-guard skip counter for this trainer run

    _base_lrs = [lr, disc_lr if disc_lr is not None else lr]
    for local_step in range(steps):
        step = start_step + local_step
        # Exponential LR decay on the absolute step (no-op at the default gamma 1.0).
        _apply_lr_decay([opt_g, opt_d], _base_lrs, step, cfg)
        batch = next(stream)
        # The loader yields (a, b), or (a, b, mask) when cfg.mask_missing is set;
        # frame_mask is None otherwise and every loss term runs unmasked.
        spec_a, spec_b = batch[0], batch[1]
        frame_mask = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
        spec_a = spec_a.to(device, non_blocking=True)
        spec_b = spec_b.to(device, non_blocking=True)

        g_terms, d_loss = _ddp_codec_train_step(
            gen_module, codec, disc, disc_raw, opt_g, opt_d,
            spec_a, spec_b, cfg, step=step, frame_mask=frame_mask,
        )
        if ema_shadow is not None:
            ema_shadow.update(codec)

        is_last = local_step == steps - 1
        if (step % eval_every == 0) or is_last:
            g = spike.compute_gate(codec, eval_pairs, frame_seq, cfg)
            g["g_total"] = float(g_terms["total"].detach())
            g["d_loss"] = float(d_loss.detach())
            g["adaptive_weight"] = float(g_terms["adaptive_weight"])
            g["lr"] = float(opt_g.param_groups[0]["lr"])
            g["step"] = step
            score = spike.gate_score(g, recon_floor=cfg.gate_recon_floor, hard_min_codes=getattr(cfg, "gate_hard_min_codes", 8))
            g["score"] = score
            is_best = score > best_score
            g["is_best"] = bool(is_best)
            if is_best:
                best_score = score
                best_step = step

            g["skipped_steps"] = spike.skipped_steps()
            if ddp.is_main:
                if log_fn is not None:
                    log_fn(
                        spike._fmt_gate(step, g)
                        + f" g_total={g['g_total']:+.4f} d_loss={g['d_loss']:.4f}"
                        + f" adv_lam={g['adaptive_weight']:.4g} score={score:+.4f}"
                        + f" skipped={g['skipped_steps']}"
                    )
                if out_path is not None:
                    spike._write_gate_json(out_path, step, g)
                    spike._save_checkpoint(out_path, step, codec, opt_g, opt_d, disc_raw, cfg)
                    if is_best:
                        spike._save_best_checkpoint(
                            out_path, step, score, codec, disc_raw, cfg, g
                        )
                        if log_fn is not None:
                            log_fn(f"[train_codec] new BEST score={score:+.4f} @ step {step}")
                    if ema_shadow is not None:
                        spike._save_ema_checkpoint(out_path, step, ema_shadow.state_dict(), cfg)
            final_gate = g
        ddp.barrier()

    final_gate["steps"] = steps
    final_gate["global_step"] = start_step + steps
    final_gate["best_score"] = best_score
    final_gate["best_step"] = best_step
    return final_gate


# Video trainer (same DDP, gate and best-checkpoint handling as train_codec)
def train_video_codec(
    cfg: VideoCodecConfig,
    modality: str,
    train_shots: Sequence[Union[str, int]],
    eval_shots: Sequence[Union[str, int]],
    *,
    steps: int,
    eval_every: int,
    batch_size: int,
    num_workers: int,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    lr: float = 1e-3,
    disc_lr: Optional[float] = None,
    ema: bool = False,
    ema_decay: float = 0.999,
    eval_batches: int = 8,
    eval_batch_size: int = 4,
    eval_frames: int = 6,
    out_dir: Optional[Union[str, Path]] = None,
    seed: int = 0,
    ddp: Optional["_DDPState"] = None,
    resume_state: Optional[Dict[str, object]] = None,
    lengths_cache_path: Optional[Union[str, Path]] = None,
    log_fn=print,
) -> Dict[str, object]:
    """Streaming DDP video codec training loop.

    Same structure as :func:`train_codec` (``_DDPState``, ``spike._EMA``,
    ``spike.gate_score`` and the ``spike._save_*`` checkpoint helpers). The differences:
    a :class:`VideoCodec` with :class:`FramePatchGAN`, the ``(frames, frame_mask)``
    loader, :func:`_ddp_video_train_step` (no δ-pair) and :func:`video_compute_gate`.
    Rank 0 logs and writes the same files with the same score contract as spectro.
    """
    if steps < 1:
        raise ValueError("train_video_codec: steps must be >= 1")
    ddp = ddp if ddp is not None else _DDPState()
    device = ddp.device
    torch.manual_seed(seed + ddp.rank)

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None and ddp.is_main:
        out_path.mkdir(parents=True, exist_ok=True)

    codec = VideoCodec(cfg).to(device)
    disc_raw = FramePatchGAN(cfg).to(device)

    start_step = 0
    if resume_state is not None:
        _grafted = _load_codec_state_allow_graft(
            codec, resume_state["codec"], log_fn=(print if ddp.is_main else None)) or []
        if resume_state.get("disc") is not None:
            disc_raw.load_state_dict(resume_state["disc"])
        start_step = int(resume_state.get("step", 0))

    # Adam betas from the cfg (defaults 0.9/0.999, as in torch; the paper uses
    # 0.8/0.99).
    _betas = (float(getattr(cfg, "adam_beta1", 0.9)), float(getattr(cfg, "adam_beta2", 0.999)))
    opt_g = torch.optim.Adam(codec.parameters(), lr=lr, betas=_betas)
    opt_d = torch.optim.Adam(
        disc_raw.parameters(), lr=disc_lr if disc_lr is not None else lr, betas=_betas
    )
    if resume_state is not None:
        if resume_state.get("opt_g") is not None:
            _load_opt_state_allow_graft(
                opt_g, resume_state["opt_g"], codec, _grafted,
                log_fn=(print if ddp.is_main else None))
        if resume_state.get("opt_d") is not None:
            opt_d.load_state_dict(resume_state["opt_d"])
    _repin_lr_after_resume(
        [opt_g, opt_d], [lr, disc_lr if disc_lr is not None else lr],
        resume_state is not None, log_fn=(print if ddp.is_main else None))

    gen_module = ddp.wrap(_VideoGenLossAdapter(codec))
    disc = ddp.wrap(disc_raw)

    ema_shadow = spike._EMA(codec, ema_decay) if ema else None

    eval_clips, frame_seq, eval_masks = _stream_video_eval_data(
        modality, eval_shots, cfg,
        data_dir=data_dir, eval_batches=eval_batches, eval_batch_size=eval_batch_size,
        eval_frames=eval_frames, device=device, seed=seed + 777,
    )

    train_ds = VideoCodecPairDataset(
        modality, train_shots, cfg,
        data_dir=data_dir, seed=seed, lengths_cache_path=lengths_cache_path,
    )
    loader = make_video_loader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        rank=ddp.rank, world_size=ddp.world_size, seed=seed, shuffle=True,
    )
    stream = _epoch_cycler(loader)

    # Start best-tracking from an existing codec_best.pt, so a resumed leg cannot
    # overwrite the overall best with a worse one of its own.
    best_score = spike.resume_best_score(out_path)
    best_step: Optional[int] = None
    final_gate: Dict[str, object] = {}
    spike.reset_skipped_steps()  # divergence-guard skip counter for this trainer run

    _base_lrs = [lr, disc_lr if disc_lr is not None else lr]
    for local_step in range(steps):
        step = start_step + local_step
        # Exponential LR decay on the absolute step (no-op at the default gamma 1.0).
        _apply_lr_decay([opt_g, opt_d], _base_lrs, step, cfg)
        frames, frame_mask = next(stream)
        frames = frames.to(device, non_blocking=True)
        frame_mask = frame_mask.to(device, non_blocking=True)

        g_terms, d_loss = _ddp_video_train_step(
            gen_module, codec, disc, disc_raw, opt_g, opt_d,
            frames, frame_mask, cfg, step=step,
        )
        if ema_shadow is not None:
            ema_shadow.update(codec)

        is_last = local_step == steps - 1
        if (step % eval_every == 0) or is_last:
            g = video_compute_gate(codec, eval_clips, frame_seq, cfg,
                                   eval_masks=eval_masks)
            g["g_total"] = float(g_terms["total"].detach())
            g["d_loss"] = float(d_loss.detach())
            g["adaptive_weight"] = float(g_terms["adaptive_weight"])
            g["step"] = step
            score = spike.gate_score(g, recon_floor=cfg.gate_recon_floor, hard_min_codes=getattr(cfg, "gate_hard_min_codes", 8))
            g["score"] = score
            is_best = score > best_score
            g["is_best"] = bool(is_best)
            if is_best:
                best_score = score
                best_step = step

            g["skipped_steps"] = spike.skipped_steps()
            if ddp.is_main:
                if log_fn is not None:
                    log_fn(
                        spike._fmt_gate(step, g)
                        + f" g_total={g['g_total']:+.4f} d_loss={g['d_loss']:.4f}"
                        + f" adv_lam={g['adaptive_weight']:.4g} score={score:+.4f}"
                        + f" skipped={g['skipped_steps']}"
                    )
                if out_path is not None:
                    spike._write_gate_json(out_path, step, g)
                    spike._save_checkpoint(out_path, step, codec, opt_g, opt_d, disc_raw, cfg)
                    if is_best:
                        spike._save_best_checkpoint(
                            out_path, step, score, codec, disc_raw, cfg, g
                        )
                        if log_fn is not None:
                            log_fn(f"[train_video_codec] new BEST score={score:+.4f} @ step {step}")
                    if ema_shadow is not None:
                        spike._save_ema_checkpoint(out_path, step, ema_shadow.state_dict(), cfg)
            final_gate = g
        ddp.barrier()

    final_gate["steps"] = steps
    final_gate["global_step"] = start_step + steps
    final_gate["best_score"] = best_score
    final_gate["best_step"] = best_step
    return final_gate


# Slow-TS trainer (same DDP, gate and best-checkpoint handling; no discriminator,
# no δ-pair)
def train_slowts_codec(
    cfg: SlowTSCodecConfig,
    signal: str,
    train_shots: Sequence[Union[str, int]],
    eval_shots: Sequence[Union[str, int]],
    *,
    steps: int,
    eval_every: int,
    batch_size: int,
    num_workers: int,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    lr: float = 1e-3,
    disc_lr: Optional[float] = None,   # ignored (no discriminator); uniform signature
    ema: bool = False,
    ema_decay: float = 0.999,
    eval_batches: int = 8,
    eval_batch_size: int = 4,
    eval_frames: int = 6,
    out_dir: Optional[Union[str, Path]] = None,
    seed: int = 0,
    ddp: Optional["_DDPState"] = None,
    resume_state: Optional[Dict[str, object]] = None,
    lengths_cache_path: Optional[Union[str, Path]] = None,
    freeze_encoder: bool = False,
    log_fn=print,
) -> Dict[str, object]:
    """Streaming DDP slow-TS codec training loop.

    Same structure as :func:`train_codec` and :func:`train_video_codec`. The
    differences: a :class:`SlowTSCodec` with no discriminator, the ``(signal, mask)``
    loader, :func:`_ddp_slowts_train_step` (no δ-pair, no GAN) and
    :func:`slowts_compute_gate`. Rank 0 logs and writes the same files with the same
    score contract as the other codecs. The ``_save_*`` helpers expect a discriminator,
    so the empty :class:`_NullDisc` is passed.
    """
    if steps < 1:
        raise ValueError("train_slowts_codec: steps must be >= 1")
    ddp = ddp if ddp is not None else _DDPState()
    device = ddp.device
    torch.manual_seed(seed + ddp.rank)

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None and ddp.is_main:
        out_path.mkdir(parents=True, exist_ok=True)

    codec = SlowTSCodec(cfg).to(device)
    disc_raw = _NullDisc().to(device)  # empty stand-in for the _save_* helpers

    start_step = 0
    if resume_state is not None:
        _grafted = _load_codec_state_allow_graft(
            codec, resume_state["codec"], log_fn=(print if ddp.is_main else None)) or []
        start_step = int(resume_state.get("step", 0))

    # Decoder-only training. On the retrained slow-TS codecs, an out-of-sample
    # 93-parameter linear read-out of the codec's own frozen codes beats its trained
    # 4-layer decoder by 0.16-0.36 pooled nRMSE (cer_rot 0.5788 vs 0.7349, cer_ti 0.5874
    # vs 0.8841, mse 0.6536 vs 0.7840) and recovers 8x more temporal amplitude
    # (slowts_std_ratio_t 0.34-0.38 vs 0.04). The decoder's last layer is already
    # Linear(128 -> 60) and the probe is strictly less expressive, so this is an
    # optimisation failure, most likely because the decoder chases a code distribution
    # that moves during joint training.
    #
    # With freeze_encoder the encoder and quantizer keep the resumed weights (fixed
    # codes, as in the read-out) and only the decoder trains. If it then matches the
    # read-out, joint training was the cause; if not, the objective is. The entropy
    # terms depend only on the frozen encoder, so they become constants without
    # gradient.
    if freeze_encoder:
        for _p in codec.encoder.parameters():
            _p.requires_grad_(False)
        for _p in codec.quantizer.parameters():
            _p.requires_grad_(False)
        _train_p = [q for q in codec.parameters() if q.requires_grad]
        if not _train_p:
            raise ValueError("freeze_encoder froze every parameter; nothing left to train")
        if ddp.is_main:
            _nfr = sum(q.numel() for q in codec.parameters() if not q.requires_grad)
            print(f"[train_codec] FREEZE ENCODER+QUANTIZER: {_nfr/1e6:.2f}M params frozen, "
                  f"{sum(q.numel() for q in _train_p)/1e6:.2f}M decoder params training")
    opt_g = torch.optim.Adam(
        [q for q in codec.parameters() if q.requires_grad] if freeze_encoder
        else codec.parameters(), lr=lr)
    if resume_state is not None and resume_state.get("opt_g") is not None and not freeze_encoder:
        _load_opt_state_allow_graft(
            opt_g, resume_state["opt_g"], codec, _grafted,
            log_fn=(print if ddp.is_main else None))
    _repin_lr_after_resume([opt_g], [lr], resume_state is not None,
                           log_fn=(print if ddp.is_main else None))
    # Optimizer over the null disc's (empty) params, so _save_checkpoint has an opt_d.
    opt_d = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=lr)

    gen_module = ddp.wrap(_SlowTSGenLossAdapter(codec))

    ema_shadow = spike._EMA(codec, ema_decay) if ema else None

    eval_windows, eval_masks, frame_seq = _stream_slowts_eval_data(
        signal, eval_shots, cfg,
        data_dir=data_dir, eval_batches=eval_batches, eval_batch_size=eval_batch_size,
        eval_frames=eval_frames, device=device, seed=seed + 777,
    )

    train_ds = SlowTSCodecPairDataset(
        signal, train_shots, cfg,
        data_dir=data_dir, seed=seed, lengths_cache_path=lengths_cache_path,
    )
    loader = make_slowts_loader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        rank=ddp.rank, world_size=ddp.world_size, seed=seed, shuffle=True,
    )
    stream = _epoch_cycler(loader)

    # Start best-tracking from an existing codec_best.pt, so a resumed leg cannot
    # overwrite the overall best with a worse one of its own.
    best_score = spike.resume_best_score(out_path)
    best_step: Optional[int] = None
    final_gate: Dict[str, object] = {}
    spike.reset_skipped_steps()  # divergence-guard skip counter for this trainer run

    for local_step in range(steps):
        step = start_step + local_step
        x, mask = next(stream)
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        g_terms = _ddp_slowts_train_step(
            gen_module, codec, opt_g, x, mask, cfg, step=step,
        )
        if ema_shadow is not None:
            ema_shadow.update(codec)

        is_last = local_step == steps - 1
        if (step % eval_every == 0) or is_last:
            g = slowts_compute_gate(codec, eval_windows, eval_masks, frame_seq, cfg)
            g["g_total"] = float(g_terms["total"].detach())
            g["d_loss"] = 0.0  # no discriminator
            g["adaptive_weight"] = float(g_terms["adaptive_weight"])
            g["step"] = step
            score = spike.gate_score(g, recon_floor=cfg.gate_recon_floor, hard_min_codes=getattr(cfg, "gate_hard_min_codes", 8))
            g["score"] = score
            is_best = score > best_score
            g["is_best"] = bool(is_best)
            if is_best:
                best_score = score
                best_step = step

            g["skipped_steps"] = spike.skipped_steps()
            if ddp.is_main:
                if log_fn is not None:
                    log_fn(
                        spike._fmt_gate(step, g)
                        + f" g_total={g['g_total']:+.4f} d_loss={g['d_loss']:.4f}"
                        + f" adv_lam={g['adaptive_weight']:.4g} score={score:+.4f}"
                        + f" skipped={g['skipped_steps']}"
                    )
                if out_path is not None:
                    spike._write_gate_json(out_path, step, g)
                    spike._save_checkpoint(out_path, step, codec, opt_g, opt_d, disc_raw, cfg)
                    if is_best:
                        spike._save_best_checkpoint(
                            out_path, step, score, codec, disc_raw, cfg, g
                        )
                        if log_fn is not None:
                            log_fn(f"[train_slowts_codec] new BEST score={score:+.4f} @ step {step}")
                    if ema_shadow is not None:
                        spike._save_ema_checkpoint(out_path, step, ema_shadow.state_dict(), cfg)
            final_gate = g
        ddp.barrier()

    final_gate["steps"] = steps
    final_gate["global_step"] = start_step + steps
    final_gate["best_score"] = best_score
    final_gate["best_step"] = best_step
    return final_gate


class _NullDisc(torch.nn.Module):
    """Zero-parameter discriminator stand-in for the slow-TS codec, which has no GAN.

    ``spike._save_checkpoint`` / ``_save_best_checkpoint`` save a discriminator
    ``state_dict`` so every codec has the same checkpoint schema; this one is empty
    (§4.3). It is never called.
    """

    def state_dict(self, *args, **kwargs):  # noqa: D401 (empty state)
        return {}


# CLI
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tokamak_foundation_model.ignite.train_codec",
        description="IGNITE Phase-A streaming DDP codec trainer (streams thousands of "
                    "shots instead of the spike's fixed ~40-shot pool).",
    )
    p.add_argument("--modality", type=str, default="ece",
                   choices=list(SPECTRO_MODALITIES) + list(VIDEO_MODALITIES)
                           + list(SLOWTS_MODALITIES) + [FASTTS_MODALITY],
                   help="Codec modality: a spectro signal (ece/co2/bes/mhr/mirnov), a "
                        "tangtv divertor video (tangtv_lower/tangtv_upper), a slow-TS "
                        "signal (ts_core_density/ts_core_temp/ts_tangential_density/"
                        "ts_tangential_temp/cer_ti/cer_rot/mse), or the fast-TS codec "
                        "(filterscopes).")
    p.add_argument("--shots", type=str, default=None,
                   help="Explicit comma-separated train shot list (overrides "
                        "--n_shots).")
    p.add_argument("--n_shots", type=int, default=1000,
                   help="Number of shots to auto-discover for the train stream.")
    p.add_argument("--eval_n_shots", type=int, default=16,
                   help="Held-out shots for the gate eval set (disjoint from train).")
    p.add_argument("--steps", type=int, default=20000, help="Training/gate steps.")
    p.add_argument("--eval_every", type=int, default=1000,
                   help="Gate/checkpoint cadence.")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Pairs per batch per rank.")
    p.add_argument("--num_workers", type=int, default=6,
                   help="DataLoader parallel workers per rank (streaming).")
    p.add_argument("--lr", type=float, default=1e-3, help="Codec (generator) Adam LR.")
    p.add_argument("--disc_lr", type=float, default=None,
                   help="Discriminator LR (default lr).")
    p.add_argument("--ema", action="store_true",
                   help="Keep and save an EMA copy of the codec.")
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay in [0,1).")
    p.add_argument("--eval_batches", type=int, default=8,
                   help="Held-out gate δ-pair batches.")
    p.add_argument("--eval_batch_size", type=int, default=4,
                   help="Pairs per eval batch.")
    p.add_argument("--eval_frames", type=int, default=6,
                   help="Consecutive frames per persistence/forecastability sequence.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output dir for gate_<step>.json / codec_{last,best,ema}.pt / "
                        "summary.")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                   help="Directory of {shot}_processed.h5 files.")
    p.add_argument("--stats_path", type=str, default=None,
                   help="preprocessing_stats.pt for fast-TS (filterscopes) and slow-TS "
                        "(Thomson/CER/MSE) input scaling: per-channel mean/std, "
                        "applied the same way as in the FM; slow-TS also takes "
                        "log_standardize or standardize per signal from it. None uses "
                        "fastts_train.DEFAULT_STATS_PATH; '' disables it (raw input, "
                        "which collapses; debugging only). Ignored for spectro and "
                        "video.")
    p.add_argument("--lengths_cache_dir", type=str, default=None,
                   help="Directory for the per-file chunk-length cache "
                        "(codec_<modality>_lengths.pt), which avoids rescanning shot "
                        "lengths at startup. None disables caching.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None,
                   help="Optional codec_last.pt to resume codec, discriminator, "
                        "optimizer and step from.")
    p.add_argument("--slowts_d_model", type=int, default=None,
                   help="Override SlowTSCodecConfig.d_model (slow-TS only; default "
                        "128). Changes every weight shape, so it only works from "
                        "scratch (no --resume).")
    p.add_argument("--slowts_enc_depth", type=int, default=None,
                   help="Override SlowTSCodecConfig.enc_depth (slow-TS only; default "
                        "4).")
    p.add_argument("--slowts_freeze_encoder", action="store_true",
                   help="Slow-TS only: freeze the encoder and quantizer at the resumed "
                        "checkpoint's weights (so the codes are fixed) and train only "
                        "the decoder. A diagnostic: a linear read-out of the frozen "
                        "codes beats the trained decoder by 0.16-0.36 pooled nRMSE, "
                        "and this tests whether joint training is the reason. Requires "
                        "--resume.")
    p.add_argument("--slowts_dec_depth", type=int, default=None,
                   help="Override SlowTSCodecConfig.dec_depth (slow-TS only; default "
                        "4). A linear read-out of the codec's own frozen codes, fitted "
                        "out of sample, beats the trained 4-layer decoder on cer_rot "
                        "(0.5794 vs 0.7380 pooled nRMSE), cer_ti (0.6044 vs 0.8892) "
                        "and mse (0.6581 vs 0.7840), so for those the information is "
                        "in the codes and the decoder fails to extract it.")
    p.add_argument("--recon_weight", type=float, default=None,
                   help="Override cfg.recon_weight, the L1 (MAE) reconstruction "
                        "weight. The slow-TS codec uses 1.0 against entropy_weight "
                        "1.0, so reconstruction and the anti-collapse term are "
                        "weighted equally (the combined loss goes negative, so "
                        "diversity can outweigh reconstruction). The spectral-codec "
                        "paper weights reconstruction 20 against a regulariser of 1.0.")
    p.add_argument("--recon_mse_weight", type=float, default=None,
                   help="Override cfg.recon_mse_weight (slow-TS; default 0 = off): "
                        "weight of a masked squared-error term next to the L1. The "
                        "gate metric (slowts_nrmse = RMSE/std) is a squared error, so "
                        "L1 alone optimises the conditional median of a metric that "
                        "wants the mean. Use with --recon_weight 0 for pure L2.")
    p.add_argument("--entropy_weight", type=float, default=None,
                   help="Override cfg.entropy_weight (anti-collapse strength). It "
                        "scales the whole entropy_loss value, including the two joint "
                        "terms below, so their effective weights are entropy_weight "
                        "times their flag.")
    p.add_argument("--joint_entropy_weight", type=float, default=None,
                   help="Override cfg.joint_entropy_weight (default 0 = off): reward "
                        "on the entropy of the batch-mean joint code distribution over "
                        "all prod(fsq_levels) codes (MagViT-2/LFQ codebook entropy). "
                        "The per-dim diversity reward is marginal, and a rank-1 "
                        "encoder saturates it while using almost none of the codebook "
                        "(prod mhr: min_dim_entropy 0.921 with 42/32768 codes, per-dim "
                        "level positions |r|>=0.997). Maximum log(codebook_size) = "
                        "6.908 nats at 1000 codes. Spectro only; refused above 4096 "
                        "codes (memory).")
    p.add_argument("--joint_entropy_ramp_steps", type=int, default=None,
                   help="Override cfg.joint_entropy_ramp_steps (default 0 = off): ramp "
                        "the joint-entropy weight linearly from 0 to "
                        "--joint_entropy_weight over the first N generator steps. On "
                        "real spectro input the encoder starts saturated (one distinct "
                        "code at step 0), and a large constant weight does not escape "
                        "it: weight 5.0 stayed at one code through step 2000, while "
                        "1.0 and 2.0 reached joint entropies of 5.00 and 5.69 nats. "
                        "Spectro only.")
    p.add_argument("--decorrelation_weight", type=float, default=None,
                   help="Override cfg.decorrelation_weight (default 0 = off): penalty "
                        "on the mean squared off-diagonal correlation of the per-dim "
                        "continuous pre-quant level positions, aimed at the rank-1 FSQ "
                        "collapse (0 = independent dims, ~1 = one scalar copied). "
                        "Spectro only.")
    p.add_argument("--fsq_noise_dropout", type=float, default=None,
                   help="Override cfg.fsq_noise_dropout (default 0 = off): FSQ's own "
                        "`noise_dropout` argument. With this probability per element, "
                        "FSQ.maybe_apply_noise adds a uniform +/-0.5 offset to the "
                        "bounded code and clamps to [-1,1], in training only. It runs "
                        "after codes_to_indices, so the indices stay clean and only "
                        "the decoder input is jittered. The offset is in the "
                        "normalised [-1,1] space where bins are 2/(L-1) apart, so for "
                        "[8,5,5] it is +/-1.75 bins on the 8-level dim and +/-1.0 on "
                        "the 5-level dims. Requires --fsq_preserve_symmetry. Spectro "
                        "only.")
    p.add_argument("--fsq_preserve_symmetry", action="store_true",
                   help="Set cfg.fsq_preserve_symmetry (default off): FSQ's own "
                        "`preserve_symmetry` argument, which uses "
                        "symmetry_preserving_bound (arXiv 2411.19842 s3.2) and the "
                        "matching _scale_and_shift. Required by --fsq_noise_dropout "
                        "(the library asserts it), but it changes the codes by itself "
                        "even at noise_dropout 0, so compare against a "
                        "preserve_symmetry-only run. Spectro only.")
    p.add_argument("--pixel_anchor_weight", type=float, default=None,
                   help="Override cfg.pixel_anchor_weight (L1 reconstruction "
                        "strength). The default 0.05 is an anchor meant to work with "
                        "the adversarial term; raise it (~1-5) when the adversarial "
                        "term is off.")
    p.add_argument("--multiscale_recon_weight", type=float, default=None,
                   help="Override cfg.multiscale_recon_weight: multi-resolution L1 "
                        "(NeMo-style), which keeps turbulent detail that plain pixel "
                        "L1 blurs. ~1-5.")
    p.add_argument("--freq_grad_weight", type=float, default=None,
                   help="Override cfg.freq_grad_weight: L1 on the frequency "
                        "derivative, which penalises a smooth envelope. ~1-5.")
    p.add_argument("--gate_hard_min_codes", type=int, default=None,
                   help="Override cfg.gate_hard_min_codes, a hard floor on distinct "
                        "codes for best-checkpoint selection (below it "
                        "gate_score=-inf). The default 8 was set on spectro (192-768 "
                        "tokens per window); for the low-token families (slow-TS 4, "
                        "fast-TS 5 tokens) set it at launch.")
    p.add_argument("--fm_weight", type=float, default=None,
                   help="Override cfg.fm_weight (discriminator feature matching in the "
                        "generator's reconstruction reference on the adaptive path). 0 "
                        "makes the codec fully GAN-free; at the default 1.0, FM keeps "
                        "pulling the generator toward a saturated discriminator's "
                        "speckle features even with adversarial_weight 0.")
    p.add_argument("--consistency_weight", type=float, default=None,
                   help="Override cfg.consistency_weight (shift-invariance strength).")
    # Fast-TS (filterscopes) raw-sample codec
    p.add_argument("--patch_w", type=int, default=None,
                   help="Fast-TS: raw samples per token. n_tok = window (500) // "
                        "patch_w, so this sets the token count and the frame size. "
                        "patch_w 100/50/20/10/5 gives 5/10/25/50/100 tokens = "
                        "0.0125/0.0249/0.0623/0.1246/0.2491 bits per raw value "
                        "(1000-code FSQ, 8x500 = 4000 values). Default 20 (25 tokens; "
                        "world-model frame 1017 - 5 + 25 = 1037, +2.0%%).")
    p.add_argument("--stem_layers", type=int, default=None,
                   help="Fast-TS: number of stride-1 conv-stem layers before patching "
                        "(default 2; 0 = plain linear patches). Used instead of "
                        "spike-weighted losses: each patch's receptive field overlaps "
                        "its neighbours, so a spike on a patch boundary is seen whole.")
    p.add_argument("--stem_channels", type=int, default=None,
                   help="Fast-TS: conv-stem width (default 64).")
    p.add_argument("--stem_kernel", type=int, default=None,
                   help="Fast-TS: conv-stem kernel, odd (default 15 = +-0.7 ms at 10 "
                        "kHz). Larger means more overlap across patches.")
    p.add_argument("--fastts_target", type=str, default=None, choices=["envelope", "raw"],
                   help="Fast-TS codec target. 'envelope' (default) is the original "
                        "ELM-activity-envelope codec, so existing checkpoints still "
                        "load. 'raw' is the sample-wise 10 kHz waveform codec (implied "
                        "by any of --patch_w / --stem_* / --recon_loss / --ssim_*).")
    p.add_argument("--ssim_weight", type=float, default=None,
                   help="Fast-TS: weight on the 1-D structural term (1 - "
                        "contrast*structure of the SSIM along the sample axis). "
                        "Default 0 = off. It guards against variance collapse: nRMSE "
                        "is minimised by the conditional mean, so it rewards shrinking "
                        "the output (a target shrunk 0.4x scores nRMSE 0.6000 with "
                        "std_ratio 0.4000), and only the SSIM contrast factor sees "
                        "that.")
    p.add_argument("--ssim_win", type=int, default=None,
                   help="Fast-TS: box-window length in samples for the 1-D SSIM "
                        "(default 17 = 1.7 ms at 10 kHz, about one ELM burst). Must "
                        "match gate._FASTTS_SSIM_WIN so loss and metric measure the "
                        "same thing.")
    p.add_argument("--recon_loss", type=str, default=None,
                   choices=["nrmse", "nmse", "mse", "l1", "huber"],
                   help="Fast-TS: sample-wise reconstruction loss. The default 'nrmse' "
                        "is the reported metric (per-(window, channel) RMSE / "
                        "std(target)). 'mse' is not a substitute: the RMS window-mean "
                        "offset is ~25x the ~0.0073 within-window std, so the MSE "
                        "optimum is a flat window (mse 5e-05 but nRMSE 1.0000). 'nmse' "
                        "is the unrooted form (worse conditioned); 'mse'/'l1'/'huber' "
                        "are unnormalised controls.")
    # Fast-TS gain-shape decomposition
    p.add_argument("--gain_shape", action="store_true",
                   help="Fast-TS envelope: code each window as level + sigma * unit "
                        "shape instead of the envelope directly. 96.2%% of the "
                        "ELM-envelope variance is the per-(window, channel) level, and "
                        "an encoder sending only that level at the codec's 49.83-bit "
                        "budget scores pooled nRMSE 0.1935, against 0.5252 for the "
                        "shipped codec. With --gain_shape the first --gain_tokens "
                        "tokens carry the level (and sigma) through an MLP->FSQ path "
                        "and the decoder's shape branch has its mean removed, so the "
                        "per-window mean becomes a floor the codec cannot fall below. "
                        "Default off.")
    p.add_argument("--gain_tokens", type=int, default=None,
                   help="Fast-TS: tokens (of 5) reserved for the gain code; the rest "
                        "carry shape. Token count and vocabulary stay 5 x 1000, so the "
                        "world-model frame is unchanged. --gain_tokens 5 = level only "
                        "(no shape path). Default 1.")
    p.add_argument("--gain_scale", dest="gain_scale", action="store_true", default=None,
                   help="Fast-TS: also send the per-(window, channel) AC scale sigma "
                        "and normalise the decoded shape to unit std, so sigma alone "
                        "sets the within-window amplitude (standard gain-shape VQ). "
                        "Targets the measured 36%% retention of burst height. On by "
                        "default with --gain_shape.")
    p.add_argument("--no_gain_scale", dest="gain_scale", action="store_false",
                   help="Fast-TS: level-only gain; the shape path's amplitude is free "
                        "(and may shrink toward the nRMSE-minimising conditional "
                        "mean).")
    p.add_argument("--gain_weight", type=float, default=None,
                   help="Fast-TS: weight on the direct |pred - (level, log1p sigma)| "
                        "supervision of the gain head (part of the reconstruction "
                        "reference). Default 1.0; 0 trains the gain path through the "
                        "reconstruction term only.")
    # Spectral-codec recipe (arXiv 2406.05298): STFT geometry and training knobs
    p.add_argument("--stft_n_fft", type=int, default=None,
                   help="Spectro: STFT n_fft (default 1024 = config.STFT_N_FFT). 512 "
                        "halves the frequency resolution, giving the 50%%-overlap Hann "
                        "COLA grid (freq_bins 256, 977 Hz per bin); the band stays "
                        "0-250 kHz (Nyquist is set by STFT_FS). Needs --freq_bins = "
                        "n_fft//2 and --patch_f halved to keep n_tok at 192.")
    p.add_argument("--stft_hop", type=int, default=None,
                   help="Spectro: STFT hop (default 256 = config.STFT_HOP).")
    p.add_argument("--freq_bins", type=int, default=None,
                   help="Spectro: codec input frequency bins (default 512). Set to "
                        "n_fft//2 (DC dropped) to cover the full band without "
                        "cropping.")
    p.add_argument("--time_frames", type=int, default=None,
                   help="Spectro: codec input time frames (default 96).")
    p.add_argument("--ms_ssim_weight", type=float, default=None,
                   help="Weight on the 1-MS-SSIM reconstruction term "
                        "(losses.ms_ssim_loss), the differentiable version of the "
                        "gate.ms_ssim metric. 0 = off (default). SSIM's contrast "
                        "factor drops on a blur, which no L-p term penalises.")
    p.add_argument("--ms_ssim_win", type=int, default=None,
                   help="Local window (bins) for the MS-SSIM term (default 7).")
    p.add_argument("--discriminator", type=str, default=None, choices=["patch", "multiscale"],
                   help="Spectro discriminator. 'patch' = FreqAwarePatchGAN (default). "
                        "'multiscale' = MultiScaleSpectroGAN, which scores the whole "
                        "spectrogram at 1x/2x/4x with one score per scale, so the "
                        "decoder cannot satisfy it by tiling one patch texture (the "
                        "measured patch lattice: recon 61.02 vs GT 1.14). The paper "
                        "uses multi-period and multi-scale complex-STFT "
                        "discriminators, i.e. global and multi-resolution, not patch.")
    p.add_argument("--multiscale_recon_scales", type=str, default=None,
                   help="Comma list of avg-pool kernel sizes for the multi-resolution "
                        "reconstruction L1 (default 2,4). Include 1 to score full "
                        "resolution: by default every term is low-passed, so a large "
                        "--multiscale_recon_weight rewards matching the blur. The "
                        "paper varies the STFT window instead.")
    p.add_argument("--conv_dec_base_ch", type=int, default=None,
                   help="Spectro conv decoder: proj_in channel width (default 128). "
                        "The paper uses 1024 initial channels (a 55M decoder against a "
                        "10M encoder).")
    p.add_argument("--conv_dec_res_blocks", type=int, default=None,
                   help="SPECTRO conv decoder: residual conv blocks per upsample stage "
                        "(2).")
    p.add_argument("--conv_dec_min_ch", type=int, default=None,
                   help="SPECTRO conv decoder: channel floor for the halving schedule "
                        "(64).")
    p.add_argument("--disc_update_every", type=int, default=None,
                   help="Run the discriminator step once every N generator steps "
                        "(default 1). Paper: 2.")
    p.add_argument("--adam_beta1", type=float, default=None,
                   help="Adam beta1 for both optimizers (default 0.9). Paper: 0.8.")
    p.add_argument("--adam_beta2", type=float, default=None,
                   help="Adam beta2 for both optimizers (default 0.999). Paper: 0.99.")
    p.add_argument("--lr_decay_gamma", type=float, default=None,
                   help="Exponential LR decay factor applied every --lr_decay_every "
                        "steps: lr = lr0 * gamma**(step/every). 1.0 = off (default). "
                        "Paper: 0.998.")
    p.add_argument("--lr_decay_every", type=int, default=None,
                   help="Step period for --lr_decay_gamma (default 1000, the paper's "
                        "period).")
    p.add_argument("--patch_f", type=int, default=None,
                   help="Override SpectroCodecConfig patch_f (24 tokens = 64/32; "
                        "192 tokens = 16/16).")
    p.add_argument("--patch_t", type=int, default=None,
                   help="Override SpectroCodecConfig patch_t (24 tokens = 64/32; "
                        "192 tokens = 16/16).")
    p.add_argument("--decoder", type=str, default=None, choices=["linear", "conv"],
                   help="Spectro decoder: 'linear' (default; transformer + nn.Linear "
                        "to_pixels, smooth patches) or 'conv' (2D transposed-conv "
                        "upsampler, SpectroConvDecoder).")
    p.add_argument("--refine_depth", type=int, default=None,
                   help="Video or spectro: depth of the decoder's residual conv "
                        "refinement head (stride 1, kernel 3, zero-init last conv, so "
                        "it starts as the identity). It gives the decoder "
                        "full-resolution context across patches, so the texture is no "
                        "longer one basis tiled on the patch lattice (the measured "
                        "checkerboard, gate.patch_lattice_metrics: mhr recon 61.6 vs "
                        "GT 1.15). 0/unset = off.")
    p.add_argument("--band_pool", type=int, default=None,
                   help="Band-power input: mean-pool freq_bins into this many equal "
                        "bands as the last step of log_power_stft (0/unset = raw STFT "
                        "bins). This changes what the codec represents, not its token "
                        "count: n_tok stays "
                        "(eff_freq_bins//patch_f)*(time_frames//patch_t), so choose "
                        "patch_f per pool to keep 192. ece's rank-192 linear floor "
                        "goes from 1.0063 (raw, above the 1.0 anchor) to 0.9274 (16 "
                        "bands) and 0.8826 (8 bands); pooling also shrinks the target, "
                        "so recalibrate every threshold in the pooled space.")
    p.add_argument("--std_weight", type=float, default=None,
                   help="Weight on the amplitude-ratio term |log std(recon) - log "
                        "std(target)| per (window, channel), i.e. the audit's "
                        "std_ratio (ideal 1.0). 0.0 = off. ece's std_ratio stays at "
                        "0.232-0.267 across patch geometries, learning rates and EMA, "
                        "and pixel_anchor_weight matches pixels, not amplitude. nRMSE "
                        "cannot see it (its minimiser is the conditional mean).")
    p.add_argument("--peak_weight", type=float, default=None,
                   help="Spectro: weight on losses.peak_l1_loss, an L1 weighted by "
                        "each frequency bin's prominence over its column mean, so the "
                        "target's spectral peaks carry most of the loss. 0 = off. A "
                        "surrogate for the track metric peak_f1: on 320 held-out co2 "
                        "windows the coherent oracle scores 0.9936, free patch means "
                        "0.6537 and the shipped ms5 codec 0.6116, so the codecs place "
                        "peaks worse than patch means, and adversarial training that "
                        "raised hf from 15%% to 82%% of ceiling moved peak_f1 by only "
                        "+0.008.")
    p.add_argument("--target_time_smooth", type=int, default=None,
                   help="Spectro: reconstruct a K-frame time boxcar of the window "
                        "instead of the window (0/1 = off, the default). The encoder "
                        "still sees the raw window and every metric is still measured "
                        "against raw GT; only the decoder target changes. GT lag-1 "
                        "autocorrelation along the STFT-frame axis is 0.61-0.68 (co2), "
                        "0.42-0.48 (mirnov) and 0.29-0.37 (ece), so a raw target is "
                        "~60-70%% unpredictable speckle and every L-p term is "
                        "minimised by a low-amplitude blur. GT smoothed over 5 frames, "
                        "scored against raw GT, beats every trained codec on both hf "
                        "and nRMSE: co2 0.250 / 0.3842 vs ms5's 0.037 / 0.6812; mirnov "
                        "0.290 / 0.6299 vs ctl_s1's 0.022 / 1.1086.")
    p.add_argument("--codec_d_model", type=int, default=None,
                   help="Codec transformer width (cfg.d_model). Internal to the codec: "
                        "independent of the Phase-B backbone width and does not change "
                        "n_tok, the vocabulary or the frame layout. It changes every "
                        "weight shape, so such a run cannot resume a checkpoint of "
                        "another width. For ece the decoder's to_pixels maps one "
                        "256-wide basis to channels*patch_f*patch_t = 10240 outputs, "
                        "and its patch_lattice_ratio is 127-282 against 1.04 for GT, "
                        "i.e. the reconstruction is dominated by that basis tiled on "
                        "the patch lattice.")
    p.add_argument("--codec_enc_depth", type=int, default=None,
                   help="Codec encoder depth (cfg.enc_depth). Same scope as "
                        "--codec_d_model.")
    p.add_argument("--codec_dec_depth", type=int, default=None,
                   help="Codec decoder depth (cfg.dec_depth). Same scope as "
                        "--codec_d_model.")
    p.add_argument("--codec_heads", type=int, default=None,
                   help="Codec attention heads (cfg.heads). Same scope as "
                        "--codec_d_model.")
    p.add_argument("--refine_hidden", type=int, default=None,
                   help="Channel width of the refinement head's hidden convs (default "
                        "64). Only meaningful with --refine_depth > 1.")
    p.add_argument("--decoder_noise", action="store_true",
                   help="Spectro: per-pixel Gaussian noise (StyleGAN-style) with a "
                        "learned per-channel scale (zero-init, so a fresh decoder is "
                        "unchanged), added at full resolution. Gives the decoder an "
                        "aperiodic source for the high-frequency texture the "
                        "adversarial and FM terms reward, which a deterministic "
                        "decoder can only produce as a tiled basis. Default off.")
    p.add_argument("--refine_dilated", action="store_true",
                   help="Spectro: dilations 1,2,4,8,... in the refinement head, so "
                        "depth D has a receptive field of 2^(D+1)-1 instead of 2D+1 "
                        "(D=4 gives 31 >= patch_f=16, a whole patch plus its "
                        "neighbours).")
    # Production-recipe knobs. The recipe that survived the multi-shot collapse test
    # (prod_ece) is fsq_levels [8,8,8,8,8] (32768 codes), entropy 0.1, adv_clamp 10000,
    # adv_warmup 0, adversarial_weight 1.0; the SpectroCodecConfig defaults are now the
    # collapsing values (1000 codes, entropy 1.0, clamp 50). These flags reproduce it
    # without editing the defaults, and are applied after apply_activity_overrides so
    # the CLI wins.
    p.add_argument("--channel_groups", type=int, default=None,
                   help="Channel-factorised tokens (config.py:79). G>1 splits the "
                        "channels into G groups so each token carries C/G channels: "
                        "patch_dim = (C/G)*patch_f*patch_t, n_tok = G*nf*nt. ece "
                        "(C=40) at G=2: 10240 -> 5120 values per token, 192 -> 384 "
                        "tokens. Must divide C.")
    p.add_argument("--fsq_levels", type=str, default=None,
                   help="Comma-separated FSQ levels, e.g. '8,8,8,8,8' (32768 codes, "
                        "production recipe) or the default '8,5,5,5' (1000 codes). Set "
                        "before the codec is built.")
    p.add_argument("--adaptive_adv_clamp", type=float, default=None,
                   help="Override cfg.adaptive_adv_clamp (production recipe 10000; "
                        "default 50).")
    p.add_argument("--adv_warmup_steps", type=int, default=None,
                   help="Override cfg.adv_warmup_steps (production recipe 0).")
    p.add_argument("--adversarial_weight", type=float, default=None,
                   help="Override cfg.adversarial_weight (production recipe 1.0).")
    p.add_argument("--skip_activity_override", action="store_true",
                   help="Skip the per-modality _activity_overrides (needed to "
                        "reproduce the plain production recipe for co2/mhr, whose "
                        "overrides change warmup, weight and bias).")
    p.add_argument("--build_presence_lengths", action="store_true",
                   help="Write codec_<modality>_presence_lengths.pt for the resolved "
                        "train shots by subsetting codec_<modality>_lengths.pt, then "
                        "exit (no HDF5 scan, no training). Run once per modality "
                        "before the first --spectro_presence run: without the sidecar "
                        "every run cold-scans every shot (~0.77 s per file) and uses "
                        "up its whole leg. Needs --spectro_presence or "
                        "--video_presence so the resolved list is the filtered one.")
    p.add_argument("--compute_logpow_stats", type=str, default=None,
                   help="Compute dataset-level per-freq log-power stats over the "
                        "resolved train shots (after the raw z-score for co2), write "
                        "them to this path and exit without training. Spectro only.")
    p.add_argument("--logpow_stats_path", type=str, default=None,
                   help="Path to codec_<mod>_perfreq_stats.pt; enables per-freq "
                        "log-power z-standardisation (the thin-modality mean-collapse "
                        "fix). Composes with raw-std; the stats must come from "
                        "--compute_logpow_stats.")
    p.add_argument("--instance_norm_quantize", type=float, default=None,
                   help="Shift-robust instance norm: quantisation step for the window "
                        "stats (sd on a log2 grid, mu in units of q*sd). 0.5 "
                        "recommended; unset keeps plain instance norm. Only used with "
                        "--input_instance_norm.")
    # Video and spectro missing-data exclusion (all off by default)
    p.add_argument("--mask_missing", action="store_true",
                   help="Video/spectro: build a per-channel validity mask and apply it "
                        "in every loss term. Video: per (channel, frame) from the "
                        "loader's channel_valid. Spectro: per (channel, STFT frame) "
                        "from data.spectro_frame_mask (NaN projected as in "
                        "data_loader._raw_to_frame_mask, plus all-zero supports, which "
                        "is how the loader stores an absent channel). Masked: pixel "
                        "anchor, adversarial and feature matching (missing windows are "
                        "dropped from the discriminator instead of shown as real), "
                        "multiscale / freq-grad / MS-SSIM, the FSQ entropy statistic, "
                        "shift consistency and the discriminator step. Default off (no "
                        "mask anywhere).")
    p.add_argument("--require_live_channels", action="store_true",
                   help="Video/spectro: redraw any window with a dead channel, so the "
                        "codec only sees fully populated windows (video: 19.0%% of "
                        "lower and 16.9%% of upper channel slots are dead even in "
                        "shots that have some video).")
    p.add_argument("--spectro_presence", type=str, default=None, choices=["off", "any", "all"],
                   help="Spectro whole-shot presence filter from the per-channel "
                        "liveness cache (no HDF5 scan): 'any' keeps shots with at "
                        "least one live channel, 'all' only fully populated shots, "
                        "'off' (default) every shot, including those whose HDF5 group "
                        "is an empty (C, 1) stub, which only yield the eps-floor "
                        "fallback pair.")
    p.add_argument("--video_presence", type=str, default=None, choices=["off", "any", "all"],
                   help="Video whole-shot presence filter from the liveness cache (no "
                        "HDF5 scan): 'any' keeps shots with at least one live camera "
                        "for this divertor, 'all' only fully populated shots, 'off' "
                        "(default) every shot, in which case 51.3%% (lower) / 68.6%% "
                        "(upper) of the stream is all-zero clips.")
    p.add_argument("--input_instance_norm", action="store_true",
                   help="Per-window instance z-score (mean ~0, std ~1) on the "
                        "log-power encoder input. Removes the large DC offset that "
                        "saturated co2's FSQ tanh bound. Composes with raw-std, which "
                        "runs first.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = build_arg_parser().parse_args(argv)

    ddp = _DDPState()
    device = ddp.device

    is_video = args.modality in VIDEO_MODALITIES
    is_slowts = args.modality in SLOWTS_MODALITIES
    is_fastts = args.modality == FASTTS_MODALITY
    is_spectro = args.modality in SPECTRO_MODALITIES

    # Config with the modality's real channel count (identical on every rank).
    channels = modality_channels(args.modality)
    if is_video:
        cfg = VideoCodecConfig(channels=channels, divertor=video_divertor(args.modality))
    elif is_slowts:
        cfg = slowts_codec_cfg(args.modality, channels)
    elif is_fastts:
        # Fast-TS (filterscopes) is trained by fastts_train, imported lazily because it
        # imports from this module. The default is the envelope codec, so old
        # checkpoints still load. --fastts_target raw selects the sample-wise codec and
        # its objective defaults (config.fastts_raw_config); any raw-only geometry flag
        # implies it.
        from .config import FastTSCodecConfig, fastts_raw_config
        _raw_implied = any(getattr(args, k, None) is not None
                           for k in ("patch_w", "stem_layers", "stem_channels", "stem_kernel",
                                     "recon_loss", "ssim_weight", "ssim_win"))
        if getattr(args, "fastts_target", None) == "raw" or _raw_implied:
            cfg = fastts_raw_config(channels=channels)
        else:
            cfg = FastTSCodecConfig(channels=channels)
    else:
        cfg = SpectroCodecConfig(channels=channels)
        # STFT geometry first, since it sets freq_bins and every later shape. Without
        # these flags the defaults (1024 / 256) are used.
        for _g in ("stft_n_fft", "stft_hop", "freq_bins", "time_frames"):
            _v = getattr(args, _g, None)
            if _v is not None:
                setattr(cfg, _g, int(_v))
        # Patch size (spectro only): 24 tokens = patch 64x32, 192 tokens = 16x16 (the
        # default). Set before standardization, overrides and the codec so n_tok is
        # fixed.
        if args.patch_f is not None:
            cfg.patch_f = args.patch_f
        if args.patch_t is not None:
            cfg.patch_t = args.patch_t
        # freq_bins should equal n_fft // 2 (DC dropped); otherwise
        # data._crop_pad_freq_time crops the top of the 0-250 kHz band or eps-pads a
        # band that does not exist. n_tok is logged because the Phase-B frame budgets
        # 192 tokens per spectro modality: halving n_fft (512 -> 256 bins) together with
        # patch_f (16 -> 8) keeps 32 x 6 = 192, so the frame stays at 1017 tokens.
        # __post_init__ only checked the defaults, so recheck divisibility here; a bad
        # combination otherwise fails as an opaque einops error in the encoder.
        if cfg.freq_bins % cfg.patch_f or cfg.time_frames % cfg.patch_t:
            raise SystemExit(
                f"geometry: freq_bins={cfg.freq_bins} must divide by patch_f={cfg.patch_f} and "
                f"time_frames={cfg.time_frames} by patch_t={cfg.patch_t}"
            )
        _bins_avail = int(cfg.stft_n_fft) // 2
        if cfg.freq_bins != _bins_avail:
            print(f"[train_codec] WARNING: freq_bins={cfg.freq_bins} != n_fft//2={_bins_avail} "
                  f"for stft_n_fft={cfg.stft_n_fft} -> the input is "
                  f"{'CROPPED' if cfg.freq_bins < _bins_avail else 'eps-PADDED'} in frequency; "
                  f"the covered band is NOT the full 0-{STFT_FS / 2e3:.0f} kHz.", flush=True)
        if ddp.is_main:
            _khz = (STFT_FS / cfg.stft_n_fft) / 1e3
            print(f"[train_codec] STFT grid: n_fft={cfg.stft_n_fft} hop={cfg.stft_hop} -> "
                  f"freq_bins={cfg.freq_bins} ({_khz:.3f} kHz/bin, "
                  f"0-{cfg.freq_bins * _khz:.1f} kHz) x time_frames={cfg.time_frames}; "
                  f"patch {cfg.patch_f}x{cfg.patch_t} -> n_tok={cfg.n_tok} "
                  f"({cfg.channels * cfg.patch_f * cfg.patch_t} values/token, "
                  f"{cfg.n_tok * math.log2(cfg.codebook_size) / (cfg.channels * cfg.freq_bins * cfg.time_frames):.4f} "
                  f"bits/value)", flush=True)
        # Decoder family (spectro only): 'conv' replaces the linear to_pixels unpatchify
        # with the transposed-conv upsampler (SpectroConvDecoder). Default 'linear'.
        if args.decoder is not None:
            cfg.decoder = args.decoder
            # Size the conv decoder before logging, so the log shows the values used.
            for _c in ("conv_dec_base_ch", "conv_dec_res_blocks", "conv_dec_min_ch"):
                _cv = getattr(args, _c, None)
                if _cv is not None:
                    setattr(cfg, _c, int(_cv))
            if ddp.is_main:
                print(f"[train_codec] decoder={cfg.decoder} "
                      f"(base_ch={cfg.conv_dec_base_ch}, res_blocks={cfg.conv_dec_res_blocks})"
                      if cfg.decoder == "conv" else f"[train_codec] decoder={cfg.decoder}")
        # Decoder noise input (spectro only; zero-init, so a no-op at step 0).
        if getattr(args, "decoder_noise", False):
            if cfg.decoder == "conv":
                raise SystemExit(
                    "--decoder_noise is wired into the LINEAR SpectroDecoder only; "
                    "it is not implemented for SpectroConvDecoder."
                )
            cfg.decoder_noise = True
            if ddp.is_main:
                print("[train_codec] decoder_noise=ON (per-channel learned scale, zero-init)")
        # Raw per-channel standardization for thin spectro modalities (co2), which lifts
        # the sparse activity off the large near-constant mean so the codec does not
        # collapse to one code. No-op for ece/bes/mhr.
        apply_spectro_standardization(cfg, args.modality, args.stats_path,
                                      log_fn=(print if ddp.is_main else None))
    for _k, _f in (("slowts_d_model", "d_model"), ("slowts_enc_depth", "enc_depth"),
                   ("slowts_dec_depth", "dec_depth")):
        _v = getattr(args, _k, None)
        if _v is None:
            continue
        if not is_slowts:
            raise SystemExit(f"--{_k} is a SLOW-TS-only override; {args.modality} is not slow-TS.")
        setattr(cfg, _f, int(_v))
        if ddp.is_main:
            print(f"[train_codec] {_f} override -> {int(_v)}")
    if args.entropy_weight is not None:
        cfg.entropy_weight = float(args.entropy_weight)
    for _w in ("recon_weight", "recon_mse_weight"):
        _v = getattr(args, _w, None)
        if _v is not None:
            if not hasattr(cfg, _w):
                raise SystemExit(f"--{_w} is not a field of {type(cfg).__name__}")
            setattr(cfg, _w, float(_v))
            if ddp.is_main:
                print(f"[train_codec] {_w} override -> {float(_v)}")
    if getattr(args, "pixel_anchor_weight", None) is not None and hasattr(cfg, "pixel_anchor_weight"):
        cfg.pixel_anchor_weight = float(args.pixel_anchor_weight)
    for _w in ("multiscale_recon_weight", "freq_grad_weight", "fm_weight"):
        if getattr(args, _w, None) is not None and hasattr(cfg, _w):
            setattr(cfg, _w, float(getattr(args, _w)))
    if getattr(args, "gate_hard_min_codes", None) is not None and hasattr(cfg, "gate_hard_min_codes"):
        cfg.gate_hard_min_codes = int(args.gate_hard_min_codes)
    if args.decoder is not None and (is_video or is_slowts or is_fastts):
        raise SystemExit(
            "--decoder {linear,conv} is a SPECTRO-only override (the conv decoder is "
            "nets.SpectroConvDecoder); it is not valid for a video / slow-TS / fast-TS modality."
        )
    # Codec architecture overrides (see --codec_d_model). Set before the codec is built
    # and before resume, and stored in the pickled cfg, so code that rebuilds a codec
    # from its checkpoint picks up the width.
    for _a, _f in (("codec_d_model", "d_model"), ("codec_enc_depth", "enc_depth"),
                   ("codec_dec_depth", "dec_depth"), ("codec_heads", "heads")):
        _v = getattr(args, _a, None)
        if _v is None:
            continue
        if not hasattr(cfg, _f):
            raise SystemExit(f"--{_a}: {type(cfg).__name__} has no field {_f!r}")
        setattr(cfg, _f, int(_v))
        if ddp.is_main:
            print(f"[train_codec] {_f} override -> {_v}")
    if getattr(args, "refine_depth", None) is not None or getattr(args, "refine_hidden", None) \
            is not None or getattr(args, "refine_dilated", False):
        # Only the video and spectro configs have the refinement-head fields.
        if not hasattr(cfg, "refine_depth"):
            raise SystemExit(
                "--refine_depth/--refine_hidden/--refine_dilated are VIDEO + SPECTRO overrides "
                "(the decoder residual conv refinement head); they are not valid for a "
                "slow-TS / fast-TS modality."
            )
        if args.refine_depth is not None:
            cfg.refine_depth = int(args.refine_depth)
        if args.refine_hidden is not None:
            cfg.refine_hidden = int(args.refine_hidden)
        if getattr(args, "refine_dilated", False):
            if not hasattr(cfg, "refine_dilated"):
                raise SystemExit("--refine_dilated is SPECTRO-only (VideoDecoder has no dilation)")
            cfg.refine_dilated = True
        if cfg.refine_depth > 0 and getattr(cfg, "decoder", "linear") == "conv":
            raise SystemExit(
                "--refine_depth is a fix for the LINEAR to_pixels unpatchify's patch lattice; "
                "it is not wired into SpectroConvDecoder. Use one or the other, not both."
            )
        if ddp.is_main:
            _dil = getattr(cfg, "refine_dilated", False)
            _rf = (2 ** (cfg.refine_depth + 1) - 1) if _dil else (2 * cfg.refine_depth + 1)
            print(f"[train_codec] decoder refine_depth={cfg.refine_depth} "
                  f"(hidden={cfg.refine_hidden}, dilated={_dil}, receptive_field={_rf}, "
                  f"residual zero-init conv head)")
    # Video missing-data knobs. A video-only flag on another modality is an error rather
    # than silently ignored.
    for _vk, _flag in (("mask_missing", "mask_missing"),
                       ("require_live_channels", "require_live_channels")):
        if getattr(args, _flag, False):
            if not hasattr(cfg, _vk):
                raise SystemExit(
                    f"--{_flag} is a VIDEO-only missing-data knob (it lives on "
                    f"VideoCodecConfig); {args.modality} is not a video modality.")
            setattr(cfg, _vk, True)
            if ddp.is_main:
                print(f"[train_codec] {_vk} -> True")
    if getattr(args, "video_presence", None) is not None:
        if not is_video:
            raise SystemExit("--video_presence is a VIDEO-only flag.")
        cfg.presence_filter = args.video_presence != "off"
        if ddp.is_main:
            print(f"[train_codec] video_presence -> {args.video_presence}")
    if getattr(args, "spectro_presence", None) is not None:
        if args.modality not in SPECTRO_MODALITIES:
            raise SystemExit("--spectro_presence is a SPECTRO-only flag.")
        cfg.presence_filter = args.spectro_presence != "off"
        if ddp.is_main:
            print(f"[train_codec] spectro_presence -> {args.spectro_presence}")

    if args.consistency_weight is not None:
        if is_video or is_slowts:
            raise SystemExit(
                "--consistency_weight is not valid for a video / slow-TS modality "
                "(those codecs have no shift-consistency term; see IGNITE_DESIGN §4.3)."
            )
        cfg.consistency_weight = float(args.consistency_weight)

    # Anti-collapse overrides (see _activity_overrides; no-op for unlisted modalities).
    # --skip_activity_override bypasses them, which reproduces the plain production
    # recipe for co2/mhr (warmup 0, adversarial_weight 1.0, no activity bias).
    if args.skip_activity_override:
        if ddp.is_main:
            print(f"[train_codec] _activity_overrides SKIPPED for {args.modality} "
                  f"(--skip_activity_override)")
    else:
        apply_activity_overrides(cfg, args.modality, log_fn=(print if ddp.is_main else None))

    # Production-recipe CLI overrides, applied last so they win over the config defaults
    # and _activity_overrides. fsq_levels changes the codebook, so it must be set before
    # the codec is built (it is). Skipped when the flag is None or the cfg lacks the
    # field.
    if getattr(args, "channel_groups", None) is not None and hasattr(cfg, "channel_groups"):
        cfg.channel_groups = int(args.channel_groups)
        _nt = cfg.n_tok
        print(f"[train_codec] channel_groups override -> {cfg.channel_groups} "
              f"(patch_dim {(cfg.channels // cfg.channel_groups) * cfg.patch_f * cfg.patch_t}, "
              f"n_tok {_nt})", flush=True)
    if getattr(args, "fsq_levels", None) is not None and hasattr(cfg, "fsq_levels"):
        lv = [int(x) for x in str(args.fsq_levels).split(",") if x.strip() != ""]
        cfg.fsq_levels = lv
        if ddp.is_main:
            import math as _m
            print(f"[train_codec] fsq_levels override -> {lv} (cb={_m.prod(lv)})")
    if getattr(args, "fsq_preserve_symmetry", False) and hasattr(cfg, "fsq_preserve_symmetry"):
        cfg.fsq_preserve_symmetry = True
        if ddp.is_main:
            print("[train_codec] fsq_preserve_symmetry -> True "
                  "(FSQ symmetry_preserving_bound; changes the codes on its own)")
    # Spectral-codec recipe knobs (arXiv 2406.05298); the defaults are the current
    # values. adam_* / lr_decay_* / disc_update_every live on the cfg so the checkpoint
    # records the recipe it was trained with.
    for _knob in ("conv_dec_base_ch", "conv_dec_res_blocks", "conv_dec_min_ch",
                  "disc_update_every", "lr_decay_every", "target_time_smooth", "band_pool"):
        _v = getattr(args, _knob, None)
        if _v is not None and hasattr(cfg, _knob):
            setattr(cfg, _knob, int(_v))
            if ddp.is_main:
                print(f"[train_codec] {_knob} override -> {int(_v)}")
    if getattr(args, "multiscale_recon_scales", None) is not None and hasattr(
            cfg, "multiscale_recon_scales"):
        cfg.multiscale_recon_scales = tuple(
            int(x) for x in str(args.multiscale_recon_scales).split(",") if x.strip())
        if ddp.is_main:
            print(f"[train_codec] multiscale_recon_scales -> {cfg.multiscale_recon_scales}"
                  f"{'  (includes FULL resolution)' if 1 in cfg.multiscale_recon_scales else ''}")
    for _knob in ("ms_ssim_weight", "peak_weight", "std_weight"):
        _v = getattr(args, _knob, None)
        if _v is not None and hasattr(cfg, _knob):
            setattr(cfg, _knob, float(_v))
            if ddp.is_main:
                print(f"[train_codec] {_knob} override -> {float(_v)}")
    if getattr(args, "ms_ssim_win", None) is not None and hasattr(cfg, "ms_ssim_win"):
        cfg.ms_ssim_win = int(args.ms_ssim_win)
    if getattr(args, "discriminator", None) is not None and hasattr(cfg, "discriminator"):
        cfg.discriminator = str(args.discriminator)
        if ddp.is_main:
            print(f"[train_codec] discriminator={cfg.discriminator}")
    for _knob in ("adam_beta1", "adam_beta2", "lr_decay_gamma"):
        _v = getattr(args, _knob, None)
        if _v is not None and hasattr(cfg, _knob):
            setattr(cfg, _knob, float(_v))
            if ddp.is_main:
                print(f"[train_codec] {_knob} override -> {float(_v)}")
    for _knob in ("adaptive_adv_clamp", "adv_warmup_steps", "adversarial_weight",
                  "joint_entropy_weight", "decorrelation_weight",
                  "joint_entropy_ramp_steps", "fsq_noise_dropout"):
        _v = getattr(args, _knob, None)
        if _v is not None and hasattr(cfg, _knob):
            setattr(cfg, _knob, _v)
            if ddp.is_main:
                print(f"[train_codec] {_knob} override -> {_v}")
    for _knob in ("joint_entropy_weight", "decorrelation_weight",
                  "joint_entropy_ramp_steps", "fsq_noise_dropout", "target_time_smooth",
                  "peak_weight"):
        if getattr(args, _knob, None) is not None and not hasattr(cfg, _knob):
            raise SystemExit(
                f"--{_knob} is a SPECTRO-only knob (it lives on SpectroCodecConfig); "
                f"{args.modality}'s {type(cfg).__name__} has no such field."
            )
    for _knob in ("patch_w", "stem_layers", "stem_channels", "stem_kernel", "recon_loss",
                  "ssim_weight", "ssim_win", "fastts_target"):
        if getattr(args, _knob, None) is not None and not is_fastts:
            raise SystemExit(
                f"--{_knob} is a FAST-TS-only knob (it lives on FastTSCodecConfig); "
                f"{args.modality} is not the fast-TS modality."
            )
    # --gain_shape and its knobs drive both the fast-TS envelope codec and the spectro
    # codec. Fast-TS splits off the per-channel mean over envelope bins, spectro the
    # per-(channel, frequency) mean and std over time; the flags and the direct gain
    # supervision are the same.
    _gs_knobs = ("gain_tokens", "gain_scale", "gain_weight")
    _gs_family = is_fastts or is_spectro
    for _knob in _gs_knobs:
        if getattr(args, _knob, None) is not None and not _gs_family:
            raise SystemExit(
                f"--{_knob} configures the gain-shape decomposition, which exists on the "
                f"FAST-TS and SPECTRO codecs only; {args.modality} is neither."
            )
    if getattr(args, "gain_shape", False) and not _gs_family:
        raise SystemExit(
            "--gain_shape is the envelope/shape decomposition of the FAST-TS and SPECTRO "
            f"codecs; {args.modality} is neither."
        )
    if any(getattr(args, k, None) is not None for k in _gs_knobs) \
            and not getattr(args, "gain_shape", False):
        raise SystemExit(
            "--gain_tokens/--gain_scale/--no_gain_scale/--gain_weight require --gain_shape "
            "(they configure the envelope/shape decomposition, which is OFF by default)."
        )
    if is_spectro and getattr(args, "gain_shape", False):
        # gain_tokens must divide freq_bins and stay below n_tok (both checked in
        # SpectroCodecConfig.__post_init__), so rebuild the dataclass instead of
        # mutating it.
        _gs = {"gain_shape": True}
        if args.gain_tokens is not None:
            _gs["gain_tokens"] = int(args.gain_tokens)
        if args.gain_scale is not None:
            _gs["gain_scale"] = bool(args.gain_scale)
        if args.gain_weight is not None:
            _gs["gain_weight"] = float(args.gain_weight)
        # Local alias: the fast-TS branch below also imports ``replace as _dc_replace``,
        # which makes that name function-local, so using it here would raise
        # UnboundLocalError.
        from dataclasses import replace as _dc_replace_local
        cfg = _dc_replace_local(cfg, **_gs)
        if ddp.is_main:
            print(
                f"[train_codec] SPECTRO ENVELOPE/SHAPE SPLIT: gain_tok={cfg.n_gain_tok} "
                f"({cfg.gain_bits:.1f} of {cfg.n_tok * math.log2(cfg.codebook_size):.1f} bits) "
                f"shape_tok={cfg.n_shape_tok} gain_scale={cfg.uses_gain_scale} "
                f"gain_patch_f={cfg.gain_patch_f} bins ({cfg.gain_values_per_tok} "
                f"values/gain token) gain_weight={cfg.gain_weight}; "
                f"recon.mean(-1) IS the coded envelope and recon.std(-1) IS the coded "
                f"amplitude BY CONSTRUCTION",
                flush=True,
            )

    # Per-frequency log-power z-score (spectro only; no-op without --logpow_stats_path).
    # It composes with the raw z-score from apply_spectro_standardization, so the stats
    # must be computed in the space the codec sees, i.e. after raw-std for co2. Without
    # the raw z-score co2's log-power is all clipped at data._LOG_CEIL=20, and stats
    # computed there are degenerate (mean 20, std 0).
    if (not (is_video or is_slowts or is_fastts)) and getattr(args, "logpow_stats_path", None):
        import torch as _t
        _st = _t.load(args.logpow_stats_path, map_location="cpu", weights_only=False)
        # Legacy stats files store tensors, newer ones nested lists, and truthiness of a
        # tensor raises, so go through torch.as_tensor and read the trailing (freq)
        # axis.
        _sf = int(_t.as_tensor(_st["mean"]).shape[-1]) if "mean" in _st else -1
        # If freq_bins is smaller than the stats width, the spectrogram is cropped to
        # the low band (data._crop_pad_freq_time keeps the first freq_bins), so the
        # stats are sliced to match. Stats narrower than freq_bins are an error. The
        # crop exists for ece, whose mode tracks sit below ~20 kHz: band-power pooling
        # would smear them out, while cropping keeps the 0.49 kHz resolution there and
        # drops the broadband range that would use most of the tokens.
        if _sf < cfg.freq_bins:
            raise SystemExit(
                f"--logpow_stats_path {args.logpow_stats_path} has F={_sf}, NARROWER than this "
                f"codec's freq_bins={cfg.freq_bins} (stft_n_fft={cfg.stft_n_fft}). Regenerate "
                f"with --compute_logpow_stats and the SAME --stft_n_fft/--freq_bins."
            )
        _mean = _t.as_tensor(_st["mean"], dtype=_t.float32)
        _std = _t.as_tensor(_st["std"], dtype=_t.float32)
        if _sf > cfg.freq_bins:
            _mean, _std = _mean[..., :cfg.freq_bins], _std[..., :cfg.freq_bins]
            if ddp.is_main:
                print(f"[train_codec] per-freq stats SLICED {_sf} -> {cfg.freq_bins} bins "
                      f"(band crop to 0-{cfg.freq_bins * 250.0 / 512:.1f} kHz)", flush=True)
        cfg.logpow_freq_mean = _mean.tolist()
        cfg.logpow_freq_std = _std.tolist()
        cfg.logpow_standardize = True
        if ddp.is_main:
            print(f"[train_codec] per-freq log-z ON from {args.logpow_stats_path} "
                  f"(C={len(cfg.logpow_freq_mean)}, F={len(cfg.logpow_freq_mean[0])}); "
                  f"COMPOSED with raw-std (raw z {'ON' if cfg.input_standardize else 'off'})",
                  flush=True)

    # Per-window instance z-score (fix for co2's DC offset saturating the FSQ). Applied
    # last in log_power_stft, after raw-std, so each window has mean ~0 and std ~1 and
    # the encoder stays off the tanh bound.
    if (not (is_video or is_slowts or is_fastts)) and getattr(args, "input_instance_norm", False):
        cfg.input_instance_norm = True
        if getattr(args, "instance_norm_quantize", None) is not None:
            cfg.instance_norm_quantize = float(args.instance_norm_quantize)
        if ddp.is_main:
            print("[train_codec] per-window instance z-score ON (mean~0/std~1) — "
                  f"DC-offset / FSQ-saturation fix; quantize={cfg.instance_norm_quantize} "
                  "(>0 = shift-robust piecewise-constant stats)", flush=True)

    # Fast-TS geometry and scaling. patch_w/stem_* set the raw-sample codec's token
    # count and receptive field; the per-channel raw mean/std bring the ~1e15 raw signal
    # to O(1) (--stats_path='' disables, None uses DEFAULT_STATS_PATH).
    if is_fastts:
        from dataclasses import replace as _dc_replace

        from .fastts_train import DEFAULT_STATS_PATH, load_fastts_channel_stats
        # patch_w sets n_tok (and so the world-model frame size) and __post_init__
        # checks divisibility, so rebuild the dataclass instead of mutating it.
        _geom = {k: v for k, v in (
            ("patch_w", args.patch_w),
            ("stem_layers", args.stem_layers),
            ("stem_channels", args.stem_channels),
            ("stem_kernel", args.stem_kernel),
        ) if v is not None}
        # Gain-shape splits the same 5 tokens into gain and shape groups, so it goes
        # through the same rebuild, whose __post_init__ checks gain_tokens.
        _gs = {k: v for k, v in (
            ("gain_shape", True if getattr(args, "gain_shape", False) else None),
            ("gain_tokens", args.gain_tokens),
            ("gain_scale", args.gain_scale),
            ("gain_weight", args.gain_weight),
        ) if v is not None}
        # (The "requires --gain_shape" check is shared with spectro and runs earlier.)
        _geom.update(_gs)
        if _geom:
            cfg = _dc_replace(cfg, **{
                k: (v if k in ("gain_shape", "gain_scale") else
                    (float(v) if k == "gain_weight" else int(v)))
                for k, v in _geom.items()})
        if getattr(args, "recon_loss", None) is not None:
            cfg.recon_loss = args.recon_loss
        if getattr(args, "ssim_weight", None) is not None:
            cfg.ssim_weight = float(args.ssim_weight)
        if getattr(args, "ssim_win", None) is not None:
            cfg.ssim_win = int(args.ssim_win)
        if ddp.is_main:
            print(
                f"[train_codec] fast-TS target={cfg.target} "
                f"{'RAW-SAMPLE' if cfg.is_raw else 'ENVELOPE'} geometry: window={cfg.window} "
                f"patch_w={cfg.patch_w} -> n_tok={cfg.n_tok} "
                f"({cfg.channels * cfg.patch_w} values/token, "
                f"{cfg.bits_per_value:.4f} bits/value over {cfg.channels * cfg.window} values); "
                f"world-model frame 1017 - 5 + {cfg.n_tok} = {1017 - 5 + cfg.n_tok}; "
                f"stem={cfg.stem_layers}x{cfg.stem_channels}k{cfg.stem_kernel} "
                f"recon_loss={cfg.recon_loss} pixel_anchor={cfg.pixel_anchor_weight} "
                f"ssim_w={cfg.ssim_weight}@win{cfg.ssim_win} "
                f"adv={cfg.adversarial_weight} fm={cfg.fm_weight} "
                f"consistency={cfg.consistency_weight}"
                + (f" | GAIN-SHAPE gain_tok={cfg.n_gain_tok} ({cfg.gain_bits:.2f} bits) "
                   f"shape_tok={cfg.n_shape_tok} gain_scale={cfg.uses_gain_scale} "
                   f"gain_values={cfg.gain_values} gain_weight={cfg.gain_weight}"
                   if cfg.n_gain_tok > 0 else ""),
                flush=True,
            )
        stats_path = DEFAULT_STATS_PATH if args.stats_path is None else args.stats_path
        if stats_path:
            mean, std = load_fastts_channel_stats(stats_path, args.modality)
            cfg.channel_mean = mean
            cfg.channel_std = std
            if ddp.is_main:
                print(
                    f"[train_codec] fast-TS raw standardization ON: per-channel raw stats "
                    f"from {stats_path} (C={len(mean)}, std range "
                    f"[{min(std):.3e}, {max(std):.3e}])",
                    flush=True,
                )
        elif ddp.is_main:
            print("[train_codec] WARNING: --stats_path='' -> fast-TS raw standardization "
                  "OFF (unstandardized ~1e15 input). Debugging only.", flush=True)

    # Slow-TS scaling: set the per-signal preprocessing (log_standardize / standardize)
    # and per-channel mean/std so the input is O(1), as the FM sees it. Raw Thomson
    # density (~1e19) collapses the codec to one code. --stats_path='' disables
    # (debugging only), None uses DEFAULT_STATS_PATH.
    if is_slowts:
        stats_path = None if args.stats_path == "" else args.stats_path
        if args.stats_path != "":
            method, mean, std = load_slowts_channel_stats(args.modality, stats_path)
            cfg.preprocess_method = method
            cfg.channel_mean = mean
            cfg.channel_std = std
            if ddp.is_main:
                from .fastts_train import DEFAULT_STATS_PATH as _DSP
                print(
                    f"[train_codec] slow-TS input standardization ON ({method}): per-channel "
                    f"{'log' if method == 'log_standardize' else 'raw'}-space stats from "
                    f"{stats_path or _DSP} (C={len(mean)}, mean range "
                    f"[{min(mean):.3e}, {max(mean):.3e}], std range "
                    f"[{min(std):.3e}, {max(std):.3e}])",
                    flush=True,
                )
        elif ddp.is_main:
            print("[train_codec] WARNING: --stats_path='' -> slow-TS input standardization OFF "
                  "(raw path; high-magnitude density signals will collapse). Debugging only.",
                  flush=True)

    # Train and disjoint eval shot lists. Discovery is sorted, so every rank derives the
    # same split.
    if args.shots is not None:
        all_shots = [s.strip() for s in args.shots.split(",") if s.strip()]
    else:
        all_shots = spike.discover_shots(args.data_dir)

    # Drop whole shots without data for this modality (co2 only, see
    # _PRESENCE_FILTER_SIGNALS), before the split so both sides draw from present shots.
    # Rank 0 scans, caches and broadcasts; the cache is keyed by (paths, signal).
    if args.modality in _PRESENCE_FILTER_SIGNALS:
        from ..data.multi_file_dataset import filter_signal_present_files

        presence_cache = (
            Path(args.lengths_cache_dir) / f"codec_{args.modality}_present.pt"
            if args.lengths_cache_dir
            else None
        )
        paths = _shot_paths(all_shots, args.data_dir)
        kept = {
            p.name.split("_")[0]
            for p in filter_signal_present_files(
                paths, args.modality, cache_path=presence_cache
            )
        }
        before = len(all_shots)
        all_shots = [s for s in all_shots if str(s) in kept]
        if ddp.is_main:
            print(
                f"[train_codec] {args.modality} presence filter: kept {len(all_shots)}/{before} "
                f"shots that contain {args.modality} data",
                flush=True,
            )
        if not all_shots:
            raise RuntimeError(
                f"presence filter left 0 shots for {args.modality} under {args.data_dir}"
            )

    # Spectro presence filter from the liveness cache, before the split.
    if args.modality in SPECTRO_MODALITIES and \
            getattr(args, "spectro_presence", None) not in (None, "off"):
        _before = len(all_shots)
        all_shots = spectro_live_shots(
            args.modality, all_shots,
            require_all=(args.spectro_presence == "all"),
            log_fn=(print if ddp.is_main else None),
        )
        if not all_shots:
            raise RuntimeError(
                f"spectro presence filter left 0 shots for {args.modality} "
                f"(started from {_before})")

    # Video presence filter (per divertor, from the liveness cache), before the split.
    if is_video and getattr(args, "video_presence", None) not in (None, "off"):
        _before = len(all_shots)
        all_shots = video_live_shots(
            args.modality, all_shots,
            require_all=(args.video_presence == "all"),
            log_fn=(print if ddp.is_main else None),
        )
        if not all_shots:
            raise RuntimeError(
                f"video presence filter left 0 shots for {args.modality} "
                f"(started from {_before})")

    # eval = last eval_n_shots; train = the rest, capped to n_shots.
    eval_shots = all_shots[-args.eval_n_shots:]
    train_pool = all_shots[: -args.eval_n_shots] if args.eval_n_shots > 0 else all_shots
    train_shots = train_pool[: args.n_shots]
    if not train_shots:
        raise RuntimeError(
            f"no train shots (discovered {len(all_shots)}, eval_n_shots={args.eval_n_shots})"
        )

    # Write the presence-filtered lengths sidecar and exit. Runs after shot resolution,
    # so train_shots is the filtered list the training runs will use.
    if getattr(args, "build_presence_lengths", False):
        if not args.lengths_cache_dir:
            raise SystemExit("--build_presence_lengths needs --lengths_cache_dir")
        if (getattr(args, "spectro_presence", None) in (None, "off")
                and getattr(args, "video_presence", None) in (None, "off")):
            raise SystemExit(
                "--build_presence_lengths needs --spectro_presence (or --video_presence) "
                "set: without it the resolved shot list is the UNFILTERED one and the "
                "derived sidecar would just duplicate codec_<mod>_lengths.pt."
            )
        derive_presence_lengths_sidecar(
            args.modality, args.lengths_cache_dir, shots=train_shots,
            data_dir=args.data_dir, log_fn=(print if ddp.is_main else None),
        )
        ddp.shutdown()
        return

    # Write dataset-level per-freq stats over the train shots (in the composed space)
    # and exit. Single process, no DDP collectives.
    if getattr(args, "compute_logpow_stats", None):
        if is_video or is_slowts or is_fastts:
            raise SystemExit("--compute_logpow_stats is spectro-only")
        # The (C, F) stats are only valid on the STFT grid they were computed on, so
        # pass the codec's geometry through.
        compute_logpow_stats(
            args.modality, train_shots, args.compute_logpow_stats,
            data_dir=args.data_dir, log_fn=(print if ddp.is_main else None),
            stft_n_fft=cfg.stft_n_fft, stft_hop=cfg.stft_hop,
            freq_bins=cfg.freq_bins, time_frames=cfg.time_frames,
        )
        ddp.shutdown()
        return

    resume_state = None
    if args.resume is not None:
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)

    if ddp.is_main:
        print(
            f"[train_codec] modality={args.modality} channels={channels} "
            f"train_shots={len(train_shots)} eval_shots={len(eval_shots)} "
            f"world_size={ddp.world_size} batch_size={args.batch_size} "
            f"num_workers={args.num_workers} steps={args.steps} device={device}",
            flush=True,
        )

    t0 = time.time()
    # The lengths cache is reused only when its stored path list matches hdf5_paths. A
    # presence filter changes that list, so the shared codec_<mod>_lengths.pt would miss
    # and every run would cold-scan all shots (a 2 h leg once spent its whole time doing
    # that). Filtered lists get their own sidecar: cold once, warm afterwards, and the
    # shared file is never written.
    _presence_on = (
        getattr(args, "spectro_presence", None) not in (None, "off")
        or getattr(args, "video_presence", None) not in (None, "off")
    )
    _suffix = "_presence" if _presence_on else ""
    lengths_cache_path = args.lengths_cache_dir and (
        Path(args.lengths_cache_dir) / f"codec_{args.modality}{_suffix}_lengths.pt"
    )
    if is_fastts:
        # The fast-TS trainer takes modality as keyword-only; reuse it as is.
        from .fastts_train import train_fastts_codec
        final_gate = train_fastts_codec(
            cfg,
            train_shots,
            eval_shots,
            steps=args.steps,
            eval_every=args.eval_every,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            modality=args.modality,
            data_dir=args.data_dir,
            lr=args.lr,
            disc_lr=args.disc_lr,
            ema=args.ema,
            ema_decay=args.ema_decay,
            eval_batches=args.eval_batches,
            eval_batch_size=args.eval_batch_size,
            eval_frames=args.eval_frames,
            out_dir=args.out_dir,
            seed=args.seed,
            ddp=ddp,
            resume_state=resume_state,
            lengths_cache_path=lengths_cache_path,
        )
    else:
        _extra_kw = {}
        if is_video:
            trainer = train_video_codec
        elif is_slowts:
            trainer = train_slowts_codec
            # slow-TS-only kwarg; the spectro/video trainers do not accept it.
            _extra_kw["freeze_encoder"] = bool(getattr(args, "slowts_freeze_encoder", False))
        else:
            trainer = train_codec
        if getattr(args, "slowts_freeze_encoder", False):
            if not is_slowts:
                raise SystemExit(
                    "--slowts_freeze_encoder is a SLOW-TS-only override; "
                    f"{args.modality} is not slow-TS.")
            if not args.resume:
                raise SystemExit(
                    "--slowts_freeze_encoder requires --resume: freezing a randomly "
                    "initialised encoder would fix MEANINGLESS codes.")
        final_gate = trainer(
            cfg,
            args.modality,
            train_shots,
            eval_shots,
            steps=args.steps,
            eval_every=args.eval_every,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
            lr=args.lr,
            disc_lr=args.disc_lr,
            ema=args.ema,
            ema_decay=args.ema_decay,
            eval_batches=args.eval_batches,
            eval_batch_size=args.eval_batch_size,
            eval_frames=args.eval_frames,
            out_dir=args.out_dir,
            seed=args.seed,
            ddp=ddp,
            resume_state=resume_state,
            lengths_cache_path=lengths_cache_path,
            **_extra_kw,
        )
    final_gate["wall_s"] = time.time() - t0

    if ddp.is_main and args.out_dir is not None:
        summary = dict(final_gate)
        summary["config"] = {
            "modality": args.modality,
            "channels": channels,
            "n_train_shots": len(train_shots),
            "n_eval_shots": len(eval_shots),
            "steps": args.steps,
            "eval_every": args.eval_every,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "world_size": ddp.world_size,
            "lr": args.lr,
            "disc_lr": args.disc_lr,
            "ema": args.ema,
            "seed": args.seed,
            "resumed_from": args.resume,
        }
        with open(Path(args.out_dir) / "summary.json", "w") as fh:
            json.dump(spike._jsonable(summary), fh, indent=2, sort_keys=True)
        print(
            f"[train_codec] DONE steps={final_gate.get('steps')} "
            f"global_step={final_gate.get('global_step')} "
            f"best_score={final_gate.get('best_score')} "
            f"best_step={final_gate.get('best_step')} wall_s={final_gate['wall_s']:.1f}",
            flush=True,
        )

    ddp.barrier()
    ddp.shutdown()
    return final_gate


if __name__ == "__main__":
    main()
