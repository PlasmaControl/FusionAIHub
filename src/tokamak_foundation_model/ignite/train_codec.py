"""IGNITE Phase-A **streaming, DDP-capable** production codec trainer.

The gate-spike (:mod:`ignite.spike`) pre-builds a FIXED batch pool via
``spike.real_ece_batches``, which opens shots SEQUENTIALLY (~19 min for 40 shots) and
caps around 40 shots. That does not scale to the thousands of shots we want a production
codec to see.

This module replaces the fixed pool with the **production data pipeline** — it reuses
:class:`~tokamak_foundation_model.data.multi_file_dataset.TokamakMultiFileDataset`, the
exact same class the main foundation-model trainer streams 7878 shots through under DDP.
That class already implements everything the old hand-rolled ``StreamingSpectroPairs``
re-invented: a global-idx → ``(file_idx, chunk_idx)`` map via cumulative-length binary
search, a per-worker LRU HDF5 file-handle cache (with correct ``__getstate__`` /
``__setstate__`` pickling into DataLoader workers), a length-cache sidecar, and a
``make_dataloader`` factory.

Design
------
* **Data**: :class:`CodecPairDataset` is a THIN subclass of ``TokamakMultiFileDataset``.
  It reuses the parent's idx-mapping + LRU file handles + length cache + num_workers
  machinery UNCHANGED and overrides ONLY the per-item transform hook
  (:meth:`_getitem_standard`) to return the codec's ``(spec_a, spec_b)`` δ-shift log-power
  pair for ONE spectro modality (ece / co2 / bes / mhr) instead of the processed
  multi-modal training dict. The parent's ``__getitem__`` still performs the binary-search
  index map and sets ``self.h5_file = self._get_file_handle(file_idx)`` before dispatching
  to our ``_getitem_standard(chunk_idx)`` — so the file-handle safety under
  ``num_workers > 0`` is inherited verbatim (each worker owns its own object copy + LRU).
* **Training**: reuses :func:`ignite.spike.codec_train_step` (the SINGLE per-step
  generator/discriminator alternation shared with the spike) so the training math is
  identical. The codec is DDP-wrapped through a thin :class:`_GenLossAdapter` whose
  ``forward`` dispatches to ``generator_losses`` — this makes DDP's autograd reducer hook
  the generator loss (the discriminator has its own optimizer / graph and is wrapped
  separately).
* **Gate + best-ckpt**: periodically streams a held-out eval set (disjoint shots) and
  computes the §4.4 gate via :func:`ignite.spike.compute_gate`, selecting the best
  checkpoint by :func:`ignite.spike.gate_score` — exactly as the spike does. Rank-0 only
  logs + writes checkpoints.

Reuse boundary (§7): only ``torch`` + sibling ``ignite`` modules + the read-only data
pipeline (``data.data_loader`` / ``data.multi_file_dataset``). No FAITH *model* code.
"""
from __future__ import annotations

import argparse
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

# spectro modalities the codec can train on. All are apply_stft=True + target_fs=500e3
# (== STFT_FS) in TokamakH5Dataset.SIGNAL_CONFIGS, so their raw windows STFT to the same
# grid data.log_power_stft expects. Channel counts (after channels_to_use) are read from
# the loader's SignalConfig at runtime — NOT hard-coded here.
# "mirnov" has NO trained FSQ codec and is not part of the Phase-A codec family — it is here
# because CodecPairDataset is also the STFT front end the band-power tokeniser reads through
# (29 magnetic channels, added 2026-08-15). Loading/STFT parameters come from its SIGNAL_CONFIGS
# entry; a caller wanting codec-side fields must borrow another modality's cfg.
SPECTRO_MODALITIES = ("ece", "co2", "bes", "mhr", "mirnov")

# video (tangtv) modalities — the two divertor codecs. Each is a MovieConfig in
# TokamakH5Dataset.MOVIE_CONFIGS: tangtv_lower keeps raw camera channels [0, 2]
# (LODIV PAR-int + LODIV PERP); tangtv_upper keeps [4, 6] (UPDIV0 PERP + UPDIV0 PAR).
# Both are 2-channel after channels_to_use (read at runtime, NOT hard-coded here).
VIDEO_MODALITIES = ("tangtv_lower", "tangtv_upper")

# slow-TS modalities — the 7 smooth kinetic-profile signals (Thomson / CER / MSE). Each is a
# non-STFT SignalConfig in TokamakH5Dataset.SIGNAL_CONFIGS at target_fs=100 Hz. Thomson uses
# zero_is_missing; CER/MSE use an explicit NaN mask (see config.SLOWTS_* + data_loader). One
# codec per signal, each frozen independently (§4.3 "lightest touch").
SLOWTS_MODALITIES = tuple(_SLOWTS_SIGNALS)

# fast-TS (filterscopes) modality — the RAW-SAMPLE (10 kHz waveform) codec. Trained by the separate
# ``fastts_train`` module (kept separate earlier to avoid a parallel-edit collision with the
# slow-TS work); ``main()`` folds its launch into this dispatch so ``--modality filterscopes``
# routes here like every other modality. The value MUST equal ``fastts_train.FASTTS_MODALITY``
# (defined here too, not imported, because fastts_train imports FROM this module — a top-level
# import back would be circular).
FASTTS_MODALITY = "filterscopes"

DEFAULT_DATA_DIR = spike.DEFAULT_DATA_DIR


# ------------------------------------------------------------------------------------- #
# per-modality anti-collapse overrides (turned ON only for the 4 collapsing codecs; every
# OTHER modality keeps its config default, so their byte-identical behavior is preserved).
# ------------------------------------------------------------------------------------- #
# Applied to the cfg in ``main()`` after channel-count sizing (like ``modality_channels``). The
# already-working codecs (ece/bes/cer_ti/cer_rot/ts_core_temp) are ABSENT here → active_bias
# stays 0 (stratification OFF, byte-identical) and adv_warmup/adversarial_weight stay at their
# config defaults. The original four failing codecs collapsed to 1 code from step 0 in the
# real 20k-step runs (frac_codes_used=0.001, minH≈0, env_corr≈0/nan); mse and ts_tangential_*
# joined later with the same missing-dominated slow-TS signature.
#
# The thresholds are read off the CPU activity diagnostic (fraction of ACTIVE windows on a spread
# of real shots):
#   * co2 (spectro): only ~33% of windows have co2 data at all, and the built log-power windows
#     have a MEDIAN std of ~0.024 (5-95 pct 0.009-0.318) vs ~0.85 for ece/bes/mhr — i.e. co2 is
#     degeneracy-dominated (mostly floored at ~-10). min_activity=0.10 selects roughly the top
#     ~10-20% most-structured windows. co2 ALSO showed adversarial sharpness spikes (13-509 in the
#     real runs), so it additionally gets the adversarial warmup + lower adversarial_weight below.
#   * filterscopes (fast-TS): ~76% of built ELM-envelope windows saturate the log1p ceiling to a
#     CONSTANT (std 0); the minority ~21% are strongly structured (env std >= 2.4). min_activity=0.5
#     cleanly separates the two (every structured window is >= 2.4). See the REPORT caveat: the
#     saturation itself is a scale artifact (raw filterscope counts ~1e13-1e15 crush log1p to the
#     ceiling), so stratification biases onto the learnable minority but does NOT rescale the
#     envelope — flagged as a residual concern.
#   * ts_core_density (slow-TS, MASKED): strongly bimodal present-fraction — median ~0.07 but ~34%
#     of windows are >= 0.75 present. min_activity=0.5 (present-fraction) biases onto the ~38% of
#     well-observed windows.
#   * tangtv_lower (video): NOT degeneracy-dominated (every built clip has frame std >= ~3.8, like
#     the working spectro modalities). Its 20k-step run briefly reached env_corr 0.15 then FELL BACK
#     with adversarial sharpness spikes (13-509) — an adversarial-instability collapse. So it gets
#     NO activity stratification (active_bias stays 0) and ONLY the adversarial warmup + lower
#     adversarial_weight.
#
# adv_warmup_steps + adversarial_weight are already cfg-driven inside generator_losses (spectro +
# video); setting them here changes ONLY the two adversarially-unstable codecs (co2, tangtv_lower).
_activity_overrides: Dict[str, Dict[str, float]] = {
    # spectro
    "co2": {"min_activity": 0.10, "active_bias": 0.5,
            "adv_warmup_steps": 1500, "adversarial_weight": 0.5},
    # mhr (spectro): adversarial-instability anti-collapse ONLY (like co2's adv knobs), no
    # activity bias needed (mhr is rich / naturally O(1), not degeneracy-dominated).
    "mhr": {"adv_warmup_steps": 1500, "adversarial_weight": 0.5},
    # mirnov (spectro, 29 magnetic channels, added 2026-08-19): NEVER had a codec, so no recipe
    # was ever tuned. With NO entry it collapsed from step 1000 (util frac 0.0000, minH 0.000);
    # with mhr's adversarial-only knobs it collapsed AGAIN. Its data is clean (no dead/flat
    # channels, per-freq std median ~1.07) but its per-freq MEAN reaches -9.996 — the same large
    # DC offset that makes co2 degeneracy-dominated. So it gets co2's FULL recipe: activity
    # stratification as well as the adversarial knobs.
    "mirnov": {"min_activity": 0.10, "active_bias": 0.5,
               "adv_warmup_steps": 1500, "adversarial_weight": 0.5},
    # video (adversarial instability only; NO activity bias)
    "tangtv_lower": {"adv_warmup_steps": 1500, "adversarial_weight": 0.5},
    # slow-TS (masked; activity = present-fraction). No discriminator -> no adv knobs.
    "ts_core_density": {"min_activity": 0.5, "active_bias": 0.5},
    # mse (Motional Stark Effect, slow-TS) is neutral-beam-dependent: ~75% of windows are <10%
    # present (NBI off), median present-fraction 0.000. Without stratification the codec drowns in
    # mostly-missing windows -> 1-code collapse + decode-corr oscillation (confirmed 2026-07-27).
    # Same present-fraction stratification that rescued ts_core_density (biases onto the ~25% of
    # well-observed windows). Pairs with the non-finite-channel masking fix in _fit_window.
    "mse": {"min_activity": 0.5, "active_bias": 0.5},
    # ts_tangential_* (slow-TS): same missing-dominated bimodal present-fraction as
    # ts_core_density (measured 2026-08-05 over 12 spread shots: median 0.067, 59% of windows
    # <= 0.1 present, 34.75% >= 0.5 — density and temp share the identical mask, same laser),
    # but both were left unstratified in the v6 fleet. ts_tangential_density DEGRADED over its
    # 80k v6 run (decode corr 0.30@20k -> 0.03@80k, distinct codes 11 -> 5) as the mostly-
    # missing windows swamped the batches. NOTE (2026-08-05): slow-TS stratification was a
    # SILENT NO-OP at production scale until the _chunks_in_current_file re-draw-bound fix —
    # the ts_core_density/mse entries above never actually stratified their production runs
    # (their improvements came from standardization/masking); these tangential runs are the
    # first slow-TS runs where the mechanism is live.
    "ts_tangential_density": {"min_activity": 0.5, "active_bias": 0.5},
    "ts_tangential_temp": {"min_activity": 0.5, "active_bias": 0.5},
    # fast-TS (envelope std). The scale-fix de-saturates the envelope (median within-window std
    # 0.648, 91% of windows structured) — the DATA is rich, NOT low-info. Yet the codec pinned to
    # 1 code (frac 0.001) even under entropy_weight=5.0 (2026-07-27). Root cause: filterscopes was
    # the ONLY collapse-prone codec running full adversarial_weight=1.0 with ZERO adv warmup, so
    # the discriminator hammered a from-scratch encoder from step 0 -> adversarial-driven 1-code
    # collapse. Fix = the SAME adv-warmup + halved adversarial_weight its siblings already get.
    #
    # RAW-MODE EXEMPT (2026-09-03): apply_activity_overrides SKIPS this entry when the config is
    # the sample-wise codec (cfg.is_raw). BOTH halves are wrong there — min_activity 0.5
    # thresholds the ELM-ENVELOPE std (median 0.648), whereas the raw window's std has median
    # ~0.007 in standardized units, so 0.5 would reject essentially every window; and
    # adversarial_weight 0.5 would SILENTLY re-enable the GAN that the raw codec deliberately
    # defaults OFF. The entry stays so the ENVELOPE codec's behaviour is byte-identical.
    "filterscopes": {"min_activity": 0.5, "active_bias": 0.5,
                     "adv_warmup_steps": 1500, "adversarial_weight": 0.5},
}

# Modalities that need WHOLE-SHOT presence filtering (fix (a) for co2 collapse). A diagnostic
# recorded on only a subset of shots stores a length-1 (C, 1) placeholder on the shots that
# lack it; that placeholder floors to silence and swamps the codec batch, collapsing the FSQ
# to one code. Within-shot activity stratification cannot rescue a shot with zero signal (no
# active window to re-draw to), so these shots are dropped up front via
# filter_signal_present_files. Only co2 qualifies today: it is absent on the older shot range
# (placeholder shape (4, 1)) and present on the newer one (shape (4, ~4.5M)). ece/bes/mhr are
# present broadly and are NOT filtered (byte-identical). Slow-TS neutral-beam gaps (cer/mse)
# are handled by masking, not shot-dropping, so they are not listed here.
_PRESENCE_FILTER_SIGNALS: frozenset = frozenset({"co2"})

# Per-modality spectro INPUT-STANDARDIZATION (global per-freq z-score of the log-power input).
# For THIN modalities the log-power window is a large near-constant plate (co2 ~20 per freq,
# across-window std ~0.005 for most bins) with real signal in only a few bins. Pure MAE on that
# plate is minimized by a constant -> the FSQ collapses to one code regardless of token count
# (confirmed: both 192-tok and coarse-16 collapsed). Subtracting the dataset per-freq mean and
# dividing by the per-freq std (clamped to a FLOOR so near-constant "noise" bins are not blown
# up) lifts the informative bins to O(1), so a constant can no longer minimize the loss and the
# few real bins become codeable. Stats are computed in the CODEC's OWN log_power_stft space
# (the FM's log_per_bin stats are in different log units -> misaligned), keyed on the modality.
# co2 keeps the FINE 192-token resolution (user 2026-07-24: coarse arch did NOT help; the lever
# is the input scale, not token count). No-op for ece/bes/mhr (rich, naturally O(1)).
_SPECTRO_STANDARDIZE_SIGNALS: frozenset = frozenset({"co2"})


def apply_spectro_standardization(cfg, modality: str, stats_path=None, log_fn=None) -> None:
    """Enable RAW per-channel input standardization on ``cfg`` IN PLACE for non-O(1)-raw spectros.

    No-op for any modality not in :data:`_SPECTRO_STANDARDIZE_SIGNALS`. Loads the FM's
    ``preprocessing_stats[modality]['raw']`` per-channel (C,) mean/std (raw is raw — no log-unit
    mismatch, unlike a log-power stat) and sets ``cfg.input_standardize`` + ``cfg.raw_mean`` /
    ``cfg.raw_std`` (ckpt-serializable). ``log_power_stft`` z-scores the raw per channel BEFORE the
    STFT, shifting co2's log-power out of the _LOG_CEIL clip. Must run BEFORE the codec/dataset are
    built. Missing/degenerate stats -> standardization stays OFF (warn), never crash.
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
    # sanitize: a non-finite / zero-std channel would poison the z-score -> center 0, scale 1.
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
    """Set the per-modality anti-collapse overrides on ``cfg`` IN PLACE (no-op if not listed).

    Only the four collapsing codecs appear in :data:`_activity_overrides`; every other modality
    is left at its config default, so this is a no-op for the already-working codecs. Each field
    is set only if ``cfg`` actually has it (video / slow-TS lack some adversarial knobs), so a
    stray field never crashes a codec family that doesn't use it.
    """
    ov = _activity_overrides.get(modality)
    if not ov:
        return
    # The fast-TS entry is calibrated for the ENVELOPE codec only (see the table note): its
    # min_activity threshold is an envelope-std threshold, and its adversarial_weight would
    # silently switch the GAN back on for the raw codec. getattr so a config pickled before
    # `target` existed still takes the envelope branch.
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
    """Dataset-level per-(channel,freq) log-power mean/std for ``--logpow_stats_path``.

    COMPOSE convention (2026-07-31): stats are computed in the space the codec actually
    SEES — i.e. AFTER ``apply_spectro_standardization``'s raw z-score for the modalities
    in ``_SPECTRO_STANDARDIZE_SIGNALS`` (co2). Without raw-z, co2's log-power is 100%
    clipped at the +20 ceiling and the stats degenerate (mean 20, std 0) — the pre-fix
    ``codec_co2_perfreq_stats.pt`` on foundation_model_meta is exactly that; REGENERATE
    it with this function, never reuse it.

    WITHIN-SHOT std convention (2026-07-31 v2, measured in gate job 5131926): the
    POOLED-across-shots std is dominated by across-shot regime variance (~5x the
    within-shot signal scale for ece/mhr) and dividing by it crushes any one shot's
    structure below loss visibility — ece/mhr reverted to 1-code decoder-only recon.
    So: ``mean`` = pooled over all sampled windows (dataset-level plate removal, which
    transfers), ``std`` = sqrt(average over shots of the per-shot variance around the
    per-shot mean) — the within-shot scale the gate campaign validated. Samples
    ``windows_per_shot`` windows spread within each shot (deterministic; no cache files
    are read or written). Writes ``{mean, std, std_kind, n_windows, n_shots, modality,
    space}`` to ``out_path`` (existing file is backed up to ``.bak`` first).
    """
    cfg = SpectroCodecConfig()
    cfg.channels = modality_channels(modality)
    # STFT GEOMETRY passthrough (all None => the 1024/256/512/96 default, byte-identical).
    # The written (C, F) stats are only valid for the grid they were computed on.
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
        # STFT grid these stats are valid for — a 512-bin file loaded into a 256-bin codec is
        # a silent correctness bug, so record the geometry alongside the numbers.
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
    """Lazily import the production multi-file dataset + samplers.

    Imported lazily so that ``import ignite.train_codec`` for module-level helpers
    (``modality_channels`` / the CLI parser) does not eagerly pull in the heavier data
    stack; the classes are needed only when a dataset/loader is actually constructed.
    """
    from tokamak_foundation_model.data.multi_file_dataset import (
        DistributedTwoLevelSampler,
        TokamakMultiFileDataset,
        TwoLevelSampler,
    )

    return TokamakMultiFileDataset, TwoLevelSampler, DistributedTwoLevelSampler


# Bind the production dataset + samplers at import time (cheap; the module is already a
# dependency of the trainer). CodecPairDataset subclasses TokamakMultiFileDataset directly.
(
    TokamakMultiFileDataset,
    TwoLevelSampler,
    DistributedTwoLevelSampler,
) = _import_multifile()


# ------------------------------------------------------------------------------------- #
# modality channel-count helper (read-only against the loader config)
# ------------------------------------------------------------------------------------- #
def _channels_after_selection(num_channels: int, channels_to_use) -> int:
    """Channel count after applying a ``channels_to_use`` slice / index-list / None."""
    if channels_to_use is None:
        return num_channels
    if isinstance(channels_to_use, slice):
        return len(range(*channels_to_use.indices(num_channels)))
    return len(list(channels_to_use))


def modality_channels(modality: str) -> int:
    """Channels actually loaded for ``modality`` after ``channels_to_use``.

    Reads ``TokamakH5Dataset.SIGNAL_CONFIGS`` (spectro) or ``MOVIE_CONFIGS`` (video),
    class-level + read-only. Mirrors ``spike._num_ece_channels`` but for any codec modality.
    """
    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

    if modality == FASTTS_MODALITY:
        # fast-TS (filterscopes) is a non-STFT signal; reuse the fast-TS trainer's helper so the
        # channel logic lives in exactly one place (lazy import: fastts_train imports FROM this
        # module, so a top-level import here would be circular).
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
    # slow-TS signals (Thomson / CER / MSE) are non-STFT raw signals; spectro signals are
    # STFT. Anything else that is non-STFT is not a codec modality we support here.
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


# Full-dataset per-channel tangtv liveness scan (scripts/data_preparation/scan_video_channels.py).
# dict {shot_int: [7 flags]}; a flag is 1 when that camera's middle frame is finite. Precomputed
# for ALL 8753 shots, so the video presence filter costs ONE torch.load and opens NO HDF5 file —
# unlike a cold scan, which is the documented NCCL-watchdog crash.
VIDEO_LIVENESS_CACHE = Path(
    "/lustre/orion/fus187/proj-shared/foundation_model_meta/video_channel_liveness.pt"
)


# Full-dataset per-channel SPECTRO liveness scan
# (scripts/data_preparation/scan_video_channels.py --family spectro). Same role and same
# one-torch.load contract as VIDEO_LIVENESS_CACHE: never a cold HDF5 scan at job time.
# dict {shot_int: {modality: {"n", "t0", "t1", "live" [C flags], "live_frac" [C floats]}}}.
SPECTRO_LIVENESS_CACHE = Path(
    "/lustre/orion/fus187/proj-shared/foundation_model_meta/spectro_channel_liveness.pt"
)


def spectro_live_shots(
    modality: str,
    shots: Sequence[Union[str, int]],
    *,
    require_all: bool = False,
    liveness_path: Union[str, Path] = SPECTRO_LIVENESS_CACHE,
    log_fn=None,
) -> List[str]:
    """Keep only the shots whose channels for THIS spectro modality actually recorded.

    The spectro codecs had NO per-channel presence filter at all (only co2 had a whole-shot
    "is the group non-empty" filter, via ``_PRESENCE_FILTER_SIGNALS``). A shot whose HDF5
    group is an empty ``(C, 1)`` stub can never yield a real window — every draw falls
    through ``_draw_valid_pair``'s re-draws to the eps-floor last-resort pair — so those
    shots contribute nothing but a constant plate, which was then reconstructed and shown to
    the discriminator as REAL.

    ``require_all`` additionally drops shots with any dead channel. Order is preserved.
    Falls back to the input list (with a warning) if the liveness cache is missing, so this
    can never harden into a hard dependency.
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
    """The RAW tangtv channel indices this divertor codec consumes (lower [0,2], upper [4,6]).

    Read from ``MOVIE_CONFIGS[modality].channels_to_use`` rather than hard-coded, so the
    presence filter can never disagree with what the loader actually slices.
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
    """Keep only the shots whose cameras for THIS divertor were actually recording.

    The video codec had NO presence filter, and the cost is large and measured (all 8753 shots,
    2026-09-03):

        tangtv_lower (ch 0, 2)   both live 2640 (30.16%)   one 1623 (18.54%)   NEITHER 4490 (51.30%)
        tangtv_upper (ch 4, 6)   both live 1820 (20.79%)   one  930 (10.62%)   NEITHER 6003 (68.58%)

    A shot with no live camera yields a fully-NaN slab that the loader ZERO-FILLS, and
    ``VideoCodecPairDataset._draw_valid_clip`` cannot re-draw out of it (no window in the shot
    has data), so it falls through to the all-zero last-resort clip. Half of the lower stream
    and two thirds of the upper stream was therefore constant zero -- fed to the discriminator
    as "real".

    ``require_all`` additionally drops the one-camera-live shots, leaving only fully-populated
    ones (2640 / 1820 shots). Order is preserved. Falls back to the input list (with a warning)
    if the liveness cache is missing, so this can never harden into a hard dependency.
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


# ------------------------------------------------------------------------------------- #
# δ-shift-pair dataset — a THIN subclass of the production TokamakMultiFileDataset
# ------------------------------------------------------------------------------------- #
def _shot_paths(shots: Sequence[Union[str, int]], data_dir: Union[str, Path]) -> List[Path]:
    """Resolve ``shots`` to existing ``{shot}_processed.h5`` paths under ``data_dir``.

    Missing files are dropped (the parent dataset would give them length 0 anyway); the
    order of the surviving shots is preserved so ranks derive a deterministic file list.
    """
    d = Path(data_dir)
    paths: List[Path] = []
    for s in shots:
        p = d / f"{s}_processed.h5"
        if p.exists():
            paths.append(p)
    return paths


# ------------------------------------------------------------------------------------- #
# activity-stratified re-draw — shared by ALL FOUR codec datasets (anti degenerate-window
# domination). Wraps a per-item "draw one valid item" callable with a biased re-draw toward
# ACTIVE windows so the batch is ~50 % active instead of the natural 35-48 %, WITHOUT dropping
# quiet windows (a below-threshold draw is still ACCEPTED as the fallback, so the codec keeps
# learning a "quiet" code for Phase-B generalization).
# ------------------------------------------------------------------------------------- #
def _stratified_draw(
    idx: int,
    draw_valid,          # callable(chunk_idx) -> item  (already re-draws degenerate windows)
    activity_of,         # callable(item) -> float      (per-modality activity score)
    n_chunks,            # callable() -> int            (re-draw bound, best-effort)
    *,
    min_activity: float,
    active_bias: float,
    max_tries: int,
    seed: int,
):
    """Draw one item for global-window ``idx`` with an activity-stratified re-draw.

    Contract (BYTE-IDENTICAL when ``active_bias <= 0``): returns exactly ``draw_valid(idx)`` and
    performs NO extra RNG / re-draw. This preserves the already-working modalities verbatim.

    When ``active_bias > 0``: draw the base item; if it is already active
    (``activity_of(item) >= min_activity``) keep it. Otherwise, with probability ``active_bias``,
    re-draw from up to ``max_tries`` nearby chunks and take the FIRST active one found. If none of
    the re-draws is active (or the biased coin came up tails), ACCEPT the base item — quiet windows
    are down-weighted, never dropped. All RNG is seeded from ``idx`` so it is deterministic per
    item (reproducible across workers / resumes), matching the degenerate-redraw convention.
    """
    base = draw_valid(idx)
    if active_bias <= 0.0:
        return base                                   # OFF: byte-identical, no extra RNG
    if float(activity_of(base)) >= min_activity:
        return base                                   # already active
    gen = torch.Generator().manual_seed(int(seed) + 90_001 * int(idx) + 13)
    if float(torch.rand((), generator=gen)) >= active_bias:
        return base                                   # biased coin tails -> keep quiet window
    n = max(1, int(n_chunks()))
    for _ in range(int(max_tries)):
        alt = int(torch.randint(0, n, (1,), generator=gen).item())
        cand = draw_valid(alt)
        if float(activity_of(cand)) >= min_activity:
            return cand                               # found an active window
    return base                                       # no active re-draw found -> keep base


def _chunks_in_current_file(ds, chunk_idx: int) -> int:
    """Re-draw bound for hooks whose ``alt`` index is a WITHIN-SHOT chunk index.

    ``_build_window`` / ``_build_clip`` consume the re-draw's ``alt`` as an index into the
    shot currently pinned on ``ds.h5_file`` (``t_start = warmup + alt * step``), so the bound
    MUST be THIS file's chunk count — the parent ``__getitem__`` records it as
    ``_cur_file_n_chunks``. The previous bound (`_cumulative_lengths_span()`, the GLOBAL
    window total) made P(alt lands inside the current shot) ≈ 1e-4 at the 9000-shot
    production scale, so every re-draw candidate was out-of-range → None → the stratified
    AND degenerate re-draws silently no-oped (proven 2026-08-05: the ts_tangential_*
    "stratified" retrains reproduced their unstratified predecessors bit-for-bit, and a
    direct probe measured 0/200 valid candidates). The spectro / fast-TS datasets were
    never affected — they already bound via ``_chunks_in_current_shot`` (xdata-derived).
    Falls back to ``chunk_idx + 1`` (never smaller than the requested chunk).
    """
    n = int(getattr(ds, "_cur_file_n_chunks", 0))
    return n if n > 0 else int(chunk_idx) + 1


class CodecPairDataset(TokamakMultiFileDataset):
    """``(spec_a, spec_b)`` δ-shift log-power pairs for ONE spectro modality.

    A **thin** subclass of :class:`~tokamak_foundation_model.data.multi_file_dataset.\
TokamakMultiFileDataset`. It re-uses the parent's production streaming machinery WHOLESALE
    — the global-idx → ``(file_idx, chunk_idx)`` binary-search map, the per-worker LRU
    HDF5 file-handle cache (with the correct ``__getstate__``/``__setstate__`` pickling into
    DataLoader workers), the length-cache sidecar, and the ``num_workers`` plumbing — and
    overrides ONLY the per-item transform hook :meth:`_getitem_standard`.

    Why ``_getitem_standard`` is the right hook
    -------------------------------------------
    The parent's :meth:`TokamakMultiFileDataset.__getitem__` does, for a global ``idx``:
    (1) a binary-search of the cumulative-length array to get ``pos``; (2) ``file_idx =
    self._valid_indices[pos]`` and ``chunk_idx = idx - cumulative_lengths[pos]``; (3)
    ``self.h5_file = self._get_file_handle(file_idx)`` (per-worker LRU handle); then (4)
    ``return self._getitem_standard(chunk_idx)`` — our override.

    So by overriding *only* ``_getitem_standard`` we inherit the binary-search index map,
    the LRU file-handle acquisition, and the num_workers file-handle safety **verbatim** —
    we do NOT re-implement any of it. When our override runs, ``self.h5_file`` is already the
    correct open handle for this item's shot (owned by this worker's own object copy), and
    ``chunk_idx`` is the within-shot window index.

    Windowing reuse
    ---------------
    The parent's :meth:`TokamakH5Dataset._getitem_standard` computes the window start as
    ``t_start = warmup_s + chunk_idx * step_size_s``. We reuse that EXACT formula (see
    :meth:`_getitem_standard` below). We construct the dataset with
    ``chunk_duration_s = CHUNK_S + span_tail_s`` (the FULL δ-extended span) so the parent's
    length computation (``floor((duration - chunk_duration_s) / step_size_s) + 1``) already
    guarantees the extended window ``[t_start, t_start + CHUNK_S + δ_max]`` fits inside the
    shot's real data for every ``chunk_idx`` — no per-item overrun guard needed for the map.
    ``step_size_s = CHUNK_S`` keeps successive windows 50 ms apart (non-overlapping cores).

    δ-pair construction
    -------------------
    ``_getitem_standard`` loads the RAW extended window for the modality's ``SignalConfig``
    over ``[t_start, t_start + span_s]`` via the parent's
    :meth:`TokamakH5Dataset._load_signal_raw` (already resampled to ``target_fs == STFT_FS``,
    zero-/NaN-padded), then builds the pair with :func:`ignite.data.shift_pair_windows`
    (raw δ-shift + re-STFT to log-power). Degenerate windows (all-NaN / near-flat / past the
    real data end) are guarded exactly as the spike's ``_load_raw_span`` does, and replaced by
    a re-draw from a nearby valid chunk so every item is a real, finite pair.

    Parameters
    ----------
    modality : str
        One of :data:`SPECTRO_MODALITIES`.
    shots : sequence of shot ids (str/int)
        Shots to stream windows from (typically thousands). Resolved to
        ``{shot}_processed.h5`` under ``data_dir``; missing files are dropped.
    cfg : SpectroCodecConfig
        Provides ``freq_bins`` / ``time_frames`` / ``consistency_delta_ms``. ``cfg`` is NOT
        mutated — the caller sets ``cfg.channels`` (use :func:`modality_channels`).
    data_dir : path
        Directory of ``{shot}_processed.h5`` files.
    delta_ms_range : (lo, hi) or None
        δ draw range in ms; defaults to ``cfg.consistency_delta_ms``.
    t0_start : float
        Earliest window start (s), forwarded to the parent as ``warmup_s`` (default 1.0,
        skips ramp-up per the windowing convention).
    min_std : float
        Minimum raw std for a window to count as non-degenerate.
    seed : int
        Base RNG seed; combined with the item ``idx`` so the δ draw / re-draw is
        deterministic per item (reproducible across workers and resumes).
    lengths_cache_path : path or None
        Forwarded to the parent's length-cache sidecar (skip re-scanning shot lengths).
    max_open_files : int
        Forwarded to the parent's per-worker LRU file-handle cap.
    max_tries : int
        Max nearby-chunk re-draws before falling back to the last finite pair for a bad item.
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
        # activity-stratified sampling knobs (0 = OFF -> byte-identical to no stratification).
        self.min_activity = float(getattr(cfg, "min_activity", 0.0))
        self.active_bias = float(getattr(cfg, "active_bias", 0.0))
        # MISSING-DATA knobs (2026-09-03). See SpectroCodecConfig.mask_missing for the
        # measured motivation.
        #
        # ITEM ARITY IS A CONSTRUCTOR ARGUMENT, NOT A cfg FIELD, and deliberately so. The
        # trainer opts in with emit_mask=cfg.mask_missing; every OTHER consumer of this
        # dataset builds it from a checkpoint's pickled cfg — the audit / figure scripts,
        # train_dynamics._single_shot_dataset (the frame-code cache), compute_logpow_stats —
        # and those all unpack a 2-tuple. Keying the arity off the cfg would silently change
        # what they receive the moment a masked codec's checkpoint is loaded, i.e. it would
        # break the cache rebuild rather than the trainer. Default False = the 2-tuple this
        # dataset has always returned, and no mask is even built.
        self.emit_mask = bool(emit_mask)
        self.require_live_channels = bool(getattr(cfg, "require_live_channels", False))
        # A fully-missing window is rejected only when someone is actually consuming the mask.
        self.mask_missing = self.emit_mask

        # δ-extended span the STFT pair needs: A covers [t0, t0+CHUNK_S]; B is shifted by up
        # to δ_max, ending at t0 + CHUNK_S + δ_max. Make the parent reserve room for the WHOLE
        # span by using it as chunk_duration_s, while step_size_s stays at CHUNK_S so windows
        # advance 50 ms at a time (non-overlapping cores).
        self.span_tail_s = float(self.delta_ms_range[1]) / 1000.0
        span_s = CHUNK_S + self.span_tail_s

        paths = _shot_paths(shots, data_dir)
        if not paths:
            raise ValueError(
                f"CodecPairDataset: no {{shot}}_processed.h5 files found under {data_dir} "
                f"for the {len(list(shots))} requested shots."
            )

        # Reuse the parent's full production machinery: idx-map + LRU handles + length cache.
        # chunk_duration_s = full δ-span so windows fit; step_size_s = CHUNK_S; warmup_s =
        # t0_start (the parent's _getitem_standard uses warmup_s + idx*step_size_s as t_start).
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
        # The parent's __getitem__ dispatches on self.signal_configs; keep only ours so
        # _load_signal_raw / duration reflect the codec modality (harmless for span math).
        self._cfg_sig = next(c for c in self.signal_configs if c.name == modality)

    # -- the ONLY overridden hook: per-item δ-pair transform -------------------------- #
    def _getitem_standard(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Return one ``(spec_a, spec_b)`` δ-shift log-power pair for window ``idx``.

        Called by the parent's ``__getitem__`` AFTER it has (a) binary-search-mapped the
        global index to ``(file_idx, chunk_idx)`` and (b) set ``self.h5_file`` to this shot's
        per-worker LRU handle. Here ``idx`` is the within-shot ``chunk_idx``.

        Reuses the parent's windowing formula (``t_start = warmup_s + idx * step_size_s``)
        and the parent's :meth:`_load_signal_raw` to pull the RAW δ-extended window, then
        builds the pair via :func:`ignite.data.shift_pair_windows`.

        With ``cfg.active_bias > 0`` an activity-stratified re-draw (:func:`_stratified_draw`)
        biases the draw toward log-power windows with std ``>= cfg.min_activity`` (co2 is mostly
        floored). With ``active_bias == 0`` (ece/bes/mhr default) this is byte-identical to the
        plain build+degenerate-redraw below.
        """
        item = _stratified_draw(
            idx, self._draw_valid_pair, lambda p: float(p[0].std()),
            lambda: self._chunks_in_current_shot(idx),
            min_activity=self.min_activity, active_bias=self.active_bias,
            max_tries=self.max_tries, seed=self.pair_seed,
        )
        # ARITY CONTRACT (see __init__): 2-tuple unless the CONSTRUCTOR asked for the mask.
        # With emit_mask=True the third element is the ``(C, cfg.time_frames)``
        # per-(channel, STFT-frame) validity mask for ``spec_a``.
        return item if self.emit_mask else (item[0], item[1])

    def _draw_valid_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the δ-pair for window ``idx``, re-drawing DEGENERATE windows (the prior path)."""
        pair = self._build_pair(idx)
        if pair is not None:
            return pair
        # Degenerate window (padded tail / all-flat / NaN): re-draw from nearby chunks so we
        # never return garbage. Bounded to this shot's chunk range.
        n_chunks = self._chunks_in_current_shot(idx)
        gen = torch.Generator().manual_seed(self.pair_seed + 100_003 * int(idx) + 7)
        for _ in range(self.max_tries):
            alt = int(torch.randint(0, max(1, n_chunks), (1,), generator=gen).item())
            pair = self._build_pair(alt)
            if pair is not None:
                return pair
        # Last resort: an eps-floor "silent" pair (finite, non-NaN). NOT rare: a shot whose
        # HDF5 group is an empty (C, 1) stub — measured in 100% of a 60-shot 190xxx sample for
        # bes and co2 — can NEVER produce a real window, so every draw from it lands here.
        # This constant plate was being reconstructed, discriminated as REAL and counted in the
        # FSQ entropy statistic: exactly the video failure. Its mask is ALL-INVALID.
        C = self._num_channels()
        floor = float(math.log10(1e-10))
        z = torch.full((C, self.codec_cfg.freq_bins, self.codec_cfg.time_frames), floor)
        return z, z.clone(), torch.zeros((C, self.codec_cfg.time_frames))

    # -- helpers (all reuse parent state; no re-implementation of the index map) ------- #
    def _build_pair(self, chunk_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Build the δ-pair for a within-shot ``chunk_idx``; None if the window is degenerate.

        ``t_start`` reuses the parent's EXACT formula. The extended span is loaded via the
        parent's ``_load_signal_raw`` (already at ``target_fs == STFT_FS``); guards mirror
        :func:`ignite.spike._load_raw_span`.
        """
        step = getattr(self, "step_size_s", self.chunk_duration_s)
        warmup = getattr(self, "warmup_s", 0.0)
        t_start = warmup + chunk_idx * step          # parent windowing, reused verbatim
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
        # NOTE the std guard is GLOBAL over ALL channels, which is exactly why a per-channel
        # mask is needed: a 40-channel ece window with 12 zero-slab channels still has a large
        # global std and sails through. `nan_mask` used to be DISCARDED here (bound to `_nan`).

        lo, hi = self.delta_ms_range
        gen = torch.Generator().manual_seed(self.pair_seed + 1_000_003 * int(chunk_idx) + 1)
        d = float(lo + (hi - lo) * float(torch.rand((), generator=gen)))
        # `raw` already starts at t_start -> rebase t0 to 0.0 for the pair slicer.
        spec_a, spec_b = data.shift_pair_windows(
            raw, t0=0.0, cfg=self.codec_cfg, delta_ms=d
        )
        # Per-(channel, STFT-frame) validity for window A (the reconstruction target), built
        # from the SAME raw samples A is STFT'd from: `raw` starts at t_start and A spans
        # [t_start, t_start + CHUNK_S] = the first cfg.window_samples samples.
        mask = data.spectro_frame_mask(
            raw[:, : self.codec_cfg.window_samples],
            nan_mask[:, : self.codec_cfg.window_samples],
            self.codec_cfg,
        )
        # `require_live_channels`: reject (-> re-draw) any window with a dead channel so the
        # codec only ever sees fully-populated windows. OFF by default.
        if self.require_live_channels and float(mask.min()) <= 0.0:
            return None
        if self.mask_missing and float(mask.sum()) <= 0.0:
            return None                              # nothing real in this window at all
        return spec_a, spec_b, mask

    def _chunks_in_current_shot(self, chunk_idx: int) -> int:
        """Number of chunks in the shot currently pinned on ``self.h5_file`` (best-effort).

        Used only to bound the re-draw range. Falls back to ``chunk_idx + 1`` if the shot
        length can't be recovered (never smaller than the requested chunk).
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
    """Stack ``(spec_a, spec_b[, mask])`` items into ``(B, C, F, T)`` batched tensors.

    Arity follows the ITEM: a 2-tuple for the legacy (unmasked) dataset — byte-identical —
    and a 3-tuple ``(a, b, (B, C, T) mask)`` when the dataset carries ``cfg.mask_missing``.
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
    """Build a δ-pair DataLoader over a :class:`CodecPairDataset`, reusing the parent's
    file-locality sampler.

    Sampler choice mirrors the main foundation-model trainer:
      * DDP (``world_size > 1``): :class:`DistributedTwoLevelSampler` — file-level sharding
        with sequential intra-file iteration, which keeps the parent's per-worker LRU
        file-handle cache warm (the DDP analogue of ``DistributedSampler`` tuned for this
        dataset; see its docstring).
      * single-process: :class:`TwoLevelSampler` — the same locality-preserving sampler
        ``make_dataloader`` uses.

    The collate stacks ``(spec_a, spec_b)`` items into ``(B, C, F, T)`` batches (the codec's
    δ-pair contract), rather than the parent's multi-modal ``collate_fn``.
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
    """Yield batches from ``loader`` forever, re-iterating (a new epoch) when exhausted.

    The production dataset/loader is a FINITE map-style epoch (one pass over the shot
    windows), while the trainer runs a fixed ``steps`` budget that may span many epochs. This
    turns the finite loader into an endless batch stream. ``set_epoch`` is called on the
    sampler each epoch (mirrors the main trainer) so DDP shuffling advances across epochs.
    """
    sampler = getattr(loader, "sampler", None)
    epoch = 0
    while True:
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


# ===================================================================================== #
# VIDEO (tangtv) codec — dataset + loader + gate, mirroring the spectro machinery above.
# ===================================================================================== #
class VideoCodecPairDataset(TokamakMultiFileDataset):
    """tangtv frame windows ``(frames (C,T,H,W), frame_mask (T,))`` for ONE divertor.

    A **thin** subclass of :class:`~tokamak_foundation_model.data.multi_file_dataset.\
TokamakMultiFileDataset`, EXACTLY as :class:`CodecPairDataset` is for spectro. It reuses the
    parent's production streaming machinery WHOLESALE — the global-idx → ``(file_idx,
    chunk_idx)`` binary-search map, the per-worker LRU HDF5 file-handle cache (with the
    ``__getstate__``/``__setstate__`` pickling into DataLoader workers), the length-cache
    sidecar, and the ``num_workers`` plumbing — and overrides ONLY the per-item transform hook
    :meth:`_getitem_standard`. It does **not** touch ``data_loader.py`` /
    ``multi_file_dataset.py``.

    Difference from :class:`CodecPairDataset` (spectro)
    ---------------------------------------------------
    The video codec has **no δ-shift consistency pair** (docs/IGNITE_DESIGN.md §4.3; video
    has no STFT-phase realization nuisance). So the transform returns the video **frames**
    plus a **per-frame validity mask** — NOT a pair. The parent's ``__getitem__`` still does
    the binary-search index map and sets ``self.h5_file`` before dispatching to our
    ``_getitem_standard(chunk_idx)``; here ``chunk_idx`` is the within-shot window index and
    ``self.h5_file`` is this worker's LRU handle. We reuse the parent's windowing formula
    (``t_start = warmup_s + chunk_idx * step_size_s``) and the parent's
    :meth:`TokamakH5Dataset._load_movie_raw` (already resampled to ``target_fps`` and
    zero-filled where the camera was off / NaN) — the video analogue of the spectro dataset's
    ``_load_signal_raw`` reuse.

    Parameters
    ----------
    modality : str
        One of :data:`VIDEO_MODALITIES` (``tangtv_lower`` / ``tangtv_upper``).
    shots, cfg, data_dir, seed, lengths_cache_path, max_open_files, max_tries, t0_start,
    min_std : as :class:`CodecPairDataset` (``cfg`` here is a :class:`VideoCodecConfig`).
        ``cfg`` is NOT mutated. ``min_std`` guards degenerate (flat / dead-camera) windows.
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
        # activity-stratified sampling knobs (0 = OFF -> byte-identical to no stratification).
        self.min_activity = float(getattr(cfg, "min_activity", 0.0))
        self.active_bias = float(getattr(cfg, "active_bias", 0.0))
        # MISSING-DATA knobs (both default False => byte-identical to the previous dataset).
        self.mask_missing = bool(getattr(cfg, "mask_missing", False))
        self.require_live_channels = bool(getattr(cfg, "require_live_channels", False))

        paths = _shot_paths(shots, data_dir)
        if not paths:
            raise ValueError(
                f"VideoCodecPairDataset: no {{shot}}_processed.h5 files found under "
                f"{data_dir} for the {len(list(shots))} requested shots."
            )

        # A video window is a single CHUNK_S (50 ms) clip: chunk_duration_s = step_size_s =
        # CHUNK_S (non-overlapping 50 ms windows), warmup_s = t0_start. The parent's length
        # formula floor((duration - chunk_duration_s)/step_size_s)+1 guarantees each window
        # fits inside the shot's real data. We request the movie by NAME so the parent loads
        # THIS divertor's channel set only.
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

    # -- the ONLY overridden hook: per-item frame-window transform -------------------- #
    def _getitem_standard(self, idx: int):  # type: ignore[override]
        """Return one ``(frames (C,T,H,W), frame_mask (T,))`` for window ``idx``.

        Called by the parent's ``__getitem__`` AFTER it mapped the global index to
        ``(file_idx, chunk_idx)`` and set ``self.h5_file``. ``idx`` here is the within-shot
        ``chunk_idx``. Reuses the parent's windowing formula + :meth:`_load_movie_raw`.

        With ``cfg.active_bias > 0`` an activity-stratified re-draw biases toward clips with
        frame std ``>= cfg.min_activity``. In production the trainer leaves ``active_bias == 0``
        for tangtv (its collapse is adversarial, not degenerate-window — see the config note), so
        this is byte-identical to the plain build+degenerate-redraw below.
        """
        return _stratified_draw(
            idx, self._draw_valid_clip, lambda c: float(c[0].std()),
            lambda: _chunks_in_current_file(self, idx),
            min_activity=self.min_activity, active_bias=self.active_bias,
            max_tries=self.max_tries, seed=self.pair_seed,
        )

    def _draw_valid_clip(self, idx: int):
        """Build the clip for window ``idx``, re-drawing DEGENERATE windows (the prior path)."""
        clip = self._build_clip(idx)
        if clip is not None:
            return clip
        # Degenerate window (dead camera / all-flat): re-draw from nearby chunks WITHIN this
        # shot (alt is a within-shot index — see _chunks_in_current_file).
        n_chunks = max(1, _chunks_in_current_file(self, idx))
        gen = torch.Generator().manual_seed(self.pair_seed + 100_003 * int(idx) + 7)
        for _ in range(self.max_tries):
            alt = int(torch.randint(0, n_chunks, (1,), generator=gen).item())
            clip = self._build_clip(alt)
            if clip is not None:
                return clip
        # Last resort: a finite all-zero clip (rare; whole shot degenerate) + all-invalid mask.
        cfg = self.codec_cfg
        z = torch.zeros((cfg.channels, cfg.frames, cfg.height, cfg.width))
        m = (torch.zeros((cfg.channels, cfg.frames), dtype=torch.float32)
             if self.mask_missing else torch.zeros(cfg.frames, dtype=torch.float32))
        return z, m

    # -- helpers (reuse parent state; no re-implementation of the index map) ---------- #
    def _build_clip(self, chunk_idx: int):
        """Build ``(frames, frame_mask)`` for a within-shot ``chunk_idx``; None if degenerate.

        ``t_start`` reuses the parent's EXACT windowing formula; the frames come from the
        parent's ``_load_movie_raw`` (already resampled to ``target_fps``, NaN→0 filled).
        A per-CHANNEL availability mask is returned by ``_load_movie_raw``; we fold it into a
        per-FRAME validity mask (a frame is valid iff ANY channel is live and the frame is
        finite + non-flat).
        """
        cfg = self.codec_cfg
        step = getattr(self, "step_size_s", self.chunk_duration_s)
        warmup = getattr(self, "warmup_s", 0.0)
        t_start = warmup + chunk_idx * step          # parent windowing, reused verbatim
        t_end = t_start + CHUNK_S

        frames, channel_valid = self._load_movie_raw(
            self.h5_file, self._cfg_movie, t_start, t_end
        )
        # frames: (C, T, H, W) at target_fps. Crop/pad T to cfg.frames defensively (a 50 ms
        # window at target_fps should already be cfg.frames; guard rounding drift).
        frames = self._fit_frames(frames)
        if not torch.isfinite(frames).all():
            return None
        if float(frames.std()) < self.min_std:
            return None                              # dead camera / flat window
        # per-frame validity: any live channel AND finite. channel_valid is (C,) bool; a frame
        # is valid if at least one channel is live (the loader zero-fills off cameras).
        any_live = bool(channel_valid.any().item())
        if not any_live:
            return None
        # `require_live_channels`: a clip with ANY dead camera is rejected (and re-drawn), so
        # the codec only ever sees fully-populated windows. MEASURED motivation: among the
        # tangtv_lower shots that have at least one live camera, 19.0% of the channel-slots
        # are still a zero slab (tangtv_upper 16.9%) -- see VideoCodecConfig.mask_missing.
        if self.require_live_channels and not bool(channel_valid.all().item()):
            return None
        if not self.mask_missing:
            # LEGACY (default): an unconditional all-ones per-frame mask. Kept so the default
            # path is byte-identical; note this mask excluded NOTHING, which is precisely the
            # 2026-09-03 audit finding.
            frame_mask = torch.ones(cfg.frames, dtype=torch.float32)
            return frames, frame_mask
        # REAL per-(C, T) validity: a (channel, frame) entry is valid iff the loader flagged
        # that camera live for this window AND the frame is finite. `channel_valid` may be
        # shorter than cfg.channels only in tests that shrink the grid; broadcast defensively.
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
        """Crop/pad the loaded ``(C, T, H, W)`` to exactly ``(cfg.channels, cfg.frames, H, W)``.

        ``_load_movie_raw`` already resamples H×W to its ``MovieConfig`` target (120×360 for
        tangtv, == the production ``VideoCodecConfig`` H/W), and a 50 ms window at target_fps
        is already ``cfg.frames``. This guard makes the codec ``cfg`` authoritative for the
        WHOLE ``(T, H, W)`` shape — the time axis can drift a frame from rounding, and H/W is
        trilinearly resized only if it differs (a no-op in production; lets tests use a smaller
        grid). Time is right-cropped / right-padded (repeat last frame); H/W is resized via
        ``F.interpolate`` (the same trilinear-family resample the loader itself uses).
        """
        cfg = self.codec_cfg
        C, T, H, W = frames.shape
        # spatial fit (usually a no-op: loader already at cfg.height × cfg.width).
        if (H, W) != (cfg.height, cfg.width):
            frames = torch.nn.functional.interpolate(
                frames.unsqueeze(0), size=(T, cfg.height, cfg.width),
                mode="trilinear", align_corners=False,
            ).squeeze(0)
            C, T = frames.shape[0], frames.shape[1]
        # time fit.
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

    Identical sampler/plumbing choice as :func:`make_codec_loader` (the file-locality
    ``DistributedTwoLevelSampler`` under DDP, ``TwoLevelSampler`` single-process), only the
    collate differs (``(frames, mask)`` instead of ``(spec_a, spec_b)``).
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


# ===================================================================================== #
# SLOW-TS (Thomson / CER / MSE) codec — dataset + loader + gate, mirroring the machinery
# above. The "lightest touch" family (§4.3): masked reconstruction + entropy, NO δ-pair,
# NO discriminator.
# ===================================================================================== #
def slowts_codec_cfg(signal: str, channels: int) -> SlowTSCodecConfig:
    """Build a :class:`SlowTSCodecConfig` for ``signal`` with the loader's real channel count.

    ``channels`` is the number of profile positions actually loaded (from
    :func:`modality_channels`). The patch sizes come from :func:`slowts_patch_for` (4 radial-zone
    tokens per window by default; ``patch_c = ceil(channels / 4)``). Kept as a helper so the
    trainer + CLI + tests build the cfg the same way.
    """
    patch_c, patch_t = slowts_patch_for(channels)
    return SlowTSCodecConfig(
        signal=signal, channels=channels, patch_c=patch_c, patch_t=patch_t, n_zones=4
    )


def load_slowts_channel_stats(
    signal: str,
    stats_path: Optional[Union[str, Path]] = None,
) -> Tuple[str, List[float], List[float]]:
    """Per-signal preprocessing method + per-channel mean/std for ``signal`` (the SCALE FIX).

    The slow-TS analogue of :func:`fastts_train.load_fastts_channel_stats`. Reads the SAME
    ``preprocessing_stats.pt`` the data_loader consumes and returns
    ``(method, mean_list, std_list)`` where:

      * ``method`` is the FM model's preprocessing for this signal
        (:data:`config.SLOWTS_PREPROCESS_METHOD` — a READ of the loader's
        ``SignalConfig.preprocess.method``: ``"log_standardize"`` for the 4 Thomson signals,
        ``"standardize"`` for cer_ti/cer_rot/mse).
      * ``mean`` / ``std`` are the per-channel stats from the sub-dict the loader would use for
        that method — the ``'log'`` sub-dict for ``log_standardize``, the ``'raw'`` sub-dict for
        ``standardize`` — mirroring ``data_loader._update_preprocessing_stats``. NaNs are mapped to
        mean 0 / std 1 exactly like the loader; length == the signal's ``C`` channels (none of the
        7 slow-TS signals use ``channels_to_use``, so no slicing is needed — asserted here).

    DATA-pipeline reuse (stats + the loader's standardize MECHANISM); no FAITH model code is
    imported. Raises if the signal / stats are absent so a mis-pointed ``--stats_path`` fails loud
    instead of silently skipping the fix. ``stats_path=None`` => the canonical
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
    mean[~np.isfinite(mean)] = 0.0   # mirror _update_preprocessing_stats NaN handling (+ inf guard)
    std[~np.isfinite(std)] = 1.0
    return method, mean.astype(np.float32).tolist(), std.astype(np.float32).tolist()


class SlowTSCodecPairDataset(TokamakMultiFileDataset):
    """slow-TS windows ``(signal (C,T), mask (C,T))`` for ONE signal (Thomson / CER / MSE).

    A **thin** subclass of :class:`~tokamak_foundation_model.data.multi_file_dataset.\
TokamakMultiFileDataset`, EXACTLY as :class:`CodecPairDataset` (spectro) and
    :class:`VideoCodecPairDataset` (video) are. It reuses the parent's production streaming
    machinery WHOLESALE — the global-idx → ``(file_idx, chunk_idx)`` binary-search map, the
    per-worker LRU HDF5 file-handle cache (with the ``__getstate__``/``__setstate__`` pickling
    into DataLoader workers), the length-cache sidecar, and the ``num_workers`` plumbing — and
    overrides ONLY the per-item transform hook :meth:`_getitem_standard`. It does **not** touch
    ``data_loader.py`` / ``multi_file_dataset.py``.

    Difference from the spectro / video datasets
    ---------------------------------------------
    The slow-TS codec has NO δ-shift consistency pair (§4.3; no realization nuisance) and NO
    adversarial term. So the transform returns the slow-TS **window** ``(C, T)`` plus a
    per-sample **validity mask** ``(C, T)`` — NOT a pair. The parent's ``__getitem__`` still
    does the binary-search index map and sets ``self.h5_file`` before dispatching to our
    ``_getitem_standard(chunk_idx)``; here ``chunk_idx`` is the within-shot window index and
    ``self.h5_file`` is this worker's LRU handle. We reuse the parent's windowing formula
    (``t_start = warmup_s + chunk_idx * step_size_s``) and the parent's
    :meth:`TokamakH5Dataset._load_signal_raw` (already resampled to ``target_fs`` == 100 Hz,
    NaN→0 filled, with the raw NaN mask returned) — the slow-TS analogue of the other datasets'
    ``_load_signal_raw`` / ``_load_movie_raw`` reuse.

    Missingness / mask contract
    ---------------------------
    ``_load_signal_raw`` returns ``(raw (C,T), valid_len, nan_mask (C,T))`` where ``nan_mask``
    is 1.0 where the raw HDF5 value was NaN. We derive the per-sample VALIDITY mask (1 = real)
    EXACTLY as ``data_loader.TokamakH5Dataset._process_signal`` would for this signal:
      * ``zero_is_missing`` signals (Thomson): valid = (raw != 0) AND (nan_mask == 0).
      * otherwise (CER / MSE): valid = (nan_mask == 0).
    This mirrors the loader's own ``element_mask = data != 0.0`` (zero_is_missing) vs the NaN
    mask, so the codec's masked reconstruction never trains toward the zero/NaN-fill of a
    missing sample.

    Parameters
    ----------
    signal : str
        One of :data:`SLOWTS_MODALITIES`.
    shots, cfg, data_dir, seed, lengths_cache_path, max_open_files, max_tries, t0_start :
        as :class:`CodecPairDataset` (``cfg`` here is a :class:`SlowTSCodecConfig`). ``cfg`` is
        NOT mutated.
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
        # activity-stratified sampling knobs (0 = OFF -> byte-identical to no stratification).
        # For slow-TS (a MASKED modality) "activity" is the window's PRESENT-FRACTION.
        self.min_activity = float(getattr(cfg, "min_activity", 0.0))
        self.active_bias = float(getattr(cfg, "active_bias", 0.0))

        paths = _shot_paths(shots, data_dir)
        if not paths:
            raise ValueError(
                f"SlowTSCodecPairDataset: no {{shot}}_processed.h5 files found under "
                f"{data_dir} for the {len(list(shots))} requested shots."
            )

        # A slow-TS window is a single CHUNK_S (50 ms) clip: chunk_duration_s = step_size_s =
        # CHUNK_S (non-overlapping 50 ms windows), warmup_s = t0_start. The parent's length
        # formula floor((duration - chunk_duration_s)/step_size_s)+1 guarantees each window
        # fits inside the shot's real data. We request the signal by NAME so the parent's
        # config reflects this modality.
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

    # -- the ONLY overridden hook: per-item window + validity-mask transform ------------ #
    def _getitem_standard(self, idx: int):  # type: ignore[override]
        """Return one ``(signal (C,T), mask (C,T))`` for window ``idx``.

        Called by the parent's ``__getitem__`` AFTER it mapped the global index to
        ``(file_idx, chunk_idx)`` and set ``self.h5_file``. ``idx`` here is the within-shot
        ``chunk_idx``. Reuses the parent's windowing formula + :meth:`_load_signal_raw`.

        With ``cfg.active_bias > 0`` an activity-stratified re-draw biases toward windows whose
        PRESENT-FRACTION (``mask.mean()``) is ``>= cfg.min_activity`` (ts_core_density is mostly
        near-empty). With ``active_bias == 0`` (default) this is byte-identical to the plain
        build+degenerate-redraw below.
        """
        return _stratified_draw(
            idx, self._draw_valid_window, lambda it: float(it[1].mean()),
            lambda: _chunks_in_current_file(self, idx),
            min_activity=self.min_activity, active_bias=self.active_bias,
            max_tries=self.max_tries, seed=self.item_seed,
        )

    def _draw_valid_window(self, idx: int):
        """Build the window for ``idx``, re-drawing DEGENERATE windows (the prior path)."""
        item = self._build_window(idx)
        if item is not None:
            return item
        # Degenerate window (all-missing / padded tail): re-draw from nearby chunks WITHIN
        # this shot (alt is a within-shot index — see _chunks_in_current_file).
        n_chunks = max(1, _chunks_in_current_file(self, idx))
        gen = torch.Generator().manual_seed(self.item_seed + 100_003 * int(idx) + 7)
        for _ in range(self.max_tries):
            alt = int(torch.randint(0, n_chunks, (1,), generator=gen).item())
            item = self._build_window(alt)
            if item is not None:
                return item
        # Last resort: a finite all-zero window + all-INVALID mask (rare; whole shot missing).
        # Shaped to padded_channels (the tensor the encoder patchifies) so collation is uniform.
        cfg = self.codec_cfg
        z = torch.zeros((cfg.padded_channels, cfg.time_steps))
        m = torch.zeros((cfg.padded_channels, cfg.time_steps), dtype=torch.float32)
        return z, m

    # -- helpers (reuse parent state; no re-implementation of the index map) ------------ #
    def _build_window(self, chunk_idx: int):
        """Build ``(signal (C,T), mask (C,T))`` for a within-shot ``chunk_idx``; None if bad.

        ``t_start`` reuses the parent's EXACT windowing formula; the raw window + NaN mask come
        from the parent's ``_load_signal_raw`` (already resampled to ``target_fs`` == 100 Hz).
        The validity mask mirrors ``_process_signal``'s missingness policy for this signal
        (see the class docstring). A window with NO valid samples at all is rejected (None) so
        the codec never trains on a fully-missing window.
        """
        cfg = self.codec_cfg
        step = getattr(self, "step_size_s", self.chunk_duration_s)
        warmup = getattr(self, "warmup_s", 0.0)
        t_start = warmup + chunk_idx * step          # parent windowing, reused verbatim
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
            return None                              # whole window missing -> reject
        return signal, mask

    def _fit_window(self, raw: torch.Tensor, nan_mask: torch.Tensor):
        """Crop/pad ``raw`` to ``(cfg.padded_channels, cfg.time_steps)`` + build the validity mask.

        ``_load_signal_raw`` returns ``(C, T)`` at ``target_fs`` (T == cfg.time_steps for a
        50 ms window at 100 Hz); this guard makes the codec ``cfg`` authoritative for the whole
        shape. The REAL ``cfg.channels`` profile positions are standardized + missingness-masked
        first (so the mask + per-channel stats stay aligned to the real profile), THEN the profile
        is padded up to ``cfg.padded_channels = n_zones * patch_c`` radial-zone-aligned positions
        with the pad tail MASKED as missing. The validity mask is built from the loader's
        missingness policy for this signal (see the class docstring).
        """
        cfg = self.codec_cfg
        C, T = raw.shape
        # time fit (usually a no-op).
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
        # channel fit to the REAL profile count (defensive; loader already yields cfg.channels).
        # The radial-zone padding to cfg.padded_channels happens AFTER standardization below, so
        # the standardize stats + missingness mask stay aligned to the real cfg.channels profile.
        if C != cfg.channels:
            raw = raw[: cfg.channels]
            nan_mask = nan_mask[: cfg.channels]
            if raw.shape[0] < cfg.channels:
                pad = torch.zeros((cfg.channels - raw.shape[0], cfg.time_steps))
                raw = torch.cat([raw, pad], dim=0)
                nan_mask = torch.cat([nan_mask, torch.ones_like(pad)], dim=0)

        # validity mask: 1 = real sample. Mirror _process_signal's missingness policy.
        # BUILT FROM THE RAW WINDOW (before standardization) so the missingness semantics are
        # identical to the FM's — `nan_mask` is the loader's raw-NaN mask and `raw != 0` is the
        # zero_is_missing policy on the RAW value (standardizing would move a real 0 off zero and
        # a missing 0 to a nonzero standardized value, so the mask MUST come from raw first).
        valid = nan_mask < 0.5                       # NaN mask 1.0 == NaN -> invalid
        if cfg.zero_is_missing:                      # Thomson: a 0 is a missing sample.
            valid = valid & (raw != 0.0)
        # Non-finite raw is masked INVALID, exactly like a missing position: excluded from the
        # loss (valid=0) and zeroed in the input below. mse has ~2 of 69 always-garbage channels
        # (inf/nan raw -> inf/nan stats); previously ANY non-finite position made _getitem_standard
        # REJECT the whole window, so mse windows touching those channels were dropped and the few
        # that leaked through poisoned the recon -> the +0.55/-0.52 decode-corr oscillation.
        valid = valid & torch.isfinite(raw)

        # SCALE FIX — standardize the codec input the SAME way the FM model does (see
        # SlowTSCodecConfig.channel_mean/std + SLOWTS_PREPROCESS_METHOD). Applied AFTER the mask
        # is built (mask semantics unchanged) and BEFORE the zone padding (stats are length
        # cfg.channels). No-op / byte-identical when the cfg carries no stats (default) or method
        # is "none"/None.
        signal = self._standardize(raw)
        # Standardizing an inf/nan raw position yields inf/nan; neutralize to 0 (== the
        # standardized mean) so the input-zeroing `signal * valid` below can't produce inf*0 = NaN
        # (which would trip the finite-check in _getitem_standard and drop the window). The
        # position is already masked invalid above, so 0 is the correct neutral input + loss-excluded.
        signal = torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        valid = valid.to(torch.float32)

        # INPUT-MASK FIX (Bug B) — after standardization a MISSING position (raw 0) maps to
        # (pp(0) - mean)/std, a LARGE-NEGATIVE artifact (~-25 for ts_core_density). The loss is
        # already masked (missing excluded via `valid`), but the ENCODER still processes the input
        # — and with a ⅔-missing signal like ts_core_density the input is dominated by that -25,
        # which collapses the codec. So zero the missing positions in the INPUT the encoder sees
        # (0 == the standardized neutral/mean value, NOT a large artifact). The recon loss still
        # excludes them via `valid`, so missing stays out of the loss. No-op when everything is
        # present (valid all 1) — byte-identical for fully-present windows.
        if self._standardize_active():
            signal = signal * valid

        # RADIAL-ZONE PADDING — pad the real profile up to cfg.padded_channels = n_zones*patch_c
        # so it splits into EXACTLY n_zones contiguous zones (=> n_zones tokens). The pad tail is
        # MASKED as missing (valid=0) so it never contributes to the loss, and zero-filled in the
        # input (0 == the standardized neutral value) so it never contributes to the encoder —
        # the SAME treatment as a genuine missing position. No-op when channels == padded_channels.
        signal, valid = self._pad_zone_tail(signal, valid)
        return signal, valid

    def _pad_zone_tail(self, signal: torch.Tensor, valid: torch.Tensor):
        """Pad ``(C, T) -> (padded_channels, T)`` with a missing (valid=0), zero-input tail.

        ``signal`` / ``valid`` are the real ``cfg.channels`` profile (already standardized +
        missingness-masked). Appends ``padded_channels - channels`` radial-zone padding rows that
        are input-zeroed (neutral, seen by the encoder as absent) and mask-invalid (excluded from
        the loss) — identical to a genuine missing position. No-op when no padding is needed.
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
        """True iff :meth:`_standardize` actually rescales (stats present + a real method).

        Gates the INPUT-mask zeroing to the standardized path ONLY: without standardization the
        raw missing fill is already 0 (Thomson zeros) or the pre-fix path we must not perturb, so
        multiplying by the mask would be a behaviour change for stat-less callers. With
        standardization, missing zeros have been moved to a large-negative artifact, so re-zeroing
        them is the fix. Byte-identical to the pre-fix path when this is False.
        """
        cfg = self.codec_cfg
        method = getattr(cfg, "preprocess_method", None)
        return (
            method not in (None, "none")
            and cfg.channel_mean is not None
            and cfg.channel_std is not None
        )

    def _standardize(self, raw: torch.Tensor) -> torch.Tensor:
        """Per-channel standardize ``raw`` (C,T) EXACTLY as ``data_loader._apply_preprocessing``.

        Mirrors the loader's ``method="standardize"`` / ``method="log_standardize"`` branches for
        this signal — same clip/log/(x-mean)/std math, same ``std.clamp(min=1e-3)``, same LOG-space
        vs RAW-space stats selection — so the codec sees the O(1) input the FM model consumes. The
        SCALE FIX for the ~1e19 Thomson density signals. GLOBAL / per-channel (broadcast over time),
        NOT per-window, so the relative profile LEVEL is preserved.

        ``cfg.channel_mean``/``cfg.channel_std`` being ``None`` OR ``preprocess_method`` in
        ``(None, "none")`` is the IDENTITY (no-op, byte-identical to the pre-fix path). Stats of
        length != ``C`` are rejected loud (a mis-pointed stats file must fail, not silently skip).
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
            # data_loader: arr = clip(x, min=-0.99); arr += 1; log10(arr, out=arr). Then standardize
            # with the LOG-space per-channel mean/std ('log' sub-dict of preprocessing_stats.pt).
            x = torch.log10(x.clamp(min=-0.99) + 1.0)
        mean = mean.reshape(C, 1)
        std = std.reshape(C, 1).clamp(min=1e-3)      # matches data_loader std.clamp(min=1e-3)
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

    Identical sampler/plumbing choice as :func:`make_codec_loader` / :func:`make_video_loader`
    (the file-locality ``DistributedTwoLevelSampler`` under DDP, ``TwoLevelSampler``
    single-process), only the collate differs (``(signal, mask)``).
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


# ------------------------------------------------------------------------------------- #
# slow-TS stability nuisance + gate (mirrors spike.compute_gate for slow-TS)
# ------------------------------------------------------------------------------------- #
def _nanmean(vals: Sequence[float]) -> float:
    """Mean of ``vals`` ignoring NaNs (NaN if all are NaN) — no numpy import needed here."""
    good = [float(v) for v in vals if float(v) == float(v)]
    return float(sum(good) / len(good)) if good else float("nan")


def slowts_nuisance(x: torch.Tensor, *, jitter: float = 0.02, seed: int = 0) -> torch.Tensor:
    """A realization-level nuisance transform of a slow-TS window for the STABILITY gate.

    Slow-TS has no STFT-phase pair (like video), so the statistics-first "stability" property —
    codes unchanged under a realization-only perturbation — is probed with a small,
    structure-preserving augmentation: a per-window multiplicative amplitude jitter (a few %).
    The profile SHAPE (which positions carry more/less, its smooth time trend) is preserved;
    only the measurement-scale realization moves — so a statistics-first codec should map ``x``
    and ``slowts_nuisance(x)`` to (nearly) the same codes. ``x`` is ``(B, C, T)``.
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
    """slow-TS analogue of :func:`spike.compute_gate` (§4.4 gate on codes + masked recon).

    Parameters
    ----------
    codec : SlowTSCodec
    eval_windows : list of (B, C, T)
        Held-out windows; used for **stability** (codes of window vs :func:`slowts_nuisance`)
        and **slow-TS decode-fidelity** (masked recon of window vs window).
    eval_masks : list of (B, C, T)
        The matching per-sample validity masks (aligned with ``eval_windows``), used to mask
        the decode-fidelity profile statistic.
    frame_seq : (B, n_windows, C, T)
        Consecutive world-model windows; used for **persistence** (window t vs t+1) and
        **forecastability** (whole sequence of codes). Each "frame" is one 50 ms slow-TS window.
    cfg : SlowTSCodecConfig

    Returns the same key set as :func:`spike.compute_gate`, so :func:`spike.gate_score` and the
    trainer plumbing consume it unchanged.
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
        # FULL-WINDOW reconstruction fidelity (nothing collapsed) + the trivial baselines in the
        # SAME units. envelope_corr above deletes the time axis first, so it reads 1.0 for a
        # predictor with ZERO temporal structure; slowts_nrmse does not. Recorded in every
        # gate_*.json from 2026-09-03. It deliberately does NOT enter spike.gate_score —
        # checkpoint selection must not shift silently under a new metric.
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


# ------------------------------------------------------------------------------------- #
# slow-TS DDP generator-loss adapter + per-step (no δ-pair, no discriminator)
# ------------------------------------------------------------------------------------- #
class _SlowTSGenLossAdapter(torch.nn.Module):
    """Thin wrapper whose ``forward`` dispatches to ``SlowTSCodec.generator_losses``.

    Same DDP purpose as :class:`_GenLossAdapter` / :class:`_VideoGenLossAdapter`: DDP's reducer
    only hooks params used in ``forward``; the codec's step is ``codec.generator_losses(...)``,
    so we DDP-wrap THIS adapter. ``self.codec`` is the underlying :class:`SlowTSCodec` (used for
    gate + EMA + state_dict). The slow-TS signature has NO ``x_shift`` and NO ``disc`` (masked
    reconstruction + entropy only).
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
    """ONE codec training step on a slow-TS window (single-GPU / bare path).

    There is NO GAN alternation (no discriminator): a single generator (codec) step on the
    masked-reconstruction + entropy loss. Shared by the DDP trainer's ``world_size==1`` path
    and the tests.
    """
    codec.train()
    opt_g.zero_grad(set_to_none=True)
    g_terms = codec.generator_losses(x, cfg, step=step, mask=mask)
    g_terms["total"].backward()
    # Divergence guard (single-process path here; DDP-collective when world_size>1). Slow-TS is
    # non-adversarial, but the guard is a cheap uniform safety net shared with the other codecs.
    if spike.is_step_diverged(g_terms["total"]):
        spike.note_skipped_step()
    else:
        opt_g.step()
    return g_terms


def _ddp_slowts_train_step(
    gen_module: torch.nn.Module,        # DDP-wrapped _SlowTSGenLossAdapter (or bare)
    codec: SlowTSCodec,                 # underlying raw codec (unused here; kept for symmetry)
    opt_g: torch.optim.Optimizer,
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    cfg: SlowTSCodecConfig,
    step: int,
) -> Dict[str, torch.Tensor]:
    """DDP-aware analogue of :func:`slowts_codec_train_step` (routes the G forward through DDP)."""
    opt_g.zero_grad(set_to_none=True)
    g_terms = gen_module(x, cfg, step, mask)
    g_terms["total"].backward()
    # DDP-safe divergence guard: uniform backward, then skip opt_g.step() identically on all ranks
    # if any rank's loss is non-finite.
    if spike.is_step_diverged(g_terms["total"]):
        spike.note_skipped_step()
    else:
        opt_g.step()
    return g_terms


# ------------------------------------------------------------------------------------- #
# held-out streamed SLOW-TS eval set (windows + masks + consecutive-window sequence)
# ------------------------------------------------------------------------------------- #
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
    """Materialize a small held-out SLOW-TS gate set (mirrors :func:`_stream_video_eval_data`).

    Returns ``(eval_windows, eval_masks, frame_seq)`` for :func:`slowts_compute_gate`:
    ``eval_windows`` / ``eval_masks`` = lists of ``(B, C, T)`` window / validity-mask batches
    (for stability + slow-TS decode-fidelity); ``frame_seq`` = ``(B, n_windows, C, T)``
    consecutive 50 ms windows (for persistence + forecastability). Streamed on THIS rank
    (world_size=1) so every rank has the same held-out data; the gate is computed on rank 0
    only. ``eval_frames`` = the number of consecutive world-model WINDOWS in the sequence.
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


# ------------------------------------------------------------------------------------- #
# video stability nuisance + gate (mirrors spike.compute_gate for video)
# ------------------------------------------------------------------------------------- #
def video_nuisance(clip: torch.Tensor, *, bright: float = 0.02, offset: float = 0.02,
                   seed: int = 0) -> torch.Tensor:
    """A realization-level nuisance transform of a video clip for the STABILITY gate.

    Video has no STFT-phase pair (unlike spectro), so the statistics-first "stability"
    property — codes unchanged under a realization-only perturbation — is probed with a
    small, structure-preserving augmentation. The divertor camera is FIXED, so a spatial
    shift is NOT a physical nuisance; the real realization noise is a camera brightness /
    gain fluctuation. So the nuisance is a per-clip **brightness/gain** jitter only: a
    multiplicative gain plus a small additive offset on the (already standardized) frames.
    The plasma *structure* (where the light sits, its relative intensity envelope) is
    preserved; only the overall level moves — so a statistics-first codec should map
    ``clip`` and ``video_nuisance(clip)`` to (nearly) the same codes.

    ``clip`` is ``(B, C, T, H, W)``; returns the same shape.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # multiplicative gain jitter (per-clip scalar) — camera-gain realization.
    gain = 1.0 + bright * (2.0 * torch.rand((), generator=gen).item() - 1.0)
    # small additive offset (per-clip scalar) on the standardized frames — brightness bias.
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
    """Video analogue of :func:`spike.compute_gate` (§4.4 gate on frame codes + recon).

    Parameters
    ----------
    codec : VideoCodec
    eval_clips : list of (B, C, T, H, W)
        Held-out clips; used for **stability** (codes of clip vs :func:`video_nuisance`) and
        **video decode-fidelity** (recon of clip vs clip).
    frame_seq : (B, n_windows, C, T, H, W)
        Consecutive world-model windows; used for **persistence** (window t vs t+1) and
        **forecastability** (whole sequence of codes). Each "frame" of the world model is one
        50 ms video window (the same stepping unit as spectro).
    cfg : VideoCodecConfig

    Returns the same key set as :func:`spike.compute_gate`, so :func:`spike.gate_score` and
    the trainer plumbing consume it unchanged: ``stability`` / ``persistence`` /
    ``forecastability`` / ``decode`` / ``utilization`` + the ``pass_*`` booleans.
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
        # recon lives in the STANDARDIZED frame space (see VideoCodec.standardize_input); compare
        # decode fidelity against the standardized input (out["x_std"]), not the raw clip.
        dm = gate.video_decode_fidelity(out["recon"], out["x_std"])
        dec_corr.append(dm["envelope_corr"])
        dec_f1.append(dm["peak_f1"])
        dec_sharp.append(dm["sharpness"])
        # FULL-array MASKED metrics (the nRMSE floor + the AMPLITUDE ratio nRMSE cannot
        # report) plus the patch-lattice / checkerboard scalar beside its own GT reference.
        # ADDITIVE: spike.gate_score does not read any of these, so best-checkpoint selection
        # is unchanged; they exist so a gate JSON records all four acceptance quantities
        # (reconstruction, amplitude, checkerboard, bit rate) instead of just one.
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
        # PATCH-LATTICE (checkerboard) beside the ``sharpness`` it can masquerade as.
        # ADDITIVE ONLY: :func:`spike.gate_score` reads exactly ``envelope_corr`` /
        # ``peak_f1`` from this dict (plus ``forecastability`` + ``utilization``), so adding
        # these two keys leaves every selection decision bit-identical. They live in
        # ``decode`` rather than only in ``full`` because :func:`spike._fmt_gate` already
        # prints ``lattice=<recon>/gt<target>`` whenever ``patch_lattice_ratio`` is present —
        # which is what makes the artifact visible in the TRAINING LOG, not just in the gate
        # JSON. Job 5413747 (video1kb) raised video ``sharpness`` 0.157 -> 0.378 with an
        # adversarial term and NOTHING in the log could say whether that was texture or
        # checkerboard; on the spectro side the same objective sustained a lattice of 26-75
        # against a ground truth of ~1.15. Always read this as a PAIR with the GT value: a
        # BLUR also drops the lattice, for free, by destroying real detail too.
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
        # ADDITIVE (2026-09-03), NOT read by spike.gate_score.
        "full": vfull,
        "rate": gate.code_rate_bits(codes_seq, cfg.codebook_size, n_tok=cfg.n_tok),
        "pass_stability": bool(stability_val >= cfg.gate_stability),
        "pass_persistence": bool(persistence_val >= cfg.gate_persistence),
        "pass_utilization": bool(not util["collapsed"]),
    }


# ------------------------------------------------------------------------------------- #
# DDP generator-loss adapter (routes generator_losses through the DDP wrapper)
# ------------------------------------------------------------------------------------- #
# Prefixes of parameters that are provably an EXACT IDENTITY at initialization, so grafting them
# onto a checkpoint that predates them leaves step 0 bit-identical: the refinement head's final
# conv is zero-init (nets.SpectroDecoder) and decoder_noise's per-channel scale is zeros.
_GRAFTABLE_PREFIXES = ("decoder.refine.", "decoder.noise_scale", "noise_scale")

# Keys whose SHAPE is a pure function of ``cfg.fsq_levels`` and therefore MUST be re-initialized
# when --fsq_levels changes the codebook DIMENSIONALITY (len(fsq_levels)). These are the FSQ
# bottleneck's two projections (d_model <-> fsq_dim); nothing else in the codec depends on
# fsq_dim. Re-sizing the VOCAB (e.g. [8,5,5,5]=1000 -> [8,5,5]=200) is an explicit, deliberate
# CLI act, so refusing it is not protecting against a silent architecture mismatch — it just
# blocks the experiment (it killed job 5411167 outright, taking 7 healthy sibling arms with it
# because KillOnBadExit=1). Dropping ONLY these keys keeps every other weight grafted from the
# parent checkpoint, which is the whole point of resuming; a same-fsq_dim vocab change (e.g.
# [8,5,5,5] -> [4,4,4,4]) still matches shapes exactly and never reaches this path.
_FSQ_DIM_DEPENDENT_KEYS = (
    "quantizer.fsq.project_in.weight", "quantizer.fsq.project_in.bias",
    "quantizer.fsq.project_out.weight", "quantizer.fsq.project_out.bias",
    "quantizer.pre_quant_levels.weight", "quantizer.pre_quant_levels.bias",
)


def _drop_fsq_dim_mismatches(codec, state, log_fn=None):
    """Return ``state`` minus the FSQ projection keys whose shape disagrees with ``codec``.

    Only the keys in :data:`_FSQ_DIM_DEPENDENT_KEYS` are eligible, and only when the shape
    actually differs — so an ordinary resume is byte-identical (nothing is dropped) and a real
    architecture mismatch anywhere else still raises in
    :func:`_load_codec_state_allow_graft`.
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
    """``codec.load_state_dict(state)``, but allow ADDING an identity-at-init module.

    Strict loading is the right default and stays the default for everything else: an
    architecture flag that silently fails to load is how seven jobs died (see
    feedback-persist-arch-flags-in-checkpoints). So this permits EXACTLY the keys that are
    zero-init identities and raises on anything else, naming the offending keys.

    Enables the one experiment strict loading blocks: take a healthy, high-utilization codec and
    add the dilated refine head to it, instead of training the head from scratch (where it lands
    at ~103/1000 codes).
    """
    # A --fsq_levels change of DIMENSIONALITY makes the FSQ projections' shapes disagree; drop
    # exactly those so they re-initialize instead of raising a size-mismatch RuntimeError.
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
    """Restore Adam moments onto a model that GAINED parameters since the checkpoint.

    ``Optimizer.load_state_dict`` matches param groups POSITIONALLY, so it raises as soon as the
    counts differ - and grafted params are inserted in the MIDDLE of ``model.parameters()`` (the
    refine head is registered inside the decoder), so a naive index restore would silently pair
    old moments with the wrong tensors. Both failure modes are avoided by remapping BY NAME:
    saved slot j belongs to the j-th non-grafted parameter. Grafted params start with fresh
    (zero) moments, which is correct - they are zero-init identities with no history.

    Exact-restore path (nothing grafted) is untouched, so ordinary resumes are unchanged.
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
    """Re-pin the REQUESTED --lr over the one ``opt.load_state_dict`` restored.

    ``torch.optim.Optimizer.load_state_dict`` restores ``param_groups`` wholesale, INCLUDING
    ``lr``. A resumed run that asks for a different learning rate therefore trains silently at
    the PARENT checkpoint's rate. train_dynamics.py already guards this (measured 2026-08-26:
    prod_nfullhilr requested 2e-3 and logged 1.00e-03 every step); the codec trainer did not.
    Detected 2026-09-02 as an accidental A/A: mhr_cont arms `c_ctl` (lr 1e-3) and `c_lr3e4`
    (lr 3e-4) produced BIT-IDENTICAL gate metrics across 15 consecutive gates.

    No-op when the rate is unchanged (every ordinary chained resume), so existing chains are
    byte-identical.
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
    """Exponential LR decay, NVIDIA Spectral Codec style (arXiv 2406.05298 section 4).

    ``lr(step) = base_lr * gamma ** (step / every)`` with ``gamma = cfg.lr_decay_gamma`` and
    ``every = cfg.lr_decay_every`` (paper: gamma 0.998 per 1,000 steps). Computed from the
    ABSOLUTE global step rather than accumulated per-step, so a chained resume lands on
    exactly the rate an uninterrupted run would have had — and so it composes correctly with
    :func:`_repin_lr_after_resume`, which re-pins ``base_lr`` before the loop starts.

    ``gamma == 1.0`` (the default) is a no-op: nothing is written to ``param_groups`` at all,
    so existing runs are byte-identical. Returns the generator LR in force this step.
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
    """Thin wrapper whose ``forward`` dispatches to ``SpectroCodec.generator_losses``.

    DDP's autograd reducer only hooks the parameters used in the module's ``forward``. The
    codec's generator step is ``codec.generator_losses(...)``, not ``codec.forward(x)``, so
    to make DDP correctly all-reduce the generator gradients we wrap the codec in this
    adapter and DDP-wrap the ADAPTER. The training step then calls the DDP-wrapped adapter,
    whose ``forward`` runs ``generator_losses`` — so every codec parameter touched there is
    seen by the reducer.

    ``self.codec`` is a registered submodule, so ``adapter.codec`` is the underlying
    :class:`SpectroCodec` (used for the detached discriminator recon + gate eval + EMA +
    state_dict, which must NOT go through the DDP graph).
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
    """DDP-aware analogue of :func:`spike.codec_train_step`.

    Same math as :func:`spike.codec_train_step` but routes the GENERATOR forward through the
    DDP-wrapped adapter (so the codec-gradient all-reduce is hooked) and the DISCRIMINATOR
    forward through its DDP wrapper. The discriminator inside ``generator_losses`` is called
    with ``disc_raw`` (no grad on D during the G step, matching the spike). ``codec.forward``
    for the detached D-step recon uses the raw codec (no DDP graph needed — it's under
    ``no_grad``).
    """
    codec.train()
    disc_raw.train()

    # ---- generator (codec) step: forward through the DDP-wrapped adapter ----
    opt_g.zero_grad(set_to_none=True)
    g_terms = gen_module(spec_a, spec_b, disc_raw, cfg, step, frame_mask)
    g_terms["total"].backward()
    # DDP-safe divergence guard: backward (+ its grad all-reduce) ran uniformly on every rank;
    # opt_g.step() is skipped IDENTICALLY on all ranks if any rank's loss is non-finite.
    if spike.is_step_diverged(g_terms["total"]):
        spike.note_skipped_step()
    else:
        opt_g.step()

    # ---- discriminator step (fresh detached recon; forward through DDP-wrapped disc) ----
    # CADENCE (NVIDIA Spectral Codec, arXiv 2406.05298 section 4: "we update the discriminators
    # only once every two steps"). cfg.disc_update_every defaults to 1 = every step, which is
    # the prior behaviour EXACTLY. The predicate uses the GLOBAL step so the cadence is
    # identical on every DDP rank and stable across chained resumes; skipping is uniform
    # across ranks, so no rank desyncs on the disc gradient all-reduce.
    every = int(getattr(cfg, "disc_update_every", 1))
    if every > 1 and (step % every) != 0:
        # Still report the (cheap, no-grad) discriminator loss so the gate line has a number,
        # but take no optimizer step and build no graph.
        # IMPORTANT: score with ``disc_raw``, NOT the DDP wrapper. A DDP forward arms the
        # reducer to expect a matching backward; skipping that backward makes the NEXT
        # iteration raise "Expected to have finished reduction in the prior iteration". The
        # raw module is the same weights with no reducer bookkeeping.
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
    """:func:`losses.discriminator_loss`, with the missing windows removed from BOTH sides.

    2026-09-03. The spectro discriminator step was the LAST unmasked consumer: with no mask
    anywhere on the spectro path, D was trained to call a constant log-eps plate (a shot whose
    HDF5 group is an empty stub) and a zero-slab channel "real". That is a direct reward for a
    mean-collapsed generator, which is the failure mode this whole family fights.

    Uses EXACTLY the selector the generator uses (``SpectroCodec._valid_windows``), so the two
    halves of the GAN alternation never disagree about what data is real. ``frame_mask`` None,
    or a mask that selects every window, keeps the ORIGINAL tensors and calls
    ``losses.discriminator_loss`` unchanged — bit-identical to the pre-fix path.
    """
    from .losses import discriminator_loss

    win = SpectroCodec._valid_windows(frame_mask, real.shape)
    if win is not None:
        real, fake = real[win], fake[win]
    return discriminator_loss(disc, real, fake, cfg)


# ------------------------------------------------------------------------------------- #
# video DDP generator-loss adapter + per-step (mirrors the spectro pair above, no x_shift)
# ------------------------------------------------------------------------------------- #
class _VideoGenLossAdapter(torch.nn.Module):
    """Thin wrapper whose ``forward`` dispatches to ``VideoCodec.generator_losses``.

    Same DDP purpose as :class:`_GenLossAdapter`: DDP's reducer only hooks params used in
    ``forward``; the codec's generator step is ``codec.generator_losses(...)``, so we DDP-wrap
    THIS adapter. ``self.codec`` is the underlying :class:`VideoCodec` (used for the detached
    D-recon + gate + EMA + state_dict). The video signature has NO ``x_shift`` (no δ-pair).
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
    """ONE generator+discriminator alternation step on a video clip (single-GPU / bare path).

    The video analogue of :func:`spike.codec_train_step` — same alternation (generator step
    on the codec's ``generator_losses``, then a fresh DETACHED-recon discriminator step) but
    with no δ-shift pair. Shared by the DDP trainer's ``world_size==1`` path and the tests.
    """
    codec.train()

    opt_g.zero_grad(set_to_none=True)
    g_terms = codec.generator_losses(frames, disc, cfg, step=step, frame_mask=frame_mask)
    g_terms["total"].backward()
    # Divergence guard (single-process path here; DDP-collective when world_size>1).
    if spike.is_step_diverged(g_terms["total"]):
        spike.note_skipped_step()
    else:
        opt_g.step()

    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        out = codec.forward(frames)
        recon, real = out["recon"], out["x_std"]     # both in the STANDARDIZED frame space
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
    """True when the DISCRIMINATOR should be updated on ``step`` (paper: every 2 steps).

    ``cfg.disc_update_every`` defaults to 1 = update every step = the pre-2026-09-03 behaviour,
    so this is byte-identical unless a run asks for the NVIDIA Spectral Codec cadence
    (arXiv 2406.05298 section 4 uses 2). The d_loss VALUE is still computed and logged every
    step, only the backward+step is skipped, so the logs stay comparable across arms.
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
    """DDP-aware analogue of :func:`video_codec_train_step` (routes the G forward through DDP)."""
    codec.train()
    disc_raw.train()

    opt_g.zero_grad(set_to_none=True)
    g_terms = gen_module(frames, disc_raw, cfg, step, frame_mask)
    g_terms["total"].backward()
    # DDP-safe divergence guard: uniform backward, then skip opt_g.step() identically on all ranks
    # if any rank's loss is non-finite (video codec collapse was one of the 2026-07 crash triggers).
    if spike.is_step_diverged(g_terms["total"]):
        spike.note_skipped_step()
    else:
        opt_g.step()

    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        out = codec.forward(frames)
        recon, real = out["recon"], out["x_std"]     # both in the STANDARDIZED frame space
    if _disc_step_due(cfg, step):
        d_loss = _video_discriminator_loss(disc, real, recon, cfg, frame_mask)
        d_loss.backward()
        if spike.is_step_diverged(d_loss):
            spike.note_skipped_step()
        else:
            opt_d.step()
    else:
        # SKIPPED D step. The value is still logged, but it must go through the RAW module
        # under no_grad: a DDP-wrapped forward with no matching backward leaves the reducer
        # armed and corrupts (or hangs) the next iteration.
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
    """Hinge GAN discriminator loss for the frame PatchGAN (reuses the shared hinge helpers).

    Identical math to :func:`losses.discriminator_loss` (``E[relu(1-D(real))] +
    E[relu(1+D(fake))]`` over the per-frame score maps); factored here only because
    ``losses.discriminator_loss`` is typed to ``SpectroCodecConfig`` (it reads no cfg field,
    but keeping a video-typed entry point is clearer and avoids any future cfg-field coupling).

    ``frame_mask`` (2026-09-03) drops the frames whose cameras were OFF from BOTH the real and
    the fake side, using the SAME selector the generator uses
    (:meth:`VideoCodec._disc_frame_selector`). Without it the discriminator is trained to call a
    constant-zero plate "real" -- and 51.3% (lower) / 68.6% (upper) of the streamed windows are
    exactly that. ``None`` / an all-valid mask keeps the ORIGINAL tensors, so the default path
    is bit-identical.
    """
    from .losses import _as_score_list, _hinge_fake, _hinge_real

    keep = VideoCodec._disc_frame_selector(frame_mask, real.shape)
    if keep is not None:
        real = VideoCodec._select_frames(real, keep)
        fake = VideoCodec._select_frames(fake, keep)
    real_maps = _as_score_list(disc(real))
    fake_maps = _as_score_list(disc(fake.detach() if fake.requires_grad else fake))
    return _hinge_real(real_maps) + _hinge_fake(fake_maps)


# ------------------------------------------------------------------------------------- #
# held-out streamed eval set (disjoint shots) for the gate
# ------------------------------------------------------------------------------------- #
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
    """Materialize a small held-out gate set by streaming ``eval_shots`` on THIS rank only.

    Returns ``(eval_pairs, frame_seq)`` matching :func:`spike.compute_gate`'s contract:
    ``eval_pairs`` = list of ``(spec_a, spec_b)`` batches; ``frame_seq`` =
    ``(B, n_frames, C, F, T)`` consecutive-frame sequence. The eval set is small + fixed
    (materialized once) so gate scores are comparable across evals. Runs single-process here
    (world_size=1) so every rank has the SAME held-out data — the gate is computed on rank 0
    only, but building it uniformly keeps the code rank-agnostic.
    """
    # δ-pairs via the SAME production dataset (single-process, deterministic draw). Iterate
    # its map-style windows in order and materialize the first ``eval_batches*eval_batch_size``.
    ds = CodecPairDataset(modality, eval_shots, cfg, data_dir=data_dir, seed=seed,
                          emit_mask=bool(getattr(cfg, "mask_missing", False)))
    # THE HELD-OUT SET MUST NOT BE ACTIVITY-STRATIFIED (2026-09-04).
    #
    # `cfg.min_activity` / `cfg.active_bias` are a TRAINING-time sampling choice (bias the
    # batch toward active windows without dropping quiet ones). Letting them reach the gate
    # set is wrong twice over:
    #
    #  * CORRECTNESS -- the gate would be measured on a sample deliberately biased toward
    #    high-activity windows, so it does not describe held-out data. Only co2 and mirnov
    #    carry the knobs, which is exactly where the gate was least trustworthy.
    #  * COST -- `_stratified_draw` re-draws up to `max_tries` times per item, each a full
    #    `_build_pair`, and this loop is SINGLE-PROCESS over `eval_batches * eval_batch_size`
    #    items. Measured on mirnov (job 5416298): 29 channels x ~29 Lustre seeks per build x
    #    up to 8 re-draws x 128 items consumed the ENTIRE 2 h leg -- all four mirnov arms
    #    wrote zero gates and zero checkpoints while the bes arms beside them reached step
    #    14000-29999. It is paid again on every resume because the eval set is never cached.
    #
    # Zeroing the two knobs on the DATASET INSTANCE (not the cfg, which the codec and the
    # checkpoint share) makes `_stratified_draw` return its base draw immediately, which is
    # its documented byte-identical path. Training sampling is untouched.
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
        # 3rd element ONLY when the dataset carries cfg.mask_missing; spike.compute_gate
        # forwards it to gate.decode_fidelity so the held-out metrics are computed on real
        # data only. Without it the tuple is the same 2-tuple as before (byte-identical).
        if len(chunk[0]) > 2:
            eval_pairs.append(
                (a, b, torch.stack([p[2] for p in chunk], dim=0).to(device))
            )
        else:
            eval_pairs.append((a, b))

    # consecutive-frame sequence for persistence / forecastability, streamed from eval shots.
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
    """Consecutive-frame ece/co2/... sequence for the gate (mirrors spike.real_frame_sequence).

    Scans ``shots`` for blocks of ``n_frames`` consecutive valid 50 ms windows; STFTs each to
    log-power. Reuses the read-only loader access + ``data.log_power_stft``. Single-process.
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
        # open via the generic per-modality path (spike._open_ece_dataset is ece-only).
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


# ------------------------------------------------------------------------------------- #
# held-out streamed VIDEO eval set (clips + consecutive-window sequence)
# ------------------------------------------------------------------------------------- #
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
    """Materialize a small held-out VIDEO gate set (mirrors :func:`_stream_eval_data`).

    Returns ``(eval_clips, frame_seq)`` for :func:`video_compute_gate`: ``eval_clips`` = list
    of ``(B, C, T, H, W)`` clip batches (for stability + video decode-fidelity); ``frame_seq``
    = ``(B, n_windows, C, T, H, W)`` consecutive 50 ms windows (for persistence +
    forecastability). Streamed on THIS rank (world_size=1) so every rank has the same held-out
    data; the gate is computed on rank 0 only. ``eval_frames`` here is the number of
    consecutive world-model WINDOWS in the sequence (each window is one 50 ms video clip).
    """
    ds = VideoCodecPairDataset(modality, eval_shots, cfg, data_dir=data_dir, seed=seed)
    n_clips = eval_batches * eval_batch_size
    if len(ds) < n_clips:
        raise RuntimeError(
            f"_stream_video_eval_data: only {len(ds)} eval windows across "
            f"{len(list(eval_shots))} eval shots; need {n_clips} "
            f"(eval_batches={eval_batches} * eval_batch_size={eval_batch_size})."
        )
    # 2026-09-03: the MASK is materialized too. It used to be dropped here ("mask unused in
    # the gate"), so every gate number was computed over zero-filled dead cameras -- and on
    # tangtv that is 51.3% (lower) / 68.6% (upper) of shots. The mask is only NON-trivial when
    # cfg.mask_missing is on; otherwise it is the same all-ones vector as before, so the gate
    # is byte-identical for a run that does not ask for masking.
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

    # consecutive-window sequence: for each sample, ``eval_frames`` back-to-back 50 ms clips.
    bsz = max(2, eval_batch_size // 2)
    n_windows = eval_frames
    n_avail = len(ds)
    seqs: List[torch.Tensor] = []
    # stride blocks of n_windows consecutive dataset indices (dataset windows are already the
    # non-overlapping 50 ms cores, in shot order per the parent's cumulative-length layout).
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


# ------------------------------------------------------------------------------------- #
# DDP init (env-based; matches utils.distributed / _srun_rank_wrapper.sh contract)
# ------------------------------------------------------------------------------------- #
class _DDPState:
    """Minimal DDP context read from the env vars set by ``_srun_rank_wrapper.sh``.

    ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` (torchrun / srun), NCCL backend, one GPU per
    rank (Frontier ``--gpus-per-task=1`` masks each rank to a single visible GCD → device
    index 0). Single-process when ``WORLD_SIZE`` is unset or 1 (the CPU smoke path).
    """

    def __init__(self, backend: str = "nccl") -> None:
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        if ws > 1:
            self.rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = ws
            visible = torch.cuda.device_count()
            self.device_index = self.local_rank if visible > 1 else 0
            self.distributed = True
            if not dist.is_initialized():
                dist.init_process_group(backend, rank=self.rank, world_size=self.world_size)
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


# ------------------------------------------------------------------------------------- #
# the streaming trainer
# ------------------------------------------------------------------------------------- #
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
    """Streaming, DDP-capable codec training loop (reuses spike's per-step + gate logic).

    Wires the streaming DataLoader → :func:`_ddp_codec_train_step` (== spike's per-step math,
    DDP-hooked) → periodic gate on a held-out streamed eval set → best-ckpt by
    :func:`spike.gate_score`. Rank-0 only logs + writes ``codec_last.pt`` / ``codec_best.pt``
    / (optional) ``codec_ema.pt`` / ``gate_<step>.json``.

    Returns the final gate dict (with ``steps`` / ``global_step`` / ``best_score`` /
    ``best_step`` keys), like :func:`spike.run_spike`.
    """
    if steps < 1:
        raise ValueError("train_codec: steps must be >= 1")
    ddp = ddp if ddp is not None else _DDPState()
    device = ddp.device
    torch.manual_seed(seed + ddp.rank)

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None and ddp.is_main:
        out_path.mkdir(parents=True, exist_ok=True)

    # --- models ---
    codec = SpectroCodec(cfg).to(device)
    # DISCRIMINATOR FAMILY. Default "patch" is FreqAwarePatchGAN, exactly as before.
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

    # Adam betas from the cfg (defaults 0.9/0.999 == torch's, so untouched runs are identical).
    # Paper (arXiv 2406.05298 section 4): "Adam optimizer with learning rate 2e-4,
    # beta1 = 0.8, beta2 = 0.99".
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

    # --- held-out gate set (same on every rank; gate computed on rank 0) ---
    eval_pairs, frame_seq = _stream_eval_data(
        modality, eval_shots, cfg,
        data_dir=data_dir, eval_batches=eval_batches, eval_batch_size=eval_batch_size,
        eval_frames=eval_frames, device=device, seed=seed + 777,
    )

    # --- train loader over the production TokamakMultiFileDataset subclass ---
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
    # The loader is a FINITE map-style epoch (unlike the old infinite IterableDataset). Wrap
    # it so the trainer's fixed ``steps`` budget cycles through epochs transparently.
    stream = _epoch_cycler(loader)

    # CHAIN-RESUME FIX (2026-08-05): seed best-tracking from an existing codec_best.pt
    # so a resume leg cannot clobber the global best with a worse leg-local one.
    best_score = spike.resume_best_score(out_path)
    best_step: Optional[int] = None
    final_gate: Dict[str, object] = {}
    spike.reset_skipped_steps()  # divergence-guard skip counter for this trainer run

    _base_lrs = [lr, disc_lr if disc_lr is not None else lr]
    for local_step in range(steps):
        step = start_step + local_step
        # Exponential LR decay on the ABSOLUTE step (no-op at the default gamma 1.0).
        _apply_lr_decay([opt_g, opt_d], _base_lrs, step, cfg)
        batch = next(stream)
        # The loader yields (a, b) by default and (a, b, mask) when cfg.mask_missing is set;
        # `frame_mask` stays None in the default case so every downstream term takes the
        # ORIGINAL (bit-identical) branch.
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


# ------------------------------------------------------------------------------------- #
# the streaming VIDEO trainer (mirrors train_codec; reuses the SAME DDP + best-ckpt + gate)
# ------------------------------------------------------------------------------------- #
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
    """Streaming, DDP-capable VIDEO codec training loop.

    The video analogue of :func:`train_codec` — it reuses the SAME infrastructure (the
    ``_DDPState`` init/wrap, ``spike._EMA``, ``spike.gate_score``, ``spike._fmt_gate``,
    ``spike._write_gate_json`` / ``spike._save_checkpoint`` / ``spike._save_best_checkpoint``
    / ``spike._save_ema_checkpoint`` best-ckpt+EMA+gate path) and differs only where video
    differs: a :class:`VideoCodec` + :class:`FramePatchGAN`, the ``(frames, frame_mask)``
    loader, the no-δ-pair :func:`_ddp_video_train_step`, and :func:`video_compute_gate`.
    Rank-0 only logs + writes ``codec_last.pt`` / ``codec_best.pt`` / (optional)
    ``codec_ema.pt`` / ``gate_<step>.json`` (identical filenames + score contract as spectro).
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

    # Adam betas from the cfg (defaults 0.9/0.999 == torch's, so untouched runs are identical).
    # Paper (arXiv 2406.05298 section 4): beta1 = 0.8, beta2 = 0.99.
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

    # CHAIN-RESUME FIX (2026-08-05): seed best-tracking from an existing codec_best.pt
    # so a resume leg cannot clobber the global best with a worse leg-local one.
    best_score = spike.resume_best_score(out_path)
    best_step: Optional[int] = None
    final_gate: Dict[str, object] = {}
    spike.reset_skipped_steps()  # divergence-guard skip counter for this trainer run

    _base_lrs = [lr, disc_lr if disc_lr is not None else lr]
    for local_step in range(steps):
        step = start_step + local_step
        # Exponential LR decay on the ABSOLUTE step (no-op at the default gamma 1.0).
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


# ------------------------------------------------------------------------------------- #
# the streaming SLOW-TS trainer (mirrors train_codec / train_video_codec; SAME DDP +
# best-ckpt + gate infra; no discriminator, no δ-pair)
# ------------------------------------------------------------------------------------- #
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
    disc_lr: Optional[float] = None,   # accepted + ignored (no discriminator); uniform signature
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
    """Streaming, DDP-capable SLOW-TS codec training loop.

    The slow-TS analogue of :func:`train_codec` / :func:`train_video_codec` — it reuses the
    SAME infrastructure (the ``_DDPState`` init/wrap, ``spike._EMA``, ``spike.gate_score``,
    ``spike._fmt_gate``, ``spike._write_gate_json`` / ``spike._save_checkpoint`` /
    ``spike._save_best_checkpoint`` / ``spike._save_ema_checkpoint`` best-ckpt+EMA+gate path)
    and differs only where slow-TS differs: a :class:`SlowTSCodec` with **no discriminator**,
    the ``(signal, mask)`` loader, the no-δ-pair / no-GAN :func:`_ddp_slowts_train_step`, and
    :func:`slowts_compute_gate`. Rank-0 only logs + writes ``codec_last.pt`` /
    ``codec_best.pt`` / (optional) ``codec_ema.pt`` / ``gate_<step>.json`` (identical filenames
    + score contract as the other codecs).

    NOTE: the ``_save_*`` helpers persist a discriminator; the slow-TS codec has none, so a
    tiny zero-parameter :class:`_NullDisc` stand-in is passed (its ``state_dict`` is empty),
    keeping the checkpoint schema uniform without a real GAN.
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
    disc_raw = _NullDisc().to(device)  # no discriminator; empty-state stand-in for _save_*

    start_step = 0
    if resume_state is not None:
        _grafted = _load_codec_state_allow_graft(
            codec, resume_state["codec"], log_fn=(print if ddp.is_main else None)) or []
        start_step = int(resume_state.get("step", 0))

    # DECODER-ONLY TRAINING (2026-09-03). Motivation, measured on the retrained slow-TS
    # codecs: an out-of-sample 93-parameter LINEAR read-out of the codec's OWN FROZEN codes
    # BEATS its trained 4-layer transformer decoder by 0.16-0.36 pooled nRMSE (cer_rot
    # 0.5788 vs 0.7349, cer_ti 0.5874 vs 0.8841, mse 0.6536 vs 0.7840) and recovers 8x more
    # temporal amplitude (slowts_std_ratio_t 0.34-0.38 vs 0.04). The decoder's final map is
    # already Linear(d_model=128 -> patch_c*patch_t=60), i.e. over-complete, and the probe
    # that beats it is STRICTLY less expressive than the decoder — so this is an
    # OPTIMIZATION failure, not a capacity one. The most likely cause is that encoder and
    # decoder train jointly, so the decoder chases a moving code distribution.
    #
    # With `freeze_encoder` the encoder + quantizer are frozen at the resumed checkpoint's
    # weights (so the CODES ARE FIXED, exactly the setting the read-out measured) and only
    # the decoder trains. If the decoder then reaches the read-out's score, joint-training
    # non-stationarity is the cause; if it does not, the objective itself is.
    #
    # The entropy/utilization terms depend only on the frozen encoder, so they become
    # constants with no gradient path — harmless, and their logged values simply stop moving.
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
    # dummy optimizer over the null disc's (empty) params so _save_checkpoint has an opt_d.
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

    # CHAIN-RESUME FIX (2026-08-05): seed best-tracking from an existing codec_best.pt
    # so a resume leg cannot clobber the global best with a worse leg-local one.
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
    """A zero-parameter discriminator stand-in for the slow-TS codec (which has no GAN).

    The shared ``spike._save_checkpoint`` / ``_save_best_checkpoint`` helpers persist a
    discriminator ``state_dict`` to keep the checkpoint schema uniform across codecs. The
    slow-TS codec is a reconstruction + entropy codec with no discriminator (§4.3), so we
    pass this empty module — ``state_dict()`` is ``{}`` — instead of a real one. It is never
    called in a forward pass.
    """

    def state_dict(self, *args, **kwargs):  # noqa: D401 (empty state)
        return {}


# ------------------------------------------------------------------------------------- #
# CLI
# ------------------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tokamak_foundation_model.ignite.train_codec",
        description="IGNITE Phase-A STREAMING, DDP codec trainer (scales past the spike's "
                    "~40-shot pre-built pool to thousands of shots on the fly).",
    )
    p.add_argument("--modality", type=str, default="ece",
                   choices=list(SPECTRO_MODALITIES) + list(VIDEO_MODALITIES)
                           + list(SLOWTS_MODALITIES) + [FASTTS_MODALITY],
                   help="Codec modality: a spectro signal (ece/co2/bes/mhr), a tangtv "
                        "divertor video (tangtv_lower/tangtv_upper), a slow-TS signal "
                        "(ts_core_density/ts_core_temp/ts_tangential_density/"
                        "ts_tangential_temp/cer_ti/cer_rot/mse), or the fast-TS ELM-envelope "
                        "codec (filterscopes).")
    p.add_argument("--shots", type=str, default=None,
                   help="Explicit comma-separated train shot list (overrides --n_shots).")
    p.add_argument("--n_shots", type=int, default=1000,
                   help="Number of shots to auto-discover for the train stream.")
    p.add_argument("--eval_n_shots", type=int, default=16,
                   help="Held-out shots for the gate eval set (disjoint from train).")
    p.add_argument("--steps", type=int, default=20000, help="Training/gate steps.")
    p.add_argument("--eval_every", type=int, default=1000, help="Gate/checkpoint cadence.")
    p.add_argument("--batch_size", type=int, default=8, help="Pairs per batch per rank.")
    p.add_argument("--num_workers", type=int, default=6,
                   help="DataLoader parallel workers per rank (streaming).")
    p.add_argument("--lr", type=float, default=1e-3, help="Codec (generator) Adam LR.")
    p.add_argument("--disc_lr", type=float, default=None, help="Discriminator LR (default lr).")
    p.add_argument("--ema", action="store_true", help="Maintain + write an EMA codec shadow.")
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay in [0,1).")
    p.add_argument("--eval_batches", type=int, default=8, help="Held-out gate δ-pair batches.")
    p.add_argument("--eval_batch_size", type=int, default=4, help="Pairs per eval batch.")
    p.add_argument("--eval_frames", type=int, default=6,
                   help="Consecutive frames per persistence/forecastability sequence.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output dir for gate_<step>.json / codec_{last,best,ema}.pt / summary.")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                   help="Directory of {shot}_processed.h5 files.")
    p.add_argument("--stats_path", type=str, default=None,
                   help="preprocessing_stats.pt for the fast-TS (filterscopes) envelope + slow-TS "
                        "(Thomson/CER/MSE) input SCALE FIX (per-channel mean/std used to "
                        "standardize the codec input the SAME way the FM model does; slow-TS also "
                        "picks log_standardize vs standardize per signal). None => "
                        "fastts_train.DEFAULT_STATS_PATH; '' => DISABLE (raw path, input will "
                        "collapse; debugging only). Ignored for spectro / video modalities.")
    p.add_argument("--lengths_cache_dir", type=str, default=None,
                   help="Directory for the per-file chunk-length sidecar cache "
                        "(codec_<modality>_lengths.pt); reuses the parent dataset's cache to "
                        "skip re-scanning shot lengths at startup. None disables caching.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None,
                   help="Optional codec_last.pt to warm-start codec/disc/opt/step from.")
    p.add_argument("--slowts_d_model", type=int, default=None,
                   help="Override SlowTSCodecConfig.d_model (slow-TS only; default 128). Changes "
                        "EVERY weight shape, so it is only usable from scratch (no --resume).")
    p.add_argument("--slowts_enc_depth", type=int, default=None,
                   help="Override SlowTSCodecConfig.enc_depth (slow-TS only; default 4).")
    p.add_argument("--slowts_freeze_encoder", action="store_true",
                   help="SLOW-TS ONLY: freeze the encoder + quantizer at the RESUMED "
                        "checkpoint's weights (the CODES become fixed) and train ONLY the "
                        "decoder. Diagnostic for the measured decoder-limitation: a linear "
                        "read-out of the frozen codes beats the trained decoder by 0.16-0.36 "
                        "pooled nRMSE, so this asks whether joint-training non-stationarity "
                        "is why. Requires --resume (freezing a random encoder is meaningless).")
    p.add_argument("--slowts_dec_depth", type=int, default=None,
                   help="Override SlowTSCodecConfig.dec_depth (slow-TS only; default 4). MEASURED "
                        "2026-09-03: an out-of-sample LINEAR read-out of the codec's OWN frozen "
                        "quantized codes BEATS the trained 4-layer decoder on cer_rot "
                        "(0.5794 vs 0.7380 pooled nRMSE), cer_ti (0.6044 vs 0.8892) and mse "
                        "(0.6581 vs 0.7840) - i.e. for those three the information is already in "
                        "the codes and the decoder is what fails to extract it.")
    p.add_argument("--recon_weight", type=float, default=None,
                   help="Override cfg.recon_weight — the L1 (MAE) reconstruction weight. On the "
                        "slow-TS codec this ships at 1.0 against entropy_weight 1.0, i.e. the "
                        "reconstruction term is weighted at PARITY with the anti-collapse "
                        "regularizer (measured: the combined loss goes NEGATIVE, so diversity can "
                        "outbid reconstruction). The NVIDIA Spectral Codec reference weights "
                        "reconstruction 20 against its 1.0 regulariser.")
    p.add_argument("--recon_mse_weight", type=float, default=None,
                   help="Override cfg.recon_mse_weight (slow-TS; default 0 = OFF): weight of a "
                        "MASKED SQUARED-error term alongside the L1. The gate metric "
                        "(slowts_nrmse = RMSE/std) is a squared error, so an L1-only objective "
                        "optimises the conditional median of a metric that wants the mean. Pair "
                        "with --recon_weight 0 for pure L2.")
    p.add_argument("--entropy_weight", type=float, default=None,
                   help="Override cfg.entropy_weight (anti-collapse strength). NOTE this scales "
                        "the WHOLE entropy_loss return value, including the two joint terms "
                        "below, so the effective joint weights are entropy_weight * <flag>.")
    p.add_argument("--joint_entropy_weight", type=float, default=None,
                   help="Override cfg.joint_entropy_weight (default 0 = off): reward on the "
                        "entropy of the batch-mean JOINT code distribution over all "
                        "prod(fsq_levels) codes (MagViT-2/LFQ codebook entropy). The existing "
                        "per-dim diversity reward is MARGINAL and a rank-1 encoder saturates it "
                        "while using ~nothing of the codebook (measured on prod mhr: "
                        "min_dim_entropy 0.921 with 42/32768 codes, per-dim level positions "
                        "|r|>=0.997). Max value log(codebook_size) = 6.908 nats at cb=1000; "
                        "spectro only, and refused above 4096 codes (memory).")
    p.add_argument("--joint_entropy_ramp_steps", type=int, default=None,
                   help="Override cfg.joint_entropy_ramp_steps (default 0 = off): linearly ramp "
                        "the joint-entropy weight from 0 to --joint_entropy_weight over the "
                        "first N generator steps. The encoder starts SATURATED on real spectro "
                        "input (every mhr arm read 1 distinct code at step 0), and a large "
                        "constant joint weight cannot climb out (measured: weight 5.0 stayed at "
                        "1 code / pre-quant level std 0.0000 through step 2000, while 1.0/2.0 "
                        "reached joint entropies of 5.00/5.69 nats). Spectro only.")
    p.add_argument("--decorrelation_weight", type=float, default=None,
                   help="Override cfg.decorrelation_weight (default 0 = off): penalty on the "
                        "mean squared OFF-DIAGONAL correlation of the per-dim continuous "
                        "pre-quant level positions. Direct attack on the rank-1 FSQ collapse "
                        "(0 = independent dims, ~1 = one scalar replicated). Spectro only.")
    p.add_argument("--fsq_noise_dropout", type=float, default=None,
                   help="Override cfg.fsq_noise_dropout (default 0 = off): FSQ's OWN "
                        "`noise_dropout` constructor arg. With this probability per element, "
                        "FSQ.maybe_apply_noise adds a uniform +/-0.5 offset to the BOUNDED code "
                        "and re-clamps to [-1,1], TRAINING ONLY. It runs AFTER codes_to_indices, "
                        "so the emitted indices stay clean and only the decoder-facing latent is "
                        "jittered. The +/-0.5 is in the NORMALIZED [-1,1] space where the bin "
                        "spacing is 2/(L-1), so for [8,5,5] it is +/-1.75 bins on the 8-level dim "
                        "and +/-1.0 on the 5-level dims. REQUIRES --fsq_preserve_symmetry. "
                        "Spectro only.")
    p.add_argument("--fsq_preserve_symmetry", action="store_true",
                   help="Set cfg.fsq_preserve_symmetry (default off): FSQ's OWN "
                        "`preserve_symmetry` constructor arg, which swaps in "
                        "symmetry_preserving_bound (arXiv 2411.19842 s3.2) and the matching "
                        "_scale_and_shift. MANDATORY with --fsq_noise_dropout (the library "
                        "asserts it), but NOT a free rider: it changes the emitted codes even at "
                        "noise_dropout 0, so always run a preserve_symmetry-only CONTROL arm or "
                        "the noise-dropout effect is unattributable. Spectro only.")
    p.add_argument("--pixel_anchor_weight", type=float, default=None,
                   help="Override cfg.pixel_anchor_weight (L1 RECONSTRUCTION strength). Default 0.05 "
                        "was an anchor meant to work WITH the adversarial; raise it (~1-5) when "
                        "adversarial is off so the decode actually reconstructs the spectrogram.")
    p.add_argument("--multiscale_recon_weight", type=float, default=None,
                   help="Override cfg.multiscale_recon_weight — multi-resolution L1 (NeMo-style) that "
                        "sharpens turbulent detail plain pixel-L1 blurs. ~1-5 recommended.")
    p.add_argument("--freq_grad_weight", type=float, default=None,
                   help="Override cfg.freq_grad_weight — L1 on the frequency-derivative; directly "
                        "penalizes a smooth envelope. ~1-5 recommended.")
    p.add_argument("--gate_hard_min_codes", type=int, default=None,
                   help="Override cfg.gate_hard_min_codes (hard best-ckpt floor on ABSOLUTE "
                        "distinct codes; below it gate_score=-inf). Default 8 was calibrated on "
                        "SPECTRO (192-768 tok/window); the low-token families (slow-TS 4 tok, "
                        "fast-TS 5 tok) have no historical n_distinct_codes record — calibrate "
                        "at launch rather than trusting the spectro default.")
    p.add_argument("--fm_weight", type=float, default=None,
                   help="Override cfg.fm_weight (disc feature-matching folded into the "
                        "generator's recon reference on the adaptive path). 0 = fully "
                        "GAN-free — v5 lesson (2026-08-04): fm at its 1.0 default kept "
                        "dragging the generator toward a SATURATED discriminator's "
                        "speckle features even with adversarial_weight 0.")
    p.add_argument("--consistency_weight", type=float, default=None,
                   help="Override cfg.consistency_weight (shift-invariance strength).")
    # --- fast-TS (filterscopes) RAW-SAMPLE codec knobs (2026-09-03 redesign) ---------- #
    p.add_argument("--patch_w", type=int, default=None,
                   help="FAST-TS: raw samples per token. n_tok = window(500) // patch_w, so the "
                        "TOKEN COUNT and hence the frame size is set here. Divisors of 500 -> "
                        "patch_w 100/50/20/10/5 = 5/10/25/50/100 tokens = 0.0125/0.0249/0.0623/"
                        "0.1246/0.2491 bits per raw value (1000-code FSQ, 8x500 = 4000 values). "
                        "Default 20 (25 tokens; world-model frame 1017 - 5 + 25 = 1037, +2.0%%).")
    p.add_argument("--stem_layers", type=int, default=None,
                   help="FAST-TS: number of stride-1 pre-patch conv-stem layers (default 2; "
                        "0 = plain linear patches, the control arm). The stem is the recorded "
                        "alternative to spike-weighted losses: it gives every patch a receptive "
                        "field that OVERLAPS its neighbours so a spike on a patch seam is seen "
                        "whole.")
    p.add_argument("--stem_channels", type=int, default=None,
                   help="FAST-TS: conv-stem width (default 64).")
    p.add_argument("--stem_kernel", type=int, default=None,
                   help="FAST-TS: conv-stem kernel, must be ODD (default 15 = +-0.7 ms at "
                        "10 kHz). Larger = more cross-patch overlap.")
    p.add_argument("--fastts_target", type=str, default=None, choices=["envelope", "raw"],
                   help="FAST-TS codec target. DEFAULT 'envelope' = the ORIGINAL ELM-activity-"
                        "envelope codec, byte-identical to the pre-2026-09-03 build so every "
                        "existing checkpoint still loads. 'raw' = the SAMPLE-WISE 10 kHz "
                        "waveform codec (implied by any of --patch_w / --stem_* / "
                        "--recon_loss / --ssim_*).")
    p.add_argument("--ssim_weight", type=float, default=None,
                   help="FAST-TS: weight on the 1-D structural term (1 - contrast*structure "
                        "of the SSIM along the SAMPLE axis). DEFAULT 0 = OFF (the control "
                        "arm). This is the guard against variance collapse: nRMSE's minimiser "
                        "is the conditional mean, so it REWARDS shrinking the output amplitude "
                        "(measured: a 0.4x-shrunk target scores nRMSE 0.6000 with std_ratio "
                        "0.4000), and only the SSIM contrast factor sees that.")
    p.add_argument("--ssim_win", type=int, default=None,
                   help="FAST-TS: box-window length in SAMPLES for the 1-D SSIM (default 17 "
                        "= 1.7 ms at 10 kHz, about one ELM burst period). Must match "
                        "gate._FASTTS_SSIM_WIN for loss and metric to be the same quantity.")
    p.add_argument("--recon_loss", type=str, default=None,
                   choices=["nrmse", "nmse", "mse", "l1", "huber"],
                   help="FAST-TS: sample-wise reconstruction loss. DEFAULT 'nrmse' IS the "
                        "reported metric (per-(window, channel) RMSE / std(target)). Plain "
                        "'mse' is NOT a surrogate for it: the RMS window-mean offset is ~25x "
                        "the ~0.0073 within-window std, so an MSE optimum emits a FLAT "
                        "window, which reads mse 5e-05 but nRMSE exactly 1.0000. 'nmse' is "
                        "the un-rooted form (worse conditioned); 'mse'/'l1'/'huber' are the "
                        "un-normalized controls.")
    # --- FAST-TS GAIN-SHAPE decomposition (2026-09-03) ------------------------------- #
    p.add_argument("--gain_shape", action="store_true",
                   help="FAST-TS ENVELOPE: code the window as level + sigma * unit-shape "
                        "instead of coding the raw envelope directly. 96.2%% of the "
                        "ELM-envelope variance is the per-(window, channel) DC LEVEL, and a "
                        "trivial encoder that transmits ONLY that level at the codec's own "
                        "49.83-bit budget scores pooled nRMSE 0.1935 against the shipped "
                        "codec's 0.5252. With --gain_shape the leading --gain_tokens tokens "
                        "carry the level (+sigma) through a dedicated MLP->FSQ path and the "
                        "decoder's shape branch is MEAN-REMOVED, so the per-window-mean "
                        "anchor becomes a FLOOR the codec cannot fall below rather than a bar "
                        "it fails. DEFAULT OFF (bit-identical control arm).")
    p.add_argument("--gain_tokens", type=int, default=None,
                   help="FAST-TS: tokens (of 5) reserved for the GAIN code; the rest carry "
                        "shape. TOKEN COUNT AND VOCAB ARE UNCHANGED (5 x 1000), so the "
                        "world-model frame layout is untouched. --gain_tokens 5 = LEVEL ONLY "
                        "(no shape path), the learned-VQ analogue of the rate-matched "
                        "baseline. Default 1.")
    p.add_argument("--gain_scale", dest="gain_scale", action="store_true", default=None,
                   help="FAST-TS: transmit the per-(window, channel) AC scale sigma with the "
                        "level and normalize the decoded shape to unit std, so sigma ALONE "
                        "sets the within-window amplitude (textbook gain-shape VQ). This is "
                        "the structural attack on the measured 36%% burst-height retention. "
                        "ON by default when --gain_shape is set.")
    p.add_argument("--no_gain_scale", dest="gain_scale", action="store_false",
                   help="FAST-TS: level-only gain; the shape path's amplitude is free (and "
                        "may shrink toward the nRMSE-minimising conditional mean).")
    p.add_argument("--gain_weight", type=float, default=None,
                   help="FAST-TS: weight on the direct |pred - (level, log1p sigma)| "
                        "supervision of the gain head (part of the reconstruction "
                        "reference). Default 1.0; 0 = train the gain path through the "
                        "reconstruction term only.")
    # --- NVIDIA Spectral Codec port (arXiv 2406.05298): STFT geometry + recipe knobs ---
    p.add_argument("--stft_n_fft", type=int, default=None,
                   help="SPECTRO: per-codec STFT n_fft (default 1024 = config.STFT_N_FFT). "
                        "512 halves the freq resolution to the canonical 50%%-overlap Hann COLA "
                        "grid (freq_bins 256, bin width 977 Hz); the 0-250 kHz band is "
                        "UNCHANGED (Nyquist is set by STFT_FS). Requires --freq_bins to match "
                        "n_fft//2 and --patch_f halved to keep n_tok at 192.")
    p.add_argument("--stft_hop", type=int, default=None,
                   help="SPECTRO: per-codec STFT hop (default 256 = config.STFT_HOP).")
    p.add_argument("--freq_bins", type=int, default=None,
                   help="SPECTRO: codec input freq bins (default 512). Set to n_fft//2 "
                        "(DC dropped) to cover the full band with no crop.")
    p.add_argument("--time_frames", type=int, default=None,
                   help="SPECTRO: codec input time frames (default 96).")
    p.add_argument("--ms_ssim_weight", type=float, default=None,
                   help="Weight on the 1-MS-SSIM reconstruction term (losses.ms_ssim_loss), the "
                        "DIFFERENTIABLE twin of the gate.ms_ssim ranking metric. 0 = OFF "
                        "(default, byte-identical). SSIM's contrast factor collapses on a blur, "
                        "which no L-p term penalises.")
    p.add_argument("--ms_ssim_win", type=int, default=None,
                   help="Local window (bins) for the MS-SSIM term (default 7).")
    p.add_argument("--discriminator", type=str, default=None, choices=["patch", "multiscale"],
                   help="SPECTRO discriminator family. 'patch' = FreqAwarePatchGAN (default). "
                        "'multiscale' = MultiScaleSpectroGAN: judges the WHOLE spectrogram at "
                        "1x/2x/4x with a single score per scale, so a decoder cannot satisfy it "
                        "by tiling one fixed patch texture (the measured patch lattice, recon "
                        "61.02 vs GT 1.14). The paper pairs a multi-period with a multi-scale "
                        "complex-STFT discriminator, i.e. global + multi-resolution, not patch.")
    p.add_argument("--multiscale_recon_scales", type=str, default=None,
                   help="Comma list of avg-pool kernel sizes for the multi-resolution recon L1 "
                        "(default 2,4). Include 1 to score FULL resolution: at the default every "
                        "term is a low-passed copy, so a large --multiscale_recon_weight rewards "
                        "matching the blur. The paper varies STFT window length instead.")
    p.add_argument("--conv_dec_base_ch", type=int, default=None,
                   help="SPECTRO conv decoder: proj_in channel width (default 128). The paper "
                        "uses 1024 initial channels for a 55M decoder vs a 10M encoder (5.5:1).")
    p.add_argument("--conv_dec_res_blocks", type=int, default=None,
                   help="SPECTRO conv decoder: residual conv blocks per upsample stage (2).")
    p.add_argument("--conv_dec_min_ch", type=int, default=None,
                   help="SPECTRO conv decoder: channel floor for the halving schedule (64).")
    p.add_argument("--disc_update_every", type=int, default=None,
                   help="Run the discriminator step once every N generator steps (default 1 = "
                        "every step). Paper: 2.")
    p.add_argument("--adam_beta1", type=float, default=None,
                   help="Adam beta1 for BOTH optimizers (default 0.9). Paper: 0.8.")
    p.add_argument("--adam_beta2", type=float, default=None,
                   help="Adam beta2 for BOTH optimizers (default 0.999). Paper: 0.99.")
    p.add_argument("--lr_decay_gamma", type=float, default=None,
                   help="Exponential LR decay factor applied every --lr_decay_every steps: "
                        "lr = lr0 * gamma**(step/every). 1.0 = OFF (default). Paper: 0.998.")
    p.add_argument("--lr_decay_every", type=int, default=None,
                   help="Step period for --lr_decay_gamma (default 1000, the paper's period).")
    p.add_argument("--patch_f", type=int, default=None,
                   help="override SpectroCodecConfig patch_f/patch_t (24-tok = 64/32; "
                        "192-tok = 16/16).")
    p.add_argument("--patch_t", type=int, default=None,
                   help="override SpectroCodecConfig patch_f/patch_t (24-tok = 64/32; "
                        "192-tok = 16/16).")
    p.add_argument("--decoder", type=str, default=None, choices=["linear", "conv"],
                   help="Spectro decoder family: 'linear' (default; transformer + nn.Linear "
                        "to_pixels, SMOOTH patches) or 'conv' (HiFi-GAN/VQGAN 2D transposed-conv "
                        "upsampler, SpectroConvDecoder — synthesizes turbulent texture). "
                        "Spectro modalities only.")
    p.add_argument("--refine_depth", type=int, default=None,
                   help="VIDEO or SPECTRO: depth of the decoder's residual conv refinement head "
                        "(stride-1 kernel-3, zero-init final conv, so it starts as an exact "
                        "identity). Gives the decoder full-resolution CROSS-PATCH context so its "
                        "texture is no longer one shared basis tiled on the patch lattice — the "
                        "measured checkerboard (gate.patch_lattice_metrics; mhr recon 61.6 vs GT "
                        "1.15). 0/unset = off (byte-identical decoder).")
    p.add_argument("--refine_hidden", type=int, default=None,
                   help="Channel width of the refinement head's hidden convs (default 64). "
                        "Only meaningful with --refine_depth > 1.")
    p.add_argument("--decoder_noise", action="store_true",
                   help="SPECTRO: StyleGAN-style per-pixel Gaussian noise with a learned "
                        "per-channel scale (ZERO-INIT, so a fresh decoder is an exact "
                        "identity) added at full resolution. Gives the decoder an APERIODIC "
                        "source of the high-frequency texture the adversarial + FM terms "
                        "reward, which a deterministic decoder can only supply as a tiled "
                        "(lattice) basis. OFF = byte-identical.")
    p.add_argument("--refine_dilated", action="store_true",
                   help="SPECTRO: use dilations 1,2,4,8,... in the refinement head so depth D "
                        "reaches a receptive field of 2^(D+1)-1 (D=4 -> 31 >= patch_f=16, i.e. a "
                        "whole patch plus its neighbours) instead of 2D+1.")
    # ---- prod-recipe knobs (the ONLY spectro recipe that survives the multi-shot collapse
    # test: prod_ece = fsq_levels [8,8,8,8,8]/cb=32768, entropy 0.1, adv_clamp 10000,
    # adv_warmup 0, adversarial_weight 1.0). The current SpectroCodecConfig defaults drifted to
    # the collapsing d2 values (cb=1000, entropy 1.0, clamp 50); these let a launch reproduce the
    # prod recipe WITHOUT editing the defaults. Applied AFTER apply_activity_overrides so the CLI
    # always wins (co2/mhr _activity_overrides would otherwise force warm=1500/advw=0.5). ----
    p.add_argument("--channel_groups", type=int, default=None,
                   help="CHANNEL-FACTORIZED tokens (config.py:79, the ece capacity lever). G>1 "
                        "splits channels into G groups so a token carries C/G channels instead "
                        "of all C: patch_dim=(C/G)*patch_f*patch_t, n_tok=G*nf*nt. ece C=40 at "
                        "g=2: 10240->5120 numbers per token, 192->384 tokens. Must divide C.")
    p.add_argument("--fsq_levels", type=str, default=None,
                   help="Comma-separated FSQ levels, e.g. '8,8,8,8,8' (cb=32768, prod recipe) "
                        "vs the collapsing d2 default '8,5,5,5' (cb=1000). Applied before codec build.")
    p.add_argument("--adaptive_adv_clamp", type=float, default=None,
                   help="Override cfg.adaptive_adv_clamp (prod=10000; drifted default=50).")
    p.add_argument("--adv_warmup_steps", type=int, default=None,
                   help="Override cfg.adv_warmup_steps (prod=0).")
    p.add_argument("--adversarial_weight", type=float, default=None,
                   help="Override cfg.adversarial_weight (prod=1.0).")
    p.add_argument("--skip_activity_override", action="store_true",
                   help="Skip the per-modality _activity_overrides entirely (needed to reproduce "
                        "the clean prod recipe for co2/mhr, whose overrides force warm/advw/bias).")
    p.add_argument("--compute_logpow_stats", type=str, default=None,
                   help="Compute dataset-level per-freq log-power stats (COMPOSE space, i.e. "
                        "after the raw z-score for co2) over the resolved train shots, write "
                        "them to this path, and exit without training. Spectro only.")
    p.add_argument("--logpow_stats_path", type=str, default=None,
                   help="Path to codec_<mod>_perfreq_stats.pt; enables per-freq log-power "
                        "z-standardization (the thin-modality mean-collapse fix) and DISABLES "
                        "raw-std since the stats are in raw-clipped space.")
    p.add_argument("--instance_norm_quantize", type=float, default=None,
                   help="Shift-robust instance norm: quantization step for the window "
                        "stats (sd on a log2 grid, mu in units of q*sd). 0.5 recommended; "
                        "unset keeps plain instance norm. Only active with "
                        "--input_instance_norm.")
    # --- VIDEO + SPECTRO missing-data exclusion; all default OFF = byte-identical.
    p.add_argument("--mask_missing", action="store_true",
                   help="VIDEO/SPECTRO: emit a REAL per-channel validity mask and honour it in "
                        "EVERY loss term. VIDEO: per-(channel, frame) from the loader's "
                        "channel_valid. SPECTRO: per-(channel, STFT-frame) from "
                        "data.spectro_frame_mask (NaN projected exactly like the production "
                        "data_loader._raw_to_frame_mask, PLUS all-zero supports, which is how "
                        "the loader represents an absent channel). Masked terms: pixel anchor, "
                        "ADVERSARIAL + FEATURE-MATCHING (missing windows are dropped from the "
                        "discriminator instead of being shown to it as 'real'), multiscale / "
                        "freq-grad / MS-SSIM, the FSQ entropy statistic, shift-consistency and "
                        "the discriminator's own step. Default OFF = no mask anywhere, which "
                        "is precisely the 2026-09-03 audit finding on BOTH families.")
    p.add_argument("--require_live_channels", action="store_true",
                   help="VIDEO/SPECTRO: re-draw any window in which a channel was dead, so the "
                        "codec only sees fully-populated windows (video: 19.0%% of lower / "
                        "16.9%% of upper channel-slots are dead even among shots that have SOME "
                        "video).")
    p.add_argument("--spectro_presence", type=str, default=None, choices=["off", "any", "all"],
                   help="SPECTRO whole-shot presence filter from the precomputed per-channel "
                        "liveness cache (no HDF5 scan): 'any' keeps shots with >=1 live channel "
                        "for this modality, 'all' keeps only fully-populated shots, 'off' "
                        "(default) keeps every shot -- including the ones whose HDF5 group is an "
                        "empty (C, 1) stub, which can only ever yield the eps-floor constant "
                        "last-resort pair.")
    p.add_argument("--video_presence", type=str, default=None, choices=["off", "any", "all"],
                   help="VIDEO whole-shot presence filter from the precomputed liveness cache "
                        "(no HDF5 scan): 'any' keeps shots with >=1 live camera for this "
                        "divertor, 'all' keeps only fully-populated shots, 'off' (default) "
                        "keeps every shot -- which means 51.3%% (lower) / 68.6%% (upper) of the "
                        "stream is an ALL-ZERO clip.")
    p.add_argument("--input_instance_norm", action="store_true",
                   help="Per-window instance z-score (mean~0/std~1) on the log-power encoder input. "
                        "ROOT-CAUSE fix for the co2 encoder death: strips the large DC offset that "
                        "saturates the FSQ tanh bound. Composes with raw-std (which un-clips first).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = build_arg_parser().parse_args(argv)

    ddp = _DDPState()
    device = ddp.device

    is_video = args.modality in VIDEO_MODALITIES
    is_slowts = args.modality in SLOWTS_MODALITIES
    is_fastts = args.modality == FASTTS_MODALITY

    # cfg with the modality's real channel count baked in (all ranks agree).
    channels = modality_channels(args.modality)
    if is_video:
        cfg = VideoCodecConfig(channels=channels, divertor=video_divertor(args.modality))
    elif is_slowts:
        cfg = slowts_codec_cfg(args.modality, channels)
    elif is_fastts:
        # fast-TS (filterscopes) ELM-envelope codec — built + trained by the fastts_train
        # module (lazy import: it imports FROM this module, so a top-level import is circular).
        # DEFAULT = the ORIGINAL envelope codec (so old checkpoints/pickles keep loading).
        # The 2026-09-03 SAMPLE-WISE codec is OPT-IN via --fastts_target raw, which also flips
        # the objective defaults (see config.fastts_raw_config). Passing any raw-only geometry
        # flag implies it, so --patch_w alone does the right thing.
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
        # --- STFT GEOMETRY first (it determines freq_bins, hence every later shape) ------
        # Defaults leave cfg.stft_n_fft / cfg.stft_hop at the module globals (1024 / 256), so
        # omitting these flags reproduces every existing codec bit-for-bit.
        for _g in ("stft_n_fft", "stft_hop", "freq_bins", "time_frames"):
            _v = getattr(args, _g, None)
            if _v is not None:
                setattr(cfg, _g, int(_v))
        # Optional patch-size override (spectro only). 24-tok = patch_f 64 / patch_t 32 (survives
        # the 192-tok mean-collapse); 192-tok = 16/16 (the collapsing default). Applied BEFORE the
        # standardization / activity overrides / codec are built so n_tok is fixed up front.
        if args.patch_f is not None:
            cfg.patch_f = args.patch_f
        if args.patch_t is not None:
            cfg.patch_t = args.patch_t
        # GEOMETRY CONTRACT. freq_bins must be exactly the DC-dropped bin count of the chosen
        # n_fft, else the crop in data._crop_pad_freq_time silently throws away the top of the
        # band (or eps-pads a band that does not exist) — the user requires the FULL 0-250 kHz
        # range. n_tok is printed because the Phase-B frame layout budgets 192 tokens/spectro
        # modality: halving n_fft (512 -> 256 bins) and halving patch_f (16 -> 8) keeps it at
        # 32 x 6 = 192, so the world-model frame stays 1017 tokens.
        # __post_init__ ran at CONSTRUCTION with the defaults, so the divisibility contract has
        # to be re-checked after the overrides — otherwise a bad combination only surfaces as an
        # opaque einops rearrange error deep in the encoder.
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
        # DECODER family override (spectro only): 'conv' swaps the linear to_pixels unpatchify for
        # the HiFi-GAN/VQGAN 2D transposed-conv upsampler (SpectroConvDecoder). Default 'linear'
        # is byte-identical. Applied before the codec is built so the right decoder is constructed.
        if args.decoder is not None:
            cfg.decoder = args.decoder
            # size the conv decoder BEFORE the print below, so the logged base_ch/res_blocks
            # are the ones actually used (they were previously applied later, and the log line
            # advertised the stale defaults).
            for _c in ("conv_dec_base_ch", "conv_dec_res_blocks", "conv_dec_min_ch"):
                _cv = getattr(args, _c, None)
                if _cv is not None:
                    setattr(cfg, _c, int(_cv))
            if ddp.is_main:
                print(f"[train_codec] decoder={cfg.decoder} "
                      f"(base_ch={cfg.conv_dec_base_ch}, res_blocks={cfg.conv_dec_res_blocks})"
                      if cfg.decoder == "conv" else f"[train_codec] decoder={cfg.decoder}")
        # StyleGAN-style decoder noise input (spectro only; zero-init => exact no-op at step 0).
        if getattr(args, "decoder_noise", False):
            if cfg.decoder == "conv":
                raise SystemExit(
                    "--decoder_noise is wired into the LINEAR SpectroDecoder only; "
                    "it is not implemented for SpectroConvDecoder."
                )
            cfg.decoder_noise = True
            if ddp.is_main:
                print("[train_codec] decoder_noise=ON (per-channel learned scale, zero-init)")
        # global per-freq input standardization for THIN spectro modalities (co2); lifts the
        # sparse activity out of the large near-constant log-power mean so the codec stops
        # collapsing to one code. No-op for ece/bes/mhr (rich, naturally O(1)).
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
    if getattr(args, "refine_depth", None) is not None or getattr(args, "refine_hidden", None) \
            is not None or getattr(args, "refine_dilated", False):
        # VIDEO + SPECTRO: only those two configs carry the refinement-head fields.
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
    # VIDEO missing-data knobs. Guarded so a typo on a non-video modality fails LOUD rather
    # than silently doing nothing (the persist-arch-flags lesson).
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

    # anti-collapse overrides for the 4 collapsing codecs (co2 / tangtv_lower / ts_core_density /
    # filterscopes). No-op for every other (already-working) modality — see _activity_overrides.
    # --skip_activity_override bypasses them entirely so the clean prod recipe can be reproduced
    # for co2/mhr (whose overrides otherwise force adv_warmup=1500 / adversarial_weight=0.5 / bias
    # that FIGHT the prod recipe — the multi-shot survivor uses warm=0 / advw=1.0 / no bias).
    if args.skip_activity_override:
        if ddp.is_main:
            print(f"[train_codec] _activity_overrides SKIPPED for {args.modality} "
                  f"(--skip_activity_override)")
    else:
        apply_activity_overrides(cfg, args.modality, log_fn=(print if ddp.is_main else None))

    # prod-recipe CLI overrides — applied LAST so they win over both the config defaults AND
    # _activity_overrides. fsq_levels changes the codebook, so it must land before the codec is
    # built (it is: construction happens after this block). No-op if the flag is None / cfg lacks
    # the field (video / slow-TS families).
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
    # NVIDIA Spectral Codec recipe knobs. All defaults are today's values, so omitting every
    # flag is byte-identical. (adam_* / lr_decay_* / disc_update_every live on the cfg so the
    # checkpoint records the recipe it was trained under — see the persist-arch-flags lesson.)
    for _knob in ("conv_dec_base_ch", "conv_dec_res_blocks", "conv_dec_min_ch",
                  "disc_update_every", "lr_decay_every"):
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
    for _knob in ("ms_ssim_weight",):
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
                  "joint_entropy_ramp_steps", "fsq_noise_dropout"):
        if getattr(args, _knob, None) is not None and not hasattr(cfg, _knob):
            raise SystemExit(
                f"--{_knob} is a SPECTRO-only anti-collapse knob (it lives on "
                f"SpectroCodecConfig); {args.modality}'s {type(cfg).__name__} has no such field."
            )
    for _knob in ("patch_w", "stem_layers", "stem_channels", "stem_kernel", "recon_loss",
                  "ssim_weight", "ssim_win", "fastts_target", "gain_tokens", "gain_scale",
                  "gain_weight"):
        if getattr(args, _knob, None) is not None and not is_fastts:
            raise SystemExit(
                f"--{_knob} is a FAST-TS-only knob (it lives on FastTSCodecConfig); "
                f"{args.modality} is not the fast-TS modality."
            )
    if getattr(args, "gain_shape", False) and not is_fastts:
        raise SystemExit(
            "--gain_shape is a FAST-TS ENVELOPE-only decomposition (level = mean over the "
            f"envelope bins); {args.modality} is not the fast-TS modality."
        )

    # per-freq LOG-POWER z-standardization (the THIN-modality mean-collapse fix; spectro ONLY).
    # Applied AFTER apply_spectro_standardization + the fsq/prod-recipe overrides. The per-freq z
    # COMPOSES with the raw z-score (raw-std stays as apply_spectro_standardization set it): the
    # stats MUST be computed in the space the codec actually SEES, i.e. AFTER raw-std for the
    # modalities in _SPECTRO_STANDARDIZE_SIGNALS (co2). The old REPLACE convention (disable
    # raw-std, stats in raw-clipped space) is BROKEN for co2 — measured 2026-07-31: without
    # raw-z, co2 log-power is 100% clipped at data._LOG_CEIL=20 (mean 20.00, per-freq std med
    # 0.0000), so replace-convention stats are degenerate and zero the signal. No-op unless
    # --logpow_stats_path is given. Guarded to the spectro branch (video/slow-TS/fast-TS lack
    # the log-power fields).
    if (not (is_video or is_slowts or is_fastts)) and getattr(args, "logpow_stats_path", None):
        import torch as _t
        _st = _t.load(args.logpow_stats_path, map_location="cpu", weights_only=False)
        # NOTE the stats files are NOT uniform: the legacy ones store torch TENSORS, the ones
        # compute_logpow_stats writes today store nested LISTS. `if _st.get("mean")` on a
        # tensor raises "Boolean value of Tensor with more than one value is ambiguous", so
        # normalise through torch first and read the trailing (frequency) axis.
        _sf = int(_t.as_tensor(_st["mean"]).shape[-1]) if "mean" in _st else -1
        if _sf != cfg.freq_bins:
            raise SystemExit(
                f"--logpow_stats_path {args.logpow_stats_path} has F={_sf} but this codec's "
                f"freq_bins={cfg.freq_bins} (stft_n_fft={cfg.stft_n_fft}). Per-freq stats are "
                f"only valid on the grid they were computed on — regenerate with "
                f"--compute_logpow_stats and the SAME --stft_n_fft/--freq_bins."
            )
        cfg.logpow_freq_mean = _t.as_tensor(_st["mean"], dtype=_t.float32).tolist()
        cfg.logpow_freq_std = _t.as_tensor(_st["std"], dtype=_t.float32).tolist()
        cfg.logpow_standardize = True
        if ddp.is_main:
            print(f"[train_codec] per-freq log-z ON from {args.logpow_stats_path} "
                  f"(C={len(cfg.logpow_freq_mean)}, F={len(cfg.logpow_freq_mean[0])}); "
                  f"COMPOSED with raw-std (raw z {'ON' if cfg.input_standardize else 'off'})",
                  flush=True)

    # Per-window instance z-score (root-cause fix for the co2 DC-offset -> FSQ-saturation collapse).
    # Applied LAST in log_power_stft (after raw-std un-clips), so it centers + unit-scales every
    # window to ece-like stats (mean~0/std~1) and the encoder no longer saturates the tanh bound.
    if (not (is_video or is_slowts or is_fastts)) and getattr(args, "input_instance_norm", False):
        cfg.input_instance_norm = True
        if getattr(args, "instance_norm_quantize", None) is not None:
            cfg.instance_norm_quantize = float(args.instance_norm_quantize)
        if ddp.is_main:
            print("[train_codec] per-window instance z-score ON (mean~0/std~1) — "
                  f"DC-offset / FSQ-saturation fix; quantize={cfg.instance_norm_quantize} "
                  "(>0 = shift-robust piecewise-constant stats)", flush=True)

    # fast-TS geometry + SCALE FIX. patch_w/stem_* set the RAW-SAMPLE codec's token count and
    # receptive field; the per-channel raw mean/std put the ~1e15 raw signal on the ~O(1) scale
    # the network is sized for (--stats_path='' disables; None => canonical DEFAULT_STATS_PATH).
    # No-op for every other modality.
    if is_fastts:
        from dataclasses import replace as _dc_replace

        from .fastts_train import DEFAULT_STATS_PATH, load_fastts_channel_stats
        # GEOMETRY FIRST: patch_w sets n_tok (and therefore the world-model frame size), and
        # __post_init__ validates the divisibility, so rebuild the dataclass rather than
        # mutating a field the invariants depend on.
        _geom = {k: v for k, v in (
            ("patch_w", args.patch_w),
            ("stem_layers", args.stem_layers),
            ("stem_channels", args.stem_channels),
            ("stem_kernel", args.stem_kernel),
        ) if v is not None}
        # GAIN-SHAPE is part of the geometry (it re-partitions the SAME 5 tokens into a gain
        # group and a shape group), so it goes through the same dataclass rebuild -> its
        # __post_init__ validates gain_tokens against n_env_patch.
        _gs = {k: v for k, v in (
            ("gain_shape", True if getattr(args, "gain_shape", False) else None),
            ("gain_tokens", args.gain_tokens),
            ("gain_scale", args.gain_scale),
            ("gain_weight", args.gain_weight),
        ) if v is not None}
        if _gs and not getattr(args, "gain_shape", False):
            raise SystemExit(
                "--gain_tokens/--gain_scale/--no_gain_scale/--gain_weight require "
                "--gain_shape (they configure the gain-shape decomposition, which is OFF "
                "by default)."
            )
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

    # slow-TS SCALE FIX — inject the per-signal preprocessing (log_standardize / standardize) +
    # per-channel mean/std so the codec input is standardized to ~O(1) (like the FM model sees)
    # instead of the unstandardized raw that collapses the high-magnitude Thomson density signals
    # (ts_core_density ~1e19 -> 1 code, env_corr=NaN). --stats_path='' disables (raw path,
    # debugging only); None => the canonical DEFAULT_STATS_PATH. No-op for every other codec.
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

    # resolve the train + disjoint eval shot lists (rank 0 discovers; the list is
    # deterministic from data_dir + sort so every rank derives the same split).
    if args.shots is not None:
        all_shots = [s.strip() for s in args.shots.split(",") if s.strip()]
    else:
        all_shots = spike.discover_shots(args.data_dir)

    # Fix (a): drop whole shots that carry no data for this modality (co2 only; no-op for
    # every other modality — see _PRESENCE_FILTER_SIGNALS). Runs BEFORE the eval/train split
    # so both draw from present shots. rank-0 scans + caches + broadcasts (DDP-safe); the
    # cache is keyed by (paths, signal) so a pre-warmed sidecar avoids any job-time scan.
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

    # SPECTRO per-channel presence filter (precomputed liveness cache). Runs BEFORE the
    # eval/train split so BOTH draw from shots that actually recorded this diagnostic.
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

    # VIDEO presence filter (per divertor, from the precomputed liveness cache). Runs BEFORE
    # the eval/train split so BOTH draw from shots that actually have this divertor's cameras.
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

    # STATS-GENERATION mode (per-freq log-z promotion prep): compute dataset-level stats
    # in the COMPOSE space over the resolved train shots (presence-filtered for co2),
    # write, and exit — no training, no DDP collectives (run single-process).
    if getattr(args, "compute_logpow_stats", None):
        if is_video or is_slowts or is_fastts:
            raise SystemExit("--compute_logpow_stats is spectro-only")
        # The stats are (C, F) in the codec's OWN log-power space, so they MUST be computed on
        # the SAME STFT grid the codec will use. A 512-bin file is silently wrong for a 256-bin
        # codec (broadcast error at best, wrong per-freq normalisation at worst), hence the
        # geometry passthrough.
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
    # LENGTHS-CACHE KEY MUST TRACK THE SHOT LIST (2026-09-04).
    #
    # TokamakMultiFileDataset._load_or_compute_lengths only reuses the sidecar when its STORED
    # PATH LIST MATCHES the current hdf5_paths. A presence filter changes that list, so the
    # shared `codec_<mod>_lengths.pt` MISSES and every arm cold-scans every shot -- the exact
    # cold-scan hazard the N_SHOTS=9000 rule exists to avoid, arriving through a new door.
    #
    # MEASURED (job 5416298, 2026-09-04): with --spectro_presence any, mirnov resolved 8693
    # shots (cache has 8737) and all four mirnov arms spent the ENTIRE 2 h leg scanning --
    # zero gates, zero checkpoints. The three masked bes arms (3279 shots) reached step 14000
    # while bes_nomask, whose 8737-shot list HITS the cache, ran the full 29999. The apparent
    # "masking is slow" and "nomask is fast" effects were this cache miss, not the mask.
    #
    # Giving the filtered list its OWN sidecar makes the scan a one-off: it is cold once, then
    # warm for every later leg. It never writes the shared file, so unfiltered runs are
    # byte-identical and the production cache cannot be clobbered.
    _presence_on = (
        getattr(args, "spectro_presence", None) not in (None, "off")
        or getattr(args, "video_presence", None) not in (None, "off")
    )
    _suffix = "_presence" if _presence_on else ""
    lengths_cache_path = args.lengths_cache_dir and (
        Path(args.lengths_cache_dir) / f"codec_{args.modality}{_suffix}_lengths.pt"
    )
    if is_fastts:
        # fast-TS trainer has a distinct signature (modality is keyword-only; no positional
        # modality). Reuse its train function verbatim — do NOT duplicate the training logic.
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
