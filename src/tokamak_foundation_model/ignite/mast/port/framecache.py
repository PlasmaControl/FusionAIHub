"""Encode MAST cache-v1 shots into dev-peter's IGNITE Phase-B frame-code cache format.

Each cached shot ``{shot}.pt`` holds:
  - ``codes``: ``{modality: int32[n_frames, n_tok * n_val]}`` (bin indices in
    ``0..n_bins-1``, ``missing_code = n_bins`` for partially invalid tokens
    within a present signal, or ``_PLACEHOLDER_CODE = 0`` across all frames for
    a signal absent/all-invalid in the shot).
  - ``actuators``: ``float16[n_frames, A]`` (per-frame mean in model units,
    z-scored with global training-set mean/sd; invalid samples filled with 0).
  - ``actuator_valid``: ``bool[n_frames, A]`` (True where >=1 sample in the
    frame was valid).
  - ``n_frames``: ``int``.
  - ``vocabs``: ``{modality: int}`` (``n_bins + 1`` so ``missing_code = n_bins``
    never collides with bin 0 or MaskGIT's ``[MASK] = codebook_size``).
  - ``t0``: ``float`` (shot time [s] of frame 0).
  - ``present``: ``{modality: bool}`` (exact ground-truth presence in the shot's
    frame range, used for ``_presence.json`` and ``--mask_absent``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from .. import cache, paths, registry, tasks, transform
from ..tokenizers import assign, codecs as tok_codecs, fit, frames
from ..tokenizers.learned import build as learned_build

DT_FRAME: float = frames.FRAME_S
MIN_CACHED_FRAMES: int = 8      # matches train_dynamics._MIN_CACHED_FRAMES
PLACEHOLDER_CODE: int = 0       # matches train_dynamics._PLACEHOLDER_CODE


@dataclass(frozen=True)
class ActuatorStats:
    """Global training-set per-channel z-scoring statistics for MAST actuators."""

    names: tuple[str, ...]
    channels: tuple[int, ...]
    mu: np.ndarray
    sd: np.ndarray
    n_train_shots: int
    n_valid_frames: np.ndarray

    @property
    def total_dim(self) -> int:
        return int(sum(self.channels))

    def to_manifest_dict(self) -> dict:
        return {
            "names": list(self.names),
            "channels": [int(c) for c in self.channels],
            "total_dim": self.total_dim,
            "mu": [float(x) for x in self.mu],
            "sd": [float(x) for x in self.sd],
            "n_train_shots": int(self.n_train_shots),
            "n_valid_frames": [int(x) for x in self.n_valid_frames],
        }

    @classmethod
    def from_manifest_dict(cls, doc: Mapping) -> ActuatorStats:
        return cls(
            names=tuple(doc["names"]),
            channels=tuple(int(c) for c in doc["channels"]),
            mu=np.asarray(doc["mu"], dtype=np.float64),
            sd=np.asarray(doc["sd"], dtype=np.float64),
            n_train_shots=int(doc["n_train_shots"]),
            n_valid_frames=np.asarray(doc["n_valid_frames"], dtype=np.int64),
        )


def recoders_root() -> Path:
    """Default directory of stage-1c learned recoders."""
    return paths.stage1_dir() / "recoders"


def actuator_specs() -> tuple[tasks.SignalSpec, ...]:
    """Union of ``task.actuators`` across all 14 TokaMark tasks, sorted by name."""
    by_name: dict[str, tasks.SignalSpec] = {}
    for task_name in tasks.ALL_TASKS:
        spec = tasks.load_task(task_name)
        for act in spec.actuators:
            prev = by_name.setdefault(act.name, act)
            if prev != act:
                raise ValueError(
                    f"actuator {act.name} differs across tasks: {prev} vs {act}"
                )
    return tuple(by_name[name] for name in sorted(by_name))


def map_family(spec: tasks.SignalSpec) -> str:
    """Map a MAST signal to dev-peter's family vocabulary (``slowts``/``fastts``/``spectro``).

    In ``dynamics.py``, ``maskgit.py`` and ``frame_layout.py``, ``ModalitySpec.family``
    does not change any model computation (only metadata stored in checkpoints). We map
    by diagnostic cadence and representation:
      - Mirnov STFT spectrograms (``BandDCT``) -> ``"spectro"``
      - 50 kHz fast time series (SXR, D-alpha, saddle voltage) -> ``"fastts"``
      - <=5 kHz time series, 1D profiles (Thomson, LCFS), 2D flux maps (psi) -> ``"slowts"``
    """
    if spec.name in tasks.SPECTROGRAM_SIGNALS:
        return "spectro"
    if registry.family_of(spec) == "high_rate":
        return "fastts"
    return "slowts"


def build_mast_codecs(
    signals: Sequence[tasks.SignalSpec],
    cache_root: Path,
    fit_dir: Path,
    assignment_path: Path,
    recoders_dir: Path | None = None,
) -> tuple[dict[str, tok_codecs.Codec], dict[str, transform.ChannelStats], dict[str, assign.Setting]]:
    """Build stage-1b/1c codecs exactly as ``scripts/mast/run_tokenizer.py::_build_predictor``."""
    fits, _ = fit.load_fit(Path(fit_dir))
    assignment = assign.load_assignment(Path(assignment_path))
    stats = transform.load_stats(Path(cache_root) / "stats.npz")
    covered = [
        s for s in signals
        if s.name in assignment and s.name in fits and fits[s.name].n_examples > 0
    ]
    rec_root = Path(recoders_dir) if recoders_dir is not None else recoders_root()
    codecs = learned_build.build_codecs(covered, stats, fits, assignment, rec_root)
    return codecs, stats, assignment


def codec_vocab(codec: tok_codecs.Codec) -> tuple[int, int, int]:
    """Return ``(n_bins, missing_code, codebook_size)`` for one codec.

    Bins ``0 .. n_bins - 1`` are the codec's grid centres; ``missing_code = n_bins``
    is a dedicated token for partially invalid tokens within a present signal so
    missing data is never encoded as bin 0 (a real physical value).
    """
    if codec.grids is None:
        raise ValueError(f"codec {codec.name!r} has no grids")
    n_bins = int(codec.grids.n_bins)
    missing_code = n_bins
    return n_bins, missing_code, n_bins + 1


def plasma_frame_range(
    shot_data: Mapping[str, cache.CachedSignal | None],
    plasma_names: Sequence[str],
    dt_frame: float = DT_FRAME,
) -> tuple[float, float, int, int, int] | None:
    """First to last 5 ms frame where any non-actuator plasma signal is valid.

    Returns ``(t0_ref, t0, f_start, f_end, n_frames)`` where ``t0_ref`` is the
    minimum ``t0`` among valid plasma signals, ``f_start`` and ``f_end`` are
    inclusive frame indices relative to ``t0_ref``, ``n_frames = f_end - f_start + 1``
    and ``t0 = t0_ref + f_start * dt_frame``. Returns ``None`` if no plasma signal
    has any finite sample.
    """
    t0_ref = math.inf
    t_min = math.inf
    t_max = -math.inf
    for name in plasma_names:
        cs = shot_data.get(name)
        if cs is None:
            continue
        t_len = cs.values.shape[-1]
        if t_len == 0:
            continue
        any_fin = np.isfinite(cs.values).reshape(-1, t_len).any(axis=0)
        if not any_fin.any():
            continue
        if cs.t0 < t0_ref:
            t0_ref = float(cs.t0)
        i0 = int(np.argmax(any_fin))
        i1 = int(t_len - 1 - np.argmax(any_fin[::-1]))
        t_min = min(t_min, float(cs.t0) + i0 * float(cs.dt))
        t_max = max(t_max, float(cs.t0) + i1 * float(cs.dt))
    if not math.isfinite(t_min) or not math.isfinite(t0_ref):
        return None
    f_start = max(0, int(math.floor((t_min - t0_ref + 1e-9) / dt_frame)))
    f_end = int(math.floor((t_max - t0_ref + 1e-9) / dt_frame))
    if f_end < f_start:
        return None
    n_frames = f_end - f_start + 1
    t0 = t0_ref + f_start * dt_frame
    return t0_ref, t0, f_start, f_end, n_frames


def raw_actuator_frames(
    shot_data: Mapping[str, cache.CachedSignal | None],
    act_specs: Sequence[tasks.SignalSpec],
    t0: float,
    n_frames: int,
    dt_frame: float = DT_FRAME,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame mean of each actuator channel in model units.

    Returns ``(means, valid)`` with ``means`` float64 ``(n_frames, A)`` (NaN where
    no valid sample fell in the frame) and ``valid`` bool ``(n_frames, A)``.
    """
    cols_mean: list[np.ndarray] = []
    cols_valid: list[np.ndarray] = []
    for spec in act_specs:
        n_ch = int(math.prod(spec.values_shape))
        cs = shot_data.get(spec.name)
        if cs is None or cs.values.shape[-1] == 0:
            cols_mean.append(np.full((n_frames, n_ch), np.nan, dtype=np.float64))
            cols_valid.append(np.zeros((n_frames, n_ch), dtype=bool))
            continue
        vals = np.asarray(cs.values, dtype=np.float64).reshape(n_ch, -1)
        t_len = vals.shape[1]
        ts = float(cs.t0) + (np.arange(t_len, dtype=np.float64) + 0.5) * float(cs.dt)
        fi = np.floor((ts - float(t0)) / dt_frame).astype(np.int64)
        in_range = (fi >= 0) & (fi < n_frames)
        if not in_range.any():
            cols_mean.append(np.full((n_frames, n_ch), np.nan, dtype=np.float64))
            cols_valid.append(np.zeros((n_frames, n_ch), dtype=bool))
            continue
        fi_r = fi[in_range]
        v_r = vals[:, in_range]  # (n_ch, M)
        ok = np.isfinite(v_r)
        sums = np.zeros((n_frames, n_ch), dtype=np.float64)
        counts = np.zeros((n_frames, n_ch), dtype=np.int64)
        np.add.at(sums, fi_r, np.where(ok, v_r, 0.0).T)
        np.add.at(counts, fi_r, ok.T.astype(np.int64))
        valid = counts > 0
        means = np.divide(
            sums,
            counts,
            out=np.full((n_frames, n_ch), np.nan, dtype=np.float64),
            where=valid,
        )
        cols_mean.append(means)
        cols_valid.append(valid)
    return np.concatenate(cols_mean, axis=1), np.concatenate(cols_valid, axis=1)


def _chan_merge(
    n_a: np.ndarray,
    mean_a: np.ndarray,
    m2_a: np.ndarray,
    n_b: np.ndarray,
    mean_b: np.ndarray,
    m2_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically stable parallel merge of channel counts, means, and M2."""
    n_tot = n_a + n_b
    has_both = (n_a > 0) & (n_b > 0)
    only_b = (n_a == 0) & (n_b > 0)
    delta = np.where(has_both, mean_b - mean_a, 0.0)
    safe_tot = np.where(n_tot > 0, n_tot, 1.0)
    mean = np.where(
        has_both,
        mean_a + delta * (n_b / safe_tot),
        np.where(only_b, mean_b, mean_a),
    )
    m2 = np.where(
        has_both,
        m2_a + m2_b + (delta ** 2) * (n_a * n_b / safe_tot),
        np.where(only_b, m2_b, m2_a),
    )
    return n_tot, mean, m2


def shot_actuator_moments(
    shot_path: Path,
    act_specs: Sequence[tasks.SignalSpec],
    plasma_names: Sequence[str],
    stats_sha256: str | None = None,
    dt_frame: float = DT_FRAME,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-channel ``(count, mean, M2)`` of valid actuator frames for one shot."""
    needed = sorted(set(plasma_names) | {s.name for s in act_specs})
    shot_data = cache.read_shot(Path(shot_path), needed, stats_sha256=stats_sha256)
    span = plasma_frame_range(shot_data, plasma_names, dt_frame=dt_frame)
    if span is None:
        return None
    _t0_ref, t0, _f_start, _f_end, n_frames = span
    if n_frames < MIN_CACHED_FRAMES:
        return None
    means, valid = raw_actuator_frames(
        shot_data, act_specs, t0=t0, n_frames=n_frames, dt_frame=dt_frame
    )
    n_ch = means.shape[1]
    count = valid.sum(axis=0).astype(np.int64)
    has = count > 0
    sums = np.where(valid, means, 0.0).sum(axis=0)
    mean = np.divide(sums, count, out=np.zeros(n_ch, dtype=np.float64), where=has)
    diff = np.where(valid, means - mean[None, :], 0.0)
    m2 = (diff ** 2).sum(axis=0)
    return count, mean, m2


def finalize_actuator_stats(
    act_specs: Sequence[tasks.SignalSpec],
    moments: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> ActuatorStats:
    """Combine per-shot ``(count, mean, M2)`` tuples into global :class:`ActuatorStats`."""
    names = tuple(s.name for s in act_specs)
    channels = tuple(int(math.prod(s.values_shape)) for s in act_specs)
    total = sum(channels)
    acc_n = np.zeros(total, dtype=np.int64)
    acc_mean = np.zeros(total, dtype=np.float64)
    acc_m2 = np.zeros(total, dtype=np.float64)
    n_shots = 0
    for count, mean, m2 in moments:
        n_shots += 1
        acc_n, acc_mean, acc_m2 = _chan_merge(acc_n, acc_mean, acc_m2, count, mean, m2)
    var = np.divide(
        acc_m2, acc_n, out=np.zeros(total, dtype=np.float64), where=acc_n > 0
    )
    sd = np.sqrt(np.maximum(var, 0.0))
    sd = np.where((acc_n > 0) & (sd > 0.0), sd, 1.0)
    return ActuatorStats(
        names=names,
        channels=channels,
        mu=acc_mean,
        sd=sd,
        n_train_shots=n_shots,
        n_valid_frames=acc_n,
    )


def normalize_actuators(
    raw_means: np.ndarray,
    valid: np.ndarray,
    act_stats: ActuatorStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Z-score actuator frame means with global training stats; invalid -> 0."""
    z = np.where(
        valid,
        (raw_means - act_stats.mu[None, :]) / act_stats.sd[None, :],
        0.0,
    )
    finite = valid & np.isfinite(z)
    z = np.where(finite, z, 0.0)
    return (
        torch.from_numpy(z.astype(np.float16)),
        torch.from_numpy(finite.astype(bool)),
    )


def encode_signal_codes(
    codec: tok_codecs.Codec,
    cs: cache.CachedSignal | None,
    t0_ref: float,
    f_start: int,
    n_frames: int,
    dt_frame: float = DT_FRAME,
) -> tuple[torch.Tensor, bool]:
    """Encode one cached signal onto frames ``[f_start, f_start + n_frames)``.

    Returns ``(codes_int32[n_frames, n_tok * n_val], is_present)``.
    Absent or all-invalid signals return ``PLACEHOLDER_CODE`` frames and ``False``.
    Within a present signal, individual invalid tokens (``grids.index == -1``)
    receive ``missing_code = n_bins``.
    """
    n_flat = int(codec.n_tok * codec.n_val)
    if n_flat < 1:
        raise ValueError(f"codec {codec.name!r} has zero tokens per frame")
    if cs is None or cs.values.shape[-1] == 0 or not np.isfinite(cs.values).any():
        return (
            torch.full((n_frames, n_flat), PLACEHOLDER_CODE, dtype=torch.int32),
            False,
        )
    vals = codec.encode_domain(codec.domain_from_model(cs.values[None]))[0]
    f_sig = vals.shape[0]
    sig_offset = int(round((float(cs.t0) - float(t0_ref)) / dt_frame))
    aligned = np.full((n_frames, codec.n_tok, codec.n_val), np.nan, dtype=np.float64)
    # Source frames [src_lo, src_hi) map to destination frames [dst_lo, dst_hi)
    src_lo = max(0, f_start - sig_offset)
    src_hi = min(f_sig, f_start + n_frames - sig_offset)
    if src_hi > src_lo:
        dst_lo = src_lo + sig_offset - f_start
        dst_hi = dst_lo + (src_hi - src_lo)
        aligned[dst_lo:dst_hi] = vals[src_lo:src_hi]
    if not np.isfinite(aligned).any():
        return (
            torch.full((n_frames, n_flat), PLACEHOLDER_CODE, dtype=torch.int32),
            False,
        )
    _n_bins, missing_code, _vocab = codec_vocab(codec)
    idx = codec.grids.index(aligned)  # (n_frames, n_tok, n_val), -1 where NaN
    code_arr = np.where(idx < 0, missing_code, idx).reshape(n_frames, n_flat)
    return torch.from_numpy(code_arr.astype(np.int32)), True


def samples_for_frames(n_frames: int, dt: float, dt_frame: float = DT_FRAME) -> int:
    """Number of samples at sampling interval ``dt`` covering ``n_frames`` frames."""
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    est = max(1, int(math.floor(n_frames * dt_frame / dt)))
    while int(frames.frame_index(est, dt)[-1]) >= n_frames and est > 1:
        est -= 1
    while int(frames.frame_index(est + 1, dt)[-1]) < n_frames:
        est += 1
    return est


def decode_signal_codes(
    codec: tok_codecs.Codec,
    codes: torch.Tensor | np.ndarray,
    n_samples: int | None = None,
    dt_frame: float = DT_FRAME,
) -> np.ndarray:
    """Decode ``(F, n_tok * n_val)`` bin codes back to physical units ``(*values_shape, n_samples)``.

    Codes ``0 .. n_bins - 1`` map to the codec's grid centres; ``missing_code``
    (``>= n_bins`` or ``< 0``) maps to ``NaN``.
    """
    arr = codes.detach().cpu().numpy() if isinstance(codes, torch.Tensor) else np.asarray(codes)
    if arr.ndim != 2 or arr.shape[1] != codec.n_tok * codec.n_val:
        raise ValueError(
            f"{codec.name}: expected (F, {codec.n_tok * codec.n_val}), got {arr.shape}"
        )
    n_frames = int(arr.shape[0])
    n_bins = int(codec.grids.n_bins)
    c_idx = arr.astype(np.int64).reshape(n_frames, codec.n_tok, codec.n_val)
    valid = (c_idx >= 0) & (c_idx < n_bins)
    safe = np.where(valid, c_idx, 0)
    centres = codec.grids.centres()  # (n_tok, n_val, n_bins)
    tok_i = np.arange(codec.n_tok)[None, :, None]
    val_i = np.arange(codec.n_val)[None, None, :]
    dec_vals = np.where(valid, centres[tok_i, val_i, safe], np.nan)
    dec_vals = codec.read(dec_vals[None], "unquantized")
    if n_samples is None:
        n_samples = samples_for_frames(n_frames, codec.dt, dt_frame=dt_frame)
    return codec.decode(dec_vals, int(n_samples))[0]


def encode_shot(
    shot_path: Path,
    codecs: Mapping[str, tok_codecs.Codec],
    modality_order: Sequence[str],
    act_specs: Sequence[tasks.SignalSpec],
    act_stats: ActuatorStats,
    plasma_names: Sequence[str],
    *,
    stats_sha256: str | None = None,
    dt_frame: float = DT_FRAME,
    min_frames: int = MIN_CACHED_FRAMES,
) -> tuple[dict | None, str]:
    """Encode one MAST cache-v1 HDF5 shot into a Phase-B frame-code cache entry.

    Returns ``(entry, "ok")`` or ``(None, reason)`` where ``reason`` is
    ``"no_plasma_data"`` or ``"too_short"``.
    """
    needed = sorted(set(modality_order) | set(plasma_names) | {s.name for s in act_specs})
    shot_data = cache.read_shot(Path(shot_path), needed, stats_sha256=stats_sha256)
    span = plasma_frame_range(shot_data, plasma_names, dt_frame=dt_frame)
    if span is None:
        return None, "no_plasma_data"
    t0_ref, t0, f_start, _f_end, n_frames = span
    if n_frames < min_frames:
        return None, "too_short"

    codes: dict[str, torch.Tensor] = {}
    vocabs: dict[str, int] = {}
    present: dict[str, bool] = {}
    for name in modality_order:
        codec = codecs[name]
        _n_bins, _missing_code, vocab = codec_vocab(codec)
        sig_codes, is_present = encode_signal_codes(
            codec, shot_data.get(name), t0_ref, f_start, n_frames, dt_frame=dt_frame
        )
        codes[name] = sig_codes
        vocabs[name] = vocab
        present[name] = is_present

    raw_act, act_valid_np = raw_actuator_frames(
        shot_data, act_specs, t0=t0, n_frames=n_frames, dt_frame=dt_frame
    )
    actuators, act_valid = normalize_actuators(raw_act, act_valid_np, act_stats)
    entry = {
        "codes": codes,
        "actuators": actuators,
        "actuator_valid": act_valid,
        "n_frames": int(n_frames),
        "vocabs": vocabs,
        "t0": float(t0),
        "present": present,
    }
    return entry, "ok"


def read_shot_split(shot_path: Path) -> str:
    """Read the TokaMark split (``train``/``val``/``test``) from a cache-v1 ``.h5`` file."""
    with h5py.File(shot_path, "r") as f:
        return str(f.attrs["split"])


def build_manifest(
    signals: Sequence[tasks.SignalSpec],
    codecs: Mapping[str, tok_codecs.Codec],
    act_stats: ActuatorStats,
    *,
    assignment_path: Path,
    assignment_sha256: str,
    fit_dir: Path,
    fit_sha256: str,
    cache_root: Path,
    stats_sha256: str,
    dt_frame: float = DT_FRAME,
    t0_start: float = 0.0,
) -> dict:
    """Build the ``_codec_manifest.json`` payload for a MAST frame-code cache."""
    modalities: dict[str, dict] = {}
    bin_counts: set[int] = set()
    for spec in signals:
        if spec.name not in codecs:
            continue
        codec = codecs[spec.name]
        n_bins, missing_code, vocab = codec_vocab(codec)
        bin_counts.add(n_bins)
        modalities[spec.name] = {
            "family": map_family(spec),
            "n_tok": int(codec.n_tok * codec.n_val),
            "codebook_size": int(vocab),
            "n_bins": int(n_bins),
            "missing_code": int(missing_code),
            "codec_n_tok": int(codec.n_tok),
            "codec_n_val": int(codec.n_val),
        }
    return {
        "t0_start": float(t0_start),
        "dt_frame": float(dt_frame),
        "modalities": modalities,
        "actuators": act_stats.to_manifest_dict(),
        "assignment_name": Path(assignment_path).stem,
        "assignment_path": str(Path(assignment_path).resolve()),
        "assignment_sha256": str(assignment_sha256),
        "fit_dir": str(Path(fit_dir).resolve()),
        "fit_sha256": str(fit_sha256),
        "cache_v1_root": str(Path(cache_root).resolve()),
        "stats_sha256": str(stats_sha256),
        "notes": {
            "family_mapping": (
                "Mapped MAST signals to dev-peter's family vocabulary: Mirnov STFT "
                "spectrograms (BandDCT) -> 'spectro', 50 kHz fast time series (SXR, "
                "D-alpha, saddle voltage) -> 'fastts', and <=5 kHz time series / 1D "
                "profiles / 2D equilibrium flux maps -> 'slowts'. Verified via grep "
                "that DynamicsBackbone, FrameTokenizer, and MaskGITDynamics do not "
                "branch on ModalitySpec.family (it is metadata only)."
            ),
            "missing_data": (
                "Signals absent or all-invalid in a shot receive _PLACEHOLDER_CODE=0 "
                "across all frames and present[name]=False in both the entry and "
                "_presence.json so --mask_absent zeroes their loss weight in "
                "MaskGITDynamics._token_weights. Partially invalid tokens within a "
                "present signal (dead channels, pre/post-EFIT or TS frames, absent "
                "X-point) receive a dedicated missing_code = n_bins (64) with "
                "codebook_size = n_bins + 1 (65) so missing values are never collapsed "
                "to bin 0 and decode back to NaN. MaskGIT's [MASK] id is codebook_size "
                "(65)."
            ),
            "frame_range": (
                "Frames run on a 5 ms grid from the shot's first to last frame where "
                "any non-actuator plasma signal has a valid (finite) sample in cache "
                "v1. Per-shot frame-0 time is stored in each {shot}.pt as 't0'; "
                "t0_start=0.0 is recorded for dev-peter's manifest reader."
            ),
            "modalities_scope": (
                f"All {len(modalities)} signals with a codec in the assignment are "
                "included as modalities in sorted signal-name order. Bin counts across "
                f"modalities: {sorted(bin_counts)} (codebook_size = n_bins + 1)."
            ),
            "actuators": (
                "Union of task.actuators across all 14 TokaMark tasks in sorted order "
                "(5 signals, 8 channels). Each frame holds the per-frame mean in "
                "cache-v1 model units, z-scored with global mean/sd fitted over "
                "training shots only. Invalid actuator frames are set to 0 after "
                "z-scoring and recorded in 'actuator_valid'."
            ),
        },
    }


def save_pt_atomic(obj: dict, path: Path) -> None:
    """Atomically ``torch.save`` ``obj`` to ``path``."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_json_atomic(doc: dict, path: Path) -> None:
    """Atomically write JSON ``doc`` to ``path``."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(doc, indent=2) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
