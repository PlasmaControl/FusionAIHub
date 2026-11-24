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
from . import dalpha, peakspec

DT_FRAME: float = frames.FRAME_S
MIN_CACHED_FRAMES: int = 8      # matches train_dynamics._MIN_CACHED_FRAMES
PLACEHOLDER_CODE: int = 0       # matches train_dynamics._PLACEHOLDER_CODE

# Cuts the pre-breakdown vacuum and the post-termination tail, which are not
# plasma dynamics; DIII-D IGNITE likewise starts its frames at 1.0 s.
PLASMA_IP_THRESHOLD_A: float = 50_000.0
PLASMA_IP_SIGNAL: str = "summary-ip"


@dataclass(frozen=True)
class ActuatorStats:
    """Global training-set per-channel z-scoring statistics for MAST actuators."""

    names: tuple[str, ...]
    channels: tuple[int, ...]
    mu: np.ndarray
    sd: np.ndarray
    n_train_shots: int
    n_valid_frames: np.ndarray
    compress: tuple[str, ...] = ()
    mu_phys: np.ndarray | None = None
    sd_phys: np.ndarray | None = None

    @property
    def total_dim(self) -> int:
        return int(sum(self.channels))

    def to_manifest_dict(self) -> dict:
        doc = {
            "names": list(self.names),
            "channels": [int(c) for c in self.channels],
            "total_dim": self.total_dim,
            "mu": [float(x) for x in self.mu],
            "sd": [float(x) for x in self.sd],
            "n_train_shots": int(self.n_train_shots),
            "n_valid_frames": [int(x) for x in self.n_valid_frames],
        }
        if self.compress:
            doc["compress"] = list(self.compress)
        if self.mu_phys is not None:
            doc["mu_phys"] = [
                float(x) if math.isfinite(float(x)) else None for x in self.mu_phys
            ]
        if self.sd_phys is not None:
            doc["sd_phys"] = [
                float(x) if math.isfinite(float(x)) else None for x in self.sd_phys
            ]
        return doc

    @classmethod
    def from_manifest_dict(cls, doc: Mapping) -> ActuatorStats:
        compress = tuple(str(c) for c in doc.get("compress", ()))
        mu_phys_raw = doc.get("mu_phys")
        sd_phys_raw = doc.get("sd_phys")
        mu_phys = (
            np.asarray(
                [float("nan") if x is None else float(x) for x in mu_phys_raw],
                dtype=np.float64,
            )
            if mu_phys_raw is not None
            else None
        )
        sd_phys = (
            np.asarray(
                [float("nan") if x is None else float(x) for x in sd_phys_raw],
                dtype=np.float64,
            )
            if sd_phys_raw is not None
            else None
        )
        return cls(
            names=tuple(doc["names"]),
            channels=tuple(int(c) for c in doc["channels"]),
            mu=np.asarray(doc["mu"], dtype=np.float64),
            sd=np.asarray(doc["sd"], dtype=np.float64),
            n_train_shots=int(doc["n_train_shots"]),
            n_valid_frames=np.asarray(doc["n_valid_frames"], dtype=np.int64),
            compress=compress,
            mu_phys=mu_phys,
            sd_phys=sd_phys,
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
    logspec_fit: Path | None = None,
    dalpha_fit: Path | None = None,
) -> tuple[dict[str, tok_codecs.Codec], dict[str, transform.ChannelStats], dict[str, assign.Setting]]:
    """Build stage-1b/1c codecs exactly as ``scripts/mast/run_tokenizer.py::_build_predictor``.

    ``logspec_fit`` swaps the Mirnov spectrogram codecs for the log-power ones (`logspec`);
    ``dalpha_fit`` swaps the D-alpha codec for the level + spikes one (`dalpha.PeakLevel`).
    """
    fits, _ = fit.load_fit(Path(fit_dir))
    assignment = assign.load_assignment(Path(assignment_path))
    stats = transform.load_stats(Path(cache_root) / "stats.npz")
    covered = [
        s for s in signals
        if s.name in assignment and s.name in fits and fits[s.name].n_examples > 0
    ]
    rec_root = Path(recoders_dir) if recoders_dir is not None else recoders_root()
    codecs = learned_build.build_codecs(covered, stats, fits, assignment, rec_root)
    if logspec_fit is not None:
        from . import logspec

        fit_path = Path(logspec_fit)
        meta = json.loads((fit_path / "logspec_fit.json").read_text())
        fit_stats_sha = meta.get("stats_sha256")
        cache_stats_sha = transform.stats_sha256(Path(cache_root) / "stats.npz")
        if not fit_stats_sha or str(fit_stats_sha) != cache_stats_sha:
            raise ValueError(
                f"logspec_fit stats_sha256 ({fit_stats_sha!r}) disagrees with "
                f"{Path(cache_root) / 'stats.npz'} ({cache_stats_sha!r})"
            )
        log_codecs = logspec.build_logspec_codecs(fit_path, stats)
        missing_specs = [
            n for n in tasks.SPECTROGRAM_SIGNALS if n in codecs and n not in log_codecs
        ]
        if missing_specs:
            raise ValueError(
                f"logspec_fit at {fit_path} is missing spectrogram signals: {missing_specs}"
            )
        for name, codec in log_codecs.items():
            if name in codecs:
                codecs[name] = codec
    if dalpha_fit is not None:
        fit_path = Path(dalpha_fit)
        meta = json.loads((fit_path / "dalpha_fit.json").read_text())
        cache_stats_sha = transform.stats_sha256(Path(cache_root) / "stats.npz")
        if str(meta.get("stats_sha256")) != cache_stats_sha:
            raise ValueError(
                f"dalpha_fit stats_sha256 ({meta.get('stats_sha256')!r}) disagrees with "
                f"{Path(cache_root) / 'stats.npz'} ({cache_stats_sha!r})"
            )
        if dalpha.SIGNAL not in codecs:
            raise ValueError(f"dalpha_fit given but {dalpha.SIGNAL} has no codec to replace")
        codecs[dalpha.SIGNAL] = dalpha.build_dalpha_codec(fit_path, stats, codecs[dalpha.SIGNAL].dt)
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


def sample_times(t_len: int, t0: float, dt: float) -> np.ndarray:
    """Sample centre times ``t0 + (i + 0.5) * dt`` in seconds."""
    return float(t0) + (np.arange(int(t_len), dtype=np.float64) + 0.5) * float(dt)


def sample_frame_index(
    t_len: int,
    sig_t0: float,
    sig_dt: float,
    window_t0: float,
    dt_frame: float = DT_FRAME,
) -> np.ndarray:
    """Frame index of each sample relative to ``window_t0``, using sample centres."""
    ts = sample_times(t_len, sig_t0, sig_dt)
    return np.floor((ts - float(window_t0)) / float(dt_frame)).astype(np.int64)


def plasma_frame_range(
    shot_data: Mapping[str, cache.CachedSignal | None],
    ip_stats: transform.ChannelStats,
    *,
    ip_threshold_a: float = PLASMA_IP_THRESHOLD_A,
    dt_frame: float = DT_FRAME,
) -> tuple[tuple[float, float, int, int, int] | None, str]:
    """First to last 5 ms frame where ``|frame_mean(summary-ip)| >= ip_threshold_a``.

    Returns ``((t0_ref, t0, f_start, f_end, n_frames), "ok")`` where ``t0_ref`` is
    ``summary-ip.t0``, ``f_start`` and ``f_end`` are inclusive frame indices relative
    to ``t0_ref``, ``n_frames = f_end - f_start + 1``, and
    ``t0 = t0_ref + f_start * dt_frame``. Returns ``(None, reason)`` if ``summary-ip``
    is absent/all-NaN (``"no_summary_ip"``) or never reaches the plasma current
    threshold (``"no_plasma_above_ip_threshold"``).
    """
    cs_ip = shot_data.get(PLASMA_IP_SIGNAL)
    if cs_ip is None or cs_ip.values.shape[-1] == 0:
        return None, "no_summary_ip"
    t_len = int(cs_ip.values.shape[-1])
    ip_phys = transform.to_physical(cs_ip.values, ip_stats).reshape(-1, t_len)
    ip_1d = ip_phys[0]
    ok = np.isfinite(ip_1d)
    if not ok.any():
        return None, "no_summary_ip"

    t0_ref = float(cs_ip.t0)
    fi = sample_frame_index(t_len, cs_ip.t0, cs_ip.dt, t0_ref, dt_frame=dt_frame)
    n_total = int(fi[-1]) + 1
    sums = np.zeros(n_total, dtype=np.float64)
    counts = np.zeros(n_total, dtype=np.int64)
    np.add.at(sums, fi, np.where(ok, ip_1d, 0.0))
    np.add.at(counts, fi, ok.astype(np.int64))
    has = counts > 0
    means = np.divide(
        sums, counts, out=np.full(n_total, np.nan, dtype=np.float64), where=has
    )
    active = np.flatnonzero(has & (np.abs(means) >= float(ip_threshold_a)))
    if active.size == 0:
        return None, "no_plasma_above_ip_threshold"
    f_start = int(active[0])
    f_end = int(active[-1])
    n_frames = f_end - f_start + 1
    t0 = t0_ref + f_start * float(dt_frame)
    return (t0_ref, t0, f_start, f_end, n_frames), "ok"


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
        fi = sample_frame_index(t_len, cs.t0, cs.dt, t0, dt_frame=dt_frame)
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


def compose_physical_actuator_stats(
    act_stats: ActuatorStats,
    stats: Mapping[str, transform.ChannelStats],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Compose model-unit actuator z-score stats into physical units for linear actuators.

    For ``compress == "none"``: ``mu_phys = mu * st.std + st.mean``, ``sd_phys = sd * st.std``.
    For ``compress == "asinh"``: ``mu_phys`` and ``sd_phys`` are ``NaN`` (linear physical
    counterfactual edits are invalid in asinh space and rejected by ``layout``).
    """
    compress: list[str] = []
    mu_phys = np.full(act_stats.total_dim, np.nan, dtype=np.float64)
    sd_phys = np.full(act_stats.total_dim, np.nan, dtype=np.float64)
    offset = 0
    for name, n_ch in zip(act_stats.names, act_stats.channels):
        st = stats[name]
        compress.append(str(st.compress))
        if st.compress == "none":
            st_mean = np.asarray(st.mean, dtype=np.float64).reshape(n_ch)
            st_std = np.asarray(st.std, dtype=np.float64).reshape(n_ch)
            mu_phys[offset : offset + n_ch] = (
                act_stats.mu[offset : offset + n_ch] * st_std + st_mean
            )
            sd_phys[offset : offset + n_ch] = (
                act_stats.sd[offset : offset + n_ch] * st_std
            )
        offset += n_ch
    return tuple(compress), mu_phys, sd_phys


def shot_training_moments(
    shot_path: Path,
    act_specs: Sequence[tasks.SignalSpec],
    modality_order: Sequence[str],
    stats: Mapping[str, transform.ChannelStats],
    *,
    stats_sha256: str | None = None,
    ip_threshold_a: float = PLASMA_IP_THRESHOLD_A,
    dt_frame: float = DT_FRAME,
    min_frames: int = MIN_CACHED_FRAMES,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    """Compute actuator model-unit moments and per-signal physical moments on one training shot."""
    needed = sorted(
        {PLASMA_IP_SIGNAL} | set(modality_order) | {s.name for s in act_specs}
    )
    shot_data = cache.read_shot(Path(shot_path), needed, stats_sha256=stats_sha256)
    span, _reason = plasma_frame_range(
        shot_data,
        stats[PLASMA_IP_SIGNAL],
        ip_threshold_a=ip_threshold_a,
        dt_frame=dt_frame,
    )
    if span is None:
        return None
    _t0_ref, t0, _f_start, _f_end, n_frames = span
    if n_frames < min_frames:
        return None

    means, valid = raw_actuator_frames(
        shot_data, act_specs, t0=t0, n_frames=n_frames, dt_frame=dt_frame
    )
    n_ch = means.shape[1]
    act_count = valid.sum(axis=0).astype(np.int64)
    has = act_count > 0
    sums = np.where(valid, means, 0.0).sum(axis=0)
    act_mean = np.divide(sums, act_count, out=np.zeros(n_ch, dtype=np.float64), where=has)
    diff = np.where(valid, means - act_mean[None, :], 0.0)
    act_m2 = (diff ** 2).sum(axis=0)

    n_sig = len(modality_order)
    sig_count = np.zeros(n_sig, dtype=np.int64)
    sig_mean = np.zeros(n_sig, dtype=np.float64)
    sig_m2 = np.zeros(n_sig, dtype=np.float64)
    for i, name in enumerate(modality_order):
        cs = shot_data.get(name)
        if cs is None or cs.values.shape[-1] == 0:
            continue
        fi = sample_frame_index(cs.values.shape[-1], cs.t0, cs.dt, t0, dt_frame=dt_frame)
        keep = (fi >= 0) & (fi < n_frames)
        if not keep.any():
            continue
        idx = np.flatnonzero(keep)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        phys = transform.to_physical(cs.values[..., lo:hi], stats[name])
        ok = np.isfinite(phys)
        if not ok.any():
            continue
        vals = phys[ok].astype(np.float64)
        c = int(vals.size)
        m = float(vals.mean())
        sig_count[i] = c
        sig_mean[i] = m
        sig_m2[i] = float(np.sum((vals - m) ** 2))

    return (act_count, act_mean, act_m2), (sig_count, sig_mean, sig_m2)


def finalize_actuator_stats(
    act_specs: Sequence[tasks.SignalSpec],
    moments: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    stats: Mapping[str, transform.ChannelStats] | None = None,
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
    base = ActuatorStats(
        names=names,
        channels=channels,
        mu=acc_mean,
        sd=sd,
        n_train_shots=n_shots,
        n_valid_frames=acc_n,
    )
    if stats is None:
        return base
    compress, mu_phys, sd_phys = compose_physical_actuator_stats(base, stats)
    return ActuatorStats(
        names=base.names,
        channels=base.channels,
        mu=base.mu,
        sd=base.sd,
        n_train_shots=base.n_train_shots,
        n_valid_frames=base.n_valid_frames,
        compress=compress,
        mu_phys=mu_phys,
        sd_phys=sd_phys,
    )


def finalize_signal_physical_std(
    modality_order: Sequence[str],
    moments: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Combine per-shot physical-unit ``(count, mean, M2)`` tuples into global per-signal std."""
    n_sig = len(modality_order)
    acc_n = np.zeros(n_sig, dtype=np.int64)
    acc_mean = np.zeros(n_sig, dtype=np.float64)
    acc_m2 = np.zeros(n_sig, dtype=np.float64)
    for count, mean, m2 in moments:
        acc_n, acc_mean, acc_m2 = _chan_merge(acc_n, acc_mean, acc_m2, count, mean, m2)
    var = np.divide(
        acc_m2, acc_n, out=np.full(n_sig, np.nan, dtype=np.float64), where=acc_n > 0
    )
    sd = np.sqrt(np.maximum(var, 0.0))
    return {
        str(name): (float(sd[i]) if acc_n[i] > 0 and math.isfinite(float(sd[i])) else float("nan"))
        for i, name in enumerate(modality_order)
    }


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


def _groups_from_frame_index(
    fi: np.ndarray, n_frames: int
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Group monotonically non-decreasing frame indices ``fi`` in ``[0, n_frames)`` by frame length.

    Returns ``[(length, frame_ids, first_sample_offsets)]``.
    """
    lens = np.bincount(fi, minlength=n_frames)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for length in np.unique(lens):
        if length == 0:
            continue
        ids = np.flatnonzero(lens == length)
        out.append((int(length), ids, starts[ids]))
    return out


def _encode_domain_by_frame_index(
    codec: tok_codecs.Codec,
    u_w: np.ndarray,
    fi_w: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    """Encode domain samples ``u_w`` ``(1, *values_shape, M)`` onto frames ``fi_w`` in ``[0, n_frames)``."""
    if isinstance(codec, (dalpha.PeakLevel, peakspec.PeakSpec)):
        return codec.encode_aligned(np.asarray(u_w, dtype=np.float64)[0], fi_w, n_frames)

    if isinstance(codec, tok_codecs.FrameDCT):
        flat = np.asarray(u_w, dtype=np.float64).reshape(1, codec.n_tok, u_w.shape[-1])
        aligned = np.full((n_frames, codec.n_tok, codec.k), np.nan, dtype=np.float64)
        for length, ids, starts in _groups_from_frame_index(fi_w, n_frames):
            cols = starts[:, None] + np.arange(length)
            v = tok_codecs.dct_values(flat[:, :, cols], codec.k)  # (1, n_tok, m, k)
            aligned[ids] = np.moveaxis(v[0], 1, 0)
        return aligned

    if isinstance(codec, tok_codecs.BandDCT):
        u64 = np.asarray(u_w, dtype=np.float64)  # (1, P, n_bins, M)
        p, _n_bins = codec.values_shape
        aligned = np.full((n_frames, codec.n_tok, codec.n_val), np.nan, dtype=np.float64)
        for length, ids, starts in _groups_from_frame_index(fi_w, n_frames):
            cols = starts[:, None] + np.arange(length)
            seg = u64[..., cols]  # (1, P, n_bins, m, length)
            ok = ~np.isnan(seg)
            count = ok.sum(-1)
            mean = np.divide(
                np.where(ok, seg, 0.0).sum(-1),
                count,
                out=np.full(count.shape, np.nan),
                where=count > 0,
            )  # (1, P, n_bins, m)
            spec = np.moveaxis(
                mean.reshape(1, p, codec.n_bands, tok_codecs.BAND_BINS, -1), -1, 2
            )  # (1, P, m, n_bands, 32)
            v = tok_codecs.dct_values(spec, codec.n_val)  # (1, P, m, n_bands, n_val)
            aligned[ids] = np.moveaxis(v[0], 1, 0).reshape(
                len(ids), codec.n_tok, codec.n_val
            )
        return codec._mask(aligned)

    if isinstance(codec, tok_codecs.LinearCodec):
        if len(fi_w) > 1 and (np.diff(fi_w) < 1).any():
            raise ValueError(
                f"{codec.name}: multiple samples fell in one frame for LinearCodec"
            )
        vals_local = codec.encode_domain(u_w)[0]
        f_local = codec._frame_of_sample(u_w.shape[-1])
        aligned = np.full((n_frames, codec.n_tok, codec.n_val), np.nan, dtype=np.float64)
        aligned[fi_w] = vals_local[f_local]
        return aligned

    raise TypeError(f"unsupported codec type for time-aligned encoding: {type(codec).__name__}")


def live_mask(codec: tok_codecs.Codec) -> np.ndarray:
    """Which of the codec's ``n_tok * n_val`` flat positions carry information, ``bool[n_tok * n_val]``.

    A band-DCT codec pads every token to ``n_val = max(k)`` values, so the values from
    ``k[band]`` on are dropped by the codec and always encode the same code. Measured
    2026-11-23: the two Mirnov arrays hold 672 such positions of their 1,536 (v1b ``k``).
    A linear codec keeps ``n_values`` of its positions (eigenvalue > 0). Anything else is live.
    """
    n_flat = int(codec.n_tok * codec.n_val)
    if hasattr(codec, "_keep"):
        return np.asarray(codec._keep, dtype=bool).reshape(n_flat)
    if isinstance(codec, tok_codecs.LinearCodec) and codec.n_values > 0:
        return (codec.eigenvalues[: codec.n_values] > 0).reshape(n_flat)
    return np.ones(n_flat, dtype=bool)


def encode_signal_codes(
    codec: tok_codecs.Codec,
    cs: cache.CachedSignal | None,
    t0: float,
    n_frames: int,
    dt_frame: float = DT_FRAME,
    live_only: bool = False,
) -> tuple[torch.Tensor, bool]:
    """`_encode_signal_codes_full`, keeping only the live positions when ``live_only``.

    ``live_only`` drops the positions the codec never uses (`live_mask`), so the frame is
    shorter and nothing is lost: `decode_signal_codes` puts them back.
    """
    codes, present = _encode_signal_codes_full(codec, cs, t0, n_frames, dt_frame=dt_frame)
    if live_only:
        codes = codes[:, torch.from_numpy(live_mask(codec))]
    return codes, present


def _encode_signal_codes_full(
    codec: tok_codecs.Codec,
    cs: cache.CachedSignal | None,
    t0: float,
    n_frames: int,
    dt_frame: float = DT_FRAME,
) -> tuple[torch.Tensor, bool]:
    """Encode one cached signal onto the ``n_frames`` window starting at ``t0``.

    Samples are assigned to frames by sample centre time ``cs.t0 + (i + 0.5) * cs.dt``
    relative to ``t0`` (no rounded start offset), eliminating sub-frame drift such as
    Mirnov's +1.022 ms start offset.

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

    t_len = int(cs.values.shape[-1])
    fi = sample_frame_index(t_len, cs.t0, cs.dt, t0, dt_frame=dt_frame)
    keep = (fi >= 0) & (fi < n_frames)
    if not keep.any():
        return (
            torch.full((n_frames, n_flat), PLACEHOLDER_CODE, dtype=torch.int32),
            False,
        )
    idx = np.flatnonzero(keep)
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    u_w = codec.domain_from_model(cs.values[None, ..., lo:hi])
    aligned = _encode_domain_by_frame_index(codec, u_w, fi[lo:hi], n_frames)

    if not np.isfinite(aligned).any():
        return (
            torch.full((n_frames, n_flat), PLACEHOLDER_CODE, dtype=torch.int32),
            False,
        )
    _n_bins, missing_code, _vocab = codec_vocab(codec)
    idx_q = codec.grids.index(aligned)  # (n_frames, n_tok, n_val), -1 where NaN
    code_arr = np.where(idx_q < 0, missing_code, idx_q).reshape(n_frames, n_flat)
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


def _neutralize_dropped(codec: tok_codecs.Codec, dec_vals: np.ndarray) -> np.ndarray:
    """Set positions a codec drops to 0.0 before ``codec.read`` so stray missing codes cannot poison kept values."""
    if isinstance(codec, tok_codecs.BandDCT):
        return np.where(codec._keep[None, :, :], dec_vals, 0.0)
    if isinstance(codec, tok_codecs.LinearCodec) and codec.n_values > 0:
        keep = (codec.eigenvalues[: codec.n_values] > 0).reshape(codec.n_tok, codec.n_val)
        if not keep.all():
            return np.where(keep[None, :, :], dec_vals, 0.0)
    return dec_vals


def _decode_domain_by_frame_index(
    codec: tok_codecs.Codec,
    dec_vals: np.ndarray,
    fi: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    """Decode ``dec_vals`` ``(1, n_frames, n_tok, n_val)`` onto samples with frame indices ``fi``."""
    n_samp = len(fi)
    if isinstance(codec, (dalpha.PeakLevel, peakspec.PeakSpec)):
        return codec.render(dec_vals[0], fi, n_frames)

    if isinstance(codec, tok_codecs.FrameDCT):
        out = np.full((1, codec.n_tok, n_samp), np.nan, dtype=np.float64)
        for length, ids, starts in _groups_from_frame_index(fi, n_frames):
            cols = starts[:, None] + np.arange(length)
            v_grp = np.moveaxis(dec_vals[:, ids], 1, 2)  # (1, n_tok, m, k)
            if length < v_grp.shape[-1]:
                v_grp = v_grp.copy()
                v_grp[..., length:] = 0.0
            out[:, :, cols] = tok_codecs.dct_decode(v_grp, length)
        return codec.from_domain(out.reshape((1,) + codec.values_shape + (n_samp,)))[0]

    if isinstance(codec, tok_codecs.BandDCT):
        p, n_bins = codec.values_shape
        out = np.full((1, p, n_bins, n_samp), np.nan, dtype=np.float64)
        for length, ids, starts in _groups_from_frame_index(fi, n_frames):
            cols = starts[:, None] + np.arange(length)
            d = tok_codecs.dct_decode(
                dec_vals[:, ids].reshape(1, len(ids), p, codec.n_bands, dec_vals.shape[3]),
                tok_codecs.BAND_BINS,
            )
            spec = np.moveaxis(d.reshape(1, len(ids), p, n_bins), 1, -1)
            out[..., cols] = spec[..., None]
        return codec.from_domain(out)[0]

    if isinstance(codec, tok_codecs.LinearCodec):
        if codec.n_values == 0:
            dom = np.broadcast_to(
                codec.profile_mean.reshape(codec.values_shape + (1,)),
                codec.values_shape + (n_samp,),
            ).copy()
        else:
            a = dec_vals[0, fi].reshape(n_samp, codec.n_values)
            dom = codec.profile_mean + a @ codec.basis[: codec.n_values]
            dom = np.moveaxis(dom, 0, 1).reshape(codec.values_shape + (n_samp,))
        return codec.from_domain(dom[None])[0]

    raise TypeError(f"unsupported codec type for time-aligned decoding: {type(codec).__name__}")


def decode_signal_codes(
    codec: tok_codecs.Codec,
    codes: torch.Tensor | np.ndarray,
    n_samples: int | None = None,
    dt_frame: float = DT_FRAME,
    sample_frame_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Decode ``(F, n_tok * n_val)`` bin codes back to physical units ``(*values_shape, n_samples)``.

    Codes ``0 .. n_bins - 1`` map to the codec's grid centres; ``missing_code``
    (``>= n_bins`` or ``< 0``) maps to ``NaN`` at kept positions. Positions dropped
    by the codec (e.g. ``BandDCT`` coefficients beyond ``k[band]``) are neutralized
    to ``0.0`` before ``codec.read`` so a stray missing code there cannot wipe a band.

    When ``sample_frame_idx`` is provided (monotonically non-decreasing frame indices
    in ``[0, F)`` for each output sample), samples are decoded onto that exact frame
    mapping instead of ``frames.frame_index(n_samples, codec.dt)``.
    """
    arr = codes.detach().cpu().numpy() if isinstance(codes, torch.Tensor) else np.asarray(codes)
    n_flat = int(codec.n_tok * codec.n_val)
    if arr.ndim == 2 and arr.shape[1] != n_flat:
        live = live_mask(codec)
        if arr.shape[1] == int(live.sum()):  # live-only frame: put the dropped positions back (code 0)
            full = np.zeros((arr.shape[0], n_flat), dtype=arr.dtype)
            full[:, live] = arr
            arr = full
    if arr.ndim != 2 or arr.shape[1] != n_flat:
        raise ValueError(
            f"{codec.name}: expected (F, {n_flat}), got {arr.shape}"
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
    dec_vals = _neutralize_dropped(codec, dec_vals)
    dec_vals = codec.read(dec_vals[None], "unquantized")
    if sample_frame_idx is not None:
        fi = np.asarray(sample_frame_idx, dtype=np.int64)
        return _decode_domain_by_frame_index(codec, dec_vals, fi, n_frames)
    if n_samples is None:
        n_samples = samples_for_frames(n_frames, codec.dt, dt_frame=dt_frame)
    return codec.decode(dec_vals, int(n_samples))[0]


def encode_shot(
    shot_path: Path,
    codecs: Mapping[str, tok_codecs.Codec],
    modality_order: Sequence[str],
    act_specs: Sequence[tasks.SignalSpec],
    act_stats: ActuatorStats,
    stats: Mapping[str, transform.ChannelStats],
    *,
    stats_sha256: str | None = None,
    ip_threshold_a: float = PLASMA_IP_THRESHOLD_A,
    dt_frame: float = DT_FRAME,
    min_frames: int = MIN_CACHED_FRAMES,
    live_only: bool = False,
) -> tuple[dict | None, str]:
    """Encode one MAST cache-v1 HDF5 shot into a Phase-B frame-code cache entry.

    Returns ``(entry, "ok")`` or ``(None, reason)`` where ``reason`` is
    ``"no_summary_ip"``, ``"no_plasma_above_ip_threshold"``, or ``"too_short"``.
    """
    needed = sorted(
        {PLASMA_IP_SIGNAL} | set(modality_order) | {s.name for s in act_specs}
    )
    shot_data = cache.read_shot(Path(shot_path), needed, stats_sha256=stats_sha256)
    span, reason = plasma_frame_range(
        shot_data,
        stats[PLASMA_IP_SIGNAL],
        ip_threshold_a=ip_threshold_a,
        dt_frame=dt_frame,
    )
    if span is None:
        return None, reason
    _t0_ref, t0, _f_start, _f_end, n_frames = span
    if n_frames < min_frames:
        return None, "too_short"

    codes: dict[str, torch.Tensor] = {}
    vocabs: dict[str, int] = {}
    present: dict[str, bool] = {}
    for name in modality_order:
        codec = codecs[name]
        _n_bins, _missing_code, vocab = codec_vocab(codec)
        sig_codes, is_present = encode_signal_codes(
            codec, shot_data.get(name), t0, n_frames, dt_frame=dt_frame, live_only=live_only
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
    stats: Mapping[str, transform.ChannelStats] | None = None,
    signal_physical_std: Mapping[str, float] | None = None,
    logspec_fit_path: Path | None = None,
    logspec_fit_sha256: str | None = None,
    camera_codec_path: Path | None = None,
    camera_codec_sha256: str | None = None,
    camera_fit_path: Path | None = None,
    camera_fit_sha256: str | None = None,
    camera_frames_dir: Path | None = None,
    camera_frames_index_sha256: str | None = None,
    camera_n_tok: int | None = None,
    camera_codebook_size: int | None = None,
    dalpha_fit_path: Path | None = None,
    dalpha_fit_sha256: str | None = None,
    plasma_ip_threshold_a: float = PLASMA_IP_THRESHOLD_A,
    dt_frame: float = DT_FRAME,
    t0_start: float = 0.0,
    live_only: bool = False,
) -> dict:
    """Build the ``_codec_manifest.json`` payload for a MAST frame-code cache.

    ``live_only``: each modality's ``n_tok`` counts only its live positions (`live_mask`).
    """
    if stats is not None and (act_stats.mu_phys is None or act_stats.sd_phys is None):
        compress, mu_phys, sd_phys = compose_physical_actuator_stats(act_stats, stats)
        act_stats = ActuatorStats(
            names=act_stats.names,
            channels=act_stats.channels,
            mu=act_stats.mu,
            sd=act_stats.sd,
            n_train_shots=act_stats.n_train_shots,
            n_valid_frames=act_stats.n_valid_frames,
            compress=compress,
            mu_phys=mu_phys,
            sd_phys=sd_phys,
        )
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
            "n_tok": int(live_mask(codec).sum()) if live_only else int(codec.n_tok * codec.n_val),
            "codebook_size": int(vocab),
            "n_bins": int(n_bins),
            "missing_code": int(missing_code),
            "codec_n_tok": int(codec.n_tok),
            "codec_n_val": int(codec.n_val),
        }
    cam_paths = (camera_codec_path, camera_fit_path, camera_frames_dir)
    cam_shas = (camera_codec_sha256, camera_fit_sha256, camera_frames_index_sha256)
    has_camera = any(p is not None for p in cam_paths) or any(s is not None for s in cam_shas)
    if has_camera:
        if any(p is None for p in cam_paths):
            raise ValueError(
                "build_manifest: camera_codec_path, camera_fit_path, and camera_frames_dir "
                "must all be provided when camera is enabled"
            )
        missing_shas = [
            name
            for name, val in (
                ("camera_codec_sha256", camera_codec_sha256),
                ("camera_fit_sha256", camera_fit_sha256),
                ("camera_frames_index_sha256", camera_frames_index_sha256),
            )
            if not val or not str(val).strip()
        ]
        if missing_shas:
            raise ValueError(
                "build_manifest: camera enabled without non-empty sha256(s): "
                + ", ".join(missing_shas)
            )
        cam_n_tok = 48 if camera_n_tok is None else int(camera_n_tok)
        cam_n_codes = 1000 if camera_codebook_size is None else int(camera_codebook_size)
        modalities["camera-rgb"] = {
            "family": "video",
            "n_tok": cam_n_tok,
            "codebook_size": cam_n_codes + 1,
            "n_codes": cam_n_codes,
            "missing_code": cam_n_codes,
        }
        modalities["camera-rgb-stats"] = {
            "family": "slowts",
            "n_tok": 2,
            "codebook_size": 65,
            "n_bins": 64,
            "missing_code": 64,
        }
    phys_std_doc: dict[str, float | None] = {}
    if signal_physical_std is not None:
        for name in modalities:
            if name in ("camera-rgb", "camera-rgb-stats"):
                continue
            val = signal_physical_std.get(name, float("nan"))
            phys_std_doc[name] = float(val) if math.isfinite(float(val)) else None
    manifest = {
        "t0_start": float(t0_start),
        "dt_frame": float(dt_frame),
        "plasma_ip_threshold_A": float(plasma_ip_threshold_a),
        "modalities": modalities,
        "actuators": act_stats.to_manifest_dict(),
        "signal_physical_std": phys_std_doc,
        "assignment_name": Path(assignment_path).stem,
        "assignment_path": str(Path(assignment_path).resolve()),
        "assignment_sha256": str(assignment_sha256),
        "fit_dir": str(Path(fit_dir).resolve()),
        "fit_sha256": str(fit_sha256),
        "cache_v1_root": str(Path(cache_root).resolve()),
        "stats_sha256": str(stats_sha256),
        "logspec_fit_path": (
            str(Path(logspec_fit_path).resolve()) if logspec_fit_path is not None else None
        ),
        "logspec_fit_sha256": (
            str(logspec_fit_sha256) if logspec_fit_sha256 is not None else None
        ),
        # written only when the D-alpha level + spikes codec is used, so older manifests are unchanged
        **(
            {
                "dalpha_fit_path": str(Path(dalpha_fit_path).resolve()),
                "dalpha_fit_sha256": str(dalpha_fit_sha256),
            }
            if dalpha_fit_path is not None
            else {}
        ),
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
                "to bin 0 and decode back to NaN at kept positions. MaskGIT's [MASK] "
                "id is codebook_size (65)."
            ),
            "frame_range": (
                f"Frames run on a 5 ms grid from the shot's first to last frame "
                f"(inclusive) where |frame_mean(summary-ip)| >= {plasma_ip_threshold_a:g} A "
                "in physical units, cutting the pre-breakdown vacuum and post-termination "
                "tail (analogous to DIII-D IGNITE starting at 1.0 s). Samples are assigned "
                "to frames by sample centre time t0 + (i + 0.5) * dt. Per-shot frame-0 "
                "time is stored in each {shot}.pt as 't0'; t0_start=0.0 is recorded for "
                "dev-peter's manifest reader."
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
                "training shots' plasma frames only. Invalid actuator frames are set "
                "to 0 after z-scoring and recorded in 'actuator_valid'. Composed "
                "physical-unit stats (mu_phys, sd_phys) are stored for linear actuators."
            ),
        },
    }
    if has_camera:
        manifest["camera_codec_path"] = str(Path(camera_codec_path).resolve())
        manifest["camera_codec_sha256"] = str(camera_codec_sha256)
        manifest["camera_fit_path"] = str(Path(camera_fit_path).resolve())
        manifest["camera_fit_sha256"] = str(camera_fit_sha256)
        manifest["camera_frames_dir"] = str(Path(camera_frames_dir).resolve())
        manifest["camera_frames_index_sha256"] = str(camera_frames_index_sha256)
    return manifest


def save_pt_atomic(obj: dict, path: Path) -> None:
    """Atomically ``torch.save`` ``obj`` to ``path``."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_json_atomic(doc: dict, path: Path, *, indent: int | None = 2) -> None:
    """Atomically write JSON ``doc`` to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(doc, indent=indent) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
