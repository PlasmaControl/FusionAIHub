"""Whole-shot autoregressive rollouts, decoding to physical units, and physics plots."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .. import cache, transform
from ..tokenizers import codecs as tok_codecs, frames
from . import framecache, layout
from .layout import ev


@dataclass(frozen=True)
class PlotSignalSpec:
    """Display settings for one physicist-facing panel in a rollout plot."""

    name: str
    label: str
    unit: str
    scale: float = 1.0


# ~10 signals a physicist reads first on MAST (plus NBI power as actuator context).
KEY_PLOT_SIGNALS: tuple[PlotSignalSpec, ...] = (
    PlotSignalSpec("summary-ip", r"$I_p$", "kA", 1e-3),
    PlotSignalSpec("interferometer-n_e_line", r"$\bar{n}_e L$", r"$10^{19}\,\mathrm{m}^{-2}$", 1e-19),
    PlotSignalSpec("spectrometer_visible-filter_spectrometer_dalpha_voltage", r"$D_\alpha$", "V", 1.0),
    PlotSignalSpec("equilibrium-beta_pol", r"$\beta_p$", "", 1.0),
    PlotSignalSpec("equilibrium-q95", r"$q_{95}$", "", 1.0),
    PlotSignalSpec("equilibrium-elongation", r"$\kappa$", "", 1.0),
    PlotSignalSpec("thomson_scattering-t_e", r"$T_e$", "keV", 1e-3),
    PlotSignalSpec("soft_x_rays-horizontal_cam_upper", "SXR", "V", 1.0),
    PlotSignalSpec("magnetics-flux_loop_flux", r"$\Psi_{\mathrm{loop}}$", "Wb", 1.0),
    PlotSignalSpec("magnetics-b_field_pol_probe_omv_voltage", "Mirnov OMV", "a.u.", 1.0),
    PlotSignalSpec("summary-power_nbi", r"$P_{\mathrm{NBI}}$ (ctx)", "MW", 1e-6),
)

_PLOT_SPECS_BY_NAME: dict[str, PlotSignalSpec] = {s.name: s for s in KEY_PLOT_SIGNALS}


def load_mast_model(ckpt_path: Path | str, cache_dir: Path | str, device: torch.device):
    """Patch layout/actuators from ``cache_dir`` and load checkpoint via ``eval_dynamics.load_model``."""
    layout.patch_ignite(cache_dir)
    return ev.load_model(Path(ckpt_path), device)


def rollout_mast_shot(
    model,
    cfg,
    cache_dir: Path | str,
    shot: str,
    *,
    k0: int = 0,
    horizon: int = 0,
    temperature: float = 1.0,
    seed: int = 0,
    device: torch.device,
    actuator_mode: str = "real",
) -> tuple[ev.Rollout, dict, torch.Tensor]:
    """Run dev-peter's ``eval_dynamics.rollout_shot`` on one cached MAST shot.

    Returns ``(rollout, shot_cache_entry, edited_actuators)``.
    """
    entry = ev.load_shot_cache(Path(cache_dir), str(shot))
    n_seed = int(k0) if k0 else int(cfg.k0_seed)
    max_n = int(entry["n_frames"])
    if horizon > 0:
        max_n = min(max_n, n_seed + int(horizon))
    capped = min(int(cfg.max_frames), max_n)
    if capped <= n_seed:
        raise ValueError(
            f"shot {shot} has {capped} usable frames <= k0={n_seed}; cannot roll out"
        )
    view = {**entry, "n_frames": capped}
    generator = torch.Generator(device=device).manual_seed(int(seed))
    rollout = ev.rollout_shot(
        model,
        cfg,
        view,
        n_seed,
        float(temperature),
        generator,
        device,
        actuator_mode=actuator_mode,
        cache_dir=Path(cache_dir),
        shot=str(shot),
    )
    edited_act = ev.apply_actuator_mode(
        entry["actuators"][: rollout.n_frames].float(),
        actuator_mode,
        n_seed,
        cache_dir=Path(cache_dir),
        F=rollout.n_frames,
    )
    return rollout, entry, edited_act


def _nrmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """``rmse(pred - gt) / (std(gt) + 1e-12)`` over mutually finite entries in float64."""
    finite = np.isfinite(pred) & np.isfinite(gt)
    if not finite.any():
        return float("nan")
    p = pred[finite].astype(np.float64)
    g = gt[finite].astype(np.float64)
    std = float(np.std(g))
    if std <= 0.0:
        return float("nan")
    err = p - g
    return float(np.sqrt(np.mean(err * err))) / (std + 1e-12)


def _align_truth_to_window(
    cs: cache.CachedSignal | None,
    st: transform.ChannelStats,
    t0: float,
    n_frames: int,
    dt_frame: float,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Slice cache-v1 signal to ``[t0, t0 + n_frames * dt_frame)`` in physical units.

    Returns ``(truth_phys, sample_times_s, n_samples)`` or ``(None, None, 0)`` if absent.
    """
    if cs is None or cs.values.shape[-1] == 0:
        return None, None, 0
    t_len = cs.values.shape[-1]
    dt = float(cs.dt)
    ts = float(cs.t0) + np.arange(t_len, dtype=np.float64) * dt
    # Align frame 0 of the signal to frame 0 of the shot window.
    sig_offset = int(round((float(cs.t0) - float(t0)) / dt_frame))
    f_all = frames.frame_index(t_len, dt) + sig_offset
    keep = (f_all >= 0) & (f_all < n_frames)
    if not keep.any():
        return None, None, 0
    idx = np.flatnonzero(keep)
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    phys = transform.to_physical(cs.values[..., lo:hi], st)
    return phys, ts[lo:hi], hi - lo


def _decode_aligned(
    codec: tok_codecs.Codec,
    codes: torch.Tensor,
    cs: cache.CachedSignal | None,
    t0: float,
    n_frames: int,
    dt_frame: float,
    truth_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``codes`` to physical units on the exact time grid of ``truth`` (or uniform grid)."""
    if cs is not None and truth_n > 0:
        sig_offset = int(round((float(cs.t0) - float(t0)) / dt_frame))
        # Decode starting from the signal's aligned frame index.
        if sig_offset <= 0:
            decoded = framecache.decode_signal_codes(
                codec, codes[:n_frames], n_samples=truth_n, dt_frame=dt_frame
            )
            sample_f = frames.frame_index(truth_n, float(cs.dt))
            return decoded, sample_f
        # Signal starts after t0 (rare): slice codes from sig_offset onward.
        sub_codes = codes[sig_offset:n_frames]
        decoded = framecache.decode_signal_codes(
            codec, sub_codes, n_samples=truth_n, dt_frame=dt_frame
        )
        sample_f = frames.frame_index(truth_n, float(cs.dt)) + sig_offset
        return decoded, sample_f
    n_samp = framecache.samples_for_frames(n_frames, codec.dt, dt_frame=dt_frame)
    decoded = framecache.decode_signal_codes(
        codec, codes[:n_frames], n_samples=n_samp, dt_frame=dt_frame
    )
    sample_f = frames.frame_index(n_samp, codec.dt)
    return decoded, sample_f


def decode_and_score_rollout(
    rollout: ev.Rollout,
    shot_h5_path: Path,
    codecs: Mapping[str, tok_codecs.Codec],
    stats: Mapping[str, transform.ChannelStats],
    t0: float,
    dt_frame: float = framecache.DT_FRAME,
    *,
    present: Mapping[str, bool] | None = None,
    stats_sha256: str | None = None,
) -> tuple[dict[str, dict], dict[str, dict[str, float]]]:
    """Decode ground-truth codes (ceiling) and predicted codes (rollout) to physical units.

    Computes per-signal NRMSE over the predicted frames ``[k0, F)`` of:
      - ``rollout_nrmse``: rollout vs cache-v1 ground truth (physical units);
      - ``ceiling_nrmse``: tokenizer ceiling (true codes decoded) vs ground truth.
    """
    mod_names = list(rollout.gt_codes)
    shot_data = cache.read_shot(Path(shot_h5_path), mod_names, stats_sha256=stats_sha256)
    k0, n_frames = int(rollout.k0), int(rollout.n_frames)
    decoded: dict[str, dict] = {}
    metrics: dict[str, dict[str, float]] = {}

    for name in mod_names:
        if name not in codecs:
            continue
        codec = codecs[name]
        is_present = True if present is None else bool(present.get(name, True))
        cs = shot_data.get(name) if is_present else None
        truth_phys, times_s, n_samp = _align_truth_to_window(
            cs, stats[name], t0=t0, n_frames=n_frames, dt_frame=dt_frame
        )
        if truth_phys is None or not is_present:
            decoded[name] = {
                "present": False,
                "times_s": t0 + np.arange(n_frames, dtype=np.float64) * dt_frame,
                "truth": None,
                "ceiling": None,
                "rollout": None,
            }
            metrics[name] = {"rollout_nrmse": float("nan"), "ceiling_nrmse": float("nan")}
            continue

        ceil_phys, sample_f = _decode_aligned(
            codec, rollout.gt_codes[name], cs, t0, n_frames, dt_frame, n_samp
        )
        pred_phys, _ = _decode_aligned(
            codec, rollout.pred_codes[name], cs, t0, n_frames, dt_frame, n_samp
        )
        pred_sel = sample_f >= k0
        if pred_sel.any():
            r_nrmse = _nrmse(pred_phys[..., pred_sel], truth_phys[..., pred_sel])
            c_nrmse = _nrmse(ceil_phys[..., pred_sel], truth_phys[..., pred_sel])
        else:
            r_nrmse = c_nrmse = float("nan")
        decoded[name] = {
            "present": True,
            "times_s": times_s,
            "truth": truth_phys,
            "ceiling": ceil_phys,
            "rollout": pred_phys,
        }
        metrics[name] = {"rollout_nrmse": r_nrmse, "ceiling_nrmse": c_nrmse}
    return decoded, metrics


def _channel_variance(arr: np.ndarray) -> np.ndarray:
    """Temporal variance per channel (axis -1) without warnings on all-NaN channels."""
    x = np.asarray(arr, dtype=np.float64)
    ok = np.isfinite(x)
    count = ok.sum(axis=-1)
    has = count > 0
    sums = np.where(ok, x, 0.0).sum(axis=-1)
    mean = np.divide(sums, count, out=np.zeros_like(sums), where=has)
    sq = np.where(ok, (x - mean[..., None]) ** 2, 0.0).sum(axis=-1)
    return np.divide(sq, count, out=np.full_like(sums, -np.inf), where=has)


def _reduce_for_panel(
    truth: np.ndarray, ceiling: np.ndarray, rollout: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Reduce a signal ``(*values_shape, T)`` to 1D traces ``(T,)`` + channel tag."""
    if truth.ndim == 2 and truth.shape[0] == 1:
        return truth[0], ceiling[0], rollout[0], ""
    if truth.ndim == 2:
        # 1D multi-channel (profiles, SXR, flux loops, D-alpha): most variable channel.
        var = _channel_variance(truth)
        ch = int(np.argmax(var)) if np.isfinite(var).any() else 0
        return truth[ch], ceiling[ch], rollout[ch], f"ch {ch}"
    # 2D spatial/spectral (e.g. Mirnov (P, F_bins, T) or psi (65, 65, T)):
    # mean over inner axis, then pick most variable outer channel.
    inner = tuple(range(1, truth.ndim - 1))

    def _mean_inner(a: np.ndarray) -> np.ndarray:
        x = np.asarray(a, dtype=np.float64)
        ok = np.isfinite(x)
        cnt = ok.sum(axis=inner)
        return np.divide(
            np.where(ok, x, 0.0).sum(axis=inner),
            cnt,
            out=np.full(cnt.shape, np.nan, dtype=np.float64),
            where=cnt > 0,
        )

    t_ch, c_ch, r_ch = _mean_inner(truth), _mean_inner(ceiling), _mean_inner(rollout)
    var = _channel_variance(t_ch)
    ch = int(np.argmax(var)) if np.isfinite(var).any() else 0
    return t_ch[ch], c_ch[ch], r_ch[ch], f"probe {ch} mean"


def select_panel_specs(decoded: Mapping[str, dict]) -> list[PlotSignalSpec]:
    """Pick the physicist-first panels present in ``decoded`` (fallback to available signals)."""
    chosen = [spec for spec in KEY_PLOT_SIGNALS if spec.name in decoded]
    if chosen:
        return chosen
    return [
        PlotSignalSpec(name, name, "", 1.0)
        for name in list(decoded)[: len(KEY_PLOT_SIGNALS)]
    ]


def counterfactual_nbi_physical(
    edited_actuators: torch.Tensor,
    act_stats: framecache.ActuatorStats,
    stats: Mapping[str, transform.ChannelStats],
) -> Optional[np.ndarray]:
    """Convert the counterfactual ``summary-power_nbi`` actuator column back to physical units (W)."""
    if "summary-power_nbi" not in act_stats.names or "summary-power_nbi" not in stats:
        return None
    offset = 0
    for name, n_ch in zip(act_stats.names, act_stats.channels):
        if name == "summary-power_nbi":
            z = edited_actuators[:, offset].detach().cpu().numpy().astype(np.float64)
            model_u = z * act_stats.sd[offset] + act_stats.mu[offset]
            return transform.to_physical(model_u[None, :], stats["summary-power_nbi"])[0]
        offset += n_ch
    return None


def plot_shot_rollout(
    shot: str,
    decoded: Mapping[str, dict],
    metrics: Mapping[str, Mapping[str, float]],
    k0: int,
    n_frames: int,
    t0: float,
    dt_frame: float,
    out_path: Path,
    *,
    actuator_mode: str = "real",
    step: int = 0,
    cf_nbi: Optional[np.ndarray] = None,
) -> Path:
    """Render one multi-panel rollout comparison figure in physical units and ms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = select_panel_specs(decoded)
    n_panels = max(1, len(panels))
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(10.5, max(3.2, 1.55 * n_panels + 0.9)), sharex=True
    )
    if n_panels == 1:
        axes = [axes]

    t_start_ms = float(t0) * 1e3
    t_seed_end_ms = (float(t0) + int(k0) * float(dt_frame)) * 1e3
    t_end_ms = (float(t0) + int(n_frames) * float(dt_frame)) * 1e3

    for idx, (ax, spec) in enumerate(zip(axes, panels)):
        entry = decoded[spec.name]
        ax.axvspan(t_start_ms, t_seed_end_ms, color="#DCE9F5", alpha=0.65, zorder=0)
        ax.axvline(t_seed_end_ms, color="0.35", ls="--", lw=0.9, zorder=4)

        ylabel = f"{spec.label}\n({spec.unit})" if spec.unit else spec.label
        if not entry["present"] or entry["truth"] is None:
            ax.set_facecolor("0.95")
            ax.text(
                0.5, 0.5, "not recorded", transform=ax.transAxes,
                ha="center", va="center", color="0.45", fontsize=9,
            )
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_yticks([])
            continue

        times_ms = np.asarray(entry["times_s"], dtype=np.float64) * 1e3
        gt_1d, ceil_1d, roll_1d, ch_tag = _reduce_for_panel(
            entry["truth"], entry["ceiling"], entry["rollout"]
        )
        s = spec.scale
        ax.plot(times_ms, gt_1d * s, color="#111111", lw=1.35, label="Ground truth", zorder=2)
        ax.plot(times_ms, ceil_1d * s, color="0.52", lw=1.05, ls="--", label="Tokenizer ceiling", zorder=3)
        ax.plot(times_ms, roll_1d * s, color="#D62728", lw=1.15, alpha=0.9, label="Rollout", zorder=4)

        if spec.name == "summary-power_nbi" and actuator_mode != "real" and cf_nbi is not None:
            f_times_ms = (float(t0) + (np.arange(len(cf_nbi)) + 0.5) * float(dt_frame)) * 1e3
            ax.plot(
                f_times_ms, cf_nbi * s, color="#1F77B4", lw=1.25, ls=":",
                label=f"Actuator ({actuator_mode})", zorder=5,
            )

        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        m = metrics.get(spec.name, {})
        r_val = m.get("rollout_nrmse", float("nan"))
        c_val = m.get("ceiling_nrmse", float("nan"))
        ann_parts = []
        if ch_tag:
            ann_parts.append(ch_tag)
        if math.isfinite(r_val):
            ann_parts.append(f"roll {r_val:.3f}")
        if math.isfinite(c_val):
            ann_parts.append(f"ceil {c_val:.3f}")
        if ann_parts:
            ax.text(
                0.995, 0.94, " | ".join(ann_parts),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color="0.2",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8),
            )
        if idx == 0:
            ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=True, framealpha=0.85, edgecolor="none")

    axes[-1].set_xlim(t_start_ms, t_end_ms)
    axes[-1].set_xlabel("Time (ms)", fontsize=10)
    fig.suptitle(
        f"MAST shot {shot} — IGNITE rollout (step {step}, k0={k0}, F={n_frames}, actuator={actuator_mode})",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
