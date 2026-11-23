"""Whole-shot autoregressive rollouts, decoding to physical units, and physics plots."""

from __future__ import annotations

import math
from collections.abc import Mapping
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


def _resolve_global_std(
    name: str,
    signal_stds: Mapping[str, float | None] | None,
    st: transform.ChannelStats,
    codec: tok_codecs.Codec,
) -> float:
    """Resolve a signal's global physical-unit std (never per-window std)."""
    if signal_stds is not None and name in signal_stds and signal_stds[name] is not None:
        val = float(signal_stds[name])
        if math.isfinite(val) and val > 0.0:
            return val
    if st.compress == "none":
        rms = float(np.sqrt(np.mean(np.asarray(st.std, dtype=np.float64) ** 2)))
        if math.isfinite(rms) and rms > 0.0:
            return rms
    val = float(codec.signal_std)
    return val if math.isfinite(val) and val > 0.0 else float("nan")


def _nrmse_global(pred: np.ndarray, gt: np.ndarray, global_std: float) -> float:
    """``rmse(pred - gt) / global_std`` over mutually finite entries in float64."""
    if not (math.isfinite(global_std) and global_std > 0.0):
        return float("nan")
    finite = np.isfinite(pred) & np.isfinite(gt)
    if not finite.any():
        return float("nan")
    err = pred[finite].astype(np.float64) - gt[finite].astype(np.float64)
    return float(np.sqrt(np.mean(err * err))) / float(global_std)


def _frame_channel_coverage(
    pred: np.ndarray,
    gt: np.ndarray,
    sample_f: np.ndarray,
    k0: int,
    n_frames: int,
) -> float:
    """Fraction of truth-finite ``(frame, channel)`` points where ``pred`` is finite."""
    n_pred_f = int(n_frames) - int(k0)
    if n_pred_f <= 0 or gt.shape[-1] == 0:
        return float("nan")
    n_ch = int(math.prod(gt.shape[:-1]))
    g = np.isfinite(gt).reshape(n_ch, -1)
    p = np.isfinite(pred).reshape(n_ch, -1)
    f_rel = np.asarray(sample_f, dtype=np.int64) - int(k0)
    gt_fc = np.zeros((n_ch, n_pred_f), dtype=bool)
    pr_fc = np.zeros((n_ch, n_pred_f), dtype=bool)
    np.logical_or.at(gt_fc.T, f_rel, g.T)
    np.logical_or.at(pr_fc.T, f_rel, p.T)
    denom = int(gt_fc.sum())
    if denom == 0:
        return float("nan")
    return float((gt_fc & pr_fc).sum() / denom)


def _missing_token_frac(codes: torch.Tensor, k0: int, n_frames: int, missing_code: int) -> float:
    """Fraction of predicted-frame tokens equal to ``missing_code``."""
    sub = codes[int(k0) : int(n_frames)]
    if sub.numel() == 0:
        return float("nan")
    return float((sub == int(missing_code)).float().mean().item())


def _align_truth_to_window(
    cs: cache.CachedSignal | None,
    st: transform.ChannelStats,
    t0: float,
    n_frames: int,
    dt_frame: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Slice cache-v1 signal to ``[t0, t0 + n_frames * dt_frame)`` in physical units.

    Samples are assigned to frames by sample centre time ``cs.t0 + (i + 0.5) * cs.dt``.
    Returns ``(truth_phys, sample_times_s, sample_frame_idx)`` or ``(None, None, None)``.
    """
    if cs is None or cs.values.shape[-1] == 0:
        return None, None, None
    t_len = int(cs.values.shape[-1])
    ts = framecache.sample_times(t_len, cs.t0, cs.dt)
    fi = framecache.sample_frame_index(t_len, cs.t0, cs.dt, t0, dt_frame=dt_frame)
    keep = (fi >= 0) & (fi < n_frames)
    if not keep.any():
        return None, None, None
    idx = np.flatnonzero(keep)
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    phys = transform.to_physical(cs.values[..., lo:hi], st)
    return phys, ts[lo:hi], fi[lo:hi]


def _decode_aligned(
    codec: tok_codecs.Codec,
    codes: torch.Tensor,
    sample_f: np.ndarray | None,
    n_frames: int,
    dt_frame: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``codes`` to physical units onto the exact frame indices ``sample_f``."""
    if sample_f is not None and len(sample_f) > 0:
        decoded = framecache.decode_signal_codes(
            codec, codes[:n_frames], dt_frame=dt_frame, sample_frame_idx=sample_f
        )
        return decoded, sample_f
    n_samp = framecache.samples_for_frames(n_frames, codec.dt, dt_frame=dt_frame)
    decoded = framecache.decode_signal_codes(
        codec, codes[:n_frames], n_samples=n_samp, dt_frame=dt_frame
    )
    return decoded, frames.frame_index(n_samp, codec.dt)


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
    signal_stds: Mapping[str, float | None] | None = None,
) -> tuple[dict[str, dict], dict[str, dict[str, float]]]:
    """Decode ground-truth codes (ceiling) and predicted codes (rollout) to physical units.

    Computes over the predicted frames ``[k0, F)`` using each signal's global physical std:
      - ``rollout_nrmse`` and ``ceiling_nrmse`` (shared global-std denominator);
      - ``pred_coverage`` and ``ceil_coverage`` (truth-finite ``(frame, channel)`` coverage);
      - ``pred_missing_frac`` and ``ceil_missing_frac`` (fraction of tokens equal to ``missing_code``).
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
        _n_bins, missing_code, _vocab = framecache.codec_vocab(codec)
        is_present = True if present is None else bool(present.get(name, True))
        cs = shot_data.get(name) if is_present else None
        truth_phys, times_s, sample_f = _align_truth_to_window(
            cs, stats[name], t0=t0, n_frames=n_frames, dt_frame=dt_frame
        )
        if truth_phys is None or sample_f is None or not is_present:
            decoded[name] = {
                "present": False,
                "times_s": framecache.sample_times(n_frames, t0, dt_frame),
                "truth": None,
                "ceiling": None,
                "rollout": None,
            }
            metrics[name] = {
                "rollout_nrmse": float("nan"),
                "ceiling_nrmse": float("nan"),
                "pred_coverage": float("nan"),
                "ceil_coverage": float("nan"),
                "pred_missing_frac": float("nan"),
                "ceil_missing_frac": float("nan"),
            }
            continue

        ceil_phys, sample_f = _decode_aligned(
            codec, rollout.gt_codes[name], sample_f, n_frames, dt_frame
        )
        pred_phys, _ = _decode_aligned(
            codec, rollout.pred_codes[name], sample_f, n_frames, dt_frame
        )
        pred_sel = sample_f >= k0
        global_std = _resolve_global_std(name, signal_stds, stats[name], codec)
        pred_miss = _missing_token_frac(rollout.pred_codes[name], k0, n_frames, missing_code)
        ceil_miss = _missing_token_frac(rollout.gt_codes[name], k0, n_frames, missing_code)
        if pred_sel.any():
            gt_p = truth_phys[..., pred_sel]
            pr_p = pred_phys[..., pred_sel]
            ce_p = ceil_phys[..., pred_sel]
            sf_p = sample_f[pred_sel]
            r_nrmse = _nrmse_global(pr_p, gt_p, global_std)
            c_nrmse = _nrmse_global(ce_p, gt_p, global_std)
            pred_cov = _frame_channel_coverage(pr_p, gt_p, sf_p, k0, n_frames)
            ceil_cov = _frame_channel_coverage(ce_p, gt_p, sf_p, k0, n_frames)
        else:
            r_nrmse = c_nrmse = pred_cov = ceil_cov = float("nan")

        decoded[name] = {
            "present": True,
            "times_s": times_s,
            "truth": truth_phys,
            "ceiling": ceil_phys,
            "rollout": pred_phys,
        }
        metrics[name] = {
            "rollout_nrmse": r_nrmse,
            "ceiling_nrmse": c_nrmse,
            "pred_coverage": pred_cov,
            "ceil_coverage": ceil_cov,
            "pred_missing_frac": pred_miss,
            "ceil_missing_frac": ceil_miss,
        }
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
        s = spec.scale
        show_cf = (
            spec.name == "summary-power_nbi"
            and actuator_mode != "real"
            and cf_nbi is not None
            and len(cf_nbi) > int(k0)
        )
        if not entry["present"] or entry["truth"] is None:
            if show_cf:
                pred_idx = np.arange(int(k0), len(cf_nbi))
                f_times_ms = (float(t0) + (pred_idx + 0.5) * float(dt_frame)) * 1e3
                (cf_line,) = ax.plot(
                    f_times_ms,
                    cf_nbi[pred_idx] * s,
                    color="#1F77B4",
                    lw=1.35,
                    ls=":",
                    label=f"Actuator ({actuator_mode})",
                    zorder=5,
                )
                ax.legend(
                    handles=[cf_line],
                    loc="upper left",
                    fontsize=8,
                    frameon=True,
                    framealpha=0.85,
                    edgecolor="none",
                )
                ax.set_ylabel(ylabel, fontsize=9)
                ax.grid(alpha=0.25, lw=0.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                continue
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
        ax.plot(times_ms, gt_1d * s, color="#111111", lw=1.35, label="Ground truth", zorder=2)
        ax.plot(times_ms, ceil_1d * s, color="0.52", lw=1.05, ls="--", label="Tokenizer ceiling", zorder=3)
        ax.plot(times_ms, roll_1d * s, color="#D62728", lw=1.15, alpha=0.9, label="Rollout", zorder=4)

        cf_line = None
        if show_cf:
            pred_idx = np.arange(int(k0), len(cf_nbi))
            f_times_ms = (float(t0) + (pred_idx + 0.5) * float(dt_frame)) * 1e3
            (cf_line,) = ax.plot(
                f_times_ms,
                cf_nbi[pred_idx] * s,
                color="#1F77B4",
                lw=1.35,
                ls=":",
                label=f"Actuator ({actuator_mode})",
                zorder=5,
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
        elif cf_line is not None:
            ax.legend(
                handles=[cf_line],
                loc="upper left",
                fontsize=8,
                frameon=True,
                framealpha=0.85,
                edgecolor="none",
            )

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
