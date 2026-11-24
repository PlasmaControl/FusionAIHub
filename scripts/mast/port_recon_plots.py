"""Truth vs tokenizer reconstruction (vs IGNITE rollout) as images, on MAST shots.

The signals a physicist reads as pictures:
- the two Mirnov spectrograms, frequency x time;
- the two soft X-ray cameras, chord x time;
- the Thomson profiles, channel x time;
- D-alpha, as traces.

"Tokenizer reconstruction" is the shot's own frame codes decoded back to physical units, the
ceiling any rollout can reach. With ``--ckpt``, a third column shows dev-peter's IGNITE rolled out
from ``k0`` true frames with the true actuators. Decoding and time alignment are
``port.rollout.decode_and_score_rollout``, the same code the rollout plots use.

MAST cache v1 holds no video camera: TokaMark's 39 signals carry none. FAIR-MAST level 1 has
video sources (rbb, rco, rbg), not downloaded here.

Usage::

    OMP_NUM_THREADS=8 scripts/mast/py.sh scripts/mast/port_recon_plots.py \
        --cache-dir <frame_codes_dir> --out-dir <plots_dir> [--ckpt <run>/dynamics_latest.pt]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tokamak_foundation_model.ignite.mast import paths, registry
from tokamak_foundation_model.ignite.mast.port import (
    camera,
    dalpha,
    framecache,
    layout,
    logspec,
    rollout as port_rollout,
)
from tokamak_foundation_model.ignite.mast.port.layout import ev, td

# TokaMark's STFT (tokamark/tools/transforms/stft_transform.py, support_n=512): 512-sample Hann
# windows without overlap on the 500 kHz Mirnov waveform, so 1.024 ms frames and 256 bins of
# 500/512 kHz each, |X|^2 (linear power).
MIRNOV_BIN_KHZ = 500.0 / 512.0

# (signal, kind, label, unit, scale)
ROWS: tuple[tuple[str, str, str, str, float], ...] = (
    ("magnetics-b_field_pol_probe_omv_voltage", "spec", "Mirnov OMV (poloidal)", "log10 power", 1.0),
    ("magnetics-b_field_tor_probe_cc_field", "spec", "Mirnov CC (toroidal)", "log10 power", 1.0),
    ("soft_x_rays-horizontal_cam_upper", "image", "SXR camera, upper", "V", 1.0),
    ("soft_x_rays-horizontal_cam_lower", "image", "SXR camera, lower", "V", 1.0),
    ("thomson_scattering-t_e", "image", "Thomson $T_e$", "keV", 1e-3),
    ("thomson_scattering-n_e", "image", "Thomson $n_e$", r"$10^{19}$ m$^{-3}$", 1e-19),
    ("spectrometer_visible-filter_spectrometer_dalpha_voltage", "trace", r"$D_\alpha$", "V", 1.0),
)
CMAPS = {"spec": "magma", "image": "inferno"}
PROFILE_CMAP = "viridis"
DEFAULT_SHOTS = "11914,16052,20384,23403,24934,28101"


def _resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _pick_probe(truth: np.ndarray) -> int:
    """The probe with the most finite samples (ties: the first)."""
    fin = np.isfinite(truth).reshape(truth.shape[0], -1).mean(axis=1)
    return int(np.argmax(fin))


def _log_power(arr: np.ndarray, floor: float) -> np.ndarray:
    """log10 of power, clipped at ``floor`` (DCT decoding can undershoot below 0); NaN stays NaN."""
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(arr)
    out[ok] = np.log10(np.maximum(arr[ok], floor))
    return out


def _limits(a: np.ndarray, lo: float = 1.0, hi: float = 99.5) -> tuple[float, float]:
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(fin, [lo, hi])
    if not vmax > vmin:
        vmax = vmin + 1e-12
    return float(vmin), float(vmax)


def _fmt_metric(m: dict, key: str) -> str:
    v = m.get(key, float("nan"))
    return f"{v:.3f}" if v is not None and math.isfinite(v) else "n/a"


def plot_recon(
    shot: str,
    decoded: dict,
    metrics: dict,
    k0: int,
    n_frames: int,
    t0: float,
    dt_frame: float,
    out_path: Path,
    *,
    step: int | None,
    with_rollout: bool,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["truth", "ceiling"] + (["rollout"] if with_rollout else [])
    titles = {
        "truth": "Ground truth",
        "ceiling": "Tokenizer reconstruction",
        "rollout": f"IGNITE rollout (step {step}, real actuators)",
    }
    rows = [r for r in ROWS if r[0] in decoded]
    n_r, n_c = len(rows), len(keys)
    fig, axes = plt.subplots(
        n_r, n_c, figsize=(4.9 * n_c + 1.3, 2.25 * n_r + 0.6), squeeze=False, layout="constrained"
    )
    t_lo = float(t0) * 1e3
    t_hi = (float(t0) + int(n_frames) * float(dt_frame)) * 1e3
    t_seed = (float(t0) + int(k0) * float(dt_frame)) * 1e3
    nrmse_scope = "predicted frames" if with_rollout else "whole window"

    for r, (name, kind, label, unit, scale) in enumerate(rows):
        ent, m = decoded[name], metrics.get(name, {})
        if not ent["present"] or ent["truth"] is None:
            for c in range(n_c):
                ax = axes[r, c]
                ax.set_facecolor("0.93")
                ax.text(0.5, 0.5, "not recorded in this shot", transform=ax.transAxes,
                        ha="center", va="center", color="0.4", fontsize=9)
                ax.set_yticks([])
                ax.set_xlim(t_lo, t_hi)
            axes[r, 0].set_ylabel(label, fontsize=9)
            continue

        times_ms = np.asarray(ent["times_s"], dtype=np.float64) * 1e3
        half = 0.5 * (times_ms[1] - times_ms[0]) if len(times_ms) > 1 else 0.5 * dt_frame * 1e3
        x0, x1 = times_ms[0] - half, times_ms[-1] + half

        if kind == "trace":
            truth = np.asarray(ent["truth"], dtype=np.float64)
            var = port_rollout._channel_variance(truth)
            ch = int(np.argmax(var)) if np.isfinite(var).any() else 0
            y0, y1 = _limits(truth[ch] * scale, 0.2, 99.8)
            pad = 0.08 * (y1 - y0)
            for c, key in enumerate(keys):
                ax = axes[r, c]
                if key != "truth":
                    ax.plot(times_ms, truth[ch] * scale, color="0.72", lw=0.8, label="truth")
                color = {"truth": "#111111", "ceiling": "#4C72B0", "rollout": "#D62728"}[key]
                ax.plot(times_ms, np.asarray(ent[key])[ch] * scale, color=color, lw=0.9)
                ax.set_ylim(y0 - pad, y1 + pad)
                ax.grid(alpha=0.25, lw=0.5)
            axes[r, 0].set_ylabel(f"{label} ch {ch}\n({unit})", fontsize=9)
        else:
            if kind == "spec":
                truth3 = np.asarray(ent["truth"], dtype=np.float64)
                p = _pick_probe(truth3)
                pos = truth3[p][np.isfinite(truth3[p]) & (truth3[p] > 0)]
                floor = float(np.percentile(pos, 0.1)) if pos.size else 1e-30
                imgs = {k: _log_power(np.asarray(ent[k], dtype=np.float64)[p], floor) for k in keys}
                n_y = imgs["truth"].shape[0]
                extent = (x0, x1, 0.0, n_y * MIRNOV_BIN_KHZ)
                ylabel = f"{label}\nprobe {p}, f (kHz)"
                cmap_name = CMAPS["spec"]
            else:
                imgs = {k: np.asarray(ent[k], dtype=np.float64) * scale for k in keys}
                n_y = imgs["truth"].shape[0]
                extent = (x0, x1, -0.5, n_y - 0.5)
                axis_name = "chord" if name.startswith("soft_x_rays") else "channel"
                ylabel = f"{label}\n{axis_name}"
                cmap_name = PROFILE_CMAP if name.startswith("thomson") else CMAPS["image"]
            vmin, vmax = _limits(imgs["truth"])
            cmap = matplotlib.colormaps[cmap_name].with_extremes(bad="0.80")
            im = None
            for c, key in enumerate(keys):
                im = axes[r, c].imshow(
                    np.ma.masked_invalid(imgs[key]), aspect="auto", origin="lower", extent=extent,
                    vmin=vmin, vmax=vmax, cmap=cmap, interpolation="antialiased",
                )
            fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.015, pad=0.005).set_label(unit, fontsize=8)
            axes[r, 0].set_ylabel(ylabel, fontsize=9)

        for c, key in enumerate(keys):
            ax = axes[r, c]
            ax.set_xlim(t_lo, t_hi)
            ax.tick_params(labelsize=7.5)
            if key == "rollout":
                ax.axvline(t_seed, color="#00E5FF", ls="--", lw=1.0)
            if key == "ceiling" or key == "rollout":
                if kind == "spec" and ("ceiling_dlog10" in m or "rollout_dlog10" in m):
                    metric = "ceiling_dlog10" if key == "ceiling" else "rollout_dlog10"
                    txt = f"|dlog10| {_fmt_metric(m, metric)} dec"
                else:
                    metric = "ceiling_nrmse" if key == "ceiling" else "rollout_nrmse"
                    txt = f"NRMSE {_fmt_metric(m, metric)}"
                if key == "rollout":
                    cov = m.get("pred_coverage", float("nan"))
                    if cov is not None and math.isfinite(cov):
                        txt += f", coverage {cov:.0%}"
                ax.text(0.99, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
                        fontsize=7.5, color="white",
                        bbox=dict(boxstyle="square,pad=0.2", fc="black", ec="none", alpha=0.55))
            if r == 0:
                ax.set_title(titles[key], fontsize=10)
            if r == n_r - 1:
                ax.set_xlabel("Time (ms)", fontsize=9)

    head = f"MAST shot {shot}: {n_frames} frames of {dt_frame * 1e3:g} ms from t0 = {t_lo:.1f} ms"
    if with_rollout:
        head += f"; rollout seeded with {k0} true frames (cyan line), then predicted"
    head += f".  NRMSE = RMSE / global physical std, over the {nrmse_scope}."
    fig.suptitle(head, fontsize=10)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _camera_panel_nrmse(u_rec: np.ndarray, u_truth: np.ndarray, u_std: float) -> float:
    fin = np.isfinite(u_rec) & np.isfinite(u_truth)
    if not fin.any():
        return float("nan")
    denom = float(u_std)
    if not (math.isfinite(denom) and denom > 0.0):
        denom = float(np.std(u_truth[fin]))
    if not (math.isfinite(denom) and denom > 0.0):
        return float("nan")
    err = u_rec[fin].astype(np.float64) - u_truth[fin].astype(np.float64)
    return float(np.sqrt(np.mean(err * err)) / denom)


def _camera_log_norm_bounds(truth_counts: np.ndarray, present_idx: np.ndarray) -> tuple[float, float]:
    vals = np.maximum(truth_counts[present_idx], 1.0)
    fin = vals[np.isfinite(vals)]
    if fin.size == 0:
        return 1.0, 4096.0
    vmin, vmax = np.percentile(fin, [1.0, 99.5])
    vmin = max(1.0, float(vmin))
    vmax = max(vmin * 2.0, float(vmax))
    return vmin, vmax


def plot_camera_recon(
    shot: str,
    cam_entry: dict,
    k0: int,
    t0: float,
    dt_frame: float,
    out_png: Path,
    out_gif: Path,
    *,
    step: int | None,
    with_rollout: bool,
    n_cols: int = 6,
    gif_fps: int = 10,
) -> tuple[Path | None, Path | None]:
    """Render ``recon_camera_<shot>.png`` (6 frames) and ``recon_camera_<shot>.gif`` (~10 fps)."""
    if not cam_entry.get("present") or cam_entry.get("truth") is None:
        return None, None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    truth = np.asarray(cam_entry["truth"], dtype=np.float32)
    ceil = np.asarray(cam_entry["ceiling"], dtype=np.float32)
    roll = np.asarray(cam_entry["rollout"], dtype=np.float32)
    truth_u = np.asarray(cam_entry["truth_u"], dtype=np.float32)
    ceil_u = np.asarray(cam_entry["ceiling_u"], dtype=np.float32)
    roll_u = np.asarray(cam_entry["rollout_u"], dtype=np.float32)
    u_std = float(cam_entry.get("u_std", float("nan")))
    times_ms = np.asarray(cam_entry["times_s"], dtype=np.float64) * 1e3

    present_idx = np.flatnonzero(np.asarray(cam_entry["frame_valid"], dtype=bool))
    if present_idx.size == 0:
        return None, None

    vmin, vmax = _camera_log_norm_bounds(truth, present_idx)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps["inferno"].with_extremes(bad="0.25")

    row_keys = ["truth", "ceiling"] + (["rollout"] if with_rollout else [])
    row_labels = {
        "truth": "Truth",
        "ceiling": "Tokenizer recon",
        "rollout": f"Rollout (step {step})",
    }
    imgs_by_key = {"truth": truth, "ceiling": ceil, "rollout": roll}
    u_by_key = {"truth": truth_u, "ceiling": ceil_u, "rollout": roll_u}

    k_cols = min(int(n_cols), int(present_idx.size))
    sel_pos = np.linspace(0, present_idx.size - 1, k_cols).round().astype(int)
    sel_frames = present_idx[sel_pos]

    fig, axes = plt.subplots(
        len(row_keys),
        k_cols,
        figsize=(2.4 * k_cols + 0.9, 2.0 * len(row_keys) + 0.7),
        squeeze=False,
        layout="constrained",
    )
    im = None
    for r, key in enumerate(row_keys):
        for c, f_idx in enumerate(sel_frames):
            ax = axes[r, c]
            frame_img = np.where(
                np.isfinite(imgs_by_key[key][f_idx]),
                np.maximum(imgs_by_key[key][f_idx], 1.0),
                np.nan,
            )
            im = ax.imshow(
                np.ma.masked_invalid(frame_img),
                cmap=cmap,
                norm=norm,
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"f={int(f_idx)} ({times_ms[f_idx]:.0f} ms)", fontsize=8.5)
            if c == 0:
                ax.set_ylabel(row_labels[key], fontsize=9)
            if key != "truth":
                nrmse_u = _camera_panel_nrmse(u_by_key[key][f_idx], truth_u[f_idx], u_std)
                txt = f"NRMSE(u) {nrmse_u:.3f}" if math.isfinite(nrmse_u) else "NRMSE(u) n/a"
                ax.text(
                    0.97, 0.94, txt, transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color="white",
                    bbox=dict(boxstyle="square,pad=0.18", fc="black", ec="none", alpha=0.6),
                )
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01).set_label(
            "counts (log scale)", fontsize=8.5
        )
    head = f"MAST shot {shot} rgb camera (t0={t0 * 1e3:.1f} ms, dt={dt_frame * 1e3:g} ms"
    if with_rollout:
        head += f", k0={k0}"
    head += ")"
    fig.suptitle(head, fontsize=10)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    # GIF: truth | reconstruction (| rollout) side by side over all present frames at ~10 fps.
    fig_g, axes_g = plt.subplots(
        1, len(row_keys), figsize=(3.2 * len(row_keys) + 0.6, 3.1), squeeze=False, layout="constrained"
    )
    ims_g = []
    f0 = int(present_idx[0])
    for c, key in enumerate(row_keys):
        ax = axes_g[0, c]
        img0 = np.where(
            np.isfinite(imgs_by_key[key][f0]),
            np.maximum(imgs_by_key[key][f0], 1.0),
            np.nan,
        )
        im_c = ax.imshow(np.ma.masked_invalid(img0), cmap=cmap, norm=norm, aspect="equal")
        ax.set_title(row_labels[key], fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ims_g.append(im_c)
    fig_g.colorbar(ims_g[-1], ax=axes_g[0, :].tolist(), fraction=0.025, pad=0.02).set_label(
        "counts", fontsize=8
    )
    title_g = fig_g.suptitle(f"MAST shot {shot} f={f0} ({times_ms[f0]:.0f} ms)", fontsize=10)

    def _update(i: int):
        fi = int(present_idx[i])
        for im_c, key in zip(ims_g, row_keys):
            arr = imgs_by_key[key][fi]
            frame_img = np.where(np.isfinite(arr), np.maximum(arr, 1.0), np.nan)
            im_c.set_data(np.ma.masked_invalid(frame_img))
        title_g.set_text(f"MAST shot {shot} f={fi} ({times_ms[fi]:.0f} ms)")
        return [*ims_g, title_g]

    anim = animation.FuncAnimation(
        fig_g, _update, frames=len(present_idx), interval=int(round(1000.0 / gif_fps)), blit=False
    )
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_gif), writer=animation.PillowWriter(fps=gif_fps))
    plt.close(fig_g)
    return out_png, out_gif


def run_recon_plots(
    *,
    cache_dir: Path,
    out_dir: Path,
    shots: list[str],
    ckpt: Path | None = None,
    k0: int = 0,
    horizon: int = 0,
    temperature: float = 1.0,
    seed: int = 0,
    device_spec: str = "auto",
) -> dict:
    cache_dir = Path(cache_dir)
    out_dir = paths.prepare_write_root(Path(out_dir))
    manifest = layout.load_manifest(cache_dir)
    dt_frame = float(manifest.get("dt_frame", framecache.DT_FRAME))
    cache_root = Path(manifest["cache_v1_root"])
    stats_sha = str(manifest.get("stats_sha256") or "") or None
    signal_stds = manifest.get("signal_physical_std")
    logspec_fit = logspec.resolve_manifest_logspec_fit(manifest)
    codecs, stats, _assignment = framecache.build_mast_codecs(
        registry.all_signals(),
        cache_root,
        Path(manifest["fit_dir"]),
        Path(manifest["assignment_path"]),
        logspec_fit=logspec_fit,
        dalpha_fit=dalpha.resolve_manifest_dalpha_fit(manifest),
    )
    names = [r[0] for r in ROWS]

    device = _resolve_device(device_spec)
    cam_codec = cam_fit = cam_frames_dir = None
    manifest_modalities = manifest.get("modalities") or {}
    has_camera = any(m in manifest_modalities for m in camera.CAMERA_MODALITIES)
    if has_camera:
        cam_paths = camera.resolve_manifest_camera(manifest)
        if cam_paths is None:
            raise ValueError(
                "manifest modalities include camera-rgb / camera-rgb-stats "
                "but camera_codec_path / camera_fit_path / camera_frames_dir are absent"
            )
        cam_codec, _vcfg = td._load_codec("video", cam_paths["codec_path"])
        cam_codec = cam_codec.to(device)
        cam_fit = camera.load_camera_fit(cam_paths["fit_path"])
        cam_frames_dir = cam_paths["frames_dir"]
        names = names + list(camera.CAMERA_MODALITIES)

    model = cfg = step = None
    if ckpt is not None:
        model, cfg, step = port_rollout.load_mast_model(ckpt, cache_dir, device)
    else:
        layout.patch_ignite(cache_dir)

    per_shot: dict[str, dict] = {}
    for shot in shots:
        if model is not None:
            ro, entry, _ = port_rollout.rollout_mast_shot(
                model, cfg, cache_dir, shot, k0=k0, horizon=horizon, temperature=temperature,
                seed=seed, device=device, actuator_mode="real",
            )
            k0_eff, n_frames = int(ro.k0), int(ro.n_frames)
            gt = {n: ro.gt_codes[n] for n in names if n in ro.gt_codes}
            pred = {n: ro.pred_codes[n] for n in names if n in ro.pred_codes}
        else:
            entry = ev.load_shot_cache(cache_dir, str(shot))
            k0_eff, n_frames = 0, int(entry["n_frames"])
            gt = {n: entry["codes"][n] for n in names if n in entry["codes"]}
            pred = gt
        view = SimpleNamespace(gt_codes=gt, pred_codes=pred, k0=k0_eff, n_frames=n_frames)
        t0 = float(entry["t0"])
        decoded, metrics = port_rollout.decode_and_score_rollout(
            view, cache_root / "shots" / f"{shot}.h5", codecs, stats, t0=t0, dt_frame=dt_frame,
            present=entry.get("present"), stats_sha256=stats_sha, signal_stds=signal_stds,
            camera_codec=cam_codec, camera_fit=cam_fit, camera_frames_dir=cam_frames_dir,
            shot=str(shot),
        )
        png = plot_recon(
            str(shot), decoded, metrics, k0_eff, n_frames, t0, dt_frame,
            out_dir / f"recon_{shot}.png", step=step, with_rollout=model is not None,
        )
        shot_rec: dict = {
            "k0": k0_eff,
            "n_frames": n_frames,
            "t0": t0,
            "figure": str(png),
            "signals": metrics,
        }
        if camera.CAMERA_MODALITY in decoded:
            cam_png, cam_gif = plot_camera_recon(
                str(shot),
                decoded[camera.CAMERA_MODALITY],
                k0_eff,
                t0,
                dt_frame,
                out_dir / f"recon_camera_{shot}.png",
                out_dir / f"recon_camera_{shot}.gif",
                step=step,
                with_rollout=model is not None,
            )
            if cam_png is not None:
                shot_rec["camera_figure"] = str(cam_png)
            if cam_gif is not None:
                shot_rec["camera_gif"] = str(cam_gif)
        per_shot[str(shot)] = shot_rec
        print(f"[port_recon] shot {shot}: k0={k0_eff} F={n_frames} -> {png}", flush=True)

    record = {
        "ckpt": str(Path(ckpt).resolve()) if ckpt is not None else None,
        "step": step,
        "cache_dir": str(cache_dir.resolve()),
        "nrmse_scope": "predicted frames" if model is not None else "whole window",
        "per_shot": per_shot,
    }
    framecache.save_json_atomic(record, out_dir / "recon_metrics.json")
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser("MAST truth vs tokenizer reconstruction (vs rollout) images")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ckpt", default=None, help="Dynamics checkpoint; adds the rollout column")
    p.add_argument("--shots", default=DEFAULT_SHOTS, help="Comma-separated shot IDs")
    p.add_argument("--k0", type=int, default=0, help="Seed frames (0 = checkpoint k0_seed)")
    p.add_argument("--horizon", type=int, default=0, help="Predicted frames (0 = to max_frames)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    run_recon_plots(
        cache_dir=Path(a.cache_dir), out_dir=Path(a.out_dir),
        shots=[s.strip() for s in a.shots.split(",") if s.strip()],
        ckpt=Path(a.ckpt) if a.ckpt else None, k0=a.k0, horizon=a.horizon,
        temperature=a.temperature, seed=a.seed, device_spec=a.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
