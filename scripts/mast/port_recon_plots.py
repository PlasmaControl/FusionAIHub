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
from tokamak_foundation_model.ignite.mast.port import framecache, layout, rollout as port_rollout
from tokamak_foundation_model.ignite.mast.port.layout import ev

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
            cmap = matplotlib.colormaps[cmap_name].copy()
            cmap.set_bad("0.80")
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
    codecs, stats, _assignment = framecache.build_mast_codecs(
        registry.all_signals(), cache_root, Path(manifest["fit_dir"]), Path(manifest["assignment_path"])
    )
    names = [r[0] for r in ROWS]

    model = cfg = step = None
    device = _resolve_device(device_spec)
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
        )
        png = plot_recon(
            str(shot), decoded, metrics, k0_eff, n_frames, t0, dt_frame,
            out_dir / f"recon_{shot}.png", step=step, with_rollout=model is not None,
        )
        per_shot[str(shot)] = {"k0": k0_eff, "n_frames": n_frames, "t0": t0,
                               "figure": str(png), "signals": metrics}
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
