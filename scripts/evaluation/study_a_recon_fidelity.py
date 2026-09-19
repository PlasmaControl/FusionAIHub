"""A3 + A4: reconstruction fidelity — hero panels & spectral-transfer curves.

A4 (``--analyses a4``): over N flat-top windows, accumulate the frequency-
marginal power of ground truth vs prediction for each spectrogram modality
(physical STFT-magnitude², via the inverse transform) and the Welch-style
PSD of the filterscopes windows. The pred/GT ratio-vs-frequency curve is the
direct, quantitative form of the "misses spectrogram detail" claim.

A3 (``--analyses a3``): stitched 4 s hero segments (stride = chunk, so
consecutive 50 ms K=1 predictions butt together) for best / median / worst
val shots ranked by the extraction sweep's mae/copy ratio — slow-TS overlays
in physical units + GT/pred/|diff| spectrogram triptychs.

Run inside an sbatch (single GCD): a4 ≈ 10 min, a3 ≈ 10 min.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from torch.utils.data import DataLoader  # noqa: E402

from tokamak_foundation_model.data.data_loader import (  # noqa: E402
    collate_fn_prediction,
)

from tfm_eval.ckpt import DEFAULT_CKPT, build_model_from_ckpt, load_ckpt  # noqa: E402
from tfm_eval.data import (  # noqa: E402
    assert_val_matches_training_cache,
    load_stats,
    make_prediction_dataset,
    resolve_split,
    shot_window_ranges,
)
from tfm_eval.inverse_preprocess import InversePreprocessor  # noqa: E402
from tfm_eval.latents import forward_with_latents  # noqa: E402
from tfm_eval.plotting import (  # noqa: E402
    psd_overlay,
    save_fig,
    set_style,
    spectro_triptych,
    ts_overlay,
)

SPECTRO_FS = 500_000.0
N_FFT, HOP = 1024, 256
TS_PANELS = [  # (signal, channel or "mean", ylabel)
    ("ts_core_density", "mean", r"$n_e$ core (avg)"),
    ("ts_core_temp", "mean", r"$T_e$ core (avg)"),
    ("cer_rot", "mean", "rotation (avg)"),
    ("cer_ti", "mean", r"$T_i$ (avg)"),
    ("filterscopes", "mean", r"D-$\alpha$ (avg)"),
]
HERO_SPECTRO_CH = {"ece": 20, "co2": 1, "bes": 8}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--analyses", default="a4,a3")
    ap.add_argument("--n-shots-a4", type=int, default=64)
    ap.add_argument("--windows-per-shot-a4", type=int, default=8)
    ap.add_argument("--n-hero-shots", type=int, default=6)
    ap.add_argument("--hero-span-s", type=float, nargs=2, default=(1.5, 5.5))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out-root",
        default=str(_HERE.parents[2] / "data" / "outputs" / "eval_suite"
                    / "e2e_stage1_best"),
    )
    return ap.parse_args()


def freq_axis(n_bins: int) -> np.ndarray:
    return np.fft.rfftfreq(N_FFT, d=1.0 / SPECTRO_FS)[:n_bins]


def run_a4(model, diagnostics, actuators, ck_args, stats, val_files, inv,
           args, fig_dir, tab_dir, t0):
    spectro = [c for c in diagnostics if c.kind == "spectrogram"]
    ck2 = dict(ck_args, warmup_s=2.0)
    ds = make_prediction_dataset(
        val_files[: args.n_shots_a4], stats, ck2, diagnostics, actuators,
        step_size_s=0.75,
        lengths_cache_path=tab_dir / "lengths_a4.pt",
        max_open_files=80,
    )
    indices = []
    for _, start, n in shot_window_ranges(ds):
        indices.extend(range(start, start + min(args.windows_per_shot_a4, n)))
    loader = DataLoader(
        ds, batch_size=args.batch_size, sampler=indices,
        num_workers=args.num_workers, collate_fn=collate_fn_prediction,
        pin_memory=args.device.startswith("cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    device = torch.device(args.device)

    acc = {c.name: {"gt": None, "pred": None, "n": 0} for c in spectro}
    fs_acc = {"gt": None, "pred": None, "n": 0}
    fs_win = torch.hann_window(500)
    done = 0
    for batch in loader:
        r = forward_with_latents(model, batch, device)
        for cfg in spectro:
            name = cfg.name
            valid = batch["inputs"][f"{name}_valid"] > 0
            if not valid.any():
                continue
            gt_z = r["targets"][name][valid].cpu().float()
            pr_z = r["predictions"][name][valid].cpu().float()
            gt_p = inv.inverse(name, gt_z) ** 2   # magnitude² power
            pr_p = inv.inverse(name, pr_z) ** 2
            g = gt_p.mean(dim=(0, 3))             # (C, F) time+batch marginal
            p = pr_p.mean(dim=(0, 3))
            a = acc[name]
            a["gt"] = g if a["gt"] is None else a["gt"] + g
            a["pred"] = p if a["pred"] is None else a["pred"] + p
            a["n"] += 1
        # filterscopes PSD (physical units, mean-removed, Hann)
        name = "filterscopes"
        valid = batch["inputs"][f"{name}_valid"] > 0
        if valid.any():
            gt = inv.inverse(name, r["targets"][name][valid].cpu().float())
            pr = inv.inverse(name, r["predictions"][name][valid].cpu().float())

            def psd(x):
                x = (x - x.mean(-1, keepdim=True)) * fs_win
                return (torch.fft.rfft(x, dim=-1).abs() ** 2).mean(dim=(0, 1))

            g, p = psd(gt), psd(pr)
            fs_acc["gt"] = g if fs_acc["gt"] is None else fs_acc["gt"] + g
            fs_acc["pred"] = p if fs_acc["pred"] is None else fs_acc["pred"] + p
            fs_acc["n"] += 1
        done += next(iter(batch["inputs"].values())).shape[0]
        if done % 80 < args.batch_size:
            print(f"[{time.time()-t0:7.1f}s] a4 {done}/{len(indices)}",
                  flush=True)

    out = {}
    set_style()
    fig, axs = plt.subplots(2, len(spectro) + 1,
                            figsize=(3.2 * (len(spectro) + 1), 5.2),
                            sharex="col")
    for i, cfg in enumerate(spectro):
        a = acc[cfg.name]
        if a["n"] == 0:
            continue
        gt = (a["gt"] / a["n"]).mean(0).numpy()      # (F,) channel-avg
        pr = (a["pred"] / a["n"]).mean(0).numpy()
        f = freq_axis(gt.size)
        out[f"{cfg.name}_gt"], out[f"{cfg.name}_pred"] = gt, pr
        out[f"{cfg.name}_f_hz"] = f
        psd_overlay(axs[0, i], f[1:], gt[1:], pr[1:])
        axs[0, i].set_title(cfg.name)
        axs[0, i].set_xlabel("")
        axs[1, i].plot(f[1:], pr[1:] / np.maximum(gt[1:], 1e-30),
                       color="#0072B2", lw=1.5)
        axs[1, i].axhline(1.0, color="0.5", lw=0.8, ls=":")
        axs[1, i].set_xscale("log"); axs[1, i].set_yscale("log")
        axs[1, i].set_xlabel("frequency (Hz)")
        axs[1, i].set_ylabel("pred/GT power" if i == 0 else "")
    if fs_acc["n"]:
        g = fs_acc["gt"].numpy() / fs_acc["n"]
        p = fs_acc["pred"].numpy() / fs_acc["n"]
        f = np.fft.rfftfreq(500, d=1e-4)
        out["filterscopes_gt"], out["filterscopes_pred"] = g, p
        out["filterscopes_f_hz"] = f
        psd_overlay(axs[0, -1], f[1:], g[1:], p[1:])
        axs[0, -1].set_title("filterscopes (PSD)")
        axs[0, -1].set_xlabel("")
        axs[1, -1].plot(f[1:], p[1:] / np.maximum(g[1:], 1e-30),
                        color="#0072B2", lw=1.5)
        axs[1, -1].axhline(1.0, color="0.5", lw=0.8, ls=":")
        axs[1, -1].set_xscale("log"); axs[1, -1].set_yscale("log")
        axs[1, -1].set_xlabel("frequency (Hz)")
    fig.suptitle("A4 — spectral fidelity: frequency-marginal power, "
                 "prediction vs ground truth (val flat-top)")
    save_fig(fig, fig_dir / "a4_spectral_fidelity")
    np.savez(tab_dir / "a4_spectral_fidelity.npz",
             **out, n_windows=np.array([acc[c.name]["n"] for c in spectro]))
    print(f"[{time.time()-t0:7.1f}s] A4 done "
          f"(windows used per spectro: "
          f"{ {c.name: acc[c.name]['n'] for c in spectro} })")


def pick_hero_shots(lat_dir: Path, n: int):
    """Rank shots by pooled mae/copy ratio from the sweep's per-shot csvs."""
    rows = []
    for csv_path in sorted(lat_dir.glob("per_shot_shard*.csv")):
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                try:
                    r = [float(row[f"mae_{m}"]) / float(row[f"copy_mae_{m}"])
                         for m in ("ts_core_density", "ts_core_temp", "ece")]
                except (ValueError, ZeroDivisionError, KeyError):
                    continue
                if all(np.isfinite(r)):
                    rows.append((row["shot"], float(np.mean(r))))
    if not rows:
        return None
    rows.sort(key=lambda x: x[1])
    k = max(1, n // 3)
    mid = len(rows) // 2
    picked = rows[:k] + rows[mid - k // 2: mid - k // 2 + k] + rows[-k:]
    return [(s, f"ratio={v:.3f}") for s, v in picked]


def run_a3(model, diagnostics, actuators, ck_args, stats, val_files, inv,
           args, fig_dir, tab_dir, t0):
    lat_dir = Path(args.out_root) / "latents" / "val"
    heroes = pick_hero_shots(lat_dir, args.n_hero_shots)
    by_name = {p.name.replace("_processed.h5", ""): p for p in val_files}
    if heroes is None:
        heroes = [(p.name.replace("_processed.h5", ""), "unranked")
                  for p in val_files[:3]]
        print("WARNING: no per-shot csvs yet — using first 3 val shots")
    heroes = [(s, note) for s, note in heroes if s in by_name]
    chunk = ck_args["chunk_duration_s"]
    t_lo, t_hi = args.hero_span_s
    n_win = int(round((t_hi - t_lo) / chunk))
    device = torch.device(args.device)
    spectro_names = [c.name for c in diagnostics if c.kind == "spectrogram"]

    for shot, note in heroes:
        ck2 = dict(ck_args, warmup_s=t_lo)
        ds = make_prediction_dataset(
            [by_name[shot]], stats, ck2, diagnostics, actuators,
            step_size_s=chunk,
            lengths_cache_path=None, max_open_files=4,
        )
        n_avail = len(ds)
        take = min(n_win, n_avail)
        loader = DataLoader(
            ds, batch_size=args.batch_size, sampler=range(take),
            num_workers=args.num_workers, collate_fn=collate_fn_prediction,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        stitched_gt, stitched_pred, stitched_valid = {}, {}, {}
        ts_names = [p[0] for p in TS_PANELS]
        for batch in loader:
            r = forward_with_latents(model, batch, device)
            for name in ts_names + spectro_names:
                gt = r["targets"][name].cpu().float()
                pr = r["predictions"][name].cpu().float()
                v = (batch["inputs"].get(f"{name}_valid",
                                         torch.ones(gt.shape[0])) > 0)
                stitched_gt.setdefault(name, []).append(gt)
                stitched_pred.setdefault(name, []).append(pr)
                stitched_valid.setdefault(name, []).append(v)

        def cat(d, name):
            # (W, C, ..., T_win) → windows concatenated on the time axis.
            x = torch.cat(d[name], 0)
            return torch.cat(list(x.unbind(0)), dim=-1)

        # slow-TS overlay figure -------------------------------------------
        set_style()
        fig, axs = plt.subplots(len(TS_PANELS), 1, sharex=True,
                                figsize=(7.0, 1.7 * len(TS_PANELS)))
        for ax, (name, ch, ylabel) in zip(axs, TS_PANELS):
            gt = cat(stitched_gt, name)               # (C, T_total)
            pr = cat(stitched_pred, name)
            gt_p = inv.inverse(name, gt).numpy()
            pr_p = inv.inverse(name, pr).numpy()
            g = gt_p.mean(0) if ch == "mean" else gt_p[ch]
            p = pr_p.mean(0) if ch == "mean" else pr_p[ch]
            t = t_lo + chunk + np.arange(g.size) * (take * chunk / g.size)
            ts_overlay(ax, t, g, p, ylabel=ylabel)
        axs[-1].set_xlabel("time (s)")
        fig.suptitle(f"shot {shot} — stitched K=1 predictions ({note})")
        save_fig(fig, fig_dir / f"a3_hero_ts_{shot}")

        # spectrogram triptychs --------------------------------------------
        for name in spectro_names:
            if not torch.cat(stitched_valid[name]).any():
                continue
            ch = HERO_SPECTRO_CH.get(name, 0)
            # Inverse needs the full channel axis (per-channel stats); select
            # the display channel after.
            gt_p = inv.inverse(name, cat(stitched_gt, name)).numpy()[ch] ** 2
            pr_p = inv.inverse(name, cat(stitched_pred, name)).numpy()[ch] ** 2
            f = freq_axis(gt_p.shape[0])
            t = t_lo + chunk + np.arange(gt_p.shape[1]) * HOP / SPECTRO_FS
            fig = spectro_triptych(
                gt_p, pr_p, t, f,
                title=f"shot {shot} — {name} ch{ch} ({note})",
            )
            save_fig(fig, fig_dir / f"a3_hero_spectro_{name}_{shot}")
        plt.close("all")
        print(f"[{time.time()-t0:7.1f}s] a3 hero {shot} done ({note})",
              flush=True)


def main() -> int:
    args = parse_args()
    t0 = time.time()
    fig_dir = Path(args.out_root) / "figures" / "study_a"
    tab_dir = Path(args.out_root) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_ckpt(args.checkpoint)
    ck_args = ckpt.get("args", {}) or {}
    model, diagnostics, actuators = build_model_from_ckpt(ckpt, device=args.device)
    print(f"[{time.time()-t0:6.1f}s] model ready on {args.device}")

    _, val_files = resolve_split(
        ck_args.get("data_dir", "/lustre/orion/fus187/proj-shared/foundation_model"),
        seed=ck_args.get("seed", 42),
        val_fraction=ck_args.get("val_fraction", 0.1),
    )
    assert_val_matches_training_cache(val_files)
    stats = load_stats(ck_args["stats_path"])
    inv = InversePreprocessor(
        str(ck_args["stats_path"]),
        dataset_kwargs={"channels_to_use": {"tangtv": [4, 6]}},
    )

    analyses = set(args.analyses.split(","))
    if "a4" in analyses:
        run_a4(model, diagnostics, actuators, ck_args, stats, val_files, inv,
               args, fig_dir, tab_dir, t0)
    if "a3" in analyses:
        run_a3(model, diagnostics, actuators, ck_args, stats, val_files, inv,
               args, fig_dir, tab_dir, t0)
    print(f"[{time.time()-t0:7.1f}s] DONE → {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
