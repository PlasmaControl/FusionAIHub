"""GROUND TRUTH vs RECONSTRUCTION of the raw 10 kHz filterscope trace — all 8 channels.

The deliverable figure for the 2026-09-03 fast-TS SAMPLE-WISE codec redesign, plus the
metrics table that must be read beside it.

Usage
-----
    python analysis/fastts_raw_recon_fig.py <codec_ckpt.pt> [more_ckpts...] \
        --out figures/fastts_raw_recon.png [--windows 3] [--n_score_shots 4]

What it draws
-------------
SMALL MULTIPLES: one panel per (channel, window) — the 8 filterscope channels are NOT eight
hues, they are eight facets, because the question ("does the reconstruction look like the
signal?") is asked once per channel. Exactly TWO series share a panel, so identity is carried
by a legend AND by line weight (thick pale ground truth under a thin saturated
reconstruction), never by colour alone.

Every panel is the FULL 50 ms window at the full 10 kHz rate — all 500 samples, no smoothing,
no decimation, no per-panel rescaling of the reconstruction. The panel title carries that
(window, channel)'s own nRMSE and correlation, so the picture and the number cannot drift
apart.

Metrics printed alongside use gate.full_fastts_metrics / gate.trivial_fastts_baselines: the
same normalisation as the training gate (RMSE / std(target) per (window, channel) over the
sample axis), with the two mandatory self-checks (self -> 0.0000, wcmean -> 1.0000).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokamak_foundation_model.ignite import gate, spike                       # noqa: E402
from tokamak_foundation_model.ignite.config import CHUNK_S, FASTTS_FS         # noqa: E402
from tokamak_foundation_model.ignite.fastts_codec import FastTSCodec          # noqa: E402

# Validated 2-slot categorical pair (dataviz reference palette, slots 1+2; all six checks
# PASS in light mode: worst CVD dE 24.7, normal-vision dE 33.6, contrast >= 3:1).
GT_COLOR = "#9aa3ad"        # ground truth: thick, recessive — it is the substrate
RECON_COLOR = "#eb6834"     # reconstruction: thin, saturated — the thing under test
INK = "#1c1c1c"
MUTED = "#6b7280"
SURFACE = "#fcfcfb"
MODALITY = "filterscopes"


def load_windows(n_fit: int, n_score: int, seed: int = 1234, max_win: int = 200):
    """Standardized raw (B, C, 500) windows from held-out shots (same picks as the floor probe)."""
    import h5py

    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset
    from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
    from tokamak_foundation_model.ignite.fastts_train import load_fastts_channel_stats
    from tokamak_foundation_model.ignite.train_codec import _shot_paths

    mean, std = load_fastts_channel_stats()
    cfg_sig = next(c for c in TokamakH5Dataset.SIGNAL_CONFIGS if c.name == MODALITY)
    W = round(CHUNK_S * FASTTS_FS)
    shots = spike.discover_shots(spike.DEFAULT_DATA_DIR)
    rng = np.random.default_rng(seed)
    cand = [shots[i] for i in rng.choice(len(shots), size=64, replace=False)]

    kept, ids, n_seen = [], [], 0
    for sid in cand:
        paths = _shot_paths([sid], spike.DEFAULT_DATA_DIR)
        if not paths:
            continue
        try:
            ds = TokamakMultiFileDataset(
                hdf5_paths=[paths[0]], chunk_duration_s=CHUNK_S, step_size_s=CHUNK_S,
                warmup_s=1.0, input_signals=[MODALITY], target_signals=[MODALITY],
                prediction_mode=False, lengths_cache_path=None, max_open_files=4)
            with h5py.File(str(paths[0]), "r") as h5:
                got = []
                for i in range(min(len(ds), max_win)):
                    ts = 1.0 + i * CHUNK_S
                    raw, valid_len, _ = ds._load_signal_raw(h5, cfg_sig, ts, ts + CHUNK_S)
                    if valid_len < W or not torch.isfinite(raw).all() or float(raw.std()) < 1e-9:
                        continue
                    got.append(raw[..., :W].numpy().astype(np.float64))
        except Exception:
            continue
        if len(got) < 30:
            continue
        n_seen += 1
        if n_seen <= n_fit:            # skip the shots the floor probe FIT on
            continue
        kept.append(np.stack(got, 0)); ids.append(sid)
        if len(kept) >= n_score:
            break
    X = np.concatenate(kept, 0)
    X = (X - np.asarray(mean)[None, :, None]) / np.maximum(np.asarray(std)[None, :, None], 1e-3)
    return X, ids


def run_codec(ckpt_path: str, X: np.ndarray, batch: int = 64):
    """Load a fast-TS codec checkpoint and reconstruct ``X`` (B, C, W). Returns (recon, cfg, ck)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    codec = FastTSCodec(cfg)
    codec.load_state_dict(ck["codec"])
    codec.eval()
    out, codes = [], []
    x = torch.from_numpy(X).float()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            o = codec.forward(x[i:i + batch])
            out.append(o["recon"]); codes.append(o["codes"])
    return torch.cat(out).numpy().astype(np.float64), torch.cat(codes), cfg, ck


def report(name: str, R: np.ndarray, X: np.ndarray, codes, cfg, ck) -> dict:
    m = gate.full_fastts_metrics(R, X)
    m.update(gate.fastts_ssim_metrics(R, X))
    b = gate.trivial_fastts_baselines(X)
    refs = gate.fastts_structural_references(X)
    u = gate.utilization(codes.reshape(1, *codes.shape), codebook_size=cfg.codebook_size)
    g = ck.get("gate", {}) or {}
    fc = g.get("forecastability", {}) or {}
    print(f"\n=== {name}  (step {ck.get('step')}, n_tok {cfg.n_tok}, "
          f"{cfg.bits_per_value:.4f} bits/value, frame {1017 - 5 + cfg.n_tok}) ===")
    print(f"  nRMSE (raw samples)      {m['fastts_nrmse']:.4f}      corr {m['fastts_corr']:+.4f}")
    print(f"  nRMSE (global, level-aware) {m['fastts_global_nrmse']:.4f}   "
          f"valid_frac {m['fastts_valid_frac']:.3f}")
    print(f"  BASELINES  self {b['base_self_fastts_nrmse']:.4f} (must be 0.0000)   "
          f"wcmean {b['base_wcmean_fastts_nrmse']:.4f} (must be 1.0000)")
    print(f"             tmean {b['base_tmean_fastts_nrmse']:.4f} (THE bar)   "
          f"cmean {b['base_cmean_fastts_nrmse']:.4f}   "
          f"binmean100 {b['base_binmean100_fastts_nrmse']:.4f} (old envelope resolution)")
    print(f"  utilization  {u['n_distinct_codes']}/{cfg.codebook_size} "
          f"= {u['frac_codes_used']:.1%}  frac_of_observable {u['frac_of_observable']:.3f}  "
          f"minH {u['min_dim_entropy']:.4f}  collapsed {u['collapsed']}")
    print(f"  forecast_margin_tr {fc.get('margin_transition', float('nan')):+.4f}  "
          f"beats_persistence {fc.get('beats_persistence')}"
          + ("   [no gate dict in this ckpt]" if not fc else ""))
    print(f"  STRUCTURAL (the ranking read; nRMSE is only a FLOOR)")
    print(f"    ssim_cs {m['fastts_ssim_cs']:.4f}   CONTRAST {m['fastts_ssim_contrast']:.4f}"
          f"   structure {m['fastts_ssim_structure']:.4f}   ms_ssim {m['fastts_ms_ssim']:.4f}")
    print(f"    std_ratio {m['fastts_std_ratio']:.4f}  "
          f"(1.0 = right dynamic range, <1 = amplitude-COMPRESSED / blurred)")
    print(f"  METRIC VALIDATION on this same batch "
          f"(a degenerate prediction can still 'beat' the nRMSE 1.0 anchor):")
    for nm in ("self", "noise", "smooth", "shrunk", "flat"):
        print(f"    ref_{nm:<7s} nrmse {refs[f'ref_{nm}_fastts_nrmse']:.4f}   "
              f"CS {refs[f'ref_{nm}_fastts_ssim_cs']:.4f}   "
              f"contrast {refs[f'ref_{nm}_fastts_ssim_contrast']:.4f}   "
              f"std_ratio {refs[f'ref_{nm}_fastts_std_ratio']:.4f}")
    return {"name": name, **m, **b, "util": u, "cfg": cfg}


def make_figure(entries, X, out_path, n_windows=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B, C, W = X.shape
    # Window picks that span the activity range: quiet (p10), typical (p50), active (p95).
    # ONLY windows whose 8 channels are ALL live are eligible — a window with a constant
    # (absent) channel has an undefined nRMSE there and would put an un-scoreable flat line in
    # the figure, which is exactly the kind of panel that makes a figure unreadable.
    live = (X.std(axis=-1) > 1e-8).all(axis=1)
    elig = np.flatnonzero(live)
    if elig.size < 3:
        elig = np.arange(B)
    act = X[elig].std(axis=(1, 2))
    order = elig[np.argsort(act)]
    picks = [int(order[int(q * (len(order) - 1))]) for q in (0.10, 0.50, 0.95)][:n_windows]
    labels = ["quiet (10th pct activity)", "typical (median)", "active (95th pct)"][:n_windows]
    t_ms = np.arange(W) / FASTTS_FS * 1e3

    n_arms = len(entries)
    fig, axes = plt.subplots(C, n_windows * n_arms, figsize=(5.4 * n_windows * n_arms, 1.7 * C),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ai, (name, R, m) in enumerate(entries):
        for wi, (w, lab) in enumerate(zip(picks, labels)):
            col = ai * n_windows + wi
            for c in range(C):
                ax = axes[c][col]
                ax.set_facecolor(SURFACE)
                gt, rc = X[w, c], R[w, c]
                ax.plot(t_ms, gt, color=GT_COLOR, lw=2.0, solid_capstyle="round",
                        label="ground truth" if (c == 0 and col == 0) else None, zorder=1)
                ax.plot(t_ms, rc, color=RECON_COLOR, lw=1.0, solid_capstyle="round",
                        label="reconstruction" if (c == 0 and col == 0) else None, zorder=2)
                sd = gt.std()
                nr = np.sqrt(((rc - gt) ** 2).mean()) / sd if sd > 1e-12 else np.nan
                a, bb = gt - gt.mean(), rc - rc.mean()
                den = np.sqrt((a * a).sum() * (bb * bb).sum())
                cr = float((a * bb).sum() / den) if den > 0 else 0.0
                ax.text(0.995, 1.02, f"ch{c}   nRMSE {nr:.3f}   r {cr:+.2f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7.5, color=MUTED)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    ax.spines[side].set_color("#d8d8d4")
                ax.tick_params(labelsize=7, colors=MUTED, length=2)
                ax.grid(True, lw=0.4, color="#ececea", zorder=0)
                if c == 0:
                    ax.set_title(f"{name}  —  {lab}", fontsize=10, color=INK, pad=18)
                if c == C - 1:
                    ax.set_xlabel("time within the 50 ms frame (ms)", fontsize=8, color=MUTED)
                if col == 0:
                    ax.set_ylabel("std. units", fontsize=7, color=MUTED)
    handles, labs = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.005))
    sub = "  |  ".join(
        f"{n}: nRMSE {m['fastts_nrmse']:.4f}, r {m['fastts_corr']:+.3f}, "
        f"SSIM-contrast {m['fastts_ssim_contrast']:.3f}, std_ratio {m['fastts_std_ratio']:.3f}, "
        f"{m['cfg'].n_tok} tok" for n, _, m in entries)
    fig.suptitle("Raw 10 kHz filterscope trace: ground truth vs codec reconstruction "
                 f"(full 500-sample window, all 8 channels)\n{sub}"
                 "\nbaseline: the window's own constant mean = nRMSE 1.0000 / std_ratio 0.000"
                 "   —   LOOK AT AMPLITUDE, not just shape: std_ratio << 1 means the recon is"
                 " a shrunken blur even when nRMSE looks good",
                 fontsize=11, color=INK, y=1.035)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpts", nargs="+", help="codec_best.pt / codec_last.pt paths")
    ap.add_argument("--out", default="figures/fastts_raw_recon.png")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--n_fit_shots", type=int, default=8,
                    help="skip this many shots first (the floor probe's FIT shots)")
    ap.add_argument("--n_score_shots", type=int, default=4)
    args = ap.parse_args()

    X, ids = load_windows(args.n_fit_shots, args.n_score_shots)
    print(f"held-out score shots {ids}: {X.shape[0]} windows x {X.shape[1]} ch x {X.shape[2]} samples")
    entries = []
    for p in args.ckpts:
        name = Path(p).parent.name
        R, codes, cfg, ck = run_codec(p, X)
        m = report(name, R, X, codes, cfg, ck)
        entries.append((name, R, m))
    make_figure(entries, X, args.out, n_windows=args.windows)


if __name__ == "__main__":
    main()
