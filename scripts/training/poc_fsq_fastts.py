"""POC: FSQ (VQ-style) codec for FAST time-series (filterscopes) — 1D analog of the
spectro/video FSQ codecs. EXPLORATORY: can fast-TS be vector-quantized well,
including its transient spikes? (Historically spikes are the hard part —
see feedback-spike-reconstruction-loss.) NOT wired to production.

FastTimeSeriesTokenizer(Conv1d patch, 50 -> 80 tokens for 8ch/500-sample window)
-> FSQBottleneck -> FastTimeSeriesHead(ConvTranspose1d), trained with the validated
adversarial recipe (1D PatchGAN + hinge + FM + R1). Reconstruction only.

Env: EVAL_SHOTS(comma) FSQ_DIM(24) FSQ_L(8) AE_STEPS(4000) N_WINDOWS(120) AE_BS(32)
  ADV_LAMBDA(0.5) FM_LAMBDA(10) R1_GAMMA(10) D_LR(1e-4) RECON_WEIGHT(1) VAL_FRAC(0.15) OUT_DIR
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.e2e.tokenizers.fast_time_series import FastTimeSeriesTokenizer
from tokamak_foundation_model.e2e.output_heads import FastTimeSeriesHead
from tokamak_foundation_model.e2e.quantizers import FSQBottleneck

D_MODEL = 256
C, WIN, PATCH = 8, 500, 50           # filterscopes: 8 ch, 0.05s @ 10 kHz, patch 50


class FastTSFSQAutoencoder(nn.Module):
    """FastTimeSeriesTokenizer -> FSQ bottleneck -> FastTimeSeriesHead."""

    def __init__(self, fsq_dim, fsq_L, d_model=D_MODEL):
        super().__init__()
        self.enc = FastTimeSeriesTokenizer(n_channels=C, window_samples=WIN,
                                           d_model=d_model, patch_size=PATCH)
        self.n_tok = C * (WIN // PATCH)          # 80
        self.fsq = FSQBottleneck(d_model, [fsq_L] * fsq_dim)
        self.dec = FastTimeSeriesHead(d_model=d_model, n_channels=C,
                                      window_samples=WIN, patch_size=PATCH)
        self.dim, self.levels = fsq_dim, fsq_L

    def forward(self, x):                        # x (B, C, WIN)
        tq, codes = self.fsq(self.enc(x))
        return self.dec(tq), codes               # (B, C, WIN), (B, n_tok, dim)


class FastTSDiscriminator1D(nn.Module):
    """1D PatchGAN over (B, C, WIN). Returns (patch_logits, [features])."""

    def __init__(self, base=32):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 15, 4, 7),
                                 nn.GroupNorm(min(8, o), o), nn.LeakyReLU(0.2, inplace=True))
        self.b1 = blk(C, base); self.b2 = blk(base, base * 2); self.b3 = blk(base * 2, base * 4)
        self.out = nn.Conv1d(base * 4, 1, 3, 1, 1)

    def forward(self, x):
        f1 = self.b1(x); f2 = self.b2(f1); f3 = self.b3(f2)
        return self.out(f3), [f1, f2, f3]


def load_fastts_windows(shot, data_dir, stats_path, n_windows):
    """(N, C, WIN) filterscopes windows, per-(window,channel) z-scored."""
    stats = torch.load(stats_path, weights_only=False)
    ds = TokamakMultiFileDataset(
        hdf5_paths=[Path(data_dir) / f"{shot}_processed.h5"], chunk_duration_s=0.05,
        prediction_mode=True, prediction_horizon_s=0.05, step_size_s=0.01, warmup_s=1.0,
        preprocessing_stats=stats, input_signals=["filterscopes"], target_signals=["filterscopes"])
    n = len(ds)
    if n == 0:
        return torch.empty(0)
    idxs = range(n) if n_windows <= 0 else range(0, n, max(1, n // n_windows))
    out = []
    for i in idxs:
        v = ds[i]["inputs"].get("filterscopes")
        if v is None:
            continue
        v = torch.nan_to_num(torch.as_tensor(v).float())      # (C, WIN)
        mu = v.mean(dim=1, keepdim=True); sd = v.std(dim=1, keepdim=True).clamp(min=1e-3)
        out.append((v - mu) / sd)
    return torch.stack(out) if out else torch.empty(0)


def plot_full_shot(shot):
    """Reconstruct an ENTIRE shot with saved frozen codec(s) and plot the full
    continuous time trace (GT vs recon). Tiles the shot into consecutive
    NON-overlapping WIN-sample windows, encode->decode each (per-window z-score,
    exactly as trained), denorm per-window, and stitch back in time order.
    Env: PLOT_SHOT=<shot[,shot...]> LOAD_CODECS=<p1.pt,p2.pt,...> OUT_DIR EVAL_DATA_DIR EVAL_STATS.
    """
    global PATCH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.environ.get("EVAL_DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
    stats_path = os.environ.get("EVAL_STATS", "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    codec_specs = [c.strip() for c in os.environ.get(
        "LOAD_CODECS",
        "eval_runs/fsq_fastts_p50/fastts_codec.pt,eval_runs/fsq_fastts_p25/fastts_codec.pt").split(",") if c.strip()]
    out_dir = Path(os.environ.get("OUT_DIR", "eval_runs/fsq_fastts_fullshot")); out_dir.mkdir(parents=True, exist_ok=True)
    fs = 10000.0  # filterscopes: WIN=500 samples over 0.05 s -> 10 kHz
    t0 = 1.0      # warmup_s skipped at shot start

    stats = torch.load(stats_path, weights_only=False)
    ds = TokamakMultiFileDataset(
        hdf5_paths=[Path(data_dir) / f"{shot}_processed.h5"], chunk_duration_s=0.05,
        prediction_mode=True, prediction_horizon_s=0.05, step_size_s=0.05, warmup_s=t0,
        preprocessing_stats=stats, input_signals=["filterscopes"], target_signals=["filterscopes"])
    raw = []
    for i in range(len(ds)):
        v = ds[i]["inputs"].get("filterscopes")
        if v is None:
            continue
        raw.append(torch.nan_to_num(torch.as_tensor(v).float()))     # (C, WIN) standardized
    if not raw:
        print(f"[fts] shot {shot}: NO filterscope windows -> abort", flush=True); return
    Wn = len(raw)
    GTw = torch.stack(raw)                                            # (Wn, C, WIN)
    mu = GTw.mean(dim=2, keepdim=True); sd = GTw.std(dim=2, keepdim=True).clamp(min=1e-3)
    Xn = (GTw - mu) / sd                                             # codec input space
    print(f"[fts] shot {shot}: {Wn} consecutive windows -> {Wn*WIN} samples "
          f"({Wn*WIN/fs:.2f} s from t={t0}s)", flush=True)

    recons = {}
    for spec in codec_specs:
        ck = torch.load(spec, map_location="cpu", weights_only=False); cfg = ck["cfg"]
        PATCH = cfg["patch"]
        ae = FastTSFSQAutoencoder(cfg["fsq_dim"], cfg["fsq_L"], d_model=cfg.get("d_model", D_MODEL))
        ae.load_state_dict(ck["ae"]); ae.eval().to(device)
        for p in ae.parameters():
            p.requires_grad_(False)
        outs = []
        with torch.no_grad():
            for i in range(0, Wn, 64):
                r, _ = ae(Xn[i:i + 64].to(device)); outs.append(r.cpu())
        Rp = torch.cat(outs, 0) * sd + mu                            # denorm -> standardized units
        recons[f"{ae.n_tok} tok"] = Rp
        print(f"[fts]   {spec} -> {ae.n_tok} tokens (patch {cfg['patch']})", flush=True)

    G = GTw.permute(1, 0, 2).reshape(C, -1).numpy()                  # (C, Wn*WIN)
    Rs = {k: v.permute(1, 0, 2).reshape(C, -1).numpy() for k, v in recons.items()}
    T = G.shape[1]; t = t0 + np.arange(T) / fs
    # channel = the ELM channel by p99-p50 (validated selection metric), NOT max-z>3
    # count (that can be an oscillatory channel).
    elev = np.percentile(G, 99, axis=1) - np.median(G, axis=1)
    chsel = int(np.argmax(elev))
    xg = G[chsel]
    med = np.median(xg); mad = np.median(np.abs(xg - med)) * 1.4826 + 1e-6
    zc = (xg - med) / mad                                            # ROBUST z
    nspk = int((zc > 3).sum())
    print(f"[fts] ELM channel = ch{chsel} (p99-p50={elev[chsel]:.2f}, {nspk} spike samples)", flush=True)
    cols = ["tab:orange", "tab:green", "tab:red"]
    ylo, yhi = np.percentile(xg, [0.5, 99.5]); ypad = 0.25 * (yhi - ylo + 1e-6)  # clip disruption

    # zoom on densest SUSTAINED ELM activity (moderate excursions, not the lone disruption)
    zwin = int(0.5 * fs)
    band = (zc > 2.5).astype(float)   # no upper cap: strongest ELMs must count (else zoom lands on flat noise)
    z_lo = int(np.convolve(band, np.ones(zwin), "valid").argmax()) if T > zwin else 0
    z_hi = min(T, z_lo + zwin)

    fig, ax = plt.subplots(3, 1, figsize=(16, 9))
    ax[0].plot(t, xg, lw=0.5, color="black", label="GT")
    for (lbl, R), c in zip(Rs.items(), cols):
        ax[0].plot(t, R[chsel], lw=0.5, alpha=0.75, color=c, label=f"recon {lbl}")
    ax[0].axvspan(t[z_lo], t[z_hi - 1], color="gold", alpha=0.15)
    ax[0].set_ylim(ylo - ypad, yhi + ypad)                          # robust scale -> ELM band visible
    ax[0].set_title(f"shot {shot} ch{chsel} (p99-p50={elev[chsel]:.2f}): FULL SHOT, robust y-scale "
                    f"({nspk} spike samples)"); ax[0].legend(fontsize=8, ncol=len(Rs) + 1)
    ax[0].set_xlabel("time (s)")
    gz = xg[z_lo:z_hi]
    ax[1].plot(t[z_lo:z_hi], gz, lw=0.9, color="black", label="GT")
    for (lbl, R), c in zip(Rs.items(), cols):
        ax[1].plot(t[z_lo:z_hi], R[chsel, z_lo:z_hi], lw=0.9, alpha=0.8, color=c, label=f"recon {lbl}")
    zylo, zyhi = np.percentile(gz, [0.5, 99.5]); zpad = 0.25 * (zyhi - zylo + 1e-6)
    ax[1].set_ylim(zylo - zpad, zyhi + zpad)
    ax[1].set_title(f"ZOOM on densest ELM burst ({(z_hi-z_lo)/fs:.2f} s)"); ax[1].legend(fontsize=8)
    ax[1].set_xlabel("time (s)")
    best = list(Rs.items())[-1]
    ax[2].plot(t, xg - best[1][chsel], lw=0.4, color="crimson")
    ax[2].set_ylim(-(yhi - ylo + 1e-6), (yhi - ylo + 1e-6))
    ax[2].set_title(f"residual (GT - recon {best[0]})"); ax[2].set_xlabel("time (s)")
    fig.tight_layout()
    p = out_dir / f"fullshot_{shot}_ch{chsel}.png"; fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[fts] FULL-SHOT FIGURE -> {p}", flush=True)

    # all-channel small multiples (GT vs best recon)
    fig, axes = plt.subplots(C, 1, figsize=(16, 1.6 * C), sharex=True)
    for c in range(C):
        axes[c].plot(t, G[c], lw=0.4, color="black")
        axes[c].plot(t, best[1][c], lw=0.4, alpha=0.75, color="tab:green")
        clo, chi = np.percentile(G[c], [0.5, 99.5]); cpad = 0.25 * (chi - clo + 1e-6)
        axes[c].set_ylim(clo - cpad, chi + cpad)            # robust per-channel scale
        axes[c].set_ylabel(f"ch{c}", fontsize=8)
    axes[0].set_title(f"shot {shot} — all filterscope channels: GT (black) vs recon {best[0]} (green)")
    axes[-1].set_xlabel("time (s)"); fig.tight_layout()
    p2 = out_dir / f"fullshot_{shot}_allch.png"; fig.savefig(p2, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[fts] ALL-CHANNEL FIGURE -> {p2}\n=== FSQ FAST-TS FULL-SHOT DONE ===", flush=True)


def load_shots_windows(shots, data_dir, stats_path, elm_zthr=4.0,
                       max_quiet_per_shot=60, step_s=0.05):
    """Load per-(window,channel) z-scored filterscope windows from MANY shots for
    final-codec training. Tags each window ELM-active (max|z| on any channel >
    elm_zthr = a sharp excursion, i.e. a crash) vs quiet, keeps ALL ELM windows +
    up to max_quiet_per_shot quiet windows/shot (bounds memory AND lifts the ELM
    fraction). Returns (X (N,C,WIN) normalized, elm_mask (N,) bool)."""
    stats = torch.load(stats_path, weights_only=False)
    norm, flags = [], []
    kept_elm = kept_quiet = 0
    for si, sh in enumerate(shots):
        try:
            ds = TokamakMultiFileDataset(
                hdf5_paths=[Path(data_dir) / f"{sh}_processed.h5"], chunk_duration_s=0.05,
                prediction_mode=True, prediction_horizon_s=0.05, step_size_s=step_s,
                warmup_s=1.0, preprocessing_stats=stats,
                input_signals=["filterscopes"], target_signals=["filterscopes"])
        except Exception as e:
            print(f"[fts] shot {sh} SKIP: {e}", flush=True); continue
        vs = []
        for i in range(len(ds)):
            v = ds[i]["inputs"].get("filterscopes")
            if v is not None:
                vs.append(torch.nan_to_num(torch.as_tensor(v).float()))
        if not vs:
            continue
        V = torch.stack(vs)                                   # (W, C, WIN)
        mu = V.mean(2, keepdim=True); sd = V.std(2, keepdim=True).clamp(min=1e-3)
        Vn = (V - mu) / sd
        elm = Vn.abs().amax(dim=(1, 2)) > elm_zthr            # (W,) sharp excursion
        eidx = torch.nonzero(elm, as_tuple=False).squeeze(1)
        qidx = torch.nonzero(~elm, as_tuple=False).squeeze(1)
        if max_quiet_per_shot > 0 and qidx.numel() > max_quiet_per_shot:
            g = torch.Generator().manual_seed(1234 + si)
            qidx = qidx[torch.randperm(qidx.numel(), generator=g)[:max_quiet_per_shot]]
        keep = torch.cat([eidx, qidx])
        if keep.numel() == 0:
            continue
        norm.append(Vn[keep]); flags.append(elm[keep])
        kept_elm += int(eidx.numel()); kept_quiet += int(qidx.numel())
        if (si + 1) % 100 == 0:
            print(f"[fts] loaded {si+1}/{len(shots)} shots  kept elm={kept_elm} quiet={kept_quiet}", flush=True)
    if not norm:
        return torch.empty(0), torch.empty(0, dtype=torch.bool)
    X = torch.cat(norm, 0); E = torch.cat(flags, 0)
    print(f"[fts] TOTAL windows={X.shape[0]}  ELM={int(E.sum())} "
          f"({100*float(E.float().mean()):.1f}%)  quiet={int((~E).sum())}", flush=True)
    return X, E


def plot_shots_grid(shots):
    """GT-only overview: for each shot, plot the FULL filterscope trace on the
    channel the ELM scan scored (best_ch from RANK_FILE, else the max-spike
    channel), with z>3 samples marked. One row per shot -> a stacked overview to
    visually validate the ELM-activity ranking BEFORE committing to training.
    Env: GRID_SHOTS=<shot,...> RANK_FILE OUT_DIR EVAL_DATA_DIR EVAL_STATS."""
    data_dir = os.environ.get("EVAL_DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
    stats_path = os.environ.get("EVAL_STATS", "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    out_dir = Path(os.environ.get("OUT_DIR", "eval_runs/fastts_elm_scan")); out_dir.mkdir(parents=True, exist_ok=True)
    fs, t0 = 10000.0, 1.0
    meta = {}
    rf = os.environ.get("RANK_FILE", "")
    if rf and os.path.exists(rf):
        for ln in open(rf):
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.split(); meta[p[0]] = (float(p[1]), int(p[2]))
    stats = torch.load(stats_path, weights_only=False)
    traces = []
    for sh in shots:
        try:
            ds = TokamakMultiFileDataset(
                hdf5_paths=[Path(data_dir) / f"{sh}_processed.h5"], chunk_duration_s=0.05,
                prediction_mode=True, prediction_horizon_s=0.05, step_size_s=0.05,
                warmup_s=t0, preprocessing_stats=stats,
                input_signals=["filterscopes"], target_signals=["filterscopes"])
        except Exception as e:
            print(f"[fts] grid shot {sh} SKIP: {e}", flush=True); continue
        vs = [torch.nan_to_num(torch.as_tensor(ds[i]["inputs"]["filterscopes"]).float())
              for i in range(len(ds)) if ds[i]["inputs"].get("filterscopes") is not None]
        if not vs:
            continue
        G = torch.stack(vs).permute(1, 0, 2).reshape(vs[0].shape[0], -1).numpy()  # (C,T)
        # per-channel diagnostics to design the ELM gate: kurtosis (heavy tail =
        # isolated bursts) vs active-fraction (duty cycle; oscillation ~ high).
        try:
            from scipy.stats import kurtosis as _kurt
            for c in range(G.shape[0]):
                xc = G[c]
                q = np.percentile(xc, [50, 99, 99.9])
                # p99-p50 = ABSOLUTE elevation of the top ~1% (ELM train, not flat, not single-spike)
                print(f"[diag] {sh} ch{c}: p99-p50={float(q[1]-q[0]):6.3f} "
                      f"p999-p50={float(q[2]-q[0]):6.3f} std={float(xc.std()):6.3f} "
                      f"kurt={float(_kurt(xc)):8.0f} max={float(xc.max()):7.1f}", flush=True)
        except Exception as e:
            print(f"[diag] {sh} stats err: {e}", flush=True)
        rate, ch = meta.get(str(sh), (None, None))
        if ch is None or ch < 0:
            z = (G - G.mean(1, keepdims=True)) / (G.std(1, keepdims=True) + 1e-6)
            ch = int((z > 3).sum(1).argmax())
        xch = G[ch]
        ps = np.percentile(xch, [50, 90, 99, 99.9, 100])
        # if max >> p99.9, ONE dominant spike flattens the plot (ELMs hidden);
        # a real ELM train shows an elevated BAND (p90..p99.9 spread above p50).
        print(f"[fts] STATS {sh} ch{ch}: p50={ps[0]:.2f} p90={ps[1]:.2f} p99={ps[2]:.2f} "
              f"p99.9={ps[3]:.2f} max={ps[4]:.2f} min={xch.min():.2f} std={xch.std():.2f}", flush=True)
        traces.append((sh, ch, rate, xch))
    if not traces:
        print("[fts] grid: no traces", flush=True); return
    N = len(traces)
    zoom_ms = float(os.environ.get("ZOOM_MS", "0"))    # >0 adds a tight full-res zoom column
    zw = int(zoom_ms * 1e-3 * fs)
    ncol = 2 if zoom_ms > 0 else 1
    fig, axes = plt.subplots(N, ncol, figsize=(16, 1.35 * N), squeeze=False)
    for i, (sh, ch, rate, x) in enumerate(traces):
        t = t0 + np.arange(len(x)) / fs
        med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
        z = (x - med) / mad                          # ROBUST z (MAD-based, immune to a lone spike)
        sp = np.nonzero(z > 3)[0]
        a = axes[i, 0]
        a.plot(t, x, lw=0.35, color="black")
        if sp.size:
            a.plot(t[sp], x[sp], ".", ms=1.3, color="red")
        ylo, yhi = np.percentile(x, [0.3, 99.7])     # robust y-lims: a lone disruption spike can't flatten the ELM band
        if yhi > ylo:
            a.set_ylim(ylo - 0.2 * (yhi - ylo), yhi + 0.2 * (yhi - ylo))
        rr = f"a={rate:.2f}" if rate is not None else "?"
        a.set_ylabel(f"{sh}\nch{ch} {rr}", fontsize=7, rotation=0, ha="right", va="center")
        a.set_yticks([]); a.margins(x=0.005)
        if zoom_ms > 0 and len(x) > zw:
            band = ((z > 2.5) & (z < 15)).astype(float)   # center on SUSTAINED moderate activity, not the lone spike
            dens = np.convolve(band, np.ones(zw), "valid")
            lo = int(dens.argmax()); hi = lo + zw
            az = axes[i, 1]
            az.plot(t[lo:hi], x[lo:hi], lw=0.7, color="black", marker=".", ms=2.0)
            spz = sp[(sp >= lo) & (sp < hi)]
            if spz.size:
                az.plot(t[spz], x[spz], ".", ms=4, color="red")
            zlo, zhi = np.percentile(x[lo:hi], [0.5, 99.5])
            if zhi > zlo:
                az.set_ylim(zlo - 0.2 * (zhi - zlo), zhi + 0.2 * (zhi - zlo))
            az.set_yticks([]); az.margins(x=0.01)
    axes[-1, 0].set_xlabel("time (s)")
    axes[0, 0].set_title("full trace (scan channel); red = z>3 samples", fontsize=10)
    if ncol == 2:
        axes[-1, 1].set_xlabel("time (s)")
        axes[0, 1].set_title(f"zoom {zoom_ms:.0f} ms on densest region (dots = samples)", fontsize=10)
    fig.tight_layout()
    name = f"top_shots_grid{'_zoom' if zoom_ms>0 else ''}.png"
    p = out_dir / name; fig.savefig(p, dpi=125, bbox_inches="tight"); plt.close(fig)
    print(f"[fts] GRID FIGURE -> {p}\n=== FSQ FAST-TS GRID DONE ===", flush=True)


def main():
    if os.environ.get("GRID_SHOTS"):
        shots = [s.strip() for s in os.environ["GRID_SHOTS"].split(",") if s.strip()]
        plot_shots_grid(shots); return
    if os.environ.get("PLOT_SHOT"):
        for sh in os.environ["PLOT_SHOT"].split(","):
            sh = sh.strip()
            if sh:
                plot_full_shot(sh)
        print("=== FSQ FAST-TS FULL-SHOT (ALL) DONE ===", flush=True); return
    data_dir = os.environ.get("EVAL_DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
    stats_path = os.environ.get("EVAL_STATS", "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    shots = [s.strip() for s in os.environ.get("EVAL_SHOTS", "200729,200226,200722,201664,201797").split(",") if s.strip()]
    fsq_dim = int(os.environ.get("FSQ_DIM", "24")); fsq_L = int(os.environ.get("FSQ_L", "8"))
    ae_steps = int(os.environ.get("AE_STEPS", "4000")); n_windows = int(os.environ.get("N_WINDOWS", "120"))
    ae_bs = int(os.environ.get("AE_BS", "32")); recon_w = float(os.environ.get("RECON_WEIGHT", "1"))
    adv_lambda = float(os.environ.get("ADV_LAMBDA", "0.5")); fm_lambda = float(os.environ.get("FM_LAMBDA", "10"))
    r1_gamma = float(os.environ.get("R1_GAMMA", "10")); d_lr = float(os.environ.get("D_LR", "1e-4"))
    val_frac = float(os.environ.get("VAL_FRAC", "0.15"))
    out_dir = Path(os.environ.get("OUT_DIR", "eval_runs/fsq_fastts")); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Finer patch = more tokens = better temporal resolution for the ELM SPIKES
    # (filterscopes are ELM detectors; the spikes are the signal). patch must
    # divide WIN=500: 50→80tok, 25→160, 10→400, 5→800.
    global PATCH
    PATCH = int(os.environ.get("PATCH_SIZE", str(PATCH)))

    shotfile = os.environ.get("EVAL_SHOTS_FILE", "").strip()
    elm_frac = float(os.environ.get("ELM_FRAC", "0.5"))
    elm_zthr = float(os.environ.get("ELM_ZTHR", "4.0"))
    max_quiet = int(os.environ.get("MAX_QUIET_PER_SHOT", "60"))
    Emask = None
    if shotfile:
        with open(shotfile) as fh:
            fshots = [ln.split()[0].strip() for ln in fh
                      if ln.strip() and not ln.lstrip().startswith("#")]
        print(f"[fts] SHOTFILE {shotfile}: {len(fshots)} shots; ELM-window oversampling "
              f"frac={elm_frac} zthr={elm_zthr} max_quiet/shot={max_quiet}", flush=True)
        X, Emask = load_shots_windows(fshots, data_dir, stats_path, elm_zthr, max_quiet)
        g = torch.Generator().manual_seed(0)                  # representative train/val split
        perm = torch.randperm(X.shape[0], generator=g); X = X[perm]; Emask = Emask[perm]
    else:
        Xs = []
        for sh in shots:
            try:
                v = load_fastts_windows(sh, data_dir, stats_path, n_windows)
            except Exception as e:
                print(f"[fts] shot {sh} SKIP: {e}", flush=True); continue
            if v.numel():
                Xs.append(v); print(f"[fts] shot {sh}: {v.shape[0]} windows", flush=True)
        X = torch.cat(Xs, 0)
    N = X.shape[0]; nv = max(1, int(N * val_frac)); ntr = N - nv
    print(f"[fts] N={N} C={X.shape[1]} WIN={X.shape[2]} train={ntr} heldout={nv}", flush=True)
    Xtr = X[:ntr]
    Etr = Emask[:ntr] if Emask is not None else None

    # DECODER-ONLY fine-tune: load an existing codec, FREEZE enc+fsq (codes stay
    # BYTE-IDENTICAL so the frozen world model's predicted codes remain valid), and
    # train ONLY the decoder. AE is rebuilt from the SAVED cfg (not env) so the
    # weights load exactly. PATCH/WIN are module globals FastTSFSQAutoencoder reads
    # at construction, so set PATCH (and WIN) from cfg FIRST.
    finetune_from = os.environ.get("FINETUNE_FROM", "").strip()
    if finetune_from:
        global WIN
        ck = torch.load(finetune_from, map_location=device, weights_only=False)
        fcfg = ck["cfg"]
        PATCH = int(fcfg["patch"]); WIN = int(fcfg["WIN"])
        fsq_dim, fsq_L = fcfg["fsq_dim"], fcfg["fsq_L"]
        ae = FastTSFSQAutoencoder(fsq_dim, fsq_L, d_model=fcfg.get("d_model", D_MODEL)).to(device)
        ae.load_state_dict(ck["ae"])
        for p in ae.enc.parameters():
            p.requires_grad_(False)
        for p in ae.fsq.parameters():
            p.requires_grad_(False)
        assert not any(p.requires_grad for p in ae.enc.parameters()), "enc must be frozen"
        assert not any(p.requires_grad for p in ae.fsq.parameters()), "fsq must be frozen"
        optG = torch.optim.Adam([p for p in ae.dec.parameters() if p.requires_grad],
                                2e-4, betas=(0.5, 0.9))
        n_frozen = sum(p.numel() for p in ae.enc.parameters()) + sum(p.numel() for p in ae.fsq.parameters())
        n_dec = sum(p.numel() for p in ae.dec.parameters() if p.requires_grad)
        ae_steps = int(os.environ.get("FT_STEPS", "2500"))
        print(f"[fts] DECODER-ONLY FINE-TUNE from {finetune_from}: "
              f"n_enc_fsq_frozen={n_frozen} n_dec_trainable={n_dec} FT_STEPS={ae_steps}", flush=True)
    else:
        ae = FastTSFSQAutoencoder(fsq_dim, fsq_L).to(device)
        optG = torch.optim.Adam(ae.parameters(), 2e-4, betas=(0.5, 0.9))
    disc = FastTSDiscriminator1D().to(device)
    optD = torch.optim.Adam(disc.parameters(), d_lr, betas=(0.5, 0.9))
    print(f"[fts] FSQ fast-TS AE: {ae.n_tok} tokens, fsq {fsq_dim}x{fsq_L}, adv{adv_lambda} "
          f"fm{fm_lambda} R1 g{r1_gamma} D-lr{d_lr}", flush=True)

    ntr_ = Xtr.shape[0]
    elm_pool = torch.nonzero(Etr, as_tuple=False).squeeze(1) if Etr is not None else None
    quiet_pool = torch.nonzero(~Etr, as_tuple=False).squeeze(1) if Etr is not None else None
    oversample = elm_pool is not None and elm_pool.numel() > 0 and quiet_pool.numel() > 0
    n_elm = int(round(ae_bs * elm_frac))
    if oversample:
        print(f"[fts] ELM oversampling: {n_elm}/{ae_bs} windows/batch from ELM pool "
              f"(elm={elm_pool.numel()} quiet={quiet_pool.numel()})", flush=True)

    def sample_idx():
        if oversample:
            ie = elm_pool[torch.randint(0, elm_pool.numel(), (n_elm,))]
            iq = quiet_pool[torch.randint(0, quiet_pool.numel(), (ae_bs - n_elm,))]
            return torch.cat([ie, iq])
        return torch.randint(0, ntr_, (ae_bs,))

    for s in range(ae_steps):
        idx = sample_idx(); x = Xtr[idx].to(device)
        with torch.no_grad():
            rec, _ = ae(x)
        xr = x.detach().requires_grad_(True)
        dr, _ = disc(xr); df, _ = disc(rec)
        dloss = F.relu(1 - dr).mean() + F.relu(1 + df).mean()
        if r1_gamma > 0:
            g = torch.autograd.grad(dr.sum(), xr, create_graph=True)[0]
            dloss = dloss + 0.5 * r1_gamma * g.pow(2).flatten(1).mean(1).mean()
        optD.zero_grad(set_to_none=True); dloss.backward(); optD.step()
        rec, _ = ae(x); mae = (rec - x).abs().mean()
        dfg, ff = disc(rec)
        with torch.no_grad():
            _, fr = disc(x)
        gadv = -dfg.mean(); fm = sum((a - b).abs().mean() for a, b in zip(ff, fr)) / len(ff)
        gloss = recon_w * mae + adv_lambda * gadv + fm_lambda * fm
        optG.zero_grad(set_to_none=True); gloss.backward(); optG.step()
        if (s + 1) % 500 == 0 or s == 0:
            print(f"  [fts] step {s+1}/{ae_steps} mae={mae.item():.4f} gadv={gadv.item():.3f} "
                  f"fm={fm.item():.3f} d={dloss.item():.3f}", flush=True)

    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    torch.save({"ae": ae.state_dict(), "cfg": dict(C=C, WIN=WIN, patch=PATCH,
                fsq_dim=fsq_dim, fsq_L=fsq_L, d_model=D_MODEL)},
               out_dir / "fastts_codec.pt")
    print(f"[fts] SAVED FROZEN CODEC -> {out_dir/'fastts_codec.pt'}", flush=True)

    # recon eval: per-channel corr + SPIKE-CAPTURE metrics (ELM spikes are the signal)
    Xv = X[ntr:].to(device)
    with torch.no_grad():
        REC = torch.cat([ae(Xv[i:i + 64])[0] for i in range(0, nv, 64)], 0)
    gt = Xv.cpu().numpy(); rc = REC.cpu().numpy()
    def corr(a, b):
        a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b); return float(a @ b / d) if d > 0 else 0.0
    pcc = [corr(gt[:, c], rc[:, c]) for c in range(C)]
    print(f"[fts] HELD-OUT per-channel corr: mean={np.mean(pcc):.3f} "
          f"min={np.min(pcc):.3f} max={np.max(pcc):.3f}  mae={np.abs(gt-rc).mean():.4f}", flush=True)

    # --- SPIKE CAPTURE (the actual figure of merit for ELM filterscopes) ---
    # Spike = sample where GT rises well above its own baseline (z>SPK_Z, positive).
    # We report, over ALL held-out spike samples: correlation on spike samples,
    # amplitude recall (mean recon / mean GT at spikes), and detection recall
    # (fraction of GT spikes where recon also exceeds the threshold).
    spk_z = float(os.environ.get("SPK_Z", "3.0"))
    g_all = gt.reshape(gt.shape[0] * C, WIN)          # (N*C, WIN) each row a window-channel
    r_all = rc.reshape(rc.shape[0] * C, WIN)
    mu = g_all.mean(1, keepdims=True); sd = g_all.std(1, keepdims=True) + 1e-6
    zg = (g_all - mu) / sd
    spike = zg > spk_z
    ns = int(spike.sum())
    if ns > 0:
        gs = g_all[spike]; rs = r_all[spike]
        spk_corr = corr(gs, rs)
        amp_recall = float(np.abs(rs).mean() / (np.abs(gs).mean() + 1e-9))
        # detection recall: recon also above the SAME per-row threshold at a GT spike
        thr = (mu + spk_z * sd)                        # (N*C,1)
        det = ((r_all > thr) & spike).sum() / max(1, ns)
        print(f"[fts] SPIKE CAPTURE (z>{spk_z}): {ns} spike-samples "
              f"({100*spike.mean():.2f}%)  spike_corr={spk_corr:.3f} "
              f"amp_recall={amp_recall:.2f} det_recall={float(det):.2f}", flush=True)
    else:
        print(f"[fts] SPIKE CAPTURE: no samples exceed z>{spk_z} in held-out set", flush=True)

    # trace overlay: the MOST SPIKE-ACTIVE windows (not the first quiet ones).
    # rank each held-out window by its peak spike count summed over channels,
    # then show the top few, each as its own zoomed 500-sample panel.
    zwin = (gt - gt.mean(axis=2, keepdims=True)) / (gt.std(axis=2, keepdims=True) + 1e-6)
    win_score = (zwin > spk_z).sum(axis=(1, 2))            # (nv,) total spike samples per window
    order = np.argsort(-win_score)
    nshow = min(4, nv)
    top = order[:nshow]
    # per top-window, the channel with the most spikes (so the panel actually shows ELMs)
    fig, ax = plt.subplots(nshow, 1, figsize=(13, 2.4 * nshow), squeeze=False)
    for i, w in enumerate(top):
        chsel = int((zwin[w] > spk_z).sum(axis=1).argmax())
        g = gt[w, chsel]; r = rc[w, chsel]
        a_ = ax[i, 0]
        a_.plot(g, lw=0.9, label="GT", color="black")
        a_.plot(r, lw=0.9, alpha=0.85, label="FSQ recon", color="tab:orange")
        thr = g.mean() + spk_z * (g.std() + 1e-6)
        a_.axhline(thr, color="tab:blue", ls=":", lw=0.7)
        a_.set_title(f"held-out window {int(w)}, ch{chsel}: {int(win_score[w])} spike-samples "
                     f"(corr {corr(g, r):.2f})", fontsize=9)
        if i == 0:
            a_.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"fast-TS FSQ ({ae.n_tok} tok, patch {PATCH}) — most ELM-active held-out windows",
                 fontsize=11)
    fig.tight_layout()
    p = out_dir / "fastts_recon.png"; fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[fts] RECON FIGURE -> {p}\n=== FSQ FAST-TS CODEC (POC) DONE ===", flush=True)


if __name__ == "__main__":
    main()
