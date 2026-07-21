"""Gate 0 — fast overfit prediction test: deterministic (MAE) vs generative (flow).

Question (minutes, one isolated component): forecasting the NEXT window's
spectrogram from the current one, does a GENERATIVE flow-matching head produce a
coherent mode where a DETERMINISTIC MAE head mean-collapses? Run in
BASELINE-SUBTRACTED (residual) space, on 200729's mode channel, overfitting the
shot. Same small U-Net capacity for both heads (fair). No production backbone —
this isolates the LOSS, not the architecture.

PASS (pre-declared): the flow SAMPLE shows the coherent mode band (sharper /
higher mode-profile peakiness than the MAE prediction). FAIL: flow also blurs ->
the generative direction is dead for ~minutes of cost.
"""
import os
import sys
from pathlib import Path

FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training"):
    if p not in sys.path:
        sys.path.insert(0, p)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from poc_fsq_stageB import load_pairs, _hard
from spectro_bg import baseline_residual

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOD = os.environ.get("MODALITY", "ece"); SHOT = os.environ.get("SHOT", "200729")
DATA = os.environ.get("DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
STATS = os.environ.get("STATS_PATH", "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
NCH = int(os.environ.get("N_CHANNELS", "40")); NWIN = int(os.environ.get("NWIN", "250"))
STEPS = int(os.environ.get("STEPS", "3000")); BG_SIGMA = float(os.environ.get("BG_SIGMA", "8.0"))
FLOW_STEPS = int(os.environ.get("FLOW_STEPS", "12")); MODE_K = float(os.environ.get("MODE_K", "2.5"))
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/eval_runs/gate0_pred")); OUT.mkdir(parents=True, exist_ok=True)
FS, NFFT, HOP = 500_000.0, 1024, 256
torch.manual_seed(0)

# ---- data: 200729, baseline-subtracted residual, mode channel, forecast pairs (Ri -> Rt) ----
xi, xt = load_pairs(SHOT, DATA, STATS, NCH, NWIN, modality=MOD)     # (N,C,F,T) current, next
ch = int(_hard(xi, MODE_K).sum(dim=(0, 2, 3)).argmax())            # strongest-mode channel
_, Ri = baseline_residual(xi); _, Rt = baseline_residual(xt)       # residual space
Ri = Ri[:, ch:ch + 1].float().to(dev); Rt = Rt[:, ch:ch + 1].float().to(dev)   # (N,1,F,T)
N, _, Fq, Tq = Ri.shape
print(f"[gate0] {MOD} {SHOT} ch{ch}: N={N} pairs, residual space, F={Fq} T={Tq}", flush=True)


def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, in_ch, w=48):
        super().__init__()
        self.e0, self.e1, self.e2 = blk(in_ch, w), blk(w, 2 * w), blk(2 * w, 4 * w)
        self.d1, self.d0 = blk(4 * w + 2 * w, 2 * w), blk(2 * w + w, w)
        self.out = nn.Conv2d(w, 1, 1)
        self.pool = nn.MaxPool2d(2); self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        s0 = self.e0(x); s1 = self.e1(self.pool(s0)); b = self.e2(self.pool(s1))
        d1 = self.d1(torch.cat([self.up(b), s1], 1))
        d0 = self.d0(torch.cat([self.up(d1), s0], 1))
        return self.out(d0)


def batch(bs=8):
    idx = torch.randint(0, N, (bs,))
    return Ri[idx], Rt[idx]

# ---- deterministic head (MAE): predict next from current ----
det = UNet(1).to(dev); od = torch.optim.Adam(det.parameters(), 2e-4)
for s in range(STEPS):
    ci, ti = batch()
    loss = (det(ci) - ti).abs().mean()
    od.zero_grad(); loss.backward(); od.step()
    if (s + 1) % 1000 == 0:
        print(f"[gate0] det step {s+1} mae={loss.item():.4f}", flush=True)

# ---- generative head (flow matching): sample next from current ----
flw = UNet(3).to(dev); of = torch.optim.Adam(flw.parameters(), 2e-4)
for s in range(STEPS):
    ci, ti = batch(); B = ci.shape[0]
    x0 = torch.randn_like(ti); t = torch.rand(B, 1, 1, 1, device=dev)
    xt_ = (1 - t) * x0 + t * ti; vtar = ti - x0
    tb = t.expand(-1, 1, Fq, Tq)
    v = flw(torch.cat([xt_, ci, tb], 1))
    loss = ((v - vtar) ** 2).mean()
    of.zero_grad(); loss.backward(); of.step()
    if (s + 1) % 1000 == 0:
        print(f"[gate0] flow step {s+1} fm={loss.item():.4f}", flush=True)


@torch.no_grad()
def flow_sample(ci):
    x = torch.randn(ci.shape[0], 1, Fq, Tq, device=dev)
    for k in range(FLOW_STEPS):
        t = torch.full((ci.shape[0], 1, 1, 1), k / FLOW_STEPS, device=dev)
        x = x + (1.0 / FLOW_STEPS) * flw(torch.cat([x, ci, t.expand(-1, 1, Fq, Tq)], 1))
    return x

# ---- evaluate on the mode-richest windows ----
with torch.no_grad():
    dpred = torch.cat([det(Ri[i:i + 16]) for i in range(0, N, 16)], 0)
    fsamp = torch.cat([flow_sample(Ri[i:i + 16]) for i in range(0, N, 16)], 0)
G, Dp, Fs = Rt.cpu().numpy(), dpred.cpu().numpy(), fsamp.cpu().numpy()
fmax = int(60 / (FS / NFFT / 1e3))


def peakiness(a):        # time-avg |profile| peak-to-median: sharp mode -> high, blur -> ~1
    p = np.abs(a[:, 0, :fmax]).mean(2)                # (N,Fbins)
    return float(np.median(p.max(1) / (np.median(p, 1) + 1e-6)))


def modecorr(a):         # corr(pred, GT) in residual/mode band, median over windows
    cs = [np.corrcoef(G[w, 0, :fmax].ravel(), a[w, 0, :fmax].ravel())[0, 1] for w in range(N)]
    return float(np.nanmedian(cs))


print(f"\n[gate0] === RESULT (residual/mode band 0-60kHz, N={N}) ===", flush=True)
print(f"[gate0] {'head':<12}{'mode_corr_vs_GT':>16}{'peakiness':>12}", flush=True)
print(f"[gate0] {'GT':<12}{1.000:>16.3f}{peakiness(G):>12.2f}", flush=True)
print(f"[gate0] {'MAE(det)':<12}{modecorr(Dp):>16.3f}{peakiness(Dp):>12.2f}", flush=True)
print(f"[gate0] {'flow(samp)':<12}{modecorr(Fs):>16.3f}{peakiness(Fs):>12.2f}", flush=True)

# ---- figure: top-mode windows, GT | MAE | flow-sample (residual, 0-60kHz) ----
order = np.argsort(-np.abs(G[:, 0, :fmax]).sum((1, 2)))[:4]
FREQ = np.arange(Fq) * FS / NFFT / 1e3
fig, ax = plt.subplots(3, len(order), figsize=(3.4 * len(order), 8))
for j, w in enumerate(order):
    vmn, vmx = np.percentile(G[w, 0, :fmax], [2, 98])
    for r, (t, d) in enumerate([("GT next", G), ("MAE pred", Dp), ("flow sample", Fs)]):
        a_ = ax[r, j]
        a_.imshow(d[w, 0, :fmax], origin="lower", aspect="auto", cmap="magma", vmin=vmn, vmax=vmx,
                  extent=(0, Tq * HOP / FS * 1e3, 0, FREQ[fmax - 1]))
        a_.set_title(f"{t} w{w}", fontsize=9)
        if j == 0:
            a_.set_ylabel("Freq (kHz)")
        if r == 2:
            a_.set_xlabel("Time (ms)")
fig.suptitle(f"Gate 0 — {MOD} {SHOT} ch{ch} residual forecast: MAE vs flow "
             f"(peakiness GT {peakiness(G):.1f} / MAE {peakiness(Dp):.1f} / flow {peakiness(Fs):.1f})", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
for e in ("png", "pdf"):
    fig.savefig(OUT / f"gate0_{MOD}_{SHOT}.{e}", dpi=130, bbox_inches="tight")
print(f"[gate0] saved {OUT}/gate0_{MOD}_{SHOT}.png", flush=True)
