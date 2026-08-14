"""Gate 0b — overfit forecast test on the REAL transformer representation.

Unlike gate0 (small U-Net on the raw window), this conditions the heads on the
FROZEN production backbone TOKENS (probe_fit setup): the actual representation
the world model's spectro head sees. Forecast = tokens(current window) ->
next-window residual spectrogram, 200729, baseline-subtracted. Deterministic MAE
head vs generative flow head, SAME capacity.

DIAGNOSTIC:
  flow >> MAE  -> tokens carry the mode; the LOSS was the problem (generative fix).
  both blur    -> the tokens don't carry the mode; the BACKBONE is the problem.
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
from torch.utils.data import DataLoader
from eval_e2e_animation_tokamak import load_model
from train_e2e_stage1 import build_datasets, forward_batch, _core
from tokamak_foundation_model.data.data_loader import collate_fn
from spectro_bg import baseline_residual
from poc_fsq_stageB import _hard

dev = torch.device("cuda")
CKPT = os.environ.get("CKPT", "/lustre/orion/fus187/proj-shared/models/e2e_stage1_allshots_b32/e2e_stage1_latest.pt")
MOD = os.environ.get("MODALITY", "ece"); SHOT = os.environ.get("SHOT", "200729")
STEPS = int(os.environ.get("STEPS", "3000")); BG_SIGMA = float(os.environ.get("BG_SIGMA", "8.0"))
FLOW_STEPS = int(os.environ.get("FLOW_STEPS", "12")); MODE_K = float(os.environ.get("MODE_K", "2.5"))
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/eval_runs/gate0b_token")); OUT.mkdir(parents=True, exist_ok=True)
FS, NFFT, HOP = 500_000.0, 1024, 256
torch.manual_seed(0)

# ---- frozen production model: cache backbone tokens (current) + target (next window) ----
model, ckpt = load_model(Path(CKPT), dev); model.eval(); core = _core(model)
a = ckpt["args"]; dn = [d["name"] for d in ckpt["diagnostics"]]; an = [c["name"] for c in ckpt["actuators"]]
dd = Path(a["data_dir"]); stats = torch.load(a["stats_path"], weights_only=False); sf = dd / f"{SHOT}_processed.h5"
_, ds = build_datasets(dd, [sf], [sf], stats, a["chunk_duration_s"],
                       a.get("prediction_horizon_s", a["chunk_duration_s"]), a["step_size_s"], a["warmup_s"],
                       dn, an, Path(f"{FMH}/eval_runs/modecode_cache"))
ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn)
TOK, TGT = [], []
with torch.no_grad():
    for batch in ld:
        _, diag_inputs, targets, _, tok = forward_batch(model, batch, dev)
        TOK.append(tok[MOD].detach().float().cpu()); TGT.append(targets[MOD].detach().float().cpu())
TOK = torch.cat(TOK, 0); TGT = torch.cat(TGT, 0)                 # (N,n_tok,d), (N,C,F,T)
ch = int(_hard(TGT, MODE_K).sum(dim=(0, 2, 3)).argmax())
Bt, Rt = baseline_residual(TGT)
Bnp = Bt[:, ch:ch + 1].float().cpu().numpy()            # baseline (for raw-magnitude recombine S=B+R)
Rt = Rt[:, ch:ch + 1].float().to(dev)                   # next-window residual, mode chan
TOK = TOK.to(dev)
N, ntok, dmodel = TOK.shape; _, _, Fq, Tq = Rt.shape
npf = a["spectro_patch_f"] and (Fq // a["spectro_patch_f"]) or 16; npt = ntok // npf
print(f"[gate0b] {MOD} {SHOT} ch{ch}: N={N} ntok={ntok} d={dmodel} grid={npf}x{npt} F={Fq} T={Tq}", flush=True)


def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class CondUNet(nn.Module):
    """Condition on backbone tokens (npf x npt x d) -> feature map upsampled to (F,T)."""
    def __init__(self, extra_in, cw=16, w=48):
        super().__init__()
        self.proj = nn.Conv2d(dmodel, cw, 1)
        self.up = nn.Upsample(size=(Fq, Tq), mode="nearest")
        ic = cw + extra_in
        self.e0, self.e1, self.e2 = blk(ic, w), blk(w, 2 * w), blk(2 * w, 4 * w)
        self.d1, self.d0 = blk(4 * w + 2 * w, 2 * w), blk(2 * w + w, w)
        self.outc = nn.Conv2d(w, 1, 1); self.pool = nn.MaxPool2d(2); self.u = nn.Upsample(scale_factor=2, mode="nearest")

    def cond(self, tok):
        g = tok.transpose(1, 2).reshape(tok.shape[0], dmodel, npf, npt)
        return self.up(self.proj(g))

    def forward(self, tok, extra=None):
        x = self.cond(tok)
        if extra is not None:
            x = torch.cat([x, extra], 1)
        s0 = self.e0(x); s1 = self.e1(self.pool(s0)); b = self.e2(self.pool(s1))
        d1 = self.d1(torch.cat([self.u(b), s1], 1)); d0 = self.d0(torch.cat([self.u(d1), s0], 1))
        return self.outc(d0)


def batch(bs=8):
    idx = torch.randint(0, N, (bs,)); return TOK[idx], Rt[idx]

det = CondUNet(extra_in=0).to(dev); od = torch.optim.Adam(det.parameters(), 2e-4)
for s in range(STEPS):
    ct, tt = batch(); loss = (det(ct) - tt).abs().mean()
    od.zero_grad(); loss.backward(); od.step()
    if (s + 1) % 1000 == 0:
        print(f"[gate0b] det step {s+1} mae={loss.item():.4f}", flush=True)

flw = CondUNet(extra_in=2).to(dev); of = torch.optim.Adam(flw.parameters(), 2e-4)
for s in range(STEPS):
    ct, tt = batch(); B = ct.shape[0]
    x0 = torch.randn_like(tt); t = torch.rand(B, 1, 1, 1, device=dev)
    xt_ = (1 - t) * x0 + t * tt; vtar = tt - x0
    v = flw(ct, torch.cat([xt_, t.expand(-1, 1, Fq, Tq)], 1))
    loss = ((v - vtar) ** 2).mean()
    of.zero_grad(); loss.backward(); of.step()
    if (s + 1) % 1000 == 0:
        print(f"[gate0b] flow step {s+1} fm={loss.item():.4f}", flush=True)


@torch.no_grad()
def sample(ct):
    x = torch.randn(ct.shape[0], 1, Fq, Tq, device=dev)
    for k in range(FLOW_STEPS):
        t = torch.full((ct.shape[0], 1, 1, 1), k / FLOW_STEPS, device=dev)
        x = x + (1.0 / FLOW_STEPS) * flw(ct, torch.cat([x, t.expand(-1, 1, Fq, Tq)], 1))
    return x

with torch.no_grad():
    Dp = torch.cat([det(TOK[i:i + 16]) for i in range(0, N, 16)], 0).cpu().numpy()
    Fs = torch.cat([sample(TOK[i:i + 16]) for i in range(0, N, 16)], 0).cpu().numpy()
G = Rt.cpu().numpy(); fmax = int(60 / (FS / NFFT / 1e3))


def peak(a):
    p = np.abs(a[:, 0, :fmax]).mean(2); return float(np.median(p.max(1) / (np.median(p, 1) + 1e-6)))


def mcorr(a):
    return float(np.nanmedian([np.corrcoef(G[w, 0, :fmax].ravel(), a[w, 0, :fmax].ravel())[0, 1] for w in range(N)]))


print(f"\n[gate0b] === RESULT (frozen backbone tokens -> next-window residual, N={N}) ===", flush=True)
print(f"[gate0b] {'head':<12}{'mode_corr':>10}{'peakiness':>12}", flush=True)
print(f"[gate0b] {'GT':<12}{1.0:>10.3f}{peak(G):>12.2f}", flush=True)
print(f"[gate0b] {'MAE(det)':<12}{mcorr(Dp):>10.3f}{peak(Dp):>12.2f}", flush=True)
print(f"[gate0b] {'flow(samp)':<12}{mcorr(Fs):>10.3f}{peak(Fs):>12.2f}", flush=True)

order = np.argsort(-np.abs(G[:, 0, :fmax]).sum((1, 2)))[:4]
FREQ = np.arange(Fq) * FS / NFFT / 1e3
fig, ax = plt.subplots(3, len(order), figsize=(3.4 * len(order), 8))
for j, w in enumerate(order):
    vmn, vmx = np.percentile(G[w, 0, :fmax], [2, 98])
    for r, (tt, d) in enumerate([("GT next", G), ("MAE pred", Dp), ("flow sample", Fs)]):
        ax[r, j].imshow(d[w, 0, :fmax], origin="lower", aspect="auto", cmap="magma", vmin=vmn, vmax=vmx,
                        extent=(0, Tq * HOP / FS * 1e3, 0, FREQ[fmax - 1]))
        ax[r, j].set_title(f"{tt} w{w}", fontsize=9)
        if j == 0:
            ax[r, j].set_ylabel("Freq (kHz)")
        if r == 2:
            ax[r, j].set_xlabel("Time (ms)")
fig.suptitle(f"Gate 0b — {MOD} {SHOT} ch{ch} forecast from FROZEN backbone tokens: MAE vs flow "
             f"(peak GT {peak(G):.1f}/MAE {peak(Dp):.1f}/flow {peak(Fs):.1f})", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
for e in ("png", "pdf"):
    fig.savefig(OUT / f"gate0b_{MOD}_{SHOT}.{e}", dpi=130, bbox_inches="tight")
print(f"[gate0b] saved {OUT}/gate0b_{MOD}_{SHOT}.png", flush=True)
