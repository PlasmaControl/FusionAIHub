#!/usr/bin/env python3
"""Self-contained FSQ autoencoder demo — single GPU, no DDP, no repo model code.

Trains a small convolutional encoder/decoder with a Finite Scalar Quantizer bottleneck on
tokamak spectrogram windows, and reports the three numbers that matter:

    nRMSE      RMSE / std(target), per (window, channel).  1.0 == predicting the window's
               own constant mean.  <1 beats a flat line.  This is a FLOOR, not a score:
               its exact minimiser is the conditional mean, i.e. blur.
    util       distinct codes used / codebook size.
    std_ratio  std(recon) / std(target), ideal 1.0.  Exposes amplitude collapse, which
               nRMSE alone rewards.

Only three repo imports, all for DATA (config shape + dataset + shot discovery). The model,
the quantizer and the training loop are implemented here.

    python fsq_autoencoder_demo.py --modality mhr --steps 2000
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from tokamak_foundation_model.ignite.config import SpectroCodecConfig
from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite import spike as spike_mod


# ----------------------------------------------------------------------------- FSQ
class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al. 2023).

    Each latent dimension is squashed with tanh and rounded to one of L integer levels, so the
    codebook is the implicit product grid: prod(levels) codes, no codebook parameters, no EMA,
    no commitment loss, and no dead entries to revive. Gradients pass straight through.
    """

    def __init__(self, levels):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))
        self.codebook_size = int(np.prod(levels))

    def forward(self, z):                                  # z: (B, N, D)
        L = self.levels
        half = (L - 1) / 2
        # even level counts need a half-step offset so the grid straddles zero symmetrically
        offset = torch.where(L % 2 == 0, torch.tensor(0.5, device=L.device), L.new_zeros(()))
        shift = torch.atanh(offset / (half + 1e-6)).nan_to_num()
        zt = torch.tanh(z + shift) * half - offset
        zq = zt + (torch.round(zt) - zt).detach()           # straight-through estimator
        codes = (torch.round(zt) + half).clamp_(min=torch.zeros_like(L), max=L - 1)
        # mixed-radix flatten -> one integer index per token
        radix = torch.cumprod(torch.cat([L.new_ones(1), L[:-1]]), 0)
        idx = (codes * radix).sum(-1).long()
        return zq / half, idx                               # normalised to [-1, 1]


# ------------------------------------------------------------------- encoder / decoder
class ConvAE(nn.Module):
    """Conv encoder -> FSQ -> conv decoder. Downsamples (F, T) by 16x16, so a
    (C, 512, 96) window becomes 32*6 = 192 tokens, matching the production token budget."""

    def __init__(self, ch, levels, width=128):
        super().__init__()
        d = len(levels)
        self.enc = nn.Sequential(
            nn.Conv2d(ch, width, 4, 2, 1), nn.GELU(),        # /2
            nn.Conv2d(width, width, 4, 2, 1), nn.GELU(),     # /4
            nn.Conv2d(width, width, 4, 2, 1), nn.GELU(),     # /8
            nn.Conv2d(width, width, 4, 2, 1), nn.GELU(),     # /16
            nn.Conv2d(width, d, 1),                          # -> latent dim
        )
        self.dec = nn.Sequential(
            nn.Conv2d(d, width, 1), nn.GELU(),
            nn.ConvTranspose2d(width, width, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(width, width, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(width, width, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(width, ch, 4, 2, 1),
        )
        self.fsq = FSQ(levels)

    def forward(self, x):
        z = self.enc(x)                                      # (B, d, f, t)
        B, d, f, t = z.shape
        zq, idx = self.fsq(z.permute(0, 2, 3, 1).reshape(B, f * t, d))
        zq = zq.reshape(B, f, t, d).permute(0, 3, 1, 2)
        return self.dec(zq), idx


# --------------------------------------------------------------------------- metrics
def report(recon, target):
    """nRMSE and std_ratio, per (window, channel), averaged. Mirrors gate.full_spectro_metrics."""
    r = recon.detach().float().flatten(2)                    # (B, C, F*T)
    t = target.detach().float().flatten(2)
    sd = t.std(dim=2)
    ok = sd > 1e-8
    nrmse = (((r - t) ** 2).mean(2).sqrt() / sd.clamp(min=1e-8))[ok].mean()
    std_ratio = (r.std(dim=2) / sd.clamp(min=1e-8))[ok].mean()
    return float(nrmse), float(std_ratio)


# ------------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", default="mhr", help="ece | bes | co2 | mhr")
    p.add_argument("--levels", default="8,5,5,5", help="FSQ levels; product = codebook size")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)         # measured codec optimum
    p.add_argument("--n_shots", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--out", default="fsq_demo.pt")
    a = p.parse_args()

    levels = [int(x) for x in a.levels.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # ---- data: the only repo dependency ------------------------------------------------
    cfg = SpectroCodecConfig()
    shots = spike_mod.discover_shots(tc.DEFAULT_DATA_DIR)
    train_shots, val_shots = shots[:a.n_shots], shots[-8:]
    train_ds = tc.CodecPairDataset(a.modality, train_shots, cfg,
                                   data_dir=tc.DEFAULT_DATA_DIR, lengths_cache_path=None)
    val_ds = tc.CodecPairDataset(a.modality, val_shots, cfg,
                                 data_dir=tc.DEFAULT_DATA_DIR, lengths_cache_path=None)
    val_ds = Subset(val_ds, np.linspace(0, len(val_ds) - 1, 64).astype(int).tolist())
    dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=4,
                    drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=a.batch_size, num_workers=2)

    x0, _ = next(iter(dl))
    ch = x0.shape[1]
    print(f"[demo] {a.modality}: window {tuple(x0.shape[1:])}  levels {levels} "
          f"= {int(np.prod(levels))} codes  device {dev}", flush=True)

    model = ConvAE(ch, levels, a.width).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.8, 0.99))
    n_par = sum(q.numel() for q in model.parameters())
    print(f"[demo] {n_par/1e6:.2f} M params", flush=True)

    # ---- train -------------------------------------------------------------------------
    step, t0, it = 0, time.time(), iter(dl)
    while step < a.steps:
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(dl); continue
        x = x.to(dev, non_blocking=True)
        recon, _ = model(x)
        loss = F.l1_loss(recon, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 200 == 0 or step == 1:
            nr, sr = report(recon, x)
            print(f"  step {step:5d}  L1 {loss.item():.4f}  nRMSE {nr:.4f}  "
                  f"std_ratio {sr:.3f}  ({(time.time()-t0)/step:.2f}s/step)", flush=True)

    # ---- validate ----------------------------------------------------------------------
    model.eval()
    nrs, srs, seen = [], [], set()
    with torch.no_grad():
        for x, _ in val_dl:
            x = x.to(dev)
            recon, idx = model(x)
            nr, sr = report(recon, x)
            nrs.append(nr); srs.append(sr)
            seen.update(idx.flatten().tolist())
    util = len(seen) / int(np.prod(levels))
    print("\n=== held-out ===")
    print(f"  nRMSE      {np.mean(nrs):.4f}   (1.0 = the window's own constant mean)")
    print(f"  std_ratio  {np.mean(srs):.4f}   (1.0 = correct amplitude; << 1 means blur)")
    print(f"  utilization {len(seen)}/{int(np.prod(levels))} = {100*util:.1f}%")
    torch.save({"model": model.state_dict(), "levels": levels, "ch": ch}, a.out)
    print(f"  saved {a.out}")


if __name__ == "__main__":
    main()
