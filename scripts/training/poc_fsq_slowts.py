"""Adversarial FSQ codec for SLOW time-series (Thomson / CER / MSE profiles) —
per-modality analog of the spectro/video/fast-TS FSQ codecs. Slow-TS is one token
per channel over a tiny 5-sample (50 ms @ 100 Hz) window; the codec autoencodes the
per-channel profile through SlowTimeSeriesTokenizer -> FSQ -> SlowTimeSeriesHead.

Trains a codec for EACH slow-TS modality in one job (channel counts differ, so one
codec per modality: slowts_codec_<modality>.pt). Data is used in the dataset-
standardized space directly (NO extra per-window z-score — the 5-sample window is too
short to z-score stably; this matches the CE branch, which encodes targets as-is).

Discriminator = global MLP over the flattened (C*WIN) profile (a conv over 5 samples
is meaningless) — a profile-shape real/fake critic + feature-matching.

Env: MODALITIES(comma; default all 7) EVAL_SHOTS_FILE|EVAL_SHOTS FSQ_DIM(8) FSQ_L(8)
  AE_STEPS(3000) N_WINDOWS(120) AE_BS(64) ADV_LAMBDA(0.5) FM_LAMBDA(10) R1_GAMMA(10)
  D_LR(1e-4) RECON_WEIGHT(1) VAL_FRAC(0.15) MAX_SHOTS(200) D_MODEL(256) OUT_DIR
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
from tokamak_foundation_model.e2e.tokenizers.slow_time_series import SlowTimeSeriesTokenizer
from tokamak_foundation_model.e2e.output_heads import SlowTimeSeriesHead
from tokamak_foundation_model.e2e.quantizers import FSQBottleneck

D_MODEL = int(os.environ.get("D_MODEL", "256"))
WIN = 5  # slow_samples = 0.05 s * 100 Hz
SLOW_TS = [("ts_core_density", 44), ("ts_core_temp", 44), ("ts_tangential_density", 10),
           ("ts_tangential_temp", 10), ("cer_ti", 48), ("cer_rot", 48), ("mse", 69)]


class SlowTSFSQAutoencoder(nn.Module):
    """SlowTimeSeriesTokenizer -> FSQ bottleneck -> SlowTimeSeriesHead. n_tok = C."""

    def __init__(self, C, fsq_dim, fsq_L, d_model=D_MODEL):
        super().__init__()
        self.enc = SlowTimeSeriesTokenizer(n_channels=C, window_samples=WIN, d_model=d_model)
        self.n_tok = C
        self.fsq = FSQBottleneck(d_model, [fsq_L] * fsq_dim)
        self.dec = SlowTimeSeriesHead(d_model=d_model, n_channels=C, window_samples=WIN)
        self.dim, self.levels, self.C = fsq_dim, fsq_L, C

    def forward(self, x):                                  # x (B, C, WIN)
        tq, codes = self.fsq(self.enc(x))
        return self.dec(tq), codes                        # (B, C, WIN), (B, C, dim)


class SlowTSDiscriminator(nn.Module):
    """Global MLP critic over the flattened (C*WIN) profile. Returns (logits, feats)."""

    def __init__(self, C, hidden=256):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(C * WIN, hidden), nn.LeakyReLU(0.2, inplace=True))
        self.l2 = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, inplace=True))
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        f1 = self.l1(x.flatten(1)); f2 = self.l2(f1)
        return self.out(f2), [f1, f2]


def load_all_slowts_windows(shots, data_dir, stats_path, modalities, n_windows):
    """Load ALL slow-TS modalities in ONE pass per shot (7000->1000 dataset opens).
    Applies the SAME cleaning the production trainer uses (``_clean_and_mask``):
    NaN/Inf -> 0 + a per-element validity mask (1=finite/valid). MSE missing is
    Inf-encoded and CER missing is NaN — this handles both the way production does.
    Keeps only MAJORITY-VALID windows (>50% cells finite) so the codec trains on real
    profiles, and returns the masks so the reconstruction loss can ignore missing
    cells. Returns {modality: (X (N,C,WIN) cleaned, M (N,C,WIN) mask, shot_count)}."""
    stats = torch.load(stats_path, weights_only=False)
    names = [m for m, _ in modalities]
    cmap = {m: c for m, c in modalities}
    acc = {m: [] for m in names}
    accm = {m: [] for m in names}
    shot_ct = {m: 0 for m in names}
    for si, sh in enumerate(shots):
        try:
            # Reconstruction codec: NO input/target split needed. Use plain
            # (non-prediction) mode so the dataset exports the per-element
            # validity mask ``{name}_mask`` (dropped in prediction mode) — no
            # production data_loader change required.
            ds = TokamakMultiFileDataset(
                hdf5_paths=[Path(data_dir) / f"{sh}_processed.h5"], chunk_duration_s=0.05,
                step_size_s=0.01, warmup_s=1.0,
                preprocessing_stats=stats, input_signals=names, target_signals=names)
        except Exception as e:
            print(f"[slow] shot {sh} SKIP: {e}", flush=True); continue
        n = len(ds)
        if n == 0:
            continue
        idxs = range(n) if n_windows <= 0 else range(0, n, max(1, n // n_windows))
        got = {m: 0 for m in names}
        for i in idxs:
            inp = ds[i]  # non-prediction: flat dict with {name} + {name}_mask
            for m in names:
                v = inp.get(m)
                if v is None or torch.as_tensor(v).shape[0] != cmap[m]:
                    continue
                v = torch.as_tensor(v).float()                    # (C, WIN) may hold NaN/Inf
                finite = torch.isfinite(v)
                # Use the dataset's EXPORTED per-element mask (1=valid): it flags
                # NaN- (CER) / zero_is_missing- (TS) encoded cells that the loader
                # already zero-filled, which isfinite alone reports as valid.
                # Combine with isfinite so MSE's Inf (NOT caught by the NaN-based
                # dataset mask) is still masked out. Fallback to isfinite.
                dm = inp.get(f"{m}_mask")
                if dm is not None:
                    valid = torch.as_tensor(dm).float() * finite.float()
                else:
                    valid = finite.float()
                cleaned = torch.where(valid > 0.5, v, torch.zeros_like(v))
                # keep majority-valid windows (real profiles); mask carries the rest
                if float(valid.mean()) > 0.5:
                    acc[m].append(cleaned); accm[m].append(valid); got[m] += 1
        for m in names:
            if got[m] > 0:
                shot_ct[m] += 1
        if (si + 1) % 100 == 0:
            print(f"[slow] loaded {si+1}/{len(shots)} shots  "
                  + " ".join(f"{m}:{shot_ct[m]}sh" for m in names), flush=True)
    out = {}
    for m in names:
        if acc[m]:
            out[m] = (torch.stack(acc[m]), torch.stack(accm[m]), shot_ct[m])
        else:
            out[m] = (torch.empty(0), torch.empty(0), shot_ct[m])
    return out


def train_one(modality, C, X, M, shot_ct, out_dir, device, hp):
    if X.numel() == 0:
        print(f"[slow] {modality}: NO windows — SKIP", flush=True); return
    N = X.shape[0]; nv = max(1, int(N * hp["val_frac"])); ntr = N - nv
    Xtr, Mtr = X[:ntr], M[:ntr]
    print(f"[slow] {modality}: N={N} C={C} WIN={X.shape[2]} train={ntr} heldout={nv} "
          f"(from {shot_ct} shots, valid-frac {float(M.mean()):.3f})", flush=True)

    # DECODER-ONLY fine-tune: if FINETUNE_FROM_DIR is set, load this modality's
    # existing codec (slowts_codec_<modality>.pt), FREEZE enc+fsq (codes stay
    # BYTE-IDENTICAL so the frozen world model's predicted codes remain valid), and
    # train ONLY the decoder. AE is rebuilt from the SAVED cfg (not hp) so the
    # weights load exactly. Uses FT_STEPS (default 2500) instead of hp["ae_steps"].
    ft_dir = os.environ.get("FINETUNE_FROM_DIR", "").strip()
    ft_steps = hp["ae_steps"]
    if ft_dir:
        ft_path = Path(ft_dir) / f"slowts_codec_{modality}.pt"
        ck = torch.load(ft_path, map_location=device, weights_only=False)
        fcfg = ck["cfg"]
        ae = SlowTSFSQAutoencoder(fcfg["C"], fcfg["fsq_dim"], fcfg["fsq_L"],
                                  d_model=fcfg.get("d_model", D_MODEL)).to(device)
        ae.load_state_dict(ck["ae"])
        # keep saved-cfg values so the re-saved codec cfg matches the loaded model
        hp["fsq_dim"], hp["fsq_L"] = fcfg["fsq_dim"], fcfg["fsq_L"]
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
        ft_steps = int(os.environ.get("FT_STEPS", "2500"))
        print(f"[slow] {modality} DECODER-ONLY FINE-TUNE from {ft_path}: "
              f"n_enc_fsq_frozen={n_frozen} n_dec_trainable={n_dec} FT_STEPS={ft_steps}", flush=True)
    else:
        ae = SlowTSFSQAutoencoder(C, hp["fsq_dim"], hp["fsq_L"]).to(device)
        optG = torch.optim.Adam(ae.parameters(), 2e-4, betas=(0.5, 0.9))
    disc = SlowTSDiscriminator(C).to(device)
    optD = torch.optim.Adam(disc.parameters(), hp["d_lr"], betas=(0.5, 0.9))
    ntr_ = Xtr.shape[0]
    for s in range(ft_steps):
        idx = torch.randint(0, ntr_, (hp["ae_bs"],))
        x = Xtr[idx].to(device); m = Mtr[idx].to(device)         # cleaned window + validity mask
        with torch.no_grad():
            rec, _ = ae(x)
        xr = x.detach().requires_grad_(True)
        dr, _ = disc(xr); df, _ = disc(rec)
        dloss = F.relu(1 - dr).mean() + F.relu(1 + df).mean()
        if hp["r1"] > 0:
            grad = torch.autograd.grad(dr.sum(), xr, create_graph=True)[0]
            dloss = dloss + 0.5 * hp["r1"] * grad.pow(2).flatten(1).mean(1).mean()
        optD.zero_grad(set_to_none=True); dloss.backward(); optD.step()
        rec, _ = ae(x)
        mae = ((rec - x).abs() * m).sum() / (m.sum() + 1e-8)      # MASKED recon (ignore missing)
        dfg, ff = disc(rec)
        with torch.no_grad():
            _, fr = disc(x)
        gadv = -dfg.mean(); fm = sum((a - b).abs().mean() for a, b in zip(ff, fr)) / len(ff)
        gloss = hp["recon_w"] * mae + hp["adv"] * gadv + hp["fm"] * fm
        optG.zero_grad(set_to_none=True); gloss.backward(); optG.step()
        if (s + 1) % 500 == 0 or s == 0:
            print(f"  [slow] {modality} step {s+1}/{ft_steps} mae={mae.item():.4f} "
                  f"gadv={gadv.item():.3f} fm={fm.item():.3f} d={dloss.item():.3f}", flush=True)

    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    ck = out_dir / f"slowts_codec_{modality}.pt"
    torch.save({"ae": ae.state_dict(),
                "cfg": dict(modality=modality, C=C, WIN=WIN,
                            fsq_dim=hp["fsq_dim"], fsq_L=hp["fsq_L"], d_model=D_MODEL)}, ck)
    # held-out recon quality (MASKED to valid cells only)
    Xv, Mv = X[ntr:].to(device), M[ntr:].to(device)
    with torch.no_grad():
        REC = torch.cat([ae(Xv[i:i + 256])[0] for i in range(0, nv, 256)], 0)
    gt = Xv.cpu().numpy(); rc = REC.cpu().numpy(); mk = Mv.cpu().numpy().astype(bool)
    a = gt[mk] - gt[mk].mean(); b = rc[mk] - rc[mk].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    corr = float(a @ b / d) if d > 0 else 0.0
    mae_v = float(np.abs(gt[mk] - rc[mk]).mean())
    print(f"[slow] {modality} SAVED -> {ck}  HELD-OUT corr={corr:.3f} mae={mae_v:.4f} "
          f"(masked, {mk.mean():.2f} valid)", flush=True)


def render_frozen():
    """Load FROZEN slow-TS codecs from RENDER_CODEC_DIR and render GT-vs-recon
    PROFILES (value vs channel = the physical Thomson/CER/MSE profile shape) on the
    most-variable held-out windows, one figure per modality. Env: RENDER_CODEC_DIR
    OUT_DIR EVAL_SHOTS_FILE/EVAL_SHOTS MAX_SHOTS(40) MODALITIES."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.environ.get("EVAL_DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
    stats_path = os.environ.get("EVAL_STATS", "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    codec_dir = Path(os.environ["RENDER_CODEC_DIR"])
    out_dir = Path(os.environ.get("OUT_DIR", str(codec_dir))); out_dir.mkdir(parents=True, exist_ok=True)
    want = os.environ.get("MODALITIES", "").strip()
    mods = [(n, c) for n, c in SLOW_TS
            if (not want or n in want.split(",")) and (codec_dir / f"slowts_codec_{n}.pt").exists()]
    sf = os.environ.get("EVAL_SHOTS_FILE", "").strip()
    max_shots = int(os.environ.get("MAX_SHOTS", "40"))
    if sf:
        shots = [ln.split()[0] for ln in open(sf) if ln.strip() and not ln.startswith("#")][:max_shots]
    else:
        shots = [s.strip() for s in os.environ.get("EVAL_SHOTS", "200729,200226,200722,201664,201797").split(",") if s.strip()]

    def corr(a, b):
        a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b); return float(a @ b / d) if d > 0 else 0.0

    data = load_all_slowts_windows(shots, data_dir, stats_path, mods, int(os.environ.get("N_WINDOWS", "60")))
    for name, C in mods:
        X, _M, sc = data[name]
        if X.numel() == 0:
            print(f"[slow] render {name}: NO windows", flush=True); continue
        ck = torch.load(codec_dir / f"slowts_codec_{name}.pt", map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        ae = SlowTSFSQAutoencoder(C, cfg["fsq_dim"], cfg["fsq_L"]).to(device)
        ae.load_state_dict(ck["ae"]); ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            REC = torch.cat([ae(X[i:i + 256].to(device))[0].cpu() for i in range(0, X.shape[0], 256)], 0)
        gt = X.numpy(); rc = REC.numpy()
        mid = gt.shape[2] // 2
        pick = np.argsort(-gt.reshape(gt.shape[0], -1).std(1))[:6]   # most-varied profiles
        fig, ax = plt.subplots(2, 3, figsize=(15, 7)); ax = ax.ravel()
        for k, w in enumerate(pick):
            a = ax[k]
            a.plot(gt[w, :, mid], color="black", marker=".", ms=4, label="GT")
            a.plot(rc[w, :, mid], color="tab:orange", marker=".", ms=4, alpha=0.85, label="FSQ recon")
            a.set_title(f"win {int(w)} profile (t={mid}) corr={corr(gt[w], rc[w]):.2f}", fontsize=9)
            a.set_xlabel("channel")
            if k == 0:
                a.legend(fontsize=8)
        fig.suptitle(f"slow-TS FSQ recon — {name} (C={C}, {sc} shots, corr(all)={corr(gt, rc):.3f})")
        fig.tight_layout()
        p = out_dir / f"recon_{name}.png"; fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"[slow] RENDER {name} -> {p}  (corr={corr(gt, rc):.3f})", flush=True)
    print("=== FSQ SLOW-TS RENDER DONE ===", flush=True)


def main():
    if os.environ.get("RENDER_CODEC_DIR"):
        render_frozen(); return
    data_dir = os.environ.get("EVAL_DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
    stats_path = os.environ.get("EVAL_STATS", "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    out_dir = Path(os.environ.get("OUT_DIR", "eval_runs/fsq_slowts")); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    want = os.environ.get("MODALITIES", "").strip()
    mods = [(n, c) for n, c in SLOW_TS if (not want or n in want.split(","))]
    sf = os.environ.get("EVAL_SHOTS_FILE", "").strip()
    max_shots = int(os.environ.get("MAX_SHOTS", "200"))
    if sf:
        shots = [ln.split()[0] for ln in open(sf) if ln.strip() and not ln.startswith("#")][:max_shots]
    else:
        shots = [s.strip() for s in os.environ.get(
            "EVAL_SHOTS", "200729,200226,200722,201664,201797").split(",") if s.strip()]
    hp = dict(fsq_dim=int(os.environ.get("FSQ_DIM", "8")), fsq_L=int(os.environ.get("FSQ_L", "8")),
              ae_steps=int(os.environ.get("AE_STEPS", "3000")), n_windows=int(os.environ.get("N_WINDOWS", "120")),
              ae_bs=int(os.environ.get("AE_BS", "64")), adv=float(os.environ.get("ADV_LAMBDA", "0.5")),
              fm=float(os.environ.get("FM_LAMBDA", "10")), r1=float(os.environ.get("R1_GAMMA", "10")),
              d_lr=float(os.environ.get("D_LR", "1e-4")), recon_w=float(os.environ.get("RECON_WEIGHT", "1")),
              val_frac=float(os.environ.get("VAL_FRAC", "0.15")))
    print(f"[slow] modalities={[m for m,_ in mods]} shots={len(shots)} fsq {hp['fsq_dim']}x{hp['fsq_L']}", flush=True)
    data = load_all_slowts_windows(shots, data_dir, stats_path, mods, hp["n_windows"])
    print("[slow] per-modality shot coverage: "
          + " ".join(f"{m}:{data[m][2]}sh/{data[m][0].shape[0] if data[m][0].numel() else 0}win"
                     for m, _ in mods), flush=True)
    for name, C in mods:
        X, M, shot_ct = data[name]
        train_one(name, C, X, M, shot_ct, out_dir, device, hp)
    print("=== FSQ SLOW-TS CODECS DONE ===", flush=True)


if __name__ == "__main__":
    main()
