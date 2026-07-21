"""Stage-B POC: FSQ code prediction (categorical) vs persistence.

The proof that a DISCRETE autoregressive objective predicts spectrogram modes
WITHOUT mean-collapse — the thing continuous regression (MAE / flow / dice) could
not do. Mirrors the production 1a->1b structure at small scale on ONE shot:

  1a  train an FSQ-AE (encoder -> FSQ -> decoder) on TRAIN windows; FREEZE it.
      (frozen tokenizer => stationary code targets, the standard discrete-AR recipe)
  1b  train a code predictor: input-window codes -> TARGET-window per-dim codes
      via cross-entropy (categorical, cannot collapse to a mean).

Evaluate on a HELD-OUT TEMPORAL split (later-time windows the predictor never
saw): sample predicted codes -> decode -> mode-Dice vs the persistence baseline
(copy the input window's modes). The prediction horizon is the dataset's
prediction_horizon_s (0.05 s ahead), so beating persistence = learning real
0.05 s-ahead mode dynamics, not copying.

SUCCESS = predictor mode-Dice > persistence on held-out, with visibly sharp modes.

Env: EVAL_SHOT(200729) FSQ_DIM(24) FSQ_L(8) AE_STEPS(3000) PRED_STEPS(4000)
     N_WINDOWS(0=all) VAL_FRAC(0.3) N_CHANNELS(8) OUT_DIR.
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
from tokamak_foundation_model.e2e.tokenizers.spectrogram import SpectrogramTokenizer
from tokamak_foundation_model.e2e.output_heads import SpectrogramOutputHead
from tokamak_foundation_model.e2e.quantizers import FSQBottleneck
from train_e2e_stage1 import (
    _spec_mode_arg, _SPEC_STRUCT_GAMMA, _SPEC_STRUCT_CUT, _SPEC_STRUCT_K,
)

PATCH_F, PATCH_T, D_MODEL = 64, 32, 256


def _hard(x, k):
    return (_spec_mode_arg(x, k).clamp(0.0, 1.0) ** _SPEC_STRUCT_GAMMA
            > _SPEC_STRUCT_CUT).float()


def pooled_dice(pred_mask, gt_mask):
    """Pooled Dice over mode-bearing (gt sum>=3) channel-windows. Masks (N,C,F,T)."""
    g = gt_mask.sum(dim=(-2, -1))
    mb = g >= 3
    if int(mb.sum()) == 0:
        return float("nan")
    ov = (pred_mask * gt_mask).sum(dim=(-2, -1))
    ps = pred_mask.sum(dim=(-2, -1))
    return float((2 * ov[mb].sum()) / (ps[mb].sum() + g[mb].sum() + 1e-6))


# --------------------------------------------------------------------------- #
class FSQAutoencoder(nn.Module):
    """encoder (patch tokenizer) -> FSQ bottleneck -> decoder (deterministic).

    per_channel=False: all C channels folded into one 24-token budget (production
    default — the extreme bottleneck). per_channel=True: a SHARED single-channel
    codec gives each channel its OWN 24 tokens -> C*24 total tokens (the capacity
    test; would 40x the production spectro token budget)."""

    def __init__(self, C, F_, T_, fsq_dim, fsq_L, per_channel=False):
        super().__init__()
        self.C, self.per_channel = C, per_channel
        enc_ch = 1 if per_channel else C
        self.enc = SpectrogramTokenizer(
            n_channels=enc_ch, d_model=D_MODEL, patch_f=PATCH_F, patch_t=PATCH_T,
            freq_bins=F_, time_frames=T_, enable_freq_stem=True)
        npf, npt = F_ // PATCH_F, T_ // PATCH_T
        self.n_tok_per = npf * npt                       # 24 tokens per channel-group
        self.n_tok = self.n_tok_per * (C if per_channel else 1)
        self.fsq = FSQBottleneck(D_MODEL, [fsq_L] * fsq_dim)
        self.dec = SpectrogramOutputHead(
            n_channels=enc_ch, d_model=D_MODEL, patch_f=PATCH_F, patch_t=PATCH_T,
            n_patches_f=npf, n_patches_t=npt)

    def _fold(self, x):                                  # (B,C,F,T) -> (B*C,1,F,T)
        return x.reshape(x.shape[0] * self.C, 1, *x.shape[2:]) if self.per_channel else x

    def forward(self, x):
        B = x.shape[0]
        tq, codes = self.fsq(self.enc._encode(self._fold(x)))
        rec = self.dec(tq)
        if self.per_channel:
            rec = rec.reshape(B, self.C, *rec.shape[2:])
            codes = codes.reshape(B, self.n_tok, -1)     # (B, C*24, dim)
        return rec, codes

    @torch.no_grad()
    def encode_codes(self, x):
        B = x.shape[0]
        _, codes = self.fsq(self.enc._encode(self._fold(x)))
        return codes.reshape(B, self.n_tok, -1) if self.per_channel else codes

    def decode_codes(self, codes):                       # codes (B, n_tok, dim)
        if self.per_channel:
            B = codes.shape[0]
            codes = codes.reshape(B * self.C, self.n_tok_per, -1)
            rec = self.dec(self.fsq.codes_to_tokens(codes))
            return rec.reshape(B, self.C, *rec.shape[2:])
        return self.dec(self.fsq.codes_to_tokens(codes))


class CodePredictor(nn.Module):
    """input-window per-dim codes -> next-window per-dim code LOGITS (categorical)."""

    def __init__(self, n_tok, dim, levels, d_pred=256, n_layers=4, n_heads=8):
        super().__init__()
        self.dim, self.levels = dim, levels
        self.embs = nn.ModuleList([nn.Embedding(levels, d_pred) for _ in range(dim)])
        self.pos = nn.Parameter(torch.randn(n_tok, d_pred) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_pred, n_heads, d_pred * 4, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.heads = nn.ModuleList([nn.Linear(d_pred, levels) for _ in range(dim)])

    def forward(self, codes):                         # codes (B, n_tok, dim) int
        h = sum(self.embs[d](codes[..., d]) for d in range(self.dim))
        h = self.tr(h + self.pos[None])
        return torch.stack([hd(h) for hd in self.heads], dim=2)  # (B,n_tok,dim,levels)


class SpectroDiscriminator(nn.Module):
    """PatchGAN discriminator on spectrograms (real vs FSQ-reconstructed) — the
    VQ-GAN / audio-codec ingredient that forces the decoder to render SHARP modes
    instead of the blurry MAE mean (which no amount of code prediction can fix).
    Returns (patch_logits, [features]) for hinge + feature-matching losses."""

    def __init__(self, C, base=64):
        super().__init__()

        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 4, s, 1),
                                 nn.GroupNorm(min(8, o), o),
                                 nn.LeakyReLU(0.2, inplace=True))
        self.b1 = blk(C, base, 2)
        self.b2 = blk(base, base * 2, 2)
        self.b3 = blk(base * 2, base * 4, 2)
        self.out = nn.Conv2d(base * 4, 1, 3, 1, 1)

    def forward(self, x):
        f1 = self.b1(x); f2 = self.b2(f1); f3 = self.b3(f2)
        return self.out(f3), [f1, f2, f3]


# --------------------------------------------------------------------------- #
def load_pairs(shot, data_dir, stats_path, n_channels, n_windows, drop_pad_std=0.4,
               modality="ece"):
    """Load ordered (input, target) spectrogram pairs for one shot (prediction
    mode, horizon 0.05 s). Returns X_in, X_tgt (N,C,F,T) cropped to patch multiples.

    Drops post-shot PADDING windows: many shots have a frozen flatline tail
    (constant signal -> std ~0.16 in norm units) that STFTs to an identical
    spectrogram every window (persistence=1.0 artifact). We keep only windows
    whose input AND target std exceed drop_pad_std, so the temporal split lands
    entirely in real, mode-active signal. Time order is preserved."""
    stats = torch.load(stats_path, weights_only=False)
    ds = TokamakMultiFileDataset(
        hdf5_paths=[Path(data_dir) / f"{shot}_processed.h5"], chunk_duration_s=0.05,
        prediction_mode=True, prediction_horizon_s=0.05, step_size_s=0.01,
        warmup_s=1.0, n_fft=1024, hop_length=256, preprocessing_stats=stats,
        input_signals=[modality], target_signals=[modality])
    n = len(ds)
    idxs = range(n) if n_windows <= 0 else range(0, n, max(1, n // n_windows))
    xin, xtg = [], []
    for i in idxs:
        s = ds[i]
        a = torch.nan_to_num(torch.as_tensor(s["inputs"][modality]).float())
        b = torch.nan_to_num(torch.as_tensor(s["targets"][modality]).float())
        xin.append(a); xtg.append(b)
    X_in, X_tgt = torch.stack(xin), torch.stack(xtg)
    C = min(n_channels, X_in.shape[1])
    cf = (X_in.shape[2] // PATCH_F) * PATCH_F
    ct = (X_in.shape[3] // PATCH_T) * PATCH_T
    X_in = X_in[:, :C, :cf, :ct].contiguous()
    X_tgt = X_tgt[:, :C, :cf, :ct].contiguous()
    # drop padding: keep windows whose input AND target carry real signal
    si = X_in.std(dim=(1, 2, 3)); st = X_tgt.std(dim=(1, 2, 3))
    keep = (si > drop_pad_std) & (st > drop_pad_std)
    n0 = X_in.shape[0]; nk = int(keep.sum())
    print(f"[stageB] padding filter (std>{drop_pad_std}): kept {nk}/{n0} windows "
          f"(dropped {n0 - nk} flatline)", flush=True)
    return X_in[keep].contiguous(), X_tgt[keep].contiguous()


def train_module(model, step_fn, steps, lr, bs, n, device, tag):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for s in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = step_fn(idx)
        loss.backward(); opt.step()
        if (s + 1) % 500 == 0 or s == 0:
            print(f"  [{tag}] step {s+1}/{steps} loss={loss.item():.4f}", flush=True)


def main():
    shot = os.environ.get("EVAL_SHOT", "200729")
    data_dir = os.environ.get("EVAL_DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
    stats_path = os.environ.get("EVAL_STATS", "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    fsq_dim = int(os.environ.get("FSQ_DIM", "24"))
    fsq_L = int(os.environ.get("FSQ_L", "8"))
    ae_steps = int(os.environ.get("AE_STEPS", "3000"))
    pred_steps = int(os.environ.get("PRED_STEPS", "4000"))
    n_windows = int(os.environ.get("N_WINDOWS", "0"))
    val_frac = float(os.environ.get("VAL_FRAC", "0.3"))
    n_channels = int(os.environ.get("N_CHANNELS", "8"))
    per_channel = bool(os.environ.get("PER_CHANNEL", ""))   # 24 tokens PER channel
    drop_pad_std = float(os.environ.get("DROP_PAD_STD", "0.4"))
    # per-channel processes B*C single-channel images + a longer token sequence,
    # so use smaller batches (overridable) to stay within GCD memory.
    ae_bs = int(os.environ.get("AE_BS", "8" if per_channel else "32"))
    pred_bs = int(os.environ.get("PRED_BS", "24" if per_channel else "64"))
    # patch size controls token count: folded tokens = (F//pf)*(T//pt); per-channel
    # multiplies by #channels. Override to sweep the spectro token budget.
    global PATCH_F, PATCH_T
    PATCH_F = int(os.environ.get("PATCH_F", str(PATCH_F)))
    PATCH_T = int(os.environ.get("PATCH_T", str(PATCH_T)))
    # WEIGHTED_CE=1: up-weight the CE loss on MODE tokens (rare) so the predictor
    # can't win by collapsing to the majority "background" code. MODE_WEIGHT =
    # loss multiplier for tokens whose patch (any channel) contains modes.
    weighted_ce = bool(os.environ.get("WEIGHTED_CE", ""))
    mode_weight = float(os.environ.get("MODE_WEIGHT", "20"))
    # SPEC_RECON_WEIGHT>1: mode-weight the AE reconstruction MAE so the codes must
    # preserve the thin modes (plain MAE is mean-seeking -> smooths them away, so
    # the codes never encode modes and no predictor can recover them).
    recon_weight = float(os.environ.get("SPEC_RECON_WEIGHT", "1"))
    # SPEC_ADV=1: train the FSQ-AE ADVERSARIALLY (VQ-GAN / audio-codec recipe) so
    # the decoder renders sharp modes instead of the MAE mean. adv/fm lambdas tune
    # the adversarial + feature-matching terms.
    spec_adv = bool(os.environ.get("SPEC_ADV", ""))
    adv_lambda = float(os.environ.get("ADV_LAMBDA", "0.5"))
    fm_lambda = float(os.environ.get("FM_LAMBDA", "10"))
    # GAN rebalance: R1 gradient penalty on real (regularizes D) + lower D lr, so
    # the discriminator can't overpower the generator (removes late-imbalance +
    # band artifacts). R1_GAMMA=0 disables R1.
    r1_gamma = float(os.environ.get("R1_GAMMA", "10"))
    d_lr = float(os.environ.get("D_LR", "1e-4"))
    out_dir = Path(os.environ.get("OUT_DIR", "eval_runs/fsq_stageB"))
    out_dir.mkdir(parents=True, exist_ok=True)
    # FIGURE_ONLY=1 reloads the saved AE+predictor and skips ALL training (for
    # figure / metric / channel-block tweaks — seconds instead of a full retrain).
    figure_only = bool(os.environ.get("FIGURE_ONLY", ""))
    ckpt_path = out_dir / "stageB_ckpt.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = _SPEC_STRUCT_K.get("ece", 2.0)

    print(f"[stageB] shot={shot} fsq_dim={fsq_dim} L={fsq_L} ae_steps={ae_steps} "
          f"pred_steps={pred_steps} val_frac={val_frac}", flush=True)
    # EVAL_SHOTS (comma list) pools MULTIPLE shots for a generalization run; each
    # shot is temporally split (early->train, late->held-out) then pooled, so the
    # held-out set is unseen late-time windows ACROSS all shots.
    shots = [s.strip() for s in os.environ.get("EVAL_SHOTS", shot).split(",") if s.strip()]
    tr_in, tr_tg, va_in, va_tg, va_shot = [], [], [], [], []
    for sh in shots:
        xi, xt = load_pairs(sh, data_dir, stats_path, n_channels, n_windows, drop_pad_std)
        ns = xi.shape[0]; nv = max(1, int(ns * val_frac)); nt = ns - nv
        tr_in.append(xi[:nt]); tr_tg.append(xt[:nt])
        va_in.append(xi[nt:]); va_tg.append(xt[nt:])
        va_shot += [sh] * nv
        print(f"[stageB]   shot {sh}: {ns} real pairs -> train {nt} / held-out {nv}", flush=True)
    X_in = torch.cat(tr_in + va_in, 0); X_tgt = torch.cat(tr_tg + va_tg, 0)
    n_tr = sum(t.shape[0] for t in tr_in); N = X_in.shape[0]
    C, Fq, Tq = X_in.shape[1:]
    va_shot = np.array(va_shot)            # per-held-out-window shot id (order = X_*[n_tr:])
    print(f"[stageB] {len(shots)} shot(s), {N} pairs (C={C} F={Fq} T={Tq}) -> "
          f"train {n_tr} / held-out {N - n_tr} (per-shot temporal split)", flush=True)
    X_in, X_tgt = X_in.to(device), X_tgt.to(device)

    # ---- persistence baseline on held-out (the bar to beat) ----
    with torch.no_grad():
        m_in_v = _hard(X_in[n_tr:], k); m_tg_v = _hard(X_tgt[n_tr:], k)
    persist = pooled_dice(m_in_v, m_tg_v)
    print(f"[stageB] PERSISTENCE (held-out, copy input modes): mode-Dice={persist:.3f}", flush=True)

    # ======================= 1a: FSQ-AE (train, freeze) =======================
    ae = FSQAutoencoder(C, Fq, Tq, fsq_dim, fsq_L, per_channel=per_channel).to(device)
    print(f"[stageB] tokenization: {'PER-CHANNEL' if per_channel else 'folded'} "
          f"-> {ae.n_tok} tokens ECE ({ae.n_tok_per}/channel-group x "
          f"{C if per_channel else 1})", flush=True)
    reload = figure_only and ckpt_path.exists()
    if reload:
        sd = torch.load(ckpt_path, map_location=device)
        ae.load_state_dict(sd["ae"])
        print(f"[stageB] FIGURE_ONLY: reloaded AE+predictor from {ckpt_path} "
              "(skipping all training)", flush=True)
    else:
        X_ae = torch.cat([X_in[:n_tr], X_tgt[:n_tr]], 0)  # AE trains on TRAIN windows only
        def _recon_mae(x, recon):
            d = (recon - x).abs()
            if recon_weight > 1:                          # mode-weighted reconstruction
                with torch.no_grad():
                    wm = 1.0 + (recon_weight - 1.0) * _hard(x, k)
                return (d * wm).sum() / wm.sum()
            return d.mean()
        if spec_adv:
            # VQ-GAN / audio-codec recipe: adversarial + feature-matching so the
            # decoder renders SHARP modes (the MAE mean is what smooths them away).
            disc = SpectroDiscriminator(C).to(device)
            optG = torch.optim.Adam(ae.parameters(), lr=2e-4, betas=(0.5, 0.9))
            optD = torch.optim.Adam(disc.parameters(), lr=d_lr, betas=(0.5, 0.9))
            print("[stageB] === 1a: train FSQ-AE ADVERSARIALLY (VQ-GAN style: "
                  f"adv x{adv_lambda} + fm x{fm_lambda} + mode-recon, R1 g{r1_gamma} "
                  f"D-lr {d_lr}) ===", flush=True)
            n_ae = X_ae.shape[0]
            for s in range(ae_steps):
                idx = torch.randint(0, n_ae, (ae_bs,), device=device)
                x = X_ae[idx]
                with torch.no_grad():                     # --- D step ---
                    recon, _ = ae(x)
                xr = x.detach().requires_grad_(True)
                dr, _ = disc(xr); df, _ = disc(recon)
                d_loss = F.relu(1 - dr).mean() + F.relu(1 + df).mean()
                if r1_gamma > 0:                          # R1 gradient penalty on real
                    g = torch.autograd.grad(dr.sum(), xr, create_graph=True)[0]
                    # mean-per-element (not sum) so gamma is input-size-independent
                    d_loss = d_loss + 0.5 * r1_gamma * g.pow(2).flatten(1).mean(1).mean()
                optD.zero_grad(set_to_none=True); d_loss.backward(); optD.step()
                recon, _ = ae(x)                          # --- G step ---
                mae = _recon_mae(x, recon)
                dfg, ff = disc(recon)
                with torch.no_grad():
                    _, fr = disc(x)
                g_adv = -dfg.mean()
                fm = sum((a - b).abs().mean() for a, b in zip(ff, fr)) / len(ff)
                g_loss = mae + adv_lambda * g_adv + fm_lambda * fm
                optG.zero_grad(set_to_none=True); g_loss.backward(); optG.step()
                if (s + 1) % 500 == 0 or s == 0:
                    print(f"  [ae-adv] step {s+1}/{ae_steps} mae={mae.item():.4f} "
                          f"g_adv={g_adv.item():.3f} fm={fm.item():.3f} "
                          f"d={d_loss.item():.3f}", flush=True)
        else:
            def ae_step(idx):
                recon, _ = ae(X_ae[idx]); return _recon_mae(X_ae[idx], recon)
            print("[stageB] === 1a: train FSQ-AE (reconstruction), then FREEZE ===", flush=True)
            train_module(ae, ae_step, ae_steps, 2e-3, ae_bs, X_ae.shape[0], device, "ae")
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    with torch.no_grad():                                # AE recon quality (held-out)
        rec_v, _ = ae(X_tgt[n_tr:])
        ae_dice = pooled_dice(_hard(rec_v, k), m_tg_v)
    print(f"[stageB] frozen-AE recon mode-Dice (held-out): {ae_dice:.3f}", flush=True)

    # ---- encode all windows -> codes (frozen) ----
    def enc_all(X):
        out = []
        for i in range(0, X.shape[0], 64):
            out.append(ae.encode_codes(X[i:i + 64]))
        return torch.cat(out, 0)
    C_in, C_tg = enc_all(X_in), enc_all(X_tgt)            # (N, n_tok, dim) int
    n_tok = C_in.shape[1]

    # CLASS-weighted CE: weight each FSQ code CLASS (per-dim level) by inverse
    # frequency over the TRAIN target codes. The "background" levels are common ->
    # down-weighted; the rare mode-encoding levels -> up-weighted, so the predictor
    # can't win by collapsing to the majority (background) code. Works at ANY token
    # granularity (unlike per-token weighting, degenerate when all tokens fold modes).
    class_w = None
    if weighted_ce:
        with torch.no_grad():
            oneh = F.one_hot(C_tg[:n_tr], fsq_L).float()      # (n_tr, n_tok, dim, L)
            freq = oneh.sum(dim=(0, 1)) / (n_tr * n_tok)      # (dim, L) level freqs
            class_w = 1.0 / (freq + 1e-4)                     # inverse frequency
            # normalize so the DATA-EXPECTED weight = 1 per dim (preserves loss
            # scale; absent levels don't distort it, unlike a plain mean).
            norm = (freq * class_w).sum(dim=1, keepdim=True) + 1e-6
            class_w = (class_w / norm).clamp(max=mode_weight)
        print(f"[stageB] WEIGHTED_CE (per-dim inv-freq class weights, data-norm, cap "
              f"x{mode_weight:.0f}): max {float(class_w.max()):.1f} "
              f"min {float(class_w.min()):.3f}", flush=True)

    # ======================= 1b: code predictor (CE) =========================
    pred = CodePredictor(n_tok, fsq_dim, fsq_L).to(device)
    if reload:
        pred.load_state_dict(sd["pred"])
    else:
        dim_idx = torch.arange(fsq_dim, device=device).view(1, 1, fsq_dim)
        def pred_step(idx):
            logits = pred(C_in[:n_tr][idx])              # (B,n_tok,dim,levels)
            tgt = C_tg[:n_tr][idx]                       # (B,n_tok,dim)
            if weighted_ce:
                ce = F.cross_entropy(logits.reshape(-1, fsq_L), tgt.reshape(-1),
                                     reduction="none").reshape(tgt.shape)
                w = class_w[dim_idx.expand_as(tgt), tgt]  # (B,n_tok,dim) per-class weight
                return (ce * w).sum() / (w.sum() + 1e-6)
            return F.cross_entropy(logits.reshape(-1, fsq_L), tgt.reshape(-1))
        print("[stageB] === 1b: train code predictor (cross-entropy on frozen codes) ===", flush=True)
        train_module(pred, pred_step, pred_steps, 1e-3, pred_bs, n_tr, device, "pred")
        torch.save({"ae": ae.state_dict(), "pred": pred.state_dict(),
                    "cfg": dict(C=C, Fq=Fq, Tq=Tq, fsq_dim=fsq_dim, fsq_L=fsq_L,
                                n_tok=n_tok, n_tr=n_tr)}, ckpt_path)
        print(f"[stageB] saved checkpoint -> {ckpt_path} (rerun with FIGURE_ONLY=1 "
              "to reload, no retrain)", flush=True)
    pred.eval()

    # ======================= eval on held-out ================================
    with torch.no_grad():
        logits = pred(C_in[n_tr:])                       # (Nv,n_tok,dim,levels)
        probs = logits.softmax(-1)
        Nv = logits.shape[0]
        # sampled (multinomial) and greedy (argmax) predicted codes
        samp = torch.multinomial(probs.reshape(-1, fsq_L), 1).reshape(Nv, n_tok, fsq_dim)
        greedy = logits.argmax(-1)
        spec_samp = ae.decode_codes(samp)
        spec_greedy = ae.decode_codes(greedy)
        d_samp = pooled_dice(_hard(spec_samp, k), m_tg_v)
        d_greedy = pooled_dice(_hard(spec_greedy, k), m_tg_v)
        # code-level accuracy vs persistence (fraction of dims predicted correctly)
        code_acc = float((greedy == C_tg[n_tr:]).float().mean())
        code_persist = float((C_in[n_tr:] == C_tg[n_tr:]).float().mean())

    print("\n[stageB] ================= HELD-OUT RESULTS =================", flush=True)
    print(f"{'method':>22} | mode-Dice", flush=True)
    print("-" * 40, flush=True)
    print(f"{'persistence (copy)':>22} | {persist:.3f}", flush=True)
    print(f"{'frozen-AE recon (ceil)':>22} | {ae_dice:.3f}", flush=True)
    print(f"{'PREDICTOR sampled':>22} | {d_samp:.3f}", flush=True)
    print(f"{'PREDICTOR greedy':>22} | {d_greedy:.3f}", flush=True)
    print(f"[stageB] code accuracy: predictor {code_acc:.3f} vs persistence {code_persist:.3f}", flush=True)
    beat = d_samp > persist + 0.02 or d_greedy > persist + 0.02
    print(f"[stageB] VERDICT: {'BEATS persistence -> mode dynamics LEARNED (categorical works)' if beat else 'does NOT beat persistence'}", flush=True)

    # ---- comparison figure: MODE-CONTRAST view on the most-dynamic channel.
    #      Chirping modes are low-freq bands invisible in raw log-power but clear
    #      under PER-FREQ CONTRAST (z per freq over time) + a low-freq zoom.
    WARMUP_S, STEP, CHUNK, HORIZON = 1.0, 0.01, 0.05, 0.05
    STRIDE = max(1, round(CHUNK / STEP))    # =5 -> pick NON-overlapping windows
    F_ZOOM = float(os.environ.get("FIG_FMAX_KHZ", "60"))
    st = torch.load(stats_path, weights_only=False)
    lm = np.asarray(st["ece"]["log"]["mean"]); ls = np.asarray(st["ece"]["log"]["std"])

    # predict codes for ALL windows (greedy), decode -> predicted spectrograms
    with torch.no_grad():
        parts = []
        for i in range(0, N, 64):
            parts.append(ae.decode_codes(pred(C_in[i:i + 64]).argmax(-1)))
        spec_all = torch.cat(parts, 0)                    # (N, C, F, T)

    # render FIG_SHOT's HELD-OUT windows (unseen late-time); for multi-shot runs
    # this isolates one shot's held-out region for a clean per-freq-contrast view.
    fig_shot = os.environ.get("FIG_SHOT", shots[0])
    holdout = np.where(va_shot == fig_shot)[0] + n_tr     # global indices of FIG_SHOT held-out
    if holdout.size == 0:
        holdout = np.arange(n_tr, N)
    sel = list(holdout[::STRIDE])                         # non-overlapping held-out windows
    def stitch_all(X4d):                                  # -> (C, F, n_sel*T) denorm
        arr = X4d[sel].cpu().numpy()                      # (n_sel, C, F, T)
        n_w, Cc, Fh, Th = arr.shape
        arr = arr * ls[None, :Cc, None, None] + lm[None, :Cc, None, None]
        return arr.transpose(1, 2, 0, 3).reshape(Cc, Fh, n_w * Th)
    G, P, PER = stitch_all(X_tgt), stitch_all(spec_all), stitch_all(X_in)
    # display channel: most time-variable (the chirping-mode channels), or FIG_CHANNEL
    fc = os.environ.get("FIG_CHANNEL", "")
    ch = int(fc) if fc else int(G.std(axis=2).mean(axis=1).argmax())

    def pfz(a2d):                                         # per-freq z over time -> mode contrast
        m = a2d.mean(1, keepdims=True); s = a2d.std(1, keepdims=True) + 1e-6
        return np.clip((a2d - m) / s, 0, 4)
    gz, pz, perz = pfz(G[ch]), pfz(P[ch]), pfz(PER[ch])
    np.savez(out_dir / f"{fig_shot}_stageB_arrays.npz", gt=G[ch], pred=P[ch], persist=PER[ch],
             gt_z=gz, pred_z=pz, persist_z=perz, ch=ch, n_tr=n_tr, stride=STRIDE,
             warmup=WARMUP_S, chunk=CHUNK, d_samp=d_samp, persist_dice=persist,
             ae_dice=ae_dice, code_acc=code_acc, code_persist=code_persist)
    fmax_bin = int(F_ZOOM / (500.0 / 1024.0))             # bins up to F_ZOOM kHz
    ext = [0, len(sel) * CHUNK, 0, F_ZOOM]                # held-out time (relative, s)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for a, (title, dat, cmap, vlo, vhi) in zip(axes, [
            ("Ground truth — per-freq contrast (modes)", gz[:fmax_bin], "magma", 0, 4),
            ("FSQ prediction — per-freq contrast", pz[:fmax_bin], "magma", 0, 4),
            ("GT - prediction (mode-contrast diff)", (gz - pz)[:fmax_bin], "RdBu_r", -3, 3)]):
        im = a.imshow(dat, aspect="auto", origin="lower", cmap=cmap,
                      vmin=vlo, vmax=vhi, extent=ext)
        a.set_title(title, fontsize=11); a.set_ylabel("Freq (kHz)")
        fig.colorbar(im, ax=a, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("held-out time (s, relative)")
    fig.suptitle(f"[FSQ Stage-B] shot {fig_shot} HELD-OUT ECE ch{ch} (of {len(shots)} "
                 f"trained shot(s)), 0-{F_ZOOM:.0f}kHz mode-contrast | code-acc "
                 f"{code_acc:.2f} (persist {code_persist:.2f})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outp = out_dir / f"{fig_shot}_fsq_stageB_comparison.png"
    fig.savefig(outp, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[stageB] FIGURE: {outp}", flush=True)


if __name__ == "__main__":
    main()
