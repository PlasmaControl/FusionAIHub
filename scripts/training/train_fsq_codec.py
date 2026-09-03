"""Phase 1a: pre-train + FREEZE the adversarial FSQ spectrogram codec (per modality).

Trains an FSQ-AE (encoder=SpectrogramTokenizer patch -> FSQBottleneck -> decoder=
SpectrogramOutputHead) with the VALIDATED VQ-GAN recipe (PatchGAN discriminator +
hinge + feature-matching + mode-weighted reconstruction + R1/low-D-lr rebalance) —
RECONSTRUCTION ONLY, no code predictor. Saves a frozen `spectro_codec_<mod>.pt` that
the e2e model (Phase 1b) loads to (a) encode target windows -> code CE targets and
(b) decode predicted codes -> sharp mode-bearing spectrograms.

Reuses the components validated in poc_fsq_stageB.py. Env:
  MODALITY(ece) EVAL_SHOTS(comma) FSQ_DIM(24) FSQ_L(8) PATCH_F(64) PATCH_T(32)
  AE_STEPS(6000) N_WINDOWS(per-shot cap) N_CHANNELS(all) AE_BS SPEC_RECON_WEIGHT(20)
  ADV_LAMBDA(0.5) FM_LAMBDA(10) R1_GAMMA(10) D_LR(1e-4) VAL_FRAC(0.2) OUT_DIR
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import poc_fsq_stageB as poc
from poc_fsq_stageB import FSQAutoencoder, SpectroDiscriminator, load_pairs, _hard
from train_e2e_stage1 import _SPEC_STRUCT_K


def main():
    modality = os.environ.get("MODALITY", "ece")
    data_dir = os.environ.get("EVAL_DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
    stats_path = os.environ.get("EVAL_STATS", "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    shots_file = os.environ.get("EVAL_SHOTS_FILE", "")
    if shots_file:
        shots = [ln.strip() for ln in open(shots_file)
                 if ln.strip() and not ln.startswith("#")]
    else:
        shots = [s.strip() for s in os.environ.get("EVAL_SHOTS", "200729").split(",") if s.strip()]
    fsq_dim = int(os.environ.get("FSQ_DIM", "24")); fsq_L = int(os.environ.get("FSQ_L", "8"))
    poc.PATCH_F = int(os.environ.get("PATCH_F", "64")); poc.PATCH_T = int(os.environ.get("PATCH_T", "32"))
    # Codec-internal encoder/decoder width. Independent of the BACKBONE d_model —
    # only n_tok (=24) is the production-VRAM-relevant interface, so raising this
    # (and fsq_dim/fsq_L) buys codec capacity/detail at NO production cost. Saved
    # in cfg so load_frozen_codec rebuilds the matching width.
    poc.D_MODEL = int(os.environ.get("D_MODEL", str(poc.D_MODEL)))
    ae_steps = int(os.environ.get("AE_STEPS", "6000"))
    n_windows = int(os.environ.get("N_WINDOWS", "250")); n_channels = int(os.environ.get("N_CHANNELS", "64"))
    drop_pad_std = float(os.environ.get("DROP_PAD_STD", "0.4")); ae_bs = int(os.environ.get("AE_BS", "32"))
    recon_weight = float(os.environ.get("SPEC_RECON_WEIGHT", "20"))
    adv_lambda = float(os.environ.get("ADV_LAMBDA", "0.5")); fm_lambda = float(os.environ.get("FM_LAMBDA", "10"))
    r1_gamma = float(os.environ.get("R1_GAMMA", "10")); d_lr = float(os.environ.get("D_LR", "1e-4"))
    val_frac = float(os.environ.get("VAL_FRAC", "0.2"))
    out_dir = Path(os.environ.get("OUT_DIR", f"eval_runs/fsq_codec_{modality}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = _SPEC_STRUCT_K.get(modality, 2.0)

    # Deterministic shuffle so the tail held-out split is representative (the shot
    # list is often mode-ranked; without this the held-out would be only the
    # lowest-density shots).
    import random as _random
    _random.Random(0).shuffle(shots)
    # multi-shot recon dataset: BOTH input & target windows are reconstruction
    # data. Kept on CPU (host RAM) and streamed batch-by-batch to the GPU — the
    # production codec set is hundreds of shots (10k+ windows), which does NOT fit
    # in GCD memory (the old all-to-device path OOMs past ~a few thousand windows).
    Xs = []
    for si, sh in enumerate(shots):
        try:
            xi, xt = load_pairs(sh, data_dir, stats_path, n_channels, n_windows,
                                drop_pad_std, modality)
        except Exception as e:
            print(f"[codec] shot {sh} SKIP: {e}", flush=True); continue
        if xi.shape[0] == 0:
            continue
        Xs += [xi, xt]
        if (si + 1) % 50 == 0:
            print(f"[codec] loaded {si+1}/{len(shots)} shots", flush=True)
    X = torch.cat(Xs, 0)                                  # CPU
    N, C, Fq, Tq = X.shape
    nv = max(1, int(N * val_frac)); ntr = N - nv
    # BG_SUBTRACT=1: train the codec on the background-subtracted RESIDUAL R = X - B
    # (B = smooth per-freq envelope). Modes dominate R, so MAE / code budget prioritize
    # them instead of the bright low-freq envelope. Recombined to original space at eval
    # (X = B + R). Self-contained (scripts/training/spectro_bg.py); NO external import.
    bg_subtract = bool(os.environ.get("BG_SUBTRACT", "").strip())
    X_orig, B_full = X, None
    if bg_subtract:
        from spectro_bg import baseline_residual
        bg_sigma = float(os.environ.get("BG_SIGMA", "8.0"))
        B_full, R = baseline_residual(X, sigma=bg_sigma)
        X = R.to(X_orig.dtype)
        print(f"[codec] BG_SUBTRACT on: residual space sigma={bg_sigma} "
              f"(R std={float(X.std()):.3f} vs S std={float(X_orig.std()):.3f})", flush=True)
    # SMOOTH_FRAMES=N: temporal moving-average of the (residual) magnitude across N
    # STFT frames before encoding — the "encode statistics, not realizations" fix.
    # Coherent ridges survive frame-averaging; STFT-phase/realization speckle (which a
    # 0.5 ms shift scrambles) dies -> codes become shift-stable/predictable. Applied to
    # BOTH the AE input and its reconstruction target (they are the same tensor X here),
    # and recorded in cfg so the gate/eval/world-model smooth identically. N<=1 = no-op.
    smooth_frames = int(os.environ.get("SMOOTH_FRAMES", "0"))
    if smooth_frames > 1:
        from spectro_bg import smooth_time_mag
        X = smooth_time_mag(X, smooth_frames).to(X_orig.dtype)
        print(f"[codec] SMOOTH_FRAMES={smooth_frames} time moving-avg on "
              f"{'residual' if bg_subtract else 'magnitude'} (std now {float(X.std()):.3f})", flush=True)
    print(f"[codec] modality={modality} shots={len(shots)} N={N} C={C} F={Fq} T={Tq} "
          f"train={ntr} heldout={nv} bg_subtract={bg_subtract} smooth_frames={smooth_frames} "
          f"(data on CPU, streamed to GPU)", flush=True)
    Xtr = X[:ntr]                                         # CPU

    # DECODER-ONLY fine-tune: load an existing codec, FREEZE enc+fsq (codes stay
    # BYTE-IDENTICAL so the frozen world model's predicted codes remain valid), and
    # train ONLY the decoder. AE is rebuilt from the SAVED cfg (not env) so the
    # weights load exactly; PATCH_F/PATCH_T/D_MODEL are module globals of poc that
    # FSQAutoencoder reads at construction, so set them from cfg FIRST.
    finetune_from = os.environ.get("FINETUNE_FROM", "").strip()
    if finetune_from:
        ck = torch.load(finetune_from, map_location=device, weights_only=False)
        fcfg = ck["cfg"]
        poc.PATCH_F = int(fcfg["patch_f"]); poc.PATCH_T = int(fcfg["patch_t"])
        poc.D_MODEL = int(fcfg["d_model"])
        ae = FSQAutoencoder(fcfg["C"], fcfg["Fq"], fcfg["Tq"], fcfg["fsq_dim"], fcfg["fsq_L"],
                            per_channel=fcfg.get("per_channel", False)).to(device)
        ae.load_state_dict(ck["ae"])
        # keep the saved-cfg values so the re-saved codec cfg matches the loaded model
        C, Fq, Tq, fsq_dim, fsq_L = fcfg["C"], fcfg["Fq"], fcfg["Tq"], fcfg["fsq_dim"], fcfg["fsq_L"]
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
        print(f"[codec] DECODER-ONLY FINE-TUNE from {finetune_from}: "
              f"n_enc_fsq_frozen={n_frozen} n_dec_trainable={n_dec} FT_STEPS={ae_steps}", flush=True)
    else:
        ae = FSQAutoencoder(C, Fq, Tq, fsq_dim, fsq_L, per_channel=False).to(device)
        optG = torch.optim.Adam(ae.parameters(), 2e-4, betas=(0.5, 0.9))
    disc = SpectroDiscriminator(C).to(device)
    optD = torch.optim.Adam(disc.parameters(), d_lr, betas=(0.5, 0.9))
    print(f"[codec] adversarial FSQ-AE: {ae.n_tok} tokens, adv{adv_lambda} fm{fm_lambda} "
          f"R1 g{r1_gamma} D-lr{d_lr} recon-wt{recon_weight}", flush=True)

    def recon_mae(x, rec):
        d = (rec - x).abs()
        if recon_weight > 1:
            with torch.no_grad():
                wm = 1.0 + (recon_weight - 1.0) * _hard(x, k)
            return (d * wm).sum() / wm.sum()
        return d.mean()

    ntr_ = Xtr.shape[0]
    for s in range(ae_steps):
        idx = torch.randint(0, ntr_, (ae_bs,)); x = Xtr[idx].to(device, non_blocking=True)
        with torch.no_grad():
            rec, _ = ae(x)
        xr = x.detach().requires_grad_(True)
        dr, _ = disc(xr); df, _ = disc(rec)
        dloss = F.relu(1 - dr).mean() + F.relu(1 + df).mean()
        if r1_gamma > 0:
            g = torch.autograd.grad(dr.sum(), xr, create_graph=True)[0]
            dloss = dloss + 0.5 * r1_gamma * g.pow(2).flatten(1).mean(1).mean()
        optD.zero_grad(set_to_none=True); dloss.backward(); optD.step()
        rec, _ = ae(x); mae = recon_mae(x, rec)
        dfg, ff = disc(rec)
        with torch.no_grad():
            _, fr = disc(x)
        gadv = -dfg.mean(); fm = sum((a - b).abs().mean() for a, b in zip(ff, fr)) / len(ff)
        gloss = mae + adv_lambda * gadv + fm_lambda * fm
        optG.zero_grad(set_to_none=True); gloss.backward(); optG.step()
        if (s + 1) % 500 == 0 or s == 0:
            print(f"  [codec] step {s+1}/{ae_steps} mae={mae.item():.4f} gadv={gadv.item():.3f} "
                  f"fm={fm.item():.3f} d={dloss.item():.3f}", flush=True)

    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    codec_path = out_dir / f"spectro_codec_{modality}.pt"
    torch.save({"ae": ae.state_dict(),
                "cfg": dict(modality=modality, C=C, Fq=Fq, Tq=Tq, fsq_dim=fsq_dim, fsq_L=fsq_L,
                            patch_f=poc.PATCH_F, patch_t=poc.PATCH_T, d_model=poc.D_MODEL,
                            per_channel=False, bg_subtract=bg_subtract,
                            smooth_frames=smooth_frames)},
               codec_path)
    print(f"[codec] SAVED FROZEN CODEC -> {codec_path}", flush=True)

    # ---- recon-quality figure on held-out (GT vs AE-recon, per-freq contrast) ----
    Xv = X[ntr:]                                          # CPU held-out
    with torch.no_grad():
        rec_parts = [ae(Xv[i:i + 64].to(device))[0].cpu() for i in range(0, nv, 64)]
        REC = torch.cat(rec_parts, 0)                     # (nv, C, F, T) on CPU
    # CODEC-CEILING capture (raw prominence ratio, mode band 5-40 kHz): the best
    # the codec can do — decode(encode(GT)). Measures whether the codec preserves
    # sharp MODE AMPLITUDE (the metric the world-model ranker uses; the pfz figure
    # below hides amplitude via per-freq contrast). Computed in RESIDUAL space
    # (matches the world-model target) BEFORE the baseline is added back.
    try:
        from scipy.ndimage import gaussian_filter1d as _gf0
        khz_per_bin = 500.0 / 1024.0
        _lo = int(5.0 / khz_per_bin); _hi = int(40.0 / khz_per_bin)
        _gt = Xv.cpu().numpy(); _rc = REC.cpu().numpy()      # (nv,C,F,T) residual
        _chp = int(np.abs(_gt[:, :, _lo:_hi]).sum(axis=(0, 2, 3)).argmax())
        caps = []
        for w in range(_gt.shape[0]):
            gp = np.abs(_gt[w, _chp, _lo:_hi]).mean(1); rp = np.abs(_rc[w, _chp, _lo:_hi]).mean(1)
            gd = gp - _gf0(gp, 6.0); rd = rp - _gf0(rp, 6.0)
            f0 = int(np.argmax(gd))
            if gd[f0] > 1e-6:
                caps.append(rd[f0] / gd[f0])
        print(f"[codec] CEILING CAPTURE (raw prominence, ch{_chp}, n={len(caps)}) = "
              f"{float(np.nanmedian(caps)):.3f}   (target >0.5; old ece codec ~0.09)", flush=True)
    except Exception as _e:
        print(f"[codec] ceiling-capture measure skipped: {_e}", flush=True)
    if bg_subtract:
        REC = REC + B_full[ntr:]                          # recombine to original space: S = B + R_rec
    st = torch.load(stats_path, weights_only=False)
    lm = np.asarray(st[modality]["log"]["mean"]); lsd = np.asarray(st[modality]["log"]["std"])
    def stitch(X4d, ch):
        step = max(1, X4d.shape[0] // 40)                 # cap ~40 windows for the view
        arr = X4d[::step, ch].cpu().numpy() * max(float(lsd[ch]), 1e-3) + float(lm[ch])
        n_w, Fh, Th = arr.shape
        return arr.transpose(1, 0, 2).reshape(Fh, n_w * Th)
    Gt = X_orig[ntr:]; ch = int(_hard(Gt, k).sum(dim=(0, 2, 3)).argmax())
    def pfz(a):
        m = a.mean(1, keepdims=True); ssd = a.std(1, keepdims=True) + 1e-6
        return np.clip((a - m) / ssd, 0, 4)
    g = pfz(stitch(Gt, ch)); r = pfz(stitch(REC, ch))
    fmax = int(60 / (500.0 / 1024.0))
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for a_, (t, d, cm, lo, hi) in zip(ax, [("GT held-out (per-freq contrast)", g[:fmax], "magma", 0, 4),
                                           ("FSQ-AE reconstruction", r[:fmax], "magma", 0, 4),
                                           ("GT - recon", (g - r)[:fmax], "RdBu_r", -3, 3)]):
        im = a_.imshow(d, aspect="auto", origin="lower", cmap=cm, vmin=lo, vmax=hi)
        a_.set_title(t, fontsize=11); a_.set_ylabel("freq bin"); fig.colorbar(im, ax=a_, fraction=0.02)
    fig.suptitle(f"[FSQ codec 1a] {modality} ch{ch} recon (frozen) | {len(shots)} shots, "
                 f"{ae.n_tok} tokens", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out_dir / f"codec_{modality}_recon.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[codec] RECON FIGURE -> {p}", flush=True)


if __name__ == "__main__":
    main()
