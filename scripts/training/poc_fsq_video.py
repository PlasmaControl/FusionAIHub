"""POC: FSQ (VQ-style) video codec for tangtv — the video analog of the spectro
FSQ codec (poc_fsq_stageB / train_fsq_codec).

VideoTokenizer(tube-patch) -> FSQBottleneck (discrete codes) -> VideoOutputHead
(resize-conv decoder), trained with the VALIDATED adversarial recipe (3D PatchGAN
discriminator + hinge + feature-matching + R1) so the frozen decoder renders SHARP
frames from discrete codes — the same fix that broke spectro mean-collapse, aimed
here at the video checkerboard + blur. Reconstruction only (no code predictor).

Per divertor view (tangtv_lower 3ch / tangtv_upper 4ch). Env:
  MODALITY(tangtv_lower) EVAL_SHOTS(comma) FSQ_DIM(24) FSQ_L(8) AE_STEPS(4000)
  N_WINDOWS(per-shot) AE_BS(8) ADV_LAMBDA(0.5) FM_LAMBDA(10) R1_GAMMA(10) D_LR(1e-4)
  RECON_WEIGHT(1) DECODER(resize_conv) VAL_FRAC(0.15) OUT_DIR
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
from tokamak_foundation_model.e2e.tokenizers.video import VideoTokenizer
from tokamak_foundation_model.e2e.output_heads import VideoOutputHead
from tokamak_foundation_model.e2e.quantizers import FSQBottleneck

D_MODEL = 256
N_FRAMES, H, W = 3, 120, 360
PATCH = (3, 12, 12)


class VideoFSQAutoencoder(nn.Module):
    """VideoTokenizer -> FSQ bottleneck -> VideoOutputHead (resize-conv)."""

    def __init__(self, C, fsq_dim, fsq_L, decoder="resize_conv", d_model=D_MODEL,
                 resize_conv_hidden_ch=64):
        super().__init__()
        self.C = C
        self.enc = VideoTokenizer(n_channels=C, n_frames=N_FRAMES, patch_size=PATCH,
                                  d_model=d_model, spatial_size=(H, W))
        self.n_tok = self.enc.n_tokens
        self.fsq = FSQBottleneck(d_model, [fsq_L] * fsq_dim)
        self.dec = VideoOutputHead(n_channels=C, n_frames=N_FRAMES, patch_size=PATCH,
                                   d_model=d_model, spatial_size=(H, W), decoder=decoder,
                                   resize_conv_hidden_ch=resize_conv_hidden_ch)
        self.dim, self.levels = fsq_dim, fsq_L

    def forward(self, x):                                   # x (B,C,T,H,W)
        tq, codes = self.fsq(self.enc._encode(x))
        rec = self.dec(tq)                                  # (B,T,C,H,W)
        return rec.permute(0, 2, 1, 3, 4), codes            # -> (B,C,T,H,W)

    @torch.no_grad()
    def encode_codes(self, x):
        return self.fsq(self.enc._encode(x))[1]

    def decode_codes(self, codes):
        return self.dec(self.fsq.codes_to_tokens(codes)).permute(0, 2, 1, 3, 4)


class VideoDiscriminator3D(nn.Module):
    """3D PatchGAN over (B,C,T,H,W). Returns (patch_logits, [features])."""

    def __init__(self, C, base=32):
        super().__init__()

        def blk(i, o, kt):
            return nn.Sequential(
                nn.Conv3d(i, o, (kt, 4, 4), (1, 2, 2), (kt // 2, 1, 1)),
                nn.GroupNorm(min(8, o), o), nn.LeakyReLU(0.2, inplace=True))
        self.b1 = blk(C, base, 1)
        self.b2 = blk(base, base * 2, 1)
        self.b3 = blk(base * 2, base * 4, 3)
        self.out = nn.Conv3d(base * 4, 1, (1, 3, 3), 1, (0, 1, 1))

    def forward(self, x):
        f1 = self.b1(x); f2 = self.b2(f1); f3 = self.b3(f2)
        return self.out(f3), [f1, f2, f3]


def load_video_windows(shot, data_dir, stats_path, modality, n_windows):
    """Load normalized video windows (N,C,T,H,W) for one shot. Per-(window,channel)
    z-score (matches the trainer's video standardize). Only the target movie is
    loaded (movie_configs restricted → no irtv / other-divertor overhead)."""
    stats = torch.load(stats_path, weights_only=False)
    ds = TokamakMultiFileDataset(
        hdf5_paths=[Path(data_dir) / f"{shot}_processed.h5"], chunk_duration_s=0.05,
        prediction_mode=True, prediction_horizon_s=0.05, step_size_s=0.01, warmup_s=1.0,
        preprocessing_stats=stats, input_signals=[modality], target_signals=[modality])
    ds.movie_configs = [mc for mc in ds.movie_configs if mc.name == modality]
    n = len(ds)
    if n == 0:
        return torch.empty(0)
    idxs = range(n) if n_windows <= 0 else range(0, n, max(1, n // n_windows))
    out = []
    for i in idxs:
        v = ds[i]["inputs"].get(modality)
        valid = ds[i]["inputs"].get(f"{modality}_valid")
        if v is None or (valid is not None and float(torch.as_tensor(valid)) < 0.5):
            continue
        v = torch.nan_to_num(torch.as_tensor(v).float())    # (C,T,H,W)
        mu = v.mean(dim=(1, 2, 3), keepdim=True)
        sd = v.std(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        out.append((v - mu) / sd)
    return torch.stack(out) if out else torch.empty(0)


def psnr(x, r):
    mse = ((x - r) ** 2).mean().item()
    return 10 * np.log10((x.max().item() - x.min().item() + 1e-6) ** 2 / (mse + 1e-9))


def render_frozen():
    """Load a FROZEN video codec and render GT-vs-recon on REPRESENTATIVE
    high-content windows: picks the highest spatial-variance windows across the
    given shots (not the arbitrary tail), shows BOTH channels, uses a FIXED shared
    gray scale from GT percentiles (no per-frame auto-stretch that turns a flat
    frame into fake noise), and prints GT variance so data-noise vs codec-noise is
    distinguishable. Env: RENDER_CODEC=<pt> MODALITY EVAL_SHOTS_FILE/EVAL_SHOTS
    OUT_DIR N_WINDOWS NSHOW MAX_SHOTS."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modality = os.environ.get("MODALITY", "tangtv_upper")
    data_dir = os.environ.get("EVAL_DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
    stats_path = os.environ.get("EVAL_STATS", "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    codec_path = os.environ["RENDER_CODEC"]
    out_dir = Path(os.environ.get("OUT_DIR", f"eval_runs/fsq_video_{modality}_render")); out_dir.mkdir(parents=True, exist_ok=True)
    n_windows = int(os.environ.get("N_WINDOWS", "40")); nshow = int(os.environ.get("NSHOW", "4"))
    max_shots = int(os.environ.get("MAX_SHOTS", "60"))
    sf = os.environ.get("EVAL_SHOTS_FILE", "")
    if sf:
        shots = [ln.split()[0] for ln in open(sf) if ln.strip() and not ln.startswith("#")][:max_shots]
    else:
        shots = [s.strip() for s in os.environ.get("EVAL_SHOTS", "").split(",") if s.strip()]
    ck = torch.load(codec_path, map_location="cpu", weights_only=False); cfg = ck["cfg"]
    ae = VideoFSQAutoencoder(cfg["C"], cfg["fsq_dim"], cfg["fsq_L"], decoder=cfg.get("decoder", "resize_conv"))
    ae.load_state_dict(ck["ae"]); ae.eval().to(device)
    for p in ae.parameters():
        p.requires_grad_(False)
    Xs, own = [], []
    for sh in shots:
        try:
            v = load_video_windows(sh, data_dir, stats_path, modality, n_windows)
        except Exception as e:
            print(f"[vid] {sh} SKIP: {e}", flush=True); continue
        if v.numel():
            Xs.append(v); own += [sh] * v.shape[0]
    if not Xs:
        print("[vid] render: NO windows loaded", flush=True); return
    X = torch.cat(Xs, 0); own = np.array(own)
    T = X.shape[2]; fr = T // 2; C = X.shape[1]
    var = X[:, :, fr].var(dim=(1, 2, 3)).numpy()        # content = spatial variance at mid frame
    pick = np.argsort(-var)[:nshow]
    with torch.no_grad():
        REC = torch.cat([ae(X[i:i + 8].to(device))[0].cpu() for i in range(0, X.shape[0], 8)], 0)
    P = psnr(X.to(device), REC.to(device))
    gt = X.numpy(); rc = REC.numpy()
    print(f"[vid] RENDER {modality}: {X.shape[0]} windows / {len(set(own))} shots, PSNR={P:.2f} dB; "
          f"GT mid-frame var range [{var.min():.3f}, {var.max():.3f}]", flush=True)
    fig, ax = plt.subplots(nshow * C, 3, figsize=(9, 2.7 * nshow * C), squeeze=False)
    row = 0
    for w in pick:
        for c in range(C):
            g = gt[w, c, fr]; r = rc[w, c, fr]
            vlo, vhi = np.percentile(g, [2, 98])
            if vhi <= vlo:
                vhi = vlo + 1e-3
            ax[row, 0].imshow(g, cmap="gray", vmin=vlo, vmax=vhi)
            ax[row, 1].imshow(r, cmap="gray", vmin=vlo, vmax=vhi)
            ax[row, 2].imshow(g - r, cmap="RdBu_r", vmin=-(vhi - vlo) / 2, vmax=(vhi - vlo) / 2)
            ax[row, 0].set_ylabel(f"{own[w]} ch{c}\nvar={var[w]:.2f}", fontsize=8)
            for k in range(3):
                ax[row, k].set_xticks([]); ax[row, k].set_yticks([])
            print(f"[vid] w={w} shot={own[w]} ch{c}: GT var={float(g.var()):.3f} "
                  f"range[{float(g.min()):.2f},{float(g.max()):.2f}]", flush=True)
            row += 1
    ax[0, 0].set_title("GT"); ax[0, 1].set_title("FSQ recon"); ax[0, 2].set_title("diff")
    fig.suptitle(f"{modality} FROZEN codec — {nshow} highest-content windows (both ch), PSNR {P:.1f} dB")
    fig.tight_layout()
    fp = out_dir / f"render_{modality}.png"; fig.savefig(fp, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[vid] RENDER FIGURE -> {fp}\n=== VIDEO RENDER DONE ===", flush=True)


def main():
    if os.environ.get("RENDER_CODEC"):
        render_frozen(); return
    modality = os.environ.get("MODALITY", "tangtv_lower")
    data_dir = os.environ.get("EVAL_DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
    stats_path = os.environ.get("EVAL_STATS", "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    shots_file = os.environ.get("EVAL_SHOTS_FILE", "")
    if shots_file:
        shots = [ln.strip() for ln in open(shots_file)
                 if ln.strip() and not ln.startswith("#")]
    else:
        shots = [s.strip() for s in os.environ.get("EVAL_SHOTS", "200729,200226,200722,201664,201797").split(",") if s.strip()]
    fsq_dim = int(os.environ.get("FSQ_DIM", "24")); fsq_L = int(os.environ.get("FSQ_L", "8"))
    ae_steps = int(os.environ.get("AE_STEPS", "4000")); n_windows = int(os.environ.get("N_WINDOWS", "80"))
    ae_bs = int(os.environ.get("AE_BS", "8"))
    adv_lambda = float(os.environ.get("ADV_LAMBDA", "0.5")); fm_lambda = float(os.environ.get("FM_LAMBDA", "10"))
    r1_gamma = float(os.environ.get("R1_GAMMA", "10")); d_lr = float(os.environ.get("D_LR", "1e-4"))
    recon_w = float(os.environ.get("RECON_WEIGHT", "1"))
    decoder = os.environ.get("DECODER", "resize_conv"); val_frac = float(os.environ.get("VAL_FRAC", "0.15"))
    out_dir = Path(os.environ.get("OUT_DIR", f"eval_runs/fsq_video_{modality}")); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xs = []
    for sh in shots:
        try:
            v = load_video_windows(sh, data_dir, stats_path, modality, n_windows)
        except Exception as e:
            print(f"[vid] shot {sh} SKIP: {e}", flush=True); continue
        if v.numel():
            Xs.append(v); print(f"[vid] shot {sh}: {v.shape[0]} windows", flush=True)
    X = torch.cat(Xs, 0)
    N, C, T, Hh, Ww = X.shape
    nv = max(1, int(N * val_frac)); ntr = N - nv
    print(f"[vid] modality={modality} shots={len(shots)} N={N} C={C} T={T} H={Hh} W={Ww} "
          f"train={ntr} heldout={nv} decoder={decoder}", flush=True)
    Xtr = X[:ntr]

    # DECODER-ONLY fine-tune: load an existing codec, FREEZE enc+fsq (codes stay
    # BYTE-IDENTICAL so the frozen world model's predicted codes remain valid), and
    # train ONLY the decoder. AE is rebuilt from the SAVED cfg (not env) so the
    # weights load exactly. PATCH/D_MODEL are module globals VideoFSQAutoencoder
    # reads at construction, so set PATCH from cfg FIRST.
    finetune_from = os.environ.get("FINETUNE_FROM", "").strip()
    if finetune_from:
        global PATCH
        ck = torch.load(finetune_from, map_location=device, weights_only=False)
        fcfg = ck["cfg"]
        PATCH = tuple(fcfg["patch"])
        C, fsq_dim, fsq_L, decoder = fcfg["C"], fcfg["fsq_dim"], fcfg["fsq_L"], fcfg.get("decoder", "resize_conv")
        # DEC_HIDDEN>64 -> build a FRESH higher-capacity decoder from scratch and load
        # ONLY the frozen enc+fsq (codes stay byte-identical -> world model unaffected).
        # ==64 -> reuse the existing trained decoder (modest FT). The aggressive PoC.
        dec_hidden = int(os.environ.get("DEC_HIDDEN", "64"))
        ae = VideoFSQAutoencoder(C, fsq_dim, fsq_L, decoder=decoder,
                                 d_model=fcfg.get("d_model", D_MODEL),
                                 resize_conv_hidden_ch=dec_hidden).to(device)
        if dec_hidden == 64:
            ae.load_state_dict(ck["ae"]); dec_init = "reused(h=64)"
        else:
            encfsq = {k: v for k, v in ck["ae"].items()
                      if k.startswith("enc.") or k.startswith("fsq.")}
            missing, unexpected = ae.load_state_dict(encfsq, strict=False)
            bad = [m for m in missing if not m.startswith("dec.")]
            assert not bad and not list(unexpected), \
                f"enc/fsq load mismatch: missing={bad[:4]} unexpected={list(unexpected)[:4]}"
            dec_init = f"FRESH(h={dec_hidden})"
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
        print(f"[vid] DECODER-ONLY FINE-TUNE from {finetune_from}: decoder={dec_init} "
              f"n_enc_fsq_frozen={n_frozen} n_dec_trainable={n_dec} FT_STEPS={ae_steps}", flush=True)
    else:
        dec_hidden = int(os.environ.get("DEC_HIDDEN", "64")); dec_init = f"fresh_train(h={dec_hidden})"
        ae = VideoFSQAutoencoder(C, fsq_dim, fsq_L, decoder=decoder,
                                 resize_conv_hidden_ch=dec_hidden).to(device)
        optG = torch.optim.Adam(ae.parameters(), 2e-4, betas=(0.5, 0.9))
    disc = VideoDiscriminator3D(C).to(device)
    optD = torch.optim.Adam(disc.parameters(), d_lr, betas=(0.5, 0.9))
    print(f"[vid] FSQ video AE: {ae.n_tok} tokens, fsq {fsq_dim}x{fsq_L}, adv{adv_lambda} "
          f"fm{fm_lambda} R1 g{r1_gamma} D-lr{d_lr}", flush=True)

    ntr_ = Xtr.shape[0]
    for s in range(ae_steps):
        idx = torch.randint(0, ntr_, (ae_bs,)); x = Xtr[idx].to(device)
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
            print(f"  [vid] step {s+1}/{ae_steps} mae={mae.item():.4f} gadv={gadv.item():.3f} "
                  f"fm={fm.item():.3f} d={dloss.item():.3f}", flush=True)

    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    ck = out_dir / f"video_codec_{modality}.pt"
    torch.save({"ae": ae.state_dict(),
                "cfg": dict(modality=modality, C=C, fsq_dim=fsq_dim, fsq_L=fsq_L,
                            patch=PATCH, d_model=D_MODEL, decoder=decoder,
                            resize_conv_hidden_ch=dec_hidden)}, ck)
    print(f"[vid] SAVED FROZEN VIDEO CODEC -> {ck}", flush=True)

    # recon eval on held-out: PSNR + mid-frame GT/recon/diff for a few windows
    Xv = X[ntr:].to(device)
    with torch.no_grad():
        REC = torch.cat([ae(Xv[i:i + 16])[0] for i in range(0, nv, 16)], 0)
    p = psnr(Xv, REC)
    print(f"[vid] HELD-OUT recon PSNR={p:.2f} dB  (mae={ (Xv-REC).abs().mean().item():.4f})", flush=True)
    # panel: 3 held-out windows, channel 0, middle frame
    nshow = min(3, nv); fig, ax = plt.subplots(3, nshow, figsize=(4 * nshow, 9))
    ax = np.array(ax).reshape(3, nshow)
    gt = Xv.cpu().numpy(); rc = REC.cpu().numpy()
    for j in range(nshow):
        fr = T // 2
        for r_, (t, d) in enumerate([("GT", gt[j, 0, fr]), ("FSQ recon", rc[j, 0, fr]),
                                     ("diff", gt[j, 0, fr] - rc[j, 0, fr])]):
            cmap = "RdBu_r" if t == "diff" else "gray"
            im = ax[r_, j].imshow(d, cmap=cmap); ax[r_, j].set_title(f"{t} w{j}", fontsize=9)
            ax[r_, j].axis("off")
    fig.suptitle(f"FSQ VIDEO codec {modality} ch0 mid-frame — held-out PSNR {p:.1f} dB, {ae.n_tok} tok")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fp = out_dir / f"video_codec_{modality}_recon.png"
    fig.savefig(fp, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[vid] RECON FIGURE -> {fp}", flush=True)

    # BEFORE/AFTER: original codec (h=64) vs this higher-capacity decoder, IDENTICAL
    # frozen codes (drop-in; world model unaffected). The PoC proof figure.
    if finetune_from:
        orig = VideoFSQAutoencoder(C, fsq_dim, fsq_L, decoder=decoder,
                                   d_model=fcfg.get("d_model", D_MODEL),
                                   resize_conv_hidden_ch=64).to(device).eval()
        orig.load_state_dict(ck["ae"])
        with torch.no_grad():
            REC0 = torch.cat([orig(Xv[i:i + 16])[0] for i in range(0, nv, 16)], 0)
        p0 = psnr(Xv, REC0)
        print(f"[vid] COMPARE orig(h=64) PSNR={p0:.2f} dB  vs  new({dec_init}) PSNR={p:.2f} dB "
              f"(delta={p-p0:+.2f} dB)", flush=True)
        order = np.argsort(-Xv.reshape(nv, -1).var(1).cpu().numpy())[:min(4, nv)]
        rc0 = REC0.cpu().numpy(); fr = T // 2; n2 = len(order)
        figc, axc = plt.subplots(3, n2, figsize=(3.4 * n2, 9))
        axc = np.array(axc).reshape(3, n2)
        for jj, w in enumerate(order):
            vmin, vmax = np.percentile(gt[w, 0, fr], [2, 98])
            rows = [(f"GT w{w}", gt[w, 0, fr]), (f"orig h64  {p0:.1f}dB", rc0[w, 0, fr]),
                    (f"new {dec_init}  {p:.1f}dB", rc[w, 0, fr])]
            for r_, (t, d) in enumerate(rows):
                axc[r_, jj].imshow(d, cmap="gray", vmin=vmin, vmax=vmax)
                axc[r_, jj].set_title(t, fontsize=9); axc[r_, jj].axis("off")
        figc.suptitle(f"Decoder-FT PoC {modality}: original h=64 vs {dec_init}, IDENTICAL frozen "
                      f"codes (delta PSNR {p-p0:+.2f} dB)")
        figc.tight_layout(rect=(0, 0, 1, 0.95))
        fpc = out_dir / f"decoder_ft_compare_{modality}.png"
        figc.savefig(fpc, dpi=120, bbox_inches="tight"); plt.close(figc)
        print(f"[vid] COMPARE FIGURE -> {fpc}", flush=True)
    print("=== FSQ VIDEO CODEC DONE ===", flush=True)


if __name__ == "__main__":
    main()
