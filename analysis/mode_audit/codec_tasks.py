"""IGNITE spectrogram mode-loss audit — codec-side tasks 0, 1, 2 (DIAGNOSTIC ONLY).

Frozen codec + data only. No world model, no training, no model/loss/rollout edits.
Task 0: metric confound check (mode-free false positives + patch-grid alignment).
Task 1: FSQ code histogram (stratified) + derived CE class weights.
Task 2: splice faithfulness test (both directions).

Mode detection is band-restricted to the PHYSICAL mode band (5-40 kHz) and uses the
SAME prominence-above-gaussian-baseline logic as proof_resid_render.py (the metric
under audit). NO human mode labels exist -> mode-positive/-free is DETECTOR-DERIVED
(z-score of band prominence); stated as a limitation in the report.

Residual codecs (bg_subtract=True): all detection/splicing is done in RESIDUAL space
(where the mode lives); baseline is only added back for optional full-spectrogram viz.

Env: MODALITIES, SHOTS_FILE, CODEC_DIR, NWIN_PER_SHOT, OUT_DIR,
     Z_POS, Z_FREE, MODE_PIX_K.
Writes analysis/mode_audit/task{0,1,2}_<mod>.json + PDFs.
"""
import json
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
from scipy.ndimage import gaussian_filter1d
import poc_fsq_stageB as poc
from poc_fsq_stageB import load_pairs
from spectro_bg import baseline_residual
from tokamak_foundation_model.e2e.quantizers.spectro_codec import load_frozen_codec

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODS = [m.strip() for m in os.environ.get("MODALITIES", "ece,co2,bes,mhr").split(",") if m.strip()]
CODEC_DIR = os.environ.get("CODEC_DIR", "/lustre/orion/fus187/proj-shared/models/fsq_resid_p8_all")
SHOTS_FILE = os.environ.get("SHOTS_FILE", "/lustre/orion/fus187/proj-shared/models/codec_shots.txt")
DATA = os.environ.get("DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model")
STATS = os.environ.get("STATS_PATH",
                       "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
NWIN_PER_SHOT = int(os.environ.get("NWIN_PER_SHOT", "800"))   # ~per shot; 8 shots -> ~5-6k
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/analysis/mode_audit"))
OUT.mkdir(parents=True, exist_ok=True)
Z_POS = float(os.environ.get("Z_POS", "4.0"))               # mode-positive z threshold
Z_FREE = float(os.environ.get("Z_FREE", "2.0"))             # mode-free z threshold
MODE_PIX_K = float(os.environ.get("MODE_PIX_K", "3.0"))     # mode-pixel z for the 2D mask
BG_SIGMA = float(os.environ.get("BG_SIGMA", "8.0"))
FS, NFFT = 500_000.0, 1024
DF = FS / NFFT / 1e3                                         # kHz per freq bin (~0.488)
MODE_LO = int(round(5.0 / DF))                              # 5 kHz
MODE_HI = int(round(40.0 / DF))                            # 40 kHz

P75_ = P25_ = FIRE_ = None            # per-modality percentile thresholds (set in driver)
SHOTS = [s.strip() for s in Path(SHOTS_FILE).read_text().split() if s.strip()]
print(f"[cfg] mods={MODS} shots={SHOTS} band=[{MODE_LO},{MODE_HI}]bin=[5,40]kHz "
      f"split=relative-quartile(top/bottom by abs band prominence) codec={CODEC_DIR}", flush=True)


# ---------------- band-restricted mode detector (the metric under audit) ----------------
def band_prominence(x_ch):
    """x_ch (F,T) -> (prominence profile over band, peak_bin_global, peak_z)."""
    prof = np.abs(x_ch[MODE_LO:MODE_HI]).mean(1)
    pd = prof - gaussian_filter1d(prof, 6.0)
    mad = np.median(np.abs(pd - np.median(pd))) * 1.4826 + 1e-9
    f0 = int(np.argmax(pd))
    return pd, MODE_LO + f0, float(pd[f0] / mad)


def window_score(x):
    """x (C,F,T) -> (peak_prominence_abs, best_ch, peak_bin).
    Absolute band-peak prominence (residual units). Mode-positive/-free is decided
    by data-driven percentiles of THIS quantity across the modality's windows
    (top/bottom quartile) — the MAD-z is scale-free and over-fires in residual space."""
    best = (-1e9, 0, MODE_LO)
    for c in range(x.shape[0]):
        pd, f0, z = band_prominence(x[c])
        P = float(pd.max())
        if P > best[0]:
            best = (P, c, f0)
    return best


def mode_pixel_mask(x_ch):
    """x_ch (F,T) -> bool (F,T): mode pixels (band only), z above freq-smoothed baseline."""
    a = np.abs(x_ch)
    base = gaussian_filter1d(a, 6.0, axis=0)
    r = a - base
    m = np.zeros_like(a, dtype=bool)
    band = r[MODE_LO:MODE_HI]
    mad = np.median(np.abs(band - np.median(band))) * 1.4826 + 1e-9
    m[MODE_LO:MODE_HI] = band > MODE_PIX_K * mad
    return m


# ---------------- codec load / encode / decode (residual-aware) ----------------
def load_codec(mod):
    codec, cfg = load_frozen_codec(f"{CODEC_DIR}/spectro_codec_{mod}.pt", map_location="cpu")
    return codec.to(dev), cfg


def to_enc_space(X, bg):
    """X (N,C,F,T) full -> (enc_in on CPU, baseline). Residual codec sees R; else X, B=0.
    Kept on CPU (N*C*F*T is tens of GB for ece); batches move to GPU inside enc()/dec()."""
    if bg:
        _, R = baseline_residual(X, sigma=BG_SIGMA)      # baseline unused (detect in residual space)
        return R.cpu(), None
    return X.cpu(), None


def enc(codec, x):                                          # (b,C,F,T) cpu -> (b,ntok,dim) int cpu
    with torch.no_grad():
        return codec.encode_codes(x.to(dev)).cpu()


def dec(codec, codes):                                      # (b,ntok,dim)->(b,C,F,T)
    with torch.no_grad():
        return codec.decode_codes(codes.to(dev)).cpu()


def load_windows(mod, cfg, shots, nwin):
    """Multi-shot GT target windows (N,C,F,T), enc-space, + per-window (z,ch,f0)."""
    poc.PATCH_F = int(cfg.get("patch_f", 8)); poc.PATCH_T = int(cfg.get("patch_t", 16))
    C = int(cfg["C"])
    xs = []
    for sh in shots:
        try:
            _, Xt = load_pairs(sh, DATA, STATS, C, nwin, modality=mod)
            xs.append(Xt)
        except Exception as e:
            print(f"[warn] {mod} shot {sh} load failed: {e}", flush=True)
    X = torch.cat(xs, 0) if xs else torch.zeros(0)
    return X


# ============================ TASK 0 — confound check ============================
def task0(mod, codec, cfg, X, enc_in, B, bg, scores):
    npf = int(cfg["Fq"]) // int(cfg.get("patch_f", 8)); patch_f = int(cfg.get("patch_f", 8))
    free_idx = [i for i, s in enumerate(scores) if s[0] <= P25_]     # bottom-quartile = relatively mode-free
    free_idx = free_idx[:20]
    n = len(free_idx)
    fp, fp_bins = 0, []
    for i in free_idx:
        rec = dec(codec, enc(codec, enc_in[i:i + 1]))[0].numpy()   # enc-space recon
        P, ch, f0 = window_score(rec)
        if P >= FIRE_:                                              # crosses the top-quartile firing cut
            fp += 1; fp_bins.append(f0)
    # patch-grid alignment: distance of FP peak bins to nearest patch_f multiple
    dists = [min(b % patch_f, patch_f - (b % patch_f)) for b in fp_bins]
    aligned = int(sum(1 for d in dists if d <= 1))
    res = {"task": 0, "modality": mod, "n_mode_free": n, "false_positives": fp,
           "fp_rate": (fp / n if n else None), "fp_peak_bins": fp_bins,
           "patch_f": patch_f, "fp_near_patch_grid": aligned,
           "verdict": ("SUSPECT: FP rate non-negligible" if (n and fp / n > 0.1)
                       else "clean")}
    print(f"[task0] {mod}: mode-free n={n} FP={fp} rate={res['fp_rate']} "
          f"grid-aligned={aligned}/{fp} ==> {res['verdict']}", flush=True)
    return res


# ============================ TASK 1 — histogram + class weights ============================
def task1(mod, codec, cfg, X, enc_in, B, bg, scores):
    dim = int(cfg["fsq_dim"]); L = int(cfg["fsq_L"])
    pos = np.array([i for i, s in enumerate(scores) if s[0] >= P75_])   # top-quartile prominence
    neg = np.array([i for i, s in enumerate(scores) if s[0] <= P25_])   # bottom-quartile
    with torch.no_grad():
        codes = torch.cat([enc(codec, enc_in[i:i + 64]) for i in range(0, enc_in.shape[0], 64)], 0)
    codes = codes.cpu().long()                              # (N,ntok,dim)
    N, ntok, _ = codes.shape

    def joint_cov(sub):                                     # top-k tuple coverage
        if len(sub) == 0:
            return {}
        f = codes[sub].reshape(-1, dim).numpy()
        v = np.ascontiguousarray(f).view([('', f.dtype)] * dim).ravel()
        _, c = np.unique(v, return_counts=True); c = np.sort(c)[::-1]
        tot = c.sum()
        return {"tokens": int(tot), "unique": int(len(c)),
                "top1": float(c[:1].sum() / tot), "top10": float(c[:10].sum() / tot),
                "top100": float(c[:100].sum() / tot)}

    # per-dim level histogram over ALL tokens -> derived CE class weights
    flat = codes.reshape(-1, dim).numpy()
    per_dim_top1 = []
    inv_freq_w, eff_num_w = [], []                          # per-dim mean weight (for reporting)
    beta = 0.9999
    for d in range(dim):
        cnt = np.bincount(flat[:, d], minlength=L).astype(np.float64)
        p = cnt / cnt.sum()
        per_dim_top1.append(float(p.max()))
        inv = 1.0 / (cnt + 1.0); inv *= L / inv.sum()       # inverse-freq, mean-normalized
        eff = (1 - beta) / (1 - np.power(beta, np.maximum(cnt, 1))); eff *= L / eff.sum()
        inv_freq_w.append(inv); eff_num_w.append(eff)
    inv_freq_w = np.stack(inv_freq_w); eff_num_w = np.stack(eff_num_w)

    # mode-pixel vs background TOKENS within mode-positive windows
    patch_f = int(cfg.get("patch_f", 8)); patch_t = int(cfg.get("patch_t", 16))
    npf = int(cfg["Fq"]) // patch_f; npt = ntok // npf
    modetok, bgtok = 0, 0
    for i in pos[:200]:
        m = np.zeros((int(cfg["Fq"]), X.shape[-1]), bool)
        xi = (enc_in[i]).cpu().numpy()
        for c in range(xi.shape[0]):
            m |= mode_pixel_mask(xi[c])
        # token = (pf,pt); mode token if any mode pixel inside
        pm = m[:npf * patch_f].reshape(npf, patch_f, npt, patch_t).any((1, 3))  # (npf,npt)
        modetok += int(pm.sum()); bgtok += int((~pm).sum())
    res = {"task": 1, "modality": mod, "dim": dim, "L": L,
           "n_windows": int(N), "n_mode_pos": int(len(pos)), "n_mode_neg": int(len(neg)),
           "per_dim_mean_top1_level": float(np.mean(per_dim_top1)),
           "per_dim_max_top1_level": float(np.max(per_dim_top1)),
           "joint_all": joint_cov(np.arange(N)),
           "joint_mode_pos": joint_cov(pos), "joint_mode_neg": joint_cov(neg),
           "mode_pixel_tokens": modetok, "background_tokens": bgtok,
           "mode_token_fraction": (modetok / (modetok + bgtok) if (modetok + bgtok) else None),
           "class_weight_scheme": "per-dim inverse-freq AND effective-number(beta=0.9999); "
                                  "mean-normalized to L; report max/mean ratio vs the flat cw=20",
           "inv_freq_weight_max": float(inv_freq_w.max()), "inv_freq_weight_mean": float(inv_freq_w.mean()),
           "eff_num_weight_max": float(eff_num_w.max()), "eff_num_weight_mean": float(eff_num_w.mean())}
    print(f"[task1] {mod}: N={N} pos={len(pos)} neg={len(neg)} | per-dim top1={res['per_dim_mean_top1_level']:.3f} "
          f"| joint_all top1={res['joint_all'].get('top1')} | mode-tok frac={res['mode_token_fraction']} "
          f"| inv-freq w max/mean={res['inv_freq_weight_max']:.1f}/{res['inv_freq_weight_mean']:.2f} "
          f"eff-num w max={res['eff_num_weight_max']:.1f}", flush=True)
    # PDF: per-dim top-1 coverage bar + tuple-coverage
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].bar(range(dim), per_dim_top1); ax[0].axhline(1.0 / L, color="r", ls="--", label=f"uniform={1/L:.3f}")
    ax[0].set_title(f"{mod}: per-dim top-1 level coverage"); ax[0].set_xlabel("FSQ dim"); ax[0].legend(fontsize=7)
    labels = ["all", "mode+", "mode-"]; t1 = [res["joint_all"].get("top1", 0),
                                              res["joint_mode_pos"].get("top1", 0), res["joint_mode_neg"].get("top1", 0)]
    ax[1].bar(labels, t1); ax[1].set_title(f"{mod}: most-common code-tuple coverage"); ax[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(OUT / f"task1_{mod}.pdf"); plt.close(fig)
    return res


# ============================ TASK 2 — splice faithfulness (both directions) ============================
def task2(mod, codec, cfg, X, enc_in, B, bg, scores):
    patch_f = int(cfg.get("patch_f", 8)); patch_t = int(cfg.get("patch_t", 16))
    npf = int(cfg["Fq"]) // patch_f
    pos = [i for i, s in enumerate(scores) if s[0] >= P75_]
    neg = [i for i, s in enumerate(scores) if s[0] <= P25_]
    pairs = list(zip(pos[:10], neg[:10]))
    fwd_pass, inv_pass, examples = 0, 0, []
    for k, (ip, ifr) in enumerate(pairs):
        xp = enc_in[ip].cpu().numpy()
        # mode token rows (union over channels) + source peak band
        m = np.zeros((int(cfg["Fq"]), X.shape[-1]), bool)
        for c in range(xp.shape[0]):
            m |= mode_pixel_mask(xp[c])
        pm = m[:npf * patch_f].reshape(npf, patch_f, m.shape[1] // patch_t, patch_t).any((1, 3))  # (npf,npt)
        mode_pf = np.where(pm.any(1))[0]
        zc, ch, f0 = scores[ip]
        if len(mode_pf) == 0:
            continue
        cp = enc(codec, enc_in[ip:ip + 1]).cpu()
        cf = enc(codec, enc_in[ifr:ifr + 1]).cpu()
        npt = cp.shape[1] // npf
        gp = cp.reshape(1, npf, npt, -1); gf = cf.reshape(1, npf, npt, -1)
        # FORWARD: graft mode freq-patch rows from pos -> free
        chi = gf.clone(); chi[:, mode_pf] = gp[:, mode_pf]
        r_chi = dec(codec, chi.reshape(1, -1, gp.shape[-1]))[0].numpy()
        z_chi, _, f0_chi = window_score(r_chi)
        fwd_ok = (z_chi >= FIRE_) and (abs(f0_chi - f0) <= patch_f)
        fwd_pass += int(fwd_ok)
        # INVERSE: replace mode rows in pos with free (background) codes
        inv = gp.clone(); inv[:, mode_pf] = gf[:, mode_pf]
        r_inv = dec(codec, inv.reshape(1, -1, gp.shape[-1]))[0].numpy()
        z_inv, _, _ = window_score(r_inv)
        inv_ok = z_inv < FIRE_
        inv_pass += int(inv_ok)
        if k < 3:
            examples.append((mod, k, ip, ifr, ch, f0, z_chi, f0_chi, fwd_ok, z_inv, inv_ok,
                             dec(codec, cp)[0, ch].numpy(), dec(codec, cf)[0, ch].numpy(),
                             r_chi[ch], r_inv[ch]))
    npair = len(pairs)
    res = {"task": 2, "modality": mod, "n_pairs": npair,
           "forward_pass_rate": (fwd_pass / npair if npair else None),
           "inverse_pass_rate": (inv_pass / npair if npair else None),
           "fire_threshold_P75": FIRE_, "P25": P25_, "detector_band_khz": [5, 40],
           "split": "relative top/bottom quartile of absolute band prominence (no human labels)",
           "verdict": ("UNFAITHFUL (<80%)" if (npair and min(fwd_pass, inv_pass) / npair < 0.8)
                       else ("FAITHFUL" if npair else "no pairs"))}
    print(f"[task2] {mod}: pairs={npair} forward_pass={res['forward_pass_rate']} "
          f"inverse_pass={res['inverse_pass_rate']} ==> {res['verdict']}", flush=True)
    # chimera example PDFs
    freqs = np.arange(int(cfg["Fq"])) * DF
    fmax = min(int(cfg["Fq"]), int(60 / DF))
    for (mm, k, ip, ifr, ch, f0, zc, f0c, ok, zi, iok, rp, rf, rchi, rinv) in examples:
        fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
        for a, (t, arr) in zip(ax, [(f"mode+ (w{ip})", rp), (f"free (w{ifr})", rf),
                                    (f"free+splice z={zc:.1f} {'PASS' if ok else 'fail'}", rchi),
                                    (f"pos-erased z={zi:.1f} {'PASS' if iok else 'fail'}", rinv)]):
            a.imshow(np.abs(arr[:fmax]), origin="lower", aspect="auto", extent=[0, arr.shape[-1], 0, freqs[fmax]])
            a.axhline(freqs[f0], color="cyan", lw=0.6, ls="--"); a.set_title(t, fontsize=8); a.set_ylabel("kHz")
        fig.suptitle(f"{mm.upper()} splice pair {k} ch{ch}", fontsize=10); fig.tight_layout()
        fig.savefig(OUT / f"task2_{mm}_pair{k}.pdf"); plt.close(fig)
    return res


# ============================ driver ============================
all_res = {}
for mod in MODS:
    print(f"\n===================== {mod} =====================", flush=True)
    try:
        codec, cfg = load_codec(mod)
        bg = bool(cfg.get("bg_subtract", False))
        X = load_windows(mod, cfg, SHOTS, NWIN_PER_SHOT)
        if X.shape[0] == 0:
            print(f"[warn] {mod}: no windows", flush=True); continue
        enc_in, B = to_enc_space(X, bg)
        scores = [window_score(enc_in[i].cpu().numpy()) for i in range(enc_in.shape[0])]
        Parr = np.array([s[0] for s in scores])
        P75_ = float(np.percentile(Parr, 75)); P25_ = float(np.percentile(Parr, 25)); FIRE_ = P75_
        n_pos = int((Parr >= P75_).sum()); n_free = int((Parr <= P25_).sum())
        print(f"[{mod}] windows={X.shape[0]} bg={bg} band-prominence P: p50={np.median(Parr):.3f} "
              f"P25={P25_:.3f} P75={P75_:.3f} | mode-pos(top-q)={n_pos} mode-free(bot-q)={n_free} "
              f"(relative quartile split; FIRE=P75)", flush=True)
        r0 = task0(mod, codec, cfg, X, enc_in, B, bg, scores)
        r1 = task1(mod, codec, cfg, X, enc_in, B, bg, scores)
        r2 = task2(mod, codec, cfg, X, enc_in, B, bg, scores)
        all_res[mod] = {"task0": r0, "task1": r1, "task2": r2,
                        "n_windows": int(X.shape[0]), "bg_subtract": bg}
        json.dump(all_res[mod], open(OUT / f"tasks012_{mod}.json", "w"), indent=2)
    except Exception as e:
        import traceback
        print(f"[WARN] {mod} FAILED: {e}", flush=True); traceback.print_exc()

json.dump(all_res, open(OUT / "tasks012_all.json", "w"), indent=2)
print("\n[codec_tasks] done", flush=True)
