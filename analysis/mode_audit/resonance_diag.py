"""RESONANCE DIAGNOSTIC — T1 (mode energy) vs T2 (roughness/realization bits).

The ece SpectrogramTokenizer.proj (patch Conv2d) turns SOME codec-decoded
feedback states into a resonant proj-absmax (~1900+ → NaN under bf16) while
others at the SAME input magnitude stay in-band (~50-210), and real INPUT
windows never resonate. Two failed fixes (per-(C,F) time-moment norm; ±1
lattice-extreme clamp) ruled out the extreme codes.

DISCRIMINATING MEASUREMENT (per mode-active ece window, side by side):
  GT-path   : codes = head.encode_target(window) → dec = head.decode(codes)
              → tok = tokenizer._encode(dec)  → record proj/out absmax.
  PRED-path : model.forward(window) → ece backbone slice
              → logits = head.code_logits(slice) → codes = argmax
              → dec = head.decode(codes) → tok = tokenizer._encode(dec)
              → record proj/out absmax.
Both decodes contain the SAME window's mode ridge.
  * GT resonates while PRED stays in-band  → T2 (roughness/realization bits):
    resonance is NOT the mode energy (both carry the ridge), it is the GT code
    realization the model never predicts.
  * BOTH resonate                          → T1 (the mode energy itself).

Also: (a) mode-band (5-40 kHz) proj-output energy concentration for resonating
vs non-resonating decodes; (b) radially-averaged 2D-FFT magnitude of resonating
decode patches vs non-resonating decode patches vs natural input-window patches
→ eval_runs/resonance_diag/spatial_spectrum.png.

Uses the g3fix β=6 ckpt via eval_e2e_animation_tokamak.load_model, and the same
one-batch data load as gate4_kprobe (shot 200729, EXTRA_DATA_DIR). READ-ONLY on
all model dirs — writes only to eval_runs/resonance_diag.

Env: CKPT(argv1), SHOT(200729), BATCH(16), MAX_WIN(64), RES_THRESH(600),
     OUT_DIR, CACHE_DIR, EXTRA_DATA_DIR.
"""
import os, sys, json
from pathlib import Path

FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training", f"{FMH}/analysis/mode_audit"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from eval_e2e_animation_tokamak import load_model
from train_e2e_stage1 import build_datasets, _core, forward_batch
from tokamak_foundation_model.data.data_loader import collate_fn
from dist_gate import MODE_LO, MODE_HI, DF   # 5-40 kHz band in 512-bin STFT index

CKPT = Path(sys.argv[1] if len(sys.argv) > 1
            else "/lustre/orion/fus187/proj-shared/models/e2e_g3fix_anneal/e2e_stage1_beta6.0_step3000.pt")
SHOT = os.environ.get("SHOT", "200729")
BATCH = int(os.environ.get("BATCH", "16"))
MAX_WIN = int(os.environ.get("MAX_WIN", "64"))
RES_THRESH = float(os.environ.get("RES_THRESH", "600"))   # proj/out-absmax "resonant" cut (natural band ~50-210)
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/eval_runs/resonance_diag")); OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, ckpt = load_model(CKPT, device); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
a = ckpt["args"]; core = _core(model)
diag_names = [d["name"] for d in ckpt["diagnostics"]]; act_names = [c["name"] for c in ckpt["actuators"]]
data_dir = Path(a["data_dir"]); extra = os.environ.get("EXTRA_DATA_DIR")
stats = torch.load(a["stats_path"], weights_only=False)
chunk = a["chunk_duration_s"]; horizon = chunk   # single window (no rollout needed here)

head = core.diag_heads["ece"]
tok = core.diag_tokenizers["ece"]
patch_f = tok.patch_f                              # 8  → n_patches_f = 512/8 = 64
n_pf, n_pt = tok.n_patches_f, tok.n_patches_t       # (64, 6)
# mode band (5-40 kHz) expressed in proj-output FREQ-TOKEN index (each token = patch_f STFT bins)
band_tok_lo, band_tok_hi = MODE_LO // patch_f, (MODE_HI + patch_f - 1) // patch_f
print(f"[res] ckpt={CKPT.name} SHOT={SHOT} BATCH={BATCH} MAX_WIN={MAX_WIN} thresh={RES_THRESH}", flush=True)
print(f"[res] ece codec bg_subtract={getattr(head,'bg_subtract',None)} sigma={getattr(head,'bg_sigma',None)} "
      f"| tok patch_f={patch_f} n_pf={n_pf} n_pt={n_pt} | mode-band STFT-bins[{MODE_LO}:{MODE_HI}] "
      f"proj-freq-tok[{band_tok_lo}:{band_tok_hi}] (DF={DF:.4f} kHz/bin)", flush=True)


def resolve(sh):
    f = data_dir / f"{sh}_processed.h5"
    if f.exists(): return f
    if extra and (Path(extra) / f"{sh}_processed.h5").exists(): return Path(extra) / f"{sh}_processed.h5"
    return None


f = resolve(SHOT); assert f is not None, f"{SHOT} not found under {data_dir} or {extra}"
cache = Path(os.environ.get("CACHE_DIR", f"{FMH}/eval_runs/resonance_diag/cache")); cache.mkdir(parents=True, exist_ok=True)
_, va = build_datasets(data_dir, [f], [f], stats, chunk, horizon, a["step_size_s"], a["warmup_s"],
                       diag_names, act_names, cache)
loader = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=2, collate_fn=collate_fn, drop_last=False)


# ---- proj/out instrumentation: run tokenizer._encode but split out the pre-add
# proj-absmax (the resonance lives in proj) AND the whole-tokenizer out-absmax, plus
# the pre-flatten proj feature map for the mode-band energy concentration. ----
def encode_probe(dec):
    """dec (B,C,F,T) residual spectrogram → (proj_am (B,), out_am (B,), proj_map (B,d,n_pf,n_pt))."""
    x = dec[..., : tok.trunc_t]
    if getattr(tok, "enable_freq_stem", False):
        import torch.nn.functional as _F
        h = x.transpose(2, 3)
        h = tok.fs_lin2(_F.gelu(tok.fs_lin1(h)))
        x = x + h.transpose(2, 3)
    pmap = tok.proj(x)                                   # (B, d_model, n_pf, n_pt)
    proj_am = pmap.flatten(1).abs().amax(1)              # (B,)
    t = pmap.flatten(2).transpose(1, 2)                  # (B, n_tok, d_model)
    t = t + tok.spatial_pe + tok.modality_embed
    for blk in tok.refine:
        t = t + blk(t)
    out_am = t.flatten(1).abs().amax(1)                  # (B,)
    return proj_am, out_am, pmap


def argmax_loc(pmap):
    """(B,d,n_pf,n_pt) → per-window (channel, freq_tok, time_tok) of the |proj| max."""
    B = pmap.shape[0]
    flat = pmap.abs().reshape(B, -1)
    idx = flat.argmax(1)                                 # (B,)
    d, nf, nt = pmap.shape[1], pmap.shape[2], pmap.shape[3]
    ch = (idx // (nf * nt)).cpu().numpy()
    rem = idx % (nf * nt)
    ft = (rem // nt).cpu().numpy()
    tt = (rem % nt).cpu().numpy()
    return ch, ft, tt                                    # each (B,)


def band_conc(pmap):
    """(B,d,n_pf,n_pt) proj feature map → fraction of |proj| energy in the 5-40 kHz freq-token band (B,)."""
    e = pmap.abs().sum(dim=(1, 3))                        # (B, n_pf) energy per freq-token
    band = e[:, band_tok_lo:band_tok_hi].sum(1)
    return (band / e.sum(1).clamp_min(1e-9))              # (B,)


def radial_spectrum(patch):
    """patch (C,F,T) → radially-averaged 2D-FFT magnitude over (F,T), channel-mean."""
    x = patch.detach().float().cpu().numpy()
    F_, T_ = x.shape[-2], x.shape[-1]
    mag = np.abs(np.fft.fftshift(np.fft.fft2(x, axes=(-2, -1)), axes=(-2, -1)))  # (C,F,T)
    mag = mag.mean(0)                                     # (F,T) channel-mean
    cy, cx = F_ // 2, T_ // 2
    yy, xx = np.ogrid[:F_, :T_]
    r = np.sqrt(((yy - cy) / max(cy, 1)) ** 2 + ((xx - cx) / max(cx, 1)) ** 2)  # normalized radius [0,~1.4]
    nb = 32
    rb = np.clip((r / r.max() * (nb - 1)).astype(int), 0, nb - 1)
    prof = np.array([mag[rb == b].mean() if np.any(rb == b) else 0.0 for b in range(nb)])
    return prof


rows = []                       # per-window records
res_maps, nonres_maps = [], []  # proj feature maps for band-conc split
res_decode, nonres_decode, input_windows = [], [], []   # decode/input patches for spatial spectrum
seen = 0
with torch.no_grad():
    for batch in loader:
        if seen >= MAX_WIN:
            break
        # forward_batch builds diag_inputs (residual-split for a bg_subtract codec),
        # runs the model, and hands back the backbone token slices. diag_inputs["ece"]
        # is the SAME residual space the codec's encode_target expects (matches the
        # SpectrogramCodeHead CE branch, which calls encode_target(diag_inputs[name])
        # under --spec_autoencode). We compare GT vs predicted codes for the ridge of
        # the SAME (input) window.
        preds, diag_inputs, tgts, masks, slices = forward_batch(model, batch, device)
        raw_r = diag_inputs["ece"]                                # (B,C,F,T) residual input window

        # ---- mode-active gate: band-prominence on the residual input ----
        band = raw_r[:, :, MODE_LO:MODE_HI].abs().mean(-1)        # (B,C,band_bins)
        prom = (band.amax(-1) - band.mean(-1)).amax(-1)          # (B,) best-channel band prominence
        thr = torch.quantile(prom, 0.5)
        active = prom > thr                                       # mode-active windows

        # ---- GT-path: encode_target(input window residual) → decode → tokenize ----
        codes_gt = head.encode_target(raw_r)
        dec_gt = head.decode(codes_gt)                            # (B,C,F,T) residual
        pgt, ogt, mgt = encode_probe(dec_gt)
        gch, gft, gtt = argmax_loc(mgt)

        # ---- PRED-path: ece backbone slice → argmax codes → decode → tokenize ----
        sl = slices["ece"]                                        # backbone token slice for ece (B,n_tok,d)
        logits = head.code_logits(sl)
        codes_pred = head.sample_codes(logits, hard=True)         # argmax
        dec_pred = head.decode(codes_pred)                        # (B,C,F,T) residual
        ppr, opr, mpr = encode_probe(dec_pred)
        pch, pft, ptt = argmax_loc(mpr)

        # ---- PRED-SAMPLE control: a DISTINCT predicted realization (temperature 1.0
        # multinomial) of the SAME ridge. If argmax collapses to GT, this forces a
        # different code pattern → a clean T1/T2 read on whether the resonance follows
        # the ridge or the specific realization. ----
        codes_samp = head.sample_codes(logits, temperature=1.0, hard=False)
        dec_samp = head.decode(codes_samp)
        psa, osa, msa = encode_probe(dec_samp)
        samp_dev = (msa - msa.mean(0, keepdim=True)).flatten(1).abs().amax(1)
        samp_agree = (codes_samp == codes_gt).float().reshape(codes_gt.shape[0], -1).mean(1)

        # ---- CONTROL 1: constant (all-zero residual) decode — NO ridge, NO content.
        # If this ALSO resonates, the proj-absmax is a content-independent DC/bias
        # saturation (T2-flavored: not the ridge). ----
        pz, oz, mz = encode_probe(torch.zeros_like(dec_gt))

        # ---- CONTROL 2: natural raw residual INPUT window (never resonates in prod). ----
        pin, oin, min_ = encode_probe(raw_r)

        # ---- DECOMPOSE: proj-absmax of the per-window DEVIATION from the batch mean.
        # If the max is carried by the batch-mean (content-independent) component, the
        # deviation absmax is small → the ridge is NOT what resonates. If the deviation
        # itself resonates, the ridge IS the driver. ----
        mgt_dev = mgt - mgt.mean(0, keepdim=True)
        gt_devproj = mgt_dev.flatten(1).abs().amax(1)             # (B,)
        mpr_dev = mpr - mpr.mean(0, keepdim=True)
        pr_devproj = mpr_dev.flatten(1).abs().amax(1)

        # ---- CODE-COLLAPSE check: fraction of pred codes equal to GT codes (per window) ----
        code_agree = (codes_pred == codes_gt).float().reshape(codes_gt.shape[0], -1).mean(1)  # (B,)

        cgt = band_conc(mgt); cpr = band_conc(mpr)
        for i in range(raw_r.shape[0]):
            if not bool(active[i]):
                continue
            rows.append(dict(
                gt_proj=float(pgt[i]), gt_out=float(ogt[i]),
                pred_proj=float(ppr[i]), pred_out=float(opr[i]),
                zero_proj=float(pz[i]), zero_out=float(oz[i]),
                input_proj=float(pin[i]), input_out=float(oin[i]),
                samp_proj=float(psa[i]), samp_devproj=float(samp_dev[i]), samp_agree=float(samp_agree[i]),
                gt_devproj=float(gt_devproj[i]), pred_devproj=float(pr_devproj[i]),
                gt_argmax_ftok=int(gft[i]), gt_argmax_ttok=int(gtt[i]), gt_argmax_ch=int(gch[i]),
                pred_argmax_ftok=int(pft[i]),
                code_agree=float(code_agree[i]),
                gt_bandconc=float(cgt[i]), pred_bandconc=float(cpr[i]),
                prom=float(prom[i])))
            # collect maps/patches for the aggregate splits (use GT-path decode: it is the one that resonates)
            if float(pgt[i]) >= RES_THRESH:
                res_maps.append(mgt[i]); res_decode.append(dec_gt[i])
            else:
                nonres_maps.append(mgt[i]); nonres_decode.append(dec_gt[i])
            input_windows.append(raw_r[i])
            seen += 1
            if seen >= MAX_WIN:
                break

print(f"[res] mode-active windows collected: {len(rows)}", flush=True)

# ---- CONTROL / DECOMPOSITION SUMMARY (the disambiguator for the pinned-max artifact) ----
_zp = np.array([r["zero_proj"] for r in rows]); _ip = np.array([r["input_proj"] for r in rows])
_gd = np.array([r["gt_devproj"] for r in rows]); _pd = np.array([r["pred_devproj"] for r in rows])
_ca = np.array([r["code_agree"] for r in rows])
_gftok = np.array([r["gt_argmax_ftok"] for r in rows]); _gttok = np.array([r["gt_argmax_ttok"] for r in rows])
print("\n[res] ===== CONTROLS & DECOMPOSITION =====", flush=True)
print(f"[res] CONTROL zero-decode  proj-absmax: mean={_zp.mean():.2f} max={_zp.max():.2f}  "
      f"(if ~resonant with NO ridge → content-independent DC/bias saturation, NOT the ridge)", flush=True)
print(f"[res] CONTROL raw-input    proj-absmax: mean={_ip.mean():.2f} max={_ip.max():.2f}  "
      f"(the never-resonates production input, as a baseline)", flush=True)
print(f"[res] GT proj-DEVIATION (per-window, batch-mean-subtracted) absmax: mean={_gd.mean():.2f} max={_gd.max():.2f}", flush=True)
print(f"[res] PRED proj-DEVIATION absmax:                                    mean={_pd.mean():.2f} max={_pd.max():.2f}", flush=True)
print(f"[res] pred-vs-GT CODE agreement (collapse check): mean={_ca.mean():.3f} "
      f"({'COLLAPSED — pred≈GT codes, both-resonate is confounded' if _ca.mean() > 0.9 else 'distinct codes — comparison is valid'})", flush=True)
_sp = np.array([r["samp_proj"] for r in rows]); _sd = np.array([r["samp_devproj"] for r in rows])
_sa = np.array([r["samp_agree"] for r in rows])
print(f"[res] SAMPLED (T=1.0, DISTINCT realization) proj-absmax: mean={_sp.mean():.2f} max={_sp.max():.2f} | "
      f"dev-absmax mean={_sd.mean():.2f} | code-agreement w/GT={_sa.mean():.3f} "
      f"(a distinct realization of the same ridge; if it too pins at ~2001 → resonance is realization-INDEPENDENT)", flush=True)
print(f"[res] GT proj-argmax freq-token: median={int(np.median(_gftok))} (band=[{band_tok_lo}:{band_tok_hi}]); "
      f"in-band frac={float(((_gftok>=band_tok_lo)&(_gftok<band_tok_hi)).mean()):.3f}; "
      f"time-token median={int(np.median(_gttok))}", flush=True)

# ---- SUMMARY TABLE ----
gt_proj = np.array([r["gt_proj"] for r in rows]); pred_proj = np.array([r["pred_proj"] for r in rows])
gt_out = np.array([r["gt_out"] for r in rows]);   pred_out = np.array([r["pred_out"] for r in rows])
gt_res = gt_proj >= RES_THRESH; pred_res = pred_proj >= RES_THRESH
n = len(rows)
n_gt_res = int(gt_res.sum())
n_gt_res_pred_inband = int((gt_res & ~pred_res).sum())
n_both_res = int((gt_res & pred_res).sum())

print("\n[res] ===== PER-WINDOW GT-proj vs PRED-proj (mode-active) =====", flush=True)
print(f"{'idx':>4} {'prom':>7} {'GT_proj':>10} {'GT_out':>10} {'PRED_proj':>10} {'PRED_out':>10} {'GT_res':>7} {'PR_res':>7}", flush=True)
order = np.argsort(-gt_proj)
for j in order[: min(40, n)]:
    r = rows[j]
    print(f"{j:>4} {r['prom']:>7.3f} {r['gt_proj']:>10.2f} {r['gt_out']:>10.2f} "
          f"{r['pred_proj']:>10.2f} {r['pred_out']:>10.2f} "
          f"{'Y' if gt_res[j] else '.':>7} {'Y' if pred_res[j] else '.':>7}", flush=True)

print("\n[res] ===== SUMMARY =====", flush=True)
print(f"[res] N mode-active windows = {n}", flush=True)
print(f"[res] resonance threshold (proj-absmax) = {RES_THRESH}", flush=True)
print(f"[res] GT-path resonant  : {n_gt_res}/{n}  (proj max={gt_proj.max():.1f} median={np.median(gt_proj):.1f})", flush=True)
print(f"[res] PRED-path resonant: {int(pred_res.sum())}/{n}  (proj max={pred_proj.max():.1f} median={np.median(pred_proj):.1f})", flush=True)
print(f"[res] of {n_gt_res} GT-resonant windows: {n_gt_res_pred_inband} stay IN-BAND under predicted codes, "
      f"{n_both_res} ALSO resonate under predicted codes", flush=True)

# ---- VERDICT ----
# The scalar proj-absmax is the GLOBAL max; if it is pinned (~identical across windows
# AND across GT/pred/zero-decode), it is a content-independent decoder-bias/DC saturation,
# NOT the per-window ridge. The controls (zero-decode proj, per-window DEVIATION proj) and
# the argmax location disambiguate this from a genuine ridge-driven resonance.
_zero_resonant = float(_zp.mean()) >= RES_THRESH
_input_resonant = float(_ip.mean()) >= RES_THRESH      # does the REAL residual input window resonate too?
_dev_resonant = float(_gd.mean()) >= RES_THRESH        # does the per-window (ridge) DEVIATION resonate?
_samp_agree = float(np.array([r["samp_agree"] for r in rows]).mean())
_amax_in_band = float(((_gftok >= band_tok_lo) & (_gftok < band_tok_hi)).mean())
_collapsed = float(_ca.mean()) > 0.9

if n_gt_res == 0:
    verdict = "INCONCLUSIVE — no GT-path resonance in this batch (raise MAX_WIN / lower gate / different shot)."
elif _collapsed:
    verdict = (f"CONFOUNDED — predicted argmax codes ≈ GT codes (agreement={_ca.mean():.2f}); consult the SAMPLED "
               f"(distinct realization, agreement={_samp_agree:.2f}) row instead of argmax.")
elif not _dev_resonant and not _zero_resonant:
    # The scalar max is carried by a batch-COMMON component; the per-window ridge
    # deviation is tiny; a blank decode is silent (so it IS content-driven). Whether
    # the input also resonates decides "realization bits" vs "shared low-freq content".
    _flavor = ("both the real INPUT window and a random SAMPLED code realization ALSO resonate at the same "
               f"level (input proj-absmax mean={_ip.mean():.0f}, sampled mean≈{float(np.array([r['samp_proj'] for r in rows]).mean()):.0f}), "
               "so the resonance is REALIZATION-INDEPENDENT and INPUT-INTRINSIC") if _input_resonant else (
               "the real input stays in-band while decodes resonate, so it is a codec-decode realization artifact")
    verdict = ("T2-adjacent (NOT the mode energy, and NOT a realization-specific roughness) — the ~2001 proj-absmax is "
               f"a FIXED single proj filter (out-ch {int(np.bincount([r['gt_argmax_ch'] for r in rows]).argmax())}) firing "
               f"at the DC/low-freq corner (proj-argmax freq-token median={int(np.median(_gftok))}, in-band frac={_amax_in_band:.2f}); "
               f"the per-window MODE-RIDGE deviation contributes only ~{_gd.mean():.0f} (<<{RES_THRESH:.0f}). "
               f"A blank (zero) decode is silent ({_zp.mean():.1f}) so it is content-driven, but {_flavor}. "
               "→ The mode ridge renders safely; the resonance lives in the shared low-frequency (near-DC) broadband "
               "structure amplified by one patch-conv filter. A SOURCE-SIDE / embed-path fix (rescale that proj filter, "
               "or high-pass / re-center the near-DC patch before proj) removes the resonance outright; feedback-renorm "
               "only treats the symptom (and would not even fire on the real INPUT window, which resonates too).")
elif _dev_resonant and n_both_res >= max(1, int(0.5 * n_gt_res)):
    verdict = ("T1 (mode energy itself) — the per-window ridge DEVIATION resonates and follows the ridge into BOTH the GT "
               f"and predicted decodes (dev proj-absmax mean={_gd.mean():.0f}; code agreement={_ca.mean():.2f} → distinct "
               "paths). The proj amplifies real coherent mode-ridge structure; bf16 can't hold it.")
else:
    verdict = (f"MIXED — input-resonant={_input_resonant}, zero-resonant={_zero_resonant}, dev-resonant={_dev_resonant}, "
               f"both-resonate {n_both_res}/{n_gt_res}, code-agree={_ca.mean():.2f}, argmax-in-band={_amax_in_band:.2f}. "
               "See controls above.")
print(f"\n[res] VERDICT: {verdict}", flush=True)

# ---- MODE-BAND CONCENTRATION (resonating vs non-resonating GT decodes) ----
gt_bc = np.array([r["gt_bandconc"] for r in rows])
bc_res = gt_bc[gt_res]; bc_non = gt_bc[~gt_res]
print("\n[res] ===== MODE-BAND (5-40 kHz) proj-output energy concentration (GT-path) =====", flush=True)
print(f"[res] resonating decodes    (n={bc_res.size}): band-fraction mean={np.nanmean(bc_res) if bc_res.size else float('nan'):.3f} "
      f"median={np.nanmedian(bc_res) if bc_res.size else float('nan'):.3f}", flush=True)
print(f"[res] non-resonating decodes(n={bc_non.size}): band-fraction mean={np.nanmean(bc_non) if bc_non.size else float('nan'):.3f} "
      f"median={np.nanmedian(bc_non) if bc_non.size else float('nan'):.3f}", flush=True)

# ---- RADIAL SPATIAL SPECTRUM PLOT ----
def stack_prof(patches):
    if not patches:
        return None
    ps = np.array([radial_spectrum(p) for p in patches])   # (n, nb)
    return ps.mean(0), (ps.std(0) if ps.shape[0] > 1 else np.zeros(ps.shape[1]))


pr_res = stack_prof(res_decode); pr_non = stack_prof(nonres_decode); pr_inp = stack_prof(input_windows)
nb = 32
rax = np.linspace(0, 1, nb)
plt.figure(figsize=(8.5, 5.5))
for prof, lab, c in [(pr_res, f"resonating GT decode (n={len(res_decode)})", "#c0392b"),
                     (pr_non, f"non-resonating GT decode (n={len(nonres_decode)})", "#2980b9"),
                     (pr_inp, f"natural input window (n={len(input_windows)})", "#2c3e50")]:
    if prof is None:
        continue
    m, s = prof
    m = m / max(m.max(), 1e-9)   # normalize each curve to its own peak (shape comparison)
    plt.plot(rax, m, "-", color=c, lw=1.8, label=lab)
plt.yscale("log"); plt.xlabel("normalized spatial frequency (radial, 0=DC → 1=Nyquist over F,T)")
plt.ylabel("radially-averaged |2D-FFT| (peak-normalized)")
plt.title(f"ECE decode-patch spatial spectrum — {SHOT} @ β6\n"
          f"where the resonant coherence lives in (freq×time) space")
plt.grid(alpha=.3); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "spatial_spectrum.png", dpi=140)
print(f"\n[res] wrote {OUT}/spatial_spectrum.png", flush=True)

# one-line where-does-it-live description
if pr_res is not None and pr_inp is not None:
    peak_res = int(np.argmax(pr_res[0][1:]) + 1)   # skip DC
    peak_inp = int(np.argmax(pr_inp[0][1:]) + 1)
    hi_res = float(pr_res[0][nb // 2:].sum() / max(pr_res[0].sum(), 1e-9))
    hi_inp = float(pr_inp[0][nb // 2:].sum() / max(pr_inp[0].sum(), 1e-9))
    spatial_note = (f"resonating decodes peak at radial-bin {peak_res}/{nb} with high-freq (r>0.5) fraction "
                    f"{hi_res:.3f} vs natural-input peak bin {peak_inp}/{nb} hi-frac {hi_inp:.3f} "
                    f"({'resonant coherence sits at HIGHER spatial freq (fine/rough structure)' if hi_res > 1.3 * hi_inp else 'resonant coherence at similar/low spatial freq (broad ridge)'})")
else:
    spatial_note = "insufficient patches for spatial-spectrum comparison"
print(f"[res] SPATIAL: {spatial_note}", flush=True)

# ---- persist JSON ----
res_json = dict(
    ckpt=CKPT.name, shot=SHOT, n_windows=n, res_thresh=RES_THRESH,
    n_gt_resonant=n_gt_res, n_pred_resonant=int(pred_res.sum()),
    n_gt_res_pred_inband=n_gt_res_pred_inband, n_both_resonant=n_both_res,
    gt_proj_max=float(gt_proj.max()) if n else None, gt_proj_median=float(np.median(gt_proj)) if n else None,
    pred_proj_max=float(pred_proj.max()) if n else None, pred_proj_median=float(np.median(pred_proj)) if n else None,
    bandconc_resonating_mean=float(np.nanmean(bc_res)) if bc_res.size else None,
    bandconc_nonresonating_mean=float(np.nanmean(bc_non)) if bc_non.size else None,
    zero_decode_proj_mean=float(_zp.mean()), input_proj_mean=float(_ip.mean()),
    gt_devproj_mean=float(_gd.mean()), gt_devproj_max=float(_gd.max()),
    pred_devproj_mean=float(_pd.mean()),
    sampled_proj_mean=float(_sp.mean()), sampled_devproj_mean=float(_sd.mean()),
    sampled_code_agreement_mean=float(_sa.mean()),
    code_agreement_mean=float(_ca.mean()),
    gt_argmax_ftok_median=int(np.median(_gftok)), gt_argmax_inband_frac=float(_amax_in_band),
    verdict=verdict, spatial_note=spatial_note,
    per_window=rows)
json.dump(res_json, open(OUT / "resonance_diag.json", "w"), indent=2)
print(f"[res] wrote {OUT}/resonance_diag.json", flush=True)
print("[res] DONE", flush=True)
