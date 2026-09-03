#!/usr/bin/env python
"""MODE-REGION spectrogram prediction metric (upgraded 2026-07-08).

The old per-dim-majority code-acc was CONFOUNDED: co2's flat/quiescent codes
inflated it to 99% while the actual modes went unpredicted. This version works at
the SIGNAL level, restricted to mode-bearing cells found by PER-FREQUENCY-BIN
contrast — so it covers modes at ANY frequency (ECE <100 kHz AND CO2 100-200 kHz),
never a fixed low-freq band.

IMPORTANT (user 2026-07-08): this is a MEASUREMENT focus only. Model TRAINING still
spans every frequency — the focal / class-weighted CE applies to all spectro code
tokens across all 512 freq bins, with no band restriction. This metric never feeds
back into the loss; it just scores where the modes are.

Per modality, on a mode-rich shot, decode three spectrograms:
  GT    = targets[name]                              (measured)
  recon = decode(encode_target(GT))                  (codec ceiling — do codes carry it)
  pred  = decode(argmax code_logits)                 (the world-model, deterministic)
Detect mode cells: GT exceeds its per-freq-bin time-median background by k*sigma
(per (channel, freq)). Report, IN THOSE MODE CELLS:
  recon-vs-GT corr (ceiling), pred-vs-GT corr (actual), pred-vs-recon corr.
"""
import os
import sys
from pathlib import Path

FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import scipy.ndimage as ndi
import torch
from torch.utils.data import DataLoader

# Gaussian-smoothing sigmas for mode detection (mirror eval_e2e_animation_tokamak
# _MASK_SMOOTH_F/_MASK_SMOOTH_T): coherent modes survive smoothing; isolated
# high-freq thermal-noise specks do NOT, so the mask stops flagging noise as modes.
MASK_SMOOTH_F = 1.0
MASK_SMOOTH_T = 2.0
from eval_e2e_animation_tokamak import load_model
from train_e2e_stage1 import build_datasets, forward_batch, _core
from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.e2e.output_heads import (
    SpectrogramCodeHead, SpectrogramMaskGITHead,
)

CKPT = Path(sys.argv[1] if len(sys.argv) > 1
            else "/lustre/orion/fus187/proj-shared/models/e2e_stage1_allshots_b32/e2e_stage1_best.pt")
SHOT = int(os.environ.get("SHOT", "200729"))
MODE_K = float(os.environ.get("MODE_K", "2.0"))
MAX_WIN = int(os.environ.get("MAX_WIN", "250"))
# SPEC_AE_EVAL=1: evaluate an AUTOENCODE-trained model correctly — compare the
# model's prediction against the CURRENT input window's recon (diag_inputs), not
# the next window's (targets). Without this an --spec_autoencode model is scored
# as if forecasting, which is the wrong reference.
AE_EVAL = os.environ.get("SPEC_AE_EVAL", "") != ""
# SAMPLE_TEMP > 0: decode PRED by SAMPLING the per-dim code distribution at this
# temperature instead of argmax. argmax snaps every patch to the dominant code
# when logits are uncertain → blocky collapse; sampling produces mode-LIKE
# texture (what a generative/predictive world model should output).
SAMPLE_TEMP = float(os.environ.get("SAMPLE_TEMP", "0"))
device = torch.device("cuda")

model, ckpt = load_model(CKPT, device)
model.eval()
a = ckpt["args"]
core = _core(model)
diag_names = [d["name"] for d in ckpt["diagnostics"]]
act_names = [c["name"] for c in ckpt["actuators"]]
data_dir = Path(a["data_dir"])
stats = torch.load(a["stats_path"], weights_only=False)
shot_file = data_dir / f"{SHOT}_processed.h5"
assert shot_file.exists(), f"missing {shot_file}"
print(f"ckpt step={ckpt.get('step')} best_step={ckpt.get('best_step')} | shot {SHOT} | k={MODE_K}", flush=True)

cache = Path(f"{FMH}/eval_runs/modecode_cache")
_, ds = build_datasets(
    data_dir, [shot_file], [shot_file], stats,
    a["chunk_duration_s"], a.get("prediction_horizon_s", a["chunk_duration_s"]),
    a["step_size_s"], a["warmup_s"], diag_names, act_names, cache)
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                    collate_fn=collate_fn, drop_last=False)

spec = [n for n in diag_names
        if isinstance(core.diag_heads[n], (SpectrogramCodeHead, SpectrogramMaskGITHead))]
print("spectro code-heads:", spec, flush=True)
G_acc = {n: [] for n in spec}
R_acc = {n: [] for n in spec}
P_acc = {n: [] for n in spec}


def _run_dist_sweep():
    """3d+3e: MaskGIT (steps x temperature) sweep -> DISTRIBUTIONAL GATE per config,
    with the persistence baseline + codec recon ceiling as reference lines. The
    objective the whole iteration optimizes toward: does a SAMPLED forecast fire the
    mode detector at ~GT rate, at the right freq, with matching band-power.

    Env: DIST_STEPS ("8,16"), DIST_TEMP ("0.3,0.5,0.7,1.0"), DIST_OUT (dir)."""
    import json
    sys.path.insert(0, f"{FMH}/analysis/mode_audit")
    from dist_gate import distributional_gate, _summary
    steps_grid = [int(s) for s in os.environ.get("DIST_STEPS", "8,16").split(",")]
    temp_grid = [float(t) for t in os.environ.get("DIST_TEMP", "0.3,0.5,0.7,1.0").split(",")]
    outdir = os.environ.get("DIST_OUT", f"{FMH}/eval_runs/dist_sweep")
    os.makedirs(outdir, exist_ok=True)
    Gt = {n: [] for n in spec}; In = {n: [] for n in spec}; Tok = {n: [] for n in spec}
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= MAX_WIN:
                break
            _, diag_inputs, targets, _, tok = forward_batch(model, batch, device)
            for n in spec:
                Gt[n].append(targets[n].float().cpu())      # forecast target (t+1)
                In[n].append(diag_inputs[n].float().cpu())  # current window (t) = persistence pred
                Tok[n].append(tok[n].cpu())
            seen += targets[spec[0]].shape[0]
            print(f"[sweep] collected {seen} windows", flush=True)
    results = {}
    for n in spec:
        head = core.diag_heads[n]
        G = torch.cat(Gt[n], 0); Ipred = torch.cat(In[n], 0); toks = torch.cat(Tok[n], 0)
        res = {"n_windows": int(G.shape[0])}
        # reference lines
        res["persistence"] = distributional_gate(Ipred, G, consecutive=True)
        print(f"[sweep {n}] PERSISTENCE baseline: " + _summary(res["persistence"]), flush=True)
        rec = torch.cat([head.decode(head.encode_target(G[i:i+32].to(device))).cpu()
                         for i in range(0, G.shape[0], 32)], 0)
        res["recon_ceiling"] = distributional_gate(rec, G, consecutive=True)
        print(f"[sweep {n}] RECON ceiling (gt-codes): " + _summary(res["recon_ceiling"]), flush=True)
        # the sweep
        best = None
        for st in steps_grid:
            for tp in temp_grid:
                preds = []
                for i in range(0, toks.shape[0], 32):
                    tb = toks[i:i+32].to(device)
                    if isinstance(head, SpectrogramMaskGITHead):
                        c = head.iterative_decode(tb, n_steps=st, temperature=tp)
                    else:
                        lg = head.code_logits(tb)
                        c = torch.distributions.Categorical(logits=lg / max(tp, 1e-6)).sample()
                    preds.append(head.decode(c).cpu())
                P = torch.cat(preds, 0)
                r = distributional_gate(P, G, consecutive=True)
                res[f"steps{st}_t{tp}"] = r
                print(f"[sweep {n}] steps={st} T={tp}: " + _summary(r), flush=True)
                if best is None or r["fire_recall"] > best[2]["fire_recall"]:
                    best = (f"steps{st}_t{tp}", P, r)
                    if st != steps_grid[0] or tp != temp_grid[0]:
                        pass
        res["best_config"] = best[0]
        Pbest = best[1]
        torch.save(Pbest, f"{outdir}/{n}_pred_best.pt"); torch.save(G, f"{outdir}/{n}_gt.pt")
        # proof figure: strongest-mode channel, GT | RECON-ceiling | BEST-PRED | PERSISTENCE
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            from dist_gate import strong_ch, win_P
            wsel = int(np.argmax([win_P(G[i].numpy()) for i in range(min(G.shape[0], 200))]))
            ch = strong_ch(G[wsel].numpy())
            imgs = [("GT", G[wsel, ch]), ("RECON ceiling", rec[wsel, ch]),
                    (f"PRED {best[0]}", Pbest[wsel, ch]), ("PERSISTENCE", Ipred[wsel, ch])]
            vlo, vhi = np.percentile(G[wsel, ch].numpy(), [2, 99.5])
            fig, ax = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
            for a2, (ttl, im) in zip(ax, imgs):
                a2.imshow(im.numpy(), origin="lower", aspect="auto", vmin=vlo, vmax=vhi, cmap="magma")
                a2.set_title(ttl, fontsize=9)
            fig.suptitle(f"{n} ch{ch} win{wsel} | fire_recall pred={best[2]['fire_recall']:.2f} "
                         f"pers={res['persistence']['fire_recall']:.2f} ceil={res['recon_ceiling']['fire_recall']:.2f}"
                         f" | freq_in_tol={best[2]['freq_in_tol']:.2f}", fontsize=10)
            fig.tight_layout(); fig.savefig(f"{outdir}/{n}_proof.png", dpi=110); plt.close(fig)
            print(f"[sweep {n}] saved {outdir}/{n}_proof.png", flush=True)
        except Exception as e:
            print(f"[sweep {n}] fig err {e}", flush=True)
        results[n] = res
    json.dump(results, open(f"{outdir}/dist_sweep.json", "w"), indent=2,
              default=lambda o: float(o) if hasattr(o, "item") else o)
    print(f"[sweep] wrote {outdir}/dist_sweep.json + per-modality pred_best/gt/proof", flush=True)


if os.environ.get("DIST_SWEEP"):
    _run_dist_sweep()
    sys.exit(0)

nwin = 0
with torch.no_grad():
    for batch in loader:
        if nwin >= MAX_WIN:
            break
        _, diag_inputs, targets, _, tok = forward_batch(model, batch, device)
        src = diag_inputs if AE_EVAL else targets   # AE: score vs INPUT-window recon
        for n in spec:
            head = core.diag_heads[n]
            gt = src[n].float()
            rec = head.decode(head.encode_target(gt))
            if isinstance(head, SpectrogramMaskGITHead):
                # JOINT decode: MaskGIT iterative parallel unmask (coherent).
                # SAMPLE_TEMP overrides the head's decode temperature if set.
                _codes = head.iterative_decode(
                    tok[n], temperature=(SAMPLE_TEMP if SAMPLE_TEMP > 0 else None))
                prd = head.decode(_codes)
            else:
                _lg = head.code_logits(tok[n])                   # (B,n_tok,dim,L)
                if SAMPLE_TEMP > 0:
                    _codes = torch.distributions.Categorical(
                        logits=_lg / SAMPLE_TEMP).sample()      # (B,n_tok,dim)
                else:
                    _codes = _lg.argmax(-1)
                prd = head.decode(_codes)
            T = min(gt.shape[-1], rec.shape[-1], prd.shape[-1])
            G_acc[n].append(gt[..., :T].cpu())
            R_acc[n].append(rec[..., :T].cpu())
            P_acc[n].append(prd[..., :T].cpu())
        nwin += targets[spec[0]].shape[0]
        print(f"windows so far: {nwin}", flush=True)


def _corr(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or x[m].std() < 1e-9 or y[m].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _ssim(a, b):
    """OBJECTIVE structural similarity between two (C,F,T) spectrogram stacks,
    Gaussian-windowed, per channel over the F-T plane, averaged. Unlike
    envelope-dominated correlation, a BLOCKY prediction scores LOW against a
    mode-structured reference — this is the metric that tracks the picture."""
    a = np.nan_to_num(np.asarray(a, float)); b = np.nan_to_num(np.asarray(b, float))
    dr = float(max(a.max(), b.max()) - min(a.min(), b.min())) or 1.0
    C1, C2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    s = (1.5, 1.5)
    vals = []
    for c in range(a.shape[0]):
        x, y = a[c], b[c]
        mux = ndi.gaussian_filter(x, s); muy = ndi.gaussian_filter(y, s)
        vx = ndi.gaussian_filter(x * x, s) - mux * mux
        vy = ndi.gaussian_filter(y * y, s) - muy * muy
        vxy = ndi.gaussian_filter(x * y, s) - mux * muy
        smap = ((2 * mux * muy + C1) * (2 * vxy + C2)) / (
            (mux * mux + muy * muy + C1) * (vx + vy + C2))
        vals.append(float(np.mean(smap)))
    return float(np.mean(vals))


fmax = 250.0  # nominal top of the STFT freq axis (kHz), for reporting bands
print(f"\n============ MODE-REGION METRIC (shot {SHOT}, k={MODE_K}, {nwin} win) ============", flush=True)
for n in spec:
    G = torch.cat(G_acc[n], 0)                       # (nw, C, F, T)
    R = torch.cat(R_acc[n], 0)
    P = torch.cat(P_acc[n], 0)
    nw, C, F, T = G.shape
    G = G.permute(1, 2, 0, 3).reshape(C, F, nw * T).numpy()   # (C, F, T_total)
    R = R.permute(1, 2, 0, 3).reshape(C, F, nw * T).numpy()
    P = P.permute(1, 2, 0, 3).reshape(C, F, nw * T).numpy()
    # Gaussian-smooth (per channel, over freq+time) so the MASK captures COHERENT
    # modes, not isolated high-freq noise specks (the eps-trap that mislabeled ece
    # at 225-249kHz). Mask on smoothed; metrics use the RAW G/R/P values.
    Gs = ndi.gaussian_filter(G, sigma=(0.0, MASK_SMOOTH_F, MASK_SMOOTH_T))
    bg = np.median(Gs, axis=2, keepdims=True)
    sd = Gs.std(axis=2, keepdims=True) + 1e-6
    mode = Gs > (bg + MODE_K * sd)
    frac = float(mode.mean())
    rc, pg, pr = _corr(R[mode], G[mode]), _corr(P[mode], G[mode]), _corr(P[mode], R[mode])
    rc_all, pg_all = _corr(R, G), _corr(P, G)
    # RESIDUAL corr (envelope removed) — the HONEST mode metric. Subtract each
    # spectrogram's per-freq time-mean so the shared broadband envelope (which
    # inflates the bulk corr to ~0.7 even for a mode-less pred) is gone; what
    # remains is the temporal MODE structure. A smooth / mean-collapsed pred has
    # ~zero residual in the mode cells -> corr -> ~0. This tracks the render.
    Gr = G - G.mean(axis=2, keepdims=True)
    Rr = R - R.mean(axis=2, keepdims=True)
    Pr = P - P.mean(axis=2, keepdims=True)
    rc_res = _corr(Rr[mode], Gr[mode])
    pg_res = _corr(Pr[mode], Gr[mode])
    # OBJECTIVE structural similarity (tracks the PICTURE; blocky pred -> LOW).
    # ssim_pr = how close PRED is to the achievable RECON (the pred≈recon bar);
    # ssim_rg = recon-vs-GT ceiling; ssim_pr_res = mode-structure (envelope removed).
    ssim_pr = _ssim(P, R)
    ssim_rg = _ssim(R, G)
    ssim_pr_res = _ssim(Pr, Rr)
    # which freq bands hold the modes (so we can cross-check ECE<100 / CO2 100-200)
    fperbin = fmax / F
    mode_by_f = mode.mean(axis=(0, 2))               # (F,) fraction of mode cells per freq
    top = np.argsort(mode_by_f)[::-1][:3]
    bands = ", ".join(f"{int(i*fperbin)}kHz" for i in sorted(top))
    print(f"\n[{n}]  C={C} F={F}  mode-cell frac={frac:.3f}  (top mode freqs ~ {bands})", flush=True)
    print(f"   recon vs GT : mode {rc:.3f}  | all {rc_all:.3f}   <- ceiling (codes carry the modes)", flush=True)
    print(f"   PRED  vs GT : mode {pg:.3f}  | all {pg_all:.3f}   <- model's MODE prediction", flush=True)
    print(f"   pred vs recon: mode {pr:.3f}   (how close pred gets to the achievable ceiling)", flush=True)
    print(f"   -- RESIDUAL (envelope removed = MODE structure; the honest number) --", flush=True)
    print(f"   recon-resid vs GT : {rc_res:.3f}   <- ceiling (codes carry mode STRUCTURE)", flush=True)
    print(f"   PRED-resid  vs GT : {pg_res:.3f}   <- model's MODE-STRUCTURE prediction (tracks eye)", flush=True)
    print(f"   == OBJECTIVE SSIM (blocky pred -> LOW; this tracks the picture) ==", flush=True)
    print(f"   SSIM pred-vs-RECON : {ssim_pr:.3f}   <- the pred≈recon bar (1.0 = indistinguishable)", flush=True)
    print(f"   SSIM recon-vs-GT   : {ssim_rg:.3f}   <- ceiling (codec's own fidelity)", flush=True)
    print(f"   SSIM pred-vs-recon RESIDUAL : {ssim_pr_res:.3f}   <- mode-structure only", flush=True)
    # PROOF PLOT (opt-in via SAVE_FIG_DIR): GT | RECON (codec ceiling) | PRED (argmax)
    # for the mode-richest channel, shared color scale. A successful overfit makes
    # the RECON and PRED rows indistinguishable — that IS the "pred==recon" proof.
    _figdir = os.environ.get("SAVE_FIG_DIR", "")
    if _figdir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(_figdir, exist_ok=True)
        ch = int(mode.sum(axis=(1, 2)).argmax())          # clearest-mode channel
        tmax = min(G.shape[2], 1200)
        gg, rr, pp = G[ch, :, :tmax], R[ch, :, :tmax], P[ch, :, :tmax]
        vlo, vhi = float(np.percentile(gg, 2)), float(np.percentile(gg, 99.5))
        fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
        for a, img, ttl in zip(
            ax, (gg, rr, pp),
            ("GROUND TRUTH", "RECON (codec ceiling)",
             f"PRED ({'sampled T=%.1f' % SAMPLE_TEMP if SAMPLE_TEMP > 0 else 'argmax'})"),
        ):
            a.imshow(img, origin="lower", aspect="auto", vmin=vlo, vmax=vhi,
                     cmap="magma", extent=[0, tmax, 0, fmax])
            a.set_ylabel(f"{ttl}\nfreq (kHz)")
        ax[-1].set_xlabel("time (frames)")
        fig.suptitle(
            f"{n}  ch{ch}  |  SSIM(pred,recon)={ssim_pr:.3f} [ceil {ssim_rg:.3f}]  "
            f"pred-resid={pg_res:.3f}  step={ckpt.get('step')}"
        )
        fig.tight_layout()
        outp = f"{_figdir}/{n}_proof_ch{ch}.png"
        fig.savefig(outp, dpi=110)
        plt.close(fig)
        print(f"   [saved proof plot] {outp}", flush=True)
    # ALL-CHANNEL GRID (opt-in via SAVE_GRID_DIR): every channel, GT | RECON | PRED
    # — a rigorous per-channel proof (no cherry-picked channel). Deterministic:
    # channels in index order, shared color scale (GT percentiles over all ch).
    _griddir = os.environ.get("SAVE_GRID_DIR", "")
    if _griddir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(_griddir, exist_ok=True)
        Cn = G.shape[0]
        tg = min(G.shape[2], 1000)
        vlo = float(np.percentile(G[..., :tg], 2))
        vhi = float(np.percentile(G[..., :tg], 99.5))
        fig, ax = plt.subplots(Cn, 3, figsize=(11, max(3.0, Cn * 0.7)),
                               squeeze=False, sharex=True, sharey=True)
        for c in range(Cn):
            for j, (img, ttl) in enumerate(zip(
                (G[c, :, :tg], R[c, :, :tg], P[c, :, :tg]), ("GT", "RECON", "PRED"))):
                ax[c][j].imshow(img, origin="lower", aspect="auto", vmin=vlo,
                                vmax=vhi, cmap="magma", extent=[0, tg, 0, fmax])
                ax[c][j].set_xticks([]); ax[c][j].set_yticks([])
                if c == 0:
                    ax[c][j].set_title(ttl, fontsize=10)
            ax[c][0].set_ylabel(f"ch{c}", fontsize=7, rotation=0, ha="right",
                                va="center")
        dec = ("argmax" if SAMPLE_TEMP <= 1e-3 and SAMPLE_TEMP > 0
               else (f"T={SAMPLE_TEMP}" if SAMPLE_TEMP > 0 else "default"))
        fig.suptitle(f"{n} — ALL {Cn} channels | GT | RECON | PRED ({dec})  "
                     f"SSIM(pred,recon)={ssim_pr:.3f}  step={ckpt.get('step')}")
        fig.tight_layout()
        outp = f"{_griddir}/{n}_grid_allch.png"
        fig.savefig(outp, dpi=90)
        plt.close(fig)
        print(f"   [saved ALL-CH grid] {outp} ({Cn} channels)", flush=True)
print("\n==================================================================", flush=True)
