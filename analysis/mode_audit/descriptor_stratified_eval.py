#!/usr/bin/env python
"""Forecast-layer (descriptor-head) mode-skill eval — STRATIFIED by window activity.

The decisive, texture-free instrument (per the 2026-07-20 ruling): decoded spectro
panels can't be read (independent-marginal decode erases coherent modes by construction).
Mode content is read from the DESCRIPTOR HEAD instead. We reproduce the trainer's exact
descriptor metrics (train_e2e_stage1.py compute_step_loss `_desc_term`, ~L1595-1646) and
aggregate `ftol` (MODEL mode-freq accuracy) vs `ftp` (PERSISTENCE baseline) + `hfrac`
(mean-collapse detector) over three window strata:

  * all        — every time-column (dominated by quiescent windows; ftol≈ftp≈1 trivially)
  * sustained  — mode present in BOTH input and target (_astatic): persistence is strong here
  * transition — mode ONSET/DEATH (presence flips input->target): persistence CANNOT copy it,
                 so ftol>ftp on THIS stratum is genuine forecast skill.

Verdict = ftol vs ftp on the transition (and sustained) strata. Not any strip of pixels.

Read-only w.r.t. the running chain: loads the checkpoint, writes nothing to the model dir.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path("/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub")
sys.path.insert(0, str(REPO / "scripts" / "training"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

KHZ_PER_BIN = 500000.0 / 1024 / 1000.0   # STFT fs=500kHz, n_fft=1024 -> 0.488 kHz/bin (matches descriptor_head_proof)

# proven checkpoint loader (rebuilds the full-modality FSQ arch incl. spec_descriptor_heads)
from eval_e2e_animation_tokamak import load_model
# exact forward pass: returns (predictions, diag_inputs, targets, masks, token_slices);
# targets[spectro] is already the descriptor-extended _desc_full_tgt (trunc_t*max_h).
from train_e2e_stage1 import forward_batch
from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
# pilot detection standard (prominence over local freq background + P75 gate + persistence).
# Its band (MODE_LO..MODE_HI, DF) aligns bin-for-bin with the descriptor head's mode band,
# so GT detection/peak (raw spectro) and model peak (descriptor forecast) share one freq axis.
from dist_gate import band_prom, win_P, strong_ch, MODE_LO as DG_LO, TOL_BINS as DG_TOL


def _core(m):
    return m.module if hasattr(m, "module") else m


def _window_peaks(tspec_np, inp_np, pe_np, mlo):
    """Per-batch window peaks/fire, all in ABSOLUTE freq bins (shared 5-40 kHz axis).

    tspec_np (B,C,F,Tw) = raw GT t+h spectro; inp_np (B,C,F,T) = raw input (persistence) spectro;
    pe_np (B,NF,TCOL) = model descriptor logit; mlo = descriptor mode_lo (== DG_LO).
    Returns arrays: gt_fire (B,) (win_P), gt_peak (B,), pers_peak (B,) (both dist_gate band_prom
    peaks), model_peak (B,) (argmax of the TCOL-mean descriptor + mlo), gt_pd (B,NF) = the GT
    prominence PROFILE of the strong channel (band_prom(...)[0], length NF = MODE_HI-MODE_LO),
    energy (B,) = raw data-present proxy = mean abs over C,F,Tw of the raw GT t+h spectro.
    """
    B = tspec_np.shape[0]
    gt_fire = np.empty(B); gt_peak = np.empty(B, int)
    pers_peak = np.empty(B, int); model_peak = np.empty(B, int)
    energy = np.empty(B)
    gt_pd = None
    mprof = np.clip(pe_np, 0, None).mean(axis=2)          # (B,NF) TCOL-averaged model profile
    for b in range(B):
        gt_fire[b] = win_P(tspec_np[b])
        gch = strong_ch(tspec_np[b]); gpd, gpk, _ = band_prom(tspec_np[b, gch]); gt_peak[b] = gpk
        ich = strong_ch(inp_np[b]); _, ipk, _ = band_prom(inp_np[b, ich]); pers_peak[b] = ipk
        model_peak[b] = int(mprof[b].argmax()) + mlo
        energy[b] = float(np.abs(tspec_np[b]).mean())     # raw data-present proxy (mean abs over C,F,T)
        if gt_pd is None:
            gt_pd = np.empty((B, gpd.shape[0]))           # (B,NF) GT prominence profile, strong channel
        gt_pd[b] = gpd
    return gt_fire, gt_peak, pers_peak, model_peak, gt_pd, energy


def _detect_gate(gt_fire, gt_peak, fire_pct=75.0, tol_bins=DG_TOL):
    """Pilot presence gate: fire >= P75(fire), AND persistent (peak agrees within tol with a
    fired neighbor — 'samples persist, not speckle'). Returns (detected mask, fire_cut)."""
    cut = float(np.percentile(gt_fire, fire_pct)) if len(gt_fire) else 0.0
    fired = gt_fire >= cut
    T = len(gt_fire); det = np.zeros(T, bool)
    for i in range(T):
        if not fired[i]:
            continue
        prev_ok = i > 0 and fired[i - 1] and abs(int(gt_peak[i]) - int(gt_peak[i - 1])) <= tol_bins
        next_ok = i < T - 1 and fired[i + 1] and abs(int(gt_peak[i]) - int(gt_peak[i + 1])) <= tol_bins
        det[i] = prev_ok or next_ok
    return det, cut


def _detect(gt_pd, gt_fire, energy, input_peak, fire_pct=75.0, tol=DG_TOL,
            max_drift=3, min_run=2, const_frac=0.40):
    """Hardened GT-mode detector operating on the RAW prominence profiles (T,NF).

    Edge-guards the band, excludes constant pickup lines, gates on a data-present-relative
    fire percentile, and admits DRIFT-TOLERANT ridge segments (chirps up to `max_drift`
    bins/window over runs of >= `min_run`). Returns a dict of per-window arrays.

    gt_pd:(T,NF) GT prominence profiles; gt_fire:(T,) [kept as fallback, unused here];
    energy:(T,) raw data-present proxy; input_peak:(T,) persistence peak (ABSOLUTE bins).
    Keys: detected(bool T), peak(int T ABSOLUTE bins), stable(bool T), transition(bool T),
    const_bins(list of ABSOLUTE bins excluded), data_present(bool T), fire_cut(float).
    """
    gt_pd = np.asarray(gt_pd, dtype=float)
    T, NF = gt_pd.shape
    energy = np.asarray(energy, dtype=float)
    input_peak = np.asarray(input_peak)

    # 1. data-present mask (relative to a robust per-shot high-energy reference)
    data_present = energy > 0.10 * np.percentile(energy, 90)

    # 2. edge-guard: mask 2 band-edge bins each side so ridge peaks can't pin to the border
    pdm = gt_pd.copy()
    if NF >= 4:
        pdm[:, :2] = -np.inf
        pdm[:, -2:] = -np.inf

    # 3. constant-line (pickup) exclusion: PRESENCE-based (NOT relative to each window's own max —
    #    that MISSED secondary lines and only killed single bins: mhr's 20 kHz line stayed marked,
    #    the 13 kHz mark hopped to the adjacent bin, and bes's bottom-band mark hopped up one bin).
    #    A bin is a pickup line if it is "notably prominent" (above an ABSOLUTE per-shot P75 level)
    #    in > 50% of data-present windows. Each core line is then DILATED +-2 bins so whole pickup
    #    BANDS (and their secondary lines) are removed, not a lone bin. Intermittent real modes have
    #    low presence-occupancy and survive.
    const_bins = []
    n_dp = int(data_present.sum())
    if n_dp > 0:
        # absolute "notably prominent" reference: P75 of finite prominence over data-present windows
        pdf = np.where(np.isfinite(pdm), pdm, np.nan)                # (T,NF); edge-guarded -inf -> nan
        dp_vals = pdf[data_present]
        with np.errstate(invalid="ignore"):
            ref = float(np.nanpercentile(dp_vals, 75)) if data_present.any() else 0.0
        if not np.isfinite(ref):
            ref = 0.0
        present = np.isfinite(pdm) & (pdm > ref)                    # (T,NF) notably-prominent mask
        occ = present[data_present].mean(axis=0)                    # (NF,) per-bin presence fraction
        const_core = np.where(occ > 0.50)[0]                        # relative bins of pickup cores
        dilated = set()
        for c in const_core:
            for r in range(max(0, int(c) - 2), min(NF - 1, int(c) + 2) + 1):
                dilated.add(int(r))
        for r in sorted(dilated):
            pdm[:, r] = -np.inf                                     # remove whole pickup band
            const_bins.append(int(r + DG_LO))                       # ABSOLUTE bin
        const_bins = sorted(set(const_bins))

    # 4. peak + fire per window (all-masked windows -> fire = -inf)
    peak_rel = np.argmax(pdm, axis=1)
    peak = peak_rel + DG_LO                                          # ABSOLUTE bins
    fire = pdm.max(axis=1)                                           # -inf where fully masked

    # 5. data-present-relative fire cut
    finite = data_present & np.isfinite(fire)
    if finite.any():
        fire_cut = float(np.percentile(fire[finite], fire_pct))
    else:
        fire_cut = float("inf")
    fired = (fire >= fire_cut) & data_present

    # 6. drift-tolerant ridge segments: maximal runs of consecutive fired windows whose
    #    peak moves <= max_drift bins/window; runs of length >= min_run are detected.
    detected = np.zeros(T, bool)
    i = 0
    while i < T:
        if not fired[i]:
            i += 1
            continue
        j = i + 1
        while j < T and fired[j] and abs(int(peak[j]) - int(peak[j - 1])) <= max_drift:
            j += 1
        if (j - i) >= min_run:
            detected[i:j] = True
        i = j

    # 7. split detected windows into stable (peak persists near input) vs transition
    dpk = np.abs(peak - input_peak)
    stable = detected & (dpk <= tol)
    transition = detected & (dpk > tol)

    # 8. MULTI-PEAK detection (for the QC figure): the single-peak path above finds only the
    #    dominant (argmax) mode per window, so it MISSES coexisting modes (ece runs 2-3
    #    simultaneous chirps). Raw per-window local maxima also admit transient NOISE SPECKLE
    #    (bes), so we RIDGE-PERSISTENCE filter: link per-window peaks into drift-tolerant ridges
    #    and keep only peaks in ridges of length >= min_run. Coexisting persistent ridges (ece)
    #    survive; isolated speckle (bes) is dropped.
    #    8a. per-window candidate peaks (unchanged logic), grouped BY WINDOW.
    cand_by_win = [[] for _ in range(T)]                             # cand_by_win[i] = [rel_bin,...]
    for i in range(T):
        if not data_present[i]:
            continue
        row = pdm[i]
        # strict interior local maxima that clear the fire cut
        cand = []
        for j in range(1, NF - 1):
            v = row[j]
            if not np.isfinite(v) or v < fire_cut:
                continue
            if v > row[j - 1] and v >= row[j + 1]:
                cand.append(j)
        if not cand:
            continue
        cand.sort(key=lambda jj: row[jj], reverse=True)             # highest first (greedy)
        taken = []
        for j in cand:
            if all(abs(j - t) >= 2 for t in taken):
                taken.append(j)
                if len(taken) >= 4:
                    break
        cand_by_win[i] = sorted(taken)

    # 8b. greedy drift-tolerant ridge linking across consecutive windows. Each ridge is a list of
    #     (win, rel_bin). For window i, match candidates one-to-one (nearest first) to active ridges
    #     whose last window == i-1 and whose last bin is within max_drift; unmatched candidates start
    #     new ridges; ridges not extended this window are closed.
    active = []                                                     # list of ridges (each a list of (win,bin))
    confirmed = []
    for i in range(T):
        cands = list(cand_by_win[i])
        # only ridges ending on the immediately-previous window can be extended
        extendable = [r for r in active if r[-1][0] == i - 1]
        stale = [r for r in active if r[-1][0] != i - 1]
        confirmed.extend(r for r in stale if len(r) >= min_run)     # close stale ridges
        # build all (drift, ridge_idx, cand_idx) pairs within max_drift, match nearest first
        pairs = []
        for ri, r in enumerate(extendable):
            lb = r[-1][1]
            for ci, cb in enumerate(cands):
                d = abs(cb - lb)
                if d <= max_drift:
                    pairs.append((d, ri, ci))
        pairs.sort(key=lambda p: p[0])
        used_r = set(); used_c = set()
        for d, ri, ci in pairs:
            if ri in used_r or ci in used_c:
                continue
            extendable[ri].append((i, cands[ci]))
            used_r.add(ri); used_c.add(ci)
        # ridges extendable but NOT matched this window are closed
        closed = [r for ri, r in enumerate(extendable) if ri not in used_r]
        confirmed.extend(r for r in closed if len(r) >= min_run)
        kept = [r for ri, r in enumerate(extendable) if ri in used_r]
        # unmatched candidates start fresh ridges
        newr = [[(i, cands[ci])] for ci in range(len(cands)) if ci not in used_c]
        active = kept + newr
    # close any still-active ridges at the end
    confirmed.extend(r for r in active if len(r) >= min_run)

    peaks_pw = [(int(win), int(rel_bin + DG_LO))                     # ABSOLUTE bin
                for ridge in confirmed for (win, rel_bin) in ridge]

    return {"detected": detected, "peak": peak.astype(int), "stable": stable,
            "transition": transition, "const_bins": const_bins,
            "data_present": data_present, "fire_cut": fire_cut,
            "peaks_pw": peaks_pw}


def _detect_broadband(gt_pd, energy, fire_pct=75.0):
    """Distributional detector for BROADBAND modalities (co2): no single ridge peak, so measure
    total band-power ACTIVITY per window instead. A window is 'active' when its integrated 5-40kHz
    prominence clears a data-present-relative percentile gate.

    gt_pd:(T,NF) GT prominence profiles; energy:(T,) raw data-present proxy.
    Returns dict: active(bool T), data_present(bool T), bandpower(float T), cut(float).
    """
    gt_pd = np.asarray(gt_pd, dtype=float)
    energy = np.asarray(energy, dtype=float)
    data_present = energy > 0.10 * np.percentile(energy, 90)
    bandpower = np.clip(gt_pd, 0, None).sum(axis=1)                 # total 5-40kHz prominence/window
    if data_present.any():
        cut = float(np.percentile(bandpower[data_present], fire_pct))
    else:
        cut = float("inf")
    active = (bandpower >= cut) & data_present
    return {"active": active, "data_present": data_present,
            "bandpower": bandpower, "cut": cut}


def _process_shot(model, core, spec_heads, spec_mods, shot_file, stats,
                  diag_names, act_names, args, device):
    """Run one shot, time-ordered; per modality return per-WINDOW arrays (shared 5-40kHz bins):
    gt_fire (win_P), gt_peak, pers_peak, model_peak, and gt_prof (TCOL-mean GT descriptor, NF)."""
    ds = TokamakMultiFileDataset(
        [str(shot_file)], chunk_duration_s=args.chunk_duration_s, prediction_mode=True,
        prediction_horizon_s=args.prediction_horizon_s, step_size_s=args.chunk_duration_s,
        warmup_s=args.warmup_s, preprocessing_stats=stats, input_signals=diag_names,
        target_signals=diag_names + act_names, lengths_cache_path=None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0, drop_last=False)
    acc = {m: {k: [] for k in ("gt_fire", "gt_peak", "pers_peak", "model_peak",
                                "gt_prof", "gt_pd", "energy")}
           for m in spec_mods}
    with torch.no_grad():
        for batch in loader:
            predictions, diag_inputs, targets, masks, token_slices = forward_batch(model, batch, device)
            for name in spec_mods:
                if name not in token_slices:
                    continue
                dh = spec_heads[name]; horizons = getattr(dh, "horizons", (1,))
                d_pred_all = dh(token_slices[name])
                inp_desc = dh.descriptor_target(diag_inputs[name])
                anc = inp_desc / inp_desc.amax(dim=1, keepdim=True).clamp_min(1e-6)
                sw = targets[name]
                pw = (predictions[name].shape[-1]
                      if torch.is_tensor(predictions.get(name)) else sw.shape[-1])
                nsw = max(1, sw.shape[-1] // max(1, pw)); Tw = pw if nsw > 1 else sw.shape[-1]
                hi = len(horizons) - 1; off = 0 if nsw <= 1 else min(horizons[hi] - 1, nsw - 1)
                tspec = sw[..., off * Tw:(off + 1) * Tw]                  # raw GT t+h spectro (B,C,F,Tw)
                pe = anc * args.anchor_beta + d_pred_all[:, hi]           # (B,NF,TCOL)
                gf, gp, pp, mp, gpd, en = _window_peaks(
                    tspec.detach().cpu().numpy(), diag_inputs[name].detach().cpu().numpy(),
                    pe.detach().cpu().numpy(), dh.mode_lo)
                acc[name]["gt_fire"].append(gf); acc[name]["gt_peak"].append(gp)
                acc[name]["pers_peak"].append(pp); acc[name]["model_peak"].append(mp)
                acc[name]["gt_pd"].append(gpd); acc[name]["energy"].append(en)          # (B,NF), (B,)
                acc[name]["gt_prof"].append(dh.descriptor_target(tspec).mean(2).cpu().numpy())  # (B,NF)
    out = {}
    for name in spec_mods:
        if not acc[name]["gt_fire"]:
            continue
        out[name] = {k: np.concatenate(acc[name][k]) for k in acc[name]}
    return out


def render_track_figure(model, core, ckpt, spec_heads, spec_mods, args, device):
    """Presence-GATED descriptor-track figure (3-panel), 1 modality/figure, narrowband only.
    Detection = dist_gate P75-fire + persistence (via _process_shot + _detect_gate). GT mode shown
    ONLY on detected windows (no-mode = its own state, masked); model+persistence scored on detected."""
    shot = args.figure_shot
    stats = torch.load(args.stats_path, weights_only=False)
    diag_names = [c.name for c in core.diagnostics]; act_names = [c.name for c in core.actuators]
    f = Path(args.data_dir) / f"{shot}_processed.h5"; assert f.exists(), f"no shot file {f}"
    R = _process_shot(model, core, spec_heads, spec_mods, f, stats, diag_names, act_names, args, device)
    outdir = Path(args.figure_out); outdir.mkdir(parents=True, exist_ok=True)
    step = ckpt.get("step", ckpt.get("global_step", "?")); BROADBAND = {"co2"}
    for name in spec_mods:
        if name not in R: continue
        if name in BROADBAND:
            print(f"[fig] {name}: BROADBAND (no ridge; descriptor forecasts a distribution, not a peak) — skip", flush=True); continue
        d = R[name]; gt_fire=d["gt_fire"]; gt_pk=d["gt_peak"]; pers_pk=d["pers_peak"]; mdl_pk=d["model_peak"]; gt_prof=d["gt_prof"]
        T=len(gt_fire); NF=gt_prof.shape[1]
        detected, cut = _detect_gate(gt_fire, gt_pk, fire_pct=args.fire_pct); nd=int(detected.sum())
        if nd < 5:
            print(f"[fig] {name}: <5 detected-mode windows — skip", flush=True); continue
        khz=(np.arange(NF)+DG_LO)*KHZ_PER_BIN; x=np.arange(T)
        def mk(pk): return np.where(detected, pk.astype(float)*KHZ_PER_BIN, np.nan)
        gt_y=mk(gt_pk); mo_y=mk(mdl_pk)
        gp=np.clip(gt_prof,0,None); vhi=float(np.percentile(gp,99.5))+1e-6; ext=[0,T,khz[0],khz[-1]]
        fig, ax = plt.subplots(3,1,figsize=(12,8), sharex=True, gridspec_kw={"height_ratios":[1,1,0.55]})
        ax[0].imshow(gp.T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=vhi, extent=ext)
        for i in np.where(detected)[0]: ax[0].axvspan(i-0.5,i+0.5,color="cyan",alpha=0.05,lw=0)
        ax[0].set_ylabel("GT ridge\n(kHz)")
        ax[0].set_title(f"{name} — shot {shot}, step {step}  ({nd}/{T} detected-mode windows)", loc="left", fontsize=10)
        ax[1].imshow(gp.T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=vhi*3, extent=ext, alpha=0.30)
        ax[1].plot(x, gt_y, ".", color="white", ms=5, label="GT mode (detected)")
        ax[1].plot(x, mo_y, ".", color="deepskyblue", ms=4, label="model forecast")
        ax[1].set_ylim(khz[0], khz[-1]); ax[1].set_ylabel("mode freq\n(kHz)"); ax[1].legend(loc="upper right", fontsize=8, framealpha=0.85)
        m_err=np.abs(mdl_pk-gt_pk).astype(float)*KHZ_PER_BIN; p_err=np.abs(pers_pk-gt_pk).astype(float)*KHZ_PER_BIN
        me=np.where(detected,m_err,np.nan); pe_=np.where(detected,p_err,np.nan)
        ytop=float(np.nanpercentile(np.concatenate([me,pe_]),99))+1e-6
        ax[2].fill_between(x,0,ytop,where=detected&(p_err>=m_err),step="mid",color="green",alpha=0.13,lw=0)
        ax[2].fill_between(x,0,ytop,where=detected&(m_err>p_err),step="mid",color="red",alpha=0.11,lw=0)
        ax[2].plot(x, me, ".", color="C0", ms=4, label="|model-GT|")
        ax[2].plot(x, pe_, ".", color="orange", ms=3, label="|persistence-GT|")
        ax[2].set_ylim(0,ytop); ax[2].set_xlim(0,T); ax[2].set_ylabel("freq err\n(kHz)")
        ax[2].set_xlabel("window (blank = no detected mode)"); ax[2].legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.85)
        tol_khz=DG_TOL*KHZ_PER_BIN; ftol=float((m_err[detected]<=tol_khz).mean()); ftp=float((p_err[detected]<=tol_khz).mean())
        cap=(f"{name}: shot {shot}, step {step}, beta={args.anchor_beta}. Detected-mode windows only "
             f"(dist_gate P{args.fire_pct:.0f}+persistence): {nd}/{T}. mode-freq within +-{tol_khz:.1f}kHz — "
             f"model {ftol*100:.0f}% vs persistence {ftp*100:.0f}%. green=model wins, red=persistence wins.")
        fig.text(0.01,0.006,cap,fontsize=7.5,wrap=True); fig.tight_layout(rect=[0,0.035,1,1])
        outp=outdir/f"{name}_track_{shot}_step{step}.png"; fig.savefig(outp,dpi=140); plt.close(fig)
        print(f"[fig] {name}: {outp}  detected={nd}/{T} ftol={ftol:.3f} ftp={ftp:.3f} (dist_gate-gated)", flush=True)
    print(f"[fig] done -> {outdir}", flush=True)


def render_full_freq_view(model, core, ckpt, spec_heads, spec_mods, args, device):
    """FULL-FREQUENCY (0-250 kHz) GT-spectrogram view per spectro modality — GT-ONLY diagnostic.

    The descriptor head only forecasts the 5-40 kHz band, but some modalities (co2) carry their
    modes ABOVE 40 kHz where the descriptor is blind. This renders the whole 512-bin GT band
    (0-250 kHz) so we can SEE where each modality's energy actually sits. No model forecast, no
    skill number — this is purely "where are the modes" for the raw GT spectrogram.
    """
    shot = args.figure_shot
    assert shot, "render_full_freq_view requires --figure_shot"
    stats = torch.load(args.stats_path, weights_only=False)
    diag_names = [c.name for c in core.diagnostics]; act_names = [c.name for c in core.actuators]
    f = Path(args.data_dir) / f"{shot}_processed.h5"; assert f.exists(), f"no shot file {f}"
    ds = TokamakMultiFileDataset(
        [str(f)], chunk_duration_s=args.chunk_duration_s, prediction_mode=True,
        prediction_horizon_s=args.prediction_horizon_s, step_size_s=args.chunk_duration_s,
        warmup_s=args.warmup_s, preprocessing_stats=stats, input_signals=diag_names,
        target_signals=diag_names + act_names, lengths_cache_path=None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0, drop_last=False)
    outdir = Path(args.figure_out); outdir.mkdir(parents=True, exist_ok=True)
    step = ckpt.get("step", ckpt.get("global_step", "?"))

    # accumulate per-window full-freq profile + data-present energy for every spectro modality
    prof_acc = {name: [] for name in spec_mods}
    energy_acc = {name: [] for name in spec_mods}
    with torch.no_grad():
        for batch in loader:
            predictions, diag_inputs, targets, masks, token_slices = forward_batch(model, batch, device)
            for name in spec_mods:
                if name not in targets:
                    continue
                sw = targets[name]                                        # raw GT spectro (B,C,F,T)
                pw = (predictions[name].shape[-1]
                      if torch.is_tensor(predictions.get(name)) else sw.shape[-1])
                nsw = max(1, sw.shape[-1] // max(1, pw)); Tw = pw if nsw > 1 else sw.shape[-1]
                dh = spec_heads[name]; horizons = getattr(dh, "horizons", (1,))
                hi = len(horizons) - 1; off = 0 if nsw <= 1 else min(horizons[hi] - 1, nsw - 1)
                tspec = sw[..., off * Tw:(off + 1) * Tw]                  # raw GT t+h spectro (B,C,F,Tw)
                # per-window full-freq magnitude profile: channel-MAX over C, mean over time frames
                prof = tspec.abs().amax(dim=1).mean(dim=-1)               # (B,F)
                energy = tspec.abs().mean(dim=(1, 2, 3))                  # (B,)
                prof_acc[name].append(prof.detach().cpu().numpy())
                energy_acc[name].append(energy.detach().cpu().numpy())

    for name in spec_mods:
        if not prof_acc[name]:
            print(f"[fullfreq] {name}: no windows — skip", flush=True); continue
        prof_ff = np.concatenate(prof_acc[name], axis=0)                  # (T, F)
        energy = np.concatenate(energy_acc[name], axis=0)                 # (T,)
        Tall, Fbins = prof_ff.shape
        data_present = energy > 0.10 * np.percentile(energy, 90)
        # clip the x-axis to the data-present span (keep everything between first & last present)
        pres_idx = np.where(data_present)[0]
        if len(pres_idx):
            lo, hi_i = int(pres_idx[0]), int(pres_idx[-1]) + 1
        else:
            lo, hi_i = 0, Tall
        prof_view = prof_ff[lo:hi_i]                                      # (T_present, F)
        T_present = prof_view.shape[0]
        n_dp = int(data_present.sum())

        # PER-FREQ normalization: subtract each freq bin's temporal median so the DC/low-freq
        # envelope (which dominates raw magnitude and buried the modes) is removed — modes at ANY
        # frequency, incl co2's high-freq modes above the 5-40 kHz descriptor band, become visible.
        logp = np.log1p(np.clip(prof_view, 0, None))
        bg = np.median(logp, axis=0, keepdims=True)                      # per-freq temporal background
        img = logp - bg                                                  # mode anomaly (any freq)
        vmin = 0.0; vmax = float(np.percentile(img, 99.5)) + 1e-6
        ext = [0, T_present, 0, Fbins * KHZ_PER_BIN]                     # y spans 0-250 kHz
        fig, ax = plt.subplots(1, 1, figsize=(13, 5))
        ax.imshow(img.T, origin="lower", aspect="auto", cmap="magma",
                  vmin=vmin, vmax=vmax, extent=ext)
        # descriptor-band guides: 5 kHz (DG_LO) and 40 kHz (DG_LO+72)
        lo_khz = DG_LO * KHZ_PER_BIN; hi_khz = (DG_LO + 72) * KHZ_PER_BIN
        ax.axhline(lo_khz, color="cyan", linestyle="--", lw=1.2,
                   label="descriptor band (5-40 kHz)")
        ax.axhline(hi_khz, color="cyan", linestyle="--", lw=1.2)
        ax.set_ylabel("freq (kHz)"); ax.set_xlabel("window (data-present)")
        ax.set_xlim(0, T_present)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.set_title(
            f"{name} FULL-FREQ GT spectrogram — shot {shot}, step {step} "
            f"(descriptor sees only 5-40 kHz dashed band)", loc="left", fontsize=10)
        fig.text(0.01, 0.006,
                 "Where do this modality's modes actually sit? Dashed = the 5-40 kHz the descriptor "
                 "head forecasts; everything above is invisible to the descriptor instrument.",
                 fontsize=8, wrap=True)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        outp = outdir / f"{name}_FULLFREQ_{shot}_step{step}.png"
        fig.savefig(outp, dpi=140); plt.close(fig)
        # report where the dominant (time-averaged) energy sits
        peak_bin = int(np.clip(prof_view, 0, None).mean(axis=0).argmax())
        peak_khz = peak_bin * KHZ_PER_BIN
        print(f"[fullfreq] {name}: {outp}  T={Tall} data_present={n_dp} "
              f"peak_freq={peak_khz:.1f}kHz (bin {peak_bin})", flush=True)
    print(f"[fullfreq] done -> {outdir}", flush=True)


def render_detector_validation(model, core, ckpt, spec_heads, spec_mods, args, device):
    """Detector-QC figure (ONE panel/modality): marks on the GT prominence ridge, human-verifiable.

    Its ONLY job is to let a human eyeball whether the hardened `_detect` fires where (and only
    where) there is a visible burst. NO skill number (ftol/ftp) is computed anywhere here.
    """
    shot = args.figure_shot
    assert shot, "render_detector_validation requires --figure_shot"
    stats = torch.load(args.stats_path, weights_only=False)
    diag_names = [c.name for c in core.diagnostics]; act_names = [c.name for c in core.actuators]
    f = Path(args.data_dir) / f"{shot}_processed.h5"; assert f.exists(), f"no shot file {f}"
    R = _process_shot(model, core, spec_heads, spec_mods, f, stats, diag_names, act_names, args, device)
    outdir = Path(args.figure_out); outdir.mkdir(parents=True, exist_ok=True)
    step = ckpt.get("step", ckpt.get("global_step", "?")); BROADBAND = {"co2"}
    for name in spec_mods:
        if name not in R:
            continue
        if name in BROADBAND:
            # BROADBAND (co2) QC: no ridge peak — render the prominence field, grey out padding,
            # and shade band-power-ACTIVE windows cyan (distributional detector). Eyeball check:
            # do the cyan windows line up with visible broadband brightening?
            d = R[name]
            gt_pd = d["gt_pd"]; energy = d["energy"]
            T = len(energy); NF = gt_pd.shape[1]
            bb = _detect_broadband(gt_pd, energy, fire_pct=args.fire_pct)
            active = bb["active"]; data_present = bb["data_present"]
            n_active = int(active.sum()); n_dp = int(data_present.sum())

            gp = np.clip(gt_pd, 0, None)
            vhi = float(np.percentile(gp, 99.5)) + 1e-6
            ext = [0, T, DG_LO * KHZ_PER_BIN, (DG_LO + NF) * KHZ_PER_BIN]
            fig, ax = plt.subplots(1, 1, figsize=(13, 5))
            ax.imshow(gp.T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=vhi, extent=ext)
            # shade non-data-present padding grey
            for i in np.where(~data_present)[0]:
                ax.axvspan(i - 0.5, i + 0.5, color="lightgrey", alpha=0.30, lw=0)
            # shade band-power-active windows translucent cyan (detected; no peak marks)
            first = True
            for i in np.where(active)[0]:
                ax.axvspan(i - 0.5, i + 0.5, color="cyan", alpha=0.18, lw=0,
                           label="band-power active" if first else None)
                first = False
            ax.set_ylabel("freq (kHz)"); ax.set_xlabel("window"); ax.set_xlim(0, T)
            if n_active:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
            ax.set_title(
                f"{name} DETECTOR QC (broadband, band-power activity) — shot {shot}, step {step}: "
                f"{n_active}/{n_dp} active windows", loc="left", fontsize=10)
            fig.text(0.01, 0.006,
                     "Eyeball check: does every mark sit on a visible burst, and does every visible "
                     "burst get a mark? NO skill number computed.", fontsize=8, wrap=True)
            fig.tight_layout(rect=[0, 0.04, 1, 1])
            outp = outdir / f"{name}_DETECTORQC_{shot}_step{step}.png"
            fig.savefig(outp, dpi=140); plt.close(fig)
            print(f"[detqc] {name}: {outp}  (broadband) active={n_active}/{n_dp} data-present",
                  flush=True)
            continue
        d = R[name]
        gt_pd = d["gt_pd"]; gt_fire = d["gt_fire"]; energy = d["energy"]; pers_pk = d["pers_peak"]
        T = len(gt_fire); NF = gt_pd.shape[1]
        det = _detect(gt_pd, gt_fire, energy, pers_pk)
        const_bins = det["const_bins"]; data_present = det["data_present"]
        peaks_pw = det["peaks_pw"]

        gp = np.clip(gt_pd, 0, None)
        vhi = float(np.percentile(gp, 99.5)) + 1e-6
        ext = [0, T, DG_LO * KHZ_PER_BIN, (DG_LO + NF) * KHZ_PER_BIN]
        fig, ax = plt.subplots(1, 1, figsize=(13, 5))
        ax.imshow(gp.T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=vhi, extent=ext)

        # shade the non-data-present region so padding is obvious
        for i in np.where(~data_present)[0]:
            ax.axvspan(i - 0.5, i + 0.5, color="lightgrey", alpha=0.30, lw=0)

        # marks: EVERY multi-peak detection (window, abs_bin) — single colour, cyan w/ black edge
        if peaks_pw:
            px = [w for (w, b) in peaks_pw]
            py = [b * KHZ_PER_BIN for (w, b) in peaks_pw]
            ax.plot(px, py, "o", color="cyan", markeredgecolor="black", markersize=4,
                    linestyle="none", label="detected peak")
        n_windows_with_peaks = len({w for (w, b) in peaks_pw})

        # excluded pickup lines
        for k, b in enumerate(const_bins):
            ax.axhline(b * KHZ_PER_BIN, color="grey", linestyle="--", lw=1.0,
                       label="excluded pickup" if k == 0 else None)

        ax.set_ylabel("freq (kHz)"); ax.set_xlabel("window")
        ax.set_xlim(0, T)
        if peaks_pw or const_bins:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.set_title(
            f"{name} DETECTOR QC — shot {shot}, step {step}: {len(peaks_pw)} peak-marks "
            f"in {n_windows_with_peaks} windows, {len(const_bins)} pickup lines excluded",
            loc="left", fontsize=10)
        fig.text(0.01, 0.006,
                 "Eyeball check: does every mark sit on a visible burst, and does every visible "
                 "burst get a mark? NO skill number computed.", fontsize=8, wrap=True)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        outp = outdir / f"{name}_DETECTORQC_{shot}_step{step}.png"
        fig.savefig(outp, dpi=140); plt.close(fig)
        print(f"[detqc] {name}: {outp}  peak_marks={len(peaks_pw)} "
              f"windows_with_peaks={n_windows_with_peaks} "
              f"pickup_excluded={len(const_bins)}", flush=True)
    print(f"[detqc] done -> {outdir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="/lustre/orion/fus187/proj-shared/foundation_model")
    ap.add_argument("--stats_path",
                    default="/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    ap.add_argument("--n_shots", type=int, default=40)
    ap.add_argument("--n_batches", type=int, default=200, help="cap total batches across shots")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--chunk_duration_s", type=float, default=0.05)
    ap.add_argument("--prediction_horizon_s", type=float, default=0.05,
                    help="dataset target horizon; 0.05 = one window = the K=1 run's config")
    ap.add_argument("--warmup_s", type=float, default=1.0)
    ap.add_argument("--anchor_beta", type=float, default=6.0,
                    help="_abeta at eval; ANCHOR_BETA_HOLDS=6 held to 100k steps -> 6.0 now")
    ap.add_argument("--tol_bins", type=int, default=2)
    ap.add_argument("--fire_pct", type=float, default=75.0)
    ap.add_argument("--out", default="analysis/mode_audit/descriptor_stratified_eval.json")
    ap.add_argument("--figure_shot", default="",
                    help="if set, render the V1 descriptor-track overlay figure for this single shot "
                         "(GT ridge + model forecast + persistence) instead of the 40-shot aggregate")
    ap.add_argument("--figure_out", default="eval_runs/descriptor_track")
    ap.add_argument("--validate_detector", action="store_true",
                    help="render the human-verifiable DETECTOR-QC figure (marks on the GT ridge) "
                         "for --figure_shot; NO skill number computed. Requires --figure_shot.")
    ap.add_argument("--full_freq_view", action="store_true",
                    help="render the FULL-FREQ (0-250 kHz) GT-spectrogram view per spectro modality "
                         "for --figure_shot; GT-only diagnostic showing where modes sit vs the "
                         "5-40 kHz descriptor band. NO skill number computed. Requires --figure_shot.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[desc-eval] device={device}  ckpt={args.ckpt}", flush=True)

    model, ckpt = load_model(Path(args.ckpt), device)
    model.eval()
    core = _core(model)
    diag_names = [c.name for c in core.diagnostics]
    act_names = [c.name for c in core.actuators]
    spec_heads = getattr(core, "spec_descriptor_heads", {}) or {}
    spec_mods = list(spec_heads.keys())
    step = ckpt.get("step", ckpt.get("global_step", "?"))
    print(f"[desc-eval] step={step}  spectro descriptor heads: {spec_mods}", flush=True)
    if not spec_mods:
        print("[desc-eval] NO descriptor heads on this checkpoint — nothing to measure."); return

    if args.full_freq_view:
        render_full_freq_view(model, core, ckpt, spec_heads, spec_mods, args, device); return

    if args.validate_detector:
        render_detector_validation(model, core, ckpt, spec_heads, spec_mods, args, device); return

    if args.figure_shot:
        render_track_figure(model, core, ckpt, spec_heads, spec_mods, args, device)
        return

    stats = torch.load(args.stats_path, weights_only=False)
    files = sorted(Path(args.data_dir).glob("*_processed.h5"))
    # deterministic val-ish sample from the TAIL (trainer splits val off the tail-fraction);
    # take a spread so we hit shots with active modes.
    files = files[-max(args.n_shots * 3, args.n_shots):]
    files = files[:args.n_shots]
    VALID_MODALITY = "ece"  # only ece's real modes fall inside the 5-40 kHz descriptor band
    STRATA = ("detected", "stable", "transition")
    print(f"[desc-eval] {len(files)} shots; horizon={args.prediction_horizon_s}s; "
          f"VALIDATED detector (_detect: data-present mask + edge-guard + presence/dilated "
          f"pickup exclusion + drift-tolerant ridge tracking); strata = {STRATA}",
          flush=True)

    # per-modality, per-stratum pooled peak arrays (all ABSOLUTE freq bins), + per-stratum counts.
    # GT peak = _detect's CLEANED peak (edge-guarded + pickup-masked), NOT R[name]["gt_peak"].
    pool = {m: {s: {"model_peak": [], "gt_peak": [], "pers_peak": []} for s in STRATA}
            for m in spec_mods}
    n_strat = {m: {s: 0 for s in STRATA} for m in spec_mods}
    n_data_present = {m: 0 for m in spec_mods}

    for si, f in enumerate(files):
        R = _process_shot(model, core, spec_heads, spec_mods, f, stats,
                          diag_names, act_names, args, device)
        for name in spec_mods:
            if name not in R:
                continue
            d = R[name]
            # VALIDATED detector on the RAW prominence profiles: returns cleaned peak + strata.
            det = _detect(d["gt_pd"], d["gt_fire"], d["energy"], d["pers_peak"],
                          fire_pct=args.fire_pct)
            det_peak = det["peak"]                       # CLEANED GT peak (edge-guard + pickup masked)
            mdl_pk = d["model_peak"]; pers_pk = d["pers_peak"]
            n_data_present[name] += int(det["data_present"].sum())
            strat_masks = {"detected": det["detected"],
                           "stable": det["stable"], "transition": det["transition"]}
            for s in STRATA:
                m = strat_masks[s]
                n_strat[name][s] += int(m.sum())
                if int(m.sum()) == 0:
                    continue
                pool[name][s]["model_peak"].append(mdl_pk[m])
                pool[name][s]["gt_peak"].append(det_peak[m])
                pool[name][s]["pers_peak"].append(pers_pk[m])
        if (si + 1) % 10 == 0:
            print(f"[desc-eval] {si + 1}/{len(files)} shots", flush=True)

    tol = args.tol_bins
    print(f"\n[desc-eval] step={step}. STRATIFIED verdict per modality via the VALIDATED detector "
          f"(GT-mode-freq forecast within ±{tol} bins).\n"
          f"[desc-eval] Strata: detected=all validated-ridge windows; stable=peak persists near "
          f"input; transition=onset/death (persistence CANNOT copy it → the forecast-skill test).\n",
          flush=True)
    report = {"step": str(step), "fire_pct": args.fire_pct, "tol_bins": tol,
              "valid_modality": VALID_MODALITY, "modalities": {}}
    for name in spec_mods:
        valid = (name == VALID_MODALITY)
        if valid:
            hdr = f"=== {name}  [gated ece — VALID BAND] ==="
        else:
            hdr = (f"=== {name}  [gated {name} — INVALID: modes are 100-250kHz, out of "
                   f"5-40kHz descriptor band; reported for completeness only] ===")
        print(hdr)
        modrec = {"valid": valid, "strata": {}, "n_data_present": int(n_data_present[name])}
        for s in STRATA:
            n = int(n_strat[name][s])
            if not pool[name][s]["model_peak"]:
                print(f"    {s:11s}: no windows (n={n})")
                modrec["strata"][s] = {"ftol": None, "ftp": None, "delta": None, "n": n}
                continue
            mp = np.concatenate(pool[name][s]["model_peak"])
            gp = np.concatenate(pool[name][s]["gt_peak"])
            pp = np.concatenate(pool[name][s]["pers_peak"])
            ftol = float((np.abs(mp - gp) <= tol).mean())
            ftp = float((np.abs(pp - gp) <= tol).mean())
            delta = ftol - ftp
            # "MODEL BEATS PERSISTENCE" is only meaningful for the VALID (ece) band.
            flag = "  <-- MODEL BEATS PERSISTENCE" if (delta > 0.02 and valid) else ""
            print(f"    {s:11s}: ftol(model)={ftol:.3f} ftp(pers)={ftp:.3f} "
                  f"Δ={delta:+.3f} n={n}{flag}")
            modrec["strata"][s] = {"ftol": ftol, "ftp": ftp, "delta": delta, "n": n}
        if valid:
            print("    (TRANSITION is the forecast-skill test: persistence MUST fail there, so a "
                  "positive Δ on transition is genuine skill.)")
        report["modalities"][name] = modrec
        print()

    print("[desc-eval] NOTE: ece is the ONLY valid descriptor-band measurement (its real modes sit "
          "in 5-40 kHz). mhr/co2/bes modes live at 100-250 kHz, OUTSIDE this band — their numbers "
          "are wrong-band artifacts; they need the full-freq code-head instrument, not the "
          "descriptor head.\n", flush=True)

    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(f"[desc-eval] wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
