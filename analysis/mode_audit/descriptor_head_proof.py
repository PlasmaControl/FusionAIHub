"""FACTORIZATION PROOF: can a descriptor readout off the (frozen) forecasting
backbone predict the NEXT window's mode-band profile — on a HELD-OUT shot?

The codes are not forecastable (audit + dist_gate: freq_in_tol 0.0-0.12). The
DESCRIPTOR (mode-band power profile) IS (pre-gate: 0.75-0.95). This trains a small
readout head on FROZEN backbone tokens to forecast the descriptor, then compares to
the persistence baseline and renders the predicted vs GT mode ridge. If the model
matches/beats persistence on a held-out shot -> the backbone forecasts modes and the
factorization head is the fix (no backbone retrain needed -- just a descriptor head).

Setup mirrors measure_modecode_rate.py (load_model + build_datasets + forward_batch).
Backbone frozen (eval); only the readout head trains. Env:
  CKPT, SHOTS_TRAIN, SHOTS_VAL, MAX_WIN_TRAIN, MAX_WIN_VAL, STEPS, OUT_DIR.
"""
import os
import sys
from pathlib import Path

FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training", f"{FMH}/analysis/mode_audit"):
    if p not in sys.path:
        sys.path.insert(0, p)
import json
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_e2e_animation_tokamak import load_model
from train_e2e_stage1 import build_datasets, forward_batch, _core
from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.e2e.output_heads import SpectrogramCodeHead, SpectrogramMaskGITHead
from dist_gate import MODE_LO, MODE_HI, TOL_BINS

CKPT = Path(sys.argv[1] if len(sys.argv) > 1
            else "/lustre/orion/fus187/proj-shared/models/e2e_step2_fsq_finer/e2e_stage1_latest.pt")
SHOTS_TRAIN = os.environ.get("SHOTS_TRAIN", "200729,190996,204811").split(",")
SHOTS_VAL = os.environ.get("SHOTS_VAL", "191001").split(",")
MAX_WIN_TRAIN = int(os.environ.get("MAX_WIN_TRAIN", "400"))
MAX_WIN_VAL = int(os.environ.get("MAX_WIN_VAL", "200"))
STEPS = int(os.environ.get("STEPS", "1500"))
TCOL = 6
NF = MODE_HI - MODE_LO
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/eval_runs/descriptor_proof")); OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, ckpt = load_model(CKPT, device); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
a = ckpt["args"]; core = _core(model)
diag_names = [d["name"] for d in ckpt["diagnostics"]]
act_names = [c["name"] for c in ckpt["actuators"]]
data_dir = Path(a["data_dir"]); stats = torch.load(a["stats_path"], weights_only=False)
spec = [n for n in diag_names if isinstance(core.diag_heads[n], (SpectrogramCodeHead, SpectrogramMaskGITHead))]
print(f"ckpt step={ckpt.get('step')} | spectro heads {spec} | train {SHOTS_TRAIN} val {SHOTS_VAL}", flush=True)

train_files = [data_dir / f"{s}_processed.h5" for s in SHOTS_TRAIN if (data_dir / f"{s}_processed.h5").exists()]
val_files = [data_dir / f"{s}_processed.h5" for s in SHOTS_VAL if (data_dir / f"{s}_processed.h5").exists()]
cache = Path(os.environ.get("CACHE_DIR", f"{FMH}/eval_runs/descr_cache"))  # per-job override avoids
cache.mkdir(parents=True, exist_ok=True)                                   # concurrent lengths-cache write races
tr_ds, va_ds = build_datasets(data_dir, train_files, val_files, stats,
                              a["chunk_duration_s"], a.get("prediction_horizon_s", a["chunk_duration_s"]),
                              a["step_size_s"], a["warmup_s"], diag_names, act_names, cache)


def descr(x_np):
    """(C,F,T) spectrogram -> (NF, TCOL) channel-max mode-band residual profile."""
    aa = np.abs(x_np)
    r = aa - ndi.gaussian_filter1d(aa, 6.0, axis=1)      # per-freq baseline subtract
    cmax = r.max(0)[MODE_LO:MODE_HI]                     # (NF, T) channel-max residual, mode band
    cols = np.array_split(np.arange(cmax.shape[1]), TCOL)
    return np.stack([cmax[:, c].mean(1) for c in cols], axis=1)   # (NF, TCOL)


def collect(ds, max_win):
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn, drop_last=False)
    TOK = {n: [] for n in spec}; TGT = {n: [] for n in spec}; INP = {n: [] for n in spec}
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= max_win:
                break
            _, diag_inputs, targets, _, tok = forward_batch(model, batch, device)
            for n in spec:
                TOK[n].append(tok[n].float().cpu())
                g = targets[n].float().cpu().numpy(); ii = diag_inputs[n].float().cpu().numpy()
                TGT[n].append(torch.tensor(np.stack([descr(g[b]) for b in range(g.shape[0])])))   # (B,NF,TCOL)
                INP[n].append(torch.tensor(np.stack([descr(ii[b]) for b in range(ii.shape[0])])))
            seen += targets[spec[0]].shape[0]
    return ({n: torch.cat(TOK[n]) for n in spec}, {n: torch.cat(TGT[n]) for n in spec},
            {n: torch.cat(INP[n]) for n in spec})


class DHead(nn.Module):
    """Frozen-backbone tokens (B, n_tok, d) -> descriptor (B, NF, TCOL). Order-agnostic:
    per-token projection then a global MLP (fixed token order, learned mapping)."""
    def __init__(self, n_tok, d, nf, tcol):
        super().__init__()
        pdrop = float(os.environ.get("DROPOUT", "0.1"))
        self.tp = nn.Linear(d, 8)
        self.mlp = nn.Sequential(nn.Linear(n_tok * 8, 512), nn.GELU(), nn.Dropout(pdrop),
                                 nn.Linear(512, 512), nn.GELU(), nn.Dropout(pdrop),
                                 nn.Linear(512, nf * tcol))
        self.nf, self.tcol = nf, tcol

    def forward(self, tok):
        B = tok.shape[0]
        h = self.tp(tok).reshape(B, -1)
        return self.mlp(h).reshape(B, self.nf, self.tcol)


def peak_in_tol(pred, tgt):
    """fraction of (window,col) where pred's peak-freq bin is within TOL of tgt's."""
    pf = pred.argmax(1); tf = tgt.argmax(1)               # (N,TCOL)
    return float((np.abs(pf - tf) <= TOL_BINS).mean())


def prof_corr(pred, tgt):
    v = []
    for i in range(pred.shape[0]):
        for c in range(pred.shape[2]):
            x, y = pred[i, :, c], tgt[i, :, c]
            if x.std() > 1e-9 and y.std() > 1e-9:
                v.append(np.corrcoef(x, y)[0, 1])
    return float(np.nanmedian(v)) if v else float("nan")


def eval_trained():
    """EVAL_TRAINED=1: use the model's OWN trained descriptor head (loaded via
    load_model) to forecast the held-out mode ridge — the deliverable render."""
    import json
    core_heads = getattr(core, "spec_descriptor_heads", {})
    specs = [n for n in spec if n in core_heads]
    if not specs:
        print("[eval] no trained descriptor heads in ckpt", flush=True); return
    anchored = bool(ckpt["args"].get("spec_descriptor_anchor", False))
    beta = float(os.environ.get("DESC_ANCHOR_BETA", ckpt["args"].get("spec_descriptor_dist_beta", 4.0)))
    print(f"[eval] anchored={anchored} beta={beta}", flush=True)
    loader = DataLoader(va_ds, batch_size=8, shuffle=False, num_workers=2,
                        collate_fn=collate_fn, drop_last=False)
    acc = {n: {"pred": [], "gt": [], "pers": []} for n in specs}
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= MAX_WIN_VAL:
                break
            _, diag_inputs, targets, _, tok = forward_batch(model, batch, device)
            for n in specs:
                h = core_heads[n]
                raw = h(tok[n])
                if raw.dim() == 4:                              # multi-horizon head (B,H,NF,TCOL) -> longest horizon
                    raw = raw[:, -1]
                if anchored:                                    # mirror the training-time persistence anchor
                    anc = h.descriptor_target(diag_inputs[n].float())
                    anc = anc / anc.amax(dim=1, keepdim=True).clamp_min(1e-6)
                    raw = anc * beta + raw
                acc[n]["pred"].append(raw.cpu())
                acc[n]["gt"].append(h.descriptor_target(targets[n].float()).cpu())
                acc[n]["pers"].append(h.descriptor_target(diag_inputs[n].float()).cpu())
            seen += targets[specs[0]].shape[0]
    results = {}
    for n in specs:
        P = torch.cat(acc[n]["pred"]).numpy(); G = torch.cat(acc[n]["gt"]).numpy(); I = torch.cat(acc[n]["pers"]).numpy()
        gtprom = G.max(1).max(1) - np.median(G.reshape(G.shape[0], -1), axis=1)
        ai = np.where(gtprom >= np.percentile(gtprom, 60))[0]
        m = {"active_model": {"peak_in_tol": peak_in_tol(P[ai], G[ai]), "prof_corr": prof_corr(P[ai], G[ai])},
             "active_persistence": {"peak_in_tol": peak_in_tol(I[ai], G[ai]), "prof_corr": prof_corr(I[ai], G[ai])},
             "all_model": {"peak_in_tol": peak_in_tol(P, G), "prof_corr": prof_corr(P, G)},
             "n_active": int(len(ai))}
        results[n] = m
        print(f"[eval {n}] TRAINED HELD-OUT (ACTIVE n={len(ai)}): model peak_in_tol="
              f"{m['active_model']['peak_in_tol']:.3f} prof_corr={m['active_model']['prof_corr']:.3f} | "
              f"persistence peak_in_tol={m['active_persistence']['peak_in_tol']:.3f} "
              f"prof_corr={m['active_persistence']['prof_corr']:.3f}", flush=True)
        # --- RENDER-vs-METRIC DIAGNOSTIC: is the model DISTRIBUTION peaked (samplable)
        # or FLAT (degenerate: argmax metric-equivalent to persistence, but samples would
        # NOT reproduce the ridge)? Compare per-window freq-entropy + argmax-trace.
        NF = P.shape[1]

        def _sm(x):                                    # softmax over freq (axis 1)
            x = x - x.max(axis=1, keepdims=True); e = np.exp(x); return e / (e.sum(axis=1, keepdims=True) + 1e-12)
        mdist = _sm(P)                                 # model distribution over freq
        pdist = _sm((I / (I.max(axis=1, keepdims=True) + 1e-6)) * beta)   # persistence as a distribution (same beta)
        _ent = lambda d: -(d * np.log(d + 1e-12)).sum(axis=1)             # (N,TCOL) freq-entropy
        ent_m = _ent(mdist)[ai].mean(axis=1); ent_p = _ent(pdist)[ai].mean(axis=1)
        argmatch = float(np.mean(P[ai].argmax(1) == I[ai].argmax(1)))
        m["entropy_model_median"] = float(np.median(ent_m))
        m["entropy_persistence_median"] = float(np.median(ent_p))
        m["entropy_uniform"] = float(np.log(NF))
        m["argmax_match_model_vs_pers"] = argmatch
        m["distributionally_degenerate"] = bool(np.median(ent_m) > 0.85 * np.log(NF) and argmatch > 0.8)
        print(f"[diag {n}] freq-entropy model={np.median(ent_m):.3f} persistence={np.median(ent_p):.3f} "
              f"(uniform={np.log(NF):.3f}) | argmax-match(model,pers)={argmatch:.3f} => "
              f"{'DEGENERATE (flat dist, argmax=persistence -> NOT samplable)' if m['distributionally_degenerate'] else 'distribution peaked'}",
              flush=True)
        pk = lambda D: D.mean(2).argmax(1)             # per-window peak bin (tcol-avg profile)
        figd, axd = plt.subplots(2, 1, figsize=(13, 6))
        axd[0].plot(pk(G)[ai], "k.", ms=5, label="GT"); axd[0].plot(pk(P)[ai], "r.", ms=3, label="model"); axd[0].plot(pk(I)[ai], "b.", ms=2, label="persistence")
        axd[0].set_ylabel("peak freq bin"); axd[0].set_title(f"{n} argmax-trace (active)"); axd[0].legend(fontsize=8)
        axd[1].plot(ent_m, "r-", label=f"model (med {np.median(ent_m):.2f})"); axd[1].plot(ent_p, "b-", label=f"persistence (med {np.median(ent_p):.2f})")
        axd[1].axhline(np.log(NF), color="gray", ls="--", label=f"uniform ({np.log(NF):.2f})")
        axd[1].set_ylabel("freq entropy"); axd[1].set_xlabel("active-window idx"); axd[1].legend(fontsize=8)
        figd.suptitle(f"{n} DIST DIAGNOSTIC — {CKPT.parent.name}"); figd.tight_layout()
        figd.savefig(OUT / f"{n}_distdiag.png", dpi=120); plt.close(figd)
        print(f"[diag {n}] saved {OUT}/{n}_distdiag.png", flush=True)
        # --- SKILL METRICS where persistence is STRUCTURALLY BLIND (from P/G/I) ---
        # window semantics: I=descriptor(input=now), G=descriptor(target=+horizon), P=model pred of target.
        def wprom(D): d = D.mean(2); return d.max(1) - np.median(d, axis=1)   # (N,) per-window prominence
        def wpk(D): return D.mean(2).argmax(1)                                # (N,) per-window peak bin
        thr = float(np.percentile(wprom(G), 60))
        pres_i = wprom(I) > thr; pres_g = wprom(G) > thr; pres_p = wprom(P) > thr
        onset = (~pres_i) & pres_g; death = pres_i & (~pres_g)               # transitions over the horizon
        pkP, pkG, pkI = wpk(P), wpk(G), wpk(I)
        rate = lambda mk, cond: float(cond[mk].mean()) if mk.sum() else float("nan")
        m["n_onset"] = int(onset.sum()); m["n_death"] = int(death.sum())
        # onset: mode ABSENT now -> PRESENT at horizon. Persistence (copies now) always says absent -> recall 0.
        m["onset_recall_model"] = rate(onset, pres_p & (np.abs(pkP - pkG) <= TOL_BINS))
        m["onset_recall_model_presence_only"] = rate(onset, pres_p)
        m["onset_recall_persistence"] = rate(onset, pres_i)                  # =0 by construction
        m["death_recall_model"] = rate(death, ~pres_p)
        m["death_recall_persistence"] = rate(death, ~pres_i)                 # =0 by construction
        # drift-direction on windows where GT actually moved > tol; persistence predicts 0 drift always.
        dG = pkG - pkI; dP = pkP - pkI; moved = np.abs(dG) > TOL_BINS
        m["n_drift"] = int(moved.sum())
        m["drift_dir_acc_model"] = rate(moved, np.sign(dP) == np.sign(dG))   # chance 0.5
        print(f"[skill {n}] ONSET n={m['n_onset']}: recall model={m['onset_recall_model']:.3f} "
              f"(presence-only {m['onset_recall_model_presence_only']:.3f}) vs persistence "
              f"{m['onset_recall_persistence']:.3f} | DEATH n={m['n_death']}: model={m['death_recall_model']:.3f} "
              f"vs pers {m['death_recall_persistence']:.3f} | DRIFT n={m['n_drift']}: dir-acc "
              f"model={m['drift_dir_acc_model']:.3f} (chance 0.5, pers=0)", flush=True)
        h = core_heads[n]
        rg = lambda D: np.concatenate([D[i] for i in range(min(D.shape[0], 60))], axis=1)
        Rg, Rm, Rp = rg(G), rg(P), rg(I)
        vlo, vhi = np.percentile(Rg, 2), np.percentile(Rg, 99)
        khz = np.arange(h.mode_lo, h.mode_hi) * (500000.0 / 1024 / 1e3)
        fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
        for a2, (ttl, R) in zip(ax, [("GT mode ridge", Rg),
                                     (f"MODEL forecast (peak_in_tol {m['active_model']['peak_in_tol']:.2f})", Rm),
                                     (f"PERSISTENCE ({m['active_persistence']['peak_in_tol']:.2f})", Rp)]):
            a2.imshow(R, origin="lower", aspect="auto", vmin=vlo, vmax=vhi, cmap="magma",
                      extent=[0, R.shape[1], khz[0], khz[-1]])
            a2.set_ylabel(ttl + "\nkHz", fontsize=8)
        ax[-1].set_xlabel("window-col (time)")
        fig.suptitle(f"{n} TRAINED descriptor forecast — held-out {SHOTS_VAL} (ckpt {CKPT.parent.name})")
        fig.tight_layout(); fig.savefig(OUT / f"{n}_trained_ridge.png", dpi=120); plt.close(fig)
        print(f"[eval {n}] saved {OUT}/{n}_trained_ridge.png", flush=True)
    json.dump(results, open(OUT / "descriptor_eval_trained.json", "w"), indent=2,
              default=lambda o: float(o) if hasattr(o, "item") else o)
    print(f"[eval] wrote {OUT}/descriptor_eval_trained.json", flush=True)


def onset_skill_eval():
    """ONSET_EVAL=1: rigorous onset/death forecasting skill. Per-shot GT presence
    timeline -> K-window HYSTERESIS (absent>=K then present>=K = physical onset, not
    detector flicker) -> pooled across many shots -> model recall + binomial 95% CI +
    SHUFFLED-chance + persistence (=0 by construction). Event-mining is detector-only,
    so pooling shots is free. Env: SHOTS_VAL (comma list), ONSET_K (default 3)."""
    import json
    from torch.utils.data import DataLoader
    core_heads = getattr(core, "spec_descriptor_heads", {})
    specs = [n for n in spec if n in core_heads]
    if not specs:
        print("[onset] no trained descriptor heads", flush=True); return
    anchored = bool(ckpt["args"].get("spec_descriptor_anchor", False))
    beta = float(os.environ.get("DESC_ANCHOR_BETA", ckpt["args"].get("spec_descriptor_dist_beta", 4.0)))
    K = int(os.environ.get("ONSET_K", "3"))
    shots = [s for s in SHOTS_VAL if (data_dir / f"{s}_processed.h5").exists()]
    print(f"[onset] anchored={anchored} K={K} shots={len(shots)}", flush=True)
    ev = {n: {"on_hit": [], "on_raw": [], "on_pk": [], "on_gtpk": [], "de_hit": [], "de_raw": [],
              "cont_fd": [], "dr_gt": [], "dr_m": []} for n in specs}
    for sh in shots:
        f = data_dir / f"{sh}_processed.h5"
        _, sds = build_datasets(data_dir, [f], [f], stats, a["chunk_duration_s"],
                                a.get("prediction_horizon_s", a["chunk_duration_s"]),
                                a["step_size_s"], a["warmup_s"], diag_names, act_names, cache)
        loader = DataLoader(sds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn, drop_last=False)
        Pd = {n: [] for n in specs}; Gd = {n: [] for n in specs}; Id = {n: [] for n in specs}
        seen = 0
        with torch.no_grad():
            for batch in loader:
                if seen >= MAX_WIN_VAL:
                    break
                _, di, tg, _, tok = forward_batch(model, batch, device)
                for n in specs:
                    h = core_heads[n]; raw = h(tok[n])
                    if raw.dim() == 4:                          # multi-horizon head -> longest horizon
                        raw = raw[:, -1]
                    if anchored:
                        anc = h.descriptor_target(di[n].float()); anc = anc / anc.amax(1, keepdim=True).clamp_min(1e-6)
                        raw = anc * beta + raw
                    Pd[n].append(raw.cpu()); Gd[n].append(h.descriptor_target(tg[n].float()).cpu())
                    Id[n].append(h.descriptor_target(di[n].float()).cpu())
                seen += tg[specs[0]].shape[0]
        for n in specs:
            P = torch.cat(Pd[n]).numpy(); G = torch.cat(Gd[n]).numpy(); I = torch.cat(Id[n]).numpy()
            wp = lambda D: (lambda d: d.max(1) - np.median(d, 1))(D.mean(2)); pk = lambda D: D.mean(2).argmax(1)
            thr = float(np.percentile(wp(G), 60))
            pg = wp(G) > thr; pp = wp(P) > thr; pi = wp(I) > thr; pkP, pkG, pkI = pk(P), pk(G), pk(I); T = len(pg)
            for t in range(K, T - K):
                if (not pg[t - K:t].any()) and pg[t:t + K].all():                 # HYSTERESIS onset
                    ev[n]["on_hit"].append(bool(pp[t] and abs(pkP[t] - pkG[t]) <= TOL_BINS))
                    ev[n]["on_raw"].append(bool(abs(pkI[t] - pkG[t]) <= TOL_BINS))  # RAW-persistence NULL (unthresholded input argmax)
                    ev[n]["on_pk"].append(int(pkP[t])); ev[n]["on_gtpk"].append(int(pkG[t]))
                if pg[t - K:t].all() and (not pg[t:t + K].any()):                 # HYSTERESIS death
                    ev[n]["de_hit"].append(bool(not pp[t]))                        # model predicts absent
                    ev[n]["de_raw"].append(bool(not pi[t]))                        # raw-copy predicts absent (input present -> ~0)
                if pg[t - K:t].all() and pg[t:t + K].all():                        # SUSTAINED present (control)
                    ev[n]["cont_fd"].append(bool(not pp[t]))                       # FALSE-death: model wrongly says absent
                if abs(pkG[t] - pkI[t]) > TOL_BINS:                                # GT drifted -> drift event
                    ev[n]["dr_gt"].append(int(np.sign(pkG[t] - pkI[t])))
                    _dm = pkP[t] - pkI[t]; ev[n]["dr_m"].append(int(np.sign(_dm)) if abs(_dm) > TOL_BINS else 0)
    res = {}
    rng = np.random.RandomState(0)
    for n in specs:
        oh = np.array(ev[n]["on_hit"]); orw = np.array(ev[n]["on_raw"]); no = len(oh); nd = len(ev[n]["de_hit"])
        r = float(oh.mean()) if no else float("nan")
        r_raw = float(orw.mean()) if no else float("nan")               # RAW-persistence NULL (THE decisive baseline)
        ci = 1.96 * np.sqrt(r * (1 - r) / no) if no else float("nan")
        miss = ~orw; nbr = int(miss.sum())                              # onsets the raw copy MISSES
        r_br = float(oh[miss].mean()) if nbr else float("nan")          # model recall THERE = genuine-beyond-copy
        opk = np.array(ev[n]["on_pk"]); gpk = np.array(ev[n]["on_gtpk"])
        sh_r = float(np.mean([np.abs(rng.permutation(opk) - gpk) <= TOL_BINS for _ in range(200)])) if no else float("nan")
        rd = float(np.mean(ev[n]["de_hit"])) if nd else float("nan")
        rd_raw = float(np.mean(ev[n]["de_raw"])) if nd else float("nan")            # raw-copy death recall (~0)
        ncont = len(ev[n]["cont_fd"]); fd = float(np.mean(ev[n]["cont_fd"])) if ncont else float("nan")  # FALSE-death rate
        ci_de = 1.96 * np.sqrt(rd * (1 - rd) / nd) if nd else float("nan")           # binomial 95% CIs
        ci_fd = 1.96 * np.sqrt(fd * (1 - fd) / ncont) if ncont else float("nan")
        # EXIT RULE (CI-separated): death-recall LOWER bound > false-death UPPER bound => real, non-copyable signal
        ci_sep = bool(not np.isnan(ci_de) and not np.isnan(ci_fd) and (rd - ci_de) > (fd + ci_fd))
        death_verdict = ("REAL (CI-separated: death >> false-death, non-copyable)" if ci_sep
                         else "BIAS (death CI overlaps false-death = absent-bias)")
        dg = np.array(ev[n]["dr_gt"]); dm = np.array(ev[n]["dr_m"]); ndr = len(dg)   # THREE-WAY drift
        com = dm != 0; ncom = int(com.sum())
        dir_acc_com = float(np.mean(dm[com] == dg[com])) if ncom else float("nan")
        frac_none = float(np.mean(dm == 0)) if ndr else float("nan")
        res[n] = {"n_onset": no, "onset_model": r, "onset_model_ci95": ci,
                  "onset_RAW_persistence_NULL": r_raw, "onset_thresh_persistence": 0.0, "onset_shuffled": sh_r,
                  "onset_beyond_raw_recall": r_br, "n_onset_raw_miss": nbr,
                  "n_death": nd, "death_model": rd, "death_model_ci95": ci_de, "death_raw_persistence": rd_raw,
                  "death_persistence": 0.0, "n_sustained": ncont, "false_death_rate": fd, "false_death_ci95": ci_fd,
                  "death_ci_separated": ci_sep, "death_verdict": death_verdict,
                  "n_drift": ndr, "drift_dir_acc_committed": dir_acc_com, "drift_frac_model_none": frac_none,
                  "K": K, "n_shots": len(shots)}
        gap = (r - r_raw) if (no and not np.isnan(r_raw)) else float("nan")
        verdict = ("LEAKAGE: model≈raw-copy (detection, NOT forecasting)" if (not np.isnan(gap) and gap < 0.10)
                   else "FORECASTING: model>>raw-copy" if (not np.isnan(gap) and gap > 0.15) else "AMBIGUOUS")
        print(f"[onset {n}] n={no} K={K} {len(shots)}sh: model={r:.3f}±{ci:.3f} | RAW-persistence NULL={r_raw:.3f} "
              f"| thresh-pers=0 | shuffled={sh_r:.3f} || beyond-raw={r_br:.3f} (n_rawmiss={nbr})", flush=True)
        print(f"[onset {n}] DEATH n={nd} model={rd:.3f}±{ci_de:.3f} vs FALSE-death(sustained n={ncont})={fd:.3f}±{ci_fd:.3f} "
              f"raw={rd_raw:.3f} => {death_verdict} | DRIFT n={ndr}: dir-acc(committed)={dir_acc_com:.3f} "
              f"model-none={frac_none:.3f} (pers always-none) || ONSET: {verdict} (gap {gap:+.3f})", flush=True)
    json.dump(res, open(OUT / "onset_skill.json", "w"), indent=2, default=lambda o: float(o) if hasattr(o, "item") else o)
    print(f"[onset] wrote {OUT}/onset_skill.json", flush=True)


def horizon_probe():
    """HORIZON_PROBE=1: does the PERSISTENCE baseline decay at longer horizons, opening
    headroom for a t+4/t+8 retrain? DATA-driven (uses the head's descriptor_target for the
    current-window peak-freq time-series; the model is NOT used to predict — this measures the
    baseline + learnability, the prerequisites for the retrain). Consecutive dataset windows are
    step_size_s apart, so a full-window horizon N = N*round(chunk/step) samples. Reports per N:
    persistence peak_in_tol (=argmax within TOL_BINS over N windows), drift-fraction (mode moved
    >TOL), and MOMENTUM-match (of drifted windows, does the N-step drift direction match the
    PRIOR N-step drift direction — a data-only 'is the drift learnable' signal, chance 0.5).
    N=1 (50 ms) should ~reproduce the Gate-1 persistence 0.576. Env: SHOTS_VAL, HORIZONS
    (default '1,2,4,8' = 50/100/200/400 ms)."""
    import json
    from torch.utils.data import DataLoader
    core_heads = getattr(core, "spec_descriptor_heads", {})
    specs = [n for n in spec if n in core_heads]
    if not specs:
        print("[horizon] no trained descriptor heads", flush=True); return
    horizons = [int(x) for x in os.environ.get("HORIZONS", "1,2,4,8").split(",")]
    spw = max(1, round(a["chunk_duration_s"] / a["step_size_s"]))   # dataset windows per full-window step
    ms = lambda N: N * a["chunk_duration_s"] * 1000.0
    shots = [s for s in SHOTS_VAL if (data_dir / f"{s}_processed.h5").exists()]
    print(f"[horizon] shots={len(shots)} horizons={horizons} stride/window={spw} samples", flush=True)
    seqs = {n: [] for n in specs}                                  # per-shot current-window descriptor sequences
    for sh in shots:
        f = data_dir / f"{sh}_processed.h5"
        _, sds = build_datasets(data_dir, [f], [f], stats, a["chunk_duration_s"],
                                a.get("prediction_horizon_s", a["chunk_duration_s"]),
                                a["step_size_s"], a["warmup_s"], diag_names, act_names, cache)
        loader = DataLoader(sds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn, drop_last=False)
        Id = {n: [] for n in specs}; seen = 0
        with torch.no_grad():
            for batch in loader:
                if seen >= MAX_WIN_VAL:
                    break
                _, di, tg, _, _ = forward_batch(model, batch, device)
                for n in specs:
                    Id[n].append(core_heads[n].descriptor_target(di[n].float()).cpu())
                seen += tg[specs[0]].shape[0]
        for n in specs:
            seqs[n].append(torch.cat(Id[n]).numpy())               # (T, NF, TCOL)
    res = {}
    for n in specs:
        per_h = {}
        for N in horizons:
            off = N * spw
            pers_hits, drifts, mom_hits = [], [], []
            for I in seqs[n]:
                T = I.shape[0]
                if T <= off:
                    continue
                prof = I.mean(2)                                   # (T, NF) tcol-avg profile
                pk = prof.argmax(1)                                # (T,) peak bin
                wp = prof.max(1) - np.median(prof, 1)              # (T,) mode prominence
                thr = float(np.percentile(wp, 60))
                act = wp > thr
                for i in range(T - off):
                    if not act[i]:
                        continue
                    d = int(pk[i + off]) - int(pk[i])
                    pers_hits.append(abs(d) <= TOL_BINS)
                    drifted = abs(d) > TOL_BINS
                    drifts.append(drifted)
                    if drifted and i - off >= 0 and act[i - off]:  # prior N-step drift (momentum)
                        prev = int(pk[i]) - int(pk[i - off])
                        if abs(prev) > TOL_BINS:
                            mom_hits.append(int(np.sign(prev) == np.sign(d)))
            npt = len(pers_hits); nmo = len(mom_hits)
            pers = float(np.mean(pers_hits)) if npt else float("nan")
            dfr = float(np.mean(drifts)) if npt else float("nan")
            mom = float(np.mean(mom_hits)) if nmo else float("nan")
            ci_p = 1.96 * np.sqrt(pers * (1 - pers) / npt) if npt else float("nan")
            ci_m = 1.96 * np.sqrt(mom * (1 - mom) / nmo) if nmo else float("nan")
            per_h[str(N)] = {"horizon_ms": ms(N), "n_active": npt, "persistence_peak_in_tol": pers,
                             "persistence_ci95": ci_p, "drift_fraction": dfr, "n_momentum": nmo,
                             "momentum_match": mom, "momentum_ci95": ci_m}
            print(f"[horizon {n}] t+{N} ({ms(N):.0f}ms): persistence peak_in_tol={pers:.3f}±{ci_p:.3f} "
                  f"(n_act={npt}) | drift_frac={dfr:.3f} | momentum={mom:.3f}±{ci_m:.3f} (n={nmo}, chance 0.5)",
                  flush=True)
        res[n] = per_h
        try:
            Ns = horizons; xs = [ms(N) for N in Ns]
            pv = [per_h[str(N)]["persistence_peak_in_tol"] for N in Ns]
            dv = [per_h[str(N)]["drift_fraction"] for N in Ns]
            mv = [per_h[str(N)]["momentum_match"] for N in Ns]
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            ax[0].plot(xs, pv, "o-", label="persistence peak_in_tol")
            ax[0].plot(xs, dv, "s--", label="drift fraction")
            ax[0].axhline(0.576, ls=":", color="grey", label="Gate-1 pers (t+1)")
            ax[0].set_xlabel("horizon (ms)"); ax[0].set_ylabel("rate"); ax[0].set_ylim(0, 1)
            ax[0].set_title(f"{n}: persistence decay + drift growth"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
            ax[1].plot(xs, mv, "o-", color="C2"); ax[1].axhline(0.5, ls=":", color="k", label="chance")
            ax[1].set_xlabel("horizon (ms)"); ax[1].set_ylabel("momentum-match"); ax[1].set_ylim(0, 1)
            ax[1].set_title(f"{n}: drift learnability (momentum)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(OUT / f"{n}_horizon_probe.png", dpi=110); plt.close(fig)
        except Exception as e:
            print(f"[horizon] fig skip: {e}", flush=True)
    json.dump(res, open(OUT / "horizon_probe.json", "w"), indent=2,
              default=lambda o: float(o) if hasattr(o, "item") else o)
    print(f"[horizon] wrote {OUT}/horizon_probe.json", flush=True)


def label_align():
    """LABEL_ALIGN=1: PRE-LAUNCH GATE for the Gate-2b t+N target slicing. Two checks:
    (A) SYNTHETIC — inject a ridge into ONLY sub-window `off` of a fake extended target;
        descriptor_target of that sub-window must peak at the injected freq, others flat.
    (B) END-TO-END vs the REAL pipeline — sub-window `off` of the extended target of sample i
        is the SAME physical window as the INPUT of sample i+(off+1)*spw (spw = chunk/step
        strided windows). Peak-freq match rate must be ~1.0; an off-by-one (t+3 vs t+4) would
        drop it to ~chance and would fake mean-reversion skill invisibly downstream.
    Env: SHOTS_VAL (uses 1st), N_SUB (default 4)."""
    import json
    from torch.utils.data import DataLoader
    core_heads = getattr(core, "spec_descriptor_heads", {})
    n = (spec and spec[0]) or None
    if n is None or n not in core_heads:
        print("[align] no descriptor head on ckpt — cannot run", flush=True); return
    dh = core_heads[n]
    # Derive n_sub from the CKPT's OWN horizon so the model's actuator tokenizer geometry
    # (patch_size scales with prediction_horizon_s — actuators are HORIZON-sized future inputs)
    # matches the data. A 0.05 (t+1) model CANNOT ingest 0.2 data (actuator patch_pos 5 vs 20).
    # => run this on a MULTI-horizon (0.2) ckpt (the smoke/production run, NOT g2).
    n_sub = max(1, round(a["prediction_horizon_s"] / a["chunk_duration_s"]))
    spw = max(1, round(a["chunk_duration_s"] / a["step_size_s"]))
    if n_sub < 2:
        print(f"[align] ckpt horizon → n_sub={n_sub}; need a multi-horizon (0.2) ckpt for the t+N test. Abort.", flush=True)
        return
    pkf = lambda D: D.mean(2).argmax(1)                      # (S,NF,TCOL)->(S,) peak bin
    out = {"n_sub": n_sub, "spw": spw, "tol_bins": int(TOL_BINS)}

    # (A) synthetic injected-ridge check
    Fq = dh.mode_hi * 2 if dh.mode_hi else 512
    Tw_s = 24; C = 4
    inj_bin = (dh.mode_lo + dh.mode_hi) // 2                 # a mid-band freq bin
    _tsub = n_sub - 1                                       # last sub-window = t+n_sub (sub3 = t+4 at n_sub=4)
    synth = torch.zeros(1, C, Fq, n_sub * Tw_s)
    synth[:, :, inj_bin, _tsub * Tw_s:(_tsub + 1) * Tw_s] = 5.0   # ridge ONLY in the last sub-window
    a_ok = True
    for off in range(n_sub):
        _d = dh.descriptor_target(synth[..., off * Tw_s:(off + 1) * Tw_s])
        pk = int(pkf(_d)[0]) + dh.mode_lo
        prom = float((_d.amax(1) - _d.mean(1)).max())
        hit = (off == _tsub and abs(pk - inj_bin) <= TOL_BINS and prom > 0.1) or (off != _tsub and prom < 0.1)
        a_ok = a_ok and hit
        print(f"[align A] sub{off}: peak_bin={pk} prom={prom:.3f} (ridge@{inj_bin} ONLY in sub{_tsub}=t+{n_sub}) -> {'ok' if hit else 'BAD'}", flush=True)
    out["synthetic_ok"] = bool(a_ok)

    # (B) end-to-end pipeline cross-check
    shots = [s for s in SHOTS_VAL if (data_dir / f"{s}_processed.h5").exists()]
    f = data_dir / f"{shots[0]}_processed.h5"
    hz = a["prediction_horizon_s"]   # == n_sub*chunk; the ckpt's own horizon → actuator geometry matches
    _, sds = build_datasets(data_dir, [f], [f], stats, a["chunk_duration_s"], hz,
                            a["step_size_s"], a["warmup_s"], diag_names, act_names, cache)
    loader = DataLoader(sds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn, drop_last=False)
    DI, TG = [], []
    with torch.no_grad():
        for batch in loader:
            _, di, tg, _, _ = forward_batch(model, batch, device)
            DI.append(di[n].float().cpu()); TG.append(tg[n].float().cpu())
            if sum(x.shape[0] for x in DI) >= 400:
                break
    DI = torch.cat(DI); TG = torch.cat(TG)                  # (S,C,F,Tin),(S,C,F,Text)
    S = DI.shape[0]; Tw = TG.shape[-1] // n_sub
    di_pk = pkf(dh.descriptor_target(DI))                   # (S,) input-window peak bins
    print(f"[align B] shot={shots[0]} S={S} Tin={DI.shape[-1]} Text={TG.shape[-1]} Tw={Tw} (Tin==Tw? {DI.shape[-1]==Tw})", flush=True)
    out["S"] = int(S); out["Tin"] = int(DI.shape[-1]); out["Text"] = int(TG.shape[-1]); out["match"] = {}
    b_ok = True
    for off in range(n_sub):
        sub = TG[..., off * Tw:(off + 1) * Tw]
        tg_pk = pkf(dh.descriptor_target(sub))              # (S,) target sub-window peak
        lag = (off + 1) * spw                               # input sample that IS this window
        if S - lag < 20:
            print(f"[align B] sub{off} (t+{off+1}): too few samples (lag {lag})", flush=True); continue
        m = float((tg_pk[:S - lag] - di_pk[lag:]).abs().le(TOL_BINS).float().mean())
        out["match"][f"t+{off+1}"] = m
        ok = m > 0.9
        b_ok = b_ok and ok
        print(f"[align B] sub{off} = t+{off+1}: peak-freq match vs input@i+{lag} = {m:.3f} (expect ~1.0) -> {'ok' if ok else 'OFF-BY-ONE?'}", flush=True)
    out["pipeline_ok"] = bool(b_ok)
    verdict = "PASS" if (out["synthetic_ok"] and out["pipeline_ok"]) else "FAIL"
    out["verdict"] = verdict
    json.dump(out, open(OUT / "label_align.json", "w"), indent=2, default=lambda o: float(o) if hasattr(o, "item") else o)
    print(f"[align] VERDICT={verdict} (synthetic={out['synthetic_ok']} pipeline={out['pipeline_ok']}) -> {OUT}/label_align.json", flush=True)


def gate2b_eval():
    """GATE2B_EVAL=1: the Gate-2b verdict for the multi-horizon descriptor head. For each horizon
    h in the head's .horizons, on held-out shots, the model's t+h descriptor forecast is scored
    against the THREE pre-registered nulls: persistence (current-window peak), momentum (continue
    the recent h-window drift), anti-momentum (reverse it). horizon h -> the window h*spw strided
    samples ahead (spw = chunk/step). Reports per horizon: peak_in_tol (model vs persistence, +95%
    CI, CI-separation), drift dir-acc(committed) vs max(momentum, anti-momentum), and false-death
    (model-absent on sustained-present windows). Env: SHOTS_VAL, MAX_WIN_VAL."""
    import json
    from torch.utils.data import DataLoader
    core_heads = getattr(core, "spec_descriptor_heads", {})
    specs = [n for n in spec if n in core_heads]
    if not specs:
        print("[g2b] no trained descriptor heads", flush=True); return
    horizons = getattr(core_heads[specs[0]], "horizons", (1,))
    # Pre-registered per-horizon nulls from the horizon probe (job 4988515), matched-horizon. The
    # eval ALSO computes each fresh on its own events (apples-to-apples); this is the reference to
    # read the fresh numbers against. drift-null to beat = max(momentum, anti-momentum).
    PROBE_NULLS = {2: {"persistence": 0.691, "momentum": 0.616, "anti_momentum": 0.384},
                   4: {"persistence": 0.545, "momentum": 0.342, "anti_momentum": 0.658}}
    anchored = bool(ckpt["args"].get("spec_descriptor_anchor", False))
    beta = float(os.environ.get("DESC_ANCHOR_BETA", ckpt["args"].get("spec_descriptor_dist_beta", 4.0)))
    spw = max(1, round(a["chunk_duration_s"] / a["step_size_s"]))
    shots = [s for s in SHOTS_VAL if (data_dir / f"{s}_processed.h5").exists()]
    print(f"[g2b] horizons={horizons} spw={spw} anchored={anchored} shots={len(shots)}", flush=True)
    seq = {n: {"I": [], **{h: [] for h in horizons}} for n in specs}   # per-shot: current desc + per-h model pred
    for sh in shots:
        f = data_dir / f"{sh}_processed.h5"
        _, sds = build_datasets(data_dir, [f], [f], stats, a["chunk_duration_s"],
                                a.get("prediction_horizon_s", a["chunk_duration_s"]),
                                a["step_size_s"], a["warmup_s"], diag_names, act_names, cache)
        loader = DataLoader(sds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn, drop_last=False)
        buf = {n: {"I": [], **{h: [] for h in horizons}} for n in specs}
        seen = 0
        with torch.no_grad():
            for batch in loader:
                if seen >= MAX_WIN_VAL:
                    break
                _, di, tg, _, tok = forward_batch(model, batch, device)
                for n in specs:
                    h_ = core_heads[n]
                    anc = h_.descriptor_target(di[n].float())              # (B,NF,TCOL) current window
                    ancn = anc / anc.amax(1, keepdim=True).clamp_min(1e-6)
                    raw = h_(tok[n])                                        # (B,H,NF,TCOL)
                    buf[n]["I"].append(anc.cpu())
                    for hi, hstep in enumerate(horizons):
                        pr = ancn * beta + raw[:, hi] if anchored else raw[:, hi]
                        buf[n][hstep].append(pr.cpu())
                seen += tg[specs[0]].shape[0]
        for n in specs:
            for key in ["I", *horizons]:
                seq[n][key].append(torch.cat(buf[n][key]).numpy())
    TOL = TOL_BINS
    pk = lambda D: D.mean(2).argmax(1)
    wp = lambda D: D.mean(2).max(1) - np.median(D.mean(2), 1)
    def wilson(p, k, z=1.96):   # Wilson score interval (lo, hi): well-behaved at small k and p in {0,1}
        if not k or (isinstance(p, float) and np.isnan(p)):
            return (float("nan"), float("nan"))
        d = 1.0 + z * z / k
        c = (p + z * z / (2 * k)) / d
        h = (z * np.sqrt(p * (1 - p) / k + z * z / (4 * k * k))) / d
        return (float(c - h), float(c + h))
    _MIN_COMMIT = int(os.environ.get("MIN_COMMIT", "30"))   # min committed-n before ANY beat-flag can fire
    res = {}
    for n in specs:
        res[n] = {}
        for hstep in horizons:
            off = hstep * spw
            m_hit, p_hit, fd = [], [], []
            dr_gt, dr_m, dr_mom = [], [], []
            n_active_all = 0; model_commit_all = 0                         # commit-rate over ALL active windows
            # SUB-THRESHOLD analysis: did the head's DISTRIBUTION move toward the true drift even
            # where its argmax stayed anchored? ΔLL = model vs pure-anchor(persistence) log-prob at
            # the true t+h bin; mass-shift direction; entropy drift-vs-static. Decides "calibrated
            # head, signal below commit threshold" vs "anchor-identical -> input channel exhausted".
            dll_list, sdir_list, absshift_list, h_drift, h_static = [], [], [], [], []
            mh_mom_ok, mh_mom_bad = [], []   # model mass-shift correct? split by whether MOMENTUM was right
            for Iarr, Parr in zip(seq[n]["I"], seq[n][hstep]):
                T = Iarr.shape[0]
                if T <= 2 * off:
                    continue
                pkI, wpI = pk(Iarr), wp(Iarr); pkP, wpP = pk(Parr), wp(Parr)
                thr = float(np.percentile(wpI, 60))
                actI = wpI > thr; ppP = wpP > thr
                for i in range(off, T - off):
                    if not actI[i]:
                        continue
                    n_active_all += 1
                    if abs(pkP[i] - pkI[i]) > TOL:                         # model moved off persistence
                        model_commit_all += 1
                    fut = i + off
                    gpk = pkI[fut]; fut_act = actI[fut]
                    if fut_act:                                            # future present -> peak_in_tol + false-death
                        m_hit.append(abs(pkP[i] - gpk) <= TOL)
                        p_hit.append(abs(pkI[i] - gpk) <= TOL)
                        fd.append(not ppP[i])                              # sustained present, model says absent
                    # full predicted vs pure-anchor freq distribution (per-tcol softmax, tcol-avg)
                    lp = Parr[i]; ep = np.exp(lp - lp.max(0, keepdims=True)); p_model = (ep / ep.sum(0, keepdims=True)).mean(1)
                    ai = Iarr[i]; an = ai / np.clip(ai.max(0, keepdims=True), 1e-6, None)
                    la = an * beta; ea = np.exp(la - la.max(0, keepdims=True)); p_anchor = (ea / ea.sum(0, keepdims=True)).mean(1)
                    Hm = float(-(p_model * np.log(p_model + 1e-9)).sum())
                    if abs(gpk - pkI[i]) > TOL:                            # GT drifted over h windows
                        dr_gt.append(int(np.sign(gpk - pkI[i])))
                        _dm = pkP[i] - pkI[i]
                        dr_m.append(int(np.sign(_dm)) if abs(_dm) > TOL else 0)     # model committed dir
                        _pv = pkI[i] - pkI[i - off]
                        dr_mom.append(int(np.sign(_pv)) if abs(_pv) > TOL else 0)   # momentum dir (recent trend)
                        dll_list.append(float(np.log(p_model[gpk] + 1e-9) - np.log(p_anchor[gpk] + 1e-9)))
                        _fb = np.arange(p_model.shape[0])
                        _shift = float((_fb * p_model).sum() - (_fb * p_anchor).sum())   # E[freq] model - anchor
                        absshift_list.append(abs(_shift))
                        _tdir = int(np.sign(gpk - pkI[i]))
                        _mdir = int(np.sign(_shift)) if abs(_shift) > 1e-3 else 0
                        sdir_list.append(int(_mdir == _tdir))
                        # HEURISTIC-FAILS SPLIT: on windows where MOMENTUM (recent trend) committed,
                        # record whether the model's mass-shift is correct, split by momentum right/wrong.
                        # The model has signal ABOVE the horizon's dominant heuristic iff it stays correct
                        # on that heuristic's WRONG windows (below).
                        _momdir = dr_mom[-1]
                        if _momdir != 0:
                            (mh_mom_ok if _momdir == _tdir else mh_mom_bad).append(int(_mdir == _tdir))
                        h_drift.append(Hm)
                    else:
                        h_static.append(Hm)
            nph = len(m_hit)
            m_pit = float(np.mean(m_hit)) if nph else float("nan")
            p_pit = float(np.mean(p_hit)) if nph else float("nan")
            fdr = float(np.mean(fd)) if fd else float("nan")
            dg = np.array(dr_gt); dm = np.array(dr_m); dmo = np.array(dr_mom)
            com = dm != 0; ncom = int(com.sum())
            dir_com = float(np.mean(dm[com] == dg[com])) if ncom else float("nan")
            mcom = dmo != 0; nmcom = int(mcom.sum())
            mom_acc = float(np.mean(dmo[mcom] == dg[mcom])) if nmcom else float("nan")   # momentum baseline
            anti_acc = float(np.mean(-dmo[mcom] == dg[mcom])) if nmcom else float("nan")  # anti-momentum
            # peak_in_tol gate: WILSON CI-separation (model lower bound > persistence upper bound).
            m_lo, m_hi = wilson(m_pit, nph); p_lo, p_hi = wilson(p_pit, nph)
            beat_pers = bool(not np.isnan(m_lo) and not np.isnan(p_hi) and m_lo > p_hi)
            drift_null = max([x for x in (mom_acc, anti_acc) if not np.isnan(x)] or [float("nan")])
            # drift gate: minimum-committed FLOOR (n<MIN_COMMIT -> DISTINCT 'insufficient_commits'
            # state, gate cannot pass); else the WILSON lower bound of committed dir-acc must clear
            # the (fresh) drift null. Wilson (not Wald) so the bound is valid at small n / p in {0,1}
            # -> closes the degenerate-CI bug class (n~1, p=1 no longer fires the gate).
            dir_lo, dir_hi = wilson(dir_com, ncom)
            if ncom < _MIN_COMMIT:
                drift_state = "insufficient_commits"; beat_drift = False
            else:
                beat_drift = bool(not np.isnan(dir_lo) and not np.isnan(drift_null) and dir_lo > drift_null)
                drift_state = "pass" if beat_drift else "fail"
            cr_all = float(model_commit_all / n_active_all) if n_active_all else float("nan")
            cr_drift = float(ncom / len(dg)) if len(dg) else float("nan")
            # SUB-THRESHOLD summary: did distribution mass move toward the truth beyond the anchor?
            ndll = len(dll_list)
            dll_mean = float(np.mean(dll_list)) if ndll else float("nan")
            dll_ci = 1.96 * float(np.std(dll_list)) / np.sqrt(ndll) if ndll > 1 else float("nan")
            sdir_acc = float(np.mean(sdir_list)) if sdir_list else float("nan")
            sdir_lo, sdir_hi = wilson(sdir_acc, len(sdir_list))
            absshift_mean = float(np.mean(absshift_list)) if absshift_list else float("nan")
            H_drift = float(np.mean(h_drift)) if h_drift else float("nan")
            H_static = float(np.mean(h_static)) if h_static else float("nan")
            anchor_identical = bool(not np.isnan(absshift_mean) and absshift_mean < 0.05)
            subthreshold_signal = bool((not np.isnan(dll_mean) and not np.isnan(dll_ci) and (dll_mean - dll_ci) > 0)
                                       or (not np.isnan(sdir_lo) and sdir_lo > 0.5))
            # HEURISTIC-FAILS SPLIT (closes the sub-threshold footnote): the horizon's DOMINANT
            # heuristic = whichever of momentum/anti-momentum scores higher. It FAILS on the opposite
            # subset (momentum fails on mom-wrong windows; anti-momentum fails on mom-correct windows).
            # If the model's mass-shift dir-acc on the heuristic's FAILING windows CI-beats 0.5, the
            # model carries signal ABOVE the heuristic; if not, the sub-threshold signal IS the heuristic.
            acc_mom_ok = float(np.mean(mh_mom_ok)) if mh_mom_ok else float("nan")
            acc_mom_bad = float(np.mean(mh_mom_bad)) if mh_mom_bad else float("nan")
            n_ok = len(mh_mom_ok); n_bad = len(mh_mom_bad)
            lo_ok, hi_ok = wilson(acc_mom_ok, n_ok); lo_bad, hi_bad = wilson(acc_mom_bad, n_bad)
            # BEYOND-HEURISTIC = model mass-shift correct on BOTH momentum-correct AND momentum-wrong
            # subsets (any trend rule is right on one, wrong on the other by construction; only trend-
            # INDEPENDENT signal clears 0.5 on BOTH). Robust — no need to guess which heuristic dominates
            # (the earlier dominant-label approach mislabeled t+2 off noisy small-n committed accuracies).
            beats_heuristic = bool(n_ok >= _MIN_COMMIT and n_bad >= _MIN_COMMIT
                                   and not np.isnan(lo_ok) and not np.isnan(lo_bad)
                                   and lo_ok > 0.5 and lo_bad > 0.5)
            res[n][f"t+{hstep}"] = {
                "n_active": n_active_all, "n_pairs_scored": nph,
                "peak_in_tol_model": m_pit, "peak_in_tol_model_wilson95": [m_lo, m_hi],
                "peak_in_tol_persistence": p_pit, "peak_in_tol_persistence_wilson95": [p_lo, p_hi],
                "beat_persistence_CIsep": beat_pers,
                "n_drift": int(len(dg)), "n_committed": ncom,
                "commit_rate": cr_all,                       # committed / ALL active windows (raw propensity)
                "commit_rate_among_drifting": cr_drift,      # committed / GT-drift windows (g2-comparable; but g2 was t+1, drift base-rate differs)
                "drift_dir_acc_committed": dir_com, "drift_dir_acc_committed_wilson95": [dir_lo, dir_hi],
                "momentum_acc": mom_acc, "anti_momentum_acc": anti_acc, "n_momentum_committed": nmcom,
                "drift_null_max": drift_null, "beat_drift_nulls": beat_drift,
                "drift_gate_state": drift_state, "min_commit_required": _MIN_COMMIT,
                "false_death_rate": fdr, "false_death_rate_wilson95": list(wilson(fdr, len(fd))), "n_false_death": len(fd),
                "subthreshold_dLL_mean": dll_mean, "subthreshold_dLL_ci95": dll_ci, "n_subthreshold": ndll,
                "subthreshold_massshift_dir_acc": sdir_acc, "subthreshold_massshift_dir_acc_wilson95": [sdir_lo, sdir_hi],
                "subthreshold_mean_abs_shift_bins": absshift_mean,
                "entropy_drift": H_drift, "entropy_static": H_static,
                "anchor_identical": anchor_identical, "subthreshold_signal": subthreshold_signal,
                "model_dir_acc_mom_correct": acc_mom_ok, "model_dir_acc_mom_correct_wilson95": [lo_ok, hi_ok], "n_mom_correct": n_ok,
                "model_dir_acc_mom_wrong": acc_mom_bad, "model_dir_acc_mom_wrong_wilson95": [lo_bad, hi_bad], "n_mom_wrong": n_bad,
                "beats_heuristic": beats_heuristic,   # True iff correct on BOTH subsets (trend-independent)
                "probe_reference": PROBE_NULLS.get(hstep, {})}
            print(f"[g2b {n} t+{hstep}] peak_in_tol model={m_pit:.3f}[{m_lo:.3f},{m_hi:.3f}] vs pers={p_pit:.3f} "
                  f"(probe {PROBE_NULLS.get(hstep,{}).get('persistence','?')}) [beat={beat_pers}] | "
                  f"commit(all)={cr_all:.3f} commit(drift)={cr_drift:.3f} n_com={ncom} "
                  f"dir-acc={dir_com:.3f}[{dir_lo:.3f},{dir_hi:.3f}] vs null={drift_null:.3f} "
                  f"[drift:{drift_state}] | false-death={fdr:.3f}", flush=True)
            print(f"[g2b {n} t+{hstep} SUBTHR] dLL(model-anchor)={dll_mean:.4f}±{dll_ci:.4f} (n={ndll}) | "
                  f"mass-shift dir-acc={sdir_acc:.3f}[{sdir_lo:.3f},{sdir_hi:.3f}] mean|shift|={absshift_mean:.3f}bins | "
                  f"H(drift)={H_drift:.3f} H(static)={H_static:.3f} | anchor_identical={anchor_identical} "
                  f"subthreshold_signal={subthreshold_signal}", flush=True)
            print(f"[g2b {n} t+{hstep} HEUR-SPLIT] model mass-shift dir-acc: "
                  f"mom-correct={acc_mom_ok:.3f}[{lo_ok:.3f},{hi_ok:.3f}](n={n_ok}) "
                  f"mom-wrong={acc_mom_bad:.3f}[{lo_bad:.3f},{hi_bad:.3f}](n={n_bad}) "
                  f"-> beats_heuristic(BOTH>0.5)={beats_heuristic}", flush=True)
    # Run metadata — incl. EFFECTIVE-INFORMATIVE-STEPS: fraction of TRAINING batches that were
    # ece-PRESENT (real descriptor gradient). Pass TRAIN_LOG=<the run's .out>. A flat verdict with
    # a low fraction has an alternative explanation (under-trained); a strong verdict with a high
    # fraction carries its own robustness note. Absent ece batches show 'ece=0.0000' in the log.
    _meta = {"horizons": list(horizons), "n_shots": len(shots), "anchored": anchored,
             "beta": beta, "tol_bins": int(TOL), "spw": spw, "max_win_val": MAX_WIN_VAL}
    _tl = os.environ.get("TRAIN_LOG")
    if _tl and os.path.exists(_tl):
        import re
        _pres = _tot = 0
        for _ln in open(_tl):
            _m = re.search(r"\bece=([0-9.]+)\s*\|\|", _ln)   # the modality-MAE field, just before '||'
            if _m:
                _tot += 1
                if float(_m.group(1)) > 1e-6:
                    _pres += 1
        if _tot:
            _meta["train_ece_present_fraction"] = round(_pres / _tot, 4)
            _meta["train_logged_steps_scanned"] = _tot
            print(f"[g2b] informative-steps: {_pres}/{_tot} logged steps ece-present "
                  f"({100 * _pres / _tot:.1f}%)", flush=True)
    res["_meta"] = _meta
    json.dump(res, open(OUT / "gate2b.json", "w"), indent=2, default=lambda o: float(o) if hasattr(o, "item") else o)
    print(f"[g2b] wrote {OUT}/gate2b.json", flush=True)


def act_cf():
    """ACT_CF=1: GATE 3 — actuator conditioning counterfactual. On mode-active windows (mode present
    now AND at t+H), perturb an actuator trajectory (+Δσ, sustained) and read the SIGNED response of
    the predicted t+H descriptor via the 2b machinery: ΔLL at the TRUE mode bin (presence-at-location;
    +ECCD suppression -> <0), Δmass-shift (E[freq]), Δentropy (flatter belief -> >0). PLACEBO actuator
    (no mode coupling) must NOT respond -> specificity, Gate-1 style. Env: SHOTS_VAL, EXTRA_DATA_DIR
    (resolve showcase shots), ACT_CF_TARGET (ech_power), ACT_CF_PLACEBO (gas_flow), ACT_CF_DELTAS ('2,1')."""
    import json
    from torch.utils.data import DataLoader
    core_heads = getattr(core, "spec_descriptor_heads", {})
    specs = [nm for nm in spec if nm in core_heads]
    if not specs:
        print("[actcf] no descriptor heads", flush=True); return
    n = specs[0]; dh = core_heads[n]
    horizons = getattr(dh, "horizons", (1,)); hstep = max(horizons); hi = list(horizons).index(hstep)
    anchored = bool(ckpt["args"].get("spec_descriptor_anchor", False))
    beta = float(os.environ.get("DESC_ANCHOR_BETA", ckpt["args"].get("spec_descriptor_dist_beta", 4.0)))
    beta_corr = float(os.environ.get("ACT_CF_BETA_CORR", "2.0"))   # lower-anchor readout — CORROBORATION ONLY (OOD)
    spw = max(1, round(a["chunk_duration_s"] / a["step_size_s"])); off = hstep * spw
    targets = [x for x in os.environ.get("ACT_CF_TARGET", "ech_power").split(",") if x]
    placebos = [x for x in os.environ.get("ACT_CF_PLACEBO", "gas_flow").split(",") if x]
    deltas = [float(x) for x in os.environ.get("ACT_CF_DELTAS", "2,1").split(",")]
    act_set = set(c["name"] for c in ckpt["actuators"])
    perts = []
    for _t in targets:                                    # each PRIMARY: base delta + dose (deltas[1:])
        perts += [(_t, deltas[0])] + [(_t, d) for d in deltas[1:]]
    for _p in placebos:                                   # each PLACEBO at base delta
        perts += [(_p, deltas[0])]
    perts = [(nm, d) for nm, d in perts if nm in act_set]
    extra_dir = os.environ.get("EXTRA_DATA_DIR")

    def resolve(sh):
        f = data_dir / f"{sh}_processed.h5"
        if f.exists():
            return f
        if extra_dir and (Path(extra_dir) / f"{sh}_processed.h5").exists():
            return Path(extra_dir) / f"{sh}_processed.h5"
        return None

    shots = [s for s in SHOTS_VAL if resolve(s)]
    print(f"[actcf] targets={targets} placebos={placebos} deltas={deltas} hstep=t+{hstep} shots={len(shots)}", flush=True)

    def dist(logit):   # (T,NF,TCOL) logit -> (T,NF) per-tcol softmax over freq, tcol-averaged
        e = np.exp(logit - logit.max(1, keepdims=True)); return (e / e.sum(1, keepdims=True)).mean(2)

    # fields: β8 output (main, near-saturated) | β_corr output (OOD corroboration) | RESIDUAL-level (primary instrument)
    acc = {f"{nm}@{d}": {"dll": [], "dfreq": [], "dent": [],
                         "dll_c": [], "dfreq_c": [],
                         "rnorm": [], "rdfreq": [], "byshot_dfreq": {}} for nm, d in perts}
    for sh in shots:
        f = resolve(sh)
        _, sds = build_datasets(data_dir, [f], [f], stats, a["chunk_duration_s"],
                                a.get("prediction_horizon_s", a["chunk_duration_s"]),
                                a["step_size_s"], a["warmup_s"], diag_names, act_names, cache)
        loader = DataLoader(sds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn, drop_last=False)
        I_list, RR_list = [], []; RP_list = {f"{nm}@{d}": [] for nm, d in perts}
        seen = 0; std_sh = {}
        with torch.no_grad():
            for batch in loader:
                if seen >= MAX_WIN_VAL:
                    break
                if not std_sh:   # per-shot actuator std (RAW units) — perturb by Δσ * std, else a fixed
                    for nm, _ in perts:   # +Δ is negligible for large-scale actuators (ech_power ~1e5)
                        std_sh[nm] = max(float(torch.nan_to_num(batch["targets"][nm].float()).std()), 1e-6)
                _, di, tg, _, tok = forward_batch(model, batch, device)
                anc = dh.descriptor_target(di[n].float())
                I_list.append(anc.cpu().numpy())
                RR_list.append(dh(tok[n])[:, hi].cpu().numpy())                     # PRE-ANCHOR residual (real)
                for nm, d in perts:
                    _, _, _, _, tokp = forward_batch(model, batch, device, act_perturb={nm: d * std_sh[nm]})
                    RP_list[f"{nm}@{d}"].append(dh(tokp[n])[:, hi].cpu().numpy())   # PRE-ANCHOR residual (perturbed)
                seen += tg[n].shape[0]
        I = np.concatenate(I_list); RR = np.concatenate(RR_list)
        anc_norm = I / np.clip(I.max(1, keepdims=True), 1e-6, None)
        # output logit at anchor weight b: reconstruct WITHOUT re-running backbone (base is actuator-independent)
        out_logit = (lambda resid, b: anc_norm * b + resid) if anchored else (lambda resid, b: resid)
        pk = lambda D: D.mean(2).argmax(1); wp = lambda D: D.mean(2).max(1) - np.median(D.mean(2), 1)
        pkI = pk(I); actI = wp(I) > float(np.percentile(wp(I), 60))
        T = I.shape[0]; fb = np.arange(I.shape[1])
        pR = dist(out_logit(RR, beta)); pR_c = dist(out_logit(RR, beta_corr)); pRR = dist(RR)   # residual-alone freq dist
        for nm, d in perts:
            key = f"{nm}@{d}"; RP = np.concatenate(RP_list[key])
            pP = dist(out_logit(RP, beta)); pP_c = dist(out_logit(RP, beta_corr)); pRP = dist(RP)
            for i in range(off, T - off):
                if not (actI[i] and actI[i + off]):
                    continue
                tb = int(pkI[i + off])
                # --- β8 anchored output (main; near-saturated softmax) ---
                acc[key]["dll"].append(float(np.log(pP[i][tb] + 1e-9) - np.log(pR[i][tb] + 1e-9)))
                _dfq_i = float((fb * pP[i]).sum() - (fb * pR[i]).sum())
                acc[key]["dfreq"].append(_dfq_i)
                acc[key]["byshot_dfreq"].setdefault(sh, []).append(_dfq_i)   # per-shot heterogeneity (AE-active vs quiet)
                acc[key]["dent"].append(float(-(pP[i] * np.log(pP[i] + 1e-9)).sum()
                                              + (pR[i] * np.log(pR[i] + 1e-9)).sum()))
                # --- lower-anchor output (CORROBORATION ONLY; OOD, never a decider) ---
                acc[key]["dll_c"].append(float(np.log(pP_c[i][tb] + 1e-9) - np.log(pR_c[i][tb] + 1e-9)))
                acc[key]["dfreq_c"].append(float((fb * pP_c[i]).sum() - (fb * pR_c[i]).sum()))
                # --- RESIDUAL-LEVEL (PRIMARY instrument): pre-anchor, anchor-mask-free ---
                acc[key]["rnorm"].append(float(np.sqrt(((RP[i] - RR[i]) ** 2).mean())))         # RMS ‖Δresidual‖
                acc[key]["rdfreq"].append(float((fb * pRP[i]).sum() - (fb * pRR[i]).sum()))      # residual freq direction

    def mci(x):
        x = np.array(x); return (float(x.mean()) if len(x) else float("nan"),
                                 1.96 * float(x.std()) / np.sqrt(len(x)) if len(x) > 1 else float("nan"), len(x))
    def boot(x, B=4000):
        # nonparametric bootstrap PERCENTILE CI of the mean (deterministic seed; robust for small/skewed
        # effects where the Wald CI misleads). Sign is "confirmed" iff this CI excludes 0.
        x = np.asarray(x, dtype=float)
        if len(x) < 2: return (float("nan"), float("nan"))
        rng = np.random.default_rng(12345)
        means = x[rng.integers(0, len(x), size=(B, len(x)))].mean(1)
        return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    prim_keys = [f"{t}@{deltas[0]}" for t in targets]
    plac_keys = [f"{p}@{deltas[0]}" for p in placebos]
    res = {}
    for key in acc:
        m, c, k = mci(acc[key]["dll"]); mf, cf, _ = mci(acc[key]["dfreq"]); me, ce, _ = mci(acc[key]["dent"])
        mlc, clc, _ = mci(acc[key]["dll_c"]); mfc, cfc, _ = mci(acc[key]["dfreq_c"])
        rn, rnc, _ = mci(acc[key]["rnorm"]); rdf, rdfc, _ = mci(acc[key]["rdfreq"])
        dfq_lo, dfq_hi = boot(acc[key]["dfreq"]); dll_lo, dll_hi = boot(acc[key]["dll"])
        responds = bool(not np.isnan(c) and (m + c < 0 or m - c > 0))     # ΔLL Wald CI excludes 0
        suppresses = bool(not np.isnan(c) and (m + c) < 0)
        dfreq_sig = bool(not np.isnan(cf) and (mf + cf < 0 or mf - cf > 0))               # dfreq WALD CI excludes 0
        dfreq_boot_sig = bool(not np.isnan(dfq_lo) and (dfq_lo > 0 or dfq_hi < 0))          # dfreq BOOTSTRAP CI excludes 0
        dll_boot_sig = bool(not np.isnan(dll_lo) and (dll_lo > 0 or dll_hi < 0))
        # PER-SHOT dfreq breakdown (heterogeneity): a REAL regime-dependent effect concentrates in AE-active
        # shots (large |dfreq|, consistent sign) while quiet non-AE shots sit near 0 — pooling would dilute it.
        # This distinguishes "true effect diluted" from "genuine wash-out", and tests the β5 sign-flip.
        byshot = {str(s): {"mean": float(np.mean(v)), "n": len(v)} for s, v in acc[key]["byshot_dfreq"].items()}
        res[key] = {"n": k, "dLL_truebin": m, "dLL_ci95": c, "dLL_boot95": [dll_lo, dll_hi], "dLL_boot_sig": dll_boot_sig,
                    "responds": responds, "suppresses": suppresses,
                    "dfreq_bins": mf, "dfreq_ci95": cf, "dfreq_sig": dfreq_sig,
                    "dfreq_boot95": [dfq_lo, dfq_hi], "dfreq_boot_sig": dfreq_boot_sig,
                    "dentropy": me, "dentropy_ci95": ce,
                    "dLL_truebin_lowbeta": mlc, "dLL_ci95_lowbeta": clc, "dfreq_bins_lowbeta": mfc, "dfreq_ci95_lowbeta": cfc,
                    "resid_dnorm": rn, "resid_dnorm_ci95": rnc, "resid_dfreq_bins": rdf, "resid_dfreq_ci95": rdfc,
                    "byshot_dfreq": byshot}
        print(f"[actcf {key}] n={k} ΔLL@truebin={m:+.4f}±{c:.4f} boot95=[{dll_lo:+.4f},{dll_hi:+.4f}]"
              f"[{'SUPPRESS' if suppresses else ('responds' if responds else 'ns')}] | "
              f"Δfreq={mf:+.4f} boot95=[{dfq_lo:+.4f},{dfq_hi:+.4f}]{'*BOOT' if dfreq_boot_sig else ''} | Δentropy={me:+.4f}", flush=True)
    # PER-SHOT dfreq breakdown for the PRIMARY channels (heterogeneity = interpretation; pooled boot CI = gate)
    for key in prim_keys:
        bs = res.get(key, {}).get("byshot_dfreq", {})
        if not bs:
            continue
        rows = sorted(bs.items(), key=lambda kv: kv[1]["mean"])   # sorted by dfreq → concentration visible
        npos = sum(1 for _, v in rows if v["mean"] > 0); nneg = sum(1 for _, v in rows if v["mean"] < 0)
        print(f"[actcf-byshot {key}] per-shot Δfreq (sorted; {nneg} neg / {npos} pos of {len(rows)} shots):", flush=True)
        for s, v in rows:
            print(f"    {s}: Δfreq={v['mean']:+.4f}  n={v['n']}", flush=True)
    # ---- RESIDUAL-LEVEL VERDICT (PRIMARY): pre-anchor specificity ordering, pin >> placebos ----
    print(f"[actcf] --- RESIDUAL-LEVEL (pre-anchor; PRIMARY instrument, anchor-mask-free) ---", flush=True)
    plac_upper = max([res[k]["resid_dnorm"] + res[k]["resid_dnorm_ci95"] for k in plac_keys
                      if not np.isnan(res[k]["resid_dnorm_ci95"])], default=0.0)
    latent = {}
    for key in prim_keys + plac_keys:
        rn = res[key]["resid_dnorm"]; rnc = res[key]["resid_dnorm_ci95"]
        rdf = res[key]["resid_dfreq_bins"]; rdfc = res[key]["resid_dfreq_ci95"]
        rdf_sig = bool(not np.isnan(rdfc) and (rdf + rdfc < 0 or rdf - rdfc > 0))
        above_plac = bool(key in prim_keys and not np.isnan(rnc) and (rn - rnc) > plac_upper)   # CI-separated from placebo band
        if key in prim_keys:
            latent[key] = above_plac
        tag = "PRIM" if key in prim_keys else "PLAC"
        print(f"[actcf-resid {key}] {tag} ‖Δresid‖={rn:.4e}±{rnc:.1e} "
              f"{'>>PLAC' if above_plac else ('(<=plac band '+format(plac_upper,'.2e')+')' if key in prim_keys else '')} | "
              f"resid_Δfreq={rdf:+.3f}±{rdfc:.3f}bins{'*' if rdf_sig else ''}", flush=True)
    latent_conditioning = bool(any(latent.values()))
    # β8 output-level verdict (kept for continuity; near-saturated softmax UNDER-reports — NOT the decider)
    prim_responds = {k: res.get(k, {}).get("responds", False) for k in prim_keys}
    plac_responds = {k: res.get(k, {}).get("responds", False) for k in plac_keys}
    conditioning_alive_output = bool(any(prim_responds.values()) and not any(plac_responds.values()))
    # lower-β corroboration (OOD; report only)
    print(f"[actcf] --- LOWER-β={beta_corr} CORROBORATION (OOD — NOT a decider) ---", flush=True)
    for key in prim_keys + plac_keys:
        print(f"[actcf-lowbeta {key}] ΔLL={res[key]['dLL_truebin_lowbeta']:+.4f}±{res[key]['dLL_ci95_lowbeta']:.4f} | "
              f"Δfreq={res[key]['dfreq_bins_lowbeta']:+.3f}±{res[key]['dfreq_ci95_lowbeta']:.3f}bins", flush=True)
    res["_verdict"] = {
        "latent_conditioning": latent_conditioning, "latent_by_channel": latent, "resid_plac_upper": plac_upper,
        "conditioning_alive_output_beta8": conditioning_alive_output,
        "primaries_respond_output": prim_responds, "placebos_fired_output": plac_responds,
        "beta_main": beta, "beta_corr": beta_corr, "hstep": hstep,
        "PRIMARY_INSTRUMENT": "residual-level ‖Δresid‖ specificity ordering (pin>>placebos => latent conditioning)",
        "note": "β8 output is near-saturated => ΔLL under-reports; residual-level is the verdict. lower-β is OOD corroboration only."}
    print(f"[actcf] VERDICT latent_conditioning={latent_conditioning} (residual) | "
          f"latent_by_channel={latent} | output_beta8_alive={conditioning_alive_output}", flush=True)
    json.dump(res, open(OUT / "act_cf.json", "w"), indent=2, default=lambda o: float(o) if hasattr(o, "item") else o)
    print(f"[actcf] wrote {OUT}/act_cf.json", flush=True)


def act_audit():
    """ACT_AUDIT=1: token-path audit for the EXACT-zero ACT_CF result. For 1 batch: (1) actuator DATA
    presence (finite-frac, std, validity) — a masked/absent actuator perturbs to nothing (false
    negative); (2) does the hook change act tokens; (3) max|Δtok[ece]| under a LARGE (+5σ) perturbation
    of {target, placebo}. Δtok>0 with data present -> perturbation reaches the spectro path (ACT_CF valid,
    conditioning genuinely weak); Δtok==0 with data present -> actuators architecturally don't reach the
    ece token path (the localized finding); data absent -> re-run on shots WITH actuator data."""
    from torch.utils.data import DataLoader
    core_heads = getattr(core, "spec_descriptor_heads", {})
    n = [x for x in spec if x in core_heads][0]
    target = os.environ.get("ACT_CF_TARGET", "ech_power"); placebo = os.environ.get("ACT_CF_PLACEBO", "gas_flow")
    extra = os.environ.get("EXTRA_DATA_DIR")

    def resolve(sh):
        f = data_dir / f"{sh}_processed.h5"
        if f.exists():
            return f
        p = Path(extra) / f"{sh}_processed.h5" if extra else None
        return p if (p and p.exists()) else None

    sh = [s for s in SHOTS_VAL if resolve(s)][0]; f = resolve(sh)
    print(f"[audit] shot={sh} file={f}", flush=True)
    _, sds = build_datasets(data_dir, [f], [f], stats, a["chunk_duration_s"],
                            a.get("prediction_horizon_s", a["chunk_duration_s"]),
                            a["step_size_s"], a["warmup_s"], diag_names, act_names, cache)
    batch = next(iter(DataLoader(sds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)))
    for act in (target, placebo):
        t = batch["targets"].get(act)
        if t is None:
            print(f"[audit] {act}: MISSING from batch['targets']", flush=True); continue
        t = t.float(); fin = float(torch.isfinite(t).float().mean()); sd = float(torch.std(torch.nan_to_num(t)))
        print(f"[audit] {act}: shape={tuple(t.shape)} finite_frac={fin:.3f} std(nan->0)={sd:.4f} "
              f"absmean={float(torch.nan_to_num(t).abs().mean()):.4f}", flush=True)
    with torch.no_grad():
        _, _, _, _, tok = forward_batch(model, batch, device)
        for act in (target, placebo):
            _, _, _, _, tokp = forward_batch(model, batch, device, act_perturb={act: 5.0})
            dtok = float((tok[n] - tokp[n]).abs().max())
            print(f"[audit] +5sigma {act}: max|delta tok[{n}]|={dtok:.6e}", flush=True)
    sys.exit(0)


def act_scale_audit():
    """ACT_SCALE_AUDIT=1: GATE3-FIX Task 1 (no training). For ALL actuators, print + write
    actuator_scaling_plan.md: finite_frac, mean, std, absmax, all-positive?, min (log-safety),
    preprocessing_stats presence (raw/log), and the token-path sensitivity max|Δtok[ece]| under a
    +5σ (std-scaled) perturbation = the PRE-FIX baseline the retrain must beat. Proposes a preprocess
    method per actuator (keep-none / standardize / log_standardize) for USER confirmation."""
    from torch.utils.data import DataLoader
    core_heads = getattr(core, "spec_descriptor_heads", {})
    n = [x for x in spec if x in core_heads][0]
    acts = [c["name"] for c in ckpt["actuators"]]
    extra = os.environ.get("EXTRA_DATA_DIR")

    def resolve(sh):
        f = data_dir / f"{sh}_processed.h5"
        if f.exists():
            return f
        p = Path(extra) / f"{sh}_processed.h5" if extra else None
        return p if (p and p.exists()) else None

    shots = [s for s in SHOTS_VAL if resolve(s)][:3]
    try:
        st = torch.load(a["stats_path"], weights_only=False)
    except Exception:
        st = {}
    agg = {act: {"n": 0, "nfin": 0, "sum": 0.0, "sumsq": 0.0, "absmax": 0.0, "min": 1e30} for act in acts}
    b0 = None
    for sh in shots:
        f = resolve(sh)
        _, sds = build_datasets(data_dir, [f], [f], stats, a["chunk_duration_s"],
                                a.get("prediction_horizon_s", a["chunk_duration_s"]),
                                a["step_size_s"], a["warmup_s"], diag_names, act_names, cache)
        b = next(iter(DataLoader(sds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)))
        if b0 is None:
            b0 = b
        for act in acts:
            t = b["targets"].get(act)
            if t is None:
                continue
            t = t.float(); tf = torch.nan_to_num(t)
            agg[act]["n"] += t.numel(); agg[act]["nfin"] += int(torch.isfinite(t).sum())
            agg[act]["sum"] += float(tf.sum()); agg[act]["sumsq"] += float((tf * tf).sum())
            agg[act]["absmax"] = max(agg[act]["absmax"], float(tf.abs().max()))
            agg[act]["min"] = min(agg[act]["min"], float(tf.min()))
    rows = []
    with torch.no_grad():
        _, _, _, _, tok = forward_batch(model, b0, device)
        for act in acts:
            g = agg[act]; nn = max(g["n"], 1); nf = max(g["nfin"], 1)
            mean = g["sum"] / nf; var = g["sumsq"] / nf - mean * mean; std = var ** 0.5 if var > 0 else 0.0
            finf = g["nfin"] / nn; allpos = g["min"] >= 0.0
            _, _, _, _, tokp = forward_batch(model, b0, device, act_perturb={act: 5.0 * max(std, 1e-6)})
            dtok = float((tok[n] - tokp[n]).abs().max())
            spk = list(st[act].keys()) if (act in st and isinstance(st[act], dict)) else []
            if std <= 10 and abs(mean) <= 10:
                method, why = "none(keep)", "already O(1)"
            elif allpos and std > 100:
                method, why = "log_standardize", "large positive power-law scale"
            else:
                method, why = "standardize", "large scale, has negatives/zero-centered"
            rows.append(dict(act=act, finf=finf, mean=mean, std=std, absmax=g["absmax"],
                             allpos=allpos, mn=g["min"], dtok=dtok, stats=spk, method=method, why=why))
            print(f"[scale] {act:16s} fin={finf:.3f} mean={mean:+.3g} std={std:.3g} absmax={g['absmax']:.3g} "
                  f"allpos={allpos} +5σ|Δtok[ece]|={dtok:.2e} stats={spk} -> PROPOSE {method} ({why})", flush=True)
    live = 3.2e-2   # gas_flow reference from the Gate-3 audit
    lines = ["# Actuator scaling plan (GATE3-FIX Task 1) — PROPOSAL, awaiting user confirmation", "",
             f"Model: {CKPT}", f"Shots: {shots}  |  token sensitivity = max|Δtok[ece]| under +5σ std-scaled perturbation.",
             f"Reference LIVE channel (gas_flow, Gate-3 audit): ~{live:.1e}. DEAD if << this.", "",
             "| actuator | finite | mean | std | absmax | all≥0 | min | +5σ \\|Δtok\\| | in stats | current | PROPOSED |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dead = " (DEAD)" if r["dtok"] < 1e-3 else ""
        lines.append(f"| {r['act']} | {r['finf']:.3f} | {r['mean']:+.3g} | {r['std']:.3g} | {r['absmax']:.3g} | "
                     f"{r['allpos']} | {r['mn']:+.3g} | {r['dtok']:.2e}{dead} | {','.join(r['stats']) or '—'} | none | "
                     f"**{r['method']}** ({r['why']}) |")
    lines += ["", "## Notes", "- Current data_loader actuator preprocess = `none` for ALL (confirmed: ech_power raw ~1e5).",
              "- `log_standardize` on all-positive channels only; if min<0 or min==0 present, needs explicit offset/clip"
              " (Task 2 must state handling — log of 0/neg is the classic failure).",
              "- Angle channels (ech_tor/pol_angle, polarization) are ~O(1) radians → likely `none(keep)`.",
              "- Proposed methods are a STARTING POINT from the numbers; the physics call (power-law vs linear vs"
              " leave-alone) is the user's. Confirm per-actuator before Task 2.",
              "- Baseline token sensitivities above are what the post-standardization smoke (Task 2) must lift"
              f" toward the live reference (~{live:.1e})."]
    (OUT / "actuator_scaling_plan.md").write_text("\n".join(lines))
    print(f"[scale] wrote {OUT}/actuator_scaling_plan.md", flush=True)
    sys.exit(0)


if os.environ.get("ACT_SCALE_AUDIT"):
    act_scale_audit()

if os.environ.get("ACT_AUDIT"):
    act_audit()

if os.environ.get("ACT_CF"):
    act_cf()
    sys.exit(0)

if os.environ.get("GATE2B_EVAL"):
    gate2b_eval()
    sys.exit(0)

if os.environ.get("LABEL_ALIGN"):
    label_align()
    sys.exit(0)

if os.environ.get("HORIZON_PROBE"):
    horizon_probe()
    sys.exit(0)

if os.environ.get("ONSET_EVAL"):
    onset_skill_eval()
    sys.exit(0)

if os.environ.get("EVAL_TRAINED"):
    from torch.utils.data import DataLoader
    eval_trained()
    sys.exit(0)


print("[proof] collecting train tokens/descriptors...", flush=True)
trTOK, trTGT, trINP = collect(tr_ds, MAX_WIN_TRAIN)
print("[proof] collecting val tokens/descriptors...", flush=True)
vaTOK, vaTGT, vaINP = collect(va_ds, MAX_WIN_VAL)

results = {}
for n in spec:
    Xtr, Ytr = trTOK[n].to(device), trTGT[n].to(device)
    Xva, Yva = vaTOK[n].to(device), vaTGT[n].to(device)
    Yin_va = vaINP[n].numpy()                              # persistence prediction (current window)
    mu, sd = Ytr.mean(), Ytr.std() + 1e-6                  # standardize target for stable MSE
    head = DHead(Xtr.shape[1], Xtr.shape[2], NF, TCOL).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3,
                           weight_decay=float(os.environ.get("WEIGHT_DECAY", "1e-4")))
    n_tr = Xtr.shape[0]
    for step in range(STEPS):
        idx = torch.randint(0, n_tr, (32,), device=device)
        pred = head(Xtr[idx])
        loss = ((pred - (Ytr[idx] - mu) / sd) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0 or step == STEPS - 1:
            print(f"[proof {n}] step {step} train_mse {loss.item():.4f}", flush=True)
    head.eval()
    with torch.no_grad():
        Pva = (head(Xva) * sd + mu).cpu().numpy()          # (Nva,NF,TCOL) de-standardized
    Yva_np = Yva.cpu().numpy()
    m_model = {"peak_in_tol": peak_in_tol(Pva, Yva_np), "prof_corr": prof_corr(Pva, Yva_np)}
    m_pers = {"peak_in_tol": peak_in_tol(Yin_va, Yva_np), "prof_corr": prof_corr(Yin_va, Yva_np)}
    # ACTIVE windows only (GT has a clear mode peak) — the fair comparison; quiescent
    # windows have a noise "peak" that penalizes model + persistence equally.
    gtprom = Yva_np.max(1).max(1) - np.median(Yva_np.reshape(Yva_np.shape[0], -1), axis=1)
    act = np.where(gtprom >= np.percentile(gtprom, 60))[0]
    m_model_act = {"peak_in_tol": peak_in_tol(Pva[act], Yva_np[act]), "prof_corr": prof_corr(Pva[act], Yva_np[act])}
    m_pers_act = {"peak_in_tol": peak_in_tol(Yin_va[act], Yva_np[act]), "prof_corr": prof_corr(Yin_va[act], Yva_np[act])}
    results[n] = {"model": m_model, "persistence": m_pers, "model_active": m_model_act,
                  "persistence_active": m_pers_act, "n_val": int(Pva.shape[0]), "n_active": int(len(act))}
    print(f"[proof {n}] HELD-OUT (all): model peak_in_tol={m_model['peak_in_tol']:.3f} prof_corr={m_model['prof_corr']:.3f}"
          f"  | persistence peak_in_tol={m_pers['peak_in_tol']:.3f} prof_corr={m_pers['prof_corr']:.3f}", flush=True)
    print(f"[proof {n}] HELD-OUT (ACTIVE n={len(act)}): model peak_in_tol={m_model_act['peak_in_tol']:.3f} "
          f"prof_corr={m_model_act['prof_corr']:.3f} | persistence peak_in_tol={m_pers_act['peak_in_tol']:.3f} "
          f"prof_corr={m_pers_act['prof_corr']:.3f}", flush=True)
    # RIDGE render: stack (NF,TCOL) over val windows -> (NF, Nval*TCOL). GT | MODEL | PERSISTENCE.
    def ridge(D):
        return np.concatenate([D[i] for i in range(min(D.shape[0], 60))], axis=1)   # (NF, up to 60*TCOL)
    Rg, Rm, Rp = ridge(Yva_np), ridge(Pva), ridge(Yin_va)
    vlo, vhi = np.percentile(Rg, 2), np.percentile(Rg, 99)
    fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    khz = np.arange(MODE_LO, MODE_HI) * (500000.0 / 1024 / 1e3)
    for a2, (ttl, R) in zip(ax, [("GT mode ridge (held-out)", Rg),
                                 (f"MODEL forecast (peak_in_tol {m_model['peak_in_tol']:.2f})", Rm),
                                 (f"PERSISTENCE (peak_in_tol {m_pers['peak_in_tol']:.2f})", Rp)]):
        a2.imshow(R, origin="lower", aspect="auto", vmin=vlo, vmax=vhi, cmap="magma",
                  extent=[0, R.shape[1], khz[0], khz[-1]])
        a2.set_ylabel(ttl + "\nkHz", fontsize=8)
    ax[-1].set_xlabel("window-col (time)")
    fig.suptitle(f"{n} descriptor forecast — held-out {SHOTS_VAL} — frozen backbone + readout")
    fig.tight_layout(); fig.savefig(OUT / f"{n}_ridge.png", dpi=120); plt.close(fig)
    print(f"[proof {n}] saved {OUT}/{n}_ridge.png", flush=True)

json.dump(results, open(OUT / "descriptor_proof.json", "w"), indent=2,
          default=lambda o: float(o) if hasattr(o, "item") else o)
print(f"\n[proof] wrote {OUT}/descriptor_proof.json", flush=True)
print("[proof] done", flush=True)
