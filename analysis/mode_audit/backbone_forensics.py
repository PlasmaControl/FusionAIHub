"""BACKBONE forensics on the d1024/48L FSQ model — is the backbone CAPABLE of carrying/
learning mode content, independent of the codec? Three sections (try/except each):

(1) THREE-TAP linear probe — mode-presence AUC at:
      A codec codes      (encode_target of GT)      — do the codes carry mode presence?
      B tokenizer output (diag_tokenizers[ece])     — the 100M+-param tokenizer question
      C backbone output  (ece slice, post-48-blocks)— does the backbone preserve it to the head?
    Where AUC drops localizes where mode info is lost.

(2) GRADIENT-share audit — after one total backward: per-modality (tokenizer+head) grad
    norm (is spectro starved vs video/TS?) + per-layer grad norm AT ECE TOKEN POSITIONS
    through the 48 blocks (does spectro-position gradient vanish/explode with depth?).

(3) OVERFIT-ONE-BATCH on current codes — Adam on one fixed batch, watch ece_codeacc.
    ~1.0 expected; sluggishness/failure = a mechanical bug report (grad flow / loss wiring).

Env: CKPT (d1024/48L FSQ), SHOTS, N_PROBE_WIN, OVERFIT_STEPS, OVERFIT_BS, OUT_DIR.
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
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from eval_e2e_animation_tokamak import load_model
from train_e2e_stage1 import build_datasets, forward_batch, compute_step_loss, _core
from tokamak_foundation_model.data.data_loader import collate_fn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.environ.get("CKPT", "/lustre/orion/fus187/proj-shared/models/e2e_stage1_allshots_b32_resid/e2e_stage1_latest.pt")
SHOTS = os.environ.get("SHOTS", "200729,190996,204811,190900,190904,201585").split(",")
MOD = "ece"
N_PROBE_WIN = int(os.environ.get("N_PROBE_WIN", "256"))
OVERFIT_STEPS = int(os.environ.get("OVERFIT_STEPS", "300"))
OVERFIT_BS = int(os.environ.get("OVERFIT_BS", "4"))
OUT = Path(os.environ.get("OUT_DIR", f"{FMH}/analysis/mode_audit"))
OUT.mkdir(parents=True, exist_ok=True)
FS, NFFT = 500_000.0, 1024
DF = FS / NFFT / 1e3
MODE_LO, MODE_HI = int(round(5.0 / DF)), int(round(40.0 / DF))
res = {"ckpt": CKPT}


def win_P(x_bcft):                       # (C,F,T) -> band-peak prominence
    out = 0.0
    for c in range(x_bcft.shape[0]):
        prof = np.abs(x_bcft[c, MODE_LO:MODE_HI]).mean(1)
        out = max(out, float((prof - gaussian_filter1d(prof, 6.0)).max()))
    return out


def auc(scores, y):                      # Mann-Whitney AUC
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def probe_auc(F, y):                      # torch logistic-regression probe, 5-fold-ish split
    F = torch.tensor(np.asarray(F), dtype=torch.float32)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    y_t = torch.tensor(y, dtype=torch.float32)
    n = len(y); tr = torch.arange(n) % 5 != 0; va = ~tr
    lin = torch.nn.Linear(F.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), 0.05)
    for _ in range(400):
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(lin(F[tr]).squeeze(-1), y_t[tr])
        l.backward(); opt.step()
    with torch.no_grad():
        s = lin(F[va]).squeeze(-1).numpy()
    return auc(s, y[va].astype(int))


model, ckpt = load_model(Path(CKPT), dev)
core = _core(model)
a = ckpt["args"]
dn = [d["name"] for d in ckpt["diagnostics"]]; an = [c["name"] for c in ckpt["actuators"]]
dd = Path(a["data_dir"]); stats = torch.load(a["stats_path"], weights_only=False)
sfiles = [dd / f"{s}_processed.h5" for s in SHOTS]; sfiles = [f for f in sfiles if f.exists()]
_, ds = build_datasets(dd, sfiles, sfiles, stats, a["chunk_duration_s"],
                       a.get("prediction_horizon_s", a["chunk_duration_s"]), a["step_size_s"],
                       a["warmup_s"], dn, an, Path(f"{FMH}/eval_runs/modecode_cache"),
                       history_windows=int(a.get("history_windows", 1)))
head = core.diag_heads[MOD]
ece_slice = next(L.slice_ for L in core.token_layout if L.name == MOD)
print(f"[fx] ckpt d_model={a['d_model']} n_layers={a['n_layers']} ece_slice={ece_slice.start}:{ece_slice.stop}", flush=True)

# ============ (1) THREE-TAP PROBE ============
try:
    model.eval()
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)
    fA, fB, fC, ys = [], [], [], []
    with torch.no_grad():
        for batch in ld:
            preds, din, targets, masks, slices = forward_batch(model, batch, dev)
            if MOD not in targets:
                continue
            tgt = torch.nan_to_num(targets[MOD].float())
            codes = head.encode_target(tgt).float()                       # (B,ntok,dim)
            tokout = core.diag_tokenizers[MOD](din[MOD])                   # (B,ntok,d) tokenizer output
            bbout = slices[MOD]                                            # (B,ntok,d) backbone output
            fA.append(codes.mean(1).cpu().numpy())                        # pool over tokens
            fB.append(tokout.mean(1).cpu().numpy())
            fC.append(bbout.mean(1).cpu().numpy())
            for b in range(tgt.shape[0]):
                ys.append(win_P(tgt[b].cpu().numpy()))
            if len(ys) >= N_PROBE_WIN:
                break
    fA = np.concatenate(fA)[:len(ys)]; fB = np.concatenate(fB)[:len(ys)]; fC = np.concatenate(fC)[:len(ys)]
    y = (np.array(ys) >= np.median(ys)).astype(int)                       # mode-present = upper half
    res["probe"] = {"n": int(len(y)), "auc_codes": probe_auc(fA, y),
                    "auc_tokenizer_out": probe_auc(fB, y), "auc_backbone_out": probe_auc(fC, y)}
    print(f"[fx] PROBE (n={len(y)}) mode-presence AUC: codes={res['probe']['auc_codes']:.3f} "
          f"tokenizer_out={res['probe']['auc_tokenizer_out']:.3f} "
          f"backbone_out={res['probe']['auc_backbone_out']:.3f}", flush=True)
except Exception as e:
    import traceback; print(f"[WARN] probe failed: {e}", flush=True); traceback.print_exc()

# ============ (2) GRADIENT-SHARE + PER-LAYER ECE GRAD ============
try:
    model.train()
    if hasattr(core.backbone, "grad_checkpoint"):
        core.backbone.grad_checkpoint = False   # need block-output grads intact for hooks
    layer_g = {}
    hooks = []
    for i, blk in enumerate(core.backbone.blocks):
        def mk(i):
            def hook(m, gi, go):
                g = go[0]
                if g is not None and g.dim() == 3:
                    layer_g[i] = float(g[:, ece_slice.start:ece_slice.stop].norm().item())
            return hook
        hooks.append(blk.register_full_backward_hook(mk(i)))
    ld1 = DataLoader(ds, batch_size=OVERFIT_BS, shuffle=False, num_workers=2, collate_fn=collate_fn)
    batch = next(iter(ld1))
    model.zero_grad(set_to_none=True)
    total, per_mod = compute_step_loss(model, batch, dev)
    total.backward()
    # per-modality param grad-norm share
    def gnorm(params):
        return float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None) + 1e-20))
    mod_share = {}
    for cfg in core.diagnostics:
        pp = list(core.diag_tokenizers[cfg.name].parameters()) + list(core.diag_heads[cfg.name].parameters())
        mod_share[cfg.name] = gnorm(pp)
    bb = gnorm(core.backbone.parameters())
    for h in hooks:
        h.remove()
    res["grad_share"] = {"backbone_grad_norm": bb,
                         "per_modality_param_grad_norm": {k: round(v, 4) for k, v in sorted(mod_share.items(), key=lambda x: -x[1])},
                         "ece_per_layer_grad": [round(layer_g.get(i, float("nan")), 5) for i in range(len(core.backbone.blocks))]}
    print(f"[fx] GRAD-SHARE backbone={bb:.3f} | per-modality(top): " +
          ", ".join(f"{k}={v:.3f}" for k, v in sorted(mod_share.items(), key=lambda x: -x[1])[:6]), flush=True)
    plg = res["grad_share"]["ece_per_layer_grad"]
    print(f"[fx] ECE per-layer grad (blocks 0..47): first={plg[0]} mid={plg[len(plg)//2]} last={plg[-1]} "
          f"min={np.nanmin(plg):.4g} max={np.nanmax(plg):.4g}", flush=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(range(len(plg)), plg, marker="."); ax.set_yscale("log")
    ax.set_xlabel("backbone block"); ax.set_ylabel("grad norm @ ece token positions")
    ax.set_title("per-layer gradient at ECE token positions (48 blocks)")
    fig.tight_layout(); fig.savefig(OUT / "backbone_ece_per_layer_grad.pdf"); plt.close(fig)
except Exception as e:
    import traceback; print(f"[WARN] grad-share failed: {e}", flush=True); traceback.print_exc()

# ============ (3) OVERFIT-ONE-BATCH ============
try:
    model.train()
    if hasattr(core.backbone, "grad_checkpoint"):
        core.backbone.grad_checkpoint = True
    ld2 = DataLoader(ds, batch_size=OVERFIT_BS, shuffle=False, num_workers=2, collate_fn=collate_fn)
    fixed = next(iter(ld2))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    traj = []
    for s in range(OVERFIT_STEPS):
        opt.zero_grad(set_to_none=True)
        total, per_mod = compute_step_loss(model, fixed, dev)
        total.backward(); opt.step()
        if s % 20 == 0 or s == OVERFIT_STEPS - 1:
            ca = per_mod.get(f"{MOD}_codeacc", float("nan"))
            traj.append({"step": s, "loss": round(float(total.item()), 4),
                         "ece_ce": round(per_mod.get(f"{MOD}_ce", float("nan")), 4),
                         "ece_codeacc": round(ca, 4)})
            print(f"[fx] OVERFIT step {s}: loss={total.item():.4f} ece_ce={per_mod.get(MOD+'_ce'):.4f} "
                  f"ece_codeacc={ca:.4f}", flush=True)
    res["overfit_one_batch"] = {"bs": OVERFIT_BS, "steps": OVERFIT_STEPS, "trajectory": traj,
                                "final_ece_codeacc": traj[-1]["ece_codeacc"] if traj else None,
                                "verdict": ("HEALTHY (codeacc->~1)" if traj and traj[-1]["ece_codeacc"] > 0.9
                                            else "SLUGGISH/FAILED — mechanical bug suspected")}
    print(f"[fx] OVERFIT verdict: {res['overfit_one_batch']['verdict']} "
          f"(final ece_codeacc={res['overfit_one_batch']['final_ece_codeacc']})", flush=True)
except Exception as e:
    import traceback; print(f"[WARN] overfit failed: {e}", flush=True); traceback.print_exc()

json.dump(res, open(OUT / "backbone_forensics.json", "w"), indent=2, default=lambda o: float(o))
print("\n[fx] done", flush=True)
