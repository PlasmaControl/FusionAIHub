"""Fittability probe: can a FRESH plain head predict encode_target(INPUT) from
the WARM backbone tokens? CE must -> 0 if the tokens carry the code info.
Isolates 'do the tokens contain it' from MaskGIT/masking/optimization."""
import os
import sys
from pathlib import Path

FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training"):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from eval_e2e_animation_tokamak import load_model
from train_e2e_stage1 import build_datasets, forward_batch, _core
from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.e2e.output_heads import (
    SpectrogramCodeHead, SpectrogramMaskGITHead,
)

dev = torch.device("cuda")
CKPT = os.environ.get(
    "CKPT", "/lustre/orion/fus187/proj-shared/models/e2e_stage1_allshots_b32/e2e_stage1_best.pt")
model, ckpt = load_model(Path(CKPT), dev)
model.eval()
core = _core(model)
a = ckpt["args"]
dn = [d["name"] for d in ckpt["diagnostics"]]
an = [c["name"] for c in ckpt["actuators"]]
dd = Path(a["data_dir"])
stats = torch.load(a["stats_path"], weights_only=False)
sf = dd / "200729_processed.h5"
_, ds = build_datasets(
    dd, [sf], [sf], stats, a["chunk_duration_s"],
    a.get("prediction_horizon_s", a["chunk_duration_s"]),
    a["step_size_s"], a["warmup_s"], dn, an, Path(f"{FMH}/eval_runs/modecode_cache"))
ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn)
batch = next(iter(ld))
with torch.no_grad():
    _, diag_inputs, targets, _, tok = forward_batch(model, batch, dev)
spec = [n for n in dn
        if isinstance(core.diag_heads[n], (SpectrogramCodeHead, SpectrogramMaskGITHead))]
print(f"ckpt step={ckpt.get('step')} | probing {spec}", flush=True)
print("PROBE: fresh plain head, WARM tokens -> encode_target(INPUT). CE must ->0 if fittable.", flush=True)
def fit_probe(tag, X, Y):
    B, N, dim = Y.shape
    L = int(Y.max().item()) + 1 if Y.numel() else 16
    d = X.shape[-1]
    probe = nn.Sequential(
        nn.Linear(d, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(),
        nn.Linear(1024, dim * 16)).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for it in range(3001):
        lg = probe(X).view(B, N, dim, 16)
        ce = F.cross_entropy(lg.reshape(-1, 16), Y.reshape(-1))
        opt.zero_grad(); ce.backward(); opt.step()
        if it % 1000 == 0:
            acc = (lg.argmax(-1) == Y).float().mean().item()
            print(f"  [{tag}] iter {it:4d}  CE={ce.item():.4f}  codeacc={acc:.3f}", flush=True)

for n in spec:
    head = core.diag_heads[n]
    X = tok[n].detach().float()                                  # (B,n_tok,d_model) tokens
    with torch.no_grad():
        Y_ae = head.encode_target(diag_inputs[n]).long()         # AUTOENCODE: input's codes
        Y_fc = head.encode_target(targets[n]).long()             # FORECAST: NEXT window's codes
    fit_probe(f"{n}/AUTOENCODE", X, Y_ae)
    fit_probe(f"{n}/FORECAST", X, Y_fc)
print("DONE", flush=True)
