"""Unit tests for the new architecture components (CPU, seconds).

Verifies IN ISOLATION that:
  1. persistence anchor: prediction_on - prediction_off == input window (exactly)
  2. anchor makes the prediction carry the input (correlates → visible-mode floor)
  3. backbone skip: gated residual exists, inits at 0.2, and changes the output
  4. anchor + skip do not break the return_tokens path (Stage-2 needs it)

Run: pixi run --frozen python scripts/training/test_arch_components.py
"""
import sys
sys.path.insert(0, "src")
import torch
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig, DiagnosticConfig, E2EFoundationModel,
)
from tokamak_foundation_model.e2e.output_heads import SpectroFreqWarpHead

DIAG = [DiagnosticConfig("ece", "spectrogram", n_channels=4, window_samples=48,
                         freq_bins=64, spectrogram_patch_size=(32, 16))]
ACT = [ActuatorConfig("nbi", n_channels=2, window_samples=48, n_tokens=2)]
B = 2
SI = torch.zeros(B, dtype=torch.long)
TO = torch.zeros(B)


def build(skip, anchor, warp=False):
    torch.manual_seed(0)
    return E2EFoundationModel(diagnostics=DIAG, actuators=ACT, d_model=32,
                              n_layers=2, n_heads=2,
                              backbone_input_skip=skip,
                              spec_persistence_anchor=anchor,
                              spec_warp_anchor=warp).eval()


def inputs():
    torch.manual_seed(1)
    d = {}
    for c in DIAG:
        if c.kind == "spectrogram":
            d[c.name] = torch.randn(B, c.n_channels, c.freq_bins, c.window_samples)
        else:
            d[c.name] = torch.randn(B, c.n_channels, c.window_samples)
    a = {c.name: torch.randn(B, c.n_channels, c.window_samples) for c in ACT}
    return d, a


npass = nfail = 0
def check(name, ok, detail=""):
    global npass, nfail
    npass += ok; nfail += (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


d, a = inputs()

# 1 + 2: anchor math + carries the input
m = build(skip=False, anchor=True)
with torch.no_grad():
    p_on = m(d, a, SI, TO)["ece"]
    m.spec_persistence_anchor = False
    p_off = m(d, a, SI, TO)["ece"]
t = p_on.shape[-1]
diff = (p_on - p_off - d["ece"][..., :t]).abs().max().item()
check("anchor: pred_on - pred_off == input window", diff < 1e-4, f"max|diff|={diff:.2e}")
corr = torch.corrcoef(torch.stack([p_on.flatten(), d["ece"][..., :t].flatten()]))[0, 1].item()
check("anchor: prediction carries the input (visible floor)", corr > 0.3, f"corr={corr:.3f}")

# 3: gated backbone skip
ms = build(skip=True, anchor=False)
check("skip: gate param exists + inits ~0.2", hasattr(ms, "backbone_skip_gate")
      and abs(float(ms.backbone_skip_gate) - 0.2) < 1e-6,
      f"gate={float(ms.backbone_skip_gate):.3f}" if hasattr(ms, "backbone_skip_gate") else "MISSING")
with torch.no_grad():
    p_skip = ms(d, a, SI, TO)["ece"]
    ms.backbone_input_skip = False
    p_noskip = ms(d, a, SI, TO)["ece"]
changed = (p_skip - p_noskip).abs().max().item()
check("skip: changes the output (residual is active)", changed > 1e-5, f"max|diff|={changed:.2e}")

# 4: return_tokens intact with both on
m2 = build(skip=True, anchor=True)
with torch.no_grad():
    out = m2(d, a, SI, TO, return_tokens=True)
ok = isinstance(out, tuple) and len(out) == 2 and "ece" in out[0] and "ece" in out[1]
check("return_tokens: (predictions, token_slices) intact with anchor+skip", ok)

# 5: WARP anchor — identity at init (zero-init shift → warp(input)==input →
#    pred_on - pred_off == input, exactly like the additive anchor).
mw = build(skip=False, anchor=False, warp=True)
check("warp: head + warp_head built", "ece" in mw.spec_warp_heads)
with torch.no_grad():
    pw_on = mw(d, a, SI, TO)["ece"]
    mw.spec_warp_anchor = False
    pw_off = mw(d, a, SI, TO)["ece"]
t = pw_on.shape[-1]
wdiff = (pw_on - pw_off - d["ece"][..., :t]).abs().max().item()
check("warp: identity at init (pred_on - pred_off == input window)",
      wdiff < 1e-3, f"max|diff|={wdiff:.2e}")

# 6: WARP MOVES a ridge. Force a known constant +K-bin shift; a delta ridge at
#    freq f0 in the input must appear at f0+K in the warped output.
Fb, T, K = 64, 32, 5
head = SpectroFreqWarpHead(d_model=32, n_channels=1, n_patches_f=2,
                           n_patches_t=2, freq_bins=Fb, trunc_t=T,
                           max_shift_bins=8.0).eval()
# proj is zero-init; set bias so tanh(bias)*8 == K → constant shift K.
import math as _m
with torch.no_grad():
    head.proj.bias.fill_(_m.atanh(K / 8.0))
ridge = torch.zeros(1, 1, Fb, T)
f0 = 20
ridge[0, 0, f0, :] = 1.0
toks = torch.zeros(1, 4, 32)  # n_tok = n_pf*n_pt = 4
with torch.no_grad():
    warped = head(toks, ridge)
peak = int(warped[0, 0, :, T // 2].argmax())
check(f"warp: +{K}-bin shift moves ridge {f0} -> {f0 + K}",
      abs(peak - (f0 + K)) <= 1, f"peak at {peak} (want {f0 + K})")

# 7: return_tokens intact with warp+skip
m3 = build(skip=True, anchor=False, warp=True)
with torch.no_grad():
    out3 = m3(d, a, SI, TO, return_tokens=True)
ok3 = isinstance(out3, tuple) and len(out3) == 2 and "ece" in out3[0]
check("return_tokens: intact with warp+skip", ok3)

print(f"\n=== UNIT TESTS: {npass} passed, {nfail} failed ===", flush=True)
sys.exit(1 if nfail else 0)
