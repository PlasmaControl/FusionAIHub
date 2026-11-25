"""MaskGIT decode, step by step, for the Mirnov OMV tokens of the first predicted frame of one shot.

    IGNITE_MASKGIT_GUMBEL=3 scripts/mast/py.sh scripts/mast/maskgit_cascade_probe.py \
        $S/runs/v1b_binspec_cam_live_da_tokeye_f32_d512L12/dynamics_step5000.pt \
        $S/frame_codes_v1b_binspec_cam_live_da 23529 --k0 12

Rolls out one frame (horizon 1) with the scoring decode ("not recorded" forbidden where the last
seed frame records) on the CPU. For each decode step it prints, over the OMV tokens still masked at
the start of the step: their number, the mean expected code E[code] under the model's distribution,
the mean probability of a code at or above the truth frame's mean, and the mean code of the tokens
committed in that step. Reference: the mean truth code of the predicted frame, of the last seed frame
and of seed frames 3 and 4 before its end. Only meaningful for the per-bin codec (BinSpec), whose
codes are ordered power levels (0 = quietest); band-DCT codes are DCT coefficients. A falling
E[code] over the steps is a cascade onto quiet. Wraps the model's methods at runtime; weights and
maskgit.py are untouched.

Measured 2026-11-25, run 10 step 5,000, Gumbel 3: on 23529 (k0 12; truth 30.7, last seed 26.2, quiet
seed frames 14.7) E[code] falls 25.2 -> 18.4 over the 10 steps and the frame ends at 17.4; on 20384
with a strongly lit seed end (k0 27; truth 49.8, last seed 51.1) it stays at 49 to 50 and ends at
51.1 (docs/models/ignite-mast-spectrogram-quality.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tokamak_foundation_model.ignite import eval_dynamics as ev
from tokamak_foundation_model.ignite.mast.port import rollout as port_rollout

OMV = "magnetics-b_field_pol_probe_omv_voltage"
MISS = 64  # framecache.codec_vocab: missing_code = n_bins (64 for the band-DCT and BinSpec codecs)

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("cache_dir")
ap.add_argument("shot")
ap.add_argument("--k0", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = torch.device("cpu")
model, cfg, step = port_rollout.load_mast_model(a.ckpt, a.cache_dir, dev)
mask_id = model._mask_id(OMV)
inputs, probs, state = [], [], {"active": True}

_orig_forward = model.backbone.forward


def _forward(sequence, *args, **kw):
    if state["active"]:
        inputs.append(sequence[OMV][0, -1].clone())
    return _orig_forward(sequence, *args, **kw)


model.backbone.forward = _forward
if hasattr(model.backbone, "next_frame_logits"):
    _orig_nfl = model.backbone.next_frame_logits

    def _nfl(cache, frame, *args, **kw):
        if state["active"]:
            inputs.append(frame[OMV][0, -1].clone())
        return _orig_nfl(cache, frame, *args, **kw)

    model.backbone.next_frame_logits = _nfl

_orig_sample = model.output.sample
names = [m.name for m in cfg.modalities]
omv_i, n_mod = names.index(OMV), len(names)
calls = {"n": 0}


def _sample(logits, temperature, generator):
    out = _orig_sample(logits, temperature, generator)
    if state["active"]:
        if calls["n"] % n_mod == omv_i:
            probs.append(torch.softmax(logits.float(), -1)[0].numpy())
        calls["n"] += 1
    return out


model.output.sample = _sample
_orig_gen = model.generate_frame
final = {}


def _gen(*args, **kw):
    out = _orig_gen(*args, **kw)
    if state["active"]:
        frame = out[0] if isinstance(out, tuple) else out
        final["codes"] = frame[OMV][0].clone()
        state["active"] = False
    return out


model.generate_frame = _gen

entry = ev.load_shot_cache(Path(a.cache_dir), str(a.shot))
truth = entry["codes"][OMV].long()
last = truth[a.k0 - 1]
forbid = {OMV: torch.where(last == MISS, torch.zeros_like(last), torch.full_like(last, MISS))}
n_seed, generator = a.k0, torch.Generator(device=dev).manual_seed(a.seed)
view = {**entry, "n_frames": min(int(entry["n_frames"]), n_seed + 1)}
ev.rollout_shot(model, cfg, view, n_seed, 1.0, generator, dev, actuator_mode="real",
                cache_dir=Path(a.cache_dir), shot=str(a.shot), decode={"forbid": forbid})


def mean_code(c):
    c = c[c < MISS]
    return float(c.float().mean()) if c.numel() else float("nan")


print(f"shot {a.shot} k0 {a.k0} ckpt step {step}: truth mean code predicted frame "
      f"{mean_code(truth[a.k0]):.1f}, last seed frame {mean_code(truth[a.k0 - 1]):.1f}, "
      f"seed frames -3/-4 {mean_code(truth[a.k0 - 3]):.1f}/{mean_code(truth[a.k0 - 4]):.1f}")
levels = np.arange(probs[0].shape[-1])
levels[MISS:] = 0
fin = final["codes"]
print(f" captured inputs {len(inputs)} probs {len(probs)}; first input all masked: {bool((inputs[0] == mask_id).all())}")
n_steps = min(len(inputs), len(probs))
for s in range(n_steps):
    masked = (inputs[s] == mask_id).numpy()
    e = (probs[s] * levels).sum(-1)
    nxt = (inputs[s + 1] == mask_id).numpy() if s + 1 < n_steps else np.zeros_like(masked)
    new = masked & ~nxt
    newc = fin.numpy()[new]
    newc = newc[newc < MISS]
    print(f" step {s}: masked {int(masked.sum()):3d}  E[code|masked] {e[masked].mean():5.1f}  "
          f"P(code>=truth mean) {(probs[s][masked][:, int(mean_code(truth[a.k0])):MISS].sum(-1)).mean():.2f}  "
          f"committed now {new.sum():3d} mean {newc.mean() if newc.size else float('nan'):5.1f}")
print(f" final frame mean code {mean_code(fin):.1f}")
