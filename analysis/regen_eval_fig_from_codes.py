"""Regenerate the seed0 comparison figure from STORED rollout codes (no new predictions).

Decodes the saved GT + prediction codes through the pinned scaling-snapshot codecs (the same
snapshot the run's fine-tune started from; GT and pred share the decoder, so the comparison
convention holds) and recomputes the denorm stats from the data via the test's own helpers.
"""
import os
import sys
from pathlib import Path

os.environ["IGNITE_OVERFIT_SHOT"] = "200729"
os.environ["IGNITE_E2E_T0"] = "0"                      # seed0 windowing (ramp-up included)
REPO = Path("/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub")
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "e2e_test", REPO / "tests/ignite/test_e2e_overfit_realshot.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

from tokamak_foundation_model.ignite import eval_dynamics as ed
from tokamak_foundation_model.ignite import train_dynamics as td

OUT = REPO / "eval_runs/ignite_e2e_overfit_200729_seed0"
PIN = REPO / "eval_runs/ignite_e2e_scaling_200729/codecs"
rc = torch.load(OUT / "rollout_codes_200729.pt", map_location="cpu", weights_only=False)
gt, pred, k0, F = rc["gt_codes"], rc["pred_codes"], rc["k0"], rc["F"]
print(f"stored rollout: k0={k0} F={F} T={rc['temperature']}", flush=True)

# Prefer the run's OWN fine-tuned codecs (saved since 2026-08-10). Decoding stored codes
# with any other codec is an ENCODER/DECODER MISMATCH — it produced the checkerboard and
# washed-out video that made an earlier re-render look far worse than the model actually is.
FT = OUT / "codecs_finetuned"
codecs, denorm, raw_gt = {}, {}, {}
for name in ed.EVAL_MODALITIES:
    fam = td.FROZEN_CODEC_CKPTS[name][0]
    ft = FT / f"{name}.pt"
    if ft.exists():
        import torch as _t
        ck = _t.load(ft, map_location="cpu", weights_only=False)
        codec, cfg = td._build_codec_from(ck) if hasattr(td, "_build_codec_from") else (None, None)
        if codec is None:                      # build via the family loader's class map
            from tokamak_foundation_model.ignite.codec import SpectroCodec
            from tokamak_foundation_model.ignite.video_codec import VideoCodec
            from tokamak_foundation_model.ignite.slow_ts_codec import SlowTSCodec
            from tokamak_foundation_model.ignite.fastts_codec import FastTSCodec
            cls = {"spectro": SpectroCodec, "video": VideoCodec,
                   "slowts": SlowTSCodec, "fastts": FastTSCodec}[ck["family"]]
            cfg = ck["cfg"]; codec = cls(cfg); codec.load_state_dict(ck["codec"])
            codec.eval()
            for _pp in codec.parameters():
                _pp.requires_grad_(False)
        print(f"  {name}: FINE-TUNED codec", flush=True)
    else:
        codec, cfg = td._load_codec(fam, PIN / name / "codec_best.pt")
    codecs[name] = (codec, cfg, fam)
    # RAW windows = the MEASURED signal, on this run's own windowing (T0=0), which is exactly
    # the frame grid the stored codes were encoded from. Used as the figure's ground truth
    # (user 2026-08-12) instead of the codec round-trip, and still as the video denorm stats.
    wins_all, masks_all = t._windows_and_masks(name, fam, cfg)
    wins = wins_all[:F]
    msk = masks_all[:F].detach().cpu().numpy() if masks_all is not None else None
    denorm[name] = t._make_denorm(name, fam, cfg, wins if fam == "video" else None)
    # Convert the raw window into the SAME space as the denormalized decode. This is
    # family-specific: video raw is already physical, slow-TS/spectro raw need the denorm,
    # and slow-TS missing samples must become NaN. Getting this wrong rendered the video
    # predictions solid black (raw inflated ~60x, blowing out the shared colour limits).
    raw_gt[name] = ed.raw_to_output_space(fam, wins.detach().cpu().numpy(),
                                          denorm_fn=denorm[name], mask=msk)
    _r = raw_gt[name]
    print(f"  {name}: codec + denorm ready, raw GT {tuple(_r.shape)} "
          f"range [{np.nanmin(_r):.4g}, {np.nanmax(_r):.4g}]", flush=True)

decoded = ed.decode_all(codecs, gt, pred, k0, F, torch.device("cpu"),
                        denorm=denorm, raw_gt=raw_gt)
_src = {n: d.get("gt_source") for n, d in decoded.items()}
print(f"GT source per modality: {_src}", flush=True)
png, pdf = ed.render_figure(decoded, "200729", 1500, rc["temperature"], k0, F, OUT,
                            t_origin=0.0)
print(f"REGEN_OK {png}", flush=True)
