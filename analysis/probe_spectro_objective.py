"""DECODER-ONLY OBJECTIVE SCREEN: which loss term buys MODE STRUCTURE, in ~1.5 min per arm.

Cheap login-GPU screen for the spectro objective before a knob costs a 2 h leg. Same pattern
the mhr lattice work used (head-only fine-tunes of a fixed baseline under different
objectives), so the codes are held FIXED and only the decoder moves -- which is exactly the
attribution we want: any difference is the OBJECTIVE, not a different codebook.

  * encoder + quantizer FROZEN and in eval mode, so every arm sees the IDENTICAL codes.
  * decoder fine-tuned from the baseline checkpoint for --steps steps at --lr 1e-4 (the
    MEASURED codec fine-tune optimum).
  * arms are ``name:field=value,field=value`` cfg overrides -- any SpectroCodecConfig field.
  * every metric is measured against the RAW held-out window, on shots DISJOINT from the
    fine-tune pool.

IT IS A SCREEN, NOT A VERDICT. It is decoder-only, short, and on ~96 windows; the recorded
rule is that only >= 300-window audits rank arms (a 16-window co2 smoke once produced a wrong
"MS-SSIM harms co2" verdict). What it is good for is killing a knob cheaply and ordering the
survivors. Calibration check that it is not nonsense: the shipped co2 ms5 arm reads 15% of
its coherent hf ceiling here (0.0477 / 0.322) and 15% on the 320-window audit
(0.037 / 0.250), i.e. the FRACTION-OF-CEILING transfers even though the absolute pool differs.

FIRST RESULT (co2, from ms5 @55001, 1200 steps): target_time_smooth is a NEGATIVE.
    K=0  hf 0.0477 (15% of ceiling)  nRMSE 0.5446  std_r 0.765  corr2d 0.7588
    K=5  hf 0.0277 ( 9% of ceiling)  nRMSE 0.5463  std_r 0.743  corr2d 0.7545
Smoothing the target does not add pressure to be sharp, it REMOVES it: the smoothed target is
easier to fit, so the decoder settles on a smoother solution. The speckle in the raw target
was, if anything, pushing hf UP.

Reported per arm: hf_ratio and its % of the coherent ceiling (the ranking key),
patch_lattice_ratio with its GT control, std_ratio, spec_nrmse (a floor), spec_corr2d.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.spectro_final_fig import (  # noqa: E402
    hf_references,
    masked_std_ratio,
    window_pool,
)
from tokamak_foundation_model.ignite import gate  # noqa: E402
from tokamak_foundation_model.ignite import spike as _spike  # noqa: E402
from tokamak_foundation_model.ignite import train_codec as _tc  # noqa: E402


def tc_spectro_live(modality: str):
    """The trainer's own shot list for this modality, presence-filtered, in trainer order."""
    return _tc.spectro_live_shots(
        modality, _spike.discover_shots(_tc.DEFAULT_DATA_DIR), log_fn=print)
from tokamak_foundation_model.ignite.codec import SpectroCodec  # noqa: E402
from tokamak_foundation_model.ignite.discriminator import FreqAwarePatchGAN  # noqa: E402
from tokamak_foundation_model.ignite.losses import time_smooth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", required=True)
    ap.add_argument("--ckpt", required=True, help="baseline checkpoint to fine-tune from")
    ap.add_argument("--arms", default="base:",
                    help="semicolon list of 'name:field=value,field=value' cfg overrides, "
                         "e.g. 'base:;fg20:freq_grad_weight=20;sm5:target_time_smooth=5'. "
                         "Any SpectroCodecConfig field; ints/floats/bools are coerced from "
                         "the existing field's type. An empty override list = the "
                         "checkpoint's own objective (the control).")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="1e-4 is the MEASURED codec fine-tune optimum (raising it was the "
                         "single biggest lever on the codec objective study)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=192, help="windows in the fit pool")
    ap.add_argument("--n_eval", type=int, default=96, help="windows in the held-out pool")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None)
    ap.add_argument("--save_dir", default=None,
                    help="write <save_dir>/<arm>/codec_best.pt per arm, so the winner can be "
                         "RENDERED (analysis/spectro_final_fig.py --mode figure). A high "
                         "hf_ratio can be checkerboard rather than modes, so a screen result "
                         "must be LOOKED AT before it costs a training leg.")
    args = ap.parse_args()

    specs = []
    for tok in args.arms.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        name, _, kv = tok.partition(":")
        over = {}
        for item in kv.split(","):
            if not item.strip():
                continue
            f, _, v = item.partition("=")
            over[f.strip()] = v.strip()
        specs.append((name, over))
    if not specs:
        raise SystemExit("--arms is required")

    dev = args.device
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg0 = ck["cfg"]
    print(f"[probe] {args.modality} from {args.ckpt} (step {ck.get('step')}), "
          f"ms_ssim={cfg0.ms_ssim_weight} pixel={cfg0.pixel_anchor_weight} "
          f"multiscale={cfg0.multiscale_recon_weight} adv={cfg0.adversarial_weight} "
          f"freq_grad={getattr(cfg0, 'freq_grad_weight', 0.0)}", flush=True)

    # SHOT POOLS. The fine-tune pool must come from the TRAINER'S TRAIN SPLIT and the eval
    # pool from its HELD-OUT split -- not two halves of the held-out 16.
    #
    # 2026-09-04: the first version fit on held_out[:8] and evaluated on held_out[-4:], which
    # is internally clean but makes the resulting checkpoint UNSCOREABLE by the >=300-window
    # audit, because that audit pools all 16 held-out shots and 8 of them were fine-tuned on.
    # Fitting inside the train split keeps the probe's own comparison clean AND lets the audit
    # score a probe checkpoint on the same footing as a real training arm.
    live = tc_spectro_live(args.modality)
    ev_shots = live[-16:][-4:]
    _train_pool = live[:-16]
    fit_shots = _train_pool[:: max(1, len(_train_pool) // 8)][:8]
    Xtr, Mtr = window_pool(args.modality, cfg0, fit_shots, args.n_train)
    Xte, Mte = window_pool(args.modality, cfg0, ev_shots, args.n_eval)
    print(f"[probe] fit {Xtr.shape} on {fit_shots}\n[probe] eval {Xte.shape} on {ev_shots}",
          flush=True)
    ref = hf_references(Xte, cfg0)
    print(f"[probe] hf calibration on the EVAL pool: coherent ceiling "
          f"{ref['hf_tsmooth']:.3f}   patch-level floor {ref['hf_patchmean']:.3f}", flush=True)

    xtr = torch.from_numpy(Xtr).to(dev, torch.float32)
    mtr = torch.from_numpy(Mtr).to(dev, torch.float32)
    rows = []
    for name, over in specs:
        torch.manual_seed(0)
        cfg = ck["cfg"]
        codec = SpectroCodec(cfg).to(dev)
        codec.load_state_dict(ck["codec"])
        # Overrides are applied AFTER the strict load, so every arm starts from the identical
        # weights; only fields that do not change a weight SHAPE may be set here (a decoder-
        # only screen cannot add parameters -- gain_shape, channel_groups, refine_depth and
        # d_model all need a full retrain and are deliberately rejected).
        _shape_changing = {"gain_shape", "gain_tokens", "channel_groups", "refine_depth",
                           "refine_hidden", "d_model", "enc_depth", "dec_depth", "heads",
                           "decoder", "fsq_levels", "patch_f", "patch_t", "freq_bins",
                           "time_frames", "decoder_noise"}
        for f, v in over.items():
            if f in _shape_changing:
                raise SystemExit(
                    f"arm {name}: {f} changes a weight shape or adds parameters, which a "
                    f"decoder-only screen cannot do -- it needs a full training arm.")
            if not hasattr(cfg, f):
                raise SystemExit(f"arm {name}: SpectroCodecConfig has no field {f!r}")
            cur = getattr(cfg, f)
            if isinstance(cur, bool):
                setattr(cfg, f, v.lower() in ("1", "true", "yes"))
            elif isinstance(cur, int):
                setattr(cfg, f, int(v))
            elif isinstance(cur, float):
                setattr(cfg, f, float(v))
            elif isinstance(cur, (tuple, list)):
                setattr(cfg, f, tuple(int(x) for x in v.split("|") if x))
            else:
                setattr(cfg, f, v)
        print(f"  --- arm {name}: " + (", ".join(f"{f}={getattr(cfg, f)}" for f in over)
                                       or "CONTROL (checkpoint objective)"), flush=True)
        disc = FreqAwarePatchGAN(cfg).to(dev)
        # FREEZE the code path: identical codes for every arm.
        for p_ in codec.encoder.parameters():
            p_.requires_grad_(False)
        for p_ in codec.quantizer.parameters():
            p_.requires_grad_(False)
        codec.encoder.eval()
        codec.quantizer.eval()
        opt = torch.optim.Adam(
            [p_ for p_ in codec.decoder.parameters() if p_.requires_grad], lr=args.lr)
        # The DISCRIMINATOR IS TRAINED TOO whenever it is weighted. Without this the
        # adversarial term is "fool a frozen random D", which is a random-feature HF reward
        # rather than the objective the arm would actually run under -- and adversarial
        # pressure is precisely what this screen exists to rank. Uses the trainer's own
        # masked discriminator step so the two halves agree about what data is real.
        want_adv = (float(cfg.adversarial_weight) > 0.0 or float(cfg.fm_weight) > 0.0)
        opt_d = (torch.optim.Adam(disc.parameters(), lr=args.lr) if want_adv else None)
        t0 = time.time()
        n = xtr.shape[0]
        for step in range(args.steps):
            i = (step * args.batch) % max(1, n - args.batch)
            xb, mb = xtr[i:i + args.batch], mtr[i:i + args.batch]
            out = codec.generator_losses(xb, xb, disc, cfg, step=10_000, frame_mask=mb)
            opt.zero_grad(set_to_none=True)
            out["total"].backward()
            opt.step()
            if opt_d is not None and step % max(1, int(cfg.disc_update_every)) == 0:
                opt_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    rec_d = codec.forward(xb)["recon"]
                d_loss = _tc._spectro_discriminator_loss(disc, xb, rec_d, cfg, mb)
                d_loss.backward()
                opt_d.step()
            if step % 400 == 0:
                print(f"    {name} step {step}: total {float(out['total']):.4f} "
                      f"pixel {float(out['pixel']):.4f} ms_ssim {float(out['ms_ssim']):.4f}",
                      flush=True)
        codec.eval()
        rec = []
        with torch.no_grad():
            for i in range(0, Xte.shape[0], 8):
                xb = torch.from_numpy(Xte[i:i + 8]).to(dev, torch.float32)
                rec.append(codec.forward(xb)["recon"].cpu().numpy())
        R = np.concatenate(rec, 0)
        # scored against the RAW held-out window, always
        met = gate.decode_fidelity(R, Xte, patch_f=cfg.patch_f, patch_t=cfg.patch_t,
                                  full_spec=True, band_bins=None, mask=Mte)
        row = {
            "arm": name,
            "overrides": over,
            "hf_ratio": met["sharpness"],
            "hf_frac_of_ceiling": met["sharpness"] / max(ref["hf_tsmooth"], 1e-12),
            # peak_f1 is the TRACK metric: overlap of the top-k spectral peaks. hf_ratio and
            # std_ratio can both be satisfied by the RIGHT KIND OF TEXTURE in the WRONG PLACE
            # (measured: the co2 `combo` arm reaches hf 82% of ceiling and std_ratio 0.964,
            # and its rendered panel has GT-like granularity and correct burst columns but
            # does NOT reproduce the coherent 10-20 kHz track). Only a peak-coincidence
            # metric separates "looks like a spectrogram" from "is THIS spectrogram".
            "peak_f1": met["peak_f1"],
            "envelope_corr": met["envelope_corr"],
            "lattice": met["patch_lattice_ratio"],
            "gt_lattice": met["target_patch_lattice_ratio"],
            "std_ratio": masked_std_ratio(R, Xte, Mte),
            "nrmse": met["spec_nrmse"],
            "corr2d": met["spec_corr2d"],
            "tmean_nrmse": met["base_tmean_spec_nrmse"],
            "wcmean_nrmse": met["base_wcmean_spec_nrmse"],
            "minutes": (time.time() - t0) / 60.0,
        }
        rows.append(row)
        if args.save_dir:
            d = Path(args.save_dir) / name
            d.mkdir(parents=True, exist_ok=True)
            # Same keys analysis/spectro_final_fig.load_codec and the audit read.
            torch.save({"cfg": cfg, "codec": codec.state_dict(),
                        "step": int(ck.get("step") or 0) + args.steps,
                        "probe": {"arm": name, "overrides": over, "from": args.ckpt}},
                       d / "codec_best.pt")
            print(f"      saved {d / 'codec_best.pt'}", flush=True)
        print(f"  {name}: hf {row['hf_ratio']:.4f} ({100 * row['hf_frac_of_ceiling']:.0f}% of "
              f"ceiling)  peak_f1 {row['peak_f1']:.4f}  lattice {row['lattice']:.1f} "
              f"(GT {row['gt_lattice']:.2f})  std_r {row['std_ratio']:.3f}  "
              f"nRMSE {row['nrmse']:.4f}  corr2d {row['corr2d']:.4f}  "
              f"[{row['minutes']:.1f} min]", flush=True)
        del codec, disc, opt, opt_d
        torch.cuda.empty_cache()

    print(f"\n{'arm':<14}{'hf':>9}{'%ceil':>8}{'peak_f1':>9}{'lattice':>9}{'std_r':>8}"
          f"{'nRMSE':>9}{'corr2d':>9}{'env_corr':>10}")
    for r in rows:
        print(f"{r['arm']:<14}{r['hf_ratio']:>9.4f}"
              f"{100 * r['hf_frac_of_ceiling']:>8.0f}{r['peak_f1']:>9.4f}"
              f"{r['lattice']:>9.1f}{r['std_ratio']:>8.3f}{r['nrmse']:>9.4f}"
              f"{r['corr2d']:>9.4f}{r['envelope_corr']:>10.4f}")
    print(f"  ceiling hf {ref['hf_tsmooth']:.3f}  patch floor {ref['hf_patchmean']:.3f}  "
          f"tmean nRMSE {rows[0]['tmean_nrmse']:.4f}  wcmean {rows[0]['wcmean_nrmse']:.4f}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"ckpt": args.ckpt, "modality": args.modality, "refs": ref, "rows": rows},
            indent=1, default=float))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
