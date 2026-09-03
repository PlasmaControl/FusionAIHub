"""FULL-IGNITE overfit gate (rung 3 of the overfit ladder) — codecs + backbone, shot 200729.

Rungs 1/2 (tests/ignite/test_fsq_overfit_realshot.py) prove the SPECTRO FSQ-AE mechanics in
isolation. This rung proves the WHOLE IGNITE stack on one real shot: every available Phase-A
codec — ALL families, not just spectro — is fine-tuned (recon-only) on shot 200729, the shot is
encoded into frame codes, and a small Phase-B MaskGIT ST-transformer backbone is trained to
overfit that code sequence, then rolled out from K0 real seed frames and decoded back for a
GT-vs-pred comparison figure.

    stage 1  codecs   EVERY modality (all 4 families, filterscopes included) <- WARM START
                      from the CURRENT production checkpoint template
                      (eval_runs/ignite_codec_{m}_v6/codec_best.pt; the ckpt's embedded cfg
                      drives BOTH the model and the window preprocessing). A modality whose
                      v6 run has not landed yet falls back to the previous generation (v5)
                      and finally the frozen manifest (train_dynamics.FROZEN_CODEC_CKPTS) —
                      loudly, per modality — and upgrades automatically once its v6
                      codec_best.pt appears. fastts (filterscopes) joined with v6 (its first
                      gate-passing checkpoint).
                      Fine-tune objective is the family-canonical RECON term only (rung-1/2
                      philosophy: no adversarial / entropy / consistency):
                        spectro  pure pixel MAE on the window
                        fast-TS  pure MAE on the ELM-envelope window (δ-pair items, like spectro)
                        video    masked pixel MAE against x_std (recon lives in the codec's
                                 standardized frame space; dead-camera frames masked out)
                        slow-TS  masked MAE (zero_is_missing / NaN gaps never trained toward)
    stage 2  encode   every codec encodes its consecutive 50 ms frame windows ->
                      {modality: (F, n_tok)} flat FSQ codes + (F, 70) actuator frames.
                      The frame layout is DERIVED from the actual codecs (n_tok / vocab per
                      modality), NOT from the static FROZEN_MODALITIES table — the _v5 codecs
                      change it (ece 768 tok at patch 8x8; bes/co2 vocab 64000 at fsq6).
    stage 3  backbone MaskGITDynamics (factorized ST-transformer, small: d256/depth4 default)
                      trained on random frame windows of THIS shot (masked-token CE), then
                      rollout: seed K0 real frames -> generate + commit codes -> decode both
                      GT and prediction through the SAME codecs -> eval_dynamics figure
                      (seed + rollout timeline, dashed rollout-start marker, seconds axes) in
                      PHYSICAL units: each family's input normalization is INVERTED via
                      _make_denorm (per-window norms use the GT frame's stats — the codes carry
                      no absolute scale). Rollout codes are saved (rollout_codes_{shot}.pt) for
                      offline re-rendering without retraining.

PASS = the full pipeline (data -> codecs -> codes -> ST-backbone -> MaskGIT decode -> rollout
-> decode) trains and memorizes one shot: codec fine-tune MAE does not diverge, masked CE
drops hard, rollout commits valid codes, slow-TS token accuracy far above chance. Asserted
floors are PROVISIONAL collapse-guards (first-run calibration pending — tighten them to the
measured operating point once this has run on 200729, as rungs 1/2 did). The primary
deliverable is the figure + metrics json in IGNITE_E2E_OUT.

REAL-DATA + GPU-scale: needs {SHOT}_processed.h5 AND IGNITE_E2E=1 (opt-in, like rung 2's
IGNITE_FULLSHOT). Launch: sbatch with E2E=1 via scripts/slurm_frontier/ignite_fsq_overfit_test.sh.

Env knobs:
    IGNITE_DATA_DIR           data dir with {shot}_processed.h5 (default: production dir)
    IGNITE_OVERFIT_SHOT       shot number                            (default 200729)
    IGNITE_E2E                "1" enables this test                  (default: off)
    IGNITE_E2E_CKPT_TMPL      extra codec ckpt template, tried FIRST (defaults try
                              ignite_codec_{m}_fsq6, then _v6, then _v5, then the manifest)
    IGNITE_E2E_CODEC_EPOCHS   codec fine-tune epochs over the shot   (default 30)
    IGNITE_E2E_CODEC_BS       codec fine-tune batch size             (default 16)
    IGNITE_E2E_STEPS          backbone MaskGIT steps                 (default 1500)
    IGNITE_E2E_DM             backbone d_model                       (default 256)
    IGNITE_E2E_DEPTH          backbone depth                         (default 4)
    IGNITE_E2E_BS             backbone batch size (frame windows)    (default 2)
    IGNITE_E2E_TRAIN_F        backbone training window (frames)      (default 40)
    IGNITE_E2E_GRAD_CKPT      "1" = backbone activation checkpointing (needed for d1024 cells)
    IGNITE_E2E_K0             rollout seed frames                    (default 20)
    IGNITE_E2E_NPRED          rollout predicted frames               (default 40; 80 = 4 s)
    IGNITE_E2E_T0             window origin in shot time [s]; 0.0 seeds from the ramp-up
                              first second -> prediction starts at t=1.0 s (default 1.0)
    IGNITE_E2E_TEMP           rollout sampling temperature           (default 0.5)
    IGNITE_E2E_OUT            figure/metrics/ckpt output dir         (default eval_runs/ignite_e2e_overfit_{shot})
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
import torch

from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite import train_dynamics as td
from tokamak_foundation_model.ignite.dynamics_config import (
    FROZEN_MODALITIES,
    DynamicsConfig,
    ModalitySpec,
)
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics

SHOT = os.environ.get("IGNITE_OVERFIT_SHOT", "200729")
DATA_DIR = Path(os.environ.get("IGNITE_DATA_DIR", tc.DEFAULT_DATA_DIR))
E2E = os.environ.get("IGNITE_E2E") == "1"
# Candidate checkpoint templates, NEWEST production generation first; per modality the first
# EXISTING checkpoint wins, with the frozen manifest (train_dynamics.FROZEN_CODEC_CKPTS) as
# the final fallback. IGNITE_E2E_CKPT_TMPL prepends an override template. _fsq6 is the
# 2026-08-06/07 video generation (64k vocab); modalities without an _fsq6 dir fall through.
_DEFAULT_TMPLS = (
    "eval_runs/ignite_codec_{m}_fsq6/codec_best.pt",
    "eval_runs/ignite_codec_{m}_v6/codec_best.pt",
    "eval_runs/ignite_codec_{m}_v5/codec_best.pt",
)
_TMPL_ENV = os.environ.get("IGNITE_E2E_CKPT_TMPL")
CKPT_TMPLS = ((_TMPL_ENV,) + _DEFAULT_TMPLS) if _TMPL_ENV else _DEFAULT_TMPLS
CODEC_EPOCHS = int(os.environ.get("IGNITE_E2E_CODEC_EPOCHS", "30"))
CODEC_BS = int(os.environ.get("IGNITE_E2E_CODEC_BS", "16"))
CODEC_LR = 3e-4        # rung-1 calibration: the FSQ path is lr-SENSITIVE; 1e-3 never settles
STEPS = int(os.environ.get("IGNITE_E2E_STEPS", "1500"))
DM = int(os.environ.get("IGNITE_E2E_DM", "256"))
DEPTH = int(os.environ.get("IGNITE_E2E_DEPTH", "4"))
BS = int(os.environ.get("IGNITE_E2E_BS", "2"))
TRAIN_F = int(os.environ.get("IGNITE_E2E_TRAIN_F", "40"))
# activation checkpointing for the BIG scaling cells (d1024 at B2/F40 exceeds a 64 GB GCD
# without it). Numerically identical (recompute in backward); rollout (eval) is unaffected.
GRAD_CKPT = os.environ.get("IGNITE_E2E_GRAD_CKPT") == "1"
K0 = int(os.environ.get("IGNITE_E2E_K0", "20"))
NPRED = int(os.environ.get("IGNITE_E2E_NPRED", "40"))   # predicted frames (80 = 4 s = production)
# Window origin in shot time. 0.0 (default; USER CONVENTION 2026-08-09) = include the
# ramp-up FIRST SECOND so the K0=20 seed covers shot time [0, 1.0) s and the prediction
# starts at t = 1.0 s. 1.0 = the historical warmup-skip windowing. NOTE: the production
# codecs never trained on ramp-up windows, and early-shot windows are more likely
# degenerate (the dataset then redraws a nearby window into that frame slot).
T0 = float(os.environ.get("IGNITE_E2E_T0", "0.0"))
TEMP = float(os.environ.get("IGNITE_E2E_TEMP", "0.5"))
OUT_DIR = Path(os.environ.get("IGNITE_E2E_OUT", f"eval_runs/ignite_e2e_overfit_{SHOT}"))
LOG_EVERY = 50

_H5 = DATA_DIR / f"{SHOT}_processed.h5"
pytestmark = [
    pytest.mark.skipif(not _H5.exists(), reason=f"real-shot data not available: {_H5}"),
    pytest.mark.skipif(not E2E, reason="full-IGNITE overfit is opt-in (GPU-scale): set IGNITE_E2E=1"),
]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------------- #
# stage 1 — codec assembly + recon-only fine-tune (family-canonical masked losses)
# --------------------------------------------------------------------------------------------- #
def _windows_and_masks(name: str, fam: str, cfg):
    """ALL frame windows of the shot in time order (frame t = chunk t; all families share the
    50 ms windowing) -> (wins (n, ...), masks (n, ...) or None), through the codec's own cfg
    pipeline. Spectro items are (spec_a, spec_b) δ-pairs (target = a, no mask); video items are
    (frames, frame_mask); slow-TS items are (signal, mask)."""
    ds = td._single_shot_dataset(name, fam, cfg, SHOT, DATA_DIR, t0_start=T0)
    n = len(ds)
    assert n > 0, f"{name}: no windows for shot {SHOT}"
    xs, ms = [], []
    for i in range(n):
        item = ds[i]
        xs.append(item[0])
        if fam in ("video", "slowts"):
            ms.append(item[1])
    wins = torch.stack(xs, 0)
    masks = torch.stack(ms, 0) if ms else None
    assert torch.isfinite(wins).all(), f"{name}: non-finite window content"
    return wins, masks


def _finetune_codec(name: str, fam: str, codec, wins: torch.Tensor, masks, epochs: int,
                    bs: int, device: torch.device):
    """Warm-started recon-only fine-tune over ALL windows of the shot.

    Uses each family's CANONICAL reconstruction term (the codec's own masked-MAE helpers), with
    every other production objective (adversarial / entropy / consistency) off — same
    philosophy as rungs 1/2: recon-only isolates "can this codec represent this shot".
    Returns (codec [eval, frozen], first_epoch_mae, last_epoch_mae).
    """
    codec = codec.to(device).train()
    for p in codec.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(codec.parameters(), lr=CODEC_LR)
    gen = torch.Generator().manual_seed(0)
    n = wins.shape[0]
    first_ep = last_ep = None
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = wins[idx].to(device)
            mb = masks[idx].to(device) if masks is not None else None
            out = codec(xb)
            if fam == "video":
                # video recon lives in the codec's STANDARDIZED frame space -> target x_std;
                # masked so dead-camera (fully-NaN -> zero-filled) frames don't drag recon to 0.
                pixel = codec._masked_pixel_mae(out["recon"], out["x_std"], mb)
            elif fam == "slowts":
                # zero_is_missing / NaN gaps are excluded — never train toward the fill value.
                pixel = codec._masked_recon_mae(out["recon"], xb, mb)
            else:
                pixel = (out["recon"] - xb).abs().mean()
            opt.zero_grad(set_to_none=True)
            pixel.backward()
            opt.step()
            tot += float(pixel) * xb.shape[0]
        ep_mae = tot / n
        assert math.isfinite(ep_mae), f"{name}: non-finite epoch MAE at epoch {ep}"
        if first_ep is None:
            first_ep = ep_mae
        last_ep = ep_mae
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[e2e:{name}] codec epoch {ep:4d}  mae={ep_mae:.4f}", flush=True)
    codec.eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    return codec, first_ep, last_ep


def _encode_windows(codec, wins: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Encode all windows -> (n_frames, n_tok) flat FSQ codes (cpu long), chunked for memory."""
    outs = []
    for s in range(0, wins.shape[0], chunk):
        outs.append(td.encode_flat(codec, wins[s:s + chunk]))
    return torch.cat(outs, 0)


# --------------------------------------------------------------------------------------------- #
# denormalization back to PHYSICAL units (inverse of each family's input normalization)
# --------------------------------------------------------------------------------------------- #
def _spectro_prenorm_stats(name: str, cfg):
    """TRUE per-(frame, channel) window stats (mu, sd) of the pre-instance-norm log-power,
    from a second dataset pass with instance norm OFF. These are the DENORM REFERENCE: the
    decoded window is affinely renormalized to (0, 1) and then mapped to (mu, sd), which
    restores the exact physical level regardless of the shift-robust stat quantization the
    forward pipeline applied (affine invariance). The codes carry no absolute scale, so the
    GT frame's stats are the reference for GT and PRED alike."""
    import dataclasses

    import numpy as np
    pre = dataclasses.replace(cfg)
    pre.input_instance_norm = False
    ds = td._single_shot_dataset(name, "spectro", pre, SHOT, DATA_DIR, t0_start=T0)
    mus, sds = [], []
    for i in range(len(ds)):
        spec = ds[i][0]                                     # (C, F, T) pre-instance-norm
        mus.append(spec.mean(dim=(-2, -1), keepdim=True))
        sds.append(spec.std(dim=(-2, -1), keepdim=True) + 1e-5)
    return np.stack(mus, 0), np.stack(sds, 0)               # (n, C, 1, 1) each


def _make_denorm(name: str, fam: str, cfg, wins: torch.Tensor):
    """-> callable(np arr (F, ...)) mapping decoded output back to physical units.

    spectro : decode affinely renormalized to (0,1) per (frame, channel), mapped to the TRUE
              GT-window stats (level/scale are exogenous under instance norm — see
              _spectro_prenorm_stats), then per-freq-z⁻¹ (cfg dataset stats) → log10 power.
              (co2's raw-z happened pre-STFT: its "physical" space is the log-power of the
              raw-standardized signal — a constant per-channel offset.)
    video   : per-clip std⁻¹ (GT-clip stats, sd clamped at 1.0 like standardize_input) →
              raw camera counts.
    slow-TS : cfg per-channel dataset stats inverse; 10^x - 1 for log_standardize (Thomson)
              → H5 physical units. The padded radial-zone tail (no stats) is dropped.
    """
    import numpy as np
    if fam == "spectro":
        inst = (_spectro_prenorm_stats(name, cfg)
                if getattr(cfg, "input_instance_norm", False) else None)
        if getattr(cfg, "logpow_standardize", False) and cfg.logpow_freq_mean is not None:
            fm = np.asarray(cfg.logpow_freq_mean, dtype=np.float64)[None, :, :, None]
            fs = np.asarray(cfg.logpow_freq_std, dtype=np.float64).clip(
                min=float(getattr(cfg, "logpow_std_floor", 0.25)))[None, :, :, None]
        else:
            fm = fs = None

        def inv(arr):
            a = np.asarray(arr, dtype=np.float64)
            if inst is not None:
                mu, sd = inst
                # The codec input was per-(window, channel) standardized — absolute window
                # level/scale are EXOGENOUS (never encoded). Renormalize the decode to (0, 1)
                # per (frame, channel) and map to the TRUE window stats: exact level
                # restoration; without this the decoder's small per-window DC errors render
                # as blocky whole-window brightness steps in the stitched physical view
                # (diagnosed 2026-08-03: inversion round-trip exact, input stitch smooth).
                am = a.mean(axis=(-2, -1), keepdims=True)
                asd = a.std(axis=(-2, -1), keepdims=True) + 1e-6
                a = (a - am) / asd
                a = a * sd[: a.shape[0]] + mu[: a.shape[0]]
            if fs is not None:
                a = a * fs + fm
            return a.astype(np.float32)
        return inv
    if fam == "fastts":
        # filterscopes is not in eval_dynamics.EVAL_MODALITIES (never decoded/rendered);
        # identity keeps the denorm contract uniform. (A physical inverse would be
        # envelope x channel_std — add it when a fast-TS render panel exists.)
        return lambda arr: arr
    if fam == "video":
        mu = wins.mean(dim=(2, 3, 4)).numpy()[:, :, None, None, None]          # (n, C, 1,1,1)
        sd = wins.std(dim=(2, 3, 4)).clamp(min=1.0).numpy()[:, :, None, None, None]

        def inv(arr):
            a = np.asarray(arr, dtype=np.float64)
            return (a * sd[: a.shape[0]] + mu[: a.shape[0]]).astype(np.float32)
        return inv
    # slow-TS: global per-channel dataset stats live on the cfg (exactly invertible)
    ch = int(cfg.channels)
    m = (np.asarray(cfg.channel_mean, dtype=np.float64)[None, :, None]
         if cfg.channel_mean is not None else None)
    s = (np.asarray(cfg.channel_std, dtype=np.float64).clip(min=1e-3)[None, :, None]
         if cfg.channel_std is not None else None)
    logspace = getattr(cfg, "preprocess_method", None) == "log_standardize"

    def inv(arr):
        a = np.asarray(arr, dtype=np.float64)[:, :ch]       # drop the padded radial-zone tail
        if s is not None:
            a = a * s + m
        if logspace:
            a = np.power(10.0, a) - 1.0                     # inverse of log10(clip(x)+1)
        return a.astype(np.float32)
    return inv


def _load_codec_set(repo: Path):
    """-> list of (name, family, codec, cfg, source) in canonical FROZEN_MODALITIES order.

    EVERY modality (filterscopes included) resolves through the same candidate chain:
    CKPT_TMPLS newest-generation-first, then the frozen manifest — first existing checkpoint
    wins, and the print tags the generation dir that was used. A modality with no checkpoint
    anywhere is skipped loudly. The _load_codec loader freezes parameters — _finetune_codec
    re-enables them.
    """
    entries, skipped = [], []
    for spec in FROZEN_MODALITIES:
        name = spec.name
        fam = td.FROZEN_CODEC_CKPTS[name][0]
        cands = ([t.format(m=name) for t in CKPT_TMPLS]
                 + [td.FROZEN_CODEC_CKPTS[name][1]])
        rel = next((c for c in cands if (repo / c).exists()), None)
        if rel is None:
            skipped.append((name, f"no checkpoint found (tried {cands})"))
            continue
        codec, cfg = td._load_codec(fam, repo / rel)
        print(f"[e2e:{name}] codec [{Path(rel).parent.name}]: {rel}", flush=True)
        entries.append((name, fam, codec, cfg, rel))
    for name, why in skipped:
        print(f"[e2e] NOTE skipping {name}: {why}", flush=True)
    return entries, [n for n, _ in skipped]


# --------------------------------------------------------------------------------------------- #
# the test
# --------------------------------------------------------------------------------------------- #
def test_ignite_model_overfits_one_shot() -> None:
    """Train the full IGNITE model (all-family codecs + MaskGIT ST-backbone) on shot 200729;
    the rollout must reproduce the shot (see module docstring for the pass criteria)."""
    device = _device()
    torch.manual_seed(0)
    repo = Path.cwd()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- stage 1: codecs (ALL families) — warm start + recon-only fine-tune ----------------- #
    entries, skipped = _load_codec_set(repo)
    assert any(f == "spectro" for _, f, _c, _cf, _s in entries), (
        f"no spectro checkpoint found anywhere (templates {CKPT_TMPLS} + manifest)"
    )
    codecs, frame_codes, codec_stats, denorm = {}, {}, {}, {}
    for name, fam, codec, cfg, src in entries:
        wins, masks = _windows_and_masks(name, fam, cfg)
        print(f"[e2e:{name}] {wins.shape[0]} windows {tuple(wins.shape[1:])}", flush=True)
        codec, first_ep, last_ep = _finetune_codec(name, fam, codec, wins, masks,
                                                   CODEC_EPOCHS, CODEC_BS, device)
        # warm-started single-shot overfit must not diverge (it may already sit at its floor)
        assert last_ep <= first_ep * 1.05, (
            f"{name}: fine-tune MAE diverged ({first_ep:.4f} -> {last_ep:.4f})"
        )
        frame_codes[name] = _encode_windows(codec, wins)
        denorm[name] = _make_denorm(name, fam, cfg, wins)   # physical-units inverse (GT stats)
        codecs[name] = (codec, cfg, fam)
        codec_stats[name] = {
            "source": src, "family": fam,
            "first_epoch_mae": first_ep, "last_epoch_mae": last_ep,
            "n_tok": int(frame_codes[name].shape[1]),
            "vocab": int(codec.quantizer.fsq.codebook_size),
        }
        del wins, masks

    # ---- stage 2: frame codes + actuators ---------------------------------------------------- #
    n_frames = {n: c.shape[0] for n, c in frame_codes.items()}
    for n in [n for n, c in n_frames.items() if c < K0 + 8]:
        print(f"[e2e] NOTE dropping {n}: only {n_frames[n]} frames on shot {SHOT}", flush=True)
        for d in (frame_codes, codecs, codec_stats, n_frames):
            d.pop(n)
    fams = {codecs[n][2] for n in codecs}
    # the point of this rung is the MULTI-MODAL state — spectro alone is rung 2's job.
    assert "spectro" in fams, "no spectro modality survived"
    assert "video" in fams, f"no video modality available on shot {SHOT} — full-state test needs it"
    assert "slowts" in fams, f"no slow-TS modality available on shot {SHOT} — full-state test needs it"
    nmin = min(n_frames.values())
    frame_codes = {n: c[:nmin] for n, c in frame_codes.items()}
    act = td.actuator_frames(SHOT, nmin, DATA_DIR, t0_start=T0)          # (nmin, 70)

    # ---- stage 3: backbone (layout DERIVED from the actual codecs) --------------------------- #
    order = [m.name for m in FROZEN_MODALITIES if m.name in codecs]
    specs = tuple(
        ModalitySpec(n, codecs[n][2], codec_stats[n]["n_tok"], codec_stats[n]["vocab"])
        for n in order
    )
    k0 = min(K0, max(2, nmin // 3))
    assert nmin >= k0 + 8, f"shot too short: nmin={nmin} frames for k0={k0}"
    n_pred = min(NPRED, nmin - k0)
    bcfg = DynamicsConfig(modalities=specs, d_model=DM, depth=DEPTH,
                          n_heads=8 if DM % 8 == 0 else 4,
                          k0_seed=k0, n_predict=n_pred, grad_checkpointing=GRAD_CKPT)
    print(f"[e2e] backbone d{DM}/depth{DEPTH}: {len(specs)} modalities, "
          f"{bcfg.tokens_per_frame} tokens/frame, nmin={nmin} k0={k0} n_pred={n_pred}", flush=True)

    model = MaskGITDynamics(bcfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    mask_gen = torch.Generator(device=device).manual_seed(0)   # MaskGIT rands live on `device`
    crop_gen = torch.Generator().manual_seed(1)
    win = min(TRAIN_F, bcfg.max_frames, nmin)
    losses = []
    for step in range(STEPS):
        starts = torch.randint(0, nmin - win + 1, (BS,), generator=crop_gen).tolist()
        cb = {n: torch.stack([frame_codes[n][s:s + win] for s in starts]).to(device)
              for n in order}
        ab = torch.stack([act[s:s + win] for s in starts]).to(device)
        opt.zero_grad(set_to_none=True)
        loss = model.training_loss(cb, ab, generator=mask_gen, ss_frac=0.0)
        loss.backward()
        opt.step()
        v = float(loss)
        assert math.isfinite(v), f"non-finite MaskGIT loss at step {step}"
        losses.append(v)
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            print(f"[e2e] backbone step {step:5d}  masked_ce={v:.4f}", flush=True)
    k = min(20, len(losses))
    ce_start, ce_end = sum(losses[:k]) / k, sum(losses[-k:]) / k

    # loss curve: every step's masked CE -> json + rendered PNG (the scaling-study instrument)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = range(1, len(losses) + 1)
    ax.plot(xs, losses, color="C0", alpha=0.3, lw=0.7, label="masked CE (per step)")
    ema, sm = None, []
    for v in losses:
        ema = v if ema is None else 0.98 * ema + 0.02 * v
        sm.append(ema)
    ax.plot(xs, sm, color="C0", lw=1.6, label="EMA")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("masked-token CE")
    ax.set_title(f"backbone d{DM}/depth{DEPTH} — masked CE overfit curve (shot {SHOT})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_curve.png", dpi=110)
    plt.close(fig)

    # ---- rollout + decode + figure ----------------------------------------------------------- #
    model.eval()
    F_ro = min(bcfg.max_frames, nmin)
    seed_codes = {n: frame_codes[n][:k0].unsqueeze(0).to(device) for n in order}
    with torch.no_grad():
        traj = model.rollout(seed_codes, act[:F_ro].unsqueeze(0).to(device),
                             n_predict=F_ro - k0, temperature=TEMP, generator=mask_gen)
    gt_codes = {n: frame_codes[n][:F_ro] for n in order}
    pred_codes = {n: traj[n][0].cpu() for n in order}
    tok_acc = {n: float((pred_codes[n][k0:] == gt_codes[n][k0:]).float().mean()) for n in order}
    # PERSISTENCE BASELINE (identical definition to eval_dynamics): freeze the last seed
    # frame across the predicted region. Discrete codes are highly persistent at 50 ms, so
    # raw accuracy/nRMSE are not interpretable without it — report the SKILL.
    pers_acc = {n: float((gt_codes[n][k0:] == gt_codes[n][k0 - 1:k0]).float().mean())
                for n in order}

    torch.save({"gt_codes": gt_codes, "pred_codes": pred_codes, "k0": k0, "F": F_ro,
                "temperature": TEMP},
               OUT_DIR / f"rollout_codes_{SHOT}.pt")           # tiny; offline re-rendering
    # persist the SHOT-FINE-TUNED codecs: without them an offline re-render decodes through
    # the pre-fine-tune snapshot and understates quality (video structure washes out).
    cdir = OUT_DIR / "codecs_finetuned"
    cdir.mkdir(parents=True, exist_ok=True)
    for _n, (_c, _cfg, _f) in codecs.items():
        torch.save({"codec": _c.state_dict(), "cfg": _cfg, "family": _f},
                   cdir / f"{_n}.pt")
    from tokamak_foundation_model.ignite import eval_dynamics as ed
    decoded = ed.decode_all(codecs, gt_codes, pred_codes, k0, F_ro, device, denorm=denorm)
    png, pdf = ed.render_figure(decoded, SHOT, STEPS, TEMP, k0, F_ro, OUT_DIR, t_origin=T0)

    metrics = {
        "shot": SHOT, "steps": STEPS, "codec_epochs": CODEC_EPOCHS, "codec_lr": CODEC_LR,
        "d_model": DM, "depth": DEPTH, "batch": BS, "train_window": win,
        "k0": k0, "rollout_frames": F_ro, "temperature": TEMP, "t0_start": T0,
        "ckpt_templates": list(CKPT_TMPLS), "skipped_modalities": skipped,
        "tokens_per_frame": bcfg.tokens_per_frame,
        "codecs": codec_stats,
        "maskgit_ce_start": ce_start, "maskgit_ce_end": ce_end,
        "loss_curve": losses,
        "token_accuracy_pred_region": tok_acc,
        "decoded_space": "physical",
        "decoded_nrmse": {n: decoded[n]["nrmse"] for n in decoded},
        "decoded_nrmse_persistence": {n: decoded[n]["nrmse_persistence"] for n in decoded},
        "decoded_nrmse_skill": {n: decoded[n]["nrmse_skill"] for n in decoded},
        "token_accuracy_persistence": pers_acc,
        "token_accuracy_skill": {n: tok_acc[n] - pers_acc[n] for n in order},
        "figure": str(png),
    }
    with open(OUT_DIR / f"e2e_overfit_{SHOT}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save({"model": model.state_dict(), "d_model": DM, "depth": DEPTH,
                "n_heads": bcfg.n_heads, "k0": k0, "n_predict": n_pred, "step": STEPS,
                "modalities": [(s.name, s.family, s.n_tok, s.codebook_size) for s in specs]},
               OUT_DIR / "dynamics_e2e_last.pt")
    print(f"[e2e] RESULT masked_ce {ce_start:.3f} -> {ce_end:.3f}  |  token acc (skill vs "
          f"persistence) "
          + " ".join(f"{n}={tok_acc[n]:.3f}({tok_acc[n] - pers_acc[n]:+.3f})" for n in order)
          + f"\n[e2e] figure: {png}\n[e2e] metrics: {OUT_DIR / f'e2e_overfit_{SHOT}.json'}",
          flush=True)

    # ---- gates (PROVISIONAL collapse-guards; tighten to measured floors after the first run) -- #
    # substantial memorization happened (masked CE starts near mean ln(vocab) ~ 7-8)
    assert ce_end < 0.6 * ce_start, (
        f"backbone barely moved (masked CE {ce_start:.3f} -> {ce_end:.3f}) — "
        f"pipeline or optimization bug"
    )
    # every committed rollout code is in its modality's vocab (MaskGIT commit contract)
    for n in order:
        lo, hi = int(pred_codes[n].min()), int(pred_codes[n].max())
        assert lo >= 0 and hi < codec_stats[n]["vocab"], (
            f"{n}: rollout committed out-of-vocab codes [{lo}, {hi}]"
        )
    # slow-TS is smooth + memorizable: token accuracy must sit FAR above chance (1/1000).
    # Spectro token accuracy is NOT gated — codes carry STFT-phase realization bits
    # (mode-audit 2026-07-13), so judge spectro from the figure, not exact-token match.
    slow_acc = [tok_acc[n] for n in order if codecs[n][2] == "slowts"]
    assert sum(slow_acc) / len(slow_acc) > 0.05, (
        f"slow-TS rollout token accuracy {slow_acc} ~ chance — dynamics learned nothing"
    )
    # decode path sanity: every rendered modality decoded to finite content
    for n, d in decoded.items():
        assert math.isfinite(d["nrmse"]), f"{n}: non-finite decoded nRMSE"
    assert png.exists(), "comparison figure missing — it is the deliverable of this gate"
