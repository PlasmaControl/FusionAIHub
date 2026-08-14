"""Real-shot FSQ-AE overfit gate (rung 1 of the overfit ladder) — shot 200729.

The production codec objective is NOT a reconstruction objective (pixel anchor 0.05 vs
consistency / entropy / feature-matching / adversarial at ~1.0), so "the codec can't
overfit" under the production loss says nothing about the FSQ-AE mechanics. This gate
isolates the mechanics:

    encoder -> FSQ -> decoder, trained with PURE pixel MAE on ONE fixed real window,
    must (nearly) perfectly reconstruct that window.

* PASS => the AE machinery (patchify, FSQ straight-through gradient, decode) is sound;
  a collapsed/blurry production run indicts the objective weighting / GAN dynamics,
  NOT the FSQ-AE itself.
* FAIL => a genuine mechanics bug in the failing modality's path — dig here before
  touching any loss weight.

CALIBRATION (ece, shot 200729, most-active window (40,512,96), 1500 steps, 2026-07-31):
all arms drop nmae 1.17 -> ~0.40 in ~100 steps, PLATEAU there for several hundred steps
(smooth structure fits fast; noise-like turbulent texture memorizes slowly), then ESCAPE:
  * FSQ-BYPASS (decoder fed PRE-FSQ feats), lr 1e-3:  nmae 0.047 / corr 0.988 — near-ideal.
  * FSQ, lr 3e-4:  nmae 0.172 / corr 0.941 at step 1500, still descending steadily.
  * FSQ, lr 1e-3:  STUCK at nmae 0.374 / corr 0.857, codes churning (flip 0.1-0.4).
So the FSQ-AE mechanics DO support near-ideal single-window overfit; the quantizer slows
the escape (STE noise) and makes it lr-SENSITIVE — lr 1e-3 (the production codec default!)
never settles, lr 3e-4 does. Hence LR=3e-4 here. A mechanics bug (collapse-to-constant,
scale saturation, dead STE) lands at corr ~0 / nmae ~1 — far outside the thresholds.

REAL-DATA test: needs ``{SHOT}_processed.h5`` under the production data dir (or
``IGNITE_DATA_DIR``). Skipped cleanly when the data is missing, so the synthetic-only
suite is unaffected. The window fed to the codec goes through the EXACT production
pipeline (``CodecPairDataset`` -> ``_load_signal_raw`` -> ``shift_pair_windows`` log-power
STFT, incl. co2 raw standardization), and the codec is the EXACT production architecture
(``SpectroCodecConfig`` defaults; d_model=256, enc/dec depth 6, FSQ [8,5,5,5]).

Runtime: a few minutes per modality on CPU (GPU used automatically when available).

RUNG 2 — whole-shot overfit + FULL-spectrogram reconstruction (GPU-scale, opt-in via
IGNITE_FULLSHOT=1; launched by scripts/slurm_frontier/ignite_fsq_overfit_test.sh): train
the codec recon-only on ALL windows of the shot, then encode->decode every window in
order and stitch the complete shot spectrogram. Writes a GT | recon | diff comparison
figure + metrics json per modality to IGNITE_FULLSHOT_OUT. This is the "can the FSQ-AE
represent the whole shot, not just memorize one window" gate — capacity now matters
(≈200 windows share one codec), unlike rung 1.

LOCKED LINEUP (end of the 2026-07-31 lever-arm campaign — see _LOCKED /
_PERFREQ_LOGPOW; measurements in jobs 5129812-5131573 + memory): the rung-2 BASELINE
now guards the user-locked production-candidate configs — bes 192tok/fsq6,
co2 192tok/fsq6 + raw-z + per-freq log-z, ece 768tok(8x8)/1k + per-freq log-z,
mhr 192tok/1k + per-freq log-z. Retired arms and their verdicts: freqgrad = dead
(no visible/metric effect); fsq6 = useful for bes/co2; tok768 = the ece pick (figures;
capacity levers were metric-flat); chfact (channel-factorized) = dead; per-freq z =
adopted for co2/ece/mhr, REJECTED for bes (whitened bes is speckle). The lever-arm
mechanism (_ARMS + sanity-only asserts + `corr_smooth`) is kept for future levers
(e.g. conv stem). METRIC CAVEAT for the z modalities: nmae/corr live in the whitened
target space and are NOT comparable to raw-space numbers.

Env knobs:
    IGNITE_DATA_DIR          data dir with {shot}_processed.h5 (default: production dir)
    IGNITE_OVERFIT_SHOT      shot number                        (default: 200729)
    IGNITE_OVERFIT_STEPS     rung-1 Adam steps                  (default: 1500)
    IGNITE_OVERFIT_DECODER   "linear" | "conv"                  (default: linear)
    IGNITE_FULLSHOT          "1" enables the whole-shot rung-2 test (default: off)
    IGNITE_FULLSHOT_EPOCHS   rung-2 epochs over the shot's windows (default: 150)
    IGNITE_FULLSHOT_BS       rung-2 batch size                  (default: 16)
    IGNITE_FULLSHOT_OUT      figure/metrics output dir (default: eval_runs/fsq_overfit_200729)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
import torch

from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite.codec import SpectroCodec
from tokamak_foundation_model.ignite.config import SpectroCodecConfig
from tokamak_foundation_model.ignite.losses import (
    freq_gradient_loss,
    multiscale_recon_loss,
)

SHOT = os.environ.get("IGNITE_OVERFIT_SHOT", "200729")
DATA_DIR = Path(os.environ.get("IGNITE_DATA_DIR", tc.DEFAULT_DATA_DIR))
STEPS = int(os.environ.get("IGNITE_OVERFIT_STEPS", "1500"))
DECODER = os.environ.get("IGNITE_OVERFIT_DECODER", "linear")
LR = 3e-4

# The 4 production spectro codecs ("magnetics" == mhr).
MODALITIES = ("bes", "co2", "ece", "mhr")

N_SCAN = 24          # windows scanned across the shot to pick the most ACTIVE one
LOG_EVERY = 50       # loss-trajectory print cadence (visible with pytest -s)

FULLSHOT = os.environ.get("IGNITE_FULLSHOT") == "1"
# RENDER mode (promotion review): reconstruct the shot through a TRAINED production
# checkpoint instead of training a gate codec. Template gets .format(m=modality), e.g.
# "eval_runs/ignite_codec_{m}_v3/codec_best.pt". The checkpoint's own embedded cfg
# drives BOTH the model and the window preprocessing (exact production pipeline).
# Asserts are sanity-only; figures/json (arm tag "prod<step>") are the deliverable —
# point OUT_DIR/IGNITE_FULLSHOT_OUT at a render directory, NOT the gate baselines.
RENDER_CKPT_TMPL = os.environ.get("IGNITE_RENDER_CKPT_TMPL")
FULLSHOT_EPOCHS = int(os.environ.get("IGNITE_FULLSHOT_EPOCHS", "150"))
FULLSHOT_BS = int(os.environ.get("IGNITE_FULLSHOT_BS", "16"))
FULLSHOT_OUT = Path(os.environ.get("IGNITE_FULLSHOT_OUT", "eval_runs/fsq_overfit_200729"))

_H5 = DATA_DIR / f"{SHOT}_processed.h5"

pytestmark = pytest.mark.skipif(
    not _H5.exists(), reason=f"real-shot data not available: {_H5}"
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Modalities whose LOCKED config includes per-(channel,freq) LOG-POWER z-standardization
# (gate campaign 2026-07-31): co2 (rescued: codes 1 -> 31, chirps appear), ece (coherent
# structure appears; adopted), mhr (best recon of the campaign: GT-matching texture,
# z-space per-window corr med 0.58; adopted). bes REJECTED (post-whitening content is
# speckle, per-window corr med 0.22). Stats are computed from THIS shot's windows in the
# space the codec actually sees (for co2: ON TOP of the raw z-score), then applied inside
# data.log_power_stft via cfg.logpow_*. NOTE: production's --logpow_stats_path REPLACES
# raw-std; here we compose on top of it — self-consistent by construction since stats
# and application share the same cfg/pipeline (convention to be settled at promotion).
_PERFREQ_LOGPOW = frozenset({"co2", "ece", "mhr"})
_STATS_STRIDE = 2

# LOCKED per-modality config (user, 2026-07-31 — the outcome of the lever-arm campaign;
# arms measured and retired: freqgrad dead, capacity/geometry levers flat for ece except
# tok768 by figure, channel factorization dead, per-freq z adopted for co2/ece/mhr).
# The gate's BASELINE now guards THIS lineup, not the old production defaults.
_LOCKED: dict = {
    "bes": {"fsq_levels": [8, 8, 8, 5, 5, 5]},                 # fsq6, no input z
    "co2": {"fsq_levels": [8, 8, 8, 5, 5, 5]},                 # fsq6 + raw-z + per-freq z
    "ece": {"patch_f": 8, "patch_t": 8},                       # 768 tokens + per-freq z
    "mhr": {},                                                 # 192 tok / 1k + per-freq z
}


# Dataset-level stats mode is the CANONICAL mode since 2026-08-01 (user verdict: the
# instance-norm v3 figures beat shot-local for BOTH ece and mhr, job 5135695): per-freq
# stats load from the production files in this dir and instance norm is enabled on top
# (see _modality_cfg). Set IGNITE_LOGPOW_STATS_DIR="" to fall back to shot-local stats
# (the 2026-07-31 campaign's original calibration mode).
LOGPOW_STATS_DIR = os.environ.get(
    "IGNITE_LOGPOW_STATS_DIR",
    "/lustre/orion/fus187/proj-shared/foundation_model_meta",
)


def _apply_shot_logpow_stats(cfg: SpectroCodecConfig, modality: str,
                             force: bool = False) -> None:
    """Enable per-(C,F) log-power z-standardization on ``cfg``.

    Applied by default only to _PERFREQ_LOGPOW; ``force=True`` applies it to any
    modality. No-op if already enabled on this cfg. Stats source:
      * IGNITE_LOGPOW_STATS_DIR set -> load the PRODUCTION dataset-level file
        ``codec_{modality}_perfreq_stats.pt`` from that dir (the gate then tests exactly
        what production trains with). NOTE: dataset-level stds are ~5-6x the shot-local
        ones (across-shot regime variance dominates), so the whitening is weaker and
        metrics live in a DIFFERENT target space than the shot-local floors.
      * unset -> compute shot-local stats from this shot's windows (the mode the
        2026-07-31 campaign was calibrated in).
    """
    if getattr(cfg, "logpow_standardize", False):
        return
    if not force and modality not in _PERFREQ_LOGPOW:
        return
    if LOGPOW_STATS_DIR:
        p = Path(LOGPOW_STATS_DIR) / f"codec_{modality}_perfreq_stats.pt"
        st = torch.load(p, map_location="cpu", weights_only=False)
        if cfg.input_standardize:
            assert str(st.get("space", "")).endswith("post_raw_std"), (
                f"{p}: stats space {st.get('space')!r} does not compose with raw-std "
                f"(regenerate with train_codec.compute_logpow_stats)"
            )
        cfg.logpow_standardize = True
        cfg.logpow_freq_mean = torch.as_tensor(st["mean"], dtype=torch.float32).tolist()
        cfg.logpow_freq_std = torch.as_tensor(st["std"], dtype=torch.float32).tolist()
        print(f"[stats:{modality}] per-freq log-power z ON from DATASET file {p} "
              f"(space={st.get('space')}, n_windows={st.get('n_windows')})", flush=True)
        return
    ds = tc.CodecPairDataset(modality, [SHOT], cfg, data_dir=DATA_DIR, seed=0)
    s1 = torch.zeros(cfg.channels, cfg.freq_bins, dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    frames = 0
    n_used = 0
    for i in range(0, len(ds), _STATS_STRIDE):
        a, _ = ds[i]                                   # (C, F, T), logpow-z still OFF here
        s1 += a.to(torch.float64).sum(dim=-1)
        s2 += (a.to(torch.float64) ** 2).sum(dim=-1)
        frames += a.shape[-1]
        n_used += 1
    mean = s1 / frames
    std = (s2 / frames - mean ** 2).clamp_min(0.0).sqrt()
    cfg.logpow_standardize = True
    cfg.logpow_freq_mean = mean.float().tolist()
    cfg.logpow_freq_std = std.float().tolist()
    print(f"[stats:{modality}] per-freq log-power z ON from {n_used} windows "
          f"(mean~{float(mean.mean()):.2f}, std med~{float(std.median()):.3f}, "
          f"floor {cfg.logpow_std_floor})", flush=True)


def _modality_cfg(modality: str, arm: str = "baseline") -> SpectroCodecConfig:
    """Production codec config for ``modality`` with a RECON-ONLY objective + arm overrides.

    Mirrors ``train_codec.main()``'s cfg construction (channel sizing + co2 raw input
    standardization; no-op for ece/bes/mhr), plus the per-freq log-power z for the
    thin modalities in _PERFREQ_LOGPOW, plus the ``_ARMS[arm]`` field overrides
    (baseline = no-op). Training uses plain recon terms directly (pixel MAE, plus
    freq-grad/multiscale when the arm sets their weights), not ``generator_losses``,
    so no discriminator is ever built.
    """
    cfg = SpectroCodecConfig()
    cfg.channels = tc.modality_channels(modality)
    cfg.decoder = DECODER
    tc.apply_spectro_standardization(cfg, modality, log_fn=print)
    for k, v in _LOCKED[modality].items():
        setattr(cfg, k, v)
    _apply_shot_logpow_stats(cfg, modality)
    cfg.pixel_anchor_weight = 1.0
    cfg.consistency_weight = 0.0
    cfg.entropy_weight = 0.0
    cfg.fm_weight = 0.0
    cfg.adversarial_weight = 0.0
    cfg.adaptive_adv_weight = False
    cfg.multiscale_recon_weight = 0.0
    cfg.freq_grad_weight = 0.0
    overrides = dict(_ARMS[arm])
    want_z = overrides.pop("_perfreq_z", False)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    if want_z:
        # lever arms may force per-freq z for a modality outside _PERFREQ_LOGPOW.
        _apply_shot_logpow_stats(cfg, modality, force=True)
    if LOGPOW_STATS_DIR and cfg.logpow_standardize:
        # PRODUCTION-CANDIDATE config (Option A, 2026-08-01): per-window instance z
        # (data.log_power_stft applies it LAST) on top of the dataset per-freq z. Kills
        # the dataset-vs-shot mean offset (~2.7σ) that kept ece at 1 code and mhr below
        # canonical in the v2 validation (job 5132199). Dataset-stats mode only — the
        # canonical shot-local mode stays as calibrated.
        cfg.input_instance_norm = True
        # shift-robust quantized stats (2026-08-03): matches the v4 production recipe;
        # IGNITE_INSTNORM_Q="" or "0" restores plain instance norm (the v3 mode).
        cfg.instance_norm_quantize = float(os.environ.get("IGNITE_INSTNORM_Q") or 0.5)
        print(f"[stats:{modality}] input_instance_norm ON (dataset-stats mode, "
              f"quantize={cfg.instance_norm_quantize})", flush=True)
    return cfg


def _most_active_window(modality: str, cfg: SpectroCodecConfig) -> torch.Tensor:
    """(1, C, F, T) log-power window with the LARGEST std among N_SCAN spread over the shot.

    Deterministic (seeded dataset, fixed scan grid). Picking the most active window matters:
    overfitting a floored/quiet plate (common for co2) would be a trivial, meaningless pass.
    """
    # Selection ranks PRE-instance-norm activity: under instance norm every window's
    # std is exactly 1.0, so ranking the final tensors is degenerate (v3 lesson: it
    # picked window 0 = ramp-up). The winning window is then loaded through the FULL
    # cfg pipeline.
    import dataclasses
    sel_cfg = dataclasses.replace(cfg)
    sel_cfg.input_instance_norm = False
    ds_sel = tc.CodecPairDataset(modality, [SHOT], sel_cfg, data_dir=DATA_DIR, seed=0)
    n = len(ds_sel)
    assert n > 0, f"{modality}: no windows for shot {SHOT}"
    idxs = sorted({round(i * (n - 1) / max(1, N_SCAN - 1)) for i in range(min(N_SCAN, n))})
    best, best_std, best_idx = None, -1.0, -1
    for i in idxs:
        spec_a, _ = ds_sel[i]
        s = float(spec_a.std())
        if s > best_std:
            best, best_std, best_idx = spec_a, s, i
    assert best is not None
    if getattr(cfg, "input_instance_norm", False):
        ds = tc.CodecPairDataset(modality, [SHOT], cfg, data_dir=DATA_DIR, seed=0)
        best = ds[best_idx][0]
    assert torch.isfinite(best).all()
    print(f"[overfit:{modality}] window idx={best_idx}/{n} sel_std={best_std:.3f} "
          f"shape={tuple(best.shape)}")
    return best[None]


def _overfit(cfg: SpectroCodecConfig, x: torch.Tensor, steps: int):
    """Train the codec on the single window ``x`` with pure pixel MAE; return diagnostics."""
    torch.manual_seed(0)
    device = _device()
    x = x.to(device)
    codec = SpectroCodec(cfg).to(device)
    opt = torch.optim.Adam(codec.parameters(), lr=LR)

    first_mae = None
    for step in range(steps):
        out = codec(x)
        loss = (out["recon"] - x).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        mae = float(loss)
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        if first_mae is None:
            first_mae = mae
        if step % LOG_EVERY == 0 or step == steps - 1:
            print(f"[overfit] step {step:4d}  mae={mae:.4f}  nmae={mae / float(x.std()):.4f}")

    codec.eval()
    with torch.no_grad():
        out = codec(x)
        final_mae = float((out["recon"] - x).abs().mean())
        r, t = out["recon"].flatten(), x.flatten()
        corr = float(torch.corrcoef(torch.stack([r, t]))[0, 1])
        # per-token flat code ids -> codebook usage of the fitted window
        flat = codec.quantizer.fsq(out["feats"])[1]
        n_codes = int(flat.unique().numel())
    return {
        "first_mae": first_mae,
        "final_mae": final_mae,
        "nmae": final_mae / float(x.std()),
        "corr": corr,
        "n_codes": n_codes,
        "recon": out["recon"].cpu(),
    }


@pytest.mark.parametrize("modality", MODALITIES)
def test_fsq_ae_overfits_one_real_window(modality: str) -> None:
    """Recon-only FSQ-AE must fit ONE active real window of shot 200729 to the measured
    sound-mechanics operating point (see module docstring CALIBRATION).

    Thresholds calibrated at STEPS=1500 / lr=3e-4 on the hardest (most active,
    broadband) ece window: a sound run reaches nmae ~0.17 / corr ~0.94 and is still
    improving; a mechanics bug (broken patchify order, dead STE, scale saturation,
    collapse-to-constant) lands at corr ~0 / nmae ~1. The asserted 0.35 / 0.80 sit
    between the two with margin for modality-to-modality variation.
    """
    cfg = _modality_cfg(modality)
    x = _most_active_window(modality, cfg)
    res = _overfit(cfg, x, STEPS)

    print(f"[overfit:{modality}] first_mae={res['first_mae']:.4f} "
          f"final_mae={res['final_mae']:.4f} nmae={res['nmae']:.4f} "
          f"corr={res['corr']:.4f} codes_used={res['n_codes']}")

    # The window must actually have structure (guards a silently-empty modality).
    assert float(x.std()) > 0.05, f"{modality}: selected window is ~flat (std={float(x.std()):.4f})"
    # Substantial optimization happened (not a frozen/dead graph).
    assert res["final_mae"] < 0.5 * res["first_mae"], (
        f"{modality}: MAE barely moved ({res['first_mae']:.4f} -> {res['final_mae']:.4f}) "
        f"— gradient path through encoder/FSQ/decoder is suspect"
    )
    # Sound-mechanics operating point. Raw-space thresholds calibrated (see module
    # docstring); whitened-target (per-freq z) thresholds calibrated on the consolidated
    # locked lineup (job 5131694): co2 0.125/0.971, ece 0.348/0.812, mhr 0.010/0.9996 —
    # floors sit below the worst measured case (ece) with margin.
    # z thresholds calibrated on the v3-canonical confirmation (job 5136281): co2
    # 0.055/0.989 (167 codes), ece 0.430/0.675 (419 codes), mhr 0.011/0.9998 (79
    # codes) — floors sit just below the worst case (ece) with margin.
    if getattr(cfg, "logpow_standardize", False):
        max_nmae, min_corr = 0.48, 0.62
    else:
        max_nmae, min_corr = 0.35, 0.80
    assert res["nmae"] < max_nmae, (
        f"{modality}: normalized MAE {res['nmae']:.3f} too high — the FSQ-AE could not "
        f"overfit one window under a PURE recon objective (mechanics bug)"
    )
    assert res["corr"] > min_corr, (
        f"{modality}: GT<->recon correlation {res['corr']:.3f} too low for a single-window "
        f"overfit (mechanics bug)"
    )


# ------------------------------------------------------------------------------------- #
# RUNG 2 — whole-shot overfit + FULL-spectrogram reconstruction (GPU-scale, opt-in)
# ------------------------------------------------------------------------------------- #
# Lever arms. Keys are cfg-field overrides applied AFTER the locked config + recon-only
# overrides in _modality_cfg (special marker `_perfreq_z` forces per-freq z for a
# modality outside _PERFREQ_LOGPOW). The 2026-07-31 campaign's arms (freqgrad, fsq6,
# tok768[_fsq6], chfact768[_fsq6], nbz*) are RETIRED: their verdicts are folded into
# _LOCKED/_PERFREQ_LOGPOW and their measurements live in the job logs
# (5130478/5130776/5130955/5131573) + memory. Add future levers (e.g. conv stem) here.
_ARMS: dict = {
    "baseline": {},
}
_ARM_CASES: list = []
_FULLSHOT_CASES = [(m, "baseline") for m in MODALITIES] + _ARM_CASES


def _all_windows(modality: str, cfg: SpectroCodecConfig) -> torch.Tensor:
    """(n, C, F, T) — ALL windows of the shot in time order (window i starts at
    ``warmup + i * CHUNK_S``), through the exact production preprocessing.

    A degenerate window (padded tail / flat / NaN) is replaced by a nearby valid one by
    the dataset's redraw — rare on an active shot and identical to what production
    training sees, so accepted here.
    """
    ds = tc.CodecPairDataset(modality, [SHOT], cfg, data_dir=DATA_DIR, seed=0)
    n = len(ds)
    assert n > 0, f"{modality}: no windows for shot {SHOT}"
    wins = torch.stack([ds[i][0] for i in range(n)], dim=0)
    assert torch.isfinite(wins).all()
    print(f"[fullshot:{modality}] {n} windows {tuple(wins.shape[1:])} "
          f"std={float(wins.std()):.3f}", flush=True)
    return wins


def _train_full_shot(cfg: SpectroCodecConfig, wins: torch.Tensor,
                     epochs: int, bs: int, device: torch.device):
    """Recon-only training over ALL windows; returns (codec, first_epoch_mae, last_epoch_mae)."""
    torch.manual_seed(0)
    codec = SpectroCodec(cfg).to(device)
    opt = torch.optim.Adam(codec.parameters(), lr=LR)
    n = wins.shape[0]
    gen = torch.Generator().manual_seed(0)
    first_ep = last_ep = None
    fg_w = float(getattr(cfg, "freq_grad_weight", 0.0))
    ms_w = float(getattr(cfg, "multiscale_recon_weight", 0.0))
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for s in range(0, n, bs):
            xb = wins[perm[s:s + bs]].to(device)
            recon = codec(xb)["recon"]
            pixel = (recon - xb).abs().mean()
            loss = pixel
            if fg_w > 0:
                loss = loss + fg_w * freq_gradient_loss(recon, xb)
            if ms_w > 0:
                loss = loss + ms_w * multiscale_recon_loss(recon, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(pixel) * xb.shape[0]   # track the PIXEL anchor across arms
        ep_mae = tot / n
        assert math.isfinite(ep_mae), f"non-finite epoch MAE at epoch {ep}"
        if first_ep is None:
            first_ep = ep_mae
        last_ep = ep_mae
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[fullshot] epoch {ep:4d}  mae={ep_mae:.4f}  "
                  f"nmae={ep_mae / float(wins.std()):.4f}", flush=True)
    return codec, first_ep, last_ep


def _reconstruct_windows(codec: SpectroCodec, wins: torch.Tensor,
                         device: torch.device, bs: int = 32) -> torch.Tensor:
    """Encode -> FSQ -> decode every window; (n, C, F, T) reconstruction on CPU."""
    codec.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, wins.shape[0], bs):
            outs.append(codec(wins[s:s + bs].to(device))["recon"].cpu())
    return torch.cat(outs, dim=0)


def _chunked_corr(a: torch.Tensor, b: torch.Tensor, chunk: int = 8) -> float:
    """Pearson corr over ALL elements, accumulated in float64 window-chunks (RAM-safe:
    bes is ~0.7G elements — never materialize a flattened copy of both arrays)."""
    n = 0
    sa = sb = saa = sbb = sab = 0.0
    for i in range(0, a.shape[0], chunk):
        x = a[i:i + chunk].to(torch.float64)
        y = b[i:i + chunk].to(torch.float64)
        n += x.numel()
        sa += float(x.sum()); sb += float(y.sum())
        saa += float((x * x).sum()); sbb += float((y * y).sum()); sab += float((x * y).sum())
    cov = sab / n - (sa / n) * (sb / n)
    va = saa / n - (sa / n) ** 2
    vb = sbb / n - (sb / n) ** 2
    return cov / max((va * vb) ** 0.5, 1e-12)


def _chunked_corr_smooth(a: torch.Tensor, b: torch.Tensor, k: int = 4,
                         chunk: int = 8) -> float:
    """Structure-sensitive corr: Pearson corr after k x k avg-pooling each window.

    Speckle noise averages out under the pool while coherent structure (mode lines,
    bursts, envelopes) survives, so this discriminates the lever arms on exactly the
    content the user wants recovered — unlike raw pixel corr, which is dominated by
    the unpredictable noise realization for broadband modalities."""
    import torch.nn.functional as F
    n = 0
    sa = sb = saa = sbb = sab = 0.0
    for i in range(0, a.shape[0], chunk):
        x = F.avg_pool2d(a[i:i + chunk], k).to(torch.float64)
        y = F.avg_pool2d(b[i:i + chunk], k).to(torch.float64)
        n += x.numel()
        sa += float(x.sum()); sb += float(y.sum())
        saa += float((x * x).sum()); sbb += float((y * y).sum()); sab += float((x * y).sum())
    cov = sab / n - (sa / n) * (sb / n)
    va = saa / n - (sa / n) ** 2
    vb = sbb / n - (sb / n) ** 2
    return cov / max((va * vb) ** 0.5, 1e-12)


def _save_fullshot_figure(gt: torch.Tensor, recon: torch.Tensor, modality: str,
                          out_dir: Path, arm: str = "baseline") -> Path:
    """GT | recon | abs-diff triptych of the stitched shot spectrogram (max-variance
    channel, time-decimated to <= 4096 columns). Returns the PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # PINNED display channel per modality: the max-variance pick changes with the
    # standardization variant (v2 lesson: canonical mhr showed ch0's mode lines, the
    # dataset-stats run showed featureless ch5), which breaks visual comparability.
    # Values = the canonical shot-local run's max-variance choices.
    _FIG_CHANNEL = {"bes": 6, "co2": 2, "ece": 5, "mhr": 0}
    ch = _FIG_CHANNEL.get(modality, int(gt.std(dim=(0, 2, 3)).argmax()))
    g = torch.cat(list(gt[:, ch].unbind(0)), dim=-1).numpy()        # (F, n*T)
    r = torch.cat(list(recon[:, ch].unbind(0)), dim=-1).numpy()
    stride = max(1, g.shape[-1] // 4096)
    g, r = g[:, ::stride], r[:, ::stride]

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if arm == "baseline" else f"_{arm}"
    path = out_dir / f"{modality}_fullshot_{SHOT}{suffix}.png"
    vmin, vmax = np.percentile(g, [1.0, 99.5])
    diff = np.abs(g - r)
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    # GT + recon share the GT color scale; the diff panel gets its OWN scale (log-power
    # GT limits are e.g. -10..-8 for co2 while |diff| is ~0..1 -> reusing them saturates
    # the diff panel to a solid plate).
    panels = ((g, f"GT {modality} ch{ch} — shot {SHOT} (full, stitched)", vmin, vmax),
              (r, f"FSQ-AE reconstruction (whole-shot recon-only overfit, arm={arm})",
               vmin, vmax),
              (diff, "abs diff (own scale)", 0.0, float(np.percentile(diff, 98.0))))
    for ax, (arr, title, lo, hi) in zip(axes, panels):
        im = ax.imshow(arr, origin="lower", aspect="auto", vmin=lo, vmax=hi,
                       cmap="magma", interpolation="none")
        ax.set_title(title)
        ax.set_ylabel("freq bin")
        fig.colorbar(im, ax=ax, pad=0.01)
    axes[-1].set_xlabel(f"STFT frame (stride {stride})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# Whole-shot REGRESSION floors for the LOCKED lineup, set just below the operating
# points measured during the 2026-07-31 campaign (same seeds/configs as the consolidated
# baselines, so the values transfer). NOT quality targets. co2/ece/mhr floors are in
# their per-freq-z target space (numbers are NOT comparable to raw-space runs: whitening
# makes every faint bin count, so pixel corr drops even as MORE structure is encoded —
# judge quality from the figures).
_FULLSHOT_FLOORS = {  # modality: (max nmae, min corr)   [measured nmae / corr, job]
    "bes": (0.55, 0.70),   # 0.472 / 0.778  (fsq6 arm, 5130478)
    "co2": (0.45, 0.50),   # 0.359 / 0.608  (fsq6 arm incl. z, 5130478)
    # ece v3 measured 0.711 / 0.280 (job 5135695): the fully-normalized target is
    # noise-dominated, so pixel corr is a weak instrument here — the FIGURE (coherent
    # structure recovery, best of the campaign per user 2026-08-01) is the ece quality
    # judge; this floor only guards against collapse.
    "ece": (0.80, 0.20),
    "mhr": (0.62, 0.55),   # 0.533 / 0.617  (nbz arm, 5131573)
}


@pytest.mark.skipif(not FULLSHOT, reason="whole-shot GPU rung is opt-in: set IGNITE_FULLSHOT=1")
@pytest.mark.parametrize("modality,arm", _FULLSHOT_CASES)
def test_fsq_ae_reconstructs_whole_shot(modality: str, arm: str) -> None:
    """Rung 2: recon-only FSQ-AE trained on ALL windows of shot 200729 must reconstruct
    the WHOLE stitched spectrogram at (or above) the measured operating point.

    Capacity now matters (~200 windows share one codec). BASELINE arms assert the
    per-modality REGRESSION floors (_FULLSHOT_FLOORS — measured values, and why pixel
    corr understates broadband quality); LEVER arms (freqgrad / fsq6, bes+ece only)
    assert sanity only — their deliverable is the figure + json comparison against the
    baseline, especially `corr_smooth` (the structure-sensitive metric).
    """
    if RENDER_CKPT_TMPL:
        ckpt_path = RENDER_CKPT_TMPL.format(m=modality)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        arm = f"prod{ckpt['step']}"
        print(f"[render:{modality}] {ckpt_path} step={ckpt['step']} "
              f"score={ckpt.get('score')}", flush=True)
        wins = _all_windows(modality, cfg)
        device = _device()
        codec = SpectroCodec(cfg).to(device)
        codec.load_state_dict(ckpt["codec"])
        first_ep, last_ep = 1.0, 0.0            # no training in render mode
    else:
        cfg = _modality_cfg(modality, arm)
        wins = _all_windows(modality, cfg)
        device = _device()
        codec, first_ep, last_ep = _train_full_shot(cfg, wins, FULLSHOT_EPOCHS,
                                                    FULLSHOT_BS, device)
    recon = _reconstruct_windows(codec, wins, device)

    nmae = float((recon - wins).abs().mean()) / float(wins.std())
    corr = _chunked_corr(wins, recon)
    corr_smooth = _chunked_corr_smooth(wins, recon)
    per_win = [float(torch.corrcoef(torch.stack(
        [recon[i].flatten(), wins[i].flatten()]))[0, 1]) for i in range(wins.shape[0])]
    per_win_t = torch.tensor(per_win)

    fig_path = _save_fullshot_figure(wins, recon, modality, FULLSHOT_OUT, arm)
    suffix = "" if arm == "baseline" else f"_{arm}"
    metrics = {
        "shot": SHOT, "modality": modality, "arm": arm,
        "n_windows": int(wins.shape[0]),
        "epochs": FULLSHOT_EPOCHS, "batch_size": FULLSHOT_BS, "lr": LR,
        "fsq_levels": list(cfg.fsq_levels),
        "patch_f": cfg.patch_f, "patch_t": cfg.patch_t, "n_tok": cfg.n_tok,
        "channel_groups": cfg.channel_groups,
        # input-conditioning state — REQUIRED to interpret the metrics: per-freq z
        # changes the target space, so numbers are only comparable at equal flags.
        "input_standardize_raw_z": bool(cfg.input_standardize),
        "logpow_standardize_perfreq_z": bool(cfg.logpow_standardize),
        "freq_grad_weight": cfg.freq_grad_weight,
        "multiscale_recon_weight": cfg.multiscale_recon_weight,
        "first_epoch_mae": first_ep, "last_epoch_mae": last_ep,
        "nmae": nmae, "corr": corr, "corr_smooth": corr_smooth,
        "per_window_corr_min": float(per_win_t.min()),
        "per_window_corr_median": float(per_win_t.median()),
    }
    with open(FULLSHOT_OUT / f"{modality}_fullshot_{SHOT}{suffix}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[fullshot:{modality}:{arm}] nmae={nmae:.4f} corr={corr:.4f} "
          f"corr_smooth={corr_smooth:.4f} per-window corr min/med="
          f"{float(per_win_t.min()):.3f}/{float(per_win_t.median()):.3f}  "
          f"figure={fig_path}", flush=True)

    if RENDER_CKPT_TMPL:
        # render mode: a held-out production checkpoint is NOT expected to meet the
        # overfit floors — figures are the deliverable; only guard against a broken load.
        assert math.isfinite(nmae) and math.isfinite(corr)
        return
    # Raw-space (envelope-dominated) targets reliably drop >2x; whitened targets
    # (per-freq z) start near the speckle floor and only drop ~15-35% even when training
    # succeeds (measured: nbz arms 0.78 -> 0.53-0.66, job 5131573), so their moved-factor
    # is 0.9. Keyed on the cfg, not the arm, since z is part of the locked configs now.
    moved = 0.9 if getattr(cfg, "logpow_standardize", False) else 0.5
    assert last_ep < moved * first_ep, (
        f"{modality}/{arm}: whole-shot training barely moved "
        f"({first_ep:.4f} -> {last_ep:.4f})"
    )
    if arm == "baseline":
        max_nmae, min_corr = _FULLSHOT_FLOORS[modality]
        assert nmae < max_nmae, (
            f"{modality}: whole-shot nmae {nmae:.3f} regressed past the measured floor "
            f"{max_nmae} (see _FULLSHOT_FLOORS)"
        )
        assert corr > min_corr, (
            f"{modality}: whole-shot corr {corr:.3f} regressed past the measured floor "
            f"{min_corr} (see _FULLSHOT_FLOORS)"
        )
    else:
        # lever arms: sanity only — a measured regression vs baseline is a FINDING
        # (recorded in the json/figure), not a suite failure.
        assert math.isfinite(nmae) and math.isfinite(corr)
        assert corr > 0.3, f"{modality}/{arm}: catastrophic recon (corr {corr:.3f})"
