"""IGNITE Phase-B (MaskGIT dynamics) EVALUATION harness.

Loads a trained dynamics checkpoint, seeds K0 real frames from the frame-code cache, rolls out
the world model, and decodes BOTH the ground-truth codes and the predicted codes through the
FROZEN Phase-A codecs, then renders per-modality GT-vs-pred panels.

Decoding BOTH gt and pred codes through the SAME frozen codec places them in an identical
normalized space, so they are directly comparable and the figure/metrics ISOLATE the dynamics
model's error from the codec's reconstruction error. No denormalization is applied (nor needed).

Usage (under the pixi frontier env)::

    python -m tokamak_foundation_model.ignite.eval_dynamics \
        --ckpt /lustre/orion/fus187/proj-shared/models/ignite_production/runs/prod_d512L8/dynamics_latest.pt \
        --shot 200729 --out_dir eval_runs/ignite_dynamics_eval

See docs/IGNITE_DESIGN.md §5 for the frame layout / rollout contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .train_dynamics import load_frozen_codecs
from .dynamics_config import DynamicsConfig, FROZEN_MODALITIES  # noqa: F401 (FROZEN_MODALITIES: API contract)
from .maskgit import MaskGITDynamics

# The exactly-11 modalities to render, and their family grouping for panel layout.
EVAL_MODALITIES: Tuple[str, ...] = (
    "ece", "bes", "mhr", "co2",
    "ts_core_density", "ts_core_temp", "ts_tangential_density", "ts_tangential_temp", "cer_ti",
    "tangtv_lower", "tangtv_upper",
)
SPECTRO = ("ece", "bes", "mhr", "co2")
VIDEO = ("tangtv_lower", "tangtv_upper")
SLOWTS = ("ts_core_density", "ts_core_temp", "ts_tangential_density", "ts_tangential_temp", "cer_ti")

# ---- FIGURE ECONOMY (2026-08-10) ------------------------------------------------------------ #
# A one-page figure has a fixed ink budget: every extra panel costs RESOLUTION in the others.
# Rendering all 11 modalities forced the spectrograms side-by-side at ~2.2 in wide, which
# destroyed exactly what the figure exists to show (rollout quality on a 5 s, 512-bin image).
# So the figure renders a SELECTION at full width — stacked GT over PRED — and the metrics
# json still carries every modality. Override per run with IGNITE_FIG_SPECTRO / _SLOWTS
# (comma-separated names, or "all").
def _fig_sel(env: str, default: Tuple[str, ...], allowed: Tuple[str, ...]) -> Tuple[str, ...]:
    import os
    v = os.environ.get(env)
    if not v:
        return default
    if v.strip().lower() == "all":
        return allowed
    return tuple(n for n in (x.strip() for x in v.split(",")) if n in allowed)


FIG_SPECTRO = _fig_sel("IGNITE_FIG_SPECTRO", ("ece", "mhr"), SPECTRO)
FIG_SLOWTS = _fig_sel("IGNITE_FIG_SLOWTS",
                      ("ts_core_density", "ts_core_temp", "cer_ti"), SLOWTS)

# PAPER GEOMETRY: the PDF must drop into \includegraphics at natural size, so the canvas
# width IS the target width (no bbox_inches="tight", which expands past it) and the font
# size is the printed point size. Defaults: 6.5 in = \textwidth for letter/1in margins,
# 12 pt to match body text. Times New Roman is not installed on Frontier; Liberation Serif
# is METRICALLY IDENTICAL to it (same advance widths) and is the standard substitute.
import os as _os
FIG_W_IN = float(_os.environ.get("IGNITE_FIG_WIDTH_IN", "6.5"))
FIG_PT = float(_os.environ.get("IGNITE_FIG_FONT_PT", "12"))
# Colour maps (user 2026-08-12): spectrograms in a colourblind-safe sequential scheme,
# video in greyscale (raw camera counts read naturally as luminance), differences on a
# diverging map centred at zero. Overridable per-run via env.
CMAP_SPECTRO = _os.environ.get("IGNITE_CMAP_SPECTRO", "Blues")
CMAP_VIDEO = _os.environ.get("IGNITE_CMAP_VIDEO", "gray")
CMAP_DIFF = _os.environ.get("IGNITE_CMAP_DIFF", "RdBu_r")
FIG_SERIF = ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"]

_DEFAULT_CACHE = "/lustre/orion/fus187/proj-shared/models/ignite_production/frame_codes"
# chunk the decode along the time (frame) axis so a codec never sees the whole rollout at once
_DECODE_CHUNK = 16
FRAME_S = 0.05           # one frame = one 50 ms chunk (the shared codec windowing)


# --------------------------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------------------------- #
@torch.no_grad()
def decode_flat(codec, flat: torch.Tensor) -> torch.Tensor:
    """Flat FSQ code indices -> decoded signal (in the codec's normalized recon space).

    flat: (T, n_tok) long. Returns SpectroCodec->(T,C,Fr,Tb) / VideoCodec->(T,C,Tv,H,W) /
    SlowTSCodec->(T,C,Tt).
    """
    quant = codec.quantizer.fsq.indices_to_codes(flat)   # (T, n_tok, d_model)
    return codec.decode(quant)


@torch.no_grad()
def decode_flat_chunked(codec, flat: torch.Tensor, device, chunk: int = _DECODE_CHUNK) -> np.ndarray:
    """decode_flat over time in ``chunk``-frame slices -> a single cpu float32 numpy array.

    Codecs are frozen / no_grad, so chunking only bounds peak memory; the result is identical to a
    single call. ``flat`` is (T, n_tok) long (any device); output leading axis is T.
    """
    outs = []
    T = flat.shape[0]
    for i in range(0, T, chunk):
        sub = flat[i: i + chunk].to(device).long()
        rec = decode_flat(codec, sub)                    # (t, ...)
        outs.append(rec.float().cpu())
    return torch.cat(outs, dim=0).numpy()


# --------------------------------------------------------------------------------------------- #
# load + rollout
# --------------------------------------------------------------------------------------------- #
def load_model(ckpt_path: Path, device) -> Tuple[MaskGITDynamics, DynamicsConfig, int]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    kw = dict(depth=int(ck["cfg_depth"]), d_model=int(ck["cfg_d_model"]))
    if "modalities" in ck:
        # checkpoints since 2026-08-08 carry the trained frame layout (v6 codecs change
        # n_tok/vocab vs the static table) + heads/horizon; older ckpts fall back to defaults.
        from .dynamics_config import ModalitySpec
        kw["modalities"] = tuple(ModalitySpec(*t) for t in ck["modalities"])
    for src, dst in (("cfg_n_heads", "n_heads"), ("cfg_k0", "k0_seed"),
                     ("cfg_n_predict", "n_predict")):
        if src in ck:
            kw[dst] = int(ck[src])
    cfg = DynamicsConfig(**kw)
    cfg.grad_checkpointing = False                        # inference: no recompute
    model = MaskGITDynamics(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    step = int(ck.get("step", 0))
    return model, cfg, step


def load_shot_cache(cache_dir: Path, shot: str) -> Dict:
    p = Path(cache_dir) / f"{shot}.pt"
    if not p.exists():
        raise FileNotFoundError(f"frame-code cache not found: {p}")
    return torch.load(p, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------------------------- #
# actuator counterfactuals
# --------------------------------------------------------------------------------------------- #
# The cache stores actuators ALREADY z-scored per shot over the cached window (train_dynamics.
# actuator_frames), so an edit here is POST-normalization: editing the predicted region cannot
# feed back into the seed frames. (The seed/predict coupling only exists if you re-derive
# actuators from the H5 with edited raw signals — a different, harder experiment.)
#
# Because the per-shot z-score already removed level and scale, a counterfactual that MULTIPLIES
# or OFFSETS a channel is a NO-OP by construction. Only trajectory SHAPE is expressible, which is
# why the modes below are structural (zero / freeze / shuffle / donor / per-group) rather than
# gain-based.
ACTUATOR_MODES = ("real", "zero", "freeze", "shuffle", "donor:<shot>", "group_zero:<group>")


def apply_actuator_mode(act: torch.Tensor, mode: str, K0: int, cache_dir=None,
                        F: int = None) -> torch.Tensor:
    """(F, 70) cached actuators -> counterfactual (F, 70). ``act`` is not modified in place.

    real                 unchanged (baseline)
    zero                 all actuators zeroed over the PREDICTED region [K0, F) — the seed
                         region is left real so the model gets the same context and only the
                         control input differs.
    freeze               hold the last seed frame's actuators across [K0, F): "control stopped".
    shuffle              randomly permute the predicted region's frames (destroys the temporal
                         trajectory, preserves the marginal distribution).
    donor:<shot>         splice another shot's actuators over [K0, F) — the in-distribution
                         counterfactual (a real control trajectory, just not this shot's).
    group_zero:<g>       zero ONE actuator group over [K0, F) (g in _ACT_SPEC: ech_power, pinj,
                         beam_voltage, tinj, gas_flow, gas_raw, rmp) — per-actuator attribution.
    """
    from .train_dynamics import _ACT_SPEC
    F = int(act.shape[0]) if F is None else F
    out = act.clone()
    if mode == "real":
        return out
    if mode == "zero":
        out[K0:F] = 0.0
        return out
    if mode == "freeze":
        out[K0:F] = act[K0 - 1:K0]
        return out
    if mode == "shuffle":
        # deterministic permutation (seeded by F) so a run is reproducible
        g = torch.Generator().manual_seed(1234 + F)
        idx = torch.randperm(F - K0, generator=g) + K0
        out[K0:F] = act[idx]
        return out
    if mode.startswith("donor:"):
        donor = mode.split(":", 1)[1]
        if cache_dir is None:
            raise ValueError("donor mode needs cache_dir")
        d = load_shot_cache(Path(cache_dir), donor)
        da = d["actuators"].float()
        if da.shape[0] < F:
            raise RuntimeError(f"donor {donor} has {da.shape[0]} frames < F={F}")
        out[K0:F] = da[K0:F]
        return out
    if mode.startswith("group_zero:"):
        g = mode.split(":", 1)[1]
        off = 0
        for key, nch in _ACT_SPEC:
            if key == g:
                out[K0:F, off:off + nch] = 0.0
                return out
            off += nch
        raise ValueError(f"unknown actuator group {g!r}; known: {[k for k, _ in _ACT_SPEC]}")
    raise ValueError(f"unknown actuator_mode {mode!r}; expected one of {ACTUATOR_MODES}")


@torch.no_grad()
def rollout_shot(model: MaskGITDynamics, cfg: DynamicsConfig, cache: Dict, K0: int,
                 temperature: float, generator: torch.Generator, device,
                 actuator_mode: str = "real", cache_dir=None
                 ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int]:
    """Seed K0 real frames, roll out, return (gt_codes, pred_codes, K0, F) as cpu long tensors.

    gt_codes[name] / pred_codes[name]: (F, n_tok) over the FULL window [0, F). F = min(cfg.max_frames,
    n_frames). Both are compared over the PREDICTED region [K0, F) downstream.

    ``actuator_mode`` applies a counterfactual to the actuators over [K0, F) — see
    :func:`apply_actuator_mode`. The rollout consumes an IDENTICAL number of RNG draws whichever
    mode is used (the decode loop is a fixed number of multinomial calls), so two runs with the
    same seed differ ONLY through the actuator conditioning — the comparison is exactly paired.
    """
    F = min(cfg.max_frames, int(cache["n_frames"]))
    if F <= K0:
        raise RuntimeError(f"shot has F={F} frames <= K0={K0}; nothing to predict")
    n_predict = F - K0
    names = [m.name for m in cfg.modalities]

    codes = cache["codes"]
    seed_codes = {n: codes[n][:K0].long().unsqueeze(0).to(device) for n in names}   # (1, K0, n_tok)
    act = cache["actuators"][:F].float()                                            # (F, 70)
    act = apply_actuator_mode(act, actuator_mode, K0, cache_dir=cache_dir, F=F)
    actuators = act.unsqueeze(0).to(device)                                         # (1, F, 70)

    traj = model.rollout(seed_codes, actuators, n_predict=n_predict,
                         temperature=temperature, generator=generator)
    gt_codes = {n: codes[n][:F].long().cpu() for n in names}
    pred_codes = {n: traj[n][0, :F].long().cpu() for n in names}
    return gt_codes, pred_codes, K0, F


# --------------------------------------------------------------------------------------------- #
# decode both + metrics
# --------------------------------------------------------------------------------------------- #
def _nrmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """rmse(pred - gt) / std(gt) over finite entries, in float64.

    float64 is REQUIRED: physical-unit decodes reach ~1e19 (Thomson density), whose square
    overflows float32 (inf/inf -> nan nRMSE — the 2026-08-03 density-nan failure)."""
    finite = np.isfinite(pred) & np.isfinite(gt)
    if not finite.any():
        return float("nan")
    p = pred[finite].astype(np.float64)
    g = gt[finite].astype(np.float64)
    d = p - g
    std = float(np.std(g))
    rmse = float(np.sqrt(np.mean(d * d)))
    return rmse / (std + 1e-12)


@torch.no_grad()
def raw_to_output_space(fam: str, raw, denorm_fn=None, mask=None) -> np.ndarray:
    """Map a codec's RAW INPUT window into the SAME space as the denormalized decode.

    The families do NOT agree on what their dataset hands back, so a single rule silently
    corrupts two of them (measured on shot 200729, per-family pre-denorm ranges):

      video   : raw is ALREADY physical camera counts (16 … 235, std 48) while the decode is
                standardized (std ~1). Applying the decode's denorm to raw inflated it to
                65 … 17 400 — a ~60x error that blew out the shared colour limits and made
                the correctly-scaled prediction render as SOLID BLACK.
      slow-TS : raw is standardized, same space as the decode -> denorm applies. But raw also
                carries MISSING-DATA sentinels the codec smooths away (min -5.66 vs decoded
                -4.47); left in, a fully-missing channel plots as a flat line and can win the
                "most variable channel" pick.
      spectro : raw is instance-normalized and denorm maps it correctly (denormed raw vs
                decode: mean -1.217 vs -1.212, std 1.03 vs 1.15).

    ``mask`` (True = valid) is applied as NaN so missing samples are neither plotted nor
    scored; the renderer and _nrmse both treat NaN as absent.
    """
    a = np.asarray(raw, dtype=np.float32)
    if mask is not None:
        m = np.asarray(mask)
        if m.shape == a.shape:
            a = np.where(m.astype(bool), a, np.nan)
        elif m.shape == a.shape[:m.ndim]:                 # per-frame / per-channel mask
            a = np.where(m.astype(bool).reshape(m.shape + (1,) * (a.ndim - m.ndim)), a, np.nan)
    if fam == "video":
        return a                                          # already raw camera counts
    return np.asarray(denorm_fn(a), dtype=np.float32) if denorm_fn else a


@torch.no_grad()
def load_raw_gt(shot: str, codecs: Dict, F: int, data_dir, t0_start: float = 1.0,
                denorm: Dict = None, log=print) -> Dict[str, np.ndarray]:
    """The MEASURED input signal per modality, (F, ...) — NOT the codec round-trip.

    The figure's reference should be the diagnostic as recorded, so the panels show what the
    model predicts against REALITY rather than against a reconstruction (user 2026-08-12).
    This is the same tensor the precompute encodes, taken straight from the codec's input
    dataset and never passed through encode/decode, so the GT-vs-pred difference now contains
    the codec's reconstruction error as well as the dynamics error — the honest end-to-end
    quantity.

    ``t0_start`` MUST match the cache's ``_codec_manifest.json``: it sets the window origin,
    and a mismatch silently shifts the raw frames against the codes (the probe and production
    caches differ by exactly 20 frames for this reason).
    """
    from .train_dynamics import _single_shot_dataset
    out = {}
    for name, (_codec, cfg, fam) in codecs.items():
        try:
            ds = _single_shot_dataset(name, fam, cfg, shot, data_dir, t0_start=t0_start)
            if len(ds) < F:
                log(f"[eval] raw GT {name}@{shot}: only {len(ds)} frames < {F}; skipping")
                continue
            frames, masks = [], []
            for t in range(F):
                item = ds[t]
                x = item[0] if isinstance(item, tuple) else item
                frames.append(np.asarray(x, dtype=np.float32))
                if isinstance(item, tuple) and len(item) > 1 and fam in ("video", "slowts"):
                    masks.append(np.asarray(item[1]))
            arr = np.stack(frames, 0)
            msk = np.stack(masks, 0) if masks else None
            # Family-specific mapping into the decode's output space (see raw_to_output_space):
            # video raw is already physical, the others need the denorm, and missing samples
            # become NaN so they are neither plotted nor scored.
            out[name] = raw_to_output_space(fam, arr, denorm_fn=(denorm or {}).get(name),
                                            mask=msk)
        except Exception as e:                       # a missing diagnostic must not kill the eval
            log(f"[eval] raw GT {name}@{shot} unavailable ({type(e).__name__}: {e})")
    return out


def decode_all(codecs: Dict, gt_codes: Dict, pred_codes: Dict, K0: int, F: int, device,
               denorm: Dict = None, raw_gt: Dict = None) -> Dict[str, Dict[str, np.ndarray]]:
    """Decode BOTH gt + pred codes for every EVAL modality over the FULL window [0, F).

    The seed region [0, K0) is included (pred codes == real codes there) so the figure can show
    the rollout taking over from real context; ``nrmse`` is computed over the PREDICTED region
    [K0, F) only. ``denorm`` optionally maps a modality's decoded array from the codec's
    normalized recon space back to PHYSICAL units ({name: callable(arr) -> arr}); per-window
    normalizations (spectro instance norm, video per-clip std) are inverted with the GT frame's
    stats — for predictions this assumes the true frame's scale, the honest eval convention
    (the codes deliberately do not carry absolute scale). Skips modalities absent from
    ``codecs`` (e.g. placeholder / not requested).

    Returns {name: {"gt": arr, "pred": arr, "nrmse": float, "family": fam, "space": str}}
    with arr leading axis = F frames and space in {"normalized", "physical"}.
    """
    out = {}
    for name in EVAL_MODALITIES:
        if name not in codecs:
            continue
        codec, _cfg, fam = codecs[name]
        gt = decode_flat_chunked(codec, gt_codes[name][:F], device)       # (F, ...)
        pred = decode_flat_chunked(codec, pred_codes[name][:F], device)
        # PERSISTENCE BASELINE — the do-nothing rollout: freeze the last seed frame for the
        # whole predicted region. The decoder is deterministic, so decoding that ONE frame
        # and tiling is exact and ~(F-K0)x cheaper than decoding the frozen sequence.
        # Denorm is applied to the tiled array so the baseline gets the SAME per-frame GT
        # stats treatment the prediction gets (fair comparison; see decode_all docstring).
        pers1 = decode_flat_chunked(codec, gt_codes[name][K0 - 1:K0], device)
        pers = np.repeat(pers1, F, axis=0)
        # GROUND TRUTH = the MEASURED input when available (user 2026-08-12), falling back to
        # the codec round-trip. Both live in the same normalized space (the codec is an
        # autoencoder), so the shared denorm below applies unchanged. Swapping the reference
        # also moves the metrics: nrmse/persistence/skill are now measured against reality,
        # so they include codec reconstruction error and are NOT comparable to numbers from
        # runs that scored against the decoded GT.
        # raw_gt entries arrive ALREADY in the output space (see raw_to_output_space) — the
        # per-family conversion cannot live here because it needs the mask and the family's
        # own denorm. So denorm applies to the decode path only.
        gt_src = "decoded"
        space = "normalized"
        if denorm and name in denorm:
            gt, pred, pers = denorm[name](gt), denorm[name](pred), denorm[name](pers)
            space = "physical"
        # Shapes must be compared AFTER denorm: the slow-TS inverse DROPS the padded
        # radial-zone tail, so the decoded array narrows (12 -> 10 channels) partway through.
        # Comparing before denorm padded the raw window to the pre-denorm width and produced a
        # 12-vs-10 mismatch at the nRMSE.
        if raw_gt and name in raw_gt and raw_gt[name].shape != gt.shape:
            r = np.asarray(raw_gt[name], dtype=np.float32)
            if (r.ndim == gt.ndim and r.shape[0] == gt.shape[0]
                    and r.shape[2:] == gt.shape[2:] and r.shape[1] < gt.shape[1]):
                # measured signal narrower than the decode -> NaN-pad; NaN already reads as
                # "absent" to the renderer and _nrmse, so the pad is ignored, not scored.
                pad = np.full((r.shape[0], gt.shape[1] - r.shape[1]) + r.shape[2:],
                              np.nan, dtype=np.float32)
                raw_gt = {**raw_gt, name: np.concatenate([r, pad], axis=1)}
            elif r.shape[1] > gt.shape[1] and r.shape[0] == gt.shape[0] \
                    and r.shape[2:] == gt.shape[2:]:
                raw_gt = {**raw_gt, name: r[:, :gt.shape[1]]}      # drop the same tail
        use_raw = bool(raw_gt) and name in raw_gt and raw_gt[name].shape == gt.shape
        if raw_gt and name in raw_gt and not use_raw:
            # a shape disagreement we cannot reconcile means the raw window and the codes are
            # not aligned; using it would silently compare different things.
            print(f"[eval] raw GT {name}: shape {raw_gt[name].shape} != decoded {gt.shape}"
                  " — falling back to the decoded round-trip", flush=True)
        if use_raw:
            gt, gt_src = np.asarray(raw_gt[name], dtype=np.float32), "measured"
        nr, nr_p = _nrmse(pred[K0:], gt[K0:]), _nrmse(pers[K0:], gt[K0:])
        out[name] = {"gt": gt, "pred": pred, "gt_source": gt_src,
                     "nrmse": nr, "nrmse_persistence": nr_p,
                     # skill > 0 => better than freezing; 1.0 => perfect; < 0 => worse
                     "nrmse_skill": (1.0 - nr / nr_p) if nr_p > 0 else float("nan"),
                     "family": fam, "space": space}
    return out


def self_check(decoded: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Guard a broken DECODE path: decoded GT must be finite + non-constant (spectro/slowts).

    The non-constant invariant holds for the codec ROUND-TRIP only. With measured ground truth
    (``gt_source == "measured"``) a constant is LEGITIMATE — it is exactly what an absent
    diagnostic looks like, and ~37% of (shot, modality) pairs are absent in this dataset. So a
    constant measured signal is reported, not asserted on; the static screening downstream
    (see curate_core / the presence map) is what handles those. Asserting here killed eval
    5245138 on the first val shot, where ece simply was not recorded.
    """
    checked, constant_measured = 0, []
    for name, d in decoded.items():
        if d["family"] not in ("spectro", "slowts"):
            continue
        gt = d["gt"]
        measured = d.get("gt_source") == "measured"
        finite = gt[np.isfinite(gt)]
        if not finite.size:
            # ALL-NaN measured GT = the diagnostic recorded nothing in this shot (missing
            # samples are masked to NaN by raw_to_output_space). Legitimate, and downstream
            # already treats it as absent: _nrmse returns NaN and the curve/figure filter it.
            # An all-NaN DECODED array is still a genuine fault.
            if measured:
                constant_measured.append(f"{name}(all-missing)")
                continue
            raise AssertionError(f"[self-check] {name}: DECODED GT has NO finite values")
        if float(finite.std()) > 0.0:
            checked += 1
            continue
        if d.get("gt_source") == "measured":
            constant_measured.append(name)          # absent diagnostic — expected, not a fault
            continue
        raise AssertionError(
            f"[self-check] {name}: DECODED GT is constant (std=0) — broken decode path")
    assert checked > 0 or constant_measured, \
        "[self-check] no spectro/slowts modalities decoded — nothing to validate"
    msg = f"[self-check] OK: {checked} spectro/slowts GT are finite + non-constant"
    if constant_measured:
        msg += (f"; {len(constant_measured)} constant because the diagnostic is ABSENT in this "
                f"shot ({', '.join(sorted(constant_measured))})")
    print(msg, flush=True)


# --------------------------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------------------------- #
# static-detection threshold: a GT decode whose temporal std (mean-reduced over all
# non-frame axes) is below this FRACTION of its overall signal scale (max|.|) is treated as
# STATIC — the codec collapsed the whole shot to ~1 constant code, so there is no dynamics to
# render. Verified: ece/co2 for shot 200729 have temporal-std EXACTLY 0.0.
_STATIC_EPS = 1e-4


def _temporal_variation(gt: np.ndarray) -> Tuple[float, float]:
    """Return (temporal_std, signal_scale) of a GT decode with leading axis = frames.

    temporal_std = mean over all non-frame elements of the per-element std across frames (masking
    non-finite). signal_scale = max|.| over finite entries (a robust magnitude reference).
    """
    a = np.asarray(gt, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        return 0.0, 0.0
    scale = float(np.max(np.abs(a[finite])))
    masked = np.where(finite, a, np.nan)
    with np.errstate(invalid="ignore"):
        per_elem_std = np.nanstd(masked, axis=0)              # std over the frame axis
        tstd = float(np.nanmean(per_elem_std))
    if not np.isfinite(tstd):
        tstd = 0.0
    return tstd, scale


def _is_static(gt: np.ndarray) -> bool:
    """True if the GT decode is temporally ~constant (below _STATIC_EPS relative to its scale)."""
    tstd, scale = _temporal_variation(gt)
    return tstd <= _STATIC_EPS * max(scale, 1e-12)


def _annotate_static(ax, name: str) -> None:
    """Draw the honest 'no dynamics to show' annotation panel for a collapsed modality."""
    import matplotlib.pyplot as plt
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                               fill=True, facecolor="0.93", edgecolor="0.55", lw=1.2))
    ax.text(0.5, 0.5,
            f"{name}: STATIC\ncodec encodes this shot to a constant\n(no dynamics to show)",
            transform=ax.transAxes, ha="center", va="center", fontsize=11,
            color="0.30", fontweight="bold", wrap=True)


WARMUP_S = 1.0                     # window origin in shot time (windowing convention)
_GT_C, _PR_C = "#1a1a19", "#eb6834"            # ground truth ink / prediction orange
_VID_C = {"tangtv_lower": "#2a78d6", "tangtv_upper": "#eb6834"}
# slow-TS display units: (scale, label) — the H5 stores eV / m^-3; temperatures display as
# keV (x1e-3), matching eval_e2e_animation_tokamak._TRACE_SCALES / _TRACE_LABELS.
# Compact symbols for the normalized case. The raw modality name is longer than a row is
# tall once rotated, so it overlaps the rows above and below whatever the font size.
_SLOWTS_SYMBOL = {
    "ts_core_density": r"$n_e$ core", "ts_core_temp": r"$T_e$ core",
    "ts_tangential_density": r"$n_e$ tang.", "ts_tangential_temp": r"$T_e$ tang.",
    "cer_ti": r"$T_i$", "cer_rot": "rot.", "mse": "MSE",
}

_SLOWTS_UNITS = {  # (display scale, SHORT ylabel, region tag for the corner annotation)
    "ts_core_density": (1.0, r"$n_e$ (m$^{-3}$)", "core"),
    "ts_core_temp": (1e-3, r"$T_e$ (keV)", "core"),
    "ts_tangential_density": (1.0, r"$n_e$ (m$^{-3}$)", "tang."),
    "ts_tangential_temp": (1e-3, r"$T_e$ (keV)", "tang."),
    "cer_ti": (1e-3, r"$T_i$ (keV)", ""),
}


def render_figure(decoded: Dict[str, Dict[str, np.ndarray]], shot: str, step: int,
                  temperature: float, K0: int, F: int, out_dir: Path,
                  t_origin: float = WARMUP_S) -> Tuple[Path, Path]:
    """Comparison figure in the 200729_comparison style (the FAITH-era reference layout).

    Every time panel shares ONE aligned absolute shot-time axis (warmup 1.0 s origin) with a
    dashed 'prediction start' marker; ground truth is ink-black, prediction orange, throughout.
    Sections: (a..) slow-TS best-channel physical traces; then per spectro modality a
    GT | prediction strip pair (stitched across the whole timeline) + a time-mean power-
    spectrum side panel over the predicted region; then per video camera GT | prediction |
    difference at mid-rollout; then per-frame video nRMSE(t). Static (codec-collapsed)
    modalities get an honest annotation strip. Saves PNG + PDF; returns paths.
    """
    import string

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    n_pred = F - K0
    t0, t1 = t_origin, t_origin + F * FRAME_S
    t_roll = t_origin + K0 * FRAME_S
    ts = t_origin + np.arange(F) * FRAME_S
    mid = K0 + n_pred // 2
    from .config import STFT_FS, STFT_N_FFT
    khz_per_bin = (STFT_FS / STFT_N_FFT) / 1e3            # 0.488 kHz/bin at 500 kHz / 1024

    # PAPER-READY: designed at PRINT size (7.2 in = full text width, placed 1:1 — never
    # shrunk), so these font sizes are the printed sizes.
    _p = FIG_PT
    plt.rcParams.update({
        "font.family": "serif", "font.serif": FIG_SERIF,
        "mathtext.fontset": "stix",              # Times-compatible math
        "font.size": _p, "axes.labelsize": _p, "axes.titlesize": _p,
        "xtick.labelsize": _p - 1, "ytick.labelsize": _p - 1, "legend.fontsize": _p - 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    _fs = _p / 7.0                                # row heights scale with type size

    # FIG_* selections keep the one-page ink budget spent on RESOLUTION, not panel count
    # (every modality is still scored in eval_metrics.json). "all" via IGNITE_FIG_* env.
    spectro = [n for n in FIG_SPECTRO if n in decoded]
    video = [n for n in VIDEO if n in decoded]
    slowts = [n for n in FIG_SLOWTS if n in decoded]
    static = {n: _is_static(decoded[n]["gt"]) for n in decoded}
    # FIXED LAYOUT (2026-08-12). The row set is a property of the FIGURE, not of the shot.
    # Static modalities used to be pulled out of their row into a compact 3-across block at the
    # bottom, so the row count, row order and page height all changed with whichever diagnostics
    # happened to be flat in that shot — measured on the production eval list: 8 distinct
    # availability patterns across 10 shots, i.e. 8 structurally different figures. Comparing
    # panels ACROSS shots is the point of this figure (seen vs unseen discharges), and that
    # requires panel k to be the same modality in every render. Every selected modality now
    # keeps its own row in a fixed order; a static one still draws (a flat trace / constant
    # image is the honest picture) and is marked STATIC in its ylabel by _static_ylab.
    dyn_slow, dyn_spec, dyn_vid = slowts, spectro, video
    statics: list = []                            # compact static block retired by the above

    def _static_ylab(name: str, lab: str) -> str:
        """Keep the collapsed-modality signal that the retired static block used to carry."""
        return f"{lab}  [STATIC]" if static.get(name) else lab
    any_phys = any(d.get("space") == "physical" for d in decoded.values())
    unit = "phys" if any_phys else "norm"

    def _best_channel(gt: np.ndarray, axis_reduce: Tuple[int, ...]) -> int:
        a = np.where(np.isfinite(gt), gt, np.nan)
        with np.errstate(invalid="ignore"):
            per_fc = np.nanmean(a, axis=axis_reduce)
            var_c = np.nanvar(per_fc, axis=0)
        var_c = np.where(np.isfinite(var_c), var_c, -np.inf)
        return int(np.argmax(var_c)) if np.isfinite(var_c).any() else 0

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson r over finite elements — the agreement the figure must make visible."""
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 2:
            return float("nan")
        x, y = a[m].astype(np.float64).ravel(), b[m].astype(np.float64).ravel()
        sx, sy = x.std(), y.std()
        if sx <= 0 or sy <= 0:
            return float("nan")
        return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))

    def _stitch(arr: np.ndarray, ch: int) -> np.ndarray:
        g = np.concatenate(list(arr[:, ch]), axis=-1)          # (Fr, F*Tb)
        stride = max(1, g.shape[-1] // 4096)
        return g[:, ::stride]

    # ---- layout: one column of aligned-time rows ------------------------------------------- #
    # spectro gets TWO FULL-WIDTH rows per modality (GT stacked over PRED). Side-by-side
    # thirds cut each strip to ~2.2 in and made the rollout quality unreadable — the whole
    # point of the figure. The power spectra move to one compact row at the end.
    base_rows: list = ([("slow", n) for n in dyn_slow]
                       + [("spec", n) for n in dyn_spec]
                       + [("vid", n) for n in dyn_vid]
                       + ([("static", None)] if statics else []))
    rows: list = []                # spacer rows between sections stop title/label collisions
    _sect = {}                                    # (spectro is one row per modality)
    for k, n in base_rows:
        if rows and _sect.get(rows[-1][0], rows[-1][0]) != _sect.get(k, k):
            rows.append(("gap", None))
        rows.append((k, n))
    # per-row heights in INCHES (the figure must fit ONE page at 100% scale: <= ~9.5 in
    # after bbox trim, so the 7 pt fonts are the printed sizes)
    # row heights in INCHES, scaled with type size so panels stay proportionate at 12 pt
    # row heights in INCHES at the 12 pt design point (a 12 pt axis needs ~0.45 in for the
    # frame plus ~0.25 in for tick labels); they scale mildly if FIG_PT is changed.
    _k = 0.55 + 0.45 * (_p / 12.0)
    heights = {"slow": 0.58 * _k, "spec": 1.05 * _k,
               "psrow": 0.62 * _k, "vid": 0.90 * _k, "vrmse": 0.60 * _k,
               "gap": 0.42 * _k,
               "static": 0.6 * _k * ((len(statics) + 2) // 3 or 1)}
    # section gap must clear a section's x-label AND the next section's titles
    heights["gap"] = 0.20
    hr = [heights[k] for k, _n in rows]
    # EXACT delivered size: the canvas IS the figure (no bbox_inches="tight", which grows
    # past the target and is why the PDF came out ~75 pt too wide for \includegraphics).
    # Margins are in inches -> converted to fractions, so they hold at any font size.
    _mar_l = 0.10 + 0.075 * _p                        # inches (ylabel + tick labels)
    _mar_r = 0.10 + 0.030 * _p                        # colorbar tick labels live here
    _mar_b, _mar_t = 0.06 + 0.046 * _p, 0.06 + 0.015 * _p
    _hsp = 0.46                                   # fraction of MEAN row height
    # 0.28 packed the blocks tight enough that the spectro row's x-label ran into
    # the video row's titles; the extra page height is worth the legibility.
    n_rows = max(len(rows), 1)
    fig_h = (sum(hr) * (1.0 + _hsp * (n_rows - 1) / n_rows)) + _mar_b + _mar_t
    fig = plt.figure(figsize=(FIG_W_IN, fig_h))
    outer = GridSpec(len(rows), 1, figure=fig, height_ratios=hr, hspace=_hsp,
                     left=_mar_l / FIG_W_IN, right=1.0 - _mar_r / FIG_W_IN,
                     top=1.0 - _mar_t / fig_h, bottom=_mar_b / fig_h)
    letters = iter(string.ascii_lowercase)

    def _letter(ax):
        # figure-relative x (left edge) + row top: independent of this axis' ylabel width,
        # so letters never sit on a neighbouring row's label.
        bb = ax.get_position()
        fig.text(0.012, bb.y1 + 0.004, next(letters), fontsize=_p, fontweight="bold",
                 va="bottom", ha="left")

    def _time_axis(ax, ticks: bool, xlabel: bool = False):
        ax.set_xlim(t0, t1)
        ax.axvline(t_roll, color="0.45", ls="--", lw=0.9)
        ax.tick_params(labelbottom=ticks)
        if xlabel:
            ax.set_xlabel("Time (s)")

    last_of = {k: max(i for i, (kk, _n) in enumerate(rows) if kk == k)
               for k in {k for k, _n in rows}}
    legend_host = None                          # first drawn time panel hosts the GT/pred legend

    # ---- slow-TS traces --------------------------------------------------------------------- #
    for i, (kind, name) in enumerate(rows):
        if kind != "slow":
            continue
        d = decoded[name]
        ax = fig.add_subplot(outer[i])
        gt_m = np.where(np.isfinite(d["gt"]), d["gt"], np.nan)
        pr_m = np.where(np.isfinite(d["pred"]), d["pred"], np.nan)
        bc = _best_channel(d["gt"], axis_reduce=(2,))
        with np.errstate(invalid="ignore"):
            tr_gt = np.nanmean(gt_m[:, bc, :], axis=-1)
            tr_pr = np.nanmean(pr_m[:, bc, :], axis=-1)
        tag = ""
        if d.get("space") == "physical" and name in _SLOWTS_UNITS:
            scale, ylab, tag = _SLOWTS_UNITS[name]
            tr_gt, tr_pr = tr_gt * scale, tr_pr * scale   # e.g. eV -> keV (display only)
        else:
            # unit is figure-global, so it goes in the corner annotation, not every y-label
            ylab = _SLOWTS_SYMBOL.get(name, name)
        ax.plot(ts, tr_gt, color=_GT_C, lw=1.7, solid_capstyle="round", zorder=2)
        ax.plot(ts, tr_pr, color=_PR_C, lw=0.9, ls=(0, (2.5, 1.6)), zorder=3)
        if legend_host is None:
            legend_host = ax
        # single-line ylabel (two-line labels bleed into neighbor rows on the one-page
        # canvas); the channel tag rides with the nRMSE corner annotation instead.
        # Rotated 12 pt on a ~0.7 in tall row is taller than the row, so full-size y-labels
        # collide with the panels above and below. Shrink them to fit their own panel.
        ax.set_ylabel(_static_ylab(name, ylab), fontsize=_p - 4)
        sk = d.get("nrmse_skill")
        corner = (f"{tag} ch{bc:02d} · nRMSE {d['nrmse']:.3f}"
                  + (f" · skill {sk:+.2f}" if sk is not None and np.isfinite(sk) else "")).lstrip()
        ax.text(0.995, 1.04, corner, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=_p - 3, color="0.35")
        _time_axis(ax, ticks=(i == last_of["slow"]), xlabel=False)
        _letter(ax)
        if i == 0:
            ax.text(t_roll + 0.02, 1.04, "prediction start",
                    transform=ax.get_xaxis_transform(), fontsize=_p - 3, color="0.45",
                    va="bottom", ha="left")
        ax.grid(alpha=0.2, lw=0.4)

    # ---- spectro: GT (left) | PREDICTION (right), shared colour scale ----------------------- #
    first_spec = min((i for i, (k, _n) in enumerate(rows) if k == "spec"), default=None)
    last_spec = max((i for i, (k, _n) in enumerate(rows) if k == "spec"), default=-1)
    _spec_ref: list = []                          # first GT axis: the shared-x reference
    for i, (kind, name) in enumerate(rows):
        if kind != "spec":
            continue
        d = decoded[name]
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.06)
        bc = _best_channel(d["gt"], axis_reduce=(2, 3))
        g_img, p_img = _stitch(d["gt"], bc), _stitch(d["pred"], bc)
        fin = g_img[np.isfinite(g_img)]
        vmin, vmax = (float(np.percentile(fin, 1.0)), float(np.percentile(fin, 99.5))) \
            if fin.size else (0.0, 1.0)
        ext = (t0, t1, 0.0, float(d["gt"].shape[2]) * khz_per_bin)
        # TRUE shared axes: prediction shares x AND y with its ground truth, and every
        # spectro row shares x with the first, so all time axes are locked together
        # (identical limits/ticks by construction, not by coincidence).
        ax_g = fig.add_subplot(inner[0], sharex=(_spec_ref[0] if _spec_ref else None))
        ax_p = fig.add_subplot(inner[1], sharex=ax_g, sharey=ax_g)
        if not _spec_ref:
            _spec_ref.append(ax_g)
        for ax, img in ((ax_g, g_img), (ax_p, p_img)):
            # The stitched spectrogram is ~512 x (F*n_time) -- far more pixels than the
            # panel gets on the page -- so "none" nearest-neighbours it and drops most
            # columns, which is what makes the image look blocky/aliased. "antialiased"
            # applies a proper downsampling filter and preserves the fine structure that
            # this panel exists to show.
            ax.imshow(img, aspect="auto", origin="lower", cmap=CMAP_SPECTRO, vmin=vmin,
                      vmax=vmax, extent=ext, interpolation="antialiased")
            ax.axvline(t_roll, color="w", ls="--", lw=0.9)
            ax.set_xlim(t0, t1)
            ax.tick_params(labelbottom=(i == last_spec))
            if i == last_spec:
                ax.set_xlabel("Time (s)")
        ax_p.tick_params(labelleft=False)          # shared scale -> one y-axis is enough
        from matplotlib.ticker import MaxNLocator
        ax_g.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))  # no "1.00.0"
        ax_p.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
        ax_g.set_ylabel(_static_ylab(name, f"{name}\nFreq (kHz)"), fontsize=_p - 4)
        _r, _sk = _corr(d["pred"][K0:], d["gt"][K0:]), d.get("nrmse_skill")
        ax_p.text(0.98, 0.94, f"r {_r:.3f} · nRMSE {d['nrmse']:.3f}"
                  + (f" · skill {_sk:+.2f}" if _sk is not None and np.isfinite(_sk) else ""),
                  transform=ax_p.transAxes, ha="right", va="top", fontsize=_p - 3,
                  color="0.15", bbox=dict(boxstyle="square,pad=0.18", fc="white",
                                          ec="none", alpha=0.8))
        if i == first_spec:
            ax_g.set_title("Ground truth")
            ax_p.set_title("Prediction")
        _letter(ax_g)

    # ---- one compact row of time-mean power spectra (statistics check) ---------------------- #
    for i, (kind, _n) in enumerate(rows):
        if kind != "psrow":
            continue
        inner = GridSpecFromSubplotSpec(1, max(len(dyn_spec), 1), subplot_spec=outer[i],
                                        wspace=0.34)
        for j, name in enumerate(dyn_spec):
            d = decoded[name]
            bc = _spec_scale.get(name, (0, 0, 1))[0]
            ax_s = fig.add_subplot(inner[j])
            with np.errstate(invalid="ignore"):
                ps_g = np.nanmean(np.where(np.isfinite(d["gt"][K0:, bc]),
                                           d["gt"][K0:, bc], np.nan), axis=(0, 2))
                ps_p = np.nanmean(np.where(np.isfinite(d["pred"][K0:, bc]),
                                           d["pred"][K0:, bc], np.nan), axis=(0, 2))
            fb = np.arange(len(ps_g)) * khz_per_bin
            ax_s.plot(fb, ps_g, color=_GT_C, lw=1.7, solid_capstyle="round", zorder=2)
            ax_s.plot(fb, ps_p, color=_PR_C, lw=0.9, ls=(0, (2.5, 1.6)), zorder=3)
            ax_s.set_xlabel("Freq (kHz)", fontsize=6.5)
            ax_s.tick_params(labelsize=6)
            ax_s.grid(alpha=0.2, lw=0.4)
            ax_s.set_title(f"{name} power spectrum", fontsize=6.5)
            if j == 0:
                ax_s.set_ylabel("log power", fontsize=6.5)
                _letter(ax_s)

    # ---- video: GT | PRED | difference at mid-rollout --------------------------------------- #
    first_vid = min((i for i, (k, _n) in enumerate(rows) if k == "vid"), default=None)
    for i, (kind, name) in enumerate(rows):
        if kind != "vid":
            continue
        d = decoded[name]
        # colorbars get their OWN columns NEXT TO their panel: attaching them with
        # ax=... steals width from the image axes and knocks this row out of alignment
        # with the spectro/trace rows (measured 2026-08-10: right edge 0.970 vs 0.985).
        inner = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[i], wspace=0.30,
                                        width_ratios=[1.0, 1.0, 0.05, 1.0, 0.05])
        gt, pr = d["gt"], d["pred"]
        mid_t = gt.shape[2] // 2
        img_g, img_p = gt[mid, 0, mid_t], pr[mid, 0, mid_t]
        # SCALE GUARD. With measured ground truth (the default since 2026-08-12) video GT is
        # raw camera counts (std ~48) while the decode is standardized (std ~1), because
        # decode_all is called without `denorm`. Painting the prediction with limits taken
        # from GT then drives every pixel to the bottom of the colormap -- the prediction
        # renders SOLID BLACK and its nRMSE compares std-1 against std-48. When the two sides
        # are that far apart they are not in the same space, so compare SHAPE: z-score each
        # side by its own statistics and say so. Matched spaces keep the old shared limits.
        _zg, _zp = img_g[np.isfinite(img_g)], img_p[np.isfinite(img_p)]
        _sg = float(np.std(_zg)) if _zg.size else 0.0
        _sp = float(np.std(_zp)) if _zp.size else 0.0
        _mg = float(np.mean(_zg)) if _zg.size else 0.0
        _mp = float(np.mean(_zp)) if _zp.size else 0.0
        # Two independent signatures of "different spaces": a scale ratio, and an OFFSET.
        # Counts-vs-standardized shows up mostly as offset (mean ~35 vs ~0) with a scale ratio
        # near 5, so a ratio-only test at >5x missed it and the prediction still rendered black.
        # Key off the SAME flag that stamps [STATIC] on the row label. A per-frame std test is
        # not equivalent: after z-scoring, a collapsed diagnostic still shows speckle, so the
        # local std is non-zero while the whole-array nRMSE has already blown up to 4.7e3 and
        # the skill to -8.9e7 by dividing through a variance that is essentially zero.
        gt_static = bool(static.get(name)) or _sg <= 1e-9 * max(1.0, abs(_mg))
        mismatched = (_sg > 0 and _sp > 0
                      and (max(_sg, _sp) / min(_sg, _sp) > 3.0
                           or abs(_mg - _mp) > 2.0 * max(_sg, _sp)))
        if mismatched:
            img_g = (img_g - float(np.mean(_zg))) / (_sg + 1e-12)
            img_p = (img_p - float(np.mean(_zp))) / (_sp + 1e-12)
        diff = img_p - img_g
        fin = img_g[np.isfinite(img_g)]
        vmin, vmax = (float(np.percentile(fin, 1.0)), float(np.percentile(fin, 99.0))) \
            if fin.size else (0.0, 1.0)
        dmax = float(np.nanpercentile(np.abs(diff), 99.0)) or 1.0
        ax_g = fig.add_subplot(inner[0])
        ax_p = fig.add_subplot(inner[1])
        cax_p = fig.add_subplot(inner[2])
        ax_d = fig.add_subplot(inner[3])
        cax_d = fig.add_subplot(inner[4])
        ax_g.imshow(img_g, cmap=CMAP_VIDEO, vmin=vmin, vmax=vmax, aspect="auto")
        im_p = ax_p.imshow(img_p, cmap=CMAP_VIDEO, vmin=vmin, vmax=vmax, aspect="auto")
        im_d = ax_d.imshow(diff, cmap=CMAP_DIFF, vmin=-dmax, vmax=dmax, aspect="auto")
        _fg = img_g[np.isfinite(img_g)]
        _rel = float(np.nanmax(np.abs(diff))) / (float(np.std(_fg)) + 1e-12) if _fg.size else float("nan")
        _dtxt = ("ground truth is constant — not scored" if gt_static
                 else f"max |diff| = {_rel:.2f}" + r"$\,\sigma_{GT}$")
        ax_d.text(0.02, 0.04, _dtxt, transform=ax_d.transAxes, ha="left", va="bottom",
                  fontsize=_p - 3, color="0.2")
        for a in (ax_g, ax_p, ax_d):
            a.set_xticks([]); a.set_yticks([])
        ax_g.set_ylabel(_static_ylab(name, name.replace("tangtv_", "tangtv\n")
                                     + ("\n(z-scored)" if mismatched else "")),
                        fontsize=_p - 4)
        _rv, _skv = _corr(pr[K0:], gt[K0:]), d.get("nrmse_skill")
        _ptxt = ("not scored (static GT)" if gt_static else
                 f"r {_rv:.3f} · nRMSE {d['nrmse']:.3f}"
                 + (f" · skill {_skv:+.2f}" if _skv is not None and np.isfinite(_skv) else ""))
        ax_p.text(0.99, 0.04, _ptxt,
                  transform=ax_p.transAxes, ha="right", va="bottom", fontsize=_p - 3,
                  color="0.15", bbox=dict(boxstyle="square,pad=0.18", fc="white",
                                          ec="none", alpha=0.78))
        if i == first_vid:
            ax_g.set_title(f"Ground truth (t = {t_origin + mid * FRAME_S:.2f} s)")
            ax_p.set_title("Prediction")
            ax_d.set_title("Difference (pred − GT)")
        for cb, im in ((cax_p, im_p), (cax_d, im_d)):
            fig.colorbar(im, cax=cb)
            cb.tick_params(labelsize=5.5, length=2, pad=1)
        _letter(ax_g)

    # ---- video nRMSE(t): per-frame error across the rollout --------------------------------- #
    for i, (kind, _n) in enumerate(rows):
        if kind != "vrmse":
            continue
        ax = fig.add_subplot(outer[i])
        for name in dyn_vid:
            gt, pr = decoded[name]["gt"], decoded[name]["pred"]
            per = []
            for f in range(K0, F):
                m = np.isfinite(gt[f]) & np.isfinite(pr[f])
                if not m.any():
                    per.append(np.nan)
                    continue
                g = gt[f][m].astype(np.float64)
                p = pr[f][m].astype(np.float64)
                per.append(float(np.sqrt(np.mean((p - g) ** 2)) / (np.std(g) + 1e-12)))
            ax.plot(ts[K0:], per, color=_VID_C.get(name, _GT_C), lw=1.2,
                    label=name.replace("tangtv_", "").capitalize() + " divertor")
        ax.set_ylabel("video nRMSE")
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        ax.grid(alpha=0.2, lw=0.4)
        _time_axis(ax, ticks=True, xlabel=True)
        _letter(ax)

    # ---- static (codec-collapsed) annotation strip ------------------------------------------ #
    for i, (kind, _n) in enumerate(rows):
        if kind != "static":
            continue
        inner = GridSpecFromSubplotSpec((len(statics) + 2) // 3, 3, subplot_spec=outer[i],
                                        wspace=0.12, hspace=0.35)
        for j, name in enumerate(statics):
            _annotate_static(fig.add_subplot(inner[j // 3, j % 3]), name)

    # GT/prediction legend INSIDE the first panel (no figure-level header text — run
    # metadata belongs in the paper caption, not the figure)
    if legend_host is not None:
        handles = [plt.Line2D([], [], color=_GT_C, lw=1.2, label="Ground truth"),
                   plt.Line2D([], [], color=_PR_C, lw=1.2, label="Prediction")]
        legend_host.legend(handles=handles, loc="upper left", frameon=True, ncol=2,
                           fontsize=_p - 2, borderaxespad=0.25, handlelength=1.6,
                           columnspacing=0.8, framealpha=0.9, edgecolor="none")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"dynamics_eval_{shot}_step{step}.png"
    pdf = out_dir / f"dynamics_eval_{shot}_step{step}.pdf"
    # NO bbox_inches: the PDF must stay exactly FIG_W_IN wide for \includegraphics.
    # The PNG is for screen reading, so give it enough pixels to be legible.
    fig.savefig(png, dpi=int(_os.environ.get("IGNITE_FIG_PNG_DPI", "400")))
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


# --------------------------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------------------------- #
def run(ckpt: str, shot: str, cache_dir: str, out_dir: str,
        temperature: float = 1.0, seed: int = 0, k0: int = 0, codec_tmpl: str = None,
        render_all: bool = False, val_tail: int = 0, val_n: int = 0,
        split_seed: int = 0, actuator_mode: str = "real", log=print) -> Dict:
    """Evaluate a trained dynamics ckpt on one or more shots (comma-separated ``shot``).

    Per shot: rollout from K0 real frames, per-modality TOKEN ACCURACY over the predicted
    region (all modalities in the ckpt's layout — no codec needed), decoded nRMSE + figure
    for the EVAL_MODALITIES whose codec resolves (``codec_tmpl`` tried before the frozen
    manifest — MUST match the cache's _codec_manifest.json codecs, or codes and decoder
    disagree). Figures render for the first shot only unless ``render_all``. Writes
    ``eval_metrics.json`` (per-shot + mean) to ``out_dir``.
    """
    import json
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[eval] device={device} ckpt={ckpt} actuator_mode={actuator_mode}", flush=True)

    if val_tail:
        # Resolve the held-out shots from the SAME split the trainer used. PREFER the
        # split_shots.json the run froze next to its checkpoint: recomputing from a cache
        # that has grown since training would hand back DIFFERENT shots — some of which the
        # model trained on. Fall back to recomputing only for runs that predate the freeze.
        from .train_dynamics import split_shots
        sp = Path(ckpt).parent / "split_shots.json"
        if sp.exists():
            d = json.loads(sp.read_text())
            have = {q.stem for q in Path(cache_dir).glob("*.pt")}
            val = [s for s in d["val"] if s in have]
            log(f"[eval] split from {sp} (frozen by the run): val={len(val)} "
                f"test={len(d.get('test', []))} HELD OUT, seed={d.get('split_seed')}",
                flush=True)
        else:
            _train, val, _test = split_shots(cache_dir, val_n or val_tail,
                                             split_seed=split_seed)
            log(f"[eval] NO split_shots.json next to the ckpt — RECOMPUTED the split "
                f"(val_n={val_n or val_tail}, split_seed={split_seed}). Valid only if the "
                f"cache is unchanged since training.", flush=True)
        shot = ",".join(val[-val_tail:])
        log(f"[eval] val_tail={val_tail} -> shots: {shot}", flush=True)

    # RAW-GT reference (user 2026-08-12). t0_start MUST come from the cache manifest: it fixes
    # the window origin, and reading raw frames on a different origin would shift them against
    # the codes (probe vs production caches differ by exactly 20 frames). If the manifest does
    # not record it, fall back to the historical 1.0 default.
    import os as _o2
    from . import spike as _spike
    use_raw_gt = _o2.environ.get("IGNITE_EVAL_RAW_GT", "1") != "0"
    data_dir = _o2.environ.get("IGNITE_DATA_DIR") or _spike.DEFAULT_DATA_DIR
    cache_t0 = 1.0
    _man = Path(cache_dir) / "_codec_manifest.json"
    if _man.exists():
        try:
            cache_t0 = float(json.loads(_man.read_text()).get("t0_start", 1.0))
        except Exception:
            pass
    if use_raw_gt:
        log(f"[eval] GROUND TRUTH = measured input (t0_start={cache_t0} from the cache "
            f"manifest); metrics therefore include codec reconstruction error and are NOT "
            f"comparable to decoded-GT runs. Set IGNITE_EVAL_RAW_GT=0 for the old behaviour.",
            flush=True)

    repo = Path.cwd()
    model, cfg, step = load_model(Path(ckpt), device)
    K0_req = int(k0) if k0 else cfg.k0_seed
    log(f"[eval] model loaded: d_model={cfg.d_model} depth={cfg.depth} heads={cfg.n_heads} "
        f"step={step} K0={K0_req} max_frames={cfg.max_frames} "
        f"frame_tokens={cfg.tokens_per_frame}", flush=True)

    from .train_dynamics import load_frozen_codecs as _lfc
    codecs = _lfc(list(EVAL_MODALITIES), repo=repo, tmpl=codec_tmpl)
    for _n, (c, _c2, _f) in codecs.items():
        c.to(device)
    log(f"[eval] loaded {len(codecs)} frozen codecs "
        f"(tmpl={codec_tmpl or 'manifest'}): {sorted(codecs)}", flush=True)

    shots_all = [s.strip() for s in str(shot).split(",") if s.strip()]
    # (a) multi-GCD parallel eval (user go 2026-08-09): under the srun rank wrapper each
    # rank evaluates shots_all[RANK::WORLD_SIZE]; rank 0 merges via part files (no process
    # group — an idle-rank NCCL watchdog cannot bite an embarrassingly parallel eval).
    import os as _os
    rank = int(_os.environ.get("RANK", "0"))
    world = int(_os.environ.get("WORLD_SIZE", "1"))
    shots = shots_all[rank::world]
    log(f"[eval] rank {rank}/{world}: {len(shots)}/{len(shots_all)} shots", flush=True)
    per_shot: Dict[str, Dict] = {}
    fig_paths = []
    for si, sh in enumerate(shots):
        cache = load_shot_cache(Path(cache_dir), sh)
        gen = torch.Generator(device=device).manual_seed(int(seed))
        # bf16 rollout is OPT-IN ONLY (IGNITE_EVAL_BF16=1) and NOT recommended: the A/B on
        # the probe N1000 ckpt (job 5218866 vs stored fp32, same shots/seed/T) measured a
        # SYSTEMATIC accuracy drop — all 14 modalities negative, mean -0.040, max -0.069.
        # Coarse bf16 logits perturb sampling + the confidence-unmask ordering, and 40
        # autoregressive frames compound it. Default = fp32; speed comes from multi-rank
        # sharding instead (fidelity-free 8x).
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda"
                                     and _os.environ.get("IGNITE_EVAL_BF16") == "1")):
            gt_codes, pred_codes, K0, F = rollout_shot(model, cfg, cache, K0_req,
                                                       temperature, gen, device,
                                                       actuator_mode=actuator_mode,
                                                       cache_dir=cache_dir)
            # ACTUATOR COUNTERFACTUAL: the effect size is the divergence from the SAME rollout
            # under real actuators, not the divergence from GT. Re-seed so the two runs share an
            # identical RNG stream — the rollout draws the same number of samples either way, so
            # any difference is attributable to the conditioning alone.
            base_codes = None
            if actuator_mode != "real":
                gen_b = torch.Generator(device=device).manual_seed(int(seed))
                _gt, base_codes, _K, _F = rollout_shot(model, cfg, cache, K0_req,
                                                       temperature, gen_b, device,
                                                       actuator_mode="real",
                                                       cache_dir=cache_dir)
        tok_acc = {n: float((pred_codes[n][K0:F] == gt_codes[n][K0:F]).float().mean())
                   for n in gt_codes}
        # persistence baseline in CODE space: fraction of tokens that simply do not change
        # from the last seed frame. Discrete codes at 50 ms are highly persistent, so raw
        # token accuracy is NOT interpretable without this (measured 2026-08-10: freezing
        # scores ~0.58 slow-TS on held-out shots) — report the SKILL, not the raw number.
        pers_acc = {n: float((gt_codes[n][K0:F] == gt_codes[n][K0 - 1:K0]).float().mean())
                    for n in gt_codes}
        # RE-RENDER WITHOUT RE-ROLLING-OUT (2026-08-12). The figure design is still in flux, and a
        # render iteration must not cost a fresh rollout — 80 generated frames x 10 MaskGIT decode
        # steps per shot, and by then dynamics_latest.pt has advanced several chain legs, so the
        # "new" figure would show a DIFFERENT model than the one it is being compared with.
        # Codes are the compact, decoder-independent state (~640 KB per shot per side), so archive
        # them beside the metrics: any later render decodes from these in seconds, pinned to
        # exactly this checkpoint. Ranks never collide — the filename carries shot and step.
        try:
            np.savez_compressed(
                Path(out_dir) / f"codes_{sh}_step{step}.npz",
                K0=np.int32(K0), F=np.int32(F), step=np.int32(step), shot=str(sh),
                temperature=np.float32(temperature),
                **{f"gt__{n}": v.cpu().numpy().astype(np.int32) for n, v in gt_codes.items()},
                **{f"pred__{n}": v.cpu().numpy().astype(np.int32) for n, v in pred_codes.items()})
        except Exception as e:                   # archiving must never take an eval down with it
            log(f"[eval] WARNING: could not archive codes for {sh}: {type(e).__name__}: {e}")
        raw = load_raw_gt(sh, codecs, F, data_dir, t0_start=cache_t0, log=log) \
            if use_raw_gt else None
        decoded = decode_all(codecs, gt_codes, pred_codes, K0, F, device, raw_gt=raw)
        if si == 0:
            self_check(decoded)
        entry = {"token_accuracy": tok_acc,
                 "token_accuracy_persistence": pers_acc,
                 "token_accuracy_skill": {n: tok_acc[n] - pers_acc[n] for n in tok_acc},
                 "nrmse": {n: d["nrmse"] for n, d in decoded.items()},
                 "nrmse_persistence": {n: d["nrmse_persistence"] for n, d in decoded.items()},
                 "nrmse_skill": {n: d["nrmse_skill"] for n, d in decoded.items()},
                 "K0": K0, "F": F}
        if base_codes is not None:
            # per-modality fraction of predicted-region tokens that CHANGED when the actuators
            # changed. 0.0 = the conditioning had literally no effect (the paired RNG makes that
            # an exact statement, not an approximation).
            entry["actuator_mode"] = actuator_mode
            entry["divergence_vs_real"] = {
                n: float((pred_codes[n][K0:F] != base_codes[n][K0:F]).float().mean())
                for n in pred_codes}
            entry["token_accuracy_real_actuators"] = {
                n: float((base_codes[n][K0:F] == gt_codes[n][K0:F]).float().mean())
                for n in pred_codes}
        if si == 0 or render_all:
            # t_origin MUST be the cache's window origin, not the WARMUP_S default: this cache
            # was built with t0_start=0.0, so defaulting to 1.0 shifted every absolute
            # shot-time label by +1 s (data correct, axis mislabelled) -- which silently
            # misaligns the figure against known shot physics. cache_t0 is already resolved
            # from _codec_manifest.json above and used for load_raw_gt; forward it here too.
            png, pdf = render_figure(decoded, sh, step, temperature, K0, F, Path(out_dir),
                                     t_origin=cache_t0)
            fig_paths.append(str(png))
            entry["figure"] = str(png)
        per_shot[sh] = entry
        log(f"[eval] shot {sh}: acc(mean)="
            f"{sum(tok_acc.values()) / max(len(tok_acc), 1):.3f} "
            f"({si + 1}/{len(shots)})", flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if world > 1:
        # rank part file (tmp+rename), then rank 0 collects all parts and merges.
        part = out / f"_eval_part_rank{rank}.json"
        tmp = out / f"_eval_part_rank{rank}.json.tmp"
        with open(tmp, "w") as f:
            json.dump(per_shot, f)
        tmp.replace(part)
        if rank != 0:
            log(f"[eval] rank {rank}: part written, exiting", flush=True)
            return {"per_shot": per_shot, "rank": rank}
        import time
        deadline = time.time() + 3600
        while time.time() < deadline:
            parts = sorted(out.glob("_eval_part_rank*.json"))
            if len(parts) >= world:
                break
            time.sleep(10)
        for pf in sorted(out.glob("_eval_part_rank*.json")):
            per_shot.update(json.loads(pf.read_text()))
            pf.unlink()
        missing = [s for s in shots_all if s not in per_shot]
        if missing:
            log(f"[eval] WARNING: no results from {len(missing)} shots: {missing}", flush=True)

    # aggregate means across shots (per modality)
    def _mean(field):
        # NAN-AWARE: with measured GT an ABSENT diagnostic yields NaN for that (shot, modality)
        # — a plain mean then poisons the whole modality (measured: 5/11 mean_nrmse came back
        # NaN because each was missing in at least one of 16 shots). Average the shots where
        # the modality was actually recorded; NaN only if it was recorded nowhere.
        keys = set().union(*(per_shot[s][field] for s in per_shot))
        out = {}
        for k in sorted(keys):
            vals = [per_shot[s][field][k] for s in per_shot if k in per_shot[s][field]]
            fin = [v for v in vals if v == v]                  # drop NaN
            out[k] = float(np.mean(fin)) if fin else float("nan")
        return out

    def _n_scored(field):
        """How many shots actually contributed to each modality's mean (provenance for _mean)."""
        keys = set().union(*(per_shot[s][field] for s in per_shot))
        return {k: int(sum(1 for s in per_shot
                           if k in per_shot[s][field] and per_shot[s][field][k] == per_shot[s][field][k]))
                for k in sorted(keys)}
    metrics = {"ckpt": str(ckpt), "step": step, "temperature": temperature,
               "seed": int(seed), "codec_tmpl": codec_tmpl, "n_shots": len(per_shot),
               "shots": sorted(per_shot), "mean_token_accuracy": _mean("token_accuracy"),
               "mean_nrmse": _mean("nrmse"),
               # skill vs the frozen-last-seed-frame rollout — the headline numbers.
               "mean_token_accuracy_persistence": _mean("token_accuracy_persistence"),
               "mean_token_accuracy_skill": _mean("token_accuracy_skill"),
               "mean_nrmse_persistence": _mean("nrmse_persistence"),
               "mean_nrmse_skill": _mean("nrmse_skill"),
               # shots contributing to each nRMSE mean — a modality present in only a few
               # shots has a far noisier mean, and that must be visible in the record.
               "n_shots_scored_nrmse": _n_scored("nrmse"),
               "per_shot": per_shot}
    with open(out / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log("[eval] predicted region — model vs PERSISTENCE (freeze last seed frame):", flush=True)
    log(f"  {'modality':24s} {'tok_acc':>8s} {'pers':>8s} {'skill':>8s} "
        f"{'nRMSE':>8s} {'pers':>8s} {'skill':>7s}", flush=True)
    for n, v in metrics["mean_token_accuracy"].items():
        nr = metrics["mean_nrmse"].get(n)
        log(f"  {n:24s} {v:8.3f} {metrics['mean_token_accuracy_persistence'][n]:8.3f} "
            f"{metrics['mean_token_accuracy_skill'][n]:+8.3f} "
            + (f"{nr:8.3f} {metrics['mean_nrmse_persistence'][n]:8.3f} "
               f"{metrics['mean_nrmse_skill'][n]:+7.3f}" if nr is not None else " " * 25),
            flush=True)
    log(f"[eval] metrics: {out / 'eval_metrics.json'}"
        + (f"\n[eval] figure: {fig_paths[0]}" if fig_paths else ""), flush=True)
    return metrics


def curate_core(cache_dir, out_path=None, k0: int = 20, n_frames: int = 60,
                thresh: float = 0.999, shots=None, require_all: bool = True, log=print):
    """Shots whose EVERY modality genuinely CHANGES over the eval window.

    A missing or parked diagnostic encodes to a frozen code sequence. That hands the model
    AND the persistence baseline a free 1.000 token accuracy, so any average over such shots
    overstates raw accuracy and understates skill (measured on the v2 probe: 9/16 eval shots
    had dead video). The standing rule is that eval/figure shots must have every modality
    present AND dynamic — this screens for exactly that, straight from the code cache.

    The statistic computed here is the SAME quantity ``decode_all`` reports as
    ``token_accuracy_persistence``: the fraction of tokens in the predicted region [k0, F)
    equal to the last seed frame (k0-1). 1.0 => frozen. No model or codec is needed.

    Returns ``(curated, report)`` where report maps shot -> {modality: persistence_accuracy}.
    """
    import json
    cache = Path(cache_dir)
    stems = sorted(p.stem for p in cache.glob("*.pt")) if shots is None else list(shots)
    curated, report, skipped_short = [], {}, 0
    for s in stems:
        try:
            d = torch.load(cache / f"{s}.pt", map_location="cpu")
        except Exception as e:
            log(f"[curate] skip {s}: {type(e).__name__}: {e}")
            continue
        codes = d["codes"]
        F = min(int(d.get("n_frames", 0)) or n_frames, n_frames)
        if F <= k0:                     # too short to have a predicted region at all
            skipped_short += 1
            continue
        per = {}
        for m, arr in codes.items():
            if arr.shape[0] < F:
                per[m] = float("nan")
                continue
            frozen = arr[k0 - 1:k0]                       # the last seed frame
            per[m] = float((arr[k0:F] == frozen).float().mean())
        report[s] = per
        vals = [v for v in per.values() if v == v]         # drop NaN (too-short modalities)
        alive = [m for m, v in per.items() if v == v and v < thresh]
        ok = (len(alive) == len(vals)) if require_all else bool(alive)
        if ok and len(vals) == len(per):
            curated.append(s)
    log(f"[curate] {len(curated)}/{len(report)} shots have ALL {len(next(iter(report.values()), {}))} "
        f"modalities dynamic over frames [{k0},{n_frames}) "
        f"(+{skipped_short} too short)")
    if report:
        n_mod = {}
        for s, per in report.items():
            for m, v in per.items():
                n_mod[m] = n_mod.get(m, 0) + (1 if (v == v and v >= thresh) else 0)
        worst = sorted(n_mod.items(), key=lambda kv: -kv[1])[:6]
        log("[curate] most frequently FROZEN modalities: "
            + ", ".join(f"{m} {c}/{len(report)}" for m, c in worst))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(out_path) + ".tmp")
        tmp.write_text(json.dumps({"cache_dir": str(cache), "k0": k0, "n_frames": n_frames,
                                   "threshold": thresh, "require_all": require_all,
                                   "curated": curated, "per_shot": report}, indent=1))
        tmp.replace(Path(out_path))
        log(f"[curate] -> {out_path}")
    return curated, report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("IGNITE Phase-B dynamics evaluation")
    p.add_argument("--ckpt", help="trained dynamics checkpoint (dynamics_latest.pt); "
                                  "not required with --curate")
    p.add_argument("--shot", default="200729",
                   help="shot id(s) to evaluate, comma-separated (figure: first shot only "
                        "unless --render_all)")
    p.add_argument("--cache_dir", default=_DEFAULT_CACHE, help="pre-encoded frame-code cache dir")
    p.add_argument("--out_dir", required=True, help="output dir for figure + eval_metrics.json")
    p.add_argument("--temperature", type=float, default=1.0, help="rollout sampling temperature")
    p.add_argument("--seed", type=int, default=0, help="rollout RNG seed (reproducible)")
    p.add_argument("--k0", type=int, default=0, help="seed frames (0 => use cfg.k0_seed)")
    p.add_argument("--codec_tmpl", default=None,
                   help="codec ckpt template tried before the frozen manifest — must match "
                        "the cache's _codec_manifest.json")
    p.add_argument("--render_all", action="store_true", help="render a figure for every shot")
    p.add_argument("--val_tail", type=int, default=0,
                   help=">0: evaluate the last N shots of the trainer's validation split "
                        "(overrides --shot)")
    p.add_argument("--val_n", type=int, default=0, help="validation split size (with --val_tail)")
    p.add_argument("--curate", action="store_true",
                   help="CURATION MODE (no model): scan the code cache and list shots whose "
                        "every modality actually CHANGES over the eval window. Frozen "
                        "diagnostics give model and persistence a free 1.000 and must not "
                        "enter an eval core or a figure.")
    p.add_argument("--curate_out", default=None,
                   help="where to write the curated list (default <out_dir>/eval_core.json)")
    p.add_argument("--curate_frames", type=int, default=60,
                   help="window length F for the dynamism test (default 60 = the 2 s eval)")
    p.add_argument("--curate_split", default=None,
                   help="a run's split_shots.json — screen only one of its partitions so "
                        "the eval core cannot accidentally contain training shots")
    p.add_argument("--curate_partition", default="val", choices=("train", "val", "test"),
                   help="which partition of --curate_split to screen (default val)")
    p.add_argument("--actuator_mode", default="real",
                   help="ACTUATOR COUNTERFACTUAL applied over the predicted region "
                        "[K0,F): real | zero | freeze | shuffle | donor:<shot> | "
                        "group_zero:<ech_power|pinj|beam_voltage|tinj|gas_flow|gas_raw|rmp>. "
                        "Anything but 'real' ALSO runs the real-actuator rollout with the "
                        "same seed and reports divergence_vs_real (the effect size). NOTE: "
                        "cached actuators are already per-shot z-scored, so gain/offset "
                        "edits would be no-ops — these modes are structural.")
    p.add_argument("--split_seed", type=int, default=0,
                   help="split seed — must match the training run's --split_seed")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.curate:
        shots = None
        if args.curate_split:                    # screen only a partition of a frozen split
            import json
            d = json.loads(Path(args.curate_split).read_text())
            shots = d[args.curate_partition]
            print(f"[curate] screening the '{args.curate_partition}' partition of "
                  f"{args.curate_split}: {len(shots)} shots", flush=True)
        curated, _ = curate_core(args.cache_dir,
                                 out_path=args.curate_out or (Path(args.out_dir) / "eval_core.json"),
                                 k0=args.k0 or 20, n_frames=args.curate_frames,
                                 shots=shots)
        return 0 if curated else 1
    if not args.ckpt:
        raise SystemExit("--ckpt is required (omit only with --curate)")
    return run(args.ckpt, args.shot, args.cache_dir, args.out_dir,
               temperature=args.temperature, seed=args.seed, k0=args.k0,
               codec_tmpl=args.codec_tmpl, render_all=args.render_all,
               val_tail=args.val_tail, val_n=args.val_n, split_seed=args.split_seed,
               actuator_mode=args.actuator_mode)


if __name__ == "__main__":
    main()
