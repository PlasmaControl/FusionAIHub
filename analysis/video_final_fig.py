"""FINAL tangtv VIDEO codec deliverable: metric validation, arm table, and the figure.

Companion to the slow-TS (``eval_runs/codec_recon_figs/slowts_FINAL_fullshot.png``) and
fast-TS (``analysis/fastts_env_final_fig.py``) deliverables. EVALUATION + RENDERING ONLY --
every checkpoint is read read-only and nothing is trained here.

Three modes, all on HELD-OUT shots:

  --validate   Score a set of REFERENCE predictors of KNOWN ordering with the same metric
               suite used to rank the arms. A metric that cannot order these is not allowed
               to order codecs (this exact check killed four 2-D spectrogram metrics on
               2026-09-03). Prints the table either way.

  --score      The arm table: nRMSE + the trivial baselines, patch_lattice_ratio, hf_ratio,
               std_ratio, utilization as a BIT RATE, and the forecastability margins, over
               >= 300 held-out windows.

  --figure     The single deliverable figure: real frames, GT vs reconstruction, dead-camera
               regions left as GAPS (never filled), % of frames with data per panel, and
               per-panel metrics so each panel is self-contained.

Windows / channels with no camera are MASKED out of every statistic (mean, std, RMSE,
correlation and each baseline's own mean), matching the slow-TS standard. This matters:
measured over all 8753 shots, tangtv_lower has NO live camera in 51.30% of shots and
tangtv_upper in 68.58%, and among the shots that do, a further 19.0% / 16.9% of channel-slots
are a zero slab.

nRMSE IS A FLOOR, NOT A RANKING KEY -- its exact minimiser is the conditional mean, so it
rewards blur and amplitude collapse. ``std_ratio`` (ideal EXACTLY 1.0) is printed beside it,
and ``patch_lattice_ratio`` is always read as a PAIR with ``hf_ratio``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# CPU scoring is dominated by the decoder's per-frame conv refinement head; give it every
# thread the caller allowed (torch defaults to a fraction of the box on a shared login node).
torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "16")))

from tokamak_foundation_model.ignite import gate
from tokamak_foundation_model.ignite.config import VideoCodecConfig
from tokamak_foundation_model.ignite.video_codec import VideoCodec
from tokamak_foundation_model.ignite.train_codec import VideoCodecPairDataset

DATA = "/lustre/orion/fus187/proj-shared/foundation_model"
PROD = "/lustre/orion/fus187/proj-shared/models/ignite_codecs_prod_noinorm"
LIVENESS = "/lustre/orion/fus187/proj-shared/foundation_model_meta/video_channel_liveness.pt"

# Held-out shots: the LAST 16 of the sorted shot list are the trainer's eval split for every
# arm (train_shots = all_shots[:-eval_n_shots]), so no arm has seen them.
HELD_OUT = [str(s) for s in range(204984, 205000)]


# ------------------------------------------------------------------------------------ #
# data
# ------------------------------------------------------------------------------------ #
def load_codec(ckpt_path: str, device="cpu", fsq_levels=None):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    if fsq_levels is not None:
        cfg.fsq_levels = list(fsq_levels)
    codec = VideoCodec(cfg).to(device)
    codec.load_state_dict(ck["codec"])
    codec.eval()
    return codec, cfg, ck


def walk_shot(shot: str, modality: str, cfg: VideoCodecConfig, limit: Optional[int] = None):
    """Consecutive-window walk of ONE shot. Returns ``(frames, mask, idx)``.

    ``_build_clip`` is called DIRECTLY so the degenerate-window re-draw is bypassed: a window
    with no data stays a GAP instead of being silently replaced by a different window (the
    same discipline the fast-TS figure uses). ``frames`` is (n, C, T, H, W) with NaN where the
    window could not be built; ``mask`` is (n, C, T) validity.
    """
    probe = VideoCodecConfig(**{**cfg.__dict__})
    probe.mask_missing = True                 # we always want the real mask here
    probe.require_live_channels = False
    ds = VideoCodecPairDataset(modality, [shot], probe, data_dir=DATA)
    n = len(ds) if limit is None else min(len(ds), limit)
    C, T, H, W = probe.channels, probe.frames, probe.height, probe.width
    fr = np.full((n, C, T, H, W), np.nan, dtype=np.float32)
    mk = np.zeros((n, C, T), dtype=np.float32)
    _ = ds[0]                                  # bind this worker's h5 handle
    for i in range(n):
        try:
            clip = ds._build_clip(i)
        except Exception:
            clip = None
        if clip is not None:
            fr[i] = clip[0].numpy()
            mk[i] = clip[1].numpy()
    return fr, mk, np.arange(n)


def held_out_pool(modality: str, cfg: VideoCodecConfig, shots: List[str],
                  per_shot: int = 24, require_live: bool = True):
    """Pool up to ``per_shot`` windows from each held-out shot. Returns (frames, mask, tags).

    Windows are sampled on an EVEN STRIDE across the whole shot (not the first N), so the pool
    spans ramp-up, flat-top and ramp-down rather than one plasma phase, and only the sampled
    windows are built (a full 219-window walk per shot is ~20x the I/O for no extra coverage).
    ``require_live`` keeps only windows in which BOTH of this divertor's cameras were
    recording, so the pool is a like-for-like comparison across arms.
    """
    F, M, tags = [], [], []
    probe = VideoCodecConfig(**{**cfg.__dict__})
    probe.mask_missing = True
    probe.require_live_channels = False
    for sh in shots:
        ds = VideoCodecPairDataset(modality, [sh], probe, data_dir=DATA)
        n = len(ds)
        if n == 0:
            continue
        _ = ds[0]
        # oversample the stride so dead windows can be skipped without losing coverage
        cand = np.unique(np.linspace(0, n - 1, min(n, per_shot * 3)).astype(int))
        got = 0
        for i in cand:
            if got >= per_shot:
                break
            try:
                clip = ds._build_clip(int(i))
            except Exception:
                clip = None
            if clip is None:
                continue
            fr, mk = clip[0].numpy(), clip[1].numpy()
            if not np.isfinite(fr).all():
                continue
            if require_live and mk.min() <= 0:
                continue
            if mk.sum() <= 0:
                continue
            F.append(fr); M.append(mk); tags.append((sh, int(i))); got += 1
    if not F:
        raise RuntimeError(f"no live held-out windows for {modality} in {shots}")
    return np.stack(F, 0), np.stack(M, 0), tags


# ------------------------------------------------------------------------------------ #
# codec run
# ------------------------------------------------------------------------------------ #
@torch.no_grad()
def reconstruct(codec, frames: np.ndarray, batch: int = 32, device="cpu"):
    """Return (recon, x_std, codes) as numpy, in the codec's STANDARDIZED frame space."""
    recs, stds, codes = [], [], []
    for i in range(0, frames.shape[0], batch):
        x = torch.from_numpy(frames[i:i + batch]).to(device)
        out = codec.forward(x)
        recs.append(out["recon"].cpu().numpy())
        stds.append(out["x_std"].cpu().numpy())
        codes.append(out["codes"].cpu().numpy())
    return (np.concatenate(recs, 0), np.concatenate(stds, 0), np.concatenate(codes, 0))


# ------------------------------------------------------------------------------------ #
# 1. METRIC VALIDATION against references of known ordering
# ------------------------------------------------------------------------------------ #
def _box_blur(x: np.ndarray, k: int) -> np.ndarray:
    """Separable box blur over (H, W) only -- the 'featureless blur' reference.

    Accumulated in place rather than np.stack-ed: a k=15 stack of a (48, 2, 5, 120, 360)
    float64 batch is 2.5 GB of temporaries per axis, which dominated the runtime.
    """
    pad = k // 2
    H, W = x.shape[-2], x.shape[-1]
    y = np.pad(x, ((0, 0), (0, 0), (0, 0), (pad, pad), (0, 0)), mode="edge")
    acc = np.zeros_like(x)
    for i in range(k):
        acc += y[..., i:i + H, :]
    acc /= k
    y = np.pad(acc, ((0, 0), (0, 0), (0, 0), (0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(x)
    for i in range(k):
        out += y[..., i:i + W]
    out /= k
    return out


def _inject_lattice(x: np.ndarray, patch_h: int, patch_w: int, amp: float) -> np.ndarray:
    """Add a TILED patch texture -- a synthetic checkerboard of known strength."""
    rng = np.random.default_rng(7)
    tile = rng.standard_normal((patch_h, patch_w))
    H, W = x.shape[-2], x.shape[-1]
    grid = np.tile(tile, (H // patch_h, W // patch_w))
    return x + amp * float(np.nanstd(x)) * grid


def validate_metrics(frames: np.ndarray, mask: np.ndarray, cfg: VideoCodecConfig) -> Dict:
    """Score reference predictors of KNOWN ordering. Returns the table + a pass/fail verdict."""
    t = frames.astype(np.float64)
    # standardize exactly as the codec does, so the numbers live in the same space
    mu = t.mean(axis=(2, 3, 4), keepdims=True)
    sd = np.maximum(t.std(axis=(2, 3, 4), keepdims=True), 1.0)
    t = (t - mu) / sd
    m5 = (mask > 0.5).astype(np.float64)[..., None, None]
    sdv = float(np.sqrt((((t - (t * m5).sum(axis=(2, 3, 4), keepdims=True)
                           / np.maximum(m5.sum(axis=(2, 3, 4), keepdims=True), 1)) * m5) ** 2).sum()
                        / max(1.0, np.broadcast_to(m5, t.shape).sum())))
    rng = np.random.default_rng(0)

    tmean = np.broadcast_to(gate._video_masked_mean(t, mask, 2), t.shape)
    wcmean = np.broadcast_to(gate._video_masked_mean(t, mask, (2, 3, 4)), t.shape)
    refs = {
        "self             (perfect)": t,
        "noise 5%         (good)": t + 0.05 * sdv * rng.standard_normal(t.shape),
        "shrunk x0.40     (amplitude collapse)": wcmean + 0.40 * (t - wcmean),
        "blur k=5         (featureless)": _box_blur(t, 5),
        "blur k=15        (very blurred)": _box_blur(t, 15),
        "tmean            (frozen video)": tmean,
        "wcmean           (constant = 1.0 anchor)": wcmean,
        "lattice+0.5      (checkerboard)": _inject_lattice(t, cfg.patch_h, cfg.patch_w, 0.5),
    }
    rows = {}
    for name, pred in refs.items():
        r = gate.full_video_metrics(pred, t, mask=mask)
        r.update(gate.video_patch_lattice(pred, cfg.patch_h, cfg.patch_w, mask=mask))
        rows[name] = r
    gt_lat = rows["self             (perfect)"]["patch_lattice_ratio"]

    def g(n, k):
        return rows[n][k]

    checks = [
        ("self nrmse == 0", abs(g("self             (perfect)", "video_nrmse")) < 1e-9),
        ("self corr == 1", abs(g("self             (perfect)", "video_corr") - 1) < 1e-9),
        ("self std_ratio == 1", abs(g("self             (perfect)", "video_std_ratio") - 1) < 1e-9),
        ("self hf_ratio == 1", abs(g("self             (perfect)", "video_hf_ratio") - 1) < 1e-9),
        ("wcmean nrmse == 1.0000 exactly",
         abs(g("wcmean           (constant = 1.0 anchor)", "video_nrmse") - 1) < 1e-9),
        ("wcmean std_ratio == 0",
         abs(g("wcmean           (constant = 1.0 anchor)", "video_std_ratio")) < 1e-9),
        ("shrunk std_ratio == 0.40 (exposes the collapse nRMSE hides)",
         abs(g("shrunk x0.40     (amplitude collapse)", "video_std_ratio") - 0.40) < 1e-6),
        ("shrunk nrmse < 1 (i.e. nRMSE alone would ACCEPT it)",
         g("shrunk x0.40     (amplitude collapse)", "video_nrmse") < 1.0),
        ("tmean std_ratio_t == 0 (frozen video detected)",
         abs(g("tmean            (frozen video)", "video_std_ratio_t")) < 1e-9),
        ("noise beats blur k=5 on nrmse",
         g("noise 5%         (good)", "video_nrmse") < g("blur k=5         (featureless)", "video_nrmse")),
        ("blur k=15 worse than blur k=5 on nrmse",
         g("blur k=15        (very blurred)", "video_nrmse") > g("blur k=5         (featureless)", "video_nrmse")),
        ("blur hf_ratio << 1 (smoothness is visible)",
         g("blur k=5         (featureless)", "video_hf_ratio") < 0.5),
        ("noise hf_ratio > 1 (added HF is visible)",
         g("noise 5%         (good)", "video_hf_ratio") > 1.0),
        ("GT lattice ratio near 1 (no checkerboard in real frames)", gt_lat < 2.0),
        ("injected lattice >> GT lattice",
         g("lattice+0.5      (checkerboard)", "patch_lattice_ratio") > 5 * gt_lat),
        ("blur DROPS the lattice for free (why it must be read with hf_ratio)",
         g("blur k=5         (featureless)", "patch_lattice_ratio") <= gt_lat * 1.5),
    ]
    return {"rows": rows, "checks": checks, "gt_lattice": gt_lat}


def print_validation(v: Dict) -> bool:
    hdr = (f"{'reference':44s} {'nrmse':>8s} {'nrmse_p':>8s} {'corr':>7s} {'std_r':>7s} "
           f"{'std_r_t':>8s} {'hf_ratio':>9s} {'lattice':>9s} {'lat_frac':>9s}")
    print(hdr); print("-" * len(hdr))
    for n, r in v["rows"].items():
        print(f"{n:44s} {r['video_nrmse']:8.4f} {r['video_nrmse_pooled']:8.4f} "
              f"{r['video_corr']:7.4f} {r['video_std_ratio']:7.4f} {r['video_std_ratio_t']:8.4f} "
              f"{r['video_hf_ratio']:9.4f} {r['patch_lattice_ratio']:9.3f} "
              f"{r['lattice_energy_frac']:9.4f}")
    print()
    ok = True
    for name, passed in v["checks"]:
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\nMETRIC SUITE {'VALIDATED' if ok else 'REJECTED'} "
          f"({sum(1 for _, p in v['checks'] if p)}/{len(v['checks'])} checks)")
    return ok


# ------------------------------------------------------------------------------------ #
# 2. ARM TABLE
# ------------------------------------------------------------------------------------ #
def score_arm(ckpt: str, frames: np.ndarray, mask: np.ndarray, modality: str,
              seq_frames: Optional[np.ndarray] = None,
              seq_mask: Optional[np.ndarray] = None, device="cpu") -> Dict:
    codec, cfg, ck = load_codec(ckpt, device=device)
    recon, x_std, codes = reconstruct(codec, frames, device=device)
    out = gate.full_video_metrics(recon, x_std, mask=mask)
    out.update({f"recon_{k}": v for k, v in
                gate.video_patch_lattice(recon, cfg.patch_h, cfg.patch_w, mask=mask).items()})
    out.update({f"gt_{k}": v for k, v in
                gate.video_patch_lattice(x_std, cfg.patch_h, cfg.patch_w, mask=mask).items()})
    out.update(gate.code_rate_bits(codes, cfg.codebook_size, n_tok=cfg.n_tok))
    ut = gate.utilization(codes, cfg=cfg)
    out["min_dim_entropy"] = ut["min_dim_entropy"]
    out["effective_codes"] = ut["effective_codes"]
    # forecastability needs a CONSECUTIVE-window sequence
    if seq_frames is not None:
        B, nw = seq_frames.shape[0], seq_frames.shape[1]
        flat = seq_frames.reshape(B * nw, *seq_frames.shape[2:])
        _, _, cs = reconstruct(codec, flat, device=device)
        cs = cs.reshape(B, nw, cfg.n_tok, cfg.fsq_dim)
        fc = gate.forecastability(cs)
        out.update({"margin_overall": fc["margin_overall"],
                    "margin_transition": fc["margin_transition"],
                    "margin_stable": fc["margin_stable"],
                    "n_transition": fc["n_transition"], "n_stable": fc["n_stable"]})
    out["codebook_size"] = cfg.codebook_size
    out["fsq_levels"] = list(cfg.fsq_levels)
    out["n_tok"] = cfg.n_tok
    out["step"] = ck.get("step")
    out["refine_dilated"] = bool(getattr(cfg, "refine_dilated", False))
    return out, recon, x_std, codes, cfg


def seq_pool(modality: str, cfg: VideoCodecConfig, shots: List[str],
             n_windows: int = 8, per_shot: int = 2, any_live: bool = False):
    """(B, n_windows, C, T, H, W) blocks of CONSECUTIVE live windows, for forecastability."""
    blocks, bmask = [], []
    probe = VideoCodecConfig(**{**cfg.__dict__})
    probe.mask_missing = True
    probe.require_live_channels = False
    for sh in shots:
        ds = VideoCodecPairDataset(modality, [sh], probe, data_dir=DATA)
        n = len(ds)
        if n < n_windows:
            continue
        _ = ds[0]
        starts = np.linspace(0, n - n_windows, min(per_shot * 3, max(1, n - n_windows + 1)))
        got = 0
        for s0 in np.unique(starts.astype(int)):
            if got >= per_shot:
                break
            blk, msk, ok = [], [], True
            for k in range(n_windows):
                try:
                    clip = ds._build_clip(int(s0) + k)
                except Exception:
                    clip = None
                if clip is None or not torch.isfinite(clip[0]).all():
                    ok = False
                    break
                if clip[1].sum() <= 0 or ((not any_live) and clip[1].min() <= 0):
                    ok = False
                    break
                blk.append(clip[0].numpy()); msk.append(clip[1].numpy())
            if ok:
                blocks.append(np.stack(blk, 0)); bmask.append(np.stack(msk, 0)); got += 1
    if not blocks:
        raise RuntimeError("no consecutive live blocks found")
    return np.stack(blocks, 0), np.stack(bmask, 0)


# ------------------------------------------------------------------------------------ #
# 3. THE FIGURE
# ------------------------------------------------------------------------------------ #
def make_figure(out_png: str, panels: List[Dict], title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    fig = plt.figure(figsize=(22, 3.9 * n))
    gs = fig.add_gridspec(n, 1, hspace=0.45)
    for pi, P in enumerate(panels):
        ncol = len(P["frames"]) + 1
        # Row 0 is a text-only strip carrying the whole panel caption, so the per-frame time
        # labels and the caption can never overlap (they did when the caption was the trace
        # axis title).
        sub = gs[pi].subgridspec(3, ncol, hspace=0.14, wspace=0.06,
                                 height_ratios=[0.26, 1, 1],
                                 width_ratios=[1] * len(P["frames"]) + [1.9])
        cap = fig.add_subplot(sub[0, :])
        cap.axis("off")
        cap.text(0.0, 0.55, P["title"], fontsize=8.5, ha="left", va="center",
                 family="monospace")
        vmin, vmax = P["vlim"]
        for k, (lbl, gtf, rcf, live) in enumerate(P["frames"]):
            for row, img, tag in ((1, gtf, "GT"), (2, rcf, "recon")):
                ax = fig.add_subplot(sub[row, k])
                if live:
                    ax.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax,
                              aspect="auto", interpolation="nearest")
                else:
                    # DEAD CAMERA -> shown as a GAP, never filled with a reconstruction.
                    ax.set_facecolor("0.85")
                    ax.text(0.5, 0.5, "NO DATA", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="0.3")
                ax.set_xticks([]); ax.set_yticks([])
                if row == 1:
                    ax.set_title(lbl, fontsize=8, pad=2)
                if k == 0:
                    ax.set_ylabel(tag, fontsize=9)
        ax = fig.add_subplot(sub[1:, -1])
        tr = P["trace"]
        for a, b in tr["gaps"]:
            ax.axvspan(a, b, color="0.85", zorder=0)
        ax.axhline(1.0, color="0.4", lw=0.8, ls=":",
                   label="1.0 = constant predictor (nRMSE) / correct amplitude (std ratio)")
        ax.plot(tr["t"], tr["err"], color="steelblue", lw=1.0,
                label="per-window nRMSE  (0 = perfect)")
        ax.plot(tr["t"], tr["rc"], color="crimson", lw=1.0, ls="--",
                label="per-window std(recon)/std(GT)  (ideal 1.0)")
        finite = np.isfinite(tr["err"]) | np.isfinite(tr["rc"])
        if finite.any():
            hi = float(np.nanmax(np.concatenate([tr["err"], tr["rc"]])))
            ax.set_ylim(0.0, max(1.35, min(3.0, hi * 1.15)))
        else:
            ax.set_ylim(0.0, 1.35)
            ax.text(0.5, 0.5, "camera off for the ENTIRE shot -- nothing reconstructed",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9, color="0.3")
        ax.set_xlabel("time (s)   |   grey = NO CAMERA DATA (gap, never filled)", fontsize=8)
        ax.set_ylabel("dimensionless (1.0 reference)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.92)
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print("wrote", out_png)


# ------------------------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "score", "figure"], required=True)
    ap.add_argument("--modality", default="tangtv_lower")
    ap.add_argument("--arms", default="", help="name=ckpt,name=ckpt,...")
    ap.add_argument("--shots", default=",".join(HELD_OUT))
    ap.add_argument("--per_shot", type=int, default=24)
    ap.add_argument("--any_live", action="store_true",
                    help="keep windows where only SOME cameras were recording (the dead ones "
                         "are masked out of every statistic). Needed for tangtv_upper, whose "
                         "held-out shots 204984-204999 have ch4 live and ch6 off.")
    ap.add_argument("--fig_shot", default=None)
    ap.add_argument("--out", default="eval_runs/codec_recon_figs/video_FINAL.png")
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--title", default="IGNITE tangtv VIDEO codec - GT vs reconstruction")
    args = ap.parse_args()

    shots = [s for s in args.shots.split(",") if s]
    base_cfg = VideoCodecConfig(channels=2,
                                divertor="lower" if args.modality.endswith("lower") else "upper")

    if args.mode == "validate":
        fr, mk, tags = held_out_pool(args.modality, base_cfg, shots, per_shot=args.per_shot,
                                     require_live=not args.any_live)
        print(f"held-out windows: {fr.shape[0]} from {len(set(t[0] for t in tags))} shots "
              f"({args.modality}), mean mask occupancy {mk.mean():.4f}\n")
        v = validate_metrics(fr, mk, base_cfg)
        ok = print_validation(v)
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            json.dump({"rows": {k: {kk: (None if isinstance(vv, float) and math.isnan(vv)
                                         else vv) for kk, vv in r.items()}
                                for k, r in v["rows"].items()},
                       "checks": [[n, bool(p)] for n, p in v["checks"]],
                       "n_windows": int(fr.shape[0]), "modality": args.modality},
                      open(args.json, "w"), indent=1)
        raise SystemExit(0 if ok else 1)

    if args.mode == "score":
        fr, mk, tags = held_out_pool(args.modality, base_cfg, shots, per_shot=args.per_shot,
                                     require_live=not args.any_live)
        sf, sm = seq_pool(args.modality, base_cfg, shots, any_live=args.any_live)
        print(f"held-out windows: {fr.shape[0]} from {len(set(t[0] for t in tags))} shots; "
              f"forecast blocks: {sf.shape[0]} x {sf.shape[1]} windows\n")
        # standardized GT + trivial baselines (computed ONCE; identical for every arm)
        t = fr.astype(np.float64)
        mu = t.mean(axis=(2, 3, 4), keepdims=True)
        sd = np.maximum(t.std(axis=(2, 3, 4), keepdims=True), 1.0)
        t = (t - mu) / sd
        base = gate.trivial_video_baselines(t, mask=mk)
        gt_lat = gate.video_patch_lattice(t, base_cfg.patch_h, base_cfg.patch_w, mask=mk)
        results = {"_baselines": base, "_gt_lattice": gt_lat,
                   "_n_windows": int(fr.shape[0]),
                   "_shots": sorted(set(x[0] for x in tags)), "arms": {}}
        for spec in [a for a in args.arms.split(",") if a]:
            name, ck = spec.split("=", 1)
            try:
                r, *_ = score_arm(ck, fr, mk, args.modality, seq_frames=sf, device=args.device)
            except Exception as e:                                    # noqa: BLE001
                print(f"  {name}: FAILED ({type(e).__name__}: {e})")
                continue
            results["arms"][name] = r
            print(f"scored {name}")
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            json.dump(results, open(args.json, "w"), indent=1, default=float)
            print("wrote", args.json)
        print_arm_table(results)
        return

    if args.mode == "figure":
        specs = [a for a in args.arms.split(",") if a]
        panels, headers = [], []
        for spec in specs:
            # "label=ckpt=modality=shot=camera"
            parts = spec.split("=")
            label, ck, mod, shot, cam = parts[0], parts[1], parts[2], parts[3], int(parts[4])
            panels.append(build_panel(label, ck, mod, shot, cam, device=args.device))
            headers.append(panels[-1]["title"])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        make_figure(args.out, panels, args.title)
        for h in headers:
            print(h)
        return

    raise SystemExit(f"unknown mode {args.mode}")


# ------------------------------------------------------------------------------------ #
def build_panel(label: str, ckpt: str, modality: str, shot: str, cam: int,
                n_frames: int = 6, device="cpu") -> Dict:
    """One figure panel: representative FRAMES + a full-shot per-frame error trace.

    Missing windows are GAPS: they are never reconstructed, never interpolated, and the
    trace shades them. The panel title carries every number needed to read it alone.
    """
    codec, cfg, ck = load_codec(ckpt, device=device)
    fr, mk, _ = walk_shot(shot, modality, cfg)
    n = fr.shape[0]
    live = (mk[:, cam].min(axis=1) > 0) & np.isfinite(fr[:, cam]).all(axis=(1, 2, 3))
    idx = np.where(live)[0]
    frac = float(live.mean())

    recon = np.full_like(fr, np.nan)
    xstd = np.full_like(fr, np.nan)
    if idx.size:
        r, xs, codes = reconstruct(codec, np.nan_to_num(fr[idx]), device=device)
        recon[idx] = r
        xstd[idx] = xs
    else:
        codes = np.zeros((0, cfg.n_tok, cfg.fsq_dim), dtype=np.int64)

    # Per-window trace over the WHOLE shot (window index -> time).
    #
    # NEITHER the frame mean NOR the raw contrast: `VideoCodec.standardize_input` z-scores each
    # window per (batch, channel) over (T, H, W), so the GT window mean is EXACTLY 0 and the GT
    # window std EXACTLY 1 by construction -- both are flat lines that say nothing. The two
    # quantities that DO vary shot-wide are the two the arm table ranks on, and both are
    # dimensionless with the SAME 1.0 reference, so they share one axis:
    #   nRMSE      per window, RMSE / std(GT). 0 = perfect, 1.0 = predicting the constant mean.
    #   std_ratio  per window, std(recon) / std(GT). IDEAL EXACTLY 1.0; < 1 is the amplitude
    #              collapse that nRMSE rewards and correlation cannot see.
    T = cfg.frames
    t_axis = 1.0 + np.arange(n) * 0.05
    err = np.full(n, np.nan)
    sratio = np.full(n, np.nan)
    for i in idx:
        a = xstd[i, cam].ravel(); b = recon[i, cam].ravel()
        sd = a.std()
        if sd > 1e-8:
            err[i] = np.sqrt(((b - a) ** 2).mean()) / sd
            sratio[i] = b.std() / sd
    gt_mean, rc_mean = err, sratio
    gaps = []
    d = np.where(~live)[0]
    if d.size:
        brk = np.where(np.diff(d) > 1)[0]
        for grp in np.split(d, brk + 1):
            gaps.append((t_axis[grp[0]] - 0.025, t_axis[grp[-1]] + 0.025))

    # metrics for THIS panel (masked, this camera only)
    if idx.size:
        sub_x = xstd[idx][:, cam:cam + 1]
        sub_r = recon[idx][:, cam:cam + 1]
        sub_m = mk[idx][:, cam:cam + 1]
        M = gate.full_video_metrics(sub_r, sub_x, mask=sub_m)
        L = gate.video_patch_lattice(sub_r, cfg.patch_h, cfg.patch_w, mask=sub_m)
        Lg = gate.video_patch_lattice(sub_x, cfg.patch_h, cfg.patch_w, mask=sub_m)
        R = gate.code_rate_bits(codes, cfg.codebook_size, n_tok=cfg.n_tok)
        title = (f"{label} (step {ck.get('step')}) | {modality} cam{cam} | shot {shot} | "
                 f"{100*frac:.0f}% of frames have "
                 f"DATA (gaps shaded, never filled) | nRMSE {M['video_nrmse']:.3f} "
                 f"(const=1.000) | amp std_ratio {M['video_std_ratio']:.2f} "
                 f"(temporal {M['video_std_ratio_t']:.2f}) | corr {M['video_corr']:.3f} | "
                 f"hf {M['video_hf_ratio']:.2f} | lattice {L['patch_lattice_ratio']:.2f} "
                 f"(GT {Lg['patch_lattice_ratio']:.2f}) | codes {R['n_distinct_codes']}/"
                 f"{cfg.codebook_size} rate {100*R['rate_positional']:.0f}%")
    else:
        title = f"{label} | {modality} cam{cam} | shot {shot} | NO DATA (camera off all shot)"

    # representative frames spread across the shot, plus a dead region if there is one
    picks = []
    if idx.size:
        want = list(np.linspace(0, idx.size - 1, min(n_frames - 1, idx.size)).astype(int))
        picks = [(int(idx[w]), True) for w in want]
    if d.size:
        picks.append((int(d[d.size // 2]), False))
    if not picks:
        picks = [(0, False)]
    picks = sorted(set(picks))[:n_frames]

    vals = xstd[idx][:, cam] if idx.size else np.zeros((1, 1, 1, 1))
    vlim = (float(np.nanpercentile(vals, 1)), float(np.nanpercentile(vals, 99)))
    frames = []
    for wi, ok in picks:
        lbl = f"t={t_axis[wi]:.2f}s" + ("" if ok else "  (camera off)")
        gtf = xstd[wi, cam, T // 2] if ok else None
        rcf = recon[wi, cam, T // 2] if ok else None
        frames.append((lbl, gtf, rcf, ok))
    return {"frames": frames, "vlim": vlim, "title": title,
            "trace": {"t": t_axis, "gt": gt_mean, "rc": rc_mean, "err": err, "gaps": gaps}}


def print_arm_table(res: Dict):
    b = res["_baselines"]
    print(f"\nHELD-OUT WINDOWS: {res['_n_windows']}  shots: {len(res['_shots'])}")
    print("\nTRIVIAL BASELINES (same mask, same units, same function)")
    hdr = f"{'baseline':10s} {'nrmse':>8s} {'nrmse_p':>8s} {'corr':>7s} {'std_r':>7s} {'std_r_t':>8s} {'hf':>7s}"
    print(hdr); print("-" * len(hdr))
    for n in ("self", "tmean", "cmean", "wcmean"):
        print(f"{n:10s} {b[f'base_{n}_video_nrmse']:8.4f} {b[f'base_{n}_video_nrmse_pooled']:8.4f} "
              f"{b[f'base_{n}_video_corr']:7.4f} {b[f'base_{n}_video_std_ratio']:7.4f} "
              f"{b[f'base_{n}_video_std_ratio_t']:8.4f} {b[f'base_{n}_video_hf_ratio']:7.4f}")
    print(f"\nGROUND-TRUTH patch_lattice_ratio = {res['_gt_lattice']['patch_lattice_ratio']:.3f} "
          f"(1.0 = no lattice), lattice_energy_frac = "
          f"{res['_gt_lattice']['lattice_energy_frac']:.4f}")
    print("\nARMS")
    hdr = (f"{'arm':12s} {'cb':>6s} {'nrmse':>7s} {'nrmse_p':>8s} {'corr':>7s} {'std_r':>7s} "
           f"{'std_r_t':>8s} {'hf':>7s} {'lattice':>8s} {'rate_pos':>9s} {'rate_pool':>9s} "
           f"{'codes':>6s} {'m_over':>8s} {'m_trans':>8s} {'n_tr':>7s}")
    print(hdr); print("-" * len(hdr))
    for n, r in res["arms"].items():
        print(f"{n:12s} {r['codebook_size']:6d} {r['video_nrmse']:7.4f} "
              f"{r['video_nrmse_pooled']:8.4f} {r['video_corr']:7.4f} "
              f"{r['video_std_ratio']:7.4f} {r['video_std_ratio_t']:8.4f} "
              f"{r['video_hf_ratio']:7.4f} {r['recon_patch_lattice_ratio']:8.3f} "
              f"{r['rate_positional']:9.4f} {r['rate_pooled']:9.4f} "
              f"{r['n_distinct_codes']:6d} {r.get('margin_overall', float('nan')):8.4f} "
              f"{r.get('margin_transition', float('nan')):8.4f} "
              f"{int(r.get('n_transition', 0)):7d}")
    # Ceilings are PER ARM: they depend on log2(codebook_size), so a 64k arm and a 1k arm on
    # the same eval set have different ceilings and their raw rates are NOT comparable.
    if res["arms"]:
        print("\nBIT BUDGET per arm (an entropy estimated from M samples cannot exceed log2 M,"
              " so quote a rate as a FRACTION OF ITS CEILING)")
        hdr = (f"{'arm':12s} {'cb':>6s} {'bits/frame':>11s} {'delivered':>10s} {'rate_pos':>9s} "
               f"{'ceil_pos':>9s} {'%of_ceil':>9s} {'rate_pool':>10s} {'ceil_pool':>10s} "
               f"{'%of_ceil':>9s}")
        print(hdr); print("-" * len(hdr))
        for n, r in res["arms"].items():
            cp, cl = r.get("rate_positional_ceiling", 1.0), r.get("rate_pooled_ceiling", 1.0)
            print(f"{n:12s} {r['codebook_size']:6d} {r['bits_available_per_frame']:11.1f} "
                  f"{r['bits_delivered_per_frame_positional']:10.1f} {r['rate_positional']:9.4f} "
                  f"{cp:9.4f} {100*r['rate_positional']/max(cp,1e-9):8.1f}% "
                  f"{r['rate_pooled']:10.4f} {cl:10.4f} "
                  f"{100*r['rate_pooled']/max(cl,1e-9):8.1f}%")


if __name__ == "__main__":
    main()
