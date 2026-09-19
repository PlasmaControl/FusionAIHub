"""Actuator-counterfactual case studies on the BAND-POWER (`bp*`) dynamics model.

The 14-modality model predicts `mhr` at or below a constant-token baseline at step 13500, so
AE and tearing-mode questions cannot be answered with it. The band-power family trains on
`mhr` + `co2` only -- the MHD channel -- which is why this exists.

Two pieces had to be combined, because neither existing script does both:
  * ``eval_dynamics`` has the actuator counterfactual (``apply_actuator_mode``) but can only
    decode learned FSQ codecs, which the band-power tokeniser does not have.
  * ``ignite_bandpower/bp_eval2.py`` knows how to invert band-power levels but has no
    counterfactual, and hardcodes a different repo checkout.

Here the rollout is driven directly through ``MaskGITDynamics.rollout``, the counterfactual is
applied to the cached actuators before the rollout (so the seed region is untouched and both
arms see identical context), and "decoding" is the explicit level -> representative log-power
map -- no learned decoder is involved, which is what makes band power directly readable.

Every non-real arm is paired against the real-actuator arm at the SAME seed, so the
difference is attributable to the actuator edit rather than to sampling. Band-restricted
traces are the point: a global spectrogram error averages a mode over 512 bins, while a
tearing mode occupies only the bottom ~8%.

Usage::

    python scripts/evaluation/ignite_bp_cases.py \
        --ckpt /lustre/orion/fus187/proj-shared/models/ignite_bp128/runs/bp128_d512L8/dynamics_latest.pt \
        --bp-root /lustre/orion/fus187/proj-shared/models/ignite_bp128 \
        --cache-dir data/outputs/ignite_bp_cache_ext \
        --shot 199597 --arms real,donor:199598,group_zero:pinj,group_zero:rmp \
        --out data/outputs/ignite_bp_cases/tm_199597
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))

from ignite_case_panels import decoded_space_skill  # noqa: E402
from tfm_eval.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402

FRAME_DT_S = 0.05
SPECTRO_FS_HZ, SPECTRO_N_FFT = 500e3, 1024
DF_HZ = SPECTRO_FS_HZ / SPECTRO_N_FFT      # 488.28 Hz per STFT bin (DC dropped)
N_LEV = 8
BANDS_HZ = [("TM 1-20 kHz", 1e3, 20e3), ("AE 50-250 kHz", 50e3, 250e3)]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bp-root", required=True, help="supplies the trained bin_edges*.npz")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--arms", default="real",
                   help="comma-separated actuator modes; 'real' is the reference arm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-predict", type=int, default=0, help="0 = the checkpoint's n_predict")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--modalities", default="mhr,co2")
    ap.add_argument("--data-dir", default="/lustre/orion/fus187/proj-shared/additional_data",
                   help="H5 dir, used ONLY to find where each diagnostic's record ends")
    return ap.parse_args()


def valid_frames(shot: str, mod: str, data_dir, n_frames: int) -> int:
    """How many frames a modality actually has DATA for.

    Diagnostics are fixed-length digitiser records -- mhr is 2**21+1 samples = 4.194 s -- so a
    5 s rollout window runs off the end of them. The cache still holds codes there because the
    encoder was handed padding, and those codes decode to plausible-looking values. Scoring or
    plotting them compares the model against fabricated ground truth, so metrics stop here.
    """
    import h5py
    p = Path(data_dir) / f"{shot}_processed.h5"
    if not p.exists():
        return n_frames
    try:
        with h5py.File(p, "r") as f:
            if mod not in f:
                return n_frames
            t_end = float(f[mod]["xdata"][-1])
    except Exception:
        return n_frames
    return max(0, min(n_frames, int(np.floor(t_end / FRAME_DT_S))))


def load_edges(bp_root: Path):
    p = next((f for f in (bp_root / "bin_edges_128.npz", bp_root / "bin_edges.npz")
              if f.exists()), None)
    if p is None:
        raise SystemExit(f"no bin_edges*.npz under {bp_root}")
    print(f"[bp] trained bin edges: {p}")
    return np.load(p)


def levels_to_value(edges, mod: str, tk: np.ndarray) -> np.ndarray:
    """Level index -> representative log-power. Interior bins take the midpoint of their two
    cut points; the two outer (unbounded) bins take the edge -/+ half the neighbouring width.
    Same construction as bp_eval2.levels_to_value so values are comparable to Peter's."""
    q = edges[mod]                                   # (N_LEV-1, D)
    D = q.shape[1]
    reps = np.zeros((N_LEV, D))
    reps[1:-1] = 0.5 * (q[:-1] + q[1:])
    w = np.maximum(q[1] - q[0], 1e-6)
    reps[0] = q[0] - 0.5 * w
    w = np.maximum(q[-1] - q[-2], 1e-6)
    reps[-1] = q[-1] + 0.5 * w
    return np.take_along_axis(reps, np.clip(tk, 0, N_LEV - 1), axis=0)


def band_slice(n_band: int, n_freq: int, f_lo: float, f_hi: float) -> tuple:
    """Which of the n_band equal-width frequency bands overlap [f_lo, f_hi]."""
    e = np.linspace(0, n_freq, n_band + 1).astype(int)
    lo_bin = max(0, int(round(f_lo / DF_HZ)) - 1)
    hi_bin = min(n_freq - 1, int(round(f_hi / DF_HZ)) - 1)
    idx = [i for i in range(n_band) if e[i + 1] > lo_bin and e[i] <= hi_bin]
    return (idx[0], idx[-1]) if idx else (0, 0)


def band_trace(edges, mod: str, codes: np.ndarray, n_ch: int, n_band: int,
               n_freq: int, f_lo: float, f_hi: float) -> np.ndarray:
    """(n_frames, C*n_band) levels -> per-frame mean log-power over the requested band."""
    vals = levels_to_value(edges, mod, codes)                       # (n_frames, C*n_band)
    v = vals.reshape(vals.shape[0], n_ch, n_band)
    b0, b1 = band_slice(n_band, n_freq, f_lo, f_hi)
    return v[:, :, b0:b1 + 1].mean(axis=(1, 2))


def main() -> int:
    from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig
    from tokamak_foundation_model.ignite.eval_dynamics import apply_actuator_mode
    from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
    from tokamak_foundation_model.ignite.train_dynamics import cache_modality_specs

    args = parse_args()
    set_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edges = load_edges(Path(args.bp_root))

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    specs = cache_modality_specs(cache)
    # Guard: the flat token frame is positional, so the cache layout must match what the
    # checkpoint was trained on exactly, or every modality is read from the wrong slice.
    ck_layout = {m[0]: (m[2], m[3]) for m in ck["modalities"]}
    cache_layout = {s.name: (s.n_tok, s.codebook_size) for s in specs}
    if ck_layout != cache_layout:
        raise SystemExit(f"layout mismatch\n  ckpt : {ck_layout}\n  cache: {cache_layout}")
    print(f"[bp] layout OK: {cache_layout}")

    cfg = DynamicsConfig(modalities=specs, d_model=ck["cfg_d_model"], depth=ck["cfg_depth"],
                         k0_seed=ck["cfg_k0"], n_predict=ck["cfg_n_predict"])
    cfg.grad_checkpointing = False
    if ck.get("cfg_n_heads"):
        cfg.n_heads = ck["cfg_n_heads"]
    model = MaskGITDynamics(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    K0 = cfg.k0_seed
    NP = args.n_predict or cfg.n_predict
    print(f"[bp] ckpt step {ck['step']}  d_model={cfg.d_model} depth={cfg.depth} "
          f"K0={K0} n_predict={NP}  dev={dev}")

    d = torch.load(cache / f"{args.shot}.pt", map_location="cpu", weights_only=False)
    F = min(int(d["n_frames"]), K0 + NP)
    NP = F - K0
    if NP <= 0:
        raise SystemExit(f"shot has {d['n_frames']} frames <= K0={K0}")
    seed = {s.name: d["codes"][s.name][:K0].long().unsqueeze(0).to(dev) for s in specs}
    act0 = d["actuators"].float()

    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
    # channel count per modality = n_tok / n_band; n_band is fixed by the trained edges, and
    # n_freq is the codec input width the bands were cut from.
    geom = {}
    for m in mods:
        n_tok = cache_layout[m][0]
        # D = C * n_band, so n_band follows from the codec's channel count; n_freq is the
        # codec input width the equal-width bands were cut from. Both come from the codec
        # cfg rather than being assumed, since a codec re-train could change either.
        cfgc = torch.load(Path("/lustre/orion/fus187/proj-shared/models/ignite_codecs_current")
                          / m / "codec_best.pt", map_location="cpu",
                          weights_only=False)["cfg"]
        n_ch, n_freq = int(cfgc.channels), int(cfgc.freq_bins)
        if n_tok % n_ch:
            raise SystemExit(f"{m}: n_tok {n_tok} not divisible by channels {n_ch}")
        geom[m] = (n_ch, n_tok // n_ch, n_freq)
        print(f"[bp] {m}: channels={n_ch} bands={n_tok // n_ch} tokens={n_tok} "
              f"n_freq={n_freq}")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    # Pre-flight the donors. Without this a missing donor is only discovered when its arm
    # runs -- after the model load and however many rollouts preceded it -- so a typo or an
    # unencoded shot costs a full GPU allocation and leaves an empty output dir.
    missing = [a.split(":", 1)[1] for a in arms
               if a.startswith("donor:") and not (cache / f"{a.split(':', 1)[1]}.pt").exists()]
    if missing:
        raise SystemExit(f"donor shot(s) not in {cache}: {missing}. Encode them first with "
                         f"scripts/evaluation/ignite_build_bp_cache.py, or drop those arms.")
    results, meta = {}, {}
    for arm in arms:
        act = apply_actuator_mode(act0.clone(), arm, K0, cache_dir=str(cache), F=F)
        t0 = time.time()
        with torch.no_grad():
            traj = model.rollout(seed, act[:F].unsqueeze(0).to(dev), n_predict=NP,
                                 temperature=args.temperature,
                                 generator=torch.Generator(device=dev).manual_seed(args.seed))
        results[arm] = {m: traj[m][0].cpu().numpy() for m in mods}
        print(f"[bp] arm {arm:24s} rollout {time.time() - t0:5.1f}s")
    ref_arm = "real" if "real" in results else arms[0]

    # token-space divergence vs the real arm: immune to any decode/GT convention
    for arm in arms:
        if arm == ref_arm:
            continue
        meta[arm] = {m: float((results[arm][m][K0:F] != results[ref_arm][m][K0:F]).mean())
                     for m in mods}
        print(f"[bp] divergence_vs_real {arm:24s} " +
              "  ".join(f"{m}={meta[arm][m]:.4f}" for m in mods))

    written, archive, kvalid = [], {}, {}
    for m in mods:
        n_ch, n_band, n_freq = geom[m]
        gt_codes = d["codes"][m][:F].numpy()
        kv = valid_frames(args.shot, m, args.data_dir, F)
        kvalid[m] = kv
        print(f"[bp] {m}: data-valid frames 0..{kv - 1} "
              f"(t < {kv * FRAME_DT_S:.2f} s); frames {kv}..{F - 1} are encoded non-data")
        archive[f"codes_gt__{m}"] = gt_codes
        for a in arms:
            archive[f"codes_pred__{a}__{m}"] = results[a][m][:F]
        for label, f_lo, f_hi in BANDS_HZ:
            gt = band_trace(edges, m, gt_codes, n_ch, n_band, n_freq, f_lo, f_hi)
            pers = np.concatenate([gt[:K0], np.repeat(gt[K0 - 1], F - K0)])
            series = {a: band_trace(edges, m, results[a][m][:F], n_ch, n_band, n_freq,
                                    f_lo, f_hi) for a in arms}
            archive[f"trace_gt__{m}__{label}"] = gt
            archive[f"trace_pers__{m}__{label}"] = pers
            for a in arms:
                archive[f"trace_pred__{a}__{m}__{label}"] = series[a]
            _panel(out, f"{m} · {label}", series, gt, pers, K0, kv, ref_arm,
                   f"{m} band power [log]", written)

    np.savez_compressed(out / "bp_cases_traces.npz", K0=np.int32(K0), F=np.int32(F),
                        **{k: np.asarray(v) for k, v in archive.items()})
    (out / "bp_cases_meta.json").write_text(json.dumps({
        "ckpt": args.ckpt, "step": int(ck["step"]), "shot": args.shot, "K0": K0, "F": F,
        "arms": arms, "reference_arm": ref_arm, "temperature": args.temperature,
        "seed": args.seed, "divergence_vs_real": meta, "geometry": geom,
        "data_valid_frames": kvalid,
        "note": "metrics are computed over [K0, data_valid_frames) only; beyond that the "
                "diagnostic's digitiser record has ended and cached codes encode padding",
    }, indent=1, default=str))
    print(f"\n[bp] {len(written)} figure file(s) + bp_cases_meta.json -> {out}")
    return 0


def _panel(out, tkey, series, gt, pers, K0, kvalid, ref_arm, ylab, written):
    t = (np.arange(len(gt)) + 0.5) * FRAME_DT_S
    t_split = K0 * FRAME_DT_S
    ref = series[ref_arm]
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.12})
    # Past the digitiser record there is no measurement, so the "ground truth" there is
    # padding pushed through the tokeniser. Shade it and exclude it from the metrics.
    if kvalid < len(gt):
        for a in (ax, axd):
            a.axvspan(kvalid * FRAME_DT_S, t[-1] + 0.5 * FRAME_DT_S,
                      color="0.85", alpha=0.55, lw=0, zorder=0)
        ax.annotate("no measured data\n(record ended)",
                    xy=(kvalid * FRAME_DT_S, ax.get_ylim()[0]), xytext=(4, 6),
                    textcoords="offset points", fontsize=7, color="#555555", va="bottom")
    ax.plot(t, gt, color="black", lw=1.1, label="ground truth (band power)", zorder=3)
    ax.plot(t, pers, color="0.55", lw=0.9, ls=":", label="persistence", zorder=1)
    ci = 0
    for arm, y in series.items():
        if arm == ref_arm:
            ax.plot(t, y, color="#0072B2", lw=1.1, ls="--", label="real actuators", zorder=4)
        else:
            ax.plot(t, y, color=OKABE_ITO[ci % len(OKABE_ITO)], lw=1.0, alpha=0.9, label=arm)
            ci += 1
    for a in (ax, axd):
        a.axvline(t_split, color="0.3", lw=0.8, alpha=0.6)
    # Metrics over the DATA-VALID predicted region only.
    nr, npers, sk = decoded_space_skill(gt[:kvalid], ref[:kvalid], pers[:kvalid], K0)
    ax.set_ylabel(ylab)
    ax.set_title(f"{tkey} — reference vs varied actuators\n"
                 f"band power over t = {K0 * FRAME_DT_S:.2f}–{kvalid * FRAME_DT_S:.2f} s: "
                 f"nRMSE {nr:.3f}  vs persistence {npers:.3f}  skill {sk:+.3f}",
                 loc="left", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    ci = 0
    for arm, y in series.items():
        if arm == ref_arm:
            continue
        axd.plot(t, y - ref, color=OKABE_ITO[ci % len(OKABE_ITO)], lw=1.0,
                 label=f"{arm} − real")
        ci += 1
    axd.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    axd.set_ylabel("Δ vs real")
    axd.set_xlabel("shot time [s]")
    if ci:
        axd.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
        span = max(float(np.nanmax(np.abs((series[a] - ref)[K0:kvalid])))
                   for a in series if a != ref_arm)
        axd.set_title(f"max |Δ| = {span:.4g} over the data-valid window  "
                      f"(flat ⇒ actuators had no effect)", loc="left", fontsize=8)
    stem = tkey.replace(" · ", "_").replace(" ", "").replace("/", "-")
    written += save_fig(fig, out / f"{stem}")
    plt.close(fig)
    print(f"[ok] {tkey}")


if __name__ == "__main__":
    sys.exit(main())
