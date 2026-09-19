"""Eyeball panels for the IGNITE control-reference case studies.

``eval_dynamics`` renders 11 modalities, but its figure code only has branches for
the spectro / video / slowts families. Two modalities the dynamics model DOES
predict are therefore never drawn:

    filterscopes  (fastts) -- D-alpha, i.e. THE ELM observable
    cer_rot       (slowts) -- rotation, i.e. the RMP/TM magnetic-braking observable

Rather than edit that figure, this decodes from the ``codes_<shot>_step<N>.npz``
archives eval_dynamics already writes beside its metrics (it archives codes
precisely so a re-render costs seconds instead of a fresh 80-frame rollout). So
this never re-runs the model and is pinned to exactly the checkpoint that rolled
those codes out.

One figure per modality. Every counterfactual arm is overlaid on the SAME axes as
the real-actuator arm and the measured ground truth, because the question is
simply "do these line up, and does varying the actuators move the answer":

    ground truth        solid black
    real actuators      dashed blue
    counterfactuals     Okabe-Ito colours
    persistence         dotted grey (freeze the last seed frame -- the do-nothing rollout)

A vertical rule marks K0, where real context ends and the rollout takes over.
The lower panel of each figure is the counterfactual MINUS the real-actuator arm,
which is the panel that matters: this model family has previously shown actuator
responses small enough to be invisible in an overlay, and a difference trace on
its own scale distinguishes "no response" from "response too small to see".

Usage::

    python scripts/evaluation/ignite_case_panels.py \
        --run real=data/outputs/ignite_cases/elm_190735_real \
        --run donor-190736=data/outputs/ignite_cases/elm_190735_donor-190736 \
        --run rmp-off=data/outputs/ignite_cases/elm_190735_group_zero-rmp \
        --modalities filterscopes,cer_rot --out data/outputs/ignite_cases/figures/elm_190735
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))

from tfm_eval.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402

DEFAULT_CODEC_TMPL = (
    "/lustre/orion/fus187/proj-shared/models/ignite_codecs_current/{m}/codec_best.pt"
)
FRAME_DT_S = 0.05  # IGNITE frame = one 50 ms chunk

# Spectrogram frequency axis: the loader STFTs at 500 kHz with n_fft=1024 and drops the DC
# bin, giving 512 bins from 488 Hz to 250 kHz. Bin i (0-based) is centred at (i+1)*df.
SPECTRO_FS_HZ = 500e3
SPECTRO_N_FFT = 1024
DF_HZ = SPECTRO_FS_HZ / SPECTRO_N_FFT          # 488.28 Hz per bin

# Band-restricted readouts. A global nRMSE over all 512 bins is dominated by the broadband
# noise floor, so a mode occupying a narrow band can move a lot while the global number says
# nothing. These are the bands the phenomena actually live in.
BANDS = {
    "mhr": [("TM 1-20 kHz", 1e3, 20e3), ("AE 50-250 kHz", 50e3, 250e3)],
    "ece": [("TM 1-20 kHz", 1e3, 20e3), ("AE 50-250 kHz", 50e3, 250e3)],
    "bes": [("TM 1-20 kHz", 1e3, 20e3)],
    "co2": [("TM 1-20 kHz", 1e3, 20e3)],
}

# Display metadata per modality: (nice label, y-axis label).
META = {
    "filterscopes": ("D-alpha (filterscopes)", "D-alpha [norm.]"),
    "cer_rot": ("Rotation (CER)", "rotation [norm.]"),
    "cer_ti": ("Ion temperature (CER)", "Ti [norm.]"),
    "ts_core_density": ("Core density (Thomson)", "ne [norm.]"),
    "mse": ("MSE", "MSE [norm.]"),
    "mhr": ("MHD array (mhr)", "band power [norm.]"),
    "ece": ("ECE", "band power [norm.]"),
    "bes": ("BES", "band power [norm.]"),
    "co2": ("CO2 interferometer", "band power [norm.]"),
}


def band_bins(f_lo: float, f_hi: float, n_bins: int) -> tuple:
    """Inclusive bin index range covering [f_lo, f_hi] on the DC-dropped STFT axis."""
    lo = max(0, int(round(f_lo / DF_HZ)) - 1)
    hi = min(n_bins - 1, int(round(f_hi / DF_HZ)) - 1)
    return lo, max(lo, hi)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                    help="repeatable; DIR is an eval_dynamics --out_dir. The run "
                         "labelled 'real' (or the first one given) is the reference arm.")
    ap.add_argument("--modalities", default="filterscopes,cer_rot")
    ap.add_argument("--codec-tmpl", default=DEFAULT_CODEC_TMPL)
    ap.add_argument("--out", required=True, help="output figure stem directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def find_codes(run_dir: Path) -> Path:
    """The single codes_*.npz an eval run archived."""
    hits = sorted(run_dir.glob("codes_*_step*.npz"))
    if not hits:
        raise FileNotFoundError(f"no codes_*.npz in {run_dir} — did the eval finish?")
    return hits[-1]


def load_codec(name: str, tmpl: str, device):
    """Frozen Phase-A codec for one modality, via the same resolver eval_dynamics uses."""
    from tokamak_foundation_model.ignite.train_dynamics import (
        FROZEN_CODEC_CKPTS, _load_codec, resolve_codec_path,
    )
    rel = resolve_codec_path(name, Path.cwd(), tmpl)
    if rel is None:
        raise FileNotFoundError(
            f"codec for {name!r} did not resolve. FROZEN_CODEC_CKPTS holds repo-relative "
            f"paths that only exist in Peter's tree; pass an absolute --codec-tmpl."
        )
    fam = FROZEN_CODEC_CKPTS[name][0]
    codec, cfg = _load_codec(fam, Path.cwd() / rel)
    return codec.to(device).eval(), fam


@torch.no_grad()
def decode(codec, codes: np.ndarray, device) -> np.ndarray:
    """(F, n_tok) int codes -> decoded window, shape preserved.

    Spectrograms come back (F, C, n_freq, n_time); time series (F, C, T) or (F, C).
    """
    from tokamak_foundation_model.ignite.eval_dynamics import decode_flat_chunked
    arr = np.asarray(decode_flat_chunked(codec, torch.from_numpy(codes.astype(np.int64)),
                                         device))
    if arr.ndim == 2:      # (F, C) -> one sample per frame
        arr = arr[:, :, None]
    return arr


def traces(name: str, arr: np.ndarray) -> dict:
    """Decoded array -> {trace_key: per-FRAME trace of length F}.

    Everything is reduced to one value per 50 ms frame, because that is the granularity at
    which the dynamics model predicts -- sub-frame structure inside a codec window is not
    something it steps independently, so plotting it only adds STFT noise and buries the
    envelope. (It also makes the persistence baseline the flat line it actually is.)

    Spectro: mean over channels and over each band's frequency bins -> that band's power.
    D-alpha: additionally a per-frame BURST AMPLITUDE (max - median within the frame), since
    for ELMs the spikes are the signal and a frame mean would wash them out.
    """
    out = {}
    with np.errstate(invalid="ignore"):
        if arr.ndim == 4:                                   # (F, C, n_freq, n_time)
            n_bins = arr.shape[2]
            for label, f_lo, f_hi in BANDS.get(name, [("full band", DF_HZ, 250e3)]):
                lo, hi = band_bins(f_lo, f_hi, n_bins)
                out[f"{name} · {label}"] = np.nanmean(arr[:, :, lo:hi + 1, :], axis=(1, 2, 3))
        else:                                               # (F, C, T)
            ch = np.nanmean(arr, axis=1)                    # (F, T)
            out[name] = np.nanmean(ch, axis=1)
            if name == "filterscopes" and ch.shape[1] > 1:
                out[f"{name} · burst amplitude"] = (np.nanmax(ch, axis=1)
                                                    - np.nanmedian(ch, axis=1))
    return out


def decoded_space_skill(gt, pred, pers, K0) -> tuple:
    """(nrmse, nrmse_persistence, skill) over the PREDICTED region, in DECODED space.

    This is the decoded-GT comparison: gt and pred both come from codes pushed through the
    SAME frozen codec, so codec reconstruction error cancels and what remains is the DYNAMICS
    model's error. eval_dynamics' own metrics default to MEASURED ground truth instead
    (IGNITE_EVAL_RAW_GT=1 since 2026-08-12), which folds the codec's error back in and is,
    by its own log message, "NOT comparable to decoded-GT runs".

    skill > 0 means the rollout beats freezing the last real frame.
    """
    g, p, q = gt[K0:], pred[K0:], pers[K0:]
    ok = np.isfinite(g) & np.isfinite(p) & np.isfinite(q)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    g, p, q = g[ok], p[ok], q[ok]
    den = g.std()
    if den < 1e-12:
        return float("nan"), float("nan"), float("nan")
    nr = float(np.sqrt(((p - g) ** 2).mean()) / den)
    npers = float(np.sqrt(((q - g) ** 2).mean()) / den)
    return nr, npers, float(1.0 - nr / npers) if npers > 0 else float("nan")


def time_axis(n_frames: int) -> np.ndarray:
    """Shot time at frame CENTRES; frame k spans [k*dt, (k+1)*dt).

    Origin is the cache's t0_start (0.0 for the production cache), NOT the legacy 1.0 s
    warmup origin -- eval_dynamics' own figure defaulted to the latter and mislabelled
    every absolute time by +1 s.
    """
    return (np.arange(n_frames, dtype=np.float64) + 0.5) * FRAME_DT_S


def main() -> int:
    args = parse_args()
    set_style()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    runs = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run needs LABEL=DIR, got {spec!r}")
        label, d = spec.split("=", 1)
        runs.append((label, find_codes(Path(d))))
    # the reference arm is the one labelled 'real', else the first given
    ref_i = next((i for i, (l, _) in enumerate(runs) if l == "real"), 0)

    written = []
    for name in [m.strip() for m in args.modalities.split(",") if m.strip()]:
        codec, fam = load_codec(name, args.codec_tmpl, device)
        nice, ylab = META.get(name, (name, f"{name} [norm.]"))

        # {trace_key: {arm_label: trace}} plus one GT / persistence trace per key
        series, gt_t, pers_t, K0 = {}, {}, {}, None
        for label, npz_path in runs:
            z = np.load(npz_path, allow_pickle=True)
            gk, pk = f"gt__{name}", f"pred__{name}"
            if pk not in z.files:
                print(f"[skip] {name}: {pk} absent in {npz_path.name}")
                break
            K0, F = int(z["K0"]), int(z["F"])
            pred = decode(codec, z[pk][:F], device)
            for k, v in traces(name, pred).items():
                series.setdefault(k, {})[label] = v
            if not gt_t:
                gt_t = traces(name, decode(codec, z[gk][:F], device))
                # persistence: freeze the last seed frame across the whole window
                pers1 = decode(codec, z[gk][K0 - 1:K0], device)
                pers_t = traces(name, np.repeat(pers1, F, axis=0))
        if not series:
            continue
        ref_label = runs[ref_i][0]
        for tkey in list(series):
            _panel(out_root, tkey, nice, ylab, series[tkey], gt_t[tkey], pers_t[tkey],
                   K0, ref_label, written)
    print(f"\n{len(written)} file(s) written under {out_root}")
    return 0


def _panel(out_root, tkey, nice, ylab, arms: dict, gt, pers, K0, ref_label, written):
    """One overlay+difference figure for a single trace key."""
    if True:
        series = arms
        t = time_axis(len(next(iter(series.values()))))
        t_split = K0 * FRAME_DT_S
        ref = series.get(ref_label)

        fig, (ax, axd) = plt.subplots(
            2, 1, figsize=(9.0, 5.6), sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.12},
        )
        ax.plot(t, gt, color="black", lw=1.0, label="ground truth (measured)", zorder=3)
        ax.plot(t, pers, color="0.55", lw=0.9, ls=":", label="persistence", zorder=1)
        ci = 0
        for label, y in series.items():
            if label == ref_label:
                ax.plot(t, y, color="#0072B2", lw=1.1, ls="--",
                        label=f"{label} actuators", zorder=4)
            else:
                ax.plot(t, y, color=OKABE_ITO[ci % len(OKABE_ITO)], lw=1.0,
                        alpha=0.9, label=label, zorder=2)
                ci += 1
        for a in (ax, axd):
            a.axvline(t_split, color="0.3", lw=0.8, ls="-", alpha=0.6)
        ax.annotate("rollout starts", xy=(t_split, ax.get_ylim()[1]),
                    xytext=(4, -10), textcoords="offset points",
                    fontsize=8, color="#333333", va="top")
        ax.set_ylabel(ylab)
        # Decoded-space accuracy of the reference arm: the dynamics-only number.
        nr, npers, sk = decoded_space_skill(gt, ref, pers, K0) if ref is not None \
            else (float("nan"),) * 3
        ax.set_title(f"{tkey} — reference vs varied actuators\n"
                     f"decoded-GT: nRMSE {nr:.3f}  vs persistence {npers:.3f}  "
                     f"skill {sk:+.3f}  (codec error cancels; dynamics only)",
                     loc="left", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

        # Difference panel: the effect size, on its own scale.
        ci = 0
        for label, y in series.items():
            if label == ref_label:
                continue
            axd.plot(t, y - ref, color=OKABE_ITO[ci % len(OKABE_ITO)], lw=1.0,
                     label=f"{label} − {ref_label}")
            ci += 1
        axd.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        axd.set_ylabel("Δ vs real")
        axd.set_xlabel("shot time [s]")
        if ci:
            axd.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
            span = float(np.nanmax(np.abs([series[l] - ref for l in series
                                           if l != ref_label])))
            axd.set_title(f"max |Δ| = {span:.4g}  (flat line ⇒ actuators had no effect)",
                          loc="left", fontsize=8)

        stem = tkey.replace(" · ", "_").replace(" ", "").replace("/", "-")
        paths = save_fig(fig, out_root / f"{stem}_cases")
        plt.close(fig)
        written += paths
        print(f"[ok] {tkey}: {paths[0]}")


if __name__ == "__main__":
    sys.exit(main())
