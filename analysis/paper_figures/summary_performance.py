"""Summary of IGNITE's predictions over many discharges (``summary_performance``).

Every quantity is in physical units and compared with the measurement of the same
discharge; no reference model is involved.

Panels
------
A   Predicted against measured core T_e, T_i, n_e and |v_tor|: one dot per predicted
    50 ms time point of every discharge.
C   Median relative error of the panel A quantities against time since the initial
    state, over the discharges whose plasma lasts to FULL_END_S (same set in every bin).
B   Predicted against measured total spectrogram power of Mirnov (5-100 kHz) and ECE
    (5-30 kHz) in dB, edge D-alpha fluctuation level (filterscope envelope, log axes)
    and lower-divertor camera pixel brightness (normalised per frame by the codec), as
    a density of pixels.
Every rollout runs under its discharge's measured actuators; no changed plan enters.

D   Change across the beam and ECH steps of the measured actuator plans: predicted
    against measured change of core T_i (beam steps) and T_e (ECH steps).

Points are the median over independently sampled rollouts (seeds) of the same discharge.
The spread over seeds is shown once, in panel C: around each median-error curve, the band
is the range of the same median error computed from each single rollout (the full range
for up to five seeds, else 10-90 %).

Two stages::

    # decode the rollout archives (CPU node, a few minutes per seed)
    python analysis/paper_figures/summary_performance.py reduce --runs DIR[,DIR...]
    # draw
    python analysis/paper_figures/summary_performance.py plot

Each DIR holds ``codes_<shot>_step<N>.npz`` archives, one sampled rollout per
discharge (e.g. a scorecard's ``val32``). The reduced data go to
``WORK/summary_performance.npz``; the figure to ``figures/summary_performance.*``.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures as mf  # noqa: E402

dc = mf.dc
NAME = "summary_performance"
REDUCED = mf.WORK / f"{NAME}.npz"
PROFILES = {  # signal -> (label, unit, scale to unit)
    "ts_core_temp": ("$T_e$", "keV", 1e-3),
    "cer_ti": ("$T_i$", "keV", 1e-3),
    "ts_core_density": ("$n_e$", "$10^{19}$ m$^{-3}$", 1e-19),
    "cer_rot": ("$|v_{tor}|$", "km/s", 1.0),     # stored in km/s
}
MIN_IP_MA = 0.1                  # below this peak plasma current: no plasma, skipped
N_CORE = 3                       # core value: mean of the 3 strongest channels
# Whole spectrogram (all frequencies above the 0 Hz bin): Mirnov 0-250 kHz, ECE 0-31 kHz.
SPECTROS = {"mirnov": ("Mirnov", (0.1, 1e9)), "ece": ("ECE", (0.1, 1e9))}
SPEC_FS_HZ = 500_000.0           # both sampled at 500 kHz (modalities.yaml)
CAMERA = "tangtv_lower"          # pixel brightness, normalised per frame by the codec
PIX_STRIDE = 4                   # camera pixels sampled on a 4 x 4 grid
ERR_BIN = 5                      # panel B: frames per bin (0.25 s)
ERR_MIN_SHOTS = 3                # ... and discharges per bin
FULL_END_S = 5.5                 # panel B: only discharges whose plasma lasts to here
POINT, BIN = "#9aa5b1", "#0b3a66"   # bars, dots
STEP_NBI_MW = 1.5                # actuator step: change of the 0.2 s means
STEP_ECH_MW = 0.4
STEP_HALF = 4                    # frames on each side of a step (0.2 s)
STEP_GAP = 10                    # frames between two steps of one discharge
CODEC_TMPL = ("/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_codecs_v4/"
              "{m}/codec_best.pt")


# ------------------------------------------------------------------ reduce
def _recorded(codes, k0):
    """True when a modality's measured codes change after the initial state."""
    c = np.asarray(codes)[k0:]
    return not (c == c[:1]).all()


def _band_power_db(img, cfg, ch, band_khz):
    """Total spectrogram power in ``band_khz`` per 50 ms frame, in dB.

    ``img`` is one modality's decoded spectrogram, (F, C, n_f, n_t): log10 power,
    z-scored per (channel, frequency) when the codec was trained that way, which is
    undone here exactly. The dB reference is the raw signal's unit (relative).
    """
    x = np.asarray(img, np.float64)[:, ch]                           # (F, n_f, n_t)
    if getattr(cfg, "logpow_standardize", False) and cfg.logpow_freq_mean is not None:
        fm = np.asarray(cfg.logpow_freq_mean, np.float64)[ch][:, None]
        fs = np.clip(np.asarray(cfg.logpow_freq_std, np.float64)[ch],
                     float(getattr(cfg, "logpow_std_floor", 0.25)), None)[:, None]
        x = x * fs + fm
    n_fft = int(getattr(cfg, "stft_n_fft", 1024))
    f = np.arange(x.shape[1]) * SPEC_FS_HZ / n_fft / 1e3
    band = (f >= band_khz[0]) & (f <= band_khz[1])
    return 10.0 * np.log10(np.mean(10.0 ** x[:, band], axis=(1, 2)))


def reduce(a):
    import torch

    from tokamak_foundation_model.ignite import eval_dynamics as ev
    from tokamak_foundation_model.ignite import train_dynamics as td
    from tokamak_foundation_model.ignite.designer import machine as mc
    from tokamak_foundation_model.ignite.designer import shots as ds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = [Path(x) for x in a.runs.split(",") if x]
    names = list(PROFILES) + list(SPECTROS) + ["filterscopes", CAMERA]
    codecs = td.load_frozen_codecs(names, repo=mf.REPO, tmpl=CODEC_TMPL)
    for codec, _cfg, _fam in codecs.values():
        codec.to(device)
    conv = ev.profile_converters(codecs, list(PROFILES))
    cfg_fs = codecs["filterscopes"][1]
    fs_std = np.asarray(cfg_fs.channel_std, np.float64)[None, :, None]

    def fs_phys(x):
        return float(cfg_fs.env_eps) * np.expm1(np.asarray(x, np.float64)) * fs_std

    shots = sorted(int(p.name.split("_")[1]) for p in runs[0].glob("codes_*_step*.npz"))
    out = {"runs": np.array([str(r) for r in runs])}
    kept = []
    for shot in shots:
        arcs = [np.load(next(r.glob(f"codes_{shot}_step*.npz")), allow_pickle=True)
                for r in runs if any(r.glob(f"codes_{shot}_step*.npz"))]
        z0 = arcs[0]
        # Measured actuators only: a changed plan's archive also holds its measured-plan
        # baseline as real__<modality>, so any such key means a changed plan.
        if any(k.startswith("real__") for z in arcs for k in z.files):
            raise ValueError(f"{shot}: an archive holds a changed actuator plan; "
                             "the summary uses measured actuators only")
        K0, F = int(z0["K0"]), int(z0["F"])
        s = ds.load(shot)
        # Frames that end before the plasma does (t_end is not on the frame grid).
        live = F if not s.t_end else min(F, int((s.t_end - mc.T0) // mc.FRAME_S))
        if not s.ip_peak_ma or s.ip_peak_ma < MIN_IP_MA:
            print(f"{shot}: no plasma (peak Ip {s.ip_peak_ma} MA); skipped")
            continue
        if live < K0 + 10:
            print(f"{shot}: plasma ends at frame {live}, before 0.5 s of prediction; skipped")
            continue
        out[f"{shot}__k0"], out[f"{shot}__live"] = K0, live
        act = s.act[:F]
        out[f"{shot}__pnbi_MW"] = act[:, mc.columns("pinj")].sum(1) * 1e-6
        out[f"{shot}__ech_MW"] = act[:, mc.columns("ech_power")].sum(1) * 1e-6

        def dec(name, key, z):
            return ev.decode_flat_chunked(codecs[name][0], ev.archived(z, key)[:F], device)

        for n, (_lab, _unit, scale) in PROFILES.items():
            if not _recorded(z0[f"gt__{n}"], K0):
                continue
            gt = np.nanmean(conv[n](dec(n, f"gt__{n}", z0)), -1) * scale
            chans = ev.core_channels(gt, K0, N_CORE)
            core = lambda x: np.abs(np.nanmean(x[:, chans], 1))  # noqa: E731
            out[f"{shot}__{n}__gt"] = core(gt)
            out[f"{shot}__{n}__pred"] = np.stack([
                core(np.nanmean(conv[n](dec(n, f"pred__{n}", z)), -1) * scale)
                for z in arcs])
        for sp, (_lab, band) in SPECTROS.items():
            if not _recorded(z0[f"gt__{sp}"], K0):
                continue
            cfg_sp = codecs[sp][1]
            g = np.asarray(dec(sp, f"gt__{sp}", z0))
            ch = ev._best_channel(g, reduce_axes=(2, 3))
            out[f"{shot}__{sp}__gt"] = _band_power_db(g, cfg_sp, ch, band)
            out[f"{shot}__{sp}__pred"] = np.stack([
                _band_power_db(dec(sp, f"pred__{sp}", z), cfg_sp, ch, band) for z in arcs])
        if _recorded(z0["gt__filterscopes"], K0):
            g = fs_phys(dec("filterscopes", "gt__filterscopes", z0)).mean(-1)  # (F, C)
            with np.errstate(all="ignore"):
                var = np.nanvar(g[K0:live], axis=0)
            if not np.isfinite(var).any():
                g = None
        if _recorded(z0["gt__filterscopes"], K0) and g is not None:
            c = int(np.nanargmax(var))
            out[f"{shot}__dalpha__gt"] = g[:, c] / 1e16
            out[f"{shot}__dalpha__pred"] = np.stack([
                fs_phys(dec("filterscopes", "pred__filterscopes", z)).mean(-1)[:, c] / 1e16
                for z in arcs])
        if _recorded(z0[f"gt__{CAMERA}"], K0):
            g = np.asarray(dec(CAMERA, f"gt__{CAMERA}", z0))       # (F, view, sub, H, W)
            # The recorded view: the one whose measured image has structure (as in the
            # paper figures' extraction); frames averaged over their sub-frames.
            vm = np.nanmean(g[K0:live], axis=(0, 2))
            view = int(np.argmax([np.nanstd(v[10:-10, 10:-10]) for v in vm]))
            frames = np.nanmean(g[:, view], axis=1)
            crop = mf.crop_of(frames, K0)

            def pixels(x):
                # Every PIX_STRIDE-th pixel of the image region, every predicted frame.
                x = np.asarray(x)[K0:live][(slice(None), *crop)]
                return x[:, ::PIX_STRIDE, ::PIX_STRIDE].reshape(-1).astype(np.float16)

            out[f"{shot}__pixels__gt"] = pixels(frames)
            out[f"{shot}__pixels__pred"] = np.stack([
                pixels(np.nanmean(np.asarray(dec(CAMERA, f"pred__{CAMERA}", z))[:, view], axis=1))
                for z in arcs])
        kept.append(shot)
        print(f"{shot}: {len(arcs)} rollouts, live to frame {live}", flush=True)
    out["shots"] = np.array(kept)
    REDUCED.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REDUCED, **out)
    print("wrote", REDUCED)


# -------------------------------------------------------------------- plot
def spread(x):
    """Seed median and error bar (low, high) over axis 0."""
    if x.shape[0] <= 5:
        lo, hi = np.nanmin(x, 0), np.nanmax(x, 0)
    else:
        lo, hi = np.nanpercentile(x, [10, 90], axis=0)
    return np.nanmedian(x, 0), lo, hi


def frame_points(d, shot, key):
    """Measured, predicted (median, lo, hi) and time since the initial state, per frame."""
    k0, live = int(d[f"{shot}__k0"]), int(d[f"{shot}__live"])
    gt, pred = np.asarray(d[f"{shot}__{key}__gt"]), np.asarray(d[f"{shot}__{key}__pred"])
    m, lo, hi = spread(pred[:, k0:live])
    g = gt[k0:live]
    t = mf.FRAME * np.arange(live - k0)
    ok = np.isfinite(g) & np.isfinite(m)
    return [(float(a), float(b), float(c), float(e), float(f))
            for a, b, c, e, f in zip(g[ok], m[ok], lo[ok], hi[ok], t[ok])]


def mf_frame(t):
    return int(round((t - mf.T0) / mf.FRAME))


def steps(d, shot, act, thresh):
    """Frames where the 0.2 s mean of ``act`` changes by more than ``thresh``."""
    k0, live = int(d[f"{shot}__k0"]), int(d[f"{shot}__live"])
    x = d[f"{shot}__{act}"]
    found, last = [], -10 ** 6
    for k in range(k0 + STEP_HALF, live - 2 * STEP_HALF):
        jump = x[k:k + STEP_HALF].mean() - x[k - STEP_HALF:k].mean()
        if abs(jump) > thresh and k - last >= STEP_GAP:
            found.append((k, float(jump)))
            last = k
    return found


def change(y, k):
    """Mean over the 0.4 s after the step's settling minus the 0.2 s before it."""
    return np.nanmean(y[..., k + 2:k + 2 + 2 * STEP_HALF], -1) - \
        np.nanmean(y[..., k - STEP_HALF:k], -1)


def same_ticks(ax):
    """The same tick positions on x and y (the panel's ranges are already equal)."""
    lo, hi = ax.get_xlim()
    from matplotlib.ticker import MaxNLocator
    ticks = [t for t in MaxNLocator(4).tick_values(lo, hi) if lo - 1e-9 <= t <= hi + 1e-9]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def parity(ax, pts, label, unit, n_shots, kind="lin", note=""):
    """Measured against predicted, one dot per time point; bar: spread of the predictions.

    ``kind``: "lin" (axes from 0, relative error), "log" (log axes, relative error) or
    "db" (values in dB, error in dB).
    """
    if not pts:
        ax.set_axis_off()
        return np.nan
    g, m, lo, hi, t = np.array(pts).T
    sc = ax.scatter(g, m, s=2.5, color=BIN, alpha=0.5, lw=0, zorder=3, rasterized=True)
    both = np.concatenate([g, m])
    if kind == "log":
        bot, top = np.nanpercentile(both[both > 0], [0.5, 99.5])
        bot, top = bot / 1.3, top * 1.3
    elif kind == "db":
        bot, top = np.nanpercentile(both, [1.0, 99.5])
        pad = 0.05 * (top - bot)
        bot, top = bot - pad, top + pad
    else:
        bot, top = 0.0, 1.05 * np.nanpercentile(both, 99.5)
    ax.plot([bot, top], [bot, top], color="0.3", lw=0.6, ls="--", zorder=1)
    if kind == "log":
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(bot, top)
    ax.set_ylim(bot, top)
    ax.set_aspect("equal")
    ax.set_xlabel(f"measured {label} ({unit})")
    ax.set_ylabel(f"predicted {label} ({unit})")
    if kind == "db":
        rel = np.nanmedian(np.abs(m - g))
        err = f"median error {rel:.1f} dB"
    else:
        rel = np.nanmedian(np.abs(m - g) / np.abs(g))
        err = f"median error {100 * rel:.0f} %"
    if kind != "log":
        same_ticks(ax)
    # Below the diagonal, bottom right: the corner a good prediction leaves empty.
    ax.text(0.97, 0.04, (f"{note}\n" if note else "") + err,
            transform=ax.transAxes, ha="right", va="bottom", fontsize=mf.FONT - 2,
            color="0.25", zorder=5)
    return sc




def pixel_scatter(ax, g, m, t, n=40000):
    """Measured against predicted camera pixels (random subsample), coloured by time."""
    lo, hi = np.nanpercentile(np.concatenate([g, m]), [0.5, 99.9])
    idx = np.random.default_rng(0).choice(len(g), size=min(n, len(g)), replace=False)
    ax.scatter(g[idx], m[idx], color=BIN, s=1.0, alpha=0.3, lw=0, zorder=2,
               rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=0.6, ls="--", zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("measured camera")
    ax.set_ylabel("predicted camera")
    same_ticks(ax)
    err = np.nanmedian(np.abs(m - g))
    ax.text(0.97, 0.04, f"median error {err:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=mf.FONT - 2, color="0.25", zorder=5)


def header(fig, ax, letter, text, dy=0.012):
    """Panel letter and a one-line description, left-aligned above ``ax``."""
    fig.canvas.draw()
    pos = ax.get_position()
    x = pos.x0 - 0.075
    fig.text(x, pos.y1 + dy, letter, fontweight="bold", fontsize=mf.FONT + 1,
             ha="left", va="bottom")
    fig.text(x + 0.03, pos.y1 + dy, text, fontsize=mf.FONT, ha="left", va="bottom")


def plot(a):
    import matplotlib.pyplot as plt

    mf.style()
    d = np.load(REDUCED, allow_pickle=True)
    shots = [int(s) for s in d["shots"]]
    n_seeds = int(max(d[k].shape[0] for k in d.files if k.endswith("__pred")))
    fig = plt.figure(figsize=(mf.WIDTH, 5.9))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.8], hspace=0.42,
                             left=0.09, right=0.985, top=0.95, bottom=0.075)
    src = {}

    # A: core profiles, predicted against measured.
    row = outer[0].subgridspec(1, 4, wspace=0.5)
    for i, (n, (lab, unit, _s)) in enumerate(PROFILES.items()):
        ax = fig.add_subplot(row[i])
        have = [s for s in shots if f"{s}__{n}__gt" in d.files]
        pts = [p for s in have for p in frame_points(d, s, n)]
        sc = parity(ax, pts, lab, unit, len(have))
        src[f"A_{n}_measured_pred_lo_hi"] = np.array(pts) if pts else np.zeros((0, 4))
        if i == 0:
            ax_a = ax

    # C: median relative error against time since the initial state.
    row_c = outer[1].subgridspec(1, 4, wspace=0.5)
    row_bd = outer[2].subgridspec(1, 2, wspace=0.0, width_ratios=[1.0, 0.0001])
    ax = fig.add_subplot(row_bd[0])
    tt = mf.T0 + mf.FRAME * np.arange(100)
    colors = ("#1b7f79", "#c1440e", "#5d6b12", "#7b3294")
    ends = []
    n_full = sum(int(d[f"{s}__live"]) >= mf_frame(FULL_END_S) for s in shots)
    for (n, (lab, _u, _s)), col in zip(PROFILES.items(), colors):
        errs = []
        for s in shots:
            if f"{s}__{n}__gt" not in d.files or int(d[f"{s}__live"]) < mf_frame(FULL_END_S):
                continue
            k0, live = int(d[f"{s}__k0"]), int(d[f"{s}__live"])
            g = d[f"{s}__{n}__gt"]
            m = np.nanmedian(d[f"{s}__{n}__pred"], 0)
            # Row 0: the seed-median prediction; rows 1..: each single rollout.
            p = np.concatenate([m[None], np.asarray(d[f"{s}__{n}__pred"])])
            e = np.full((len(p), 100), np.nan)
            e[:, k0:live] = np.abs(p[:, k0:live] - g[k0:live]) / np.abs(g[k0:live])
            errs.append(e)
        errs = np.array(errs).transpose(1, 0, 2)          # (1 + seeds, shots, frames)
        # Median over discharges and frames in 0.25 s bins (aggregation, no smoothing of
        # any trace); a bin needs ERR_MIN_SHOTS discharges.
        edges = np.arange(mf_frame(2.0), mf_frame(FULL_END_S) + 1, ERR_BIN)
        meds = np.full((len(errs), len(edges) - 1), np.nan)
        for b in range(len(edges) - 1):
            blk = errs[:, :, edges[b]:edges[b + 1]]
            if (np.isfinite(blk[0]).any(1)).sum() >= ERR_MIN_SHOTS:
                meds[:, b] = np.nanmedian(blk.reshape(len(blk), -1), 1)
        med = meds[0]
        _m, lo, hi = spread(meds[1:])
        tb = mf.T0 + mf.FRAME * 0.5 * (edges[:-1] + edges[1:]) - 2.0
        if len(meds) > 2:
            ax.fill_between(tb, 100 * lo, 100 * hi, color=col, alpha=0.18, lw=0,
                            zorder=1)
            src[f"C_{n}_seed_rel_err_pct_lo_hi"] = 100 * np.stack([lo, hi])
        ax.plot(tb, 100 * med, color=col, lw=1.1, marker="o", ms=2.5, zorder=2)
        k = np.flatnonzero(np.isfinite(med))[-1]
        ends.append([100 * med[k], tb[k], lab, col])
        src[f"C_{n}_median_rel_err_pct"] = 100 * med
    # End labels, nudged apart so that no two overlap.
    ends.sort(key=lambda e: e[0])
    ax.set_ylim(0, None)
    gap = 0.12 * ax.get_ylim()[1]
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, x, lab, col in ends:
        ax.text(x + 0.08, y, lab, color=col, fontsize=mf.FONT - 1, va="center")
    ax.set_xlim(0, FULL_END_S - 2.0 + 0.45)
    ax.set_ylim(0, None)
    ax.set_xlabel("time since the initial state (s)")
    ax.set_ylabel("median error (%)")
    ax_c = ax

    # B: spectrogram power (Mirnov, ECE), edge D-alpha level and camera contrast.
    for jj, (sp, (lab, band)) in enumerate(SPECTROS.items()):
        ax = fig.add_subplot(row_c[jj])
        have = [s for s in shots if f"{s}__{sp}__gt" in d.files]
        pts = [p for s in have for p in frame_points(d, s, sp)]
        parity(ax, pts, lab, "dB", len(have), kind="db")
        src[f"B_{sp}_power_db_measured_pred_lo_hi"] = np.array(pts) if pts else np.zeros((0, 4))
        if jj == 0:
            ax_b = ax
    ax = fig.add_subplot(row_c[2])
    have = [s for s in shots if f"{s}__dalpha__gt" in d.files]
    pts = [p for s in have for p in frame_points(d, s, "dalpha")]
    parity(ax, pts, "D$_\\alpha$", "a.u.", len(have), kind="log")
    src["C_dalpha_1e16_measured_pred_lo_hi"] = np.array(pts) if pts else np.zeros((0, 4))
    ax = fig.add_subplot(row_c[3])
    have = [s for s in shots if f"{s}__pixels__gt" in d.files]
    if have:
        gs, ms, ts = [], [], []
        for s_ in have:
            gt = d[f"{s_}__pixels__gt"].astype(np.float32)
            n_fr = int(d[f"{s_}__live"]) - int(d[f"{s_}__k0"])
            gs.append(gt)
            ms.append(np.median(d[f"{s_}__pixels__pred"].astype(np.float32), 0))
            ts.append(np.repeat(mf.FRAME * np.arange(n_fr), len(gt) // n_fr))
        g, m, t = (np.concatenate(x) for x in (gs, ms, ts))
        pixel_scatter(ax, g, m, t)
        src["B_camera_pixels_measured_pred_time"] = np.stack([g, m, t]).astype(np.float16)

    # Self-contained: what each panel shows, and what each mark is.
    header(fig, ax_a, "A", "Core plasma: predicted against measured")
    header(fig, ax_b, "B", "Spectrograms, edge D$_\\alpha$ and divertor camera: "
                          "predicted against measured")
    header(fig, ax_c, "C", "Error against time")
    mf.OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(mf.OUT / f"{NAME}.pdf")
    fig.savefig(mf.OUT / f"{NAME}.png", dpi=300)
    np.savez_compressed(mf.OUT / f"{NAME}_source_data.npz", **src)
    print("wrote", mf.OUT / f"{NAME}.pdf", f"({n_seeds} seeds, {len(shots)} discharges)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reduce")
    r.add_argument("--runs", required=True, help="comma list of rollout dirs (one per seed)")
    sub.add_parser("plot")
    a = p.parse_args()
    {"reduce": reduce, "plot": plot}[a.cmd](a)


if __name__ == "__main__":
    main()
