"""RMP branching experiment: does the model's ELM activity follow the RMP coil current?

From one branching time ``t0`` inside a shot's ELM-suppressed RMP phase, five rollouts
share the same measured context (the ``K0`` frames before ``t0``) and the same random
streams, and differ only in the RMP coil currents after ``t0``; every other actuator keeps
its recorded trajectory. After ``t0`` the model runs fully autoregressively: no measured
plasma frame enters a branch. The model (MaskGIT) keeps no cache or recurrent state; each
frame is predicted from the frames before it, so a shared seed and generator seed make the
branches identical up to ``t0``.

Branches (per coil, physical units, ``I0`` = the current at ``t0``)::

    A  recorded-waveform replay
    B  hold I0
    C  ramp to 0.75 I0 over T_ramp, then hold
    D  ramp to 0.50 I0
    E  ramp to 0.20 I0
    I(t) = I0 + (f I0 - I0) clip((t - t0) / T_ramp, 0, 1)    (eval_dynamics ``ramp:`` mode)

Dα: the filterscope codec stores, per channel and 10 ms bin, ``log1p(rms / env_eps)`` of the
standardized signal minus its 2 ms moving mean (``data.elm_envelope``). The plot inverts the
invertible steps, ``rms * std = env_eps * expm1(env) * channel_std``: the RMS of the fast Dα
component in raw filterscope units, where ELMs stand out (the log compresses them). The slow
Dα level is removed by the 2 ms detrend and cannot be recovered.
ELMs are local maxima of that RMS above ``elm_frac`` times the 99th percentile of the measured
context, at least 20 ms apart. One threshold per shot, from the measured context, serves
every branch and the measurement. Burst rate: ELMs in a sliding 250 ms window.

Stages::

    rollout   one shot, every branch and seed (GPU); writes <out>/<shot>_branches.npz
    plot      figure and metrics table from those archives (CPU)

Usage::

    python analysis/rmp_branching.py rollout --ckpt <dynamics_stepN.pt> --shot 190739 \
        --t0 2.5 --t_ramp 0.3 --seeds 3 --branches A,B --cache_dir <frame_codes> \
        --codec_tmpl '<codecs>/{m}/codec_best.pt' --out <dir>
    python analysis/rmp_branching.py plot --shot 190739 --out <dir> --codec_tmpl ...

Branch archives of one shot may come from several jobs (``--branches``); ``plot`` merges
every ``<shot>_branch<X>.npz`` in ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_controllability_fig as dm  # noqa: E402

BRANCHES = {
    "A": (None, "A  recorded RMP (replay)"),
    "B": (1.0, "B  hold I$_0$"),
    "C": (0.75, "C  ramp to 0.75 I$_0$"),
    "D": (0.50, "D  ramp to 0.50 I$_0$"),
    "E": (0.20, "E  ramp to 0.20 I$_0$"),
}
FRAME_S, T0_SHOT = 0.05, 1.0  # frame length; shot time of cache frame 0
BIN_S = 0.01  # filterscope envelope bin
# t0 per shot, from the measured Dα and RMP traces (see SELECTION in the plot stage).
DEFAULT_T0 = {"190738": 2.50, "190739": 2.50}


def _frame(t):
    return int(round((t - T0_SHOT) / FRAME_S))


def _rmp_amplitude(phys, columns):
    """Mean |I| over the RMP coils, kA."""
    return np.abs(phys[:, columns]).mean(1) * 1e-3


# ----------------------------------------------------------------------------- rollout
def rollout(a):
    import torch

    from tokamak_foundation_model.ignite import eval_dynamics as ev

    device = ev._rank_device()
    model, cfg, step = ev.load_model(Path(a.ckpt), device)
    model.eval()
    k0 = int(cfg.k0_seed)
    branch_frame = _frame(a.t0)  # first predicted frame, in shot frames
    start = branch_frame - k0
    if start < 0:
        raise SystemExit(f"t0={a.t0} s leaves fewer than K0={k0} context frames")
    ramp_frames = max(1, int(round(a.t_ramp / FRAME_S)))
    mu, sd = ev._global_actuator_stats()
    columns = ev._actuator_group_columns("rmp")
    names = [m.name for m in cfg.modalities]
    for key in a.branches.split(","):
        factor = BRANCHES[key][0]
        mode = "real" if factor is None else f"ramp:rmp={factor}@{k0}/{ramp_frames}"
        out = {"shot": a.shot, "branch": key, "mode": mode, "t0": a.t0, "t_ramp": a.t_ramp,
               "start": start, "k0": k0, "step": step, "ckpt": str(a.ckpt), "seeds": a.seeds}
        for seed in range(a.seeds):
            cache = ev.load_shot_cache(Path(a.cache_dir), a.shot)
            generator = torch.Generator(device=device).manual_seed(seed)
            result = ev.rollout_shot(model, cfg, cache, k0, a.temperature, generator, device,
                                     actuator_mode=mode, shot=a.shot, start=start)
            for name in names:
                out[f"pred__{seed}__{name}"] = result.pred_codes[name].numpy().astype(np.int16)
                if seed == 0:
                    out[f"gt__{name}"] = result.gt_codes[name].numpy().astype(np.int16)
            if seed == 0:
                F = result.n_frames
                edited = ev.apply_actuator_mode(cache["actuators"][start:start + F].float(), mode, k0, F=F)
                out["rmp_ka"] = _rmp_amplitude((edited * sd + mu).numpy(), columns)
                out["rmp_coils_a"] = (edited * sd + mu).numpy()[:, columns]
                out["n_frames"] = F
            print(f"[rmp] {a.shot} {key} seed {seed} done", flush=True)
        path = Path(a.out) / f"{a.shot}_branch{key}.npz"
        np.savez_compressed(path, **out)
        print("wrote", path, flush=True)


# ------------------------------------------------------------------------------- plot
def _envelope(codec, codes, device):
    """(F, n_tok) filterscope codes -> (F * bins, channels) decoded log1p envelope."""
    import torch

    from tokamak_foundation_model.ignite import eval_dynamics as ev

    x = ev.decode_flat_chunked(codec, torch.from_numpy(codes.astype(np.int64)), device)
    return np.asarray(x).transpose(0, 2, 1).reshape(-1, x.shape[1])  # (F, C, E) -> (F*E, C)


def _fast_rms(envelope, cfg):
    """Invert the envelope's log1p and standardization: fast-component RMS, raw units."""
    std = np.asarray(cfg.channel_std, dtype=np.float64)[: envelope.shape[1]]
    return float(cfg.env_eps) * np.expm1(np.asarray(envelope, dtype=np.float64)) * std


def _detect(hp, threshold, min_gap):
    """Indices of local maxima of ``hp`` above ``threshold``, at least ``min_gap`` apart."""
    peaks = [i for i in range(1, len(hp) - 1)
             if hp[i] > threshold and hp[i] >= hp[i - 1] and hp[i] > hp[i + 1]]
    kept = []
    for i in sorted(peaks, key=lambda i: -hp[i]):
        if all(abs(i - j) >= min_gap for j in kept):
            kept.append(i)
    return np.array(sorted(kept), dtype=int)


def _rate(events, n_bins, window_bins):
    """ELMs per second in a centred window of ``window_bins``, at every bin."""
    hits = np.zeros(n_bins)
    hits[events] = 1
    kernel = np.ones(window_bins)
    return np.convolve(hits, kernel, mode="same") / (window_bins * BIN_S)


def _missing(envelopes, min_bins=5):
    """Bins where every channel reads ~0 for at least ``min_bins``: no measurement."""
    flat = np.all(np.abs(envelopes) < 0.05, axis=1)
    out = np.zeros_like(flat)
    run = 0
    for i, f in enumerate(flat):
        run = run + 1 if f else 0
        if run >= min_bins:
            out[i - run + 1:i + 1] = True
    return out


def plot(a):
    import matplotlib.pyplot as plt
    import torch

    from tokamak_foundation_model.ignite import train_dynamics as td

    out_dir = Path(a.out)
    archives = {key: dict(np.load(out_dir / f"{a.shot}_branch{key}.npz", allow_pickle=True))
                for key in BRANCHES if (out_dir / f"{a.shot}_branch{key}.npz").exists()}
    if "A" not in archives:
        raise SystemExit(f"no replay (branch A) archive for {a.shot} in {out_dir}")
    ref = archives["A"]
    k0, start, F = int(ref["k0"]), int(ref["start"]), int(ref["n_frames"])
    t0, t_ramp, seeds = float(ref["t0"]), float(ref["t_ramp"]), int(ref["seeds"])
    codec, codec_cfg, _ = td.load_frozen_codecs(["filterscopes"], tmpl=a.codec_tmpl)["filterscopes"]
    device = torch.device("cpu")
    channel_names = dm.channel_names("filterscopes")
    ch = channel_names.index(a.channel)
    measured_all = _envelope(codec, ref["gt__filterscopes"][:F], device)
    missing = _missing(measured_all)
    measured = np.where(missing, np.nan, _fast_rms(measured_all, codec_cfg)[:, ch])
    n_bins = len(measured)
    t = T0_SHOT + (start * FRAME_S) + BIN_S * np.arange(n_bins)
    t_frames = T0_SHOT + FRAME_S * (start + np.arange(F))
    context = slice(0, k0 * (n_bins // F))
    min_gap, window_bins = 2, int(round(a.rate_window / BIN_S))
    threshold = float(a.elm_frac * np.nanpercentile(measured[context], 99))

    traces = {"measured": (measured, np.nan_to_num(measured))}
    for key, arc in archives.items():
        for seed in range(seeds):
            env = _envelope(codec, arc[f"pred__{seed}__filterscopes"][:F], device)
            rms = _fast_rms(env, codec_cfg)[:, ch]
            traces[(key, seed)] = (rms, rms)
    events = {k: _detect(x, threshold, min_gap) for k, (_, x) in traces.items()}
    events["measured"] = np.array([i for i in events["measured"] if not missing[i]], dtype=int)
    rates = {k: _rate(ev_, n_bins, window_bins) for k, ev_ in events.items()}
    rates["measured"] = np.where(missing, np.nan, rates["measured"])

    # Spectrogram of one channel, stitched along time as in the demo figures (seed 0).
    images, spec = {}, None
    if a.spectro:
        from tokamak_foundation_model.ignite import eval_dynamics as ev

        scodec = td.load_frozen_codecs([a.spectro], tmpl=a.codec_tmpl)[a.spectro][0]

        def decode(codes):
            return ev.decode_flat_chunked(scodec, torch.from_numpy(codes[:F].astype(np.int64)), device)

        gt_spec = decode(ref[f"gt__{a.spectro}"])
        chan = a.spectro_channel if a.spectro_channel >= 0 else ev._best_channel(gt_spec, (2, 3))
        images["measured"] = ev._stitch(gt_spec, chan)
        for key, arc in archives.items():
            images[key] = ev._stitch(decode(arc[f"pred__0__{a.spectro}"]), chan)
        finite = images["measured"][np.isfinite(images["measured"])]
        names = dm.channel_names(a.spectro)
        spec = {"extent": (t_frames[0], t_frames[-1] + FRAME_S, 0, gt_spec.shape[2] * (500_000.0 / 1024) / 1e3),
                "lim": (float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.5))),
                "label": f"{a.spectro.upper() if a.spectro != 'mirnov' else 'Mirnov'} "
                         f"{names[chan] if chan < len(names) else chan}"}

    # ---- metrics: per branch, mean (min..max) over seeds
    after_ramp = t0 + t_ramp
    windows = {"context": (t[0], t0), "ramp": (t0, after_ramp),
               "plateau": (after_ramp, min(after_ramp + a.plateau_s, t[-1]))}

    def window_stats(key):
        rows = {}
        for wname, (lo, hi) in windows.items():
            sel = (t >= lo) & (t < hi)
            dur = sel.sum() * BIN_S
            if key == "measured":
                sel = sel & ~missing
                dur = sel.sum() * BIN_S
                vals = [np.isin(events["measured"], np.where(sel)[0]).sum() / dur if dur > 0 else np.nan]
                mean = [np.nanmean(measured[sel]) if sel.any() else np.nan]
            else:
                vals = [np.isin(events[(key, s)], np.where(sel)[0]).sum() / dur for s in range(seeds)]
                mean = [np.mean(traces[(key, s)][0][sel]) for s in range(seeds)]
            rows[wname] = (vals, mean, dur)
        first = []
        for s in ([None] if key == "measured" else range(seeds)):
            ev_ = events["measured"] if key == "measured" else events[(key, s)]
            after = t[ev_][t[ev_] >= t0]
            first.append(after[0] - t0 if len(after) else np.nan)
        return rows, first

    table = []
    for key in ["measured", *archives]:
        rows, first = window_stats(key)
        label = "measured" if key == "measured" else key
        rmp = ref["rmp_ka"] if key in ("measured", "A") else archives[key]["rmp_ka"]
        plateau = (t_frames >= after_ramp) & (t_frames < windows["plateau"][1])
        entry = {"branch": label, "RMP plateau (kA)": float(np.mean(rmp[plateau]))}
        for wname, (vals, mean, dur) in rows.items():
            entry[f"ELM rate {wname} (Hz)"] = _fmt(vals)
            entry[f"Dα fast RMS mean {wname} (1e15)"] = _fmt(np.asarray(mean) / 1e15, 2)
        entry["first ELM after t0 (s)"] = _fmt(first, 2)
        entry["seeds"] = 1 if key == "measured" else seeds
        table.append(entry)
    with open(out_dir / f"{a.shot}_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with open(out_dir / f"{a.shot}_metrics.md", "w") as f:
        f.write(_markdown(table, a, t0, t_ramp, threshold, windows, channel_names[ch], ref))
    print(open(out_dir / f"{a.shot}_metrics.md").read())

    # ---- figure: the demo scenario layout (demo_controllability_fig), one column per branch.
    # Measured blue, prediction under the recorded RMP orange; a changed RMP waveform and the
    # prediction it drives share the scenario purple. Rows share their axes.
    plt.rcParams.update({"font.size": a.font, "axes.spines.top": False, "axes.spines.right": False})
    shown = [k for k in ("A", "B", "C", "D", "E") if k in archives]
    seed_bins = k0 * (n_bins // F)
    channel = f"Filterscope {channel_names[ch]}"
    fig = plt.figure(figsize=(a.width, a.height), constrained_layout=True)
    columns = np.atleast_1d(fig.subfigures(1, len(shown)))
    ylim = (np.nanmin([np.nanmin(v[0]) for v in traces.values()]),
            np.nanmax([np.nanmax(v[0]) for v in traces.values()]))
    rate_max = np.nanmax([np.nanmax(r) for r in rates.values()])
    first_axes = None
    for col, key in zip(columns, shown):
        replay = key == "A"
        label = "recorded RMP" if replay else BRANCHES[key][1][3:]
        col.suptitle(label, fontsize=a.font + 2)
        n_rows = 6 if spec else 4
        ratios = [1.1, 1, 1] + ([1.1, 1.1] if spec else []) + [0.9]
        axes = col.subplots(n_rows, 1, sharex=True, gridspec_kw={"height_ratios": ratios})
        if first_axes is not None:
            for ax, ref_ax in zip(axes, first_axes):
                ax.sharey(ref_ax)
        else:
            first_axes = axes
        # Row 0: the imposed RMP coil current.
        ax = axes[0]
        ax.plot(t_frames, ref["rmp_ka"], color=dm.COLORS["truth"], lw=2, label="measured actuators")
        if not replay:
            rmp = archives[key]["rmp_ka"]
            ax.plot(t_frames[k0 - 1:], rmp[k0 - 1:], color=dm.COLORS["scenario"], lw=2, ls="--", label=label)
            ax.axhline(rmp[-1], color=dm.COLORS["scenario"], lw=0.8, ls=":")
        ax.set_ylabel("RMP coil current (kA)" if key == shown[0] else "")
        ax.legend(frameon=False, fontsize=a.font - 2, loc="upper left", bbox_to_anchor=(0.0, 1.32), ncol=2)
        # Rows 1-2: Dα, the reference above the branch, on separate panels to keep the bursts visible.
        pairs = ([("measured", "measured", dm.COLORS["truth"]),
                  (("A", 0), "prediction, measured actuators", dm.COLORS["real"])] if replay else
                 [(("A", 0), "prediction, measured actuators", dm.COLORS["real"]),
                  ((key, 0), "prediction, " + label, dm.COLORS["scenario"])])
        for ax, (trace_key, trace_name, c) in zip(axes[1:3], pairs):
            env = traces[trace_key][0]
            if trace_key == "measured":
                ax.plot(t, env, color=c, lw=0.8, label=trace_name)
            else:
                ax.plot(t[:seed_bins], measured[:seed_bins], color=dm.COLORS["truth"], lw=0.8)
                ax.plot(t[seed_bins - 1:], env[seed_bins - 1:], color=c, lw=0.8, label=trace_name)
            elm = events[trace_key]
            ax.plot(t[elm], np.full(len(elm), ylim[1] * 1.04), "|", color=c, ms=6, mew=1.2)
            ax.set_ylim(ylim[0], ylim[1] * 1.1)
            ax.set_ylabel("fast D$_\\alpha$ RMS\n(raw units)" if key == shown[0] else "")
            dm.label_box(ax, f"{channel}, {trace_name}", a.font)
        # Rows 3-4: the spectrogram, reference above the branch (seed 0).
        if spec:
            roles = ([("measured", "measured", "truth"), ("A", "prediction, measured actuators", "real")]
                     if replay else [("A", "prediction, measured actuators", "real"),
                                     (key, "prediction, " + label, "scenario")])
            for ax, (image_key, image_name, role) in zip(axes[3:5], roles):
                ax.imshow(images[image_key], aspect="auto", origin="lower", cmap=dm.CMAPS[role],
                          vmin=spec["lim"][0], vmax=spec["lim"][1], extent=spec["extent"],
                          interpolation="none")
                ax.set_ylim(0, a.spectro_ylim)
                ax.set_ylabel("freq (kHz)" if key == shown[0] else "")
                dm.label_box(ax, f"{spec['label']}, {image_name}", a.font)
        # Last row: ELM rate, measured and the two predictions of the column (mean over seeds).
        ax = axes[-1]
        ax.plot(t, rates["measured"], color=dm.COLORS["truth"], lw=1.5, label="measured")
        for branch, c, ls in (("A", dm.COLORS["real"], "-"), (key, dm.COLORS["scenario"], "--")):
            if replay and branch == key and ls == "--":
                continue
            r = np.mean([rates[(branch, s)] for s in range(seeds)], axis=0)
            ax.plot(t[seed_bins - 1:], r[seed_bins - 1:], color=c, lw=1.8, ls=ls)
        ax.set_ylim(0, rate_max * 1.1)
        ax.set_ylabel(f"ELM rate (Hz)" if key == shown[0] else "")
        dm.label_box(ax, f"{a.rate_window * 1e3:.0f} ms window, mean of {seeds} rollouts", a.font)
        ax.set_xlabel("time (s)")
        for i, ax in enumerate(axes):
            in_image = spec is not None and i in (3, 4)
            ax.axvline(t0, color="w" if in_image else dm.COLORS["scenario"], lw=1.2, ls=":")
            if not replay and not in_image:
                ax.axvspan(t0, after_ramp, color=dm.COLORS["scenario"], alpha=0.08, lw=0)
            ax.set_xlim(t[0], t[-1])
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{a.shot}_rmp_branches.{ext}", dpi=a.dpi)
    print("wrote", out_dir / f"{a.shot}_rmp_branches.png")


def _fmt(values, digits=1):
    v = np.asarray(values, dtype=float)
    if np.all(np.isnan(v)):
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{np.nanmean(v):.{digits}f} ({np.nanmin(v):.{digits}f}..{np.nanmax(v):.{digits}f})"


def _markdown(table, a, t0, t_ramp, threshold, windows, channel, ref):
    lines = [f"# RMP branching, shot {a.shot}", "",
             f"checkpoint `{ref['ckpt']}` (step {int(ref['step'])}); t0 = {t0:.2f} s; "
             f"T_ramp = {t_ramp:.2f} s; {int(ref['seeds'])} seeds per branch (mean, min..max).",
             f"Dα: fast-component RMS per 10 ms (codec envelope with log1p and scaling inverted). "
             f"ELMs on {channel}: local maxima above {threshold:.3g} = {a.elm_frac} x the 99th "
             f"percentile of the measured context, >= 20 ms apart.",
             "Windows: " + ", ".join(f"{k} {lo:.2f}-{hi:.2f} s" for k, (lo, hi) in windows.items()) + ".",
             f"t0 selection: {SELECTION.get(a.shot, 'see --t0')}", ""]
    keys = list(table[0])
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "---|" * len(keys))
    for row in table:
        lines.append("| " + " | ".join(str(row[k]) if not isinstance(row[k], float) else f"{row[k]:.2f}"
                                       for k in keys) + " |")
    return "\n".join(lines) + "\n"


# Evidence for DEFAULT_T0, from the cached measurement (filterscopes decoded with the frozen
# codec, RMP coil currents from the act_global side cache), 2026-10-28.
SELECTION = {
    "190738": "t0 = 2.50 s: end of the RMP plateau, inside the ELM-suppressed interval, "
              "before the recorded RMP reduction.",
    "190739": "t0 = 2.50 s: end of the RMP plateau, inside the ELM-suppressed interval, "
              "before the recorded RMP reduction.",
}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    r = sub.add_parser("rollout")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--shot", required=True)
    r.add_argument("--t0", type=float, default=None, help="branching time (s); default per shot")
    r.add_argument("--t_ramp", type=float, default=0.3, help="ramp duration of B-E (s)")
    r.add_argument("--seeds", type=int, default=3, help="rollouts per branch (generator seeds 0..N-1)")
    r.add_argument("--branches", default="A,B,C,D,E")
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--cache_dir", required=True)
    r.add_argument("--codec_tmpl", default=None)
    r.add_argument("--out", required=True)
    q = sub.add_parser("plot")
    q.add_argument("--shot", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--codec_tmpl", required=True)
    q.add_argument("--channel", default="FS04", help="filterscope channel for ELM detection (Dα)")
    q.add_argument("--elm_frac", type=float, default=0.2,
                   help="ELM threshold as a fraction of the measured context's 99th percentile")
    q.add_argument("--spectro", default="mhr", help="spectrogram rows (mhr, mirnov, ece, ...); '' = none")
    q.add_argument("--spectro_channel", type=int, default=-1,
                   help="spectrogram channel; -1 = the one whose power varies most in the measurement")
    q.add_argument("--spectro_ylim", type=float, default=100.0, help="max frequency shown (kHz)")
    q.add_argument("--rate_window", type=float, default=0.25, help="burst-rate window (s)")
    q.add_argument("--plateau_s", type=float, default=1.5, help="length of the scored plateau window (s)")
    q.add_argument("--font", type=int, default=11)
    q.add_argument("--width", type=float, default=24)
    q.add_argument("--height", type=float, default=15)
    q.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.stage == "rollout":
        if a.t0 is None:
            a.t0 = DEFAULT_T0[a.shot]
        if not os.environ.get("IGNITE_ACT_GLOBAL"):
            raise SystemExit("set IGNITE_ACT_GLOBAL: the RMP edits are in physical units")
        rollout(a)
    else:
        plot(a)


if __name__ == "__main__":
    main()
