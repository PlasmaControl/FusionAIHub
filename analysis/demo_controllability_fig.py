"""Controllability figure for one shot: validation against the measurement, then
model-only actuator scenarios (NBI off, ECH off, no torque) from one switch time.

Two stages, so the figure can be reworked without decoding again:

``extract``
    Reads finished evaluations (``codes_<shot>_step<N>.npz``) of the same checkpoint,
    one folder per scenario, decodes the slow profiles and one spectrogram with the
    frozen codecs, converts n_e, T_e, T_i and v_tor to physical units with the codecs'
    own normalisation, and writes everything the figure needs to one small ``.npz``:
    per-channel frame means (measured and predicted), the stitched spectrograms and the
    actuators in physical units, edited exactly as the rollout saw them. Needs the
    codecs (CPU is enough, about a minute).
``plot``
    Draws the figure from that ``.npz`` alone, in seconds: smoothing, core channels,
    columns, spectrogram and labels are plotting options.

Usage::

    python analysis/demo_controllability_fig.py extract --shot 201240 --switch_frame 50 \
        --real <dir> --nbi_off <dir> --ech_off <dir> --torque0 <dir> --spectro ece \
        --cache_dir <frame_codes> --codec_tmpl '<codecs>/{m}/codec_best.pt' --data fig_data.npz
    python analysis/demo_controllability_fig.py plot --data fig_data.npz --out fig.png \
        --smooth 3 --n_core 3 --columns nbi_off,ech_off
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FRAME_S, T0 = 0.05, 1.0
SCENARIOS = {
    "nbi_off": ("scale:pinj=0@{f}+beam_voltage=0@{f}+tinj=0@{f}", "NBI off"),
    "ech_off": ("scale:ech_power=0@{f}", "ECH off"),
    "torque0": ("scale:tinj=0@{f}", "No beam torque"),
}
RUNS = ("real", *SCENARIOS)
# Fixed scheme for talks and the paper: measurement blue, prediction orange, and a changed
# actuator shares its colour with the prediction it produces.
COLORS = {"truth": "#1f5c9e", "real": "#e07b1a", "scenario": "#7b3294"}
CMAPS = {"truth": "Blues", "real": "Oranges", "scenario": "Purples"}
SLOW = ("ts_core_temp", "ts_core_density", "cer_ti", "cer_rot")
LABELS = {"ts_core_temp": (r"$T_e$ (keV)", 1e-3), "ts_core_density": (r"$n_e$ ($10^{19}$ m$^{-3}$)", 1e-19),
          "cer_ti": (r"$T_i$ (keV)", 1e-3), "cer_rot": (r"$v_{tor}$ (km/s)", 1.0)}
ACTUATORS = {"NBI (MW)": ("pinj", 1e-6), "ECH (MW)": ("ech_power", 1e-6), "torque (N m)": ("tinj", 1.0)}
CAMERAS = ("tangtv_lower", "tangtv_upper")
CAMERA_NAMES = {"tangtv_lower": "lower divertor camera", "tangtv_upper": "upper divertor camera"}


# --------------------------------------------------------------------------- extract
def channel_names(modality):
    """Name of each codec channel of ``modality``: the input keys of ``modalities.yaml``,
    reduced by the loader's ``channels_to_use`` subset (e.g. ECE TECEF01-40, BES BESFU49-64)."""
    import yaml

    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

    yaml_path = Path(__file__).resolve().parents[1] / "src/tokamak_foundation_model/data/config/modalities/modalities.yaml"
    keys = yaml.safe_load(open(yaml_path))["signals"][modality]["input_key"]
    cfg = next((c for c in TokamakH5Dataset.SIGNAL_CONFIGS if c.name == modality), None)
    n_raw = cfg.num_channels if cfg is not None else len(keys)
    use = cfg.channels_to_use if cfg is not None and cfg.channels_to_use is not None else slice(0, n_raw)
    index = list(range(n_raw))[use] if isinstance(use, slice) else list(use)
    if len(keys) == 1:  # one node for all chords (Thomson): chord numbers
        return [f"chord {i + 1}" for i in index]
    names = []
    for i in index:
        node, _, leaf = keys[i].rpartition(":")
        part = node.rsplit(".", 1)[-1] if node and not any(ch.isdigit() for ch in leaf) else leaf
        # CER chords are named T01-T48 (tangential), not by their MDSplus node CHANNELnn.
        names.append(f"T{int(part[7:]):02d}" if part.startswith("CHANNEL") else part.lstrip("\\"))
    return names


_DIAG = {"ts_core_temp": "TS core", "ts_core_density": "TS core", "cer_ti": "CER", "cer_rot": "CER"}


def trace_label(name, channels):
    """e.g. 'CER tangential ch 25, 3, 2' for the channels averaged in a trace."""
    names = channel_names(name)
    shown = [names[c] if c < len(names) else str(c) for c in channels]
    head, _, _ = shown[0].rpartition(" ")
    head += "s" if head == "chord" and len(shown) > 1 else ""
    return f"{_DIAG.get(name, name)} {head} " + ", ".join(x.rpartition(" ")[2] for x in shown) if head else \
        f"{_DIAG.get(name, name)} " + ", ".join(shown)


def spectro_label(d, name):
    """'ECE TECEF01' when the data file records the channel, else 'ECE'."""
    shown = name.upper() if name != "mirnov" else "Mirnov"
    key = f"spectro_channel__{name}"
    if key in d.files:
        names = channel_names(name)
        c = int(d[key])
        shown += f" {names[c] if c < len(names) else c}"
    return shown


def _label_trace(ax, ctx, n, font):
    ax.text(0.01, 0.98, trace_label(n, ctx["channels"][n]), transform=ax.transAxes, ha="left", va="top",
            fontsize=font - 3, color="0.35", bbox={"facecolor": "white", "alpha": 0.7, "lw": 0, "pad": 1})


def _physical(codecs):
    """Per-signal converters to physical units; channels with broken stats become NaN."""
    from tokamak_foundation_model.ignite.config import SLOWTS_PREPROCESS_METHOD

    conv = {}
    for name in SLOW:
        cfg = codecs[name][1]
        mean = np.asarray(cfg.channel_mean, dtype=np.float64)
        raw_std = np.asarray(cfg.channel_std, dtype=np.float64)
        std = np.clip(raw_std, 1e-3, None)
        broken = (np.abs(mean) > 1e4) | (raw_std > 1e4) | (raw_std <= 1e-3)
        log = SLOWTS_PREPROCESS_METHOD[name] == "log_standardize"

        def f(x, mean=mean, std=std, broken=broken, log=log):
            x = np.asarray(x, dtype=np.float64)[:, : mean.size]
            y = x * std[None, :, None] + mean[None, :, None]
            y = np.power(10.0, y) - 1.0 if log else y
            y[:, broken] = np.nan
            return y

        conv[name] = f
    return conv


def extract(a):
    import torch

    from tokamak_foundation_model.ignite import eval_dynamics as ev
    from tokamak_foundation_model.ignite import train_dynamics as td

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spectros = [x for x in a.spectro.split(",") if x]
    names = list(SLOW) + spectros + list(CAMERAS)
    codecs = td.load_frozen_codecs(names, repo=Path.cwd(), tmpl=a.codec_tmpl)
    for codec, _cfg, _fam in codecs.values():
        codec.to(device)
    conv = _physical(codecs)
    out = {"shot": a.shot, "spectro": a.spectro, "switch_frame": a.switch_frame}
    for key in RUNS:
        out[f"dir__{key}"] = str(getattr(a, key))
    spec_ch = {}
    forced = {k: int(v) for k, v in (x.split("=") for x in a.spectro_channel.split(",") if x)}
    for key in RUNS:
        codes = np.load(Path(getattr(a, key)) / f"codes_{a.shot}_step{a.step}.npz", allow_pickle=True)
        K0, F = int(codes["K0"]), int(codes["F"])
        out.update(k0=K0, n_frames=F)
        for name in names:
            codec = codecs[name][0]
            decoded = {
                which: ev.decode_flat_chunked(codec, torch.from_numpy(codes[f"{which}__{name}"]).long()[:F], device)
                for which in ("gt", "pred")
            }
            if name in CAMERAS:
                gt_codes = codes[f"gt__{name}"][K0:F]
                if (gt_codes == gt_codes[:1]).all():
                    continue  # constant code over the window: camera not recorded
                for which, x in decoded.items():
                    if which == "gt" and f"camera_gt__{name}" in out:
                        continue
                    # (F, H, W): first view, averaged over the frame's sub-frames.
                    frames = np.nanmean(np.asarray(x)[:, 0], axis=1).astype(np.float16)
                    out[f"camera_gt__{name}" if which == "gt" else f"camera_pred__{key}__{name}"] = frames
                continue
            if name in spectros:
                first = name == spectros[0]
                if name not in spec_ch:
                    spec_ch[name] = forced.get(name, None)
                    if spec_ch[name] is None:
                        spec_ch[name] = ev._best_channel(decoded["gt"], reduce_axes=(2, 3))
                    out[f"spectro_channel__{name}"] = spec_ch[name]
                    fmax = decoded["gt"].shape[2] * (500_000.0 / 1024) / 1e3
                    gt_img = ev._stitch(decoded["gt"], spec_ch[name]).astype(np.float32)
                    out[f"spectro_fmax_khz__{name}"], out[f"spectro_gt__{name}"] = fmax, gt_img
                    if first:
                        out["spectro_fmax_khz"], out["spectro_gt"] = fmax, gt_img
                img = ev._stitch(decoded["pred"], spec_ch[name]).astype(np.float32)
                out[f"spectro_pred__{key}__{name}"] = img
                if first:
                    out[f"spectro_pred__{key}"] = img
                continue
            for which, x in decoded.items():
                with np.errstate(invalid="ignore"):
                    # (F, C): physical value of each channel, averaged over the frame.
                    out[f"{which}__{key}__{name}"] = np.nanmean(conv[name](x), axis=-1)
        mode = "real" if key == "real" else SCENARIOS[key][0].format(f=a.switch_frame)
        entry = torch.load(Path(os.environ["IGNITE_ACT_GLOBAL"]) / f"{a.shot}.pt",
                           map_location="cpu", weights_only=False)
        edited = ev.apply_actuator_mode(entry["actuators_global"][:F].float(), mode, K0, F=F)
        mu, sd = ev._global_actuator_stats()
        phys = (edited * sd + mu).numpy()
        for label, (group, scale) in ACTUATORS.items():
            out[f"act__{key}__{label}"] = phys[:, ev._actuator_group_columns(group)].sum(1) * scale
    np.savez_compressed(a.data, **out)
    print("wrote", a.data)


# ------------------------------------------------------------------------------ plot
def _core(gt, k0, n_core):
    """The ``n_core`` channels with the largest mean |value| over the predicted window."""
    with np.errstate(invalid="ignore"):
        level = np.nanmean(np.abs(gt[k0:]), axis=0)
    finite = np.isfinite(level)
    order = [int(c) for c in np.argsort(np.where(finite, level, -np.inf))[::-1] if finite[c]]
    return order[: max(1, n_core)]


def _trace(x, channels, smooth):
    """Mean over ``channels`` of (F, C) frame values, then a centred moving average of
    ``smooth`` frames (the same for measurement and prediction)."""
    with np.errstate(invalid="ignore"):
        trace = np.nanmean(x[:, channels], axis=1)
    if smooth > 1:
        kernel = np.ones(smooth)
        num = np.convolve(np.nan_to_num(trace), kernel, mode="same")
        den = np.convolve(np.isfinite(trace).astype(float), kernel, mode="same")
        trace = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    return trace


def _setup(a):
    d = np.load(a.data, allow_pickle=True)
    k0, F, sw = int(d["k0"]), int(d["n_frames"]), int(d["switch_frame"])
    ctx = {"d": d, "shot": str(d["shot"]), "spectro": str(d["spectro"]).split(",")[0], "k0": k0, "F": F, "sw": sw,
           "spectros": str(d["spectro"]).split(","),
           "t": T0 + FRAME_S * np.arange(F), "t_switch": T0 + FRAME_S * sw}
    ctx["display"] = ctx["spectro"].upper() if ctx["spectro"] != "mirnov" else "Mirnov"
    ctx["channels"] = {n: _core(d[f"gt__real__{n}"], k0, a.n_core) for n in SLOW}
    ctx["picked"] = {}
    for term in [x for x in a.channels.split(";") if x]:
        name, _, chans = term.partition("=")
        ctx["channels"][name.strip()] = ctx["picked"][name.strip()] = [int(c) for c in chans.split(",")]
    ctx["extent"] = (ctx["t"][0], ctx["t"][-1] + FRAME_S, 0, float(d["spectro_fmax_khz"]))
    finite = d["spectro_gt"][np.isfinite(d["spectro_gt"])]
    ctx["spec_lim"] = (float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.5)))
    ctx["camera"] = next((c for c in ([a.camera] if getattr(a, "camera", "") else CAMERAS) if f"camera_gt__{c}" in d.files), None)
    plt.rcParams.update({"font.size": a.font, "axes.spines.top": False, "axes.spines.right": False})
    return ctx


def _tr(ctx, a, which, key, n):
    return _trace(ctx["d"][f"{which}__{key}__{n}"], ctx["channels"][n], a.smooth) * LABELS[n][1]


def _spec(ctx, ax, img, role="truth"):
    vmin, vmax = ctx["spec_lim"]
    ax.imshow(img, aspect="auto", origin="lower", cmap=CMAPS[role], vmin=vmin, vmax=vmax,
              extent=ctx["extent"], interpolation="none")


def _note(a, ctx=None):
    note = (f"traces: core average of {a.n_core} channels, {a.smooth * 50} ms moving average"
            if (a.smooth > 1 or a.n_core > 1) else "")
    if ctx and ctx.get("picked"):
        short = {"cer_ti": "T_i", "cer_rot": "v_tor", "ts_core_temp": "T_e", "ts_core_density": "n_e"}
        note += ("; " if note else "") + ", ".join(
            f"{short.get(n, n)} chords {','.join(map(str, c))}" for n, c in ctx["picked"].items())
    return note


def _draw_validation(ctx, a, sub, title):
    """Measured actuators: four profiles and the spectrogram, prediction against the measurement."""
    d, t, k0 = ctx["d"], ctx["t"], ctx["k0"]
    if title:
        sub.suptitle(title, fontsize=a.font + 3, x=0.01, ha="left")
    spectros = [x for x in ctx["spectros"] if f"spectro_gt__{x}" in d.files] or [ctx["spectro"]]
    axes = sub.subplots(2, 2 + len(spectros), gridspec_kw={"width_ratios": [1, 1] + [1.4 if len(spectros) == 1 else 1.1] * len(spectros)})
    for ax, n in zip(axes[:, :2].ravel(), SLOW):
        ax.plot(t, _tr(ctx, a, "gt", "real", n), color=COLORS["truth"], lw=2, label="measured")
        ax.plot(t[k0 - 1:], _tr(ctx, a, "pred", "real", n)[k0 - 1:], color=COLORS["real"], lw=2, ls="--",
                label="prediction")
        ax.axvline(T0 + FRAME_S * k0, color="0.6", lw=1, ls=":")
        ax.set_ylabel(LABELS[n][0])
        ax.set_xlim(t[0], t[-1])
        _label_trace(ax, ctx, n, a.font)
    axes[0, 0].legend(frameon=False, loc="lower left")
    for ax in axes[1, :2]:
        ax.set_xlabel("time (s)")
    ylim = {k: float(v) for k, v in (x.split("=") for x in a.spectro_ylim.split(",") if x)}
    for j, name in enumerate(spectros):
        gt_key = f"spectro_gt__{name}" if f"spectro_gt__{name}" in d.files else "spectro_gt"
        pred_key = f"spectro_pred__real__{name}" if f"spectro_pred__real__{name}" in d.files else "spectro_pred__real"
        fmax_key = f"spectro_fmax_khz__{name}" if f"spectro_fmax_khz__{name}" in d.files else "spectro_fmax_khz"
        finite = d[gt_key][np.isfinite(d[gt_key])]
        lim = (float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.5)))
        extent = (t[0], t[-1] + FRAME_S, 0, float(d[fmax_key]))
        shown = spectro_label(d, name)
        for ax, img, label, role in ((axes[0, 2 + j], d[gt_key], f"{shown} measured", "truth"),
                                     (axes[1, 2 + j], d[pred_key], f"{shown} prediction", "real")):
            ax.imshow(img, aspect="auto", origin="lower", cmap=CMAPS[role], vmin=lim[0], vmax=lim[1],
                      extent=extent, interpolation="none")
            ax.axvline(T0 + FRAME_S * k0, color="w", lw=1, ls=":")
            ax.set_title(label, fontsize=a.font)
            ax.set_ylabel("freq (kHz)")
            if name in ylim:
                ax.set_ylim(0, ylim[name])
        axes[1, 2 + j].set_xlabel("time (s)")


def _draw_scenarios(ctx, a, sub, title, with_truth):
    """Model scenarios, one column each: actuator, then the responding quantities."""
    d, t, k0, spectro = ctx["d"], ctx["t"], ctx["k0"], ctx["spectro"]
    if title:
        sub.suptitle(title, fontsize=a.font + 3, x=0.01, ha="left")
    plan = {"nbi_off": ("NBI (MW)", ["cer_ti", "cer_rot"]), "ech_off": ("ECH (MW)", ["ts_core_temp"]),
            "torque0": ("torque (N m)", ["cer_rot", spectro])}
    columns = [c for c in a.columns.split(",") if c]
    for col, key in zip(list(np.atleast_1d(sub.subfigures(1, len(columns)))), columns):
        act_label, shown = plan[key]
        label = SCENARIOS[key][1]
        col.suptitle(label, fontsize=a.font + 2)
        n_rows = 1 + len(shown) + (1 if spectro in shown else 0)
        ax_all = np.atleast_1d(col.subplots(n_rows, 1, sharex=True))
        ax_all[0].plot(t, d[f"act__real__{act_label}"], color=COLORS["truth"], lw=2, label="measured actuators")
        ax_all[0].plot(t, d[f"act__{key}__{act_label}"], color=COLORS["scenario"], lw=2, ls="--", label=label)
        ax_all[0].set_ylabel(act_label)
        ax_all[0].legend(frameon=False, fontsize=a.font - 2, loc="upper left", bbox_to_anchor=(0.0, 1.32), ncol=2)
        row = 1
        for n in shown:
            if n == spectro:
                for img, which, role in ((d["spectro_pred__real"], "measured actuators", "real"),
                                         (d[f"spectro_pred__{key}"], label, "scenario")):
                    _spec(ctx, ax_all[row], img, role)
                    ax_all[row].set_ylabel(f"{spectro_label(d, spectro)} (kHz)\n{which}", fontsize=a.font - 2)
                    row += 1
                continue
            ax = ax_all[row]
            row += 1
            if with_truth:
                ax.plot(t, _tr(ctx, a, "gt", "real", n), color=COLORS["truth"], lw=1.5, label="measured")
            ax.plot(t[k0 - 1:], _tr(ctx, a, "pred", "real", n)[k0 - 1:], color=COLORS["real"], lw=2,
                    label="prediction, measured actuators")
            ax.plot(t[k0 - 1:], _tr(ctx, a, "pred", key, n)[k0 - 1:], color=COLORS["scenario"], lw=2, ls="--",
                    label="prediction, " + label)
            ax.set_ylabel(LABELS[n][0])
            _label_trace(ax, ctx, n, a.font)
        for ax in ax_all:
            ax.axvline(ctx["t_switch"], color=COLORS["scenario"], lw=1.2, ls=":")
            ax.set_xlim(t[0], t[-1])
        if len(ax_all) > 1 and ax_all[1].get_lines():
            ax_all[1].legend(frameon=False, fontsize=a.font - 3, loc="upper left", bbox_to_anchor=(0.0, 1.38), ncol=2)
        ax_all[-1].set_xlabel("time (s)")


def _save(fig, a, ctx):
    fig.savefig(a.out, dpi=a.dpi)
    fig.savefig(Path(a.out).with_suffix(".pdf"))
    caption = f"DIII-D #{ctx['shot']} (held-out)"
    note = _note(a, ctx)
    if a.kind != "camera":
        caption += f"; prediction from {T0 + FRAME_S * ctx['k0']:.0f} s; scenarios switch at {ctx['t_switch']:.2f} s"
    print("wrote", a.out, "| caption:", caption + (f"; {note}" if note and a.kind != "camera" else ""))


def _validation_title(ctx, a):
    note = _note(a, ctx)
    if not a.with_title:
        return ""
    return a.title or (f"DIII-D #{ctx['shot']} (held-out): prediction from {T0 + FRAME_S * ctx['k0']:.0f} s "
                       "under the measured actuators" + (f"\n{note}" if note else ""))


def _scenario_title(ctx, a, combined):
    if not a.with_title:
        return ""
    lead = "Model scenarios" if combined else f"DIII-D #{ctx['shot']}: model scenarios"
    note = _note(a, ctx)
    return (f"{lead}: actuator changed at {ctx['t_switch']:.2f} s (no measurement exists for these)"
            + (f"\n{note}" if note else ""))


def plot_combined(a):
    """The validation on top, the scenarios below (with the measurement in the scenario panels)."""
    ctx = _setup(a)
    fig = plt.figure(figsize=(a.width, a.height), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.25])
    _draw_validation(ctx, a, top, _validation_title(ctx, a))
    _draw_scenarios(ctx, a, bottom, _scenario_title(ctx, a, True), with_truth=True)
    _save(fig, a, ctx)


def plot_validation(a):
    ctx = _setup(a)
    fig = plt.figure(figsize=(a.width, a.height * 0.45), constrained_layout=True)
    _draw_validation(ctx, a, fig, _validation_title(ctx, a))
    _save(fig, a, ctx)


def plot_scenarios(a):
    ctx = _setup(a)
    fig = plt.figure(figsize=(a.width, a.height * 0.58), constrained_layout=True)
    _draw_scenarios(ctx, a, fig, _scenario_title(ctx, a, False), with_truth=False)
    _save(fig, a, ctx)


def _camera_image(frame, crop, sigma):
    from scipy.ndimage import gaussian_filter

    x = np.asarray(frame, dtype=np.float32)[crop]
    return gaussian_filter(x, sigma) if sigma > 0 else x


def plot_camera(a):
    """Camera snapshots, measured against predicted (and the scenario predictions after the switch)."""
    ctx = _setup(a)
    d, k0, cam = ctx["d"], ctx["k0"], ctx["camera"]
    if cam is None:
        raise SystemExit("no recorded camera in this data file")
    gt = np.asarray(d[f"camera_gt__{cam}"], dtype=np.float32)
    mean = gt[k0:].mean(0)
    floor = np.median(mean) - 1.0
    rows, cols = np.where(mean.mean(1) > floor)[0], np.where(mean.mean(0) > floor)[0]
    crop = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    times = [float(x) for x in a.camera_times.split(",")]
    series = [("measured", gt), ("prediction", d[f"camera_pred__real__{cam}"])]
    roles = ["truth", "real"]
    for key in [c for c in a.camera_scenarios.split(",") if c]:
        series.append((f"prediction, {SCENARIOS[key][1]}", d[f"camera_pred__{key}__{cam}"]))
        roles.append("scenario")
    images = [[_camera_image(x[int(round((tt - T0) / FRAME_S))], crop, a.camera_sigma) for tt in times]
              for _label, x in series]
    flat = np.concatenate([im.ravel() for row in images for im in row])
    vmin, vmax = np.percentile(flat, [1.0, 99.7])
    fig, axes = plt.subplots(len(series), len(times), figsize=(a.width, 1.35 * len(series) + 0.8),
                             constrained_layout=True, squeeze=False)
    if a.with_title:
        fig.suptitle(f"DIII-D #{ctx['shot']}: {CAMERA_NAMES[cam]}", fontsize=a.font + 2, x=0.01, ha="left")
    for r, (label, _x) in enumerate(series):
        for c, tt in enumerate(times):
            ax = axes[r, c]
            cmap = CMAPS[roles[r]] if a.camera_cmap == "auto" else a.camera_cmap
            ax.imshow(images[r][c], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(f"{tt:.1f} s", fontsize=a.font)
        axes[r, 0].set_ylabel(label, fontsize=a.font - 1)
    _save(fig, a, ctx)


def plot(a):
    {"combined": plot_combined, "validation": plot_validation, "scenarios": plot_scenarios,
     "camera": plot_camera}[a.kind](a)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("extract", help="decode once and write the figure data")
    e.add_argument("--shot", required=True)
    e.add_argument("--step", type=int, default=20600)
    e.add_argument("--switch_frame", type=int, default=50)
    for key in RUNS:
        e.add_argument(f"--{key}", required=True, help=f"evaluation folder of the {key} rollout")
    e.add_argument("--spectro", default="ece", help="spectrograms, comma-separated (ece, mhr, bes, co2, mirnov); "
                   "the first is the one in the scenario panels")
    e.add_argument("--spectro_channel", default="", help="channel per spectrogram, e.g. co2=1,bes=11 "
                   "(default: the channel whose frame means vary most; see analysis/demo_spectro_channels.py)")
    e.add_argument("--cache_dir", required=True)
    e.add_argument("--codec_tmpl", required=True)
    e.add_argument("--data", required=True, help="output .npz")
    q = sub.add_parser("plot", help="draw a figure from the data file (seconds, no codecs)")
    q.add_argument("--data", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--kind", choices=("combined", "validation", "scenarios", "camera"), default="combined")
    q.add_argument("--smooth", type=int, default=3, help="centred moving average, frames")
    q.add_argument("--n_core", type=int, default=3, help="core channels averaged")
    q.add_argument("--channels", default="", help='override chords, e.g. "cer_ti=24,2,1;cer_rot=19,18,20"')
    q.add_argument("--spectro_ylim", default="mhr=100", help="max frequency (kHz) per spectrogram, e.g. mhr=100,ece=30")
    q.add_argument("--columns", default="nbi_off,ech_off,torque0", help="scenario columns")
    q.add_argument("--camera", default="", help="camera to show (tangtv_lower or tangtv_upper); default: first recorded")
    q.add_argument("--camera_times", default="2.5,3.5,4.5,5.5", help="camera snapshots (s)")
    q.add_argument("--camera_scenarios", default="", help="scenario rows in the camera figure, e.g. nbi_off")
    q.add_argument("--camera_sigma", type=float, default=2.0, help="Gaussian smoothing of camera images (px)")
    q.add_argument("--camera_cmap", default="auto", help="'auto': measured Blues, prediction Oranges, scenario Purples")
    q.add_argument("--title", default="")
    q.add_argument("--with_title", action="store_true", help="draw the figure titles (off by default)")
    q.add_argument("--font", type=int, default=11)
    q.add_argument("--width", type=float, default=16)
    q.add_argument("--height", type=float, default=13)
    q.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()
    if a.stage == "extract":
        extract(a)
    else:
        plot(a)


if __name__ == "__main__":
    main()
