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
    # Zero net torque with sources of equal size: two co sources, 30L and 330L (channels 4, 6,
    # the CER beams), against the two counter sources 210L/R (2, 3), which cancel the co
    # torque frame by frame at the torque per MW of 201240's 210R. 150L/R and 30R/330R
    # (0, 1, 5, 7) are off, so NBI power drops to about four sources.
    "balanced": ("balance:off=0,1,5,7;counter=2,3;nm_per_mw=-0.979;kv=72@{f}",
                 "Balanced injection"),
    # Ramps over 1 s (20 frames) from the switch: every channel moves linearly from its
    # measured value to the target setting and then follows the target, so the measured
    # waveforms (later beam or gyrotron turn-offs) carry on. All four gyrotrons in use already
    # run near their limit, so ECH can only go down. Torque changes at fixed hardware
    # use the balanced mix above; half way is every source at half its duty cycle.
    "ech_lin_off": ("linear:20:scale:ech_power=0@{f}", "ECH ramped off"),
    "ech_lin_half": ("linear:20:scale:ech_power=0.5@{f}", "ECH ramped to half"),
    "tq_lin_zero": ("linear:20:balance:off=0,1,5,7;counter=2,3;nm_per_mw=-0.979;kv=72@{f}",
                    "Torque ramped to zero"),
    "tq_lin_half": ("linear:20/0.5:balance:off=0,1,5,7;counter=2,3;nm_per_mw=-0.979;kv=72@{f}",
                    "Torque ramped to half"),
}
OPTIONAL_RUNS = ("balanced", "ech_lin_off", "ech_lin_half", "tq_lin_zero", "tq_lin_half")
RUNS = ("real", *SCENARIOS)
# Fixed scheme for talks and the paper: measurement blue, prediction orange, and a changed
# actuator shares its colour with the prediction it produces.
COLORS = {"truth": "#1f5c9e", "real": "#e07b1a", "scenario": "#7b3294"}
CMAPS = {"truth": "Blues", "real": "Oranges", "scenario": "Purples"}
SLOW = ("ts_core_temp", "ts_core_density", "cer_ti", "cer_rot")
LABELS = {"ts_core_temp": (r"$T_e$ (keV)", 1e-3), "ts_core_density": (r"$n_e$ ($10^{19}$ m$^{-3}$)", 1e-19),
          "cer_ti": (r"$T_i$ (keV)", 1e-3), "cer_rot": (r"$|v_{tor}|$ (km/s)", 1.0)}
# Plotted as magnitudes: the sign of v_tor is a CER convention and confuses talk audiences.
ABSOLUTE = ("cer_rot",)
# Temperatures are drawn from 0 keV, so a drop reads as its true fraction.
ZERO_BASED = ("ts_core_temp", "cer_ti")
# Momentum confinement time: the rotation answers a torque change within about this time.
TAU_PHI_S = 0.1
TORQUE_SCENARIOS = ("torque0", "balanced", "tq_lin_zero", "tq_lin_half")
# Energy confinement time: T_e answers an ECH change within about this time.
TAU_E_S = 0.1
ECH_SCENARIOS = ("ech_off", "ech_lin_off", "ech_lin_half")
# Response time marked after the switch, per scenario: (seconds, symbol, what changed).
TIMESCALES = {**{k: (TAU_PHI_S, r"\tau_\phi", "torque") for k in TORQUE_SCENARIOS},
              **{k: (TAU_E_S, r"\tau_E", "ECH") for k in ECH_SCENARIOS}}
ACTUATORS = {"NBI (MW)": ("pinj", 1e-6), "ECH (MW)": ("ech_power", 1e-6), "torque (N m)": ("tinj", 1.0),
             "gas (Torr L/s)": ("gas_flow", 1.0)}
# Axis labels of the actuators (the ACTUATORS keys name them in the data files).
ACT_SHOWN = {"NBI (MW)": r"$P_\mathrm{inj}$ (MW)", "torque (N m)": r"$T_\mathrm{inj}$ (N m)",
             "ECH (MW)": r"$P_\mathrm{ECH}$ (MW)", "gas (Torr L/s)": r"gas (Torr L/s)"}
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


def label_box(ax, text, font):
    """The grey channel label in the top-left corner of a trace panel."""
    ax.text(0.01, 0.98, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=font - 3, color="0.35", bbox={"facecolor": "white", "alpha": 0.7, "lw": 0, "pad": 1})


def _label_trace(ax, ctx, n, font):
    label_box(ax, trace_label(n, ctx["channels"][n]), font)


def _physical(codecs):
    """Per-signal converters to physical units; channels with broken stats become NaN."""
    from tokamak_foundation_model.ignite.eval_dynamics import profile_converters

    return profile_converters(codecs, SLOW)


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
    runs = [key for key in RUNS if getattr(a, key)]
    # --donor <shot>=<dirs>: rollouts under another shot's actuators (mode donor:<shot>).
    donors = dict(x.split("=", 1) for x in a.donor)
    for shot in donors:
        setattr(a, f"donor_{shot}", donors[shot])
        runs.append(f"donor_{shot}")
    for key in runs:
        out[f"dir__{key}"] = str(getattr(a, key))
    spec_ch = {}
    member = None  # ensemble member shown in the spectrograms (--spectro_member medoid)
    forced = {k: int(v) for k, v in (x.split("=") for x in a.spectro_channel.split(",") if x)}
    for key in runs:
        # A rollout folder, or several joined by "+": an ensemble whose members are
        # averaged after decoding (independent seeds of the same scenario).
        members = [
            np.load(Path(d) / f"codes_{a.shot}_step{a.step}.npz", allow_pickle=True)
            for d in str(getattr(a, key)).split("+")
        ]
        codes = members[0]
        K0, F = int(codes["K0"]), int(codes["F"])
        out.update(k0=K0, n_frames=F)
        # nearest:<m> picks the member on modality m, so decode m first.
        target = a.spectro_member.partition(":")[2]
        for name in sorted(names, key=lambda n: n != target):
            codec = codecs[name][0]
            each = [np.asarray(ev.decode_flat_chunked(codec, ev.archived(z, f"{a.pred}__{name}")[:F], device))
                    for z in members]
            decoded = {
                "gt": ev.decode_flat_chunked(codec, ev.archived(codes, f"gt__{name}")[:F], device),
                "pred": np.mean(each, axis=0),
            }
            if name in CAMERAS:
                gt_codes = codes[f"gt__{name}"][K0:F]
                if (gt_codes == gt_codes[:1]).all():
                    continue  # constant code over the window: camera not recorded
                # The camera has two views and some shots record only one: show the view
                # whose measured image has structure (the other decodes to a flat field).
                # Judge it inside a 10 px margin: a dead view can decode to a flat field
                # with a dark edge band (204011 lower, raw channel 0), which wins on the
                # whole image.
                gt_mean = np.nanmean(np.asarray(decoded["gt"])[K0:F], axis=(0, 2))
                view = int(np.argmax([np.nanstd(v[10:-10, 10:-10]) for v in gt_mean]))
                for which, x in decoded.items():
                    if which == "gt" and f"camera_gt__{name}" in out:
                        continue
                    # (F, H, W): the recorded view, averaged over the frame's sub-frames.
                    frames = np.nanmean(np.asarray(x)[:, view], axis=1).astype(np.float16)
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
                if a.spectro_member.startswith("nearest:") and len(each) > 1 and member is None \
                        and name == target:
                    # The seed closest to the measurement on modality <target>.
                    ref = ev._stitch(decoded["gt"][K0:], spec_ch[name])
                    member = int(np.argmin([np.mean((ev._stitch(x[K0:], spec_ch[name]) - ref) ** 2)
                                            for x in each]))
                    out["spectro_member"] = member
                if member is not None and a.spectro_member.startswith("nearest:"):
                    img = ev._stitch(each[member], spec_ch[name]).astype(np.float32)
                if a.spectro_member == "medoid" and len(each) > 1:
                    # Averaging seeds halves the spectrogram's texture and flattens the mode
                    # lines' wander, so show one sampled rollout: the seed closest to the mean
                    # over the predicted window. It is chosen once, on the measured-actuator
                    # run, so every scenario shows the same seed and matches before the switch.
                    if member is None:
                        mean = ev._stitch(decoded["pred"][K0:], spec_ch[name])
                        member = int(np.argmin([np.mean((ev._stitch(x[K0:], spec_ch[name]) - mean) ** 2)
                                                for x in each]))
                        out["spectro_member"] = member
                    img = ev._stitch(each[member], spec_ch[name]).astype(np.float32)
                out[f"spectro_pred__{key}__{name}"] = img
                if first:
                    out[f"spectro_pred__{key}"] = img
                continue
            for which, x in decoded.items():
                with np.errstate(invalid="ignore"):
                    # (F, C): physical value of each channel, averaged over the frame.
                    out[f"{which}__{key}__{name}"] = np.nanmean(conv[name](x), axis=-1)
        if key.startswith("donor_"):
            mode = "donor:" + key[len("donor_"):]
        else:
            mode = "real" if key == "real" else SCENARIOS[key][0].format(f=a.switch_frame)
        entry = torch.load(Path(os.environ["IGNITE_ACT_GLOBAL"]) / f"{a.shot}.pt",
                           map_location="cpu", weights_only=False)
        edited = ev.apply_actuator_mode(entry["actuators_global"][:F].float(), mode, K0,
                                        cache_dir=a.cache_dir, F=F)
        mu, sd = ev._global_actuator_stats()
        phys = (edited * sd + mu).numpy()
        for label, (group, scale) in ACTUATORS.items():
            out[f"act__{key}__{label}"] = phys[:, ev._actuator_group_columns(group)].sum(1) * scale
    np.savez_compressed(a.data, **out)
    print("wrote", a.data)


# ------------------------------------------------------------------------------ plot
def _core(gt, k0, n_core):
    """The ``n_core`` channels with the largest mean |value| over the predicted window."""
    from tokamak_foundation_model.ignite.eval_dynamics import core_channels

    return core_channels(gt, k0, n_core)


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
    _style(a)
    return ctx


def _style(a):
    """Presentation figures use seaborn's talk theme; ``--style paper`` keeps the plain look."""
    if a.style == "talk":
        import seaborn as sns

        # A half-slide figure is drawn at about the size it is shown, so the theme is scaled down.
        sns.set_theme(context="talk", font_scale=0.7 if getattr(a, "layout", "") == "half" else 1.0)
        a.font = int(round(plt.rcParams["font.size"]))
    else:
        plt.rcParams.update({"font.size": a.font, "axes.spines.top": False, "axes.spines.right": False})


def shared_legend(fig, a, extra=()):
    """One legend for the whole figure, above it: every labelled artist once, plus ``extra``."""
    handles, labels = [], []
    stack = [fig]
    while stack:
        f = stack.pop(0)
        for ax in f.axes:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in labels:
                    handles.append(h)
                    labels.append(lab)
        stack.extend(getattr(f, "subfigs", []))
    for h, lab in extra:
        handles.append(h)
        labels.append(lab)
    ncol = 2 if getattr(a, "layout", "") == "half" else 3
    if ncol == 2 and len(labels) > 2:
        # Two columns, filled column by column: inputs on the left, predictions on the right.
        pairs = list(zip(handles, labels))
        first = [x for x in pairs if not x[1].startswith("prediction")]
        second = [x for x in pairs if x[1].startswith("prediction")]
        rows = max(len(first), len(second))
        blank = (plt.Line2D([], [], lw=0), "")
        pairs = first + [blank] * (rows - len(first)) + second + [blank] * (rows - len(second))
        handles, labels = [h for h, _ in pairs], [lab for _, lab in pairs]
    fig.legend(handles, labels, loc="outside upper center", ncol=min(len(labels), ncol), frameon=False,
               fontsize=a.font - 2)


def _tau_box(ax, ctx, key):
    """Mark the confinement time after an actuator change, in which the plasma should respond, as
    a bar on the time axis (not over the data)."""
    from matplotlib.transforms import blended_transform_factory

    t0 = ctx["t_switch"]
    ax.plot([t0, t0 + TIMESCALES[key][0]], [0, 0], transform=blended_transform_factory(ax.transData, ax.transAxes),
            color="0.25", lw=7, solid_capstyle="butt", clip_on=False, zorder=5)


def _tau_handle(key):
    seconds, symbol, what = TIMESCALES[key]
    return (plt.Line2D([], [], color="0.25", lw=7, solid_capstyle="butt"),
            rf"${symbol} \approx$ {seconds * 1e3:.0f} ms after {what} change")


def _tr(ctx, a, which, key, n):
    trace = _trace(ctx["d"][f"{which}__{key}__{n}"], ctx["channels"][n], a.smooth) * LABELS[n][1]
    return np.abs(trace) if n in ABSOLUTE else trace


def _spec(ctx, ax, img, role="truth", name=None):
    """One spectrogram; ``name`` selects its own range and colour limits in a multi-spectrogram file."""
    d = ctx["d"]
    vmin, vmax = ctx["spec_lim"]
    extent = ctx["extent"]
    if name and f"spectro_gt__{name}" in d.files:
        finite = d[f"spectro_gt__{name}"][np.isfinite(d[f"spectro_gt__{name}"])]
        vmin, vmax = float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.5))
        extent = (*extent[:3], float(d[f"spectro_fmax_khz__{name}"]))
    ax.imshow(img, aspect="auto", origin="lower", cmap=CMAPS[role], vmin=vmin, vmax=vmax,
              extent=extent, interpolation="none")
    ax.grid(False)


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
    if a.style == "talk":
        # Slide-shaped: the four profiles 2 x 2 on top, the spectrograms below (measured row,
        # predicted row, one column per spectrogram).
        top, bottom = sub.subfigures(2, 1, height_ratios=[1.0, 0.9])
        trace_axes = top.subplots(2, 2, sharex=True)
        spec_axes = np.asarray(bottom.subplots(2, len(spectros), sharex=True)).reshape(2, len(spectros))
    else:
        axes = sub.subplots(2, 2 + len(spectros),
                            gridspec_kw={"width_ratios": [1, 1] + [1.4 if len(spectros) == 1 else 1.1] * len(spectros)})
        trace_axes, spec_axes = axes[:, :2], axes[:, 2:]
    for ax, n in zip(trace_axes.ravel(), SLOW):
        ax.plot(t, _tr(ctx, a, "gt", "real", n), color=COLORS["truth"], lw=2, label="measured")
        ax.plot(t[k0 - 1:], _tr(ctx, a, "pred", "real", n)[k0 - 1:], color=COLORS["real"], lw=2,
                label="prediction, measured actuators")
        ax.axvline(T0 + FRAME_S * k0, color="0.6", lw=1, ls=":")
        ax.set_ylabel(LABELS[n][0])
        ax.set_xlim(t[0], a.tmax or t[-1])
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        if n in ZERO_BASED:
            ax.set_ylim(bottom=0)
        _label_trace(ax, ctx, n, a.font)
    for ax in trace_axes[1]:
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
        for ax, img, label, role in ((spec_axes[0, j], d[gt_key], f"{shown} measured", "truth"),
                                     (spec_axes[1, j], d[pred_key], f"{shown} prediction", "real")):
            ax.imshow(img, aspect="auto", origin="lower", cmap=CMAPS[role], vmin=lim[0], vmax=lim[1],
                      extent=extent, interpolation="none")
            ax.grid(False)
            ax.axvline(T0 + FRAME_S * k0, color="w", lw=1, ls=":")
            ax.set_title(label, fontsize=a.font)
            ax.set_ylabel("freq (kHz)")
            if name in ylim:
                ax.set_ylim(0, ylim[name])
            ax.set_xlim(t[0], a.tmax or t[-1] + FRAME_S)
        spec_axes[1, j].set_xlabel("time (s)")


def _draw_scenarios(ctx, a, sub, title, with_truth):
    """Model scenarios, one column each: actuator, then the responding quantities."""
    d, t, k0, spectro = ctx["d"], ctx["t"], ctx["k0"], ctx["spectro"]
    if title:
        sub.suptitle(title, fontsize=a.font + 3, x=0.01, ha="left")
    plan = {"nbi_off": ("NBI (MW)", ["cer_ti", "cer_rot"]), "ech_off": ("ECH (MW)", ["ts_core_temp"]),
            # Torque changes: every spectrogram in the file, so mode frequencies can be compared.
            "torque0": (("torque (N m)", "NBI (MW)"), ["cer_ti", "cer_rot", *ctx["spectros"]]),
            "balanced": (("torque (N m)", "NBI (MW)"), ["cer_ti", "cer_rot", *ctx["spectros"]])}
    for key in ("ech_lin_off", "ech_lin_half"):
        plan[key] = ("ECH (MW)", ["ts_core_temp", *ctx["spectros"]])
    for key in ("tq_lin_zero", "tq_lin_half"):
        plan[key] = (("torque (N m)", "NBI (MW)"), ["cer_ti", "cer_rot", *ctx["spectros"]])
    ylim = {k: float(v) for k, v in (x.split("=") for x in a.spectro_ylim.split(",") if x)}
    columns = [c for c in a.columns.split(",") if c]
    for col, key in zip(list(np.atleast_1d(sub.subfigures(1, len(columns)))), columns):
        act_labels, shown = plan[key]
        act_labels = (act_labels,) if isinstance(act_labels, str) else act_labels
        label = SCENARIOS[key][1]
        col.suptitle(label, fontsize=a.font + 2)
        n_rows = len(act_labels) + len(shown) + sum(1 for n in shown if n in ctx["spectros"])
        ax_all = np.atleast_1d(col.subplots(n_rows, 1, sharex=True))
        for ax, act_label in zip(ax_all, act_labels):
            ax.plot(t, d[f"act__real__{act_label}"], color=COLORS["truth"], lw=2, label="measured actuators")
            ax.plot(t, d[f"act__{key}__{act_label}"], color=COLORS["scenario"], lw=2, ls="--",
                    label="changed actuator")
            ax.set_ylabel(ACT_SHOWN.get(act_label, act_label))
        row = len(act_labels)
        for n in shown:
            if n in ctx["spectros"]:
                real_key = f"spectro_pred__real__{n}" if f"spectro_pred__real__{n}" in d.files else "spectro_pred__real"
                key_key = f"spectro_pred__{key}__{n}" if f"spectro_pred__{key}__{n}" in d.files else f"spectro_pred__{key}"
                for img, which, role in ((d[real_key], "measured actuators", "real"),
                                         (d[key_key], label, "scenario")):
                    _spec(ctx, ax_all[row], img, role, name=n)
                    if n in ylim:
                        ax_all[row].set_ylim(0, ylim[n])
                    ax_all[row].set_ylabel("freq (kHz)")
                    label_box(ax_all[row], f"{spectro_label(d, n)}, prediction, {which}", a.font)
                    row += 1
                continue
            ax = ax_all[row]
            row += 1
            if with_truth:
                ax.plot(t, _tr(ctx, a, "gt", "real", n), color=COLORS["truth"], lw=1.5, label="measured")
            ax.plot(t[k0 - 1:], _tr(ctx, a, "pred", "real", n)[k0 - 1:], color=COLORS["real"], lw=2,
                    label="prediction, measured actuators")
            ax.plot(t[k0 - 1:], _tr(ctx, a, "pred", key, n)[k0 - 1:], color=COLORS["scenario"], lw=2, ls="--",
                    label="prediction, changed actuator")
            ax.set_ylabel(LABELS[n][0])
            if n in ZERO_BASED:
                ax.set_ylim(bottom=0)
            _label_trace(ax, ctx, n, a.font)
        if key in TIMESCALES:
            _tau_box(ax_all[-1], ctx, key)
        for ax in ax_all:
            ax.axvline(ctx["t_switch"], color=COLORS["scenario"], lw=1.2, ls=":")
            ax.set_xlim(t[0], t[-1])
        ax_all[-1].set_xlabel("time (s)")


def _save(fig, a, ctx):
    if a.kind != "camera":
        columns = [c for c in a.columns.split(",") if c]
        marked = [c for c in columns if c in TIMESCALES] if a.kind != "validation" else []
        extra = [_tau_handle(marked[0])] if marked else []
        shared_legend(fig, a, extra)
    fig.savefig(a.out, dpi=a.dpi)
    fig.savefig(Path(a.out).with_suffix(".pdf"))
    caption = f"DIII-D #{ctx['shot']}"
    note = _note(a, ctx)
    if a.kind != "camera":
        caption += f"; prediction from {T0 + FRAME_S * ctx['k0']:.0f} s; scenarios switch at {ctx['t_switch']:.2f} s"
    print("wrote", a.out, "| caption:", caption + (f"; {note}" if note and a.kind != "camera" else ""))


def _validation_title(ctx, a):
    note = _note(a, ctx)
    if not a.with_title:
        return ""
    return a.title or (f"DIII-D #{ctx['shot']}: prediction from {T0 + FRAME_S * ctx['k0']:.0f} s "
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
    size = ((a.width, a.height) if a.layout == "half" else (16, 13) if a.style == "talk"
            else (a.width, a.height * 0.45))
    fig = plt.figure(figsize=size, constrained_layout=True)
    _draw_validation(ctx, a, fig, _validation_title(ctx, a))
    _save(fig, a, ctx)


def _draw_scenario_half(ctx, a, fig, key):
    """One scenario on half a slide, two panels per row: the actuators, then the responding
    profiles, then each spectrogram's prediction under the measured and the changed actuators."""
    d, t, k0 = ctx["d"], ctx["t"], ctx["k0"]
    torque = (("torque (N m)", "NBI (MW)"), ["cer_ti", "cer_rot"])
    ech = (("ECH (MW)", "NBI (MW)"), ["ts_core_temp", "ts_core_density"])
    act_labels, shown = {"torque0": torque, "balanced": torque, "tq_lin_zero": torque, "tq_lin_half": torque,
                         "nbi_off": (("NBI (MW)",), ["cer_ti", "cer_rot"]),
                         "ech_off": ech, "ech_lin_off": ech, "ech_lin_half": ech}[key]
    label = SCENARIOS[key][1]
    spectros = [n for n in (a.sim_spectro.split(",") if a.sim_spectro else ctx["spectros"])
                if f"spectro_pred__{key}__{n}" in d.files]
    ylim = {k: float(v) for k, v in (x.split("=") for x in a.spectro_ylim.split(",") if x)}
    # Each group starts a new row of two panels.
    groups = [[("act", x) for x in act_labels], [("trace", n) for n in shown]]
    groups += [[("spec", (n, "real")), ("spec", (n, key))] for n in spectros]
    rows = [g[i:i + 2] for g in groups for i in range(0, len(g), 2)]
    axes = fig.subplots(len(rows), 2, sharex=True, squeeze=False)
    for r, row in enumerate(rows):
        if len(row) == 1:
            axes[r, 1].set_visible(False)
        for c, (kind, what) in enumerate(row):
            ax = axes[r, c]
            if kind == "act":
                ax.plot(t, d[f"act__real__{what}"], color=COLORS["truth"], lw=1.8, label="measured")
                ax.plot(t, d[f"act__{key}__{what}"], color=COLORS["scenario"], lw=1.8, ls="--",
                        label="changed actuator")
                ax.set_ylabel(ACT_SHOWN.get(what, what))
            elif kind == "trace":
                ax.plot(t, _tr(ctx, a, "gt", "real", what), color=COLORS["truth"], lw=1.5, label="measured")
                ax.plot(t[k0 - 1:], _tr(ctx, a, "pred", "real", what)[k0 - 1:], color=COLORS["real"], lw=1.8,
                        label="prediction, measured actuators")
                ax.plot(t[k0 - 1:], _tr(ctx, a, "pred", key, what)[k0 - 1:], color=COLORS["scenario"], lw=1.8,
                        ls="--", label="prediction, changed actuator")
                ax.set_ylabel(LABELS[what][0])
                _label_trace(ax, ctx, what, a.font)
            else:
                n, which = what
                _spec(ctx, ax, d[f"spectro_pred__{which}__{n}"], "real" if which == "real" else "scenario", name=n)
                if n in ylim:
                    ax.set_ylim(0, ylim[n])
                ax.set_ylabel("freq (kHz)" if c == 0 else "")
                # "Balanced injection" -> "balanced injection", but "ECH off" keeps its acronym.
                tag = ("measured actuators" if which == "real"
                       else label if label.split()[0].isupper() else label.lower())
                label_box(ax, f"{spectro_label(d, n)}, {tag}", a.font)
            ax.axvline(ctx["t_switch"], color=COLORS["scenario"], lw=1.0, ls=":")
            ax.set_xlim(t[0], a.tmax or t[-1])
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            if kind != "spec":
                # Scale y to the shown window, not to data cut off by --tmax.
                t_end = a.tmax or t[-1]
                ys = np.concatenate([np.asarray(ln.get_ydata(), float)[np.asarray(ln.get_xdata(), float) <= t_end]
                                     for ln in ax.get_lines() if len(ln.get_ydata()) > 2])
                lo, hi = np.nanmin(ys), np.nanmax(ys)
                pad = 0.08 * (hi - lo + 1e-9)
                floor = kind == "act" or what in ZERO_BASED
                ax.set_ylim(min(lo - pad, 0) if floor else lo - pad, hi + pad)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
        if key in TIMESCALES:
            _tau_box(ax, ctx, key)


def plot_scenarios(a):
    ctx = _setup(a)
    if a.layout == "half":
        fig = plt.figure(figsize=(a.width, a.height), constrained_layout=True)
        _draw_scenario_half(ctx, a, fig, [c for c in a.columns.split(",") if c][0])
        _save(fig, a, ctx)
        return
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
        optional = key in OPTIONAL_RUNS
        e.add_argument(f"--{key}", required=not optional, default="",
                       help=f"evaluation folder of the {key} rollout" + (" (optional)" if optional else ""))
    e.add_argument("--donor", action="append", default=[], metavar="SHOT=DIRS",
                   help="rollouts under another shot's actuators (actuator mode donor:SHOT), "
                        "repeatable; DIRS as for --real")
    e.add_argument("--spectro", default="ece", help="spectrograms, comma-separated (ece, mhr, bes, co2, mirnov); "
                   "the first is the one in the scenario panels")
    e.add_argument("--spectro_channel", default="", help="channel per spectrogram, e.g. co2=1,bes=11 "
                   "(default: the channel whose frame means vary most; see analysis/demo_spectro_channels.py)")
    e.add_argument("--pred", choices=("pred", "mean"), default="pred",
                   help="prediction to show: sampled codes, or the expected frames (mean__*)")
    e.add_argument("--spectro_member", default="mean",
                   help="spectrograms of an ensemble: the decoded mean, or the one member closest to it "
                   "(keeps the measured texture and mode-line wander; traces stay the mean), or "
                   "nearest:<modality>, the member closest to the measurement on that modality")
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
    q.add_argument("--style", choices=("talk", "paper"), default="talk",
                   help="talk: seaborn talk theme (presentation); paper: the plain small-font look")
    q.add_argument("--font", type=int, default=11, help="font size of --style paper")
    q.add_argument("--width", type=float, default=16)
    q.add_argument("--height", type=float, default=13)
    q.add_argument("--dpi", type=int, default=200)
    q.add_argument("--layout", choices=("full", "half"), default="full",
                   help="half: compact figure for half a 16:9 slide (validation, or one scenario column)")
    q.add_argument("--sim_spectro", default="", help="spectrograms in the scenario panels of --layout half, "
                   "e.g. mhr; default: every spectrogram in the data file")
    q.add_argument("--tmax", type=float, default=0.0, help="end of the time axis (s); 0: end of the data")
    a = p.parse_args()
    if a.stage == "plot" and a.layout == "half" and (a.width, a.height) == (16, 13):
        a.width, a.height = 6.4, 6.8  # inches: shown at about this size on half a slide
    elif a.stage == "plot" and a.style == "talk" and (a.width, a.height) == (16, 13):
        a.width, a.height = 24, 20  # room for the talk theme's 18 pt text
    if a.stage == "extract":
        extract(a)
    else:
        plot(a)


if __name__ == "__main__":
    main()
