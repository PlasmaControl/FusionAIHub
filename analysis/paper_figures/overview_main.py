"""Overview figures of the IGNITE paper: ``overview_main`` (panels A and B)
and ``overview_shot_204011`` (the shot rows, formerly panel C).

From a 1 s measured initial state and an actuator plan, IGNITE generates
every diagnostic. With the measured plan it replays the discharge; with a
changed plan it simulates a discharge that was never run. The figure shows
this as a fork at the time the plan changes.

Panels
------
A
    DIII-D cross-section (EFIT01 wall and flux surfaces, fetched with
    ``fetch_geqdsk.py``), actuators in on the left, diagnostics out on the
    right.
Timescale bar
    Sampling interval of each modality (from the dataset config), the
    frame and the rollout length.
B
    Filmstrip: measured frames, IGNITE, predicted frames, the actuator
    plan underneath and the autoregressive feedback. Pictograms, arrow,
    flame and wordmark are the vector artwork of
    ``overview/ignite_logos/ignite-overview.svg`` (from
    ``ignite-logos.zip``), recoloured to the modality palette.
Shot rows (``overview_shot_204011``)
    Shot 204011: torque, one CER chord, one Mirnov channel, one
    filterscope and the lower divertor camera, measured, replayed and
    under the changed plan (torque ramped to zero through the beam mix).

Data (no rollouts are run here):

* ``ens4_soup/ramps/demo_204011_ramps_data.npz``: actuators and the
  measured profile traces.
* ``one_204011_cpu_data.npz``: every modality of one rollout (seed 3) of the
  replay and the changed plan (``claude_scratch/efit/extract_one.sh``,
  decode only); the profile traces.
* ``rank_204011.npz``: Mirnov spectrograms of one rollout (the medoid
  seed).
* ``fs_204011_data.npz``: filterscope envelopes and camera frames of
  every seed (``demo_controllability_fig.py extract --filterscopes
  --camera_member``).
* ``geqdsk_204011.npz``: EFIT01 equilibrium.

Run on a login node::

    pixi run --frozen -e stellarai \\
        python analysis/paper_figures/overview_main.py

writes ``overview_main.{pdf,png}``, ``overview_shot_204011.{pdf,png}`` and
their ``*_source_data.npz`` to ``analysis/paper_figures/figures/``.
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    FancyArrowPatch, PathPatch, Rectangle)
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import (  # noqa: E402
    Affine2D, blended_transform_factory)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures as mf  # noqa: E402

dc = mf.dc

# --------------------------------------------------------------- constants
FIG_W_CM = 16.51           # 6.5 in, the text width of main.tex
FIG_H_CM = 6.6             # overview_main: panels A and B
SHOT_H_CM = 15.5           # SHOT_NAME: the shot rows (old panel C); fits a page with its caption
MIN_FONT_PT = 7
SHOT = 204011
T_SEED = (1.0, 2.0)
T_CHANGE = 3.5
T_END = 6.0

NAME = "overview_main"
SHOT_NAME = f"overview_shot_{SHOT}"
CM = 1 / 2.54
FONT_PT = 8
LETTER_PT = 10
RASTER_DPI = 600
RIGHT = 0.955              # right edge of the timescale, panel B and panel C

# Trace colours and styles: measurement, replay (measured plan) and the
# changed plan. Changed and replay differ in line style as well as colour.
TRACE = {"measured": "#16191d", "replay": "#0072b2", "changed": "#c1440e"}
TRACE_LS = {"measured": "-", "replay": "-", "changed": (0, (3.0, 1.6))}
TRACE_LW = {"measured": 1.3, "replay": 1.0, "changed": 1.0}
TRACE_LABEL = {"measured": "measured", "replay": "replay (measured plan)",
               "changed": "prediction, changed plan"}
END_LABEL = {"measured": "measured", "replay": "replay",
             "changed": "changed plan"}
# Modality colours, for labels and blocks only.
MODALITY = {"profiles": "#1b7f79", "filterscopes": "#b0306a",
            "spectrograms": "#4b47a8", "camera": "#5a5a5a",
            "actuators": "#2b3440"}
CHANGE_TINT = "#fbf0ea"    # background after the plan change
SEED_SHADE = "#e6e8eb"
WALL_FILL = "#f3f4f6"
FLUX_LINE = "#a3aab2"

# Runs in the data files: measured plan and changed plan (torque linearly
# to zero over 1 s through the beam mix, from T_CHANGE).
REPLAY_RUN = "real"
CHANGED_RUN = "tq_lin_zero"
TORQUE = "torque (N m)"
POWER = "NBI (MW)"
PROFILE = "cer_rot"
DENSITY = "ts_core_density"   # second profile row, chord from PROFILE_CHORDS
PROFILE_CHORDS = mf.CHORDS["204011_torque"]
SPECTRO = "mhr"
# MHR B4: its broadband bursts are the ELMs (they fade under the changed plan, as the
# D-alpha spikes do), and the coherent mode slows with the rotation. claude_scratch/elm_spectro/
SPECTRO_CH = 1
CAMERA = mf.CAM
CAMERA_T = 5.0             # time of the camera frames (s)
CAMERA_SIGMA_PX = 1.5
CAMERA_H_FRAC = 0.84       # camera images: at most this fraction of the row
CAMERA_W_FRAC = 0.225      # ... and of the width each
FS_CHANNEL = "FS04"        # largest response to the changed plan
SPEC_PCT = (20.0, 99.7)    # spectrogram colour limits (measured, live); the
                           # lower limit hides the broadband background
SPEC_FMAX_KHZ = 100.0      # frequency range shown: the mode lines, enlarged
CAMERA_PCT = (1.0, 99.7)   # camera colour limits (all frames shown)
DIFF_PCT = 99.5            # camera difference: symmetric colour limit
SPEC_ARROW_T = 4.5         # changed-plan mode ridge marked at this time (s)
SPEC_ARROW_FMAX = 30.0     # ... searched below this frequency (kHz)
SPEC_TEXT_T = 4.95         # left edge of the annotation (s)
FS_UNIT = 1e16
LABEL_FRAMES = 20          # end labels sit at the mean of the last frames
FS_SPIKE_LW = 0.35         # filterscope spike line width (pt)

# Equilibrium for panel A.
EQ_TIME_S = 3.0
PSI_LEVELS = (0.2, 0.4, 0.6, 0.8)

# Modality groups of the timescale bar (dataset names).
GROUPS = {
    "spectrograms": ("ece", "bes", "mhr", "co2", "mirnov"),
    "filterscopes": ("filterscopes",),
    "profiles": ("ts_core_temp", "ts_core_density", "ts_tangential_temp",
                 "ts_tangential_density", "cer_ti", "cer_rot", "mse"),
    "camera": ("tangtv_lower", "tangtv_upper"),
}
ZOOM_FRAME = 28            # rollout frame enlarged in the timescale
ZOOM_X0 = 0.4              # left edge of the enlarged frame (axes)
DENSE_N = 1000             # above this, samples are drawn as a fill

# Panel A schematic: actuator inputs (left) and diagnostic outputs (right)
# at heights Z (m), with an optional second line of text.
INPUTS = (("neutral\nbeams", 0.95), ("ECH", 0.35), ("gas", -0.25),
          ("coils", -0.85))
OUTPUTS = (("profiles", "Thomson, CER, MSE", 0.95),
           ("spectrograms", "ECE, BES, interferometer,\nmagnetics", 0.42),
           ("filterscopes", "edge emission", -0.42),
           ("camera", "lower and upper divertor", -1.0))
ARROW_LEN_M = 0.32         # arrow length outside the wall (m)
TEXT_LEFT_M = 0.85         # room for the input labels (m)
TEXT_RIGHT_M = 1.95        # room for the output labels (m)

# Panel B schematic, in its own axis units.
N_MEASURED = 3
N_PREDICTED = 3
FRAME_W, FRAME_GAP, BLOCK_GAP, ARROW_GAP = 6.0, 4.0, 0.6, 5.0
LIGHTEN = 0.45             # predicted frames: blend this far to white

# IGNITE artwork (vector, from ignite-logos.zip). The overview drawing's
# modality colours (outline, tint) are mapped to MODALITY.
ART_SVG = mf.REPO / "overview" / "ignite_logos" / "ignite-overview.svg"
ART_ROWS = ("profiles", "filterscopes", "spectrograms", "camera")
ART_FAMILY = {"profiles": ("#4D3C98", "#F1EEFA"),
              "filterscopes": ("#B35F1C", "#FDF1E6"),
              "spectrograms": ("#9E2A22", "#FCECEB"),
              "camera": ("#0E6B66", "#EDFBF9")}
ART_INK = "#2C2E31"
ART_ARROW_FILL = "#EEF6EA"
ART_ARROW_BODY = (407.5 - 365.5) / 300.0   # body bottom, fraction of height
ART_TINT = 0.86            # pictogram background: blend towards white
ART_CAMERA_LIFT = 0.4      # camera shapes on the dark frame: lighter grey

DATA_RAMPS = mf.DATA / "ramps" / f"demo_{SHOT}_ramps_data.npz"
DATA_FS = mf.WORK / f"fs_{SHOT}_data.npz"
DATA_ONE = mf.WORK / f"one_{SHOT}_cpu_data.npz"
DATA_EQ = mf.WORK / f"geqdsk_{SHOT}.npz"


# ----------------------------------------------------------------- helpers
def style():
    """Sans-serif paper style at the final print size."""
    # Register Nimbus Sans (the Helvetica clone) explicitly: matplotlib's font cache can be
    # rebuilt on a node without the system fonts, and the figure then falls back to the
    # wider DejaVu Sans, whose labels collide.
    from glob import glob
    from matplotlib import font_manager
    for path in glob("/usr/share/fonts/urw-base35/NimbusSans-*.otf"):
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold",
        "mathtext.cal": "Nimbus Sans", "mathtext.sf": "Nimbus Sans",
        "font.size": FONT_PT, "axes.labelsize": FONT_PT,
        "xtick.labelsize": MIN_FONT_PT, "ytick.labelsize": MIN_FONT_PT,
        "legend.fontsize": FONT_PT, "axes.titlesize": FONT_PT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "xtick.major.size": 2.0,
        "ytick.major.size": 2.0, "xtick.minor.size": 1.2,
        "axes.labelpad": 1.5, "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5, "axes.grid": False,
        "pdf.fonttype": 42, "savefig.dpi": RASTER_DPI,
        "image.interpolation": "none"})


def frame_index(t):
    """Frame index of time ``t`` (s)."""
    return int(round((t - T_SEED[0]) / mf.FRAME))


def lighten(color, amount=LIGHTEN):
    """``color`` blended towards white by ``amount``."""
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def channel_box(ax, text):
    """Channel name in the top-left corner of ``ax``."""
    ax.text(0.006, 0.96, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=MIN_FONT_PT, color="0.25", zorder=10,
            bbox={"facecolor": "white", "alpha": 0.85, "lw": 0, "pad": 0.6})


def row_label(fig, axes, text, color, sub=""):
    """Row label with a modality-colour swatch, left of ``axes``.

    ``sub`` is an optional second line in grey (e.g. view and time).
    """
    y0 = min(a.get_position().y0 for a in axes)
    y1 = max(a.get_position().y1 for a in axes)
    yc = 0.5 * (y0 + y1)
    h_cm = fig.get_size_inches()[1] / CM
    w, h = 0.006, 0.2 / h_cm
    fig.patches.append(Rectangle((0.006, yc - h / 2), w, h, color=color,
                                 transform=fig.transFigure, lw=0))
    fig.text(0.016, yc, text, ha="left", va="center", fontsize=FONT_PT,
             color=color, fontweight="bold")
    if sub:
        fig.text(0.016, yc - 0.2 / h_cm, sub, ha="left", va="top",
                 fontsize=MIN_FONT_PT, color="0.35", linespacing=1.1)


def panel_letter(fig, x, y, letter):
    """Bold panel letter at figure coordinates ``(x, y)``."""
    fig.text(x, y, letter, fontsize=LETTER_PT, fontweight="bold",
             ha="left", va="top")


def sampling_intervals():
    """Sampling interval (s) of each modality group, from the dataset config.

    Returns
    -------
    dict of str to list of float
        Distinct intervals ``1 / target_fs`` (signals) or
        ``1 / target_fps`` (movies) of the group's modalities.
    """
    from tokamak_foundation_model.data.data_loader import TokamakH5Dataset

    rate = {c.name: c.target_fs for c in TokamakH5Dataset.SIGNAL_CONFIGS}
    rate.update({c.name: c.target_fps
                 for c in TokamakH5Dataset.MOVIE_CONFIGS})
    return {g: sorted({1.0 / rate[n] for n in names})
            for g, names in GROUPS.items()}


# -------------------------------------------------------------------- data
def load_data():
    """Everything the figure plots, from the cached rollouts.

    Returns
    -------
    dict
        ``ctx``/``args`` (the figure scripts' context of the ramp file),
        ``fs`` (filterscope and camera file), ``spec`` (Mirnov images),
        ``member`` (the single rollout shown) and ``eq`` (equilibrium).
    """
    for path in (DATA_RAMPS, DATA_FS, DATA_ONE, DATA_EQ):
        if not path.exists():
            raise FileNotFoundError(path)
    # smooth=1: one value per 50 ms frame, the model's own resolution; no
    # moving average over time.
    ctx, args = mf.context(DATA_RAMPS, channels=PROFILE_CHORDS, smooth=1)
    if abs(ctx["t_switch"] - T_CHANGE) > 1e-9:
        raise ValueError(f"plan change at {ctx['t_switch']} s in the data")
    fs = np.load(DATA_FS, allow_pickle=True)
    best = mf.best_spectro(str(SHOT), key="replay", names=(SPECTRO,), n=1)
    if not best:
        raise FileNotFoundError(f"rank_{SHOT} has no {SPECTRO} ranking")
    ch = SPECTRO_CH if SPECTRO_CH is not None else best[0][1]
    scores = mf.rank(str(SHOT))[0]["scores"][SPECTRO]
    member = next(x for x in scores if x["ch"] == ch)[f"medoid:{REPLAY_RUN}"]
    if int(fs["camera_member"]) != member:
        raise ValueError("camera frames are not of the spectrogram rollout")
    # The single-rollout file must hold that same rollout.
    # Same chords and smoothing as the ramp file's context, other data.
    one_ctx, one_args = dict(ctx), args
    one_ctx["d"] = np.load(DATA_ONE, allow_pickle=True)
    for run in (REPLAY_RUN, CHANGED_RUN):
        a = np.asarray(one_ctx["d"][f"filterscopes_members__{run}"])[0]
        b = np.asarray(fs[f"filterscopes_members__{run}"])[member]
        # Decoded on another device: equal up to float rounding.
        if not np.allclose(a, b, rtol=1e-3, atol=1e-3 * float(np.abs(b).max())):
            raise ValueError(f"{DATA_ONE.name} is not rollout {member}")
    fmax, gt, preds = mf.spectro_images(str(SHOT), SPECTRO, ch,
                                        [REPLAY_RUN, CHANGED_RUN])
    spec = {"ch": ch, "fmax": fmax, "measured": gt,
            "replay": preds[REPLAY_RUN], "changed": preds[CHANGED_RUN],
            "name": mf.spectro_name(SPECTRO, ch)}
    eq = np.load(DATA_EQ)
    return {"ctx": ctx, "args": args, "fs": fs, "spec": spec,
            "member": member, "eq": eq, "one": (one_ctx, one_args)}


# ------------------------------------------------------------------ panels
def panel_a(fig, spec, data, src):
    """DIII-D cross-section with actuators in and diagnostics out.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    spec : SubplotSpec
        Area of the panel.
    data : dict
        Output of :func:`load_data`.
    src : dict
        Source data, filled in place.
    """
    eq = data["eq"]
    k = int(np.argmin(np.abs(eq["GTIME"] - 1e3 * EQ_TIME_S)))
    wall_r, wall_z = np.asarray(eq["XLIM"]), np.asarray(eq["YLIM"])
    nb = int(eq["NBBBS"][k])
    bdry_r, bdry_z = eq["RBBBS"][k][:nb], eq["ZBBBS"][k][:nb]
    psi = np.asarray(eq["PSIRZ"][k], dtype=float)
    psin = (psi - eq["SSIMAG"][k]) / (eq["SSIBRY"][k] - eq["SSIMAG"][k])
    rr, zz = np.meshgrid(eq["R"], eq["Z"])
    inside = MplPath(np.column_stack([bdry_r, bdry_z])).contains_points(
        np.column_stack([rr.ravel(), zz.ravel()])).reshape(rr.shape)
    psin_in = np.where(inside, psin, np.nan)

    ax = fig.add_subplot(spec)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.fill(wall_r, wall_z, color=WALL_FILL, lw=0, zorder=1)
    ax.plot(np.r_[wall_r, wall_r[:1]], np.r_[wall_z, wall_z[:1]],
            color=MODALITY["actuators"], lw=0.8, zorder=3)
    ax.contour(rr, zz, psin_in, levels=PSI_LEVELS, colors=FLUX_LINE,
               linewidths=0.4, zorder=2)
    ax.plot(np.r_[bdry_r, bdry_r[:1]], np.r_[bdry_z, bdry_z[:1]],
            color="0.35", lw=0.8, zorder=3)

    def wall_at(z, side):
        """Inner (side -1) or outer (side +1) wall crossing at height z."""
        r0, z0 = wall_r, wall_z
        r1, z1 = np.roll(wall_r, -1), np.roll(wall_z, -1)
        hit = (z0 - z) * (z1 - z) <= 0
        frac = (z - z0[hit]) / np.where(z1[hit] == z0[hit], 1.0,
                                        z1[hit] - z0[hit])
        r = r0[hit] + frac * (r1[hit] - r0[hit])
        return r.min() if side < 0 else r.max()

    left = wall_r.min() - ARROW_LEN_M
    right = wall_r.max() + ARROW_LEN_M
    for text, z in INPUTS:
        end = wall_at(z, -1)
        ax.add_patch(FancyArrowPatch(
            (left, z), (end, z), arrowstyle="-|>", mutation_scale=6,
            lw=0.9, color=MODALITY["actuators"], zorder=4,
            shrinkA=0, shrinkB=0))
        ax.text(left - 0.05, z, text, ha="right", va="center",
                fontsize=FONT_PT, color=MODALITY["actuators"])
    for key, sub, z in OUTPUTS:
        start = wall_at(z, +1)
        color = MODALITY[key]
        ax.add_patch(FancyArrowPatch(
            (start, z), (right, z), arrowstyle="-|>", mutation_scale=6,
            lw=0.9, color=color, zorder=4, shrinkA=0, shrinkB=0))
        ax.text(right + 0.06, z + 0.02, key, ha="left", va="bottom",
                fontsize=FONT_PT, color=color, fontweight="bold")
        ax.text(right + 0.06, z - 0.02, sub, ha="left", va="top",
                fontsize=MIN_FONT_PT, color="0.35")
    top = wall_z.max() + 0.12
    ax.text(left - TEXT_LEFT_M, top, "actuator plan", ha="left",
            va="bottom",
            fontsize=FONT_PT, color=MODALITY["actuators"], style="italic")
    ax.text(right + 0.06, top, "diagnostics", ha="left", va="bottom",
            fontsize=FONT_PT, color="0.25", style="italic")
    ax.set_xlim(left - TEXT_LEFT_M, right + TEXT_RIGHT_M)
    ax.set_ylim(wall_z.min() - 0.05, top + 0.3)
    src.update({"A_wall_R_m": wall_r, "A_wall_Z_m": wall_z,
                "A_lcfs_R_m": bdry_r, "A_lcfs_Z_m": bdry_z,
                "A_grid_R_m": np.asarray(eq["R"]),
                "A_grid_Z_m": np.asarray(eq["Z"]),
                "A_psin_inside_lcfs": psin_in,
                "A_equilibrium_time_ms": float(eq["GTIME"][k])})
    return ax


def timescale_bar(fig, spec, src):
    """Rollout of 80 frames, one frame enlarged with each group's samples.

    Linear in time on both bars: the rollout bar spans the prediction,
    the enlarged frame one frame, with one tick per sample of each
    modality group (a fill where ticks would merge).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    spec : SubplotSpec
    src : dict
        Source data, filled in place.
    """
    ax = fig.add_subplot(spec)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ink = TRACE["measured"]
    rollout = T_END - T_SEED[1]
    n_frames = int(round(rollout / mf.FRAME))
    src["bar_rollout_s"] = rollout
    src["bar_frame_s"] = mf.FRAME
    # Rollout bar: one cell per frame, the enlarged one dark.
    y0, h = 0.78, 0.1
    ax.add_patch(Rectangle((0, y0), 1, h, color="0.82", lw=0))
    ax.vlines(np.arange(1, n_frames) / n_frames, y0, y0 + h,
              color="white", lw=0.3)
    x0, x1 = ZOOM_FRAME / n_frames, (ZOOM_FRAME + 1) / n_frames
    ax.add_patch(Rectangle((x0, y0), x1 - x0, h, color=ink, lw=0))
    ax.text(0, y0 + h + 0.05, f"rollout: {rollout:g} s = {n_frames} frames",
            ha="left", va="bottom", fontsize=MIN_FONT_PT, color=ink,
            fontweight="bold")
    # Enlarged frame: one row per sampling interval.
    by_dt = {}
    for group, dts in sampling_intervals().items():
        for dt in dts:
            by_dt.setdefault(dt, []).append(group)
        src[f"bar_sampling_interval_s__{group}"] = np.asarray(dts)
    rows = sorted(by_dt.items(), reverse=True)
    row_h, row_gap = 0.11, 0.05
    top = 0.6
    bottom = top - len(rows) * row_h - (len(rows) - 1) * row_gap
    for xa, xb in ((x0, ZOOM_X0), (x1, 1.0)):
        ax.plot([xa, xb], [y0, top + 0.02], color="0.55", lw=0.4)
    ax.text(0.5 * (ZOOM_X0 + 1.0), bottom - 0.04,
            f"one frame: {mf.FRAME * 1e3:.0f} ms", ha="center", va="top",
            fontsize=MIN_FONT_PT, color=ink, fontweight="bold")
    for i, (dt, groups) in enumerate(rows):
        yt = top - i * (row_h + row_gap)
        n = int(round(mf.FRAME / dt))
        color = MODALITY[groups[0]]
        if n > DENSE_N:
            ax.add_patch(Rectangle((ZOOM_X0, yt - row_h), 1 - ZOOM_X0,
                                   row_h, color=color, lw=0))
        else:
            xs = ZOOM_X0 + (np.arange(n) + 0.5) / n * (1 - ZOOM_X0)
            ax.vlines(xs, yt - row_h, yt, color=color,
                      lw=0.8 if n < 50 else 0.12)
        dt_text = (f"{dt * 1e3:g} ms" if dt >= 1e-3 else
                   f"{dt * 1e6:g} µs")
        ax.text(ZOOM_X0 - 0.015, yt - row_h / 2,
                f"{', '.join(groups)}: every {dt_text} ({n:,} per frame)",
                ha="right", va="center", fontsize=MIN_FONT_PT, color=color)
    return ax


# ---------------------------------------------------------- IGNITE artwork
_TAG = re.compile(r"<(/?)(\w+)([^>]*?)/?>")
_ATTR = re.compile(r'([\w:-]+)="([^"]*)"')
_TOK = re.compile(r"[MmLlHhVvCcQqAaZz]"
                  r"|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def read_svg(path):
    """Drawable elements and clip rectangles of a flat SVG file.

    Parameters
    ----------
    path : Path
        SVG without nested groups other than ``<g clip-path=...>``.

    Returns
    -------
    elements : list of (str, dict, str or None)
        ``(tag, attributes, clip id)`` in document order.
    clips : dict of str to dict
        Attributes of the rectangle of each ``<clipPath>``.
    """
    elements, clips = [], {}
    clip, defining = None, None
    for m in _TAG.finditer(path.read_text()):
        close, tag, attrs = m.groups()
        a = dict(_ATTR.findall(attrs))
        if tag == "clipPath":
            defining = None if close else a["id"]
        elif defining is not None:
            clips[defining] = a
        elif tag == "g":
            clip = None if close else a.get("clip-path", "")[5:-1] or None
        elif not close and tag in ("rect", "circle", "line", "polyline",
                                   "path"):
            elements.append((tag, a, clip))
    return elements, clips


def _arc(p0, rx, ry, phi, large, sweep, p1, n=24):
    """Points of an SVG elliptical arc from ``p0`` to ``p1`` (no ``p0``)."""
    c, s = np.cos(np.radians(phi)), np.sin(np.radians(phi))
    d = (p0 - p1) / 2
    x1, y1 = c * d[0] + s * d[1], -s * d[0] + c * d[1]
    lam = x1 ** 2 / rx ** 2 + y1 ** 2 / ry ** 2
    if lam > 1:
        rx, ry = rx * np.sqrt(lam), ry * np.sqrt(lam)
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1 ** 2 - ry ** 2 * x1 ** 2
    k = np.sqrt(max(num, 0.0) / (rx ** 2 * y1 ** 2 + ry ** 2 * x1 ** 2))
    k = -k if large == sweep else k
    cx1, cy1 = k * rx * y1 / ry, -k * ry * x1 / rx
    centre = np.array([c * cx1 - s * cy1, s * cx1 + c * cy1]) + (p0 + p1) / 2
    a0 = np.arctan2((y1 - cy1) / ry, (x1 - cx1) / rx)
    a1 = np.arctan2((-y1 - cy1) / ry, (-x1 - cx1) / rx)
    da = a1 - a0
    if sweep and da < 0:
        da += 2 * np.pi
    elif not sweep and da > 0:
        da -= 2 * np.pi
    t = a0 + da * np.linspace(0, 1, n + 1)[1:]
    ex, ey = rx * np.cos(t), ry * np.sin(t)
    return np.column_stack([c * ex - s * ey, s * ex + c * ey]) + centre


def svg_path(d):
    """Matplotlib path of an SVG path string (M L H V C Q A Z, both cases).

    Parameters
    ----------
    d : str
        The ``d`` attribute.

    Returns
    -------
    matplotlib.path.Path
    """
    toks = _TOK.findall(d)
    verts, codes = [], []
    cur, start = np.zeros(2), np.zeros(2)
    i, cmd = 0, "M"

    def nums(n):
        nonlocal i
        out = np.array([float(x) for x in toks[i:i + n]])
        i += n
        return out

    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                verts.append(start.copy())
                codes.append(MplPath.CLOSEPOLY)
                cur = start.copy()
                continue
        base = cur if cmd.islower() else np.zeros(2)
        op = cmd.upper()
        if op == "M":
            cur = start = base + nums(2)
            verts.append(cur.copy())
            codes.append(MplPath.MOVETO)
            cmd = "l" if cmd.islower() else "L"
        elif op == "L":
            cur = base + nums(2)
            verts.append(cur.copy())
            codes.append(MplPath.LINETO)
        elif op in "HV":
            v = nums(1)[0]
            ax_ = 0 if op == "H" else 1
            cur = cur.copy()
            cur[ax_] = v + (cur[ax_] if cmd.islower() else 0.0)
            verts.append(cur.copy())
            codes.append(MplPath.LINETO)
        elif op in "CQ":
            n = 3 if op == "C" else 2
            pts = nums(2 * n).reshape(n, 2) + base
            verts.extend(pts)
            codes.extend([MplPath.CURVE4 if op == "C" else MplPath.CURVE3]
                         * n)
            cur = pts[-1].copy()
        elif op == "A":
            rx, ry, phi, large, sweep, x, y = nums(7)
            end = base + np.array([x, y])
            pts = _arc(cur, rx, ry, phi, int(large), int(sweep), end)
            verts.extend(pts)
            codes.extend([MplPath.LINETO] * len(pts))
            cur = end
    return MplPath(np.asarray(verts, float), codes)


def _rounded_rect(x, y, w, h, r, n=6):
    """Closed path of a rectangle with corner radius ``r``."""
    pts = []
    for cx, cy, a0 in ((x + w - r, y + r, -90), (x + w - r, y + h - r, 0),
                       (x + r, y + h - r, 90), (x + r, y + r, 180)):
        a = np.radians(a0 + np.linspace(0, 90, n))
        pts.extend(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]))
    pts.append(pts[0])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 2) \
        + [MplPath.CLOSEPOLY]
    return MplPath(np.asarray(pts), codes)


def element_path(tag, a):
    """Matplotlib path of one SVG element, in SVG coordinates."""
    f = lambda k, v=0.0: float(a.get(k, v))  # noqa: E731
    if tag == "rect":
        return _rounded_rect(f("x"), f("y"), f("width"), f("height"),
                             f("rx"))
    if tag == "circle":
        return MplPath.circle((f("cx"), f("cy")), f("r"))
    if tag == "line":
        return MplPath([[f("x1"), f("y1")], [f("x2"), f("y2")]])
    if tag == "polyline":
        pts = np.array(a["points"].replace(",", " ").split(), float)
        return MplPath(pts.reshape(-1, 2))
    path = svg_path(a["d"])
    m = re.match(r"translate\(([-\d.]+),([-\d.]+)\)", a.get("transform", ""))
    if m:
        path = MplPath(path.vertices + [float(m[1]), float(m[2])],
                       path.codes)
    return path


def art_color(color, light):
    """Overview colour mapped to this figure's palette.

    Each modality's outline and tint become that modality's colour and a
    pale tint of it; the overview's ink becomes the figure's ink. With
    ``light`` the result is lightened (predicted frames).
    """
    if color in (None, "none"):
        return None
    c = color.upper()
    out = color
    for key, (dark, tint) in ART_FAMILY.items():
        if c == dark:
            out = lighten(MODALITY[key], ART_CAMERA_LIFT) \
                if key == "camera" else MODALITY[key]
        elif c == tint:
            out = lighten(MODALITY[key], ART_TINT)
    if c == ART_INK:
        out = TRACE["measured"]
    return lighten(out, LIGHTEN) if light else out


def draw_svg(ax, items, xform, pt_per_unit, light=False, clip=None,
             recolor=art_color):
    """Draw SVG elements on ``ax`` through the affine ``xform``.

    Parameters
    ----------
    ax : Axes
    items : list of (str, dict)
        ``(tag, attributes)`` from :func:`read_svg`.
    xform : matplotlib.transforms.Affine2D
        SVG coordinates to data coordinates.
    pt_per_unit : float
        Points per SVG unit after ``xform``, for the stroke widths.
    light : bool
        Lighten every colour (predicted frames).
    clip : matplotlib.path.Path, optional
        Clip path in SVG coordinates.
    recolor : callable
        ``(colour, light) -> colour``.
    """
    trans = xform + ax.transData
    clip_patch = None
    if clip is not None:
        clip_patch = PathPatch(clip, transform=trans, fc="none", ec="none")
    for tag, a in items:
        fill = a.get("fill", "none" if tag in ("line", "polyline")
                     else "#000000")
        patch = PathPatch(
            element_path(tag, a), transform=trans,
            fc=recolor(fill, light) or "none",
            ec=recolor(a.get("stroke"), light) or "none",
            lw=float(a.get("stroke-width", 1.0)) * pt_per_unit,
            alpha=float(a.get("opacity", 1.0)), capstyle="round",
            joinstyle="round", zorder=3)
        ax.add_patch(patch)
        if clip_patch is not None:
            patch.set_clip_path(clip_patch)


def load_art():
    """Pictograms and the IGNITE arrow of the overview artwork.

    Returns
    -------
    dict
        ``"cells"``: ``{(row, column): (clip rect, [(tag, attrs)])}`` with
        rows in :data:`ART_ROWS` order and column 0 the context column;
        ``"arrow"``, ``"mark"``: the arrow and the flame and wordmark.
    """
    elements, clips = read_svg(ART_SVG)
    by_clip = {}
    for tag, a, clip in elements:
        if clip is not None:
            by_clip.setdefault(clip, []).append((tag, a))
    frames = {(a["x"], a["y"]): (tag, a) for tag, a, clip in elements
              if tag == "rect" and clip is None and "stroke" in a}
    xs = sorted({float(c["x"]) for c in clips.values()})
    ys = sorted({float(c["y"]) for c in clips.values()})
    cells = {}
    for cid, c in clips.items():
        x, y = float(c["x"]), float(c["y"])
        # Context and rollout rows are offset slightly in y: nearest row.
        col = xs.index(x)
        row = int(np.argmin([abs(y - yy) for yy in ys[::2]])) if \
            len(ys) == 2 * len(ART_ROWS) else ys.index(y)
        items = [frames[(c["x"], c["y"])]] + by_clip.get(cid, [])
        cells[(row, col)] = (c, items)
    k = next(i for i, (tag, a, _c) in enumerate(elements)
             if a.get("fill", "").upper() == ART_ARROW_FILL)
    return {"cells": cells, "arrow": [elements[k][:2]],
            "mark": [e[:2] for e in elements[k + 1:k + 3]]}


def panel_b(fig, spec):
    """Filmstrip: measured frames, IGNITE, predicted frames, plan.

    The pictograms, the arrow and the IGNITE flame and wordmark are the
    vector artwork of ``ignite-overview.svg``, recoloured to the modality
    palette.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    spec : SubplotSpec
    """
    art = load_art()
    ax = fig.add_subplot(spec)
    ax.axis("off")
    clip0 = art["cells"][(0, 0)][0]
    cw, ch = float(clip0["width"]), float(clip0["height"])
    block_h = FRAME_W * ch / cw
    n_rows = len(ART_ROWS)
    frame_h = n_rows * block_h + (n_rows - 1) * BLOCK_GAP
    y0, bar_y, bar_h = 7.8, 4.6, 2.0
    arrow_items = art["arrow"]
    a_ext = element_path(*arrow_items[0]).get_extents()
    arrow_w = frame_h * a_ext.width / a_ext.height
    xs_meas = [i * (FRAME_W + FRAME_GAP) for i in range(N_MEASURED)]
    arrow_x = xs_meas[-1] + FRAME_W + ARROW_GAP
    x_pred0 = arrow_x + arrow_w + ARROW_GAP
    xs_pred = [x_pred0 + i * (FRAME_W + FRAME_GAP)
               for i in range(N_PREDICTED)]
    x_end = xs_pred[-1] + FRAME_W
    top = y0 + frame_h + 7.0
    # Equal units in x and y: widen whichever range the panel's shape
    # leaves room for; the content sits at the right edge (aligned with
    # the timescale and panel C) and vertically centred.
    pos = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    w_pt = pos.width * fw * 72
    h_pt = pos.height * fh * 72
    xr, yr = x_end + 2.0, top
    pt_per_data = min(w_pt / xr, h_pt / yr)
    pad_x = (w_pt / pt_per_data - xr) / 2
    pad_y = (h_pt / pt_per_data - yr) / 2
    ax.set_xlim(-1.0 - 2 * pad_x, x_end + 1.0)
    ax.set_ylim(-pad_y, top + pad_y)

    def place(x, y, scale, x0, y0_svg, h_svg):
        """SVG box with top-left (x0, y0_svg) to data, bottom-left (x, y)."""
        return (Affine2D().translate(-x0, -(y0_svg + h_svg)).scale(
            scale, -scale).translate(x, y))

    def frame(x, col, light):
        for r in range(n_rows):
            clip, items = art["cells"][(r, col)]
            cx, cy = float(clip["x"]), float(clip["y"])
            s = FRAME_W / cw
            y = y0 + frame_h - (r + 1) * block_h - r * BLOCK_GAP
            draw_svg(ax, items[:1], place(x, y, s, cx, cy, ch),
                     s * pt_per_data, light=light)
            draw_svg(ax, items[1:], place(x, y, s, cx, cy, ch),
                     s * pt_per_data, light=light,
                     clip=element_path("rect", clip))

    n_cols = 1 + max(c for _r, c in art["cells"])
    for x in xs_meas:
        frame(x, 0, light=False)
    for i, x in enumerate(xs_pred):
        frame(x, 1 + min(i, n_cols - 2), light=True)

    # The IGNITE arrow spans the frame height; flame and wordmark inside,
    # as in the overview artwork.
    s = frame_h / a_ext.height
    xf = place(arrow_x, y0, s, a_ext.x0, a_ext.y0, a_ext.height)
    draw_svg(ax, arrow_items, xf, s * pt_per_data,
             recolor=lambda c, _l: {ART_ARROW_FILL: WALL_FILL}.get(
                 (c or "").upper(), TRACE["measured"] if c not in
                 (None, "none") else None))
    draw_svg(ax, art["mark"], xf, s * pt_per_data)
    arrow = dict(arrowstyle="-|>", mutation_scale=7, lw=0.8,
                 color=TRACE["measured"])
    # Feedback: the newest predicted frame becomes input to the next step.
    ax.add_patch(FancyArrowPatch(
        (xs_pred[-1] + FRAME_W / 2, y0 + frame_h + 0.6),
        (arrow_x + 0.45 * arrow_w, y0 + frame_h * 0.93),
        connectionstyle="arc3,rad=0.3", **arrow))
    ax.text(0.5 * (arrow_x + 0.45 * arrow_w + xs_pred[-1] + FRAME_W / 2),
            y0 + frame_h + 5.5, "fed back", ha="center", va="bottom",
            fontsize=MIN_FONT_PT, color="0.35")
    # Actuator plan under the whole strip, into the model. A second,
    # changed plan forks off part way through the prediction.
    plan_h = 2.6
    ax.add_patch(Rectangle((0, bar_y + bar_h - plan_h), x_end, plan_h,
                           color=MODALITY["actuators"], lw=0))
    ax.text(0.8, bar_y + bar_h - plan_h / 2,
            "measured plan → replay", ha="left", va="center",
            fontsize=MIN_FONT_PT, color="white", fontweight="bold")
    x_fork = xs_pred[0] + FRAME_W + FRAME_GAP / 2
    y_chg = bar_y + bar_h - 2 * plan_h - 0.5
    ax.add_patch(Rectangle((x_fork, y_chg), x_end - x_fork, plan_h,
                           color=TRACE["changed"], lw=0))
    ax.plot([x_fork, x_fork], [y_chg + plan_h, bar_y + bar_h - plan_h],
            color=TRACE["changed"], lw=0.8)
    ax.text(x_fork - 1.4, y_chg + plan_h / 2,
            "or changed plan → a discharge never run", ha="right",
            va="center", fontsize=MIN_FONT_PT, color=TRACE["changed"],
            fontweight="bold")
    body_bottom = y0 + frame_h * ART_ARROW_BODY
    ax.add_patch(FancyArrowPatch((arrow_x + 0.45 * arrow_w, bar_y + bar_h),
                                 (arrow_x + 0.45 * arrow_w, body_bottom),
                                 arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color=MODALITY["actuators"],
                                 shrinkA=1, shrinkB=1))
    ax.text(xs_meas[0], y0 + frame_h + 0.8, "measured", ha="left",
            va="bottom", fontsize=FONT_PT, color=TRACE["measured"])
    ax.text(xs_pred[0], y0 + frame_h + 0.8, "prediction", ha="left",
            va="bottom", fontsize=FONT_PT,
            color=TRACE["replay"])
    return ax


def _trace(ax, t, y, role, start=0):
    """One trace from frame ``start`` on, in the style of ``role``."""
    ax.plot(t[start:], y[start:], color=TRACE[role], ls=TRACE_LS[role],
            lw=TRACE_LW[role], zorder=4 if role == "measured" else 5,
            solid_capstyle="round")


def spread_bands(ax, ctx, args, n, t, k0, sw):
    """Band of the sampled rollouts per run: mean -/+ one standard deviation
    (``make_figures.band``), centred on the line, which is their mean."""
    for run, role, start in ((REPLAY_RUN, "replay", k0 - 1), (CHANGED_RUN, "changed", sw - 1)):
        b = mf.band(ctx, args, run, n)
        if b is None:
            continue
        lo, hi = mf.cut(ctx, b[0]), mf.cut(ctx, b[1])
        ax.fill_between(t[start:], lo[start:], hi[start:], color=TRACE[role], alpha=0.3,
                        lw=0, zorder=2.5)


def end_labels(ax, t, ys, min_gap=None):
    """Name each trace at its right end, nudged apart vertically.

    Parameters
    ----------
    ax : Axes
    t : ndarray
        Frame times (s).
    ys : dict of str to ndarray
        Trace per role; NaN after the plasma end.
    min_gap : float, optional
        Smallest vertical distance between labels, in axes fraction;
        by default one label height.
    """
    lo, hi = ax.get_ylim()
    if min_gap is None:
        h_pt = ax.get_position().height * ax.figure.get_size_inches()[1] * 72
        min_gap = 1.1 * MIN_FONT_PT / h_pt
    items = []
    for role, y in ys.items():
        live = np.flatnonzero(np.isfinite(y))
        k = int(live[-1])
        level = float(np.mean(y[live[-LABEL_FRAMES:]]))
        items.append([(level - lo) / (hi - lo), role, t[k]])
    items.sort()
    # Keep the lowest label clear of the x axis, then stack upwards.
    items[0][0] = max(items[0][0], 0.75 * min_gap)
    for i in range(1, len(items)):
        items[i][0] = max(items[i][0], items[i - 1][0] + min_gap)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for yf, role, tk in items:
        ax.text(tk + 0.04, yf, END_LABEL[role],
                transform=trans, ha="left", va="center",
                fontsize=MIN_FONT_PT, color=TRACE[role])


def phase_header(ax):
    """Phases above the top row: initial state, prediction, plan change."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for (a, b), text, color in (
            (T_SEED, "measured initial state", "0.45"),
            ((T_SEED[1], T_END), "IGNITE prediction", TRACE["measured"])):
        ax.plot([a + 0.02, b - 0.02], [1.12, 1.12], color=color, lw=0.8,
                transform=trans, clip_on=False, solid_capstyle="butt")
        ax.text(0.5 * (a + b), 1.17, text, transform=trans, ha="center",
                va="bottom", fontsize=FONT_PT, color=color)
    ax.text(T_CHANGE - 0.04, 0.06, "plan changed", transform=trans,
            ha="right", va="bottom", fontsize=MIN_FONT_PT,
            color=TRACE["changed"], fontweight="bold")


def time_marks(ax, dark=True):
    """Seed interval, changed-plan background and the plan change line."""
    ax.axvspan(*T_SEED, color=SEED_SHADE, lw=0, zorder=0)
    ax.axvspan(T_CHANGE, T_END, color=CHANGE_TINT, lw=0, zorder=0)
    ax.axvline(T_CHANGE, color=TRACE["changed"] if dark else "white",
               lw=0.7, ls=(0, (2.5, 1.5)), zorder=6)
    ax.set_xlim(T_SEED[0], T_END)


def panel_c(fig, spec, data, src):
    """Shot rows: torque, profile, spectrograms, filterscope, camera.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    spec : SubplotSpec
    data : dict
        Output of :func:`load_data`.
    src : dict
        Source data, filled in place.

    Returns
    -------
    dict
        Checks: maximum difference between replay and changed plan before
        ``T_CHANGE`` per row.
    list of Axes
    """
    ctx, args, fs, sp = data["ctx"], data["args"], data["fs"], data["spec"]
    d, t, k0, sw = ctx["d"], ctx["t"], ctx["k0"], ctx["sw"]
    end = mf.live_end(ctx)
    checks = {}
    rows = spec.subgridspec(
        6, 1, height_ratios=[1.45, 1.65, 2.4, 1.5, 0.14, 0.8], hspace=0.22)
    src["C_time_s"] = t

    # 1. Actuators: injected torque and beam power, measured and changed
    # plan. The torque is removed through the beam mix, so the power
    # changes with it.
    act = rows[0].subgridspec(2, 1, hspace=0.18)
    axes = []
    for i, (name, text) in enumerate(((TORQUE, "injected torque (N m)"),
                                      (POWER, "neutral-beam power (MW)"))):
        ax = fig.add_subplot(act[i], sharex=axes[0] if axes else None)
        meas = mf.cut(ctx, d[f"act__{REPLAY_RUN}__{name}"])
        chg = mf.cut(ctx, d[f"act__{CHANGED_RUN}__{name}"])
        _trace(ax, t, meas, "measured")
        _trace(ax, t, chg, "changed", start=sw - 1)
        ax.set_ylabel(dc.ACT_SHOWN[name].split(" (")[0])
        mf.headroom(ax, 0.35, zero=True)
        ax.yaxis.set_major_locator(plt.MaxNLocator(2, integer=True))
        time_marks(ax)
        plt.setp(ax.get_xticklabels(), visible=False)
        channel_box(ax, text)
        key = "C1_torque_Nm" if name == TORQUE else "C1_nbi_MW"
        checks[key.split("_")[1]] = float(np.nanmax(np.abs(meas[:sw]
                                                            - chg[:sw])))
        src.update({f"{key}__measured_plan": meas,
                    f"{key}__changed_plan": chg})
        axes.append(ax)
    trans = blended_transform_factory(axes[0].transData,
                                      axes[0].transAxes)
    k_end = int(np.flatnonzero(np.isfinite(
        src["C1_torque_Nm__measured_plan"]))[-1])
    axes[0].text(t[k_end] + 0.04, src["C1_torque_Nm__measured_plan"][k_end],
                 "measured", ha="left", va="bottom",
                 fontsize=MIN_FONT_PT, color=TRACE["measured"])
    axes[0].text(4.55, 0.1, "changed plan: beams rebalanced",
                 transform=trans, ha="left",
                 va="bottom", fontsize=MIN_FONT_PT, color=TRACE["changed"])
    phase_header(axes[0])
    row_label(fig, axes, "Actuator", MODALITY["actuators"])
    ax = axes[0]

    # 2. Profiles: |v_tor| on one CER chord and n_e on one TS chord, the
    # mean of the sampled rollouts, band: one standard deviation.
    prof = rows[1].subgridspec(2, 1, hspace=0.18)
    ax = fig.add_subplot(prof[0], sharex=axes[0])
    one, one_args = data["one"]
    y = {"measured": mf.cut(ctx, dc._tr(ctx, args, "gt", REPLAY_RUN,
                                        PROFILE)),
         "replay": mf.cut(ctx, mf.pred_trace(ctx, args, REPLAY_RUN, PROFILE)),
         "changed": mf.cut(ctx, mf.pred_trace(ctx, args, CHANGED_RUN, PROFILE))}
    _trace(ax, t, y["measured"], "measured")
    _trace(ax, t, y["replay"], "replay", start=k0 - 1)
    _trace(ax, t, y["changed"], "changed", start=sw - 1)
    spread_bands(ax, ctx, args, PROFILE, t, k0, sw)
    ax.set_ylabel(r"$|v_{tor}|$")
    mf.headroom(ax, 0.3)
    time_marks(ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    end_labels(ax, t, y)
    chord = mf.chord_label(ctx, PROFILE).replace(",\n", ", ")
    channel_box(ax, f"toroidal rotation (km/s), {chord}")
    raw_r = np.asarray(one["d"][f"pred__{REPLAY_RUN}__{PROFILE}"])[:sw]
    raw_c = np.asarray(one["d"][f"pred__{CHANGED_RUN}__{PROFILE}"])[:sw]
    checks["profile"] = float(np.nanmax(np.abs(raw_r - raw_c)))
    for role, v in y.items():
        src[f"C2_vtor_kms__{role}"] = v
    src["C2_cer_chord_index"] = np.asarray(ctx["channels"][PROFILE])
    axes.append(ax)

    ax = fig.add_subplot(prof[1], sharex=axes[0])
    ne = {"measured": mf.cut(ctx, dc._tr(ctx, args, "gt", REPLAY_RUN,
                                         DENSITY)),
          "replay": mf.cut(ctx, mf.pred_trace(ctx, args, REPLAY_RUN, DENSITY)),
          "changed": mf.cut(ctx, mf.pred_trace(ctx, args, CHANGED_RUN, DENSITY))}
    _trace(ax, t, ne["measured"], "measured")
    _trace(ax, t, ne["replay"], "replay", start=k0 - 1)
    _trace(ax, t, ne["changed"], "changed", start=sw - 1)
    spread_bands(ax, ctx, args, DENSITY, t, k0, sw)
    ax.set_ylabel(r"$n_e$")
    mf.headroom(ax, 0.45)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    time_marks(ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    channel_box(ax, "electron density (10$^{19}$ m$^{-3}$), "
                    f"{mf.chord_label(ctx, DENSITY).replace(chr(44) + chr(10), chr(44) + chr(32))}")
    raw_r = np.asarray(one["d"][f"pred__{REPLAY_RUN}__{DENSITY}"])[:sw]
    raw_c = np.asarray(one["d"][f"pred__{CHANGED_RUN}__{DENSITY}"])[:sw]
    checks["density"] = float(np.nanmax(np.abs(raw_r - raw_c)))
    for role, v in ne.items():
        src[f"C2_ne_1e19m3__{role}"] = v
    src["C2_ts_chord_index"] = np.asarray(ctx["channels"][DENSITY])
    row_label(fig, axes[-1:] + [ax], "Profile", MODALITY["profiles"])
    axes.append(ax)

    # 3. Spectrograms: one Mirnov channel, the shown rollout, common limits.
    sub = rows[2].subgridspec(3, 1, hspace=0.08)
    ylim = min(SPEC_FMAX_KHZ, mf.spectro_ylim(sp["measured"], sp["fmax"],
                                              cap=mf.SPEC_CAP[SPECTRO]))
    n_f, n_t = sp["measured"].shape
    f_khz = (np.arange(n_f) + 0.5) * sp["fmax"] / n_f
    keep = f_khz <= ylim
    t_img = T_SEED[0] + (np.arange(n_t) + 0.5) * (t[-1] + mf.FRAME
                                                    - T_SEED[0]) / n_t
    live_cols = (t_img >= T_SEED[1]) & (t_img < t[0] + end * mf.FRAME)
    vmin, vmax = np.nanpercentile(sp["measured"][keep][:, live_cols],
                                  SPEC_PCT)
    cmap = LinearSegmentedColormap.from_list(
        "spectro", ["#ffffff", MODALITY["spectrograms"], "#120f33"])
    spec_axes = []
    for i, role in enumerate(("measured", "replay", "changed")):
        ax = fig.add_subplot(sub[i], sharex=axes[0])
        img = sp[role][keep]
        ax.imshow(img, aspect="auto", origin="lower", cmap=cmap, vmin=vmin,
                  vmax=vmax, extent=(t[0], t[-1] + mf.FRAME, 0, f_khz[keep][-1]
                                     + 0.5 * sp["fmax"] / n_f),
                  rasterized=True, zorder=1)
        ax.axvline(T_CHANGE, color=TRACE["changed"], lw=0.7,
                   ls=(0, (2.5, 1.5)), zorder=6)
        ax.axvspan(*T_SEED, color="white", alpha=0.45, lw=0, zorder=2)
        ax.set_xlim(T_SEED[0], T_END)
        ax.set_ylim(0, ylim)
        ax.yaxis.set_major_locator(plt.MaxNLocator(2, integer=True,
                                                   prune="upper"))
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.text(0.995, 0.92, TRACE_LABEL[role], transform=ax.transAxes,
                ha="right", va="top", fontsize=MIN_FONT_PT,
                color=TRACE[role], zorder=10,
                bbox={"facecolor": "white", "alpha": 0.85, "lw": 0,
                      "pad": 0.6})
        src[f"C3_{SPECTRO}_ch{sp['ch']}__{role}"] = img
        spec_axes.append(ax)
    spec_axes[1].set_ylabel("frequency (kHz)")
    channel_box(spec_axes[0], f"magnetics, {sp['name']}")
    # Arrow from the right onto the changed plan's mode ridge (strongest
    # frequency below the annotation height) at SPEC_ARROW_T.
    img = sp["changed"][keep]
    col = int(np.argmin(np.abs(t_img - SPEC_ARROW_T)))
    low = f_khz[keep] < SPEC_ARROW_FMAX
    f_mode = float(f_khz[keep][low][np.argmax(img[low, col])])
    spec_axes[2].annotate(
        "mode slows with the rotation",
        xy=(SPEC_ARROW_T, f_mode), xytext=(SPEC_TEXT_T, 0.3 * ylim),
        fontsize=MIN_FONT_PT, color=TRACE["changed"], ha="left",
        va="center", zorder=11,
        bbox={"facecolor": "white", "alpha": 0.85, "lw": 0, "pad": 0.6},
        arrowprops={"arrowstyle": "-|>", "color": TRACE["changed"],
                    "lw": 0.7, "mutation_scale": 6, "shrinkA": 1,
                    "shrinkB": 1})
    src["C3_mode_arrow_t_s_f_khz"] = np.asarray([SPEC_ARROW_T, f_mode])
    src["C3_frequency_khz"] = f_khz[keep]
    src["C3_time_s"] = t_img
    cols = int(round(sw * n_t / len(t)))
    checks["spectrogram"] = float(np.nanmax(np.abs(
        sp["replay"][:, :cols] - sp["changed"][:, :cols])))
    row_label(fig, spec_axes, "Spectrogram", MODALITY["spectrograms"])
    axes += spec_axes

    # 4. Filterscope: D-alpha spike amplitude per 10 ms bin (the codec's
    # RMS envelope), measured, replay and changed plan, one rollout.
    names = list(dc.channel_names("filterscopes"))
    c = names.index(FS_CHANNEL)
    raw = {"measured": np.asarray(fs["filterscopes_gt"], float),
           "replay": np.asarray(fs[f"filterscopes_members__{REPLAY_RUN}"],
                                float)[data["member"]],
           "changed": np.asarray(fs[f"filterscopes_members__{CHANGED_RUN}"],
                                 float)[data["member"]]}
    n_bin = raw["measured"].shape[2]
    if raw["measured"].shape[0] != len(t):
        raise ValueError("filterscope frames do not match the time axis")
    checks["filterscope"] = float(np.max(np.abs(
        raw["replay"][:sw] - raw["changed"][:sw])))
    t_fs = T_SEED[0] + (np.arange(len(t) * n_bin) + 0.5) * mf.FRAME / n_bin
    start = {"measured": 0, "replay": k0 * n_bin, "changed": sw * n_bin}
    stop = end * n_bin
    level = {r: v[:, c].ravel() / FS_UNIT for r, v in raw.items()}
    top_fs = max(float(np.max(v[start[r]:stop])) for r, v in level.items())
    sub = rows[3].subgridspec(3, 1, hspace=0.08)
    fs_axes = []
    for i, role in enumerate(("measured", "replay", "changed")):
        ax = fig.add_subplot(sub[i], sharex=axes[0])
        sl = slice(start[role], stop)
        ax.vlines(t_fs[sl], 0, level[role][sl], color=TRACE[role],
                  lw=FS_SPIKE_LW, zorder=4)
        if role != "measured":
            ax.vlines(t_fs[:start[role]], 0, level["measured"][
                :start[role]], color=TRACE["measured"], lw=FS_SPIKE_LW,
                zorder=4)
        ax.set_ylim(0, 1.15 * top_fs)
        ax.set_yticks([1])
        time_marks(ax)
        ax.text(0.995, 0.92, TRACE_LABEL[role], transform=ax.transAxes,
                ha="right", va="top", fontsize=MIN_FONT_PT,
                color=TRACE[role], zorder=10,
                bbox={"facecolor": "white", "alpha": 0.85, "lw": 0,
                      "pad": 0.6})
        if i < 2:
            plt.setp(ax.get_xticklabels(), visible=False)
        src[f"C4_{FS_CHANNEL}_rms_per_10ms_1e16__{role}"] = level[role]
        fs_axes.append(ax)
    src["C4_time_s"] = t_fs
    fs_axes[1].set_ylabel("D$_\\alpha$")
    channel_box(fs_axes[0], f"{FS_CHANNEL} edge D$_\\alpha$ per 10 ms "
                            "(10$^{16}$ ph cm$^{-2}$ sr$^{-1}$ s$^{-1}$)")
    ax = fs_axes[-1]
    ax.set_xlabel("time (s)", labelpad=0.5)
    row_label(fig, fs_axes, "Filterscope", MODALITY["filterscopes"])
    axes += fs_axes
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([CAMERA_T], [0.0], marker="^", ms=3.5, color=MODALITY["camera"],
            transform=trans, clip_on=False, zorder=8)

    # 5. Camera: one frame of the lower divertor at CAMERA_T, measured,
    # replay and changed plan (one rollout), and changed minus replay.
    ax = fig.add_subplot(rows[5])
    ax.axis("off")
    cam = {"measured": np.asarray(fs[f"camera_gt__{CAMERA}"], np.float32),
           "replay": np.asarray(fs[f"camera_member__{REPLAY_RUN}__{CAMERA}"],
                                np.float32),
           "changed": np.asarray(
               fs[f"camera_member__{CHANGED_RUN}__{CAMERA}"], np.float32)}
    crop = mf.crop_of(cam["measured"], k0)
    k = frame_index(CAMERA_T)
    imgs = {role: dc._camera_image(x[k], crop, CAMERA_SIGMA_PX)
            for role, x in cam.items()}
    checks["camera"] = float(np.max(np.abs(
        cam["replay"][:sw] - cam["changed"][:sw])))
    lo, hi = np.percentile(np.concatenate([v.ravel()
                                           for v in imgs.values()]),
                           CAMERA_PCT)
    diff = imgs["changed"] - imgs["replay"]
    lim = np.percentile(np.abs(diff), DIFF_PCT)
    diff_cmap = LinearSegmentedColormap.from_list(
        "diff", [TRACE["replay"], "#ffffff", TRACE["changed"]])
    pos = ax.get_position()
    fw, fh = fig.get_size_inches()
    ax_w_in = pos.width * fw
    ax_h_in = pos.height * fh
    aspect = imgs["measured"].shape[1] / imgs["measured"].shape[0]
    # As tall as the row allows, but four images must fit the width.
    w_frac = min(CAMERA_H_FRAC * ax_h_in * aspect / ax_w_in, CAMERA_W_FRAC)
    h_frac = w_frac * ax_w_in / (ax_h_in * aspect)
    gap = (1.0 - 4 * w_frac) / 3
    panels = (("measured", imgs["measured"], "gray", (lo, hi)),
              ("replay", imgs["replay"], "gray", (lo, hi)),
              ("changed", imgs["changed"], "gray", (lo, hi)),
              ("diff", diff, diff_cmap, (-lim, lim)))
    titles = {"measured": "measured", "replay": "replay (measured plan)",
              "changed": "prediction, changed plan",
              "diff": "changed plan minus replay"}
    for i, (role, im, cm, (v0, v1)) in enumerate(panels):
        # Images at the top of the row (titles above), the key below.
        ins = ax.inset_axes([i * (w_frac + gap), max(0.0, 0.86 - h_frac),
                             w_frac, h_frac],
                            zorder=7)
        ins.imshow(im, cmap=cm, vmin=v0, vmax=v1, aspect="auto",
                   interpolation="bilinear", rasterized=True)
        ins.set_xticks([])
        ins.set_yticks([])
        for s in ins.spines.values():
            s.set_linewidth(0.6)
            s.set_color(TRACE.get(role, "0.6"))
        ins.text(0.0, 1.04, titles[role], transform=ins.transAxes,
                 ha="left", va="bottom", fontsize=MIN_FONT_PT,
                 color=TRACE.get(role, TRACE["measured"]))
        src[f"C5_{CAMERA}_t{CAMERA_T:.1f}s__{role}"] = im
        if role == "diff":
            # Key below the image, clear of the emission.
            for yy, text, col in ((-0.06, "\u25a0 more light, changed plan",
                                   TRACE["changed"]),
                                  (-0.36, "\u25a0 less light, changed plan",
                                   TRACE["replay"])):
                ins.text(0.0, yy, text, transform=ins.transAxes,
                         ha="left", va="top", fontsize=MIN_FONT_PT,
                         color=col)
    sub = f"lower divertor,\nvisible light,\nt = {CAMERA_T:g} s (\u25b2)"
    row_label(fig, [ax], "Camera", MODALITY["camera"], sub=sub)
    axes.append(ax)
    return checks, axes


# -------------------------------------------------------------------- main
def smallest_font(fig):
    """Smallest font size (pt) of any visible, non-empty text."""
    fig.canvas.draw()
    sizes = [t.get_fontsize() for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    return min(sizes)


def save(fig, name, src):
    """Write ``name`` as PDF and PNG and its source data; return the size."""
    size_cm = fig.get_size_inches() / CM
    min_pt = smallest_font(fig)
    if min_pt < MIN_FONT_PT:
        raise ValueError(f"{name}: text at {min_pt} pt, below {MIN_FONT_PT} pt")
    out = mf.OUT
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", dpi=RASTER_DPI,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{name}.png", dpi=RASTER_DPI)
    plt.close(fig)
    np.savez_compressed(out / f"{name}_source_data.npz", **src)
    print(f"{name}: {size_cm[0]:.2f} x {size_cm[1]:.2f} cm, "
          f"smallest font {min_pt:g} pt")
    print("wrote", *(out / f"{name}{x}" for x in
                     (".pdf", ".png", "_source_data.npz")))


def main():
    """Build both figures and write their PDF, PNG and source data.

    ``overview_main`` holds panels A (cross-section, timescales) and B
    (filmstrip); ``SHOT_NAME`` holds the shot rows, each at a size
    readable on the page.
    """
    data = load_data()
    style()
    shot_only = "--shot-only" in sys.argv  # re-render only SHOT_NAME

    if not shot_only:
        # Figure 1: A left, timescale bar and B right.
        fig = plt.figure(figsize=(FIG_W_CM * CM, FIG_H_CM * CM))
        top = fig.add_gridspec(1, 2, width_ratios=[0.38, 0.62], wspace=0.06,
                               left=0.01, right=RIGHT, top=0.97, bottom=0.03)
        right = top[1].subgridspec(2, 1, height_ratios=[0.36, 0.64], hspace=0.3)
        src = {}
        ax_a = panel_a(fig, top[0], data, src)
        timescale_bar(fig, right[0], src)
        ax_b = panel_b(fig, right[1])
        pb = ax_b.get_position()
        panel_letter(fig, 0.005, 0.995, "A")
        # B sits just left of the first measured frame (the panel's axes are
        # wider than its content, which is centred for equal units).
        fig.canvas.draw()
        x_b = fig.transFigure.inverted().transform(
            ax_b.transData.transform((0.0, 0.0)))[0]
        # Above the frames' labels, clear of panel A's text.
        panel_letter(fig, x_b - 0.012, pb.y1 + 0.13, "B")
        save(fig, NAME, src)

    # Figure 2: the shot rows, full width.
    fig = plt.figure(figsize=(FIG_W_CM * CM, SHOT_H_CM * CM))
    bottom = fig.add_gridspec(1, 1, left=0.215, right=0.93, top=0.955,
                              bottom=0.045)
    src = {}
    checks, axes_c = panel_c(fig, bottom[0], data, src)
    fig.align_ylabels([a for a in axes_c if a.get_ylabel()])
    src.update({"meta_shot": SHOT, "meta_t_seed_s": np.asarray(T_SEED),
                "meta_t_change_s": T_CHANGE, "meta_t_end_s": T_END,
                "meta_rollout_member_shown": data["member"],
                "meta_changed_run": CHANGED_RUN})
    save(fig, SHOT_NAME, src)
    print(f"spectrogram {data['spec']['name']} (ch {data['spec']['ch']}), "
          f"filterscope {FS_CHANNEL}, rollout member "
          f"{data['member']} (seed {data['member'] + 1})")
    for row, diff in checks.items():
        print(f"replay vs changed before {T_CHANGE} s, {row}: "
              f"max |diff| = {diff:g}")


if __name__ == "__main__":
    main()
