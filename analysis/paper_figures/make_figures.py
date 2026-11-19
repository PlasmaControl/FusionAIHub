"""Results figures of the paper, from the soup's rollouts.

Model: ``soup_lu18_h16_30`` (1:1 weight average of ``ss75g_live_unroll@18000`` and
``ss75g_spec3_h16@30000``), Gumbel reveal noise 3, seeds 1-4; traces are the decoded mean
of the seeds, spectrograms the medoid seed. Data files of
``analysis/demo_controllability_fig.py extract`` in ``ens4_soup/`` (written by
``paper_rollouts.sh``). ``--gt_twice`` puts the measurement in every prediction slot
(layout check). Helpers (core channels, smoothing, labels) are that script's own.

Paper style: 6.5 in wide (main.tex's \\textwidth), 9 pt Times-like text (Nimbus Roman, STIX math), no titles, channel names in the
grey label box, little text. Palette A (colour-blind safe, see ``cvd_check.py``):
measured sky blue (thick, underneath), prediction orange (thin, on top), changed
actuator and its prediction purple (dashed).

Run on a login node::

    PYTHONPATH=/scratch/gpfs/EKOLEMEN/ps9551/ignite/pyvendor \
        pixi run --frozen -e stellarai python analysis/paper_figures/make_figures.py
"""
import argparse
import functools
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis"))
import demo_controllability_fig as dc  # noqa: E402

# Rollout extracts; PAPER_FIG_ENS points at another set (e.g. the 8-seed ens8_soup).
DATA = Path(os.environ.get("PAPER_FIG_ENS", "/scratch/gpfs/EKOLEMEN/ps9551/ignite/demo_figures_live_unroll/ens4_soup"))
GT_TWICE = "--gt_twice" in sys.argv
CAM = "tangtv_lower"
HERE = Path(__file__).resolve().parent
# PAPER_FIG_OUT renders another model's figures elsewhere (the paper's stay in figures/).
OUT = Path(os.environ.get("PAPER_FIG_OUT", HERE / "figures"))
# Intermediate data (rankings, measured gas, layout-check copies), 3 GB, kept out of git.
WORK = Path(os.environ.get("PAPER_FIG_DATA", "/scratch/gpfs/EKOLEMEN/ps9551/ignite/paper_figures/data"))
PALETTE = {"truth": "#56b4e9", "real": "#e07b1a", "scenario": "#7b3294"}
# \textwidth of main.tex (letter paper, 1 in margins) and its Times text (newtx).
WIDTH, FONT = 6.5, 9
# Heights (in) at which figures 1 and 2 fit one page with their captions in main.tex
# (letterpaper, 1 in margins, 12 pt); included at 1:1, never scaled.
HEIGHT_FIG1, HEIGHT_FIG2 = 5.65, 5.4
LW = {"truth": 2.6, "real": 1.3, "scenario": 1.3}
T0, FRAME = 1.0, 0.05


def style():
    # Register Nimbus Roman (the Times clone of the paper's text font) explicitly: the
    # matplotlib font cache can be rebuilt on a node without the system fonts, and the
    # figures then fall back to STIXGeneral.
    from glob import glob
    from matplotlib import font_manager
    for path in glob("/usr/share/fonts/urw-base35/NimbusRoman-*.otf"):
        font_manager.fontManager.addfont(path)
    dc.COLORS.update(PALETTE)
    plt.rcParams.update({
        "font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT - 0.5,
        "ytick.labelsize": FONT - 0.5, "legend.fontsize": FONT, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.labelpad": 2, "savefig.dpi": 300,
        "font.family": "serif", "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42})


def gt_twice(src: Path) -> Path:
    """Copy of a data file with every prediction replaced by the measurement."""
    d = np.load(src, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    for k in d.files:
        parts = k.split("__")
        if parts[0] in ("pred", "gt") and len(parts) == 3:  # <which>__<run>__<signal>
            out[k] = d[f"gt__real__{parts[2]}"]
        elif parts[0] == "spectro_pred":  # spectro_pred__<run>[__<name>]
            name = parts[2] if len(parts) > 2 else None
            out[k] = d[f"spectro_gt__{name}"] if name else d["spectro_gt"]
        elif parts[0] == "camera_pred":
            out[k] = d[f"camera_gt__{parts[-1]}"]
    dst = WORK / f"gt2_{src.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)
    return dst


def context(path, smooth=3, n_core=3, channels=""):
    """The figure script's context of a data file (chords, times, switch). ``channels``
    picks single chords, e.g. ``"cer_rot=1;ts_core_temp=42"`` (see ``CHORDS``)."""
    if isinstance(channels, dict):
        channels = ";".join(f"{n}={c}" for n, c in channels.items())
    a = argparse.Namespace(data=str(gt_twice(path) if GT_TWICE else path), smooth=smooth, n_core=n_core,
                           channels=channels,
                           camera="tangtv_lower", style="paper", font=FONT)
    ctx = dc._setup(a)
    ctx["src_name"] = Path(path).name  # the sidecar of sampled rollouts is keyed by it
    style()  # _setup applies the script's own rcParams; ours go on top
    return ctx, a


# Best single chords per figure, from rank_chords.py (replay: correlation with the
# measurement; scenarios: largest change against the
# measured-actuator prediction). Channel indices of the codec (TS core: row of TS_RZ,
# CER T<index + 1>).
CHORDS = {
    "204011": {"ts_core_temp": 41, "ts_core_density": 19, "cer_ti": 24, "cer_rot": 1},
    # n_e on the core chord (Z 0.03 m, as T_e): edge chords (Z > 0.6 m) respond to the beam
    # rebalancing about three times as strongly as the core (-19 % vs -7 % at ch 19 / 41).
    "204011_torque": {"cer_rot": 1, "cer_ti": 0, "ts_core_temp": 41, "ts_core_density": 41},
    # ECH of 204011: T_e and n_e on the innermost core chord (Z 0.03 m). Switching ECH off
    # raises the core n_e (density pump-out reversed, 8/8 seeds on the chords Z < 0.65 m);
    # the edge chord of Fig 1 (ch 19, Z 0.66 m) barely responds.
    "204011_ech": {"ts_core_temp": 41, "ts_core_density": 41, "cer_ti": 24, "cer_rot": 1},
    "201240_torque": {"cer_rot": 1, "cer_ti": 0, "ts_core_temp": 43, "ts_core_density": 21},
    "201240_ech": {"ts_core_temp": 43, "ts_core_density": 21, "cer_ti": 24, "cer_rot": 1},
    "201081": {"ts_core_temp": 41, "ts_core_density": 22, "cer_ti": 18, "cer_rot": 16},
    "201085": {"ts_core_temp": 42, "ts_core_density": 34, "cer_ti": 18, "cer_rot": 16},
    "200729": {"ts_core_temp": 42, "ts_core_density": 34, "cer_ti": 16, "cer_rot": 16},
}
# Thomson scattering chord positions per shot (system, index, R_m, Z_m, phi_deg), in the
# channel order of the data files. Shots with fewer core chords than the dataset's 44
# (204011 has 42) are zero-padded at the end: those slots are not channels.
TS_RZ = HERE / "figures" / "ts_rz.zip"


def ts_positions(shot, system="core"):
    """``(R, Z)`` in m of every chord of one Thomson system of ``shot``, in index order."""
    import csv
    import io
    import zipfile
    with zipfile.ZipFile(TS_RZ) as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(f"ts_rz/ts_rz_{shot}.csv"))))
    rows = sorted((r for r in rows if r["system"] == system), key=lambda r: int(r["index"]))
    if [int(r["index"]) for r in rows] != list(range(len(rows))):
        raise ValueError(f"ts_rz_{shot}.csv: {system} indices are not 0..n-1")
    return np.array([[float(r["R_m"]), float(r["Z_m"])] for r in rows])


def ts_z(ctx, chans):
    """Z (m) of TS core channels ``chans`` of the context's shot; refuses padded slots."""
    z = ts_positions(ctx["shot"])[:, 1]
    bad = [c for c in chans if c >= len(z)]
    if bad:
        raise ValueError(f"{ctx['shot']} has {len(z)} TS core chords; {bad} are padding")
    return z[list(chans)]


# CER tangential chords: major radius (cm) per chord and beam, from the tables in
# CER_GEOM (one file per shot range). The model uses the tangential chords T01-T48 only
# (codec index = chord - 1). A chord's radius is where its view crosses the beam; the
# left sources (30LT, 330LT) ran in the paper shots, so their radius is used.
CER_GEOM = HERE / "figures" / "cer_geometry"
CER_BEAMS = ("30LT", "30RT", "330LT", "330RT")
CER_PREFER = ("30LT", "330LT", "30RT", "330RT")
RAW = Path("/tigerdata/EKOLEMEN/d3d_fusion_data")


def cer_radii(shot):
    """``{"T01": {"30LT": R_m, ...}, ...}`` for the shot range that holds ``shot``."""
    import re
    shot = int(shot)
    for path in sorted(CER_GEOM.glob("*_CER_Chord_Major_Radii_(cm).txt")):
        lo, hi = path.name.split("_")[0].split("-")
        if int(lo) <= shot <= (10 ** 9 if hi == "Present" else int(hi)):
            words = path.read_text(encoding="utf-8").replace("\u200b", " ").split()
            out = {}
            for i, w in enumerate(words):
                if re.fullmatch(r"T\d\d", w):
                    vals = words[i + 1:i + 5]
                    out[w] = {b: float(v) / 100 for b, v in zip(CER_BEAMS, vals)
                              if re.fullmatch(r"\d+(\.\d+)?", v)}
            return out
    raise FileNotFoundError(f"no CER geometry table covers shot {shot}")


def cer_r(ctx, chans):
    """Major radius (m) of CER tangential channels ``chans`` (codec indices)."""
    radii = cer_radii(ctx["shot"])
    out = []
    for c in chans:
        beams = radii[f"T{c + 1:02d}"]
        out.append(next(beams[b] for b in CER_PREFER if b in beams))
    return np.array(out)


def cer_heights(shot):
    """``{"T02": {"30LT": Z_m, ...}, ...}``: the height where each tangential chord's view
    crosses each beam, the middle of its Zmax/Zmin (cm) in the CER_GEOM geometry table
    (``*_CER_Chord_Geometries.txt``) for the shot range that holds ``shot``. The tables
    write the chords without the leading zero (T2)."""
    import re
    shot = int(shot)
    for path in sorted(CER_GEOM.glob("*_CER_Chord_Geometries.txt")):
        lo, hi = path.name.split("_")[0].split("-")
        if not int(lo) <= shot <= (10 ** 9 if hi == "Present" else int(hi)):
            continue
        text = path.read_text(encoding="utf-8").replace("\u200b", " ")
        out = {}
        parts = re.split(r"For the (\w+) neutral beam", text)
        for beam, body in zip(parts[1::2], parts[2::2]):
            words = body.split()
            for i, w in enumerate(words):
                m = re.fullmatch(r"T(\d+)", w)
                num = r"[+-]?\d+(\.\d+)?"
                if m and i + 4 < len(words) and re.fullmatch(num, words[i + 3]) \
                        and re.fullmatch(num, words[i + 4]):
                    z = (float(words[i + 3]) + float(words[i + 4])) / 200
                    out.setdefault(f"T{int(m.group(1)):02d}", {}).setdefault(beam, z)
        return out
    raise FileNotFoundError(f"no CER geometry table covers shot {shot}")


def cer_z(ctx, chans):
    """Height (m) of CER tangential channels ``chans`` (codec indices), same beam as
    :func:`cer_r`."""
    radii, heights = cer_radii(ctx["shot"]), cer_heights(ctx["shot"])
    out = []
    for c in chans:
        name = f"T{c + 1:02d}"
        beam = next(b for b in CER_PREFER if b in radii[name])
        out.append(heights[name][beam])
    return np.array(out)


def cer_recorded(shot, n):
    """Codec indices of the CER channels of signal ``n`` that the shot recorded.

    Read once from the raw file's MDSplus node list (CERAUTO.TANGENTIAL) and cached in
    WORK. Channels without a node are zero in the data and their "measurement" is the
    codec's decoding of nothing.
    """
    import json
    cache = WORK / "cer_recorded.json"
    table = json.loads(cache.read_text()) if cache.exists() else {}
    if str(shot) not in table:
        import re
        import h5py
        with h5py.File(RAW / f"{shot}.h5", "r") as f:
            g = f[str(shot)]
            keys = [k for top in g for k in g[top].keys() if "CERAUTO.TANGENTIAL" in k]
        table[str(shot)] = {kind: sorted({int(re.search(r"CHANNEL(\d+)", k).group(1)) - 1
                                          for k in keys if k.endswith(f":{kind}")})
                            for kind in ("ROT", "TEMP")}
        cache.write_text(json.dumps(table, indent=1))
    return table[str(shot)]["ROT" if n == "cer_rot" else "TEMP"]


def chord_label(ctx, n):
    """Channel label of a trace: the channel and where it measures, R and Z (m)."""
    chans = ctx["channels"][n]
    if n.startswith("ts_core"):
        ts_z(ctx, chans)  # refuses padded slots
        rz = ts_positions(ctx["shot"])[list(chans)]
        nums = ", ".join(str(c) for c in chans)
        rs = ", ".join(f"{r:.2f}" for r in rz[:, 0])
        zs = ", ".join(f"{z:.2f}".replace("-", "\u2212") for z in rz[:, 1])
        return f"TS ch {nums} (R {rs}, Z {zs} m)"
    if n.startswith("cer_"):
        missing = [c for c in chans if c not in cer_recorded(ctx["shot"], n)]
        if missing:
            raise ValueError(f"{ctx['shot']} {n}: T{[c + 1 for c in missing]} not recorded")
        rs = ", ".join(f"{r:.2f}" for r in cer_r(ctx, chans))
        zs = ", ".join(f"{z:.2f}".replace("-0.00", "0.00").replace("-", "\u2212")
                       for z in cer_z(ctx, chans))
        return f"{dc.trace_label(n, chans)} (R {rs}, Z {zs} m)"
    return dc.trace_label(n, chans)


# Waterfall channel sets: every TS core chord; the CER tangential chords T01-T16 (one
# radial fan, core to edge).
WATERFALL = {"ts_core_temp": range(0, 44), "ts_core_density": range(0, 44)}
# CER profiles: outboard chords only (inboard views, R < 1.6 m, see a different projection of
# the rotation and would zigzag against the outboard ones at the same psi_N).
CER_R_MIN = 1.6


def cer_profile_chords(ctx):
    """Codec indices of the CER chords drawn in T_i and |v_tor| profiles: the same chords
    for both, i.e. every tangential chord with geometry, outboard of CER_R_MIN, that the
    shot recorded for both TEMP and ROT and whose measurement decodes to finite values in
    both (cer_rot T01/T03 have corrupt stats and decode to NaN)."""
    d, shot = ctx["d"], ctx["shot"]
    radii = cer_radii(shot)
    both = set(cer_recorded(shot, "cer_ti")) & set(cer_recorded(shot, "cer_rot"))
    out = []
    for c in sorted(both):
        if not any(b in radii.get(f"T{c + 1:02d}", {}) for b in CER_PREFER):
            continue
        if cer_r(ctx, [c])[0] < CER_R_MIN:
            continue
        if all(np.isfinite(np.asarray(d[f"gt__real__{n}"], float)[:, c]).any()
               for n in ("cer_ti", "cer_rot")):
            out.append(c)
    return np.array(out, dtype=int)


def live_end(ctx):
    """Last frame with plasma: core T_e above 30 % of its peak (204011 ends ~5.65 s)."""
    te = np.asarray(ctx["d"]["gt__real__ts_core_temp"], float)
    core = np.nanmax(te, axis=1)
    on = np.where(core > 0.3 * np.nanmax(core))[0]
    return int(on[-1]) + 1 if len(on) else te.shape[0]


def cut(ctx, y):
    """``y`` with the frames after the plasma end blanked."""
    y = np.array(y, dtype=float)
    y[live_end(ctx):] = np.nan
    return y


def profile_chords(ctx, n):
    """Chords of profile signal ``n`` the shot recorded and their positions (Z for TS, R
    for CER), sorted by position."""
    if n.startswith("ts_core"):
        chans = np.array(list(WATERFALL[n]))
        chans = chans[chans < len(ts_positions(ctx["shot"]))]
        xpos = ts_z(ctx, chans)
    else:
        chans = cer_profile_chords(ctx)
        xpos = cer_r(ctx, chans)
    order = np.argsort(xpos)
    return chans[order], xpos[order]


DASH = (0, (3, 1.5))


@functools.lru_cache(maxsize=None)
def _efit_slice(shot, k):
    """psi_N on the (Z, R) grid of EFIT01 slice ``k`` of ``shot``, as a spline."""
    from scipy.interpolate import RectBivariateSpline
    e = _efit(shot)
    pn = (e["PSIRZ"][k] - e["SSIMAG"][k]) / (e["SSIBRY"][k] - e["SSIMAG"][k])
    return RectBivariateSpline(e["Z"], e["R"], pn)


@functools.lru_cache(maxsize=None)
def _efit(shot):
    path = WORK / f"geqdsk_{shot}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run fetch_geqdsk.py {shot}")
    return dict(np.load(path))


def chord_psin(shot, n, chans, times):
    """psi_N of the chords ``chans`` of profile signal ``n`` at each of ``times`` (s),
    ``(len(times), len(chans))``, from the EFIT01 slice nearest each time
    (``geqdsk_<shot>.npz`` in WORK, written by ``fetch_geqdsk.py``). TS core chords sit
    at their scattering volume (R, Z); CER tangential chords where their view crosses
    the beam (:func:`cer_r`, :func:`cer_z`). Profiles of every figure are plotted against
    it; a changed-actuator prediction is placed on the measured equilibrium."""
    shot = str(shot)
    if n.startswith("ts"):
        rz = ts_positions(shot)[list(chans)]
    else:
        ctx = {"shot": shot}
        rz = np.stack([cer_r(ctx, chans), cer_z(ctx, chans)], 1)
    gt = _efit(shot)["GTIME"]
    out = []
    for tt in np.atleast_1d(times):
        sp = _efit_slice(shot, int(np.argmin(np.abs(gt - 1e3 * tt))))
        out.append(sp.ev(rz[:, 1], rz[:, 0]))
    return np.array(out)


PSIN_LABEL = r"$\psi_N$"


def smooth_profile(n, x):
    """(F, C) profile frames: absolute value where the signal is shown as one, then the
    3-frame running mean of the traces."""
    x = np.abs(x) if n in dc.ABSOLUTE else x
    return np.stack([np.nanmean(x[max(0, k - 1):k + 2], axis=0) for k in range(len(x))])


def profile_frames(ctx, which, key, n, chans):
    """Per-frame profile of ``n`` on ``chans`` (physical units, 3-frame smoothing as the
    traces), and its standard deviation over the sampled rollouts (None for measurements
    or without a sidecar). Predictions are the mean of the sampled rollouts."""
    scale = dc.LABELS[n][1]

    def smooth(x):
        return smooth_profile(n, x)

    m = members(ctx) if which == "pred" else None
    k = f"members__{key}__{n}"
    if m is not None and k in m.files:
        each = np.stack([smooth(np.asarray(x, float)[:, chans] * scale) for x in m[k]])
        return np.nanmean(each, 0), np.nanstd(each, 0)
    return smooth(np.asarray(ctx["d"][f"{which}__{key}__{n}"], float)[:, chans] * scale), None


def profile_time_panel(fig, spec, ctx, n, runs, labels, start, step=5, colorbar=True,
                       t_range=(3.5, 6.0), legend=True, cbar_spec=None,
                       free_bottom=0.32):
    """Profile of ``n`` against psi_N (EFIT01, per frame), one line every ``step`` frames from
    frame ``start`` to the plasma end, coloured by time (one time scale per figure, so
    one colour bar serves every panel). ``runs``: ``(which, key)`` pairs drawn solid and
    dashed, named ``labels`` in the panel legend. Predictions carry a band of one standard
    deviation of the sampled rollouts in the line's colour."""
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    t = ctx["t"]
    end = live_end(ctx)
    chans, _ = profile_chords(ctx, n)
    frames = list(range(start, min(end + 1, len(t)), step))
    norm = Normalize(vmin=t_range[0], vmax=t_range[1])
    cmap = matplotlib.colormaps["viridis"]
    # Bottom part of the cell left free for the position axis labels.
    if free_bottom:
        spec = spec.subgridspec(2, 1, height_ratios=[1.0, free_bottom], hspace=0.0)[0]
    if cbar_spec is None:
        sub = spec.subgridspec(1, 3, width_ratios=[1.0, 0.035, 0.12], wspace=0.04)
        ax, cax_spec = fig.add_subplot(sub[0]), sub[1]
    else:
        ax, cax_spec = fig.add_subplot(spec), cbar_spec
    styles = ("-", DASH)
    psin = chord_psin(ctx["shot"], n, chans, t[frames])
    for (which, key), ls in zip(runs, styles):
        mu, sd = profile_frames(ctx, which, key, n, chans)
        for j, k in enumerate(frames):
            col = cmap(norm(t[k]))
            o = np.argsort(psin[j])
            x = psin[j][o]
            if sd is not None:
                ax.fill_between(x, (mu[k] - sd[k])[o], (mu[k] + sd[k])[o], color=col,
                                alpha=0.2, lw=0)
            ax.plot(x, mu[k][o], color=col, ls=ls, lw=1.0 if ls == "-" else 1.2)
    ax.axvline(1.0, color="0.6", lw=0.6, ls=":", zorder=0)
    ax.set_xlabel(PSIN_LABEL)
    ax.set_ylabel(dc.LABELS[n][0])
    headroom(ax, 0.3, zero=True)
    if legend:
        ax.legend([Line2D([], [], color="0.3", lw=1.0, ls=ls) for ls in styles], labels,
                  fontsize=FONT - 2, frameon=False, loc="upper right", handlelength=1.8)
    box(ax, ("TS core" if n.startswith("ts") else "CER") + " profile", size=FONT - 2.5)
    if colorbar:
        cax = fig.add_subplot(cax_spec)
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cb.set_label("time (s)", fontsize=FONT - 1)
        cb.ax.tick_params(labelsize=FONT - 1.5)
    return ax


def waterfall(fig, spec, ctx, n, runs, step=5, elev=24, azim=-58, label=None, start=None,
              thick=(2.5, 3.5, 4.5, 5.5), late_front=False, slices=False):
    """3D waterfall of the profile of ``n`` over psi_N (EFIT01, per frame) and time, on the chords the shot recorded.

    ``runs``: ``(which, key, role)`` triples, e.g. ``("gt", "real", "truth")``. A thin,
    faint profile every ``step`` frames (0.25 s) from ``start`` (default: the measurement
    from the first frame, predictions from the first predicted frame) to the plasma end,
    and thick profiles at the times ``thick`` (s), which carry the comparison.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    d, t, k0 = ctx["d"], ctx["t"], ctx["k0"]
    ax = fig.add_subplot(spec, projection="3d", computed_zorder=False)
    is_ts = n.startswith("ts_core")
    if is_ts:  # real chords only, plotted at their height Z
        chans = np.array(list(WATERFALL[n]))
        chans = chans[chans < len(ts_positions(ctx["shot"]))]
        xpos = ts_z(ctx, chans)
    else:  # the T_i/|v_tor| chord pairs (cer_profile_chords), at their major radius R
        chans = cer_profile_chords(ctx)
        xpos = cer_r(ctx, chans)
    order = np.argsort(xpos)
    chans = chans[order]
    scale = dc.LABELS[n][1]
    end = live_end(ctx)
    psin = chord_psin(ctx["shot"], n, chans, t)
    k_thick = {int(round((tt - T0) / FRAME)) for tt in thick}
    for which, key, role in runs:
        x = np.asarray(d[f"{which}__{key}__{n}"], float)[:, chans] * scale
        if n in dc.ABSOLUTE:
            x = np.abs(x)
        # Light time smoothing (3 frames), as in the traces.
        x = np.stack([np.nanmean(x[max(0, k - 1):k + 2], axis=0) for k in range(len(x))])
        first = start if start is not None else (0 if which == "gt" else k0)
        ls = (0, (3, 1.5)) if role == "scenario" else "-"
        for k in range(first, min(end + 1, len(t))):
            o = np.argsort(psin[k])
            xpos, xk = psin[k][o], x[k][o]
            if k in k_thick:
                if slices:
                    # The time slice as a translucent sheet under the profile.
                    ok = np.isfinite(xk)
                    verts = [list(zip(xpos[ok], np.full(ok.sum(), t[k]), xk[ok]))
                             + [(xpos[ok][-1], t[k], 0.0), (xpos[ok][0], t[k], 0.0)]]
                    sheet = Poly3DCollection(verts, facecolor=PALETTE[role], alpha=0.13,
                                             edgecolor="none")
                    sheet.set_zorder(3.5)
                    ax.add_collection3d(sheet)
                ax.plot(xpos, np.full(len(chans), t[k]), xk, color=PALETTE[role],
                        lw=2.2 if role == "truth" else 1.4, ls=ls,
                        zorder=4 if role == "truth" else 5)
            elif (k - first) % step == 0:
                ax.plot(xpos, np.full(len(chans), t[k]), xk, color=PALETTE[role],
                        lw=0.5, ls=ls, alpha=0.35, zorder=2 if role == "truth" else 3)
    ax.view_init(elev=elev, azim=azim)
    if slices:  # long time axis: the slices spread out in depth
        ax.set_box_aspect((1.0, 2.2, 0.8), zoom=1.0)
        for tt in thick:
            ax.text(np.nanmin(psin), tt, 0.0, f"{tt:g} s", fontsize=FONT - 2.5, color="0.35",
                    ha="right", va="top", zorder=6)
    else:
        ax.set_box_aspect(None, zoom=0.86)  # room for the z tick labels inside the cell
    ax.set_xlabel(PSIN_LABEL, fontsize=FONT - 1, labelpad=-8)
    ax.set_ylabel("time (s)", fontsize=FONT - 1, labelpad=-8)
    ax.set_zlabel(dc.LABELS[n][0], fontsize=FONT - 1, labelpad=-6)
    ax.tick_params(labelsize=FONT - 1.5, pad=-3)
    from matplotlib.ticker import MaxNLocator
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
    ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
    lo_t = 1 if start is None else t[start] - 0.1
    # late_front: later times nearest the viewer, so a profile that falls with time is not
    # hidden behind the earlier, higher ones.
    ax.set_ylim((6, lo_t) if late_front else (lo_t, 6))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0)); axis.pane.set_edgecolor("0.8")
        axis._axinfo["grid"]["linewidth"] = 0.3
    ax.set_title(label or dc.LABELS[n][0], fontsize=FONT - 0.5, pad=-2, y=0.98)
    return ax


def box(ax, text, size=None):
    ax.text(0.015, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=size or FONT - 1.5,
            color="0.3", bbox={"facecolor": "white", "alpha": 0.8, "lw": 0, "pad": 0.8}, zorder=10)


def headroom(ax, frac=0.3, zero=False):
    lo, hi = ax.get_ylim()
    lo = 0 if zero else lo
    ax.set_ylim(lo, hi + frac * (hi - lo))


# Sampled rollouts, one file per figure data file: members__<run>__<signal>, (n, F, C) frame
# values in the data file's units. Written by ``make_figures.py --members`` from the rollout
# archives the data file names (dir__<run>); decode only, no rollout.
MEMBERS = WORK / "members"


def members(ctx):
    path = MEMBERS / f"members_{ctx['src_name']}"
    if "members" not in ctx:
        ctx["members"] = np.load(path) if path.exists() else None
    return ctx["members"]


def member_traces(ctx, a, key, n):
    """Traces of ``n`` of every sampled rollout of run ``key`` (or None), each made as the
    plotted ones (same channels, same smoothing)."""
    m = members(ctx)
    k = f"members__{key}__{n}"
    if m is None or k not in m.files:
        return None
    tr = np.stack([dc._trace(x, ctx["channels"][n], a.smooth) * dc.LABELS[n][1]
                   for x in m[k]])
    return np.abs(tr) if n in dc.ABSOLUTE else tr


def pred_trace(ctx, a, key, n):
    """The prediction shown: the mean of the sampled rollouts, in physical units (the
    data file's stored prediction when there is no sidecar)."""
    tr = member_traces(ctx, a, key, n)
    return dc._tr(ctx, a, "pred", key, n) if tr is None else np.nanmean(tr, 0)


def band(ctx, a, key, n):
    """Mean minus and plus one standard deviation of the sampled rollouts (or None):
    centred on :func:`pred_trace`."""
    tr = member_traces(ctx, a, key, n)
    if tr is None:
        return None
    mu, sd = np.nanmean(tr, 0), np.nanstd(tr, 0)
    return mu - sd, mu + sd


def draw_band(ax, ctx, a, key, n, role, start, color=None):
    b = band(ctx, a, key, n)
    if b is not None:
        t = ctx["t"]
        lo, hi = cut(ctx, b[0]), cut(ctx, b[1])
        ax.fill_between(t[start:], lo[start:], hi[start:], color=color or PALETTE[role],
                        alpha=0.35, lw=0, zorder=2.5)


def extract_members(path):
    """Decode every sampled rollout's profiles of a figure data file into its sidecar."""
    import torch
    from tokamak_foundation_model.ignite import eval_dynamics as ev
    from tokamak_foundation_model.ignite import train_dynamics as td
    d = np.load(path, allow_pickle=True)
    shot = int(d["shot"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmpl = "/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_codecs_v4/{m}/codec_best.pt"
    codecs = td.load_frozen_codecs(list(dc.SLOW), repo=REPO, tmpl=tmpl)
    for codec, _cfg, _fam in codecs.values():
        codec.to(device)
    conv = ev.profile_converters(codecs, list(dc.SLOW))
    out = {}
    for k in d.files:
        if not k.startswith("dir__"):
            continue
        key = k[len("dir__"):]
        arcs = [np.load(next(Path(x).glob(f"codes_{shot}_step*.npz")), allow_pickle=True)
                for x in str(d[k]).split("+")]
        for n in dc.SLOW:
            vals = []
            for z in arcs:
                F = int(z["F"])
                x = ev.decode_flat_chunked(codecs[n][0], ev.archived(z, f"pred__{n}")[:F], device)
                with np.errstate(invalid="ignore"):
                    vals.append(np.nanmean(conv[n](x), axis=-1).astype(np.float32))
            out[f"members__{key}__{n}"] = np.stack(vals)
        print(f"{Path(path).name}: {key}, {len(arcs)} rollouts", flush=True)
    MEMBERS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MEMBERS / f"members_{Path(path).name}", **out)


# Data files whose predicted traces get a band of the sampled rollouts.
# Missing-diagnostic inference (eval_dynamics --infer, run_shot.sh INFER=...): per case the
# shot, a tag, the hidden modalities. Archives in RUN_DIR/_infer_<shot>_<tag>_s30000_g3/
# seed<N>; decoded into WORK/infer/infer_<shot>_<tag>.npz by --infer-data.
# PAPER_FIG_RUN / PAPER_FIG_STEP point at another checkpoint's archives.
RUN_DIR = Path(os.environ.get("PAPER_FIG_RUN", "/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_prod_v4/runs/soup_lu18_h16_30"))
STEP = int(os.environ.get("PAPER_FIG_STEP", 30000))
INFER_CASES = (
    ("204011", "ts", ("ts_core_temp", "ts_core_density", "ts_tangential_temp",
                      "ts_tangential_density")),
    ("204011", "cer", ("cer_ti", "cer_rot")),
    ("204011", "tvlower", ("tangtv_lower",)),
    ("200729", "cer", ("cer_ti", "cer_rot")),
    # Hidden including their own history (--infer_blind --infer_bootstrap; cameras also
    # --forbid_absent, so the "not recorded" frame is never sampled). The upper camera of
    # 204011 did not record: inferred where there is no measurement.
    ("204011", "ts_bs", ("ts_core_temp", "ts_core_density", "ts_tangential_temp",
                         "ts_tangential_density")),
    ("204011", "cer_bs", ("cer_ti", "cer_rot")),
    ("204011", "tvlower_bs", ("tangtv_lower",)),
    ("204011", "tvupper_bs", ("tangtv_upper",)),
    # Removed from the initial state (eval_dynamics --hide): every diagnostic is then
    # predicted autoregressively, nothing held at its measurement. Archives
    # RUN_DIR/_hide_<shot>_<what>_s30000_g3/. These are the ones Fig 5 shows.
    ("204011", "hide_ts", ("ts_core_temp", "ts_core_density", "ts_tangential_temp",
                           "ts_tangential_density")),
    ("204011", "hide_cer", ("cer_ti", "cer_rot")),
    ("204011", "hide_tvlower", ("tangtv_lower",)),
    ("204011", "hide_tvupper", ("tangtv_upper",)),
)
INFER = WORK / "infer"


def extract_infer():
    """Decode the inferred rollouts: per seed, the profiles in the units of the members
    sidecars; the lower camera as the seed mean of the recorded view (as the data
    files). Decode only."""
    import torch
    from tokamak_foundation_model.ignite import eval_dynamics as ev
    from tokamak_foundation_model.ignite import train_dynamics as td
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmpl = "/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_codecs_v4/{m}/codec_best.pt"
    cameras = (CAM, "tangtv_upper")
    codecs = td.load_frozen_codecs(list(dc.SLOW) + list(cameras), repo=REPO, tmpl=tmpl)
    for codec, _cfg, _fam in codecs.values():
        codec.to(device)
    conv = ev.profile_converters(codecs, list(dc.SLOW))
    INFER.mkdir(parents=True, exist_ok=True)
    for shot, tag, mods in INFER_CASES:
        name = (f"_hide_{shot}_{tag[5:]}" if tag.startswith("hide_") else f"_infer_{shot}_{tag}")
        dirs = sorted((RUN_DIR / f"{name}_s{STEP}_g3").glob("seed*"))
        arcs = [np.load(next(x.glob(f"codes_{shot}_step*.npz")), allow_pickle=True)
                for x in dirs]
        if not arcs:
            print(f"infer {shot} {tag}: no archives yet")
            continue
        out = {"seeds": np.array([x.name for x in dirs]), "k0": int(arcs[0]["K0"]),
               "F": int(arcs[0]["F"])}
        F = out["F"]
        for n in dc.SLOW:
            if n not in mods:
                continue
            vals = []
            for z in arcs:
                x = ev.decode_flat_chunked(codecs[n][0], ev.archived(z, f"pred__{n}")[:F], device)
                with np.errstate(invalid="ignore"):
                    vals.append(np.nanmean(conv[n](x), axis=-1).astype(np.float32))
            out[f"members__infer__{n}"] = np.stack(vals)
        for cam in cameras:
            if cam not in mods:
                continue
            codec = codecs[cam][0]
            k0 = out["k0"]
            gt_codes = ev.archived(arcs[0], f"gt__{cam}")[:F]
            full = [np.asarray(ev.decode_flat_chunked(codec, ev.archived(z, f"pred__{cam}")[:F], device))
                    for z in arcs]
            # The view with structure: in the measurement, or, where the camera did not
            # record (constant codes), in the inferred frames.
            ref = (np.mean(full, axis=0) if bool((np.asarray(gt_codes) == np.asarray(gt_codes)[:1]).all()) else
                   np.asarray(ev.decode_flat_chunked(codec, gt_codes, device)))
            view = int(np.argmax([np.nanstd(v[10:-10, 10:-10])
                                  for v in np.nanmean(ref[k0:F], axis=(0, 2))]))
            each = [np.nanmean(x[:, view], axis=1) for x in full]
            out[f"camera_pred__infer__{cam}"] = np.mean(each, axis=0).astype(np.float16)
            out[f"camera_view__{cam}"] = view
        np.savez_compressed(INFER / f"infer_{shot}_{tag}.npz", **out)
        print(f"infer {shot} {tag}: {len(arcs)} seeds", flush=True)


MEMBER_FILES = ("demo_204011_val_data.npz", "ramps/demo_204011_ramps_data.npz",
                "ramps/demo_201240_ramps_data.npz", "demo_201081_cm_data.npz",
                "demo_201085_cm_data.npz", "demo_200729_g1_data.npz")


def trace_panel(ax, ctx, a, n, runs=(("real", "real"),), start=True):
    """Measured (thick, underneath) and predicted traces of signal ``n``, each prediction
    with the band of its sampled rollouts (min to max) when the sidecar exists."""
    t, k0 = ctx["t"], ctx["k0"]
    ax.plot(t, cut(ctx, dc._tr(ctx, a, "gt", "real", n)), color=PALETTE["truth"], lw=LW["truth"], zorder=2,
            solid_capstyle="round")
    for run in runs:
        # (key, role) or (key, role, colour, linestyle)
        key, role = run[:2]
        col = run[2] if len(run) > 2 else PALETTE[role]
        ls = run[3] if len(run) > 3 else ("--" if role == "scenario" else "-")
        draw_band(ax, ctx, a, key, n, role, k0 - 1, color=col)
        ax.plot(t[k0 - 1:], cut(ctx, pred_trace(ctx, a, key, n))[k0 - 1:], color=col, lw=LW[role],
                ls=ls, zorder=3)
    if start:
        ax.axvline(T0 + FRAME * k0, color="0.55", lw=0.7, ls=":", zorder=1)
    ax.set_ylabel(dc.LABELS[n][0])
    headroom(ax, zero=n in dc.ZERO_BASED)
    box(ax, chord_label(ctx, n), size=FONT - 2.5)


def spectro_panel(ax, ctx, img, name, role, ylim):
    d, t = ctx["d"], ctx["t"]
    gt = d[f"spectro_gt__{name}"]
    finite = gt[np.isfinite(gt)]
    vmin, vmax = np.percentile(finite, [1.0, 99.5])
    fmax = float(d[f"spectro_fmax_khz__{name}"])
    ax.imshow(img, aspect="auto", origin="lower", cmap=dc.CMAPS[role], vmin=vmin, vmax=vmax,
              extent=(t[0], t[-1] + FRAME, 0, fmax), interpolation="none", rasterized=True)
    ax.set_ylim(0, ylim)
    ax.grid(False)


def camera_strip(fig, spec, frames_by_role, times, crop):
    """Rows of camera frames (one per role) at ``times``; common colour limits."""
    imgs = [[dc._camera_image(x[int(round((tt - T0) / FRAME))], crop, 1.5) for tt in times]
            for _role, x in frames_by_role]
    flat = np.concatenate([im.ravel() for row in imgs for im in row])
    vmin, vmax = np.percentile(flat, [1.0, 99.7])
    grid = spec.subgridspec(len(frames_by_role), len(times), wspace=0.03, hspace=0.06)
    axes = []
    for r, (role, _x) in enumerate(frames_by_role):
        for c, tt in enumerate(times):
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(imgs[r][c], cmap=dc.CMAPS[role], vmin=vmin, vmax=vmax, interpolation="bilinear")
            ax.set_anchor("S" if r == 0 else "N")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.text(0.98, 0.95, f"{tt:.1f} s", transform=ax.transAxes, ha="right", va="top",
                        fontsize=FONT - 1, color="0.3")
            axes.append(ax)
    return axes


def camera_grid(fig, spec, cells, row_labels=(), col_labels=(), crop=None, wspace=0.03, hspace=0.05):
    """Grid of camera frames. ``cells[r][c]`` is ``(role, frames, time)`` or ``(role, None, text)``
    for a placeholder of the same size. Common colour limits over all frames."""
    imgs = [[None if x is None else dc._camera_image(x[int(round((tt - T0) / FRAME))], crop, 1.5)
             for role, x, tt in row] for row in cells]
    real = [im for row in imgs for im in row if im is not None]
    vmin, vmax = np.percentile(np.concatenate([im.ravel() for im in real]), [1.0, 99.7])
    shape = real[0].shape
    grid = spec.subgridspec(len(cells), len(cells[0]), wspace=wspace, hspace=hspace)
    axes = []
    for r, row in enumerate(cells):
        out = []
        for c, (role, x, tt) in enumerate(row):
            ax = fig.add_subplot(grid[r, c])
            if x is None:
                ax.imshow(np.zeros(shape), cmap=dc.CMAPS[role], vmin=0, vmax=12)
                ax.text(0.5, 0.5, tt, transform=ax.transAxes, ha="center", va="center",
                        fontsize=FONT - 1.5, color=PALETTE[role])
            else:
                ax.imshow(imgs[r][c], cmap=dc.CMAPS[role], vmin=vmin, vmax=vmax, interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0 and c < len(col_labels):
                ax.set_title(col_labels[c], fontsize=FONT - 0.5, pad=2)
            if c == 0 and r < len(row_labels):
                ax.set_ylabel(row_labels[r], fontsize=FONT - 0.5, rotation=0, ha="right", va="center",
                              labelpad=3)
            out.append(ax)
        axes.append(out)
    return axes


def cams(ctx, key="real"):
    """Measured and predicted lower-camera frames (F, H, W) of run ``key``."""
    d = ctx["d"]
    return (np.asarray(d[f"camera_gt__{CAM}"], np.float32),
            np.asarray(d[f"camera_pred__{key}__{CAM}"], np.float32))


def brightness(frames, crop, q=99.5, smooth=5):
    """Emission peaking of each camera frame: the ``q``-th percentile inside ``crop``.

    The camera codec standardises every frame (mean 0, std ~1), so the mean
    brightness carries no information. The percentile tracks how concentrated the
    strike-leg emission is: it drops when the emission spreads out on detachment.
    A ``smooth``-frame running mean (edge-padded) takes out frame-to-frame noise.
    """
    x = np.asarray(frames, np.float32)[(slice(None), *crop)]
    y = np.percentile(x.reshape(len(x), -1), q, axis=1)
    pad = np.pad(y, smooth // 2, mode="edge")
    return np.convolve(pad, np.ones(smooth) / smooth, mode="valid")


def crop_of(frames, k0):
    mean = np.asarray(frames[k0:], dtype=np.float32).mean(0)
    floor = np.median(mean) - 1.0
    rows, cols = np.where(mean.mean(1) > floor)[0], np.where(mean.mean(0) > floor)[0]
    return (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))


def legend(fig, entries, y=1.0, ncol=None, extra=()):
    handles = [plt.Line2D([], [], color=PALETTE[r], lw=LW[r] if r != "truth" else 2.2,
                          ls="--" if r == "scenario" else "-") for r, _l in entries]
    labels = [l for _r, l in entries]
    for h, l in extra:
        handles.append(h); labels.append(l)
    # Hung from the top edge, so a larger font cannot push it off the page; ``y`` is unused.
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=ncol or len(labels), frameon=False, handlelength=2.2, columnspacing=1.4)


# ---------------------------------------------------------------------------- spectrogram channels
RANK = WORK
SPEC_ORDER = ("ece", "co2", "mhr", "mirnov", "bes")


def rank(shot):
    """Scores and images of ``rank_spectro.py`` for ``shot``, or ``None`` before it has run."""
    import json
    js, nz = RANK / f"rank_{shot}.json", RANK / f"rank_{shot}.npz"
    if not (js.exists() and nz.exists()):
        return None
    return json.loads(js.read_text()), np.load(nz)


def best_spectro(shot, key="replay", names=SPEC_ORDER, n=4):
    """``[(modality, channel)]``: the best channel of each modality in ``names``.

    Channels are ranked within a modality by ``key``. The mode strength has a different
    scale per diagnostic, so modalities are ordered as given for ``replay`` and, for a
    response key, by the relative change ``resp / replay`` of their best channel. A
    modality whose prediction matches exactly (match 1) was not recorded and is skipped.
    """
    r = rank(shot)
    if r is None:
        return []
    scores, imgs = r
    best = []
    for i, m in enumerate(names):
        rows = [x for x in scores["scores"].get(m, []) if f"gt__{m}__{x['ch']}" in imgs.files and key in x
                and x["match"] < 0.999]
        if rows:
            top = max(rows, key=lambda x: x[key])
            order = -i if key == "replay" else top[key] / max(top["replay"], 1e-9)
            best.append((order, m, top["ch"]))
    best.sort(reverse=True)
    return [(m, ch) for _o, m, ch in best[:n]]


def spectro_images(shot, m, ch, runs):
    """``(fmax_khz, measured, {run: medoid-seed prediction})`` from the rank file."""
    _scores, imgs = rank(shot)
    return (float(imgs[f"fmax__{m}"]), np.asarray(imgs[f"gt__{m}__{ch}"], np.float32),
            {r: np.asarray(imgs[f"pred__{r}__{m}__{ch}"], np.float32) for r in runs})


# Upper frequency limits per diagnostic (kHz): MHR's coherent modes sit below 100 kHz,
# the broadband bursts above it would otherwise set the range.
SPEC_CAP = {"mhr": 100.0, "mirnov": 100.0, "co2": 100.0}


def spectro_ylim(gt, fmax, k0_frac=0.2, keep=0.97, cap=np.inf):
    """Frequency (kHz) below which ``keep`` of the measured mode excess lies, rounded up."""
    sys.path.insert(0, str(HERE))
    from rank_spectro import excess
    # Per frequency, the 75th percentile over time: broadband bursts (ELMs, vertical
    # stripes) light one column at a time and drop out; mode lines persist.
    e = np.percentile(excess(gt)[:, int(gt.shape[1] * k0_frac):], 75, axis=1)
    c = np.cumsum(e) / max(e.sum(), 1e-9)
    f = fmax * (np.searchsorted(c, keep) + 1) / len(c)
    step = 10 if f < 60 else 25 if f < 150 else 50
    return min(fmax, cap, step * np.ceil(f / step))


def spectro_name(m, ch):
    names = dc.channel_names(m)
    shown = m.upper() if m != "mirnov" else "Mirnov"
    return f"{shown} {names[ch] if ch < len(names) else ch}"


_SPEC_CFG = {}


def spectro_cfg(m):
    """Codec config of spectrogram modality ``m`` (its per-frequency log-power stats)."""
    if m not in _SPEC_CFG:
        from tokamak_foundation_model.ignite import train_dynamics as td
        tmpl = "/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_codecs_v4/{m}/codec_best.pt"
        _SPEC_CFG[m] = td.load_frozen_codecs([m], repo=REPO, tmpl=tmpl)[m][1]
    return _SPEC_CFG[m]


def mode_amplitude_db(img, m, ch, fmax, ylim, n_frames, fmin=2.0):
    """Power of the strongest spectral line per 50 ms frame, in dB (relative).

    ``img`` is a stitched decoded spectrogram (n_f, columns): log10 power, z-scored per
    frequency by the codec, which is undone here. Within each frame the power is
    averaged over time, then the maximum over ``fmin``-``ylim`` kHz is taken, so the
    trace follows the mode as its frequency changes.
    """
    cfg = spectro_cfg(m)
    x = np.asarray(img, np.float64)
    if getattr(cfg, "logpow_standardize", False) and cfg.logpow_freq_mean is not None:
        fm = np.asarray(cfg.logpow_freq_mean, np.float64)[ch][:, None]
        fs = np.clip(np.asarray(cfg.logpow_freq_std, np.float64)[ch],
                     float(getattr(cfg, "logpow_std_floor", 0.25)), None)[:, None]
        x = x * fs + fm
    f = (np.arange(x.shape[0]) + 0.5) * fmax / x.shape[0]
    band = (f >= fmin) & (f <= ylim)
    cols = np.array_split(np.arange(x.shape[1]), n_frames)
    p = np.stack([np.mean(10.0 ** x[:, c], axis=1) for c in cols], axis=1)  # (n_f, frames)
    return 10.0 * np.log10(p[band].max(axis=0))


def row_tag(ax, text, role):
    """Name of a spectrogram row, top right inside the panel."""
    ax.text(0.99, 0.95, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT - 2, color="0.15" if role == "truth" else PALETTE[role],
            bbox={"facecolor": "white", "alpha": 0.8, "lw": 0, "pad": 0.8}, zorder=6)


def spectro_img(ax, img, gt, fmax, role, ylim, t):
    finite = gt[np.isfinite(gt)]
    vmin, vmax = np.percentile(finite, [1.0, 99.5])
    ax.imshow(img, aspect="auto", origin="lower", cmap=dc.CMAPS[role], vmin=vmin, vmax=vmax,
              extent=(t[0], t[-1] + FRAME, 0, fmax), interpolation="none", rasterized=True)
    ax.set_ylim(0, ylim)
    ax.grid(False)


def actuator_panel(ax, ctx, act, label_above=False):
    """Measured actuator ``act`` (one of the data file's ``act__real__*`` keys) to the plasma
    end. ``label_above``: its name above the panel instead of in a box inside it (short
    panels, where the box covers the trace)."""
    ax.plot(ctx["t"], cut(ctx, ctx["d"][f"act__real__{act}"]), color="0.25", lw=1.1, zorder=2)
    ax.axvline(T0 + FRAME * ctx["k0"], color="0.55", lw=0.7, ls=":", zorder=1)
    if label_above:
        headroom(ax, 0.1, zero=True)
        ax.text(0.0, 1.06, act, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=FONT - 1.5, color="0.3")
    else:
        headroom(ax, 0.35, zero=True)
        box(ax, act)


# ---------------------------------------------------------------------------- figure 1
def frame_at(ctx, time_s):
    return int(np.argmin(np.abs(ctx["t"] - time_s)))


def profile_snapshot(ax, ctx, n, time_s, runs=(("real", "real"),), ref=None,
                     measured=True, label_x=True, show_time=True):
    """Profile of ``n`` against psi_N at one time (EFIT01 slice nearest it): measured
    chords as dots,
    each prediction (``runs``: ``(key, role)``) as a line with a band of one standard
    deviation of the sampled rollouts, and ``ref`` = ``(which, key, time_s)`` as a thin
    grey dashed reference profile (e.g. the initial state)."""
    chans, _ = profile_chords(ctx, n)

    def placed(k):
        """psi_N of the chords at frame ``k`` and the order that sorts them."""
        x = chord_psin(ctx["shot"], n, chans, [ctx["t"][k]])[0]
        o = np.argsort(x)
        return x[o], o

    if ref is not None:
        which, key, t_ref = ref
        mu, _sd = profile_frames(ctx, which, key, n, chans)
        kr = frame_at(ctx, t_ref)
        xr, o = placed(kr)
        ax.plot(xr, mu[kr][o], color="0.45", lw=0.8, ls=(0, (3, 2)), zorder=2)
    k = frame_at(ctx, time_s)
    xpos, o = placed(k)
    for run in runs:
        # (key, role) or (key, role, colour, linestyle)
        key, role = run[:2]
        col = run[2] if len(run) > 2 else PALETTE[role]
        ls = run[3] if len(run) > 3 else ("--" if role == "scenario" else "-")
        mu, sd = profile_frames(ctx, "pred", key, n, chans)
        if sd is not None:
            ax.fill_between(xpos, (mu[k] - sd[k])[o], (mu[k] + sd[k])[o], color=col,
                            alpha=0.3, lw=0, zorder=3)
        ax.plot(xpos, mu[k][o], color=col, lw=LW[role], ls=ls, zorder=4)
    if measured:
        mu, _sd = profile_frames(ctx, "gt", "real", n, chans)
        ax.plot(xpos, mu[k][o], ls="none", marker="o", ms=2.6, mew=0,
                color=PALETTE["truth"], zorder=5)
    if show_time:
        ax.text(0.97, 0.97, f"{ctx['t'][k]:.2f}".rstrip("0").rstrip(".") + " s",
                transform=ax.transAxes, ha="right", va="top", fontsize=FONT - 1.5,
                color="0.3", zorder=10)
    if label_x:
        ax.set_xlabel(PSIN_LABEL)
    else:
        plt.setp(ax.get_xticklabels(), visible=False)  # shared x: keep the ticker
    return ax


def psin_axes(axes):
    """One psi_N range for a row of profile panels, from 0 to just past the outermost
    chord, with two ticks (0.5, 1 when it reaches the separatrix, else 0.3, 0.6)."""
    hi = max(np.nanmax(ln.get_xdata()) for ax in axes for ln in ax.get_lines()
             if len(ln.get_xdata()))
    hi *= 1.04
    for ax in axes:
        ax.set_xlim(0, hi)
        ax.set_xticks([0.5, 1.0] if hi >= 1.0 else [0.3, 0.6])
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%g"))


def mark_times(ax, times):
    """Thin grey lines at the times whose profiles are shown."""
    for tt in times:
        ax.axvline(tt, color="0.7", lw=0.6, zorder=0.5)


PROFILE_TIMES_FIG1 = (2.5, 3.5, 4.5, 5.5)   # s, as the camera frames; initial state to 2 s
PROFILE_TIMES_FIG2 = (4.0, 4.5, 5.0, 5.5)   # s; the ramps run from 3.5 s to 4.5 s
HALF = "#b48ccb"                            # the halved change: a lighter purple


def profile_row(fig, spec, ctx, a, n, times, runs, trace_runs, ref=None, label_x=True,
                label_t=True, head=False, sharex=None, trace_width=1.7):
    """One profile signal over time: its profile at each of ``times`` (shared y), then
    the time trace of its chord on the right with grey lines at those times."""
    cols = spec.subgridspec(1, len(times) + 2, wspace=0.08,
                            width_ratios=[1] * len(times) + [0.12, trace_width])
    axes = []
    for j, ts in enumerate(times):
        ax = fig.add_subplot(cols[j], sharey=axes[0] if axes else None,
                             sharex=sharex[j] if sharex else None)
        profile_snapshot(ax, ctx, n, ts, runs=runs, ref=ref, label_x=label_x,
                         show_time=head)
        if j:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax.set_ylabel(dc.LABELS[n][0])
        if head:
            ax.set_title(f"{ts:g} s", fontsize=FONT - 0.5, pad=2)
            ax.texts[-1].remove()
        axes.append(ax)
    psin_axes(axes)
    lo = min(x.get_ylim()[0] for x in axes)
    hi = max(x.get_ylim()[1] for x in axes)
    axes[0].set_ylim(0 if n in dc.ZERO_BASED else lo, hi + 0.25 * (hi - lo))
    box(axes[0], "TS" if n.startswith("ts") else "CER", size=FONT - 2.5)
    # The chord's time trace, y axis on the right.
    tax = fig.add_subplot(cols[-1])
    trace_panel(tax, ctx, a, n, runs=trace_runs)
    mark_times(tax, times)
    tax.set_xlim(1, 6)
    tax.yaxis.tick_right(); tax.yaxis.set_label_position("right")
    tax.set_ylabel("")
    tax.spines["left"].set_visible(False); tax.spines["right"].set_visible(True)
    if label_t:
        tax.set_xlabel("time (s)")
    else:
        plt.setp(tax.get_xticklabels(), visible=False)
    return axes, tax


def figure1():
    ctx, a = context(DATA / "demo_204011_val_data.npz", channels=CHORDS["204011"])
    d, t, k0 = ctx["d"], ctx["t"], ctx["k0"]
    cam_gt, cam_pred = cams(ctx)
    fig = plt.figure(figsize=(WIDTH, HEIGHT_FIG1))
    # Margins in inches (legend 0.32 in on top), so they survive a change of height.
    outer = fig.add_gridspec(4, 1, height_ratios=[0.36, 3.9, 1.7, 0.95], hspace=0.36,
                             left=0.085, right=0.94, top=1 - 0.47 / HEIGHT_FIG1,
                             bottom=0.02 / HEIGHT_FIG1)
    # a: the measured actuators that drive the rollout.
    acts = outer[0].subgridspec(1, 4, wspace=0.42)
    for i, act in enumerate(("NBI (MW)", "torque (N m)", "ECH (MW)", "gas (Torr L/s)")):
        ax = fig.add_subplot(acts[0, i])
        actuator_panel(ax, ctx, act, label_above=True)
        ax.set_xlim(1, 6)  # time (s), as every trace below; no label, as in Fig. 2
    # b: per profile signal, its profile at four times (initial state as a grey dashed
    # reference) and the time trace of one chord with those times marked.
    rows = outer[1].subgridspec(5, 1, hspace=0.2, height_ratios=[1, 1, 0.55, 1, 1])
    rows = [rows[0], rows[1], rows[3], rows[4]]
    t_init = float(t[k0 - 1])
    shared = {}
    for i, n in enumerate(dc.SLOW):
        # T_e and n_e share Z, T_i and |v_tor| share R: label x once per pair.
        first_of_pair = i in (0, 2)
        axes, tax = profile_row(fig, rows[i], ctx, a, n, PROFILE_TIMES_FIG1,
                                runs=(("real", "real"),), trace_runs=(("real", "real"),),
                                ref=("gt", "real", t_init), label_x=not first_of_pair,
                                label_t=i == 3, head=i == 0,
                                sharex=None if first_of_pair else shared[i - 1])
        shared[i] = axes
        if n == "ts_core_density":  # two lines, clear of the T_e label above
            axes[0].set_ylabel(r"$n_e$" + "\n" + r"($10^{19}$ m$^{-3}$)", linespacing=1.0)
            axes[0].yaxis.set_label_coords(-0.29, 0.4)  # below the T_e label
        if n == "ts_core_temp":
            axes[0].yaxis.set_label_coords(-0.25, 0.62)
        if first_of_pair:
            for x in axes:
                x.set_xlabel("")
    # d: the best channel of each spectrogram diagnostic, measured over predicted.
    shown = best_spectro("204011", names=("ece", "co2", "mhr", "mirnov")) or [("ece", None), ("mhr", None)]
    mid = outer[2].subgridspec(2, len(shown), wspace=0.28, hspace=0.08)
    for j, (m, ch) in enumerate(shown):
        if ch is None:  # before rank_spectro.py has run: the data file's own channel
            fmax, gt, pr = float(d[f"spectro_fmax_khz__{m}"]), d[f"spectro_gt__{m}"], d[f"spectro_pred__real__{m}"]
            name = dc.spectro_label(d, m)
        else:
            fmax, gt, preds = spectro_images("204011", m, ch, ["real"])
            pr, name = preds["real"], spectro_name(m, ch)
        ylim = spectro_ylim(gt, fmax, cap=SPEC_CAP.get(m, np.inf))
        for r, (img, role) in enumerate(((gt, "truth"), (pr, "real"))):
            ax = fig.add_subplot(mid[r, j])
            spectro_img(ax, img, gt, fmax, role, ylim, t)
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.axvline(T0 + FRAME * k0, color="w", lw=0.7, ls=":")
            ax.set_xlim(1, 6)
            if r == 0:
                ax.set_xticklabels([]); box(ax, name)
            else:
                ax.set_xlabel("time (s)")
            ax.set_ylabel("f (kHz)" if j == 0 else "")  # short: the two rows' labels fit
    crop = crop_of(cam_gt, k0)
    axes = camera_strip(fig, outer[3], [("truth", cam_gt), ("real", cam_pred)], (2.5, 3.5, 4.5, 5.5), crop)
    # rotated beside the frames: inside them it would cover the time label
    axes[0].text(-0.03, 0.0, "lower\ndivertor", transform=axes[0].transAxes, rotation=90,
                 ha="right", va="center", fontsize=FONT - 2.5, color="0.3", linespacing=1.0)
    legend(fig, [("truth", "measured"), ("real", "prediction")], y=0.985, ncol=4,
           extra=[(plt.Line2D([], [], color="0.45", lw=0.8, ls=(0, (3, 2))),
                   f"initial state ({t_init:.1f} s)"),
                  (plt.Line2D([], [], color="0.7", lw=0.6), "profile times")])
    fig.savefig(OUT / "fig1_replay_204011.png"); fig.savefig(OUT / "fig1_replay_204011.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------- figure 2
# Torque on 204011 (beam mix, so P_inj falls too), ECH on 201240.
SIDES_ALL = [
    {"shot": "204011", "data": "ramps/demo_204011_ramps_data.npz", "chords": "204011_torque",
     "act": "torque (N m)", "resp": "cer_rot", "second": "cer_ti", "spec_default": "mhr",
     "spec_names": ("mhr", "mirnov"),
     "cols": [("tq_lin_zero", r"torque $\to$ 0"), ("tq_lin_half", r"torque $\to$ ½")]},
    {"shot": "204011", "data": "ramps/demo_204011_ramps_data.npz", "chords": "204011_ech",
     "act": "ECH (MW)", "resp": "ts_core_temp", "second": "ts_core_density", "spec_default": "ece",
     "spec_names": ("ece", "co2"),
     "cols": [("ech_lin_off", r"ECH $\to$ 0"), ("ech_lin_half", r"ECH $\to$ ½")]},
    {"shot": "201240", "data": "ramps/demo_201240_ramps_data.npz", "chords": "201240_torque",
     "act": "torque (N m)", "resp": "cer_rot", "second": "cer_ti", "spec_default": "mhr",
     "spec_names": ("mhr", "mirnov"),
     "cols": [("tq_lin_zero", r"torque $\to$ 0"), ("tq_lin_half", r"torque $\to$ ½")]},
    {"shot": "201240", "data": "ramps/demo_201240_ramps_data.npz", "chords": "201240_ech",
     "act": "ECH (MW)", "resp": "ts_core_temp", "second": "ts_core_density", "spec_default": "ece",
     "spec_names": ("ece", "co2"),
     "cols": [("ech_lin_off", r"ECH $\to$ 0"), ("ech_lin_half", r"ECH $\to$ ½")]},
]


def figure2():
    """One figure per discharge: torque and ECH of 204011, ECH of 201240."""
    figure2_shot("204011")
    figure2_shot("201240")


def figure2_shot(shot):
    SIDES = [x for x in SIDES_ALL if x["shot"] == shot]
    two = len(SIDES) == 2
    # A discharge with one change (201240: ECH only) is drawn as one half
    # of the two-change figure: same panel sizes and margins, half the width; its legend
    # takes two columns, so the figure is taller by one legend row and the panels keep
    # their height.
    left_in, right_in, top_in, bottom_in = 0.075 * WIDTH, 0.01 * WIDTH, 0.72, 0.36
    unit = (WIDTH - left_in - right_in) / (4.2 + 4 * 0.1 * 0.84)  # one column, in inches
    width = WIDTH if two else left_in + right_in + unit * (2 + 0.1)
    if not two:
        top_in += 0.42
    height = HEIGHT_FIG2 + (top_in - 0.72)
    fig = plt.figure(figsize=(width, height))
    g = fig.add_gridspec(7, 5 if two else 2, height_ratios=[0.6, 0.95, 0.85, 1.6, 0.6, 0.6, 0.6],
                         width_ratios=[1, 1, 0.2, 1, 1] if two else [1, 1],
                         hspace=0.2, wspace=0.1, left=left_in / width, right=1 - right_in / width,
                         top=1 - top_in / height, bottom=bottom_in / height)  # legend + titles; time label
    for side, gcols in zip(SIDES, ((0, 1), (3, 4))):  # one side: columns 0, 1
        spec_ch = None
        ctx, a = context(DATA / side["data"], channels=CHORDS[side["chords"]])
        d, t = ctx["d"], ctx["t"]
        keys = [k for k, _h in side["cols"]]
        # Best response channel of the first scenario, if ranked; else the data file's.
        r = rank(side["shot"])
        spec = None
        if r is not None:
            cand = best_spectro(side["shot"], key=f"resp:{keys[0]}", names=side["spec_names"], n=1)
            if cand:
                m, ch = cand[0]
                spec_ch = ch
                fmax, gt, preds = spectro_images(side["shot"], m, ch, ["real"] + keys)
                spec = (m, fmax, gt, preds, spectro_name(m, ch))
        if spec is None:
            m = side["spec_default"]
            if f"spectro_gt__{m}" in d.files:
                preds = {k: d[f"spectro_pred__{k}__{m}"] for k in ["real"] + keys}
                spec = (m, float(d[f"spectro_fmax_khz__{m}"]), d[f"spectro_gt__{m}"], preds, dc.spectro_label(d, m))
                if f"spectro_channel__{m}" in d.files:
                    spec_ch = int(d[f"spectro_channel__{m}"])
        for c, ((key, head), gc) in enumerate(zip(side["cols"], gcols)):
            first = c == 0
            axs = [fig.add_subplot(g[r_, gc]) for r_ in (0, 1, 2, 4, 5, 6)]
            ax = axs[0]
            act = side["act"]
            ax.plot(t, cut(ctx, d[f"act__real__{act}"]), color=PALETTE["truth"], lw=LW["truth"], zorder=2)
            ax.plot(t, cut(ctx, d[f"act__{key}__{act}"]), color=PALETTE["scenario"], lw=LW["scenario"], ls="--", zorder=3)
            ax.set_title(f"#{side['shot']}: {head}", fontsize=FONT + 0.5, pad=3)
            ax.set_ylabel(dc.ACT_SHOWN[act] if first else "")
            headroom(ax, 0.15, zero=True)
            for ax, n in ((axs[1], side["resp"]), (axs[2], side["second"])):
                trace_panel(ax, ctx, a, n, runs=(("real", "real"), (key, "scenario")))
                mark_times(ax, PROFILE_TIMES_FIG2)
                if not first:
                    ax.set_ylabel("")
            dc._tau_box(axs[1], ctx, key)
            if spec is not None:
                m, fmax, gt, preds, name = spec
                ylim = spectro_ylim(gt, fmax, cap=SPEC_CAP.get(m, np.inf))
                # Measured, then the prediction under the measured actuators (replay),
                # then under the changed actuator; each row named in the panel.
                rows_sp = ((gt, "truth", "measured"),
                           (preds["real"], "real", "replay (measured actuators)"),
                           (preds[key], "scenario", "changed actuator"))
                for ax, (img, role, tag) in zip(axs[3:6], rows_sp):
                    spectro_img(ax, img, gt, fmax, role, ylim, t)
                    ax.set_ylabel("f (kHz)" if first else "")  # short: three rows of labels fit
                    row_tag(ax, f"{tag}, {name}" if role == "truth" else tag, role)
            for r_, ax in enumerate(axs):
                ax.set_xlim(1, 6)
                ax.axvline(ctx["t_switch"], color=PALETTE["scenario"], lw=0.8, ls=":", zorder=4)
                if r_ < 5:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("time (s)")
                if not first:
                    ax.set_yticklabels([])
        # Shared y ranges within a side.
        rows = [[ax for ax in fig.axes if ax.get_subplotspec() is not None and
                 ax.get_subplotspec().rowspan.start == r_ and ax.get_subplotspec().colspan.start in gcols]
                for r_ in (0, 1, 2)]
        for pair in rows:
            if len(pair) == 2:
                lo = min(x.get_ylim()[0] for x in pair); hi = max(x.get_ylim()[1] for x in pair)
                for x in pair:
                    x.set_ylim(lo, hi)
        # d: the whole profile of the main response at four times after the change:
        # measured, replay, full change and halved change.
        n = side["resp"]
        span = g[3, gcols[0]:gcols[1] + 1].subgridspec(2, 1, height_ratios=[1.0, 0.42],
                                                         hspace=0.0)[0]
        sl = span.subgridspec(1, len(PROFILE_TIMES_FIG2), wspace=0.1)
        prof = []
        for j, ts in enumerate(PROFILE_TIMES_FIG2):
            ax = fig.add_subplot(sl[j], sharey=prof[0] if prof else None)
            profile_snapshot(ax, ctx, n, ts, runs=(("real", "real"),
                                                    (keys[0], "scenario"),
                                                    (keys[1], "scenario", HALF, ":")))
            if j == 0:
                ax.set_ylabel(dc.LABELS[n][0])
                box(ax, "TS" if n.startswith("ts") else "CER", size=FONT - 2.5)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            prof.append(ax)
        psin_axes(prof)
        headroom(prof[0], 0.3, zero=True)
    legend(fig, [("truth", "measured"), ("real", "replay (measured actuators)"),
                 ("scenario", "changed actuator and its prediction")], y=1 - 0.36 / HEIGHT_FIG2,
           ncol=3 if two else 2,
           extra=[(plt.Line2D([], [], color="0.25", lw=5, solid_capstyle="butt"),
                   r"$\tau \approx$ 100 ms after the change"),
                  (plt.Line2D([], [], color=HALF, lw=1.3, ls=":"), "halved change (profiles)"),
                  (plt.Line2D([], [], color="0.7", lw=0.6), "profile times")])
    fig.savefig(OUT / f"fig2_actuators_{shot}.png"); fig.savefig(OUT / f"fig2_actuators_{shot}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------- figure 3
FS_SHOWN = "FS03"  # recorded in every shot of the detachment session
FS_UNIT = 1e16


def detach(shot):
    """Measured gas (GASA, Torr L/s) and filterscope (FS03) per frame, from ``detach_gt.py``."""
    d = np.load(WORK / f"detach_{shot}.npz")
    j = list(d["fs_names"]).index(FS_SHOWN)
    return {"t": T0 + FRAME * np.arange(len(d["gas"])), "gas": d["gas"].sum(1),
            "fs": d["fs"][:, j] / FS_UNIT, "pinj": d["pinj"] / 1e6}


def simple_trace(ax, t, y, role, k0=None, label=None, ls=None, color=None):
    lw = LW[role]
    ax.plot(t if k0 is None else t[k0 - 1:], y if k0 is None else y[k0 - 1:], color=color or PALETTE[role], lw=lw,
            ls=ls or ("--" if role == "scenario" else "-"), zorder=2 if role == "truth" else 3,
            solid_capstyle="round", label=label)


def short_label(ax, n):
    """y label without units; the units go in the label box."""
    full = dc.LABELS[n][0]
    ax.set_ylabel(full.split(" (")[0])
    return full[len(full.split(" (")[0]) + 1:]


GAS = "gas (Torr L/s)"


# Emission-front height Z_E (m) in the lower divertor camera, from the detachment
# analysis of the exported camera files (export_camera_sav.py, EFIT01 geometry). Those files
# come from the maskfree soup, so other models' figures keep the peaking measure.
ZE_FILE = HERE / "ze_dz_paper.pk"
ZE_ENS = "ens8_soup_maskfree"


@functools.lru_cache(maxsize=None)
def front_height(fig, shot):
    """{run key or 'truth': (t, Z_E)} of one figure's shot, or None without Z_E data."""
    if DATA.name != ZE_ENS or not ZE_FILE.exists():
        return None
    import pickle
    with open(ZE_FILE, "rb") as f:
        rows = pickle.load(f)["data"][fig]
    out = {}
    for r in rows:
        if str(int(r["shot"])) != shot:
            continue
        series = r["series"]
        key = "truth" if series.startswith("measured") else series.split(", ")[1]
        key = key.replace("donor", "donor_") if key.startswith("donor") else key
        out[key] = (np.asarray(r["time"], float), np.asarray(r["ze"], float))
    return out


def camera_series(ctx, crop, fig, key):
    """(t, y, k0) of the camera trace of run ``key`` ('truth' = measured): Z_E where the
    detachment analysis has it, else the emission peaking."""
    ze = front_height(fig, ctx["shot"])
    if ze:
        t, y = ze[key]
        return t, y, None
    frames = cams(ctx)[0] if key == "truth" else cams(ctx, key)[1]
    return ctx["t"], brightness(frames, crop), None if key == "truth" else ctx["k0"]


def camera_label(ax, ctx, fig):
    if front_height(fig, ctx["shot"]):
        ax.set_ylabel(r"$Z_E$"); box(ax, "lower divertor emission front height (m)")
    else:
        ax.set_ylabel("peaking"); box(ax, "lower divertor camera emission peaking (σ)")


def brightness_panel(ax, ctx, crop, fig, runs=(("real", "real", None),)):
    simple_trace(ax, *camera_series(ctx, crop, fig, "truth")[:2], "truth")
    for key, role, ls in runs:
        t, y, k0 = camera_series(ctx, crop, fig, key)
        simple_trace(ax, t, y, role, k0=k0, ls=ls)
    camera_label(ax, ctx, fig)


def figure3():
    shots = ("201081", "201085")
    times = (2.5, 4.0, 5.5)
    fig = plt.figure(figsize=(WIDTH, 5.0))
    outer = fig.add_gridspec(2, 1, hspace=0.3, left=0.085, right=0.99, top=0.885, bottom=0.07)
    for i, shot in enumerate(shots):
        ctx, a = context(DATA / f"demo_{shot}_cm_data.npz", channels=CHORDS[shot])
        d, t = ctx["d"], ctx["t"]
        cam_gt, cam_pred = cams(ctx)
        crop = crop_of(cam_gt, ctx["k0"])
        row = outer[i].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.17)
        left = row[0].subgridspec(4, 1, hspace=0.22)
        axes = [fig.add_subplot(left[r]) for r in range(4)]
        simple_trace(axes[0], t, d[f"act__real__{GAS}"], "truth")
        axes[0].set_ylabel("gas"); headroom(axes[0], zero=True); box(axes[0], "GASA (Torr L/s)")
        for ax, n in zip(axes[1:3], ("ts_core_density", "ts_core_temp")):
            trace_panel(ax, ctx, a, n)
            unit = short_label(ax, n)
            ax.texts[-1].set_text(ax.texts[-1].get_text() + f" {unit}")
        brightness_panel(axes[3], ctx, crop, "fig3_detachment"); headroom(axes[3], 0.45)
        for r, ax in enumerate(axes):
            ax.set_xlim(1, 6)
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
            ax.axvline(T0 + FRAME * ctx["k0"], color="0.55", lw=0.7, ls=":", zorder=1)
            if r < 3:
                ax.set_xticklabels([])
        axes[3].set_xlabel("time (s)")
        fig.align_ylabels(axes)
        axes[0].set_title(f"#{shot}", loc="left", fontsize=FONT + 0.5, fontweight="bold", pad=3)
        cells = [[("truth", cam_gt, tt), ("real", cam_pred, tt)] for tt in times]
        cg = camera_grid(fig, row[1], cells, row_labels=[f"{tt:.1f} s" for tt in times],
                         col_labels=("measured", "prediction") if i == 0 else (), crop=crop)
        box(cg[0][0], "lower divertor camera")
    legend(fig, [("truth", "measured"), ("real", "prediction")], y=0.945)
    fig.savefig(OUT / "fig3_detachment_replay.png"); fig.savefig(OUT / "fig3_detachment_replay.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------- figure 4
# Until the recommended trajectories exist, the actuators of measured discharges of the same
# session stand in: 201081's initial state (1-2 s) rolled out under the actuators of 201079 and
# 201083 (gas-fuelled like 201081) and of 201084 (the gas trajectory that should
# not detach). Actuator mode donor:<shot>.
# Each path differs from the others in shade and in line style (dash, dot, dash-dot), so that
# no two are told apart by dash pattern alone.
# Named by what the gas path does (source shot in brackets): A keeps the puff on longer
# than 201081, B starts it 0.5 s later, C has no puff (gas off at 2.5 s).
# Measured is solid, so the paths take the next styles in the standard order: dashed,
# dotted, dash-dot.
CANDIDATES = (("donor_201079", "A", (0, (4.0, 1.6)), "#201079", "#3d1150", "longer puff"),
              ("donor_201083", "B", (0, (1.0, 1.3)), "#201083", "#7b3294", "delayed puff"),
              ("donor_201084", "C", (0, (4.0, 1.4, 1.0, 1.4)), "#201084", "#b884d0", "no puff"))


def figure4():
    ctx, a = context(DATA / "demo_201081_cm_data.npz", channels=CHORDS["201081"])
    d, t, k0 = ctx["d"], ctx["t"], ctx["k0"]
    times = (2.5, 4.0, 5.5)
    fig = plt.figure(figsize=(WIDTH, 3.9))
    g = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.38, left=0.075, right=0.99,
                         top=0.88, bottom=0.1)
    left = g[0].subgridspec(3, 1, hspace=0.12)
    ax_gas, ax_ne, ax_br = (fig.add_subplot(left[r]) for r in range(3))
    cam_gt, _ = cams(ctx)
    crop = crop_of(cam_gt, k0)
    simple_trace(ax_gas, t, d[f"act__real__{GAS}"], "truth")
    trace_panel(ax_ne, ctx, a, "ts_core_density", runs=())
    brightness_panel(ax_br, ctx, crop, "fig4_gas_paths", runs=())
    n = "ts_core_density"
    for key, letter, ls, _src, col, _what in CANDIDATES:
        simple_trace(ax_gas, t, d[f"act__{key}__{GAS}"], "scenario", k0=k0, ls=ls, color=col)
        ne = pred_trace(ctx, a, key, n)
        draw_band(ax_ne, ctx, a, key, n, "scenario", k0 - 1, color=col)
        simple_trace(ax_ne, t, ne, "scenario", k0=k0, ls=ls, color=col)
        tb, br, kb = camera_series(ctx, crop, "fig4_gas_paths", key)
        simple_trace(ax_br, tb, br, "scenario", k0=kb, ls=ls, color=col)
    unit = short_label(ax_ne, n)
    ax_ne.texts[-1].set_text(ax_ne.texts[-1].get_text() + f" {unit}")
    ax_gas.set_ylabel("gas"); box(ax_gas, "GASA (Torr L/s)")
    for r, ax in enumerate((ax_gas, ax_ne, ax_br)):
        ax.set_xlim(1, 6)
        ax.axvline(T0 + FRAME * k0, color="0.55", lw=0.7, ls=":", zorder=1)
        headroom(ax, 0.15, zero=r == 0)
        if r < 2:
            ax.set_xticklabels([])
    ax_br.set_xlabel("time (s)")
    fig.align_ylabels([ax_gas, ax_ne, ax_br])
    ax_gas.set_title("#201081 initial state", loc="left", fontsize=FONT + 0.5, fontweight="bold", pad=3)
    cells = [[("truth", cam_gt, tt) for tt in times]]
    labels = ["experiment\n(measured)"]
    for key, letter, _ls, src, _col, what in CANDIDATES:
        cells.append([("scenario", cams(ctx, key)[1], tt) for tt in times])
        labels.append(f"{letter}: {what}\n#{src.lstrip(chr(35))}")
    cg = camera_grid(fig, g[1], cells, row_labels=labels, col_labels=[f"{tt:.1f} s" for tt in times],
                     crop=crop)
    box(cg[0][0], "lower divertor camera")
    extra = [(plt.Line2D([], [], color=col, lw=LW["scenario"], ls=ls), f"{letter}: {what}")
             for _k, letter, ls, src, col, what in CANDIDATES]
    legend(fig, [("truth", "experiment, measured")], y=0.94, extra=extra, ncol=4)
    fig.savefig(OUT / "fig4_detachment_paths.png"); fig.savefig(OUT / "fig4_detachment_paths.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------- figure 5
FIG5_FREE = os.environ.get("FIG5_FREE") == "1"
FREE = "0.55"  # rollout without the other diagnostics' measured tokens (baseline)
dc.CMAPS.setdefault("free", "Greys")
PALETTE.setdefault("free", FREE)
CER_GAPS_200729 = ((2.0, 2.5), (3.0, 3.5))  # beam modulation, Section "Further replay: 200729"


def infer_traces(ctx, a, shot, tag, n):
    """Traces of signal ``n`` of every inferred rollout (``INFER_CASES``), made as the
    plotted ones, or None before ``--infer-data`` has run. ``tag`` None: the replay's
    sampled rollouts (complete initial state)."""
    if tag is None:
        return member_traces(ctx, a, "real", n)
    path = INFER / f"infer_{shot}_{tag}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    k = f"members__infer__{n}"
    if k not in z.files:
        return None
    tr = np.stack([dc._trace(x, ctx["channels"][n], a.smooth) * dc.LABELS[n][1]
                   for x in z[k]])
    return np.abs(tr) if n in dc.ABSOLUTE else tr


def infer_panel(ax, ctx, a, n, shot, tag, gaps=(), free=False):
    """Measured chord (blank in ``gaps``) and the inferred prediction (mean of the seeds,
    band of one standard deviation); ``free`` adds the rollout without the other
    diagnostics' measured tokens (mean of its seeds)."""
    t, k0 = ctx["t"], ctx["k0"]
    gt = cut(ctx, dc._tr(ctx, a, "gt", "real", n))
    hole = np.zeros(len(t), bool)
    for lo, hi in gaps:
        hole |= (t >= lo) & (t < hi)
        ax.axvspan(lo, hi, color="0.9", lw=0, zorder=0)
    ax.plot(t, np.where(hole, np.nan, gt), color=PALETTE["truth"], lw=LW["truth"], zorder=2,
            solid_capstyle="round")
    if free:
        ax.plot(t[k0 - 1:], cut(ctx, pred_trace(ctx, a, "real", n))[k0 - 1:], color=FREE,
                lw=1.0, ls="--", zorder=3)
    tr = infer_traces(ctx, a, shot, tag, n)
    if tr is not None:
        mu, sd = cut(ctx, np.nanmean(tr, 0)), cut(ctx, np.nanstd(tr, 0))
        ax.fill_between(t[k0 - 1:], (mu - sd)[k0 - 1:], (mu + sd)[k0 - 1:],
                        color=PALETTE["real"], alpha=0.35, lw=0, zorder=3.5)
        ax.plot(t[k0 - 1:], mu[k0 - 1:], color=PALETTE["real"], lw=LW["real"], zorder=4)
    ax.axvline(T0 + FRAME * k0, color="0.55", lw=0.7, ls=":", zorder=1)
    ax.set_xlim(1, 6); ax.set_xlabel("time (s)"); ax.set_ylabel(dc.LABELS[n][0])
    headroom(ax, zero=n in dc.ZERO_BASED)


def figure5():
    ctx, a = context(DATA / "demo_204011_val_data.npz", channels=CHORDS["204011"])
    t, k0 = ctx["t"], ctx["k0"]

    def inferred_camera(tag, name):
        path = INFER / f"infer_204011_{tag}.npz"
        return (np.asarray(np.load(path)[f"camera_pred__infer__{name}"], np.float32)
                if path.exists() else None)

    # TS and CER of 204011 removed from the initial state (--hide), and the upper divertor
    # camera, which did not record: all diagnostics predicted autoregressively from 2 s,
    # nothing held at its measurement.
    upper_inf = inferred_camera("hide_tvupper", "tangtv_upper")
    height = 5.9
    fig = plt.figure(figsize=(WIDTH, height))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.05, 0.3],
                             hspace=0.55, left=0.08,
                             right=0.99, top=1 - 0.5 / height, bottom=0.01)
    top = outer[0].subgridspec(1, 3, wspace=0.35)
    # a, b: a measured diagnostic of 204011 removed from the initial state.
    for i, (n, tag, what) in enumerate((("ts_core_temp", "hide_ts", "TS"),
                                        ("cer_rot", "hide_cer", "CER"))):
        ax = fig.add_subplot(top[0, i])
        infer_panel(ax, ctx, a, n, "204011", tag, free=FIG5_FREE)
        box(ax, f"no {what} initial state: " + chord_label(ctx, n), size=FONT - 2.5)
        ax.text(-0.2, 1.04, "#204011", transform=ax.transAxes, fontsize=FONT, fontweight="bold")
    # c: CER of 200729 through the beam-modulation gaps, where there is no measurement:
    # the replay from the complete initial state.
    c2, a2 = context(DATA / "demo_200729_g1_data.npz", channels=CHORDS["200729"])
    ax = fig.add_subplot(top[0, 2])
    infer_panel(ax, c2, a2, "cer_ti", "200729", None, gaps=CER_GAPS_200729)
    ax.text(np.mean(CER_GAPS_200729[0]), 0.04, "no CER", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=FONT - 1.5, color="0.35")
    box(ax, chord_label(c2, "cer_ti"), size=FONT - 2.5)
    ax.text(-0.2, 1.04, "#200729", transform=ax.transAxes, fontsize=FONT, fontweight="bold")
    # d: profiles of the removed diagnostics against psi_N at the camera times: measured
    # chords as dots, the inferred prediction as the mean of its seeds with a band of one
    # standard deviation.
    times = (2.5, 3.5, 4.5, 5.5)
    rows = outer[1].subgridspec(2, len(times), hspace=0.12, wspace=0.08)
    for r, (n, tag, what) in enumerate((("ts_core_temp", "hide_ts", "TS"),
                                        ("cer_rot", "hide_cer", "CER"))):
        chans, _ = profile_chords(ctx, n)
        gt_p, _ = profile_frames(ctx, "gt", "real", n, chans)
        path = INFER / f"infer_204011_{tag}.npz"
        inf = None
        if path.exists():
            z = np.load(path)
            each = np.stack([smooth_profile(n, np.asarray(x, float)[:, chans] * dc.LABELS[n][1])
                             for x in z[f"members__infer__{n}"]])
            inf = np.nanmean(each, 0), np.nanstd(each, 0)
        axes = []
        for j, ts in enumerate(times):
            ax = fig.add_subplot(rows[r, j], sharey=axes[0] if axes else None)
            k = frame_at(ctx, ts)
            x = chord_psin("204011", n, chans, [t[k]])[0]
            o = np.argsort(x)
            if inf is not None:
                mu, sd = inf
                ax.fill_between(x[o], (mu[k] - sd[k])[o], (mu[k] + sd[k])[o],
                                color=PALETTE["real"], alpha=0.3, lw=0, zorder=3)
                ax.plot(x[o], mu[k][o], color=PALETTE["real"], lw=LW["real"], zorder=4)
            ax.plot(x[o], gt_p[k][o], ls="none", marker="o", ms=2.6, mew=0,
                    color=PALETTE["truth"], zorder=5)
            if r == 0:
                ax.set_title(f"{ts:g} s", fontsize=FONT - 0.5, pad=2)
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel(PSIN_LABEL)
            if j == 0:
                ax.set_ylabel(dc.LABELS[n][0])
                box(ax, f"no {what} initial state", size=FONT - 2.5)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            axes.append(ax)
        psin_axes(axes)
        headroom(axes[0], 0.25, zero=True)
    # e: the upper divertor camera of 204011, which did not record: predicted only.
    up_cells = [[("real", upper_inf, tt) if upper_inf is not None else ("real", None, "pending")
                 for tt in times]]
    axs = camera_grid(fig, outer[2], up_cells, col_labels=[f"{tt:.1f} s" for tt in times],
                      row_labels=["predicted"],
                      crop=crop_of(upper_inf, k0) if upper_inf is not None else None,
                      wspace=0.03, hspace=0.06)
    p0 = axs[0][0].get_position()
    fig.text(p0.x0, p0.y1 + 0.03, "#204011, upper divertor camera: not recorded",
             fontsize=FONT, color="0.25", ha="left", va="bottom")
    legend(fig, [("truth", "measured"), ("real", "predicted")], y=0.985,
           extra=[(plt.Line2D([], [], color=FREE, lw=1.0, ls="--"), "replay")]
           if FIG5_FREE else ())
    fig.savefig(OUT / "fig5_missing_diagnostics.png"); fig.savefig(OUT / "fig5_missing_diagnostics.pdf")
    plt.close(fig)


def profile_diff_panel(fig, spec, ctx, n, key, start, step=5, t_range=(3.5, 6.0),
                       colorbar=True):
    """Changed actuator minus replay of profile ``n`` against psi_N, one line
    every ``step`` frames from ``start``, coloured by time; band: the two runs' standard
    deviations over the sampled rollouts, added in quadrature."""
    from matplotlib import cm
    from matplotlib.colors import Normalize
    t = ctx["t"]
    end = live_end(ctx)
    chans, _ = profile_chords(ctx, n)
    frames = list(range(start, min(end + 1, len(t)), step))
    psin = chord_psin(ctx["shot"], n, chans, t[frames])
    norm = Normalize(vmin=t_range[0], vmax=t_range[1])
    cmap = matplotlib.colormaps["viridis"]
    sub = spec.subgridspec(1, 2, width_ratios=[1.0, 0.04], wspace=0.05)
    ax = fig.add_subplot(sub[0])
    mu_r, sd_r = profile_frames(ctx, "pred", "real", n, chans)
    mu_c, sd_c = profile_frames(ctx, "pred", key, n, chans)
    for j, k in enumerate(frames):
        col = cmap(norm(t[k]))
        o = np.argsort(psin[j])
        dlt = (mu_c[k] - mu_r[k])[o]
        if sd_r is not None and sd_c is not None:
            err = np.hypot(sd_r[k], sd_c[k])[o]
            ax.fill_between(psin[j][o], dlt - err, dlt + err, color=col, alpha=0.2, lw=0)
        ax.plot(psin[j][o], dlt, color=col, lw=1.1)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.axvline(1.0, color="0.6", lw=0.6, ls=":", zorder=0)
    ax.set_xlabel(PSIN_LABEL)
    ax.set_ylabel(r"$\Delta$" + dc.LABELS[n][0])
    box(ax, "changed minus replay", size=FONT - 2.5)
    if colorbar:
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_subplot(sub[1]))
        cb.set_label("time (s)", fontsize=FONT - 1)
        cb.ax.tick_params(labelsize=FONT - 1.5)
    return ax


DESIGNER_RUN = Path(os.environ.get("PAPER_FIG_DESIGNER", "/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/designer_runs/agent_204011_rotation_v5/soup"))
DESIGNER_DATA = WORK / "designer_204011_data.npz"


def extract_designer(run=DESIGNER_RUN):
    """Decode the designer run's sampled rollouts (CER rotation and T_i): measured
    (``gt``), the same seed under the measured actuators (``real``) and under the plan
    (``pred``), each (seeds, F, C) frame values in physical units; plus the plan's and the
    measured actuators. Decode only."""
    import torch
    from tokamak_foundation_model.ignite import eval_dynamics as ev
    from tokamak_foundation_model.ignite import train_dynamics as td
    names = ("cer_rot", "cer_ti")
    tmpl = "/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_codecs_v4/{m}/codec_best.pt"
    codecs = td.load_frozen_codecs(list(names), repo=REPO, tmpl=tmpl)
    conv = ev.profile_converters(codecs, list(names))
    dev = torch.device("cpu")
    arcs = [np.load(p, allow_pickle=True) for p in sorted(run.glob("seed*/codes_204011_step*.npz"))]
    out = {"K0": int(arcs[0]["K0"]), "F": int(arcs[0]["F"])}
    F = out["F"]
    for n in names:
        for which, key in (("gt", "gt"), ("real", "real"), ("pred", "pred")):
            vals = []
            for z in arcs[:1] if which == "gt" else arcs:
                x = ev.decode_flat_chunked(codecs[n][0], ev.archived(z, f"{key}__{n}")[:F], dev)
                with np.errstate(invalid="ignore"):
                    vals.append(np.nanmean(conv[n](x), axis=-1).astype(np.float32))
            out[f"{which}__{n}"] = np.stack(vals)
    # A run is the plan dir itself or one model's rollouts inside it (<plan>/soup).
    plan = np.load(run / "plan.npz" if (run / "plan.npz").exists() else run.parent / "plan.npz")
    out["act_plan"], out["act_measured"] = plan["act"][:F], plan["measured"][:F]
    np.savez_compressed(DESIGNER_DATA, **out)
    print("wrote", DESIGNER_DATA, len(arcs), "rollouts")


def figure6():
    """Designed trajectory: reactor-like rotation in 204011 (zero torque, CER kept).

    Top: mean beam power per source over 4.5-5 s, measured and planned; injected torque
    and total beam power over time. Middle: core |v_tor| (CER T02) and T_i (CER T01),
    measured, replay and plan (mean of the sampled rollouts, shaded one standard
    deviation), profile times in grey. Bottom: the rotation profile against R at the
    profile times of Fig. 2, one panel per time, as in Figs. 1 and 2."""
    from tokamak_foundation_model.ignite.designer import machine as mc
    d = np.load(DESIGNER_DATA)
    K0, F = int(d["K0"]), int(d["F"])
    t = T0 + FRAME * np.arange(F)
    end = min(F, int((5.778 - T0) / FRAME))  # plasma end of 204011 (Ip)
    sw = int(round((3.5 - T0) / FRAME))
    P, T = mc.columns("pinj"), mc.columns("tinj")
    plan, meas = d["act_plan"], d["act_measured"]
    COL = {"truth": PALETTE["truth"], "real": PALETTE["real"], "scenario": PALETTE["scenario"]}

    def smooth(x):  # 3-frame centred mean, as in the other traces
        return np.stack([np.nanmean(x[..., max(0, k - 1):k + 2], axis=-1) for k in range(x.shape[-1])], -1)

    def chord(which, n, c):
        x = np.abs(d[f"{which}__{n}"][:, :, c]) if n == "cer_rot" else d[f"{which}__{n}"][:, :, c]
        return smooth(x * dc.LABELS[n][1])

    height = 5.6
    fig = plt.figure(figsize=(WIDTH, height))
    g = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.5, wspace=0.42, left=0.08,
                         right=0.98, top=1 - 0.55 / height, bottom=0.07)
    # a: mean power per source.
    ax = fig.add_subplot(g[0, 0])
    # Same window as the beam powers quoted in the text: after the balance ramp of
    # Fig. 2, before the measured beams ramp down at 5.1 s.
    w = slice(int(round((4.5 - T0) / FRAME)), int(round((5.0 - T0) / FRAME)))
    xb = np.arange(len(mc.BEAMS))
    pm = meas[w, P].mean(0) * 1e-6
    pp = plan[w, P].mean(0) * 1e-6
    ax.bar(xb - 0.2, pm, 0.38, color=COL["truth"], label="measured")
    ax.bar(xb + 0.2, pp, 0.38, color=COL["scenario"], label="plan")
    ax.set_xticks(xb)
    # The paper names the beam sources by beamline angle (NB21L is the 210L source).
    names = {"NB15L": "150L", "NB15R": "150R", "NB21L": "210L", "NB21R": "210R",
             "NB30L": "30L", "NB30R": "30R", "NB33L": "330L", "NB33R": "330R"}
    ax.set_xticklabels([names.get(b, b) for b in mc.BEAMS], rotation=60, fontsize=FONT - 2)
    ax.set_ylabel("beam power (MW)")
    box(ax, "4.5\u20135.0 s", size=FONT - 2.5)
    # b, c: torque and total power over time.
    for col_, (cols, lab, scale) in zip((1, 2), ((T, r"$T_\mathrm{inj}$ (N m)", 1.0),
                                                 (P, r"$P_\mathrm{NBI}$ (MW)", 1e-6))):
        ax = fig.add_subplot(g[0, col_])
        ax.plot(t[:end], meas[:end, cols].sum(1) * scale, color=COL["truth"], lw=LW["truth"])
        ax.plot(t[sw - 1:end], plan[sw - 1:end, cols].sum(1) * scale, color=COL["scenario"],
                lw=LW["scenario"], ls="--")
        ax.axvline(3.5, color=COL["scenario"], lw=0.8, ls=":")
        ax.set_ylabel(lab)
        ax.set_xlim(1, 6)
        headroom(ax, 0.2, zero=True)
        ax.set_xlabel("time (s)")
    # d, e: core rotation and T_i over time.
    mid = g[1, :].subgridspec(1, 2, wspace=0.25)
    for col_, (n, c, lab) in enumerate((("cer_rot", 1, "CER T02 (R 1.85, Z -0.01 m)"),
                                        ("cer_ti", 0, "CER T01 (R 1.76, Z 0.00 m)"))):
        ax = fig.add_subplot(mid[col_])
        gt = chord("gt", n, c)[0]
        ax.plot(t[:end], gt[:end], color=COL["truth"], lw=LW["truth"], zorder=2)
        for which, role, ls, k0_ in (("real", "real", "-", K0 - 1), ("pred", "scenario", "--", K0 - 1)):
            x = chord(which, n, c)
            mu, sd = np.nanmean(x, 0), np.nanstd(x, 0)
            ax.fill_between(t[k0_:end], (mu - sd)[k0_:end], (mu + sd)[k0_:end], color=COL[role],
                            alpha=0.35, lw=0, zorder=2.5)
            ax.plot(t[k0_:end], mu[k0_:end], color=COL[role], lw=LW[role], ls=ls, zorder=3)
        ax.axvline(3.5, color=COL["scenario"], lw=0.8, ls=":")
        ax.axvline(T0 + FRAME * K0, color="0.55", lw=0.7, ls=":")
        mark_times(ax, PROFILE_TIMES_FIG2)
        ax.set_ylabel(dc.LABELS[n][0])
        ax.set_xlim(1, 6)
        ax.set_xlabel("time (s)")
        headroom(ax, 0.25, zero=True)
        box(ax, lab.replace("-", "\u2212"), size=FONT - 2.5)
    # f: the rotation profile against psi_N at the profile times of Fig. 2, one panel per
    # time: measured chords as dots, replay and plan as lines with a band of one standard
    # deviation of the sampled rollouts (3-frame smoothing, as the traces).
    # The chords of Fig. 1's profiles (same shot and measurement).
    rec = list(profile_chords({"shot": "204011",
                               "d": np.load(DATA / "demo_204011_val_data.npz",
                                            allow_pickle=True)}, "cer_rot")[0])

    def profile(which):
        x = np.abs(d[f"{which}__cer_rot"][:, :, rec]) * dc.LABELS["cer_rot"][1]
        x = np.stack([np.nanmean(x[:, max(0, k - 1):k + 2], axis=1) for k in range(F)], 1)
        return np.nanmean(x, 0), np.nanstd(x, 0)

    gt_p, _ = profile("gt")
    sl = g[2, :].subgridspec(1, len(PROFILE_TIMES_FIG2), wspace=0.1)
    prof = []
    for j_, ts in enumerate(PROFILE_TIMES_FIG2):
        ax = fig.add_subplot(sl[j_], sharey=prof[0] if prof else None)
        k = int(round((ts - T0) / FRAME))
        R = chord_psin("204011", "cer_rot", rec, [ts])[0]
        o = np.argsort(R)
        R = R[o]
        for which, role, ls in (("real", "real", "-"), ("pred", "scenario", "--")):
            mu, sd = profile(which)
            ax.fill_between(R, (mu[k] - sd[k])[o], (mu[k] + sd[k])[o], color=COL[role],
                            alpha=0.3, lw=0, zorder=3)
            ax.plot(R, mu[k][o], color=COL[role], lw=LW[role], ls=ls, zorder=4)
        ax.plot(R, gt_p[k][o], ls="none", marker="o", ms=2.6, mew=0, color=COL["truth"], zorder=5)
        ax.text(0.97, 0.97, f"{ts:g} s", transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT - 1.5, color="0.3", zorder=10)
        ax.set_xlabel(PSIN_LABEL)
        if j_ == 0:
            ax.set_ylabel(dc.LABELS["cer_rot"][0])
            box(ax, "CER", size=FONT - 2.5)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        prof.append(ax)
    psin_axes(prof)
    headroom(prof[0], 0.3, zero=True)
    legend(fig, [("truth", "measured"), ("real", "replay (measured actuators)"),
                 ("scenario", "designed trajectory and its prediction")], y=0.985, ncol=2,
           extra=[(plt.Line2D([], [], color="0.7", lw=0.6), "profile times")])
    fig.savefig(OUT / "fig6_designed_trajectory.png"); fig.savefig(OUT / "fig6_designed_trajectory.pdf")
    plt.close(fig)


def extra_ts_profiles():
    """Not a paper figure: TS T_e and n_e from core to edge (psi_N, EFIT01) for the
    replay of 204011 (measured vs prediction) and the torque -> 0 and ECH -> 0 changes
    (replay vs changed), time-coloured with the band of the sampled predictions."""
    cases = [("demo_204011_val_data.npz", "204011", [("gt", "real"), ("pred", "real")],
              ["measured", "prediction"], "k0", 10, (2.0, 6.0), "#204011 replay"),
             ("ramps/demo_204011_ramps_data.npz", "204011_torque",
              [("pred", "real"), ("pred", "tq_lin_zero")], ["replay", r"torque $\to$ 0"],
              "sw", 5, (3.5, 6.0), r"#204011, torque $\to$ 0"),
             ("ramps/demo_201240_ramps_data.npz", "201240_ech",
              [("pred", "real"), ("pred", "ech_lin_off")], ["replay", r"ECH $\to$ 0"],
              "sw", 5, (3.5, 6.0), r"#201240, ECH $\to$ 0")]
    fig = plt.figure(figsize=(WIDTH, 6.6))
    g = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.35, left=0.09, right=0.93, top=0.95,
                         bottom=0.07)
    for r, (name, chords, runs, labels, start, step, rng, title) in enumerate(cases):
        ctx, a = context(DATA / name, channels=CHORDS[chords])
        for c, n in enumerate(("ts_core_temp", "ts_core_density")):
            ax = profile_time_panel(fig, g[r, c], ctx, n, runs, labels, start=ctx[start],
                                    step=step, t_range=rng, colorbar=c == 1, legend=c == 0,
                                    free_bottom=0)
            if c == 0:
                ax.set_title(title, loc="left", fontsize=FONT, pad=3)

    out = OUT / "extra"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "ts_profiles_psin.png"); fig.savefig(out / "ts_profiles_psin.pdf")
    plt.close(fig)
    print("wrote", out / "ts_profiles_psin.pdf")


if __name__ == "__main__":
    if "--designer-data" in sys.argv:
        extract_designer()
        sys.exit(0)
    if "--extra" in sys.argv:
        style()
        extra_ts_profiles()
        sys.exit(0)
    if "--infer-data" in sys.argv:
        extract_infer()
        sys.exit(0)
    if "--members" in sys.argv:
        for name in MEMBER_FILES:
            extract_members(DATA / name)
        sys.exit(0)
    style()
    only = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--only=")), "1,2,3,4,5")
    for i in only.split(","):
        for old in OUT.glob(f"fig{i}_*.p*"):
            old.unlink()
        globals()[f"figure{i}"]()
    print("figures in", OUT)
