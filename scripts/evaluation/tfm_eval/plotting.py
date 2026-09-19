"""Single plotting/style layer for the tokamak-foundation-model eval suite.

Every eval driver imports figures from here so the whole suite reads as one
system (Nature/NeurIPS print style). Call :func:`set_style` once per process
before plotting; figure-producing helpers call it themselves.

Design rules (do not deviate in drivers):

* **Categorical color** = Okabe-Ito palette in fixed order (``OKABE_ITO``),
  never cycled beyond 8 series — fold extras into "other" or facet instead.
  Color follows the *entity*: a modality keeps its ``MODALITY_COLORS`` hue in
  every figure. Hues may repeat across modality groups (slow-TS vs
  spectrogram); identity is therefore never color-alone — legends or direct
  labels always accompany multi-series panels.
* **Ground truth** solid black, **prediction** dashed blue, **persistence**
  dotted gray (``GT_STYLE`` / ``PRED_STYLE`` / ``COPY_STYLE``).
* **Colormaps**: sequential magnitude = viridis, |error| = magma, signed
  diverging = RdBu_r centered at 0 via ``TwoSlopeNorm``. Never rainbow/jet.
* **One y-axis per axes** — never ``twinx``.
* Recessive grid (alpha 0.15, below data), no top/right spines, legend only
  when >= 2 series, direct labels where cleaner. Text is always neutral ink
  (black/gray), never set in a series color.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker as mticker
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

# --------------------------------------------------------------------------- #
# palette                                                                      #
# --------------------------------------------------------------------------- #

#: Okabe-Ito colorblind-safe palette, FIXED assignment order. Validated
#: (adjacent-pair CVD ΔE >= 8, normal-vision ΔE 20). Yellow is a last resort
#: on white; black doubles as the neutral/GT ink.
OKABE_ITO: list[str] = [
    "#0072B2",  # 0 blue
    "#D55E00",  # 1 vermillion
    "#009E73",  # 2 green
    "#E69F00",  # 3 orange
    "#CC79A7",  # 4 purple-pink
    "#56B4E9",  # 5 sky
    "#F0E442",  # 6 yellow (last resort on white)
    "#000000",  # 7 black
]

#: Stable per-diagnostic colors — the same modality wears the same hue in
#: every figure of the suite. Groups may share a hue *family*: the 7 slow
#: time-series take all non-yellow slots (TS densities = blue family, TS
#: temperatures = warm family); filterscopes gets the remaining unshared slot;
#: spectrograms/video are distinct from each other and reuse slow-TS hues
#: across groups only. Blue stays unique to ts_core_density and green to
#: cer_ti, so the common mixed panel (core density / Ti / mse / filterscopes
#: / ece) is fully distinct. In any panel that does mix colliding pairs
#: (e.g. ece with ts_core_temp), direct labels carry identity.
MODALITY_COLORS: dict[str, str] = {
    # slow time-series (7)
    "ts_core_density": OKABE_ITO[0],        # blue
    "ts_tangential_density": OKABE_ITO[5],  # sky        (TS density family)
    "ts_core_temp": OKABE_ITO[1],           # vermillion (TS temp family)
    "ts_tangential_temp": OKABE_ITO[3],     # orange     (TS temp family)
    "cer_ti": OKABE_ITO[2],                 # green
    "cer_rot": OKABE_ITO[4],                # purple-pink
    "mse": OKABE_ITO[7],                    # black
    # fast photodiodes — own (unshared) slot; pair with labels on white
    "filterscopes": OKABE_ITO[6],           # yellow
    # spectrograms + video — distinct from each other, hue reuse across groups
    "ece": OKABE_ITO[1],                    # vermillion (~ ts_core_temp)
    "co2": OKABE_ITO[5],                    # sky        (~ ts_tang_density)
    "bes": OKABE_ITO[3],                    # orange     (~ ts_tang_temp)
    "tangtv": OKABE_ITO[4],                 # purple     (~ cer_rot)
}

#: Line styles for the three canonical series roles (splat into ``ax.plot``).
#: zorder layers dashed prediction above solid GT, persistence underneath.
GT_STYLE: dict = {
    "color": "#000000", "linestyle": "-", "linewidth": 1.5, "zorder": 2.2,
}
PRED_STYLE: dict = {
    "color": "#0072B2", "linestyle": "--", "linewidth": 1.5, "zorder": 2.5,
}
COPY_STYLE: dict = {
    "color": "#7F7F7F", "linestyle": ":", "linewidth": 1.5, "zorder": 1.9,
}

#: Colormap roles — sequential magnitude / |error| / signed-diverging.
SEQ_CMAP = "viridis"
ERR_CMAP = "magma"
DIV_CMAP = "RdBu_r"

_NEUTRAL_INK = "#333333"      # annotation/direct-label ink (never series color)
_ROW_LABEL_INK = "#444444"

# --------------------------------------------------------------------------- #
# style                                                                        #
# --------------------------------------------------------------------------- #

_RC: dict = {
    # typography — Nature single column (~89 mm) reads at 8-9 pt
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.titleweight": "normal",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 9.5,
    "figure.titleweight": "normal",
    "mathtext.fontset": "dejavusans",
    # line/frame weights for print
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.5,
    "lines.markersize": 3.5,
    # recessive grid below the data; quiet frame
    "axes.grid": True,
    "grid.color": "#000000",
    "grid.alpha": 0.15,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # color defaults
    "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    "image.cmap": SEQ_CMAP,
    # legends: compact, borderless white scrim so text stays readable when
    # loc="best" lands over data (text never sits raw on a trace)
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.facecolor": "white",
    "legend.framealpha": 0.8,
    "legend.edgecolor": "none",
    "legend.handlelength": 1.8,
    "legend.borderaxespad": 0.4,
    # layout / output
    "figure.dpi": 130,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # keep text editable in vector output
    "ps.fonttype": 42,
}


def set_style() -> None:
    """Apply the suite's print rcParams (idempotent; call freely)."""
    mpl.rcParams.update(_RC)


def save_fig(
    fig: Figure,
    stem: str | Path,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 300,
) -> list[Path]:
    """Save ``fig`` as ``<stem>.<ext>`` for each format; returns the paths.

    Parent directories are created; bounding box is tight.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in formats:
        path = stem.parent / f"{stem.name}.{ext.lstrip('.')}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- #
# time-series / spectral panels                                                #
# --------------------------------------------------------------------------- #


def ts_overlay(
    ax: Axes,
    t,
    gt,
    pred=None,
    copy=None,
    ylabel: str = "",
    title: str = "",
    pred_label: str = "prediction",
) -> None:
    """Overlay ground truth (solid black) with prediction / persistence.

    Legend appears only when >= 2 series are drawn; the caller owns the
    x-label (drivers often share a time axis across stacked panels).
    """
    ax.plot(t, gt, label="ground truth", **GT_STYLE)
    n_series = 1
    if pred is not None:
        ax.plot(t, pred, label=pred_label, **PRED_STYLE)
        n_series += 1
    if copy is not None:
        ax.plot(t, copy, label="persistence", **COPY_STYLE)
        n_series += 1
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if n_series >= 2:
        ax.legend(loc="best")


def _as_freq_time(a, n_freq: int, n_time: int, name: str) -> np.ndarray:
    """Coerce a spectrogram to ``(n_freq, n_time)``, transposing if needed."""
    a = np.asarray(a, dtype=float)
    if a.shape == (n_freq, n_time):
        return a
    if a.shape == (n_time, n_freq):
        return a.T
    raise ValueError(
        f"{name} shape {a.shape} matches neither (n_freq={n_freq}, "
        f"n_time={n_time}) nor its transpose"
    )


def spectro_triptych(
    gt,
    pred,
    t_s,
    f_hz,
    title: str = "",
    log_power: bool = True,
    figwidth: float = 7.0,
) -> Figure:
    """GT / prediction / |difference| spectrogram rows on a shared time axis.

    GT and prediction share ``vmin/vmax`` from the GT [1, 99] percentiles
    (of the log10 display when ``log_power``); the |difference| row (in dex
    when ``log_power``) gets its own magma scale. Frequency is shown in kHz;
    ``t_s``/``f_hz`` are assumed uniformly spaced (STFT grids).
    """
    set_style()
    t_s = np.asarray(t_s, dtype=float)
    f_hz = np.asarray(f_hz, dtype=float)
    gt = _as_freq_time(gt, f_hz.size, t_s.size, "gt")
    pred = _as_freq_time(pred, f_hz.size, t_s.size, "pred")

    if log_power:
        pos = gt[gt > 0]
        floor = float(pos.min()) if pos.size else 1e-12
        floor = max(floor, float(np.nanmax(np.abs(gt))) * 1e-12, 1e-300)
        disp_gt = np.log10(np.maximum(gt, floor))
        disp_pred = np.log10(np.maximum(pred, floor))
        mag_label = r"$\log_{10}$ power (a.u.)"
        diff_label = r"$|\Delta|$ (dex)"
    else:
        disp_gt, disp_pred = gt, pred
        mag_label = "power (a.u.)"
        diff_label = r"$|\Delta|$ (a.u.)"

    vmin, vmax = (float(v) for v in np.nanpercentile(disp_gt, [1.0, 99.0]))
    if not vmin < vmax:  # degenerate (constant) field
        vmin, vmax = vmin - 0.5, vmax + 0.5
    diff = np.abs(disp_gt - disp_pred)
    dmax = float(np.nanpercentile(diff, 99.0))
    if dmax <= 0:
        dmax = max(float(np.nanmax(diff)), 1e-12)

    f_khz = f_hz / 1e3
    extent = (t_s[0], t_s[-1], f_khz[0], f_khz[-1])
    imshow_kw = dict(origin="lower", aspect="auto", extent=extent)

    fig, axs = plt.subplots(
        3, 1, sharex=True, figsize=(figwidth, 0.62 * figwidth)
    )
    im_gt = axs[0].imshow(disp_gt, cmap=SEQ_CMAP, vmin=vmin, vmax=vmax,
                          **imshow_kw)
    axs[1].imshow(disp_pred, cmap=SEQ_CMAP, vmin=vmin, vmax=vmax, **imshow_kw)
    im_df = axs[2].imshow(diff, cmap=ERR_CMAP, vmin=0.0, vmax=dmax,
                          **imshow_kw)

    row_labels = ("ground truth", "prediction", "|difference|")
    for ax, lab in zip(axs, row_labels):
        ax.grid(False)
        # row label on the left as the per-axes ylabel (layout-managed, so it
        # can never collide); the shared physical axis label is a supylabel.
        ax.set_ylabel(lab, fontsize=8.0, color=_ROW_LABEL_INK)
    fig.supylabel("frequency (kHz)", fontsize=8.5)
    axs[2].set_xlabel("time (s)")

    cb = fig.colorbar(im_gt, ax=list(axs[:2]), fraction=0.03, pad=0.02)
    cb.set_label(mag_label)
    cb_df = fig.colorbar(im_df, ax=axs[2], fraction=0.03, pad=0.02)
    cb_df.set_label(diff_label)
    if title:
        fig.suptitle(title)
    return fig


def psd_overlay(
    ax: Axes,
    f_hz,
    psd_gt,
    psd_pred,
    xlog: bool = True,
    ylog: bool = True,
) -> None:
    """Overlay GT (solid black) and predicted (dashed blue) power spectra."""
    ax.plot(f_hz, psd_gt, label="ground truth", **GT_STYLE)
    ax.plot(f_hz, psd_pred, label="prediction", **PRED_STYLE)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (a.u.)")
    ax.legend(loc="best")


# --------------------------------------------------------------------------- #
# matrices / summary panels                                                    #
# --------------------------------------------------------------------------- #


def signed_heatmap(
    matrix,
    row_labels,
    col_labels,
    annot=None,
    title: str = "",
    cbar_label: str = "",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Signed matrix on RdBu_r with 0 pinned to the neutral midpoint.

    ``annot`` (optional) is an array of strings, one per cell, drawn in
    neutral ink (near-black or near-white chosen per cell luminance —
    identity never rides on colored text).
    """
    set_style()
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {m.shape}")
    n_rows, n_cols = m.shape

    finite = m[np.isfinite(m)]
    lo = float(finite.min()) if finite.size else -1.0
    hi = float(finite.max()) if finite.size else 1.0
    eps = max(abs(lo), abs(hi), 1e-12) * 1e-6  # TwoSlopeNorm needs lo < 0 < hi
    norm = TwoSlopeNorm(vcenter=0.0, vmin=min(lo, -eps), vmax=max(hi, eps))

    if figsize is None:
        figsize = (
            min(7.0, 2.0 + 0.55 * n_cols),
            min(7.0, 1.0 + 0.42 * n_rows),
        )
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(m, cmap=DIV_CMAP, norm=norm, aspect="auto")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(range(n_cols), labels=[str(c) for c in col_labels])
    ax.set_yticks(range(n_rows), labels=[str(r) for r in row_labels])
    ax.tick_params(length=0)
    if max(len(str(c)) for c in col_labels) > 5:
        plt.setp(
            ax.get_xticklabels(), rotation=40, ha="right",
            rotation_mode="anchor",
        )

    if annot is not None:
        annot = np.asarray(annot, dtype=object)
        if annot.shape != m.shape:
            raise ValueError(
                f"annot shape {annot.shape} != matrix shape {m.shape}"
            )
        cmap = plt.get_cmap(DIV_CMAP)
        for i in range(n_rows):
            for j in range(n_cols):
                text = annot[i, j]
                if text is None or str(text) == "":
                    continue
                if np.isfinite(m[i, j]):
                    r, g, b, _ = cmap(norm(m[i, j]))
                    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    ink = "#1a1a1a" if lum > 0.45 else "#f2f2f2"
                else:
                    ink = "#1a1a1a"
                ax.text(j, i, str(text), ha="center", va="center",
                        fontsize=7.0, color=ink)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if cbar_label:
        cb.set_label(cbar_label)
    if title:
        ax.set_title(title)
    return fig


def efficiency_curve(
    ax: Axes,
    x,
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    xlabel: str = "labeled shots",
    ylabel: str = "AUROC",
) -> None:
    """Label-efficiency curves: median line + shaded IQR band per method.

    ``curves`` maps method name -> ``(median, lo, hi)``; colors follow the
    fixed ``OKABE_ITO`` order of insertion (max 8 methods — fold extras into
    "other" or facet). The x-axis is log2 with ticks at the given ``x``.
    """
    if len(curves) > len(OKABE_ITO):
        raise ValueError(
            f"{len(curves)} methods > {len(OKABE_ITO)} fixed colors; fold "
            "extras into 'other' or use small multiples (never cycle hues)"
        )
    x = np.asarray(x, dtype=float)
    for i, (name, (med, lo, hi)) in enumerate(curves.items()):
        color = OKABE_ITO[i]
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, med, color=color, linewidth=1.5, marker="o",
                markersize=3.2, label=name, zorder=2.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(curves) >= 2:
        ax.legend(loc="best")


def rollout_divergence(
    ax: Axes,
    k,
    curves: dict[str, np.ndarray],
    ratio_line: float | None = 1.0,
    xlabel: str = "rollout step k",
) -> None:
    """Error growth vs rollout step, one line per modality, direct-labeled.

    Curve names found in ``MODALITY_COLORS`` wear their fixed modality color;
    other names take ``OKABE_ITO`` in order of insertion. Lines are labeled
    directly at their right end in neutral ink (no legend); ``ratio_line``
    draws a thin gray reference (e.g. parity with persistence) — pass
    ``None`` to omit.
    """
    k = np.asarray(k, dtype=float)
    if ratio_line is not None:
        ax.axhline(ratio_line, color="#999999", linewidth=0.8,
                   linestyle=(0, (2, 2)), zorder=1.5)

    n_fallback = 0
    ends: list[tuple[str, float]] = []
    for name, y in curves.items():
        y = np.asarray(y, dtype=float)
        color = MODALITY_COLORS.get(name)
        if color is None:
            if n_fallback >= len(OKABE_ITO):
                raise ValueError(
                    "ran out of fixed categorical colors; facet or fold "
                    "series into 'other' (never cycle hues)"
                )
            color = OKABE_ITO[n_fallback]
            n_fallback += 1
        ax.plot(k, y, color=color, linewidth=1.5, zorder=2.3)
        finite = y[np.isfinite(y)]
        ends.append((name, float(finite[-1]) if finite.size else 0.0))

    ax.set_xlim(k.min(), k.max())
    ax.set_xlabel(xlabel)

    # direct end labels in neutral ink, nudged apart when line ends crowd
    ymin, ymax = ax.get_ylim()
    gap = 0.055 * (ymax - ymin)
    ends.sort(key=lambda item: item[1])
    placed: list[float] = []
    for _, y_end in ends:
        placed.append(y_end if not placed else max(y_end, placed[-1] + gap))
    if placed and placed[-1] > ymax:  # keep the top label inside the frame
        ax.set_ylim(ymin, placed[-1] + 0.5 * gap)
    for (name, _), y_lab in zip(ends, placed):
        ax.annotate(
            name, xy=(k[-1], y_lab), xytext=(5, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=7.0, color=_NEUTRAL_INK, annotation_clip=False,
        )


# --------------------------------------------------------------------------- #
# self-test                                                                    #
# --------------------------------------------------------------------------- #


def _demo(out_dir: Path) -> list[Path]:
    """Render one example of every helper with synthetic data."""
    mpl.use("Agg", force=True)  # explicit headless backend for the self-test
    set_style()
    rng = np.random.default_rng(7)
    files: list[Path] = []

    # 1 -- ts_overlay -------------------------------------------------------
    t = np.linspace(0.0, 2.0, 400)
    gt = (
        1.2
        + 0.45 * np.sin(2 * np.pi * 2.2 * t)
        + 0.12 * np.sin(2 * np.pi * 9.0 * t)
        + 0.02 * rng.standard_normal(t.size)
    )
    pred = np.interp(t - 0.03, t, gt) + 0.05 * rng.standard_normal(t.size)
    copy = np.interp(t - 0.12, t, gt)  # persistence-style lagged copy
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ts_overlay(
        ax, t, gt, pred=pred, copy=copy,
        ylabel=r"$n_e$ ($10^{19}\,$m$^{-3}$)",
        title="ts_core_density — shot 191914, ch 12",
    )
    ax.set_xlabel("time (s)")
    files += save_fig(fig, out_dir / "ts_overlay")
    plt.close(fig)

    # 2 -- spectro_triptych --------------------------------------------------
    n_f, n_t = 120, 240
    t_s = np.linspace(0.0, 2.0, n_t)
    f_hz = np.linspace(0.0, 100e3, n_f)
    fgrid, tgrid = np.meshgrid(f_hz, t_s, indexing="ij")
    ridge = 30e3 + 12e3 * np.sin(2 * np.pi * 0.7 * tgrid) + 6e3 * tgrid
    gt_spec = (
        1e-6
        + np.exp(-(((fgrid - ridge) / 4e3) ** 2))
        + 0.4 * np.exp(-(((fgrid - 2 * ridge) / 6e3) ** 2))
    ) * np.exp(0.25 * rng.standard_normal((n_f, n_t)))
    pred_spec = (
        1e-6
        + np.exp(-(((fgrid - 1.04 * ridge) / 4.5e3) ** 2))
        + 0.3 * np.exp(-(((fgrid - 2 * ridge) / 8e3) ** 2))
    ) * np.exp(0.25 * rng.standard_normal((n_f, n_t)))
    fig = spectro_triptych(
        gt_spec, pred_spec, t_s, f_hz,
        title="ece ch 20 — spectrogram reconstruction",
    )
    files += save_fig(fig, out_dir / "spectro_triptych")
    plt.close(fig)

    # 3 -- psd_overlay -------------------------------------------------------
    f = np.logspace(1, 5, 300)

    def lorentzian(f0: float, width: float) -> np.ndarray:
        return 1.0 / (1.0 + ((f - f0) / width) ** 2)

    psd_gt = 1e-2 * f**-1.4 * (
        1 + 25 * lorentzian(3e3, 400) + 8 * lorentzian(12e3, 1.5e3)
    )
    psd_pred = 1e-2 * f**-1.45 * (
        1 + 18 * lorentzian(3.2e3, 600) + 8 * lorentzian(12e3, 2e3)
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    psd_overlay(ax, f, psd_gt, psd_pred)
    ax.set_title("co2 ch 1 — Welch PSD")
    files += save_fig(fig, out_dir / "psd_overlay")
    plt.close(fig)

    # 4 -- signed_heatmap ----------------------------------------------------
    rows = ["ts_core_density", "ts_core_temp", "cer_ti", "mse",
            "filterscopes", "ece"]
    cols = ["k=1", "k=5", "k=10", "k=20", "k=40"]
    m = rng.normal(0.0, 0.25, size=(len(rows), len(cols)))
    m[:, 0] = 0.5 * np.abs(m[:, 0])
    m[:, -1] -= 0.3
    annot = np.array([[f"{v:+.2f}" for v in row] for row in m])
    fig = signed_heatmap(
        m, rows, cols, annot=annot,
        title="skill vs persistence by rollout depth",
        cbar_label=r"$\Delta$NRMSE (pred $-$ persistence)",
    )
    files += save_fig(fig, out_dir / "signed_heatmap")
    plt.close(fig)

    # 5 -- efficiency_curve --------------------------------------------------
    x = 2.0 ** np.arange(3, 10)  # 8 .. 512 labeled shots
    log_x = np.log2(x)
    spec = {  # name -> (start, asymptote, time constant)
        "scratch": (0.55, 0.88, 3.0),
        "linear probe": (0.68, 0.90, 2.2),
        "fine-tune": (0.74, 0.94, 2.0),
    }
    curves = {}
    for name, (start, asym, tau) in spec.items():
        med = asym - (asym - start) * np.exp(-(log_x - 3.0) / tau)
        band = 0.06 * np.exp(-(log_x - 3.0) / 4.0) + 0.01
        curves[name] = (med, med - band, med + band)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    efficiency_curve(ax, x, curves)
    ax.set_title("disruption probe — label efficiency")
    files += save_fig(fig, out_dir / "efficiency_curve")
    plt.close(fig)

    # 6 -- rollout_divergence ------------------------------------------------
    k = np.arange(1, 81)
    growth = {  # modality -> (start ratio, asymptote, rate)
        "ts_core_density": (0.30, 0.85, 30.0),
        "cer_ti": (0.38, 1.05, 25.0),
        "mse": (0.28, 0.72, 35.0),
        "filterscopes": (0.45, 1.30, 20.0),
        "ece": (0.50, 1.15, 28.0),
    }
    roll = {
        name: a + (c - a) * (1.0 - np.exp(-k / tau))
        + 0.01 * rng.standard_normal(k.size)
        for name, (a, c, tau) in growth.items()
    }
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    rollout_divergence(ax, k, roll, ratio_line=1.0)
    ax.set_ylabel("NRMSE / persistence NRMSE")
    ax.set_title("rollout divergence by modality")
    files += save_fig(fig, out_dir / "rollout_divergence")
    plt.close(fig)

    return files


if __name__ == "__main__":
    import sys
    import tempfile

    if len(sys.argv) > 1:
        demo_dir = Path(sys.argv[1])
    else:
        demo_dir = Path(tempfile.gettempdir()) / "tfm_eval_plotting_demo"
    demo_files = _demo(demo_dir)

    empty = [p for p in demo_files if not (p.exists() and p.stat().st_size)]
    first_png = next(p for p in demo_files if p.suffix == ".png")
    image = plt.imread(first_png)

    print(f"demo dir: {demo_dir}")
    for p in sorted(demo_files):
        print(f"  {p.name:26s} {p.stat().st_size:9,d} B")
    print(f"read-back {first_png.name}: array {image.shape}")
    if empty:
        raise SystemExit(f"ERROR — empty outputs: {empty}")
    print("all outputs verified (non-empty; PNG read back OK)")
