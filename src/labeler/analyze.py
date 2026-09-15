"""`analyze`: one shot in, the labels a config names out, with a picture.

The question a physicist actually asks is "what does every model say about
shot N?", not "run stage X over shot list Y". `analyze` answers it: a YAML
file names the labels wanted (and a few context signals), the runner makes
sure the features and labels exist, and this module turns the label file into
a JSON summary and one figure with a panel per label.

Nothing here computes a label. The label file under `labels/` stays the one
canonical output; `analysis/` is a view of it that is cheap to regenerate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .features import namespace as ns
from .features.store import present, read_feature
from .labels.store import labelled, read_label
from .models import registry

DEFAULT_CONFIG = Path(__file__).with_name("analyze_default.yaml")

# dataviz reference palette, light mode
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_SERIES = "#2a78d6"
_TRUTH = "#4a3aa7"


class ConfigError(ValueError):
    """The analysis config asks for something labeler cannot produce."""


@dataclass(frozen=True)
class AnalysisConfig:
    labels: tuple[str, ...]
    context: tuple[str, ...]
    threshold: float
    #: per-label overrides of `threshold`; one scale does not fit every model
    thresholds: dict[str, float] = field(default_factory=dict)
    #: labels drawn as markers, not a line: a value that is defined only while
    #: something is happening (a frequency while a mode is active) has no
    #: meaningful path between its islands
    scatter: tuple[str, ...] = ()

    def threshold_for(self, label: str) -> float:
        return float(self.thresholds.get(label, self.threshold))

    @property
    def slugs(self) -> tuple[str, ...]:
        out: list[str] = []
        for label in self.labels:
            slug = label.split("/", 1)[0]
            if slug not in out:
                out.append(slug)
        return tuple(out)

    def as_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "context": list(self.context),
            "threshold": self.threshold,
            "thresholds": dict(self.thresholds),
            "scatter": list(self.scatter),
        }


def load_config(path) -> AnalysisConfig:
    """Parse and check a config; every problem is a `ConfigError` before any work."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    labels = raw.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ConfigError(f"{path}: `labels` must be a non-empty list of slug/name")
    fields_of: dict[str, set[str]] = {}
    for label in labels:
        if not isinstance(label, str) or label.count("/") != 1:
            raise ConfigError(f"{label!r}: a label is written slug/name")
        slug, name = label.split("/")
        if slug not in fields_of:
            try:
                adapter = registry.load_adapter(slug)
            except Exception as exc:  # any load failure is a config error
                raise ConfigError(
                    f"{label}: no model {slug!r} ({type(exc).__name__}: {exc})"
                ) from exc
            fields_of[slug] = {f.name for f in adapter.output_spec.fields}
        if name not in fields_of[slug]:
            raise ConfigError(
                f"{label}: {slug} produces {sorted(fields_of[slug])}, not {name!r}"
            )
    context = raw.get("context") or []
    if not isinstance(context, list):
        raise ConfigError(f"{path}: `context` must be a list of canonical features")
    for name in context:
        try:
            ns.by_name(name)
        except KeyError:
            raise ConfigError(f"{name!r} is not a canonical feature") from None
    threshold = raw.get("threshold", 0.5)
    if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        raise ConfigError(f"threshold must be a number in [0, 1], got {threshold!r}")
    overrides = raw.get("thresholds") or {}
    if not isinstance(overrides, dict):
        raise ConfigError(f"{path}: `thresholds` must map a label to a number")
    for label, value in overrides.items():
        if label not in labels:
            raise ConfigError(f"thresholds: {label!r} is not one of the `labels`")
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ConfigError(f"thresholds: {label} must be a number in [0, 1], got {value!r}")
    scatter = raw.get("scatter") or []
    if not isinstance(scatter, list) or not all(isinstance(x, str) for x in scatter):
        raise ConfigError(f"{path}: `scatter` must be a list of labels")
    for label in scatter:
        if label not in labels:
            raise ConfigError(f"scatter: {label!r} is not one of the `labels`")
    return AnalysisConfig(
        labels=tuple(labels), context=tuple(context), threshold=float(threshold),
        thresholds={k: float(v) for k, v in overrides.items()}, scatter=tuple(scatter),
    )


def _scalar(value):
    """h5py attribute -> plain Python, for JSON."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_label(labels_path, slug: str, name: str, *, threshold: float,
                    infer_row: dict, truth: dict | None = None) -> dict:
    """What one label series says about the shot, as numbers.

    `infer_row` carries the provenance `InputSpec.build` produced for this
    model on this shot: `invalid_reasons`, `resolvers`, `missing_inputs`.
    """
    lab = read_label(labels_path, slug, name)
    valid = read_label(labels_path, slug, f"{name}_valid").y[0].astype(bool)
    t, y = lab.x, lab.y[0]
    task = str(_scalar(lab.attrs["task"]))
    finite = np.isfinite(y)
    peak = int(np.argmax(np.where(finite, y, -np.inf))) if finite.any() else None
    binary = task == "binary"
    above = np.flatnonzero(finite & (y >= threshold)) if binary else np.array([], int)
    out = {
        "card_id": str(_scalar(lab.attrs["card_id"])),
        "artifact_sha256": str(_scalar(lab.attrs["artifact_sha256"])),
        "task": task,
        "units": str(_scalar(lab.attrs.get("units", ""))),
        "time_step_ms": float(_scalar(lab.attrs["time_step_ms"])),
        "n_rows": int(t.size),
        "n_valid": int(valid.sum()),
        "valid_fraction": float(valid.mean()) if t.size else 0.0,
        "max": float(y[peak]) if peak is not None else None,
        "t_at_max": float(t[peak]) if peak is not None else None,
        "threshold": threshold if binary else None,
        "first_above_threshold": float(t[above[0]]) if above.size else None,
        "invalid_reasons": dict(infer_row.get("invalid_reasons") or {}),
        "resolvers": dict(infer_row.get("resolvers") or {}),
        "missing_inputs": list(infer_row.get("missing_inputs") or []),
    }
    if name.endswith("_p50"):
        out["p50_at_onset_minus_1s"] = None
        onset = (truth or {}).get("onset_s")
        if onset is not None and np.isfinite(onset) and t.size:
            i = int(np.argmin(np.abs(t - (onset - 1.0))))
            if valid[i] and finite[i]:
                out["p50_at_onset_minus_1s"] = float(y[i])
    return out


def empty_summary(infer_row: dict, note: str) -> dict:
    """The summary of a label that was not produced, saying why."""
    return {
        "n_rows": 0, "n_valid": 0, "valid_fraction": 0.0, "max": None,
        "t_at_max": None, "threshold": None, "first_above_threshold": None,
        "note": note,
        "invalid_reasons": dict(infer_row.get("invalid_reasons") or {}),
        "resolvers": dict(infer_row.get("resolvers") or {}),
        "missing_inputs": list(infer_row.get("missing_inputs") or []),
    }


def panels_for(features_path, labels_path, cfg: AnalysisConfig,
               truth: dict | None = None) -> list[dict]:
    """The series the figure draws: context features first, then each label.

    `truth` is `validate.archived_truth` for this shot when it has any: its
    tearing label is shaded on binary panels, its `betan` column drawn on the
    matching regression panel, and the archived onset marked on every label
    panel, so the picture answers "how did it do" and not only "what did it
    say".
    """
    panels: list[dict] = []
    stored = present(features_path) if Path(features_path).exists() else set()
    for name in cfg.context:
        spec = ns.by_name(name)
        panel = {"kind": "context", "name": name, "units": spec.units, "t": None,
                 "y": None, "note": "not resolved"}
        if name in stored:
            fa = read_feature(features_path, name)
            many = fa.y.shape[0] > 1
            panel.update(
                t=fa.x, y=np.nanmean(fa.y, axis=0) if many else fa.y[0],
                note="channel mean" if many else "",
            )
        panels.append(panel)
    have = labelled(labels_path)
    for label in cfg.labels:
        slug, name = label.split("/", 1)
        panel = {"kind": "label", "name": label, "t": None, "note": "no labels",
                 "threshold": cfg.threshold_for(label), "truth_t": None,
                 "scatter": label in cfg.scatter,
                 "truth_mask": None, "truth_y": None,
                 "onset_s": (truth or {}).get("onset_s")}
        if truth is not None and truth.get("available"):
            panel["truth_t"] = truth["t"]
            # The archive's own tearing label, and its `betan` column for the
            # one label that predicts it.
            panel["truth_mask"] = truth["tm_label"]
            if name == "betan":
                panel["truth_y"] = truth["betan"]
        if label in have:
            lab = read_label(labels_path, slug, name)
            spread = read_label(labels_path, slug, f"{name}_spread")
            valid = read_label(labels_path, slug, f"{name}_valid")
            panel.update(
                task=str(_scalar(lab.attrs["task"])),
                units=str(_scalar(lab.attrs.get("units", ""))),
                t=lab.x, y=lab.y[0], lo=spread.y[0], hi=spread.y[1],
                valid=valid.y[0].astype(bool), note="",
            )
            if name.endswith("_p50") and all(
                f"{slug}/{name[:-3]}{q}" in have for q in ("p10", "p90")
            ):
                panel.update(
                    kind="band", units="ms",
                    lo=read_label(labels_path, slug, f"{name[:-3]}p10").y[0],
                    hi=read_label(labels_path, slug, f"{name[:-3]}p90").y[0],
                )
                onset = panel["onset_s"]
                if (truth or {}).get("available") and onset is not None:
                    before = lab.x < onset
                    panel["truth_t"] = lab.x[before]
                    panel["truth_y"] = (onset - lab.x[before]) * 1000.0
        panels.append(panel)
    return panels


def _spans(mask, x):
    """Contiguous `[start, end]` runs of a boolean mask over `x`.

    The archived rows are a subset of the label's grid and can have gaps, so
    a run ends at the last row that is true, not at the next grid step.
    """
    mask = np.asarray(mask, dtype=bool)
    x = np.asarray(x, dtype=float)
    out, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = x[i]
        if start is not None and (not on or i == mask.size - 1):
            out.append((start, x[i] if on else x[i - 1]))
            start = None
    return out


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_SECONDARY, labelsize=8, length=3)
    ax.yaxis.label.set_color(_INK)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_shot(shot: int, panels: list[dict], out_png, *, title_ids) -> Path:
    """One column of panels on a shared time axis; a label panel shows the
    series, the ensemble spread, the invalid rows, and its threshold."""
    # Figure objects directly, not pyplot: no global state to leak between
    # shots in one worker, and no display backend to negotiate.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n = max(len(panels), 1)
    fig = Figure(figsize=(10, 2.1 * n + 0.9), dpi=130, facecolor=_SURFACE)
    FigureCanvasAgg(fig)
    axes = fig.subplots(n, 1, sharex=True, squeeze=False)[:, 0]
    # The labels are the subject: the time axis spans their grid, so a
    # context signal recorded from -4 s to 20 s does not squeeze a 6 s
    # plasma into a third of the width.
    spans = [(p["t"][0], p["t"][-1]) for p in panels
             if p["kind"] in ("label", "band") and p["t"] is not None and len(p["t"])]
    if spans:
        lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
        pad = 0.02 * (hi - lo) if hi > lo else 0.1
        axes[-1].set_xlim(lo - pad, hi + pad)
    for ax, panel in zip(axes, panels, strict=False):
        _style(ax)
        short = panel["name"].rsplit("/", 1)[-1]
        label = short + (f" ({panel['units']})" if panel.get("units") else "")
        if panel["t"] is None:
            ax.set_ylabel(label, fontsize=8)
            ax.text(0.5, 0.5, panel["note"], transform=ax.transAxes, ha="center",
                    va="center", color=_MUTED, fontsize=9)
            continue
        if panel["kind"] == "context":
            ax.plot(panel["t"], panel["y"], color=_SECONDARY, linewidth=1.0)
            ax.set_ylabel(label, fontsize=8)
            if panel["note"]:
                ax.text(0.995, 0.92, panel["note"], transform=ax.transAxes,
                        ha="right", va="top", color=_MUTED, fontsize=7)
            continue
        t, y, valid = panel["t"], panel["y"], panel["valid"]
        band = panel["kind"] == "band"
        binary = panel["task"] == "binary"
        if band:
            ax.set_yscale("log")
        if binary and panel.get("truth_mask") is not None:
            # Labelled only once: one legend entry, not one per run.
            for k, (lo, hi) in enumerate(_spans(panel["truth_mask"], panel["truth_t"])):
                ax.axvspan(lo, hi, color=_TRUTH, alpha=0.16, linewidth=0, zorder=0,
                           label="archived label: mode present" if k == 0 else None)
        if (~valid).any():
            ax.fill_between(
                t, 0, 1, where=~valid, transform=ax.get_xaxis_transform(),
                color=_GRID, alpha=0.75, linewidth=0, label="invalid rows",
            )
        if panel.get("scatter"):
            # Markers with the spread as a whisker: the value exists only on
            # islands, and a line would invent a path between them.
            ax.errorbar(t, y, yerr=[np.clip(y - panel["lo"], 0, None), np.clip(panel["hi"] - y, 0, None)],
                        fmt="o", color=_SERIES, markersize=3, linewidth=0, elinewidth=0.8,
                        ecolor=_SERIES, alpha=0.9, label=panel["name"])
        else:
            ax.fill_between(t, panel["lo"], panel["hi"], color=_SERIES, alpha=0.18,
                            linewidth=0, label="p10..p90" if band else "ensemble spread")
            ax.plot(t, y, color=_SERIES, linewidth=1.5, label=panel["name"])
        if panel.get("truth_y") is not None:
            ax.plot(panel["truth_t"], panel["truth_y"], color=_TRUTH if band else _INK, linewidth=1.1,
                    linestyle=(0, (5, 2)), label="archived truth")
        if panel.get("onset_s") is not None:
            ax.axvline(panel["onset_s"], color=_TRUTH, linewidth=1.4,
                       label=f"archived onset {panel['onset_s']:.2f} s")
        if binary:
            threshold = panel["threshold"]
            ax.axhline(threshold, color=_MUTED, linestyle="--", linewidth=0.9,
                       label=f"threshold {threshold:g}")
            ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(label, fontsize=8)
        # Above the panel, in the gap `hspace` leaves, so it never sits on
        # the curve it names.
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False,
                  fontsize=7, ncol=4, labelcolor=_SECONDARY, borderaxespad=0.2)
    axes[-1].set_xlabel("time (s)", fontsize=8, color=_INK)
    # One model id per line once there are more than two: the ids are long and
    # a single line runs off the right edge with five models.
    joiner = ", " if len(title_ids) <= 2 else "\n"
    fig.suptitle(f"shot {shot}  |  " + joiner.join(title_ids), fontsize=10,
                 color=_INK, x=0.01, ha="left", va="top", y=0.995)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.07, hspace=0.42)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=_SURFACE)
    return out_png
