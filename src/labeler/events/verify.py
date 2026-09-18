"""The human review surface: show a shot's signals, take back corrections.

This module knows nothing about plasma physics. A notebook computes whatever
arrays settle its own phenomenon - a spectrogram, a set of raw channels, a
scalar trace - and hands them over as `Panel`s. Deciding which traces settle
which phenomenon is case-by-case work that does not generalise, so it lives
in each category's `verification.ipynb` rather than here.

What is shared is everything else: reading the corpus without loading it,
stacking the panels on one time axis, turning a dragged range into an
interval, and writing the two files a review produces - the corrections under
`review/` and the roster row in `shots.csv`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..config import Paths
from ..features.store import FeatureArray

if TYPE_CHECKING:
    import pandas as pd

#: A corpus group whose `ydata` is this narrow carries the absent-signal
#: sentinel - `(C, 1)` - rather than a record. `resolve_corpus` writes it for
#: a diagnostic that did not run, and it is not data.
SENTINEL_WIDTH = 1


class NoDataError(RuntimeError):
    """A signal a panel asked for is not on disk for this shot."""


def corpus_path(shot: int, *, corpus: Path | None = None) -> Path:
    """The corpus file for one shot."""
    corpus = Paths.from_env().corpus if corpus is None else Path(corpus)
    return corpus / f"{int(shot)}_processed.h5"


def corpus_signal(
    shot: int,
    group: str,
    *,
    channels: Sequence[int] | None = None,
    t_range: tuple[float, float] | None = None,
    corpus: Path | None = None,
) -> FeatureArray:
    """One corpus group, sliced: `x` in milliseconds, `y` as `(C, T)` float32.

    ECE is `(48, 3.1e6)` and CO2 `(4, 4.5e6)`; neither is ever read whole.
    `t_range` is milliseconds and is applied with an h5py slice, so a 100 ms
    window costs a 100 ms read. `channels` selects rows by index.
    """
    import h5py

    path = corpus_path(shot, corpus=corpus)
    if not path.is_file():
        raise NoDataError(
            f"shot {int(shot)} has no corpus file at {path}. The corpus covers "
            f"185601-204999; a shot outside it has to be fetched."
        )
    with h5py.File(path, "r") as f:
        if group not in f:
            raise NoDataError(f"shot {int(shot)} has no {group!r} group in {path}")
        x = f[group]["xdata"]
        y = f[group]["ydata"]
        if y.shape[-1] <= SENTINEL_WIDTH:
            raise NoDataError(
                f"shot {int(shot)} carries the absent-signal sentinel for "
                f"{group!r}: ydata is {y.shape}, so this diagnostic did not run"
            )
        start, stop = 0, x.shape[0]
        if t_range is not None:
            times = np.asarray(x, dtype="float64") * 1000.0
            start = int(np.searchsorted(times, t_range[0], side="left"))
            stop = int(np.searchsorted(times, t_range[1], side="right"))
            if stop <= start:
                raise NoDataError(
                    f"shot {int(shot)} {group!r} has no samples in "
                    f"{t_range[0]}-{t_range[1]} ms"
                )
        rows = list(range(y.shape[0])) if channels is None else list(channels)
        values = np.stack([y[row, start:stop] for row in rows]).astype("float32")
        times_ms = np.asarray(x[start:stop], dtype="float64") * 1000.0
    return FeatureArray(
        x=times_ms,
        y=values,
        attrs={
            "group": group,
            "shot": str(int(shot)),
            "channels": ",".join(str(row) for row in rows),
            "units": "ms",
        },
    )


#: The ECE tree and point spelling, from
#: `src/tokamak_foundation_model/data/config/modalities/modalities.yaml`.
#: `TECEF20`..`TECEF36` etc are the channel points on the `D3D` tree.
ECE_TREE = "D3D"
ECE_POINT = r"\D3D::TOP.ELECTRONS.ECE.TECEF:TECEF{channel:02d}"

#: The four CO2 interferometer chords, R0/V1/V2/V3, in the order every
#: crosspower pair in `alfven_eigenmode/verification.ipynb` depends on.
CO2_CHORDS = ("DENR0UF", "DENV1UF", "DENV2UF", "DENV3UF")

#: The `via` values `fdp_signal` accepts. A typo here must raise, not
#: silently fall back to `"mds"`.
FDP_VIA_ROUTES = ("mds", "ptdata")

#: The wrapper every live fdp fetch has to run under. PTDATA and MDSplus
#: both fail outside it - PTDATA with `getservbyname failed for task
#: 'PTSERVER'`, MDSplus with `TREE-E-FOPENR` - so this is what a reviewer
#: who has not started under it needs to see.
FDP_RESTART_COMMAND = "pixi run -e labelmaker fdp run jupyter lab"


def fdp_signal(
    shot: int,
    exprs: Sequence[str],
    *,
    tree: str = ECE_TREE,
    via: str = "mds",
    t_range: tuple[float, float] | None = None,
    cache: Path | None = None,
) -> FeatureArray:
    """One or more MDSplus or PTDATA points, fetched live or from a cache.

    Deliberately mirrors `corpus_signal`'s contract: `x` in milliseconds,
    `y` as `(C, T)` float32, one row per expression in `exprs` order - so a
    `Panel` cannot tell a fetched array from a corpus one.

    `via="mds"` (the default) fetches each expression with `_fetch_mds(expr,
    tree, shot, dims=["dim0"])`; its time key is `dim0`. `via="ptdata"`
    fetches with `_fetch_ptdata(expr, shot)` instead; its time key is
    `times`, and `tree` is unused on this route. Either way the record's
    time unit is checked rather than trusted, because a silent unit change
    would put every recorded correction off by a factor of a thousand with
    nothing downstream noticing. Any other `via` raises `ValueError` before
    anything is fetched - a typo must not fall through to a default route.

    `cache`, when given, holds the FULL record (never the `t_range` window),
    so widening a window later costs nothing and a shot already on disk is
    never refetched. Nothing is cached when `cache` is None.
    """
    if via not in FDP_VIA_ROUTES:
        legal = " or ".join(repr(route) for route in FDP_VIA_ROUTES)
        raise ValueError(f"fdp_signal: unknown via={via!r}; must be {legal}")
    exprs = list(exprs)
    if not exprs:
        raise NoDataError(f"shot {int(shot)}: no expressions to fetch")
    cache_path = None if cache is None else Path(cache)

    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path) as npz:
            times_ms = np.asarray(npz["x"], dtype="float64")
            values = np.asarray(npz["y"], dtype="float32")
            # The cache is keyed only by its filename, which says nothing
            # about WHICH points it holds. Without this check, asking for
            # four channels against a cache of five returns five rows
            # silently mislabelled as the four - wrong data in front of a
            # reviewer, with nothing to see.
            cached = [str(e) for e in npz.get("exprs", np.array([]))]
        if cached != exprs:
            raise NoDataError(
                f"{cache_path} holds {cached or 'unrecorded points'}, not "
                f"{exprs}. Delete it or pass a different cache path."
            )
    else:
        # `_fetch_mds` and `_fetch_ptdata` are the only places in labeler
        # that know how to call toksearch; reused deliberately here rather
        # than kept as a second copy of either call.
        from ..features.resolve_fdp import _fetch_mds, _fetch_ptdata

        time_key = "dim0" if via == "mds" else "times"

        def fetch_one(expr: str) -> dict:
            if via == "mds":
                return _fetch_mds(expr, tree, int(shot), dims=["dim0"])
            return _fetch_ptdata(expr, int(shot))

        rows = []
        times_ms = None
        expected_length = None
        first_expr = exprs[0] if exprs else None
        for expr in exprs:
            try:
                record = fetch_one(expr)
            except Exception as error:
                raise NoDataError(
                    f"shot {int(shot)} {expr!r} failed to fetch over fdp: "
                    f"{error}. If this kernel was not started under "
                    f"'{FDP_RESTART_COMMAND}', restart it that way first."
                ) from error
            unit = record["units"][time_key]
            if unit != "ms":
                raise NoDataError(
                    f"shot {int(shot)} {expr!r} {time_key} units are "
                    f"{unit!r}, not 'ms'; refusing to guess at a conversion "
                    f"that would put every correction's timebase off."
                )
            data = np.asarray(record["data"], dtype="float32")
            if expected_length is None:
                expected_length = len(data)
                times_ms = np.asarray(record[time_key], dtype="float64")
            elif len(data) != expected_length:
                raise NoDataError(
                    f"shot {int(shot)} expressions disagree on length: "
                    f"{expected_length} for {first_expr!r} vs {len(data)} "
                    f"for {expr!r}"
                )
            rows.append(data)
        values = np.stack(rows).astype("float32")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # A fixed-width unicode array, not dtype=object: an object array
            # would only load back under allow_pickle, and nothing should
            # unpickle a file off disk to read a channel list.
            np.savez(cache_path, x=times_ms, y=values, exprs=np.array(exprs))

    start, stop = 0, times_ms.shape[0]
    if t_range is not None:
        start = int(np.searchsorted(times_ms, t_range[0], side="left"))
        stop = int(np.searchsorted(times_ms, t_range[1], side="right"))
        if stop <= start:
            raise NoDataError(
                f"shot {int(shot)} has no samples in "
                f"{t_range[0]}-{t_range[1]} ms"
            )
    return FeatureArray(
        x=times_ms[start:stop],
        y=values[:, start:stop].astype("float32"),
        attrs={
            "shot": str(int(shot)),
            "channels": ",".join(exprs),
            "units": "ms",
            "source": "fdp",
        },
    )


REVIEW_DIRECTORY = "review"


@dataclass
class Panel:
    """One row of a review figure, already computed by the notebook.

    `kind="line"` draws each row of `y` against `x`, so `y` is `(C, len(x))`.
    `kind="heatmap"` draws `z` with `y` as the vertical axis.

    A heatmap has TWO legitimate shapes, because plotly reads N coordinates
    against N columns as cell CENTRES and N+1 as cell EDGES. A spectrogram
    arrives on centres (`z.shape == (len(y), len(x))`); `label_panel` hands
    over the label grid's own bin edges (`z.shape == (len(y)-1, len(x)-1)`).
    Both render correctly and both are accepted; anything else is a mistake
    plotly would draw without complaining, so it raises here instead.

    `bands` marks horizontal regions of interest, such as the 80-250 kHz AE
    band. `hlines` draws single dashed lines, such as a class threshold -
    a band 0.01 wide is not a line. `zmin`/`zmax` pin a heatmap's colour
    scale so one value means one colour across every shot.
    """

    title: str
    x: np.ndarray
    y: np.ndarray
    kind: str = "line"
    z: np.ndarray | None = None
    ylabel: str = ""
    legend: Sequence[str] | None = None
    bands: Sequence[tuple[float, float]] = ()
    hlines: Sequence[float] = ()
    zmin: float | None = None
    zmax: float | None = None

    def __post_init__(self) -> None:
        # `FeatureArray` two modules over validates its own arrays in
        # `__post_init__`; a panel that disagrees with itself is worse,
        # because plotly renders most of these silently - a transposed `z`
        # as-is, a `z=None` as an empty row, and a `y` shorter than `x` as a
        # trace truncated to the shorter, i.e. drawn at the wrong times on
        # the one surface whose output is corrected timings.
        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)
        if self.kind == "heatmap":
            if self.z is None:
                raise ValueError(
                    f"panel {self.title!r}: kind='heatmap' needs z; without it "
                    f"plotly draws an empty row and reports nothing"
                )
            self.z = np.asarray(self.z)
            rows, columns = (len(self.y), len(self.x))
            if self.z.ndim != 2 or (
                self.z.shape[0] not in (rows, rows - 1)
                or self.z.shape[1] not in (columns, columns - 1)
            ):
                raise ValueError(
                    f"panel {self.title!r}: z is {self.z.shape} against "
                    f"len(y)={rows} and len(x)={columns}; a heatmap needs "
                    f"({rows}, {columns}) for bin centres or "
                    f"({rows - 1}, {columns - 1}) for bin edges"
                )
        elif self.kind == "line":
            if self.y.ndim != 2:
                raise ValueError(
                    f"panel {self.title!r}: kind='line' needs y as "
                    f"(channels, len(x)); got {self.y.ndim}-D {self.y.shape}"
                )
            if self.y.shape[-1] != len(self.x):
                raise ValueError(
                    f"panel {self.title!r}: y is {self.y.shape} against "
                    f"len(x)={len(self.x)}; plotly would truncate to the "
                    f"shorter and put the trace at the wrong times"
                )
        # An unknown `kind` is left to `_build_figure`, which names the panel
        # and the kind - validating it twice would make that branch dead.


def review_path(event: str, shot: int, *, root: Path | None = None) -> Path:
    """Where one shot's corrections live."""
    root = Paths.from_env().label_tables if root is None else Path(root)
    return root / event / REVIEW_DIRECTORY / f"{int(shot)}.csv"


def read_corrections(path) -> pd.DataFrame:
    """Read one shot's corrections through the public interval schema."""
    import pandas as pd

    from .interval_tables import validate_intervals

    return validate_intervals(pd.read_csv(path, keep_default_na=False))


def write_corrections(frame, path) -> None:
    """Validate and write one shot's corrections."""
    from .interval_tables import validate_intervals

    validated = validate_intervals(frame)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(path, index=False)


class ReviewSession:
    """One reviewer, one shot: the figure, the marks, and what gets written.

    Marks accumulate in memory. `save()` is the only thing that touches disk,
    and it writes two files: the corrections under `review/` and the roster
    row in `shots.csv`. `verify()` records that this reviewer looked; without
    it `save()` writes corrections alone, which is what a half-finished
    review should leave behind.
    """

    def __init__(
        self,
        *,
        event: str,
        shot: int,
        panels: Sequence[Panel],
        root: Path | None = None,
        reviewer: str | None = None,
        source: str = "format/shots",
        note: str = "",
    ) -> None:
        import os

        self.event = event
        self.shot = int(shot)
        self.panels = list(panels)
        self.source = source
        # Something the reviewer must not miss, shown in the figure title
        # rather than only as a warning above the widget: `review()` sets it
        # to "NO LABEL ROW" when no saved grid exists for this shot.
        self.note = note
        self.root = Paths.from_env().label_tables if root is None else Path(root)
        self.reviewer = reviewer or os.environ.get("USER", "unknown")
        self._marks: list[tuple[float, float, int]] = []
        self._verify = False
        self._notes: str | None = None
        self.figure = self._build_figure()
        self.controls = self._build_controls()

    @property
    def corrections(self) -> pd.DataFrame:
        import pandas as pd

        from .interval_tables import INTERVAL_COLUMNS

        return pd.DataFrame(
            [
                [self.shot, category, t_start, t_end, ""]
                for t_start, t_end, category in self._marks
            ],
            columns=list(INTERVAL_COLUMNS),
        )

    def mark(self, t_start: float, t_end: float, category: int = 1) -> None:
        """Record one corrected interval, in milliseconds."""
        if t_end < t_start:
            raise ValueError(f"t_end {t_end} precedes t_start {t_start}")
        self._marks.append((float(t_start), float(t_end), int(category)))

    def verify(self, *, notes: str | None = None) -> None:
        """Say this reviewer has looked; `save()` then records them."""
        self._verify = True
        self._notes = notes

    def save(self) -> None:
        """Write the corrections, and the roster row when verified."""
        from .rosters import record_review, roster_path

        if self._marks:
            write_corrections(
                self.corrections, review_path(self.event, self.shot, root=self.root)
            )
        if self._verify:
            record_review(
                roster_path(self.event, root=self.root),
                self.shot,
                self.reviewer,
                notes=self._notes,
            )

    def _build_figure(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        rows = max(len(self.panels), 1)
        figure = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[panel.title for panel in self.panels] or [""],
        )
        for index, panel in enumerate(self.panels, start=1):
            # Each row gets its own legend (plotly's multiple-legend support:
            # a trace's `legend` names a `layout[legend_name]` object), so a
            # multi-row figure shows names beside their own row instead of
            # one merged box - or none at all, which is what a single
            # figure-wide `showlegend=False` used to do.
            legend_name = f"legend{index}"
            if panel.kind == "heatmap":
                figure.add_trace(
                    go.Heatmap(
                        x=panel.x,
                        y=panel.y,
                        z=panel.z,
                        # Pinned by the panel, not autoscaled per shot: with
                        # `minimum_safety_factor`'s five classes an all-class-1
                        # shot would otherwise render in exactly the colours of
                        # an all-class-4 one. None leaves plotly autoscaling,
                        # which is right for a spectrogram.
                        zmin=panel.zmin,
                        zmax=panel.zmax,
                        showscale=False,
                        legend=legend_name,
                        # plotly.js lists heatmap under "showLegend", so with
                        # the figure-wide showlegend now on, a label row would
                        # otherwise emit a nameless "trace N" entry beside it.
                        showlegend=False,
                    ),
                    row=index,
                    col=1,
                )
            elif panel.kind == "line":
                names = panel.legend or [f"ch {i}" for i in range(len(panel.y))]
                for values, name in zip(panel.y, names, strict=True):
                    figure.add_trace(
                        go.Scattergl(
                            x=panel.x,
                            y=values,
                            name=name,
                            mode="lines",
                            legend=legend_name,
                        ),
                        row=index,
                        col=1,
                    )
            else:
                raise ValueError(
                    f"panel {panel.title!r} has unknown kind {panel.kind!r}"
                )
            for low, high in panel.bands:
                if panel.kind == "heatmap":
                    # plotly's `shape.layer` defaults to "above", so a filled
                    # rect washes 8% black over the very image under review -
                    # darker reads as less power on every sequential colormap,
                    # biasing the reviewer toward under-calling the mode. And
                    # `layer="below"` would hide the marker behind the image
                    # entirely. Boundary lines mark the band without touching
                    # what is inside it.
                    for edge in (low, high):
                        figure.add_hline(
                            y=edge, line_width=1, line_dash="dot",
                            line_color="black", row=index, col=1,
                        )
                else:
                    figure.add_hrect(
                        y0=low, y1=high, line_width=0, fillcolor="black",
                        opacity=0.08, layer="below", row=index, col=1,
                    )
            for level in panel.hlines:
                figure.add_hline(
                    y=level, line_width=1, line_dash="dash",
                    line_color="#b2182b", row=index, col=1,
                )
            figure.update_yaxes(title_text=panel.ylabel, row=index, col=1)
            axis_name = "yaxis" if index == 1 else f"yaxis{index}"
            y0, y1 = figure.layout[axis_name].domain
            figure.layout[legend_name] = {"y": (y0 + y1) / 2, "yanchor": "middle"}
        figure.update_xaxes(title_text="Time (ms)", row=rows, col=1)
        title = f"{self.event} - shot {self.shot} ({self.source})"
        if self.note:
            title = f"{title} - {self.note}"
        figure.update_layout(
            title=title,
            height=200 * rows + 120,
            dragmode="select",
            selectdirection="h",
            showlegend=True,
            margin={"l": 60, "r": 20, "t": 60, "b": 40},
        )
        return go.FigureWidget(figure)

    def _build_controls(self):
        import ipywidgets as widgets

        present = widgets.Button(description="Mark present", button_style="primary")
        absent = widgets.Button(description="Mark absent")
        verify = widgets.Button(description="Verify", button_style="success")
        save = widgets.Button(description="Save", button_style="warning")
        status = widgets.HTML(value=self._status())

        def selected() -> tuple[float, float]:
            # Box/lasso select (`dragmode="select"`) writes a persistent,
            # axis-level entry to `layout.selections` regardless of trace
            # type - confirmed against plotly 6.9.0: `go.Heatmap` carries no
            # `selectedpoints` attribute, so the older per-trace
            # `on_selection`/`plotly_selected` callback (built on
            # `selectedpoints`) never fires for a heatmap row, and this
            # figure always has at least one (the label panel). Reading
            # `layout.selections` here, lazily, is what works uniformly
            # across every panel kind. `[-1]` is the most recent drag: the
            # docs describe Shift as how a user *accumulates* more than one
            # persistent selection, so a plain new drag should replace the
            # prior one, but taking the last rather than the first is
            # correct either way.
            selections = self.figure.layout.selections
            if not selections:
                raise ValueError("drag a time range on the figure first")
            box = selections[-1]
            # The default modebar offers Lasso Select alongside Box Select.
            # A lasso writes `{type: "path", path: "M...Z"}` with x0/x1 both
            # None, so reject anything that is not a plain rectangle before
            # touching x0/x1 - checking `x0 is None` too, not just the type
            # string, so a future plotly spelling the type differently is
            # still caught.
            if box.type != "rect" or box.x0 is None or box.x1 is None:
                raise ValueError("use the Box Select tool, not Lasso")
            return float(min(box.x0, box.x1)), float(max(box.x0, box.x1))

        def on_mark(category):
            def handler(_):
                try:
                    t_start, t_end = selected()
                    self.mark(t_start, t_end, category)
                except ValueError as error:
                    status.value = f"<b style='color:#b2182b'>{error}</b>"
                    return
                # Clear the drag so a second click cannot silently record the
                # same interval twice, and so a reviewer who drags again
                # faster than the comm round trip sees an empty selection
                # rather than a stale one being recorded underneath them.
                self.figure.layout.selections = ()
                status.value = self._status()

            return handler

        def on_verify(_):
            self.verify()
            status.value = self._status()

        def on_save(_):
            try:
                self.save()
            except (ValueError, OSError) as error:
                status.value = f"<b style='color:#b2182b'>{error}</b>"
                return
            status.value = self._status() + " <b>saved</b>"

        present.on_click(on_mark(1))
        absent.on_click(on_mark(0))
        verify.on_click(on_verify)
        save.on_click(on_save)
        return widgets.VBox(
            [widgets.HBox([present, absent, verify, save]), status]
        )

    def _status(self) -> str:
        verified = "verified" if self._verify else "not verified"
        return (
            f"{len(self._marks)} correction(s), {verified}, "
            f"reviewer <code>{self.reviewer}</code>"
        )

    def _ipython_display_(self):
        from IPython.display import display

        display(self.figure, self.controls)


def review(
    event: str,
    shot: int,
    panels: Sequence[Panel],
    *,
    source: str = "format/shots",
    root: Path | None = None,
    reviewer: str | None = None,
) -> ReviewSession:
    """Open a review of one shot, with the label row appended to the panels."""
    panels = list(panels)
    note = ""
    try:
        panels.append(label_panel(event, shot, source=source, root=root))
    except OSError as error:
        warnings.warn(
            f"no saved label grid for {event!r} shot {shot}: {error}",
            stacklevel=2,
        )
        # The warning goes to stderr above the widget, which is not where the
        # reviewer is looking. Say it on the figure too.
        note = "NO LABEL ROW"
    return ReviewSession(
        event=event,
        shot=shot,
        panels=panels,
        root=root,
        reviewer=reviewer,
        source=source,
        note=note,
    )


def label_panel(
    event: str, shot: int, *, source: str = "format/shots", root: Path | None = None
) -> Panel:
    """The saved label grid as a heatmap row, unknown cells left as NaN."""
    from .interval_tables import SAMPLE_MS
    from .notebooks import load_shot

    grid = load_shot(event, shot, source=source, root=root)
    # `time_ms` holds each half-open 50 ms bin's LEFT edge, so the edge array
    # is one longer than the grid - exactly what `plot_shot` builds for
    # pcolormesh's `shading="flat"`. Plotly reads N coordinates against N
    # columns as cell CENTRES and N+1 as cell EDGES, so passing the N left
    # edges alone drew every label cell 25 ms early and half a rho bin low,
    # with nothing on screen to give it away - on the surface whose whole job
    # is deciding whether a label boundary sits on the right feature.
    ids = sorted(int(key) for key in grid.get("categories", {}))
    return Panel(
        title=f"labels ({source})",
        kind="heatmap",
        x=np.r_[grid["time_ms"], grid["time_ms"][-1] + SAMPLE_MS],
        y=grid["rho_edges"],
        z=grid["label"].T,
        ylabel="rho",
        # The grid's own declared class ids, derived the way `plot_shot`
        # derives them, so a class keeps one colour from shot to shot. A grid
        # written without a category mapping leaves plotly to autoscale.
        zmin=float(min(ids)) if ids else None,
        zmax=float(max(ids)) if ids else None,
    )
