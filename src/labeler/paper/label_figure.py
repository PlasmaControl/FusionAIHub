"""The paper's teaser figure, `fig_interpreter`: one shot with the catalog's labels.

    PYTHONPATH=src pixi run --frozen -e labelmaker \\
        python -m labeler.paper.label_figure --out DIR [--shot SHOT] [--no-gate]

It replaces the suggestions figure `paper.roster` drew, whose tracks came from
models the paper no longer keeps (AE-XP, SegNet, the frame models' tables).
Here every track is a label: nothing on it is a model's output, and no model is
run to make one.

**The rows** are the review page's, read from the shot's review stores, never
fetched: the CO2 R0 power spectrogram to 125 kHz (the ELM event's row
`CO2_TITLE`, built from the shot's raw CO2 record), then `roster.PANELS`: the
toroidal n map, D-alpha, NBI power and the four ECE Te groups. The n map is
gated by TokEye's coherent mask as `paper.roster` gates it (the owner's F4),
unless `gate=False` (`--no-gate`). A row is found by every comma part of its
role's title (`matches`), so the role "ECE Te, ch 20-23" finds the row
"ECE Te, inversion side A, ch 20-23 (...)". A store or row that is not there
gives a panel that says so (`shots.NO_DATA`).

**The tracks** (`TRACKS`) hold one phenomenon each, from the first of its
sources (`Source`) that has the shot:

- the expert's review, the review page's saved label
  (`<event>/review/labels.csv`), the paper's silver tier (`SILVER`);
- else an imported table, the paper's legacy tier (`LEGACY`): the tearing-mode
  archive's format table, the merged confinement table (Gill's and Butt's
  tables, `labeler.confinement`'s `merged_intervals.csv`, regimes by letter:
  `REGIME_CODES`) and the ELM onset table's format table.

AE and sawteeth have no imported source: the AE format table is a CO2
detector's output, and the sawtooth targets a crash detector's. A shot that no
source holds says `NO_LABEL`. A row is drawn as its file has it: category 0 is
absent; a span is present, uncertain or not observable, or for confinement a
regime (high, low, QH, WPQH) or uncertain; an ELM crowd span (an ELMing period
whose single ELMs are not separated, `iscrowd` 1) is hatched. Time with no row
was not assessed and is left blank: it is never drawn as absent. Each track
names its tier at its right.

**The shot** is `PICK_RULE`'s among the non-blind cohort shots, whose review
stores hold the rows (`candidates`), or the one `--shot` names. Its record
(`record`, written beside the figure as `MANIFEST`) holds the shot, the rule,
each track's source and tier with the file's path and sha256, the stores read
and the n map's gate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import confinement
from ..ae.xpower.gallery import STATE_COLOURS
from ..config import Paths, atomic_path, git_sha, sha256_of
from ..events.catalog.cohort import read_cohort
from ..events.catalog.points import validate_csv_fields
from ..events.catalog.states import NOT_OBSERVABLE, PRESENT, UNCERTAIN
from ..events.interval_tables import parse_attrs, validate_intervals
from ..events.review import labels
from ..events.review import rows as store_rows
from ..events.spans import cohort_path
from . import FONT_PT, LOGIN_THREADS, PAGE_IN, roster, save, style
from .roster import ROW_COLOURS, TRACE_LW, Read, Signal, read_row
from .shots import MARGIN_MS, NO_DATA, TEXT_COLOUR, show_image, text_track

STEM = "fig_interpreter"
MANIFEST = "label_figure.json"
SILVER = "silver: expert review"
LEGACY = "legacy: imported table"
NO_LABEL = "no label on this shot"
TITLE = "shot {shot} ({year}): the catalog's labels, no model output"
PICK_RULE = (
    "the non-blind cohort shot on which the most phenomena carry a label "
    "(expert-reviewed or imported), then the most expert-reviewed, then the most "
    "labelled time, then the lower shot number"
)
NAMED = "named by --shot"
#: The top spectrogram: the ELM event's interferometer row (0-125 kHz).
CO2_EVENT = "edge_localized_mode"
CO2_TITLE = "CO2 R0 power"
CO2_TOP_KHZ = 125.0
CO2_YLABEL = "CO2 R0 power\nkHz"
#: The merged confinement table's regime letters, as the review page's
#: confinement event numbers them (1 high, 2 low, 3 qh, 4 wpqh, 5 uncertain).
REGIME_CODES = {"H": 1, "L": 2, "QH": 3, "WP": 4, "H|L": 5}
ABSENT_NAME = "absent"
BINARY = {PRESENT: "present", UNCERTAIN: "uncertain", NOT_OBSERVABLE: "not observable"}
REGIMES = {1: "H-mode", 2: "L-mode", 3: "QH-mode", 4: "WPQH-mode", 5: "uncertain"}
COLOURS = {
    ABSENT_NAME: "#e2e2e2",
    "present": STATE_COLOURS[PRESENT],
    "uncertain": STATE_COLOURS[UNCERTAIN],
    "not observable": STATE_COLOURS[NOT_OBSERVABLE],
    "H-mode": "#2c7fb8",
    "L-mode": "#a6d96a",
    "QH-mode": "#41b6c4",
    "WPQH-mode": "#253494",
}
CROWD = "crowd (ELMing period)"
CROWD_HATCH = "//////"
UNLABELLED = "blank: not labelled"
BAR = (0.1, 0.8)
TRACK_HEIGHT = 0.45
#: The layout's side pad, in: a little over the default 3 pt, so the tiers at
#: the tracks' right keep clear of the page's edge.
W_PAD_IN = 8 / 72


@dataclass(frozen=True)
class Row:
    """One label row, ms: its category and, where the file says, `iscrowd`."""

    t_start: float
    t_end: float
    category: int
    crowd: int | None = None


@dataclass(frozen=True)
class Source:
    """A label file of one tier: `locate(paths)` is it; `regimes` reads the
    merged confinement table (regime letters) rather than a format table."""

    tier: str
    what: str
    locate: Callable[[Paths], Path]
    regimes: bool = False


@dataclass(frozen=True)
class TrackSpec:
    """A phenomenon's track: its `title`, the names of its categories, and its
    sources, the expert's review first."""

    key: str
    title: str
    states: Mapping[int, str]
    sources: tuple[Source, ...]


def _review(event: str) -> Callable[[Paths], Path]:
    return lambda paths: labels.labels_path(paths.label_tables / event)


def _format(relative: str) -> Callable[[Paths], Path]:
    return lambda paths: paths.label_tables / relative


def merged_confinement(paths: Paths) -> Path:
    """Gill's and Butt's merged regime intervals (`labeler.confinement`)."""
    return confinement.run_directory() / "merged_intervals.csv"


def _expert(event: str) -> Source:
    return Source(SILVER, "the expert's review", _review(event))


TRACKS = (
    TrackSpec("alfven_eigenmode", "AE", BINARY, (_expert("alfven_eigenmode"),)),
    TrackSpec(
        "neoclassical_tearing_mode",
        "NTM",
        BINARY,
        (
            _expert("neoclassical_tearing_mode"),
            Source(
                LEGACY,
                "the tearing-mode archive",
                _format(
                    "neoclassical_tearing_mode/format/"
                    "neoclassical_tearing_mode_format_2026_v1.csv"
                ),
            ),
        ),
    ),
    TrackSpec(
        "confinement",
        "confinement",
        REGIMES,
        (
            _expert("confinement"),
            Source(
                LEGACY,
                "Gill's and Butt's merged regime tables",
                merged_confinement,
                regimes=True,
            ),
        ),
    ),
    TrackSpec(
        "edge_localized_mode",
        "ELMs",
        BINARY,
        (
            _expert("edge_localized_mode"),
            Source(
                LEGACY,
                "the ELM onset table",
                _format(
                    "edge_localized_mode/format/edge_localized_mode_format_2026_v1.csv"
                ),
            ),
        ),
    ),
    TrackSpec(
        "sawtooth_oscillation", "sawteeth", BINARY, (_expert("sawtooth_oscillation"),)
    ),
)


@dataclass(frozen=True)
class Track:
    """A drawn track: its spec, the source it came from (None: no source has
    the shot) and that source's rows for the shot."""

    spec: TrackSpec
    source: Source | None = None
    file: Path | None = None
    rows: tuple[Row, ...] = ()

    @property
    def labelled_ms(self) -> float:
        """The time some non-absent row covers, overlaps counted once."""
        return union_ms((r.t_start, r.t_end) for r in self.rows if r.category)


def union_ms(spans) -> float:
    """The total length of the union of `(start, end)` spans."""
    total, end = 0.0, -math.inf
    for a, b in sorted(spans):
        if b > end:
            total += b - max(a, end)
            end = b
    return total


def read_rows(path: Path, regimes: bool = False) -> dict[int, tuple[Row, ...]]:
    """A label file's rows by shot, in file order; none if it does not exist.
    A format table (or a review's `labels.csv`) is validated as one; `regimes`
    reads the merged confinement table's letters (`REGIME_CODES`). A row with
    no length (a point) is dropped: it has no span to draw."""
    path = Path(path)
    if not path.is_file():
        return {}
    validate_csv_fields(path)
    frame = pd.read_csv(path, index_col=False)
    if regimes:
        unknown = set(frame["regimes"]) - set(REGIME_CODES)
        if unknown:
            raise ValueError(f"{path}: unknown regimes {sorted(unknown)}")
        frame = frame.assign(category=frame["regimes"].map(REGIME_CODES))
        crowd = [None] * len(frame)
    else:
        frame = validate_intervals(frame)
        cells = frame["attrs"] if "attrs" in frame else [None] * len(frame)
        crowd = [parse_attrs(cell).get("iscrowd") for cell in cells]
    found: dict[int, list[Row]] = {}
    for shot, a, b, c, flag in zip(
        frame["shot"],
        frame["t_start"],
        frame["t_end"],
        frame["category"],
        crowd,
        strict=True,
    ):
        if b > a:
            found.setdefault(int(shot), []).append(
                Row(float(a), float(b), int(c), flag)
            )
    return {shot: tuple(rows) for shot, rows in found.items()}


def read_sources(
    paths: Paths, specs: tuple[TrackSpec, ...] = TRACKS
) -> dict[tuple[str, int], tuple[Path, dict[int, tuple[Row, ...]]]]:
    """Every source's file and rows, keyed by (track key, source index)."""
    out = {}
    for spec in specs:
        for i, source in enumerate(spec.sources):
            file = source.locate(paths)
            out[spec.key, i] = (file, read_rows(file, source.regimes))
    return out


def tracks_of(
    shot: int,
    read: Mapping[tuple[str, int], tuple[Path, dict]],
    specs: tuple[TrackSpec, ...] = TRACKS,
) -> tuple[Track, ...]:
    """Each spec's track on `shot`: its first source that has the shot."""
    out = []
    for spec in specs:
        track = Track(spec)
        for i, source in enumerate(spec.sources):
            file, rows = read[spec.key, i]
            if shot in rows:
                track = Track(spec, source, file, rows[shot])
                break
        out.append(track)
    return tuple(out)


@dataclass(frozen=True)
class Candidate:
    shot: int
    year: int
    window: tuple[int, int]


def candidates(paths: Paths) -> list[Candidate]:
    """The frozen cohort's non-blind shots, by shot: the review queue, whose
    stores hold the review page's rows."""
    cohort = read_cohort(cohort_path(paths))
    keep = cohort[~cohort["blind"]]
    columns = ("shot", "year", "window_start_ms", "window_end_ms")
    found = [
        Candidate(int(s), int(y), (int(lo), int(hi)))
        for s, y, lo, hi in zip(*(keep[c] for c in columns), strict=True)
    ]
    return sorted(found, key=lambda c: c.shot)


def score(tracks: tuple[Track, ...]) -> tuple[int, int, float]:
    """`PICK_RULE`'s measures: phenomena labelled, of them expert-reviewed, and
    the labelled time, ms."""
    held = [t for t in tracks if t.source is not None]
    expert = sum(t.source.tier == SILVER for t in held)
    return len(held), expert, sum(t.labelled_ms for t in held)


def pick(
    found: list[Candidate],
    read: Mapping[tuple[str, int], tuple[Path, dict]],
    specs: tuple[TrackSpec, ...] = TRACKS,
) -> Candidate:
    """`PICK_RULE`'s shot among `found`."""
    if not found:
        raise ValueError("no candidate shot to pick from")

    def key(c: Candidate):
        n, expert, ms = score(tracks_of(c.shot, read, specs))
        return (-n, -expert, -ms, c.shot)

    return min(found, key=key)


def matches(title: str, role_title: str) -> bool:
    """Whether a store row's `title` is the role's: it starts with the role's
    first comma part and holds every other part."""
    head, *rest = [p.strip() for p in role_title.split(",")]
    return title.startswith(head) and all(p in title for p in rest)


def find_row(described: list[dict], role_title: str) -> dict | None:
    return next((r for r in described if matches(r["title"], role_title)), None)


@dataclass(frozen=True)
class LabelShot:
    """The drawn shot: its spectrogram (`spec`, a `Read` or None), its signal
    panels and its tracks over `window` (ms)."""

    shot: int
    year: int
    window: tuple[int, int]
    spec: Read | None
    signals: tuple[Signal, ...]
    tracks: tuple[Track, ...]
    stores: dict
    n_gate: dict | None = None
    no_gate: str | None = None

    @property
    def span(self) -> tuple[float, float]:
        return self.window[0] - MARGIN_MS, self.window[1] + MARGIN_MS


def signals(
    paths: Paths, shot: int, t0: float, t1: float
) -> tuple[Read | None, tuple[Signal, ...], dict]:
    """The CO2 spectrogram and `roster.PANELS`' signals of `shot` over
    `t0`-`t1` ms from its review stores, and each store event's file (None
    where there is none). Never fetches."""
    events = [CO2_EVENT, *(p.event for p in roster.PANELS)]
    stores = {}
    for event in events:
        path = paths.spectrogram_file(event, shot)
        stores.setdefault(event, path if path.is_file() else None)
    described = {
        event: [] if path is None else store_rows.meta(path)["rows"]
        for event, path in stores.items()
    }
    row = find_row(described[CO2_EVENT], CO2_TITLE)
    spec = None if row is None else read_row(stores[CO2_EVENT], row, t0, t1)
    out = []
    for panel in roster.PANELS:
        found = []
        for role in panel.roles:
            row = find_row(described[panel.event], role.title)
            read = None if row is None else read_row(stores[panel.event], row, t0, t1)
            if read is not None and all(
                r.meta["name"] != read.meta["name"] for r in found
            ):
                found.append(read)
        text = None if found else NO_DATA.format(panel.title)
        out.append(Signal(panel, roster._label(panel, found), tuple(found), text))
    return spec, tuple(out), stores


def label_shot(
    paths: Paths,
    candidate: Candidate,
    read: Mapping[tuple[str, int], tuple[Path, dict]],
    *,
    gate: bool = True,
) -> LabelShot:
    """`candidate`'s rows and tracks, the n map gated by TokEye if `gate`."""
    t0, t1 = candidate.window[0] - MARGIN_MS, candidate.window[1] + MARGIN_MS
    spec, drawn, stores = signals(paths, candidate.shot, t0, t1)
    n_gate = no_gate = None
    if gate:
        drawn, n_gate, why = roster.gate_signals(paths, candidate.shot, drawn)
        no_gate = None if why is None else why.why
    return LabelShot(
        shot=candidate.shot,
        year=candidate.year,
        window=candidate.window,
        spec=spec,
        signals=drawn,
        tracks=tracks_of(candidate.shot, read),
        stores=stores,
        n_gate=n_gate,
        no_gate=no_gate,
    )


_CHANNELS = re.compile(r"ch \d+-\d+")


def _trace_name(title: str) -> str:
    """A trace row's key: its channel group ("ch 20-23") where it names one,
    else its title after the first comma."""
    found = _CHANNELS.search(title)
    if found:
        return found.group(0)
    return title.split(",", 1)[-1].split("(")[0].strip()


def _traces(ax, sig: Signal) -> None:
    """Each channel's minimum-to-maximum band, as `paper.roster` draws it; a
    colour per row, keyed by its channel group, when there are several."""
    handles = []
    several = len(sig.rows) > 1
    for i, read in enumerate(sig.rows):
        low, high = read.values
        for c in range(len(low)):
            colour = ROW_COLOURS[(i if several else c) % len(ROW_COLOURS)]
            ax.fill_between(read.centres, low[c], high[c], color=colour, lw=TRACE_LW)
        if several:
            colour = ROW_COLOURS[i % len(ROW_COLOURS)]
            name = _trace_name(read.meta["title"])
            handles.append(Line2D([], [], color=colour, lw=1.0, label=name))
    roster._key(ax, handles)


def _signal(ax, sig: Signal) -> None:
    ax.set_ylabel(sig.label)
    if sig.text is not None:
        ax.set_yticks([])
        text_track(ax, sig.text)
    elif sig.panel.kind == "trace":
        _traces(ax, sig)
    else:
        (read,) = sig.rows
        roster._modes(ax, read)
        ax.set_ylim(0, read.extent[3])


def _bars(ax, spans, colour, *, hatch=None) -> None:
    if spans:
        ax.broken_barh(
            spans,
            BAR,
            facecolor=colour,
            edgecolor="white" if hatch else colour,
            hatch=hatch,
            lw=0,
        )


def draw_track(ax, track: Track) -> set[str]:
    """`track`'s rows: absent rows first, then crowd spans hatched, then the
    other spans; the names of the states drawn (and `CROWD` for a crowd)."""
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(track.spec.title, rotation=0, ha="right", va="center")
    # The tier as a right-hand axis label, so the layout keeps room for it.
    right = ax.secondary_yaxis("right")
    right.set_yticks([])
    right.spines["right"].set_visible(False)
    right.set_ylabel(
        NO_LABEL if track.source is None else track.source.tier,
        rotation=0,
        ha="left",
        va="center",
        color=TEXT_COLOUR,
        fontsize=FONT_PT - 1,
    )
    if track.source is None:
        return set()
    drawn = set()
    absent = [(r.t_start, r.t_end - r.t_start) for r in track.rows if not r.category]
    if absent:
        _bars(ax, absent, COLOURS[ABSENT_NAME])
        drawn.add(ABSENT_NAME)
    for crowd in (True, False):
        for category, name in track.spec.states.items():
            spans = [
                (r.t_start, r.t_end - r.t_start)
                for r in track.rows
                if r.category == category and (r.crowd == 1) == crowd
            ]
            if spans:
                _bars(ax, spans, COLOURS[name], hatch=CROWD_HATCH if crowd else None)
                drawn.add(name)
                if crowd:
                    drawn.add(CROWD)
    return drawn


LEGEND_ORDER = (
    "present",
    ABSENT_NAME,
    "uncertain",
    "not observable",
    "H-mode",
    "L-mode",
    "QH-mode",
    "WPQH-mode",
    CROWD,
    UNLABELLED,
)


def _handle(name: str) -> Patch:
    if name == CROWD:
        return Patch(facecolor="#9a9a9a", edgecolor="white", hatch=CROWD_HATCH, lw=0)
    if name == UNLABELLED:
        return Patch(facecolor="white", edgecolor="#9a9a9a", lw=0.5)
    return Patch(facecolor=COLOURS[name], lw=0)


def legend(fig: Figure, drawn: set[str]) -> list[str]:
    """One key per state some track draws, in `LEGEND_ORDER`, and the blank
    of unlabelled time; the names keyed."""
    names = [n for n in LEGEND_ORDER if n in drawn or n == UNLABELLED]
    fig.legend(
        [_handle(n) for n in names],
        names,
        loc="outside lower center",
        ncols=min(len(names), 6),
    )
    return names


def draw(s: LabelShot, stem: Path | None = None) -> Figure:
    """The CO2 spectrogram; the signal panels (the n map, D-alpha, NBI, ECE
    Te); then one track per phenomenon, each the catalog's label with its tier.
    Saved as `stem`.pdf and .png when `stem` is given."""
    t0, t1 = s.span
    heights = [
        4,
        *(g.panel.height for g in s.signals),
        *[TRACK_HEIGHT] * len(s.tracks),
    ]
    n_signals = len(s.signals)
    with style():
        fig = Figure(
            figsize=(PAGE_IN, 3.4 + 0.45 * math.fsum(heights[1 : 1 + n_signals])),
            layout="constrained",
        )
        fig.get_layout_engine().set(w_pad=W_PAD_IN)
        spec, *axes = fig.subplots(
            len(heights), 1, sharex=True, gridspec_kw={"height_ratios": heights}
        )
        panels, tracks = axes[:n_signals], axes[n_signals:]
        if s.spec is None:
            spec.set_ylabel(CO2_YLABEL)
            spec.set_yticks([])
            text_track(spec, NO_DATA.format("CO2"))
        else:
            show_image(spec, s.spec.values, s.spec.extent, CO2_TOP_KHZ, CO2_YLABEL)
        spec.set_xlim(t0, t1)
        spec.set_title(TITLE.format(shot=s.shot, year=s.year))
        for ax, sig in zip(panels, s.signals, strict=True):
            _signal(ax, sig)
        drawn: set[str] = set()
        for ax, track in zip(tracks, s.tracks, strict=True):
            drawn |= draw_track(ax, track)
        for ax in (spec, *panels, *tracks[:-1]):
            ax.tick_params(labelbottom=False)
        for ax in tracks[:-1]:
            ax.tick_params(bottom=False)
        tracks[-1].set_xlabel("time (ms)")
        legend(fig, drawn)
        if stem is not None:
            save(fig, stem)
    return fig


def _sha(path: Path | None) -> str | None:
    return None if path is None or not Path(path).is_file() else sha256_of(path)


def record(s: LabelShot, paths: Paths, *, rule: str, n_candidates: int) -> dict:
    """What the figure of `s` was drawn from: the shot and the rule that picked
    it, each track's source (tier, what, path, sha256; null where no source
    has the shot), each store read, and the n map's gate."""
    return {
        "shot": s.shot,
        "year": s.year,
        "window": list(s.window),
        "pick_rule": rule,
        "candidates": n_candidates,
        "tracks": {
            t.spec.key: None
            if t.source is None
            else {
                "tier": t.source.tier,
                "what": t.source.what,
                "path": str(t.file),
                "sha256": _sha(t.file),
                "rows": len(t.rows),
            }
            for t in s.tracks
        },
        "stores": {
            event: None if path is None else {"path": str(path), "sha256": _sha(path)}
            for event, path in s.stores.items()
        },
        "n_gate": s.n_gate,
        "no_gate": s.no_gate,
        "corpus": str(paths.corpus_file(s.shot)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="a scratch build")
    parser.add_argument("--shot", type=int, help=f"default: {PICK_RULE}")
    parser.add_argument(
        "--no-gate", action="store_true", help="draw the n map without TokEye's gate"
    )
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    found = candidates(paths)
    read = read_sources(paths)
    if args.shot is None:
        chosen, rule = pick(found, read), PICK_RULE
    else:
        named = [c for c in found if c.shot == args.shot]
        if not named:
            raise SystemExit(f"{args.shot}: not a non-blind cohort shot")
        chosen, rule = named[0], NAMED
    torch.set_num_threads(LOGIN_THREADS)  # TokEye's gate: one shot, on the login node
    s = label_shot(paths, chosen, read, gate=not args.no_gate)
    args.out.mkdir(parents=True, exist_ok=True)
    draw(s, args.out / STEM)
    drawn_from = record(s, paths, rule=rule, n_candidates=len(found))
    manifest = args.out / MANIFEST
    with atomic_path(manifest) as tmp:
        tmp.write_text(json.dumps(drawn_from | {"git": git_sha()}, indent=1) + "\n")
    print(json.dumps({"out": str(args.out), "shot": s.shot, "score": score(s.tracks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
