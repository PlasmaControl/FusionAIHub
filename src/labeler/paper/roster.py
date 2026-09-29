"""One roster shot's suggestions from the models, drawn for a scratch build (D51).

    PYTHONPATH=src pixi run -e labelmaker python -m labeler.paper.roster \\
        --out DIR [--shot SHOT] [--version V] [--seg-version V] \\
        [--tables-version V]

The AE shots the paper's interpreter figure draws carry none of the other
events' groups, so no figure built on them can show the other phenomena. This
one draws a non-blind roster shot of the frozen cohort with at least 2 s of
corpus CO2 (`candidates`: 153 of its 450 non-blind shots):

- AE from the frame model (`AE_VERSION`) and SegNet's mask (`SEG_VERSION`),
  both run here over the shot's corpus CO2 rows (`data.raw_rows`), as the AE
  extension runs them; the picture is those rows pooled to SegNet's store level
  (`shots.PICTURE_LEVEL`), and the mask lies on its pixels;
- NTM, H-mode, ELMing and sawteeth from their frame models' suggestion tables
  (`TABLES`, `TABLE_VERSION`): a track says `NO_TABLE` where a table does not
  exist and `NOT_APPLIED` where the shot is not in it.

**What it is not.** Everything on it is a suggestion: the title says so
(`TITLE`), every track's key is `SUGGESTED`'s (the mask's is
`shots.MASK_LABEL`), no track is the owner's, and `roster.json` says
`"tier": "suggestions"`. It is never a reviewed label, and it goes into a
scratch build only (the Part B build's
`$LABELER_ROOT/scratch/paper-round-three-b/`): `main` refuses the paper's own
`paper/` and anything inside it (`ScratchOnly`) before it reads anything.

**The shot** is `PICK_RULE`'s (`pick`), or the one `--shot` names (`NAMED`),
which must be a candidate. `roster.json` records it, the rule, the number of
candidates, and each model and table it read, by path and sha256 (a table that
does not exist as null).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from ..ae import seg as ae_seg
from ..ae import xpower
from ..ae.seg import train as seg_train
from ..ae.seg.poi import ae_pixels
from ..ae.xpower.data import raw_rows, targets, window_frames
from ..ae.xpower.evaluate import chosen_model
from ..ae.xpower.extend import MIN_CO2_S, frame_states
from ..ae.xpower.gallery import STATE_COLOURS
from ..ae.xpower.train import load, probabilities
from ..config import Paths, atomic_path, git_sha, sha256_of
from ..events import suggestions
from ..events.catalog.cohort import read_cohort
from ..events.catalog.states import PRESENT
from ..events.review import labels
from ..events.review.labels import Label
from ..events.review.rows import Grid, pool
from ..events.spans import cohort_path
from ..events.verify import corpus_signal
from ..scoring.frames import FRAME_MS
from . import AE, ORDER, PAGE_IN, paper_dir, save, style, title
from .build import LOGIN_THREADS
from .shots import (
    MARGIN_MS,
    PICTURE_LEVEL,
    STATE_NAMES,
    _legend,
    _mask,
    _spectrogram,
    runs,
)

#: The other paper phenomena, in `ORDER`'s order, and each one's frame model
#: (a `labeler.frames.SPECS` key), whose suggestion table the track reads.
TABLES = {
    "neoclassical_tearing_mode": "ntm_frames",
    "high_confinement_mode": "hmode_frames",
    "edge_localized_mode": "elm_frames",
    "sawtooth_oscillation": "sawtooth_frames",
}
TABLE_VERSION = "v1"
AE_VERSION = "v3"
SEG_VERSION = "v2"  # v3 was worse than v2 on MHD lines, so the build keeps v2
STEM = "fig_roster_interpreter"
MANIFEST = "roster.json"
PICK_RULE = (
    "the non-blind roster shot with at least 2 s of corpus CO2 on which the most "
    "of the four frame models suggest a present span, then the most present time "
    "over the four, then the lower shot number"
)
NAMED = "named by --shot"
NO_TABLE = "no table"
NOT_APPLIED = "not applied to this shot"
SUGGESTED = "suggested: {}"
TITLE = "shot {shot} ({year}): the models' suggestions, not reviewed"
TIER = "suggestions"
TEXT_COLOUR = "#888888"


class ScratchOnly(ValueError):
    """The roster figure goes into a scratch build, never the paper's own."""


@dataclass(frozen=True)
class Candidate:
    shot: int
    year: int
    window: tuple[int, int]  # the cohort's window_start_ms, window_end_ms


@dataclass(frozen=True)
class RosterShot:
    """The drawn shot. `prob` is v3's P(AE) per 10 ms frame from `first`; the
    picture (`grid`, `image`, `y0`, `dy`) and SegNet's `mask` are the corpus
    rows pooled to `PICTURE_LEVEL`; `tracks` holds, per `ORDER` category, the
    suggested state per frame, or the text `NO_TABLE` or `NOT_APPLIED`."""

    shot: int
    year: int
    window: tuple[int, int]
    grid: Grid
    image: np.ndarray  # (n_y, grid.n) uint8: the first cross-power row, pooled
    y0: float
    dy: float
    first: int
    prob: np.ndarray
    threshold: float
    mask: np.ndarray | None  # (n_y, grid.n) bool: SegNet's AE pixels; None: not run
    tracks: dict

    @property
    def edges(self) -> np.ndarray:
        return (self.first + np.arange(len(self.prob) + 1)) * FRAME_MS


def candidates(paths: Paths) -> list[Candidate]:
    """The frozen cohort's non-blind shots with at least `MIN_CO2_S` of corpus
    CO2, by shot."""
    cohort = read_cohort(cohort_path(paths))
    keep = cohort[~cohort["blind"] & (cohort["span_co2_s"] >= MIN_CO2_S)]
    columns = ("shot", "year", "window_start_ms", "window_end_ms")
    found = [
        Candidate(int(shot), int(year), (int(lo), int(hi)))
        for shot, year, lo, hi in zip(*(keep[c] for c in columns), strict=True)
    ]
    return sorted(found, key=lambda c: c.shot)


def table_file(paths: Paths, category: str, version: str = TABLE_VERSION) -> Path:
    """The suggestion table `category`'s frame model wrote."""
    return suggestions.table_path(paths, category, TABLES[category], version)


def read_tables(
    paths: Paths, version: str = TABLE_VERSION
) -> dict[str, dict[int, Label] | None]:
    """Each frame model's table as labels by shot (`labels.read_labels`: its
    window from the first to the last row, the absent rows dropped), in
    `TABLES`' order; None where the table does not exist."""
    out = {}
    for category in TABLES:
        file = table_file(paths, category, version)
        out[category] = labels.read_labels(file) if file.is_file() else None
    return out


def present_ms(label: Label | None) -> int:
    """The present spans' total time, ms; 0 for no label."""
    if label is None:
        return 0
    return int(sum(b - a for a, b, c in label.intervals if c == PRESENT))


def pick(
    found: list[Candidate], tables: Mapping[str, Mapping[int, Label] | None]
) -> Candidate:
    """`PICK_RULE`'s shot among `found`: the most of the four tables with some
    present time on it, then the most present time over the four, then the
    lower shot number. A missing table (None) counts 0."""
    if not found:
        raise ValueError("no candidate shot to pick from")

    def key(c: Candidate) -> tuple[int, int, int]:
        times = [present_ms((tables.get(t) or {}).get(c.shot)) for t in TABLES]
        return (-sum(t > 0 for t in times), -sum(times), c.shot)

    return min(found, key=key)


def roster_shot(
    paths: Paths,
    candidate: Candidate,
    *,
    model_file: Path,
    seg_file: Path | None = None,
    tables: Mapping[str, Mapping[int, Label] | None],
) -> RosterShot:
    """`candidate` with the frame model in `model_file`, and SegNet in
    `seg_file` if given, run over its corpus CO2 rows, and its tracks from
    `tables` (`read_tables`)."""
    net, blob = load(model_file)
    co2 = corpus_signal(candidate.shot, "co2", corpus=paths.corpus)
    rows = raw_rows(co2.x, co2.y)
    first, n = window_frames(candidate.window)
    prob, observed = probabilities(net, rows, first, n, band=blob["band_khz"])
    threshold = float(blob["threshold"])
    grid, values, y0, dy = rows
    pooled = pool(values, PICTURE_LEVEL, "image")
    picture = Grid(grid.t0_ms, grid.dt_ms * PICTURE_LEVEL, -(-grid.n // PICTURE_LEVEL))
    mask = None
    if seg_file is not None:
        seg_net, seg_blob = seg_train.load(seg_file)
        mask = ae_pixels(
            seg_train.predict(seg_net, pooled),
            float(seg_blob["threshold"]),
            y0,
            dy,
            band=seg_train.blob_band(seg_blob),
        )
    tracks: dict = {AE: frame_states(prob, observed, threshold)}
    for category in TABLES:
        table = tables.get(category)
        if table is None:
            tracks[category] = NO_TABLE
        elif candidate.shot not in table:
            tracks[category] = NOT_APPLIED
        else:
            tracks[category] = targets(table[candidate.shot], first, n)
    return RosterShot(
        shot=candidate.shot,
        year=candidate.year,
        window=candidate.window,
        grid=picture,
        image=pooled[0],
        y0=y0,
        dy=dy,
        first=first,
        prob=prob,
        threshold=threshold,
        mask=mask,
        tracks=tracks,
    )


def _text_track(ax, text: str) -> None:
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=TEXT_COLOUR,
        style="italic",
    )


def _state_bars(ax, s: RosterShot, states: np.ndarray) -> None:
    """One bar per suggested state the track holds, keyed as a suggestion."""
    edges = s.edges
    for state, colour in STATE_COLOURS.items():
        spans = [(edges[a], edges[b] - edges[a]) for a, b in runs(states == state)]
        if spans:
            label = SUGGESTED.format(STATE_NAMES[state])
            ax.broken_barh(spans, (0.1, 0.8), color=colour, lw=0, label=label)


def draw(s: RosterShot, stem: Path) -> Figure:
    """The spectrogram with SegNet's mask over one track per phenomenon, each
    the models' suggestions."""
    t0, t1 = s.edges[0] - MARGIN_MS, s.edges[-1] + MARGIN_MS
    with style():
        fig = Figure(figsize=(PAGE_IN, 3.4), layout="constrained")
        spec, *tracks = fig.subplots(
            1 + len(ORDER),
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [4] + [0.45] * len(ORDER)},
        )
        _spectrogram(spec, s)
        spec.set_xlim(t0, t1)
        _mask(spec, s)
        spec.set_title(TITLE.format(shot=s.shot, year=s.year))
        for ax, category in zip(tracks, ORDER, strict=True):
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_ylabel(title(category), rotation=0, ha="right", va="center")
            track = s.tracks[category]
            if isinstance(track, str):
                _text_track(ax, track)
            else:
                _state_bars(ax, s, np.asarray(track))
        for ax in tracks[:-1]:
            ax.tick_params(bottom=False)
        tracks[-1].set_xlabel("time (ms)")
        _legend(fig)
        save(fig, stem)
    return fig


def _pinned(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_of(path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out", type=Path, required=True, help="a scratch build, never paper/"
    )
    parser.add_argument("--shot", type=int, help=f"default: {PICK_RULE}")
    parser.add_argument(
        "--version",
        default=AE_VERSION,
        help=f"the AE frame model's version (default {AE_VERSION})",
    )
    parser.add_argument(
        "--seg-version",
        default=SEG_VERSION,
        help=f"the segmentation's version (default {SEG_VERSION})",
    )
    parser.add_argument(
        "--tables-version",
        default=TABLE_VERSION,
        help=f"the frame models' tables' version (default {TABLE_VERSION})",
    )
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    paper = paper_dir(paths).resolve()
    if args.out.resolve().is_relative_to(paper):
        raise ScratchOnly(
            f"{args.out}: the roster figure is the models' suggestions and goes "
            f"into a scratch build, never {paper} or inside it"
        )
    found = candidates(paths)
    if args.shot is None:
        named = None
    else:
        named = next((c for c in found if c.shot == args.shot), None)
        if named is None:
            raise ValueError(
                f"{args.shot}: not a non-blind roster shot with corpus CO2"
            )
    tables = read_tables(paths, args.tables_version)
    if named is None:
        chosen, rule = pick(found, tables), PICK_RULE
    else:
        chosen, rule = named, NAMED
    model_file = chosen_model(xpower.model_dir(paths, args.version))
    seg_file = ae_seg.model_dir(paths, args.seg_version) / "model.pt"
    seg_file = seg_file if seg_file.is_file() else None
    torch.set_num_threads(LOGIN_THREADS)  # one shot, on the login node
    s = roster_shot(
        paths, chosen, model_file=model_file, seg_file=seg_file, tables=tables
    )
    args.out.mkdir(parents=True, exist_ok=True)
    draw(s, args.out / STEM)
    record = {
        "shot": s.shot,
        "year": s.year,
        "window": list(s.window),
        "pick_rule": rule,
        "candidates": len(found),
        "ae_version": args.version,
        "ae_model": _pinned(model_file),
        "seg_version": args.seg_version,
        "seg_model": None if seg_file is None else _pinned(seg_file),
        "tables": {
            method: (
                None
                if tables[category] is None
                else _pinned(table_file(paths, category, args.tables_version))
            )
            for category, method in TABLES.items()
        },
        "tier": TIER,
        "git": git_sha(),
    }
    manifest = args.out / MANIFEST
    with atomic_path(manifest) as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"out": str(args.out), "shot": s.shot, "manifest": str(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
