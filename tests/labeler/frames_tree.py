"""A synthetic tree for Part B's frame models (`labeler.frames`, Tasks 2.5-2.11).

`build` makes, under `editor_tree.paths(tmp_path)`:
- one shot per event (`SHOTS`), in its event's roster, with a small review store
  (`spectrograms/<event>/<shot>.h5`, levels 1/8/64, the real stores' column
  widths `DT_MS`) holding the rows its spec's roles name, lit over `EVENT_MS`.
  The ELM store has a PCPHD03 row with data of its own (no ELMs) before the
  filterscope's, whose title carries the clipped suffix; the H-mode store's
  filterscope row has 7 lit channels (`HMODE_FS`), each with its own L-H drop
  (`HMODE_DROP_MS`), and the store has the density row the roles skip; the four
  ECE rows differ (`ECE_TE_KEV`, `ECE_RISE_KEV`) and each crashes at its own
  phase (`ECE_PHASE_MS`), so no robust scaling maps one row or channel onto
  another; the sawtooth store has no SXR row, an optional role left out;
- the format grids (`<event>/format/shots/<shot>.npz`, 50 ms cells, `grid_path`):
  Hiro's ELM onsets and the tearing archive's, present over `EVENT_MS`, and
  Jalal Butt's H and L grids, H over `EVENT_MS` and L before and after it, both
  on the 600 ms cell. `IP_SHOT` and `LEGACY` are legacy ELM shots beyond the
  cohort and every roster;
- the `ece_sawtooth` v2 table (`SAWTOOTH_SPANS`): present over `EVENT_MS` but
  for an uncertain 1000-1050 ms, and not observable over 1850-1900 ms;
- a cohort of 6 (`COHORT`: `BLIND` the blind one, which has every target, and
  `SPARE` with none) over `WINDOW`, and a population with one shot more
  (`POPULATION_ONLY`, over `POPULATION_WINDOW`);
- one owner save per event on its `SHOTS` shot, present over `EVENT_MS`;
- Ip: `IP_SHOT`'s in the raw cache and the ELM shot's in the corpus, each a
  1 MA plateau; `LEGACY` has none, so its window is its grid's hull.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from labeler.config import Paths
from labeler.events import rosters, suggestions
from labeler.events.catalog.cohort import POPULATION_COLUMNS
from labeler.events.catalog.states import NOT_OBSERVABLE, PRESENT, UNCERTAIN
from labeler.events.interval_tables import SAMPLE_MS, write_label_grid
from labeler.events.panels._shared import CLIPPED
from labeler.events.panels.neoclassical_tearing_mode import N_COLOURS, Z_DB
from labeler.events.panels.sawtooth_oscillation import CHANNEL_ROWS, ECE_BIN_MS
from labeler.events.review import labels, rows
from labeler.events.verify import mode_bytes

from . import editor_tree

ELM = "edge_localized_mode"
HMODE = "high_confinement_mode"
LMODE = "low_confinement_mode"
NTM = "neoclassical_tearing_mode"
SAWTOOTH = "sawtooth_oscillation"

SHOTS = {ELM: 190001, HMODE: 190002, NTM: 190003, SAWTOOTH: 190004}
SPARE = 190005
BLIND = 190006
COHORT = (*SHOTS.values(), SPARE, BLIND)
POPULATION_ONLY = 190007
IP_SHOT = 160001
LEGACY = 160002

WINDOW = (100, 1900)
POPULATION_WINDOW = (150, 1850)
EVENT_MS = (600, 1400)
#: Where each Ip record is over 50 kA: `assessed_window`'s answer.
IP_WINDOW = (200, 2200)  # IP_SHOT's, in the raw cache
CORPUS_IP_WINDOW = (50, 1950)  # the ELM shot's, which its catalog window beats
#: The legacy ELM grids' known cells, so their hulls.
IP_SHOT_HULL = (300, 2000)
LEGACY_HULL = (300, 2500)
#: The H and L grids' cells: unknown on the first and last (50 and 1900 ms).
HL_CELLS = (50, 1950)
L_MS = ((100, 650), (1400, 1900))
SAWTOOTH_TABLE = ("ece_sawtooth", "v2")
SAWTOOTH_SPANS = (
    (600, 1000, PRESENT),
    (1000, 1050, UNCERTAIN),
    (1050, 1400, PRESENT),
    (1850, 1900, NOT_OBSERVABLE),
)

STORE_T0_MS = -100.0
STORE_MS = 2200.0
DT_MS = {ELM: 0.05, HMODE: 0.1, NTM: 2.56, SAWTOOTH: 0.05}
#: The tearing mode's line in the NTM store: bin 102 (9.96 kHz), n = 1, 30 dB.
NTM_BIN = 102
NTM_N = 1
NTM_DB = 30.0
NTM_DY_KHZ = 100 / 1024
NTM_BINS = 308
#: The H-mode filterscope row's lit channels, as on 186561, where FS01 is dark,
#: and each one's L-H drop: 10 ms apart, each in the middle of its own 10 ms
#: sub-frame (`hmode_frames`'), so the features tell each drop from the others.
HMODE_FS = tuple(f"FS{c:02d}" for c in range(2, 9))
HMODE_DROP_MS = tuple(EVENT_MS[0] + 5.0 + 10.0 * c for c in range(len(HMODE_FS)))
#: Each ECE row's Te on its first channel, and its rise between crashes (keV),
#: core to edge: a channel is 0.1 keV cooler than the one before, and the outer
#: row's rise is inverted, as outside the inversion radius.
ECE_TE_KEV = (2.0, 1.6, 1.2, 0.8)
ECE_RISE_KEV = (0.3, 0.2, 0.1, -0.1)
#: The sawtooth period, and each ECE row's crash phase in it: 5 ms apart, and off
#: the 2 ms sub-frames' edges, so each crash is inside one of them.
SAWTOOTH_MS = 20.0
ECE_PHASE_MS = (2.5, 7.5, 12.5, 17.5)


def grid_path(paths: Paths, event: str, shot: int) -> Path:
    """A format grid: `<label tables>/<event>/format/shots/<shot>.npz`."""
    return paths.label_tables / event / "format" / "shots" / f"{int(shot)}.npz"


def _during(t, span=EVENT_MS) -> np.ndarray:
    return (t >= span[0]) & (t < span[1])


def sawtooth_rise(t, phase_ms: float) -> np.ndarray:
    """0 to 1 between crashes `phase_ms` into each period, over `EVENT_MS`; 0
    elsewhere."""
    return np.where(_during(t), ((t - phase_ms) % SAWTOOTH_MS) / SAWTOOTH_MS, 0.0)


def _trace(name: str, title: str, y, units: str, legend=()) -> rows.TraceRow:
    """A trace row whose columns' minimum and maximum are both `y`, `(C, n)`."""
    y = np.atleast_2d(np.asarray(y, dtype=np.float32))
    values = np.stack([y, y])
    return rows.TraceRow(name, title, values, y_units=units, legend=list(legend))


def _elm_rows(t, rng) -> list:
    spikes = _during(t) & (t % 20.0 < 1.0)  # an ELM every 20 ms
    fs01 = 1.0 + 4.0 * spikes + 0.05 * rng.standard_normal(len(t))
    # The row the role skips: a level of its own and a slow swing, no ELMs.
    pcphd03 = 3.0 + 0.5 * np.sin(2 * np.pi * t / 50.0)
    pcphd03 = pcphd03 + 0.05 * rng.standard_normal(len(t))
    co2 = np.full((16, len(t)), 30, dtype=np.uint8)
    return [
        rows.ImageRow(
            "p0",
            "CO2 R0 power",
            co2,
            y0=0.0,
            dy=0.9765625,
            y_units="kHz",
            z_lo=-3.0,
            z_hi=27.0,
            z_units="",
        ),
        _trace("p1", "D-alpha PCPHD03", pcphd03, "D-α"),
        _trace("p2", "D-alpha FS01, the ELM spans' channel" + CLIPPED, fs01, "D-α"),
    ]


def _hmode_rows(t, rng) -> list:
    h = _during(t)
    gain = 1.0 + 0.1 * np.arange(len(HMODE_FS))[:, None]  # each channel its own
    in_h = (t >= np.asarray(HMODE_DROP_MS)[:, None]) & (t < EVENT_MS[1])
    dalpha = gain * np.where(in_h, 0.5, 2.0)
    dalpha = dalpha + 0.05 * rng.standard_normal(dalpha.shape)
    nbi = np.where((t >= 300.0) & (t < 1700.0), 6.0, 0.0)
    return [
        _trace("p0", "D-alpha filterscopes", dalpha, "D-α", HMODE_FS),
        _trace("p1", "density, CO2 R0 (1 ms mean)", np.where(h, 5.0, 2.5), "n_e"),
        _trace("p2", "NBI power", nbi, "MW"),
    ]


def _ntm_rows(t, rng) -> list:
    lo, hi = Z_DB
    z = rng.uniform(lo, lo + 6.0, (NTM_BINS, len(t)))
    z[NTM_BIN, _during(t)] = NTM_DB
    power = np.clip(np.rint((z - lo) * 255 / (hi - lo)), 0, 255).astype(np.uint8)
    modes = np.zeros(z.shape, dtype=np.int64)
    modes[NTM_BIN] = NTM_N
    n = sorted(N_COLOURS)
    meta = {"n": n, "levels": 256 // len(n), "colours": [N_COLOURS[v] for v in n]}
    axis = {
        "y0": 0.0,
        "dy": NTM_DY_KHZ,
        "y_units": "kHz",
        "z_lo": lo,
        "z_hi": hi,
        "z_units": "",
    }
    return [
        rows.ImageRow("p0", "MPI66M322D power", power, **axis),
        rows.ImageRow(
            "p1",
            "toroidal n, MPI66M probes",
            mode_bytes(z, modes, N_COLOURS, lo, hi),
            **axis,
            modes=meta,
        ),
    ]


def _sawtooth_rows(t, rng) -> list:
    out = []
    shapes = zip(CHANNEL_ROWS, ECE_TE_KEV, ECE_RISE_KEV, ECE_PHASE_MS, strict=True)
    for i, (channels, te0, size, phase) in enumerate(shapes):
        rise = size * sawtooth_rise(t, phase)  # a crash every 20 ms
        te = te0 - 0.1 * np.arange(len(channels))[:, None] + rise
        te = te + 0.01 * rng.standard_normal(te.shape)
        title = f"ECE Te, ch {channels[0]}-{channels[-1]} ({ECE_BIN_MS:g} ms median)"
        legend = [f"ch {c}" for c in channels]
        out.append(_trace(f"p{i}", title, te, "keV", legend))
    return out


STORE_ROWS = {
    ELM: _elm_rows,
    HMODE: _hmode_rows,
    NTM: _ntm_rows,
    SAWTOOTH: _sawtooth_rows,
}


def _store(paths: Paths, event: str, shot: int, rng) -> None:
    dt = DT_MS[event]
    grid = rows.Grid(t0_ms=STORE_T0_MS, dt_ms=dt, n=round(STORE_MS / dt))
    t = grid.t0_ms + (np.arange(grid.n) + 0.5) * dt
    rows.write(
        paths.spectrogram_file(event, shot),
        grid,
        STORE_ROWS[event](t, rng),
        builder="panel_rows",
        event=event,
        shot=int(shot),
    )


def _roster(paths: Paths, event: str, shots) -> None:
    frame = pd.DataFrame(
        [
            {"shot": shot, "tier": "unverified", "holdout": "false"}
            | dict.fromkeys(("reviewers", "verified_on", "notes"), "")
            for shot in shots
        ],
        columns=list(rosters.ROSTER_COLUMNS),
    )
    rosters.write_roster(frame, rosters.roster_path(event, root=paths.label_tables))


def _grid(path: Path, cells, known, present, *, counts: bool = False) -> None:
    """50 ms cells from `cells[0]` to `cells[1]`: unknown off `known`, 1 on the
    spans `present` (inside `known`), 0 elsewhere; with `counts`, ELM onsets'
    `event_count`, 2 in every present cell."""
    t = np.arange(cells[0], cells[1], SAMPLE_MS)
    label = np.where(_during(t, known), 0.0, np.nan)
    for span in present:
        label[_during(t, span)] = 1.0
    extras = {"event_count": np.where(label == 1.0, 2, 0)} if counts else {}
    categories = {"0": "absent", "1": "present"}
    write_label_grid(path, t, label, categories=categories, **extras)


def _grids(paths: Paths) -> None:
    for shot in (SHOTS[ELM], BLIND):
        _grid(grid_path(paths, ELM, shot), (0, 2000), WINDOW, [EVENT_MS], counts=True)
    legacy = ((IP_SHOT, (0, 2500), IP_SHOT_HULL), (LEGACY, (0, 3000), LEGACY_HULL))
    for shot, cells, hull in legacy:
        _grid(grid_path(paths, ELM, shot), cells, hull, [EVENT_MS], counts=True)
    for shot in (SHOTS[HMODE], BLIND):
        _grid(grid_path(paths, HMODE, shot), HL_CELLS, WINDOW, [EVENT_MS])
        _grid(grid_path(paths, LMODE, shot), HL_CELLS, WINDOW, L_MS)
    for shot in (SHOTS[NTM], BLIND):
        _grid(grid_path(paths, NTM, shot), (0, 2000), (0, 2000), [EVENT_MS])


def sawtooth_table(paths: Paths) -> Path:
    """The `ece_sawtooth` v2 suggestion table's path."""
    return suggestions.table_path(paths, SAWTOOTH, *SAWTOOTH_TABLE)


def _sawtooth(paths: Paths) -> None:
    method, version = SAWTOOTH_TABLE
    lines = [
        row
        for shot in (SHOTS[SAWTOOTH], BLIND)
        for row in suggestions.span_rows(shot, WINDOW, SAWTOOTH_SPANS)
    ]
    meta = {"event": SAWTOOTH, "method": method, "version": version}
    suggestions.write_table(sawtooth_table(paths), lines, meta)


def _population(paths: Paths) -> None:
    """`catalog/population.csv` (`spans.population`); unnamed columns blank."""
    windows = dict.fromkeys(COHORT, WINDOW) | {POPULATION_ONLY: POPULATION_WINDOW}
    blank = dict.fromkeys(POPULATION_COLUMNS, "")
    frame = pd.DataFrame(
        [
            blank
            | {
                "shot": shot,
                "in_cohort": shot in COHORT,
                "window_start_ms": window[0],
                "window_end_ms": window[1],
            }
            for shot, window in windows.items()
        ],
        columns=list(POPULATION_COLUMNS),
    )
    path = paths.catalog / "population.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _ip(path: Path, plasma) -> None:
    """A corpus-format `ip` group, 0-2500 ms: 1 MA from `plasma[0]` to
    `plasma[1]` inclusive, 0 elsewhere."""
    t = np.arange(0.0, 2500.0, 0.5)
    ip = np.where((t >= plasma[0]) & (t <= plasma[1]), 1e6, 0.0)
    editor_tree.write(path, {"ip": (t, ip)})


def build(tmp_path: Path, seed: int = 0) -> Paths:
    """The tree above under `tmp_path`; its `Paths`."""
    paths = editor_tree.paths(tmp_path)
    rng = np.random.default_rng(seed)
    for event, shot in SHOTS.items():
        _store(paths, event, shot, rng)
        _roster(paths, event, [shot])
        saved = labels.normalise(WINDOW, [(*EVENT_MS, PRESENT)])
        labels.save(paths.label_tables / event, shot, saved, source=None)
    _grids(paths)
    _sawtooth(paths)
    editor_tree.cohort(
        paths,
        [
            editor_tree.queue_row(shot, rank, blind=shot == BLIND, window=WINDOW)
            for rank, shot in enumerate(COHORT, 1)
        ],
    )
    _population(paths)
    _ip(paths.raw_cache / f"{IP_SHOT}_processed.h5", IP_WINDOW)
    _ip(paths.corpus_file(SHOTS[ELM]), CORPUS_IP_WINDOW)
    return paths
