r"""fig_event_coverage: for each event, which DIII-D shots have been scanned and on which
of them the event is active, over the whole shot range (a supplementary figure).

    PYTHONPATH=src pixi run --frozen -e labeler \
        python scripts/labeler/paper/fig_event_coverage.py \
        [--out PATH.pdf] [--png PATH.png] [--numbers dev/label_paper/numbers.tex] \
        [--ae-shots PATH/shots.parquet]

One horizontal strip per event (`EVENTS`; only AE for now), shot number on x from
`FIRST_SHOT` to the latest available shot (the d3drdb snapshot's last shot, or a
published table's, whichever is later). The strip says only three things: the event is
in this shot, it is not, or the shot is not scanned yet. Every shot number in the range
falls in one class:

- activity: scanned, with at least `ACTIVITY_MS` ms of the event in its assessed window;
- quiet: scanned, with less;
- not scanned: a plasma shot with no label yet, or whose run is pending or failed
  (hatched);
- blank: not a plasma shot (no d3drdb row with pulse_length > 0.1 s and |ipmax| > 100 kA,
  the all-shot run's candidate rule), or a plasma shot with no usable record for the
  event (`NO_DATA`: not on FDP, no CO2 file, CO2 below 500 kHz, ...). Blank is left
  unfilled inside the strip's outline, so it never reads as quiet.

Rendering: about 109,000 shot numbers across about 6.3 in of strip is some 30 shots to a
printer dot at 600 dpi, so a per-shot image would be resampled by whatever viewer or
printer shows it, which keeps or drops a sparse class at will. The strip is binned
instead: each column is `BIN_SHOTS` consecutive shot numbers and is filled full height
in one class, the first of activity, quiet, not scanned that any shot in it has (blank
if none), so one shot with the event marks its column. Runs of equal columns are drawn
as vector rectangles. The header counts are per shot.

Sources, per event, in order. AE: the all-shot run's `shots.parquet` (written by
`scripts/labeler/ae_fdp_label.py publish`, one row per plasma candidate with its status
and `present_ms`) when it exists; until then the labels made so far, a placeholder: the
model's silver extension (`extend_ae_seldnet/silver_shots.csv`, 10 ms frames) with the
expert-reviewed table (`review/labels.csv`) painted over it where it has rows, every
other plasma candidate (`ae_fdp_1ms/candidates.csv`, else the same rule on the d3drdb
snapshot) not scanned. The source each row used is printed, written to the sidecar JSON
next to the PDF, named in the row's header, and carried into the caption macros:
`--numbers` rewrites their block in numbers.tex from this run, and counts from a
placeholder source are wrapped in `\scanProvisional`, which prints them in orange.

To add an event: write a loader that returns a `Status` (a table of shot, status,
present_ms for the plasma candidates and the scanned shots) and append an `Event` to
`EVENTS`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths
from labeler.paper import FONT_PT, PAGE_IN, STYLE, theme

# ---- what the figure claims ------------------------------------------------------
#: "significant activity": at least this much of the event, summed over the shot's
#: assessed window (on the labels' own frames: 1 ms for the all-shot run, 10 ms for
#: the placeholder labels)
ACTIVITY_MS = 50
FIRST_SHOT = 100_000
#: shot numbers per strip column
BIN_SHOTS = 100
#: run statuses (ae_fdp_label.py) that mean the shot has no usable record for the
#: event, so it can never be scanned: blank, like a non-plasma shot. Every other status
#: that is not "labeled" (pending, failed, infer_failed, extract_error, bad_record,
#: chord_timebase_mismatch, timebase_not_increasing, ...) counts as not scanned yet.
NO_DATA = frozenset(
    {
        "no_view",  # FDP has no per-shot view
        "no_bci",  # the view has no CO2 PTDATA file
        "no_co2_points",
        "rate_below_500khz",
        "no_record_in_window",
        "nonfinite",
        "uncalibrated",
    }
)
#: in priority order: a column takes the first class any of its shots has
CLASSES = ("activity", "quiet", "not_scanned", "blank")
ACTIVITY, QUIET, NOT_SCANNED, BLANK = range(4)

# ---- inputs ----------------------------------------------------------------------
PATHS = Paths.from_env()
TABLES = PATHS.label_tables  # /scratch/gpfs/EKOLEMEN/D3D/events
D3DRDB = Path("/scratch/gpfs/EKOLEMEN/D3D/text/d3drdb_20261006T135944Z/SUMMARIES.parquet")
#: ae_fdp_label.py's candidate rule (MIN_PULSE_S, MIN_IP_A)
MIN_PULSE_S, MIN_IP_A = 0.1, 1e5
AE_PUBLISHED = Path(
    "/projects/EKOLEMEN/fusionaihub_labels/alfven_eigenmode/ae_seldnet_fdp_1ms/shots.parquet"
)
AE_CANDIDATES = PATHS.root / "ae_fdp_1ms" / "candidates.csv"
AE_SILVER = TABLES / "alfven_eigenmode" / "extend_ae_seldnet" / "silver_shots.csv"
AE_REVIEW = TABLES / "alfven_eigenmode" / "review" / "labels.csv"

# ---- outputs ---------------------------------------------------------------------
DEFAULT_OUT = REPO / "dev" / "label_paper" / "figures" / "event_coverage.pdf"
DEFAULT_PNG = PATHS.root / "scratch" / "event_coverage.png"
BLOCK_BEGIN = (
    "% ---- Event coverage figure, supplementary [figures/event_coverage.json]. "
    "scripts/labeler/paper/fig_event_coverage.py --numbers rewrites this block; "
    "rerun it, do not edit ----"
)
BLOCK_END = "% ---- end of the event coverage block ----"

# ---- look ------------------------------------------------------------------------
INK = "#333333"
MUTED = "#6b6b6b"
QUIET_FILL = "#cfcfcf"  # light grey, the confinement figure's gate grey
HATCH_INK = "#8c8c8c"  # not scanned: white with 45-degree grey lines (reads in greyscale)
HATCH = "//////"
OUTLINE = "#9a9a9a"  # hairline round each strip; blank is the unfilled inside
# figure geometry, inches
LEFT_IN, RIGHT_IN = 0.22, 0.22
LEGEND_IN = 0.17
HEADER_IN, STRIP_IN, GAP_IN = 0.17, 0.26, 0.05
AXIS_IN = 0.31
ROW_IN = HEADER_IN + STRIP_IN + GAP_IN

sources: dict[str, str] = {}


def note_source(path: Path) -> str:
    """Keep `path`'s sha256 for the sidecar; return the path as text."""
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(path)


@dataclass
class Status:
    """One event's per-shot status: `table` has shot, status ("labeled" or a run
    status) and present_ms (labeled shots) for the plasma candidates and every scanned
    shot; shots it does not list are not plasma candidates."""

    table: pd.DataFrame
    final: bool
    name: str  # in the figure's row header
    caption: str  # LaTeX, the \scan<Event>source macro
    paths: list[str]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Event:
    key: str  # catalog category
    macro: str  # letters only: \scan<macro>active, ...
    label: str  # row header
    colour: str
    load: Callable[[argparse.Namespace], Status]


# ---- loaders ---------------------------------------------------------------------


def summaries() -> pd.DataFrame:
    note_source(D3DRDB)
    return pd.read_parquet(D3DRDB, columns=["shot", "pulse_length", "ipmax"])


def plasma_rule(s: pd.DataFrame) -> pd.Series:
    return (s.pulse_length > MIN_PULSE_S) & (s.ipmax.abs() > MIN_IP_A)


def frames_ms(rows: pd.DataFrame, out: dict[int, np.ndarray]) -> None:
    """Paint `rows` (shot, category, t_start, t_end in ms) onto 1 ms arrays, one per
    shot (-1 unassessed); a later call paints over an earlier one."""
    for shot, g in rows.groupby("shot"):
        end = int(np.ceil(g.t_end.max()))
        arr = out.get(int(shot))
        if arr is None or len(arr) < end:
            grown = np.full(end, -1, np.int8)
            if arr is not None:
                grown[: len(arr)] = arr
            arr = grown
        for r in g.itertuples(index=False):
            arr[int(round(r.t_start)) : int(round(r.t_end))] = int(r.category)
        out[int(shot)] = arr


def ae_published(path: Path) -> Status:
    note_source(path)
    t = pd.read_parquet(path, columns=["shot", "status", "present_ms"])
    t["present_ms"] = t.present_ms.astype("float")
    made = datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m-%d")
    return Status(
        table=t,
        final=True,
        name=f"all-shot run, published {made}",
        caption=r"\seldnet{} over every plasma shot, on 1\,ms frames",
        paths=[str(path)],
    )


def ae_fallback() -> Status:
    """The labels made so far: the silver extension, the reviewed table over it."""
    if AE_CANDIDATES.is_file():
        note_source(AE_CANDIDATES)
        cands = pd.read_csv(AE_CANDIDATES, usecols=["shot"]).shot.astype(int)
        cand_from = str(AE_CANDIDATES)
    else:
        s = summaries()
        cands = s.shot[plasma_rule(s)].astype(int)
        cand_from = f"{D3DRDB} (pulse_length > {MIN_PULSE_S} s, |ipmax| > {MIN_IP_A:.0f} A)"
    note_source(AE_SILVER)
    note_source(AE_REVIEW)
    silver = pd.read_csv(AE_SILVER, usecols=["shot", "category", "t_start", "t_end"])
    review = pd.read_csv(AE_REVIEW, usecols=["shot", "category", "t_start", "t_end"])
    arrays: dict[int, np.ndarray] = {}
    frames_ms(silver, arrays)
    frames_ms(review, arrays)  # reviewed rows (any category) win where they exist
    rows = []
    for shot, arr in arrays.items():
        assessed = int(np.isin(arr, (0, 1)).sum())
        if assessed:
            rows.append({"shot": shot, "status": "labeled",
                         "present_ms": float((arr == 1).sum())})
    scanned = pd.DataFrame(rows)
    rest = pd.DataFrame({"shot": sorted(set(cands) - set(scanned.shot)),
                         "status": "pending", "present_ms": np.nan})
    n_silver, n_review = silver.shot.nunique(), review.shot.nunique()
    n_both = len(set(silver.shot) & set(review.shot))
    return Status(
        table=pd.concat([scanned, rest], ignore_index=True),
        final=False,
        name=f"placeholder, labels so far ({len(scanned)} shots)",
        caption=(
            r"a placeholder until the all-shot run is published: the labels made so far, "
            rf"\seldnet{{}} on {fmt(n_silver)} shots with the expert-reviewed labels "
            rf"({fmt(n_review)} shots) in its place where they exist"
        ),
        paths=[str(AE_SILVER), str(AE_REVIEW), cand_from],
        notes=[
            f"silver extension {n_silver} shots, reviewed table {n_review} shots, "
            f"{n_both} in both (the reviewed rows win where they exist), "
            f"{len(scanned)} scanned in all",
            "the all-shot run's shots.parquet does not exist yet: "
            f"{AE_PUBLISHED}",
        ],
    )


def ae_status(args: argparse.Namespace) -> Status:
    path = args.ae_shots
    return ae_published(path) if path.is_file() else ae_fallback()


EVENTS = (
    Event("alfven_eigenmode", "AE", "AE", theme.SEA_GREEN, ae_status),
)

# ---- classes and counts ----------------------------------------------------------


def classify(table: pd.DataFrame) -> pd.Series:
    """Class code per shot of a `Status.table`."""
    st = table.status.astype(str)
    labeled = st.eq("labeled")
    active = labeled & (table.present_ms.fillna(0) >= ACTIVITY_MS)
    code = np.where(active, ACTIVITY,
                    np.where(labeled, QUIET,
                             np.where(st.isin(NO_DATA), BLANK, NOT_SCANNED)))
    return pd.Series(code.astype(np.int8), index=table.shot.astype(int).to_numpy())


def per_shot(table: pd.DataFrame, first: int, last: int) -> np.ndarray:
    """Class code for every shot number first..last (blank where the table is silent)."""
    codes = np.full(last - first + 1, BLANK, np.int8)
    cls = classify(table)
    cls = cls[(cls.index >= first) & (cls.index <= last)]
    codes[cls.index.to_numpy() - first] = cls.to_numpy()
    return codes


def columns(codes: np.ndarray, first: int) -> list[tuple[int, int, int]]:
    """Runs (x0, x1, class) of strip columns: each column is BIN_SHOTS shot numbers in the
    first class (CLASSES order) any of them has; equal neighbours are merged."""
    last = first + len(codes) - 1
    edges = np.r_[np.arange(first, last + 1, BIN_SHOTS), last + 1]
    col = np.minimum.reduceat(codes, edges[:-1] - first)
    change = np.flatnonzero(np.diff(col)) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(col)]
    return [(int(edges[a]), int(edges[b]), int(col[a])) for a, b in zip(starts, ends, strict=True)]


def breakdown(status: Status, s: pd.DataFrame, first: int, last: int) -> dict:
    """What the not-scanned and blank shot numbers are, for the sidecar."""
    t = status.table
    t = t[(t.shot >= first) & (t.shot <= last)]
    st = t.status.astype(str)
    rng = s[(s.shot >= first) & (s.shot <= last)]
    listed = set(t.shot.astype(int))
    unlisted = rng[~rng.shot.isin(listed)]
    return {
        "not_scanned_statuses": st[~st.eq("labeled") & ~st.isin(NO_DATA)]
        .value_counts().to_dict(),
        "blank": {
            "no_usable_record_statuses": st[st.isin(NO_DATA)].value_counts().to_dict(),
            "d3drdb_row_not_plasma": int(
                (unlisted.pulse_length.notna() & ~plasma_rule(unlisted)).sum()
            ),
            "d3drdb_row_without_summary": int(unlisted.pulse_length.isna().sum()),
            "plasma_by_rule_but_not_listed": int(plasma_rule(unlisted).sum()),
            "not_in_d3drdb": int((last - first + 1) - len(rng) - len(listed - set(rng.shot))),
        },
        "plasma_candidates_in_range": len(t),
        "listed_outside_range": int(
            len(status.table) - len(t)
        ),
    }


# ---- drawing ---------------------------------------------------------------------


def fmt(n: int) -> str:
    """4{,}872 for LaTeX."""
    return f"{int(n):,}".replace(",", "{,}")


def draw(rows: list[dict], first: int, last: int) -> Figure:
    n = len(rows)
    height = LEGEND_IN + n * ROW_IN + AXIS_IN
    fig = Figure(figsize=(PAGE_IN, height))
    ax = fig.add_axes(
        (LEFT_IN / PAGE_IN, AXIS_IN / height,
         (PAGE_IN - LEFT_IN - RIGHT_IN) / PAGE_IN, n * ROW_IN / height)
    )
    ax.set_xlim(first, last + 1)
    ax.set_ylim(0, n * ROW_IN)  # y in inches, row 0 at the top
    for k, row in enumerate(rows):
        y0 = (n - 1 - k) * ROW_IN + GAP_IN
        style = {
            ACTIVITY: {"facecolor": row["colour"]},
            QUIET: {"facecolor": QUIET_FILL},
            NOT_SCANNED: {"facecolor": "white", "hatch": HATCH, "hatchcolor": HATCH_INK},
        }
        for x0, x1, code in row["columns"]:
            if code in style:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, STRIP_IN, linewidth=0,
                                       edgecolor="none", **style[code]))
        ax.add_patch(Rectangle((first, y0), last + 1 - first, STRIP_IN, facecolor="none",
                               edgecolor=OUTLINE, linewidth=0.5))
        c = row["counts"]
        head = ax.text(first, y0 + STRIP_IN + 0.03, row["label"], ha="left",
                       va="bottom", color=INK, fontweight="bold", fontsize=FONT_PT)
        ax.annotate(
            f"   {c['activity']:,} with {row['label']}  ·  {c['quiet']:,} without  ·  "
            f"{c['not_scanned']:,} not scanned (shots)",
            xy=(1, 0), xycoords=head, va="bottom", ha="left", color=MUTED,
            fontsize=FONT_PT - 1,
        )
        ax.text(last + 1, y0 + STRIP_IN + 0.03, f"source: {row['source']}", ha="right",
                va="bottom", color=MUTED, fontsize=FONT_PT - 1)
    ticks = [t for t in range(first, last + 1, 10_000) if last - t > 4_000] + [last]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=2, pad=1.5, color=MUTED, labelcolor=INK)
    ax.set_xlabel("shot number", color=INK, labelpad=2)
    colour = rows[0]["colour"]
    handles = [
        Patch(facecolor=colour, label=f"event in a shot (≥ {ACTIVITY_MS} ms)"),
        Patch(facecolor=QUIET_FILL, label="scanned, no event"),
        Patch(facecolor="white", hatch=HATCH, hatchcolor=HATCH_INK, edgecolor=OUTLINE,
              linewidth=0.5, label="not scanned yet"),
    ]
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(LEFT_IN / PAGE_IN - 0.006, 1.0), ncol=3, frameon=False,
               handlelength=1.4, handleheight=0.9, columnspacing=1.4, handletextpad=0.5,
               borderaxespad=0.15, fontsize=FONT_PT - 1, labelcolor=INK)
    return fig


# ---- macros ----------------------------------------------------------------------


def macros(record: dict) -> str:
    provisional = not all(e["source"]["final"] for e in record["events"])
    lines = [
        BLOCK_BEGIN,
        f"% [made {record['made_at']}, git {record['git']}; "
        + "; ".join(f"{e['key']}: {'final' if e['source']['final'] else 'placeholder'} source"
                    for e in record["events"])
        + "]",
        r"\newcommand{\scanProvisional}[1]{"
        + (r"\textcolor{orange!80!black}{#1}}  % placeholder source: counts print orange"
           if provisional else r"#1}  % every row from its final source"),
        rf"\newcommand{{\scanFirstShot}}{{{record['first_shot']}}}"
        rf"\newcommand{{\scanLastShot}}{{{record['last_shot']}}}"
        f"  % last: {record['last_shot_from']}",
        rf"\newcommand{{\scanThreshMs}}{{{ACTIVITY_MS}}}"
        rf"\newcommand{{\scanBinShots}}{{{BIN_SHOTS}}}"
        "  % activity: present time >= this in the assessed window; shot numbers per strip column",
    ]
    for e in record["events"]:
        m, c = e["macro"], e["counts"]
        wrap = (lambda v: v) if e["source"]["final"] else (lambda v: rf"\scanProvisional{{{v}}}")
        lines.append(rf"\newcommand{{\scan{m}source}}{{{e['source']['caption']}}}")
        lines.append(
            "".join(
                rf"\newcommand{{\scan{m}{name}}}{{{wrap(fmt(val))}}}"
                for name, val in (
                    ("active", c["activity"]),
                    ("quiet", c["quiet"]),
                    ("unscanned", c["not_scanned"]),
                    ("blank", c["blank"]),
                )
            )
        )
        lines.append(
            rf"\newcommand{{\scan{m}scanned}}{{{wrap(fmt(c['activity'] + c['quiet']))}}}"
            rf"\newcommand{{\scan{m}plasma}}{{{wrap(fmt(e['breakdown']['plasma_candidates_in_range']))}}}"
            "  % scanned shots; plasma candidates in the range"
        )
    lines.append(BLOCK_END)
    return "\n".join(lines) + "\n"


def write_block(numbers: Path, block: str) -> str:
    """Replace the block in numbers.tex, or append it; returns what was done."""
    text = numbers.read_text()
    pattern = re.compile(
        re.escape(BLOCK_BEGIN.split(" [")[0]) + r".*?" + re.escape(BLOCK_END) + r"\n?",
        re.S,
    )
    if pattern.search(text):
        numbers.write_text(pattern.sub(lambda _: block, text, count=1))
        return "replaced"
    numbers.write_text(text.rstrip("\n") + "\n" + block)
    return "appended"


# ---- main ------------------------------------------------------------------------


def head() -> str:
    run = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=False)
    return run.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--png", type=Path, default=DEFAULT_PNG,
                    help="a preview, kept out of the paper's figures folder")
    ap.add_argument("--numbers", type=Path, default=None,
                    help="numbers.tex whose event coverage block to rewrite")
    ap.add_argument("--ae-shots", type=Path, default=AE_PUBLISHED,
                    help="the all-shot AE run's shots.parquet")
    args = ap.parse_args(argv)
    sources.clear()

    s = summaries()
    statuses = {e.key: e.load(args) for e in EVENTS}
    snap_last = int(s.shot.max())
    pub_last = max((int(st.table.shot.max()) for st in statuses.values() if st.final),
                   default=None)
    last = max(snap_last, pub_last or 0)
    last_from = (
        f"max shot of the d3drdb snapshot {D3DRDB.parent.name}"
        if last == snap_last
        else "max shot of the published table"
    )
    first = FIRST_SHOT

    rows, events = [], []
    for e in EVENTS:
        st = statuses[e.key]
        codes = per_shot(st.table, first, last)
        counts = {name: int((codes == i).sum()) for i, name in enumerate(CLASSES)}
        rows.append({"label": e.label, "colour": e.colour, "counts": counts,
                     "columns": columns(codes, first), "source": st.name})
        events.append({
            "key": e.key, "macro": e.macro, "label": e.label, "colour": e.colour,
            "source": {"final": st.final, "name": st.name, "caption": st.caption,
                       "paths": st.paths, "notes": st.notes},
            "counts": counts,
            "breakdown": breakdown(st, s, first, last),
        })

    record = {
        "schema": "event_coverage",
        "made_by": "scripts/labeler/paper/fig_event_coverage.py",
        "git": head(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "first_shot": first,
        "last_shot": last,
        "last_shot_from": last_from,
        "d3drdb_last_shot": snap_last,
        "published_last_shot": pub_last,
        "activity_threshold_ms": ACTIVITY_MS,
        "bin_shots": BIN_SHOTS,
        "classes": {
            "activity": f"scanned, present time >= {ACTIVITY_MS} ms in the assessed window",
            "quiet": f"scanned, present time < {ACTIVITY_MS} ms",
            "not_scanned": "plasma candidate with no label, or a run status other than "
                           "labeled and not in no_usable_record",
            "blank": "not a plasma candidate, or a no_usable_record status (drawn unfilled)",
        },
        "no_usable_record": sorted(NO_DATA),
        "rendering": f"vector rectangles, {BIN_SHOTS} shot numbers a column, each column "
                     "filled full height in the first class (activity, quiet, not_scanned) "
                     "any of its shots has, blank (unfilled) if none; header counts per shot",
        "events": events,
        "inputs_sha256": dict(sorted(sources.items())),
    }
    record["macros"] = macros(record)

    with matplotlib.rc_context({**STYLE, "hatch.linewidth": 0.45}):
        fig = draw(rows, first, last)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out)
        if args.png:
            args.png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.png, dpi=300)
    sidecar = args.out.with_suffix(".json")
    sidecar.write_text(json.dumps(record, indent=1) + "\n")

    print(f"range {first}-{last} ({last_from})")
    for ev in events:
        kind = "FINAL" if ev["source"]["final"] else "PLACEHOLDER"
        print(f"{ev['key']}: source {kind}: {ev['source']['name']}")
        for p in ev["source"]["paths"]:
            print(f"    {p}")
        for note in ev["source"]["notes"]:
            print(f"    note: {note}")
        print("    " + ", ".join(f"{k} {v:,}" for k, v in ev["counts"].items())
              + f" (activity >= {ACTIVITY_MS} ms)")
    print(f"wrote {args.out}, {sidecar}" + (f", {args.png}" if args.png else ""))
    if args.numbers:
        numbers = args.numbers if args.numbers.is_absolute() else REPO / args.numbers
        print(f"{write_block(numbers, record['macros'])} the macro block in {numbers}")
    else:
        print("macros (pass --numbers to write them):\n" + record["macros"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
