"""pseudo-v3's MHD markers: one line at a time, below 80 kHz, and their gate.

pseudo-v2's rules (`mhdlines`) judge a whole 8-connected region, so an MHD line
that touches other lit structure, or drifts a little, escapes them, and an
owner-absent frame is 0 at every bin although the owner, labelling on the
80-250 kHz view, said nothing about what lies below 80 kHz. pseudo-v3
(`python -m labeler.ae.seg.pseudo --version v3`) keeps pseudo-v2's grid, lit
pixels and bright rule, and changes three things:
- below 80 kHz an owner-absent frame is IGNORED, except the pixels these
  markers take, which stay 0 (hard negatives);
- the steady test runs per line, not per region;
- the catalog's own NTM intervals, and the harmonics of a marked line, mark MHD
  too. The absent-run rule is dropped: an absent frame says nothing below 80 kHz.

**Lines** (`segments`). In each column of the store's level-8 grid a run of lit
pixels (TokEye's two-chord pixels, as pseudo-v2 pools them) is split at the
local minima of its brightness (the brightest cross-power row, smoothed 1-2-1
over frequency). Each piece is a segment; its brightest bin is its ridge. Two
lines touching in a column are two segments, so one can be taken and not the
other.

**Markers** (`mhd_lines`), all on the pixels below `below_khz` (80 kHz) only:
- **steady** (`steady_lines`): a segment whose ridge lies on a held line. A band
  of w = floor(drift_khz / dy) + 1 bins, wholly below `guard_khz`, is held
  where it holds a ridge in every column of a run of at least `steady_ms`, gaps
  of up to `gap_ms` bridged;
- **ntm**: a segment whose ridge lies at `f0_min_khz` to `guard_khz` in a column
  inside one of the catalog's NTM intervals (`ntm_intervals`: category 1 of
  `$LABELER_LABEL_TABLES/neoclassical_tearing_mode/format/`, read-only);
- **comb** (`comb`): a segment whose ridge lies, in the same column, at m x f0
  (m = 2 .. `harmonics`, within min(ceil(m / 2), floor(f0 / (4 dy))) bins) of a
  steady or ntm ridge f0 >= `f0_min_khz`: an MHD line's harmonics.

**The values** (`Markers`) are fixed, not picked: `drift_khz` and `steady_ms`
are pseudo-v2's chosen pair (2 kHz over 100 ms); `gap_ms` 20 ms bridges the
breaks TokEye leaves in a line; `guard_khz` 30 kHz keeps the steady test off
the slowly drifting AE lines at 40-80 kHz (176041, 176053), which a 60 kHz
guard with bridging took. They were set by looking at the three gate shots
below, all SegNet v2 training shots; SegNet v2's test shots were never looked
at for them. The gate is therefore a check that the markers do what the owner
said of those shots, not an independent test.

**The gate** (`GATE`, `gate_rows`), before any mask is written: on the rule
shots' masks, each box's share of lit pixels with the given mask value must
meet its bound, and the markers, run without the 80 kHz cut, may take at most
`MAX_LOSS` (5 %) of the lit present-frame pixels at 80-250 kHz, where the
owner's labels are truth (`costs`). A gate shot that is not a rule shot fails
its rows: the gate never reads a val or test shot.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ...config import Paths, atomic_path
from ...events.catalog.states import ABSENT, PRESENT
from ..xpower.data import BAND_KHZ, band_slice
from .mhdlines import MAX_LOSS, RULE_BELOW_KHZ, ShotLines

#: The catalog's NTM table (the owner's label tables; read-only).
NTM_TABLE = Path(
    "neoclassical_tearing_mode",
    "format",
    "neoclassical_tearing_mode_format_2026_v1.csv",
)
NTM_PRESENT = 1  # the table's category for a tearing mode present
COST_BAND_KHZ = BAND_KHZ  # where the owner's present frames are AE
REPORT_BANDS_KHZ = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0))
KINDS = ("steady", "ntm", "comb")


@dataclass(frozen=True)
class Markers:
    """pseudo-v3's markers and its bright rule (`bright_u8`, set from the rule
    shots' first-pass masks as pseudo-v2 sets it; no bright rule when None)."""

    drift_khz: float = 2.0
    steady_ms: float = 100.0
    gap_ms: float = 20.0
    guard_khz: float = 30.0
    harmonics: int = 5
    f0_min_khz: float = 4.0
    below_khz: float = RULE_BELOW_KHZ
    bright_u8: int | None = None


@dataclass(frozen=True)
class GateBox:
    """A gate row: of the lit pixels of `shot` in `t_ms` x `khz` whose column the
    owner calls `state`, the share whose mask value is `value` must be at most
    (`most`) or at least `bound`."""

    shot: int
    says: str  # what the owner said, or what was seen, of the box
    t_ms: tuple[float, float]
    khz: tuple[float, float]
    state: int
    value: int
    most: bool
    bound: float


#: SegNet v2 training shots only; 170720 and 170790, test shots, are never used.
GATE = (
    GateBox(
        176035,
        "the owner: the 3-13 kHz line is not AE",
        (600.0, 1600.0),
        (3.0, 13.0),
        PRESENT,
        1,
        True,
        0.05,
    ),
    GateBox(
        176041,
        "the owner: the 40-80 kHz lines at 0.4-1.5 s (absent at 80-250 kHz) are AE,"
        " so not hard negatives",
        (424.0, 1504.0),
        (40.0, 80.0),
        ABSENT,
        0,
        True,
        0.05,
    ),
    GateBox(
        176041,
        "the same AE lines where the owner calls AE present (0.2-0.42 s) stay AE",
        (200.0, 424.0),
        (40.0, 80.0),
        PRESENT,
        1,
        False,
        0.5,
    ),
    GateBox(
        176053,
        "the owner's early AE, 50-80 kHz at 0.2-1.2 s, stays AE",
        (205.0, 1195.0),
        (50.0, 80.0),
        PRESENT,
        1,
        False,
        0.8,
    ),
    GateBox(
        176053,
        "seen on the page: the steady 12-22 kHz line at 4.1-4.5 s is not AE",
        (4100.0, 4500.0),
        (12.0, 22.0),
        PRESENT,
        1,
        True,
        0.05,
    ),
)


def segments(lit, bright) -> tuple[np.ndarray, np.ndarray]:
    """`(ids, ridge)`, both `(n_y, n)`: each lit pixel's segment id (0 where
    unlit; unique over the whole array) and each segment's ridge, its brightest
    pixel (ties to the lower bin). A segment is a run of lit pixels in one column
    cut before each local minimum of the 1-2-1 smoothed brightness that has lit
    neighbours above and below it."""
    lit = np.asarray(lit, dtype=bool)
    n_y, n = lit.shape
    b = np.asarray(bright, dtype=np.float32)
    s = b.copy()
    s[1:-1] = (b[:-2] + 2 * b[1:-1] + b[2:]) / 4
    lower = np.zeros_like(lit)
    lower[1:] = lit[:-1]
    upper = np.zeros_like(lit)
    upper[:-1] = lit[1:]
    s_lower = np.full_like(s, np.inf)
    s_lower[1:] = s[:-1]
    s_upper = np.full_like(s, np.inf)
    s_upper[:-1] = s[1:]
    minimum = lit & lower & upper & (s < s_lower) & (s <= s_upper)
    start = lit & (~lower | minimum)
    offset = np.arange(n, dtype=np.int64)[None, :] * (n_y + 1)
    ids = np.where(lit, np.cumsum(start, axis=0, dtype=np.int64) + offset, 0)
    rows, cols = np.nonzero(lit)
    keys = ids[rows, cols]
    order = np.lexsort((rows, -s[rows, cols], keys))
    first = np.r_[True, np.diff(keys[order]) != 0] if keys.size else keys.astype(bool)
    ridge = np.zeros_like(lit)
    ridge[rows[order][first], cols[order][first]] = True
    return ids, ridge


def _edges(a: np.ndarray, value: bool) -> tuple[np.ndarray, ...]:
    """Rows, starts and stops of the runs of `value` along axis 1."""
    d = np.diff(np.pad((a == value).astype(np.int8), ((0, 0), (1, 1))), axis=1)
    rows, starts = np.nonzero(d == 1)
    _, stops = np.nonzero(d == -1)
    return rows, starts, stops


def _fill(shape, rows, starts, stops) -> np.ndarray:
    """True over columns starts .. stops - 1 of each row."""
    delta = np.zeros((shape[0], shape[1] + 1), dtype=np.int32)
    np.add.at(delta, (rows, starts), 1)
    np.add.at(delta, (rows, stops), -1)
    return np.cumsum(delta, axis=1)[:, : shape[1]] > 0


def held_bands(points, w: int, k: int, gap: int = 0) -> np.ndarray:
    """`(n_y - w + 1, n)`: band r0 (bins r0 .. r0 + w - 1) is held in a column
    when it lies in a run of at least `k` columns, each holding a point in the
    band, after the band's gaps of at most `gap` columns between two such
    columns are bridged."""
    points = np.asarray(points, dtype=bool)
    n_y, n = points.shape
    if w > n_y:
        return np.zeros((0, n), dtype=bool)
    total = np.zeros((n_y + 1, n), dtype=np.int32)
    total[1:] = np.cumsum(points, axis=0, dtype=np.int32)
    near = (total[w:] - total[:-w]) > 0
    if gap > 0:
        rows, starts, stops = _edges(near, False)
        inner = ((stops - starts) <= gap) & (starts > 0) & (stops < n)
        near = near | _fill(near.shape, rows[inner], starts[inner], stops[inner])
    rows, starts, stops = _edges(near, True)
    long = (stops - starts) >= k
    return _fill(near.shape, rows[long], starts[long], stops[long])


def _in_held(points, held, w: int) -> np.ndarray:
    """The points lying in some held band (bin r is in bands r - w + 1 .. r)."""
    n_y = points.shape[0]
    bands = held.shape[0]
    total = np.zeros((bands + 1, held.shape[1]), dtype=np.int32)
    total[1:] = np.cumsum(held, axis=0, dtype=np.int32)
    r = np.arange(n_y)
    hi = np.minimum(bands, r + 1)
    lo = np.minimum(np.maximum(0, r - w + 1), hi)
    return points & ((total[hi] - total[lo]) > 0)


def _freqs(lines: ShotLines) -> np.ndarray:
    return lines.y0_khz + np.arange(lines.lit.shape[0]) * lines.dy_khz


def _taken(ids, chosen, lit, low) -> np.ndarray:
    """The lit pixels below the cut of the segments holding a `chosen` pixel."""
    picked = np.unique(ids[chosen])
    return np.isin(ids, picked[picked > 0]) & lit & low[:, None]


def steady_lines(
    lines: ShotLines, markers: Markers, ids, ridge, below_khz: float
) -> tuple[np.ndarray, np.ndarray]:
    """`(taken, anchors)`: the steady test's pixels (lit, below `below_khz`) and
    the ridge points on its held lines, the comb's fundamentals."""
    f = _freqs(lines)
    w = math.floor(markers.drift_khz / lines.dy_khz + 1e-9) + 1
    k = max(1, math.ceil(markers.steady_ms / lines.dt_ms - 1e-9))
    gap = round(markers.gap_ms / lines.dt_ms)
    low = f < below_khz
    held = held_bands(ridge & low[:, None], w, k, gap)
    tops = f[np.arange(held.shape[0]) + w - 1]
    held &= (tops < markers.guard_khz)[:, None]
    anchors = _in_held(ridge & low[:, None], held, w)
    return _taken(ids, anchors, lines.lit, low), anchors


def comb(
    lines: ShotLines, markers: Markers, ids, ridge, fundamentals, below_khz: float
) -> np.ndarray:
    """The segments whose ridge is a harmonic, m = 2 .. `harmonics`, of a
    fundamental ridge point at or above `f0_min_khz` in the same column."""
    lit = lines.lit
    n_y = lit.shape[0]
    f = _freqs(lines)
    low = f < below_khz
    rows, cols = np.nonzero(fundamentals & (f >= markers.f0_min_khz)[:, None])
    near = np.zeros_like(lit)
    for m in range(2, markers.harmonics + 1):
        target = np.round((m * f[rows] - lines.y0_khz) / lines.dy_khz).astype(int)
        tol = np.minimum(
            math.ceil(m / 2), np.floor(f[rows] / (4 * lines.dy_khz))
        ).astype(int)
        for t in range(int(tol.max(initial=0)) + 1):
            for sign in (-1, 1):
                at = target + sign * t
                ok = (tol >= t) & (at >= 0) & (at < n_y)
                near[at[ok], cols[ok]] = True
    harmonic = ridge & near & low[:, None] & ~fundamentals
    return _taken(ids, harmonic, lit, low)


def ntm_intervals(paths: Paths) -> tuple[dict[int, list[tuple[float, float]]], str]:
    """The catalog's NTM intervals by shot (ms, category NTM_PRESENT rows only)
    and the table's sha256. FileNotFoundError without the table."""
    file = Path(paths.label_tables) / NTM_TABLE
    data = file.read_bytes()
    out: dict[int, list[tuple[float, float]]] = {}
    for row in csv.DictReader(data.decode().splitlines()):
        if int(float(row["category"])) == NTM_PRESENT:
            span = (float(row["t_start"]), float(row["t_end"]))
            out.setdefault(int(row["shot"]), []).append(span)
    return out, hashlib.sha256(data).hexdigest()


def ntm_columns(spans, t0_ms: float, dt_ms: float, n: int) -> np.ndarray:
    """`(n,)`: the columns whose centre lies in one of `spans` ([start, end) ms)."""
    centres = t0_ms + (np.arange(n) + 0.5) * dt_ms
    out = np.zeros(n, dtype=bool)
    for a, b in spans:
        out |= (centres >= a) & (centres < b)
    return out


def mhd_lines(
    lines: ShotLines, markers: Markers, ntm, *, below_khz: float | None = None
) -> dict[str, np.ndarray]:
    """Each marker's pixels and their union ("mhd"), `(n_y, n)` bool, lit and
    below `below_khz` (`markers.below_khz` when None; `math.inf`, no cut, for
    the cost). `ntm` is `(n,)` bool, the columns inside an NTM interval."""
    below_khz = markers.below_khz if below_khz is None else below_khz
    f = _freqs(lines)
    ids, ridge = segments(lines.lit, lines.bright)
    held, anchors = steady_lines(lines, markers, ids, ridge, below_khz)
    band = (f >= markers.f0_min_khz) & (f < min(markers.guard_khz, below_khz))
    ntm_ridge = ridge & np.asarray(ntm, dtype=bool)[None, :] & band[:, None]
    tearing = _taken(ids, ntm_ridge, lines.lit, f < below_khz)
    harmonic = comb(lines, markers, ids, ridge, anchors | ntm_ridge, below_khz)
    return {
        "steady": held,
        "ntm": tearing,
        "comb": harmonic,
        "mhd": held | tearing | harmonic,
    }


def _rows_in(lines: ShotLines, khz) -> np.ndarray:
    """`(n_y,)`: the bins whose frequency lies in [lo, hi)."""
    f = _freqs(lines)
    return (f >= khz[0]) & (f < khz[1])


def costs(lines: list[ShotLines], markers: Markers, ntm: dict) -> dict:
    """The markers over the rule shots: at COST_BAND_KHZ, the lit present-frame
    pixels and what the markers, run without the cut, take of them (each and
    all); per REPORT_BANDS_KHZ band, the lit present and absent pixels and what
    each marker takes of them as pseudo-v3 applies it. `ntm` maps a shot to its
    (n,) NTM columns."""
    area = taken = 0
    by_kind = dict.fromkeys(KINDS, 0)
    bands = [
        {
            "lo_khz": lo,
            "hi_khz": hi,
            **{f"lit_{s}_px": 0 for s in ("present", "absent")},
            **{
                f"{k}_{s}_px": 0 for k in (*KINDS, "mhd") for s in ("present", "absent")
            },
        }
        for lo, hi in REPORT_BANDS_KHZ
    ]
    for line in lines:
        present = line.state == PRESENT
        absent = line.state == ABSENT
        cols = ntm[line.shot]
        rows = band_slice(line.y0_khz, line.dy_khz, line.lit.shape[0], COST_BAND_KHZ)
        uncut = mhd_lines(line, markers, cols, below_khz=math.inf)
        lit = np.zeros_like(line.lit)
        lit[rows] = line.lit[rows]
        lit &= present[None, :]
        area += int(lit.sum())
        taken += int((uncut["mhd"] & lit).sum())
        for kind in KINDS:
            by_kind[kind] += int((uncut[kind] & lit).sum())
        cut = mhd_lines(line, markers, cols)
        for band in bands:
            inside = _rows_in(line, (band["lo_khz"], band["hi_khz"]))
            for state, where in (("present", present), ("absent", absent)):
                box = inside[:, None] & where[None, :]
                band[f"lit_{state}_px"] += int((line.lit & box).sum())
                for kind in (*KINDS, "mhd"):
                    band[f"{kind}_{state}_px"] += int((cut[kind] & box).sum())
    return {
        "rule_shots": [int(line.shot) for line in lines],
        "max_loss": MAX_LOSS,
        "cost_band_khz": list(COST_BAND_KHZ),
        "lit_present_px": area,
        "cost_px": taken,
        "cost_px_by_marker": by_kind,
        "within": area > 0 and taken <= MAX_LOSS * area,
        "bands": bands,
    }


def gate_rows(masks: dict, lines: dict, rule_shots) -> list[dict]:
    """One row per GATE box: its lit pixels, the share of them whose mask value
    is the box's, and whether that meets the bound. `masks` and `lines` map a
    shot to its mask and ShotLines. A box whose shot is not a rule shot, or was
    not read, fails with `share` None."""
    rows = []
    for box in GATE:
        row = {**asdict(box), "t_ms": list(box.t_ms), "khz": list(box.khz)}
        line, mask = lines.get(box.shot), masks.get(box.shot)
        if box.shot not in rule_shots or line is None or mask is None:
            why = "not a rule shot" if box.shot not in rule_shots else "not read"
            rows.append(
                {**row, "lit_px": 0, "share": None, "passed": False, "why": why}
            )
            continue
        centres = line.t0_ms + (np.arange(line.lit.shape[1]) + 0.5) * line.dt_ms
        cols = (centres >= box.t_ms[0]) & (centres < box.t_ms[1])
        cols &= line.state == box.state
        lit = line.lit & _rows_in(line, box.khz)[:, None] & cols[None, :]
        n = int(lit.sum())
        share = float((np.asarray(mask)[lit] == box.value).mean()) if n else None
        ok = share is not None and (
            share <= box.bound + 1e-12 if box.most else share >= box.bound - 1e-12
        )
        rows.append({**row, "lit_px": n, "share": share, "passed": bool(ok), "why": ""})
    return rows


def _table(head, rows) -> list[str]:
    return [
        "| " + " | ".join(map(str, head)) + " |",
        "|" + "---|" * len(head),
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ]


def rules_md(markers: Markers, report: dict) -> str:
    """rules.md: the values, the gate and its verdict, the cost, the bands."""
    gate, passed = report["gate"], report["gate_passed"]
    area, cost = report["lit_present_px"], report["cost_px"]
    band = "{:g}-{:g} kHz".format(*report["cost_band_khz"])
    share = f"{report['max_loss']:.0%}"
    names = {1: "AE (1)", 0: "hard negative (0)"}
    states = {PRESENT: "present", ABSENT: "absent"}
    gate_table = [
        (
            box["shot"],
            box["says"],
            "{:g}-{:g} ms".format(*box["t_ms"]),
            "{:g}-{:g} kHz".format(*box["khz"]),
            states.get(box["state"], box["state"]),
            box["lit_px"],
            f"{names[box['value']]} {'<=' if box['most'] else '>='} {box['bound']:.0%}",
            "-" if box["share"] is None else f"{box['share']:.3f}",
            ("pass" if box["passed"] else f"FAIL {box['why']}".strip()),
        )
        for box in gate
    ]
    budget = f"{cost} of {area} px ({cost / area:.2%})" if area else f"{cost} of 0 px"
    kinds = report["cost_px_by_marker"]
    band_rows = [
        (
            f"{b['lo_khz']:g}-{b['hi_khz']:g}",
            b["lit_present_px"],
            b["mhd_present_px"],
            *(b[f"{k}_present_px"] for k in KINDS),
            b["lit_absent_px"],
            b["mhd_absent_px"],
        )
        for b in report["bands"]
    ]
    bright = report.get("bright_background_share")
    values = asdict(markers)
    out = [
        "# pseudo-v3's MHD markers",
        "",
        (
            "The values are fixed (`labeler.ae.seg.markers.Markers`), not picked. "
            "They were set by looking at the gate shots below, all SegNet v2 "
            "training shots; SegNet v2's test shots, 170720 and 170790 among them, "
            "were never used to choose or gate anything. The gate checks that the "
            "markers do what the owner said of those shots; it is not an "
            "independent test."
        ),
        "",
        *_table(("value", "setting"), [(f"`{k}`", v) for k, v in values.items()]),
        "",
        f"**The gate: {'PASS' if passed else 'FAIL'}.**"
        + ("" if passed else " No mask was written."),
        "",
        *_table(
            (
                "shot",
                "what holds",
                "time",
                "band",
                "owner",
                "lit px",
                "criterion",
                "share",
                "verdict",
            ),
            gate_table,
        ),
        "",
        (
            f"The budget: run without the {markers.below_khz:g} kHz cut, the "
            f"markers take {budget} of the lit present-frame pixels at {band} "
            f"over the {len(report['rule_shots'])} rule shots, where the owner's "
            f"labels are truth; at most {share} may go: "
            f"{'within' if report['within'] else 'OVER'} (steady {kinds['steady']}, "
            f"ntm {kinds['ntm']}, comb {kinds['comb']} px)."
        ),
        "",
        (
            "`bright_u8` is the median brightest cross-power row at the AE pixels "
            "of the rule shots' first-pass masks, as pseudo-v2 sets it"
            + (
                f"; it IGNOREs {bright:.1%} of those masks' present-frame background."
                if bright is not None
                else "."
            )
        ),
        "",
        "## By band: what the markers take below 80 kHz (px, rule shots)",
        "",
        *_table(
            (
                "band (kHz)",
                "lit, present",
                "taken, present",
                *(f"{k}, present" for k in KINDS),
                "lit, absent",
                "taken, absent",
            ),
            band_rows,
        ),
        "",
        (
            "In a present frame a taken pixel is IGNORED; in an absent frame it is "
            "0 (a hard negative), and every other pixel below 80 kHz is IGNORED."
        ),
    ]
    return "\n".join(out) + "\n"


def write_rules(out: Path, markers: Markers, report: dict) -> None:
    """out / "rules.json": {"rules": asdict(markers), **report}; out / "rules.md"."""
    out = Path(out)
    record = {"rules": asdict(markers), **report}
    with atomic_path(out / "rules.json") as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    with atomic_path(out / "rules.md") as tmp:
        tmp.write_text(rules_md(markers, report))
