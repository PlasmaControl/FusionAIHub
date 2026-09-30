"""pseudo-v2's MHD-line rules, and how their values are picked (spec §2.4).

pseudo-v2 (`python -m labeler.ae.seg.pseudo --version v2`) reads TokEye's
whole-shot masks over 0-250 kHz. Below 80 kHz TokEye lights every coherent line,
so in a frame the owner calls AE present an MHD mode's lines are lit beside the
mode. Such a line is neither AE nor background there, and pseudo-v2 IGNORES it,
never 0 or 1. The two MHD-line rules act below `RULE_BELOW_KHZ` (80 kHz) only, on
the regions lit there (`below`; 8-connected, on the store's level-8 grid), so a
region that crosses 80 kHz is cut there and only its part below can be taken.
Above 80 kHz the owner labelled on the 80+ kHz view, so a present frame's lit
pixels stay AE, as in pseudo-v1: the edge marks where the owner's labels speak,
not a physics floor. A lit pixel of a present frame is MHD-like (`mhd_like`) when
its region below 80 kHz is taken by either rule:
- **absent-run** (`absent_run`): at least `run_ms` of the region's columns lie in
  frames the owner calls absent, as a tearing line and its harmonics run on where
  the owner saw no AE;
- **steady** (`steady`): the region's column-centroid frequency stays below
  `guard_khz` (`MHD_GUARD_KHZ`, the edge `mhd_frames` uses), within `drift_khz`,
  for at least `steady_ms`. AE chirps and sweeps drift; an MHD line holds.

A third rule, **bright**, IGNORES a present pixel TokEye left unlit, at any
frequency, when its brightest cross-power row (`brightest`) reaches `bright_u8`:
TokEye may have missed the mode there, and 0 would teach the network that there
is none.

**Picking the values** (`pick`; D22), on SegNet v2's training shots only, never
its val or test shots. The costs are measured at 80-250 kHz (`COST_BAND_KHZ`),
where the owner's labels are truth: a candidate's cost is the lit pixels of the
owner's present frames there that its test takes, run on the regions lit there
(the rules themselves act below 80 kHz only), and at most `MAX_LOSS` (5 %) of
them may go:
- `run_ms`: the smallest of `RUN_MS` whose absent-run test, on the regions lit
  inside that band, is within the budget; else the largest;
- (`drift_khz`, `steady_ms`): of the `PAIRS` whose steady test, run on those
  present pixels alone and without the guard, is within the budget, the one that
  takes the most below 80 kHz as `mhd_like` applies it (its take), ties going to
  the more inclusive; else the least inclusive, 1 kHz over 200 ms;
- `bright_u8` (`bright_level`; D23): the median (`BRIGHT_PERCENTILE`) of the
  brightest row at the AE pixels of the rule shots' masks made with the two rules
  above, so an unlit pixel is IGNORED only when it is as bright as a typical AE
  pixel. `bright_background_share` is the share of those masks' present-frame
  background that it IGNOREs.

A fallback is over the budget, and `fallbacks` says so: the command prints it as
a WARNING and rules.md states it. `write_rules` records the values, every
candidate's cost and take and, per 20 kHz band, the lit present-frame pixels and
what the chosen rules take (`rules.json`, `rules.md`).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from ...config import atomic_path
from ...events.catalog.states import ABSENT, PRESENT
from ..xpower.data import BAND_KHZ, FULL_BAND_KHZ, band_slice

MHD_GUARD_KHZ = 60.0  # mhd_frames' edge (TokEye bins 0-121)
RUN_MS = (20.0, 50.0, 100.0, 200.0, 500.0)
DRIFTS_KHZ = (1.0, 2.0, 3.0, 5.0)
STEADY_MS = (50.0, 100.0, 200.0)
#: The steady pairs, most inclusive first: larger drift, then shorter span.
PAIRS = tuple((d, s) for d in sorted(DRIFTS_KHZ, reverse=True) for s in STEADY_MS)
MAX_LOSS = 0.05  # of the lit present-frame area at COST_BAND_KHZ
BRIGHT_PERCENTILE = 50  # the median lit AE pixel
COST_BAND_KHZ = BAND_KHZ  # where the owner's present frames are AE
#: The MHD-line rules act below it; above it the owner's labels speak.
RULE_BELOW_KHZ = COST_BAND_KHZ[0]
REPORT_BAND_KHZ = 20.0  # rules.md's bands: 0-20, ..., 220-240, 240-250
TOKEYE_VERSION = "v3"  # masks-full
EIGHT = np.ones((3, 3), dtype=bool)
#: The report's bands, (lo, hi) kHz: REPORT_BAND_KHZ wide over FULL_BAND_KHZ.
_BANDS = tuple(
    (lo, min(lo + REPORT_BAND_KHZ, FULL_BAND_KHZ[1]))
    for lo in np.arange(*FULL_BAND_KHZ, REPORT_BAND_KHZ).tolist()
)


@dataclass(frozen=True)
class Rules:
    """pseudo-v2's rules: absent-run (`run_ms`), steady (`drift_khz` over
    `steady_ms`, below `guard_khz`; no guard when None) and bright (`bright_u8`;
    no bright rule when None)."""

    run_ms: float
    drift_khz: float
    steady_ms: float
    guard_khz: float | None = MHD_GUARD_KHZ
    bright_u8: int | None = None


@dataclass(frozen=True)
class ShotLines:
    """One shot on the store's level-8 grid: `lit` (n_y, n) bool, TokEye's
    two-chord pixels pooled as pseudo-v1 pools them; `state` (n,) the owner's
    state per column, OUTSIDE where masks-full does not cover the column;
    `bright` (n_y, n) uint8, the brightest of the three cross-power rows."""

    shot: int
    t0_ms: float
    dt_ms: float
    y0_khz: float
    dy_khz: float
    lit: np.ndarray
    state: np.ndarray
    bright: np.ndarray


def below(lit, y0_khz, dy_khz, khz=RULE_BELOW_KHZ) -> np.ndarray:
    """`lit` with its rows from `khz` up cleared: the rows kept are those whose
    frequency y0 + row * dy lies below it, under band_slice's edge at `khz`."""
    out = np.array(lit, dtype=bool)
    out[max(0, math.ceil((khz - y0_khz) / dy_khz - 1e-9)) :] = False
    return out


def absent_run(lit, absent, dt_ms, run_ms) -> np.ndarray:
    """The pixels of every 8-connected region of `lit` whose distinct columns
    inside `absent` ((n,) bool) number at least run_ms / dt_ms (count * dt_ms
    >= run_ms - 1e-9). The whole region is taken, its present part too."""
    regions, count = ndimage.label(np.asarray(lit, dtype=bool), structure=EIGHT)
    rows, cols = np.nonzero(regions)
    inside = np.asarray(absent, dtype=bool)[cols]
    rows, cols = rows[inside], cols[inside]
    width = max(regions.shape[1], 1)
    # One key per (region, absent column) pair the region occupies.
    keys = np.unique(regions[rows, cols].astype(np.int64) * width + cols)
    columns = np.bincount(keys // width, minlength=count + 1)
    taken = columns * dt_ms >= run_ms - 1e-9
    taken[0] = False
    return taken[regions]


def steady(
    lit, y0_khz, dy_khz, dt_ms, drift_khz, steady_ms, guard_khz=MHD_GUARD_KHZ
) -> np.ndarray:
    """The pixels of every 8-connected region of `lit` holding k =
    ceil(steady_ms / dt_ms - 1e-9) consecutive occupied columns whose
    column-centroid frequencies (y0 + row * dy, averaged over the region's
    pixels in each column) span at most drift_khz + 1e-9, all below guard_khz
    (no guard when None). The whole region is taken. One pass per region: the
    centroids once, then a sliding window over each run of occupied columns."""
    lit = np.asarray(lit, dtype=bool)
    k = max(1, math.ceil(steady_ms / dt_ms - 1e-9))
    regions, _ = ndimage.label(lit, structure=EIGHT)
    freqs = y0_khz + np.arange(lit.shape[0]) * dy_khz
    out = np.zeros(lit.shape, dtype=bool)
    for label, (rows, cols) in enumerate(ndimage.find_objects(regions), start=1):
        if cols.stop - cols.start < k:
            continue
        region = regions[rows, cols] == label
        # A region is 8-connected, so it occupies every column of its box: its
        # occupied columns are one run.
        centroid = (region * freqs[rows, None]).sum(axis=0) / region.sum(axis=0)
        window = np.lib.stride_tricks.sliding_window_view(centroid, k)
        top = window.max(axis=1)
        held = top - window.min(axis=1) <= drift_khz + 1e-9
        if guard_khz is not None:
            held &= top < guard_khz
        if held.any():
            out[rows, cols] |= region
    return out


def _takes(lines: ShotLines, rules: Rules) -> tuple[np.ndarray, np.ndarray]:
    """What the absent-run and the steady rule take, as `mhd_like` applies them:
    on the regions of the lit pixels below RULE_BELOW_KHZ."""
    low = below(lines.lit, lines.y0_khz, lines.dy_khz)
    run = absent_run(low, lines.state == ABSENT, lines.dt_ms, rules.run_ms)
    held = steady(
        low,
        lines.y0_khz,
        lines.dy_khz,
        lines.dt_ms,
        rules.drift_khz,
        rules.steady_ms,
        rules.guard_khz,
    )
    return run, held


def mhd_like(lines: ShotLines, rules: Rules) -> np.ndarray:
    """absent_run(low, state == ABSENT, dt, run_ms) | steady(low, ..., guard_khz),
    where low = below(lit, y0, dy): the rules act on the regions lit below
    RULE_BELOW_KHZ, so a region crossing it is cut there."""
    run, held = _takes(lines, rules)
    return run | held


def brightest(values) -> np.ndarray:
    """(3, n_y, n) -> (n_y, n): the maximum over the rows."""
    return np.asarray(values).max(axis=0)


def _band_rows(lines: ShotLines) -> list[slice]:
    """Each report band's bins: those whose centre lies in [lo, hi), and in the
    last band [lo, hi], so the top bin (250 kHz) is counted once."""
    n_y = lines.lit.shape[0]

    def first(khz: float) -> int:  # the first bin at or above khz
        return min(n_y, max(0, math.ceil((khz - lines.y0_khz) / lines.dy_khz - 1e-9)))

    top = math.floor((_BANDS[-1][1] - lines.y0_khz) / lines.dy_khz + 1e-9) + 1
    stops = [first(hi) for _, hi in _BANDS[:-1]] + [min(n_y, max(0, top))]
    return [slice(first(lo), stop) for (lo, _), stop in zip(_BANDS, stops)]


def _band_report(lines: list[ShotLines], rules: Rules) -> list[dict]:
    """Per report band, over the shots' present columns: the lit pixels, and those
    the chosen rules take as `mhd_like` applies them (below RULE_BELOW_KHZ)."""
    bands = [
        {
            "lo_khz": lo,
            "hi_khz": hi,
            "lit_present_px": 0,
            "absent_run_px": 0,
            "steady_px": 0,
        }
        for lo, hi in _BANDS
    ]
    for line in lines:
        present = line.state == PRESENT
        run, held = _takes(line, rules)
        per_bin = {
            "lit_present_px": line.lit[:, present].sum(axis=1),
            "absent_run_px": run[:, present].sum(axis=1),
            "steady_px": held[:, present].sum(axis=1),
        }
        for band, rows in zip(bands, _band_rows(line), strict=True):
            for key, counts in per_bin.items():
                band[key] += int(counts[rows].sum())
    return bands


def pick(lines: list[ShotLines]) -> tuple[Rules, dict]:
    """The rules for these shots (spec §2.4), and the report rules.json records.

    The costs are counted in present columns at COST_BAND_KHZ (band_slice):
    area = the lit pixels there. run_ms is the first of RUN_MS whose
    absent_run(lit_band, absent, ...) pixels there are <= MAX_LOSS * area, where
    lit_band is lit with its rows outside the band cleared, so its regions form
    inside it; else RUN_MS[-1]. A pair is within the budget when its steady(lit &
    present & band, ..., guard_khz=None) pixels are <= MAX_LOSS * area. Its take
    is steady(below(lit), ..., guard_khz=MHD_GUARD_KHZ) counted in present
    columns, as mhd_like applies it. The pair is the within-budget one with the
    largest take, ties in PAIRS order; else (min(DRIFTS_KHZ), max(STEADY_MS)).
    guard_khz is MHD_GUARD_KHZ, bright_u8 None (`bright_level` sets it).
    ValueError "... no present pixel ..." when the area is 0.

    Report keys: rule_shots (list of int), max_loss, cost_band_khz (list),
    rule_below_khz, lit_present_px, absent_run_px {f"{r:g}": px}, steady_px and
    steady_take_px {f"{d:g}-{s:g}": px} (all 12 pairs), run_ms_within,
    steady_within (False when the fallback was taken), and bands: one dict per
    REPORT_BAND_KHZ band of the page's bins (lo_khz, hi_khz, lit_present_px,
    absent_run_px, steady_px), the chosen rules as mhd_like applies them, counted
    in present columns; 13 bands, the last 240-250 with 250 kHz inside it."""
    area = 0
    run_px = dict.fromkeys(RUN_MS, 0)
    steady_px = dict.fromkeys(PAIRS, 0)
    take_px = dict.fromkeys(PAIRS, 0)
    for line in lines:
        present = line.state == PRESENT
        absent = line.state == ABSENT
        rows = band_slice(line.y0_khz, line.dy_khz, line.lit.shape[0], COST_BAND_KHZ)
        band = np.zeros(line.lit.shape, dtype=bool)
        band[rows] = line.lit[rows]  # lit_band: its regions form inside the band
        lit = band & present
        area += int(lit.sum())
        for run_ms in RUN_MS:
            taken = absent_run(band, absent, line.dt_ms, run_ms)
            run_px[run_ms] += int(taken[:, present].sum())
        low = below(line.lit, line.y0_khz, line.dy_khz)
        for pair in PAIRS:
            args = (line.y0_khz, line.dy_khz, line.dt_ms, *pair)
            steady_px[pair] += int(steady(lit, *args, guard_khz=None).sum())
            taken = steady(low, *args, guard_khz=MHD_GUARD_KHZ)
            take_px[pair] += int(taken[:, present].sum())
    if not area:
        lo, hi = COST_BAND_KHZ
        raise ValueError(
            f"the rule shots have no present pixel lit at {lo:g}-{hi:g} kHz"
        )
    budget = MAX_LOSS * area
    run_ms = next((r for r in RUN_MS if run_px[r] <= budget), None)
    within = [p for p in PAIRS if steady_px[p] <= budget]
    # max keeps the first of equal takes, so a tie goes to PAIRS order.
    pair = max(within, key=take_px.__getitem__, default=None)
    drift_khz, steady_ms = pair or (min(DRIFTS_KHZ), max(STEADY_MS))
    rules = Rules(RUN_MS[-1] if run_ms is None else run_ms, drift_khz, steady_ms)
    report = {
        "rule_shots": [int(line.shot) for line in lines],
        "max_loss": MAX_LOSS,
        "cost_band_khz": list(COST_BAND_KHZ),
        "rule_below_khz": RULE_BELOW_KHZ,
        "lit_present_px": area,
        "absent_run_px": {f"{r:g}": px for r, px in run_px.items()},
        "steady_px": {f"{d:g}-{s:g}": px for (d, s), px in steady_px.items()},
        "steady_take_px": {f"{d:g}-{s:g}": px for (d, s), px in take_px.items()},
        "run_ms_within": run_ms is not None,
        "steady_within": pair is not None,
        "bands": _band_report(lines, rules),
    }
    return rules, report


def bright_level(masks, lines) -> int | None:
    """floor(BRIGHT_PERCENTILE-th percentile) of lines[i].bright where masks[i]
    == 1, over all the shots; None when no pixel is 1."""
    values = [
        np.asarray(line.bright)[np.asarray(mask) == 1]
        for mask, line in zip(masks, lines, strict=True)
    ]
    values = np.concatenate(values) if values else np.zeros(0, dtype=np.uint8)
    if not values.size:
        return None
    return int(np.floor(np.percentile(values, BRIGHT_PERCENTILE)))


def bright_background_share(masks, lines, bright_u8) -> float | None:
    """The share of the shots' present-frame background that the bright rule
    IGNOREs: of the pixels of present columns that `masks` (the first pass's,
    made without the bright rule) score 0, all bins, and so unlit and not
    MHD-like, those whose brightest row reaches bright_u8. None when bright_u8
    is None or no such pixel is 0."""
    if bright_u8 is None:
        return None
    background = ignored = 0
    for mask, line in zip(masks, lines, strict=True):
        zero = (np.asarray(mask) == 0) & (line.state == PRESENT) & ~line.lit
        background += int(zero.sum())
        ignored += int((zero & (line.bright >= bright_u8)).sum())
    return ignored / background if background else None


def fallbacks(rules: Rules, report: dict) -> list[str]:
    """One sentence per rule whose value fell back, no candidate being within
    the budget: the rule, the fallback and what it costs."""
    area = report["lit_present_px"]
    band = "{:g}-{:g} kHz".format(*report["cost_band_khz"])
    budget = f"{report['max_loss']:.0%} budget ({report['max_loss'] * area:.0f} px)"
    taken = []
    if not report["run_ms_within"]:
        px = report["absent_run_px"][f"{rules.run_ms:g}"]
        taken.append(("absent-run", f"{rules.run_ms:g} ms", px))
    if not report["steady_within"]:
        px = report["steady_px"][f"{rules.drift_khz:g}-{rules.steady_ms:g}"]
        value = f"{rules.drift_khz:g} kHz over {rules.steady_ms:g} ms"
        taken.append(("steady", value, px))
    return [
        f"no {rule} candidate is within the {budget}; its fallback, {value}, "
        f"costs {px} of the {area} lit present-frame px at {band} "
        f"({px / area:.1%}), over the budget"
        for rule, value, px in taken
    ]


def _table(head, rows) -> list[str]:
    """A Markdown table's lines."""
    return [
        "| " + " | ".join(map(str, head)) + " |",
        "|" + "---|" * len(head),
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ]


def rules_md(rules: Rules, report: dict) -> str:
    """rules.md: the values, every candidate's cost and take, and the bands."""
    area = report["lit_present_px"]
    band = "{:g}-{:g} kHz".format(*report["cost_band_khz"])
    low = f"{report['rule_below_khz']:g} kHz"
    share = f"{report['max_loss']:.0%}"
    background = report.get("bright_background_share")

    def cost(px) -> str:
        if px is None:
            return "-"
        return f"{px} px ({px / area:.1%})" if area else f"{px} px"

    def within(ok: bool) -> str:
        return "yes" if ok else "no: the fallback, the least inclusive"

    run = report["absent_run_px"].get(f"{rules.run_ms:g}")
    pair = report["steady_px"].get(f"{rules.drift_khz:g}-{rules.steady_ms:g}")
    guard = "no guard" if rules.guard_khz is None else f"below {rules.guard_khz:g} kHz"
    if background is None:
        ignored = "-"
        bright = "`bright_background_share` is None: no bright rule, or no background."
    else:
        ignored = f"IGNOREs {background:.1%} of the present background"
        bright = (
            f"It IGNOREs {background:.1%} of those masks' present-frame background "
            "(`bright_background_share`: the pixels of present columns they score "
            "0, at any frequency)."
        )
    values = [
        (
            "absent-run",
            f"`run_ms` {rules.run_ms:g} ms",
            cost(run),
            within(report["run_ms_within"]),
        ),
        (
            "steady",
            (
                f"`drift_khz` {rules.drift_khz:g} kHz over `steady_ms` "
                f"{rules.steady_ms:g} ms, {guard}"
            ),
            cost(pair),
            within(report["steady_within"]),
        ),
        ("bright", f"`bright_u8` {rules.bright_u8}", ignored, ""),
    ]
    drifts = sorted(DRIFTS_KHZ, reverse=True)

    def pairs(key: str) -> list[tuple]:
        return [
            (f"{d:g} kHz", *(report[key][f"{d:g}-{s:g}"] for s in STEADY_MS))
            for d in drifts
        ]

    band_rows = [
        (
            f"{b['lo_khz']:g}-{b['hi_khz']:g}",
            b["lit_present_px"],
            b["absent_run_px"],
            b["steady_px"],
        )
        for b in report["bands"]
    ]
    over = fallbacks(rules, report)
    verdict = (
        " **Over the budget:** " + " ".join(f"{s[0].upper()}{s[1:]}." for s in over)
        if over
        else " Every value is within it."
    )
    intro = (
        f"Picked on SegNet v2's {len(report['rule_shots'])} training shots only "
        "(`labeler.ae.seg.mhdlines`, spec §2.4). The absent-run and steady rules "
        f"act below {low} only, on the regions lit there; above it a present "
        "frame's lit pixels stay AE, as in pseudo-v1. A candidate's cost is what "
        f"its test takes of the lit pixels of the owner's present frames at {band} "
        f"({area} in all), run on the regions lit there. The budget is {share} of "
        "them "
        f"({report['max_loss'] * area:.0f} px).{verdict}"
    )
    note = (
        f"The absent-run cost is taken on the regions lit inside {band}, and the "
        "steady cost on the present pixels there, without the guard. The steady "
        "pair is the one within the budget that takes the most below "
        f"{low}, as the rule applies it (its take, below); a tie goes to the "
        f"more inclusive. `bright_u8` is the {BRIGHT_PERCENTILE}th percentile of "
        "the brightest cross-power row at the AE pixels of the rule shots' masks "
        "made with the two rules above; None, and no bright rule, when they have "
        f"none. {bright}"
    )
    out = [
        "# pseudo-v2's MHD-line rules",
        "",
        intro,
        "",
        *_table(("Rule", "Value", "Cost", f"Within {share}"), values),
        "",
        note,
        "",
        f"## Every candidate's cost at {band} (px)",
        "",
        *_table(
            ("`run_ms`", *(f"{r:g} ms" for r in RUN_MS)),
            [("absent-run", *(report["absent_run_px"][f"{r:g}"] for r in RUN_MS))],
        ),
        "",
        *_table(("`drift_khz`", *(f"{s:g} ms" for s in STEADY_MS)), pairs("steady_px")),
        "",
        f"## Every steady pair's take below {low}, in present frames (px)",
        "",
        *_table(
            ("`drift_khz`", *(f"{s:g} ms" for s in STEADY_MS)),
            pairs("steady_take_px"),
        ),
        "",
        "## By band: the chosen rules over the rule shots' present frames (px)",
        "",
        *_table(("Band (kHz)", "Lit", "Absent-run", "Steady"), band_rows),
    ]
    return "\n".join(out) + "\n"


def write_rules(out: Path, rules: Rules, report: dict) -> None:
    """out / "rules.json": {"rules": asdict(rules), **report}; out / "rules.md":
    "# pseudo-v2's MHD-line rules", the values, and a table of the bands."""
    out = Path(out)
    record = {"rules": asdict(rules), **report}
    with atomic_path(out / "rules.json") as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    with atomic_path(out / "rules.md") as tmp:
        tmp.write_text(rules_md(rules, report))
