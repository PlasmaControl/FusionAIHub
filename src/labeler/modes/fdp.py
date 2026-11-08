"""The 1 ms label databases of `docs/labeler/fdp_labels.md`, one a parent event (`mhd_n<K>`).

Same files as the AE, ELM and confinement databases: `spans.parquet`, `shots.parquet`,
`valid.csv`, `meta.json`, `README.md` and `grid_1ms/`. Frame `k` is `[k, k + 1)` ms. The labels
are the label view of `labeler.modes.cells` (only labelled modes, harmonics dropped):

- `spans.parquet`: the present spans of the parent (`subcategory` empty: some labelled n = K mode)
  and of each of its subcategories (`m<m>`, a labelled m/K mode): `shot, t_start_ms, t_end_ms`
  (int32, half-open), `subcategory`, `m` (null on a parent row), `n`, `mn`, `confidence` (the
  largest P(resonant) the span holds; a parent's is the largest of its subcategories' under it),
  `tier`. A subcategory's spans lie inside its parent's, and sibling subcategories overlap freely.
- `shots.parquet`: one row per listed shot: `status` (labeled, no_labels, no_window),
  the assessed window, `present_ms`, `n_spans`, `tracks_kept`, `harmonics_dropped`, `tree`, and
  the review tier (`tier`, `reviewers`, `verified_on`, `notes`).
- `valid.csv`: `shot, valid` for the listed shots, valid 1 when the shot is labeled.
- `grid_1ms/<shot // 1000>/<shot>.bin`: one byte a ms of the window (`start_ms` to `end_ms` in
  `grid_1ms/index.csv`); bit 0 the parent present, bits 1-6 the subcategories in `meta.json`'s
  `grid_bits` order, bit 7 not observable (the parent could not be assessed).

Writing is explicit: `write` takes the folder; nothing here chooses a destination.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.config import atomic_path
from labeler.mhd import rows

from . import cells as C
from . import suggest
from . import tracks as T

GRID_DIR = "grid_1ms"
NOT_OBSERVABLE_BIT = 7
MAX_KIDS = 6  # bits 1-6
UNVERIFIED = {"tier": "unverified", "reviewers": "", "verified_on": "", "notes": ""}
SPAN_COLUMNS = [
    "shot",
    "t_start_ms",
    "t_end_ms",
    "subcategory",
    "m",
    "n",
    "mn",
    "confidence",
    "tier",
]
STATUSES = {
    "labeled": "run; its window, spans and grid file are in the database",
    "no_labels": "no port-v6 labels were computed for the shot (no files)",
    "no_window": "omnimode found no assessed window for the shot (no Ip record)",
    "failed": "the labeller failed on the shot (a read error or a model error in its info record)",
}


def kids_of(event: str, kids: dict[str, str]) -> dict[str, str]:
    """The event's subcategories, `{event.m<m>: "m/n"}`, by m."""
    out = {e: name for e, name in kids.items() if C.parent_of(e) == event}
    if len(out) > MAX_KIDS:
        raise ValueError(
            f"{event} has {len(out)} subcategories; the grid byte holds {MAX_KIDS}"
        )
    return out


def bits(event: str, kids: dict[str, str]) -> dict[str, int]:
    """Grid bit of each name: `parent`, then the subcategories (`m<m>`) from bit 1."""
    out = {"parent": 0}
    for i, child in enumerate(kids_of(event, kids), start=1):
        out[child.partition(".")[2]] = i
    out["not_observable"] = NOT_OBSERVABLE_BIT
    return out


def shot_spans(event, kids, cells, shot, tier) -> list[dict]:
    """The shot's present spans for `event` and its subcategories, from `shot_cells(strict=True)`."""
    n = C.PARENTS[event]
    mine = kids_of(event, kids)
    child_runs = {child: C.present_runs(cells[child]) for child in mine}
    out = []
    for a, b, conf in C.present_runs(cells[event]):
        under = [
            c
            for runs in child_runs.values()
            for x, y, c in runs
            if x < b and y > a and np.isfinite(c)
        ]
        out.append((a, b, "", None, n, "", max(under) if under else conf))
    for child, runs in child_runs.items():
        m, _ = T.parse_class(mine[child])
        out += [
            (a, b, child.partition(".")[2], m, n, mine[child], conf)
            for a, b, conf in runs
        ]
    return [
        {
            "shot": int(shot),
            "t_start_ms": a,
            "t_end_ms": b,
            "subcategory": sub,
            "m": m,
            "n": n,
            "mn": mn,
            "confidence": conf,
            "tier": tier,
        }
        for a, b, sub, m, n, mn, conf in out
    ]


def grid_bytes(event, kids, cells) -> tuple[int, int, bytes]:
    """`(start_ms, end_ms, bytes)`: one byte a ms of the parent's window, bits as `bits`."""
    parent = cells[event]
    byte = np.zeros(len(parent.state), np.uint8)
    byte[parent.state == rows.PRESENT] |= 1
    byte[parent.state == rows.UNSEEN] |= 1 << NOT_OBSERVABLE_BIT
    for child in kids_of(event, kids):
        byte[cells[child].state == rows.PRESENT] |= (
            1 << bits(event, kids)[child.partition(".")[2]]
        )
    return parent.lo, parent.hi, byte.tobytes()


def build(event, ready, kids, listed, skipped, dropped, tiers) -> dict:
    """The event's tables and grid bytes over the ready shots.

    `listed`: every shot the run covers (a shot with no labels still gets a `shots` row);
    `tiers`: `{shot: {tier, reviewers, verified_on, notes}}` for the shots not unverified."""
    spans, shots, grid = [], [], {}
    for shot in sorted(listed):
        tier = {**UNVERIFIED, **tiers.get(shot, {})}
        row = {"shot": shot, "status": "no_labels", **tier}
        got = dict(ready).get(shot)
        if got is None:
            if shot in skipped and skipped[shot].startswith(suggest.NO_WINDOW):
                row["status"] = "no_window"
            elif shot in skipped and skipped[shot].startswith(suggest.FAILED):
                row["status"] = "failed"
            shots.append(row)
            continue
        cells = C.shot_cells(got, kids, strict=True)
        found = shot_spans(event, kids, cells, shot, tier["tier"])
        a, b, data = grid_bytes(event, kids, cells)
        grid[shot] = (a, b, data)
        parent = cells[event]
        row |= {
            "status": "labeled",
            "window_start_ms": a,
            "window_end_ms": b,
            "present_ms": int((parent.state == rows.PRESENT).sum()),
            "not_observable_ms": int((parent.state == rows.UNSEEN).sum()),
            "n_spans": sum(1 for s in found if not s["subcategory"]),
            "n_sub_spans": sum(1 for s in found if s["subcategory"]),
            "tracks_kept": dropped[shot]["kept"],
            "harmonics_dropped": dropped[shot]["harmonics"],
            "tree": (got.info.get("inputs") or {}).get("tree"),
        }
        shots.append(row)
        spans += found
    spans = pd.DataFrame(spans, columns=SPAN_COLUMNS)
    spans = spans.astype(
        {
            "shot": "int32",
            "t_start_ms": "int32",
            "t_end_ms": "int32",
            "m": "Int8",
            "n": "int8",
            "confidence": "float32",
        }
    )
    table = pd.DataFrame(shots)
    for column in (
        "window_start_ms",
        "window_end_ms",
        "present_ms",
        "not_observable_ms",
        "n_spans",
        "n_sub_spans",
        "tracks_kept",
        "harmonics_dropped",
    ):
        if column not in table:
            table[column] = pd.NA
        table[column] = table[column].astype("Int32")
    table["shot"] = table["shot"].astype("int32")
    valid = pd.DataFrame(
        {"shot": table["shot"], "valid": (table.status == "labeled").astype("uint8")}
    )
    return {"spans": spans, "shots": table, "valid": valid, "grid": grid}


def write_grid(out: Path, grid: dict, job: str) -> int:
    """`grid_1ms/<shot // 1000>/<shot>.bin` for every labeled shot and `grid_1ms/index.csv`."""
    gdir = Path(out) / GRID_DIR
    gdir.mkdir(parents=True, exist_ok=True)
    for shot, (_, _, data) in sorted(grid.items()):
        f = gdir / str(shot // 1000) / f"{shot}.bin"
        with atomic_path(f) as tmp:
            tmp.write_bytes(data)
    index = pd.DataFrame(
        [(s, a, b, job) for s, (a, b, _) in sorted(grid.items())],
        columns=["shot", "start_ms", "end_ms", "job"],
    )
    with atomic_path(gdir / "index.csv") as tmp:
        index.to_csv(tmp, index=False)
    return len(grid)


def meta_json(event, kids, built, *, run, source, rules, repo, note) -> dict:
    spans, shots = built["spans"], built["shots"]
    labeled = shots[shots.status == "labeled"]
    return {
        "name": run,
        "category": event,
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "made_by": "scripts/labeler/mode_fdp_label.py publish",
        "git_sha": suggest.git_sha(repo),
        "source": source,
        "rules": rules,
        "files": {
            "spans.parquet": (
                "present spans: shot (int32), t_start_ms, t_end_ms (int32, half-open [start, end) "
                "on the 1 ms grid), subcategory ('' = the parent's own span, else m<m>), m (null "
                "on a parent row), n, mn (the m/n class, '' on a parent row), confidence (largest "
                "P(resonant) the span holds), tier"
            ),
            "shots.parquet": (
                "one row per listed shot: status, window_start_ms, window_end_ms (the assessed "
                "window, half-open), present_ms (the parent's), not_observable_ms, n_spans, "
                "n_sub_spans, tracks_kept, harmonics_dropped, tree (the equilibrium), tier, "
                "reviewers, verified_on, notes"
            ),
            "valid.csv": "shot, valid: the listed shots, valid 1 when the shot is labeled",
            f"{GRID_DIR}/<shot // 1000>/<shot>.bin": (
                "one byte a 1 ms frame of the window, bit 0 the parent present, bits 1-6 the "
                "subcategories (grid_bits), bit 7 not observable"
            ),
            f"{GRID_DIR}/index.csv": "shot, start_ms, end_ms (end_ms - start_ms bytes), job",
        },
        "grid_bits": bits(event, kids),
        "subcategories": {
            c.partition(".")[2]: name for c, name in kids_of(event, kids).items()
        },
        "counts": {
            "listed": len(shots),
            "labeled": len(labeled),
            "parent_spans": int((spans.subcategory == "").sum()),
            "subcategory_spans": int((spans.subcategory != "").sum()),
            "by_subcategory": {
                k: int(v)
                for k, v in spans[spans.subcategory != ""].mn.value_counts().items()
            },
            "harmonic_tracks_dropped": int(labeled.harmonics_dropped.sum())
            if len(labeled)
            else 0,
            "present_ms": int(labeled.present_ms.sum()) if len(labeled) else 0,
        },
        "statuses": STATUSES,
        "tiers": {k: int(v) for k, v in shots.tier.value_counts().items()},
        "caveats": note,
    }


def readme(event, meta: dict) -> str:
    c = meta["counts"]
    subs = ", ".join(f"`{k}` ({v})" for k, v in meta["subcategories"].items())
    return f"""# MHD {event.replace("mhd_", "")} modes, 1 ms grid, with their m/n subcategories

Labelled modes of `{event}` from {meta["source"]["labels"]}; subcategories: {subs}. Harmonics are
not labelled, only labelled modes are written. {meta["caveats"][0]}
Built {meta["made_at"]} by `{meta["made_by"]}` (git {meta["git_sha"]}).

Listed {c["listed"]}; labeled {c["labeled"]}; parent spans {c["parent_spans"]};
subcategory spans {c["subcategory_spans"]}. Tiers: {meta["tiers"]}. Details and caveats: `meta.json`.

```python
import numpy as np, pandas as pd
spans = pd.read_parquet("spans.parquet")   # shot, t_start_ms, t_end_ms, subcategory, m, n, mn, confidence, tier
shots = pd.read_parquet("shots.parquet").set_index("shot")
def mode_1ms(shot, sub=""):
    \"\"\"(t_ms, 0/1) on the assessed window: `sub` '' the parent, else 'm2' etc.\"\"\"
    w0, w1 = shots.loc[shot, ["window_start_ms", "window_end_ms"]].astype(int)
    lab = np.zeros(w1 - w0, dtype=np.uint8)
    pick = spans[(spans.shot == shot) & (spans.subcategory == sub)]
    for a, b in pick[["t_start_ms", "t_end_ms"]].itertuples(index=False):
        lab[a - w0:b - w0] = 1
    return np.arange(w0, w1), lab
```

The same labels dense, one byte a ms (bits in `meta.json` `grid_bits`):

```python
grid = pd.read_csv("{GRID_DIR}/index.csv").set_index("shot")
b = np.fromfile(f"{GRID_DIR}/{{shot // 1000}}/{{shot}}.bin", np.uint8)   # e.g. 199/199597.bin
parent, not_observable = b & 1, (b >> 7) & 1
sub = lambda i: (b >> i) & 1      # i from grid_bits, e.g. m2
t_ms = grid.start_ms[shot] + np.arange(len(b))                           # frame [t, t + 1) ms
```

Frame `k` is `[k, k + 1)` ms. Outside `[window_start_ms, window_end_ms)` nothing was assessed.
"""


def write(
    out,
    event,
    ready,
    kids,
    listed,
    skipped,
    dropped,
    tiers,
    *,
    run,
    source,
    rules,
    repo,
    note,
) -> dict:
    """Write the event's database into the folder `out` (made if missing); returns its meta.

    `out` is the caller's: a folder is never chosen here, and an existing database is never
    written into (a `meta.json` there is an error)."""
    out = Path(out)
    if (out / "meta.json").exists():
        raise FileExistsError(f"{out} already holds a database")
    built = build(event, ready, kids, listed, skipped, dropped, tiers)
    out.mkdir(parents=True, exist_ok=True)
    meta = meta_json(
        event, kids, built, run=run, source=source, rules=rules, repo=repo, note=note
    )
    with atomic_path(out / "spans.parquet") as tmp:
        built["spans"].to_parquet(tmp, index=False)
    with atomic_path(out / "shots.parquet") as tmp:
        built["shots"].to_parquet(tmp, index=False)
    with atomic_path(out / "valid.csv") as tmp:
        built["valid"].to_csv(tmp, index=False)
    write_grid(out, built["grid"], run)
    with atomic_path(out / "README.md") as tmp:
        tmp.write_text(readme(event, meta))
    with atomic_path(
        out / "meta.json"
    ) as tmp:  # last: its presence says the database is whole
        tmp.write_text(json.dumps(meta, indent=1) + "\n")
    return meta
