"""The corpus census: what every FAITH corpus file actually carries, one row per (file, group).

This exists because the question was answered wrongly once. The shotsearch `manifest.parquet`
counted a group as populated when `ydata.shape[0] > 1` -- the CHANNEL axis -- so every `(C, 1)`
placeholder for a diagnostic that did not record counted as present, and the availability table
built on it was wrong in the direction that matters (too optimistic) for exactly the groups that
are absent most often. Nothing in this project reuses that manifest; this module recounts, on the
time axis, and writes the count with the corpus directory, the file count and the git sha of the
code that produced it so the next reader can tell which census they are holding.

Two properties make a 16,909-file census cheap enough to redo whenever it is in doubt:

* **Header-only.** A row costs the group's `ydata` shape (metadata, no data pages) and two scalar
  reads from `xdata`. No `ydata` is ever read -- a single 500 kHz group is 3 M samples x 48
  channels, and reading one would make the scan hours instead of minutes.
* **One file per worker task.** The files are independent, so `multiprocessing.Pool` over the
  sorted file list is the whole parallel structure. Results come back unordered and are sorted by
  (shot, group) before anything is written, so the table on disk does not depend on scheduling.

An unreadable file (2.3 % of the corpus, truncated writes) is a row of its own, with `openable`
False and the exception text in `error` -- never a silently missing shot, which would read as
"that shot has no diagnostics" rather than "that shot was not measured here".
"""

from __future__ import annotations

import datetime as dt
import json
import multiprocessing as mp
from collections.abc import Iterable, Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# `_git_sha` is build's, imported rather than copied: a census sidecar and a build manifest that
# disagreed about what a sha means would be worse than the private-name import.
from .build import _git_sha

# The layout rules -- what counts as a channel, a sample, a video group, a placeholder -- are
# corpus.py's, so that the census and the reader can never come to different answers about which
# groups a shot has.
from .corpus import MIN_SAMPLES, SUFFIX, describe, shot_of

# The census table, in order, with the dtype each column is stored as. `str` columns are pandas'
# string dtype, which round-trips through Parquet unchanged; the two counts are integers because
# they are counts of things that exist, while every measured quantity is float64 so that "not
# measured" can be NaN rather than a zero indistinguishable from a measurement.
COLUMNS: dict[str, str] = {
    "shot": "int32",
    "group": "str",
    "kind": "str",  # "signal" | "video" | "" for a file that did not open
    "n_channels": "int32",
    "n_samples": "int64",  # on the group's TIME axis, as stored (the pad sample included)
    "t0_s": "float64",  # seconds, as the corpus stores them -- this table is not converted to ms
    "t1_s": "float64",
    "fs_hz": "float64",
    "present": "bool",  # the diagnostic recorded something: >= 2 samples on the time axis
    "openable": "bool",
    "error": "str",
}

# Files per task handed to a worker. The per-file work is a few milliseconds of metadata reads on
# a shared filesystem, so a chunk of one would spend more time in the queue than in h5py; 16 is
# the floor, and larger chunks are used when there are enough files to keep every worker fed
# several times over (which keeps a straggler chunk from being the whole tail of the scan).
MIN_CHUNKSIZE = 16


def files(
    corpus_dir: Path, *, limit: int | None = None, shots: Iterable[int] | None = None
) -> list[Path]:
    """The corpus files to scan, sorted by shot.

    `shots` selects; a requested shot with no file is simply not scanned, because this census
    reports on files that exist and a shot outside the corpus is not a finding about the corpus.
    `limit` then takes the first N, which is what makes a pilot run a prefix of the full run
    rather than a different sample.
    """
    corpus_dir = Path(corpus_dir)
    found = sorted(corpus_dir.glob(f"*{SUFFIX}"), key=shot_of)
    if shots is not None:
        want = {int(s) for s in shots}
        found = [p for p in found if shot_of(p) in want]
    return found[:limit] if limit is not None else found


def scan_file(path: Path) -> list[dict]:
    """One file's rows: one per group, or a single `openable=False` row if it cannot be read.

    Failing to OPEN the file is a fact about the file, and is one row saying so. Failing on one
    GROUP of a file that opened is a fact about that group: it becomes that group's row with the
    exception in `error`, and the file's other groups are still measured. The two used to be the
    same case -- every group in one `try` -- which meant that a single malformed group anywhere in
    a 16,909-file corpus discarded the whole shot and reported it as unreadable, understating that
    shot's coverage in exactly the direction this table exists to correct.
    """
    path = Path(path)
    shot = shot_of(path)
    try:
        with h5py.File(path, "r", locking=False) as f:
            rows = [_safe_group_row(shot, name, f) for name in sorted(f)]
    except (OSError, KeyError, RuntimeError) as e:
        return [_row(shot, openable=False, error=f"{type(e).__name__}: {e}")]
    # Every file scanned is at least one row, including one that opens and holds nothing. The
    # file count -- and so every fraction computed against it -- is the number of distinct shots
    # in this table, and a file that contributed no row would quietly leave the denominator.
    return rows or [_row(shot, error="no groups")]


def scan(
    corpus_dir: Path,
    *,
    workers: int,
    limit: int | None = None,
    shots: Iterable[int] | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Census `corpus_dir` with `workers` processes, sorted by (shot, group).

    `progress` is the caller's decision, not this function's: the CLI turns it on only for a tty,
    and a SLURM job's log is not one.
    """
    todo = files(corpus_dir, limit=limit, shots=shots)
    rows: list[dict] = []
    with mp.Pool(max(1, int(workers))) as pool:
        done = pool.imap_unordered(scan_file, todo, chunksize=_chunksize(len(todo), workers))
        for one in _progress(done, len(todo)) if progress else done:
            rows.extend(one)
    return table(rows)


def table(rows: Sequence[dict]) -> pd.DataFrame:
    """`rows` as the census table: exactly COLUMNS, in dtype, sorted by (shot, group)."""
    df = pd.DataFrame(list(rows), columns=list(COLUMNS)).astype(COLUMNS)
    return df.sort_values(["shot", "group"], kind="stable").reset_index(drop=True)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-group presence over the openable files -- the availability table.

    The denominator is the files that OPENED, not the files that exist: a truncated file says
    nothing about whether DIII-D recorded mhr that day, and counting it against the diagnostic
    would blend a filesystem fact into a coverage fact.
    """
    openable = int(df.loc[df["openable"], "shot"].nunique())
    groups = df[df["group"] != ""]
    out = []
    for name, g in groups.groupby("group", sort=True):
        seen = g[g["present"]]
        out.append(
            {
                "group": name,
                "kind": (seen["kind"].iloc[0] if len(seen) else g["kind"].iloc[0]),
                "n_present": len(seen),
                "n_openable": openable,
                "frac_present": (len(seen) / openable) if openable else float("nan"),
                "median_fs_hz": float(seen["fs_hz"].median()) if len(seen) else float("nan"),
                "median_span_s": (
                    float((seen["t1_s"] - seen["t0_s"]).median()) if len(seen) else float("nan")
                ),
            }
        )
    cols = ["group", "kind", "n_present", "n_openable", "frac_present"]
    frame = pd.DataFrame(out, columns=[*cols, "median_fs_hz", "median_span_s"])
    order = frame.sort_values(["frac_present", "group"], ascending=[False, True])
    return order.reset_index(drop=True)


def write(
    df: pd.DataFrame, out: Path, *, corpus_dir: Path, workers: int, elapsed_s: float
) -> Path:
    """Write the table to `out` and, beside it, the JSON that says what run produced it.

    The sidecar is the part that keeps this census from becoming the next stale manifest: it
    names the directory scanned, how many files were found and how many of them opened, and the
    git sha of the code that counted -- so a table whose numbers look wrong can be traced to a
    run rather than argued about.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {
        "n_files": int(df["shot"].nunique()),
        "n_openable": int(df.loc[df["openable"], "shot"].nunique()),
        "n_rows": len(df),
        "elapsed_s": round(float(elapsed_s), 3),
        "workers": int(workers),
        "corpus_dir": str(corpus_dir),
        "written_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return sidecar


# ------------------------------------------------------------------------------------ internals


def _row(shot: int, **over) -> dict:
    """One row with every column at its "nothing measured" value: NaN for the measurements, zero
    for the counts, False for the claims. Never 0.0 for a quantity -- a rate of zero and a rate
    nobody measured are different rows in any table built on this one."""
    row = {
        "shot": int(shot),
        "group": "",
        "kind": "",
        "n_channels": 0,
        "n_samples": 0,
        "t0_s": float("nan"),
        "t1_s": float("nan"),
        "fs_hz": float("nan"),
        "present": False,
        "openable": True,
        "error": "",
    }
    row.update(over)
    return row


def _safe_group_row(shot: int, name: str, f: h5py.File) -> dict:
    """`_group_row`, with any exception it raises turned into that group's row.

    Deliberately `Exception` and not a list: the point is that a group this scan has never seen
    before cannot cost the file, and the exceptions h5py raises on a malformed member are not a
    closed set (`TypeError` from a shape it cannot interpret, `AttributeError` from a member that
    is not a dataset, `ValueError`, `RuntimeError`, ...). The row that comes back names the
    exception, so an unfamiliar failure is visible in the table rather than absent from it.
    """
    try:
        return _group_row(shot, name, f[name])
    except Exception as e:  # noqa: BLE001 -- one odd group may not cost the file's other groups
        return _row(shot, group=name, error=f"{type(e).__name__}: {e}")


def _group_row(shot: int, name: str, g) -> dict:
    if not isinstance(g, h5py.Group) or "ydata" not in g or "xdata" not in g:
        return _row(shot, group=name, error="no xdata/ydata")
    kind, n_channels, n_samples = describe(g["ydata"].shape)
    present = n_samples >= MIN_SAMPLES
    row = _row(shot, group=name, kind=kind, n_channels=n_channels, n_samples=n_samples)
    if not present:
        # A placeholder's xdata is a single 0.0. Reporting that as the group's t0 would put a
        # measurement where there is none, so the span stays NaN and only the shape is recorded.
        return row
    x = g["xdata"]
    # The shape, not merely the length -- the same guard `CorpusReader.coverage` applies, for the
    # same reason: `float(x[0])` on a 2-D xdata raises rather than answering. The shape stays in
    # the table because it is still a fact about the file; the span does not, because it was not
    # measured.
    if x.ndim != 1 or x.shape[0] < MIN_SAMPLES:
        row.update(error=f"xdata is {tuple(x.shape)}, not a 1-D time axis")
        return row
    t0, t1 = float(x[0]), float(x[-1])
    fs_hz = (n_samples - 1) / (t1 - t0) if t1 > t0 else float("nan")
    row.update(present=True, t0_s=t0, t1_s=t1, fs_hz=fs_hz)
    return row


def _chunksize(n_files: int, workers: int) -> int:
    return max(MIN_CHUNKSIZE, n_files // (max(1, int(workers)) * 8))


def _progress(it, total: int):
    # Imported here and not at the top: a progress bar is for a human watching a terminal, and
    # `scan()` must not need one to run under SLURM.
    from tqdm import tqdm

    return tqdm(it, total=total, unit="file", smoothing=0.05)


def format_summary(df: pd.DataFrame) -> str:
    """The availability table as text, widest coverage first."""
    head = f"{'group':<22}{'kind':<8}{'present':>9}{'of':>8}{'fraction':>10}"
    lines = [head + f"{'rate':>13}{'span':>10}"]
    for _, r in df.iterrows():
        rate = "" if not np.isfinite(r["median_fs_hz"]) else _rate(r["median_fs_hz"])
        span = "" if not np.isfinite(r["median_span_s"]) else f"{r['median_span_s']:.2f} s"
        lines.append(
            f"{r['group']:<22}{r['kind']:<8}{r['n_present']:>9,}{r['n_openable']:>8,}"
            f"{r['frac_present']:>9.1%} {rate:>12}{span:>10}"
        )
    return "\n".join(lines)


def _rate(fs_hz: float) -> str:
    return f"{fs_hz / 1000.0:.1f} kHz" if fs_hz >= 1000.0 else f"{fs_hz:.1f} Hz"
