"""Check `events.heuristics`'s sawtooth port against omnimode, crash for crash.

`src/labelmaker/events/heuristics.py` is a port of
`omnimode.mrms.ece.find_sawteeth` with one thing changed: the reference
recomputes the 1 ms envelope inside `crash_steps` for every candidate crash,
which on shot 198658's 429 candidates is 430 passes over a 594 MB record.
A port is only trustworthy if somebody has run both on real data and
compared the ANSWERS - not the counts, the times - so this script does that
and writes what it found into
`tests/labelmaker/data/sawtooth_198658_reference.json`, which
`tests/labelmaker/test_events_heuristics.py` then reads on every run.

What it does:

1. reads one shot's `ece` group straight from the corpus (48 channels,
   3,096,576 samples on 198658), optionally truncated to `--max-seconds`,
   which is the knob for bounding the reference's O(candidates x record)
   run - it takes about eight minutes on the whole of 198658 against the
   port's 0.4 s;
2. runs the **port**, `heuristics.sawtooth_events`, and times it;
3. runs the **reference**, `omnimode.mrms.ece.find_sawteeth`, on the same
   array and times it. omnimode is put on `sys.path` HERE and nowhere else:
   no library code and no test in labelmaker may import it;
4. pairs the two crash lists nearest-neighbour and reports every crash's
   time difference, the largest of them, and anything unpaired;
5. prints counts, median and mean periods and elapsed times, and with
   `--json` writes the lot - plus both repositories' git shas - as the
   committed record. `labelmaker_sha` is HEAD at RUN time, which is the
   PARENT of the commit that carries the record: the script has to run
   before the commit its output goes into.

    PYTHONPATH=$PWD/src \\
        pixi run --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \\
        -e labelmaker python scripts/labelmaker/sawtooth_reference_check.py \\
        --shot 198658 --json

Re-run it when the port's crash search changes. The acceptance it exists to
justify is in `heuristics`'s module docstring: 47 +/- 3 crashes with a
69 +/- 5 ms median period on 198658, which are the REFERENCE's own numbers
on that shot rather than the plan's remembered "45 sawteeth, 76 ms".

The reference side costs about eight minutes, so a change to the port that
plainly cannot move the reference does not need it re-run - but it does need
the PORT re-run, or the record's numbers are a measurement of code that no
longer exists. `--port-only` is that: it re-runs step 2 alone on the shot and
the record the file already names, and amends it with a `port_rerun` stanza -
the crash count, the median period, the elapsed time, the largest distance
from the crashes the record already holds, the sha it was measured at, and
whether anything under `src/labelmaker` was uncommitted when it ran (`dirty`;
a dirty record names a tree that is not the tree that was measured, so re-run
it once the change is committed).
`tests/labelmaker/test_events_heuristics.py` then checks that count against
the reference's, so a port that drifts away from omnimode fails the suite
rather than waiting for somebody to spend the eight minutes.

    PYTHONPATH=$PWD/src \\
        pixi run --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \\
        -e labelmaker python scripts/labelmaker/sawtooth_reference_check.py \\
        --port-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from labelmaker.events import heuristics

#: Read only here. `heuristics` is a PORT of this and imports nothing from
#: it; a test that imported omnimode would pin the reference rather than the
#: port, and would skip wherever omnimode is not installed.
OMNIMODE_SRC = Path("/scratch/gpfs/nc1514/omnimode/src")

CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
#: Two crash times count as the same crash within this, in ms. See `main`.
AGREE_MS = 1e-9
OUT = REPO / "tests" / "labelmaker" / "data" / "sawtooth_198658_reference.json"


def reference_find_sawteeth(y: np.ndarray, t_ms: np.ndarray,
                            omnimode_src: Path) -> list[float]:
    """omnimode's own answer, in ms, with the reference's own envelope."""
    sys.path.insert(0, str(omnimode_src))
    from omnimode.mrms.ece import find_sawteeth

    return [float(tc) for tc in find_sawteeth(y, t_ms)]


def read_ece(corpus_file: Path, max_seconds: float | None):
    """`(t_s, y)` of the shot's ECE array, truncated to `max_seconds`.

    Non-finite samples are left exactly where they are: both detectors turn
    a non-finite bin ratio into no change (`np.where(isfinite, d, 0)` here,
    `np.nan_to_num` there), so a NaN channel or a NaN end votes for nothing
    in either, and stripping them would be a difference between what this
    script runs and what the pipeline runs.
    """
    with h5py.File(corpus_file, "r", locking=False) as f:
        t_s = np.asarray(f["ece"]["xdata"][:], dtype=np.float64)
        n = t_s.size
        if max_seconds is not None:
            n = int(np.searchsorted(t_s, t_s[0] + float(max_seconds), "right"))
            n = max(2, min(n, t_s.size))
        t_s = t_s[:n]
        y = np.asarray(f["ece"]["ydata"][:, :n], dtype=np.float32)
    return t_s, y


def periods_ms(times_s: np.ndarray) -> dict[str, float]:
    """The train's count and its median and mean interval, in ms."""
    times_s = np.sort(np.asarray(times_s, dtype=np.float64))
    if times_s.size < 2:
        return {"n": int(times_s.size), "median_period_ms": float("nan"),
                "mean_period_ms": float("nan")}
    gaps = np.diff(times_s) * 1e3
    return {
        "n": int(times_s.size),
        "median_period_ms": float(np.median(gaps)),
        "mean_period_ms": float(gaps.mean()),
    }


def pair(ref_ms: np.ndarray, ours_ms: np.ndarray) -> list[dict[str, float]]:
    """Every reference crash with its nearest port crash and their gap.

    Nearest-neighbour rather than by index, so that one crash found by only
    one of the two shows up as a large difference on ONE row instead of
    shifting every row after it.
    """
    if ref_ms.size == 0 or ours_ms.size == 0:
        return []
    j = np.abs(ref_ms[:, None] - ours_ms[None, :]).argmin(axis=1)
    return [
        {"reference_ms": float(a), "port_ms": float(ours_ms[k]),
         "delta_ms": float(ours_ms[k] - a)}
        for a, k in zip(ref_ms.tolist(), j.tolist(), strict=True)
    ]


def git_sha(repo: Path) -> str:
    """`repo`'s HEAD, or "unknown" where there is no git to ask."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def git_dirty(repo: Path, pathspec: str) -> bool:
    """Is anything under `pathspec` uncommitted? True where git cannot say.

    `labelmaker_sha` alone is a claim the record cannot back: HEAD names a
    tree, and the code that actually ran is HEAD plus whatever was sitting
    in the working tree. So the stanza carries both, and a `dirty` record
    is a measurement of code nobody can get back. Unknown counts as dirty:
    a record that cannot prove it was clean is not clean.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--", pathspec],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(out.stdout.strip())


def port_rerun(record_path: Path, corpus: Path) -> int:
    """Re-run the port on the record's own shot and amend the record.

    The shot, the corpus file and the `--max-seconds` prefix come from the
    record rather than from the command line, so the re-run is the same
    measurement as the one it is amending and not a second, differently
    scoped one. `labelmaker_sha` is HEAD at RUN time, which is the PARENT of
    the commit that carries the amended record - the same convention as the
    full run's.

    RUN THIS ON AN IDLE NODE. `elapsed_s` is wall clock on whatever the
    node was doing at the time, and the suite asserts the recorded number
    is under a second (`test_events_heuristics.py`), so a re-run taken on a
    busy login node commits a timing that has nothing to do with the port.
    """
    record = json.loads(record_path.read_text())
    corpus_file = corpus / Path(record["corpus_file"]).name
    shot = int(record["shot"])
    t_s, y = read_ece(corpus_file, record["max_seconds"])
    span = (float(t_s[0]), float(t_s[-1]))
    start = time.perf_counter()
    events = heuristics.sawtooth_events(y, t_s, shot=shot, t_cov=span)
    elapsed = time.perf_counter() - start
    ours_ms = np.array([e.t0_s for e in events], dtype=np.float64) * 1e3
    stats = periods_ms(ours_ms * 1e-3)
    was_ms = np.array([r["port_ms"] for r in record["crashes"]],
                      dtype=np.float64)
    rows = pair(was_ms, ours_ms)
    deltas = np.array([r["delta_ms"] for r in rows], dtype=np.float64)
    record["port_rerun"] = {
        "labelmaker_sha": git_sha(REPO),
        "dirty": git_dirty(REPO, "src/labelmaker"),
        "n": stats["n"],
        "median_ms": stats["median_period_ms"],
        "elapsed_s": elapsed,
        "max_abs_dt_ms": float(np.abs(deltas).max()) if deltas.size else float("nan"),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"shot          {shot}  {corpus_file}")
    print(f"port re-run   {stats['n']:4d} crashes in {elapsed:8.3f} s   "
          f"median {stats['median_period_ms']:.1f} ms")
    print(f"recorded      {record['reference']['n']:4d} reference crashes   "
          f"max |dt| {record['port_rerun']['max_abs_dt_ms']:.3e} ms")
    print(f"tree          {record['port_rerun']['labelmaker_sha'][:12]}"
          f"{'  DIRTY' if record['port_rerun']['dirty'] else '  clean'}")
    print(f"amended       {record_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shot", type=int, default=198658)
    parser.add_argument("--corpus", type=Path, default=CORPUS,
                        help="the processed-HDF5 corpus directory")
    parser.add_argument(
        "--omnimode-src", type=Path, default=OMNIMODE_SRC,
        help="omnimode's src/ directory, for the reference implementation",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=None,
        help="compare only the first N seconds of the record; the default "
             "is the WHOLE record, which is the comparison that counts and "
             "the one the reference takes minutes over",
    )
    parser.add_argument("--json", nargs="?", type=Path, const=OUT, default=None,
                        help=f"write the record (default {OUT})")
    parser.add_argument(
        "--port-only", action="store_true",
        help="re-run only the port, on the shot the committed record names, "
             "and amend that record with a `port_rerun` stanza; omnimode is "
             "not imported and not run. Run it on an IDLE node: the stanza's "
             "`elapsed_s` is wall clock and the suite asserts it",
    )
    args = parser.parse_args(argv)

    if args.port_only:
        return port_rerun(args.json or OUT, args.corpus)

    corpus_file = args.corpus / f"{args.shot}_processed.h5"
    t_s, y = read_ece(corpus_file, args.max_seconds)
    span = (float(t_s[0]), float(t_s[-1]))
    print(f"shot          {args.shot}  {corpus_file}")
    print(f"ece           {y.shape[0]} x {y.shape[1]}  "
          f"{span[0]:.4f} to {span[1]:.4f} s")

    start = time.perf_counter()
    events = heuristics.sawtooth_events(y, t_s, shot=args.shot, t_cov=span)
    port_s = time.perf_counter() - start
    ours_ms = np.array([e.t0_s for e in events], dtype=np.float64) * 1e3

    start = time.perf_counter()
    ref_ms = np.array(
        reference_find_sawteeth(y, t_s * 1e3, args.omnimode_src),
        dtype=np.float64,
    )
    ref_s = time.perf_counter() - start

    ours = periods_ms(ours_ms * 1e-3)
    ref = periods_ms(ref_ms * 1e-3)
    rows = pair(ref_ms, ours_ms)
    deltas = np.array([r["delta_ms"] for r in rows], dtype=np.float64)
    max_abs = float(np.abs(deltas).max()) if deltas.size else float("nan")
    same_count = bool(ours_ms.size == ref_ms.size)
    # `AGREE_MS` is not a tolerance on the physics - it is below the last
    # bit of a millisecond-grid time a few seconds into the shot. The
    # residual it forgives is the REFERENCE's own arithmetic: its crash
    # times are `t[0] + i + 0.5` in floating point, and ours are the same
    # sum in a different order, so a handful of crashes differ by a
    # femtosecond. Two crashes that are actually different crashes are a
    # millisecond apart at the very least.
    agree = bool(same_count and deltas.size and max_abs <= AGREE_MS)
    identical = bool(same_count and deltas.size and np.all(deltas == 0.0))

    print(f"port          {ours['n']:4d} crashes in {port_s:8.3f} s   "
          f"median {ours['median_period_ms']:.1f} ms  "
          f"mean {ours['mean_period_ms']:.1f} ms")
    print(f"reference     {ref['n']:4d} crashes in {ref_s:8.3f} s   "
          f"median {ref['median_period_ms']:.1f} ms  "
          f"mean {ref['mean_period_ms']:.1f} ms")
    print(f"speedup       {ref_s / max(port_s, 1e-12):.0f}x")
    print(f"same crashes  {agree}   (bitwise {identical})   "
          f"max |delta| {max_abs:.3e} ms")
    for row in rows:
        if abs(row["delta_ms"]) > AGREE_MS:
            print(f"  reference {row['reference_ms']:10.3f} ms  ->  port "
                  f"{row['port_ms']:10.3f} ms   delta {row['delta_ms']:+.3f}")

    if args.json is not None:
        record = {
            "shot": int(args.shot),
            "corpus_file": str(corpus_file),
            "span_s": list(span),
            "max_seconds": args.max_seconds,
            "n_samples": int(y.shape[1]),
            "port": {**ours, "elapsed_s": port_s},
            "reference": {**ref, "elapsed_s": ref_s},
            "speedup": ref_s / max(port_s, 1e-12),
            "same_count": same_count,
            "agree": agree,
            "agree_tolerance_ms": AGREE_MS,
            "bitwise_identical": identical,
            "max_abs_delta_ms": max_abs,
            "crashes": rows,
            "labelmaker_sha": git_sha(REPO),
            "omnimode_sha": git_sha(args.omnimode_src.parent),
            "script": Path(__file__).name,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote         {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
