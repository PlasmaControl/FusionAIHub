#!/usr/bin/env python
"""Fetch named canonical features through fdp for shots whose feature file lacks them, and merge.

WHY THIS EXISTS. `python -m labeler.run features --models ...` resolves only the features the
named models consume. `shot_design` reads more than that: `corpus select`'s rule (d) needs `ip`, and
`shot_design build --reader corpus` addresses the EFIT/plasma scalars `configs/shot_design/signals.yaml`
names in its `labelmaker: {feature: ...}` blocks -- `bt betan kappa tritop tribot li qmin gapin
aminor volume r0 pcbcoil ne_zipfit te_zipfit`. None of those is an input of any shipped model, so
no features job ever fetches them, and I6b's build reported 4,615 (shot, field) pairs `pending`
for exactly that reason. This script closes that gap and nothing else.

It began as `fetch_ip_features.py` (one feature, hard-coded) and is the same script generalised:
`--features NAME [NAME ...]`, defaulting to `ip`, so the invocation that fetched `ip` for the 500
shots of `recommender_v1` still reads the same but for the file name.

WHERE IT RUNS. The LOGIN node, not SLURM. The compute nodes have no route to DIII-D's PTDATA
server, so this is not a batch job and there are no jobstats for it. And it must run under the
`fdp run` wrapper, which supplies the server configuration:

    pixi run -e labelmaker fdp run python scripts/labeler/fetch_features.py \
        --shot-file $LABELER_ROOT/recommender_v1.txt \
        --features bt betan kappa tritop tribot li qmin gapin aminor volume r0 pcbcoil \
                   ne_zipfit te_zipfit \
        --workers 4 --retries 5

Without the wrapper every PTDATA fetch fails with `PtDataError` and the process reports
`getservbyname failed for task 'PTSERVER'` -- see `labeler.features.store.TRANSIENT_CAUSES`,
where that trap is recorded, and `resolve_fdp`'s module docstring.

HOW IT IS SAFE. `store.write_features(..., merge=True)` keeps every group already in the file and
drops a name from the recorded misses when it resolves; it writes through a temporary file and
renames, so a file is never left half-written.

**This script never CREATES a feature file** -- `select.preferred_shots` and the corpus census
both read the mere existence of one as "this shot has features", so a file holding nothing but
what this run fetched would promote a shot labeler has never featured. The `path.exists()`
guard is therefore on EVERY path, the successful one included, and it is checked before the
network call rather than only before the write: there is nothing this run may do for such a shot,
so spending a fetch on it is waste. Those shots are reported `failed / no features file`.

WHAT COUNTS AS MISSING. A feature is fetched when the file has no group for it, or has one the
consumer could not use -- fewer than two samples is the corpus' "signal absent" sentinel -- or
holds it only as a recorded miss (a miss leaves no group, so the same test catches it). So a
rerun over the same list is nearly free and does retry what a previous run could not reach.

The worker pool is forked before toksearch is imported anywhere (its ptserver reader is not
fork-safe), which is why the labeler imports sit inside `_resolver()`/`_store()` and those are
called from `_fetch`. They are functions rather than a bare `from ... import` so a test can
substitute them without importing labeler at all. `--workers 1` skips the pool entirely and
runs in this process, which is what a smoke test and a debugger want.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from multiprocessing import Pool
from pathlib import Path

from labeler.env import getenv

#: What a bare invocation fetches -- what `fetch_ip_features.py` fetched.
DEFAULT_FEATURES = ("ip",)

#: The cause reported for a shot this script must not write a file for.
NO_FILE = "no features file (this script merges, it never creates one)"

# Set once in each worker by `_init`, because `Pool.imap_unordered` passes one argument and the
# alternative -- a tuple per shot -- would repeat the same three values once for every shot.
_FEATURES_DIR: Path | None = None
_FEATURES: tuple[str, ...] = DEFAULT_FEATURES
_RETRIES = 1


def read_shots(path: Path) -> list[int]:
    """One shot per line, `#` starting a comment. The pending file's own format."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        head = line.split("#", 1)[0].strip()
        if head:
            out.append(int(head))
    return out


def canonical(names: Iterable[str]) -> list[str]:
    """`names` deduplicated, in the order they were given.

    Asking for the same feature twice is not an error worth refusing a run over, but sending it
    to the resolver twice would fetch it twice.
    """
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def unresolvable(names: Sequence[str]) -> dict[str, str]:
    """`{name: why}` for every name fdp cannot be asked for, checked in the PARENT.

    `resolve_fdp.resolve` raises KeyError on a name with no fdp source -- correctly, it is a
    programming error and not a per-shot gap -- but it raises it inside a worker, once per shot,
    after the pool has been forked and the wrapper has connected. A typo in a flag belongs on the
    first line of output instead. `labeler.features.namespace` is a pure table: importing it
    here does not import toksearch, so the fork rule above is untouched.
    """
    from labeler.features import namespace as ns

    bad: dict[str, str] = {}
    for name in names:
        try:
            ns.by_name(name).locator_for("fdp")
        except KeyError as exc:
            bad[name] = str(exc).strip("'\"")
    return bad


def present_features(path: Path, names: Sequence[str]) -> set[str]:
    """Which of `names` the file already holds in a form a consumer could use.

    The test is the CONSUMER's, not the store's: `shot_design.shotdb.select.measured_flattop` wants
    `<name>/xdata` and `<name>/ydata` with something in them, so a group that exists but carries
    one sample -- the corpus' "signal absent" sentinel, and what `store.write_features` demotes a
    degenerate fetch into -- is not a feature for this purpose. A recorded miss leaves no group
    at all, so the same test catches it and the feature is fetched again.
    """
    if not path.exists():
        return set()
    import h5py

    try:
        with h5py.File(path, "r", locking=False) as f:
            return {
                name
                for name in names
                if name in f
                and "xdata" in f[name]
                and "ydata" in f[name]
                and f[name]["ydata"].shape[-1] >= 2
            }
    except OSError:
        return set()


def _init(features_dir: str, features: Sequence[str], retries: int) -> None:
    global _FEATURES_DIR, _FEATURES, _RETRIES
    _FEATURES_DIR = Path(features_dir)
    _FEATURES = tuple(features)
    _RETRIES = retries


def _resolver():
    """labeler's fdp resolver, imported HERE and not at module scope.

    The parent process must never import toksearch -- its ptserver reader is not fork-safe, and
    the pool below is forked. Deferring the import into a function called from the worker keeps
    that true and, as a side effect, lets a test substitute the resolver without labeler.
    """
    from labeler.features import resolve_fdp

    return resolve_fdp


def _store():
    """labeler's feature-file writer. Deferred for the same reason as `_resolver`."""
    from labeler.features import store

    return store


def _row(shot: int, secs: float = 0.0) -> dict:
    """One shot's report: a verdict per requested feature, plus the causes and sample counts."""
    return {"shot": shot, "secs": secs, "status": {}, "causes": {}, "n": {}}


def _fetch(shot: int) -> dict:
    """One shot: fetch every requested feature the file lacks, and merge them in with one write.

    One `resolve` call for all of them, because `resolve_fdp` caches a shot's time axes across
    the features of a single call and asking per feature would re-fetch each axis.
    """
    path = _FEATURES_DIR / f"{shot}_features.h5"
    t0 = time.monotonic()
    have = present_features(path, _FEATURES)
    row = _row(shot)
    for name in _FEATURES:
        if name in have:
            row["status"][name] = "present"
    want = [name for name in _FEATURES if name not in have]
    if not want:
        return row
    if not path.exists():
        for name in want:
            row["status"][name], row["causes"][name] = "failed", NO_FILE
        return row

    resolve_fdp, store = _resolver(), _store()
    try:
        arrays, missing = resolve_fdp.resolve(shot, want, retries=_RETRIES)
    except Exception as exc:  # noqa: BLE001 - one shot's failure is not the run's
        cause = f"{type(exc).__name__}: {exc}"[:120]
        for name in want:
            row["status"][name], row["causes"][name] = "failed", cause
        row["secs"] = time.monotonic() - t0
        return row
    row["secs"] = time.monotonic() - t0

    got, miss = {}, {}
    for name in want:
        arr = arrays.get(name)
        if arr is None:
            miss[name] = str(missing.get(name, "NoArrayNoCause"))[:120]
        elif arr.y.shape[-1] < 2:
            # The store demotes a group of fewer than two samples into `missing` rather than
            # writing the corpus' "signal absent" sentinel as a resolved feature. Said here too,
            # so the table this run prints agrees with the file it just wrote.
            miss[name] = f"OneSampleAmbiguous({arr.y.shape[-1]})"
        else:
            got[name] = arr
    if got or miss:
        try:
            store.write_features(path, shot, got, miss, merge=True)
        except Exception as exc:  # noqa: BLE001
            cause = f"write: {type(exc).__name__}: {exc}"[:120]
            for name in want:
                row["status"][name], row["causes"][name] = "failed", cause
            return row
    for name, arr in got.items():
        row["status"][name], row["n"][name] = "fetched", int(arr.y.shape[-1])
    for name, cause in miss.items():
        row["status"][name], row["causes"][name] = "failed", cause
    return row


def _line(i: int, total: int, row: dict, features: Sequence[str]) -> str:
    """One shot's progress line: what was fetched, with sample counts, and what was not."""
    fetched = [f"{n}({row['n'][n]:,})" for n in features if row["status"].get(n) == "fetched"]
    failed = [f"{n}:{row['causes'][n]}" for n in features if row["status"].get(n) == "failed"]
    parts = [f"[{i:>4}/{total}] {row['shot']} {row['secs']:5.1f}s"]
    if fetched:
        parts.append("fetched " + " ".join(fetched))
    if failed:
        parts.append("FAILED " + " ".join(failed))
    if not fetched and not failed:
        parts.append("nothing to do")
    return "  ".join(parts)


def _table(rows: Sequence[dict], features: Sequence[str]) -> list[str]:
    """The per-feature summary: `fetched / present / failed` over every shot of the run."""
    counts = {name: Counter(r["status"].get(name, "present") for r in rows) for name in features}
    width = max([len("feature"), *(len(n) for n in features)])
    out = [f"{'feature':<{width}}  {'fetched':>8}  {'present':>8}  {'failed':>8}"]
    for name in features:
        c = counts[name]
        out.append(
            f"{name:<{width}}  {c['fetched']:>8,}  {c['present']:>8,}  {c['failed']:>8,}"
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shot-file", required=True, help="one shot per line")
    ap.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURES),
        metavar="NAME",
        help="canonical feature names (labeler.features.namespace); default: ip",
    )
    ap.add_argument("--features-dir", help="default: $LABELER_ROOT/features")
    ap.add_argument(
        "--workers", type=int, default=4, help="network-bound; 4 is polite. 1 skips the pool"
    )
    ap.add_argument("--retries", type=int, default=5, help="per-signal retries inside resolve()")
    ap.add_argument("--limit", type=int, help="only the first N shots (a smoke test)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    features = canonical(args.features)

    root = getenv("LABELER_ROOT")
    features_dir = (
        Path(args.features_dir)
        if args.features_dir
        else (Path(root) / "features" if root else None)
    )
    if features_dir is None:
        print("no --features-dir and no $LABELER_ROOT", file=sys.stderr)
        return 2
    if not features_dir.is_dir():
        print(f"no feature store at {features_dir}", file=sys.stderr)
        return 2
    bad = unresolvable(features)
    if bad:
        for name, why in bad.items():
            print(f"{name}: not a canonical feature fdp can be asked for ({why})", file=sys.stderr)
        return 2

    shots = read_shots(Path(args.shot_file))
    if args.limit:
        shots = shots[: args.limit]
    # Checked in the parent, before the fork: a shot that already has everything costs no worker
    # and no network call, and the printed table then separates "was already there" from
    # "fetched now" without either of them having to be inferred.
    done, todo = [], []
    for shot in shots:
        have = present_features(features_dir / f"{shot}_features.h5", features)
        (done if len(have) == len(features) else todo).append(shot)
    print(
        f"{len(shots):,} shot(s) from {args.shot_file}, {len(features)} feature(s) "
        f"({' '.join(features)}): {len(done):,} complete, {len(todo):,} to fetch, "
        f"{args.workers} worker(s), {args.retries} retries",
        flush=True,
    )

    rows = [_row(shot) for shot in done]
    for row in rows:
        row["status"] = dict.fromkeys(features, "present")
    t0 = time.monotonic()
    if todo and args.workers > 1:
        with Pool(
            args.workers, initializer=_init, initargs=(str(features_dir), features, args.retries)
        ) as pool:
            for i, row in enumerate(pool.imap_unordered(_fetch, todo), start=1):
                rows.append(row)
                print(_line(i, len(todo), row, features), flush=True)
    elif todo:
        # One worker is this process. A pool of one is a fork for no reason, and it is the shape
        # a smoke test, a debugger and the tests below want.
        _init(str(features_dir), features, args.retries)
        for i, shot in enumerate(todo, start=1):
            row = _fetch(shot)
            rows.append(row)
            print(_line(i, len(todo), row, features), flush=True)

    verdicts = Counter(v for r in rows for v in r["status"].values())
    print(
        f"\n{len(shots):,} shot(s) x {len(features)} feature(s) in "
        f"{time.monotonic() - t0:.0f}s: {verdicts['fetched']:,} fetched, "
        f"{verdicts['present']:,} already present, {verdicts['failed']:,} failed"
    )
    print()
    print("\n".join(_table(rows, features)))
    causes = Counter(c for r in rows for c in r["causes"].values())
    if causes:
        print("\ncauses:")
        for cause, n in causes.most_common():
            print(f"  {n:>6,}  {cause}")
    # Any feature this run was asked for and did not get, transient or not. A partial run must
    # not look like a complete one to whatever wrapper called it; `labeler.features.store`'s
    # `is_transient` is what says which of the causes above a rerun would clear.
    return 1 if verdicts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
