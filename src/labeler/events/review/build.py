"""Build the review store: one HDF5 file of rows per shot.

    python -m labeler.events.review.build --event alfven_eigenmode [--shots ...]
        [--workers 8] [--force]

AE rows come from the Heidbrink recipe (`alfven.py`); every other event's
from its panel builder (`panel_rows.py`). The review server calls `build`
itself for a shot that has no file yet. The frame models build their legacy
shots' stores with `out=` (`labeler.frames.prepare`), outside `spectrograms/`.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from ...config import Paths, git_sha
from .. import rosters
from . import alfven, panel_rows, rows

BUILDERS = {"alfven_eigenmode": alfven.build}
# Rows an older build wrote that the review no longer shows, until it is rebuilt.
HIDDEN = {"alfven_eigenmode": alfven.DROPPED}


def build(
    event: str, shot: int, paths: Paths | None = None, *, force: bool = False,
    out: Path | None = None,
) -> Path:
    """Build one shot's rows file and return its path; a file already there is
    kept unless `force`.

    The file is the review's (`spectrogram_file`), or `out/<shot>.h5` with `out`,
    which the review page never serves, so `out` may not be under `spectrograms/`.
    """
    paths = Paths.from_env() if paths is None else paths
    if out is None:
        path = paths.spectrogram_file(event, shot)
    else:
        served, where = paths.spectrograms.resolve(), Path(out).resolve()
        if where == served or served in where.parents:
            raise ValueError(f"{out}: a store built with out= is never under {served}")
        path = Path(out) / f"{int(shot)}.h5"
    if path.is_file() and not force:
        return path
    builder = BUILDERS.get(event, panel_rows.build)
    grid, built, info = builder(event, int(shot), paths)
    rows.write(
        path, grid, built, event=event, shot=int(shot),
        builder=builder.__module__.rsplit(".", 1)[-1], **info,
        made_at=datetime.now(UTC).isoformat(timespec="seconds"), git_sha=git_sha(),
    )
    return path


def _timed(event: str, shot: int, force: bool) -> tuple[int, float, str | None]:
    started = time.monotonic()
    try:
        build(event, shot, force=force)
    except Exception as error:  # noqa: BLE001 - one bad shot must not stop the rest
        return shot, time.monotonic() - started, f"{type(error).__name__}: {error}"
    return shot, time.monotonic() - started, None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m labeler.events.review.build",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--event", required=True)
    parser.add_argument("--shots", nargs="*", type=int, help="default: the roster")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="rebuild existing files")
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    roster = rosters.roster_path(args.event, root=paths.label_tables)
    shots = args.shots or list(rosters.read_roster(roster).shot)
    todo = [
        int(shot) for shot in shots
        if args.force or not paths.spectrogram_file(args.event, shot).is_file()
    ]
    print(f"{args.event}: {len(todo)} of {len(shots)} shots to build", flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_timed, args.event, shot, args.force) for shot in todo]
        for future in as_completed(futures):
            shot, seconds, error = future.result()
            failed += error is not None
            print(f"{shot} {seconds:.1f}s {error or 'ok'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
