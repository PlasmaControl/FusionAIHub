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
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from ...config import Paths, git_sha, store_event
from .. import panels, rosters, taxonomy
from . import alfven, detachment, panel_rows, rows

BUILDERS = {"alfven_eigenmode": alfven.build, "detachment": detachment.build}
# Rows an older build wrote that the review no longer shows, until it is rebuilt.
HIDDEN = {"alfven_eigenmode": alfven.DROPPED}
# The band an event's image rows show, whatever band the stored rows were built with.
BANDS = {"alfven_eigenmode": alfven.BAND_KHZ}
# Changes to these diagnostic recipes replace earlier cached review rows. Keyed by
# the STORE event (`config.store_event`): `confinement` is `high_confinement_mode`,
# `mhd_n2` is `mhd_n1`. The edge-Te row came to the H-mode panels with 1.
# `detachment` 7: the row titles changed (qualifiers moved to the caveats);
# `resistive_wall_mode` 1: its rows were retitled the same way. The folded events
# (sawtooth, fishbone, NTM) are stored under their own parent and are not here.
PANEL_VERSIONS = {
    "edge_localized_mode": 4, "sawtooth_oscillation": 1, "detachment": 7,
    "high_confinement_mode": 1, "mhd_n1": 1, "mhd_n1.m1": 1,
    "resistive_wall_mode": 1,
}
# Recipe 7 has not been delivered yet. Reject early recipe-7 stores lacking
# the independent-review target/bolometer context introduced in the fix round
# before it (revision 1 of recipe 6, kept for recipe 7's stores).
PANEL_REVISIONS = {"detachment": 1}


def current(path: Path, event: str) -> bool:
    """Whether a review store exists and has the event's current panel recipe."""
    if not path.is_file():
        return False
    event = store_event(event)
    version = PANEL_VERSIONS.get(event)
    if version is None:
        return True
    import h5py

    with h5py.File(path, "r") as store:
        return store.attrs.get("panel_version") == version and (
            event not in PANEL_REVISIONS
            or store.attrs.get("panel_revision") == PANEL_REVISIONS[event]
        )


def _markers(store: str, shot: int, grid, paths: Paths) -> dict:
    """The store's marker sets, as the `markers` file attribute.

    `panels.MARKERS[store]`, when a panel module registers one, is
    `markers(shot, *, t_range, paths) -> [{"label": str, "t_ms": [float, ...]}]`:
    thin ticks the page draws over every row. A store without a markers function
    gets no attribute and reads back as `[]`. A markers function that fails fails
    the build, like any panel: a store never keeps `[]` for markers it could not read.
    """
    found = getattr(panels, "MARKERS", {}).get(store)
    if found is None:
        return {}
    t_range = (grid.t0_ms, grid.t0_ms + grid.n * grid.dt_ms)
    sets = [
        {"label": str(item["label"]),
         "t_ms": sorted(float(t) for t in item["t_ms"] if math.isfinite(float(t)))}
        for item in found(int(shot), t_range=t_range, paths=paths)
    ]
    return {"markers": sets}


def build(
    event: str, shot: int, paths: Paths | None = None, *, force: bool = False,
    out: Path | None = None,
) -> Path:
    """Build one shot's rows file and return its path; a file already there is
    kept unless `force` or its diagnostic recipe has changed.

    The file is the review's (`spectrogram_file`), or `out/<shot>.h5` with `out`,
    which the review page never serves, so `out` may not be under `spectrograms/`.
    The rows, recipe version and markers are those of the event's store
    (`config.store_event`): `mhd_n2` builds as `mhd_n1`, `alfven_eigenmode.rsae` as
    `alfven_eigenmode`.
    """
    paths = Paths.from_env() if paths is None else paths
    if out is None:
        path = paths.spectrogram_file(event, shot)
    else:
        served, where = paths.spectrograms.resolve(), Path(out).resolve()
        if where == served or served in where.parents:
            raise ValueError(f"{out}: a store built with out= is never under {served}")
        path = Path(out) / f"{int(shot)}.h5"
    if not force and (path.is_file() if out is not None else current(path, event)):
        return path
    store = store_event(event)
    builder = BUILDERS.get(store, panel_rows.build)
    grid, built, info = builder(store, int(shot), paths)
    rows.write(
        path, grid, built, event=store, shot=int(shot),
        builder=builder.__module__.rsplit(".", 1)[-1], **info,
        **_markers(store, shot, grid, paths),
        **({"panel_version": PANEL_VERSIONS[store]}
           if store in PANEL_VERSIONS and out is None else {}),
        **({"panel_revision": PANEL_REVISIONS[store]}
           if store in PANEL_REVISIONS and out is None else {}),
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
    # A subcategory is labelled over its parent's roster.
    parent = taxonomy.load(paths.events_manifest).parent(args.event)
    roster = rosters.roster_path(parent or args.event, root=paths.label_tables)
    shots = args.shots or list(rosters.read_roster(roster).shot)
    todo = [
        int(shot) for shot in shots
        if args.force or not current(paths.spectrogram_file(args.event, shot), args.event)
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
