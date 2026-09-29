"""The legacy shots' stores and every shot's features, for the frame models.

    python -m labeler.frames.prepare --method M --stores [--shots ...] [--limit N]
        [--force] [--index I --count N] [--workers W]
    python -m labeler.frames.prepare --method M [--shots ...] [--limit N]
        [--index I --count N] [--workers W]

**Shots.** The method's shots file's (`frames.shots_file`, every split), or
`--shots`. Shard I of N (`shard`) takes every N-th of them in shot order from
the I-th, and `--limit` the shard's first N. I and N default to
SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT (D67), so each task of an array
is its own shard.

**Stores** (`--stores`, `build_stores`). Each shot outside the store event's
roster gets its store built under `frames/stores/<store_event>/`
(`frames.store_path`) by `review.build.build(out=)`, never into `spectrograms/`,
which the review page serves; nothing is fetched under LABELER_NO_FETCH=1. A
roster shot's store is the review's and is left as it is. A store already built
is kept unless `--force`, or unless it cannot be opened, or it is stale (a
tearing-mode store from before d9fb57d, `features.StaleStore`, D65): those are
rebuilt, never read.

**Features** (`prepare`). Each shot's `features.features` over the whole bins
inside its window (the split's, `frames.shots_meta_file`'s `windows`) go to
`features_dir/<shot>.npz`:
- `x`, `(C, n_sub)` float16 in [0, 1]: the sub-frames of those bins;
- `observed`, `(n_frames,)`: each 10 ms frame's;
- `bins`, `(n_bins,)`: each bin's start, ms; `states`, `(n_bins,)` int8: its
  target (`targets.target_bins`; for the owner split, the owner's label as the
  split froze it, `frames.owner_file`, D40), UNKNOWN where the target has no bin;
- `first`, the first frame, and `window`, the window in ms.

A shot whose features cannot be made is dropped: its npz is removed and its
reason written to `features_dir/<shot>.dropped.json` (`dropped` reads them all).
The reasons: not in the split, no target, no whole bin in the window, no store,
a stale store, a store that cannot be read, and no observed frame (a store whose
required rows hold nothing inside the window, as sawtooth 187154's ECE).

**Records.** Each run writes what it did shot by shot, with its shard, to
`features_dir/records/<I>-of-<N>.json`, or for `--stores` to
`frames/stores/<store_event>/records/<method>-<I>-of-<N>.json`. The command
exits 1 when a store could not be built (as `review.build` does), else 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from multiprocessing import get_context

import h5py
import numpy as np
import pandas as pd

from ..ae.xpower.data import window_frames
from ..config import Paths, atomic_path, git_sha
from ..events.review import build as review_build
from ..events.review import labels
from ..scoring.frames import FRAME_MS
from . import (
    SPECS,
    EventSpec,
    features_dir,
    owner_file,
    roster_shots,
    shots_file,
    shots_meta_file,
    store_path,
    stores_dir,
    targets,
)
from .features import StaleStore, features, match_roles
from .targets import UNKNOWN

log = logging.getLogger(__name__)

#: A dropped shot's reason sits beside where its npz would be.
DROPPED = ".dropped.json"


def shard(shots, index: int, count: int) -> list[int]:
    """Shard `index` of `count`: every `count`-th of the distinct `shots`, in
    order, from the `index`-th; the shards cover each shot once."""
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard {index} of {count}: need 0 <= index < count")
    return sorted({int(shot) for shot in shots})[index::count]


def split_shots(paths: Paths, method: str) -> dict[int, str]:
    """The method's shots file, shot -> split; raises before the split is made."""
    path = shots_file(paths, method)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: no split; run python -m labeler.frames.shots --method {method}"
        )
    frame = pd.read_csv(path)
    return {int(s): str(v) for s, v in zip(frame.shot, frame.split, strict=True)}


def _described(path) -> list[dict]:
    """A store's row descriptions, as `features` reads them."""
    with h5py.File(path, "r") as f:
        return [
            {"name": name, **json.loads(f["rows"][name].attrs["meta"])}
            for name in json.loads(f.attrs["rows"])
        ]


def store_state(path, spec: EventSpec) -> str | None:
    """Why a built store must be built again: "missing", "unreadable" or "stale"
    (D65); None for one to keep. A store without a required row is kept: its
    features drop it, and `--force` rebuilds it after a panel change."""
    if not path.is_file():
        return "missing"
    try:
        described = _described(path)
    except (OSError, KeyError, ValueError):
        return "unreadable"
    try:
        match_roles(described, spec, str(path))
    except StaleStore:
        return "stale"
    except ValueError:
        return None
    return None


def _run(work, jobs: list, workers: int):
    """`work(job)` for each job, in order; spawned workers when more than one."""
    if workers <= 1 or len(jobs) <= 1:
        yield from map(work, jobs)
        return
    context = get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        yield from pool.map(work, jobs)


def _build_one(job) -> tuple[int, float, str | None]:
    event, shot, paths, out = job
    started = time.monotonic()
    try:
        review_build.build(event, shot, paths, force=True, out=out)
    except Exception as error:  # noqa: BLE001 - one bad shot must not stop the rest
        return shot, time.monotonic() - started, f"{type(error).__name__}: {error}"
    return shot, time.monotonic() - started, None


def build_stores(paths: Paths, method: str, shots, *, force=False, workers=1) -> dict:
    """Build the stores of `shots` outside the roster (module docstring); what was
    done, shot by shot: `roster`, `kept`, `built`, `rebuilt` (shot -> why) and
    `failed` (shot -> the error), and each build's `seconds`."""
    spec = SPECS[method]
    roster = roster_shots(paths, spec.store_event)
    out = stores_dir(paths, spec)
    wanted = sorted({int(shot) for shot in shots})
    record = {
        "method": method,
        "store_event": spec.store_event,
        "shots": wanted,
        "roster": [],
        "kept": [],
        "built": [],
        "rebuilt": {},
        "failed": {},
        "seconds": {},
    }
    why = {}
    for shot in wanted:
        if shot in roster:
            record["roster"].append(shot)
            continue
        reason = "forced" if force else store_state(store_path(paths, spec, shot), spec)
        if reason is None:
            record["kept"].append(shot)
        else:
            why[shot] = reason
    jobs = [(spec.store_event, shot, paths, out) for shot in why]
    for shot, seconds, error in _run(_build_one, jobs, workers):
        record["seconds"][str(shot)] = round(seconds, 2)
        if error is not None:
            log.warning("%s: shot %s: no store: %s", method, shot, error)
            record["failed"][str(shot)] = error
        elif why[shot] == "missing":
            record["built"].append(shot)
        else:
            record["rebuilt"][str(shot)] = why[shot]
    return record


def _drop(folder, shot: int, reason: str) -> tuple[int, str]:
    """Record why `shot` has no features, and remove any it had."""
    (folder / f"{shot}.npz").unlink(missing_ok=True)
    with atomic_path(folder / f"{shot}{DROPPED}") as tmp:
        tmp.write_text(json.dumps({"shot": shot, "reason": reason}) + "\n")
    return shot, reason


def dropped(paths: Paths, method: str) -> dict[int, str]:
    """Every dropped shot's reason, by shot."""
    found = {}
    for path in sorted(features_dir(paths, method).glob(f"*{DROPPED}")):
        record = json.loads(path.read_text())
        found[int(record["shot"])] = record["reason"]
    return found


def _aligned(starts, states, k0: int, k1: int, bin_ms: float) -> np.ndarray:
    """The target's states on bins `k0 .. k1 - 1`; UNKNOWN where it has none."""
    k = np.rint(np.asarray(starts, dtype=np.float64) / bin_ms).astype(np.int64)
    states = np.asarray(states, dtype=np.int8)
    out = np.full(k1 - k0, UNKNOWN, dtype=np.int8)
    inside = (k >= k0) & (k < k1)
    out[k[inside] - k0] = states[inside]
    return out


def bin_range(window, bin_ms: float) -> tuple[int, int]:
    """`(k0, k1)`: the whole bins `k0 .. k1 - 1` of `bin_ms` inside the whole 10 ms
    frames of `window`."""
    per = round(bin_ms / FRAME_MS)
    first, n = window_frames(window)
    return -(-first // per), (first + n) // per


def _features_one(job) -> tuple[int, str | None]:
    paths, method, shot, window, target = job
    spec = SPECS[method]
    folder = features_dir(paths, method)
    per = round(spec.bin_ms / FRAME_MS)
    k0, k1 = bin_range(window, spec.bin_ms)
    if k1 <= k0:
        return _drop(folder, shot, "no whole bin in the window")
    path = store_path(paths, spec, shot)
    if not path.is_file():
        return _drop(folder, shot, "no store")
    try:
        x, observed = features(path, spec, (k0 * per * FRAME_MS, k1 * per * FRAME_MS))
    except StaleStore as error:
        return _drop(folder, shot, f"stale store: {error}")
    except (OSError, KeyError, ValueError) as error:
        return _drop(folder, shot, f"unusable store: {type(error).__name__}: {error}")
    if not observed.any():
        return _drop(folder, shot, "no observed frame")
    starts, states = target
    bins = (k0 + np.arange(k1 - k0)) * float(spec.bin_ms)
    arrays = {
        "x": x.astype(np.float16),
        "observed": observed,
        "bins": bins,
        "states": _aligned(starts, states, k0, k1, spec.bin_ms),
        "first": np.int64(k0 * per),
        "window": np.asarray(window, dtype=np.int64),
    }
    with atomic_path(folder / f"{shot}.npz") as tmp, open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    (folder / f"{shot}{DROPPED}").unlink(missing_ok=True)
    return shot, None


def prepare(paths: Paths, method: str, shots, *, workers: int = 1) -> dict:
    """Write each of `shots`' features (module docstring); what was done: the
    shots `written`, and the `dropped` ones with their reasons."""
    spec = SPECS[method]
    windows = json.loads(shots_meta_file(paths, method).read_text())["windows"]
    split = split_shots(paths, method)
    wanted = sorted({int(shot) for shot in shots})
    folder = features_dir(paths, method)
    folder.mkdir(parents=True, exist_ok=True)
    saved = None
    record = {"method": method, "shots": wanted, "written": [], "dropped": {}}
    jobs = []
    for shot in wanted:
        if shot not in split or str(shot) not in windows:
            _drop(folder, shot, "not in the split")
            record["dropped"][str(shot)] = "not in the split"
            continue
        lo, hi = windows[str(shot)][:2]
        try:
            if split[shot] == "owner":
                if saved is None:
                    saved = labels.read_labels(owner_file(paths, method))
                target = targets.label_bins(saved[shot], spec.bin_ms)
            else:
                target = targets.target_bins(paths, spec, shot)
        except (OSError, ValueError, KeyError) as error:
            reason = f"no target: {type(error).__name__}: {error}"
            record["dropped"][str(shot)] = _drop(folder, shot, reason)[1]
            continue
        jobs.append((paths, method, shot, (int(lo), int(hi)), target))
    for shot, reason in _run(_features_one, jobs, workers):
        if reason is None:
            record["written"].append(shot)
        else:
            record["dropped"][str(shot)] = reason
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--method", required=True, choices=list(SPECS))
    p.add_argument(
        "--stores", action="store_true", help="build the legacy shots' stores"
    )
    p.add_argument("--shots", type=int, nargs="*", help="default: the shots file's")
    p.add_argument("--limit", type=int, default=0, help="the shard's first N shots")
    p.add_argument(
        "--force", action="store_true", help="with --stores, rebuild built stores"
    )
    p.add_argument(
        "--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    p.add_argument(
        "--count", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    )
    p.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    args = p.parse_args(argv)
    if args.limit < 0:
        p.error("--limit must be nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = Paths.from_env()
    try:
        shots = args.shots or sorted(split_shots(paths, args.method))
        mine = shard(shots, args.index, args.count)
    except (OSError, ValueError) as error:
        p.error(str(error))
    if args.limit:
        mine = mine[: args.limit]
    started = time.monotonic()
    spec = SPECS[args.method]
    if args.stores:
        record = build_stores(
            paths, args.method, mine, force=args.force, workers=args.workers
        )
        out = stores_dir(paths, spec) / "records"
        name = f"{args.method}-{args.index}-of-{args.count}.json"
    else:
        record = prepare(paths, args.method, mine, workers=args.workers)
        out = features_dir(paths, args.method) / "records"
        name = f"{args.index}-of-{args.count}.json"
    record |= {
        "index": args.index,
        "count": args.count,
        "limit": args.limit,
        "workers": args.workers,
        "wall_seconds": round(time.monotonic() - started, 1),
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with atomic_path(out / name) as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    counts = {
        k: len(v)
        for k, v in record.items()
        if k in ("roster", "kept", "built", "rebuilt", "failed", "written", "dropped")
    }
    line = {"method": args.method, "stores": args.stores, "shots": len(mine), **counts}
    print(json.dumps(line | {"record": str(out / name)}), flush=True)
    return 1 if record.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
