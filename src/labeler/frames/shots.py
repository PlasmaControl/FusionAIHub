"""Each frame model's shots, their split and the owner's snapshot (spec §3.3).

A shot's target is its original table's bins (`targets.target_shots`; the
sawtooth's, the ece_sawtooth v3 detector's table, D56 as amended) with the
owner's saved label over them (`targets.merged`, F2), and the shots are the
original's with the owner's saved ones added. `eligible` takes them through its
checks cheapest first, so that the windows, the slowest, are only asked for the
shots every other check passed:
1. not blind (the cohort's blind shots);
2. a labelled bin: the target is not unknown in every bin (D60);
3. every required group on disk (`raw.record_tier`, never fetched); a corpus or
   cache file h5py cannot open is its own reason, since `record_tier` takes it
   for a missing group;
4. a window (`targets.window_for`, the catalog's windows read once) holding a
   labelled bin; no `ip.jsonl` window is needed (D66);
5. a store: a roster shot's review store must hold the spec's rows, each
   required trace with a value in the window (a cheap read, `_store_problem`);
   any other shot's is built under `frames/stores/` from the groups of 3, so
   its store is not read here, and a shot whose store then gives no observed
   frame is Task 2.8's to drop.

`make` freezes the owner's saves (`labels.labels_path`) as `frames.owner_file`,
with their sha256, and every target is merged with that copy, never the live
file. There is no owner split (F2 supersedes D40's "owner split, never trained
on"): every eligible shot is split 70/15/15 by shot within each (positive,
roster) stratum at `frames.SEED` (`split`), and the shots file's `owner` is 1
where the owner saved the shot. A shot's `positive` is whether its merged
target has a present bin in its window.

The meta (`frames.shots_meta_file`) holds D60's `labelled_shots` and
`positive_shots`, counted over every merged target, blind and left-out shots
too, and `legacy_labelled_shots`, the original's alone; `present_s`, the merged
targets' present bins times `bin_ms`, in s (F4); `owner`, the saved shots, those
also in the original ("overriding") and those it lacks ("added"); the counts and
positives by split; the owner's snapshot; the left-out shots by reason; each
shot's window and where it came from; and, for the shots whose window is their
target's hull, how many labelled bins lie outside the plasma's window in the
catalog's Ip log (`labels_window`). `paper.coverage` counts the frame
phenomena from this meta alone.

The files are `frames.VERSION`'s (F1); v1's, with their owner split, stay
where v1 wrote them.

    python -m labeler.frames.shots --method M [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime

import h5py
import numpy as np
import pandas as pd

from ..config import Paths, atomic_path, git_sha
from ..events import raw, spans
from ..events.catalog.check import CatalogError
from ..events.catalog.cohort import read_cohort
from ..events.catalog.window import read_log
from ..events.review import labels
from ..events.review.rows import LEVELS
from . import (
    SEED,
    SPECS,
    VERSION,
    EventSpec,
    model_dir,
    owner_file,
    roster_shots,
    shots_file,
    shots_meta_file,
    store_path,
    targets,
)
from .features import StaleStore, match_roles
from .targets import PRESENT_T, UNKNOWN

log = logging.getLogger(__name__)

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train, val, test; seed frames.SEED
SPLITS = ("train", "val", "test")
#: The shots file's columns, exactly; `owner` marks the owner's saved shots.
COLUMNS = ("shot", "split", "positive", "roster", "owner")
#: `eligible`'s columns.
ELIGIBLE_COLUMNS = (
    "shot",
    "owner",
    "positive",
    "roster",
    "window_start_ms",
    "window_end_ms",
    "window_from",
)


def blind_shots(paths: Paths) -> frozenset[int]:
    """The frozen cohort's blind shots (`spans.cohort_path`)."""
    cohort = read_cohort(spans.cohort_path(paths))
    return frozenset(int(shot) for shot in cohort.shot[cohort.blind])


def read_targets(paths: Paths, spec: EventSpec) -> dict:
    """Each original target shot's `targets.target_bins`; None, logged, for one
    that cannot be read."""
    found = {}
    for shot in targets.target_shots(paths, spec):
        try:
            found[shot] = targets.target_bins(paths, spec, shot)
        except (OSError, ValueError, KeyError) as error:
            log.warning("%s: shot %s: no target: %s", spec.method, shot, error)
            found[shot] = None
    return found


def merge_targets(found: dict, saved: dict, bin_ms: float) -> dict:
    """Every shot's `targets.merged` target, by shot: `found`'s (the originals,
    `read_targets`) with `saved`'s labels over them, and `saved`'s shots the
    originals lack. An original that cannot be read (None) leaves the owner's
    label alone, and a shot with neither is None."""
    return {
        int(shot): targets.merged(found.get(shot), saved.get(shot), bin_ms)
        for shot in sorted(set(found) | set(saved))
    }


def _unreadable(path) -> bool:
    """Whether a file is there that h5py cannot open."""
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r"):
            return False
    except OSError:
        return True


def _missing_group(paths: Paths, spec: EventSpec, shot: int) -> str | None:
    """Why a required group is not on disk, or None when every one is."""
    for group in spec.required_groups:
        if raw.record_tier(shot, group, paths=paths) is not None:
            continue
        for tier, path in (
            ("corpus", paths.corpus_file(shot)),
            ("cache", raw.cache_path(shot, paths=paths)),
        ):
            if _unreadable(path):
                return f"unreadable {tier} file"
        return f"no {group}"
    return None


def _inside(starts, bin_ms, window) -> np.ndarray:
    """Which bins lie wholly inside `window`."""
    starts = np.asarray(starts, dtype=np.float64)
    return (starts >= window[0]) & (starts + bin_ms <= window[1])


def _store_problem(paths: Paths, spec: EventSpec, shot: int, window) -> str | None:
    """Why a roster shot's review store gives nothing in `window`, or None.

    Its rows must hold every required role (`features.match_roles`), and each
    required trace row a value inside the window at the coarsest level, which
    holds one wherever a column below it does (`rows.pool`). A store whose
    required rows are all NaN (the ECE of sawtooth 187154) would give no
    observed frame. The rest of `features`' reading is left to Task 2.8's.

    A stale tearing-mode store (D65) passes here, and the shot stays in the
    split, but it is never read: `prepare` rebuilds only the stores outside the
    roster, so it drops this one from the features ("stale store: ...", in the
    shot's `.dropped.json`), and a test shot so dropped is counted in the
    evaluation's md. Rebuilding it is the review's (`review.build --force`). No
    roster store was stale when Task 2.7 ran.
    """
    path = store_path(paths, spec, shot)
    if not path.is_file():
        return "no review store"
    level = str(max(LEVELS))
    try:
        with h5py.File(path, "r") as f:
            described = [
                {"name": name, **json.loads(f["rows"][name].attrs["meta"])}
                for name in json.loads(f.attrs["rows"])
            ]
            try:
                found = match_roles(described, spec, str(path))
            except StaleStore:
                return None
            t0, dt, n = (f.attrs[key] for key in ("t0_ms", "dt_ms", "n"))
            width = float(dt) * int(level)
            centres = float(t0) + (np.arange(-(-int(n) // int(level))) + 0.5) * width
            cols = np.flatnonzero((centres >= window[0]) & (centres < window[1]))
            for role, row in zip(spec.roles, found, strict=True):
                if role.kind != "trace" or role.optional:
                    continue
                if not cols.size:
                    return "no signal in the window"
                top = f["rows"][row["name"]][level][1, :, cols[0] : cols[-1] + 1]
                if not np.isfinite(top).any():
                    return "no signal in the window"
    except (OSError, KeyError, ValueError) as error:
        log.warning("%s: shot %s: its review store: %s", spec.method, shot, error)
        return "unusable review store"
    return None


def _check(paths, spec, shot, bins, *, blind, catalog, roster) -> dict | str:
    """A shot's row of `eligible`, or why it is left out."""
    if shot in blind:
        return "blind"
    if bins is None:
        return "no target"
    starts, states = bins
    if not targets.labelled(states):
        return "no labelled bin"
    missing = _missing_group(paths, spec, shot)
    if missing is not None:
        return missing
    hull = targets.hull(starts, states, spec.bin_ms)
    window, window_from = targets.window_for(shot, paths, hull, catalog=catalog)
    inside = states[_inside(starts, spec.bin_ms, window)]
    if not targets.labelled(inside):
        return "no labelled bin in the window"
    if shot in roster:
        problem = _store_problem(paths, spec, shot, window)
        if problem is not None:
            return problem
    return {
        "shot": shot,
        "positive": int(np.any(inside == PRESENT_T)),
        "roster": int(shot in roster),
        "window_start_ms": int(window[0]),
        "window_end_ms": int(window[1]),
        "window_from": window_from,
    }


def eligible(
    paths: Paths, spec: EventSpec, *, found=None, saved=None
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """The shots the method may use (`ELIGIBLE_COLUMNS`, by shot) and the
    left-out shots by reason.

    `found` is `read_targets`' answer and `saved` the owner's labels, each read
    here when not given (the live saves; `make` passes its frozen copy's); a
    shot's target is `merge_targets`'.
    """
    found = read_targets(paths, spec) if found is None else found
    if saved is None:
        saved = labels.read_saved(paths.label_tables / spec.event)
    context = {
        "blind": blind_shots(paths),
        "catalog": targets.catalog_windows(paths),
        "roster": roster_shots(paths, spec.store_event),
    }
    table, left = [], defaultdict(list)
    for shot, bins in merge_targets(found, saved, spec.bin_ms).items():
        row = _check(paths, spec, shot, bins, **context)
        if isinstance(row, str):
            left[row].append(shot)
        else:
            table.append(row | {"owner": int(shot in saved)})
    frame = pd.DataFrame(table, columns=list(ELIGIBLE_COLUMNS))
    return frame, dict(sorted(left.items()))


def split(frame: pd.DataFrame, seed: int = SEED) -> dict[int, str]:
    """Each shot's split: within each (positive, roster) stratum, in order, a
    permutation of its shots at `seed`, the first 15 % (rounded half up) test,
    the next 15 % val and the rest train."""
    _, val, test = SPLIT_FRACTIONS
    rng = np.random.default_rng(seed)
    out = {}
    for _, stratum in frame.groupby(["positive", "roster"], sort=True):
        shots = sorted(int(shot) for shot in stratum.shot)
        n = len(shots)
        n_test = math.floor(test * n + 0.5)
        n_val = math.floor(val * n + 0.5)
        for rank, i in enumerate(rng.permutation(n)):
            if rank < n_test:
                out[shots[i]] = "test"
            elif rank < n_test + n_val:
                out[shots[i]] = "val"
            else:
                out[shots[i]] = "train"
    return out


def _ip_log(paths: Paths) -> dict[int, tuple[int, int]]:
    """The catalog's measured Ip windows (`catalog/ip.jsonl`), by shot."""
    path = paths.catalog / "ip.jsonl"
    try:
        frame = read_log(path)
    except (OSError, CatalogError) as error:
        log.warning("no Ip windows from %s: %s", path, error)
        return {}
    frame = frame[frame.status == "ok"]
    return {
        int(row.shot): (int(row.window_start_ms), int(row.window_end_ms))
        for row in frame.itertuples(index=False)
    }


def _labels_window(paths, spec, frame, merged) -> dict:
    """The "labels"-window shots, whose window is their target's hull (a legacy
    grid's can span its whole axis), and their labelled bins outside the
    plasma's Ip window, where the catalog's Ip log has one (for Task 2.10)."""
    ip = _ip_log(paths)
    out = dict.fromkeys(
        ("shots", "with_ip_window", "labelled_bins", "outside_ip_window"), 0
    )
    for row in frame[frame.window_from == "labels"].itertuples(index=False):
        out["shots"] += 1
        plasma = ip.get(int(row.shot))
        if plasma is None:
            continue
        starts, states = merged[row.shot]
        window = (row.window_start_ms, row.window_end_ms)
        known = _inside(starts, spec.bin_ms, window) & (states != UNKNOWN)
        out["with_ip_window"] += 1
        out["labelled_bins"] += int(known.sum())
        outside = known & ~_inside(starts, spec.bin_ms, plasma)
        out["outside_ip_window"] += int(outside.sum())
    return out


def make(paths: Paths, method: str, *, force: bool = False) -> pd.DataFrame:
    """Write the method's shots file and its meta (module docstring), both
    `frames.VERSION`'s; the shots.

    A split is made once: `force` makes it again, and freezes the owner's saves
    again, unless a model is trained on it."""
    spec = SPECS[method]
    out = shots_file(paths, method, VERSION)
    meta_path = shots_meta_file(paths, method, VERSION)
    if out.exists() or meta_path.exists():
        if not force:
            raise FileExistsError(
                f"{out}: {method}'s split is made once; --force makes it again, "
                "and freezes the owner's saves again"
            )
        model = model_dir(paths, method, VERSION) / "model.pt"
        if model.exists():
            raise FileExistsError(f"{model}: a model is trained on the split")
    found = read_targets(paths, spec)
    source = labels.labels_path(paths.label_tables / spec.event)
    snapshot = {"path": None, "sha256": None, "source": str(source)}
    saved = {}
    if source.is_file():
        data = source.read_bytes()
        copy = owner_file(paths, method, VERSION)
        with atomic_path(copy) as tmp:
            tmp.write_bytes(data)
        saved = labels.read_labels(copy)
        snapshot |= {"path": str(copy), "sha256": hashlib.sha256(data).hexdigest()}
    merged = merge_targets(found, saved, spec.bin_ms)
    frame, left = eligible(paths, spec, found=found, saved=saved)
    assigned = split(frame)
    frame["split"] = [assigned[int(shot)] for shot in frame.shot]
    frame = frame.sort_values("shot", kind="stable").reset_index(drop=True)
    shots = frame[list(COLUMNS)].astype(
        {"shot": "int64", "positive": "int64", "roster": "int64", "owner": "int64"}
    )
    known = [bins[1] for bins in merged.values() if bins is not None]
    legacy = [bins[1] for bins in found.values() if bins is not None]
    present = sum(int(np.sum(states == PRESENT_T)) for states in known)
    roster = roster_shots(paths, spec.store_event)
    meta = {
        "method": method,
        "version": VERSION,
        "target": spec.target,
        "labelled_shots": sum(targets.labelled(states) for states in known),
        "positive_shots": sum(bool(np.any(states == PRESENT_T)) for states in known),
        "legacy_labelled_shots": sum(targets.labelled(states) for states in legacy),
        "present_s": round(present * spec.bin_ms / 1000.0, 3),
        "owner": {
            "saved": len(saved),
            "overriding": sum(shot in found for shot in saved),
            "added": sum(shot not in found for shot in saved),
        },
        "seed": SEED,
        "fractions": dict(zip(("train", "val", "test"), SPLIT_FRACTIONS, strict=True)),
        "counts": {s: int((shots.split == s).sum()) for s in SPLITS},
        "positive": {s: int(shots.positive[shots.split == s].sum()) for s in SPLITS},
        "owner_snapshot": snapshot,
        "left_out": {reason: len(shots_) for reason, shots_ in left.items()},
        "left_out_shots": left,
        "window_from": dict(sorted(Counter(frame.window_from).items())),
        "labels_window": _labels_window(paths, spec, frame, merged),
        "windows": {
            str(row.shot): [row.window_start_ms, row.window_end_ms, row.window_from]
            for row in frame.itertuples(index=False)
        },
        "roster_without_store": sorted(
            shot for shot in roster if not store_path(paths, spec, shot).is_file()
        ),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
    }
    with atomic_path(meta_path) as tmp:
        tmp.write_text(json.dumps(meta, indent=1, default=int) + "\n")
    with atomic_path(out) as tmp:
        shots.to_csv(tmp, index=False)
    return shots


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", required=True, choices=list(SPECS))
    parser.add_argument(
        "--force", action="store_true", help="make the split again (module docstring)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = Paths.from_env()
    try:
        make(paths, args.method, force=args.force)
    except FileExistsError as error:
        raise SystemExit(f"error: {error}") from None
    meta = json.loads(shots_meta_file(paths, args.method, VERSION).read_text())
    keys = (
        "method",
        "version",
        "counts",
        "labelled_shots",
        "positive_shots",
        "owner",
        "window_from",
    )
    line = {k: meta[k] for k in keys} | {
        "left_out": meta["left_out"],
        "shots": str(shots_file(paths, args.method, VERSION)),
    }
    print(json.dumps(line), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
