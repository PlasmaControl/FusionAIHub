"""Refresh existing per-shot HDF5 files in place from the FDP pipeline.

For each shot: fetch through ``fetch_shot_fast.sh``, merge anything the existing
file has that the fetch did not produce, drop the duplicate lower-case tree
group, and only then replace the original.

The ordering matters.  The existing files are the only copy of the omega
pipeline's output, so a fetch that silently produced fewer signals would destroy
data that cannot be re-derived.  Every shot therefore passes a gate before its
original is touched:

* **Signals the old file has and the new one does not are copied across**, not
  dropped.  On shot 115401 there were none, but the dataset is heterogeneous
  (410 signals for 103425, 1393 for 146992, and only some eras carry
  SPECTROSCOPY, AOT or EFIT01), so this cannot be assumed for every shot.
* **Any real disagreement aborts that shot** and leaves the original in place.
  ``units`` differences do not count: the old pipeline stored blanks where the
  FDP path returns the real strings.
* A shot is replaced only after its merged file has been re-opened and
  re-counted.

Failures are per-shot: one bad shot is logged and skipped, not fatal.  Progress
goes to a JSONL log so a re-run resumes rather than re-fetching.

Usage::

    python refresh_shots.py --dir <dir> --below 150000 [--limit N] [--shots ...]
                            [--log run.jsonl] [--keep-going]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FETCH = SCRIPT_DIR / "fetch_shot_fast.sh"


def signal_map(path: Path) -> dict:
    """Map ``(TREE_UPPER, signal)`` to its child dataset names.

    Upper-casing the tree collapses the duplicate ``D3D``/``d3d`` groups that
    older files carry, so a duplicate does not read as extra content.
    """
    out: dict = {}
    with h5py.File(path, "r") as handle:
        shot_group = handle[list(handle.keys())[0]]
        for tree in shot_group:
            for signal in shot_group[tree]:
                node = shot_group[tree][signal]
                if isinstance(node, h5py.Group):
                    out[(tree.upper(), signal)] = {
                        k for k, v in node.items() if isinstance(v, h5py.Dataset)
                    }
    return out


def _values_differ(a, b) -> bool:
    """Compare two HDF5 payload values the way the validation diff does."""
    if isinstance(a, bytes) or isinstance(b, bytes):
        return a != b
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return True
    if a.dtype.kind in "SUO" or b.dtype.kind in "SUO":
        return not np.array_equal(a, b)
    return not np.allclose(a.astype("f8"), b.astype("f8"),
                           rtol=1e-5, atol=0, equal_nan=True)


def compare(old: Path, new: Path) -> dict:
    """Diff two shot files, ignoring tree-name case and ``units`` strings."""
    old_map, new_map = signal_map(old), signal_map(new)
    old_keys, new_keys = set(old_map), set(new_map)

    real: list[str] = []
    units_only = identical = extra_keys = 0
    with h5py.File(old, "r") as fo, h5py.File(new, "r") as fn:
        go = fo[list(fo.keys())[0]]
        gn = fn[list(fn.keys())[0]]
        omap = {(t.upper(), s): go[t][s] for t in go for s in go[t]}
        nmap = {(t.upper(), s): gn[t][s] for t in gn for s in gn[t]}
        for key in old_keys & new_keys:
            bad = []
            for child in sorted(old_map[key] & new_map[key]):
                if _values_differ(omap[key][child][()], nmap[key][child][()]):
                    bad.append(child)
            # A key set that only *grew* is added information, not a
            # disagreement: the old pipeline stored `data` alone for plenty of
            # nodes where this one also resolves `dim0`/`dim1`/`units` (all three
            # AOT signals on shot 146918, and 19 signals on 200000). Losing a
            # key the old file had is a real difference, so the direction of the
            # difference is what matters, not its existence.
            missing = old_map[key] - new_map[key]
            gained = new_map[key] - old_map[key]
            if missing:
                bad.append(f"__lost__{sorted(missing)}")
            if not bad:
                if gained:
                    extra_keys += 1
                else:
                    identical += 1
            elif bad == ["units"]:
                units_only += 1
            else:
                real.append(f"{key[0]}/{key[1]}: {','.join(bad[:3])}")

    return {
        "old_signals": len(old_keys),
        "new_signals": len(new_keys),
        "identical": identical,
        "units_only": units_only,
        "extra_keys": extra_keys,
        "real_differences": len(real),
        "real_examples": real[:5],
        "old_only": sorted(f"{t}/{s}" for t, s in old_keys - new_keys),
        "new_only_count": len(new_keys - old_keys),
    }


def merge_old_only(old: Path, new: Path, old_only: list) -> int:
    """Copy signals present only in the old file into the new one."""
    copied = 0
    with h5py.File(old, "r") as fo, h5py.File(new, "a") as fn:
        go = fo[list(fo.keys())[0]]
        gn = fn[list(fn.keys())[0]]
        # Resolve each entry back to a real group, preferring the upper-case
        # tree so the duplicate lower-case one is never the source.
        lookup: dict = {}
        for tree in go:
            for signal in go[tree]:
                lookup.setdefault((tree.upper(), signal), (tree, signal))
                if tree.isupper():
                    lookup[(tree.upper(), signal)] = (tree, signal)
        for entry in old_only:
            tree_upper, signal = entry.split("/", 1)
            source = lookup.get((tree_upper, signal))
            if source is None:
                continue
            target = gn.require_group(tree_upper)
            if signal in target:
                continue
            fo.copy(go[source[0]][source[1]], target, name=signal)
            copied += 1
    return copied


def drop_lowercase_duplicates(path: Path) -> list:
    """Delete lower-case tree groups whose upper-case twin holds the same signals.

    Only an exact duplicate is removed.  A lower-case group carrying anything
    its twin lacks is left alone, because deleting it would lose data.
    """
    dropped = []
    with h5py.File(path, "a") as handle:
        shot_group = handle[list(handle.keys())[0]]
        for tree in [t for t in shot_group if not t.isupper()]:
            twin = tree.upper()
            if twin not in shot_group or twin == tree:
                continue
            extra = set(shot_group[tree]) - set(shot_group[twin])
            if extra:
                continue
            del shot_group[tree]
            dropped.append(tree)
    return dropped


def describe(path: Path) -> dict:
    """Return tree sizes and total signal count."""
    with h5py.File(path, "r") as handle:
        shot_group = handle[list(handle.keys())[0]]
        trees = {k: len(shot_group[k]) for k in shot_group}
    return {"trees": trees, "total": sum(trees.values())}


def _finish(result: dict, **fields) -> dict:
    """Update the shared result dict in place and return it.

    Returning ``{**result, ...}`` instead would leave ``result["status"]`` as
    ``"unknown"``, and :func:`refresh_one`'s ``finally`` block reads that to
    decide whether to keep a rejected fetch -- which is how the first version of
    this driver silently discarded the very files needed to diagnose a gate
    failure.
    """
    result.update(fields)
    return result


def refresh_one(shot: int, target_dir: Path, work_dir: Path,
                stage_parent: Optional[str] = None,
                keep_rejects: bool = True) -> dict:
    """Fetch, gate, merge and replace one shot.  Never raises for one shot."""
    started = time.time()
    original = target_dir / f"{shot}.h5"
    result: dict = {"shot": shot, "status": "unknown"}
    if not original.is_file():
        return _finish(result, status="skipped", reason="no existing file")

    # A truncated original is an aborted write from the old pipeline (147642 has
    # 25 MB on disk against a stored EOF of 10.4 GB).  There is nothing in it to
    # preserve and nothing to compare against, so the fetch simply replaces it
    # -- for these shots this is a repair, not a refresh.
    readable = True
    try:
        with h5py.File(original, "r") as probe:
            list(probe.keys())
    except Exception as error:  # noqa: BLE001
        readable = False
        result["original_unreadable"] = f"{type(error).__name__}: {str(error)[:120]}"

    scratch = work_dir / f"refresh_{shot}"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [str(FETCH), str(shot), str(scratch)]
        if stage_parent:
            cmd.append(stage_parent)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        fetched = scratch / f"{shot}.h5"
        if proc.returncode != 0 or not fetched.is_file():
            tail = (proc.stdout or "")[-400:] + (proc.stderr or "")[-400:]
            return _finish(result, status="fetch_failed", rc=proc.returncode,
                                   log_tail=tail,
                                   seconds=round(time.time() - started))

        if readable:
            diff = compare(original, fetched)
            result.update(diff)
            if diff["real_differences"]:
                return _finish(result, status="gate_failed",
                                       reason="real data differences",
                                       seconds=round(time.time() - started))
            result["merged_old_only"] = (
                merge_old_only(original, fetched, diff["old_only"])
                if diff["old_only"] else 0)
        else:
            diff = {"old_signals": 0, "new_signals": len(signal_map(fetched)),
                    "old_only": []}
            result.update(diff)
            result["merged_old_only"] = 0

        result["dropped_groups"] = drop_lowercase_duplicates(fetched)

        after = describe(fetched)
        result["final"] = after
        expected = max(diff["old_signals"], diff["new_signals"])
        if not readable and after["total"] == 0:
            return _finish(result, status="gate_failed",
                                   reason="original unreadable and fetch "
                                          "produced nothing",
                                   seconds=round(time.time() - started))
        if after["total"] < expected:
            return _finish(result, status="gate_failed",
                                   reason=f"merged file has {after['total']} "
                                          f"signals, expected at least "
                                          f"{expected}",
                                   seconds=round(time.time() - started))

        # Same filesystem, so this is a rename: the original is replaced only
        # once the merged file has been re-opened and re-counted above.
        backup = target_dir / f".{shot}.h5.replacing"
        os.replace(original, backup)
        try:
            os.replace(fetched, original)
        except Exception:
            os.replace(backup, original)
            raise
        backup.unlink(missing_ok=True)
        result["status"] = "replaced" if readable else "repaired"
        result["old_bytes"] = None
        result["seconds"] = round(time.time() - started)
        return result
    except Exception as error:  # noqa: BLE001
        return _finish(result, status="error",
                               reason=f"{type(error).__name__}: {error}",
                               seconds=round(time.time() - started))
    finally:
        # A gate failure is exactly when the fetched file is worth looking at, so
        # keep it instead of discarding the evidence.  Successful shots leave
        # nothing behind.
        if result.get("status") in ("gate_failed", "error") and keep_rejects:
            reject = work_dir / f"rejected_{shot}.h5"
            try:
                if (scratch / f"{shot}.h5").is_file():
                    os.replace(scratch / f"{shot}.h5", reject)
                    result["rejected_file"] = str(reject)
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, required=True,
                        help="Directory of <shot>.h5 files to refresh")
    parser.add_argument("--below", type=int, default=None,
                        help="Only shots with a number below this")
    parser.add_argument("--shots", type=int, nargs="+", default=None,
                        help="Explicit shot numbers instead of scanning --dir")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many shots")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Where fetches land before replacing; must be on "
                             "the same filesystem as --dir (default: --dir)")
    parser.add_argument("--stage-parent", default=None,
                        help="Passed to fetch_shot_fast.sh (default /dev/shm)")
    parser.add_argument("--log", type=Path, required=True,
                        help="JSONL progress log; existing entries are skipped")
    parser.add_argument("--discard-rejects", action="store_true",
                        help="Delete a fetch that failed the gate instead of "
                             "keeping it as rejected_<shot>.h5 for inspection")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be refreshed and exit")
    args = parser.parse_args()

    if args.shots:
        shots = sorted(args.shots)
    else:
        shots = sorted(int(p.stem) for p in args.dir.glob("*.h5")
                       if p.stem.isdigit()
                       and (args.below is None or int(p.stem) < args.below))

    done: set = set()
    if args.log.is_file():
        for line in args.log.read_text().splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("status") in ("replaced", "repaired", "skipped"):
                done.add(entry["shot"])
    todo = [s for s in shots if s not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(shots)} shots matched, {len(done)} already done, "
          f"{len(todo)} to process", flush=True)
    if args.dry_run:
        print("dry run:", todo[:20], "..." if len(todo) > 20 else "")
        return

    work_dir = args.work_dir or args.dir
    args.log.parent.mkdir(parents=True, exist_ok=True)
    counts: dict = {}
    t0 = time.time()
    for index, shot in enumerate(todo, 1):
        outcome = refresh_one(shot, args.dir, work_dir, args.stage_parent,
                              keep_rejects=not args.discard_rejects)
        counts[outcome["status"]] = counts.get(outcome["status"], 0) + 1
        with args.log.open("a") as handle:
            handle.write(json.dumps(outcome) + "\n")
        rate = (time.time() - t0) / index
        print(f"[{index}/{len(todo)}] {shot}: {outcome['status']}"
              f" ({outcome.get('seconds','?')}s)"
              f" old={outcome.get('old_signals','?')}"
              f" new={outcome.get('new_signals','?')}"
              f" merged={outcome.get('merged_old_only','-')}"
              f" xkeys={outcome.get('extra_keys','-')}"
              f" | mean {rate/60:.1f} min/shot,"
              f" eta {(len(todo)-index)*rate/3600:.1f} h", flush=True)
        if outcome["status"] not in ("replaced", "skipped"):
            print(f"    -> {outcome.get('reason') or outcome.get('log_tail','')}"[:400],
                  flush=True)

    print(f"\ndone in {(time.time()-t0)/3600:.2f} h: {counts}", flush=True)
    sys.exit(0 if (counts.get("replaced") or counts.get("repaired")) else 1)


if __name__ == "__main__":
    main()
