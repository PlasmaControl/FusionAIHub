#!/usr/bin/env python
"""Fetch missing Ip/Bt for the reviewed DSM shots into an isolated evaluation store.

Run on the login node via the mandated pixi labelmaker `fdp run python` wrapper.
Use `--workers 1 --pace 1`; at most three workers are accepted. No production data
store is modified. Stop immediately when an authentication/login error is seen.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import dsm, labels, prepare

OUT = Path(__file__).resolve().parents[2] / "outputs/labeler/elm/dsm/fetch.json"
AUTH = re.compile(
    r"auth|login|credential|expired|unauthor|forbidden|permission denied|token",
    re.IGNORECASE,
)


def fetch_one(task):
    """One shot, preserving exceptions so authentication errors cannot be hidden."""
    from labeler.features import namespace as ns
    from labeler.features import resolve_fdp

    shot, want, target, pace = task
    path = Path(target) / f"{shot}.npz"
    payload = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            payload.update({k: z[k] for k in z.files})
    row = {"shot": shot, "requested": want, "fetched": {}, "missing": {}}
    cache = {}

    def cached(key, thunk):
        if key not in cache:
            cache[key] = thunk()
        return cache[key]

    for name in want:
        try:
            diagnosis = resolve_fdp._import_diagnosis()
            if diagnosis is not None:
                raise RuntimeError(diagnosis)
            spec = ns.by_name(name)
            arr = resolve_fdp._resolve_one(spec, spec.locator_for("fdp"), shot, cached)
            if arr.y.shape[-1] < 2 or not np.isfinite(arr.y).any():
                raise ValueError("empty or nonfinite signal")
        except Exception as exc:  # noqa: BLE001 - record the shot's fetch failure
            cause = f"{type(exc).__name__}: {exc}"
            row["missing"][name] = cause
            if AUTH.search(cause):
                row["auth_error"] = True
                break
        else:
            payload[f"{name}_x"] = arr.x
            payload[f"{name}_y"] = arr.y
            payload[f"{name}_attrs"] = json.dumps(arr.attrs)
            row["fetched"][name] = {"samples": int(arr.y.shape[-1])}
        time.sleep(pace)
    if row["fetched"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)
    if path.exists():
        row["file"] = str(path)
        row["sha256"] = sha256_of(path)
    return row


def fetch_native_photodiodes(paths, out, pace, fresh=()):
    """Fetch PCPHD02/03 for every review shot, stopping on any auth error.

    A shot listed in `fresh` is fetched even where the upstream pickle holds its
    photodiodes, so no row of the detector rests on the upstream export.
    """
    from labeler.events.verify import fdp_signal

    source = Path("/projects/EKOLEMEN/wpqh_elm_hiro/data/dalpha_wpqh.pkl")
    with source.open("rb") as fh:
        existing = pickle.load(fh)
    review = labels.review_table(prepare.review_csv(paths))
    cohort = pd.read_csv(paths.catalog / "cohort.csv").set_index("shot")
    if any(cohort.split.get(int(s)) == "test" for s in review.shot.unique()):
        raise ValueError("cohort test shots may not be fetched for native evaluation")
    target = paths.root / "round4/elm/dsm/native_photodiodes"
    target.mkdir(parents=True, exist_ok=True)
    record = {
        "git": git_sha(full=True),
        "script_sha256": sha256_of(__file__),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "scope": "PCPHD02/03 on all 119 reviewed shots, regardless of H5 coverage",
        "rows": [],
        "pace_seconds": pace,
        "workers": 1,
        "stopped_on_auth_error": False,
        "store": str(target),
    }
    for shot in sorted(map(int, review.shot.unique())):
        for name in ("pcphd02", "pcphd03"):
            value = existing.get(str(shot), {}).get(name, {})
            if shot not in fresh and np.asarray(value.get("data", [])).size > 2:
                record["rows"].append(
                    {
                        "shot": shot,
                        "column": name,
                        "source": "existing upstream photodiode pickle",
                        "samples": int(np.asarray(value["data"]).size),
                    }
                )
                continue
            cache = target / f"{shot}_{name}.npz"
            row = {"shot": shot, "column": name, "path": str(cache)}
            if shot in fresh:
                row["fresh_beside_upstream_pickle"] = True
            try:
                arr = fdp_signal(shot, [name.upper()], via="ptdata", cache=cache)
                row["samples"] = int(arr.y.size)
                row["sha256"] = sha256_of(cache)
            except Exception as exc:  # noqa: BLE001 - preserve authentication failures
                cause = f"{type(exc).__name__}: {exc}"
                row["error"] = cause
                if AUTH.search(cause):
                    record["stopped_on_auth_error"] = True
            record["rows"].append(row)
            print(json.dumps(row), flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(record, indent=1))
            if record["stopped_on_auth_error"]:
                return 1
            time.sleep(pace)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--pace", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--native-photodiodes", action="store_true")
    ap.add_argument(
        "--fresh-shots",
        type=int,
        nargs="+",
        default=(),
        help="with --native-photodiodes: fetch these shots although the upstream "
        "pickle holds them",
    )
    args = ap.parse_args(argv)
    if args.pace < 1:
        ap.error("--pace must be at least 1 second")
    paths = Paths.from_env()
    if args.native_photodiodes:
        return fetch_native_photodiodes(
            paths, args.out, args.pace, {int(s) for s in args.fresh_shots}
        )
    table = labels.review_table(prepare.review_csv(paths))
    shots = sorted(int(s) for s in table.shot.unique())
    cohort = pd.read_csv(paths.catalog / "cohort.csv").set_index("shot")
    if any(cohort.split.get(s) == "test" for s in shots):
        raise ValueError("cohort test shots may not be fetched for this evaluation")
    record = {
        "git": git_sha(full=True),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "display_name": dsm.DISPLAY_NAME,
        "shots": shots,
        "workers": args.workers,
        "pace_seconds": args.pace,
        "cohort_test_shots_used": 0,
        "store": str(dsm.fetched_features_dir(paths)),
        "source": "DIII-D PTDATA via canonical fdp resolver",
        "rows": [],
        "stopped_on_auth_error": False,
    }
    tasks = []
    for shot in shots:
        arrays = dsm.shot_features(paths, shot, names=("ip", "bt"))
        want = [n for n in ("ip", "bt") if n not in arrays]
        record["rows"].append(
            {"shot": shot, "already_served": [n for n in ("ip", "bt") if n in arrays]}
        )
        if want:
            tasks.append((shot, want, str(dsm.fetched_features_dir(paths)), args.pace))
    record["requested_shots"] = len(tasks)
    print(f"{len(shots)} reviewed shots; {len(tasks)} need Ip/Bt", flush=True)
    pool = Pool(args.workers) if args.workers > 1 else None
    iterator = pool.imap_unordered(fetch_one, tasks) if pool else map(fetch_one, tasks)
    try:
        for row in iterator:
            record["rows"].append(row)
            print(json.dumps(row), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            if row.get("auth_error"):
                record["stopped_on_auth_error"] = True
                if pool:
                    pool.terminate()
                break
            args.out.write_text(json.dumps(record, indent=1))
    finally:
        if pool:
            pool.close()
            pool.join()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1))
    return 1 if record["stopped_on_auth_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
