"""Scan non-blind cohort shots for live lower-divertor TangTV and build stores.

Read-only corpus and cohort. Candidate roster goes to shots_review.csv; existing
shots.csv curation is preserved. The server's isolated shots.csv and all stores
are under --out. Each eligibility decision records an actual native frame.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from labeler.config import Paths, git_sha, sha256_of
from labeler.events import rosters
from labeler.events.review import build, rows, video

REPO = Path(__file__).resolve().parents[2]


def scan(paths, cohort):
    records = []
    selected = cohort[cohort.split.isin(["train", "val"])].sort_values("queue_rank")
    for row in selected.itertuples(index=False):
        path = paths.corpus_file(int(row.shot))
        record = {
            "shot": int(row.shot),
            "split": row.split,
            "corpus": str(path),
            "window_ms": [int(row.window_start_ms), int(row.window_end_ms)],
            "lower_channels": [],
            "reason": "corpus missing",
        }
        records.append(record)
        if not path.is_file():
            continue
        try:
            with h5py.File(path, "r") as source:
                if "tangtv" not in source:
                    record["reason"] = "tangtv missing"
                    continue
                t, data, channels = video._layout(source["tangtv"])
                record["source_shape"] = list(data.shape)
                # Same preview cadence as the store, confined to the plasma.
                indices = video.frame_indices(t)
                indices = indices[
                    (t[indices] * 1000 >= row.window_start_ms)
                    & (t[indices] * 1000 <= row.window_end_ms)
                ]
                record["preview_indices_checked"] = {}
                for channel in (0, 2):
                    if channel >= channels:
                        continue
                    checked = 0
                    # Try the centre first, then every retained native frame;
                    # a negative decision checks all candidates, never a stride
                    # over pixels that could miss a lit image region.
                    order = list(indices)
                    if order:
                        order.insert(0, order.pop(len(order) // 2))
                    for index in order:
                        checked += 1
                        frame = video._frame(
                            data, channel, int(index), video.CAMERAS["tangtv"]
                        )
                        finite = frame[np.isfinite(frame)]
                        if finite.size and float(np.ptp(finite)) > 0:
                            record["lower_channels"].append(
                                {
                                    **video.view("tangtv", channel),
                                    "source_index": int(index),
                                    "time_ms": float(t[index] * 1000),
                                    "finite_pixels": int(finite.size),
                                    "min": float(finite.min()),
                                    "max": float(finite.max()),
                                }
                            )
                            break
                    record["preview_indices_checked"][str(channel)] = checked
                record["reason"] = (
                    ""
                    if record["lower_channels"]
                    else "no live lower-divertor preview in plasma window"
                )
        except (ValueError, OSError) as error:
            record["reason"] = str(error)
        print(f"scan {row.shot}: {record['reason'] or 'eligible'}", flush=True)
    return records


def build_store(r, paths, indicators, resume=False):
    started = time.monotonic()
    store = paths.spectrogram_file("detachment", r["shot"])
    current = build.current(store, "detachment")
    if current:
        with h5py.File(store, "r") as source:
            for camera in source.get("videos", {}).values():
                shape = json.loads(camera.attrs.get("source_shape", "[]"))
                if shape and min(shape[-2:]) < 2:
                    current = False
    rebuild = not (resume and current)
    store = build.build("detachment", r["shot"], paths, force=rebuild)
    manifest = video.meta(store)
    return {
        "shot": r["shot"],
        "split": r["split"],
        "store": str(store),
        "bytes": store.stat().st_size,
        "seconds": time.monotonic() - started,
        "action": "rebuilt" if rebuild else "kept current valid-geometry store",
        "frame_counts": {
            c["name"]: {str(ch["channel"]): len(ch["times_ms"]) for ch in c["channels"]}
            for c in manifest["cameras"]
        },
        "rows": rows.meta(store)["rows"],
        "indicator_source": str(indicators / f"{r['shot']}.npz"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--reuse-scan", action="store_true")
    parser.add_argument(
        "--resume-build",
        action="store_true",
        help="keep current stores with known image geometry",
    )
    parser.add_argument("--indicator-root", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    original = Paths.from_env()
    cohort_path = REPO / "data/events/catalog/cohort.csv"
    cohort = pd.read_csv(cohort_path)
    scan_path = args.out / "corpus_scan.json"
    if args.reuse_scan:
        record = json.loads(scan_path.read_text())
        if record["cohort_sha256"] != sha256_of(cohort_path):
            raise ValueError("cohort changed since scan")
    else:
        started = time.monotonic()
        records = scan(original, cohort)
        record = {
            "git_sha": git_sha(full=True),
            "script_sha256": sha256_of(Path(__file__)),
            "cohort": str(cohort_path),
            "cohort_sha256": sha256_of(cohort_path),
            "policy": "train/val only; channel 0 or 2; finite, spatially nonconstant preview inside plasma window",
            "blind_shots_excluded": int((cohort.split == "test").sum()),
            "records": records,
            "scan_seconds": time.monotonic() - started,
        }
    eligible = [r for r in record["records"] if r["lower_channels"]]
    record["summary"] = {
        "scanned": len(record["records"]),
        "eligible": len(eligible),
        "by_split": {
            s: sum(r["split"] == s for r in eligible) for s in ("train", "val")
        },
    }
    scan_path.write_text(json.dumps(record, indent=2) + "\n")
    frame = pd.DataFrame(
        [
            [
                r["shot"],
                "unverified",
                "false",
                "",
                "",
                f"{r['split']} cohort; live lower TangTV "
                + "; ".join(str(ch["channel"]) for ch in r["lower_channels"]),
            ]
            for r in eligible
        ],
        columns=rosters.ROSTER_COLUMNS,
    )
    roster_path = REPO / "data/events/detachment/shots_review.csv"
    rosters.write_roster(frame, roster_path, keep_order=True)
    isolated = args.out / "tables/detachment/shots.csv"
    isolated.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(roster_path, isolated)
    if args.build:
        indicators = args.indicator_root or original.root / "round4/detach/bins"
        os.environ["LABELER_DETACHMENT_INDICATORS"] = str(indicators)
        paths = Paths(
            root=args.out, corpus=original.corpus, label_tables=args.out / "tables"
        )
        built = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {
                pool.submit(build_store, r, paths, indicators, args.resume_build): r
                for r in eligible
            }
            for future in as_completed(pending):
                result = future.result()
                built.append(result)
                print(f"build {result['shot']}: ok", flush=True)
        built.sort(key=lambda r: r["shot"])
        (args.out / "roster_build.json").write_text(
            json.dumps(
                {
                    "git_sha": git_sha(full=True),
                    "source_sha256": {
                        str(p.relative_to(REPO)): sha256_of(p)
                        for p in (
                            Path(__file__),
                            REPO / "src/labeler/events/review/video.py",
                            REPO / "src/labeler/events/panels/detachment.py",
                            REPO / "src/labeler/events/review/detachment.py",
                        )
                    },
                    "scan": str(scan_path),
                    "roster": str(roster_path),
                    "roster_sha256": sha256_of(roster_path),
                    "indicator_root": str(indicators),
                    "stores": built,
                },
                indent=2,
            )
            + "\n"
        )
    print(json.dumps(record["summary"]))


if __name__ == "__main__":
    main()
