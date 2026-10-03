"""Write the real detachment review queue using local, read-only evidence.

Union nonblind train/val cohort shots with a live lower TangTV preview coincident
with the producer's EFIT shelf gate, and every shot with a producer label/vote
already on disk. Re-running refreshes producer outputs and geometry. Retained
curation fields are preserved; corpus and producer inputs are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from zipfile import BadZipFile

import h5py
import numpy as np
import pandas as pd

from labeler.config import Paths, git_sha, sha256_of
from labeler.events import rosters
from labeler.events.review import build, detachment, geometry, rows, video

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "docs/labeler/results/detachment_review_queue.json"


def label_keys(keys):
    """Recognise producer states, grids, and draft indicator-vote outputs."""
    return any(
        key in {"category", "state", "state_lm", "state_rule", "labels", "prob"}
        or key.endswith("_vote")
        for key in keys
    )


def producer_snapshot(roots):
    """Read completed local label/vote files, retaining an exact input manifest.

    A producer can still be running: partial/unreadable files are recorded and
    retried on the next invocation. Camera inversions, surveys and placeholder
    rosters carry no label/vote field and do not count.
    """
    shots, explicit_test, sources, errors = set(), set(), [], []
    csvs = sorted({p for root in roots if root.is_dir() for p in root.rglob("*.csv*")})
    for path in csvs:
        if path.suffix not in (".csv", ".gz"):
            continue
        try:
            payload = path.read_bytes()
            compression = "gzip" if path.suffix == ".gz" else None
            columns = pd.read_csv(
                io.BytesIO(payload), nrows=0, compression=compression
            ).columns
            if "shot" in columns and label_keys(columns):
                usecols = ["shot"] + (["split"] if "split" in columns else [])
                table = pd.read_csv(
                    io.BytesIO(payload), usecols=usecols, compression=compression
                )
                values = pd.to_numeric(table.shot, errors="coerce")
                valid = values.notna() & (values % 1 == 0) & (values >= 100000)
                found = set(values[valid].astype(int))
                if "split" in table:
                    explicit_test.update(
                        values[valid & table.split.eq("test")].astype(int)
                    )
            elif path.stem.isdigit() and label_keys(columns):
                found = {int(path.stem)}
            else:
                continue
            shots.update(found)
            sources.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "shots": sorted(found),
                }
            )
        except (OSError, ValueError, pd.errors.ParserError, EOFError) as error:
            errors.append({"path": str(path), "error": str(error)})
    npzs = sorted({p for root in roots if root.is_dir() for p in root.rglob("*.npz")})
    for path in npzs:
        try:
            with path.open("rb") as stream, np.load(stream, allow_pickle=False) as grid:
                if not label_keys(grid.files):
                    continue
            payload = path.read_bytes()
            with np.load(io.BytesIO(payload), allow_pickle=False) as grid:
                if not label_keys(grid.files):
                    continue
                if "shot" in grid:
                    values = np.asarray(grid["shot"], dtype=float).ravel()
                    valid = np.isfinite(values) & (values % 1 == 0) & (values >= 100000)
                    found = set(values[valid].astype(int).tolist())
                    if "split" in grid:
                        split = np.asarray(grid["split"]).astype(str).ravel()
                        if split.shape == values.shape:
                            explicit_test.update(
                                values[valid & (split == "test")].astype(int).tolist()
                            )
                elif path.stem.isdigit() and int(path.stem) >= 100000:
                    found = {int(path.stem)}
                    if "split" in grid and np.any(
                        np.asarray(grid["split"]).astype(str) == "test"
                    ):
                        explicit_test.update(found)
                else:
                    continue
            shots.update(found)
            sources.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "shots": sorted(found),
                }
            )
        except (OSError, ValueError, BadZipFile, EOFError) as error:
            errors.append({"path": str(path), "error": str(error)})
    return {
        "shots": sorted(shots),
        "explicit_test_shots": sorted(explicit_test),
        "sources": sources,
        "errors": errors,
    }


def original_candidates(out, original_roster=None):
    """Archive the earlier camera-only roster once, for repair coverage audits."""
    archive = out / "original_camera_candidates.json"
    if original_roster is None and archive.is_file():
        return json.loads(archive.read_text())
    if original_roster is not None:
        payload = original_roster.read_bytes()
        source = str(original_roster)
    else:
        old_roster = REPO / "data/events/detachment/shots_review.csv"
        if old_roster.is_file():
            payload, source = old_roster.read_bytes(), str(old_roster)
        else:
            commit = git_sha(full=True)
            got = subprocess.run(
                ["git", "show", f"{commit}:data/events/detachment/shots_review.csv"],
                cwd=REPO,
                capture_output=True,
                check=False,
            )
            if got.returncode:
                return {
                    "shots": [],
                    "reason": "original camera-only roster unavailable",
                }
            payload = got.stdout
            source = f"git:{commit}:data/events/detachment/shots_review.csv"
    frame = rosters.validate_roster(pd.read_csv(io.BytesIO(payload), dtype=str))
    record = {
        "source": source,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "shots": sorted(frame.shot.astype(int).tolist()),
    }
    archive.write_text(json.dumps(record, indent=2) + "\n")
    return record


def scan(paths, cohort, previous=None):
    """Refresh live previews at EFIT-valid, in-plasma times on every invocation.

    Accept the older caller's ``previous`` argument without reusing its negative
    decisions: a prior scan may have used an older camera-layout reader.
    """
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
        record["geometry"] = geometry.load(row.shot, paths, record["window_ms"])
        geometry_path = Path(record["geometry"]["source"])
        record["geometry_sha256"] = (
            sha256_of(geometry_path) if geometry_path.is_file() else None
        )
        records.append(record)
        if not record["geometry"]["shelf_gate_samples"]:
            record["reason"] = "no EFIT-valid lower-null shelf sample in plasma window"
            continue
        if not path.is_file():
            continue
        try:
            with h5py.File(path, "r") as source:
                if "tangtv" not in source:
                    record["reason"] = "tangtv missing"
                    continue
                t, data, channels = video._layout(source["tangtv"])
                record["source_shape"] = list(data.shape)
                indices = video.frame_indices(t)
                indices = indices[
                    (t[indices] * 1000 >= row.window_start_ms)
                    & (t[indices] * 1000 <= row.window_end_ms)
                ]
                _, valid_geometry = geometry.at_times(
                    record["geometry"], t[indices] * 1000
                )
                indices = indices[valid_geometry]
                record["geometry_valid_preview_times"] = len(indices)
                record["preview_indices_checked"] = {}
                for channel in (0, 2):
                    if channel >= channels:
                        continue
                    checked, order = 0, list(indices)
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
                    else "no live lower TangTV preview coincident with EFIT shelf gate"
                )
        except (ValueError, OSError) as error:
            record["reason"] = str(error)
        print(f"scan {row.shot}: {record['reason'] or 'eligible'}", flush=True)
    return records


def queue_records(records, producer, cohort):
    """Deterministic union, excluding blind shots after both branches combine."""
    cohort_rows = {int(r.shot): r for r in cohort.itertuples(index=False)}
    blind = {int(r.shot) for r in cohort_rows.values() if r.split == "test"}
    blind.update(producer["explicit_test_shots"])
    eligible = {r["shot"]: dict(r) for r in records if r["lower_channels"]}
    labelled = set(producer["shots"])
    queue = []
    for shot in sorted(set(eligible) | labelled):
        if shot in blind:
            continue
        row = cohort_rows.get(shot)
        record = eligible.get(shot) or {
            "shot": shot,
            "split": row.split if row else "producer_external",
            "lower_channels": [],
            "window_ms": [int(row.window_start_ms), int(row.window_end_ms)]
            if row
            else None,
        }
        record["queue_sources"] = (
            ["cohort_camera_geometry"] if shot in eligible else []
        ) + (["producer_labels_or_votes"] if shot in labelled else [])
        queue.append(record)
    return queue, sorted(blind & (set(eligible) | labelled))


def roster_frame(queue, roster_path):
    """Keep real retained curation fields, replacing the example-only roster."""
    old = {}
    if roster_path.is_file():
        old = {
            int(r.shot): r
            for r in rosters.read_roster(roster_path).itertuples(index=False)
            if int(r.shot) >= 100000 and "EXAMPLE" not in r.notes
        }
    output = []
    for record in queue:
        shot = record["shot"]
        note = f"{record['split']}; " + "; ".join(record["queue_sources"])
        if "geometry" in record:
            g = record["geometry"]
            note += (
                f"; EFIT shelf {g['shelf_gate_samples']}/{g['total_samples']} samples"
            )
            note += "; config " + "/".join(k for k, v in g["counts"].items() if v)
        if shot in old:
            r = old[shot]
            base = (
                "" if r.notes.startswith("queue: ") else r.notes.split(" | queue: ")[0]
            )
            note = (base + " | " if base else "") + "queue: " + note
            output.append([shot, r.tier, r.holdout, r.reviewers, r.verified_on, note])
        else:
            output.append([shot, "unverified", "false", "", "", "queue: " + note])
    return pd.DataFrame(output, columns=rosters.ROSTER_COLUMNS)


def build_store(record, paths, indicators, resume=False):
    started = time.monotonic()
    store = paths.spectrogram_file("detachment", record["shot"])
    current = build.current(store, "detachment")
    desired_sources = detachment.context_sources(record["shot"], paths)
    if resume and current:
        with h5py.File(store, "r") as source:
            previous_sources = json.loads(source.attrs.get("params", "{}")).get(
                "context_sources"
            )
        if previous_sources is None:
            # Earlier stores without source fingerprints can only be reused if
            # neither producer input exists. All source-bearing stores refresh.
            current = not any(v["sha256"] for v in desired_sources.values())
        else:
            current = previous_sources == desired_sources
    rebuild = not (resume and current)
    store = build.build("detachment", record["shot"], paths, force=rebuild)
    manifest, row_metadata = video.meta(store), rows.meta(store)["rows"]
    store_range = rows.meta(store)["t_range"]
    plasma_window = detachment.plasma_window(record["shot"], paths)
    video_audit = []
    for camera in manifest["cameras"]:
        for channel in camera["channels"]:
            times = np.asarray(channel["times_ms"], dtype=float)
            outside_store = (times < store_range[0]) | (times > store_range[1])
            outside_plasma = (
                (times < plasma_window[0]) | (times > plasma_window[1])
                if plasma_window
                else np.zeros(len(times), dtype=bool)
            )
            video_audit.append(
                {
                    "camera": camera["name"],
                    "channel": channel["channel"],
                    "frames": len(times),
                    "min_ms": float(times.min()) if len(times) else None,
                    "max_ms": float(times.max()) if len(times) else None,
                    "outside_store_frames": int(outside_store.sum()),
                    "outside_plasma_frames": int(outside_plasma.sum()),
                }
            )
    with h5py.File(store, "r") as source:
        stored_geometry = json.loads(source.attrs.get("params", "{}")).get(
            "detachment_geometry", {}
        )
    titles = [r["title"] for r in row_metadata if "density" in r["title"].lower()]
    density = (
        "CO2"
        if any("CO2" in t for t in titles)
        else "Thomson"
        if any("Thomson" in t for t in titles)
        else "aux_ne"
        if titles
        else "unavailable"
    )
    return {
        "shot": record["shot"],
        "split": record["split"],
        "store": str(store),
        "bytes": store.stat().st_size,
        "seconds": time.monotonic() - started,
        "action": "rebuilt" if rebuild else "kept current store",
        "density_source": density,
        "context_sources": desired_sources,
        "store_range_ms": store_range,
        "plasma_window_ms": list(plasma_window) if plasma_window else None,
        "video_audit": video_audit,
        "geometry_counts": stored_geometry.get("counts", {}),
        "geometry_shelf_gate_samples": stored_geometry.get("shelf_gate_samples", 0),
        "geometry_total_samples": stored_geometry.get("total_samples", 0),
        "frame_counts": {
            c["name"]: {str(ch["channel"]): len(ch["times_ms"]) for ch in c["channels"]}
            for c in manifest["cameras"]
        },
        "rows": row_metadata,
        "indicator_source": str(indicators / f"{record['shot']}.npz"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--rebuild-existing",
        action="store_true",
        help="also repair all nonblind prior isolated stores",
    )
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--reuse-scan", action="store_true")
    parser.add_argument("--resume-build", action="store_true")
    parser.add_argument("--indicator-root", type=Path)
    parser.add_argument("--geometry-root", type=Path)
    parser.add_argument("--producer-root", type=Path)
    parser.add_argument("--producer-tables", type=Path)
    parser.add_argument("--record", type=Path, default=SUMMARY)
    parser.add_argument(
        "--original-roster",
        type=Path,
        help="optional historical camera-only roster for coverage audit",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    original_camera = original_candidates(args.out, args.original_roster)
    original = Paths.from_env()
    producer_root = args.producer_root or original.root / "round4/detach"
    producer_tables = (
        args.producer_tables
        or REPO.with_name("FusionAIHub-r4-detach") / "data/events/detachment"
    )
    geometry_root, indicators = (
        args.geometry_root or producer_root / "cache",
        args.indicator_root or producer_root / "bins",
    )
    os.environ["LABELER_DETACHMENT_GEOMETRY_ROOT"] = str(geometry_root)
    cohort_path, scan_path = (
        REPO / "data/events/catalog/cohort.csv",
        args.out / "corpus_scan.json",
    )
    cohort, previous = pd.read_csv(cohort_path), None
    if args.reuse_scan and scan_path.is_file():
        previous_scan = json.loads(scan_path.read_text())
        if previous_scan["cohort_sha256"] != sha256_of(cohort_path):
            raise ValueError("cohort changed since scan")
        previous = previous_scan["records"]
    started = time.monotonic()
    records = scan(original, cohort, previous)
    producer = producer_snapshot([producer_root, producer_tables])
    queue, excluded = queue_records(records, producer, cohort)
    roster_path = REPO / "data/events/detachment/shots.csv"
    rosters.write_roster(roster_frame(queue, roster_path), roster_path, keep_order=True)
    (REPO / "data/events/detachment/shots_review.csv").unlink(missing_ok=True)
    isolated = args.out / "tables/detachment/shots.csv"
    isolated.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(roster_path, isolated)
    cohort_copy = args.out / "tables/catalog/cohort.csv"
    cohort_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cohort_path, cohort_copy)
    population = original.catalog / "population.csv"
    if population.is_file():
        population_copy = args.out / "catalog/population.csv"
        population_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(population, population_copy)
    camera_shots = [
        r["shot"] for r in queue if "cohort_camera_geometry" in r["queue_sources"]
    ]
    producer_shots = [
        r["shot"] for r in queue if "producer_labels_or_votes" in r["queue_sources"]
    ]
    summary = {
        "scanned_train_val": len(records),
        "camera_and_geometry_eligible": len(camera_shots),
        "producer_labels_or_votes_nonblind": len(producer_shots),
        "overlap": len(set(camera_shots) & set(producer_shots)),
        "queue": len(queue),
        "by_split": {
            s: sum(r["split"] == s for r in queue)
            for s in ("train", "val", "producer_external")
        },
        "blind_cohort_not_scanned": int((cohort.split == "test").sum()),
        "producer_blind_excluded": excluded,
    }
    record = {
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(full=True),
        "source_sha256": {
            str(p.relative_to(REPO)): sha256_of(p)
            for p in (Path(__file__), REPO / "src/labeler/events/review/geometry.py")
        },
        "cohort": str(cohort_path),
        "cohort_sha256": sha256_of(cohort_path),
        "population": str(population) if population.is_file() else None,
        "population_sha256": sha256_of(population) if population.is_file() else None,
        "policy": "union: train/val with finite spatially nonconstant lower TangTV corpus preview inside plasma window and coincident with producer EFIT shelf gate (nearest <=40 ms), plus producer labelled or draft-vote shots; exclude every cohort or explicit producer test shot",
        "geometry_thresholds": geometry.THRESHOLDS,
        "geometry_root": str(geometry_root),
        "producer_roots": [str(producer_root), str(producer_tables)],
        "producer_snapshot": producer,
        "roster": str(roster_path),
        "roster_sha256": sha256_of(roster_path),
        "summary": summary,
        "camera_geometry_shots": camera_shots,
        "producer_labelled_or_vote_shots": producer_shots,
        "queue_shots": [r["shot"] for r in queue],
        "original_camera_candidates": original_camera,
        "scan_seconds": time.monotonic() - started,
    }
    scan_path.write_text(json.dumps({**record, "records": records}, indent=2) + "\n")
    if args.build:
        os.environ["LABELER_DETACHMENT_INDICATORS"] = str(indicators)
        paths = Paths(
            root=args.out, corpus=original.corpus, label_tables=args.out / "tables"
        )
        targets = {r["shot"]: r for r in queue}
        blind = set(cohort[cohort.split.eq("test")].shot) | set(
            producer["explicit_test_shots"]
        )
        if args.rebuild_existing:
            split = dict(zip(cohort.shot, cohort.split, strict=True))
            for store in (args.out / "spectrograms/detachment").glob("*.h5"):
                if store.stem.isdigit() and int(store.stem) not in blind:
                    shot = int(store.stem)
                    targets.setdefault(
                        shot,
                        {"shot": shot, "split": split.get(shot, "producer_external")},
                    )
        built = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = [
                pool.submit(build_store, r, paths, indicators, args.resume_build)
                for r in targets.values()
            ]
            for future in as_completed(pending):
                result = future.result()
                built.append(result)
                print(f"build {result['shot']}: ok", flush=True)
        built.sort(key=lambda r: r["shot"])
        for name, subset in (
            ("all_rebuilt", built),
            ("queue", [r for r in built if r["shot"] in set(record["queue_shots"])]),
            (
                "original_camera_candidates",
                [r for r in built if r["shot"] in set(original_camera["shots"])],
            ),
        ):
            record["summary"][f"density_sources_{name}"] = {
                s: sum(r["density_source"] == s for r in subset)
                for s in ("CO2", "Thomson", "aux_ne", "unavailable")
            }
        record["summary"]["stores_built"] = len(built)
        record["summary"]["stores_rebuilt"] = sum(
            r["action"] == "rebuilt" for r in built
        )
        record["summary"]["stores_kept_current"] = sum(
            r["action"] != "rebuilt" for r in built
        )
        record["summary"]["outside_store_frames"] = sum(
            a["outside_store_frames"] for r in built for a in r["video_audit"]
        )
        record["summary"]["outside_plasma_frames"] = sum(
            a["outside_plasma_frames"] for r in built for a in r["video_audit"]
        )
        record["summary"]["stores_without_plasma_window"] = [
            r["shot"] for r in built if r["plasma_window_ms"] is None
        ]
        source_files = (
            REPO / "src/labeler/events/review/video.py",
            REPO / "src/labeler/events/panels/detachment.py",
            REPO / "src/labeler/events/review/detachment.py",
        )
        record["source_sha256"].update(
            {str(p.relative_to(REPO)): sha256_of(p) for p in source_files}
        )
        (args.out / "roster_build.json").write_text(
            json.dumps(
                {**record, "indicator_root": str(indicators), "stores": built}, indent=2
            )
            + "\n"
        )
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["summary"]))


if __name__ == "__main__":
    main()
