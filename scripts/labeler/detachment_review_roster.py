"""Write the real detachment review queue using local, read-only evidence.

Consume the producer-owned roster and label outputs, checking camera availability
independently of EFIT. The delivery overlay excludes blind shots and, by default,
shots without live lower-divertor video. Producer inputs are never written.
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
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from zipfile import BadZipFile

import h5py
import numpy as np
import pandas as pd

from labeler.config import Paths, git_sha, sha256_of
from labeler.events import rosters
from labeler.events.review import (
    build,
    detachment,
    geometry,
    labels,
    recipe,
    rows,
    video,
)
from labeler.events.review import producer as producer_review

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "docs/labeler/results/detachment_review_queue.json"


def write_recipe_help(labels_path):
    """Update the docs' machine block from the same read-only interpretation."""
    document = REPO / "docs/labeler/detachment_review.md"
    if not document.is_file():
        return
    before, separator, rest = document.read_text().partition(
        "<!-- MACHINE_RECIPE_START -->"
    )
    if not separator:
        return
    _, end, after = rest.partition("<!-- MACHINE_RECIPE_END -->")
    if not end:
        raise ValueError("missing machine-recipe end marker")
    frozen = recipe.load(labels_path)
    text = frozen["documentation"] or json.dumps(frozen["record"], indent=2)
    sources = "\n".join(
        f"- `{key}`: `{value['sha256']}`"
        for key, value in frozen["sources"].items()
        if value["sha256"]
    )
    block = (
        "\n<details>\n<summary>Producer wording (generated recipe snapshot)</summary>\n\n"
        "````text\n"
        + text
        + "\n````\n\nSource SHA256:\n\n"
        + sources
        + "\n\n</details>\n"
    )
    document.write_text(before + separator + block + end + after)


def label_keys(keys):
    """Recognise producer states, grids, and draft indicator-vote outputs."""
    return any(
        key in {"category", "state", "state_lm", "state_rule", "labels", "prob"}
        or key.endswith("_vote")
        for key in keys
    )


def blind_flags(table, size):
    """Per-row split/holdout flags, with scalar NPZ flags applied to all rows."""
    blind = np.zeros(size, dtype=bool)
    for key, reserved in (("split", {"test"}), ("holdout", {"true", "1", "yes"})):
        if key not in table:
            continue
        values = np.asarray(table[key]).astype(str).ravel()
        if values.size == 1:
            values = np.repeat(values, size)
        if values.size != size:
            raise ValueError(f"{key} flag count does not match shot count")
        blind |= np.isin(np.char.lower(np.char.strip(values)), list(reserved))
    return blind


def input_files(roots, pattern):
    """Accept explicit files or directories, without discovering unrelated runs."""
    return sorted(
        {
            path
            for root in map(Path, roots)
            for path in ([root] if root.is_file() else root.rglob(pattern))
            if path.is_file()
        }
    )


def producer_snapshot(roots, roster_path=None):
    """Read completed local label/vote files, retaining an exact input manifest.

    A producer can still be running: partial/unreadable files are recorded and
    retried on the next invocation. Camera inversions, surveys and placeholder
    rosters carry no label/vote field and do not count.
    """
    shots, explicit_test, sources, errors = set(), set(), [], []
    csvs = input_files(roots, "*.csv*")
    for path in csvs:
        if not path.name.endswith((".csv", ".csv.gz")):
            continue
        try:
            payload = path.read_bytes()
            compression = "gzip" if path.suffix == ".gz" else None
            columns = pd.read_csv(
                io.BytesIO(payload), nrows=0, compression=compression
            ).columns
            if not label_keys(columns):
                continue
            usecols = [key for key in ("shot", "split", "holdout") if key in columns]
            filename_shot = path.name.removesuffix(".gz").removesuffix(".csv")
            if "shot" in columns:
                table = pd.read_csv(
                    io.BytesIO(payload), usecols=usecols, compression=compression
                )
                values = pd.to_numeric(table.shot, errors="coerce")
                valid = values.notna() & (values % 1 == 0) & (values >= 100000)
                found = set(values[valid].astype(int))
                explicit_test.update(
                    values[valid & blind_flags(table, len(table))].astype(int)
                )
            elif filename_shot.isdigit() and int(filename_shot) >= 100000:
                table = pd.read_csv(io.BytesIO(payload), compression=compression)
                found = {int(filename_shot)}
                if blind_flags(table, len(table)).any():
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
        except (OSError, ValueError, pd.errors.ParserError, EOFError) as error:
            errors.append({"path": str(path), "error": str(error)})
    npzs = input_files(roots, "*.npz")
    for path in npzs:
        if path.suffix != ".npz":
            continue
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
                    explicit_test.update(
                        values[valid & blind_flags(grid, len(values))]
                        .astype(int)
                        .tolist()
                    )
                elif path.stem.isdigit() and int(path.stem) >= 100000:
                    found = {int(path.stem)}
                    size = max(
                        (
                            np.asarray(grid[k]).size
                            for k in ("split", "holdout")
                            if k in grid
                        ),
                        default=1,
                    )
                    if blind_flags(grid, size).any():
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
    roster, roster_shots = None, []
    if roster_path is not None:
        frame = rosters.read_roster(roster_path)
        roster_shots = frame.shot.astype(int).tolist()
        explicit_test.update(frame.loc[frame.holdout.eq("true"), "shot"].astype(int))
        roster = {
            "path": str(roster_path),
            "sha256": sha256_of(roster_path),
            "shots": roster_shots,
        }
    return {
        "shots": sorted(shots),
        "roster_shots": sorted(roster_shots),
        "roster": roster,
        "explicit_test_shots": sorted(explicit_test),
        "sources": sources,
        "errors": errors,
    }


def snapshot_suggestions(
    source_path,
    queue_shots,
    event_dir,
    reserved_shots=(),
    indicator_root=None,
    windows=None,
):
    """Freeze primary 50 ms producer labels for the isolated Source lane.

    Category zero preserves the assessed window while remaining an empty span in
    the review reader. Missing/unassessed bins never become attached labels.
    """
    source_path, event_dir = Path(source_path), Path(event_dir)
    payload = source_path.read_bytes()
    fingerprint = hashlib.sha256(payload).hexdigest()
    frame = pd.read_csv(
        io.BytesIO(payload),
        keep_default_na=False,
        low_memory=False,
        compression="gzip" if source_path.name.endswith(".gz") else None,
    )
    required = {"shot", "start_ms", "state_lm"}
    if not required <= set(frame.columns):
        raise ValueError(f"producer label table requires {sorted(required)}")
    values = pd.to_numeric(frame.shot, errors="coerce")
    valid_shot = values.notna() & (values % 1 == 0) & (values >= 100000)
    explicit_blind = set(
        values[valid_shot & blind_flags(frame, len(frame))].astype(int)
    )
    blind = explicit_blind | set(reserved_shots)
    selected = frame[valid_shot & values.isin(set(queue_shots) - blind)].copy()
    selected["shot"] = values[selected.index].astype(int)
    state = pd.to_numeric(selected.state_lm, errors="coerce")
    start = pd.to_numeric(selected.start_ms, errors="coerce")
    if not (state.isin(range(5)) & np.isfinite(start)).all():
        raise ValueError("producer label bins need finite starts and states 0..4")
    if selected.duplicated(["shot", "start_ms"]).any():
        raise ValueError("duplicate producer label bin")
    indicator_fingerprints, inconsistencies = {}, []
    if indicator_root is not None:
        indicator_root = Path(indicator_root)
        for shot, shot_rows in selected.groupby("shot", sort=True):
            bins_path = indicator_root / f"{int(shot)}.npz"
            indicator_fingerprints[str(shot)] = {
                "path": str(bins_path),
                "sha256": None,
            }
            if not bins_path.is_file():
                continue
            try:
                bins_payload = bins_path.read_bytes()
                bins_fingerprint = hashlib.sha256(bins_payload).hexdigest()
                indicator_fingerprints[str(shot)]["sha256"] = bins_fingerprint
                with np.load(io.BytesIO(bins_payload), allow_pickle=False) as bins:
                    if not set(shot_rows.start_ms).issubset(set(bins["start_ms"])):
                        raise ValueError("producer labels and vote bin clocks disagree")
                    producer_review._verify_snapshot(shot_rows, bins)
            except (
                OSError,
                ValueError,
                KeyError,
                AttributeError,
                BadZipFile,
                EOFError,
                IndexError,
                TypeError,
            ) as error:
                inconsistencies.append(
                    {
                        "shot": int(shot),
                        "reason": f"Producer snapshot inconsistent: {error}",
                        "producer_sha256": fingerprint,
                        "indicator": indicator_fingerprints[str(shot)],
                    }
                )
        excluded = {item["shot"] for item in inconsistencies}
        selected = selected[~selected.shot.isin(excluded)]
        state, start = state[selected.index], start[selected.index]
    consistency = {
        "indicator_root": str(indicator_root) if indicator_root is not None else None,
        "indicator_fingerprints": indicator_fingerprints,
        "excluded_inconsistent_shots": [item["shot"] for item in inconsistencies],
        "inconsistencies": inconsistencies,
    }
    suggestions = pd.DataFrame(
        {
            "shot": selected.shot,
            "category": state.astype(int),
            "t_start": start,
            "t_end": start + 50.0,
            "confidence": selected.get("confidence", np.nan),
            "attrs": [
                json.dumps(
                    {
                        "producer_bin_start_ms": float(value),
                        "producer_bin_end_ms": float(value + 50.0),
                    }
                )
                for value in start
            ],
        }
    ).sort_values(["shot", "t_start"])
    windows = windows or {}
    for shot, window in windows.items():
        if window is None:
            continue
        mask = suggestions.shot.eq(shot)
        suggestions.loc[mask, "t_start"] = suggestions.loc[mask, "t_start"].clip(
            lower=window[0]
        )
        suggestions.loc[mask, "t_end"] = suggestions.loc[mask, "t_end"].clip(
            upper=window[1]
        )
    suggestions = suggestions[suggestions.t_end > suggestions.t_start]
    path = event_dir / "review/suggestions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    suggestions.to_csv(path, index=False)
    entry = labels.write_pointer(
        event_dir,
        path,
        method="detach_vote unverified producer suggestions",
        version=fingerprint,
    )
    entry.update(
        producer_table=str(source_path),
        producer_sha256=fingerprint,
        state_column="state_lm",
        bin_ms=50,
        category_zero="not assessed",
        **consistency,
    )
    labels.pointer_path(event_dir).write_text(json.dumps(entry, indent=2) + "\n")
    return {
        "producer_table": str(source_path),
        "producer_sha256": fingerprint,
        "table": str(path),
        "sha256": sha256_of(path),
        "pointer": str(labels.pointer_path(event_dir)),
        "state_column": "state_lm",
        "bin_ms": 50,
        "rows": len(suggestions),
        "shots": sorted(suggestions.shot.unique().astype(int).tolist()),
        "explicit_test_shots": sorted(explicit_blind),
        "reserved_shots": sorted(reserved_shots),
        "synchronized_windows_ms": {str(k): v for k, v in windows.items()},
        **consistency,
    }


def delivery_holdouts(roster_path):
    """Retain UI reservations after reserved rows leave the review queue.

    The registry is delivery-owned; no producer table is changed. Reservations
    persist across regenerations until explicitly removed from the registry.
    """
    roster_path = Path(roster_path)
    registry = roster_path.parent / "review/holdouts.json"
    retained = (
        set(json.loads(registry.read_text())["shots"]) if registry.is_file() else set()
    )
    current = set()
    if roster_path.is_file():
        frame = rosters.read_roster(roster_path)
        current = set(frame.loc[frame.holdout.eq("true"), "shot"].astype(int))
    shots = retained | current
    if any(not isinstance(shot, int) or shot < 100000 for shot in shots):
        raise ValueError("delivery holdouts must be physical integer shot numbers")
    record = {
        "path": str(registry),
        "shots": sorted(shots),
        "overlay": str(roster_path),
        "overlay_sha256": sha256_of(roster_path) if roster_path.is_file() else None,
        "overlay_holdout_shots": sorted(current),
        "retained_holdout_shots": sorted(retained),
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(record, indent=2) + "\n")
    return {**record, "sha256": sha256_of(registry)}


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
    """Refresh live in-plasma previews independently of EFIT availability.

    Accept the older caller's ``previous`` argument without reusing its negative
    decisions: a prior scan may have used an older camera-layout reader.
    """
    records = []
    selected = cohort[~cohort.split.eq("test")].sort_values("queue_rank")
    for row in selected.itertuples(index=False):
        path = paths.corpus_file(int(row.shot))
        record = {
            "shot": int(row.shot),
            "split": row.split,
            "corpus": str(path),
            "window_ms": (
                [float(row.window_start_ms), float(row.window_end_ms)]
                if np.isfinite(row.window_start_ms) and np.isfinite(row.window_end_ms)
                else None
            ),
            "lower_channels": [],
            "camera_available": False,
            "camera_geometry_eligible": False,
            "reason": "corpus missing",
        }
        record["geometry"] = geometry.load(row.shot, paths, record["window_ms"])
        geometry_path = Path(record["geometry"]["source"])
        record["geometry_sha256"] = (
            sha256_of(geometry_path) if geometry_path.is_file() else None
        )
        record["geometry_reason"] = record["geometry"].get("reason") or (
            "" if record["geometry"]["shelf_gate_samples"] else "no shelf-gate sample"
        )
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
                if len(t) < 2:
                    record["reason"] = "tangtv one-sample stub"
                    continue
                indices = video.frame_indices(t)
                if record["window_ms"]:
                    indices = indices[
                        (t[indices] * 1000 >= row.window_start_ms)
                        & (t[indices] * 1000 <= row.window_end_ms)
                    ]
                _, valid_geometry = geometry.at_times(
                    record["geometry"], t[indices] * 1000
                )
                record["geometry_valid_preview_times"] = int(valid_geometry.sum())
                record["preview_indices_checked"] = {}
                for channel in (2, 0):
                    if channel >= channels:
                        continue
                    checked, order = 0, list(indices[valid_geometry])
                    gate_indices = set(order)
                    if order:
                        order.insert(0, order.pop(len(order) // 2))
                    remainder = [i for i in indices if i not in gate_indices]
                    if not order and remainder:
                        remainder.insert(0, remainder.pop(len(remainder) // 2))
                    order.extend(remainder)
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
                            if index in indices[valid_geometry]:
                                record["camera_geometry_eligible"] = True
                            break
                    record["preview_indices_checked"][str(channel)] = checked
                record["camera_available"] = bool(record["lower_channels"])
                record["reason"] = (
                    "" if record["camera_available"] else "no live lower TangTV preview"
                )
        except (ValueError, OSError) as error:
            record["reason"] = str(error)
        print(f"scan {row.shot}: {record['reason'] or 'eligible'}", flush=True)
    return records


def queue_records(records, producer, cohort, include_no_video=False, reserved_shots=()):
    """Source union with blind exclusions and a camera-first review overlay."""
    cohort_rows = {int(r.shot): r for r in cohort.itertuples(index=False)}
    blind = {int(r.shot) for r in cohort_rows.values() if r.split == "test"}
    blind.update(producer["explicit_test_shots"])
    blind.update(reserved_shots)
    scanned = {r["shot"]: r for r in records}
    eligible = {r["shot"]: dict(r) for r in records if r["lower_channels"]}
    labelled = set(producer["shots"])
    rostered = set(producer.get("roster_shots", []))
    candidates = set(eligible) | labelled | rostered
    queue = []
    for shot in sorted(candidates):
        if shot in blind:
            continue
        row = cohort_rows.get(shot)
        record = dict(
            scanned.get(shot)
            or {
                "shot": shot,
                "split": row.split if row else "producer_external",
                "lower_channels": [],
                "window_ms": [int(row.window_start_ms), int(row.window_end_ms)]
                if row
                else None,
            }
        )
        record["camera_available"] = bool(record["lower_channels"])
        if not record["camera_available"] and not include_no_video:
            continue
        record["queue_sources"] = (
            (["cohort_camera"] if shot in eligible and row is not None else [])
            + (["producer_roster"] if shot in rostered else [])
            + (["producer_labels_or_votes"] if shot in labelled else [])
        )
        queue.append(record)
    queue.sort(key=lambda record: (not record["camera_available"], record["shot"]))
    return queue, sorted(blind & candidates)


def roster_frame(queue, roster_path, producer_roster=None):
    """Keep producer curation and retained UI edits in the delivery overlay."""
    old = {}
    for path in (producer_roster, roster_path):
        if path is not None and Path(path).is_file():
            old.update(
                {
                    int(r.shot): r
                    for r in rosters.read_roster(path).itertuples(index=False)
                    if int(r.shot) >= 100000 and "EXAMPLE" not in r.notes
                }
            )
    output = []
    for record in queue:
        shot = record["shot"]
        note = f"{record['split']}; " + "; ".join(record["queue_sources"])
        note += (
            "; camera available"
            if record.get("camera_available", False)
            else "; NO LOWER-DIVERTOR VIDEO"
        )
        if "geometry" in record:
            g = record["geometry"]
            if g.get("reason", "").startswith("efit_"):
                note += "; EFIT missing or unreadable"
            note += (
                f"; EFIT shelf {g['shelf_gate_samples']}/{g['total_samples']} samples"
            )
            if g.get("configuration_available"):
                note += "; config " + "/".join(
                    k for k, v in g["counts"].items() if v and k != "missing"
                )
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


def context_summary(panel_metadata, producer):
    """Audit the density hierarchy, independent Te and front-height provenance."""
    panels = list(panel_metadata.values())
    density = "unavailable"
    for key, value, name in (
        ("quantity", "aux_ne", "aux_ne"),
        ("node", "DENR0UF", "cached_DENR0UF"),
        ("corpus_group", "co2", "corpus_CO2"),
        ("corpus_group", "ts_core_density", "Thomson"),
    ):
        if any(panel.get(key) == value for panel in panels):
            density = name
            break
    tangtv = next((p for p in panels if p.get("indicator") == "tangtv"), {})
    source_bins = Counter(
        source
        for source, valid in zip(
            tangtv.get("tangtv_source", []), tangtv.get("valid", []), strict=True
        )
        if valid
    )
    return {
        "density_source": density,
        "divertor_te_available": any(p.get("quantity") == "aux_te_div" for p in panels),
        "tangtv_valid_source_bins": dict(source_bins),
        "producer_strip_bins": len(producer.get("bin_start_ms", [])),
        "producer_strip_reason": producer.get("reason"),
        "producer_label_available": producer.get("label_available", False),
        "producer_state_bins": dict(Counter(producer.get("state_lm", []))),
    }


def build_targets(queue, cohort, blind, store_root, rebuild_existing=False):
    """Select queue/prior stores only after every holdout source is combined."""
    targets = {
        record["shot"]: record for record in queue if record["shot"] not in blind
    }
    if rebuild_existing:
        split = dict(zip(cohort.shot, cohort.split, strict=True))
        for store in Path(store_root).glob("*.h5"):
            if store.stem.isdigit() and int(store.stem) not in blind:
                shot = int(store.stem)
                targets.setdefault(
                    shot, {"shot": shot, "split": split.get(shot, "producer_external")}
                )
    return targets


def build_store(record, paths, indicators, resume=False):
    started = time.monotonic()
    store = paths.spectrogram_file("detachment", record["shot"])
    current = build.current(store, "detachment")
    desired_sources = detachment.context_sources(record["shot"], paths)
    refreshed_recipe = False
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
            without_recipe = lambda values: {
                k: v for k, v in values.items() if not k.startswith("recipe_")
            }
            if not current and without_recipe(previous_sources) == without_recipe(
                desired_sources
            ):
                # Interpretation-only updates do not change any stored pixels
                # or diagnostic values. Refresh the help and guides in place.
                frozen = recipe.load(producer_review.source_path(paths))
                with h5py.File(store, "r+") as source:
                    params = json.loads(source.attrs["params"])
                    params["detachment_producer"]["recipe"] = frozen
                    params["detachment_producer"]["definitions"] = frozen["record"].get(
                        "definitions", {}
                    )
                    params["context_sources"] = desired_sources
                    for name, panel in params.get("panel_metadata", {}).items():
                        if "indicator" not in panel:
                            continue
                        panel["recipe_sources"] = frozen["sources"]
                        group = source[f"rows/{name}"]
                        meta = json.loads(group.attrs["meta"])
                        meta["hlines"] = recipe.guides(
                            frozen["record"], panel["indicator"]
                        )
                        group.attrs["meta"] = json.dumps(meta)
                    source.attrs["params"] = json.dumps(params)
                current = refreshed_recipe = True
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
        params = json.loads(source.attrs.get("params", "{}"))
    stored_geometry = params.get("detachment_geometry", {})
    context = context_summary(
        params.get("panel_metadata", {}), params.get("detachment_producer", {})
    )
    return {
        "shot": record["shot"],
        "split": record["split"],
        "store": str(store),
        "bytes": store.stat().st_size,
        "seconds": time.monotonic() - started,
        "action": "rebuilt"
        if rebuild
        else "refreshed recipe"
        if refreshed_recipe
        else "kept current store",
        **context,
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
    parser.add_argument(
        "--producer-roster",
        type=Path,
        help="read-only producer roster (default: producer tables/shots.csv)",
    )
    parser.add_argument(
        "--producer-labels",
        type=Path,
        help="primary full-bin labels (default: producer root/labels_bins.csv.gz)",
    )
    parser.add_argument(
        "--label-source",
        type=Path,
        action="append",
        help="additional read-only label/vote file or directory; repeatable",
    )
    parser.add_argument(
        "--include-no-video",
        action="store_true",
        help="keep explicitly flagged no-video producer shots at the end",
    )
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
    producer_roster = args.producer_roster or producer_tables / "shots.csv"
    producer_labels = args.producer_labels or producer_root / "labels_bins.csv.gz"
    producer_inputs = [
        producer_labels,
        producer_root / "labels_rule.csv",
        producer_root / "bins",
        *(args.label_source or []),
    ]
    geometry_root, indicators = (
        args.geometry_root or producer_root / "cache",
        args.indicator_root or producer_root / "bins",
    )
    os.environ["LABELER_DETACHMENT_GEOMETRY_ROOT"] = str(geometry_root)
    os.environ["LABELER_DETACHMENT_INDICATORS"] = str(indicators)
    os.environ["LABELER_DETACHMENT_LABELS"] = str(producer_labels)
    os.environ["LABELER_DETACHMENT_CACHE_ROOT"] = str(producer_root / "cache")
    write_recipe_help(producer_labels)
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
    producer = producer_snapshot(producer_inputs, producer_roster)
    isolated = args.out / "tables/detachment/shots.csv"
    if isolated.resolve() == producer_roster.resolve():
        raise ValueError("review overlay cannot replace the producer-owned roster")
    reservations = delivery_holdouts(isolated)
    reserved_shots = set(reservations["shots"])
    blind = (
        set(cohort.loc[cohort.split.eq("test"), "shot"].astype(int))
        | set(producer["explicit_test_shots"])
        | reserved_shots
    )
    candidate_cohort = cohort[~cohort.shot.isin(blind)].reset_index(drop=True)
    external = (
        (set(producer["shots"]) | set(producer["roster_shots"]))
        - set(cohort.shot)
        - blind
    )
    for rank, shot in enumerate(sorted(external), start=len(cohort) + 1):
        window = geometry.current_window(shot, original)
        candidate_cohort.loc[len(candidate_cohort)] = {
            "shot": shot,
            "split": "producer_external",
            "queue_rank": rank,
            "window_start_ms": window[0] if window else np.nan,
            "window_end_ms": window[1] if window else np.nan,
        }
    records = scan(original, candidate_cohort, previous)
    queue, excluded = queue_records(
        records,
        producer,
        cohort,
        include_no_video=args.include_no_video,
        reserved_shots=reserved_shots,
    )
    isolated.parent.mkdir(parents=True, exist_ok=True)
    rosters.write_roster(
        roster_frame(queue, isolated, producer_roster), isolated, keep_order=True
    )
    suggestions = snapshot_suggestions(
        producer_labels,
        [r["shot"] for r in queue],
        isolated.parent,
        reserved_shots=reserved_shots,
        indicator_root=indicators,
        windows={r["shot"]: r["window_ms"] for r in queue},
    )
    cohort_copy = args.out / "tables/catalog/cohort.csv"
    cohort_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cohort_path, cohort_copy)
    population = original.catalog / "population.csv"
    if population.is_file():
        population_copy = args.out / "catalog/population.csv"
        population_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(population, population_copy)
    camera_shots = [r["shot"] for r in queue if r["camera_available"]]
    producer_shots = [
        r["shot"] for r in queue if "producer_labels_or_votes" in r["queue_sources"]
    ]
    summary = {
        "scanned_train_val": sum(r["split"] in ("train", "val") for r in records),
        "scanned_producer_external": sum(
            r["split"] == "producer_external" for r in records
        ),
        "camera_available": len(camera_shots),
        "camera_and_geometry_eligible": sum(
            r["camera_geometry_eligible"] for r in records
        ),
        "camera_with_any_shelf_sample_scan": sum(
            r["geometry"]["shelf_gate_samples"] > 0 for r in queue
        ),
        "missing_efit": sum(r["geometry_reason"] == "efit_missing" for r in records),
        "unreadable_efit": sum(
            r["geometry_reason"].startswith("efit_unreadable") for r in records
        ),
        "no_video_excluded": sum(
            not r["camera_available"]
            and r["shot"] in (set(producer["shots"]) | set(producer["roster_shots"]))
            for r in records
        )
        if not args.include_no_video
        else 0,
        "producer_roster_nonblind": len(set(producer["roster_shots"]) - blind),
        "producer_labels_or_votes_nonblind": len(producer_shots),
        "overlap": len(set(camera_shots) & set(producer_shots)),
        "queue": len(queue),
        "by_split": {
            s: sum(r["split"] == s for r in queue)
            for s in ("train", "val", "producer_external")
        },
        "blind_cohort_not_scanned": int((cohort.split == "test").sum()),
        "producer_blind_excluded": sorted(set(excluded) - reserved_shots),
        "delivery_holdout_excluded": sorted(reserved_shots),
    }
    record = {
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(full=True),
        "source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[2])): sha256_of(p)
            for p in (Path(__file__), REPO / "src/labeler/events/review/geometry.py")
            if p.is_file()
        },
        "cohort": str(cohort_path),
        "cohort_sha256": sha256_of(cohort_path),
        "population": str(population) if population.is_file() else None,
        "population_sha256": sha256_of(population) if population.is_file() else None,
        "policy": (
            "union of producer roster, labels/votes and cohort live lower TangTV "
            "previews inside the plasma window; exclude cohort test, explicit "
            "split=test, producer holdout=true and retained UI holdout shots; "
            "camera availability is separate "
            "from EFIT shelf gate; camera shots first; "
            + (
                "flag no-video shots last"
                if args.include_no_video
                else "exclude no-video shots"
            )
        ),
        "geometry_thresholds": geometry.THRESHOLDS,
        "geometry_root": str(geometry_root),
        "producer_roots": [str(p) for p in producer_inputs],
        "producer_roster": str(producer_roster),
        "producer_roster_after_sha256": sha256_of(producer_roster),
        "producer_snapshot": producer,
        "roster": str(isolated),
        "roster_sha256": sha256_of(isolated),
        "suggestions": suggestions,
        "delivery_holdouts": reservations,
        "summary": summary,
        "camera_shots": camera_shots,
        "camera_geometry_shots": [
            r["shot"] for r in queue if r["camera_geometry_eligible"]
        ],
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
        targets = build_targets(
            queue,
            cohort,
            blind,
            args.out / "spectrograms/detachment",
            rebuild_existing=args.rebuild_existing,
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
                for s in (
                    "aux_ne",
                    "cached_DENR0UF",
                    "corpus_CO2",
                    "Thomson",
                    "unavailable",
                )
            }
            record["summary"][f"divertor_te_available_{name}"] = sum(
                r["divertor_te_available"] for r in subset
            )
            record["summary"][f"producer_strip_available_{name}"] = sum(
                r["producer_strip_bins"] > 0 for r in subset
            )
            record["summary"][f"producer_label_available_{name}"] = sum(
                r["producer_label_available"] for r in subset
            )
            source_bins = Counter()
            for item in subset:
                source_bins.update(item["tangtv_valid_source_bins"])
            record["summary"][f"tangtv_valid_source_bins_{name}"] = dict(source_bins)
        record["summary"]["stores_built"] = len(built)
        record["summary"]["camera_with_any_shelf_sample"] = sum(
            r["geometry_shelf_gate_samples"] > 0
            for r in built
            if r["shot"] in set(record["queue_shots"])
        )
        record["summary"]["stores_rebuilt"] = sum(
            r["action"] == "rebuilt" for r in built
        )
        record["summary"]["stores_kept_current"] = sum(
            r["action"] == "kept current store" for r in built
        )
        record["summary"]["stores_refreshed_recipe"] = sum(
            r["action"] == "refreshed recipe" for r in built
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
            REPO / "src/labeler/events/review/producer.py",
            REPO / "src/labeler/events/review/recipe.py",
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
