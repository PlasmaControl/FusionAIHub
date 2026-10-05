"""Pin the sawtooth export that Figure 1 reads and record competing catalog sources.

Read-only inputs; writes immutable copies under `$LABELER_ROOT/round4/fig1b/` and a
small manifest in this worktree. The source is the saw stream's final population
export, `$LABELER_ROOT/round4/saw/fix5/`, which outlives any stream worktree: its
`labels/SHA256SUMS` pins every population shard, and that file's own digest is pinned
here (`--expect-sums-sha256`). The figure renders from the shot's physics record
(`shots/<shot>.json`, which also carries the density guard); this script checks that
record's states and crash points equal the shot's rows in the hashed population
shard, so the label table and the drawn row are the same labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from labeler.config import Paths, sha256_of
from labeler.paper import figure_sources as fs
from labeler.paper import label_figure as lf
from labeler.paper import mode_tags as mt

#: Figure 1's shot and window (ms): AE, ELMs, H-mode, an NTM and sawtooth all appear.
WINDOWS = {199563: (700, 5800)}

#: sha256 of `labels/SHA256SUMS` in the pinned export (the stream's documented value).
EXPORT_SUMS_SHA256 = "47081aebcad868911bf3e38817b3e0c79e4ba9f02efe5fcee84780ad00a0ef14"

#: Earlier catalog suggestions, compared in the manifest but never drawn.
CATALOG = (("ece_sawtooth", "v3"), ("sawtooth_frames", "v2"))


def read_sums(path: Path) -> dict[str, str]:
    """`SHA256SUMS` as {file name: digest}."""
    sums = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip().removeprefix("*")] = digest
    return sums


def shard_of(labels: Path, sums: dict[str, str], shot: int) -> Path:
    """The one CSV shard that holds the shot's rows."""
    found = [
        labels / name
        for name in sorted(sums)
        if name.endswith(".csv")
        and (pd.read_csv(labels / name, usecols=["shot"])["shot"] == shot).any()
    ]
    if len(found) != 1:
        raise SystemExit(f"{shot}: expected one population shard, found {found}")
    return found[0]


def population_rows(shard: Path, shot: int):
    """(interval rows, present point times) of the shot in a population shard."""
    frame = pd.read_csv(shard)
    frame = frame.loc[frame["shot"] == shot]
    spans = sorted(
        (round(float(r.t_start), 1), round(float(r.t_end), 1), int(r.category))
        for r in frame.itertuples()
        if r.t_end > r.t_start
    )
    points = sorted(
        round(float(r.t_start), 1)
        for r in frame.itertuples()
        if r.t_end == r.t_start and int(r.category) == fs.PRESENT
    )
    return spans, points


def physics_rows(record: dict):
    """The same quantities from the shot's physics record."""
    codes = {
        "absent": fs.ABSENT,
        "present": fs.PRESENT,
        "uncertain": fs.UNCERTAIN,
        "q_prior_ece_contradicted": fs.UNCERTAIN,
        "q_prior_untested": fs.UNCERTAIN,
        "unassessed": fs.NOT_OBSERVABLE,
    }
    spans = sorted(
        (
            round(r["start_s"] * 1000, 1),
            round(r["end_s"] * 1000, 1),
            codes[r["state"]],
        )
        for r in record["states"]
    )
    points = sorted(
        round(c["time_s"] * 1000, 1)
        for c in record["crashes"]
        if c.get("attrs", {}).get("state", "present") == "present"
    )
    return spans, points


#: Population shards are rounded to the 0.1 ms sample grid; an edge may differ by one.
EDGE_TOLERANCE_MS = 0.15


def same_labels(physics, population) -> bool:
    """Equal state intervals (edges within the grid rounding) and equal crash points."""
    (spans, points), (other_spans, other_points) = physics, population
    if len(spans) != len(other_spans) or len(points) != len(other_points):
        return False
    return all(
        a[2] == b[2]
        and abs(a[0] - b[0]) < EDGE_TOLERANCE_MS
        and abs(a[1] - b[1]) < EDGE_TOLERANCE_MS
        for a, b in zip(spans, other_spans, strict=True)
    ) and all(
        abs(a - b) < EDGE_TOLERANCE_MS
        for a, b in zip(points, other_points, strict=True)
    )


def main():
    paths = Paths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        type=Path,
        default=paths.root / "round4/saw/fix5",
        help="the saw stream's final export (default $LABELER_ROOT/round4/saw/fix5)",
    )
    parser.add_argument("--expect-sums-sha256", default=EXPORT_SUMS_SHA256)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=paths.root / "round4/fig1b/saw_source",
        help="immutable copies read by the render",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "outputs/labeler/paper/fig_interpreter_tokeye/fig1b/"
            "sawtooth_source_manifest.json"
        ),
    )
    args = parser.parse_args()
    labels = args.export / "labels"
    sums_file = labels / "SHA256SUMS"
    assert sha256_of(sums_file) == args.expect_sums_sha256, "export SHA256SUMS changed"
    sums = read_sums(sums_file)
    completion_file = args.export / "cohort_labels.json"
    complete = json.loads(completion_file.read_text())
    assert not set(map(str, WINDOWS)) & set(complete["errors"])
    assert set(WINDOWS) <= set(complete["processed_shots"])
    args.snapshot.mkdir(parents=True, exist_ok=True)
    for sub in ("shots", "labels"):
        (args.snapshot / sub).mkdir(exist_ok=True)

    def snapshot(source: Path, target: Path) -> Path:
        if target.exists():
            assert sha256_of(target) == sha256_of(source), "immutable snapshot changed"
        else:
            shutil.copyfile(source, target)
        return target

    completion = snapshot(completion_file, args.snapshot / "cohort_labels.json")
    snapshot(sums_file, args.snapshot / "labels/SHA256SUMS")
    files = []
    for shot in WINDOWS:
        source = args.export / "shots" / f"{shot}.json"
        target = snapshot(source, args.snapshot / "shots" / source.name)
        physics = json.loads(target.read_text())
        assert physics["shot"] == shot and physics["rule"] == complete["rule"]
        shard = shard_of(labels, sums, shot)
        assert sha256_of(shard) == sums[shard.name], f"{shard.name} changed"
        kept = snapshot(shard, args.snapshot / "labels" / shard.name)
        # The drawn physics record and the population shard are the same labels.
        assert same_labels(physics_rows(physics), population_rows(kept, shot)), (
            f"{shot}: physics record and population shard disagree"
        )
        files.append(
            {
                "shot": shot,
                "original_path": str(source),
                "snapshot_path": str(target),
                "sha256": sha256_of(target),
                "population_shard": str(shard),
                "population_shard_snapshot_path": str(kept),
                "population_shard_sha256": sha256_of(kept),
                "states_equal_population_shard": True,
                "state_seconds": physics["state_seconds"],
            }
        )
    spec = next(s for s in lf.TRACKS if s.key == mt.SAWTOOTH)
    comparisons = []
    for model, version in CATALOG:
        source = (
            paths.root
            / f"suggestions/{model}/{version}/sawtooth_oscillation_suggest_{model}_{version}.csv"
        )
        if not source.is_file():
            comparisons.append({"model": model, "version": version, "missing": True})
            continue
        table = lf.read_rows(source)
        comparisons.append(
            {
                "model": model,
                "version": version,
                "path": str(source),
                "sha256": sha256_of(source),
                "shots": {
                    str(shot): fs.state_intervals(
                        lf.Track(spec, rows=table.get(shot, ())), window
                    )
                    for shot, window in WINDOWS.items()
                },
            }
        )
    manifest = {
        "selected_at_utc": datetime.now(UTC).isoformat(),
        "export_root": str(args.export),
        "export_sums_path": str(sums_file),
        "export_sums_sha256": args.expect_sums_sha256,
        "completion_record": str(completion_file),
        "completion_snapshot_path": str(completion),
        "completion_sha256": sha256_of(completion),
        "completion_requested_count": complete["requested_count"],
        "validation_status": complete["validation_status"],
        "snapshot_source": str(args.snapshot / "shots"),
        "windows_ms": {str(shot): list(w) for shot, w in WINDOWS.items()},
        "files": files,
        "catalog_comparison": comparisons,
        "integration_decision": "controller pins the sawtooth source shipped with the paper",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        f"Pinned {len(files)} physics record(s) from {args.export}; "
        "states equal the population shard; catalog comparison recorded"
    )


if __name__ == "__main__":
    main()
