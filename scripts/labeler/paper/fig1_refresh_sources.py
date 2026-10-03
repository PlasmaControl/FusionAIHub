"""Pin the requested sawtooth revision and record competing catalog sources.

Read-only inputs; writes immutable copies only in round4/fig1 and a small
manifest in this worktree. The completion record is read from the named commit;
its external shot directory supplies the physics records, hashed individually.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from labeler.config import Paths, sha256_of
from labeler.paper import figure_sources as fs
from labeler.paper import label_figure as lf
from labeler.paper import mode_tags as mt

WINDOWS = {
    201978: (1500, 3300),
    201973: (1600, 3350),
    203187: (1700, 3150),
    186636: (1300, 3900),
    191376: (1500, 2900),
    191782: (1800, 3700),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("/scratch/gpfs/nc1514/FusionAIHub-r4-saw"),
    )
    parser.add_argument("--revision", default="ad0ca40f")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "outputs/labeler/paper/fig_interpreter_tokeye/sawtooth_source_manifest.json"
        ),
    )
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.repository), "rev-parse", args.revision], text=True
    ).strip()
    completion_rel = "outputs/labeler/sawtooth/fix2/cohort_labels.json"
    committed = subprocess.check_output(
        ["git", "-C", str(args.repository), "show", f"{revision}:{completion_rel}"]
    )
    complete = json.loads(committed)
    assert not complete["errors"]
    assert set(complete["requested_shots"]) == set(complete["processed_shots"])
    original = Path(complete["source_records"])
    # External records have no git objects. Require the current source worktree
    # and its external completion to agree with the requested committed release.
    head = subprocess.check_output(
        ["git", "-C", str(args.repository), "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == revision, "external shot records require the pinned release HEAD"
    assert json.loads((original.parent / "cohort_labels.json").read_text()) == complete
    paths = Paths.from_env()
    snapshot = paths.root / "round4/fig1/saw_source" / f"ad0ca40f-{revision[:12]}"
    (snapshot / "shots").mkdir(parents=True, exist_ok=True)
    completion = snapshot / "cohort_labels.json"
    completion.write_bytes(committed)
    files = []
    for shot in WINDOWS:
        source = original / f"{shot}.json"
        target = snapshot / "shots" / source.name
        if target.exists():
            assert sha256_of(target) == sha256_of(source), "immutable snapshot changed"
        else:
            shutil.copyfile(source, target)
        physics = json.loads(target.read_text())
        assert physics["rule"] == complete["rule"]
        files.append(
            {
                "shot": shot,
                "original_path": str(source),
                "snapshot_path": str(target),
                "sha256": sha256_of(target),
            }
        )
    spec = next(s for s in lf.TRACKS if s.key == mt.SAWTOOTH)
    comparisons = []
    for model, version in (("ece_sawtooth", "v3"), ("sawtooth_frames", "v2")):
        source = (
            paths.root
            / f"suggestions/{model}/{version}/sawtooth_oscillation_suggest_{model}_{version}.csv"
        )
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
        "source_repository": str(args.repository),
        "source_commit": revision,
        "completion_git_path": completion_rel,
        "original_source": str(original),
        "snapshot_source": str(snapshot / "shots"),
        "completion_record": str(original.parent / "cohort_labels.json"),
        "completion_snapshot_path": str(completion),
        "completion_sha256": sha256_of(completion),
        "completion_requested_count": complete["requested_count"],
        "validation_status": complete["validation_status"],
        "files": files,
        "catalog_comparison": comparisons,
        "integration_decision": "controller pins the sawtooth source shipped with the paper",
    }
    args.manifest.write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        f"Pinned {len(files)} physics records from {revision}; catalog comparison recorded"
    )


if __name__ == "__main__":
    main()
