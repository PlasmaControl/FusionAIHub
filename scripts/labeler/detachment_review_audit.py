"""Read-only audit of gas corrections, Source coverage and resume fingerprints."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from detachment_review_roster import build_store

from labeler.config import Paths, git_sha, sha256_of
from labeler.events.review import build, detachment, rows

REPO = Path(__file__).resolve().parents[2]
GAS_SHOTS = [190094, 190102, 190212, 190218, 190450, *range(190109, 190117), 190182]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument(
        "--refresh-recipes",
        action="store_true",
        help="refresh changed interpretation only; never rebuild pixels",
    )
    args = parser.parse_args()
    original = Paths.from_env()
    producer = original.root / "round4/detach"
    for name, path in {
        "INDICATORS": producer / "bins",
        "LABELS": producer / "labels_bins.csv.gz",
        "GEOMETRY_ROOT": producer / "cache",
        "CACHE_ROOT": producer / "cache",
    }.items():
        os.environ[f"LABELER_DETACHMENT_{name}"] = str(path)
    paths = Paths(
        root=args.out, corpus=original.corpus, label_tables=args.out / "tables"
    )
    build_record = json.loads((args.out / "roster_build.json").read_text())
    queue = set(build_record["queue_shots"])
    resumed, refreshed = [], []
    for entry in build_record["stores"]:
        path = paths.spectrogram_file("detachment", entry["shot"])
        with h5py.File(path, "r") as source:
            params = json.loads(source.attrs["params"])
        desired = detachment.context_sources(entry["shot"], paths)
        changed = {
            key for key in desired if params["context_sources"].get(key) != desired[key]
        }
        if (
            args.refresh_recipes
            and changed
            and all(k.startswith("recipe_") for k in changed)
        ):
            result = build_store(entry, paths, producer / "bins", resume=True)
            assert result["action"] == "refreshed recipe"
            refreshed.append(entry["shot"])
            with h5py.File(path, "r") as source:
                params = json.loads(source.attrs["params"])
        current = (
            build.current(path, "detachment") and params["context_sources"] == desired
        )
        resumed.append(
            {
                **entry,
                "current": current,
                "geometry_shelf_gate_samples": params["detachment_geometry"][
                    "shelf_gate_samples"
                ],
            }
        )
    gas = []
    for shot in GAS_SHOTS:
        store_path = paths.spectrogram_file("detachment", shot)
        entry = {"shot": shot, "store_available": store_path.is_file()}
        if not store_path.is_file():
            gas.append(entry)
            continue
        meta = rows.meta(store_path)
        row = next(r for r in meta["rows"] if r["title"] == "Gas flow")
        channel = next(
            i for i, name in enumerate(row["legend"]) if name.startswith("LOB2 ")
        )
        stored = meta["params"]["panel_metadata"][row["name"]]["preplasma_baseline"][
            channel
        ]
        with h5py.File(paths.corpus_file(shot), "r") as corpus:
            group = corpus["gas_flow"]
            stop = int(np.searchsorted(group["xdata"][:], 0))
            native = group["ydata"][6, :stop]
            baseline = float(native[np.isfinite(native)].astype(float).mean())
        with h5py.File(store_path, "r") as store:
            values = store[f"rows/{row['name']}/1"][:, channel]
        entry.update(
            preplasma_samples=stop,
            native_baseline=baseline,
            stored_baseline=stored,
            baseline_matches=bool(np.isclose(stored, baseline, rtol=0, atol=1e-9)),
            corrected_min=float(np.nanmin(values)),
            corrected_max=float(np.nanmax(values)),
        )
        gas.append(entry)
    suggestions = pd.read_csv(args.out / "tables/detachment/review/suggestions.csv")
    clipped = []
    for shot in (202205, 202206):
        subset = suggestions[suggestions.shot == shot]
        window = rows.meta(paths.spectrogram_file("detachment", shot))["t_range"]
        clipped.append(
            {
                "shot": shot,
                "window_ms": window,
                "delivered_rows": len(subset),
                "within_window": bool(
                    ((subset.t_start >= window[0]) & (subset.t_end <= window[1])).all()
                ),
                "last_row": json.loads(subset.tail(1).to_json(orient="records")),
            }
        )
    record = {
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(full=True),
        "script_sha256": sha256_of(Path(__file__)),
        "build_record": str(args.out / "roster_build.json"),
        "resume_stores": len(resumed),
        "resume_would_rebuild": sum(not r["current"] for r in resumed),
        "recipes_refreshed": refreshed,
        "resume_stale_shots": [r["shot"] for r in resumed if not r["current"]],
        "camera_with_any_shelf_sample": sum(
            r["geometry_shelf_gate_samples"] > 0 for r in resumed if r["shot"] in queue
        ),
        "gas": gas,
        "clipped_suggestions": clipped,
        "queue_shots": len(queue),
        "suggestion_rows": len(suggestions),
        "suggestion_shots": int(suggestions.shot.nunique()),
        "producer_snapshot_consistent": not build_record["suggestions"][
            "inconsistencies"
        ],
        "blind_overlap": sorted(
            queue
            & set(
                pd.read_csv(REPO / "data/events/catalog/cohort.csv")
                .query("split == 'test'")
                .shot
            )
        ),
    }
    args.record.write_text(json.dumps(record, indent=2) + "\n")
    assert not record["blind_overlap"]
    assert all(r.get("baseline_matches", True) for r in gas)
    assert all(r["within_window"] for r in clipped)
    print(
        json.dumps(
            {k: v for k, v in record.items() if k not in {"gas", "clipped_suggestions"}}
        )
    )


if __name__ == "__main__":
    main()
