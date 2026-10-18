#!/usr/bin/env python
"""Grade submission folders with the DIII-D ELM detection challenge's own merit function.

Their code is never copied: this imports `merit_function` from a pinned clone of
git@github.com:DIII-D/elm_detection_challenge.git that lives outside the repository, and
writes only here (`--out`). Run it with the interpreter of a venv that has numpy, scipy,
pandas and `uncertainties` and nothing of ours (`labeler` is not imported)::

    $LABELER_ROOT/envs/elm_challenge/bin/python scripts/labeler/elm_challenge_merit.py \\
        --clone $LABELER_ROOT/benchmarks/elm/challenge/elm_detection_challenge \\
        --key $LABELER_LABEL_TABLES/edge_localized_mode/raw/elm_challenge_key_b71f991.json \\
        --folder name=DIR [--folder name=DIR ...] --out scores.json

The key given must be byte-identical to the clone's own, so what is scored is the key at
the clone's commit. A folder with no JSON would be replaced by the challenge's samples
(`find_submission_files`' backup folder), so an empty folder is an error here. Scores are
the merit function's: lower is better, unnormalised, and a case with no file costs
`missing_case_penalty` (20,000).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clone", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--folder", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--shots", type=int, nargs="+", help="grade only these shots")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    clone = args.clone.resolve()
    own_key = clone / "src/elm_detection_challenge/answer_key/ELM_database.json"
    if sha256(args.key) != sha256(own_key):
        raise SystemExit(f"{args.key} is not the clone's own key {own_key}")
    head = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(clone), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"the clone has uncommitted changes:\n{dirty}")

    sys.path.insert(0, str(clone / "src"))
    import elm_detection_challenge
    from elm_detection_challenge.merit_function import merit_function

    elm_detection_challenge._config["verbose"] = False
    result = {"clone_commit": head, "key_sha256": sha256(own_key), "folders": {}}
    for item in args.folder:
        name, _, folder = item.partition("=")
        folder = Path(folder)
        if not any(folder.glob("*.json")):
            raise SystemExit(
                f"{folder}: no *.json (the samples would be graded instead)"
            )
        scores = merit_function(
            answer_key_filename=str(args.key),
            submission_folder=str(folder),
            devices_to_grade=["DIII-D"],
            only_these_shots=args.shots,
        )
        result["folders"][name] = scores
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    for name, scores in result["folders"].items():
        total = scores["total"]
        print(
            name,
            {
                k: round(float(total[k]), 1)
                for k in (
                    "total",
                    "timing_mismatch",
                    "false_positive",
                    "overcounting",
                    "false_negative",
                    "real_events",
                    "detected_events",
                )
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
