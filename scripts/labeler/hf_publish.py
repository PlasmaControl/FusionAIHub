#!/usr/bin/env python
"""Build, check, upload and pull the labels-only Hugging Face release.

    python scripts/labeler/hf_publish.py build  [--out DIR] [--release v1] [--silver-grid DIR]
                                                [--gold-grid DIR] [--exclude DATASET ...]
                                                [--include DATASET ...] [--allow-unshipped]
    python scripts/labeler/hf_publish.py verify [--dir DIR]
    python scripts/labeler/hf_publish.py upload [--dir DIR] [--repo nc1/tokamak-si] --yes
    python scripts/labeler/hf_publish.py pull   [--repo nc1/tokamak-si] [--revision v1] [--into DIR]

`build` writes release `--release` (default v1; `labeler.events.hf_export`) from the
current label tables into `$LABELER_ROOT/hf/release` by default, exporting the silver
and gold 10 ms grids into `<out>.work/` unless given, and writes the build report to
`<out>.build.json`. `--out` must be new, empty or a previous release (never the label
tables, the labeler root or anything holding them); the release is built in a hidden
sibling and swapped in once it verifies. It fails if the built directory names a blind
cohort shot, an absolute or home path, a login, a reviewer or a roster placeholder,
or puts a gold or holdout shot in another split, and when the grid or a review pointer
uses a table the release does not hold (no `.meta.json`, not the interval schema, or
`--exclude`d) unless `--allow-unshipped`, which flags those tables in the card and
`datasets.parquet`. The withheld tables (`hf_export.WITHHELD`: the ELM challenge
answer key) stay out unless `--include` names them; the card says where to get them.
`upload` re-verifies, then sends the directory to a PRIVATE dataset repo (created
private when missing; a public repo is refused) and tags it with the release; it
refuses a release name the repo already has as a tag, and needs `--yes`, plus
`--allow-pending` while `signals.yaml` has `todo` markers and `--allow-unshipped`
while tables are flagged. `pull` downloads a release's `tables/` into
`$LABELER_ROOT/hf/pull/<commit>` (an earlier download there is reused only when it is
exactly that release), checks every file against `release.json`, and copies them
under the label tables (suggestions under `$LABELER_ROOT`), or under `--into DIR`
(`DIR/events`, `DIR/suggestions`), setting each review pointer. It never replaces a
file or a pointer that differs, and refuses label tables whose catalog lists the
blind cohort (the full tables a release is cut from). The Hub calls go through
`huggingface_hub.HfApi` (the `hf` CLI crashes after printing).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from labeler.config import Paths
from labeler.events import hf_export


def _paths() -> Paths:
    return Paths.from_env()


def build(args) -> int:
    paths = _paths()
    out = args.out or paths.root / "hf" / "release"
    try:
        report = hf_export.build(
            paths,
            out,
            silver_grid=args.silver_grid,
            gold_grid=args.gold_grid,
            work=args.work,
            exclude=args.exclude,
            include=args.include,
            allow_unshipped=args.allow_unshipped,
            release=args.release,
        )
    except hf_export.ReleaseError as error:
        print(f"build failed: {error}", file=sys.stderr)
        return 1
    Path(f"{out}.build.json").write_text(
        json.dumps(report, indent=1, default=str) + "\n"
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "release": report["release"],
                "bytes": report["bytes"],
                "files": report["n_files"],
                "labels": report["labels"],
                "grids": {
                    k: {kk: vv for kk, vv in v.items() if kk != "events"}
                    for k, v in report["grids"].items()
                },
                "withheld_blind_rows": report["withheld_blind_rows"],
                "holdout": report["holdout"],
                "shots": report["shots"],
                "withheld": sorted(report["withheld"]),
                "unshipped": report["unshipped_groups"],
                "dropped_attrs": report["dropped_attrs"],
                "skipped": len(report["skipped"]),
                "mirror": report["mirror"],
                "pending": sorted(report["pending"]),
            },
            indent=1,
        )
    )
    return 0


def verify(args) -> int:
    paths = _paths()
    folder = args.dir or paths.root / "hf" / "release"
    problems = hf_export.verify(folder, hf_export.context(paths))
    for problem in problems:
        print(problem)
    print(f"{folder}: {'clean' if not problems else f'{len(problems)} problems'}")
    return 1 if problems else 0


def upload(args) -> int:
    if not args.yes:
        print(
            "upload sends the release to the Hub; pass --yes to do it", file=sys.stderr
        )
        return 2
    paths = _paths()
    folder = args.dir or paths.root / "hf" / "release"
    try:
        result = hf_export.upload(
            folder,
            hf_export.context(paths),
            repo_id=args.repo,
            allow_pending=args.allow_pending,
            allow_unshipped=args.allow_unshipped,
            message=args.message,
        )
    except hf_export.ReleaseError as error:
        print(f"upload refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    return 0


def pull(args) -> int:
    paths = _paths()
    dest = args.work or paths.root / "hf" / "pull"
    if args.into is not None:
        paths = dataclasses.replace(
            paths, root=args.into, label_tables=args.into / "events"
        )
    try:
        folder = hf_export.download(args.repo, dest, revision=args.revision)
        result = hf_export.restore(folder, paths)
    except hf_export.ReleaseError as error:
        print(f"pull refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"from": str(folder), **result}, indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build", help="write and verify a release directory")
    command.add_argument("--out", type=Path)
    command.add_argument(
        "--release",
        default=hf_export.RELEASE,
        help="the release name and tag (v1, v2, ...); upload refuses one the repo has",
    )
    command.add_argument(
        "--silver-grid", type=Path, help="an exported silver grid to read"
    )
    command.add_argument("--gold-grid", type=Path, help="an exported gold grid to read")
    command.add_argument("--work", type=Path, help="where fresh grids go (<out>.work)")
    command.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="DATASET",
        help="leave this dataset id out (a table not cleared for release); "
        "an id that is no table of the labeler is an error",
    )
    command.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="DATASET",
        choices=sorted(hf_export.WITHHELD),
        help="ship this withheld dataset id (only once its owners allow it)",
    )
    command.add_argument(
        "--allow-unshipped",
        action="store_true",
        help="build although the grid or a pointer uses tables the release lacks",
    )
    command.set_defaults(run=build)
    command = commands.add_parser("verify", help="check a built release")
    command.add_argument("--dir", type=Path)
    command.set_defaults(run=verify)
    command = commands.add_parser(
        "upload", help="send a built release to a private repo"
    )
    command.add_argument("--dir", type=Path)
    command.add_argument("--repo", default=hf_export.REPO_ID)
    command.add_argument("--message")
    command.add_argument(
        "--allow-pending",
        action="store_true",
        help="upload although signals.yaml has todo markers",
    )
    command.add_argument(
        "--allow-unshipped",
        action="store_true",
        help="upload although tables the grid uses are not in the release",
    )
    command.add_argument("--yes", action="store_true", help="really upload")
    command.set_defaults(run=upload)
    command = commands.add_parser(
        "pull", help="restore the label tables from a release"
    )
    command.add_argument("--repo", default=hf_export.REPO_ID)
    command.add_argument("--revision", help="a tag or commit (default: main)")
    command.add_argument(
        "--work", type=Path, help="where downloads go (<work>/<commit>)"
    )
    command.add_argument(
        "--into",
        type=Path,
        help="restore into DIR/events and DIR/suggestions instead",
    )
    command.set_defaults(run=pull)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
