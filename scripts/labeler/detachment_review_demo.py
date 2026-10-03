"""Build an isolated real-shot review and record reproducible UI evidence.

No fetching, GPUs, production label writes or existing server changes. The
browser is the repository's existing DevTools driver; its own server uses a
free loopback port and is stopped in finally. All outputs stay under --out.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import h5py
import pandas as pd
import uvicorn
from detachment_review_roster import snapshot_suggestions

from labeler.config import Paths, git_dirty, git_sha, sha256_of
from labeler.events.review import build, detachment, rows, video
from labeler.events.ui.app import create_app

REPO = Path(__file__).resolve().parents[2]
TESTS = [
    f"tests/labeler/test_{name}.py"
    for name in (
        "review_detachment",
        "review_detachment_browser",
        "review_detachment_geometry",
        "review_detachment_producer",
        "detachment_diagnostics",
        "detachment_queue",
        "review_agreement",
        "review_build",
        "review_rows",
        "review_panel_rows",
        "review_labels",
        "interval_tables",
        "events_ui",
        "review_versions",
        "review_page",
        "review_browser",
        "review_browser_api1",
        "review_browser_race",
    )
]
NEW_PYTHON = [
    "src/labeler/events/review/video.py",
    "src/labeler/events/review/detachment.py",
    "src/labeler/events/panels/detachment.py",
    "tests/labeler/test_review_detachment.py",
    "tests/labeler/test_review_detachment_browser.py",
    "scripts/labeler/detachment_review_demo.py",
    "scripts/labeler/detachment_review_roster.py",
    "src/labeler/events/review/geometry.py",
    "src/labeler/events/review/producer.py",
    "src/labeler/events/review/recipe.py",
    "tests/labeler/test_review_detachment_producer.py",
    "tests/labeler/test_detachment_diagnostics.py",
    "tests/labeler/test_detachment_queue.py",
    "scripts/labeler/detachment_review_export.py",
    "scripts/labeler/detachment_review_audit.py",
    "tests/labeler/test_review_detachment_geometry.py",
]
CHANGED_PYTHON = NEW_PYTHON + [
    "tests/labeler/test_review_page.py",
    "src/labeler/events/review/panel_rows.py",
    "tests/labeler/test_review_panel_rows.py",
    "src/labeler/events/interval_tables.py",
    "src/labeler/events/panels/__init__.py",
    "src/labeler/events/review/build.py",
    "src/labeler/events/review/rows.py",
    "src/labeler/events/review/labels.py",
    "src/labeler/events/ui/app.py",
    "src/labeler/events/review/agreement.py",
    "tests/labeler/test_review_build.py",
    "tests/labeler/test_events_ui.py",
    "tests/labeler/test_review_versions.py",
    "tests/labeler/test_review_agreement.py",
]


def provenance():
    return {
        "git_sha": git_sha(full=True),
        "git_dirty": git_dirty(),
        "source_sha256": {
            name: sha256_of(REPO / name)
            for name in CHANGED_PYTHON
            + [
                "src/labeler/events/ui/static/app.js",
                "src/labeler/events/ui/static/style.css",
                "src/labeler/events/ui/static/index.html",
                "tests/labeler/review_browser.mjs",
                "docs/labeler/detachment.md",
                "docs/labeler/detachment_review.md",
            ]
        },
    }


def verify(out):
    """Run only covering files and persist commands/results as JSON evidence."""
    runner = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh")
    pixi = [
        "pixi",
        "run",
        "--frozen",
        "--no-install",
        "--manifest-path",
        "/scratch/gpfs/nc1514/FusionAIHub/pyproject.toml",
        "-e",
        "labelmaker",
    ]
    record = {**provenance(), "checks": []}
    with tempfile.TemporaryDirectory(
        prefix="review-verification-", dir=os.environ["TMPDIR"]
    ) as temporary:
        junit = Path(temporary) / "pytest.xml"
        commands = [
            [
                "bash",
                str(runner),
                str(REPO),
                *TESTS,
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={junit}",
            ],
            [*pixi, "ruff", "check", *CHANGED_PYTHON],
            [*pixi, "ruff", "format", "--check", *NEW_PYTHON],
            ["node", "--check", "src/labeler/events/ui/static/app.js"],
            ["node", "--check", "tests/labeler/review_browser.mjs"],
        ]
        for command in commands:
            result = subprocess.run(
                command,
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            record["checks"].append(
                {
                    "command": command,
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-5000:],
                    "stderr_tail": result.stderr[-3000:],
                }
            )
        if junit.is_file():
            record["pytest"] = [
                dict(suite.attrib) for suite in ET.parse(junit).getroot()
            ]
    (out / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    if any(check["returncode"] for check in record["checks"]):
        raise RuntimeError(f"verification failed: see {out / 'verification.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=200977)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indicator-root", type=Path)
    parser.add_argument("--producer-labels", type=Path)
    parser.add_argument("--capture-time-ms", type=float)
    parser.add_argument(
        "--frozen-from",
        type=Path,
        help="copy and serve an existing demo snapshot; prohibit all store builds",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run covering tests/lint and write verification.json",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.verify:
        verify(out)
    cohort_path = REPO / "data/events/catalog/cohort.csv"
    cohort = pd.read_csv(cohort_path)
    match = cohort[cohort.shot == args.shot]
    if not match.empty and match.split.iloc[0] == "test":
        raise ValueError("demo cannot use a blind test cohort shot")
    original = Paths.from_env()
    if match.empty:
        audited = original.root / "round4/detach-ui/tables/detachment/shots.csv"
        if not audited.is_file() or args.shot not in set(pd.read_csv(audited).shot):
            raise ValueError("external demo must belong to the nonblind delivery queue")
    os.environ["LABELER_DETACHMENT_INDICATORS"] = str(
        args.indicator_root or original.root / "round4/detach/bins"
    )
    os.environ.setdefault(
        "LABELER_DETACHMENT_GEOMETRY_ROOT", str(original.root / "round4/detach/cache")
    )
    os.environ["LABELER_DETACHMENT_LABELS"] = str(
        args.producer_labels or original.root / "round4/detach/labels_bins.csv.gz"
    )
    os.environ.setdefault(
        "LABELER_DETACHMENT_CACHE_ROOT", str(original.root / "round4/detach/cache")
    )
    paths = Paths(root=out, corpus=original.corpus, label_tables=out / "tables")
    split = "producer_external" if match.empty else str(match.split.iloc[0])
    frozen_store = (
        args.frozen_from / "spectrograms/detachment" / f"{args.shot}.h5"
        if args.frozen_from
        else None
    )
    window = (
        rows.meta(frozen_store)["t_range"]
        if frozen_store
        else detachment.plasma_window(args.shot, original)
    )
    event = paths.label_tables / "detachment"
    if args.frozen_from:
        if args.frozen_from.resolve() == out:
            raise ValueError("frozen demonstration must use a separate --out")
        shutil.copytree(
            args.frozen_from / "tables/detachment", event, dirs_exist_ok=True
        )
        pointer = event / "review/source.json"
        frozen_source = json.loads(pointer.read_text())
        frozen_source["table"] = str(event / "review/suggestions.csv")
        pointer.write_text(json.dumps(frozen_source, indent=2) + "\n")
    event.mkdir(parents=True, exist_ok=True)
    roster = event / "shots.csv"
    if roster.is_file():
        if args.shot not in set(pd.read_csv(roster).shot):
            raise ValueError("demo shot is not in the isolated scanned roster")
    else:
        roster.write_text(
            "shot,tier,holdout,reviewers,verified_on,notes\n"
            f"{args.shot},unverified,false,,,isolated UI demonstration\n"
        )
    overlay = pd.read_csv(roster, dtype=str, keep_default_na=False)
    reserved = set(cohort.loc[cohort.split.eq("test"), "shot"].astype(int))
    reserved.update(overlay.loc[overlay.holdout.eq("true"), "shot"].astype(int))
    if args.shot in reserved:
        raise ValueError("demo shot is a reserved holdout")
    producer_snapshot = (
        {
            "mode": "frozen snapshot; no producer reread or store rebuild",
            "source": str(args.frozen_from),
            "source_sha256": sha256_of(
                args.frozen_from / "tables/detachment/review/source.json"
            ),
        }
        if args.frozen_from
        else snapshot_suggestions(
            os.environ["LABELER_DETACHMENT_LABELS"],
            overlay.shot.astype(int),
            event,
            reserved_shots=reserved,
            indicator_root=os.environ["LABELER_DETACHMENT_INDICATORS"],
            windows={args.shot: list(window) if window else None},
        )
    )
    source_shapes = {}
    with h5py.File(paths.corpus_file(args.shot), "r") as source:
        for name in (*video.CAMERAS, "filterscopes", "langmuir", "gas_flow", "co2"):
            if name in source:
                source_shapes[name] = {
                    key: list(source[name][key].shape)
                    for key in ("xdata", "ydata")
                    if key in source[name]
                }
    started = time.monotonic()
    if frozen_store:
        store = paths.spectrogram_file("detachment", args.shot)
        store.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(frozen_store, store)
    else:
        store = build.build("detachment", args.shot, paths, force=True)
    manifest = video.meta(store)
    record = {
        **provenance(),
        "shot": args.shot,
        "split": split,
        "cohort": str(cohort_path),
        "corpus": str(paths.corpus_file(args.shot)),
        "indicator_root": os.environ["LABELER_DETACHMENT_INDICATORS"],
        "producer_snapshot": producer_snapshot,
        "neutral_reviewer": "Reviewer",
        "source_shapes": source_shapes,
        "store": str(store),
        "store_bytes": store.stat().st_size,
        "store_sha256": sha256_of(store),
        "frozen_store_sha256": sha256_of(frozen_store) if frozen_store else None,
        "store_rebuilt": not bool(frozen_store),
        "build_seconds": time.monotonic() - started,
        "rows": rows.meta(store),
        "video": manifest,
        "frame_counts": {
            c["name"]: {str(ch["channel"]): len(ch["times_ms"]) for ch in c["channels"]}
            for c in manifest["cameras"]
        },
        "network_fetch": False,
        "label_writes": False,
        "browser_checks": [],
    }
    evidence = out / f"demo_{args.shot}.json"
    evidence.write_text(json.dumps(record, indent=2) + "\n")
    node = shutil.which("node")
    shells = sorted(
        Path.home().glob(
            ".cache/ms-playwright/chromium_headless_shell-*/chrome-linux/headless_shell"
        )
    )
    if not node or not shells:
        raise RuntimeError("no existing Playwright Chromium; browser evidence required")
    token = "detachment-local-demonstration"
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(paths, token),
            host="127.0.0.1",
            port=0,
            log_level="warning",
            access_log=False,
        )
    )
    frozen_guards = contextlib.ExitStack()
    if frozen_store:
        current = build.current
        frozen_guards.enter_context(
            patch.object(
                build,
                "current",
                side_effect=lambda path, event: (
                    path.is_file() if event == "detachment" else current(path, event)
                ),
            )
        )
        frozen_guards.enter_context(
            patch.object(
                build,
                "build",
                side_effect=RuntimeError("Frozen demo prohibits rebuilding"),
            )
        )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while not server.started:
            if time.monotonic() > deadline or not thread.is_alive():
                raise RuntimeError("demo server did not start")
            time.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]
        png = out / f"detachment_review_{args.shot}.png"
        with tempfile.TemporaryDirectory(
            prefix="demo-browser-", dir=os.environ["TMPDIR"]
        ) as profile_root:
            result = subprocess.run(
                [
                    node,
                    str(REPO / "tests/labeler/review_browser.mjs"),
                    f"http://127.0.0.1:{port}",
                    token,
                    str(shells[-1]),
                    str(Path(profile_root) / "profile"),
                    "detachment-demo",
                    str(png),
                ],
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
                env={
                    **os.environ,
                    "DETACHMENT_DEMO_SHOT": str(args.shot),
                    "DETACHMENT_CAPTURE_TIME_MS": str(args.capture_time_ms or ""),
                },
            )
        record["browser_returncode"] = result.returncode
        record["browser_checks"] = (
            json.loads(result.stdout.splitlines()[-1]) if result.stdout.strip() else []
        )
        record["screenshot"] = str(png) if png.is_file() else None
        record["screenshot_viewport"] = [1366, 768]
        record["browser_stderr_tail"] = result.stderr[-2000:]
        evidence.write_text(json.dumps(record, indent=2) + "\n")
        if (
            result.returncode
            or len(record["browser_checks"])
            != (34 + 2 * any(len(c["channels"]) > 1 for c in manifest["cameras"]))
            or not {
                "camera, full diagnostic, time axis and annotations fit at 1366x768",
                "camera, full diagnostic, time axis and annotations fit at 1400x900",
                "no script error",
            }
            <= {c["name"] for c in record["browser_checks"]}
            or any(not c["ok"] for c in record["browser_checks"])
        ):
            raise RuntimeError(f"browser check failed: see {evidence}")
    finally:
        server.should_exit = True
        thread.join(10)
        frozen_guards.close()
        record["server_stopped"] = not thread.is_alive()
        evidence.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "evidence": str(evidence),
                "shot": args.shot,
                "frame_counts": record["frame_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
