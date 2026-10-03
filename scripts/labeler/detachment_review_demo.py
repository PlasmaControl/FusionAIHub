"""Build an isolated real-shot review and record reproducible UI evidence.

No fetching, GPUs, production label writes or existing server changes. The
browser is the repository's existing DevTools driver; its own server uses a
free loopback port and is stopped in finally. All outputs stay under --out.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import matplotlib
import pandas as pd
import uvicorn

from labeler.config import Paths, git_dirty, git_sha, sha256_of
from labeler.events.review import build, rows, video
from labeler.events.ui.app import create_app

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
TESTS = [
    f"tests/labeler/test_{name}.py"
    for name in (
        "review_detachment",
        "review_detachment_browser",
        "review_detachment_geometry",
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
    "tests/labeler/test_review_detachment_geometry.py",
]
CHANGED_PYTHON = NEW_PYTHON + [
    "src/labeler/events/review/panel_rows.py",
    "tests/labeler/test_review_panel_rows.py",
    "src/labeler/events/interval_tables.py",
    "src/labeler/events/panels/__init__.py",
    "src/labeler/events/review/build.py",
    "src/labeler/events/review/rows.py",
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


def paper_export(screenshot, out, shot):
    """Annotate a cropped live-page image with vector, column-readable text."""
    layout_path = Path(str(screenshot) + ".paper.json")
    crop_path = Path(str(screenshot) + ".paper.png")
    layout = json.loads(layout_path.read_text())
    with Image.open(crop_path) as image:
        pixels = image.copy()
    # 16 px page text at this width is >=7 pt at the final 3.25 inch width.
    width = 3.25
    page_height = width * pixels.height / pixels.width
    fig = plt.figure(figsize=(width, page_height + 0.9), facecolor="white")
    ax = fig.add_axes(
        [0, 0.35 / (page_height + 0.9), 1, page_height / (page_height + 0.9)]
    )
    ax.imshow(pixels)
    ax.set_axis_off()
    fig.text(
        0.02,
        1 - 0.12 / (page_height + 0.9),
        f"Shot {shot} · TangTV lower-divertor review",
        fontsize=8,
        va="top",
    )
    fig.text(0.02, 1 - 0.30 / (page_height + 0.9), layout["view"], fontsize=8, va="top")
    fig.text(
        0.02,
        1 - 0.46 / (page_height + 0.9),
        f"Corpus image {layout['frame_ms']:.1f} ms · 50 Hz linear blend",
        fontsize=7.5,
        va="top",
    )
    for letter, key in (
        ("A", "camera"),
        ("B", "timestamp"),
        ("C", "cursor"),
        ("D", "labels"),
    ):
        box = layout[key]
        x = max(12, min(pixels.width - 12, box["x"] + 12))
        y = max(12, min(pixels.height - 12, box["y"] + 12))
        if letter == "B":
            x = box["x"] - 14
        elif letter == "C":
            y += 28
        elif letter == "D":
            x = box["x"] + box["width"] - 20
        ax.text(
            x,
            y,
            letter,
            fontsize=8,
            weight="bold",
            va="center",
            ha="center",
            bbox={"boxstyle": "circle,pad=0.2", "fc": "white", "ec": "#0072b2"},
        )
    fig.text(
        0.02,
        0.20 / (page_height + 0.9),
        "A Camera · B Shared time · C Cursor · D State controls",
        fontsize=7,
        va="center",
    )
    fig.text(
        0.02,
        0.06 / (page_height + 0.9),
        "Cropped live review page; annotations added for readability.",
        fontsize=7,
        va="center",
    )
    base = out / f"detachment_review_{shot}_paper"
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return {
        "pdf": str(base.with_suffix(".pdf")),
        "png": str(base.with_suffix(".png")),
        "dpi": 150,
        "width_inches": width,
        "minimum_annotation_font_pt": 7,
        "layout": str(layout_path),
        "page_crop": str(crop_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=190102)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indicator-root", type=Path)
    parser.add_argument("--capture-time-ms", type=float)
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
    if match.empty or match.split.iloc[0] == "test":
        raise ValueError("demo must use a train/val cohort shot")
    original = Paths.from_env()
    os.environ["LABELER_DETACHMENT_INDICATORS"] = str(
        args.indicator_root or original.root / "round4/detach/bins"
    )
    os.environ.setdefault(
        "LABELER_DETACHMENT_GEOMETRY_ROOT", str(original.root / "round4/detach/cache")
    )
    paths = Paths(root=out, corpus=original.corpus, label_tables=out / "tables")
    event = paths.label_tables / "detachment"
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
    store = build.build("detachment", args.shot, paths, force=True)
    manifest = video.meta(store)
    record = {
        **provenance(),
        "shot": args.shot,
        "split": str(match.split.iloc[0]),
        "cohort": str(cohort_path),
        "corpus": str(paths.corpus_file(args.shot)),
        "indicator_root": os.environ["LABELER_DETACHMENT_INDICATORS"],
        "neutral_reviewer": "Reviewer",
        "source_shapes": source_shapes,
        "store": str(store),
        "store_bytes": store.stat().st_size,
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
    evidence = out / "demo.json"
    evidence.write_text(json.dumps(record, indent=2) + "\n")
    node = shutil.which("node")
    shells = sorted(
        Path.home().glob(
            ".cache/ms-playwright/chromium_headless_shell-*/chrome-linux/headless_shell"
        )
    )
    if not node or not shells:
        raise RuntimeError("no existing headless browser; render a matplotlib mock")
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
        record["browser_stderr_tail"] = result.stderr[-2000:]
        evidence.write_text(json.dumps(record, indent=2) + "\n")
        if (
            result.returncode
            or not record["browser_checks"]
            or any(not c["ok"] for c in record["browser_checks"])
        ):
            raise RuntimeError(f"browser check failed: see {evidence}")
        record["paper_export"] = paper_export(png, out, args.shot)
    finally:
        server.should_exit = True
        thread.join(10)
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
