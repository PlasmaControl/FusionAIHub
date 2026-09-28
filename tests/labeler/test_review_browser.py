"""The review page in a real browser: draw, save, reload, restore, move on."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn

from labeler.config import Paths
from labeler.events.review import labels
from labeler.events.review import rows as review_rows
from labeler.events.review.rows import Grid, ImageRow, TraceRow
from labeler.events.ui.app import create_app

NODE = shutil.which("node")
SHELLS = sorted(
    Path.home().glob(
        ".cache/ms-playwright/chromium_headless_shell-*/chrome-linux/headless_shell"
    )
)
DRIVER = Path(__file__).with_name("review_browser.mjs")
TOKEN = "0123456789abcdef" * 2
ROSTER = (
    "shot,tier,holdout,reviewers,verified_on,notes\n"
    "170815,gold,false,,,\n"
    "170816,unverified,false,,,\n"
    "170817,unverified,false,,,\n"
)
SOURCE = (
    "shot,category,t_start,t_end,confidence\n"
    "170815,0,0,100,\n"
    "170815,1,100,300,\n"
    "170815,0,300,2000,\n"
)

pytestmark = pytest.mark.skipif(
    NODE is None or not SHELLS, reason="needs node and a headless Chromium"
)


def _store(paths: Paths, shot: int) -> None:
    """Two rows over 0-4000 ms at 0.5 ms: a chirping image and a trace."""
    t = np.arange(8000) * 0.5
    bins = np.arange(64)[:, None]
    chirp = np.exp(-((bins - 32 - 20 * np.sin(t / 300)) ** 2) / 6)
    image = ImageRow(
        "R0xV1",
        "R0 × V1",
        (40 + 200 * chirp).astype("uint8"),
        y0=0.0,
        dy=4.0,
        y_units="kHz",
        z_lo=-3.0,
        z_hi=27.0,
        z_units="dB",
    )
    wave = np.sin(t / 120)[None]
    trace = TraceRow("p1", "Density", np.stack([wave, wave]), hlines=[0.0])
    review_rows.write(
        paths.spectrogram_file("alfven_eigenmode", shot),
        Grid(0.0, 0.5, 8000),
        [image, trace],
        event="alfven_eigenmode",
    )


@pytest.fixture
def served(tmp_path):
    """The review server on a free loopback port, over three built shots.

    170816 is saved before the page opens, so the next unreviewed shot after
    170815 is 170817 while the next shot in the queue is 170816.
    """
    paths = Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tmp_path / "tables",
        raw_cache=tmp_path / "raw",
    )
    event = paths.label_tables / "alfven_eigenmode"
    (event / "format").mkdir(parents=True)
    (event / "shots.csv").write_text(ROSTER)
    (event / "format/alfven_eigenmode_format_2026_v1.csv").write_text(SOURCE)
    for shot in (170815, 170816, 170817):
        _store(paths, shot)
    labels.save(
        event, 170816, labels.normalise((0, 2000), [(400, 600, 1)]), source=None
    )
    app = create_app(paths=paths, token=TOKEN)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", event
    server.should_exit = True
    thread.join(10)


def test_a_drawn_label_is_saved_found_again_and_left_behind(served, tmp_path):
    base, event = served
    result = subprocess.run(
        [NODE, str(DRIVER), base, TOKEN, str(SHELLS[-1]), str(tmp_path / "profile")],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    checks = json.loads(result.stdout.splitlines()[-1])
    assert [c for c in checks if not c["ok"]] == []
    assert len(checks) == 20
    saved = labels.read_saved(event)
    assert sorted(saved) == [170815, 170816]
    kept, (a, b, c) = saved[170815].intervals
    assert kept == (100, 300, 1) and c == 1 and abs(a - 500) <= 2 and abs(b - 800) <= 2
    history = labels.read_history(event)
    assert [entry["shot"] for entry in history] == [170816] + [170815] * 3
    assert [entry["name"] for entry in history] == [None] + ["Ada Lovelace"] * 3
    first, second, restored = history[1:]
    assert len(second["intervals"]) == 3, "the second save added a span"
    assert restored["intervals"] == first["intervals"], "the restored version was saved"
