"""The AE pseudo-mask in a real browser: drawn, clicked, hidden, remembered."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn

from labeler.ae.seg import pseudo_dir, regions
from labeler.ae.seg.pseudo import IGNORE, PseudoMask
from labeler.config import Paths
from labeler.events.ui.app import create_app

from .test_review_browser import NODE, ROSTER, SHELLS, SOURCE, TOKEN, _store

DRIVER = Path(__file__).with_name("review_masks.mjs")

pytestmark = pytest.mark.skipif(
    NODE is None or not SHELLS, reason="needs node and a headless Chromium"
)


def _pseudo(paths: Paths, shot: int) -> None:
    """Two regions: 97-107 kHz over 410-532 ms, and 176-186 kHz over 614-737 ms."""
    mask = np.zeros((257, 900), dtype=np.uint8)
    mask[:82] = IGNORE
    mask[100:111, 200:260] = 1
    mask[180:191, 300:360] = 1
    pseudo_dir(paths).mkdir(parents=True, exist_ok=True)
    PseudoMask(shot, 0.0, 2.048, 0.0, 500 / 512, mask).save(
        regions.pseudo_file(paths, shot)
    )


@pytest.fixture
def served(tmp_path):
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
    for shot in (170815, 170816):
        _store(paths, shot)
    _pseudo(paths, 170815)
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(paths=paths, token=TOKEN),
            host="127.0.0.1",
            port=0,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", event
    server.should_exit = True
    thread.join(10)


def test_a_region_is_rejected_by_a_click_and_the_choice_is_kept(served, tmp_path):
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
    assert len(checks) == 9
    lines = regions.log_path(event).read_text().splitlines()
    saves = [json.loads(line) for line in lines]
    assert [(s["rejected"], s["name"]) for s in saves] == [([2], "Ada"), ([], "Ada")]


@pytest.mark.parametrize(
    "case", ["race", "failed", "away_failed", "conflict", "stale_mask"]
)
def test_mask_save_races_and_failures(served, tmp_path, case):
    base, event = served
    if case == "stale_mask":
        regions.save_decision(event, 170815, [2], pseudo_sha256="old mask")
    result = subprocess.run(
        [
            NODE,
            str(DRIVER),
            base,
            TOKEN,
            str(SHELLS[-1]),
            str(tmp_path / "profile"),
            case,
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    checks = json.loads(result.stdout.splitlines()[-1])
    assert [c for c in checks if not c["ok"]] == []
    log = regions.log_path(event)
    saves = [json.loads(line)["rejected"] for line in log.read_text().splitlines()]
    expected = {
        "race": [[2], [1, 2]],
        "failed": [[2]],
        "away_failed": [[2]],
        "conflict": [[2], [1, 2]],
        "stale_mask": [[2], [1]],
    }
    assert saves == expected[case]
