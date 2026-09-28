"""History and edits cannot cross a pending shot or event navigation."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time

import pytest
import uvicorn

from labeler.config import Paths
from labeler.events.review import labels
from labeler.events.ui.app import create_app

from .test_review_browser import DRIVER, NODE, ROSTER, SHELLS, SOURCE, TOKEN, _store

pytestmark = pytest.mark.skipif(
    NODE is None or not SHELLS, reason="needs node and a headless Chromium"
)


@pytest.fixture
def served_events(tmp_path):
    """Two events with different saved labels for the same shot, all local."""
    paths = Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tmp_path / "tables",
        raw_cache=tmp_path / "raw",
    )
    for shot in (170815, 170816, 170817):
        _store(paths, shot)
    events = []
    for name, span in (
        ("alfven_eigenmode", (100, 300, 1)),
        ("neoclassical_tearing_mode", (900, 1100, 1)),
    ):
        event = paths.label_tables / name
        (event / "format").mkdir(parents=True)
        # B has only 170815, so the event menu resumes the same shot too.
        roster = ROSTER if not events else "\n".join(ROSTER.splitlines()[:2]) + "\n"
        (event / "shots.csv").write_text(roster)
        (event / f"format/{name}_format_2026_v1.csv").write_text(SOURCE)
        labels.save(event, 170815, labels.normalise((0, 2000), [span]), source=None)
        events.append(event)
    labels.save(
        events[0], 170816, labels.normalise((0, 2000), [(400, 600, 1)]), source=None
    )
    target = paths.spectrogram_file(events[1].name, 170815)
    target.parent.mkdir(parents=True)
    shutil.copyfile(paths.spectrogram_file(events[0].name, 170815), target)
    app = create_app(paths=paths, token=TOKEN)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        while not server.started:
            time.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}", events
    finally:
        server.should_exit = True
        thread.join(10)


def test_pending_navigation_cannot_mix_history_or_labels(served_events, tmp_path):
    base, (event_a, event_b) = served_events
    result = subprocess.run(
        [
            NODE,
            str(DRIVER),
            base,
            TOKEN,
            str(SHELLS[-1]),
            str(tmp_path / "profile"),
            "race",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    checks = json.loads(result.stdout.splitlines()[-1])
    failed = [c for c in checks if not c["ok"]]
    assert not failed, json.dumps(failed, indent=2)
    assert len(checks) == 30
    history_a = labels.read_history(event_a)
    assert [(v["shot"], v["intervals"]) for v in history_a] == [
        (170815, [[100, 300, 1]]),
        (170816, [[400, 600, 1]]),
        (170816, [[400, 600, 1]]),
    ]
    history_b = labels.read_history(event_b)
    assert [(v["shot"], v["intervals"]) for v in history_b] == [
        (170815, [[900, 1100, 1]])
    ]
