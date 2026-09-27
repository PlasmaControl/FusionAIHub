"""An empty event clears the previous shot and finishes opening."""

from __future__ import annotations

import json
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
def served_empty_event(tmp_path):
    """One saved shot and a second event with a header-only roster, all local."""
    paths = Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tmp_path / "tables",
        raw_cache=tmp_path / "raw",
    )
    _store(paths, 170815)
    events = []
    for name, count in (("alfven_eigenmode", 2), ("neoclassical_tearing_mode", 1)):
        event = paths.label_tables / name
        (event / "format").mkdir(parents=True)
        (event / "shots.csv").write_text("\n".join(ROSTER.splitlines()[:count]) + "\n")
        source = SOURCE if count == 2 else SOURCE.splitlines()[0] + "\n"
        (event / f"format/{name}_format_2026_v1.csv").write_text(source)
        events.append(event)
    labels.save(
        events[0], 170815, labels.normalise((0, 2000), [(100, 300, 1)]), source=None
    )
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


def test_empty_event_clears_and_settles_then_allows_navigation(
    served_empty_event, tmp_path
):
    base, (event_a, event_b) = served_empty_event
    result = subprocess.run(
        [
            NODE,
            str(DRIVER),
            base,
            TOKEN,
            str(SHELLS[-1]),
            str(tmp_path / "profile"),
            "empty",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:] + result.stdout[-4000:]
    checks = json.loads(result.stdout.splitlines()[-1])
    failed = [c for c in checks if not c["ok"]]
    assert not failed, json.dumps(failed, indent=2)
    assert len(checks) == 41
    history = labels.read_history(event_a)
    assert len(history) == 2
    assert history[0]["intervals"] == [[100, 300, 1]]
    assert history[1]["shot"] == 170815
    kept, (start, end, category) = history[1]["intervals"]
    assert kept == [100, 300, 1]
    assert category == 1 and abs(start - 500) <= 2 and abs(end - 800) <= 2
    assert labels.read_history(event_b) == []
    assert not labels.history_path(event_b).exists()
