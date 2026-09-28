"""An older review server still accepts a drawn label from the current page."""

from __future__ import annotations

import getpass
import json
import subprocess

import pytest

from labeler.events.review import labels

from .test_review_browser import DRIVER, NODE, SHELLS, TOKEN, served  # noqa: F401

pytestmark = pytest.mark.skipif(
    NODE is None or not SHELLS, reason="needs node and a headless Chromium"
)


def test_an_older_server_saves_without_a_name(served, tmp_path):  # noqa: F811
    base, event = served
    result = subprocess.run(
        [
            NODE,
            str(DRIVER),
            base,
            TOKEN,
            str(SHELLS[-1]),
            str(tmp_path / "profile"),
            "api1",
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
    assert len(checks) == 12
    history = [entry for entry in labels.read_history(event) if entry["shot"] == 170815]
    assert len(history) == 1
    assert history[0]["name"] is None
    assert history[0]["reviewer"] == getpass.getuser()
    kept, (a, b, c) = history[0]["intervals"]
    assert kept == [100, 300, 1] and c == 1 and abs(a - 500) <= 2 and abs(b - 800) <= 2
