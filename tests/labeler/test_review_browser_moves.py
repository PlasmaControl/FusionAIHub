"""History keys preserve drafts, and shot moves follow the queue."""

from __future__ import annotations

import json
import subprocess

import pytest

from labeler.events.review import labels

from .test_review_browser import DRIVER, NODE, SHELLS, TOKEN, served  # noqa: F401

pytestmark = pytest.mark.skipif(
    NODE is None or not SHELLS, reason="needs node and a headless Chromium"
)


def test_history_keys_and_moves_keep_the_reviewers_edit(served, tmp_path):  # noqa: F811
    base, event = served
    result = subprocess.run(
        [
            NODE,
            str(DRIVER),
            base,
            TOKEN,
            str(SHELLS[-1]),
            str(tmp_path / "profile"),
            "moves",
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
    assert len(checks) == 20
    history = labels.read_history(event)
    assert [(v["shot"], v["window"], v["intervals"]) for v in history] == [
        (170816, [0, 2000], [[400, 600, 1]]),
        (170817, [0, 4000], []),
    ]
