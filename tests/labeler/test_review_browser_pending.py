"""Edits and acknowledged saves survive pending save and queue responses."""

from __future__ import annotations

import json
import subprocess

import pytest

from labeler.events.review import labels

from .test_review_browser import DRIVER, NODE, SHELLS, TOKEN
from .test_review_browser_race import served_events as pending_events  # noqa: F401

pytestmark = pytest.mark.skipif(
    NODE is None or not SHELLS, reason="needs node and a headless Chromium"
)


def _run_browser(base, tmp_path, case, count):
    result = subprocess.run(
        [
            NODE,
            str(DRIVER),
            base,
            TOKEN,
            str(SHELLS[-1]),
            str(tmp_path / "profile"),
            "pending",
            case,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-2000:]
    checks = json.loads(result.stdout.splitlines()[-1])
    failed = [check for check in checks if not check["ok"]]
    assert not failed, json.dumps(failed, indent=2)
    assert len(checks) == count


@pytest.mark.parametrize("action", ["restore", "undo"])
@pytest.mark.parametrize("route", ["move", "type", "switch"])
def test_later_label(pending_events, tmp_path, action, route):  # noqa: F811
    base, (event_a, event_b) = pending_events
    _run_browser(base, tmp_path, f"{action}_{route}", 9)
    history = labels.read_history(event_a)
    assert [entry["shot"] for entry in history] == [170815, 170816, 170815]
    saved = labels.read_saved(event_a)[170815]
    assert history[-1]["intervals"] == [list(span) for span in saved.intervals]
    first, (start, stop, category) = saved.intervals
    assert first == (100, 300, 1)
    assert abs(start - 500) <= 2 and abs(stop - 800) <= 2 and category == 1
    assert [(v["shot"], v["intervals"]) for v in labels.read_history(event_b)] == [
        (170815, [[900, 1100, 1]])
    ]
