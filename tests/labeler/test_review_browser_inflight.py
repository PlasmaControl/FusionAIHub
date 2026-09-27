"""Delayed save answers preserve later edits, restores and navigation."""

from __future__ import annotations

import json
import subprocess

import pytest

from labeler.events.review import labels

from .test_review_browser import DRIVER, NODE, SHELLS, TOKEN, served  # noqa: F401

pytestmark = pytest.mark.skipif(
    NODE is None or not SHELLS, reason="needs node and a headless Chromium"
)


@pytest.mark.parametrize("case", ["move", "edit", "restore", "next", "return"])
def test_save_finishes_without_losing_a_later_change(served, tmp_path, case):  # noqa: F811
    base, event = served
    result = subprocess.run(
        [
            NODE,
            str(DRIVER),
            base,
            TOKEN,
            str(SHELLS[-1]),
            str(tmp_path / "profile"),
            "inflight",
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
    assert len(checks) == (9 if case in {"move", "next", "restore"} else 8)
    history = labels.read_history(event)
    assert [entry["shot"] for entry in history] == [170816] + [170815] * (
        2 if case == "restore" else 1
    )
    saved = labels.read_saved(event)[170815]
    assert history[-1]["intervals"] == [list(span) for span in saved.intervals]
    first, (start, stop, category) = saved.intervals
    assert first == (100, 300, 1)
    assert abs(start - 500) <= 2 and abs(stop - 800) <= 2 and category == 1
