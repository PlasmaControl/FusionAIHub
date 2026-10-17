"""R2_CHIPS in a real browser: a list of `N` shots on the event page's chips and in All (drawn
in part; K and Find shot keep the open shot's chip or row in view and marked; the chips' line
measured on a real chip), the owner's Delete on a drafted span with the focus left in the event
dropdown or in Resolution (Save writes the label without it), keys after a pick in the list
dropdown, an event's default class drawn in its unmarked time on the event page and in All, and
All's roster walk across events. The scenario is review_long_list.mjs, run on
review_browser.mjs's transport as the D1 checks are."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from labeler.config import Paths
from labeler.events.review import labels
from labeler.events.ui.app import create_app

from .test_review_browser import DRIVER, NODE, ROSTER, SHELLS, SOURCE, TOKEN, _store
from .test_review_page_d1_browser import _lock_store

SCENARIO = Path(__file__).with_name("review_long_list.mjs")
#: Shots on the big list: past `LongList.WHOLE` (500), so both places draw it in part.
N = 3000
REGISTRY = """\
lists:
  big: {title: "Big list", kind: dataset}
events:
  alfven_eigenmode: {recommended: [big]}
"""
#: AE's unmarked time reads as uncertain here; locked_mode keeps no default.
EVENTS = """\
taxonomy:
  alfven_eigenmode: {display: "Alfven eigenmode", default: 2}
"""

pytestmark = pytest.mark.skipif(
    NODE is None or not SHELLS, reason="needs node and a headless Chromium"
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def served(tmp_path, monkeypatch):
    """AE (170815 opening on its source's span, 170816 saved, 170817), locked_mode
    (170815), and `big`: N shots around them, in order. Nothing is fetched."""
    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    paths = Paths(
        root=tmp_path / "root", corpus=tmp_path / "corpus",
        text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tmp_path / "tables", raw_cache=tmp_path / "raw",
        config_dir=tmp_path / "config",
    )
    tables = paths.label_tables
    ae = tables / "alfven_eigenmode"
    _write(ae / "shots.csv", ROSTER)
    _write(ae / "format/alfven_eigenmode_format_2026_v1.csv", SOURCE)
    for shot in (170815, 170816, 170817):
        _store(paths, shot)
    labels.save(ae, 170816, labels.normalise((0, 2000), [(400, 600, 1)]), source=None)
    _write(tables / "locked_mode/shots.csv",
           "shot,tier,holdout,reviewers,verified_on,notes\n170815,unverified,false,,,\n")
    _lock_store(paths, 170815)
    start = 170815 - N // 2
    _write(tables / "shotlists/big.csv",
           "shot\n" + "".join(f"{shot}\n" for shot in range(start, start + N)))
    _write(paths.shotlists_manifest, REGISTRY)
    _write(paths.events_manifest, EVENTS)
    server = uvicorn.Server(uvicorn.Config(
        create_app(paths=paths, token=TOKEN), host="127.0.0.1", port=0,
        log_level="warning",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", paths
    server.should_exit = True
    thread.join(10)


def test_a_long_list_and_the_label_in_a_browser(served, tmp_path):
    base, paths = served
    script = tmp_path / SCENARIO.name
    # The driver's browser transport, without its scenarios.
    script.write_text(DRIVER.read_text().split("const opened =", 1)[0] + SCENARIO.read_text())
    result = subprocess.run(
        [NODE, str(script), base, TOKEN, str(SHELLS[-1]), str(tmp_path / "profile"), str(N)],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    checks = json.loads(result.stdout.strip().splitlines()[-1])
    failed = [item for item in checks if not item["ok"]]
    assert not failed, json.dumps(failed, indent=1)[:6000]
    assert len(checks) == 12
    # Save wrote the drafted label without its span: the window alone, nothing marked (the
    # unmarked time's default class is how it reads, never a stored span).
    saved = labels.read_labels(labels.labels_path(paths.label_tables / "alfven_eigenmode"))
    assert saved[170815].window == (0, 2000)
    assert saved[170815].intervals == ()
