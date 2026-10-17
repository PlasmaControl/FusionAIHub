"""Plan D.8's twelve D1 checks in a real browser (shot lists, All, the activity line and
the TokEye chips) and 13 (the flat event dropdown with All), against an api-11 server (with stubbed api-10, failing and
expired-fdp answers where a check needs them). The scenario is review_page_d1.mjs,
run on review_browser.mjs's transport as test_review_silver.py runs its own."""

from __future__ import annotations

import json
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

from .test_review_browser import DRIVER, NODE, ROSTER, SHELLS, SOURCE, TOKEN, _store
from .test_review_masks_browser import _layer, _pseudo

SCENARIO = Path(__file__).with_name("review_page_d1.mjs")
LOCK = "locked_mode"
#: The six TokEye states the scenario stubs on locked_mode 170815, one image row each.
STATE_ROWS = ("k_done", "k_stale", "k_queued", "k_computing", "k_failed", "k_new")
REGISTRY = """\
lists:
  silver_batch_1: {title: "Silver batch 1", parent: silver}
  silver_batch_2: {title: "Silver batch 2", parent: silver}
  ae_a: {title: "AE list A"}
  ae_b: {title: "AE list B"}
events:
  alfven_eigenmode: {recommended: [ae_a, gold]}
"""

pytestmark = pytest.mark.skipif(
    NODE is None or not SHELLS, reason="needs node and a headless Chromium"
)


def _lock_store(paths: Paths, shot: int) -> None:
    """Six spectrograms (kHz), one image that is not (TokEye does not apply), a trace."""
    t = np.arange(4000) * 0.5
    bins = np.arange(32)[:, None]
    image = (40 + 200 * np.exp(-((bins - 16 - 8 * np.sin(t / 200)) ** 2) / 4))
    image = image.astype("uint8")
    rows = [
        ImageRow(name, name.replace("_", " "), image, y0=0.0, dy=4.0, y_units="kHz",
                 z_lo=-3.0, z_hi=27.0, z_units="dB")
        for name in STATE_ROWS
    ]
    rows.append(ImageRow("k_plain", "plain", image, y0=0.0, dy=1.0, y_units="ch",
                         z_lo=0.0, z_hi=1.0, z_units=""))
    wave = np.sin(t / 120)[None]
    rows.append(TraceRow("p1", "Density", np.stack([wave, wave]), hlines=[0.0]))
    review_rows.write(paths.spectrogram_file(LOCK, shot), Grid(0.0, 0.5, 4000), rows,
                      event=LOCK)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def served(tmp_path, monkeypatch):
    """AE (170815 with a pseudo-mask and a tokeye-full layer, 170816 saved and with
    neither, 170817, and 170899 off its roster on silver batch 2), locked_mode
    (170815, 170816, and 170899 off its roster), gold (170815 and 171234, which no
    store has), silver in two batches, and two equal-length AE lists. Nothing is
    fetched: a shot with no store fails its build."""
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
    _write(ae / "format/extra.csv", "shot,category,t_start,t_end,confidence\n"
                                     "170899,0,0,1000,\n170899,1,1000,1500,\n")
    for shot in (170815, 170816, 170817, 170899):
        _store(paths, shot)
    _pseudo(paths, 170815)
    _layer(paths, 170815)
    labels.save(ae, 170816, labels.normalise((0, 2000), [(400, 600, 1)]), source=None)
    lock = tables / LOCK
    _write(lock / "shots.csv", "shot,tier,holdout,reviewers,verified_on,notes\n"
                               "170815,unverified,false,,,\n170816,unverified,false,,,\n")
    for shot in (170815, 170816, 170899):
        _lock_store(paths, shot)
    _write(tables / "gold/gold_shots.csv",
           "shot,run_day,year,classes,sources,notes\n"
           "170815,20260101,2026,alfven_eigenmode:saved,saved,\n"
           "171234,20260101,2026,,,\n")
    _write(tables / "silver/silver_shots.csv",
           "shot,batch,order,pool,classes,sources,notes\n"
           "170815,1,1,corpus,alfven_eigenmode:present,format,\n"
           "170899,2,1,ae_audited,alfven_eigenmode:present,format,\n")
    _write(tables / "shotlists/silver_batch_1.csv",
           "shot,batch,order,pool,classes\n170815,1,1,corpus,alfven_eigenmode:present\n")
    _write(tables / "shotlists/silver_batch_2.csv",
           "shot,batch,order,pool,classes\n"
           "170899,2,1,ae_audited,alfven_eigenmode:present\n")
    _write(tables / "shotlists/ae_a.csv", "shot\n170815\n170816\n")
    _write(tables / "shotlists/ae_b.csv", "shot\n170816\n170817\n")
    _write(paths.shotlists_manifest, REGISTRY)
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


def test_the_d1_page_in_a_browser(served, tmp_path):
    base, paths = served
    script = tmp_path / "review_page_d1.mjs"
    # The driver's browser transport, without its scenarios.
    prefix = DRIVER.read_text().split("const opened =", 1)[0]
    script.write_text(prefix + SCENARIO.read_text())
    result = subprocess.run(
        [NODE, str(script), base, TOKEN, str(SHELLS[-1]), str(tmp_path / "profile")],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    checks = json.loads(result.stdout.splitlines()[-1])
    print(json.dumps(checks, indent=1))  # the report's numbers (pytest -rP)
    assert [c for c in checks if not c["ok"]] == [], checks
    assert [c["name"].split(".")[0] for c in checks] == [str(n) for n in range(1, 14)]
    saved = labels.read_saved(paths.label_tables / "alfven_eigenmode")
    assert 170899 in saved, "By shot's save reached the event's table"
