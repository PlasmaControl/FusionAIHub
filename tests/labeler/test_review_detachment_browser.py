"""Detachment playback, timeline seek, asynchronous frames and label editing."""

from __future__ import annotations

import json
import subprocess

import h5py
import numpy as np
import pytest

from labeler.config import Paths
from labeler.events.review import build, labels

from .test_review_browser import DRIVER, NODE, SHELLS, TOKEN, served  # noqa: F401
from .test_review_detachment import corpus

pytestmark = pytest.mark.skipif(
    NODE is None or not SHELLS, reason="needs node and a headless Chromium"
)


def test_video_slider_playback_clicks_and_labels(served, tmp_path):  # noqa: F811
    base, existing = served
    paths = Paths(
        root=tmp_path / "root", corpus=tmp_path / "corpus", label_tables=existing.parent
    )
    source = corpus(paths)
    with h5py.File(source, "a") as store:
        store["tangtv/ydata"][1] = np.broadcast_to(
            np.arange(13)[:, None, None], (13, 4, 6)
        )
        for name, channels in (("filterscopes", 8), ("gas_flow", 11), ("co2", 4)):
            group = store.create_group(name)
            group["xdata"] = np.arange(13) * 0.02
            group["ydata"] = np.broadcast_to(np.arange(13), (channels, 13))
    event = paths.label_tables / "detachment"
    (event / "format").mkdir(parents=True)
    (event / "shots.csv").write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n170815,unverified,false,,,\n"
    )
    (event / "format/detachment_format_test.csv").write_text(
        "shot,category,t_start,t_end,confidence\n170815,1,0,60,\n170815,0,60,240,\n"
    )
    build.build("detachment", 170815, paths)
    result = subprocess.run(
        [
            NODE,
            str(DRIVER),
            base,
            TOKEN,
            str(SHELLS[-1]),
            str(tmp_path / "video-profile"),
            "detachment",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-2000:]
    checks = json.loads(result.stdout.splitlines()[-1])
    assert not [c for c in checks if not c["ok"]], checks
    assert any(c == 2 for _, _, c in labels.read_saved(event)[170815].intervals)
