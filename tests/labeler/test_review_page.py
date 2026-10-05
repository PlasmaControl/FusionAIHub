"""The review page's own rules, run under node: they must agree with the server's."""

from __future__ import annotations

import json
import random
import shutil
import subprocess

import numpy as np
import pytest

from labeler.events.review.labels import normalise
from labeler.events.ui.app import STATIC
from labeler.events.verify import mode_palette

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node runs the page's script")
APP = STATIC / "app.js"


def _node(expression: str, payload=None):
    """`expression` over `m` (the page's exports) and `input` (the payload)."""
    script = (
        "const m = require(process.argv[1]);"
        "const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        f"process.stdout.write(JSON.stringify({expression}));"
    )
    result = subprocess.run(
        [NODE, "-e", script, str(APP)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def test_the_page_loads_nothing_from_the_network():
    for name in ("index.html", "app.js", "style.css"):
        assert "http" not in (STATIC / name).read_text().lower(), name


@needs_node
def test_the_script_parses():
    subprocess.run([NODE, "--check", str(APP)], check=True, timeout=30)


@needs_node
def test_producer_strips_use_half_open_bins_and_preserve_vote_validity():
    found = _node(
        "[99,100,149,150,200].map(t => m.producerAt(input, t))",
        {
            "bin_start_ms": [100, 150],
            "bin_end_ms": [150, 200],
            "state_lm": [2, 4],
            "state_rule": [2, 4],
            "votes": {
                "afrac": {"vote": [-1, -1], "valid": [True, False],
                          "reason": ["", "no_probes"]},
            },
            "tangtv_source": ["surrogate", "none"],
            "confidence": [0.9, 0.5],
        },
    )
    assert found[0] is None and found[-1] is None
    assert found[1] == found[2]
    assert found[1]["state"] == 2
    assert found[1]["votes"]["afrac"] == {
        "vote": -1, "valid": True, "reason": "",
    }
    assert found[3]["votes"]["afrac"] == {
        "vote": -1, "valid": False, "reason": "no_probes",
    }


def _cases():
    rng = random.Random(0)
    cases = []
    for _ in range(300):
        lo = rng.uniform(-50, 50)
        width = rng.choice([0.3, 0.5, 40.0, 900.0, 25000.0])
        spans = []
        for _ in range(rng.randint(0, 4)):
            a = rng.uniform(-100, 1000)
            spans.append([a, a + rng.uniform(-3, 200), rng.choice([0, 1, 1, 1, 2])])
        cases.append({"window": [lo, lo + width], "intervals": spans})
    # Halves round up on both sides, and a span may land on the window's edge.
    cases.append({"window": [0.5, 10.5], "intervals": [[1.5, 2.5, 1], [-0.5, 0.5, 1]]})
    cases.append({"window": [-2.5, 2.5], "intervals": [[-1.5, -0.5, 1], [0.5, 0.5, 1]]})
    return cases


def _server(case):
    try:
        return normalise(case["window"], case["intervals"], known={1}).as_json()
    except ValueError:
        return None


@needs_node
def test_the_page_normalises_a_label_exactly_as_the_server_does():
    """What the page shows while editing is what a save will store."""
    cases = _cases()
    page = _node("input.map((c) => m.normalise(c.window, c.intervals, [1]))", cases)
    server = [_server(case) for case in cases]
    assert page == server
    assert None in server and any(server)


@needs_node
def test_a_change_is_where_the_label_leaves_its_source():
    source = {"window": [0, 100], "intervals": [[10, 20, 1]]}
    labels = [
        source,
        {"window": [0, 100], "intervals": [[10, 30, 1]]},
        {"window": [0, 120], "intervals": [[10, 20, 1]]},
        {"window": [0, 100], "intervals": []},
    ]
    found = _node("input.labels.map((b) => m.diffRuns(input.source, b))",
                  {"source": source, "labels": labels})
    assert found == [[], [[20, 30]], [[100, 120]], [[10, 20]]]


@needs_node
def test_each_saved_version_counts_the_ms_it_changed_from_the_one_before():
    """The history's `ms changed`: the first version against the source, if any."""
    source = {"window": [0, 2000], "intervals": [[100, 300, 1]]}
    versions = [
        {"window": [0, 2000], "intervals": [[100, 300, 1]]},
        {"window": [0, 2000], "intervals": [[100, 300, 1], [500, 800, 1]]},
        {"window": [0, 2000], "intervals": [[150, 300, 1], [500, 800, 1]]},
        {"window": [0, 2100], "intervals": [[150, 300, 1], [500, 800, 1]]},
    ]
    found = _node(
        "[m.versionChanges(input.versions, input.source),"
        " m.versionChanges(input.versions.slice(0, 2), null),"
        " m.versionChanges([], input.source)]",
        {"source": source, "versions": versions},
    )
    assert found == [[0, 300, 50, 100], [None, 300], []]


@needs_node
def test_ticks_step_by_one_two_or_five():
    spans = [[1000, 10], [9000, 8], [50, 10], [3.7, 5]]
    assert _node("input.map(([s, n]) => m.niceStep(s, n))", spans) == [100, 2000, 5, 1]


@needs_node
def test_a_press_on_the_label_grabs_the_nearest_thing():
    """Window edges only in the foot strip; span edges before span bodies."""
    label = {"window": [0, 1000], "intervals": [[100, 300, 1], [500, 520, 1]]}
    points = [[100, 5], [200, 5], [700, 5], [0, 35], [1000, 35], [510, 5], [298, 35]]
    found = _node(
        "input.points.map(([x, y]) => m.hitTest(input.label, x, y, 40, (t) => t))",
        {"label": label, "points": points},
    )
    assert found == [
        {"kind": "edge", "index": 0, "edge": 0},
        {"kind": "move", "index": 0},
        {"kind": "new"},
        {"kind": "window", "edge": 0},
        {"kind": "window", "edge": 1},
        {"kind": "move", "index": 1},
        {"kind": "edge", "index": 0, "edge": 1},
    ]


@needs_node
def test_the_colour_map_runs_from_inferno_black_to_its_yellow():
    """Packed as RGBA bytes; values under the contrast floor take its colour."""
    found = _node("[m.lut(0)[0], m.lut(0)[255], m.lut(128)[100] === m.lut(128)[0]]")
    assert found == [0xFF040000, 0xFFA4FFFC, True]


@needs_node
def test_a_modes_row_draws_each_code_in_its_mode_colour_scaled_by_its_level():
    modes = {"n": [1, 2], "levels": 128, "colours": ["#ff0000", "#00ff00"]}
    found = _node("[0, 128].map((lo) => Array.from(m.modeLut(input, lo)))", modes)
    rgba = lambda r, g, b: (255 << 24) | (b << 16) | (g << 8) | r
    plain, floored = [np.array(lut) for lut in found]
    assert plain[:2].tolist() == [rgba(0, 0, 0)] * 2
    assert plain[254:].tolist() == [rgba(255, 0, 0), rgba(0, 255, 0)]
    # The page draws what `verify.mode_palette` gives the notebook.
    palette = mode_palette({1: "#ff0000", 2: "#00ff00"})
    assert plain.tolist() == [
        rgba(*(int(c[i : i + 2], 16) for i in (1, 3, 5))) for c in palette
    ]
    # The contrast floor blacks out the lower half and stretches the rest.
    assert (floored[:128] == rgba(0, 0, 0)).all() and floored[254] == rgba(255, 0, 0)


@needs_node
def test_a_click_picks_the_mask_region_nearest_it_within_reach():
    """Runs are `[bin, first column, length]`; reach is in bins and columns."""
    found = _node(
        "input.points.map(([j, k, tj, tk]) => m.regionAt(input.regions, j, k, tj, tk)?.id ?? null)",
        {
            "regions": [
                {"id": 1, "runs": [[100, 10, 5], [101, 10, 5]]},
                {"id": 2, "runs": [[104, 16, 3]]},
            ],
            "points": [
                [100, 12, 0, 0],
                [102, 12, 0, 0],
                [102, 12, 1, 0],
                [103, 15, 2, 2],
                [104, 20, 0, 2],
                [104, 21, 0, 2],
                [90, 12, 3, 3],
            ],
        },
    )
    assert found == [1, None, 1, 2, 2, None, None]
