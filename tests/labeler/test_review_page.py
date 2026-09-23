"""The review page's own rules, run under node: they must agree with the server's."""

from __future__ import annotations

import json
import random
import shutil
import subprocess

import pytest

from labeler.events.review.labels import normalise
from labeler.events.ui.app import STATIC

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
