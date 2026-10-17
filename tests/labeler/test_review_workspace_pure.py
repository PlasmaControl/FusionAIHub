"""The workspace scripts' pure rules (plan D.1-D.3, D.A, D.B), run under node.

signals.js, slice.js, dock.js and views.js are classic scripts loaded before app.js: each
defines only the names plan D.7 gives it and does nothing at load. Their pure helpers run here
in one node context with D1's status.js; the few page flows run against stubs of app.js.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from labeler.events.review import catalog
from labeler.events.ui.app import STATIC

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node runs the page's scripts")

#: D2's scripts, in load order, and the globals plan D.7 gives each.
GLOBALS = {
    "signals.js": {"Signals", "Profiles", "loadSignal", "loadProfile", "profileAt", "unpackTrace",
                   "sliceIndex"},
    "slice.js": {"Slice", "sliceTime", "notePointer", "restAfter", "pinAt", "unpinNear",
                 "axisClickPin", "undoAxisClicks", "drawPins", "placePinTabs", "valueAt",
                 "copyValues", "cutMargin", "drawMargins", "axisKind"},
    "dock.js": {"Dock", "buildDock", "drawDock", "toggleDock", "panelProfile", "panelValues"},
    "views.js": {"Views", "defaultLayout", "applyView", "layoutRows", "drawAddedRows", "saveView",
                 "renameView", "deleteView", "resetView", "openEditor", "openPicker",
                 "matchCatalog"},
}
#: D1's scripts, loaded first on the page.
EARLIER = ("status.js", "lists.js", "overlays.js")
TOP_LEVEL = re.compile(
    r"^(?:async\s+)?(?:function\*?|const|let|var|class)\s+([A-Za-z_$][\w$]*)", re.MULTILINE
)


def _page(expression: str, payload=None, *, setup: str = ""):
    """`expression`, awaited, in one node context holding status.js and the four scripts
    after `setup`; `input` is `payload`."""
    script = (
        "const fs = require('fs'), vm = require('vm');"
        "const [dir, files, setup, expression] = JSON.parse(process.argv[1]);"
        "const input = JSON.parse(fs.readFileSync(0, 'utf8'));"
        "const context = vm.createContext({ module: { exports: {} }, input });"
        "vm.runInContext(setup, context, { filename: 'setup' });"
        "for (const name of files)"
        "  vm.runInContext(fs.readFileSync(dir + '/' + name, 'utf8'), context, { filename: name });"
        "Promise.resolve(vm.runInContext(expression, context, { filename: 'test' })).then("
        "  (value) => process.stdout.write(JSON.stringify(value === undefined ? null : value)),"
        "  (error) => { process.stderr.write(String(error && error.stack || error)); process.exit(1); });"
    )
    files = ["status.js", *GLOBALS]
    arguments = json.dumps([str(STATIC), files, setup, expression])
    result = subprocess.run(
        [NODE, "-e", script, arguments],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _top_level(name: str) -> list[str]:
    return TOP_LEVEL.findall((STATIC / name).read_text())


# -- the files


@needs_node
@pytest.mark.parametrize("name", list(GLOBALS))
def test_each_script_parses(name):
    subprocess.run([NODE, "--check", str(STATIC / name)], check=True, timeout=30)


@pytest.mark.parametrize("name", list(GLOBALS))
def test_each_script_loads_nothing_from_the_network(name):
    assert "http" not in (STATIC / name).read_text().lower()


@pytest.mark.parametrize("name", list(GLOBALS))
def test_each_script_defines_exactly_its_globals(name):
    found = _top_level(name)
    assert sorted(found) == sorted(GLOBALS[name]), found


@pytest.mark.parametrize("name", list(GLOBALS))
def test_no_global_is_also_app_or_d1s(name):
    others = set()
    for other in ("app.js", *EARLIER):
        if (STATIC / other).exists():
            others.update(_top_level(other))
    assert not GLOBALS[name] & others


@needs_node
@pytest.mark.parametrize("name", list(GLOBALS))
def test_each_script_exports_its_globals_for_node(name):
    script = "process.stdout.write(JSON.stringify(Object.keys(require(process.argv[1]))))"
    result = subprocess.run(
        [NODE, "-e", script, str(STATIC / name)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    assert set(json.loads(result.stdout)) == GLOBALS[name]


@needs_node
def test_the_scripts_load_with_d1s_and_app_and_do_nothing_at_load():
    """In page order in one context with no document: work at load would throw."""
    files = [*(name for name in EARLIER if (STATIC / name).exists()), *GLOBALS, "app.js"]
    script = (
        "const fs = require('fs'), vm = require('vm');"
        "const [dir, files] = JSON.parse(process.argv[1]);"
        "const context = vm.createContext({ module: { exports: {} } });"
        "for (const name of files)"
        "  vm.runInContext(fs.readFileSync(dir + '/' + name, 'utf8'), context, { filename: name });"
        "process.stdout.write(JSON.stringify(vm.runInContext("
        "  '[typeof Signals, typeof Slice, typeof Dock, typeof Views, typeof boot]', context)));"
    )
    result = subprocess.run(
        [NODE, "-e", script, json.dumps([str(STATIC), files])],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["object", "object", "object", "object", "function"]


# -- signals.js


@needs_node
def test_slice_index_is_the_nearest_time_and_a_tie_goes_earlier():
    got = _page(
        "[[], NaN, -100, 4, 5, 6, 15, 100].map((t, k) => k === 0 ? sliceIndex([], 1)"
        " : sliceIndex([0, 10, 20], t))"
    )
    assert got == [-1, -1, 0, 0, 0, 1, 1, 2]


@needs_node
def test_a_trace_unpacks_to_its_channels_and_a_profile_slices_by_time():
    got = _page(
        "(() => {"
        " const trace = unpackTrace(new Float32Array(12).buffer, 3);"
        " const header = { n_t: 2, n_x: 3, x_per_time: false };"
        " const body = new Float32Array([100, 200, 0, 0.5, 1, 1, 2, 3, 4, 5, 6]);"
        " const fixed = Profiles.parse(header, body.buffer);"
        " const moving = Profiles.parse({ n_t: 2, n_x: 2, x_per_time: true },"
        "   new Float32Array([100, 200, 0, 1, 2, 3, 7, 8, 9, 10]).buffer);"
        " let short = null;"
        " try { Profiles.parse(header, new Float32Array(5).buffer); } catch (error) { short = error.message; }"
        " return { channels: trace.channels, at160: profileAt(fixed, 160), at120: profileAt(fixed, 120),"
        "   moving: profileAt(moving, 190), none: profileAt({ t: [] }, 1), short,"
        "   span: Profiles.span([5, NaN, 1, 3, 2, 4], 0, 1) };"
        "})()"
    )
    assert got["channels"] == 2
    assert got["at160"]["i"] == 1 and list(got["at160"]["y"].values()) == [4, 5, 6]
    assert got["at120"]["i"] == 0 and list(got["at120"]["x"].values()) == [0, 0.5, 1]
    assert list(got["moving"]["x"].values()) == [2, 3]
    assert list(got["moving"]["y"].values()) == [9, 10]
    assert got["none"] is None
    assert "shorter" in got["short"]
    assert got["span"] == [1, 5]


@needs_node
def test_a_signal_polls_its_202_then_takes_its_window():
    setup = r"""
var S = { shot: 189081 };
var calls = [];
var answers = [
  { status: 202, body: { state: "queued", stage: "", elapsed_s: 2, queue_position: 2 } },
  { status: 202, body: { state: "fetching", stage: "PTDATA", elapsed_s: 4, queue_position: 0 } },
  { status: 200, values: [1, 2, 3, 4], headers: {
    "X-Grid": JSON.stringify({ t0: 0, t1: 100, n: 2 }),
    "X-Signal": JSON.stringify({ label: "DUSBRADIAL", units: "V", legend: [], source: "ptdata" }) } },
];
async function api(path, options = {}) {
  calls.push({ path, quiet: Boolean(options.quiet) });
  const answer = answers.shift();
  return { status: answer.status, json: async () => answer.body,
    arrayBuffer: async () => new Float32Array(answer.values).buffer,
    headers: { get: (key) => (answer.headers || {})[key] ?? null } };
}
"""
    got = _page(
        "(async () => {"
        " Signals.wait = async () => {};"
        " const states = [];"
        " const entry = loadSignal(189081, 'custom:DUSBRADIAL', 0, 100, 2, { name: 'DUSBRADIAL',"
        "   source: 'ptdata' }, (e) => states.push([e.state, e.queue_position]));"
        " for (let k = 0; k < 500 && entry.running; k++) await Promise.resolve();"
        " const again = loadSignal(189081, 'custom:DUSBRADIAL', 0, 100, 2);"
        " return { states, calls, data: { ...entry.data, values: Array.from(entry.data.values) },"
        "   label: entry.signal.label, same: again === entry, asked: calls.length,"
        "   bar: Activity.items.size };"
        "})()",
        setup=setup,
    )
    assert got["states"] == [["queued", 2], ["loading", 0], ["done", 0]]
    assert got["calls"][0]["path"].startswith("/api/signal?shot=189081&id=custom%3ADUSBRADIAL&t0=0")
    assert all(call["quiet"] for call in got["calls"])
    assert got["data"] == {"t0": 0, "t1": 100, "n": 2, "values": [1, 2, 3, 4], "channels": 1}
    assert got["label"] == "DUSBRADIAL"
    assert got["same"] and got["asked"] == 3  # the same window asks nothing more
    assert got["bar"] == 0  # the bar's item ended


@needs_node
def test_a_failed_signal_waits_for_retry_and_names_its_kind():
    setup = r"""
var S = { shot: 7 };
var calls = [];
async function api(path) {
  calls.push(path);
  const error = new Error("No DUSB data");
  error.status = 404;
  throw error;
}
"""
    got = _page(
        "(async () => {"
        " const entry = loadSignal(7, 'custom:DUSB', 0, 10, 4);"
        " for (let k = 0; k < 500 && entry.running; k++) await Promise.resolve();"
        " loadSignal(7, 'custom:DUSB', 0, 20, 4);"
        " return { state: entry.state, kind: entry.error.kind, asked: calls.length,"
        "   kinds: [400, 422, 503, 502, 504, 500].map((status) => Signals.kindOf({ status })),"
        "   named: Signals.kindOf({ status: 404, kind: 'auth' }) };"
        "})()",
        setup=setup,
    )
    assert got == {"state": "error", "kind": "missing", "asked": 1,
                   "kinds": ["invalid", "invalid", "auth", "upstream", "timeout", "upstream"],
                   "named": "auth"}


# -- slice.js


@needs_node
def test_the_rest_time_needs_a_dwell_and_ignores_jitter():
    got = _page(
        "(() => {"
        " const first = restAfter(null, 100, 2000, 0);"
        " const jitter = restAfter(first.anchor, 103, 2003, 50);"
        " const quick = restAfter(first.anchor, 110, 2010, 80);"
        " const dwelt = restAfter(first.anchor, 110, 2010, 120);"
        " const later = restAfter(dwelt.anchor, 300, 2600, 900);"
        " return { first, jitterKept: jitter.anchor === first.anchor, jitter: jitter.rest,"
        "   quick, dwelt: dwelt.rest, later: later.rest };"
        "})()"
    )
    assert got["first"] == {"anchor": {"x": 100, "t": 2000, "ts": 0}, "rest": None}
    assert got["jitterKept"] and got["jitter"] is None
    assert got["quick"] == {"anchor": {"x": 110, "t": 2010, "ts": 80}, "rest": None}
    assert got["dwelt"] == 2000
    assert got["later"] == 2010


@needs_node
def test_pins_set_a_then_b_then_move_the_nearer_and_undo():
    got = _page(
        "(() => {"
        " let pins = { A: null, B: null };"
        " const steps = [];"
        " for (const t of [2140, 2610, 2200, 2590]) { pins = Slice.withPin(pins, t); steps.push({ ...pins }); }"
        " const tie = Slice.withoutNearest({ A: 100, B: 300 }, 200);"
        " const keepB = Slice.withoutNearest({ A: 100, B: 300 }, 120);"
        " const none = Slice.withoutNearest({ A: null, B: null }, 5);"
        " const clicks = [{ pins: { A: 1, B: null }, at: 100 }, { pins: { A: 2, B: null }, at: 400 }];"
        " return { steps, tie, keepB, none, undo: Slice.undone(clicks, 550),"
        "   late: Slice.undone(clicks, 650), stale: Slice.undone(clicks, 1000) };"
        "})()"
    )
    assert got["steps"] == [
        {"A": 2140, "B": None}, {"A": 2140, "B": 2610},
        {"A": 2200, "B": 2610}, {"A": 2200, "B": 2590},
    ]
    assert got["tie"] == {"A": None, "B": 300}
    assert got["keepB"] == {"A": None, "B": 300}  # B is not promoted to A
    assert got["none"] == {"A": None, "B": None}
    assert got["undo"] == {"A": 1, "B": None}  # the oldest snapshot of the last 500 ms
    assert got["late"] == {"A": 2, "B": None}
    assert got["stale"] is None


@needs_node
def test_slice_time_follows_a_pin_else_the_hover_else_the_rest_or_a():
    got = _page(
        "(() => {"
        " const out = [];"
        " Object.assign(Slice, { inRows: true, hover: 10, rest: 20, pins: { A: 30, B: 40 } });"
        " out.push(sliceTime());"
        " Slice.inRows = false; out.push(sliceTime());"
        " Slice.rest = null; out.push(sliceTime());"
        " Slice.follow = 'B'; Slice.inRows = true; out.push(sliceTime());"
        " Slice.pins.B = null; out.push(sliceTime());"
        " return out;"
        "})()"
    )
    assert got == [10, 20, 30, 40, 10]


@needs_node
def test_buckets_cells_and_peaks():
    got = _page(
        "(() => {"
        " const grid = { t0: 0, t1: 100, n: 4 };"
        " const trace = new Float32Array([1, 2, 3, NaN, 10, 20, 30, 40, 1.5, 2.5, 3.5, NaN, 11, 21, 31, 41]);"
        " const image = new Uint8Array([0, 5, 0, 0,  0, 9, 0, 0,  0, 9, 0, 0]);"
        " const modes = new Uint8Array([0, 4 * 3 + 1, 0, 0,  0, 2 * 3 + 2, 0, 0]);"
        " return {"
        "   buckets: [-1, 0, 24.9, 25, 99.9, 100].map((t) => Slice.bucket(grid, t)),"
        "   cell: Slice.traceCell(trace, 4, 2, 1, 1), empty: Slice.traceCell(trace, 4, 2, 0, 3),"
        "   peak: Slice.peak(image, 4, 1, 0, 3), band: Slice.peak(image, 4, 1, 0, 1),"
        "   dark: Slice.peak(image, 4, 0, 0, 3), mode: Slice.peak(modes, 4, 1, 0, 2, 3) };"
        "})()"
    )
    assert got["buckets"] == [-1, 0, 0, 1, 3, -1]
    assert got["cell"] == {"lo": 20, "hi": 21, "mid": 20.5}
    assert got["empty"] is None
    assert got["peak"] == {"bin": 1, "level": 9, "mode": None}  # the first of equal peaks
    assert got["band"] == {"bin": 0, "level": 5, "mode": None}
    assert got["dark"] is None  # no data reads "–", never 0
    assert got["mode"] == {"bin": 0, "level": 4, "mode": 1}


@needs_node
def test_readout_numbers_times_and_values():
    got = _page(
        "(() => {"
        " const trace = (lo, hi) => ({ kind: 'trace', cells: [{ lo, hi, mid: (lo + hi) / 2 }], range: [0, 100] });"
        " return {"
        "   numbers: [2140.4, 0.12345, -3.456, 1.5e6, 0.0002, 0, NaN].map(Slice.number),"
        "   times: [Slice.time(2140), Slice.time(-20), Slice.signed(470), Slice.signed(-20), Slice.time(null)],"
        "   values: [Slice.valueText(trace(5, 5.5)).text, Slice.valueText(trace(5, 7)).text,"
        "     Slice.valueText({ kind: 'image', y: 8.12, units: 'kHz', level: 182 }).text,"
        "     Slice.valueText({ kind: 'modes', n: -2, level: 3 }).text, Slice.valueText(null).text],"
        "   plain: [Slice.plain(1 / 3), Slice.plain(NaN)] };"
        "})()"
    )
    assert got["numbers"] == ["2,140", "0.123", "−3.46", "1.50e+6", "2.00e-4", "0", "–"]
    assert got["times"] == ["2,140", "−20", "+470 ms", "−20 ms", "–"]
    assert got["values"] == ["5.25", "5–7", "peak 8.12 kHz · 182", "n = −2", "–"]
    assert got["plain"] == ["0.333333", ""]


@pytest.mark.parametrize(
    ("units", "kind"),
    [
        ("kHz", "freq_khz"), ("f (kHz)", "freq_khz"), ("R (m)", "R_m"), ("r_m", "R_m"),
        ("major radius (m)", "R_m"), ("channel", "channel"), ("chord", "channel"),
        ("ψN", "psi_n"), ("psi_n", "psi_n"), ("normalised flux", "psi_n"), ("ρ", "rho"),
        ("rhovn", "rho"), ("rho_tor", "rho"), ("ms", "time_ms"), ("time (ms)", "time_ms"),
        ("keV", None), ("", None), (None, None),
    ],
)
@needs_node
def test_axis_kinds_never_convert(units, kind):
    assert _page("axisKind(input)", units) == kind


# -- dock.js (plan D.B)


@needs_node
def test_r_z_axes_keep_their_aspect_and_put_z_up():
    got = _page(
        "(() => {"
        " const fit = Dock.fitRZ([1, 2.5], [-1.5, 1.5], 300, 300);"
        " const wide = Dock.fitRZ([1, 2.5], [-1.5, 1.5], 900, 300);"
        " return { fit, corners: [Dock.toPixel(fit, 1, 1.5), Dock.toPixel(fit, 2.5, -1.5),"
        "   Dock.toPixel(fit, 1.75, 0)], wide, wideCorner: Dock.toPixel(wide, 1, 1.5) };"
        "})()"
    )
    assert got["fit"] == {"k": 100, "left": 75, "top": 0, "r0": 1, "z1": 1.5}
    assert got["corners"] == [[75, 0], [225, 300], [150, 150]]
    assert got["wide"]["k"] == 100 and got["wide"]["left"] == 375  # centred, 1 m = 100 px both ways
    assert got["wideCorner"] == [375, 0]


@needs_node
def test_the_slice_pick_asks_the_rounded_time_within_40_ms_of_the_range():
    kind = {"kind": "inversion", "n_t": 61, "t_range_ms": [2000, 3000]}
    got = _page(
        "[2000, 2010.4, 2008.6, 1965, 1959, 3039, 3041, NaN].map((t) => Dock.pickSlice(input, t))"
        ".concat([Dock.pickSlice({ n_t: 0, t_range_ms: [0, 1] }, 0),"
        " Dock.pickSlice({ n_t: 1, t_range_ms: [2340.4, 2340.4] }, 2350)])",
        kind,
    )
    assert got == [
        {"t_ms": 2000}, {"t_ms": 2010}, {"t_ms": 2009}, {"t_ms": 1965}, None, {"t_ms": 3039},
        None, None, None, {"t_ms": 2350},
    ]


@needs_node
def test_unevenly_spaced_slices_show_the_nearest_one_cached_by_its_own_time():
    """Review M2: EFIT times are not evenly spaced. The dock asks the server at the rounded
    cursor time and caches each answer by the t_ms it carries; a time between one asked and the
    slice answered is known without asking."""
    setup = r"""
var S = { event: "detachment", shot: 190001, ticket: 3 };
var calls = [];
var TIMES = [2000, 2003.5, 2010, 2100, 2101, 2240.25, 2500];
function nearest(t) {
  let best = 0;
  TIMES.forEach((time, i) => { if (Math.abs(time - t) < Math.abs(TIMES[best] - t)) best = i; });
  return Math.abs(TIMES[best] - t) <= 40 ? TIMES[best] : null;
}
async function api(path) {
  const t = Number(/[?&]t_ms=([^&]*)/.exec(path)[1]);
  calls.push(t);
  const found = nearest(t);
  if (found == null) {
    const error = new Error("no equilibrium slice within 40 ms");
    error.status = 404;
    throw error;
  }
  return { json: async () => ({ t_ms: found, source: "EFIT02" }) };
}
"""
    got = _page(
        "(async () => {"
        " Dock.redraw = () => {};"
        " const kind = { kind: 'equilibrium', n_t: TIMES.length, t_range_ms: [2000, 2500] };"
        " const cursor = [2001.6, 2004, 2006.4, 2009, 2055.4, 2056, 2099, 2101.2, 2170.6, 2171, 2230,"
        "   2300, 2380, 2002, 2003, 2050, 2100.4, 2238];"
        " const out = [];"
        " for (const t of cursor) {"
        "   const pick = Dock.pickSlice(kind, t);"
        "   await Dock.askSlice('equilibrium', pick);"
        "   const item = Dock.itemAt('equilibrium', pick.t_ms, (found) => found.eq);"
        "   out.push([t, item ? item.eq.t_ms : null, nearest(Math.round(t))]);"
        " }"
        " const even = cursor.map((t) => { const step = (2500 - 2000) / (TIMES.length - 1);"
        "   return 2000 + Math.round((t - 2000) / step) * step; });"
        " return { out, calls, even, cached: [...Dock.sliceCache('equilibrium').items.keys()].sort((a, b) => a - b) };"
        "})()",
        setup=setup,
    )
    shown = [item for _, item, _ in got["out"]]
    assert shown == [want for _, _, want in got["out"]]  # the nearest slice, or none within 40 ms
    assert shown == [2003.5, 2003.5, 2003.5, 2010, None, None, 2100, 2101, None, None, 2240.25, None,
                     None, 2003.5, 2003.5, 2010, 2100, 2240.25]
    # The even-spacing guess of review M2 would show another slice at every one of these times.
    assert all(abs(even - item) > 1 for even, item in zip(got["even"], shown) if item is not None)
    # One request per time not yet known: 2171 and 2002 again, 2003 (between 2002 and its answer
    # 2003.5), 2100 (between 2099 and 2100) and 2238 (between 2230 and 2240.25) ask nothing.
    assert got["calls"] == [2002, 2004, 2006, 2009, 2055, 2056, 2099, 2101, 2171, 2230, 2300, 2380, 2050]
    assert got["cached"] == [2003.5, 2010, 2100, 2101, 2240.25]


@needs_node
def test_q_marks_and_shared_profile_ranges():
    got = _page(
        "({ marks: Dock.qMarks(0.8, 2.5).map(([q]) => q), signed: Dock.qMarks(-3, -0.9, -1).map(([q]) => q),"
        "  text: Dock.qMarks(1.2, 1.6)[0][1],"
        "  union: Dock.union([['psi_n', [0, 1]], ['rho', [0, 0.9]], ['psi_n', [-0.1, 0.8]]]) })"
    )
    assert got == {"marks": [1, 1.5, 2], "signed": [-1, -1.5, -2], "text": "q = 3/2",
                   "union": {"psi_n": [-0.1, 1], "rho": [0, 0.9]}}


@needs_node
def test_a_profile_note_names_the_diagnostic_not_the_radius_map():
    got = _page(
        "[{ label: 'q (EFIT02)', source: 'park' }, { label: 'Thomson core Te (R from EFIT02)', source: 'corpus' },"
        " { label: 'ECE Te (R from EFIT01)', source: 'corpus' }, { label: 'Rotation (ZIPFIT)', source: 'park' },"
        " { label: 'q', source: 'EFIT01' }, { label: 'Density', source: 'corpus' }]"
        ".map((profile) => Dock.sliceNote(profile, { i: 3, t: 1900 }, null, null, 1800))"
        ".concat([Dock.sliceNote({ label: 'q (EFIT02)' }, { i: 3, t: 1900 }, { i: 3 }, { i: 3 }, 1900)])",
        setup="var S = { video: null };",
    )
    assert got == ["EFIT 1,900 ms, 100 ms from cursor", "TS 1,900 ms, 100 ms from cursor",
                   "ECE 1,900 ms, 100 ms from cursor", "ZIPFIT 1,900 ms, 100 ms from cursor",
                   "EFIT 1,900 ms, 100 ms from cursor", "slice 1,900 ms, 100 ms from cursor",
                   "A and B share one EFIT slice"]


# -- views.js


@needs_node
def test_default_layouts():
    got = _page("['detachment', 'tearing_mode'].map(defaultLayout)")
    detachment, other = got
    assert detachment["dock"]["open"] is True
    assert detachment["dock"]["panels"] == [{"kind": "slice", "id": "inversion", "h": 300},
                                            {"kind": "cameras"}]
    assert detachment["rows"] == [] and other["rows"] == []
    assert other["dock"] == {"open": False, "width": None, "follow": "cursor", "panels": []}
    assert {detachment["v"], other["other_store_rows"]} == {1, "end"}


@needs_node
def test_store_keys_and_a_layout_over_the_store():
    got = _page(
        "(() => {"
        " const keys = Views.storeKeys(['D-alpha', 'ECE', 'D-alpha']);"
        " const layout = { rows: [{ key: 'store:ECE', h: 220 }, { key: 'cat:dusb' },"
        "   { key: 'store:gone' }, { key: 'store:D-alpha', hidden: true }, { key: 'junk' },"
        "   { key: 'store:ECE' }, { key: 'mds:EFIT01:\\\\EFIT01::BETAN', draw: 'map' }] };"
        " const plan = Views.arrange(layout, keys);"
        " const plain = Views.arrange({ rows: [] }, keys);"
        " return { keys, plan, notes: Views.notes(plan), plain: plain.rows.map((row) => row.i),"
        "   quiet: Views.notes(plain), many: Views.notes({ missing: ['a', 'b'], fresh: ['c', 'd', 'e'] }),"
        "   ids: ['cat:ip', 'custom:DUSB', 'mds:EFIT01:\\\\EFIT01::Q95', 'store:x', 'cat:', 7].map(Views.idOf) };"
        "})()"
    )
    assert got["keys"] == ["store:D-alpha", "store:ECE", "store:D-alpha#2"]
    rows = got["plan"]["rows"]
    assert [(row["key"], row["i"]) for row in rows] == [
        ("store:ECE", 1), ("cat:dusb", None), ("store:D-alpha", 0),
        ("mds:EFIT01:\\EFIT01::BETAN", None), ("store:D-alpha#2", 2),
    ]
    assert rows[0]["h"] == 220 and rows[2]["hidden"] is True and rows[3]["draw"] == "map"
    assert got["plan"]["missing"] == ["gone"] and got["plan"]["fresh"] == ["D-alpha"]
    assert got["notes"] == {"missing": "1 row of this view is not in this store: gone",
                            "fresh": "1 row is new in this store: D-alpha"}
    assert got["plain"] == [0, 1, 2]  # Default: store order, no notes
    assert got["quiet"] == {"missing": None, "fresh": None}
    assert got["many"] == {"missing": "2 rows of this view are not in this store: a, b",
                           "fresh": "3 rows are new in this store: c, d, e"}
    assert got["ids"] == ["ip", "custom:DUSB", "mds:EFIT01:\\EFIT01::Q95", None, None, None]


@needs_node
def test_layout_rows_give_sizes_rows_its_heights():
    setup = r"""
var S = { meta: { rows: [{ title: "a" }, { title: "b" }] } };
var canvases = [{}, {}, {}, {}];
var $ = (id) => (id === "rows" ? { children: canvases } : null);
var document = {};
"""
    got = _page(
        "(() => {"
        " Views.storeSpec = [{ key: 'store:a', h: 220 }, { key: 'store:b', hidden: true }];"
        " Views.added = [{ canvas: canvases[2], h: 150, entry: { state: 'error', error: { kind: 'missing' } } },"
        "   { canvas: canvases[3], h: null, hidden: true, entry: null }];"
        " return layoutRows();"
        "})()",
        setup=setup,
    )
    assert got == [
        {"h": 220, "hidden": False, "fixed": False},
        {"h": None, "hidden": True, "fixed": False},
        {"h": 28, "hidden": False, "fixed": True},
        {"h": 110, "hidden": True, "fixed": False},
    ]


ENTRIES = [
    {"id": "ip", "label": "Plasma current", "group": "Global", "units": "MA",
     "suggested": False, "source": "corpus", "where": "ip"},
    {"id": "ece_te", "label": "ECE Te", "group": "Kinetic", "units": "keV",
     "suggested": True, "source": "raw", "where": "tecef"},
    {"id": "betan", "label": "beta_N", "group": "Global", "units": "",
     "suggested": False, "source": "mds", "where": "\\EFIT01::BETAN"},
]


def _match(query, source="ptdata"):
    return _page(
        "matchCatalog(input.entries, input.query, input.source, input.trees)",
        {"entries": ENTRIES, "query": query, "source": source, "trees": list(catalog.TREES)},
    )


@needs_node
def test_the_picker_searches_every_word_suggested_first():
    assert [m["id"] for m in _match("")["matches"]] == ["ece_te", "ip", "betan"]
    assert [m["id"] for m in _match("plasma CURRENT")["matches"]] == ["ip"]
    assert [m["id"] for m in _match("kev")["matches"]] == ["ece_te"]
    assert [m["id"] for m in _match("efit01::betan")["matches"]] == ["betan"]
    assert [m["id"] for m in _match("global")["matches"]] == ["ip", "betan"]
    found = _match("plasma")
    assert found["offer"] is None and found["refusal"] is None


@needs_node
def test_the_picker_offers_or_refuses_a_ptdata_point():
    assert _match("dusbradial") == {"matches": [], "offer": {
        "id": "custom:DUSBRADIAL", "text": "Add PTDATA DUSBRADIAL"}, "refusal": None}
    for name in ("1abc", "ABCDEFGHIJKLM", "dim_of(x)", "a-b"):
        got = _match(name)
        assert got["offer"] is None
        assert got["refusal"] == (f"{name} is not a PTDATA point name: use a letter, then letters,"
                                  " digits or _, up to 12 characters.")
    assert _match("no such words") == {"matches": [], "offer": None, "refusal": None}


@needs_node
def test_the_picker_prefixes_and_checks_an_mds_node():
    got = _match("q95", "efit01")
    assert got["offer"] == {"id": "mds:EFIT01:\\EFIT01::q95", "text": "Add MDS \\EFIT01::q95"}
    assert _match("\\EFIT01::TOP.RESULTS:BETAP", "EFIT01")["offer"]["id"] == (
        "mds:EFIT01:\\EFIT01::TOP.RESULTS:BETAP")
    assert "is in EFIT02, not EFIT01" in _match("\\EFIT02::BETAN", "EFIT01")["refusal"]
    assert "DIM_OF and PTDATA2" in _match("EFIT01::DIM_OF_X", "EFIT01")["refusal"]
    assert "not an allowed MDS tree" in _match("XYZQ", "NOTATREE")["refusal"]
    generic = ("is not an MDS node name: use letters, digits and _ in parts that start with a"
               " letter, up to 96 characters.")
    for node in ("dim_of(q95)", "A..B", "_X", "A" * 100):
        assert _match(node, "MHD")["refusal"].endswith(generic), node


@needs_node
@pytest.mark.parametrize(
    ("tree", "node"),
    [
        ("EFIT01", "\\EFIT01::BETAN"), ("efit02", "\\EFIT02::TOP.RESULTS.AEQDSK:Q95"),
        ("MHD", "\\MHD::B_DOT"), ("EFIT01", "\\EFIT02::BETAN"), ("EFIT01", "\\EFIT01::DIM_OF"),
        ("EFIT01", "\\EFIT01::PTDATA2X"), ("EFIT01", "\\EFIT01::A..B"), ("EFIT01", "\\EFIT01::1A"),
        ("EFIT01", "\\EFIT01::A$B"), ("EFIT01", "\\EFIT01::" + "A" * 90), ("BOGUS", "\\BOGUS::A"),
        ("ECE", "\\ECE::TECEF01"), ("EFIT01", "\\EFIT01::A:B:C"), ("EFIT01", "\\EFIT01::_A"),
    ],
)
def test_the_pickers_node_rule_agrees_with_the_servers(tree, node):
    """Never send what the server refuses, and refuse nothing it takes."""
    got = _page(
        "Views.nodeProblem(input.tree.toUpperCase(), input.node, input.trees)",
        {"tree": tree, "node": node, "trees": list(catalog.TREES)},
    )
    assert (got is None) == (catalog.node_problem(tree, node) is None), got


@needs_node
@pytest.mark.parametrize("name", ["DUSBRADIAL", "a", "A_1", "ABCDEFGHIJKL", "ABCDEFGHIJKLM", "1A",
                                  "_A", "A-B", "A B", ""])
def test_the_pickers_ptdata_rule_agrees_with_the_servers(name):
    got = _page("Views.PTDATA.test(input)", name)
    assert got == (catalog.ptdata_problem(name) is None)


# -- a page flow: Copy values


@needs_node
def test_copy_values_writes_every_row_and_channel_at_a_and_b():
    setup = r"""
var copied = null;
var navigator = { clipboard: { writeText: async (text) => { copied = text; } } };
var S = { meta: { rows: [
    { title: "Ip", kind: "trace", n_channels: 1, y_units: "MA", legend: [] },
    { title: "Spec", kind: "image", y_units: "kHz", y0: 0, dy: 10, n_y: 3 },
  ] },
  data: { t0: 0, t1: 100, n: 4, rows: [
    new Float32Array([1, 2, 3, 4, 1, 2, 3, 4]),
    new Uint8Array([0, 0, 0, 0,  9, 0, 0, 0,  0, 0, 0, 7]),
  ] } };
var traceRange = () => [0, 10];
var bandBins = (row) => [0, row.n_y];
var canvas = () => ({ style: { order: "" }, classList: { contains: () => false } });
var rows = [canvas(), canvas(), canvas()];
var $ = (id) => (id === "rows" ? { children: rows } : null);
"""
    got = _page(
        "(async () => {"
        " Slice.pins = { A: 10, B: 90 };"
        " Views.added = [{ key: 'custom:DUSB', title: 'DUSB', units: 'V', canvas: rows[2], draw: 'lines',"
        "   entry: { data: { t0: 0, t1: 100, n: 2, channels: 2, values: new Float32Array([1, 2, 3, 4, 1, 2, 3, 4]) },"
        "     signal: { legend: ['in', 'out'], units: 'V' } } }];"
        " await copyValues();"
        " return copied;"
        "})()",
        setup=setup,
    )
    assert got.splitlines() == [
        "row\tunits\tA\tB\tB−A",
        "t (ms)\tms\t10\t90\t80",
        "Ip\tMA\t1\t4\t3",
        "Spec\tkHz\t10\t20\t10",
        "DUSB / in\tV\t1\t2\t1",
        "DUSB / out\tV\t3\t4\t1",
    ]
