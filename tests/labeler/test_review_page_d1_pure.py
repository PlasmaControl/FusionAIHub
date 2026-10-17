"""The review page's status, shot-list and TokEye scripts (plan D.4, D.5, D.7), run under node.

status.js, lists.js and overlays.js are classic scripts loaded before app.js: they share its
globals, so each defines only the names D.7 gives it and does nothing at load. Their pure
parts run here in one node context; the rest is exercised against stubs of app.js's helpers.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from labeler.events.ui.app import STATIC

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node runs the page's scripts")

#: The page's new scripts, in load order, and the globals plan D.7 gives each.
GLOBALS = {
    "status.js": {"Activity", "ERROR_COPY", "rowNote", "renderActivity", "stopWaiting"},
    "lists.js": {"Lists", "fillLists", "chooseList", "toggleByShot", "openByShot", "byShotKey",
                 "listAddress"},
    "overlays.js": {"Overlays", "overlayNote", "loadOverlays", "requestOverlay",
                    "pollOverlays"},
}
#: D2's scripts, loaded after D1's when they exist.
LATER = ("signals.js", "slice.js", "dock.js", "views.js")
TOP_LEVEL = re.compile(
    r"^(?:async\s+)?(?:function\*?|const|let|var|class)\s+([A-Za-z_$][\w$]*)", re.MULTILINE
)

#: Stubs of what app.js gives the scripts, for the flows below: requests are recorded in
#: `calls` and answered from `answers` (path prefix -> {status, body}); timers wait in
#: `timers` until a test runs them.
STUBS = r"""
var calls = [], timers = [], drawn = 0, answers = {};
var MASK_EVENT = "alfven_eigenmode";
var S = { api: 11, ticket: 1, event: "tearing_mode", shot: 189081, meta: null, tokeye: null,
  showMasks: true };
var render = () => { drawn++; };
var showMasks = () => {};
var setTimeout = (run, delay) => { timers.push({ run, delay }); return timers.length; };
var clearTimeout = () => {};
async function api(path, options = {}) {
  calls.push({ path, method: options.method || "GET", quiet: Boolean(options.quiet),
    body: options.body ? JSON.parse(options.body) : null });
  const key = Object.keys(answers).find((prefix) => path.startsWith(prefix));
  const answer = answers[key] || { status: 404, body: { error: "not found" } };
  if (answer.status >= 400) {
    const error = new Error(answer.body.error || String(answer.status));
    error.status = answer.status;
    throw error;
  }
  return { status: answer.status, json: async () => answer.body };
}
async function loadTokeye(event, shot) {
  try {
    return await (await api(`/api/tokeye?event=${event}&shot=${shot}`)).json();
  } catch {
    return null;
  }
}
var tick = async () => {
  const due = timers.splice(0);
  for (const timer of due) await timer.run();
};
"""


def _page(expression: str, payload=None, *, setup: str = ""):
    """`expression`, awaited, in one node context holding the three scripts after `setup`;
    `input` is `payload`."""
    script = (
        "const fs = require('fs'), vm = require('vm');"
        "const [dir, files, setup, expression] = JSON.parse(process.argv[1]);"
        "const input = JSON.parse(fs.readFileSync(0, 'utf8'));"
        "const context = vm.createContext({ module: { exports: {} }, URLSearchParams, input });"
        "vm.runInContext(setup, context, { filename: 'setup' });"
        "for (const name of files)"
        "  vm.runInContext(fs.readFileSync(dir + '/' + name, 'utf8'), context, { filename: name });"
        "Promise.resolve(vm.runInContext(expression, context, { filename: 'test' })).then("
        "  (value) => process.stdout.write(JSON.stringify(value === undefined ? null : value)),"
        "  (error) => { process.stderr.write(String(error && error.stack || error)); process.exit(1); });"
    )
    arguments = json.dumps([str(STATIC), list(GLOBALS), setup, expression])
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


def _flow(body: str, payload=None):
    """An async `body` over the app.js stubs; it returns what the test checks."""
    return _page(f"(async () => {{ {body} }})()", payload, setup=STUBS)


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


def test_no_two_page_scripts_share_a_top_level_name():
    """Classic scripts share one global scope: a repeated function silently replaces the
    other, a repeated const stops the page."""
    owners: dict[str, list[str]] = {}
    for path in sorted(STATIC.glob("*.js")):
        for name in _top_level(path.name):
            owners.setdefault(name, []).append(path.name)
    assert not {name: files for name, files in owners.items() if len(files) > 1}


@needs_node
def test_the_scripts_load_together_and_do_nothing_at_load():
    """D1's scripts, D2's that exist, then app.js, in one context with no document: any work
    at load (or a repeated declaration) would throw."""
    files = [*GLOBALS, *(name for name in LATER if (STATIC / name).exists()), "app.js"]
    script = (
        "const fs = require('fs'), vm = require('vm');"
        "const [dir, files] = JSON.parse(process.argv[1]);"
        "const context = vm.createContext({ module: { exports: {} } });"
        "for (const name of files)"
        "  vm.runInContext(fs.readFileSync(dir + '/' + name, 'utf8'), context, { filename: name });"
        "process.stdout.write(JSON.stringify(vm.runInContext("
        "  '[typeof Activity, typeof Lists, typeof Overlays, typeof boot]', context)));"
    )
    result = subprocess.run(
        [NODE, "-e", script, json.dumps([str(STATIC), files])],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["object", "object", "object", "function"]


@needs_node
@pytest.mark.parametrize("name", list(GLOBALS))
def test_each_script_exports_its_globals_for_node(name):
    script = "process.stdout.write(JSON.stringify(Object.keys(require(process.argv[1]))))"
    result = subprocess.run(
        [NODE, "-e", script, str(STATIC / name)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    assert set(json.loads(result.stdout)) == GLOBALS[name]


# -- status.js


@needs_node
def test_requests_are_named_as_the_bar_says_them():
    found = _page("""[
        Activity.describe("/api/rows?event=elm&shot=1"),
        Activity.describe("/api/queue?event=elm"),
        Activity.describe("/api/shot?event=elm&shot=189081"),
        Activity.describe("/api/overview?list=silver"),
        Activity.describe("/api/overview?list=silver&shot=170899"),
        Activity.describe("/api/masks", "POST"),
        Activity.describe("/api/masks?event=alfven_eigenmode&shot=1"),
        Activity.describe("/api/overlays/request", "POST"),
        Activity.describe("/api/signal?shot=1&name=ip"),
        Activity.describe("/api/somewhere"),
    ]""")
    assert found == [
        "Loading rows", "Loading the queue", "Opening 189081", "Loading the list",
        "Loading the shot overview", "Saving the mask decisions", "Loading the AE mask",
        "Asking for TokEye", "Fetching a signal", "Waiting for the server",
    ]


@needs_node
def test_clock_and_ordinals():
    clocks = _page("[0.8, 0.05, 14.9, 60, 134, 120].map(Activity.clock)")
    assert clocks == ["0.8 s", "0.0 s", "14 s", "1 min", "2 min 14 s", "2 min"]
    ordinals = _page("[1, 2, 3, 4, 11, 12, 13, 21, 22, 111, 112].map(Activity.ordinal)")
    assert ordinals == ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd",
                        "111th", "112th"]


@needs_node
def test_the_activity_line():
    """An item shows after 300 ms and its elapsed time after 2 s; one kind is counted, mixed
    kinds name the longest; quiet items never show; a build's own text wins."""
    found = _page("""(() => {
        const now = 100000;
        const item = (path, age, quiet = false) =>
            ({ kind: path.split("?")[0].slice(5), what: Activity.describe(path), started: now - age, quiet });
        return [
            Activity.line([item("/api/rows?x", 200)], now),
            Activity.line([item("/api/rows?x", 800)], now),
            Activity.line([item("/api/rows?x", 2500)], now),
            Activity.line([item("/api/shot?event=e&shot=1", 5000, true)], now),
            Activity.line([item("/api/signal?a", 9000), item("/api/signal?b", 3000)], now),
            Activity.line([item("/api/rows?x", 1000), item("/api/overview?list=a&shot=1", 14000),
                           item("/api/signal?a", 500)], now),
            Activity.line([item("/api/rows?x", 9000), { ...item("/api/shot?shot=1", 10), text: "Building 1" }], now),
        ];
    })()""")
    assert found == [
        "",
        "Loading rows",
        "Loading rows · 2 s",
        "",
        "Fetching 2 signals · 9 s",
        "3 loading · longest: Loading the shot overview · 14 s",
        "Building 1",
    ]


@needs_node
def test_a_store_build_says_its_stage_queue_and_list():
    found = _page("""[
        Activity.buildText(189081, "Tearing (n=1)", { stage: "fetching", elapsed_s: 14.2, queue_position: 1,
            prebuilt: false, progress: { done: 2, total: 5, stage: "fetching mirnov" } }),
        Activity.buildText(170816, "ELM", { stage: "queued", elapsed_s: 1, queue_position: 3, prebuilt: true }),
        Activity.buildText(170816, "ELM", { stage: "writing", elapsed_s: 75, queue_position: 0 }),
        Activity.buildText(170816, "ELM", { stage: "fetching" }, 120),
        Activity.stopped(189081),
        Activity.buildText(170816, "ELM", { stage: "fetching", fetching: ["pinj_total"],
            note: "Fetching pinj_total", elapsed_s: 4, queue_position: 0 }),
        Activity.buildText(170816, "ELM", { stage: "building", note: "Building the store.",
            elapsed_s: 4, queue_position: 0 }),
        Activity.buildText(170816, "ELM", { stage: "queued", note: "Queued, 2 ahead", queue_position: 3 }),
    ]""")
    assert found == [
        "Building 189081 for Tearing (n=1): fetching mirnov (2 of 5) · 14 s · 1st in line",
        "Building 170816 for ELM: queued · 3rd in line",
        "Building 170816 for ELM: writing the store · 1 min 15 s",
        "Still building after 2 min. The server may be stuck.",
        "189081 is still building on the server. Open it again later.",
        # R2 (contract §2): the 202's short `note` says what the server is doing.
        "Building 170816 for ELM: fetching pinj_total · 4 s",
        "Building 170816 for ELM: building the store · 4 s",
        # A note that gives the queue is not followed by the place in line (review R2 M8).
        "Building 170816 for ELM: queued, 2 ahead",
    ]


@needs_node
def test_a_build_without_a_stage_change_for_two_minutes_may_be_stuck():
    found = _page("""(() => {
        let clock = 0, stopped = null;
        Activity.now = () => clock;
        const body = { stage: "fetching", elapsed_s: 3, queue_position: 0, progress: { done: 2, total: 5, stage: "fetching mirnov" } };
        const id = Activity.build(null, { shot: 1, name: "ELM", body, stop: () => { stopped = 1; } });
        const first = Activity.items.get(id).text, fraction = Activity.items.get(id).fraction;
        clock = 121000;
        Activity.build(id, { shot: 1, name: "ELM", body });
        const stuck = Activity.items.get(id).text;
        Activity.build(id, { shot: 1, name: "ELM", body: { ...body, progress: { ...body.progress, done: 3 } } });
        const moved = Activity.items.get(id).text;
        const offered = Activity.summary(clock).stop;
        const ended = stopWaiting();
        return [first, fraction, stuck, moved, offered, ended, stopped, Activity.items.size, stopWaiting()];
    })()""")
    assert found == [
        "Building 1 for ELM: fetching mirnov (2 of 5) · 3 s",
        0.4,
        "Still building after 2 min. The server may be stuck.",
        "Building 1 for ELM: fetching mirnov (3 of 5) · 3 s",
        True,
        True,
        1,
        0,
        False,
    ]


@needs_node
def test_failures_are_kept_but_an_absent_mask_is_not_one():
    found = _page("""(() => {
        const fail = (path, status) => {
            const id = Activity.start(path);
            Activity.end(id, Object.assign(new Error(`${status} no`), { status }));
        };
        fail("/api/masks?event=alfven_eigenmode&shot=1", 404);
        fail("/api/tokeye?event=alfven_eigenmode&shot=1", 404);
        fail("/api/masks?event=alfven_eigenmode&shot=1", 500);
        for (let i = 0; i < 12; i++) fail("/api/rows?event=elm&shot=1", 502);
        Activity.end(Activity.start("/api/rows?event=elm&shot=2"));
        return [Activity.errors.length, Activity.errors[0].what, Activity.errors.at(-1).what, Activity.items.size];
    })()""")
    assert found == [10, "Loading rows", "Loading rows", 0]


@needs_node
def test_notes_show_while_nothing_loads_and_an_error_note_first():
    found = _page("""(() => {
        let clock = 1000;
        Activity.now = () => clock;
        Activity.note("prefetch", ERROR_COPY.prefetch({ shot: 170816, message: "500 boom" }), "error");
        clock = 2000;
        Activity.note("fdp", "fdp login expires in 5 h");
        const idle = Activity.summary(clock);
        const id = Activity.start("/api/rows?event=elm&shot=1");
        clock = 2500;
        const busy = Activity.summary(clock);
        Activity.end(id);
        Activity.clear("prefetch");
        const after = Activity.summary(clock);
        Activity.clear("fdp");
        return [idle, busy.text, after.text, Activity.summary(clock), Activity.errors[0].message];
    })()""")
    assert found[0] == {"text": "Next shot 170816 failed to build: 500 boom", "tone": "error",
                        "busy": False, "stop": False, "fraction": None}
    assert found[1:] == ["Loading rows", "fdp login expires in 5 h", None,
                         "Next shot 170816 failed to build: 500 boom"]


@needs_node
def test_the_fdp_note():
    found = _page("""(() => {
        const now = Date.parse("2026-10-16T12:00:00Z");
        const at = (hours) => new Date(now + hours * 3600e3).toISOString();
        return [
            Activity.fdpNote({ env: false, expires_at: null }, now),
            Activity.fdpNote({ env: true, expires_at: at(-1) }, now),
            Activity.fdpNote({ env: true, expires_at: at(5.5) }, now),
            Activity.fdpNote({ env: true, expires_at: at(0.5) }, now),
            Activity.fdpNote({ env: true, expires_at: at(48) }, now),
            Activity.fdpNote({ env: true, expires_at: null }, now),
            Activity.fdpNote(undefined, now),
        ];
    })()""")
    assert found == [
        {"text": "This server cannot fetch new signals. Restart it with pixi run -e labeler labeler-verify.", "tone": ""},
        {"text": "fdp login expired: no new signals. Run fdp login on the server, then Retry.",
         "tone": "error"},
        {"text": "fdp login expires in 5 h", "tone": ""},
        {"text": "fdp login expires in 30 min", "tone": ""},
        None, None, None,
    ]


#: /api/fetch/status's `reason` while LABELER_NO_FETCH is set (fetch_worker.off_reason).
OFF = ("fetching is off on this server (LABELER_NO_FETCH is set): only signals on disk can "
       "be shown")


@needs_node
def test_the_fetch_status_reason_wins_over_the_generic_copy():
    """B's /api/fetch/status says `fdp: "off"` with the 503's text in `reason` when nothing can
    be fetched; the bar shows that reason for "off" and for any other state that has one."""
    found = _page(f"""[
        Activity.fetchNote({{ fdp: "ok", queue: 0, running: null }}),
        Activity.fetchNote({{ fdp: "off", reason: {json.dumps(OFF)} }}),
        Activity.fetchNote({{ fdp: "expired", reason: "the fdp token has expired: run 'fdp login', then try again" }}),
        Activity.fetchNote({{ fdp: "unavailable", reason: "this server was not started under the fdp wrapper" }}),
        Activity.fetchNote({{ fdp: "expired" }}),
        Activity.fetchNote({{ fdp: "unavailable" }}),
        Activity.fetchNote({{ fdp: "off" }}),
        Activity.fetchNote(null),
    ]""")
    assert found == [
        None,
        {"text": "Fetching is off on this server (LABELER_NO_FETCH is set): only signals on "
                 "disk can be shown.", "tone": ""},
        {"text": "The fdp token has expired: run 'fdp login', then try again.",
         "tone": "error"},
        {"text": "This server was not started under the fdp wrapper.", "tone": ""},
        {"text": "fdp login expired: no new signals. Run fdp login on the server, then Retry.",
         "tone": "error"},
        {"text": "This server cannot fetch new signals. Restart it with pixi run -e labeler labeler-verify.", "tone": ""},
        {"text": "This server cannot fetch new signals. Restart it with pixi run -e labeler labeler-verify.", "tone": ""},
        None,
    ]


@needs_node
def test_retry_checks_the_fetch_status_and_shows_its_reason():
    found = _flow(f"""
        Activity.enabled = true;
        answers["/api/fetch/status"] = {{ status: 200, body: {{ fdp: "off", reason: {json.dumps(OFF)},
          queue: 0, running: null }} }};
        const off = await Activity.fetchReady();
        const said = Activity.summary(0);
        answers["/api/fetch/status"] = {{ status: 200, body: {{ fdp: "ok", queue: 0, running: null }} }};
        const ok = await Activity.fetchReady();
        return [off, said.text, said.tone, ok, Activity.notes.has("fdp"), calls.map((c) => c.path)];
    """)
    assert found == [
        False,
        ("Fetching is off on this server (LABELER_NO_FETCH is set): only signals on disk "
         "can be shown."),
        "",
        True,
        False,
        ["/api/fetch/status", "/api/fetch/status"],
    ]


@needs_node
def test_stale_server_and_version_copy():
    found = _page("""[
        [1, 4, 6, 10, 11].map(Activity.staleText),
        Activity.versionNote({ status: 404, message: "404 Not Found" }),
        Activity.versionNote({ status: 500, message: "500 Internal Server Error" }),
        Activity.versionNote(new TypeError("Failed to fetch")),
    ]""")
    assert found == [
        ["Restart the server for names and history",
         "Restart the server for the list of names",
         "Restart the server for individual/span annotations",
         "Restart the server for shot lists, TokEye status and views",
         None],
        None,
        "Could not read the server version (500 Internal Server Error). Reload the page.",
        "Could not read the server version (Failed to fetch). Reload the page.",
    ]


@needs_node
def test_row_chips_for_added_rows():
    found = _page("""[
        rowNote("queued", { queue_position: 2, elapsed_s: 4 }),
        rowNote("loading", { source: "ptdata", elapsed_s: 9 }),
        rowNote("loading", { source: "mds", tree: "EFIT01", elapsed_s: 12 }),
        rowNote("loading", { source: "disk", elapsed_s: 0.5 }),
        rowNote("auth", {}),
        rowNote("missing", { name: "DUSBRADIAL", shot: 180625 }),
        rowNote("upstream", { message: "MDSplus said no" }),
        rowNote("timeout", {}),
        rowNote("invalid", { name: "DUSB-RADIAL" }),
        rowNote("invalid", { name: "X", message: "the formula does not parse" }),
        rowNote("teapot", {}),
        ERROR_COPY.mask({ message: "500 Internal Server Error" }),
        ERROR_COPY.timeout({ limit_s: 120 }),
    ]""")
    assert found == [
        {"text": "Queued, 2nd in line · 4 s", "button": "Stop waiting", "tone": "queued"},
        {"text": "Fetching from PTDATA · 9 s", "button": "Stop waiting", "tone": "queued"},
        {"text": "Fetching from MDS (EFIT01) · 12 s", "button": "Stop waiting", "tone": "queued"},
        {"text": "Reading from disk", "button": "Stop waiting", "tone": "queued"},
        {"text": "fdp login expired. Run fdp login on the server, then Retry.",
         "button": "Retry", "tone": "error"},
        {"text": "No DUSBRADIAL for 180625.", "button": "Remove from view", "tone": "",
         "collapse": True},
        {"text": "The data server failed: MDSplus said no. Try again soon.",
         "button": "Retry", "tone": "error"},
        {"text": "Stopped after 30 s. A formula node can take 2 min.",
         "button": "Retry", "tone": "error"},
        {"text": "DUSB-RADIAL is not a valid PTDATA name: a letter, then letters, digits or _, up to 12.",
         "button": "Edit", "tone": "error"},
        {"text": "The formula does not parse.", "button": "Edit", "tone": "error"},
        None,
        "Could not load the AE mask. Press M to retry.",
        "Stopped after 2 min. Try again soon.",
    ]


# -- lists.js

LISTS = [
    {"name": "ae_co2_ece", "title": "AE CO2 + ECE", "n": 193, "kind": "derived",
     "recommended": True, "default": True, "parent": None, "n_saved": 40},
    {"name": "silver", "title": "Silver set", "n": 450, "kind": "curated",
     "recommended": True, "default": False, "parent": None, "n_saved": 12},
    {"name": "silver_batch_1", "title": "Silver batch 1", "n": 90, "kind": "derived",
     "recommended": False, "default": False, "parent": "silver", "n_saved": 3},
    {"name": "silver_batch_2", "title": "Silver batch 2", "n": 90, "kind": "derived",
     "recommended": False, "default": False, "parent": "silver", "n_saved": 0},
    {"name": "elm", "title": "ELM labels", "n": 576, "kind": "dataset",
     "recommended": False, "default": False, "parent": None, "n_saved": 120},
    {"name": "elm_fast", "title": "Fast ELMs", "n": 40, "kind": "derived",
     "recommended": False, "default": False, "parent": "elm", "n_saved": 1},
    {"name": "elm_fast_big", "title": "Big fast ELMs", "n": 9, "kind": "derived",
     "recommended": False, "default": False, "parent": "elm_fast", "n_saved": 0},
    {"name": "roster", "title": "Event roster", "n": 193, "kind": "roster",
     "recommended": False, "default": False, "parent": None, "n_saved": 40},
]


@needs_node
def test_the_list_dropdown_groups_and_options():
    found = _page('Lists.groups(input, "Alfvén eigenmode")', LISTS)
    em = " "
    assert found == [
        {"label": "Recommended for Alfvén eigenmode", "options": [
            {"value": "ae_co2_ece", "text": "ae_co2_ece · 193 · 40 saved", "title": "AE CO2 + ECE"},
            {"value": "silver", "text": "silver · 450 · 12 saved", "title": "Silver set"},
        ]},
        {"label": "Other lists", "options": [
            {"value": "silver_batch_1", "text": "silver_batch_1 · 90 · 3 saved",
             "title": "Silver batch 1"},
            {"value": "silver_batch_2", "text": "silver_batch_2 · 90 · 0 saved",
             "title": "Silver batch 2"},
            {"value": "elm", "text": "elm · 576 · 120 saved", "title": "ELM labels"},
            {"value": "elm_fast", "text": f"{em}elm_fast · 40 · 1 saved", "title": "Fast ELMs"},
            {"value": "elm_fast_big", "text": f"{em}{em}elm_fast_big · 9 · 0 saved",
             "title": "Big fast ELMs"},
        ]},
        {"label": "Event roster", "options": [
            {"value": "roster", "text": "Event roster · 193 · 40 saved", "title": "Event roster"},
        ]},
    ]
    assert _page('Lists.groups(input.filter((e) => e.kind === "roster"), "ELM").length', LISTS) == 1


@needs_node
def test_the_closed_dropdown_says_whether_the_list_is_recommended():
    found = _page("""[
        Lists.mark(input[0]), Lists.mark(input[4]),
        Lists.mark({ ...input[7], default: true }), Lists.mark(undefined),
    ]""", LISTS)
    assert found == [
        {"recommended": True, "title": "AE CO2 + ECE"},
        {"recommended": False, "title": "Not recommended for this event"},
        {"recommended": False, "title": "Event roster"},
        {"recommended": False, "title": ""},
    ]


@needs_node
def test_queue_urls_and_list_addresses():
    found = _page("""[
        Lists.queueUrl("alfven_eigenmode", null),
        Lists.queueUrl("alfven_eigenmode", "silver"),
        Lists.queueUrl("a b", "c&d"),
        Lists.storageKey("detachment"),
        listAddress("silver"),
        listAddress("silver_batch_2", 170899),
        listAddress("silver", 170899, "edge_localized_mode"),
        Lists.queueKey("elm", "silver"),
    ]""")
    assert found == [
        "/api/queue?event=alfven_eigenmode",
        "/api/queue?event=alfven_eigenmode&list=silver",
        "/api/queue?event=a%20b&list=c%26d",
        "labeler:list:detachment",
        "#list/silver",
        "#list/silver_batch_2/170899",
        "#list/silver/170899/edge_localized_mode",
        "elm/silver",
    ]


@needs_node
def test_list_and_silver_addresses_are_read():
    found = _page("""[
        "#list/silver/170899/edge_localized_mode", "#list/gold", "#list/gold/x",
        "#silver/2/170899/detachment", "#silver/", "#silver/x", "#silver",
        "#detachment/170899", "#list/", "",
    ].map(Lists.parseAddress)""")
    assert found == [
        {"list": "silver", "shot": 170899, "event": "edge_localized_mode"},
        {"list": "gold", "shot": None, "event": None},
        {"list": "gold", "shot": None, "event": None},
        {"list": "silver_batch_2", "shot": 170899, "event": "detachment"},
        {"list": "silver", "shot": None, "event": None},
        {"list": "silver", "shot": None, "event": None},
        {"list": "silver", "shot": None, "event": None},
        None, None, None,
    ]


@needs_node
def test_the_gold_box_migrates_once_to_the_gold_list():
    found = _page("""(() => {
        const kept = new Map(Object.entries(input));
        const read = (key) => kept.get(key) ?? null;
        const write = (key, value) => (value == null ? kept.delete(key) : kept.set(key, value));
        const events = [{ event: "alfven_eigenmode", subcategories: [{ event: "alfven_eigenmode.tae" }] },
                        { event: "detachment" }];
        const first = Lists.migrate(read, write, events);
        const second = Lists.migrate(read, write, events);
        return [first, second, Object.fromEntries(kept)];
    })()""", {"labeler:gold-only": "1", "labeler:list:detachment": "silver"})
    assert found == [2, 0, {"labeler:list:detachment": "silver",
                            "labeler:list:alfven_eigenmode": "gold",
                            "labeler:list:alfven_eigenmode.tae": "gold"}]


@needs_node
def test_newer_saves_merge_into_the_queue_and_keep_the_extra_mark():
    found = _page("""Lists.merge(
        [{ shot: 1, state: "unreviewed" }, { shot: 2, state: "unreviewed", extra: true }, { shot: 3, state: "unreviewed" }],
        new Map([[1, { count: 5, row: { shot: 1, state: "confirmed" } }],
                 [2, { count: 5, row: { shot: 2, state: "confirmed" } }],
                 [3, { count: 2, row: { shot: 3, state: "confirmed" } }]]),
        2)""")
    assert found == [{"shot": 1, "state": "confirmed"},
                     {"shot": 2, "state": "confirmed", "extra": True},
                     {"shot": 3, "state": "unreviewed"}]


@needs_node
def test_a_lists_family_becomes_the_batch_buttons():
    found = _page("""(() => {
        const child = { list: "silver_batch_1", title: "Silver batch 1", children: [],
                        shots: [{ shot: 1, done: true }, { shot: 2, done: false }] };
        const root = { list: "silver", title: "Silver set", shots: [{ shot: 1, done: true }],
                       children: [{ name: "silver_batch_1", title: "Silver batch 1", n: 90 }] };
        const lone = { list: "elm", title: "ELM labels", shots: [{ shot: 5, done: false }], children: [] };
        return [Lists.family("silver_batch_1", child, input), Lists.family("silver", root, []),
                Lists.family("elm", lone, input), Lists.family("elm", null, input)];
    })()""", LISTS)
    assert found == [
        [{"name": "silver", "text": "Silver set · 450", "pressed": False},
         {"name": "silver_batch_1", "text": "Silver batch 1 · 1/2 done", "pressed": True},
         {"name": "silver_batch_2", "text": "Silver batch 2 · 90", "pressed": False}],
        [{"name": "silver", "text": "Silver set · 1/1 done", "pressed": True},
         {"name": "silver_batch_1", "text": "Silver batch 1 · 90", "pressed": False}],
        [{"name": "elm", "text": "ELM labels · 0/1 done", "pressed": True}],
        [{"name": "elm", "text": "ELM labels · 576", "pressed": True}],
    ]


@needs_node
def test_by_shot_steps_and_the_next_event():
    found = _page("""(() => {
        const shots = [{ shot: 1, done: true }, { shot: 2, done: false }, { shot: 3, done: true }];
        const events = [{ event: "a", saved: true }, { event: "b", locked: true }, { event: "c" }, { event: "d" }];
        return [
            [[2, 1], [3, 1], [1, -1], [null, 1], [null, -1]].map(([at, by]) => Lists.step(shots, at, by)),
            Lists.step([], 1, 1),
            [2, 3, null].map((at) => Lists.nextShot(shots, at)?.shot ?? null),
            ["c", "d", null].map((at) => Lists.nextOnShot(events, at)?.event ?? null),
            Lists.nextOnShot([{ event: "a", saved: true }], "a"),
            Lists.classes("alfven_eigenmode.rsae:present;mhd_n1:present;"),
            Lists.classes(["a:b"]), Lists.classes(null),
            Lists.short("alfven_eigenmode.rsae"), Lists.short("minimum_safety_factor"),
            Lists.timeText({ t_range: [0, 5200], grid_ms: 10 }),
        ];
    })()""")
    assert found == [
        [3, 1, 3, 1, 3], None, [None, 2, 2], ["d", "c", "c"], None,
        ["alfven_eigenmode.rsae:present", "mhd_n1:present"], ["a:b"], [],
        "AE.rsae", "q min", "0–5200 ms · 10 ms frames",
    ]


# -- overlays.js


@needs_node
def test_tokeye_chips_say_each_state():
    found = _page("""(() => {
        const now = 100000;
        return [
            overlayNote({ state: "done" }),
            overlayNote({ state: "done" }, { hidden: true }),
            overlayNote({ state: "stale" }),
            overlayNote({ state: "queued", queue_position: 2 }),
            overlayNote({ state: "queued", queue_position: 0 }),
            overlayNote({ state: "computing", elapsed_s: 40, seen: now }, { now }),
            overlayNote({ state: "computing", elapsed_s: 40, seen: now - 5000 }, { now }),
            overlayNote({ state: "computing", elapsed_s: 1, seen: now }, { now }),
            overlayNote({ state: "failed", reason: "CUDA out of memory while computing the second window" }),
            overlayNote({ state: "failed", reason: "" }),
            overlayNote({ state: "not_computed" }),
            overlayNote({ state: "not_applicable" }),
            overlayNote({ state: "sideways" }),
        ];
    })()""")
    plain = {"tone": "", "title": ""}
    assert found == [
        {"text": "TokEye", "button": None, **plain},
        {"text": "TokEye hidden (M)", "button": None, **plain},
        {"text": "TokEye from an older model", "button": "Recompute", **plain},
        {"text": "TokEye queued, 2nd in line", "button": None, "tone": "queued", "title": ""},
        {"text": "TokEye queued", "button": None, "tone": "queued", "title": ""},
        {"text": "TokEye computing · 40 s", "button": None, "tone": "queued", "title": ""},
        {"text": "TokEye computing · 45 s", "button": None, "tone": "queued", "title": ""},
        {"text": "TokEye computing", "button": None, "tone": "queued", "title": ""},
        {"text": "TokEye failed: CUDA out of memory while computing the…", "button": "Retry",
         "tone": "error",
         "title": "TokEye failed: CUDA out of memory while computing the second window"},
        {"text": "TokEye failed", "button": "Retry", "tone": "error", "title": ""},
        {"text": "No TokEye yet", "button": "Compute", **plain},
        None,
        None,
    ]


LAYER = {"grid": {"t0_ms": 0, "dt_ms": 1, "n": 4}, "n_y": 2, "y0": 10, "dy": 5, "bits": "AA=="}


@needs_node
def test_layers_states_and_titles():
    found = _page("""(() => {
        const meta = { rows: [
            { name: "ece", kind: "image", overlay: { state: "done", reason: "" } },
            { name: "mirnov", kind: "image", overlay: { state: "not_applicable", reason: "a camera" } },
            { name: "ip", kind: "trace", overlay: { state: "not_applicable", reason: "" } },
            { name: "co2", kind: "image" },
        ] };
        Overlays.states = Overlays.statesFrom(meta, 7);
        const layer = Overlays.layer(input);
        return [
            Overlays.states,
            [layer.y0_khz, layer.dy_khz, layer.grid.n],
            Overlays.layer({ ...input, y0_khz: 2, dy_khz: 1 }).y0_khz,
            Overlays.layer({ state: "not_computed", reason: "no layer yet" }),
            Overlays.layer({ ...input, grid: { t0_ms: 0, dt_ms: 1, n: 0 } }),
            Object.keys(Overlays.normalise({ rows: { ece: input, co2: { bits: 1 } } })),
            Overlays.normalise({ state: "failed", reason: "x" }),
            meta.rows.map(Overlays.title),
            Overlays.waiting(), Overlays.waiting({ a: { state: "computing" } }),
        ];
    })()""", LAYER)
    assert found == [
        {"ece": {"state": "done", "reason": "", "seen": 7},
         "mirnov": {"state": "not_applicable", "reason": "a camera", "seen": 7}},
        [10, 5, 4],
        2,
        None,
        None,
        ["ece"],
        {},
        ["", "No TokEye: not a spectrogram", "", ""],
        False, True,
    ]


META = {"rows": [
    {"name": "ece", "kind": "image", "overlay": {"state": "done", "reason": ""}},
    {"name": "co2", "kind": "image", "overlay": {"state": "not_computed", "reason": ""}},
    {"name": "ip", "kind": "trace", "overlay": {"state": "not_applicable", "reason": ""}},
]}


@needs_node
def test_a_default_open_asks_only_for_the_layers_of_done_rows():
    """Another event's done rows bring one /api/tokeye; nothing waits, so nothing polls. AE's
    layer comes with loadMasks; a shot with nothing done asks for nothing."""
    found = _flow("""
        answers["/api/tokeye"] = { status: 200, body: { rows: { ece: input.layer } } };
        S.meta = input.meta;
        Overlays.open(S.meta, 1);
        await loadOverlays(1);
        const other = [calls.map((c) => c.path), Object.keys(Overlays.layers), Overlays.states.ece.state, timers.length];
        calls.length = 0;
        S.event = "alfven_eigenmode";
        Overlays.open(S.meta, 1);
        await loadOverlays(1);
        const ae = calls.length;
        S.event = "tearing_mode";
        S.meta = { rows: [{ name: "co2", kind: "image", overlay: { state: "not_computed", reason: "" } }] };
        Overlays.open(S.meta, 1);
        await loadOverlays(1);
        S.api = 10;
        S.meta = input.meta;
        Overlays.open(S.meta, 1);
        await loadOverlays(1);
        return [other, ae, calls.length, timers.length];
    """, {"meta": META, "layer": LAYER})
    assert found == [
        [["/api/tokeye?event=tearing_mode&shot=189081"], ["ece"], "done", 0], 0, 0, 0,
    ]


@needs_node
def test_a_missing_layer_fails_the_chip_and_retry_reads_it_again():
    found = _flow("""
        S.meta = input.meta;
        Overlays.open(S.meta, 1);
        await loadOverlays(1);
        const failed = { ...Overlays.states.ece };
        delete failed.seen;
        answers["/api/tokeye"] = { status: 200, body: { rows: { ece: input.layer } } };
        await Overlays.act("ece");
        return [failed, overlayNote(failed).text, calls.length, Overlays.states.ece.state, Object.keys(Overlays.layers)];
    """, {"meta": META, "layer": LAYER})
    assert found == [
        {"state": "failed", "reason": "the layer was not found", "local": True},
        "TokEye failed: the layer was not found",
        2, "done", ["ece"],
    ]


@needs_node
def test_a_page_failure_is_done_again_once_its_layer_arrives():
    found = _flow("""
        S.meta = input.meta;
        Overlays.open(S.meta, 1);
        await loadOverlays(1);
        const failed = Overlays.states.ece.state;
        Overlays.layers = { ece: Overlays.layer(input.layer) };
        Overlays.check();
        return [failed, Overlays.states.ece.state, Boolean(Overlays.states.ece.local)];
    """, {"meta": META, "layer": LAYER})
    assert found == ["failed", "done", False]


@needs_node
def test_compute_posts_once_and_the_poll_runs_until_nothing_waits():
    found = _flow("""
        S.meta = input.meta;
        answers["/api/overlays/request"] = { status: 202, body: { state: "queued", queue_position: 1 } };
        answers["/api/overlays?"] = { status: 200, body: { rows: { co2: { state: "computing", reason: "", elapsed_s: 3 } } } };
        answers["/api/tokeye"] = { status: 200, body: { rows: { ece: input.layer, co2: input.layer } } };
        Overlays.open(S.meta, 1);
        await loadOverlays(1);
        calls.length = 0;
        const first = requestOverlay(), second = requestOverlay();
        const queued = overlayNote(Overlays.states.co2).text;
        await Promise.all([first, second]);
        const posts = calls.filter((c) => c.method === "POST").map((c) => c.body);
        const after = overlayNote(Overlays.states.co2).text;
        const delays = timers.map((t) => t.delay);
        await tick();
        const computing = Overlays.states.co2.state;
        answers["/api/overlays?"] = { status: 200, body: { rows: { co2: { state: "done", reason: "" } } } };
        await tick();
        const gets = calls.filter((c) => c.method === "GET").map((c) => [c.path.split("?")[0], c.quiet]);
        return [queued, posts, after, delays, computing, Overlays.states.co2.state, gets, timers.length,
                Object.keys(Overlays.layers).sort()];
    """, {"meta": META, "layer": LAYER})
    assert found == [
        "TokEye queued",
        [{"event": "tearing_mode", "shot": 189081}],
        "TokEye queued, 1st in line",
        [5000],
        "computing",
        "done",
        [["/api/overlays", True], ["/api/overlays", True], ["/api/tokeye", False]],
        0,
        ["co2", "ece"],
    ]


@needs_node
def test_the_poll_stops_when_another_shot_opens():
    found = _flow("""
        S.meta = { rows: [{ name: "co2", kind: "image", overlay: { state: "queued", reason: "", queue_position: 2 } }] };
        Overlays.open(S.meta, 1);
        await loadOverlays(1);
        const scheduled = timers.length;
        S.ticket = 2;
        await tick();
        return [scheduled, calls.length, timers.length];
    """)
    assert found == [1, 0, 0]


@needs_node
def test_a_reopen_of_the_same_shot_keeps_its_layers():
    found = _flow("""
        answers["/api/tokeye"] = { status: 200, body: { rows: { ece: input.layer } } };
        S.meta = input.meta;
        Overlays.open(S.meta, 1);
        await loadOverlays(1);
        S.ticket = 2;
        Overlays.open(S.meta, 2);
        await loadOverlays(2);
        const same = [calls.length, Object.keys(Overlays.layers)];
        S.shot = 189082;
        S.ticket = 3;
        Overlays.open(S.meta, 3);
        await loadOverlays(3);
        Overlays.open(null, 3);
        return [same, calls.length, Overlays.states, Object.keys(Overlays.layers)];
    """, {"meta": META, "layer": LAYER})
    assert found == [[1, ["ece"]], 2, {}, []]


@needs_node
def test_a_request_answered_done_reads_the_layers():
    found = _flow("""
        S.meta = { rows: [{ name: "co2", kind: "image", overlay: { state: "stale", reason: "" } }] };
        answers["/api/overlays/request"] = { status: 202, body: { state: "done", queue_position: null } };
        answers["/api/tokeye"] = { status: 200, body: { rows: { co2: input } } };
        Overlays.open(S.meta, 1);
        await loadOverlays(1);
        const before = calls.length;
        await Overlays.act("co2");
        return [before, calls.map((c) => [c.method, c.path.split("?")[0]]), Overlays.states.co2.state,
                Object.keys(Overlays.layers), timers.length];
    """, LAYER)
    assert found == [0, [["POST", "/api/overlays/request"], ["GET", "/api/tokeye"]], "done",
                     ["co2"], 0]
