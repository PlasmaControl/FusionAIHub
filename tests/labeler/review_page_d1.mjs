// Plan D.8's twelve D1 checks: shot lists, By shot, #activity and the TokEye chips. Appended
// to review_browser.mjs's transport (everything before its `const opened =`) by
// test_review_page_d1_browser.py, and run as
//   node <script> <base-url> <token> <headless-shell> <profile-dir>
// Prints one JSON line: the twelve checks, [{name, ok, detail}], in D.8's order.

const AE = "alfven_eigenmode", LOCK = "locked_mode";
const results = new Map();
const check = (n, name, ok, detail) => results.set(n, { name: `${n}. ${name}`, ok: Boolean(ok), detail });
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const tally = (requests) => requests.map((r) => JSON.parse(r).kind).reduce((m, k) => ({ ...m, [k]: (m[k] || 0) + 1 }), {});
let loads = 0;

/** A real left click at (x, y) in CSS px, through the browser's hit test. */
async function click(x, y) {
  const mouse = (type, more = {}) => send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, ...more });
  await mouse("mouseMoved", { button: "none" });
  await mouse("mousePressed", { buttons: 1 });
  await mouse("mouseReleased", { buttons: 0 });
}

/** A key press on the page (after the focus leaves any control). */
async function press(key) {
  await js("document.activeElement?.blur?.()");
  const named = { Enter: [13, "\r"], Escape: [27, ""] }[key];
  const [code, text] = named || [key.toUpperCase().charCodeAt(0), key];
  const event = { key, code: named ? key : `Key${key.toUpperCase()}`, windowsVirtualKeyCode: code };
  await send("Input.dispatchKeyEvent", { type: "keyDown", ...event, ...(text ? { text } : {}) });
  await send("Input.dispatchKeyEvent", { type: "keyUp", ...event });
}

/** Load the page at `hash` in a new document. `fixed` answers a kind of request with a fixed
 * {status, body} (a stubbed /api/version); `reset` clears localStorage first; `storage` sets
 * keys before the page's scripts run. */
async function navigate(hash, { fixed = {}, reset = false, storage = {} } = {}) {
  const setup = `window.d1Fixed = ${JSON.stringify(fixed)};${reset ? "localStorage.clear();" : ""}` +
    Object.entries(storage).map(([k, v]) => `localStorage.setItem(${JSON.stringify(k)}, ${JSON.stringify(v)});`).join("");
  const { identifier } = await send("Page.addScriptToEvaluateOnNewDocument", { source: setup });
  await js("window.d1Left = true");
  await send("Page.navigate", { url: `${BASE}/?token=${TOKEN}&n=${++loads}${hash}` });
  await until(`!window.d1Left && document.readyState === "complete" && typeof S !== "undefined" && typeof Lists !== "undefined"`);
  await send("Page.removeScriptToEvaluateOnNewDocument", { identifier });
}

const opened = (shot, event) => until(`S.shot === ${shot} && S.event === "${event}" && S.data !== null &&
  S.opened?.ticket === S.ticket`);

/** Wait until no request has been made for `stable` ms and none is in flight. */
async function quiet(stable = 700, limit = 15000) {
  let last = null, since = Date.now();
  for (const start = Date.now(); Date.now() - start < limit; await sleep(100)) {
    const now = await js("[window.requests.length, Activity.items.size]");
    if (!same(now, last)) [last, since] = [now, Date.now()];
    else if (now[1] === 0 && Date.now() - since >= stable) return;
  }
  throw new Error("the page did not settle");
}

const count = (kind, where = "true") => js(`window.requests.filter((r) => r.kind === "${kind}" && (${where})).length`);
const activity = () => js(`({ hidden: $("activity").hidden, text: $("activity").querySelector(".activity-text")?.textContent ?? "",
  error: $("activity").classList.contains("error"), stop: !($("activity").querySelector(".activity-stop")?.hidden ?? true) })`);
const chips = () => js(`Object.fromEntries([...document.querySelectorAll("#row-ui .row-chip")].map((chip) => {
  const canvas = $("rows").children[S.meta.rows.findIndex((row) => row.name === chip.dataset.overlay)];
  const [a, b] = [chip.getBoundingClientRect(), canvas.getBoundingClientRect()];
  return [chip.dataset.overlay, { text: chip.children[0].textContent, button: chip.children[1].hidden ? null : chip.children[1].textContent,
    className: chip.className, title: chip.title, shown: chip.style.visibility !== "hidden",
    gutter: a.left >= b.left && a.right <= b.left + GUTTER && a.top >= b.top && a.top < b.bottom }];
}))`);
const snapshot = () => js(`({
  requests: window.requests.map((r) => JSON.stringify(r)).sort(),
  rows: [...$("rows").children].map((c) => [c.tagName, Math.round(c.getBoundingClientRect().height)]),
  chips: [...$("queue").children].map((c) => [c.dataset.shot, c.className]),
  count: $("count").textContent, status: $("status").textContent,
  masks: [$("masks").hidden, $("masks").textContent], tokeye: S.tokeye?.layer ?? null,
})`);

/** The bar's height now; with D1's and D2's controls swapped back for today's (no list, By
 * shot, activity or view; the Gold set box; #saved in full); and with a long #activity note. */
const barHeights = () => js(`(() => {
  const height = () => Math.round($("bar").getBoundingClientRect().height);
  const layout = () => [...$("bar").children].filter((c) => c.getClientRects().length).map((c) => {
    const r = c.getBoundingClientRect();
    return [c.id || c.tagName, Math.round(r.top), Math.round(r.left), Math.round(r.width)];
  });
  const now = height(), lines = layout();
  const swap = [["shotlist", true], ["by-shot", true], ["activity", true], ["view", true], ["gold-only-label", false]];
  const before = swap.map(([id]) => $(id).hidden), saved = $("saved").textContent, title = $("saved").title;
  swap.forEach(([id, hide]) => { $(id).hidden = hide; });
  $("saved").textContent = title || saved;
  const today = height();
  swap.forEach(([id], i) => { $(id).hidden = before[i]; });
  $("saved").textContent = saved;
  Activity.note("d1-width", "A long note to see that the bar keeps its lines: ".repeat(6));
  const noted = height(), notedLines = layout();
  Activity.clear("d1-width");
  return { now, today, noted, saved, title, ...(now !== today || now !== noted ? { lines, notedLines } : {}) };
})()`);

await send("Runtime.enable");
await send("Page.enable");
await send("Page.addScriptToEvaluateOnNewDocument", { source: `
  sessionStorage.setItem("labeler:who", "D1 reviewer");
  window.requests = [];
  window.gates = {};
  window.stubs = {};
  window.json = (body, status = 200) => new Response(JSON.stringify(body),
    { status, headers: { "content-type": "application/json" } });
  const realFetch = window.fetch.bind(window);
  window.realFetch = realFetch;
  window.fetch = async (input, options) => {
    const request = new Request(input, options);
    const url = new URL(request.url);
    const kind = url.pathname.replace(/^\\/api\\//, "");
    const params = Object.fromEntries(url.searchParams);
    window.requests.push({ kind, method: request.method, ...params });
    const fixed = window.d1Fixed?.[kind];
    if (fixed) return window.json(fixed.body, fixed.status ?? 200);
    const stub = window.stubs[kind];
    const answer = stub && (await stub(params, request));
    if (answer) return answer;
    const gate = window.gates[kind];
    const held = gate && !gate.released && (!gate.event || gate.event === params.event) &&
      (!gate.shot || gate.shot === params.shot);
    const response = await realFetch(input, options);
    if (held) {
      await new Promise((done) => (gate.waiters ||= []).push(done));
      if (gate.reject) throw new TypeError("Failed to fetch");
    }
    return response;
  };
` });

try {
  // 1 and 9: a default open at api 11 against the same page told it is api 10.
  const API10 = { version: { body: { api: 10 } } };
  const opens = {};
  let older = null;
  for (const [event, shot] of [[AE, 170815], [LOCK, 170815]]) {
    const runs = {};
    for (const [label, fixed] of [["api10", API10], ["api11", {}]]) {
      await navigate(`#${event}/${shot}`, { fixed, reset: true });
      await opened(shot, event);
      await quiet();
      runs[label] = await snapshot();
      if (label === "api10" && event === AE) {
        older = await js(`({ shotlist: $("shotlist").hidden, byShot: $("by-shot").hidden, activity: $("activity").hidden,
          line: $("activity-line").hidden, gold: $("gold-only-label").hidden, stale: [$("stale").hidden, $("stale").textContent],
          silver: $("event").options[0].textContent, chips: document.querySelectorAll("#row-ui .row-chip").length,
          enabled: Activity.enabled })`);
      }
    }
    const { api10, api11 } = runs;
    const queues = api11.requests.map((r) => JSON.parse(r)).filter((r) => r.kind === "queue");
    opens[event] = {
      same: Object.fromEntries(["requests", "rows", "chips", "count", "masks", "tokeye"].map((key) => [key, same(api10[key], api11[key])])),
      requests: tally(api11.requests), api10: tally(api10.requests), queue: queues, rows: api11.rows, chips: api11.chips.length,
    };
    opens[event].ok = Object.values(opens[event].same).every(Boolean) && queues.length === 1 && !("list" in queues[0]);
  }
  check(1, "a default open at api 11 makes api 10's requests by kind and count, rows, heights and chips; the queue URL has no list=",
    opens[AE].ok && opens[LOCK].ok, opens);

  // 9 (the rest): a failing /api/version says so on #stale.
  await navigate(`#${AE}/170815`, { fixed: { version: { status: 500, body: { error: "fixture version failure" } } }, reset: true });
  await opened(170815, AE);
  const failed = await js(`[$("stale").hidden, $("stale").textContent, S.api, $("shotlist").hidden]`);
  check(9, "an api-10 server shows no list, By shot or activity controls and #stale says why; a failing /api/version shows the version copy",
    older.shotlist && older.byShot && older.activity && older.line && !older.gold && older.silver === "Silver set" &&
      older.chips === 0 && !older.enabled &&
      same(older.stale, [false, "Restart the server for shot lists, TokEye status and views"]) &&
      same(failed, [false, "Could not read the server version (fixture version failure). Reload the page.", 1, true]),
    { older, failed });

  // 2: a held /api/rows.
  await navigate(`#${AE}/170815`, { reset: true });
  await opened(170815, AE);
  await quiet();
  await js(`window.statusMutations = 0;
    new MutationObserver((list) => { window.statusMutations += list.length; })
      .observe($("status"), { childList: true, characterData: true, subtree: true, attributes: true });
    window.gates.rows = {}; S.asked = ""; fetchRows(0); void 0`);
  await until("window.gates.rows.waiters?.length > 0");
  await sleep(2300);
  const loading = { ...(await activity()), line: await js(`!$("activity-line").hidden && $("activity-line").classList.contains("indeterminate")`) };
  await js("window.gates.rows.released = true; window.gates.rows.waiters.forEach((done) => done()); delete window.gates.rows");
  await until(`Activity.items.size === 0 && $("activity-line").hidden`);
  const touched = await js("window.statusMutations");
  check(2, "a held /api/rows shows Loading rows with its elapsed time on #activity, and #status is not mutated",
    !loading.hidden && /^Loading rows · 2(\.\d)? s$/.test(loading.text) && loading.line && touched === 0, { loading, touched });

  // 10: the silent sites (prefetch, mask, TokEye) each say so; M retries the mask.
  await js(`window.stubs.shot = (p) => (p.shot === "170816" ? json({ error: "fixture build failure" }, 500) : null);
    window.stubs.masks = (p, r) => (r.method === "GET" ? json({ error: "fixture mask failure" }, 500) : null);
    window.stubs.tokeye = (p) => (p.event === "${AE}" ? json({ error: "fixture layer failure" }, 500) : null); void 0`);
  await js("void openShot(170815)");
  await opened(170815, AE);
  await quiet();
  const silent = await js(`({ prefetch: Activity.notes.get("prefetch"), mask: Activity.notes.get("masks"),
    status: [$("status").textContent, $("status").className], masks: $("masks").hidden, tokeye: S.tokeye })`);
  silent.chips = await chips();
  silent.activity = await activity();
  await js("window.stubs = {}");
  await press("m");
  await until(`S.masks !== null && S.tokeye !== null && !Activity.notes.has("masks")`);
  const retried = { chips: await chips(), masks: await js(`$("masks").hidden`) };
  const chip = silent.chips.R0xV1 || {};
  check(10, "a failed prefetch, a mask 500 and a TokEye 500 each give their note; M retries the mask",
    silent.prefetch?.text === "Next shot 170816 failed to build: fixture build failure" && silent.prefetch.tone === "" &&
      silent.mask?.text === "Could not load the AE mask. Press M to retry." && silent.mask.tone === "error" &&
      chip.text?.startsWith("TokEye failed: the layer could not be loaded") && chip.title.includes("fixture layer failure") &&
      chip.className.includes("error") && chip.button === "Retry" &&
      same(silent.status, ["", ""]) && silent.masks && silent.tokeye === null &&
      silent.activity.error && silent.activity.text === silent.mask.text &&
      retried.chips.R0xV1?.text === "TokEye" && retried.masks === false,
    { silent, retried });

  // 3: a store build (202) with its stage, Stop waiting, and the stuck message, at 1280 px
  // (the narrowest pinned width) with the mask shown: Stop waiting must be the element under
  // its own centre, and a real click on it ends the poll.
  await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 610, deviceScaleFactor: 1, mobile: false });
  await sleep(300);
  await js(`window.stubs.shot = (p) => (p.shot === "170817" ? json({ building: true,
    progress: { done: 2, total: 5, stage: "fetching mirnov" }, stage: "fetching", elapsed_s: 14, queue_position: 1,
    prebuilt: false }, 202) : null); void 0`);
  await js("void openShot(170817)");
  await until(`$("activity").textContent.includes("Building 170817")`);
  const building = { ...(await activity()), status: await js(`$("status").textContent`),
    width: await js(`$("activity-line").style.width`) };
  await js("Activity.now = () => Date.now() + 121000; void 0");
  await until(`$("activity").textContent.startsWith("Still building after")`);
  const stuck = await activity();
  await js("Activity.now = () => Date.now(); void 0");
  const stop = await js(`(() => {
    const button = $("activity").querySelector(".activity-stop"), text = $("activity").querySelector(".activity-text");
    const r = button.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
    const hit = document.elementFromPoint(x, y);
    return { x, y, hit: hit === button || button.contains(hit), under: hit ? (hit.id || hit.className || hit.tagName) : null,
      text: Math.round(text.getBoundingClientRect().width), masks: !$("masks").hidden, width: innerWidth };
  })()`);
  await click(stop.x, stop.y);
  await until(`S.shot === 170817 && S.meta === null && $("rows").textContent === Activity.stopped(170817)`);
  const asked = await count("shot", `r.shot === "170817"`);
  await sleep(2000);
  const after = { asked, later: await count("shot", `r.shot === "170817"`), activity: await activity(),
    status: await js(`$("status").textContent`) };
  await js("window.stubs = {}");
  await send("Emulation.clearDeviceMetricsOverride");
  await sleep(300);
  check(3, "a 202 shows its stage, progress and queue; #status keeps \"<shot>: building\"; at 1280 px Stop waiting is hit-testable and a real click on it ends the poll; 120 s still gives the stuck copy",
    /^Building 170817 for .+: fetching mirnov \(2 of 5\) · 14 s · 1st in line$/.test(building.text) &&
      building.stop && building.status.startsWith("170817: building") && building.width === "40%" &&
      stuck.text === "Still building after 2 min. The server may be stuck." &&
      stop.hit && stop.text >= 40 && stop.width === 1280 &&
      after.later === after.asked && !after.activity.text.includes("Building") && after.status === "",
    { building, stuck, stop, after });

  // 5: AE 170816 has no layer: #masks as review_masks.mjs:193 expects, and the chip says so.
  await js("void openShot(170816)");
  await opened(170816, AE);
  await quiet();
  const bare = await js(`({ tokeye: S.tokeye, masks: S.masks, hidden: $("masks").hidden })`);
  bare.chips = await chips();
  await js("window.statusMutations = 0");
  await press("m");
  await sleep(200);
  bare.m = await js(`[Activity.notes.get("layers")?.text, window.statusMutations, $("status").textContent]`);
  check(5, "AE 170816 with no layer keeps S.tokeye null and #masks hidden; its chip reads No TokEye yet; M says why on #activity",
    bare.tokeye === null && bare.masks === null && bare.hidden &&
      same(Object.keys(bare.chips), ["R0xV1"]) && bare.chips.R0xV1.text === "No TokEye yet" && bare.chips.R0xV1.button === "Compute" &&
      same(bare.m, ["This shot has no pseudo-mask or TokEye layer.", 0, ""]),
    bare);

  // 4: every TokEye state's chip on locked_mode 170815; Compute posts once; the poll.
  const STATES = {
    k_done: { state: "done", reason: "" },
    k_stale: { state: "stale", reason: "an older model" },
    k_queued: { state: "queued", reason: "", queue_position: 2 },
    k_computing: { state: "computing", reason: "", elapsed_s: 40 },
    k_failed: { state: "failed", reason: "CUDA out of memory on the GPU node after three retries" },
    k_new: { state: "not_computed", reason: "No TokEye yet for this row" },
  };
  const LAYER = { shot: 170815, layer: "tokeye", grid: { t0_ms: 0, dt_ms: 250, n: 8 }, y0_khz: 0, dy_khz: 16, y0: 0, dy: 16,
    n_y: 8, bits: "/wAAAAAAAAA=" };
  await js(`Overlays.POLL_MS = 300;
    window.stubs.shot = async (p, request) => {
      if (p.event !== "${LOCK}" || p.shot !== "170815") return null;
      const body = await (await window.realFetch(request)).json();
      const states = ${JSON.stringify(STATES)};
      for (const row of body.rows) if (states[row.name]) row.overlay = states[row.name];
      window.pollRows = Object.fromEntries(body.rows.map((row) => [row.name, row.overlay]));
      return json(body);
    };
    window.stubs.tokeye = (p) => (p.event === "${LOCK}" ? json({ rows: { k_done: ${JSON.stringify(LAYER)} } }) : null);
    window.stubs.overlays = (p) => json({ event: p.event, shot: Number(p.shot), rows: window.pollRows });
    window.stubs["overlays/request"] = () => {
      for (const name of ["k_stale", "k_failed", "k_new"]) window.pollRows[name] = { state: "queued", reason: "", queue_position: 3 };
      return json({ state: "queued", queue_position: 3 }, 202);
    };
    void openEvent("${LOCK}", 170815)`);
  await opened(170815, LOCK);
  await until(`Overlays.any() && document.querySelectorAll("#row-ui .row-chip").length === 6`);
  await until(`[...document.querySelectorAll("#row-ui .row-chip")].every((chip) => chip.style.visibility !== "hidden")`);
  const states = await chips();
  const plain = await js(`$("rows").children[S.meta.rows.findIndex((row) => row.name === "k_plain")].title`);
  await until(`window.requests.filter((r) => r.kind === "overlays").length >= 2`);
  await js(`(() => {
    const button = document.querySelector('#row-ui .row-chip[data-overlay="k_new"] button');
    button.click();
    button.click();
  })()`);
  await until(`document.querySelector('#row-ui .row-chip[data-overlay="k_new"]').textContent.startsWith("TokEye queued")`);
  await sleep(700);
  const posts = await js(`window.requests.filter((r) => r.kind === "overlays/request").map((r) => r.method)`);
  const queued = (await chips()).k_new;
  await js("void openShot(170816)");
  await opened(170816, LOCK);
  const polls = await count("overlays");
  await sleep(1200);
  const stopped = { polls, later: await count("overlays") };
  await js("window.stubs = {}");
  const want = {
    k_done: ["TokEye", null, "row-chip"],
    k_stale: ["TokEye from an older model", "Recompute", "row-chip"],
    k_queued: ["TokEye queued, 2nd in line", null, "row-chip queued"],
    k_new: ["No TokEye yet", "Compute", "row-chip"],
  };
  const fits = Object.entries(want).every(([name, [text, button, className]]) =>
    states[name]?.text === text && states[name].button === button && states[name].className === className) &&
    /^TokEye computing · 4\d s$/.test(states.k_computing?.text) &&
    states.k_failed?.text.startsWith("TokEye failed: CUDA out of memory") && states.k_failed.text.endsWith("…") &&
    states.k_failed.title === `TokEye failed: ${STATES.k_failed.reason}` && states.k_failed.button === "Retry" &&
    states.k_failed.className === "row-chip error" && Object.values(states).every((found) => found.shown && found.gutter);
  check(4, "each TokEye state has its gutter copy; Compute posts once and shows queued; the poll runs only while rows wait and stops on a shot change",
    fits && !("k_plain" in states) && plain === "No TokEye: not a spectrogram" && same(posts, ["POST"]) &&
      queued.text === "TokEye queued, 3rd in line" && stopped.later === stopped.polls,
    { states, plain, posts, queued, stopped });

  // 7: a queue that fails, then the event chosen again: exactly two queue requests.
  const before = await count("queue", `r.event === "${AE}"`);
  await js(`window.gates.queue = { event: "${AE}", reject: true };
    $("event").value = "${AE}"; $("event").dispatchEvent(new Event("change")); void 0`);
  await until("window.gates.queue.waiters?.length > 0");
  await js("window.gates.queue.released = true; window.gates.queue.waiters.forEach((done) => done())");
  await until(`$("rows").textContent.includes("the queue could not be read")`);
  const refused = await js(`[$("rows").textContent, $("shotlist").disabled]`);
  await js(`delete window.gates.queue; $("event").dispatchEvent(new Event("change")); void 0`);
  await until(`S.event === "${AE}" && S.data !== null && S.opened?.ticket === S.ticket`);
  const retry = { refused, queues: (await count("queue", `r.event === "${AE}"`)) - before,
    lists: await js(`[$("shotlist").disabled, $("shotlist").options.length]`) };
  check(7, "a queue retry after a failure makes exactly two queue requests and fills the list again",
    refused[0] === `${AE}: the queue could not be read: Failed to fetch. Choose the event again to retry.` &&
      retry.queues === 2 && retry.lists[0] === false && retry.lists[1] > 0,
    retry);

  // 11: an expired fdp login (a stubbed /api/version).
  await navigate(`#${AE}/170815`, { fixed: { version: { body: { api: 11, fdp: { env: true, expires_at: "2020-01-01T00:00:00Z" },
    workspace: true } } }, reset: true });
  await opened(170815, AE);
  await quiet();
  const fdp = { ...(await activity()), stale: await js(`$("stale").hidden`) };
  check(11, "an expired fdp login shows the activity message, in the error colour",
    !fdp.hidden && fdp.error && fdp.stale &&
      fdp.text === "fdp login expired: no new signals. Run fdp login on the server, then Retry.",
    fdp);

  // 6: the list dropdown, the Gold set migration, equal-length lists, the recommendation.
  await navigate(`#${AE}/170815`, { reset: true, storage: { "labeler:gold-only": "1" } });
  await opened(170815, AE);
  await quiet();
  const list = (since) => js(`({ queues: window.requests.slice(${since}).filter((r) => r.kind === "queue").map((r) => r.list ?? null),
    value: $("shotlist").value, recommended: $("shotlist").classList.contains("recommended"), title: $("shotlist").title,
    chips: [...$("queue").children].map((c) => c.dataset.shot),
    extra: [...$("queue").children].filter((c) => c.classList.contains("extra")).map((c) => [c.dataset.shot, c.title]) })`);
  const migrated = { ...(await list(0)), ae: await js(`localStorage.getItem("labeler:list:${AE}")`),
    lock: await js(`localStorage.getItem("labeler:list:${LOCK}")`), box: await js(`localStorage.getItem("labeler:gold-only")`),
    groups: await js(`[...$("shotlist").children].map((g) => [g.label, [...g.children].map((o) => o.value)])`) };
  const choose = async (name) => {
    const since = await js("window.requests.length");
    await js(`$("shotlist").value = "${name}"; $("shotlist").dispatchEvent(new Event("change")); void 0`);
    await until(`Lists.name === "${name}" && S.data !== null && S.opened?.ticket === S.ticket`);
    await quiet();
    return list(since);
  };
  const a = await choose("ae_a");
  const b = await choose("ae_b");
  await navigate(`#${AE}`);
  await until(`S.event === "${AE}" && S.data !== null && S.opened?.ticket === S.ticket`);
  await quiet();
  const reloaded = { ...(await list(0)), box: await js(`localStorage.getItem("labeler:gold-only")`) };
  const roster = await choose("roster");
  roster.stored = await js(`localStorage.getItem("labeler:list:${AE}")`);
  check(6, "one queue request per list chosen (list= only off the default); equal-length lists rebuild the chips; the recommended group comes first and the border follows; Gold set migrates once",
    same(migrated.queues, ["gold"]) && migrated.ae === "gold" && migrated.lock === "gold" && migrated.box === null &&
      migrated.value === "gold" && migrated.recommended && migrated.groups[0][0].startsWith("Recommended for") &&
      same(migrated.groups[0][1], ["ae_a", "gold"]) && same(migrated.extra, [["171234", "Not on this event's roster"]]) &&
      same(a.queues, ["ae_a"]) && same(a.chips, ["170815", "170816"]) && a.recommended &&
      same(b.queues, ["ae_b"]) && same(b.chips, ["170816", "170817"]) && !b.recommended &&
      b.title === "Not recommended for this event" &&
      same(reloaded.queues, ["ae_b"]) && reloaded.value === "ae_b" && reloaded.box === null &&
      same(roster.queues, [null]) && same(roster.chips, ["170815", "170816", "170817"]) && roster.stored === null &&
      same(roster.extra, []),
    { migrated, a, b, reloaded, roster });

  // 8: By shot, from an old #silver/ address.
  await navigate("#silver/2/170899", { reset: true });
  await until(`Lists.byShot && Lists.strips?.shot === 170899`);
  const walk = await js(`({ hash: location.hash, top: $("top").hidden, pressed: $("by-shot").getAttribute("aria-pressed"),
    batches: [...$("silver-batches").children].map((b) => [b.textContent, b.getAttribute("aria-pressed")]),
    audited: $("silver-shots").textContent.includes("AE audited"),
    strips: [...$("silver-strips").children].map((s) => s.dataset.event) })`);
  await press("Enter");
  await until(`Lists.context && S.event === "${AE}" && S.shot === 170899 && S.data !== null && S.opened?.ticket === S.ticket`);
  const inside = await js(`({ hash: location.hash, crumb: [$("silver-breadcrumb").hidden, $("silver-breadcrumb").textContent],
    queue: S.queue.map((r) => [r.shot, Boolean(r.extra)]), top: $("top").hidden })`);
  await js("edit(S.label.window, [[1001, 1004, 1]]); void save(true)");
  await until(`S.event === "${LOCK}" && S.shot === 170899 && S.data !== null && S.opened?.ticket === S.ticket`);
  const next = await js(`({ hash: location.hash, context: Lists.context })`);
  await press("g");
  await until(`Lists.byShot && Lists.strips?.shot === 170899`);
  const back = await js(`({ hash: location.hash, saved: Lists.strips.events.find((e) => e.event === "${AE}").saved })`);
  await press("Escape");
  await until(`Lists.byShot && Lists.shot === null && Lists.overview?.list === "silver_batch_2"`);
  await press("Escape");
  await until(`!Lists.byShot && S.data !== null && S.opened?.ticket === S.ticket`);
  const left = await js(`[location.hash, $("top").hidden, $("by-shot").getAttribute("aria-pressed")]`);
  check(8, "By shot: #silver/2/170899 redirects to #list/, Enter opens the event, Save goes to the next event, g and Escape go back",
    walk.hash === "#list/silver_batch_2/170899" && walk.top && walk.pressed === "true" && walk.audited &&
      walk.batches.length === 3 && same(walk.batches[2], ["Silver batch 2 · 0/1 done", "true"]) &&
      same(walk.strips, [AE, LOCK]) &&
      inside.hash === `#list/silver_batch_2/170899/${AE}` && same(inside.crumb, [false, "← 170899 overview"]) &&
      same(inside.queue, [[170899, true]]) && !inside.top &&
      next.hash === `#list/silver_batch_2/170899/${LOCK}` && next.context &&
      back.hash === "#list/silver_batch_2/170899" && back.saved === true &&
      /^#[a-z_]+\/\d+$/.test(left[0]) && !left[1] && left[2] === "false",
    { walk, inside, next, back, left });

  // 12: the bar's height at 1440, 1400 and 1280 px, on AE (saved under a name, the mask shown;
  // measured after the save and again on a fresh open of the saved shot once its mask is
  // shown) and on locked_mode.
  const WIDTHS = [[1440, 790], [1400, 800], [1280, 610]];
  await navigate(`#${AE}/170815`, { reset: true });
  await opened(170815, AE);
  await quiet();
  await js("edit(S.label.window, [[500, 800, 1]]); void save(false)");
  await until("S.meta.saved !== null && !S.saving");
  const bars = {};
  for (const [width, height] of WIDTHS) {
    await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
    await sleep(300);
    bars[`${AE} ${width}`] = await barHeights();
  }
  for (const [width, height] of WIDTHS) {
    await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
    await navigate(`#${AE}/170815`);
    await opened(170815, AE);
    await until(`S.meta.saved !== null && !$("masks").hidden`);
    await quiet();
    await sleep(1500);
    bars[`${AE} ${width} reopened`] = { ...(await barHeights()), masks: await js(`$("masks").textContent`),
      innerWidth: await js("innerWidth") };
  }
  await send("Emulation.clearDeviceMetricsOverride");
  await js(`void openEvent("${LOCK}", 170815)`);
  await opened(170815, LOCK);
  await quiet();
  for (const [width, height] of WIDTHS) {
    await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
    await sleep(300);
    bars[`${LOCK} ${width}`] = await barHeights();
  }
  await send("Emulation.clearDeviceMetricsOverride");
  check(12, "at 1440, 1400 and 1280 px the bar keeps today's height on the fixture events (a fresh open of a saved AE shot and a long #activity note included)",
    Object.values(bars).every((bar) => bar.now === bar.today && bar.noted === bar.now) &&
      / by D1 reviewer$/.test(bars[`${AE} 1440`].saved) &&
      WIDTHS.every(([width]) => bars[`${AE} ${width} reopened`].innerWidth === width && bars[`${AE} ${width} reopened`].masks),
    bars);
} catch (error) {
  console.log(JSON.stringify([...results.keys()].sort((x, y) => x - y).map((n) => results.get(n))));
  console.error(error.stack);
  console.error(JSON.stringify(errors));
  process.exit(1);
}

console.log(JSON.stringify([...results.keys()].sort((x, y) => x - y).map((n) => results.get(n))));
ws.close();
browser.kill();
if (errors.length) {
  console.error(JSON.stringify(errors));
  process.exit(1);
}
