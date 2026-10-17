// Plan D.8's twenty-one D2 checks, 22-24 (D2B's dock width, D2C's colour bar) and plan D.B's
// dock checks: views, the dock, pins and their
// tabs, the Values readout, the rest time, the cut margins and probe guides, added signals with
// their fetch and error states, saved views, the narrow overlay; on detachment the dock that
// opens with the TangTV inversion, the EFIT equilibrium and the cameras. Appended to
// review_browser.mjs's transport (everything before its `const opened =`) by
// test_review_workspace_browser.py, and run as
//   node <script> <base-url> <token> <headless-shell> <profile-dir> checks
//   node <script> <base-url> <token> <headless-shell> <profile-dir> shots <out-dir>
// "checks" prints one JSON line: [{name, ok, detail}] in the order 1-24, B1-B5, R1-R9, E (22-24 are
// review D2B's dock width at 1280 and D2C's colour bar). "shots"
// saves a "q and Te" view on mhd_n1 and writes d2_<w>x<h>.png at three window sizes into
// <out-dir>, then prints what it saw.

const AE = "alfven_eigenmode", MHD = "mhd_n1", DET = "detachment", EMPTY = "disruption";
const INVERSION_MS = [0, 50, 100, 150, 200, 240], EFIT_MS = [0, 30, 80, 110, 190, 240];
const API10 = { version: { body: { api: 10 } } };
const FORBIDDEN = ["views", "catalog", "signal", "profile", "slices", "slice", "fetch/status"];
const ORDER = [...Array.from({ length: 24 }, (_, k) => String(k + 1)), "B1", "B2", "B3", "B4", "B5",
  "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "E"];
const Q_DEFAULT = [{ kind: "profile", id: "q_profile", h: 200 }];
const results = new Map();
const check = (n, name, ok, detail) => results.set(String(n), { name: `${n}. ${name}`, ok: Boolean(ok), detail });
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const tally = (requests) => requests.reduce((m, r) => ({ ...m, [r.kind]: (m[r.kind] || 0) + 1 }), {});
const near = (a, b, tol) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= tol;
const nearest = (list, t) => list.reduce((best, v) => (Math.abs(v - t) < Math.abs(best - t) ? v : best));
let loads = 0;

// -- input

const mouse = (type, x, y, more = {}) =>
  send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, ...more });
/** Two animation frames: what a pointer move or a key asked for has been drawn. */
const frames = () => js("new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(() => done(true))))");

async function move(x, y, more = {}) {
  await mouse("mouseMoved", x, y, { button: "none", ...more });
}

async function click(x, y, { modifiers = 0, clickCount = 1 } = {}) {
  await mouse("mouseMoved", x, y, { button: "none", modifiers });
  await mouse("mousePressed", x, y, { buttons: 1, modifiers, clickCount });
  await mouse("mouseReleased", x, y, { buttons: 0, modifiers, clickCount });
}

async function drag(x0, y0, x1, y1, modifiers = 0) {
  await mouse("mouseMoved", x0, y0, { button: "none", modifiers });
  await mouse("mousePressed", x0, y0, { buttons: 1, modifiers });
  for (let i = 1; i <= 8; i++) await mouse("mouseMoved", x0 + ((x1 - x0) * i) / 8, y0 + ((y1 - y0) * i) / 8, { buttons: 1, modifiers });
  await mouse("mouseReleased", x1, y1, { buttons: 0, modifiers });
}

/** A key press; `modifiers` is CDP's bit set (1 Alt, 2 Ctrl, 8 Shift). The focus leaves any
 * control first unless `blur` is false. */
async function key(name, { modifiers = 0, blur = true } = {}) {
  if (blur) await js("document.activeElement?.blur?.()");
  const named = { Enter: [13, "\r"], Escape: [27, ""], ArrowUp: [38, ""], ArrowDown: [40, ""],
    ArrowLeft: [37, ""], ArrowRight: [39, ""], Delete: [46, ""], Backspace: [8, ""] }[name];
  const [code, text] = named || [name.toUpperCase().charCodeAt(0), name];
  const event = { key: name, code: named ? name : `Key${name.toUpperCase()}`, windowsVirtualKeyCode: code, modifiers };
  await send("Input.dispatchKeyEvent", { type: "keyDown", ...event, ...(modifiers || !text ? {} : { text }) });
  await send("Input.dispatchKeyEvent", { type: "keyUp", ...event });
}

// -- the page

/** Load the page at `hash` in a new document; `fixed` answers a kind of request with a fixed
 * {status, body}; `reset` clears localStorage first. */
async function navigate(hash, { fixed = {}, reset = false } = {}) {
  const setup = `window.fixed = ${JSON.stringify(fixed)};${reset ? "localStorage.clear();" : ""}`;
  const { identifier } = await send("Page.addScriptToEvaluateOnNewDocument", { source: setup });
  await js("window.left = true");
  await send("Page.navigate", { url: `${BASE}/?token=${TOKEN}&n=${++loads}${hash}` });
  await until(`!window.left && document.readyState === "complete" && typeof S !== "undefined" && typeof Views !== "undefined"`);
  await send("Page.removeScriptToEvaluateOnNewDocument", { identifier });
}

const opened = (shot, event) => until(`S.shot === ${shot} && S.event === "${event}" && S.data !== null &&
  S.opened?.ticket === S.ticket`);

/** Wait until no request has been made for `stable` ms and none is in flight. */
async function quiet(stable = 700, limit = 20000) {
  let last = null, since = Date.now();
  for (const start = Date.now(); Date.now() - start < limit; await sleep(100)) {
    const now = await js("[window.requests.length, Activity.items.size]");
    if (!same(now, last)) [last, since] = [now, Date.now()];
    else if (now[1] === 0 && Date.now() - since >= stable) return;
  }
  throw new Error("the page did not settle");
}

const requests = (since = 0) => js(`window.requests.slice(${since})`);
const mark = () => js("window.requests.length");
const X = (t) => js(`$("axis-row").getBoundingClientRect().left + px(${t})`);
/** The vertical middle of #rows' child `i` (a store row in store order). */
const rowY = (i) => js(`(() => { const r = $("rows").children[${i}].getBoundingClientRect(); return r.top + r.height / 2; })()`);
const axisY = () => js(`(() => { const r = $("axis-row").getBoundingClientRect(); return r.top + r.height / 2; })()`);
const barPoint = () => js(`(() => { const r = $("bar").getBoundingClientRect(); return [r.left + 30, r.top + r.height / 2]; })()`);
const msPerPx = () => js("(S.view[1] - S.view[0]) / plotWidth()");
/** The middle of the AE pseudo-mask's region 2 (176-186 kHz, 614-737 ms) on the image row. */
const region2 = () => js(`(() => {
  const r = $("rows").children[0].getBoundingClientRect();
  const [lo, hi] = imageRange(S.meta.rows[0]);
  return [$("axis-row").getBoundingClientRect().left + px(675), r.top + (r.height - 1) * (1 - (181 - lo) / (hi - lo))];
})()`);
const footer = (text) => js(`[...document.querySelectorAll("#view-footer button")].find((b) => b.textContent === ${JSON.stringify(text)}).click()`);
const showBox = (title) => js(`document.querySelector('#dock-edit tr[data-key="store:${title}"] input[type=checkbox]').click()`);

await send("Runtime.enable");
await send("Page.enable");
await send("Page.addScriptToEvaluateOnNewDocument", { source: `
  sessionStorage.setItem("labeler:who", "D2 reviewer");
  window.requests = [];
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
    window.requests.push({ ...params, kind, method: request.method, ...(params.kind ? { slice: params.kind } : {}) });
    const fixed = window.fixed?.[kind];
    if (fixed) return window.json(fixed.body, fixed.status ?? 200);
    const stub = window.stubs[kind];
    const answer = stub && (await stub(params, request));
    if (answer) return answer;
    return realFetch(input, options);
  };
  /** RGBA bytes of a canvas box given in CSS px. */
  window.pixels = (canvas, x, y, w, h) => {
    const k = window.devicePixelRatio || 1;
    return [...canvas.getContext("2d").getImageData(Math.round(x * k), Math.round(y * k),
      Math.max(1, Math.round(w * k)), Math.max(1, Math.round(h * k))).data];
  };
  /** A colour as RGB bytes. */
  window.rgb = (colour) => {
    const c = document.createElement("canvas");
    c.width = c.height = 1;
    const g = c.getContext("2d");
    g.fillStyle = colour;
    g.fillRect(0, 0, 1, 1);
    return [...g.getImageData(0, 0, 1, 1).data.slice(0, 3)];
  };
` });

/** Plan D.8's checks and D.B's. */
async function checks() {
  // 1: a default open at api 11 makes api 10's requests, and shows today's page.
  const opens = {};
  for (const [event, shot] of [[AE, 170815], [MHD, 180001]]) {
    const runs = {};
    for (const [label, fixed] of [["api10", API10], ["api11", {}]]) {
      await navigate(`#${event}/${shot}`, { fixed, reset: true });
      await opened(shot, event);
      await quiet();
      runs[label] = await js(`({
        requests: window.requests.map((r) => JSON.stringify(r)).sort(),
        kinds: [...new Set(window.requests.map((r) => r.kind))],
        dock: [$("dock").hidden, typeof Dock === "undefined" ? false : Dock.open], right: right(),
        panels: typeof Dock === "undefined" ? [] : Dock.panels,
        margin: typeof Slice === "undefined" ? 0 : Slice.margin,
        view: [$("view").hidden, $("view").selectedOptions[0]?.textContent ?? null],
        plain: [...$("rows").children].every((c) => !c.style.order && !c.classList.contains("view-hidden")),
      })`);
    }
    const { api10, api11 } = runs;
    opens[event] = { kinds: api11.kinds, sameRequests: same(api10.requests, api11.requests), api11: { ...api11, requests: undefined } };
    if (event === AE) {
      opens[event].ok = opens[event].sameRequests && !api11.kinds.some((kind) => FORBIDDEN.includes(kind)) &&
        same(api11.dock, [true, false]) && api11.right === 12 && api11.margin === 0 &&
        same(api11.view, [false, "Default"]) && api11.plain;
    } else {
      // R2: q on mhd_n1 (and the q_min and poloidal-beta events) opens the dock with q, which
      // adds /api/profile and nothing else that api 10 does not ask.
      opens[event].ok = api11.kinds.every((kind) => kind === "profile" || api10.kinds.includes(kind)) &&
        api11.kinds.includes("profile") && !api11.kinds.some((kind) => kind !== "profile" && FORBIDDEN.includes(kind)) &&
        same(api11.dock, [false, true]) && same(api11.panels, Q_DEFAULT) && same(api11.view, [false, "Default"]) && api11.plain;
    }
  }
  check(1, "a default open at api 11 makes api 10's requests on AE (no views, catalog, signal, profile or slice request; the dock hidden, right() 12); on mhd_n1 it adds only /api/profile for the q panel the dock opens with; #view says Default and no row is reordered or hidden",
    opens[AE].ok && opens[MHD].ok, opens);

  // 2: every key on an event with no shots asks nothing and throws nothing.
  await navigate(`#${AE}/170815`, { reset: true });
  await opened(170815, AE);
  await quiet();
  await js(`$("event").value = "${EMPTY}"; $("event").dispatchEvent(new Event("change")); void 0`);
  await until(`S.event === "${EMPTY}" && S.meta === null`);
  await quiet();
  const sweep = { before: await mark(), errors: errors.length };
  sweep.keys = await js(`(() => {
    const keys = [...Object.keys(KEYS), ...Object.keys(WORKSPACE_KEYS), "1", "5", "9"];
    const fire = (key, more = {}) => document.body.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...more }));
    for (const name of keys) fire(name.replace(/^Shift\\+/, ""), { shiftKey: name.startsWith("Shift+") });
    fire("z", { ctrlKey: true });
    return keys;
  })()`);
  await sleep(1500);
  sweep.after = await mark();
  sweep.errorsAfter = errors.length;
  sweep.made = await requests(sweep.before);
  check(2, "on an event with no shots every key (p, x, l, d and today's) makes no request and no error",
    sweep.after === sweep.before && sweep.errorsAfter === sweep.errors && ["p", "x", "l", "d"].every((k) => sweep.keys.includes(k)),
    sweep);

  await js(`void openEvent("${AE}", 170815)`);
  await opened(170815, AE);
  await quiet();
  const trace = await rowY(1);

  // 3: p and x at the hover, with no request.
  const pinned = { before: await mark(), hovers: [] };
  for (const [t, name] of [[500, "p"], [1500, "p"], [1400, "p"], [600, "x"]]) {
    await move(await X(t), trace);
    await frames();
    pinned.hovers.push(await js("Slice.hover"));
    await key(name);
  }
  pinned.pins = await js("Slice.pins");
  pinned.made = await requests(pinned.before);
  const [h500, h1500, h1400] = pinned.hovers;
  check(3, "p pins A then B at the hover, a third p moves the nearer pin, x removes the pin nearest the hover; no request",
    near(h500, 500, 3) && near(h1500, 1500, 3) && near(h1400, 1400, 3) && pinned.pins.A === null && pinned.pins.B === h1400 &&
      h1500 !== h1400 && pinned.made.length === 0, pinned);

  // 4 (AE): a click on the time axis pins A at the time under it.
  await js("Slice.setPins({ A: null, B: null }); void 0");
  const ax = Math.round(await X(800));
  await click(ax, await axisY());
  await frames();
  const axisPin = { x: ax, want: await js(`timeAt(${ax})`), pins: await js("Slice.pins") };
  const fourAE = axisPin.pins.A === axisPin.want && axisPin.pins.B === null && near(axisPin.want, 800, 3);

  // 5: a double-click on the axis fits the view and leaves the pins as they were. Check 4's
  // axis click must be older than Slice.UNDO_MS, or the double-click takes it back too.
  await sleep(await js("Slice.UNDO_MS + 100"));
  await js("Slice.setPins({ A: 300, B: 1700 }); fit(); void 0");
  const fitted = await js("S.view.slice()");
  await js(`(() => {
    window.axisClicks = 0;
    const real = axisClickPin;
    axisClickPin = (t) => { window.axisClicks++; return real(t); };
    setView(800, 1200);
  })()`);
  await frames();
  const dx = await X(1000), dy = await axisY();
  await mouse("mouseMoved", dx, dy, { button: "none" });
  await mouse("mousePressed", dx, dy, { buttons: 1, clickCount: 1 });
  await mouse("mouseReleased", dx, dy, { buttons: 0, clickCount: 1 });
  const between = await js("Slice.pins");
  await mouse("mousePressed", dx, dy, { buttons: 1, clickCount: 2 });
  await mouse("mouseReleased", dx, dy, { buttons: 0, clickCount: 2 });
  await frames();
  const doubled = { fitted, between, view: await js("S.view.slice()"), pins: await js("Slice.pins"), clicks: await js("window.axisClicks") };
  // A double-click on a pin's tab fits too, and pins nothing: the tab takes its presses. The
  // axis clicks above must be older than Slice.UNDO_MS first, or the fit takes them back.
  await sleep(await js("Slice.UNDO_MS + 100"));
  await js("Slice.setPins({ A: 1000, B: null }); setView(800, 1200); window.axisClicks = 0; void 0");
  await frames();
  const [px5, py5] = await js(`(() => {
    const r = document.querySelector("#pins .pin-a .pin-text").getBoundingClientRect();
    return [r.left + r.width / 2, r.top + r.height / 2];
  })()`);
  await mouse("mouseMoved", px5, py5, { button: "none" });
  await mouse("mousePressed", px5, py5, { buttons: 1, clickCount: 1 });
  await mouse("mouseReleased", px5, py5, { buttons: 0, clickCount: 1 });
  await mouse("mousePressed", px5, py5, { buttons: 1, clickCount: 2 });
  await mouse("mouseReleased", px5, py5, { buttons: 0, clickCount: 2 });
  await frames();
  doubled.tab = { view: await js("S.view.slice()"), pins: await js("Slice.pins"), clicks: await js("window.axisClicks") };
  check(5, "a double-click on the axis fits the view and puts the pins back as they were before its two clicks; a double-click on a pin tab fits the view and leaves the pins",
    same(doubled.view, fitted) && same(doubled.pins, { A: 300, B: 1700 }) && doubled.clicks === 2 &&
      same(doubled.tab.view, fitted) && same(doubled.tab.pins, { A: 1000, B: null }) && doubled.tab.clicks === 0, doubled);

  // 6: the tabs: drag A, × removes B; an axis drag pans and a Shift-drag adds a span, pins kept;
  // Ctrl+Z undoes the span.
  await js("Slice.setPins({ A: 500, B: 900 }); void 0");
  await frames();
  const tabAt = (pin, part) => js(`(() => {
    const r = document.querySelector("#pins .pin-${pin} ${part}").getBoundingClientRect();
    return [r.left + r.width / 2, r.top + r.height / 2];
  })()`);
  const [ta, tay] = await tabAt("a", ".pin-text");
  await drag(ta, tay, await X(700), tay);
  await frames();
  const tabs = { dragged: await js("Slice.pins.A"), msPerPx: await msPerPx() };
  const [tb, tby] = await tabAt("b", ".pin-text");
  await move(tb, tby);
  await frames();
  const [rx, ry] = await tabAt("b", ".pin-remove");
  await click(rx, ry);
  await frames();
  tabs.removed = await js("Slice.pins");
  await js("setView(1000, 1600); void 0");
  await frames();
  const view0 = await js("S.view.slice()");
  const pan0 = await X(1300);
  await drag(pan0, await axisY(), pan0 + 120, await axisY());
  tabs.panned = { before: view0, after: await js("S.view.slice()"), pins: await js("Slice.pins") };
  await js("fit(); void 0");
  await frames();
  const spans0 = await js("JSON.stringify(S.label.intervals)");
  await drag(await X(1500), trace, await X(1600), trace, 8);
  tabs.added = { spans: await js("S.label.intervals.length"), before: JSON.parse(spans0).length, pins: await js("Slice.pins") };
  await key("z", { modifiers: 2 });
  tabs.undone = await js(`JSON.stringify(S.label.intervals) === ${JSON.stringify(spans0)}`);
  check(6, "dragging A's tab moves A, the tab's × removes B, an axis drag pans and a Shift-drag adds a span with the pins kept, Ctrl+Z takes the span back",
    near(tabs.dragged, 700, tabs.msPerPx + 0.5) && tabs.removed.B === null && tabs.removed.A === tabs.dragged &&
      tabs.panned.after[0] < tabs.panned.before[0] && same(tabs.panned.pins, tabs.removed) &&
      tabs.added.spans === tabs.added.before + 1 && same(tabs.added.pins, tabs.removed) && tabs.undone,
    tabs);

  // 7: Alt-click on the mask pins A and decides nothing; Alt on the label lane pins B and edits nothing.
  await js("Slice.setPins({ A: null, B: null }); void 0");
  await until("S.masks !== null && S.showMasks");
  const alt = { rejected: await js("JSON.stringify(S.masks.rejected)"), view: await js("S.view.slice()"), msPerPx: await msPerPx(),
    label: await js("JSON.stringify(S.label)") };
  const [mx, my] = await region2();
  await click(mx, my, { modifiers: 1 });
  await sleep(300);
  alt.afterMask = { pins: await js("Slice.pins"), saves: await js("maskSaves.size"), rejected: await js("JSON.stringify(S.masks.rejected)"),
    view: await js("S.view.slice()"), want: await js(`timeAt(${mx})`) };
  const lane = await js(`(() => { const r = $("label-track").getBoundingClientRect(); return r.top + r.height / 2; })()`);
  const lx = await X(1000);
  await click(lx, lane, { modifiers: 1 });
  await frames();
  alt.afterLane = { pins: await js("Slice.pins"), label: await js("JSON.stringify(S.label)"), want: await js(`timeAt(${lx})`) };
  check(7, "Alt-click on a mask region sets A without rejecting it or moving the view; Alt-press on the label lane sets B and leaves the label",
    near(alt.afterMask.pins.A, alt.afterMask.want, alt.msPerPx) && alt.afterMask.saves === 0 && alt.afterMask.rejected === alt.rejected &&
      same(alt.afterMask.view, alt.view) && near(alt.afterLane.pins.B, alt.afterLane.want, alt.msPerPx) &&
      alt.afterLane.pins.A === alt.afterMask.pins.A && alt.afterLane.label === alt.label,
    alt);

  // 10 (AE): with the dock closed, #cursor follows the pointer over the rows and goes over the bar.
  const [bx, by] = await barPoint();
  await move(await X(1200), trace);
  await frames();
  const cursor = { rows: await js(`!$("cursor").hidden`), dock: await js("Dock.open") };
  await move(bx, by);
  await frames();
  cursor.bar = await js(`$("cursor").hidden`);
  const tenAE = cursor.rows && cursor.bar && !cursor.dock;

  // 11: d on Default opens Values and makes one rows request; the view says "(changed)".
  await js(`void resetView()`);
  await quiet();
  const before11 = await mark();
  await key("d");
  await until("Dock.open");
  await quiet();
  const opened11 = { made: tally(await requests(before11)), panels: await js("Dock.panels"),
    view: await js(`$("view").selectedOptions[0]?.textContent`), title: await js(`document.querySelector("#dock .dock-title")?.textContent`) };
  check(11, "d on Default opens a dock holding Values with one rows request, and #view and the dock title read Default (changed)",
    same(opened11.made, { rows: 1 }) && same(opened11.panels, [{ kind: "values" }]) &&
      opened11.view === "Default (changed)" && opened11.title === "Default (changed)",
    opened11);

  // 12: the cut margin narrows the time column; times and a drawn span stay exact.
  const narrowed = await js(`({ margin: Slice.margin, right: right(), at: timeAt($("axis-row").getBoundingClientRect().left + px(1300)),
    end: px(S.view[1]), plot: $("axis-row").clientWidth - 12 - Slice.margin })`);
  const lane12 = await js(`(() => { const r = $("label-track").getBoundingClientRect(); return r.top + r.height / 2; })()`);
  await drag(await X(1240), lane12, await X(1360), lane12);
  await frames();
  const drawn = await js("S.label.intervals.map((s) => [s[0], s[1]])");
  await key("s");
  await until("S.meta.saved !== null && !S.saving && !dirty()");
  narrowed.drawn = drawn;
  narrowed.saved = await js("S.meta.saved !== null");
  check(12, "with the dock open an image row gets a 96 or 120 px cut margin, right() grows by it, and timeAt(px(t)) is t; a span drawn at 1240-1360 saves exactly",
    [96, 120].includes(narrowed.margin) && narrowed.right === 12 + narrowed.margin && near(narrowed.at, 1300, 1e-6) &&
      near(narrowed.end, narrowed.plot, 0.5) && drawn.some(([a, b]) => a === 1240 && b === 1360) && narrowed.saved,
    narrowed);

  // 8: the Values readout at A and B, and Copy values with a hidden row.
  await js("Slice.setPins({ A: 700, B: 900 }); void 0");
  await frames();
  await sleep(100);
  const values = await js(`(() => {
    const out = {};
    for (const item of Slice.rows()) {
      const tr = document.querySelector('#dock tr[data-key="' + item.key + '"]');
      const [a, b] = [valueAt(item, 700), valueAt(item, 900)];
      const want = [Slice.valueText(a).text, Slice.valueText(b).text, Slice.number(Slice.valueNumber(b) - Slice.valueNumber(a))];
      out[item.title] = { cells: tr ? [...tr.children].slice(2).map((td) => td.textContent) : null, want,
        a: Slice.valueNumber(a), b: Slice.valueNumber(b) };
    }
    out.t = [...document.querySelector('#dock tr[data-key="t"]').children].map((c) => c.textContent);
    return out;
  })()`);
  await js(`Object.defineProperty(navigator, "clipboard", { configurable: true,
    value: { writeText: async (text) => { window.copied = text; } } });
    Views.setRow("store:Density", { hidden: true }); void 0`);
  await frames();
  const [cx, cy] = await tabAt("b", ".pin-copy");
  await click(cx, cy);
  await until("typeof window.copied === 'string'");
  const copied = await js("window.copied");
  const hiddenRow = await js(`$("rows").children[1].classList.contains("view-hidden")`);
  await js(`Views.setRow("store:Density", { hidden: false }); void 0`);
  const density = values.Density, image = values["R0 × V1"];
  check(8, "Values shows each row at A and B as valueText says and B−A; Density and the image peak are the store's values; Copy values copies a hidden row too",
    density && image && same(density.cells, density.want) && same(image.cells, image.want) &&
      near(density.a, Math.sin(700 / 120), 0.05) && near(density.b, Math.sin(900 / 120), 0.05) &&
      near(image.a, 4 * (32 + 20 * Math.sin(700 / 300)), 8) && same(values.t.slice(2), ["700", "900", "+200"]) &&
      hiddenRow && /^Density\tV?[^\t]*\t[-\d.e]+\t[-\d.e]+\t/m.test(copied) && copied.startsWith("row\tunits\tA\tB\tB−A\nt (ms)\tms\t700\t900\t200\n"),
    { values, copied, hiddenRow });

  // 9: the rest time from event timestamps: a dwell of 120 ms or more is held when the pointer leaves.
  await js(`window.stamps = []; window.addEventListener("pointermove", (e) => window.stamps.push([e.clientX, e.timeStamp]), true); void 0`);
  const x1 = Math.round(await X(1100)), x2 = Math.round(await X(1500));
  const run = async (timed) => {
    await move(bx, by);
    await frames();
    await js("window.stamps = []");
    const base = Date.now() / 1000 - 1;
    const plan = [[x1, 0], [x1 + 1, 60], [x1 + 2, 150], [x2, 290], [x2 + 20, 320], [x2 + 40, 350]];
    let last = 0;
    for (const [x, ms] of plan) {
      if (!timed) await sleep(Math.max(0, ms - last - (ms - last > 100 ? 0 : 25)));
      await move(x, trace, timed ? { timestamp: base + ms / 1000 } : {});
      await frames();
      last = ms;
    }
    await move(bx, by, timed ? { timestamp: base + 0.38 } : {});
    await frames();
    return js("window.stamps");
  };
  let stamps = await run(true);
  const honoured = stamps.length >= 7 && near(stamps[3][1] - stamps[0][1], 290, 5) && near(stamps[5][1] - stamps[3][1], 60, 5);
  if (!honoured) stamps = await run(false);
  await frames();
  const rest = { honoured, stamps, want: await js(`timeAt(${x1})`), rest: await js("Slice.rest"), slice: await js("sliceTime()"),
    held: await js("Slice.time(Slice.rest)"),
    reading: await js(`document.querySelector("#dock .dock-reading")?.textContent`),
    cursor: await js(`[$("cursor").hidden, $("cursor").classList.contains("held")]`) };
  check(9, "a pointer that dwelt 120 ms or more at a time leaves its rest time there: the dock follows it, its header says held at, and #cursor holds dotted",
    near(rest.rest, rest.want, 1e-6) && near(rest.slice, rest.want, 1e-6) &&
      Boolean(rest.reading?.startsWith(`held at ${rest.held} ms`)) && same(rest.cursor, [false, true]),
    rest);

  // 17: Alt+↑ on a grip moves the row by CSS order only; a mask click still finds its region.
  await js("openEditor(); void 0");
  await until(`document.querySelector('#dock-edit tr[data-key="store:Density"] .grip')`);
  await js(`document.querySelector('#dock-edit tr[data-key="store:Density"] .grip').focus()`);
  await key("ArrowUp", { modifiers: 1, blur: false });
  await frames();
  const moved = await js(`({ order: [...$("rows").children].map((c) => c.style.order), first: S.meta.rows[0].title,
    above: $("rows").children[1].getBoundingClientRect().top < $("rows").children[0].getBoundingClientRect().top,
    focus: document.activeElement?.closest?.("tr")?.dataset.key ?? null, rejected: S.masks.rejected.slice() })`);
  const [qx, qy] = await region2();
  await click(qx, qy);
  await until("maskSaves.size === 0 && S.masks.rejected.length === 1");
  moved.rejected1 = await js("S.masks.rejected.slice()");
  const [qx2, qy2] = await region2();
  await click(qx2, qy2);
  await until("maskSaves.size === 0 && S.masks.rejected.length === 0");
  moved.rejected2 = await js("S.masks.rejected.slice()");
  check(17, "Alt+↑ on Density's grip shows it above the image with style.order only (store order kept, focus kept); a click on mask region 2 still rejects it and a second takes it back",
    same(moved.order, ["1", "0"]) && moved.first === "R0 × V1" && moved.above && moved.focus === "store:Density" &&
      moved.rejected.length === 0 && moved.rejected1.length === 1 && moved.rejected2.length === 0,
    moved);

  // 18: Save, reload (one GET), Rename, a conflict from another tab, Delete, Reset to default;
  // the editor stays open through Delete and Reset (review D2B M1), and Done closes the dock
  // Default has none of.
  await showBox("Density");
  await js(`$("view-name").value = "Mine"; void 0`);
  await footer("Save");
  await until(`Views.current?.name === "Mine" && !Views.changed`);
  const views = { stored: await js(`stored(Views.storeKey("${AE}"))`) };
  await navigate(`#${AE}/170815`);
  await opened(170815, AE);
  await quiet();
  views.reload = await js(`({ gets: window.requests.filter((r) => r.kind === "views" && r.method === "GET").length,
    hidden: $("rows").children[1].classList.contains("view-hidden"), order: [...$("rows").children].map((c) => c.style.order),
    dock: Dock.open, current: Views.current?.name, value: $("view").value })`);
  await js("openEditor(); void 0");
  await js(`$("view-name").value = "Mine 2"; void 0`);
  await footer("Rename");
  await until(`Views.current?.name === "Mine 2"`);
  views.renamed = { stored: await js(`stored(Views.storeKey("${AE}"))`), list: await js("Views.list.filter((v) => !v.preset).map((v) => v.name)") };
  views.otherTab = await js(`(async () => {
    const body = { event: S.event, reviewer: S.name, op: "save", name: "Mine 2", layout: Views.snapshot("Mine 2"), etag: Views.current.etag };
    const response = await window.realFetch("/api/views", { method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    return response.status;
  })()`);
  await footer("Save");
  await until(`document.querySelector("#view-footer .view-message")?.textContent.includes("another tab")`);
  views.conflict = await js(`document.querySelector("#view-footer .view-message").textContent`);
  await until("Views.listing === null && Views.list !== null");
  await js(`Views.choose("view:Mine 2")`);
  await until(`Views.current?.name === "Mine 2" && document.querySelector("#view-footer")`);
  await footer("Delete");
  await until("Views.current === null");
  const editorShown = () => js(`({ dock: !$("dock").hidden, edit: !$("dock-edit").hidden, editing: Views.editing,
    footer: Boolean(document.querySelector("#view-footer")) })`);
  views.deleted = { stored: await js(`stored(Views.storeKey("${AE}"))`), label: await js("Views.label()"),
    list: await js("(Views.list || []).filter((v) => !v.preset).map((v) => v.name)"), shown: await editorShown() };
  await js("openEditor(); void 0");
  await showBox("Density");
  views.hidden = await js("Views.label()");
  await footer("Reset to default");
  await until("!Views.changed");
  views.reset = await js(`({ label: Views.label(), hidden: [...$("rows").children].some((c) => c.classList.contains("view-hidden")),
    value: $("view").value })`);
  views.reset.shown = await editorShown();
  await footer("Done");
  views.done = await js(`({ dock: !$("dock").hidden, open: Dock.open, editing: Views.editing })`);
  check(18, "Save remembers the view, a reload asks /api/views once and applies it, Rename keeps it, a save from another tab gives the conflict copy, Delete and Reset to default go back to Default with the editor still open, and Done closes the dock",
    views.stored === "view:Mine" && views.reload.gets === 1 && views.reload.hidden && same(views.reload.order, ["1", "0"]) &&
      views.reload.dock && views.reload.current === "Mine" && views.reload.value === "view:Mine" &&
      views.renamed.stored === "view:Mine 2" && same(views.renamed.list, ["Mine 2"]) && views.otherTab === 200 &&
      views.conflict === "This view changed in another tab. Reload it, or save yours under a new name." &&
      views.deleted.stored === null && views.deleted.label === "Default" && same(views.deleted.list, []) &&
      views.hidden === "Default (changed)" && views.reset.label === "Default" && !views.reset.hidden && views.reset.value === "" &&
      Object.values(views.deleted.shown).every(Boolean) && Object.values(views.reset.shown).every(Boolean) &&
      !views.done.dock && !views.done.open && !views.done.editing,
    views);

  // 19: a view naming a row the store lacks, and lacking one it has.
  await js(`Views.post({ op: "save", name: "Partial", layout: { v: 1, name: "Partial", event: S.event, reviewer: S.name,
    rows: [{ key: "store:Gone row" }, { key: "store:Density", h: 150 }], other_store_rows: "end",
    dock: { open: false, width: null, follow: "cursor", panels: [] } } })`);
  await js(`Views.choose("view:Partial")`);
  await frames();
  const partial = await js(`({ note: Activity.notes.get("view-rows")?.text ?? null, order: [...$("rows").children].map((c) => c.style.order),
    image: [$("rows").children[0].classList.contains("view-hidden"), Math.round($("rows").children[0].getBoundingClientRect().height)],
    label: Views.label() })`);
  check(19, "a view with a row the store lacks and without one it has says both in one muted note and shows the new row",
    partial.note === "1 row of this view is not in this store: Gone row. 1 row is new in this store: R0 × V1" &&
      same(partial.order, ["1", "0"]) && partial.image[0] === false && partial.image[1] > 0 && partial.label === "Partial",
    partial);

  // 20: below 1,100 px the dock lies over the rows: no margin, the time column keeps its width.
  await send("Emulation.setDeviceMetricsOverride", { width: 1024, height: 700, deviceScaleFactor: 1, mobile: false });
  await sleep(400);
  await quiet();
  const narrow = { before: await js(`$("axis-row").clientWidth`), open: await js("Dock.open") };
  await key("d");
  await until("Dock.open");
  await frames();
  await quiet();
  Object.assign(narrow, await js(`(() => {
    const style = getComputedStyle($("dock"));
    return { column: style.gridColumnStart, z: style.zIndex, after: $("axis-row").clientWidth, right: right(), margin: Slice.margin,
      dockLeft: Math.round($("dock").getBoundingClientRect().left) };
  })()`));
  const lane20 = await js(`(() => { const r = $("label-track").getBoundingClientRect(); return r.top + r.height / 2; })()`);
  await drag(await X(500), lane20, await X(600), lane20);
  await frames();
  narrow.drawn = await js("S.label.intervals.map((s) => [s[0], s[1]])");
  await key("s");
  await until("S.meta.saved !== null && !S.saving && !dirty()");
  await key("d");
  await until("!Dock.open");
  narrow.closed = await js(`$("dock").hidden`);
  await send("Emulation.clearDeviceMetricsOverride");
  await sleep(300);
  check(20, "at 1024 px d lays the dock over the rows (grid column 1, z-index 15) with no margin and the time column unchanged; a span drawn at 500-600 saves exactly; d hides it",
    !narrow.open && narrow.column === "1" && narrow.z === "15" && narrow.after === narrow.before && narrow.right === 12 &&
      narrow.margin === 0 && narrow.drawn.some(([a, b]) => a === 500 && b === 600) && narrow.closed,
    narrow);

  // mhd_n1: the margins on spectrograms and modes, profiles, added signals.
  const beforeMHD = await mark();
  await js(`void openEvent("${MHD}", 180001)`);
  await opened(180001, MHD);
  await quiet();
  await until(`Signals.entry(Profiles.shots, "profile", S.shot, "q_profile")?.state === "done"`);
  await frames();

  // R5: q is in the dock by default on mhd_n1, asked once; its readout follows the cursor.
  const qDefault = await js(`({ open: Dock.open, panels: Dock.panels, view: $("view").selectedOptions[0]?.textContent,
    title: Dock.nodes.get("profile:q_profile")?.name.textContent ?? null })`);
  qDefault.asked = (await requests(beforeMHD)).filter((r) => r.kind === "profile").map((r) => r.id ?? r.name ?? r.signal ?? "");
  await move(await X(1000), await rowY(2));
  await frames();
  await sleep(250); // a dwell: the rows' time rests at 1,000 ms while the pointer is in the dock
  const qCanvas = await js(`(() => { const r = Dock.nodes.get("profile:q_profile").canvas.getBoundingClientRect();
    return [r.left + r.width * 0.6, r.top + r.height / 2]; })()`);
  await move(qCanvas[0], qCanvas[1]);
  await frames();
  qDefault.probe = await js("Slice.probe && { kind: Slice.probe.kind, from: Slice.probe.from }");
  qDefault.note = await js(`Dock.nodes.get("profile:q_profile")?.note.textContent ?? ""`);
  qDefault.readout = await js(`(() => {
    const texts = [], proto = CanvasRenderingContext2D.prototype, real = proto.fillText;
    proto.fillText = function (text, ...rest) { texts.push(String(text)); return real.call(this, text, ...rest); };
    try { drawDock(); } finally { proto.fillText = real; }
    return texts.filter((text) => text.startsWith("ψN "));
  })()`);
  check("R5", "mhd_n1 opens with q in the dock (one /api/profile), following the cursor (nearest slice 1,000 ms); a pointer on the panel gives its own ψN · q readout",
    qDefault.open && same(qDefault.panels, Q_DEFAULT) && qDefault.view === "Default" && qDefault.asked.length === 1 &&
      /^q/.test(qDefault.title || "") && /1,000 ms/.test(qDefault.note) && qDefault.probe?.kind === "psi_n" &&
      qDefault.readout.length === 1 && / · /.test(qDefault.readout[0]),
    qDefault);

  // 13: the cut at A, the margin's pixels, a probe from the margin, n at A in Values.
  await js("Views.setPanels([{ kind: \"values\" }]); void 0");
  await until("Dock.open && Slice.margin > 0 && Dock.panels.length === 1 && Dock.panels[0].kind === \"values\"");
  await js("Slice.setPins({ A: 1000, B: null }); void 0");
  await quiet();
  await frames();
  const cut = await js(`(() => {
    const items = Slice.rows();
    const power = items.find((item) => item.title === "MPI66M322D power");
    const line = Slice.cutAt(power, 1000);
    const data = Slice.storeData(power.i, 1000), j = Slice.bucket(data, 1000);
    const column = line.length > 0 && line.every(([y, v]) => v === data.rows[power.i][Math.round((y - power.row.y0) / power.row.dy) * data.n + j]);
    const peak = line.reduce((best, p) => (p[1] > best[1] ? p : best));
    const canvas = $("cut-margins"), box = canvas.getBoundingClientRect(), rect = power.canvas.getBoundingClientRect();
    const w = box.width;
    const block = window.pixels(canvas, w - Slice.margin + 2, rect.top - box.top + 2, Slice.margin - 4, rect.height - 4);
    let painted = 0;
    for (let k = 3; k < block.length; k += 4) painted += block[k] > 0;
    return { column, peak, margin: Slice.margin, painted, area: block.length / 4, hidden: canvas.hidden };
  })()`);
  const probeAt = await js(`(() => {
    const power = Slice.rows().find((item) => item.title === "MPI66M322D power").canvas.getBoundingClientRect();
    const axis = $("axis-row").getBoundingClientRect();
    return [axis.left + axis.width - 12 - Slice.margin / 2, power.top + power.height * 0.3];
  })()`);
  await move(probeAt[0], probeAt[1]);
  await frames();
  const probe = await js(`(() => {
    const p = Slice.probe;
    const modes = Slice.rows().find((item) => item.title === "toroidal n, MPI66M");
    const canvas = $("cut-margins"), box = canvas.getBoundingClientRect(), rect = modes.canvas.getBoundingClientRect();
    const [lo, hi] = Slice.yRange(modes);
    const y = rect.top - box.top + Math.round(((hi - p.v) / (hi - lo)) * (rect.height - 1));
    const line = window.pixels(canvas, GUTTER + 10, y - 1, 200, 3);
    let ink = 0;
    for (let k = 3; k < line.length; k += 4) ink += line[k] > 0;
    const off = window.pixels(canvas, GUTTER + 10, Math.max(0, y - 12), 200, 3);
    let stray = 0;
    for (let k = 3; k < off.length; k += 4) stray += off[k] > 0;
    return { kind: p?.kind, from: p?.from, v: p?.v, ink, stray };
  })()`);
  const nAtA = await js(`document.querySelector('#dock tr[data-key="store:toroidal n, MPI66M"]')?.children[2]?.textContent ?? null`);
  const [b13x, b13y] = await barPoint();
  await move(b13x, b13y);
  check(13, "on mhd_n1 the power row's cut at A is the store's column (peak near 92 kHz) and is drawn in the margin; a probe from the margin draws a guide across the modes row; Values says n = 1 at A",
    cut.column && near(cut.peak[0], 92, 4) && [96, 120].includes(cut.margin) && !cut.hidden && cut.painted > cut.area * 0.9 &&
      probe.kind === "freq_khz" && probe.from === "margin" && probe.ink > 10 && probe.stray === 0 && nAtA === "n = 1",
    { cut, probe, nAtA });

  // 24: an image row's colour bar in its 12 px right edge, labelled with z_lo, z_hi and z_units,
  // the cut naming the scale, the value under the cursor in the cursor's label and the peak's
  // value in Values; the modes row keeps its key and gets no bar. The store's power row is
  // 30 (-35.3 dB) away from its line and about 240 (-2.4 dB) on it, near 92 kHz at 1,000 ms.
  const bar = await js(`(() => {
    const items = Slice.rows();
    const power = items.find((item) => item.title === "MPI66M322D power");
    const modes = items.find((item) => item.title === "toroidal n, MPI66M");
    const texts = [], proto = CanvasRenderingContext2D.prototype, real = proto.fillText;
    proto.fillText = function (text, ...rest) { texts.push(String(text)); return real.call(this, text, ...rest); };
    try { drawRows(); drawMargins(); } finally { proto.fillText = real; }
    const c = power.canvas, w = c.clientWidth, h = c.clientHeight;
    const words = new Uint32Array(new Uint8ClampedArray(window.pixels(c, w - 6, 3, 1, h - 8)).buffer);
    const levels = [...words].map((word) => S.lut.indexOf(word));
    const known = levels.filter((v) => v >= 0);
    const m = modes.canvas;
    const blank = window.pixels(m, m.clientWidth - 6, 3, 1, m.clientHeight - 8).filter((v, k) => k % 4 === 3).every((a) => a === 0);
    return { labels: ["0 dB", "−40 dB", "−40 to 0 dB"].map((text) => texts.includes(text)), top: known[0], bottom: known[known.length - 1],
      inLut: known.length / levels.length, falling: known.every((v, k) => k === 0 || v <= known[k - 1]),
      modes: Slice.zScale(modes.row), blank };
  })()`);
  const point24 = await js(`(() => {
    const r = Slice.rows().find((item) => item.title === "MPI66M322D power").canvas.getBoundingClientRect();
    return [Math.round($("axis-row").getBoundingClientRect().left + px(1000)), Math.round(r.top + r.height * 0.1)];
  })()`);
  await move(point24[0], point24[1]);
  await frames();
  Object.assign(bar, await js(`(() => {
    const power = Slice.rows().find((item) => item.title === "MPI66M322D power");
    const tr = document.querySelector('#dock tr[data-key="store:MPI66M322D power"]');
    return { label: $("cursor-time").textContent, at: Math.round(timeAt(${point24[0]})), cell: tr?.children[2]?.textContent ?? null,
      want: Slice.valueText(valueAt(power, 1000)).text };
  })()`));
  bar.read = /^(\d+) ms · ([\d.]+) kHz · (.+)$/.exec(bar.label)?.slice(1) ?? null;
  bar.peak = /^peak ([\d.]+) kHz · −([\d.]+) dB$/.exec(bar.cell ?? "")?.slice(1).map(Number) ?? null;
  check(24, "the power row has a colour bar of the page's colours at its right edge (bright on top) labelled 0 dB and −40 dB, its cut says −40 to 0 dB; the cursor reads −35.3 dB off the line and Values the peak near 92 kHz within 5 dB of the top; the modes row has no bar",
    bar.labels.every(Boolean) && bar.inLut >= 0.95 && bar.top >= 230 && bar.bottom >= 0 && bar.bottom <= 25 && bar.falling &&
      bar.read && Number(bar.read[0]) === bar.at && Number(bar.read[1]) > 200 && bar.read[2] === "−35.3 dB" &&
      bar.peak && near(bar.peak[0], 92, 4) && bar.peak[1] < 5 && bar.want === bar.cell && bar.modes === null && bar.blank,
    bar);

  // 14: a profile panel: one request, nearest slice, the q marks. The q panel mhd_n1 opens
  // with (R5) asked for it already, so the one request is counted from the shot's open.
  const before14 = beforeMHD;
  await js(`void openPicker("q")`);
  await until(`[...document.querySelectorAll("#dock-edit .picker-result")].some((r) => r.querySelector("span")?.textContent === "q")`);
  await js(`[...document.querySelectorAll("#dock-edit .picker-result")].find((r) => r.querySelector("span")?.textContent === "q").querySelector("button").click()`);
  await until(`Signals.entry(Profiles.shots, "profile", S.shot, "q_profile").state === "done"`);
  await quiet();
  const asked14 = (await requests(before14)).filter((r) => r.kind === "profile").length;
  for (const t of [400, 900, 1700, 2600, 3300]) {
    await move(await X(t), await rowY(2));
    await frames();
  }
  await quiet();
  const profile = await js(`(() => {
    const p = Signals.entry(Profiles.shots, "profile", S.shot, "q_profile").data;
    const linear = (t) => { let best = 0; for (let i = 1; i < p.t.length; i++) if (Math.abs(p.t[i] - t) < Math.abs(p.t[best] - t)) best = i; return best; };
    const times = [305, 1425, 2000, 3990, 10, 4100, 1450];
    return { times, got: times.map((t) => profileAt(p, t).i), want: times.map(linear), n: p.t.length, x_kind: p.x_kind };
  })()`);
  profile.asked = asked14;
  profile.later = (await requests(before14)).filter((r) => r.kind === "profile").length;
  await js("Views.closeEditor(); void 0");
  await frames();
  profile.texts = await js(`(() => {
    const texts = [], proto = CanvasRenderingContext2D.prototype, real = proto.fillText;
    proto.fillText = function (text, ...rest) { texts.push(String(text)); return real.call(this, text, ...rest); };
    try { drawDock(); } finally { proto.fillText = real; }
    return texts.filter((text) => text.startsWith("q ="));
  })()`);
  // The slice the panel shows, read from its note with the panel following A at each time
  // (review D2B M10): 305 and 1425 tie 300/310 and 1400/1450, so the earlier one is shown.
  profile.shown = await js(`(() => {
    const p = Signals.entry(Profiles.shots, "profile", S.shot, "q_profile").data;
    const [follow, pins] = [Slice.follow, { ...Slice.pins }];
    const note = () => [...document.querySelectorAll("#dock .dock-profile")]
      .find((panel) => panel.querySelector(".dock-panel-title")?.textContent.startsWith("q"))?.querySelector(".dock-panel-note")?.textContent ?? "";
    const out = [];
    Slice.follow = "A";
    try {
      for (const t of ${JSON.stringify([305, 1425, 2000, 3990, 10, 4100, 1450])}) {
        Slice.pins = { A: t, B: null };
        drawDock();
        out.push([t, note()]);
      }
    } finally {
      [Slice.follow, Slice.pins] = [follow, pins];
      drawDock();
    }
    return { notes: out, t: [...p.t] };
  })()`);
  const nearestEarlier = (list, t) => list.reduce((best, v) => (Math.abs(v - t) < Math.abs(best - t) ? v : best));
  profile.read = profile.shown.notes.map(([t, text]) => [t, Number((/^[A-Za-z]+ ([\d,]+) ms/.exec(text)?.[1] ?? "NaN").replace(/,/g, "")),
    Math.round(nearestEarlier(profile.shown.t, t))]);
  check(14, "Add as panel on q asks /api/profile once and a hover asks nothing; the panel shows the nearest slice (the earlier on a tie: 300 at 305, 1,400 at 1,425); the q = 1, 3/2 and 2 marks are drawn",
    profile.asked === 1 && profile.later === 1 && same(profile.got, profile.want) && profile.x_kind === "psi_n" &&
      profile.read.every(([, shown, want]) => shown === want) && profile.read[0][1] === 300 && profile.read[1][1] === 1400 &&
      ["q = 1", "q = 3/2", "q = 2"].every((text) => profile.texts.includes(text)),
    profile);

  // 15: a PTDATA point through the fetch worker: queued, fetching, drawn.
  await js(`void openPicker("DUSBRADIAL")`);
  await until(`[...document.querySelectorAll("#dock-edit button.picker-result")].some((b) => b.textContent === "Add PTDATA DUSBRADIAL")`);
  await js(`window.chipTexts = [];
    new MutationObserver(() => {
      const chip = document.querySelector('#row-ui .row-chip[data-signal="custom:DUSBRADIAL"]');
      const text = chip?.firstChild?.textContent;
      if (text && window.chipTexts[window.chipTexts.length - 1] !== text) window.chipTexts.push(text);
    }).observe($("row-ui"), { childList: true, subtree: true, characterData: true });
    [...document.querySelectorAll("#dock-edit button.picker-result")].find((b) => b.textContent === "Add PTDATA DUSBRADIAL").click(); void 0`);
  await until(`Views.added.find((a) => a.id === "custom:DUSBRADIAL")?.entry?.state === "done"`, 30000);
  await frames();
  const fetched = await js(`(() => {
    const added = Views.added.find((a) => a.id === "custom:DUSBRADIAL");
    const c = added.canvas, w = c.clientWidth;
    const block = window.pixels(c, GUTTER + 2, 1, w - GUTTER - right() - 4, c.clientHeight - 2);
    let ink = 0;
    for (let k = 3; k < block.length; k += 4) ink += block[k] > 0;
    return { texts: window.chipTexts, chip: Boolean(document.querySelector('#row-ui .row-chip[data-signal="custom:DUSBRADIAL"]')),
      ink, n: added.entry.data.n, channels: added.entry.data.channels };
  })()`);
  const before15 = await mark();
  await js(`void openPicker("9BAD")`);
  await until(`document.querySelector("#dock-edit p.error")`);
  await sleep(300);
  fetched.refusal = await js(`document.querySelector("#dock-edit p.error").textContent`);
  fetched.refusedAsked = (await requests(before15)).length;
  const queuedAt = fetched.texts.findIndex((text) => text.startsWith("Queued, 1st in line"));
  // The chip may show the page's own "Fetching" for a frame before the server's answer says the
  // job is queued; the order checked is the server's: Queued, then Fetching.
  const fetchingAt = fetched.texts.findIndex((text, k) => k > queuedAt && text.startsWith("Fetching from PTDATA"));
  check(15, "Add PTDATA DUSBRADIAL shows Queued, 1st in line, then Fetching from PTDATA, then the drawn row without a chip; 9BAD is refused before any request",
    queuedAt >= 0 && fetchingAt > queuedAt && !fetched.chip && fetched.ink > 20 && fetched.channels === 1 &&
      fetched.refusal === "9BAD is not a PTDATA point name: use a letter, then letters, digits or _, up to 12 characters." &&
      fetched.refusedAsked === 0,
    fetched);

  // 16: each error kind's chip.
  await js("Views.closeEditor(); void 0");
  await js(`window.stubs.signal = (p) => {
    const answers = {
      "custom:BADPT": [{ error: "BADPT is not a point this fixture knows", kind: "invalid" }, 422],
      "custom:GONEPT": [{ error: "no such point", kind: "missing" }, 404],
      "custom:AUTHPT": [{ error: "token expired", kind: "auth" }, 503],
      "custom:UPPT": [{ error: "fixture upstream failure", kind: "upstream" }, 502],
      "custom:SLOWPT": [{ error: "took too long", kind: "timeout" }, 504],
    }[p.id];
    return answers ? json(...answers) : null;
  };
  for (const id of ["BADPT", "GONEPT", "AUTHPT", "UPPT", "SLOWPT"]) Views.addRow("custom:" + id); void 0`);
  await until(`["BADPT", "GONEPT", "AUTHPT", "UPPT", "SLOWPT"].every((id) => Views.added.find((a) => a.id === "custom:" + id)?.entry?.state === "error")`);
  await frames();
  const failed = await js(`(() => {
    const out = {};
    for (const id of ["BADPT", "GONEPT", "AUTHPT", "UPPT", "SLOWPT"]) {
      const chip = document.querySelector('#row-ui .row-chip[data-signal="custom:' + id + '"]');
      const added = Views.added.find((a) => a.id === "custom:" + id);
      out[id] = { text: chip?.firstChild.textContent ?? null, button: chip && !chip.lastChild.hidden ? chip.lastChild.textContent : null,
        error: chip?.classList.contains("error") ?? null, height: Math.round(added.canvas.getBoundingClientRect().height) };
    }
    out.sentence = Activity.sentence("BADPT is not a point this fixture knows");
    return out;
  })()`);
  await js(`window.stubs = {}; Views.closeEditor(); void Views.choose("")`);
  await quiet();
  check(16, "invalid, missing, auth, upstream and timeout each give their chip copy and button; a missing row collapses to 28 px",
    failed.BADPT.text === failed.sentence && failed.BADPT.button === "Edit" && failed.BADPT.error &&
      failed.GONEPT.text === "No GONEPT for 180001." && failed.GONEPT.button === "Remove from view" && failed.GONEPT.height === 28 &&
      failed.AUTHPT.text === "fdp login expired. Run fdp login on the server, then Retry." &&
      failed.AUTHPT.button === "Retry" &&
      failed.UPPT.text === "The data server failed: fixture upstream failure. Try again soon." && failed.UPPT.button === "Retry" &&
      failed.SLOWPT.text?.startsWith("Stopped after ") && failed.SLOWPT.button === "Retry",
    failed);

  // 21: the margin follows the image rows; a ≥ 8-channel catalog trace draws as a map in the page's colours.
  // (Default on mhd_n1 opens the dock with q since R2; d opens it otherwise.)
  if (!(await js("Dock.open"))) await key("d");
  await until("Dock.open");
  await quiet();
  const mapped = { open: await js("Slice.margin") };
  await js(`Views.setRow("store:MPI66M322D power", { hidden: true }); Views.setRow("store:toroidal n, MPI66M", { hidden: true }); void 0`);
  await quiet();
  mapped.hidden = await js("Slice.margin");
  await js(`void openPicker("ECE")`);
  await until(`[...document.querySelectorAll("#dock-edit .picker-result")].some((r) => r.querySelector("span")?.textContent === "ECE Te, channels 1-40")`);
  await js(`[...document.querySelectorAll("#dock-edit .picker-result")].find((r) => r.querySelector("span")?.textContent === "ECE Te, channels 1-40").querySelector("button").click()`);
  await until(`Views.added.find((a) => a.id === "ece_map")?.entry?.state === "done"`);
  await quiet();
  await frames();
  Object.assign(mapped, await js(`(() => {
    const added = Views.added.find((a) => a.id === "ece_map");
    const c = added.canvas, data = added.entry.data;
    const x0 = Math.max(GUTTER, px(Math.max(data.t0, S.label.window[0], S.view[0]))) + 2;
    const x1 = Math.min(c.clientWidth - right(), px(Math.min(data.t1, S.label.window[1], S.view[1]))) - 2;
    const bytes = window.pixels(c, x0, 2, x1 - x0, c.clientHeight - 4);
    const words = new Uint32Array(new Uint8ClampedArray(bytes).buffer);
    const lut = new Set(S.lut);
    let hit = 0;
    for (const word of words) hit += lut.has(word);
    return { draw: added.draw, h: Math.round(c.getBoundingClientRect().height), share: hit / words.length, margin: Slice.margin,
      channels: data.channels };
  })()`));
  await js("Views.closeEditor(); void 0");
  check(21, "with the dock open image rows get a margin and lose it when hidden; ECE Te added from the picker draws as a 40-channel map in the page's colours and brings the margin back",
    mapped.open > 0 && mapped.hidden === 0 && mapped.draw === "map" && mapped.channels === 40 && mapped.share >= 0.8 && mapped.margin > 0,
    mapped);

  // 22 and 23 (review D2B I1): at 1280×720 the "q and Te" preset keeps the plan's budget, and a
  // dock width saved on a wider screen is capped by this window.
  await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
  await sleep(300);
  await js(`Views.choose("preset:q and Te")`);
  await until(`Views.current?.name === "q and Te" && Dock.open`);
  await quiet();
  await frames();
  const budget = await js(`({ dock: Math.round($("dock").getBoundingClientRect().width), plot: plotWidth(), width: Dock.width,
    stored: Views.layout.dock.width ?? null, margin: Slice.margin, column: $("axis-row").clientWidth })`);
  check(22, "at 1280×720 the q and Te preset opens the dock at the default width (at most 320 + 6 px) and the plot keeps at least 730 px",
    budget.dock <= 326 && budget.dock >= 300 && budget.plot >= 730 && budget.width === null && budget.stored === null, budget);
  await js(`applyView({ ...Views.clone(Views.layout), dock: { ...Views.layout.dock, width: 1100 } })`);
  await quiet();
  await frames();
  const capped = await js(`({ dock: Math.round($("dock").getBoundingClientRect().width), plot: plotWidth(), width: Dock.width,
    vw: window.innerWidth, most: Math.round(Dock.maxGridPx()), margin: Slice.margin })`);
  await send("Emulation.clearDeviceMetricsOverride");
  await sleep(300);
  check(23, "a view with a 1,100 px dock opened at 1280 px is capped (at most 60vw and the window less 820 px): the plot keeps at least 400 px",
    capped.width === 1100 && capped.dock <= 0.6 * capped.vw && Math.abs(capped.dock - capped.most) <= 1 && capped.plot >= 400, capped);

  // R6: the Add row box in the axis row's gutter: catalog matches, any PTDATA name or MDS node
  // offered as is, a bare tag refused with its fix, and Enter adds the pick to the view.
  const typeInto = async (text) => {
    await js(`(() => { const box = $("add-row-input"); box.focus(); box.value = ""; box.dispatchEvent(new Event("input")); })()`);
    for (const ch of text) await key(ch, { blur: false });
    await frames();
  };
  await js("resetView()"); // Default on mhd_n1: beta_N is not in it yet
  await quiet();
  // The fixture's beta_N reads the corpus; give it the MDS node the real catalog lists as its
  // `points` (review fix I4: a typed pointname finds the curated entry).
  await js(`window.stubs.catalog = async (params, request) => {
    const response = await window.realFetch(request.url, { credentials: "same-origin" });
    const body = await response.json();
    for (const entry of body.entries || []) if (entry.id === "betan") entry.points = ["\\\\efit01::top.results.aeqdsk:betan"];
    return window.json(body, response.status);
  }; Views.catalog = null; Views.loadCatalog().then(() => true)`);
  const addBox = { shown: await js(`!$("add-row").hidden && $("add-row").getBoundingClientRect().left < GUTTER`) };
  const listed = () => js(`[...document.querySelectorAll("#add-row-list li")].filter(() => !$("add-row-list").hidden).map((li) => li.firstChild.textContent)`);
  const said = () => js(`[$("add-row-note").hidden, $("add-row-note").className, $("add-row-note").textContent]`);
  await typeInto("beta");
  await until(`!$("add-row-list").hidden`);
  addBox.beta = await listed();
  await typeInto("ECEVS01");
  addBox.ptdata = await js(`[...document.querySelectorAll("#add-row-list li")].map((li) => li.textContent)`);
  await typeInto("\\BETAN");
  addBox.tag = { list: await listed(), note: await said() };
  await typeInto("EFIT01::BETAN");
  addBox.node = await listed();
  await typeInto("\\ZZQ9");
  addBox.bare = await said();
  await typeInto("beta");
  await until(`!$("add-row-list").hidden`);
  await key("Enter", { blur: false });
  await until(`Views.added.some((a) => a.id === "betan")`);
  addBox.added = await js(`({ ids: Views.added.map((a) => a.id), inLayout: Views.layout.rows.some((r) => r.key === "cat:betan"),
    value: $("add-row-input").value })`);
  await typeInto("efit01::betan");
  addBox.known = { list: await listed(), note: await said() };
  await key("Escape", { blur: false });
  await js("delete window.stubs.catalog; Views.catalog = null; void 0");
  check("R6", "the Add row box offers beta_N from the catalog, PTDATA ECEVS01 to fetch; \\BETAN and EFIT01::BETAN find beta_N by its points (no MDS row offered), a bare \\ZZQ9 is refused with its fix; Enter adds beta_N, after which its pointname says it is in the view",
    addBox.shown && addBox.beta.includes("beta_N") && addBox.ptdata.some((t) => t.startsWith("PTDATA ECEVS01")) &&
      same(addBox.tag.list, ["beta_N"]) && addBox.tag.note[0] && same(addBox.node, ["beta_N"]) &&
      same(addBox.bare, [false, "error", "Add the tree, e.g. EFIT01::ZZQ9"]) &&
      addBox.added.ids.includes("betan") && addBox.added.inLayout && addBox.added.value === "" &&
      same(addBox.known.list, []) && same(addBox.known.note, [false, "hint", "beta_N is already in the view."]),
    addBox);

  // detachment: the dock opens with the inversion, EFIT and the cameras.
  const det = {};
  for (const [label, fixed] of [["api10", API10], ["api11", {}]]) {
    await navigate(`#${DET}/170815`, { fixed, reset: true });
    await opened(170815, DET);
    await until(`S.video && S.video.cards.some((c) => c.img?.naturalWidth > 0)`);
    await quiet();
    det[label] = tally(await requests());
  }
  await until(`Dock.nodes.get("slice:inversion")?.note.textContent.includes("TangTV") && Dock.nodes.get("slice:inversion").note.textContent.includes("EFIT02")`);
  await frames();
  const docked = await js(`({ open: Dock.open, hidden: $("dock").hidden, panels: Dock.panels,
    cameras: $("dock").contains($("video-cameras")) && $("dock").contains($("video-controls")),
    top: [...$("video-panel").children].filter((c) => !c.hidden && c.getClientRects().length).map((c) => c.id),
    panel: !$("video-panel").hidden, cursor: Dock.following() })`);
  check("B1", "detachment opens with the dock showing the TangTV inversion and EFIT, then the cameras, moved out of the top panel",
    docked.open && !docked.hidden && same(docked.panels, [{ kind: "slice", id: "inversion", h: 300 }, { kind: "cameras" }]) &&
      docked.cameras && docked.cursor === "video",
    docked);
  const added = Object.keys(det.api11).filter((kind) => (det.api11[kind] || 0) > (det.api10[kind] || 0));
  check("B4", "no camera panel above the rows (the strips and details only), and the default adds only /api/slices, /api/slice and /api/frame",
    docked.top.every((id) => ["detachment-strips", "detachment-details"].includes(id)) && docked.top.includes("detachment-strips") &&
      added.every((kind) => ["slices", "slice", "frame"].includes(kind)) &&
      !Object.keys(det.api11).some((kind) => ["views", "catalog", "signal", "profile", "fetch/status"].includes(kind)),
    { det, added, top: docked.top });

  const drawn2 = await js(`(() => {
    const node = Dock.nodes.get("slice:inversion");
    const canvas = node.canvas, w = canvas.clientWidth, h = canvas.clientHeight;
    const item = [...Dock.slices.get("inversion").items.values()].find((found) => found.record);
    const record = item.record;
    const fit = Dock.fitRZ(record.r_range, record.z_range, w - 6 - 34, h - 20 - 6);
    fit.left += 34;
    fit.top += 6;
    const inked = (r, z, colour) => {
      const ink = window.rgb(colour);
      const [x, y] = Dock.toPixel(fit, r, z);
      const box = window.pixels(canvas, Math.round(x) - 1, Math.round(y) - 1, 3, 3);
      for (let k = 0; k < box.length; k += 4) {
        if (box[k + 3] > 200 && [0, 1, 2].every((c) => Math.abs(box[k + c] - ink[c]) <= 40)) return true;
      }
      return false;
    };
    const [ix, iy] = Dock.toPixel(fit, 1.85, -1.3);
    return { note: node.note.textContent, title: node.name.textContent,
      xpoint: inked(1.4, -1.1, Dock.OVERLAY.xpoint), boundary: inked(1.7, -0.9, Dock.OVERLAY.boundary),
      image: window.pixels(canvas, Math.round(ix), Math.round(iy), 1, 1)[3], r: record.r_range, z: record.z_range };
  })()`);
  check("B2", "the inversion is drawn with the EFIT02 separatrix (orange) and X-point (sky blue) over it, and both are named with their times",
    drawn2.title === "TangTV inversion and EFIT" && /TangTV [\d,]+ ms/.test(drawn2.note) && /EFIT02 [\d,]+ ms/.test(drawn2.note) &&
      drawn2.xpoint && drawn2.boundary && drawn2.image === 255,
    drawn2);

  // 4 (detachment) and B3: an axis click seeks the video, and the slices follow the video cursor.
  const seeks = [];
  const pins4 = await js("Slice.pins");
  for (const [how, t] of [["click", 95], ["seek", 170]]) {
    const since = await mark();
    if (how === "click") await click(await X(t), await axisY());
    else await js(`void seekVideo(${t})`);
    await until(`S.video.time === S.video.times.reduce((b, v) => (Math.abs(v - ${t}) < Math.abs(b - ${t}) ? v : b))`);
    const time = await js("S.video.time");
    const want = { inversion: nearest(INVERSION_MS, time), efit: nearest(EFIT_MS, time) };
    const text = (ms) => ms.toLocaleString("en-US");
    let shown = true;
    try {
      await until(`Dock.nodes.get("slice:inversion").note.textContent.includes("TangTV ${text(want.inversion)} ms") &&
        Dock.nodes.get("slice:inversion").note.textContent.includes("EFIT02 ${text(want.efit)} ms")`, 8000);
    } catch {
      shown = false;
    }
    await quiet();
    const asked = (await requests(since)).filter((r) => r.kind === "slice");
    seeks.push({ how, t, time, times: await js("S.video.times"), want, shown, note: await js(`Dock.nodes.get("slice:inversion").note.textContent`),
      asked: asked.map((r) => r.t_ms), pins: await js("Slice.pins") });
  }
  const fourDet = seeks[0].time === 80 && same(seeks[0].pins, pins4);
  check(4, "a click on the time axis pins A at its time; on detachment it seeks the video to the nearest frame and pins nothing",
    fourAE && fourDet, { ae: axisPin, detachment: seeks[0] });
  check("B3", "the slices follow the video cursor: each seek asks /api/slice at the video time (rounded) and shows the nearest inversion and EFIT slices",
    seeks.every((s) => s.shown && s.asked.length >= 2 && s.asked.every((ms) => ms === String(Math.round(s.time)))) &&
      seeks[0].time === 80 && seeks[1].time === 160 && same(seeks[0].want, { inversion: 100, efit: 80 }) &&
      same(seeks[1].want, { inversion: 150, efit: 190 }),
    seeks);

  // 10 (detachment): #cursor stays the video cursor.
  await move(await X(60), await rowY(0));
  await frames();
  const [dbx, dby] = await barPoint();
  await move(dbx, dby);
  await frames();
  const videoCursor = await js(`({ hidden: $("cursor").hidden, left: parseFloat($("cursor").style.left),
    want: $("axis-row").getBoundingClientRect().left + px(S.video.time), time: S.video.time })`);
  check(10, "#cursor follows the pointer over the rows and goes over the bar with the dock closed; on detachment it stays the video cursor",
    tenAE && !videoCursor.hidden && near(videoCursor.left, videoCursor.want, 1), { ae: cursor, detachment: videoCursor });

  // B5: other events are unchanged.
  const before5 = await mark();
  await js(`void openEvent("${AE}", 170815)`);
  await opened(170815, AE);
  await quiet();
  const back = await js(`({ open: Dock.open, hidden: $("dock").hidden, panels: Dock.panels, video: $("video-panel").hidden,
    cameras: $("video-cameras").parentNode?.id, margin: Slice.margin, right: right() })`);
  back.made = [...new Set((await requests(before5)).map((r) => r.kind))];
  check("B5", "leaving detachment for another event closes the dock, puts the cameras back and asks for no slice",
    !back.open && back.hidden && same(back.panels, []) && back.video && back.cameras === "video-panel" && back.margin === 0 &&
      back.right === 12 && !back.made.some((kind) => FORBIDDEN.includes(kind)),
    back);

  // R2: the dock tab at the window's right edge (the dock's left edge when open) opens and
  // closes the dock; an event with no shot hides it.
  const tabState = () => js(`(() => { const tab = $("dock-tab"), r = tab?.getBoundingClientRect();
    return tab && { hidden: tab.hidden || !r.width, text: tab.textContent, expanded: tab.getAttribute("aria-expanded"), label: tab.getAttribute("aria-label"),
      right: Math.round(r.right), x: r.left + r.width / 2, y: r.top + r.height / 2, open: Dock.open,
      dockLeft: Math.round($("dock").getBoundingClientRect().left), width: window.innerWidth }; })()`);
  const tab = { closed: await tabState() };
  await click(tab.closed.x, tab.closed.y);
  await until("Dock.open");
  await frames();
  tab.opened = await tabState();
  await click(tab.opened.x, tab.opened.y);
  await until("!Dock.open");
  await frames();
  tab.again = await tabState();
  await js(`$("event").value = "${EMPTY}"; $("event").dispatchEvent(new Event("change")); void 0`);
  await until(`S.event === "${EMPTY}" && S.meta === null`);
  await quiet();
  tab.empty = await tabState();
  await js(`void openEvent("${AE}", 170815)`);
  await opened(170815, AE);
  await quiet();
  check("R2", "the dock tab sits at the window's right edge (‹, Open the dock), a click opens the dock with the tab on its left edge (›), a second closes it; no shot, no tab",
    !tab.closed.hidden && tab.closed.text === "‹" && tab.closed.expanded === "false" && tab.closed.label === "Open the dock (D)" &&
      Math.abs(tab.closed.right - tab.closed.width) <= 1 && tab.opened.open && tab.opened.text === "›" &&
      tab.opened.expanded === "true" && Math.abs(tab.opened.right - tab.opened.dockLeft) <= 1 &&
      !tab.again.open && tab.again.text === "‹" && tab.empty.hidden,
    tab);

  // R4: a click on a row's name box selects it; Delete removes it from the view, Ctrl+Z puts it
  // back and the label is untouched; Pop out moves it to the dock and × puts it back.
  const firstKey = await js("Slice.rows()[0].key");
  const label4 = await js("JSON.stringify(S.label)");
  await click(30, await rowY(0));
  await frames();
  const sel = { picked: await js(`[Views.selected, document.querySelectorAll("#rows canvas.row-selected").length, Boolean($("row-select"))]`) };
  await key("Delete");
  await frames();
  sel.removed = await js(`({ rows: Slice.rows().map((r) => r.key), hidden: Views.layout.rows.find((r) => r.key === "${firstKey}")?.hidden ?? null,
    selected: Views.selected, note: Activity.notes.get("view")?.text ?? "" })`);
  await key("z", { modifiers: 2 });
  await frames();
  sel.back = await js(`({ rows: Slice.rows().map((r) => r.key), label: JSON.stringify(S.label) === ${JSON.stringify(label4)} })`);
  await click(30, await rowY(0));
  await frames();
  await js(`[...$("row-select").querySelectorAll("button")].find((b) => b.textContent === "Pop out").click()`);
  await until(`Dock.open && Dock.nodes.has("row:${firstKey}")`);
  await frames();
  sel.popped = await js(`({ panels: Dock.panels, inRows: Slice.rows().some((r) => r.key === "${firstKey}"),
    note: Dock.nodes.get("row:${firstKey}").note.textContent, title: Dock.nodes.get("row:${firstKey}").name.textContent })`);
  await js(`Dock.nodes.get("row:${firstKey}").close.click()`);
  await frames();
  sel.returned = await js(`({ inRows: Slice.rows().some((r) => r.key === "${firstKey}"), panels: Dock.panels })`);
  if (await js("Dock.open")) await key("d");
  await until("!Dock.open");
  check("R4", "a click on a row's name selects it; Delete hides it from the view and Ctrl+Z brings it back (the label kept); Pop out shows it in the dock and its × puts it back",
    sel.picked[0] === firstKey && sel.picked[1] === 1 && sel.picked[2] && !sel.removed.rows.includes(firstKey) &&
      sel.removed.hidden === true && sel.removed.note.startsWith("Removed ") && sel.back.rows[0] === firstKey && sel.back.label &&
      sel.popped.panels.some((p) => p.kind === "row" && p.id === firstKey) && !sel.popped.inRows && sel.popped.title.length > 0 &&
      sel.returned.inRows && !sel.returned.panels.some((p) => p.kind === "row"),
    sel);

  // R7 (review fix I1): one selection. A row and an annotation clear each other, so Delete
  // acts on what was clicked last; Ctrl+Z puts each back.
  const lane7 = await js(`(() => { const r = $("label-track").getBoundingClientRect(); return r.top + r.height / 2; })()`);
  const one = { n: await js("S.label.intervals.length"), at: await js("S.label.intervals.findIndex((s) => s[0] === 1240 && s[1] === 1360)"),
    rows: await js("Slice.rows().map((r) => r.key)") };
  const label7 = await js("JSON.stringify(S.label)");
  await click(await X(1300), lane7);
  await frames();
  await click(30, await rowY(0));
  await frames();
  one.rowLast = await js(`[Views.selected, S.selected]`);
  const key7 = one.rowLast[0];
  await key("Delete");
  await frames();
  one.rowGone = await js(`({ n: S.label.intervals.length, hidden: Views.layout.rows.find((r) => r.key === ${JSON.stringify(key7)})?.hidden ?? null })`);
  await key("z", { modifiers: 2 });
  await frames();
  one.rowBack = await js(`Slice.rows().map((r) => r.key)`);
  await click(30, await rowY(0));
  await frames();
  await click(await X(1300), lane7);
  await frames();
  one.spanLast = await js(`[Views.selected, S.selected, document.querySelectorAll("#rows canvas.row-selected").length, Boolean($("row-select")?.offsetParent)]`);
  await key("Delete");
  await frames();
  one.spanGone = await js(`({ n: S.label.intervals.length, gone: !S.label.intervals.some((s) => s[0] === 1240 && s[1] === 1360),
    rows: Slice.rows().map((r) => r.key) })`);
  await key("z", { modifiers: 2 });
  await frames();
  one.spanBack = await js(`({ n: S.label.intervals.length, same: JSON.stringify(S.label) === ${JSON.stringify(label7)}, undo: S.undo.length })`);
  check("R7", "a row click clears the annotation selection and an annotation click the row's: Delete removes the row (label kept), then the annotation (rows kept); Ctrl+Z restores each",
    one.at >= 0 && key7 && one.rowLast[1] === -1 && one.rowGone.n === one.n && one.rowGone.hidden === true && same(one.rowBack, one.rows) &&
      same(one.spanLast, [null, one.at, 0, false]) && one.spanGone.n === one.n - 1 && one.spanGone.gone && same(one.spanGone.rows, one.rows) &&
      one.spanBack.n === one.n && one.spanBack.same,
    one);

  // R8 (review fix I2): a popped-out added row whose fetch failed shows the entry's chip (its
  // copy, Retry, the server's detail behind a toggle), not a "Loading…" that never ends; Retry
  // fetches it and the chip goes.
  await js(`window.stubs.signal = (p) => p.id === "custom:UPPT"
    ? window.json({ error: "fixture upstream failure", kind: "upstream", detail: "fixture traceback, line 1" }, 502) : null;
    Views.addRow("custom:UPPT"); void 0`);
  await until(`Views.added.find((a) => a.id === "custom:UPPT")?.entry?.state === "error"`);
  const upKey = await js(`Views.added.find((a) => a.id === "custom:UPPT").key`);
  await js(`Dock.popRow(${JSON.stringify(upKey)}); void 0`);
  await until(`Dock.open && Dock.nodes.has("row:" + ${JSON.stringify(upKey)})`);
  await js(`window.tagged = []; window.realTag = Dock.tag; Dock.tag = (g, text, ...rest) => { window.tagged.push(text); return window.realTag(g, text, ...rest); }; Dock.redraw(); void 0`);
  await frames();
  await frames();
  const upChip = `Dock.nodes.get("row:" + ${JSON.stringify(upKey)}).chip`;
  const failedRow = await js(`(() => { const chip = ${upChip};
    return { text: chip?.firstChild.textContent ?? null, button: chip && !chip.lastChild.hidden ? chip.lastChild.textContent : null,
      error: chip?.classList.contains("error") ?? null, toggle: chip?.querySelector(".failure-detail summary")?.textContent ?? null,
      detail: chip?.querySelector(".failure-detail pre")?.textContent ?? null, closed: chip ? !chip.querySelector(".failure-detail").open : null,
      tags: window.tagged.slice() };
  })()`);
  await js(`Dock.tag = window.realTag; delete window.stubs.signal; ${upChip}.lastChild.click(); void 0`);
  await until(`Views.added.find((a) => a.id === "custom:UPPT")?.entry?.state === "done"`, 30000);
  await frames();
  failedRow.retried = await js(`({ chip: !${upChip}, panel: Dock.panels.some((p) => p.kind === "row" && p.id === ${JSON.stringify(upKey)}) })`);
  await js(`Dock.nodes.get("row:" + ${JSON.stringify(upKey)}).close.click(); void 0`);
  await frames();
  await js(`Views.removeKey("custom:UPPT"); void 0`);
  if (await js("Dock.open")) await key("d");
  await until("!Dock.open");
  check("R8", "a popped-out row whose fetch failed (502) shows the chip \"The data server failed: … Try again soon.\" with Retry and its detail behind a closed toggle, never \"Loading…\"; Retry fetches it and the chip goes",
    failedRow.text === "The data server failed: fixture upstream failure. Try again soon." && failedRow.button === "Retry" && failedRow.error &&
      failedRow.toggle === "details" && failedRow.detail === "fixture traceback, line 1" && failedRow.closed &&
      !failedRow.tags.includes("Loading…") && failedRow.retried.chip && failedRow.retried.panel,
    failedRow);

  // R9 (review fix I3): the removed-rows undo belongs to its layout: after Reset (a view
  // choice) or an event change Ctrl+Z puts no row into the new layout.
  await click(30, await rowY(0));
  await frames();
  const undo9 = { key: await js("Views.selected") };
  await key("Delete");
  await frames();
  undo9.removed = await js("Views.removed.length");
  await js("void resetView()");
  await quiet();
  const rows9 = await js("JSON.stringify(Views.layout.rows)");
  const label9 = await js("JSON.stringify(S.label)");
  undo9.afterReset = await js("[Views.removed.length, S.undo.length]");
  await key("z", { modifiers: 2 });
  await frames();
  undo9.same = await js(`[JSON.stringify(Views.layout.rows) === ${JSON.stringify(rows9)}, Views.changed, JSON.stringify(S.label) === ${JSON.stringify(label9)}]`);
  await click(30, await rowY(0));
  await frames();
  await key("Delete");
  await frames();
  undo9.again = await js("Views.removed.length");
  await js(`void openEvent("${MHD}", 180001)`);
  await opened(180001, MHD);
  await quiet();
  undo9.otherEvent = await js("Views.removed.length");
  await js(`void openEvent("${AE}", 170815)`);
  await opened(170815, AE);
  await quiet();
  undo9.back = await js("[Views.removed.length, Slice.rows().some((r) => r.key === " + JSON.stringify(undo9.key) + ")]");
  check("R9", "a row removed before Reset or before an event change stays out of Ctrl+Z: the undo list empties and Ctrl+Z changes neither the new layout nor the label",
    undo9.key && undo9.removed === 1 && same(undo9.afterReset, [0, 0]) && same(undo9.same, [true, false, true]) &&
      undo9.again === 1 && undo9.otherEvent === 0 && undo9.back[0] === 0,
    undo9);
  if (await js("Dock.open")) await key("d");

  // R1: the shot box's search: digits ask /api/shots/search for the event, up to 8 matches
  // show with their lists, ↓ and Enter open the pick; a server without the route (404) leaves
  // the box opening the number typed, with no error. The box sits in the label panel above the
  // shot chips, uncovered with the dock open, its list opens upward on screen, and the bar
  // shows the open shot as text.
  const dockWasOpen = await js("Dock.open");
  if (!dockWasOpen) await key("d");
  await js(`window.stubs["shots/search"] = async (params) => window.json({ shots: [
    { shot: 170815, lists: ["gold", "roster"], cached: true }, { shot: 170816, lists: ["roster"], cached: true },
    ...Array.from({ length: 10 }, (_, k) => ({ shot: 170820 + k, lists: [], cached: false })) ] }); void 0`);
  const since1 = await mark();
  const typeShot = async (text) => {
    await js(`(() => { const box = $("shot"); box.focus(); box.value = ""; })()`);
    for (const ch of text) await key(ch, { blur: false });
  };
  await typeShot("1708");
  await until(`!$("shot-matches").hidden`);
  const search = { asked: (await requests(since1)).filter((r) => r.kind === "shots/search"),
    shown: await js(`[...document.querySelectorAll("#shot-matches li")].map((li) => [li.dataset.shot, li.querySelector("span").textContent, li.querySelector("small").textContent])`),
    bar: await js(`Math.round($("bar").getBoundingClientRect().height)`),
    place: await js(`(() => {
      const box = $("shot").getBoundingClientRect(), list = $("shot-matches").getBoundingClientRect();
      const find = $("shot-find").getBoundingClientRect(), chip = $("queue").firstElementChild.getBoundingClientRect();
      const hit = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
      return { panel: $("bottom").contains($("shot")) && !$("bar").contains($("shot")),
        above: find.bottom <= $("queue").getBoundingClientRect().top + 1,
        aligned: Math.abs($("shot-label").getBoundingClientRect().left - chip.left) <= 2,
        dock: Dock.open, uncovered: hit === $("shot"), right: Math.round(box.right),
        dockLeft: Math.round($("dock").getBoundingClientRect().left),
        up: list.bottom <= box.top, onScreen: list.top >= 0 && list.bottom <= innerHeight };
    })()`) };
  await key("ArrowDown", { blur: false });
  await key("ArrowDown", { blur: false });
  search.active = await js(`$("shot").getAttribute("aria-activedescendant")`);
  await key("Enter", { blur: false });
  await opened(170816, AE);
  search.hash = await js("location.hash");
  search.now = await js(`[$("shot-now").textContent, $("bar").querySelector("input#shot") === null]`);
  search.closed = await js(`$("shot-matches").hidden`);
  await js(`window.stubs["shots/search"] = async () => window.json({ detail: "Not Found" }, 404); void 0`);
  await typeShot("170815");
  await sleep(500);
  await key("Enter", { blur: false });
  await opened(170815, AE);
  await quiet();
  search.old = await js(`({ off: ShotSearch.off, hash: location.hash, hidden: $("shot-matches").hidden, error: $("activity").classList.contains("error") })`);
  check("R1", "typing 1708 in the label panel's box (above the chips, left of the open dock) asks /api/shots/search for the event; 8 of 12 matches show upward on screen with their lists and \"fetch\" for shots not on disk; ↓↓ Enter opens 170816 and the bar says Shot 170816; on a 404 the box still opens the typed shot, quietly",
    search.asked.length >= 1 && search.asked.at(-1).q === "1708" && search.asked.every((r) => r.event === AE && r.limit === "20") && search.shown.length === 8 &&
      same(search.shown[0], ["170815", "gold · roster", ""]) && same(search.shown[2], ["170820", "", "fetch"]) &&
      search.active === "shot-match-1" && search.hash === `#${AE}/170816` && search.closed &&
      search.old.off && search.old.hash === `#${AE}/170815` && search.old.hidden && !search.old.error &&
      search.place.panel && search.place.above && search.place.aligned && search.place.dock && search.place.uncovered &&
      search.place.right < search.place.dockLeft && search.place.up && search.place.onScreen &&
      same(search.now, ["Shot 170816", true]),
    search);
  if (!dockWasOpen && (await js("Dock.open"))) await key("d");

  // R3: a detachment shot without camera frames (as 170659): the slices follow the rows'
  // cursor, never stay "Loading…", and show the TangTV inversion and EFIT at the cursor.
  await js(`window.stubs.shot = async (params, request) => {
    if (params.event !== "${DET}") return null;
    const response = await window.realFetch(request.url, { credentials: "same-origin" });
    const body = await response.json();
    for (const camera of body.video?.cameras || []) Object.assign(camera, { channels: [], views: [], reason: "camera group missing" });
    return window.json(body, response.status);
  }; void 0`);
  await js(`void openEvent("${DET}", 170815)`);
  await opened(170815, DET);
  await quiet();
  await frames();
  const efit = { before: await js(`({ led: Dock.videoLed(), following: Dock.following(), open: Dock.open,
    note: Dock.nodes.get("slice:inversion")?.note.textContent ?? null, kinds: (Dock.kinds || []).map((k) => k.kind) })`) };
  const since3 = await mark();
  await move(await X(95), await rowY(0));
  try {
    await until(`Dock.nodes.get("slice:inversion").note.textContent.includes("TangTV 100 ms") &&
      Dock.nodes.get("slice:inversion").note.textContent.includes("EFIT02 80 ms")`, 8000);
    efit.shown = true;
  } catch {
    efit.shown = false;
  }
  await quiet();
  efit.after = await js(`({ note: Dock.nodes.get("slice:inversion").note.textContent, time: Dock.time() })`);
  efit.asked = (await requests(since3)).filter((r) => r.kind === "slice").map((r) => [r.slice, r.t_ms]);
  const [r3x, r3y] = await barPoint();
  await move(r3x, r3y);
  check("R3", "with no camera frames the detachment dock follows the rows' cursor: it asks /api/slice at the cursor (95 ms) and shows TangTV 100 ms and EFIT02 80 ms, never stuck on Loading",
    !efit.before.led && efit.before.following === "cursor" && efit.before.open && !/Loading/.test(efit.before.note || "") &&
      efit.shown && efit.asked.length >= 2 && efit.asked.every(([, t]) => Math.abs(Number(t) - 95) <= 1),
    efit);
  await js("delete window.stubs.shot; void 0");

  check("E", "no script error", errors.length === 0, errors);
}

/** A saved "q and Te" view on mhd_n1, photographed at three window sizes. */
async function shots() {
  await navigate(`#${MHD}/180001`, { reset: true });
  await opened(180001, MHD);
  await quiet();
  await js(`Views.choose("preset:q and Te")`);
  await until(`Views.added.length > 0 && Views.added.every((a) => a.hidden || a.entry?.state === "done") &&
    Dock.panels.filter((p) => p.kind === "profile").every((p) => Signals.entry(Profiles.shots, "profile", S.shot, p.id).state === "done")`, 60000);
  await quiet();
  await js("openEditor(); void 0");
  await js(`$("view-name").value = "My q and Te"; void 0`);
  await footer("Save as");
  await until(`Views.current?.name === "My q and Te" && !Views.changed`);
  await footer("Done");
  await js("Slice.setPins({ A: 1200, B: 2600 }); void 0");
  const facts = {};
  for (const [width, height] of [[1920, 1080], [1440, 900], [1280, 720]]) {
    await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
    await sleep(600);
    await quiet();
    await move(await X(1800), await js(`(() => { const r = $("rows").children[0].getBoundingClientRect(); return r.top + 40; })()`));
    await frames();
    await sleep(200);
    const { data } = await send("Page.captureScreenshot", { format: "png" });
    writeFileSync(`${CASE}/d2_${width}x${height}.png`, Buffer.from(data, "base64"));
    facts[`${width}x${height}`] = await js(`({ view: Views.label(), dock: Math.round($("dock").getBoundingClientRect().width), margin: Slice.margin,
      rows: [...$("rows").children].filter((c) => !c.classList.contains("view-hidden")).map((c) => [c.dataset.signal || S.meta.rows[[...$("rows").children].indexOf(c)]?.title, Math.round(c.getBoundingClientRect().height)]),
      panels: Dock.panels.map((p) => p.id || p.kind), notes: [...document.querySelectorAll("#dock .dock-panel-note")].map((n) => n.textContent),
      reading: document.querySelector("#dock .dock-reading")?.textContent, pins: Slice.pins, hover: Slice.hover })`);
  }
  await send("Emulation.clearDeviceMetricsOverride");
  facts.errors = errors;
  return facts;
}

let output;
try {
  if (SCENARIO === "shots") output = await shots();
  else {
    await checks();
    output = ORDER.filter((n) => results.has(n)).map((n) => results.get(n));
  }
} catch (error) {
  console.log(JSON.stringify(ORDER.filter((n) => results.has(n)).map((n) => results.get(n))));
  console.error(error.stack);
  console.error(JSON.stringify(errors));
  process.exit(1);
}

console.log(JSON.stringify(output));
ws.close();
browser.kill();
if (errors.length) {
  console.error(JSON.stringify(errors));
  process.exit(1);
}
