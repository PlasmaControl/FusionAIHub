// The review page in headless Chromium, driven over the DevTools protocol.
//   node review_browser.mjs <base-url> <token> <headless-shell> <profile-dir> [api1|race|moves|empty|inflight] [case]
// Prints one JSON line: every check made, [{name, ok, detail}].
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

const [BASE, TOKEN, SHELL, PROFILE, SCENARIO, CASE] = process.argv.slice(2);
const env = { ...process.env };
delete env.DISPLAY;
delete env.WAYLAND_DISPLAY;
const browser = spawn(
  SHELL,
  ["--no-sandbox", "--disable-gpu", "--remote-debugging-port=0", `--user-data-dir=${PROFILE}`,
    "--window-size=1400,900", "about:blank"],
  { stdio: "ignore", env }
);
process.on("exit", () => browser.kill());

async function within(promise, what) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, fail) => {
      timer = setTimeout(() => fail(new Error(`${what} did not answer within 15 s`)), 15000);
    })]);
  } finally {
    clearTimeout(timer);
  }
}

const page = await within((async () => {
  let port;
  while (!port) {
    try {
      port = readFileSync(`${PROFILE}/DevToolsActivePort`, "utf8").split("\n")[0];
    } catch {
      await sleep(50);
    }
  }
  const response = await fetch(`http://127.0.0.1:${port}/json/list`);
  const page = (await response.json()).find((t) => t.type === "page");
  if (!page) throw new Error("browser DevTools endpoint returned no page");
  return page;
})(), "browser DevTools endpoint");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await within(new Promise((open, fail) => {
  ws.addEventListener("open", open);
  ws.addEventListener("error", () => fail(new Error("browser DevTools WebSocket did not open")));
}), "browser DevTools WebSocket");

let id = 0;
const pending = new Map();
const errors = [];
ws.addEventListener("message", ({ data }) => {
  const msg = JSON.parse(data);
  if (pending.has(msg.id)) {
    const [ok, fail] = pending.get(msg.id);
    pending.delete(msg.id);
    if (msg.error) fail(new Error(msg.error.message));
    else ok(msg.result);
  } else if (msg.method === "Runtime.exceptionThrown") {
    errors.push(msg.params.exceptionDetails.exception?.description);
  }
});
const send = (method, params = {}) =>
  new Promise((ok, fail) => {
    pending.set(++id, [ok, fail]);
    ws.send(JSON.stringify({ id, method, params }));
  });

async function js(expression) {
  const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description);
  return r.result.value;
}

async function until(expression, timeout = 20000) {
  for (const start = Date.now(); Date.now() - start < timeout; await sleep(50)) {
    if (await js(expression)) return;
  }
  throw new Error(`timed out waiting for ${expression}`);
}

const opened = (shot) => until(`typeof S !== "undefined" && S.shot === ${shot} && S.data !== null`);
const mouse = (type, x, y, more = {}) =>
  send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, ...more });

async function drag(x0, x1, y, modifiers = 0) {
  await mouse("mouseMoved", x0, y, { button: "none", modifiers });
  await mouse("mousePressed", x0, y, { buttons: 1, modifiers });
  for (let i = 1; i <= 8; i++) await mouse("mouseMoved", x0 + ((x1 - x0) * i) / 8, y, { buttons: 1, modifiers });
  await mouse("mouseReleased", x1, y, { buttons: 0, modifiers });
}

/** Draw a span over t0-t1 ms on the label track. */
async function draw(t0, t1) {
  const [x0, x1, y] = await js(`(() => {
    const r = $("label-track").getBoundingClientRect();
    return [r.left + px(${t0}), r.left + px(${t1}), r.top + 15];
  })()`);
  await drag(x0, x1, y);
}

/** A key press; `modifiers` is CDP's bit set: 2 is Ctrl, 8 is Shift. */
async function press(key, modifiers = 0) {
  const named = { Enter: [13, "\r"], Escape: [27, ""], ArrowLeft: [37, ""], ArrowRight: [39, ""],
    Delete: [46, ""], Backspace: [8, ""], " ": [32, " "] }[key];
  const [code, text] = named || [key.toUpperCase().charCodeAt(0), key];
  const event = { key, code: named ? key : `Key${key.toUpperCase()}`, windowsVirtualKeyCode: code, modifiers };
  await send("Input.dispatchKeyEvent", { type: "keyDown", ...event, ...(modifiers || !text ? {} : { text }) });
  await send("Input.dispatchKeyEvent", { type: "keyUp", ...event });
}

const checks = [];
const check = (name, ok, detail) => checks.push({ name, ok: Boolean(ok), detail });
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

async function currentServer() {
  const source = await js("S.meta.source");
  check("the link opens its shot on its source label", same(await js("S.label"), source), source);

  await js(`$("reviewer-name").focus()`);
  await send("Input.insertText", { text: "Ada Lovelace" });
  await press("Enter");
  check(
    "the name box keeps the reviewer's name and gives the keys back",
    await js(`localStorage.getItem("labeler:name") === "Ada Lovelace" && document.activeElement === document.body`),
    await js(`[localStorage.getItem("labeler:name"), document.activeElement.id]`)
  );

  await draw(500, 800);
  check("a drag on the label adds a span", (await js("S.label.intervals.length")) === 2, await js("S.label"));
  check(
    "the edit shows as unsaved and is kept as a draft",
    await js(`!$("dirty").hidden && localStorage.getItem("labeler:alfven_eigenmode:170815") !== null`)
  );
  await press("z", 2);
  check("ctrl-z undoes it", same(await js("S.label"), source) && (await js(`$("dirty").hidden`)));

  await draw(500, 800);
  await press("s");
  await until("S.meta.saved !== null && !S.saving");
  const saved = await js("S.label");
  check(
    "s saves it and says who saved it",
    await js(`$("state").textContent === "changed" && /^saved .* by Ada Lovelace$/.test($("saved").textContent)`),
    await js(`[$("state").textContent, $("saved").textContent]`)
  );

  await send("Page.reload");
  await opened(170815);
  check(
    "a reload opens the saved label and the name",
    same(await js("S.label"), saved) &&
      (await js(`$("count").textContent === "2/3" && $("reviewer-name").value === "Ada Lovelace"`)),
    await js(`[S.label, $("count").textContent, $("reviewer-name").value]`)
  );

  const [x, y, before] = await js(`(() => {
    const r = $("top").getBoundingClientRect();
    return [r.left + px(1000), r.top + 60, S.view[1] - S.view[0]];
  })()`);
  await send("Input.dispatchMouseEvent", { type: "mouseWheel", x, y, deltaX: 0, deltaY: -400, modifiers: 2 });
  await until(`S.view[1] - S.view[0] < ${before / 2} && S.data.t1 - S.data.t0 < ${before}`);
  check("ctrl-wheel zooms in and fetches the rows it shows", true, await js("[S.view, S.data.n]"));

  // Out again, drawn before any rows can be asked for: the overview fills what the zoomed ones miss.
  const pixel = await js(`new Promise((done) => {
    S.view = [S.overview.t0, S.overview.t1];
    render();
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const t = S.overview.t0 + 0.03 * (S.overview.t1 - S.overview.t0);
      const canvas = $("rows").children[0];
      const x = Math.round((px(t) * canvas.width) / canvas.clientWidth);
      done([t < S.data.t0, [...canvas.getContext("2d").getImageData(x, 20, 1, 1).data]]);
    }));
  })`);
  check("a zoom-out shows the overview until its own rows arrive", pixel[0] && pixel[1][3] > 0, pixel);

  await draw(1200, 1400);
  await press("s");
  await until(`!S.saving && $("dirty").hidden`);
  await press("h");
  await until(`$("versions").open`);
  const listed = await js(`[...$("version-list").children].map((item) => item.textContent)`);
  check(
    "h lists every saved version, newest first, with who saved it",
    listed.length === 2 && listed[0].startsWith("v2") && listed.every((text) => text.includes("Ada Lovelace")),
    listed
  );
  await press("Escape");
  await until(`!$("versions").open`);
  check("escape shuts the list and gives the keys back", await js(`document.activeElement === document.body`),
    await js(`document.activeElement.outerHTML`));
  await press("h");
  await until(`$("versions").open`);
  await js(`$("version-list").querySelector('[data-version="1"]').click()`);
  check(
    "restore loads a version as an unsaved draft",
    same(await js("S.label"), saved) && (await js(`!$("versions").open && !$("dirty").hidden`)),
    await js("S.label")
  );

  // 170816 is saved already, so the next unreviewed shot would be 170817.
  await press("Enter");
  await opened(170816);
  check("enter saves and opens the next shot in the queue", (await js(`$("next-shot").textContent`)) === "→ 170817");

  await press("ArrowRight");
  await opened(170817);
  check("→ opens the next shot", true);
  await press("ArrowLeft");
  await opened(170816);
  check("← opens the previous shot", true);
  const start = await js("zoomAt(middle(), 0.5), S.view[0]");
  await press("ArrowRight", 8);
  await until(`S.view[0] > ${start}`);
  check("shift-→ pans instead", (await js("S.shot")) === 170816, await js("[S.shot, S.view]"));

  await js(`$("shot").focus(); $("shot").value = "1"`);
  await press("Enter");
  await until(`$("rows").querySelector(".failure") !== null`);
  check("a shot off the roster says why", await js(`$("rows").textContent.includes("not on this event")`),
    await js(`$("rows").textContent`));
  await js("document.activeElement.blur()");
  await press("ArrowRight");
  await opened(170815);
  check("→ carries on from it", true);
}

// Hold real responses at a chosen boundary; requests and saves still reach the fixture.
async function recordFetches(holdVersion = false) {
  await send("Page.enable");
  await send("Page.addScriptToEvaluateOnNewDocument", { source: `
    window.requests = [];
    window.saves = [];
    window.gates = ${holdVersion ? '{ version: {} }' : '{}'};
    const realFetch = window.fetch.bind(window);
    window.fetch = async (input, options) => {
      const request = new Request(input, options);
      const url = new URL(request.url);
      const kind = url.pathname.split("/").pop();
      const params = Object.fromEntries(url.searchParams);
      window.requests.push({ kind, ...params });
      if (kind === "label" && request.method === "POST") {
        window.saves.push(JSON.parse(await request.clone().text()));
      }
      const gate = window.gates[kind];
      const held = gate && !gate.released &&
        (!gate.event || gate.event === params.event) && (!gate.shot || gate.shot === params.shot);
      const response = await realFetch(input, options);
      if (held) {
        gate.waiters ||= [];
        if (!gate.released) await new Promise((done) => gate.waiters.push(done));
      }
      return response;
    };
  ` });
}

const arm = (kind, event, shot) => js(`window.gates[${JSON.stringify(kind)}] = {
  event: ${JSON.stringify(event)}, shot: ${JSON.stringify(shot == null ? null : String(shot))}
}; void 0`);
const held = (kind) => until(`window.gates.${kind}.waiters?.length > 0`);
const release = (kind) => js(`window.gates.${kind}.released = true;
  window.gates.${kind}.waiters?.forEach((done) => done())`);
const drafts = () => js(`Object.fromEntries(Object.entries(localStorage).filter(([key]) => /^labeler:.*:\\d+$/.test(key)))`);
const settled = (event, shot) => until(`S.event === "${event}" && S.shot === ${shot} &&
  S.meta?.event === "${event}" && S.data !== null`);
const visit = async (event, shot) => {
  await js(`void openEvent("${event}", ${shot})`);
  await settled(event, shot);
};

async function navigationRace() {
  const a = "alfven_eigenmode", b = "neoclassical_tearing_mode";
  await held("version");
  check("name is hidden while API version is pending", await js(`$("reviewer-name").hidden`));
  check("History is hidden while API version is pending", await js(`$("show-versions").hidden`));
  await release("version");
  await opened(170815);
  check("API 2 reveals name and History", await js(`!$("reviewer-name").hidden && !$("show-versions").hidden`));

  await draw(1200, 1400);
  await press("h");
  await until(`$("versions").open`);
  await js(`window.oldRestore = $("version-list").querySelector("button")`);
  await arm("shot", a, 170816);
  await js(`void openShot(170816)`);
  check("starting a shot closes its open History", await js(`!$("versions").open && S.versions.length === 0`));
  await held("shot");
  await js(`closeDialog($("versions"))`); // keep testing refusals even on the broken page
  const histories = await js(`window.requests.filter((r) => r.kind === "history").length`);
  await press("h");
  await sleep(100);
  check("H during opening neither opens nor requests History", await js(`!$("versions").open &&
    window.requests.filter((r) => r.kind === "history").length === ${histories}`));
  await js(`closeDialog($("versions"))`);
  const before = await js("S.label"), kept = await drafts();
  await draw(1500, 1700);
  check("drawing during opening preserves label and drafts", same(await js("S.label"), before) && same(await drafts(), kept));
  check("an edit refused during opening explains why", await js(`$("status").textContent.includes("still opening")`));
  for (const action of ["edit([0, 2000], [[700, 800, 1]])", "undo()", "touch()"]) {
    await js(action);
    check(`${action} cannot change a pending shot or draft`,
      same(await js("S.label"), before) && same(await drafts(), kept));
  }
  await press("s");
  await sleep(100);
  check("S during opening posts nothing", (await js("window.saves.length")) === 0);
  await press("Enter");
  await js(`$("save-next").click(); document.activeElement.blur()`);
  check("Enter and Save during opening post nothing", (await js("window.saves.length")) === 0);
  await press("r");
  await js(`$("revert").click()`);
  check("Revert during opening preserves label and drafts",
    same(await js("S.label"), before) && same(await drafts(), kept));
  await release("shot");
  await opened(170816);
  await until("!S.saving");
  check("170816 opens with its own saved label", same(await js("S.label.intervals"), [[400, 600, 1]]));
  const target = await js("S.label"), targetDrafts = await drafts();
  await js("window.oldRestore.click(); restoreVersion(1)");
  check("a stale Restore cannot change the new shot or store a draft",
    same(await js("S.label"), target) && same(await drafts(), targetDrafts));
  await press("s");
  await until("!S.saving");
  const saved = await js("window.saves.at(-1)");
  check("Save posts 170816 with its own label", saved?.event === a && saved?.shot === 170816 &&
    same(saved?.intervals, [[400, 600, 1]]), saved);

  await visit(a, 170815);
  await arm("history", a, 170815);
  await press("h");
  await held("history");
  await press("ArrowRight");
  await opened(170816);
  await release("history");
  await sleep(100);
  check("History arriving after navigation opens no dialog", await js(`!$("versions").open && S.versions.length === 0`));
  await js(`closeDialog($("versions"))`);

  for (const route of ["address", "menu"]) {
    await visit(a, 170815);
    await press("h");
    await until(`$("versions").open`);
    await js(`window.oldRestore = $("version-list").querySelector("button")`);
    if (route === "menu") await press("Escape");
    await arm("shot", b, 170815);
    if (route === "address") await js(`location.hash = "#${b}/170815"`);
    else await js(`$("event").value = "${b}"; $("event").dispatchEvent(new Event("change"))`);
    await held("shot");
    check(`${route}: switching event closes and invalidates History`, await js(`!$("versions").open && S.versions.length === 0`));
    await js(`closeDialog($("versions")); document.activeElement.blur()`);
    const label = await js("S.label"), stored = await drafts(), posts = await js("window.saves.length");
    await draw(1500, 1700);
    await press("z", 2);
    await press("r");
    await press("s");
    await sleep(100);
    check(`${route}: the same shot in a pending event cannot be edited or saved`,
      same(await js("S.label"), label) && same(await drafts(), stored) && (await js("window.saves.length")) === posts);
    const requests = await js(`window.requests.filter((r) => r.kind === "history").length`);
    await press("h");
    await sleep(100);
    check(`${route}: H cannot request the new event's history while opening`, await js(`!$("versions").open &&
      window.requests.filter((r) => r.kind === "history").length === ${requests}`));
    await release("shot");
    await settled(b, 170815);
    check(`${route}: the new event opens its own 170815 label`, same(await js("S.label.intervals"), [[900, 1100, 1]]));
    const own = await js("S.label"), ownDrafts = await drafts();
    await js("window.oldRestore.click(); restoreVersion(1)");
    check(`${route}: A's version cannot be restored into B's same shot`,
      same(await js("S.label"), own) && same(await drafts(), ownDrafts));
  }

  // A queue response must not undo a newer event or shot choice.
  await visit(a, 170815);
  await arm("queue", b);
  await js(`void openEvent("${b}", 170815)`);
  await held("queue");
  await visit(a, 170816);
  await release("queue");
  await sleep(100);
  check("a late event queue cannot replace a newer event", await js(`S.event === "${a}" && S.shot === 170816 && S.queue.length === 3`));
  await arm("queue", b);
  await js(`void openEvent("${b}", 170815)`);
  await held("queue");
  await js("void openShot(170815)");
  await settled(b, 170815);
  const requests = await js(`window.requests.filter((r) => r.kind === "shot").length`);
  await release("queue");
  await sleep(100);
  check("a late event queue cannot open over a newer shot", await js(`S.event === "${b}" && S.shot === 170815 &&
    window.requests.filter((r) => r.kind === "shot").length === ${requests}`));
}

async function inflight() {
  if (CASE === "queue") return await eventQueueInflight();
  const a = "alfven_eigenmode", key = `labeler:${a}:170815`;
  const original = await js("S.label");
  if (CASE === "restore") {
    await press("s");
    await until("!S.saving && S.meta.saved !== null");
  }
  // Observe every status change, including messages cleared by the next navigation.
  await js(`window.statuses = [];
    new MutationObserver(() => window.statuses.push($("status").textContent))
      .observe($("status"), { childList: true, subtree: true, characterData: true })`);
  await draw(500, 800);
  const sent = await js("S.label");
  await arm("label");
  await press(CASE === "next" ? "Enter" : "s");
  await held("label");

  if (["move", "next", "return"].includes(CASE)) {
    await press("ArrowRight");
    await opened(170816);
    if (CASE === "return") {
      await press("ArrowLeft");
      await opened(170815);
    }
  }
  if (["edit", "return"].includes(CASE)) await draw(1200, 1400);
  if (CASE === "restore") {
    await press("h");
    await until(`$("versions").open`);
    await js(`$("version-list").querySelector('[data-version="1"]').click()`);
    check("Restore loads the older version while the save answer is held",
      same(await js("S.label"), original));
  }
  const later = await js("S.label"), undo = await js("S.undo");
  const requests = await js(`window.requests.filter((r) => r.kind === "shot").length`);
  await release("label");
  await until("!S.saving");

  const versions = await js(`fetch("/api/history?event=${a}&shot=170815").then((r) => r.json())`);
  const saved = versions.versions.at(-1);
  check("the server saved the submitted label, without any later edit or Restore",
    same({ window: saved.window, intervals: saved.intervals }, sent), saved);

  if (["move", "next"].includes(CASE)) {
    check("moving during a save never reports that shot as not saved",
      await js(`!window.statuses.some((text) => text.includes("not saved"))`), await js("window.statuses"));
    check("a moved save clears its submitted draft", await js(`localStorage.getItem("${key}") === null`));
    check("the saved shot's chip loses its unsaved mark",
      await js(`!$("queue").querySelector('[data-shot="170815"]').classList.contains("dirty")`));
    check("the saved shot's queue row and reviewed count update after moving",
      await js(`S.queue.find((r) => r.shot === 170815).state === "changed" && $("count").textContent === "2/3"`),
      await js(`[S.queue, $("count").textContent]`));
    check("a save answer does not move again or replace the new shot's label",
      await js(`S.shot === 170816 && window.requests.filter((r) => r.kind === "shot").length === ${requests}`) &&
      same(await js("S.label.intervals"), [[400, 600, 1]]));
    await press("ArrowRight");
    await opened(170817);
    await press("u");
    await until(`$("status").textContent === "all reviewed" || S.shot === 170815`);
    check("U skips the shot whose save finished after moving", (await js("S.shot")) === 170817);
    await send("Page.reload");
    await until(`typeof window.statuses === "undefined" && typeof S !== "undefined" &&
      S.data !== null && !pendingNavigation()`);
    check("a reload keeps the saved shot free of an unsaved mark",
      await js(`localStorage.getItem("${key}") === null &&
        !$("queue").querySelector('[data-shot="170815"]').classList.contains("dirty")`));
  } else {
    check("the later edit or Restore remains on screen", same(await js("S.label"), later), await js("S.label"));
    check("the later label remains in localStorage as a draft",
      same(await js(`JSON.parse(localStorage.getItem("${key}"))`), later));
    check("the later label is marked dirty against the completed save",
      await js(`!$("dirty").hidden && $("queue").querySelector('[data-shot="170815"]').classList.contains("dirty")`));
    check("the save updates the displayed baseline and timestamp",
      same(await js("S.meta.saved"), sent) && await js(`Boolean(S.meta.last_save) && $("saved").textContent.startsWith("saved ")`));
    check("the later edit's undo stack is preserved", same(await js("S.undo"), undo));
    await press("z", 2);
    check("Ctrl+Z still undoes the later edit or Restore to the submitted label",
      same(await js("S.label"), sent) && await js(`$("dirty").hidden && localStorage.getItem("${key}") === null`));
  }
}

async function eventQueueInflight() {
  const b = "neoclassical_tearing_mode";
  await arm("queue", b);
  await js(`$("event").value = "${b}"; $("event").dispatchEvent(new Event("change"))`);
  await held("queue");
  check("changing events empties the previous roster in state and on screen",
    await js(`S.queue.length === 0 && $("queue").children.length === 0`), await js("S.queue"));
  check("a pending event queue shows neither the previous count nor next-shot hint",
    await js(`$("count").textContent === "0/0" && $("next-shot").textContent === ""`));
  await js(`$("shot").focus(); $("shot").value = "170815"`);
  await press("Enter");
  await settled(b, 170815);
  await js("document.activeElement.blur()");
  check("the typed shot opens its own event's label while the queue is held",
    same(await js("S.label.intervals"), [[900, 1100, 1]]));
  check("opening a shot before its queue arrives never prefetches another event's roster",
    await js(`window.requests.filter((r) => r.kind === "shot" && r.event === "${b}")
      .every((r) => r.shot === "170815")`));
  const ticket = await js("S.ticket");
  const requests = await js(`window.requests.filter((r) => r.kind === "shot").length`);
  await release("queue");
  await sleep(100);
  check("the delayed queue belongs to the event even after a newer shot navigation",
    same(await js("S.queue.map((r) => r.shot)"), [170815]), await js("S.queue"));
  check("the delayed event queue refreshes its reviewed count",
    await js(`$("count").textContent === "1/1"`), await js(`$("count").textContent`));
  check("the next-shot hint stays on the event's one-shot roster",
    await js(`$("next-shot").textContent === ""`), await js(`$("next-shot").textContent`));
  check("accepting a delayed queue does not reopen the chosen shot",
    await js(`S.ticket === ${ticket} &&
      window.requests.filter((r) => r.kind === "shot").length === ${requests}`));
  await press("Enter");
  await until("window.saves.length === 1 && !S.saving && !pendingNavigation()");
  check("Save-and-next wraps within the event's roster",
    await js(`S.event === "${b}" && S.shot === 170815 && S.meta?.event === "${b}"`),
    await js(`({ event: S.event, shot: S.shot, meta: S.meta, status: $("status").textContent })`));
  check("every navigation and prefetch in the new event stays on its roster",
    await js(`window.requests.filter((r) => r.kind === "shot" && r.event === "${b}")
      .every((r) => r.shot === "170815")`));
}

async function moves() {
  const fetched = await js(`window.requests.filter((r) => r.kind === "shot").map((r) => r.shot)`);
  check("opening 170815 prefetches its next queue shot, the saved 170816",
    same(fetched, ["170815", "170816"]), fetched);
  await press("ArrowRight");
  await opened(170816);
  await draw(1500, 1700);
  const edited = await js("S.label"), kept = await drafts();
  await press("h");
  await until(`$("versions").open`);
  check("History opens with focus on Close", await js(`document.activeElement.textContent === "Close" &&
    $("versions").contains(document.activeElement)`));
  await press("Enter");
  check("Enter closes History and keeps the unsaved edit and draft", await js(`!$("versions").open && !$("dirty").hidden`) &&
    same(await js("S.label"), edited) && same(await drafts(), kept));
  check("Enter in History posts nothing", (await js("window.saves.length")) === 0);
  const versions = await js(`fetch("/api/history?event=alfven_eigenmode&shot=170816").then((r) => r.json())`);
  check("Enter in History adds no history line", versions.versions.length === 1);

  await draw(1200, 1400);
  const replaced = await js("S.label");
  await press("h");
  await until(`$("versions").open`);
  await js(`$("version-list").querySelector("button").click()`);
  check("Restore says it replaced an unsaved edit and Ctrl+Z brings it back", await js(`
    $("status").textContent.includes("unsaved edit") && $("status").textContent.includes("Ctrl+Z")`),
    await js(`$("status").textContent`));
  await press("z", 2);
  check("Ctrl+Z brings back the edit replaced by Restore", same(await js("S.label"), replaced) &&
    (await js(`!$("dirty").hidden && localStorage.getItem("labeler:alfven_eigenmode:170816") !== null`)));

  await press("h");
  await until(`$("versions").open`);
  await js(`$("version-list").querySelector("button").focus()`);
  await press("Enter");
  check("Enter on a focused Restore still closes without restoring", await js(`!$("versions").open && !$("dirty").hidden`) &&
    same(await js("S.label"), replaced));
  await press("h");
  await until(`$("versions").open`);
  await js(`$("version-list").querySelector("button").focus()`);
  await press(" ");
  check("Space on a focused Restore loads the saved label", await js(`!$("versions").open && $("dirty").hidden`) &&
    same(await js("S.label.intervals"), [[400, 600, 1]]));

  const undoCount = await js("S.undo.length");
  await press("h");
  await until(`$("versions").open`);
  await js(`$("version-list").querySelector("button").click()`);
  check("Restore of the current label stays clean and does not invite a save", await js(`
    $("dirty").hidden && localStorage.getItem("labeler:alfven_eigenmode:170816") === null &&
    S.undo.length === ${undoCount} && $("status").textContent.includes("current label") &&
    !$("status").textContent.includes("saves it")`), await js(`$("status").textContent`));
  await press("h");
  await until(`$("versions").open`);
  await press("h");
  check("H closes History and gives the keys back", await js(`!$("versions").open && document.activeElement === document.body`));
  await press("h");
  await until(`$("versions").open`);
  await press("Escape");
  check("Escape closes History and gives the keys back", await js(`!$("versions").open && document.activeElement === document.body`));

  await press("ArrowLeft");
  await opened(170815);
  await draw(1200, 1400);
  const draft = await js("S.label");
  await press("ArrowRight");
  await opened(170816);
  check("leaving an edit with → says it is kept, not saved", await js(`
    $("status").textContent === "170815: the edit is kept as a draft, not saved"`));
  check("the queue marks the shot with its unsaved draft", await js(`
    $("queue").querySelector('[data-shot="170815"]').classList.contains("dirty") &&
    localStorage.getItem("labeler:alfven_eigenmode:170815") !== null`));
  await press("ArrowLeft");
  await opened(170815);
  check("← brings back the unsaved edit", same(await js("S.label"), draft) && (await js(`!$("dirty").hidden`)));
  await press("u");
  await opened(170817);
  check("U skips saved 170816 to the next unreviewed shot in queue order", true);
  await press("k");
  await opened(170815);
  await press("j");
  await opened(170817);
  check("J on the first shot wraps to the last", true);
  await press("k");
  await opened(170815);
  check("K on the last shot wraps to the first", true);
  await press("j");
  await opened(170817);
  await press("Enter");
  await opened(170815);
  const saves = await js("window.saves");
  check("Enter saves the last shot and wraps to the first with its draft intact",
    saves.length === 1 && saves[0].shot === 170817 && same(saves[0].intervals, []) &&
    same(await js("S.label"), draft) && (await js(`!$("dirty").hidden`)), saves);
}

async function emptyEvent() {
  const a = "alfven_eigenmode", b = "neoclassical_tearing_mode";
  await draw(500, 800);
  const edited = await js("S.label"), kept = await drafts();
  check("the previous shot has a label, an undo and a selection to clear", await js(`
    S.label.intervals.length === 2 && S.undo.length > 0 && S.selected >= 0 &&
    $("tier").textContent === "gold" && $("saved").textContent.startsWith("saved ")`));

  await arm("queue", b);
  await js(`$("event").value = "${b}"; $("event").dispatchEvent(new Event("change"))`);
  await held("queue");
  // A paint queued for the old shot must not run after the empty queue arrives.
  await js(`render(); window.gates.queue.released = true;
    window.gates.queue.waiters.forEach((done) => done())`);
  await until(`S.event === "${b}" && S.queue.length === 0`);
  await js("new Promise(requestAnimationFrame)");
  check("the empty roster is a usable event", await js(`
    S.events.some((row) => row.event === "${b}" && row.n_shots === 0 && !row.error)`));
  check("an empty queue clears the previous shot and all label state", await js(`
    [S.shot, S.meta, S.data, S.overview, S.label].every((value) => value === null) &&
    S.undo.length === 0 && S.selected === -1`), await js(`
    ({shot: S.shot, hasMeta: !!S.meta, hasData: !!S.data, hasOverview: !!S.overview,
      label: S.label, undo: S.undo.length, selected: S.selected})`));
  check("the rows show only the empty-event note", await js(`
    $("rows").children.length === 1 && $("rows").firstElementChild.className === "failure" &&
    $("rows").textContent === "${b} has no shots to review."`), await js(`$("rows").textContent`));
  check("the old shot's top and label canvases are cleared", await js(`
    [...document.querySelectorAll("#top canvas, .track canvas")].every((canvas) =>
      canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data.every((v) => v === 0))`));
  check("empty navigation is finished", await js(`
    S.opened.ticket === S.ticket && S.opened.event === "${b}" && S.opened.shot === null &&
    !pendingNavigation()`), await js("[S.ticket, S.opened]"));
  check("the empty event has an event-only address and an empty header", await js(`
    location.hash === "#${b}" && $("shot").value === "" && $("count").textContent === "0/0" &&
    ["tier", "state", "saved", "next-shot"].every((id) => $(id).textContent === "") &&
    $("dirty").hidden && $("cursor").hidden`));
  check("empty navigation disables Save, Revert and History", await js(`
    ["save-next", "revert", "show-versions"].every((id) => $(id).disabled)`));

  const requests = await js("window.requests.length");
  await js("document.activeElement.blur()");
  for (const key of [...await js("Object.keys(KEYS)"), "J", "K", "U"]) {
    await press(key.replace("Shift+", ""), key.startsWith("Shift+") || /^[JKU]$/.test(key) ? 8 : 0);
    check(`${key} on an empty event neither opens nor saves nor throws`, await js(`
      S.shot === null && S.label === null && S.opened.ticket === S.ticket &&
      window.requests.length === ${requests} && window.saves.length === 0 &&
      !$("versions").open && !$("status").textContent.includes("still opening")`) && errors.length === 0, errors.slice());
  }
  assert.equal(await js(`$("status").textContent`), "no shots to review",
    "U on an empty queue says there are no shots to review");
  await press("z", 2);
  await js(`$("save-next").click(); $("revert").click(); $("show-versions").click()`);
  // Exercise the entry points too: disabled controls and key filtering must not hide a null dereference.
  const actionErrors = await js(`(async () => {
    const errors = [];
    for (const action of [() => edit([0, 2000], [[900, 1100, 1]]), undo, revert,
      () => save(false), () => save(true), toggleVersions]) {
      try { await action(); } catch (error) { errors.push(String(error)); }
    }
    return errors;
  })()`);
  check("direct edit, undo, revert, save and History are harmless without a shot",
    actionErrors.length === 0 && (await js("S.label === null && S.undo.length === 0")), actionErrors);
  const [x, y] = await js(`(() => {
    const r = $("rows").getBoundingClientRect(); return [r.left + 150, r.top + 20];
  })()`);
  await drag(x, x + 150, y, 8);
  await draw(1200, 1400);
  check("drags on the rows and label track add no label or draft", await js(`
    S.label === null && S.drag === null && S.undo.length === 0`) && same(await drafts(), kept));
  check("all empty-event actions send no request and show no opening message", await js(`
    window.requests.length === ${requests} && window.saves.length === 0 &&
    !$("status").textContent.includes("still opening")`));
  // Report the empty-state failures before trying navigation that depends on it.
  if (await js("S.shot !== null || pendingNavigation()")) return;

  // Typing a shot in an empty roster still reaches the server's normal missing-shot path.
  await js(`$("shot").focus(); $("shot").value = "170815"`);
  await press("Enter");
  await until(`S.shot === 170815 && S.opened.ticket === S.ticket`);
  check("a shot typed from empty uses the normal missing-shot path", await js(`
    S.meta === null && $("rows").textContent.includes("170815 has nothing to show:") &&
    location.hash === "#${b}/170815" && window.requests.at(-1).kind === "shot"`));
  await js(`openShot(null).then(() => document.activeElement.blur())`);
  await js(`location.hash = "#${a}/170815"`);
  await settled(a, 170815);
  check("an address opens another event's shot from empty with its draft", same(await js("S.label"), edited));

  await js(`openEvent("${b}")`);
  await js(`location.hash = "#${b}/170815"`);
  await until(`S.shot === 170815 && S.opened.ticket === S.ticket`);
  check("an address opens a shot in the same empty event through the normal error path", await js(`
    S.meta === null && $("rows").textContent.includes("170815 has nothing to show:")`));
  await js("openShot(null)");
  await js(`$("event").value = "${a}"; $("event").dispatchEvent(new Event("change"))`);
  await settled(a, 170815);
  check("switching back from empty opens the first event's shot and draft", same(await js("S.label"), edited));
  await press("r");
  await draw(500, 800);
  const drawn = await js("S.label");
  check("an edit works after returning from empty", drawn.intervals.length === 2 && (await js("dirty()")));
  await press("s");
  await until("!S.saving && !dirty()");
  const saved = await js("window.saves");
  check("saving after returning posts only the first event's own label", saved.length === 1 &&
    saved[0].event === a && saved[0].shot === 170815 && same(saved[0].intervals, drawn.intervals) &&
    same(await js("S.meta.saved"), drawn), saved);
}

async function olderServer() {
  check(
    "the link opens its shot on its source label",
    await js(`S.event === "alfven_eigenmode" && S.shot === 170815 &&
      location.hash === "#alfven_eigenmode/170815" && same(S.label, S.meta.source)`),
    await js("[S.event, S.shot, S.label]")
  );
  check("a missing version route selects API level 1", (await js("S.api")) === 1, await js("S.api"));
  check("the page remembers the name it must not send", (await js("S.name")) === "Ada Lovelace");
  check("the name box is hidden", await js(`$("reviewer-name").hidden`));
  check("the History button is hidden", await js(`$("show-versions").hidden`));
  check(
    "the page asks for a server restart for names and history",
    await js(`!$("stale").hidden && $("stale").textContent === "Restart the server for names and history"`)
  );
  await press("h");
  check(
    "h opens no dialog and asks for no history",
    await js(`document.querySelector("dialog[open]") === null && window.api1HistoryRequests === 0`),
    await js(`[$("versions").open, window.api1HistoryRequests]`)
  );

  await draw(500, 800);
  const drawn = await js("S.label");
  const [kept, span] = drawn.intervals;
  check(
    "a drag on the label adds the drawn span",
    drawn.intervals.length === 2 && same(kept, [100, 300, 1]) &&
      span[2] === 1 && Math.abs(span[0] - 500) <= 2 && Math.abs(span[1] - 800) <= 2,
    drawn
  );
  await press("s");
  await until(`!S.saving && (S.meta.saved !== null || $("status").classList.contains("error"))`);
  check(
    "s saves the label without an error toast",
    await js(`S.meta.saved !== null && $("dirty").hidden && $("state").textContent === "changed" &&
      $("saved").textContent.startsWith("saved ") && !$("status").classList.contains("error")`),
    await js(`[$("state").textContent, $("saved").textContent, $("status").textContent]`)
  );
  check(
    "the saved label is the one drawn",
    same(await js("S.meta.saved"), drawn) && same(await js("S.label"), drawn),
    await js("S.meta.saved")
  );
  const saves = await js("window.api1Saves");
  check(
    "every save body is JSON without a name key",
    saves.length > 0 && saves.every(({ method, path, body }) => {
      try {
        const label = JSON.parse(body);
        return method === "POST" && path === "/api/label" &&
          label !== null && typeof label === "object" && !Object.hasOwn(label, "name");
      } catch {
        return false;
      }
    }),
    saves
  );
}

try {
  await within(send("Runtime.enable"), "first page command (Runtime.enable)");
  await send("Emulation.setDeviceMetricsOverride", { width: 1400, height: 900, deviceScaleFactor: 1, mobile: false });
  if (SCENARIO === "api1") {
    await send("Page.enable");
    await send("Page.addScriptToEvaluateOnNewDocument", { source: `
      localStorage.setItem("labeler:name", "Ada Lovelace");
      window.api1Saves = [];
      window.api1HistoryRequests = 0;
      const realFetch = window.fetch.bind(window);
      window.fetch = async (input, options) => {
        const request = new Request(input, options);
        const path = new URL(request.url).pathname;
        if (path === "/api/version") {
          return new Response('{"detail": "Not Found"}', {
            status: 404, statusText: "Not Found", headers: { "content-type": "application/json" }
          });
        }
        if (path === "/api/label" && request.method === "POST") {
          window.api1Saves.push({ method: request.method, path, body: await request.clone().text() });
        }
        if (path === "/api/history") window.api1HistoryRequests++;
        return realFetch(input, options);
      };
    ` });
  }
  if (["race", "moves", "empty", "inflight"].includes(SCENARIO)) await recordFetches(SCENARIO === "race");
  await send("Page.navigate", { url: `${BASE}/?token=${TOKEN}#alfven_eigenmode/170815` });
  if (SCENARIO !== "race") await opened(170815);
  if (SCENARIO === "race") await navigationRace();
  else if (SCENARIO === "inflight") await inflight();
  else if (SCENARIO === "moves") await moves();
  else if (SCENARIO === "empty") await emptyEvent();
  else if (SCENARIO === "api1") await olderServer();
  else await currentServer();
} catch (error) {
  check("the page did what was asked", false, String(error));
  process.exitCode = 1;
}
check("no script error", errors.length === 0, errors);
console.log(JSON.stringify(checks));
process.exit(process.exitCode || 0);
