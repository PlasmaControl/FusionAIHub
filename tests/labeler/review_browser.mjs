// The review page in headless Chromium, driven over the DevTools protocol.
//   node review_browser.mjs <base-url> <token> <headless-shell> <profile-dir> [api1|race|moves|empty|inflight|pending|equilibrium] [case]
// Prints one JSON line: every check made, [{name, ok, detail}].
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
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
const labelContent = (label) => ({
  window: label.window,
  intervals: label.intervals,
  ...(Object.hasOwn(label, "iscrowd") ? { iscrowd: label.iscrowd } : {}),
});

async function currentServer() {
  const source = await js("S.meta.source");
  check("the link opens its shot on its source label", same(await js("S.label"), source), source);

  const title = await js(`(() => {
    const g = document.createElement("canvas").getContext("2d");
    g.font = \`600 \${FONT}\`;
    const width = GUTTER - 12;
    const lines = wrapped(g, "toroidal mode number n (MPI66M probes)", width);
    const long = "D-alpha FSnn, the ELM spans' channel, clipped to its plasma range (PCPHD03 flat, left out)";
    const cut = capped(g, wrapped(g, long, width), width, 3);
    const widest = (ls) => Math.max(...ls.map((l) => g.measureText(l).width));
    return { lines, widest: widest(lines), width, cut, cutWidest: widest(cut) };
  })()`);
  check(
    "a long row title wraps between words to the gutter, never squeezed",
    title.lines.join(" ") === "toroidal mode number n (MPI66M probes)" && title.lines.length > 1 &&
      title.widest <= title.width,
    title
  );
  check(
    "a title too long for its row keeps the lines that fit, the last ending in an ellipsis",
    title.cut.length === 3 && title.cut[2].endsWith("…") && title.cutWidest <= title.width,
    title
  );

  await until(`$("who").open && $("who-empty").hidden === false`);
  const who = await js(`(() => {
    const r = $("who").getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2, w: innerWidth, h: innerHeight,
      go: $("who-continue").disabled, focus: document.activeElement.id };
  })()`);
  check(
    "a first open asks who is reviewing, centred, with Continue off until a name is picked",
    Math.abs(who.x - who.w / 2) < 2 && Math.abs(who.y - who.h / 2) < 2 && who.go && who.focus === "who-new",
    who
  );
  await press("Escape");
  await sleep(100);
  check("Escape does not get past the list", await js(`$("who").open && !S.picked`));
  await js(`$("who-new").focus()`);
  await send("Input.insertText", { text: "Ada Lovelace" });
  await press("Enter");
  await until(`$("who-names").value === "Ada Lovelace"`);
  const go = await js(`(() => {
    const r = $("who-continue").getBoundingClientRect();
    return [r.left + r.width / 2, r.top + r.height / 2, $("who-continue").disabled];
  })()`);
  await mouse("mousePressed", go[0], go[1], { buttons: 1 });
  await mouse("mouseReleased", go[0], go[1], { buttons: 0 });
  await until(`!$("who").open`);
  check(
    "Add Name lists the name and Continue closes the list, keeping the name and giving the keys back",
    !go[2] && await js(`localStorage.getItem("labeler:name") === "Ada Lovelace" &&
      sessionStorage.getItem("labeler:who") === "Ada Lovelace" && $("reviewer").textContent === "Ada Lovelace" &&
      !$("reviewer").hidden && $("reviewer-name").hidden && document.activeElement === document.body`),
    await js(`[localStorage.getItem("labeler:name"), $("reviewer").textContent, document.activeElement.id]`)
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
  check(
    "the reviewers are listed under the queue",
    await js(`$("contributors").textContent === "170815 reviewed by Ada Lovelace" &&
      $("contributors").getBoundingClientRect().top >= $("queue").getBoundingClientRect().bottom`),
    await js(`$("contributors").textContent`)
  );

  await send("Page.reload");
  await opened(170815);
  check(
    "a reload opens the saved label and the name, without asking again",
    same(await js("S.label"), saved) &&
      (await js(`$("count").textContent === "2/3" && $("reviewer").textContent === "Ada Lovelace" &&
        !$("who").open`)),
    await js(`[S.label, $("count").textContent, $("reviewer").textContent, $("who").open]`)
  );

  await js(`sessionStorage.removeItem("labeler:who")`);
  await send("Page.reload");
  await opened(170815);
  await until(`$("who").open && $("who-names").options.length === 1`);
  check(
    "a new tab asks again, with the last name picked",
    await js(`$("who-names").value === "Ada Lovelace" && !$("who-continue").disabled &&
      document.activeElement === $("who-continue")`),
    await js(`[$("who-names").value, document.activeElement.id]`)
  );
  await press("Enter");
  await until(`!$("who").open`);

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
  check("a second save by the same reviewer lists them once",
    await js(`$("contributors").textContent === "170815 reviewed by Ada Lovelace"`),
    await js(`$("contributors").textContent`));
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

  await js(`$("next").click()`);
  await opened(170815);
  const unsaved = await js(`fetch("/api/history?event=alfven_eigenmode&shot=170817").then((r) => r.json())`);
  check("Next opens the next shot without saving the one it leaves", unsaved.versions.length === 0, unsaved);

  await send("Emulation.setDeviceMetricsOverride", { width: 900, height: 900, deviceScaleFactor: 1, mobile: false });
  await until(`window.innerWidth === 900`);
  check("a narrow window keeps the reviewer's name on screen",
    await js(`(() => { const r = $("reviewer").getBoundingClientRect();
      return r.width > 0 && r.right <= window.innerWidth && r.left >= 0; })()`),
    await js(`[$("reviewer").getBoundingClientRect().toJSON(), window.innerWidth]`));
  await send("Emulation.clearDeviceMetricsOverride");
  await until(`window.innerWidth === 1400`);
  await js(`$("next").click()`);
  await opened(170816);
  await js(`$("next").click()`);
  await opened(170817);
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
        if (gate.reject) throw new TypeError("Failed to fetch");
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
  check(
    "the API version reveals the name and History",
    await js(`!$("reviewer").hidden && $("reviewer-name").hidden && !$("show-versions").hidden && !$("who").open`)
  );

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
  await js(`$("save").click(); document.activeElement.blur()`);
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
    same(labelContent(saved), labelContent(sent)), saved);

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

async function pendingResponses() {
  if (["queue_fail", "queue_reject"].includes(CASE)) return pendingQueueFailure();
  if (CASE.startsWith("queue_")) return pendingQueue();
  if (["next_edit", "save_fail"].includes(CASE)) return pendingSaveMessage();
  const a = "alfven_eigenmode", b = "neoclassical_tearing_mode";
  const key = `labeler:${a}:170815`;
  const original = await js("S.label");
  await draw(500, 800);
  const sent = await js("S.label");
  await arm("label");
  await press("s");
  await held("label");
  if (CASE.startsWith("restore_")) {
    await press("h");
    await until(`$("versions").open`);
    await js(`$("version-list").querySelector('[data-version="1"]').click()`);
  } else {
    await press("z", 2);
  }
  check("Restore or Undo returns to the old baseline while saving",
    same(await js("S.label"), original));
  check("the later label has a draft even when it equals the old baseline",
    same(await js(`JSON.parse(localStorage.getItem("${key}"))`), original));

  if (CASE.endsWith("_switch")) {
    await js(`$("event").value = "${b}"; $("event").dispatchEvent(new Event("change"))`);
    await settled(b, 170815);
  } else if (CASE.endsWith("_type")) {
    await js(`$("shot").focus(); $("shot").value = "170816"`);
    await press("Enter");
    await opened(170816);
    await js("document.activeElement.blur()");
  } else {
    await press("ArrowRight");
    await opened(170816);
  }
  await release("label");
  await until("!S.saving");
  const history = await js(`fetch("/api/history?event=${a}&shot=170815").then((r) => r.json())`);
  const saved = history.versions.at(-1);
  check("navigation does not change the label actually saved by the server",
    same(labelContent(saved), labelContent(sent)));
  check("save completion on another shot or event keeps the later draft",
    same(await js(`JSON.parse(localStorage.getItem("${key}"))`), original));
  if (CASE.endsWith("_switch")) {
    await js(`$("event").value = "${a}"; $("event").dispatchEvent(new Event("change"))`);
    // The event resumes its unreviewed shot; type the shot whose draft was left.
    await settled(a, 170817);
    await js(`$("shot").focus(); $("shot").value = "170815"`);
    await press("Enter");
    await opened(170815);
    await js("document.activeElement.blur()");
  } else {
    await press("ArrowLeft");
    await opened(170815);
  }
  check("returning shows the restored or undone label", same(await js("S.label"), original));
  check("returning marks the later label unsaved in the header and queue", await js(`
    !$("dirty").hidden && $("queue").querySelector('[data-shot="170815"]').classList.contains("dirty")`));
  check("returning keeps the later draft in storage",
    same(await js(`JSON.parse(localStorage.getItem("${key}"))`), original));
  check("returning reads the acknowledged save as its new baseline", same(await js("S.meta.saved"), sent));
}

async function pendingQueue() {
  const a = "alfven_eigenmode", b = "neoclassical_tearing_mode";
  await visit(b, 170815);
  await arm("queue", a);
  await js(`$("event").value = "${a}"; $("event").dispatchEvent(new Event("change"))`);
  await held("queue");
  check("the header explains that this event's queue is loading",
    await js(`$("state").textContent === "the queue is still loading"`), await js(`$("state").textContent`));
  if (CASE === "queue_moves") {
    const ticket = await js("S.ticket"), requests = await js("window.requests.length");
    for (const key of ["ArrowRight", "ArrowLeft", "u"]) {
      await press(key);
      check(`${key} says the queue is loading`,
        await js(`$("status").textContent === "the queue is still loading"`), await js(`$("status").textContent`));
      check(`${key} does not navigate or replace the rows while the queue is held`, await js(`
        S.ticket === ${ticket} && window.requests.length === ${requests} &&
        !$("rows").textContent.includes("no shots to review")`));
    }
    await release("queue");
    await until("S.queue.length === 3");
    check("the arriving queue removes refused-navigation notes", await js(`
      !$("rows").textContent.includes("no shots to review") &&
      !$("status").textContent.includes("queue is still loading")`));
    // Do not wait indefinitely if the broken arrow stole the navigation ticket.
    await until("!pendingNavigation()");
    if (await js("S.shot === 170817")) await opened(170817);
    check("the event opens its resume shot after the queue arrives",
      await js(`S.shot === 170817 && S.meta?.event === "${a}" && !pendingNavigation()`));
    return;
  }
  await js(`$("shot").focus(); $("shot").value = "170817"`);
  await press("Enter");
  await settled(a, 170817);
  await js("document.activeElement.blur()");
  await draw(500, 800);
  const sent = await js("S.label"), ticket = await js("S.ticket");
  await press(CASE === "queue_next" ? "Enter" : "s");
  await until("window.saves.length === 1 && !S.saving");
  const history = await js(`fetch("/api/history?event=${a}&shot=170817").then((r) => r.json())`);
  const saved = history.versions.at(-1);
  check("saving before the queue arrives writes the drawn label to the server",
    same(labelContent(saved), labelContent(sent)));
  check("saving before the queue arrives stays on the typed shot", await js(`
    S.shot === 170817 && S.ticket === ${ticket} && S.meta?.shot === 170817 && !pendingNavigation()`));
  check("Enter explains why it stays; plain S does not invent an empty queue", await js(CASE === "queue_next"
    ? `$("status").textContent === "the queue is still loading"`
    : `!$("rows").textContent.includes("no shots to review")`), await js(`$("status").textContent`));
  await release("queue");
  await until("S.queue.length === 3");
  check("the arriving queue retains the acknowledged save in its count",
    await js(`$("count").textContent === "3/3"`), await js(`$("count").textContent`));
  const serverQueue = await js(`fetch("/api/queue?event=${a}").then((r) => r.json())`);
  check("the arriving queue matches the server's acknowledged review rows",
    same(await js("S.queue"), serverQueue.shots));
  check("the saved shot's chip is reviewed and has no unsaved mark", await js(`
    $("queue").querySelector('[data-shot="170817"]').classList.contains("changed") &&
    !$("queue").querySelector('[data-shot="170817"]').classList.contains("dirty")`));
  check("the next-shot hint wraps from the typed shot in queue order",
    await js(`$("next-shot").textContent === "→ 170815"`), await js(`$("next-shot").textContent`));
  check("queue arrival removes the loading message without reopening the shot", await js(`
    S.ticket === ${ticket} && S.shot === 170817 &&
    !$("status").textContent.includes("queue is still loading") &&
    !$("state").textContent.includes("queue is still loading")`));
  await press("ArrowRight");
  await opened(170815);
  check("the arrow after queue arrival opens the next shot in queue order", true);
  await press("u");
  await until("!pendingNavigation()");
  check("U skips the newly acknowledged shot and reports all reviewed", await js(`
    S.shot === 170815 && $("status").textContent === "all reviewed"`));
}

async function pendingSaveMessage() {
  const a = "alfven_eigenmode", key = `labeler:${a}:170815`;
  await draw(500, 800);
  const sent = await js("S.label");
  await arm("label");
  await press(CASE === "next_edit" ? "Enter" : "s");
  await held("label");
  if (CASE === "next_edit") await draw(1200, 1400);
  const later = await js("S.label");
  if (CASE === "save_fail") {
    await press("ArrowRight");
    await opened(170816);
  }
  await release("label");
  await until("!S.saving");
  await opened(170816);
  if (CASE === "next_edit") {
    check("Save-and-next uses the same unsaved-draft notice as an arrow", await js(`
      $("status").textContent === "170815: the edit is kept as a draft, not saved"`),
      await js(`$("status").textContent`));
    const history = await js(`fetch("/api/history?event=${a}&shot=170815").then((r) => r.json())`);
    const saved = history.versions.at(-1);
    check("Save-and-next saves only what it sent",
      same(labelContent(saved), labelContent(sent)));
  } else {
    check("a failed save names its shot after the reviewer moved", await js(`
      $("status").textContent === "170815 not saved: [Errno 122] Disk quota exceeded" &&
      $("status").classList.contains("error")`), await js(`$("status").textContent`));
  }
  check("the next shot keeps its own label", same(await js("S.label.intervals"), [[400, 600, 1]]));
  check("the unsaved label is kept in the original shot's draft",
    same(await js(`JSON.parse(localStorage.getItem("${key}"))`), later));
  await press("ArrowLeft");
  await opened(170815);
  check("returning shows the unsaved label", same(await js("S.label"), later));
  check("returning marks the unsaved label in the header and queue", await js(`
    !$("dirty").hidden && $("queue").querySelector('[data-shot="170815"]').classList.contains("dirty")`));
}

async function pendingQueueFailure() {
  const b = "neoclassical_tearing_mode";
  await arm("queue", b);
  if (CASE === "queue_reject") await js("window.gates.queue.reject = true");
  await js(`$("event").value = "${b}"; $("event").dispatchEvent(new Event("change"))`);
  await held("queue");
  // Observe completion of the event handler even when the old page leaves navigation pending.
  const message = CASE === "queue_fail" ? "fixture queue failure" : "Failed to fetch";
  await release("queue");
  await until(`$("rows").textContent.includes("${message}") || $("status").textContent.includes("${message}")`);
  check("a queue failure names its event, explains the error and gives the retry", await js(`
    $("rows").textContent === "${b}: the queue could not be read: ${message}. Choose the event again to retry."`),
    await js(`$("rows").textContent`));
  check("a failed queue clears the old shot and label", await js(`
    [S.shot, S.meta, S.data, S.overview, S.label].every((value) => value === null) &&
    S.undo.length === 0 && S.selected === -1`));
  check("a failed queue finishes navigation and uses an event-only address", await js(`
    !pendingNavigation() && S.opened.event === "${b}" && S.opened.shot === null &&
    location.hash === "#${b}" && $("shot").value === ""`));
  check("a failed queue clears the header and disables shot actions", await js(`
    $("count").textContent === "0/0" && ["tier", "state", "saved", "next-shot"].every((id) => $(id).textContent === "") &&
    ["save", "revert", "show-versions"].every((id) => $(id).disabled)`));
  await js("document.activeElement.blur(); edit([0, 2000], [[500, 800, 1]])");
  await press("s");
  check("a failed queue refuses no edit as still opening and cannot save", await js(`
    !$("status").textContent.includes("still opening") && window.saves.length === 0`));
  await js(`$("event").dispatchEvent(new Event("change"))`);
  await settled(b, 170815);
  check("choosing the event again retries and opens its shot", await js(`
    S.meta.event === "${b}" && location.hash === "#${b}/170815" && !pendingNavigation() &&
    window.requests.filter((r) => r.kind === "queue" && r.event === "${b}").length === 2`));
  check("retry opens the event's own label", same(await js("S.label.intervals"), [[900, 1100, 1]]));
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
    ["save", "revert", "show-versions"].every((id) => $(id).disabled)`));

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
  await js(`$("save").click(); $("revert").click(); $("show-versions").click()`);
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

async function equilibriumEditors() {
  await visit("resistive_wall_mode", 170815);
  const points = await js("S.meta.onsets");
  check("RWM points keep exact times and toroidal modes", same(
    points.map(p => [p.t_ms, p.ntor, p.mode_type]),
    [[900.1234, 1, "rwm"], [900.6789, 2, "n2rwm"]]), points);
  const buttons = await js(`Array.from($("onsets").querySelectorAll("button"))
    .map(b => [b.textContent, b.title])`);
  check("neighboring onsets have separate exact clickable annotations", same(buttons,
    [["900.1234 ms · n=1 · rwm", "curated"],
     ["900.6789 ms · n=2 · n2rwm", "curated"]]), buttons);
  check("point annotations do not create a duration label", await js("S.meta.source === null"));
  const ticks = await js(`(() => {
    const canvas = $("source-track"), g = canvas.getContext("2d");
    const original = g.stroke, strokes = [];
    g.stroke = () => strokes.push(true);
    try { drawTrack(canvas, null, false); } finally { g.stroke = original; }
    return strokes.length;
  })()`);
  check("a source without intervals still draws both exact onset ticks", ticks === 2, ticks);
  await js(`$("onsets").querySelectorAll("button")[1].click()`);
  check("an onset button zooms to the exact time", same(await js("S.view"), [800.6789, 1000.6789]));
  await visit("minimum_safety_factor", 170815);
  check("RWM annotations clear when changing events", await js(`$("onsets").hidden &&
    $("onsets").children.length === 0`));
  check("all five qmin categories have different track colors, and not observable is not offered", await js(`
    new Set(known().map(categoryColour)).size === 5 &&
    S.categories[5] === "uncertain" && !(6 in S.categories)`));
  await press("5");
  const [x0, x1, y] = await js(`(() => {
    const r = $("rows").querySelector("canvas").getBoundingClientRect();
    return [r.left + px(500), r.left + px(800), r.top + r.height / 2];
  })()`);
  await drag(x0, x1, y, 8); // Shift draws over the existing regime.
  await press("s");
  await until("!S.saving && !dirty()");
  check("saving qmin uncertainty preserves its multiclass category", await js(`
    S.meta.saved.intervals.some(([a,b,c]) => c === 5 && Math.abs(a-500) <= 25 && Math.abs(b-800) <= 25)`),
    await js("S.meta.saved"));
  await visit("poloidal_beta", 170815);
  check("beta_p opens its signal and threshold with the source draft", await js(`
    S.meta.rows[0].title === "beta_p (EFIT01 aeqdsk)" &&
    same(S.meta.rows[0].hlines, [1]) && S.label.intervals[0][2] === 1`));
}

async function crowdAnnotations() {
  check("legacy labels stay unspecified", await js(`
    !S.label.iscrowd && !$("resolution-control").hidden`));
  const select = async (t) => {
    const [x, y] = await js(`(() => {
      const i = S.label.intervals.findIndex(([a,b]) => a <= ${t} && ${t} <= b);
      const r = $(S.label.iscrowd?.[i] === 1 ? "crowd-track" : "label-track").getBoundingClientRect();
      return [r.left + px(${t}), r.top + 15];
    })()`);
    await mouse("mousePressed", x, y, { buttons: 1 });
    await mouse("mouseReleased", x, y, { buttons: 0 });
  };
  const resolution = async (value) => js(`
    $("resolution").value = "${value}";
    $("resolution").dispatchEvent(new Event("change", {bubbles: true}));
  `);
  await select(200);
  await resolution("1");
  check("the selected source span becomes a group", same(await js("S.label.iscrowd"), [1]));
  await press("z", 2);
  check("undo restores unspecified metadata", await js("!S.label.iscrowd && !dirty()"));
  await select(200);
  await resolution("1");
  await press("s");
  await until("!S.saving && !dirty()");
  await send("Page.reload");
  await opened(170815);
  check("groups survive saving and reloading", same(await js("S.label.iscrowd"), [1]));
  await select(200);
  await press("2");
  check("changing category preserves group resolution", await js(`
    S.label.intervals[0][2] === 2 && S.label.iscrowd[0] === 1`));
  const [x0, x1, y] = await js(`(() => {
    const r = $("crowd-track").getBoundingClientRect();
    return [r.left + px(200), r.left + px(1200), r.top + 15];
  })()`);
  await drag(x0, x1, y);
  check("moving an annotation keeps its group flag", await js(`
    Math.abs(S.label.intervals[0][0] - 1100) <= 2 && S.label.iscrowd[0] === 1`));
  await press("z", 2);
  await press("1");
  await press("Escape");
  await resolution("0");
  const drawAdjacent = async (a, b) => {
    const [x0, x1, y] = await js(`(() => {
      const r = $("rows").children[0].getBoundingClientRect();
      return [r.left + px(${a}), r.left + px(${b}), r.top + 40];
    })()`);
    await drag(x0, x1, y, 8);
  };
  await drawAdjacent(500, 600);
  await drawAdjacent(600, 700);
  check("touching individuals remain separate annotations", await js(`
    S.label.intervals.length === 3 && same(S.label.iscrowd, [1,0,0]) &&
    S.label.intervals[1][1] === S.label.intervals[2][0]`), await js("S.label"));
  await press("Delete");
  check("deleting an individual keeps the other resolutions aligned", same(
    await js("S.label.iscrowd"), [1, 0]));
  await press("z", 2);
  await press("s");
  await until("!S.saving && !dirty()");
  await press("h");
  await until(`$("versions").open`);
  await js(`$("version-list").querySelector('[data-version="1"]').click()`);
  check("history restore brings back group metadata", await js(`
    S.label.intervals.length === 1 && same(S.label.iscrowd, [1]) && dirty()`));
  await press("z", 2);
  check("undo restores the individual boundaries after history restore", await js(`
    S.label.intervals.length === 3 && same(S.label.iscrowd, [1,0,0])`));
  await press("Escape");
  await resolution("1");
  await drawAdjacent(900, 1000);
  await press("k");
  await opened(170816);
  await press("j");
  await opened(170815);
  check("navigation restores the draft with its group metadata", same(
    await js("S.label.iscrowd"), [1, 0, 0, 1]));
  await press("r");
  check("revert restores the unspecified source", await js(`
    S.label.intervals.length === 1 && !S.label.iscrowd`));
  await press("z", 2);
  check("undo restores all draft annotation flags", same(
    await js("S.label.iscrowd"), [1, 0, 0, 1]));
}

async function overlappingAnnotations() {
  const select = async (t, lane) => {
    const [x, y] = await js(`(() => {
      const r = $("${lane === 1 ? "crowd-track" : "label-track"}").getBoundingClientRect();
      return [r.left + px(${t}), r.top + 15];
    })()`);
    await mouse("mousePressed", x, y, { buttons: 1 });
    await mouse("mouseReleased", x, y, { buttons: 0 });
  };
  const resolution = async (value) => js(`
    $("resolution").value = "${value}";
    $("resolution").dispatchEvent(new Event("change", {bubbles: true}));
  `);
  check("the editor shows named individual and crowd lanes", await js(`
    $("individual-lane-name").textContent === "Individual" && !$("crowd-lane").hidden`));
  await select(200, 0);
  await resolution("1");
  check("switching resolution moves the selected span into the crowd lane", await js(`
    S.label.iscrowd[S.selected] === 1 && $("resolution").value === "1"`));
  await press("Escape");
  await draw(150, 200);
  check("drawing an individual over a crowd keeps its complete envelope", await js(`
    same(S.label.intervals[0], [100,300,1]) && S.label.intervals.length === 2 &&
    same(S.label.iscrowd, [1,0]) && Math.abs(S.label.intervals[1][0]-150) <= 2 &&
    Math.abs(S.label.intervals[1][1]-200) <= 2`), await js("S.label"));
  check("the newly drawn individual is selected instead of the underlying crowd", await js(`
    S.label.iscrowd[S.selected] === 0 && $("resolution").value === "0"`));
  await press("s");
  await until("!S.saving && !dirty()");
  await send("Page.reload");
  await opened(170815);
  check("save and reload retain both overlapping annotations", await js(`
    same(S.label.intervals[0], [100,300,1]) && S.label.intervals.length === 2 &&
    same(S.label.iscrowd, [1,0])`));
  const [x0, x1, y] = await js(`(() => {
    const r = $("label-track").getBoundingClientRect();
    return [r.left+px(175), r.left+px(225), r.top+15];
  })()`);
  await drag(x0, x1, y);
  check("moving an overlapping individual preserves the crowd", await js(`
    same(S.label.intervals[0], [100,300,1]) && S.label.iscrowd[S.selected] === 0 &&
    Math.abs(S.label.intervals[1][0]-200) <= 2 && Math.abs(S.label.intervals[1][1]-250) <= 2`));
  const [edge0, edge1, edgeY] = await js(`(() => {
    const r = $("label-track").getBoundingClientRect();
    return [r.left+px(S.label.intervals[1][1]), r.left+px(280), r.top+15];
  })()`);
  await drag(edge0, edge1, edgeY);
  check("resizing an individual keeps the overlapping crowd unchanged", await js(`
    same(S.label.intervals[0], [100,300,1]) && Math.abs(S.label.intervals[1][1]-280) <= 2`));
  await select(230, 1);
  check("clicking the crowd lane selects the crowd under the individual", await js(`
    S.label.iscrowd[S.selected] === 1`));
  await press("Delete");
  check("deleting a crowd preserves the individual inside it", await js(`
    S.label.intervals.length === 1 && same(S.label.iscrowd, [0])`));
  await press("z", 2);
  check("undo restores the overlapping crowd and individual", await js(`
    same(S.label.intervals[0], [100,300,1]) && same(S.label.iscrowd, [1,0])`));
  await press("s");
  await until("!S.saving && !dirty()");
  await press("h");
  await until(`$("versions").open`);
  await js(`$("version-list").querySelector('[data-version="1"]').click()`);
  check("history restore retains the overlap and original individual boundary", await js(`
    same(S.label.intervals[0], [100,300,1]) && same(S.label.iscrowd, [1,0]) &&
    Math.abs(S.label.intervals[1][0]-150) <= 2 && Math.abs(S.label.intervals[1][1]-200) <= 2`));
  await press("z", 2);
  await draw(350, 400);
  await press("k");
  await opened(170816);
  await press("j");
  await opened(170815);
  check("navigation restores overlapping annotations in the draft", await js(`
    same(S.label.intervals[0], [100,300,1]) && same(S.label.iscrowd, [1,0,0]) &&
    Math.abs(S.label.intervals[1][1]-280) <= 2`));
  await select(375, 0);
  await press("Delete");
  await until("!dirty()");
  if (CASE) {
    const screenshot = await send("Page.captureScreenshot", { format: "png" });
    writeFileSync(CASE, Buffer.from(screenshot.data, "base64"));
  }
}

async function detachmentCameras(shot = 170815, demo = false) {
  if (!demo) await visit("detachment", shot);
  await until(`S.video && S.video.cards.some(c => c.img?.naturalWidth > 0)`);
  check("detachment offers all four states", same(await js("S.categories"),
    {1: "attached", 2: "detached", 3: "marfe", 4: "uncertain"}));
  if (!demo) check("magnetic configuration and active lower-outer-leg gate are visible", await js(`
    $("video-geometry").textContent.includes("LSN (lower single null)") &&
    $("video-geometry").textContent.includes("strike-point gate valid")`));
  check("cameras have small manifests and lazy pixels", await js(`
    S.meta.video.cameras.every(c => c.channels.every(ch => !ch.frames)) &&
    S.video.cards.some(c => c.img?.src.startsWith("blob:"))`));
  check("missing cameras explain their absence", await js(`
    S.video.cards.filter(c => !c.channel).every(c => c.note.textContent.includes("No frames"))`));
  check("selectors name physical views and default to a live lower divertor", await js(`
    S.video.cards.filter(c => c.channel && c.camera.name === "tangtv").every(c =>
      [0,2].includes(c.channel.channel) &&
      c.figure.querySelector("select").selectedOptions[0].textContent.includes("lower divertor"))`));
  await js(`window.realVideoFetch = window.fetch.bind(window);
    window.frameActive = 0; window.framePeak = 0;
    window.fetch = async (input, options) => {
      if (!String(input).startsWith("/api/frame?")) return window.realVideoFetch(input, options);
      window.framePeak = Math.max(window.framePeak, ++window.frameActive);
      try {
        const result = await window.realVideoFetch(input, options);
        await new Promise(ok => setTimeout(ok, 180));
        return result;
      } finally { window.frameActive--; }
    };
    window.oldFrames = S.video.cards.filter(c => c.channel).map(c => c.img.dataset.frameTime);`);
  const middle = await js(`S.video.times[Math.floor(S.video.times.length / 2)]`);
  await js(`$("video-time").value = ${middle};
    $("video-time").dispatchEvent(new Event("input", {bubbles: true}));`);
  check("delayed seek hides old pixels and keeps their time until decoded", await js(`
    S.video.cards.filter(c => c.channel).every((c,i) =>
      c.figure.getAttribute("aria-busy") === "true" &&
      c.img.dataset.frameTime === window.oldFrames[i] &&
      getComputedStyle(c.img).visibility === "hidden" && c.note.textContent.includes("Loading"))`));
  await until(`S.video.cards.filter(c => c.channel).every(c =>
    Math.abs(Number(c.img.dataset.frameTime) - Number(c.note.textContent.split(" ")[0])) <= 0.06 && c.img.complete)`);
  check("slider seeks the stored corpus clock and persistent timeline cursor", await js(`
    Math.abs(S.video.time - ${middle}) < 0.01 && !$("cursor").hidden &&
    Math.abs(parseFloat($("cursor").style.left) - ($("axis-row").getBoundingClientRect().left + px(S.video.time))) < 1`));
  check("decoded image, physical identity and caption time agree", await js(`
    S.video.cards.filter(c => c.channel).every(c =>
      c.img.dataset.channel === String(c.channel.channel) &&
      c.img.alt.includes(c.channel.view_name) &&
      c.figure.getAttribute("aria-busy") === "false" &&
      c.note.textContent.includes(c.channel.view_name))`));
  check("detached and uncertain use distinct colour-blind-safe blue and orange", await js(`
    categoryColour(2) === "#0072b2" && categoryColour(4) === "#d55e00"`));
  if (!demo) {
    await send("Emulation.setDeviceMetricsOverride", { width: 1400, height: 2600, deviceScaleFactor: 1, mobile: false });
    await sleep(100);
    await js(`window.rowHeightBefore = $("rows").children[0].clientHeight;
      window.panelHeightBefore = $("video-panel").style.height;
      $("video-panel").style.height = ($("video-panel").clientHeight + 60) + "px";`);
    await sleep(100);
    check("rows resize when the video panel grows after delivery", await js(`
      $("rows").children[0].clientHeight < window.rowHeightBefore`));
    await js(`$("video-panel").style.height = window.panelHeightBefore`);
    await send("Emulation.setDeviceMetricsOverride", { width: 1400, height: 900, deviceScaleFactor: 1, mobile: false });
    await sleep(50);
  }
  const [hoverX, hoverY] = await js(`(() => {
    const r = $("rows").querySelector("canvas")?.getBoundingClientRect() || $("axis-row").getBoundingClientRect();
    return [r.left + px(S.video.times[0]), r.top + r.height/2];
  })()`);
  await mouse("mouseMoved", hoverX, hoverY);
  check("detachment hover reads time independently of the pinned video cursor", await js(`
    !$('hover-cursor').hidden && $('hover-time').textContent !== $('cursor-time').textContent &&
    Math.abs(S.video.time - ${middle}) < 0.01`));
  await js(`$("top").scrollTop = 10000`);
  await sleep(50);
  check("video stays visible and cursor starts below the sticky panel while scrolling", await js(`
    Math.abs($("video-panel").getBoundingClientRect().top - $("top").getBoundingClientRect().top) < 1 &&
    parseFloat($("cursor").style.top) >= $("video-panel").getBoundingClientRect().bottom - 1`));
  await js(`$("top").scrollTop = 0`);
  await sleep(50);
  const channelCard = await js(`S.video.cards.findIndex(c => c.camera.channels.length > 1)`);
  if (channelCard >= 0) {
    await js(`(() => {
      const card = S.video.cards[${channelCard}], select = card.figure.querySelector("select");
      select.value = card.camera.channels.at(-1).channel;
      select.dispatchEvent(new Event("change", {bubbles: true}));
    })()`);
    check("changing views hides pixels until their channel identity is delivered", await js(`
      getComputedStyle(S.video.cards[${channelCard}].img).visibility === "hidden"`));
    await until(`S.video.cards[${channelCard}].img.complete &&
      S.video.cards[${channelCard}].img.dataset.channel === String(S.video.cards[${channelCard}].channel.channel)`);
    check("camera channel selectors retain the shared time", await js(`Math.abs(S.video.time - ${middle}) < 0.01`));
  }
  const target = await js(`S.video.times[Math.floor(S.video.times.length / 3)]`);
  const [x, y] = await js(`(() => {
    const r = $("rows").querySelector("canvas")?.getBoundingClientRect() ||
      $("axis-row").getBoundingClientRect();
    return [r.left + px(${target}), r.top + r.height / 2];
  })()`);
  await mouse("mousePressed", x, y, {buttons: 1});
  await mouse("mouseReleased", x, y, {buttons: 0});
  await until(`S.video.cards.filter(c => c.channel).every(c => c.figure.getAttribute("aria-busy") === "false")`);
  check("clicking a timeline seeks without changing labels", await js(`
    Math.abs(S.video.time - ${target}) < 2 && !dirty()`));
  await js(`window.playStart = S.video.time; window.playFrames = S.video.cards.filter(c => c.channel).map(c => c.img.src);
    window.framePeak = 0;`);
  await js(`$("video-play").click()`);
  await sleep(100);
  check("playback waits for frames delayed beyond their cadence", await js(`S.video.time === window.playStart`));
  await until(`S.video.time > ${target} + 0.1`);
  check("playback delivers new pixels before advancing and bounds requests", await js(`
    S.video.cards.filter(c => c.channel).every((c,i) =>
      c.img.src !== window.playFrames[i] &&
      Math.abs(Number(c.img.dataset.frameTime)-S.video.time) <= 65) &&
    window.framePeak <= S.video.cards.filter(c => c.channel).length`));
  await js(`$("video-play").click()`);
  const paused = await js("S.video.time");
  await sleep(150);
  check("play steps forward and pause stops the clock", await js(`
    S.video.timer === null && S.video.time === ${paused} &&
    $("video-play").getAttribute("aria-pressed") === "false"`));
  await js("window.fetch = window.realVideoFetch");
  // Fast seeks must finish on the latest request, even when an older fetch arrives late.
  await js(`window.realVideoFetch = window.fetch.bind(window);
    window.fetch = async (input, options) => {
      const result = await window.realVideoFetch(input, options);
      if (String(input).startsWith("/api/frame?") &&
          new URL(String(input), location.origin).searchParams.get("t_ms") === String(S.video?.times[0]))
        await new Promise(ok => setTimeout(ok, 150));
      return result;
    };
    seekVideo(S.video.times[0]); seekVideo(${middle});`);
  await sleep(250);
  check("late frame responses cannot replace a newer seek", await js(`
    S.video.cards.filter(c => c.channel).every(c =>
      Math.abs(Number(c.img.dataset.frameTime) - Number(c.note.textContent.split(" ")[0])) <= 0.06) &&
      Math.abs(S.video.time - ${middle}) < 0.01`));
  await js("window.fetch = window.realVideoFetch");
  if (demo) {
    await js(`(async () => {
      const card = S.video.cards.find(c => c.camera.name === "tangtv");
      card.channel = card.camera.channels.find(c => c.channel === 2 && c.region === "lower divertor") ||
        card.camera.channels.find(c => c.region === "lower divertor");
      card.figure.querySelector("select").value = card.channel.channel;
      videoTimes();
      const requested = ${JSON.stringify(process.env.DETACHMENT_CAPTURE_TIME_MS || "")};
      const time = requested === "" ? S.video.time : S.video.times.reduce((best, item) =>
        Math.abs(item - Number(requested)) < Math.abs(best - Number(requested)) ? item : best);
      await seekVideo(time);
    })()`);
    await until(`S.video.cards.filter(c => c.channel).every(c => c.img.complete)`);
    check("evidence and paper capture use a delivered lower-divertor view", await js(`
      S.video.cards.find(c => c.camera.name === "tangtv").channel.region === "lower divertor"`));
    await js(`(() => {
      const i = S.meta.rows.findIndex(row => row.title === "Gas flow");
      if (i >= 0) $("top").scrollTop += $("rows").children[i].getBoundingClientRect().top -
        $("top").getBoundingClientRect().top - $("video-panel").offsetHeight;
      showVideoCursor(); $("hover-cursor").hidden = true;
    })()`);
    await sleep(100);
    const screenshot = await send("Page.captureScreenshot", {format: "png"});
    writeFileSync(CASE, Buffer.from(screenshot.data, "base64"));
    // A separate crop for paper use, from the same live page and delivered data.
    await send("Emulation.setDeviceMetricsOverride", { width: 500, height: 780, deviceScaleFactor: 1, mobile: false });
    await js(`(() => {
      S.video.cards.find(c => c.camera.name === "tangtv").figure.classList.add("paper-camera");
      const style = document.createElement("style");
      style.textContent = ":root {font-size:16px} .camera {display:none} .paper-camera {display:block}" +
        "#video-cameras {display:block} .camera select {font-size:16px}" +
        ".camera small, #video-context, #bar, #queue, #contributors, #resolution-control," +
        "#next, #revert, #show-versions {display:none!important}" +
        "#cursor-time, kbd {font-size:16px} #hover-cursor {display:none!important}" +
        "#controls {padding:0 8px} #swatches {flex-wrap:wrap} #bottom {max-height:none}" +
        "#rows canvas {display:none} body {width:500px;height:auto;grid-template-rows:auto auto auto}" +
        "#top, #bottom {min-width:0} #swatches {flex:1 1 100%;min-width:0}" +
        "#top {overflow:visible} #video-controls {flex-wrap:wrap} .camera img {height:180px}";
      document.head.append(style); $("top").scrollTop = 0;
      sizeCanvases(); render();
    })()`);
    await sleep(150);
    const layout = await js(`(() => {
      const box = e => { const r = e.getBoundingClientRect();
        return {x:r.x,y:r.y,width:r.width,height:r.height}; };
      const card = S.video.cards.find(c => c.camera.name === "tangtv");
      return {shot:S.shot,view:card.channel.view_name,region:card.channel.region,
        clock_ms:S.video.time,frame_ms:Number(card.img.dataset.frameTime),
        configuration:$("video-geometry").textContent, viewport:{width:500,height:780},
        camera:box(card.img),timestamp:box($("video-clock")),cursor:box($("cursor")),
        labels:box($("swatches")),save:box($("save")), crop_bottom:$("controls").getBoundingClientRect().bottom+8};
    })()`);
    check("paper crop retains readable camera pixels, cursor and all four label controls", await js(`
      S.video.cards.find(c => c.camera.name === "tangtv").img.naturalWidth > 0 &&
      !$("cursor").hidden && $("swatches").children.length === 4 &&
      $("controls").getBoundingClientRect().bottom < innerHeight &&
      $("save").getBoundingClientRect().right <= innerWidth &&
      $("video-clock").getBoundingClientRect().right <= innerWidth`));
    writeFileSync(CASE + ".paper.json", JSON.stringify(layout, null, 2) + "\n");
    const paper = await send("Page.captureScreenshot", {format:"png",clip:{x:0,y:0,width:500,
      height:Math.ceil(layout.crop_bottom),scale:1}});
    writeFileSync(CASE + ".paper.png", Buffer.from(paper.data, "base64"));
    return;
  }
  await press("2");
  await draw(80, 160);
  await press("s");
  await until("!S.saving && !dirty()");
  check("detached labels are saved on the usual timeline", await js(`
    S.meta.saved.intervals.some(([a,b,c]) => c === 2 && Math.abs(a-80) <= 2 && Math.abs(b-160) <= 2)`));
  await js(`$("video-play").click()`);
  await visit("alfven_eigenmode", 170815);
  check("navigation clears playback, previews and pinned cursor", await js(`
    S.video === null && $("video-panel").hidden && $("video-cameras").children.length === 0`));
  await js(`$("top").scrollTop = 10000; showCursor($("axis-row").getBoundingClientRect().left + px(100));`);
  check("other events clamp the cursor top to the visible rows viewport", await js(`
    parseFloat($("cursor").style.top) >= $("top").getBoundingClientRect().top`));
}

async function atomicDetachmentPlayback() {
  await visit("detachment", 170815);
  await js(`window.earlyRealFetch = window.fetch.bind(window); window.earlyReleases = [];
    window.fetch = async (input, options) => {
      const response = await window.earlyRealFetch(input, options);
      if (String(input).startsWith("/api/frame?"))
        await new Promise(resolve => window.earlyReleases.push(resolve));
      return response;
    }; clearVideo(); buildVideo();`);
  await until(`window.earlyReleases.length === 2`);
  check("Play stays disabled before the first complete camera transaction", await js(`
    $("video-play").disabled && !S.video.playing`));
  await js(`$("video-play").click()`);
  check("early Play cannot publish a NaN timestamp", await js(`
    !$("video-clock").textContent.includes("NaN") && !S.video.playing`));
  await js(`window.fetch = window.earlyRealFetch; window.earlyReleases.forEach(resolve => resolve());`);
  await until(`S.video?.cards.filter(c => c.channel).length === 2 &&
    S.video.cards.filter(c => c.channel).every(c => c.img?.naturalWidth > 0)`);
  await js(`window.fetch = async (input, options) => {
    if (String(input).startsWith("/api/frame?")) throw new Error("initial frame failure");
    return window.earlyRealFetch(input, options);
  }; clearVideo(); buildVideo();`);
  await until(`S.video.cards.filter(c => c.channel).every(c => c.figure.getAttribute("aria-busy") === "false")`);
  check("an initial frame failure leaves Play disabled and no committed clock", await js(`
    $("video-play").disabled && $("video-clock").textContent === "" && !S.video.playing &&
    S.video.cards.filter(c => c.channel).every(c => c.note.textContent.includes("unavailable"))`));
  await js(`(async () => { window.fetch = window.earlyRealFetch;
    await seekVideo(S.video.times[0]); })()`);
  check("a successful retry enables playback at a finite timestamp", await js(`
    !$("video-play").disabled && Number.isFinite(S.video.time)`));
  await js(`window.fetch = async (input, options) => {
    if (String(input).startsWith("/api/frame?")) throw new Error("seek failure");
    return window.earlyRealFetch(input, options);
  }; $("video-time").value = 60; $("video-time").dispatchEvent(new Event("input", {bubbles:true}));`);
  await until(`S.video.cards.filter(c => c.channel).every(c => c.figure.getAttribute("aria-busy") === "false")`);
  check("a failed seek restores slider, clock and both delivered frames", await js(`
    S.video.time === 0 && Number($("video-time").value) === 0 &&
    $("video-clock").textContent === "0.0 ms" && S.video.cards.filter(c => c.channel).every(c =>
      Number(c.img.dataset.frameTime) === 0 && c.note.textContent.startsWith("0.0 ms"))`));
  await js("window.fetch = window.earlyRealFetch");
  await js(`(async () => { await seekVideo(S.video.times[0]);
    window.atomicBefore = S.video.cards.filter(c => c.channel).map(c =>
      [c.img.src, c.img.dataset.frameTime, c.note.textContent]);
    window.atomicTime = S.video.time;
    window.atomicSlowRelease = null;
    window.realVideoFetch = window.fetch.bind(window);
    window.fetch = async (input, options) => {
      const response = await window.realVideoFetch(input, options);
      if (String(input).startsWith("/api/frame?")) {
        const camera = new URL(String(input), location.origin).searchParams.get("camera");
        if (camera === "irtv") await new Promise(resolve => { window.atomicSlowRelease = resolve; });
      }
      return response;
    };
    $("video-play").click(); })()`);
  await until(`window.atomicSlowRelease !== null`);
  // Wait for the fast camera's detached Image.decode(), while IRTV is held.
  await until(`S.video.cards.find(c => c.camera.name === "tangtv").staged != null ||
    S.video.cards.find(c => c.camera.name === "tangtv").img.src !== window.atomicBefore[0][0]`);
  check("unequal camera deliveries keep both published images and clock at the old time", await js(`
    S.video.time === window.atomicTime && S.video.cards.filter(c => c.channel).every((c,i) =>
      c.img.src === window.atomicBefore[i][0] && c.img.dataset.frameTime === window.atomicBefore[i][1] &&
      c.note.textContent === window.atomicBefore[i][2])`));
  await js(`pauseVideo(); window.atomicSlowRelease();`);
  await until(`S.video.cards.filter(c => c.channel).every(c => c.pendingKey === "")`);
  check("pause between deliveries discards staged images and preserves both captions and clock", await js(`
    S.video.time === window.atomicTime && S.video.cards.filter(c => c.channel).every((c,i) =>
      !c.staged && c.img.src === window.atomicBefore[i][0] &&
      c.img.dataset.frameTime === window.atomicBefore[i][1] && c.note.textContent === window.atomicBefore[i][2] &&
      getComputedStyle(c.img).visibility === "visible" && c.figure.getAttribute("aria-busy") === "false")`));
  await js(`window.fetch = window.realVideoFetch;
    window.playStart = S.video.time; $("video-play").click();`);
  await until(`S.video.time > window.playStart`);
  check("all cameras and the shared clock publish together after resume", await js(`
    S.video.cards.filter(c => c.channel).every(c =>
      Math.abs(Number(c.img.dataset.frameTime) - S.video.time) < 0.01 &&
      c.note.textContent.startsWith(Number(c.img.dataset.frameTime).toFixed(1)))`));
  await js("pauseVideo()");
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
  if (["race", "moves", "empty", "inflight", "pending"].includes(SCENARIO)) await recordFetches(SCENARIO === "race");
  if (SCENARIO && SCENARIO !== "api1") {
    // Every scenario but the first open has picked its name already (API 1 has no list).
    await send("Page.enable");
    await send("Page.addScriptToEvaluateOnNewDocument", { source: `
      sessionStorage.setItem("labeler:who", ${JSON.stringify(SCENARIO === "detachment-demo" ? "Reviewer" : "Grace Hopper")});
    ` });
  }
  const demoShot = Number(process.env.DETACHMENT_DEMO_SHOT || 190010);
  const initial = SCENARIO === "detachment-demo" ? `detachment/${demoShot}` : "alfven_eigenmode/170815";
  await send("Page.navigate", { url: `${BASE}/?token=${TOKEN}#${initial}` });
  if (SCENARIO !== "race") await opened(SCENARIO === "detachment-demo" ? demoShot : 170815);
  if (SCENARIO === "race") await navigationRace();
  else if (SCENARIO === "inflight") await inflight();
  else if (SCENARIO === "pending") await pendingResponses();
  else if (SCENARIO === "moves") await moves();
  else if (SCENARIO === "empty") await emptyEvent();
  else if (SCENARIO === "api1") await olderServer();
  else if (SCENARIO === "equilibrium") await equilibriumEditors();
  else if (SCENARIO === "crowds") await crowdAnnotations();
  else if (SCENARIO === "overlaps") await overlappingAnnotations();
  else if (SCENARIO === "detachment") await detachmentCameras();
  else if (SCENARIO === "detachment-atomic") await atomicDetachmentPlayback();
  else if (SCENARIO === "detachment-demo") await detachmentCameras(demoShot, true);
  else await currentServer();
} catch (error) {
  check("the page did what was asked", false, String(error));
  process.exitCode = 1;
}
check("no script error", errors.length === 0, errors);
console.log(JSON.stringify(checks));
process.exit(process.exitCode || 0);
