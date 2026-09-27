// The review page in headless Chromium, driven over the DevTools protocol.
//   node review_browser.mjs <base-url> <token> <headless-shell> <profile-dir> [api1]
// Prints one JSON line: every check made, [{name, ok, detail}].
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

const [BASE, TOKEN, SHELL, PROFILE, SCENARIO] = process.argv.slice(2);
const browser = spawn(
  SHELL,
  ["--no-sandbox", "--disable-gpu", "--remote-debugging-port=0", `--user-data-dir=${PROFILE}`,
    "--window-size=1400,900", "about:blank"],
  { stdio: "ignore" }
);
process.on("exit", () => browser.kill());

let port;
for (let i = 0; i < 200 && !port; i++) {
  try {
    port = readFileSync(`${PROFILE}/DevToolsActivePort`, "utf8").split("\n")[0];
  } catch {
    await sleep(50);
  }
}
const page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((open) => ws.addEventListener("open", open));

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

async function drag(x0, x1, y) {
  await mouse("mouseMoved", x0, y, { button: "none" });
  await mouse("mousePressed", x0, y, { buttons: 1 });
  for (let i = 1; i <= 8; i++) await mouse("mouseMoved", x0 + ((x1 - x0) * i) / 8, y, { buttons: 1 });
  await mouse("mouseReleased", x1, y, { buttons: 0 });
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
  const named = { Enter: [13, "\r"], Escape: [27, ""], ArrowLeft: [37, ""], ArrowRight: [39, ""] }[key];
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
  await send("Runtime.enable");
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
  await send("Page.navigate", { url: `${BASE}/?token=${TOKEN}#alfven_eigenmode/170815` });
  await opened(170815);
  if (SCENARIO === "api1") await olderServer();
  else await currentServer();
} catch (error) {
  check("the page did what was asked", false, String(error));
}
check("no script error", errors.length === 0, errors);
console.log(JSON.stringify(checks));
process.exit(0);
