// The review page in headless Chromium, driven over the DevTools protocol.
//   node review_browser.mjs <base-url> <token> <headless-shell> <profile-dir>
// Prints one JSON line: every check made, [{name, ok, detail}].
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

const [BASE, TOKEN, SHELL, PROFILE] = process.argv.slice(2);
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

async function press(key, modifiers = 0) {
  const named = { Enter: [13, "\r"] }[key];
  const [code, text] = named || [key.toUpperCase().charCodeAt(0), key];
  const event = { key, code: named ? key : `Key${key.toUpperCase()}`, windowsVirtualKeyCode: code, modifiers };
  await send("Input.dispatchKeyEvent", { type: "keyDown", ...event, ...(modifiers ? {} : { text }) });
  await send("Input.dispatchKeyEvent", { type: "keyUp", ...event });
}

const checks = [];
const check = (name, ok, detail) => checks.push({ name, ok: Boolean(ok), detail });
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

try {
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1400, height: 900, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: `${BASE}/?token=${TOKEN}#alfven_eigenmode/170815` });
  await opened(170815);
  const source = await js("S.meta.source");
  check("the link opens its shot on its source label", same(await js("S.label"), source), source);

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
    "s saves it and says so",
    await js(`$("state").textContent === "changed" && $("saved").textContent.startsWith("saved")`),
    await js(`[$("state").textContent, $("saved").textContent]`)
  );

  await send("Page.reload");
  await opened(170815);
  check(
    "a reload opens the saved label",
    same(await js("S.label"), saved) && (await js(`$("count").textContent`)) === "1/2",
    await js("S.label")
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

  await press("Enter");
  await opened(170816);
  check("enter saves and opens the next unreviewed shot", true);

  await js(`$("shot").focus(); $("shot").value = "1"`);
  await press("Enter");
  await until(`$("rows").querySelector(".failure") !== null`);
  check("a shot off the roster says why", await js(`$("rows").textContent.includes("not on this event")`),
    await js(`$("rows").textContent`));
  await js("document.activeElement.blur()");
  await press("k");
  await opened(170815);
  check("k carries on from it", true);
} catch (error) {
  check("the page did what was asked", false, String(error));
}
check("no script error", errors.length === 0, errors);
console.log(JSON.stringify(checks));
process.exit(0);
