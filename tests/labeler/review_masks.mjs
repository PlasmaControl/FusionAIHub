// The AE pseudo-mask on the review page, in headless Chromium over the DevTools protocol.
//   node review_masks.mjs <base-url> <token> <headless-shell> <profile-dir>
// Prints one JSON line: every check made, [{name, ok, detail}].
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

const [BASE, TOKEN, SHELL, PROFILE, CASE = "happy"] = process.argv.slice(2);
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

const mouse = (type, x, y, more = {}) =>
  send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, ...more });

async function click(x, y) {
  await mouse("mouseMoved", x, y, { button: "none" });
  await mouse("mousePressed", x, y, { buttons: 1 });
  await mouse("mouseReleased", x, y, { buttons: 0 });
}

async function drag(x0, x1, y) {
  await mouse("mouseMoved", x0, y, { button: "none" });
  await mouse("mousePressed", x0, y, { buttons: 1 });
  for (let i = 1; i <= 8; i++) await mouse("mouseMoved", x0 + ((x1 - x0) * i) / 8, y, { buttons: 1 });
  await mouse("mouseReleased", x1, y, { buttons: 0 });
}

async function press(key) {
  const event = { key, code: `Key${key.toUpperCase()}`, windowsVirtualKeyCode: key.toUpperCase().charCodeAt(0) };
  await send("Input.dispatchKeyEvent", { type: "keyDown", ...event, text: key });
  await send("Input.dispatchKeyEvent", { type: "keyUp", ...event });
}

/** Where region `n`'s middle pixel is on the first row, in page px, and its colour there. */
const region = (n) =>
  js(`new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(() => {
    const m = S.masks;
    const found = m.regions.find((r) => r.id === ${n});
    const [j, k, len] = found.runs[Math.floor(found.runs.length / 2)];
    const t = m.grid.t0_ms + (k + len / 2) * m.grid.dt_ms;
    const f = m.y0_khz + j * m.dy_khz;
    const canvas = $("rows").children[0];
    const r = canvas.getBoundingClientRect();
    const [lo, hi] = imageRange(S.meta.rows[0]);
    const [x, y] = [px(t), (r.height - 1) * (1 - (f - lo) / (hi - lo))];
    const scale = canvas.width / canvas.clientWidth;
    const rgb = [...canvas.getContext("2d").getImageData(Math.round(x * scale), Math.round(y * scale), 1, 1).data];
    done({ x: r.left + x, y: r.top + y, rgb: rgb.slice(0, 3) });
  })))`);
const cyan = ([r, g, b]) => g > r + 60 && b > r + 60;
const grey = ([r, g, b]) => Math.max(r, g, b) - Math.min(r, g, b) < 40 && r > 60;

const checks = [];
const check = (name, ok, detail) => checks.push({ name, ok: Boolean(ok), detail });
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

try {
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1400, height: 900, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: `${BASE}/?token=${TOKEN}#alfven_eigenmode/170815` });
  await until(`typeof S !== "undefined" && S.shot === 170815 && S.data !== null && S.masks !== null`);
  if (CASE === "happy") {
  await js(`$("reviewer-name").focus()`);
  await send("Input.insertText", { text: "Ada" });
  await js("document.activeElement.blur()");
  check("an AE shot with a pseudo-mask says so", (await js(`$("masks").textContent`)) === "mask 2/2 kept",
    await js(`$("masks").textContent`));
  const one = await region(1);
  const two = await region(2);
  check("its regions are drawn over the rows", cyan(one.rgb) && cyan(two.rgb), [one, two]);

  await click(two.x, two.y);
  await until("!S.masks.saving && S.masks.last_save !== null");
  const after = await region(2);
  check(
    "a click rejects the region under it, saved with the name",
    same(await js("S.masks.rejected"), [2]) && grey(after.rgb) &&
      (await js(`$("masks").textContent === "mask 1/2 kept · saved by Ada"`)),
    [await js("S.masks.rejected"), after.rgb, await js(`$("masks").textContent`)]
  );

  const view = await js("S.view[0]");
  await drag(one.x, one.x - 120, one.y);
  await until(`S.view[0] !== ${view}`);
  check("a drag still pans and rejects nothing", same(await js("S.masks.rejected"), [2]), await js("S.masks.rejected"));

  await press("m");
  const hidden = await region(1);
  check(
    "m hides the mask, and the browser remembers",
    !cyan(hidden.rgb) && (await js(`localStorage.getItem("labeler:masks") === "hidden"`)) &&
      (await js(`$("masks").textContent.endsWith("· hidden")`)),
    [hidden.rgb, await js(`$("masks").textContent`)]
  );

  await send("Page.reload");
  await until(`typeof S !== "undefined" && S.shot === 170815 && S.data !== null && S.masks !== null`);
  check(
    "a reload keeps the rejection and the hiding",
    same(await js("S.masks.rejected"), [2]) && (await js("!S.showMasks")),
    await js("[S.masks.rejected, S.showMasks]")
  );
  await press("m");
  const back = await region(2);
  await click(back.x, back.y);
  await until("!S.masks.saving && S.masks.rejected.length === 0");
  check("a second click takes the rejection back", cyan((await region(2)).rgb), await region(2));

  await press("k");
  await until(`S.shot === 170816 && S.data !== null`);
  await sleep(300);
  check("a shot without a pseudo-mask shows none", await js(`S.masks === null && $("masks").hidden`),
    await js(`[S.masks, $("masks").hidden]`));
  } else if (CASE === "race") {
    await js(`window.realFetch = window.fetch.bind(window);
      window.fetch = async (url, options) => {
        if (String(url).includes('/api/masks') && options?.method === 'POST') {
          window.fetch = window.realFetch;
          await new Promise(resolve => { window.releaseMask = resolve; });
        }
        return window.realFetch(url, options);
      };`);
    const two = await region(2);
    await click(two.x, two.y);
    await until('S.masks.saving && typeof window.releaseMask === "function"');
    await click(two.x, two.y);
    check("a pending save blocks another click and says so",
      await js('$("status").textContent.includes("saving")'), await js('$("status").textContent'));
    await press("k");
    await until('S.shot === 170816 && S.data !== null');
    await press("j");
    await until('S.shot === 170815 && S.data !== null');
    await sleep(150);
    await js('window.releaseMask()');
    await until('S.masks !== null && !S.masks.saving && S.masks.last_save !== null');
    check("returning to the shot adopts the pending rejection",
      same(await js('S.masks.rejected'), [2]), await js('S.masks.rejected'));
    const one = await region(1);
    await click(one.x, one.y);
    await until('!S.masks.saving');
    check("the next rejection keeps both regions", same(await js('S.masks.rejected'), [1, 2]),
      await js('S.masks.rejected'));
  } else if (CASE === "failed") {
    await js(`window.realFetch = window.fetch.bind(window);
      window.fetch = async (url, options) => {
        if (String(url).includes('/api/masks') && options?.method === 'POST') {
          window.fetch = window.realFetch;
          return new Response(JSON.stringify({error: 'mask save failed'}), {status: 500});
        }
        return window.realFetch(url, options);
      };`);
    const two = await region(2);
    await click(two.x, two.y);
    await until('!S.masks.saving');
    check("a failed rejection reverts and shows the error",
      same(await js('S.masks.rejected'), []) && await js('$("status").textContent.includes("mask save failed")'),
      await js('[S.masks.rejected, $("status").textContent]'));
    await click(two.x, two.y);
    await until('!S.masks.saving && S.masks.last_save !== null');
  } else if (CASE === "conflict") {
    await js(`(async () => { await api('/api/masks', {
      method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify({event: S.event, shot: S.shot, rejected: [2],
        pseudo_sha256: S.masks.pseudo_sha256, revision: S.masks.revision || 0})
    }); })()`);
    const one = await region(1);
    await click(one.x, one.y);
    await until('!S.masks.saving');
    await sleep(200);
    check("a conflict reloads the current decisions and says why",
      same(await js('S.masks.rejected'), [2]) && await js('$("status").textContent.includes("decisions changed")'),
      await js('[S.masks.rejected, $("status").textContent]'));
    await click(one.x, one.y);
    await until('!S.masks.saving');
    check("the next click uses the new revision", same(await js('S.masks.rejected'), [1, 2]),
      await js('S.masks.rejected'));
  } else if (CASE === "stale_mask") {
    check("the page explains an old mask decision",
      await js('$("masks").textContent.includes("older mask was dropped")'), await js('$("masks").textContent'));
    const one = await region(1);
    await click(one.x, one.y);
    await until('!S.masks.saving');
    check("a click saves on the current mask",
      await js('!S.masks.stale && S.masks.last_save.pseudo_sha256 === S.masks.pseudo_sha256') &&
      same(await js('S.masks.rejected'), [1]), await js('S.masks.last_save'));
  }
} catch (error) {
  check("the page did what was asked", false, String(error));
}
check("no script error", errors.length === 0, errors);
console.log(JSON.stringify(checks));
process.exit(0);
