// R2_CHIPS in a real browser: a list of N shots (the chips and All's rows drawn in part, K and
// Find shot keeping the current one in view and marked, the chips' geometry measured on a real
// chip), a drafted span deleted with the focus left in the event dropdown or in Resolution, keys
// after a pick in the list dropdown, the event's default class drawn in the label's unmarked time
// on the event page and in All, and All's roster walk across events (fix round I1-I4).
// Appended to review_browser.mjs's transport (everything before its `const opened =`) by
// test_review_long_list_browser.py, and run as
//   node <script> <base-url> <token> <headless-shell> <profile-dir> <N>
// Prints one JSON line: the checks, [{name, ok, detail}], in order.

const N = Number(SCENARIO);
const AE = "alfven_eigenmode", LOCK = "locked_mode", SHOT = 170815;
const results = [];
const check = (name, ok, detail) => results.push({ name, ok: Boolean(ok), detail });
let loads = 0;

/** A key press where the focus is (`blur` first leaves any control); `modifiers` 2 is Ctrl. */
async function key(name, { blur = true, modifiers = 0 } = {}) {
  if (blur) await js("document.activeElement?.blur?.()");
  const named = { Enter: [13, "\r"], Delete: [46, ""] }[name];
  const [code, text] = named || [name.toUpperCase().charCodeAt(0), name];
  const event = { key: name, code: named ? name : `Key${name.toUpperCase()}`, windowsVirtualKeyCode: code, modifiers };
  await send("Input.dispatchKeyEvent", { type: "keyDown", ...event, ...(modifiers || !text ? {} : { text }) });
  await send("Input.dispatchKeyEvent", { type: "keyUp", ...event });
}

/** A left click at (x, y) without moving, through the browser's hit test. */
async function click(x, y) {
  const mouse = (type, more = {}) => send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, ...more });
  await mouse("mouseMoved", { button: "none" });
  await mouse("mousePressed", { buttons: 1 });
  await mouse("mouseReleased", { buttons: 0 });
}

/** Load the page at `hash` in a new document, with `storage` set before its scripts run. */
async function navigate(hash, storage = {}) {
  const setup = "localStorage.clear();" +
    Object.entries(storage).map(([k, v]) => `localStorage.setItem(${JSON.stringify(k)}, ${JSON.stringify(v)});`).join("");
  const { identifier } = await send("Page.addScriptToEvaluateOnNewDocument", { source: setup });
  await js("window.left = true");
  await send("Page.navigate", { url: `${BASE}/?token=${TOKEN}&n=${++loads}${hash}` });
  await until(`!window.left && document.readyState === "complete" && typeof S !== "undefined" && typeof Lists !== "undefined"`);
  await send("Page.removeScriptToEvaluateOnNewDocument", { identifier });
}

const opened = (shot, event = AE) => until(`S.shot === ${shot} && S.event === "${event}" && S.data !== null &&
  S.opened?.ticket === S.ticket`, 30000);

/** ms from the last key down to the first frame where `condition` holds. */
async function timed(press, condition) {
  await js(`window.watch = { seen: null }; (function loop() {
    if (${condition}) watch.seen = performance.now(); else requestAnimationFrame(loop); })(); void 0`);
  await press();
  await until("watch.seen !== null", 30000);
  return js("Math.round(watch.seen - keys.at(-1))");
}

/** Pick `value` in the select `id` as a person does: it takes the focus, then changes. */
const pick = (id, value) => js(`(() => { const e = $("${id}"); e.focus(); e.value = ${JSON.stringify(value)};
  e.dispatchEvent(new Event("change", { bubbles: true })); })()`);

/** Keys pressed at once (one task), where the focus is not in a control. */
const press = (keys) => js(`${JSON.stringify(keys)}.forEach((key) =>
  document.body.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true }))); void 0`);

/** The current element in `box` (a selector): its shot, whether it lies inside the box, and how
 * many elements the box holds. */
const current = (box, selector) => js(`(() => {
  const box = document.querySelector(${JSON.stringify(box)}), b = box.getBoundingClientRect();
  const found = box.querySelectorAll(${JSON.stringify(selector)}), r = found[0]?.getBoundingClientRect();
  return { shot: found.length === 1 ? Number(found[0].dataset.shot) : null,
    inside: Boolean(r && r.height && r.top >= b.top - 1 && r.bottom <= b.bottom + 1),
    elements: box.querySelectorAll("*").length };
})()`);

await send("Runtime.enable");
await send("Page.enable");
await send("Page.addScriptToEvaluateOnNewDocument", { source: `
  sessionStorage.setItem("labeler:who", "Long list reviewer");
  window.keys = [];
  addEventListener("keydown", (event) => keys.push(event.timeStamp), true);
` });

try {
  // 1. The label panel's chips on a list of N: a window of them, the open shot's in view.
  await navigate(`#${AE}/${SHOT}`, { [`labeler:list:${AE}`]: "big" });
  await opened(SHOT);
  await until(`S.queue.length === ${N} && document.querySelector("#queue .chip.current")`);
  const loaded = await current("#queue", ".chip.current");
  const states = await js(`[...document.querySelectorAll("#queue .chip")].every((chip) => {
    const row = S.queue.find((r) => r.shot === Number(chip.dataset.shot));
    return chip.classList.contains(row.state) && chip.classList.contains("current") === (row.shot === S.shot);
  })`);
  check(`the chips of a ${N}-shot list are drawn in part, the open shot's in view and marked, each chip in its state`,
    loaded.shot === SHOT && loaded.inside && loaded.elements < 1000 && states, { loaded, states });

  // I4 + M1: the line LongList reckons is a real chip's (pitch and chips per line); scrolled far
  // down, the view stays where it was put (no snap back), the chips sit where the line says and
  // the scroll height does not change.
  const geometry = (to) => js(`(() => {
    const box = $("queue"), chips = [...box.querySelectorAll(".chip")];
    const tops = chips.map((chip) => Math.round(chip.getBoundingClientRect().top));
    const { line, per } = queueChips.size, at = Math.floor(box.scrollTop / line);
    const top = box.querySelector('[data-shot="' + S.queue[at * per].shot + '"]')?.getBoundingClientRect().top;
    const own = getComputedStyle(box), b = box.getBoundingClientRect();
    return { line, per, real_per: tops.filter((t) => t === tops[0]).length, pitch: tops.find((t) => t > tops[0]) - tops[0],
      height: box.scrollHeight, scroll: box.scrollTop, wanted: ${to},
      off: top == null ? null : Math.round((top - b.top - box.clientTop + box.scrollTop - parseFloat(own.paddingTop) - at * line) * 10) / 10 };
  })()`);
  const before = await geometry(null);
  await js(`$("queue").scrollTop = 2500; void 0`);
  await sleep(300);
  const after = await geometry(2500);
  check("the chips' line is a real chip's: the pitch and chips per line match, a far scroll stays put with the chips where the line says",
    Math.abs(before.line - before.pitch) < 0.5 && before.per === before.real_per && Math.abs(after.scroll - 2500) <= 1 &&
    after.off !== null && Math.abs(after.off) <= 1 && Math.abs(after.height - before.height) <= 1, { before, after });

  // 2. K: the next shot's chip marked and in view, quickly.
  const k = await timed(() => key("k"), `S.shot === ${SHOT + 1} && document.querySelector("#queue .chip.current")?.dataset.shot === "${SHOT + 1}"`);
  await opened(SHOT + 1);
  const afterK = await current("#queue", ".chip.current");
  check("K marks the next chip in place, in view, within 500 ms", k < 500 && afterK.shot === SHOT + 1 && afterK.inside,
    { k_ms: k, afterK });

  // 3. Find shot after the chips were scrolled away: the shot's chip is drawn, in view, marked.
  await js(`$("queue").scrollTop = 0; void 0`);
  await until(`!document.querySelector("#queue .chip.current")`);
  await js(`$("shot").focus(); $("shot").value = ""; void 0`);
  await send("Input.insertText", { text: String(SHOT + 2) });
  await key("Enter", { blur: false });
  await opened(SHOT + 2);
  await until(`document.querySelector("#queue .chip.current")?.dataset.shot === "${SHOT + 2}"`);
  const found = await current("#queue", ".chip.current");
  check("Find shot brings the found shot's chip back into view, marked", found.shot === SHOT + 2 && found.inside, found);

  // I2 + M2: a pick in the list dropdown gives the keys back (K steps on the new list, wrapping
  // to its first shot); back on the long list, the open shot's chip is drawn, marked and in view.
  await pick("shotlist", "roster");
  await until(`Lists.name === "roster" && S.queue.length === 3`);
  const focus = await js("document.activeElement?.id || null");
  await key("k", { blur: false });
  const stepped = await opened(SHOT).then(() => true, () => false); // a failed check, not the end
  await pick("shotlist", "big");
  await until(`Lists.name === "big" && S.queue.length === ${N}`);
  await until(`document.querySelector("#queue .chip.current")`, 5000).catch(() => null);
  const shown = await current("#queue", ".chip.current");
  check("after a pick in the list dropdown K steps on the new list; back on the long list the open shot's chip is in view",
    focus !== "shotlist" && stepped && shown.shot === SHOT && shown.inside, { focus, stepped, shown });

  // 4. All on the same list: a window of rows; K marks the next row in view.
  await navigate(`#list/big/${SHOT}`);
  await until(`Lists.byShot && Lists.overview?.shots.length === ${N} && Lists.strips?.shot === ${SHOT} &&
    document.querySelector("#silver-shots .silver-shot.current")`, 30000);
  const rows = await current("#silver-shots", ".silver-shot.current");
  const allK = await timed(() => key("k"), `Lists.shot === ${SHOT + 1} &&
    document.querySelector("#silver-shots .silver-shot.current")?.dataset.shot === "${SHOT + 1}"`);
  const rowsK = await current("#silver-shots", ".silver-shot.current");
  check("All draws the list's rows in part; K marks the next row in place, in view, within 500 ms",
    rows.shot === SHOT && rows.inside && rows.elements < 2000 && allK < 500 && rowsK.shot === SHOT + 1 && rowsK.inside,
    { rows, all_k_ms: allK, rowsK });

  // 5. The drafted span, with the focus left in the event dropdown (as choosing the event leaves
  // it): a click selects it and Delete removes it; Ctrl+Z puts it back; Delete again; Save.
  await navigate(`#${AE}/${SHOT}`);
  await opened(SHOT);
  const drafted = await js("S.meta.saved === null && same(S.label, S.meta.source) && same(S.label.iscrowd, [1])");
  const spanAt = (t) => js(`(() => { const r = $("crowd-track").getBoundingClientRect(); return [r.left + px(${t}), r.top + 15]; })()`);
  await js(`$("event").focus(); void 0`);
  await click(...(await spanAt(200)));
  const picked = await js(`[S.selected, document.activeElement === $("event")]`);
  await key("Delete", { blur: false });
  const gone = await js("S.label.intervals.length === 0 && dirty()");
  await key("z", { blur: false, modifiers: 2 });
  const back = await js("same(S.label, S.meta.source)");
  await click(...(await spanAt(200)));
  await key("Delete", { blur: false });
  const goneAgain = await js("S.label.intervals.length === 0");
  // I2: back again, made Individual in Resolution (the focus left there), then Delete.
  await key("z", { blur: false, modifiers: 2 });
  await click(...(await spanAt(200)));
  await pick("resolution", "0");
  const individual = await js(`[S.selected, S.label.iscrowd?.[S.selected] ?? null, document.activeElement?.id || null]`);
  await key("Delete", { blur: false });
  const goneResolved = await js("S.label.intervals.length === 0");
  await key("s");
  await until("!S.saving && !dirty() && S.meta.saved !== null");
  check("a drafted span is selected and deleted with the focus left in the event dropdown, Ctrl+Z restores it, Save writes the label without it",
    drafted && picked[0] === 0 && !picked[1] && gone && back && goneAgain, { drafted, picked, gone, back, goneAgain });
  check("a span made Individual in Resolution is deleted by the next Delete (the dropdown gives the keys back)",
    individual[0] === 0 && individual[1] === 0 && individual[2] !== "resolution" && goneResolved, { individual, goneResolved });

  // 6. The event's default class (here uncertain) in the unmarked time: said once in the class
  // bar, drawn lighter in the window and only there, never a span: a click there selects nothing.
  const pixel = (t) => js(`(() => { const c = $("crowd-track"), r = devicePixelRatio;
    return [...c.getContext("2d").getImageData(Math.round(px(${t}) * r), Math.round(c.height / 2), 1, 1).data]; })()`);
  const note = await js(`$("unmarked")?.textContent ?? null`);
  await sleep(100);
  const tinted = await pixel(1500);
  await js("window.keep = unmarkedClass; unmarkedClass = () => 0; render(); void 0");
  await sleep(100);
  const plain = await pixel(1500);
  await js("unmarkedClass = window.keep; render(); void 0");
  await click(...(await spanAt(1500)));
  const none = await js("S.selected === -1 && S.label.intervals.length === 0 && !dirty()");
  await navigate(`#${LOCK}/${SHOT}`);
  await opened(SHOT, LOCK);
  const lockNote = await js(`$("unmarked")?.textContent ?? null`);
  check("an event's default class is said in the class bar and drawn in the unmarked time, not as a span; an event without one shows neither",
    note === "Unmarked: uncertain" && JSON.stringify(tinted) !== JSON.stringify(plain) && none && lockNote === null, { note, tinted, plain, none, lockNote });

  // I1: All on 170816 (AE saved: window 0-2000, one span 400-600): the strip's unmarked time is
  // AE's default class, as the export reads it, drawn lighter than the span.
  await navigate(`#list/big/${SHOT + 1}`);
  await until(`Lists.byShot && Lists.strips?.shot === ${SHOT + 1} && $("silver-strips").querySelector("canvas")`, 30000);
  await sleep(200);
  const strip = await js(`(() => {
    const event = Lists.strips.events.find((row) => row.event === "${AE}"), [lo, hi] = Lists.strips.t_range;
    const frames = silverFrames(event, [lo, hi]), k = (t) => Math.floor((t - lo) / GRID_MS);
    const canvas = document.querySelector('#silver-strips section[data-event="${AE}"] canvas'), r = devicePixelRatio;
    const alpha = (t) => canvas.getContext("2d").getImageData(Math.round((t - lo) / (hi - lo) * canvas.clientWidth * r),
      Math.round(canvas.height / 2), 1, 1).data[3];
    return { fill: event.layers.find((layer) => layer.kind === "saved")?.fill, range: [lo, hi],
      unmarked: [frames.cells[k(1000)], frames.implied?.[k(1000)], alpha(1000)], span: [frames.cells[k(500)], frames.implied?.[k(500)], alpha(500)] };
  })()`);
  check("All's strip draws a save's unmarked time in the event's default class, lighter than its span",
    strip.fill === 2 && strip.unmarked[0] === 2 && strip.unmarked[1] === 1 && strip.unmarked[2] > 90 && strip.unmarked[2] < 170 &&
    strip.span[0] === 1 && strip.span[1] === 0 && strip.span[2] === 255, strip);

  // I3: All on locked_mode's roster; AE opened from its strip; G and K at once: All walks AE's
  // roster (G shows the open event's) and keeps it. Then on `big`: G and K at once step on the
  // shots in hand, and the list's fresh answer is still taken.
  await navigate(`#${LOCK}/${SHOT}`);
  await opened(SHOT, LOCK);
  await pick("event", "__all");
  await until(`Lists.byShot && Lists.overview?.event === "${LOCK}" && Lists.strips?.shot === ${SHOT}`, 30000);
  const lockShots = await js("Lists.overview.shots.length");
  await js(`document.querySelector('#silver-strips section[data-event="${AE}"] .silver-strip').click()`);
  await opened(SHOT);
  await press(["g", "k"]);
  await until(`Lists.byShot && Lists.overview?.event === "${AE}" && Lists.strips?.shot === Lists.shot`, 30000).catch(() => null);
  await sleep(300);
  const walk = await js(`({ event: Lists.overview?.event, shots: Lists.overview?.shots.length, name: Lists.name,
    rows: $("silver-shots").querySelectorAll(".silver-shot").length })`);
  await navigate(`#list/big/${SHOT}`);
  await until(`Lists.byShot && Lists.overview?.shots.length === ${N} && Lists.strips?.shot === ${SHOT}`, 30000);
  await js(`document.querySelector('#silver-strips section[data-event="${AE}"] .silver-strip').click()`);
  await opened(SHOT);
  await js("window.held = Lists.overview; void 0");
  await press(["g", "k"]);
  const fresh = await until(`Lists.byShot && Lists.overview && Lists.overview !== window.held && Lists.strips?.shot === Lists.shot`, 15000)
    .then(() => true, () => false);
  const moved = await js(`[Lists.shot, $("silver-shots").querySelector(".silver-shot.current")?.dataset.shot ?? null]`);
  check("G from an event opened off a roster walks that event's roster, and a K at once neither drops the fetch nor is undone",
    lockShots === 1 && walk.event === AE && walk.shots === 3 && walk.rows === 3 && walk.name === "roster" &&
    fresh && moved[0] === SHOT + 1 && moved[1] === String(SHOT + 1), { lockShots, walk, fresh, moved });
} catch (error) {
  check("the scenario ran to its end", false, String(error?.stack || error));
}

check("no script errors", errors.length === 0, errors);
console.log(JSON.stringify(results));
process.exit(0);
