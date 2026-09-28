// The label review page: spectrogram rows on top, one editable label per shot below.
"use strict";

// ---- The label rules, as the server applies them; exported for the tests ----

const LONGEST_WINDOW_MS = 20000;
const HANDLE_BAND = 10; // px at the foot of the label track that hold the window's edges
const GRAB = 5; // px either side of an edge that grab it
const CLICK_PX = 4; // a press that moves less than this is a click; a click this close picks a region
const MASK_EVENT = "alfven_eigenmode"; // the one event with pseudo-masks
const INFERNO = [
  "000004", "0b0724", "210c4a", "3d0965", "57106e", "71196e", "8a226a", "a32c61", "bc3754",
  "d24644", "e45a31", "f1731d", "f98e09", "fcac11", "f9cb35", "f2ea69", "fcffa4",
];

/** Whole ms, halves up, as the server's `_ms`. */
function ms(t) {
  return Math.floor(Number(t) + 0.5);
}

/** One cell per ms of the window, holding the category painted there last. */
function paint(window, spans) {
  const [lo, hi] = window;
  const cells = new Int32Array(hi - lo);
  for (const [a, b, c] of spans) {
    const start = Math.max(a, lo) - lo;
    const stop = Math.min(b, hi) - lo;
    if (stop > start) cells.fill(c, start, stop);
  }
  return cells;
}

/** [start, stop, value] of each run of equal cells, as offsets. */
function runs(cells) {
  const out = [];
  for (let start = 0, i = 1; i <= cells.length; i++) {
    if (i === cells.length || cells[i] !== cells[start]) {
      out.push([start, i, cells[start]]);
      start = i;
    }
  }
  return out;
}

/** The server's `normalise`: snap to whole ms, grow the window over every span, merge
 * and clip; later spans paint over earlier ones. Null where the server would refuse. */
function normalise(window, intervals, known) {
  let lo = ms(window[0]);
  let hi = ms(window[1]);
  const spans = [];
  for (const [a0, b0, c0] of intervals) {
    const [a, b, c] = [ms(a0), ms(b0), Math.trunc(c0)];
    if (b < a || (known && c && !known.includes(c))) return null;
    if (b > a) spans.push([a, b, c]);
  }
  for (const [a, b, c] of spans) {
    if (c) [lo, hi] = [Math.min(lo, a), Math.max(hi, b)];
  }
  if (!(hi - lo > 0 && hi - lo <= LONGEST_WINDOW_MS)) return null;
  const painted = runs(paint([lo, hi], spans)).filter(([, , c]) => c);
  return { window: [lo, hi], intervals: painted.map(([a, b, c]) => [lo + a, lo + b, c]) };
}

/** The ms ranges where two labels disagree; outside a window counts as unknown. */
function diffRuns(a, b) {
  const lo = Math.min(a.window[0], b.window[0]);
  const hi = Math.max(a.window[1], b.window[1]);
  const cells = (label) => {
    const out = new Int32Array(hi - lo).fill(-1);
    out.fill(0, label.window[0] - lo, label.window[1] - lo);
    for (const [s, e, c] of label.intervals) out.fill(c, s - lo, e - lo);
    return out;
  };
  const [x, y] = [cells(a), cells(b)];
  return runs(x.map((v, i) => (v === y[i] ? 0 : 1)))
    .filter(([, , differs]) => differs)
    .map(([s, e]) => [lo + s, lo + e]);
}

/** Per saved version, the ms it changed from the one before; the first, from the source or null. */
function versionChanges(versions, source) {
  return versions.map((version, i) => {
    const before = i ? versions[i - 1] : source;
    return before ? diffRuns(before, version).reduce((n, [s, e]) => n + e - s, 0) : null;
  });
}

/** A tick step giving about `n` ticks over `span`: 1, 2 or 5 times a power of ten. */
function niceStep(span, n) {
  const raw = span / n;
  const power = 10 ** Math.floor(Math.log10(raw));
  const f = raw / power;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * power;
}

/** What a press at (x, y) on the label track grabs: a window edge (only in the foot
 * strip), a span edge, a span, or nothing. `px` maps ms to the track's px. */
function hitTest(label, x, y, height, px) {
  if (y >= height - HANDLE_BAND) {
    for (const edge of [0, 1]) {
      if (Math.abs(x - px(label.window[edge])) <= GRAB) return { kind: "window", edge };
    }
  }
  const spans = label.intervals;
  for (let index = spans.length - 1; index >= 0; index--) {
    for (const edge of [0, 1]) {
      if (Math.abs(x - px(spans[index][edge])) <= GRAB) return { kind: "edge", index, edge };
    }
  }
  for (let index = spans.length - 1; index >= 0; index--) {
    if (x > px(spans[index][0]) && x < px(spans[index][1])) return { kind: "move", index };
  }
  return { kind: "new" };
}

/** The pseudo-mask region with a pixel nearest bin `j`, column `k`, at most `tj` bins
 * and `tk` columns away; null if none. A region's `runs` are `[bin, first column,
 * length]`; on a tie the lower-numbered region wins. */
function regionAt(regions, j, k, tj = 0, tk = 0) {
  let [best, nearest] = [null, Infinity];
  for (const region of regions) {
    for (const [y, x, n] of region.runs) {
      const dy = Math.abs(y - j);
      const dx = k < x ? x - k : k >= x + n ? k - (x + n - 1) : 0;
      if (dy > tj || dx > tk) continue;
      const d = Math.hypot(dy / (tj + 1), dx / (tk + 1));
      if (d < nearest) [best, nearest] = [region, d];
    }
  }
  return best;
}

/** Colours for the byte values 0-255: inferno from `lo` up, its floor below `lo`. */
function lut(lo, hi = 255) {
  const anchors = INFERNO.map((hex) => [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)));
  const out = new Uint32Array(256);
  for (let v = 0; v < 256; v++) {
    const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo))) * (anchors.length - 1);
    const i = Math.min(anchors.length - 2, Math.floor(t));
    const [r, g, b] = anchors[i].map((c, k) => Math.round(c + (anchors[i + 1][k] - c) * (t - i)));
    out[v] = ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0; // RGBA bytes, little-endian
  }
  return out;
}

// ---- The page ----

const GUTTER = 96; // px left of every plot: a row's title and units, then its y ticks
const RIGHT = 12;
const IMAGE_H = 150;
const TRACE_H = 110;
const CATEGORY_COLOURS = { 2: "#e69f00", 3: "#cc79a7", 4: "#56b4e9" }; // 1 is --label
const TRACE_COLOURS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"];
const FONT = "11px system-ui, sans-serif";
const GUTTER_LINE = 14; // px between the lines of a row's title and units

const S = {
  events: [],
  event: null,
  categories: {},
  category: 1,
  queue: [],
  queueEvent: null, // whose queue has arrived; null while loading
  saveCount: 0,
  savedRows: new Map(), // event -> shot -> {count, row}, from acknowledged saves
  shot: null,
  meta: null, // what /api/shot said: grid, t_range, rows, source, saved, state, last_save
  label: null, // the label being edited, always normalised
  selected: -1,
  view: [0, 1], // the ms on screen
  data: null, // the last /api/rows: {t0, t1, n, rows, bitmaps}
  overview: null, // the widest /api/rows of this shot, drawn under `data`
  asked: "", // the last /api/rows query sent
  lo: 0,
  lut: lut(0),
  undo: [],
  drag: null,
  ticket: 0, // bumped per shot opened, so an answer for an older shot is dropped
  opened: null, // the finished navigation: ticket, event and shot, recorded by arrive
  versionsAt: null, // the finished navigation that asked for these versions
  frame: 0,
  timer: 0,
  saving: false, // or the event and shot whose save is in flight
  api: 1, // what /api/version said: 2 takes a name with each save and lists versions
  name: "", // the reviewer's typed name, sent with each save
  versions: [], // the open shot's saved versions, as /api/history listed them
  masks: null, // the open AE shot's pseudo-mask regions, as /api/masks gave them (api 3)
  showMasks: true, // M shows and hides them; the browser remembers which
};
let T = {}; // colour tokens, read from the stylesheet
const $ = (id) => document.getElementById(id);
const enc = encodeURIComponent;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const sleep = (delay) => new Promise((done) => setTimeout(done, delay));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const known = () => Object.keys(S.categories).map(Number);
const draftKey = (shot) => `labeler:${S.event}:${shot}`;

function stored(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function store(key, value) {
  try {
    if (value == null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // storage refused (a private window): a draft then lasts as long as the page
  }
}

async function api(path, options) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(body.error || `${response.status} ${response.statusText}`);
    error.status = response.status;
    throw error;
  }
  return response;
}

/** A save time as the header shows it. */
function when(iso) {
  const style = { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" };
  return new Date(iso).toLocaleString([], style);
}

function say(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function readTokens() {
  const style = getComputedStyle(document.documentElement);
  const names = ["panel", "ink", "muted", "rule", "label", "changed", "veil", "mask", "rejected"];
  T = Object.fromEntries(names.map((name) => [name, style.getPropertyValue(`--${name}`).trim()]));
}

// -- time and geometry

const plotWidth = () => Math.max(1, $("axis-row").clientWidth - GUTTER - RIGHT);
const px = (t) => GUTTER + ((t - S.view[0]) / (S.view[1] - S.view[0])) * plotWidth();
const middle = () => (S.view[0] + S.view[1]) / 2;

function timeAt(clientX) {
  const x = clientX - $("axis-row").getBoundingClientRect().left;
  return S.view[0] + ((x - GUTTER) / plotWidth()) * (S.view[1] - S.view[0]);
}

function setView(t0, t1) {
  const lo = Math.min(S.meta.t_range[0], S.label.window[0]);
  const hi = Math.max(S.meta.t_range[1], S.label.window[1]);
  const span = clamp(t1 - t0, 5, hi - lo);
  const start = clamp(t0, lo, hi - span);
  S.view = [start, start + span];
  render();
  fetchRows();
}

function zoomAt(t, factor) {
  const [v0, v1] = S.view;
  setView(t - (t - v0) * factor, t + (v1 - t) * factor);
}

function pan(fraction) {
  const shift = (S.view[1] - S.view[0]) * fraction;
  setView(S.view[0] + shift, S.view[1] + shift);
}

function fit() {
  const [lo, hi] = S.label.window;
  setView(lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.05);
}

// -- the label

function emptyLabel() {
  const lo = ms(S.meta.t_range[0]);
  return { window: [lo, Math.min(ms(S.meta.t_range[1]), lo + LONGEST_WINDOW_MS)], intervals: [] };
}

const baseline = () => S.meta.saved || S.meta.source || emptyLabel();
const dirty = () => Boolean(S.meta) && !same(S.label, baseline());

function draft(shot) {
  try {
    const label = JSON.parse(stored(draftKey(shot)));
    return label && normalise(label.window, label.intervals, known());
  } catch {
    return null;
  }
}

function keepForUndo(label) {
  S.undo.push(label);
  if (S.undo.length > 100) S.undo.shift();
}

/** Take a new label if the server would accept it. */
function edit(window, intervals) {
  if (stillOpening()) return;
  if (!S.meta) return;
  const next = normalise(window, intervals, known());
  if (!next) return;
  keepForUndo(S.label);
  S.label = next;
  touch();
}

/** Keep the draft until it is saved or reverted, and mark the shot unsaved. */
function touch() {
  if (stillOpening()) return;
  const saving = S.saving && S.saving.event === S.event && S.saving.shot === S.shot;
  const keep = dirty() || (saving && !same(S.label, S.saving.label));
  store(draftKey(S.shot), keep ? JSON.stringify(S.label) : null);
  $("dirty").hidden = !dirty();
  renderQueue();
  render();
}

/** Select the span holding `t`, if any. */
function selectAt(t) {
  S.selected = S.label.intervals.findIndex(([a, b]) => a <= t && t <= b);
}

function undo() {
  if (stillOpening()) return;
  if (!S.undo.length) return;
  S.label = S.undo.pop();
  S.selected = -1;
  touch();
}

function revert() {
  if (stillOpening()) return;
  if (!S.meta) return;
  const source = S.meta.source || emptyLabel();
  S.selected = -1;
  edit(source.window, source.intervals);
}

function removeSelected() {
  if (stillOpening()) return;
  if (S.selected < 0) return;
  const kept = S.label.intervals.filter((_, i) => i !== S.selected);
  S.selected = -1;
  edit(S.label.window, kept);
}

function setCategory(c) {
  if (stillOpening()) return;
  if (!known().includes(c)) return;
  S.category = c;
  renderSwatches();
  if (S.selected < 0) return;
  const [a, b] = S.label.intervals[S.selected];
  edit(S.label.window, S.label.intervals.map((s, i) => (i === S.selected ? [a, b, c] : s)));
  selectAt((a + b) / 2);
}

function contrast(delta) {
  S.lo = clamp(S.lo + delta, 0, 192);
  S.lut = lut(S.lo);
  store("labeler:contrast", String(S.lo));
  for (const data of [S.data, S.overview]) data?.bitmaps.clear();
  render();
}

// -- loading

async function boot() {
  readTokens();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    readTokens();
    renderSwatches();
    render();
  });
  S.lo = clamp(Number(stored("labeler:contrast")) || 0, 0, 192);
  S.lut = lut(S.lo);
  S.name = stored("labeler:name") || "";
  $("reviewer-name").value = S.name;
  S.showMasks = stored("labeler:masks") !== "hidden";
  wire();
  try {
    S.api = (await (await api("/api/version")).json()).api;
  } catch {
    S.api = 1; // a server older than this page: save without a name, no history
  }
  for (const id of ["reviewer-name", "show-versions"]) $(id).hidden = S.api < 2;
  $("stale").hidden = S.api >= 2;
  try {
    S.events = (await (await api("/api/events")).json()).events;
    $("event").replaceChildren(
      ...S.events.map((row) => {
        const option = new Option(row.error ? `${row.event} (broken)` : row.event, row.event);
        option.disabled = Boolean(row.error);
        option.title = row.error || "";
        return option;
      })
    );
    const usable = S.events.filter((row) => !row.error);
    if (!usable.length) return say("no events", true);
    const [hashEvent, hashShot] = fromHash();
    const pick =
      [hashEvent, stored("labeler:event")].find((name) => usable.some((r) => r.event === name)) ||
      usable.reduce((a, b) => (b.n_shots > a.n_shots ? b : a)).event;
    await openEvent(pick, hashShot);
  } catch (error) {
    say(error.message, true);
  }
}

/** `#event/shot`, the address of what is open. */
function fromHash() {
  const [event, shot] = decodeURIComponent(location.hash.slice(1)).split("/");
  return [event, Number(shot) || null];
}

/** A link pasted into the address bar opens what it names. */
function followHash() {
  const [event, shot] = fromHash();
  if (event !== S.event && S.events.some((row) => row.event === event && !row.error)) {
    openEvent(event, shot).catch((error) => say(error.message, true));
  } else if (event === S.event && shot && shot !== S.shot) {
    openShot(shot);
  }
}

const pendingNavigation = () => S.ticket !== S.opened?.ticket;

/** Refuse changes to the shot left on screen until its replacement has arrived. */
function stillOpening() {
  if (!pendingNavigation()) return false;
  say("the shot is still opening");
  return true;
}

function leave() {
  closeDialog($("versions"));
  S.versions = [];
  S.versionsAt = null;
  S.drag = null;
  clearTimeout(S.timer);
}

async function openEvent(event, shot) {
  const ticket = ++S.ticket;
  leave();
  S.queue = [];
  S.queueEvent = null;
  S.event = event;
  S.categories = S.events.find((row) => row.event === event).categories;
  S.category = known()[0] || 1;
  $("event").value = event;
  store("labeler:event", event);
  renderSwatches();
  renderQueue();
  showHeader();
  const saveCount = S.saveCount;
  let queue;
  try {
    queue = await (await api(`/api/queue?event=${enc(event)}`)).json();
  } catch (error) {
    if (S.event !== event) return;
    S.queueEvent = event;
    showNothing(null, `${event}: the queue could not be read: ${error.message}. Choose the event again to retry.`);
    return;
  }
  if (S.event !== event) return;
  const saved = S.savedRows.get(event);
  S.queue = queue.shots.map((row) => {
    const newer = saved?.get(row.shot);
    return newer && newer.count > saveCount ? newer.row : row;
  });
  S.queueEvent = event;
  if ($("status").textContent === "the queue is still loading") say("");
  renderQueue();
  showHeader();
  if (ticket !== S.ticket) return;
  await openShot(S.queue.some((row) => row.shot === shot) ? shot : queue.resume);
}

/** Show `shot` (or no shot) with nothing to edit, and the note `text` where the rows go. */
function showNothing(shot, text) {
  cancelAnimationFrame(S.frame);
  Object.assign(S, { shot, meta: null, data: null, overview: null, label: null,
    undo: [], selected: -1, asked: "", frame: 0, masks: null });
  showMasks();
  const note = document.createElement("p");
  note.className = "failure";
  note.textContent = text;
  $("rows").replaceChildren(note);
  for (const canvas of document.querySelectorAll("#top canvas, .track canvas")) context(canvas);
  arrive();
}

async function openShot(shot) {
  const ticket = ++S.ticket;
  leave();
  if (shot == null) {
    showNothing(null, `${S.event} has no shots to review.`);
    return;
  }
  try {
    let response, meta;
    for (;;) {
      response = await api(`/api/shot?event=${enc(S.event)}&shot=${shot}`);
      meta = await response.json();
      if (ticket !== S.ticket) return;
      if (response.status !== 202) break;
      const p = meta.progress;
      busy(`${shot}: building${p ? ` ${p.done}/${p.total} ${p.stage}` : ""}`);
      await sleep(800);
      if (ticket !== S.ticket) return;
    }
    Object.assign(S, { shot, meta, data: null, overview: null, asked: "", undo: [], selected: -1, masks: null });
    S.label = draft(shot) || baseline();
    buildRows();
    arrive();
    fit();
    fetchRows(0);
    prefetch();
    loadMasks(ticket);
  } catch (error) {
    if (ticket !== S.ticket) return;
    // Stay on the shot, with nothing to edit, so J, K and U carry on from it.
    showNothing(shot, `${shot} has nothing to show: ${error.message}. K opens the next shot.`);
  }
}

/** The header, the address and the queue follow the shot just opened. */
function arrive() {
  S.opened = { ticket: S.ticket, event: S.event, shot: S.shot };
  busy(null);
  $("cursor").hidden ||= !S.meta;
  for (const id of ["save-next", "revert", "show-versions"]) $(id).disabled = !S.meta;
  history.replaceState(null, "", `#${S.event}${S.shot == null ? "" : `/${S.shot}`}`);
  $("shot").value = S.shot ?? "";
  showHeader();
  renderQueue();
  $("queue").querySelector(".current")?.scrollIntoView({ block: "nearest" });
}

/** Dim the shot on screen while the next one builds. */
function busy(text) {
  document.body.classList.toggle("busy", Boolean(text));
  say(text || "");
}

/** Ask for the next shot now, so a build it needs is done when the reviewer gets there. */
function prefetch() {
  const next = neighbour(1);
  if (next != null && next !== S.shot) api(`/api/shot?event=${enc(S.event)}&shot=${next}`).catch(() => {});
}

function stillLoadingQueue() {
  if (S.queueEvent === S.event) return false;
  say("the queue is still loading");
  return true;
}

function nextUnreviewed() {
  const i = S.queue.findIndex((row) => row.shot === S.shot);
  const after = [...S.queue.slice(i + 1), ...S.queue.slice(0, Math.max(i, 0))];
  return after.find((row) => row.state === "unreviewed")?.shot ?? null;
}

/** The shot `delta` places from `shot` in queue order, wrapping at the ends. */
function neighbour(delta, shot = S.shot) {
  const i = S.queue.findIndex((row) => row.shot === shot);
  return S.queue[(i + delta + S.queue.length) % S.queue.length]?.shot ?? null;
}

/** Open the shot `delta` places along; an unsaved edit stays behind as a draft. */
async function go(delta) {
  if (stillLoadingQueue()) return;
  const saving = S.saving && S.saving.event === S.event && S.saving.shot === S.shot &&
    same(S.label, S.saving.label);
  const left = !pendingNavigation() && !saving && dirty() ? S.shot : null;
  await openShot(neighbour(delta));
  if (left != null && left !== S.shot) say(`${left}: the edit is kept as a draft, not saved`);
}

function fetchRows(delay = 90) {
  clearTimeout(S.timer);
  S.timer = setTimeout(loadRows, delay);
}

async function loadRows() {
  if (!S.meta || pendingNavigation()) return;
  const [v0, v1] = S.view;
  // A fifth wider than the view either side, so a pan has data while the next one loads.
  const t0 = Math.max(S.meta.t_range[0], v0 - (v1 - v0) * 0.2);
  const t1 = Math.min(S.meta.t_range[1], v1 + (v1 - v0) * 0.2);
  if (t1 <= t0) return;
  const cols = clamp(Math.round((plotWidth() * (t1 - t0)) / (v1 - v0)), 16, 8192);
  const query = `event=${enc(S.event)}&shot=${S.shot}&t0=${t0}&t1=${t1}&cols=${cols}`;
  if (query === S.asked) return;
  S.asked = query;
  const ticket = S.ticket;
  try {
    const response = await api(`/api/rows?${query}`);
    const grid = JSON.parse(response.headers.get("X-Grid"));
    const buffer = await response.arrayBuffer();
    if (ticket !== S.ticket || query !== S.asked) return;
    S.data = { ...grid, rows: unpack(buffer, grid.n), bitmaps: new Map() };
    if (!S.overview || grid.t1 - grid.t0 > S.overview.t1 - S.overview.t0) S.overview = S.data;
    render();
  } catch (error) {
    if (ticket === S.ticket) say(error.message, true);
  }
}

/** Split /api/rows' bytes into rows: images uint8 (n_y, n), traces float32 (2, C, n). */
function unpack(buffer, n) {
  let offset = 0;
  return S.meta.rows.map((row) => {
    if (row.kind === "image") {
      const values = new Uint8Array(buffer, offset, row.n_y * n);
      offset += row.n_y * n;
      return values;
    }
    const bytes = 2 * row.n_channels * n * 4;
    const values = new Float32Array(buffer.slice(offset, offset + bytes)); // a copy: aligned
    offset += bytes;
    return values;
  });
}

async function save(next) {
  if (stillOpening()) return;
  if (!S.meta || S.saving) return;
  const [event, shot, key, ticket, label] = [S.event, S.shot, draftKey(S.shot), S.ticket, S.label];
  S.saving = { event, shot, label };
  const name = S.api >= 2 ? { name: S.name || null } : {};
  try {
    const response = await api("/api/label", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ event, shot, ...label, ...name }),
    });
    const body = await response.json();
    const saved = S.savedRows.get(event) || new Map();
    saved.set(shot, { count: ++S.saveCount, row: body.row });
    S.savedRows.set(event, saved);
    try {
      if (same(JSON.parse(stored(key)), label)) store(key, null);
    } catch {
      // A malformed draft does not prevent the completed save from being shown.
    }
    if (!pendingNavigation() && S.opened?.event === event && S.opened.shot === shot && S.meta) {
      Object.assign(S.meta, { saved: body.saved, last_save: body.last_save });
      if (same(S.label, label)) {
        S.label = body.saved;
        render();
      } else {
        touch(); // a Restore may have been clean against the previous save
      }
    }
    if (S.event === event) {
      S.queue = S.queue.map((row) => (row.shot === shot ? body.row : row));
      renderQueue();
      showHeader();
    }
    if (next && ticket === S.ticket) await go(1);
  } catch (error) {
    say(`${shot} not saved: ${error.message}`, true);
  } finally {
    S.saving = false;
  }
}

// -- drawing

function buildRows() {
  $("rows").replaceChildren(...S.meta.rows.map(() => document.createElement("canvas")));
  sizeCanvases();
}

/** Each row at its own height, or taller when the rows would not fill the view. */
function sizeRows() {
  if (!S.meta) return;
  const base = S.meta.rows.map((row) => (row.kind === "image" ? IMAGE_H : TRACE_H));
  const room = $("top").clientHeight - $("axis-row").offsetHeight;
  const scale = Math.max(1, room / base.reduce((a, b) => a + b, 0));
  [...$("rows").children].forEach((canvas, i) => {
    canvas.style.height = `${Math.floor(base[i] * scale)}px`;
  });
}

function sizeCanvases() {
  sizeRows();
  const ratio = window.devicePixelRatio || 1;
  for (const canvas of document.querySelectorAll("canvas")) {
    const [w, h] = [canvas.clientWidth * ratio, canvas.clientHeight * ratio].map(Math.round);
    if (canvas.width !== w || canvas.height !== h) [canvas.width, canvas.height] = [w, h];
  }
}

/** A cleared 2D context drawing in CSS px. */
function context(canvas) {
  const g = canvas.getContext("2d");
  const ratio = canvas.width / Math.max(1, canvas.clientWidth);
  g.setTransform(ratio, 0, 0, ratio, 0, 0);
  g.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  return g;
}

function render() {
  if (S.frame || !S.meta) return;
  S.frame = requestAnimationFrame(() => {
    S.frame = 0;
    drawRows();
    drawAxis();
    drawTrack($("source-track"), S.meta.source, false);
    drawTrack($("label-track"), S.label, true);
  });
}

const categoryColour = (c) => (c === 1 ? T.label : CATEGORY_COLOURS[c] || T.muted);

function drawRows() {
  const canvases = $("rows").children;
  S.meta.rows.forEach((row, i) => {
    const canvas = canvases[i];
    const [w, h] = [canvas.clientWidth, canvas.clientHeight];
    const g = context(canvas);
    const values = S.data && S.data.rows[i];
    const range = row.kind === "image" ? imageRange(row) : values && traceRange(row, values);
    g.save();
    g.beginPath();
    g.rect(GUTTER, 0, w - GUTTER - RIGHT, h);
    g.clip();
    if (row.kind === "image") drawImage(g, row, i, h);
    if (row.kind === "image") drawMasks(g, row, h);
    if (values && row.kind === "trace") drawTrace(g, row, values, range, w, h);
    drawOverlay(g, w, h);
    g.restore();
    drawGutter(g, row, range, h);
    g.fillStyle = T.rule;
    g.fillRect(0, h - 1, w, 1);
  });
}

/** The bins an image row shows, [first, stop): its band if it has one, else all. */
function bandBins(row) {
  if (!row.band) return [0, row.n_y];
  const bin = (y) => clamp(Math.round((y - row.y0) / row.dy), 0, row.n_y - 1);
  return [bin(row.band[0]), bin(row.band[1]) + 1];
}

function imageRange(row) {
  const [first, stop] = bandBins(row);
  return [row.y0 + (first - 0.5) * row.dy, row.y0 + (stop - 0.5) * row.dy];
}

function drawImage(g, row, i, h) {
  const [first, stop] = bandBins(row);
  g.imageSmoothingEnabled = false;
  // The overview goes first, so a zoom-out shows it where its own rows have not arrived.
  for (const data of new Set([S.overview, S.data])) {
    if (!data) continue;
    const [x0, x1] = [px(data.t0), px(data.t1)];
    g.drawImage(bitmap(row, data, i), 0, row.n_y - stop, data.n, stop - first, x0, 0, x1 - x0, h - 1);
  }
}

/** The pseudo-mask over an image row: `--mask` where TokEye's line lies inside the
 * label's AE frames, `--rejected` where the reviewer rejected the region. */
function drawMasks(g, row, h) {
  const m = S.masks;
  if (!m || !S.showMasks) return;
  const [lo, hi] = imageRange(row);
  const y = (f) => (h - 1) * (1 - (f - lo) / (hi - lo));
  const x = (k) => px(m.grid.t0_ms + k * m.grid.dt_ms);
  const rejected = new Set(m.rejected);
  for (const region of m.regions) {
    g.fillStyle = rejected.has(region.id) ? T.rejected : T.mask;
    for (const [j, k, n] of region.runs) {
      const [top, bottom] = [y(m.y0_khz + (j + 0.5) * m.dy_khz), y(m.y0_khz + (j - 0.5) * m.dy_khz)];
      if (bottom < 0 || top > h) continue;
      const left = x(k);
      g.fillRect(left, top, Math.max(1, x(k + n) - left), Math.max(1, bottom - top));
    }
  }
}

/** A row's image at the current contrast: a pixel per column and bin, top bin first. */
function bitmap(row, data, i) {
  let canvas = data.bitmaps.get(i);
  if (canvas) return canvas;
  const [n, values] = [data.n, data.rows[i]];
  canvas = document.createElement("canvas");
  [canvas.width, canvas.height] = [n, row.n_y];
  const g = canvas.getContext("2d");
  const image = g.createImageData(n, row.n_y);
  const pixels = new Uint32Array(image.data.buffer);
  for (let y = 0; y < row.n_y; y++) {
    const from = (row.n_y - 1 - y) * n; // the store's row 0 is the lowest bin
    for (let x = 0; x < n; x++) pixels[y * n + x] = S.lut[values[from + x]];
  }
  g.putImageData(image, 0, 0);
  data.bitmaps.set(i, canvas);
  return canvas;
}

/** A trace row's y range over the columns on screen, its dashed lines included. */
function traceRange(row, values) {
  const n = S.data.n;
  const step = (S.data.t1 - S.data.t0) / n;
  const j0 = clamp(Math.floor((S.view[0] - S.data.t0) / step), 0, n);
  const j1 = clamp(Math.ceil((S.view[1] - S.data.t0) / step), 0, n);
  let [lo, hi] = [Math.min(Infinity, ...row.hlines), Math.max(-Infinity, ...row.hlines)];
  for (let p = 0; p < 2 * row.n_channels; p++) {
    for (let j = j0; j < j1; j++) {
      const v = values[p * n + j];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!(hi > lo)) [lo, hi] = Number.isFinite(lo) ? [lo - 1, lo + 1] : [0, 1];
  const pad = (hi - lo) * 0.06;
  return [lo - pad, hi + pad];
}

function drawTrace(g, row, values, [lo, hi], w, h) {
  const [n, channels] = [S.data.n, row.n_channels];
  const step = (S.data.t1 - S.data.t0) / n;
  const y = (v) => ((hi - v) / (hi - lo)) * (h - 1);
  g.lineWidth = 1;
  g.strokeStyle = T.muted;
  g.setLineDash([4, 3]);
  for (const v of row.hlines) {
    g.beginPath();
    g.moveTo(0, y(v));
    g.lineTo(w, y(v));
    g.stroke();
  }
  g.setLineDash([]);
  for (let c = 0; c < channels; c++) {
    // Each column's minimum then its maximum: an envelope zoomed out, a line zoomed in.
    g.strokeStyle = TRACE_COLOURS[c % TRACE_COLOURS.length];
    g.beginPath();
    let pen = false;
    for (let j = 0; j < n; j++) {
      const [low, high] = [values[c * n + j], values[(channels + c) * n + j]];
      if (!Number.isFinite(low)) {
        pen = false;
        continue;
      }
      const x = px(S.data.t0 + (j + 0.5) * step);
      if (pen) g.lineTo(x, y(low));
      else g.moveTo(x, y(low));
      g.lineTo(x, y(high));
      pen = true;
    }
    g.stroke();
  }
  if (row.legend.length < 2) return;
  g.font = FONT;
  g.textAlign = "right";
  let x = w - RIGHT - 6;
  for (let c = row.legend.length - 1; c >= 0; c--) {
    g.fillStyle = TRACE_COLOURS[c % TRACE_COLOURS.length];
    g.fillText(row.legend[c], x, 14);
    x -= g.measureText(row.legend[c]).width + 10;
  }
}

/** The label on a row: outside its window veiled, each span's edges and a bar on top. */
function drawOverlay(g, w, h) {
  const [lo, hi] = S.label.window;
  g.fillStyle = T.veil;
  g.fillRect(0, 0, px(lo), h);
  g.fillRect(px(hi), 0, w - px(hi), h);
  S.label.intervals.forEach(([a, b, c], i) => {
    const width = i === S.selected ? 2 : 1;
    g.fillStyle = categoryColour(c);
    g.fillRect(px(a), 0, px(b) - px(a), 3);
    g.fillRect(px(a) - width / 2, 0, width, h);
    g.fillRect(px(b) - width / 2, 0, width, h);
  });
}

function ticks(lo, hi, n) {
  const step = niceStep(hi - lo, n);
  if (!(step > 0 && Number.isFinite(step))) return [];
  const digits = Math.max(0, -Math.floor(Math.log10(step)));
  const text = (t) => (step >= 1e4 || step < 1e-3 ? t.toExponential(1) : t.toFixed(digits));
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) out.push([t, text(t)]);
  return out;
}

/** `text` in lines no wider than `width` in `g`'s font, broken between words; a
 * word wider than a line has a line of its own. */
function wrapped(g, text, width) {
  const lines = [];
  for (const word of String(text ?? "").split(/\s+/).filter(Boolean)) {
    const longer = lines.length ? `${lines[lines.length - 1]} ${word}` : word;
    if (lines.length && g.measureText(longer).width <= width) lines[lines.length - 1] = longer;
    else lines.push(word);
  }
  return lines;
}

/** The first `most` of `lines`, the last one kept ending in an ellipsis when
 * any are left out; measured in the context's current font. */
function capped(g, lines, width, most) {
  if (lines.length <= most) return lines;
  const kept = lines.slice(0, Math.max(most, 0));
  if (!kept.length) return kept;
  let last = `${kept[kept.length - 1]}…`;
  while (last.length > 1 && g.measureText(last).width > width) last = `${last.slice(0, -2)}…`;
  kept[kept.length - 1] = last;
  return kept;
}

/** A row's title, wrapped to the gutter, then its units, from the top left,
 * cut with an ellipsis only where it would run into the row's last 20 px; then
 * the y ticks, each one that would touch that text left off. */
function drawGutter(g, row, range, h) {
  const width = GUTTER - 12;
  const most = Math.max(1, Math.floor((h - 20) / GUTTER_LINE));
  const text = [];  // [right, top, bottom] of each line drawn
  let y = 16;
  const line = (words) => {
    g.fillText(words, 8, y, width);  // only a word wider than the gutter is squeezed
    text.push([8 + Math.min(width, g.measureText(words).width), y - 10, y + 3]);
    y += GUTTER_LINE;
  };
  g.textAlign = "left";
  g.fillStyle = T.ink;
  g.font = `600 ${FONT}`;
  const title = wrapped(g, row.title, width);
  capped(g, title, width, most).forEach(line);
  g.font = FONT;
  g.fillStyle = T.muted;
  capped(g, wrapped(g, row.y_units, width), width, most - title.length).forEach(line);
  if (!range) return;
  const [lo, hi] = range;
  g.textAlign = "right";
  for (const [t, label] of ticks(lo, hi, h / 40)) {
    const y = ((hi - t) / (hi - lo)) * (h - 1);
    const left = GUTTER - 6 - g.measureText(label).width;
    const touches = text.some(([right, top, bottom]) => right + 2 > left && top < y + 7 && bottom > y - 7);
    if (y > 6 && y < h - 4 && !touches) g.fillText(label, GUTTER - 6, y + 4);
  }
}

function drawAxis() {
  const canvas = $("axis-row");
  const g = context(canvas);
  g.font = FONT;
  g.fillStyle = T.muted;
  g.fillText("ms", 8, 16);
  g.textAlign = "center";
  for (const [t, text] of ticks(S.view[0], S.view[1], plotWidth() / 90)) {
    const x = px(t);
    if (x < GUTTER || x > canvas.clientWidth - RIGHT) continue;
    g.fillRect(x, 0, 1, 4);
    g.fillText(text, x, 16);
  }
}

/** A label as a track: its window lit, its spans filled; the editable one also shows
 * the window's edges in its foot strip and, along its top, where it leaves the source. */
function drawTrack(canvas, label, editable) {
  const g = context(canvas);
  const [w, h] = [canvas.clientWidth, canvas.clientHeight];
  if (!label) return;
  g.save();
  g.beginPath();
  g.rect(GUTTER, 0, w - GUTTER - RIGHT, h);
  g.clip();
  const [lo, hi] = label.window;
  const [top, foot] = editable ? [5, HANDLE_BAND + 2] : [4, 4];
  g.fillStyle = T.panel;
  g.fillRect(px(lo), 0, px(hi) - px(lo), h);
  label.intervals.forEach(([a, b, c], i) => {
    g.globalAlpha = editable ? 1 : 0.55;
    g.fillStyle = categoryColour(c);
    g.fillRect(px(a), top, Math.max(1, px(b) - px(a)), h - top - foot);
    g.globalAlpha = 1;
    if (editable && i === S.selected) {
      g.strokeStyle = T.ink;
      g.lineWidth = 2;
      g.strokeRect(px(a), top, px(b) - px(a), h - top - foot);
    }
  });
  if (editable) {
    g.fillStyle = T.ink;
    g.fillRect(px(lo), h - HANDLE_BAND / 2 - 0.5, px(hi) - px(lo), 1);
    for (const t of label.window) g.fillRect(px(t) - 2, h - HANDLE_BAND, 4, HANDLE_BAND);
    const source = S.meta.source || { window: label.window, intervals: [] };
    g.fillStyle = T.changed;
    for (const [a, b] of diffRuns(source, label)) g.fillRect(px(a), 0, Math.max(1, px(b) - px(a)), 3);
  }
  g.restore();
}

function showHeader() {
  const next = neighbour(1);
  $("next-shot").textContent = next == null || next === S.shot ? "" : `→ ${next}`;
  const row = S.queue.find((r) => r.shot === S.shot) || {};
  const last = S.meta?.last_save;
  $("tier").textContent = row.tier || "";
  $("state").textContent = S.queueEvent !== S.event ? "the queue is still loading" : row.state || "";
  $("state").className = `pill ${row.state || ""}`;
  $("saved").textContent = last ? `saved ${when(last.saved_at)}${last.name ? ` by ${last.name}` : ""}` : "";
  $("dirty").hidden = !dirty();
  const reviewed = S.queue.filter((row) => row.state !== "unreviewed").length;
  $("count").textContent = `${reviewed}/${S.queue.length}`;
}

function renderQueue() {
  const nav = $("queue");
  if (nav.children.length !== S.queue.length) {
    nav.replaceChildren(
      ...S.queue.map((row) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.dataset.shot = row.shot;
        chip.textContent = row.shot;
        return chip;
      })
    );
  }
  S.queue.forEach((row, i) => {
    const chip = nav.children[i];
    const marks = [row.state, row.shot === S.shot && "current", stored(draftKey(row.shot)) && "dirty"];
    chip.className = ["chip", ...marks.filter(Boolean)].join(" ");
    chip.title = row.tier;
  });
}

function renderSwatches() {
  $("swatches").replaceChildren(
    ...Object.entries(S.categories).map(([c, name]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "swatch";
      button.textContent = `${c} ${name}`;
      button.style.setProperty("--c", categoryColour(Number(c)));
      button.setAttribute("aria-pressed", String(Number(c) === S.category));
      button.addEventListener("click", () => setCategory(Number(c)));
      return button;
    })
  );
}

// -- input

function startDrag(event, drag) {
  if (stillOpening()) return;
  S.drag = { ...drag, base: S.label };
  event.currentTarget.setPointerCapture(event.pointerId);
  event.preventDefault();
}

function onRowsDown(event) {
  if (event.button !== 0 || !S.meta || event.target.tagName !== "CANVAS") return;
  if (event.shiftKey) startDrag(event, { kind: "new", from: timeAt(event.clientX) });
  else startDrag(event, { kind: "pan", x: event.clientX, y: event.clientY, view: S.view, canvas: event.target });
}

function onLabelDown(event) {
  if (event.button !== 0 || !S.meta) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const hit = hitTest(S.label, event.clientX - rect.left, event.clientY - rect.top, rect.height, px);
  S.selected = hit.kind === "edge" || hit.kind === "move" ? hit.index : -1;
  startDrag(event, { ...hit, from: timeAt(event.clientX) });
  render();
}

function onLabelHover(event) {
  if (S.drag || !S.meta) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const hit = hitTest(S.label, event.clientX - rect.left, event.clientY - rect.top, rect.height, px);
  const cursors = { edge: "ew-resize", window: "ew-resize", move: "grab", new: "crosshair" };
  event.currentTarget.style.cursor = cursors[hit.kind];
}

function dragTo(clientX) {
  const d = S.drag;
  const t = timeAt(clientX);
  if (d.kind === "pan") {
    d.moved ||= Math.abs(clientX - d.x) > CLICK_PX;
    const shift = ((clientX - d.x) / plotWidth()) * (d.view[1] - d.view[0]);
    return setView(d.view[0] - shift, d.view[1] - shift);
  }
  if (stillOpening()) return;
  const edges = [...d.base.window];
  const spans = d.base.intervals.map((span) => [...span]);
  let moved = null;
  if (d.kind === "new") moved = [Math.min(d.from, t), Math.max(d.from, t), S.category];
  if (d.kind === "edge") {
    const span = spans.splice(d.index, 1)[0];
    span[d.edge] = t;
    moved = [Math.min(span[0], span[1]), Math.max(span[0], span[1]), span[2]];
  }
  if (d.kind === "move") {
    const [a, b, c] = spans.splice(d.index, 1)[0];
    moved = [a + t - d.from, b + t - d.from, c];
  }
  if (d.kind === "window") {
    const [lo, hi] = edges;
    edges[d.edge] = d.edge
      ? clamp(t, lo + 1, lo + LONGEST_WINDOW_MS)
      : clamp(t, hi - LONGEST_WINDOW_MS, hi - 1);
    // The window is the edge of what was looked at: spans are cut back to it.
    for (const span of spans) [span[0], span[1]] = [0, 1].map((k) => clamp(span[k], ...edges));
  }
  if (moved) spans.push(moved); // last, so it paints over what it crosses
  const next = normalise(edges, spans, known());
  if (!next) return;
  S.label = next;
  if (moved) selectAt((moved[0] + moved[1]) / 2);
  render();
}

function endDrag(event) {
  const d = S.drag;
  S.drag = null;
  if (d?.kind === "pan" && d.canvas && !d.moved && event?.type === "pointerup") return clickMask(d);
  if (!d || d.kind === "pan" || same(d.base, S.label)) return;
  keepForUndo(d.base);
  touch();
}

// -- the AE pseudo-mask: TokEye's lines inside the label, one click per region

const maskSaves = new Map(); // shot -> pending save, survives navigation
const maskErrors = new Map(); // shot -> last failed save, until a successful retry

/** The open shot's pseudo-mask, if the server has one (api 3, AE only). */
async function loadMasks(ticket) {
  if (S.api >= 3 && S.event === MASK_EVENT) {
    const event = S.event, shot = S.shot;
    try {
      await maskSaves.get(shot);
      if (ticket !== S.ticket) return;
      const body = await (await api(`/api/masks?event=${enc(event)}&shot=${shot}`)).json();
      if (ticket !== S.ticket) return;
      S.masks = body;
      const failure = maskErrors.get(shot);
      if (failure) say(failure, true);
    } catch {
      // no pseudo-mask for this shot: nothing to draw
    }
  }
  if (ticket !== S.ticket) return;
  showMasks();
  render();
}

function showMasks() {
  const m = S.masks;
  $("masks").hidden = !m;
  if (!m) return;
  const n = m.regions.length;
  const kept = n - m.rejected.length;
  const last = !m.stale && m.last_save;
  const by = last ? ` · saved${last.name ? ` by ${last.name}` : ""}` : "";
  const notes = [m.stale ? " · a decision on an older mask was dropped" : "", by, S.showMasks ? "" : " · hidden"];
  $("masks").textContent = `mask ${kept}/${n} kept${notes.join("")}`;
}

function toggleMasks() {
  if (!S.masks) return say("this shot has no pseudo-mask");
  S.showMasks = !S.showMasks;
  store("labeler:masks", S.showMasks ? null : "hidden");
  showMasks();
  render();
}

/** A click on an image row, not a drag: reject the region under it, or take that back. */
function clickMask(d) {
  if (maskSaves.has(S.shot)) return say("mask is saving; wait for it to finish");
  const m = S.masks;
  if (!m || !S.showMasks || m.saving) return;
  const row = S.meta.rows[[...$("rows").children].indexOf(d.canvas)];
  if (!row || row.kind !== "image") return;
  const rect = d.canvas.getBoundingClientRect();
  const [lo, hi] = imageRange(row);
  const f = hi - ((d.y - rect.top) / (rect.height - 1)) * (hi - lo);
  const j = Math.round((f - m.y0_khz) / m.dy_khz);
  const k = Math.floor((timeAt(d.x) - m.grid.t0_ms) / m.grid.dt_ms);
  const binPx = ((rect.height - 1) * m.dy_khz) / (hi - lo);
  const columnPx = (plotWidth() * m.grid.dt_ms) / (S.view[1] - S.view[0]);
  const reach = (size) => Math.ceil(CLICK_PX / Math.max(size, 1e-6));
  const region = regionAt(m.regions, j, k, reach(binPx), reach(columnPx));
  if (!region) return;
  const rejected = new Set(m.rejected);
  if (!rejected.delete(region.id)) rejected.add(region.id);
  saveMasks(m, [...rejected].sort((a, b) => a - b));
}

/** Save the rejected regions at once, with the typed name; drawn before the answer. */
function saveMasks(m, rejected) {
  if (maskSaves.has(m.shot)) return say("mask is saving; wait for it to finish");
  const pending = persistMasks(m, rejected, S.event);
  maskSaves.set(m.shot, pending);
  return pending;
}

async function persistMasks(m, rejected, event) {
  const before = m.rejected;
  const current = () => S.event === event && S.masks?.shot === m.shot &&
    S.masks.pseudo_sha256 === m.pseudo_sha256;
  let conflict = false;
  Object.assign(m, { rejected, saving: true });
  showMasks();
  render();
  try {
    const response = await api("/api/masks", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        event,
        shot: m.shot,
        pseudo_sha256: m.pseudo_sha256,
        revision: m.revision,
        rejected,
        name: S.name || null,
      }),
    });
    const body = await response.json();
    const saved = { rejected: body.rejected, last_save: body.last_save,
      revision: body.revision, stale: false };
    Object.assign(m, saved);
    maskErrors.delete(m.shot);
    if (current()) Object.assign(S.masks, saved);
    const n = body.rejected.length;
    if (current()) say(`mask saved: ${n} region${n === 1 ? "" : "s"} rejected`);
  } catch (error) {
    m.rejected = before;
    conflict = error.status === 409;
    maskErrors.set(m.shot, conflict
      ? "mask decisions changed; reloaded the current masks — click again"
      : error.message);
    if (current()) {
      S.masks.rejected = before;
      say(error.message, true);
    }
  } finally {
    m.saving = false;
    if (current()) S.masks.saving = false;
    maskSaves.delete(m.shot);
  }
  if (conflict && S.event === event && S.shot === m.shot) {
    await loadMasks(S.ticket);
    if (S.event === event && S.shot === m.shot) {
      say("mask decisions changed; reloaded the current masks — click again", true);
    }
  }
  if (!current()) return;
  showMasks();
  render();
}

function onWheel(event, zoomAlways) {
  if (!S.meta) return;
  const scale = event.deltaMode === 1 ? 16 : 1; // lines, not px
  const [dx, dy] = [event.deltaX * scale, event.deltaY * scale];
  if (Math.abs(dx) > Math.abs(dy)) {
    event.preventDefault();
    return pan(dx / plotWidth());
  }
  if (!zoomAlways && !event.ctrlKey && !event.metaKey) return; // a plain wheel scrolls
  event.preventDefault();
  zoomAt(timeAt(event.clientX), Math.exp(dy * 0.002));
}

function showCursor(clientX) {
  const cursor = $("cursor");
  const axis = $("axis-row").getBoundingClientRect();
  const x = clientX - axis.left;
  cursor.hidden = !S.meta || x < GUTTER || x > axis.width - RIGHT;
  if (cursor.hidden) return;
  const top = $("top").getBoundingClientRect().top;
  const bottom = $("label-track").getBoundingClientRect().bottom;
  Object.assign(cursor.style, { left: `${clientX}px`, top: `${top}px`, height: `${bottom - top}px` });
  $("cursor-time").textContent = `${Math.round(timeAt(clientX))} ms`;
}

function toggleKeys() {
  const dialog = $("keys");
  if (dialog.open) closeDialog(dialog);
  else dialog.showModal();
}

/** Close a dialog and give the keys back to the page, not to a button left inside it. */
function closeDialog(dialog) {
  if (dialog.contains(document.activeElement)) document.activeElement.blur();
  dialog.close();
}

/** The shot's saved versions, newest first; H again, Escape or Close shuts them. */
async function toggleVersions() {
  if (stillOpening()) return;
  const dialog = $("versions");
  if (dialog.open) return closeDialog(dialog);
  if (!S.meta || S.api < 2) return;
  const { ticket, event, shot } = S.opened;
  try {
    const body = await (await api(`/api/history?event=${enc(event)}&shot=${shot}`)).json();
    if (ticket !== S.ticket) return;
    S.versions = body.versions;
    S.versionsAt = S.opened;
    renderVersions();
    dialog.showModal();
  } catch (error) {
    say(error.message, true);
  }
}

function renderVersions() {
  const changes = versionChanges(S.versions, S.meta.source);
  const items = S.versions.map((version, i) => {
    const item = document.createElement("li");
    const who = version.name ? `${version.name} (${version.reviewer})` : version.reviewer || "unknown";
    const n = version.intervals.length;
    const moved = changes[i] == null ? "first label" : `${changes[i]} ms changed`;
    const text = document.createElement("span");
    text.textContent = `v${version.version} · ${when(version.saved_at)} · ${who} · ${n} span${n === 1 ? "" : "s"} · ${moved}`;
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.version = version.version;
    button.textContent = "Restore";
    item.append(text, button);
    return item;
  });
  const none = Object.assign(document.createElement("li"), { textContent: "Not saved yet." });
  $("version-list").replaceChildren(...(items.length ? items.reverse() : [none]));
}

/** Load a saved version as the draft: saving it appends a new version, so none is lost. */
function restoreVersion(number) {
  if (stillOpening()) return;
  const at = S.versionsAt;
  if (!at || at !== S.opened || at.event !== S.event || at.shot !== S.shot) {
    return say("this history belongs to a shot that is no longer open");
  }
  const found = S.versions.find((version) => version.version === number);
  if (!found || !S.meta) return;
  closeDialog($("versions"));
  const label = normalise(found.window, found.intervals, known());
  if (!label) return say(`version ${number} does not fit this event's categories`, true);
  if (same(label, S.label)) return say(`version ${number} is already the current label`);
  const replaced = dirty() ? "; replaced an unsaved edit; Ctrl+Z brings it back" : "";
  S.selected = -1;
  edit(label.window, label.intervals);
  say(`version ${number} restored as a draft: Enter or S saves it as a new version${replaced}`);
}

const KEYS = {
  Enter: () => save(true),
  s: () => save(false),
  r: revert,
  h: toggleVersions,
  m: toggleMasks,
  j: () => go(-1),
  k: () => go(1),
  ArrowLeft: () => go(-1),
  ArrowRight: () => go(1),
  u: () => {
    if (stillLoadingQueue()) return;
    if (nextUnreviewed() == null) say(S.queue.length ? "all reviewed" : "no shots to review");
    else openShot(nextUnreviewed());
  },
  "[": () => contrast(-16),
  "]": () => contrast(16),
  Escape: () => {
    S.selected = -1;
    render();
  },
  Delete: removeSelected,
  Backspace: removeSelected,
  "Shift+ArrowLeft": () => pan(-0.1),
  "Shift+ArrowRight": () => pan(0.1),
  "-": () => zoomAt(middle(), 1.25),
  "=": () => zoomAt(middle(), 0.8),
  "+": () => zoomAt(middle(), 0.8),
  0: fit,
};

// The keys that work on a shot with nothing to show.
const MOVES = new Set(["j", "k", "u", "ArrowLeft", "ArrowRight"]);

function onKey(event) {
  const target = event.target;
  const typed = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  const key = event.shiftKey && typed.startsWith("Arrow") ? `Shift+${typed}` : typed;
  if (target.closest("input, select, textarea")) return;
  if ($("versions").open) {
    if (key === "Enter" || key === "h") {
      event.preventDefault();
      closeDialog($("versions"));
    }
    return;
  }
  if (target.closest("button") && (key === "Enter" || key === " ")) return;
  if (key === "?" || $("keys").open) return key === "?" && toggleKeys();
  const mod = event.ctrlKey || event.metaKey;
  if (event.altKey || (mod && key !== "z") || (!S.meta && !MOVES.has(key))) return;
  if (mod) {
    event.preventDefault();
    return undo();
  }
  if (/^[1-9]$/.test(key)) return setCategory(Number(key));
  if (KEYS[key]) {
    event.preventDefault();
    KEYS[key]();
  }
}

function wire() {
  const top = $("top");
  const tracks = [$("source-track"), $("label-track")];
  top.addEventListener("pointerdown", onRowsDown);
  top.addEventListener("wheel", (event) => onWheel(event, event.target.id === "axis-row"), {
    passive: false,
  });
  for (const track of tracks) {
    track.addEventListener("wheel", (event) => onWheel(event, true), { passive: false });
  }
  for (const target of [top, ...tracks]) target.addEventListener("dblclick", () => S.meta && fit());
  tracks[0].addEventListener("pointerdown", (event) => {
    if (event.button === 0 && S.meta) startDrag(event, { kind: "pan", x: event.clientX, view: S.view });
  });
  tracks[1].addEventListener("pointerdown", onLabelDown);
  tracks[1].addEventListener("pointermove", onLabelHover);
  window.addEventListener("pointermove", (event) => {
    if (S.drag) dragTo(event.clientX);
    if (S.drag || event.target.closest?.("#top, .track")) showCursor(event.clientX);
    else $("cursor").hidden = true;
  });
  window.addEventListener("hashchange", followHash);
  document.documentElement.addEventListener("pointerleave", () => ($("cursor").hidden = true));
  window.addEventListener("pointerup", endDrag);
  window.addEventListener("pointercancel", endDrag);
  document.addEventListener("keydown", onKey);
  $("event").addEventListener("change", () =>
    openEvent($("event").value, null).catch((error) => say(error.message, true))
  );
  $("shot").addEventListener("keydown", (event) => {
    const typed = $("shot").value.trim();
    if (event.key === "Enter" && /^\d+$/.test(typed)) openShot(Number(typed));
  });
  $("queue").addEventListener("click", (event) => {
    const chip = event.target.closest(".chip");
    if (chip) openShot(Number(chip.dataset.shot));
  });
  $("save-next").addEventListener("click", () => save(true));
  $("revert").addEventListener("click", () => S.meta && revert());
  $("show-versions").addEventListener("click", toggleVersions);
  $("help").addEventListener("click", toggleKeys);
  for (const dialog of document.querySelectorAll("dialog")) {
    // Chrome can leave focus on a button in a closed dialog, and Enter would then click it
    // instead of saving: let go of it as Escape cancels the dialog, and again once closed.
    const release = () => dialog.contains(document.activeElement) && document.activeElement.blur();
    dialog.addEventListener("cancel", release);
    dialog.addEventListener("close", release);
  }
  // A button clicked with the mouse gives the keys back, so Enter saves instead of clicking it again.
  document.addEventListener("click", (event) => {
    const button = event.target.closest?.("button");
    if (button && event.detail > 0) button.blur();
  });
  $("version-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-version]");
    if (button) restoreVersion(Number(button.dataset.version));
  });
  $("reviewer-name").addEventListener("input", () => {
    S.name = $("reviewer-name").value.trim();
    store("labeler:name", S.name || null);
  });
  $("reviewer-name").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === "Escape") $("reviewer-name").blur();
  });
  new ResizeObserver(() => {
    sizeCanvases();
    render();
    fetchRows();
  }).observe(top);
}

if (typeof module !== "undefined") {
  module.exports = { ms, paint, runs, normalise, diffRuns, versionChanges, niceStep, hitTest, regionAt, lut };
} else {
  boot();
}
