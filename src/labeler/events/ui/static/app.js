// The label review page: spectrogram rows on top, one editable label per shot below.
"use strict";

// ---- The label rules, as the server applies them; exported for the tests ----

const LONGEST_WINDOW_MS = 20000;
const HANDLE_BAND = 10; // px at the foot of the label track that hold the window's edges
const GRAB = 5; // px either side of an edge that grab it
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

const S = {
  events: [],
  event: null,
  categories: {},
  category: 1,
  queue: [],
  shot: null,
  meta: null, // what /api/shot said: grid, t_range, rows, source, saved, state, last_save
  label: null, // the label being edited, always normalised
  selected: -1,
  view: [0, 1], // the ms on screen
  data: null, // the last /api/rows: {t0, t1, n, rows}
  asked: "", // the last /api/rows query sent
  bitmaps: new Map(),
  lo: 0,
  lut: lut(0),
  undo: [],
  drag: null,
  ticket: 0, // bumped per shot opened, so an answer for an older shot is dropped
  frame: 0,
  timer: 0,
  saving: false,
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
    throw new Error(body.error || `${response.status} ${response.statusText}`);
  }
  return response;
}

function say(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function readTokens() {
  const style = getComputedStyle(document.documentElement);
  const names = ["panel", "ink", "muted", "rule", "label", "changed", "veil"];
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
  const next = normalise(window, intervals, known());
  if (!next) return;
  keepForUndo(S.label);
  S.label = next;
  touch();
}

/** Keep the draft until it is saved or reverted, and mark the shot unsaved. */
function touch() {
  store(draftKey(S.shot), dirty() ? JSON.stringify(S.label) : null);
  $("dirty").hidden = !dirty();
  renderQueue();
  render();
}

/** Select the span holding `t`, if any. */
function selectAt(t) {
  S.selected = S.label.intervals.findIndex(([a, b]) => a <= t && t <= b);
}

function undo() {
  if (!S.undo.length) return;
  S.label = S.undo.pop();
  S.selected = -1;
  touch();
}

function revert() {
  const source = S.meta.source || emptyLabel();
  S.selected = -1;
  edit(source.window, source.intervals);
}

function removeSelected() {
  if (S.selected < 0) return;
  const kept = S.label.intervals.filter((_, i) => i !== S.selected);
  S.selected = -1;
  edit(S.label.window, kept);
}

function setCategory(c) {
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
  S.bitmaps.clear();
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
  wire();
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

async function openEvent(event, shot) {
  S.event = event;
  S.categories = S.events.find((row) => row.event === event).categories;
  S.category = known()[0] || 1;
  $("event").value = event;
  store("labeler:event", event);
  renderSwatches();
  $("queue").replaceChildren();
  const queue = await (await api(`/api/queue?event=${enc(event)}`)).json();
  S.queue = queue.shots;
  renderQueue();
  await openShot(S.queue.some((row) => row.shot === shot) ? shot : queue.resume);
}

async function openShot(shot) {
  if (shot == null) return;
  const ticket = ++S.ticket;
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
    Object.assign(S, { shot, meta, data: null, asked: "", undo: [], selected: -1 });
    S.bitmaps.clear();
    S.label = draft(shot) || baseline();
    buildRows();
    arrive();
    fit();
    fetchRows(0);
    prefetch();
  } catch (error) {
    if (ticket !== S.ticket) return;
    // Stay on the shot, with nothing to edit, so J, K and U carry on from it.
    Object.assign(S, { shot, meta: null, data: null, label: null, undo: [], selected: -1 });
    const note = document.createElement("p");
    note.className = "failure";
    note.textContent = `${shot} has nothing to show: ${error.message}. K opens the next shot.`;
    $("rows").replaceChildren(note);
    for (const canvas of document.querySelectorAll("#top canvas, .track canvas")) context(canvas);
    arrive();
  }
}

/** The header, the address and the queue follow the shot just opened. */
function arrive() {
  busy(null);
  $("cursor").hidden ||= !S.meta;
  for (const id of ["save-next", "revert"]) $(id).disabled = !S.meta;
  history.replaceState(null, "", `#${S.event}/${S.shot}`);
  $("shot").value = S.shot;
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
  const next = nextUnreviewed() ?? neighbour(1);
  if (next != null && next !== S.shot) api(`/api/shot?event=${enc(S.event)}&shot=${next}`).catch(() => {});
}

function nextUnreviewed() {
  const i = S.queue.findIndex((row) => row.shot === S.shot);
  const after = [...S.queue.slice(i + 1), ...S.queue.slice(0, Math.max(i, 0))];
  return after.find((row) => row.state === "unreviewed")?.shot ?? null;
}

function neighbour(delta) {
  const i = S.queue.findIndex((row) => row.shot === S.shot);
  return S.queue[(i + delta + S.queue.length) % S.queue.length]?.shot ?? null;
}

function fetchRows(delay = 90) {
  clearTimeout(S.timer);
  S.timer = setTimeout(loadRows, delay);
}

async function loadRows() {
  if (!S.meta) return;
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
    S.data = { ...grid, rows: unpack(buffer, grid.n) };
    S.bitmaps.clear();
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
  if (!S.meta || S.saving) return;
  S.saving = true;
  const [shot, key] = [S.shot, draftKey(S.shot)];
  try {
    const response = await api("/api/label", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ event: S.event, shot, ...S.label }),
    });
    const body = await response.json();
    store(key, null);
    S.queue = S.queue.map((row) => (row.shot === shot ? body.row : row));
    if (S.shot === shot) {
      Object.assign(S.meta, { saved: body.saved, last_save: body.last_save });
      S.label = body.saved;
      showHeader();
      touch();
    }
    if (next) await openShot(nextUnreviewed() ?? neighbour(1));
  } catch (error) {
    say(error.message, true);
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
    if (values && row.kind === "image") drawImage(g, row, values, i, h);
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

function drawImage(g, row, values, i, h) {
  const [first, stop] = bandBins(row);
  const [x0, x1] = [px(S.data.t0), px(S.data.t1)];
  g.imageSmoothingEnabled = false;
  g.drawImage(bitmap(row, values, i), 0, row.n_y - stop, S.data.n, stop - first, x0, 0, x1 - x0, h - 1);
}

/** A row's image at the current contrast: a pixel per column and bin, top bin first. */
function bitmap(row, values, i) {
  let canvas = S.bitmaps.get(i);
  if (canvas) return canvas;
  const n = S.data.n;
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
  S.bitmaps.set(i, canvas);
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

function drawGutter(g, row, range, h) {
  g.textAlign = "left";
  g.fillStyle = T.ink;
  g.font = `600 ${FONT}`;
  g.fillText(row.title, 8, 16, GUTTER - 44);
  g.font = FONT;
  g.fillStyle = T.muted;
  g.fillText(row.y_units, 8, 30, GUTTER - 44);
  if (!range) return;
  const [lo, hi] = range;
  g.textAlign = "right";
  for (const [t, text] of ticks(lo, hi, h / 40)) {
    const y = ((hi - t) / (hi - lo)) * (h - 1);
    if (y > 6 && y < h - 4) g.fillText(text, GUTTER - 6, y + 4);
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
  const row = S.queue.find((r) => r.shot === S.shot) || {};
  const last = S.meta?.last_save;
  $("tier").textContent = row.tier || "";
  $("state").textContent = row.state || "";
  $("state").className = `pill ${row.state || ""}`;
  $("saved").textContent = last
    ? `saved ${new Date(last.saved_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`
    : "";
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
  S.drag = { ...drag, base: S.label };
  event.currentTarget.setPointerCapture(event.pointerId);
  event.preventDefault();
}

function onRowsDown(event) {
  if (event.button !== 0 || !S.meta || event.target.tagName !== "CANVAS") return;
  if (event.shiftKey) startDrag(event, { kind: "new", from: timeAt(event.clientX) });
  else startDrag(event, { kind: "pan", x: event.clientX, view: S.view });
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
    const shift = ((clientX - d.x) / plotWidth()) * (d.view[1] - d.view[0]);
    return setView(d.view[0] - shift, d.view[1] - shift);
  }
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

function endDrag() {
  const d = S.drag;
  S.drag = null;
  if (!d || d.kind === "pan" || same(d.base, S.label)) return;
  keepForUndo(d.base);
  touch();
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
  if (dialog.open) dialog.close();
  else dialog.showModal();
}

const KEYS = {
  Enter: () => save(true),
  s: () => save(false),
  r: revert,
  j: () => openShot(neighbour(-1)),
  k: () => openShot(neighbour(1)),
  u: () => (nextUnreviewed() == null ? say("all reviewed") : openShot(nextUnreviewed())),
  "[": () => contrast(-16),
  "]": () => contrast(16),
  Escape: () => {
    S.selected = -1;
    render();
  },
  Delete: removeSelected,
  Backspace: removeSelected,
  ArrowLeft: () => pan(-0.1),
  ArrowRight: () => pan(0.1),
  "-": () => zoomAt(middle(), 1.25),
  "=": () => zoomAt(middle(), 0.8),
  "+": () => zoomAt(middle(), 0.8),
  0: fit,
};

const MOVES = new Set(["j", "k", "u"]); // the keys that work on a shot with nothing to show

function onKey(event) {
  const target = event.target;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  if (target.closest("input, select, textarea")) return;
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
  $("help").addEventListener("click", toggleKeys);
  new ResizeObserver(() => {
    sizeCanvases();
    render();
    fetchRows();
  }).observe(top);
}

if (typeof module !== "undefined") {
  module.exports = { ms, paint, runs, normalise, diffRuns, niceStep, hitTest, lut };
} else {
  boot();
}
