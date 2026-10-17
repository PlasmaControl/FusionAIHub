// The label review page: spectrogram rows on top, one editable label per shot below.
"use strict";

// ---- The label rules, as the server applies them; exported for the tests ----

const LONGEST_WINDOW_MS = 20000;
const GRID_MS = 10;
const HANDLE_BAND = 10; // px at the foot of the label track that hold the window's edges
const GRAB = 5; // px either side of an edge that grab it
const CLICK_PX = 4; // a press that moves less than this is a click; a click this close picks a region
const MASK_EVENT = "alfven_eigenmode"; // the one event with pseudo-masks
/** The silver list's pools, as their tags (`silver.POOLS`). */
const SILVER_POOLS = { ae_audited: "AE audited", ae_ece_box: "AE ECE box", rwm_onsets: "RWM onset" };
const INFERNO = [
  "000004", "0b0724", "210c4a", "3d0965", "57106e", "71196e", "8a226a", "a32c61", "bc3754",
  "d24644", "e45a31", "f1731d", "f98e09", "fcac11", "f9cb35", "f2ea69", "fcffa4",
];

/** Nearest 10 ms edge, halves up, as `grid.snap`. */
function ms(t) {
  return Math.floor(Number(t) / GRID_MS + 0.5) * GRID_MS;
}

function snapSpan(a, b) {
  if (![a, b].every(Number.isFinite) || b < a) return null;
  const [lo, hi] = [ms(a), ms(b)];
  if (hi > lo || a === b) return [lo, hi];
  const start = Math.floor((a + b) / 2 / GRID_MS) * GRID_MS;
  return [start, start + GRID_MS];
}

/** A window's ends, snapped as a span's are (`grid.snap_window`). */
function snapWindow(lo, hi) {
  return snapSpan(lo, hi);
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

/** The server's `normalise`: snap and clip, with later annotations painting only
 * their own lane. Individual/unspecified annotations overlap spans. */
function normalise(window, intervals, known, iscrowd) {
  const snapped = snapWindow(...window.map(Number));
  if (!snapped) return null;
  let [lo, hi] = snapped;
  if (iscrowd != null && (iscrowd.length !== intervals.length ||
    iscrowd.some((flag) => flag != null && ![0, 1, false, true].includes(flag)))) return null;
  const spans = [];
  const flags = [];
  for (let i = 0; i < intervals.length; i++) {
    const [a0, b0, c0] = intervals[i];
    const edges = snapSpan(Number(a0), Number(b0));
    if (!edges) return null;
    const [a, b] = edges;
    const c = Math.trunc(c0);
    if (b < a || (known && c && !known.includes(c))) return null;
    if (b > a) {
      spans.push([a, b, c]);
      flags.push(iscrowd?.[i] == null ? null : Number(iscrowd[i]));
    }
  }
  for (const [a, b, c] of spans) {
    if (c) [lo, hi] = [Math.min(lo, a), Math.max(hi, b)];
  }
  if (!(hi - lo > 0 && hi - lo <= LONGEST_WINDOW_MS)) return null;
  if (flags.some((flag) => flag != null)) {
    const annotated = [false, true].flatMap((crowd) => {
      const ids = spans.flatMap(([a, b, c], i) =>
        (c || flags[i] != null) && (flags[i] === 1) !== crowd
          ? [] : [[a, b, c ? i + 1 : 0]]);
      return runs(paint([lo, hi], ids)).filter(([, , id]) => id);
    }).sort((a, b) => a[0] - b[0] || a[1] - b[1] ||
      Number(flags[a[2] - 1] === 1) - Number(flags[b[2] - 1] === 1));
    const keptFlags = annotated.map(([, , id]) => flags[id - 1]);
    const keptSpans = annotated.map(([a, b, id]) => [lo + a, lo + b, spans[id - 1][2]]);
    if (keptFlags.some((flag) => flag != null)) return {
      window: [lo, hi],
      intervals: keptSpans,
      iscrowd: keptFlags,
    };
    spans.splice(0, spans.length, ...keptSpans);
  }
  const painted = runs(paint([lo, hi], spans)).filter(([, , c]) => c);
  return { window: [lo, hi], intervals: painted.map(([a, b, c]) => [lo + a, lo + b, c]) };
}

/** The ms ranges where two labels disagree; outside a window counts as unknown. */
function diffRuns(a, b) {
  const lo = Math.min(a.window[0], b.window[0]);
  const hi = Math.max(a.window[1], b.window[1]);
  const cells = (label) => {
    const out = [0, 1].map(() => {
      const lane = new Int32Array(hi - lo).fill(-1);
      lane.fill(0, label.window[0] - lo, label.window[1] - lo);
      return lane;
    });
    label.intervals.forEach(([s, e, c], i) => {
      const flag = label.iscrowd?.[i];
      out[flag === 1 ? 1 : 0].fill(c * 2 + (flag === 0 ? 1 : 0), s - lo, e - lo);
    });
    return out;
  };
  const [x, y] = [cells(a), cells(b)];
  return runs(x[0].map((v, i) => (v === y[0][i] && x[1][i] === y[1][i] ? 0 : 1)))
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
function hitTest(label, x, y, height, px, lane) {
  if (y >= height - HANDLE_BAND) {
    for (const edge of [0, 1]) {
      if (Math.abs(x - px(label.window[edge])) <= GRAB) return { kind: "window", edge };
    }
  }
  const spans = label.intervals;
  const matches = (i) => lane == null || (label.iscrowd?.[i] === 1 ? 1 : 0) === lane;
  for (let index = spans.length - 1; index >= 0; index--) {
    if (!matches(index)) continue;
    for (const edge of [0, 1]) {
      if (Math.abs(x - px(spans[index][edge])) <= GRAB) return { kind: "edge", index, edge };
    }
  }
  for (let index = spans.length - 1; index >= 0; index--) {
    if (!matches(index)) continue;
    if (x > px(spans[index][0]) && x < px(spans[index][1])) return { kind: "move", index };
  }
  return { kind: "new" };
}


/** A client using one timeline would silently remove the overlap. */
function hasOverlaps(label) {
  if (!label) return false;
  let end = label.window[0];
  for (const [a, b] of [...label.intervals].sort((a, b) => a[0] - b[0] || a[1] - b[1])) {
    if (a < end) return true;
    end = Math.max(end, b);
  }
  return false;
}

/** `intervals` cut to the ranges in `allowed` ([[a, b], ...], sorted and disjoint): a span
 * that crosses a gap comes back as its pieces, each keeping its category and lane flag.
 * The server's `Region.clip`; a subcategory may be labelled only inside its parent. */
function clipToAllowed(intervals, iscrowd, allowed) {
  const kept = [];
  const flags = [];
  intervals.forEach(([a, b, c], i) => {
    for (const [lo, hi] of allowed) {
      const [start, stop] = [Math.max(a, lo), Math.min(b, hi)];
      if (stop > start) {
        kept.push([start, stop, c]);
        flags.push(iscrowd?.[i] ?? null);
      }
    }
  });
  return { intervals: kept, iscrowd: iscrowd ? flags : undefined };
}

/** The ranges of `window` that `allowed` leaves out, where a subcategory cannot be labelled. */
function gaps(window, allowed) {
  const out = [];
  let from = window[0];
  for (const [a, b] of allowed) {
    if (a > from) out.push([from, Math.min(a, window[1])]);
    from = Math.max(from, b);
  }
  if (from < window[1]) out.push([from, window[1]]);
  return out.filter(([a, b]) => b > a);
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

/** A modes row's colours for its bytes, `level * K + i` (`verify.mode_bytes`): mode
 * i's colour scaled by its level, black at or below the contrast floor as in `lut`. */
function modeLut(modes, lo) {
  const [k, top] = [modes.colours.length, modes.levels - 1];
  const floor = (lo / 255) * top;
  const rgb = modes.colours.map((hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)));
  const out = new Uint32Array(256);
  for (let v = 0; v < 256; v++) {
    const above = Math.max(0, Math.min(top, Math.floor(v / k)) - floor);
    const [r, g, b] = rgb[v % k].map((c) => Math.round((c * above) / (top - floor)));
    out[v] = ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
  }
  return out;
}

// ---- The page ----

const GUTTER = 96; // px left of every plot: a row's title and units, then its y ticks
const RIGHT = 12;
const IMAGE_H = 150;
const TRACE_H = 110;
const CATEGORY_COLOURS = {
  2: "#e69f00", 3: "#cc79a7", 4: "#56b4e9", 5: "#d55e00", 6: "#009e73",
}; // 1 is --label; q_min and confinement keep 1-4 as their bands, 5 is uncertain, 6 is I-mode.
const MARKER_COLOURS = ["#e69f00", "#56b4e9", "#cc79a7", "#009e73", "#d55e00", "#0072b2"];
const TRACE_COLOURS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"];
const FONT = "11px system-ui, sans-serif";
const GUTTER_LINE = 14; // px between the lines of a row's title and units

const S = {
  events: [],
  event: null,
  categories: {},
  category: 1,
  crowd: 1, // resolution for new annotations: 0 individual, 1 span (iscrowd), null unspecified
  queue: [], // what the page offers: `allQueue`, or only its gold shots when "Gold set" is ticked
  allQueue: [], // the event's whole queue, as the server sent it
  queueEvent: null, // whose queue has arrived; null while loading
  saveCount: 0,
  savedRows: new Map(), // event -> shot -> {count, row}, from acknowledged saves
  shot: null,
  meta: null, // what /api/shot said: grid, t_range, rows, source, saved, state, last_save
  video: null, // camera cards, selected time and playback; detachment only
  blind: false, // independent detachment review: no producer-derived drawings
  suggestionsShown: false,
  prefilled: false,
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
  name: "", // the reviewer's name, sent with each save: picked from the list (api 5) or typed
  picked: false, // a name was picked in this tab: the picker can then be closed without one
  versions: [], // the open shot's saved versions, as /api/history listed them
  masks: null, // the open AE shot's pseudo-mask regions, as /api/masks gave them (api 3)
  tokeye: null, // TokEye's lines over the whole shot, as /api/tokeye gave them (api 4)
  showMasks: true, // M shows and hides both; the browser remembers which
  showMarkers: true, // T shows and hides the event markers; the browser remembers which
  workspace: false, // /api/version said `workspace: true`: views, the dock and pins (api 11)
};
const Silver = { mode: false, context: false, batch: 1, shot: null,
  data: null, overview: null, layers: false, sequence: 0 };
const Gold = { only: false, shots: null }; // the "Gold set" box; the gold shot numbers, once read
let T = {}; // colour tokens, read from the stylesheet
const $ = (id) => document.getElementById(id);
const enc = encodeURIComponent;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const sleep = (delay) => new Promise((done) => setTimeout(done, delay));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const known = () => Object.keys(S.categories).map(Number);
const EDGE_MARKERS = new Set(["ELM onset"]); // marker sets drawn as short ticks on a row's lower edge
const EDGE_TICK = 6; // their height, px
const markerColour = (i) => MARKER_COLOURS[i % MARKER_COLOURS.length];

/** Where `event` sits in the events listing: its row, and its chip if it is a subcategory. */
function locate(event) {
  for (const row of S.events) {
    if (row.event === event) return { row, chip: null };
    const chip = row.subcategories?.find((sub) => sub.event === event);
    if (chip) return { row, chip };
  }
  return null;
}
const hasEvent = (event) => Boolean(locate(event)) && !locate(event).row.error;
const draftKey = (shot) => `labeler:${S.event}:${shot}${S.event === "detachment" ? ":independent" : ""}`;
const exposureKey = (shot) => `${draftKey(shot)}:exposure`;

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

/** One request: the global fetch, once. `Activity` tracks it (a `quiet` one, such as a
 * prefetch or a poll, is tracked but never shown); a failure keeps the server's short `error`
 * as its message, and its `kind`, `code` and full `detail` (a string FastAPI `detail` is the
 * message when there is no `error`). */
async function api(path, options = {}) {
  const { quiet = false, ...rest } = options || {};
  const id = Activity.start(path, { quiet, method: rest.method });
  let response;
  try {
    response = await fetch(path, { credentials: "same-origin", ...rest });
  } catch (error) {
    Activity.end(id, error);
    throw error;
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const plain = typeof body.detail === "string" ? body.detail : "";
    const error = new Error(body.error || plain || `${response.status} ${response.statusText}`);
    error.status = response.status;
    if (body.kind) error.kind = body.kind;
    if (body.code) error.code = body.code;
    if (body.error && plain && plain !== body.error) error.detail = plain;
    Activity.end(id, error);
    throw error;
  }
  Activity.end(id);
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
  const names = ["panel", "ink", "muted", "rule", "label", "changed", "veil", "mask", "rejected", "tokeye", "unreviewed"];
  T = Object.fromEntries(names.map((name) => [name, style.getPropertyValue(`--${name}`).trim()]));
}

// -- time and geometry

/** The workspace (views, dock, pins) is on: api 11 and a server that says so. */
const ws = () => S.api >= 11 && S.workspace;
/** The px right of every plot: RIGHT, plus the cut margin while the dock shows one. */
const right = () => RIGHT + (ws() ? Slice.margin : 0);
const plotWidth = () => Math.max(1, $("axis-row").clientWidth - GUTTER - right());
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
  // A subcategory is looked at over its parent's saved window, nothing else.
  if (S.meta.parent) return { window: [...S.meta.parent.label.window], intervals: [] };
  // The whole frames the shot's data touch.
  const [t0, t1] = S.meta.t_range;
  const [lo, hi] = [Math.floor(t0 / GRID_MS) * GRID_MS, Math.ceil(t1 / GRID_MS) * GRID_MS];
  return { window: [lo, Math.min(hi, lo + LONGEST_WINDOW_MS)], intervals: [] };
}

/** A subcategory's label, as the page may make it: the parent's window, and only spans
 * inside the parent's saved spans (the server's rule, applied as the reviewer draws). */
function constrain(window, intervals, iscrowd) {
  const parent = S.meta?.parent;
  if (!parent) return [window, intervals, iscrowd];
  const kept = clipToAllowed(intervals, iscrowd, S.meta.allowed);
  return [parent.label.window, kept.intervals, kept.iscrowd];
}

/** The label `normalise` makes of these arguments once `constrain` has had them; null if refused. */
function constrained(window, intervals, iscrowd) {
  const [w, spans, flags] = constrain(window, intervals, iscrowd);
  return normalise(w, spans, known(), flags);
}

const independent = () => S.event === "detachment";
const isBlind = () => independent() && S.blind;
// Unsaved detachment opens on its Source's states 1–3 (a recorded prefill), empty in Blind mode.
const baseline = () => S.meta.saved ||
  (independent() ? (S.blind ? emptyLabel() : sourceCopy()) : S.meta.source || emptyLabel());
const dirty = () => Boolean(S.meta) && !same(S.label, baseline());

/** The Source's states 1–3 as an editable label; state 4 is a machine abstention. */
function sourceCopy() {
  const source = S.meta.source;
  const keep = source ? source.intervals.map((_, i) => i).filter(i => [1, 2, 3].includes(source.intervals[i][2])) : [];
  if (!keep.length) return emptyLabel();
  return constrained(source.window, keep.map(i => source.intervals[i]),
    source.iscrowd && keep.map(i => source.iscrowd[i])) || emptyLabel();
}

/** An untouched, unsaved detachment shot as a fresh load opens it in the current mode. */
function reopenFresh() {
  if (stillOpening() || !S.meta) return;
  S.label = baseline();
  S.selected = -1;
  if (!S.blind && S.label.intervals.length) S.prefilled = mergeExposure(S.prefilled, true);
  touch();
}

function keepExposure() {
  if (independent() && S.meta) store(exposureKey(S.shot), JSON.stringify({
    suggestions_shown: S.suggestionsShown, prefilled: S.prefilled,
  }));
}

const mergeExposure = (a, b) => a === true || b === true ? true :
  a == null || b == null ? null : false;

function previousExposure(shot) {
  try { return JSON.parse(stored(exposureKey(shot)) || "null"); }
  catch { return null; }
}

function draft(shot) {
  try {
    const label = JSON.parse(stored(draftKey(shot)));
    if (independent() && label) {
      S.suggestionsShown = mergeExposure(S.suggestionsShown, label.suggestions_shown);
      S.prefilled = mergeExposure(S.prefilled, label.prefilled);
    }
    return label && constrained(label.window, label.intervals, label.iscrowd);
  } catch {
    return null;
  }
}

function keepForUndo(label) {
  // `depth` counts the edits Ctrl+Z can step back through, past the 100 kept: a row taken
  // out (Views.removed) compares with it, so the cap never makes a later edit look older.
  S.undo.depth = undoDepth() + 1;
  S.undo.push(label);
  if (S.undo.length > 100) S.undo.shift();
}

/** How many label edits deep the undo is (a new `S.undo` array starts at its length). */
function undoDepth() {
  return S.undo.depth ?? S.undo.length;
}

/** Take a new label if the server would accept it. */
function edit(window, intervals, iscrowd) {
  if (stillOpening()) return;
  if (!S.meta) return;
  const next = constrained(window, intervals, iscrowd);
  if (!next) return;
  keepForUndo(S.label);
  S.label = next;
  touch();
}

/** Keep the draft until it is saved or reverted, and mark the shot unsaved. */
function touch() {
  if (stillOpening()) return;
  keepExposure();
  const saving = S.saving && S.saving.event === S.event && S.saving.shot === S.shot;
  const keep = dirty() || (saving && !same(S.label, S.saving.label));
  store(draftKey(S.shot), keep ? JSON.stringify({ ...S.label,
    ...(independent() ? { suggestions_shown: S.suggestionsShown, prefilled: S.prefilled } : {}) }) : null);
  $("dirty").hidden = !dirty();
  renderQueue();
  render();
}

/** Select the span holding `t`, if any. */
function selectAt(t, flag) {
  S.selected = S.label.intervals.findIndex(([a, b], i) => a <= t && t <= b &&
    (flag === undefined || (S.label.iscrowd?.[i] ?? null) === flag));
}

function undo() {
  if (stillOpening()) return;
  if (!S.undo.length) return;
  const depth = undoDepth() - 1;
  S.label = S.undo.pop();
  S.undo.depth = depth;
  S.selected = -1;
  touch();
}

function revert() {
  if (stillOpening()) return;
  if (!S.meta) return;
  if (isBlind()) return;
  const source = S.meta.source || emptyLabel();
  if (independent()) {
    const copy = sourceCopy();
    S.prefilled = mergeExposure(S.prefilled, copy.intervals.length > 0);
    S.suggestionsShown = true;
    S.selected = -1;
    edit(copy.window, copy.intervals, copy.iscrowd);
    return;
  }
  S.selected = -1;
  edit(source.window, source.intervals, source.iscrowd);
}

/** Clear only Individual/unspecified annotations, retaining the Span lane. */
function startBlank() {
  if (stillOpening() || !S.meta) return;
  const crowd = S.label.intervals.filter((_, i) => S.label.iscrowd?.[i] === 1);
  S.selected = -1;
  edit(S.label.window, crowd, crowd.map(() => 1));
}

/** Delete or Backspace: the selected annotation, else the selected row (workspace). */
function deleteSelected() {
  if (S.selected >= 0 || !(ws() && Views.removeSelectedRow())) removeSelected();
}

function removeSelected() {
  if (stillOpening()) return;
  if (S.selected < 0) return;
  const kept = S.label.intervals.filter((_, i) => i !== S.selected);
  const flags = S.label.iscrowd?.filter((_, i) => i !== S.selected);
  S.selected = -1;
  edit(S.label.window, kept, flags);
}

function setCategory(c) {
  if (stillOpening()) return;
  if (!known().includes(c)) return;
  S.category = c;
  renderSwatches();
  if (S.selected < 0) return;
  const [a, b] = S.label.intervals[S.selected];
  const flag = S.label.iscrowd?.[S.selected] ?? null;
  edit(S.label.window, S.label.intervals.map((s, i) => (i === S.selected ? [a, b, c] : s)),
    S.label.iscrowd);
  selectAt((a + b) / 2, flag);
}

/** Resolution is independent of the event category. Legacy labels stay unspecified. */
function setCrowd(value) {
  if (stillOpening() || S.api < 7 || !S.meta) return;
  const flag = value === "unspecified" ? null : Number(value);
  if (flag != null && ![0, 1].includes(flag)) return;
  S.crowd = flag;
  if (S.selected < 0) return renderResolution();
  const spans = S.label.intervals.filter((_, i) => i !== S.selected);
  const flags = S.label.intervals.flatMap((_, i) => i === S.selected ? [] : [S.label.iscrowd?.[i] ?? null]);
  const moved = S.label.intervals[S.selected];
  spans.push(moved);
  flags.push(flag);
  edit(S.label.window, spans, flags);
  selectAt((moved[0] + moved[1]) / 2, flag);
  render();
}

function renderResolution() {
  $("resolution-control").hidden = S.api < 7;
  $("crowd-lane").hidden = S.api < 8;
  $("individual-lane-name").textContent = S.api >= 8 ? "Individual" : "Label";
  const selected = S.selected >= 0 && S.label?.intervals[S.selected];
  $("resolution-label").textContent = selected ? "Selected annotation" : "New annotations";
  const flag = selected ? S.label.iscrowd?.[S.selected] ?? null : S.crowd;
  $("resolution").value = flag == null ? "unspecified" : String(flag);
  $("resolution").disabled = !S.meta || pendingNavigation();
}

function contrast(delta) {
  S.lo = clamp(S.lo + delta, 0, 192);
  S.lut = lut(S.lo);
  store("labeler:contrast", String(S.lo));
  for (const data of [S.data, S.overview]) data?.bitmaps.clear();
  render();
}

// -- loading

function showSilverMode() {
  document.body.classList.toggle("silver-mode", Silver.mode);
  $("silver-page").hidden = !Silver.mode;
  $("top").hidden = $("bottom").hidden = Silver.mode;
  $("silver-breadcrumb").hidden = !Silver.context;
  $("silver-breadcrumb").textContent = `Silver · batch ${Silver.batch} · shot ${Silver.shot}`;
  $("silver-next-event").hidden = !Silver.context || !S.meta?.saved;
}

function silverAddress(event) {
  history.replaceState(null, "", `#silver/${Silver.batch}/${Silver.shot || ""}${event ? `/${event}` : ""}`);
}

async function openSilver(batch = 1, shot = null, event = null) {
  const ticket = ++S.ticket;
  leave();
  busy(null);
  Object.assign(S, { meta: null, label: null, event: "silver", data: null, overview: null });
  Object.assign(Silver, { mode: true, context: false, batch: clamp(batch, 1, 5), shot, overview: null });
  Silver.sequence++;
  $("event").value = "silver";
  $("silver-strips").replaceChildren();
  $("silver-title").textContent = "Silver set";
  showSilverMode();
  silverAddress();
  try {
    const data = await (await api("/api/silver")).json();
    if (ticket !== S.ticket) return;
    Silver.data = data;
    renderSilverList();
    if (data.error) return say(data.error, true);
    say("");
    if (shot != null) {
      await selectSilverShot(shot);
      if (event && ticket === S.ticket && hasEvent(event)) await openSilverEvent(event);
    }
  } catch (error) {
    if (ticket === S.ticket) say(error.message, true);
  }
}

function renderSilverList() {
  $("silver-batches").replaceChildren(...(Silver.data?.batches || []).map(batch => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${batch.batch} · ${batch.shots.filter(shot => shot.done).length}/${batch.shots.length} done`;
    button.setAttribute("aria-pressed", String(batch.batch === Silver.batch));
    button.addEventListener("click", () => openSilver(batch.batch));
    return button;
  }));
  const shots = Silver.data?.batches.find(batch => batch.batch === Silver.batch)?.shots || [];
  $("silver-shots").replaceChildren(...shots.map(shot => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `silver-shot${shot.shot === Silver.shot ? " current" : ""}`;
    button.dataset.shot = shot.shot;
    button.setAttribute("aria-pressed", String(shot.shot === Silver.shot));
    const order = Object.assign(document.createElement("span"), { textContent: shot.order, className: "silver-order" });
    const number = Object.assign(document.createElement("strong"), { textContent: shot.shot });
    const classes = Object.assign(document.createElement("span"), { className: "silver-classes" });
    if (SILVER_POOLS[shot.pool]) classes.append(Object.assign(document.createElement("span"), {
      textContent: SILVER_POOLS[shot.pool], className: "silver-tag",
    }));
    for (const name of shot.classes) {
      const [event, category] = name.split(":");
      const short = event.replace("alfven_eigenmode", "AE").replace("edge_localized_mode", "ELM")
        .replace("minimum_safety_factor", "q min").replace("confinement", "conf");
      const chip = Object.assign(document.createElement("span"), {
        textContent: `${short}:${category}`, className: "silver-class",
      });
      chip.setAttribute("aria-label", name);
      classes.append(chip);
    }
    const progress = Object.assign(document.createElement("span"), {
      textContent: `${shot.saved}/${shot.total}${shot.done ? " ✓" : ""}`, className: "silver-progress",
    });
    button.append(order, number, classes, progress);
    button.addEventListener("click", () => selectSilverShot(shot.shot));
    return button;
  }));
}

async function selectSilverShot(shot) {
  const sequence = ++Silver.sequence;
  Silver.shot = shot;
  Silver.overview = null;
  $("silver-strips").replaceChildren();
  $("silver-title").textContent = `Shot ${shot}`;
  renderSilverList();
  silverAddress();
  try {
    const overview = await (await api(`/api/silver/shot?shot=${shot}`)).json();
    if (!Silver.mode || sequence !== Silver.sequence) return;
    Silver.overview = overview;
    Silver.batch = overview.batch;
    silverAddress();
    renderSilverList();
    renderSilverOverview();
    say("");
  } catch (error) {
    if (Silver.mode && sequence === Silver.sequence) say(error.message, true);
  }
}

/** One common grid; a higher layer's unassessed frames fall through. */
function silverFrames(event, range, layers = event.layers) {
  const [lo, hi] = range;
  const n = Math.max(0, (hi - lo) / GRID_MS);
  const cells = new Int8Array(n).fill(-1), origin = new Int16Array(n).fill(-1);
  const index = t => clamp((t - lo) / GRID_MS, 0, n);
  layers.forEach((layer, i) => {
    const label = layer.label;
    const values = new Int8Array(n).fill(-1);
    values.fill(0, index(label.window[0]), index(label.window[1]));
    for (const [a, b, c] of label.intervals) values.fill(c, index(a), index(b));
    // A table's not-observable stretches say nothing: a lower layer may. The
    // owner's own in a save (held) stay unassessed: no lower layer fills them.
    for (const [a, b] of layer.unknown || []) values.fill(-1, index(a), index(b));
    const held = new Uint8Array(n);
    for (const [a, b] of layer.held || []) held.fill(1, index(a), index(b));
    for (let k = 0; k < n; k++) {
      const t = lo + k * GRID_MS;
      if (event.allowed && !event.allowed.some(([a, b]) => a <= t && t < b)) continue;
      if (origin[k] !== -1) continue;
      if (values[k] !== -1) { cells[k] = values[k]; origin[k] = i; } else if (held[k]) origin[k] = i;
    }
  });
  return { cells, origin };
}

function drawSilverCanvas(canvas, event, layers = event.layers) {
  const range = Silver.overview.t_range;
  const { cells, origin } = silverFrames(event, range, layers);
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * ratio);
  canvas.height = Math.round(canvas.clientHeight * ratio);
  const g = context(canvas), w = canvas.clientWidth, h = canvas.clientHeight;
  const x = k => k / Math.max(1, cells.length) * w;
  for (const [a, b, c] of runs(cells)) {
    if (c === -1) {
      g.save(); g.beginPath(); g.rect(x(a), 0, x(b) - x(a), h); g.clip();
      g.strokeStyle = T.rule;
      for (let edge = x(a) - h; edge < x(b); edge += 7) {
        g.beginPath(); g.moveTo(edge, h); g.lineTo(edge + h, 0); g.stroke();
      }
      g.restore();
    } else {
      g.fillStyle = c === 0 ? T.unreviewed : categoryColour(c, event.event, event.categories);
      g.fillRect(x(a), 0, x(b) - x(a), h);
    }
  }
  return [...new Set(origin)].filter(i => i >= 0).map(i => layers[i].name).join(" · ") || "unassessed";
}

function renderSilverOverview() {
  if (!Silver.mode || !Silver.overview) return;
  const overview = Silver.overview;
  $("silver-title").textContent = `Shot ${overview.shot}`;
  $("silver-time").textContent = `${overview.t_range[0]}–${overview.t_range[1]} ms · 10 ms frames`;
  $("silver-strips").replaceChildren(...overview.events.map(event => {
    const section = document.createElement("section");
    section.className = `silver-event${event.parent ? " child" : ""}${event.locked ? " locked" : ""}`;
    const row = document.createElement("button");
    row.type = "button";
    row.className = "silver-strip";
    row.disabled = event.locked;
    const name = Object.assign(document.createElement("span"), {
      textContent: `${event.display_name.slice(0, 24)}${event.saved ? " ✓" : ""}`, className: "silver-event-name",
    });
    row.setAttribute("aria-label", event.display_name);
    const canvas = document.createElement("canvas");
    const origin = Object.assign(document.createElement("span"), { className: "silver-origin" });
    row.append(name, canvas, origin);
    row.addEventListener("click", () => openSilverEvent(event.event));
    section.append(row);
    if (Silver.layers) for (const layer of event.layers) {
      const item = Object.assign(document.createElement("div"), { className: "silver-strip silver-layer" });
      const caption = Object.assign(document.createElement("span"), { textContent: layer.kind });
      const canvas = document.createElement("canvas");
      const origin = Object.assign(document.createElement("span"), {
        textContent: layer.name, className: "silver-origin",
      });
      item.append(caption, canvas, origin);
      item.dataset.layer = event.layers.indexOf(layer);
      section.append(item);
    }
    section.dataset.event = event.event;
    return section;
  }));
  requestAnimationFrame(drawSilverOverview);
}

function drawSilverOverview() {
  if (!Silver.mode || !Silver.overview) return;
  for (const section of $("silver-strips").children) {
    const event = Silver.overview.events.find(event => event.event === section.dataset.event);
    const row = section.firstChild;
    row.lastChild.textContent = event.locked ? "Review parent first" : drawSilverCanvas(row.querySelector("canvas"), event);
    for (const item of [...section.children].slice(1))
      drawSilverCanvas(item.querySelector("canvas"), event, [event.layers[Number(item.dataset.layer)]]);
  }
  const canvas = $("silver-axis"), ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * ratio);
  canvas.height = Math.round(canvas.clientHeight * ratio);
  const g = context(canvas), [lo, hi] = Silver.overview.t_range;
  const step = niceStep(hi - lo, 6);
  g.fillStyle = T.muted; g.font = FONT;
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) {
    const x = (t - lo) / (hi - lo) * canvas.clientWidth;
    g.fillRect(x, 0, 1, 5);
    g.fillText(`${t}`, clamp(x + 3, 0, Math.max(0, canvas.clientWidth - 45)), 19);
  }
}

async function openSilverEvent(event) {
  const shot = Silver.shot;
  Silver.context = true;
  Silver.mode = false;
  await openEvent(event, shot, true);
}

async function nextSilverEvent() {
  const shot = Silver.shot, event = S.event;
  const overview = await (await api(`/api/silver/shot?shot=${shot}`)).json();
  if (!Silver.context || Silver.shot !== shot || S.event !== event) return;
  const index = overview.events.findIndex(row => row.event === event);
  const ordered = [...overview.events.slice(index + 1), ...overview.events.slice(0, index)];
  const next = ordered.find(row => !row.saved && !row.locked);
  if (next) return openSilverEvent(next.event);
  Silver.data = await (await api("/api/silver")).json();
  if (!Silver.context || Silver.shot !== shot || S.event !== event) return;
  const shots = Silver.data.batches.flatMap(batch => batch.shots.map(shot => ({ ...shot, batch: batch.batch })));
  const at = shots.findIndex(row => row.shot === shot);
  const after = [...shots.slice(at + 1), ...shots.slice(0, at)].find(row => !row.done);
  await openSilver(after?.batch || Silver.batch, after?.shot || null);
  if (after && Silver.overview) {
    const next = Silver.overview.events.find(row => !row.saved && !row.locked);
    if (next) await openSilverEvent(next.event);
  }
}

function silverKey(key) {
  if (!Silver.mode) {
    if (Silver.context && ["g", "Escape"].includes(key)) { openSilver(Silver.batch, Silver.shot); return true; }
    return false;
  }
  const shots = Silver.data?.batches.find(batch => batch.batch === Silver.batch)?.shots || [];
  if (["ArrowDown", "ArrowUp", "j", "k"].includes(key)) {
    const at = shots.findIndex(row => row.shot === Silver.shot);
    const delta = ["ArrowDown", "j"].includes(key) ? 1 : -1;
    const next = shots[(at + delta + shots.length) % shots.length];
    if (next) selectSilverShot(next.shot);
  } else if (key === "Enter") {
    if (!Silver.overview && shots.length) selectSilverShot(Silver.shot || shots[0].shot);
    else {
      const next = Silver.overview?.events.find(row => !row.saved && !row.locked)
        || Silver.overview?.events.find(row => !row.locked);
      if (next) openSilverEvent(next.event);
    }
  } else if (["g", "Escape"].includes(key)) {
    if (Silver.shot) openSilver(Silver.batch);
    else {
      const event = [stored("labeler:event"), ...S.events.map(row => row.event)].find(hasEvent);
      if (event) openEvent(event);
    }
  } else return false;
  return true;
}

async function boot() {
  readTokens();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    readTokens();
    renderSwatches();
    render();
  });
  S.lo = clamp(Number(stored("labeler:contrast")) || 0, 0, 192);
  S.lut = lut(S.lo);
  S.picked = Boolean(picked());
  S.name = picked() || stored("labeler:name") || "";
  $("reviewer-name").value = S.name;
  S.showMasks = stored("labeler:masks") !== "hidden";
  S.blind = stored("labeler:detachment-blind") === "true";
  Gold.only = $("gold-only").checked = stored("labeler:gold-only") === "1";
  wire();
  let version = null, versionError = null;
  try {
    version = await (await api("/api/version")).json();
    S.api = version.api;
    S.workspace = version.workspace === true;
  } catch (error) {
    S.api = 1; // a server older than this page: save without a name, no history
    versionError = error;
  }
  $("show-versions").hidden = S.api < 2;
  $("reviewer-name").hidden = S.api < 2 || S.api >= 5;
  $("reviewer").hidden = S.api < 5;
  // api 11: shot lists, All, the activity line and TokEye chips (status.js, lists.js,
  // overlays.js); an older server keeps the page it had, and #stale says what a restart adds.
  const failed = versionError && Activity.versionNote(versionError);
  $("stale").hidden = S.api >= 11 && !failed;
  $("stale").textContent = failed || (Activity.staleText(S.api) ?? "");
  Activity.enabled = S.api >= 11;
  $("shotlist").hidden = $("all-key").hidden = S.api < 11;
  $("gold-only-label").hidden = S.api >= 11;
  if (S.api >= 11) {
    Gold.only = $("gold-only").checked = false; // the "Gold set" box became the gold list
    const fdp = Activity.fdpNote(version?.fdp);
    if (fdp) Activity.note("fdp", fdp.text, fdp.tone);
  }
  if (ws()) {
    // Views, the dock and pins (signals.js, slice.js, dock.js, views.js); without them the page
    // is today's. #view lists Default and Edit view… until it gets focus (no request here).
    $("view").hidden = false;
    Views.fillSelect();
    buildDock();
    for (const line of document.querySelectorAll("#keys .workspace-key")) line.hidden = false;
  }
  renderActivity();
  if (S.api >= 5) {
    showName();
    if (!S.picked) chooseName(); // not awaited: the shot loads behind the list
  }
  try {
    S.events = (await (await api("/api/events")).json()).events;
    fillEvents();
    if (S.api >= 11) {
      Lists.migrate(stored, store, S.events);
      const at = Lists.parseAddress(location.hash); // #list/…, or #silver/… redirected to it
      if (at) return await openByShot(at.list, at.shot, at.event);
    } else if (location.hash.startsWith("#silver/")) {
      const [, batch, shot, event] = location.hash.slice(1).split("/");
      return await openSilver(Number(batch) || 1, Number(shot) || null, event);
    }
    const usable = S.events.filter((row) => !row.error);
    if (!usable.length) return say("no events", true);
    const [hashEvent, hashShot] = fromHash();
    const pick =
      [hashEvent, stored("labeler:event")].find(hasEvent) ||
      usable.reduce((a, b) => (b.n_shots > a.n_shots ? b : a)).event;
    await openEvent(pick, hashShot);
  } catch (error) {
    say(error.message, true);
  }
}

/** The event dropdown: All first (every event's label on the open shot; "Silver set" before
 * api 11), then every event by name in one flat list (the taxonomy's `group` is not shown).
 * Subcategories are not listed: the bar under it has them. */
function fillEvents() {
  const option = (row) => {
    const name = row.display_name || row.event;
    const found = new Option(row.error ? `${name} (broken)` : name, row.event);
    found.disabled = Boolean(row.error);
    found.title = row.error || "";
    return found;
  };
  const byName = (a, b) => (a.display_name || a.event).localeCompare(b.display_name || b.event);
  const first = S.api >= 11
    ? Object.assign(new Option("All", Lists.ALL), { title: "Every event's label on the shot" })
    : new Option("Silver set", "silver");
  $("event").replaceChildren(first, ...[...S.events].sort(byName).map(option));
}

/** `#event/shot`, the address of what is open (`#list/<name>/<shot>/<event>` at api 11). */
function fromHash() {
  const at = S.api >= 11 && location.hash.startsWith("#list/") ? Lists.parseAddress(location.hash) : null;
  if (at) return [at.event, at.shot];
  const parts = decodeURIComponent(location.hash.slice(1)).split("/");
  const [event, shot] = parts[0] === "silver" ? [parts[3] || "silver", parts[2]] : parts;
  return [event, Number(shot) || null];
}

/** A link pasted into the address bar opens what it names. */
function followHash() {
  const at = S.api >= 11 ? Lists.parseAddress(location.hash) : null;
  if (at) {
    const shown = Lists.name === at.list && (Lists.byShot
      ? Lists.shot === at.shot && !at.event
      : Lists.context && Lists.shot === at.shot && S.event === at.event && S.shot === at.shot);
    if (!shown) openByShot(at.list, at.shot, at.event).catch((error) => Activity.note("by-shot", error.message, "error"));
    return;
  }
  if (S.api < 11 && location.hash.startsWith("#silver/")) {
    const [, batch, shot, event] = location.hash.slice(1).split("/");
    if (Silver.batch === Number(batch) && Silver.shot === Number(shot) &&
      (Silver.mode && !event || Silver.context && S.event === event && S.shot === Number(shot))) return;
    openSilver(Number(batch) || 1, Number(shot) || null, event).catch(error => say(error.message, true));
    return;
  }
  Silver.context = false;
  Lists.context = false;
  const [event, shot] = fromHash();
  if ((event !== S.event || Lists.byShot) && hasEvent(event)) {
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
  clearVideo();
  closeDialog($("versions"));
  S.versions = [];
  S.versionsAt = null;
  S.drag = null;
  clearTimeout(S.timer);
}

async function openEvent(event, shot, silverReview = false) {
  if (event === "silver") return openSilver(Silver.batch, Silver.shot);
  if (S.api >= 11) {
    // `silverReview`: opened from By shot, so the queue is By shot's list and G goes back
    Lists.context = silverReview || (Lists.context && Lists.shot === shot);
    Lists.byShot = false;
    Lists.show();
  } else {
    Silver.context = silverReview || (Silver.context && Silver.shot === shot);
    Silver.mode = false;
    showSilverMode();
  }
  const ticket = ++S.ticket;
  leave();
  S.queue = S.allQueue = [];
  S.queueEvent = null;
  S.event = event;
  if (ws()) Views.restore(event); // Default asks nothing; a saved view asks /api/views once
  const { row } = locate(event);
  S.categories = row.categories; // a subcategory offers its parent's present and uncertain
  S.category = known()[0] || 1;
  S.crowd = 1; // a span; the Resolution selector picks Individual or Unspecified
  $("event").value = row.event; // the dropdown holds the parent; the bar under it, the child
  store("labeler:event", event);
  renderSwatches();
  renderQueue();
  showHeader();
  const saveCount = S.saveCount;
  // api 11: `list=` only for a list other than the event's default, so a default open asks
  // today's exact URL; the answer carries the lists for the dropdown.
  const asked = S.api >= 11 ? (Lists.context ? Lists.name : Lists.chosen(event)) : null;
  if (S.api >= 11) $("shotlist").disabled = true; // the lists are the event's once its queue is in
  let queue;
  try {
    queue = await (await api(S.api >= 11 ? Lists.queueUrl(event, asked) : `/api/queue?event=${enc(event)}`)).json();
  } catch (error) {
    if (S.event !== event) return;
    S.queueEvent = event;
    if (S.api >= 11) fillLists({ list: null, lists: [] }, event);
    showNothing(null, `${event}: the queue could not be read: ${error.message}. Choose the event again to retry.`);
    return;
  }
  if (S.event !== event) return;
  if (S.api >= 11) {
    Lists.take(event, queue, saveCount, asked);
    if ($("status").textContent === "the queue is still loading") say("");
    if (ticket !== S.ticket) return;
    return await openShot(shot || (inQueue(queue.resume) ? queue.resume : firstToReview()));
  }
  const saved = S.savedRows.get(event);
  S.allQueue = queue.shots.map((row) => {
    const newer = saved?.get(row.shot);
    return newer && newer.count > saveCount ? newer.row : row;
  });
  if (Gold.only) {
    try {
      await loadGoldShots();
    } catch (error) {
      say(`Could not read the gold shots: ${error.message}`, true);
    }
    if (S.event !== event) return;
  }
  filterQueue();
  S.queueEvent = event;
  if ($("status").textContent === "the queue is still loading") say("");
  renderQueue();
  showHeader();
  if (ticket !== S.ticket) return;
  await openShot(shot || (inQueue(queue.resume) ? queue.resume : firstToReview()));
}

const inQueue = (shot) => S.queue.some((row) => row.shot === shot);

/** The first unreviewed shot on offer, else the first shot on offer. */
function firstToReview() {
  return (S.queue.find((row) => row.state === "unreviewed") || S.queue[0])?.shot ?? null;
}

async function loadGoldShots() {
  if (Gold.shots) return;
  const found = await (await api("/api/gold/shots")).json();
  if (found.error) throw new Error(found.error);
  Gold.shots = new Set(found.shots);
}

/** The queue the page offers: with "Gold set" ticked, only the gold list's shots. */
function filterQueue() {
  // api 11: the server applied the list; the event's own view still leaves out the gold shots
  // its roster lacks, as before, while a list's off-roster shots are its own
  if (S.api >= 11) {
    S.queue = Lists.own ? S.allQueue.filter((row) => !row.extra) : S.allQueue;
    return;
  }
  // `extra` rows are gold shots the event's roster lacks: offered only with the box ticked
  S.queue = Gold.only && Gold.shots
    ? S.allQueue.filter((row) => Gold.shots.has(row.shot))
    : S.allQueue.filter((row) => !row.extra);
}

async function toggleGoldOnly() {
  Gold.only = $("gold-only").checked;
  store("labeler:gold-only", Gold.only ? "1" : null);
  if (Silver.mode || S.queueEvent !== S.event) return; // the silver page lists silver anyway; a loading queue filters on arrival
  if (Gold.only) {
    try {
      await loadGoldShots();
    } catch (error) {
      Gold.only = $("gold-only").checked = false;
      store("labeler:gold-only", null);
      return say(`Could not read the gold shots: ${error.message}`, true);
    }
  }
  filterQueue();
  renderQueue();
  showHeader();
  if (!S.queue.length) return say(`No gold shots on ${S.event}'s list.`);
  say("");
  if (S.shot == null || !inQueue(S.shot)) await openShot(firstToReview());
}

/** Show `shot` (or no shot) with nothing to edit, and the note `text` where the rows go; a
 * `detail` (the server's full text) sits behind a small "details" toggle. */
function showNothing(shot, text, detail = "") {
  clearVideo();
  cancelAnimationFrame(S.frame);
  Object.assign(S, { shot, meta: null, data: null, overview: null, label: null,
    undo: [], selected: -1, asked: "", frame: 0, masks: null, tokeye: null });
  Overlays.open(null); // no chips, no poll
  showMasks();
  const note = document.createElement("p");
  note.className = "failure";
  note.textContent = text;
  if (detail && detail !== text) {
    const more = Object.assign(document.createElement("details"), { className: "failure-detail" });
    more.append(Object.assign(document.createElement("summary"), { textContent: "details" }),
      Object.assign(document.createElement("pre"), { textContent: detail }));
    note.append(" ", more);
  }
  $("rows").replaceChildren(note);
  for (const canvas of document.querySelectorAll("#top canvas, .track canvas")) context(canvas);
  if (typeof Dock !== "undefined") {
    // Its panels are the last shot's and D needs a shot: the dock closes, the next shot opens it.
    if (Dock.open && Dock.show(false)) Dock.reopen = true;
    Dock.placeTab(); // no shot: no dock tab, no Add row
  }
  if (typeof Views !== "undefined") {
    Views.shot = null; // the next shot, even this one again, starts fresh (slices, pins, row undo)
    Views.placeAddBox();
  }
  arrive();
}

async function openShot(shot) {
  clearVideo();
  const ticket = ++S.ticket;
  leave();
  Activity.clear("prefetch", "masks", "layers"); // notes about the shot left behind
  if (shot == null) {
    const { row, chip } = locate(S.event);
    showNothing(null, chip
      ? `No shot has ${row.display_name || row.event} saved yet: save it on a shot, then come back to ${chip.display_name}.`
      : `${S.event} has no shots to review.`);
    return;
  }
  let wait = null; // the bar's item for a store build, with Stop waiting (api 11)
  try {
    let response, meta;
    for (;;) {
      response = await api(`/api/shot?event=${enc(S.event)}&shot=${shot}`);
      meta = await response.json();
      if (ticket !== S.ticket) return;
      if (response.status !== 202) break;
      const p = meta.progress;
      busy(`${shot}: building${p ? ` ${p.done}/${p.total} ${p.stage}` : ""}`);
      if (S.api >= 11) {
        const found = locate(S.event);
        wait = Activity.build(wait, { shot, name: found?.chip?.display_name || found?.row.display_name || S.event, body: meta });
      }
      await sleep(800);
      if (ticket !== S.ticket) return;
    }
    if (wait) wait = Activity.end(wait);
    Object.assign(S, { shot, meta, data: null, overview: null, asked: "", undo: [], selected: -1, masks: null,
      tokeye: null });
    if (S.api >= 11) Overlays.open(meta, ticket); // the rows' TokEye states, before the rows are built
    if (meta.categories) { // a subcategory says what it offers
      S.categories = meta.categories;
      if (!known().includes(S.category)) S.category = known()[0] || 1;
      renderSwatches();
    }
    // Confirmed BCI units also apply to frozen stores whose old UI called
    // the CO2 line integral's native units unverified; preserve store bytes.
    if (independent()) for (const row of meta.rows) {
      const metadata = meta.params?.panel_metadata?.[row.name];
      if (!metadata?.measurement?.includes("CO2 line-integrated")) continue;
      row.y_units = "m·cm⁻³";
      row.legend = row.legend.map(text => text.replace("native units (unverified)", "m·cm⁻³"));
      metadata.caveat = "BCI line integral in m·cm⁻³; no chord-length division, not a line average.";
    }
    const exposure = independent() ? previousExposure(shot) : null;
    S.suggestionsShown = independent() ? mergeExposure(exposure ? exposure.suggestions_shown : false,
      meta.saved ? meta.last_save?.suggestions_shown ?? null : false) : true;
    S.prefilled = independent() ? mergeExposure(exposure ? exposure.prefilled : false,
      meta.saved ? meta.last_save?.prefilled ?? null : false) : Boolean(meta.source && !meta.saved);
    const pending = draft(shot);
    S.label = isBlind() ? baseline() : pending || baseline();
    if (independent() && !S.blind) S.suggestionsShown = true;
    if (independent() && !S.blind && !pending && !meta.saved && S.label.intervals.length)
      S.prefilled = mergeExposure(S.prefilled, true); // opened on the Source
    keepExposure();
    buildRows();
    if (ws()) applyView(); // order, heights and added rows; the dock; a new shot's pins reset
    arrive();
    fit();
    buildVideo();
    fetchRows(0);
    prefetch();
    loadMasks(ticket);
    loadOverlays(ticket);
  } catch (error) {
    if (ticket !== S.ticket) return;
    // Stay on the shot, with nothing to edit, so J, K and U carry on from it. The server's
    // short `error` only (it may name the shot already); its full text behind "details".
    const said = Activity.sentence(String(error.message || "it failed"));
    showNothing(shot, said.includes(String(shot)) ? said : `${shot}: ${said}`, error.detail || "");
  } finally {
    if (wait) Activity.end(wait); // a newer shot or Stop waiting ended the wait
  }
}

/** The header, the address and the queue follow the shot just opened. */
function arrive() {
  S.opened = { ticket: S.ticket, event: S.event, shot: S.shot };
  busy(null);
  $("cursor").hidden ||= !S.meta;
  for (const id of ["save", "revert", "show-versions"]) $(id).disabled = !S.meta;
  if (Silver.context && Silver.shot !== S.shot) Silver.context = false;
  if (Lists.context && Lists.shot !== S.shot) Lists.context = false;
  if (S.api >= 11) Lists.show();
  else showSilverMode();
  history.replaceState(null, "", Lists.context
    ? listAddress(Lists.name, S.shot, S.event)
    : Silver.context
    ? `#silver/${Silver.batch}/${S.shot}/${S.event}`
    : `#${S.event}${S.shot == null ? "" : `/${S.shot}`}`);
  $("shot").value = S.shot ?? "";
  $("shot-now").textContent = S.shot == null ? "" : `Shot ${S.shot}`;
  showHeader();
  if (S.api >= 7 && S.api < 11) { // older servers: what a restart adds
    const cameras = S.event === "detachment" && S.api < 9;
    $("stale").hidden = false;
    $("stale").textContent = cameras ? "Restart the server for detachment cameras" : Activity.staleText(S.api);
  }
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
  if (next == null || next === S.shot) return;
  // Quiet: the bar does not show it; a failed build says so there (api 11), muted.
  api(`/api/shot?event=${enc(S.event)}&shot=${next}`, { quiet: true }).catch((error) => {
    if (S.api >= 11) Activity.note("prefetch", ERROR_COPY.prefetch({ shot: next, message: error.message }));
  });
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
  if (ws()) Views.fetchAdded(); // an unchanged window asks nothing
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
  if (S.api < 7 && S.label.iscrowd) {
    return say("Restart the server before saving individual/span annotations", true);
  }
  if (S.api < 8 && [S.label, S.meta.source, S.meta.saved].some(hasOverlaps)) {
    return say("Restart the server before saving overlapping annotations", true);
  }
  const [event, shot, key, ticket, label] = [S.event, S.shot, draftKey(S.shot), S.ticket, S.label];
  S.saving = { event, shot, label };
  const name = S.api >= 2 ? { name: S.name || null } : {};
  const resolution = S.api >= 7
    ? { iscrowd: label.iscrowd || label.intervals.map(() => null) } : {};
  const overlap = S.api >= 8 ? { overlap_edit: true } : {};
  if (independent() && S.api < 10) {
    S.saving = false;
    return say("Restart the server before saving detachment exposure flags", true);
  }
  const exposure = S.api >= 10
    ? { suggestions_shown: S.suggestionsShown, prefilled: S.prefilled } : {};
  try {
    const response = await api("/api/label", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ event, shot, ...label, ...resolution, ...overlap, ...name, ...exposure }),
    });
    const body = await response.json();
    const saved = S.savedRows.get(event) || new Map();
    saved.set(shot, { count: ++S.saveCount, row: body.row });
    S.savedRows.set(event, saved);
    try {
      const pendingDraft = JSON.parse(stored(key));
      const pendingLabel = pendingDraft && normalise(pendingDraft.window, pendingDraft.intervals,
        undefined, pendingDraft.iscrowd);
      if (same(pendingLabel, label)) store(key, null);
    } catch {
      // A malformed draft does not prevent the completed save from being shown.
    }
    if (!pendingNavigation() && S.opened?.event === event && S.opened.shot === shot && S.meta) {
      Object.assign(S.meta, { saved: body.saved, last_save: body.last_save, reviewers: body.reviewers });
      if (same(S.label, label)) {
        S.label = body.saved;
        render();
      } else {
        touch(); // a Restore may have been clean against the previous save
      }
    }
    if (S.event === event) {
      const before = S.queue.find((row) => row.shot === shot)?.state;
      // a list's off-roster shot stays marked as one (api 11)
      const keep = (row) => (row.extra && S.api >= 11 ? { ...body.row, extra: true } : body.row);
      S.allQueue = S.allQueue.map((row) => (row.shot === shot ? keep(row) : row));
      filterQueue();
      const chip = locate(event)?.chip;
      if (chip && before === "unreviewed") chip.n_reviewed += 1;
      renderQueue();
      showHeader();
    }
    if (Silver.context && Silver.shot === shot) $("silver-next-event").hidden = false;
    if (Lists.context && Lists.shot === shot) $("silver-next-event").hidden = false;
    if (next && ticket === S.ticket) {
      if (Lists.context) await Lists.nextEvent(); // its own catch: the save stands either way
      else if (Silver.context) await nextSilverEvent();
      else await go(1);
    }
  } catch (error) {
    say(`${shot} not saved: ${error.message}`, true);
  } finally {
    S.saving = false;
  }
}

// -- drawing

function buildRows() {
  $("rows").replaceChildren(...S.meta.rows.map(row => {
    const canvas = document.createElement("canvas");
    const metadata = S.meta.params?.panel_metadata?.[row.name];
    canvas.title = [metadata?.caveat, Overlays.title(row)].filter(Boolean).join(" · ");
    return canvas;
  }));
  sizeCanvases();
}

/** Each row at its own height, or taller when the rows would not fill the view. */
function sizeRows() {
  if (!S.meta) return;
  const base = S.meta.rows.map((row) => (isBlind() && S.meta.params?.panel_metadata?.[row.name]?.indicator
    ? 0 : row.kind === "image" ? IMAGE_H : TRACE_H));
  const room = $("top").clientHeight - $("axis-row").offsetHeight - $("video-panel").offsetHeight;
  if (ws()) {
    // The view's heights (layoutRows): store rows at theirs or today's, hidden ones at 0, added
    // rows after them; only the rows not fixed scale up to fill the view.
    const plan = layoutRows().map((item, i) => ({ fixed: item.fixed,
      h: item.hidden || base[i] === 0 ? 0 : item.h ?? base[i] ?? TRACE_H }));
    const fixed = plan.reduce((sum, item) => sum + (item.fixed ? item.h : 0), 0);
    const flexible = plan.reduce((sum, item) => sum + (item.fixed ? 0 : item.h), 0);
    const grow = Math.max(1, (room - fixed) / (flexible || 1));
    [...$("rows").children].forEach((canvas, i) => {
      canvas.style.height = `${Math.floor(plan[i].fixed ? plan[i].h : plan[i].h * grow)}px`;
    });
    return;
  }
  const scale = Math.max(1, room / (base.reduce((a, b) => a + b, 0) || 1));
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
  renderResolution();
  if (S.frame || !S.meta) return;
  S.frame = requestAnimationFrame(() => {
    S.frame = 0;
    if (!S.meta) return; // the silver page closed the shot before this frame
    drawRows();
    drawAxis();
    drawTrack($("source-track"), isBlind() ? null : S.meta.source, false);
    drawProducerStrips();
    if (S.meta.parent) drawTrack($("parent-track"), S.meta.parent.label, false);
    drawTrack($("label-track"), S.label, true, S.api >= 8 ? 0 : undefined);
    if (S.api >= 8) drawTrack($("crowd-track"), S.label, true, 1);
    showVideoCursor();
    if (typeof Dock !== "undefined") Dock.placeTab();
    if (typeof Views !== "undefined") Views.placeAddBox();
    if (ws()) {
      drawDock();
      placePinTabs();
      Views.placeRowChip();
      drawMargins();
      if (!Slice.inRows && !S.drag) Slice.showHeld(); // where a zoom or pan moved the held time
    }
  });
}

const categoryColour = (c, event = S.event, categories = S.categories) =>
  (event === "detachment" && categories[c] === "detached" ? "#0072b2" :
  event === "detachment" && categories[c] === "uncertain" ? CATEGORY_COLOURS[5] :
  c === 1 ? T.label : CATEGORY_COLOURS[c] || T.muted);

function drawRows() {
  const canvases = $("rows").children;
  const workspace = ws();
  S.meta.rows.forEach((row, i) => {
    const canvas = canvases[i];
    const [w, h] = [canvas.clientWidth, canvas.clientHeight];
    const g = context(canvas);
    if (isBlind() && S.meta.params?.panel_metadata?.[row.name]?.indicator) return;
    if (workspace && canvas.classList.contains("view-hidden")) return;
    const values = S.data && S.data.rows[i];
    const range = row.kind === "image" ? imageRange(row) : values && traceRange(row, values);
    g.save();
    g.beginPath();
    g.rect(GUTTER, 0, w - GUTTER - right(), h);
    g.clip();
    if (row.kind === "image") drawImage(g, row, i, h);
    if (row.kind === "image") drawTokeye(g, row, w, h, true);
    if (row.kind === "image") drawMasks(g, row, h);
    if (values && row.kind === "trace") drawTrace(g, row, values, range, w, h);
    drawUngated(g, row, h);
    drawOverlay(g, w, h);
    if (row.kind === "image") drawTokeye(g, row, w, h, false);
    if (workspace) drawPins(g, h);
    g.restore();
    drawGutter(g, row, range, h);
    if (row.kind === "image" && !row.modes) drawColourBar(g, row, w, h);
    const note = S.event === "detachment" && indicatorNote(row, S.video?.time);
    if (note) {
      g.font = FONT; g.textAlign = "left";
      const width = w - GUTTER - right() - 10;
      g.fillStyle = T.panel;
      g.fillRect(GUTTER + 3, 1, Math.min(width, g.measureText(note).width + 4), 15);
      g.fillStyle = T.muted;
      g.fillText(note, GUTTER + 5, 13, width);
    }
    g.fillStyle = T.rule;
    g.fillRect(0, h - 1, w, 1);
  });
  if (workspace) drawAddedRows();
}

/** The colours of `S.lut` as a 1 x 256 canvas, z_hi at the top; remade when S.lut changes. */
const colourBar = { lut: null, canvas: null };

/** An image row's colour bar (a modes row keeps its key in the gutter instead): a 6 px strip of
 * the row's colours, contrast floor included, in the row's 12 px right edge (RIGHT), so the plot
 * keeps its width; z_hi with z_units at the top right corner of the plot and z_lo at the bottom
 * (", log" on a log row). The cut margin, when open, lies over the two labels and names the
 * scale itself; the strip stays in view. Nothing on a row without a usable z_lo < z_hi. */
function drawColourBar(g, row, w, h) {
  const scale = Slice.zScale(row);
  if (!scale || h < 12) return;
  if (colourBar.lut !== S.lut) {
    const canvas = document.createElement("canvas");
    [canvas.width, canvas.height] = [1, 256];
    const c = canvas.getContext("2d");
    const image = c.createImageData(1, 256);
    const pixels = new Uint32Array(image.data.buffer);
    for (let y = 0; y < 256; y++) pixels[y] = S.lut[255 - y];
    c.putImageData(image, 0, 0);
    Object.assign(colourBar, { lut: S.lut, canvas });
  }
  const [x, top, height] = [w - RIGHT + 3, 1, h - 3];
  g.save();
  g.imageSmoothingEnabled = false;
  g.drawImage(colourBar.canvas, 0, 0, 1, 256, x, top, 6, height);
  g.strokeStyle = T.rule;
  g.lineWidth = 1;
  g.strokeRect(x - 0.5, top - 0.5, 7, height + 1);
  if (h >= 36) {
    const units = scale.units ? ` ${scale.units}` : "";
    const edge = w - RIGHT - 2;
    g.font = FONT;
    g.textAlign = "right";
    for (const [text, y] of [[`${Slice.number(scale.hi)}${units}`, 12],
      [`${Slice.number(scale.lo)}${units}${scale.log ? ", log" : ""}`, h - 5]]) {
      const width = Math.min(g.measureText(text).width, w - GUTTER - RIGHT - 8);
      g.globalAlpha = 0.85;
      g.fillStyle = T.panel;
      g.fillRect(edge - width - 3, y - 10, width + 5, 13);
      g.globalAlpha = 1;
      g.fillStyle = T.ink;
      g.fillText(text, edge, y, width);
    }
  }
  g.restore();
}

/** The value under the pointer on an image row with a z scale, for the cursor's label:
 * " · 8.12 kHz · −12.3 dB", the byte drawn there mapped back through z_lo..z_hi (log-aware);
 * "" off such a row. */
function cursorValue(event) {
  const canvas = event?.target?.closest?.("#rows > canvas");
  const i = canvas ? Array.prototype.indexOf.call($("rows").children, canvas) : -1;
  const row = S.meta?.rows?.[i];
  const scale = row?.kind === "image" ? Slice.zScale(row) : null;
  if (!scale) return "";
  const r = canvas.getBoundingClientRect();
  const x = event.clientX - r.left;
  if (x < GUTTER || x > r.width - right()) return "";
  const t = timeAt(event.clientX);
  const data = Slice.storeData(i, t);
  const j = data ? Slice.bucket(data, t) : -1;
  if (j < 0) return "";
  // drawImage puts bins [first, stop) over the row's h - 1 px, the highest at the top.
  const [first, stop] = bandBins(row);
  const f = clamp((event.clientY - r.top) / Math.max(1, r.height - 1), 0, 1 - 1e-9);
  const bin = stop - 1 - Math.floor(f * (stop - first));
  const y = `${Slice.number(row.y0 + bin * row.dy)}${row.y_units ? ` ${row.y_units}` : ""}`;
  return ` · ${y} · ${Slice.zText(scale, data.rows[i][bin * data.n + j])}`;
}

/** Afrac lower's bins drawn from the ungated ratio, where the producer marked the bin invalid
 * (stream E's per-bin `metadata.ungated`): hatched, like the invalid votes on the strips, so the
 * second line reads as a weak aid. The reason is in the "Afrac method" readout. */
function drawUngated(g, row, h) {
  const metadata = S.meta.params?.panel_metadata?.[row.name];
  const flags = metadata?.indicator === "afrac" ? metadata.ungated : null;
  if (!flags?.length || !metadata.bin_start_ms) return;
  for (let i = 0; i < flags.length; i++) {
    if (!flags[i]) continue;
    let k = i; // a run of ungated bins is one hatch, so its marks run on across bin edges
    while (flags[k + 1] && metadata.bin_start_ms[k + 1] <= metadata.bin_end_ms[k]) k++;
    const [a, b] = [px(metadata.bin_start_ms[i]), px(metadata.bin_end_ms[k])];
    hatch(g, a, 0, b - a, h - 1);
    i = k;
  }
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

/** TokEye's lines over the whole shot: `--tokeye` wherever TokEye lights two of the four
 * chords, AE or not. A picture: a click never lands on it. Drawn in two parts, `inside` the
 * label's window under the pseudo-mask and the label's lines, and outside it over the veil,
 * which would otherwise wash it out where the pseudo-mask never reaches. */
function drawTokeye(g, row, w, h, inside) {
  const t = Overlays.layerFor(row) ?? S.tokeye; // a row's own layer (api 11), else AE's
  if (!t || !S.showMasks) return;
  const [lo, hi] = imageRange(row);
  const y = (f) => t.n_y - ((f - t.y0_khz) / t.dy_khz + 0.5); // the layer's own pixel rows
  const [x0, x1] = [px(t.grid.t0_ms), px(t.grid.t0_ms + t.grid.n * t.grid.dt_ms)];
  const [a, b] = S.label.window.map(px);
  g.save();
  g.beginPath();
  if (inside) g.rect(a, 0, b - a, h);
  else {
    g.rect(0, 0, a, h);
    g.rect(b, 0, w - b, h);
  }
  g.clip();
  g.imageSmoothingEnabled = false;
  g.drawImage(tokeyeBitmap(t), 0, y(hi), t.grid.n, y(lo) - y(hi), x0, 0, x1 - x0, h - 1);
  g.restore();
}

/** The layer as a canvas, a pixel per column and bin, top bin first; built once per colour. */
function tokeyeBitmap(t) {
  if (t.bitmap?.colour === T.tokeye) return t.bitmap.canvas;
  const bytes = Uint8Array.from(atob(t.bits), (c) => c.charCodeAt(0));
  const [n, stride] = [t.grid.n, Math.ceil(t.grid.n / 8)];
  const lit = (j, k) => (bytes[j * stride + (k >> 3)] >> (7 - (k & 7))) & 1;
  const canvas = document.createElement("canvas");
  [canvas.width, canvas.height] = [n, t.n_y];
  const g = canvas.getContext("2d");
  g.fillStyle = T.tokeye;
  for (let j = 0; j < t.n_y; j++) {
    for (let k = 0; k < n; k++) {
      if (!lit(j, k)) continue;
      let end = k + 1;
      while (end < n && lit(j, end)) end++;
      g.fillRect(k, t.n_y - 1 - j, end - k, 1);
      k = end;
    }
  }
  t.bitmap = { colour: T.tokeye, canvas };
  return canvas;
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
  const colours = row.modes ? modeLut(row.modes, S.lo) : S.lut;
  for (let y = 0; y < row.n_y; y++) {
    const from = (row.n_y - 1 - y) * n; // the store's row 0 is the lowest bin
    for (let x = 0; x < n; x++) pixels[y * n + x] = colours[values[from + x]];
  }
  g.putImageData(image, 0, 0);
  data.bitmaps.set(i, canvas);
  return canvas;
}

/** A trace row's y range over the columns on screen, its dashed lines included. */
function traceGuides(row) {
  const metadata = S.meta.params?.panel_metadata?.[row.name];
  return metadata?.indicator === "prad" ? [...new Set([...row.hlines, 1])] : row.hlines;
}

/** Report stored Afrac methods verbatim; never invent their calibration recipe. */
function indicatorNote(row, time) {
  const metadata = S.meta.params?.panel_metadata?.[row.name];
  if (metadata?.quantity === "aux_jsat_peak") {
    const i = metadata.bin_start_ms?.findIndex((start, i) => start <= time && time < metadata.bin_end_ms[i]) ?? -1;
    return `Jsat peak probe: ${metadata.probe_id?.[i] ?? "unavailable at cursor"} · switching is not rollover`;
  }
  if (metadata?.indicator === "prad") {
    const values = S.data?.rows[S.meta.rows.indexOf(row)];
    if (!values) return "";
    const step = (S.data.t1 - S.data.t0) / S.data.n;
    const exceeds = Array.from(values).some((value, i) => {
      const j = i % S.data.n, t0 = S.data.t0 + j * step;
      return Number.isFinite(value) && value > 1 && t0 < S.view[1] && t0 + step > S.view[0];
    });
    return exceeds ? "f_div > 1: check heating-power denominator and radiation estimate" : "";
  }
  if (metadata?.indicator !== "afrac") return "";
  const i = metadata.bin_start_ms?.findIndex((start, i) => start <= time && time < metadata.bin_end_ms[i]) ?? -1;
  return `Afrac method: ${metadata.afrac_method?.[i] || "not recorded at cursor"}`;
}

function traceRange(row, values, data = S.data) {
  const n = data.n;
  const step = (data.t1 - data.t0) / n;
  const j0 = clamp(Math.floor((S.view[0] - data.t0) / step), 0, n);
  const j1 = clamp(Math.ceil((S.view[1] - data.t0) / step), 0, n);
  let [lo, hi] = [Math.min(Infinity, ...traceGuides(row)), Math.max(-Infinity, ...traceGuides(row))];
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

/** A trace row's channels on `data`'s grid (S.data's, or an added row's own). */
function drawTrace(g, row, values, [lo, hi], w, h, data = S.data) {
  const [n, channels] = [data.n, row.n_channels];
  const step = (data.t1 - data.t0) / n;
  const y = (v) => ((hi - v) / (hi - lo)) * (h - 1);
  g.lineWidth = 1;
  g.strokeStyle = T.muted;
  g.setLineDash([4, 3]);
  for (const v of traceGuides(row)) {
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
      const x = px(data.t0 + (j + 0.5) * step);
      if (pen) g.lineTo(x, y(low));
      else g.moveTo(x, y(low));
      g.lineTo(x, y(high));
      if (S.meta.params?.panel_metadata?.[row.name]?.indicator === "prad" && high > 1) {
        g.fillStyle = "#d55e00";
        g.fillRect(x - 2, y(high) - 2, 4, 4);
      }
      pen = true;
    }
    g.stroke();
  }
  if (row.legend.length < 2) return;
  g.font = FONT;
  g.textAlign = "right";
  let x = w - right() - 6;
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
    g.fillRect(px(a), S.api >= 8 && S.label.iscrowd?.[i] === 1 ? 4 : 0, px(b) - px(a), 3);
    g.fillRect(px(a) - width / 2, 0, width, h);
    g.fillRect(px(b) - width / 2, 0, width, h);
  });
  drawMarkers(g, w, h);
}

/** Each marker set as thin ticks over the row, one colour per label (T hides them). A dense
 * set (ELM onsets, dozens a shot) is a short tick on the row's lower edge, not a line down it. */
function drawMarkers(g, w, h) {
  if (!S.showMarkers || !S.meta?.markers) return;
  g.save();
  S.meta.markers.forEach((set, i) => {
    const edge = EDGE_MARKERS.has(set.label);
    g.globalAlpha = edge ? 0.55 : 0.75;
    g.fillStyle = markerColour(i);
    const [top, height] = edge ? [Math.max(0, h - EDGE_TICK), Math.min(h, EDGE_TICK)] : [0, h];
    for (const t of set.t_ms) {
      if (t < S.view[0] || t > S.view[1]) continue;
      g.fillRect(Math.round(px(t)) - 0.5, top, 1, height);
    }
  });
  g.restore();
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
  if (row.modes) modeKey(g, row.modes, width, y, text);
  // The row's TokEye chip (api 11) sits under that text; the y ticks keep clear of it too.
  const slot = Overlays.slot(row, Math.max(6, ...text.map((r) => r[2])) + 4, g.canvas);
  if (slot) text.push(slot);
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

/** A modes row's key under its units: each n on a swatch of its colour, wrapped to
 * the gutter; the swatches join `text`, so the y ticks keep clear of them. */
function modeKey(g, modes, width, top, text) {
  let [x, y] = [8, top - 10];
  g.save();
  modes.n.forEach((n, i) => {
    const label = String(n).replace("-", "\u2212");
    const w = g.measureText(label).width + 4;
    if (x > 8 && x + w > 8 + width) [x, y] = [8, y + GUTTER_LINE];
    g.fillStyle = modes.colours[i];
    g.fillRect(x, y, w, 12);
    g.fillStyle = "#000";
    g.fillText(label, x + 2, y + 10);
    text.push([x + w, y, y + 12]);
    x += w + 2;
  });
  g.restore();
}

function drawAxis() {
  const canvas = $("axis-row");
  const g = context(canvas);
  g.font = FONT;
  g.fillStyle = T.muted;
  g.textAlign = "right"; // at the gutter's right end: its left holds the Add row box (R2)
  g.fillText("ms", GUTTER - 4, 16);
  g.textAlign = "center";
  for (const [t, text] of ticks(S.view[0], S.view[1], plotWidth() / 90)) {
    const x = px(t);
    if (x < GUTTER || x > canvas.clientWidth - right()) continue;
    g.fillRect(x, 0, 1, 4);
    g.fillText(text, x, 16);
  }
}

/** A label as a track: its window lit, its spans filled; the editable one also shows
 * the window's edges in its foot strip and, along its top, where it leaves the source. */
function drawTrack(canvas, label, editable, lane) {
  const g = context(canvas);
  const [w, h] = [canvas.clientWidth, canvas.clientHeight];
  if (!label) {
    if (editable || !S.meta.onsets?.length) return;
    label = { window: S.meta.t_range, intervals: [] };
  }
  g.save();
  g.beginPath();
  g.rect(GUTTER, 0, w - GUTTER - right(), h);
  g.clip();
  const [lo, hi] = label.window;
  const [top, foot] = editable ? [5, HANDLE_BAND + 2] : [4, 4];
  g.fillStyle = T.panel;
  g.fillRect(px(lo), 0, px(hi) - px(lo), h);
  if (editable && S.meta.allowed) {
    // Time the parent does not mark: veiled and hatched, nothing can be drawn there.
    g.fillStyle = T.veil;
    for (const [a, b] of gaps(label.window, S.meta.allowed)) {
      g.fillRect(px(a), 0, px(b) - px(a), h);
      hatch(g, px(a), 0, px(b) - px(a), h);
    }
  }
  const split = !editable && hasOverlaps(label);
  label.intervals.forEach(([a, b, c], i) => {
    const flag = label.iscrowd?.[i];
    if (lane != null && (flag === 1 ? 1 : 0) !== lane) return;
    const spanHeight = (h - top - foot) / (split ? 2 : 1);
    const spanTop = top + (split && flag === 1 ? spanHeight : 0);
    g.globalAlpha = editable ? 1 : 0.55;
    g.fillStyle = categoryColour(c);
    g.fillRect(px(a), spanTop, Math.max(1, px(b) - px(a)), spanHeight);
    g.globalAlpha = 1;
    if (flag === 1) hatch(g, px(a), spanTop, px(b) - px(a), spanHeight);
    if (flag != null) {
      g.fillStyle = T.ink;
      g.globalAlpha = 0.5;
      g.fillRect(px(a), spanTop, 1, spanHeight);
      g.globalAlpha = 1;
    }
    if (editable && i === S.selected) {
      g.strokeStyle = T.ink;
      g.lineWidth = 2;
      g.strokeRect(px(a), spanTop, px(b) - px(a), spanHeight);
    }
    if (editable && px(b) - px(a) > 45) {
      const width = px(b) - px(a) - 8;
      const text = `${S.categories[c]} · ${flag === 1 ? "Span" : flag === 0 ? "Individual" : "Unspecified"}`;
      g.save();
      g.font = FONT;
      const title = capped(g, wrapped(g, text, width), width, 1)[0];
      const x = px(a) + 4, y = spanTop + (spanHeight - 14) / 2;
      g.fillStyle = T.panel;
      g.globalAlpha = 0.9;
      g.fillRect(x - 1, y, g.measureText(title).width + 2, 14);
      g.globalAlpha = 1;
      g.fillStyle = T.ink;
      g.fillText(title, x, y + 11);
      g.restore();
    }
  });
  if (editable) {
    g.fillStyle = T.ink;
    if (!S.meta.parent) { // a subcategory's window is its parent's: no handles to move
      g.fillRect(px(lo), h - HANDLE_BAND / 2 - 0.5, px(hi) - px(lo), 1);
      for (const t of label.window) g.fillRect(px(t) - 2, h - HANDLE_BAND, 4, HANDLE_BAND);
    }
    if (!isBlind()) {
      const source = S.meta.source || { window: label.window, intervals: [] };
      g.fillStyle = T.changed;
      for (const [a, b] of diffRuns(source, label)) g.fillRect(px(a), 0, Math.max(1, px(b) - px(a)), 3);
    }
  } else {
    g.strokeStyle = T.ink;
    g.lineWidth = 1;
    for (const point of S.meta.onsets || []) {
      g.beginPath();
      g.moveTo(px(point.t_ms), 0);
      g.lineTo(px(point.t_ms), h);
      g.stroke();
    }
  }
  if (ws()) drawPins(g, h);
  g.restore();
}

/** Diagonal marks keep unresolved groups recognizable at every zoom. */
function hatch(g, x, y, width, height) {
  const left = Math.max(GUTTER, x), right = Math.min(px(S.view[1]), x + width);
  if (right <= left) return;
  g.save();
  g.beginPath();
  g.rect(left, y, right - left, height);
  g.clip();
  g.strokeStyle = T.ink;
  g.globalAlpha = 0.35;
  g.lineWidth = 1;
  g.beginPath();
  for (let t = left - height; t < right; t += 7) {
    g.moveTo(t, y + height);
    g.lineTo(t + height, y);
  }
  g.stroke();
  g.restore();
}

/** The bar of subcategory chips under the dropdown: "all" (the parent) and one per child. A
 * child opens on the shot on screen, once the parent is saved on it with something marked. */
function renderSubbar() {
  const bar = $("subbar");
  const { row } = locate(S.event) || {};
  const subs = row?.subcategories || [];
  bar.hidden = !subs.length;
  bar.replaceChildren();
  if (!subs.length) return;
  const name = row.display_name || row.event;
  const parent = S.meta?.parent;
  const saved = parent ? parent.label : S.meta?.saved;
  const ready = Boolean(saved?.intervals.length);
  const chip = (event, text, title, disabled) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sub";
    button.textContent = text;
    button.title = title;
    button.disabled = disabled;
    button.setAttribute("aria-pressed", String(event === S.event));
    button.addEventListener("click", () => {
      if (event !== S.event) openEvent(event, S.shot).catch((error) => say(error.message, true));
    });
    return button;
  };
  bar.append(chip(row.event, "all", `All of ${name}`, false));
  for (const sub of subs) {
    const title = ready ? `${sub.display_name}: label inside ${name}`
      : saved ? `${name} marks nothing on this shot` : `review ${name} first`;
    bar.append(chip(sub.event, `${sub.display_name} · ${sub.n_reviewed}`, title,
      !ready && sub.event !== S.event));
  }
}

/** The parent's read-only lane above the label lane, on a subcategory's page only. */
function showParent() {
  const parent = S.meta?.parent;
  const lane = $("parent-lane");
  const was = lane.hidden;
  lane.hidden = !parent;
  $("parent-lane-name").textContent = parent ? parent.display_name : "";
  lane.title = parent ? `${parent.display_name}: saved, read-only` : "";
  if (was !== lane.hidden) sizeCanvases();
}

/** The legend of the open shot's markers: one colour per label, T hides the ticks. */
function showMarkers() {
  const line = $("markers");
  const sets = S.meta?.markers || [];
  line.hidden = !sets.length;
  line.replaceChildren();
  sets.forEach((set, i) => {
    const key = document.createElement("span");
    key.className = "marker-key";
    key.style.setProperty("--c", markerColour(i));
    key.textContent = `${set.label} ${set.t_ms.length}`;
    if (S.meta.onset_candidates) key.title = "These onsets are also in the draft, as Individual annotations";
    line.append(key);
  });
  line.classList.toggle("off", !S.showMarkers);
}

function toggleMarkers() {
  if (!S.meta?.markers?.length) return say("This shot has no markers.");
  S.showMarkers = !S.showMarkers;
  store("labeler:markers", S.showMarkers ? null : "hidden");
  showMarkers();
  render();
}

function showHeader() {
  renderResolution();
  renderSubbar();
  showParent();
  showMarkers();
  $("detachment-help").hidden = S.event !== "detachment";
  const next = neighbour(1);
  $("next-shot").textContent = next == null || next === S.shot ? "" : `→ ${next}`;
  const row = S.queue.find((r) => r.shot === S.shot) || S.meta || {};
  const last = S.meta?.last_save;
  $("tier").textContent = row.tier || "";
  const state = isBlind() && row.state && row.state !== "unreviewed" ? "reviewed" : row.state;
  $("state").textContent = S.queueEvent !== S.event ? "the queue is still loading" : state || "";
  $("state").className = `pill ${state || ""}`;
  $("dirty").hidden = !dirty();
  showContributors();
  showOnsets();
  const reviewed = S.queue.filter((row) => row.state !== "unreviewed").length;
  $("count").textContent = `${reviewed}/${S.queue.length}`;
  showSaved(last);
}

/** #saved: "saved <when> by <name>". At api 11, when that would put the bar on more lines
 * than the shortest form does, the longest form that does not: the day or time (today's
 * save) with the name, then without it; the full text is then the title. */
function showSaved(last = S.meta?.last_save) {
  const saved = $("saved");
  const by = last?.name ? ` by ${last.name}` : "";
  const full = last ? `saved ${when(last.saved_at)}${by}` : "";
  saved.textContent = full;
  saved.title = "";
  if (!full || S.api < 11) return;
  const at = new Date(last.saved_at);
  const brief = at.toLocaleString([], new Date().toDateString() === at.toDateString()
    ? { hour: "2-digit", minute: "2-digit" } : { month: "short", day: "numeric" });
  const forms = [full, `saved ${brief}${by}`, `saved ${brief}`];
  const heights = forms.map((text) => {
    saved.textContent = text;
    return $("bar").getBoundingClientRect().height;
  });
  const fits = heights.findIndex((height) => height <= heights[heights.length - 1]);
  saved.textContent = forms[fits];
  saved.title = fits ? full : "";
}

let savedFrame = 0; // the pending fitSaved

/** Choose #saved's form again on the next frame (api 11): the bar's width changed, or an item
 * beside it did (#masks arrives after the header; #status, #view and #stale change). */
function fitSaved() {
  if (savedFrame || S.api < 11) return;
  savedFrame = requestAnimationFrame(() => {
    savedFrame = 0;
    if (S.meta) showSaved();
  });
}

/** Exact database times remain available even when two ticks share a pixel. */
function showOnsets() {
  const line = $("onsets");
  const points = S.meta?.onsets || [];
  line.hidden = !points.length;
  line.replaceChildren();
  if (!points.length) return;
  line.append("Curated onsets (times only): ");
  for (const point of points) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${point.t_ms} ms` +
      (point.ntor == null ? "" : ` · n=${point.ntor}`) +
      (point.mode_type ? ` · ${point.mode_type}` : "");
    button.title = point.raw_source || point.source;
    button.addEventListener("click", () => setView(point.t_ms - 100, point.t_ms + 100));
    line.append(button, " ");
  }
}

/** Everyone who has saved the open shot under a name, under the queue; nothing from a server without the list. */
function showContributors() {
  const line = $("contributors");
  const people = S.meta?.reviewers;
  if (!people) return line.replaceChildren();
  if (!people.length) return (line.textContent = S.meta.saved ? "" : `${S.shot}: no reviewers yet`);
  line.replaceChildren(
    `${S.shot} reviewed by `,
    ...people.flatMap((p, i) => {
      const who = Object.assign(document.createElement("b"), { textContent: p.name });
      who.title = `${p.saves} save${p.saves === 1 ? "" : "s"}, first ${when(p.first)}, last ${when(p.last)}`;
      return i ? [", ", who] : [who];
    })
  );
}

function renderQueue() {
  const nav = $("queue");
  // api 11: another list of the same length must not keep the old chips' shots
  const key = S.api >= 11 ? Lists.queueKey(S.event) : null;
  if (nav.children.length !== S.queue.length || key !== S.queueKey) {
    S.queueKey = key;
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
    const state = isBlind() && row.state !== "unreviewed" ? "reviewed" : row.state;
    const extra = S.api >= 11 && row.extra; // on the list, not on this event's roster
    const marks = [state, row.shot === S.shot && "current", stored(draftKey(row.shot)) && "dirty", extra && "extra"];
    chip.className = ["chip", ...marks.filter(Boolean)].join(" ");
    chip.title = extra ? "Not on this event's roster" : row.tier;
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
  if (ws() && event.target.parentNode === $("rows") && !event.shiftKey && !event.altKey &&
      event.clientX - event.target.getBoundingClientRect().left < GUTTER) {
    // A click on a row's name box selects the row (again: lets go); Delete removes it (R2).
    // The selection waits for the release, so a drag from the name box still pans.
    const key = Views.keyOf(event.target);
    return startDrag(event, { kind: "pan", x: event.clientX, y: event.clientY, view: S.view, canvas: event.target,
      select: key && key !== Views.selected ? key : null });
  }
  if (ws() && Views.selected) Views.selectRow(null);
  if (ws() && Slice.inMargin(event.clientX)) return; // the cut margin is not time
  if (ws() && event.altKey) return pinAt(timeAt(event.clientX));
  if (event.shiftKey) startDrag(event, { kind: "new", from: timeAt(event.clientX) });
  else startDrag(event, { kind: "pan", x: event.clientX, y: event.clientY, view: S.view, canvas: event.target });
}

function onLabelDown(event) {
  if (event.button !== 0 || !S.meta) return;
  if (ws() && event.altKey) return pinAt(timeAt(event.clientX)); // a pin, never an edit
  if (ws() && Views.selected) Views.selectRow(null); // one selection: Delete acts on the annotation
  if (S.video) { pauseVideo(); seekVideo(timeAt(event.clientX)); }
  const rect = event.currentTarget.getBoundingClientRect();
  const lane = S.api >= 8 ? Number(event.currentTarget.dataset.lane) : undefined;
  const hit = hitTest(S.label, event.clientX - rect.left, event.clientY - rect.top, rect.height, px, lane);
  if (hit.kind === "window" && S.meta.parent) hit.kind = "new"; // the parent's window is fixed
  S.selected = hit.kind === "edge" || hit.kind === "move" ? hit.index : -1;
  const crowd = lane == null ? S.crowd : lane === 0 && S.crowd == null ? null : lane;
  if (hit.kind === "new") S.crowd = crowd;
  startDrag(event, { ...hit, crowd, from: timeAt(event.clientX) });
  render();
}

function onLabelHover(event) {
  if (S.drag || !S.meta) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const lane = S.api >= 8 ? Number(event.currentTarget.dataset.lane) : undefined;
  const hit = hitTest(S.label, event.clientX - rect.left, event.clientY - rect.top, rect.height, px, lane);
  if (hit.kind === "window" && S.meta.parent) hit.kind = "new";
  const cursors = { edge: "ew-resize", window: "ew-resize", move: "grab", new: "crosshair" };
  event.currentTarget.style.cursor = cursors[hit.kind];
  const span = S.label.intervals[hit.index];
  const flag = S.label.iscrowd?.[hit.index];
  event.currentTarget.title = span
    ? `${span[0]}–${span[1]} ms · ${S.categories[span[2]]} · ${flag === 1 ? "span" : flag === 0 ? "individual" : "resolution unspecified"}`
    : `Drag to add ${lane === 1 ? "a span" : lane === 0 ? "an individual" : "an annotation"}`;
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
  const flags = spans.map((_, i) => d.base.iscrowd?.[i] ?? null);
  let moved = null;
  let flag = null;
  if (d.kind === "new") {
    moved = [Math.min(d.from, t), Math.max(d.from, t), S.category];
    flag = S.api >= 7 ? ("crowd" in d ? d.crowd : S.crowd) : null;
  }
  if (d.kind === "edge") {
    const span = spans.splice(d.index, 1)[0];
    flag = flags.splice(d.index, 1)[0];
    span[d.edge] = t;
    moved = [Math.min(span[0], span[1]), Math.max(span[0], span[1]), span[2]];
  }
  if (d.kind === "move") {
    const [a, b, c] = spans.splice(d.index, 1)[0];
    flag = flags.splice(d.index, 1)[0];
    moved = [a + t - d.from, b + t - d.from, c];
  }
  if (d.kind === "window") {
    const [lo, hi] = edges;
    edges[d.edge] = d.edge
      ? clamp(ms(t), lo + GRID_MS, lo + LONGEST_WINDOW_MS)
      : clamp(ms(t), hi - LONGEST_WINDOW_MS, hi - GRID_MS);
    // The window is the edge of what was looked at: spans are cut back to it.
    for (const span of spans) [span[0], span[1]] = [0, 1].map((k) => clamp(span[k], ...edges));
  }
  if (moved) {
    spans.push(moved); // last, so it paints over its own lane
    flags.push(flag);
  }
  const next = constrained(edges, spans, flags);
  if (!next) return;
  S.label = next;
  if (moved) selectAt((moved[0] + moved[1]) / 2, flag);
  render();
}

function endDrag(event) {
  const d = S.drag;
  S.drag = null;
  if (d?.kind === "pan" && !d.moved && event?.type === "pointerup") {
    if (d.select !== undefined) { // a click on a row's name box: the row, never an annotation
      const had = S.selected >= 0;
      Views.selectRow(d.select);
      if (had) render();
      return;
    }
    if (S.video) { pauseVideo(); return seekVideo(timeAt(event.clientX)); }
    if (ws() && d.canvas?.id === "axis-row") return axisClickPin(timeAt(event.clientX));
    if (d.canvas) return clickMask(d);
  }
  if (!d || d.kind === "pan" || same(d.base, S.label)) return;
  keepForUndo(d.base);
  touch();
}

// -- the AE pseudo-mask: TokEye's lines inside the label, one click per region

const maskSaves = new Map(); // shot -> pending save, survives navigation
const maskErrors = new Map(); // shot -> last failed save, until a successful retry

/** The open shot's pseudo-mask (api 3) and TokEye layer (api 4), if the server has them (AE only). */
async function loadMasks(ticket) {
  if (S.api >= 3 && (S.meta?.parent?.event ?? S.event) === MASK_EVENT) {
    const event = S.event, shot = S.shot;
    // Once per shot: a reload after a save conflict keeps the layer it has.
    const layer = S.api >= 4 && S.tokeye?.shot !== shot ? loadTokeye(event, shot) : null;
    try {
      await maskSaves.get(shot);
      if (ticket !== S.ticket) return;
      const body = await (await api(`/api/masks?event=${enc(event)}&shot=${shot}`)).json();
      if (ticket !== S.ticket) return;
      S.masks = body;
      Activity.clear("masks");
      const failure = maskErrors.get(shot);
      if (failure) say(failure, true);
    } catch (error) {
      // a 404: no pseudo-mask for this shot, nothing to draw; anything else says so (api 11)
      if (S.api >= 11 && ticket === S.ticket && error?.status !== 404) {
        Activity.note("masks", ERROR_COPY.mask({ message: error.message }), "error");
      }
    }
    if (ticket !== S.ticket) return;
    showMasks();
    render();
    if (!layer) return;
    const tokeye = await layer;
    if (ticket !== S.ticket) return;
    S.tokeye = tokeye;
    if (S.api >= 11) Overlays.check(); // a row the server calls done, with no layer here, says so
  }
  if (ticket !== S.ticket) return;
  showMasks();
  render();
}

/** The shot's whole-shot TokEye layer, or null. At api 11 an answer that is not a layer is
 * null too, and a failure other than a 404 goes on the done rows' chips. */
async function loadTokeye(event, shot) {
  try {
    const body = await (await api(`/api/tokeye?event=${enc(event)}&shot=${shot}`)).json();
    return S.api >= 11 && !Overlays.layer(body) ? null : body;
  } catch (error) {
    if (S.api >= 11 && error?.status !== 404 && S.event === event && S.shot === shot) {
      Overlays.mark(Overlays.named("done"), { state: "failed", reason: `the layer could not be loaded: ${error.message}`, local: true });
      Overlays.renderChips();
    }
    return null; // no layer for this shot: nothing to draw
  }
}

function showMasks() {
  const m = S.masks;
  $("masks").hidden = !m && !S.tokeye;
  if (!m) {
    if (S.tokeye) $("masks").textContent = `TokEye only${S.showMasks ? "" : " · hidden"}`;
    return;
  }
  const n = m.regions.length;
  const kept = n - m.rejected.length;
  const last = !m.stale && m.last_save;
  const by = last ? ` · saved${last.name ? ` by ${last.name}` : ""}` : "";
  const notes = [m.stale ? " · a decision on an older mask was dropped" : "", by, S.showMasks ? "" : " · hidden"];
  $("masks").textContent = `mask ${kept}/${n} kept${notes.join("")}`;
}

function toggleMasks() {
  if (S.api >= 11 && Activity.notes.has("masks")) { // the mask failed to load: M retries
    Activity.clear("masks");
    return loadMasks(S.ticket);
  }
  if (!S.masks && !S.tokeye && !Overlays.any()) {
    if (S.api >= 11) return Activity.note("layers", "This shot has no pseudo-mask or TokEye layer.");
    return say("this shot has no pseudo-mask");
  }
  S.showMasks = !S.showMasks;
  store("labeler:masks", S.showMasks ? null : "hidden");
  showMasks();
  Overlays.renderChips(); // "TokEye hidden (M)"
  render();
}

/** A click on an image row, not a drag: reject the region under it, or take that back. */
function clickMask(d) {
  if (S.meta?.parent) return; // an AE subcategory shows the mask; rejecting is a decision on AE
  if (maskSaves.has(S.shot)) return say("The mask is saving. Wait a moment.");
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

/** Save the rejected regions at once, with the reviewer's name; drawn before the answer. */
function saveMasks(m, rejected) {
  if (maskSaves.has(m.shot)) return say("The mask is saving. Wait a moment.");
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
    if (current() && S.meta && body.reviewers) {
      S.meta.reviewers = body.reviewers;
      showHeader();
    }
    const n = body.rejected.length;
    if (current()) say(`Mask saved: ${n} region${n === 1 ? "" : "s"} rejected.`);
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

function showCursor(clientX, hover = false, pointer = null) {
  const cursor = $(hover ? "hover-cursor" : "cursor");
  const axis = $("axis-row").getBoundingClientRect();
  const x = clientX - axis.left;
  cursor.hidden = !S.meta || x < GUTTER || x > axis.width - right();
  if (cursor.hidden) return;
  const viewport = $("top").getBoundingClientRect();
  const videoBottom = S.video ? $("video-panel").getBoundingClientRect().bottom : viewport.top;
  const top = Math.max(viewport.top, videoBottom, $("rows").getBoundingClientRect().top);
  const bottom = $(S.api >= 8 ? "crowd-track" : "label-track").getBoundingClientRect().bottom;
  Object.assign(cursor.style, { left: `${clientX}px`, top: `${top}px`, height: `${bottom - top}px` });
  // On an image row the label adds the value under the pointer; near the right end it turns
  // to the left of the line so it stays over the rows.
  const value = pointer ? cursorValue(pointer) : "";
  const label = $(hover ? "hover-time" : "cursor-time");
  label.textContent = `${Math.round(timeAt(clientX))} ms${value}`;
  label.style.left = "";
  if (value && clientX + 5 + label.offsetWidth > axis.right - RIGHT) label.style.left = `${-5 - label.offsetWidth}px`;
}

// -- detachment cameras: small manifests, one lazy frame per camera

function pauseVideo() {
  if (S.video) {
    clearTimeout(S.video.timer);
    S.video.timer = null;
    S.video.playing = false;
    S.video.epoch++;
    S.video.seekSeq++;
    S.video.cards.forEach(cancelVideoFrame);
    if (Number.isFinite(S.video.time)) $("video-time").value = S.video.time;
  }
  $("video-play").textContent = "Play";
  $("video-play").setAttribute("aria-pressed", "false");
}

/** Release transaction-owned pixels without touching the last published frame. */
function cancelVideoFrame(card) {
  card.abort?.abort();
  card.seq++;
  if (card.staged?.url) URL.revokeObjectURL(card.staged.url);
  card.staged = null;
  card.pendingKey = "";
  card.figure.setAttribute("aria-busy", "false");
  if (card.img) card.img.style.visibility = "visible";
  if (card.renderedNote) card.note.textContent = card.renderedNote;
  else if (card.channel && card.note.textContent.startsWith("Loading"))
    card.note.textContent = "Frame canceled; seek to retry.";
}

function clearVideo() {
  pauseVideo();
  closeDialog($("camera-enlarged"));
  for (const card of S.video?.cards || []) {
    card.abort?.abort();
    if (card.url) URL.revokeObjectURL(card.url);
  }
  S.video = null;
  $("video-panel").hidden = true;
  $("video-cameras").replaceChildren();
  $("cursor").hidden = true;
  $("hover-cursor").hidden = true;
}

function buildVideo() {
  $("start-blank").hidden = S.event !== "detachment";
  showBlindMode();
  if (S.event !== "detachment" || !S.meta?.video || S.api < 9) return;
  const cards = [];
  for (const camera of S.meta.video.cameras) {
    const figure = document.createElement("figure");
    figure.className = "camera";
    const caption = document.createElement("figcaption");
    caption.textContent = camera.name;
    const note = document.createElement("p");
    const channel = camera.channels.find(c => c.channel === camera.default_channel) || camera.channels[0];
    const card = { camera, figure, note, channel, key: "", seq: 0 };
    figure.append(caption);
    if (!card.channel) {
      figure.classList.add("unavailable");
      note.textContent = `No frames for this camera: ${camera.reason}`;
    } else {
      const select = document.createElement("select");
      select.setAttribute("aria-label", `${camera.name} channel`);
      const views = camera.views?.length ? camera.views : camera.channels;
      for (const channel of views) {
        const option = document.createElement("option");
        option.value = channel.channel;
        option.textContent = `${channel.channel}: ${channel.view_name || "unknown view"} · ${channel.region || "unknown"}`;
        option.disabled = !camera.channels.some(c => c.channel === channel.channel);
        if (option.disabled) option.textContent += " · unavailable";
        select.append(option);
      }
      select.value = card.channel.channel;
      select.addEventListener("change", () => {
        pauseVideo();
        card.channel = camera.channels.find((c) => c.channel === Number(select.value));
        videoTimes();
        seekVideo(S.video.time);
      });
      caption.append(select);
      card.img = document.createElement("img");
      card.img.alt = `${camera.name} camera frame`;
      figure.addEventListener("click", event => {
        if (event.target === card.img) enlargeCamera(card);
      });
      figure.addEventListener("keydown", event => {
        if (event.target === card.img && ["Enter", " "].includes(event.key)) {
          event.preventDefault(); enlargeCamera(card);
        }
      });
      figure.append(card.img);
    }
    figure.append(note);
    if (card.channel) {
      const details = document.createElement("details"), summary = document.createElement("summary");
      summary.textContent = "Camera details";
      const spectral = document.createElement("small");
      spectral.textContent = [camera.name === "tangtv" ? camera.spectral_note : "",
        camera.sampling_note].filter(Boolean).join(" ");
      details.append(summary, spectral); figure.append(details);
    }
    cards.push(card);
  }
  S.video = { cards, times: [], timer: null, playing: false, epoch: 0, seekSeq: 0 };
  videoTimes();
  $("video-cameras").replaceChildren(...cards.map((card) => card.figure));
  buildProducerStrips();
  fitVideoPanel();
  const slider = $("video-time");
  [slider.min, slider.max] = S.meta.t_range;
  slider.disabled = !S.video.times.length;
  $("video-play").disabled = true;
  $("video-clock").textContent = "";
  sizeCanvases();
  seekVideo(S.video.times[0] ?? S.meta.t_range[0]);
}

/** #video-panel shows while it holds something: the cameras and their controls, unless the
 * dock has adopted them (api 11, workspace; plan D.B), then the producer strips and details,
 * which blind mode hides. */
function fitVideoPanel() {
  const panel = $("video-panel");
  const hidden = !S.video || ![...panel.children].some((child) => !child.hidden);
  if (panel.hidden !== hidden) panel.hidden = hidden;
}

function videoTimes() {
  const [lo, hi] = S.meta.t_range;
  const primary = S.video.cards.find(c => c.camera.name === "tangtv" && c.channel) ||
    S.video.cards.find(c => c.channel);
  S.video.times = [...new Set(primary?.channel.times_ms || [])]
    .filter(t => t >= lo && t <= hi).sort((a, b) => a - b);
}

async function seekVideo(t, playbackEpoch = null) {
  const v = S.video;
  if (!v || !Number.isFinite(Number(t)) || pendingNavigation()) return false;
  if (playbackEpoch == null && v.playing) pauseVideo();
  const seq = ++v.seekSeq;
  const requested = clamp(Number(t), ...S.meta.t_range);
  const time = v.times.reduce((best, frame) =>
    Math.abs(frame - requested) < Math.abs(best - requested) ? frame : best, v.times[0] ?? requested);
  const current = () => S.video === v && seq === v.seekSeq &&
    (playbackEpoch == null || (v.playing && v.epoch === playbackEpoch));
  const commit = () => {
    v.time = time;
    $("video-time").value = time;
    $("video-clock").textContent = `${time.toFixed(1)} ms`;
    $("video-play").disabled = !v.times.length;
    showVideoGeometry(time);
    drawProducerStrips();
    render();
    showVideoCursor();
  };
  const cards = v.cards.filter(c => c.channel);
  cards.forEach(cancelVideoFrame);
  const delivered = await Promise.all(cards
    .map(card => loadVideoFrame(card, time, current, playbackEpoch != null)));
  if (!current()) return false;
  if (delivered.some(frame => !frame)) {
    cards.forEach(cancelVideoFrame);
    if (Number.isFinite(v.time)) $("video-time").value = v.time;
    return false;
  }
  // No await or browser paint can occur within this publication transaction.
  for (const frame of delivered) {
    const { card, img, url, key, note } = frame;
    if (img) {
      card.img.replaceWith(img);
      card.img = img;
      if (card.url) URL.revokeObjectURL(card.url);
      card.url = url;
    }
    card.key = key;
    card.staged = null;
    card.pendingKey = "";
    card.note.textContent = card.renderedNote = note;
    card.img.style.visibility = "visible";
    card.figure.setAttribute("aria-busy", "false");
  }
  commit();
  return true;
}

function showVideoGeometry(time) {
  const geometry = S.meta?.params?.detachment_geometry;
  const line = $("video-geometry");
  const samples = geometry?.samples || [];
  const sample = samples.reduce((best, item) => !best ||
    Math.abs(item.time_ms - time) < Math.abs(best.time_ms - time) ? item : best, null);
  const nearby = sample && Math.abs(sample.time_ms - time) <= 40;
  const configuration = nearby ? sample.configuration : null;
  const names = { LSN: "LSN (lower single null)", USN: "USN (upper single null)",
    DN: "DN (double null)" };
  line.textContent = (configuration ? `Magnetic configuration: ${names[configuration]} · ` : "") +
    (nearby ? `Lower outer strike-point gate ${sample.shelf_gate ? "valid" : "invalid"}` :
      "EFIT shelf gate unavailable at this time") +
    (nearby && !configuration ? " · topology unavailable (DRSEP required)" : "");
  line.title = nearby ? `${sample.time_ms.toFixed(1)} ms: ${sample.reason || geometry.note}` :
    (geometry?.note || "No local EFIT geometry source; use uncertain when geometry is required.");
}

/** Exact half-open producer bins; an uncovered time has no inferred state. */
function producerAt(data, time) {
  const i = data?.bin_start_ms?.findIndex((start, i) => start <= time && time < data.bin_end_ms[i]) ?? -1;
  if (i < 0) return null;
  return { state: data.label_available === false ? null : data.state_lm[i],
    rule: data.label_available === false ? null : data.state_rule[i],
    tangtv_source: data.tangtv_source[i], confidence: data.confidence[i],
    votes: Object.fromEntries(Object.entries(data.votes).map(([name, vote]) =>
      [name, { vote: vote.vote[i], valid: vote.valid[i], reason: vote.reason[i] }])) };
}

const PRODUCER_LANES = [
  ["state_lm", "Producer"], ["state_rule", "Rule"],
  ["afrac", "Afrac vote"], ["prad", "f_div vote"], ["tangtv", "TangTV vote"],
];

function buildProducerStrips() {
  const data = S.meta?.params?.detachment_producer;
  const recipe = data?.recipe;
  $("detachment-machine-help").textContent = recipe?.reason ||
    [recipe?.documentation, recipe?.record && Object.keys(recipe.record).length ?
      JSON.stringify(recipe.record, null, 2) : ""].filter(Boolean).join("\n\n") ||
    "Producer interpretation record unavailable.";
  $("detachment-strips").hidden = S.blind;
  $("detachment-lanes").replaceChildren(...PRODUCER_LANES.map(([key, title]) => {
    const canvas = document.createElement("canvas");
    canvas.dataset.producerLane = key;
    canvas.setAttribute("aria-label", title);
    canvas.addEventListener("click", event => { pauseVideo(); seekVideo(timeAt(event.clientX)); });
    return canvas;
  }));
  $("detachment-strips").title = data?.source || "Producer label source unavailable";
}

function drawProducerStrips() {
  if (S.event !== "detachment" || !S.video || S.blind) return;
  const data = S.meta?.params?.detachment_producer;
  const canvases = $("detachment-lanes").children;
  PRODUCER_LANES.forEach(([key, title], lane) => {
    const canvas = canvases[lane];
    if (!canvas) return;
    const g = context(canvas), h = canvas.clientHeight, w = canvas.clientWidth;
    g.font = FONT; g.fillStyle = T.muted;
    g.fillText(title, 8, h / 2 + 4);
    g.save(); g.beginPath(); g.rect(GUTTER, 0, w - GUTTER - right(), h); g.clip();
    (data?.bin_start_ms || []).forEach((start, i) => {
      const a = px(start), b = px(data.bin_end_ms[i]);
      const vote = data.votes[key];
      const code = vote ? vote.vote[i] : data[key][i];
      if (code === 0 && !vote) return;
      g.fillStyle = code > 0 ? categoryColour(code) : T.unreviewed;
      g.fillRect(a, 3, b - a, h - 6);
      if (vote && !vote.valid[i]) hatch(g, a, 3, b - a, h - 6);
    });
    if (Number.isFinite(S.video.time)) {
      g.fillStyle = T.ink; g.fillRect(px(S.video.time), 0, 1, h);
    }
    g.restore();
  });
  const at = producerAt(data, S.video.time);
  const stateName = c => c === null ? "label not published" :
    c === 0 ? "unassessed" : S.categories[c] || "unavailable";
  const voteName = v => !v.valid ? `invalid (${v.reason || "reason unavailable"})` :
    v.vote === -1 ? `abstains (${v.reason || "producer evidence gates; see stored recipe"})` : stateName(v.vote);
  $("detachment-reading").textContent = at ?
    `At ${S.video.time.toFixed(1)} ms: producer ${stateName(at.state)} · rule ${stateName(at.rule)} · ` +
    Object.entries(at.votes).map(([name, vote]) => `${name === "prad" ? "f_div" : name}: ${voteName(vote)}`).join(" · ") +
    (at.tangtv_source === "surrogate" ? " · TangTV source: surrogate regression (model estimate)" :
      at.tangtv_source === "inversion" ? " · TangTV source: tomographic inversion" : "") :
    data?.reason || "No producer bin at this time (unassessed)";
}

function loadVideoFrame(card, time, current, playback) {
  const { channel, times_ms: times } = card.channel;
  let right = times.findIndex((t) => t >= time);
  if (right < 0) right = times.length - 1;
  if (right && Math.abs(time - times[right - 1]) <= Math.abs(times[right] - time)) right--;
  const key = `${channel}/${right}`;
  const outside = time < times[0] || time > times.at(-1);
  const view = `${card.channel.view_name} · ${card.channel.region}`;
  const note = `${times[right].toFixed(1)} ms · ${view} · nearest corpus frame${outside ? " · outside camera coverage" : ""}`;
  if (card.key === key) {
    return Promise.resolve({ card, key, note });
  }
  card.abort = new AbortController();
  const seq = ++card.seq, ticket = S.ticket;
  card.pendingKey = key;
  card.figure.setAttribute("aria-busy", "true");
  if (!playback) {
    card.img.style.visibility = "hidden";
    card.note.textContent = `Loading ${view} at ${times[right].toFixed(1)} ms…`;
  }
  const query = new URLSearchParams({ event: S.event, shot: S.shot,
    camera: card.camera.name, channel, t_ms: times[right] });
  card.pending = (async () => {
    let url;
    try {
      const response = await api(`/api/frame?${query}`, { signal: card.abort.signal });
      url = URL.createObjectURL(await response.blob());
      const img = new Image();
      img.src = url;
      await img.decode();
      if (ticket !== S.ticket || seq !== card.seq || !current()) return false;
      const header = response.headers.get("X-Frame-Time-Ms");
      const actual = Number(header);
      if (header == null || !Number.isFinite(actual)) throw new Error("frame timestamp missing");
      img.alt = `${card.camera.name}: ${view} at ${actual.toFixed(1)} ms`;
      img.dataset.frameTime = actual;
      img.dataset.channel = channel;
      img.tabIndex = 0;
      img.title = "Click or press Enter to enlarge this frame";
      card.staged = { card, img, url, key,
        note: note.replace(times[right].toFixed(1), actual.toFixed(1)) };
      url = null;
      return card.staged;
    } catch (error) {
      if (ticket === S.ticket && seq === card.seq && error.name !== "AbortError") {
        if (!card.renderedNote) card.note.textContent = `Frame unavailable: ${error.message}. Seek to retry.`;
        say(`Frame unavailable: ${error.message}`, true);
      }
      return false;
    } finally {
      if (url) URL.revokeObjectURL(url);
      if (seq === card.seq && !card.staged) {
        card.pendingKey = "";
        card.figure.setAttribute("aria-busy", "false");
        if (card.renderedNote) card.note.textContent = card.renderedNote;
        card.img.style.visibility = "visible";
      }
    }
  })();
  return card.pending;
}

function showVideoCursor() {
  if (!S.video || !S.meta) return;
  const axis = $("axis-row").getBoundingClientRect();
  showCursor(axis.left + px(S.video.time));
}

function enlargeCamera(card) {
  if (!card.url || card.figure.getAttribute("aria-busy") === "true") return;
  pauseVideo();
  $("camera-enlarged-image").src = card.img.src;
  $("camera-enlarged-image").alt = card.img.alt;
  $("camera-enlarged-caption").textContent = card.note.textContent;
  $("camera-enlarged").showModal();
}

function showBlindMode() {
  const active = S.event === "detachment" && S.blind;
  $("blind-control").hidden = S.event !== "detachment";
  $("blind-mode").checked = S.blind;
  for (const id of ["source-lane", "detachment-strips", "detachment-details", "detachment-machine-help"])
    $(id).hidden = active;
  $("revert").disabled = active || !S.meta;
  $("revert").textContent = independent() ? "Use Source (1–3)" : "Revert";
  $("detachment-reading").textContent = active ? "" : $("detachment-reading").textContent;
  for (const [i, canvas] of Array.from($("rows").children).entries())
    canvas.hidden = active && Boolean(S.meta?.params?.panel_metadata?.[S.meta.rows[i]?.name]?.indicator);
  if (active) for (const canvas of [$("source-track"), ...$("detachment-lanes").children]) {
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }
  fitVideoPanel();
}

async function playVideo() {
  if (!S.video || !S.video.times.length || !Number.isFinite(S.video.time)) return;
  if (S.video.playing) return pauseVideo();
  const v = S.video;
  v.playing = true;
  const epoch = ++v.epoch;
  $("video-play").textContent = "Pause";
  $("video-play").setAttribute("aria-pressed", "true");
  const active = () => S.video === v && v.playing && v.epoch === epoch;
  const ready = await seekVideo(v.time >= v.times.at(-1) ? v.times[0] : v.time, epoch);
  if (!active()) return;
  if (!ready) return pauseVideo();
  const step = () => {
    if (!active() || pendingNavigation()) return;
    const next = v.times.find((t) => t > v.time + 0.001);
    if (next == null) return pauseVideo();
    const delay = Math.max(50, next - v.time);
    v.timer = setTimeout(async () => {
      const ready = await seekVideo(next, epoch);
      if (!active()) return;
      if (!ready) return pauseVideo();
      step();
    }, delay);
  };
  step();
}

function toggleKeys() {
  const dialog = $("keys");
  if (dialog.open) closeDialog(dialog);
  else dialog.showModal();
}

// -- who is reviewing (api 5)

/** This tab's picked name: asked again in a new tab or window, not on a reload. */
function picked() {
  try {
    return sessionStorage.getItem("labeler:who");
  } catch {
    return null;
  }
}

function showName() {
  $("reviewer").textContent = S.name || "Your name";
}

/** The list of names, centred over the page; the first time, a name must be picked. */
async function chooseName() {
  const dialog = $("who");
  $("who-error").textContent = "";
  if (!dialog.open) dialog.showModal();
  try {
    listNames((await (await api("/api/names")).json()).names, S.name);
  } catch (error) {
    listNames(S.name ? [S.name] : [], S.name);
    $("who-error").textContent = error.message;
  }
  ($("who-names").value ? $("who-continue") : $("who-new")).focus();
}

function listNames(names, pick) {
  const list = $("who-names");
  list.replaceChildren(...names.map((name) => new Option(name, name, false, name === pick)));
  list.size = clamp(names.length, 3, 8);
  list.hidden = names.length === 0;
  $("who-empty").hidden = names.length > 0;
  $("who-continue").disabled = !list.value;
}

async function addName() {
  const name = $("who-new").value.trim();
  if (!name) return $("who-new").focus();
  $("who-error").textContent = "";
  try {
    const response = await api("/api/names", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const body = await response.json();
    listNames(body.names, body.name);
    $("who-new").value = "";
    $("who-continue").focus();
  } catch (error) {
    $("who-error").textContent = error.message;
  }
}

function pickName() {
  const name = $("who-names").value;
  if (!name) return;
  S.name = name;
  S.picked = true;
  store("labeler:name", name);
  try {
    sessionStorage.setItem("labeler:who", name);
  } catch {
    // storage refused (a private window): the list then asks again on a reload
  }
  showName();
  closeDialog($("who"));
}

function wireNames() {
  const dialog = $("who");
  // No way past the first list without a name: Escape is refused, and a close
  // forced through anyway (Chrome allows a second Escape) opens it again.
  dialog.addEventListener("cancel", (event) => S.picked || event.preventDefault());
  dialog.addEventListener("close", () => S.picked || chooseName());
  $("reviewer").addEventListener("click", chooseName);
  $("who-names").addEventListener("change", () => {
    $("who-continue").disabled = !$("who-names").value;
  });
  $("who-names").addEventListener("dblclick", pickName);
  $("who-names").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      pickName();
    }
  });
  $("who-new").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addName();
    }
  });
  $("who-add").addEventListener("click", addName);
  $("who-continue").addEventListener("click", pickName);
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
  const changes = versionChanges(S.versions, isBlind() ? null : S.meta.source);
  const items = S.versions.map((version, i) => {
    const item = document.createElement("li");
    const who = version.name || "no name";
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
    return say("These versions are for a shot that is no longer open.");
  }
  const found = S.versions.find((version) => version.version === number);
  if (!found || !S.meta) return;
  closeDialog($("versions"));
  const label = normalise(found.window, found.intervals, known(), found.iscrowd);
  if (!label) return say(`Version ${number} does not fit this event's categories.`, true);
  if (independent()) {
    S.suggestionsShown = mergeExposure(S.suggestionsShown, found.suggestions_shown);
    S.prefilled = mergeExposure(S.prefilled, found.prefilled);
    keepExposure();
  }
  if (same(label, S.label)) return say(`Version ${number} is already the current label.`);
  const replaced = dirty() ? " It replaced an unsaved edit: Ctrl+Z brings that back." : "";
  S.selected = -1;
  edit(label.window, label.intervals, label.iscrowd);
  say(`Version ${number} restored as a draft. S saves it as a new version.${replaced}`);
}

/** The shot box's search (R2, contract §3): digits typed in #shot list up to `SHOW` shots
 * from GET /api/shots/search, gold, silver and the roster first, each with the lists that hold
 * it; ↑/↓ pick, Enter or a click opens `#<event>/<shot>`. An older server without the route
 * (404) leaves the box as it was: Enter opens the number typed. */
const ShotSearch = {
  SHOW: 8,
  DELAY_MS: 150,
  off: false, // this server has no /api/shots/search
  timer: 0,
  serial: 0, // the latest request; an older answer is dropped
  shots: [], // the matches shown: [{shot, lists, cached}]
  active: -1, // the match ↑/↓ picked, or -1

  /** The list of matches, made once at the end of the body (#bottom scrolls and would cut it). */
  list() {
    let list = $("shot-matches");
    if (!list) {
      list = Object.assign(document.createElement("ul"), { id: "shot-matches", hidden: true });
      list.setAttribute("role", "listbox");
      list.setAttribute("aria-label", "Matching shots");
      document.body.append(list);
    }
    return list;
  },

  /** The box changed: ask for its digits after a short pause. */
  typed() {
    clearTimeout(ShotSearch.timer);
    const q = $("shot").value.trim();
    if (ShotSearch.off || !/^\d{2,6}$/.test(q) || S.api < 11) return ShotSearch.show([]);
    ShotSearch.timer = setTimeout(() => ShotSearch.ask(q), ShotSearch.DELAY_MS);
  },

  async ask(q) {
    const serial = ++ShotSearch.serial;
    try {
      const response = await api(`/api/shots/search?q=${q}&event=${enc(S.event || "")}&limit=20`, { quiet: true });
      const body = await response.json();
      if (serial !== ShotSearch.serial || $("shot").value.trim() !== q) return;
      ShotSearch.show((body.shots || []).filter((row) => Number.isInteger(row?.shot)).slice(0, ShotSearch.SHOW));
    } catch (error) {
      if (serial !== ShotSearch.serial) return;
      if (error.status === 404 || error.status === 405) ShotSearch.off = true; // an older server
      ShotSearch.show([]);
    }
  },

  /** A match's lists as the menu says them. */
  tags: (lists) => (lists || []).map((name) => (name === "catalog" ? "all" : name)).join(" · "),

  /** Show `shots` under the box (none hides it). */
  show(shots) {
    ShotSearch.shots = shots;
    ShotSearch.active = -1;
    const list = ShotSearch.list();
    const input = $("shot");
    list.replaceChildren(...shots.map((row, k) => {
      const item = Object.assign(document.createElement("li"), { id: `shot-match-${k}` });
      item.setAttribute("role", "option");
      item.setAttribute("aria-selected", "false");
      item.dataset.shot = row.shot;
      item.append(Object.assign(document.createElement("b"), { textContent: String(row.shot) }),
        Object.assign(document.createElement("span"), { textContent: ShotSearch.tags(row.lists), title: ShotSearch.tags(row.lists) }),
        Object.assign(document.createElement("small"), { textContent: row.cached === false ? "fetch" : "" }));
      item.title = row.cached === false ? "Not on disk: opening it fetches its data" : "";
      item.addEventListener("pointerdown", (event) => {
        event.preventDefault(); // the box keeps focus until the shot opens
        ShotSearch.open(row.shot);
      });
      return item;
    }));
    list.hidden = !shots.length || document.activeElement !== input;
    ShotSearch.place();
    input.setAttribute("aria-expanded", String(!list.hidden));
    input.removeAttribute("aria-activedescendant");
  },

  /** Put the list against the box, on the side with more room (above, as the box sits low in
   * the label panel), never past the window's edge. */
  place() {
    const list = $("shot-matches");
    if (!list || list.hidden) return;
    const box = $("shot").getBoundingClientRect();
    const above = box.top - 8, below = innerHeight - box.bottom - 8;
    const up = above >= below;
    Object.assign(list.style, { left: `${Math.max(4, Math.min(box.left, innerWidth - list.offsetWidth - 4))}px`,
      minWidth: `${box.width}px`, maxHeight: `${Math.max(above, below)}px`,
      top: up ? "" : `${box.bottom + 2}px`, bottom: up ? `${innerHeight - box.top + 2}px` : "" });
  },

  /** ↑/↓ move the pick; the box keeps the typed digits. */
  move(step) {
    const n = ShotSearch.shots.length;
    if (!n) return;
    ShotSearch.active = ShotSearch.active < 0 ? (step > 0 ? 0 : n - 1) : (ShotSearch.active + step + n) % n;
    [...ShotSearch.list().children].forEach((item, k) => item.setAttribute("aria-selected", String(k === ShotSearch.active)));
    $("shot").setAttribute("aria-activedescendant", `shot-match-${ShotSearch.active}`);
  },

  /** Open `shot` on the event, as the address #<event>/<shot> does. */
  open(shot) {
    clearTimeout(ShotSearch.timer);
    ShotSearch.serial++;
    ShotSearch.show([]);
    $("shot").blur();
    if (Lists.byShot) return Lists.select(shot); // All: the shot on every event
    if (shot === S.shot && S.meta?.event === S.event) return; // after an event change the shot opens anew
    openShot(shot);
  },

  /** The box's keys: ↑/↓, Enter (the pick, else the number typed), Escape. */
  key(event) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      ShotSearch.move(event.key === "ArrowDown" ? 1 : -1);
    } else if (event.key === "Enter") {
      const typed = $("shot").value.trim();
      const picked = ShotSearch.shots[ShotSearch.active]?.shot;
      // A partial number opens the first match shown; six digits open as typed.
      const first = typed.length < 6 && !ShotSearch.list().hidden ? ShotSearch.shots[0]?.shot : null;
      if (picked != null) ShotSearch.open(picked);
      else if (first != null) ShotSearch.open(first);
      else if (/^\d+$/.test(typed)) ShotSearch.open(Number(typed));
    } else if (event.key === "Escape") {
      $("shot").value = (Lists.byShot ? Lists.shot : S.shot) ?? "";
      ShotSearch.show([]);
      $("shot").blur();
    }
  },
};

const KEYS = {
  Enter: () => save(true),
  s: () => save(false),
  r: revert,
  h: toggleVersions,
  m: toggleMasks,
  t: toggleMarkers,
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
    if (ws() && Views.selected) Views.selectRow(null);
    render();
  },
  // The selected annotation, else a row picked by its name box (R2); the two never coexist.
  Delete: () => deleteSelected(),
  Backspace: () => deleteSelected(),
  "Shift+ArrowLeft": () => pan(-0.1),
  "Shift+ArrowRight": () => pan(0.1),
  "-": () => zoomAt(middle(), 1.25),
  "=": () => zoomAt(middle(), 0.8),
  "+": () => zoomAt(middle(), 0.8),
  0: fit,
};

/** The workspace's keys, read by onKey only when the server has the workspace (ws()). They
 * stay out of KEYS so a page without the workspace has exactly today's keys. */
const WORKSPACE_KEYS = {
  p: () => pinAt(Slice.hover ?? middle()),
  x: () => unpinNear(Slice.hover ?? middle()),
  l: () => Slice.toggleFollow(),
  d: () => toggleDock(),
};

// The keys that work on a shot with nothing to show.
const MOVES = new Set(["j", "k", "u", "ArrowLeft", "ArrowRight"]);

function onKey(event) {
  const target = event.target;
  const typed = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  const key = event.shiftKey && typed.startsWith("Arrow") ? `Shift+${typed}` : typed;
  if (target.closest("input, select, textarea") || $("who").open) return;
  if ($("versions").open) {
    if (key === "Enter" || key === "h") {
      event.preventDefault();
      closeDialog($("versions"));
    }
    return;
  }
  if (target.closest("button") && (key === "Enter" || key === " ")) return;
  if (key === "?" || $("keys").open) return key === "?" && toggleKeys();
  if (!event.ctrlKey && !event.metaKey && !event.altKey && (S.api >= 11 ? byShotKey(key) : silverKey(key))) {
    event.preventDefault();
    return;
  }
  const mod = event.ctrlKey || event.metaKey;
  if (event.altKey || (mod && key !== "z") || (!S.meta && !MOVES.has(key))) return;
  if (mod) {
    event.preventDefault();
    return (ws() && Views.undoRow()) || undo();
  }
  if (/^[1-9]$/.test(key)) return setCategory(Number(key));
  const action = KEYS[key] || (ws() && WORKSPACE_KEYS[key]);
  if (action) {
    event.preventDefault();
    action();
  }
}

function wire() {
  const top = $("top");
  const tracks = [$("source-track"), $("label-track"), $("crowd-track")];
  top.addEventListener("pointerdown", onRowsDown);
  top.addEventListener("wheel", (event) => onWheel(event, event.target.id === "axis-row"), {
    passive: false,
  });
  for (const track of tracks) {
    track.addEventListener("wheel", (event) => onWheel(event, true), { passive: false });
  }
  for (const target of [top, ...tracks, $("pins")]) target.addEventListener("dblclick", () => {
    if (!S.meta) return;
    if (ws()) undoAxisClicks(); // a double-click on the axis takes back its two pins
    fit();
  });
  tracks[0].addEventListener("pointerdown", (event) => {
    if (event.button === 0 && S.meta) startDrag(event, { kind: "pan", x: event.clientX, view: S.view });
  });
  for (const track of tracks.slice(1)) {
    track.addEventListener("pointerdown", onLabelDown);
    track.addEventListener("pointermove", onLabelHover);
  }
  window.addEventListener("pointermove", (event) => {
    if (ws()) notePointer(event);
    if (S.drag) dragTo(event.clientX);
    // Over a row popped out into the dock, the rows' cursor follows its time (R2).
    const docked = ws() && !S.drag ? Dock.pointerTime(event) : null;
    if (docked != null) {
      const line = $(S.video ? "hover-cursor" : "cursor");
      line.classList.remove("held");
      if (docked >= S.view[0] && docked <= S.view[1]) showCursor($("axis-row").getBoundingClientRect().left + px(docked), Boolean(S.video));
      else line.hidden = true;
      return;
    }
    // Out of the rows with the dock open, the hover line holds at the rest time (dotted).
    if (S.video) {
      showVideoCursor();
      if (S.drag || event.target.closest?.("#rows, .track, #axis-row")) {
        $("hover-cursor").classList.remove("held");
        showCursor(event.clientX, true, event);
      } else if (!(ws() && Slice.showHeld())) $("hover-cursor").hidden = true;
      return;
    }
    if (S.drag || event.target.closest?.("#top, .track")) {
      $("cursor").classList.remove("held");
      showCursor(event.clientX, false, event);
    } else if (!(ws() && Slice.showHeld())) $("cursor").hidden = true;
  });
  window.addEventListener("hashchange", followHash);
  document.documentElement.addEventListener("pointerleave", () => {
    $("hover-cursor").hidden = true;
    if (!S.video) $("cursor").hidden = true;
    if (ws()) {
      Slice.inRows = false;
      Slice.redraw();
    }
  });
  $("video-time").addEventListener("input", () => {
    const requested = $("video-time").value;
    pauseVideo();
    seekVideo(requested);
  });
  $("video-play").addEventListener("click", playVideo);
  $("start-blank").addEventListener("click", startBlank);
  $("blind-mode").addEventListener("change", () => {
    // An untouched, unsaved shot switches between its two fresh openings: Source or empty.
    const fresh = Boolean(S.meta) && !S.meta.saved && !dirty();
    S.blind = $("blind-mode").checked;
    store("labeler:detachment-blind", String(S.blind));
    if (S.blind) { if (fresh) reopenFresh(); else startBlank(); S.undo = []; }
    else { S.suggestionsShown = true; if (fresh) reopenFresh(); }
    keepExposure();
    showBlindMode(); showHeader(); renderQueue(); sizeCanvases(); render();
  });
  $("top").addEventListener("scroll", showVideoCursor);
  window.addEventListener("pointerup", endDrag);
  window.addEventListener("pointercancel", endDrag);
  document.addEventListener("keydown", onKey);
  const listFailed = (error) => Activity.note("list", error.message, "error");
  $("event").addEventListener("change", () => {
    const value = $("event").value;
    $("event").blur(); // the keys go to the page, as after a click on it
    // All (api 11) shows the shot on every event; an event picked from All opens at its shot
    if (value === Lists.ALL) return toggleByShot(true).catch(listFailed);
    (Lists.byShot ? Lists.openEvent(value) : openEvent(value, null)).catch((error) => say(error.message, true));
  });
  $("shotlist").addEventListener("change", () => chooseList($("shotlist").value).catch(listFailed));
  $("all-next").addEventListener("click", () => Lists.stepShot(1));
  $("silver-breadcrumb").addEventListener("click", () =>
    (S.api >= 11 ? openByShot(Lists.name, Lists.shot).catch(listFailed) : openSilver(Silver.batch, Silver.shot)));
  $("silver-next-event").addEventListener("click", () =>
    (S.api >= 11 ? Lists.nextEvent() : nextSilverEvent()).catch(error => say(error.message, true)));
  $("gold-only").addEventListener("change", () => toggleGoldOnly().catch((error) => say(error.message, true)));
  $("silver-all-layers").addEventListener("change", () => {
    Silver.layers = $("silver-all-layers").checked;
    if (S.api >= 11) Lists.renderStrips();
    else renderSilverOverview();
  });
  $("shot").addEventListener("keydown", ShotSearch.key);
  $("shot").addEventListener("input", ShotSearch.typed);
  $("shot").addEventListener("focus", () => $("shot").select());
  $("bottom").addEventListener("scroll", ShotSearch.place);
  $("silver-page").addEventListener("scroll", ShotSearch.place);
  window.addEventListener("resize", ShotSearch.place);
  $("shot").addEventListener("blur", () => setTimeout(() => {
    if (document.activeElement !== $("shot")) ShotSearch.show([]);
  }, 100));
  $("queue").addEventListener("click", (event) => {
    const chip = event.target.closest(".chip");
    if (chip) openShot(Number(chip.dataset.shot));
  });
  $("save").addEventListener("click", () => save(false));
  $("next").addEventListener("click", () => go(1));
  $("revert").addEventListener("click", () => S.meta && revert());
  $("show-versions").addEventListener("click", toggleVersions);
  $("resolution").addEventListener("change", (event) => setCrowd(event.target.value));
  $("help").addEventListener("click", toggleKeys);
  wireNames();
  for (const dialog of document.querySelectorAll("dialog")) {
    // Chrome can leave focus on a button in a closed dialog, and Enter would then click it
    // instead of saving: let go of it as Escape cancels the dialog (unless the dialog refuses to
    // close), and again once closed.
    const release = (event) =>
      !event.defaultPrevented && dialog.contains(document.activeElement) && document.activeElement.blur();
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
    if (ws()) cutMargin(); // the cut margin follows the window (review D2B M2); the refetch below covers it
    sizeCanvases();
    render();
    fetchRows();
  }).observe(top);
  new ResizeObserver(() => {
    sizeRows();
    sizeCanvases();
    render();
  }).observe($("video-panel"));
  new ResizeObserver(() => {
    sizeCanvases();
    drawDock();
  }).observe($("dock"));
  $("view").addEventListener("change", (event) => {
    if (ws()) Views.choose(event.target.value).catch((error) => Activity.note("view", error.message, "error"));
  });
  $("view").addEventListener("focus", () => ws() && Views.ensureList());
  new ResizeObserver(() => {
    drawSilverOverview();
    Lists.drawStrips();
  }).observe($("silver-strips"));
  renderActivity(); // #activity's line, Stop waiting and detail list
  // The short #saved follows the bar: its width (a resize), and the items beside it. #saved's own
  // text and #activity (which takes no room on the line) are left out, so a fit never refits.
  new ResizeObserver(fitSaved).observe($("bar"));
  const own = (node) => (node.nodeType === 1 ? node : node.parentElement)
    ?.closest("#saved, #activity, #activity-detail, #activity-line");
  new MutationObserver((records) => {
    if (records.some((record) => !own(record.target))) fitSaved();
  }).observe($("bar"), { subtree: true, childList: true, characterData: true, attributes: true,
    attributeFilter: ["hidden", "class"] });
}

if (typeof module !== "undefined") {
  module.exports = { ms, snapSpan, snapWindow, silverFrames, paint, runs, normalise, diffRuns, versionChanges, niceStep, hitTest, hasOverlaps, regionAt, lut, modeLut, producerAt, clipToAllowed, gaps };
} else {
  boot();
}
