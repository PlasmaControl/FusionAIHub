// Views (plan D.3): which rows show, in what order and how tall; signals added as rows; the
// dock's panels; and the named views that keep them, per reviewer and event. Also the view
// editor and the signal picker in the dock. A classic script loaded before app.js: it only
// defines names, and app.js calls them (api 11, workspace).
"use strict";

/** The view in use. `layout` is the working layout (D.3's stored form); `current` the saved
 * view it came from ({name, layout, etag, preset}) or null for Default; `changed` whether it
 * differs; `list` the reviewer's views and the presets for `listFor` ("<event>/<reviewer>");
 * `added` the added rows of the open shot in view order ({key, id, title, units, h, draw,
 * hidden, canvas, entry, range}); `storeSpec` the layout's row for each store row. */
const Views = {
  HEIGHTS: { S: 80, M: 110, L: 150, XL: 220 },
  MISSING_H: 28, // an added row with no data on this shot
  MAP_CHANNELS: 8, // an added trace with this many channels may draw as a map
  NAME_MAX: 64,
  PTDATA: /^[A-Za-z][A-Za-z0-9_]{0,11}$/,
  SEGMENT: /^[A-Za-z][A-Za-z0-9_]*$/,
  NODE_MAX: 96,
  REFUSED: ["DIM_OF", "PTDATA2"],
  Q_EVENTS: ["minimum_safety_factor", "poloidal_beta", "mhd_n1", "mhd_n2", "mhd_n3"], // q in the dock by default
  SUGGEST: 8, // matches the Add row box lists
  SOURCES: { corpus: "corpus", raw: "raw cache", feature: "features", park: "MHD park", recipe: "recipe", mds: "MDS",
    ptdata: "PTDATA" },
  event: null,
  layout: null,
  current: null,
  changed: false,
  list: null,
  listFor: "",
  listing: null, // the /api/views request in flight
  added: [],
  storeSpec: [],
  canvases: new Map(), // key -> the added row's canvas, kept across shots
  chips: new Map(), // key -> the added row's chip in #row-ui
  bitmaps: new Map(), // key -> {data, lo, canvas}: a map row's picture
  catalog: null, // /api/catalog for `catalogFor`
  catalogFor: "",
  catalogFailed: "", // the event/shot whose /api/catalog failed (the Add menu and box say so)
  shot: null, // the shot the layout was last applied to
  editing: false,
  openBefore: false, // the dock was open when the editor opened
  panelsEdited: false, // panels were changed in this edit
  pending: true, // the layout's dock is still to be applied (a view was chosen)
  picking: null, // the picker's state while it shows: {query, source}
  message: "", // the editor's last message
  selected: null, // the row key picked by a click on its name box (R2)
  removed: [], // rows Delete took out of this event's layout, newest last: {key, at, row, depth}; Ctrl+Z puts one back
  box: null, // the Add row box's parts, made on first use

  // -- pure helpers (node tests)

  /** The store's row keys in store order: "store:<title>", a repeat "store:<title>#2". */
  storeKeys(titles) {
    const seen = new Map();
    return titles.map((title) => {
      const n = (seen.get(title) || 0) + 1;
      seen.set(title, n);
      return `store:${title}${n > 1 ? `#${n}` : ""}`;
    });
  },

  /** A store key's title, without its "#2". */
  titleOf: (key) => key.slice("store:".length).replace(/#\d+$/, ""),

  /** The signal id an added row's key names: a catalog id, custom:<NAME> or mds:<TREE>:<node>;
   * null for anything else. */
  idOf(key) {
    if (typeof key !== "string") return null;
    if (key.startsWith("cat:")) return key.slice(4) || null;
    if (key.startsWith("custom:") || key.startsWith("mds:")) return key;
    return null;
  },

  /** A layout laid over the store's `keys`: `rows` in view order, each the layout's row plus
   * `i` (the store index, or null for an added row); `missing` the titles of store keys the
   * store lacks; `fresh` the titles of store rows the layout does not name (appended visible,
   * `other_store_rows: "end"`), said only when the layout names any store row. */
  arrange(layout, keys) {
    const index = new Map(keys.map((key, i) => [key, i]));
    const seen = new Set();
    const rows = [], missing = [];
    let named = false;
    for (const row of layout?.rows || []) {
      if (typeof row?.key !== "string") continue;
      if (row.key.startsWith("store:")) {
        named = true;
        if (!index.has(row.key)) missing.push(Views.titleOf(row.key));
        else if (!seen.has(row.key)) {
          seen.add(row.key);
          rows.push({ ...row, i: index.get(row.key) });
        }
      } else if (Views.idOf(row.key) && !seen.has(row.key)) {
        seen.add(row.key);
        rows.push({ ...row, i: null });
      }
    }
    const fresh = keys.filter((key) => !seen.has(key));
    for (const key of fresh) rows.push({ key, i: index.get(key) });
    return { rows, missing, fresh: named ? fresh.map(Views.titleOf) : [] };
  },

  /** The muted notes for a layout's missing and new store rows. */
  notes({ missing, fresh }) {
    const list = (titles) => titles.join(", ");
    return {
      missing: missing.length ? (missing.length === 1 ? `1 row of this view is not in this store: ${missing[0]}`
        : `${missing.length} rows of this view are not in this store: ${list(missing)}`) : null,
      fresh: fresh.length ? (fresh.length === 1 ? `1 row is new in this store: ${fresh[0]}`
        : `${fresh.length} rows are new in this store: ${list(fresh)}`) : null,
    };
  },

  /** Why `node` in `tree` is not a node the server fetches, as the picker says it; null when it
   * is one. Mirrors the server's rule (plan D.A). */
  nodeProblem(tree, node, trees) {
    const generic = `${node} is not an MDS node name: use letters, digits and _ in parts that start with a letter, up to 96 characters.`;
    if (!trees.includes(String(tree).toUpperCase())) return `${tree} is not an allowed MDS tree${trees.length ? `: ${trees.join(", ")}` : ""}.`;
    if (!node || node.length > Views.NODE_MAX) return generic;
    const path = node.replace(/^\\/, "");
    const at = path.indexOf("::");
    if (at < 0) return `${node}: name its tree, as \\${tree.toUpperCase()}::…`;
    const head = path.slice(0, at);
    if (head.toUpperCase() !== String(tree).toUpperCase()) return `${node} is in ${head}, not ${tree.toUpperCase()}: pick ${head} as the source.`;
    const segments = path.split(/::|[.:]/);
    if (segments.some((segment) => !Views.SEGMENT.test(segment))) return generic;
    if (segments.some((segment) => Views.REFUSED.some((refused) => segment.toUpperCase().startsWith(refused)))) {
      return `${node}: DIM_OF and PTDATA2 are not fetched; name the node itself.`;
    }
    return null;
  },

  /** A name typed in Add row as the server fetches it: an MDS node ("\\EFIT01::BETAN" or
   * "EFIT01::BETAN") as {id: "mds:<TREE>:<node>", text}, a PTDATA point ("ecevs01") as
   * {id: "custom:<NAME>", text}, else {refusal}; null for nothing typed. */
  parsePoint(typed, trees = []) {
    const text = String(typed ?? "").trim();
    if (!text) return null;
    const allowed = trees.map((name) => String(name).toUpperCase());
    if (text.includes("::")) {
      const bare = text.replace(/^\\/, "");
      const tree = bare.slice(0, bare.indexOf("::")).toUpperCase();
      const node = `\\${tree}${bare.slice(bare.indexOf("::"))}`.toUpperCase();
      const problem = Views.nodeProblem(tree, node, allowed);
      return problem ? { refusal: problem } : { id: `mds:${tree}:${node}`, text: `MDS ${node}` };
    }
    if (text.startsWith("\\")) return { refusal: `Add the tree, e.g. EFIT01::${text.slice(1).toUpperCase()}` };
    if (Views.PTDATA.test(text)) return { id: `custom:${text.toUpperCase()}`, text: `PTDATA ${text.toUpperCase()}` };
    return { refusal: "Not a PTDATA name or an MDS node (TREE::NODE)." };
  },

  /** A pointname as the catalog's `points` are compared: no leading \\, lower case. */
  pointKey: (text) => String(text ?? "").trim().replace(/^\\/, "").toLowerCase(),

  /** The names a catalog entry answers to as a pointname: each of its `points` (a PTDATA name,
   * or an MDS node also as TREE::LAST and LAST, so EFIT01::BETAN finds
   * \\efit01::top.results.aeqdsk:betan). */
  pointKeys(entry) {
    const keys = [];
    for (const point of entry?.points || []) {
      const key = Views.pointKey(point);
      const last = key.split(/::|[.:]/).pop();
      keys.push(key);
      if (key.includes("::")) keys.push(`${key.slice(0, key.indexOf("::"))}::${last}`, last);
    }
    return keys;
  },

  /** A copy of a layout (views keep JSON only). */
  clone: (layout) => JSON.parse(JSON.stringify(layout)),

  /** Availability as the picker says it. */
  availability(entry, shot) {
    return entry.local === "disk" ? "on disk" : entry.local === "fetch" ? "fetch, about 5 s" : `no data for ${shot}`;
  },

  // -- the page: views

  /** "Default", the view's name, or either with " (changed)". */
  label() {
    const name = Views.current?.name || "Default";
    return Views.changed ? `${name} (changed)` : name;
  },

  /** The layout differs from its saved view (an edit, `d` on Default). */
  touch() {
    Views.changed = true;
    Views.fillSelect();
    if (typeof Dock !== "undefined" && Dock.open) Dock.drawHead();
  },

  storeKey: (event) => `labeler:view:${event}`,

  /** The select value of a view: "view:<name>" or "preset:<name>"; "" is Default. */
  valueOf: (view) => (view ? `${view.preset ? "preset" : "view"}:${view.name}` : ""),

  find(value) {
    return (Views.list || []).find((view) => Views.valueOf(view) === value) || null;
  },

  /** The reviewer's views and the presets for the open event, asked once per event and
   * reviewer (one GET /api/views). */
  async loadList() {
    const at = `${S.event}/${S.name || ""}`;
    if (Views.listFor === at && (Views.list || Views.listing)) return Views.listing || Views.list;
    Views.listFor = at;
    const query = `event=${encodeURIComponent(S.event)}${S.name ? `&reviewer=${encodeURIComponent(S.name)}` : ""}`;
    Views.listing = (async () => {
      try {
        const body = await (await api(`/api/views?${query}`)).json();
        if (Views.listFor === at) Views.list = body.views || [];
      } catch (error) {
        if (Views.listFor === at) {
          Views.list = [];
          Activity.note("view", `Could not load the views: ${error.message}. Open the View list again to retry.`, "error");
          Views.listFor = "";
        }
      } finally {
        Views.listing = null;
      }
      Views.fillSelect();
      return Views.list;
    })();
    return Views.listing;
  },

  /** #view gets focus: its list loads (once per event and reviewer). */
  ensureList() {
    if (S.event) Views.loadList();
  },

  /** On opening an event: the view last used on it (localStorage labeler:view:<event>). Default
   * or none asks nothing; a named view asks /api/views once. */
  async restore(event) {
    Views.event = event;
    Object.assign(Views, { current: null, changed: false, layout: defaultLayout(event), list: null, listFor: "",
      pending: true, removed: [] }); // Ctrl+Z never puts a row of another event's layout here
    Activity.clear("view", "view-rows");
    const saved = stored(Views.storeKey(event));
    if (saved) {
      await Views.loadList();
      if (Views.event !== event) return;
      const found = Views.find(saved);
      if (found) Object.assign(Views, { current: found, layout: Views.clone(found.layout), pending: true });
      else {
        store(Views.storeKey(event), null);
        Activity.note("view", `The view ${saved.replace(/^(view|preset):/, "")} is gone. Showing Default.`);
      }
      if (S.meta && S.event === event) {
        applyView();
        Views.fetchAdded();
      }
    }
    Views.fillSelect();
  },

  /** Fill #view: Default, your views, the presets, and Edit view…; the selected one says
   * "(changed)" when the layout differs. */
  fillSelect() {
    const select = typeof document === "undefined" ? null : $("view");
    if (!select) return;
    const option = (value, text) => Object.assign(document.createElement("option"), { value, textContent: text });
    const value = Views.valueOf(Views.current);
    const own = (Views.list || []).filter((view) => !view.preset);
    const presets = (Views.list || []).filter((view) => view.preset);
    const parts = [option("", "Default")];
    parts.push(...own.map((view) => option(Views.valueOf(view), view.name)));
    if (Views.current && !(Views.list || []).some((view) => Views.valueOf(view) === value)) {
      parts.push(option(value, Views.current.name));
    }
    if (presets.length) {
      const group = Object.assign(document.createElement("optgroup"), { label: "Presets" });
      group.append(...presets.map((view) => option(Views.valueOf(view), view.name)));
      parts.push(group);
    }
    parts.push(option("__edit", "Edit view…"));
    select.replaceChildren(...parts);
    select.value = value;
    const chosen = select.selectedOptions[0];
    if (chosen && Views.changed) chosen.textContent = Views.label();
    select.title = Views.label();
  },

  /** #view changed: apply the chosen view (remembered for the event), or open the editor. */
  async choose(value) {
    if (value === "__edit") {
      Views.fillSelect();
      return openEditor();
    }
    if (value && !Views.find(value)) await Views.loadList();
    const view = value ? Views.find(value) : null;
    Object.assign(Views, { current: view, changed: false, pending: true, removed: [], // Ctrl+Z stays in this layout
      layout: view ? Views.clone(view.layout) : defaultLayout(S.event) });
    store(Views.storeKey(S.event), view ? value : null);
    Activity.clear("view", "view-rows");
    Views.fillSelect();
    applyView();
    Views.fetchAdded();
    if (Views.editing) {
      // Reset and Delete inside the editor keep it open (review D2B M1): the dock stays shown
      // while editing, and Done leaves it as the chosen layout has it.
      Object.assign(Views, { openBefore: Dock.open, panelsEdited: false });
      if (!Dock.open) Dock.show(true);
      Views.renderEditor();
    }
  },

  /** The layout as it stands, in the stored form (D.3); pins are never in it. */
  snapshot(name) {
    const rows = Views.order().map((spec) => {
      const row = { key: spec.key };
      if (spec.h) row.h = spec.h;
      if (spec.draw === "map") row.draw = "map";
      if (spec.hidden) row.hidden = true;
      return row;
    });
    const open = Views.editing && !Views.openBefore && !Views.panelsEdited ? false : Dock.open;
    return { v: 1, name, event: S.event, reviewer: S.name, saved_at: new Date().toISOString(), rows,
      other_store_rows: "end",
      dock: { open, width: Dock.width, follow: Slice.follow, panels: Views.clone(Dock.panels) } };
  },

  /** Every row in view order as the layout's rows: [{key, h?, draw?, hidden?, i}]. */
  order() {
    if (!S.meta) return Views.layout?.rows || [];
    return Views.arrange(Views.layout, Views.storeKeys(S.meta.rows.map((row) => row.title))).rows;
  },

  /** Give the layout a row for every store row (the default names none), so the editor can
   * move them. */
  materialize() {
    Views.layout.rows = Views.order().map(({ i, ...row }) => row);
  },

  /** Send a change to /api/views; the answer's views become the list. Returns the list, or null
   * after saying why it failed. */
  async post(body) {
    const doing = { delete: "delete", rename: "rename" }[body.op] || "save";
    try {
      const response = await api("/api/views", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event: S.event, reviewer: S.name, ...body }) });
      const answer = await response.json();
      Views.list = answer.views || [];
      Views.listFor = `${S.event}/${S.name || ""}`;
      return Views.list;
    } catch (error) {
      const conflict = error.status === 409 && !/exists/.test(error.message);
      if (conflict) {
        Views.listFor = ""; // the list is stale: ask for it again, with the other tab's etags
        Views.loadList();
      }
      Views.say(conflict ? "This view changed in another tab. Reload it, or save yours under a new name."
        : `Could not ${doing} the view: ${error.message.replace(/\.$/, "")}.`, true);
      return null;
    }
  },

  /** A message in the editor's footer, and in #activity (never #status). */
  say(text, error = false) {
    Views.message = text;
    Activity.note("view", text || null, error ? "error" : "");
    if (Views.editing) Views.renderFooter();
  },

  /** A reviewer name is needed to save: without one, say so and open #who. */
  named() {
    if (S.name) return true;
    Views.say("Pick your name to save views");
    chooseName();
    return false;
  },

  /** The name typed in the editor's footer, checked; null after saying why not. */
  typedName() {
    const name = ($("view-name")?.value || "").trim();
    if (!name || name.length > Views.NAME_MAX) {
      Views.say(name ? `A view name is at most ${Views.NAME_MAX} characters.` : "Type a name for the view.", true);
      return null;
    }
    if (name.toLowerCase() === "default") {
      Views.say("Default is built in. Pick another name.", true);
      return null;
    }
    return name;
  },

  /** After a save or a rename: the named view is the current one, unchanged. */
  adopt(name) {
    Views.current = (Views.list || []).find((view) => !view.preset && view.name === name) || null;
    Views.changed = false;
    if (Views.current) store(Views.storeKey(S.event), Views.valueOf(Views.current));
    Views.fillSelect();
    if (Dock.open) Dock.drawHead();
  },

  // -- the page: added rows

  /** An added row's record for a layout row `spec`, its canvas kept across shots. */
  makeAdded(spec) {
    const id = Views.idOf(spec.key);
    let canvas = Views.canvases.get(spec.key);
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvas.dataset.signal = id;
      Views.canvases.set(spec.key, canvas);
    }
    const entry = S.shot == null ? null : Signals.peek(S.shot, id);
    const catalog = Views.catalogEntry(id);
    const title = catalog?.label || entry?.signal?.label || (id.startsWith("mds:") ? id.slice(id.indexOf(":", 4) + 1) : id.replace(/^custom:/, ""));
    return { key: spec.key, id, title, units: catalog?.units ?? "", h: spec.h || null, draw: spec.draw || "lines",
      hidden: Boolean(spec.hidden), canvas, entry, range: null };
  },

  catalogEntry: (id) => Views.catalog?.entries?.find((entry) => entry.id === id) || null,

  catalogLabel: (id) => Views.catalogEntry(id)?.label || null,

  /** What an added row's chip says where its data come from. */
  facts(added) {
    const id = added.id;
    if (id.startsWith("custom:")) return { name: id.slice(7), source: "ptdata" };
    if (id.startsWith("mds:")) {
      const tree = id.slice(4, id.indexOf(":", 4));
      return { name: id.slice(id.indexOf(":", 4) + 1), source: "mds", tree };
    }
    const entry = Views.catalogEntry(id);
    const mds = entry?.source === "mds";
    return { name: added.title, source: !entry || entry.local === "disk" ? "disk" : mds ? "mds" : "ptdata",
      tree: mds ? String(entry.where || "").replace(/^\\/, "").split("::")[0] : undefined };
  },

  /** Ask for every added row's window, as loadRows does for the store's: a fifth wider than
   * the view either side, one column per pixel. An entry already holding it asks nothing. */
  fetchAdded() {
    if (!S.meta || !Views.added.length) return;
    const [v0, v1] = S.view;
    const t0 = Math.max(S.meta.t_range[0], v0 - (v1 - v0) * 0.2);
    const t1 = Math.min(S.meta.t_range[1], v1 + (v1 - v0) * 0.2);
    if (!(t1 > t0)) return;
    const cols = clamp(Math.round((plotWidth() * (t1 - t0)) / (v1 - v0)), 16, 8192);
    for (const added of Views.added) {
      added.entry = loadSignal(S.shot, added.id, t0, t1, cols, Views.facts(added), Views.signalChanged);
    }
  },

  /** A signal's state changed: its row and chip redraw (and the editor's status). */
  signalChanged(entry) {
    if (entry.shot !== S.shot) return;
    for (const added of Views.added) if (added.id === entry.id) added.entry = entry;
    sizeCanvases();
    render();
    if (Views.editing) Views.renderEditor();
  },

  /** Remove the added row of `id` from the view (Remove from view on a chip). */
  removeKey(id) {
    if (!Views.layout) return;
    Views.materialize();
    Views.layout.rows = Views.layout.rows.filter((row) => Views.idOf(row.key) !== id);
    Views.touch();
    applyView();
    if (Views.editing) Views.renderEditor();
  },

  /** A trace row's y range over the columns on screen, padded like the store's. */
  traceRange(data) {
    const step = (data.t1 - data.t0) / data.n;
    const j0 = clamp(Math.floor((S.view[0] - data.t0) / step), 0, data.n);
    const j1 = clamp(Math.ceil((S.view[1] - data.t0) / step), 0, data.n);
    let [lo, hi] = [Infinity, -Infinity];
    for (let p = 0; p < 2 * data.channels; p++) {
      for (let j = j0; j < j1; j++) {
        const v = data.values[p * data.n + j];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (!(hi > lo)) [lo, hi] = Number.isFinite(lo) ? [lo - 1, lo + 1] : [0, 1];
    const pad = (hi - lo) * 0.06;
    return [lo - pad, hi + pad];
  },

  /** A map row's picture: a pixel per column and channel (channel 0 at the bottom), each
   * bucket's midpoint over the window's 1st to 99th percentile, in the page's colours. */
  mapBitmap(added) {
    const data = added.entry.data;
    const cached = Views.bitmaps.get(added.key);
    if (cached?.data === data && cached.lo === S.lo) {
      added.range = cached.range;
      return cached.canvas;
    }
    const { n, channels, values } = data;
    const mids = new Float32Array(n * channels);
    for (let c = 0; c < channels; c++) {
      for (let j = 0; j < n; j++) mids[c * n + j] = (values[c * n + j] + values[(channels + c) * n + j]) / 2;
    }
    const range = Profiles.span(mids) || [0, 1];
    if (!(range[1] > range[0])) range[1] = range[0] + 1;
    const canvas = document.createElement("canvas");
    [canvas.width, canvas.height] = [n, channels];
    const g = canvas.getContext("2d");
    const image = g.createImageData(n, channels);
    const pixels = new Uint32Array(image.data.buffer);
    for (let y = 0; y < channels; y++) {
      const c = channels - 1 - y;
      for (let j = 0; j < n; j++) {
        const v = mids[c * n + j];
        const level = Number.isFinite(v) ? Math.round(clamp((v - range[0]) / (range[1] - range[0]), 0, 1) * 255) : 0;
        pixels[y * n + j] = S.lut[level];
      }
    }
    g.putImageData(image, 0, 0);
    Views.bitmaps.set(added.key, { data, lo: S.lo, canvas, range });
    added.range = range;
    return canvas;
  },

  /** An added row as the store's drawing code takes a row. */
  rowLike(added) {
    const data = added.entry?.data;
    const signal = added.entry?.signal;
    const facts = Views.facts(added);
    const source = facts.source === "ptdata" ? "PTDATA" : facts.source === "mds" ? `MDS ${facts.tree || ""}`.trim()
      : Views.SOURCES[signal?.source] || "";
    const units = signal?.units ?? added.units ?? "";
    return { name: added.key, kind: "trace", title: signal?.label || added.title, hlines: [],
      y_units: [units, source].filter(Boolean).join(" · "), legend: signal?.legend || [],
      n_channels: data?.channels || 1 };
  },

  /** Place an added row's chip, centred over its plot (or at the left of a collapsed row);
   * none when `note` is null. */
  placeChip(added, note) {
    let chip = Views.chips.get(added.key);
    const ui = $("row-ui");
    if (!note || !ui || added.hidden) {
      chip?.remove();
      Views.chips.delete(added.key);
      return;
    }
    if (!chip) {
      chip = Object.assign(document.createElement("div"), { className: "row-chip" });
      chip.dataset.signal = added.id;
      const button = Object.assign(document.createElement("button"), { type: "button" });
      button.addEventListener("click", () => Dock.entryAct(added.entry));
      chip.append(document.createElement("span"), button);
      ui.append(chip);
      Views.chips.set(added.key, chip);
    }
    chip.className = `row-chip signal-chip${note.tone ? ` ${note.tone}` : ""}`;
    chip.firstChild.textContent = note.text;
    chip.lastChild.textContent = note.button || "";
    chip.lastChild.hidden = !note.button;
    const offset = (element) => {
      let [x, y] = [0, 0];
      for (let at = element; at; at = at.offsetParent) [x, y] = [x + at.offsetLeft, y + at.offsetTop];
      return [x, y];
    };
    const [[cx, cy], [ux, uy]] = [offset(added.canvas), offset(ui)];
    const [w, h] = [added.canvas.clientWidth, added.canvas.clientHeight];
    chip.style.maxWidth = `${Math.max(160, w - GUTTER - RIGHT - Slice.margin - 16)}px`;
    const collapsed = note.collapse;
    const left = collapsed ? cx - ux + GUTTER + 8 : cx - ux + GUTTER + (w - GUTTER - RIGHT - Slice.margin - chip.offsetWidth) / 2;
    const top = cy - uy + (h - chip.offsetHeight) / 2;
    chip.style.left = `${Math.max(cx - ux + GUTTER, left)}px`;
    chip.style.top = `${top}px`;
  },

  // -- the editor and the picker, in #dock-edit

  /** Show #dock-edit (the editor or the picker) in place of the panels. */
  showEdit(on) {
    if (on && !Views.editing) Object.assign(Views, { openBefore: Dock.open, panelsEdited: false });
    if (on && !Dock.open) Dock.show(true);
    Views.editing = on;
    Dock.editing = on;
    const [edit, panels] = [$("dock-edit"), $("dock-panels")];
    if (edit) edit.hidden = !on;
    if (panels) panels.hidden = on;
    if (!on) Views.picking = null;
  },

  /** The editor: the rows table, the panels, Add signal, and the footer. */
  renderEditor() {
    const box = $("dock-edit");
    if (!box || !Views.editing) return;
    if (Views.picking) return Views.renderPicker();
    const el = (tag, props = {}, ...children) => {
      const node = Object.assign(document.createElement(tag), props);
      node.append(...children);
      return node;
    };
    const rows = Views.order();
    const table = el("table", { className: "editor-table" });
    table.append(el("thead", {}, el("tr", {}, ...["", "Row", "Height", "Draw", "Show", "", "Status"]
      .map((text) => el("th", { scope: "col", textContent: text })))));
    const body = el("tbody");
    const store = S.meta?.rows || [];
    rows.forEach((spec, at) => {
      const added = spec.i == null ? Views.added.find((row) => row.key === spec.key) : null;
      const title = spec.i != null ? store[spec.i].title : added?.title || spec.key;
      const source = spec.i != null ? "event row" : Views.sourceName(spec.key);
      const tr = el("tr", { draggable: true });
      tr.dataset.key = spec.key;
      const grip = el("button", { type: "button", className: "grip", textContent: "⋮⋮", title: "Drag to reorder (Alt+↑ and Alt+↓ also move it)" });
      grip.setAttribute("aria-label", `Move ${title}`);
      grip.addEventListener("keydown", (event) => {
        if (!event.altKey || !["ArrowUp", "ArrowDown"].includes(event.key)) return;
        event.preventDefault();
        Views.moveRow(spec.key, at + (event.key === "ArrowUp" ? -1 : 1));
        box.querySelector(`tr[data-key="${CSS.escape(spec.key)}"] .grip`)?.focus();
      });
      tr.addEventListener("dragstart", (event) => event.dataTransfer.setData("text/plain", spec.key));
      tr.addEventListener("dragover", (event) => event.preventDefault());
      tr.addEventListener("drop", (event) => {
        event.preventDefault();
        Views.moveRow(event.dataTransfer.getData("text/plain"), at);
      });
      const height = el("select", {}, ...(spec.i != null ? [el("option", { value: "", textContent: "Auto" })] : []),
        ...Object.entries(Views.HEIGHTS).map(([name, px]) => el("option", { value: String(px), textContent: name })));
      height.setAttribute("aria-label", `Height of ${title}`);
      height.value = spec.h ? String(spec.h) : spec.i != null ? "" : String(Views.HEIGHTS.M);
      height.addEventListener("change", () => Views.setRow(spec.key, { h: height.value ? Number(height.value) : null }));
      const channels = added?.entry?.data?.channels ?? Views.catalogEntry(added?.id)?.channels ?? 0;
      let draw = el("span");
      if (added && channels >= Views.MAP_CHANNELS) {
        draw = el("select", {}, el("option", { value: "lines", textContent: "lines" }), el("option", { value: "map", textContent: "map" }));
        draw.setAttribute("aria-label", `Draw ${title} as`);
        draw.value = spec.draw === "map" ? "map" : "lines";
        draw.addEventListener("change", () => Views.setRow(spec.key, { draw: draw.value }));
      }
      const show = el("input", { type: "checkbox", checked: !spec.hidden });
      show.setAttribute("aria-label", `Show ${title}`);
      show.addEventListener("change", () => Views.setRow(spec.key, { hidden: !show.checked }));
      const remove = added ? el("button", { type: "button", textContent: "Remove" }) : el("span");
      if (added) remove.addEventListener("click", () => Views.removeKey(added.id));
      const note = added ? Dock.entryNote(added.entry, added.title) : null;
      tr.append(el("td", {}, grip), el("td", {}, el("span", { textContent: title }), el("small", { textContent: ` ${source}` })),
        el("td", {}, height), el("td", {}, draw), el("td", {}, show), el("td", {}, remove),
        el("td", { className: note?.tone === "error" ? "error" : "", textContent: note ? note.text : added ? "ready" : "" }));
      body.append(tr);
    });
    table.append(body);
    const addSignal = el("button", { type: "button", textContent: "Add signal" });
    addSignal.addEventListener("click", () => openPicker());
    box.replaceChildren(el("h3", { textContent: "Rows" }), table, addSignal, Views.panelsEditor(el), el("div", { id: "view-footer" }));
    Views.renderFooter();
  },

  /** The editor's panels: one line each with ↑, ↓ and Remove, and Add panel. */
  panelsEditor(el) {
    const list = el("ol", { className: "editor-panels" });
    Dock.panels.forEach((panel, at) => {
      const name = Views.panelName(panel);
      const up = el("button", { type: "button", textContent: "↑", disabled: at === 0 });
      up.setAttribute("aria-label", `Move ${name} up`);
      up.addEventListener("click", () => Views.movePanel(at, at - 1));
      const down = el("button", { type: "button", textContent: "↓", disabled: at === Dock.panels.length - 1 });
      down.setAttribute("aria-label", `Move ${name} down`);
      down.addEventListener("click", () => Views.movePanel(at, at + 1));
      const remove = el("button", { type: "button", textContent: "Remove" });
      remove.addEventListener("click", () => Views.setPanels(Dock.panels.filter((_, k) => k !== at)));
      list.append(el("li", {}, el("span", { textContent: name }), up, down, remove));
    });
    const add = el("select", {}, el("option", { value: "", textContent: "Add panel…" }));
    add.setAttribute("aria-label", "Add a panel");
    const offer = (value, text) => add.append(el("option", { value, textContent: text }));
    for (const entry of (Views.catalog?.entries || []).filter((item) => item.kind === "profile")) offer(`profile:${entry.id}`, entry.label);
    if (!Views.catalog) offer("catalog", "Profiles: load the signal list");
    offer("values", "Values");
    offer("slice:inversion", "TangTV inversion and EFIT");
    offer("slice:equilibrium", "EFIT equilibrium");
    if (S.video) offer("cameras", "Cameras");
    add.addEventListener("change", async () => {
      const [kind, id] = add.value.split(":");
      if (kind === "catalog") {
        await Views.loadCatalog();
        return Views.renderEditor();
      }
      if (kind && !Dock.panels.some((panel) => panel.kind === kind && (panel.id ?? "") === (id ?? ""))) {
        Views.setPanels([...Dock.panels, id ? { kind, id, ...(kind === "values" || kind === "cameras" ? {} : { h: kind === "slice" ? Dock.SLICE_H : Dock.PROFILE_H }) } : { kind }]);
      }
      add.value = "";
    });
    return el("section", {}, el("h3", { textContent: "Panels" }), list, add);
  },

  panelName(panel) {
    if (panel.kind === "values") return "Values";
    if (panel.kind === "cameras") return "Cameras";
    if (panel.kind === "slice") return panel.id === "equilibrium" ? "EFIT equilibrium" : "TangTV inversion and EFIT";
    return Views.catalogLabel(panel.id) || Signals.entry(Profiles.shots, "profile", S.shot, panel.id).signal?.label || panel.id;
  },

  /** Where an added row's key comes from, as the editor says it. */
  sourceName(key) {
    if (key.startsWith("custom:")) return "PTDATA";
    if (key.startsWith("mds:")) return `MDS ${key.slice(4, key.indexOf(":", 4))}`;
    const entry = Views.catalogEntry(Views.idOf(key));
    return entry ? Views.SOURCES[entry.source] || entry.source : "catalog";
  },

  /** The footer: the name, Save, Save as, Rename, Delete, Reset to default and Done. */
  renderFooter() {
    const box = $("view-footer");
    if (!box) return;
    const button = (text, act, disabled = false) => {
      const node = Object.assign(document.createElement("button"), { type: "button", textContent: text, disabled });
      node.addEventListener("click", act);
      return node;
    };
    const own = Views.current && !Views.current.preset;
    const name = Object.assign(document.createElement("input"), { id: "view-name", value: own ? Views.current.name : "",
      placeholder: "View name", maxLength: Views.NAME_MAX, spellcheck: false });
    name.setAttribute("aria-label", "View name");
    const label = Object.assign(document.createElement("span"), { className: "view-label", textContent: Views.label() });
    const message = Object.assign(document.createElement("p"), { className: "view-message", textContent: Views.message });
    message.setAttribute("role", "status");
    box.replaceChildren(label, name,
      button("Save", () => saveView(false)), button("Save as", () => saveView(true)),
      button("Rename", () => renameView(), !own), button("Delete", () => deleteView(), !own),
      button("Reset to default", () => resetView()), button("Done", () => Views.closeEditor()), message);
  },

  closeEditor() {
    Views.showEdit(false);
    Views.message = "";
    if (Activity.notes.get("view")?.tone !== "error") Activity.clear("view"); // "Saved the view …" ends with the edit
    if (!Views.openBefore && !Views.panelsEdited) Dock.show(false);
    else if (!Dock.panels.length) toggleDock(false);
    else {
      Dock.layout();
      drawDock();
    }
  },

  /** Change a layout row (h, draw, hidden) and show it. */
  setRow(key, change) {
    Views.materialize();
    const row = Views.layout.rows.find((item) => item.key === key);
    if (!row) return;
    Object.assign(row, change);
    for (const name of Object.keys(change)) if (row[name] == null || row[name] === false) delete row[name];
    Views.touch();
    applyView();
    Views.fetchAdded();
    Views.renderEditor();
  },

  /** Move a row to `to` in the view's order. */
  moveRow(key, to) {
    Views.materialize();
    const rows = Views.layout.rows;
    const from = rows.findIndex((row) => row.key === key);
    if (from < 0 || to < 0 || to >= rows.length || from === to) return;
    rows.splice(to, 0, ...rows.splice(from, 1));
    Views.touch();
    applyView();
    Views.renderEditor();
  },

  setPanels(panels) {
    Dock.panels = panels;
    Views.panelsEdited = true;
    Views.layout.dock = { ...(Views.layout.dock || {}), open: true, panels: Views.clone(panels) };
    Views.touch();
    Dock.layout();
    Dock.relayout();
    Views.renderEditor();
  },

  movePanel(from, to) {
    if (to < 0 || to >= Dock.panels.length) return;
    const panels = Dock.panels.slice();
    panels.splice(to, 0, ...panels.splice(from, 1));
    Views.setPanels(panels);
  },

  /** /api/catalog for the open event and shot, asked when the picker first opens. */
  async loadCatalog() {
    const at = `${S.event}/${S.shot}`;
    if (Views.catalogFor === at && Views.catalog) return Views.catalog;
    try {
      const body = await (await api(`/api/catalog?event=${encodeURIComponent(S.event)}&shot=${S.shot}`)).json();
      Views.catalog = body;
      Views.catalogFor = at;
      Views.catalogFailed = "";
    } catch (error) {
      Views.catalogFailed = at;
      Views.say(`Could not load the signal list: ${error.message}. Try again in a minute.`, true);
    }
    return Views.catalog;
  },

  /** The picker: search, Source (PTDATA or an allowed tree), the matches by group with their
   * availability, and the offer of a PTDATA point or an MDS node. */
  renderPicker() {
    const box = $("dock-edit");
    const state = Views.picking;
    if (!box || !state) return;
    const el = (tag, props = {}, ...children) => {
      const node = Object.assign(document.createElement(tag), props);
      node.append(...children);
      return node;
    };
    const trees = Views.catalog?.trees || [];
    const query = el("input", { type: "search", value: state.query, placeholder: "Search signals, or type a point or node name",
      spellcheck: false, id: "picker-query" });
    query.setAttribute("aria-label", "Search signals");
    const source = el("select", { id: "picker-source" }, el("option", { value: "ptdata", textContent: "PTDATA" }),
      ...trees.map((tree) => el("option", { value: tree, textContent: `MDS ${tree}` })));
    source.setAttribute("aria-label", "Source of a typed name");
    source.value = state.source;
    const results = el("div", { className: "picker-results" });
    const back = el("button", { type: "button", textContent: "Back to the view" });
    back.addEventListener("click", () => {
      Views.picking = null;
      Views.renderEditor();
    });
    const fill = () => {
      state.query = query.value;
      state.source = source.value;
      results.replaceChildren(...Views.pickerResults(el));
    };
    query.addEventListener("input", fill);
    source.addEventListener("change", fill);
    query.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      results.querySelector("button")?.click();
    });
    box.replaceChildren(el("h3", { textContent: "Add signal" }), el("div", { className: "picker-bar" }, query, source), results, back);
    fill();
    query.focus();
  },

  /** The picker's result lines for its state. */
  pickerResults(el) {
    const state = Views.picking;
    if (!Views.catalog) return [el("p", { className: "hint", textContent: "Loading the signal list…" })];
    const { matches, offer, refusal } = matchCatalog(Views.catalog.entries || [], state.query, state.source, Views.catalog.trees || []);
    const out = [];
    if (refusal) out.push(el("p", { className: "error", textContent: refusal }));
    if (offer) {
      const add = el("button", { type: "button", className: "picker-result", textContent: offer.text });
      add.addEventListener("click", () => Views.addRow(offer.id));
      out.push(add);
    }
    const name = S.events?.find?.((event) => event.event === S.event)?.display_name || S.event;
    const groups = new Map();
    for (const entry of matches) {
      const group = entry.suggested ? `Suggested for ${name}` : entry.group || "Other";
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push(entry);
    }
    for (const [group, entries] of groups) {
      out.push(el("h4", { textContent: group }));
      for (const entry of entries.slice(0, 60)) {
        const inView = entry.kind === "profile" ? Dock.panels.some((panel) => panel.kind === "profile" && panel.id === entry.id)
          : Views.order().some((row) => Views.idOf(row.key) === entry.id);
        const button = el("button", { type: "button", textContent: entry.kind === "profile" ? "Add as panel" : "Add as row",
          disabled: inView || entry.local === "missing" });
        button.addEventListener("click", () => (entry.kind === "profile" ? Views.addPanel(entry.id) : Views.addRow(entry.id, entry)));
        const facts = [entry.units, entry.kind, `${Views.SOURCES[entry.source] || entry.source}${entry.where ? `: ${entry.where}` : ""}`,
          Views.availability(entry, S.shot)].filter(Boolean).join(" · ");
        out.push(el("div", { className: "picker-result" }, el("span", { textContent: entry.label }), el("small", { textContent: ` ${facts}` }), button));
      }
    }
    if (!matches.length && !offer && !refusal && state.query.trim()) {
      out.push(el("p", { className: "hint", textContent: state.source === "ptdata"
        ? "No signal in the list matches."
        : "No signal in the list matches. Type a node of this tree: TREE::NODE." }));
    }
    return out;
  },

  /** Add the signal `id` as a row below the store's (a ≥ 8-channel catalog map draws as one). */
  addRow(id, entry = null) {
    if (!Views.layout) return;
    Views.materialize();
    const key = id.startsWith("custom:") || id.startsWith("mds:") ? id : `cat:${id}`;
    if (!Views.layout.rows.some((row) => row.key === key)) {
      const map = entry?.draw === "map" && entry.channels >= Views.MAP_CHANNELS;
      Views.layout.rows.push(map ? { key, h: Views.HEIGHTS.L, draw: "map" } : { key, h: Views.HEIGHTS.M });
    }
    Views.touch();
    applyView();
    Views.fetchAdded();
    Views.say(`Added ${entry?.label || id.replace(/^(custom|mds:[^:]+):/, "")} below the rows.`);
    Views.picking = null;
    Views.renderEditor();
  },

  /** Add the profile `id` as a dock panel. */
  addPanel(id) {
    if (!Dock.panels.some((panel) => panel.kind === "profile" && panel.id === id)) {
      Views.setPanels([...Dock.panels, { kind: "profile", id, h: Dock.PROFILE_H }]);
    }
    Views.picking = null;
    Views.renderEditor();
  },

  // -- row editing on the rows themselves (R2): select by the name box, Delete, Ctrl+Z

  /** The view key of a row's canvas in #rows, or null. */
  keyOf(canvas) {
    return Slice.rows(true).find((item) => item.canvas === canvas)?.key || null;
  },

  /** Select the row `key` (a click on its name box; again, or Escape, lets go): it is outlined,
   * with Pop out and Remove beside its name. */
  selectRow(key) {
    Views.selected = key || null;
    if (key && typeof S !== "undefined") S.selected = -1; // one selection at a time (Delete)
    for (const item of Slice.rows(true)) item.canvas?.classList.toggle("row-selected", item.key === Views.selected);
    Views.placeRowChip();
  },

  /** The selected row's chip: Pop out (to the dock) and Remove (Delete), over its name box. */
  placeRowChip() {
    const ui = typeof document === "undefined" ? null : $("row-ui");
    if (!ui) return;
    let chip = $("row-select");
    const item = Views.selected ? Slice.rows().find((found) => found.key === Views.selected) : null;
    if (!item) {
      if (Views.selected && !Slice.rows(true).some((found) => found.key === Views.selected)) Views.selected = null;
      chip?.remove();
      return;
    }
    if (!chip) {
      chip = Object.assign(document.createElement("div"), { id: "row-select", className: "row-chip row-select" });
      const pop = Object.assign(document.createElement("button"), { type: "button", textContent: "Pop out", title: "Show this row in the dock" });
      pop.addEventListener("click", () => {
        const key = Views.selected;
        Views.selectRow(null);
        Dock.popRow(key);
      });
      const remove = Object.assign(document.createElement("button"), { type: "button", textContent: "Remove", title: "Remove from the view (Delete); Ctrl+Z undoes" });
      remove.addEventListener("click", () => Views.removeSelectedRow());
      chip.append(pop, remove);
      ui.append(chip);
    }
    const canvas = item.canvas;
    let [x, y] = [0, 0];
    for (let at = canvas; at && at !== ui.offsetParent; at = at.offsetParent) [x, y] = [x + at.offsetLeft, y + at.offsetTop];
    chip.style.left = `${x + GUTTER + 6}px`;
    chip.style.top = `${y + 4}px`;
  },

  /** Delete on a selected row: take it out of the view (a store row hides, an added row goes);
   * Ctrl+Z puts it back, and Reset to default always does. False with no row selected. */
  removeSelectedRow() {
    const key = Views.selected;
    if (!key || !Views.layout || !S.meta) return false;
    Views.materialize();
    const at = Views.layout.rows.findIndex((row) => row.key === key);
    if (at < 0) return false;
    const title = Slice.rows(true).find((item) => item.key === key)?.title || key;
    Views.removed.push({ key, at, row: Views.clone(Views.layout.rows[at]), depth: S.undo.depth ?? S.undo.length });
    Views.selectRow(null);
    if (key.startsWith("store:")) Views.layout.rows[at].hidden = true;
    else Views.layout.rows.splice(at, 1);
    Views.touch();
    applyView();
    Views.renderEditor();
    Activity.note("view", `Removed ${title}. Ctrl+Z puts it back.`);
    return true;
  },

  /** Ctrl+Z after Delete on a row, before any later label edit: the row comes back where it
   * was. False when the last undoable change was not a row. */
  undoRow() {
    const last = Views.removed[Views.removed.length - 1];
    if (!last || last.depth !== (S.undo.depth ?? S.undo.length) || !Views.layout || !S.meta) return false;
    Views.removed.pop();
    Views.materialize();
    const rows = Views.layout.rows;
    const at = rows.findIndex((row) => row.key === last.key);
    if (at >= 0) rows[at] = last.row;
    else rows.splice(Math.min(last.at, rows.length), 0, last.row);
    Views.touch();
    applyView();
    Views.fetchAdded();
    Views.renderEditor();
    Activity.note("view", null);
    return true;
  },

  // -- Add row: a box under the rows' names that takes any fdp pointname (R2)

  /** The Add row box, made once, fixed over the time axis's left gutter. */
  addBox() {
    if (typeof document === "undefined" || !document.body) return null;
    if (Views.box) return Views.box;
    const wrap = Object.assign(document.createElement("div"), { id: "add-row" });
    const input = Object.assign(document.createElement("input"), { id: "add-row-input", type: "text", placeholder: "+ Add row",
      autocomplete: "off", spellcheck: false, title: "Add a row: a signal, a PTDATA name (ECEVS01) or an MDS node (EFIT01::BETAN)" });
    input.setAttribute("aria-label", "Add a row by signal or point name");
    input.setAttribute("role", "combobox");
    input.setAttribute("aria-autocomplete", "list");
    input.setAttribute("aria-controls", "add-row-list");
    input.setAttribute("aria-expanded", "false");
    const list = Object.assign(document.createElement("ul"), { id: "add-row-list", hidden: true });
    list.setAttribute("role", "listbox");
    const note = Object.assign(document.createElement("p"), { id: "add-row-note", hidden: true });
    note.setAttribute("role", "status");
    const above = Object.assign(document.createElement("div"), { className: "add-row-above" }); // opens upward
    above.append(list, note);
    wrap.append(above, input);
    document.body.append(wrap);
    const state = { input, list, note, wrap, options: [], active: -1 };
    input.addEventListener("focus", async () => {
      wrap.classList.add("open");
      if (!Views.catalog || Views.catalogFor !== `${S.event}/${S.shot}`) await Views.loadCatalog();
      Views.suggest();
    });
    input.addEventListener("blur", () => setTimeout(() => {
      if (document.activeElement === input) return;
      wrap.classList.remove("open");
      list.hidden = true;
      input.setAttribute("aria-expanded", "false");
    }, 150));
    input.addEventListener("input", async () => {
      if (Views.catalogFailed === `${S.event}/${S.shot}`) await Views.loadCatalog(); // typing retries
      Views.suggest();
    });
    input.addEventListener("keydown", (event) => {
      const n = state.options.length;
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        if (!n) return;
        const down = event.key === "ArrowDown";
        state.active = state.active < 0 ? (down ? 0 : n - 1) : (state.active + (down ? 1 : n - 1)) % n;
        Views.showSuggestions();
      } else if (event.key === "Enter") {
        event.preventDefault();
        Views.takeSuggestion(state.options[Math.max(0, state.active)]);
      } else if (event.key === "Escape") {
        event.preventDefault();
        input.value = "";
        note.hidden = true;
        input.blur();
      }
    });
    Views.box = state;
    return state;
  },

  /** Place the Add row box over the time axis's gutter; shown on a shot with the workspace. */
  placeAddBox() {
    const show = typeof document !== "undefined" && Boolean(S.meta) && ws() && !$("top").hidden;
    if (!show && !Views.box) return;
    const box = Views.addBox();
    if (!box) return;
    box.wrap.hidden = !show;
    if (!show) return;
    const axis = $("axis-row").getBoundingClientRect();
    box.wrap.style.left = `${axis.left + 2}px`;
    box.wrap.style.top = `${axis.top + 2}px`;
  },

  /** The box's suggestions for what is typed: catalog rows that match (suggested first), then
   * the typed name as a PTDATA point or an MDS node, which the server checks when it fetches. */
  suggest() {
    const box = Views.box;
    if (!box) return;
    const typed = box.input.value.trim();
    const entries = (Views.catalog?.entries || []).filter((entry) => entry.kind !== "profile");
    const inView = new Set(Views.order().filter((row) => !row.hidden).map((row) => Views.idOf(row.key)).filter(Boolean));
    const found = typed ? matchCatalog(entries, typed).matches : [];
    const matches = found.filter((entry) => !inView.has(entry.id)).slice(0, Views.SUGGEST);
    // A pointname the catalog curates is that entry, never a second raw row of it.
    const exact = found.filter((entry) => Views.pointKeys(entry).includes(Views.pointKey(typed)));
    const known = exact.length && exact.every((entry) => inView.has(entry.id)) ? exact[0] : null;
    const parsed = exact.length ? null : Views.parsePoint(typed, Views.catalog?.trees || []);
    const options = matches.map((entry) => ({ id: entry.id, entry, text: entry.label,
      detail: [entry.units, Views.SOURCES[entry.source] || entry.source].filter(Boolean).join(" · ") }));
    if (parsed?.id && !matches.some((entry) => entry.id === parsed.id)) options.push({ id: parsed.id, text: parsed.text, detail: "fetch with fdp" });
    box.options = options;
    box.active = options.length ? 0 : -1;
    box.note.hidden = !(parsed?.refusal && !matches.length);
    box.note.className = "error";
    box.note.textContent = parsed?.refusal && !matches.length ? parsed.refusal : "";
    const failed = Views.catalogFailed === `${S.event}/${S.shot}`;
    if ((!Views.catalog || failed) && typed) {
      box.note.hidden = false;
      box.note.className = failed ? "error" : "hint";
      box.note.textContent = failed ? "Could not load the signal list. Type to try again." : "Loading the signal list…";
    }
    if (known) {
      box.note.hidden = false;
      box.note.className = "hint";
      box.note.textContent = `${known.label || known.id} is already in the view.`;
    }
    Views.showSuggestions();
  },

  showSuggestions() {
    const box = Views.box;
    box.list.replaceChildren(...box.options.map((option, k) => {
      const li = Object.assign(document.createElement("li"), { id: `add-row-option-${k}` });
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", String(k === box.active));
      li.append(Object.assign(document.createElement("span"), { textContent: option.text }),
        Object.assign(document.createElement("small"), { textContent: option.detail || "" }));
      li.addEventListener("pointerdown", (event) => {
        event.preventDefault(); // the input keeps focus
        Views.takeSuggestion(option);
      });
      return li;
    }));
    box.list.hidden = !box.options.length;
    box.input.setAttribute("aria-expanded", String(!box.list.hidden));
    if (box.active >= 0) box.input.setAttribute("aria-activedescendant", `add-row-option-${box.active}`);
    else box.input.removeAttribute("aria-activedescendant");
  },

  /** Add the chosen suggestion as a row of the view; the server fetches it, and its chip says
   * why if it cannot. */
  takeSuggestion(option) {
    const box = Views.box;
    if (!option || !box) {
      if (box && box.input.value.trim()) Views.suggest();
      return;
    }
    if (!Views.layout || !S.meta) return;
    Views.addRow(option.id, option.entry || null);
    box.input.value = "";
    box.options = [];
    box.note.hidden = true;
    Views.showSuggestions();
  },
};

/** The default layout of `event`: the store's rows in store order at today's heights, and no
 * dock; detachment's opens the dock with the TangTV inversion and EFIT, then the cameras
 * (plan D.B); q_min, poloidal beta and the MHD events (their subcategories too) open it with
 * the q profile (R2). */
function defaultLayout(event) {
  const family = String(event ?? "").split(".")[0];
  const dock = family === "detachment"
    ? { open: true, width: null, follow: "cursor", panels: [{ kind: "slice", id: "inversion", h: 300 }, { kind: "cameras" }] }
    : Views.Q_EVENTS.includes(family)
      ? { open: true, width: null, follow: "cursor", panels: [{ kind: "profile", id: "q_profile", h: 200 }] }
      : { open: false, width: null, follow: "cursor", panels: [] };
  return { v: 1, name: "Default", event, rows: [], other_store_rows: "end", dock };
}

/** Lay the view over the open shot's rows (after buildRows): store rows reordered with
 * `style.order` and hidden with `.view-hidden` (they stay children[0..n-1]); added rows as
 * canvases after them; the dock's panels. A new shot also clears the pins and the slices. */
function applyView(layout = null) {
  if (layout) Views.layout = Views.clone(layout);
  if (!S.meta) return;
  if (layout) Views.pending = true;
  if (!Views.layout || Views.layout.event !== S.event) Object.assign(Views, { layout: defaultLayout(S.event), pending: true, removed: [] });
  if (Views.shot !== S.shot) {
    Views.shot = S.shot;
    Slice.reset();
    Dock.shotChanged();
    // A new shot starts a new label undo: a row taken out before it comes back first.
    Views.removed = Views.removed.map((removed) => ({ ...removed, depth: 0 }));
    Views.selected = null;
    for (const chip of Views.chips.values()) chip.remove();
    Views.chips.clear();
  }
  const box = $("rows");
  const n = S.meta.rows.length;
  const keys = Views.storeKeys(S.meta.rows.map((row) => row.title));
  const plan = Views.arrange(Views.layout, keys);
  const named = plan.rows.some((spec, at) => spec.i !== at) || (Views.layout.rows || []).length > 0;
  for (const canvas of [...box.children].slice(n)) canvas.remove();
  Views.storeSpec = new Array(n).fill(null);
  const added = [];
  plan.rows.forEach((spec, at) => {
    if (spec.i != null) {
      const canvas = box.children[spec.i];
      Views.storeSpec[spec.i] = spec;
      canvas.style.order = named ? String(at) : "";
      canvas.classList.toggle("view-hidden", Boolean(spec.hidden));
      return;
    }
    const row = Views.makeAdded(spec);
    row.canvas.style.order = String(at);
    row.canvas.classList.toggle("view-hidden", row.hidden);
    box.append(row.canvas);
    added.push(row);
  });
  for (const [key, chip] of Views.chips) {
    if (!added.some((row) => row.key === key && !row.hidden)) {
      chip.remove();
      Views.chips.delete(key);
    }
  }
  Views.added = added;
  Views.selectRow(Views.selected); // the outline and chip follow the row (or go with it)
  const notes = Views.notes(plan);
  Activity.note("view-rows", [notes.missing, notes.fresh].filter(Boolean).join(". ") || null);
  if (Views.pending) {
    // A view was chosen: its dock replaces the one shown. Between shots the dock stays as the
    // reviewer left it (`d`, the resize handle and Follow are changes to the view).
    Views.pending = false;
    Dock.reopen = false;
    const dock = Views.layout.dock || {};
    Dock.panels = Views.clone(dock.panels || []);
    Dock.width = Number(dock.width) > 0 ? Number(dock.width) : null;
    Slice.follow = ["cursor", "A", "B"].includes(dock.follow) ? dock.follow : "cursor";
    if (Boolean(dock.open) !== Dock.open) Dock.show(Boolean(dock.open));
    else if (Dock.open) {
      Dock.layout();
      Dock.relayout();
    }
  } else if (Dock.reopen) {
    Dock.reopen = false; // a failed shot closed it (showNothing): this shot opens it again
    if (!Dock.open) Dock.show(true);
  }
  if (cutMargin()) fetchRows();
  sizeCanvases();
  render();
}

/** Each child of #rows as sizeRows lays it out: {h, hidden, fixed}. `h` is the view's height
 * in px or null for today's (IMAGE_H or TRACE_H); `fixed` keeps it from the fill-the-view
 * scale (an added row with no data on this shot is 28 px). Store rows first, in store order. */
function layoutRows() {
  const children = typeof document === "undefined" ? [] : [...($("rows")?.children || [])];
  const n = S.meta?.rows?.length || 0;
  return children.map((canvas, i) => {
    if (i < n) {
      const spec = Views.storeSpec[i];
      return { h: spec?.h || null, hidden: Boolean(spec?.hidden), fixed: false };
    }
    const added = Views.added.find((row) => row.canvas === canvas);
    const missing = added?.entry?.state === "error" && added.entry.error?.kind === "missing";
    return { h: missing ? Views.MISSING_H : added?.h || Views.HEIGHTS.M, hidden: Boolean(added?.hidden), fixed: missing };
  });
}

/** Draw the added rows (drawRows calls it): a trace through drawTrace with the signal's own
 * grid, a map as an image, the label's overlay and the pins over each; while one loads its
 * plot is hatched and its chip centred over it. */
function drawAddedRows() {
  for (const added of Views.added) {
    const canvas = added.canvas;
    if (added.hidden || !canvas.isConnected) {
      Views.placeChip(added, null);
      continue;
    }
    const [w, h] = [canvas.clientWidth, canvas.clientHeight];
    const g = context(canvas);
    const entry = added.entry;
    const data = entry?.data;
    const row = Views.rowLike(added);
    const map = added.draw === "map" && data && data.channels >= 2;
    const range = !data ? null : map ? [-0.5, data.channels - 0.5] : (added.range = Views.traceRange(data));
    const plot = w - GUTTER - RIGHT - Slice.margin;
    g.save();
    g.beginPath();
    g.rect(GUTTER, 0, plot, h);
    g.clip();
    if (map) {
      g.imageSmoothingEnabled = false;
      g.drawImage(Views.mapBitmap(added), px(data.t0), 0, px(data.t1) - px(data.t0), h - 1);
    } else if (data) drawTrace(g, row, data.values, range, w, h, data);
    if (!data || ["queued", "loading"].includes(entry?.state)) hatch(g, GUTTER, 0, plot, h);
    drawOverlay(g, w, h);
    drawPins(g, h);
    g.restore();
    drawGutter(g, map ? { ...row, y_units: "channel" } : row, range, h);
    g.fillStyle = T.rule;
    g.fillRect(0, h - 1, w, 1);
    Views.placeChip(added, Dock.entryNote(entry, added.title));
  }
}

/** Save: the current view under its name (Save), or under the name typed (Save as, or a
 * Default or preset layout). Needs a reviewer name. */
async function saveView(asNew = false) {
  if (!Views.named()) return;
  const own = Views.current && !Views.current.preset;
  const name = !asNew && own ? Views.current.name : Views.typedName();
  if (!name) return;
  const replacing = (Views.list || []).find((view) => !view.preset && view.name === name);
  const list = await Views.post({ op: "save", name, layout: Views.snapshot(name), ...(replacing ? { etag: replacing.etag } : {}) });
  if (!list) return;
  Views.adopt(name);
  Views.say(`Saved the view ${name}.`);
}

/** Rename the current view to the name typed. */
async function renameView() {
  if (!Views.named() || !Views.current || Views.current.preset) return;
  const name = Views.typedName();
  if (!name || name === Views.current.name) return;
  const list = await Views.post({ op: "rename", name: Views.current.name, new_name: name, etag: Views.current.etag });
  if (!list) return;
  const changed = Views.changed;
  Views.adopt(name);
  Views.changed = changed;
  Views.fillSelect();
  Views.say(`Renamed the view to ${name}.`);
}

/** Delete the current view; the page goes back to Default. */
async function deleteView() {
  if (!Views.named() || !Views.current || Views.current.preset) return;
  const name = Views.current.name;
  const list = await Views.post({ op: "delete", name, etag: Views.current.etag });
  if (!list) return;
  await Views.choose("");
  Views.say(`Deleted the view ${name}.`);
}

/** Reset to default: the store's rows in store order and the event's default dock (none, or
 * detachment's). Saved views are kept. */
function resetView() {
  return Views.choose("");
}

/** Edit mode: the dock opens (even when closed) with the rows table, the panels and the
 * footer; labeling keeps working beside it. */
function openEditor() {
  if (!S.meta) return;
  Views.showEdit(true);
  Views.renderEditor();
}

/** The signal picker in the dock, its search filled with `query` (Edit on an invalid chip
 * passes the id). The catalog is asked when it first opens for the shot. */
async function openPicker(query = "") {
  if (!S.meta) return;
  Views.showEdit(true);
  let [text, source] = [String(query), "ptdata"];
  if (text.startsWith("custom:")) text = text.slice(7);
  else if (text.startsWith("mds:")) [source, text] = [text.slice(4, text.indexOf(":", 4)), text.slice(text.indexOf(":", 4) + 1)];
  Views.picking = { query: text, source };
  Views.renderPicker();
  if (!Views.catalog || Views.catalogFor !== `${S.event}/${S.shot}`) {
    await Views.loadCatalog();
    if (Views.picking) Views.renderPicker();
  }
}

/** Search the catalog `entries` for `query` (every word in the id, label, group, units, node,
 * description or pointnames, a leading \\ ignored); suggested entries first. With no match, a name typed for `source` ("ptdata" or an
 * allowed tree of `trees`) is offered as {id, text} ("Add PTDATA DUSBRADIAL", custom:<NAME>;
 * "Add MDS \EFIT01::BETAN", mds:<TREE>:<node>) or refused before any request. Returns
 * {matches, offer, refusal}. */
function matchCatalog(entries, query, source = "ptdata", trees = []) {
  const text = String(query ?? "").trim();
  const words = text.toLowerCase().split(/\s+/).filter(Boolean).map((word) => word.replace(/^\\/, ""));
  const haystack = (entry) => [entry.id, entry.label, entry.group, entry.units, entry.where, entry.description,
    ...Views.pointKeys(entry)].join(" ").toLowerCase();
  const matches = !words.length ? (entries || []).slice()
    : (entries || []).filter((entry) => words.every((word) => haystack(entry).includes(word)));
  matches.sort((a, b) => Number(Boolean(b.suggested)) - Number(Boolean(a.suggested)));
  let offer = null, refusal = null;
  if (text && !matches.length && !/\s/.test(text)) {
    if (source === "ptdata") {
      if (Views.PTDATA.test(text)) offer = { id: `custom:${text.toUpperCase()}`, text: `Add PTDATA ${text.toUpperCase()}` };
      else refusal = `${text} is not a PTDATA point name: use a letter, then letters, digits or _, up to 12 characters.`;
    } else {
      const tree = String(source).toUpperCase();
      const bare = text.replace(/^\\/, "");
      const node = `\\${bare.includes("::") ? bare : `${tree}::${bare}`}`;
      const problem = Views.nodeProblem(tree, node, trees.map((name) => String(name).toUpperCase()));
      if (problem) refusal = problem;
      else offer = { id: `mds:${tree}:${node}`, text: `Add MDS ${node}` };
    }
  }
  return { matches, offer, refusal };
}

if (typeof module !== "undefined") {
  module.exports = { Views, defaultLayout, applyView, layoutRows, drawAddedRows, saveView, renameView, deleteView,
    resetView, openEditor, openPicker, matchCatalog };
}
