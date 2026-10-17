// Times across the rows (plan D.2): the hover and its rest time, pins A and B with their tabs,
// the values at those times, Copy values, the cut margins of image rows and the probe guides
// that link axes of one kind. A classic script loaded before app.js: it only defines names,
// and app.js, dock.js and views.js call them (api 11, workspace).
"use strict";

/** The times the dock and the margins follow. `hover` is the time under the pointer while it
 * is `inRows`; `rest` the last time it dwelt on (see `restAfter`); `pins` A and B are times
 * (never stored in views, cleared per shot); `follow` picks what the dock follows; `clicks`
 * the pins before each recent axis click (a double-click undoes them); `probe` a value on an
 * axis kind ({kind, v, from}) that draws guides; `margin` the cut margin's width in px. */
const Slice = {
  REST_MS: 120, // a pointer still this long has dwelt
  REST_PX: 3, // a move this small is still a dwell
  UNDO_MS: 500, // axis clicks this recent are undone by a double-click
  MARGIN_PX: [96, 120], // the cut margin's width, and its width on a wide time column
  WIDE_PX: 1300, // a time column at least this wide gets the wider margin
  NARROW_PX: 1100, // below this window width there is no margin (the dock covers the rows)
  SPREAD: 0.01, // a bucket wider than this share of its row's range reads as a range
  hover: null,
  inRows: false,
  rest: null,
  anchor: null,
  pins: { A: null, B: null },
  follow: "cursor",
  clicks: [],
  probe: null,
  margin: 0,
  frame: 0, // the pending redraw of the dock and the margins
  tabs: null, // the pin tabs in #pins, made on first use
  drag: null, // a tab being dragged: {pin, pointerId}

  // -- pure helpers (node tests)

  /** `pins` with a pin at `t`: A first, then B; with both set, the nearer one moves. */
  withPin(pins, t) {
    if (pins.A == null) return { ...pins, A: t };
    if (pins.B == null) return { ...pins, B: t };
    return Math.abs(t - pins.A) <= Math.abs(t - pins.B) ? { ...pins, A: t } : { ...pins, B: t };
  },

  /** `pins` without the pin nearest `t` (A on a tie); unchanged when there is none. */
  withoutNearest(pins, t) {
    const set = ["A", "B"].filter((name) => pins[name] != null);
    if (!set.length) return pins;
    const nearest = set.reduce((a, b) => (Math.abs(t - pins[b]) < Math.abs(t - pins[a]) ? b : a));
    return { ...pins, [nearest]: null };
  },

  /** The pins to restore when a double-click lands at `now`: the oldest snapshot of the last
   * `UNDO_MS`, or null when no axis click was that recent. */
  undone(clicks, now, ms = Slice.UNDO_MS) {
    const recent = clicks.filter((click) => now - click.at <= ms);
    return recent.length ? recent[0].pins : null;
  },

  /** The column of a grid {t0, t1, n} holding `t`, or -1 outside it. */
  bucket(grid, t) {
    if (!grid || !(grid.n > 0) || !(t >= grid.t0) || !(t < grid.t1)) return -1;
    return Math.min(grid.n - 1, Math.floor(((t - grid.t0) / (grid.t1 - grid.t0)) * grid.n));
  },

  /** Channel `c` of a min/max trace (float32 (2, C, n)) at column `j`: {lo, hi, mid}, or null
   * when the bucket holds no sample. */
  traceCell(values, n, channels, c, j) {
    if (j < 0) return null;
    const [lo, hi] = [values[c * n + j], values[(channels + c) * n + j]];
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
    return { lo, hi, mid: (lo + hi) / 2 };
  },

  /** The brightest bin of an image column (uint8 (n_y, n), bin 0 lowest) in [first, stop):
   * {bin, level}, or null when the column is dark (no data shows "–", never 0). For a modes
   * row (`k` colours, byte = level * k + mode) the level is `byte / k` and `mode` is kept. */
  peak(values, n, j, first, stop, k = 0) {
    if (j < 0) return null;
    let best = null;
    for (let b = first; b < stop; b++) {
      const byte = values[b * n + j];
      const level = k ? Math.floor(byte / k) : byte;
      if (level > 0 && (!best || level > best.level)) best = { bin: b, level, mode: k ? byte % k : null };
    }
    return best;
  },

  /** A number as the readout shows it: three figures, thousands grouped, a true minus. */
  number(v) {
    if (!Number.isFinite(v)) return "–";
    const a = Math.abs(v);
    const text = a !== 0 && (a >= 1e5 || a < 1e-3) ? v.toExponential(2)
      : a >= 100 ? Math.round(v).toLocaleString("en-US") : String(Number(v.toPrecision(3)));
    return text.replace(/^-/, "−");
  },

  /** A time in ms as the tabs show it: "2,140". */
  time: (t) => (Number.isFinite(t) ? Math.round(t).toLocaleString("en-US").replace(/^-/, "−") : "–"),

  /** A signed duration: "+470 ms", "−20 ms". */
  signed: (dt) => `${dt < 0 ? "−" : "+"}${Slice.time(Math.abs(dt))} ms`,

  /** A value for copying: plain digits and a hyphen minus, empty for no data. */
  plain: (v) => (Number.isFinite(v) ? String(Number(v.toPrecision(6))) : ""),

  // -- the page

  /** The cursor's time: the hover in the rows, else the last dwell (null before any). */
  cursorTime() {
    return Slice.inRows ? Slice.hover : Slice.rest;
  },

  /** Forget the shot's pins and pointer history (a shot opens). `follow` stays. */
  reset() {
    Object.assign(Slice, { hover: null, inRows: false, rest: null, anchor: null, pins: { A: null, B: null },
      clicks: [], probe: null, drag: null });
    placePinTabs();
  },

  /** Redraw what follows the pointer, the dock and the margins, once per frame; the rows and
   * lanes are left alone (render redraws them). */
  redraw() {
    if (Slice.frame || typeof requestAnimationFrame === "undefined") return;
    Slice.frame = requestAnimationFrame(() => {
      Slice.frame = 0;
      if (!S.meta) return;
      drawMargins();
      if (typeof drawDock === "function") drawDock();
    });
  },

  /** The time the hover line holds at while the pointer is out of the rows: the rest time, or
   * A; null with the dock closed or the pointer in the rows. */
  held() {
    if (Slice.inRows || typeof Dock === "undefined" || !Dock.open) return null;
    return Slice.rest ?? Slice.pins.A ?? null;
  },

  /** Draw the held hover line (`.held`, dotted) at `held()`; false when nothing is held. The
   * hover line is #hover-cursor on detachment, where #cursor is the video cursor. */
  showHeld() {
    const t = Slice.held();
    const line = $(S.video ? "hover-cursor" : "cursor");
    if (t == null || !S.meta || t < S.view[0] || t > S.view[1]) {
      line.classList.remove("held");
      return false;
    }
    showCursor($("axis-row").getBoundingClientRect().left + px(t), Boolean(S.video));
    line.classList.add("held");
    return true;
  },

  /** True when `clientX` is over the cut margin, right of the time plot. */
  inMargin(clientX) {
    if (!Slice.margin || !S.meta) return false;
    const axis = $("axis-row").getBoundingClientRect();
    const x = clientX - axis.left;
    return x >= axis.width - RIGHT - Slice.margin && x < axis.width - RIGHT;
  },

  /** Follow A, or the cursor again (`l`). */
  toggleFollow() {
    Slice.follow = Slice.follow === "cursor" ? "A" : "cursor";
    if (typeof Dock !== "undefined") Dock.followChanged();
    else Slice.redraw();
  },

  /** Set the pins and draw them on the rows, the lanes, the tabs, the dock and the margins. */
  setPins(pins) {
    Slice.pins = pins;
    render();
    placePinTabs();
    Slice.redraw();
  },

  // -- the rows the readout and the margins read

  /** Every time row in view order: the store's ({i, row}) and the added ones ({entry}), each
   * {key, title, units, canvas, hidden, kind: trace|image|modes|map, yKind}. Hidden rows are
   * left out unless `all`. */
  rows(all = false) {
    const out = [];
    const canvases = $("rows")?.children || [];
    const store = S.meta?.rows || [];
    const keys = typeof Views !== "undefined" ? Views.storeKeys(store.map((row) => row.title))
      : store.map((row) => `store:${row.title}`);
    store.forEach((row, i) => {
      const canvas = canvases[i];
      const kind = row.kind === "image" ? (row.modes ? "modes" : "image") : "trace";
      out.push({ key: keys[i], title: row.title, units: row.y_units || "", canvas, i, row, kind,
        yKind: kind === "trace" ? null : axisKind(row.y_units), hidden: Slice.hiddenCanvas(canvas) });
    });
    for (const added of (typeof Views !== "undefined" && Views.added) || []) {
      const units = added.entry?.signal?.units ?? added.units ?? "";
      const map = added.draw === "map";
      const title = added.entry?.signal?.label || added.title; // the server's label once it answered
      out.push({ key: added.key, title, units, canvas: added.canvas, entry: added.entry, added,
        kind: map ? "map" : "trace", yKind: map ? "channel" : null, hidden: Slice.hiddenCanvas(added.canvas) });
    }
    const order = (item) => Number(item.canvas?.style.order) || 0;
    const sorted = out.map((item, at) => [item, at]).sort((a, b) => order(a[0]) - order(b[0]) || a[1] - b[1])
      .map(([item]) => item);
    return all ? sorted : sorted.filter((item) => !item.hidden);
  },

  hiddenCanvas: (canvas) => !canvas || canvas.hidden || canvas.classList.contains("view-hidden"),

  /** A store row's grids that may hold `t`: the last window first, then the overview. */
  storeData(i, t) {
    for (const data of [S.data, S.overview]) {
      if (data?.rows?.[i] && Slice.bucket(data, t) >= 0) return data;
    }
    return null;
  },

  /** The y range an image-like row is drawn over: [lo, hi] at the row's bottom and top. */
  yRange(item) {
    if (item.kind === "map") return [-0.5, (item.entry?.data?.channels || 1) - 0.5];
    return item.row ? imageRange(item.row) : null;
  },

  /** The probe of the cut margin at `clientY`: the y value of the image row under it. */
  marginProbe(clientY) {
    for (const item of Slice.rows()) {
      if (item.kind === "trace" || !item.canvas) continue;
      const rect = item.canvas.getBoundingClientRect();
      if (clientY < rect.top || clientY >= rect.bottom || !item.yKind) continue;
      const [lo, hi] = Slice.yRange(item);
      return { kind: item.yKind, v: hi - ((clientY - rect.top) / Math.max(1, rect.height - 1)) * (hi - lo), from: "margin" };
    }
    return null;
  },

  /** One pin's tab. */
  pinTab(name, box) {
    const tab = Object.assign(document.createElement("div"), { className: `pin-tab pin-${name.toLowerCase()}`, hidden: true });
    tab.dataset.pin = name;
    const text = Object.assign(document.createElement("span"), { className: "pin-text" });
    text.title = `Pin ${name}: drag along the axis to move it`;
    const remove = Object.assign(document.createElement("button"), { type: "button", className: "pin-remove", textContent: "×", hidden: true });
    remove.setAttribute("aria-label", `Remove pin ${name}`);
    remove.addEventListener("click", () => Slice.setPins({ ...Slice.pins, [name]: null }));
    const copy = Object.assign(document.createElement("button"), { type: "button", className: "pin-copy", textContent: "Copy values" });
    copy.title = "Copy every row's values at A and B, tab-separated";
    copy.addEventListener("click", () => copyValues());
    tab.append(text, remove, copy);
    tab.addEventListener("pointerenter", () => { remove.hidden = false; });
    tab.addEventListener("pointerleave", () => { remove.hidden = true; });
    tab.addEventListener("wheel", (event) => onWheel(event, true), { passive: false });
    tab.addEventListener("pointerdown", (event) => {
      event.stopPropagation(); // the second click of a double-click lands here and pins nothing
      if (event.button !== 0 || event.target.closest("button")) return;
      event.preventDefault();
      tab.setPointerCapture(event.pointerId);
      Slice.drag = { pin: name, pointerId: event.pointerId };
    });
    tab.addEventListener("pointermove", (event) => {
      if (Slice.drag?.pin !== name || !S.meta) return;
      const t = clamp(timeAt(event.clientX), S.view[0], S.view[1]);
      Slice.setPins({ ...Slice.pins, [name]: t });
    });
    const end = () => { if (Slice.drag?.pin === name) Slice.drag = null; };
    tab.addEventListener("pointerup", end);
    tab.addEventListener("pointercancel", end);
    box.append(tab);
    return tab;
  },

  /** The number B−A subtracts for a value (channel `c` of a trace). */
  valueNumber(value, c = 0) {
    if (!value) return NaN;
    if (value.kind === "image") return value.y;
    if (value.kind === "modes") return value.n;
    return value.cells[c]?.mid ?? NaN;
  },

  /** A value as the readout shows it: {text, title}. A trace bucket wider than 1 % of its row's
   * range reads "min–max"; an image "peak 8.1 kHz · 182" (the 0–255 level, not power). */
  valueText(value, c = 0) {
    if (!value) return { text: "–", title: "" };
    if (value.kind === "image") {
      return { text: `peak ${Slice.number(value.y)} ${value.units} · ${value.level}`, title: "Brightest bin in the band, and its 0–255 level" };
    }
    if (value.kind === "modes") return { text: `n = ${String(value.n).replace("-", "−")}`, title: "" };
    const cell = value.cells[c];
    if (!cell) return { text: "–", title: "" };
    const span = value.range ? value.range[1] - value.range[0] : 0;
    if (span > 0 && cell.hi - cell.lo > Slice.SPREAD * span) {
      return { text: `${Slice.number(cell.lo)}–${Slice.number(cell.hi)}`, title: "Range at this zoom; zoom in for one sample" };
    }
    return { text: Slice.number(cell.mid), title: "" };
  },

  /** The values in a dialog, selected for Ctrl+C, when the clipboard is refused. */
  copyDialog(text) {
    let dialog = $("copy-values");
    if (!dialog) {
      dialog = Object.assign(document.createElement("dialog"), { id: "copy-values" });
      dialog.setAttribute("aria-labelledby", "copy-values-title");
      const title = Object.assign(document.createElement("h2"), { id: "copy-values-title", textContent: "Values at A and B" });
      const hint = Object.assign(document.createElement("p"), { className: "hint",
        textContent: "The browser did not allow copying. The text is selected: press Ctrl+C." });
      const area = Object.assign(document.createElement("textarea"), { rows: 12, cols: 60, readOnly: true, spellcheck: false });
      const form = Object.assign(document.createElement("form"), { method: "dialog" });
      form.append(Object.assign(document.createElement("button"), { textContent: "Close" }));
      dialog.append(title, hint, area, form);
      document.body.append(dialog);
    }
    const area = dialog.querySelector("textarea");
    area.value = text;
    if (!dialog.open) dialog.showModal();
    area.focus();
    area.select();
  },

  /** The canvas over the rows that carries the margins and the probe guides (#cut-margins, in
   * #row-ui, never in #rows); its press goes through to the rows. */
  marginCanvas() {
    let canvas = $("cut-margins");
    if (!canvas) {
      canvas = Object.assign(document.createElement("canvas"), { id: "cut-margins", className: "cut-margin" });
      canvas.setAttribute("aria-hidden", "true");
      Object.assign(canvas.style, { position: "fixed", pointerEvents: "none" });
      ($("row-ui") || document.body).append(canvas);
    }
    return canvas;
  },

  /** One row's cut in the margin at `x0`, `width` px wide and `h` tall. */
  drawCut(g, item, times, x0, width, h) {
    g.fillStyle = T.panel;
    g.fillRect(x0, 0, width, h);
    g.fillStyle = T.rule;
    g.fillRect(x0, 0, 1, h);
    g.fillRect(x0, h - 1, width, 1);
    const [lo, hi] = Slice.yRange(item);
    const yAt = (v) => ((hi - v) / (hi - lo)) * (h - 1);
    const [left, span] = [x0 + 4, width - 8];
    g.textAlign = "left";
    if (item.kind === "modes") {
      g.fillStyle = T.ink;
      times.forEach(([t, name], line) => {
        const value = valueAt(item, t);
        const n = value ? String(value.n).replace("-", "−") : "–";
        const text = name === "cursor" ? `n = ${n} at ${Slice.time(t)}` : `${name}: n = ${n}`;
        g.fillText(text, left, 14 + line * 14, span);
      });
      return;
    }
    let scale = [0, 255];
    if (item.kind === "map") scale = item.added?.range || [0, 1];
    g.fillStyle = T.muted;
    g.textAlign = "center";
    if (h >= 40) g.fillText(item.kind === "map" ? (item.units || "value") : "8-bit level", x0 + width / 2, h - 4, span);
    for (const [t, name] of times) {
      const line = Slice.cutAt(item, t);
      if (!line) continue;
      g.save();
      g.strokeStyle = name === "cursor" ? T.muted : T.ink;
      g.lineWidth = name === "cursor" ? 1 : 2;
      g.setLineDash(name === "B" ? [6, 4] : []);
      g.beginPath();
      let pen = false;
      for (const [y, v] of line) {
        if (!Number.isFinite(v)) {
          pen = false;
          continue;
        }
        const x = left + clamp((v - scale[0]) / (scale[1] - scale[0] || 1), 0, 1) * span;
        if (pen) g.lineTo(x, yAt(y));
        else g.moveTo(x, yAt(y));
        pen = true;
      }
      g.stroke();
      g.restore();
    }
  },

  /** A dotted --muted guide across a row at `v` on its y axis. */
  drawGuide(g, item, v, w, h) {
    const [lo, hi] = Slice.yRange(item) || [];
    if (!(hi > lo) || v < lo || v > hi) return;
    const y = Math.round(((hi - v) / (hi - lo)) * (h - 1)) + 0.5;
    g.save();
    g.strokeStyle = T.muted;
    g.lineWidth = 1;
    g.setLineDash([1, 3]);
    g.beginPath();
    g.moveTo(GUTTER, y);
    g.lineTo(w, y);
    g.stroke();
    g.restore();
  },

  /** An image-like row's column at `t` as [[y, value], ...] bottom up: a store image's levels
   * (0–255) over its band, a map row's channel values (the bucket midpoints). */
  cutAt(item, t) {
    if (item.row) {
      const data = Slice.storeData(item.i, t);
      const j = Slice.bucket(data, t);
      if (j < 0) return null;
      const [first, stop] = bandBins(item.row);
      const values = data.rows[item.i];
      const out = [];
      for (let b = first; b < stop; b++) out.push([item.row.y0 + b * item.row.dy, values[b * data.n + j]]);
      return out;
    }
    const value = valueAt(item, t);
    return value ? value.cells.map((cell, c) => [c, cell ? cell.mid : NaN]) : null;
  },
};

/** The time the dock and the margins follow: the hover in the rows, else the rest time or A;
 * or the pin `follow` names when it is set. */
function sliceTime() {
  const pinned = Slice.follow !== "cursor" ? Slice.pins[Slice.follow] : null;
  if (pinned != null) return pinned;
  return Slice.inRows ? Slice.hover : Slice.rest ?? Slice.pins.A;
}

/** The rest-time rule, from event timestamps only: a move more than `REST_PX` from `anchor`
 * commits the anchor's time as the rest when the pointer stayed there `REST_MS` or longer,
 * and moves the anchor. Returns {anchor, rest}; `rest` is null when it does not change. */
function restAfter(anchor, x, t, ts) {
  if (anchor && Math.abs(x - anchor.x) <= Slice.REST_PX) return { anchor, rest: null };
  const rest = anchor && ts - anchor.ts >= Slice.REST_MS ? anchor.t : null;
  return { anchor: { x, t, ts }, rest };
}

/** The window's pointermove, first: the hover, whether it is in the rows, the rest time and a
 * probe from the cut margin. Never makes a request; redraws the dock and the margins. */
function notePointer(event) {
  if (!S.meta) return;
  const axis = $("axis-row").getBoundingClientRect();
  const x = event.clientX - axis.left;
  const over = Boolean(event.target?.closest?.("#rows, #axis-row, .track, #row-ui"));
  const margin = over && Slice.inMargin(event.clientX);
  const inPlot = over && !margin && x >= GUTTER && x <= axis.width - RIGHT - Slice.margin;
  const t = timeAt(event.clientX);
  if (inPlot || Slice.anchor) {
    const next = restAfter(Slice.anchor, x, t, event.timeStamp);
    if (next.rest != null) Slice.rest = next.rest;
    Slice.anchor = inPlot ? next.anchor : null;
  }
  Slice.inRows = inPlot;
  if (inPlot) Slice.hover = t;
  const probe = margin ? Slice.marginProbe(event.clientY) : null;
  if (probe || Slice.probe?.from === "margin") Slice.probe = probe;
  Slice.redraw();
}

/** `p`: a pin at `t` (A, then B, then the nearer moves). */
function pinAt(t) {
  if (!Number.isFinite(t) || !S.meta) return;
  Slice.setPins(Slice.withPin(Slice.pins, t));
}

/** `x`: remove the pin nearest `t`. */
function unpinNear(t) {
  if (!S.meta || (Slice.pins.A == null && Slice.pins.B == null)) return;
  Slice.setPins(Slice.withoutNearest(Slice.pins, Number.isFinite(t) ? t : middle()));
}

/** A click (no drag) on the time axis: remember the pins, so a double-click can undo it, then
 * pin. endDrag calls it after the detachment seek and before clickMask. */
function axisClickPin(t) {
  Slice.clicks.push({ pins: { ...Slice.pins }, at: performance.now() });
  Slice.clicks = Slice.clicks.filter((click) => performance.now() - click.at <= Slice.UNDO_MS);
  pinAt(t);
}

/** A double-click: put back the pins from before its clicks (the oldest snapshot of the last
 * 500 ms) and forget them; the caller then fits. */
function undoAxisClicks() {
  const pins = Slice.undone(Slice.clicks, performance.now());
  Slice.clicks = [];
  if (pins) Slice.setPins(pins);
}

/** Draw the pins over a row or a lane `h` px tall, inside its clip: 2 px --ink, A solid, B
 * dashed. */
function drawPins(g, h) {
  const { A, B } = Slice.pins;
  if (A == null && B == null) return;
  g.save();
  g.strokeStyle = T.ink;
  g.lineWidth = 2;
  for (const [t, dash] of [[A, []], [B, [6, 4]]]) {
    if (t == null || t < S.view[0] || t > S.view[1]) continue;
    const x = Math.round(px(t));
    g.setLineDash(dash);
    g.beginPath();
    g.moveTo(x, 0);
    g.lineTo(x, h);
    g.stroke();
  }
  g.restore();
}

/** The tabs of pins A and B in #pins, on the time axis: "A 2,140" and "B 2,610 · +470 ms",
 * a × on hover, and Copy values on B (or on A while it is the only pin). Called on every
 * render. A tab's press never reaches the rows; its wheel zooms; a drag moves its pin. */
function placePinTabs() {
  const box = typeof document === "undefined" ? null : $("pins");
  if (!box) return;
  if (!Slice.tabs) Slice.tabs = { A: Slice.pinTab("A", box), B: Slice.pinTab("B", box) };
  const axis = $("axis-row").getBoundingClientRect();
  const { A, B } = Slice.pins;
  for (const name of ["A", "B"]) {
    const tab = Slice.tabs[name];
    const t = Slice.pins[name];
    const x = t == null || !S.meta ? NaN : px(t);
    tab.hidden = !(x >= GUTTER && x <= axis.width - RIGHT - Slice.margin);
    if (tab.hidden) continue;
    tab.querySelector(".pin-text").textContent = name === "B" && A != null
      ? `B ${Slice.time(B)} · ${Slice.signed(B - A)}` : `${name} ${Slice.time(t)}`;
    tab.querySelector(".pin-copy").hidden = !(name === "B" || B == null);
    Object.assign(tab.style, { left: `${axis.left + x}px`, top: `${axis.top}px` });
  }
}

/** A row's value at `t`: for a trace {kind: "trace", cells: [{lo, hi, mid} | null per channel],
 * range}; for an image {kind: "image", y, units, level}; for a modes row {kind: "modes", n,
 * level}; for a map {kind: "map", cells}; null when the row has no data at `t`. `item` is one
 * of `Slice.rows()`. Store rows read the min/max bucket of S.data (or the overview); added
 * rows their own grid. */
function valueAt(item, t) {
  if (!item || !Number.isFinite(t)) return null;
  if (item.row) {
    const data = Slice.storeData(item.i, t);
    if (!data) return null;
    const j = Slice.bucket(data, t);
    const values = data.rows[item.i];
    const row = item.row;
    if (row.kind === "image") {
      const [first, stop] = bandBins(row);
      const k = row.modes ? row.modes.colours.length : 0;
      const best = Slice.peak(values, data.n, j, first, stop, k);
      if (!best) return null;
      if (row.modes) return { kind: "modes", n: row.modes.n[best.mode], level: best.level };
      return { kind: "image", y: row.y0 + best.bin * row.dy, units: row.y_units || "", level: best.level };
    }
    const cells = Array.from({ length: row.n_channels }, (_, c) => Slice.traceCell(values, data.n, row.n_channels, c, j));
    if (!cells.some(Boolean)) return null;
    return { kind: "trace", cells, range: data === S.data ? traceRange(row, values) : null };
  }
  const data = item.entry?.data;
  const j = Slice.bucket(data, t);
  if (j < 0) return null;
  const cells = Array.from({ length: data.channels }, (_, c) => Slice.traceCell(data.values, data.n, data.channels, c, j));
  if (!cells.some(Boolean)) return null;
  return { kind: item.kind === "map" ? "map" : "trace", cells, range: item.added?.range || null };
}

/** Copy every row's values at A and B (hidden and scrolled-out rows, store and added; one
 * line per channel) as tab-separated columns row, units, A, B, B−A. When the clipboard
 * refuses, the table opens in a dialog with its text selected. */
async function copyValues() {
  const { A, B } = Slice.pins;
  const lines = [["row", "units", "A", "B", "B−A"].join("\t"),
    ["t (ms)", "ms", Slice.plain(A), Slice.plain(B), A != null && B != null ? Slice.plain(B - A) : ""].join("\t")];
  for (const item of Slice.rows(true)) {
    const [a, b] = [valueAt(item, A), valueAt(item, B)];
    const channels = item.kind === "trace" || item.kind === "map"
      ? (item.row?.n_channels ?? item.entry?.data?.channels ?? 1) : 1;
    const legend = item.row?.legend || item.entry?.signal?.legend || [];
    const units = item.kind === "image" ? item.units : item.kind === "modes" ? "n" : item.units;
    for (let c = 0; c < channels; c++) {
      const [va, vb] = [Slice.valueNumber(a, c), Slice.valueNumber(b, c)];
      const name = channels > 1 ? `${item.title} / ${legend[c] ?? `channel ${c + 1}`}` : item.title;
      lines.push([name, units, Slice.plain(va), Slice.plain(vb), Slice.plain(vb - va)].join("\t"));
    }
  }
  const text = `${lines.join("\n")}\n`;
  try {
    await navigator.clipboard.writeText(text);
    if (typeof Activity !== "undefined") Activity.note("copy", `Copied the values of ${lines.length - 2} lines at A and B.`);
  } catch {
    Slice.copyDialog(text);
  }
}

/** Size the cut margin for the dock and the rows: 120 px on a time column at least 1,300 px
 * wide, else 96 px, while the dock is open and an image row shows; 0 otherwise and below
 * 1,100 px of window. Returns true when it changed (the caller refetches the rows once). */
function cutMargin() {
  const open = typeof Dock !== "undefined" && Dock.open;
  const images = open && Slice.rows().some((item) => item.kind !== "trace");
  const wide = $("axis-row").clientWidth >= Slice.WIDE_PX;
  const want = open && images && window.innerWidth >= Slice.NARROW_PX ? Slice.MARGIN_PX[wide ? 1 : 0] : 0;
  if (want === Slice.margin) return false;
  Slice.margin = want;
  return true;
}

/** Draw every visible image row's cut in its margin, on the row's own y axis (horizontal: the
 * 8-bit level; a map row its own units): the slice time 1 px --muted, A 2 px --ink solid, B
 * 2 px --ink dashed; a modes row says "n = 2 at 2,388". Then the probe's guides: a dotted
 * --muted line across each visible row of the probe's kind. A hover redraws only this. */
function drawMargins() {
  const canvas = Slice.marginCanvas();
  const probe = Slice.probe;
  if (!S.meta || (!Slice.margin && !probe)) {
    canvas.hidden = true;
    return;
  }
  const axis = $("axis-row").getBoundingClientRect();
  const top = $("top").getBoundingClientRect();
  const rows = $("rows").getBoundingClientRect();
  const [y0, y1] = [Math.max(top.top, rows.top), Math.min(top.bottom, rows.bottom, axis.top)];
  canvas.hidden = y1 <= y0;
  if (canvas.hidden) return;
  const [w, h] = [axis.width - RIGHT, y1 - y0];
  Object.assign(canvas.style, { left: `${axis.left}px`, top: `${y0}px`, width: `${w}px`, height: `${h}px` });
  const ratio = window.devicePixelRatio || 1;
  const [cw, ch] = [Math.round(w * ratio), Math.round(h * ratio)];
  if (canvas.width !== cw || canvas.height !== ch) [canvas.width, canvas.height] = [cw, ch];
  const g = context(canvas);
  g.font = FONT;
  const t = sliceTime();
  const times = [[t, "cursor"], [Slice.pins.A, "A"], [Slice.pins.B, "B"]]
    .filter(([time, name], k) => time != null && (k > 0 || (time !== Slice.pins.A && time !== Slice.pins.B)) && name);
  for (const item of Slice.rows()) {
    if (!item.canvas) continue;
    const rect = item.canvas.getBoundingClientRect();
    const [ry, rh] = [rect.top - y0, rect.height];
    if (ry + rh <= 0 || ry >= h) continue;
    g.save();
    g.translate(0, ry);
    if (Slice.margin && item.kind !== "trace") Slice.drawCut(g, item, times, w - Slice.margin, Slice.margin, rh);
    if (probe && item.yKind === probe.kind) Slice.drawGuide(g, item, probe.v, w, rh);
    g.restore();
  }
}

/** The kind of an axis from its units, or null: time_ms, psi_n, rho, R_m, channel, freq_khz.
 * Kinds never convert into each other (psi_N and rho stay apart: the rhovn trap). */
function axisKind(units) {
  const u = String(units ?? "").trim().toLowerCase().replace(/\s+/g, " ");
  if (!u) return null;
  if (/^(k|m)?hz$|^f(requency)? \((k|m)?hz\)$|^freq_khz$/.test(u)) return "freq_khz";
  if (/^r ?(\(m\)|\[m\])$|^r_m$|^major radius/.test(u)) return "R_m";
  if (/^(channel|chord|ch)s?( \(index\)| index| number)?$/.test(u)) return "channel";
  if (/^(ψ|psi)[ _]?n$|^normali[sz]ed (poloidal )?flux|^psi_n$/.test(u)) return "psi_n";
  if (/^(ρ|rho)([ _]?(n|tor|vn))?$/.test(u)) return "rho";
  if (/^(ms|time \(ms\)|t \(ms\)|time_ms)$/.test(u)) return "time_ms";
  return null;
}

if (typeof module !== "undefined") {
  module.exports = { Slice, sliceTime, notePointer, restAfter, pinAt, unpinNear, axisClickPin, undoAxisClicks,
    drawPins, placePinTabs, valueAt, copyValues, cutMargin, drawMargins, axisKind };
}
