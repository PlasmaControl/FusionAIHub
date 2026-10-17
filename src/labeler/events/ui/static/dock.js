// The dock (plan D.1, D.2, D.B, R2): the right-hand column of 2-D and secondary views beside
// the rows. Profile panels with the cursor, A and B; the Values readout; the TangTV inversion
// and the EFIT equilibrium on R-Z axes; the detachment cameras; any row popped out of the
// stack. Every panel has the same frame (title, note, ×) and a crosshair linked to the rows'
// cursor. A classic script loaded before app.js: it only defines names, and app.js and
// views.js call them (api 11, workspace).
"use strict";

/** The dock's state. `panels` are the view's ({kind: "profile", id, h} | {kind: "values"} |
 * {kind: "slice", id: "inversion" | "equilibrium", h} | {kind: "cameras"} | {kind: "row",
 * id: <row key>, h}); `nodes` their elements by key; `kinds` what /api/slices said for
 * `kindsAt` ("<event>/<shot>"); `slices` the slice images and equilibria by kind, `asked` and
 * `want` per kind (one request in flight per kind, the latest time wins). */
const Dock = {
  WIDTH: "clamp(320px, 24vw, 480px)", // the default width; a dragged width is in px
  HANDLE_PX: 6,
  MIN_PX: 280,
  KEEP_TIME_PX: 820, // a stored width never leaves the time column narrower than this (review D2B I1)
  PROFILE_H: 200,
  SLICE_H: 300,
  ROW_H: 160, // a row popped out of the stack
  // Overlays on the slices, from the page's palette (Okabe-Ito), each over a dark halo so it
  // reads on a light panel and on a dark greyscale frame alike (R2): the separatrix and flux
  // surfaces orange, X-points sky blue, strike points bluish green.
  OVERLAY: { boundary: "#e69f00", xpoint: "#56b4e9", strike: "#009e73", halo: "rgba(0, 0, 0, 0.7)" },
  NEAR_MS: 40, // a slice farther than this from the time asked is not shown
  KEEP_IMAGES: 24, // slice images kept per kind (scrubbing back is free)
  KEEP_ASKED: 2000, // asked times remembered per kind, with the slice the server answered
  Q_MARKS: [[1, "q = 1"], [1.5, "q = 3/2"], [2, "q = 2"]],
  X_TITLES: { psi_n: "ψN", rho: "ρ", R_m: "R (m)", channel: "channel", freq_khz: "kHz", time_ms: "ms" },
  open: false,
  reopen: false, // a failed shot closed it (showNothing): the next shot opens it again
  width: null,
  editing: false,
  panels: [],
  nodes: new Map(),
  kinds: null,
  kindsAt: "",
  kindsError: null,
  slices: new Map(), // kind -> {items: Map(slice t_ms -> {image, record} | {eq}), asked, want, busy, error}
  cameras: null, // the camera nodes moved out of #video-panel: [[node, parent, next], ...]
  expanded: new Set(), // Values lines opened to show each channel
  frame: 0,
  headParts: null, // the header's parts, made once
  menuOpen: false, // the Add menu shows

  // -- pure helpers (node tests)

  /** Fit an R-Z box into `w` x `h` px with the aspect kept, centred: {k (px per m), left, top,
   * r0, z1}. */
  fitRZ([r0, r1], [z0, z1], w, h) {
    const k = Math.max(0, Math.min(w / Math.max(1e-9, r1 - r0), h / Math.max(1e-9, z1 - z0)));
    return { k, left: (w - (r1 - r0) * k) / 2, top: (h - (z1 - z0) * k) / 2, r0, z1 };
  },

  /** A point (r, z) in metres to [x, y] px of a fit: z up, r right. */
  toPixel(fit, r, z) {
    return [fit.left + (r - fit.r0) * fit.k, fit.top + (fit.z1 - z) * fit.k];
  },

  /** What to ask /api/slice for at `t`, given a /api/slices kind ({n_t, t_range_ms}): {t_ms},
   * `t` rounded to the integer ms the server takes, or null when `t` is more than `NEAR_MS`
   * outside the range or the kind has no slice. The server answers with the slice nearest
   * t_ms, wherever its slices fall (EFIT times are not evenly spaced). */
  pickSlice(kind, t) {
    const n = kind?.n_t || 0;
    const [t0, t1] = kind?.t_range_ms || [];
    if (!n || !Number.isFinite(t) || !Number.isFinite(t0) || t < t0 - Dock.NEAR_MS || t > t1 + Dock.NEAR_MS) return null;
    return { t_ms: Math.round(t) };
  },

  /** The slice time the server answers for `ms`, from its earlier answers in `cache`: null when
   * it has none within NEAR_MS, undefined when not known (or no longer cached). A slice is the
   * nearest one to every time between a time it answered and its own time (its cell is an
   * interval), so those times are known without asking. */
  answered(cache, ms) {
    const direct = cache.asked.get(ms);
    if (direct === null) return null;
    if (direct != null && cache.items.has(direct)) return direct;
    for (const [at, t] of cache.asked) {
      if (t != null && Math.min(at, t) <= ms && ms <= Math.max(at, t) && cache.items.has(t)) return t;
    }
    return undefined;
  },

  /** The union of x ranges by axis kind: {kind: [lo, hi]} from [[kind, [lo, hi]], ...]. */
  union(ranges) {
    const out = {};
    for (const [kind, [lo, hi]] of ranges) {
      const was = out[kind];
      out[kind] = was ? [Math.min(was[0], lo), Math.max(was[1], hi)] : [lo, hi];
    }
    return out;
  },

  /** The q marks inside [lo, hi] (q may be signed): [[q, text], ...]. */
  qMarks(lo, hi, sign = 1) {
    return Dock.Q_MARKS.map(([q, text]) => [sign * q, text]).filter(([q]) => q >= lo && q <= hi);
  },

  /** y at `v` on the polyline (xs, ys), linear between its finite points (xs either way up);
   * null outside it. */
  interpolate(xs, ys, v) {
    for (let k = 1; k < xs.length; k++) {
      const [x0, x1, y0, y1] = [xs[k - 1], xs[k], ys[k - 1], ys[k]];
      if (![x0, x1, y0, y1].every(Number.isFinite)) continue;
      if ((v - x0) * (v - x1) > 0) continue;
      return x1 === x0 ? y0 : y0 + ((v - x0) / (x1 - x0)) * (y1 - y0);
    }
    return null;
  },

  /** The R-Z box of an equilibrium: its own, else the wall's or the boundary's; null when it
   * has no geometry. */
  extentOf(eq) {
    if (eq?.r_range && eq?.z_range) return [eq.r_range, eq.z_range];
    const points = (eq?.wall?.length ? eq.wall : eq?.boundary) || [];
    if (points.length < 2) return null;
    const rs = points.map((p) => p[0]), zs = points.map((p) => p[1]);
    return [[Math.min(...rs), Math.max(...rs)], [Math.min(...zs), Math.max(...zs)]];
  },

  // -- the page

  /** True when the slices follow the video: detachment with camera frames to seek. With no
   * frames the video's time never moves, so the slices follow the cursor instead (R2: a shot
   * without camera frames held EFIT at the store's first ms, outside EFIT's times, and the
   * panel said "Loading…" for ever). */
  videoLed() {
    return Boolean(S.video?.times?.length);
  },

  /** The time the slice views follow: the video cursor while the video leads (#cursor, which
   * stays where it was put), else `sliceTime()`; a followed pin wins in both. */
  time() {
    const pinned = Slice.follow !== "cursor" ? Slice.pins[Slice.follow] : null;
    if (pinned != null) return pinned;
    return Dock.videoLed() ? S.video.time ?? null : sliceTime();
  },

  /** What the slice time follows, for the headers: "cursor", "video", "A" or "B". */
  following() {
    if (Slice.follow !== "cursor" && Slice.pins[Slice.follow] != null) return Slice.follow;
    return Dock.videoLed() ? "video" : "cursor";
  },

  /** The dock's width as the body grid takes it (`--dock-w`), handle included. A stored width
   * (dragged, or from a view saved on another screen) is capped in CSS, so the cap follows the
   * window: at most 60vw and at most the window less `KEEP_TIME_PX`, but never below the default
   * width (review D2B I1: a 1,100 px width saved at 1920 left a 1 px plot at 1280). */
  gridWidth() {
    if (!Dock.open) return "0px";
    const handle = Dock.HANDLE_PX;
    if (!Dock.width) return `calc(${Dock.WIDTH} + ${handle}px)`;
    const cap = `max(calc(${Dock.WIDTH} + ${handle}px), min(60vw, calc(100vw - ${Dock.KEEP_TIME_PX}px)))`;
    return `min(${Dock.width + handle}px, ${cap})`;
  },

  /** The cap of `gridWidth` in px for this window, handle included (the drag stops there). */
  maxGridPx(width = window.innerWidth) {
    const base = Math.min(480, Math.max(320, width * 0.24)) + Dock.HANDLE_PX;
    return Math.max(base, Math.min(width * 0.6, width - Dock.KEEP_TIME_PX));
  },

  /** Lay the page out again after the dock opened, closed or changed width: the cut margin,
   * and one row refetch when it changed (the #top ResizeObserver refetches for the width). */
  relayout() {
    document.body.style.setProperty("--dock-w", Dock.gridWidth());
    if (cutMargin()) fetchRows();
    sizeCanvases();
    render();
    Slice.redraw();
    if (Activity.parts) Activity.place(); // a strip under the bar moves clear of the dock
  },

  /** Redraw the dock alone on the next frame. */
  redraw() {
    Slice.redraw();
  },

  /** A panel's key in `nodes`. */
  key: (panel) => `${panel.kind}:${panel.id ?? ""}`,

  /** Make or reuse the elements of the view's panels in #dock-panels, in order; the cameras
   * panel adopts the camera nodes, which go back to #video-panel when it goes. An empty dock
   * says how to fill it. */
  layout() {
    const box = $("dock-panels");
    if (!box) return;
    const keep = new Set(Dock.panels.map(Dock.key));
    for (const [key, node] of Dock.nodes) {
      if (keep.has(key)) continue;
      if (key === "cameras:") Dock.returnCameras();
      node.el.remove();
      Dock.nodes.delete(key);
    }
    const nodes = Dock.panels.map((panel) => {
      const key = Dock.key(panel);
      let node = Dock.nodes.get(key);
      if (!node) {
        node = panel.kind === "profile" ? panelProfile(panel) : panel.kind === "values" ? panelValues(panel)
          : panel.kind === "slice" ? Dock.panelSlice(panel) : panel.kind === "cameras" ? Dock.panelCameras(panel)
          : panel.kind === "row" ? Dock.panelRow(panel) : null;
        if (!node) return null;
        const made = node;
        made.close.addEventListener("click", () => Dock.removePanel(made.panel));
        Dock.nodes.set(key, node);
      }
      node.panel = panel;
      if (node.canvas) node.canvas.style.height = `${panel.h || Dock.heightOf(panel)}px`;
      return node.el;
    }).filter(Boolean);
    if (!nodes.length) {
      nodes.push(Object.assign(document.createElement("p"), { className: "dock-empty",
        textContent: "Empty. Use Add for a profile, a slice or a row." }));
    }
    box.replaceChildren(...nodes);
    if (Dock.panels.some((panel) => panel.kind === "cameras")) Dock.adoptCameras();
  },

  /** A panel's default height in px. */
  heightOf: (panel) => (panel.kind === "slice" ? Dock.SLICE_H : panel.kind === "row" ? Dock.ROW_H : Dock.PROFILE_H),

  /** A panel's frame, the same for every kind: a header with a title, a note and × (remove
   * from the dock), then the body. */
  frameFor(kind, title) {
    const el = Object.assign(document.createElement("section"), { className: `dock-panel dock-${kind}` });
    const head = Object.assign(document.createElement("header"), { className: "dock-panel-head" });
    const name = Object.assign(document.createElement("span"), { className: "dock-panel-title", textContent: title });
    const note = Object.assign(document.createElement("span"), { className: "dock-panel-note" });
    const close = Object.assign(document.createElement("button"), { type: "button", className: "dock-panel-close",
      textContent: "×", title: kind === "row" ? "Put back in the rows" : "Remove from the dock" });
    close.setAttribute("aria-label", close.title);
    head.append(name, note, close);
    el.append(head);
    return { el, name, note, close };
  },

  /** Remove `panel` from the dock (its ×); a popped-out row goes back to the stack. */
  removePanel(panel) {
    const key = Dock.key(panel);
    const panels = Dock.panels.filter((item) => Dock.key(item) !== key);
    if (typeof Views === "undefined" || !Views.layout) {
      Dock.panels = panels;
      Dock.layout();
      return Dock.redraw();
    }
    Views.setPanels(panels);
    if (panel.kind === "row") Views.setRow(panel.id, { hidden: false });
  },

  /** Pop the row `key` out of the stack into the dock (the dock opens). */
  popRow(key) {
    if (typeof Views === "undefined" || !Views.layout || !key) return;
    if (!Dock.panels.some((panel) => panel.kind === "row" && panel.id === key)) {
      Views.setRow(key, { hidden: true });
      Views.setPanels([...Dock.panels, { kind: "row", id: key, h: Dock.ROW_H }]);
    }
    if (!Dock.open) Dock.show(true);
  },

  /** Add a panel unless the dock has it (Add): a profile, Values, a slice or the cameras. */
  addPanel(panel) {
    if (Dock.panels.some((item) => Dock.key(item) === Dock.key(panel))) return;
    const sized = panel.kind === "values" || panel.kind === "cameras" ? panel : { ...panel, h: Dock.heightOf(panel) };
    if (typeof Views !== "undefined" && Views.layout) Views.setPanels([...Dock.panels, sized]);
    else {
      Dock.panels = [...Dock.panels, sized];
      Dock.layout();
    }
    if (!Dock.open) Dock.show(true);
  },

  /** A status line inside a panel (loading, an error and its button), or none. */
  chip(node, note, act) {
    if (!note) {
      node.chip?.remove();
      node.chip = null;
      return;
    }
    if (!node.chip) {
      node.chip = Object.assign(document.createElement("div"), { className: "row-chip dock-chip" });
      node.chip.append(document.createElement("span"),
        Object.assign(document.createElement("button"), { type: "button" }));
      node.chip.lastChild.addEventListener("click", () => node.chipAct?.());
      node.el.append(node.chip);
    }
    node.chip.className = `row-chip dock-chip${note.tone ? ` ${note.tone}` : ""}`;
    node.chip.firstChild.textContent = note.text;
    node.chip.lastChild.textContent = note.button || "";
    node.chip.lastChild.hidden = !note.button;
    node.chipAct = act;
    let more = node.chip.querySelector(".failure-detail");
    if (note.detail && !more) {
      more = Object.assign(document.createElement("details"), { className: "failure-detail" });
      more.append(Object.assign(document.createElement("summary"), { textContent: "details" }), document.createElement("pre"));
      node.chip.firstChild.after(more);
    }
    if (more && !note.detail) more.remove();
    else if (more && more.lastChild.textContent !== note.detail) more.lastChild.textContent = note.detail;
  },

  /** A canvas sized to its box in device pixels, and its context in CSS px. */
  paint(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const [w, h] = [canvas.clientWidth * ratio, canvas.clientHeight * ratio].map(Math.round);
    if (canvas.width !== w || canvas.height !== h) [canvas.width, canvas.height] = [w, h];
    return context(canvas);
  },

  /** Draw a profile panel: the slice at the dock's time (1 px --muted), A (2 px --ink) and B
   * (dashed), the q marks on q, the probe, and a header naming the slices used. */
  drawProfile(node, xShared) {
    const panel = node.panel;
    const entry = S.shot == null ? null : loadProfile(S.shot, panel.id, { name: panel.id, source: "disk" }, Dock.redraw);
    const label = entry?.signal?.label || Views.catalogLabel?.(panel.id) || panel.id;
    node.name.textContent = label;
    const g = Dock.paint(node.canvas);
    const [w, h] = [node.canvas.clientWidth, node.canvas.clientHeight];
    Dock.chip(node, Dock.entryNote(entry, label), () => (entry?.state === "error" && entry.error?.kind === "missing"
      ? Dock.removePanel(panel) : Dock.entryAct(entry)));
    const profile = entry?.state === "done" ? entry.data : null;
    if (!profile) {
      node.note.textContent = "";
      return;
    }
    const { y } = Profiles.ranges(profile);
    const x = xShared[profile.x_kind] || Profiles.ranges(profile).x;
    const box = { left: 40, right: w - 8, top: 8, bottom: h - 22 };
    const X = (v) => box.left + ((v - x[0]) / (x[1] - x[0])) * (box.right - box.left);
    const Y = (v) => box.bottom - ((v - y[0]) / (y[1] - y[0])) * (box.bottom - box.top);
    node.scale = { x, X, Y, box, kind: profile.x_kind };
    Dock.axes(g, box, x, y, X, Y, `${Dock.X_TITLES[profile.x_kind] || profile.x_kind}`, profile.units || "");
    if (panel.id === "q_profile" || /^q\b/i.test(label)) {
      const median = Profiles.span(profile.y, 0.5, 0.5)?.[0] ?? 1;
      g.save();
      g.strokeStyle = T.muted;
      g.fillStyle = T.muted;
      g.setLineDash([2, 3]);
      g.textAlign = "right";
      for (const [q, text] of Dock.qMarks(y[0], y[1], Math.sign(median) || 1)) {
        g.beginPath();
        g.moveTo(box.left, Math.round(Y(q)) + 0.5);
        g.lineTo(box.right, Math.round(Y(q)) + 0.5);
        g.stroke();
        g.fillText(text, box.right - 2, Y(q) - 3);
      }
      g.restore();
    }
    const t = Dock.time();
    const { A, B } = Slice.pins;
    const at = (time) => (time == null ? null : profileAt(profile, time));
    const curves = [[at(t), "cursor"], [at(A), "A"], [at(B), "B"]];
    g.save();
    g.beginPath();
    g.rect(box.left, box.top - 4, box.right - box.left, box.bottom - box.top + 8);
    g.clip();
    for (const [slice, name] of curves) {
      if (!slice || (name === "cursor" && (t === A || t === B))) continue;
      g.strokeStyle = name === "cursor" ? T.muted : T.ink;
      g.lineWidth = name === "cursor" ? 1 : 2;
      g.setLineDash(name === "B" ? [6, 4] : []);
      g.beginPath();
      let pen = false;
      for (let k = 0; k < slice.x.length; k++) {
        if (!Number.isFinite(slice.x[k]) || !Number.isFinite(slice.y[k])) {
          pen = false;
          continue;
        }
        if (pen) g.lineTo(X(slice.x[k]), Y(slice.y[k]));
        else g.moveTo(X(slice.x[k]), Y(slice.y[k]));
        pen = true;
      }
      g.stroke();
    }
    g.restore();
    Dock.probeLine(g, node.scale);
    Dock.probeReadout(g, node.scale, at(t) || at(A), profile.units || "");
    node.note.textContent = Dock.sliceNote(profile, at(t), at(A), at(B), t);
  },

  /** The crosshair's readout on a profile, wherever the probe comes from (this panel, another
   * of its axis, an image row's cut margin): a dot on the curve at the probe and
   * "ψN 0.42 · 1.83" in the top right corner. */
  probeReadout(g, scale, slice, units) {
    const probe = Slice.probe;
    if (!probe || !scale || probe.kind !== scale.kind || probe.v < scale.x[0] || probe.v > scale.x[1]) return;
    const y = slice ? Dock.interpolate(slice.x, slice.y, probe.v) : null;
    const [name, unit] = Dock.READ[scale.kind] || [scale.kind, ""];
    const text = `${name} ${Slice.number(probe.v)}${unit}${y == null ? "" : ` · ${Slice.number(y)}${units ? ` ${units}` : ""}`}`;
    if (y != null) {
      const [px0, py0] = [scale.X(probe.v), scale.Y(y)];
      if (py0 >= scale.box.top - 4 && py0 <= scale.box.bottom + 4) {
        g.save();
        g.fillStyle = T.panel;
        g.beginPath();
        g.arc(px0, py0, 4.5, 0, 2 * Math.PI);
        g.fill();
        g.fillStyle = T.ink;
        g.beginPath();
        g.arc(px0, py0, 3, 0, 2 * Math.PI);
        g.fill();
        g.restore();
      }
    }
    Dock.tag(g, text, scale.box.right - 2, scale.box.top + 1);
  },

  /** The axis names of a readout: [name, unit]. */
  READ: { psi_n: ["ψN", ""], rho: ["ρ", ""], R_m: ["R", " m"], channel: ["channel", ""], freq_khz: ["", " kHz"], time_ms: ["", " ms"] },

  /** A short text on a --panel box, its right edge at `x` and its top at `y`. */
  tag(g, text, x, y) {
    g.save();
    g.font = FONT;
    g.textAlign = "right";
    const width = g.measureText(text).width;
    g.globalAlpha = 0.9;
    g.fillStyle = T.panel;
    g.fillRect(x - width - 4, y, width + 6, 14);
    g.globalAlpha = 1;
    g.fillStyle = T.ink;
    g.fillText(text, x, y + 11);
    g.restore();
  },

  /** The header note of a profile: "EFIT 2,380 ms (8 ms from cursor)", or "A and B share one
   * EFIT slice". The diagnostic comes from the label and source ("q (EFIT02)" from park,
   * "Thomson core Te (R from EFIT02)" from the corpus): Thomson, ZIPFIT and ECE before EFIT,
   * which may only map their radii. */
  sliceNote(profile, here, a, b, t) {
    const source = `${profile.label || ""} ${profile.source || ""}`;
    const name = /thomson|\bts\b/i.test(source) ? "TS" : /zipfit/i.test(source) ? "ZIPFIT"
      : /\bece\b/i.test(source) ? "ECE" : /efit/i.test(source) ? "EFIT" : "slice";
    if (a && b && a.i === b.i) return `A and B share one ${name} slice`;
    if (!here) return "";
    const off = Math.round(Math.abs(here.t - t));
    return `${name} ${Slice.time(here.t)} ms${off ? ` (${Slice.time(off)} ms from ${Dock.following()})` : ""}`;
  },

  /** Axes inside `box`: ticks, the x title (its kind) and the y units. */
  axes(g, box, x, y, X, Y, xTitle, yTitle) {
    g.font = FONT;
    g.fillStyle = T.rule;
    g.fillRect(box.left, box.bottom, box.right - box.left, 1);
    g.fillRect(box.left, box.top, 1, box.bottom - box.top);
    g.fillStyle = T.muted;
    g.textAlign = "center";
    for (const [v, text] of ticks(x[0], x[1], (box.right - box.left) / 60)) g.fillText(text, X(v), box.bottom + 12);
    g.textAlign = "right";
    for (const [v, text] of ticks(y[0], y[1], (box.bottom - box.top) / 30)) g.fillText(text, box.left - 4, Y(v) + 4);
    g.fillText(xTitle, box.right, box.bottom + 21);
    g.textAlign = "left";
    if (yTitle) g.fillText(yTitle, box.left + 4, box.top + 10);
  },

  /** The probe's vertical guide in a panel of its kind. */
  probeLine(g, scale) {
    const probe = Slice.probe;
    if (!probe || !scale || probe.kind !== scale.kind || probe.v < scale.x[0] || probe.v > scale.x[1]) return;
    const x = Math.round(scale.X(probe.v)) + 0.5;
    g.save();
    g.strokeStyle = T.muted;
    g.setLineDash([1, 3]);
    g.beginPath();
    g.moveTo(x, scale.box.top);
    g.lineTo(x, scale.box.bottom);
    g.stroke();
    g.restore();
  },

  /** A profile or signal entry's status line: rowNote's copy for loading, queued and errors;
   * null once it is done. */
  entryNote(entry, name) {
    if (!entry || entry.state === "done" || entry.state === "idle") return null;
    const facts = { name, shot: entry.shot, source: entry.facts.source, tree: entry.facts.tree,
      elapsed_s: entry.elapsed_s, queue_position: entry.queue_position, message: entry.error?.message };
    if (entry.state === "stopped") return { text: "Stopped waiting. The server is still fetching.", button: "Retry", tone: "" };
    if (entry.state === "error") {
      const kind = entry.error?.kind;
      const note = rowNote(kind, kind === "invalid" ? { ...facts, source: entry.id.startsWith("mds:") ? "mds" : "ptdata" } : facts)
        || { text: Activity.sentence(entry.error?.message || "it failed"), button: "Retry", tone: "error" };
      return entry.error?.detail ? { ...note, detail: entry.error.detail } : note; // the server's detail, behind a toggle
    }
    return rowNote(entry.state === "queued" ? "queued" : "loading", facts);
  },

  /** A status line's button: Stop waiting, Retry (an auth failure checks the fdp login
   * first), Remove from view or Edit. */
  async entryAct(entry) {
    if (!entry) return;
    if (entry.state === "queued" || entry.state === "loading") return Signals.stop(entry);
    if (entry.state === "error" && entry.error?.kind === "missing") return Views.removeKey?.(entry.id);
    if (entry.state === "error" && entry.error?.kind === "invalid") return openPicker(entry.id);
    if (entry.state === "error" && entry.error?.kind === "auth" && !(await Activity.fetchReady())) return;
    Signals.retry(entry);
  },

  /** Draw the Values panel: Cursor, A, B and B−A for every visible time row, a t (ms) line
   * first; a multi-channel row folds into one line that opens to its channels. */
  drawValues(node) {
    const { A, B } = Slice.pins;
    const t = Slice.cursorTime();
    // On detachment this column follows the hover while the slices follow the video (D2B M6).
    const heading = node.el.querySelector("thead th:nth-child(2)");
    const named = Dock.videoLed() ? "Hover" : "Cursor";
    if (heading && heading.textContent !== named) heading.textContent = named;
    const lines = [["t", "t (ms)", [t, A, B].map(Slice.time).concat(A != null && B != null ? Slice.signed(B - A).replace(/ ms$/, "") : "–"), ""]];
    const docked = new Set(Dock.panels.filter((panel) => panel.kind === "row").map((panel) => panel.id));
    for (const item of Slice.rows(true).filter((found) => !found.hidden || docked.has(found.key))) {
      const values = [valueAt(item, t), valueAt(item, A), valueAt(item, B)];
      const channels = item.kind === "trace" || item.kind === "map" ? (item.row?.n_channels ?? item.entry?.data?.channels ?? 1) : 1;
      const legend = item.row?.legend || item.entry?.signal?.legend || [];
      const cells = (c) => {
        const shown = values.map((value) => Slice.valueText(value, c));
        const [a, b] = [Slice.valueNumber(values[1], c), Slice.valueNumber(values[2], c)];
        return [...shown.map((cell) => cell.text), Slice.number(b - a)].map((text, k) => [text, shown[k]?.title || ""]);
      };
      if (channels > 1) {
        const open = Dock.expanded.has(item.key);
        lines.push([item.key, item.title, [`${channels} channels`, "", "", ""], open ? "open" : "closed"]);
        if (open) for (let c = 0; c < channels; c++) lines.push([`${item.key}#${c}`, `  ${legend[c] ?? `channel ${c + 1}`}`, cells(c), ""]);
      } else lines.push([item.key, item.title, cells(0), ""]);
    }
    const body = node.body;
    const rows = [...body.children];
    lines.forEach(([key, name, cells, fold], k) => {
      let tr = rows[k];
      if (!tr) {
        tr = document.createElement("tr");
        const th = Object.assign(document.createElement("th"), { scope: "row" });
        tr.append(th, ...[0, 1, 2, 3].map(() => document.createElement("td")));
        body.append(tr);
      }
      tr.dataset.key = key;
      const th = tr.firstChild;
      if (fold) {
        let button = th.querySelector("button");
        if (!button) {
          button = Object.assign(document.createElement("button"), { type: "button", className: "fold" });
          button.addEventListener("click", () => {
            const id = tr.dataset.key;
            if (Dock.expanded.has(id)) Dock.expanded.delete(id);
            else Dock.expanded.add(id);
            Dock.redraw();
          });
          th.replaceChildren(button);
        }
        button.textContent = `${fold === "open" ? "▾" : "▸"} ${name}`;
        button.setAttribute("aria-expanded", String(fold === "open"));
      } else if (th.textContent !== name || th.firstChild?.tagName === "BUTTON") th.textContent = name;
      cells.forEach((cell, c) => {
        const [text, title] = Array.isArray(cell) ? cell : [cell, ""];
        const td = tr.children[c + 1];
        if (td.textContent !== text) td.textContent = text;
        if (td.title !== title) td.title = title;
        td.classList.toggle("wrap", text.includes(" · "));
      });
    });
    while (body.children.length > lines.length) body.lastChild.remove();
  },

  // -- slices (plan D.B): stream E's /api/slices and /api/slice

  /** What the open shot has on disk, asked once per shot: /api/slices. */
  async loadKinds() {
    const at = `${S.event}/${S.shot}`;
    if (Dock.kindsAt === at) return;
    Dock.kindsAt = at;
    Dock.kinds = null;
    Dock.slices = new Map();
    const ticket = S.ticket;
    try {
      const body = await (await api(`/api/slices?event=${encodeURIComponent(S.event)}&shot=${S.shot}`)).json();
      if (ticket !== S.ticket) return Dock.kindsStale(at);
      Dock.kinds = body.kinds || [];
    } catch (error) {
      if (ticket !== S.ticket) return Dock.kindsStale(at);
      Dock.kinds = [];
      Dock.kindsError = error.message;
    }
    Dock.redraw();
  },

  /** An answer for a shot opened since: when it is the same shot again, ask anew (a reopen
   * mid-request would otherwise leave the slices on "Loading…"). */
  kindsStale(at) {
    if (Dock.kindsAt !== at) return;
    Dock.kindsAt = "";
    Dock.redraw();
  },

  kind: (name) => Dock.kinds?.find((kind) => kind.kind === name) || null,

  /** The cache of one slice kind: `items` by the slice time the server answered with (oldest
   * use first), `asked` each ms asked -> that slice time (null: none within NEAR_MS), `failed`
   * the ms whose request last failed for a reason other than 404 (asked again after a move). */
  sliceCache(kind) {
    let cache = Dock.slices.get(kind);
    if (!cache) {
      cache = { items: new Map(), asked: new Map(), want: null, busy: false, error: null, failed: null };
      Dock.slices.set(kind, cache);
    }
    return cache;
  },

  /** Ask /api/slice for `kind` at `pick.t_ms` unless the answer is known; one request in flight
   * per kind, and the latest pick is asked next. The answer is cached by the t_ms it carries
   * (X-Slice, or the equilibrium's own), never by the time asked. */
  async askSlice(kind, pick) {
    const cache = Dock.sliceCache(kind);
    if (pick.t_ms === cache.failed) return; // it failed just now; the redraw does not retry it
    cache.want = pick.t_ms;
    if (cache.busy || Dock.answered(cache, pick.t_ms) !== undefined) return;
    cache.busy = true;
    const ticket = S.ticket;
    try {
      while (cache.want != null && Dock.answered(cache, cache.want) === undefined && ticket === S.ticket) {
        const ms = cache.want;
        const query = `event=${encodeURIComponent(S.event)}&shot=${S.shot}&kind=${kind}&t_ms=${ms}`;
        try {
          const response = await api(`/api/slice?${query}`);
          const item = kind === "equilibrium" ? { eq: await response.json() } : await Dock.decode(response);
          if (ticket !== S.ticket) return;
          const t = Number(item.eq?.t_ms ?? item.record?.t_ms);
          if (!Number.isFinite(t)) throw new Error("the slice came without its time");
          cache.items.delete(t);
          cache.items.set(t, item);
          cache.asked.set(ms, t);
          cache.error = null;
          cache.failed = null;
          Dock.trim(cache);
        } catch (error) {
          if (ticket !== S.ticket) return;
          if (error.status === 404) {
            cache.error = null;
            cache.asked.set(ms, null); // the server has no slice near ms: a lasting answer
          } else {
            // A transient failure (5xx, the network) is not an answer (review D2B M4): say so,
            // leave ms unasked and stop; the next pointer or video move asks again.
            cache.error = error.message;
            cache.failed = ms;
            cache.want = null;
          }
        }
        Dock.redraw();
      }
    } finally {
      cache.busy = false;
    }
  },

  /** Keep `KEEP_IMAGES` slices (the least recently used go) and `KEEP_ASKED` asked times. */
  trim(cache) {
    while (cache.items.size > Dock.KEEP_IMAGES) {
      const [t, gone] = cache.items.entries().next().value;
      gone.image?.close?.();
      cache.items.delete(t);
    }
    while (cache.asked.size > Dock.KEEP_ASKED) cache.asked.delete(cache.asked.keys().next().value);
  },

  /** A /api/slice inversion answer: its decoded grayscale image and its X-Slice record. */
  async decode(response) {
    const record = JSON.parse(response.headers.get("X-Slice") || "null");
    if (!record?.r_range || !record?.z_range) throw new Error("the slice came without its R-Z extent");
    const blob = await response.blob();
    if (typeof createImageBitmap === "function") return { image: await createImageBitmap(blob), record };
    const url = URL.createObjectURL(blob);
    try {
      const image = new Image();
      image.src = url;
      await image.decode();
      return { image, record };
    } finally {
      URL.revokeObjectURL(url);
    }
  },

  /** The slice panel: the TangTV inversion with the equilibrium over it, or the equilibrium
   * alone when the shot has no inversion. Its crosshair reads R and Z and probes R across the
   * panels and rows on an R axis (ECE). */
  panelSlice(panel) {
    const frame = Dock.frameFor("slice", "TangTV inversion");
    const canvas = Object.assign(document.createElement("canvas"), { className: "dock-plot" });
    canvas.setAttribute("aria-label", "Poloidal cross-section at the slice time");
    frame.el.append(canvas);
    const node = { ...frame, canvas, panel, rz: null, hover: null };
    canvas.addEventListener("pointermove", (event) => {
      const rz = node.rz;
      if (!rz) return;
      const box = canvas.getBoundingClientRect();
      const [x, y] = [event.clientX - box.left, event.clientY - box.top];
      const r = rz.extent[0][0] + (x - rz.x0) / rz.fit.k;
      const z = rz.extent[1][1] - (y - rz.y0) / rz.fit.k;
      const inside = x >= rz.x0 && x <= rz.x1 && y >= rz.y0 && y <= rz.y1;
      node.hover = inside ? { r, z } : null;
      Slice.probe = inside ? { kind: "R_m", v: r, from: "dock" } : Slice.probe?.from === "dock" ? null : Slice.probe;
      Slice.redraw();
    });
    canvas.addEventListener("pointerleave", () => {
      node.hover = null;
      if (Slice.probe?.from === "dock") Slice.probe = null;
      Slice.redraw();
    });
    return node;
  },

  /** A slice kind's state at `t` for a panel: {state, item, error}. `state` is "shown" (the
   * slice the server answers for t), "near" (another cached slice while t's loads),
   * "loading", "none" (no slice within NEAR_MS), "out" (t outside the kind's times) or "error"
   * (the request for t failed; the next move asks again). Asks for t when needed. */
  sliceState(name, kind, t, has) {
    const pick = Dock.pickSlice(kind, t);
    if (!pick) return { state: "out", item: null };
    Dock.askSlice(name, pick);
    const cache = Dock.sliceCache(name);
    const answer = Dock.answered(cache, pick.t_ms);
    if (answer === null) return { state: "none", item: null };
    const item = Dock.itemAt(name, pick.t_ms, has);
    if (answer !== undefined && item) return { state: "shown", item };
    if (cache.failed === pick.t_ms) return { state: "error", item, error: cache.error };
    return { state: item ? "near" : "loading", item };
  },

  /** Draw a slice panel at the dock's time. Every state says what it is: loading only while a
   * request is out, else what is missing and the times that exist. */
  drawSlice(node) {
    if (S.shot == null) return;
    Dock.loadKinds();
    const g = Dock.paint(node.canvas);
    const [w, h] = [node.canvas.clientWidth, node.canvas.clientHeight];
    const t = Dock.time();
    const inversion = node.panel.id !== "equilibrium" ? Dock.kind("inversion") : null;
    const equilibrium = Dock.kind("equilibrium");
    node.name.textContent = inversion ? "TangTV inversion and EFIT" : equilibrium ? equilibrium.label || "EFIT equilibrium" : "TangTV inversion";
    node.rz = null;
    g.font = FONT;
    g.fillStyle = T.muted;
    const say = (text, note = "") => {
      node.note.textContent = note;
      g.fillText(text, 8, 18, w - 16);
    };
    if (!Dock.kinds) return say("Loading…");
    if (!inversion && !equilibrium) {
      return say(Dock.kindsError ? "Could not list this shot's slices." : "No TangTV inversion or EFIT for this shot.", Dock.kindsError || "");
    }
    if (t == null) return say(Dock.videoLed() ? "Move the video to show a slice." : "Point at the rows to show a slice.");
    const parts = [];
    if (inversion) parts.push({ name: "TangTV", kind: inversion, ...Dock.sliceState("inversion", inversion, t, (item) => item.image) });
    if (equilibrium) parts.push({ name: "EFIT", kind: equilibrium, ...Dock.sliceState("equilibrium", equilibrium, t, (item) => item.eq) });
    const image = inversion ? parts[0].item : null;
    const eq = (equilibrium ? parts[parts.length - 1].item : null)?.eq || null;
    // Each slice's offset from the time followed sits by its own time (review D2B M11).
    const from = (shown) => {
      const off = Math.round(Math.abs(shown - t));
      return off ? ` (${Slice.time(off)} ms from ${Dock.following()})` : "";
    };
    const notes = parts.map((part) => {
      const shown = part.name === "TangTV" ? image?.record : eq;
      const label = part.name === "EFIT" ? eq?.source || "EFIT" : "TangTV";
      if (shown && part.state !== "error") return `${label} ${Slice.time(shown.t_ms)} ms${from(shown.t_ms)}`;
      const [t0, t1] = part.kind.t_range_ms || [];
      if (part.state === "out") return `${label} ${Slice.time(t0)}–${Slice.time(t1)} ms`;
      if (part.state === "none") return `no ${label} within ${Dock.NEAR_MS} ms`;
      if (part.state === "error") return `${label} failed: ${part.error}`;
      return `${label} loading`;
    });
    const box = { left: 34, right: w - 6, top: 6, bottom: h - 20 };
    const extent = image ? [image.record.r_range, image.record.z_range] : eq ? Dock.extentOf(eq) : null;
    if (!extent) {
      const loading = parts.some((part) => part.state === "loading");
      const out = parts.every((part) => part.state === "out");
      return say(loading ? "Loading…" : out ? `No slice at ${Slice.time(t)} ms. Move into the range above.` : `No slice at ${Slice.time(t)} ms.`,
        notes.join(" · "));
    }
    const fit = Dock.fitRZ(extent[0], extent[1], box.right - box.left, box.bottom - box.top);
    fit.left += box.left;
    fit.top += box.top;
    const P = (r, z) => Dock.toPixel(fit, r, z);
    const [x0, y0] = P(extent[0][0], extent[1][1]);
    const [x1, y1] = P(extent[0][1], extent[1][0]);
    node.rz = { fit, extent, x0, y0, x1, y1 };
    g.save();
    g.beginPath();
    g.rect(x0, y0, x1 - x0, y1 - y0);
    g.clip();
    if (image) {
      g.imageSmoothingEnabled = true;
      g.drawImage(image.image, x0, y0, x1 - x0, y1 - y0);
    }
    if (eq) Dock.drawEquilibrium(g, eq, P, !image);
    Dock.crosshairRZ(g, node, P, [x0, y0, x1, y1]);
    g.restore();
    g.fillStyle = T.rule;
    g.fillRect(x0, y1, x1 - x0, 1);
    g.fillRect(x0 - 1, y0, 1, y1 - y0);
    g.fillStyle = T.muted;
    g.textAlign = "center";
    const R = (v) => P(v, extent[1][0])[0];
    const Z = (v) => P(extent[0][0], v)[1];
    for (const [v, text] of ticks(extent[0][0], extent[0][1], (x1 - x0) / 50)) g.fillText(text, R(v), y1 + 12);
    g.textAlign = "right";
    for (const [v, text] of ticks(extent[1][0], extent[1][1], (y1 - y0) / 30)) g.fillText(text, x0 - 4, Z(v) + 4);
    g.fillText("R (m)", x1, y1 + 21 > h ? h - 2 : y1 + 21);
    g.textAlign = "left";
    g.fillText("Z (m)", 2, box.top + 8);
    if (image) {
      const r = image.record;
      node.el.title = `${r.source || ""}${r.units ? ` · ${r.vmin}–${r.vmax} ${r.units}` : ""}`.trim();
    }
    if (node.hover) Dock.tag(g, `R ${Slice.number(node.hover.r)} m · Z ${Slice.number(node.hover.z)} m`, x1 - 2, y0 + 1);
    node.note.textContent = notes.join(" · ");
  },

  /** The R-Z crosshair: dotted guides through the pointer on this panel, or a guide at R when
   * another panel or row probes R (ECE). */
  crosshairRZ(g, node, P, [x0, y0, x1, y1]) {
    const probe = Slice.probe;
    const r = node.hover ? node.hover.r : probe?.kind === "R_m" ? probe.v : null;
    if (r == null) return;
    const [x] = P(r, 0);
    const y = node.hover ? P(0, node.hover.z)[1] : null;
    g.save();
    g.setLineDash([2, 3]);
    for (const [colour, width] of [[Dock.OVERLAY.halo, 3], ["#ffffff", 1]]) {
      g.strokeStyle = colour;
      g.lineWidth = width;
      g.beginPath();
      g.moveTo(Math.round(x) + 0.5, y0);
      g.lineTo(Math.round(x) + 0.5, y1);
      if (y != null) {
        g.moveTo(x0, Math.round(y) + 0.5);
        g.lineTo(x1, Math.round(y) + 0.5);
      }
      g.stroke();
    }
    g.restore();
  },

  /** The cached item of `kind` for the asked `ms`: the slice the server answers for it when
   * known; null when it has none within NEAR_MS; else, while the answer loads, the cached one
   * nearest `ms` that `has` (the panel keeps the last image). The item shown is the most
   * recently used, so trimming never closes it. */
  itemAt(kind, ms, has) {
    const cache = Dock.sliceCache(kind);
    const t = Dock.answered(cache, ms);
    if (t === null) return null;
    let [item, at] = [t === undefined ? null : cache.items.get(t), t];
    if (!item || !has(item)) {
      let gap = Infinity;
      item = null;
      for (const [k, found] of cache.items) {
        if (has(found) && Math.abs(k - ms) < gap) [item, at, gap] = [found, k, Math.abs(k - ms)];
      }
    }
    if (item) {
      cache.items.delete(at);
      cache.items.set(at, item);
    }
    return item;
  },

  /** The equilibrium over a slice: the wall in --ink on a --panel halo; the separatrix
   * (boundary) and, with `levels`, the flux surfaces in orange, X-points (×) in sky blue and
   * strike points (●) in bluish green (`OVERLAY`), each on a dark halo so it reads over a
   * light panel and a dark frame alike. */
  drawEquilibrium(g, eq, P, levels) {
    const colours = Dock.OVERLAY;
    const path = (points, close = false) => {
      g.beginPath();
      points.forEach(([r, z], k) => {
        const [x, y] = P(r, z);
        if (k) g.lineTo(x, y);
        else g.moveTo(x, y);
      });
      if (close) g.closePath();
    };
    const stroke = (points, width, colour, halo, close) => {
      if (!points?.length) return;
      path(points, close);
      g.strokeStyle = halo;
      g.lineWidth = width + 2;
      g.stroke();
      g.strokeStyle = colour;
      g.lineWidth = width;
      g.stroke();
    };
    g.save();
    g.lineJoin = "round";
    if (levels) {
      g.globalAlpha = 0.55;
      g.strokeStyle = colours.boundary;
      g.lineWidth = 1;
      for (const level of eq.levels || []) {
        if (level.psi_n === 1) continue;
        for (const points of level.paths || []) {
          path(points);
          g.stroke();
        }
      }
      g.globalAlpha = 1;
    }
    stroke(eq.wall, 1, T.ink, T.panel, true);
    stroke(eq.boundary, 2, colours.boundary, colours.halo, true);
    for (const [r, z] of eq.xpoints || []) {
      const [x, y] = P(r, z);
      for (const [width, colour] of [[4.5, colours.halo], [2, colours.xpoint]]) {
        g.lineWidth = width;
        g.strokeStyle = colour;
        g.beginPath();
        g.moveTo(x - 5, y - 5);
        g.lineTo(x + 5, y + 5);
        g.moveTo(x - 5, y + 5);
        g.lineTo(x + 5, y - 5);
        g.stroke();
      }
    }
    for (const [r, z] of eq.strikes || []) {
      const [x, y] = P(r, z);
      for (const [radius, colour] of [[5, colours.halo], [3.5, colours.strike]]) {
        g.fillStyle = colour;
        g.beginPath();
        g.arc(x, y, radius, 0, 2 * Math.PI);
        g.fill();
      }
    }
    g.restore();
  },

  // -- cameras (plan D.B): today's camera cards, moved into the dock

  /** The cameras panel: it takes #video-controls, #video-context and #video-cameras out of
   * #video-panel while it shows, so buildVideo, seekVideo and loadVideoFrame work unchanged. */
  panelCameras(panel) {
    const frame = Dock.frameFor("cameras", "Cameras");
    frame.body = Object.assign(document.createElement("div"), { className: "dock-cameras" });
    frame.el.append(frame.body);
    return { ...frame, panel };
  },

  adoptCameras() {
    const node = Dock.nodes.get("cameras:");
    if (!node || Dock.cameras) return;
    const ids = ["video-controls", "video-context", "video-cameras"];
    Dock.cameras = ids.map($).filter(Boolean).map((el) => [el, el.parentNode, el.nextSibling]);
    node.body.append(...Dock.cameras.map(([el]) => el));
    if (typeof fitVideoPanel === "function") fitVideoPanel(); // the top panel keeps the strips, or hides
  },

  /** Put the camera nodes back where they were. */
  returnCameras() {
    for (const [el, parent, next] of (Dock.cameras || []).slice().reverse()) {
      parent.insertBefore(el, next && next.parentNode === parent ? next : null);
    }
    Dock.cameras = null;
    if (typeof fitVideoPanel === "function") fitVideoPanel();
  },

  drawCameras(node) {
    const time = S.video?.time;
    node.name.textContent = "Cameras";
    node.note.textContent = !S.video ? "No cameras for this shot." : !Dock.videoLed() ? "No camera frames for this shot."
      : Number.isFinite(time) ? `video ${Slice.time(time)} ms` : "";
  },

  // -- a row popped out of the stack (R2)

  /** A popped-out row's panel: the row over the view's time, with the rows' cursor and pins;
   * the pointer over it moves the rows' cursor (notePointer reads `pointerTime`). */
  panelRow(panel) {
    const frame = Dock.frameFor("row", panel.id);
    const canvas = Object.assign(document.createElement("canvas"), { className: "dock-plot" });
    canvas.setAttribute("aria-label", "A row over the view's time");
    frame.el.append(canvas);
    return { ...frame, canvas, panel, scale: null };
  },

  /** The time under the pointer on a popped-out row's plot, or null off one. */
  pointerTime(event) {
    const canvas = event?.target?.closest?.(".dock-row canvas");
    if (!canvas) return null;
    let scale = null;
    for (const node of Dock.nodes.values()) if (node.canvas === canvas) scale = node.scale;
    if (!scale) return null;
    const x = event.clientX - canvas.getBoundingClientRect().left;
    if (x < scale.box.left || x > scale.box.right) return null;
    return scale.t0 + ((x - scale.box.left) / (scale.box.right - scale.box.left)) * (scale.t1 - scale.t0);
  },

  /** Draw a popped-out row: an image row's picture, a map, or a trace's channels over the
   * view, the cursor (dotted) and the pins A and B, and a readout of the value at the cursor. */
  drawRow(node) {
    const item = Slice.rows(true).find((found) => found.key === node.panel.id);
    const g = Dock.paint(node.canvas);
    const [w, h] = [node.canvas.clientWidth, node.canvas.clientHeight];
    const title = item?.title || (typeof Views !== "undefined" ? Views.titleOf(node.panel.id) : node.panel.id);
    if (node.name.textContent !== title) node.name.textContent = title;
    g.font = FONT;
    g.fillStyle = T.muted;
    node.scale = null;
    if (!item || !S.view) {
      node.note.textContent = "";
      g.fillText("This shot has no such row.", 8, 18, w - 16);
      return;
    }
    const [t0, t1] = S.view;
    const box = { left: 44, right: w - 8, top: 6, bottom: h - 22 };
    const X = (t) => box.left + ((t - t0) / (t1 - t0)) * (box.right - box.left);
    g.save();
    g.beginPath();
    g.rect(box.left, box.top, box.right - box.left, box.bottom - box.top);
    g.clip();
    const range = Dock.rowPicture(g, item, X, box);
    g.restore();
    const y = range || [0, 1];
    const Y = (v) => box.bottom - ((v - y[0]) / (y[1] - y[0])) * (box.bottom - box.top);
    node.scale = { t0, t1, box, X };
    Dock.axes(g, box, [t0, t1], y, X, Y, "ms", item.kind === "map" ? "channel" : item.units);
    // The rows' cursor and pins over the plot.
    const t = Slice.cursorTime();
    const { A, B } = Slice.pins;
    g.save();
    const picture = item.kind !== "trace"; // over an image each line gets a --panel halo
    for (const [at, dash, colour, width] of [[t, [2, 3], picture ? T.ink : T.muted, 1], [A, [], T.ink, 2], [B, [6, 4], T.ink, 2]]) {
      if (at == null || at < t0 || at > t1) continue;
      const x = Math.round(X(at)) + 0.5;
      for (const [stroke, wide] of picture ? [[T.panel, width + 2], [colour, width]] : [[colour, width]]) {
        g.strokeStyle = stroke;
        g.lineWidth = wide;
        g.setLineDash(stroke === T.panel ? [] : dash);
        g.beginPath();
        g.moveTo(x, box.top);
        g.lineTo(x, box.bottom);
        g.stroke();
      }
    }
    g.restore();
    // An added row says its fetch's state as a profile does: queued, fetching, or the error
    // with its button (review I2); "Loading…" is only a store row still arriving.
    // "Remove from view" on a popped-out row takes its panel too, not only the stack row.
    const missing = item.entry?.state === "error" && item.entry.error?.kind === "missing";
    const drop = () => {
      Views.setPanels(Dock.panels.filter((panel) => Dock.key(panel) !== Dock.key(node.panel)));
      Views.removeKey(item.entry.id);
    };
    if (item.entry || item.added) Dock.chip(node, Dock.entryNote(item.entry, title), () => (missing ? drop() : Dock.entryAct(item.entry)));
    if (!range) {
      node.note.textContent = "";
      const state = item.entry?.state;
      if (state === "done") return Dock.tag(g, "No data in this window.", box.right - 2, box.top + 1);
      if (["queued", "loading", "error", "stopped"].includes(state)) return; // the chip says it
      return Dock.tag(g, "Loading…", box.right - 2, box.top + 1);
    }
    const value = t == null ? null : valueAt(item, t);
    const text = value ? Slice.valueText(value, 0).text : "";
    node.note.textContent = t == null ? "" : `${Slice.time(t)} ms${text && text !== "–" ? ` · ${text}` : ""}`;
  },

  /** A row's picture inside `box` with time at X: returns its y range, or null with no data. */
  rowPicture(g, item, X, box) {
    const height = box.bottom - box.top;
    if (item.row?.kind === "image") {
      const [first, stop] = bandBins(item.row);
      g.imageSmoothingEnabled = false;
      let drew = false;
      for (const data of new Set([S.overview, S.data])) {
        if (!data?.rows?.[item.i]) continue;
        g.drawImage(bitmap(item.row, data, item.i), 0, item.row.n_y - stop, data.n, stop - first,
          X(data.t0), box.top, X(data.t1) - X(data.t0), height);
        drew = true;
      }
      return drew ? imageRange(item.row) : null;
    }
    const data = item.row ? S.data : item.entry?.data;
    const values = item.row ? data?.rows?.[item.i] : data?.values;
    if (!data || !values) return null;
    if (item.kind === "map" && data.channels >= 2) {
      g.imageSmoothingEnabled = false;
      g.drawImage(Views.mapBitmap(item.added), X(data.t0), box.top, X(data.t1) - X(data.t0), height);
      return [-0.5, data.channels - 0.5];
    }
    const [lo, hi] = item.row ? traceRange(item.row, values) : Views.traceRange(data);
    const channels = item.row ? item.row.n_channels : data.channels;
    const Y = (v) => box.bottom - ((v - lo) / (hi - lo)) * height;
    const step = (data.t1 - data.t0) / data.n;
    g.lineWidth = 1;
    for (let c = 0; c < channels; c++) {
      g.strokeStyle = TRACE_COLOURS[c % TRACE_COLOURS.length];
      g.beginPath();
      let pen = false;
      for (let j = 0; j < data.n; j++) {
        const [low, high] = [values[c * data.n + j], values[(channels + c) * data.n + j]];
        if (!Number.isFinite(low)) {
          pen = false;
          continue;
        }
        const x = X(data.t0 + (j + 0.5) * step);
        if (pen) g.lineTo(x, Y(low));
        else g.moveTo(x, Y(low));
        g.lineTo(x, Y(high));
        pen = true;
      }
      g.stroke();
    }
    return [lo, hi];
  },

  // -- Add: profiles from the server, the other views, rows to pop out

  /** Show or hide the Add menu under the header; the catalog is asked for on first open. */
  async toggleMenu(open = !Dock.menuOpen) {
    const parts = Dock.head();
    if (!parts) return;
    Dock.menuOpen = open;
    parts.add.setAttribute("aria-expanded", String(open));
    parts.menu.hidden = !open;
    if (!open) return;
    Dock.renderMenu();
    if (typeof Views !== "undefined" && (!Views.catalog || Views.catalogFor !== `${S.event}/${S.shot}`)) {
      await Views.loadCatalog();
      if (Dock.menuOpen) Dock.renderMenu();
    }
    parts.menu.querySelector("button:not(:disabled)")?.focus();
  },

  /** The Add menu: Profiles (every profile the server offers for this event and shot), Views
   * (Values, the slices, the cameras) and Rows (pop one out of the stack). */
  renderMenu() {
    const parts = Dock.head();
    if (!parts) return;
    const el = (tag, props = {}, ...children) => {
      const node = Object.assign(document.createElement(tag), props);
      node.append(...children);
      return node;
    };
    const has = (panel) => Dock.panels.some((item) => Dock.key(item) === Dock.key(panel));
    const offer = (text, panel, act) => {
      const button = el("button", { type: "button", textContent: text, disabled: has(panel) });
      button.addEventListener("click", () => {
        Dock.toggleMenu(false);
        act();
      });
      return button;
    };
    const ready = typeof Views !== "undefined" && Views.catalog && Views.catalogFor === `${S.event}/${S.shot}`;
    const profiles = ready ? (Views.catalog.entries || []).filter((entry) => entry.kind === "profile") : [];
    const groups = [];
    const failed = typeof Views !== "undefined" && Views.catalogFailed === `${S.event}/${S.shot}`;
    const retry = el("button", { type: "button", textContent: "Retry" });
    retry.addEventListener("click", () => {
      Views.catalogFailed = "";
      Dock.renderMenu();
      Dock.toggleMenu(true);
    });
    groups.push(["Profiles", !ready ? (failed ? [el("p", { className: "hint", textContent: "Could not load the list." }), retry]
      : [el("p", { className: "hint", textContent: "Loading…" })])
      : profiles.length ? profiles.map((entry) => offer(entry.label || entry.id, { kind: "profile", id: entry.id },
        () => Dock.addPanel({ kind: "profile", id: entry.id })))
        : [el("p", { className: "hint", textContent: "None for this shot." })]]);
    const views = [["Values", { kind: "values" }], ["TangTV inversion and EFIT", { kind: "slice", id: "inversion" }],
      ["EFIT equilibrium", { kind: "slice", id: "equilibrium" }]];
    if (S.video) views.push(["Cameras", { kind: "cameras" }]);
    groups.push(["Views", views.map(([text, panel]) => offer(text, panel, () => Dock.addPanel(panel)))]);
    const rows = Slice.rows().filter((item) => item.canvas?.isConnected);
    groups.push(["Pop out a row", rows.length ? rows.map((item) => offer(item.title, { kind: "row", id: item.key }, () => Dock.popRow(item.key)))
      : [el("p", { className: "hint", textContent: "No rows." })]]);
    parts.menu.replaceChildren(...groups.map(([name, items]) => el("section", {}, el("h4", { textContent: name }), ...items)));
  },

  // -- the header

  /** The dock's header: the view's name, Follow, Add and Edit (the tab on its edge closes it); under it, "held at …", A, B
   * and B−A; under that, the Add menu while it is open. */
  head() {
    const box = $("dock-head");
    if (!box) return null;
    if (Dock.headParts) return Dock.headParts;
    const title = Object.assign(document.createElement("span"), { className: "dock-title" });
    const follow = Object.assign(document.createElement("select"), { className: "dock-follow" });
    follow.setAttribute("aria-label", "What the dock follows");
    follow.addEventListener("change", () => {
      Slice.follow = follow.value;
      Dock.followChanged();
    });
    const followLabel = Object.assign(document.createElement("label"), { textContent: "Follow " });
    followLabel.append(follow);
    const add = Object.assign(document.createElement("button"), { type: "button", textContent: "Add", className: "dock-add",
      title: "Add a profile, a slice or a row" });
    add.setAttribute("aria-haspopup", "true");
    add.setAttribute("aria-expanded", "false");
    add.addEventListener("click", () => Dock.toggleMenu());
    const edit = Object.assign(document.createElement("button"), { type: "button", textContent: "Edit", title: "Edit the view: rows, panels, save" });
    edit.addEventListener("click", () => openEditor());
    const reading = Object.assign(document.createElement("div"), { className: "dock-reading" });
    const menu = Object.assign(document.createElement("div"), { className: "dock-menu", hidden: true });
    menu.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      Dock.toggleMenu(false);
      add.focus();
    });
    document.addEventListener("pointerdown", (event) => {
      if (Dock.menuOpen && !menu.contains(event.target) && event.target !== add) Dock.toggleMenu(false);
    });
    const top = Object.assign(document.createElement("div"), { className: "dock-head-line" });
    top.append(title, followLabel, add, edit); // the tab on the dock's edge closes it
    box.replaceChildren(top, reading, menu);
    Dock.headParts = { title, follow, reading, add, menu };
    return Dock.headParts;
  },

  /** Fill the header from the view and the times. */
  drawHead() {
    const parts = Dock.head();
    if (!parts) return;
    const name = Views.label?.() || "Default";
    if (parts.title.textContent !== name) parts.title.textContent = name;
    const options = [["cursor", Dock.videoLed() ? "Video" : "Cursor"], ["A", "A"], ["B", "B"]];
    if (parts.follow.options.length !== 3 || parts.follow.options[0].textContent !== options[0][1]) {
      parts.follow.replaceChildren(...options.map(([value, text]) => Object.assign(document.createElement("option"), { value, textContent: text })));
    }
    parts.follow.value = Slice.follow;
    const { A, B } = Slice.pins;
    const bits = [];
    const held = Slice.held();
    if (held != null) bits.push(`held at ${Slice.time(held)} ms`);
    if (A != null) bits.push(`A ${Slice.time(A)}`);
    if (B != null) bits.push(`B ${Slice.time(B)}`);
    if (A != null && B != null) bits.push(`B−A ${Slice.time(B - A)} ms`);
    const text = bits.join("   ");
    if (parts.reading.textContent !== text) parts.reading.textContent = text;
  },

  /** Drag the handle to resize the dock (px; the view keeps it). */
  wireResize() {
    const handle = $("dock-resize");
    if (!handle || handle.dataset.wired) return;
    handle.dataset.wired = "1";
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-label", "Resize the dock");
    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      const move = (e) => {
        const most = Dock.maxGridPx() - Dock.HANDLE_PX;
        Dock.width = Math.round(clamp(window.innerWidth - e.clientX - Dock.HANDLE_PX, Math.min(Dock.MIN_PX, most), most));
        document.body.style.setProperty("--dock-w", Dock.gridWidth());
      };
      const up = () => {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
        Views.touch?.();
        Dock.relayout();
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });
  },

  /** Follow changed (the select, `l`): a change to the view; the dock redraws. */
  followChanged() {
    if (typeof Views !== "undefined" && Views.layout) Views.touch();
    Dock.redraw();
  },

  /** Show or hide the dock as a view says (applyView) or as toggleDock asks; an open dock with
   * no panels gets Values. False on a page without aside#dock. */
  show(open) {
    const aside = buildDock();
    if (!aside) return false;
    if (open && !Dock.panels.length) Dock.panels = [{ kind: "values" }];
    Dock.open = open;
    aside.hidden = !open;
    if (open) Dock.layout();
    else {
      if (Dock.cameras) Dock.returnCameras();
      if (Dock.menuOpen) Dock.toggleMenu(false);
    }
    Dock.relayout();
    Dock.placeTab();
    return true;
  },

  /** The arrow tab on the dock's left edge (at the window's right edge while the dock is
   * closed): it opens and closes the dock, like D. Made on first use; shown on a shot with
   * the workspace. */
  placeTab() {
    if (typeof document === "undefined" || !document.body) return;
    const show = Boolean(S.meta) && ws() && Boolean($("dock")) && !$("top").hidden;
    let tab = $("dock-tab");
    if (!tab && !show) return;
    if (!tab) {
      tab = Object.assign(document.createElement("button"), { id: "dock-tab", type: "button" });
      tab.addEventListener("click", () => toggleDock());
      document.body.append(tab);
    }
    if (tab.hidden !== !show) tab.hidden = !show;
    const text = Dock.open ? "›" : "‹";
    if (tab.textContent !== text) tab.textContent = text;
    const label = Dock.open ? "Close the dock (D)" : "Open the dock (D)";
    if (tab.title !== label) {
      tab.title = label;
      tab.setAttribute("aria-label", label);
    }
    tab.setAttribute("aria-expanded", String(Dock.open));
  },

  /** On a shot change: forget the shot's slices (the next draw asks /api/slices once). */
  shotChanged() {
    Dock.kindsAt = "";
    Dock.kinds = null;
    Dock.kindsError = null;
    for (const cache of Dock.slices.values()) for (const item of cache.items.values()) item.image?.close?.();
    Dock.slices = new Map();
  },
};

/** Make the dock's parts in aside#dock once: the header, the panels, the editor and the
 * resize handle. Returns the aside, or null on a page without one. */
function buildDock() {
  const aside = $("dock");
  if (!aside) return null;
  for (const [id, tag] of [["dock-head", "header"], ["dock-edit", "div"], ["dock-panels", "div"], ["dock-resize", "div"]]) {
    if (!$(id)) aside.append(Object.assign(document.createElement(tag), { id }));
  }
  $("dock-edit").hidden = !Dock.editing;
  Dock.head();
  Dock.wireResize();
  return aside;
}

/** Draw the dock: its header and each panel at the dock's time. Called by render and on every
 * pointer move (through Slice.redraw); it asks only for what a panel lacks (a profile once per
 * shot, /api/slices once per shot, a slice per new time). */
function drawDock() {
  if (!Dock.open || !buildDock() || !S.meta) return;
  Dock.drawHead();
  const shared = [];
  for (const node of Dock.nodes.values()) {
    if (node.panel.kind !== "profile" || S.shot == null) continue;
    const entry = Signals.entry(Profiles.shots, "profile", S.shot, node.panel.id);
    if (entry.state === "done") shared.push([entry.data.x_kind, Profiles.ranges(entry.data).x]);
  }
  const xShared = Dock.union(shared);
  for (const node of Dock.nodes.values()) {
    const kind = node.panel.kind;
    if (kind === "profile") Dock.drawProfile(node, xShared);
    else if (kind === "values") Dock.drawValues(node);
    else if (kind === "slice") Dock.drawSlice(node);
    else if (kind === "cameras") Dock.drawCameras(node);
    else if (kind === "row") Dock.drawRow(node);
  }
}

/** Show or hide the dock (`d`, its ×): a change to the view, which then reads "<name>
 * (changed)". Opening a view with no panels gives a dock holding only Values. */
function toggleDock(open = !Dock.open) {
  if (!S.meta || open === Dock.open) return;
  if (!Dock.show(open)) return;
  if (typeof Views !== "undefined" && Views.layout) {
    Views.layout.dock = { ...(Views.layout.dock || {}), open, panels: JSON.parse(JSON.stringify(Dock.panels)) };
    Views.touch();
  }
}

/** A profile panel: a canvas whose hover probes its x kind across the image rows. */
function panelProfile(panel) {
  const frame = Dock.frameFor("profile", panel.id);
  const canvas = Object.assign(document.createElement("canvas"), { className: "dock-plot" });
  canvas.setAttribute("aria-label", `Profile ${panel.id} at the slice time`);
  frame.el.append(canvas);
  const node = { ...frame, canvas, panel, scale: null };
  canvas.addEventListener("pointermove", (event) => {
    const scale = node.scale;
    if (!scale) return;
    const x = event.clientX - canvas.getBoundingClientRect().left;
    const v = scale.x[0] + ((x - scale.box.left) / (scale.box.right - scale.box.left)) * (scale.x[1] - scale.x[0]);
    Slice.probe = x >= scale.box.left && x <= scale.box.right ? { kind: scale.kind, v, from: "dock" } : null;
    Slice.redraw();
  });
  canvas.addEventListener("pointerleave", () => {
    if (Slice.probe?.from === "dock") Slice.probe = null;
    Slice.redraw();
  });
  return node;
}

/** The Values panel: a table of Cursor, A, B and B−A. */
function panelValues(panel) {
  const frame = Dock.frameFor("values", "Values");
  const table = Object.assign(document.createElement("table"), { className: "dock-values" });
  const head = document.createElement("thead");
  const tr = document.createElement("tr");
  for (const text of ["", Dock.videoLed() ? "Hover" : "Cursor", "A", "B", "B−A"]) tr.append(Object.assign(document.createElement("th"), { scope: "col", textContent: text }));
  head.append(tr);
  const body = document.createElement("tbody");
  table.append(head, body);
  frame.el.append(table);
  return { ...frame, body, panel };
}

if (typeof module !== "undefined") {
  module.exports = { Dock, buildDock, drawDock, toggleDock, panelProfile, panelValues };
}
