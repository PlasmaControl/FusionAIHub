// The dock (plan D.1, D.2, D.B): the right-hand column of 2-D views beside the rows. Profile
// panels with the cursor, A and B; the Values readout; the TangTV inversion and the EFIT
// equilibrium on R-Z axes; the detachment cameras. A classic script loaded before app.js: it
// only defines names, and app.js and views.js call them (api 11, workspace).
"use strict";

/** The dock's state. `panels` are the view's ({kind: "profile", id, h} | {kind: "values"} |
 * {kind: "slice", id: "inversion" | "equilibrium", h} | {kind: "cameras"}); `nodes` their
 * elements by key; `kinds` what /api/slices said for `kindsAt` ("<event>/<shot>"); `slices`
 * the slice images and equilibria by kind, `asked` and `want` per kind (one request in flight
 * per kind, the latest time wins). */
const Dock = {
  WIDTH: "clamp(320px, 24vw, 480px)", // the default width; a dragged width is in px
  HANDLE_PX: 6,
  MIN_PX: 280,
  PROFILE_H: 200,
  SLICE_H: 300,
  NEAR_MS: 40, // a slice farther than this from the time asked is not shown
  KEEP_IMAGES: 24, // slice images kept per kind (scrubbing back is free)
  KEEP_ASKED: 2000, // asked times remembered per kind, with the slice the server answered
  Q_MARKS: [[1, "q = 1"], [1.5, "q = 3/2"], [2, "q = 2"]],
  X_TITLES: { psi_n: "ψN", rho: "ρ", R_m: "R (m)", channel: "channel", freq_khz: "kHz", time_ms: "ms" },
  open: false,
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

  // -- the page

  /** The time the slice views follow: on detachment the video cursor (#cursor, which stays
   * where it was put), else `sliceTime()`; a followed pin wins in both. */
  time() {
    const pinned = Slice.follow !== "cursor" ? Slice.pins[Slice.follow] : null;
    if (pinned != null) return pinned;
    return S.video ? S.video.time ?? null : sliceTime();
  },

  /** What the slice time follows, for the headers: "cursor", "video", "A" or "B". */
  following() {
    if (Slice.follow !== "cursor" && Slice.pins[Slice.follow] != null) return Slice.follow;
    return S.video ? "video" : "cursor";
  },

  /** The dock's width as the body grid takes it (`--dock-w`), handle included. */
  gridWidth() {
    if (!Dock.open) return "0px";
    return Dock.width ? `${Dock.width + Dock.HANDLE_PX}px` : `calc(${Dock.WIDTH} + ${Dock.HANDLE_PX}px)`;
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
   * panel adopts the camera nodes, which go back to #video-panel when it goes. */
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
          : panel.kind === "slice" ? Dock.panelSlice(panel) : panel.kind === "cameras" ? Dock.panelCameras(panel) : null;
        if (!node) return null;
        Dock.nodes.set(key, node);
      }
      node.panel = panel;
      if (node.canvas) node.canvas.style.height = `${panel.h || (panel.kind === "slice" ? Dock.SLICE_H : Dock.PROFILE_H)}px`;
      return node.el;
    }).filter(Boolean);
    box.replaceChildren(...nodes);
    if (Dock.panels.some((panel) => panel.kind === "cameras")) Dock.adoptCameras();
  },

  /** A panel's frame: a header with a title and a note, and a body. */
  frameFor(kind, title) {
    const el = Object.assign(document.createElement("section"), { className: `dock-panel dock-${kind}` });
    const head = Object.assign(document.createElement("header"), { className: "dock-panel-head" });
    const name = Object.assign(document.createElement("span"), { className: "dock-panel-title", textContent: title });
    const note = Object.assign(document.createElement("span"), { className: "dock-panel-note" });
    head.append(name, note);
    el.append(head);
    return { el, name, note };
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
    Dock.chip(node, Dock.entryNote(entry, label), () => Dock.entryAct(entry));
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
    node.scale = { x, X, box, kind: profile.x_kind };
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
    node.note.textContent = Dock.sliceNote(profile, at(t), at(A), at(B), t);
  },

  /** The header note of a profile: "EFIT 2,380 ms, 8 ms from cursor", or "A and B share one
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
    return `${name} ${Slice.time(here.t)} ms${off ? `, ${Slice.time(off)} ms from ${Dock.following()}` : ""}`;
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
    if (entry.state === "stopped") {
      return { text: "Stopped waiting for this signal. The server keeps fetching it; press Retry to check again.",
        button: "Retry", tone: "" };
    }
    if (entry.state === "error") {
      const kind = entry.error?.kind;
      return rowNote(kind, kind === "invalid" ? { ...facts, source: entry.id.startsWith("mds:") ? "mds" : "ptdata" } : facts)
        || { text: Activity.sentence(entry.error?.message || "it failed"), button: "Retry", tone: "error" };
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
    const lines = [["t", "t (ms)", [t, A, B].map(Slice.time).concat(A != null && B != null ? Slice.signed(B - A).replace(/ ms$/, "") : "–"), ""]];
    for (const item of Slice.rows()) {
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
      if (ticket !== S.ticket) return;
      Dock.kinds = body.kinds || [];
    } catch (error) {
      if (ticket !== S.ticket) return;
      Dock.kinds = [];
      Dock.kindsError = error.message;
    }
    Dock.redraw();
  },

  kind: (name) => Dock.kinds?.find((kind) => kind.kind === name) || null,

  /** The cache of one slice kind: `items` by the slice time the server answered with (oldest
   * use first), `asked` each ms asked -> that slice time (null: none within NEAR_MS). */
  sliceCache(kind) {
    let cache = Dock.slices.get(kind);
    if (!cache) {
      cache = { items: new Map(), asked: new Map(), want: null, busy: false, error: null };
      Dock.slices.set(kind, cache);
    }
    return cache;
  },

  /** Ask /api/slice for `kind` at `pick.t_ms` unless the answer is known; one request in flight
   * per kind, and the latest pick is asked next. The answer is cached by the t_ms it carries
   * (X-Slice, or the equilibrium's own), never by the time asked. */
  async askSlice(kind, pick) {
    const cache = Dock.sliceCache(kind);
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
          Dock.trim(cache);
        } catch (error) {
          if (ticket !== S.ticket) return;
          cache.error = error.status === 404 ? null : error.message;
          cache.asked.set(ms, null);
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
   * alone when the shot has no inversion. */
  panelSlice(panel) {
    const frame = Dock.frameFor("slice", "TangTV inversion");
    const canvas = Object.assign(document.createElement("canvas"), { className: "dock-plot" });
    canvas.setAttribute("aria-label", "Poloidal cross-section at the slice time");
    frame.el.append(canvas);
    return { ...frame, canvas, panel };
  },

  /** Draw a slice panel at the dock's time. */
  drawSlice(node) {
    if (S.shot == null) return;
    Dock.loadKinds();
    const g = Dock.paint(node.canvas);
    const [w, h] = [node.canvas.clientWidth, node.canvas.clientHeight];
    const t = Dock.time();
    const inversion = node.panel.id !== "equilibrium" ? Dock.kind("inversion") : null;
    const equilibrium = Dock.kind("equilibrium");
    node.name.textContent = inversion ? "TangTV inversion and EFIT" : equilibrium ? equilibrium.label || "EFIT equilibrium" : "TangTV inversion";
    g.font = FONT;
    g.fillStyle = T.muted;
    if (!Dock.kinds) {
      node.note.textContent = "";
      g.fillText("Reading what the shot has…", 8, 18);
      return;
    }
    if (!inversion && !equilibrium) {
      node.note.textContent = "";
      g.fillText(Dock.kindsError ? `Could not read the shot's slices: ${Dock.kindsError}.` : "No inversion or EFIT on disk for this shot.", 8, 18, w - 16);
      return;
    }
    if (t == null) {
      node.note.textContent = "";
      g.fillText(S.video ? "Seek the video to show a slice." : "Point at the rows to show a slice.", 8, 18, w - 16);
      return;
    }
    const pickI = inversion && Dock.pickSlice(inversion, t);
    const pickE = equilibrium && Dock.pickSlice(equilibrium, t);
    if (pickI) Dock.askSlice("inversion", pickI);
    if (pickE) Dock.askSlice("equilibrium", pickE);
    const image = pickI ? Dock.itemAt("inversion", pickI.t_ms, (item) => item.image) : null;
    const eq = pickE ? Dock.itemAt("equilibrium", pickE.t_ms, (item) => item.eq)?.eq : null;
    const none = (name, pick) => !pick || Dock.answered(Dock.sliceCache(name), pick.t_ms) === null;
    const box = { left: 34, right: w - 6, top: 6, bottom: h - 20 };
    const extent = image ? [image.record.r_range, image.record.z_range] : eq ? [eq.r_range, eq.z_range] : null;
    const notes = [];
    if (inversion && none("inversion", pickI)) notes.push(`no inversion within ${Dock.NEAR_MS} ms`);
    for (const name of ["inversion", "equilibrium"]) {
      const error = Dock.slices.get(name)?.error;
      if (error) notes.push(`${name === "inversion" ? "inversion" : "EFIT"}: ${error}`);
    }
    if (!extent) {
      node.note.textContent = notes.join(" · ");
      g.fillText(notes.length ? `Nothing to show at ${Slice.time(t)} ms.` : "Loading…", 8, 18, w - 16);
      return;
    }
    const fit = Dock.fitRZ(extent[0], extent[1], box.right - box.left, box.bottom - box.top);
    fit.left += box.left;
    fit.top += box.top;
    const P = (r, z) => Dock.toPixel(fit, r, z);
    const [x0, y0] = P(extent[0][0], extent[1][1]);
    const [x1, y1] = P(extent[0][1], extent[1][0]);
    g.save();
    g.beginPath();
    g.rect(x0, y0, x1 - x0, y1 - y0);
    g.clip();
    if (image) {
      g.imageSmoothingEnabled = true;
      g.drawImage(image.image, x0, y0, x1 - x0, y1 - y0);
    }
    if (eq) Dock.drawEquilibrium(g, eq, P, !image);
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
      notes.push(`TangTV ${Slice.time(r.t_ms)} ms`);
      node.el.title = `${r.source || ""}${r.units ? ` · ${r.vmin}–${r.vmax} ${r.units}` : ""}`.trim();
    }
    if (eq) notes.push(`${eq.source || "EFIT"} ${Slice.time(eq.t_ms)} ms`);
    else if (equilibrium && none("equilibrium", pickE)) notes.push(`no EFIT within ${Dock.NEAR_MS} ms`);
    else if (equilibrium) notes.push("EFIT loading");
    const shown = image?.record.t_ms ?? eq?.t_ms;
    const off = Math.round(Math.abs(shown - t));
    node.note.textContent = `${notes.join(" · ")}${off ? `, ${Slice.time(off)} ms from ${Dock.following()}` : ""}`;
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

  /** The equilibrium over a slice in --ink: the wall, the boundary, X-points (×) and strike
   * points (●); with `levels`, the flux surfaces in --muted. A --panel halo keeps the ink
   * visible over the image. */
  drawEquilibrium(g, eq, P, levels) {
    const path = (points, close = false) => {
      g.beginPath();
      points.forEach(([r, z], k) => {
        const [x, y] = P(r, z);
        if (k) g.lineTo(x, y);
        else g.moveTo(x, y);
      });
      if (close) g.closePath();
    };
    const stroke = (points, width, close) => {
      if (!points?.length) return;
      path(points, close);
      g.strokeStyle = T.panel;
      g.lineWidth = width + 2;
      g.stroke();
      g.strokeStyle = T.ink;
      g.lineWidth = width;
      g.stroke();
    };
    g.save();
    if (levels) {
      g.strokeStyle = T.muted;
      g.lineWidth = 1;
      for (const level of eq.levels || []) {
        if (level.psi_n === 1) continue;
        for (const points of level.paths || []) {
          path(points);
          g.stroke();
        }
      }
    }
    stroke(eq.wall, 1, true);
    stroke(eq.boundary, 1.5, true);
    for (const [r, z] of eq.xpoints || []) {
      const [x, y] = P(r, z);
      for (const width of [3, 1.5]) {
        g.lineWidth = width;
        g.strokeStyle = width === 3 ? T.panel : T.ink;
        g.beginPath();
        g.moveTo(x - 4, y - 4);
        g.lineTo(x + 4, y + 4);
        g.moveTo(x - 4, y + 4);
        g.lineTo(x + 4, y - 4);
        g.stroke();
      }
    }
    for (const [r, z] of eq.strikes || []) {
      const [x, y] = P(r, z);
      g.fillStyle = T.panel;
      g.beginPath();
      g.arc(x, y, 4, 0, 2 * Math.PI);
      g.fill();
      g.fillStyle = T.ink;
      g.beginPath();
      g.arc(x, y, 2.5, 0, 2 * Math.PI);
      g.fill();
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
    node.note.textContent = !S.video ? "No cameras for this shot." : Number.isFinite(time) ? `video ${Slice.time(time)} ms` : "";
  },

  // -- the header

  /** The dock's header: the view's name, Follow, Edit and ×; under it, "held at …", A, B and
   * B−A. */
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
    const edit = Object.assign(document.createElement("button"), { type: "button", textContent: "Edit" });
    edit.addEventListener("click", () => openEditor());
    const close = Object.assign(document.createElement("button"), { type: "button", textContent: "×", className: "dock-close" });
    close.setAttribute("aria-label", "Close the dock (D)");
    close.addEventListener("click", () => toggleDock(false));
    const reading = Object.assign(document.createElement("div"), { className: "dock-reading" });
    const top = Object.assign(document.createElement("div"), { className: "dock-head-line" });
    top.append(title, followLabel, edit, close);
    box.replaceChildren(top, reading);
    Dock.headParts = { title, follow, reading };
    return Dock.headParts;
  },

  /** Fill the header from the view and the times. */
  drawHead() {
    const parts = Dock.head();
    if (!parts) return;
    const name = Views.label?.() || "Default";
    if (parts.title.textContent !== name) parts.title.textContent = name;
    const options = [["cursor", S.video ? "Video" : "Cursor"], ["A", "A"], ["B", "B"]];
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
        Dock.width = Math.round(clamp(window.innerWidth - e.clientX - Dock.HANDLE_PX, Dock.MIN_PX, window.innerWidth * 0.6));
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
    else if (Dock.cameras) Dock.returnCameras();
    Dock.relayout();
    return true;
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
  for (const text of ["", "Cursor", "A", "B", "B−A"]) tr.append(Object.assign(document.createElement("th"), { scope: "col", textContent: text }));
  head.append(tr);
  const body = document.createElement("tbody");
  table.append(head, body);
  frame.el.append(table);
  return { ...frame, body, panel };
}

if (typeof module !== "undefined") {
  module.exports = { Dock, buildDock, drawDock, toggleDock, panelProfile, panelValues };
}
