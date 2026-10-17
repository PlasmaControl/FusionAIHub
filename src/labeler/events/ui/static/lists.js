// Named shot lists on the review page (plan D.5): the list dropdown and its memory, and All (the
// event dropdown's first entry, once the By shot button), which walks a list shot by shot across
// every event over the #silver-page DOM. A classic script loaded before app.js: it only defines
// names, and app.js calls them (api 11).
"use strict";

/** The list the queue shows and the walk All makes of it. `name` and `lists` come from the last
 * /api/queue answer (`list`, `lists`) for `event`; `overview` and `strips` are the
 * /api/overview answers for the list and for its selected `shot` (`byShot`, `openByShot` and
 * the `by-shot` notes keep the old name). */
const Lists = {
  ALL: "__all", // the event dropdown's All
  ROSTER: "roster", // the event's own list: All walks it from the event's queue
  name: null, // the list the queue shows, as the server named it ("roster" for the event's own)
  lists: [], // the lists offered on `event`: /api/shotlists?event=
  event: null,
  byShot: false, // All is chosen: #silver-page shows the list and the shot on every event
  context: false, // an event opened from All: the breadcrumb leads back to its shot
  own: true, // the queue is the event's own view (no list, the roster or a fallback): gold extras stay out of it
  shot: null, // the shot All has selected
  overview: null, // /api/overview?list=
  strips: null, // /api/overview?list=&shot=
  sequence: 0, // the latest All request; an older answer is dropped
  INDENT: " ", // an em space per level a child list sits under its parent

  /** Where the browser keeps the list chosen on `event`; nothing kept means its default. */
  storageKey: (event) => `labeler:list:${event}`,

  /** The queue's address: today's exact URL unless a list other than the default was chosen. */
  queueUrl: (event, name) => `/api/queue?event=${encodeURIComponent(event)}${name ? `&list=${encodeURIComponent(name)}` : ""}`,

  /** What renderQueue compares to rebuild every chip when the event or its list changes. */
  queueKey: (event, name = Lists.name) => `${event}/${name ?? ""}`,

  /** An option's text: "elm · 576 · 120 saved" (the roster by its title). */
  /** A strip layer's origin in one word (its full name is the tooltip). */
  originText: (layer) => ({ saved: "saved", folded: "saved", format: "curated", extend: "model",
    source: "source", suggest: "suggestion" })[layer.kind] || layer.kind,

  optionText: (entry) => `${entry.kind === "roster" ? entry.title : entry.name} · ${entry.n} · ${entry.n_saved ?? 0} saved`,

  /** The dropdown's three groups, [{label, options: [{value, text, title}]}], empty ones left out:
   * the lists recommended for the event, the others (a child indented under its parent), the
   * event's own roster. */
  groups(lists, eventName) {
    const option = (entry, depth = 0) => ({
      value: entry.name, text: `${Lists.INDENT.repeat(depth)}${Lists.optionText(entry)}`, title: entry.title || entry.name,
    });
    const other = lists.filter((entry) => !entry.recommended && entry.kind !== "roster");
    const names = new Set(other.map((entry) => entry.name));
    const depth = (entry, seen = new Set()) => {
      const parent = other.find((found) => found.name === entry.parent);
      if (!parent || !names.has(parent.name) || seen.has(parent.name)) return 0;
      seen.add(parent.name);
      return 1 + depth(parent, seen);
    };
    return [
      { label: `Recommended for ${eventName}`, options: lists.filter((entry) => entry.recommended).map((entry) => option(entry)) },
      { label: "Other lists", options: other.map((entry) => option(entry, depth(entry))) },
      { label: "Event roster", options: lists.filter((entry) => entry.kind === "roster").map((entry) => option(entry)) },
    ].filter((group) => group.options.length);
  },

  /** How the closed dropdown looks: `.recommended` when the list is recommended for the event;
   * otherwise a warning title, except on the event's default. */
  mark(entry) {
    if (!entry) return { recommended: false, title: "" };
    const title = entry.recommended || entry.default ? entry.title || entry.name : "Not recommended for this event";
    return { recommended: Boolean(entry.recommended), title };
  },

  /** `#list/<name>[/<shot>[/<event>]]`, the address of All on a list (once By shot's). */
  listAddress(name, shot = null, event = null) {
    const tail = shot == null ? "" : `/${shot}${event ? `/${encodeURIComponent(event)}` : ""}`;
    return `#list/${encodeURIComponent(name)}${tail}`;
  },

  /** {list, shot, event} from a `#list/` address, or from a `#silver/<batch>/…` one (batch b is
   * the list silver_batch_<b>; no batch is the whole silver list); null for any other. */
  parseAddress(hash) {
    const parts = String(hash || "").replace(/^#/, "").split("/").map((part) => {
      try {
        return decodeURIComponent(part);
      } catch {
        return part;
      }
    });
    const shot = (text) => (/^\d+$/.test(text || "") ? Number(text) : null);
    if (parts[0] === "list" && parts[1]) return { list: parts[1], shot: shot(parts[2]), event: parts[3] || null };
    if (parts[0] !== "silver") return null;
    const batch = /^[1-9]\d*$/.test(parts[1] || "") ? `silver_batch_${parts[1]}` : "silver";
    return { list: batch, shot: shot(parts[2]), event: parts[3] || null };
  },

  /** The old "Gold set" box, once: an event or subcategory without a chosen list gets gold, and
   * the box's key goes. `read` and `write` are the page's storage calls. Returns how many
   * events it set. */
  migrate(read, write, events) {
    if (read("labeler:gold-only") !== "1") return 0;
    let n = 0;
    for (const row of events) {
      for (const event of [row.event, ...(row.subcategories || []).map((sub) => sub.event)]) {
        if (read(Lists.storageKey(event)) != null) continue;
        write(Lists.storageKey(event), "gold");
        n++;
      }
    }
    write("labeler:gold-only", null);
    return n;
  },

  /** Queue rows from the server with this tab's newer saves in place (as openEvent merges
   * them); a save keeps the row's off-roster `extra` mark. */
  merge(rows, saved, saveCount) {
    return rows.map((row) => {
      const newer = saved?.get(row.shot);
      if (!newer || !(newer.count > saveCount)) return row;
      return row.extra ? { ...newer.row, extra: true } : newer.row;
    });
  },

  /** The family a list sits in, as the batch buttons show it: the root list, then its
   * children. The open one reads "<title> · <done>/<n> done" once its overview is in, the
   * others "<title> · <n>".
   * `lists` (the offered entries) say who a list's parent is; `overview` holds its children. */
  family(name, overview, lists = []) {
    const entry = (found) => lists.find((item) => item.name === found);
    const parent = entry(name)?.parent && entry(entry(name).parent) ? entry(name).parent : null;
    const root = parent ?? name;
    const children = parent ? lists.filter((item) => item.parent === parent) : overview?.children || [];
    const shots = overview?.shots || [];
    return [root, ...children.map((child) => child.name)].map((member) => {
      const known = entry(member) || (overview?.children || []).find((child) => child.name === member);
      const title = known?.title || (member === name ? overview?.title : null) || member;
      const text = member === name && overview
        ? `${title} · ${shots.filter((row) => row.done).length}/${shots.length} done`
        : `${title} · ${known?.n ?? "?"}`;
      return { name: member, text, pressed: member === name };
    });
  },

  /** A shot's classes column: "a:present;b:absent" or a list. */
  classes: (value) => (Array.isArray(value) ? value : String(value ?? "").split(";").map((part) => part.trim()).filter(Boolean)),

  /** A class's event as the shot list writes it: AE, ELM, q min, conf. */
  short: (name) => name.replace("alfven_eigenmode", "AE").replace("edge_localized_mode", "ELM")
    .replace("minimum_safety_factor", "q min").replace("confinement", "conf"),

  /** The shot `delta` places from `shot` in `shots`, wrapping; the first or last when `shot` is
   * not among them; null for no shots. */
  step(shots, shot, delta) {
    if (!shots.length) return null;
    const at = shots.findIndex((row) => row.shot === shot);
    if (at === -1) return (delta > 0 ? shots[0] : shots[shots.length - 1]).shot;
    return shots[(((at + delta) % shots.length) + shots.length) % shots.length].shot;
  },

  /** The next shot after `shot` (wrapping) that is not done, or null. */
  nextShot(shots, shot) {
    const at = shots.findIndex((row) => row.shot === shot);
    return [...shots.slice(at + 1), ...shots.slice(0, Math.max(0, at))].find((row) => !row.done) || null;
  },

  /** The next event on a shot after `event` (wrapping; the first with no `event`) that is
   * neither saved nor locked behind its parent, or null. */
  nextOnShot(events, event) {
    const at = events.findIndex((row) => row.event === event);
    return [...events.slice(at + 1), ...events.slice(0, Math.max(0, at))].find((row) => !row.saved && !row.locked) || null;
  },

  /** The time line under the strips: "1000–5200 ms · 10 ms frames". */
  timeText: (strips) => `${strips.t_range[0]}–${strips.t_range[1]} ms · ${strips.grid_ms ?? GRID_MS} ms frames`,

  /** The list chosen on `event`, or null for its default. */
  chosen: (event) => stored(Lists.storageKey(event)),

  /** Read the lists offered on `event` when the queue has not given them. */
  async load(event) {
    const lists = await (await api(`/api/shotlists?event=${encodeURIComponent(event)}`)).json();
    if (S.event === event) fillLists({ list: Lists.name, lists }, event);
  },

  /** Take a queue answer for `event` as openEvent and chooseList do: the rows (with this tab's
   * newer saves), the dropdown, the chips and the header. `asked` is the list the request named
   * (null for none): the server answers the event's own view, with the gold shots off the roster
   * marked `extra`, for none, for "roster" and when it fell back, and filterQueue keeps those
   * extras out of the event's own view, as today. */
  take(event, queue, saveCount, asked = null) {
    S.allQueue = Lists.merge(queue.shots, S.savedRows.get(event), saveCount);
    Lists.own = !asked || asked === "roster" || queue.list !== asked;
    // The server fell back from the list kept for this event (the Gold set migration writes gold
    // for every event): forget it, so the next open asks the default and says nothing.
    if (asked && asked !== "roster" && queue.list !== asked && Lists.chosen(event) === asked) {
      store(Lists.storageKey(event), null);
    }
    fillLists(queue, event);
    filterQueue();
    S.queueEvent = event;
    renderQueue();
    showHeader();
  },

  /** Show or hide All in place of the rows (the event dropdown says All, and the Find shot box
   * moves above the list's shots, as it sits above the chips on an event); with an event open
   * from it, the breadcrumb back to its shot and Next event. */
  show() {
    const all = Lists.byShot;
    document.body.classList.toggle("silver-mode", all);
    $("silver-page").hidden = !all;
    $("top").hidden = $("bottom").hidden = all;
    if (all) {
      $("event").value = Lists.ALL;
      $("silver-page").setAttribute("aria-label", "All");
      $("silver-help").textContent = "J/K step · U next not done · Enter opens · Esc back";
    }
    const find = $("shot-find"), side = $("silver-find");
    if (find && side && (find.parentElement === side) !== all) {
      if (all) side.prepend(find);
      else $("bottom").insertBefore(find, $("queue"));
    }
    if (side) side.hidden = !all;
    $("silver-breadcrumb").hidden = !Lists.context;
    $("silver-breadcrumb").textContent = `← All · ${Lists.shot}`;
    $("silver-breadcrumb").title = "Back to All on this shot (G)";
    $("silver-next-event").hidden = !Lists.context || !S.meta?.saved;
  },

  /** Say on the page why the list or shot could not be shown, and keep it with the failures. */
  fail(what, error) {
    const text = `Could not load ${what.charAt(0).toLowerCase()}${what.slice(1)}: ${error.message}. Choose it again to retry.`;
    $("silver-strips").replaceChildren(Object.assign(document.createElement("p"), { className: "failure", textContent: text }));
    Activity.fail(what, error.message);
  },

  /** The family buttons and the list's shots, each with its classes and progress. */
  renderShots() {
    $("silver-batches").replaceChildren(...Lists.family(Lists.name, Lists.overview, Lists.lists).map((member) => {
      const button = Object.assign(document.createElement("button"), { type: "button", textContent: member.text });
      button.setAttribute("aria-pressed", String(member.pressed));
      button.addEventListener("click", () => openByShot(member.name));
      return button;
    }));
    const shots = Lists.overview?.shots || [];
    $("silver-shots").replaceChildren(...shots.map((row, i) => {
      const button = Object.assign(document.createElement("button"), {
        type: "button", className: `silver-shot${row.shot === Lists.shot ? " current" : ""}`,
      });
      button.dataset.shot = row.shot;
      button.setAttribute("aria-pressed", String(row.shot === Lists.shot));
      const order = Object.assign(document.createElement("span"), { textContent: row.order ?? i + 1, className: "silver-order" });
      const number = Object.assign(document.createElement("strong"), { textContent: row.shot });
      const classes = Object.assign(document.createElement("span"), { className: "silver-classes" });
      if (SILVER_POOLS[row.pool]) {
        classes.append(Object.assign(document.createElement("span"), { textContent: SILVER_POOLS[row.pool], className: "silver-tag" }));
      }
      for (const name of Lists.classes(row.classes)) {
        const [event, category] = name.split(":");
        const chip = Object.assign(document.createElement("span"), {
          textContent: category ? `${Lists.short(event)}:${category}` : Lists.short(event), className: "silver-class",
        });
        chip.setAttribute("aria-label", name);
        classes.append(chip);
      }
      const progress = Object.assign(document.createElement("span"), { // a roster row has no counts
        textContent: row.total == null ? (row.done ? "✓" : "") : `${row.saved}/${row.total}${row.done ? " ✓" : ""}`,
        className: "silver-progress",
      });
      button.append(order, number, classes, progress);
      button.addEventListener("click", () => Lists.select(row.shot));
      return button;
    }));
    $("silver-shots").querySelector(".current")?.scrollIntoView({ block: "nearest" });
    // As on an event: Next names the shot K opens, the bar and the box the shot on screen.
    const next = Lists.step(shots, Lists.shot, 1);
    if ($("all-next-shot")) $("all-next-shot").textContent = next == null || next === Lists.shot ? "" : `→ ${next}`;
    $("shot-now").textContent = Lists.shot == null ? "" : `Shot ${Lists.shot}`;
    if (document.activeElement !== $("shot")) $("shot").value = Lists.shot ?? "";
  },

  /** K/J and Next: the list's next or previous shot (wrapping). */
  stepShot(delta) {
    const next = Lists.step(Lists.overview?.shots || [], Lists.shot, delta);
    if (next != null) Lists.select(next);
  },

  /** A named list for /api/overview?list=&shot=: the strips are the same on every list, but the
   * route takes a named one and the roster has none. The walked list, else the smallest list
   * offered with shots. */
  stripsList() {
    if (Lists.name && Lists.name !== Lists.ROSTER) return Lists.name;
    const named = Lists.lists.filter((entry) => entry.kind !== "roster" && entry.n > 0);
    return named.length ? named.reduce((a, b) => (b.n < a.n ? b : a)).name : null;
  },

  /** The walked list's shots: /api/overview?list= for a named list; for the roster (each
   * event's own), the event's queue in its order, done once saved (no counts across events). */
  async fetchOverview(name, event = S.event) {
    if (name !== Lists.ROSTER) return (await api(`/api/overview?list=${encodeURIComponent(name)}`)).json();
    const queue = await (await api(Lists.queueUrl(event, Lists.ROSTER))).json();
    const rows = (queue.shots || []).filter((row) => !row.extra);
    return {
      list: Lists.ROSTER, title: Lists.lists.find((entry) => entry.kind === "roster")?.title || "Event roster",
      shots: rows.map((row, i) => ({ shot: row.shot, order: i + 1, classes: [], done: row.state !== "unreviewed" })),
    };
  },

  /** /api/overview?list=&shot=: `shot` on every event. */
  async fetchStrips(shot) {
    const list = Lists.stripsList();
    if (!list) throw new Error("this event offers no shot list");
    return (await api(`/api/overview?list=${encodeURIComponent(list)}&shot=${shot}`)).json();
  },

  /** Select `shot`: its strips on every event, from /api/overview?list=&shot=. */
  async select(shot) {
    const sequence = ++Lists.sequence, name = Lists.name;
    Object.assign(Lists, { shot, strips: null });
    $("silver-strips").replaceChildren();
    $("silver-title").textContent = `Shot ${shot}`;
    $("silver-time").textContent = "";
    Lists.renderShots();
    history.replaceState(null, "", listAddress(name, shot));
    try {
      const strips = await Lists.fetchStrips(shot);
      if (!Lists.byShot || sequence !== Lists.sequence) return;
      Lists.strips = strips;
      Lists.renderStrips();
    } catch (error) {
      if (Lists.byShot && sequence === Lists.sequence) Lists.fail(`Shot ${shot}`, error);
    }
  },

  /** One strip per event on the selected shot (and its layers with "All layers" ticked); a
   * strip opens its event at the shot. */
  renderStrips() {
    if (!Lists.byShot || !Lists.strips) return;
    const strips = Lists.strips, layers = $("silver-all-layers").checked;
    $("silver-title").textContent = `Shot ${strips.shot}`;
    $("silver-time").textContent = Lists.timeText(strips);
    $("silver-strips").replaceChildren(...strips.events.map((event) => {
      const section = Object.assign(document.createElement("section"), {
        className: `silver-event${event.parent ? " child" : ""}${event.locked ? " locked" : ""}`,
      });
      const row = Object.assign(document.createElement("button"), { type: "button", className: "silver-strip", disabled: event.locked });
      row.setAttribute("aria-label", event.display_name);
      row.append(
        Object.assign(document.createElement("span"), {
          textContent: `${event.display_name.slice(0, 24)}${event.saved ? " ✓" : ""}`, className: "silver-event-name",
        }),
        document.createElement("canvas"),
        Object.assign(document.createElement("span"), { className: "silver-origin" }),
      );
      row.addEventListener("click", () => Lists.openEvent(event.event));
      section.append(row);
      if (layers) {
        event.layers.forEach((layer, i) => {
          const item = Object.assign(document.createElement("div"), { className: "silver-strip silver-layer" });
          item.append(
            Object.assign(document.createElement("span"), { textContent: layer.kind }),
            document.createElement("canvas"),
            Object.assign(document.createElement("span"), {
              textContent: Lists.originText(layer), title: layer.name, className: "silver-origin",
            }),
          );
          item.dataset.layer = i;
          section.append(item);
        });
      }
      section.dataset.event = event.event;
      return section;
    }));
    requestAnimationFrame(Lists.drawStrips);
  },

  /** Paint the strips and the time axis under them (on a resize too). */
  drawStrips() {
    if (!Lists.byShot || !Lists.strips) return;
    const range = Lists.strips.t_range;
    for (const section of $("silver-strips").children) {
      const event = Lists.strips.events.find((found) => found.event === section.dataset.event);
      if (!event) continue;
      const row = section.firstChild;
      const used = event.locked ? [] : Lists.drawStrip(row.querySelector("canvas"), event, range);
      row.lastChild.textContent = event.locked ? "Review parent first"
        : [...new Set(used.map(Lists.originText))].join(" · ") || "unassessed";
      row.lastChild.title = used.map((layer) => layer.name).join("\n");
      for (const item of [...section.children].slice(1)) {
        Lists.drawStrip(item.querySelector("canvas"), event, range, [event.layers[Number(item.dataset.layer)]]);
      }
    }
    const canvas = $("silver-axis"), ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    const g = context(canvas), [lo, hi] = range;
    const step = niceStep(hi - lo, 6);
    g.fillStyle = T.muted;
    g.font = FONT;
    for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) {
      const x = ((t - lo) / (hi - lo)) * canvas.clientWidth;
      g.fillRect(x, 0, 1, 5);
      g.fillText(`${t}`, clamp(x + 3, 0, Math.max(0, canvas.clientWidth - 45)), 19);
    }
  },

  /** One strip: each 10 ms frame in its category's colour, hatched where nothing assessed it.
   * Returns the layers it drew from. */
  drawStrip(canvas, event, range, layers = event.layers) {
    const { cells, origin } = silverFrames(event, range, layers);
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    const g = context(canvas), w = canvas.clientWidth, h = canvas.clientHeight;
    const x = (k) => (k / Math.max(1, cells.length)) * w;
    for (const [a, b, c] of runs(cells)) {
      if (c === -1) {
        g.save();
        g.beginPath();
        g.rect(x(a), 0, x(b) - x(a), h);
        g.clip();
        g.strokeStyle = T.rule;
        for (let edge = x(a) - h; edge < x(b); edge += 7) {
          g.beginPath();
          g.moveTo(edge, h);
          g.lineTo(edge + h, 0);
          g.stroke();
        }
        g.restore();
      } else {
        g.fillStyle = c === 0 ? T.unreviewed : categoryColour(c, event.event, event.categories);
        g.fillRect(x(a), 0, x(b) - x(a), h);
      }
    }
    return [...new Set(origin)].filter((i) => i >= 0).map((i) => layers[i]);
  },

  /** Open `event` at the selected shot, leaving All with the way back; with no shot selected,
   * the event as usual. */
  async openEvent(event) {
    if (Lists.shot == null) {
      Object.assign(Lists, { byShot: false, context: false });
      return openEvent(event);
    }
    Object.assign(Lists, { byShot: false, context: true });
    return openEvent(event, Lists.shot, true);
  },

  /** After a save in an event opened from All: the shot's next event still to review, else the
   * list's next shot not done, on its first such event. A failure says so and stays. */
  async nextEvent() {
    const shot = Lists.shot, event = S.event, name = Lists.name;
    const still = () => Lists.context && Lists.shot === shot && S.event === event;
    try {
      const strips = await Lists.fetchStrips(shot);
      if (!still()) return;
      const next = Lists.nextOnShot(strips.events, event);
      if (next) return await Lists.openEvent(next.event);
      const overview = await Lists.fetchOverview(name, event);
      if (!still()) return;
      Lists.overview = overview;
      const after = Lists.nextShot(overview.shots, shot);
      await openByShot(name, after?.shot ?? null);
      const first = after && Lists.byShot && Lists.strips ? Lists.nextOnShot(Lists.strips.events, null) : null;
      if (first) await Lists.openEvent(first.event);
    } catch (error) {
      Activity.note("by-shot", `Could not open the next event: ${error.message}. G goes back to All.`, "error");
    }
  },
};

/** Fill #shotlist from a queue answer (or {list, lists}) for `event`: the groups, the list in
 * use, how the closed control looks and the server's fallback note. */
function fillLists(body, event = S.event) {
  Object.assign(Lists, { event, lists: body?.lists || [], name: body?.list ?? null });
  Activity.note("list", body?.note || null);
  const select = $("shotlist");
  if (!select) return;
  const name = locate(event)?.chip?.display_name || locate(event)?.row.display_name || event;
  select.replaceChildren(...Lists.groups(Lists.lists, name).map((group) => {
    const element = document.createElement("optgroup");
    element.label = group.label;
    element.append(...group.options.map((item) => Object.assign(new Option(item.text, item.value), { title: item.title })));
    return element;
  }));
  select.value = Lists.name ?? "";
  select.disabled = !Lists.lists.length;
  const look = Lists.mark(Lists.lists.find((entry) => entry.name === Lists.name));
  select.classList.toggle("recommended", look.recommended);
  select.title = look.title;
}

/** The list dropdown changed: remember the choice (nothing for the default) and show the list,
 * with one queue request, keeping the old queue usable until it arrives; in All, walk the new
 * list. A failure says so and puts the dropdown back. */
async function chooseList(name) {
  const event = S.event;
  if (!event || (name === Lists.name && !Lists.byShot)) return;
  store(Lists.storageKey(event), Lists.lists.find((entry) => entry.name === name)?.default ? null : name);
  if (Lists.byShot) return openByShot(name, Lists.shot); // the strips are the shot's on any list
  if (Lists.context) {
    Lists.context = false;
    Lists.show();
  }
  const ticket = S.ticket, saveCount = S.saveCount;
  let queue;
  try {
    queue = await (await api(Lists.queueUrl(event, Lists.chosen(event)))).json();
  } catch (error) {
    if (S.event !== event) return;
    fillLists({ list: Lists.name, lists: Lists.lists }, event);
    Activity.note("list", `Could not load the list ${name}: ${error.message}. Choose it again to retry.`, "error");
    return;
  }
  if (S.event !== event) return;
  Lists.take(event, queue, saveCount, Lists.chosen(event));
  if (ticket !== S.ticket || (S.shot != null && inQueue(S.shot))) return;
  await openShot(inQueue(queue.resume) ? queue.resume : firstToReview());
}

/** Turn All on (the list the queue shows, at the open shot) or off (the event, at the shot All
 * selected). */
function toggleByShot(on = !Lists.byShot) {
  if (on) return Lists.byShot ? Promise.resolve() : openByShot(Lists.name || Lists.ROSTER, S.shot ?? null);
  const event = [S.event, stored("labeler:event"), ...S.events.map((row) => row.event)].find((found) => found && hasEvent(found));
  Object.assign(Lists, { byShot: false, context: false });
  if (!event) {
    Lists.show();
    return Promise.resolve();
  }
  return openEvent(event, Lists.shot);
}

/** All: list `name` shot by shot over #silver-page: its shots with their progress, its family
 * as the batch buttons, `shot` selected and, with `event`, that event opened at it. The event
 * dropdown says All; the shot on screen is closed. */
async function openByShot(name = Lists.name, shot = null, event = null) {
  name ||= Lists.ROSTER;
  const ticket = ++S.ticket;
  leave();
  document.body.classList.remove("busy"); // as busy(null) does, without writing #status
  Object.assign(S, { meta: null, label: null, data: null, overview: null });
  Object.assign(Lists, {
    byShot: true, context: false, name, shot, strips: null, overview: Lists.overview?.list === name ? Lists.overview : null,
  });
  const sequence = ++Lists.sequence;
  Activity.clear("by-shot");
  if (!S.event) {
    const pick = [stored("labeler:event"), ...S.events.map((row) => row.event)].find((found) => found && hasEvent(found));
    if (pick) S.event = pick; // the event whose lists are offered (and whose roster All walks)
  }
  $("silver-strips").replaceChildren();
  $("silver-time").textContent = "";
  $("silver-title").textContent = Lists.lists.find((entry) => entry.name === name)?.title || name;
  Lists.show();
  Lists.renderShots();
  history.replaceState(null, "", listAddress(name, shot));
  try {
    if (S.event && (Lists.event !== S.event || !Lists.lists.some((entry) => entry.name === name))) await Lists.load(S.event);
    const overview = await Lists.fetchOverview(name);
    if (ticket !== S.ticket || sequence !== Lists.sequence) return;
    Lists.overview = overview;
    $("silver-title").textContent = overview.title || name;
    Lists.renderShots();
    if (shot == null) return;
    await Lists.select(shot);
    if (event && ticket === S.ticket && Lists.strips && hasEvent(event)) await Lists.openEvent(event);
  } catch (error) {
    if (ticket === S.ticket) Lists.fail(`The list ${name}`, error);
  }
}

/** All's keys (silverKey's at api 11), as on an event: K, → or ↓ the next shot, J, ← or ↑ the
 * previous, U the next not done; Enter opens, g or Escape go back. True when the key was All's. */
function byShotKey(key) {
  if (!Lists.byShot) {
    if (Lists.context && ["g", "Escape"].includes(key)) {
      openByShot(Lists.name, Lists.shot);
      return true;
    }
    return false;
  }
  const shots = Lists.overview?.shots || [];
  if (["ArrowDown", "ArrowUp", "ArrowRight", "ArrowLeft", "j", "k"].includes(key)) {
    Lists.stepShot(["ArrowDown", "ArrowRight", "k"].includes(key) ? 1 : -1);
  } else if (key === "u") {
    const next = Lists.nextShot(shots, Lists.shot);
    if (next) Lists.select(next.shot);
  } else if (key === "Enter") {
    if (!Lists.strips && shots.length) Lists.select(Lists.shot ?? shots[0].shot);
    else {
      const events = Lists.strips?.events || [];
      const next = Lists.nextOnShot(events, null) || events.find((row) => !row.locked);
      if (next) Lists.openEvent(next.event);
    }
  } else if (["g", "Escape"].includes(key)) {
    if (Lists.shot != null) openByShot(Lists.name);
    else toggleByShot(false);
  } else return false;
  return true;
}

/** `#list/<name>[/<shot>[/<event>]]`: see Lists.listAddress. */
function listAddress(name, shot = null, event = null) {
  return Lists.listAddress(name, shot, event);
}

if (typeof module !== "undefined") {
  module.exports = { Lists, fillLists, chooseList, toggleByShot, openByShot, byShotKey, listAddress };
}
