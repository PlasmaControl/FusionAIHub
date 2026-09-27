---
title: "Label review"
sidebar_position: 2
---

# Label review

A browser page for checking an event's labels one shot at a time. The
diagnostic rows fill the top of the page on one time axis. The shot's source
label and your label sit below them, with the whole roster as a row of chips.
Each shot has one label. You change it, save it, and it is still there when you
come back.

Code: `src/labeler/events/review/` (labels and rows) and
`src/labeler/events/ui/` (server and page). Tests: `tests/labeler/test_review_*.py`
and `tests/labeler/test_events_ui.py`.

## Opening it

```bash
pixi run -e labelmaker labeler-verify
```

This prints a link carrying a fresh token and an `ssh -L 8811:localhost:8811 <node>`
line. Run the `ssh` line on your laptop, then open the link there. The server
listens on the loopback only. The first visit trades the token for a cookie, so
reloads and `#<event>/<shot>` links work until the server restarts. `--port`
and `--token` change either one.

The page opens on the event you last had open, at the first unreviewed shot
after the newest save.

## The page

- **Rows.** Alfvén eigenmode shots show the CO2 interferometer as three rows:
  the cross-power of chord R0 with V1, V2 and V3. A mode seen by several chords
  shows in all three rows; noise on a single chord averages out. Colour is dB above each
  frequency's own quiet level (its median over 0-6 s), from -3 to 27 dB. Other
  events show the rows of their panel builder.
- **Source** is the label the event's newest `format/*_format_*.csv` gives the
  shot. **Label** is yours; it starts as a copy of the source. Where the two
  differ, a strip along the top of the label track marks the difference.
- **Chips**, one per roster shot, coloured by state: *unreviewed* (never saved),
  *confirmed* (saved as the source had it) or *changed*. A dot marks a shot with
  unsaved edits. Unsaved edits live in the browser until you save or revert, so
  a reload keeps them.
- **Your name**, in the box at the top right, goes with every save and is shown
  beside it (`saved Sep 26, 14:02 by Ada Lovelace`). The browser remembers it.
  It is an attribution, not a login: the history also records the login that
  runs the server.
- **History** (`H`) lists the shot's saved versions, newest first: who saved
  each one, when, how many spans, and how many ms changed. The first version
  is compared with the event's current source table; later ones with the version
  before. **Restore** loads a version as an unsaved edit; a save appends a
  version and never rewrites one, but a crash between writing the label and
  its history line can leave the current label without its version line.
- **Mask** (Alfvén eigenmode shots with a pseudo-mask): TokEye's coherent lines
  inside the label's AE frames, 80-250 kHz, drawn in cyan over the rows
  (`labeler.ae.seg.pseudo` builds them). The segmentation model learns from
  them. A click on a region that is not the mode (an MHD harmonic, pickup)
  rejects it, grey; a second click takes that back. Each click is saved at
  once, with your name. `M` hides and shows the mask, and the browser
  remembers which. The header counts the regions kept.
- **Save and next** shows the shot it goes to. It is the next shot in the
  queue, reviewed or not, wrapping at the end; `U` still jumps to the next
  unreviewed one.

While the next shot is still opening, you cannot edit, save or restore a
label. History closes when you move on. Enter closes History too; click
Restore to load a version. If it replaces an unsaved edit, Ctrl+Z brings
that edit back. Restoring the current label leaves it alone.

| Key or gesture | Does |
|---|---|
| Drag on empty label track | add a span |
| Drag a span, or its edge | move it, or resize it |
| Drag the foot of the label track | move the window's edges |
| Shift-drag on the rows | add a span |
| Drag on the rows | pan |
| Wheel | scroll the rows |
| Ctrl/⌘ + wheel, or any wheel over the axis or tracks | zoom at the cursor |
| Double-click, `0` | fit the window |
| Shift + `←` `→`, `-` `=` | pan, zoom |
| `1`-`9` | category for new spans and the selected one |
| `Delete` | remove the selected span |
| Ctrl/⌘ + `Z` | undo |
| `Enter` | save and open the next shot in the queue |
| `S` / `R` | save / revert to the source |
| `←` `→` or `J` `K` / `U` | previous, next shot (an edit stays as a draft) / next unreviewed |
| `H` | saved versions, and restore one |
| `M` / click a mask region | AE: hide or show the pseudo-mask / reject the region, or take that back |
| `[` `]` | contrast |
| `?` | this list |

## What a save writes

Saves go under the event's directory in the label tables
(`data/events/<event>/` unless `LABELER_LABEL_TABLES` says otherwise):

- `review/labels.csv` holds the current label of every reviewed shot, in the
  format-table schema (`shot, category, t_start, t_end, confidence`, whole ms).
  A shot's rows tile its window, and the gaps are category 0. Each save replaces
  that shot's rows and rewrites the file atomically. The result validates like
  any other format table.
- `review/history.jsonl` gets one line per save: shot, `reviewer` (the login
  running the server), `name` (what the name box held, or null), time, the
  window and spans saved, and the source file they were compared with. It is
  only ever appended to: a shot's versions are its lines in order, numbered
  from 1, and `GET /api/history?event=&shot=` lists them. Lines written before
  names existed have no `name` and read as unnamed.
- `review/masks.jsonl` (Alfvén eigenmode only) gets one line per mask click:
  shot, the pseudo-mask's version and sha256, the regions rejected, `name`
  and time. A shot's last line is its decision. A decision made on an older
  pseudo-mask (another sha256) is dropped: the page says so and training
  ignores it.

A page newer than its server asks `/api/version` first. From an older server it
saves without a name, hides the name box and history, and says to restart the
server, so a page reload before a restart never breaks a save.

## The row store

The page reads rows from `$LABELER_ROOT/spectrograms/<event>/<shot>.h5`. Each
file holds one time grid. Every row is stored at full resolution and again
max-pooled by 8 and by 64, so a zoomed-out view reads a small slice and a
zoomed-in one reads the full columns. The Alfvén eigenmode rows are built ahead
of time, because each shot takes about 8 s:

```bash
sbatch scripts/labeler/spectrograms.sbatch          # every roster shot; skips built ones
python -m labeler.events.review.build --event alfven_eigenmode --shots 170659 170660
```

The AE recipe resamples the four chords to 500 kHz and applies a Hann STFT of
512 samples every 128. That gives 0.256 ms columns and 257 bins up to 250 kHz.
The cross-power rows average R0 × conj(chord) over 8 columns before taking the
magnitude. Other events build their file the first time a shot is opened, which
takes a few seconds. The page says so while it waits.

A shot's CO2 comes from the corpus if the corpus has it, else from the fetch
cache `$LABELER_ROOT/raw`. Failing both, it is fetched live, which needs the
`fdp` wrapper and a login node. All 180 AE roster shots are in the fetch cache.
