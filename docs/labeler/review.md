---
title: "Label review"
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
  events show the rows of their panel builder; the four cohort editors' rows
  are under [The cohort editors](#the-cohort-editors).
- **Source** is the label the page opens a shot on. It comes from the table the
  event's `review/source.json` names, when it has one: the four cohort editors
  point at their draft, `$LABELER_ROOT/suggestions/<method>/v1/<event>_suggest_<method>_v1.csv`.
  Without a pointer it is the event's newest `format/*_format_*.csv`. **Label**
  is yours; it starts as a copy of the source. Where the two differ, a strip
  along the top of the label track marks the difference.
- **Chips**, one per roster shot, coloured by state: *unreviewed* (never saved),
  *confirmed* (saved as the source had it) or *changed*. A dot marks a shot with
  unsaved edits. Unsaved edits live in the browser until you save or revert, so
  a reload keeps them.
- **Your name.** The page first asks who is reviewing: pick your name from the
  list, or type it and press Add Name if it is not there, then press Continue.
  A new tab or window asks again, with your last name already picked; a reload
  does not. The name, at the top right (click it to change), goes with every
  save and is shown beside it (`saved Sep 26, 14:02 by Ada Lovelace`). It is an
  attribution, not a login: the history also records the login that runs the
  server. The list is `data/events/reviewers.txt`, one name per line, which you
  may edit (Add Name only appends a line); until a name is added it is the
  names already saved in the review logs.
- **History** (`H`) lists the shot's saved versions, newest first: who saved
  each one, when, how many spans, and how many ms changed. The first version
  is compared with the event's current source table; later ones with the version
  before. **Restore** loads a version as an unsaved edit; a save appends a
  version and never rewrites one, but a crash between writing the label and
  its history line can leave the current label without its version line.
- **Mask** (Alfvén eigenmode shots with a pseudo-mask): TokEye's coherent lines
  inside the label's AE frames, 80-250 kHz, over the label's whole window, drawn
  in cyan over the rows: `pseudo-v1-full`, pseudo-v1's rules
  (`labeler.ae.seg.pseudo`) over TokEye's whole-shot masks, built by
  `python -m labeler.ae.seg.whole` from the labels saved when it ran. SegNet v1
  learnt from pseudo-v1 (TokEye's 0-2 s masks), so a decision saved here is
  stale to it, as to v2 and v3. A click on a region that is not the mode (an MHD harmonic, pickup)
  rejects it, grey; a second click takes that back. Each click is saved at
  once, with your name. `M` hides and shows the mask, and the browser
  remembers which. The header counts the regions kept.
- **TokEye layer** (Alfvén eigenmode shots): every line TokEye's whole-shot
  masks light on two of the four chords, AE or not, in the band each row
  shows, drawn in faint cyan over TokEye's 0-6 s, past the label windows' 2 s:
  under the mask inside the label's window, over the veil outside it
  (`python -m labeler.ae.seg.whole` builds it into
  `segmentation/alfven_eigenmode/tokeye-full/`) from the same TokEye masks as
  the mask. A picture only: nothing trains on it and a click never lands on it.
  `M` hides it with the mask; a shot with the layer and no mask says
  "TokEye only" in the header.
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
| `M` / click a mask region | AE: hide or show the pseudo-mask and TokEye layer / reject the region, or take that back |
| `[` `]` | contrast |
| `?` | this list |

## The cohort editors

The ELM (`edge_localized_mode`), H-mode (`high_confinement_mode`), sawtooth
(`sawtooth_oscillation`) and tearing-mode (`neoclassical_tearing_mode`) editors
review the frozen cohort's 450 non-blind shots in its queue order
(`queue_rank`). Any prefix of the queue is therefore a random subsample of
every group. Blind shots are left to v1's blind review and never get a draft.
The page opens each shot nobody has saved on its draft, a suggestion table
written by `labeler.events.spans`.

A draft covers the shot's catalog window, the v1 rule-4 Ip window. Its states
are these:

- *present*: the method saw the phenomenon.
- *absent*: the inputs were measured and showed nothing.
- *uncertain*: see H-mode below.
- *not observable*: the method's inputs did not measure that time. A shot the
  method could not run on is not observable throughout, and the table's
  `.meta.json` records why under `skipped`.

The meta's `rule` holds only the constants that method uses. Its `per_shot`
records, for each shot, what the page does not show: where the draft started,
and for ELMs the filterscope read and whether the H-mode gate ran. `gold`
holds the score from the last `--gold` run.

**ELM and sawtooth drafts start in the plasma.** A span is a run of at least 3
events (ELMs at most 200 ms apart, sawtooth crashes at most 300 ms), padded by
5 ms. Only events between the plasma's start and the window's end form runs.
Events before the start are dropped before the runs are grouped, so the steps
and spikes of the ramp-up neither make a run nor join one. The start is the
first time the 25 ms centred mean of |Ip| inside the window reaches 0.8 of its
plateau (its 95th percentile), the catalog's flat-top fraction. Ip comes from
the corpus or the raw cache. A shot without Ip starts 700 ms into its window,
the median on the 450 shots. Time before the start is absent. Sawteeth in the
ramp-up are left out with it: 189061's 227-530 ms, for one. Add them by hand
where you see them.

**ELMs** (`elm_clock`).

- Rows: the CO2 R0 chord's power from 0 to 125 kHz, in dB above each
  frequency's floor over the plasma window, from -3 to 27 dB. Then two traces:
  the PCPHD03 photodiode, and the filterscope the ELM spans were found on
  (FS01, or FS02 where FS01 is dark). The filterscope row's title names its
  channel.
- PCPHD03 is left out where it cannot be read, or where it is flat over the
  plasma: its 0.5-99.5 percentile range under 0.011 V, on 33 of the 450
  shots. The filterscope row's title then says "(PCPHD03 not found)" or
  "(PCPHD03 flat, left out)".
- Both traces are clipped to their robust range: the 0.5-99.5 percentiles over
  the plasma window, widened on each side by the distance between them. The
  title says so when anything was cut, so a spike at the end of the discharge
  does not flatten the ELMs.
- Draft: the runs of ELMs, less any time the H-mode method saw the shot in
  L-mode, because the clock also counts L-mode D-alpha spikes. A shot whose
  H-mode inputs are missing keeps its runs whole, and `per_shot` records why
  under `hmode_gate`.

**H-mode** (`dalpha_lh`).

- Rows:
  - the D-alpha filterscopes FS01-FS08, what the method reads;
  - the density, the CO2 R0 chord averaged over 1 ms;
  - the NBI power summed over the beams, in MW;
  - beta_N, only on the shots the features store holds (62 of the 450).
- Draft: present from each L-H transition to the next H-L, or to the end of
  the stretch the inputs measured. An H-L with no L-H before it makes the time
  back to the transition before it, or to the start of that stretch,
  uncertain. The shot was in H-mode then, but the detector did not see it
  begin: mark the L-H where you can see it.
- 33 queue shots have no beam power (`pinj`) in the corpus or the raw cache.
  Nothing fetches it for them, so their drafts are not observable throughout.
- H-mode has no Ip start: the transitions set their own.

**Sawteeth** (`ece_sawtooth`).

- Rows: the ECE Te of channels 20-35, as four rows of four adjacent channels
  in keV, so the inversion (inner channels drop as outer ones rise) reads from
  row to row. Each sample is the median of its 0.05 ms, the page's finest
  column: the radiometer's 1-4-sample spikes (to 44 keV over a 3 keV core)
  otherwise set the row's range and flatten the crashes. Then one Te row from
  Thomson scattering (`ts_core_temp`, every 10 ms), in keV: of the chords with
  a Te on at least half as many of the plasma window's samples as the best-lit
  chord (without a window, the record's), the 4 hottest by their median over
  it. A failed fit (0 or below) is a gap. A bad fit reads high (17 keV over a
  2 keV core) and would set the row's range, so the row is clipped a quarter of
  its span above its 99.5th percentile over the window, and the title says so;
  it is not clipped below, so a ramp's cooler Te stays. Then one SXR row: the
  first fan of SX90RM1F, SX90RP1F, SX90RM1S and SX90RP1S with 8 chords finite
  over half the record.
- The SXR row draws the fan's 4 chords with the most crash-like drops over the
  Ip flat-top, not its brightest: on about 25 shots the brightest sit near
  4.6 V and barely move. A crash-like drop is a sample where the 5-sample mean
  falls by more than 6 standard deviations of its own change, taken second by
  second. The Thomson and SXR chords are chosen over the whole record,
  whatever the view.
- A shot without ECE, Thomson or SXR gets the others' rows alone.
- Draft: the runs of crashes, starting in the plasma as above.

**Tearing modes** (`window`).

- Rows:
  - The spectrogram of MPI66M322D, in dB above each frequency's 20th
    percentile over the plasma window (without one, over the columns louder
    than the record's median), from -3 to 42 dB.
  - The toroidal mode number n, drawn the way pyspecview draws a probe
    array. Every time-frequency cell of the six MPI66M midplane probes' STFTs
    takes the n from -4 to 5 whose phases fit the probes' best, and is drawn
    in that n's colour, as bright as the probes' mean power over the same
    floor and scale as the row above. A line is one colour, and each line gets
    its own n, so a 2/1 and its harmonics or a 3/2 beside it read apart. The
    key under the row's units gives each n's colour. The probes' angles are
    the measured ones in pyspecview's DIII-D probe table (MPI66M322D sits at
    317.4°); pyspecview's own fit takes the angle in the name (322°), which
    changes a line's fit by under 0.1 % at n = 1.
  - beta_N, as for H-mode.
- Sign convention: n > 0 is a mode travelling counter-clockwise seen from
  above, the co-current direction of DIII-D's normal plasma current, so a
  rotating 2/1 shows n = 1, as the catalog counts it. DIII-D's toroidal angle
  runs clockwise, and pyspecview, which fits that angle as it stands, shows the
  same mode as n = -1. On a reversed-current shot a co-current mode shows n < 0.
- Draft: there is no method yet, so the whole window is absent. It gives the
  page each shot's window.

### Commands

```bash
# The raw cache the ELM and H-mode rows draw (PCPHD03, and CO2 for old shots):
# on the login node, one editor at a time.
pixi run -e labelmaker fdp run python -m labeler.events.raw --event edge_localized_mode
pixi run -e labelmaker fdp run python -m labeler.events.raw --event high_confinement_mode
# The drafts, over the queue; --force redoes shots already drafted, --gold scores
# the drafts on the roster's gold shots into the meta (sbatch: scripts/labeler/spans.sbatch).
pixi run -e labelmaker python -m labeler.events.spans --event sawtooth_oscillation --gold
# The roster in queue order, and --point opens the page on the draft.
pixi run -e labelmaker python -m labeler.events.review.cohort_rosters \
    --event sawtooth_oscillation --point
# The row store ahead of the review; --force rebuilds built shots
# (sbatch: scripts/labeler/review_build.sbatch).
pixi run -e labelmaker python -m labeler.events.review.build --event sawtooth_oscillation --workers 8
# The gate: how close the saved labels came to the draft.
pixi run -e labelmaker python -m labeler.events.review.agreement --event sawtooth_oscillation
```

`agreement` counts a saved shot only when its last save in
`review/history.jsonl` was opened on the table the pointer names now. Saves
opened on another table, before the pointer moved, are counted under
`excluded_saves`. It reports frame precision and recall over 10 ms frames. It
is `ready` at 50 shots with both at least 0.75. `spans --gold` uses the same
scorer on the roster's gold-tier shots. Its reference is the saved labels
(`review/labels.csv`), or the label table whose path follows `--gold`. The ten
gold sawtooth shots have no gold label yet, so their score counts 0 shots, and
`missing` names the ten.

## What a save writes

Saves go under the event's directory in the label tables
(`data/events/<event>/` unless `LABELER_LABEL_TABLES` says otherwise):

- `review/labels.csv` holds the current label of every reviewed shot, in the
  format-table schema (`shot, category, t_start, t_end, confidence`, whole ms).
  A shot's rows tile its window, and the gaps are category 0. Each save replaces
  that shot's rows and rewrites the file atomically. The result validates like
  any other format table.
- `review/history.jsonl` gets one line per save: shot, `reviewer` (the login
  running the server), `name` (the reviewer's name, or null), time, the
  window and spans saved, and the source file they were compared with. It is
  only ever appended to: a shot's versions are its lines in order, numbered
  from 1, and `GET /api/history?event=&shot=` lists them. Lines written before
  names existed have no `name` and read as unnamed.
- `review/masks.jsonl` (Alfvén eigenmode only) gets one line per mask click:
  shot, the pseudo-mask's version and sha256, the regions rejected, `reviewer`
  (the server's login), `name` (the reviewer's name, or null), and time. A shot's last
  line is its decision. Its `revision` is the number of that shot's log lines,
  including decisions on older masks; it is derived from the log rather than
  stored as a separate field. GET returns the revision and each POST must name
  the revision it replaces. A successful save returns the incremented revision.
  A changed revision or pseudo-mask returns HTTP 409: the page reloads the
  current mask decisions, explains the conflict, and asks for another click.
  A decision made on an older pseudo-mask (another sha256) is dropped: the page
  says so and training
  ignores it. A failed save restores the previous rejection list and keeps its
  error for that shot across navigation, until a successful retry. Returning to
  the shot shows the error and the saved decisions.

A page newer than its server asks `/api/version` first. From an older server it
saves without a name, hides the name box and history, and says to restart the
server, so a page reload before a restart never breaks a save. From a server
older than the list of names the page keeps the typed name box.

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
