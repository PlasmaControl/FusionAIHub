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
pixi run --frozen -e labeler labeler-verify
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
  server, but the page never shows it. The credit under the queue (`170815
  reviewed by Ada Lovelace`) and the history list show names only, and a save
  made without a name credits no one. The list is
  `data/events/reviewers.txt`, one name per line, which you
  may edit (Add Name only appends a line); until a name is added it is the
  names already saved in the review logs.
- **History** (`H`) lists the shot's saved versions, newest first: who saved
  each one, when, how many spans, and how many ms changed. The first version
  is compared with the event's current source table; later ones with the version
  before. **Restore** loads a version as an unsaved edit; a save appends a
  version and never rewrites one, but a crash between writing the label and
  its history line can leave the current label without its version line.
- **Mask** (Alfvén eigenmode shots with a pseudo-mask): TokEye's coherent lines
  inside the label's AE frames, 60-250 kHz (80-250 until 2026-09-30), over the
  label's whole window, drawn
  in cyan over the rows: `pseudo-v1-full`, pseudo-v1's rules
  (`labeler.ae.seg.pseudo`) over TokEye's whole-shot masks, built by
  `python -m labeler.ae.seg.whole` from the labels saved when it ran; a rerun
  keeps a mask whose content is unchanged, and so the decisions on it. A click
  on a region that is not the mode (an MHD harmonic, pickup) rejects it, grey; a
  second click takes that back. SegNet trains on its own version's masks, not
  these, and each rejection reaches it (`labeler.ae.seg.regions.transfer`): the
  region's pixels are background in that version's mask where it scores them
  (never scored where it does not, as pseudo-v1 after 2 s), and the rest of that
  mask's line the region touches, outside the regions drawn here (below 60 kHz
  in pseudo-v2 and v3), is left unscored. Each click is saved at once, with your
  name. `M` hides and shows the mask, and the browser remembers which. The
  header counts the regions kept.
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
| `T` | show or hide the event markers (ticks over every row, one colour per label, listed in the header) |
| `M` / click a mask region | AE: hide or show the pseudo-mask and TokEye layer / reject the region, or take that back |
| `[` `]` | contrast |
| `?` | this list |

### Individual events and spans

The **Resolution** selector applies to the selected annotation and to new ones.
Choose **Span** for a stretch of events that have not been individually
delineated (the stored `iscrowd` 1, COCO's "crowd"), **Individual** for one
resolved event, or **Unspecified** when that distinction has not been reviewed.
Resolution is independent of the event category: the uncertain category still
describes the evidence, while a span describes annotation granularity.
The **Individual** and **Span** bars are separate editable lanes. Drag in the
Individual bar to delineate an event inside a span; the span stays intact.
Drag in the Span bar to annotate a stretch over existing individuals.
Clicking, moving, resizing or deleting an annotation affects its own lane.
Switching the selected annotation's resolution moves it to the other lane. Each
wide annotation shows its category and resolution; spans are also hatched.
Touching individual annotations retain their separate boundaries, even when
categories match. Within a lane, a newly drawn or moved annotation replaces the
coverage it crosses.

**New annotations default to Span, for every event** (owner 2026-10-11; before,
only ELM, sawtooth and MHD 1/1 did). This sets the type for Shift-drag on the
diagnostic rows; dragging directly in a lane uses that lane's type. The
**Alfvén eigenmode** and **confinement** events (and the AE subcategories) go
further: a label from any layer (saved, format, extend, source, suggestion) that
carries no resolution at all opens with every annotation a Span
(`labels.SPAN_EVENTS`, `labels.span_default`). A table that sets any flag keeps
its own, and nothing is written to the tables until a save. For other events,
existing labels without resolution metadata remain Unspecified; opening or
saving them does not invent individual events. Unspecified annotations appear in
the Individual bar with an Unspecified caption. Changing resolution, moving or
resizing an annotation, undo, drafts and history restore preserve the
distinction. To delineate two touching events, Shift-drag on the diagnostic
rows; dragging an existing annotation's edge resizes it.

**ELM onset candidates are in the draft** (owner 2026-10-11). The yellow ticks on
the ELM page are the jump-and-hold onsets of the D-alpha detector
(`panels.edge_localized_mode.markers`). A shot nobody has saved now opens with
each onset as a one-frame **Individual** annotation (present, on the 10 ms frame
that holds it) beside the source's ELMy-period spans, which open as **Span**
annotations. They are edited like any source label (move, resize, delete,
re-categorise, add), and Revert returns to this draft. The ticks are still drawn
(T hides them). Onsets in one frame make one annotation, onsets outside the window
are left out, and a draft of more than 1,000 annotations is not offered (the ticks
remain). Once a shot is saved the page offers the plain source and the save, no
candidates: `labels.with_onsets` is applied only by `/api/shot` to a shot with no
save, never by a save or the queue, so an unedited draft that is saved counts as
changed from the source table. `/api/shot` reports the number added as
`onset_candidates`.

Annotation boundaries still snap to whole milliseconds. A span may contain
bursts finer than this grid; resolving sub-millisecond ELM boundaries would
require a separate precision change. The selector records annotation facts;
it does not change a trained model. Model options and proposed loss/evaluation
semantics are in [Group versus individual event annotations](crowd_models.md).
The page says "Span"; the stored attribute and the API field stay `iscrowd`
(`{"iscrowd": 1}`), so saved tables, history and every reader are unchanged.

## Subcategories

A subcategory is a subset of its parent, not another category: whatever is on
as the parent may also be on as a subcategory, and siblings overlap freely.
The dropdown groups events (`MHD mode`: `mhd_n1`..`mhd_n3`); an event with
subcategories shows a bar of chips under it, `all` and one per subcategory with
its reviewed count (`1/1 (sawtooth, fishbone)`, `RSAE`). The chips and display
names come from the `taxonomy:` block of `configs/labeler/events.yaml`
(`labeler.events.taxonomy`); a new kind is one more entry there, and
`review: false` leaves an event off the dropdown and the silver set (disruption
and VDE). `default: <class>` names the class a saved label's unmarked time reads
as inside its window: detachment 1 (attached), confinement 2 (low), every other
event 0 (absent); the stored label is unchanged, and the page reads it from the
event's `default` field. A subcategory
id is `<event>.<key>` (`mhd_n1.m1`, `alfven_eigenmode.rsae`) and the address
`#<id>/<shot>` opens it.

Label the parent first. A chip opens on the shot on screen once the parent is
saved on it with something marked; before that it is disabled (`review <event>
first`). A subcategory's page shows the parent's saved label read-only above the
label lane, over the parent's rows and window. Where the parent marks nothing
is hatched, and drawing, moving and resizing stop at its edge; the server cuts
every saved span (both lanes) to the parent's spans the same way. Its queue is
the parent's shots with a saved, non-empty label. An AE subcategory draws the
mask read-only: rejecting a region is a decision on AE.

A subcategory's saves are in `<event>/subcategories/<key>/review/` (`labels.csv`,
`history.jsonl`, an optional `source.json`); each history line names the parent
version it was cut to (`parent_saved_at`, `parent_sha256`). Re-saving the parent
smaller hides the child's spans outside it on read, and a later child save cannot
bring them back; the file keeps them until then. In a child's table time that is
not a span is a gap, not an assessed absence: read it with the parent's label.

`sawtooth_oscillation`, `fishbone` and `neoclassical_tearing_mode` are reviewed
as `mhd_n1` (1/1, 2/1 and 3/1 are its subcategories) and
`improved_energy_confinement_mode` as `confinement`; their saved labels are not
touched, and the page neither lists nor opens them.

## The cohort editors

Poloidal beta, minimum safety factor and resistive wall mode also have
signal-specific rows and drafts; their class mappings, scientific limits and
preparation commands are in [Equilibrium and RWM review](equilibrium_review.md).

The ELM (`edge_localized_mode`), H-mode (`high_confinement_mode`, which the page
offers as `confinement`), sawtooth
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
- *uncertain*: see H-mode, and in v2 the ramp-up, below.
- *not observable*: the method's inputs did not measure that time; in v2 that
  includes an ELM filterscope that stopped reading (below). A shot the method
  could not run on is not observable throughout, and the table's `.meta.json`
  records why under `skipped`.

The page offers *present* and *uncertain* only (q_min and confinement add their
regimes): whatever the reviewer leaves unmarked is not observable. A draft's or a
saved label's not-observable stretch therefore opens as a gap, and does not make
a saved label differ from its source. The tables keep the state: the frame models
and scoring read it as masked.

**Versions.** `spans --version` names the table a run writes,
`$LABELER_ROOT/suggestions/<method>/<version>/`. The pages open on v1's: the
four editors' `review/source.json` name v1 tables, and `cohort_rosters --point`
writes v1's. v2's tables, beside them, add two rules to the ELM and sawtooth
drafts: the ramp-up's uncertain time and, for ELMs, dead filterscopes. The
H-mode and tearing-mode v2 tables hold v1's rows. A table keeps the rules it
was drafted under: a run into a table whose meta records other rules than the
method's now is refused and changes nothing (`check_rule`), so v1's tables keep
the old rules and a changed rule goes into a new version. A page opens on v2
only once its pointer is moved to the v2 table (`labels.write_pointer`).

The meta's `rule` holds only the constants that method uses; v2's adds
`min_dead_ms` to the ELM rule, `ramp` to both `start` rules, and the ramp-up to
the ELM `l_mode`. Its `per_shot` records, for each shot, what the page does not
show: where the draft started (`start_ms`, and `start_from`: "ip", or the
fallback and why) and, in v2, how many events made the ramp-up uncertain
(`ramp_events`); for ELMs also the filterscope read (`channel`), whether the
H-mode gate ran (`hmode_gate`: "ran", or why not) and, in v2, how many ms of the
window that filterscope's dead stretches take (`dead_ms`). `ramp_events` and
`dead_ms` appear only when not zero. `gold` holds the score from the last
`--gold` run.

**ELM and sawtooth drafts start in the plasma.** A span is a run of at least 3
events (ELMs at most 200 ms apart, sawtooth crashes at most 300 ms), padded by
5 ms. Only events between the plasma's start and the window's end form runs.
Events before the start are dropped before the runs are grouped, so the steps
and spikes of the ramp-up neither make a run nor join one. The start is the
first time the 25 ms centred mean of |Ip| inside the window reaches 0.8 of its
plateau (its 95th percentile), the catalog's flat-top fraction. Ip comes from
the corpus or the raw cache. A shot without Ip starts 700 ms into its window,
the median on the 450 shots.

**The ramp-up**, from the window's start to the plasma's, is where v1 and v2
differ. In v1 it is absent, and the events in it are left out with it: 189061's
sawtooth at 227-530 ms, for one. In v2 it is uncertain where the detector saw
any event there, one is enough, and absent where it saw none (`ramp_up`):

- Sawtooth: the whole ramp-up, once the ECE array saw a crash in it. 189061's
  is uncertain from 6 to 557 ms, on 7 crashes. 443 of the queue's 450 v2
  drafts open this way, 43 of them on one or two crashes.
- ELMs: the ramp-up less the time the H-mode draft calls absent (L-mode),
  piece by piece. Each piece that holds an ELM is uncertain, and the L-mode
  time stays absent, so a ramp-up all in L-mode stays absent. The H-mode
  draft's uncertain time, and time it did not measure, are not L-mode, so ELMs
  there count: 189324's ramp-up is uncertain at 8-302 ms, where the H-mode
  draft is uncertain too (56 ELMs), and absent at 302-830 ms. Without the
  H-mode gate the ramp-up is one piece. 80 of the 450 v2 drafts have an
  uncertain ramp-up, 13 of them without the gate.

Like any span, the uncertain time is clipped to what the inputs measured. The
ramp-up's events still neither make a run nor join one, so in either version
mark the sawtooth or ELMs you see there by hand.

**ELMs** (`elm_clock`).

- Rows: the CO2 R0 chord's power from 0 to 125 kHz, in dB above each
  frequency's floor over the plasma window, from -3 to 27 dB. Then two traces:
  the PCPHD03 photodiode, and the filterscope the ELM spans were found on
  (FS01, or FS02 where FS01 is dark). The filterscope row's title names its
  channel. Optional energy rows show the **diamagnetic loop**, corrected for
  linear drift and scaled to EFIT WMHD, then provisional **ELM energy loss**
  in kJ and as a percentage of the pre-ELM energy. Drops are detected on the
  loop and matched one-to-one against D-alpha peaks before a size is assigned.
- EFIT WMHD is the calibration reference. Linear drift is fitted only over
  explicit quiet baseline windows. Noise estimation and loss sizing stay
  inside the calibration interval; all energy rows are unavailable outside it.
  Each accepted drop and D-alpha peak must have a unique admissible counterpart;
  competing matches remain unsized. Native loop times and gaps are retained;
  missing or overlapping pre/post windows leave the event unsized. Drift fit,
  gain, calibration residual, detector settings and measured losses are saved
  with the diagnostic row store. Without a calibrated loop the other rows
  remain available. The local signal contract, defaults and scientific limits
  are in [Diamagnetic ELM energy](elm_energy.md).
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
  under `hmode_gate`. In v2 the ramp-up is uncertain where it holds ELMs
  outside L-mode (above).
- Dead filterscopes (v2): a stretch of at least 200 ms (`min_dead_ms`) over
  which the filterscope read repeats one value is a channel that stopped
  reading (`dead_stretches`). It is not observable, not absent, and no span
  covers it; `dead_ms` gives its length inside the window. 186636's FS02 is
  dead from 3877 ms to the window's end at 7285 ms. Only the filterscope the
  draft reads is checked, and the draft never switches to another, even when
  another filterscope is live over the dead stretch. The H-mode editor draws
  each lit filterscope. v1 has no such rule: its drafts call that time absent.

**H-mode** (`dalpha_lh`).

- Rows:
  - the D-alpha filterscopes FS01-FS08, what the method reads, each clipped
    to its robust range over the plasma window as the ELM traces are (the
    title says so when anything was cut);
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
- The page offers this editor as `confinement`, and not `high_confinement_mode`,
  `low_confinement_mode`, `quiescent_high_confinement_mode` or
  `wide_pedestal_quiescent_high_confinement_mode`: a span is 1 high, 2 low, 3 qh,
  4 wpqh, 5 uncertain or 6 imode (I-mode; the former
  `improved_energy_confinement_mode` event, folded in), and unmarked time inside a
  saved label's window is 2 low (the event's default; outside the window it is a
  gap). It shares these rows. Its
  draft is this one renumbered (present is high, uncertain is 5, absent and not
  observable are left unmarked), except on the shots that have curated regime
  intervals (Gill's and Butt's: H, L, QH, WP), which open on those. Its roster is
  the 450 queue shots and then the 389 curated shots that have a D-alpha record
  (corpus or raw cache); the other 45 curated shots have none yet, and nothing
  fetches it for them while the page runs.
  See `docs/labeler/events/confinement/README.md`.

**Sawtooth** (`ece_sawtooth`).

- ECE rows use calibrated per-channel normalized poloidal flux (`ece_psi`)
  and local EFIT `qpsi`, when available, to separate **core q<1** and
  **outside q=1**. These let the reviewer compare a core temperature drop
  with an outer temperature rise. The surface is the innermost q=1 crossing
  connected to an axis with q<1; reverse-shear profiles with axis q>=1,
  missing profiles, and profiles without that crossing leave membership
  unknown. Membership is retained between geometry samples only when both
  neighboring samples agree, without extrapolation across missing data.
- Without flux calibration, explicit local `ece_q` supports rows named
  **q<1 inversion group** and **q>1 inversion group**, which do not assert
  spatial core membership. With neither calibration, channels 20-35 remain
  as four rows of adjacent channels labeled **inversion side A/B** and
  **q=1 mapping unavailable**. An additional comparison subtracts each
  channel's full-record mean and averages channels 20-27 versus 28-35 on one
  axis; opposite changes are visible and the baseline is stable on zoom.
  Channel number alone does not establish which side of q=1 a channel sees.
- Live source checks on 189061 retrieved the 48 channel frequencies and
  validity flags, viewing height/angle, and EFIT01 flux grid and q profile.
  EFIT established an axis-connected q=1 surface at 69 of 294 native times
  (25 of 172 times in the trial 1.8–5.3 s interval). A separate research plot
  estimates positions from cold second-harmonic resonance using the full
  magnetic-field magnitude. It does not supply the editor's calibrated
  `ece_psi`: relativistic shifts, radiation transport, cutoff/optical depth,
  fast-channel validity and q-profile uncertainty remain unvalidated.
  OMFIT's [sawtooth channel selection](https://omfit.io/_modules/omfit_classes/omfit_elm.html#OMFITelm.select_sawtooth_signal)
  documents the frequency/validity sources and a cold-resonance mapping; its
  approximate detector selection does not validate a channel's q=1 membership.
- Each ECE sample is the median of its 0.05 ms, the page's finest
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
  whatever the view. The SXR chords are clipped to their robust range over
  the plasma window, as the ELM traces are, and the title says so when
  anything was cut. The ECE rows are not: each sample is already its
  0.05 ms median.
- A shot without ECE, Thomson or SXR gets the others' rows alone.
- Draft: the runs of crashes, starting in the plasma as above. In v2 the
  ramp-up is uncertain once the array saw a crash in it (above).

The calibrated view expects an `ece_psi` feature group in the features store,
corpus, or raw cache: `x` is seconds, `y` has shape `(ECE channels, times)`
in normalized poloidal flux, with the same channel order as `ece`. This is
neither channel radius nor normalized toroidal flux. Local canonical EFIT
`qpsi` has the uniform normalized-poloidal-flux grid from 0 to 1. An optional
`ece_q` group uses the same shape/clock and contains the safety factor at each
channel's location. The current corpus's ECE channel numbers alone do not
provide this calibration, so those shots use the labeled fallback comparison.

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
pixi run -e labeler fdp run python -m labeler.events.raw --event edge_localized_mode
pixi run -e labeler fdp run python -m labeler.events.raw --event high_confinement_mode
# The drafts, over the queue; --force redoes shots already drafted, --gold scores
# the drafts on the roster's gold shots into the meta (sbatch: scripts/labeler/spans.sbatch).
# --version v2 writes beside v1's tables, which refuse a rerun under today's rules.
pixi run -e labeler python -m labeler.events.spans --event sawtooth_oscillation --version v2 --gold
# The roster in queue order, and --point opens the page on the v1 draft.
pixi run -e labeler python -m labeler.events.review.cohort_rosters \
    --event sawtooth_oscillation --point
# The row store ahead of the review; --force rebuilds built shots
# (sbatch: scripts/labeler/review_build.sbatch).
pixi run -e labeler python -m labeler.events.review.build --event sawtooth_oscillation --workers 8
# The gate: how close the saved labels came to the draft.
pixi run -e labeler python -m labeler.events.review.agreement --event sawtooth_oscillation
```

`agreement` counts a saved shot only when its last save in
`review/history.jsonl` was made against the table the pointer names now: the
save's `source_sha256` is that table's sha256, so a table rebuilt under its own
name leaves out the saves made against the old one. A save with no
`source_sha256` (a line written before the page recorded it) is matched by the
table's name. Saves made against another table, before the pointer moved, or
against this one before it was rewritten, are counted under `excluded_saves` by
that table's name. The table is hashed when the save is made, not when the page
opened the shot: a draft begun before a pointer moved is recorded against the
new table, so save or discard pending drafts before moving a pointer. It
reports frame precision and recall over 10 ms frames. It is `ready` at 50 shots
with both at least 0.75. `spans --gold` uses the same scorer on the roster's
gold-tier shots. Its reference is the saved labels (`review/labels.csv`), or the
label table whose path follows `--gold`. The ten gold sawtooth shots have no
gold label yet, so their score counts 0 shots, and `missing` names the ten.

## 10 ms grid

Every review event and subcategory uses the same grid: frame k covers
[10k, 10k+10) ms. Boundaries round to the nearest frame edge, halves up;
a window's ends snap the same way. A short span that would vanish keeps the
frame containing its midpoint; a zero-length point is dropped. The page
and server use the same rule. Evidence-bearing tables snap shared boundaries
and drop collapsed spans so their explicit coverage still tiles the window.
The page, its queues and the silver view read every table through one rule,
`labels.page_label` (the page's categories, then the grid). `read_labels`,
`read_saved` and `read_source` keep whole-ms values by default, so evaluation,
training and the models' frozen label copies read what was written.

## Silver and gold sets

The page's `Silver set` entry (first in the event list) opens the 450-shot list that is
reviewed for every event, in five batches of 90 (`data/events/silver/`). The `Gold set`
box beside the event list offers only the gold set's 50 shots (`data/events/gold/`:
shots from 170000 up with hand labels or a curated source, every observed class, a few
run days, about the same number from each shot-number decade 17xxxx-20xxxx, plus three the owner
picked from the 2024-07-11 tearing-mode experiment) on
whichever event is open, including those the event's own roster lacks; the choice is
remembered in the browser. A shot on neither list is still opened by typing its number.
Silver and gold share no shot.

## What a save writes

Saves go under the event's directory in the label tables
(`data/events/<event>/` unless `LABELER_LABEL_TABLES` says otherwise). Each save is
first one line in the account's own file, `review/saves/history.<login>.<host>.jsonl`
(`masks.<login>.<host>.jsonl` for a mask click), and is then merged into the files
below, so several accounts can save into one folder; see
[Shared label folder](shared_labels.md). The files below are what everything reads:

- `review/labels.csv` holds the current label of every reviewed shot, in the
  format-table schema (`shot, category, t_start, t_end, confidence`, on the
  10 ms grid since 2026-10-10).
  Explicit resolution adds the optional `attrs` JSON column, with
  `{"iscrowd":0}` for an individual and `{"iscrowd":1}` for a span. Blank
  attrs remain unspecified. Category 0 gaps carry no `iscrowd` flag. Other shots'
  attrs are preserved; editing a shot with additional unsupported attrs is
  refused so the editor cannot discard its metadata.
  Individual/unspecified annotations may overlap span annotations. Shared category-0 gaps
  cover only time outside both lanes. Each save replaces
  that shot's rows and rewrites the file atomically. The result validates like
  any other format table.
- `review/history.jsonl` gets one line per save: shot, `reviewer` (the login
  running the server), `name` (the reviewer's name, or null), time, the
  window and spans saved, the source file they were compared with and that
  file's sha256 (`source_sha256`; absent when the event has no source table).
  Explicit resolution is stored as a parallel `iscrowd` array aligned with
  the intervals, also returned by the label/history API; null entries mean
  unspecified. Older history entries without it remain unspecified (the history view
  restores them as spans for the span events above).
  Saves add lines (a save made earlier on another account that reaches the merge late
  goes before the later ones; only owner scripts, the Jalal import's rename and
  withdrawal, ever rewrite one, under the folder's lock). Every process that saves into
  a folder must run the code that writes `review/saves/`; an older server's append can be
  lost (see [Shared label folder](shared_labels.md)). A shot's versions are its lines in
  order, numbered from 1, and `GET /api/history?event=&shot=` lists them. Lines
  written before names existed have no `name` and read as unnamed; lines
  written before the sha256 was recorded have no `source_sha256` and match the
  source table by name.
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
Span (`iscrowd`) controls require API version 7; overlapping lanes require version 8.
An older client cannot overwrite
`iscrowd`-bearing labels without supplying the resolution array, and a newer
page refuses to send an `iscrowd`-bearing draft to an older server. Version-8 saves
send `overlap_edit: true`; clients without this marker cannot overwrite a shot
whose saved or source annotations overlap. This prevents a cached older page
from silently flattening the lanes. Restart an older server and reload the page
to use overlapping annotations.

## Detachment cameras

The `detachment` editor offers spans **1 attached, 2 detached, 3 marfe,
4 uncertain**; unmarked time inside a saved review's window reads as 1 attached
(the event's default), and time outside it is not reviewed. The Individual lane starts from
the producer label, so reviewers correct suggestions. **Start blank** clears
the editable lane while keeping Source visible. Drag **Time (ms)** or click the
timeline to seek the nearest camera frame and move the cursor across all
panels. The video panel stays visible while scrolling. **Play/Pause** waits for
decoded frames before advancing. All selected camera images, physical views,
captions and the shared clock publish together; pausing discards staged frames.
TangTV corpus images are 50 Hz linear resamples blending adjacent exposures;
seeking shows a loading state. Labels still use the
usual timeline lanes, saves and history.

Each available `bolo`, `tangtv` or `irtv` image channel uses a fixed per-shot
grayscale scale. Missing/stub cameras have a one-line unavailable card.
Selectors name the source views; the default is lower-divertor TangTV channel
**2, then 0** if 2 is unavailable. The TangTV filter/emission line is not recorded.
Raw Langmuir sweeps are omitted; bolo raw chord voltages have a
qualitative chord×time context row, and optional Jsat peaks show probe ids; filterscope, density and gas-flow context uses block means over the
plasma window. Optional producer files add validity-gated Afrac, lower-divertor
radiation fraction and normalized TangTV DZ. Movies stay in the per-shot store;
the page requests one frame at a time. Camera controls require server API 9.
The state track is exclusive: MARFE can physically coexist with detachment,
and partial detachment has no separate state. The queue is the delivery overlay
`round4/detach-ui/tables/detachment/shots.csv`, gated on camera availability,
separately from the EFIT shelf gate. Frozen delivery counts are recorded in
[the fix-five delivery record](results/detachment_ui_fix5_delivery.json): 247
queued shots, 87 with camera/shelf coverage. The earlier “94” shelf-sample count
is superseded. `results/detachment_review_queue.json` is a future controller
rebuild output; it is not present evidence. Blind test and reserved shots are excluded.
When DRSEP is absent, the page says **topology unavailable**, alongside the
lower outer strike-point gate. See the
[human camera protocol](detachment_review.md#human-camera-review-protocol) and
the page's separate **What the machine label means** block, generated from the
producer recipe stored with the shot.

See [Detachment camera review](detachment_review.md) for data contracts,
launch commands and the isolated real-shot screenshot check.

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

ELM and sawtooth row stores carry a panel version. Opening an older store
rebuilds it once to include the diamagnetic energy rows and revised ECE view.
If calibration or previously missing diagnostic data are added later,
rebuild the affected shots explicitly:

```bash
pixi run --frozen -e labeler python -m labeler.events.review.build \
    --event sawtooth_oscillation --shots 192238 --force
```

A shot's CO2 comes from the corpus if the corpus has it, else from the fetch
cache `$LABELER_ROOT/raw`. Failing both, it is fetched live, which needs the
`fdp` wrapper and a login node. All 180 AE roster shots are in the fetch cache.
