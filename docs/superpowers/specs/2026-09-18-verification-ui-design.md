# Browser verification surface for event labels

Replace the sixteen per-event `verification.ipynb` notebooks with one
browser application, served the way `shot_design serve` is served, and route
every raw-signal read through the corpus.

`alfven_eigenmode` is the acceptance target: it must work end to end on a
real shot before anything here is called done. The other fifteen events are
registered against a generic builder and are expected to render, but are not
validated by this work.

## Why

The notebooks are three separate problems wearing one coat.

The panel code that decides which traces settle a phenomenon lives in a
notebook cell, so it cannot be called from anywhere else, cannot be tested,
and exists in sixteen copies that drift. Twelve of those copies are the same
placeholder (`ip`, `betan`, `pinj_total`) with a paragraph asking the reader
to replace it.

The window is fixed. `alfven_eigenmode`'s notebook hard-codes `t_range =
(0.0, 2000.0)` and block-averages to a thousand time bins, because a
full-rate spectrogram of even that window is ~3300 columns and three of them
as plotly heatmaps wedges the tab. A reviewer cannot see the rest of the
shot, and cannot zoom into a feature to decide where its boundary sits --
which is the one thing the surface exists for.

Raw data comes from three unrelated places: the corpus, a per-event npz
cache under `data/events/<event>/review/_cache/`, and, in
`example.ipynb`, a hard-coded `/scratch/gpfs/nc1514/aemodes/data/.cache/`
path belonging to a different project entirely.

## What does not change

`src/labeler/events/verify.py` is not edited. `write_corrections`,
`correction_path`, `corrections_for`, `record_review`, `ReviewSession` and
`Panel` all stay exactly as they are, and the existing `tests/labeler`
coverage of them runs green without a single edit. That is the check that
the write path really did not move.

Corrections stay append-only: every save writes its own
`review/<shot>__<reviewer>__<stamp>.csv`, `write_corrections` refuses a path
that already exists, and nothing there is ever overwritten or deleted.

What changes is only which of those functions the app calls. See
"`shots.csv` stays manual" below.

`Panel` and `label_panel` stay in `verify.py`. The registry composes them; it
does not redefine them.

## `shots.csv` stays manual

`shots.csv` is how a person tracks what has been processed and what has not,
and it is the file holding the hand-set `tier` and `holdout` calls. It stays
a file people edit, so that it stays simple and so that nobody overwrites
somebody else's curation by clicking something.

So the app never writes it. Save writes the corrections file and nothing
else. Concretely, the app calls `write_corrections` directly rather than
`ReviewSession.save()`, because `save()` also calls `record_review`, which
edits `shots.csv` in place. `ReviewSession` itself is left alone for any
caller that still wants that behaviour.

The roster instead appears as a read-only panel: for each shot, the `tier`,
`holdout`, `reviewers` and `verified_on` as recorded, beside the count of
correction files actually on disk under `review/`. A review you just saved
shows as a correction file against an unrecorded reviewer, which is the
prompt to go edit the row by hand -- and also the one view that shows the two
disagreeing.

## Components

### `src/labeler/events/raw.py` -- one way to get a signal

```python
raw_signal(shot, group, *, channels=None, t_range=None) -> FeatureArray
```

Three tiers, in order:

1. `/scratch/gpfs/EKOLEMEN/foundation_model/<shot>_processed.h5`
2. `<repo>/.cache/raw/<shot>_processed.h5`
3. fetch over fdp/toksearch, write tier 2, return

Contract matches `corpus_signal`: `x` in milliseconds, `y` as `(C, T)`
float32, one row per channel in the order asked for. A `Panel` cannot tell
which tier its array came from.

The two roots are different kinds of storage and the split is deliberate.
EKOLEMEN is the long-term home for bulk raw signal data and has the capacity
for it. The project directory has decent room but is meant for temporary and
smaller things -- labels, tables, outputs -- so a fetch lands there as scratch
and stays scratch until somebody decides otherwise. `.cache` is already
gitignored (`.gitignore:49`), and deleting `.cache/raw/` at any time is safe:
the next open of that shot refetches it.

Both tiers use the corpus's own layout -- flat groups holding `xdata` and
`ydata`, as described at `src/labeler/features/resolve_corpus.py:3` -- so
promotion is a move and nothing else changes.

Writes are temp-file-plus-rename, and additive: fetching `ece` for a shot
whose cache file already holds `co2` adds a group rather than replacing the
file. The server will be doing this concurrently with itself.

Cost: about 240 MB per AE shot, four CO2 chords at 1.667 MHz as float32.
That is a lot for the project directory, which is the reason the cache is
disposable rather than durable. There is no eviction and no cap -- exactly as
the npz cache it replaces had none -- but `labeler raw clean` removes the
whole directory, and the README says what it costs.

`fdp_signal` survives as the fetch primitive but loses its `cache=`
argument. `raw.py` owns caching now. The npz cache directory under
`data/events/<event>/review/_cache/` is deleted, as are `TS_CACHE` and the
feather-reading `read_co2` in `data/events/alfven_eigenmode/example.ipynb`.

Live fetches only work under the fdp wrapper. `FDP_RESTART_COMMAND` in
`verify.py:125` records what happens without it: PTDATA fails with
`getservbyname failed for task 'PTSERVER'`, MDSplus with `TREE-E-FOPENR`.

### `labeler raw promote <shot>` -- scratch to long-term

Moves `.cache/raw/<shot>_processed.h5` into the corpus root. Separate and
manual: nothing moves hundreds of megabytes as a side effect of clicking
Save, and a promotion that needs repeating is one command rather than a
review done again.

It refuses a shot whose groups are incomplete, because training loaders glob
`*_processed.h5` in the corpus root --
`src/tokamak_foundation_model/ignite/spike.py:220`,
`fastts_train.py:214`, `train_codec.py:633` -- and a verification fetch
materializes the one group a panel asked for, not all thirty-two. Such a
file in the root is one those globs hand to training with thirty-one groups
missing. `--partial` overrides the refusal for someone who knows they want
it; the refusal names the groups that are present and the ones that are not.

A promotion is a move, not a copy: afterwards tier 1 serves the shot and the
cache entry is gone.

### `src/labeler/events/panels/` -- what to plot

```
__init__.py              build(event, shot, *, t_range) -> list[Panel]
                         BUILDERS: dict[str, Builder]
_generic.py              ip / betan / pinj_total
alfven_eigenmode.py      CO2 crosspower over three chord pairs
fishbone.py
sawtooth_oscillation.py
minimum_safety_factor.py
```

Each builder is `(shot, *, t_range) -> list[Panel]`, lifted from the
notebook cell it replaces. An event with no entry falls back to `_generic`,
which is what the twelve placeholder notebooks already were.

A builder also carries a `guidance` string -- the prose the notebook's
markdown cell held about what to look for. The page shows it beside the
panels, which is where a reviewer is actually looking.

`alfven_eigenmode.py` keeps `crosspower()` verbatim, including its two
load-bearing comments: the sample rate is derived from the span and never
from a median of successive differences, because a float32 time vector
quantises its spacing at t ~ 3 s; and the block average is taken over the
power with the log applied after, because averaging the log is a geometric
mean that is pulled down by quiet bins and so suppresses exactly the short
bursts the panel exists to show.

Its band is 80-250 kHz over chord pairs R0xV1, R0xV2, R0xV3. A feature on a
single pair is more likely chord-specific noise than a real mode.

### `src/labeler/events/ui/` -- the surface

Ported from `src/shot_design/ui/`, structure for structure:

```
app.py            FastAPI, token gate, JSON endpoints
serve.py          launcher; binds 127.0.0.1, prints token link + ssh line
static/index.html
static/app.js     vanilla JS, plotly from the browser
static/style.css
```

Endpoints:

```
GET  /?token=...            the page; sets the cookie
GET  /api/events            registered events, with roster counts
GET  /api/shots?event=      roster rows, read-only, unioned with the shots
                            present in format/, plus per-shot correction
                            counts from review/
GET  /api/panels?event=&shot=&t0=&t1=
                            panels over [t0, t1] at 1000 time bins
POST /api/save              {event, shot, marks[]}
```

`/api/shots` unions the roster with `format/`'s shots because
`alfven_eigenmode/shots.csv` currently holds three example rows and no real
shot, while the annotated set spans 170659-178879. A picker fed by the
roster alone would be empty. The union is a read: it is presented, never
written back.

Resolution: the page opens on the whole shot at a thousand bins, and
re-requests the visible window at a thousand bins on pan or zoom, debounced
about 200 ms, driven by plotly's `relayout`. A window is always as sharp as
the screen can show, whatever its width. The raw record sits in a
server-side LRU of two or three shots, so a re-render is a spectrogram and
not a refetch -- warm round trip in the low hundreds of milliseconds, against
several minutes for the fetch.

Marking keeps the notebook's semantics, with one deliberate subtraction.
Box-select a range; *Mark present* and *Mark absent* accumulate in memory;
*Save* posts, and the server calls `write_corrections`. There is no *Verify*
button, because the only thing it did was drive the `shots.csv` write the
app no longer performs. Two guards carry over into JS: a lasso writes `{type: "path"}` with `x0`/`x1`
both `None` and is rejected in favour of box select, and the selection is
cleared after a mark so a second click cannot silently record the same
interval twice.

The page also carries a fixed block saying what *Save* writes: one new file
under `review/`, never an overwrite, and nothing else. Not `shots.csv`, not
`format/`, and no promotion of raw data to EKOLEMEN -- those are three
separate deliberate acts. That text is a corrected version of what sits in
every notebook's closing markdown cell.

### Launch

```
pixi run -e labelmaker fdp run python -m labeler.events.ui
```

`serve.py` refuses any host but `127.0.0.1`, mints a token, prints the link
and the `ssh -L` line, exactly as `shot_design/ui/serve.py:27` does. It also
checks for the fdp wrapper at startup and prints `FDP_RESTART_COMMAND`'s
equivalent if it is missing, rather than letting the first fetch fail with a
`PTSERVER` error several minutes in.

`fastapi` and `uvicorn` are added to `[tool.pixi.feature.labelmaker.dependencies]`.
That feature already declares `plotly` but neither of those.

## Removals

The sixteen `data/events/*/verification.ipynb` files are deleted. Each
event's `README.md` gains the launch line in their place.

## Testing

`tests/labeler/test_events_raw.py`

- the three-tier lookup returns from each tier in order
- a cache write is additive: a second group joins the file
- a cache write is atomic: an interrupted write leaves no partial file
- the cache honours `channels` and `t_range` the way `corpus_signal` does
- a promoted shot reads identically from tier 1 as it did from tier 2
- `promote` refuses an incomplete shot, names the missing groups, and moves
  it under `--partial`
- `clean` removes the cache and a subsequent read refetches

`tests/labeler/test_events_panels.py`

- every registered event builds panels against a synthetic shot
- an unregistered event falls back to `_generic`
- `crosspower` derives its rate from the span: a float32 time vector whose
  successive differences quantise still yields the right frequency axis

`tests/labeler/test_events_ui.py`

- no token, or a wrong one, is refused
- `/api/panels` honours `t0`/`t1` and returns a thousand bins at any width
- `/api/shots` unions the roster with `format/` and reports correction counts
- `/api/save` writes one corrections file, and a second save writes a second
  file rather than overwriting the first
- **`/api/save` does not modify `shots.csv`** -- asserted on the file's bytes
  before and after, because this is the guarantee the reviewer is being
  given and the one an accidental `ReviewSession.save()` would quietly break

Existing `verify.py` tests run unchanged.

## Acceptance

Against `alfven_eigenmode`, on a real annotated shot, over the full shot
rather than 0-2000 ms:

1. `pixi run -e labelmaker fdp run python -m labeler.events.ui` prints a
   token link; the link opens the page over an SSH forward.
2. A shot not in the corpus fetches, writes `.cache/raw/<shot>_processed.h5`
   in the project directory, and draws three crosspower panels plus the
   label row over the whole discharge.
3. Reopening that shot is instant and touches no network.
4. Panning and zooming re-renders at the visible window's resolution.
5. Box-select, *Mark present*, *Save* writes
   `review/<shot>__<reviewer>__<stamp>.csv`; a second save writes a second
   file and destroys nothing; `shots.csv` is byte-identical afterwards.
6. `labeler raw promote <shot>` refuses the partial shot by name, moves it
   under `--partial`, and the shot then reads from EKOLEMEN with the cache
   entry gone.

The remaining fifteen events are registered and expected to render their
generic panels. They are not part of acceptance.

## Out of scope

Eviction or a size cap for `.cache/raw/`; `clean` is the whole story.
Any change to what `tier` or `holdout` mean, or any automatic writing of
`shots.csv`. Merging `review/` rows into `format/`, which stays a manual
step. Public network exposure -- the server binds the loopback and is reached
by SSH forward, as `shot_design` is.
