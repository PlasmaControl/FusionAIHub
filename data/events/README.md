# Event label datasets

Each category keeps its untouched originals, a small interval table, and any
producer extensions. `events.yaml` lists the raw sources, output filenames,
formatters, and shared storage conventions.

```text
data/events/<category>/
  raw/                                      # unchanged original data
  formatter.py
  example.ipynb                             # executed reading/plotting example
  shots.csv                                 # review roster: who looked at what
  verification.ipynb                        # review one shot, take corrections
  format/
    <dataset>.csv
    <dataset>.meta.json
    shots/
      <shot>.npz                            # saved time × rho labels
  extend_<producer>/
    <shot-list>.csv
    <shot-list>.meta.json
    <shot-list>/
      <shot>.npz                            # sampled time × rho labels
  review/
    <shot>.csv                              # what a human asserts after looking
    _cache/                                 # fetch cache, not a claim; deletable
```

## Interval CSVs

Both `format/` and `extend_<producer>/` use exactly these columns:

```csv
shot,category,t_start,t_end,confidence
170815,1,450.0,700.0,
```

Times are milliseconds. Equal start and end times represent a point event.
Blank confidence means unknown. Source, run IDs, raw checksums, and
conversion assumptions belong in the JSON metadata, not repeated CSV columns.
Original annotations remain in `raw/`; internal event products retain their
full provenance separately. Overlapping annotations can remain separate rows.
The per-shot binary grid represents their union.

The interval table is retained even for large exports; it is no longer replaced
by a shot-count summary above 50,000 rows. The CSV category is a nonnegative integer, matching the grid IDs. Every JSON
sidecar has a `categories` ID-to-name mapping, also listed under **Category**
in the event README. Binary labels use 0=absent, 1=present. Missing is not 0.

`labeler.events.interval_tables.validate_intervals` validates this public
schema. The event loader reads it through `events.yaml` and converts milliseconds
to seconds internally. It also supports the older internal event-table schema.
Undated `format_datasets` entries are pending and are not loaded.

## Review rosters

`shots.csv` in each category is a shot-level roster of who has looked at what.
It is not the category's shot list: a shot enters it when somebody puts it up
for review, and the interval tables stay the record of what is labelled.

```csv
shot,tier,holdout,reviewers,verified_on,notes
170815,gold,false,nc1514;aj17,2026-09-17,retimed first onset -30 ms
178631,silver,false,nc1514,2026-09-17,
185945,unverified,false,,,
```

`tier` is a curation judgement, set by hand, and says nothing about how many
people have reviewed a shot: `gold`/`silver`/`unverified` are legal values,
not a count. `holdout` is a required `true`/`false` reserving a shot from
training and tuning for final evaluation only; blank is invalid. Reviewer
ids are `$USER`, separated by `;`, in the order they reviewed. A reviewer
appears at most once per shot, so pressing Verify twice re-dates the row
without touching `tier`. `verified_on` is the most recent review, blank when
unverified. `notes` is one line.

Ten gold shots per category is the target, not something the file enforces.
The three placeholder rows every category ships are meant to be deleted.

`labeler.events.rosters.validate_roster` validates this schema: legal `tier`
and `holdout` values, no duplicate shots or reviewers, and `verified_on`
agreeing with `reviewers`. It does not check `tier` against the reviewer
list.

## Verification notebooks

`verification.ipynb` in each category shows one shot's signals against its
saved labels, where there are any, and takes back corrections:

```bash
pixi run -e labelmaker jupyter lab data/events/<category>/verification.ipynb
```

A category with no saved grid yet - `fishbone` and `sawtooth_oscillation`
among the four hand-written ones, and any category whose `format/shots/` and
`extend_*/` are still empty - has no label row to show. `review()` warns, puts
`NO LABEL ROW` in the figure title and stacks the signal panels alone; the
reviewer's own marks are then that category's first labels.

Set `shot`, drag a time range on any panel, then press *Mark present* /
*Mark absent*, *Verify* and *Save*. Nothing touches disk until *Save*, which
writes two files: the corrected intervals to `review/<shot>.csv`, in the same
five-column schema as `format/`, and the reviewer's name into `shots.csv`.

Corrections under `review/` are a separate claim from `format/` and
`extend_*/`: they are what a human asserts after looking. No formatter reads
or overwrites them, and merging them back into a formatted table is not yet
decided.

`review/_cache/` is the one thing under `review/` that is not a human claim.
The two fetching notebooks cache each shot's raw fetched record there, as
`<shot>_<group>.npz`, so that reopening a shot costs nothing: about **240 MB
per `alfven_eigenmode` shot** (four CO2 chords at 1.667 MHz) and about **62 MB
per fetched `sawtooth_oscillation` shot**. It has no cap and no eviction. It is
safe to delete at any time - the next open of that shot simply refetches it -
and `*.npz` is gitignored (`.gitignore:261`), so it cannot be committed by
accident. Anything that later reads reviewer output by globbing `review/*` has
to skip it.

Four categories have panels chosen for the phenomenon -
`minimum_safety_factor` (qmin against the rule's class thresholds),
`sawtooth_oscillation` (raw ECE channels 20-35, four to a row),
`alfven_eigenmode` (CO2 crosspower R0xV1/V2/V3) and `fishbone` (the magnetic
spectrogram). The rest carry generic `ip`/`betan`/`pinj_total` panels, which show
that a shot exists and not that a phenomenon happened; each says so and asks
to be replaced. Scaffold a new one with:

```bash
pixi run -e labelmaker python scripts/labeler/make_verification_notebook.py --event <category>
```

Two categories fetch, and both need their kernel started under the `fdp run`
wrapper (`pixi run -e labelmaker fdp run jupyter lab`):

- `alfven_eigenmode`, for every shot: its 180 annotated shots are
  170659-178879 and the corpus covers 185601-204999, so its notebook pulls
  the CO2 chords from PTDATA.
- `sawtooth_oscillation`, for 8 of its 10 rostered shots (178640, 178641,
  178642, 179310, 180090, 180406, 180627 and 184084 all predate the corpus),
  over MDSplus, at 80-145 s PER CHANNEL on the first fetch. Only 192238 and
  195032 are in the corpus.

## Per-shot sampled grids

Every extension writes one compressed `.npz` file per requested shot. The logical
array is **time × 20 rho bins**, with rho edges `0, 0.05, …, 1.0`.
Values are binary 0/1 or nonnegative integer class IDs. These are actual sampled
label values, not transition markers. Storage is sparse:

- `time_ms`: the time coordinate for each row.
- `rho_edges`: the 21 edges defining the 20 radial bins.
- `shape`: the two-dimensional label-array shape.
- `indices`, `values`: coordinates and integer values of nonzero cells.
- `unknown_indices`: cells with missing/unknown values; separate from zero.

When a dataset has no radial localization, each scalar time label is broadcast
across all 20 rho bins. An active binary sample fills the whole radial column
with 1; an integer class fills it with that class ID. This is the requested
storage assumption, not a measured radial extent.

The extension exporter aggregates stored intervals into half-open 50 ms bins
(start coordinates 0–5950 ms). Any event in a bin gives binary category 1,
including short and point events. Categorical bins use greatest total overlap
duration; ties favor the larger ID. No off-grid timestamps are inserted. Metadata records this mapping. A future producer can write
its native time coordinates and radial labels directly with `write_label_grid`.
Zeros mean no represented event inside recorded source coverage; outside coverage
and for missing products, cells remain unknown. Coverage bounds do not establish
that every intervening sample was observed.

```python
from labeler.events.interval_tables import read_label_grid

grid = read_label_grid("extend_qmin_rule/recommender_v1/191515.npz")
time_ms = grid["time_ms"]
rho_edges = grid["rho_edges"]
labels = grid["label"]  # (n_times, 20); NaN for unknown cells
```

Minimum-safety-factor extensions currently use the existing `qmin_rule` classes:
0 = absent (reserved), 1 = low, 2 = hybrid, 3 = elevated, 4 = high. The rule does not yet emit
low-q intervals; unclassified times remain unknown rather than being assigned 0. The rule's original
flat-top, finite-data, and minimum-duration requirements remain in force.

## Generate tables

From the repository root, format all implemented datasets:

```bash
pixi run -e labelmaker python scripts/labeler/labels_format.py
```

Or run a category formatter directly:

```bash
pixi run -e labelmaker python data/events/edge_localized_mode/formatter.py
pixi run -e labelmaker python data/events/alfven_eigenmode/formatter.py
pixi run -e labelmaker python data/events/resistive_wall_mode/formatter.py
pixi run -e labelmaker python data/events/neoclassical_tearing_mode/formatter.py
```

AE's original pickle has no timestamps. Its formatter explicitly maps labels
onto the upstream reader's 0–2 s window; `--start-s` and `--stop-s` override that
mapping. AE uses any-positive aggregation in 50 ms bins. This limitation is recorded in its metadata and example notebook.

Export an existing producer's results without rerunning the producer:

```bash
pixi run -e labelmaker python scripts/labeler/labels_extend.py \
  --category minimum_safety_factor --producer qmin_rule \
  --shot-list configs/ideate/shot_lists/recommender_v1.yaml \
  --out data/events/minimum_safety_factor/extend_qmin_rule/recommender_v1.csv
```

Each producing source has its own `extend_<producer>/` folder. A selector spanning
multiple producers must be narrowed before exporting. `--root` and `--events-root`
locate existing labelmaker products; these stores are read-only. Missing-shot and
source-status information stays in metadata. A rerun replaces generated files for
that shot list and removes stale per-shot files.

The existing RWM database has 56 onsets across 33 shots, none in `recommender_v1`.
Its interval export remains empty and its 500 grids contain unknown cells, not
fabricated negative labels. Minimum-safety-factor's current export has 465
recorded intervals and 500 sampled grids.

`discrete_labels.csv` remains the editable scope inventory. Keep raw data,
checkpoints, and large generated artifacts out of commits. No formatter alters,
extracts over, or normalizes the source files in `raw/`.

ELM formatting counts the original 1 ms onset labels within each 50 ms bin,
uses category 1 for nonzero counts, and saves per-shot grids and `event_count`
arrays under `format/shots/`. Matching WPQH subsets are deduplicated.
AE and ELM CSVs compress consecutive equal 50 ms bins into half-open intervals.
RWM and TM interval CSVs preserve original annotated bounds; their sampled
grid examples and extensions aggregate onto the same 50 ms grid.

H-mode and L-mode formatters read the supplied Jalal regime table. H-mode combines
H/QH/WP, and L-mode uses L. Both produce binary 50 ms intervals and per-shot
rho grids, using other explicitly labelled regimes as 0 and gaps as unknown.
The JSON source audit records skipped unlabelled rows and duplicate removal.

## Open saved shot labels

All populated formatted datasets write `format/shots/<shot>.npz`, using the same
sparse NumPy format as extensions. Existing per-shot directories under `format/`
have been moved into `shots/`. The formatters regenerate this layout directly.

Each `example.ipynb` compares original annotations with saved `.npz` labels. Set `source` to `format/shots`
or an available `extend_<producer>/<shot-list>` directory and select `shot`.
The notebook loads the selected shot, displays its categories, and plots the
grid through shared helpers. Category names are embedded as `category_ids` and `category_names` arrays;
`read_label_grid` returns them as `categories`. No raw pickle, archive, CSV, or
sidecar is required for the formatted plot; the original-label plot reads its source. ELM files also retain their onset counts.

```python
from labeler.events.interval_tables import read_label_grid

grid = read_label_grid("data/events/alfven_eigenmode/format/shots/170815.npz")
time = grid["time_ms"]
labels = grid["label"]  # (time, 20 rho bins), NaN for unknown cells
categories = grid["categories"]
```

## Notebook kernel and imports

Select **Python (FAITH labelmaker)** in Jupyter or VS Code. The project is already
installed editable in the labelmaker Pixi environment (`pyproject.toml`), so
normal imports work from any directory; no `sys.path` insertion is needed.
The kernel has been registered locally. On another machine, register it once
from the repository root:

```bash
pixi run -e labelmaker python -m ipykernel install --user --name faith-labelmaker --display-name "Python (FAITH labelmaker)"
```

The notebooks use shared helpers:

```python
from labeler.events.notebooks import load_shot, plot_shot

grid = load_shot("alfven_eigenmode", 170815)
plot_shot(grid)
```

Use `source="extend_qmin_rule/recommender_v1"` for a minimum-safety-factor
extension, or another existing extension folder for its event. The loader uses
`LABELMAKER_LABEL_TABLES` when set and otherwise finds this checkout's
`data/events` directory through the installed package. It reads only the selected
NPZ file. The plotting helper also shows ELM onset counts when present.

Reading a category's original annotation format is case-by-case work: each
`example.ipynb` carries its own cell for it rather than a shared function. AE
shows its five original classes against sample index (the pickle has no
timestamps and loading it temporarily needs several GB). ELM shows native
1 ms annotations; tearing shows HDF5 and archive traces separately; RWM shows
original onsets and mode numbers; confinement shows the four original regime
flags and bounds. Q-min shows the source `qmin_rule` intervals before
aggregation, not raw EFIT measurements. The original files and source event
store remain unchanged.
