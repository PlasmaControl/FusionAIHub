# Event shot rosters and human verification

Date: 2026-09-17
Status: design, approved for planning

Every phenomenon under `data/events/` gets a shot-level review roster
(`shots.csv`) and a notebook (`verification.ipynb`) that shows a reviewer the
signals that decide whether the event is real, lets them correct the label
intervals by dragging on the plot, and records who reviewed what.

## Problem

`data/events/<category>/` holds interval CSVs and per-shot label grids, and
`example.ipynb` plots them. Nothing records whether a human has ever *looked*
at a shot's labels, and nothing lets a human change them. Label quality is
therefore unmeasured: the ELM table has 576 shots and the NTM table 4,820, all
of them at the same unknown standard as each other.

## Scope

Sixteen category directories under `data/events/`, each gaining:

- `shots.csv` — the review roster, three placeholder rows.
- `verification.ipynb` — the review surface.

Four categories get real, hand-written panel definitions:
`minimum_safety_factor`, `sawtooth_oscillation`, `alfven_eigenmode` and
`fishbone`. The other twelve get a generic fallback panel set until somebody
writes theirs.

`fishbone` is a new, empty category directory: it gets the full skeleton
(`README.md`, `raw/`, `format/`) as well as the roster and the notebook.

Out of scope: choosing the actual gold shots (none are chosen yet), rerunning
any formatter, and the repository-wide `labelmaker`/`ideate` rename sweep.

## Roster schema: `data/events/<category>/shots.csv`

```csv
shot,tier,holdout,reviewers,verified_on,notes
000001,gold,false,alice;bob,2026-01-01,EXAMPLE - replace
000002,silver,false,alice,2026-01-01,EXAMPLE - replace
000003,unverified,false,,,EXAMPLE - replace
```

| Column | Meaning |
| --- | --- |
| `shot` | DIII-D shot number, integer, unique within the file |
| `tier` | `gold`, `silver`, or `unverified` |
| `holdout` | `true` or `false`, required on every row |
| `reviewers` | `;`-separated reviewer ids, in the order they reviewed |
| `verified_on` | ISO date of the most recent review; blank when unverified |
| `notes` | free text, one line |

`tier` is a curation judgement, set by hand, not something derived from the
review count:

- `gold` — a shot the curator trusts as a clean, representative example.
- `silver` — usable but less certain, or not yet fully checked.
- `unverified` — no curation judgement has been made.

`validate_roster` checks only that `tier` is one of the three legal values; it
does not compare it against `reviewers`, the same way `validate_intervals`
guards the interval schema without judging label quality. A reviewer id is
`$USER`, and a reviewer appears at most once per shot; pressing Verify
records that reviewer and today's date and leaves `tier` untouched.

`holdout` marks a shot as reserved from training and tuning, used only for
final evaluation. It is required and has no default: a blank value is
invalid, and `validate_roster` rejects it the same way it rejects an unknown
`tier`.

Ten gold shots per category is the target, not a rule the file enforces. The
roster is not the category's shot list: a shot enters it when someone puts it
up for review, and the interval tables under `format/` and `extend_*/` remain
the record of what is labelled.

The three placeholder rows ship in every category and are meant to be deleted.

## Corrections: `data/events/<category>/review/<shot>.csv`

A reviewer's edits are written per shot, in the public interval schema already
documented in `data/events/README.md`:

```csv
shot,category,t_start,t_end,confidence
170815,1,450.0,700.0,
```

The review directory is a separate claim from `format/` and `extend_*/`: it is
what a human asserts after looking, and no formatter reads or overwrites it.
Merging reviews back into a formatted table is a later decision, deliberately
not made here.

## Verification notebook

One `verification.ipynb` per category. Shared machinery lives in
`src/labeler/events/verify.py`; the panel definitions live **inline in each
notebook**, because deciding which traces settle a phenomenon is case-by-case
work that does not generalise.

### Shared: `src/labeler/events/verify.py`

```python
corpus_signal(shot, group, channels=None, t_range=None) -> FeatureArray
```

Reads one group from `<corpus>/<shot>_processed.h5` with h5py slicing. ECE is
`(48, 3.1e6)` float32 and CO2 `(4, 4.5e6)`; neither is ever read whole. Raises
`NoDataError` naming the shot and the corpus span when the file is absent or
the group carries the `(C, 1)` absent-signal sentinel.

```python
review(event, shot, panels, *, source="format/shots") -> ReviewSession
```

Builds a stacked `plotly.graph_objects.FigureWidget`: one row per panel, a
shared time axis in milliseconds, and a final row showing the current labels
read from the category's `.npz` grid. Box-select on any row captures a time
range; `Mark present` / `Mark absent` turn that range into a correction row;
`Verify` records `$USER` and today's date on the roster row, leaving
`tier` alone; `Save` writes
both files. Nothing is written until a button is pressed.

`FigureWidget` requires `anywidget`, which is not currently installed — one
entry in `[tool.pixi.feature.labelmaker.pypi-dependencies]` and a lock update.

### Panels: `minimum_safety_factor`

`qmin` from the feature store (all 337 rostered shots have it), `qpsi` as a
profile heatmap, and `ip` for context. The existing `qmin_rule` classes
(0 absent, 1 low, 2 hybrid, 3 elevated, 4 high) are the label row.

### Panels: `sawtooth_oscillation`

Raw ECE from the **corpus** `ece` group — `(48, ~3.1e6)` at roughly 500 kHz —
in four rows of four channels each, covering channels 20–36. The feature
store's `ece` is decimated to 1 ms and is useless here: a sawtooth crash is
100 µs to 1 ms, so the decimated record cannot show one. Channels are
overplotted within a row so the inversion radius shows as the phase flip
between rows.

### Panels: `alfven_eigenmode`

CO2 crosspower spectrograms for the chord pairs R0×V1, R0×V2, R0×V3, computed
with `scipy.signal.csd` over the 500 kHz record, plotted as log magnitude
against time and frequency with the 80–250 kHz AE band marked. The corpus
`co2` group carries the four chords in order `r0, v1, v2, v3`.

The CO2 the corpus holds does not cover this category. The 180 annotated AE
shots span 170659-178879 and the corpus covers 185601-204999, so none of them
has a corpus file; and where the corpus does carry `co2`, it is filled on only
about half of its shots, all of them above 198279.

So the panel fetches. When the corpus has no CO2 for a shot, it pulls
`DENR0UF`, `DENV1UF`, `DENV2UF` and `DENV3UF` through
`toksearch_d3d.PtDataSignal` and caches the arrays under the review directory,
keyed by shot. The `co2` feature declares `sources=("corpus",)` and has no fdp
resolver, so this fetch lives in the notebook rather than in the feature
namespace.

The fetch needs the kernel started under the `fdp run` wrapper. The only
reviewer today has it, so the notebook does not work around its absence: it
checks, and if the wrapper is missing it says so and names the command to
restart under.

### Panels: `fishbone`

The magnetic spectrogram. `scipy.signal.spectrogram` of the corpus `mhr` group
(8 probes, `B1`-`B8`, 500 kHz), plotted as log power against time and frequency
over 0-40 kHz with the 2-30 kHz fishbone band marked, plus the cross-phase
between a probe pair on the row beneath it.

What the reviewer is looking for is a burst that chirps *downward* - the test
fixture's synthetic fishbone sweeps 20 -> 12 kHz at -0.8 kHz/ms, which is the
shape - repeating on the beam-heated part of the discharge.

The toroidal mode number is judged, not computed. The corpus does not record
the probes' toroidal angles, so the notebook shows the pair cross-phase and
leaves n = 1 to the reviewer's eye; it does not claim to measure n. Saying
otherwise would be the panel asserting something its inputs cannot support.

### Panels: the other twelve categories

`ip`, `betan`, and `pinj_total` from the feature store, plus the label row. Enough
to confirm a shot exists and its labels sit inside the discharge; not enough to
verify a phenomenon. Each notebook says so in its first markdown cell.

## Moving `plot_original` out of `src/`

`src/labeler/events/notebooks.py` is 278 lines, of which `plot_original` is
189 across six `if event ==` branches. Each branch reads one category's
original annotation format and exists only to serve that category's
`example.ipynb`. The branches move into their own notebooks as plain cells.

`notebooks.py` keeps `load_shot` and `plot_shot`, which are generic over the
saved grid format and are what the new `verify.py` builds on.

## Dead package directories

`src/labelmaker/` and `src/ideate/` are the pre-R1 names. Both now contain
`__pycache__` and nothing else — zero `.py` files — and neither is tracked by
git. Both are deleted.

Stale references inside `data/events/**` are corrected: the module path
`labelmaker.events.*` becomes `labeler.events.*`, and `src/labelmaker/`
becomes `src/labeler/`.

Three things keep their old names on purpose, matching the R1 decision:

- the pixi environments `labelmaker` and `ideate` (pixi rejects underscores,
  so `shot_design` is not available as an environment name);
- the data roots `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker` and
  `.../nc1514/ideate`;
- `.meta.json` sidecars, which are provenance records of what produced a file.

The wider sweep — about 190 files, mostly historical run records under
`outputs/` — is a separate job and is not part of this work.

## New category: `fishbone`

`data/events/fishbone/` exists and is empty. It gets the same skeleton every
other category has - `README.md`, `raw/`, `format/` - plus the roster and the
notebook.

Its `README.md` follows the section order the other categories use
(Description, Method, Provenance, Models, Alias, Reference, Contact, Tables,
Category), with:

- **Description** - the bursting m/n = 1/1 internal kink driven by fast ions
  resonating with the trapped-ion toroidal precession, named for the burst
  envelope on Mirnov signals, first seen on PDX under near-perpendicular NBI
  (McGuire et al. 1983). On DIII-D a burst sits in roughly 2-30 kHz and chirps
  downward as the resonant fast-ion energy falls. Fishbones expel fast ions and
  can seed sawteeth and NTMs.
- **Method** - the magnetic spectrogram: look for the n = 1 chirp. No detector
  exists, so this is the stated method, not a description of one that runs.
- **Provenance** - none. No curated table, so `raw/` is empty.
- **Models** - none, at stable, latest and all.
- **Alias** - `fishbone`, `fishbones`, matching the `fishbone` entry already in
  `src/labeler/events/lexicons.yaml`.
- **Tables** - inventory row `Fishbone`, lexicon id `fishbone`, no registered
  raw table. The row is already in `discrete_labels.csv`.
- **Category** - the default 0 = absent / 1 = present mapping.

Nothing is registered in `events.yaml`: that file lists datasets that a
formatter produces, and fishbone has neither. The lexicon entry already exists
and is not changed.

## Documentation

`data/events/README.md` gains a section covering the roster schema, the tier
rules, the review directory, and how to run a verification notebook. Each
category README gains a link to its `verification.ipynb`, beside the
`example.ipynb` link where the category has one - eight of the sixteen do.

## Testing

`tests/labeler/test_events_verify.py` covers, without a display or a kernel:

- `validate_roster` accepts the placeholder file and rejects each way tier and
  reviewer count can disagree, a duplicate shot, and a duplicate reviewer;
- the roster writer appends each new reviewer and leaves `tier` alone, and a
  repeated Verify by the same reviewer updates `verified_on` without
  appearing twice;
- `holdout` is required and rejects anything but `true` or `false`;
- corrections round-trip through `validate_intervals`;
- `corpus_signal` slices rather than loading, and raises `NoDataError` for an
  absent file and for the `(C, 1)` sentinel;
- the AE fallback raises its wrapper message when toksearch is unavailable.

Every `verification.ipynb` is checked to parse as JSON and to import cleanly;
none is executed in the suite, because execution needs the corpus.

## What this does not decide

Which shots become gold. How a review merges back into `format/`. Whether a
third reviewer's disagreement demotes a shot. Panel definitions for the
twelve categories on the generic fallback.
