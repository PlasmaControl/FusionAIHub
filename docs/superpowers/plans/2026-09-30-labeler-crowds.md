# Labeler crowd annotations and diagnostic views

**Goal:** Review resolved individual events and unresolved event groups across
all labeler editors, see EFIT-scaled diamagnetic energy and ELM losses, and compare
sawtooth ECE signals using measured q=1 geometry where available.

**Design:** Keep category intervals compatible with existing consumers. Add
optional, aligned `iscrowd` metadata: 0 individual, 1 group, null unspecified.
Store it in the existing CSV `attrs` field and history. Explicit annotations
retain their boundaries even when neighboring categories match. Existing
labels without this metadata remain unspecified. Preserve unsupported attrs
by refusing edits to those shots. The browser exposes the same resolution
choices for every event and keeps metadata through every editing operation.

**Constraints:** Preserve existing workspace edits; production stores are
read-only; no dependency or training changes. Whole-millisecond label precision
stays explicit. Missing diagnostic geometry cannot establish q=1 membership.
EFIT calibrates the loop; native loop response limits individual ELM sizing.
User correction: fit quiet-window linear drift, scale to EFIT energy, independently
detect loop drops and corroborate them with D-alpha before estimating their size.

- [x] Annotation storage/API: regressions for metadata, touching individuals,
  mixed legacy labels, overlaps, malformed flags, save/reload/history, and
  preservation of other shots' attrs. Add optional `Label.iscrowd`, normalize
  with annotation identity, validate CSV metadata and preserve it in projections.
- [x] Browser: matching normalization, resolution selector and group hatching,
  preserving flags through drafts, drag/category/delete, undo, revert, and
  history restore. Check actual browser save/reload and older-server behavior.
- [x] Diagnostics implementation: retain native EFIT WMHD as the reference for
  calibrated diamagnetic energy, with drift/gain/measurement provenance and missing-data
  handling; sawtooth grouping uses real geometry or an explicitly unmapped fallback.
  Test finite coverage, changing geometry, missing sources and time slicing.
- [x] Research: delegated primary-source note distinguishes COCO annotation,
  training and evaluation semantics; recommends crowd-aware temporal objectives
  and separate individual/group evaluation.
- [x] Focused regressions, independent review, scoped Ruff and JS syntax checks.
- [x] Run the complete labeler suite, review the final changes, and document
  cached-row rebuild and scientific limitations.

The user initially chose offline validation, then refreshed the FDP token.
PTDATA `DIAMAG3` and EFIT WMHD now read successfully for both probed shots,
192238 and 189061; configured FS01 records were read from the corpus.
An exploratory 192238 calibration had a 2.68% median residual within the chosen
regime; 189061 failed the polarity-consistency guard. These trials set no
campaign defaults or production metadata. Response and compensation validation
remain necessary. Explicit quiet-baseline/calibration metadata is required
before displaying measured loop energy. The provided DSL channels
are classified as poloidal-field probes in MHDIN and are excluded from this
path. Physical sawtooth grouping requires supplied calibrated `ece_psi` and
local EFIT `qpsi`; unmapped records retain the clearly labeled fallback.

## Initial validation

The full `labeler-test` run completed with 4,586 passed, 10 skipped and 19
failures. Fourteen failures came from browser test comparisons that omitted
the newly saved `iscrowd` field. The comparisons now check it explicitly;
all 19 tests in the affected inflight/pending browser suites pass, including
new assertions for flags retained during delayed saves and navigation.
The energy, geometry and crowd regressions passed in the full run.
Scoped Ruff, JavaScript syntax checks and `git diff --check` passed.

Five unrelated failures remain in the existing workspace: the unfinished
`data/events/confinement` category lacks a roster and verification notebook
and has an empty README; the paper coverage test expects a linear energy
axis where the figure uses a log axis; and an AE test supplies a parameterless
model to a trainer that obtains its device from its first parameter. These
failures were present in the cached failure list before the full run. Those
files were preserved.

## Overlap correction

The user requested independent individual/crowd bars so resolved events can
be drawn inside crowd envelopes. Both normalization implementations now paint
separate lanes; same-lane replacement, explicit boundaries, unspecified legacy
labels and whole-ms precision remain intact. CSV gaps describe combined
coverage, while annotations retain overlaps in saved tables and history.
The browser shows named Individual/Crowd bars with category/resolution captions,
filters pointer hits by lane, and follows the edited span's resolution through
normalization. API 8 uses an overlap-aware client marker to protect stored/source
overlaps from cached older clients. Catalog checks allow cross-lane overlaps,
recognize binary `iscrowd`, and respect individual-scoped absent rows.

- [x] Reproduce lost envelopes, CSV gaps/round trips, hidden history changes,
  cross-lane pointer hits and old-client flattening with failing regressions.
- [x] Implement independent storage and editing lanes; validate drawing,
  selection, conversion, movement, resizing, deletion, undo, history and drafts
  in actual Chromium.
- [x] Independent read-only review, scoped Ruff, JS syntax and whitespace checks.
- [x] Complete the fresh full labeler-suite run and record its result.

The fresh `pixi run --frozen -e labelmaker labeler-test` run, with CPU thread
counts limited to one, finished with **4,616 passed, 10 skipped and 5 failed**
in 32 minutes 28 seconds. The five failures are exactly the existing workspace
failures listed above. The annotation, catalog, API, energy, geometry and browser
regressions passed, including drawing individuals inside crowds and preserving
both through editing, saving, history restoration and draft navigation.
Independent review found no remaining actionable issues. Fresh scoped Ruff,
JavaScript syntax checks and `git diff --check` passed.

## Continued live validation

- [x] Audit calibration and measurement sensitivity on the fetched native shot
  records, retaining production stores as read-only.
- [x] Reproduce quiet-record contamination of the plasma noise threshold,
  measurements outside calibration, and incomplete boundary support with
  failing regressions; restrict noise, sizing and visible rows to the calibrated
  interval. Record that interval and update the method and ELM cache versions.
- [x] Reproduce competing loop/D-alpha matches; require unique admissible
  counterparts and leave ambiguous individual attribution unsized.
- [x] Independent read-only review of the corrections; no remaining actionable
  findings. Related regression suite: 119 passed. Scoped Ruff passed.
- [x] Probe native ECE frequencies/validity, sightline and EFIT geometry;
  produce an explicitly illustrative cold-resonance plot without installing
  it as calibrated production geometry.

For 192238, the calibration still fits WMHD with 2.68% median residual. The
loss threshold changes from 4.41 kJ over the quiet-dominated full record to
16.56 kJ within the calibration interval; its largest positive measured step
is 11.87 kJ. No events receive sizes under the corrected estimator, including
the earlier 6.80 kJ example. Native signals and historical v1 trial artifacts
remain available for comparison. No production calibration or geometry was
written. The earlier full-suite result above is the baseline; this correction
was verified with the focused energy, cache, panel, API and geometry suites.
