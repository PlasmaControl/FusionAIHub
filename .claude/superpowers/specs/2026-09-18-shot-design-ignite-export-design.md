# Editable shot programs and IGNITE export

The user approved the reference-first flow on 2026-09-18: select one primary
reference and optional comparison shots, edit actuator waveforms with handles,
see basic checks, save an editable design, and export an IGNITE input.

## Scope and interface

Add a Design view to the existing plain JavaScript browser application. Reuse
the existing corpus reader, 88-channel actuator layout and frame-code caches.
Do not change retrieval rankings, labels, model weights, or production data.
Do not add dependencies. This is a simulation input editor; PCS is a future
adapter. Provisional limits are advisory, not proof a discharge will succeed.

The primary reference provides all 88 original control channels and the plasma
state tokens. Comparison shots are overlays only. Start with copied waveforms;
optimization and median synthesis are outside this version.

The pinned production contract is 50 ms frames, 20 seed frames, at most 80
prediction frames. The user chooses prediction start/end in shot seconds,
aligned to 50 ms; the start must leave 20 earlier reference frames. The export
contains the chosen history plus prediction window. Statistics are computed
over the complete reference cache, never the edited or sliced window.

## Shared wire contract

`DesignProgram` JSON has `schema_version: "shot-design/1"`, nullable server `id`
and `created`, `reference_shot`, `comparison_shots` (at most five distinct shots),
`start_s` (prediction start, default 1), `end_s` (default 5), `notes`,
`reference_digest` (server-computed source identity), `units` (server-populated
key-to-native-unit mapping), and `edits` mapping
actuator keys to ordered `{t_s, y}` vertices in native physical units.
An empty edits mapping means an exact reference replay. Source identity detects
changes to reference arrays or tokens; a stale design must be reseeded explicitly.
Supplied units that disagree with the canonical units of an edited key are an
error, so values labelled MW cannot silently be interpreted as W.

Editable keys are the native `ACT_SPEC` channel keys (`pinj[0]`, etc.), with
friendly member labels, plus `nbi.total` and `ech.total`. Native channel order
and units are explicit. Unknown or overlapping total/member edits are errors.
Totals preserve the reference member split; nonzero demand where no reference
member was active is an error, not an arbitrary distribution to idle sources.
Related controls that remain at reference values (e.g. torque after a power
edit, calibrated gas flow after a valve edit) produce explicit warnings.
Newly requested NBI/ECH injection power must be nonnegative, both for native
power channels and totals. Historical negative measurement noise in an unchanged
reference is preserved; editing a different point must not turn that unchanged
noise into a new error. Signed torque and coil currents remain valid.

Waveform vertices define frame values at frame start times with linear
interpolation between them; each resulting value is the model's mean for that
50 ms frame. They are not sub-frame PCS switching commands. Endpoints must
cover prediction start through the last prediction frame. The editor starts
with all prediction frame values, avoiding approximation on an untouched trace.
Edits never overwrite the 20 context frames. Reset removes the edit entirely.

HTTP endpoints, under the existing token gate:

- `POST /api/design/preview` accepts a DesignProgram (null digest for a new
  reference). Returns `{program, channels, validation, context_start_s,
  seed_frames:20, frame_s:0.05}`. Each channel has `{key,label,units,editable,
  reason,reference:[{t_s,y}],vertices:[{t_s,y}],comparisons:[{shot,vertices}]}`.
  `reference` includes context; `vertices` covers prediction only. All values
  are finite JSON numbers. Missing channels have `editable:false` and a reason;
  they are never advertised as measured off. Missing comparisons yield warnings.
- `validation` is `{errors:[string],warnings:[string],can_export:bool}`. Errors
  identify invalid edits, time bounds, source changes, missing seed caches,
  missing channels, zero-spread normalization and float16 overflow. Warnings
  include configured provisional amplitude limits, model extrapolation, and
  related controls retained from the reference. Baseline excursions are
  distinguished from excursions introduced by edits.
- `POST /api/design` saves the DesignProgram as a new immutable revision and
  returns the preview payload with assigned id. Drafts with edit validation
  errors can be saved; malformed schema/source references are rejected.
- `GET /api/design` returns `[{id,reference_shot,created,start_s,end_s}]`.
- `GET /api/design/{id}` returns the preview payload for a saved revision.
- `GET /api/design/{id}/json` downloads the editable design.
- `GET /api/design/{id}/ignite` downloads `design-{id}.pt`, or returns 422 with
  actionable validation errors. Export always revalidates server-side.

## Numerical and storage contract

Model inputs are built from `design.actuators.build_actuators` and the selected
reference cache. Preserve unedited cached z-scores exactly. For edited channels
compute `(edited_raw-reference_mean)/(reference_std+EPS)`; zero-spread changed
channels are blocked. Check baseline cache normalization agrees with the raw
reference for edited channels. Missing groups/individual channels retain model
padding for replay but are unavailable for editing. Do not silently ignore keys.

The IGNITE artifact is the existing four-key dictionary: `codes`, `actuators`
(`F,88` float16), `n_frames`, `vocabs`. Slice all codes and controls to the same
window. Seed controls and tokens stay identical to the reference; diagnostic
tokens after the seed describe the real reference, not ground truth for the
proposed scenario. The artifact loads with `ignite_infer.load_shot` and runs
with the existing `rollout` API. Check cache dimensions, finite actuators,
modalities, vocabulary metadata, and bounds before exporting.

Keep new JSON revisions under `paths.actuations_dir/designs/` and exported
artifacts under `paths.ignite_inputs_dir/designs/`. Use strict generated IDs,
atomic writes, safe tensor loading, and no client-selected filesystem paths.
Read source caches from `paths.data_root/frame_codes/<shot>.pt`, with the local
bundle's shipped cache as a fallback. Corpus and source caches are read-only.
Derived reference data may be bounded-cached using source file identity; raw
data must not be reread on every mouse movement. Checks are debounced and stale
responses cannot overwrite more recent edits.

## Browser behavior

Add Design navigation and a Use as reference action from search results/shot
details. Provide primary shot, comparison shots, prediction start/end, channel
selection, reference/comparison overlays, visible seed region, draggable points,
numeric time/value editing, reset, notes, check status, save, saved revision
selection, and JSON/IGNITE downloads. Use SVG and existing styles. Support
keyboard/numeric editing and mobile widths. Mark modified drafts, invalidate
download links after edits, and handle loading/errors without losing edits.

## Verification

Use deterministic local HDF5 and frame-cache fixtures. Prove exact unedited
replay, seed preservation, channel/time order, edits surviving normalization,
total splitting, zero-total and zero-spread failures, source drift rejection,
round-trip persistence, invalid export refusal, token authentication, and
comparison handling. Exercise the real browser save/reopen/download flow and
load the downloaded artifact with the actual inference loader when available.
A full GPU rollout is not required to verify this file interface; distinguish
loader/model-interface verification from numerical prediction-quality evidence.
