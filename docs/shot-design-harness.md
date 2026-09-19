# Gemma shot-design harness

The **Design a shot** form runs a small, fixed workflow. Gemma interprets the
request, the existing retrieval system finds real shots, and Gemma chooses a
reference shots using their measured actuator summaries.
It can retain the reference, average selected references, and multiply available
waveforms by factors between 0.5 and 1.5. The existing design service validates
and saves the result. Each model step permits one JSON-format repair if its
first response is prose. Generated edits keep waveform corners within 5% of
the scaled waveform's peak magnitude, with zero-demand spans preserved; the
HDF5 and saved editor curves describe the same proposal. Original reference
measurements remain unchanged. This does not run a plasma-response simulation or establish
that the proposed controls achieve the requested physics objective.

The configured model is Gemma through the existing Ollama client in
`configs/shot_design/llm.yaml`. If unavailable, start the existing model service
with `sbatch scripts/shot_design/serve_llm.sbatch`. The harness reports the actual
model error; it does not fabricate an assistant result. Deterministic model
replies can use the existing LLM cache; artifact metadata records cache hits.

## API and progress

All routes use the application's existing authentication cookie.

- `POST /api/design-assistant` accepts `{"prompt":"...","model":"quality"}`.
  `model` can also be `fast`; prompts allow 1–4000 characters. Returns HTTP 202
  and a job snapshot.
- `GET /api/design-assistant/{id}` returns the current snapshot. Job statuses
  are `queued`, `running`, `complete`, and `failed`. The ordered stages are
  `interpret`, `retrieve`, `propose`, `validate`, and `save`, with statuses
  `pending`, `running`, `complete`, or `failed`. Each stage has a readable detail.
- `GET /api/design-assistant/{id}/hdf5` downloads a completed artifact.

Only one job runs per application instance; another submission receives HTTP
409. Job status is kept in memory for the last 50 jobs and disappears after a
server restart. Saved HDF5 artifacts and editable design revisions persist.
A completed result contains `design_id`, `artifact_path`, `hdf5_url`, selected
references, model, rationale, and validation checks. Open the revision through
`GET /api/design/{design_id}` and continue editing in the existing editor.

## HDF5 contract

Artifacts are atomically written to `<configured data_root>/outputs/<design_id>.h5`.
Source corpus files remain read-only. Root attributes include
`schema_version = "shot-design-actuators/1"`, `layout = "time,channel"`,
`time_basis = "absolute shot time"`, and `frame_s = 0.05`.

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `time_s` | `(T,)` | Absolute shot time in seconds at each 50 ms frame start |
| `actuators` | `(T, 88)` | Physical actuator frame means; each column uses its own units |
| `available` | `(T, 88)` | Boolean measurement coverage; unavailable entries are NaN |
| `channel_names` | `(88,)` | Canonical native keys in `ACT_SPEC` order |
| `channel_units` | `(88,)` | Physical unit of each column |
| `metadata` | scalar UTF-8 JSON | Request, model/cache provenance, retrieval evidence, references, explanation, checks |
| `program` | scalar UTF-8 JSON | Saved editable `DesignProgram`, including any averaged proposal snapshot |

The 88 channels follow `ech_power` (12, W), `pinj` (8, W), `beam_voltage`
(8, V), `tinj` (8, N m), `gas_flow` (11, Torr.L/s), `gas_raw` (11, V),
`rmp` (12, A), and `i_coil` (18, A). Missing real channels are explicitly masked;
they are never silently replaced with invented zero measurements.

The HDF5 is a portable physical actuator proposal. Current IGNITE inference
consumes the existing `.pt` artifact containing normalized actuators and
reference diagnostic seed tokens. In the editor, use **Prepare IGNITE input**
when required, then the existing IGNITE export after its checks pass. HDF5
availability does not imply IGNITE readiness: `checks.hdf5_valid` and
`checks.can_export` deliberately describe different contracts.

Validation reports `physical_errors` separately from `model_errors` and keeps
their union in `errors` for the existing editor. Cache availability, cache/raw
normalization disagreement, zero-spread normalization, and float16 normalization
overflow block IGNITE export without invalidating finite physical waveforms.
Gemma also sees incompatible control channels in the retrieved evidence so it
can prefer references and controls that support the intended simulation.

## Saved demo examples

Three real Gemma runs are saved under `<data_root>/outputs/demo_examples/`:

| Example | First reference | Proposed change |
| --- | --- | --- |
| `tearing_mode_control` | 195071 | Increase measured ECH channels 5, 7 and 10 by 30% |
| `alfven_eigenmode_control` | 193353 | Reduce total NBI power by 30% |
| `elm_control` | 190736 | Increase compatible RMP channels 0–10 by 20% |

Each has `.h5`, editable `.json`, and IGNITE `.pt` files. The manifest records
the original requests, all reference shots, model rationale, and checks.
All three HDF5 trajectories were verified against their saved programs; the
`.pt` inputs passed the installed production checkpoint's input validator.
No plasma-response rollout was run. The generated artifacts are outside the
repository and are not source-controlled.
