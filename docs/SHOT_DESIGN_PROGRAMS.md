# Actuator programs for IGNITE

The Shot Designer's **Actuator editor** starts with real reference shots, lets you
edit their actuator waveforms, and saves inputs for IGNITE inference. The first
listed reference supplies the initial state and default actuation.

## Design a program

1. Run `pixi run -e shot-design shot_design serve` and open its token link.
2. Choose **Use as reference** on a search result or shot, or open **Actuator editor**
   and enter up to six reference shots separated by commas. The first supplies
   IGNITE's seed. **Average reference actuation** creates an equal physical-unit
   average at matching shot times across compatible channels, replacing current
   edits. Channels missing from any reference retain the first shot's values.
3. Set the prediction start and end in seconds. Times must lie on the 50 ms
   frame grid. The model needs one second of reference history before the
   prediction start and supports up to four seconds of prediction. Click
   **Check preview** after changing the references or time window
   to refresh the waveforms before editing.
4. Select an actuator. Handles mark waveform corners; flat and linear runs have
   only endpoint handles. Small variations within 5% of the trace's amplitude
   are ignored for handle display. Original samples are preserved until edited.
   Click the editable plot to add a joint, drag it or edit its numeric values,
   and use **Delete joint** (or Delete/Backspace on a focused joint) to remove it.
   Endpoint times stay fixed. User-created joints survive saving and reopening;
   **Simplify joints** explicitly reduces a dense edited curve. All references
   stay visible. **Reset channel** restores the reference or averaged proposal.
5. Read the checks, add notes, and save a revision. The editable JSON is available
   immediately. If the shot has no simulation cache, click **Prepare IGNITE input**;
   progress or errors appear beside the buttons. This uses the installed codecs
   and may take a few minutes. Download the IGNITE input once preparation and
   validation succeed. Edits after saving require another save before export.

Editing and saving need the shot's processed corpus file. A compatible frame-code
cache is required only for IGNITE export. Existing caches are read from
`<data_root>/frame_codes/<shot>.pt` or the shipped model bundle. **Prepare IGNITE
input** creates one under `<ignite_inputs_dir>/reference_cache/<source-id>/` using
the production encoder. The source ID binds that cache to the exact corpus file;
changed source data requires fresh preparation. Original corpus files and existing
caches are not modified. Missing actuator measurements remain unavailable for editing.
Cache token shapes and vocabulary sizes must match the pinned production model;
caches from another codec generation cannot be exported.

## What is saved

Every save creates a new JSON revision under
`<actuations_dir>/designs/`. It records the reference shot, comparison shots,
prediction window, notes, source identity, units, any averaged proposal snapshot,
and edited vertices with native
actuator keys. The source identity prevents a saved design from silently being
applied to changed reference data. Reloading a JSON design requires its source
data; the exported IGNITE input already contains the reference tokens it needs.

The IGNITE export is written under `<ignite_inputs_dir>/designs/` and downloaded
as a `.pt` file. It uses the same dictionary layout as IGNITE's frame-code caches:

- `codes`: reference plasma-state tokens for the selected history and window.
- `actuators`: float16 tensor of shape `(n_frames, 88)`, in the model's channel order.
- `n_frames`: number of history plus prediction frames.
- `vocabs`: the reference codec vocabulary metadata.

Until preparation, traces come directly from the actuator measurements and model
normalization checks are deferred. Preparation determines the model's complete
reference duration and revalidates the draft before allowing export.

The first 20 frames provide the fixed reference history. Edited controls begin
at prediction frame 20. Unedited controls retain the cached values exactly.
Edits use the complete reference shot's normalization statistics, so scaling a
power trace survives normalization. A changed channel with zero reference
standard deviation cannot be represented by this policy and blocks export.

The editor's time axis is absolute shot time. Exported frame zero corresponds
to `prediction_start - 1 second`; keep the JSON alongside the model input to
retain that time origin and the physical-unit edits.

## Run inference

Use the local IGNITE bundle and its matching production checkpoint. With the
bundle directory on Python's module search path:

```python
from ignite_infer import load_dynamics, load_shot, rollout

model, cfg = load_dynamics("/path/to/IGNITE/ignite_dynamics_prod_nfullrs2_step13400.pt")
program = load_shot("/path/to/design-export.pt")
result = rollout(model, cfg, program, seed=0)
```

This uses the saved actuator trajectory directly. The JSON file is for editing
and provenance; it is not the model input. For a paired comparison, export the
same reference/window without edits and run both inputs with the same sampling
settings and random seed.

The `codes` after the history window still describe the real reference shot.
Consequently the inference wrapper's `gt` output is reference-shot data, not
ground truth for the proposed scenario. Compare generated outputs with that
distinction in mind.

## Checks and interpretation

Checks distinguish malformed or unrepresentable model inputs from advisory
limits. A draft can be saved with edit errors; IGNITE export requires those
errors to be resolved. The configured operating limits are provisional and
are displayed as warnings. Passing these checks does not predict discharge
success or certify a PCS command sequence.

Newly requested NBI/ECH power cannot be negative. This bound applies to power
totals and individual power channels; signed torque and coil currents retain
their physical meaning. Unedited reference measurements are preserved.

Waveforms describe values on IGNITE's 50 ms frame grid. Each point is the mean
control value for that frame; interpolation between handles supplies those
frame values. This is not a sub-frame switching schedule. NBI/ECH total edits
preserve the reference split between members. Turning on a total where no
member was active requires an explicit per-member edit and usable reference
normalization; the designer does not invent a split.

Related channels are independent model inputs. For example, changing NBI power
does not automatically recalculate beam torque, and changing a gas valve
command does not calibrate its flow. Checks identify related controls that
remain at their reference values. Gas valve commands are volts; calibrated
gas flow is a separate channel. PCS integration needs its own actuator and
timing adapter.
