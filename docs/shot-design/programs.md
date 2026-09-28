---
title: "Actuator Programs"
---

# Actuator programs for IGNITE

The Shot Designer's **Actuator editor** starts with real reference shots, lets you
edit their actuator waveforms, and saves inputs for IGNITE inference. The first
listed reference supplies the initial state and default actuation.

## Start with the assistant

In **Design a shot**, describe the experiment and submit it. Reference shot numbers
in the result link to **Shot**. A saved design ready for simulation offers
**Simulate**, which opens the saved revision and submits it through the editor.
A design that still needs preparation offers **Open in actuator editor**.

The assistant job ID stays in the URL. Reload reattaches to that job without
starting another one. Jobs currently live in the server's memory; if a restart or
history eviction removed the job, the page says it is no longer on the server.
Saved design revisions remain available from the editor.

## Design a program

1. Run `pixi run -e shot-design shot_design serve` and open its token link.
2. Select up to six search results and choose **Design from selected**, or choose
   **Use as reference** on a shot. You can also enter comma-separated shot numbers
   in **Actuator editor**. The first supplies IGNITE's seed; the rest are comparisons.
   Search selection carries your query into Notes. **Average reference actuation** creates an equal physical-unit
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

## Simulate a saved revision

**Simulate** becomes available once a revision is saved and can export. The editor
fetches its simulation status when reopened, including jobs still in the queue.
Queued and running jobs show elapsed time. Editing keeps the last result visible,
labelled with its saved revision; the edits are not included in that result.
Save the changes before submitting another simulation.

Once a result exists, **Run again** asks for confirmation. Results appear under the
waveform: one table and plot per simulated diagnostic. The error columns compare
the mean prediction and hold-last-frame baseline with measurements, scaled by
measured variation. Skill compares the ensemble with that baseline; negative skill
means worse than holding the last frame. An edit effect below twice the run-to-run
noise is marked **No — unresolved**. Absent diagnostics appear once as
**not simulated (diagnostic absent)**, without scores or plots.

The editor reads `/api/design/{id}/simulate/metrics` and the corresponding
`panels/{modality}.png` routes. A run without `metrics.json` shows a plain message;
its markdown report remains available through **View simulation report**.

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

model, cfg = load_dynamics("/path/to/IGNITE_v4/ignite_dynamics_prod_v4_mskfull_step3200.pt")
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
When the assistant scales a measured power waveform, it clamps the scaled
handles at zero: a beam or gyrotron trace sits a few hundred watts below zero
between pulses, and without the clamp every scale factor on such a channel
failed this bound (69 of the 100 rejected designs in the 2026-09-21 Stellar
batch). Only the sub-zero noise is touched; the pulse shape and the member
split are unchanged. A total whose split would drive an individual member
negative is still rejected.

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
