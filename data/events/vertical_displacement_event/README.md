# Vertical Displacement Event

## Status

Not started. No raw dataset, reference labels, formatter, detector, or trained
model is available here yet. The pending entry in `../events.yaml` is not loaded
as a formatted dataset. The existing scope-inventory row is `VDE` in
[`discrete_labels.csv`](../discrete_labels.csv).

## Description

The intended label identifies an observed vertical displacement event (VDE).
An estimate of vertical instability or its growth rate is supporting information;
it does not by itself establish that a VDE occurred.

## Scientist guidance

Source: Jayson's response to Nathan, supplied in the project conversation.
The message did not include a date, example shots, signal units, or thresholds.

| PCS pointname | What Jayson described | Potential use |
| --- | --- | --- |
| `rtslambda` | Real-time estimate of the linear open-loop VDE growth rate, when available for a shot. Calculated similarly to TokSys. | Assess the likelihood of vertical instability; not an automatic VDE label. |
| `vpsdfz` | A fast vertical-position metric in the PCS. | Examine its evolution relative to the timing of other events to identify actual VDEs. |

Jayson attributed the original growth-rate algorithm to Erike Oloffson, with
later updates by Stefano Marchioni and Olivier Sauter. These names and the
algorithm description are recorded as supplied; implementation details have
not yet been checked.

Jayson offered to meet and explain how he identifies VDEs shot by shot in the
control room.

## Follow-up tasks

- [ ] Arrange a walkthrough with Jayson. Ask for confirmed VDE and non-VDE
  example shots, the event timings he uses, and examples that are hard to classify.
- [ ] Confirm how to retrieve `rtslambda` and `vpsdfz`: archive/access route,
  units, sign convention, native sample rate, time reference, validity indicators,
  and which shots contain them. Missing signals must remain unknown.
- [ ] Plot both signals at their native time resolution for the example shots.
  Ask Jayson which other event timings and control signals should be aligned
  with the vertical-position trace, including how to distinguish a VDE from
  commanded motion or motion associated with another event.
- [ ] Agree on an annotation definition: what qualifies as a VDE, its onset and
  end, and when a reviewed interval can be labelled as VDE-free. Do not assign
  thresholds to `rtslambda` or `vpsdfz` before checking their meanings and examples.
- [ ] Build a small reviewed reference set, keeping confirmed labels separate
  from uncertain cases and from growth-rate estimates. Record annotation
  provenance and any confidence information actually supplied.
- [ ] Implement and validate a candidate detector against the reviewed labels.
  Inspect false positives and missed events with Jayson before treating its
  output as an automatic label.
- [ ] Add the formatter and an executed `example.ipynb` showing original signals,
  reviewed labels, and the output grid. Register the event ID with the loaders
  when a usable dataset or producer exists.

## Planned format

Follow the shared [event storage conventions](../README.md):

- CSV columns: `shot,category,t_start,t_end,confidence`; time units are
  milliseconds, as declared in `../events.yaml`.
- Sampled output: 50 ms time bins and 20 rho bins over 0–1. Inspect the native
  fast signals before aggregation so a short event is not lost by subsampling.
  A confirmed event overlapping a bin can mark that bin as present.
- Without radial localization, broadcast each scalar label across all rho bins.
- Store unknown cells separately from zero. Blank confidence means unknown.
- Put source provenance, signal interpretation, aggregation rules, and category
  names in the JSON sidecar when outputs are generated.

## Category

Planned binary IDs for the CSV, grids, and future JSON `categories` mapping:

| ID | Label |
| --- | --- |
| 0 | VDE absent within a reviewed or validated observation interval |
| 1 | VDE present |

No data or no label does not mean category 0. No output files have been generated.

## Alias

- vertical displacement event
- vde

## Contact

- **Jayson**: offered a control-room identification walkthrough in the supplied response.
- **Nathaniel Chen**: follow up on the signals, example shots, and annotation criteria.

## Verification

[`verification.ipynb`](verification.ipynb) plots one shot's signals against its
saved labels and takes back corrections. The review roster is
[`shots.csv`](shots.csv). See the [table guide](../README.md) for the roster
schema.
