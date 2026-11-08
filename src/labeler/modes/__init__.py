"""MHD mode labels from omnimode's port-v6 track tables, in the labeler's two label formats.

omnimode (the owner's labeller, port of `labeler.mhd` v6) writes one set of files a shot:
`<shot>.json` (the run's info record), `<shot>.tracks.parquet` (one row a mode track: n, a time
span, the MAP m/n and its posterior, the harmonic flag `harm` with `fund_id`, status and why)
and `<shot>.frames.parquet` (the track's ridge, one row a 4 ms frame). This package turns them
into the formats of the AE, ELM and confinement labels:

- `tracks`: read a shot as a `labeler.mhd.pipeline.Shot` and apply the owner's display rules
  (harmonics are not labelled; |n| = 1-3 only).
- `cells`: the events' whole-ms states, `mhd_n<K>` and its `m<m>` subcategories, from the same
  rules `labeler.mhd.rows` applies to the v5 tracks (same span padding, merging and states).
- `suggest`: the `mhd_modes` suggestion tables the review page opens on (`review/source.json`).
- `fdp`: the 1 ms databases of `docs/labeler/fdp_labels.md` (spans, shots, valid, grid_1ms).
- `roster`: the `shots.csv` roster of an event, replaced by a proposed shot set.

`scripts/labeler/mode_fdp_label.py` is the command line. Nothing here reads omnimode's code or
its sealed data: a shot is read only when the caller lists it.
"""
