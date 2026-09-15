"""The join layer: labeler's per-shot labels and events become shot_design's DB tables.

`join.py` turns `$LABELER_ROOT/labels/<shot>_labels.h5` into `labels_wide.parquet` and
`$LABELER_ROOT/events/<shot>_events.parquet` (plus the risk forecasts derived here) into
`events.parquet`; `claims.py` turns the operator text into `text_claims.parquet`.

shot_design reads labeler and never writes into it.
"""
