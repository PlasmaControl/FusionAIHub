"""The join layer: labelmaker's per-shot labels and events become ideate's DB tables.

`join.py` turns `$LABELMAKER_ROOT/labels/<shot>_labels.h5` into `labels_wide.parquet` and
`$LABELMAKER_ROOT/events/<shot>_events.parquet` (plus the risk forecasts derived here) into
`events.parquet`; `claims.py` turns the operator text into `text_claims.parquet`.

ideate reads labelmaker and never writes into it.
"""
