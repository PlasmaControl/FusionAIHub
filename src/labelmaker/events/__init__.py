"""Discrete phenomenon events: what happened, when, and on whose word.

Labels are a probability at every time step; an event is a single thing with
a start, an end, an optional frequency band and a named source. Detectors,
heuristics, forecasts, text and human annotation all land in one table
(`events/<shot>_events.parquet`) so a consumer can ask "who says so" of
every row. See docs/superpowers/specs/2026-09-07-recommender-labelmaker-v2.md.

`lexicons.yaml` sits here too, and is not only labelmaker's: it is the
single source of the round-1 phenomenon ids and their aliases, and ideate
reads this very file rather than keeping a second list (plan 5.6).
`lexicon.py` is its reader and its matcher - what a phrase means, over
text and nothing else - and `text_weak.py` is where the text comes from:
a shot's own logbook entries out of `sql/logs.jsonl` (shot scope) and its
run's session context out of the per-shot bundle (run scope).

`windows.py` is the other end of the table: masks and events reduced onto a
0.34 s window every 0.17 s, as the 46 diagnostics-only features a prior
scores and a classifier is trained on (`features/resolve_events.py` serves
them as the canonical `phenomenon_window_features`).
"""
