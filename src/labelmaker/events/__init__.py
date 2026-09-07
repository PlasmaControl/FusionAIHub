"""Discrete phenomenon events: what happened, when, and on whose word.

Labels are a probability at every time step; an event is a single thing with
a start, an end, an optional frequency band and a named source. Detectors,
heuristics, forecasts, text and human annotation all land in one table
(`events/<shot>_events.parquet`) so a consumer can ask "who says so" of
every row. See docs/superpowers/specs/2026-09-07-recommender-labelmaker-v2.md.
"""
