"""Scoring one assessment of a shot against another.

One library serves the catalog's reader agreement, its method report cards and
the Phase C model tests, so all three count frames, match events and weight
shots the same way:

- `frames`: assessments, 10 ms frame states, frame counts, shot-level presence;
- `events`: onsets and ends, nearest-first matching within a tolerance;
- `stats`: precision, recall, F1, Cohen's and Fleiss' kappa, shot weights and the
  stratified shot bootstrap.
"""
