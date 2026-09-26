"""Scoring one assessment of a shot against another.

One library serves the catalog's reader agreement, its method report cards and
the Phase C model tests, so all three count frames, match events and weight
shots the same way:

- `frames`: assessments, 10 ms frame states, frame counts, shot-level presence;
- `events`: onsets and ends, nearest-first matching within a tolerance;
- `stats`: precision, recall, F1, Cohen's and Fleiss' kappa, shot weights and the
  stratified shot bootstrap.

D21 scores events strictly inside the common window, matches nearest first,
then excludes unmatched boundaries near the other reader's uncertain or
not-observable time. Matched pairs always count. Method abstentions read as
absent and never exclude; reader exclusions apply in both directions.
"""
