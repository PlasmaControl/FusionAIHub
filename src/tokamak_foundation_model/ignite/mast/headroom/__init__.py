"""Headroom predictors: how far simple references are from the best published scores.

Each predictor is a `harness.PredictFn` scored by `scripts/mast/run_headroom.py`.
"""

# Predictors that are fitted on data: they are scored on validation shots only, because the spec
# reads the test shots once, for the final table. The floor reads the targets and fits nothing.
FITTED = ("shrunk", "ridge", "direct")
