"""Operating-limit flags: the "what's possible" half of the recommender.

`load_rules()` reads configs/shot_design/flags.yaml; `evaluate_flags(values, category, cfg)` runs it
against either a shot's measured scalars or a user's proposed actuator settings. See `flags.rules`.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from .rules import DERIVED, OPS, envelopes_from_frame, evaluate_flags, load_rules

__all__ = ["DERIVED", "OPS", "envelopes_from_frame", "evaluate_flags", "load_rules"]
