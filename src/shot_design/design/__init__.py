"""Counterfactual design: the IGNITE actuator contract and the frame-code seed cache.

`actuators` is the 88-channel control vector the production dynamics checkpoint consumes, and
`seed` turns one corpus shot into the `frame_codes/<shot>.pt` that checkpoint seeds a rollout
from. Both are pinned against the ten caches shipped in the model bundle -- see
`scripts/shot_design/g_enc.py`, the gate that re-runs that comparison.
"""

from __future__ import annotations

__all__ = ["actuators", "seed"]
