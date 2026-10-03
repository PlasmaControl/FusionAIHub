"""Serialize undefined scientific metrics as JSON null, never NaN/Infinity."""

from __future__ import annotations

import json
import math


def _finite(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def dumps(value, **kwargs):
    return json.dumps(_finite(value), allow_nan=False, **kwargs)
