"""Shared time conversion for review and method label writers."""

import math


def whole_ms(t) -> int:
    """Round finite milliseconds half up, matching the page's Math.floor(t + 0.5)."""
    value = float(t)
    if not math.isfinite(value):
        raise ValueError("time must be finite")
    return math.floor(value + 0.5)
