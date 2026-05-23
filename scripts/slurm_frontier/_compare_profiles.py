"""Diff two memory.json outputs from profile_stage1.py and print a table.

Usage:
    python _compare_profiles.py <baseline.json> <treatment.json>

Prints rows: step_time_s, throughput_steps_per_s, peak_alloc_GB,
peak_reserved_GB. Each row has baseline value, treatment value, delta
(treatment - baseline), and ratio (treatment / baseline). Pure stdlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fmt(x: float | None) -> str:
    if x is None:
        return "  n/a"
    return f"{x:>7.3f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("treatment", type=Path)
    args = p.parse_args()

    with args.baseline.open() as f:
        base = json.load(f)
    with args.treatment.open() as f:
        treat = json.load(f)

    rows = [
        ("step_time_s",            "active_mean_step_s",       True),
        ("throughput_steps_per_s", "throughput_steps_per_s",   False),
        ("peak_alloc_GB",          "peak_alloc_GB",            True),
        ("peak_reserved_GB",       "peak_reserved_GB",         True),
    ]

    print(f"baseline  ({base.get('attn_impl')}): {args.baseline}")
    print(f"treatment ({treat.get('attn_impl')}): {args.treatment}")
    print()
    print(f"{'metric':<24} {'baseline':>9} {'treatment':>10} {'delta':>9} {'ratio':>8}")
    print("-" * 64)
    for label, key, lower_is_better in rows:
        b = base.get(key)
        t = treat.get(key)
        delta = (t - b) if (b is not None and t is not None) else None
        ratio = (t / b) if (b not in (None, 0) and t is not None) else None
        arrow = ""
        if delta is not None:
            if lower_is_better:
                arrow = "↓" if delta < 0 else "↑"
            else:
                arrow = "↑" if delta > 0 else "↓"
        print(
            f"{label:<24} {fmt(b):>9} {fmt(t):>10} "
            f"{fmt(delta):>9} {fmt(ratio):>8} {arrow}"
        )
    print()
    # Headline line for grep-friendly summary.
    b_step = base.get("active_mean_step_s")
    t_step = treat.get("active_mean_step_s")
    if b_step and t_step:
        speedup = b_step / t_step
        print(f"SUMMARY: {speedup:.2f}x speedup with {treat.get('attn_impl')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
