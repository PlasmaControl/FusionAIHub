#!/usr/bin/env python3
"""Format the original neoclassical_tearing_mode labels; run with the labelmaker environment."""

import sys
from pathlib import Path

# Support direct execution from any working directory in this checkout.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from labeler.events.source_formatters import main

if __name__ == "__main__":
    raise SystemExit(
        main("neoclassical_tearing_mode", Path(__file__).resolve().parents[1])
    )
