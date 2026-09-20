"""Simulation stage: paired real/proposed IGNITE rollout over a design seed."""

from __future__ import annotations

# The brief's own default list. Lives here (not in `simulate.cli`, which pulls
# torch + the IGNITE dynamics machinery in at module import time) so that
# `shot_design/cli.py`'s argparse setup -- which runs on every invocation, not
# just `simulate` -- can build the `--decode` help text without paying for
# that import.
DEFAULT_DECODE = ("filterscopes", "mhr", "mirnov", "ts_core_density", "ts_core_temp")
