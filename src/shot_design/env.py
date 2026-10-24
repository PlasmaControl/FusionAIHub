"""Resolve environment settings for Python readers and shell launchers."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


def getenv(name: str, default: str | None = None) -> str | None:
    """Return the environment value of `name`, or `default` when it is unset.

    An empty value counts as set. Shell launchers call this module as a script.
    """
    return os.environ.get(name, default)


if __name__ == "__main__":
    # The cluster table lives in labeler/env.py, so a launcher that asks here for
    # SHOT_DESIGN_DATA_ROOT gets the same answer as one that asks there.
    table = Path(__file__).resolve().parents[1] / "labeler" / "env.py"
    runpy.run_path(str(table), run_name="__main__")
