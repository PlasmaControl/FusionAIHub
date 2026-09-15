"""Resolve renamed environment settings for Python readers and shell launchers."""

from __future__ import annotations

import logging
import os
import sys


def getenv(name: str, default: str | None = None) -> str | None:
    """Prefer SHOT_DESIGN_* even when empty; warn only when IDEATE_* supplies it.

    Accept either spelling so an older interpolated configuration also observes
    the new name's precedence. Unrelated environment names pass through unchanged.
    """
    if name.startswith("IDEATE_"):
        name = "SHOT_DESIGN_" + name[len("IDEATE_"):]
    if name in os.environ:
        return os.environ[name]
    if name.startswith("SHOT_DESIGN_"):
        old = "IDEATE_" + name[len("SHOT_DESIGN_"):]
        if old in os.environ:
            logging.getLogger(__name__).warning("%s is deprecated; use %s", old, name)
            return os.environ[old]
    return default


if __name__ == "__main__":
    print(getenv(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""))
