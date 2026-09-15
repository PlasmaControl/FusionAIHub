"""Resolve renamed environment settings for Python readers and shell launchers."""

from __future__ import annotations

import logging
import os
import sys


def getenv(name: str, default: str | None = None) -> str | None:
    """Prefer LABELER_* even when empty; warn only when LABELMAKER_* supplies it.

    Accept either spelling so an older interpolated configuration also observes
    the new name's precedence. Unrelated environment names pass through unchanged.
    """
    if name.startswith("LABELMAKER_"):
        name = "LABELER_" + name[len("LABELMAKER_"):]
    if name in os.environ:
        return os.environ[name]
    if name.startswith("LABELER_"):
        old = "LABELMAKER_" + name[len("LABELER_"):]
        if old in os.environ:
            logging.getLogger(__name__).warning("%s is deprecated; use %s", old, name)
            return os.environ[old]
    return default


if __name__ == "__main__":
    print(getenv(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""))
