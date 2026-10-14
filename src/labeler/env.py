"""Resolve environment settings for Python readers and shell launchers."""

from __future__ import annotations

import os
import sys


def getenv(name: str, default: str | None = None) -> str | None:
    """Return the environment value of `name`, or `default` when it is unset.

    An empty value counts as set. Shell launchers call this module as a script.
    """
    return os.environ.get(name, default)


if __name__ == "__main__":
    print(getenv(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""))
