"""The package exists, is importable from every environment that lists it, and
carries the version the pyproject wheel target promises."""
from __future__ import annotations

import ideate


def test_version() -> None:
    assert ideate.__version__ == "0.1.0"
