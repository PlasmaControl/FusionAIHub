"""The package exists, is importable from every environment that lists it, and
carries the version the pyproject wheel target promises."""
from __future__ import annotations

import importlib
import subprocess
import sys

import shot_design


def test_version() -> None:
    assert shot_design.__version__ == "0.1.0"


def test_importing_the_module_entry_point_does_not_run_the_cli() -> None:
    """`raise SystemExit(main())` at module level ran the whole CLI on a bare
    `import shot_design.__main__` -- which is what a doc tool, a coverage run or a reload does."""
    mod = importlib.import_module("shot_design.__main__")
    assert callable(mod.main)
    assert importlib.reload(mod) is mod  # a second import is still not an invocation


def test_python_dash_m_shot_design_still_runs_the_cli() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "shot_design", "--help"], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0 and "usage: shot_design" in r.stdout
