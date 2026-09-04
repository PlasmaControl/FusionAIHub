"""Opt-in gating for the tests that fetch live data through fdp.

Everything else in this directory is hermetic: it must pass under a plain
`pixi run -e labelmaker python -m pytest tests/labelmaker`, with no server,
no token and no network. A live test is marked `live` and is skipped unless
it is asked for explicitly, either with `--run-live` or with
`LABELMAKER_FDP=1` (the environment switch the plan's command blocks use).

A live run also needs the `fdp run` wrapper, which supplies the server
configuration: without it PTDATA fails with `getservbyname failed for task
'PTSERVER'` and MDSplus with `TREE-E-FOPENR`.

    pixi run -e labelmaker fdp run python -m pytest tests/labelmaker \
        -q -W error --run-live
"""
from __future__ import annotations

import os

import pytest

#: Environment switch, equivalent to `--run-live`.
LIVE_ENV = "LABELMAKER_FDP"


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run the tests that fetch real data through fdp",
    )


def pytest_configure(config) -> None:
    # Registered, not just used: `-W error` promotes pytest's
    # PytestUnknownMarkWarning to an error, so an unregistered marker would
    # fail the hermetic gate rather than merely warn.
    config.addinivalue_line(
        "markers",
        "live: fetches real data through fdp; needs `fdp run` and a SciToken",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--run-live") or os.environ.get(LIVE_ENV) == "1":
        return
    skip = pytest.mark.skip(
        reason="live fdp fetch is opt-in: --run-live or LABELMAKER_FDP=1"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
