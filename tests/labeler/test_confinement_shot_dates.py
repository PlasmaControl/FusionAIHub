"""Run-day dating of the curated confinement shots: the year and consistency of each
shot from the time its EFIT reconstruction was inserted."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/labeler/confinement_shot_dates_fetch.py"
)
spec = importlib.util.spec_from_file_location("confinement_shot_dates_fetch", SCRIPT)
dates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dates)


def test_vms_ticks_are_100_ns_since_1858():
    when = dates.vms_to_utc(40587 * 86400 * 10_000_000)  # 1970-01-01 in VMS days
    assert (when.year, when.month, when.day) == (1970, 1, 1)


def test_assign_years_takes_the_neighbours_year_for_an_undated_or_late_shot():
    stamps = [f"2019-05-{d:02d}T15:00:00+00:00" for d in range(1, 12)]
    stamps[4] = ""  # no stamp
    stamps[7] = "2021-01-20T10:00:00+00:00"  # EFIT run two years later
    frame = pd.DataFrame(
        {"shot": range(100, 111), "inserted_utc": stamps, "source_node": "EFIT01"}
    )
    out = dates.assign_years(frame).set_index("shot")
    assert out.loc[104, "year"] == 2019 and not out.loc[104, "consistent"]
    assert out.loc[107, "year"] == 2019 and not out.loc[107, "consistent"]
    assert out.loc[100, "year"] == 2019 and out.loc[100, "consistent"]
    assert dates.summary(out.reset_index())["inconsistent"] == [104, 107]
