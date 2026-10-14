"""The gold set: 50 shots from 170000 up with hand labels or a curated source, from a few run days,
spread over the shot-number decades."""

import pandas as pd
import pytest

from labeler.config import DEFAULT_LABEL_TABLES
from labeler.events.review import gold, silver

HEADER = "shot,run_day,year,classes,sources,notes\n"


def _tables(tmp_path, rows, blind=()):
    tables = tmp_path / "events"
    (tables / "gold").mkdir(parents=True)
    (tables / "gold" / "gold_shots.csv").write_text(HEADER + "".join(rows))
    if blind:
        (tables / "catalog").mkdir()
        pd.DataFrame({"shot": list(blind), "blind": True}).to_csv(tables / "catalog" / "cohort.csv", index=False)
    return tables


def test_a_gold_list_reads_with_its_run_days_as_text(tmp_path):
    tables = _tables(tmp_path, ["170815,20170504,2017,confinement:high,source,\n"])
    got = gold.read_gold(tables)
    assert got.shot.tolist() == [170815] and got.run_day.tolist() == ["20170504"]


@pytest.mark.parametrize("rows, message", [
    (["169999,20170504,2017,,saved,\n"], "below 170000"),
    (["170815,20170504,2017,,saved,\n", "170815,20170504,2017,,saved,\n"], "duplicate"),
    (["170815,2017-05-04,2017,,saved,\n"], "YYYYMMDD"),
])
def test_a_bad_gold_list_is_refused(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        gold.read_gold(_tables(tmp_path, rows))


def test_a_blind_shot_may_not_be_gold(tmp_path):
    tables = _tables(tmp_path, ["170815,20170504,2017,,saved,\n"], blind=[170815])
    with pytest.raises(ValueError, match="blind"):
        gold.read_gold(tables)


@pytest.mark.label_tables
def test_the_shipped_gold_list_is_fifty_shots_from_few_run_days_and_not_silver():
    got = gold.read_gold(DEFAULT_LABEL_TABLES)
    assert len(got) == 50 and got.shot.is_unique and (got.shot >= 170_000).all()
    assert got.run_day.nunique() <= 13
    assert got.groupby("run_day").size().max() >= 6, "several shots from the same run day"
    assert all(set(row.sources.split(",")) & {"saved", "format", "source", "logbook"}
               or "added by the owner" in row.notes
               for row in got.itertuples()), "hand labels or a curated source, or the owner's pick"
    assert {199597, 199603, 199607} <= set(got.shot), "the owner's 2024-07-11 TM experiment shots"
    assert not set(got.shot) & set(silver.read_silver(DEFAULT_LABEL_TABLES).shot)


@pytest.mark.label_tables
def test_the_shipped_gold_list_has_about_the_same_number_of_shots_from_each_shot_number_decade():
    """17xxxx, 18xxxx, 19xxxx and 20xxxx (no 21xxxx shot exists in the corpus) share the 50."""
    per_decade = gold.read_gold(DEFAULT_LABEL_TABLES).shot.floordiv(10_000).value_counts()
    assert set(per_decade.index) == {17, 18, 19, 20}
    assert per_decade.max() - per_decade.min() <= 2


@pytest.mark.label_tables
def test_the_shipped_gold_list_covers_every_class_the_silver_list_does():
    seen = {c for classes in gold.read_gold(DEFAULT_LABEL_TABLES).classes for c in classes.split(";") if c}
    assert "resistive_wall_mode:present" in seen and "confinement:imode" in seen
    assert {"alfven_eigenmode.tae:present", "alfven_eigenmode.rsae:present", "detachment:detached",
            "mhd_n1.m2:present", "mhd_n2.m3:present", "minimum_safety_factor:low"} <= seen
