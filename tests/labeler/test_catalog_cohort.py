"""The cohort draw: groups, allocation, keys, weights, splits and the queue."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from labeler.events import databases
from labeler.events.catalog import cohort, window
from labeler.events.catalog import population as pop
from labeler.events.catalog.check import CatalogError
from labeler.literature.papers import papers_frame, write_papers

REPO = Path(__file__).resolve().parents[2]
#: N per cell of a made-up population: L small enough to take whole, G and R not.
SIZES = {
    "L": {2021: 5, 2022: 40, 2023: 50, 2024: 45, 2025: 10},
    "G": {2021: 300, 2022: 150, 2023: 80, 2024: 50, 2025: 2},
    "R": {2021: 400, 2022: 1600, 2023: 900, 2024: 1100, 2025: 800},
}


def _population(sizes=SIZES):
    """A population as `assign_groups` leaves it, its shots numbered in order."""
    rows = []
    for group, years in sizes.items():
        for year, n in years.items():
            rows += [{"year": year, "group": group, "cell": f"{group}{year}"}] * n
    frame = pd.DataFrame(rows)
    frame.insert(0, "shot", range(185_601, 185_601 + len(frame)))
    return frame.assign(
        year=frame["year"].astype("Int64"),
        window_start_ms=7,
        window_end_ms=5000,
        ip_ma=1.1,
        pulse_length_s=5.0,
        flattop_s=3.0,
        ip_peak_ma=1.2,
        pbeam_max_mw=2.5,
        pech_max_mw=0.0,
        run_id=frame["year"].astype(str) + "0301",
        legacy_sets=frame["group"].map({"G": "edge_localized_mode"}).fillna(""),
        n_links_verified=frame["group"].eq("L").astype(int),
        **{span: 5.0 for span in cohort.SPANS},
    )


def _paper(shot, source="osti", record_id="1"):
    return {
        "shot": shot,
        "source": source,
        "record_id": record_id,
        "doi": "",
        "title": "A paper",
        "year": 2024,
        "venue": "",
        "context": "shot",
        "match_type": "exact",
        "verified_by": "auto",
    }


def _tables():
    population = _population()
    population["flattop_s"] = 1.2345678901234567
    drawn, _ = cohort.draw(population)
    population["in_cohort"] = population["shot"].isin(drawn["shot"])
    return drawn, population[list(cohort.POPULATION_COLUMNS)]


@pytest.mark.parametrize("name", ["cohort", "population"])
def test_tables_read_back_every_column_exactly(tmp_path, name):
    drawn, population = _tables()
    expected = {"cohort": drawn, "population": population}[name].astype(
        {"year": "int64", "window_start_ms": "Int64", "window_end_ms": "Int64"}
    )
    path = tmp_path / f"{name}.csv"
    expected.to_csv(path, index=False)
    got = getattr(cohort, f"read_{name}")(path)
    pd.testing.assert_frame_equal(got, expected, check_exact=True)


@pytest.mark.parametrize(
    "name, column", [("cohort", "blind"), ("population", "in_cohort")]
)
@pytest.mark.parametrize("value", ["", "Ture", "true", "1", " False"])
def test_table_booleans_are_strict(tmp_path, name, column, value):
    drawn, population = _tables()
    frame = {"cohort": drawn, "population": population}[name].copy()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(CatalogError) as error:
        getattr(cohort, f"read_{name}")(path)
    assert str(path) in str(error.value)
    assert "row 2" in str(error.value)
    assert column in str(error.value) and repr(value) in str(error.value)


@pytest.mark.parametrize("name", ["cohort", "population"])
@pytest.mark.parametrize("change", ["missing", "extra", "reordered"])
def test_table_columns_are_exact(tmp_path, name, change):
    drawn, population = _tables()
    frame = {"cohort": drawn, "population": population}[name]
    if change == "missing":
        frame = frame.drop(columns="shot")
    elif change == "extra":
        frame = frame.assign(extra=1)
    else:
        frame = frame[frame.columns[::-1]]
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(CatalogError, match="expected the columns"):
        getattr(cohort, f"read_{name}")(path)


def test_a_key_depends_on_the_seed_the_purpose_and_the_shot_alone():
    assert cohort.key(20260923, "draw", 190000) == 0.884898099725107
    assert cohort.key(20260923, "order", 190000) == 0.11216141712330072
    assert cohort.key(1, "draw", 190000) != cohort.key(20260923, "draw", 190000)
    keys = pd.Series([cohort.key(cohort.SEED, "draw", s) for s in range(20_000)])
    assert keys.between(0, 1, inclusive="left").all()
    assert keys.mean() == pytest.approx(0.5, abs=0.01)
    assert (keys < 0.25).mean() == pytest.approx(0.25, abs=0.01)


def test_allocation_is_largest_remainder_with_one_per_non_empty_cell():
    assert cohort.allocate({"a": 50, "b": 30, "c": 20}, 10) == {"a": 5, "b": 3, "c": 2}
    # a tie on the remainder goes to the first cell by name
    assert cohort.allocate({"a": 3, "b": 3, "c": 2}, 4) == {"a": 2, "b": 1, "c": 1}
    # quotas 3.5, 1, 0.5: a rounds up, then gives c its one shot, being furthest over
    assert cohort.allocate({"a": 7, "b": 2, "c": 1}, 5) == {"a": 3, "b": 1, "c": 1}
    # quotas 51.55, 25.77, 13.75, 8.59, 0.34: the three largest remainders round up,
    # then G2025 takes its shot from G2024, the cell furthest over (9 - 8.59)
    sizes = {"G2021": 300, "G2022": 150, "G2023": 80, "G2024": 50, "G2025": 2}
    assert cohort.allocate(sizes, 100) == {
        "G2021": 51,
        "G2022": 26,
        "G2023": 14,
        "G2024": 8,
        "G2025": 1,
    }
    assert cohort.allocate({"a": 10, "b": 0, "c": 5}, 3) == {"a": 2, "b": 0, "c": 1}
    assert cohort.allocate({"a": 3, "b": 0}, 10) == {"a": 3, "b": 0}  # takes everything
    with pytest.raises(ValueError, match="cannot give each"):
        cohort.allocate({"a": 5, "b": 5, "c": 5}, 2)


def test_a_shot_takes_the_first_group_that_applies():
    frame = pd.DataFrame(
        {
            "shot": [1, 2, 3, 4],
            "year": pd.array([2021, 2022, 2023, 2025], dtype="Int64"),
        }
    )
    papers = papers_frame([_paper(1), _paper(1, record_id="2"), _paper(3, "arxiv")])
    links = cohort.osti_links(papers)
    assert links == {1: 2}  # an arXiv link never moves a shot
    legacy = {
        1: ("neoclassical_tearing_mode",),
        2: ("edge_localized_mode", "high_confinement_mode"),
    }
    got = cohort.assign_groups(frame, links, legacy)
    assert got[
        ["group", "cell", "legacy_sets", "n_links_verified"]
    ].values.tolist() == [
        ["L", "L2021", "neoclassical_tearing_mode", 2],
        ["G", "G2022", "edge_localized_mode;high_confinement_mode", 0],
        ["R", "R2023", "", 0],
        ["R", "R2025", "", 0],
    ]


def test_a_campaign_year_outside_the_cells_is_refused():
    frame = pd.DataFrame(
        {"shot": [1, 2], "year": pd.array([2020, None], dtype="Int64")}
    )
    with pytest.raises(CatalogError, match=r"\[2020, 'none'\]"):
        cohort.assign_groups(frame, {}, {})


def test_the_legacy_sets_are_the_registered_tables():
    root = REPO / "data" / "events"
    legacy, specs = cohort.legacy_sets(root)
    assert {
        "neoclassical_tearing_mode",
        "edge_localized_mode",
        "high_confinement_mode",
        "resistive_wall_mode",
    } <= {s.dir for s in specs}
    elm = next(s for s in specs if s.dir == "edge_localized_mode")
    assert "edge_localized_mode" in legacy[min(databases.shots(elm, root))]


def test_each_cell_gets_its_share_and_the_weights_reproduce_n():
    drawn, cells = cohort.draw(_population())
    assert len(drawn) == 500
    assert drawn["cell"].value_counts().to_dict() == {
        "L2021": 5,
        "L2022": 40,
        "L2023": 50,
        "L2024": 45,
        "L2025": 10,
        "G2021": 51,
        "G2022": 26,
        "G2023": 14,
        "G2024": 8,
        "G2025": 1,
        # quotas 20.83, 83.33, 46.88, 57.29, 41.67: .88, .83 and .67 round up
        "R2021": 21,
        "R2022": 83,
        "R2023": 47,
        "R2024": 57,
        "R2025": 42,
    }
    assert cells["G2021"] == {"N": 300, "n": 51} and len(cells) == 15
    assert drawn.loc[drawn["cell"].eq("G2021"), "weight"].unique().tolist() == [
        300 / 51
    ]
    assert cohort.check_cohort(drawn, {c: v["N"] for c, v in cells.items()}) == []


def test_each_cell_is_its_smallest_draw_keys_and_the_seed_moves_them():
    frame = _population()
    drawn, _ = cohort.draw(frame)
    keyed = frame.assign(
        k=frame["shot"].map(lambda s: cohort.key(cohort.SEED, "draw", s)),
        picked=frame["shot"].isin(drawn["shot"]),
    )
    for cell, part in keyed.groupby("cell"):
        if not part["picked"].all():
            assert (
                part.loc[part.picked, "k"].max() < part.loc[~part.picked, "k"].min()
            ), cell
    pd.testing.assert_frame_equal(cohort.draw(frame)[0], drawn)
    other, _ = cohort.draw(frame, seed=1)
    assert set(other["shot"]) != set(drawn["shot"])
    assert set(other.loc[other.group.eq("L"), "shot"]) == set(
        frame.loc[frame.group.eq("L"), "shot"]
    )


def test_the_splits_follow_u_within_each_group_and_the_queue_starts_blind():
    drawn, _ = cohort.draw(_population())
    counts = drawn.groupby(["group", "split"]).size().unstack()
    assert counts.loc[["L", "G", "R"], ["test", "val", "train"]].values.tolist() == [
        [15, 15, 120],
        [10, 10, 80],
        [25, 25, 200],
    ]
    for _, part in drawn.groupby("group"):
        u = {s: part.loc[part["split"].eq(s), "u"] for s in cohort.SPLITS}
        assert u["test"].max() < u["val"].min() and u["val"].max() < u["train"].min()
    assert drawn["blind"].eq(drawn["split"].eq("test")).all()
    queue = drawn.sort_values("queue_rank")
    assert queue["queue_rank"].tolist() == list(range(1, 501))
    assert queue["blind"].iloc[:50].all() and not queue["blind"].iloc[50:].any()
    assert queue["u"].iloc[:50].is_monotonic_increasing
    assert queue["u"].iloc[50:].is_monotonic_increasing


def test_the_check_finds_each_kind_of_break():
    drawn, cells = cohort.draw(_population())
    n_pop = {c: v["N"] for c, v in cells.items()}

    def broken(**changes):
        frame = drawn.copy()
        for column, value in changes.items():
            frame.loc[0, column] = value
        return {f.check for f in cohort.check_cohort(frame, n_pop)}

    assert broken(weight=drawn.loc[0, "weight"] * 2) == {"weights"}
    assert broken(weight=0.5) == {"weight", "weights"}
    assert broken(split="holdout") == {"split"}
    assert broken(blind=not drawn.loc[0, "blind"]) == {"split"}
    assert broken(cell="Q2021") == {"cell", "weights"}
    doubled = pd.concat([drawn, drawn.iloc[[0]]], ignore_index=True)
    assert {f.check for f in cohort.check_cohort(doubled, n_pop)} == {
        "unique",
        "weights",
    }


def _inputs(folder: Path):
    """700 shots pass rules 1-3, one fails rule 1; every tenth has a short flat-top."""
    rows, lines = [], []
    for i in range(700):
        shot, year = 185_601 + 20 * i, 2021 + i % 5
        rows.append(
            {
                "shot": shot,
                "year": year,
                "run_id": f"{year}0301",
                "ip_ma": 1.0,
                "pulse_length_s": 5.0,
                "pbeam_max_mw": 2.0,
                "pech_max_mw": 0.0,
                **{span: 5.0 for span in cohort.SPANS},
                "reasons": "",
            }
        )
        lines.append(
            {
                "shot": shot,
                "status": "ok",
                "window_start_ms": 7,
                "window_end_ms": 5000,
                "flattop_s": 0.5 if i % 10 == 0 else 3.0,
                "ip_peak_ma": 1.0,
                "dt_ms": 0.05,
                "version": window.LOG_VERSION,
            }
        )
    rows.append({"shot": 185_602, "year": 2021, "reasons": "heating"})
    folder.mkdir(parents=True)
    pd.DataFrame(rows, columns=list(pop.POOL_COLUMNS)).to_csv(
        folder / "pool.csv", index=False
    )
    (folder / "ip.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    write_papers(
        papers_frame([_paper(185_601 + 20 * k) for k in (1, 2, 3)]),
        folder / "papers.csv",
    )
    return {185_601 + 20 * k: ("edge_localized_mode",) for k in range(11, 111)}


def test_the_command_writes_the_cohort_and_its_manifest(tmp_path, monkeypatch, capsys):
    folder, tables = tmp_path / "root" / "catalog", tmp_path / "tables"
    legacy = _inputs(folder)
    (tables / "catalog").mkdir(parents=True)
    (folder / "papers.csv").rename(tables / "catalog" / "papers.csv")
    monkeypatch.setattr(cohort, "legacy_sets", lambda root: (legacy, []))
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("LABELER_LABEL_TABLES", str(tables))
    assert cohort.main([]) == 0
    drawn = cohort.read_cohort(folder / "cohort.csv")
    doc = yaml.safe_load((folder / "cohort_manifest.yaml").read_text())
    counts = doc["counts"]
    assert counts["population"] == 630 and len(drawn) == 500
    assert counts["after_rule_1"] == 700 and counts["after_rule_4"] == 630
    assert list(counts["rejections"]) == [
        r for rule in pop.RULES.values() for r in rule
    ]
    assert {k: v for k, v in counts["rejections"].items() if v} == {
        "heating": 1,
        "flattop": 70,
    }
    # L: 3 shots with links; G: 90 legacy shots (10 fail rule 4); R: 407 of 537
    assert counts["groups"] == {
        "L": {"N": 3, "n": 3, "test": 1, "val": 1, "train": 1},
        "G": {"N": 90, "n": 90, "test": 9, "val": 9, "train": 72},
        "R": {"N": 537, "n": 407, "test": 40, "val": 40, "train": 327},
    }
    assert sum(c["N"] for c in doc["cells"].values()) == 630
    assert (
        cohort.check_cohort(drawn, {c: v["N"] for c, v in doc["cells"].items()}) == []
    )
    assert doc["seed"] == cohort.SEED and doc["replacements"] == []
    assert (
        len(doc["inputs"]["ip_log"]["sha256"]) == 64
        and doc["inputs"]["legacy_tables"] == []
    )
    # an input under the label tables is recorded relative to them
    assert doc["inputs"]["papers"]["path"] == "catalog/papers.csv"
    assert doc["inputs"]["pool"]["path"] == str(folder / "pool.csv")
    population = pd.read_csv(folder / "population.csv")
    assert len(population) == 630 and population["in_cohort"].sum() == 500
    assert json.loads(capsys.readouterr().out)["population"] == 630


def test_the_command_needs_the_verified_links(tmp_path, monkeypatch, capsys):
    _inputs(tmp_path / "root" / "catalog")
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("LABELER_LABEL_TABLES", str(tmp_path / "tables"))
    with pytest.raises(SystemExit):
        cohort.main([])
    assert "papers.csv does not exist" in capsys.readouterr().err
