"""The cohort draw: groups, allocation, keys, weights, splits and the queue."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

from labeler.config import sha256_of
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
    assert {s.dir for s in specs} == {s.dir for s in databases.load_manifest(root)}
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


def test_l_over_its_cap_shares_200_across_years_with_a_minimum_of_one():
    sizes = {**SIZES, "L": {2021: 300, 2022: 150, 2023: 80, 2024: 50, 2025: 1}}
    drawn, cells = cohort.draw(_population(sizes))
    assert sum(sizes["L"].values()) == 581
    assert drawn.group.eq("L").sum() == 200
    # Quotas 103.27, 51.64, 27.54, 17.21, 0.34: 2022 and 2023 round up;
    # then 2025 takes its one from 2023, the cell furthest above its quota.
    assert {c: v["n"] for c, v in cells.items() if c.startswith("L")} == {
        "L2021": 103,
        "L2022": 52,
        "L2023": 27,
        "L2024": 17,
        "L2025": 1,
    }


def test_removing_undrawn_shots_preserves_the_draw_when_allocation_is_stable():
    frame = _population(
        {
            g: dict.fromkeys(cohort.YEARS, n)
            for g, n in (("L", 60), ("G", 40), ("R", 200))
        }
    )
    drawn, cells = cohort.draw(frame)
    removed = frame[~frame.shot.isin(drawn.shot)].groupby("cell").head(5)
    reduced = frame[~frame.shot.isin(removed.shot)]
    assert len(frame) - len(reduced) == 75
    other, new_cells = cohort.draw(reduced.sample(frac=1, random_state=7))
    assert {c: v["n"] for c, v in new_cells.items()} == {
        c: v["n"] for c, v in cells.items()
    }
    stable = ["shot", "year", "group", "cell", "split", "blind", "queue_rank", "u"]
    pd.testing.assert_frame_equal(drawn[stable], other[stable], check_exact=True)


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

    assert broken(weight=drawn.loc[0, "weight"] * 2) == {"weight", "weights"}
    assert broken(weight=0.5) == {"weight", "weights"}
    assert broken(split="holdout") == {"split"}
    assert broken(blind=not drawn.loc[0, "blind"]) == {"split", "queue_rank"}
    assert broken(cell="Q2021") == {"cell", "weight", "weights"}
    doubled = pd.concat([drawn, drawn.iloc[[0]]], ignore_index=True)
    assert {f.check for f in cohort.check_cohort(doubled, n_pop)} == {
        "unique",
        "weights",
        "weight",
        "queue_rank",
    }


def test_the_check_rejects_offsetting_weight_errors():
    drawn, cells = cohort.draw(_population())
    i, j = drawn.index[drawn.cell.eq("R2022")][:2]
    drawn.loc[i, "weight"] += 1
    drawn.loc[j, "weight"] -= 1
    found = cohort.check_cohort(drawn, {c: v["N"] for c, v in cells.items()})
    assert {f.shot for f in found if f.check == "weight"} == {
        drawn.loc[i, "shot"],
        drawn.loc[j, "shot"],
    }


@pytest.mark.parametrize("shot", [float("nan"), 185601.5, -1, float("inf"), 0])
def test_the_check_rejects_invalid_shots(shot):
    drawn, cells = cohort.draw(_population())
    drawn["shot"] = drawn["shot"].astype(float)
    drawn.loc[0, "shot"] = shot
    found = cohort.check_cohort(drawn, {c: v["N"] for c, v in cells.items()})
    assert any(f.check == "shot" and repr(shot) in f.detail for f in found)


@pytest.mark.parametrize("u", [-0.1, 1.0, float("nan"), float("inf")])
def test_the_check_rejects_invalid_order_keys(u):
    drawn, cells = cohort.draw(_population())
    drawn.loc[0, "u"] = u
    found = cohort.check_cohort(drawn, {c: v["N"] for c, v in cells.items()})
    assert any(f.check == "u" and f.shot == drawn.loc[0, "shot"] for f in found)


@pytest.mark.parametrize("change", ["one_duplicate", "all_duplicate", "blind_late"])
def test_the_check_rejects_bad_queue_ranks(change):
    drawn, cells = cohort.draw(_population())
    if change == "one_duplicate":
        drawn.loc[0, "queue_rank"] = drawn.loc[1, "queue_rank"]
    elif change == "all_duplicate":
        drawn["queue_rank"] = 1
    else:
        i = drawn.index[drawn.queue_rank.eq(1)][0]
        j = drawn.index[drawn.queue_rank.eq(51)][0]
        drawn.loc[[i, j], "queue_rank"] = [51, 1]
    found = cohort.check_cohort(drawn, {c: v["N"] for c, v in cells.items()})
    assert any(f.check == "queue_rank" and f.shot is not None for f in found)


@pytest.mark.parametrize(
    "change",
    [
        "55_test",
        "all_train",
        "val_before_test",
        "train_before_test",
        "train_before_val",
    ],
)
def test_the_check_rejects_split_counts_and_order(change):
    drawn, cells = cohort.draw(_population())
    if change == "55_test":
        indices = drawn.index[drawn.split.eq("train")][:5]
        drawn.loc[indices, "split"] = "test"
        drawn.loc[indices, "blind"] = True
    elif change == "all_train":
        drawn["split"] = "train"
        drawn["blind"] = False
    else:
        split, before = change.split("_before_")
        group = drawn[drawn.group.eq("R")]
        index = group.index[group.split.eq(split)][0]
        drawn.loc[index, "u"] = group.loc[group.split.eq(before), "u"].min() / 2
    found = cohort.check_cohort(drawn, {c: v["N"] for c, v in cells.items()})
    assert any(f.check == "split" and f.shot is not None for f in found)


def test_verification_rederives_an_untouched_draw():
    drawn, population = _tables()
    cells = cohort._cells(population)
    assert cohort.verify_cohort(drawn, population, cells) == []
    assert cohort.verify_cohort(drawn, population) == []
    assert cohort.verify_cohort(drawn, population.sample(frac=1, random_state=7)) == []


@pytest.mark.parametrize("change", ["outside", "flag_off", "flag_on", "duplicate"])
def test_verification_checks_population_membership(change):
    drawn, population = _tables()
    shot = int(drawn.loc[0, "shot"])
    if change == "outside":
        shot = 999999
        drawn.loc[0, "shot"] = shot
    elif change == "flag_off":
        population.loc[population.shot.eq(shot), "in_cohort"] = False
    elif change == "flag_on":
        i = population.index[~population.in_cohort][0]
        shot = int(population.loc[i, "shot"])
        population.loc[i, "in_cohort"] = True
    else:
        population = pd.concat([population, population.iloc[[0]]], ignore_index=True)
    found = cohort.verify_cohort(drawn, population)
    assert any(f.shot == shot for f in found)


@pytest.mark.parametrize(
    "column, value",
    [
        ("year", 2025),
        ("group", "R"),
        ("cell", "L2025"),
        ("weight", 2.0),
        ("split", "holdout"),
        ("blind", None),
        ("queue_rank", 501),
        ("u", None),
    ],
)
def test_verification_compares_each_draw_column_exactly(column, value):
    drawn, population = _tables()
    if column == "u":
        value = math.nextafter(drawn.loc[0, column], 1.0)
    elif column == "blind":
        value = not drawn.loc[0, column]
    drawn.loc[0, column] = value
    found = cohort.verify_cohort(drawn, population)
    assert any(f.shot == drawn.loc[0, "shot"] and column in f.detail for f in found)


def test_verification_checks_manifest_cells_and_seed():
    drawn, population = _tables()
    cells = cohort._cells(population)
    cells["R2022"] += 1
    assert any(
        "R2022" in f.detail for f in cohort.verify_cohort(drawn, population, cells)
    )
    assert cohort.verify_cohort(drawn, population, seed=1)


def test_an_infeasible_total_is_refused_without_exceeding_caps():
    frame = _population({"L": {2021: 231}, "G": {2021: 573}, "R": {2021: 100}})
    assert len(frame) == 904
    with pytest.raises(CatalogError) as error:
        cohort.draw(frame)
    message = str(error.value)
    for text in ("N", "L", "231", "G", "573", "R", "100", "caps", "200", "500"):
        assert text in message


def test_an_exactly_feasible_total_draws():
    frame = _population({"L": {2021: 231}, "G": {2021: 573}, "R": {2021: 200}})
    drawn, _ = cohort.draw(frame)
    assert len(drawn) == 500
    assert drawn.group.value_counts().to_dict() == {"L": 200, "G": 100, "R": 200}


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
    shots = folder / "pool_shots.txt"
    shots.write_text("".join(f"{line['shot']}\n" for line in lines))
    (folder / "pool.meta.json").write_text(
        json.dumps(
            {
                "pool_sha256": sha256_of(folder / "pool.csv"),
                "pool_shots_sha256": sha256_of(shots),
                "rules": pop.rules_record(),
            }
        )
    )
    (folder / "ip.meta.json").write_text(
        json.dumps(
            {
                "version": window.LOG_VERSION,
                "git_sha": "fixture",
                "log": str(folder / "ip.jsonl"),
                "shot_file": str(shots),
                "shot_file_sha256": sha256_of(shots),
                "shots": len(lines),
                "definition": window.definition(),
            }
        )
    )
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
    assert cohort.main(["--out", str(folder)]) == 0
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
    assert (
        cohort.verify_cohort(
            drawn,
            cohort.read_population(folder / "population.csv"),
            {c: v["N"] for c, v in doc["cells"].items()},
        )
        == []
    )
    assert json.loads(capsys.readouterr().out)["population"] == 630


def test_the_command_needs_the_verified_links(tmp_path, monkeypatch, capsys):
    _inputs(tmp_path / "root" / "catalog")
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("LABELER_LABEL_TABLES", str(tmp_path / "tables"))
    with pytest.raises(SystemExit):
        cohort.main(["--out", str(tmp_path / "out")])
    assert "papers.csv does not exist" in capsys.readouterr().err


@pytest.mark.parametrize("change", ["weight", "u", "serialized_u"])
def test_the_command_checks_the_serialized_draw_before_writing(
    tmp_path, monkeypatch, change
):
    folder, out = tmp_path / "inputs", tmp_path / "out"
    legacy = _inputs(folder)
    monkeypatch.setattr(cohort, "legacy_sets", lambda root: (legacy, []))
    if change == "serialized_u":
        original = cohort.read_cohort

        def corrupt_read(path):
            frame = original(path)
            frame.loc[0, "u"] = math.nextafter(frame.loc[0, "u"], 1.0)
            return frame

        monkeypatch.setattr(cohort, "read_cohort", corrupt_read)
    else:
        original = cohort.draw

        def corrupt_draw(*args, **kwargs):
            frame, cells = original(*args, **kwargs)
            frame.loc[0, change] = math.nextafter(frame.loc[0, change], 1.0)
            if change == "weight":
                frame.loc[0, change] += 1
            return frame, cells

        monkeypatch.setattr(cohort, "draw", corrupt_draw)
    with pytest.raises(CatalogError):
        cohort.main(
            [
                "--pool",
                str(folder / "pool.csv"),
                "--ip-log",
                str(folder / "ip.jsonl"),
                "--papers",
                str(folder / "papers.csv"),
                "--out",
                str(out),
            ]
        )
    assert not out.exists()


def test_the_command_refuses_an_infeasible_total(tmp_path, monkeypatch):
    folder, out = tmp_path / "inputs", tmp_path / "out"
    legacy = _inputs(folder)
    monkeypatch.setattr(cohort, "legacy_sets", lambda root: (legacy, []))
    small = _population({"L": {2021: 231}, "G": {2021: 573}, "R": {2021: 100}})
    monkeypatch.setattr(cohort, "assign_groups", lambda *args: small)
    with pytest.raises(CatalogError, match="500"):
        cohort.main(
            [
                "--pool",
                str(folder / "pool.csv"),
                "--ip-log",
                str(folder / "ip.jsonl"),
                "--papers",
                str(folder / "papers.csv"),
                "--out",
                str(out),
            ]
        )
    assert not out.exists()


@pytest.mark.parametrize("relative_root", [False, True])
def test_relative_papers_are_recorded_under_resolved_label_tables(
    tmp_path, monkeypatch, relative_root
):
    folder, out = tmp_path / "tables" / "catalog", tmp_path / "out"
    legacy = _inputs(folder)
    monkeypatch.setattr(cohort, "legacy_sets", lambda root: (legacy, []))
    monkeypatch.chdir(tmp_path)
    root = Path("tables") if relative_root else tmp_path / "tables"
    monkeypatch.setenv("LABELER_LABEL_TABLES", str(root))
    assert (
        cohort.main(
            [
                "--pool",
                str(folder / "pool.csv"),
                "--ip-log",
                str(folder / "ip.jsonl"),
                "--papers",
                "tables/catalog/papers.csv",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    doc = yaml.safe_load((out / "cohort_manifest.yaml").read_text())
    assert doc["inputs"]["papers"]["path"] == "catalog/papers.csv"


def test_an_input_outside_the_root_uses_its_resolved_absolute_path(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    papers = tmp_path / "papers.csv"
    papers.write_text("papers")
    alias = outside / ".." / "papers.csv"
    assert cohort._input(alias, outside)["path"] == str(papers.resolve())


@pytest.mark.parametrize(
    "change, filename, detail",
    [
        ("no_pool_meta", "pool.meta.json", "missing"),
        ("stale_sha", "pool.meta.json", "pool_sha256"),
        ("other_rules", "pool.meta.json", "rules"),
        ("no_ip_meta", "ip.meta.json", "missing"),
    ],
)
def test_the_command_refuses_unverified_provenance(
    tmp_path, monkeypatch, change, filename, detail
):
    folder, out = tmp_path / "inputs", tmp_path / "out"
    legacy = _inputs(folder)
    monkeypatch.setattr(cohort, "legacy_sets", lambda root: (legacy, []))
    meta_path = folder / filename
    if change.startswith("no_"):
        meta_path.unlink()
    else:
        meta = json.loads(meta_path.read_text())
        meta[detail] = "0" * 64 if change == "stale_sha" else {"old": "rules"}
        meta_path.write_text(json.dumps(meta))
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    with pytest.raises(CatalogError) as error:
        cohort.main(
            [
                "--pool",
                str(folder / "pool.csv"),
                "--ip-log",
                str(folder / "ip.jsonl"),
                "--papers",
                str(folder / "papers.csv"),
                "--out",
                str(out),
            ]
        )
    assert str(meta_path) in str(error.value) and detail in str(error.value)
    assert not out.exists()
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}


@pytest.mark.parametrize("field", ["outputs", "ip_log_meta", "keys"])
def test_the_manifest_records_its_outputs_and_provenance(tmp_path, monkeypatch, field):
    folder, out = tmp_path / "inputs", tmp_path / "out"
    legacy = _inputs(folder)
    monkeypatch.setattr(cohort, "legacy_sets", lambda root: (legacy, []))
    assert (
        cohort.main(
            [
                "--pool",
                str(folder / "pool.csv"),
                "--ip-log",
                str(folder / "ip.jsonl"),
                "--papers",
                str(folder / "papers.csv"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    doc = yaml.safe_load((out / "cohort_manifest.yaml").read_text())
    expected = {
        "outputs": {
            name: sha256_of(out / name) for name in ("cohort.csv", "population.csv")
        },
        "ip_log_meta": json.loads((folder / "ip.meta.json").read_text()),
        "keys": "sha256 of '<seed>:<purpose>:<shot>', its first 8 bytes as a "
        "big-endian unsigned integer, / 2**64; "
        "purposes draw (the cell sample) and order (u)",
    }
    assert doc[field] == expected[field]
    assert doc["pool"] == json.loads((folder / "pool.meta.json").read_text())


@pytest.mark.parametrize("dirty", [True, False, None])
def test_the_manifest_records_git_dirty(tmp_path, monkeypatch, dirty):
    folder, out = tmp_path / "inputs", tmp_path / "out"
    legacy = _inputs(folder)
    monkeypatch.setattr(cohort, "legacy_sets", lambda root: (legacy, []))
    monkeypatch.setattr(cohort, "git_dirty", lambda: dirty, raising=False)
    assert (
        cohort.main(
            [
                "--pool",
                str(folder / "pool.csv"),
                "--ip-log",
                str(folder / "ip.jsonl"),
                "--papers",
                str(folder / "papers.csv"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    doc = yaml.safe_load((out / "cohort_manifest.yaml").read_text())
    assert doc["git_dirty"] is dirty and "git_sha" in doc
