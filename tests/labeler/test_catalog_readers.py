"""Structural guards at every frozen-table and pool reader."""

import io
from pathlib import Path

import pandas as pd
import pytest

from labeler.events.catalog import cohort, population
from labeler.literature import papers

READERS = [
    (cohort.read_cohort, cohort.COHORT_COLUMNS),
    (cohort.read_population, cohort.POPULATION_COLUMNS),
    (population.read_pool, population.POOL_COLUMNS),
    (papers.read_papers, papers.PAPER_COLUMNS),
]
ROOT = Path(__file__).resolve().parents[2] / "data/events/catalog"


@pytest.mark.parametrize("reader, columns", READERS)
@pytest.mark.parametrize("kind", ["extra", "short", "duplicate"])
def test_readers_refuse_structure_with_physical_lines(tmp_path, reader, columns, kind):
    path = tmp_path / "table.csv"
    header = list(columns)
    if kind == "duplicate":
        header[-1] = header[0]
        text = ",".join(header) + "\n"
        line, detail = 1, "duplicate headers"
    else:
        # One complete quoted record starts on line 3 and spans two physical lines.
        text = ",".join(header) + '\n\n"a\nb",' + ",".join(["0"] * (len(header) - 1))
        width = len(header) + (1 if kind == "extra" else -1)
        text += "\n" + ",".join(["0"] * width) + "\n"
        line, detail = 5, "expected.*fields"
    path.write_text(text)
    with pytest.raises(pd.errors.ParserError, match=f"{path}: row {line}:.*{detail}"):
        reader(path)


@pytest.mark.parametrize("name", ["cohort", "population", "papers"])
@pytest.mark.parametrize("stream", [False, True])
def test_committed_reader_preserves_pandas_values(name, stream):
    path = ROOT / f"{name}.csv"
    data = path.read_bytes()
    source = io.StringIO(data.decode()) if stream else path
    if name == "papers":
        expected = pd.read_csv(
            io.BytesIO(data),
            dtype={"record_id": str, "doi": str},
            keep_default_na=False,
            na_values={"year": [""]},
        )
        expected = expected.astype({"year": "Int64"})
        actual = papers.read_papers(source)
    else:
        columns = (
            cohort.COHORT_COLUMNS if name == "cohort" else cohort.POPULATION_COLUMNS
        )
        boolean = "blind" if name == "cohort" else "in_cohort"
        strings = ("run_id", "legacy_sets", "split", "group", "cell", boolean)
        expected = pd.read_csv(
            io.BytesIO(data),
            dtype={c: str for c in strings},
            float_precision="round_trip",
            keep_default_na=False,
            na_values={c: [""] for c in columns if c not in strings},
        )
        expected[boolean] = expected[boolean].eq("True")
        expected = expected.astype(
            {"window_start_ms": "Int64", "window_end_ms": "Int64"}
        )
        actual = getattr(cohort, f"read_{name}")(source)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_pool_reader_accepts_a_stream_with_its_missing_value_contract():
    values = {c: "" for c in population.POOL_COLUMNS}
    values.update(shot=190001, year=2022, has_shot_table=True, reasons="")
    frame = population.read_pool(
        io.StringIO(pd.DataFrame([values]).to_csv(index=False))
    )
    assert frame.reasons.tolist() == [""]
    assert frame.title.isna().all()
