"""The corpus census: `ideate corpus scan` and the table it writes.

The census exists because the availability question was answered wrongly once already: the old
shotsearch manifest counted a group as populated when `shape[0] > 1`, which is the CHANNEL axis,
so every `(C, 1)` placeholder counted as present. So what this file pins hardest is the sentinel
axis, and the census's own honesty about the ~2.3 % of corpus files that do not open at all --
a file that cannot be read is a row saying so, never a silently missing shot.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ideate import cli
from ideate.shotdb import census

from .conftest import CORPUS_ABSENT, CORPUS_FULL, CORPUS_SMALL, CORPUS_TRUNCATED

COLUMNS = {
    "shot": "int32",
    "group": "str",
    "kind": "str",
    "n_channels": "int32",
    "n_samples": "int64",
    "t0_s": "float64",
    "t1_s": "float64",
    "fs_hz": "float64",
    "present": "bool",
    "openable": "bool",
    "error": "str",
}


@pytest.fixture
def scanned(corpus_dir: Path) -> pd.DataFrame:
    return census.scan(corpus_dir, workers=2)


def row(df: pd.DataFrame, shot: int, group: str) -> pd.Series:
    hit = df[(df["shot"] == shot) & (df["group"] == group)]
    assert len(hit) == 1, f"{shot}/{group}: {len(hit)} rows"
    return hit.iloc[0]


# ------------------------------------------------------------------------------- the table


def test_one_row_per_file_and_group_sorted_by_shot_then_group(scanned):
    assert list(zip(scanned["shot"], scanned["group"], strict=True)) == [
        (CORPUS_FULL, "co2"),
        (CORPUS_FULL, "gas_flow"),
        (CORPUS_FULL, "mhr"),
        (CORPUS_FULL, "pinj"),
        (CORPUS_FULL, "tangtv"),
        (CORPUS_TRUNCATED, ""),
        (CORPUS_SMALL, "co2"),
        (CORPUS_SMALL, "pinj"),
    ]
    assert list(scanned.index) == list(range(len(scanned)))


def test_column_names_and_dtypes_are_exact(scanned):
    assert {c: str(dt) for c, dt in scanned.dtypes.items()} == COLUMNS


def test_the_table_survives_a_parquet_round_trip_unchanged(scanned, tmp_path):
    p = tmp_path / "c.parquet"
    scanned.to_parquet(p, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(p), scanned)


# --------------------------------------------------------------------------- what a row says


def test_a_placeholder_group_is_a_row_that_says_not_present(scanned):
    co2 = row(scanned, CORPUS_FULL, "co2")
    # (4, 1): four channels, one sample. Counting on shape[0] would call this present.
    assert co2["n_channels"] == 4 and co2["n_samples"] == 1
    assert bool(co2["present"]) is False and bool(co2["openable"]) is True
    # Its xdata is a single 0.0 that measures nothing, so the span is NaN and not (0.0, 0.0).
    assert all(math.isnan(co2[c]) for c in ("t0_s", "t1_s", "fs_hz"))


def test_a_recorded_group_reports_its_span_and_rate(scanned):
    pinj = row(scanned, CORPUS_FULL, "pinj")
    assert pinj["kind"] == "signal" and pinj["n_channels"] == 8 and pinj["n_samples"] == 5
    assert (pinj["t0_s"], pinj["t1_s"]) == pytest.approx((0.0, 0.004))
    assert pinj["fs_hz"] == pytest.approx(1000.0)  # (5 - 1) / (0.004 - 0.0)
    assert bool(pinj["present"]) and pinj["error"] == ""
    assert row(scanned, CORPUS_FULL, "mhr")["fs_hz"] == pytest.approx(500_000.0)


def test_the_census_reports_the_shapes_on_disk_not_the_read_view(scanned):
    # mhr's ninth sample is the all-NaN pad CorpusReader.read strips. The census is header-only --
    # shapes and two xdata scalars -- so it counts 9, and 9 is what the file holds.
    assert row(scanned, CORPUS_FULL, "mhr")["n_samples"] == 9
    assert row(scanned, CORPUS_FULL, "gas_flow")["n_samples"] == 6


def test_a_video_group_is_measured_on_its_own_time_axis(scanned):
    tangtv = row(scanned, CORPUS_FULL, "tangtv")  # ydata is (7, 3, 2, 4)
    assert tangtv["kind"] == "video" and tangtv["n_channels"] == 7 and tangtv["n_samples"] == 3
    assert bool(tangtv["present"]) and tangtv["fs_hz"] == pytest.approx(50.0)


def test_a_file_that_opens_and_holds_nothing_is_still_a_row(corpus_dir):
    """Otherwise it would leave the denominator of every fraction without saying so."""
    import h5py

    h5py.File(corpus_dir / "100005_processed.h5", "w").close()
    empty = row(census.scan(corpus_dir, workers=2), 100005, "")
    assert bool(empty["openable"]) is True and bool(empty["present"]) is False
    assert empty["error"] == "no groups"


def test_an_unopenable_file_is_one_row_that_names_the_error(scanned):
    bad = row(scanned, CORPUS_TRUNCATED, "")
    assert bool(bad["openable"]) is False and bool(bad["present"]) is False
    assert bad["error"] and bad["kind"] == ""
    assert bad["n_channels"] == 0 and bad["n_samples"] == 0
    assert all(math.isnan(bad[c]) for c in ("t0_s", "t1_s", "fs_hz"))


# ----------------------------------------------------------------------------- what is scanned


def test_limit_takes_the_first_files_in_shot_order(corpus_dir):
    df = census.scan(corpus_dir, workers=2, limit=1)
    assert set(df["shot"]) == {CORPUS_FULL}


def test_shots_selects_those_files_and_silently_skips_the_ones_with_none(corpus_dir):
    df = census.scan(corpus_dir, workers=2, shots=[CORPUS_SMALL, CORPUS_ABSENT])
    assert set(df["shot"]) == {CORPUS_SMALL} and len(df) == 2


# --------------------------------------------------------------------------------- summary


def test_summary_is_the_presence_fraction_of_each_group_over_the_openable_files(scanned):
    s = census.summary(scanned).set_index("group")
    assert s.loc["pinj", "n_present"] == 2 and s.loc["pinj", "frac_present"] == pytest.approx(1.0)
    assert s.loc["mhr", "frac_present"] == pytest.approx(0.5)  # in one of the two openable files
    assert s.loc["co2", "n_present"] == 0 and s.loc["co2", "frac_present"] == pytest.approx(0.0)
    assert s.loc["tangtv", "kind"] == "video"
    assert "" not in s.index  # the unopenable file's row is not a group


# ------------------------------------------------------------------------------------- CLI


def test_cli_scan_writes_the_parquet_and_its_json_sidecar(corpus_dir, tmp_path, capsys):
    out = tmp_path / "corpus_coverage.parquet"
    code = cli.main(
        ["corpus", "scan", "--corpus", str(corpus_dir), "--out", str(out), "--workers", "2"]
    )
    assert code == 0 and out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 8 and {c: str(dt) for c, dt in df.dtypes.items()} == COLUMNS
    doc = json.loads((out.with_suffix(".json")).read_text())
    assert doc["n_files"] == 3 and doc["n_openable"] == 2 and doc["n_rows"] == 8
    assert doc["workers"] == 2 and doc["corpus_dir"] == str(corpus_dir)
    assert doc["written_at"] and doc["git_sha"] and doc["elapsed_s"] >= 0.0
    assert "3 files" in capsys.readouterr().out


def test_cli_summary_prints_the_availability_table(corpus_dir, tmp_path, capsys):
    out = tmp_path / "corpus_coverage.parquet"
    assert cli.main(["corpus", "scan", "--corpus", str(corpus_dir), "--out", str(out)]) == 0
    capsys.readouterr()
    assert cli.main(["corpus", "summary", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "pinj" in printed and "100.0%" in printed and "50.0%" in printed


def test_cli_scan_refuses_a_corpus_directory_that_is_not_there(tmp_path, capsys):
    assert cli.main(["corpus", "scan", "--corpus", str(tmp_path / "nope")]) == 1
    assert "nope" in capsys.readouterr().err


def test_the_worker_reads_no_sample_data(corpus_dir, monkeypatch):
    """Header-only is the whole reason a 16,909-file census is affordable: shapes and two xdata
    scalars per group, never a `ydata` read."""
    import h5py

    reads: list[str] = []
    real = h5py.Dataset.__getitem__

    def spy(self, key):
        reads.append(self.name)
        return real(self, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    census.scan_file(corpus_dir / f"{CORPUS_FULL}_processed.h5")
    assert reads and not any(name.endswith("ydata") for name in reads)


def test_every_scanned_row_is_finite_or_declared_missing(scanned):
    """No column may carry a silent zero: an absent measurement is NaN, never 0.0."""
    for _, r in scanned.iterrows():
        if not r["openable"] or not r["present"]:
            continue
        assert np.isfinite([r["t0_s"], r["t1_s"], r["fs_hz"]]).all()
