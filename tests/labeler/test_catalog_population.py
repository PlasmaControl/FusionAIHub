"""Rules 1-3 on text bundles made by hand, and rule 4 from an Ip log."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from labeler.events.catalog import population as pop
from labeler.events.catalog.check import CatalogError
from labeler.events.catalog.window import read_log

GOOD = {
    "SHOT_TYPE": "plasma",
    "IP-(MA)": "1.2",
    "PULSE-LENGTH": "5.0",
    "PBEAM-MAX-(MW)": "3.0",
    "PECH-MAX-(MW)": "0.0",
}
SPANS = {"mhr": 5.0, "ece": 5.0, "filterscopes": 5.0}


def _bundle(shot, row, *, title="Tearing mode avoidance", run_id="20220301"):
    """The parts of a `shot_<N>.txt` that `select.parse_facts` reads; `row=None` is the
    session fallback."""
    meta = json.dumps({"run_id": run_id, "title": title}, indent=1)
    if row is None:
        table = "(Shot table key/value mapping not found.)"
    else:
        table = "\n".join([f"- SHOT: {shot}"] + [f"- {k}: {v}" for k, v in row.items()])
    return "\n\n".join(
        [
            f"RUN_ID: {run_id}",
            f"METADATA (selected)\n{meta}",
            "## Shot-specific context (from summary.html)",
            f"SHOT: {shot}\n\nSHOT TABLE ROW (name -> value)\n{table}",
        ]
    )


def _pool():
    bundles = {
        1: _bundle(1, GOOD),
        2: _bundle(2, GOOD | {"PULSE-LENGTH": "1.5", "PBEAM-MAX-(MW)": "0.0"}),
        3: _bundle(3, None),
        4: _bundle(4, GOOD, title="PS test"),
        5: _bundle(5, GOOD),
        7: _bundle(
            7,
            GOOD | {"IP-(MA)": "-0.9", "PBEAM-MAX-(MW)": "0.0", "PECH-MAX-(MW)": "1.5"},
            run_id="20240110",
        ),
        # 2.1 s of pulse: the proxy's flat-top is 0.83 s, but rule 4 is measured
        8: _bundle(8, GOOD | {"PULSE-LENGTH": "2.1"}),
    }
    spans = {s: dict(SPANS) for s in bundles} | {5: SPANS | {"ece": 1.5}}
    spans[1]["co2"] = 4.25
    return pop.screen([8, 7, 6, 5, 4, 3, 2, 1], bundles, spans)


def test_the_screen_names_every_rule_a_shot_fails():
    pool = _pool()
    assert pool["shot"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert pool["reasons"].tolist() == [
        "",
        "pulse_length;heating",
        "shot_type;ip;pulse_length;heating;session_fallback",
        "title",
        "census_ece",
        "no_bundle",
        "",  # reversed Ip, heated by ECH alone
        "",
    ]
    first = pool.iloc[0]
    assert (first.year, first.run_id, first.ip_ma, first.pulse_length_s) == (
        2022,
        "20220301",
        1.2,
        5.0,
    )
    assert (first.span_co2_s, first.span_sxr_s) == (4.25, 0.0)
    assert pool.loc[6, "year"] == 2024 and pool.loc[6, "ip_ma"] == -0.9
    assert pool.loc[5, ["year", "ip_ma"]].isna().all()


def test_the_pool_round_trips(tmp_path):
    pool = _pool()
    pool.to_csv(tmp_path / "pool.csv", index=False)
    got = pop.read_pool(tmp_path / "pool.csv")
    pd.testing.assert_frame_equal(got, pool, check_dtype=False)
    assert got["reasons"].tolist()[0] == "" and str(got["year"].dtype) == "Int64"


def _log(tmp_path, lines):
    path = tmp_path / "ip.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return read_log(path)


def _ok(shot, flattop, window=(7, 5000)):
    return {
        "shot": shot,
        "status": "ok",
        "window_start_ms": window[0],
        "window_end_ms": window[1],
        "flattop_s": flattop,
        "ip_peak_ma": 1.2,
    }


def test_rule_4_comes_from_the_ip_log(tmp_path):
    log = _log(
        tmp_path,
        [
            _ok(1, 3.2),
            {"shot": 7, "status": "error", "error": "fdp could not answer"},
            _ok(8, 0.6),
            _ok(4, 3.0),  # measured in a pilot, but rule 2 already rejected it
        ],
    )
    frame = pop.population(_pool(), log)
    assert frame["reasons"].tolist() == [
        "",
        "pulse_length;heating",
        "shot_type;ip;pulse_length;heating;session_fallback",
        "title",
        "census_ece",
        "no_bundle",
        "flattop_unmeasured",
        "flattop",
    ]
    assert frame.loc[0, ["window_start_ms", "window_end_ms", "flattop_s"]].tolist() == [
        7,
        5000,
        3.2,
    ]
    assert frame.loc[3, ["ip_status", "window_start_ms"]].isna().all()
    assert str(frame["window_start_ms"].dtype) == "Int64"


def test_no_plasma_is_its_own_rejection(tmp_path):
    log = _log(tmp_path, [_ok(1, 3.2), {"shot": 7, "status": "no_plasma"}, _ok(8, 1.0)])
    frame = pop.population(_pool(), log)
    assert frame.loc[[0, 6, 7], "reasons"].tolist() == ["", "no_plasma", ""]


def test_a_screened_shot_missing_from_the_log_is_refused(tmp_path):
    log = _log(tmp_path, [_ok(1, 3.2)])
    with pytest.raises(
        CatalogError, match=r"2 shots pass rules 1-3 but have no Ip log"
    ):
        pop.population(_pool(), log)


def test_the_funnel_counts_what_each_rule_leaves(tmp_path):
    log = _log(
        tmp_path,
        [_ok(1, 3.2), {"shot": 7, "status": "error", "error": "x"}, _ok(8, 0.6)],
    )
    frame = pop.population(_pool(), log)
    assert pop.funnel(frame) == {
        "corpus_in_range": 8,
        "after_bundle": 7,
        "after_rule_1": 5,
        "after_rule_2": 4,
        "after_rule_3": 3,
        "after_rule_4": 1,
    }
    assert pop.rejections(frame) == {
        "no_bundle": 1,
        "shot_type": 1,
        "ip": 1,
        "pulse_length": 2,
        "heating": 2,
        "session_fallback": 1,
        "title": 1,
        "census_ece": 1,
        "flattop": 1,
        "flattop_unmeasured": 1,
    }


def test_the_command_writes_the_pool_under_the_root(tmp_path, monkeypatch, capsys):
    corpus, text = tmp_path / "corpus", tmp_path / "text"
    corpus.mkdir()
    text.mkdir()
    for shot in (185_600, 185_601, 185_602, 205_000):
        (corpus / f"{shot}_processed.h5").touch()
        (text / f"shot_{shot}.txt").write_text(_bundle(shot, GOOD))
    census = pd.DataFrame(
        [(s, g, True, 0.0, 5.0) for s in (185_601, 185_602) for g in SPANS]
        + [(185_602, "ece", True, 0.0, 1.0)],
        columns=["shot", "group", "present", "t0_s", "t1_s"],
    )
    census = census.drop_duplicates(["shot", "group"], keep="last")
    census.to_parquet(tmp_path / "census.parquet")
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("LABELER_CORPUS", str(corpus))
    monkeypatch.setenv("LABELER_TEXT_ROOT", str(text))
    assert pop.main(["--census", str(tmp_path / "census.parquet")]) == 0
    out = tmp_path / "root" / "catalog"
    assert pop.read_pool(out / "pool.csv")["reasons"].tolist() == ["", "census_ece"]
    assert (out / "pool_shots.txt").read_text() == "185601\n"
    meta = json.loads((out / "pool.meta.json").read_text())
    assert (
        meta["census"] == str(tmp_path / "census.parquet")
        and len(meta["census_sha256"]) == 64
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["corpus_in_range"] == 2 and summary["after_rule_3"] == 1
