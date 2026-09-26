"""Rules 1-3 on text bundles made by hand, and rule 4 from an Ip log."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from labeler.config import sha256_of
from labeler.events.catalog import population as pop
from labeler.events.catalog import window
from labeler.events.catalog.check import CatalogError
from labeler.events.catalog.window import read_log
from shot_design import config as shot_config

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
        4: _bundle(4, GOOD, title="Power Systems Testing continued."),
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
    assert str(got["has_shot_table"].dtype) == "boolean"
    assert got.loc[0, ["shot_type", "has_shot_table", "title"]].tolist() == [
        "plasma",
        True,
        "Tearing mode avoidance",
    ]
    assert not got.loc[2, "has_shot_table"]
    assert pd.isna(got.loc[5, "has_shot_table"])
    assert "mp_subject" in got and got["mp_subject"].isna().all()


def _log(tmp_path, lines):
    path = tmp_path / "ip.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return read_log(path)


def _ok(shot, flattop, bounds=(7, 5000)):
    return {
        "shot": shot,
        "status": "ok",
        "window_start_ms": bounds[0],
        "window_end_ms": bounds[1],
        "flattop_s": flattop,
        "ip_peak_ma": 1.2,
        "dt_ms": 0.05,
        "version": window.LOG_VERSION,
    }


def test_rule_4_comes_from_the_ip_log(tmp_path):
    log = _log(
        tmp_path,
        [
            _ok(1, 3.2),
            {
                "shot": 7,
                "status": "error",
                "error": "fdp could not answer",
                "dt_ms": 0.05,
                "version": window.LOG_VERSION,
            },
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
    log = _log(
        tmp_path,
        [
            _ok(1, 3.2),
            {
                "shot": 7,
                "status": "no_plasma",
                "dt_ms": 0.05,
                "version": window.LOG_VERSION,
            },
            _ok(8, 1.0),
        ],
    )
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
        [
            _ok(1, 3.2),
            {
                "shot": 7,
                "status": "error",
                "error": "x",
                "dt_ms": 0.05,
                "version": window.LOG_VERSION,
            },
            _ok(8, 0.6),
        ],
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
        "bundle_mismatch": 0,
        "shot_type": 1,
        "ip": 1,
        "pulse_length": 2,
        "heating": 2,
        "session_fallback": 1,
        "title": 1,
        "census_mhr": 0,
        "census_ece": 1,
        "census_filterscopes": 0,
        "flattop": 1,
        "no_plasma": 0,
        "flattop_unmeasured": 1,
    }


@pytest.mark.parametrize("default_census", [False, True])
def test_the_command_writes_the_pool_under_the_root(
    tmp_path, monkeypatch, capsys, default_census
):
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
    census_path = tmp_path / "corpus_coverage.parquet"
    census.to_parquet(census_path)
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("LABELER_CORPUS", str(corpus))
    monkeypatch.setenv("LABELER_TEXT_ROOT", str(text))
    monkeypatch.setattr(
        shot_config, "load_paths", lambda: SimpleNamespace(db_dir=tmp_path)
    )
    assert pop.main([] if default_census else ["--census", str(census_path)]) == 0
    out = tmp_path / "root" / "catalog"
    assert pop.read_pool(out / "pool.csv")["reasons"].tolist() == ["", "census_ece"]
    assert (out / "pool_shots.txt").read_text() == "185601\n"
    meta = json.loads((out / "pool.meta.json").read_text())
    assert meta["census"] == str(census_path) and len(meta["census_sha256"]) == 64
    summary = json.loads(capsys.readouterr().out)
    assert summary["corpus_in_range"] == 2 and summary["after_rule_3"] == 1
    inputs = pd.read_csv(out / "pool_inputs.csv")
    assert inputs["shot"].tolist() == [185601, 185602]
    for shot, digest in inputs.itertuples(index=False, name=None):
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert digest == sha256_of(text / f"shot_{shot}.txt")
    for name in ("pool", "pool_shots", "pool_inputs"):
        suffix = ".txt" if name == "pool_shots" else ".csv"
        assert meta[f"{name}_sha256"] == sha256_of(out / (name + suffix))
    lexicon = shot_config.CONFIG_DIR / "labels.yaml"
    assert meta["lexicon"] == str(lexicon)
    assert meta["lexicon_sha256"] == sha256_of(lexicon)
    assert meta["corpus_in_range"] == 2 and meta["bundles"] == 2
    assert meta["funnel"] == {
        "corpus_in_range": 2,
        "after_bundle": 2,
        "after_rule_1": 2,
        "after_rule_2": 2,
        "after_rule_3": 1,
    }
    assert meta["rejections"]["census_ece"] == 1
    assert not set(pop.RULES["rule_4"]) & meta["rejections"].keys()
    rules = meta["rules"]
    assert rules == pop.rules_record()
    assert rules["shot_range"] == [185601, 204999]
    assert rules["rule_1"]["min_abs_ip_ma"] == 0.5
    assert rules["rule_1"]["min_pulse_length_s"] == 2.0
    assert rules["rule_1"]["min_pbeam_mw"] == 1.0
    assert rules["rule_2"]["own_shot_table_block"] is True
    assert rules["rule_3"]["groups"] == ["mhr", "ece", "filterscopes"]
    assert rules["rule_3"]["min_group_span_s"] == 2.0
    assert rules["rule_4"]["max_ip_dt_ms"] == 0.5
    assert rules["rule_4"]["log_version"] == 2
    assert rules["rule_4"]["measured_on"] == "every shot passing rules 1-3"
    assert rules["dropped"]["shot_text"] == {"min_shot_chars": 0}


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("IP-(MA)", "NaN", "ip"),
        ("PULSE-LENGTH", "NaN", "pulse_length"),
        ("PBEAM-MAX-(MW)", "inf", "heating"),
        ("PECH-MAX-(MW)", "inf", "heating"),
    ],
)
def test_nonfinite_shot_table_numbers_are_missing(column, value, reason):
    row = GOOD | {column: value}
    if column == "PECH-MAX-(MW)":
        row["PBEAM-MAX-(MW)"] = "0"
    pool = pop.screen([1], {1: _bundle(1, row)}, {1: SPANS})
    assert pool.loc[0, "reasons"] == reason


def test_a_bundle_cannot_supply_another_shots_row():
    bundle = _bundle(190001, GOOD).replace("- SHOT: 190001", "- SHOT: 190002")
    pool = pop.screen([190001], {190001: bundle}, {190001: SPANS})
    assert pool.loc[0, "reasons"] == "bundle_mismatch"
    assert pool.drop(columns=["shot", "reasons"]).isna().all().all()


@pytest.mark.parametrize("bad", ["duplicate", "reason", "columns"])
def test_read_pool_refuses_invalid_tables(tmp_path, bad):
    pool = _pool()
    if bad == "duplicate":
        pool = pd.concat([pool, pool.iloc[[0]]])
        message = "duplicate.*1"
    elif bad == "reason":
        pool.loc[0, "reasons"] = "new_rejection"
        message = "1.*new_rejection"
    else:
        pool = pool.drop(columns=["run_id"])
        message = "expected the columns"
    path = tmp_path / "pool.csv"
    pool.to_csv(path, index=False)
    with pytest.raises(CatalogError, match=message):
        pop.read_pool(path)


@pytest.mark.parametrize(
    ("flat", "dt", "want"),
    [
        (None, 0.05, "flattop_unmeasured"),
        (2.0, 1.0, "flattop_unmeasured"),
        (2.0, None, "flattop_unmeasured"),
        (2.0, 0.5, ""),
        (0.99995, 0.05, "flattop"),
    ],
)
def test_rule_4_requires_a_measured_high_rate_flattop(tmp_path, flat, dt, want):
    line = _ok(1, flat)
    if dt is None:
        del line["dt_ms"]
    else:
        line["dt_ms"] = dt
    pool = pop.screen([1], {1: _bundle(1, GOOD)}, {1: SPANS})
    frame = pop.population(pool, _log(tmp_path, [line]))
    assert frame.loc[0, "reasons"] == want
    if dt is None:
        assert pd.isna(frame.loc[0, "ip_dt_ms"])
    else:
        assert frame.loc[0, "ip_dt_ms"] == dt


def test_an_unknown_ip_status_is_refused():
    log = pd.DataFrame([_ok(1, 3.2) | {"status": "pending"}, _ok(7, 2), _ok(8, 2)])
    with pytest.raises(CatalogError, match="status.*1"):
        pop.population(_pool(), log)


@pytest.mark.parametrize("old_shots", [[1], [1, 7, 8]])
def test_old_measurements_are_refused_even_in_a_mixed_log(tmp_path, old_shots):
    lines = [_ok(shot, 2.0) for shot in (1, 7, 8)]
    for line in lines:
        if line["shot"] in old_shots:
            del line["version"]
    with pytest.raises(
        CatalogError,
        match=rf"{len(old_shots)} shots.*version.*1.*catalog.window.*pool_shots.txt",
    ):
        pop.population(_pool(), _log(tmp_path, lines))


@pytest.mark.parametrize("function", [pop.funnel, pop.rejections])
def test_unknown_rejection_codes_are_refused(function):
    pool = _pool()
    pool.loc[0, "reasons"] = "new_rejection"
    with pytest.raises(CatalogError, match="1.*new_rejection"):
        function(pool)


def test_a_missing_bundle_has_a_blank_input_digest(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "185601_processed.h5").touch()
    census = tmp_path / "census.parquet"
    pd.DataFrame(columns=["shot", "group", "present", "t0_s", "t1_s"]).to_parquet(
        census
    )
    monkeypatch.setenv("LABELER_CORPUS", str(corpus))
    monkeypatch.setenv("LABELER_TEXT_ROOT", str(tmp_path / "absent-text"))
    out = tmp_path / "out"
    assert pop.main(["--census", str(census), "--out", str(out)]) == 0
    inputs = pd.read_csv(out / "pool_inputs.csv", keep_default_na=False)
    assert inputs.to_dict("records") == [{"shot": 185601, "bundle_sha256": ""}]
    meta = json.loads((out / "pool.meta.json").read_text())
    assert meta["bundles"] == 0 and meta["corpus_in_range"] == 1


@pytest.mark.parametrize(
    ("title", "subject", "want"),
    [
        (
            (
                "Test FSRM at Low Collisionality and Spectroscopic Measurements "
                "of ELMy Redeposition"
            ),
            (
                "Spectroscopic Measurements of Tungsten S/XB | Test FSRM at Low "
                "Collisionality and Spectroscopic Measurements of ELMy Redeposition"
            ),
            False,
        ),
        (
            (
                "Impurity-seeded detachment in upper-ceiling closed divertor and test "
                "detachment scaling model"
            ),
            None,
            False,
        ),
        (
            "Test the root cause of 2,1 instability of low-torque stable IBS plasmas",
            None,
            False,
        ),
        (
            (
                "Detachment dynamics and control with multi-sine system identification "
                "and IRTV heat flux control test"
            ),
            None,
            False,
        ),
        (
            "Test Off-axis Helicon Current Drive using High Beta_e Plasma coupling",
            None,
            False,
        ),
        (
            (
                "Characterization of runaway electron generation during ohmic plasma "
                "startup"
            ),
            None,
            False,
        ),
        (
            (
                "Investigation of electron-cyclotron assisted startup and the ITER "
                "first plasma scenario with ITER-like actuators (part 1)"
            ),
            (
                "Investigation of EC assisted startup and the ITER | Investigation of "
                "electron-cyclotron assisted startup and the ITER first plasma "
                "scenario with ITER-like actuators (part 1)"
            ),
            False,
        ),
        (
            "Diagnostic checkout for WPQH",
            (
                "Diagnostic checkout for QH/WPQH-Mode D3DMP No. 2022-35-51 | Divertor "
                "diagnostic checkout for QH/WPQH-Mode"
            ),
            False,
        ),
        ("Helicon Commissioning - Determine Edge Conditions for Coupling", None, False),
        ("Testing of private industry plasma-facing materials", None, False),
        (None, None, False),
        ("Power Systems Testing continued.", None, True),
        ("Startup", None, True),
        ("Startup", "Startup", True),
        ("Plasma Startup - reboot - Day 9", None, True),
        ("Plasma Starup - reboot - Day 5", None, True),
        (
            "Plasma Startup and Systems Checkout - post NT armor vent - Day 4",
            None,
            True,
        ),
        ("Plasma Startup - PostECG port blank off", "Startup", True),
        ("Plasma Startup after Helicon Vent", None, True),
        ("IWL Plasmas for diagnostic checkout", None, True),
        (
            "RF Systems Commissioning",
            "RF Systems Commissioning D3DMP No. 2024-23-59 | RF Systems Commissioning",
            True,
        ),
    ],
)
def test_machine_time_separates_physics_from_machine_activity(title, subject, want):
    assert pop.machine_time(title, subject) is want


@pytest.mark.parametrize(
    ("title", "want"),
    [
        (
            "Test the root cause of 2,1 instability of low-torque stable IBS plasmas",
            "",
        ),
        ("Power Systems Testing continued.", "title"),
    ],
)
def test_screen_rejects_machine_titles_and_keeps_physics_tests(title, want):
    pool = pop.screen([1], {1: _bundle(1, GOOD, title=title)}, {1: SPANS})
    assert pool.loc[0, "reasons"] == want


def test_the_subject_can_resolve_a_machine_title_with_supplied_themes():
    title = (
        "Investigation of electron-cyclotron assisted startup and the ITER first "
        "plasma scenario with ITER-like actuators (part 1)"
    )
    subject = "Investigation of EC assisted startup and the ITER"
    themes = [
        {"id": "startup_checkout", "keywords": ["startup"]},
        {"id": "physics", "keywords": ["ec assisted"]},
    ]
    assert pop.machine_time(title, None, themes)
    assert not pop.machine_time(title, subject, themes)


def test_the_screen_loads_themes_once_and_honours_supplied_themes(monkeypatch):
    calls = []
    themes = [{"id": "startup_checkout", "keywords": ["fixture machine"]}]

    def lexicon_themes():
        calls.append(True)
        return themes

    monkeypatch.setattr(pop.select, "lexicon_themes", lexicon_themes)
    bundles = {shot: _bundle(shot, GOOD, title="fixture machine") for shot in (1, 2)}
    spans = {1: SPANS, 2: SPANS}
    assert pop.screen([1, 2], bundles, spans)["reasons"].tolist() == ["title", "title"]
    assert calls == [True]
    pool = pop.screen([1, 2], bundles, spans, themes=[])
    assert pool["reasons"].tolist() == ["", ""]
    assert calls == [True]


def test_the_rule_record_names_the_machine_rule_and_dropped_title_regex():
    rules = pop.rules_record()
    assert rules["rule_2"] == {
        "own_shot_table_block": True,
        "machine_time": {
            "pattern": (
                r"(?i)^\s*(?:plasma\s+)?(?:start-?up|starup)\b"
                r"|^\s*power\s+systems?\s+test"
            ),
            "lexicon_theme": "startup_checkout",
            "assignment": "physics themes first, title and mini-proposal subject",
        },
    }
    assert rules["dropped"]["title"] == pop.select.TITLE_EXCLUDE.pattern
