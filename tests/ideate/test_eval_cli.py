"""`ideate eval prompts|latency|recall`: the exit codes, the flags, and what each one prints.

The exit codes carry meaning here and are asserted rather than assumed: 0 measured and within the
bar, 2 refused (there is nothing to measure), 3 measured and outside the bar. A harness that
exited 0 on a refusal would let a CI job report "eval passed" on a run that evaluated nothing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ideate import cli
from ideate.eval import prompts as ev

from .conftest import shot_record, write_db


@pytest.fixture(autouse=True)
def _no_minilm(monkeypatch):
    from ideate.shotdb import text as text_mod

    monkeypatch.setattr(
        text_mod, "embed_texts", lambda texts: np.tile([1.0, 0.0, 0.0], (len(texts), 1))
    )


@pytest.fixture
def db_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "ideate"
    (root / "db").mkdir(parents=True)
    monkeypatch.setenv("IDEATE_DATA_ROOT", str(root))
    monkeypatch.delenv("IDEATE_PATHS", raising=False)
    write_db(
        root / "db",
        [
            shot_record(300, "20210412", 1.0e6, 5.0e6, "QH-mode with an EHO at low torque"),
            shot_record(301, "20210412", 1.1e6, 5.5e6, "ELM suppression with n=3 RMP"),
            shot_record(302, "20230616", 1.2e6, 6.0e6, "sawtooth crashes on the core ECE"),
        ],
    )
    return root


@pytest.fixture
def small_evalset(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(
            ["prompt_id", "category", "prompt", "expect_phenomena", "expect_constraints",
             "expect_segment", "hand_graded", "notes"]
        )
        w.writerow(["p001", "qh_mode", "QH-mode with an EHO", "qh|eho", "", "flat_top", "0", ""])
        w.writerow(["p002", "elm_rmp", "ELM suppression", "elm", "", "flat_top", "0", ""])
    return path


# ---------------------------------------------------------------------------- eval prompts


def test_eval_prompts_prints_the_table_and_says_which_split_it_ran(
    db_root, small_evalset, capsys
):
    code = cli.main(["eval", "prompts", "--split", "all", "--evalset", str(small_evalset)])
    out = capsys.readouterr().out
    assert code == 0
    assert "split `all`" in out
    assert "coverage (prompts with >= 1 result)" in out
    assert "| channel | prompts it contributed to |" in out


def test_eval_prompts_defaults_to_the_eval_split(db_root, small_evalset, capsys):
    cli.main(["eval", "prompts", "--evalset", str(small_evalset)])
    assert "split `eval`" in capsys.readouterr().out


def test_eval_prompts_json_is_the_report_model(db_root, small_evalset, capsys):
    code = cli.main(
        ["eval", "prompts", "--split", "all", "--evalset", str(small_evalset), "--json"]
    )
    doc = json.loads(capsys.readouterr().out)
    assert code == 0
    assert doc["split"] == "all" and doc["n_prompts"] == 2
    assert doc["proxy"]["caveat"].startswith("PROXY, NOT A HUMAN GRADE")
    assert len(doc["outcomes"]) == 2


def test_eval_prompts_exits_three_when_coverage_is_under_the_bar(db_root, tmp_path, capsys):
    path = tmp_path / "impossible.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(
            ["prompt_id", "category", "prompt", "expect_phenomena", "expect_constraints",
             "expect_segment", "hand_graded", "notes"]
        )
        w.writerow(["p001", "x", "nothing is this big", "", '{"ip_mean":[9e9,null]}',
                    "flat_top", "0", ""])
    code = cli.main(["eval", "prompts", "--split", "all", "--evalset", str(path)])
    assert code == 3
    assert "FAIL" in capsys.readouterr().out


def test_eval_prompts_uses_the_frozen_evalset_when_none_is_named(db_root, capsys):
    cli.main(["eval", "prompts", "--split", "all", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["evalset"] == ev.EVALSET_NAME
    assert doc["n_prompts"] == 200


# ---------------------------------------------------------------------------- eval latency


def test_eval_latency_prints_the_budget_table(db_root, capsys):
    code = cli.main(["eval", "latency", "--repeats", "2"])
    out = capsys.readouterr().out
    assert code in (0, 3)  # a wall-clock verdict on a shared node is not asserted
    assert "| operation | what | median | p95 | budget | verdict |" in out
    assert "`phenomenon_locate`" in out


def test_eval_latency_json_carries_every_row(db_root, capsys):
    cli.main(["eval", "latency", "--repeats", "2", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert [r["name"] for r in doc["rows"]] == [
        "load", "search_text", "search_no_text", "phenomenon_locate", "describe"
    ]
    assert doc["warm"] is True


def test_eval_latency_exits_three_when_a_budgeted_row_is_over(db_root, monkeypatch, capsys):
    from ideate.eval import latency as lat_mod

    monkeypatch.setitem(lat_mod.BUDGETS_S, "load", 0.0)
    code = cli.main(["eval", "latency", "--repeats", "1"])
    assert code == 3
    assert "FAIL" in capsys.readouterr().out


# ----------------------------------------------------------------------------- eval recall


def test_eval_recall_exits_two_when_there_is_no_sheet(db_root, tmp_path, capsys):
    code = cli.main(["eval", "recall", "eho", "--labelmaker-root", str(tmp_path)])
    assert code == 2
    assert "no annotation sheet at" in capsys.readouterr().err


def test_eval_recall_refuses_an_unknown_phenomenon_with_the_registry(db_root, tmp_path, capsys):
    code = cli.main(["eval", "recall", "sawtoth", "--labelmaker-root", str(tmp_path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "unknown phenomenon" in err and "sawtooth" in err


def test_eval_recall_defaults_to_the_whole_discharge_not_the_flat_top():
    """An annotation window is indexed against the shot; clipping to the flat top would score a
    ramp-down ELM as a miss."""
    args = cli.build_parser().parse_args(["eval", "recall", "eho"])
    assert args.segment == "full"
    assert cli.build_parser().parse_args(["eval", "prompts"]).segment == "flat_top"


# ------------------------------------------------------------------------------- no database


def test_every_eval_subcommand_says_so_when_there_is_no_database(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("IDEATE_DATA_ROOT", str(tmp_path / "empty"))
    monkeypatch.delenv("IDEATE_PATHS", raising=False)
    for argv in (["eval", "prompts"], ["eval", "latency"], ["eval", "recall", "eho"]):
        assert cli.main(argv) == 1
    assert capsys.readouterr().err.count("run `ideate build` first") == 3
