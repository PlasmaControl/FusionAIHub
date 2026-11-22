"""The headroom table: reads the stored scores, trains and scores nothing."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import baselines, tasks
from tokamak_foundation_model.ignite.mast.headroom import FITTED, table

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "mast" / "run_headroom.py"
REPL = {t: 0.2 for t in tasks.STAGE0_TASKS}


def _run_headroom():
    spec = importlib.util.spec_from_file_location("run_headroom_table_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_score(scores, task, predictor, split, per_shot, *, code=None, **extra):
    """A stored score: `score.json` and the evaluator's `shots_metrics.csv`, as the harness writes them.

    `per_shot` is the list of the shots' task scores; each shot gets two signal rows whose mean it is.
    """
    out_dir = scores / task / predictor / split
    (out_dir / task).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, value in enumerate(per_shot):
        for j, delta in enumerate((-0.01, 0.01)):
            rows.append({"shot_id": 1000 + i, "feature_name": f"sig{j}", "n_windows": 10,
                         "NRMSE_mean": value + delta})
    pd.DataFrame(rows).to_csv(out_dir / task / "shots_metrics.csv", index=False)
    record = {"task": task, "predictor": predictor, "split": split, "nrmse": float(np.mean(per_shot)),
              "n_shots": len(per_shot), "n_windows": 10 * len(per_shot), "git_revision": "abc1234",
              "date": "2026-11-22T00:00:00+00:00",
              "code_sha256": baselines.scoring_code_sha256() if code is None else code, **extra}
    (out_dir / "score.json").write_text(json.dumps(record))
    return out_dir


def write_log(stage0, task, variant, val_nrmse, config=None):
    run = stage0 / "direct" / task / variant
    run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"step": range(len(val_nrmse)), "train_loss": 1.0, "val_nrmse": val_nrmse}).to_csv(
        run / "train_log.csv", index=False)
    (run / "config.json").write_text(json.dumps({"task": task, "cfg": config or {"depth": 8, "max_steps": 100}}))


def test_bootstrap_ci_brackets_the_mean():
    x = np.random.default_rng(0).normal(0.2, 0.05, 500)
    lo, hi = table.bootstrap_ci(x)
    assert lo < x.mean() < hi and 0.003 < (hi - lo) / 2 < 0.006 and table.bootstrap_ci(x) == (lo, hi)


def test_verdict():
    assert table.verdict(0.08, 0.075, 0.085, 0.0941) == "ahead"
    assert table.verdict(0.09, 0.085, 0.095, 0.0941) == "tie"
    assert table.verdict(0.11, 0.10, 0.12, 0.0941) == "behind"


def test_fitted_predictors_are_the_ones_the_command_refuses_on_test():
    assert table.FITTED is FITTED and FITTED == ("shrunk", "ridge", "direct")


def test_projected_mean_uses_best_published_elsewhere():
    t = pd.DataFrame({"direct_val": [tasks.best_published(k) for k in tasks.STAGE0_TASKS]},
                     index=pd.Index(tasks.STAGE0_TASKS, name="task"))
    expected = np.mean([tasks.best_published(k) for k in tasks.ALL_TASKS])
    assert abs(table.projected_mean(t) - expected) < 1e-4 and abs(expected - 0.194) < 5e-4
    t.loc["task_4-5", "direct_val"] = t.loc["task_4-5", "direct_val"] - 0.14  # 0.14 / 14 = 0.01
    assert abs(table.projected_mean(t) - (expected - 0.01)) < 1e-9


def test_projected_mean_is_nan_while_a_stage0_task_has_no_direct_score():
    t = pd.DataFrame({"direct_val": [0.1] * 7}, index=pd.Index(tasks.STAGE0_TASKS, name="task"))
    t.loc["task_3-1", "direct_val"] = np.nan
    assert math.isnan(table.projected_mean(t))


def test_build_table_reads_score_files(tmp_path):
    scores = tmp_path / "stage0" / "scores"
    write_score(scores, "task_1-2", "mean", "test", [0.99, 1.01])
    write_score(scores, "task_1-2", "mean", "val_report", [0.5, 0.7])
    write_score(scores, "task_1-2", "floor", "test", [0.02, 0.03])
    write_score(scores, "task_1-2", "shrunk", "val_report", [0.2, 0.3])
    write_score(scores, "task_3-1", "mean", "val_report", [0.9])
    t = table.build_table(scores, {**REPL, "task_3-1": None})
    assert list(t.index) == list(tasks.STAGE0_TASKS) and t.index.name == "task"
    assert t.loc["task_1-2", "mean_test"] == pytest.approx(1.0)
    assert t.loc["task_1-2", "mean_val"] == pytest.approx(0.6)
    assert t.loc["task_1-2", "floor_test"] == pytest.approx(0.025)
    assert t.loc["task_1-2", "shrunk_val"] == pytest.approx(0.25)
    assert t.loc["task_3-1", "mean_val"] == pytest.approx(0.9)
    # a cell nobody scored is NaN, not 0, and so is a replication that is not there
    for cell in (("task_1-2", "floor_val"), ("task_1-2", "ridge_val"), ("task_1-2", "direct_val"),
                 ("task_3-1", "mean_test"), ("task_3-1", "tokamind_repl"), ("task_4-5", "mean_val")):
        assert math.isnan(t.loc[cell])
    assert t.loc["task_1-2", "tokamind_repl"] == 0.2
    assert t.loc["task_1-2", "best_published"] == tasks.best_published("task_1-2")
    assert t.loc["task_1-2", "best_published_by"] == "gentoka"
    assert t.loc["task_4-1", "best_published_by"] == "tokamind_rerun"
    assert math.isnan(t.loc["task_1-2", "direct_over_best"])
    assert t.loc["task_1-2", "verdict"] == "missing"
    assert (t["stale"] == "").all() and (t["unfinished"] == "").all()


def test_the_direct_cell_has_an_interval_a_ratio_and_a_verdict(tmp_path):
    stage0 = tmp_path / "stage0"
    per_shot = np.random.default_rng(1).normal(0.03, 0.005, 400)
    write_score(stage0 / "scores", "task_1-2", "direct", "val_report", per_shot, trained_steps=100, max_steps=100)
    write_log(stage0, "task_1-2", "base", [0.05, 0.03])
    t = table.build_table(stage0 / "scores", REPL)
    row = t.loc["task_1-2"]
    best = tasks.best_published("task_1-2")
    assert row["direct_val"] == pytest.approx(per_shot.mean())
    assert row["direct_lo"] < row["direct_val"] < row["direct_hi"]
    assert (row["direct_lo"], row["direct_hi"]) == table.bootstrap_ci(per_shot)
    assert row["direct_over_best"] == pytest.approx(per_shot.mean() / best)
    assert row["verdict"] == "ahead" and row["unfinished"] == "" and row["not_competitive"] == ""


def test_a_direct_score_worse_than_the_replication_is_marked_not_competitive(tmp_path):
    stage0 = tmp_path / "stage0"
    done = {"trained_steps": 10, "max_steps": 10}
    write_score(stage0 / "scores", "task_1-2", "direct", "val_report", [0.05] * 5, **done)   # replication 0.0433
    write_score(stage0 / "scores", "task_3-1", "direct", "val_report", [0.30] * 5, **done)   # replication 0.3457
    t = table.build_table(stage0 / "scores", table.TOKAMIND_REPLICATION)
    assert t.loc["task_1-2", "not_competitive"] == "yes" and t.loc["task_3-1", "not_competitive"] == ""
    assert t.loc["task_4-5", "not_competitive"] == ""  # no direct score: nothing to say
    assert "behind; not competitive" in table.to_markdown(t)


def test_a_stored_score_with_other_scoring_code_is_shown_and_named(tmp_path):
    scores = tmp_path / "stage0" / "scores"
    write_score(scores, "task_1-2", "mean", "test", [1.0], code="0" * 64)
    write_score(scores, "task_1-2", "mean", "val_report", [0.6])
    t = table.build_table(scores, REPL)
    assert t.loc["task_1-2", "mean_test"] == pytest.approx(1.0)  # shown, not dropped
    assert t.loc["task_1-2", "mean_val"] == pytest.approx(0.6)
    assert t.loc["task_1-2", "stale"].startswith("mean_test: scoring code changed since abc1234")
    assert "mean_val" not in t.loc["task_1-2", "stale"]
    assert (t.drop(index="task_1-2")["stale"] == "").all()


def test_the_command_decides_what_is_stale(tmp_path):
    scores = tmp_path / "stage0" / "scores"
    write_score(scores, "task_1-2", "shrunk", "val_report", [0.2])
    write_score(scores, "task_1-2", "ridge", "val_report", [0.2])
    write_score(scores, "task_1-2", "mean", "test", [1.0], code="0" * 64)
    asked = []

    def stale(task, predictor, split):
        asked.append((task, predictor, split))
        return "fit_sha256 changed since abc1234" if predictor == "ridge" else ""

    t = table.build_table(scores, REPL, stale_reason=stale)
    # only stored records are asked about, once each, and the function replaces the scoring-code rule
    assert sorted(asked) == [("task_1-2", "mean", "test"), ("task_1-2", "ridge", "val_report"),
                             ("task_1-2", "shrunk", "val_report")]
    assert t.loc["task_1-2", "stale"] == "ridge_val: fit_sha256 changed since abc1234"
    assert t.loc["task_1-2", "ridge_val"] == pytest.approx(0.2)


def test_a_direct_score_of_an_unfinished_training_counts_and_is_marked(tmp_path):
    stage0 = tmp_path / "stage0"
    write_score(stage0 / "scores", "task_4-5", "direct", "val_report", [0.6, 0.7], trained_steps=12000,
                max_steps=30000)
    write_score(stage0 / "scores", "task_4-4", "direct", "val_report", [0.3], max_steps=30000)  # no trained_steps
    write_score(stage0 / "scores", "task_4-1", "direct", "val_report", [0.2], trained_steps=None, max_steps=30000)
    write_score(stage0 / "scores", "task_3-1", "direct", "val_report", [0.3], trained_steps=30000, max_steps=30000)
    t = table.build_table(stage0 / "scores", REPL)
    assert t.loc["task_4-5", "direct_val"] == pytest.approx(0.65)
    assert t.loc["task_4-5", "unfinished"] == "direct_val: trained 12000 of 30000 steps"
    assert t.loc["task_4-4", "unfinished"] == "direct_val: trained steps not recorded"
    assert t.loc["task_4-1", "unfinished"] == "direct_val: trained steps not recorded"
    assert t.loc["task_3-1", "unfinished"] == ""
    filled = pd.Series(0.2, index=t.index)
    t["direct_val"] = t["direct_val"].fillna(filled)
    assert not math.isnan(table.projected_mean(t))  # an unfinished score counts


def test_the_direct_cell_is_the_variant_with_the_lowest_selection_error(tmp_path):
    stage0 = tmp_path / "stage0"
    scores = stage0 / "scores"
    done = {"trained_steps": 100, "max_steps": 100}
    write_score(scores, "task_3-1", "direct", "val_report", [0.40], **done)
    write_score(scores, "task_3-1", "direct.wide", "val_report", [0.45], **done)  # lower on val_select
    write_score(scores, "task_3-1", "direct.deep", "val_report", [0.30], **done)  # no train log: ranked last
    write_log(stage0, "task_3-1", "base", [0.9, 0.52, 0.55], {"depth": 8, "d_model": 384, "max_steps": 100})
    write_log(stage0, "task_3-1", "wide", [0.9, 0.50, 0.60], {"depth": 8, "d_model": 768, "max_steps": 100})
    t = table.build_table(scores, REPL)
    assert t.loc["task_3-1", "direct_variant"] == "wide"
    assert t.loc["task_3-1", "direct_val"] == pytest.approx(0.45)
    variants = table.direct_variants(scores, "task_3-1")
    assert list(variants["variant"]) == ["base", "deep", "wide"]
    by = variants.set_index("variant")
    assert by.loc["base", "best_val_nrmse"] == pytest.approx(0.52)
    assert by.loc["wide", "best_val_nrmse"] == pytest.approx(0.50)
    assert math.isnan(by.loc["deep", "best_val_nrmse"])
    assert by.loc["wide", "val_report"] == pytest.approx(0.45)
    assert by.loc["wide", "differs"] == "d_model=768"
    assert by.loc["base", "differs"] == ""


def test_a_per_shot_file_that_disagrees_with_the_score_gives_no_interval(tmp_path):
    scores = tmp_path / "stage0" / "scores"
    out_dir = write_score(scores, "task_1-2", "direct", "val_report", [0.03, 0.04], trained_steps=1, max_steps=1)
    record = json.loads((out_dir / "score.json").read_text())
    record["nrmse"] = 0.5
    (out_dir / "score.json").write_text(json.dumps(record))
    t = table.build_table(scores, REPL)
    assert t.loc["task_1-2", "direct_val"] == 0.5 and math.isnan(t.loc["task_1-2", "direct_lo"])
    assert t.loc["task_1-2", "verdict"] == "no interval"
    assert "per-shot file" in t.loc["task_1-2", "stale"]


def test_markdown_marks_missing_stale_and_unfinished_cells(tmp_path):
    scores = tmp_path / "stage0" / "scores"
    write_score(scores, "task_1-2", "mean", "test", [1.0], code="0" * 64)
    write_score(scores, "task_4-5", "direct", "val_report", [0.6], trained_steps=5, max_steps=10)
    text = table.to_markdown(table.build_table(scores, REPL))
    def rows(label):  # the references table first, the fitted-and-direct table second
        return [line for line in text.splitlines() if line.startswith(f"| {label} ")]

    assert "1.0000 †" in rows("1-2")[0] and "missing" in rows("1-2")[0]
    assert "0.6000 ‡" in rows("4-5")[1]
    row11 = rows("1-1")[0]
    assert "undefined" in row11  # the benchmark defines no persistence for it
    assert "† stale" in text
    assert "0.0000" not in text  # a missing cell is never printed as a score


def test_the_command_has_one_rule_for_a_current_score(tmp_path, monkeypatch):
    run_headroom = _run_headroom()
    stage0 = tmp_path / "stage0"
    write_score(stage0 / "scores", "task_1-2", "mean", "test", [1.0])
    write_score(stage0 / "scores", "task_1-2", "persistence", "test", [1.0], code="0" * 64)
    write_score(stage0 / "scores", "task_1-2", "floor", "val_report", [0.02], predictor_sha256="x",
                tokamind_representation_sha256="y")
    assert run_headroom.stale_reason(stage0, "task_1-2", "mean", "test") == ""
    assert run_headroom.stale_reason(stage0, "task_1-2", "persistence", "test").startswith(
        "scoring code changed since abc1234")
    # a headroom predictor is checked with the same `_kept_score` that decides whether the command scores again
    monkeypatch.setattr(run_headroom.floor, "PREDICTOR_SHA256", "x")
    monkeypatch.setattr(run_headroom.floor, "tokamind_output_indices", lambda name: {})
    monkeypatch.setattr(run_headroom.floor, "tokamind_nan_imputation", lambda name: "interpolate")
    monkeypatch.setattr(run_headroom.floor, "representation_sha256", lambda *a: "y")
    assert run_headroom.stale_reason(stage0, "task_1-2", "floor", "val_report") == ""
    monkeypatch.setattr(run_headroom.floor, "representation_sha256", lambda *a: "z")
    assert run_headroom.stale_reason(stage0, "task_1-2", "floor", "val_report") == \
        "tokamind_representation_sha256 changed since abc1234"


def test_a_missing_checkpoint_or_fit_is_a_stale_reason_not_a_crash(tmp_path):
    pytest.importorskip("tokamark")  # the fit's metadata names TokaMark's training shots
    run_headroom = _run_headroom()
    stage0 = tmp_path / "stage0"
    write_score(stage0 / "scores", "task_1-2", "direct", "val_report", [0.03], predictor_sha256="x",
                checkpoint_sha256="y", trained_steps=1, max_steps=1)
    write_score(stage0 / "scores", "task_1-2", "shrunk", "val_report", [0.2], predictor_sha256="x",
                fit_sha256="y")
    reason = run_headroom.stale_reason(stage0, "task_1-2", "direct", "val_report")
    assert "provenance cannot be computed" in reason and "best.pt" in reason
    assert run_headroom.stale_reason(stage0, "task_1-2", "shrunk", "val_report") != ""
    assert "changed since" in run_headroom.stale_reason(stage0, "task_1-2", "shrunk", "val_report")


def test_the_table_command_writes_the_csv_and_prints_the_projection(tmp_path, monkeypatch, capsys):
    run_headroom = _run_headroom()
    stage0 = tmp_path / "stage0"
    write_score(stage0 / "scores", "task_1-2", "mean", "test", [1.0])
    monkeypatch.setattr(run_headroom.paths, "stage0_dir", lambda: stage0)
    monkeypatch.setattr(run_headroom.paths, "prepare_write_root", lambda p: Path(p))
    assert run_headroom.main(["--table"]) == 0
    out = capsys.readouterr().out
    assert "projected mean" in out and "7 of 7" in out and "| task " in out
    csv = pd.read_csv(stage0 / "headroom_table.csv", index_col="task")
    assert csv.loc["task_1-2", "mean_test"] == pytest.approx(1.0) and math.isnan(csv.loc["task_1-2", "mean_val"])
    assert sorted(p.name for p in stage0.iterdir()) == ["headroom_table.csv", "scores"]


def test_the_table_command_needs_no_task(capsys):
    run_headroom = _run_headroom()
    with pytest.raises(SystemExit) as e:
        run_headroom.main([])
    assert e.value.code == 2
    assert "--task" in capsys.readouterr().err


def test_the_table_lists_every_trained_variant_including_one_without_a_score(tmp_path):
    stage0 = tmp_path / "stage0"
    scores = stage0 / "scores"
    done = {"trained_steps": 100, "max_steps": 100}
    write_score(scores, "task_3-1", "direct", "val_report", [0.40], **done)
    write_log(stage0, "task_3-1", "base", [0.9, 0.52], {"depth": 8, "d_model": 384, "max_steps": 100})
    write_log(stage0, "task_3-1", "wide", [0.9, 0.45], {"depth": 8, "d_model": 768, "max_steps": 100})  # no score yet
    text = table.variants_markdown(scores)
    assert "3-1" in text and "d_model=768" in text and "0.4500" in text and "0.5200" in text
    wide = next(line for line in text.splitlines() if line.startswith("| wide "))
    base = next(line for line in text.splitlines() if line.startswith("| base "))
    assert "no score" in wide and "no" in wide.split("|")[-2]      # lower selection error, but not in the table
    assert "0.4000" in base and "yes" in base.split("|")[-2]       # the variant the table takes
    assert "1-2" not in text                                       # tasks without a variant are not listed


def test_the_table_command_prints_the_variants(tmp_path, monkeypatch, capsys):
    run_headroom = _run_headroom()
    stage0 = tmp_path / "stage0"
    write_score(stage0 / "scores", "task_1-2", "mean", "test", [1.0])
    write_log(stage0, "task_3-1", "base", [0.9, 0.5])
    monkeypatch.setattr(run_headroom.paths, "stage0_dir", lambda: stage0)
    monkeypatch.setattr(run_headroom.paths, "prepare_write_root", lambda p: Path(p))
    assert run_headroom.main(["--table"]) == 0
    out = capsys.readouterr().out
    assert "Direct variants" in out and "| base " in out and "no score" in out


# ------------------------------------------------------------------ final fix wave: C2

def _variants_stage0(tmp_path):
    """task_1-2 with the base run (finished), `s90k` (stopped at 30,000 of 90,000: the case that exists) and `old`
    (a finished run whose score is stale)."""
    stage0 = tmp_path / "stage0"
    scores = stage0 / "scores"
    write_score(scores, "task_1-2", "direct", "val_report", [0.040], trained_steps=30000, max_steps=30000)
    write_score(scores, "task_1-2", "direct.s90k", "val_report", [0.038], trained_steps=30000, max_steps=90000)
    write_score(scores, "task_1-2", "direct.old", "val_report", [0.045], trained_steps=100, max_steps=100)
    write_log(stage0, "task_1-2", "base", [0.9, 0.050], {"depth": 8, "max_steps": 30000})
    write_log(stage0, "task_1-2", "s90k", [0.9, 0.049], {"depth": 8, "max_steps": 90000})
    write_log(stage0, "task_1-2", "old", [0.9, 0.060], {"depth": 8, "max_steps": 100})
    return stage0, scores


def _old_is_stale(task, predictor, split):
    return "trained by other code than the code now loaded" if predictor == "direct.old" else ""


def test_the_variants_have_the_stale_and_unfinished_marks_of_the_table(tmp_path):
    stage0, scores = _variants_stage0(tmp_path)
    by = table.direct_variants(scores, "task_1-2", stale_reason=_old_is_stale).set_index("variant")
    assert by.loc["s90k", "unfinished"] == "trained 30000 of 90000 steps" and by.loc["s90k", "stale"] == ""
    assert by.loc["old", "stale"] == "trained by other code than the code now loaded"
    assert by.loc["old", "unfinished"] == "" and by.loc["base", "stale"] == "" == by.loc["base", "unfinished"]
    # without a callback the scoring-code rule decides, as in `build_table`
    plain = table.direct_variants(scores, "task_1-2").set_index("variant")
    assert (plain["stale"] == "").all()
    write_score(scores, "task_1-2", "direct.old", "val_report", [0.045], code="0" * 64, trained_steps=1, max_steps=1)
    assert table.direct_variants(scores, "task_1-2").set_index("variant").loc["old", "stale"].startswith(
        "scoring code changed since")


def test_a_variant_without_a_score_or_with_an_unreadable_one_is_marked_correctly(tmp_path):
    stage0, scores = _variants_stage0(tmp_path)
    (scores / "task_1-2" / "direct.old" / "val_report" / "score.json").write_text("{not json")
    write_log(stage0, "task_1-2", "fresh", [0.9, 0.5], {"depth": 8, "max_steps": 100})        # no score at all
    by = table.direct_variants(scores, "task_1-2").set_index("variant")
    assert by.loc["old", "stale"] == "unreadable score.json" and math.isnan(by.loc["old", "val_report"])
    assert by.loc["fresh", "stale"] == "" == by.loc["fresh", "unfinished"] and math.isnan(by.loc["fresh", "val_report"])


def test_the_variants_listing_prints_the_marks_and_their_reasons(tmp_path):
    stage0, scores = _variants_stage0(tmp_path)
    text = table.variants_markdown(scores, stale_reason=_old_is_stale)
    rows = {line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("| ") and "variant" not in line}
    assert "0.0380 ‡" in rows["s90k"] and "0.0450 †" in rows["old"] and "0.0400" in rows["base"]
    assert "†" not in rows["base"] and "‡" not in rows["base"]
    assert "- 1-2 s90k val_report: trained 30000 of 90000 steps (unfinished)" in text
    assert "- 1-2 old val_report: trained by other code than the code now loaded (stale)" in text
    assert "† stale" in text and "‡ unfinished" in text                       # the marks are explained where they occur


def test_a_listing_without_marks_has_no_legend(tmp_path):
    stage0 = tmp_path / "stage0"
    write_score(stage0 / "scores", "task_3-1", "direct", "val_report", [0.4], trained_steps=10, max_steps=10)
    write_log(stage0, "task_3-1", "base", [0.9, 0.5])
    text = table.variants_markdown(stage0 / "scores")
    assert "†" not in text and "‡" not in text and "stale" not in text


def test_the_legend_states_every_reason_for_a_mark_and_the_limit_of_the_rule(tmp_path):
    scores = tmp_path / "stage0" / "scores"
    write_score(scores, "task_4-5", "direct", "val_report", [0.6], trained_steps=5, max_steps=10)
    text = table.to_markdown(table.build_table(scores, REPL))
    legend = " ".join(line for line in text.splitlines() if line.startswith("missing:"))
    for reason in ("scoring code", "predictor's code", "fit", "checkpoint", "TokaMind", "unreadable", "per-shot",
                   "cannot be computed", "trained by other code than the code now loaded",
                   "training code not recorded"):
        assert reason in legend, reason
    assert "TokaMark" in legend and "Zarr" in legend and "not marked" in legend     # the limit of the rule (A-I1)


def test_the_names_the_command_shares_are_the_tables():
    rh = _run_headroom()
    assert rh.BASE_VARIANT is table.BASE_VARIANT and rh.predictor_name is table.predictor_name
    assert rh.direct_run_dir is table.direct_run_dir
    assert table.predictor_name("direct", "base") == "direct" and table.predictor_name("direct", "wide") == "direct.wide"
    assert table.direct_run_dir(Path("/x"), "task_1-2", "wide") == Path("/x/direct/task_1-2/wide")


# ---- the table command

def _table_stage0(tmp_path, monkeypatch, stale=None):
    """All seven stage-0 tasks with a finished, competitive, current direct score. Returns (module, stage0)."""
    rh = _run_headroom()
    stage0 = tmp_path / "stage0"
    for task in tasks.STAGE0_TASKS:
        write_score(stage0 / "scores", task, "direct", "val_report", [table.TOKAMIND_REPLICATION[task] - 0.01],
                    trained_steps=10, max_steps=10)
    monkeypatch.setattr(rh.paths, "stage0_dir", lambda: stage0)
    monkeypatch.setattr(rh.paths, "prepare_write_root", lambda p: Path(p))
    monkeypatch.setattr(rh, "stale_reason", lambda root, task, name, split: (stale or {}).get((task, name), ""))
    return rh, stage0


def _rewrite(stage0, task, name, **changes):
    path = stage0 / "scores" / task / name / "val_report" / "score.json"
    record = json.loads(path.read_text())
    record.update(changes)
    path.write_text(json.dumps(record))


def test_the_projected_mean_line_counts_rows_whose_direct_score_is_marked(tmp_path, monkeypatch, capsys):
    stale = {("task_3-1", "direct"): "checkpoint_sha256 changed since abc1234",     # stale and unfinished: one row
             ("task_1-1", "direct"): "trained by other code than the code now loaded",
             ("task_1-2", "mean"): "scoring code changed since abc1234"}            # not the direct score: not counted
    rh, stage0 = _table_stage0(tmp_path, monkeypatch, stale)
    write_score(stage0 / "scores", "task_1-2", "mean", "test", [1.0])
    _rewrite(stage0, "task_3-1", "direct", trained_steps=5)                          # unfinished as well
    write_score(stage0 / "scores", "task_4-5", "direct", "val_report", [0.9], trained_steps=10, max_steps=10)  # worse
    assert rh.main(["--table"]) == 0
    out = capsys.readouterr().out
    assert "projected mean over the 14 tasks" in out
    line = next(line for line in out.splitlines() if line.startswith("projected mean"))
    assert "3 row(s)" in line                                                       # 1-1, 3-1 (once), 4-5
    assert "stale, unfinished or not competitive" in line


def test_the_projected_mean_line_has_no_count_when_nothing_is_marked(tmp_path, monkeypatch, capsys):
    rh, stage0 = _table_stage0(tmp_path, monkeypatch)
    assert rh.main(["--table"]) == 0
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("projected mean"))
    assert "row(s)" not in line and "the bar is 0.183" in line


def test_the_variants_listing_of_the_command_carries_the_marks(tmp_path, monkeypatch, capsys):
    rh, stage0 = _table_stage0(tmp_path, monkeypatch, {("task_1-2", "direct.old"): "checkpoint_sha256 changed"})
    write_score(stage0 / "scores", "task_1-2", "direct.old", "val_report", [0.05], trained_steps=1, max_steps=2)
    write_log(stage0, "task_1-2", "base", [0.9, 0.5])
    write_log(stage0, "task_1-2", "old", [0.9, 0.6])
    assert rh.main(["--table"]) == 0
    out = capsys.readouterr().out
    assert "- 1-2 old val_report: checkpoint_sha256 changed (stale)" in out
    assert "- 1-2 old val_report: trained 1 of 2 steps (unfinished)" in out


@pytest.mark.parametrize("option", [["--force"], ["--train"], ["--gpu", "0"], ["--variant", "wide"], ["--share-gpu"],
                                    ["--num-workers", "2"], ["--batch-size", "8"], ["--max-steps", "5"],
                                    ["--variant", "base"], ["--num-workers", "64"]])
def test_the_table_command_refuses_every_other_option(tmp_path, monkeypatch, capsys, option):
    rh, stage0 = _table_stage0(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        rh.main(["--table", *option])
    assert e.value.code == 2
    assert option[0] in capsys.readouterr().err and not (stage0 / "headroom_table.csv").exists()


def test_a_provenance_error_of_any_kind_marks_the_cell_and_does_not_stop_the_table(tmp_path, monkeypatch):
    rh = _run_headroom()
    stage0 = tmp_path / "stage0"
    write_score(stage0 / "scores", "task_1-2", "direct", "val_report", [0.03], trained_steps=1, max_steps=1)

    def broken(predictor, ctx):
        raise RuntimeError("the fit file is a directory")

    monkeypatch.setattr(rh, "provenance_of", broken)
    reason = rh.stale_reason(stage0, "task_1-2", "direct", "val_report")
    assert reason.startswith("provenance cannot be computed now (RuntimeError: the fit file is a directory)")


def test_the_table_can_cover_all_fourteen_tasks(tmp_path):
    stage0 = tmp_path / "stage0"
    write_score(stage0 / "scores", "task_2-3", "mean", "val_report", [0.4, 0.6])
    write_score(stage0 / "scores", "task_2-3", "direct", "val_report", [0.07, 0.09], trained_steps=100, max_steps=100)
    write_log(stage0, "task_2-3", "base", [0.09, 0.08])
    t = table.build_table(stage0 / "scores", table.TOKAMIND_REPLICATION, task_names=tasks.ALL_TASKS)
    assert list(t.index) == list(tasks.ALL_TASKS) and t.index.name == "task"
    assert t.loc["task_2-3", "mean_val"] == pytest.approx(0.5)
    assert t.loc["task_2-3", "direct_val"] == pytest.approx(0.08)
    assert t.loc["task_2-3", "tokamind_repl"] == 0.0917
    assert t.loc["task_2-3", "best_published"] == tasks.best_published("task_2-3")
    for task in ("task_1-3", "task_4-2"):        # nothing stored: NaN cells and a verdict that says so
        assert math.isnan(t.loc[task, "direct_val"]) and math.isnan(t.loc[task, "mean_val"])
        assert t.loc[task, "verdict"] == "missing"


def test_the_default_table_is_still_the_seven_stage0_tasks(tmp_path):
    t = table.build_table(tmp_path / "scores", REPL)
    assert list(t.index) == list(tasks.STAGE0_TASKS)


def test_the_projection_of_a_table_is_over_its_own_rows():
    t = pd.DataFrame({"direct_val": [0.1] * 14}, index=pd.Index(tasks.ALL_TASKS, name="task"))
    assert table.projected_mean(t) == pytest.approx(0.1)          # 14 measured rows, no published score used
    t.loc["task_3-2", "direct_val"] = np.nan
    assert math.isnan(table.projected_mean(t))                    # not a silent fall-back to the published score


def test_the_variants_listing_takes_the_task_list(tmp_path):
    stage0 = tmp_path / "stage0"
    write_score(stage0 / "scores", "task_2-3", "direct", "val_report", [0.08], trained_steps=100, max_steps=100)
    write_log(stage0, "task_2-3", "base", [0.09, 0.08])
    assert table.variants_markdown(stage0 / "scores") == ""
    assert "Task 2-3:" in table.variants_markdown(stage0 / "scores", task_names=tasks.ALL_TASKS)
