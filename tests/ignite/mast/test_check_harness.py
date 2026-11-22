"""The gate command `scripts/mast/check_harness.py`: keep or re-score, the table and the exit codes.

`main()` is run with stubs: `tasks.load_task` returns the task's name, `_predictor` returns no predictor
and `harness.score` returns a made-up `ScoreResult` (and counts its calls). Nothing reads TokaMark or
its data, the scores are written to `tmp_path` (`IGNITE_MAST_ROOT`), and the real `baselines`
(`read_kept_score`, `write_score_json`, `outside_tolerance`) run unstubbed, because they are what is tested.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import baselines, harness, tasks

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "mast" / "check_harness.py"


@pytest.fixture
def gate(monkeypatch, tmp_path):
    """The command as a module, wired to stubs; ``gate.scored`` lists the (task, split) of every harness.score call.

    ``gate.nrmse`` maps (task, predictor) to the score the stub returns; the default is the published value
    (the persistence_all rows have none and get 0.9).
    """
    spec = importlib.util.spec_from_file_location("check_harness_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path / "root"))
    module.scored = []
    module.nrmse = {}
    module.current = {}

    def predictor(name, task):
        module.current["predictor"] = name
        return None, task

    def score(task, predict_fn, out_dir, *, split, batch_size, num_workers):
        predictor_name = module.current["predictor"]
        module.scored.append((task, predictor_name, split))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        published = tasks.PUBLISHED[task].get(predictor_name)
        value = module.nrmse.get((task, predictor_name), 0.9 if published is None else published)
        return harness.ScoreResult(task=task, task_nrmse=value, per_signal=pd.DataFrame(), n_windows=7,
                                   per_shot=pd.DataFrame({"shot_id": [1, 2], "NRMSE": [value, value]}),
                                   out_dir=Path(out_dir))

    monkeypatch.setattr(module, "_predictor", predictor)
    monkeypatch.setattr(tasks, "load_task", lambda name: name)
    monkeypatch.setattr(harness, "score", score)
    module.scores_dir = tmp_path / "root" / "stage0" / "scores"
    return module


ARGS = ["--tasks", "task_3-1", "task_1-1"]  # rows: task_3-1 mean, persistence, persistence_all; task_1-1 mean


def test_the_rows_are_the_mean_and_the_persistence_pair_where_one_is_published():
    rows = {(t, p): pub for t, p, pub in _load_rows()}
    assert rows[("task_3-1", "mean")] == 0.984 and rows[("task_3-1", "persistence")] == 0.536
    assert rows[("task_3-1", "persistence_all")] is None  # never judged against a published value
    assert ("task_1-1", "mean") in rows and ("task_1-1", "persistence") not in rows
    assert ("task_1-1", "persistence_all") not in rows


def _load_rows():
    spec = importlib.util.spec_from_file_location("check_harness_rows", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._rows_to_score(list(ARGS[1:]))


def test_scores_that_match_the_published_values_exit_0_and_are_written(gate, capsys):
    assert gate.main(ARGS) == 0
    out = capsys.readouterr().out
    assert len(gate.scored) == 4 and {p for _, p, _ in gate.scored} == {"mean", "persistence", "persistence_all"}
    record = json.loads((gate.scores_dir / "task_3-1" / "persistence" / "test" / "score.json").read_text())
    assert record["nrmse"] == 0.536 and record["code_sha256"] == baselines.scoring_code_sha256()
    assert "re-scoring" not in out  # a first score has no earlier one to explain
    table = out.split("\n\n", 1)[1]
    assert table.splitlines()[0].split() == ["task", "predictor", "ours", "published", "diff"]
    assert "+0.0000" in table and "OUTSIDE" not in out


def test_a_second_run_keeps_every_score_and_scores_nothing(gate, capsys):
    assert gate.main(ARGS) == 0
    gate.scored.clear()
    capsys.readouterr()
    assert gate.main(ARGS) == 0
    out = capsys.readouterr().out
    assert gate.scored == []
    assert out.count("kept ") == 4 and "re-scoring" not in out
    assert "task_3-1  persistence_all" in out  # a kept score is still in the table


def test_force_scores_again_and_says_why(gate, capsys):
    gate.main(ARGS)
    gate.scored.clear()
    capsys.readouterr()
    assert gate.main([*ARGS, "--force"]) == 0
    out = capsys.readouterr().out
    assert len(gate.scored) == 4 and out.count("re-scoring") == 4 and "kept " not in out
    assert "--force" in out


def test_a_score_of_other_scoring_code_is_made_again(gate, capsys):
    gate.main(ARGS)
    path = gate.scores_dir / "task_3-1" / "mean" / "test" / "score.json"
    stale = json.loads(path.read_text())
    stale["code_sha256"] = "0" * 64
    path.write_text(json.dumps(stale))
    gate.scored.clear()
    capsys.readouterr()
    assert gate.main(ARGS) == 0
    out = capsys.readouterr().out
    assert gate.scored == [("task_3-1", "mean", "test")]
    assert "re-scoring task_3-1 mean test: scoring code changed since" in out
    assert json.loads(path.read_text())["code_sha256"] == baselines.scoring_code_sha256()


def test_a_test_score_off_the_published_value_exits_1_and_names_the_row(gate, capsys):
    gate.nrmse[("task_3-1", "persistence")] = 0.536 + 0.01
    assert gate.main(ARGS) == 1
    out = capsys.readouterr().out
    assert "OUTSIDE 0.002: task_3-1 persistence" in out and out.count("OUTSIDE") == 1


def test_a_test_score_inside_the_tolerance_exits_0(gate):
    gate.nrmse[("task_3-1", "persistence")] = 0.536 + 0.0019
    assert gate.main(ARGS) == 0


def test_val_report_judges_no_difference_and_exits_0(gate, capsys):
    """The published values are test-split values: far from them on other shots is not a failure."""
    gate.nrmse[("task_3-1", "mean")] = 5.0
    assert gate.main([*ARGS, "--split", "val_report"]) == 0
    out = capsys.readouterr().out
    table = out.split("\n\n")[1].splitlines()
    assert len(table) == 1 + 4 and all(line.split()[-1] == "n/a" for line in table[1:])
    assert "OUTSIDE" not in out and "no diff" in out
    assert (gate.scores_dir / "task_3-1" / "mean" / "val_report" / "score.json").exists()
    assert not (gate.scores_dir / "task_3-1" / "mean" / "test").exists()


@pytest.mark.parametrize("split", ["test", "val_report"])
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_score_that_is_not_finite_exits_1_on_every_split(gate, capsys, split, bad):
    gate.nrmse[("task_1-1", "mean")] = bad
    assert gate.main([*ARGS, "--split", split]) == 1
    assert "OUTSIDE 0.002: task_1-1 mean" in capsys.readouterr().out


def test_a_kept_score_of_the_other_split_is_not_used(gate, capsys):
    gate.main(ARGS)  # test split
    gate.scored.clear()
    capsys.readouterr()
    gate.main([*ARGS, "--split", "val_report"])
    assert len(gate.scored) == 4 and "kept " not in capsys.readouterr().out


def test_the_default_is_all_fourteen_tasks_and_every_row_is_scored(gate):
    assert gate.main([]) == 0
    persistence_tasks = [t for t in tasks.ALL_TASKS if tasks.PUBLISHED[t]["persistence"] is not None]
    assert len(persistence_tasks) == 6
    assert len(gate.scored) == 14 + 2 * len(persistence_tasks)


def test_an_unknown_task_is_refused_before_anything_is_scored(gate):
    with pytest.raises(SystemExit):
        gate.main(["--tasks", "task_9-9"])
    assert gate.scored == []
