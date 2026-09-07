"""`ideate logs missing|import`: the d3dlogfetching contract.

The text corpus is the output of a private tool (PlasmaControl/d3dlogfetching) that only runs
inside the GA network, so this side of the boundary can do exactly two things: say precisely what
has to be run over there, and re-compose a synced `runs/` tree here without changing a byte of
what the tool would have written. Both are what this file pins.

The byte-compatibility test is the load-bearing one, and it is a real-data test: a composition
that is merely "close" produces bundles that parse differently from their 22,950 neighbours, and
no synthetic fixture can catch that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ideate import cli, config
from ideate.shotdb import logs

# ------------------------------------------------------------------------------ fixtures

SUMMARY_HTML = """<html><body>
<table>
<tr><td><a href="shot_summaries.php?searchstring=SUMMARIES.SHOT">SHOT</a></td>
    <td><a href="x.php?searchstring=SUMMARIES.SHOT_TYPE">SHOT TYPE</a></td></tr>
<tr><td><a href="shot_summaries.php?ShotNumber=190001">190001</a>,plasma</td></tr>
<tr><td><a href="shot_summaries.php?ShotNumber=190002">190002</a>,plasma</td></tr>
</table>
<p>Run: 20220301</p>
<p>Session Leader: Someone</p>
</body></html>
"""


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    """A synced `runs/` tree, in the layout `RunStorage.save_run` writes."""
    d = tmp_path / "runs"
    run = d / "20220301"
    run.mkdir(parents=True)
    (run / "summary.html").write_text(SUMMARY_HTML, encoding="utf-8")
    (run / "summary.md").write_text("Run: 20220301\nShot Range: 190001 - 190002\n", encoding="utf-8")
    (run / "miniproposal.md").write_text("Subject: A synthetic experiment\n", encoding="utf-8")
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "20220301",
                "mpid": "2022-11-05",
                "shot_range": "190001 - 190002",
                "title": "A synthetic experiment",
                "session_leaders": ["someone"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (d / "index.json").write_text(json.dumps({"190001": "20220301", "190002": "20220301"}), "utf-8")
    return d


@pytest.fixture
def corpus_text(tmp_path: Path) -> Path:
    d = tmp_path / "per_shot_txt"
    d.mkdir()
    (d / "shot_190001.txt").write_text("already here", encoding="utf-8")
    return d


# ------------------------------------------------------------------------------ logs missing


def test_missing_is_the_shots_with_no_bundle(corpus_text):
    assert logs.missing([190001, 190002, 190003], corpus_text) == [190002, 190003]


def test_missing_is_sorted_and_deduplicated(corpus_text):
    assert logs.missing([190005, 190002, 190005], corpus_text) == [190002, 190005]


def test_the_fetch_command_is_the_tool_s_own_and_chunks_at_two_hundred_shots():
    shots = list(range(190000, 190450))
    cmds = logs.fetch_commands(shots, chunk=200)
    assert len(cmds) == 3
    assert cmds[0].startswith("uv run python main.py 190000 190001 ")
    assert cmds[0].split()[-1] == "190199"
    assert cmds[1].split()[4] == "190200"
    assert cmds[2].split()[-1] == "190449"


def test_the_fetch_command_of_a_single_shot_is_one_line():
    assert logs.fetch_commands([190001]) == ["uv run python main.py 190001"]


def test_the_textprocess_command_is_one_per_run_with_the_tool_s_own_flags():
    cmds = logs.textprocess_commands(["20220301", "20220302"], base_dir="data/runs", outdir="out")
    assert cmds == [
        "uv run python textprocess.py --base-dir data/runs --outdir out --run-id 20220301",
        "uv run python textprocess.py --base-dir data/runs --outdir out --run-id 20220302",
    ]


def test_the_textprocess_commands_are_deduplicated_and_sorted():
    assert len(logs.textprocess_commands(["20220301", "20220301"], base_dir="d", outdir="o")) == 1


def test_run_ids_come_from_the_shot_index_and_are_none_when_it_does_not_know(tmp_path):
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"190001": "20220301", "bogus": "x"}), encoding="utf-8")
    assert logs.run_ids([190001, 190002], index) == {190001: "20220301", 190002: None}


def test_run_ids_of_a_missing_index_are_all_unknown(tmp_path):
    assert logs.run_ids([190001], tmp_path / "nope.json") == {190001: None}


def test_cli_logs_missing_prints_the_count_the_shots_and_the_exact_commands(
    corpus_text, tmp_path, capsys
):
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"190002": "20220301"}), encoding="utf-8")
    shots = tmp_path / "list.txt"
    shots.write_text("190001\n190002\n190003\n", encoding="utf-8")
    rc = cli.main(
        [
            "logs", "missing", "--list", str(shots),
            "--text-dir", str(corpus_text), "--index", str(index),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 of 3" in out
    assert "190002 190003" in out
    assert "uv run python main.py 190002 190003" in out
    assert "--run-id 20220301" in out
    # The shot the index cannot place has to be visible as such, not silently absent from the
    # textprocess step -- its run id only exists after main.py has written index.json over there.
    assert "190003" in out.split("run id not yet known")[1].split("\n")[0]


def test_cli_logs_missing_over_the_whole_corpus_reads_the_census(corpus_text, tmp_path, capsys):
    from .conftest import census_frame

    parquet = tmp_path / "census.parquet"
    census_frame([(190001, "mhr", 0.0, 5.0, True), (190002, "mhr", 0.0, 5.0, True)]).to_parquet(
        parquet, index=False
    )
    rc = cli.main(
        ["logs", "missing", "--all-corpus", "--census", str(parquet), "--text-dir", str(corpus_text)]
    )
    assert rc == 0 and "1 of 2" in capsys.readouterr().out


def test_cli_logs_missing_says_so_when_nothing_is_missing(corpus_text, tmp_path, capsys):
    shots = tmp_path / "list.txt"
    shots.write_text("190001\n", encoding="utf-8")
    rc = cli.main(["logs", "missing", "--list", str(shots), "--text-dir", str(corpus_text)])
    out = capsys.readouterr().out
    assert rc == 0 and "0 of 1" in out and "uv run" not in out


# ------------------------------------------------------------------------------ logs import


def test_import_copies_the_run_into_the_raw_layout_and_writes_the_bundles(runs_dir, tmp_path):
    raw, txt = tmp_path / "raw", tmp_path / "txt"
    report = logs.import_runs(runs_dir, raw_dir=raw, txt_dir=txt)
    assert report.runs == ["20220301"]
    assert (raw / "20220301" / "summary.html").read_text(encoding="utf-8") == SUMMARY_HTML
    assert json.loads((raw / "20220301" / "metadata.json").read_text())["mpid"] == "2022-11-05"
    assert sorted(p.name for p in txt.glob("*.txt")) == ["shot_190001.txt", "shot_190002.txt"]
    assert report.bundles_written == 2


def test_an_imported_bundle_has_the_three_sections_of_the_real_ones(runs_dir, tmp_path):
    logs.import_runs(runs_dir, raw_dir=tmp_path / "raw", txt_dir=tmp_path / "txt")
    got = (tmp_path / "txt" / "shot_190001.txt").read_text(encoding="utf-8")
    assert got.startswith("# DIII-D per-shot text bundle")
    assert "\n## General session context\n" in got
    assert "\n## Shot-specific context (from summary.html)\n" in got
    assert "SHOT TABLE ROW (name -> value)" in got


def test_import_never_overwrites_an_existing_bundle_without_force(runs_dir, tmp_path):
    txt = tmp_path / "txt"
    txt.mkdir()
    (txt / "shot_190001.txt").write_text("the corpus's own", encoding="utf-8")
    report = logs.import_runs(runs_dir, raw_dir=tmp_path / "raw", txt_dir=txt)
    assert (txt / "shot_190001.txt").read_text(encoding="utf-8") == "the corpus's own"
    assert report.bundles_written == 1 and report.bundles_kept == 1


def test_force_rewrites_an_existing_bundle(runs_dir, tmp_path):
    txt = tmp_path / "txt"
    txt.mkdir()
    (txt / "shot_190001.txt").write_text("the corpus's own", encoding="utf-8")
    report = logs.import_runs(runs_dir, raw_dir=tmp_path / "raw", txt_dir=txt, force=True)
    assert (txt / "shot_190001.txt").read_text(encoding="utf-8") != "the corpus's own"
    assert report.bundles_written == 2 and report.bundles_kept == 0


def test_a_dry_run_writes_nothing_and_reports_what_it_would_have_written(runs_dir, tmp_path):
    raw, txt = tmp_path / "raw", tmp_path / "txt"
    report = logs.import_runs(runs_dir, raw_dir=raw, txt_dir=txt, dry_run=True)
    assert report.bundles_written == 2 and report.files_copied == 4
    assert not raw.exists() and not txt.exists()


def test_importing_twice_changes_nothing_the_second_time(runs_dir, tmp_path):
    raw, txt = tmp_path / "raw", tmp_path / "txt"
    logs.import_runs(runs_dir, raw_dir=raw, txt_dir=txt)
    before = {p.name: p.read_bytes() for p in txt.glob("*.txt")}
    second = logs.import_runs(runs_dir, raw_dir=raw, txt_dir=txt)
    assert second.bundles_written == 0 and second.bundles_kept == 2
    assert {p.name: p.read_bytes() for p in txt.glob("*.txt")} == before
    # ... and with --force the bytes are still the same, which is what makes the port idempotent
    logs.import_runs(runs_dir, raw_dir=raw, txt_dir=txt, force=True)
    assert {p.name: p.read_bytes() for p in txt.glob("*.txt")} == before


def test_import_skips_a_directory_that_is_not_a_run(runs_dir, tmp_path):
    (runs_dir / "not-a-run").mkdir()
    report = logs.import_runs(runs_dir, raw_dir=tmp_path / "raw", txt_dir=tmp_path / "txt")
    assert report.runs == ["20220301"] and report.skipped == ["not-a-run"]


def test_cli_logs_import_dry_run(runs_dir, tmp_path, capsys):
    rc = cli.main(
        [
            "logs", "import", str(runs_dir), "--dry-run",
            "--raw-dir", str(tmp_path / "raw"), "--text-dir", str(tmp_path / "txt"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0 and "would write" in out and "shot_190001.txt" in out
    assert not (tmp_path / "txt").exists()


def test_cli_logs_import_refuses_a_runs_directory_that_is_not_there(tmp_path, capsys):
    rc = cli.main(["logs", "import", str(tmp_path / "nope")])
    assert rc == 1 and "nope" in capsys.readouterr().err


# ------------------------------------------------------------------------------ byte-compatible

_paths = config.load_paths()
_RAW = _paths.shotsummary_raw_dir
_TXT = _paths.per_shot_txt_dir


@pytest.mark.real_data
@pytest.mark.skipif(
    not (_RAW.exists() and _TXT.exists()), reason="the DIII-D text corpus is not mounted"
)
def test_recomposing_a_real_run_reproduces_its_bundles_byte_for_byte():
    """The whole point of porting `compose_per_shot_bundle` instead of writing a new one.

    A bundle this project regenerates has to be indistinguishable from the 22,950 the tool wrote,
    or every parser downstream -- `shot_table_row`, `mp_text`, the phenomenon search -- sees two
    populations where it should see one. Byte equality is the only check that says so.

    Only the shots whose bundle on disk carries THIS run's `RUN_ID:` header are compared, and that
    is not a convenience. `load_session` takes a run's shots to be every six-digit number in its
    summary text (upstream's rule), so a shot merely quoted as a comparison is in `shot_ids` --
    and `shot_<N>.txt` was written once per such mention, with the last writer winning. Run
    20210622 mentions 185805, whose file on disk says `RUN_ID: 20210413` and holds that run's
    material. Re-composing it from 20210622 SHOULD differ; the same quirk is measured from the
    other side in `text.mp_text`'s docstring (1,876 of 22,950 bundles disagree with index.json).
    """
    run_id = "20210622"  # shot 187199's run: a PDF mini-proposal and a full shot table
    session = logs.load_session(_RAW, run_id)
    checked = 0
    for shot in session.shot_ids:
        existing = _TXT / f"shot_{shot}.txt"
        if not existing.exists():
            continue
        on_disk = existing.read_text(encoding="utf-8", errors="replace")
        if not on_disk.startswith(f"# DIII-D per-shot text bundle\n\nRUN_ID: {run_id}\n"):
            continue  # written by a later run that merely mentioned this shot -- see the docstring
        assert logs.compose_per_shot_bundle(session, shot) == on_disk, f"shot_{shot}.txt differs"
        checked += 1
        if checked >= 5:
            break
    assert checked >= 3, f"only {checked} of run {run_id}'s shots are the corpus's own"
