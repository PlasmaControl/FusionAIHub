"""The `shot_design` command line, end to end over the synthetic fixtures.

Every assertion here is about what a user sees or gets back: an exit code, a line of terminal
output, a file on disk. The CLI owns no physics -- build/store/legacy_raw are tested elsewhere -- so
what is worth pinning is the shape of the interface: which subcommands exist, what they print,
what they refuse, and that a query's flags land in the right QueryState fields.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from shot_design import cli
from shot_design.schema import ShotRecord, ShotSummary
from shot_design.shotdb import text


@pytest.fixture
def stub_embeddings(monkeypatch):
    """MiniLM stands aside: nothing here tests the encoder, and loading it costs seconds."""
    monkeypatch.setattr(text, "embed_texts", lambda texts: np.zeros((len(texts), 384), np.float32))


@pytest.fixture
def shot_list_file(tmp_path, staged_shot_a, staged_shot_b) -> Path:
    p = tmp_path / "list.yaml"
    p.write_text(
        f"shots:\n  - {{shot: {staged_shot_a}, reason: a}}\n  - {{shot: {staged_shot_b}, reason: b}}\n"
    )
    return p


@pytest.fixture
def built(paths, text_fixtures, stub_embeddings, shot_list_file, capsys) -> str:
    assert cli.main(["build", "--list-file", str(shot_list_file), "--workers", "1"]) == 0
    return capsys.readouterr().out


# ------------------------------------------------------------------------------------- build


def test_build_reports_and_prints_the_coverage_table(built, paths, staged_shot_a):
    assert "built 2 shots" in built and "8 segments" in built
    assert "coverage:" in built and "[nbi]" in built and "pnbi_15L" in built
    # the encoder is Task 14; the build must say so rather than leave it unexplained
    assert "not_installed" in built
    manifest = json.loads((paths.db_dir / "manifest.json").read_text())
    assert manifest["n_shots"] == 2 and manifest["ignite"]["status"] == "not_installed"


def test_build_skips_shots_with_no_ip_on_disk(paths, text_fixtures, stub_embeddings, capsys):
    """A shot whose raw file is absent (or still mid-fetch) would build as a record with zero
    segments, which is junk in the database rather than a failure anybody notices."""
    assert cli.main(["build", "--shots", "999999", "--workers", "1"]) == 1
    out = capsys.readouterr()
    assert "no Ip signal on disk" in out.out and "999999" in out.out
    assert "nothing to build" in out.err


def test_build_all_overrides_the_skip(paths, text_fixtures, stub_embeddings, capsys):
    assert cli.main(["build", "--shots", "999999", "--workers", "1", "--all"]) == 0
    assert "built 1 shots / 0 segments" in capsys.readouterr().out


def test_build_reader_corpus_reads_the_corpus_and_says_so(
    paths, signal_corpus, labelmaker_features, text_fixtures, stub_embeddings, capsys
):
    """`--reader corpus` has to reach the Ip-on-disk filter too: that check is what decides
    whether a shot is built at all, and asking the legacy layout about a corpus shot would skip
    every one of the 500."""
    assert (
        cli.main(["build", "--shots", str(signal_corpus), "--workers", "1", "--reader", "corpus",
                  "--no-encode"]) == 0
    )  # fmt: skip
    out = capsys.readouterr().out
    assert "built 1 shots" in out and "no Ip signal on disk" not in out
    manifest = json.loads((paths.db_dir / "manifest.json").read_text())
    assert manifest["reader"] == "corpus"


def test_build_defaults_to_the_legacy_reader(tmp_path):
    args = cli.build_parser().parse_args(["build"])
    assert args.reader == "legacy" and args.limit is None


def test_build_limit_takes_the_first_n_of_the_list(paths, shot_list_file, staged_shot_a,
                                                   text_fixtures, stub_embeddings, capsys):
    """A pilot builds the first N shots of the real list, so its timing is measured against the
    same shots, the same reader and the same code the full run will use."""
    assert cli.main(["build", "--list-file", str(shot_list_file), "--workers", "1",
                     "--limit", "1", "--no-encode"]) == 0  # fmt: skip
    assert "built 1 shots" in capsys.readouterr().out


# -------------------------------------------------------------------------------------- show


def test_show_prints_a_legible_record(built, staged_shot_a, capsys):
    assert cli.main(["show", str(staged_shot_a)]) == 0
    out = capsys.readouterr().out
    assert f"Shot {staged_shot_a}" in out
    assert "MP 2014-21-20" in out
    assert "regime QH" in out and "source: text" in out
    assert "[flat_top]" in out and re.search(r"\bip 1\.\d+e\+06 A", out)
    assert "nbi " in out and "15L" in out  # the three beams that are on
    assert "coverage" in out and "provenance" in out and "EFIT01" in out
    assert "verdict   good" in out


def test_show_and_export_use_the_one_describe_template(built, paths, staged_shot_a, capsys):
    """Three renderers of the one-line shot description (cli.describe, describe._segment_line,
    rank._fallback_description) printed the same shot differently -- `q95 3.38` in show and
    `q95 3.4` in query. There is one now: show prints describe.segment_line, export stores
    describe.describe, and a query result carries describe.describe."""
    from shot_design.retrieval import describe as D
    from shot_design.shotdb import store

    rec = store.ShotDB.load(paths.db_dir).get(staged_shot_a)
    assert cli.main(["show", str(staged_shot_a)]) == 0
    shown = " ".join(capsys.readouterr().out.split())  # _fill wraps the line at 98 columns
    headline = D.segment_line(rec, "flat_top")
    assert headline and headline.startswith("Flat top") and headline in shown
    assert cli.main(["show", str(staged_shot_a), "--segment", "ramp_up"]) == 0
    assert D.segment_line(rec, "ramp_up") in " ".join(capsys.readouterr().out.split())
    assert cli.main(["export", str(staged_shot_a), "--format", "json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["description"] == D.describe(rec)


def test_show_segment_selection_and_full(built, staged_shot_a, capsys):
    assert cli.main(["show", str(staged_shot_a), "--segment", "ramp_up"]) == 0
    out = capsys.readouterr().out
    assert "[ramp_up]" in out and "[flat_top]" not in out
    assert cli.main(["show", str(staged_shot_a), "--segment", "all", "--full"]) == 0
    out = capsys.readouterr().out
    for name in ("full", "ramp_up", "flat_top", "ramp_down"):
        assert f"[{name}]" in out
    assert "betan_mean=" in out  # --full dumps every scalar by its column name


def test_show_json_round_trips_through_record(built, staged_shot_a, capsys):
    assert cli.main(["show", str(staged_shot_a), "--json"]) == 0
    rec = ShotRecord.model_validate_json(capsys.readouterr().out)
    assert rec.shot == staged_shot_a and rec.segment("flat_top") is not None


def test_show_says_so_when_the_shot_is_not_in_the_database(built, capsys):
    assert cli.main(["show", "123456"]) == 1
    err = capsys.readouterr().err
    assert "123456" in err and "shot_design add" in err


def test_commands_refuse_cleanly_without_a_database(paths, capsys):
    assert cli.main(["show", "900001"]) == 1
    assert "no database" in capsys.readouterr().err


# ------------------------------------------------------------------------------------ export


def test_export_json_to_a_file_and_to_stdout(built, staged_shot_a, tmp_path, capsys):
    out_path = tmp_path / "s.json"
    assert cli.main(["export", str(staged_shot_a), "--format", "json", "--out", str(out_path)]) == 0
    rows = json.loads(out_path.read_text())
    assert len(rows) == 1 and rows[0]["mpid"] == "2014-21-20"
    assert ShotSummary.model_validate(rows[0]).shot == staged_shot_a

    capsys.readouterr()
    assert cli.main(["export", str(staged_shot_a), "--format", "json"]) == 0
    stdout_rows = json.loads(capsys.readouterr().out)
    assert ShotSummary.model_validate(stdout_rows[0]).description == rows[0]["description"]
    assert stdout_rows[0]["description"]  # the description is composed, not left empty


def test_export_parquet(built, staged_shot_a, staged_shot_b, tmp_path):
    import pandas as pd

    out_path = tmp_path / "s.parquet"
    argv = [
        "export",
        str(staged_shot_a),
        str(staged_shot_b),
        "--format",
        "parquet",
        "--out",
        str(out_path),
    ]
    assert cli.main(argv) == 0
    df = pd.read_parquet(out_path)
    assert list(df["shot"]) == [staged_shot_a, staged_shot_b]
    assert "flat_top.ip_mean" in df.columns


def test_export_parquet_needs_an_out_path(built, staged_shot_a, capsys):
    assert cli.main(["export", str(staged_shot_a), "--format", "parquet"]) == 2
    assert "--out" in capsys.readouterr().err


# ---------------------------------------------------------------------------- add + coverage


def test_add_upserts_one_shot(built, paths, staged_shot_a, our_shot_c, text_fixtures, capsys):
    assert cli.main(["add", str(our_shot_c)]) == 0
    out = capsys.readouterr().out
    assert f"added 1 shot: {our_shot_c}" in out
    assert cli.main(["show", str(our_shot_c)]) == 0
    assert "fast_current_quench" in capsys.readouterr().out


def test_coverage_recomputes_from_the_records(built, capsys):
    assert cli.main(["coverage"]) == 0
    out = capsys.readouterr().out
    assert "fields over 2 shots" in out and "present" in out


def test_coverage_splits_the_encoded_shots_by_the_device_that_encoded_them(built, paths, capsys):
    """"500 shots are encoded" is not actionable on its own: the codes differ between cuda and
    cpu, and between cpu at four threads and cpu at eight. The census has to show the split, and
    show a cache with no provenance sidecar as unknown rather than folding it into a device."""
    from shot_design.design import provenance
    from shot_design.shotdb import build as build_mod

    codes = build_mod.frame_codes_dirs(paths)[0]
    codes.mkdir(parents=True, exist_ok=True)
    for shot, device in ((190001, "cuda"), (190002, "cpu"), (190003, None)):
        (codes / f"{shot}.pt").write_bytes(b"")
        if device:
            provenance.write_sidecar(
                codes,
                shot,
                provenance.build_sidecar(shot, device=device, input_file=None, bundle=None),
            )

    assert cli.main(["coverage"]) == 0
    out = capsys.readouterr().out
    assert "frame codes  3 shot(s) encoded" in out
    assert "cuda 1" in out and "cpu 1" in out and "no sidecar 1" in out


def test_query_arguments_become_one_query_state():
    """The parser's whole job: every flag lands in the right QueryState field. Retrieval itself
    is tested in test_retrieval.py, against a database; this needs none."""
    argv = [
        "query",
        "--text", "wide pedestal QH",
        "--negative", "disruptive",
        "--ref-shot", "161172",
        "--segment", "flat_top",
        "--constraint", "ip_mean=1.0e6:1.4e6",
        "--constraint", "q95_mean=:4.5",
        "--actuator", "nbi.total=5e6",
        "--require", "QH",
        "--avoid", "dud",
        "--exclude-shot", "160904",
        "--exclude-run", "20150120",
        "--prefer-outcome", "success",
        "-n", "5",
    ]  # fmt: skip
    state = cli._query_state(cli.build_parser().parse_args(argv))
    assert state.text == "wide pedestal QH" and state.negatives == ["disruptive"]
    assert state.ref_shot == 161172 and state.n == 5 and state.segment == "flat_top"
    assert state.constraints["ip_mean"].lo == 1.0e6 and state.constraints["ip_mean"].hi == 1.4e6
    assert state.constraints["q95_mean"].lo is None and state.constraints["q95_mean"].hi == 4.5
    assert state.actuators == {"nbi.total": 5.0e6}
    assert state.require_labels == {"QH"} and state.avoid_labels == {"dud"}
    assert state.exclude_shots == {160904} and state.exclude_runs == {"20150120"}
    assert state.prefer_outcome == "success"


def test_the_short_query_flags_mean_the_same_thing():
    """--ref/--where/--n are the spellings the Phase 2 brief used; --ref-shot/--constraint/-n are
    the ones docs/CLI.md documents. They have to be the same flag, not two half-wired ones."""
    long = cli._query_state(
        cli.build_parser().parse_args(
            ["query", "--ref-shot", "161172", "--constraint", "ip_mean=1e6:1.4e6", "-n", "5"]
        )
    )
    short = cli._query_state(
        cli.build_parser().parse_args(
            ["query", "--ref", "161172", "--where", "ip_mean:1e6:1.4e6", "--n", "5"]
        )
    )
    assert long == short


def test_query_rejects_a_malformed_constraint(capsys):
    with pytest.raises(SystemExit):
        cli.main(["query", "--constraint", "ip_mean"])
    assert "COLUMN=LO:HI" in capsys.readouterr().err


def test_query_prints_the_proposals_own_flags_before_the_results(built, capsys):
    """`shot_design query --actuator nbi.total=5e7` -- 2.5x the installed beam power -- printed no flag,
    because the rules ran on the result rows only. The two-shot fixture database has no fitted
    PCA, so no channel fires and the command exits 2; the proposal is still answered first."""
    assert cli.main(["query", "--actuator", "nbi.total=5e7"]) == 2
    out, err = capsys.readouterr()
    flat = " ".join(out.split())  # _fill wraps the flag message at 98 columns
    assert "proposal  nbi.total=5e+07" in out
    assert "error     pnbi_total_peak = 5e+07 > 2e+07" in out
    assert "requested NBI power above the total installed beam power" in flat
    assert "rules had no input to check" in flat
    assert out.index("proposal") < out.index("rules had no input")  # the block, then its skips
    assert "no channel had anything to search on" in err
    assert cli.main(["query", "--actuator", "nbi.total=5e6"]) == 2
    assert "ok        no configured limit violated" in capsys.readouterr().out


def test_query_rejects_an_unknown_segment_before_touching_the_database(capsys):
    """`--segment flattop` used to load the database and then die with a pydantic traceback."""
    with pytest.raises(SystemExit) as e:
        cli.main(["query", "--ref", "161172", "--segment", "flattop"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'flattop'" in err and "flat_top" in err
    # `show --segment all` is legitimate and untouched.
    assert cli.build_parser().parse_args(["show", "1", "--segment", "all"]).segment == "all"


def test_a_string_system_exit_is_printed_not_turned_into_a_valueerror(monkeypatch, capsys):
    """A library a command calls may exit with a message rather than a status -- a staging
    script refusing a read-only raw_dir with `SystemExit("refusing ...")` is the shape this was
    written against. `int()` on that message raised ValueError out of main."""

    def refuse(args):
        raise SystemExit("refusing to write into the read-only store /x: raw_dir=/x/raw")

    monkeypatch.setattr(cli, "cmd_coverage", refuse)
    assert cli.main(["coverage"]) == 1
    assert "refusing to write into the read-only store" in capsys.readouterr().err
    monkeypatch.setattr(cli, "cmd_coverage", lambda args: (_ for _ in ()).throw(SystemExit(3)))
    assert cli.main(["coverage"]) == 3


def test_build_unions_list_and_list_file(tmp_path, monkeypatch):
    """`--list poc_v1 --list-file extra.yaml` silently dropped every poc_v1 shot when the two
    were `elif` branches; `fetch` already unioned them."""
    extra = tmp_path / "extra.yaml"
    extra.write_text("shots:\n  - {shot: 1}\n  - {shot: 2}\n")
    ns = cli.build_parser().parse_args(
        ["build", "--list", "poc_v1", "--list-file", str(extra), "--shots", "3"]
    )
    got = cli._shots(ns, "poc_v1")
    assert {1, 2, 3} <= set(got) and len(got) == 3 + len(cli.config.load_shot_list("poc_v1"))
    # the default list applies only when no selector at all was given
    assert cli._shots(cli.build_parser().parse_args(["build"]), "poc_v1") == (
        cli.config.load_shot_list("poc_v1")
    )
    assert cli._shots(cli.build_parser().parse_args(["build", "--shots", "5"]), "poc_v1") == [5]


def test_the_eval_subcommand_is_the_harness_and_not_the_old_stub(capsys):
    """`eval` was refused outright while the harness was deferred, because a stub that parsed
    --task/--folds/--out only to print a Phase-2 notice advertised a command that did nothing.
    Task I11 built the real one, so the word is now taken -- by three subcommands, none of which
    accepts the stub's flags."""
    assert set(cli.build_parser().parse_args(["eval", "prompts"]).__dict__) >= {"split", "json"}
    assert "eval" in cli.build_parser().format_help()
    with pytest.raises(SystemExit) as e:
        cli.main(["eval", "--task", "mp2shots"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'mp2shots'" in err
    assert "prompts" in err and "latency" in err and "recall" in err


# --------------------------------------------------------------------------------- packaging


def test_console_script_is_installed_and_runnable():
    for argv in (["-m", "shot_design", "--help"], ["-m", "shot_design", "show", "--help"]):
        r = subprocess.run([sys.executable, *argv], capture_output=True, text=True, check=False)
        assert r.returncode == 0, r.stderr
    assert (
        "build"
        in subprocess.run(
            [sys.executable, "-m", "shot_design", "--help"], capture_output=True, text=True, check=False
        ).stdout
    )


def test_the_encoder_is_kept_offline_by_default():
    """Task 12 measured this: sentence_transformers hangs for minutes on a node with no outbound
    route unless huggingface_hub is told the cache is all there is."""
    import os

    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert "SHOT_DESIGN_HF_ONLINE" in Path(cli.__file__).read_text()


@pytest.mark.parametrize("argv", [["show", "1"], ["export", "1"], ["coverage"], ["query", "--ref", "1"], ["add", "1"]])
def test_data_commands_report_an_empty_database(tmp_path, monkeypatch, capsys, argv):
    monkeypatch.delenv("SHOT_DESIGN_PATHS", raising=False)
    monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", str(tmp_path))
    assert cli.main(argv) == 1
    error = capsys.readouterr().err
    assert "no database at" in error and str(tmp_path) in error
    assert "Traceback" not in error
    assert list(tmp_path.iterdir()) == []


def test_explicitly_empty_data_root_reports_a_clear_error(monkeypatch, capsys):
    monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", "")
    assert cli.main(["coverage"]) == 1
    assert "SHOT_DESIGN_DATA_ROOT is empty" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["fetch", "serve", "curate"])
def test_retired_commands_and_restored_serve(command):
    parser = cli.build_parser()
    if command == "serve":
        assert parser.parse_args([command]).func is cli.cmd_serve
        return
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([command])
    assert exc.value.code == 2




def test_actuation_list_and_show(paths, capsys):
    from shot_design.retrieval import actuation
    from shot_design.schema import ActuationSet, ActuatorWaveform, Vertex

    assert cli.main(["actuation", "list"]) == 0
    assert "no saved actuation sets" in capsys.readouterr().out
    stored = actuation.save(
        ActuationSet(
            source_shot=900001,
            waveforms={
                "nbi.total": ActuatorWaveform(
                    key="nbi.total", units="W", vertices=[Vertex(t_s=0, y=0), Vertex(t_s=1, y=5e6)]
                )
            },
            gas_species={"GASA": "D2", "GASC": "N2"},
        ),
        paths,
    )
    assert cli.main(["actuation", "list"]) == 0
    out = capsys.readouterr().out
    assert stored.id in out and "shot 900001" in out and "1 waveforms" in out
    assert cli.main(["actuation", "show", stored.id]) == 0
    out = capsys.readouterr().out
    assert "nbi.total [W] seed=reference" in out and "(1 s, 5e+06)" in out
    assert "gas species: GASA=D2, GASC=N2" in out
    assert cli.main(["actuation", "show", stored.id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == stored.id
    assert cli.main(["actuation", "show", "nope"]) == 2
    assert cli.main(["actuation", "show", actuation.new_id(5)]) == 1


def test_the_manifest_records_where_the_shot_list_came_from(
    paths, shot_list_file, staged_shot_a, text_fixtures, stub_embeddings
):
    """`list: args.list` was null for a --list-file or --shots build and named the whole list for
    a `--limit 20` one, so the field either said nothing or overstated. The source, how many shots
    it named, how many were built and the limit that was applied are four separate facts."""
    assert cli.main(["build", "--list-file", str(shot_list_file), "--workers", "1",
                     "--limit", "1", "--no-encode"]) == 0  # fmt: skip
    m = json.loads((paths.db_dir / "manifest.json").read_text())
    assert m["shot_source"] == f"list-file:{shot_list_file}"
    assert m["n_requested"] == 2 and m["limit"] == 1 and m["n_built"] == 1


def test_the_manifest_names_an_explicit_shot_list_as_its_source(
    paths, staged_shot_a, text_fixtures, stub_embeddings
):
    assert cli.main(["build", "--shots", str(staged_shot_a), "--workers", "1", "--no-encode"]) == 0
    m = json.loads((paths.db_dir / "manifest.json").read_text())
    assert m["shot_source"] == "shots:1" and m["n_requested"] == 1 and m["limit"] is None


def test_the_no_encode_flag_is_named_by_the_cli_and_not_by_the_library(
    paths, staged_shot_a, text_fixtures, stub_embeddings, capsys
):
    """`build(..., encode=False)` is a library call: writing "--no-encode" into the manifest of a
    database nobody passed a flag to is the library speaking for a caller it does not have. The
    manifest says what was skipped; the CLI adds the flag that skipped it."""
    assert cli.main(["build", "--shots", str(staged_shot_a), "--workers", "1", "--no-encode"]) == 0
    assert "--no-encode" in capsys.readouterr().out
    reason = json.loads((paths.db_dir / "manifest.json").read_text())["ignite"]["reason"]
    assert "IGNITE waveform channel" in reason and "--no-encode" not in reason
