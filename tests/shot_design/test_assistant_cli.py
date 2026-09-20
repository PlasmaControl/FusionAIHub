"""`shot_design assistant`: run the design harness once from the command line, writing a
JSONL trace of every stage transition and every LLM round trip, and printing the saved
design id alone on the final stdout line.

`design.assistant.run_design` is monkeypatched throughout: this file pins the CLI's own
plumbing (the trace file, the --provider override, the stdout/stderr contract), not the
assistant pipeline itself, which test_assistant.py already covers end to end. Hermetic:
no network, no agy, no Slurm -- `LLMClient.chat` is monkeypatched wherever the fake
pipeline calls it.

Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import json

import pytest

from shot_design import cli
from shot_design.design import assistant as assistant_mod
from shot_design.llm.client import LLMClient, Reply

DESIGN_ID = "a" * 32


@pytest.fixture(autouse=True)
def fake_db(monkeypatch):
    """`assistant` opens a database before calling `run_design`; the pipeline itself is
    monkeypatched below, so a sentinel stands in for a built `ShotDB` and no fixture
    corpus or `shot_design build` is needed here."""
    monkeypatch.setattr(cli, "_open_db", lambda paths: object())


def _install_fake_run_design(monkeypatch, *, use_client=False, fail=None):
    """A stand-in `run_design` that walks the real 5 stages (each `running` then
    `complete`, exactly as the real pipeline does) and, if asked, makes one real-shaped
    LLM call through the client the CLI built, so the trace's "llm" lines can be tested
    without a model."""
    calls = []

    def fake_run_design(
        prompt, paths, db, *, client=None, model="quality", progress=None
    ):
        calls.append({"prompt": prompt, "client": client, "model": model})
        if fail is not None:
            raise fail
        for key, label in assistant_mod.STAGES:
            progress(key, "running", label)
            if use_client and key == "propose":
                client.chat([{"role": "user", "content": prompt}], model=model)
            progress(key, "complete", label)
        return {"design_id": DESIGN_ID}

    monkeypatch.setattr(assistant_mod, "run_design", fake_run_design)
    return calls


def _trace_lines(trace):
    return [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]


def test_writes_five_stage_lines_and_prints_the_design_id(
    tmp_path, paths, monkeypatch, capsys
):
    _install_fake_run_design(monkeypatch)
    trace = tmp_path / "t.jsonl"

    rc = cli.main(
        ["assistant", "--prompt", "control tearing modes", "--trace", str(trace)]
    )

    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == DESIGN_ID  # the demo script pipes this through `tail -1`

    lines = _trace_lines(trace)
    assert len(lines) >= 5
    stage_lines = [line for line in lines if line.get("kind") != "llm"]
    stages = {line["stage"] for line in stage_lines}
    assert stages == {key for key, _ in assistant_mod.STAGES}
    for line in stage_lines:
        assert set(line) >= {"stage", "status", "detail", "t"}
        assert isinstance(line["t"], (int, float))


def test_logs_every_llm_round_to_the_same_trace_file(
    tmp_path, paths, monkeypatch, capsys
):
    monkeypatch.setattr(
        LLMClient,
        "chat",
        lambda self, messages, **kw: Reply(content="ready", model="stub-model"),
    )
    _install_fake_run_design(monkeypatch, use_client=True)
    trace = tmp_path / "t.jsonl"

    rc = cli.main(["assistant", "--prompt", "hello there", "--trace", str(trace)])

    assert rc == 0
    llm_lines = [line for line in _trace_lines(trace) if line.get("kind") == "llm"]
    assert len(llm_lines) == 1
    assert llm_lines[0]["messages"] == [{"role": "user", "content": "hello there"}]
    assert llm_lines[0]["reply"]["content"] == "ready"


def test_provider_flag_overrides_the_configured_default(
    tmp_path, paths, monkeypatch, capsys
):
    calls = _install_fake_run_design(monkeypatch)
    trace = tmp_path / "t.jsonl"

    rc = cli.main(
        ["assistant", "--prompt", "x", "--trace", str(trace), "--provider", "agy"]
    )

    assert rc == 0
    assert calls[0]["client"].cfg["provider"] == "agy"


def test_provider_defaults_to_the_configured_yaml_value(
    tmp_path, paths, monkeypatch, capsys
):
    calls = _install_fake_run_design(monkeypatch)
    trace = tmp_path / "t.jsonl"

    rc = cli.main(["assistant", "--prompt", "x", "--trace", str(trace)])

    assert rc == 0
    # configs/shot_design/llm.yaml's own default; untouched, --provider was not given
    assert calls[0]["client"].cfg["provider"] == "ollama"


def test_a_failed_run_is_reported_on_stderr_with_exit_1(
    tmp_path, paths, monkeypatch, capsys
):
    _install_fake_run_design(monkeypatch, fail=ValueError("no usable references"))
    trace = tmp_path / "t.jsonl"

    rc = cli.main(["assistant", "--prompt", "x", "--trace", str(trace)])

    assert rc == 1
    assert "no usable references" in capsys.readouterr().err
