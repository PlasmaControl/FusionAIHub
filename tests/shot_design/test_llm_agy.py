"""agy (Antigravity CLI) provider: prompt rendering, emulated tool calls, error
mapping. A fake `runner` plays agy's stdout; no real subprocess is invoked -- except
in the launch-error tests below, which exercise the real `_run_subprocess` against a
nonexistent binary or a script that outlives its timeout, never the real `agy` CLI.
"""

import json
import sys

import pytest

from shot_design.llm import agy
from shot_design.llm.client import LLMUnavailable

# Captured at import time, before conftest's autouse `_no_real_agy_subprocess` fixture
# ever runs, so a test that wants the real subprocess path can restore it for itself.
_REAL_RUN_SUBPROCESS = agy.AgyProvider._run_subprocess

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_shots",
            "description": "d",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]


def test_render_prompt_lists_tools_and_roles():
    p = agy.render_prompt(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], TOOLS
    )
    assert "S" in p and "U" in p and "search_shots" in p and "tool_calls" in p


def test_chat_parses_structured_output_into_tool_calls():
    seen = {}

    def runner(cmd, prompt):
        seen["cmd"] = cmd
        return json.dumps(
            {
                "result": json.dumps(
                    {
                        "content": "",
                        "tool_calls": [
                            {"name": "search_shots", "arguments": {"q": "ELM"}}
                        ],
                    }
                )
            }
        )

    cfg = {"bin": "agy", "effort": "low", "timeout_s": 5}
    body = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": TOOLS,
        "model": "quality",
    }
    r = agy.AgyProvider(cfg, runner=runner).chat(body)
    assert r.tool_calls[0].name == "search_shots"
    assert r.tool_calls[0].arguments == {"q": "ELM"}
    assert "--json-schema" in seen["cmd"] and seen["cmd"][-1].startswith("-p=")


def test_over_encoded_structured_output_is_unwrapped():
    def runner(cmd, prompt):
        return json.dumps(
            {
                "status": "SUCCESS",
                "response": "",
                "structured_output": {
                    "content": json.dumps({"content": "ready", "tool_calls": []})
                },
            }
        )

    body = {
        "messages": [{"role": "user", "content": "x"}],
        "model": "gemini-3.8-flash-low",
    }
    r = agy.AgyProvider({"bin": "agy"}, runner=runner).chat(body)
    assert r.content == "ready" and r.tool_calls == []


def test_nonzero_exit_is_unavailable():
    def runner(cmd, prompt):
        raise agy.AgyError("exit 2: boom")

    with pytest.raises(LLMUnavailable, match="boom"):
        body = {"messages": [], "model": "m"}
        agy.AgyProvider({"bin": "agy"}, runner=runner).chat(body)


def test_missing_binary_raises_llmunavailable_naming_it(monkeypatch):
    monkeypatch.setattr(agy.AgyProvider, "_run_subprocess", _REAL_RUN_SUBPROCESS)
    provider = agy.AgyProvider({"bin": "definitely-not-a-real-binary-xyz"})
    body = {"messages": [{"role": "user", "content": "hi"}], "model": "m"}
    with pytest.raises(LLMUnavailable, match="definitely-not-a-real-binary-xyz"):
        provider.chat(body)


def test_timeout_error_includes_the_stderr_tail(monkeypatch):
    monkeypatch.setattr(agy.AgyProvider, "_run_subprocess", _REAL_RUN_SUBPROCESS)
    provider = agy.AgyProvider({"timeout_s": 0.2})
    script = (
        "import sys, time\n"
        "sys.stderr.write('partial trace\\n')\n"
        "sys.stderr.flush()\n"
        "time.sleep(5)\n"
    )
    cmd = [sys.executable, "-c", script]
    with pytest.raises(agy.AgyError, match="partial trace"):
        provider._run_subprocess(cmd, "prompt")


def test_retries_on_a_transient_failure_then_succeeds():
    calls = {"n": 0}

    def flaky(cmd, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise agy.AgyError("agy timed out after 5s")
        payload = {"content": "ready", "tool_calls": []}
        return json.dumps({"result": json.dumps(payload)})

    cfg = {"bin": "agy", "retries": 2}
    body = {"messages": [{"role": "user", "content": "x"}], "model": "m"}
    r = agy.AgyProvider(cfg, runner=flaky).chat(body)
    assert r.content == "ready" and calls["n"] == 2


def test_retries_are_exhausted_and_raise_the_last_error():
    def always_times_out(cmd, prompt):
        raise agy.AgyError("agy timed out after 5s")

    cfg = {"bin": "agy", "retries": 1}
    body = {"messages": [], "model": "m"}
    with pytest.raises(LLMUnavailable, match="timed out"):
        agy.AgyProvider(cfg, runner=always_times_out).chat(body)


def test_a_real_exit_error_is_not_retried_even_with_retries_configured():
    calls = {"n": 0}

    def runner(cmd, prompt):
        calls["n"] += 1
        raise agy.AgyError("exit 2: boom")

    cfg = {"bin": "agy", "retries": 2}
    body = {"messages": [], "model": "m"}
    with pytest.raises(LLMUnavailable, match="boom"):
        agy.AgyProvider(cfg, runner=runner).chat(body)
    assert calls["n"] == 1
