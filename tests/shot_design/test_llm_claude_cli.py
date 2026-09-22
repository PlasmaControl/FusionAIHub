"""Claude Code CLI provider: command shape, structured-output decoding, error mapping
and retries. A fake `runner` plays `claude -p`'s stdout; no real subprocess runs.
"""

import json

import pytest

from shot_design.llm import agy, claude_cli
from shot_design.llm.client import LLMClient, LLMUnavailable

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


def _ok(structured):
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": json.dumps(structured),
            "structured_output": structured,
        }
    )


def test_command_is_claude_print_with_schema_and_stripped_context():
    seen = {}

    def runner(cmd, prompt):
        seen["cmd"] = cmd
        return _ok({"content": "ready", "tool_calls": []})

    body = {"messages": [{"role": "user", "content": "hi"}], "model": "sonnet"}
    claude_cli.ClaudeCliProvider({"bin": "claude"}, runner=runner).chat(body)
    cmd = seen["cmd"]
    assert cmd[:4] == ["claude", "-p", "--model", "sonnet"]
    assert cmd[cmd.index("--json-schema") + 1] == json.dumps(agy.SCHEMA)
    for flag in claude_cli.DEFAULT_EXTRA_ARGS:
        assert flag in cmd
    assert cmd[-1].startswith("USER: hi")  # the rendered prompt is the last argument


def test_structured_output_becomes_tool_calls():
    structured = {
        "content": "",
        "tool_calls": [{"name": "search_shots", "arguments": {"q": "ELM"}}],
    }
    body = {
        "messages": [{"role": "user", "content": "x"}],
        "model": "m",
        "tools": TOOLS,
    }
    r = claude_cli.ClaudeCliProvider({}, runner=lambda c, p: _ok(structured)).chat(body)
    assert r.tool_calls[0].name == "search_shots"
    assert r.tool_calls[0].arguments == {"q": "ELM"}
    assert r.model == "m"


def test_json_object_inside_result_is_accepted_without_structured_output():
    raw = json.dumps(
        {
            "subtype": "success",
            "is_error": False,
            "result": '```json\n{"content": "plain", "tool_calls": []}\n```',
        }
    )
    r = claude_cli.ClaudeCliProvider({}, runner=lambda c, p: raw).chat(
        {"messages": [], "model": "m"}
    )
    assert r.content == "plain" and r.tool_calls == []


def test_cli_error_payload_is_unavailable_with_its_message():
    raw = json.dumps(
        {
            "subtype": "error",
            "is_error": True,
            "result": "Not logged in · Please run /login",
        }
    )
    with pytest.raises(LLMUnavailable, match="Not logged in"):
        claude_cli.ClaudeCliProvider({"retries": 2}, runner=lambda c, p: raw).chat(
            {"messages": [], "model": "m"}
        )


def test_rate_limit_is_retried_with_backoff_then_succeeds(monkeypatch):
    calls = {"n": 0}
    slept = []
    monkeypatch.setattr(agy.time, "sleep", slept.append)

    def runner(cmd, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise agy.AgyError("exit 1: API Error: 429 rate_limit_error")
        return _ok({"content": "ok", "tool_calls": []})

    cfg = {"retries": 2, "backoff_s": 3}
    r = claude_cli.ClaudeCliProvider(cfg, runner=runner).chat(
        {"messages": [], "model": "m"}
    )
    assert r.content == "ok" and calls["n"] == 2 and slept == [3.0]


def test_client_registers_the_claude_provider_and_checks_the_binary(paths, monkeypatch):
    cfg = {"provider": "claude", "claude": {"bin": "definitely-not-a-binary-xyz"}}
    client = LLMClient(cfg=cfg, paths=paths)
    ok, hint = client.available()
    assert not ok and "claude CLI" in hint
    assert "claude" in client._providers
