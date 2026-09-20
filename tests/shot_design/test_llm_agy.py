"""agy (Antigravity CLI) provider: prompt rendering, emulated tool calls, error
mapping. A fake `runner` plays agy's stdout; no real subprocess is invoked.
"""

import json

import pytest

from shot_design.llm import agy
from shot_design.llm.client import LLMUnavailable

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
