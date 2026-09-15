"""LLM client: endpoint discovery, OpenAI-shaped chat with tools, disk cache, offline behaviour.
No real model is contacted; httpx.MockTransport plays the server.

Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import json
import os
import time

import httpx
import pytest

from ideate import config
from ideate.llm.client import LLMClient, LLMUnavailable, Reply
from ideate.retrieval import describe

CFG = {
    "provider": "openai_compatible",
    "base_url": None,
    "endpoint_file": "llm/endpoint.json",
    "models": {"quality": "gemma4:26b", "fast": "gemma4:e4b"},
    "default": "quality",
    "timeout_s": 120,
    "max_tool_rounds": 8,
    "temperature": 0.0,
    "max_tokens": 600,
    "cache": True,
    "blurb": {"model": "quality", "max_words": 70, "prompt_version": 1},
}


def _completion(content="hello", tool_calls=None, model="gemma4:26b"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "x",
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
    }


def _transport(handler_calls: list, reply):
    def handler(request: httpx.Request):
        handler_calls.append(json.loads(request.content))
        return httpx.Response(200, json=reply)

    return httpx.MockTransport(handler)


def test_llm_yaml_matches_the_shape_the_client_reads():
    cfg = config.load_yaml("llm.yaml")
    assert cfg["provider"] == "off"
    assert cfg["models"] == {"quality": "gemma4:26b", "fast": "gemma4:e4b"}
    assert cfg["default"] == "quality"
    assert cfg["max_tool_rounds"] == 8 and cfg["timeout_s"] == 120
    assert cfg["endpoint_file"] == "llm/endpoint.json"


def test_no_endpoint_means_unavailable_with_the_start_command(paths):
    c = LLMClient(CFG, paths)
    assert c.endpoint() is None
    ok, hint = c.available()
    assert ok is False and "set base_url" in hint
    with pytest.raises(LLMUnavailable):
        c.chat([{"role": "user", "content": "hi"}])


def test_endpoint_file_is_discovered_and_recached_on_mtime(paths):
    f = paths.data_root / "llm" / "endpoint.json"
    f.parent.mkdir(parents=True)
    f.write_text(
        json.dumps({"url": "http://127.0.0.1:11434", "models": ["gemma4:26b"], "host": "vis2"})
    )
    c = LLMClient(CFG, paths)
    assert c.endpoint().url == "http://127.0.0.1:11434"
    f.write_text(json.dumps({"url": "http://node1:11434", "models": []}))
    # two writes can land in the same filesystem timestamp tick; force a distinct mtime
    os.utime(f, (time.time() + 5, time.time() + 5))
    assert c.endpoint().url == "http://node1:11434"
    f.unlink()
    assert c.endpoint() is None


def test_chat_posts_openai_shape_and_parses_tool_calls(paths):
    calls: list = []
    tc = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "set_query", "arguments": json.dumps({"patch": {"n": 5}})},
        }
    ]
    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"},
        paths,
        transport=_transport(calls, _completion("", tc)),
    )
    r = c.chat(
        [{"role": "user", "content": "five results"}],
        tools=[{"type": "function", "function": {"name": "set_query", "parameters": {}}}],
        model="fast",
    )
    assert isinstance(r, Reply)
    assert r.tool_calls[0].name == "set_query" and r.tool_calls[0].arguments == {"patch": {"n": 5}}
    sent = calls[0]
    assert sent["model"] == "gemma4:e4b" and sent["messages"][0]["content"] == "five results"
    assert sent["tools"][0]["function"]["name"] == "set_query" and sent["temperature"] == 0.0
    assert sent["stream"] is False and sent["max_tokens"] == 600


def test_provider_off_raises_without_opening_a_socket(paths):
    def handler(request):
        raise AssertionError("no request should be made when the provider is off")

    c = LLMClient(
        {**CFG, "provider": "off", "base_url": "http://llm.test"},
        paths,
        transport=httpx.MockTransport(handler),
    )
    assert c.endpoint() is None
    # the hint names the config file the reader has to edit, at its real path
    assert c.available() == (False, "configs/ideate/llm.yaml has provider: off")
    with pytest.raises(LLMUnavailable, match=r"configs/ideate/llm\.yaml has provider: off"):
        c.chat([{"role": "user", "content": "x"}])


def test_chat_url_is_v1_chat_completions(paths):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=_completion())

    c = LLMClient(
        {**CFG, "base_url": "http://llm.test/"}, paths, transport=httpx.MockTransport(handler)
    )
    c.chat([{"role": "user", "content": "x"}], cache=False)
    assert seen == ["http://llm.test/v1/chat/completions"]


def test_malformed_tool_arguments_are_kept_as_raw_text(paths):
    tc = [
        {"id": "c", "type": "function", "function": {"name": "set_query", "arguments": "{not json"}}
    ]
    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"}, paths, transport=_transport([], _completion("", tc))
    )
    r = c.chat([{"role": "user", "content": "x"}], cache=False)
    assert r.tool_calls[0].arguments == {
        "_raw": "{not json",
        "_error": "arguments are not a JSON object",
    }


def test_cache_hits_for_identical_deterministic_requests_only(paths):
    calls: list = []
    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"},
        paths,
        transport=_transport(calls, _completion("cached answer")),
    )
    m = [{"role": "user", "content": "same"}]
    a = c.chat(m)
    b = c.chat(m)
    assert (a.content, a.cached, b.content, b.cached) == (
        "cached answer",
        False,
        "cached answer",
        True,
    )
    assert len(calls) == 1
    c.chat(m, temperature=0.7)
    assert len(calls) == 2, "a sampled request is never served from the cache"
    assert any(paths.llm_cache_dir.rglob("*.json"))


def test_server_error_becomes_unavailable_with_the_status(paths):
    def handler(request):
        return httpx.Response(503, text="loading")

    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"}, paths, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMUnavailable, match="503"):
        c.chat([{"role": "user", "content": "x"}], cache=False)


def test_200_html_body_becomes_unavailable(paths):
    def handler(request):
        return httpx.Response(200, text="<html>not json</html>")

    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"}, paths, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMUnavailable):
        c.chat([{"role": "user", "content": "x"}], cache=False)


def test_200_choices_as_dict_becomes_unavailable(paths):
    def handler(request):
        return httpx.Response(200, json={"id": "x", "model": "gemma4:26b", "choices": {}})

    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"}, paths, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMUnavailable):
        c.chat([{"role": "user", "content": "x"}], cache=False)


def test_connect_error_keeps_the_start_command(paths):
    def handler(request):
        raise httpx.ConnectError("refused")

    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"}, paths, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMUnavailable, match="set base_url"):
        c.chat([{"role": "user", "content": "x"}], cache=False)


def test_openai_error_shape_raises_with_the_message(paths):
    def handler(request):
        return httpx.Response(
            200, json={"error": {"message": "model not loaded", "type": "server_error"}}
        )

    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"}, paths, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMUnavailable, match="model not loaded"):
        c.chat([{"role": "user", "content": "x"}], cache=False)


def test_api_key_is_sent_as_bearer_and_never_written_to_the_cache(paths, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret")
    seen_auth = []

    def handler(request):
        seen_auth.append(request.headers.get("authorization"))
        return httpx.Response(200, json=_completion("hi"))

    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"}, paths, transport=httpx.MockTransport(handler)
    )
    c.chat([{"role": "user", "content": "x"}])
    assert seen_auth == ["Bearer secret"]
    for f in paths.llm_cache_dir.rglob("*"):
        if f.is_file():
            assert "secret" not in f.read_text(encoding="utf-8")


def test_polish_keeps_the_template_when_the_model_changes_a_number(paths, monkeypatch):
    template = "Shot 161172 (2015-01-20). Flat top 1.20-4.80 s: Ip 1.21 MA."
    monkeypatch.setattr(
        describe,
        "_client_for_polish",
        lambda: _FakeClient("Shot 161172 ran 1.25 MA in a 1.20-4.80 s flat top."),
    )
    assert describe.polish(template) == (template, False)
    monkeypatch.setattr(
        describe,
        "_client_for_polish",
        lambda: _FakeClient(
            "Shot 161172 (2015-01-20) held Ip 1.21 MA over a 1.20-4.80 s flat top."
        ),
    )
    text, ok = describe.polish(template)
    assert ok is True and "1.21 MA" in text


def test_polish_is_silent_when_no_server_is_up(paths, recwarn):
    template = "Shot 161172."
    assert describe.polish(template) == (template, False)
    assert not [w for w in recwarn if "LLM" in str(w.message) or "llm" in str(w.message)]


class _FakeClient:
    def __init__(self, content):
        self._content = content

    def available(self):
        return True, ""

    def chat(self, messages, **kw):
        return Reply(content=self._content, tool_calls=[], model="fake")


def test_reasoning_effort_is_sent_only_when_configured(paths):
    """gemma4:26b answers with empty content unless reasoning is turned off, so the key has to
    reach the wire when llm.yaml sets it -- and stay off it for a provider that has no such field."""
    calls: list = []
    c = LLMClient(
        {**CFG, "base_url": "http://llm.test", "reasoning_effort": "none"},
        paths,
        transport=_transport(calls, _completion("ready")),
    )
    c.chat([{"role": "user", "content": "hi"}])
    assert calls[0]["reasoning_effort"] == "none"

    calls2: list = []
    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"},
        paths,
        transport=_transport(calls2, _completion("ready")),
    )
    c.chat([{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in calls2[0]


def test_loaded_lists_the_tags_ollama_holds_in_memory(paths):
    def handler(request):
        assert request.url.path == "/api/ps" and request.method == "GET"
        return httpx.Response(
            200, json={"models": [{"name": "gemma4:26b", "model": "gemma4:26b", "size": 1}]}
        )

    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"}, paths, transport=httpx.MockTransport(handler)
    )
    assert c.loaded() == ["gemma4:26b"]


def test_loaded_is_empty_and_silent_when_the_call_fails_or_there_is_no_endpoint(paths):
    def handler(request):
        return httpx.Response(500, text="nope")

    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"}, paths, transport=httpx.MockTransport(handler)
    )
    assert c.loaded() == []
    assert LLMClient(CFG, paths).loaded() == []  # no endpoint file in tmp_path


def test_loaded_opens_no_socket_when_the_provider_is_off(paths):
    def handler(request):
        raise AssertionError("provider off must not connect")

    c = LLMClient({**CFG, "provider": "off"}, paths, transport=httpx.MockTransport(handler))
    assert c.loaded() == []


def test_loaded_is_empty_when_the_body_is_valid_json_but_not_an_object(paths):
    def handler(request):
        return httpx.Response(200, json=[])

    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"}, paths, transport=httpx.MockTransport(handler)
    )
    assert c.loaded() == []


def test_loaded_is_empty_when_models_is_null(paths):
    def handler(request):
        return httpx.Response(200, json={"models": None})

    c = LLMClient(
        {**CFG, "base_url": "http://llm.test"}, paths, transport=httpx.MockTransport(handler)
    )
    assert c.loaded() == []
