"""`shot_design llm` on a provider with no HTTP endpoint (agy)."""

import argparse
import json

from shot_design import cli
from shot_design.llm import client as client_mod


class _AgyClient:
    cfg = {
        "provider": "agy",
        "agy": {"bin": "agy"},
        "models": {"quality": "gemini-3.8-flash-high", "fast": "gemini-3.8-flash-low"},
        "default": "quality",
    }

    def __init__(self, *a, **kw):
        pass

    def available(self):
        return True, ""

    def endpoint(self):
        return None  # agy is a subprocess, not a served endpoint

    def model(self, key=None):
        return self.cfg["models"][key or self.cfg["default"]]


def test_llm_status_names_the_provider_when_there_is_no_endpoint(monkeypatch, capsys):
    # Before the fix a healthy agy config raised AttributeError on `ep.url` (ep is None).
    monkeypatch.setattr(client_mod, "LLMClient", _AgyClient)
    assert cli.cmd_llm(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "agy" in out and "gemini-3.8-flash-high" in out


class _NullProviderBlockClient(_AgyClient):
    """`llm.yaml` writes an absent provider block as a null value (the same style
    `ollama:` uses), not as an omitted key -- `cfg.get(provider, {})` only covers the
    omitted case."""

    cfg = {**_AgyClient.cfg, "agy": None}


def test_llm_status_does_not_crash_when_the_provider_block_is_null(monkeypatch, capsys):
    # Before the fix, `client.cfg.get(provider, {}).get("bin", provider)` raised
    # AttributeError on `None.get(...)` because "agy" IS present in cfg, just null.
    monkeypatch.setattr(client_mod, "LLMClient", _NullProviderBlockClient)
    assert cli.cmd_llm(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "bin agy" in out


def test_provider_block_models_override_the_top_level_map(paths):
    from shot_design.llm.client import LLMClient

    cfg = {
        "provider": "ollama",
        "models": {"quality": "gemini-3.8-flash-high", "fast": "gemini-3.8-flash-low"},
        "ollama": {"models": {"quality": "gemma4:26b", "fast": "gemma4:e4b"}},
    }
    client = LLMClient(cfg=cfg, paths=paths)
    assert client.model("quality") == "gemma4:26b"
    assert client.model("fast") == "gemma4:e4b"
    cfg["provider"] = "agy"
    assert LLMClient(cfg=cfg, paths=paths).model("quality") == "gemini-3.8-flash-high"


def test_provider_block_reasoning_effort_is_sent_only_for_that_provider(paths):
    import httpx

    from shot_design.llm.client import LLMClient

    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    cfg = {
        "provider": "ollama",
        "cache": False,
        "models": {"quality": "gemini-3.8-flash-high"},
        "ollama": {
            "base_url": "http://llm.test",
            "models": {"quality": "gemma4:26b"},
            "reasoning_effort": "none",
        },
    }
    client = LLMClient(cfg=cfg, paths=paths, transport=httpx.MockTransport(handler))
    client.chat([{"role": "user", "content": "hi"}])
    assert seen["body"]["model"] == "gemma4:26b"
    assert seen["body"]["reasoning_effort"] == "none"
