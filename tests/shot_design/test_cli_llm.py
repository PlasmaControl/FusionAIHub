"""`shot_design llm` on a provider with no HTTP endpoint (agy)."""

import argparse

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
