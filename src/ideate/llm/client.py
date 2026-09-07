"""One client for every model call in the project.

OpenAI chat-completions shape over httpx, so Ollama today and a hosted provider later are the
same code path. Discovery goes through configs/llm.yaml: an explicit base_url, else the endpoint
file the serving script writes under the data root (so moving Ollama between the vis node and a
Slurm GPU changes nothing here). Deterministic requests (temperature 0) are cached on disk by
request hash; sampled ones never are. Every failure surfaces as LLMUnavailable carrying the
command that would fix it, and `provider: off` opens no socket at all.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from ideate import config
from ideate.config import Paths


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class Reply(BaseModel):
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str
    cached: bool = False


class Endpoint(BaseModel):
    url: str
    models: list[str] = Field(default_factory=list)
    host: str | None = None
    job_id: str | None = None
    started: str | None = None


class LLMUnavailable(RuntimeError):
    def __init__(self, hint: str):
        super().__init__(hint)
        self.hint = hint


def start_hint(paths: Paths) -> str:
    return (
        "no language model is configured: set base_url in configs/ideate/llm.yaml "
        "to an OpenAI-compatible server, or write its URL to "
        f"{paths.data_root / 'llm' / 'endpoint.json'}"
    )


class LLMClient:
    def __init__(
        self,
        cfg: dict | None = None,
        paths: Paths | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.cfg = cfg or config.load_yaml("llm.yaml")
        self.paths = paths or config.load_paths()
        self._transport = transport
        self._ep_cache: tuple[int, Endpoint | None] | None = None  # (mtime_ns, endpoint)

    # ------------------------------------------------------------------ discovery

    @property
    def off(self) -> bool:
        return str(self.cfg.get("provider", "off")).strip().lower() in ("", "off", "none", "false")

    def endpoint_path(self) -> Path:
        return self.paths.data_root / str(self.cfg.get("endpoint_file", "llm/endpoint.json"))

    def endpoint(self) -> Endpoint | None:
        if self.off:
            return None
        if self.cfg.get("base_url"):
            return Endpoint(url=str(self.cfg["base_url"]).rstrip("/"))
        p = self.endpoint_path()
        try:
            mtime = p.stat().st_mtime_ns
        except OSError:
            self._ep_cache = None
            return None
        if self._ep_cache and self._ep_cache[0] == mtime:
            return self._ep_cache[1]
        try:
            ep = Endpoint.model_validate_json(p.read_text(encoding="utf-8"))
            ep.url = ep.url.rstrip("/")
        except Exception:  # noqa: BLE001 — a half-written file while the script is starting up
            ep = None
        self._ep_cache = (mtime, ep)
        return ep

    def available(self) -> tuple[bool, str]:
        if self.off:
            return False, "configs/llm.yaml has provider: off"
        ep = self.endpoint()
        if ep is None:
            return False, start_hint(self.paths)
        return True, ""

    def loaded(self) -> list[str]:
        """The model tags the server holds in memory (Ollama's `GET /api/ps`), for the status
        line only: [] when the provider is off, there is no endpoint, the server is not Ollama or
        the call fails. Never raises and never counts as unavailability."""
        ep = self.endpoint()
        if ep is None:
            return []
        try:
            with httpx.Client(
                transport=self._transport, timeout=httpx.Timeout(3.0, connect=2.0)
            ) as http:
                r = http.get(f"{ep.url}/api/ps")
            if r.status_code != 200:
                return []
            body = r.json()
            models = body.get("models") if isinstance(body, dict) else None
            if not isinstance(models, list):
                models = []
        except Exception:  # noqa: BLE001 — a status decoration only: any failure here means "nothing known"
            return []
        return [
            str(m.get("name") or m.get("model"))
            for m in models
            if isinstance(m, dict) and (m.get("name") or m.get("model"))
        ]

    def model(self, key: str | None = None) -> str:
        models = self.cfg["models"]
        key = key or self.cfg.get("default", "quality")
        return str(models.get(key, key))  # an unknown key is taken as a literal tag

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool | None = None,
    ) -> Reply:
        if self.off:
            raise LLMUnavailable("configs/llm.yaml has provider: off")
        ep = self.endpoint()
        if ep is None:
            raise LLMUnavailable(start_hint(self.paths))
        temperature = float(
            self.cfg.get("temperature", 0.0) if temperature is None else temperature
        )
        body: dict = {
            "model": self.model(model),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": int(max_tokens or self.cfg.get("max_tokens", 600)),
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        # A reasoning model otherwise spends the whole max_tokens budget thinking and returns
        # empty content (configs/llm.yaml says what was measured); absent or null, nothing is sent.
        if self.cfg.get("reasoning_effort") is not None:
            body["reasoning_effort"] = self.cfg["reasoning_effort"]
        use_cache = (self.cfg.get("cache", True) if cache is None else cache) and temperature == 0.0
        key = (
            hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            if use_cache
            else None
        )
        cache_file = self.paths.llm_cache_dir / "chat" / f"{key}.json" if key else None
        if cache_file and cache_file.exists():
            reply = Reply.model_validate_json(cache_file.read_text(encoding="utf-8"))
            reply.cached = True
            return reply
        headers = {"content-type": "application/json"}
        if os.environ.get("LLM_API_KEY"):
            headers["authorization"] = f"Bearer {os.environ['LLM_API_KEY']}"
        timeout = httpx.Timeout(float(self.cfg.get("timeout_s", 120)), connect=5.0)
        try:
            with httpx.Client(transport=self._transport, timeout=timeout) as http:
                r = http.post(f"{ep.url}/v1/chat/completions", json=body, headers=headers)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise LLMUnavailable(
                f"cannot reach the language model at {ep.url} ({e}); " + start_hint(self.paths)
            ) from e
        except httpx.HTTPError as e:
            raise LLMUnavailable(f"cannot reach the language model at {ep.url}: {e}") from e
        if r.status_code != 200:
            raise LLMUnavailable(
                f"language model at {ep.url} answered {r.status_code}: {r.text[:200]}"
            )
        try:
            data = r.json()
        except ValueError as e:
            raise LLMUnavailable(
                f"language model at {ep.url} returned an unusable response: body is not JSON ({e})"
            ) from e
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            msg = err.get("message") if isinstance(err, dict) else None
            raise LLMUnavailable(
                f"language model at {ep.url} returned an unusable response: {msg or err}"
            )
        try:
            reply = _parse(data, body["model"])
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as e:
            raise LLMUnavailable(
                f"language model at {ep.url} returned an unusable response: "
                f"malformed completion body ({e})"
            ) from e
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".json.part")
            tmp.write_text(reply.model_dump_json(), encoding="utf-8")
            os.replace(tmp, cache_file)
        return reply


def _parse(data: dict, model: str) -> Reply:
    choices = data.get("choices")
    if choices is None:
        choices = [{}]
    if not isinstance(choices, list):
        raise ValueError("'choices' is not a list")  # noqa: TRY004 — malformed response uses the ValueError contract
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    calls: list[ToolCall] = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        raw = fn.get("arguments", "{}")
        args: dict
        if isinstance(raw, dict):
            args = raw
        else:
            try:
                parsed = json.loads(raw or "{}")
                args = (
                    parsed
                    if isinstance(parsed, dict)
                    else {"_raw": raw, "_error": "arguments are not a JSON object"}
                )
            except json.JSONDecodeError:
                args = {"_raw": raw, "_error": "arguments are not a JSON object"}
        calls.append(
            ToolCall(
                id=str(tc.get("id") or f"call_{i}"), name=str(fn.get("name", "")), arguments=args
            )
        )
    return Reply(
        content=msg.get("content") or "", tool_calls=calls, model=str(data.get("model") or model)
    )
