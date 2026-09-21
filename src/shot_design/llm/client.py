"""One client for every model call in the project.

OpenAI chat-completions shape over httpx, so Ollama today and a hosted provider later are the
same code path. Discovery goes through configs/shot_design/llm.yaml: an explicit base_url, else the
endpoint file the serving script writes under the data root (so moving Ollama between the vis node
and a Slurm GPU changes nothing here). Deterministic requests (temperature 0) are cached on disk by
request hash; sampled ones never are. Every failure surfaces as LLMUnavailable carrying the
command that would fix it, and `provider: off` opens no socket at all.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from shot_design import config
from shot_design.config import Paths


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
        "no language model is configured: start it with "
        "sbatch scripts/shot_design/serve_llm.sbatch, set base_url in configs/shot_design/llm.yaml "
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
        self.cfg = dict(cfg or config.load_yaml("llm.yaml"))
        # SHOT_DESIGN_LLM_PROVIDER overrides llm.yaml's `provider` for one process. The
        # Frontier build wrapper sets it to "off": a compute node has the agy binary on the
        # shared filesystem, so `available()` would say yes, but it has no route to Google
        # and every blurb then waits ~50 s for an auth timeout. Blurbs are backfilled on the
        # login node by `shot_design blurb` instead. A blank value is no override.
        override = os.environ.get("SHOT_DESIGN_LLM_PROVIDER", "").strip()
        if override:
            self.cfg["provider"] = override
        self.paths = paths or config.load_paths()
        self._transport = transport
        self._ep_cache: tuple[int, Endpoint | None] | None = None  # (mtime_ns, endpoint)
        # Local import: shot_design.llm.agy imports Reply/ToolCall/LLMUnavailable back
        # from this module, so importing it at module scope would be circular.
        from shot_design.llm import agy

        # Every provider takes the already-built request `body` and returns a Reply;
        # `chat` owns discovery, cache and body-building so a provider only has to speak
        # its own transport. "ollama" is also the fallback for any unregistered provider
        # string -- the shape any such OpenAI-compatible server is expected to speak.
        self._providers: dict = {
            "ollama": lambda body: self._chat_openai(body, self.endpoint()),
            # A hosted OpenAI-compatible server is the same wire shape as Ollama, just a
            # different base_url/API key -- see the module docstring.
            "openai_compatible": lambda body: self._chat_openai(body, self.endpoint()),
            "agy": agy.AgyProvider(self.cfg.get("agy", {})).chat,
        }

    # ------------------------------------------------------------------ discovery

    @property
    def off(self) -> bool:
        return str(self.cfg.get("provider", "off")).strip().lower() in ("", "off", "none", "false")

    def _ollama_cfg(self, key: str, default=None):
        """`base_url`/`endpoint_file`/`ollama_*_dir` moved under an `ollama:` block; a
        caller that still passes a flat cfg dict (an unmigrated script, this project's
        own tests) is read the same way for one release. A nested key that is present
        but null (llm.yaml sets every Frontier-absent path that way) falls through
        to the flat key too, so overriding only the flat key still takes effect."""
        value = (self.cfg.get("ollama") or {}).get(key)
        return value if value is not None else self.cfg.get(key, default)

    def endpoint_path(self) -> Path:
        rel = self._ollama_cfg("endpoint_file", "llm/endpoint.json")
        return self.paths.data_root / str(rel)

    def endpoint(self) -> Endpoint | None:
        if self.off:
            return None
        base_url = self._ollama_cfg("base_url")
        if base_url:
            return Endpoint(url=str(base_url).rstrip("/"))
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
            return False, "configs/shot_design/llm.yaml has provider: off"
        provider = str(self.cfg.get("provider", "")).strip().lower()
        if provider == "agy":
            # agy is a CLI, not a server: "available" means the binary is on PATH, not
            # that any endpoint file/base_url exists -- those are ollama-only concepts.
            bin_name = str((self.cfg.get("agy") or {}).get("bin", "agy"))
            if shutil.which(bin_name) is None:
                return False, (
                    f"the agy CLI ({bin_name!r}) is not installed or not on PATH; "
                    "load/install it or set agy.bin in configs/shot_design/llm.yaml"
                )
            return True, ""
        if provider not in self._providers:
            # Mirrors chat()'s own refusal: a typo'd provider must not report
            # "available" off the back of an unrelated ollama endpoint file, then
            # raise a differently-worded LLMUnavailable the first time it is used.
            return False, (
                f"unknown llm provider {provider!r}; known providers: "
                f"{', '.join(sorted(self._providers))}"
            )
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
        models = self.cfg.get("models", {})
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
            raise LLMUnavailable("configs/shot_design/llm.yaml has provider: off")
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
        # empty content (configs/shot_design/llm.yaml says what was measured); absent or null,
        # nothing is sent.
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
        provider = str(self.cfg.get("provider", "")).strip().lower()
        try:
            provider_fn = self._providers[provider]
        except KeyError:
            # A typo'd provider must not silently fall back to ollama and print a
            # "start an Ollama server" hint that is wrong for whatever was actually
            # misconfigured.
            raise LLMUnavailable(
                f"unknown llm provider {provider!r}; known providers: "
                f"{', '.join(sorted(self._providers))}"
            ) from None
        reply = provider_fn(body)
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".json.part")
            tmp.write_text(reply.model_dump_json(), encoding="utf-8")
            os.replace(tmp, cache_file)
        return reply

    def _chat_openai(self, body: dict, ep: Endpoint | None) -> Reply:
        if ep is None:
            raise LLMUnavailable(start_hint(self.paths))
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
            return _parse(data, body["model"])
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as e:
            raise LLMUnavailable(
                f"language model at {ep.url} returned an unusable response: "
                f"malformed completion body ({e})"
            ) from e


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
