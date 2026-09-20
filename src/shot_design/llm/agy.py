"""agy (Antigravity CLI, Gemini Flash) provider.

There is no OpenAI-compatible endpoint on Frontier -- the only model is Gemini Flash
through the `agy` CLI, run once per chat turn in --print mode. The client's
messages/tools are rendered into a single text prompt; a JSON schema on the CLI
invocation asks agy to emulate the OpenAI tool-calling shape (a `content` string plus
a `tool_calls` array) since agy itself has no notion of the caller's tool contract.
agy's own stdout is JSON-wrapped around that structured reply, and has been observed
encoding it one level too deep, so decoding is defensive.

Ported from shot-recommender-system (shotrec) @565d548 (this provider is new for the
Frontier port; there is no prior Ollama-only equivalent).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable

from .client import LLMUnavailable, Reply, ToolCall

SCHEMA = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    },
    "required": ["content", "tool_calls"],
}


class AgyError(RuntimeError):
    """The agy subprocess failed, timed out, or answered something unusable."""


def render_prompt(messages: list[dict], tools: list[dict] | None = None) -> str:
    """One text prompt: each message as a labelled section, tools appended as a
    schema-emulation instruction plus their name/description/parameters."""
    sections = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = message.get("content", "")
        if message.get("role") == "tool":
            label = message.get("name") or message.get("tool_call_id") or "result"
            sections.append(f"TOOL RESULT ({label}): {content}")
        else:
            sections.append(f"{role}: {content}")
    if tools:
        lines = [
            "You may call tools. Reply with JSON matching the schema; put a plain "
            "answer in `content` and zero or more tool calls in `tool_calls`."
        ]
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            name = fn.get("name", "")
            description = fn.get("description", "")
            parameters = fn.get("parameters", {})
            lines.append(f"- {name}: {description} parameters={json.dumps(parameters)}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _fenced_json_object(text: str):
    """The first JSON object in `text`, stripping a ``` fence if the whole text is
    one."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) > 1 and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _extract_structured(payload: dict) -> dict:
    """A dict with `content` and `tool_calls`, however deep agy nested/encoded it:
    prefer `structured_output`, unwrapping one level of JSON-string-in-a-`content`
    field if present; else a `result` string holding the whole thing JSON-encoded;
    else the first JSON object inside `response`."""
    structured = payload.get("structured_output")
    if isinstance(structured, dict):
        content = structured.get("content")
        if isinstance(content, str):
            unwrapped = _fenced_json_object(content)
            if isinstance(unwrapped, dict) and "content" in unwrapped:
                return unwrapped
        return structured
    result = payload.get("result")
    if isinstance(result, str):
        parsed = _fenced_json_object(result)
        if isinstance(parsed, dict):
            return parsed
    response = payload.get("response")
    if isinstance(response, str):
        parsed = _fenced_json_object(response)
        if isinstance(parsed, dict):
            return parsed
    detail = json.dumps(payload)[:2000]
    raise AgyError(f"no usable structured output in agy's reply: {detail}")


def _parse_output(raw: str, model: str) -> Reply:
    raw = raw.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise AgyError(f"agy printed unparsable output: {raw[-4000:]}") from e
    if not isinstance(payload, dict):
        raise AgyError(f"agy printed unparsable output: {raw[-4000:]}")
    status = payload.get("status")
    if status is not None and status != "SUCCESS":
        raise AgyError(f"agy status {status}: {raw[-4000:]}")
    structured = _extract_structured(payload)
    calls = []
    for i, call in enumerate(structured.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        calls.append(
            ToolCall(
                id=f"call_{i}",
                name=str(call.get("name", "")),
                arguments=call.get("arguments") or {},
            )
        )
    content = str(structured.get("content") or "")
    return Reply(content=content, tool_calls=calls, model=model or "agy")


class AgyProvider:
    def __init__(
        self, cfg: dict, runner: Callable[[list[str], str], str] | None = None
    ):
        self.cfg = cfg or {}
        self._runner = runner or self._run_subprocess

    def _run_subprocess(self, cmd: list[str], prompt: str) -> str:
        # `prompt` is already the last element of `cmd`; kept as a separate argument
        # so a custom runner (tests, or a future batching runner) need not re-parse it.
        del prompt
        timeout_s = float(self.cfg.get("timeout_s", 300))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired as e:
            raise AgyError(f"agy timed out after {timeout_s:.0f}s") from e
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-4000:]
            raise AgyError(f"exit {result.returncode}: {tail}")
        return result.stdout

    def chat(self, body: dict) -> Reply:
        model = str(body.get("model") or "")
        prompt = render_prompt(body.get("messages") or [], body.get("tools"))
        cmd = [
            str(self.cfg.get("bin", "agy")),
            "--model",
            model,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(SCHEMA),
            *[str(a) for a in self.cfg.get("extra_args", [])],
            f"-p={prompt}",
        ]
        try:
            raw = self._runner(cmd, prompt)
        except AgyError as e:
            raise LLMUnavailable(str(e)) from e
        try:
            return _parse_output(raw, model)
        except AgyError as e:
            raise LLMUnavailable(str(e)) from e
