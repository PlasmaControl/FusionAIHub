"""Claude Code CLI provider (`claude -p`), the Claude sibling of the agy provider.

House rule (owner, 2026-09-21): agy is for Gemini only, Claude goes through Claude Code
itself, GPT through Codex. So when the Gemini quota is spent on Stellar, the assistant
runs on Claude by shelling out to the `claude` CLI in --print mode, exactly as the agy
provider shells out to `agy`: the chat is rendered to one text prompt, the CLI is asked
for structured output against the same {content, tool_calls} schema, and the reply's
`structured_output` (or a JSON object inside `result`) is decoded into a Reply.

Per-call overhead is what the CLI loads into its context, so `extra_args` in llm.yaml
strips it: no tools, no MCP servers, only the user's own settings (auth), no session
file. Measured 2026-09-21: 12.7 k cached system tokens per call vs 54 k with the
defaults from inside this repo. The CLAUDECODE variables are removed from the child's
environment because a headless `claude` refuses to start inside another Claude Code
session otherwise.
"""

from __future__ import annotations

import json
import os
import subprocess

from .agy import (
    SCHEMA,
    AgyError,
    AgyProvider,
    _decode,
    _fenced_json_object,
)
from .client import Reply, ToolCall

_NESTED_SESSION_VARS = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")

DEFAULT_EXTRA_ARGS = (
    "--tools",
    "none",
    "--strict-mcp-config",
    "--setting-sources",
    "user",
    "--no-session-persistence",
)


def _extract_structured(payload: dict) -> dict:
    """`structured_output` when the CLI honoured --json-schema, else the first JSON
    object inside the plain `result` text."""
    structured = payload.get("structured_output")
    if isinstance(structured, dict) and "content" in structured:
        return structured
    result = payload.get("result")
    if isinstance(result, str):
        parsed = _fenced_json_object(result)
        if isinstance(parsed, dict) and "content" in parsed:
            return parsed
    detail = json.dumps(payload)[:2000]
    raise AgyError(f"no usable structured output in claude's reply: {detail}")


def parse_output(raw: str, model: str) -> Reply:
    raw = raw.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise AgyError(f"claude printed unparsable output: {raw[-4000:]}") from e
    if not isinstance(payload, dict):
        raise AgyError(f"claude printed unparsable output: {raw[-4000:]}")
    if payload.get("is_error") or payload.get("subtype", "success") != "success":
        # `result` carries the CLI's own message ("Not logged in", an API error, ...)
        raise AgyError(
            f"claude {payload.get('subtype', 'error')}: "
            f"{str(payload.get('result') or '')[:500]}"
        )
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
    return Reply(
        content=str(structured.get("content") or ""),
        tool_calls=calls,
        model=model or "claude",
    )


def is_retryable(message: str) -> bool:
    if message.startswith("claude timed out after"):
        return True
    if message.startswith("exit ") and message.split(":", 1)[-1].strip() == "":
        return True
    lowered = message.lower()
    # Rate limits and overloads clear on their own; a login failure or a refused
    # prompt does not.
    return any(
        marker in lowered
        for marker in ("rate_limit", "rate limit", "overloaded", "529", "429")
    )


class ClaudeCliProvider(AgyProvider):
    """`claude -p` behind the agy provider's render/launch/retry loop."""

    def _command(self, model: str, prompt: str) -> list[str]:
        extra = self.cfg.get("extra_args")
        extra = list(DEFAULT_EXTRA_ARGS) if extra is None else [str(a) for a in extra]
        return [
            str(self.cfg.get("bin", "claude")),
            "-p",
            "--model",
            model,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(SCHEMA),
            *extra,
            prompt,
        ]

    def _parse(self, raw: str, model: str) -> Reply:
        return parse_output(raw, model)

    def _retryable(self, message: str) -> bool:
        return is_retryable(message)

    def _backoff_s(self, message: str, attempt: int) -> float:
        lowered = message.lower()
        if any(
            m in lowered
            for m in ("rate_limit", "rate limit", "overloaded", "529", "429")
        ):
            return float(self.cfg.get("backoff_s", 20.0)) * (2**attempt)
        return 0.0

    def _run_subprocess(self, cmd: list[str], prompt: str) -> str:
        del prompt
        timeout_s = float(self.cfg.get("timeout_s", 300))
        env = {k: v for k, v in os.environ.items() if k not in _NESTED_SESSION_VARS}
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s, env=env
            )
        except subprocess.TimeoutExpired as e:
            tail = _decode(e.stderr) or _decode(e.stdout)
            detail = f": {tail}" if tail else ""
            raise AgyError(f"claude timed out after {timeout_s:.0f}s{detail}") from e
        except OSError as e:
            raise AgyError(f"could not launch claude ({cmd[0]!r}): {e}") from e
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-4000:]
            raise AgyError(f"exit {result.returncode}: {tail}")
        return result.stdout
