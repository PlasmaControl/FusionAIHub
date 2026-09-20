### Task E2: `agy` provider with emulated tool calls

**Files:**
- Create: `src/shot_design/llm/agy.py`
- Modify: `src/shot_design/llm/client.py` (`_providers["agy"] = agy.AgyProvider(self.cfg["agy"]).chat`)
- Test: `tests/shot_design/test_llm_agy.py`

**Interfaces:**
- `AgyProvider(cfg: dict, runner: Callable[[list[str], str], str] | None = None)`; `chat(body: dict) -> Reply`.
- `render_prompt(messages, tools) -> str`: system/user/assistant/tool messages rendered as labelled sections; when `tools` is present appends "You may call tools. Reply with JSON matching the schema; put a plain answer in `content` and zero or more tool calls in `tool_calls`." and each tool's name, description and JSON parameters.
- `SCHEMA = {"type": "object", "properties": {"content": {"type": "string"}, "tool_calls": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}}}, "required": ["content", "tool_calls"]}`
- Command: `[bin, "--model", model, "--output-format", "json", "--json-schema", json.dumps(SCHEMA), *extra_args, f"-p={prompt}"]` — `-p` MUST be the last argument and attached with `=`: `agy -p --model x` takes `--model` as the prompt (measured 2026-09-19). `model` is the resolved id (`client.model(body["model"])`, e.g. `gemini-3.8-flash-low`); `runner` defaults to `subprocess.run(..., capture_output=True, text=True, timeout=timeout_s)`; stdout is JSON — an object with `status`, `response`, `structured_output`, `usage`, `duration_seconds`. Prefer `structured_output` (a dict); if its `content` is a string that itself `json.loads` to a dict carrying `content`, unwrap that one level (measured: `{"content": "{\"content\": \"ready\"}"}`); else fall back to the first JSON object inside `response` (strip ``` fences); a `status` other than `SUCCESS` is an `AgyError`; wrap into `Reply(content, tool_calls=[ToolCall(id=f"call_{i}", name, arguments)], model=model or "agy")`.
- Errors → `LLMUnavailable` with the stderr tail (non-zero exit, timeout, unparsable JSON).

- [ ] **Step 1: Failing tests**

```python
import json, pytest
from shot_design.llm import agy
from shot_design.llm.client import LLMUnavailable

TOOLS = [{"type": "function", "function": {"name": "search_shots", "description": "d", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]

def test_render_prompt_lists_tools_and_roles():
    p = agy.render_prompt([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], TOOLS)
    assert "S" in p and "U" in p and "search_shots" in p and "tool_calls" in p

def test_chat_parses_structured_output_into_tool_calls():
    seen = {}
    def runner(cmd, prompt):
        seen["cmd"] = cmd
        return json.dumps({"result": json.dumps({"content": "", "tool_calls": [{"name": "search_shots", "arguments": {"q": "ELM"}}]})})
    r = agy.AgyProvider({"bin": "agy", "effort": "low", "timeout_s": 5}, runner=runner).chat({"messages": [{"role": "user", "content": "x"}], "tools": TOOLS, "model": "quality"})
    assert r.tool_calls[0].name == "search_shots" and r.tool_calls[0].arguments == {"q": "ELM"}
    assert "--json-schema" in seen["cmd"] and seen["cmd"][-1].startswith("-p=")

def test_over_encoded_structured_output_is_unwrapped():
    def runner(cmd, prompt):
        return json.dumps({"status": "SUCCESS", "response": "", "structured_output": {"content": json.dumps({"content": "ready", "tool_calls": []})}})
    r = agy.AgyProvider({"bin": "agy"}, runner=runner).chat({"messages": [{"role": "user", "content": "x"}], "model": "gemini-3.8-flash-low"})
    assert r.content == "ready" and r.tool_calls == []

def test_nonzero_exit_is_unavailable():
    def runner(cmd, prompt): raise agy.AgyError("exit 2: boom")
    with pytest.raises(LLMUnavailable, match="boom"):
        agy.AgyProvider({"bin": "agy"}, runner=runner).chat({"messages": [], "model": "m"})
```

- [ ] **Step 2–4:** fail → implement → pass.

- [ ] **Step 5: Live smoke (login node)**

```bash
pixi run --frozen -e shot-design-frontier python -c "
from shot_design.llm.client import LLMClient
r = LLMClient().chat([{'role':'user','content':'Reply with the single word ready.'}], cache=False); print(repr(r.content))"
```
Expected: a reply containing `ready`. If `agy -p` needs a project/agent flag on this machine, add it to `extra_args` in `llm.yaml` and note it in the docs.

- [ ] **Step 6: De-Gemma the assistant** — in `src/shot_design/design/assistant.py` replace the `"gemma" not in model_name.lower()` guard with `if client.cfg.get("provider") == "off": raise LLMUnavailable("The design assistant needs a configured model (llm.yaml provider)")`; change every user-facing "Gemma" in the stage labels, error strings and system prompts to "the model" / `model_name`; do the same in `ui/assistant_routes.py`, `ui/static/index.html` (grep -i gemma). Update the tests that assert on those strings. Commit `git commit -m "shot_design: assistant is model-neutral (Gemini Flash via agy)"`.

- [ ] **Step 7: Commit** `git commit -m "shot_design llm: agy (Antigravity CLI) provider with schema-emulated tool calls"`

