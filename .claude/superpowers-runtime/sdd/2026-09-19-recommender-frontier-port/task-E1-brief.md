### Task E1: Provider seam in `LLMClient`

**Files:**
- Modify: `src/shot_design/llm/client.py` (`chat` → dispatch on `self.cfg["provider"]`; the current HTTP body becomes `_chat_openai(body, ep)`), `configs/shot_design/llm.yaml`
- Test: `tests/shot_design/test_llm_client.py` (existing tests must still pass unchanged)

**Interfaces:**
- `LLMClient.chat(...)` keeps its signature and `Reply` return. Providers: `ollama` (today's path), `agy` (Task E2), `off`.
- `llm.yaml`: `provider: agy`, new block

```yaml
agy:
  bin: agy
  timeout_s: 300
  retries: 2
  extra_args: ["--disable-slash-commands"]
models:                    # replaces the gemma4 entries; `agy models` lists the ids
  quality: gemini-3.8-flash-high
  fast: gemini-3.8-flash-low
default: quality
blurb:
  model: fast
ollama:
  base_url: null
  endpoint_file: llm/endpoint.json
  ollama_bin_dir: /lustre/orion/fus187/proj-shared/nchen/ollama/bin
  ollama_models_dir: /lustre/orion/fus187/proj-shared/nchen/ollama/models
  ollama_home_dir: /lustre/orion/fus187/proj-shared/nchen/ollama/home
```
Keep the existing top-level `ollama_*`/`base_url`/`endpoint_file` keys readable for one release (`self.cfg.get("ollama", {}).get(k, self.cfg.get(k))`). Drop `reasoning_effort` from the yaml (it was a Gemma workaround); the `agy` provider passes no `--effort` flag (the model id carries the effort level).

- [ ] **Step 1: Failing test**

```python
def test_chat_dispatches_to_a_registered_provider(paths):
    from shot_design.llm.client import LLMClient, Reply
    c = LLMClient(cfg={"provider": "fake", "cache": False}, paths=paths)
    c._providers["fake"] = lambda body: Reply(content="hi", model="fake")
    assert c.chat([{"role": "user", "content": "x"}]).content == "hi"
```

- [ ] **Step 2–4:** fail → refactor `chat` so it builds `body`, handles the cache, then `reply = self._providers[provider](body)`; `_providers = {"ollama": self._chat_openai}` in `__init__` → pass (whole `test_llm_client.py`).

- [ ] **Step 5: Commit** `git commit -m "shot_design llm: provider dispatch seam; ollama block in llm.yaml"`

