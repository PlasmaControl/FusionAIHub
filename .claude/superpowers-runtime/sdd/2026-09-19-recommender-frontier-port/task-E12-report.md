# Task E1 + E2 report: provider seam + agy (Antigravity CLI) provider

Worktree: `/lustre/orion/fus187/scratch/nchen/FusionAIHub-wt-e`, branch `nathan_dev-e`, base `b8ea4f4`.

Commits (in order):

```
197918d shot_design llm: provider dispatch seam; ollama block in llm.yaml          (E1)
1e70ec8 shot_design: assistant is model-neutral (Gemini Flash via agy)             (E2 step 6)
a1c0f87 shot_design llm: agy (Antigravity CLI) provider with schema-emulated tool calls  (E2 step 7)
```

E2 produced two commits, not one: the brief's Step 6 text itself says "Commit
`git commit -m \"shot_design: assistant is model-neutral (Gemini Flash via agy)\"`",
separate from Step 7's commit. I followed the brief literally and made two commits for E2,
ordered as the brief lists the steps (de-Gemma before the final agy-provider commit),
even though I discovered the de-Gemma work's test interaction only after building the
provider (narrative below).

## E1: provider dispatch seam

**Files:** `src/shot_design/llm/client.py`, `configs/shot_design/llm.yaml`,
`tests/shot_design/test_llm_client.py`, plus (unplanned, see Adaptations)
`tests/shot_design/test_blurb.py`, `tests/shot_design/test_blurb_provenance.py`.

### What changed

- `LLMClient.__init__` now builds `self._providers: dict`, seeded with
  `"ollama": lambda body: self._chat_openai(body, self.endpoint())`.
- `chat()` still builds `body`, resolves the cache, but now dispatches via
  `self._providers.get(provider, self._providers["ollama"])(body)` instead of
  calling the HTTP path inline. The old inline HTTP/parsing code became
  `_chat_openai(self, body, ep)`, unchanged apart from taking `ep` as a parameter
  and no longer owning the cache.
- `model()` reads `self.cfg.get("models", {})` instead of `self.cfg["models"]` (see
  Adaptations — required by the brief's own Step-1 test).
- New `_ollama_cfg(key, default)` helper: reads `cfg["ollama"][key]`, falling back to
  the flat top-level `cfg[key]` when the nested value is absent **or null** (see
  Adaptations — the null case was needed for an existing test to keep passing).
  `endpoint_path()`/`endpoint()` now go through it for `endpoint_file`/`base_url`.
- `configs/shot_design/llm.yaml`: `provider: agy` is now the default; a new `agy:` block
  (`bin`, `timeout_s: 300`, `retries: 2`, `extra_args: ["--disable-slash-commands"]`);
  `models:` now holds `gemini-3.8-flash-high`/`gemini-3.8-flash-low`; `blurb.model` moved
  from `quality` to `fast`; `reasoning_effort` removed (Gemma-only workaround); the
  Stellar-only keys (`base_url`, `endpoint_file`, `ollama_bin_dir`, `ollama_models_dir`,
  `ollama_home_dir`) moved under a new `ollama:` block, with the three `ollama_*_dir`
  values set to `null` with a one-line "not installed on Frontier; Stellar-only" comment
  each (ruling 1), and the old `/scratch/gpfs/...` paths removed entirely. Header comment
  rewritten to describe Gemini Flash via agy.

### Design decision: permissive provider fallback

The brief's pseudocode was `reply = self._providers[provider](body)` (strict indexing).
I used `.get(provider, self._providers["ollama"])` instead: any provider string that
isn't explicitly registered (not just "ollama") falls back to the OpenAI-compatible HTTP
path. This was necessary because `tests/shot_design/test_llm_client.py`'s module-level
`CFG` constant uses `"provider": "openai_compatible"`, a value that predates any
dispatch and was never meant to be a routed key — before E1 the provider string was
inert except for the `off` check. Strict indexing would `KeyError` on every one of the
~15 tests built on that constant. The permissive default also matches the module's own
docstring ("Ollama today and a hosted provider later are the same code path") and keeps
the change additive: registering "agy" in E2 is the only thing that changes behavior for
`provider: agy` configs; everything else keeps working exactly as before.

## E2: agy provider

**Files:** `src/shot_design/llm/agy.py` (new), `src/shot_design/llm/client.py`
(registration), `tests/shot_design/test_llm_agy.py` (new), plus Step 6's
`src/shot_design/design/assistant.py`, `src/shot_design/ui/assistant_routes.py`,
`src/shot_design/ui/static/index.html`, `tests/shot_design/test_assistant.py`, and
(unplanned, see Adaptations) another `tests/shot_design/test_blurb.py` fix.

### `agy.py`

- `SCHEMA`, `render_prompt(messages, tools)` exactly as specified: labelled
  `ROLE: content` sections (a `tool` message becomes `TOOL RESULT (name): content`),
  and when `tools` is given, the instruction sentence plus each tool's
  name/description/JSON parameters.
- `AgyProvider(cfg, runner=None)`; `chat(body)` builds
  `[bin, "--model", model, "--output-format", "json", "--json-schema", <SCHEMA json>,
  *extra_args, f"-p={prompt}"]` (no `--effort` — the model id carries it), calls
  `runner(cmd, prompt)` (default: `subprocess.run(cmd, capture_output=True, text=True,
  timeout=timeout_s)`), and maps every failure mode to `LLMUnavailable`.
- Output decoding (`_parse_output` / `_extract_structured`) handles, in order:
  `structured_output` as a dict, unwrapping one level if its `content` is itself a
  JSON string carrying `{"content": ..., "tool_calls": ...}`; else a top-level
  `result` string that JSON-decodes straight to the shaped dict (this is what the
  brief's own Step-1 test actually sends — see Adaptations); else the first JSON
  object found inside `response` (``` fences stripped). A `status` key present and
  not `"SUCCESS"` raises; a `status` key absent is treated as success (needed for the
  `result`-shaped payload, which the brief's test never gives a `status`).
- `client.py`: `__init__` now does a **local** `from shot_design.llm import agy`
  (agy.py imports `Reply`/`ToolCall`/`LLMUnavailable` back from client.py, so a
  module-level import would be circular) and adds
  `"agy": agy.AgyProvider(self.cfg.get("agy", {})).chat` to `self._providers`
  (`.get(..., {})` rather than the brief's `self.cfg["agy"]` — see Adaptations).

### Live smoke (Step 5, ruling 2)

Ruling 2 said `agy` works and to run the smoke for real, using `gemini-3.8-flash-low`,
and to record command/timing/reply. The worktree has no pixi env, so I ran the same
frozen interpreter used for tests instead of `pixi run --frozen`:

```
cd /lustre/orion/fus187/scratch/nchen/FusionAIHub-wt-e
PYTHONPATH=$PWD/src /lustre/orion/fus187/scratch/nchen/FusionAIHub/.pixi/envs/shot-design-frontier/bin/python -c "
from shot_design.llm.client import LLMClient
c = LLMClient()
print('provider:', c.cfg.get('provider'), 'model:', c.model('fast'))
r = c.chat([{'role':'user','content':'Reply with the single word ready.'}], model='fast', cache=False)
print(repr(r.content))
print('reply.model:', r.model)
"
```

Output:
```
provider: agy model: gemini-3.8-flash-low
'ready'
reply.model: gemini-3.8-flash-low

real  0m5.700s
```

No extra flags were needed beyond what's already in `llm.yaml`'s `agy.extra_args`
(`--disable-slash-commands`); `extra_args` was not changed for the smoke.

### Step 6: de-Gemma

In `assistant.py`: module docstring, `STAGES` labels, the JSON-repair error, the
"empty time window"/"outside references"/"duplicate shots"/"average without
comparisons"/"channel without measurements" `ValueError`s, the `notes=` provenance
string, and the planning system prompt all had "Gemma" replaced with "the model" /
`model_name` (or removed where the sentence didn't need it, e.g. "You are the
planning step..."). The stale-model guard became exactly what the brief specified:

```python
if client.cfg.get("provider") == "off":
    raise LLMUnavailable(
        "The design assistant needs a configured model (llm.yaml provider)"
    )
```

`assistant_routes.py`: one docstring line. `index.html`: the two user-facing "Gemma"
mentions (intro paragraph, "Gemma mode" label -> "Model"). `tests/shot_design/
test_assistant.py`: the one assertion tied to the changed error string
(`match="Gemma returned an invalid"` -> `match="The model returned an invalid"`).
`grep -i gemma` on all three source files now returns nothing; the only remaining
`gemma4:*` strings anywhere in the tree are literal model-id fixtures in
`test_llm_client.py`, `test_assistant.py`, `test_blurb.py` (arbitrary tag strings for
local `CFG` dicts, unrelated to the word "Gemma" or the real yaml) and one
test-authored mock error string in `test_assistant_browser.py` that never touches
`index.html`'s actual copy — none of these were in scope per the brief.

## Adaptations beyond the two briefs (all necessary to keep `tests/shot_design` green)

1. **`model()` reads `cfg.get("models", {})` not `cfg["models"]`.** The brief's own
   E1 Step-1 test constructs `LLMClient(cfg={"provider": "fake", "cache": False},
   paths=paths)` — no `"models"` key at all — then calls `.chat(...)`, which builds
   `body["model"] = self.model(model)`. With the original strict `self.cfg["models"]`
   this test as given would `KeyError` before ever reaching the new dispatch code.
   Made it lenient (unknown/missing models config resolves to the literal key, same
   spirit as the existing "an unknown key is taken as a literal tag" fallback).

2. **`_ollama_cfg` falls through on an explicitly-null nested value, not just an
   absent key.** `configs/shot_design/llm.yaml` sets `ollama.base_url: null` by
   design (Frontier has no Ollama). `tests/shot_design/test_blurb.py::
   test_make_marks_prompt_v6_when_using_the_yaml_config` merges the real yaml with a
   flat top-level `"base_url": "http://llm.test"` override, expecting that override
   to take effect. A naive `nested.get(key, self.cfg.get(key, default))` never falls
   through because the nested key *exists* (it's just `None`), so the override was
   silently ignored and the test regressed from `source == "llm"` to `"template"`.
   Fixed by treating a `None` nested value as "unset" for fallback purposes.

3. **`tests/shot_design/test_llm_client.py::test_llm_yaml_matches_the_shape_the_client_
   reads`** literally asserts the real yaml's old Gemma/Ollama shape
   (`provider == "ollama"`, `models == {"quality": "gemma4:26b", ...}`, flat
   `ollama_bin_dir`, etc.) — exactly the shape E1 is required to change. Rewrote its
   assertions to the new shape (`provider == "agy"`, gemini model ids, `agy:` block,
   nested `ollama:` block with the three dirs `None`). This is the same category of
   fix as the brief's own Step-1 test, just for a pre-existing assertion the brief
   didn't quote.

4. **Three more real-yaml-shape assertions, in files outside either brief's file
   list**, broke for the identical reason as #3 and were fixed the same way:
   - `test_blurb.py::test_default_ollama_without_endpoint_uses_template_without_a_
     request`: `cfg["provider"] == "ollama"` -> `== "agy"`.
   - `test_blurb.py::test_shots_parquet_has_blurb_columns_after_build` and
     `::test_write_blurbs_updates_the_manifest_counts`, and
     `test_blurb_provenance.py::test_add_to_legacy_database_keeps_unknown_versions_
     nullable_and_new_versions_integer`: these build a database via
     `build.load_build_cfg()` (the real yaml) with no LLM endpoint present, and the
     manifest/parquet provenance columns record the *would-be* model
     (`client.model(blurb.model)`). Since `blurb.model` moved from `quality` to
     `fast`, the recorded id moved from `gemma4:26b` to `gemini-3.8-flash-low`.
     Updated the literal expected strings. (`test_write_blurbs_backfills_only_
     missing_and_is_atomic` and `test_write_blurbs_uses_client_config_for_prompt_and_
     provenance` were **not** touched — they pass an explicit local `CFG`/`FakeClient`
     cfg, so they never read the real yaml and were unaffected.)
   - `test_write_blurbs_updates_the_manifest_counts` needed one more fix beyond the
     literal-string swap: it reuses one `provenance` dict across two assertions, but
     the first half's manifest comes from `build.build()`'s internal client (real
     yaml, `gemini-3.8-flash-low`) while the second half's comes from an explicit
     `FakeClient(GOOD, paths)` (local module `CFG`, `gemma4:26b`) — two different
     clients that used to coincidentally resolve to the same literal. Split into
     `provenance_before` (real yaml) and `provenance` (FakeClient's local CFG).

5. **`test_make_marks_prompt_v6_when_using_the_yaml_config` (regression introduced
   by E2, not E1)**: once `"agy"` was registered as a provider (Step 7), this test's
   `yaml_cfg = {**config.load_yaml("llm.yaml"), "base_url": "http://llm.test"}` +
   `FakeClient(...)` stopped exercising `FakeClient`'s mocked httpx transport at all
   — `FakeClient` only fakes the OpenAI-shaped HTTP path, but with the real yaml's
   `provider: agy` now actually registered, `chat()` routed to `AgyProvider`, which
   shells out to the real `agy` binary (observed: the test took ~8s and the real CLI
   answered, but `client.calls` — populated only by the httpx mock — stayed empty,
   failing on `client.calls[0]`). Fixed by overriding `"provider": "ollama"` in this
   test's merged config, since the test's actual purpose (per its name/body) is to
   verify the yaml's prompt/model wiring over a mocked transport, not to exercise
   agy's real subprocess path.

6. **`client.__init__` uses `self.cfg.get("agy", {})`, not the brief's literal
   `self.cfg["agy"]`.** Every existing `LLMClient` construction in the test suite
   (module-level `CFG` in `test_llm_client.py`/`test_blurb.py`, `model_client()` in
   `test_assistant.py`, etc.) has no `"agy"` key at all. Since `_providers["agy"]` is
   now built unconditionally in `__init__` regardless of which provider is
   configured, strict indexing would `KeyError` on construction for literally every
   other test in the repo.

None of these needed a judgment call on the *emulated tool-call schema* itself (the
one case the task flagged as plausibly ambiguous) — the schema and its unwrap rule
were unambiguous once I matched them against the brief's own four tests (see
`_extract_structured` above for the one place I had to pick a reading: an absent
`status` key is treated as success, since the brief's own `result`-shaped test never
sets one).

## Tests

### TDD evidence, E1

RED (before implementing the seam):
```
$ PYTHONPATH=$PWD/src .../python -m pytest tests/shot_design/test_llm_client.py -q \
    -k "test_chat_dispatches_to_a_registered_provider or test_llm_yaml_matches_the_shape_the_client_reads"
FF
AttributeError: 'LLMClient' object has no attribute '_providers'
AssertionError: assert 'ollama' == 'agy'
2 failed, 21 deselected
```

GREEN (after client.py + llm.yaml):
```
$ PYTHONPATH=$PWD/src .../python -m pytest tests/shot_design/test_llm_client.py -q
23 passed
```

### TDD evidence, E2

RED (agy.py moved aside):
```
$ mv src/shot_design/llm/agy.py /tmp/.../agy.py.bak
$ PYTHONPATH=$PWD/src .../python -m pytest tests/shot_design/test_llm_agy.py -q
ImportError: cannot import name 'agy' from 'shot_design.llm'
1 error in 0.32s
```

GREEN (agy.py restored):
```
$ mv /tmp/.../agy.py.bak src/shot_design/llm/agy.py
$ PYTHONPATH=$PWD/src .../python -m pytest tests/shot_design/test_llm_agy.py -q
4 passed in 0.19s
```

### Full suite, after every commit

Run three times over the course of the work (once after each commit's fixes landed),
final state:
```
$ PYTHONPATH=$PWD/src .../python -m pytest tests/shot_design -q -p no:cacheprovider
1 failed, 1703 passed, 23 skipped, 2 warnings in 121.60s
FAILED tests/shot_design/test_mcp.py::test_the_project_mcp_config_points_at_this_server
```
That one failure is the pre-declared known pre-existing failure (unrelated `git
rev-parse` against a Stellar path baked into that test's fixture); everything else in
the 1699-test-file suite passes, including all of `test_llm_client.py` (23),
`test_llm_agy.py` (4), `test_blurb.py`, `test_blurb_provenance.py`,
`test_assistant.py`/`test_assistant_routes.py`/`test_assistant_browser.py`.

## Files changed

- `configs/shot_design/llm.yaml` — provider/agy/models/ollama restructure (E1)
- `src/shot_design/llm/client.py` — dispatch seam, `_ollama_cfg`, agy registration
  (E1 + E2)
- `src/shot_design/llm/agy.py` — new, the provider (E2)
- `src/shot_design/design/assistant.py` — de-Gemma (E2 step 6)
- `src/shot_design/ui/assistant_routes.py` — de-Gemma docstring (E2 step 6)
- `src/shot_design/ui/static/index.html` — de-Gemma copy (E2 step 6)
- `tests/shot_design/test_llm_client.py` — new dispatch test, real-yaml-shape test
  rewritten (E1)
- `tests/shot_design/test_llm_agy.py` — new (E2)
- `tests/shot_design/test_assistant.py` — one assertion string (E2 step 6)
- `tests/shot_design/test_blurb.py` — real-yaml literal fixes, `provenance_before`
  split, provider-override fix for the FakeClient/agy interaction (E1 + E2, see
  Adaptations)
- `tests/shot_design/test_blurb_provenance.py` — real-yaml literal fix (E1, see
  Adaptations)

## Self-review checklist

- Both briefs' steps done, including both rulings (Frontier-null ollama paths /
  header comment; live smoke run for real with the resolved gemini-3.8-flash-low id).
- `test_llm_client.py`'s original 22 tests pass unchanged in behavior (one of them —
  the real-yaml-shape test — had its *assertions* updated because the yaml it reads
  changed underneath it, which is the whole point of E1; the test's role/shape is
  otherwise identical).
- `provider: agy` is the real yaml's default; `provider: off` still raises
  `LLMUnavailable` before touching cache, endpoint, or any provider, and opens no
  socket (`test_provider_off_raises_without_opening_a_socket`,
  `test_loaded_opens_no_socket_when_the_provider_is_off` both still pass).
- Legacy flat `ollama_*`/`base_url`/`endpoint_file` keys are still readable for one
  release, verified both by the untouched flat-style `CFG` fixtures across the suite
  and by the null-fallback fix in Adaptation #2.
- YAGNI: `AgyProvider` has exactly the surface the brief asked for; no retry logic
  was added even though `llm.yaml` has an unused `agy.retries: 2` key — the brief
  didn't ask `agy.py` to implement retries (client-level retry policy, if any, is a
  future task), and I didn't invent one.
- Testing: all four `test_llm_agy.py` tests inject a fake `runner`, never touch a
  real subprocess; the one real call is the Step-5 smoke, run once, output pasted
  above verbatim.
- Ruff line length: checked every changed/new line in every touched file with
  `awk 'length>88'` restricted to diff `+` lines; zero violations remain outside
  pre-existing untouched lines (the docstrings this task didn't touch already
  exceeded 88 before E1/E2, e.g. `client.py`'s original module docstring and the
  `off` property's one-liner — left as found, since the rule is "every new line",
  not a retroactive repo-wide reflow).

## Concerns

- The two extra test-file fixes outside either brief's stated file list (`test_blurb.
  py`, `test_blurb_provenance.py`) were unavoidable side effects of changing the real
  `llm.yaml` provider/models — anyone reviewing should sanity-check that I didn't
  paper over a real behavioral regression rather than a stale literal. I re-derived
  each expected value from first principles (which client each test actually
  exercises: real yaml vs. a local `CFG` constant vs. `FakeClient`) rather than just
  copying whatever the run happened to print.
- `agy.retries: 2` in `llm.yaml` is currently inert — nothing in `agy.py` reads it.
  It was part of the brief's exact yaml block, so I kept the key, but flagging that
  it's a no-op today in case a future task assumes it's already wired up.
- The de-Gemma commit (Step 6) is dated/ordered before the agy-provider commit
  (Step 7) per the brief's literal step order, even though chronologically I built
  and smoke-tested the provider first and only discovered the `test_blurb.py`
  FakeClient/agy interaction (Adaptation #5) afterward. The two commits are
  independent in content (disjoint files) so the reordering is cosmetic, but noting
  it since "TDD evidence" timeline and "commit order" don't perfectly match my
  actual work order.

## Fix round 1

Review found 2 Critical, 1 Important, 3 Minor issues plus a docstring-note ask.
Commit `f6018be` (`shot_design llm: provider-aware available(); strict provider
lookup; agy launch errors and retries`), on top of E3's concurrent `e91fb99`/
`b0eec53` (blurb `--workers`/dry-run work, `src/shot_design/shotdb/build.py` and
`src/shot_design/cli.py` — untouched by this round, confirmed by `git diff
--name-only` before staging). Same worktree/branch, same test command.

### 1. Critical -- `available()` was hard-wired to the ollama endpoint check

`src/shot_design/llm/client.py:available()` now branches on `self.cfg["provider"]`:
`off` keeps its existing message; `agy` checks `shutil.which(agy.bin)` (no subprocess,
no network) and names the missing binary in the hint if absent; anything else
(`ollama`, `openai_compatible`, an unrecognised string) falls through to the existing
endpoint-file/base_url check. One judgment call: the review's shorthand for the `off`
case was `(False, "llm provider is off")`, but the existing (unflagged, still-passing)
`test_provider_off_raises_without_opening_a_socket` asserts the current literal string
`"configs/shot_design/llm.yaml has provider: off"`. I kept the existing string rather
than break that test over wording the review didn't call out as wrong -- the
functional fix (`off` -> `(False, <hint>)`) is what matters and is unchanged.

New tests in `tests/shot_design/test_llm_client.py`: `test_available_checks_the_agy_
binary_on_path` (monkeypatches `shutil.which` both ways), `test_available_is_false_
when_the_provider_is_off`.

**The safety net and the audit it triggered.** Added an autouse fixture in
`tests/shot_design/conftest.py`, `_no_real_agy_subprocess`, that monkeypatches
`shot_design.llm.agy.AgyProvider._run_subprocess` to raise `AssertionError("real agy
call from a test")`; a test that wants the fake agy path still passes `runner=` to
`AgyProvider`, which bypasses `_run_subprocess` entirely and is unaffected. `git grep
-n 'load_yaml("llm.yaml")' tests/` found 4 hits: `test_llm_client.py:63` and
`test_blurb.py:~530` are structural shape assertions with no `.chat()`/`.available()`
call downstream (safe); `test_serve_llm.py:76` feeds a temp yaml to a `serve_llm.sh`
subprocess lifecycle test that never constructs a Python `LLMClient` (safe, confirmed
by reading the whole fixture). Running the fixture against the full suite exposed the
real leaks -- not from that grep, but from every place a bare `LLMClient()` or
`build.build()`/`build.add()` builds its own client from the real yaml and the test's
assertion depends on that client being *unavailable*: `available()` was hard-wired to
ollama before this fix, so those tests passed by accident regardless of `provider:
agy`; after the fix, `agy` is genuinely on this machine's PATH (`shutil.which("agy")`
-> `/ccs/home/nchen/.local/bin/agy`), so `available()` now correctly returns `True`,
and each of these would otherwise have gone on to place a real `agy` call (caught
loudly by the safety net instead). Fixed, all in `tests/shot_design/`:
- `test_blurb.py`'s `rec` fixture (used by most of the file) and
  `test_shots_parquet_has_blurb_columns_after_build` and
  `test_write_blurbs_updates_the_manifest_counts` (the two the review named, plus the
  fixture, which the review's line numbers for `test_default_ollama_...` pointed near)
  now wrap their `build.build(...)` call in a new `force_ollama_provider()` context
  manager (defined once in `conftest.py`): it monkeypatches `config.load_yaml` so
  `name == "llm.yaml"` comes back with `provider: ollama` forced, for the duration of
  the `with` block only, then restores the real loader.
- `test_default_ollama_without_endpoint_uses_template_without_a_request` (the review's
  other named test): kept its `assert cfg["provider"] == "agy"` (documents the real
  production default) and builds the actual test client from `{**cfg, "provider":
  "ollama"}` instead -- this test's assertions are about the *ollama* no-endpoint
  hint text (`"scripts/shot_design/serve_llm.sbatch"`), so it needs the true ollama
  branch, not just an unavailable agy.
- `test_blurb_provenance.py`'s `test_add_to_legacy_database_...`: same
  `force_ollama_provider()` wrap around both its `build.build(...)` and `build.add(...)`
  calls (`build.add` also builds its own client via `_blurb_client()`).
- `test_describe.py`'s `test_polish_returns_the_template_when_no_endpoint_is_
  published` and `test_llm_client.py`'s `test_polish_is_silent_when_no_server_is_up`:
  both call `describe.polish(...)` for real (no client override), same wrap.
- Added a docstring paragraph to `test_blurb.py`'s module docstring per the review's
  ask, describing the landmine and pointing at the conftest fixture.

I also grepped every other `build.build(`/`build.add(` call site with no explicit
client across the whole `tests/shot_design/` tree (`test_build_phases.py`,
`test_build_staging.py`, `test_build_store.py`, `test_publish_guard.py`,
`test_retrieval.py`, `test_scenarios.py`, `test_suggest.py` -- none of them named by
the review) expecting the same landmine there, since `agy` being on PATH makes
`available()` true everywhere, not just in the two named tests. It did not
materialize: the full-suite run (below) shows zero failures and zero "real agy call
from a test" assertions anywhere outside the five sites above, so whatever those
tests actually exercise does not route through `_blurb_client()`'s result the way the
five fixed tests do (I did not chase the exact reason once the evidence -- a clean
full run -- was in; noting this so a reviewer knows it was checked empirically, not
assumed).

### 2. Critical -- unregistered provider silently fell back to ollama

`chat()`'s dispatch is now `self._providers[provider]` in a `try`/`except KeyError`,
raising `LLMUnavailable(f"unknown llm provider {provider!r}; known providers: ...")`
instead of the old `.get(provider, self._providers["ollama"])`. This could not be a
literal strict lookup against only `{"ollama", "agy"}`, though: `test_llm_client.py`'s
module-level `CFG` constant (~20 tests) uses `"provider": "openai_compatible"` and
exercises real HTTP dispatch through it (`test_chat_posts_openai_shape_and_parses_
tool_calls` et al.) -- a strict lookup against just the two provider names would have
`KeyError`'d every one of those. Rather than treat that as a fallback case, I
registered `"openai_compatible"` as its own provider key pointing at the same
`_chat_openai` lambda as `"ollama"`, matching the module's own docstring ("Ollama
today and a hosted provider later are the same code path"): a hosted OpenAI-compatible
server is not a typo of `ollama`, it is a legitimate second name for the identical
wire format. `test_chat_dispatches_to_a_registered_provider`'s dynamically-registered
`"fake"` provider is unaffected (it's added to `_providers` before `chat()` is ever
called, so it's a genuine lookup hit, not a fallback).

New test: `test_unknown_provider_raises_llmunavailable_naming_the_known_ones`
(`provider: "gemma-cli"` -> `LLMUnavailable` matching `"unknown llm provider
'gemma-cli'"`).

### 3. Important -- a missing `agy` binary raised a raw `FileNotFoundError`

`AgyProvider._run_subprocess` now catches `OSError` around `subprocess.run(...)`
(after the more specific `subprocess.TimeoutExpired`, which is not an `OSError`
subclass) and re-raises `AgyError(f"could not launch agy ({cmd[0]!r}): {e}")`, caught
by `chat()` like every other `AgyError` and turned into `LLMUnavailable`.

New test: `test_missing_binary_raises_llmunavailable_naming_it`, which restores the
*real* `_run_subprocess` (captured at module-import time in `test_llm_agy.py`, before
conftest's autouse safety net patches it) via `monkeypatch.setattr`, then points
`cfg["bin"]` at `"definitely-not-a-real-binary-xyz"` and asserts the name appears in
the `LLMUnavailable` message. This is the one test in the file that intentionally
exercises the real subprocess path; it never touches the real `agy` CLI (the binary
name is nonsense), so it stays hermetic despite the autouse fixture's default block.

### 4 & 5. Minor -- `agy.retries` was inert; the timeout error dropped stderr

Implemented together in `AgyProvider.chat()`: `retries = int(self.cfg.get("retries",
0))` bounds a retry loop around `self._runner(cmd, prompt)` (not around
`_run_subprocess` directly, so an injected test `runner=` is retried too, matching
what "test with a runner that fails once then succeeds" asks for). A new
`_is_retryable(message)` helper retries only a timeout or an exit with an empty
tail (both plausibly transient); an exit that actually said something (a bad prompt,
a quota error, ...) is not retried even if `retries` is configured, since repeating it
would not help. `_run_subprocess`'s `TimeoutExpired` handler now includes the stderr
(falling back to stdout) tail in the `AgyError` message, decoding it if it comes back
as bytes -- I verified empirically that `subprocess.TimeoutExpired.stderr`/`.stdout`
are raw `bytes` even when `subprocess.run` was called with `text=True` (confirmed with
a throwaway script against the frozen interpreter before writing `_decode()`).

New tests: `test_retries_on_a_transient_failure_then_succeeds` (fails once with a
timeout-shaped `AgyError`, succeeds on the second `runner` call),
`test_retries_are_exhausted_and_raise_the_last_error`, `test_a_real_exit_error_is_not_
retried_even_with_retries_configured` (asserts the runner was called exactly once),
and `test_timeout_error_includes_the_stderr_tail` (real `_run_subprocess`, restored
the same way as the missing-binary test, against a short-lived `python -c` child that
writes to stderr then sleeps past a 0.2s timeout).

### 6. Minor -- dead `reasoning_effort` handling

**Deviation, not implemented as asked.** I did not remove `client.py`'s
`reasoning_effort` handling in `chat()`. It is not dead: `test_reasoning_effort_is_
sent_only_when_configured` in `test_llm_client.py` and an assertion in
`test_blurb.py` (`good.calls[0]["reasoning_effort"] == "none"`) both actively exercise
it, and the first test's own docstring explains why it exists -- "gemma4:26b answers
with empty content unless reasoning is turned off." The real `llm.yaml` no longer
sets a top-level `reasoning_effort` (removed in E1), which is presumably why the
review read it as dead from the diff/config alone, but the field is still a supported
per-cfg option for the `ollama`/`openai_compatible` path (kept "for one release" per
the module docstring), gated by real tests documenting a real Gemma quirk. Removing it
would have meant deleting or rewriting two passing, purposeful tests to satisfy a
cosmetic cleanup with no correctness benefit, which reads as exactly the kind of
should-push-back case `receiving-code-review` describes rather than one to
comply with silently. Flagging this explicitly for the reviewer to overrule if there's
context I'm missing (e.g. a plan to drop Ollama/Gemma support entirely next release,
in which case removing it then, together with its tests, would make sense).

### Tests and full-suite run

Targeted runs after implementing all of the above:
```
$ PYTHONPATH=$PWD/src .../python -m pytest tests/shot_design/test_llm_client.py tests/shot_design/test_llm_agy.py -q -p no:cacheprovider
35 passed in 0.64s
$ PYTHONPATH=$PWD/src .../python -m pytest tests/shot_design/test_blurb.py tests/shot_design/test_blurb_provenance.py tests/shot_design/test_describe.py -q -p no:cacheprovider
84 passed, 1 skipped in 12.92s
```
Full suite:
```
$ PYTHONPATH=$PWD/src .../python -m pytest tests/shot_design -q -p no:cacheprovider
1 failed, 1714 passed, 23 skipped, 2 warnings in 122.17s (0:02:02)
FAILED tests/shot_design/test_mcp.py::test_the_project_mcp_config_points_at_this_server
```
Same single pre-approved known failure as before (an unrelated `git rev-parse`
against a Stellar path baked into that test's fixture; `git status`/`git diff
--name-only` before staging confirm this round touched none of E3's four files).
1714 vs. the prior report's 1703 passed reflects tests E3 added concurrently
(`e91fb99`/`b0eec53`), not anything from this round. No "real agy call from a test"
assertion appears anywhere in the full log, confirming the safety net is armed and
no leak remains.

Note on process: I implemented the production code changes for items 1-5 before
writing their tests (not strict RED-then-GREEN), because the fixes interact tightly
(the `available()` fix's correctness is what *exposed* the test-fixture leaks, which
had to be understood and fixed together with the strict-lookup change before any
single new test could be meaningfully red/green in isolation). I did verify each new
test actually exercises the intended branch by inspection (e.g. confirming
`shutil.which` is looked up via the plain module attribute so `monkeypatch.setattr
("shutil.which", ...)` binds correctly, and empirically confirming the
`TimeoutExpired.stderr` bytes-vs-str behavior before relying on it), rather than
trusting the tests passed for the wrong reason.

### Self-review for this round

- Ruff line length: every new/changed line in all 8 touched files checked with
  `git diff --unified=0 | grep '^+' | awk 'length>88'`; zero violations.
- Did not touch `src/shot_design/shotdb/build.py`, `src/shot_design/cli.py`,
  `scripts/shot_design/blurb_frontier.sh`, or `tests/shot_design/test_blurb_backfill.py`
  (E3's files) -- confirmed via `git diff --name-only` immediately before `git add`.
- Committed with explicit pathspecs (`git add <8 paths>`, `git commit -m "..." --
  <8 paths>`); no `git add -A`; no index.lock contention (single commit, no retry
  needed).
- YAGNI: did not add a CLI flag or config knob for the retry backoff delay (the
  review didn't ask for one and `agy.yaml`'s `retries: 2` implies immediate retry is
  fine); did not touch `cmd_llm`'s `ep = client.endpoint()` line even though it will
  now `AttributeError` on `ep.url` for a healthy `agy` config (available() -> True,
  but `endpoint()` -> `None` for agy, since endpoint() only knows the ollama
  base_url/endpoint-file shape) -- that line is in `cli.py`, off-limits this round;
  flagged below instead of silently fixed or silently ignored.

### Concerns for the reviewer

- **`cmd_llm` in `src/shot_design/cli.py` (off-limits this round, owned by E3) will
  crash on a healthy agy config.** `cmd_llm()` does `ok, hint = client.available()`
  then, if `ok`, unconditionally `ep = client.endpoint(); print(f"{ep.url} ...")`.
  With `provider: agy` and the binary present, `available()` now correctly returns
  `True`, but `client.endpoint()` still only understands the ollama
  base_url/endpoint-file shape and returns `None` for `agy` -- so `ep.url` raises
  `AttributeError`. This is a real bug my Critical #1 fix exposes (it was previously
  masked by `available()` always being hard-wired False), but the fix belongs in
  `cli.py`, which I was told never to touch this round. Someone needs to either give
  `cmd_llm` an agy-specific branch or route this through E3's next pass.
- Item 6 (`reasoning_effort`) was deliberately not done -- see above; I'd like either
  an explicit "yes remove it and update/delete the two tests" or a "leave it" from
  the reviewer rather than assuming either way.
- The `off`-case message in `available()` was kept as the existing literal string
  rather than changed to the review's shorthand `"llm provider is off"` -- functionally
  equivalent, but flagging the wording deviation explicitly in case the reviewer's
  literal string was intentional for some downstream consumer I didn't find.
