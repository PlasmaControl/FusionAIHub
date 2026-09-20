---
title: "LLM providers"
sidebar_position: 4
---

## Local LLM and per-shot blurbs

`configs/shot_design/llm.yaml` is one shared file for both clusters — unlike
`paths.yaml`/`paths.frontier.yaml`, there is no per-cluster override, so
whatever is checked in applies everywhere. Its checked-in default is now
`provider: agy` (see [Frontier: Gemini Flash via `agy`](#frontier-gemini-flash-via-agy)
below); `shot_design.llm.client.LLMClient` dispatches purely on that key, so
a Stellar operator running the Ollama path sets `provider: ollama` in this
same file. Without an endpoint file, the client is unavailable and builds
use the existing header + outcome template. The UI reads stored blurbs;
loading a page does not generate them.

Prompt v6 asks for exactly **three plain sentences**, at most **90 words**:

1. What the shot or experiment set out to do.
2. Whether it succeeded, plainly naming a disruption, fast current quench or early
   termination when the source records one; say when success is unknown.
3. One interesting finding from the operator entries or shot brief, or the literal
   sentence **No notable findings were logged.**

The model uses the mini-proposal title/purpose, run title, shot brief, up to eight
quotable operator entries, chief-operator status, outcome and verdict. It writes
in its own words, without quotation marks, headings, lists or invented/rounded
numbers. Abbreviations, acronyms and symbols must be copied verbatim, never expanded,
translated or explained, even when the model thinks it knows their meaning; it may
not introduce an abbreviation absent from the source. Quantities must use digits
and the unit exactly as given, or be omitted; numbers must never be written as words
(such as `thirteen`, `several hundred` or `two-one`). The 2026-09-15 audit of the v5
backfill motivated these rules: shot 200729 incorrectly expanded the source's `AE`
to `aeroelastic instabilities`.

The gate rejects empty text, overlong text, double/curly quotation marks, numbers
or shot references absent from the source, and anything other than three complete
sentences. Before counting sentences, it also checks:

- `number written as a word: [...]`: a number word absent as a whole word from
  the source, ignoring case and splitting candidate text on whitespace and hyphens.
  The list covers two through nineteen, tens through ninety, hundred, thousand,
  million and dozen. `one` and ordinals are exempt; source wording such as
  `two-day experiment` may be retained.
- `abbreviation not in source: [...]`: an ALL-CAPS token of two to six letters,
  with an optional plural `s`, absent from the source. The comparison removes
  hyphens, ignores case, strips the token's plural `s`, and matches substrings:
  `L-H` passes for `PLH`, `ELMs` for `ELM`, and `NTMs` for `NTM`. `DIII` is always
  allowed; single letters, including the `D` in `DIII-D`, are exempt.

Sentence terminators are `.`, `!` and `?` followed by whitespace or end of text;
decimal points do not split sentences. A rejected reply retains the header + outcome
template. These mechanical checks do not establish semantic accuracy: inspect the
dry-run candidates before the full backfill.

Every generated table row carries `blurb`, `blurb_source` (`llm`, `template`, or `human`
for a summary written by hand from the shot's own text -- shown with a `hand` tag in
the browser, never selected by a model backfill unless named with `--shots`),
`blurb_model` (resolved tag, e.g. `gemma4:26b`) and `blurb_prompt_version` (integer).
The latter two record the configuration used for the attempt, including template
fallbacks; `blurb_source` says whether the model supplied the final text. Legacy
rows retain unknown provenance until processed. `manifest.json["blurbs"]` contains
the whole table's `llm`, `template` and `human` counts plus the latest pass's `model` and
`prompt_version`. Its `prompt_versions` histogram counts every non-null row version
with string keys, for example `{"5": 502, "6": 2}`, making mixed versions visible;
legacy rows with unknown versions are excluded. Builds and additions update the
same histogram.

`blurb` normally selects templates, empty blurbs, and rows with missing or older
prompt versions. `--all` selects every row, including current model blurbs.
`--shots N [N ...]` regenerates exactly the named shots in ascending order, regardless
of their existing text or prompt version; unknown shots fail before any model call.
Combining `--shots` with `--all` reports an error on stderr and exits with status 2.
`--limit N` takes the first N eligible shots after selection (`0` does nothing).
`--dry-run` prints each shot number, source-text character count, candidate, gate
verdict/reason and final text; it writes neither database files nor request cache.
A normal pass atomically replaces `shots.parquet`, then atomically updates the
manifest. Those are two file replacements, not a transaction across both files.

For a targeted preview, use `shot_design blurb --shots 200729 190511 --dry-run`;
remove `--dry-run` to regenerate those rows. Invoke the CLI directly for `--shots`:
the `blurb_all.sh` wrapper already supplies `--all`.

### Operator runbook (after merge)

The existing Ollama 0.33.3 binary and both Gemma models are reused in place.
`serve_llm.sh` resolves these three directories itself, from hardcoded
fallback defaults baked into the script (the literal paths below) — it reads
each as a FLAT top-level `llm.yaml` key, not through the nested `ollama:`
block that now holds unrelated `null` placeholders for the agy default, so
that block has no effect on `serve_llm.sh`. Set a flat top-level key of the
same name in `llm.yaml` to override one:

| `llm.yaml` key (flat, top-level) | default (hardcoded in `serve_llm.sh`) |
| --- | --- |
| `ollama_bin_dir` | `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/bin/ollama` |
| `ollama_models_dir` | `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/ollama` |
| `ollama_home_dir` | `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/ollama_home` |

`serve_llm.sh` requires `<ollama_bin_dir>/bin/ollama`; a missing binary exits 2
with its path. Installed model tags bypass pulling; a missing tag uses the retained
pull loop. No binary installation or copying occurs. Defaults are a 16384-token
context, 24h keep-alive and two loaded models. The Slurm allocation is one GPU,
eight CPUs, 64 GB RAM and four hours. See the upstream
[Ollama serving documentation](https://github.com/ollama/ollama/blob/main/docs/cli.mdx)
and [environment settings](https://github.com/ollama/ollama/blob/main/docs/faq.mdx).

Run these commands as the operator, from the merged checkout, on the login node:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# 1. Create the Slurm log directory, then submit the model server.
mkdir -p /scratch/gpfs/EKOLEMEN/nc1514/ideate/llm
sbatch scripts/shot_design/serve_llm.sbatch

# 2. Wait for readiness; check the job/log if this does not appear.
until test -s /scratch/gpfs/EKOLEMEN/nc1514/ideate/llm/endpoint.json; do sleep 2; done
cat /scratch/gpfs/EKOLEMEN/nc1514/ideate/llm/endpoint.json
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e shot-design-cpu python -m shot_design llm

# 3. Inspect five candidates, gate verdicts and final summaries without writing.
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e shot-design-cpu python -m shot_design blurb --all --dry-run --limit 5

# 4. After reviewing the preview, backfill all shots from the login node.
bash scripts/shot_design/blurb_all.sh

# 5. Stop the existing UI serve process with Ctrl-C in its terminal, then restart
#    it so its loaded database snapshot contains the new blurbs.
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e shot-design-cpu python -m shot_design serve --port 8765
```

The blurb wrapper runs the mandated frozen Pixi command with
`python -m shot_design blurb --all`; additional flags are forwarded. The client
runs on the CPU and discovers the GPU server through `llm/endpoint.json` under
`paths.data_root`. A configured `base_url` takes precedence over that file.
Match the endpoint's `job_id` to the submitted job and check its log for `ready:`;
the `llm` command prints discovery metadata and does not probe the server.

The Slurm script binds `$(hostname -s):11434` and uses the default data root in its
literal `#SBATCH -o` directive. For another root, override `sbatch --output` and
create its `llm` directory first; account for Pixi activation overriding
`SHOT_DESIGN_DATA_ROOT` as described above. Endpoint JSON contains `url`, `models`,
`host`, `job_id`, `started` and `version`, and is published with an atomic rename.
An answering port is refused before any write. On exit, the script removes the
endpoint only if its URL and start time still match this run, then stops its child.
To stop the GPU allocation, use `scancel JOB_ID` with the ID printed by `sbatch`;
to renew it, wait for that job to exit and submit the same script again.



## Frontier: Gemini Flash via `agy`

Everything above is the Stellar path: `provider: ollama`, an Ollama server on
a GPU node, an `endpoint.json` file the CPU client discovers. Frontier has no
route to that binary — it is an x86_64 build, no ROCm build for MI250X was
ever produced, and compute nodes have no outbound network to pull one even
if there were. Frontier's `llm.yaml` instead configures `provider: agy`, a
second `LLMClient` backend that shells out to the Antigravity CLI (`agy`)
from the login node, which does have network access and an OAuth cache.

`LLMClient.chat(...)` keeps the same signature and `Reply` return for both
providers; only the body that builds the request and reads the response
differs. The dispatch lives in `self._providers[provider]`, keyed by
`llm.yaml`'s `provider` value (lower-cased and stripped), so adding `agy`
did not touch `ollama`'s code path. An unregistered provider string raises
`LLMUnavailable` immediately, naming the known providers, instead of
silently falling back to Ollama.

```yaml
provider: agy
agy:
  bin: agy
  timeout_s: 300
  retries: 2
  extra_args: ["--disable-slash-commands"]
models:
  quality: gemini-3.8-flash-high
  fast: gemini-3.8-flash-low
default: quality
blurb:
  model: fast
```

`gemini-3.8-flash-high` is the quality tier (the design chat, `describe`
polish); `gemini-3.8-flash-low` is the fast tier used for the per-shot blurb
backfill. Unlike the Ollama configuration, no `reasoning_effort` key is
needed — the `agy` provider passes no separate effort flag, since the model
id itself carries the effort level.

Because `agy` has no native chat-with-tools protocol, tool calls are
emulated: the client renders the conversation (system/user/assistant/tool
messages, plus each available tool's name, description and JSON parameters)
into one prompt, and asks for a JSON reply matching a fixed schema —
`{"content": string, "tool_calls": [{"name", "arguments"}]}` — via `agy`'s
`--json-schema` flag. The `-p=<prompt>` argument must be last and attached
with `=`, not a separate token: passing `-p` and `--model` as adjacent
arguments makes `agy` swallow `--model` as part of the prompt text. The
response comes back as JSON on stdout: a `structured_output` field is
preferred, falling back to a JSON-encoded `result` string, then the first
embedded JSON object found inside `response`;
a non-`SUCCESS` `status`, a non-zero exit or a timeout all surface as the
same `LLMUnavailable` error the Ollama path raises when its endpoint is
unreachable, so callers do not need a provider-specific except clause.

The blurb backfill runs from the login node, not a compute job, because
`agy` needs the network and the OAuth cache — there is no `sbatch` wrapper
for it, unlike the census/build/encode stages:

```bash
source scripts/slurm_frontier/_shot_design_common.sh
bash scripts/shot_design/blurb_frontier.sh --dry-run --limit 5   # preview
WORKERS=8 bash scripts/shot_design/blurb_frontier.sh              # full backfill
```

`write_blurbs(..., workers=N)` runs the per-shot LLM calls in a thread pool
(safe here because each `agy` call is its own subprocess); the parquet and
manifest rewrite at the end stay single-threaded. The three-sentence gate,
the acronym/number rules and the `blurb_source`/`blurb_model`/
`blurb_prompt_version` bookkeeping described above are unchanged by the
provider — a Gemini Flash blurb is gated exactly as a Gemma one was.

Ollama/Gemma is Stellar-only; nothing under `configs/shot_design/llm.yaml`
on Frontier references it. See [Frontier](../clusters/frontier.md#llm-gemini-flash-via-agy)
for the operational summary and [Database build](./database-build.md) for
where the blurb backfill fits in the Frontier database pipeline.
