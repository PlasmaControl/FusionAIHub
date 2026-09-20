# Recommender on Frontier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate `recommender` + `dev-peter` + the `nathan_fm` working tree into a new `nathan_dev`, run the Shot Designer recommender on Frontier against ~5000 shots with the IGNITE v4 codec/dynamics generation, add a simulation stage and two LLM providers, and replace `docs/` with a Docusaurus site.

**Architecture:** The recommender (`src/shot_design`) already does interpret → retrieve → propose → validate → save and exports a v2-layout IGNITE seed. This plan (a) makes its IGNITE layer manifest-driven so the same code handles the 15-modality v4 set, (b) adds `shot_design.simulate` which turns a saved design into a paired IGNITE rollout on one Frontier GCD and renders panels, (c) adds an `agy` (Gemini Flash) LLM provider behind the existing `LLMClient.chat` contract and backfills the shot blurbs with it, (d) ports the Slurm/paths layer to Frontier via a `paths.frontier.yaml` + `shot-design-frontier` pixi env, and (e) moves `docs/` under a Docusaurus site with Claude material relocated to `.claude/`.

**Tech Stack:** Python 3.11, pixi/uv, PyTorch 2.10 ROCm 7.1, FastAPI, pydantic v2, h5py, pytest; Slurm on Frontier (`-A fus187`); Docusaurus 3 (Node 22, TypeScript); `agy` 1.2.7 (Gemini 3.8 Flash).

**Spec:** `.claude/superpowers/specs/2026-09-19-recommender-frontier-port-design.md`

## Global Constraints

- Repo: `/lustre/orion/fus187/scratch/nchen/FusionAIHub`. Work on branch `nathan_dev` (created in Task A1). Never force-push; never touch `nathan_fm`, `dev-nathan`, `main`.
- Python for tests: `pixi run --frozen -e shot-design-frontier pytest tests/shot_design tests/labeler` (env exists since Task B2). Every pixi call on Frontier uses `--frozen`; never re-solve the lock.
- Frontier data roots (spec §2): `SHOT_DESIGN_DATA_ROOT=/lustre/orion/fus187/proj-shared/nchen/shot_design`, `LABELER_ROOT=/lustre/orion/fus187/proj-shared/nchen/labeler`, `SHOT_DESIGN_CORPUS=/lustre/orion/fus187/proj-shared/foundation_model`. Never write there from a test; tests use the `paths` fixture (tmp_path).
- v4 contract: 15 modalities (spec §"Facts"), every vocab 1000, `t0_start_s: 1.0`, 1209 tokens/frame, actuators `(F, 88)`. Codec template `/lustre/orion/fus187/proj-shared/models/ignite_codecs_v4/{m}/codec_best.pt`; cache `/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes`; dynamics `/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/runs/mskfull/dynamics_best.pt` (step 3200). Pinned copies with sha256 live under `<models_dir>/IGNITE_v4/`.
- Slurm: `-A fus187 -p batch`; short demo/validation jobs `-q debug` (≤ 2 h, one at a time). Logs to `$SHOT_DESIGN_DATA_ROOT/runs/slurm/%j.out`. Never run production writes (`build`, `add`, `labels join`) without the owner's go-ahead stated in the task.
- No Claude/superpowers/planning material under `docs/`; it lives under `.claude/`.
- Ruff `line-length = 88`. Every commit message is one line, imperative, `<area>: <what>`.
- Tests are TDD: failing test first, then the code, then green, then commit.
- Nothing is copied from Stellar. `foundation_model_text/sql/` is absent by decision: `mpid` is None everywhere and no logbook-text claims exist.
- The only LLM is Gemini Flash through `agy`; no Gemma/Ollama serving on Frontier.

---

## Phase A: Git consolidation

### Task A1: Create `nathan_dev` and carry the `nathan_fm` working tree

**Files:**
- Branch only. Untracked files listed by `git status --short` on `nathan_fm` (35 evaluation scripts, `docs/eval_improvement_recommendations.md`, `scripts/data_fetching_omega/n1rms_fetch/`, `scripts/slurm_frontier/_frontier_common.sh`).

**Interfaces:**
- Produces: local + remote branch `nathan_dev` at `origin/recommender` + one commit.

- [ ] **Step 1: Record the starting state**

```bash
cd /lustre/orion/fus187/scratch/nchen/FusionAIHub
git rev-parse HEAD                      # expect 702aaae
git status --short > /tmp/claude-20641/nathan_fm_status.txt
git diff > /tmp/claude-20641/nathan_fm_tracked.patch
```

- [ ] **Step 2: Stash tracked changes, create the branch, restore**

```bash
git stash push -m "nathan_fm working tree 2026-09-19"
git switch -c nathan_dev origin/recommender
git stash pop
```
If `stash pop` reports conflicts in `scripts/slurm_frontier/_frontier_settings.sh`, `scripts/slurm_frontier/benchmark_plugin_perf.sh` or `src/tokamak_foundation_model/ignite/train_dynamics.py`: open each, keep the recommender side's structure and re-apply the `nathan_fm` hunks from `/tmp/claude-20641/nathan_fm_tracked.patch` by hand (they add the RCCL plugin toggle block and train_dynamics timing/log lines). Then `git add` and `git stash drop`.

- [ ] **Step 3: Verify the IGNITE trainer still imports**

```bash
PYTHONPATH=src .pixi/envs/frontier/bin/python -c "import tokamak_foundation_model.ignite.train_dynamics as t; print('ok')"
PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite -x -q 2>&1 | tail -3
```
Expected: `ok`; pytest passes or reports only pre-existing skips (record any failure verbatim in the commit body; do not fix here).

- [ ] **Step 4: Commit the carried work**

```bash
git add -A scripts/evaluation scripts/data_fetching_omega/n1rms_fetch scripts/slurm_frontier docs/eval_improvement_recommendations.md src/tokamak_foundation_model/ignite/train_dynamics.py
git commit -m "nathan_fm: carry Frontier settings, plugin benchmark, train_dynamics edits and Stage-1 evaluation scripts onto nathan_dev"
git push -u origin nathan_dev
```

### Task A2: Merge `dev-peter`

**Files:**
- Modify (conflicts): `src/tokamak_foundation_model/ignite/eval_dynamics.py`, `src/tokamak_foundation_model/ignite/maskgit.py`
- Also touched by the merge: `scripts/slurm_frontier/eval_dynamics.sh`, `scripts/slurm_frontier/train_dynamics.sh`, `src/tokamak_foundation_model/ignite/train_dynamics.py`

- [ ] **Step 1: Merge and list conflicts**

```bash
git merge --no-ff origin/dev-peter -m "merge dev-peter: IGNITE actuator counterfactuals, dropout, Gumbel reveal, honest video panel"
git diff --name-only --diff-filter=U
```

- [ ] **Step 2: Resolve `maskgit.py`**

Peter's side adds a Gumbel-noise reveal option and actuator dropout in `MaskGITDynamics`; the recommender side has its own edits in the same region. Keep both: Peter's new parameters/branches plus the recommender's code paths. Ensure the constructor signature keeps every keyword the recommender's callers use (`grep -n "MaskGITDynamics(" src tests`).

- [ ] **Step 3: Resolve `eval_dynamics.py`**

Keep Peter's `--actuator_mode` extensions and honest video panel, and the recommender-side changes. After resolving, the file must still define `load_model(ckpt_path, device)` returning `(model, cfg, step)` exactly as before (Task C3 edits it).

- [ ] **Step 4: Verify**

```bash
PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite -x -q 2>&1 | tail -3
PYTHONPATH=src .pixi/envs/frontier/bin/python -c "from tokamak_foundation_model.ignite import eval_dynamics, maskgit; print('ok')"
```
Expected: same pass/skip set as A1 step 3, `ok`.

- [ ] **Step 5: Commit and push**

```bash
git add -A && git commit --no-edit
git push
```

### Task A3: Move Claude material out of `docs/` and delete `origin/recommender`

**Files:**
- Move: `docs/superpowers/` → `.claude/superpowers/` (merge with the existing `.claude/superpowers/specs`), `.superpowers/` → `.claude/superpowers-runtime/`
- Move to `.claude/notes/`: `docs/phase_c_step1_status.md`, `docs/spectro_video_status.md`, `docs/spectrogram_step0_findings.md`, `docs/eval_stage1_panels_patch.md`, `docs/eval_stage1_plan.md`, `docs/eval_improvement_recommendations.md`, `docs/shot-design-harness.md`, `fsq_e2e_wiring_scope.md`
- Keep in `docs/`: `CLUSTERS.md`, `E2E_ARCHITECTURE.md`, `IGNITE_CODEC_RETRAIN_SPEC.md`, `IGNITE_DESIGN.md`, `IGNITE_ROLLOUT_QUALITY_PLAN.md`, `LABELER.md`, `ResearchPlan.MD`, `SHOT_DESIGN.md`, `SHOT_DESIGN_PROGRAMS.md`, `spectrogram_tokenizer_plan.md`, `stage2_genvid_integration_plan.md`, `stage2_with_video_plan.md`, `video_tokenizer_plan.md`
- Modify: every file that links to a moved path (`git grep -l "docs/superpowers\|\.superpowers/"`).

- [ ] **Step 1: Move**

```bash
mkdir -p .claude/superpowers .claude/notes
git mv docs/superpowers/plans .claude/superpowers/plans-recommender
git mv docs/superpowers/specs/* .claude/superpowers/specs/
git rm -r --cached docs/superpowers 2>/dev/null; rm -rf docs/superpowers
git mv .superpowers .claude/superpowers-runtime
for f in phase_c_step1_status spectro_video_status spectrogram_step0_findings eval_stage1_panels_patch eval_stage1_plan eval_improvement_recommendations shot-design-harness; do git mv docs/$f.md .claude/notes/$f.md; done
git mv fsq_e2e_wiring_scope.md .claude/notes/fsq_e2e_wiring_scope.md
```

- [ ] **Step 2: Fix links**

```bash
git grep -n "docs/superpowers" -- ':!.claude' | cut -d: -f1 | sort -u
```
For each file: replace `docs/superpowers/` with `.claude/superpowers/` (plans that were under `docs/superpowers/plans/` are now `.claude/superpowers/plans-recommender/`). `tests/shot_design/test_phenomena.py` asserts a sentence appears in `docs/SHOT_DESIGN.md`; that file did not move, so no change.

- [ ] **Step 3: Verify nothing references a dead path and tests still load**

```bash
git grep -n "\.superpowers/\|docs/superpowers" -- ':!.claude' && echo "STALE LINKS" || echo "clean"
PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/shot_design/test_phenomena.py -q 2>&1 | tail -2
```

- [ ] **Step 4: Commit, push, delete the remote branch**

```bash
git add -A && git commit -m "repo: move superpowers specs/plans and working notes under .claude, keep docs/ for documentation"
git push
test "$(git rev-parse origin/nathan_dev)" = "$(git rev-parse HEAD)" && git push origin --delete recommender
git branch -r | grep -c recommender   # expect 0
```

---

## Phase B: Frontier roots, paths and environment

### Task B1: `paths.frontier.yaml` and the Frontier data roots

**Files:**
- Create: `configs/shot_design/paths.frontier.yaml`
- Test: `tests/shot_design/test_paths_frontier.py`

**Interfaces:**
- Produces: a paths file selectable by `SHOT_DESIGN_PATHS`; every key present in `paths.yaml` is present here (`config.Paths` requires them).

- [ ] **Step 1: Failing test**

```python
# tests/shot_design/test_paths_frontier.py
from pathlib import Path
import yaml
from shot_design import config

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "shot_design"

def test_frontier_paths_has_every_key_of_the_default_file():
    default = yaml.safe_load((CONFIG / "paths.yaml").read_text())
    frontier = yaml.safe_load((CONFIG / "paths.frontier.yaml").read_text())
    assert set(frontier) == set(default)

def test_frontier_paths_loads_and_points_at_proj_shared(monkeypatch):
    monkeypatch.delenv("SHOT_DESIGN_DATA_ROOT", raising=False)
    p = config.load_paths(CONFIG / "paths.frontier.yaml")
    assert str(p.data_root) == "/lustre/orion/fus187/proj-shared/nchen/shot_design"
    assert str(p.foundation_model_processed_dir) == "/lustre/orion/fus187/proj-shared/foundation_model"
    assert str(p.models_dir) == "/lustre/orion/fus187/proj-shared/nchen/shot_design/models"
```

- [ ] **Step 2: Run to fail**

`PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/shot_design/test_paths_frontier.py -q` → FAIL (file missing).

- [ ] **Step 3: Write the file**

Copy `configs/shot_design/paths.yaml` to `paths.frontier.yaml` and set:

```yaml
data_root: /lustre/orion/fus187/proj-shared/nchen/shot_design
models_dir: ${data_root}/models
staged_raw_dir: /lustre/orion/fus187/proj-shared/nchen/shot_design/staged_raw     # none on Frontier; kept so Paths validates; reader treats a missing dir as "unavailable"
text_root: /lustre/orion/fus187/proj-shared/foundation_model_text
logs_jsonl: ${text_root}/sql/logs.jsonl
shot_index_json: ${text_root}/sql/index.json
shotsummary_raw_dir: ${text_root}/shotsummary/raw
per_shot_txt_dir: ${text_root}/shotsummary/processed/per_shot_txt
qh_database_csv: ${data_root}/external/QH_Database.csv
foundation_model_processed_dir: /lustre/orion/fus187/proj-shared/foundation_model
fdp_project_dir: ${data_root}/fdp                                                 # MDSplus unreachable from OLCF
```
Keep every other key relative to `${data_root}` exactly as in `paths.yaml`. Add a header comment naming the Stellar file as the source and the rule "paths.yaml stays Stellar; this file is selected by SHOT_DESIGN_PATHS".

- [ ] **Step 4: Create the roots (one-time, idempotent)**

```bash
for d in shot_design/{db,raw,text_cache,ignite_inputs,eval,sessions,actuations,llm_cache,llm,models,outputs,runs/slurm,frame_codes,external} labeler/{events,labels,models} ollama/{bin,models,home}; do mkdir -p /lustre/orion/fus187/proj-shared/nchen/$d; done
chmod -R g+rwXs /lustre/orion/fus187/proj-shared/nchen/{shot_design,labeler,ollama}
```

- [ ] **Step 5: Run to pass, commit**

`... pytest tests/shot_design/test_paths_frontier.py -q` → 2 passed.
```bash
git add configs/shot_design/paths.frontier.yaml tests/shot_design/test_paths_frontier.py
git commit -m "shot_design: Frontier paths file selected by SHOT_DESIGN_PATHS"
```

### Task B2: `shot-design-frontier` pixi environment

**Files:**
- Modify: `pyproject.toml` (`[tool.pixi.feature.shot-design-frontier...]`, `[tool.pixi.environments]`)
- Modify: `pixi.lock` (regenerated)

**Interfaces:**
- Produces: `pixi run --frozen -e shot-design-frontier python -m shot_design ...` with `SHOT_DESIGN_PATHS`, `SHOT_DESIGN_DATA_ROOT`, `LABELER_ROOT`, `SHOT_DESIGN_CORPUS`, `HF_HUB_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false` set by activation.

- [ ] **Step 1: Add the feature**

Read `pyproject.toml` lines 211–372 first. Add after the `frontier` feature:

```toml
[tool.pixi.feature.shot-design-frontier]
platforms = ["linux-64"]

[tool.pixi.feature.shot-design-frontier.pypi-dependencies]
torch       = { version = ">=2.10,<2.11", index = "https://download.pytorch.org/whl/rocm7.1" }
torchvision = { version = ">=0.25,<0.27", index = "https://download.pytorch.org/whl/rocm7.1" }
triton-rocm = { version = "*",            index = "https://download.pytorch.org/whl/rocm7.1" }
x-transformers = "*"
vector-quantize-pytorch = "*"
einops = "*"
loguru = "*"

[tool.pixi.feature.shot-design-frontier.target.unix.activation.env]
SHOT_DESIGN_PATHS     = "/lustre/orion/fus187/scratch/nchen/FusionAIHub/configs/shot_design/paths.frontier.yaml"
SHOT_DESIGN_DATA_ROOT = "/lustre/orion/fus187/proj-shared/nchen/shot_design"
LABELER_ROOT          = "/lustre/orion/fus187/proj-shared/nchen/labeler"
SHOT_DESIGN_CORPUS    = "/lustre/orion/fus187/proj-shared/foundation_model"
HF_HUB_OFFLINE = "1"
TOKENIZERS_PARALLELISM = "false"
HDF5_USE_FILE_LOCKING = "FALSE"
```
and `shot-design-frontier = ["shot-design", "shot-design-frontier"]` under `[tool.pixi.environments]`. The `shot-design` feature's own activation block also sets the three roots (Stellar values); pixi merges activation env with later features winning — confirm with step 3, and if `shot-design`'s values win, rename this feature so it sorts after or move the three roots into a wrapper `scripts/shot_design/frontier_env.sh` that the sbatch scripts source (document which was needed).

- [ ] **Step 2: Solve and install**

```bash
pixi install --frozen -e shot-design-frontier 2>&1 | tail -5
```
If uv cannot resolve `shot-design`'s `sentence-transformers` against ROCm torch, pin `sentence-transformers = ">=3,<6"` in the new feature. If pixi rejects the two torch sources, the fallback is a uv venv: `uv venv .venv-shot-design-frontier --python 3.11 && uv pip install --index-url https://download.pytorch.org/whl/rocm7.1 torch torchvision && uv pip install -e . sentence-transformers ...` and a `scripts/shot_design/frontier_env.sh` exporting the same variables. Record the outcome in the commit body.

- [ ] **Step 3: Verify the activation and the suite**

```bash
pixi run --frozen -e shot-design-frontier bash -c 'echo $SHOT_DESIGN_DATA_ROOT $SHOT_DESIGN_PATHS; python -c "import torch, shot_design, sentence_transformers; print(torch.__version__)"'
pixi run --frozen -e shot-design-frontier pytest tests/shot_design -q -x 2>&1 | tail -3
```
Expected: the Frontier root and paths file printed, `2.10.0+rocm7.1`, suite green (or list failures verbatim; fix only ones caused by the env).

- [ ] **Step 4: Pre-populate the sentence-transformers cache (login node, network)**

```bash
HF_HUB_OFFLINE=0 pixi run --frozen -e shot-design-frontier python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
pixi run --frozen -e shot-design-frontier python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2'); print('offline ok')"
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml pixi.lock
git commit -m "pixi: shot-design-frontier env (ROCm torch + shot_design deps, Frontier activation roots)"
```

### Task B3: Frontier Slurm wrappers for shot_design

**Files:**
- Create: `scripts/slurm_frontier/shot_design_census.sh`, `shot_design_build.sh`, `shot_design_encode.sh`, `shot_design_simulate.sh` (filled in Task D4), `scripts/slurm_frontier/_shot_design_common.sh`
- Test: `tests/shot_design/test_slurm_frontier_scripts.py`

**Interfaces:**
- Produces: `_shot_design_common.sh` exporting `REPO`, `ROOT` (= `$SHOT_DESIGN_DATA_ROOT`), `PY` (= `$REPO/.pixi/envs/shot-design-frontier/bin/python`), and sourcing `_frontier_settings.sh` with `RCCL_PLUGIN=0` (single-GPU jobs need no plugin).

- [ ] **Step 1: Failing test (static checks on the scripts)**

```python
# tests/shot_design/test_slurm_frontier_scripts.py
import re
from pathlib import Path
import pytest

SLURM = Path(__file__).resolve().parents[2] / "scripts" / "slurm_frontier"
SCRIPTS = ["shot_design_census.sh", "shot_design_build.sh", "shot_design_encode.sh",
           "shot_design_simulate.sh"]

@pytest.mark.parametrize("name", SCRIPTS)
def test_frontier_shot_design_scripts_follow_house_style(name):
    text = (SLURM / name).read_text()
    assert "#SBATCH -A fus187" in text
    assert re.search(r"#SBATCH -p (batch|extended)", text)
    assert "_shot_design_common.sh" in text
    assert "/scratch/gpfs" not in text, "Stellar path leaked into a Frontier script"
    assert "gpu-stellar" not in text and "pppl" not in text
    assert "runs/slurm/%j" in text or "runs/slurm/%A_%a" in text
```

- [ ] **Step 2: Run to fail** → FAIL (files missing).

- [ ] **Step 3: Write `_shot_design_common.sh`**

```bash
#!/bin/bash
# Sourced by every scripts/slurm_frontier/shot_design_*.sh. Single-GPU or CPU jobs: no RCCL plugin.
set -euo pipefail
REPO="${REPO:-/lustre/orion/fus187/scratch/${USER}/FusionAIHub}"
cd "$REPO"
export RCCL_PLUGIN=0
source "$REPO/scripts/slurm_frontier/_frontier_settings.sh"
export SHOT_DESIGN_PATHS="$REPO/configs/shot_design/paths.frontier.yaml"
export SHOT_DESIGN_DATA_ROOT=/lustre/orion/fus187/proj-shared/nchen/shot_design
export LABELER_ROOT=/lustre/orion/fus187/proj-shared/nchen/labeler
export SHOT_DESIGN_CORPUS=/lustre/orion/fus187/proj-shared/foundation_model
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
ROOT="$SHOT_DESIGN_DATA_ROOT"
PY="$REPO/.pixi/envs/shot-design-frontier/bin/python"
mkdir -p "$ROOT/runs/slurm"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is)"
rocm-smi --showproductname 2>/dev/null | grep -m1 'Card series' || true
```
`_frontier_settings.sh` prepends the `frontier` env to PATH; `PY` points at the `shot-design-frontier` interpreter explicitly so the two envs cannot be confused.

- [ ] **Step 4: Write the three data scripts**

`shot_design_census.sh` (CPU-heavy, one node):
```bash
#!/bin/bash
#SBATCH -A fus187
#SBATCH -p batch
#SBATCH -J sd-census
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --output=/lustre/orion/fus187/proj-shared/nchen/shot_design/runs/slurm/%j.out
source "$(dirname "$0")/_shot_design_common.sh"
srun -n1 -c56 "$PY" -m shot_design corpus scan --workers 48 --out "$ROOT/db/corpus_coverage.parquet"
"$PY" -m shot_design corpus summary "$ROOT/db/corpus_coverage.parquet"
```
`shot_design_build.sh`: same header, `-t 06:00:00`, body `srun -n1 -c56 "$PY" -m shot_design build --workers 48 --list "${SHOT_LIST:-recommender_frontier_v1}" ${BUILD_ARGS:-}`.
`shot_design_encode.sh`: array of 8 tasks on one node, one GCD each:
```bash
#SBATCH --array=0-7%8
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH -c 7
#SBATCH -t 06:00:00
#SBATCH --output=/lustre/orion/fus187/proj-shared/nchen/shot_design/runs/slurm/%A_%a.out
source "$(dirname "$0")/_shot_design_common.sh"
srun "$PY" -m shot_design encode --list "${SHOT_LIST:-recommender_frontier_v1}" \
    --out "$ROOT/frame_codes" --device cuda --workers 6 --skip-existing \
    --chunk "$SLURM_ARRAY_TASK_ID" --n-chunks "${N_CHUNKS:-8}"
```
Read the Stellar `scripts/shot_design/encode.sbatch` header for the argument semantics (`--chunk/--n-chunks` split the sorted list). Create `shot_design_simulate.sh` as a header-only stub (same `#SBATCH` lines, `-q debug`, `--gres=gpu:1`, body `echo "filled in by Task D4"; exit 1`) so the style test passes now.

- [ ] **Step 5: Run to pass, commit**

```bash
git add scripts/slurm_frontier/_shot_design_common.sh scripts/slurm_frontier/shot_design_*.sh tests/shot_design/test_slurm_frontier_scripts.py
git commit -m "slurm_frontier: shot_design census/build/encode wrappers and shared env"
```

### Task B4: `labeler.jobstats` Frontier backend

**Files:**
- Modify: `src/labeler/jobstats.py` (read it fully first: `JobStats`, `parse_jobstats`, `has_utilisation`, `__main__`)
- Create: `scripts/slurm_frontier/_gpu_sampler.sh`
- Test: `tests/labeler/test_jobstats_frontier.py`

**Interfaces:**
- Produces: `parse_frontier(sacct_text: str, rocm_samples: str | None) -> JobStats`; CLI `python -m labeler.jobstats <jobid>` auto-detects Frontier when `jobstats` is not on PATH and `sacct` is.
- `_gpu_sampler.sh` writes `$ROOT/runs/slurm/<jobid>.gpu.csv` with lines `epoch_s,gpu_pct,vram_used_mb,vram_total_mb` every 30 s (`rocm-smi --showuse --showmemuse --csv`).

- [ ] **Step 1: Failing tests**

```python
# tests/labeler/test_jobstats_frontier.py
from labeler import jobstats

SACCT = """JobID|Elapsed|AllocCPUS|ReqMem|MaxRSS|TotalCPU|State
123|01:00:00|56|100G||56:00:00|COMPLETED
123.0|01:00:00|56||40G|50:24:00|COMPLETED
"""
GPU = "epoch_s,gpu_pct,vram_used_mb,vram_total_mb\n1,80,32000,65536\n2,60,32000,65536\n"

def test_parse_frontier_reads_cpu_and_gpu_utilisation():
    s = jobstats.parse_frontier(SACCT, GPU)
    assert s.job_id == "123"
    assert round(s.cpu_pct) == 90          # 50.4 h TotalCPU / (1 h * 56 cores)
    assert round(s.cpu_mem_pct) == 40      # 40G / 100G
    assert round(s.gpu_pct) == 70          # mean of samples
    assert round(s.gpu_mem_pct) == 49      # 32000/65536
    assert jobstats.has_utilisation(s)

def test_parse_frontier_without_gpu_samples_is_cpu_only():
    s = jobstats.parse_frontier(SACCT, None)
    assert s.gpu_pct is None and s.gpu_mem_pct is None
```
Adapt field names to the real `JobStats` attributes after reading the class; keep the numbers.

- [ ] **Step 2: Run to fail** → `AttributeError: parse_frontier`.

- [ ] **Step 3: Implement `parse_frontier` and the auto-detect**

Parse `sacct -P -o JobID,Elapsed,AllocCPUS,ReqMem,MaxRSS,TotalCPU,State` output: batch row supplies Elapsed/AllocCPUS/ReqMem; the `.0` step supplies MaxRSS/TotalCPU (use the max over steps). Reuse `parse_duration`, `parse_size`, `parse_req_mem`. GPU: mean of `gpu_pct`, and `max(vram_used)/vram_total`. In `main()`, when `shutil.which("jobstats") is None and shutil.which("sacct")`, run the `sacct` command above for the job id and read `$SHOT_DESIGN_DATA_ROOT/runs/slurm/<jobid>.gpu.csv` if present.

- [ ] **Step 4: Sampler script**

```bash
#!/bin/bash
# Background GPU sampler for Frontier jobs: source _shot_design_common.sh first, then
#   bash "$REPO/scripts/slurm_frontier/_gpu_sampler.sh" "$ROOT/runs/slurm/$SLURM_JOB_ID.gpu.csv" &
out="$1"; echo "epoch_s,gpu_pct,vram_used_mb,vram_total_mb" > "$out"
while true; do
  use=$(rocm-smi --showuse --csv 2>/dev/null | awk -F, 'NR==2{print $2}')
  mem=$(rocm-smi --showmemuse --csv 2>/dev/null | awk -F, 'NR==2{print $2}')
  tot=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR==2{print int($2/1048576)}')
  echo "$(date +%s),${use:-0},${mem:-0},${tot:-0}" >> "$out"; sleep 30
done
```
Add the sampler line (backgrounded, killed at exit with `trap 'kill %1' EXIT`) to `shot_design_encode.sh` from B3.

- [ ] **Step 5: Run to pass, commit**

```bash
git add src/labeler/jobstats.py scripts/slurm_frontier/_gpu_sampler.sh scripts/slurm_frontier/shot_design_encode.sh tests/labeler/test_jobstats_frontier.py
git commit -m "labeler: jobstats Frontier backend from sacct + rocm-smi samples"
```

---

### Task B5: Rename `ideate` to `shot-design` everywhere live

User directive 2026-09-19 16:11: "rename all the ideate to shot_design". Pixi rejects underscores in environment names (measured: `Failed to parse environment name 'shot_design', please use only lowercase letters, numbers and dashes`), so environment/feature names use the dash form; the Python package stays `shot_design`.

**Mapping (exact):**
- pixi features and environments: `ideate` → `shot-design`, `ideate-cpu` → `shot-design-cpu`, `ideate-frontier` → `shot-design-frontier` (in `pyproject.toml` `[tool.pixi.feature.*]`, `[tool.pixi.environments]`, task/comment text; regenerate `pixi.lock` so its environment keys follow; the installed dir becomes `.pixi/envs/shot-design-frontier`).
- `.mcp.json` `-e ideate-cpu` → `-e shot-design-cpu`; `src/shot_design/mcp/server.py` resource `ideate://manifest` → `shot-design://manifest`; `src/shot_design/ui/app.py` `COOKIE = "ideate_token"` → `"shot_design_token"`.
- Schema tags: `src/shot_design/shotdb/legacy_raw.py` `SCHEMA = "ideate-raw-v1"` → `"shot-design-raw-v1"`, `src/shot_design/design/provenance.py` `SCHEMA = "ideate-frame-codes-provenance-v1"` → `"shot-design-frame-codes-provenance-v1"`; wherever a reader compares the tag, also accept the old string (`LEGACY_SCHEMAS = {"ideate-raw-v1"}`) so Stellar files still load. Test: a file stamped with the old tag validates.
- Every `pixi run -e ideate*`, `.pixi/envs/ideate*` and prose "ideate" in `scripts/`, `src/`, `tests/`, `docs/`, `AGENTS.md`, `data/events/README.md`, `configs/shot_design/*.yaml` comments, and the CURRENT plan/spec (`.claude/superpowers/{plans,specs}/2026-09-19-*`).
- Slurm scripts under `scripts/shot_design/*.sbatch` and `scripts/slurm_frontier/_shot_design_common.sh`: env dir names as above.

**Do NOT rename:** the Stellar filesystem path string `/scratch/gpfs/EKOLEMEN/nc1514/ideate` (a real directory on another cluster; renaming the string cannot rename the directory), and the archival `.claude/superpowers/plans-recommender/`, `.claude/superpowers/specs/2026-09-0*`/`2026-09-1[0-8]*`, `.claude/superpowers-runtime/` files (history), and `tests/labeler/data/jobstats/*.txt` fixtures (verbatim Slurm output).

- [ ] **Step 1:** `grep -rIn "ideate" --exclude-dir=.pixi --exclude-dir=.git --exclude-dir=.superpowers --exclude-dir=node_modules . | grep -v "^./.claude/superpowers-runtime\|^./.claude/superpowers/plans-recommender\|^./.claude/superpowers/specs/2026-09-0\|^./.claude/superpowers/specs/2026-09-1[0-8]\|/scratch/gpfs/EKOLEMEN/nc1514/ideate\|tests/labeler/data/jobstats" > /tmp/ideate_before.txt; wc -l /tmp/ideate_before.txt`
- [ ] **Step 2:** Apply the mapping with `sed -i` per file class (dash form for env names, `shot_design_token` for the cookie, `shot-design://` for the MCP resource, `shot-design-` for schema tags, plain "shot design"/"shot_design" for prose as reads naturally). Add the legacy-tag acceptance + its test.
- [ ] **Step 3:** Re-run the Step 1 grep: expected 0 lines. `pixi install --frozen -e shot-design-frontier` (this replaces B2's install if B2 has not finished); `pixi run --frozen -e shot-design-frontier pytest tests/shot_design tests/labeler -q`.
- [ ] **Step 4:** `git mv` nothing (no file is named ideate outside archives); commit `git commit -m "repo: rename ideate environments and tags to shot-design"`.

## Phase C: IGNITE v4 migration

### Task C0: Merge Peter's v4 training code (`peter/dev-peter` @ 35cfed1)

Added 2026-09-19 18:55 by controller ruling. C1 found that the v4 codecs and the `mskfull` dynamics
checkpoint were produced by code that is not on `nathan_dev`: Peter's clone
`/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub` is on `dev-peter` at `35cfed1`, 47
unpushed commits past the `f71acd4` merged in Task A2. Measured on the login node with the same
interpreter: with Peter's `src` on `PYTHONPATH`, `train_dynamics._load_codec` loads all 15 v4
codecs strictly and `eval_dynamics.load_model` loads `mskfull/dynamics_best.pt` (step 3200,
300,815,000 params, 15 modalities, `actuator_dim 70 -> 88` inferred from
`backbone.act_embed.weight`); with `nathan_dev`'s `src`, 6 codecs fail (`SpectroCodec` lacks
`decoder.refine.*`, `FastTSCodec` lacks `encoder.gain_to_tokens.*`) and the dynamics load fails on
`act_embed` 70 vs 88. His 5 uncommitted files are NOT needed for loading (verified against his
committed HEAD) and are not taken. The ref is already fetched as `refs/remotes/peter/dev-peter`
(if absent: `git -c safe.directory='*' fetch /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub dev-peter:refs/remotes/peter/dev-peter`).

**Files (the seven that conflict):**
- Modify: `scripts/slurm_frontier/_frontier_common.sh`, `scripts/slurm_frontier/train_dynamics.sh`
- Modify: `src/tokamak_foundation_model/ignite/dynamics.py`, `dynamics_config.py`, `eval_dynamics.py`, `maskgit.py`, `train_dynamics.py`
- Create: `tests/ignite/test_v4_assets_load.py`
- Everything else merges cleanly (Peter deleted no files since the base; our `sampling.py`, `scoring.py`, `selfforce.py`, `text_embed.py` stay).

**Interfaces:**
- Produces: after the merge, `train_dynamics._load_codec(family, path)` loads every `ignite_codecs_v4/<m>/codec_best.pt` strictly; `eval_dynamics.load_model(ckpt_path, device)` loads `mskfull/dynamics_best.pt` and returns `actuator_dim == 88` on its config; `dynamics_config.modalities_from_manifest` (C1) is unchanged.

**Conflict rules (the spec is v4 end to end, so Peter's semantics win wherever they decide how a v4 checkpoint is built or loaded):**
1. Model construction and checkpoint loading (`maskgit.py`, `dynamics.py`, `dynamics_config.py` fields, `eval_dynamics.load_model`, codec loading in `train_dynamics.py`): take Peter's side.
2. Our additive, flag-gated features stay and must still be off by default and bit-identical when off: per-shot text-embedding conditioning (`--text_*`, `text_embed.py`), `--snapshot_every`, the Gumbel reveal (`IGNITE_MASKGIT_GUMBEL`), the actuator-counterfactual and honest-video eval panels, the removed diff-panel remnants (do not resurrect `CMAP_DIFF`). Where both sides changed the same function, the result carries both behaviours.
3. `dynamics_config.py`: keep C1's `modalities_from_manifest` verbatim and Peter's new fields.
4. Slurm wrappers: keep our `_frontier_common.sh` / `train_dynamics.sh` structure (they source `_frontier_settings.sh` and carry the RCCL plugin fix); bring over Peter's new environment variables and flags only where they are not already present. `bash -n` both.

- [ ] **Step 1: Baseline failure list before merging**

```bash
pixi run --frozen -e shot-design-frontier pytest tests/ignite tests/shot_design -q -p no:cacheprovider --ignore=tests/ignite/test_train_codec.py -rf 2>&1 | grep -E "^FAILED|passed|failed" > /tmp/c0_before.txt
```
(`test_train_codec.py` trainer-loop tests abort natively on a login node — Peter's note 34dc6c3. Never run `tests/labeler/test_labels_layout_integration.py` here.)

- [ ] **Step 2: Failing test**

```python
# tests/ignite/test_v4_assets_load.py
from pathlib import Path
import pytest

CODECS = Path("/lustre/orion/fus187/proj-shared/models/ignite_codecs_v4")
DYN = Path("/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/runs/mskfull/dynamics_best.pt")
FAMILIES = {"ece": "spectro", "bes": "spectro", "mhr": "spectro", "co2": "spectro",
            "mirnov": "spectro", "tangtv_lower": "video", "tangtv_upper": "video",
            "ts_core_density": "slowts", "ts_core_temp": "slowts",
            "ts_tangential_density": "slowts", "ts_tangential_temp": "slowts",
            "cer_ti": "slowts", "cer_rot": "slowts", "mse": "slowts", "filterscopes": "fastts"}

pytestmark = pytest.mark.skipif(not CODECS.exists(), reason="v4 assets are Frontier-only")


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_v4_codec_loads_strictly(name):
    from tokamak_foundation_model.ignite import train_dynamics as td
    codec = td._load_codec(FAMILIES[name], CODECS / name / "codec_best.pt")
    assert codec is not None


def test_mskfull_dynamics_checkpoint_loads_with_88_actuators():
    from tokamak_foundation_model.ignite import eval_dynamics as ed
    model, cfg, step = ed.load_model(DYN, "cpu")
    assert step == 3200
    assert len(cfg.modalities) == 15
    assert model.backbone.act_embed.weight.shape[1] == 88
```

Run: `pixi run --frozen -e shot-design-frontier pytest tests/ignite/test_v4_assets_load.py -q` — expected: 6 codec cases and the dynamics case FAIL on `nathan_dev` before the merge.

- [ ] **Step 3: Merge and resolve**

```bash
git merge --no-ff --no-commit peter/dev-peter
git diff --name-only --diff-filter=U     # the seven files
# resolve per the conflict rules; then
bash -n scripts/slurm_frontier/_frontier_common.sh scripts/slurm_frontier/train_dynamics.sh
git add -A -- scripts/slurm_frontier src/tokamak_foundation_model tests
git commit -m "merge peter/dev-peter 35cfed1: v4 codec stack, live-frame weighting, actuator_dim inference"
```

- [ ] **Step 4: Verify**

```bash
pixi run --frozen -e shot-design-frontier pytest tests/ignite/test_v4_assets_load.py -q          # 16 passed
pixi run --frozen -e shot-design-frontier pytest tests/ignite tests/shot_design -q -p no:cacheprovider --ignore=tests/ignite/test_train_codec.py -rf 2>&1 | grep -E "^FAILED|passed|failed" > /tmp/c0_after.txt
diff /tmp/c0_before.txt /tmp/c0_after.txt
```
Expected: every FAILED line in `before` that is not in `after` is a fix; any FAILED line in `after` not in `before` is a regression to fix before reporting (except `tests/shot_design/test_ignite.py::test_codecs_expose_the_encode_then_quantize_contract` if it now fails only on the encodable-name set — that is C2's `mirnov` work; report it).

- [ ] **Step 5: Commit the test**

```bash
git add tests/ignite/test_v4_assets_load.py
git commit -m "ignite: test that the pinned v4 codecs and mskfull dynamics checkpoint load"
```

### Task C1: Manifest-driven modality table and `shot_design model --pin`

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/dynamics_config.py` (add `modalities_from_manifest`)
- Modify: `src/shot_design/shotdb/ignite.py` (`model_cfg`, `bundle_dir`, new `pin_bundle`, `load_codecs` manifest read, vocab check)
- Modify: `configs/shot_design/ignite_modalities.yaml` (`model:` block)
- Modify: `src/shot_design/cli.py` (`model` subcommand: `--pin`, `--check`)
- Test: `tests/shot_design/test_ignite_v4.py`, `tests/ignite/test_dynamics_config_manifest.py`

**Interfaces:**
- Produces: `dynamics_config.modalities_from_manifest(path: Path) -> tuple[ModalitySpec, ...]` reading `{"modalities": {name: {"family", "n_tok", "codebook_size"}}}` in file order.
- `shotdb.ignite.pin_bundle(paths, *, codec_tmpl: str, dynamics_src: Path, names: list[str], t0_start: float) -> Path` writes `<models_dir>/<local_name>/codecs/<m>/codec_best.pt` (real copies, `shutil.copy2`, symlinks resolved), `codecs/MANIFEST.json`, copies the dynamics file to `<local_name>/<dynamics_file>` and records sha256 of every file in the manifest under `"sha256"`.
- `shotdb.ignite.check_bundle(paths) -> list[str]` returns mismatches (empty = ok). `load_codecs` calls it and raises `CheckpointMissing` listing the mismatches when non-empty.
- `ignite_modalities.yaml` `model:` block:

```yaml
model:
  generation: v4
  local_name: IGNITE_v4
  codec_tmpl: /lustre/orion/fus187/proj-shared/models/ignite_codecs_v4/{m}/codec_best.pt
  dynamics_src: /lustre/orion/fus187/proj-shared/models/ignite_prod_v4/runs/mskfull/dynamics_best.pt
  dynamics_file: ignite_dynamics_prod_v4_mskfull_step3200.pt
  frame_codes_cache: /lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes
  frame_tokens: 1209
  t0_start_s: 1.0
  window_ms: 250
  production_vocabs: {ece: 1000, bes: 1000, mhr: 1000, co2: 1000, mirnov: 1000, tangtv_lower: 1000, tangtv_upper: 1000, ts_core_density: 1000, ts_core_temp: 1000, ts_tangential_density: 1000, ts_tangential_temp: 1000, cer_ti: 1000, cer_rot: 1000, mse: 1000, filterscopes: 1000}
  families: {ece: spectro, bes: spectro, mhr: spectro, co2: spectro, mirnov: spectro, tangtv_lower: video, tangtv_upper: video, ts_core_density: slowts, ts_core_temp: slowts, ts_tangential_density: slowts, ts_tangential_temp: slowts, cer_ti: slowts, cer_rot: slowts, mse: slowts, filterscopes: fastts}
  n_tok: {ece: 192, bes: 192, mhr: 192, co2: 192, mirnov: 192, tangtv_lower: 108, tangtv_upper: 108, ts_core_density: 4, ts_core_temp: 4, ts_tangential_density: 4, ts_tangential_temp: 4, cer_ti: 4, cer_rot: 4, mse: 4, filterscopes: 5}
  # v2 bundle kept for reference / rollback: repo_id nc1/IGNITE rev d0cfe4f73d0287d5e12d724f8d77ed651fcfa47a
```
Keep the existing long header comment but rewrite the SCOPE paragraph for 15 modalities / 1209 tokens. Add a `mirnov:` entry to the `modalities:` map below (copy `mhr`'s structure; `staged.cols` listing the 29 mirnov channels from `data_loader.SIGNAL_CONFIGS`; `n_channels: 29`).

- [ ] **Step 1: Failing tests**

```python
# tests/ignite/test_dynamics_config_manifest.py
import json
from tokamak_foundation_model.ignite import dynamics_config as dc

def test_modalities_from_manifest_preserves_order_and_totals(tmp_path):
    m = {"modalities": {"ece": {"family": "spectro", "n_tok": 192, "codebook_size": 1000},
                        "mirnov": {"family": "spectro", "n_tok": 192, "codebook_size": 1000},
                        "filterscopes": {"family": "fastts", "n_tok": 5, "codebook_size": 1000}}}
    p = tmp_path / "MANIFEST.json"; p.write_text(json.dumps(m))
    mods = dc.modalities_from_manifest(p)
    assert [x.name for x in mods] == ["ece", "mirnov", "filterscopes"]
    assert sum(x.n_tok for x in mods) == 389
    assert mods[1] == dc.ModalitySpec("mirnov", "spectro", 192, 1000)
```

```python
# tests/shot_design/test_ignite_v4.py
import hashlib, json, torch, pytest
from shot_design.shotdb import ignite

def _fake_codec(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": {"d_model": 8}, "codec": {}}, path)

def test_pin_bundle_copies_resolves_symlinks_and_records_sha(paths, tmp_path):
    src = tmp_path / "codecs_v4"; real = tmp_path / "real"
    _fake_codec(real / "ece.pt"); (src / "ece").mkdir(parents=True)
    (src / "ece" / "codec_best.pt").symlink_to(real / "ece.pt")
    dyn = tmp_path / "dyn.pt"; torch.save({"step": 3200, "modalities": [("ece", "spectro", 192, 1000)], "model": {}}, dyn)
    out = ignite.pin_bundle(paths, codec_tmpl=str(src / "{m}" / "codec_best.pt"), dynamics_src=dyn, names=["ece"], t0_start=1.0)
    copied = out / "codecs" / "ece" / "codec_best.pt"
    assert copied.exists() and not copied.is_symlink()
    man = json.loads((out / "codecs" / "MANIFEST.json").read_text())
    assert man["modalities"]["ece"] == {"family": "spectro", "n_tok": 192, "codebook_size": 1000}
    assert man["sha256"]["codecs/ece/codec_best.pt"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert man["t0_start_s"] == 1.0 and man["frame_tokens"] == 192
    assert ignite.check_bundle(paths) == []

def test_check_bundle_reports_a_changed_codec(paths, tmp_path):
    test_pin_bundle_copies_resolves_symlinks_and_records_sha(paths, tmp_path)
    p = ignite.bundle_dir(paths) / "codecs" / "ece" / "codec_best.pt"
    p.write_bytes(b"tampered")
    bad = ignite.check_bundle(paths)
    assert bad and "codecs/ece/codec_best.pt" in bad[0]
    with pytest.raises(ignite.CheckpointMissing):
        ignite.load_codecs(ignite.bundle_dir(paths))

def test_model_cfg_declares_fifteen_v4_modalities():
    cfg = ignite.model_cfg()
    assert cfg["generation"] == "v4"
    assert len(cfg["production_vocabs"]) == 15 and set(cfg["production_vocabs"].values()) == {1000}
    assert sum(cfg["n_tok"].values()) == cfg["frame_tokens"] == 1209
    assert cfg["t0_start_s"] == 1.0
```
The `pin_bundle` test uses `names=["ece"]` so `frame_tokens` in that manifest is 192; the function computes it from `n_tok` of the pinned names.

- [ ] **Step 2: Run to fail** → `AttributeError`.

- [ ] **Step 3: Implement**

`dynamics_config.modalities_from_manifest`: 6 lines, `json.loads`, tuple comprehension in dict order.

`shotdb/ignite.py`:
- `pin_bundle`: for each name, `src = Path(codec_tmpl.format(m=name)).resolve()`; copy to `bundle_dir(paths)/codecs/<name>/codec_best.pt`; copy `dynamics_src` to `bundle_dir(paths)/model_cfg()["dynamics_file"]`; write `MANIFEST.json`:
  `{"_meta": {"created", "generation", "codec_tmpl", "dynamics_src", "copy_mode": "shutil.copy2, symlinks resolved"}, "modalities": {name: {"family", "n_tok", "codebook_size"}}, "t0_start_s", "frame_tokens", "sha256": {relpath: hex}}` with families/n_tok/vocab from `model_cfg()["families"|"n_tok"|"production_vocabs"]`.
- `check_bundle`: recompute sha256 for each entry; return `[f"{rel}: expected {a[:12]} got {b[:12]}"]`.
- `load_codecs`: after reading `entries`, call `check_bundle` when `manifest` has `sha256`; raise `CheckpointMissing("pinned bundle changed on disk: ...")` if non-empty. Everything else unchanged (`entries[name]["family"]` is still what it reads).
- `download_bundle` stays for `generation: v2`; `cli.py` `model` gains `--pin` (calls `pin_bundle` with the yaml values and all 15 names) and `--check` (prints `check_bundle` result, exit 1 if non-empty); `--download` prints "not used for generation v4" when `generation != "v2"`.

- [ ] **Step 4: Run to pass**

```bash
PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_dynamics_config_manifest.py tests/shot_design/test_ignite_v4.py tests/shot_design/test_ignite.py -q
```

- [ ] **Step 5: Pin the real bundle (login node, ~4 GB copy) and commit**

```bash
pixi run --frozen -e shot-design-frontier python -m shot_design model --pin && pixi run --frozen -e shot-design-frontier python -m shot_design model --check
ls -la /lustre/orion/fus187/proj-shared/nchen/shot_design/models/IGNITE_v4/codecs | head -20
git add configs/shot_design/ignite_modalities.yaml src/shot_design/shotdb/ignite.py src/shot_design/cli.py src/tokamak_foundation_model/ignite/dynamics_config.py tests/ignite/test_dynamics_config_manifest.py tests/shot_design/test_ignite_v4.py
git commit -m "ignite: v4 generation pinned by sha256 manifest; 15-modality table from manifest"
```

### Task C2: `mirnov` through the encode path and cache-first frame codes

**Files:**
- Modify: `src/shot_design/shotdb/ignite.py` (`frame_codes`: cache lookup; `_frames`: accept `mirnov`)
- Modify: `src/shot_design/design/program_reference.py` (`_cache_path` searches `model_cfg()["frame_codes_cache"]` first)
- Modify: `src/shot_design/design/seed.py` (docstring 14 → 15; `wanted_modalities` uses `model_cfg()["families"]`)
- Modify: `src/tokamak_foundation_model/ignite/train_dynamics.py` (`FROZEN_CODEC_CKPTS` gets a `"mirnov": ("spectro", "eval_runs/ignite_d5_mirnov/codec_best.pt")` entry so `resolve_codec_path` and `_load_codec` accept the name; `actuator_frames` unchanged)
- Test: `tests/shot_design/test_ignite_v4.py` (extend), `tests/shot_design/test_seed.py` (update counts)

**Interfaces:**
- Produces: `program_reference._cache_path(shot, paths)` order = `[model_cfg()["frame_codes_cache"], paths.data_root/"frame_codes", bundle_dir(paths)/"frame_codes"]`; `validate_cache` also checks `set(cache["vocabs"]) == set(model_cfg()["production_vocabs"])` and values equal.

- [ ] **Step 1: Failing tests**

```python
def test_cache_path_prefers_the_production_v4_cache(paths, tmp_path, monkeypatch):
    from shot_design.design import program_reference as pr
    prod = tmp_path / "prod"; prod.mkdir(); (prod / "190000.pt").write_bytes(b"x")
    monkeypatch.setitem(ignite.model_cfg(), "frame_codes_cache", str(prod))
    assert pr._cache_path(190000, paths) == prod / "190000.pt"

def test_validate_cache_rejects_v2_vocabs():
    from shot_design.design import program_reference as pr
    cache = {"codes": {}, "actuators": torch.zeros(1, 88), "n_frames": 1,
             "vocabs": {m: 1000 for m in ignite.model_cfg()["production_vocabs"]}}
    cache["vocabs"]["ece"] = 32768
    with pytest.raises(ValueError, match="ece"):
        pr.validate_cache(cache)

def test_wanted_modalities_includes_mirnov():
    from shot_design.design import seed
    assert "mirnov" in seed.wanted_modalities(ignite.model_cfg()["families"])   # adapt to the real signature after reading seed.py:100
```
`model_cfg()` returns the cached yaml dict; `monkeypatch.setitem` on it is enough because `load_yaml` caches by mtime.

- [ ] **Step 2: Run to fail.**

- [ ] **Step 3: Implement.** Read `program_reference.py:46-70`, `seed.py:100-135`, `shotdb/ignite.py:380-415`. `frame_codes(shot, ...)`: if `Path(cfg["frame_codes_cache"]) / f"{shot}.pt"` exists, `torch.load` it, run the vocab check, return it; else encode as today. Update `tests/shot_design/test_seed.py` expectations from 14 to 15 modalities where they count.

- [ ] **Step 4: Live check on 3 cached shots (login node, CPU is fine for this)**

```bash
pixi run --frozen -e shot-design-frontier python - <<'EOF'
import torch
from shot_design.config import load_paths
from shot_design.design import program_reference as pr
p = load_paths()
for s in (190000, 190090, 199597):
    c = torch.load(pr._cache_path(s, p), map_location="cpu", weights_only=False)
    pr.validate_cache(c); print(s, c["n_frames"], tuple(c["actuators"].shape), len(c["codes"]))
EOF
```
Expected: three lines, 15 codes each, `(F, 88)`.

- [ ] **Step 5: Run the suite, commit**

```bash
pixi run --frozen -e shot-design-frontier pytest tests/shot_design -q 2>&1 | tail -3
git add -A src/shot_design src/tokamak_foundation_model/ignite/train_dynamics.py tests/shot_design
git commit -m "ignite v4: mirnov in the encode path, production cache first, vocab validation against the pinned generation"
```

### Task C3: Loader reads `actuator_dim`; G-ENC gate against the v4 cache (GPU)

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/eval_dynamics.py:121-142` (`load_model`)
- Modify: `scripts/shot_design/g_enc.py` (compare fresh encodes with `frame_codes_cache`, 15 modalities)
- Modify: `src/shot_design/shotdb/build.py:296-309` (`frame_codes_dirs`: production cache first, v4 bundle dir)
- Modify: `src/shot_design/design/program_reference.py` (`_cache_path` reuses `build.frame_codes_dirs`)
- Create: `scripts/slurm_frontier/shot_design_genc.sh`
- Test: `tests/ignite/test_load_model_actuator_dim.py`
- Test: `tests/shot_design/test_build.py` (or the existing build test module) — `frame_codes_dirs` order

**Controller amendment 2026-09-19 20:10 (ruling, from the C2 implementer's follow-ups):**
`shotdb.build.frame_codes_dirs` still returns the v2 pair `(<data_root>/frame_codes, <models_dir>/IGNITE/frame_codes)`. It is the single source for `build`'s `has_frame_codes` column, `corpus select`'s preferred-shots set (`cli.py:608, 978`) and `frame_codes_path`. Left as is, Task F1 would select 5000 shots blind to the 8752 shots already in the production cache and Task F2 Step 5 would re-encode them on 8 GCDs. Fix it here, before any F-phase job runs.

- [ ] **Step 0a: Failing test** (append to the module that already tests `shotdb.build`; otherwise create `tests/shot_design/test_build_frame_codes_dirs.py`):

```python
from pathlib import Path
from shot_design.shotdb import build, ignite

def test_frame_codes_dirs_production_cache_first_then_v4_bundle(paths, monkeypatch):
    monkeypatch.setattr(ignite, "model_cfg", lambda: {**ignite.model_cfg(), "frame_codes_cache": "/prod/frame_codes"})
    dirs = build.frame_codes_dirs(paths)
    assert dirs[0] == Path("/prod/frame_codes")
    assert dirs[1] == Path(paths.data_root) / "frame_codes"
    assert dirs[2] == ignite.bundle_dir(paths) / "frame_codes"
    assert Path(paths.models_dir) / "IGNITE" / "frame_codes" not in dirs

def test_frame_codes_dirs_without_a_production_cache(paths, monkeypatch):
    monkeypatch.setattr(ignite, "model_cfg", lambda: {k: v for k, v in ignite.model_cfg().items() if k != "frame_codes_cache"})
    dirs = build.frame_codes_dirs(paths)
    assert dirs == (Path(paths.data_root) / "frame_codes", ignite.bundle_dir(paths) / "frame_codes")
```

(`monkeypatch.setitem(ignite.model_cfg(), …)` is a no-op because `load_yaml` deep-copies — patch the function, as Task C2's tests do.)

- [ ] **Step 0b: Implement.** `frame_codes_dirs` returns `(Path(production), data_root/"frame_codes", bundle_dir(paths)/"frame_codes")` with `production = ignite.model_cfg().get("frame_codes_cache")`, omitting the first entry when unset; drop `<models_dir>/IGNITE`. Then make `program_reference._cache_path` iterate `build.frame_codes_dirs(paths)` instead of its own `roots` list (same order; one source of truth). `build.py` deliberately never imports `ignite` at module level (its line-14 comment; see the lazy `from . import ignite` at `build.py:1074, 1293`) — keep that: import `ignite` inside `frame_codes_dirs`. `design/program_reference.py` may import `build` at module level only if `build` does not import `design`; check with `grep -n "^from\|^import" src/shot_design/shotdb/build.py`, and if it does, put the tuple in `shotdb/ignite.py` as `frame_codes_dirs(paths)` and have both callers use it. Run `pytest tests/shot_design -q`; the existing `_cache_path` tests must stay green.

- [ ] **Step 0b-ii: Hermeticity (C2 review, Important, plan-mandated).** `_cache_path`/`frame_codes_dirs` read `model_cfg()["frame_codes_cache"]`, an absolute Frontier path, so any test using a real shot number in the production range silently reads production data instead of its tmp tree. Add to `tests/shot_design/conftest.py` an autouse fixture that patches `shot_design.shotdb.ignite.model_cfg` to return the yaml WITHOUT `frame_codes_cache`; tests that want a production cache set it themselves afterwards (their own `monkeypatch.setattr(ignite, "model_cfg", ...)` runs later and wins — Task C2's `test_ignite_v4.py` cache tests do exactly this; confirm they still pass). Test for the fixture itself: `def test_tests_never_see_the_production_cache(): assert "frame_codes_cache" not in ignite.model_cfg()`. Also point `program_reference._cache_path` at `ignite.production_cache_path(shot)` / `frame_codes_dirs` rather than re-deriving the root (C2 review, Minor 1) — Step 0b already does this if `_cache_path` iterates `frame_codes_dirs`.

- [ ] **Step 0c:** In `g_enc.py`, `FULL_SHOT_FRAMES = 239` is the v2 count (t0 0.05 s). v4 starts at `t0_start_s: 1.0`, so a full shot is 219 frames (the C2 validation measured 219 on shots 190000, 190090, 204346). Set `FULL_SHOT_FRAMES = 219` and derive the expected count from the cache file's `n_frames` when it is present, with the constant as the fallback; update its comment.

- [ ] **Step 1: Failing test**

```python
import torch
from tokamak_foundation_model.ignite import eval_dynamics, dynamics_config as dc, maskgit

def test_load_model_builds_the_checkpoints_actuator_width(tmp_path):
    mods = (dc.ModalitySpec("ece", "spectro", 4, 16), dc.ModalitySpec("mse", "slowts", 2, 16))
    cfg = dc.DynamicsConfig(modalities=mods, d_model=32, depth=1, n_heads=2, actuator_dim=88, grad_checkpointing=False)
    m = maskgit.MaskGITDynamics(cfg)
    ck = {"model": m.state_dict(), "cfg_depth": 1, "cfg_d_model": 32, "cfg_n_heads": 2,
          "modalities": [tuple(x) for x in mods], "step": 7}
    p = tmp_path / "d.pt"; torch.save(ck, p)
    model, cfg2, step = eval_dynamics.load_model(p, "cpu")
    assert cfg2.actuator_dim == 88 and step == 7
```

- [ ] **Step 2: Run.** Controller amendment 2026-09-19 19:45: Task C0 merged Peter's `eval_dynamics.load_model`, which already infers `actuator_dim` from `backbone.act_embed.weight` (it printed `actuator_dim 70 -> 88` when loading `mskfull`). Expected: this test PASSES as written. Keep the test (it pins the behaviour), skip Step 3, and go to Step 4. Only if it fails, do Step 3.

- [ ] **Step 3 (only if Step 2 failed): Implement** in `load_model` before `DynamicsConfig(**kw)`:

```python
    if "cfg_actuator_dim" in ck:
        kw["actuator_dim"] = int(ck["cfg_actuator_dim"])
    else:
        w = ck["model"].get("backbone.act_embed.weight")
        if w is not None:
            kw["actuator_dim"] = int(w.shape[1])
```
Confirm the key name by `grep -n "act_embed" src/tokamak_foundation_model/ignite/maskgit.py`.

- [ ] **Step 4: G-ENC on Frontier (one GCD, debug queue)**

Rewrite `scripts/shot_design/g_enc.py` to take `--cache-dir` (default `model_cfg()["frame_codes_cache"]`) and `--shots` (default 5 shots present in the cache: 190000 190090 199597 190735 190736), encode each with the pinned v4 codecs at `t0_start=1.0`, and print per-modality exact-match fraction; exit 1 if any non-spectro/video modality is below 1.0 or any spectro/video below 0.99. `shot_design_genc.sh`: `-q debug -t 01:00:00 --gres=gpu:1`, body `srun "$PY" scripts/shot_design/g_enc.py --device cuda --out "$ROOT/runs/genc_v4.json"`.

```bash
sbatch scripts/slurm_frontier/shot_design_genc.sh && squeue -u $USER
```
Wait; read `$ROOT/runs/genc_v4.json`. If a spectro modality misses the bar, the STFT/standardisation used by `CodecPairDataset` differs from Peter's cache builder for v4: record the measured fractions in the commit body and in `docs/models/ignite.md` (Task H2) and continue, since rollouts seed from the cache anyway.

- [ ] **Step 5: Commit**

```bash
git add src/shot_design/shotdb/build.py src/shot_design/design/program_reference.py tests/shot_design/test_build_frame_codes_dirs.py
git commit -m "shot_design: frame_codes_dirs lists the production cache first and the v4 bundle"
git add src/tokamak_foundation_model/ignite/eval_dynamics.py scripts/shot_design/g_enc.py scripts/slurm_frontier/shot_design_genc.sh tests/ignite/test_load_model_actuator_dim.py
git commit -m "ignite: pin actuator_dim inference in load_model; G-ENC gate against the v4 production cache"
```

---

## Phase D: Simulation stage

### Task D1: `shot_design.simulate.core` — seed assembly and paired rollout (torch only)

**Files:**
- Create: `src/shot_design/simulate/__init__.py`, `src/shot_design/simulate/core.py`
- Test: `tests/shot_design/test_simulate_core.py`

**Interfaces:**
- Consumes: the design seed `.pt` written by `program.export_ignite` = `{"codes": {m: (F, n_tok) int32}, "actuators": (F, 88) float16 | None, "n_frames": F, "vocabs": {m: int}}` (this is the reference cache sliced to the design window with the PROPOSED actuators already z-scored by `design.actuators.z_score` — verify by reading `program.py:340-500` before coding; if `output` there is the reference's real actuators with the proposal applied, then the real arm is `reference cache["actuators"]` over the same slice).
- Produces:

```python
@dataclass
class SimulationArms:
    seed_frames: int          # k0, default 20
    predict_frames: int       # default 80
    real: dict[str, torch.Tensor]       # {m: (F, n_tok)} predicted codes with the reference actuators
    proposed: dict[str, torch.Tensor]   # same shape, proposed actuators
    gt: dict[str, torch.Tensor]         # reference cache codes over the same frames
    divergence_vs_real: dict[str, float]   # fraction of tokens that differ, predicted region only
    token_accuracy: dict[str, float]       # vs gt, predicted region only
    persistence_accuracy: dict[str, float] # last seed frame repeated, vs gt

def load_dynamics(paths, device) -> tuple[torch.nn.Module, DynamicsConfig, int]   # eval_dynamics.load_model on bundle_dir/dynamics_file
def actuator_arms(reference_cache: dict, design_seed: dict, k0: int, n_predict: int) -> tuple[torch.Tensor, torch.Tensor]  # (real, proposed), each (k0+n_predict, 88) float32; the first k0 frames are identical (seed uses measured controls)
def run_paired(model, cfg, codes: dict, real_act, prop_act, *, seed: int, temperature: float = 1.0, decode_steps: int = 10) -> SimulationArms
```
`run_paired` calls the repo's own rollout: read `src/tokamak_foundation_model/ignite/eval_dynamics.py` for the function it uses to generate frames (`rollout`/`generate` on `MaskGITDynamics`) and `scripts/../ignite_release/ignite_infer.py:131-170` for the bundle's version; use the repo one. Same `torch.manual_seed(seed)` before each arm.

- [ ] **Step 1: Failing test (tiny synthetic model, CPU)**

```python
import torch
from tokamak_foundation_model.ignite import dynamics_config as dc, maskgit
from shot_design.simulate import core

MODS = (dc.ModalitySpec("ece", "spectro", 4, 16), dc.ModalitySpec("mse", "slowts", 2, 16))

def _model():
    cfg = dc.DynamicsConfig(modalities=MODS, d_model=32, depth=1, n_heads=2, actuator_dim=88,
                            grad_checkpointing=False, k0_seed=2, n_predict=3)
    return maskgit.MaskGITDynamics(cfg).eval(), cfg

def _codes(F):
    return {"ece": torch.randint(0, 16, (F, 4), dtype=torch.int32), "mse": torch.randint(0, 16, (F, 2), dtype=torch.int32)}

def test_actuator_arms_share_the_seed_frames():
    ref = {"actuators": torch.randn(10, 88).half()}
    des = {"actuators": torch.randn(5, 88).half()}
    real, prop = core.actuator_arms(ref, des, k0=2, n_predict=3)
    assert real.shape == prop.shape == (5, 88)
    assert torch.equal(real[:2], prop[:2])
    assert not torch.equal(real[2:], prop[2:])

def test_run_paired_is_deterministic_and_reports_token_metrics():
    model, cfg = _model()
    codes = _codes(5)
    real = torch.zeros(5, 88); prop = torch.ones(5, 88)
    a = core.run_paired(model, cfg, codes, real, prop, seed=1, decode_steps=2)
    b = core.run_paired(model, cfg, codes, real, prop, seed=1, decode_steps=2)
    assert torch.equal(a.real["ece"], b.real["ece"])
    assert a.real["ece"].shape == (5, 4) and a.proposed["mse"].shape == (5, 2)
    assert set(a.divergence_vs_real) == {"ece", "mse"} and 0 <= a.token_accuracy["ece"] <= 1
    assert torch.equal(a.real["ece"][:2], codes["ece"][:2])   # seed frames are copied through
```
Check `DynamicsConfig` field names (`k0_seed`, `n_predict`) in `dynamics_config.py:70-140` and adapt.

- [ ] **Step 2: Run to fail.**

- [ ] **Step 3: Implement `core.py`** (≤ 200 lines). Persistence baseline: `gt[k0-1]` repeated over the predicted frames, compared token-wise to `gt[k0:]`.

- [ ] **Step 4: Run to pass, commit**

```bash
git add src/shot_design/simulate tests/shot_design/test_simulate_core.py
git commit -m "shot_design.simulate: seed assembly and paired real/proposed IGNITE rollout with token metrics"
```

### Task D2: `shot_design.simulate.decode` + `report` — physical units, panels, report

**Files:**
- Create: `src/shot_design/simulate/decode.py`, `src/shot_design/simulate/report.py`
- Test: `tests/shot_design/test_simulate_report.py`

**Interfaces:**
- `decode.decode_modalities(codecs: dict[str, tuple], arms: SimulationArms, names: list[str]) -> dict[str, dict[str, np.ndarray]]` → `{m: {"real": (F, ...), "proposed": ..., "gt": ...}}` using `train_dynamics` decode helpers (read `eval_dynamics.decode_all` and the bundle's `decode.py`; slowts/fastts decode to `(F, C)`, spectro to `(F, freq, time)` reduced to band power `(F, C)` by `np.mean(abs, axis=freq)` within 10–60 kHz for `mhr`/`mirnov`).
- `report.write(out_dir: Path, arms, decoded, meta: dict) -> Path` writes `simulation.h5` (groups `tokens/{real,proposed,gt}/<m>`, `decoded/<m>/{real,proposed,gt}`, `actuators/{real,proposed}`, attrs from `meta` incl. `dynamics_sha256`, `codec_generation`, `design_id`, `window_s`), `panels/<m>.png` (three-line plot per channel or channel mean: gt, real, proposed; vertical line at the seed/predict boundary), and `report.md` with a table `modality | frac_static | token_acc | persistence_acc | skill (= token_acc - persistence_acc) | divergence_vs_real` and the sentence "IGNITE v4 dynamics (step N) is an early checkpoint; treat results as qualitative." when any skill < 0.

- [ ] **Step 1: Failing test**

```python
import h5py, numpy as np, torch
from shot_design.simulate import core, report

def _arms():
    F = 5
    z = lambda: torch.randint(0, 16, (F, 2), dtype=torch.int32)
    return core.SimulationArms(seed_frames=2, predict_frames=3, real={"mse": z()}, proposed={"mse": z()}, gt={"mse": z()},
        divergence_vs_real={"mse": 0.5}, token_accuracy={"mse": 0.4}, persistence_accuracy={"mse": 0.6})

def test_report_writes_h5_panels_and_markdown(tmp_path):
    arms = _arms()
    decoded = {"mse": {k: np.random.rand(5, 3).astype(np.float32) for k in ("real", "proposed", "gt")}}
    out = report.write(tmp_path, arms, decoded, {"design_id": "abc", "dynamics_sha256": "0" * 64, "codec_generation": "v4", "window_s": [1.0, 5.0], "dynamics_step": 3200})
    with h5py.File(tmp_path / "simulation.h5") as f:
        assert f["decoded/mse/proposed"].shape == (5, 3)
        assert f["tokens/real/mse"].shape == (5, 2)
        assert f.attrs["codec_generation"] == "v4"
    assert (tmp_path / "panels" / "mse.png").exists()
    md = (tmp_path / "report.md").read_text()
    assert "| mse |" in md and "-0.20" in md and "qualitative" in md
```

- [ ] **Step 2: Run to fail.** **Step 3: Implement** (matplotlib `Agg`). **Step 4: Pass, commit**

```bash
git add src/shot_design/simulate tests/shot_design/test_simulate_report.py
git commit -m "shot_design.simulate: decode to physical units, panels and honest report"
```

### Task D3: `shot_design simulate` CLI

**Files:**
- Create: `src/shot_design/simulate/cli.py`
- Modify: `src/shot_design/cli.py` (register `simulate` subcommand → `simulate.cli.run(args)`)
- Test: `tests/shot_design/test_simulate_cli.py`

**Interfaces:**
- `python -m shot_design simulate <design-ident> [--seed 0] [--k0 20] [--n-predict 80] [--decode filterscopes,mhr,mirnov,ts_core_density,ts_core_temp] [--device cuda|cpu] [--out DIR]`
- Default out: `paths.data_root / "outputs" / <ident> / "simulation"`; writes `status.json` `{"state": "running|complete|failed", "started", "finished", "error", "report": "report.md"}` first thing and last thing.
- Steps: `program.load_program(ident)` → `program.export_ignite(program, paths)` (design seed) → `program_reference.reference(program.reference_shot, paths).cache` (real) → `core.load_dynamics` → `core.actuator_arms` → `core.run_paired` → `decode` → `report.write`.

- [ ] **Step 1: Failing test** — monkeypatch `core.load_dynamics`, `core.run_paired`, `decode.decode_modalities`, `program.load_program`, `program.export_ignite`, `program_reference.reference` with small fakes; assert `status.json` goes `running → complete`, `report.md` exists, and a raising fake leaves `status.json` `failed` with the error text and exit code 1.

- [ ] **Step 2–4:** fail → implement → pass.

- [ ] **Step 5: Commit**

```bash
git add src/shot_design/simulate/cli.py src/shot_design/cli.py tests/shot_design/test_simulate_cli.py
git commit -m "shot_design: simulate subcommand with status file"
```

### Task D4: Frontier sbatch + UI route for simulation

**Files:**
- Modify: `scripts/slurm_frontier/shot_design_simulate.sh` (replace the stub)
- Create: `src/shot_design/ui/simulate_routes.py`
- Modify: `src/shot_design/ui/app.py` (include router), `src/shot_design/ui/static/` (a "Simulate" button on the design page that POSTs and polls; read the existing design page JS first), `configs/shot_design/ui.yaml` (`simulate: {submit_cmd: "sbatch scripts/slurm_frontier/shot_design_simulate.sh {ident}", poll_s: 15}`)
- Test: `tests/shot_design/test_ui_simulate.py`

**Interfaces:**
- `POST /api/design/{ident}/simulate` → 202 `{"ident", "job_id", "status_url"}`; runs `ui.yaml simulate.submit_cmd` via `subprocess.run` (mockable through `app.state.submit`), parses `Submitted batch job N`.
- Controller amendment 2026-09-19 22:05: the wrapper locates `_shot_design_common.sh` through `$SLURM_SUBMIT_DIR`, so `sbatch` MUST be invoked with the repo root as its working directory. The default `app.state.submit` runs `subprocess.run(shlex.split(cmd), cwd=REPO_ROOT, capture_output=True, text=True, check=True)` with `REPO_ROOT = Path(__file__).resolve().parents[3]` (verify that index resolves to the repo root from `src/shot_design/ui/simulate_routes.py`), and the test asserts the fake submit receives the command string (the cwd is the production default's concern; cover it with one test that inspects the default submit's `cwd` via `monkeypatch.setattr(subprocess, "run", fake)`). The existing house-style test `tests/shot_design/test_slurm_frontier_scripts.py::test_wrappers_source_the_common_file_from_the_submit_dir` already pins the source line; keep it green.
- `GET /api/design/{ident}/simulate` → contents of `outputs/<ident>/simulation/status.json` or `{"state": "not_started"}`.
- `GET /api/design/{ident}/simulate/report` → `report.md`; `GET .../simulate/panels/{name}.png` → the image (name validated `^[a-z_]+\.png$`).

- [ ] **Step 1: Failing test** with FastAPI `TestClient`, `app.state.submit = lambda cmd: "Submitted batch job 4242"`, then write a `status.json` under the tmp data root and assert GET returns it, and a panel file is served with `image/png`.

- [ ] **Step 2–4:** fail → implement → pass.

- [ ] **Step 5: Slurm script**

```bash
#!/bin/bash
#SBATCH -A fus187
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -J sd-simulate
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH -c 7
#SBATCH -t 01:00:00
#SBATCH --output=/lustre/orion/fus187/proj-shared/nchen/shot_design/runs/slurm/%j.out
# sbatch runs a spool COPY of this file, so `dirname "$0"` is not the repo; the submit
# directory is (every wrapper is submitted from the repo root). Local runs fall back.
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/slurm_frontier/_shot_design_common.sh"
IDENT="${1:?design ident}"
srun "$PY" -m shot_design simulate "$IDENT" --device cuda "${@:2}"
```

- [ ] **Step 6: Commit**

```bash
git add scripts/slurm_frontier/shot_design_simulate.sh src/shot_design/ui configs/shot_design/ui.yaml tests/shot_design/test_ui_simulate.py
git commit -m "shot_design ui: simulate via sbatch submit + status polling; Frontier simulate wrapper"
```

---

## Phase E: LLM providers

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

### Task E3: Parallel blurb backfill with Gemini Flash

**Files:**
- Modify: `src/shot_design/shotdb/build.py` (`write_blurbs(..., workers: int = 1)`), `src/shot_design/cli.py` (`blurb --workers N`, default 1)
- Create: `scripts/shot_design/blurb_frontier.sh`
- Note: no Frontier serve script exists (nothing is served on Frontier); `scripts/shot_design/serve_llm.sbatch` is Stellar-only and stays.
- Test: `tests/shot_design/test_blurb_backfill.py`

**Interfaces:**
- `write_blurbs(paths, client, only_missing=True, *, limit=None, dry_run=False, shots=None, workers=1) -> int`. With `workers > 1` the `_blurb.make(rec, client, ...)` calls run in a `concurrent.futures.ThreadPoolExecutor(max_workers=workers)`; results are written back into `df` in shot order after all futures resolve; the parquet/manifest rewrite is unchanged and single-threaded. The `agy` provider is a subprocess per call so threads are safe; the request cache must be safe for concurrent writers (write `<sha>.json.part` then `os.replace`).
- `scripts/shot_design/blurb_frontier.sh`: `source _shot_design_common.sh`; `"$PY" -m shot_design blurb --workers "${WORKERS:-8}" "$@"`; runs on the login node (no sbatch — `agy` needs the network and the OAuth cache in `~/.gemini`).

- [ ] **Step 1: Failing test**

```python
def test_write_blurbs_workers_writes_every_row(tmp_db, fake_client):
    from shot_design.shotdb.build import write_blurbs
    n = write_blurbs(tmp_db.paths, fake_client, workers=4)
    df = pd.read_parquet(tmp_db.paths.db_dir / "shots.parquet")
    assert n == len(df) and (df["blurb_source"] == "llm").all()
    assert fake_client.max_concurrent >= 2
```
(`fake_client` returns a gate-passing three-sentence blurb after `time.sleep(0.05)` and records the peak number of in-flight calls; reuse `tests/shot_design/conftest.py` fixtures for `tmp_db`.)

- [ ] **Step 2–4:** fail → implement → pass (whole `tests/shot_design`).
- [ ] **Step 5: Dry run on 5 shots** (after F2's build): `bash scripts/shot_design/blurb_frontier.sh --dry-run --limit 5` — read the five candidates and gate verdicts; every acronym must be verbatim from the source text.
- [ ] **Step 6: Commit** `git commit -m "shot_design blurb: --workers thread pool; Frontier backfill script"`

The full backfill itself is F2 Step 4.

## Phase F: Database at ~5000 shots (text from the Frontier `shotsummary` bundles; no `sql/`)

### Task F1: Census and selection

- [ ] **Step 1:** `sbatch scripts/slurm_frontier/shot_design_census.sh`; then `pixi run --frozen -e shot-design-frontier python -m shot_design corpus summary $SHOT_DESIGN_DATA_ROOT/db/corpus_coverage.parquet | head -40`. Record eligible count.
- [ ] **Step 2:** Read `src/shot_design/shotdb/select.py` and `configs/shot_design/shot_lists/recommender_v1.yaml`. Run `python -m shot_design corpus select --n 5000 --name recommender_frontier_v1 --seed 20260919` with theme quotas ×10 (`fast_ion_ae` first). If eligible < 5000, take all eligible and say so.
- [ ] **Step 3:** Commit the list: `git add configs/shot_design/shot_lists/recommender_frontier_v1.yaml && git commit -m "shot_design: recommender_frontier_v1 shot list (N shots, full Frontier corpus)"`.

### Task F2: Labels, build, blurbs, encode

- [ ] **Step 1:** `python -m shot_design logs missing --list recommender_frontier_v1 | tail -3` — record how many selected shots have no `per_shot_txt` bundle (informational; nothing is imported, `sql/` is absent by decision, and the select step already prefers shots with text).
- [ ] **Step 2:** `python -m shot_design labels join --list recommender_frontier_v1` (from `data/events` + whatever is under `LABELER_ROOT`).
- [ ] **Step 3:** `SHOT_LIST=recommender_frontier_v1 sbatch scripts/slurm_frontier/shot_design_build.sh`; then `python -m labeler.jobstats <jobid>`.
- [ ] **Step 4:** `bash scripts/shot_design/blurb_frontier.sh` (login node, WORKERS=8, Gemini Flash low via agy; ~1 h for 5000). Then read `db/manifest.json`'s `blurbs` counts: expect > 90% `llm`; list the gate failures' reasons from a `--dry-run --shots <20 failing>` pass.
- [ ] **Step 5:** `SHOT_LIST=recommender_frontier_v1 sbatch scripts/slurm_frontier/shot_design_encode.sh` (8 GCDs); afterwards `python -m shot_design coverage` and `python -m shot_design describe 190736` (compare to the Stellar record the user can paste).
- [ ] **Step 6:** Record job ids, durations, blurb counts and jobstats in `.claude/notes/frontier-db-build-2026-09.md`; commit.

---

## Phase G: Demo

### Task G1: `scripts/shot_design/demo_frontier.sh` and the three prompts

**Files:**
- Create: `configs/shot_design/evalsets/frontier_demo_prompts.yaml` (the three prompts verbatim from the user, slugs `tearing_eccd`, `elm_rmp`, `ae_nbi`), `scripts/shot_design/demo_frontier.sh`
- Modify: `src/shot_design/cli.py` — `assistant` subcommand if absent (`--prompt`, `--provider`, `--trace out.jsonl`), wrapping `design.assistant.run_design` with a `progress` callback that appends `{"stage", "status", "detail", "t"}` lines and an `LLMClient` wrapper that appends every request/reply (`{"kind": "llm", "messages", "reply"}`) to the same trace.

- [ ] **Step 1:** Test `tests/shot_design/test_assistant_cli.py`: with `run_design` monkeypatched, `python -m shot_design assistant --prompt "x" --trace t.jsonl` writes a trace with at least the 5 stage lines and prints the design id.
- [ ] **Step 2–4:** fail → implement → pass; commit `git commit -m "shot_design: assistant CLI with JSONL trace"`.
- [ ] **Step 5: Demo script**

```bash
#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../slurm_frontier/_shot_design_common.sh"
OUT="$ROOT/outputs/recommender_frontier_demo"; mkdir -p "$OUT"
for slug in tearing_eccd elm_rmp ae_nbi; do
  d="$OUT/$slug"; mkdir -p "$d"
  "$PY" -m shot_design evalsets prompt frontier_demo_prompts "$slug" > "$d/prompt.md"
  ident=$("$PY" -m shot_design assistant --prompt "$(cat "$d/prompt.md")" --provider agy --trace "$d/trace.jsonl" | tail -1)
  echo "$ident" > "$d/design_id"
  "$PY" -m shot_design design show "$ident" --references > "$d/references.md"
  "$PY" -m shot_design design show "$ident" --actuation-csv > "$d/actuation.csv"
  cp "$ROOT/outputs/$ident.h5" "$d/design.h5"
  jid=$(sbatch --parsable "$REPO/scripts/slurm_frontier/shot_design_simulate.sh" "$ident")
  echo "$jid" > "$d/simulate_job"
done
echo "simulations queued; run scripts/shot_design/demo_frontier_collect.sh when they finish"
```
Add `design show` flags if they do not exist (read `cli.py:1839-1912` and `design/program.py:634-660`). `demo_frontier_collect.sh` copies `outputs/<ident>/simulation/{report.md,panels}` into each `$d/` and writes `$OUT/index.md` (prompt, model, reference shots, explanation excerpt, skill table, panel images).

- [ ] **Step 6: Run it** (after F2). Read every `report.md`; the deliverable is `$OUT/index.md`. Copy `index.md` + panels into `docs/examples/` (Task H2).

---

## Phase H: Documentation site

### Task H1: Docusaurus scaffold in docs-only mode

**Files:**
- Create: `website/` (`package.json`, `docusaurus.config.ts`, `sidebars.ts`, `tsconfig.json`, `src/css/custom.css`, `static/img/`), `.github/workflows/docs.yml`
- Modify: `.gitignore` (`website/node_modules`, `website/build`, `website/.docusaurus`)

- [ ] **Step 1: Scaffold**

```bash
cd /lustre/orion/fus187/scratch/nchen/FusionAIHub
npx create-docusaurus@latest website classic --typescript --skip-install
cd website && npm install 2>&1 | tail -2
rm -rf blog docs src/pages/index.tsx src/components
```

- [ ] **Step 2: Configure** `docusaurus.config.ts`: `title: 'FusionAIHub (FAITH)'`, `url: 'https://plasmacontrol.github.io'`, `baseUrl: '/FusionAIHub/'`, `organizationName: 'PlasmaControl'`, `projectName: 'FusionAIHub'`, `onBrokenLinks: 'throw'`, `onBrokenMarkdownLinks: 'throw'`; preset-classic `docs: { path: '../docs', routeBasePath: '/', sidebarPath: './sidebars.ts', editUrl: 'https://github.com/PlasmaControl/FusionAIHub/tree/nathan_dev/' }`, `blog: false`; navbar items: Docs, GitHub. `sidebars.ts`: `docs: [{type: 'autogenerated', dirName: '.'}]`. Add `plugins: [require.resolve('@docusaurus/theme-mermaid')]` only if a page uses mermaid.

- [ ] **Step 3: Workflow** `.github/workflows/docs.yml`: the Docusaurus GitHub Actions template (checkout, setup-node 20 with `cache: npm` and `cache-dependency-path: website/package-lock.json`, `npm ci` and `npm run build` with `working-directory: website`, `upload-pages-artifact` path `website/build`, deploy job with `pages: write`/`id-token: write`), trigger `on: push: branches: [nathan_dev, main]`.

- [ ] **Step 4: Build** `cd website && npm run build 2>&1 | tail -5` → expect the build to fail on broken links until H2 fixes them; that failure list is H2's input.

- [ ] **Step 5: Commit** `git add website .github/workflows/docs.yml .gitignore && git commit -m "docs: Docusaurus site in docs-only mode with GitHub Pages workflow"`

### Task H2: Reorganise `docs/` into the sidebar structure and write the new pages

**Files:**
- Move (git mv) into: `docs/intro.md`, `docs/getting-started/`, `docs/clusters/`, `docs/data/`, `docs/models/`, `docs/shot-design/`, `docs/labeler/`, `docs/examples/`, `docs/reference/`; each directory gets `_category_.json` `{"label": "...", "position": N}`.
- Map: `CLUSTERS.md` → `clusters/stellar.md` + `clusters/frontier.md` (split §1/§2, update §2 with the actual roots, env, scripts, LLM and model pins from this work; the checklist becomes "done" items) + `clusters/adding-a-cluster.md` (the path map + the three env variables + the paths file pattern); `E2E_ARCHITECTURE.md` → `models/foundation-model.md`; `IGNITE_DESIGN.md` → `models/ignite.md` (append a "v4 generation" section: 15 modalities, 1209 tokens, t0 1.0, pin/check commands, G-ENC result, dynamics caveat); `IGNITE_CODEC_RETRAIN_SPEC.md`, `IGNITE_ROLLOUT_QUALITY_PLAN.md`, `spectrogram_tokenizer_plan.md`, `video_tokenizer_plan.md`, `stage2_*.md` → `models/`; `SHOT_DESIGN.md` → `shot-design/overview.md` (+ split the CLI / MCP / LLM sections into `shot-design/cli.md`, `shot-design/mcp.md`, `shot-design/llm-providers.md`), `SHOT_DESIGN_PROGRAMS.md` → `shot-design/programs.md`; new `shot-design/simulation.md`, `shot-design/database-build.md` (census → select → logs → labels → build → encode with the Frontier commands); `LABELER.md` → `labeler/overview.md`; `ResearchPlan.MD` → `reference/research-plan.md`; `README.md` body → `intro.md` + `getting-started/install.md` (pixi envs per platform, table of env → platform → use); `reference/environment-variables.md`, `reference/slurm-scripts.md` (tables generated by reading `scripts/slurm_frontier/*.sh` headers), `data/corpus.md` (from CLAUDE.md's modality table and `data_loader.py` docstring; CLAUDE.md itself stays untouched).
- `README.md` at repo root shrinks to: one paragraph, install commands, link to the site, link to `docs/clusters/frontier.md`.
- `tests/shot_design/test_phenomena.py` asserts a sentence in `docs/SHOT_DESIGN.md`: update the path in the test to `docs/shot-design/overview.md`.

- [ ] **Step 1:** Do the moves with `git mv`, add front matter (`title`, `sidebar_position`) to every page, fix relative links (`git grep -n "](docs/\|](\.\./\|\.md)" docs | ...`).
- [ ] **Step 2:** Write the new pages listed above (each 60–200 lines, commands copied from the scripts in this plan, no Claude references).
- [ ] **Step 3:** `cd website && npm run build 2>&1 | tail -20` → zero broken links; `npm run serve -- --port 3111 &` and `curl -s localhost:3111/ | grep -c FusionAIHub` ≥ 1; kill it.
- [ ] **Step 4:** `pixi run --frozen -e shot-design-frontier pytest tests/shot_design -q 2>&1 | tail -2` (the docs-path assertion).
- [ ] **Step 5:** `git add -A docs README.md tests/shot_design/test_phenomena.py && git commit -m "docs: reorganise into Docusaurus sections; Frontier, LLM providers, simulation and database pages"`

### Task H3: Examples page from the demo and final push

- [ ] **Step 1:** After G1 completes: copy `outputs/recommender_frontier_demo/index.md` to `docs/examples/recommender-demo.md` with panels under `docs/examples/img/`; front matter; `npm run build` clean.
- [ ] **Step 2:** `git push`; confirm the Pages workflow is green on GitHub (`gh run list --workflow docs.yml -L 1`); if Pages is not enabled for the repo, say so in the final report (enabling it is a repo-settings action for the owner).

---

## Self-review (done while writing)

- Spec §1 → A1–A3; §2 → B1–B4; §3 → C1–C3; §4 → D1–D4; §5 → E1–E3; §6 → F1–F2; §7 → G1; §8 → H1–H3; tests section → each task's Step 1; "no Claude material in docs" → A3 + H2.
- Names used across tasks: `pin_bundle/check_bundle/bundle_dir/model_cfg` (C1, C2, D3), `SimulationArms/actuator_arms/run_paired/load_dynamics` (D1, D2, D3), `report.write` (D2, D3), `AgyProvider/render_prompt/AgyError` (E2), `_shot_design_common.sh` variables `REPO/ROOT/PY` (B3, C3, D4, E3, G1), status file `outputs/<ident>/simulation/status.json` (D3, D4).
- Open risks stated in-task: pixi activation precedence (B2), CodecPairDataset vs Peter's v4 STFT for mirnov (C3), agy rate limits / OAuth expiry during the 5000-blurb backfill (E3/F2), eligible pool < 5000 (F1), Pages enablement (H3).
