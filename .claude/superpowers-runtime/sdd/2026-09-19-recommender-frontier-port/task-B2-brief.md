### Task B2: `ideate-frontier` pixi environment

**Files:**
- Modify: `pyproject.toml` (`[tool.pixi.feature.ideate-frontier...]`, `[tool.pixi.environments]`)
- Modify: `pixi.lock` (regenerated)

**Interfaces:**
- Produces: `pixi run -e ideate-frontier python -m shot_design ...` with `SHOT_DESIGN_PATHS`, `SHOT_DESIGN_DATA_ROOT`, `LABELER_ROOT`, `SHOT_DESIGN_CORPUS`, `HF_HUB_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false` set by activation.

- [ ] **Step 1: Add the feature**

Read `pyproject.toml` lines 211–372 first. Add after the `frontier` feature:

```toml
[tool.pixi.feature.ideate-frontier]
platforms = ["linux-64"]

[tool.pixi.feature.ideate-frontier.pypi-dependencies]
torch       = { version = ">=2.10,<2.11", index = "https://download.pytorch.org/whl/rocm7.1" }
torchvision = { version = ">=0.25,<0.27", index = "https://download.pytorch.org/whl/rocm7.1" }
triton-rocm = { version = "*",            index = "https://download.pytorch.org/whl/rocm7.1" }
x-transformers = "*"
vector-quantize-pytorch = "*"
einops = "*"
loguru = "*"

[tool.pixi.feature.ideate-frontier.target.unix.activation.env]
SHOT_DESIGN_PATHS     = "/lustre/orion/fus187/scratch/nchen/FusionAIHub/configs/shot_design/paths.frontier.yaml"
SHOT_DESIGN_DATA_ROOT = "/lustre/orion/fus187/proj-shared/nchen/shot_design"
LABELER_ROOT          = "/lustre/orion/fus187/proj-shared/nchen/labeler"
SHOT_DESIGN_CORPUS    = "/lustre/orion/fus187/proj-shared/foundation_model"
HF_HUB_OFFLINE = "1"
TOKENIZERS_PARALLELISM = "false"
HDF5_USE_FILE_LOCKING = "FALSE"
```
and `ideate-frontier = ["ideate", "ideate-frontier"]` under `[tool.pixi.environments]`. The `ideate` feature's own activation block also sets the three roots (Stellar values); pixi merges activation env with later features winning — confirm with step 3, and if `ideate`'s values win, rename this feature so it sorts after or move the three roots into a wrapper `scripts/shot_design/frontier_env.sh` that the sbatch scripts source (document which was needed).

- [ ] **Step 2: Solve and install**

```bash
pixi install -e ideate-frontier 2>&1 | tail -5
```
If uv cannot resolve `ideate`'s `sentence-transformers` against ROCm torch, pin `sentence-transformers = ">=3,<6"` in the new feature. If pixi rejects the two torch sources, the fallback is a uv venv: `uv venv .venv-ideate-frontier --python 3.11 && uv pip install --index-url https://download.pytorch.org/whl/rocm7.1 torch torchvision && uv pip install -e . sentence-transformers ...` and a `scripts/shot_design/frontier_env.sh` exporting the same variables. Record the outcome in the commit body.

- [ ] **Step 3: Verify the activation and the suite**

```bash
pixi run -e ideate-frontier bash -c 'echo $SHOT_DESIGN_DATA_ROOT $SHOT_DESIGN_PATHS; python -c "import torch, shot_design, sentence_transformers; print(torch.__version__)"'
pixi run -e ideate-frontier pytest tests/shot_design -q -x 2>&1 | tail -3
```
Expected: the Frontier root and paths file printed, `2.10.0+rocm7.1`, suite green (or list failures verbatim; fix only ones caused by the env).

- [ ] **Step 4: Pre-populate the sentence-transformers cache (login node, network)**

```bash
HF_HUB_OFFLINE=0 pixi run -e ideate-frontier python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"
pixi run -e ideate-frontier python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2'); print('offline ok')"
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml pixi.lock
git commit -m "pixi: ideate-frontier env (ROCm torch + shot_design deps, Frontier activation roots)"
```

