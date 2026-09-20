### Task B3: Frontier Slurm wrappers for shot_design

**Files:**
- Create: `scripts/slurm_frontier/shot_design_census.sh`, `shot_design_build.sh`, `shot_design_encode.sh`, `shot_design_simulate.sh` (filled in Task D4), `scripts/slurm_frontier/_shot_design_common.sh`
- Test: `tests/shot_design/test_slurm_frontier_scripts.py`

**Interfaces:**
- Produces: `_shot_design_common.sh` exporting `REPO`, `ROOT` (= `$SHOT_DESIGN_DATA_ROOT`), `PY` (= `$REPO/.pixi/envs/ideate-frontier/bin/python`), and sourcing `_frontier_settings.sh` with `RCCL_PLUGIN=0` (single-GPU jobs need no plugin).

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
PY="$REPO/.pixi/envs/ideate-frontier/bin/python"
mkdir -p "$ROOT/runs/slurm"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is)"
rocm-smi --showproductname 2>/dev/null | grep -m1 'Card series' || true
```
`_frontier_settings.sh` prepends the `frontier` env to PATH; `PY` points at the `ideate-frontier` interpreter explicitly so the two envs cannot be confused.

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
srun -n1 -c56 "$PY" -m shot_design corpus scan --workers 48 --out "$ROOT/db/census.parquet"
"$PY" -m shot_design corpus summary "$ROOT/db/census.parquet"
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

