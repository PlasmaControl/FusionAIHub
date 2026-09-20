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

