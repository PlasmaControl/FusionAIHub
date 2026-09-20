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

