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

