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

