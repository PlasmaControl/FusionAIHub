### Task F1: Census and selection

- [ ] **Step 1:** `sbatch scripts/slurm_frontier/shot_design_census.sh`; then `pixi run --frozen -e shot-design-frontier python -m shot_design corpus summary $SHOT_DESIGN_DATA_ROOT/db/corpus_coverage.parquet | head -40`. Record eligible count.
- [ ] **Step 2:** Read `src/shot_design/shotdb/select.py` and `configs/shot_design/shot_lists/recommender_v1.yaml`. Run `python -m shot_design corpus select --n 5000 --name recommender_frontier_v1 --seed 20260919` with theme quotas ×10 (`fast_ion_ae` first). If eligible < 5000, take all eligible and say so.
- [ ] **Step 3:** Commit the list: `git add configs/shot_design/shot_lists/recommender_frontier_v1.yaml && git commit -m "shot_design: recommender_frontier_v1 shot list (N shots, full Frontier corpus)"`.

