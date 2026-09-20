### Task F2: Labels, build, blurbs, encode

- [ ] **Step 1:** `python -m shot_design logs missing --list recommender_frontier_v1 | tail -3` — record how many selected shots have no `per_shot_txt` bundle (informational; nothing is imported, `sql/` is absent by decision, and the select step already prefers shots with text).
- [ ] **Step 2:** `python -m shot_design labels join --list recommender_frontier_v1` (from `data/events` + whatever is under `LABELER_ROOT`).
- [ ] **Step 3:** `SHOT_LIST=recommender_frontier_v1 sbatch scripts/slurm_frontier/shot_design_build.sh`; then `python -m labeler.jobstats <jobid>`.
- [ ] **Step 4:** `bash scripts/shot_design/blurb_frontier.sh` (login node, WORKERS=8, Gemini Flash low via agy; ~1 h for 5000). Then read `db/manifest.json`'s `blurbs` counts: expect > 90% `llm`; list the gate failures' reasons from a `--dry-run --shots <20 failing>` pass.
- [ ] **Step 5:** `SHOT_LIST=recommender_frontier_v1 sbatch scripts/slurm_frontier/shot_design_encode.sh` (8 GCDs); afterwards `python -m shot_design coverage` and `python -m shot_design describe 190736` (compare to the Stellar record the user can paste).
- [ ] **Step 6:** Record job ids, durations, blurb counts and jobstats in `.claude/notes/frontier-db-build-2026-09.md`; commit.

---

