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

