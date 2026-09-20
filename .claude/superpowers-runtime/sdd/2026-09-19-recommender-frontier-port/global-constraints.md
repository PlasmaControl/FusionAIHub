## Global Constraints

- Repo: `/lustre/orion/fus187/scratch/nchen/FusionAIHub`. Work on branch `nathan_dev` (created in Task A1). Never force-push; never touch `nathan_fm`, `dev-nathan`, `main`.
- Python for tests (env exists since Task B2): `pixi run --frozen -e shot-design-frontier pytest tests/shot_design tests/labeler`. Every pixi call on Frontier uses `--frozen`; never re-solve the lock (bare `pixi run`/`pixi install` re-solves all envs and dies on `default`/win-64).
- Frontier data roots (spec §2): `SHOT_DESIGN_DATA_ROOT=/lustre/orion/fus187/proj-shared/nchen/shot_design`, `LABELER_ROOT=/lustre/orion/fus187/proj-shared/nchen/labeler`, `SHOT_DESIGN_CORPUS=/lustre/orion/fus187/proj-shared/foundation_model`. Never write there from a test; tests use the `paths` fixture (tmp_path).
- v4 contract: 15 modalities (spec §"Facts"), every vocab 1000, `t0_start_s: 1.0`, 1209 tokens/frame, actuators `(F, 88)`. Codec template `/lustre/orion/fus187/proj-shared/models/ignite_codecs_v4/{m}/codec_best.pt`; cache `/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes`; dynamics `/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/runs/mskfull/dynamics_best.pt` (step 3200). Pinned copies with sha256 live under `<models_dir>/IGNITE_v4/`.
- Slurm: `-A fus187 -p batch`; short demo/validation jobs `-q debug` (≤ 2 h, one at a time). Logs to `$SHOT_DESIGN_DATA_ROOT/runs/slurm/%j.out`. Never run production writes (`build`, `add`, `labels join`) without the owner's go-ahead stated in the task.
- No Claude/superpowers/planning material under `docs/`; it lives under `.claude/`.
- Ruff `line-length = 88`. Every commit message is one line, imperative, `<area>: <what>`.
- Tests are TDD: failing test first, then the code, then green, then commit.
- Nothing is copied from Stellar. `foundation_model_text/sql/` is absent by decision (mpid None everywhere). The only LLM is Gemini Flash through `agy`; no Gemma/Ollama on Frontier.

