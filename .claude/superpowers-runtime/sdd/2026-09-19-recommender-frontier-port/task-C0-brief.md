### Task C0: Merge Peter's v4 training code (`peter/dev-peter` @ 35cfed1)

Added 2026-09-19 18:55 by controller ruling. C1 found that the v4 codecs and the `mskfull` dynamics
checkpoint were produced by code that is not on `nathan_dev`: Peter's clone
`/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub` is on `dev-peter` at `35cfed1`, 47
unpushed commits past the `f71acd4` merged in Task A2. Measured on the login node with the same
interpreter: with Peter's `src` on `PYTHONPATH`, `train_dynamics._load_codec` loads all 15 v4
codecs strictly and `eval_dynamics.load_model` loads `mskfull/dynamics_best.pt` (step 3200,
300,815,000 params, 15 modalities, `actuator_dim 70 -> 88` inferred from
`backbone.act_embed.weight`); with `nathan_dev`'s `src`, 6 codecs fail (`SpectroCodec` lacks
`decoder.refine.*`, `FastTSCodec` lacks `encoder.gain_to_tokens.*`) and the dynamics load fails on
`act_embed` 70 vs 88. His 5 uncommitted files are NOT needed for loading (verified against his
committed HEAD) and are not taken. The ref is already fetched as `refs/remotes/peter/dev-peter`
(if absent: `git -c safe.directory='*' fetch /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub dev-peter:refs/remotes/peter/dev-peter`).

**Files (the seven that conflict):**
- Modify: `scripts/slurm_frontier/_frontier_common.sh`, `scripts/slurm_frontier/train_dynamics.sh`
- Modify: `src/tokamak_foundation_model/ignite/dynamics.py`, `dynamics_config.py`, `eval_dynamics.py`, `maskgit.py`, `train_dynamics.py`
- Create: `tests/ignite/test_v4_assets_load.py`
- Everything else merges cleanly (Peter deleted no files since the base; our `sampling.py`, `scoring.py`, `selfforce.py`, `text_embed.py` stay).

**Interfaces:**
- Produces: after the merge, `train_dynamics._load_codec(family, path)` loads every `ignite_codecs_v4/<m>/codec_best.pt` strictly; `eval_dynamics.load_model(ckpt_path, device)` loads `mskfull/dynamics_best.pt` and returns `actuator_dim == 88` on its config; `dynamics_config.modalities_from_manifest` (C1) is unchanged.

**Conflict rules (the spec is v4 end to end, so Peter's semantics win wherever they decide how a v4 checkpoint is built or loaded):**
1. Model construction and checkpoint loading (`maskgit.py`, `dynamics.py`, `dynamics_config.py` fields, `eval_dynamics.load_model`, codec loading in `train_dynamics.py`): take Peter's side.
2. Our additive, flag-gated features stay and must still be off by default and bit-identical when off: per-shot text-embedding conditioning (`--text_*`, `text_embed.py`), `--snapshot_every`, the Gumbel reveal (`IGNITE_MASKGIT_GUMBEL`), the actuator-counterfactual and honest-video eval panels, the removed diff-panel remnants (do not resurrect `CMAP_DIFF`). Where both sides changed the same function, the result carries both behaviours.
3. `dynamics_config.py`: keep C1's `modalities_from_manifest` verbatim and Peter's new fields.
4. Slurm wrappers: keep our `_frontier_common.sh` / `train_dynamics.sh` structure (they source `_frontier_settings.sh` and carry the RCCL plugin fix); bring over Peter's new environment variables and flags only where they are not already present. `bash -n` both.

- [ ] **Step 1: Baseline failure list before merging**

```bash
pixi run --frozen -e shot-design-frontier pytest tests/ignite tests/shot_design -q -p no:cacheprovider --ignore=tests/ignite/test_train_codec.py -rf 2>&1 | grep -E "^FAILED|passed|failed" > /tmp/c0_before.txt
```
(`test_train_codec.py` trainer-loop tests abort natively on a login node — Peter's note 34dc6c3. Never run `tests/labeler/test_labels_layout_integration.py` here.)

- [ ] **Step 2: Failing test**

```python
# tests/ignite/test_v4_assets_load.py
from pathlib import Path
import pytest

CODECS = Path("/lustre/orion/fus187/proj-shared/models/ignite_codecs_v4")
DYN = Path("/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/runs/mskfull/dynamics_best.pt")
FAMILIES = {"ece": "spectro", "bes": "spectro", "mhr": "spectro", "co2": "spectro",
            "mirnov": "spectro", "tangtv_lower": "video", "tangtv_upper": "video",
            "ts_core_density": "slowts", "ts_core_temp": "slowts",
            "ts_tangential_density": "slowts", "ts_tangential_temp": "slowts",
            "cer_ti": "slowts", "cer_rot": "slowts", "mse": "slowts", "filterscopes": "fastts"}

pytestmark = pytest.mark.skipif(not CODECS.exists(), reason="v4 assets are Frontier-only")


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_v4_codec_loads_strictly(name):
    from tokamak_foundation_model.ignite import train_dynamics as td
    codec = td._load_codec(FAMILIES[name], CODECS / name / "codec_best.pt")
    assert codec is not None


def test_mskfull_dynamics_checkpoint_loads_with_88_actuators():
    from tokamak_foundation_model.ignite import eval_dynamics as ed
    model, cfg, step = ed.load_model(DYN, "cpu")
    assert step == 3200
    assert len(cfg.modalities) == 15
    assert model.backbone.act_embed.weight.shape[1] == 88
```

Run: `pixi run --frozen -e shot-design-frontier pytest tests/ignite/test_v4_assets_load.py -q` — expected: 6 codec cases and the dynamics case FAIL on `nathan_dev` before the merge.

- [ ] **Step 3: Merge and resolve**

```bash
git merge --no-ff --no-commit peter/dev-peter
git diff --name-only --diff-filter=U     # the seven files
# resolve per the conflict rules; then
bash -n scripts/slurm_frontier/_frontier_common.sh scripts/slurm_frontier/train_dynamics.sh
git add -A -- scripts/slurm_frontier src/tokamak_foundation_model tests
git commit -m "merge peter/dev-peter 35cfed1: v4 codec stack, live-frame weighting, actuator_dim inference"
```

- [ ] **Step 4: Verify**

```bash
pixi run --frozen -e shot-design-frontier pytest tests/ignite/test_v4_assets_load.py -q          # 16 passed
pixi run --frozen -e shot-design-frontier pytest tests/ignite tests/shot_design -q -p no:cacheprovider --ignore=tests/ignite/test_train_codec.py -rf 2>&1 | grep -E "^FAILED|passed|failed" > /tmp/c0_after.txt
diff /tmp/c0_before.txt /tmp/c0_after.txt
```
Expected: every FAILED line in `before` that is not in `after` is a fix; any FAILED line in `after` not in `before` is a regression to fix before reporting (except `tests/shot_design/test_ignite.py::test_codecs_expose_the_encode_then_quantize_contract` if it now fails only on the encodable-name set — that is C2's `mirnov` work; report it).

- [ ] **Step 5: Commit the test**

```bash
git add tests/ignite/test_v4_assets_load.py
git commit -m "ignite: test that the pinned v4 codecs and mskfull dynamics checkpoint load"
```

