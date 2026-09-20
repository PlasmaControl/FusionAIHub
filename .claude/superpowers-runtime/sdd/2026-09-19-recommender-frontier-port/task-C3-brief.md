### Task C3: Loader reads `actuator_dim`; G-ENC gate against the v4 cache (GPU)

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/eval_dynamics.py:121-142` (`load_model`)
- Modify: `scripts/shot_design/g_enc.py` (compare fresh encodes with `frame_codes_cache`, 15 modalities)
- Modify: `src/shot_design/shotdb/build.py:296-309` (`frame_codes_dirs`: production cache first, v4 bundle dir)
- Modify: `src/shot_design/design/program_reference.py` (`_cache_path` reuses `build.frame_codes_dirs`)
- Create: `scripts/slurm_frontier/shot_design_genc.sh`
- Test: `tests/ignite/test_load_model_actuator_dim.py`
- Test: `tests/shot_design/test_build.py` (or the existing build test module) — `frame_codes_dirs` order

**Controller amendment 2026-09-19 20:10 (ruling, from the C2 implementer's follow-ups):**
`shotdb.build.frame_codes_dirs` still returns the v2 pair `(<data_root>/frame_codes, <models_dir>/IGNITE/frame_codes)`. It is the single source for `build`'s `has_frame_codes` column, `corpus select`'s preferred-shots set (`cli.py:608, 978`) and `frame_codes_path`. Left as is, Task F1 would select 5000 shots blind to the 8752 shots already in the production cache and Task F2 Step 5 would re-encode them on 8 GCDs. Fix it here, before any F-phase job runs.

- [ ] **Step 0a: Failing test** (append to the module that already tests `shotdb.build`; otherwise create `tests/shot_design/test_build_frame_codes_dirs.py`):

```python
from pathlib import Path
from shot_design.shotdb import build, ignite

def test_frame_codes_dirs_production_cache_first_then_v4_bundle(paths, monkeypatch):
    monkeypatch.setattr(ignite, "model_cfg", lambda: {**ignite.model_cfg(), "frame_codes_cache": "/prod/frame_codes"})
    dirs = build.frame_codes_dirs(paths)
    assert dirs[0] == Path("/prod/frame_codes")
    assert dirs[1] == Path(paths.data_root) / "frame_codes"
    assert dirs[2] == ignite.bundle_dir(paths) / "frame_codes"
    assert Path(paths.models_dir) / "IGNITE" / "frame_codes" not in dirs

def test_frame_codes_dirs_without_a_production_cache(paths, monkeypatch):
    monkeypatch.setattr(ignite, "model_cfg", lambda: {k: v for k, v in ignite.model_cfg().items() if k != "frame_codes_cache"})
    dirs = build.frame_codes_dirs(paths)
    assert dirs == (Path(paths.data_root) / "frame_codes", ignite.bundle_dir(paths) / "frame_codes")
```

(`monkeypatch.setitem(ignite.model_cfg(), …)` is a no-op because `load_yaml` deep-copies — patch the function, as Task C2's tests do.)

- [ ] **Step 0b: Implement.** `frame_codes_dirs` returns `(Path(production), data_root/"frame_codes", bundle_dir(paths)/"frame_codes")` with `production = ignite.model_cfg().get("frame_codes_cache")`, omitting the first entry when unset; drop `<models_dir>/IGNITE`. Then make `program_reference._cache_path` iterate `build.frame_codes_dirs(paths)` instead of its own `roots` list (same order; one source of truth). `build.py` deliberately never imports `ignite` at module level (its line-14 comment; see the lazy `from . import ignite` at `build.py:1074, 1293`) — keep that: import `ignite` inside `frame_codes_dirs`. `design/program_reference.py` may import `build` at module level only if `build` does not import `design`; check with `grep -n "^from\|^import" src/shot_design/shotdb/build.py`, and if it does, put the tuple in `shotdb/ignite.py` as `frame_codes_dirs(paths)` and have both callers use it. Run `pytest tests/shot_design -q`; the existing `_cache_path` tests must stay green.

- [ ] **Step 0b-ii: Hermeticity (C2 review, Important, plan-mandated).** `_cache_path`/`frame_codes_dirs` read `model_cfg()["frame_codes_cache"]`, an absolute Frontier path, so any test using a real shot number in the production range silently reads production data instead of its tmp tree. Add to `tests/shot_design/conftest.py` an autouse fixture that patches `shot_design.shotdb.ignite.model_cfg` to return the yaml WITHOUT `frame_codes_cache`; tests that want a production cache set it themselves afterwards (their own `monkeypatch.setattr(ignite, "model_cfg", ...)` runs later and wins — Task C2's `test_ignite_v4.py` cache tests do exactly this; confirm they still pass). Test for the fixture itself: `def test_tests_never_see_the_production_cache(): assert "frame_codes_cache" not in ignite.model_cfg()`. Also point `program_reference._cache_path` at `ignite.production_cache_path(shot)` / `frame_codes_dirs` rather than re-deriving the root (C2 review, Minor 1) — Step 0b already does this if `_cache_path` iterates `frame_codes_dirs`.

- [ ] **Step 0c:** In `g_enc.py`, `FULL_SHOT_FRAMES = 239` is the v2 count (t0 0.05 s). v4 starts at `t0_start_s: 1.0`, so a full shot is 219 frames (the C2 validation measured 219 on shots 190000, 190090, 204346). Set `FULL_SHOT_FRAMES = 219` and derive the expected count from the cache file's `n_frames` when it is present, with the constant as the fallback; update its comment.

- [ ] **Step 1: Failing test**

```python
import torch
from tokamak_foundation_model.ignite import eval_dynamics, dynamics_config as dc, maskgit

def test_load_model_builds_the_checkpoints_actuator_width(tmp_path):
    mods = (dc.ModalitySpec("ece", "spectro", 4, 16), dc.ModalitySpec("mse", "slowts", 2, 16))
    cfg = dc.DynamicsConfig(modalities=mods, d_model=32, depth=1, n_heads=2, actuator_dim=88, grad_checkpointing=False)
    m = maskgit.MaskGITDynamics(cfg)
    ck = {"model": m.state_dict(), "cfg_depth": 1, "cfg_d_model": 32, "cfg_n_heads": 2,
          "modalities": [tuple(x) for x in mods], "step": 7}
    p = tmp_path / "d.pt"; torch.save(ck, p)
    model, cfg2, step = eval_dynamics.load_model(p, "cpu")
    assert cfg2.actuator_dim == 88 and step == 7
```

- [ ] **Step 2: Run.** Controller amendment 2026-09-19 19:45: Task C0 merged Peter's `eval_dynamics.load_model`, which already infers `actuator_dim` from `backbone.act_embed.weight` (it printed `actuator_dim 70 -> 88` when loading `mskfull`). Expected: this test PASSES as written. Keep the test (it pins the behaviour), skip Step 3, and go to Step 4. Only if it fails, do Step 3.

- [ ] **Step 3 (only if Step 2 failed): Implement** in `load_model` before `DynamicsConfig(**kw)`:

```python
    if "cfg_actuator_dim" in ck:
        kw["actuator_dim"] = int(ck["cfg_actuator_dim"])
    else:
        w = ck["model"].get("backbone.act_embed.weight")
        if w is not None:
            kw["actuator_dim"] = int(w.shape[1])
```
Confirm the key name by `grep -n "act_embed" src/tokamak_foundation_model/ignite/maskgit.py`.

- [ ] **Step 4: G-ENC on Frontier (one GCD, debug queue)**

Rewrite `scripts/shot_design/g_enc.py` to take `--cache-dir` (default `model_cfg()["frame_codes_cache"]`) and `--shots` (default 5 shots present in the cache: 190000 190090 199597 190735 190736), encode each with the pinned v4 codecs at `t0_start=1.0`, and print per-modality exact-match fraction; exit 1 if any non-spectro/video modality is below 1.0 or any spectro/video below 0.99. `shot_design_genc.sh`: `-q debug -t 01:00:00 --gres=gpu:1`, body `srun "$PY" scripts/shot_design/g_enc.py --device cuda --out "$ROOT/runs/genc_v4.json"`.

```bash
sbatch scripts/slurm_frontier/shot_design_genc.sh && squeue -u $USER
```
Wait; read `$ROOT/runs/genc_v4.json`. If a spectro modality misses the bar, the STFT/standardisation used by `CodecPairDataset` differs from Peter's cache builder for v4: record the measured fractions in the commit body and in `docs/models/ignite.md` (Task H2) and continue, since rollouts seed from the cache anyway.

- [ ] **Step 5: Commit**

```bash
git add src/shot_design/shotdb/build.py src/shot_design/design/program_reference.py tests/shot_design/test_build_frame_codes_dirs.py
git commit -m "shot_design: frame_codes_dirs lists the production cache first and the v4 bundle"
git add src/tokamak_foundation_model/ignite/eval_dynamics.py scripts/shot_design/g_enc.py scripts/slurm_frontier/shot_design_genc.sh tests/ignite/test_load_model_actuator_dim.py
git commit -m "ignite: pin actuator_dim inference in load_model; G-ENC gate against the v4 production cache"
```

---

