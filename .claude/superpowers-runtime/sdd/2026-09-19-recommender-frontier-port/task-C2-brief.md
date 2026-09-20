### Task C2: `mirnov` through the encode path and cache-first frame codes

**Files:**
- Modify: `src/shot_design/shotdb/ignite.py` (`frame_codes`: cache lookup; `_frames`: accept `mirnov`)
- Modify: `src/shot_design/design/program_reference.py` (`_cache_path` searches `model_cfg()["frame_codes_cache"]` first)
- Modify: `src/shot_design/design/seed.py` (docstring 14 → 15; `wanted_modalities` uses `model_cfg()["families"]`)
- Modify: `src/tokamak_foundation_model/ignite/train_dynamics.py` (`FROZEN_CODEC_CKPTS` gets a `"mirnov": ("spectro", "eval_runs/ignite_d5_mirnov/codec_best.pt")` entry so `resolve_codec_path` and `_load_codec` accept the name; `actuator_frames` unchanged)
- Test: `tests/shot_design/test_ignite_v4.py` (extend), `tests/shot_design/test_seed.py` (update counts)

**Interfaces:**
- Produces: `program_reference._cache_path(shot, paths)` order = `[model_cfg()["frame_codes_cache"], paths.data_root/"frame_codes", bundle_dir(paths)/"frame_codes"]`; `validate_cache` also checks `set(cache["vocabs"]) == set(model_cfg()["production_vocabs"])` and values equal.

- [ ] **Step 1: Failing tests**

```python
def test_cache_path_prefers_the_production_v4_cache(paths, tmp_path, monkeypatch):
    from shot_design.design import program_reference as pr
    prod = tmp_path / "prod"; prod.mkdir(); (prod / "190000.pt").write_bytes(b"x")
    monkeypatch.setitem(ignite.model_cfg(), "frame_codes_cache", str(prod))
    assert pr._cache_path(190000, paths) == prod / "190000.pt"

def test_validate_cache_rejects_v2_vocabs():
    from shot_design.design import program_reference as pr
    cache = {"codes": {}, "actuators": torch.zeros(1, 88), "n_frames": 1,
             "vocabs": {m: 1000 for m in ignite.model_cfg()["production_vocabs"]}}
    cache["vocabs"]["ece"] = 32768
    with pytest.raises(ValueError, match="ece"):
        pr.validate_cache(cache)

def test_wanted_modalities_includes_mirnov():
    from shot_design.design import seed
    assert "mirnov" in seed.wanted_modalities(ignite.model_cfg()["families"])   # adapt to the real signature after reading seed.py:100
```
`model_cfg()` returns the cached yaml dict; `monkeypatch.setitem` on it is enough because `load_yaml` caches by mtime.

- [ ] **Step 2: Run to fail.**

- [ ] **Step 3: Implement.** Read `program_reference.py:46-70`, `seed.py:100-135`, `shotdb/ignite.py:380-415`. `frame_codes(shot, ...)`: if `Path(cfg["frame_codes_cache"]) / f"{shot}.pt"` exists, `torch.load` it, run the vocab check, return it; else encode as today. Update `tests/shot_design/test_seed.py` expectations from 14 to 15 modalities where they count.

- [ ] **Step 4: Live check on 3 cached shots (login node, CPU is fine for this)**

```bash
pixi run --frozen -e shot-design-frontier python - <<'EOF'
import torch
from shot_design.config import load_paths
from shot_design.design import program_reference as pr
p = load_paths()
for s in (190000, 190090, 199597):
    c = torch.load(pr._cache_path(s, p), map_location="cpu", weights_only=False)
    pr.validate_cache(c); print(s, c["n_frames"], tuple(c["actuators"].shape), len(c["codes"]))
EOF
```
Expected: three lines, 15 codes each, `(F, 88)`.

- [ ] **Step 5: Run the suite, commit**

```bash
pixi run --frozen -e shot-design-frontier pytest tests/shot_design -q 2>&1 | tail -3
git add -A src/shot_design src/tokamak_foundation_model/ignite/train_dynamics.py tests/shot_design
git commit -m "ignite v4: mirnov in the encode path, production cache first, vocab validation against the pinned generation"
```

