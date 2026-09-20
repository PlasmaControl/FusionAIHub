### Task D1: `shot_design.simulate.core` — seed assembly and paired rollout (torch only)

**Files:**
- Create: `src/shot_design/simulate/__init__.py`, `src/shot_design/simulate/core.py`
- Test: `tests/shot_design/test_simulate_core.py`

**Interfaces:**
- Consumes: the design seed `.pt` written by `program.export_ignite` = `{"codes": {m: (F, n_tok) int32}, "actuators": (F, 88) float16 | None, "n_frames": F, "vocabs": {m: int}}` (this is the reference cache sliced to the design window with the PROPOSED actuators already z-scored by `design.actuators.z_score` — verify by reading `program.py:340-500` before coding; if `output` there is the reference's real actuators with the proposal applied, then the real arm is `reference cache["actuators"]` over the same slice).
- Produces:

```python
@dataclass
class SimulationArms:
    seed_frames: int          # k0, default 20
    predict_frames: int       # default 80
    real: dict[str, torch.Tensor]       # {m: (F, n_tok)} predicted codes with the reference actuators
    proposed: dict[str, torch.Tensor]   # same shape, proposed actuators
    gt: dict[str, torch.Tensor]         # reference cache codes over the same frames
    divergence_vs_real: dict[str, float]   # fraction of tokens that differ, predicted region only
    token_accuracy: dict[str, float]       # vs gt, predicted region only
    persistence_accuracy: dict[str, float] # last seed frame repeated, vs gt

def load_dynamics(paths, device) -> tuple[torch.nn.Module, DynamicsConfig, int]   # eval_dynamics.load_model on bundle_dir/dynamics_file
def actuator_arms(reference_cache: dict, design_seed: dict, k0: int, n_predict: int) -> tuple[torch.Tensor, torch.Tensor]  # (real, proposed), each (k0+n_predict, 88) float32; the first k0 frames are identical (seed uses measured controls)
def run_paired(model, cfg, codes: dict, real_act, prop_act, *, seed: int, temperature: float = 1.0, decode_steps: int = 10) -> SimulationArms
```
`run_paired` calls the repo's own rollout: read `src/tokamak_foundation_model/ignite/eval_dynamics.py` for the function it uses to generate frames (`rollout`/`generate` on `MaskGITDynamics`) and `scripts/../ignite_release/ignite_infer.py:131-170` for the bundle's version; use the repo one. Same `torch.manual_seed(seed)` before each arm.

- [ ] **Step 1: Failing test (tiny synthetic model, CPU)**

```python
import torch
from tokamak_foundation_model.ignite import dynamics_config as dc, maskgit
from shot_design.simulate import core

MODS = (dc.ModalitySpec("ece", "spectro", 4, 16), dc.ModalitySpec("mse", "slowts", 2, 16))

def _model():
    cfg = dc.DynamicsConfig(modalities=MODS, d_model=32, depth=1, n_heads=2, actuator_dim=88,
                            grad_checkpointing=False, k0_seed=2, n_predict=3)
    return maskgit.MaskGITDynamics(cfg).eval(), cfg

def _codes(F):
    return {"ece": torch.randint(0, 16, (F, 4), dtype=torch.int32), "mse": torch.randint(0, 16, (F, 2), dtype=torch.int32)}

def test_actuator_arms_share_the_seed_frames():
    ref = {"actuators": torch.randn(10, 88).half()}
    des = {"actuators": torch.randn(5, 88).half()}
    real, prop = core.actuator_arms(ref, des, k0=2, n_predict=3)
    assert real.shape == prop.shape == (5, 88)
    assert torch.equal(real[:2], prop[:2])
    assert not torch.equal(real[2:], prop[2:])

def test_run_paired_is_deterministic_and_reports_token_metrics():
    model, cfg = _model()
    codes = _codes(5)
    real = torch.zeros(5, 88); prop = torch.ones(5, 88)
    a = core.run_paired(model, cfg, codes, real, prop, seed=1, decode_steps=2)
    b = core.run_paired(model, cfg, codes, real, prop, seed=1, decode_steps=2)
    assert torch.equal(a.real["ece"], b.real["ece"])
    assert a.real["ece"].shape == (5, 4) and a.proposed["mse"].shape == (5, 2)
    assert set(a.divergence_vs_real) == {"ece", "mse"} and 0 <= a.token_accuracy["ece"] <= 1
    assert torch.equal(a.real["ece"][:2], codes["ece"][:2])   # seed frames are copied through
```
Check `DynamicsConfig` field names (`k0_seed`, `n_predict`) in `dynamics_config.py:70-140` and adapt.

- [ ] **Step 2: Run to fail.**

- [ ] **Step 3: Implement `core.py`** (≤ 200 lines). Persistence baseline: `gt[k0-1]` repeated over the predicted frames, compared token-wise to `gt[k0:]`.

- [ ] **Step 4: Run to pass, commit**

```bash
git add src/shot_design/simulate tests/shot_design/test_simulate_core.py
git commit -m "shot_design.simulate: seed assembly and paired real/proposed IGNITE rollout with token metrics"
```

