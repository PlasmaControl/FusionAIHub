### Task D2: `shot_design.simulate.decode` + `report` — physical units, panels, report

**Files:**
- Create: `src/shot_design/simulate/decode.py`, `src/shot_design/simulate/report.py`
- Test: `tests/shot_design/test_simulate_report.py`

**Interfaces:**
- `decode.decode_modalities(codecs: dict[str, tuple], arms: SimulationArms, names: list[str]) -> dict[str, dict[str, np.ndarray]]` → `{m: {"real": (F, ...), "proposed": ..., "gt": ...}}` using `train_dynamics` decode helpers (read `eval_dynamics.decode_all` and the bundle's `decode.py`; slowts/fastts decode to `(F, C)`, spectro to `(F, freq, time)` reduced to band power `(F, C)` by `np.mean(abs, axis=freq)` within 10–60 kHz for `mhr`/`mirnov`).
- `report.write(out_dir: Path, arms, decoded, meta: dict) -> Path` writes `simulation.h5` (groups `tokens/{real,proposed,gt}/<m>`, `decoded/<m>/{real,proposed,gt}`, `actuators/{real,proposed}`, attrs from `meta` incl. `dynamics_sha256`, `codec_generation`, `design_id`, `window_s`), `panels/<m>.png` (three-line plot per channel or channel mean: gt, real, proposed; vertical line at the seed/predict boundary), and `report.md` with a table `modality | frac_static | token_acc | persistence_acc | skill (= token_acc - persistence_acc) | divergence_vs_real` and the sentence "IGNITE v4 dynamics (step N) is an early checkpoint; treat results as qualitative." when any skill < 0.

- [ ] **Step 1: Failing test**

```python
import h5py, numpy as np, torch
from shot_design.simulate import core, report

def _arms():
    F = 5
    z = lambda: torch.randint(0, 16, (F, 2), dtype=torch.int32)
    return core.SimulationArms(seed_frames=2, predict_frames=3, real={"mse": z()}, proposed={"mse": z()}, gt={"mse": z()},
        divergence_vs_real={"mse": 0.5}, token_accuracy={"mse": 0.4}, persistence_accuracy={"mse": 0.6})

def test_report_writes_h5_panels_and_markdown(tmp_path):
    arms = _arms()
    decoded = {"mse": {k: np.random.rand(5, 3).astype(np.float32) for k in ("real", "proposed", "gt")}}
    out = report.write(tmp_path, arms, decoded, {"design_id": "abc", "dynamics_sha256": "0" * 64, "codec_generation": "v4", "window_s": [1.0, 5.0], "dynamics_step": 3200})
    with h5py.File(tmp_path / "simulation.h5") as f:
        assert f["decoded/mse/proposed"].shape == (5, 3)
        assert f["tokens/real/mse"].shape == (5, 2)
        assert f.attrs["codec_generation"] == "v4"
    assert (tmp_path / "panels" / "mse.png").exists()
    md = (tmp_path / "report.md").read_text()
    assert "| mse |" in md and "-0.20" in md and "qualitative" in md
```

- [ ] **Step 2: Run to fail.** **Step 3: Implement** (matplotlib `Agg`). **Step 4: Pass, commit**

```bash
git add src/shot_design/simulate tests/shot_design/test_simulate_report.py
git commit -m "shot_design.simulate: decode to physical units, panels and honest report"
```

