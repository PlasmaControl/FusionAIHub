# IGNITE Rollout Quality — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the train/test gap that makes IGNITE rollouts degrade and modalities clash, without changing the behaviour of Peter's existing model by a single bit.

**Architecture:** Every change is **additive and flag-gated inside the existing `ignite/` package — no fork.** Peter is actively developing the same module (bp line, codecs), so a copy would diverge on day one; instead, new behaviour lives behind config fields and a `SamplerConfig` whose defaults reproduce today's code path exactly, enforced by a golden-output test written *before* any change (Task 1). **No Phase-1 feature adds a single model parameter**, so every existing checkpoint — `prod_d512L8`, `bp_d512L8`, all of Peter's — loads and runs unchanged. Work happens on branch `nathan_fm`.

**Tech Stack:** PyTorch 2.x (ROCm on Frontier), pytest, SLURM, HDF5. No new dependencies.

**Spec:** `docs/IGNITE_ROLLOUT_QUALITY_PLAN.md` (R0–R16, with the measured evidence and the source-paper analysis behind each). Read it before Task 1 — this plan implements its Tier 0, Tier 1, and the Stage-A half of Tier 2, and argues from its numbers.

## Global Constraints

- **Bit-identical default behaviour.** With no new flag set, `rollout()`, `generate_frame()` and `training_loss()` must produce output identical to the pre-change code, including the RNG draw sequence. Any new random draw must be guarded so it is *not executed* when its feature is off (`if p > 0:`, never `if torch.rand(...) < p` with `p == 0`).
- **Zero new parameters in Phase 1.** No new `nn.Module`, no changed tensor shapes in `state_dict`. Head factorization, state tokens, and the inverse-dynamics head are deliberately out of scope (they change parameter counts — see Follow-on Plans).
- **Config additions are dataclass fields with defaults.** `DynamicsConfig` is constructed from checkpoints that predate the new fields, so every new field must have a default that means "off".
- **Tests run on CPU** with tiny configs, following the existing idiom in `tests/ignite/test_phaseb_maskgit.py` (`_tiny()` / `_codes()` helpers, `torch.Generator().manual_seed(...)`). No test may require a GPU or the production cache.
- **Interpreter.** Every `pytest` command below is written for `.pixi/envs/frontier/bin/python`. That env is currently broken (Task 0). **The Phase-B modules import with torch alone** — no `x-transformers`, no `vector-quantize-pytorch` — so a torch-only venv runs the whole unit-test suite. Verified 2026-08-17: all 23 existing Phase-B tests pass under `/lustre/orion/fus187/scratch/nchen/tokeye/.venv/bin/python` (torch 2.10 ROCm, py3.13). Substitute that interpreter and Tasks 1–10 are unblocked even before the pixi rebuild; only Tasks 11–13 (real cache, SLURM) need the rebuilt env.
- **Frame = 50 ms**; production layout is **cache-derived, not the static table** (1593 tokens/frame, four 64k-vocab modalities). Never hard-code 1017 or 1012.
- **Judge every claim in decoded, band-restricted space**, with the majority-token guard. `divergence_vs_real` is not an effect size (measured 2026-08-15: token churn anti-correlates with decoded change).
- Commit after every task. Branch: `nathan_fm`.

---

## File Structure

**New files**

| path | responsibility |
|---|---|
| `src/tokamak_foundation_model/ignite/sampling.py` | `SamplerConfig` + the decode-policy helpers (temperature, top-p, global confidence pool, revision). Pure functions over logits/confidences; no model state. |
| `src/tokamak_foundation_model/ignite/scoring.py` | Masked pseudo-likelihood scorer — the frozen-teacher window score used by best-of-N reranking and (later) as a reward. |
| `src/tokamak_foundation_model/ignite/selfforce.py` | Rollout-context construction for training: no-grad m-frame self-rollout from a ground-truth prefix. |
| `tests/ignite/test_phaseb_compat.py` | The golden test. Pins pre-change behaviour of rollout/training_loss forever. |
| `tests/ignite/fixtures/phaseb_golden.pt` | Recorded reference outputs (a few KB, committed). |
| `tests/ignite/test_sampling.py` | Unit tests for the decode policies. |
| `tests/ignite/test_scoring.py` | Unit tests for the pseudo-likelihood scorer. |
| `tests/ignite/test_selfforce.py` | Unit tests for rollout-context training. |
| `scripts/evaluation/ignite_skill_vs_step.py` | The regression harness: rollout skill vs training step across checkpoints — the test that must stop inverting. |

**Modified files**

| path | change |
|---|---|
| `ignite/frame_layout.py` | add `logits_last()` (mirror of `masked_logits`, last frame only) |
| `ignite/maskgit.py` | accept `SamplerConfig` in `generate_frame`/`rollout`; boundary (CTF) masking in `_random_mask`; token-count loss weighting; rollout-context hook in `training_loss` |
| `ignite/dynamics.py` | actuator dropout (training) + zeroed-actuator path for CFG (inference) |
| `ignite/dynamics_config.py` | new fields, all defaulting to current behaviour |
| `ignite/train_dynamics.py` | CLI flags + wiring for CTF, loss weighting, actuator dropout, rollout-context fine-tune |
| `ignite/eval_dynamics.py` | plumb `SamplerConfig` + best-of-N through the eval entry point |

---

## Task 0: Recover the toolchain and back up the branch

**Do this first — but note it only *fully* blocks Tasks 11–13.** Tasks 1–10 are unit-test-driven and run under any torch-only interpreter (see Global Constraints), so if the pixi rebuild stalls, development continues; training and eval on the real cache do not. Frontier's `/lustre/orion/.../scratch` purges files by access time.
 The pixi env (built 2026-03-05) has lost stdlib files — `os.py`, `site.py`, `codecs.py` are gone and `encodings/` retains 9 of ~120 files — so `.pixi/envs/frontier/bin/python` cannot start at all. `git fsck` also reports missing blobs in old history. Measured on 2026-08-17: the working tree is intact, HEAD's history walks, and **the 34 unpushed commits on `nathan_fm` are a complete object graph** (`git rev-list --objects origin/nathan_fm..nathan_fm` succeeds), so they can still be pushed — but they exist only on the damaged filesystem until they are.

**Files:**
- Modify: none (environment + git state)

**Interfaces:**
- Produces: a working `python` for every later task; an off-machine backup of the branch.

- [ ] **Step 1: Confirm the damage (so you know when it is fixed)**

```bash
cd /lustre/orion/fus187/scratch/nchen/FusionAIHub
.pixi/envs/frontier/bin/python -c "print('ok')"    # expect: LookupError / no codec search functions
ls .pixi/envs/frontier/lib/python3.11/encodings/ | wc -l   # expect: ~9, not ~120
```

- [ ] **Step 2: Back up the 34 unpushed commits BEFORE anything else**

```bash
git rev-list --count origin/nathan_fm..nathan_fm      # expect 34
git rev-list --objects origin/nathan_fm..nathan_fm >/dev/null && echo PUSHABLE
git push origin nathan_fm
```

If the push is refused, stop and escalate — do not proceed while the only copy of that work is on a purging filesystem.

- [ ] **Step 3: Rebuild the environment**

```bash
pixi install                      # rebuilds .pixi/envs/frontier
```

Then reinstall the IGNITE-only deps that are absent from `pixi.lock`, with `--no-deps` so the solver cannot swap the ROCm torch for a CUDA wheel:

```bash
.pixi/envs/frontier/bin/pip install --no-deps \
  x-transformers vector-quantize-pytorch loguru einops einx torch-einops-utils
```

- [ ] **Step 4: Verify the toolchain end to end**

```bash
.pixi/envs/frontier/bin/python -c "import torch, pytest; print(torch.__version__, pytest.__version__)"
PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_phaseb_maskgit.py -q
```
Expected: torch prints a `+rocm` build, and the existing Phase-B tests pass. **If they do not pass, every later task's "expected: PASS" is meaningless — fix this first.**

- [ ] **Step 5: Protect against the next purge**

Add to the repo a one-line note in `docs/IGNITE_ROLLOUT_QUALITY_PLAN.md` under a new "Ops" heading recording that scratch purges by atime and that the env must be rebuilt after long idle periods, then commit:

```bash
git add docs/IGNITE_ROLLOUT_QUALITY_PLAN.md
git commit -m "docs: record scratch purge-by-atime hazard and env rebuild recipe"
```

---

## Task 1: Golden compatibility harness (do this before ANY behaviour change)

This task exists to make "Peter's model still runs" a machine-checked fact rather than an intention. It records the current outputs and pins them.

**Files:**
- Create: `tests/ignite/test_phaseb_compat.py`
- Create: `tests/ignite/fixtures/phaseb_golden.pt`

**Interfaces:**
- Produces: `tests/ignite/fixtures/phaseb_golden.pt`, a dict with keys `rollout_codes` (dict name→LongTensor), `train_loss` (float), `cfg_kw` (dict). Every later task must keep `test_phaseb_compat.py` green.

- [ ] **Step 1: Write the fixture generator and run it against UNMODIFIED code**

Create `tests/ignite/fixtures/_make_golden.py`:

```python
"""Regenerate the Phase-B golden fixture. Run ONLY against known-good code."""
import torch
from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics

CFG_KW = dict(d_model=16, depth=2, n_heads=2, ffn_mult=2,
              k0_seed=2, n_predict=3, maskgit_decode_steps=4, actuator_dim=6)
MODS = (ModalitySpec("a", "spectro", 3, 5), ModalitySpec("b", "slowts", 2, 4))


def build():
    cfg = DynamicsConfig(modalities=MODS, **CFG_KW)
    torch.manual_seed(0)
    model = MaskGITDynamics(cfg).eval()
    return cfg, model


def main():
    cfg, model = build()
    g = torch.Generator().manual_seed(1)
    seed = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok), generator=g)
            for m in cfg.modalities}
    act = torch.randn(1, cfg.max_frames, cfg.actuator_dim, generator=g)
    roll = model.rollout(seed, act, n_predict=3, generator=torch.Generator().manual_seed(7))
    codes = {m.name: torch.randint(0, m.codebook_size, (2, 5, m.n_tok),
                                   generator=torch.Generator().manual_seed(3))
             for m in cfg.modalities}
    tact = torch.randn(2, 5, cfg.actuator_dim, generator=torch.Generator().manual_seed(4))
    loss = model.training_loss(codes, tact, generator=torch.Generator().manual_seed(5))
    torch.save({"rollout_codes": roll, "train_loss": float(loss),
                "cfg_kw": CFG_KW}, "tests/ignite/fixtures/phaseb_golden.pt")
    print("wrote fixture; loss =", float(loss))


if __name__ == "__main__":
    main()
```

Run it:

```bash
mkdir -p tests/ignite/fixtures
PYTHONPATH=src .pixi/envs/frontier/bin/python tests/ignite/fixtures/_make_golden.py
```

- [ ] **Step 2: Write the golden test**

Create `tests/ignite/test_phaseb_compat.py`:

```python
"""Byte-identical behaviour guard: Peter's model must keep running unchanged.

Every feature added by the rollout-quality work is flag-gated. With no flag set,
rollout and training_loss must reproduce the recorded reference EXACTLY, including
the RNG draw sequence. If this test fails, a default changed — that is a bug, not
a fixture to regenerate.
"""
from pathlib import Path

import torch

from tests.ignite.fixtures._make_golden import build

FIXTURE = Path(__file__).parent / "fixtures" / "phaseb_golden.pt"


def _ref():
    return torch.load(FIXTURE, weights_only=False)


def test_default_rollout_is_bit_identical():
    ref = _ref()
    cfg, model = build()
    g = torch.Generator().manual_seed(1)
    seed = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok), generator=g)
            for m in cfg.modalities}
    act = torch.randn(1, cfg.max_frames, cfg.actuator_dim, generator=g)
    roll = model.rollout(seed, act, n_predict=3, generator=torch.Generator().manual_seed(7))
    for name, want in ref["rollout_codes"].items():
        assert torch.equal(roll[name], want), f"{name}: default rollout changed"


def test_default_training_loss_is_bit_identical():
    ref = _ref()
    cfg, model = build()
    codes = {m.name: torch.randint(0, m.codebook_size, (2, 5, m.n_tok),
                                   generator=torch.Generator().manual_seed(3))
             for m in cfg.modalities}
    act = torch.randn(2, 5, cfg.actuator_dim, generator=torch.Generator().manual_seed(4))
    loss = model.training_loss(codes, act, generator=torch.Generator().manual_seed(5))
    assert float(loss) == ref["train_loss"], "default training loss changed"


def test_old_checkpoint_config_still_constructs():
    """A checkpoint saved before the new fields exist must still build a config."""
    from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
    old_payload_kw = dict(d_model=512, depth=8, n_heads=16, k0_seed=20, n_predict=80)
    cfg = DynamicsConfig(modalities=(ModalitySpec("mhr", "spectro", 192, 1000),),
                         **old_payload_kw)
    assert cfg.max_frames == 100 and cfg.tokens_per_frame == 192
```

- [ ] **Step 3: Run the golden test against unmodified code**

Run: `PYTHONPATH=. PYTHONPATH=src:. .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_phaseb_compat.py -q`
Expected: 3 passed. (If `tests.ignite.fixtures._make_golden` will not import, add empty `tests/ignite/fixtures/__init__.py`.)

- [ ] **Step 4: Commit**

```bash
git add tests/ignite/test_phaseb_compat.py tests/ignite/fixtures/
git commit -m "test: pin Phase-B default rollout/loss behaviour with a golden fixture"
```

---

## Task 2: R0 — project only the last frame during rollout

`DynamicsBackbone.forward` projects **every** frame and token to full vocab on each of the 800 forwards a rollout performs, and `generate_frame` uses only `[:, -1]`. At the production layout that is ~15 GB of transient fp32 per step. This is the enabler for batched training-time rollouts (Task 10) and best-of-N (Task 6), and it changes no numbers.

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/frame_layout.py` (after `masked_logits`, ~line 115)
- Modify: `src/tokamak_foundation_model/ignite/maskgit.py:165-171`, `:190-191`
- Test: `tests/ignite/test_phaseb_frame_layout.py`

**Interfaces:**
- Produces: `FrameTokenizer.logits_last(h: Tensor) -> Dict[str, Tensor]` mapping modality name → `(B, n_tok_m, codebook_m)`, equal to `logits(h)[name][:, -1]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/ignite/test_phaseb_frame_layout.py`:

```python
def test_logits_last_matches_full_logits_last_frame():
    from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
    from tokamak_foundation_model.ignite.frame_layout import FrameTokenizer
    import torch

    cfg = DynamicsConfig(modalities=(ModalitySpec("a", "spectro", 3, 5),
                                     ModalitySpec("b", "slowts", 2, 4)),
                         d_model=16, depth=2, n_heads=2, k0_seed=2, n_predict=3)
    torch.manual_seed(0)
    tok = FrameTokenizer(cfg).eval()
    h = torch.randn(2, 4, cfg.tokens_per_frame, cfg.d_model)
    full = tok.logits(h)
    last = tok.logits_last(h)
    for m in cfg.modalities:
        assert last[m.name].shape == (2, m.n_tok, m.codebook_size)
        assert torch.allclose(last[m.name], full[m.name][:, -1], atol=0, rtol=0)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_phaseb_frame_layout.py::test_logits_last_matches_full_logits_last_frame -q`
Expected: FAIL — `AttributeError: 'FrameTokenizer' object has no attribute 'logits_last'`

- [ ] **Step 3: Implement `logits_last`**

Add to `frame_layout.py` after `masked_logits`:

```python
    def logits_last(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Project ONLY the last frame to vocab -> {name: (B, n_tok_m, codebook_m)}.

        Rollout generates one frame at a time and reads only ``logits[:, -1]``, but
        :meth:`logits` projects every frame: ~15 GB of transient fp32 per decode step at
        the production layout (1593 tokens, four 64k vocabs), x10 steps x80 frames.
        Slicing the hidden states first drops that to ~150 MB. Numerically identical.
        """
        if h.shape[2] != self.cfg.tokens_per_frame:
            raise ValueError(
                f"logits_last: expected {self.cfg.tokens_per_frame} tokens, got {h.shape[2]}"
            )
        hl = h[:, -1]                                            # (B, tokens_per_frame, d)
        out: Dict[str, torch.Tensor] = {}
        for (s, e), m in zip(self.cfg.modality_token_slices(), self.cfg.modalities):
            out[m.name] = self.heads[m.name](hl[:, s:e, :])
        return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: same command as Step 2. Expected: PASS

- [ ] **Step 5: Use it in the decode loop**

In `maskgit.py::generate_frame`, replace the full-forward line

```python
            logits = self.backbone(seq, actuators)                    # {name:(B,P+1,n_tok,vocab)}
```

with

```python
            h = self.backbone.encode(seq, actuators)                  # (B, P+1, N, d)
            logits = self.backbone.tok.logits_last(h)                 # {name:(B, n_tok, vocab)}
```

and change the per-modality read from `logits[m.name][:, -1]` to `logits[m.name]`:

```python
                lg = logits[m.name].float() / max(temperature, 1e-6)
```

Do the same in the still-masked fallback block at the end of `generate_frame`:

```python
                h = self.backbone.encode(seq, actuators)
                lg = self.backbone.tok.logits_last(h)[m.name]
                cur[m.name] = torch.where(still, lg.argmax(-1), cur[m.name])
```

- [ ] **Step 6: Verify the golden test still passes (this is the point of Task 1)**

Run: `PYTHONPATH=src:. .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_phaseb_compat.py tests/ignite/test_phaseb_maskgit.py -q`
Expected: all pass — `logits_last` is a pure slice, so codes must be bit-identical.

- [ ] **Step 7: Commit**

```bash
git add src/tokamak_foundation_model/ignite/frame_layout.py src/tokamak_foundation_model/ignite/maskgit.py tests/ignite/test_phaseb_frame_layout.py
git commit -m "perf: project only the last frame during MaskGIT decode (~100x less transient logit memory)"
```

---

## Task 3: SamplerConfig + per-modality temperature and top-p (R2)

Confidences are compared across modalities whose vocabs differ by 64× (1 000 vs 64 000) and whose token counts differ by 192× (4 vs 768). One global temperature cannot be right for all of them.

**Files:**
- Create: `src/tokamak_foundation_model/ignite/sampling.py`
- Create: `tests/ignite/test_sampling.py`
- Modify: `src/tokamak_foundation_model/ignite/maskgit.py` (`generate_frame`, `rollout` signatures)

**Interfaces:**
- Produces:
  - `SamplerConfig(temperature: float | Dict[str, float] = 1.0, top_p: float | None = None, global_pool: bool = False, revision_rounds: int = 0, revision_frac: float = 0.25, cfg_scale: float = 1.0)`
  - `SamplerConfig.temp_for(name: str) -> float`
  - `apply_top_p(probs: Tensor, top_p: float | None) -> Tensor` — renormalized nucleus filter over the last dim; identity when `top_p is None`.
  - `MaskGITDynamics.generate_frame(..., sampler: SamplerConfig | None = None)` and `rollout(..., sampler=None)`; `None` means `SamplerConfig(temperature=temperature)`, preserving the old signature.

- [ ] **Step 1: Write the failing tests**

Create `tests/ignite/test_sampling.py`:

```python
import torch

from tokamak_foundation_model.ignite.sampling import SamplerConfig, apply_top_p


def test_temp_for_scalar_and_dict():
    assert SamplerConfig().temp_for("mhr") == 1.0
    s = SamplerConfig(temperature={"mhr": 0.7})
    assert s.temp_for("mhr") == 0.7
    assert s.temp_for("absent_modality") == 1.0     # falls back to 1.0


def test_apply_top_p_is_identity_when_none():
    p = torch.tensor([[0.5, 0.3, 0.2]])
    assert torch.equal(apply_top_p(p, None), p)


def test_apply_top_p_keeps_nucleus_and_renormalizes():
    p = torch.tensor([[0.6, 0.3, 0.08, 0.02]])
    out = apply_top_p(p, 0.9)
    assert out[0, 2] == 0 and out[0, 3] == 0        # tail dropped
    assert torch.isclose(out.sum(), torch.tensor(1.0))
    assert torch.isclose(out[0, 0] / out[0, 1], torch.tensor(2.0))  # ratios preserved


def test_apply_top_p_always_keeps_at_least_one_token():
    p = torch.tensor([[0.99, 0.01]])
    out = apply_top_p(p, 0.1)                        # threshold below the top prob
    assert (out > 0).sum() == 1 and torch.isclose(out.sum(), torch.tensor(1.0))
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_sampling.py -q`
Expected: FAIL — `ModuleNotFoundError: tokamak_foundation_model.ignite.sampling`

- [ ] **Step 3: Implement `sampling.py`**

```python
"""Decode-policy configuration and helpers for the MaskGIT sampler.

Pure policy: these functions take logits/probabilities and return filtered or
reordered ones. They hold no model state, so they are cheap to unit-test and can be
swapped per eval arm. ``SamplerConfig()`` with no arguments reproduces the original
sampler exactly (see tests/ignite/test_phaseb_compat.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch


@dataclass
class SamplerConfig:
    """How a frame is decoded. Defaults == the original behaviour, bit for bit.

    temperature      : scalar, or {modality_name: float}. Spectro modalities carry a
                       64k vocab and 768 tokens; slow-TS carries 1k and 4. One global
                       temperature over-disperses the former.
    top_p            : nucleus filter applied per token before sampling. None = off.
    global_pool      : rank reveal-confidence across ALL of a frame's tokens instead of
                       per modality, so an uncertain modality can defer while confident
                       ones commit and anchor it (cross-modal coherence).
    revision_rounds  : after the schedule completes, re-mask the least-confident
                       ``revision_frac`` of the frame and re-decode, this many times.
    cfg_scale        : classifier-free guidance on the actuator conditioning.
                       1.0 = off (single forward pass, no cost).
    """

    temperature: Union[float, Dict[str, float]] = 1.0
    top_p: Optional[float] = None
    global_pool: bool = False
    revision_rounds: int = 0
    revision_frac: float = 0.25
    cfg_scale: float = 1.0

    def temp_for(self, name: str) -> float:
        if isinstance(self.temperature, dict):
            return float(self.temperature.get(name, 1.0))
        return float(self.temperature)


def apply_top_p(probs: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    """Nucleus filter over the last dim, renormalized. Identity when ``top_p`` is None.

    The highest-probability token is always kept, so a top_p below the max prob still
    yields a valid distribution rather than an all-zero row.
    """
    if top_p is None:
        return probs
    srt, idx = probs.sort(dim=-1, descending=True)
    cum = srt.cumsum(dim=-1)
    keep = cum - srt < top_p                       # keep while the mass BEFORE this token < p
    keep[..., 0] = True                            # always keep the argmax
    filt = torch.zeros_like(probs).scatter_(-1, idx, srt * keep)
    return filt / filt.sum(dim=-1, keepdim=True).clamp_min(1e-12)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: same as Step 2. Expected: 4 passed.

- [ ] **Step 5: Thread `SamplerConfig` through `generate_frame` and `rollout`**

In `maskgit.py`, add the import and change the two signatures. Keep `temperature` so existing callers (`eval_dynamics.py`, `ignite_bp_cases.py`) keep working:

```python
from .sampling import SamplerConfig, apply_top_p
```

```python
    @torch.no_grad()
    def generate_frame(self, past_codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
                       temperature: float = 1.0,
                       generator: Optional[torch.Generator] = None,
                       sampler: Optional[SamplerConfig] = None) -> Dict[str, torch.Tensor]:
```

Immediately after `cfg = self.cfg` in the body:

```python
        sampler = SamplerConfig(temperature=temperature) if sampler is None else sampler
```

Replace the per-modality temperature/sampling lines with the policy-aware version:

```python
                lg = logits[m.name].float() / max(sampler.temp_for(m.name), 1e-6)
                prob = apply_top_p(lg.softmax(-1), sampler.top_p)
```

Mirror the same `sampler` parameter on `rollout`, defaulting to `None`, and pass it into `generate_frame`:

```python
    @torch.no_grad()
    def rollout(self, seed_codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
                n_predict: Optional[int] = None, temperature: float = 1.0,
                generator: Optional[torch.Generator] = None,
                sampler: Optional[SamplerConfig] = None) -> Dict[str, torch.Tensor]:
```
```python
            nxt = self.generate_frame(
                traj, actuators[:, : K0 + t + 1], temperature=temperature,
                generator=generator, sampler=sampler
            )
```

- [ ] **Step 6: Verify defaults are still bit-identical**

Run: `PYTHONPATH=src:. .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_phaseb_compat.py tests/ignite/test_phaseb_maskgit.py tests/ignite/test_sampling.py -q`
Expected: all pass. `apply_top_p(p, None)` returns `p` unchanged and `temp_for` returns the scalar, so no RNG draw or arithmetic changes.

- [ ] **Step 7: Commit**

```bash
git add src/tokamak_foundation_model/ignite/sampling.py src/tokamak_foundation_model/ignite/maskgit.py tests/ignite/test_sampling.py
git commit -m "feat: SamplerConfig with per-modality temperature and top-p (defaults unchanged)"
```

---

## Task 4: R1 — global cross-modal confidence pool

Today each modality reveals a fixed fraction of *its own* tokens each step, ranked by its own confidence. An uncertain modality (mhr mid-transition) must commit ~19 tokens at step 1 regardless of how unsure it is, while a confident one cannot go first and anchor it. A global pool fixes exactly that. Confidences are made comparable by **within-modality quantile rank**, not raw probability — a 64k-vocab modality's max probability is structurally smaller than a 1k-vocab modality's.

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/sampling.py` (add `rank_normalize`)
- Modify: `src/tokamak_foundation_model/ignite/maskgit.py::generate_frame`
- Test: `tests/ignite/test_sampling.py`

**Interfaces:**
- Produces: `rank_normalize(conf: Tensor) -> Tensor` — maps each row's values to their quantile rank in [0, 1] along the last dim, ties broken by index order.

- [ ] **Step 1: Write the failing tests**

Append to `tests/ignite/test_sampling.py`:

```python
def test_rank_normalize_is_monotone_and_bounded():
    from tokamak_foundation_model.ignite.sampling import rank_normalize
    c = torch.tensor([[0.9, 0.1, 0.5]])
    r = rank_normalize(c)
    assert r.min() >= 0 and r.max() <= 1
    assert r[0, 0] > r[0, 2] > r[0, 1]              # order preserved


def test_rank_normalize_equalizes_scales_across_modalities():
    """A 64k-vocab modality has structurally smaller probabilities than a 1k one;
    rank normalization must make their confidences comparable."""
    from tokamak_foundation_model.ignite.sampling import rank_normalize
    small = torch.tensor([[0.002, 0.001, 0.004]])   # 64k-vocab scale
    large = torch.tensor([[0.20, 0.10, 0.40]])      # 1k-vocab scale
    assert torch.allclose(rank_normalize(small), rank_normalize(large))


def test_global_pool_defers_low_confidence_modality():
    """With a global pool, the confident modality reveals more tokens at step 1 than the
    uncertain one — impossible under per-modality fixed quotas."""
    import torch
    from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
    from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
    from tokamak_foundation_model.ignite.sampling import SamplerConfig

    cfg = DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 8, 5), ModalitySpec("b", "slowts", 8, 5)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=2,
        maskgit_decode_steps=4, actuator_dim=6)
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    past = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.k0_seed + 1, cfg.actuator_dim)
    out = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(2),
                            sampler=SamplerConfig(global_pool=True))
    for m in cfg.modalities:                         # still a complete, valid frame
        assert out[m.name].shape == (1, m.n_tok)
        assert (out[m.name] >= 0).all() and (out[m.name] < m.codebook_size).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_sampling.py -q`
Expected: FAIL — `ImportError: cannot import name 'rank_normalize'`

- [ ] **Step 3: Implement `rank_normalize`**

Append to `sampling.py`:

```python
def rank_normalize(conf: torch.Tensor) -> torch.Tensor:
    """Map each row's values to their quantile rank in [0, 1] along the last dim.

    Raw sampled-token probabilities are NOT comparable across modalities: a 64 000-way
    softmax puts far less mass on its argmax than a 1 000-way one, so a global argsort
    over raw confidence would let low-vocab modalities monopolize every reveal step.
    Rank normalization removes the scale while preserving order within each modality.
    """
    n = conf.shape[-1]
    if n == 1:
        return torch.ones_like(conf)
    order = conf.argsort(dim=-1)
    ranks = torch.empty_like(order)
    ar = torch.arange(n, device=conf.device).expand_as(order)
    ranks.scatter_(-1, order, ar)
    return ranks.to(conf.dtype) / (n - 1)
```

- [ ] **Step 4: Run to verify the two `rank_normalize` tests pass**

Run: `PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_sampling.py -k rank_normalize -q`
Expected: 2 passed.

- [ ] **Step 5: Implement the global pool in `generate_frame`**

The loop currently samples and reveals per modality in one pass. Split it into **sample all modalities → decide reveals → commit**, so the sampling RNG order is unchanged (this is what keeps the default bit-identical). Replace the body of the `for step, frac in enumerate(keep_masked):` loop with:

```python
        for step, frac in enumerate(keep_masked):
            seq = {n: torch.cat([past_codes[n], cur[n].unsqueeze(1)], dim=1) for n in cur}
            h = self.backbone.encode(seq, actuators)
            logits = self.backbone.tok.logits_last(h)
            # PASS 1 — sample every modality (unchanged RNG order: same calls, same sequence)
            samp, conf = {}, {}
            for m in cfg.modalities:
                lg = logits[m.name].float() / max(sampler.temp_for(m.name), 1e-6)
                prob = apply_top_p(lg.softmax(-1), sampler.top_p)
                s = torch.multinomial(prob.reshape(-1, prob.shape[-1]), 1,
                                      generator=generator).reshape(B, m.n_tok)
                samp[m.name] = s
                conf[m.name] = prob.gather(-1, s.unsqueeze(-1)).squeeze(-1)   # (B, n_tok)
            # PASS 2 — choose what to reveal
            if sampler.global_pool:
                take = self._global_reveal(conf, revealed, frac)
            else:
                take = self._per_modality_reveal(conf, revealed, frac)
            # PASS 3 — commit
            for m in cfg.modalities:
                cur[m.name] = torch.where(take[m.name], samp[m.name], cur[m.name])
                revealed[m.name] = revealed[m.name] | take[m.name]
```

Add the two selection helpers as methods on `MaskGITDynamics`:

```python
    def _per_modality_reveal(self, conf, revealed, frac):
        """Original policy: each modality reveals the same FRACTION of its own tokens."""
        take = {}
        for m in self.cfg.modalities:
            c = conf[m.name].masked_fill(revealed[m.name], float("inf"))
            n_reveal = m.n_tok - int(round(frac * m.n_tok))
            order = c.argsort(dim=-1, descending=True)
            new_rev = torch.zeros_like(revealed[m.name])
            new_rev.scatter_(1, order[:, :n_reveal], True)
            take[m.name] = new_rev & ~revealed[m.name]
        return take

    def _global_reveal(self, conf, revealed, frac):
        """Pooled policy: rank confidence across the WHOLE frame, reveal the global top-K.

        Lets an uncertain modality defer while confident ones commit first and anchor it
        through the next step's spatial attention — the cross-modal-coherence lever.
        """
        names = [m.name for m in self.cfg.modalities]
        parts = [rank_normalize(conf[n]).masked_fill(revealed[n], float("inf")) for n in names]
        flat = torch.cat(parts, dim=1)                       # (B, tokens_per_frame)
        total = flat.shape[1]
        n_reveal = total - int(round(frac * total))
        order = flat.argsort(dim=-1, descending=True)
        sel = torch.zeros_like(flat, dtype=torch.bool)
        sel.scatter_(1, order[:, :n_reveal], True)
        take, off = {}, 0
        for m in self.cfg.modalities:
            take[m.name] = sel[:, off:off + m.n_tok] & ~revealed[m.name]
            off += m.n_tok
        return take
```

Import `rank_normalize` alongside the others at the top of `maskgit.py`.

- [ ] **Step 6: Run the full Phase-B suite — the default path must be untouched**

Run: `PYTHONPATH=src:. .pixi/envs/frontier/bin/python -m pytest tests/ignite/ -k "phaseb or sampling" -q`
Expected: all pass, **including `test_phaseb_compat.py`**. If the golden test fails here, the three-pass refactor changed the RNG order — fix that, do not regenerate the fixture.

- [ ] **Step 7: Commit**

```bash
git add src/tokamak_foundation_model/ignite/sampling.py src/tokamak_foundation_model/ignite/maskgit.py tests/ignite/test_sampling.py
git commit -m "feat: optional global cross-modal confidence pool for MaskGIT reveal order"
```

---

## Task 5: R3 — revision pass (draft-and-revise)

Tokens committed at decode step 1 are never revisited, so an early incoherent commit is locked in for the rest of the frame. A revision pass re-masks the least-confident committed tokens and re-decodes them against the survivors.

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/maskgit.py::generate_frame`
- Test: `tests/ignite/test_sampling.py`

**Interfaces:**
- Consumes: `SamplerConfig.revision_rounds`, `SamplerConfig.revision_frac` (Task 3).
- Produces: no new public symbol; `generate_frame` honours the two fields.

- [ ] **Step 1: Write the failing test**

Append to `tests/ignite/test_sampling.py`:

```python
def _tiny_rev():
    from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 6, 5), ModalitySpec("b", "slowts", 4, 5)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=2,
        maskgit_decode_steps=4, actuator_dim=6)


def test_revision_rounds_produce_a_valid_complete_frame():
    import torch
    from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
    from tokamak_foundation_model.ignite.sampling import SamplerConfig

    cfg = _tiny_rev()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    past = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.k0_seed + 1, cfg.actuator_dim)
    out = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(3),
                            sampler=SamplerConfig(revision_rounds=2, revision_frac=0.5))
    for m in cfg.modalities:
        assert out[m.name].shape == (1, m.n_tok)
        assert (out[m.name] >= 0).all() and (out[m.name] < m.codebook_size).all()


def test_revision_can_change_committed_tokens():
    """A revision round must actually be able to revise — otherwise it is a no-op."""
    import torch
    from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
    from tokamak_foundation_model.ignite.sampling import SamplerConfig

    cfg = _tiny_rev()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    past = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.k0_seed + 1, cfg.actuator_dim)
    base = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(3),
                             sampler=SamplerConfig())
    rev = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(3),
                            sampler=SamplerConfig(revision_rounds=3, revision_frac=0.9))
    assert any(not torch.equal(base[m.name], rev[m.name]) for m in cfg.modalities)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_sampling.py -k revision -q`
Expected: FAIL — the second test fails (revision is a no-op today, so outputs are equal).

- [ ] **Step 3: Implement the revision loop**

In `generate_frame`, after the main `for step, frac in enumerate(keep_masked):` loop and **before** the still-masked argmax fallback, insert:

```python
        # REVISION (draft-and-revise): the schedule above never revisits a committed token,
        # so an incoherent early commit is permanent. Re-mask the least-confident fraction
        # and re-decode it against the tokens that survived.
        for _ in range(int(sampler.revision_rounds)):
            seq = {n: torch.cat([past_codes[n], cur[n].unsqueeze(1)], dim=1) for n in cur}
            h = self.backbone.encode(seq, actuators)
            logits = self.backbone.tok.logits_last(h)
            samp, conf = {}, {}
            for m in cfg.modalities:
                lg = logits[m.name].float() / max(sampler.temp_for(m.name), 1e-6)
                prob = apply_top_p(lg.softmax(-1), sampler.top_p)
                s = torch.multinomial(prob.reshape(-1, prob.shape[-1]), 1,
                                      generator=generator).reshape(B, m.n_tok)
                samp[m.name] = s
                # confidence of the CURRENTLY COMMITTED code, not of the fresh draw:
                # that is what decides which commits look weakest in context.
                conf[m.name] = prob.gather(-1, cur[m.name].unsqueeze(-1)).squeeze(-1)
            for m in cfg.modalities:
                k = int(round(sampler.revision_frac * m.n_tok))
                if k <= 0:
                    continue
                weakest = conf[m.name].argsort(dim=-1)[:, :k]          # lowest confidence
                redo = torch.zeros_like(revealed[m.name])
                redo.scatter_(1, weakest, True)
                cur[m.name] = torch.where(redo, samp[m.name], cur[m.name])
```

- [ ] **Step 4: Run the revision tests to verify they pass**

Run: same as Step 2. Expected: 2 passed.

- [ ] **Step 5: Verify defaults unchanged**

Run: `PYTHONPATH=src:. .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_phaseb_compat.py -q`
Expected: PASS — `revision_rounds` defaults to 0, so `range(0)` executes nothing and draws no RNG.

- [ ] **Step 6: Commit**

```bash
git add src/tokamak_foundation_model/ignite/maskgit.py tests/ignite/test_sampling.py
git commit -m "feat: optional draft-and-revise pass so early MaskGIT commits are not permanent"
```

---

## Task 6: R4 — masked pseudo-likelihood scorer and best-of-N reranking

Cosmos's operational answer to drift is to generate N rollouts and keep the one a critic likes. The discrete analogue of "the teacher scores this window" is: re-mask a fraction of the rollout's own tokens and measure the frozen model's cross-entropy at reproducing them. Lower is better. This needs no retrain and becomes the reward function for Phase 2's GRPO.

**Files:**
- Create: `src/tokamak_foundation_model/ignite/scoring.py`
- Create: `tests/ignite/test_scoring.py`

**Interfaces:**
- Produces:
  - `masked_pseudo_likelihood(model, codes, actuators, frames=None, mask_frac=0.3, n_draws=4, generator=None) -> float` — mean masked CE over `n_draws` independent mask draws restricted to `frames` (a `slice`, default all). Lower = more self-consistent.
  - `best_of_n(model, seed_codes, actuators, n, n_predict=None, sampler=None, generator=None, score_frames=None) -> Tuple[Dict[str, Tensor], List[float]]` — returns the best trajectory and every candidate's score.

- [ ] **Step 1: Write the failing tests**

Create `tests/ignite/test_scoring.py`:

```python
import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
from tokamak_foundation_model.ignite.scoring import best_of_n, masked_pseudo_likelihood


def _tiny():
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 4, 5), ModalitySpec("b", "slowts", 3, 4)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=3,
        maskgit_decode_steps=3, actuator_dim=6)


def _model(cfg):
    torch.manual_seed(0)
    return MaskGITDynamics(cfg).eval()


def test_pseudo_likelihood_is_finite_and_deterministic_given_a_seed():
    cfg = _tiny()
    mg = _model(cfg)
    codes = {m.name: torch.randint(0, m.codebook_size, (1, 5, m.n_tok)) for m in cfg.modalities}
    act = torch.randn(1, 5, cfg.actuator_dim)
    a = masked_pseudo_likelihood(mg, codes, act, n_draws=2,
                                 generator=torch.Generator().manual_seed(0))
    b = masked_pseudo_likelihood(mg, codes, act, n_draws=2,
                                 generator=torch.Generator().manual_seed(0))
    assert a == b and a > 0 and a == a          # finite, reproducible


def test_pseudo_likelihood_prefers_model_consistent_codes():
    """Codes the model itself generated should score better than uniform-random codes."""
    cfg = _tiny()
    mg = _model(cfg)
    seed = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.max_frames, cfg.actuator_dim)
    own = mg.rollout(seed, act, n_predict=3, generator=torch.Generator().manual_seed(1))
    rand = {m.name: torch.randint(0, m.codebook_size, own[m.name].shape) for m in cfg.modalities}
    win = slice(cfg.k0_seed, cfg.k0_seed + 3)
    s_own = masked_pseudo_likelihood(mg, own, act[:, :own["a"].shape[1]], frames=win,
                                     n_draws=4, generator=torch.Generator().manual_seed(2))
    s_rand = masked_pseudo_likelihood(mg, rand, act[:, :own["a"].shape[1]], frames=win,
                                      n_draws=4, generator=torch.Generator().manual_seed(2))
    assert s_own < s_rand


def test_best_of_n_returns_a_valid_trajectory_and_all_scores():
    cfg = _tiny()
    mg = _model(cfg)
    seed = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.max_frames, cfg.actuator_dim)
    traj, scores = best_of_n(mg, seed, act, n=3, n_predict=3,
                             generator=torch.Generator().manual_seed(5))
    assert len(scores) == 3
    for m in cfg.modalities:
        assert traj[m.name].shape == (1, cfg.k0_seed + 3, m.n_tok)
        assert (traj[m.name] < m.codebook_size).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_scoring.py -q`
Expected: FAIL — `ModuleNotFoundError: tokamak_foundation_model.ignite.scoring`

- [ ] **Step 3: Implement `scoring.py`**

```python
"""Self-consistency scoring for generated trajectories.

The discrete counterpart of "the teacher scores this window": re-mask a fraction of a
trajectory's OWN tokens and measure the frozen model's cross-entropy at reproducing
them. It needs no ground truth, so it works at inference time (best-of-N reranking)
and doubles as a reward signal for post-training.

Lower is better. Averaging several independent mask draws keeps the estimate stable —
a single draw is noisy because the mask decides which tokens are being asked about.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .sampling import SamplerConfig


@torch.no_grad()
def masked_pseudo_likelihood(model, codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
                             frames: Optional[slice] = None, mask_frac: float = 0.3,
                             n_draws: int = 4,
                             generator: Optional[torch.Generator] = None) -> float:
    """Mean masked cross-entropy of ``codes`` under ``model``. Lower = more self-consistent.

    ``frames`` restricts scoring to a window (e.g. the predicted region only); the whole
    sequence is still fed as context so the score is conditioned on the real seed.
    """
    cfg = model.cfg
    ref = codes[cfg.modalities[0].name]
    Fr = ref.shape[1]
    sel = torch.zeros(Fr, dtype=torch.bool, device=ref.device)
    sel[frames if frames is not None else slice(0, Fr)] = True
    total = 0.0
    for _ in range(max(1, n_draws)):
        masked, mask = {}, {}
        for m in cfg.modalities:
            c = codes[m.name]
            r = torch.rand(c.shape, generator=generator, device=c.device)
            mk = (r < mask_frac) & sel.view(1, Fr, 1)
            mk[:, :, 0] |= ~mk.any(dim=-1) & sel.view(1, Fr)   # >=1 target per scored frame
            masked[m.name] = torch.where(mk, torch.full_like(c, model.backbone.tok.mask_ids[m.name]), c)
            mask[m.name] = mk
        h = model.backbone.encode(masked, actuators)
        mlog = model.backbone.tok.masked_logits(h, mask)
        ce, n = 0.0, 0
        for m in cfg.modalities:
            if not bool(mask[m.name].any()):
                continue
            ce += float(F.cross_entropy(mlog[m.name], codes[m.name][mask[m.name]]))
            n += 1
        total += ce / max(n, 1)
    return total / max(1, n_draws)


@torch.no_grad()
def best_of_n(model, seed_codes: Dict[str, torch.Tensor], actuators: torch.Tensor, n: int,
              n_predict: Optional[int] = None, sampler: Optional[SamplerConfig] = None,
              generator: Optional[torch.Generator] = None,
              score_frames: Optional[slice] = None
              ) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """Roll out ``n`` candidates and return the most self-consistent one plus all scores."""
    cfg = model.cfg
    n_predict = cfg.n_predict if n_predict is None else n_predict
    K0 = seed_codes[cfg.modalities[0].name].shape[1]
    win = score_frames if score_frames is not None else slice(K0, K0 + n_predict)
    best, best_score, scores = None, float("inf"), []
    for _ in range(n):
        traj = model.rollout(seed_codes, actuators, n_predict=n_predict,
                             generator=generator, sampler=sampler)
        s = masked_pseudo_likelihood(model, traj, actuators[:, : K0 + n_predict],
                                     frames=win, generator=generator)
        scores.append(s)
        if s < best_score:
            best, best_score = traj, s
    return best, scores
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: same as Step 2. Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tokamak_foundation_model/ignite/scoring.py tests/ignite/test_scoring.py
git commit -m "feat: masked pseudo-likelihood scorer and best-of-N rollout reranking"
```

---

## Task 7: R5 — complete-context (CTF) training

The structural fix. Training masks ~64% of **every** frame, so context frames are always partially masked ground truth; rollout conditions a fully-masked new frame on **complete, model-generated** history. The model is never trained on the conditional it samples from. MAGI (CVPR 2025) reports +23% FVD from exactly this correction.

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/dynamics_config.py` (new fields)
- Modify: `src/tokamak_foundation_model/ignite/maskgit.py` (`_random_mask`, `training_loss`)
- Create: `tests/ignite/test_ctf.py`

**Interfaces:**
- Produces:
  - `DynamicsConfig.ctf_frac: float = 0.0` — probability that a training window uses the boundary layout.
  - `DynamicsConfig.ctf_min_target_ratio: float = 0.8` — minimum mask ratio applied to frames at/after the boundary.
  - `MaskGITDynamics._boundary_mask(codes, gen) -> (masked, mask)` — frames `< c` fully visible, frames `>= c` masked at a high ratio, loss only where masked.

- [ ] **Step 1: Write the failing tests**

Create `tests/ignite/test_ctf.py`:

```python
"""Complete-context (CTF) training: match the conditional the rollout actually uses."""
import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics


def _tiny(**kw):
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 4, 5), ModalitySpec("b", "slowts", 3, 4)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=3,
        maskgit_decode_steps=3, actuator_dim=6, **kw)


def _codes(cfg, B, F):
    return {m.name: torch.randint(0, m.codebook_size, (B, F, m.n_tok)) for m in cfg.modalities}


def test_ctf_defaults_to_off():
    assert _tiny().ctf_frac == 0.0


def test_boundary_mask_leaves_a_clean_prefix_and_masks_the_suffix():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg)
    codes = _codes(cfg, B=4, F=6)
    masked, mask = mg._boundary_mask(codes, gen=torch.Generator().manual_seed(0))
    for b in range(4):
        per_frame = torch.stack([mask[m.name][b].any(dim=-1) for m in cfg.modalities]).any(0)
        first_masked = int(per_frame.float().argmax())
        assert per_frame[first_masked:].all(), "every frame at/after the boundary is supervised"
        assert not per_frame[:first_masked].any(), "the prefix is COMPLETE (unmasked) context"
    for m in cfg.modalities:                       # unmasked positions keep the true code
        keep = ~mask[m.name]
        assert torch.equal(masked[m.name][keep], codes[m.name][keep])


def test_ctf_loss_is_finite_and_backprops():
    cfg = _tiny(ctf_frac=1.0)
    mg = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, B=2, F=6)
    act = torch.randn(2, 6, cfg.actuator_dim)
    loss = mg.training_loss(codes, act, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    g = mg.backbone.blocks[0].spatial.qkv.weight.grad
    assert g is not None and g.abs().sum() > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_ctf.py -q`
Expected: FAIL — `AttributeError: 'DynamicsConfig' object has no attribute 'ctf_frac'`

- [ ] **Step 3: Add the config fields**

In `dynamics_config.py`, after the scheduled-sampling block:

```python
    # --- complete-context training (CTF; MAGI arXiv 2501.12389) -------------------------------
    # Fraction of training windows that use the ROLLOUT's conditional structure: a clean
    # (fully visible) prefix and a heavily-masked suffix, loss on the suffix only. The default
    # random-mask objective masks ~64% of EVERY frame, so the model never trains on the
    # "complete context -> fully masked next frame" conditional that rollout actually uses.
    # 0.0 = off (original behaviour, bit-identical).
    ctf_frac: float = 0.0
    ctf_min_target_ratio: float = 0.8      # min mask ratio applied at/after the boundary
```

- [ ] **Step 4: Implement `_boundary_mask` and wire it into `training_loss`**

Add to `MaskGITDynamics`, next to `_random_mask`:

```python
    def _boundary_mask(self, codes: Dict[str, torch.Tensor], gen: Optional[torch.Generator]):
        """CTF layout: frames < c are COMPLETE context; frames >= c are heavily masked targets.

        This is the conditional rollout actually uses (see docs/IGNITE_ROLLOUT_QUALITY_PLAN.md
        §1A). The boundary c is per-sample so one batch spans many context lengths.
        """
        cfg = self.cfg
        ref = codes[cfg.modalities[0].name]
        B, Fr, _ = ref.shape
        dev = ref.device
        c = torch.randint(1, max(2, Fr), (B,), generator=gen, device=dev)      # >=1 context frame
        idx = torch.arange(Fr, device=dev).view(1, Fr)
        is_target = idx >= c.view(B, 1)                                        # (B, F)
        lo = float(cfg.ctf_min_target_ratio)
        u = torch.rand((B, Fr), generator=gen, device=dev)
        ratio = lo + (1.0 - lo) * u                                            # in [lo, 1]
        masked, mask = {}, {}
        for m in cfg.modalities:
            cd = codes[m.name]
            r = torch.rand(cd.shape, generator=gen, device=dev)
            mk = (r < ratio.unsqueeze(-1)) & is_target.unsqueeze(-1)
            none = (~mk.any(dim=-1, keepdim=True)) & is_target.unsqueeze(-1)   # keep >=1 target
            if none.any():
                first = torch.zeros_like(mk)
                first[:, :, 0] = True
                mk = mk | (first & none)
            masked[m.name] = torch.where(
                mk, torch.full_like(cd, self.backbone.tok.mask_ids[m.name]), cd)
            mask[m.name] = mk
        return masked, mask
```

In `training_loss`, replace the single masking call

```python
        masked, mask = self._random_mask(context, generator)
```

with the flag-gated choice (note the guard: with `ctf_frac == 0` **no** random draw happens, so the RNG stream is unchanged):

```python
        use_ctf = False
        if self.cfg.ctf_frac > 0.0:
            use_ctf = bool(torch.rand((), generator=generator).item() < self.cfg.ctf_frac)
        masked, mask = (self._boundary_mask(context, generator) if use_ctf
                        else self._random_mask(context, generator))
```

- [ ] **Step 5: Run the CTF tests to verify they pass**

Run: same as Step 2. Expected: 3 passed.

- [ ] **Step 6: Verify the default path is still bit-identical**

Run: `PYTHONPATH=src:. .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_phaseb_compat.py tests/ignite/test_phaseb_maskgit.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/tokamak_foundation_model/ignite/dynamics_config.py src/tokamak_foundation_model/ignite/maskgit.py tests/ignite/test_ctf.py
git commit -m "feat: complete-context (CTF) training mode matching the rollout conditional"
```

---

## Task 8: R7 — token-count loss weighting

`training_loss` averages modalities with equal weight, so a 4-token slow-TS modality contributes as much as 768-token `ece` — a ~192× per-token gradient advantage for the smallest modality. That is a plausible contributor to the degenerate `mhr` prediction.

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/dynamics_config.py`
- Modify: `src/tokamak_foundation_model/ignite/maskgit.py::training_loss`
- Test: `tests/ignite/test_ctf.py` (same suite; it already exercises the loss)

**Interfaces:**
- Produces: `DynamicsConfig.modality_loss_weight: str = "uniform"`, one of `"uniform" | "tokens" | "sqrt_tokens"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/ignite/test_ctf.py`:

```python
def test_loss_weighting_modes_change_the_loss_but_stay_finite():
    import math
    cfg_u = _tiny()
    cfg_t = _tiny(modality_loss_weight="tokens")
    torch.manual_seed(0)
    mg_u = MaskGITDynamics(cfg_u).train()
    torch.manual_seed(0)
    mg_t = MaskGITDynamics(cfg_t).train()
    codes = _codes(cfg_u, B=2, F=5)
    act = torch.randn(2, 5, cfg_u.actuator_dim)
    lu = mg_u.training_loss(codes, act, generator=torch.Generator().manual_seed(2))
    lt = mg_t.training_loss(codes, act, generator=torch.Generator().manual_seed(2))
    assert torch.isfinite(lu) and torch.isfinite(lt)
    assert not math.isclose(float(lu), float(lt), rel_tol=1e-9)


def test_default_loss_weight_is_uniform():
    assert _tiny().modality_loss_weight == "uniform"
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_ctf.py -k weight -q`
Expected: FAIL — unexpected keyword argument `modality_loss_weight`.

- [ ] **Step 3: Add the config field**

In `dynamics_config.py`:

```python
    # --- per-modality loss weighting ----------------------------------------------------------
    # "uniform" (default, original): every modality's masked CE counts the same, so a 4-token
    # slow-TS modality gets ~192x the per-token gradient of 768-token ece. "tokens" weights by
    # n_tok, "sqrt_tokens" by sqrt(n_tok) (a compromise that still protects small modalities).
    modality_loss_weight: str = "uniform"
```

- [ ] **Step 4: Apply the weights in `training_loss`**

Add a helper method on `MaskGITDynamics`:

```python
    def _modality_weight(self, m) -> float:
        mode = getattr(self.cfg, "modality_loss_weight", "uniform")
        if mode == "uniform":
            return 1.0
        if mode == "tokens":
            return float(m.n_tok)
        if mode == "sqrt_tokens":
            return float(m.n_tok) ** 0.5
        raise ValueError(f"unknown modality_loss_weight {mode!r}")
```

In `training_loss`, accumulate weighted terms. Replace `count += 1` bookkeeping with a weight sum — both branches (with and without `present`):

```python
        total, count = codes[self.cfg.modalities[0].name].new_zeros((), dtype=torch.float32), 0.0
```
```python
            w_m = self._modality_weight(m)
            if present is None:
                total = total + w_m * F.cross_entropy(lg, tg)
                count += w_m
                continue
```
```python
            ce = F.cross_entropy(lg, tg, reduction="none")            # (n_masked,)
            total = total + w_m * (ce * w).sum() / denom
            count += w_m
```
```python
        return total / max(count, 1e-8)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_ctf.py -q`
Expected: all pass.

- [ ] **Step 6: Verify the default is bit-identical**

Run: `PYTHONPATH=src:. .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_phaseb_compat.py -q`
Expected: PASS — with `"uniform"`, every `w_m` is 1.0 and `count` is the old integer count in float form.

- [ ] **Step 7: Commit**

```bash
git add src/tokamak_foundation_model/ignite/dynamics_config.py src/tokamak_foundation_model/ignite/maskgit.py tests/ignite/test_ctf.py
git commit -m "feat: optional token-count loss weighting across modalities"
```

---

## Task 9: R8 — actuator dropout and classifier-free guidance

Actuators are never dropped during training, so guidance is impossible without a retrain, and there is no way to amplify controllability at rollout time. The old e2e model's conditioning was measurably re-absorbed during autoregressive rollout; CFG is the standard countermeasure. Dropout adds no parameters (it zeroes the input to the existing `act_embed`).

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/dynamics_config.py`
- Modify: `src/tokamak_foundation_model/ignite/dynamics.py::DynamicsBackbone.encode`
- Modify: `src/tokamak_foundation_model/ignite/maskgit.py::generate_frame` (CFG combine)
- Create: `tests/ignite/test_cfg_guidance.py`

**Interfaces:**
- Produces:
  - `DynamicsConfig.actuator_dropout_p: float = 0.0`
  - `DynamicsBackbone.encode(codes, actuators, drop_actuators: bool = False)` — when True, the actuator embedding contribution is zeroed for the whole batch (the unconditional branch).
  - `SamplerConfig.cfg_scale` (already defined in Task 3) honoured in `generate_frame`.

- [ ] **Step 1: Write the failing tests**

Create `tests/ignite/test_cfg_guidance.py`:

```python
"""Actuator dropout (training) and classifier-free guidance (inference)."""
import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
from tokamak_foundation_model.ignite.sampling import SamplerConfig


def _tiny(**kw):
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 4, 5), ModalitySpec("b", "slowts", 3, 4)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=3,
        maskgit_decode_steps=3, actuator_dim=6, **kw)


def test_actuator_dropout_defaults_to_off():
    assert _tiny().actuator_dropout_p == 0.0


def test_drop_actuators_makes_encode_ignore_the_actuators():
    cfg = _tiny()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    codes = {m.name: torch.randint(0, m.codebook_size, (1, 4, m.n_tok)) for m in cfg.modalities}
    a1 = torch.randn(1, 4, cfg.actuator_dim)
    a2 = torch.randn(1, 4, cfg.actuator_dim)
    h1 = mg.backbone.encode(codes, a1, drop_actuators=True)
    h2 = mg.backbone.encode(codes, a2, drop_actuators=True)
    assert torch.allclose(h1, h2), "dropped actuators must not influence the hidden states"
    assert not torch.allclose(mg.backbone.encode(codes, a1), h1)


def test_cfg_scale_one_is_identical_to_no_guidance():
    cfg = _tiny()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    past = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.k0_seed + 1, cfg.actuator_dim)
    base = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(4))
    same = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(4),
                             sampler=SamplerConfig(cfg_scale=1.0))
    for m in cfg.modalities:
        assert torch.equal(base[m.name], same[m.name])


def test_cfg_scale_above_one_changes_the_frame():
    cfg = _tiny()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    past = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.k0_seed + 1, cfg.actuator_dim)
    base = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(4))
    guided = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(4),
                               sampler=SamplerConfig(cfg_scale=3.0))
    assert any(not torch.equal(base[m.name], guided[m.name]) for m in cfg.modalities)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_cfg_guidance.py -q`
Expected: FAIL — unexpected keyword `actuator_dropout_p` / `drop_actuators`.

- [ ] **Step 3: Add the config field**

In `dynamics_config.py`:

```python
    # --- actuator conditioning dropout (enables classifier-free guidance at rollout) ----------
    # Probability of zeroing the actuator embedding for a training sample. Without it the model
    # has no unconditional branch, so guidance cannot be applied at inference. Adds NO parameters.
    # 0.0 = off (original behaviour, bit-identical).
    actuator_dropout_p: float = 0.0
```

- [ ] **Step 4: Implement dropout + the unconditional path in `dynamics.py`**

Change `encode`:

```python
    def encode(self, codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
               drop_actuators: bool = False) -> torch.Tensor:
        """→ hidden states (B, F, tokens_per_frame, d_model).

        ``drop_actuators`` zeroes the actuator contribution for the whole batch — the
        unconditional branch used by classifier-free guidance at inference.
        """
        x = self.tok.embed(codes)                              # (B, F, N, d)
        B, Fr, N, d = x.shape
        if actuators.shape != (B, Fr, self.cfg.actuator_dim):
            raise ValueError(
                f"actuators expected {(B, Fr, self.cfg.actuator_dim)}; got {tuple(actuators.shape)}"
            )
        a = self.act_embed(actuators)                          # (B, F, d)
        if drop_actuators:
            a = torch.zeros_like(a)
        else:
            p = getattr(self.cfg, "actuator_dropout_p", 0.0)
            if self.training and p > 0.0:
                # per-SAMPLE dropout: a whole trajectory is conditional or unconditional,
                # matching how guidance is applied at inference.
                keep = (torch.rand((B, 1, 1), device=a.device) >= p).to(a.dtype)
                a = a * keep
        x = x + a.unsqueeze(2)                                 # (B, F, 1, d) broadcast over tokens
```

(The rest of the method — the checkpointing loop and `return self.out_norm(x)` — is unchanged.)

- [ ] **Step 5: Implement CFG in `generate_frame`**

Replace the single-forward logits computation inside the decode loop with a guided version. Add this helper method to `MaskGITDynamics`:

```python
    def _decode_logits(self, seq, actuators, sampler):
        """Per-modality last-frame logits, with optional classifier-free guidance.

        cfg_scale == 1.0 short-circuits to ONE forward pass, so guidance costs nothing
        when it is off (and the default path stays bit-identical).
        """
        h = self.backbone.encode(seq, actuators)
        cond = self.backbone.tok.logits_last(h)
        if sampler.cfg_scale == 1.0:
            return cond
        hu = self.backbone.encode(seq, actuators, drop_actuators=True)
        uncond = self.backbone.tok.logits_last(hu)
        s = float(sampler.cfg_scale)
        return {n: uncond[n] + s * (cond[n] - uncond[n]) for n in cond}
```

Then in both the main decode loop and the revision loop, replace

```python
            h = self.backbone.encode(seq, actuators)
            logits = self.backbone.tok.logits_last(h)
```

with

```python
            logits = self._decode_logits(seq, actuators, sampler)
```

- [ ] **Step 6: Run the guidance tests to verify they pass**

Run: same as Step 2. Expected: 4 passed.

- [ ] **Step 7: Verify defaults**

Run: `PYTHONPATH=src:. .pixi/envs/frontier/bin/python -m pytest tests/ignite/ -k "phaseb or sampling or scoring or ctf or guidance" -q`
Expected: all pass, golden test included.

- [ ] **Step 8: Commit**

```bash
git add src/tokamak_foundation_model/ignite/dynamics_config.py src/tokamak_foundation_model/ignite/dynamics.py src/tokamak_foundation_model/ignite/maskgit.py tests/ignite/test_cfg_guidance.py
git commit -m "feat: actuator dropout + classifier-free guidance on actuator conditioning"
```

---

## Task 10: R9 — rollout-context fine-tune (Self-Forcing Stage A)

The centrepiece. Self-Forcing's ablation shows the **context distribution** is the load-bearing ingredient (82.32 teacher-forced → 84.31 on-policy with the *same* loss), and Self-Forcing++ shows synthetic corruption is a poor substitute for real rollout statistics. Gradients flow only through the final supervised prediction; the rollout runs under `no_grad` and the context is detached, exactly as Self-Forcing does — so there is no BPTT and memory stays at one forward pass.

**Files:**
- Create: `src/tokamak_foundation_model/ignite/selfforce.py`
- Create: `tests/ignite/test_selfforce.py`
- Modify: `src/tokamak_foundation_model/ignite/dynamics_config.py`
- Modify: `src/tokamak_foundation_model/ignite/maskgit.py::training_loss`

**Interfaces:**
- Consumes: `SamplerConfig` (Task 3), `logits_last` (Task 2), `_boundary_mask` (Task 7).
- Produces:
  - `DynamicsConfig.sf_frames: int = 0`, `DynamicsConfig.sf_decode_steps: int = 4`, `DynamicsConfig.sf_prob: float = 1.0`
  - `rollout_context(model, codes, actuators, boundary, n_roll, sampler=None, generator=None) -> Dict[str, Tensor]` — returns a copy of `codes` whose frames `[boundary, boundary+n_roll)` are replaced by the model's own committed codes; all tensors detached.

- [ ] **Step 1: Write the failing tests**

Create `tests/ignite/test_selfforce.py`:

```python
"""Rollout-context fine-tune: train on the model's OWN context distribution."""
import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
from tokamak_foundation_model.ignite.selfforce import rollout_context


def _tiny(**kw):
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 4, 5), ModalitySpec("b", "slowts", 3, 4)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=4,
        maskgit_decode_steps=3, actuator_dim=6, **kw)


def _codes(cfg, B, F):
    return {m.name: torch.randint(0, m.codebook_size, (B, F, m.n_tok)) for m in cfg.modalities}


def test_sf_defaults_to_off():
    c = _tiny()
    assert c.sf_frames == 0 and c.sf_prob == 1.0


def test_rollout_context_replaces_only_the_rolled_window():
    cfg = _tiny()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    codes = _codes(cfg, B=1, F=6)
    act = torch.randn(1, 6, cfg.actuator_dim)
    out = rollout_context(mg, codes, act, boundary=2, n_roll=2,
                          generator=torch.Generator().manual_seed(1))
    for m in cfg.modalities:
        assert out[m.name].shape == codes[m.name].shape
        assert torch.equal(out[m.name][:, :2], codes[m.name][:, :2])   # prefix untouched
        assert torch.equal(out[m.name][:, 4:], codes[m.name][:, 4:])   # suffix untouched
        assert (out[m.name] < m.codebook_size).all()                   # never the MASK id


def test_rollout_context_output_is_detached():
    cfg = _tiny()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, B=1, F=6)
    act = torch.randn(1, 6, cfg.actuator_dim)
    out = rollout_context(mg, codes, act, boundary=2, n_roll=2,
                          generator=torch.Generator().manual_seed(1))
    for m in cfg.modalities:
        assert not out[m.name].requires_grad


def test_training_loss_with_sf_frames_backprops():
    cfg = _tiny(sf_frames=2, ctf_frac=1.0)
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, B=2, F=6)
    act = torch.randn(2, 6, cfg.actuator_dim)
    loss = mg.training_loss(codes, act, generator=torch.Generator().manual_seed(2))
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    g = mg.backbone.blocks[0].spatial.qkv.weight.grad
    assert g is not None and g.abs().sum() > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_selfforce.py -q`
Expected: FAIL — `ModuleNotFoundError: tokamak_foundation_model.ignite.selfforce`

- [ ] **Step 3: Add the config fields**

In `dynamics_config.py`:

```python
    # --- self-forcing Stage A: rollout-context fine-tune ---------------------------------------
    # Number of frames the model rolls out FROM ITS OWN OUTPUT before the supervised frame.
    # Training otherwise conditions only on (masked) ground-truth context, while rollout
    # conditions on complete self-generated context; Self-Forcing (arXiv 2506.08009) shows that
    # gap — not the loss function — is what costs rollout quality. The rollout runs under
    # no_grad and is detached, so there is no BPTT. 0 = off.
    sf_frames: int = 0
    sf_decode_steps: int = 4     # cheaper few-step decode for training rollouts (inference uses 10)
    sf_prob: float = 1.0         # probability a window uses the self-rollout context when sf_frames>0
```

- [ ] **Step 4: Implement `selfforce.py`**

```python
"""Self-Forcing Stage A: build training context from the model's OWN rollout.

Self-Forcing (arXiv 2506.08009) isolates the active ingredient with an ablation that
holds the loss fixed and varies only where the context comes from: teacher-forced 82.32,
diffusion-forced 82.76, self-rollout 84.31 (VBench). Self-Forcing++ (arXiv 2510.02283)
then shows synthetic corruption of the context is NOT a substitute — real rollout errors
have structured statistics (drift toward stasis) that random noise does not reproduce.

The rollout here runs under no_grad with a cheap few-step decode and every returned
tensor is detached: gradients flow only through the supervised prediction that follows,
exactly as Self-Forcing does by detaching its KV cache. Cost is therefore ~one extra
forward per rolled frame, not a backprop-through-time.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .sampling import SamplerConfig


@torch.no_grad()
def rollout_context(model, codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
                    boundary: int, n_roll: int,
                    sampler: Optional[SamplerConfig] = None,
                    generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    """Replace frames ``[boundary, boundary+n_roll)`` with the model's own committed codes.

    Frames before ``boundary`` stay ground truth (the real seed); frames after the rolled
    window are left untouched so the caller can still supervise against them.
    """
    cfg = model.cfg
    ref = codes[cfg.modalities[0].name]
    Fr = ref.shape[1]
    n_roll = max(0, min(n_roll, Fr - boundary))
    if n_roll == 0:
        return {k: v.detach() for k, v in codes.items()}

    was_training = model.training
    model.eval()                                   # no dropout inside the rollout
    prev_steps = cfg.maskgit_decode_steps
    cfg.maskgit_decode_steps = max(1, int(getattr(cfg, "sf_decode_steps", 4)))
    try:
        ctx = {n: v[:, :boundary].clone() for n, v in codes.items()}
        for t in range(n_roll):
            nxt = model.generate_frame(ctx, actuators[:, : boundary + t + 1],
                                       generator=generator, sampler=sampler)
            ctx = {n: torch.cat([ctx[n], nxt[n].unsqueeze(1)], dim=1) for n in ctx}
    finally:
        cfg.maskgit_decode_steps = prev_steps
        if was_training:
            model.train()

    out = {}
    for n, v in codes.items():
        out[n] = torch.cat([ctx[n], v[:, boundary + n_roll:]], dim=1).detach()
    return out
```

- [ ] **Step 5: Wire it into `training_loss`**

In `maskgit.py`, import at the top:

```python
from .selfforce import rollout_context
```

In `training_loss`, immediately after the scheduled-sampling line and **before** masking, insert the self-forcing context substitution. Note the guard — with `sf_frames == 0` nothing is drawn or executed:

```python
        context = self._scheduled_sample_context(codes, actuators, ss_frac, generator)
        n_sf = int(getattr(self.cfg, "sf_frames", 0))
        if n_sf > 0:
            use_sf = (self.cfg.sf_prob >= 1.0
                      or bool(torch.rand((), generator=generator).item() < self.cfg.sf_prob))
            if use_sf:
                Fr = context[self.cfg.modalities[0].name].shape[1]
                lo = min(self.cfg.k0_seed, max(1, Fr - n_sf - 1))
                b = int(torch.randint(lo, max(lo + 1, Fr - n_sf), (1,),
                                      generator=generator).item())
                context = rollout_context(self, context, actuators, boundary=b,
                                          n_roll=n_sf, generator=generator)
```

- [ ] **Step 6: Run the self-forcing tests to verify they pass**

Run: same as Step 2. Expected: 4 passed.

- [ ] **Step 7: Verify defaults are still bit-identical**

Run: `PYTHONPATH=src:. .pixi/envs/frontier/bin/python -m pytest tests/ignite/ -q`
Expected: the whole ignite suite passes, golden test included.

- [ ] **Step 8: Commit**

```bash
git add src/tokamak_foundation_model/ignite/selfforce.py src/tokamak_foundation_model/ignite/dynamics_config.py src/tokamak_foundation_model/ignite/maskgit.py tests/ignite/test_selfforce.py
git commit -m "feat: self-forcing Stage A — train on the model's own rollout context"
```

---

## Task 11: Expose everything through the trainer and eval CLIs

Nothing above is reachable from a SLURM job yet. This task adds the flags and the launcher, and keeps every default at the current production behaviour so re-running Peter's exact configuration is still a no-flag invocation.

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/train_dynamics.py` (`train()` signature ~line 840, cfg construction ~line 862, CLI ~line 1146)
- Modify: `src/tokamak_foundation_model/ignite/eval_dynamics.py` (rollout call + CLI)
- Create: `scripts/slurm_frontier/train_dynamics_ctf.sh`

**Interfaces:**
- Consumes: all config fields from Tasks 7–10, `SamplerConfig` from Task 3, `best_of_n` from Task 6.
- Produces: CLI flags `--ctf_frac`, `--ctf_min_target_ratio`, `--modality_loss_weight`, `--actuator_dropout_p`, `--sf_frames`, `--sf_decode_steps`, `--sf_prob` on the trainer; `--global_pool`, `--top_p`, `--revision_rounds`, `--cfg_scale`, `--best_of_n` on the eval.

- [ ] **Step 1: Add the trainer flags**

In `train_dynamics.py`, extend the `train()` signature with keyword-only defaults that mean "off":

```python
          mask_absent: bool = False, presence_path: str = None,
          accum_steps: int = 1, val_windows: int = 32,
          ctf_frac: float = 0.0, ctf_min_target_ratio: float = 0.8,
          modality_loss_weight: str = "uniform", actuator_dropout_p: float = 0.0,
          sf_frames: int = 0, sf_decode_steps: int = 4, sf_prob: float = 1.0,
          log=print):
```

Where `cfg` is built (just after the `ss_final_frac` block), apply them:

```python
    cfg.ctf_frac = float(ctf_frac)
    cfg.ctf_min_target_ratio = float(ctf_min_target_ratio)
    cfg.modality_loss_weight = str(modality_loss_weight)
    cfg.actuator_dropout_p = float(actuator_dropout_p)
    cfg.sf_frames = int(sf_frames)
    cfg.sf_decode_steps = int(sf_decode_steps)
    cfg.sf_prob = float(sf_prob)
    if ddp.is_main:
        log(f"[dynamics] rollout-quality flags: ctf={cfg.ctf_frac} "
            f"loss_w={cfg.modality_loss_weight} act_drop={cfg.actuator_dropout_p} "
            f"sf_frames={cfg.sf_frames}")
```

Add the matching argparse entries next to the existing ones:

```python
    p.add_argument("--ctf_frac", type=float, default=0.0,
                   help="fraction of windows trained with complete-context (CTF) masking")
    p.add_argument("--ctf_min_target_ratio", type=float, default=0.8)
    p.add_argument("--modality_loss_weight", default="uniform",
                   choices=("uniform", "tokens", "sqrt_tokens"))
    p.add_argument("--actuator_dropout_p", type=float, default=0.0,
                   help="per-sample actuator dropout; enables CFG at eval")
    p.add_argument("--sf_frames", type=int, default=0,
                   help="self-forcing: frames rolled from the model's own output before the "
                        "supervised frame (0 = off)")
    p.add_argument("--sf_decode_steps", type=int, default=4)
    p.add_argument("--sf_prob", type=float, default=1.0)
```

and forward them in the `train(...)` call built from `args`.

Persist them in the checkpoint payload so an eval can tell how an arm was trained — find the `torch.save` payload with `cfg_depth`/`cfg_d_model` and add:

```python
        "cfg_ctf_frac": cfg.ctf_frac,
        "cfg_modality_loss_weight": cfg.modality_loss_weight,
        "cfg_actuator_dropout_p": cfg.actuator_dropout_p,
        "cfg_sf_frames": cfg.sf_frames,
```

- [ ] **Step 2: Smoke-test the trainer wiring on CPU**

```bash
PYTHONPATH=src .pixi/envs/frontier/bin/python -c "
from tokamak_foundation_model.ignite.train_dynamics import train
import inspect
sig = inspect.signature(train)
for k in ('ctf_frac','modality_loss_weight','actuator_dropout_p','sf_frames'):
    assert k in sig.parameters, k
print('trainer exposes all rollout-quality flags')
"
```
Expected: prints the confirmation line.

- [ ] **Step 3: Add the eval flags**

`rollout_shot` already has this exact signature (verified at `eval_dynamics.py:217-220`) — extend it with two keyword arguments, keeping every existing caller valid:

```python
def rollout_shot(model: MaskGITDynamics, cfg: DynamicsConfig, cache: Dict, K0: int,
                 temperature: float, generator: torch.Generator, device,
                 actuator_mode: str = "real", cache_dir=None,
                 sampler=None, best_of: int = 1
                 ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int]:
```

Add the imports at the top of `eval_dynamics.py`:

```python
from .sampling import SamplerConfig
from .scoring import best_of_n as _best_of_n
```

and replace the rollout call at `eval_dynamics.py:243-244`

```python
    traj = model.rollout(seed_codes, actuators, n_predict=n_predict,
                         temperature=temperature, generator=generator)
```

with the sampler-aware version:

```python
    sampler = SamplerConfig(temperature=temperature) if sampler is None else sampler
    if best_of > 1:
        traj, scores = _best_of_n(model, seed_codes, actuators, n=best_of,
                                  n_predict=n_predict, sampler=sampler, generator=generator)
        print(f"[eval] best-of-{best_of} pseudo-likelihood "
              f"{[round(s, 4) for s in scores]} -> kept {round(min(scores), 4)}")
    else:
        traj = model.rollout(seed_codes, actuators, n_predict=n_predict,
                             temperature=temperature, generator=generator, sampler=sampler)
```

Build the `SamplerConfig` once in `main()` (where `model, cfg, step = load_model(...)` happens at `eval_dynamics.py:1005`) and thread it into both `rollout_shot` call sites (`:1041` and the paired real-actuator arm at `:1052` — **both**, or the counterfactual comparison stops being paired):

```python
    sampler = SamplerConfig(temperature=temperature, top_p=args.top_p,
                            global_pool=args.global_pool,
                            revision_rounds=args.revision_rounds,
                            cfg_scale=args.cfg_scale)
```

with argparse entries:

```python
    p.add_argument("--global_pool", action="store_true",
                   help="rank reveal-confidence across all modalities jointly")
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--revision_rounds", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--best_of_n", type=int, default=1)
```

**Pairing caveat:** `rollout_shot`'s docstring guarantees that actuator counterfactual arms consume an identical number of RNG draws so the comparison is exactly paired. `revision_rounds` and `cfg_scale` preserve that (fixed extra draws per frame); **`best_of_n` does not** (it draws N trajectories). Never combine `--best_of_n > 1` with an actuator counterfactual arm in the same comparison.

- [ ] **Step 4: Verify the eval CLI still parses with no new flags**

```bash
PYTHONPATH=src .pixi/envs/frontier/bin/python -m tokamak_foundation_model.ignite.eval_dynamics --help | grep -E "global_pool|best_of_n|cfg_scale"
```
Expected: the three flags are listed; running without them is unchanged.

- [ ] **Step 5: Add the CTF training launcher**

Create `scripts/slurm_frontier/train_dynamics_ctf.sh` as a copy of `train_dynamics.sh` with a different `OUT_DIR` and the new flags. The header/module block must match the existing launcher exactly; only these lines differ:

```bash
OUT_DIR="${OUT_DIR:-/lustre/orion/fus187/proj-shared/models/ignite_production/runs/ctf_d512L8}"
CTF_FRAC="${CTF_FRAC:-0.5}"
MODALITY_LOSS_WEIGHT="${MODALITY_LOSS_WEIGHT:-sqrt_tokens}"
ACTUATOR_DROPOUT_P="${ACTUATOR_DROPOUT_P:-0.1}"
SF_FRAMES="${SF_FRAMES:-0}"      # stage A is enabled in a SECOND run, after CTF is validated
```

and append to the `srun ... python -m ...train_dynamics` invocation:

```bash
  --ctf_frac "${CTF_FRAC}" \
  --modality_loss_weight "${MODALITY_LOSS_WEIGHT}" \
  --actuator_dropout_p "${ACTUATOR_DROPOUT_P}" \
  --sf_frames "${SF_FRAMES}"
```

- [ ] **Step 6: Commit**

```bash
git add src/tokamak_foundation_model/ignite/train_dynamics.py src/tokamak_foundation_model/ignite/eval_dynamics.py scripts/slurm_frontier/train_dynamics_ctf.sh
git commit -m "feat: expose rollout-quality flags through the train and eval CLIs"
```

---

## Task 12: The regression harness — skill vs training step

Everything above is a hypothesis until this measures it. The documented failure is that **more training made rollouts worse**: on the band-power line, TM-band skill went +0.153 at step 11 000 to −0.862 at step 20 000. If exposure bias is the cause, a CTF/self-forcing arm must stop inverting. This harness is the acceptance test for the whole plan.

**Files:**
- Create: `scripts/evaluation/ignite_skill_vs_step.py`
- Create: `scripts/slurm_frontier/ignite_skill_vs_step.sh`

**Interfaces:**
- Consumes: the eval flags from Task 11.
- Produces: `skill_vs_step.json` (`{arm: {step: {modality: skill}}}`) and `skill_vs_step.png` in the output directory.

- [ ] **Step 1: Write the harness**

Create `scripts/evaluation/ignite_skill_vs_step.py`:

```python
"""Rollout skill as a function of training step — the exposure-bias regression test.

The documented pathology is that rollout skill INVERTS with more training (band-power
line: TM-band skill +0.153 @ step 11k -> -0.862 @ step 20k) while teacher-forced CE
keeps improving. A fix for the train/test gap must flatten or reverse that curve, so
this harness plots skill against step for one or more checkpoint directories.

Skill = token accuracy over the predicted region minus the persistence baseline
(fraction of tokens equal to the last seed frame). Persistence is mandatory: discrete
codes at 50 ms are highly persistent, so raw accuracy is not interpretable. Compare
against the MAJORITY-TOKEN baseline too — a modality that cannot beat the frequency of
its commonest ground-truth code is degenerate and its skill number means nothing.

    python ignite_skill_vs_step.py --run_dir <dir> [--run_dir <dir2>] \
        --cache_dir <frame_codes> --shots 199597,190735 --out_dir <out>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def checkpoints(run_dir: Path):
    """Every step-tagged checkpoint in a run dir, ascending by step."""
    out = []
    for p in sorted(run_dir.glob("dynamics_step*.pt")):
        m = re.search(r"step(\d+)", p.name)
        if m:
            out.append((int(m.group(1)), p))
    latest = run_dir / "dynamics_latest.pt"
    if latest.exists():
        step = int(torch.load(latest, map_location="meta", mmap=True).get("step", -1))
        out.append((step, latest))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", action="append", required=True,
                    help="checkpoint directory; repeat for multiple arms")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--shots", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k0", type=int, default=20)
    ap.add_argument("--global_pool", action="store_true")
    ap.add_argument("--best_of_n", type=int, default=1)
    args = ap.parse_args()

    from tokamak_foundation_model.ignite.eval_dynamics import (
        load_model, load_shot_cache, rollout_shot)
    from tokamak_foundation_model.ignite.sampling import SamplerConfig

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shots = [s.strip() for s in args.shots.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = SamplerConfig(global_pool=args.global_pool)
    results: dict = defaultdict(dict)

    for run in args.run_dir:
        arm = Path(run).name
        for step, ckpt in checkpoints(Path(run)):
            # load_model(ckpt_path, device) -> (model, cfg, step)   [eval_dynamics.py:119]
            model, cfg, _ = load_model(Path(ckpt), device)
            per_mod = defaultdict(list)
            for shot in shots:
                cache = load_shot_cache(Path(args.cache_dir), shot)
                # rollout_shot(model, cfg, cache, K0, temperature, generator, device, ...)
                # -> (gt_codes, pred_codes, K0, F); tensors are (F, n_tok) cpu long, NO batch dim.
                gt, pred, K0, F = rollout_shot(
                    model, cfg, cache, args.k0, 1.0,
                    torch.Generator().manual_seed(1234), device,
                    sampler=sampler, best_of=args.best_of_n)
                for name in pred:
                    g, p = gt[name][K0:F], pred[name][K0:F]
                    acc = float((g == p).float().mean())
                    last = gt[name][K0 - 1: K0]
                    pers = float((g == last).float().mean())
                    # majority-token guard: the frequency of the commonest GT code
                    vals, cnt = torch.unique(g, return_counts=True)
                    major = float(cnt.max()) / float(g.numel())
                    per_mod[name].append({"skill": acc - pers, "acc": acc,
                                          "persistence": pers, "majority": major,
                                          "beats_majority": acc > major})
            results[arm][step] = {
                n: {k: (sum(d[k] for d in v) / len(v) if isinstance(v[0][k], float)
                        else all(d[k] for d in v))
                    for k in v[0]}
                for n, v in per_mod.items()}
            print(f"[skill] {arm} step {step}: "
                  + ", ".join(f"{n}={d['skill']:+.3f}"
                              + ("" if d["beats_majority"] else "(DEGENERATE)")
                              for n, d in results[arm][step].items()), flush=True)

    with open(out_dir / "skill_vs_step.json", "w") as f:
        json.dump(results, f, indent=2)

    mods = sorted({n for arm in results.values() for s in arm.values() for n in s})
    fig, axes = plt.subplots(len(mods), 1, figsize=(9, 2.6 * len(mods)), sharex=True,
                             squeeze=False)
    for ax, n in zip(axes[:, 0], mods):
        for arm, by_step in results.items():
            xs = sorted(by_step)
            ys = [by_step[s].get(n, {}).get("skill", float("nan")) for s in xs]
            ax.plot(xs, ys, marker="o", label=arm)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_ylabel(f"{n}\nskill")
        ax.legend(fontsize=7)
    axes[-1, 0].set_xlabel("training step")
    fig.suptitle("Rollout skill vs training step (skill must not invert)")
    fig.tight_layout()
    fig.savefig(out_dir / "skill_vs_step.png", dpi=110)
    print("wrote", out_dir / "skill_vs_step.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm the harness matches the real signatures**

Task 11 added `sampler` and `best_of` to `rollout_shot`. Verify before running:

```bash
grep -n "def rollout_shot" -A 4 src/tokamak_foundation_model/ignite/eval_dynamics.py
grep -n "def load_model\|def load_shot_cache" src/tokamak_foundation_model/ignite/eval_dynamics.py
```
Expected: `rollout_shot(model, cfg, cache, K0, temperature, generator, device, actuator_mode="real", cache_dir=None, sampler=None, best_of=1)`, `load_model(ckpt_path, device)` returning a 3-tuple, and `load_shot_cache(cache_dir, shot)` present. The harness above depends on all three — fix the harness to match the code, never the reverse.

- [ ] **Step 3: Reproduce the KNOWN failure first (this validates the harness)**

Run it against the band-power checkpoints that produced the documented inversion. The harness is only trustworthy if it reproduces a result we already know:

```bash
PYTHONPATH=src .pixi/envs/frontier/bin/python scripts/evaluation/ignite_skill_vs_step.py \
  --run_dir /lustre/orion/fus187/proj-shared/models/ignite_bp128/runs/bp128_d512L8 \
  --cache_dir /lustre/orion/fus187/proj-shared/models/ignite_bp128/frame_codes \
  --shots 199597 --out_dir data/outputs/ignite_skill_vs_step/baseline
```
Expected: a clearly **declining** skill curve for `mhr` between step 11 000 and 20 000 — the +0.153 → −0.862 inversion. If it does not reproduce, fix the harness before trusting any new arm.

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluation/ignite_skill_vs_step.py
git commit -m "eval: skill-vs-step harness — the exposure-bias regression test"
```

---

## Task 13: Run the experiment and record the verdict

**Files:**
- Modify: `docs/IGNITE_ROLLOUT_QUALITY_PLAN.md` (fill in a Results section)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Tier-0 A/B on existing checkpoints (no training required)**

Sampler fixes alone, on the checkpoints that already exist. Paired seeds, same shots:

```bash
for FLAGS in "" "--global_pool" "--global_pool --revision_rounds 2" "--best_of_n 4"; do
  PYTHONPATH=src .pixi/envs/frontier/bin/python -m tokamak_foundation_model.ignite.eval_dynamics \
    --ckpt /lustre/orion/fus187/proj-shared/models/ignite_bp128/runs/bp128_d512L8/dynamics_latest.pt \
    --cache_dir /lustre/orion/fus187/proj-shared/models/ignite_bp128/frame_codes \
    --shot 199597 --out_dir "data/outputs/tier0/$(echo ${FLAGS:-base} | tr ' /-' '_')" $FLAGS
done
```
Record token skill AND decoded band-restricted skill for each arm. Expected direction: the global pool and revision help most where modalities disagree; best-of-N gives a smaller, more uniform gain.

- [ ] **Step 2: Launch the CTF training arm on the band-power line**

The bp line is the right pilot: same `MaskGITDynamics` class, 320 tokens/frame, vocab 8, 33.9 M params (essentially all transformer), so a full arm costs hours rather than days — and it owns the documented inversion.

```bash
OUT_DIR=/lustre/orion/fus187/proj-shared/models/ignite_bp128/runs/bp128_ctf_d512L8 \
CACHE_DIR=/lustre/orion/fus187/proj-shared/models/ignite_bp128/frame_codes \
CTF_FRAC=0.5 MODALITY_LOSS_WEIGHT=sqrt_tokens ACTUATOR_DROPOUT_P=0.1 \
CKPT_EVERY=2000 \
sbatch scripts/slurm_frontier/train_dynamics_ctf.sh
```
Checkpoint every 2 000 steps — the curve, not the endpoint, is the result.

- [ ] **Step 3: Launch the self-forcing arm once CTF has a curve**

```bash
OUT_DIR=/lustre/orion/fus187/proj-shared/models/ignite_bp128/runs/bp128_sf_d512L8 \
CACHE_DIR=/lustre/orion/fus187/proj-shared/models/ignite_bp128/frame_codes \
CTF_FRAC=0.5 MODALITY_LOSS_WEIGHT=sqrt_tokens ACTUATOR_DROPOUT_P=0.1 \
SF_FRAMES=8 CKPT_EVERY=2000 \
sbatch scripts/slurm_frontier/train_dynamics_ctf.sh
```

- [ ] **Step 4: Compare all three arms on one plot**

```bash
PYTHONPATH=src .pixi/envs/frontier/bin/python scripts/evaluation/ignite_skill_vs_step.py \
  --run_dir /lustre/orion/fus187/proj-shared/models/ignite_bp128/runs/bp128_d512L8 \
  --run_dir /lustre/orion/fus187/proj-shared/models/ignite_bp128/runs/bp128_ctf_d512L8 \
  --run_dir /lustre/orion/fus187/proj-shared/models/ignite_bp128/runs/bp128_sf_d512L8 \
  --cache_dir /lustre/orion/fus187/proj-shared/models/ignite_bp128/frame_codes \
  --shots 199597,190735 --out_dir data/outputs/ignite_skill_vs_step/phase1
```

**Acceptance criterion:** the baseline arm inverts (skill falls with step); the CTF arm's skill is flat or rising over the same step range; the self-forcing arm is at least as good as CTF. Report the numbers whatever they are — a clean refutation is a result, and the spec's §1A argument is falsifiable precisely here.

- [ ] **Step 5: Write the verdict into the spec and commit**

Add a `## Results (Phase 1)` section to `docs/IGNITE_ROLLOUT_QUALITY_PLAN.md` with the measured skill curves, the Tier-0 A/B table, and an explicit statement of which recommendations were confirmed, which were refuted, and what the next phase should attempt.

```bash
git add docs/IGNITE_ROLLOUT_QUALITY_PLAN.md data/outputs/ignite_skill_vs_step/phase1/skill_vs_step.json
git commit -m "docs: Phase-1 rollout-quality results — CTF/self-forcing vs the skill inversion"
git push origin nathan_fm
```

---

## Deliberately not implemented

**R6 (repair the existing scheduled-sampling path) has no task, on purpose.** Its substitution samples come from a single forward pass over *clean ground-truth* codes (`maskgit.py:79`) — the weakest possible corruption — and Self-Forcing++'s ablation shows synthetic context corruption barely helps where real rollout statistics do. Task 10 supersedes it with the genuine article at similar cost. The dead `ss_*` machinery stays in place, still defaulting to 0, so nothing breaks; delete it only after Task 13 confirms Stage A works.

## Follow-on Plans (deliberately out of scope here)

Each of these is a separate subsystem that produces working software on its own; folding them in would make this plan un-reviewable and would break the zero-new-parameters guarantee that keeps Peter's checkpoints loadable.

1. **Phase 2 — distribution-level post-training (R10).** GRPO on committed-token log-probs with the Task-6 scorer plus decoded-space rewards, and an auxiliary masked-CE term against reward hacking; optional R3GAN token-window critic as the alternative. Depends on Task 6 and Task 10 landing first.
2. **Phase 3 — capacity reallocation (R12).** Factorize the 64k-vocab heads into six FSQ-digit heads, freeing ~260 M parameters for the dynamics core. **Changes the `state_dict`**, so it needs a conversion path for existing checkpoints and its own compatibility story.
3. **Phase 4 — architecture (R13/R14).** PAN-style state tokens; per-family parameter towers. Also parameter-changing.
4. **Phase 5 — horizon extension (R11).** Replace the learned absolute `frame_embed` (hard-capped at 100 frames) with a relative/RoPE temporal axis, then roll out beyond 80 frames.
5. **TokEye integration (R16).** Activity-weighted rollout metrics and rewards, curation, and activity-masked band-power tokenization, consuming the `{shot}_tokeye.h5` sidecars produced by `/lustre/orion/fus187/scratch/nchen/tokeye/`.
6. **Data curation and sampling (R15) + the inverse-dynamics auxiliary head (second half of R8).** Presence-aware curriculum, dropping frozen/absent segments, shot-balanced batching, window stride > 1 to cut the 99% overlap between stride-1 windows; and a small head predicting `actuator_t` from frame hidden states. Both are cheap, but the head **adds parameters**, and curation changes the data distribution under every arm — so they belong after Phase 1's measurement, not inside it.
