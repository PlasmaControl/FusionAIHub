# Labelmaker Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up `src/labelmaker/` in FusionAIHub so that `python -m labelmaker.run all --models d3d_tearing_onset_cnn1d` turns 100 corpus shots into per-shot tearing-mode probability label files in the corpus HDF5 layout, with a measured reliability report.

**Architecture:** Three stages with per-shot HDF5 files between them — `features` (a 25 ms archive store, the corpus HDF5, and fdp efit01/PTDATA/ZIPFIT → `<shot>_features.h5`), `infer` (per-model adapter builds arrays from the feature file, runs the trained weights, writes `<shot>_labels.h5`), `validate` (wrapper fidelity, reconstruction fidelity, label quality). One canonical feature namespace serves every model; a model folder contributes only a name mapping, a card, and a loader.

**Tech Stack:** Python 3.11 (pixi), numpy, h5py, scipy, pandas + pyarrow, toksearch/toksearch_d3d (fdp, `ga-fdp` channel), pytest. Trained weights are Keras 2.8 legacy HDF5, evaluated in numpy — **TensorFlow is not a runtime dependency** (see Global Constraints).

**Design spec:** `docs/superpowers/specs/2026-09-03-labelmaker-design.md`. Read Sections 9-13 before starting; this plan implements them.

## Global Constraints

- **Python** `>=3.11,<3.12` (pinned by `[tool.pixi.dependencies]`). The env has numpy 2.4.2, scipy 1.17.0, pandas 3.0.0, h5py 3.15.1, pyyaml 6.0.3, pytest 9.0.2.
- **Lint: `pixi run -e labelmaker ruff check src/labelmaker tests/labelmaker` must be clean.** Measured on 2026-09-03, and not what `pyproject.toml` appears to say: the repo declares only `[tool.ruff] line-length = 88`, but the installed ruff (0.16.5) applies a broad default rule set - `UP`, `B`, `S`, `BLE`, `ASYNC`, `YTT` among others - and **`E501` is not in it**. So the 88-column figure is a formatter target that nothing enforces, and the rest of the repo exceeds it 2,659 times. Stay near 88 columns to match the surrounding code, but a long line is not a defect; a `ruff check` finding is. Run it over the whole package, not just the files a task touched - two real errors (`UP037`, `UP017`) survived five task reviews because each one checked only its own files.
- **`except Exception` is deliberate where this plan uses it**, and ruff's default `BLE001` will flag it. Per-shot and per-signal isolation is the pipeline's core resilience property, so add `# noqa: BLE001` on those handlers with a short reason rather than narrowing the catch - an unforeseen exception class from h5py or toksearch is exactly what must not kill a 100-shot run.
- **No Pydantic.** Repo convention is `@dataclass(frozen=True)` and plain functions. Config over code.
- **No imports from `tokamak_foundation_model`.** Labelmaker sits on the `ignite/gate.py` side of the reuse boundary. Its dependencies are numpy, h5py, scipy, pandas + pyarrow, pyyaml (card front matter), stdlib, and import-guarded toksearch - every one of them declared in the `labelmaker` pixi feature or in `[project]`, never merely inherited. Conventions that are shared (the time base) are *copied with a comment naming the source*, not imported.
- **No TensorFlow anywhere in `src/labelmaker/`.** conda-forge's only `tensorflow-cpu` is 2.21.0 with a **py312** build, which cannot be installed next to this repo's `python <3.12` pin. TensorFlow appears exactly once, in a throwaway uv venv, in Task 14's fidelity check.
- **Data root** `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/`, override `LABELMAKER_ROOT`. Never write artifacts into `/scratch/gpfs/nc1514/` (near quota) and never into the repo.
- **Corpus** `/scratch/gpfs/EKOLEMEN/foundation_model/<shot>_processed.h5`, override `LABELMAKER_CORPUS`. Read-only.
- **HDF5 layout for everything we write:** one group per quantity, `xdata` float64 **seconds**, `ydata` float32 `(C, T)`. A group with `ydata.shape[-1] < 2` means "absent" — the corpus' own sentinel.
- **Every file we write is written atomically:** write `path.with_suffix(".tmp")` (or `path + ".tmp"`), close, then `Path.replace(final)`.
- **Every shot is isolated:** per-shot `try/except Exception`, record the exception class name, keep going. A per-shot `SIGALRM` timeout guards HDF5 reads that hang.
- **Time base:** sample rate from the span `fs = (n-1)/(x[-1]-x[0])`, sample index `round((t-x[0])*fs)` **clamped** into the record. Never `1/median(diff(x))` (float32 `xdata` loses the step to cancellation). Source: `src/tokamak_foundation_model/ignite/train_dynamics.py:177-192`.
- **Tests:** `tests/labelmaker/`, mirroring source modules, synthetic-first. `tests/` is a package — every new test directory needs `__init__.py`. There is no `conftest.py` and no pytest config in this repo; do not add one. Real-data tests gate with module-level `pytestmark = pytest.mark.skipif(not PATH.exists(), reason=...)`, matching `tests/ignite/test_fsq_overfit_realshot.py:111-113`. fdp network tests gate additionally on `os.environ.get("LABELMAKER_FDP") == "1"`.
- **Commits:** one per task, message style `labelmaker: <imperative summary>`. **No `Co-Authored-By` trailer of any kind.** Commit only the paths the task names. Work on branch `labelmaker` (Task 1 creates it); never commit to `nathan_fm` or `main`, never push.
- **Model artifact provenance:** weights are copied once into `<root>/models/<slug>/` with a `PROVENANCE.json` holding the upstream path, mtime and sha256 of every file. Inference reads the copy, never the upstream path.

## Deviations from the design spec (found while verifying the plan)

Fix the spec in the same commit as the task that touches each item.

1. **§10.4 upstream path.** The 10-member ensemble is `/projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w/best_model_{0..9}_4c.h5`, not the parent `rt_multi_io/` directory, which holds only members 0-4. `mse_bin_os_w` (MSE + binary cross-entropy, oversampled, class-weighted) is the variant the upstream reference harness `test/test.py` loads.
2. **§7, §10.3 framework runner.** The Keras runner is `models/runners/keras_h5.py`, a numpy evaluator for Keras-2 legacy HDF5 graphs, because no TensorFlow build exists for this env's Python (see Global Constraints). The `_4c.h5` artifacts contain only `InputLayer / BatchNormalization / Conv1D / MaxPooling1D / Flatten / Dense / Concatenate / Dropout` and a single `(None, 2)` output, so they evaluate exactly. (The sibling `best_model_i.h5` files add two `SlicingOpLambda` output heads and are *not* used.) `models/runners/torch.py` is **not** created in Phase 1 — no Phase 1 model is torch. YAGNI.
3. **§9.1 names carry provenance in the data, not in the name.** The spec's `<quantity>_<source>` naming assumed one source per quantity. Twelve of the Phase 1 features have two (an archive store and fdp), with different shot coverage, so provenance is per shot and cannot live in a static name. A `FeatureSpec` therefore has an ordered `sources` tuple, the canonical name is the physics quantity (with a fit qualifier only where the fit is part of the identity, e.g. `ne_zipfit`), and the group attribute `resolver` records which source actually produced it for that shot.
4. **§7 environment.** The `labelmaker` pixi feature adds `pyarrow` only, plus a CPU-torch index override, and the environment is `labelmaker = ["labelmaker", "fdp"]` — no `cuda`, no `tensorflow-cpu`. Labelmaker never uses a GPU; the cu124 torch wheels would cost ~3 GB in an env that only needs them because the workspace installs `faith` editable into every environment.
5. **§11 output grid.** Labels and adapter inputs live on a fixed 240-point grid `t = 0.000, 0.025, ..., 5.975 s` — the `times` dataset of the upstream reference file, and (verified below) the grid of the archive store, so archived rows and reconstructed rows are directly comparable. Timesteps whose inputs are missing or out of domain are written with a `_valid = 0` flag rather than dropped.
6. **§9.4, §16: the training features' actual upstream store was found, and it changes the Phase 1 critical path.** `/projects/EKOLEMEN/profile_predictor/DATA/new_h5_data/example_191450_183224.h5` holds 5,000 shots (183224-191450, 209 datasets each) on exactly the 240-point 25 ms grid. Measured on shot 185945 against the model's own training arrays (`x0/x1/y.npy`, 106 surviving rows, row mapping recovered by nearest-neighbour match on `[bt, ip, tritop, tribot, gapin]` at a median distance of 2.3e-7 and strictly monotonic):

   | training column | store column | median relative difference | correlation |
   |---|---|---|---|
   | `bt`, `ip`, `pinj`, `tinj` | same names | **0.0** (bit-identical) | 1.0000 |
   | `tritop_EFIT01`, `tribot_EFIT01`, `gapin_EFIT01` | same names | **0.0** (bit-identical) | 1.0000 |
   | `pres_EFIT01` (profile, at `t`) | `pres_EFIT01` | **0.0** (bit-identical) | 0.9995 |
   | `R0_EFITRT1` | `rmaxis_EFIT01` | 8.8e-3 | 0.882 |
   | `kappa_EFITRT1` | `kappa_EFIT01` | 3.1e-3 | 0.945 |
   | `1/qpsi_EFITRT1` (profile) | `1/qpsi_EFIT01` | 6.4e-2 | 0.978 |
   | `thomson_density_mtanh_1d` | `zipfit_edensfit_rho` | 2.0e-1 | 0.984 |
   | `thomson_temp_mtanh_1d` | `zipfit_etempfit_rho` | 1.8e-1 | 0.991 |
   | `cer_rot_csaps_1d` | `zipfit_trotfit_rho` | 1.7e-1 | 0.980 |
   | `betan_EFITRT1` (output) | `betan_EFIT01` | 1.7e-2 | 0.995 |

   The bit-identical rows also confirm the lag convention: the 0-D block is row `m` (`t+dt`) and the profile block is row `m-1` (`t`), matching `train.py`'s `[jump::jump]` / `[:-jump:jump]` slicing.

7. **§9.5 geometry leaves Phase 1.** Thomson and CER channel `(R, Z)` were needed only to re-fit the kinetic profiles the store already carries as ZIPFIT. So `features/geometry.py` and `features/resolve_fits.py` are **not** built in Phase 1, and the canonical profile features are `ne_zipfit`, `te_zipfit`, `rot_zipfit` - served from the archive store for the proof of concept and from the `zipfit01` MDSplus tree through fdp for everything else (omnimode.py:387 confirms that tree is reachable). Fitting our own mtanh/csaps profiles from raw Thomson/CER becomes a Phase 2 improvement measured against the 20% ZIPFIT baseline above, which is a far better justification for that work than a guess. (Verified for the record: the corpus files carry **zero** attributes and no channel positions; the only channel geometry on disk is `/scratch/gpfs/nc1514/omnimode/data/geometry/{shot}_geom.h5`, 17 shots, Thomson present in 2, none in the corpus range — produced by `/scratch/gpfs/nc1514/fdp/scripts/geometry.py` via `ImasSignal`.)
8. **§12 scaling is now quantified.** The corpus is 16,909 shots spanning **185601-204999**; only **3,621 (21.4%)** fall inside the archive store's range. The archive resolver serves the proof of concept (1,497 of the 1,503 overlap shots) and the fdp resolver is what makes the other 79% reachable. Phase 1 therefore builds and validates the fdp path on a handful of overlap shots, where the store provides an exact reference, instead of needing it for all 100.
9. **§9.4 corpus units and sampling, measured.** Corpus `pinj` is in **W** (shot 185945 at t=1.025 s: 1.002e7 summed over 8 beams) while the store and the model are in **kW** (10,995.6) — a factor of 1000 plus a ~10% difference that is a *sampling* difference, not a unit one: the store's 25 ms value is a window statistic and the corpus is a 10 kHz instantaneous sample. Corpus `tinj` is already in N m (8.43 vs 9.24). Corpus `ech_power` contains NaN channels (channel 3 on that shot), so totals need `nansum` before the upstream NaN-or-negative-to-zero rule. Which sampling convention reproduces the store is decided by measurement in Task 14, not by assumption.

## File Structure

Created under `/scratch/gpfs/nc1514/FusionAIHub/`:

| File | Responsibility |
|---|---|
| `src/labelmaker/__init__.py` | version string only |
| `src/labelmaker/config.py` | `Paths` (all filesystem roots, env overrides), `git_sha()` |
| `src/labelmaker/timebase.py` | span sample rate, clamped index, nearest sample, window mean, decimation |
| `src/labelmaker/catalog.py` | corpus shot enumeration, per-shot group availability, archive overlap, seeded PoC sample |
| `src/labelmaker/features/namespace.py` | `FeatureSpec` registry: canonical name → kind, units, ordered sources, locators |
| `src/labelmaker/features/store.py` | read/write `<shot>_features.h5`, per-shot `missing` bookkeeping, skip-if-complete |
| `src/labelmaker/features/resolve_corpus.py` | actuator totals (NBI power, torque, ECH) out of `<shot>_processed.h5`, in canonical units |
| `src/labelmaker/features/resolve_archive.py` | the 25 ms archive store: EFIT scalars/profiles, ZIPFIT profiles, `EC.RHO_ECH` |
| `src/labelmaker/features/resolve_fdp.py` | efit01 scalars/profiles + PTDATA `ip`/`bt` via toksearch (import-guarded) |
| `src/labelmaker/models/base.py` | `InputField`, `InputSpec`, `OutputField`, `OutputSpec`, `ModelAdapter`, transforms |
| `src/labelmaker/models/registry.py` | discover model folders, parse card front matter, cross-check against `spec.py` |
| `src/labelmaker/models/runners/keras_h5.py` | numpy evaluator for Keras-2 legacy HDF5 graphs |
| `src/labelmaker/models/d3d_tearing_onset_cnn1d/` | `spec.py` (the only per-model code), `README.md` (card), `__init__.py` |
| `src/labelmaker/models/<six scaffolds>/` | `README.md` (card, `status: scaffold`), `spec.py` raising `NotImplementedError` |
| `src/labelmaker/labels/schema.py` | `LabelSpec`; group naming rules |
| `src/labelmaker/labels/store.py` | write `<shot>_labels.h5` + provenance attrs; append to `labels_index.parquet` |
| `src/labelmaker/run.py` | CLI: `features`, `infer`, `validate`, `all`; shot isolation, timeout, run manifest |
| `src/labelmaker/validate.py` | the three reliability reports |
| `tests/labelmaker/` | one test module per source module, plus `data/` goldens |

---

### Task 1: Branch, package skeleton, config, packaging

**Files:**
- Create: `src/labelmaker/__init__.py`, `src/labelmaker/config.py`
- Create: `tests/labelmaker/__init__.py`, `tests/labelmaker/test_config.py`
- Modify: `pyproject.toml` (add `[tool.hatch.build.targets.wheel]`, `[tool.pixi.feature.labelmaker.*]`, one line in `[tool.pixi.environments]`)

**Interfaces:**
- Consumes: nothing.
- Produces: `labelmaker.__version__: str`; `labelmaker.config.Paths` frozen dataclass with fields `root: Path`, `corpus: Path`, classmethod `from_env() -> Paths`, properties `features/labels/models/runs/validation -> Path`, `labels_index -> Path`, methods `features_file(shot: int) -> Path`, `labels_file(shot: int) -> Path`, `corpus_file(shot: int) -> Path`, `mkdirs() -> None`; `labelmaker.config.git_sha() -> str`.

- [ ] **Step 1: Get the working tree into a committable state and create the branch**

The tree is mid-merge: `git status` shows 460 staged files and `scripts/training/train_e2e_stage1.py` unmerged (one conflict hunk). Git refuses every commit until that is resolved, so this must happen first. **This is Nathan's merge — do not resolve the conflict yourself.** Ask him to finish it, then:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git status --short | head            # expect: no "UU" / "AA" lines
git rev-parse -q --verify MERGE_HEAD # expect: no output (exit 1)
git checkout -b labelmaker
git branch --show-current            # expect: labelmaker
```

If the merge is still in progress, stop and report — do not start writing files, and do not run `git merge --abort` or `git checkout --ours/--theirs`.

- [ ] **Step 2: Write the failing test**

Create `tests/labelmaker/__init__.py` as an empty file (the `tests` tree is a package: `tests/__init__.py` and `tests/ignite/__init__.py` both exist).

`tests/labelmaker/test_config.py`:

```python
"""Paths resolve from the environment and nothing else hard-codes a root."""
from pathlib import Path

from labelmaker.config import Paths, git_sha


def test_default_root_is_group_storage():
    p = Paths()
    assert p.root == Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
    assert p.corpus == Path("/scratch/gpfs/EKOLEMEN/foundation_model")


def test_from_env_overrides_both_roots(monkeypatch, tmp_path):
    monkeypatch.setenv("LABELMAKER_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("LABELMAKER_CORPUS", str(tmp_path / "corpus"))
    p = Paths.from_env()
    assert p.root == tmp_path / "out"
    assert p.corpus == tmp_path / "corpus"


def test_per_shot_paths_and_mkdirs(tmp_path):
    p = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    assert p.features_file(190000) == tmp_path / "features" / "190000_features.h5"
    assert p.labels_file(190000) == tmp_path / "labels" / "190000_labels.h5"
    assert p.corpus_file(190000) == tmp_path / "corpus" / "190000_processed.h5"
    assert p.labels_index == tmp_path / "labels_index.parquet"
    p.mkdirs()
    for sub in ("features", "labels", "models", "runs", "validation"):
        assert (tmp_path / sub).is_dir()
    p.mkdirs()  # idempotent


def test_git_sha_is_a_string():
    sha = git_sha()
    assert isinstance(sha, str) and sha
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
.pixi/envs/default/bin/python -m pytest tests/labelmaker/test_config.py -q
```
Expected: collection error, `ModuleNotFoundError: No module named 'labelmaker'`.

- [ ] **Step 4: Write the package skeleton**

`src/labelmaker/__init__.py`:

```python
"""labelmaker - run the group's trained models over the FAITH shot corpus.

Stages are `features` -> `infer` -> `validate`, with a per-shot HDF5 file
between each, driven by `python -m labelmaker.run`. See
docs/superpowers/specs/2026-09-03-labelmaker-design.md.
"""

__version__ = "0.1.0"
```

`src/labelmaker/config.py`:

```python
"""Every filesystem root labelmaker reads or writes.

Nothing else in the package hard-codes a path, so pointing labelmaker at a
different data root (a scratch copy, a test fixture) is one environment
variable. Defaults are group storage: Nathan's own scratch is near quota.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
DEFAULT_CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")


@dataclass(frozen=True)
class Paths:
    """Where labelmaker's inputs and outputs live."""

    root: Path = DEFAULT_ROOT
    corpus: Path = DEFAULT_CORPUS

    @classmethod
    def from_env(cls) -> Paths:
        return cls(
            root=Path(os.environ.get("LABELMAKER_ROOT", str(DEFAULT_ROOT))),
            corpus=Path(os.environ.get("LABELMAKER_CORPUS", str(DEFAULT_CORPUS))),
        )

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def labels(self) -> Path:
        return self.root / "labels"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def validation(self) -> Path:
        return self.root / "validation"

    @property
    def labels_index(self) -> Path:
        return self.root / "labels_index.parquet"

    def features_file(self, shot: int) -> Path:
        return self.features / f"{shot}_features.h5"

    def labels_file(self, shot: int) -> Path:
        return self.labels / f"{shot}_labels.h5"

    def corpus_file(self, shot: int) -> Path:
        return self.corpus / f"{shot}_processed.h5"

    def mkdirs(self) -> None:
        for d in (self.features, self.labels, self.models, self.runs, self.validation):
            d.mkdir(parents=True, exist_ok=True)


def git_sha() -> str:
    """Short sha of the checkout that produced an artifact, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"
```

- [ ] **Step 5: Make `import labelmaker` work**

`src/labelmaker` is a third top-level package in a `src/` layout, and hatchling currently autodiscovers only `src/faith` (there is no `packages` list). Add after `[tool.hatch.version]` (pyproject.toml line 48):

```toml
[tool.hatch.build.targets.wheel]
# Three top-level packages live under src/. Without this list hatchling
# autodiscovers only the one matching the project name (faith), so
# `import labelmaker` fails in every pixi env.
packages = ["src/faith", "src/tokamak_foundation_model", "src/labelmaker"]
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
.pixi/envs/default/bin/python -m pytest tests/labelmaker/test_config.py -q
```
Expected: `4 passed`. The editable install exposes `src/` on `sys.path`, so the tests pass before the env is re-materialized; Step 7 is what makes a *fresh* env carry `labelmaker`.

- [ ] **Step 7: Add the `labelmaker` pixi feature and environment**

Append to the `[tool.pixi.feature.*]` block (after the `fdp` feature, pyproject.toml line 84) :

```toml
[tool.pixi.feature.labelmaker]
platforms = ["linux-64"]

[tool.pixi.feature.labelmaker.dependencies]
# labels_index.parquet. pandas 3 can write parquet only through pyarrow,
# and the default env does not have it.
pyarrow = ">=17,<22"
# models/registry.py parses model-card front matter with yaml. It is present
# in this env today only as a transitive dependency of hydra-core ->
# omegaconf, and hydra-core is in [project] for IGNITE's sake rather than
# labelmaker's - so an unrelated change to IGNITE's dependencies would break
# every card read. Declare what we import.
pyyaml = ">=6,<7"

[tool.pixi.feature.labelmaker.pypi-dependencies]
# The workspace installs `faith` editable into every environment, which
# drags torch in. Labelmaker never uses a GPU (its heaviest model is 12k
# parameters evaluated in numpy), so take the CPU wheels and save ~3 GB
# in an environment that would otherwise duplicate cu124.
torch = { version = ">=2.5.1", index = "https://download.pytorch.org/whl/cpu" }
torchvision = { version = ">=0.20.1", index = "https://download.pytorch.org/whl/cpu" }
```

and in `[tool.pixi.environments]` (line 119-122) add:

```toml
labelmaker = ["labelmaker", "fdp"]
```

Then materialize it:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi install -e labelmaker 2>&1 | tail -5
pixi run -e labelmaker python -c "import labelmaker, h5py, numpy, toksearch, pyarrow; print(labelmaker.__version__, numpy.__version__)"
```
Expected: `0.1.0 1.26.4` (the `fdp` feature's toksearch pins numpy `<2`; that is fine, labelmaker uses no numpy-2-only API).

If the solve fails on a `toksearch` / `torch` conflict, drop the two `pypi-dependencies` lines and re-run — the CPU-wheel override is a size optimization, not a requirement. Record whichever happened in the task's commit message.

- [ ] **Step 8: Run the whole new test directory in the new environment**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker -q
```
Expected: `4 passed`.

- [ ] **Step 9: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/__init__.py src/labelmaker/config.py \
        tests/labelmaker/__init__.py tests/labelmaker/test_config.py \
        pyproject.toml pixi.lock docs/superpowers/specs/2026-09-03-labelmaker-design.md \
        docs/superpowers/plans/2026-09-03-labelmaker-phase1.md
git commit -m "labelmaker: package skeleton, path config and pixi environment"
```

---

### Task 2: Time-base helpers

**Files:**
- Create: `src/labelmaker/timebase.py`
- Test: `tests/labelmaker/test_timebase.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `sample_rate(x: np.ndarray) -> float`; `index_at(x: np.ndarray, t: np.ndarray | float) -> np.ndarray` (int64, clamped); `sample_at(x, y, t, *, max_gap: float | None = None) -> np.ndarray` (y may be `(T,)` or `(C, T)`; samples the last axis, NaN where the nearest sample is further than `max_gap` seconds); `window_mean(x, y, t, width: float) -> np.ndarray`; `decimate_to_step(x, y, step: float) -> tuple[np.ndarray, np.ndarray]`; `MS_PER_S = 1000.0`.

- [ ] **Step 1: Write the failing test**

`tests/labelmaker/test_timebase.py`:

```python
"""The time-base conventions labelmaker shares with IGNITE."""
import warnings

import numpy as np
import pytest

from labelmaker.timebase import (
    decimate_to_step,
    index_at,
    sample_at,
    sample_rate,
    window_mean,
)


def test_sample_rate_is_exact_on_float32_records():
    # A real actuator group: 10 kHz, starting 10 s before the shot, stored
    # float32. Quantization puts up to 0.82% of a step of error on any single
    # diff - the figure train_dynamics.py:190 quotes, reproduced here as
    # 8.18e-7 s. See train_dynamics.py:189-191.
    x64 = np.arange(-10.0, 1.0, 1e-4)
    x32 = x64.astype(np.float32)
    fs = sample_rate(x32)
    assert abs(fs - 10_000.0) / 10_000.0 < 1e-6          # measured 1.5e-9
    worst_step = np.abs(np.diff(x32.astype(np.float64)) - 1e-4).max()
    assert worst_step > 0.005 * 1e-4                     # measured 0.82% of a step

    # The median is a *robust* estimator, so a rate taken from diff() is only
    # ~1.7e-4 off - small enough to look fine and still wrong. The damage is
    # cumulative: that rate misplaces a sample 18 deep into an 11 s record,
    # which is precisely what index_at must never do.
    naive = 1.0 / float(np.median(np.diff(x32.astype(np.float64))))
    assert abs(naive - 10_000.0) / 10_000.0 < 1e-3       # NOT a big rate error
    elapsed = 0.9 - float(x32[0])
    assert abs(round(elapsed * naive) - round(elapsed * fs)) >= 10
    assert index_at(x32, 0.9)[0] == 109_000              # span-based is exact


def test_sample_rate_rejects_degenerate_axes():
    with pytest.raises(ValueError):
        sample_rate(np.array([1.0]))
    with pytest.raises(ValueError):
        sample_rate(np.array([1.0, 1.0, 1.0]))


def test_index_at_counts_from_record_start_and_clamps():
    x = np.arange(-10.0, 1.0, 1e-4)  # 110000 samples
    assert index_at(x, 0.0)[0] == 100_000
    assert index_at(x, -10.0)[0] == 0
    assert index_at(x, -50.0)[0] == 0        # clamped, not wrapped
    assert index_at(x, 99.0)[0] == x.size - 1
    np.testing.assert_array_equal(index_at(x, [0.0, 0.5]), [100_000, 105_000])


def test_sample_at_handles_1d_and_2d_and_gaps():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y1 = np.array([10.0, 11.0, 12.0, 13.0])
    y2 = np.stack([y1, y1 * 2])
    np.testing.assert_allclose(sample_at(x, y1, [0.1, 2.9]), [10.0, 13.0])
    np.testing.assert_allclose(sample_at(x, y2, [0.1])[:, 0], [10.0, 20.0])
    out = sample_at(x, y1, [10.0], max_gap=0.5)
    assert np.isnan(out[0])


def test_window_mean_is_silent_when_a_window_holds_no_finite_sample():
    # np.nanmean returns the right value here and raises "Mean of empty
    # slice" through the warnings module, which np.errstate does NOT catch.
    # Nine later tasks import this module and real channels have dropout
    # stretches, so that warning would become permanent noise.
    x = np.array([0.0, 0.1, 0.2])
    y = np.array([np.nan, np.nan, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = window_mean(x, y, [0.0], 0.15)
    assert np.isnan(got[0])


def test_window_mean_averages_the_following_window():
    x = np.arange(0.0, 1.0, 0.1)
    y = np.arange(10.0)
    # window [0.0, 0.5) covers samples 0..4 -> mean 2.0
    np.testing.assert_allclose(window_mean(x, y, [0.0], 0.5), [2.0])
    np.testing.assert_allclose(window_mean(x, y, [0.5], 0.5), [7.0])
    assert np.isnan(window_mean(x, y, [5.0], 0.5)[0])  # window outside record


def test_decimate_to_step_bin_averages_and_marks_empty_bins():
    x = np.array([0.0, 0.001, 0.002, 0.05, 0.051])
    y = np.array([1.0, 3.0, 5.0, 10.0, 20.0])
    xg, yg = decimate_to_step(x, y, 0.01)
    assert xg[0] == 0.0 and yg.shape == (1, 6)
    np.testing.assert_allclose(yg[0, 0], 3.0)     # (1+3+5)/3
    np.testing.assert_allclose(yg[0, 5], 15.0)    # (10+20)/2
    assert np.isnan(yg[0, 1:5]).all()             # nothing sampled there


def test_decimate_to_step_ignores_nans_within_a_bin():
    x = np.array([0.0, 0.001, 0.002])
    y = np.array([1.0, np.nan, 5.0])
    _, yg = decimate_to_step(x, y, 0.01)
    np.testing.assert_allclose(yg[0, 0], 3.0)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_timebase.py -q
```
Expected: `ModuleNotFoundError: No module named 'labelmaker.timebase'`.

- [ ] **Step 3: Write the implementation**

`src/labelmaker/timebase.py`:

```python
"""Sampling a per-shot record at chosen times.

These are IGNITE's conventions, copied rather than imported: labelmaker
depends on no model code (see ignite/gate.py's reuse note). The original
and the reason both halves matter are in
src/tokamak_foundation_model/ignite/train_dynamics.py:177-192.
"""
from __future__ import annotations

import numpy as np

MS_PER_S = 1000.0


def sample_rate(x: np.ndarray) -> float:
    """Samples per second, from the record span.

    Not `1/median(diff(x))`: `xdata` is float32 in the corpus, which loses
    the sample step to cancellation and drifts by up to 0.8% over a shot.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError(f"need at least two samples, got {x.size}")
    span = float(x[-1] - x[0])
    if span <= 0.0:
        raise ValueError(f"non-increasing time axis: {x[0]} .. {x[-1]}")
    return (x.size - 1) / span


def index_at(x: np.ndarray, t) -> np.ndarray:
    """Nearest sample index for each requested time, clamped into the record.

    Counted from the record start, not from zero: actuator groups begin at
    negative times, where an unclamped index would wrap to the array tail.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    idx = np.round((t - x[0]) * sample_rate(x)).astype(np.int64)
    return np.clip(idx, 0, x.size - 1)


def sample_at(x: np.ndarray, y: np.ndarray, t, *, max_gap: float | None = None):
    """`y` at each time in `t` (nearest sample), last axis sampled.

    `max_gap` guards the clamp: a time further than `max_gap` seconds from
    the nearest sample yields NaN instead of the edge value.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    idx = index_at(x, t)
    out = y[..., idx].astype(np.float64, copy=True)
    if max_gap is not None:
        far = np.abs(x[idx] - t) > max_gap
        out[..., far] = np.nan
    return out


def window_mean(x: np.ndarray, y: np.ndarray, t, width: float) -> np.ndarray:
    """Mean of `y` over each half-open window `[t, t + width)`.

    NaN where the window holds no finite sample, so a missing stretch never
    silently becomes an edge value.

    The mean is accumulated by hand rather than with `np.nanmean`, which
    raises "Mean of empty slice" through the `warnings` module on an
    all-NaN window - `np.errstate` does not catch that, and real channels
    have dropout stretches. Same approach as `decimate_to_step` below.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[None, :]
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    out = np.full((y.shape[0], t.size), np.nan, dtype=np.float64)
    lo = np.searchsorted(x, t, side="left")
    hi = np.searchsorted(x, t + width, side="left")
    for j, (a, b) in enumerate(zip(lo, hi)):
        if b <= a:
            continue
        seg = y[:, a:b]
        good = np.isfinite(seg)
        counts = good.sum(axis=1)
        totals = np.where(good, seg, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, j] = np.where(counts > 0, totals / counts, np.nan)
    return out[0] if out.shape[0] == 1 else out


def decimate_to_step(x: np.ndarray, y: np.ndarray, step: float):
    """Bin-average `y` onto a uniform grid of the given step.

    Used to bound the size of high-rate PTDATA before it is stored: a 10 kHz
    channel over 6 s is 60k samples per shot, and nothing downstream asks
    for more than 1 ms resolution. Empty bins are NaN; NaNs inside a bin are
    ignored.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[None, :]
    if x.size < 2:
        raise ValueError(f"need at least two samples, got {x.size}")
    n = int(np.floor((x[-1] - x[0]) / step)) + 1
    bins = np.clip(((x - x[0]) / step).astype(np.int64), 0, n - 1)
    out = np.empty((y.shape[0], n), dtype=np.float64)
    for c in range(y.shape[0]):
        good = np.isfinite(y[c])
        totals = np.bincount(bins, weights=np.where(good, y[c], 0.0), minlength=n)
        counts = np.bincount(bins, weights=good.astype(np.float64), minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[c] = np.where(counts > 0, totals / counts, np.nan)
    return x[0] + step * np.arange(n, dtype=np.float64), out
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_timebase.py -q
```
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/timebase.py tests/labelmaker/test_timebase.py
git commit -m "labelmaker: span-based time-base helpers"
```

---

### Task 3: Feature namespace and feature store

**Files:**
- Create: `src/labelmaker/features/__init__.py`, `src/labelmaker/features/namespace.py`, `src/labelmaker/features/store.py`
- Test: `tests/labelmaker/test_namespace.py`, `tests/labelmaker/test_feature_store.py`

**Interfaces:**
- Consumes: `labelmaker.config.Paths`, `labelmaker.__version__`, `labelmaker.config.git_sha`.
- Produces:
  - `namespace.RHO_GRID: np.ndarray` (33 points, `linspace(0, 1, 33)`, float64), `namespace.GRID_S: np.ndarray` (240 points, `0.025 * arange(240)`), `namespace.STEP_S = 0.025`
  - `namespace.FeatureSpec` frozen dataclass, fields `name: str`, `kind: str`, `units: str`, `sources: tuple[str, ...]`, `locators: tuple[str, ...]`, `step: float = 0.0`, `notes: str = ""`, method `locator_for(source: str) -> str`
  - `namespace.FEATURES: tuple[FeatureSpec, ...]`, `namespace.by_name(name) -> FeatureSpec`, `namespace.by_source(source) -> tuple[FeatureSpec, ...]`, `namespace.KINDS`, `namespace.SOURCES`
  - `store.FeatureArray` frozen dataclass, fields `x: np.ndarray` (T,) float64 seconds, `y: np.ndarray` (C, T) float64, `attrs: dict[str, str]`
  - `store.write_features(path, shot, arrays: dict[str, FeatureArray], missing: dict[str, str], *, merge: bool = True) -> None`
  - `store.read_feature(path, name) -> FeatureArray`, `store.present(path) -> set[str]`, `store.missing_names(path) -> dict[str, str]`, `store.is_complete(path, names) -> bool`

- [ ] **Step 1: Write the failing tests**

`tests/labelmaker/test_namespace.py`:

```python
"""The canonical feature namespace is the package's contract with models."""
import numpy as np
import pytest

from labelmaker.features import namespace as ns


def test_grids_match_the_upstream_reference_file():
    # test_shots.h5 stores spatial_coordinates = 0, 1/32, ..., 1 and
    # times = 0, 25, ..., 5975 ms. The archive store is on the same grid.
    assert ns.RHO_GRID.shape == (33,)
    assert ns.RHO_GRID[0] == 0.0 and ns.RHO_GRID[-1] == 1.0
    np.testing.assert_allclose(np.diff(ns.RHO_GRID), 1.0 / 32.0)
    assert ns.GRID_S.shape == (240,)
    np.testing.assert_allclose(ns.GRID_S[[0, 1, -1]], [0.0, 0.025, 5.975])
    assert ns.STEP_S == 0.025


def test_specs_are_unique_and_well_formed():
    names = [f.name for f in ns.FEATURES]
    assert len(names) == len(set(names))
    for f in ns.FEATURES:
        assert f.name == f.name.lower()
        assert f.kind in ns.KINDS
        assert f.sources, f"{f.name} has no source"
        assert len(f.sources) == len(f.locators)
        assert set(f.sources) <= set(ns.SOURCES)
        assert len(set(f.sources)) == len(f.sources)


def test_every_feature_the_phase1_model_needs_exists():
    required = {
        "bt", "ip", "pinj_total", "tinj_total", "ech_power_total", "ech_rho",
        "r0", "kappa", "tritop", "tribot", "gapin", "betan",
        "qpsi", "pres", "ne_zipfit", "te_zipfit", "rot_zipfit",
    }
    assert required <= {f.name for f in ns.FEATURES}


def test_profiles_and_scalars_are_labelled_correctly():
    for name in ("qpsi", "pres", "ne_zipfit", "te_zipfit", "rot_zipfit"):
        assert ns.by_name(name).kind == "profile"
    for name in ("ip", "bt", "kappa", "ech_rho"):
        assert ns.by_name(name).kind == "scalar"


def test_lookup_helpers_and_locators():
    assert ns.by_name("ip").sources[0] == "archive"
    assert ns.by_name("pres").locator_for("archive") == "pres_EFIT01"
    assert ns.by_name("pinj_total").locator_for("corpus") == "pinj"
    assert all("archive" in f.sources for f in ns.by_source("archive"))
    assert {f.name for f in ns.by_source("corpus")} == {
        "pinj_total", "tinj_total", "ech_power_total"
    }
    assert "ip" in {f.name for f in ns.by_source("fdp")}
    assert "ech_rho" not in {f.name for f in ns.by_source("fdp")}
    with pytest.raises(KeyError):
        ns.by_name("no_such_feature")
    # ech_rho is the only Phase 1 feature with no second source, so it is the
    # only one whose locator_for("fdp") can raise.
    with pytest.raises(KeyError):
        ns.by_name("ech_rho").locator_for("fdp")


def test_source_preference_and_the_one_archive_only_feature():
    # The archive store comes first everywhere: for twelve features it is
    # bit-identical to what the Phase 1 model was trained on. ZIPFIT is
    # reachable through fdp too, which is what lets the pipeline leave the
    # archive's 21% of the corpus behind.
    for f in ns.FEATURES:
        assert f.sources[0] == "archive" or "archive" not in f.sources
    for name in ("ne_zipfit", "te_zipfit", "rot_zipfit"):
        assert ns.by_name(name).sources == ("archive", "fdp")
    # ech_rho is the one feature with no second source: TORBEAM deposition
    # locations exist only as the archived column.
    assert ns.by_name("ech_rho").sources == ("archive",)
```

`tests/labelmaker/test_feature_store.py`:

```python
"""Feature files round-trip, merge, and record what was missing."""
import h5py
import numpy as np
import pytest

from labelmaker.features.store import (
    FeatureArray,
    is_complete,
    missing_names,
    present,
    read_feature,
    write_features,
)


def _scalar(n=10):
    x = 0.001 * np.arange(n)
    return FeatureArray(
        x=x, y=np.arange(n, dtype=float)[None, :],
        attrs={"resolver": "corpus"},
    )


def _profile(n=4):
    x = 0.025 * np.arange(n)
    y = np.tile(np.linspace(1.0, 0.0, 33)[:, None], (1, n))
    return FeatureArray(x=x, y=y, attrs={"resolver": "archive"})


def test_round_trip_scalar_and_profile(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar(), "ne_zipfit": _profile()}, {})
    got = read_feature(p, "ip")
    np.testing.assert_allclose(got.x, _scalar().x)
    np.testing.assert_allclose(got.y, _scalar().y)
    assert got.attrs["resolver"] == "corpus"
    assert got.attrs["units"] == "A"          # filled in from the namespace
    assert read_feature(p, "ne_zipfit").y.shape == (33, 4)
    assert present(p) == {"ip", "ne_zipfit"}


def test_stored_in_corpus_layout_with_provenance(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {})
    with h5py.File(p, "r") as f:
        assert f["ip"]["xdata"].dtype == np.float64
        assert f["ip"]["ydata"].dtype == np.float32
        assert f["ip"]["ydata"].shape == (1, 10)
        assert f.attrs["shot"] == 190000
        assert f.attrs["labelmaker_version"] and f.attrs["git_sha"]
        assert f["ip"].attrs["complete"] == 1
        assert f["ip"].attrs["fetched_at"]


def test_profile_groups_carry_the_rho_grid(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ne_zipfit": _profile()}, {})
    with h5py.File(p, "r") as f:
        assert f["ne_zipfit"]["rho"].shape == (33,)


def test_missing_is_recorded_not_raised(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {}, {"bt": "KeyError", "pres": "TimeoutError"})
    assert missing_names(p) == {"bt": "KeyError", "pres": "TimeoutError"}
    assert present(p) == set()


def test_merge_keeps_earlier_groups_and_clears_resolved_misses(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {"bt": "KeyError"})
    write_features(p, 190000, {"bt": _scalar()}, {})
    assert present(p) == {"ip", "bt"}
    assert missing_names(p) == {}


def test_merge_false_replaces_the_file(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {})
    write_features(p, 190000, {"bt": _scalar()}, {}, merge=False)
    assert present(p) == {"bt"}


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {})
    assert list(tmp_path.iterdir()) == [p]


def test_is_complete_needs_every_name_present_or_missing(tmp_path):
    p = tmp_path / "190000_features.h5"
    assert not is_complete(p, ["ip"])          # no file yet
    write_features(p, 190000, {"ip": _scalar()}, {"bt": "KeyError"})
    assert is_complete(p, ["ip"])
    assert is_complete(p, ["ip", "bt"])        # a recorded miss counts
    assert not is_complete(p, ["ip", "pres"])


def test_mismatched_shapes_are_rejected_at_construction():
    with pytest.raises(ValueError):
        FeatureArray(x=np.zeros(3), y=np.zeros((1, 4)))
    with pytest.raises(ValueError):
        FeatureArray(x=np.zeros(3), y=np.zeros(3))     # must be (C, T)


def test_a_one_sample_feature_is_demoted_to_a_miss(tmp_path):
    # The corpus layout reads ydata.shape[-1] < 2 as "signal absent", so a
    # resolved one-sample group would be silently misread downstream. It is
    # recorded as a miss instead - and NOT raised on, which would cost this
    # shot the other feature resolved in the same call.
    p = tmp_path / "190000_features.h5"
    one = FeatureArray(x=np.zeros(1), y=np.zeros((1, 1)), attrs={"resolver": "corpus"})
    write_features(p, 190000, {"ip": one, "bt": _scalar()}, {})
    assert present(p) == {"bt"}
    assert "OneSampleAmbiguous" in missing_names(p)["ip"]


def test_write_features_leaves_the_callers_dicts_alone(tmp_path):
    p = tmp_path / "190000_features.h5"
    arrays = {"ip": FeatureArray(x=np.zeros(1), y=np.zeros((1, 1)))}
    missing: dict[str, str] = {}
    write_features(p, 190000, arrays, missing)
    assert set(arrays) == {"ip"} and missing == {}


def test_read_feature_raises_for_absent_group(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {})
    with pytest.raises(KeyError):
        read_feature(p, "bt")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_namespace.py tests/labelmaker/test_feature_store.py -q
```
Expected: `ModuleNotFoundError: No module named 'labelmaker.features'`.

- [ ] **Step 3: Write `features/__init__.py` and `features/namespace.py`**

`src/labelmaker/features/__init__.py`:

```python
"""Resolving canonical physics features for a shot, one module per source."""
```

`src/labelmaker/features/namespace.py`:

```python
"""The canonical feature namespace.

One name per physical quantity. A model never names an MDSplus node, a corpus
group or an archive column; it names a canonical feature, so every
substitution is written down once, here, and shows up in the model card.

Most quantities are available from more than one place with different shot
coverage, so `sources` is an ordered preference list and the group attribute
`resolver` records which source actually produced a feature for a given
shot. Provenance is therefore per shot, which a static name could not be.

`locators` is parallel to `sources`: the archive column name, the corpus
group name, the MDSplus node. Registry order is not meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Radial grid every profile feature is evaluated on: the
#: `spatial_coordinates` dataset of
#: /projects/EKOLEMEN/simple_ae_predictor/test/test_shots.h5.
RHO_GRID = np.linspace(0.0, 1.0, 33)

#: The 25 ms output grid, in seconds. Identical to that file's `times`
#: dataset (0..5975 ms) and to the archive store's 240 rows.
STEP_S = 0.025
GRID_S = STEP_S * np.arange(240, dtype=np.float64)

KINDS = ("scalar", "profile")
SOURCES = ("archive", "corpus", "fdp")


@dataclass(frozen=True)
class FeatureSpec:
    """What a canonical feature is and where it can come from."""

    name: str
    kind: str
    units: str
    sources: tuple[str, ...]
    locators: tuple[str, ...]
    step: float = 0.0  # storage step in seconds; 0.0 keeps the native rate
    notes: str = ""

    def locator_for(self, source: str) -> str:
        """This feature's name in the given source, or KeyError."""
        for src, loc in zip(self.sources, self.locators):
            if src == source:
                return loc
        raise KeyError(f"{self.name} has no {source} source")


# The `archive` locators are columns of
# /projects/EKOLEMEN/profile_predictor/DATA/new_h5_data/example_191450_183224.h5
# (5,000 shots, 183224..191450, 240 rows at 25 ms), which is the store the
# Phase 1 model's training features were built from. `fdp` locators are
# verified against that store in Task 11 Step 1 before any bulk fetch.
FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="ip", kind="scalar", units="A",
        sources=("archive", "fdp"),
        locators=("ip", "ip"),
        step=0.001,
        notes="archive column is bit-identical to the model's training input",
    ),
    FeatureSpec(
        name="bt", kind="scalar", units="T",
        sources=("archive", "fdp"),
        locators=("bt", "bt"),
        step=0.001,
        notes="archive column is bit-identical to the model's training input",
    ),
    FeatureSpec(
        name="pinj_total", kind="scalar", units="kW",
        sources=("archive", "corpus"),
        locators=("pinj", "pinj"),
        step=0.001,
        notes="corpus holds 8 beams in W; sum and divide by 1000",
    ),
    FeatureSpec(
        name="tinj_total", kind="scalar", units="N m",
        sources=("archive", "corpus"),
        locators=("tinj", "tinj"),
        step=0.001,
        notes="corpus holds 8 beams already in N m; sum only",
    ),
    FeatureSpec(
        name="ech_power_total", kind="scalar", units="W",
        sources=("archive", "corpus"),
        locators=("ech_pwr", "ech_power"),
        step=0.001,
        notes="corpus holds 12 gyrotrons and can contain NaN channels; "
              "nansum, then NaN or negative -> 0 (upstream rule). Units "
              "confirmed against the archive in Task 10 Step 5",
    ),
    FeatureSpec(
        name="r0", kind="scalar", units="m",
        sources=("archive", "fdp"),
        locators=("rmaxis_EFIT01", r"\efit01::top.results.geqdsk:rmaxis"),
        notes="stands in for R0_EFITRT1; measured 8.8e-3 median relative "
              "difference on shot 185945",
    ),
    FeatureSpec(
        name="kappa", kind="scalar", units="",
        sources=("archive", "fdp"),
        locators=("kappa_EFIT01", r"\efit01::top.results.aeqdsk:kappa"),
        notes="stands in for kappa_EFITRT1; measured 3.1e-3",
    ),
    FeatureSpec(
        name="tritop", kind="scalar", units="",
        sources=("archive", "fdp"),
        locators=("tritop_EFIT01", r"\efit01::top.results.aeqdsk:tritop"),
        notes="bit-identical to the training input",
    ),
    FeatureSpec(
        name="tribot", kind="scalar", units="",
        sources=("archive", "fdp"),
        locators=("tribot_EFIT01", r"\efit01::top.results.aeqdsk:tribot"),
        notes="bit-identical to the training input",
    ),
    FeatureSpec(
        name="gapin", kind="scalar", units="m",
        sources=("archive", "fdp"),
        locators=("gapin_EFIT01", r"\efit01::top.results.aeqdsk:gapin"),
        notes="bit-identical to the training input",
    ),
    FeatureSpec(
        name="betan", kind="scalar", units="",
        sources=("archive", "fdp"),
        locators=("betan_EFIT01", r"\efit01::top.results.aeqdsk:betan"),
        notes="not a model input; the model predicts it, and validation "
              "compares against it",
    ),
    FeatureSpec(
        name="qpsi", kind="profile", units="",
        sources=("archive", "fdp"),
        locators=("qpsi_EFIT01", r"\efit01::top.results.geqdsk:qpsi"),
        notes="stands in for qpsi_EFITRT1; measured 6.4e-2. The model "
              "consumes 1/qpsi, applied by the adapter, not here",
    ),
    FeatureSpec(
        name="pres", kind="profile", units="Pa",
        sources=("archive", "fdp"),
        locators=("pres_EFIT01", r"\efit01::top.results.geqdsk:pres"),
        notes="bit-identical to the training input",
    ),
    FeatureSpec(
        name="ne_zipfit", kind="profile", units="1e19 m^-3",
        sources=("archive", "fdp"),
        locators=("zipfit_edensfit_rho", r"\ZIPFIT01::TOP.PROFILES.EDENSFIT"),
        notes="stands in for thomson_density_mtanh_1d; measured 2.0e-1 "
              "median relative difference, correlation 0.984. Our own mtanh "
              "fit to raw Thomson is Phase 2",
    ),
    FeatureSpec(
        name="te_zipfit", kind="profile", units="keV",
        sources=("archive", "fdp"),
        locators=("zipfit_etempfit_rho", r"\ZIPFIT01::TOP.PROFILES.ETEMPFIT"),
        notes="stands in for thomson_temp_mtanh_1d; measured 1.8e-1, 0.991",
    ),
    FeatureSpec(
        name="rot_zipfit", kind="profile", units="krad/s",
        sources=("archive", "fdp"),
        locators=("zipfit_trotfit_rho", r"\ZIPFIT01::TOP.PROFILES.TROTFIT"),
        notes="stands in for cer_rot_csaps_1d; measured 1.7e-1, 0.980",
    ),
    FeatureSpec(
        name="ech_rho", kind="scalar", units="",
        sources=("archive",),
        locators=("EC.RHO_ECH",),
        notes="TORBEAM deposition location; 0 where unavailable, which the "
              "upstream training filter admitted (x0[:, 10] >= 0)",
    ),
)

_BY_NAME = {f.name: f for f in FEATURES}


def by_name(name: str) -> FeatureSpec:
    """The spec for a canonical name, or KeyError."""
    return _BY_NAME[name]


def by_source(source: str) -> tuple[FeatureSpec, ...]:
    """Every feature a single resolver can produce."""
    return tuple(f for f in FEATURES if source in f.sources)
```

- [ ] **Step 4: Write `features/store.py`**

```python
"""Reading and writing `<shot>_features.h5`.

Same layout as the corpus itself - one group per quantity, `xdata` seconds,
`ydata` (C, T) - so anything that can read a corpus file can read a feature
file. Files are small (a few MB), so a merge rewrites the whole file and
renames it into place; a killed run therefore never leaves a half-written
feature file behind.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

from .. import __version__
from ..config import git_sha
from . import namespace as ns

MISSING_ATTR = "missing"


@dataclass(frozen=True)
class FeatureArray:
    """One resolved feature: seconds on `x`, `(C, T)` values on `y`."""

    x: np.ndarray
    y: np.ndarray
    attrs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if np.asarray(self.y).ndim != 2:
            raise ValueError(f"ydata must be (C, T), got {np.shape(self.y)}")
        if np.shape(self.y)[-1] != np.shape(self.x)[-1]:
            raise ValueError(f"x {np.shape(self.x)} and y {np.shape(self.y)} disagree")


def _read_group(g) -> FeatureArray:
    """One stored group back into a FeatureArray, in float64."""
    return FeatureArray(
        x=np.asarray(g["xdata"], dtype=np.float64),
        y=np.asarray(g["ydata"], dtype=np.float64),
        attrs={k: str(v) for k, v in g.attrs.items()},
    )


def _load_all(path: Path) -> tuple[dict[str, FeatureArray], dict[str, str]]:
    if not Path(path).exists():
        return {}, {}
    with h5py.File(path, "r") as f:
        arrays = {name: _read_group(f[name]) for name in f}
        missing = json.loads(f.attrs.get(MISSING_ATTR, "{}"))
    return arrays, missing


def write_features(
    path,
    shot: int,
    arrays: dict[str, FeatureArray],
    missing: dict[str, str],
    *,
    merge: bool = True,
) -> None:
    """Write a feature file atomically.

    With `merge`, groups already in the file are kept and any name now
    resolved is dropped from the recorded misses, so a rerun that fetches
    one more feature does not throw away the previous ones.

    A feature carrying fewer than two samples is moved into `missing`
    instead of being stored; see the comment below. The caller's `arrays`
    and `missing` dicts are never mutated.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, missing = dict(arrays), dict(missing)
    if merge:
        kept, kept_missing = _load_all(path)
        kept.update(arrays)
        kept_missing.update(missing)
        for name in arrays:
            kept_missing.pop(name, None)
        arrays, missing = kept, kept_missing
    # A group with fewer than two samples is the corpus' "signal absent"
    # sentinel, and these files share the corpus layout - so a *resolved*
    # one-sample group would read as absent to any consumer applying the
    # corpus rule, which `catalog.available_groups` does. It is reachable:
    # `decimate_to_step` on a degenerate time axis returns one sample.
    #
    # Such a feature is demoted into `missing` rather than persisted, and
    # rather than raised on. Raising would cost the shot every other feature
    # resolved in the same call, against this pipeline's rule that one bad
    # feature never costs a shot the rest; `missing` is the channel built to
    # carry exactly this. Demotion also heals a file that somehow already
    # holds a short group, which a raise would have made unwritable forever.
    for name in [n for n, a in arrays.items() if a.y.shape[-1] < 2]:
        missing[name] = f"OneSampleAmbiguous({arrays[name].y.shape[-1]})"
        del arrays[name]
    now = datetime.now(UTC).isoformat(timespec="seconds")
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["shot"] = int(shot)
        f.attrs["labelmaker_version"] = __version__
        f.attrs["git_sha"] = git_sha()
        f.attrs["written_at"] = now
        f.attrs[MISSING_ATTR] = json.dumps(missing, sort_keys=True)
        for name, arr in sorted(arrays.items()):
            g = f.create_group(name)
            g.create_dataset("xdata", data=np.asarray(arr.x, dtype=np.float64))
            g.create_dataset("ydata", data=np.asarray(arr.y, dtype=np.float32))
            for k, v in arr.attrs.items():
                g.attrs[k] = v
            if "fetched_at" not in g.attrs:
                g.attrs["fetched_at"] = now
            g.attrs["complete"] = 1
            try:
                spec = ns.by_name(name)
            except KeyError:
                continue
            g.attrs["units"] = spec.units
            if spec.kind == "profile" and np.shape(arr.y)[0] == ns.RHO_GRID.size:
                g.create_dataset("rho", data=ns.RHO_GRID)
    tmp.replace(path)


def read_feature(path, name: str) -> FeatureArray:
    """One feature out of a feature file, or KeyError."""
    with h5py.File(path, "r") as f:
        if name not in f:
            raise KeyError(f"{name} not in {path}")
        return _read_group(f[name])


def present(path) -> set[str]:
    """Feature names stored in the file (empty if the file does not exist)."""
    if not Path(path).exists():
        return set()
    with h5py.File(path, "r") as f:
        return set(f.keys())


def missing_names(path) -> dict[str, str]:
    """Feature name -> cause, for everything a run tried and could not get."""
    if not Path(path).exists():
        return {}
    with h5py.File(path, "r") as f:
        return json.loads(f.attrs.get(MISSING_ATTR, "{}"))


def is_complete(path, names) -> bool:
    """True when every requested name is either stored or a recorded miss."""
    if not Path(path).exists():
        return False
    return set(names) <= (present(path) | set(missing_names(path)))
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_namespace.py tests/labelmaker/test_feature_store.py -q
```
Expected: `16 passed`.

- [ ] **Step 6: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/features/__init__.py src/labelmaker/features/namespace.py \
        src/labelmaker/features/store.py tests/labelmaker/test_namespace.py \
        tests/labelmaker/test_feature_store.py
git commit -m "labelmaker: canonical feature namespace and per-shot feature store"
```

---

### Task 4: Shot catalog and the proof-of-concept shot list

**Files:**
- Create: `src/labelmaker/catalog.py`
- Test: `tests/labelmaker/test_catalog.py`

**Interfaces:**
- Consumes: `labelmaker.config.Paths`.
- Produces: `catalog.TM_ARCHIVE: Path`; `corpus_shots(paths: Paths) -> list[int]`; `corpus_groups(path) -> dict[str, int]` (group name → `ydata.shape[-1]`, absent groups included with their length 1); `available_groups(path) -> set[str]` (only those with length > 1); `archive_shots(archive: Path = TM_ARCHIVE) -> np.ndarray`; `overlap_shots(paths: Paths, archive: Path = TM_ARCHIVE) -> list[int]`; `sample_shots(shots, n: int, seed: int) -> list[int]`; `write_shot_file(path, shots) -> None`; `read_shot_file(path) -> list[int]`.

- [ ] **Step 1: Write the failing test**

`tests/labelmaker/test_catalog.py`:

```python
"""Which shots exist, which are usable, and which 100 the PoC runs on."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from labelmaker import catalog
from labelmaker.config import Paths

CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
TM = catalog.TM_ARCHIVE


def _fake_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for shot, n_ech in ((190000, 24), (190001, 1), (190002, 24)):
        with h5py.File(corpus / f"{shot}_processed.h5", "w") as f:
            g = f.create_group("pinj")
            g.create_dataset("xdata", data=np.arange(10, dtype=np.float32))
            g.create_dataset("ydata", data=np.zeros((8, 10), dtype=np.float32))
            g = f.create_group("ech_power")
            g.create_dataset("xdata", data=np.arange(n_ech, dtype=np.float32))
            g.create_dataset("ydata", data=np.zeros((12, n_ech), dtype=np.float32))
    (corpus / "not_a_shot.h5").touch()
    return Paths(root=tmp_path / "out", corpus=corpus)


def test_corpus_shots_are_sorted_ints_from_filenames(tmp_path):
    paths = _fake_corpus(tmp_path)
    assert catalog.corpus_shots(paths) == [190000, 190001, 190002]


def test_corpus_groups_reports_lengths_and_absence(tmp_path):
    paths = _fake_corpus(tmp_path)
    groups = catalog.corpus_groups(paths.corpus_file(190001))
    assert groups == {"pinj": 10, "ech_power": 1}
    assert catalog.available_groups(paths.corpus_file(190001)) == {"pinj"}
    assert catalog.available_groups(paths.corpus_file(190000)) == {"pinj", "ech_power"}


def test_sample_shots_is_deterministic_and_bounded():
    shots = list(range(1000, 1100))
    a = catalog.sample_shots(shots, 10, seed=0)
    b = catalog.sample_shots(shots, 10, seed=0)
    c = catalog.sample_shots(shots, 10, seed=1)
    assert a == b and a != c
    assert len(a) == 10 and set(a) <= set(shots) and a == sorted(a)
    assert catalog.sample_shots(shots, 500, seed=0) == shots  # n > len is all


def test_shot_file_round_trip(tmp_path):
    p = tmp_path / "shots.txt"
    catalog.write_shot_file(p, [190002, 190000])
    assert catalog.read_shot_file(p) == [190000, 190002]
    p.write_text("# a comment\n190005\n\n190004\n")
    assert catalog.read_shot_file(p) == [190004, 190005]


@pytest.mark.skipif(not CORPUS.exists(), reason=f"corpus not available: {CORPUS}")
def test_real_corpus_is_enumerated():
    shots = catalog.corpus_shots(Paths())
    assert len(shots) > 16_000
    assert all(100_000 < s < 300_000 for s in shots)


@pytest.mark.skipif(
    not (CORPUS.exists() and TM.exists()), reason="corpus or tm archive missing"
)
def test_overlap_with_the_tearing_archive_is_the_poc_pool():
    overlap = catalog.overlap_shots(Paths())
    assert len(overlap) > 1_000       # measured 1,503 on 2026-09-03
    assert min(overlap) >= 185_000
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_catalog.py -q
```
Expected: `ModuleNotFoundError: No module named 'labelmaker.catalog'`.

- [ ] **Step 3: Write the implementation**

`src/labelmaker/catalog.py`:

```python
"""Which shots exist and which of them a run should touch.

The proof-of-concept pool is the intersection of the corpus with the
tearing-mode training archive: only there can a reconstructed feature row be
compared against the row the model was actually trained on, which is what
makes Task 15 and Task 16 possible.
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from .config import Paths

#: Filtered training arrays of the Phase 1 model: x0 (11 scalars), x1
#: (33 x 5 profiles), y (betan, tm_label), z (shot per row).
TM_ARCHIVE = Path("/projects/EKOLEMEN/tm_data")

_SHOT_FILE = re.compile(r"^(\d+)_processed\.h5$")


def corpus_shots(paths: Paths) -> list[int]:
    """Every shot with a corpus file, sorted."""
    out = []
    for p in paths.corpus.iterdir():
        m = _SHOT_FILE.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def corpus_groups(path) -> dict[str, int]:
    """Group name -> number of time samples, for one corpus file.

    A length of 1 is the corpus' "signal absent" sentinel (a `(C, 1)`
    placeholder), documented in
    src/tokamak_foundation_model/data/multi_file_dataset.py:845-861.
    """
    with h5py.File(path, "r") as f:
        return {
            name: int(f[name]["ydata"].shape[-1])
            for name in f
            if "ydata" in f[name]
        }


def available_groups(path) -> set[str]:
    """Corpus groups that actually carry a series."""
    return {name for name, n in corpus_groups(path).items() if n > 1}


def archive_shots(archive: Path = TM_ARCHIVE) -> np.ndarray:
    """Sorted unique shots present in the tearing-mode training arrays."""
    z = np.load(archive / "z.npy")
    return np.unique(z.astype(np.int64))


def overlap_shots(paths: Paths, archive: Path = TM_ARCHIVE) -> list[int]:
    """Shots in both the corpus and the training archive."""
    return sorted(set(corpus_shots(paths)) & set(archive_shots(archive).tolist()))


def sample_shots(shots, n: int, seed: int) -> list[int]:
    """A reproducible sample; all of `shots` when `n` is larger."""
    shots = sorted(int(s) for s in shots)
    if n >= len(shots):
        return shots
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(shots), size=n, replace=False)
    return sorted(shots[i] for i in idx)


def write_shot_file(path, shots) -> None:
    """One shot per line, sorted - the unit of a reproducible run."""
    Path(path).write_text("\n".join(str(s) for s in sorted(shots)) + "\n")


def read_shot_file(path) -> list[int]:
    """Shots from a file, ignoring blank lines and `#` comments."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(int(line))
    return sorted(out)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_catalog.py -q
```
Expected: `6 passed` (the two real-data tests run on stellar; they skip elsewhere).

- [ ] **Step 5: Freeze the proof-of-concept shot list**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
from labelmaker.catalog import overlap_shots, sample_shots, write_shot_file
from labelmaker.config import Paths

paths = Paths.from_env()
pool = overlap_shots(paths)
shots = sample_shots(pool, 100, seed=20260903)
paths.mkdirs()
write_shot_file(paths.root / "poc_shots.txt", shots)
print(f"pool={len(pool)} sampled={len(shots)} first={shots[:3]} last={shots[-3:]}")
PY
```
Expected: `pool=1503 sampled=100 ...` (pool size may differ if either archive changed; record the number printed in the commit message).

- [ ] **Step 6: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/catalog.py tests/labelmaker/test_catalog.py
git commit -m "labelmaker: shot catalog and seeded proof-of-concept shot list"
```

---

### Task 5: Numpy evaluator for Keras-2 legacy HDF5 models

**Files:**
- Create: `src/labelmaker/models/__init__.py`, `src/labelmaker/models/runners/__init__.py`, `src/labelmaker/models/runners/keras_h5.py`
- Test: `tests/labelmaker/test_keras_h5.py`

**Why this exists:** the roster's Keras artifacts were written by Keras 2.8; conda-forge's only `tensorflow-cpu` (2.21.0) is a py312 build and this repo pins `python <3.12`, so TensorFlow cannot be installed here at all. The `_4c.h5` graphs contain nothing but `InputLayer / BatchNormalization / Conv1D / MaxPooling1D / Flatten / Dense / Concatenate / Dropout`, which evaluate exactly in numpy. Task 14 proves the two paths agree to 1e-5 once, in a throwaway TensorFlow env, and freezes the result as a golden file.

**Interfaces:**
- Consumes: nothing (numpy + h5py + stdlib only).
- Produces: `keras_h5.UnsupportedLayer(RuntimeError)`; `keras_h5.KerasGraph` frozen dataclass with fields `config: dict`, `weights: dict[str, dict[str, np.ndarray]]`, `input_names: tuple[str, ...]`, `output_names: tuple[str, ...]`, `input_shapes: dict[str, tuple[int | None, ...]]`, and `__call__(inputs: dict[str, np.ndarray] | list[np.ndarray] | np.ndarray) -> list[np.ndarray]`; `keras_h5.load_graph(path) -> KerasGraph`; `keras_h5.load_ensemble(paths) -> tuple[KerasGraph, ...]`; `keras_h5.predict_members(graphs, inputs) -> np.ndarray` of shape `(n_members, n_rows, n_out)`.

- [ ] **Step 1: Write the failing test**

`tests/labelmaker/test_keras_h5.py`:

```python
"""The numpy evaluator reproduces Keras semantics layer by layer.

Every expectation here is hand-computed from the layer definition, so the
tests are an independent oracle rather than a recording of our own output.
Equality with TensorFlow itself is Task 14.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from labelmaker.models.runners.keras_h5 import (
    UnsupportedLayer,
    load_ensemble,
    load_graph,
    predict_members,
)

TM_UPSTREAM = Path(
    "/projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w"
)


def _layer(cls, name, inbound, **cfg):
    node = [[[i, 0, 0, {}] for i in inbound]] if inbound else []
    return {"class_name": cls, "config": {"name": name, **cfg}, "inbound_nodes": node}


def _write_legacy_h5(path, layers, input_layers, output_layers, weights):
    """A Keras-2.8-shaped legacy HDF5 file, written without TensorFlow."""
    cfg = {
        "class_name": "Functional",
        "config": {
            "name": "m",
            "layers": layers,
            "input_layers": input_layers,
            "output_layers": output_layers,
        },
    }
    with h5py.File(path, "w") as f:
        f.attrs["keras_version"] = "2.8.0"
        f.attrs["backend"] = "tensorflow"
        f.attrs["model_config"] = json.dumps(cfg)
        mw = f.create_group("model_weights")
        for lname in [lay["config"]["name"] for lay in layers]:
            g = mw.create_group(lname)
            names = []
            for wname, arr in weights.get(lname, {}).items():
                full = f"{lname}/{wname}:0"
                g.create_dataset(full, data=np.asarray(arr, dtype=np.float32))
                names.append(full.encode())
            g.attrs["weight_names"] = names


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_dense_with_sigmoid_matches_hand_computation(tmp_path):
    p = tmp_path / "m.h5"
    # Every value here is an exact binary fraction, so the float32 round trip
    # through the HDF5 is lossless and rtol=1e-12 tests the arithmetic rather
    # than the storage. 0.1 would not be: float32(0.1) differs from
    # float64(0.1) by ~1.5e-9 relative.
    kernel = np.array([[1.0, -2.0], [0.5, 0.25], [0.0, 3.0]])
    bias = np.array([0.25, -0.5])
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "in", [], batch_input_shape=[None, 3], dtype="float32"),
            _layer("Dense", "d", ["in"], units=2, activation="sigmoid", use_bias=True),
        ],
        [["in", 0, 0]],
        [["d", 0, 0]],
        {"d": {"kernel": kernel, "bias": bias}},
    )
    g = load_graph(p)
    assert g.input_names == ("in",) and g.output_names == ("d",)
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out = g([x])[0]
    np.testing.assert_allclose(out, _sigmoid(x @ kernel + bias), rtol=1e-12)


def test_batchnorm_uses_moving_statistics(tmp_path):
    p = tmp_path / "m.h5"
    stats = {
        "gamma": np.array([2.0, 1.0]),
        "beta": np.array([0.5, -0.5]),
        "moving_mean": np.array([1.0, 2.0]),
        "moving_variance": np.array([4.0, 9.0]),
    }
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "in", [], batch_input_shape=[None, 2], dtype="float32"),
            _layer("BatchNormalization", "bn", ["in"], axis=[1], epsilon=1e-3),
        ],
        [["in", 0, 0]],
        [["bn", 0, 0]],
        {"bn": stats},
    )
    x = np.array([[3.0, 5.0]])
    out = load_graph(p)([x])[0]
    want = stats["gamma"] * (x - stats["moving_mean"]) / np.sqrt(
        stats["moving_variance"] + 1e-3
    ) + stats["beta"]
    np.testing.assert_allclose(out, want, rtol=1e-12)


def test_conv1d_valid_then_maxpool_matches_hand_computation(tmp_path):
    p = tmp_path / "m.h5"
    kernel = np.array([[[1.0]], [[1.0]]])          # (k=2, cin=1, cout=1), sum of pairs
    _write_legacy_h5(
        p,
        [
            _layer(
                "InputLayer", "in", [], batch_input_shape=[None, 5, 1], dtype="float32"
            ),
            _layer(
                "Conv1D", "c", ["in"], filters=1, kernel_size=[2], strides=[1],
                padding="valid", dilation_rate=[1], activation="linear", use_bias=True,
            ),
            _layer("MaxPooling1D", "mp", ["c"], pool_size=[2], strides=[2],
                   padding="valid"),
        ],
        [["in", 0, 0]],
        [["mp", 0, 0]],
        {"c": {"kernel": kernel, "bias": np.array([0.0])}},
    )
    x = np.arange(5.0).reshape(1, 5, 1)            # 0 1 2 3 4
    out = load_graph(p)([x])[0]
    # conv -> [1, 3, 5, 7]; maxpool/2 -> [3, 7]
    np.testing.assert_allclose(out[0, :, 0], [3.0, 7.0], rtol=1e-12)


def test_flatten_is_channels_last(tmp_path):
    p = tmp_path / "m.h5"
    _write_legacy_h5(
        p,
        [
            _layer(
                "InputLayer", "in", [], batch_input_shape=[None, 2, 3], dtype="float32"
            ),
            _layer("Flatten", "fl", ["in"]),
        ],
        [["in", 0, 0]],
        [["fl", 0, 0]],
        {},
    )
    x = np.arange(6.0).reshape(1, 2, 3)
    np.testing.assert_allclose(load_graph(p)([x])[0][0], np.arange(6.0))


def test_dropout_is_identity_and_graph_order_may_be_arbitrary(tmp_path):
    # `input_1` is listed *after* the layer that consumes it, exactly as in the
    # real artifact, so the evaluator cannot assume the list is topological.
    p = tmp_path / "m.h5"
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "a", [], batch_input_shape=[None, 2], dtype="float32"),
            _layer("Concatenate", "cat", ["a", "b"], axis=-1),
            _layer("Dropout", "dr", ["cat"], rate=0.2),
            _layer("InputLayer", "b", [], batch_input_shape=[None, 3], dtype="float32"),
        ],
        [["a", 0, 0], ["b", 0, 0]],
        [["dr", 0, 0]],
        {},
    )
    g = load_graph(p)
    a = np.array([[1.0, 2.0]])
    b = np.array([[3.0, 4.0, 5.0]])
    np.testing.assert_allclose(g({"a": a, "b": b})[0], [[1, 2, 3, 4, 5]])
    np.testing.assert_allclose(g([a, b])[0], [[1, 2, 3, 4, 5]])


def test_a_dict_feed_missing_an_input_is_named_in_the_error(tmp_path):
    p = tmp_path / "m.h5"
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "a", [], batch_input_shape=[None, 2], dtype="float32"),
            _layer("Concatenate", "cat", ["a", "b"], axis=-1),
            _layer("InputLayer", "b", [], batch_input_shape=[None, 3], dtype="float32"),
        ],
        [["a", 0, 0], ["b", 0, 0]],
        [["cat", 0, 0]],
        {},
    )
    with pytest.raises(ValueError, match="'b'"):
        load_graph(p)({"a": np.zeros((1, 2))})


def test_unsupported_layer_names_the_class(tmp_path):
    p = tmp_path / "m.h5"
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "in", [], batch_input_shape=[None, 2], dtype="float32"),
            _layer("LSTM", "l", ["in"], units=4),
        ],
        [["in", 0, 0]],
        [["l", 0, 0]],
        {},
    )
    with pytest.raises(UnsupportedLayer, match="LSTM"):
        load_graph(p)([np.zeros((1, 2))])


def test_wrong_number_of_inputs_is_an_error(tmp_path):
    p = tmp_path / "m.h5"
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "in", [], batch_input_shape=[None, 2], dtype="float32"),
            _layer("Dense", "d", ["in"], units=1, activation="linear", use_bias=False),
        ],
        [["in", 0, 0]],
        [["d", 0, 0]],
        {"d": {"kernel": np.ones((2, 1))}},
    )
    g = load_graph(p)
    with pytest.raises(ValueError):
        g([np.zeros((1, 2)), np.zeros((1, 2))])
    with pytest.raises(ValueError):
        g([np.zeros((1, 7))])        # wrong feature width


def test_predict_members_refuses_a_multi_output_graph(tmp_path):
    p = tmp_path / "m.h5"
    _write_legacy_h5(
        p,
        [
            _layer("InputLayer", "in", [], batch_input_shape=[None, 2], dtype="float32"),
            _layer("Dense", "a", ["in"], units=1, activation="linear", use_bias=False),
            _layer("Dense", "b", ["in"], units=1, activation="linear", use_bias=False),
        ],
        [["in", 0, 0]],
        [["a", 0, 0], ["b", 0, 0]],
        {"a": {"kernel": np.ones((2, 1))}, "b": {"kernel": np.ones((2, 1))}},
    )
    graphs = load_ensemble([p])
    with pytest.raises(ValueError, match="single-output"):
        predict_members(graphs, [np.zeros((1, 2))])


pytestmark_upstream = pytest.mark.skipif(
    not TM_UPSTREAM.exists(), reason=f"upstream weights not available: {TM_UPSTREAM}"
)


@pytestmark_upstream
def test_real_ensemble_members_have_different_layer_names():
    # MEASURED: all ten members were built in one Keras session, so the
    # global name counter ran on - inputs are input_1/input_2 for member 0,
    # input_3/input_4 for member 1, through input_19/input_20 for member 9,
    # and the outputs are dense_4, dense_9, ... dense_49. Only the
    # POSITIONAL order is common, which is why an ensemble is fed a
    # sequence and never a dict keyed on one member's names.
    graphs = load_ensemble(sorted(TM_UPSTREAM.glob("best_model_?_4c.h5")))
    assert len({g.input_names for g in graphs}) == 10
    assert len({g.output_names for g in graphs}) == 10
    for g in graphs:
        assert len(g.input_names) == 2 and len(g.output_names) == 1
        assert [g.input_shapes[n] for n in g.input_names] == [
            (None, 11), (None, 33, 5)
        ]


@pytestmark_upstream
def test_real_tearing_ensemble_loads_and_predicts():
    paths = sorted(TM_UPSTREAM.glob("best_model_?_4c.h5"))
    assert len(paths) == 10
    graphs = load_ensemble(paths)
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(7, 11))
    x1 = rng.normal(size=(7, 33, 5))
    members = predict_members(graphs, [x0, x1])       # positional, per above
    assert members.shape == (10, 7, 2)
    assert np.isfinite(members).all()
    # the members are different models, not ten copies
    assert members.std(axis=0).max() > 1e-6
    np.testing.assert_allclose(members, predict_members(graphs, [x0, x1]))


@pytestmark_upstream
def test_feeding_an_ensemble_by_name_fails_loudly():
    # The trap this guards: a dict keyed on member 0's names fits member 0
    # and raises for the other nine, rather than quietly mispredicting.
    graphs = load_ensemble(sorted(TM_UPSTREAM.glob("best_model_?_4c.h5")))
    rng = np.random.default_rng(0)
    feed = {"input_1": rng.normal(size=(3, 11)),
            "input_2": rng.normal(size=(3, 33, 5))}
    graphs[0](feed)                                    # member 0 is fine
    with pytest.raises(ValueError, match="missing input"):
        graphs[1](feed)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_keras_h5.py -q
```
Expected: `ModuleNotFoundError: No module named 'labelmaker.models'`.

- [ ] **Step 3: Write the implementation**

`src/labelmaker/models/__init__.py`:

```python
"""One folder per trained model: a card, an I/O spec, and a loader."""
```

`src/labelmaker/models/runners/__init__.py`:

```python
"""Framework-specific model loading. The only place a framework appears."""
```

`src/labelmaker/models/runners/keras_h5.py`:

```python
"""Evaluate a Keras-2 legacy HDF5 model in numpy.

The group's Keras artifacts were saved by Keras 2.8, and there is no
TensorFlow build for this environment's Python (conda-forge ships
tensorflow-cpu 2.21 for py312 only, against this repo's `python <3.12`
pin). The graphs in the roster are small feed-forward networks whose layers
have closed-form inference semantics, so instead of a framework we read the
serialized config and the moving statistics out of the file and evaluate the
graph directly.

Supported layers: InputLayer, BatchNormalization, Conv1D, MaxPooling1D,
Flatten, Dense, Concatenate, Dropout (identity at inference). Anything else
raises UnsupportedLayer, naming the class, rather than silently skipping it.

Equality with TensorFlow is not assumed: `labelmaker.validate adapter`
compares this evaluator against a real Keras load of the same file to 1e-5
and stores the golden outputs under tests/labelmaker/data/.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


class UnsupportedLayer(RuntimeError):
    """A layer class or option this evaluator does not implement."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


_ACTIVATIONS = {
    None: lambda x: x,
    "linear": lambda x: x,
    "sigmoid": _sigmoid,
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
    "softmax": _softmax,
}


def _activation(name):
    if name not in _ACTIVATIONS:
        raise UnsupportedLayer(f"activation {name!r}")
    return _ACTIVATIONS[name]


def _inbound_names(layer: dict) -> list[str]:
    """Names this layer consumes, in order, from either nesting Keras uses."""
    names: list[str] = []

    def walk(node):
        if isinstance(node, list):
            if node and isinstance(node[0], str):
                names.append(node[0])
            else:
                for item in node:
                    walk(item)

    walk(layer.get("inbound_nodes") or [])
    return names


def _conv1d(x, kernel, bias, cfg):
    k, stride = kernel.shape[0], int(cfg["strides"][0])
    dil = int(cfg.get("dilation_rate", [1])[0])
    eff = (k - 1) * dil + 1
    padding = cfg.get("padding", "valid")
    if padding == "same":
        need = max(0, (int(np.ceil(x.shape[1] / stride)) - 1) * stride + eff - x.shape[1])
        left = need // 2
        x = np.pad(x, ((0, 0), (left, need - left), (0, 0)))
    elif padding != "valid":
        raise UnsupportedLayer(f"Conv1D padding {padding!r}")
    out_len = (x.shape[1] - eff) // stride + 1
    if out_len <= 0:
        raise ValueError(f"Conv1D input too short: {x.shape} for kernel {k}")
    acc = np.zeros((x.shape[0], out_len, kernel.shape[2]), dtype=np.float64)
    for i in range(k):
        stop = i * dil + (out_len - 1) * stride + 1
        acc += x[:, i * dil : stop : stride, :] @ kernel[i]
    if bias is not None:
        acc += bias
    return acc


def _maxpool1d(x, cfg):
    pool = int(cfg["pool_size"][0])
    stride = int((cfg.get("strides") or [pool])[0])
    if cfg.get("padding", "valid") != "valid":
        raise UnsupportedLayer(f"MaxPooling1D padding {cfg['padding']!r}")
    out_len = (x.shape[1] - pool) // stride + 1
    idx = np.arange(out_len)[:, None] * stride + np.arange(pool)[None, :]
    return x[:, idx, :].max(axis=2)


def _batchnorm(x, w, cfg):
    axis = cfg.get("axis", -1)
    axis = int(axis[0] if isinstance(axis, (list, tuple)) else axis)
    n = x.shape[axis]
    shape = [1] * x.ndim
    shape[axis] = n
    gamma = w.get("gamma", np.ones(n))
    beta = w.get("beta", np.zeros(n))
    mean = w.get("moving_mean", np.zeros(n))
    var = w.get("moving_variance", np.ones(n))
    eps = float(cfg.get("epsilon", 1e-3))
    scaled = (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + eps)
    return gamma.reshape(shape) * scaled + beta.reshape(shape)


def _apply(layer: dict, ins: list[np.ndarray], w: dict[str, np.ndarray]):
    cls, cfg = layer["class_name"], layer["config"]
    if cls == "Dense":
        out = ins[0] @ w["kernel"]
        if cfg.get("use_bias", True):
            out = out + w["bias"]
        return _activation(cfg.get("activation"))(out)
    if cls == "BatchNormalization":
        return _batchnorm(ins[0], w, cfg)
    if cls == "Conv1D":
        out = _conv1d(ins[0], w["kernel"], w.get("bias"), cfg)
        return _activation(cfg.get("activation"))(out)
    if cls == "MaxPooling1D":
        return _maxpool1d(ins[0], cfg)
    if cls == "Flatten":
        return ins[0].reshape(ins[0].shape[0], -1)
    if cls == "Concatenate":
        return np.concatenate(ins, axis=int(cfg.get("axis", -1)))
    if cls == "Dropout":
        return ins[0]
    raise UnsupportedLayer(f"layer class {cls} ({cfg.get('name')})")


@dataclass(frozen=True)
class KerasGraph:
    """A loaded graph: config, weights, and the order of its inputs."""

    config: dict
    weights: dict[str, dict[str, np.ndarray]]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_shapes: dict[str, tuple]

    def __call__(self, inputs) -> list[np.ndarray]:
        feed = self._as_dict(inputs)
        tensors: dict[str, np.ndarray] = {}
        layers = {lay["config"]["name"]: lay for lay in self.config["config"]["layers"]}
        for name, lay in layers.items():
            if lay["class_name"] == "InputLayer":
                tensors[name] = np.asarray(feed[name], dtype=np.float64)
        pending = [lay for lay in layers.values() if lay["class_name"] != "InputLayer"]
        while pending:
            ready = [
                lay for lay in pending
                if all(dep in tensors for dep in _inbound_names(lay))
            ]
            if not ready:
                stuck = [lay["config"]["name"] for lay in pending]
                raise UnsupportedLayer(f"cannot resolve graph at {stuck}")
            for lay in ready:
                name = lay["config"]["name"]
                ins = [tensors[dep] for dep in _inbound_names(lay)]
                tensors[name] = _apply(lay, ins, self.weights.get(name, {}))
                pending.remove(lay)
        return [tensors[name] for name in self.output_names]

    def _as_dict(self, inputs) -> dict[str, np.ndarray]:
        if isinstance(inputs, dict):
            feed = dict(inputs)
        else:
            seq = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
            if len(seq) != len(self.input_names):
                raise ValueError(
                    f"expected {len(self.input_names)} inputs "
                    f"{self.input_names}, got {len(seq)}"
                )
            feed = dict(zip(self.input_names, seq))
        for name in self.input_names:
            if name not in feed:
                raise ValueError(
                    f"missing input {name!r}; this graph expects {self.input_names}"
                )
        for name, want in self.input_shapes.items():
            got = np.asarray(feed[name]).shape
            if len(got) != len(want) or got[1:] != tuple(want[1:]):
                raise ValueError(f"input {name}: expected (N, *{want[1:]}), got {got}")
        return feed


def load_graph(path) -> KerasGraph:
    """Read one legacy `.h5` model file."""
    with h5py.File(path, "r") as f:
        raw = f.attrs["model_config"]
        config = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        weights: dict[str, dict[str, np.ndarray]] = {}
        group = f["model_weights"]
        for lname in group:
            g = group[lname]
            per_layer: dict[str, np.ndarray] = {}
            for wname in g.attrs.get("weight_names", []):
                key = (wname.decode() if isinstance(wname, bytes) else wname)
                short = key.split("/")[-1].split(":")[0]
                per_layer[short] = np.asarray(g[key], dtype=np.float64)
            weights[lname] = per_layer
    inner = config["config"]
    shapes = {
        lay["config"]["name"]: tuple(lay["config"]["batch_input_shape"])
        for lay in inner["layers"]
        if lay["class_name"] == "InputLayer"
    }
    return KerasGraph(
        config=config,
        weights=weights,
        input_names=tuple(entry[0] for entry in inner["input_layers"]),
        output_names=tuple(entry[0] for entry in inner["output_layers"]),
        input_shapes=shapes,
    )


def load_ensemble(paths) -> tuple[KerasGraph, ...]:
    """Load ensemble members in the order given."""
    return tuple(load_graph(Path(p)) for p in paths)


def predict_members(graphs, inputs) -> np.ndarray:
    """`(n_members, n_rows, n_out)` for single-output graphs.

    Pass `inputs` as a SEQUENCE, not a dict. Members trained in one Keras
    session carry different layer names - this project's tearing ensemble
    runs input_1/input_2 through input_19/input_20, with outputs dense_4
    through dense_49 - so only the positional order is common across
    members. A dict keyed on one member's names fits that member and raises
    for the rest.

    Statistics across members are the caller's business: the label store
    keeps the mean as the label and the min/max as its spread.
    """
    outs = []
    for g in graphs:
        got = g(inputs)
        if len(got) != 1:
            raise ValueError(
                f"expected a single-output graph, got {len(got)} outputs: "
                f"{g.output_names}"
            )
        outs.append(np.atleast_2d(got[0]))
    return np.stack(outs, axis=0)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_keras_h5.py -q
```
Expected: `8 passed` on stellar (7 synthetic + the upstream-weights test).

- [ ] **Step 5: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/models/__init__.py src/labelmaker/models/runners/__init__.py \
        src/labelmaker/models/runners/keras_h5.py tests/labelmaker/test_keras_h5.py
git commit -m "labelmaker: numpy evaluator for Keras-2 legacy HDF5 graphs"
```

---

### Task 6: The model I/O contract

**Files:**
- Create: `src/labelmaker/models/base.py`
- Test: `tests/labelmaker/test_model_base.py`

**Interfaces:**
- Consumes: `features.namespace`, `features.store.FeatureArray`, `timebase.sample_at`.
- Produces: `base.TRANSFORMS: dict[str, Callable]`; `base.ACTIVATIONS: dict[str, Callable]`; frozen dataclasses `InputField(model_name, canonical, lag="t", transform=None, scale=1.0)` with property `kind`; `DomainRule(canonical, stat="value", lo=None, hi=None, lo_inclusive=False, hi_inclusive=False)`; `BuiltInputs(t, scalars, profiles, valid, missing, resolvers)`; `InputSpec(fields, dt_s, rho_grid=ns.RHO_GRID, nan_policy="zero", domain=())` with properties `scalar_fields`, `profile_fields`, `canonical_names` and method `build(features, grid) -> BuiltInputs`; `OutputField(name, task, column, activation="none", units="", classes=())`; `Decoded(mean, lo, hi)`; `OutputSpec(fields)` with `decode(members) -> dict[str, Decoded]`; `ModelAdapter(slug, card_id, framework, time_step_ms, artifacts, upstream, input_spec, output_spec, load, ensemble_n=1)`.

- [ ] **Step 1: Write the failing test**

`tests/labelmaker/test_model_base.py`:

```python
"""Building model inputs from canonical features, and reading outputs back."""
import numpy as np
import pytest

from labelmaker.features import namespace as ns
from labelmaker.features.store import FeatureArray
from labelmaker.models.base import (
    BuiltInputs,
    Decoded,
    DomainRule,
    InputField,
    InputSpec,
    OutputField,
    OutputSpec,
)

GRID = 0.025 * np.arange(6)          # 0.000 .. 0.125 s


def _scalar_feature(values, resolver="archive"):
    return FeatureArray(
        x=0.025 * np.arange(len(values)),
        y=np.asarray(values, dtype=float)[None, :],
        attrs={"resolver": resolver},
    )


def _profile_feature(rows):
    y = np.asarray(rows, dtype=float).T          # (33, T)
    return FeatureArray(x=0.025 * np.arange(y.shape[1]), y=y, attrs={"resolver": "archive"})


def test_lag_shifts_the_scalar_by_one_step():
    spec = InputSpec(
        fields=(
            InputField("bt_at_t", "bt", lag="t"),
            InputField("bt_next", "ip", lag="t+dt"),
        ),
        dt_s=0.025,
    )
    feats = {
        "bt": _scalar_feature([0, 1, 2, 3, 4, 5, 6]),
        "ip": _scalar_feature([0, 1, 2, 3, 4, 5, 6]),
    }
    built = spec.build(feats, GRID)
    assert built.scalars.shape == (6, 2)
    np.testing.assert_allclose(built.scalars[:, 0], [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(built.scalars[:, 1], [1, 2, 3, 4, 5, 6])
    assert built.valid.all()
    assert built.resolvers == {"bt": "archive", "ip": "archive"}


def test_transform_and_scale_are_applied_after_sampling():
    spec = InputSpec(
        fields=(
            InputField("inv_q", "qpsi", transform="reciprocal"),
            InputField("ip_ma", "ip", scale=1e-6),
        ),
        dt_s=0.025,
    )
    q_rows = [np.full(33, 2.0)] * 6
    feats = {"qpsi": _profile_feature(q_rows), "ip": _scalar_feature([1e6] * 6)}
    built = spec.build(feats, GRID)
    assert built.profiles.shape == (6, 33, 1)
    np.testing.assert_allclose(built.profiles[:, :, 0], 0.5)
    np.testing.assert_allclose(built.scalars[:, 0], 1.0)


def test_profile_field_order_is_the_stacking_order():
    spec = InputSpec(
        fields=(
            InputField("ne", "ne_zipfit"),
            InputField("te", "te_zipfit"),
        ),
        dt_s=0.025,
    )
    feats = {
        "ne_zipfit": _profile_feature([np.full(33, 3.0)] * 6),
        "te_zipfit": _profile_feature([np.full(33, 7.0)] * 6),
    }
    built = spec.build(feats, GRID)
    np.testing.assert_allclose(built.profiles[:, :, 0], 3.0)
    np.testing.assert_allclose(built.profiles[:, :, 1], 7.0)


def test_missing_feature_is_recorded_and_zero_filled_but_invalid():
    spec = InputSpec(fields=(InputField("bt", "bt"), InputField("ip", "ip")), dt_s=0.025)
    built = spec.build({"bt": _scalar_feature([1.0] * 6)}, GRID)
    assert built.missing == ("ip",)
    np.testing.assert_allclose(built.scalars[:, 1], 0.0)   # nan_policy="zero"
    assert not built.valid.any()                            # nothing is trustworthy


def test_a_gap_larger_than_one_step_is_not_extrapolated():
    spec = InputSpec(fields=(InputField("bt", "bt"),), dt_s=0.025)
    short = FeatureArray(x=np.array([0.0, 0.025]), y=np.array([[1.0, 2.0]]),
                         attrs={"resolver": "archive"})
    built = spec.build({"bt": short}, GRID)
    assert built.valid.tolist() == [True, True, False, False, False, False]


def test_the_last_step_of_a_t_plus_dt_field_is_never_extrapolated():
    # The real archive record is exactly 240 rows at 25 ms, so a t+dt field's
    # final query lands one step past the end. Upstream training dropped that
    # row for the same reason (x0 = rows[1:], x1 = rows[:-1]); reusing the
    # last row would fabricate a label, so the step must be flagged.
    spec = InputSpec(fields=(InputField("bt", "bt", lag="t+dt"),), dt_s=ns.STEP_S)
    x = ns.STEP_S * np.arange(240)
    feats = {
        "bt": FeatureArray(
            x=x, y=np.arange(240, dtype=float)[None, :], attrs={"resolver": "archive"}
        )
    }
    built = spec.build(feats, ns.GRID_S)
    assert built.valid.sum() == 239
    assert not built.valid[-1]
    assert built.valid[:-1].all()


def test_the_trust_boundary_is_not_decided_by_float_residue():
    # A gap comfortably inside half a step is trusted, one comfortably
    # outside is not, and neither verdict sits on a knife edge.
    spec = InputSpec(fields=(InputField("bt", "bt"),), dt_s=0.025)
    x = np.array([0.0, 0.05])          # 50 ms apart, so 0.025 is the midpoint
    feats = {"bt": FeatureArray(x=x, y=np.array([[1.0, 2.0]]))}
    built = spec.build(feats, np.array([0.010, 0.020]))
    assert built.valid.tolist() == [True, False]


def test_domain_rules_flag_rows_without_dropping_them():
    spec = InputSpec(
        fields=(
            InputField("kappa", "kappa"),
            InputField("ne", "ne_zipfit"),
            InputField("rot", "rot_zipfit"),
        ),
        dt_s=0.025,
        domain=(
            DomainRule("kappa", "value", lo=1.6, hi=2.0),
            DomainRule("ne_zipfit", "min", lo=0.0, lo_inclusive=True),
            DomainRule("ne_zipfit", "max", hi=12.0),
            DomainRule("rot_zipfit", "absmax", hi=150.0),
        ),
    )
    kappa = _scalar_feature([1.8, 1.5, 1.8, 1.8, 1.8, 1.8])
    ne = [np.full(33, 3.0) for _ in range(6)]
    ne[2] = np.full(33, 20.0)                     # too high
    ne[3] = np.full(33, -1.0)                     # negative
    rot = [np.full(33, 10.0) for _ in range(6)]
    rot[4] = np.full(33, -200.0)                  # |rot| too large
    built = spec.build(
        {"kappa": kappa, "ne_zipfit": _profile_feature(ne),
         "rot_zipfit": _profile_feature(rot)},
        GRID,
    )
    assert built.valid.tolist() == [True, False, False, False, False, True]
    assert built.scalars.shape == (6, 1) and built.profiles.shape == (6, 33, 2)


def test_domain_rule_for_an_unused_feature_is_a_programming_error():
    spec = InputSpec(
        fields=(InputField("bt", "bt"),),
        dt_s=0.025,
        domain=(DomainRule("kappa", "value", lo=0.0),),
    )
    with pytest.raises(KeyError, match="kappa"):
        spec.build({"bt": _scalar_feature([1.0] * 6)}, GRID)


def test_a_profile_field_can_also_carry_the_t_plus_dt_lag():
    # The lag arithmetic is kind-agnostic, but only scalars were covered.
    spec = InputSpec(
        fields=(
            InputField("ne_now", "ne_zipfit", lag="t"),
            InputField("ne_next", "te_zipfit", lag="t+dt"),
        ),
        dt_s=0.025,
    )
    rows = [np.full(33, float(i)) for i in range(7)]
    feats = {"ne_zipfit": _profile_feature(rows), "te_zipfit": _profile_feature(rows)}
    built = spec.build(feats, GRID)
    np.testing.assert_allclose(built.profiles[:, 0, 0], [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(built.profiles[:, 0, 1], [1, 2, 3, 4, 5, 6])


def test_a_domain_rule_whose_stat_mismatches_the_field_kind_is_rejected():
    # Otherwise this surfaces as numpy.exceptions.AxisError from reducing a
    # 1-D array along axis 1, well away from the typo that caused it.
    with pytest.raises(ValueError, match="needs a profile"):
        DomainRule("kappa", "max", hi=2.0)
    with pytest.raises(ValueError, match="needs a scalar"):
        DomainRule("ne_zipfit", "value", hi=12.0)


def test_an_unknown_nan_policy_is_rejected():
    with pytest.raises(ValueError, match="nan_policy"):
        InputSpec(fields=(InputField("bt", "bt"),), dt_s=0.025, nan_policy="zeros")


def test_an_ambiguous_domain_rule_is_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        InputSpec(
            fields=(
                InputField("bt_now", "bt", lag="t"),
                InputField("bt_next", "bt", lag="t+dt"),
            ),
            dt_s=0.025,
            domain=(DomainRule("bt", "value", lo=0.0),),
        )


def test_bad_field_definitions_are_rejected_at_construction():
    with pytest.raises(ValueError, match="lag"):
        InputField("bt", "bt", lag="tomorrow")
    with pytest.raises(ValueError, match="transform"):
        InputField("bt", "bt", transform="logarithm")


def test_decode_applies_the_activation_after_the_ensemble_mean():
    spec = OutputSpec(
        fields=(
            OutputField("betan", "regression", column=0),
            OutputField("tm_prob", "binary", column=1, activation="sigmoid"),
        )
    )

    def sigmoid(a):
        return 1.0 / (1.0 + np.exp(-np.asarray(a, dtype=float)))

    # Two members, three rows. The tearing logits are deliberately
    # asymmetric: with symmetric logits both orderings collapse to 0.5 and
    # the test would prove nothing.
    members = np.array(
        [[[1.0, 1.0], [2.0, -2.0], [3.0, 0.0]],
         [[3.0, 3.0], [4.0, -1.0], [5.0, 4.0]]]
    )
    got = spec.decode(members)
    np.testing.assert_allclose(got["betan"].mean, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(got["betan"].lo, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(got["betan"].hi, [3.0, 4.0, 5.0])
    # mean over members in logit space, THEN the activation
    np.testing.assert_allclose(got["tm_prob"].mean, sigmoid([2.0, -1.5, 2.0]))
    np.testing.assert_allclose(got["tm_prob"].lo, sigmoid([1.0, -2.0, 0.0]))
    np.testing.assert_allclose(got["tm_prob"].hi, sigmoid([3.0, -1.0, 4.0]))
    # averaging probabilities instead would give a materially different answer
    averaged_probs = (sigmoid([1.0, -2.0, 0.0]) + sigmoid([3.0, -1.0, 4.0])) / 2
    assert np.abs(averaged_probs - got["tm_prob"].mean).max() > 0.01
    assert isinstance(got["tm_prob"], Decoded)


def test_built_inputs_is_the_declared_shape_contract():
    built = BuiltInputs(
        t=GRID, scalars=np.zeros((6, 2)), profiles=np.zeros((6, 33, 3)),
        valid=np.ones(6, bool), missing=(), resolvers={},
    )
    assert built.scalars.shape[0] == built.profiles.shape[0] == built.t.size
    assert built.profiles.shape[1] == ns.RHO_GRID.size
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_model_base.py -q
```
Expected: `ImportError: cannot import name 'BuiltInputs' from 'labelmaker.models.base'`.

- [ ] **Step 3: Write the implementation**

`src/labelmaker/models/base.py`:

```python
"""What a model consumes and produces, in canonical terms.

A model folder contributes a `spec.py` holding these dataclasses and nothing
else: the mapping from the names the model was trained under to canonical
features, the lag structure, the training-time admissible domain, and how to
read its output columns. Fetching, sampling, ensembling and writing are
shared, so adding a model adds no pipeline code.

The domain is kept as data (a tuple of `DomainRule`) rather than a
hand-written predicate, so the same clauses can be printed into the model
card and checked by a test.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..features import namespace as ns
from ..features.store import FeatureArray
from ..timebase import sample_at

#: Named pure transforms a spec may apply after sampling - named rather than
#: inline lambdas so the model card can list them and a test can compare the
#: card against the spec.
TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "reciprocal": lambda a: 1.0 / a,
}

ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": lambda a: a,
    "sigmoid": lambda a: 1.0 / (1.0 + np.exp(-a)),
}

STATS = ("value", "min", "max", "absmax")
NAN_POLICIES = ("zero",)


@dataclass(frozen=True)
class InputField:
    """One model input: its trained-on name and where it now comes from."""

    model_name: str
    canonical: str
    lag: str = "t"
    transform: str | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.lag not in ("t", "t+dt"):
            raise ValueError(f"lag must be 't' or 't+dt', got {self.lag!r}")
        if self.transform is not None and self.transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {self.transform!r}")
        ns.by_name(self.canonical)  # fail loudly on a typo, at import time

    @property
    def kind(self) -> str:
        return ns.by_name(self.canonical).kind


@dataclass(frozen=True)
class DomainRule:
    """One clause of the upstream training filter, kept as data.

    `stat` says how a profile is reduced along the radial axis before the
    bound is applied. Rows outside a clause are flagged, never dropped: a
    label file covers the whole shot and says where not to trust it.
    """

    canonical: str
    stat: str = "value"
    lo: float | None = None
    hi: float | None = None
    lo_inclusive: bool = False
    hi_inclusive: bool = False

    def __post_init__(self) -> None:
        if self.stat not in STATS:
            raise ValueError(f"stat must be one of {STATS}, got {self.stat!r}")
        # Model authors write these tuples by hand - the tearing spec alone has
        # fourteen - so a stat that does not match the field's kind is a
        # plausible copy-paste error. Caught here, at import, rather than as an
        # opaque numpy AxisError from reducing a 1-D array along axis 1.
        kind = ns.by_name(self.canonical).kind
        if self.stat == "value" and kind != "scalar":
            raise ValueError(
                f"stat='value' compares a field directly, so it needs a scalar; "
                f"{self.canonical!r} is a profile - reduce it with 'min', 'max' "
                "or 'absmax'"
            )
        if self.stat != "value" and kind != "profile":
            raise ValueError(
                f"stat={self.stat!r} reduces along the radial axis, so it needs a "
                f"profile; {self.canonical!r} is a scalar - use 'value'"
            )


@dataclass(frozen=True)
class BuiltInputs:
    """Model-ready arrays for one shot, plus what is trustworthy."""

    t: np.ndarray                  # (T,) seconds - the label time stamps
    scalars: np.ndarray            # (T, n_scalar)
    profiles: np.ndarray           # (T, n_rho, n_profile)
    valid: np.ndarray              # (T,) bool
    missing: tuple[str, ...]
    resolvers: dict[str, str]


@dataclass(frozen=True)
class InputSpec:
    """How to turn a feature file into this model's input arrays."""

    fields: tuple[InputField, ...]
    dt_s: float
    rho_grid: np.ndarray = field(default_factory=lambda: ns.RHO_GRID)
    nan_policy: str = "zero"
    domain: tuple[DomainRule, ...] = ()

    def __post_init__(self) -> None:
        if self.nan_policy not in NAN_POLICIES:
            raise ValueError(
                f"nan_policy must be one of {NAN_POLICIES}, got {self.nan_policy!r}"
            )
        # A domain rule on a canonical that two fields share would check an
        # unspecified one of them. `canonical_names` deduplicates, so the same
        # physical quantity at two lags is an anticipated configuration - make
        # the ambiguity loud rather than arbitrary.
        for rule in self.domain:
            shared = [f.model_name for f in self.fields if f.canonical == rule.canonical]
            if len(shared) > 1:
                raise ValueError(
                    f"domain rule on {rule.canonical!r} is ambiguous: fields "
                    f"{shared} all use it"
                )

    @property
    def scalar_fields(self) -> tuple[InputField, ...]:
        return tuple(f for f in self.fields if f.kind == "scalar")

    @property
    def profile_fields(self) -> tuple[InputField, ...]:
        return tuple(f for f in self.fields if f.kind == "profile")

    @property
    def canonical_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f.canonical for f in self.fields))

    def build(self, features: Mapping[str, FeatureArray], grid) -> BuiltInputs:
        """Sample, transform, flag, and stack - in that order.

        The flags are computed on the sampled values *before* the NaN policy
        runs, so a zero-filled hole is never mistaken for a real zero.
        """
        grid = np.asarray(grid, dtype=np.float64)
        n = grid.size
        sampled: dict[str, np.ndarray] = {}
        missing: list[str] = []
        resolvers: dict[str, str] = {}
        for f in self.fields:
            arr = features.get(f.canonical)
            if arr is None:
                missing.append(f.canonical)
                shape = (n,) if f.kind == "scalar" else (n, self.rho_grid.size)
                sampled[f.model_name] = np.full(shape, np.nan)
                continue
            resolvers[f.canonical] = str(arr.attrs.get("resolver", "unknown"))
            t = grid + self.dt_s if f.lag == "t+dt" else grid
            # Half a step, not a whole one, for two reasons. It is the
            # correct nearest-neighbour rule: a query is trustworthy only
            # if a real sample lies within half a sampling interval. And a
            # whole step puts the record-edge case exactly on the boundary
            # - for a t+dt field on a 240-row 25 ms record the final query
            # sits 0.025 s past the last sample, so `gap > dt_s` is decided
            # by a 3.5e-16 float residue. It happens to fall the right way
            # on this data; half a step clears it by 0.0125.
            vals = sample_at(arr.x, arr.y, t, max_gap=self.dt_s / 2)
            v = vals[0] if f.kind == "scalar" else vals.T
            if f.transform is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = TRANSFORMS[f.transform](v)
            sampled[f.model_name] = v * f.scale
        valid = self._validity(sampled, n)
        scalars = (
            np.stack([sampled[f.model_name] for f in self.scalar_fields], axis=1)
            if self.scalar_fields
            else np.zeros((n, 0))
        )
        profiles = (
            np.stack([sampled[f.model_name] for f in self.profile_fields], axis=2)
            if self.profile_fields
            else np.zeros((n, self.rho_grid.size, 0))
        )
        if self.nan_policy == "zero":
            scalars = np.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)
            profiles = np.nan_to_num(profiles, nan=0.0, posinf=0.0, neginf=0.0)
        return BuiltInputs(
            t=grid,
            scalars=scalars,
            profiles=profiles,
            valid=valid,
            missing=tuple(sorted(set(missing))),
            resolvers=resolvers,
        )

    def _validity(self, sampled: dict[str, np.ndarray], n: int) -> np.ndarray:
        ok = np.ones(n, dtype=bool)
        for v in sampled.values():
            ok &= np.isfinite(v) if v.ndim == 1 else np.isfinite(v).all(axis=1)
        by_canonical = {f.canonical: f.model_name for f in self.fields}
        for rule in self.domain:
            key = by_canonical.get(rule.canonical)
            if key is None:
                raise KeyError(
                    f"domain rule names {rule.canonical}, which is not an input"
                )
            v = sampled[key]
            # Non-finite rows are already invalid, so filling them keeps the
            # reductions warning-free without changing any verdict.
            v = np.where(np.isfinite(v), v, 0.0)
            if rule.stat == "value":
                red = v
            elif rule.stat == "min":
                red = v.min(axis=1)
            elif rule.stat == "max":
                red = v.max(axis=1)
            else:
                red = np.abs(v).max(axis=1)
            if rule.lo is not None:
                ok &= (red >= rule.lo) if rule.lo_inclusive else (red > rule.lo)
            if rule.hi is not None:
                ok &= (red <= rule.hi) if rule.hi_inclusive else (red < rule.hi)
        return ok


@dataclass(frozen=True)
class OutputField:
    """One label a model produces, and which output column carries it."""

    name: str
    task: str
    column: int
    activation: str = "none"
    units: str = ""
    classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}")


@dataclass(frozen=True)
class Decoded:
    """A label series and the ensemble's spread around it."""

    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


@dataclass(frozen=True)
class OutputSpec:
    fields: tuple[OutputField, ...]

    def decode(self, members: np.ndarray) -> dict[str, Decoded]:
        """`(M, T, C)` raw member outputs -> one `Decoded` per label.

        The activation is applied *after* the ensemble mean, matching the
        upstream harness (`test.py` plots `sigmoid(mean(members))`).
        Averaging probabilities instead would be a different - arguably
        better calibrated - number, but it would not be this model's output.
        """
        members = np.asarray(members, dtype=np.float64)
        if members.ndim != 3:
            raise ValueError(f"expected (M, T, C) members, got {members.shape}")
        out: dict[str, Decoded] = {}
        for f in self.fields:
            col = members[:, :, f.column]
            act = ACTIVATIONS[f.activation]
            out[f.name] = Decoded(
                mean=act(col.mean(axis=0)),
                lo=act(col.min(axis=0)),
                hi=act(col.max(axis=0)),
            )
        return out


@dataclass(frozen=True)
class ModelAdapter:
    """Everything labelmaker needs to run one trained model."""

    slug: str
    card_id: str
    framework: str
    time_step_ms: float
    artifacts: tuple[str, ...]
    upstream: str
    input_spec: InputSpec
    output_spec: OutputSpec
    load: Callable[[Any], Callable[[BuiltInputs], np.ndarray]]
    ensemble_n: int = 1
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_model_base.py -q
```
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/models/base.py tests/labelmaker/test_model_base.py
git commit -m "labelmaker: model input/output contract with data-driven domain rules"
```

---

### Task 7: Model registry, card format, and the scaffolded roster

**Files:**
- Create: `src/labelmaker/models/registry.py`, `src/labelmaker/models/README.md`
- Create: six scaffold folders under `src/labelmaker/models/`, each with `__init__.py`, `spec.py`, `README.md`
- Test: `tests/labelmaker/test_registry.py`

**Interfaces:**
- Consumes: `models.base.ModelAdapter`.
- Produces: `registry.MODELS_DIR: Path`; `model_slugs() -> list[str]`; `card_path(slug) -> Path`; `parse_card(text: str) -> dict`; `read_card(slug) -> dict`; `status(slug) -> str`; `implemented() -> list[str]`; `scaffolds() -> list[str]`; `load_adapter(slug) -> ModelAdapter`; `card_discrepancies(slug) -> list[str]`.

- [ ] **Step 1: Write the failing test**

`tests/labelmaker/test_registry.py`:

```python
"""Every model folder is discoverable and its card is machine-readable."""
import pytest

from labelmaker.models import registry

REQUIRED_TOP = ("pipeline_tag", "tags", "library_name", "labelmaker")
REQUIRED_LM = ("status", "slug", "card_id", "framework", "upstream", "inputs", "outputs")


def test_front_matter_is_parsed():
    card = registry.parse_card(
        "---\nlibrary_name: keras\nlabelmaker:\n  status: scaffold\n---\n\n# Title\n"
    )
    assert card["library_name"] == "keras"
    assert card["labelmaker"]["status"] == "scaffold"


def test_a_card_without_front_matter_is_an_error():
    with pytest.raises(ValueError, match="front matter"):
        registry.parse_card("# Just a heading\n")


def test_every_model_folder_has_a_well_formed_card():
    slugs = registry.model_slugs()
    assert slugs, "no model folders found"
    for slug in slugs:
        card = registry.read_card(slug)
        for key in REQUIRED_TOP:
            assert key in card, f"{slug}: card is missing {key}"
        lm = card["labelmaker"]
        for key in REQUIRED_LM:
            assert key in lm, f"{slug}: labelmaker block is missing {key}"
        assert lm["slug"] == slug
        assert lm["card_id"] == f"plasmacontrol/{slug.replace('_', '-')}"
        assert lm["status"] in ("implemented", "scaffold")


def test_the_roster_readme_lists_every_folder_and_the_exclusions():
    text = (registry.MODELS_DIR / "README.md").read_text()
    for slug in registry.model_slugs():
        assert slug in text, f"{slug} missing from models/README.md"
    for excluded in ("tokeye", "ae_tf_maskrcnn", "tokamind", "diag2diag"):
        assert excluded in text.lower(), f"{excluded} not recorded as excluded"


def test_scaffolds_declare_what_blocks_them_and_refuse_to_load():
    assert registry.scaffolds(), "expected scaffolded models"
    for slug in registry.scaffolds():
        card = registry.read_card(slug)
        assert card["labelmaker"]["blocked_on"], f"{slug}: no blocked_on entries"
        with pytest.raises(NotImplementedError, match=slug):
            registry.load_adapter(slug)


def test_implemented_models_load_and_match_their_card():
    for slug in registry.implemented():
        adapter = registry.load_adapter(slug)
        assert adapter.slug == slug
        assert registry.card_discrepancies(slug) == []
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_registry.py -q
```
Expected: `ModuleNotFoundError: No module named 'labelmaker.models.registry'`.

- [ ] **Step 3: Write `models/registry.py`**

```python
"""Discovering model folders and reading their cards.

A model folder is `models/<slug>/` with a `README.md` whose YAML front
matter follows the HuggingFace model-card standard plus one custom
`labelmaker:` block, and a `spec.py` exposing `ADAPTER`. The card is the
single place a physicist reads to learn what a model is, where its weights
came from, what it consumes, and how its labels scored; `card_discrepancies`
keeps the card honest about the first two.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import yaml

from .base import ModelAdapter

MODELS_DIR = Path(__file__).resolve().parent

_FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.S)


def model_slugs() -> list[str]:
    """Model folder names, sorted."""
    return sorted(
        p.name
        for p in MODELS_DIR.iterdir()
        if p.is_dir() and (p / "README.md").exists() and (p / "spec.py").exists()
    )


def card_path(slug: str) -> Path:
    return MODELS_DIR / slug / "README.md"


def parse_card(text: str) -> dict:
    """The YAML front matter of a model card."""
    m = _FRONT_MATTER.match(text)
    if not m:
        raise ValueError("card has no YAML front matter (expected a leading --- block)")
    data = yaml.safe_load(m.group(1))
    if not isinstance(data, dict):
        # ValueError, not TypeError, despite ruff's TRY004: `text` is always a
        # str, so this is a malformed *data file*, not a caller passing the
        # wrong argument type - and the sibling branch above raises ValueError
        # for the same category. A caller wanting to catch "bad card" should
        # need one except clause, not two.
        raise ValueError(  # noqa: TRY004
            f"card front matter is not a mapping: {type(data).__name__}"
        )
    return data


def read_card(slug: str) -> dict:
    return parse_card(card_path(slug).read_text())


def status(slug: str) -> str:
    return read_card(slug)["labelmaker"]["status"]


def implemented() -> list[str]:
    return [s for s in model_slugs() if status(s) == "implemented"]


def scaffolds() -> list[str]:
    return [s for s in model_slugs() if status(s) == "scaffold"]


def load_adapter(slug: str) -> ModelAdapter:
    """The `ADAPTER` of a model's `spec.py`.

    A scaffold's `spec.py` raises `NotImplementedError` at import, which is
    the intended behaviour: the folder documents a model that cannot run yet.
    """
    module = importlib.import_module(f"labelmaker.models.{slug}.spec")
    adapter = getattr(module, "ADAPTER", None)
    if adapter is None:
        raise AttributeError(f"{slug}/spec.py defines no ADAPTER")
    return adapter


def card_discrepancies(slug: str) -> list[str]:
    """Where a card disagrees with its own `spec.py`.

    Checked: the input mapping (`"<model name> <- <canonical>"`), the output
    names with their task and activation, the framework, the artifact list
    and the ensemble size. A card that drifts from the code is worse than no
    card, so a test fails on any entry here.
    """
    card = read_card(slug)["labelmaker"]
    adapter = load_adapter(slug)
    out: list[str] = []
    want_in = [f"{f.model_name} <- {f.canonical}" for f in adapter.input_spec.fields]
    got_in = list(card.get("inputs") or [])
    if got_in != want_in:
        out.append(f"inputs: card {got_in} != spec {want_in}")
    want_out = [
        {"name": f.name, "task": f.task, "activation": f.activation}
        for f in adapter.output_spec.fields
    ]
    got_out = [
        {"name": d.get("name"), "task": d.get("task"), "activation": d.get("activation")}
        for d in (card.get("outputs") or [])
    ]
    if got_out != want_out:
        out.append(f"outputs: card {got_out} != spec {want_out}")
    if card.get("framework") != adapter.framework:
        out.append(f"framework: card {card.get('framework')} != {adapter.framework}")
    if card.get("card_id") != adapter.card_id:
        out.append(f"card_id: card {card.get('card_id')} != {adapter.card_id}")
    artifacts = list((card.get("upstream") or {}).get("artifacts") or [])
    if artifacts != list(adapter.artifacts):
        out.append(f"artifacts: card {artifacts} != spec {list(adapter.artifacts)}")
    if int(card.get("ensemble_n") or 1) != adapter.ensemble_n:
        out.append(f"ensemble_n: card {card.get('ensemble_n')} != {adapter.ensemble_n}")
    return out
```

- [ ] **Step 4: Write `models/README.md` (the roster)**

```markdown
# Model roster

One folder per trained model. A folder is `<slug>/` with:

| file | contents |
|---|---|
| `README.md` | HuggingFace-style model card. YAML front matter plus a custom `labelmaker:` block; the front matter is what `registry.py` parses. |
| `spec.py` | the only per-model code: name mapping, lag structure, training domain, output columns, loader. Exposes `ADAPTER`. |
| `__init__.py` | empty. |

## Naming

Slug grammar: `<device>_<phenomenon>_<what-is-predicted>_<architecture>[_<variant>]`,
lower snake case for the folder and Python module. The card id is the same slug
hyphenated under the `plasmacontrol/` namespace, which is what a HuggingFace repo
would be called. Label groups in `<shot>_labels.h5` use the folder name.

| Folder | Card id | Task | Status |
|---|---|---|---|
| `d3d_tearing_onset_cnn1d` | `plasmacontrol/d3d-tearing-onset-cnn1d` | binary + regression | implemented |
| `d3d_elm_time_to_event_dsm` | `plasmacontrol/d3d-elm-time-to-event-dsm` | survival | scaffold |
| `d3d_tearing_time_to_event_dsm` | `plasmacontrol/d3d-tearing-time-to-event-dsm` | survival | scaffold |
| `d3d_ech_beam_fate_mlp` | `plasmacontrol/d3d-ech-beam-fate-mlp` | 3-class + regression | scaffold |
| `d3d_ech_deposition_torbeamnn` | `plasmacontrol/d3d-ech-deposition-torbeamnn` | regression | scaffold |
| `d3d_kinetic_equilibrium_rtcakenn` | `plasmacontrol/d3d-kinetic-equilibrium-rtcakenn` | profile regression | scaffold |
| `d3d_inpa_image_cnn` | `plasmacontrol/d3d-inpa-image-cnn` | image regression | scaffold |

`status: scaffold` means the folder documents a model that labelmaker cannot run
yet; its card's `blocked_on` list says exactly what is missing, and importing its
`spec.py` raises `NotImplementedError`.

## Deliberately excluded

Recorded so they are not re-added by mistake:

- **TokEye** (`tokeye_unet`) and **`ae_tf_maskrcnn`** - spectrogram-to-pixel-mask
  models. A different I/O contract (image in, mask out) and they already ship as
  their own packaged application.
- **TokaMind** (`tokamind_base_v2`) - MAST-pretrained; its tokenizer and inverse
  decode live outside the saved graph.
- **diag2diag** - excluded at the project owner's direction. No technical reason
  was recorded, so do not infer one; ask before re-adding it.
- Anything from `tokamak_deploy_bench`'s `models/` directory. That repo is a
  latency benchmark: it feeds random noise to models and stores only timings. It
  is a useful *index* of what exists (`MODEL_ROSTER.md`), never a source of
  weights. Labelmaker loads every model from its upstream source of truth.
```

- [ ] **Step 5: Write the six scaffold folders**

For each row of the table below, create `models/<slug>/__init__.py` (empty),
`models/<slug>/spec.py`, and `models/<slug>/README.md`.

Two values the table does not carry, because they are derived from the slug and
must be derived the same way every time (`test_registry.py` asserts the first):

- `<card_id>` is `plasmacontrol/` followed by the slug with every underscore
  replaced by a hyphen. For `d3d_elm_time_to_event_dsm` that is
  `plasmacontrol/d3d-elm-time-to-event-dsm`.
- `<card_id without the namespace>`, used as the `model-index` `name`, is just
  the hyphenated slug: `d3d-elm-time-to-event-dsm`.

`spec.py`, with `<slug>`, `<upstream>` and `<blocked>` substituted:

```python
"""<slug> - scaffold. See README.md for what this model is and what blocks it."""

raise NotImplementedError(
    "<slug> is a scaffold: no adapter yet. "
    "Upstream weights: <upstream>. "
    "Blocked on: <blocked>"
)
```

`README.md`, with the table's values substituted:

```markdown
---
language: en
license: other
library_name: <library_name>
pipeline_tag: <pipeline_tag>
tags:
<tags as a YAML list, two-space indent, "  - " per item>
datasets:
  - plasmacontrol/d3d-faith-corpus
metrics:
<metrics>
model-index:
  - name: <card_id without the namespace>
    results: []
labelmaker:
  status: scaffold
  slug: <slug>
  card_id: <card_id>
  framework: <framework>
  time_step_ms: null
  ensemble_n: 1
  upstream:
    path: <upstream>
    artifacts: <artifacts as a YAML list, or [] if not yet identified>
    sha256: {}
  inputs: []
  outputs: []
  approximations: []
  blocked_on:
<blocked_on as a YAML list, two-space indent>
---

# <card_id>

**Status: scaffold.** Labelmaker cannot run this model yet. Nothing here loads
weights and no labels are produced; the folder exists so the roster, the naming
scheme and the known upstream location are recorded in one place.

## Model details

<description>

## Uses

Intended use is offline label generation over the FAITH shot corpus, for
comparison against IGNITE and against other models' labels. Not for real-time
control and not for physics conclusions without the reliability numbers in the
Evaluation section.

## Bias, risks and limitations

Unmeasured. This model has not been run through labelmaker's validation, so
nothing is known here about how its labels behave on corpus shots.

## Training details

Upstream, outside this repository: `<upstream>`. Recovering the trained-on
feature list and preprocessing constants is part of the work listed in
`blocked_on`.

## Evaluation

None yet. When implemented, `python -m labelmaker.run validate --models <slug>`
writes adapter fidelity, reconstruction fidelity and label quality into
`model-index` above.

## Technical specifications

<technical>

## Citation

Unpublished internal model. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
```

`<metrics>` per row, as a YAML list with two-space indent - a fixed `roc_auc` on
every card would assert a meaningless evaluation commitment on the pure-regression
models, which a light-touch reader could mistake for a real one:

| slug | `<metrics>` |
|---|---|
| `d3d_elm_time_to_event_dsm` | `  - concordance_index` |
| `d3d_tearing_time_to_event_dsm` | `  - concordance_index` |
| `d3d_ech_beam_fate_mlp` | `  - roc_auc`<br>`  - rmse` |
| `d3d_ech_deposition_torbeamnn` | `  - rmse` |
| `d3d_kinetic_equilibrium_rtcakenn` | `  - rmse` |
| `d3d_inpa_image_cnn` | `  - rmse` |

Substitution table:

| slug | library_name | pipeline_tag | tags | framework | upstream | artifacts | blocked_on | description | technical |
|---|---|---|---|---|---|---|---|---|---|
| `d3d_elm_time_to_event_dsm` | `keras` | `time-series-forecasting` | diii-d, tokamak, elm, survival-analysis | `keras` | `/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/` | `wpqh1_embedding_with_norm.keras`, `wpqh1_gate.keras`, `wpqh1_scaleg.keras`, `wpqh1_shapeg.keras` | "recover the ~124 trained-on feature names and their order from `hiro_scripts/`"; "decide how a deep-survival-machines head (gate, scale, shape) becomes a label series: hazard at fixed horizons, or expected time to event"; "the embedding network bakes in its own input normalisation, so the constants need no recovery - confirm that" | Predicts time to the next ELM from a wide 0-D and profile feature vector, as a mixture of Weibull distributions (deep survival machines). Four saved graphs: an embedding network with normalisation, and gate, scale and shape heads. | Embedding 143,276 parameters; gate/scale/shape ~3,000 each. Legacy Keras format; `runners/keras_h5.py` may need additional layer support. |
| `d3d_tearing_time_to_event_dsm` | `keras` | `time-series-forecasting` | diii-d, tokamak, tearing-mode, survival-analysis | `keras` | `/projects/EKOLEMEN/survival_tm/` | `[]` | "three candidate directories exist (`survival_tm`, `survival_tm_2`, `survival_tm_depreciated`) - identify the production checkpoint"; "recover the trained-on feature list"; "same survival-head-to-label-series question as the ELM model" | Time to tearing-mode onset as a survival problem, the same family as the ELM model. Complements `d3d_tearing_onset_cnn1d`, which answers the fixed-horizon binary question instead. | Not yet inspected. |
| `d3d_ech_beam_fate_mlp` | `keras` | `tabular-classification` | diii-d, tokamak, ech, multitask | `keras` | `/projects/EKOLEMEN/ECH_interlock/models_v15/` | `multitask_skip.keras`, `norm_stats.npy` | "the file is a legacy Keras-2 HDF5 containing `TFOpLambda`, which `runners/keras_h5.py` does not implement - add that layer or re-save the graph once via `tf_keras`"; "its profiles are on a 101-point grid, so `ne` and `Te` need their own canonical features rather than the 33-point `RHO_GRID`"; "confirm whether the normalisation constants in `norm_stats.npy` are already baked into the graph" | Classifies the fate of an ECH beam (absorbed, shine-through, or reflected) and regresses absorption efficiency, deposition height and toroidal angle. Four inputs: `ne` (101), `Te` (101), three scalars, fourteen EFIT scalars. | 186,726 parameters. Four outputs returned as a dict; the config order (`logits, eta, z, phi`) differs from the tensor order, which the adapter must pin explicitly. |
| `d3d_ech_deposition_torbeamnn` | `keras` | `tabular-regression` | diii-d, tokamak, ech, torbeam, surrogate | `keras` | `/projects/EKOLEMEN/torbeamNN/models_v2/` | `s1_omode.h5`, `s1_xmode.h5` | "recover the input list and ordering from the torbeamNN training code"; "O-mode and X-mode are separate graphs, so the adapter needs the per-gyrotron polarisation, which the corpus does carry as `ech_polarization`"; "decide whether to emit one label per gyrotron or one aggregate" | A neural surrogate for the TORBEAM ray-tracing code: given the equilibrium, profiles and launcher geometry, predicts where ECH power deposits and how much is absorbed. Would give a physics-consistent ECH deposition label for every corpus shot with ECH. | 748,735 (O-mode) and 750,040 (X-mode) parameters. A third, smaller graph (`models/mini_torbeamNN_model_0.h5`, 32,454 parameters) exists and may be the better choice for bulk inference. |
| `d3d_kinetic_equilibrium_rtcakenn` | `keras` | `tabular-regression` | diii-d, tokamak, equilibrium, cake, surrogate | `keras` | `/projects/EKOLEMEN/rtcakenn_optimization/` | `[]` | "four candidate directories exist (`fall2024_rtcakenn`, `rtcakenn_optimization`, `rscake_nn`, `cake_nn`) - identify the production checkpoint and its author"; "recover the input list"; "profile outputs need canonical *output* features, which the label schema does not yet cover (it stores `(C, T)` per label, so a profile label is `C = n_rho`)" | A real-time surrogate for CAKE kinetic-equilibrium reconstruction. Its outputs are profiles rather than scalars, which makes it the roster's test of whether the label format handles profile-valued labels. | Not yet inspected. |
| `d3d_inpa_image_cnn` | `keras` | `image-to-image` | diii-d, tokamak, inpa, fast-ion, cnn | `keras` | `/projects/EKOLEMEN/agarcia/inpa_net/` | `cnn.keras` | "**there is no input source**: the corpus has no INPA group, so no corpus shot can be fed to this model. Label generation is impossible until INPA frames are added to the corpus or fetched separately"; "recover the expected image size and preprocessing" | Maps INPA (Imaging Neutral Particle Analyzer) camera frames to a fast-ion phase-space quantity. Listed for completeness of the roster; it is the one candidate whose inputs the corpus does not contain at all. | 1,441,344 parameters. Legacy Keras format. |

- [ ] **Step 6: Run the test to verify it passes**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_registry.py -q
```
Expected: `6 passed`. `test_implemented_models_load_and_match_their_card` passes trivially here (no implemented model yet) and becomes meaningful in Task 8.

- [ ] **Step 7: Check the cards render as YAML the way a card reader would see them**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
from labelmaker.models import registry
for slug in registry.model_slugs():
    card = registry.read_card(slug)
    lm = card["labelmaker"]
    print(f"{slug:34s} {lm['status']:12s} {card['pipeline_tag']:26s} "
          f"blocked_on={len(lm.get('blocked_on') or [])}")
PY
```
Expected: six rows, all `scaffold`, each with a non-zero `blocked_on` count.

- [ ] **Step 8: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/models/registry.py src/labelmaker/models/README.md \
        src/labelmaker/models/d3d_elm_time_to_event_dsm \
        src/labelmaker/models/d3d_tearing_time_to_event_dsm \
        src/labelmaker/models/d3d_ech_beam_fate_mlp \
        src/labelmaker/models/d3d_ech_deposition_torbeamnn \
        src/labelmaker/models/d3d_kinetic_equilibrium_rtcakenn \
        src/labelmaker/models/d3d_inpa_image_cnn \
        tests/labelmaker/test_registry.py
git commit -m "labelmaker: model registry, card format, and the scaffolded roster"
```

---

### Task 8: The tearing-onset adapter

**Files:**
- Create: `src/labelmaker/models/d3d_tearing_onset_cnn1d/__init__.py`, `spec.py`, `README.md`
- Modify: `src/labelmaker/models/base.py` (add the `nonneg_zero_fill` transform)
- Modify: `src/labelmaker/models/registry.py` (add `sha256_of`, `verify_artifacts`)
- Test: `tests/labelmaker/test_tearing_adapter.py`

**Interfaces:**
- Consumes: `models.base` (all dataclasses), `models.runners.keras_h5.load_ensemble`, `predict_members`.
- Produces: `d3d_tearing_onset_cnn1d.spec.ADAPTER: ModelAdapter`, `spec.ARTIFACTS: tuple[str, ...]`, `spec.UPSTREAM: Path`, `spec.load(model_dir) -> Callable[[BuiltInputs], np.ndarray]`; `registry.sha256_of(path) -> str`; `registry.verify_artifacts(slug, model_dir) -> None`.

**Upstream facts this task encodes** (verified 2026-09-03):
- Weights: `/projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w/best_model_{0..9}_4c.h5`, Keras 2.8, ten members, two inputs `input_1 (None, 11)` and `input_2 (None, 33, 5)`, one output `dense_4 (None, 2)`.
- Column 0 is `betan` (linear). Column 1 is the tearing logit; the head was trained with `from_logits=True`, so a sigmoid is applied on read.
- The first layer of each branch is a `BatchNormalization` carrying the training-set moving statistics, so **there are no external normalisation constants to recover**.
- Input order is `train.py:31-32` verbatim. 0-D inputs are at `t+dt`, profiles at `t`, `dt = 25 ms`.
- The domain rules are the `idx` filter of `train.py:83`.

- [ ] **Step 1: Write the failing test**

`tests/labelmaker/test_tearing_adapter.py`:

```python
"""The Phase 1 adapter: name mapping, lags, transforms, domain, decode."""
from pathlib import Path

import numpy as np
import pytest

from labelmaker.features import namespace as ns
from labelmaker.features.store import FeatureArray
from labelmaker.models import registry
from labelmaker.models.d3d_tearing_onset_cnn1d import spec as tm

UPSTREAM = Path(
    "/projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w"
)

# train.py:31-32, verbatim.
INPUTS_0D = [
    "bt", "ip", "pinj", "tinj", "R0_EFITRT1", "kappa_EFITRT1",
    "tritop_EFIT01", "tribot_EFIT01", "gapin_EFIT01", "ech_pwr_total",
    "EC.RHO_ECH",
]
INPUTS_1D = [
    "thomson_density_mtanh_1d", "thomson_temp_mtanh_1d", "1/qpsi_EFITRT1",
    "pres_EFIT01", "cer_rot_csaps_1d",
]


def test_field_order_is_the_upstream_column_order():
    got_0d = [f.model_name for f in tm.ADAPTER.input_spec.scalar_fields]
    got_1d = [f.model_name for f in tm.ADAPTER.input_spec.profile_fields]
    assert got_0d == INPUTS_0D
    assert got_1d == INPUTS_1D


def test_lags_and_step():
    assert tm.ADAPTER.input_spec.dt_s == 0.025
    assert tm.ADAPTER.time_step_ms == 25.0
    assert all(f.lag == "t+dt" for f in tm.ADAPTER.input_spec.scalar_fields)
    assert all(f.lag == "t" for f in tm.ADAPTER.input_spec.profile_fields)


def test_outputs_are_betan_then_the_tearing_logit():
    fields = tm.ADAPTER.output_spec.fields
    assert [f.name for f in fields] == ["betan", "tm_prob"]
    assert [f.column for f in fields] == [0, 1]
    assert [f.activation for f in fields] == ["none", "sigmoid"]
    assert [f.task for f in fields] == ["regression", "binary"]


def test_every_upstream_filter_clause_is_a_domain_rule():
    rules = {(r.canonical, r.stat): r for r in tm.ADAPTER.input_spec.domain}
    assert rules[("ne_zipfit", "min")].lo == 0.0
    assert rules[("ne_zipfit", "min")].lo_inclusive
    assert rules[("ne_zipfit", "max")].hi == 12.0
    assert rules[("te_zipfit", "max")].hi == 10.0
    assert rules[("qpsi", "max")].hi == 3.0
    assert rules[("pres", "max")].lo == 0.0 and rules[("pres", "max")].hi == 2e5
    assert rules[("rot_zipfit", "absmax")].hi == 150.0
    assert (rules[("r0", "value")].lo, rules[("r0", "value")].hi) == (1.65, 1.9)
    assert (rules[("kappa", "value")].lo, rules[("kappa", "value")].hi) == (1.6, 2.0)
    assert (rules[("tritop", "value")].lo, rules[("tritop", "value")].hi) == (0.0, 1.0)
    assert (rules[("tribot", "value")].lo, rules[("tribot", "value")].hi) == (0.0, 1.0)
    assert rules[("gapin", "value")].hi == 0.2
    assert rules[("ech_rho", "value")].lo == 0.0
    assert rules[("ech_rho", "value")].lo_inclusive


def _features(n=8, *, ech_nan=False, rho_nan=False):
    t = ns.STEP_S * np.arange(n + 1)          # one extra step for the t+dt lag
    scalars = {
        "bt": 2.0, "ip": 1.0e6, "pinj_total": 5000.0, "tinj_total": 4.0,
        "r0": 1.75, "kappa": 1.8, "tritop": 0.4, "tribot": 0.3,
        "gapin": 0.05, "ech_power_total": 1.0e6, "ech_rho": 0.4,
    }
    out = {
        name: FeatureArray(x=t, y=np.full((1, t.size), v), attrs={"resolver": "archive"})
        for name, v in scalars.items()
    }
    if ech_nan:
        y = np.full((1, t.size), np.nan)
        out["ech_power_total"] = FeatureArray(x=t, y=y, attrs={"resolver": "archive"})
    if rho_nan:
        y = np.full((1, t.size), np.nan)
        out["ech_rho"] = FeatureArray(x=t, y=y, attrs={"resolver": "archive"})
    profiles = {
        "ne_zipfit": 3.0, "te_zipfit": 2.0, "qpsi": 2.5, "pres": 5.0e4,
        "rot_zipfit": 50.0,
    }
    for name, v in profiles.items():
        out[name] = FeatureArray(
            x=t, y=np.full((33, t.size), v), attrs={"resolver": "archive"}
        )
    return out, ns.STEP_S * np.arange(n)


def test_build_produces_the_model_input_shapes():
    feats, grid = _features()
    built = tm.ADAPTER.input_spec.build(feats, grid)
    assert built.scalars.shape == (8, 11)
    assert built.profiles.shape == (8, 33, 5)
    assert built.valid.all()
    assert built.missing == ()
    np.testing.assert_allclose(built.scalars[:, 0], 2.0)          # bt
    np.testing.assert_allclose(built.profiles[:, :, 2], 1 / 2.5)  # 1/qpsi
    np.testing.assert_allclose(built.profiles[:, :, 3], 5.0e4)    # pres, unscaled


def test_nan_ech_power_becomes_zero_without_invalidating_the_row():
    # Upstream rule (train.py:81): NaN or negative ECH power is set to zero
    # *before* the filter runs, so those rows survived training.
    feats, grid = _features(ech_nan=True)
    built = tm.ADAPTER.input_spec.build(feats, grid)
    np.testing.assert_allclose(built.scalars[:, 9], 0.0)
    assert built.valid.all()


def test_negative_ech_power_also_becomes_zero():
    feats, grid = _features()
    t = feats["ech_power_total"].x
    feats["ech_power_total"] = FeatureArray(
        x=t, y=np.full((1, t.size), -1.0e5), attrs={"resolver": "archive"}
    )
    built = tm.ADAPTER.input_spec.build(feats, grid)
    np.testing.assert_allclose(built.scalars[:, 9], 0.0)
    assert built.valid.all()


def test_missing_ech_deposition_location_becomes_zero_which_is_in_domain():
    feats, grid = _features(rho_nan=True)
    built = tm.ADAPTER.input_spec.build(feats, grid)
    np.testing.assert_allclose(built.scalars[:, 10], 0.0)
    assert built.valid.all()


def test_out_of_domain_kappa_is_flagged():
    feats, grid = _features()
    t = feats["kappa"].x
    y = np.full((1, t.size), 1.8)
    y[0, 4] = 1.2                                   # below the 1.6 floor
    feats["kappa"] = FeatureArray(x=t, y=y, attrs={"resolver": "archive"})
    built = tm.ADAPTER.input_spec.build(feats, grid)
    # kappa is a t+dt field, so grid step i reads feature sample i+1:
    # spoiling sample 4 flags grid step 3, not grid step 4.
    assert built.valid.sum() == 7 and not built.valid[3]


pytestmark_upstream = pytest.mark.skipif(
    not UPSTREAM.exists(), reason=f"upstream weights not available: {UPSTREAM}"
)


@pytestmark_upstream
def test_card_matches_the_spec():
    assert registry.card_discrepancies("d3d_tearing_onset_cnn1d") == []
    assert registry.implemented() == ["d3d_tearing_onset_cnn1d"]


@pytestmark_upstream
def test_predict_runs_end_to_end_from_the_upstream_directory():
    predict = tm.load(UPSTREAM)
    feats, grid = _features()
    built = tm.ADAPTER.input_spec.build(feats, grid)
    members = predict(built)
    assert members.shape == (10, 8, 2)
    assert np.isfinite(members).all()
    decoded = tm.ADAPTER.output_spec.decode(members)
    assert set(decoded) == {"betan", "tm_prob"}
    assert ((decoded["tm_prob"].mean >= 0) & (decoded["tm_prob"].mean <= 1)).all()
    assert (decoded["tm_prob"].lo <= decoded["tm_prob"].mean).all()
    assert (decoded["tm_prob"].hi >= decoded["tm_prob"].mean).all()


@pytestmark_upstream
def test_sha256_verification_rejects_a_tampered_artifact(tmp_path):
    import shutil

    for name in tm.ARTIFACTS:
        shutil.copy(UPSTREAM / name, tmp_path / name)
    registry.verify_artifacts("d3d_tearing_onset_cnn1d", tmp_path)  # passes
    with open(tmp_path / tm.ARTIFACTS[0], "ab") as fh:
        fh.write(b"\x00")
    with pytest.raises(ValueError, match="sha256"):
        registry.verify_artifacts("d3d_tearing_onset_cnn1d", tmp_path)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_tearing_adapter.py -q
```
Expected: `ModuleNotFoundError: No module named 'labelmaker.models.d3d_tearing_onset_cnn1d'`.

- [ ] **Step 3: Add the upstream ECH rule as a named transform**

In `src/labelmaker/models/base.py`, extend `TRANSFORMS`:

```python
TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "reciprocal": lambda a: 1.0 / a,
    # train.py:81 - NaN or negative ECH power was set to zero before the
    # training filter ran, so those rows were kept. Applied before the
    # validity flags so a zeroed ECH channel does not invalidate a row.
    "nonneg_zero_fill": lambda a: np.where(np.isfinite(a) & (a > 0.0), a, 0.0),
}
```

- [ ] **Step 4: Write `spec.py`**

`src/labelmaker/models/d3d_tearing_onset_cnn1d/__init__.py`: empty file.

`src/labelmaker/models/d3d_tearing_onset_cnn1d/spec.py`:

```python
"""d3d_tearing_onset_cnn1d - tearing-mode onset 25 ms ahead, plus betan.

Upstream: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
(ten Keras 2.8 members, `best_model_{i}_4c.h5`), training code `../train.py`,
reference harness `../../test/test.py`.

The model takes eleven 0-D quantities at `t + 25 ms` and five 33-point
profiles at `t`, and emits two columns: `betan` and a tearing logit. Input
names and their order are `train.py:31-32` verbatim; the domain rules are
`train.py:83`. Each branch starts with a BatchNormalization holding the
training-set moving statistics, so no external scaler is needed.

Substitutions, all measured in validation/d3d_tearing_onset_cnn1d/:
  R0_EFITRT1, kappa_EFITRT1, 1/qpsi_EFITRT1  <- offline EFIT01 equivalents
  thomson_*_mtanh_1d, cer_rot_csaps_1d       <- ZIPFIT fitted profiles
"""
from __future__ import annotations

from pathlib import Path

from ...features import namespace as ns
from ..base import (
    DomainRule,
    InputField,
    InputSpec,
    ModelAdapter,
    OutputField,
    OutputSpec,
)
from ..runners import keras_h5

SLUG = "d3d_tearing_onset_cnn1d"
CARD_ID = "plasmacontrol/d3d-tearing-onset-cnn1d"
UPSTREAM = Path(
    "/projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w"
)
ARTIFACTS = tuple(f"best_model_{i}_4c.h5" for i in range(10))
DT_S = 0.025

INPUT_SPEC = InputSpec(
    fields=(
        # 0-D block, at t + dt. Order is x0's column order.
        InputField("bt", "bt", lag="t+dt"),
        InputField("ip", "ip", lag="t+dt"),
        InputField("pinj", "pinj_total", lag="t+dt"),
        InputField("tinj", "tinj_total", lag="t+dt"),
        InputField("R0_EFITRT1", "r0", lag="t+dt"),
        InputField("kappa_EFITRT1", "kappa", lag="t+dt"),
        InputField("tritop_EFIT01", "tritop", lag="t+dt"),
        InputField("tribot_EFIT01", "tribot", lag="t+dt"),
        InputField("gapin_EFIT01", "gapin", lag="t+dt"),
        InputField(
            "ech_pwr_total", "ech_power_total", lag="t+dt",
            transform="nonneg_zero_fill",
        ),
        InputField(
            "EC.RHO_ECH", "ech_rho", lag="t+dt", transform="nonneg_zero_fill",
        ),
        # profile block, at t. Order is x1's channel order.
        InputField("thomson_density_mtanh_1d", "ne_zipfit", lag="t"),
        InputField("thomson_temp_mtanh_1d", "te_zipfit", lag="t"),
        InputField("1/qpsi_EFITRT1", "qpsi", lag="t", transform="reciprocal"),
        InputField("pres_EFIT01", "pres", lag="t"),
        InputField("cer_rot_csaps_1d", "rot_zipfit", lag="t"),
    ),
    dt_s=DT_S,
    rho_grid=ns.RHO_GRID,
    nan_policy="zero",
    # train.py:83, clause by clause. Rows outside these ranges are labelled
    # anyway and flagged: the model never saw such states in training.
    domain=(
        DomainRule("ne_zipfit", "min", lo=0.0, lo_inclusive=True),
        DomainRule("ne_zipfit", "max", hi=12.0),
        DomainRule("te_zipfit", "min", lo=0.0, lo_inclusive=True),
        DomainRule("te_zipfit", "max", hi=10.0),
        DomainRule("qpsi", "max", hi=3.0),
        DomainRule("pres", "min", lo=0.0, lo_inclusive=True),
        DomainRule("pres", "max", lo=0.0, hi=2.0e5),
        DomainRule("rot_zipfit", "absmax", hi=150.0),
        DomainRule("r0", "value", lo=1.65, hi=1.9),
        DomainRule("kappa", "value", lo=1.6, hi=2.0),
        DomainRule("tritop", "value", lo=0.0, hi=1.0),
        DomainRule("tribot", "value", lo=0.0, hi=1.0),
        DomainRule("gapin", "value", hi=0.2),
        DomainRule("ech_rho", "value", lo=0.0, lo_inclusive=True),
    ),
)

OUTPUT_SPEC = OutputSpec(
    fields=(
        OutputField("betan", "regression", column=0, activation="none", units=""),
        OutputField("tm_prob", "binary", column=1, activation="sigmoid",
                    classes=("no_tearing", "tearing")),
    )
)


def load(model_dir):
    """Load the ten members once and return a predictor over BuiltInputs."""
    graphs = keras_h5.load_ensemble(Path(model_dir) / name for name in ARTIFACTS)

    def predict(built):
        # Positional, not keyed by name: the ten members carry different
        # layer names (input_1/input_2 .. input_19/input_20), so only the
        # order - 0-D block first, profile block second - is common.
        return keras_h5.predict_members(graphs, [built.scalars, built.profiles])

    return predict


ADAPTER = ModelAdapter(
    slug=SLUG,
    card_id=CARD_ID,
    framework="keras_h5",
    time_step_ms=DT_S * 1000.0,
    artifacts=ARTIFACTS,
    upstream=str(UPSTREAM),
    input_spec=INPUT_SPEC,
    output_spec=OUTPUT_SPEC,
    load=load,
    ensemble_n=len(ARTIFACTS),
)
```

- [ ] **Step 5: Add artifact verification to `registry.py`**

Append to `src/labelmaker/models/registry.py`:

```python
def sha256_of(path) -> str:
    """Hex digest of a file, read in 1 MiB blocks."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_artifacts(slug: str, model_dir) -> None:
    """Raise unless every artifact in `model_dir` matches the card's sha256.

    Labels are only worth what the weights behind them are, so inference
    refuses to run against unexpected bytes rather than silently producing
    a label file whose provenance is wrong.
    """
    card = read_card(slug)["labelmaker"]
    expected = (card.get("upstream") or {}).get("sha256") or {}
    if not expected:
        raise ValueError(f"{slug}: card records no sha256 for its artifacts")
    model_dir = Path(model_dir)
    problems = []
    for name, want in sorted(expected.items()):
        path = model_dir / name
        if not path.exists():
            problems.append(f"{name}: missing from {model_dir}")
            continue
        got = sha256_of(path)
        if got != want:
            problems.append(f"{name}: sha256 {got[:12]} != card {want[:12]}")
    if problems:
        raise ValueError(f"{slug}: artifact sha256 mismatch: " + "; ".join(problems))
```

- [ ] **Step 6: Copy the weights into the data root and record provenance**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
import json, shutil
from datetime import UTC, datetime
from pathlib import Path

from labelmaker.config import Paths
from labelmaker.models import registry
from labelmaker.models.d3d_tearing_onset_cnn1d import spec as tm

dest = Paths.from_env().models / tm.SLUG
dest.mkdir(parents=True, exist_ok=True)
records = {}
for name in tm.ARTIFACTS:
    src = tm.UPSTREAM / name
    shutil.copy2(src, dest / name)
    records[name] = {
        "sha256": registry.sha256_of(dest / name),
        "bytes": src.stat().st_size,
        "upstream_mtime": datetime.fromtimestamp(
            src.stat().st_mtime, timezone.utc
        ).isoformat(timespec="seconds"),
    }
(dest / "PROVENANCE.json").write_text(
    json.dumps(
        {
            "slug": tm.SLUG,
            "card_id": tm.CARD_ID,
            "upstream": str(tm.UPSTREAM),
            "copied_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "artifacts": records,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(f"copied {len(records)} artifacts to {dest}")
for name, rec in sorted(records.items()):
    print(f"  {name}  {rec['sha256']}")
PY
```
Expected: ten lines with 64-character digests. Paste those digests into the card's
`upstream.sha256` map in Step 7 — they are what `verify_artifacts` checks.

- [ ] **Step 7: Write the model card**

`src/labelmaker/models/d3d_tearing_onset_cnn1d/README.md`. Replace each
`<sha256 from step 6>` with the matching digest printed above.

```markdown
---
language: en
license: other
library_name: keras
pipeline_tag: tabular-classification
tags:
  - diii-d
  - tokamak
  - plasma
  - tearing-mode
  - neoclassical-tearing-mode
  - cnn
datasets:
  - plasmacontrol/d3d-faith-corpus
metrics:
  - roc_auc
  - f1
model-index:
  - name: d3d-tearing-onset-cnn1d
    results: []
labelmaker:
  status: implemented
  slug: d3d_tearing_onset_cnn1d
  card_id: plasmacontrol/d3d-tearing-onset-cnn1d
  framework: keras_h5
  time_step_ms: 25.0
  ensemble_n: 10
  upstream:
    path: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
    trained: 2022-12
    training_code: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/train.py
    reference_harness: /projects/EKOLEMEN/simple_ae_predictor/test/test.py
    artifacts:
      - best_model_0_4c.h5
      - best_model_1_4c.h5
      - best_model_2_4c.h5
      - best_model_3_4c.h5
      - best_model_4_4c.h5
      - best_model_5_4c.h5
      - best_model_6_4c.h5
      - best_model_7_4c.h5
      - best_model_8_4c.h5
      - best_model_9_4c.h5
    sha256:
      best_model_0_4c.h5: <sha256 from step 6>
      best_model_1_4c.h5: <sha256 from step 6>
      best_model_2_4c.h5: <sha256 from step 6>
      best_model_3_4c.h5: <sha256 from step 6>
      best_model_4_4c.h5: <sha256 from step 6>
      best_model_5_4c.h5: <sha256 from step 6>
      best_model_6_4c.h5: <sha256 from step 6>
      best_model_7_4c.h5: <sha256 from step 6>
      best_model_8_4c.h5: <sha256 from step 6>
      best_model_9_4c.h5: <sha256 from step 6>
  inputs:
    - "bt <- bt"
    - "ip <- ip"
    - "pinj <- pinj_total"
    - "tinj <- tinj_total"
    - "R0_EFITRT1 <- r0"
    - "kappa_EFITRT1 <- kappa"
    - "tritop_EFIT01 <- tritop"
    - "tribot_EFIT01 <- tribot"
    - "gapin_EFIT01 <- gapin"
    - "ech_pwr_total <- ech_power_total"
    - "EC.RHO_ECH <- ech_rho"
    - "thomson_density_mtanh_1d <- ne_zipfit"
    - "thomson_temp_mtanh_1d <- te_zipfit"
    - "1/qpsi_EFITRT1 <- qpsi"
    - "pres_EFIT01 <- pres"
    - "cer_rot_csaps_1d <- rot_zipfit"
  outputs:
    - name: betan
      task: regression
      activation: none
      units: ""
    - name: tm_prob
      task: binary
      activation: sigmoid
      units: ""
  approximations:
    - "R0_EFITRT1 and kappa_EFITRT1 are served by offline EFIT01 (rmaxis, kappa);
      measured median relative difference 8.8e-3 and 3.1e-3 on shot 185945"
    - "1/qpsi_EFITRT1 is served by offline EFIT01 qpsi; measured 6.4e-2"
    - "the three kinetic profiles are ZIPFIT fits, not the pipeline's own mtanh
      and csaps fits; measured 2.0e-1 (ne), 1.8e-1 (Te), 1.7e-1 (rotation)
      median relative difference, correlation 0.98-0.99"
    - "NaN or negative ECH power becomes 0 (upstream rule, train.py:81)"
    - "a missing EC.RHO_ECH becomes 0, which the training filter admitted;
      validation reports how often that happens while ECH power is non-zero"
    - "inference evaluates the Keras graph in numpy (models/runners/keras_h5.py);
      equality with TensorFlow is checked to 1e-5 in validation"
---

# plasmacontrol/d3d-tearing-onset-cnn1d

Predicts, for a DIII-D discharge, the probability that a tearing mode is present
25 ms from now, together with the normalised beta at the same instant.

## Model details

A ten-member ensemble of small multi-input networks (12,086 parameters each).
The profile branch is two `Conv1D` layers over the 33-point radial axis, pooled
and compressed to four numbers; those are concatenated with the eleven 0-D
inputs and passed through three dense layers to a two-column output. Every
block is preceded by a `BatchNormalization` carrying training-set statistics, so
the graph normalises its own inputs.

- Developed by: PlasmaControl group, Princeton (upstream author recorded in
  `PROVENANCE.json`)
- Model type: 1-D CNN plus MLP, multi-input, multi-task
- Trained: December 2022, on 639,555 filtered timeslices from 8,505 DIII-D shots
  (147000-190997)
- Tearing-positive fraction in training: 7.9%

## Uses

Offline label generation over the FAITH shot corpus: a per-shot tearing-mode
probability series at 25 ms resolution that can be compared against IGNITE and
against other models. The `_valid` companion series says where the inputs were
inside the training domain; a probability outside it is an extrapolation.

Not intended for real-time control (the real-time variant of this model is a
different artifact) and not a substitute for magnetics-based mode detection.

## Bias, risks and limitations

- The label answers "is a tearing mode present at t+25 ms", not "will one appear"
  - a mode already present is the easy majority of positives.
- Trained on 2011-2021 shots; corpus shots beyond 190997 are outside the
  training shot range even when their parameters are in domain.
- The three kinetic profiles are substituted (see `approximations`), which is
  the dominant reconstruction error. The measured effect on label quality is in
  the Evaluation section.
- `betan` is a secondary head; the upstream training filter kept only rows with
  `0 < betan < 5`, so predictions far outside that band are unreliable.

## Training details

Preprocessing, filtering and the ensemble loop are `train.py` in the upstream
directory. Loss: MSE on `betan`, binary cross-entropy `from_logits=True` on the
tearing column, with oversampling and class weighting (the `mse_bin_os_w`
variant). Inputs and outputs both at 25 ms; 0-D inputs read one step ahead of
the profiles.

## Evaluation

Written by `python -m labelmaker.run validate --models d3d_tearing_onset_cnn1d`
into `model-index` above and, in full, into
`<LABELMAKER_ROOT>/validation/d3d_tearing_onset_cnn1d/`:

- `adapter_fidelity.json` - numpy evaluator against TensorFlow on the upstream
  reference file, max absolute difference.
- `reconstruction.json` - per-feature agreement between labelmaker's features
  and the model's own training rows on the corpus/archive overlap shots.
- `label_quality.json` - AUROC, F1 at 0.5 and calibration against the archived
  labels, computed twice: with archived inputs and with labelmaker's inputs. The
  difference is the reconstruction penalty.

## Technical specifications

- Inputs: `input_1 (None, 11)` float32, `input_2 (None, 33, 5)` float32
- Output: `dense_4 (None, 2)` - column 0 `betan` (linear), column 1 tearing
  logit (apply sigmoid)
- Radial grid: 33 points, `rho = linspace(0, 1, 33)`
- Ensemble: mean over ten members in logit space, then the activation; the
  member min and max are stored as the label's spread
- Framework: Keras 2.8 legacy HDF5, evaluated in numpy

## Citation

Unpublished internal model. Attribute to the PlasmaControl group, Princeton.

## Contact

`nc1514@princeton.edu`.
```

- [ ] **Step 8: Run the tests to verify they pass**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_tearing_adapter.py tests/labelmaker/test_registry.py -q
```
Expected: `18 passed` (12 adapter tests plus the 6 registry tests, which now include the
implemented model).

- [ ] **Step 9: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/models/base.py src/labelmaker/models/registry.py \
        src/labelmaker/models/d3d_tearing_onset_cnn1d \
        tests/labelmaker/test_tearing_adapter.py
git commit -m "labelmaker: tearing-onset adapter, card and artifact verification"
```

---

### Task 9: Label schema, label store, and the label index

**Files:**
- Create: `src/labelmaker/labels/__init__.py`, `src/labelmaker/labels/schema.py`, `src/labelmaker/labels/store.py`
- Modify: `src/labelmaker/config.py` (add `sha256_of`), `src/labelmaker/models/registry.py` (re-export it)
- Test: `tests/labelmaker/test_label_store.py`

**Interfaces:**
- Consumes: `models.base.Decoded`, `models.base.ModelAdapter`, `config.Paths`, `config.git_sha`.
- Produces:
  - `config.sha256_of(path) -> str` (moved here from `registry`, which now re-exports it)
  - `schema.LabelSpec(name, task, activation, units, classes, slug, card_id, time_step_ms, ensemble_n, artifact_sha256)`; `schema.specs_for(adapter, artifact_sha256) -> tuple[LabelSpec, ...]`; `schema.group_path(slug, label) -> str`; `schema.artifact_digest(sha_by_name: Mapping[str, str]) -> str`
  - `store.LabelArray(x, y, attrs)`; `store.write_labels(path, shot, t, decoded, specs, valid, *, run_id, features_sha256, merge=True) -> None`; `store.read_label(path, slug, label) -> LabelArray`; `store.labelled(path) -> set[str]`; `store.index_rows(path) -> list[dict]`; `store.append_index(index_path, rows) -> None`

- [ ] **Step 1: Write the failing test**

`tests/labelmaker/test_label_store.py`:

```python
"""Label files are corpus-shaped, provenanced, and indexable."""
import h5py
import numpy as np
import pandas as pd
import pytest

from labelmaker.labels.schema import LabelSpec, artifact_digest, group_path
from labelmaker.labels.store import (
    append_index,
    index_rows,
    labelled,
    read_label,
    write_labels,
)
from labelmaker.models.base import Decoded

SLUG = "d3d_tearing_onset_cnn1d"
T = 0.025 * np.arange(6)


def _specs():
    return (
        LabelSpec(
            name="tm_prob", task="binary", activation="sigmoid", units="",
            classes=("no_tearing", "tearing"), slug=SLUG,
            card_id="plasmacontrol/d3d-tearing-onset-cnn1d",
            time_step_ms=25.0, ensemble_n=10, artifact_sha256="abc123",
        ),
        LabelSpec(
            name="betan", task="regression", activation="none", units="",
            classes=(), slug=SLUG,
            card_id="plasmacontrol/d3d-tearing-onset-cnn1d",
            time_step_ms=25.0, ensemble_n=10, artifact_sha256="abc123",
        ),
    )


def _decoded():
    p = np.linspace(0.1, 0.6, 6)
    return {
        "tm_prob": Decoded(mean=p, lo=p - 0.05, hi=p + 0.05),
        "betan": Decoded(mean=p * 5, lo=p * 5 - 0.2, hi=p * 5 + 0.2),
    }


def _write(path, valid=None):
    valid = np.ones(6, bool) if valid is None else valid
    write_labels(
        path, 190000, T, _decoded(), _specs(), valid,
        run_id="run-test", features_sha256="f" * 64,
    )


def test_group_path_and_artifact_digest_are_stable():
    assert group_path(SLUG, "tm_prob") == f"{SLUG}/tm_prob"
    a = artifact_digest({"m0.h5": "aa", "m1.h5": "bb"})
    b = artifact_digest({"m1.h5": "bb", "m0.h5": "aa"})
    assert a == b and len(a) == 64            # order-independent, sha256 hex
    assert a != artifact_digest({"m0.h5": "aa"})


def test_round_trip_in_corpus_layout(tmp_path):
    p = tmp_path / "190000_labels.h5"
    _write(p)
    got = read_label(p, SLUG, "tm_prob")
    np.testing.assert_allclose(got.x, T)
    assert got.y.shape == (1, 6)
    np.testing.assert_allclose(got.y[0], np.linspace(0.1, 0.6, 6), rtol=1e-6)
    assert got.attrs["task"] == "binary"
    assert got.attrs["card_id"] == "plasmacontrol/d3d-tearing-onset-cnn1d"
    assert labelled(p) == {f"{SLUG}/tm_prob", f"{SLUG}/betan"}


def test_spread_and_validity_companions(tmp_path):
    p = tmp_path / "190000_labels.h5"
    valid = np.array([1, 1, 0, 0, 1, 1], dtype=bool)
    _write(p, valid)
    with h5py.File(p, "r") as f:
        g = f[SLUG]
        assert g["tm_prob_spread"]["ydata"].shape == (2, 6)
        np.testing.assert_allclose(
            g["tm_prob_spread"]["ydata"][0], np.linspace(0.1, 0.6, 6) - 0.05,
            rtol=1e-6,
        )
        v = g["tm_prob_valid"]["ydata"]
        assert v.shape == (1, 6) and v.dtype == np.uint8
        assert v[0].tolist() == [1, 1, 0, 0, 1, 1]


def test_provenance_attrs_on_the_file(tmp_path):
    p = tmp_path / "190000_labels.h5"
    _write(p)
    with h5py.File(p, "r") as f:
        assert f.attrs["shot"] == 190000
        assert f.attrs["run_id"] == "run-test"
        assert f.attrs["features_sha256"] == "f" * 64
        assert f.attrs["labelmaker_version"] and f.attrs["git_sha"]
        assert f[SLUG]["tm_prob"].attrs["ensemble_n"] == 10
        assert f[SLUG]["tm_prob"].attrs["time_step_ms"] == 25.0
        assert f[SLUG]["tm_prob"].attrs["artifact_sha256"] == "abc123"
        assert list(f[SLUG]["tm_prob"].attrs["classes"]) == ["no_tearing", "tearing"]


def test_write_is_atomic_and_merges_other_models(tmp_path):
    p = tmp_path / "190000_labels.h5"
    _write(p)
    other = (
        LabelSpec(
            name="elm_hazard", task="regression", activation="none", units="1/s",
            classes=(), slug="d3d_elm_time_to_event_dsm", card_id="x/y",
            time_step_ms=50.0, ensemble_n=1, artifact_sha256="def456",
        ),
    )
    d = {"elm_hazard": Decoded(mean=np.zeros(6), lo=np.zeros(6), hi=np.zeros(6))}
    write_labels(p, 190000, T, d, other, np.ones(6, bool),
                 run_id="run-2", features_sha256="f" * 64)
    assert labelled(p) == {
        f"{SLUG}/tm_prob", f"{SLUG}/betan",
        "d3d_elm_time_to_event_dsm/elm_hazard",
    }
    assert list(tmp_path.iterdir()) == [p]


def test_index_rows_summarise_each_label(tmp_path):
    p = tmp_path / "190000_labels.h5"
    _write(p, np.array([1, 1, 0, 0, 1, 1], bool))
    rows = {r["label"]: r for r in index_rows(p)}
    assert set(rows) == {"tm_prob", "betan"}
    r = rows["tm_prob"]
    assert r["shot"] == 190000 and r["slug"] == SLUG and r["task"] == "binary"
    assert r["n_total"] == 6 and r["n_valid"] == 4
    assert 0.0 <= r["mean_valid"] <= 1.0
    assert r["run_id"] == "run-test"


def test_append_index_is_idempotent_per_shot_and_label(tmp_path):
    p = tmp_path / "190000_labels.h5"
    _write(p)
    idx = tmp_path / "labels_index.parquet"
    append_index(idx, index_rows(p))
    append_index(idx, index_rows(p))          # same shot again
    df = pd.read_parquet(idx)
    assert len(df) == 2                        # not 4
    assert set(df["label"]) == {"tm_prob", "betan"}
    assert list(tmp_path.iterdir()).count(idx) == 1


def test_read_label_raises_for_an_absent_label(tmp_path):
    p = tmp_path / "190000_labels.h5"
    _write(p)
    with pytest.raises(KeyError):
        read_label(p, SLUG, "no_such_label")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_label_store.py -q
```
Expected: `ModuleNotFoundError: No module named 'labelmaker.labels'`.

- [ ] **Step 3: Move `sha256_of` down into `config.py`**

Add to `src/labelmaker/config.py`:

```python
def sha256_of(path) -> str:
    """Hex digest of a file, read in 1 MiB blocks.

    Lives here beside `git_sha` because both answer the same question about
    an artifact: exactly which bytes produced this output.
    """
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
```

and in `src/labelmaker/models/registry.py`, replace the local definition with a
re-export so existing callers keep working:

```python
from ..config import sha256_of  # re-exported: registry.sha256_of is the public name
```

- [ ] **Step 4: Write `labels/schema.py`**

`src/labelmaker/labels/__init__.py`:

```python
"""The label file: what labelmaker produces for one shot."""
```

`src/labelmaker/labels/schema.py`:

```python
"""What one label is.

A `LabelSpec` is everything a consumer needs to interpret a stored series
without opening the model card: the task, the activation already applied,
the class names, the model's time step, and the digest of the weights that
produced it. It is written into the HDF5 group's attributes, so a label file
is self-describing even when moved.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class LabelSpec:
    name: str
    task: str
    activation: str
    units: str
    classes: tuple[str, ...]
    slug: str
    card_id: str
    time_step_ms: float
    ensemble_n: int
    artifact_sha256: str


def group_path(slug: str, label: str) -> str:
    """Where a label lives inside a label file."""
    return f"{slug}/{label}"


def artifact_digest(sha_by_name: Mapping[str, str]) -> str:
    """One digest standing for a whole set of weight files.

    Order-independent: the per-file digests are sorted by file name before
    hashing, so re-listing the ensemble cannot change the result.
    """
    h = hashlib.sha256()
    for name in sorted(sha_by_name):
        h.update(name.encode())
        h.update(sha_by_name[name].encode())
    return h.hexdigest()


def specs_for(adapter, artifact_sha256: str) -> tuple[LabelSpec, ...]:
    """One `LabelSpec` per output field of a model."""
    return tuple(
        LabelSpec(
            name=f.name,
            task=f.task,
            activation=f.activation,
            units=f.units,
            classes=tuple(f.classes),
            slug=adapter.slug,
            card_id=adapter.card_id,
            time_step_ms=float(adapter.time_step_ms),
            ensemble_n=int(adapter.ensemble_n),
            artifact_sha256=artifact_sha256,
        )
        for f in adapter.output_spec.fields
    )
```

- [ ] **Step 5: Write `labels/store.py`**

```python
"""Reading and writing `<shot>_labels.h5`.

Corpus layout, so an IGNITE loader can read a label like any other signal:
`xdata` float64 seconds, `ydata` float32 `(C, T)`. Each label gets two
companions - `<label>_spread` `(2, T)` with the ensemble min and max, and
`<label>_valid` `(1, T)` uint8, one where every input was present and inside
the training domain. Probabilities are stored, never thresholded labels:
thresholding is the consumer's decision and it is not recoverable once lost.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

from .. import __version__
from ..config import git_sha
from .schema import LabelSpec


@dataclass(frozen=True)
class LabelArray:
    x: np.ndarray
    y: np.ndarray
    attrs: dict[str, str] = field(default_factory=dict)


def _put(parent, name: str, x: np.ndarray, y: np.ndarray, dtype) -> h5py.Group:
    g = parent.create_group(name)
    g.create_dataset("xdata", data=np.asarray(x, dtype=np.float64))
    g.create_dataset("ydata", data=np.asarray(y, dtype=dtype))
    return g


def write_labels(
    path,
    shot: int,
    t: np.ndarray,
    decoded: Mapping[str, "object"],
    specs: Sequence[LabelSpec],
    valid: np.ndarray,
    *,
    run_id: str,
    features_sha256: str,
    merge: bool = True,
) -> None:
    """Write one model's labels for one shot, atomically.

    With `merge`, groups written by other models are preserved, so a second
    model's run adds to the same per-shot file instead of replacing it. The
    model's own group is always rewritten: re-running a model means its old
    numbers are stale.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    slug = specs[0].slug
    now = datetime.now(UTC).isoformat(timespec="seconds")
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        if merge and path.exists():
            with h5py.File(path, "r") as old:
                for key, value in old.attrs.items():
                    f.attrs[key] = value
                for group in old:
                    if group != slug:
                        old.copy(group, f, name=group)
        f.attrs["shot"] = int(shot)
        f.attrs["labelmaker_version"] = __version__
        f.attrs["git_sha"] = git_sha()
        f.attrs["written_at"] = now
        f.attrs["run_id"] = run_id
        f.attrs["features_sha256"] = features_sha256
        model_group = f.create_group(slug)
        valid_u8 = np.asarray(valid, dtype=bool).astype(np.uint8)[None, :]
        for spec in specs:
            dec = decoded[spec.name]
            g = _put(model_group, spec.name, t, np.asarray(dec.mean)[None, :],
                     np.float32)
            g.attrs["task"] = spec.task
            g.attrs["activation"] = spec.activation
            g.attrs["units"] = spec.units
            g.attrs["classes"] = list(spec.classes)
            g.attrs["card_id"] = spec.card_id
            g.attrs["slug"] = spec.slug
            g.attrs["time_step_ms"] = float(spec.time_step_ms)
            g.attrs["ensemble_n"] = int(spec.ensemble_n)
            g.attrs["artifact_sha256"] = spec.artifact_sha256
            _put(
                model_group, f"{spec.name}_spread", t,
                np.stack([np.asarray(dec.lo), np.asarray(dec.hi)]), np.float32,
            )
            _put(model_group, f"{spec.name}_valid", t, valid_u8, np.uint8)
    tmp.replace(path)


def read_label(path, slug: str, label: str) -> LabelArray:
    """One label series out of a label file, or KeyError."""
    with h5py.File(path, "r") as f:
        key = f"{slug}/{label}"
        if key not in f:
            raise KeyError(f"{key} not in {path}")
        g = f[key]
        return LabelArray(
            x=np.asarray(g["xdata"], dtype=np.float64),
            y=np.asarray(g["ydata"], dtype=np.float64),
            attrs={k: v for k, v in g.attrs.items()},
        )


def labelled(path) -> set[str]:
    """`"<slug>/<label>"` for every real label in the file (not companions)."""
    if not Path(path).exists():
        return set()
    out = set()
    with h5py.File(path, "r") as f:
        for slug in f:
            for name in f[slug]:
                if name.endswith("_spread") or name.endswith("_valid"):
                    continue
                out.add(f"{slug}/{name}")
    return out


def index_rows(path) -> list[dict]:
    """One summary row per label, for `labels_index.parquet`."""
    rows: list[dict] = []
    with h5py.File(path, "r") as f:
        shot = int(f.attrs["shot"])
        run_id = str(f.attrs.get("run_id", ""))
        written_at = str(f.attrs.get("written_at", ""))
        for key in sorted(labelled(path)):
            slug, label = key.split("/", 1)
            g = f[slug][label]
            y = np.asarray(g["ydata"], dtype=np.float64)[0]
            v = np.asarray(f[slug][f"{label}_valid"]["ydata"])[0].astype(bool)
            rows.append(
                {
                    "shot": shot,
                    "slug": slug,
                    "label": label,
                    "card_id": str(g.attrs.get("card_id", "")),
                    "task": str(g.attrs.get("task", "")),
                    "artifact_sha256": str(g.attrs.get("artifact_sha256", "")),
                    "n_total": int(y.size),
                    "n_valid": int(v.sum()),
                    "mean_valid": float(y[v].mean()) if v.any() else float("nan"),
                    "max_valid": float(y[v].max()) if v.any() else float("nan"),
                    "run_id": run_id,
                    "written_at": written_at,
                }
            )
    return rows


def append_index(index_path, rows) -> None:
    """Merge rows into the parquet index, replacing any (shot, slug, label).

    Rewritten whole and renamed into place: the index is small (one row per
    shot per label) and an interrupted run must never leave it truncated.
    """
    import pandas as pd

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if new.empty:
        return
    if index_path.exists():
        old = pd.read_parquet(index_path)
        keys = ["shot", "slug", "label"]
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=keys, keep="last")
    else:
        merged = new
    merged = merged.sort_values(["shot", "slug", "label"]).reset_index(drop=True)
    tmp = index_path.with_name(index_path.name + ".tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(index_path)
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_label_store.py -q
```
Expected: `8 passed`.

- [ ] **Step 7: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/labels src/labelmaker/config.py \
        src/labelmaker/models/registry.py tests/labelmaker/test_label_store.py
git commit -m "labelmaker: label schema, per-shot label store and parquet index"
```

---

### Task 10: The archive resolver

**Files:**
- Create: `src/labelmaker/features/resolve_archive.py`
- Test: `tests/labelmaker/test_resolve_archive.py`

**What this reads:** `/projects/EKOLEMEN/profile_predictor/DATA/new_h5_data/example_191450_183224.h5` — 5,000 shots (183224-191450), one group per shot, 209 datasets per shot, scalars `(240,)` and profiles `(240, 33)`, all float64, on the implicit 25 ms grid `t = 0.025 * arange(240)`. This is the store the Phase 1 model's training features were built from (see Deviation 6), so twelve of its columns are bit-identical to what the model was trained on. Dataset availability varies by shot: shot 190204's group exists but has no `bt`, so a missing column is normal and is recorded, not raised.

`gold_h5_data/all_shots.h5` (30,002 groups, 140888-191450) is a superset for older shots and is deliberately *not* read: it is 30 GB, its coverage of the corpus range is identical, and the corpus starts at 185601.

**Interfaces:**
- Consumes: `features.namespace`, `features.store.FeatureArray`.
- Produces: `resolve_archive.ARCHIVE_FILES: tuple[Path, ...]`; `shot_index(files=ARCHIVE_FILES) -> dict[int, Path]`; `resolve(shot, names, *, files=ARCHIVE_FILES) -> tuple[dict[str, FeatureArray], dict[str, str]]`.

- [ ] **Step 1: Write the failing test**

`tests/labelmaker/test_resolve_archive.py`:

```python
"""The archive resolver: 25 ms columns straight onto the canonical grid."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from labelmaker.features import namespace as ns
from labelmaker.features import resolve_archive as ra

REAL = ra.ARCHIVE_FILES[0]


def _fake_archive(tmp_path, n=240):
    p = tmp_path / "archive.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("190000")
        g.create_dataset("bt", data=np.full(n, 2.0))
        g.create_dataset("ip", data=np.linspace(0.0, 1e6, n))
        g.create_dataset("ech_pwr", data=np.full((1, n), 3.0))       # stored (1, N)
        g.create_dataset("pres_EFIT01", data=np.full((n, 33), 5.0e4))
        g.create_dataset("zipfit_edensfit_rho", data=np.tile(
            np.linspace(4.0, 1.0, 33), (n, 1)))
        f.create_group("190001").create_dataset("bt", data=np.full(n, 1.9))
    return p


def test_shot_index_maps_numeric_groups_to_their_file(tmp_path):
    p = _fake_archive(tmp_path)
    idx = ra.shot_index((p,))
    assert idx == {190000: p, 190001: p}


def test_scalars_and_profiles_land_on_the_canonical_grid(tmp_path):
    p = _fake_archive(tmp_path)
    got, missing = ra.resolve(190000, ["bt", "pres", "ne_zipfit"], files=(p,))
    assert missing == {}
    np.testing.assert_allclose(got["bt"].x, ns.GRID_S)
    assert got["bt"].y.shape == (1, 240)
    np.testing.assert_allclose(got["bt"].y[0], 2.0)
    assert got["pres"].y.shape == (33, 240)
    np.testing.assert_allclose(got["pres"].y, 5.0e4)
    assert got["ne_zipfit"].y.shape == (33, 240)
    np.testing.assert_allclose(got["ne_zipfit"].y[:, 0], np.linspace(4.0, 1.0, 33))
    assert got["bt"].attrs["resolver"] == "archive"
    assert got["bt"].attrs["locator"] == "bt"


def test_a_column_stored_with_a_leading_axis_is_squeezed(tmp_path):
    p = _fake_archive(tmp_path)
    got, _ = ra.resolve(190000, ["ech_power_total"], files=(p,))
    assert got["ech_power_total"].y.shape == (1, 240)
    np.testing.assert_allclose(got["ech_power_total"].y[0], 3.0)


def test_a_missing_column_is_recorded_per_feature(tmp_path):
    p = _fake_archive(tmp_path)
    got, missing = ra.resolve(190000, ["bt", "kappa"], files=(p,))
    assert set(got) == {"bt"}
    assert missing == {"kappa": "KeyError"}


def test_a_shot_outside_the_archive_misses_everything(tmp_path):
    p = _fake_archive(tmp_path)
    got, missing = ra.resolve(999999, ["bt", "pres"], files=(p,))
    assert got == {}
    assert missing == {"bt": "ShotNotInArchive", "pres": "ShotNotInArchive"}


def test_an_unknown_feature_name_is_refused(tmp_path):
    # Every Phase 1 feature has an archive source, so the refusal this
    # resolver can actually raise is for a name outside the namespace.
    p = _fake_archive(tmp_path)
    with pytest.raises(KeyError, match="no_such_feature"):
        ra.resolve(190000, ["no_such_feature"], files=(p,))


def test_a_shorter_record_keeps_its_own_grid(tmp_path):
    p = _fake_archive(tmp_path, n=120)
    got, _ = ra.resolve(190000, ["bt"], files=(p,))
    assert got["bt"].y.shape == (1, 120)
    np.testing.assert_allclose(got["bt"].x[-1], 0.025 * 119)
    assert got["bt"].attrs["n_rows"] == "120"


@pytest.mark.skipif(not REAL.exists(), reason=f"archive not available: {REAL}")
def test_real_archive_resolves_every_archive_feature_for_an_overlap_shot():
    names = [f.name for f in ns.by_source("archive")]
    got, missing = ra.resolve(185945, names)
    assert missing == {}, missing
    for name in names:
        arr = got[name]
        assert arr.x.size == 240
        want_c = 33 if ns.by_name(name).kind == "profile" else 1
        assert arr.y.shape == (want_c, 240), name
    # pres is bit-identical to the model's training column; check the raw read
    with h5py.File(REAL, "r") as f:
        raw = np.asarray(f["185945"]["pres_EFIT01"])
    np.testing.assert_array_equal(got["pres"].y, raw.T)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_resolve_archive.py -q
```
Expected: `ImportError: cannot import name 'resolve_archive'`.

- [ ] **Step 3: Write the implementation**

`src/labelmaker/features/resolve_archive.py`:

```python
"""Features from the 25 ms archive store.

`/projects/EKOLEMEN/profile_predictor/DATA/new_h5_data/example_191450_183224.h5`
holds 5,000 shots (183224..191450) as one group per shot, 209 columns each,
scalars `(240,)` and profiles `(240, 33)` on an implicit 25 ms grid. It is
the store the Phase 1 model's training features were built from: twelve of
its columns are bit-identical to that model's training inputs (see the
plan's Deviation 6), which is why this resolver comes first in every
feature's `sources`.

It covers only 21% of the corpus (3,621 of 16,909 shots), so it is the
proof-of-concept path, not the scaling path. `resolve_fdp` is the latter.

Availability varies per shot: some groups are missing individual columns, so
a missing column is recorded as a per-feature miss rather than raised.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

from . import namespace as ns
from .store import FeatureArray

SOURCE = "archive"

ARCHIVE_FILES: tuple[Path, ...] = (
    Path(
        "/projects/EKOLEMEN/profile_predictor/DATA/new_h5_data/"
        "example_191450_183224.h5"
    ),
)


@lru_cache(maxsize=4)
def shot_index(files: tuple[Path, ...] = ARCHIVE_FILES) -> dict[int, Path]:
    """shot -> the archive file holding it.

    Metadata only: listing the group names of a 5.8 GB file reads no arrays.
    Cached because a bulk run asks for it once per worker.
    """
    index: dict[int, Path] = {}
    for path in files:
        if not Path(path).exists():
            continue
        with h5py.File(path, "r") as f:
            for key in f:
                if key.isdigit():
                    index.setdefault(int(key), Path(path))
    return index


def resolve(
    shot: int,
    names: Sequence[str],
    *,
    files: tuple[Path, ...] = ARCHIVE_FILES,
) -> tuple[dict[str, FeatureArray], dict[str, str]]:
    """Read the requested canonical features for one shot.

    Returns `(arrays, missing)`; `missing` maps a feature name to a short
    cause so a run records why a shot is incomplete instead of failing.
    """
    specs = [ns.by_name(n) for n in names]
    for spec in specs:
        spec.locator_for(SOURCE)  # KeyError names the feature and the source
    path = shot_index(tuple(files)).get(int(shot))
    if path is None:
        return {}, {n: "ShotNotInArchive" for n in names}
    arrays: dict[str, FeatureArray] = {}
    missing: dict[str, str] = {}
    with h5py.File(path, "r") as f:
        group = f[str(int(shot))]
        for spec in specs:
            locator = spec.locator_for(SOURCE)
            if locator not in group:
                missing[spec.name] = "KeyError"
                continue
            raw = np.asarray(group[locator], dtype=np.float64)
            raw = np.squeeze(raw)
            if spec.kind == "profile":
                if raw.ndim != 2:
                    missing[spec.name] = "ShapeError"
                    continue
                y = raw.T                      # (n_rho, T)
                n = y.shape[1]
            else:
                if raw.ndim != 1:
                    missing[spec.name] = "ShapeError"
                    continue
                y = raw[None, :]
                n = y.shape[1]
            arrays[spec.name] = FeatureArray(
                x=ns.STEP_S * np.arange(n, dtype=np.float64),
                y=y,
                attrs={
                    "resolver": SOURCE,
                    "locator": locator,
                    "archive_file": str(path),
                    "n_rows": str(n),
                },
            )
    return arrays, missing
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_resolve_archive.py -q
```
Expected: `8 passed`.

- [ ] **Step 5: Record how much of the corpus this resolver covers**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
from labelmaker.catalog import corpus_shots, overlap_shots
from labelmaker.config import Paths
from labelmaker.features.resolve_archive import shot_index

paths = Paths.from_env()
corpus = set(corpus_shots(paths))
arch = set(shot_index())
overlap = set(overlap_shots(paths))
print(f"corpus            {len(corpus):6d}  {min(corpus)}..{max(corpus)}")
print(f"archive           {len(arch):6d}  {min(arch)}..{max(arch)}")
print(f"corpus & archive  {len(corpus & arch):6d}  ({100*len(corpus & arch)/len(corpus):.1f}% of corpus)")
print(f"poc pool in arch  {len(overlap & arch):6d}  of {len(overlap)} overlap shots")
PY
```
Expected (measured 2026-09-03): corpus 16,909 / archive 5,000 / corpus & archive 3,621 (21.4%) / poc pool 1,497 of 1,503. Record the numbers printed in the commit message; if the archive coverage of the PoC pool has dropped below ~1,400, stop and re-check the shot list before continuing.

- [ ] **Step 6: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/features/resolve_archive.py \
        tests/labelmaker/test_resolve_archive.py
git commit -m "labelmaker: archive resolver for the 25 ms feature store"
```

---

### Task 11: The corpus resolver

**Files:**
- Create: `src/labelmaker/features/resolve_corpus.py`
- Test: `tests/labelmaker/test_resolve_corpus.py`

**What this reads:** `<LABELMAKER_CORPUS>/<shot>_processed.h5`. Verified layout (shot 185945): exactly 32 flat groups, each with only `xdata` (1-D float32 **seconds**) and `ydata` `(C, T)` float32, and **no attributes anywhere in the file**. Relevant groups: `pinj` `(8, 131002)` from t=0, `tinj` `(8, 131002)`, `ech_power` `(12, 102501)` from t=-0.25. A group with `ydata.shape[-1] < 2` is the corpus' "absent" sentinel. `ech_power` can contain all-NaN channels (channel 3 on that shot), so totals need `nansum`.

Measured units: corpus `pinj` is in **W** (1.002e7 summed at t=1.025 s) where the model wants **kW** (10,995.6 in the archive), and corpus `tinj` is already in N m (8.43 vs 9.24). The residual ~10% is a *sampling* difference — the corpus is a 10 kHz instantaneous sample, the archive value is a window statistic — which Task 15 measures and settles.

**Interfaces:**
- Consumes: `features.namespace`, `features.store.FeatureArray`, `timebase.decimate_to_step`.
- Produces: `resolve_corpus.SOURCE = "corpus"`; `resolve_corpus.SCALE_TO_CANONICAL: dict[str, float]`; `resolve(shot, names, *, corpus: Path) -> tuple[dict[str, FeatureArray], dict[str, str]]`.

- [ ] **Step 1: Write the failing test**

`tests/labelmaker/test_resolve_corpus.py`:

```python
"""The corpus resolver: channel sums, units, decimation, absent groups."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from labelmaker.features import resolve_corpus as rc

CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")


def _fake_corpus(tmp_path, *, ech_absent=False, ech_nan_channel=True):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    n = 2000                                  # 2 s at 1 kHz, for speed
    t = np.linspace(0.0, 2.0, n, dtype=np.float32)
    with h5py.File(corpus / "190000_processed.h5", "w") as f:
        g = f.create_group("pinj")            # 8 beams, 1 MW each, in W
        g.create_dataset("xdata", data=t)
        g.create_dataset("ydata", data=np.full((8, n), 1.0e6, dtype=np.float32))
        g = f.create_group("tinj")            # 8 beams, 1 N m each
        g.create_dataset("xdata", data=t)
        g.create_dataset("ydata", data=np.full((8, n), 1.0, dtype=np.float32))
        g = f.create_group("ech_power")
        g.create_dataset("xdata", data=np.array([0.0], dtype=np.float32)
                         if ech_absent else t)
        y = np.full((12, 1 if ech_absent else n), 1.0e5, dtype=np.float32)
        if ech_nan_channel and not ech_absent:
            y[3] = np.nan
        g.create_dataset("ydata", data=y)
    return corpus


def test_beam_channels_are_summed_and_scaled_to_the_model_units(tmp_path):
    corpus = _fake_corpus(tmp_path)
    got, missing = rc.resolve(190000, ["pinj_total", "tinj_total"], corpus=corpus)
    assert missing == {}
    # 8 beams x 1 MW = 8e6 W = 8000 kW
    np.testing.assert_allclose(np.nanmax(got["pinj_total"].y), 8000.0, rtol=1e-5)
    np.testing.assert_allclose(np.nanmax(got["tinj_total"].y), 8.0, rtol=1e-5)
    assert got["pinj_total"].attrs["resolver"] == "corpus"
    assert got["pinj_total"].attrs["scale_to_canonical"] == "0.001"


def test_nan_channels_do_not_poison_the_total(tmp_path):
    corpus = _fake_corpus(tmp_path)
    got, _ = rc.resolve(190000, ["ech_power_total"], corpus=corpus)
    # 11 finite channels x 1e5 W; the NaN channel is skipped, not propagated
    np.testing.assert_allclose(np.nanmax(got["ech_power_total"].y), 11.0e5, rtol=1e-5)
    assert got["ech_power_total"].attrs["nan_channels"] == "1"


def test_series_are_decimated_to_the_declared_step(tmp_path):
    corpus = _fake_corpus(tmp_path)
    got, _ = rc.resolve(190000, ["pinj_total"], corpus=corpus)
    x = got["pinj_total"].x
    assert x.size == 2001                      # 2 s at the 1 ms declared step
    np.testing.assert_allclose(np.diff(x), 0.001)


def test_an_absent_group_is_recorded_not_raised(tmp_path):
    corpus = _fake_corpus(tmp_path, ech_absent=True)
    got, missing = rc.resolve(190000, ["pinj_total", "ech_power_total"], corpus=corpus)
    assert set(got) == {"pinj_total"}
    assert missing == {"ech_power_total": "SignalAbsent"}


def test_a_missing_shot_file_misses_everything(tmp_path):
    corpus = _fake_corpus(tmp_path)
    got, missing = rc.resolve(999999, ["pinj_total"], corpus=corpus)
    assert got == {}
    assert missing == {"pinj_total": "FileNotFoundError"}


def test_a_feature_with_no_corpus_source_is_refused(tmp_path):
    corpus = _fake_corpus(tmp_path)
    with pytest.raises(KeyError, match="corpus"):
        rc.resolve(190000, ["ne_zipfit"], corpus=corpus)


@pytest.mark.skipif(not CORPUS.exists(), reason=f"corpus not available: {CORPUS}")
def test_real_corpus_shot_resolves_all_three_actuator_totals():
    got, missing = rc.resolve(
        185945, ["pinj_total", "tinj_total", "ech_power_total"], corpus=CORPUS
    )
    assert missing == {}
    for name in ("pinj_total", "tinj_total", "ech_power_total"):
        assert got[name].y.shape[0] == 1
        assert got[name].x[0] <= 0.0 or name == "pinj_total"
    # 10 MW-class beam power on this shot, expressed in kW
    assert 5_000.0 < np.nanmax(got["pinj_total"].y) < 30_000.0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_resolve_corpus.py -q
```
Expected: `ImportError: cannot import name 'resolve_corpus'`.

- [ ] **Step 3: Write the implementation**

`src/labelmaker/features/resolve_corpus.py`:

```python
"""Features from the FAITH corpus itself.

`<shot>_processed.h5` is 32 flat groups, each holding only `xdata` (1-D
float32 seconds) and `ydata` `(C, T)` float32, with no attributes anywhere
in the file - so units are not discoverable from the data and are recorded
here, measured. A group whose `ydata` last axis is shorter than 2 is the
corpus' absent-signal sentinel (see
tokamak_foundation_model/data/multi_file_dataset.py:845-861).

This resolver covers every corpus shot, which the archive resolver does not,
but it can only serve the actuator totals: the corpus holds raw diagnostics
and actuators, no equilibrium and no fitted profiles.

Measured on shot 185945 at t = 1.025 s: summed `pinj` = 1.002e7 where the
model's own training column reads 10,995.6, so the corpus is in W and the
canonical feature is kW. `tinj` needs no scaling (8.43 vs 9.24 N m). The
residual difference is a sampling difference, not a unit one, and Task 15
decides between nearest-sample and window-mean by measurement.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from . import namespace as ns
from ..timebase import decimate_to_step
from .store import FeatureArray

SOURCE = "corpus"

#: Multiplier taking a corpus channel sum to the canonical feature's units.
SCALE_TO_CANONICAL: dict[str, float] = {
    "pinj_total": 1e-3,        # W -> kW
    "tinj_total": 1.0,         # already N m
    "ech_power_total": 1.0,    # already W
}


def resolve(
    shot: int,
    names: Sequence[str],
    *,
    corpus: Path,
) -> tuple[dict[str, FeatureArray], dict[str, str]]:
    """Sum the channels of each requested corpus group, in canonical units."""
    specs = [ns.by_name(n) for n in names]
    for spec in specs:
        spec.locator_for(SOURCE)
    path = Path(corpus) / f"{int(shot)}_processed.h5"
    if not path.exists():
        return {}, {n: "FileNotFoundError" for n in names}
    arrays: dict[str, FeatureArray] = {}
    missing: dict[str, str] = {}
    with h5py.File(path, "r") as f:
        for spec in specs:
            group = spec.locator_for(SOURCE)
            if group not in f or "ydata" not in f[group]:
                missing[spec.name] = "KeyError"
                continue
            y = np.asarray(f[group]["ydata"], dtype=np.float64)
            if y.shape[-1] < 2:
                missing[spec.name] = "SignalAbsent"
                continue
            x = np.asarray(f[group]["xdata"], dtype=np.float64)
            if x.size != y.shape[-1]:
                missing[spec.name] = "ShapeError"
                continue
            nan_channels = int((~np.isfinite(y)).all(axis=1).sum())
            with np.errstate(invalid="ignore"):
                total = np.nansum(y, axis=0) * SCALE_TO_CANONICAL[spec.name]
            # A time where every channel is NaN is genuinely unknown; nansum
            # would report 0, which for ECH power is a different claim.
            total[(~np.isfinite(y)).all(axis=0)] = np.nan
            step = spec.step or 0.001
            xg, yg = decimate_to_step(x, total[None, :], step)
            arrays[spec.name] = FeatureArray(
                x=xg,
                y=yg,
                attrs={
                    "resolver": SOURCE,
                    "locator": group,
                    "n_channels": str(y.shape[0]),
                    "nan_channels": str(nan_channels),
                    "scale_to_canonical": str(SCALE_TO_CANONICAL[spec.name]),
                    "decimated_to_s": str(step),
                },
            )
    return arrays, missing
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_resolve_corpus.py -q
```
Expected: `7 passed`.

- [ ] **Step 5: Settle the ECH power units against the archive**

The archive column `ech_pwr` and the corpus group `ech_power` are both zero on
shot 185945, so that shot cannot decide the scale. Find a shot with ECH and
compare:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
import h5py, numpy as np
from labelmaker.catalog import overlap_shots
from labelmaker.config import Paths
from labelmaker.features import resolve_archive as ra, resolve_corpus as rc

paths = Paths.from_env()
pool = [s for s in overlap_shots(paths) if s in ra.shot_index()]
checked = 0
for shot in pool:
    arch, _ = ra.resolve(shot, ["ech_power_total"])
    if "ech_power_total" not in arch:
        continue
    a = arch["ech_power_total"].y[0]
    if not np.isfinite(a).any() or np.nanmax(a) <= 0:
        continue
    cor, miss = rc.resolve(shot, ["ech_power_total"], corpus=paths.corpus)
    if "ech_power_total" not in cor:
        continue
    c = cor["ech_power_total"].y[0]
    print(f"shot {shot}: archive max={np.nanmax(a):.4g}  corpus max={np.nanmax(c):.4g}  "
          f"ratio={np.nanmax(c)/np.nanmax(a):.4g}")
    checked += 1
    if checked == 5:
        break
if checked == 0:
    print("no overlap shot has non-zero ECH in both sources")
PY
```

Read the ratio: **~1.0** means both are in W and `SCALE_TO_CANONICAL["ech_power_total"]`
stays `1.0`; **~1000** means the corpus is in W and the archive in kW, so set the
scale to `1e-3` and change the `ech_power_total` units in `namespace.py` to `kW`;
**~1e-3** is the reverse. Whatever the answer, update the `notes` on that
`FeatureSpec` to state the measured ratio and the shots it was measured on, and
add a test in `test_resolve_corpus.py` asserting the scale constant. If no
overlap shot has ECH in both sources, leave the scale at `1.0`, record that in
the notes, and add the check to Task 15's report instead.

- [ ] **Step 6: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/features/resolve_corpus.py \
        src/labelmaker/features/namespace.py \
        tests/labelmaker/test_resolve_corpus.py
git commit -m "labelmaker: corpus resolver for actuator totals, units measured"
```

---

### Task 12: The fdp resolver (the scaling path)

**Files:**
- Create: `src/labelmaker/features/resolve_fdp.py`
- Test: `tests/labelmaker/test_resolve_fdp.py`
- Modify: `src/labelmaker/features/namespace.py` (correct any locator the probe disproves; add ZIPFIT locators)

**Why this task exists:** the archive resolver covers 21% of the corpus. Everything else — 13,288 shots, and every shot recorded after 191450 — can only be reached through fdp. Phase 1 builds this path and *measures* it against the archive on overlap shots, where the archive is a bit-identical reference for seven of the features. It is not run over all 100 PoC shots; that is a scheduling decision for later, not a design one.

**Environment:** the `labelmaker` pixi env includes the `fdp` feature (toksearch 2.2.3, toksearch_d3d 0.1.5). Fetching needs a cached SciToken: `pixi run fdp login` once, then `FDP_NO_AUTO_LOGIN=1` in batch. The reference implementation to copy from is `/scratch/gpfs/nc1514/fdp/scripts/omnimode.py` — Nathan's working fetch script, which is where every pattern below comes from:

- `MdsSignal(expr, tree).fetch(shot)` returns a dict with `"data"`, `"units"`, and, when built with `dims=[...]`, one entry per named dim. Trees confirmed reachable: `efit01`, `zipfit01`, `electrons` (omnimode.py:332, 365, 387).
- `PtDataSignal(name).fetch(shot)` for PTDATA points (`toksearch_d3d`).
- **Dim-order trap** (omnimode.py:147-149, 166-172): profile `data` comes back `(n_t, n_x)` while dimension 0 is the *profile coordinate* and dimension 1 is *time in ms*. So with `dims=["x", "t_ms"]`, `rec["x"]` has length `data.shape[1]` and `rec["t_ms"]` length `data.shape[0]`. Settle this from shapes at read time; never assert it.
- `dim_of(...)` TDI strings do not work through either signal class — use the `dims` mechanism.
- **Fork before the first fetch**: `toksearch_d3d`'s ptserver reader is not fork-safe (omnimode.py's `Pool(12)` is created before any fetch).
- fdp has no retries and no caching. Retry once per signal, then record the miss.

**Interfaces:**
- Consumes: `features.namespace`, `features.store.FeatureArray`, `timebase.decimate_to_step`.
- Produces: `resolve_fdp.SOURCE = "fdp"`; `available() -> bool`; `resolve(shot, names, *, retries=1) -> tuple[dict[str, FeatureArray], dict[str, str]]`; helpers `_fetch_ptdata(name, shot)`, `_fetch_mds(expr, tree, shot, dims=())`, `_to_rho_grid(values, coord, grid)`.

- [ ] **Step 1: Probe the node names and record what comes back**

Nothing in this task is guesswork that survives this step. Run it once, on a
shot the archive also covers, so every answer has a reference:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker fdp login          # once per token lifetime
FDP_NO_AUTO_LOGIN=1 pixi run -e labelmaker python - <<'PY'
import numpy as np
from toksearch import MdsSignal
from toksearch_d3d import PtDataSignal

SHOT = 185945

def show(label, thunk):
    try:
        rec = thunk()
    except Exception as exc:
        print(f"  {label:52s} FAIL {type(exc).__name__}: {str(exc)[:70]}")
        return
    keys = sorted(rec)
    data = np.asarray(rec["data"])
    extra = {k: np.shape(np.asarray(rec[k])) for k in keys if k not in ("data", "units")}
    print(f"  {label:52s} keys={keys} data={data.shape} {data.dtype} "
          f"dims={extra} units={rec.get('units')}")
    flat = data.ravel()
    print(f"  {'':52s} first={flat[:3]} last={flat[-3:]}")

print("PTDATA:")
for name in ("ip", "bt"):
    show(name, lambda name=name: PtDataSignal(name).fetch(SHOT))

print("efit01 aeqdsk scalars:")
for leaf in ("kappa", "tritop", "tribot", "gapin", "betan", "atime", "time"):
    expr = rf"\efit01::top.results.aeqdsk:{leaf}"
    show(expr, lambda expr=expr: MdsSignal(expr, "efit01").fetch(SHOT))

print("efit01 geqdsk:")
for leaf in ("gtime", "rmaxis", "qpsi", "pres", "rhovn"):
    expr = rf"\efit01::top.results.geqdsk:{leaf}"
    show(expr, lambda expr=expr: MdsSignal(expr, "efit01").fetch(SHOT))

print("zipfit01 profiles (with dims):")
for leaf in ("EDENSFIT", "ETEMPFIT", "TROTFIT"):
    expr = rf"\ZIPFIT01::TOP.PROFILES.{leaf}"
    show(expr, lambda expr=expr: MdsSignal(expr, "zipfit01",
                                           dims=["x", "t_ms"]).fetch(SHOT))
PY
```

Write the printed block verbatim into the module docstring of
`resolve_fdp.py` as a dated measurement record, and act on it:

| probe result | action |
|---|---|
| an `aeqdsk` leaf fails | try `\efit01::{leaf}` (the shortcut form the legacy HDF5 archives use), then `\efit01::top.results.a{leaf}`; record which spelling worked in `namespace.py`'s `locators` |
| the aeqdsk time node is `atime` rather than `time` | set `AEQDSK_TIME` accordingly in the implementation below |
| a `dims` entry length matches `data.shape[0]` instead of `shape[1]` | the trap is inverted for that node; `_to_rho_grid` reads it from shapes, so only the docstring record needs updating |
| `zipfit01` is unreachable | leave `ne_zipfit`/`te_zipfit`/`rot_zipfit` with `sources=("archive",)` (they already are) and note in the module docstring that the kinetic profiles have no fdp path; the fdp resolver then serves scalars and EFIT profiles only, and full-corpus scaling needs Phase 2's own fits |
| units come back in ms | already handled: every time axis is divided by 1000 on read |

- [ ] **Step 2: Write the failing test**

`tests/labelmaker/test_resolve_fdp.py`:

```python
"""The fdp resolver. Network-free by default; live fetches are opt-in."""
import os
from pathlib import Path

import numpy as np
import pytest

from labelmaker.features import namespace as ns
from labelmaker.features import resolve_fdp as rf

LIVE = os.environ.get("LABELMAKER_FDP") == "1"


def test_to_rho_grid_resamples_a_profile_onto_the_canonical_grid():
    coord = np.linspace(0.0, 1.2, 121)           # ZIPFIT's own x axis
    values = np.tile(coord[None, :], (4, 1))     # value == coordinate
    out = rf._to_rho_grid(values, coord, ns.RHO_GRID)
    assert out.shape == (33, 4)                  # (n_rho, T)
    np.testing.assert_allclose(out[:, 0], ns.RHO_GRID, atol=1e-12)


def test_to_rho_grid_handles_a_time_dependent_coordinate():
    # EFIT profiles live on a psi grid whose rho mapping changes with time.
    coord = np.stack([np.linspace(0.0, 1.0, 65), np.linspace(0.0, 0.5, 65)])
    values = np.stack([np.linspace(0.0, 1.0, 65), np.linspace(0.0, 1.0, 65)])
    out = rf._to_rho_grid(values, coord, ns.RHO_GRID)
    assert out.shape == (33, 2)
    np.testing.assert_allclose(out[:, 0], ns.RHO_GRID, atol=1e-12)
    # second slice: rho only reaches 0.5, so beyond it the profile is NaN
    assert np.isnan(out[ns.RHO_GRID > 0.5, 1]).all()


def test_to_rho_grid_rejects_mismatched_axes():
    with pytest.raises(ValueError):
        rf._to_rho_grid(np.zeros((4, 65)), np.linspace(0, 1, 33), ns.RHO_GRID)


def test_resolve_records_a_miss_when_toksearch_is_unavailable(monkeypatch):
    monkeypatch.setattr(rf, "available", lambda: False)
    got, missing = rf.resolve(190000, ["ip", "kappa"])
    assert got == {}
    assert missing == {"ip": "ToksearchUnavailable", "kappa": "ToksearchUnavailable"}


def test_resolve_records_per_signal_failures(monkeypatch):
    monkeypatch.setattr(rf, "available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("ptserver said no")

    monkeypatch.setattr(rf, "_fetch_ptdata", boom)
    monkeypatch.setattr(
        rf, "_fetch_mds",
        lambda expr, tree, shot, dims=(): {
            "data": np.ones(10), "t_s": np.linspace(0, 1, 10)
        },
    )
    got, missing = rf.resolve(190000, ["ip", "kappa"], retries=0)
    assert missing == {"ip": "RuntimeError"}
    assert "kappa" in got


def test_a_feature_with_no_fdp_source_is_refused():
    with pytest.raises(KeyError, match="fdp"):
        rf.resolve(190000, ["ech_rho"])


@pytest.mark.skipif(not LIVE, reason="live fdp fetch is opt-in: LABELMAKER_FDP=1")
def test_live_fetch_of_every_fdp_feature_for_one_shot():
    names = [f.name for f in ns.by_source("fdp")]
    got, missing = rf.resolve(185945, names)
    assert set(got), f"nothing fetched; misses were {missing}"
    for name, arr in got.items():
        spec = ns.by_name(name)
        want_c = 33 if spec.kind == "profile" else 1
        assert arr.y.shape[0] == want_c, name
        assert arr.x.ndim == 1 and arr.x.size == arr.y.shape[1]
        assert arr.x[-1] < 30.0, f"{name}: time axis looks like ms, not s"
        assert arr.attrs["resolver"] == "fdp"
    print("missing:", missing)
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_resolve_fdp.py -q
```
Expected: `ImportError: cannot import name 'resolve_fdp'`.

- [ ] **Step 4: Write the implementation**

`src/labelmaker/features/resolve_fdp.py`:

```python
"""Features from fdp: PTDATA points and MDSplus trees.

This is the path that scales. The archive store covers 3,621 of the corpus'
16,909 shots; everything else has to be fetched. Patterns here are copied
from /scratch/gpfs/nc1514/fdp/scripts/omnimode.py, which is a working fetch
script against this cluster's Pelican/OSDF route:

  * `MdsSignal(expr, tree).fetch(shot)` -> {"data", "units", *dims}
  * `PtDataSignal(name).fetch(shot)` for PTDATA
  * profile `data` is (n_t, n_x) while dim 0 is the profile coordinate and
    dim 1 is time in ms - resolved from shapes below, never assumed
  * toksearch has no retries and no caching; one retry here, then a miss
  * toksearch_d3d's ptserver reader is not fork-safe, so a worker pool must
    be forked before the first fetch (see run.py)

Imports of toksearch are deferred into `available()` and the fetch helpers,
so importing labelmaker in the default pixi environment never touches it.

MEASUREMENT RECORD (paste the output of Task 12 Step 1 here, with its date).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import namespace as ns
from ..timebase import decimate_to_step
from .store import FeatureArray

SOURCE = "fdp"

#: The aeqdsk time node, as confirmed by the Task 12 Step 1 probe.
AEQDSK_TIME = r"\efit01::top.results.aeqdsk:atime"
#: The geqdsk time node (omnimode.py:138 uses gtime's own data as the axis).
GEQDSK_TIME = r"\efit01::top.results.geqdsk:gtime"
#: Radial coordinate of the EFIT profile grid, per time slice.
GEQDSK_RHO = r"\efit01::top.results.geqdsk:rhovn"

MS_PER_S = 1000.0


def available() -> bool:
    """True when toksearch can be imported (the `labelmaker`/`fdp` envs)."""
    try:
        import toksearch  # noqa: F401
        import toksearch_d3d  # noqa: F401
    except Exception:
        return False
    return True


def _fetch_ptdata(name: str, shot: int) -> dict:
    """One PTDATA point, with its time axis converted to seconds."""
    from toksearch_d3d import PtDataSignal

    rec = PtDataSignal(name).fetch(int(shot))
    data = np.asarray(rec["data"], dtype=np.float64)
    times = np.asarray(rec.get("times", rec.get("t_ms")), dtype=np.float64)
    return {"data": data, "t_s": times / MS_PER_S}


def _fetch_mds(expr: str, tree: str, shot: int, dims: Sequence[str] = ()) -> dict:
    """One MDSplus node. `dims` uses toksearch's getDimensionAt mechanism."""
    from toksearch import MdsSignal

    signal = MdsSignal(expr, tree, dims=list(dims)) if dims else MdsSignal(expr, tree)
    rec = signal.fetch(int(shot))
    out = {"data": np.asarray(rec["data"], dtype=np.float64)}
    for dim in dims:
        out[dim] = np.asarray(rec[dim], dtype=np.float64)
    if "t_ms" in out:
        out["t_s"] = out.pop("t_ms") / MS_PER_S
    return out


def _to_rho_grid(values: np.ndarray, coord: np.ndarray, grid: np.ndarray):
    """Resample `(n_t, n_x)` profiles onto `grid`, returning `(n_grid, n_t)`.

    `coord` is either one shared axis `(n_x,)` (ZIPFIT) or one axis per time
    slice `(n_t, n_x)` (EFIT, whose psi grid maps to a different rho each
    slice). Outside a slice's coordinate range the result is NaN rather than
    an edge value: an extrapolated profile edge is not a measurement.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    coord = np.asarray(coord, dtype=np.float64)
    n_t, n_x = values.shape
    if coord.ndim == 1:
        if coord.size != n_x:
            raise ValueError(f"coord {coord.shape} does not match values {values.shape}")
        coord = np.broadcast_to(coord, (n_t, n_x))
    elif coord.shape != values.shape:
        raise ValueError(f"coord {coord.shape} does not match values {values.shape}")
    out = np.full((grid.size, n_t), np.nan)
    for i in range(n_t):
        x, y = coord[i], values[i]
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 2:
            continue
        x, y = x[good], y[good]
        order = np.argsort(x)
        x, y = x[order], y[order]
        inside = (grid >= x[0]) & (grid <= x[-1])
        out[inside, i] = np.interp(grid[inside], x, y)
    return out


def _series(name: str, rec: dict, spec) -> FeatureArray:
    """A scalar record -> a decimated `(1, T)` feature."""
    x, y = decimate_to_step(rec["t_s"], rec["data"][None, :], spec.step or 0.001)
    return FeatureArray(
        x=x, y=y,
        attrs={"resolver": SOURCE, "locator": spec.locator_for(SOURCE),
               "decimated_to_s": str(spec.step or 0.001)},
    )


def resolve(
    shot: int,
    names: Sequence[str],
    *,
    retries: int = 1,
) -> tuple[dict[str, FeatureArray], dict[str, str]]:
    """Fetch the requested canonical features for one shot through fdp."""
    specs = [ns.by_name(n) for n in names]
    for spec in specs:
        spec.locator_for(SOURCE)
    if not available():
        return {}, {n: "ToksearchUnavailable" for n in names}

    arrays: dict[str, FeatureArray] = {}
    missing: dict[str, str] = {}
    axes: dict[str, dict] = {}      # cached time / coordinate nodes

    def cached(key: str, thunk):
        if key not in axes:
            axes[key] = thunk()
        return axes[key]

    for spec in specs:
        locator = spec.locator_for(SOURCE)
        for attempt in range(retries + 1):
            try:
                if not locator.startswith("\\"):
                    rec = _fetch_ptdata(locator, shot)
                    arrays[spec.name] = _series(spec.name, rec, spec)
                elif spec.kind == "scalar":
                    rec = _fetch_mds(locator, "efit01", shot)
                    time_node = (
                        AEQDSK_TIME if "aeqdsk" in locator else GEQDSK_TIME
                    )
                    axis = cached(
                        time_node, lambda n=time_node: _fetch_mds(n, "efit01", shot)
                    )
                    t_s = np.asarray(axis["data"], dtype=np.float64) / MS_PER_S
                    if t_s.size != rec["data"].size:
                        raise ValueError(
                            f"{locator}: {rec['data'].size} samples vs "
                            f"{t_s.size} times from {time_node}"
                        )
                    arrays[spec.name] = _series(
                        spec.name, {"data": rec["data"], "t_s": t_s}, spec
                    )
                elif "ZIPFIT" in locator:
                    rec = _fetch_mds(locator, "zipfit01", shot, dims=["x", "t_ms"])
                    y = _to_rho_grid(rec["data"], rec["x"], ns.RHO_GRID)
                    arrays[spec.name] = FeatureArray(
                        x=rec["t_s"], y=y,
                        attrs={"resolver": SOURCE, "locator": locator,
                               "resampled_from": "zipfit x axis"},
                    )
                else:
                    rec = _fetch_mds(locator, "efit01", shot)
                    axis = cached(
                        GEQDSK_TIME,
                        lambda: _fetch_mds(GEQDSK_TIME, "efit01", shot),
                    )
                    rho = cached(
                        GEQDSK_RHO, lambda: _fetch_mds(GEQDSK_RHO, "efit01", shot)
                    )
                    t_s = np.asarray(axis["data"], dtype=np.float64) / MS_PER_S
                    y = _to_rho_grid(rec["data"], rho["data"], ns.RHO_GRID)
                    if y.shape[1] != t_s.size:
                        raise ValueError(
                            f"{locator}: {y.shape[1]} slices vs {t_s.size} times"
                        )
                    arrays[spec.name] = FeatureArray(
                        x=t_s, y=y,
                        attrs={"resolver": SOURCE, "locator": locator,
                               "resampled_from": "geqdsk rhovn"},
                    )
                missing.pop(spec.name, None)
                break
            except Exception as exc:  # per-signal isolation, as in omnimode
                missing[spec.name] = type(exc).__name__
    return arrays, missing
```

- [ ] **Step 5: Run the tests to verify they pass, then measure against the archive**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_resolve_fdp.py -q
LABELMAKER_FDP=1 FDP_NO_AUTO_LOGIN=1 pixi run -e labelmaker \
    python -m pytest tests/labelmaker/test_resolve_fdp.py -q -k live -s
```
Expected: `6 passed, 1 skipped`, then the live test passing with its `missing:` line printed. Any feature in that `missing` dict must be explained by Step 1's probe — if a node failed there, correct the locator; if it fetched there and misses here, the bug is in this module.

Then compare the fdp path against the archive, which is a bit-identical
reference for seven features:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
LABELMAKER_FDP=1 FDP_NO_AUTO_LOGIN=1 pixi run -e labelmaker python - <<'PY'
import numpy as np
from labelmaker.features import namespace as ns
from labelmaker.features import resolve_archive as ra, resolve_fdp as rf
from labelmaker.timebase import sample_at

SHOTS = [185945, 186697, 187327]
names = [f.name for f in ns.by_source("fdp") if "archive" in f.sources]
print(f"{'shot':7s} {'feature':12s} {'median rel':>12s} {'corr':>7s}  n")
for shot in SHOTS:
    arch, _ = ra.resolve(shot, names)
    fdp, miss = rf.resolve(shot, names)
    for name in names:
        if name not in arch or name not in fdp:
            print(f"{shot:<7d} {name:12s} {'-':>12s} {'-':>7s}  "
                  f"missing ({miss.get(name, 'no archive')})")
            continue
        a = arch[name]
        b = sample_at(fdp[name].x, fdp[name].y, a.x, max_gap=0.025)
        ya, yb = a.y.ravel(), np.asarray(b).ravel()
        good = np.isfinite(ya) & np.isfinite(yb)
        if good.sum() < 10:
            print(f"{shot:<7d} {name:12s} {'-':>12s} {'-':>7s}  too few points")
            continue
        rel = np.median(np.abs(ya[good] - yb[good]) / (np.abs(ya[good]) + 1e-12))
        corr = np.corrcoef(ya[good], yb[good])[0, 1]
        print(f"{shot:<7d} {name:12s} {rel:12.3e} {corr:7.4f}  {int(good.sum())}")
PY
```

Expected shape of the answer: `bt`, `ip`, `tritop`, `tribot`, `gapin`, `pres` should agree to ~1e-3 or better (the archive built them from these same nodes; residual is decimation and sampling), `kappa` and `r0` similar, `qpsi` possibly worse because of the rho remapping. Save this table into `docs/superpowers/plans/` notes or the module docstring. **A relative difference above ~0.05 on any of the bit-identical features means a wrong node, a wrong time axis, or an inverted dim order — fix it before moving on**, because the same code is what scales to the other 79% of the corpus.

- [ ] **Step 6: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/features/resolve_fdp.py \
        src/labelmaker/features/namespace.py \
        tests/labelmaker/test_resolve_fdp.py
git commit -m "labelmaker: fdp resolver for PTDATA, EFIT01 and ZIPFIT, measured against the archive"
```

---

### Task 13: The runner

**Files:**
- Create: `src/labelmaker/run.py`
- Test: `tests/labelmaker/test_run.py`

**Interfaces:**
- Consumes: `config.Paths`, `config.sha256_of`, `catalog`, `features.{namespace,store,resolve_archive,resolve_corpus,resolve_fdp}`, `models.registry`, `labels.{schema,store}`.
- Produces: `run.main(argv=None) -> int`; `run.build_parser() -> argparse.ArgumentParser`; `run.shot_list(args, paths) -> list[int]`; `run.features_for_shot(shot, names, ctx) -> dict`; `run.infer_for_shot(shot, slug, ctx) -> dict`; `run.RunContext` frozen dataclass; `run.time_limit(seconds)` context manager; `run.write_manifest(paths, run_id, payload) -> Path`.

**Command surface:**

```
python -m labelmaker.run features --models SLUG [SLUG ...] [shot selection] [--workers N] [--timeout S] [--force]
python -m labelmaker.run infer    --models SLUG [SLUG ...] [shot selection] [--workers N] [--timeout S] [--force]
python -m labelmaker.run all      --models SLUG [SLUG ...] [shot selection] [--workers N] [--timeout S] [--force]
shot selection: --shots N [N ...] | --shot-file PATH | --corpus [--sample N] [--seed S] | --overlap [--sample N] [--seed S]
overrides:      --root PATH | --corpus-dir PATH | --archive PATH [PATH ...]
```

`validate` is added to this parser in Task 16, and `all` gains it there. Until then `all` is `features` followed by `infer`.

- [ ] **Step 1: Write the failing test**

`tests/labelmaker/test_run.py`:

```python
"""End-to-end through the CLI on synthetic data, in a temp root."""
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from labelmaker import run
from labelmaker.features import namespace as ns
from labelmaker.labels.store import labelled, read_label
from labelmaker.models.base import (
    InputField,
    InputSpec,
    ModelAdapter,
    OutputField,
    OutputSpec,
)

SLUG = "fake_model"
NAMES = ["bt", "ip", "pinj_total", "ne_zipfit"]


def _archive(tmp_path, shots=(190000, 190001), n=240):
    p = tmp_path / "archive.h5"
    with h5py.File(p, "w") as f:
        for shot in shots:
            g = f.create_group(str(shot))
            g.create_dataset("bt", data=np.full(n, 2.0))
            g.create_dataset("ip", data=np.full(n, 1.0e6))
            g.create_dataset(
                "zipfit_edensfit_rho", data=np.tile(np.linspace(4.0, 1.0, 33), (n, 1))
            )
            # pinj deliberately absent: it must fall through to the corpus
    return p


def _corpus(tmp_path, shots=(190000, 190001)):
    d = tmp_path / "corpus"
    d.mkdir()
    n, t = 2000, np.linspace(0.0, 2.0, 2000, dtype=np.float32)
    for shot in shots:
        with h5py.File(d / f"{shot}_processed.h5", "w") as f:
            g = f.create_group("pinj")
            g.create_dataset("xdata", data=t)
            g.create_dataset("ydata", data=np.full((8, n), 1.0e6, dtype=np.float32))
    return d


def _fake_adapter():
    spec = InputSpec(
        fields=(
            InputField("bt", "bt", lag="t+dt"),
            InputField("ip", "ip", lag="t+dt"),
            InputField("pinj", "pinj_total", lag="t+dt"),
            InputField("ne", "ne_zipfit", lag="t"),
        ),
        dt_s=0.025,
    )
    out = OutputSpec(
        fields=(
            OutputField("score", "binary", column=0, activation="sigmoid"),
        )
    )

    def load(model_dir):
        def predict(built):
            # two "members": the scalar sum, and the profile mean
            a = built.scalars.sum(axis=1)
            b = built.profiles.mean(axis=(1, 2))
            return np.stack([a[:, None], b[:, None]])
        return predict

    return ModelAdapter(
        slug=SLUG, card_id="test/fake-model", framework="none", time_step_ms=25.0,
        artifacts=(), upstream="none", input_spec=spec, output_spec=out,
        load=load, ensemble_n=2,
    )


@pytest.fixture
def wired(tmp_path, monkeypatch):
    from labelmaker.models import registry

    monkeypatch.setattr(registry, "load_adapter", lambda slug: _fake_adapter())
    monkeypatch.setattr(registry, "verify_artifacts", lambda slug, d: None)
    monkeypatch.setattr(
        registry, "read_card",
        lambda slug: {"labelmaker": {"upstream": {"sha256": {"fake.h5": "00"}}}},
    )
    return {
        "archive": _archive(tmp_path),
        "corpus": _corpus(tmp_path),
        "root": tmp_path / "out",
    }


def _argv(wired, stage, *extra):
    return [
        stage, "--models", SLUG, "--shots", "190000", "190001",
        "--root", str(wired["root"]), "--corpus-dir", str(wired["corpus"]),
        "--archive", str(wired["archive"]), "--workers", "1", *extra,
    ]


def test_features_stage_writes_one_file_per_shot(wired):
    assert run.main(_argv(wired, "features")) == 0
    for shot in (190000, 190001):
        p = wired["root"] / "features" / f"{shot}_features.h5"
        assert p.exists()
        with h5py.File(p, "r") as f:
            assert set(f.keys()) == {"bt", "ip", "pinj_total", "ne_zipfit"}
            assert f["bt"].attrs["resolver"] == "archive"
            assert f["pinj_total"].attrs["resolver"] == "corpus"
            assert json.loads(f.attrs["missing"]) == {}


def test_features_stage_skips_complete_shots_and_force_overrides(wired, capsys):
    run.main(_argv(wired, "features"))
    run.main(_argv(wired, "features"))
    assert "skipped" in capsys.readouterr().out
    assert run.main(_argv(wired, "features", "--force")) == 0


def test_infer_stage_writes_labels_and_the_index(wired):
    run.main(_argv(wired, "features"))
    assert run.main(_argv(wired, "infer")) == 0
    p = wired["root"] / "labels" / "190000_labels.h5"
    assert labelled(p) == {f"{SLUG}/score"}
    got = read_label(p, SLUG, "score")
    assert got.y.shape == (1, 240)
    assert ((got.y >= 0) & (got.y <= 1)).all()          # sigmoid applied
    assert got.attrs["task"] == "binary"
    df = pd.read_parquet(wired["root"] / "labels_index.parquet")
    assert set(df["shot"]) == {190000, 190001}
    assert set(df["label"]) == {"score"}
    assert (df["n_total"] == 240).all()


def test_all_chains_the_stages(wired):
    assert run.main(_argv(wired, "all")) == 0
    assert (wired["root"] / "labels" / "190001_labels.h5").exists()


def test_a_broken_shot_does_not_stop_the_run(wired):
    bad = wired["corpus"] / "190001_processed.h5"
    bad.write_bytes(b"not an hdf5 file")
    assert run.main(_argv(wired, "all")) == 0
    assert (wired["root"] / "labels" / "190000_labels.h5").exists()
    with h5py.File(wired["root"] / "features" / "190001_features.h5", "r") as f:
        assert "pinj_total" in json.loads(f.attrs["missing"])


def test_every_run_writes_a_manifest(wired):
    run.main(_argv(wired, "features"))
    runs = sorted((wired["root"] / "runs").iterdir())
    assert len(runs) == 1
    manifest = json.loads((runs[0] / "manifest.json").read_text())
    assert manifest["stage"] == "features"
    assert manifest["models"] == [SLUG]
    assert manifest["shots"] == [190000, 190001]
    assert manifest["git_sha"] and manifest["labelmaker_version"]
    assert manifest["hostname"]
    assert (runs[0] / "log.txt").exists()


def test_shot_selection_from_a_file(wired, tmp_path):
    f = tmp_path / "shots.txt"
    f.write_text("190001\n# comment\n")
    argv = [
        "features", "--models", SLUG, "--shot-file", str(f),
        "--root", str(wired["root"]), "--corpus-dir", str(wired["corpus"]),
        "--archive", str(wired["archive"]), "--workers", "1",
    ]
    assert run.main(argv) == 0
    assert (wired["root"] / "features" / "190001_features.h5").exists()
    assert not (wired["root"] / "features" / "190000_features.h5").exists()


def test_a_shot_selection_is_required(wired):
    with pytest.raises(SystemExit):
        run.main(["features", "--models", SLUG, "--root", str(wired["root"])])


def test_time_limit_raises_rather_than_hanging():
    import time

    with pytest.raises(run.StageTimeout):
        with run.time_limit(1):
            time.sleep(3)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_run.py -q
```
Expected: `ModuleNotFoundError: No module named 'labelmaker.run'`.

- [ ] **Step 3: Teach the store which misses are worth retrying**

`is_complete` currently counts every recorded miss as known, so a shot whose
fdp fetch timed out is skipped on the next run exactly like a shot whose
MDSplus node genuinely does not exist. That defeats the resume behaviour this
stage is built around: a transient failure should be retried, a permanent one
should not, and `--force` should not be the only way to distinguish them.

Add to `src/labelmaker/features/store.py`:

```python
#: Miss causes worth another attempt on a later run. Everything else - an
#: absent node, a shape mismatch, a genuinely one-sample record - is a
#: property of the data and will fail again identically, so a rerun skips it.
TRANSIENT_CAUSES = ("TimeoutError", "OSError", "ConnectionError", "StageTimeout")


def is_transient(cause: str) -> bool:
    """True when a recorded miss is worth retrying on a later run."""
    return any(t in cause for t in TRANSIENT_CAUSES)
```

and change `is_complete` to ignore transient misses:

```python
def is_complete(path, names, *, retry_transient: bool = True) -> bool:
    """True when every requested name is stored or permanently missed.

    A transient miss (a timeout, a dropped connection) does not count as
    known: the next run should try it again. Pass `retry_transient=False`
    to treat any recorded miss as final.
    """
    if not Path(path).exists():
        return False
    misses = missing_names(path)
    if retry_transient:
        misses = {n: c for n, c in misses.items() if not is_transient(c)}
    return set(names) <= (present(path) | set(misses))
```

Add to `tests/labelmaker/test_feature_store.py`:

```python
def test_a_transient_miss_does_not_count_as_complete(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {"bt": "fdp:TimeoutError"})
    assert not is_complete(p, ["ip", "bt"])                    # retry the timeout
    assert is_complete(p, ["ip", "bt"], retry_transient=False)  # unless told not to


def test_a_permanent_miss_counts_as_complete(tmp_path):
    p = tmp_path / "190000_features.h5"
    write_features(p, 190000, {"ip": _scalar()}, {"bt": "archive:KeyError"})
    assert is_complete(p, ["ip", "bt"])


def test_a_demoted_one_sample_feature_is_permanent(tmp_path):
    # A one-sample record is a property of the data, not of the attempt, so
    # it is not worth retrying - unlike a timeout.
    p = tmp_path / "190000_features.h5"
    one = FeatureArray(x=np.zeros(1), y=np.zeros((1, 1)))
    write_features(p, 190000, {"ip": one}, {})
    assert is_complete(p, ["ip"])
```

Import `is_complete` and `is_transient` where the tests need them.

- [ ] **Step 5: Write the implementation**

`src/labelmaker/run.py`:

```python
"""The labelmaker CLI: `python -m labelmaker.run <stage>`.

Three stages, each independently rerunnable, with a per-shot HDF5 file
between them:

    features  resolve the union of canonical features the requested models
              need, per shot, into <root>/features/<shot>_features.h5
    infer     build each model's inputs from that file, predict, and write
              <root>/labels/<shot>_labels.h5
    all       features, then infer

Every shot is isolated: one try/except and one SIGALRM timeout per shot, so
a corrupt HDF5 or a hung read costs one shot and not the run (IGNITE
measured ~56 of 3,000 corpus shots hanging on reads). Workers return summary
rows; the parent writes the parquet index, because many processes appending
to one parquet file would race.

The worker pool is forked before the first fetch and before toksearch is
imported anywhere: toksearch_d3d's ptserver reader is not fork-safe. This is
why `resolve_fdp` defers its imports into its own functions.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import __version__
from .catalog import corpus_shots, overlap_shots, read_shot_file, sample_shots
from .config import Paths, git_sha, sha256_of
from .features import namespace as ns
from .features import resolve_archive, resolve_corpus
from .features.store import (
    FeatureArray,
    is_complete,
    present,
    read_feature,
    write_features,
)
from .labels.schema import artifact_digest, specs_for
from .labels.store import append_index, index_rows, labelled, write_labels
from .models import registry

STAGES = ("features", "infer", "all")


class StageTimeout(Exception):
    """A single shot exceeded its time budget."""


@contextmanager
def time_limit(seconds: int):
    """SIGALRM guard. Works in each worker, which is its process' main thread."""

    def handler(signum, frame):
        raise StageTimeout(f"exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@dataclass(frozen=True)
class RunContext:
    """Everything a worker needs, picklable and read-only."""

    paths: Paths
    archive_files: tuple[Path, ...]
    models: tuple[str, ...]
    run_id: str
    timeout_s: int
    force: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m labelmaker.run",
        description="Run trained models over the FAITH shot corpus.",
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--models", nargs="+", required=True, metavar="SLUG")
    picker = parser.add_mutually_exclusive_group(required=True)
    picker.add_argument("--shots", nargs="+", type=int, metavar="SHOT")
    picker.add_argument("--shot-file", type=Path)
    picker.add_argument("--corpus", action="store_true",
                        help="every shot in the corpus")
    picker.add_argument("--overlap", action="store_true",
                        help="shots present in both the corpus and the "
                             "tearing-mode training archive")
    parser.add_argument("--sample", type=int, default=0,
                        help="with --corpus/--overlap: take a seeded sample")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds per shot per stage")
    parser.add_argument("--force", action="store_true",
                        help="redo shots that are already complete")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--archive", nargs="+", type=Path, default=None)
    return parser


def shot_list(args, paths: Paths) -> list[int]:
    if args.shots:
        shots = sorted(set(args.shots))
    elif args.shot_file:
        shots = read_shot_file(args.shot_file)
    elif args.corpus:
        shots = corpus_shots(paths)
    else:
        shots = overlap_shots(paths)
    if args.sample:
        shots = sample_shots(shots, args.sample, args.seed)
    return shots


def _feature_names(models) -> list[str]:
    """The union of canonical features the requested models consume."""
    names: list[str] = []
    for slug in models:
        adapter = registry.load_adapter(slug)
        for name in adapter.input_spec.canonical_names:
            if name not in names:
                names.append(name)
    return names


def _resolve_one_source(source, shot, want, ctx):
    if source == "archive":
        return resolve_archive.resolve(shot, want, files=tuple(ctx.archive_files))
    if source == "corpus":
        return resolve_corpus.resolve(shot, want, corpus=ctx.paths.corpus)
    from .features import resolve_fdp  # deferred: keeps the parent fork-safe

    return resolve_fdp.resolve(shot, want)


def features_for_shot(shot: int, names, ctx: RunContext) -> dict:
    """Resolve every requested feature for one shot, cheapest source first."""
    path = ctx.paths.features_file(shot)
    if not ctx.force and is_complete(path, names):
        return {"shot": shot, "status": "skipped"}
    have = set() if ctx.force else present(path)
    todo = [n for n in names if n not in have]
    arrays: dict[str, FeatureArray] = {}
    causes: dict[str, list[str]] = {}
    for source in ns.SOURCES:
        want = [
            n for n in todo
            if source in ns.by_name(n).sources and n not in arrays
        ]
        if not want:
            continue
        try:
            got, missed = _resolve_one_source(source, shot, want, ctx)
        except Exception as exc:
            got, missed = {}, {n: type(exc).__name__ for n in want}
        arrays.update(got)
        for name, cause in missed.items():
            causes.setdefault(name, []).append(f"{source}:{cause}")
    missing = {n: ",".join(c) for n, c in causes.items() if n not in arrays}
    write_features(path, shot, arrays, missing, merge=not ctx.force)
    return {
        "shot": shot,
        "status": "ok" if not missing else "partial",
        "resolved": len(arrays),
        "missing": missing,
    }


_PREDICTORS: dict[str, object] = {}


def _predictor(slug: str, ctx: RunContext):
    """Load a model once per worker process, not once per shot."""
    if slug not in _PREDICTORS:
        model_dir = ctx.paths.models / slug
        registry.verify_artifacts(slug, model_dir)
        adapter = registry.load_adapter(slug)
        _PREDICTORS[slug] = (adapter, adapter.load(model_dir))
    return _PREDICTORS[slug]


def infer_for_shot(shot: int, slug: str, ctx: RunContext) -> dict:
    """Build inputs, predict, and write one model's labels for one shot."""
    features_path = ctx.paths.features_file(shot)
    labels_path = ctx.paths.labels_file(shot)
    if not features_path.exists():
        return {"shot": shot, "status": "no-features"}
    adapter, predict = _predictor(slug, ctx)
    if not ctx.force and f"{slug}/{adapter.output_spec.fields[0].name}" in labelled(
        labels_path
    ):
        return {"shot": shot, "status": "skipped"}
    stored = present(features_path)
    features = {
        name: read_feature(features_path, name)
        for name in adapter.input_spec.canonical_names
        if name in stored
    }
    built = adapter.input_spec.build(features, ns.GRID_S)
    members = predict(built)
    decoded = adapter.output_spec.decode(members)
    sha_map = (registry.read_card(slug)["labelmaker"]["upstream"] or {}).get(
        "sha256"
    ) or {}
    specs = specs_for(adapter, artifact_digest(sha_map))
    write_labels(
        labels_path,
        shot,
        built.t,
        decoded,
        specs,
        built.valid,
        run_id=ctx.run_id,
        features_sha256=sha256_of(features_path),
    )
    return {
        "shot": shot,
        "status": "ok",
        "n_valid": int(np.asarray(built.valid).sum()),
        "n_total": int(built.t.size),
        "missing_inputs": list(built.missing),
        "rows": index_rows(labels_path),
    }


def _guarded(fn, shot, ctx, *args):
    started = time.monotonic()
    try:
        with time_limit(ctx.timeout_s):
            row = fn(shot, *args, ctx)
    except Exception as exc:
        row = {"shot": shot, "status": "error", "error": type(exc).__name__,
               "detail": str(exc)[:200]}
    row["seconds"] = round(time.monotonic() - started, 2)
    return row


def _features_worker(payload):
    shot, names, ctx = payload
    return _guarded(features_for_shot, shot, ctx, names)


def _infer_worker(payload):
    shot, slug, ctx = payload
    return _guarded(infer_for_shot, shot, ctx, slug)


def write_manifest(paths: Paths, run_id: str, payload: dict) -> Path:
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def _log(paths: Paths, run_id: str, rows) -> None:
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "log.txt", "a") as fh:
        for row in rows:
            fh.write(json.dumps({k: v for k, v in row.items() if k != "rows"},
                                default=str) + "\n")


def _summarise(stage: str, rows) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    order = sorted(counts.items())
    print(f"{stage}: " + ", ".join(f"{n} {status}" for status, n in order))
    for row in rows:
        if row["status"] == "error":
            print(f"  {row['shot']}: {row.get('error')} {row.get('detail', '')}")


def _run_pool(worker, payloads, workers: int):
    if workers <= 1:
        return [worker(p) for p in payloads]
    # Forked before the first fetch: the ptserver reader is not fork-safe.
    with Pool(workers) as pool:
        return pool.map(worker, payloads)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    base = Paths.from_env()
    paths = Paths(
        root=args.root or base.root,
        corpus=args.corpus_dir or base.corpus,
    )
    paths.mkdirs()
    archive_files = tuple(args.archive) if args.archive else resolve_archive.ARCHIVE_FILES
    shots = shot_list(args, paths)
    if not shots:
        print("no shots selected", file=sys.stderr)
        return 1
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{args.stage}-{stamp}"
    ctx = RunContext(
        paths=paths,
        archive_files=archive_files,
        models=tuple(args.models),
        run_id=run_id,
        timeout_s=args.timeout,
        force=args.force,
    )
    names = _feature_names(args.models)
    write_manifest(
        paths,
        run_id,
        {
            "stage": args.stage,
            "run_id": run_id,
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "labelmaker_version": __version__,
            "git_sha": git_sha(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "workers": args.workers,
            "timeout_s": args.timeout,
            "models": list(args.models),
            "cards": {
                slug: registry.read_card(slug)["labelmaker"].get("upstream", {})
                for slug in args.models
            },
            "features": names,
            "root": str(paths.root),
            "corpus": str(paths.corpus),
            "archive_files": [str(p) for p in archive_files],
            "shots": shots,
        },
    )

    if args.stage in ("features", "all"):
        rows = _run_pool(
            _features_worker, [(s, names, ctx) for s in shots], args.workers
        )
        _log(paths, run_id, rows)
        _summarise("features", rows)

    if args.stage in ("infer", "all"):
        all_rows = []
        for slug in args.models:
            rows = _run_pool(
                _infer_worker, [(s, slug, ctx) for s in shots], args.workers
            )
            _log(paths, run_id, rows)
            _summarise(f"infer {slug}", rows)
            all_rows.extend(rows)
        index = [r for row in all_rows for r in row.get("rows", [])]
        if index:
            append_index(paths.labels_index, index)
            print(f"index: {len(index)} rows -> {paths.labels_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_run.py -q
```
Expected: `9 passed`. The timeout test takes ~1 s of real time; that is the only slow test in the suite.

- [ ] **Step 6: Run the whole suite**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker -q
```
Expected: everything green, with the fdp live test skipped.

- [ ] **Step 7: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/run.py tests/labelmaker/test_run.py
git commit -m "labelmaker: three-stage runner with per-shot isolation and run manifests"
```


---

### Task 14: Validation 1 of 3 — adapter fidelity against TensorFlow

**Files:**
- Create: `src/labelmaker/validate.py` (first section), `tests/labelmaker/data/tearing_golden.npz`
- Create: `tests/labelmaker/test_adapter_fidelity.py`

**The question this answers:** does `runners/keras_h5.py` compute what Keras computes? Everything downstream is worthless if not — and TensorFlow cannot live in this environment, so it runs exactly once, in a throwaway uv environment, and its answer is frozen into a golden file the test suite checks from then on.

**Interfaces:**
- Consumes: `models.registry`, `models.runners.keras_h5`, the tearing adapter.
- Produces: `validate.GOLDEN: Path`; `validate.adapter_fidelity(slug, golden=GOLDEN) -> dict`; `validate.write_report(paths, slug, name, payload) -> Path`.

**About the reference file.** `/projects/EKOLEMEN/simple_ae_predictor/test/test_shots.h5` holds 7 shots x 240 timesteps, scalars `(240,)` and profiles `(240, 33)`, with `times` in ms and `spatial_coordinates` on the 33-point grid. Its columns differ from the training names in three ways, all handled by zero-filling **both** sides identically, so this check tests the evaluator and not the column mapping:

- `tritop_EFITRT1` / `tribot_EFITRT1` / `gapin_EFITRT1` / `pres_EFITRT1` where training used `_EFIT01` — matched by stripping either suffix.
- no `ech_pwr_total`, no `EC.RHO_ECH` — zero-filled.
- no `tm_label` — irrelevant: this compares two implementations, not predictions against truth.

Note also that the upstream harness `test/test.py` cannot read this file at all (it indexes `data[shot]` as a pandas DataFrame, `shot_data[k].values`, while the file stores plain HDF5 datasets — which is why every shot lands in its bare `except`). Build the arrays here rather than reusing that code.

- [ ] **Step 1: Write the one-off script that produces the golden file**

Write it to the scratchpad — it is not package code and runs once:

```python
# /tmp/claude-labelmaker/make_golden.py
"""Run the tearing ensemble under real Keras and freeze inputs + outputs."""
import json

import h5py
import numpy as np
from tensorflow import keras

REF = "/projects/EKOLEMEN/simple_ae_predictor/test/test_shots.h5"
MODELS = "/projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w"
OUT = "/tmp/claude-labelmaker/tearing_golden.npz"

INPUTS_0D = ["bt", "ip", "pinj", "tinj", "R0_EFITRT1", "kappa_EFITRT1",
             "tritop_EFIT01", "tribot_EFIT01", "gapin_EFIT01",
             "ech_pwr_total", "EC.RHO_ECH"]
INPUTS_1D = ["thomson_density_mtanh_1d", "thomson_temp_mtanh_1d",
             "1/qpsi_EFITRT1", "pres_EFIT01", "cer_rot_csaps_1d"]


def base(name):
    for suffix in ("_EFIT01", "_EFITRT1"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def column(group, name, n, width):
    """The reference file's column for a training name, or zeros."""
    want = base(name)
    for key in group:
        if base(key) == want:
            return np.asarray(group[key], dtype=np.float64)
    return np.zeros((n,) if width == 1 else (n, width))


x0_all, x1_all, shots_all = [], [], []
with h5py.File(REF, "r") as f:
    n = f["times"].shape[0]
    for shot in [k for k in f if k.isdigit()]:
        g = f[shot]
        x0 = np.stack([column(g, name, n, 1) for name in INPUTS_0D], axis=1)
        cols = []
        for name in INPUTS_1D:
            if name.startswith("1/"):
                with np.errstate(divide="ignore", invalid="ignore"):
                    cols.append(1.0 / column(g, name[2:], n, 33))
            else:
                cols.append(column(g, name, n, 33))
        x1 = np.stack(cols, axis=2)
        x0_all.append(x0[1:])          # 0-D block at t+dt
        x1_all.append(x1[:-1])         # profile block at t
        shots_all.append(np.full(n - 1, int(shot)))

x0 = np.nan_to_num(np.concatenate(x0_all), nan=0.0, posinf=0.0, neginf=0.0)
x1 = np.nan_to_num(np.concatenate(x1_all), nan=0.0, posinf=0.0, neginf=0.0)
x0, x1 = x0.astype(np.float32), x1.astype(np.float32)
shot_of_row = np.concatenate(shots_all)
print("x0", x0.shape, "x1", x1.shape)

members = []
for i in range(10):
    model = keras.models.load_model(f"{MODELS}/best_model_{i}_4c.h5", compile=False)
    members.append(np.asarray(model.predict([x0, x1], verbose=0)))
members = np.stack(members)
print("members", members.shape, "finite", bool(np.isfinite(members).all()))

np.savez_compressed(
    OUT, x0=x0, x1=x1, members=members.astype(np.float32), shot=shot_of_row,
    meta=json.dumps({"reference_file": REF, "models": MODELS,
                     "keras": keras.__version__,
                     "inputs_0d": INPUTS_0D, "inputs_1d": INPUTS_1D}),
)
print("wrote", OUT)
```

- [ ] **Step 2: Run it under TensorFlow, once**

```bash
mkdir -p /tmp/claude-labelmaker
which uv    # expected: a uv on PATH; Nathan's other projects use it
uv run --no-project --isolated --python 3.11 \
    --with "tensorflow-cpu==2.15.1" --with h5py --with numpy \
    python /tmp/claude-labelmaker/make_golden.py
```
Expected: `x0 (1673, 11) x1 (1673, 33, 5)`, `members (10, 1673, 2) finite True`, then the
`wrote` line. TF 2.15 is the last release whose bundled Keras is version 2, which is
what wrote these files; if the load fails with `Unknown layer`, add
`--with "tf_keras==2.15.1"` and use `import tf_keras as keras` instead — the
`_4c.h5` graphs contain only standard layers, so this should not be needed.

Then move the golden file into the test suite:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
mkdir -p tests/labelmaker/data
cp /tmp/claude-labelmaker/tearing_golden.npz tests/labelmaker/data/
ls -l tests/labelmaker/data/tearing_golden.npz     # expect ~1 MB
```

- [ ] **Step 3: Write the failing test**

`tests/labelmaker/test_adapter_fidelity.py`:

```python
"""The numpy evaluator equals TensorFlow on the upstream reference inputs.

The golden file was produced once by tests/../make_golden.py under
tensorflow-cpu 2.15.1 (see the plan, Task 14). It pins both the inputs and
Keras' own outputs, so this test needs no TensorFlow.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from labelmaker import validate
from labelmaker.models.d3d_tearing_onset_cnn1d import spec as tm
from labelmaker.models.runners.keras_h5 import load_ensemble, predict_members

GOLDEN = Path(__file__).parent / "data" / "tearing_golden.npz"
UPSTREAM = Path(
    "/projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w"
)

pytestmark = [
    pytest.mark.skipif(not GOLDEN.exists(), reason=f"golden file missing: {GOLDEN}"),
    pytest.mark.skipif(not UPSTREAM.exists(), reason=f"weights missing: {UPSTREAM}"),
]


def test_golden_file_records_its_provenance():
    with np.load(GOLDEN, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    assert meta["keras"].startswith("2.")
    assert meta["inputs_0d"][0] == "bt" and len(meta["inputs_0d"]) == 11
    assert len(meta["inputs_1d"]) == 5


def test_numpy_evaluator_matches_keras_to_1e5():
    with np.load(GOLDEN, allow_pickle=False) as z:
        x0, x1, want = z["x0"], z["x1"], z["members"]
    graphs = load_ensemble(UPSTREAM / name for name in tm.ARTIFACTS)
    got = predict_members(graphs, {"input_1": x0, "input_2": x1})
    assert got.shape == want.shape
    assert np.abs(got - want).max() < 1e-5


def test_adapter_fidelity_report_is_a_pass(tmp_path):
    report = validate.adapter_fidelity("d3d_tearing_onset_cnn1d", golden=GOLDEN)
    assert report["passed"] is True
    assert report["max_abs_diff"] < 1e-5
    assert report["n_rows"] > 1000 and report["n_members"] == 10
    out = validate.write_report(
        validate.Paths(root=tmp_path), "d3d_tearing_onset_cnn1d",
        "adapter_fidelity", report,
    )
    assert json.loads(out.read_text())["passed"] is True
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_adapter_fidelity.py -q
```
Expected: `ModuleNotFoundError: No module named 'labelmaker.validate'`.

- [ ] **Step 5: Write the first section of `validate.py`**

`src/labelmaker/validate.py`:

```python
"""How much a model's labels can be trusted, in three measurements.

1. adapter fidelity      - does our evaluator equal the framework's?
2. reconstruction fidelity - do our features equal the model's own training
                             inputs, feature by feature? (Task 15)
3. label quality         - how do the labels score against archived truth,
                           with archived inputs and with ours? The gap
                           between the two is the reconstruction penalty.
                           (Task 16)

Each writes JSON under `<root>/validation/<slug>/`; the headline numbers go
into the model card's `model-index`, so the card is the one place to read how
a model performed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .catalog import TM_ARCHIVE
from .config import Paths
from .models import registry
from .models.runners.keras_h5 import load_ensemble, predict_members

GOLDEN = Path(__file__).resolve().parents[2] / (
    "tests/labelmaker/data/tearing_golden.npz"
)


def write_report(paths: Paths, slug: str, name: str, payload: dict) -> Path:
    """Write one validation report, atomically."""
    out_dir = paths.validation / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return path


def adapter_fidelity(slug: str, golden: Path = GOLDEN, *, tol: float = 1e-5) -> dict:
    """Compare our evaluator against the framework's frozen outputs.

    The golden file holds the reference inputs and the outputs a real Keras
    load produced for the same weights, so this runs with no framework
    installed. A failure here invalidates every label the model has ever
    written, which is why it is a hard pass/fail rather than a metric.
    """
    adapter = registry.load_adapter(slug)
    with np.load(golden, allow_pickle=False) as z:
        x0, x1, want = z["x0"], z["x1"], z["members"]
        meta = json.loads(str(z["meta"]))
    graphs = load_ensemble(Path(meta["models"]) / name for name in adapter.artifacts)
    got = predict_members(graphs, [x0, x1])   # positional; see predict_members
    diff = np.abs(got - want)
    return {
        "slug": slug,
        "golden": str(golden),
        "framework_version": meta.get("keras"),
        "n_rows": int(x0.shape[0]),
        "n_members": int(got.shape[0]),
        "max_abs_diff": float(diff.max()),
        "median_abs_diff": float(np.median(diff)),
        "max_abs_diff_by_column": [float(c) for c in diff.max(axis=(0, 1))],
        "tolerance": tol,
        "passed": bool(diff.max() < tol),
    }
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_adapter_fidelity.py -q
```
Expected: `3 passed`. If `max_abs_diff` lands between 1e-5 and 1e-4, the likely cause is
float32-versus-float64 accumulation in the conv or the batch-norm; check by rerunning
`predict_members` with `x0.astype(np.float64)` — a genuine implementation error shows up
as a difference of order 1e-2 or larger, not 1e-5.

- [ ] **Step 7: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/validate.py tests/labelmaker/data/tearing_golden.npz \
        tests/labelmaker/test_adapter_fidelity.py
git commit -m "labelmaker: adapter fidelity against a frozen TensorFlow reference"
```


---

### Task 15: Validation 2 of 3 — reconstruction fidelity

**Files:**
- Modify: `src/labelmaker/validate.py` (second section)
- Test: `tests/labelmaker/test_reconstruction.py`
- Conditionally modify: `src/labelmaker/features/namespace.py`, `src/labelmaker/models/base.py` (see Step 5)

**The question this answers:** feature by feature, how close are labelmaker's inputs to the rows this model was actually trained on? This is where the substitutions are priced. It is possible at all because the archived training arrays `/projects/EKOLEMEN/tm_data/{x0,x1,y,z}.npy` carry the shot of every row, and five of the eleven scalar columns are bit-identical to the archive store — so the row-to-timestep mapping can be recovered by nearest-neighbour matching on those five, with no timestamps anywhere. Measured on shot 185945: median match distance 2.3e-7, strictly monotonic, 106 of 240 timesteps kept by the upstream filter.

**Interfaces:**
- Consumes: `catalog.TM_ARCHIVE`, `features.store.read_feature`, the adapter's `input_spec`, `scipy.stats`.
- Produces: `validate.MATCH_COLUMNS: tuple[int, ...]`; `validate.archive_rows(shot, archive=TM_ARCHIVE) -> dict | None`; `validate.match_rows(archived_x0, built) -> dict`; `validate.reconstruction_fidelity(slug, shots, paths, *, archive=TM_ARCHIVE) -> dict`.

- [ ] **Step 1: Write the failing test**

`tests/labelmaker/test_reconstruction.py`:

```python
"""Recovering the archived row mapping, and pricing the substitutions."""
from pathlib import Path

import numpy as np
import pytest

from labelmaker import validate
from labelmaker.catalog import TM_ARCHIVE
from labelmaker.config import Paths
from labelmaker.models.base import BuiltInputs

CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")


def _built(n=240, seed=0):
    rng = np.random.default_rng(seed)
    scalars = np.zeros((n, 11))
    scalars[:, 0] = 2.0 + 0.001 * np.arange(n)            # bt, monotone
    scalars[:, 1] = 1e6 + 1e3 * np.arange(n)              # ip
    scalars[:, 6] = 0.4 + 1e-4 * np.arange(n)             # tritop
    scalars[:, 7] = 0.3 + 1e-4 * np.arange(n)             # tribot
    scalars[:, 8] = 0.05 + 1e-5 * np.arange(n)            # gapin
    scalars[:, 2] = rng.normal(size=n)
    profiles = rng.normal(size=(n, 33, 5))
    return BuiltInputs(
        t=0.025 * np.arange(n), scalars=scalars, profiles=profiles,
        valid=np.ones(n, bool), missing=(), resolvers={},
    )


def test_match_rows_recovers_a_known_subset():
    built = _built()
    take = np.array([41, 58, 59, 100, 199])
    archived = built.scalars[take]
    got = validate.match_rows(archived, built)
    np.testing.assert_array_equal(got["index"], take)
    assert got["median_distance"] < 1e-9
    assert got["monotonic"] is True
    assert got["n_matched"] == 5


def test_match_rows_reports_a_bad_match_rather_than_hiding_it():
    built = _built()
    archived = built.scalars[[10, 20]] + 5.0        # nothing like the real rows
    got = validate.match_rows(archived, built)
    assert got["median_distance"] > 1.0
    assert got["passed"] is False


def test_match_rows_needs_the_match_columns_to_vary():
    built = _built()
    built = BuiltInputs(
        t=built.t, scalars=np.zeros_like(built.scalars), profiles=built.profiles,
        valid=built.valid, missing=(), resolvers={},
    )
    with pytest.raises(ValueError, match="constant"):
        validate.match_rows(np.zeros((3, 11)), built)


@pytest.mark.skipif(not TM_ARCHIVE.exists(), reason="tm archive not available")
def test_archive_rows_reads_one_shot():
    got = validate.archive_rows(185945)
    assert got["x0"].shape[1] == 11
    assert got["x1"].shape[1:] == (33, 5)
    assert got["y"].shape[1] == 2
    assert got["x0"].shape[0] == got["y"].shape[0] > 50
    assert validate.archive_rows(1) is None            # not in the archive


@pytest.mark.skipif(
    not (TM_ARCHIVE.exists() and CORPUS.exists()),
    reason="archive or corpus not available",
)
@pytest.mark.skipif(
    not (Paths.from_env().features / "185945_features.h5").exists(),
    reason="run `features` on shot 185945 first",
)
def test_reconstruction_report_on_one_real_shot():
    report = validate.reconstruction_fidelity(
        "d3d_tearing_onset_cnn1d", [185945], Paths.from_env()
    )
    per = report["per_feature"]
    # the five match columns must be near-exact, or the mapping is wrong
    for name in ("bt", "ip", "tritop_EFIT01", "tribot_EFIT01", "gapin_EFIT01"):
        assert per[name]["median_rel"] < 1e-2, (name, per[name])
    # the substituted profiles are the priced ones
    assert per["thomson_density_mtanh_1d"]["corr"] > 0.9
    assert report["match"]["185945"]["median_distance"] < 1e-3
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_reconstruction.py -q
```
Expected: `AttributeError: module 'labelmaker.validate' has no attribute 'match_rows'`.

- [ ] **Step 3: Write the second section of `validate.py`**

Append to `src/labelmaker/validate.py`:

```python
#: Columns of the archived x0 that are bit-identical to the archive store,
#: measured on shot 185945: bt, ip, tritop, tribot, gapin. They are what
#: makes the archived rows addressable - the arrays carry no timestamps.
MATCH_COLUMNS = (0, 1, 6, 7, 8)


def archive_rows(shot: int, archive: Path = TM_ARCHIVE) -> dict | None:
    """The archived training rows for one shot, or None if it has none.

    `z.npy` holds the shot of every row, so this is a mask, not a lookup.
    The arrays are memory-mapped: x1 alone is 422 MB.
    """
    z = np.load(archive / "z.npy", mmap_mode="r")
    rows = np.where(np.asarray(z).astype(np.int64) == int(shot))[0]
    if rows.size == 0:
        return None
    out = {}
    for name in ("x0", "x1", "y"):
        arr = np.load(archive / f"{name}.npy", mmap_mode="r")
        out[name] = np.asarray(arr[rows], dtype=np.float64)
    out["rows"] = rows
    return out


def match_rows(archived_x0: np.ndarray, built, *, tol: float = 1e-3) -> dict:
    """Map each archived row to the timestep of our own inputs.

    Nearest neighbour on the columns that are bit-identical between the two
    sources, each scaled by its own spread so no single column dominates.
    The upstream filter dropped rows, so the mapping is a strictly increasing
    subsequence; `monotonic` is the check that it really is one, and a large
    `median_distance` means the mapping is not to be trusted at all.
    """
    ours = np.asarray(built.scalars, dtype=np.float64)[:, MATCH_COLUMNS]
    theirs = np.asarray(archived_x0, dtype=np.float64)[:, MATCH_COLUMNS]
    scale = np.nanstd(ours, axis=0)
    if not np.all(scale > 0):
        raise ValueError(
            f"match columns are constant in our inputs (std={scale}); "
            "cannot align without variation"
        )
    d = np.linalg.norm(
        (theirs[:, None, :] - ours[None, :, :]) / scale, axis=2
    )
    index = np.nanargmin(d, axis=1)
    distance = np.nanmin(d, axis=1)
    median = float(np.median(distance))
    return {
        "index": index,
        "distance": distance,
        "median_distance": median,
        "max_distance": float(np.nanmax(distance)),
        "monotonic": bool(np.all(np.diff(index) > 0)),
        "n_matched": int(index.size),
        "n_unique": int(np.unique(index).size),
        "passed": bool(median < tol and np.unique(index).size == index.size),
    }


def _stats(ours: np.ndarray, theirs: np.ndarray) -> dict:
    """Agreement between two samples of the same quantity."""
    from scipy import stats

    a = np.asarray(ours, dtype=np.float64).ravel()
    b = np.asarray(theirs, dtype=np.float64).ravel()
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 10:
        return {"n": int(good.sum()), "median_rel": None, "corr": None, "ks": None}
    a, b = a[good], b[good]
    rel = np.median(np.abs(a - b) / (np.abs(b) + 1e-12))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    return {
        "n": int(good.size),
        "median_rel": float(rel),
        "corr": corr,
        "ks": float(stats.ks_2samp(a, b).statistic),
        "mean_ours": float(a.mean()),
        "mean_archive": float(b.mean()),
    }


def reconstruction_fidelity(
    slug: str,
    shots,
    paths: Paths,
    *,
    archive: Path = TM_ARCHIVE,
) -> dict:
    """Price every substitution, per feature, against the training rows."""
    from .features import namespace as ns
    from .features.store import missing_names, present, read_feature
    from .timebase import sample_at

    adapter = registry.load_adapter(slug)
    spec = adapter.input_spec
    names_0d = [f.model_name for f in spec.scalar_fields]
    names_1d = [f.model_name for f in spec.profile_fields]
    pooled: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    match_info: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    incomplete: dict[str, dict] = {}
    resolvers: dict[str, set] = {}
    used: list[int] = []
    ech_conflicts = 0
    ech_rows = 0

    for shot in shots:
        got = archive_rows(shot, archive)
        if got is None:
            skipped[str(shot)] = "no archived rows"
            continue
        fpath = paths.features_file(shot)
        if not fpath.exists():
            skipped[str(shot)] = "no feature file"
            continue
        stored = present(fpath)
        features = {
            name: read_feature(fpath, name)
            for name in spec.canonical_names
            if name in stored
        }
        built = spec.build(features, ns.GRID_S)
        for canonical, source in built.resolvers.items():
            resolvers.setdefault(canonical, set()).add(source)
        info = match_rows(got["x0"], built)
        match_info[str(shot)] = {
            k: v for k, v in info.items() if k not in ("index", "distance")
        }
        if not info["passed"]:
            skipped[str(shot)] = (
                f"match rejected (median {info['median_distance']:.3g})"
            )
            continue
        idx = info["index"]
        used.append(int(shot))
        for j, name in enumerate(names_0d):
            pooled.setdefault(name, []).append(
                (built.scalars[idx, j], got["x0"][:, j])
            )
        for j, name in enumerate(names_1d):
            pooled.setdefault(name, []).append(
                (built.profiles[idx, :, j], got["x1"][:, :, j])
            )
        # Diagnostic for the zero-filled ECH deposition location: how often
        # is the location unknown while power is actually being injected?
        if "ech_power_total" in features and "ech_rho" in features:
            raw = features["ech_rho"]
            rho = np.asarray(
                sample_at(raw.x, raw.y, ns.GRID_S + spec.dt_s, max_gap=spec.dt_s)
            ).ravel()[idx]
            power = built.scalars[idx, names_0d.index("ech_pwr_total")]
            ech_rows += int(power.size)
            ech_conflicts += int(((power > 0) & ~np.isfinite(rho)).sum())
        incomplete[str(shot)] = missing_names(fpath)

    per_feature = {
        name: _stats(
            np.concatenate([np.asarray(a).ravel() for a, _ in pairs]),
            np.concatenate([np.asarray(b).ravel() for _, b in pairs]),
        )
        for name, pairs in pooled.items()
    }
    return {
        "slug": slug,
        "n_shots_requested": len(list(shots)),
        "n_shots_used": len(used),
        "shots_used": used,
        "skipped": skipped,
        "incomplete_features": {k: v for k, v in incomplete.items() if v},
        "match": match_info,
        "per_feature": per_feature,
        "resolvers": {k: sorted(v) for k, v in resolvers.items()},
        "ech_location_unknown_while_powered": {
            "rows": ech_rows,
            "conflicts": ech_conflicts,
        },
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_reconstruction.py -q
```
Expected: `3 passed, 2 skipped` before any real run; after Task 17's `features` stage
the two real-data tests run as well.

- [ ] **Step 5: Decide the sampling convention by measurement**

The corpus is a 10 kHz instantaneous record; the archive's 25 ms value is a window
statistic. On shot 185945 at t = 1.025 s the corpus sum reads 10.02 MW where the
archive reads 10.996 MW — about 10%, which is sampling, not units. Measure which
convention reproduces the archive:

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
import numpy as np
from labelmaker.config import Paths
from labelmaker.features import namespace as ns
from labelmaker.features import resolve_archive as ra, resolve_corpus as rc
from labelmaker.timebase import sample_at, window_mean

paths = Paths.from_env()
for shot in (185945, 186697, 187327):
    for name in ("pinj_total", "tinj_total"):
        arch, _ = ra.resolve(shot, [name])
        cor, _ = rc.resolve(shot, [name], corpus=paths.corpus)
        if name not in arch or name not in cor:
            print(f"{shot} {name}: unavailable")
            continue
        ref = arch[name].y[0]
        near = np.asarray(sample_at(cor[name].x, cor[name].y, ns.GRID_S,
                                    max_gap=0.025)).ravel()
        wind = np.asarray(window_mean(cor[name].x, cor[name].y, ns.GRID_S,
                                      0.025)).ravel()
        back = np.asarray(window_mean(cor[name].x, cor[name].y,
                                      ns.GRID_S - 0.025, 0.025)).ravel()
        for label, series in (("nearest", near), ("window", wind),
                              ("window-back", back)):
            good = np.isfinite(ref) & np.isfinite(series) & (np.abs(ref) > 1e-6)
            rel = np.median(np.abs(series[good] - ref[good]) / np.abs(ref[good]))
            print(f"{shot} {name:11s} {label:12s} median rel = {rel:.4f}"
                  f"  (n={int(good.sum())})")
PY
```

Act on the answer:

- If `nearest` is within ~1.5x of the best, keep it and record the numbers in the
  `notes` of those two `FeatureSpec`s. Nothing else changes.
- If a window convention is materially better (more than ~2x lower), adopt it. Add
  `sample: str = "nearest"` to `FeatureSpec` in `namespace.py`, set
  `sample="window_mean"` (or `"window_mean_back"`) on `pinj_total`, `tinj_total` and
  `ech_power_total`, and in `InputSpec.build` in `models/base.py` replace the
  sampling line:

  ```python
            t = grid + self.dt_s if f.lag == "t+dt" else grid
            mode = ns.by_name(f.canonical).sample
            if mode == "window_mean":
                vals = window_mean(arr.x, arr.y, t, self.dt_s)
            elif mode == "window_mean_back":
                vals = window_mean(arr.x, arr.y, t - self.dt_s, self.dt_s)
            else:
                vals = sample_at(arr.x, arr.y, t, max_gap=self.dt_s)
            vals = np.atleast_2d(vals)
  ```

  with `from ..timebase import sample_at, window_mean` at the top, then add a test in
  `test_model_base.py` asserting a `window_mean` feature averages its window, and
  rerun the whole suite.

- [ ] **Step 6: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/validate.py tests/labelmaker/test_reconstruction.py \
        src/labelmaker/features/namespace.py src/labelmaker/models/base.py \
        tests/labelmaker/test_model_base.py
git commit -m "labelmaker: reconstruction fidelity against the archived training rows"
```


---

### Task 16: Validation 3 of 3 — label quality, and `validate` in the CLI

**Files:**
- Modify: `src/labelmaker/validate.py` (third section), `src/labelmaker/models/registry.py` (add `update_model_index`), `src/labelmaker/run.py` (add the `validate` stage)
- Test: `tests/labelmaker/test_label_quality.py`

**The question this answers:** are the labels any good, and how much of any shortfall is ours? Both are answered at once by scoring the model twice against the same archived truth — once on the rows it was trained on, once on labelmaker's reconstruction of those rows. The first number is the model's own ceiling; the difference is the price of the reconstruction.

**Truth source.** `y.npy` column 1 (`tm_label`), row-aligned with `x0`/`x1` and addressable per shot through `z.npy`. `ntm_labels.pkl` (5,834 shots of 50 ms series) is deliberately *not* used in Phase 1: its per-shot lengths vary (82-127 entries) and it records no time origin, so aligning it is guesswork where `y.npy` is exact. It returns in Phase 2 as an independent cross-check once its origin is established.

**No scikit-learn.** It is not in this environment and is not worth adding for three metrics. AUROC comes from the rank identity (Mann-Whitney U over `scipy.stats.rankdata`), which is exact, and F1, Brier and the calibration bins are a few lines each.

**Interfaces:**
- Produces: `validate.binary_metrics(prob, truth, *, bins=10) -> dict`; `validate.label_quality(slug, shots, paths, *, archive=TM_ARCHIVE) -> dict`; `validate.model_index_results(reports) -> list[dict]`; `registry.update_model_index(slug, results) -> None`; `run` gains the `validate` stage and `all` chains it.

- [ ] **Step 1: Write the failing test**

`tests/labelmaker/test_label_quality.py`:

```python
"""Metrics, and the card that records them."""
import numpy as np
import pytest

from labelmaker import validate
from labelmaker.models import registry


def test_auroc_matches_hand_computed_cases():
    truth = np.array([0, 0, 1, 1])
    assert validate.binary_metrics(np.array([0.1, 0.2, 0.8, 0.9]), truth)["auroc"] == 1.0
    assert validate.binary_metrics(np.array([0.9, 0.8, 0.2, 0.1]), truth)["auroc"] == 0.0
    assert validate.binary_metrics(np.array([0.5, 0.5, 0.5, 0.5]), truth)["auroc"] == 0.5
    # one swapped pair out of four -> 0.75
    got = validate.binary_metrics(np.array([0.1, 0.85, 0.8, 0.9]), truth)["auroc"]
    assert abs(got - 0.75) < 1e-12


def test_f1_and_brier_and_calibration():
    truth = np.array([0, 0, 1, 1, 1])
    prob = np.array([0.2, 0.7, 0.9, 0.6, 0.3])
    got = validate.binary_metrics(prob, truth, bins=2)
    # at 0.5: predictions 0,1,1,1,0 -> tp=2, fp=1, fn=1 -> f1 = 2*2/(2*2+1+1)
    assert abs(got["f1_at_0.5"] - 4 / 6) < 1e-12
    assert abs(got["brier"] - np.mean((prob - truth) ** 2)) < 1e-12
    assert got["n"] == 5 and got["n_positive"] == 3
    assert len(got["calibration"]) == 2
    assert 0.0 <= got["ece"] <= 1.0


def test_metrics_refuse_a_degenerate_truth_vector():
    got = validate.binary_metrics(np.array([0.1, 0.2]), np.array([0, 0]))
    assert got["auroc"] is None and got["n_positive"] == 0


def test_regression_metrics():
    got = validate.regression_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    assert abs(got["rmse"] - 0.5) < 1e-12
    assert abs(got["bias"] + 0.5) < 1e-12


def test_model_index_results_shape():
    results = validate.model_index_results(
        {
            "adapter_fidelity": {"passed": True, "max_abs_diff": 1e-7},
            "label_quality": {
                "archived_inputs": {"tm_prob": {"auroc": 0.9, "f1_at_0.5": 0.5}},
                "reconstructed_inputs": {"tm_prob": {"auroc": 0.8, "f1_at_0.5": 0.4}},
            },
        }
    )
    names = {r["metrics"][0]["name"] for r in results}
    assert "auroc (archived inputs)" in names
    assert "auroc (reconstructed inputs)" in names
    for r in results:
        assert r["task"]["type"] and r["dataset"]["name"]
        assert isinstance(r["metrics"][0]["value"], float)


def test_update_model_index_rewrites_only_the_results(tmp_path, monkeypatch):
    card = tmp_path / "README.md"
    card.write_text(
        "---\n"
        "library_name: keras\n"
        "model-index:\n"
        "  - name: d3d-tearing-onset-cnn1d\n"
        "    results: []\n"
        "labelmaker:\n"
        "  status: implemented\n"
        "  slug: s\n"
        "---\n\n# Title\n\nProse that must survive.\n"
    )
    monkeypatch.setattr(registry, "card_path", lambda slug: card)
    registry.update_model_index(
        "s",
        [
            {
                "task": {"type": "tabular-classification"},
                "dataset": {"name": "d3d overlap shots", "type": "d3d"},
                "metrics": [{"name": "auroc (archived inputs)", "type": "roc_auc",
                             "value": 0.91}],
            }
        ],
    )
    text = card.read_text()
    assert "Prose that must survive." in text
    parsed = registry.parse_card(text)
    assert parsed["model-index"][0]["results"][0]["metrics"][0]["value"] == 0.91
    assert parsed["labelmaker"]["status"] == "implemented"
    assert parsed["library_name"] == "keras"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_label_quality.py -q
```
Expected: `AttributeError: module 'labelmaker.validate' has no attribute 'binary_metrics'`.

- [ ] **Step 3: Write the metrics and the third report**

Append to `src/labelmaker/validate.py`:

```python
def binary_metrics(prob: np.ndarray, truth: np.ndarray, *, bins: int = 10) -> dict:
    """AUROC, F1 at 0.5, Brier, and a calibration curve.

    AUROC is the rank identity rather than a trapezoid over a sampled ROC,
    so it is exact and needs no scikit-learn (which this environment does
    not have). Ties get mid-ranks, which is the correct convention for a
    model that emits identical probabilities for different rows.
    """
    from scipy import stats

    prob = np.asarray(prob, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    good = np.isfinite(prob) & np.isfinite(truth)
    prob, truth = prob[good], (truth[good] > 0.5)
    n_pos, n_neg = int(truth.sum()), int((~truth).sum())
    out: dict = {
        "n": int(prob.size),
        "n_positive": n_pos,
        "positive_fraction": float(n_pos / prob.size) if prob.size else None,
        "auroc": None,
        "f1_at_0.5": None,
        "brier": None,
        "ece": None,
        "calibration": [],
    }
    if prob.size == 0 or n_pos == 0 or n_neg == 0:
        return out
    ranks = stats.rankdata(prob)
    out["auroc"] = float(
        (ranks[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )
    pred = prob >= 0.5
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    out["f1_at_0.5"] = float(2 * tp / (2 * tp + fp + fn)) if tp or fp or fn else 0.0
    out["precision_at_0.5"] = float(tp / (tp + fp)) if tp + fp else None
    out["recall_at_0.5"] = float(tp / (tp + fn)) if tp + fn else None
    out["brier"] = float(np.mean((prob - truth.astype(np.float64)) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = which == b
        if not sel.any():
            out["calibration"].append(
                {"bin": b, "n": 0, "mean_prob": None, "observed": None}
            )
            continue
        mean_prob = float(prob[sel].mean())
        observed = float(truth[sel].mean())
        out["calibration"].append(
            {"bin": b, "n": int(sel.sum()), "mean_prob": mean_prob,
             "observed": observed}
        )
        ece += sel.sum() / prob.size * abs(mean_prob - observed)
    out["ece"] = float(ece)
    return out


def regression_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    good = np.isfinite(pred) & np.isfinite(truth)
    pred, truth = pred[good], truth[good]
    if pred.size == 0:
        return {"n": 0, "rmse": None, "bias": None, "corr": None}
    return {
        "n": int(pred.size),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "bias": float(np.mean(pred - truth)),
        "corr": float(np.corrcoef(pred, truth)[0, 1])
        if pred.std() > 0 and truth.std() > 0
        else None,
    }


def label_quality(
    slug: str,
    shots,
    paths: Paths,
    *,
    archive: Path = TM_ARCHIVE,
) -> dict:
    """Score the model twice against the same truth: their rows, then ours.

    The first score is the model's own ceiling on these shots. The second is
    what labelmaker actually produces. The difference is the price of the
    reconstruction, and it is the number that answers "are these labels
    reliable".
    """
    from .features import namespace as ns
    from .features.store import present, read_feature
    from .models.base import BuiltInputs

    adapter = registry.load_adapter(slug)
    spec = adapter.input_spec
    predict = adapter.load(paths.models / slug)
    truth_col = {"tm_prob": 1, "betan": 0}

    arch_pred: dict[str, list[np.ndarray]] = {}
    ours_pred: dict[str, list[np.ndarray]] = {}
    truths: list[np.ndarray] = []
    used: list[int] = []
    skipped: dict[str, str] = {}

    for shot in shots:
        got = archive_rows(shot, archive)
        if got is None:
            skipped[str(shot)] = "no archived rows"
            continue
        fpath = paths.features_file(shot)
        if not fpath.exists():
            skipped[str(shot)] = "no feature file"
            continue
        stored = present(fpath)
        features = {
            name: read_feature(fpath, name)
            for name in spec.canonical_names
            if name in stored
        }
        built = spec.build(features, ns.GRID_S)
        info = match_rows(got["x0"], built)
        if not info["passed"]:
            skipped[str(shot)] = f"match rejected ({info['median_distance']:.3g})"
            continue
        idx = info["index"]
        theirs = BuiltInputs(
            t=built.t[idx], scalars=got["x0"], profiles=got["x1"],
            valid=np.ones(idx.size, bool), missing=(), resolvers={},
        )
        ours = BuiltInputs(
            t=built.t[idx], scalars=built.scalars[idx],
            profiles=built.profiles[idx], valid=built.valid[idx],
            missing=built.missing, resolvers=built.resolvers,
        )
        a = adapter.output_spec.decode(predict(theirs))
        b = adapter.output_spec.decode(predict(ours))
        for name in a:
            arch_pred.setdefault(name, []).append(a[name].mean)
            ours_pred.setdefault(name, []).append(b[name].mean)
        truths.append(got["y"])
        used.append(int(shot))

    report: dict = {
        "slug": slug,
        "truth": str(archive / "y.npy"),
        "n_shots_used": len(used),
        "shots_used": used,
        "skipped": skipped,
        "archived_inputs": {},
        "reconstructed_inputs": {},
    }
    if not used:
        return report
    y = np.concatenate(truths)
    for name in arch_pred:
        col = truth_col.get(name)
        if col is None:
            continue
        t = y[:, col]
        pa = np.concatenate(arch_pred[name])
        po = np.concatenate(ours_pred[name])
        field = next(f for f in adapter.output_spec.fields if f.name == name)
        scorer = binary_metrics if field.task == "binary" else regression_metrics
        report["archived_inputs"][name] = scorer(pa, t)
        report["reconstructed_inputs"][name] = scorer(po, t)
    for name in report["archived_inputs"]:
        a = report["archived_inputs"][name]
        o = report["reconstructed_inputs"][name]
        key = "auroc" if "auroc" in a else "rmse"
        if a.get(key) is not None and o.get(key) is not None:
            report.setdefault("reconstruction_penalty", {})[name] = {
                key: float(o[key] - a[key])
            }
    return report


def model_index_results(reports: dict) -> list[dict]:
    """The headline numbers, in HuggingFace `model-index` shape."""
    quality = reports.get("label_quality") or {}
    results: list[dict] = []
    for source, suffix in (
        ("archived_inputs", "archived inputs"),
        ("reconstructed_inputs", "reconstructed inputs"),
    ):
        for label, metrics in (quality.get(source) or {}).items():
            entries = []
            for key, mtype in (("auroc", "roc_auc"), ("f1_at_0.5", "f1"),
                               ("rmse", "rmse")):
                value = metrics.get(key)
                if value is None:
                    continue
                entries.append(
                    {"name": f"{key} ({suffix})", "type": mtype,
                     "value": float(value)}
                )
            if not entries:
                continue
            kinds = {e["type"] for e in entries}
            task_type = (
                "tabular-regression" if kinds == {"rmse"} else "tabular-classification"
            )
            results.append(
                {
                    "task": {"type": task_type, "name": label},
                    "dataset": {
                        "name": f"d3d overlap shots (n={quality.get('n_shots_used')})",
                        "type": "d3d-faith-corpus",
                    },
                    "metrics": entries,
                }
            )
    return results
```

- [ ] **Step 4: Add `update_model_index` to `registry.py`**

```python
def update_model_index(slug: str, results: list[dict]) -> None:
    """Write validation results into the card's `model-index`, in place.

    Only the front matter is rewritten - the prose below it is copied
    verbatim - so a card keeps its human-written sections while its numbers
    stay generated. The front matter carries no comments, so a safe_dump
    round trip is lossless.
    """
    path = card_path(slug)
    text = path.read_text()
    m = _FRONT_MATTER.match(text)
    if not m:
        raise ValueError(f"{slug}: card has no front matter to update")
    data = yaml.safe_load(m.group(1))
    index = data.get("model-index") or [{"name": slug.replace("_", "-")}]
    index[0]["results"] = results
    data["model-index"] = index
    body = text[m.end():]
    dumped = yaml.safe_dump(data, sort_keys=False, allow_unicode=True,
                            default_flow_style=False, width=100)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(f"---\n{dumped}---\n{body}")
    tmp.replace(path)
```

- [ ] **Step 5: Add the `validate` stage to `run.py`**

Change `STAGES` and add the branch:

```python
STAGES = ("features", "infer", "validate", "all")
```

```python
    if args.stage in ("validate", "all"):
        from . import validate as validation

        for slug in args.models:
            reports = {}
            try:
                reports["adapter_fidelity"] = validation.adapter_fidelity(slug)
            except FileNotFoundError as exc:
                reports["adapter_fidelity"] = {"skipped": str(exc)}
            reports["reconstruction"] = validation.reconstruction_fidelity(
                slug, shots, paths
            )
            reports["label_quality"] = validation.label_quality(slug, shots, paths)
            for name, payload in reports.items():
                out = validation.write_report(paths, slug, name, payload)
                print(f"validate {slug}: {name} -> {out}")
            results = validation.model_index_results(reports)
            if results:
                registry.update_model_index(slug, results)
                print(f"validate {slug}: {len(results)} results written to the card")
            fidelity = reports["adapter_fidelity"]
            if fidelity.get("passed") is False:
                print(f"validate {slug}: ADAPTER FIDELITY FAILED "
                      f"(max_abs_diff={fidelity.get('max_abs_diff')})",
                      file=sys.stderr)
                return 2
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m pytest tests/labelmaker/test_label_quality.py -q
pixi run -e labelmaker python -m pytest tests/labelmaker -q
```
Expected: `6 passed`, then the whole suite green.

- [ ] **Step 7: Commit**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add src/labelmaker/validate.py src/labelmaker/models/registry.py \
        src/labelmaker/run.py tests/labelmaker/test_label_quality.py
git commit -m "labelmaker: label-quality metrics, reconstruction penalty, validate stage"
```

---

### Task 17: The proof of concept — 100 shots end to end

**Files:**
- Create: `docs/LABELMAKER.md`
- Modify: `docs/superpowers/specs/2026-09-03-labelmaker-design.md` (fold in the plan's Deviations)
- Modify: `src/labelmaker/models/d3d_tearing_onset_cnn1d/README.md` (written by the run)

This task produces the deliverable: tearing-mode labels for 100 corpus shots, and the numbers that say how much to trust them.

- [ ] **Step 1: Run the feature stage**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
time pixi run -e labelmaker python -m labelmaker.run features \
    --models d3d_tearing_onset_cnn1d \
    --shot-file "$(pixi run -e labelmaker python -c 'from labelmaker.config import Paths; print(Paths.from_env().root / "poc_shots.txt")')" \
    --workers 8 --timeout 300
```
Expected: `features: 100 ok` (or a few `partial`), in a couple of minutes — the archive
resolver is a local HDF5 read and the corpus resolver reads three groups per shot. Any
`error` rows are printed with their exception class; investigate those before continuing.
Check one file by hand:

```bash
pixi run -e labelmaker python - <<'PY'
import json
import h5py
from labelmaker.catalog import read_shot_file
from labelmaker.config import Paths

paths = Paths.from_env()
shots = read_shot_file(paths.root / "poc_shots.txt")
with h5py.File(paths.features_file(shots[0]), "r") as f:
    print("shot", f.attrs["shot"], "features", sorted(f.keys()))
    print("missing", json.loads(f.attrs["missing"]))
    for name in sorted(f):
        g = f[name]
        print(f"  {name:18s} {str(g['ydata'].shape):12s} "
              f"resolver={g.attrs.get('resolver')}")
PY
```
Expected: all 16 features present, `missing` empty or naming only `ech_rho`, resolvers
`archive` for everything except the three actuator totals, which read `corpus` only if the
archive lacked them.

- [ ] **Step 2: Run inference**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
time pixi run -e labelmaker python -m labelmaker.run infer \
    --models d3d_tearing_onset_cnn1d \
    --shot-file "$(pixi run -e labelmaker python -c 'from labelmaker.config import Paths; print(Paths.from_env().root / "poc_shots.txt")')" \
    --workers 8 --timeout 300
```
Expected: `infer d3d_tearing_onset_cnn1d: 100 ok`, then `index: 200 rows -> .../labels_index.parquet`
(two labels per shot). Ten 12k-parameter graphs over 240 rows is milliseconds; the time
is dominated by reading the feature files.

- [ ] **Step 3: Look at the labels before trusting the metrics**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
import numpy as np
import pandas as pd
from labelmaker.config import Paths

paths = Paths.from_env()
df = pd.read_parquet(paths.labels_index)
tm = df[df["label"] == "tm_prob"]
print(f"shots labelled: {tm['shot'].nunique()}")
print(f"valid timesteps: {tm['n_valid'].sum()} of {tm['n_total'].sum()} "
      f"({100 * tm['n_valid'].sum() / tm['n_total'].sum():.1f}%)")
print(f"mean tm_prob over valid rows: {tm['mean_valid'].mean():.4f}")
print(f"shots with any tm_prob > 0.5: "
      f"{int((tm['max_valid'] > 0.5).sum())} of {len(tm)}")
print(tm[["shot", "n_valid", "n_total", "mean_valid", "max_valid"]].head(10)
        .to_string(index=False))
PY
```

Sanity expectations, from the training distribution (7.9% positive) and the domain filter
(which kept 106 of 240 timesteps on shot 185945):
- **valid fraction 30-70%.** Much higher means a domain rule is not being applied; much
  lower means a feature is missing or in the wrong units. Either way, stop and find it.
- **mean `tm_prob` over valid rows of order 0.05-0.3**, and a minority of shots reaching
  above 0.5. A mean near 0.5 with no structure means the inputs are not reaching the model
  (all-zero inputs through a batch-norm produce a constant), which the next step catches
  numerically.

- [ ] **Step 4: Run validation**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python -m labelmaker.run validate \
    --models d3d_tearing_onset_cnn1d \
    --shot-file "$(pixi run -e labelmaker python -c 'from labelmaker.config import Paths; print(Paths.from_env().root / "poc_shots.txt")')"
cat "$(pixi run -e labelmaker python -c 'from labelmaker.config import Paths; print(Paths.from_env().validation / "d3d_tearing_onset_cnn1d" / "label_quality.json")')"
```

What the numbers should look like, and what each failure would mean:

| number | expectation | if it is off |
|---|---|---|
| `adapter_fidelity.max_abs_diff` | < 1e-5 | the numpy evaluator is wrong; nothing else matters until it is fixed |
| `reconstruction.per_feature[bt/ip/tritop/tribot/gapin].median_rel` | < 1e-6 | the row match or the archive read is wrong |
| `reconstruction.per_feature[pres_EFIT01].median_rel` | < 1e-6 | as above; this column is bit-identical upstream |
| `reconstruction.per_feature[R0_EFITRT1].median_rel` | ~9e-3 | a different EFIT node than `rmaxis` |
| `reconstruction.per_feature[thomson_density_mtanh_1d].corr` | > 0.95, `median_rel` ~0.2 | the ZIPFIT substitution behaving as measured on 185945 |
| `label_quality.archived_inputs.tm_prob.auroc` | > 0.9 | the model scored on its own training rows; a low value means the ensemble or the sigmoid is wrong |
| `label_quality.reconstructed_inputs.tm_prob.auroc` | 0.75-0.9 | this is the headline result |
| `reconstruction_penalty.tm_prob.auroc` | -0.20 to 0.00 | a penalty worse than -0.2 makes the ZIPFIT substitution the first thing Phase 2 should fix |
| `ech_location_unknown_while_powered.conflicts` | small next to `rows` | if large, the zero-filled `EC.RHO_ECH` is a real bias and needs the conditional fill |

Note that `archived_inputs` scores the model on rows it was trained on, so it is an
optimistic ceiling, not a held-out result. The honest reading of this task is the *gap*.

- [ ] **Step 5: Confirm the card now carries the numbers**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
pixi run -e labelmaker python - <<'PY'
from labelmaker.models import registry

card = registry.read_card("d3d_tearing_onset_cnn1d")
for result in card["model-index"][0]["results"]:
    for metric in result["metrics"]:
        print(f"{result['task']['name']:10s} {metric['name']:34s} {metric['value']:.4f}")
print("discrepancies:", registry.card_discrepancies("d3d_tearing_onset_cnn1d"))
PY
pixi run -e labelmaker python -m pytest tests/labelmaker -q
```
Expected: four to six metric lines, `discrepancies: []`, and a green suite — the card
rewrite must not have broken the front matter that `test_registry.py` checks.

- [ ] **Step 6: Write the operator guide**

`docs/LABELMAKER.md`:

```markdown
# labelmaker

Runs the group's trained models over the FAITH shot corpus and writes their
predictions as per-shot label files in the corpus HDF5 layout, with a measured
statement of how much each model's labels can be trusted.

Design: `docs/superpowers/specs/2026-09-03-labelmaker-design.md`.
Phase 1 build log: `docs/superpowers/plans/2026-09-03-labelmaker-phase1.md`.

## Running it

```bash
pixi install -e labelmaker            # once
pixi run -e labelmaker fdp login      # only if you need the fdp resolver

pixi run -e labelmaker python -m labelmaker.run all \
    --models d3d_tearing_onset_cnn1d \
    --overlap --sample 100 --seed 20260903 --workers 8
```

Shot selection is one of `--shots N N`, `--shot-file PATH`, `--corpus`, or
`--overlap` (shots present in both the corpus and a model's training archive),
with `--sample N --seed S` to take a reproducible subset. Stages are
`features`, `infer`, `validate` and `all`; each is independently rerunnable and
skips work that is already complete unless given `--force`.

## What it writes

Everything under `$LABELMAKER_ROOT` (default
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker`):

| path | contents |
|---|---|
| `features/<shot>_features.h5` | canonical features, one group each, `xdata`/`ydata` |
| `labels/<shot>_labels.h5` | `<slug>/<label>` plus `_spread` and `_valid` companions |
| `labels_index.parquet` | one row per shot, model and label - the "which shots have labels" query |
| `models/<slug>/` | the weights, copied once, with `PROVENANCE.json` |
| `runs/<run_id>/` | `manifest.json` (config, git sha, shot list, artifact digests) and `log.txt` |
| `validation/<slug>/` | the three reliability reports |

Labels are probabilities, never thresholded, on the model's own time step.
`<label>_valid` is 1 where every input was present and inside the model's
training domain; a probability where it is 0 is an extrapolation.

## Reading a label

```python
import h5py

with h5py.File(".../labels/190000_labels.h5") as f:
    g = f["d3d_tearing_onset_cnn1d/tm_prob"]
    t, p = g["xdata"][:], g["ydata"][0]          # seconds, probability
    valid = f["d3d_tearing_onset_cnn1d/tm_prob_valid/ydata"][0].astype(bool)
    print(g.attrs["card_id"], g.attrs["task"], g.attrs["artifact_sha256"][:12])
```

## Adding a model

1. `src/labelmaker/models/<device>_<phenomenon>_<predicted>_<arch>/`, with
   `README.md` (HuggingFace card plus the `labelmaker:` block), `spec.py`
   (`ADAPTER`), and an empty `__init__.py`. Copy the closest existing folder.
2. Map each trained-on input name to a canonical feature in `namespace.py`. Add
   a `FeatureSpec` only if the quantity is genuinely new, and give it every
   source that can serve it, cheapest first.
3. Put the training-time filter in as `DomainRule`s. Labelmaker flags rows
   outside them instead of dropping them.
4. Load the weights from their upstream location, copy them into
   `<root>/models/<slug>/`, and record the sha256 in the card - inference
   refuses to run against bytes the card does not know.
5. `pixi run -e labelmaker python -m pytest tests/labelmaker` - the registry
   tests check that the card and `spec.py` agree.

## Known limits

- The archive resolver covers 3,621 of 16,909 corpus shots (21%); everything
  else needs the fdp resolver, which is built and measured but has not been run
  at corpus scale.
- The three kinetic profiles are ZIPFIT fits standing in for the tearing model's
  own mtanh/csaps fits, which costs about 20% in median relative profile error.
  Fitting them from raw Thomson and CER needs channel geometry and is Phase 2.
- Only one model is implemented. The other six roster folders are scaffolds
  whose cards say what blocks each of them.
```

- [ ] **Step 7: Fold the plan's deviations back into the spec**

Edit `docs/superpowers/specs/2026-09-03-labelmaker-design.md` so it describes what was
built, not what was planned. Minimum edits, each one line to a short paragraph:

- §6 package layout: drop `geometry.py`, `resolve_fits.py` and `runners/torch.py`; add
  `resolve_archive.py`; note `runners/keras_h5.py` is a numpy evaluator.
- §7: the `labelmaker` environment is `["labelmaker", "fdp"]` with `pyarrow` and CPU torch
  wheels, no `tensorflow-cpu` (no py3.11 build exists) and no `cuda`.
- §9.1: canonical names are physics quantities with an ordered `sources` tuple; provenance
  is the per-shot `resolver` attribute.
- §9.4: add the archive resolver as the primary source, with the measured agreement table
  from the plan's Deviation 6, and the measured corpus units (`pinj` in W).
- §9.5: geometry moves to Phase 2, with the reason (ZIPFIT already covers the profiles at a
  measured 20% cost) and the finding that the corpus carries no channel positions at all.
- §10.4: correct the upstream path to `mse_bin_os_w/` and note the two-column single output
  of the `_4c.h5` artifacts.
- §13: replace the cross-correlation alignment with the exact row match on the
  bit-identical columns.
- §16: mark the geometry risk as retired-for-Phase-1 and record the measured fdp throughput
  if Task 12 produced a number.

- [ ] **Step 8: Commit the deliverable**

```bash
cd /scratch/gpfs/nc1514/FusionAIHub
git add docs/LABELMAKER.md docs/superpowers/specs/2026-09-03-labelmaker-design.md \
        src/labelmaker/models/d3d_tearing_onset_cnn1d/README.md
git commit -m "labelmaker: proof of concept on 100 shots, operator guide, spec brought up to date"
git log --oneline labelmaker ^main | cat
```
Expected: one commit per task, in order, on the `labelmaker` branch. Do not push; Nathan
pushes to `nathan_dev` himself.

---

## Done means

- `pixi run -e labelmaker python -m pytest tests/labelmaker` is green, with only the
  live-fdp test skipped.
- 100 shots have `features/<shot>_features.h5` and `labels/<shot>_labels.h5`, and
  `labels_index.parquet` has 200 rows.
- `validation/d3d_tearing_onset_cnn1d/` holds three reports, and
  `adapter_fidelity.json` says `passed: true`.
- The model card carries AUROC with archived inputs, AUROC with reconstructed inputs, and
  therefore the reconstruction penalty.
- Six scaffold folders exist, each with a card that parses and a `blocked_on` list.
- `import labelmaker` works in a freshly materialised `labelmaker` environment.
- Every task is one commit on the `labelmaker` branch, and nothing is pushed.
