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
import torch

from .config import Paths, atomic_path
from .models import registry
from .models.runners.keras_h5 import load_ensemble, predict_members

# This resolves correctly from a source checkout (three parents up from
# src/labelmaker/validate.py lands on the repo root), but not from an
# installed wheel - tests/ is not packaged, so the file would not exist at
# this path there. `adapter_fidelity` takes `golden=` so any caller that
# needs to run outside a checkout can point it elsewhere; there is no
# fallback search implemented here.
GOLDEN = Path(__file__).resolve().parents[2] / (
    "tests/labelmaker/data/tearing_golden.npz"
)


def write_report(paths: Paths, slug: str, name: str, payload: dict) -> Path:
    """Write one validation report, atomically."""
    out_dir = paths.validation / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def adapter_fidelity(slug: str, golden: Path = GOLDEN, *, tol: float = 1e-4) -> dict:
    """Compare our evaluator against the framework's frozen outputs.

    The golden file holds the reference inputs and the outputs a real Keras
    load produced for the same weights, so this runs with no framework
    installed. A failure here invalidates every label the model has ever
    written, which is why it is a hard pass/fail rather than a metric.

    The weights loaded here come from the golden file's own `meta["models"]`,
    not from `paths.models` - deliberately: the comparison is only meaningful
    against the exact weights TensorFlow saw when the golden file was made,
    so `paths` is otherwise unused by this function.

    `tol` is 1e-4, not 1e-5, and stays there after Task 14b's measurement -
    but the reasoning changed. Task 14 (numpy evaluator) measured
    `max_abs_diff` ~5.6e-5 and guessed this was "TensorFlow's own float32
    rounding against our float64 arithmetic", since the numpy evaluator
    upcast everything to float64 and never computed in float32 at all. That
    hypothesis predicted a decisive test: if it were true, re-running THIS
    evaluator's own arithmetic in float32 should reproduce something close
    to TensorFlow's float32 forward pass, so the float32-vs-golden diff
    should collapse toward float32 machine epsilon (~1e-6).

    It does not. Loading the same ten members at `dtype=torch.float32`
    measures `max_abs_diff_float32` ~8.2e-5 - the same order of magnitude as
    float64's ~5.6e-5, not two orders of magnitude smaller. So the
    hypothesis is falsified: the residual is not "our exact float64 answer
    minus TensorFlow's float32 rounding". Layer-by-layer tracing (see
    `models/runners/keras_h5.py`'s module docstring) instead localizes it to
    ordinary float32-scale rounding noise, compounded across the ~15
    sequential BatchNorm/Conv1D/Dense layers and expressed almost entirely
    on the ensemble's one unbounded (linear-activation) output column - the
    tearing logit, never `betan` - because every other activation in the
    graph is a saturating sigmoid that absorbs small input perturbations
    near saturation. That noise floor differs between TensorFlow's own
    float32 kernels and any other implementation (torch's included, at
    either dtype) simply because "float32 arithmetic" does not mean one
    fixed rounding sequence - different reduction orders and fused
    multiply-adds land on different, similarly-sized, residuals. Both
    measured numbers (5.6e-5, 8.2e-5) remain two orders of magnitude below
    the ~1e-2 a genuine implementation error produced when this was checked
    (Task 14), so 1e-4 stays a tight, bug-catching gate - it is left
    unchanged, not re-derived to fit either number.
    """
    adapter = registry.load_adapter(slug)
    with np.load(golden, allow_pickle=False) as z:
        x0, x1, want = z["x0"], z["x1"], z["members"]
        meta = json.loads(str(z["meta"]))
    artifact_paths = [Path(meta["models"]) / name for name in adapter.artifacts]
    graphs = load_ensemble(artifact_paths, dtype=torch.float64)
    got = predict_members(graphs, [x0, x1])  # positional; see predict_members
    # float32 is measured purely to settle the tolerance question above; the
    # dtype actually used for `got`/`max_abs_diff` stays float64, matching
    # this evaluator's historical precision and every hand-computed oracle.
    graphs32 = load_ensemble(artifact_paths, dtype=torch.float32)
    got32 = predict_members(graphs32, [x0, x1])
    diff = np.abs(got - want)
    diff32 = np.abs(got32.astype(np.float64) - want)
    return {
        "slug": slug,
        "golden": str(golden),
        "framework_version": meta.get("keras"),
        "n_rows": int(x0.shape[0]),
        "n_members": int(got.shape[0]),
        "max_abs_diff": float(diff.max()),
        "max_abs_diff_float64": float(diff.max()),
        "max_abs_diff_float32": float(diff32.max()),
        "median_abs_diff": float(np.median(diff)),
        "max_abs_diff_by_column": [float(c) for c in diff.max(axis=(0, 1))],
        "tolerance": tol,
        "passed": bool(diff.max() < tol),
    }
